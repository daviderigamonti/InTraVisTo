from enum import Enum
from typing import List, Self

import re

import torch
from torch.distributions import Categorical
from scipy.special import kl_div # (ufuncs in scipy.special are written in C) pylint:disable=E0611


DEFAULT_SESSION_ID = "0"

class EmbeddingTypes(str, Enum):
    BLOCK_OUTPUT = "block_output"
    BLOCK_INPUT = "block_input"
    POST_ATTENTION = "post_attention"
    POST_FF = "post_ff"
    POST_ATTENTION_RESIDUAL = "post_attention_residual"

class ProbabilityType(Enum):
    ATT_RES_PERCENT = (
        "compute_prob_residual",
        {"emb_res": EmbeddingTypes.BLOCK_INPUT, "emb": EmbeddingTypes.POST_ATTENTION}
    )
    FFNN_RES_PERCENT = (
        "compute_prob_residual",
        {"emb_res": EmbeddingTypes.POST_ATTENTION_RESIDUAL, "emb": EmbeddingTypes.POST_FF}
    )
    ENTROPY = (
        "compute_prob_entropy", {}
    )


class CellWrapper:
    def __init__(self, layer_number=-1, token_number=-1, emb_keys=(), emb_block=torch.Tensor(), probabilities=None):
        self.layer_number = layer_number
        self.token_number = token_number
        self.embeddings = dict(zip(emb_keys, emb_block))
        self.probabilities = probabilities if probabilities else {}

    def __reduce__(self):
        emb_keys = list(self.embeddings.keys())
        emb_block = torch.stack(list(self.embeddings.values()), dim=0)
        return (self.__class__, (self.layer_number, self.token_number, emb_keys, emb_block, self.probabilities))

    def add_embedding(self, tensor: torch.Tensor, emb_type: EmbeddingTypes):
        self.embeddings[emb_type] = tensor

    def get_embedding(self, emb_type: EmbeddingTypes):
        return self.embeddings[emb_type]

    def add_probability(self, prob, decoding_strategy: str):
        self.probabilities[decoding_strategy] = prob

    def get_probability(self, decoding_strategy):
        return self.probabilities[decoding_strategy]

    def get_logits_info(self, emb_type: EmbeddingTypes, decoding_matrix):
        logits = torch.matmul(self.embeddings[emb_type], decoding_matrix.T)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        token_id = logits.argmax()
        return logits, probs, token_id

    # TODO: remove
    def __sub__(self, other):
        l = CellWrapper((self.layer_number + other.layer_number) / 2)
        for k in self.embeddings.keys() & other.embeddings.keys():
            l.add_embedding(self.embeddings[k] - other.embeddings[k], k)
        return l

    def compute_prob_residual(self, emb_res: EmbeddingTypes, emb: EmbeddingTypes, return_0_on_error: bool = False, **kwargs):
        if emb not in self.embeddings or emb_res not in self.embeddings:
            if return_0_on_error:
                return 0.0
            raise KeyError(f"Missing embeddings {emb, emb_res} to compute cell probability")
        initial_residual = self.embeddings[emb_res]
        x_emb = self.embeddings[emb]
        return (
            initial_residual.norm(2, dim=-1) / (initial_residual.norm(2, dim=-1) + x_emb.norm(2, dim=-1))
        ).squeeze().detach().float().cpu()

    def compute_prob_entropy(self, emb_type, decoding_matrix, return_0_on_error: bool = False, **kwargs):
        return Categorical(probs=self.get_logits_info(emb_type, decoding_matrix)[1]).entropy()


class LayerWrapper:
    def __init__(
        self,
        layer_number : int = -1,
        session_id: str = DEFAULT_SESSION_ID,
        token_numbers = (),
        emb_keys = (),
        emb_block = torch.Tensor(),
        prob_list = (),
        cells : List[CellWrapper] = None
    ):
        self.layer_number = layer_number
        self.session_id = session_id
        self.cells = [
            CellWrapper(layer_number, token_n, block_keys, emb, prob)
            for token_n, block_keys, emb, prob
            in zip(token_numbers, emb_keys, emb_block, prob_list)
        ] if cells is None else cells

    def __reduce__(self):
        token_numbers = []
        emb_block = []
        emb_keys = []
        prob_list = []
        for cell in self.cells:
            _, (_, token_n, block_keys, emb, prob) = cell.__reduce__()
            token_numbers.append(token_n)
            emb_keys.append(block_keys)
            emb_block.append(emb)
            prob_list.append(prob)
        # Assumption that all cells in layer share the same number of embeddings
        emb_block = torch.stack(emb_block, dim=0)
        return (self.__class__, (self.layer_number, self.session_id, token_numbers, emb_keys, emb_block, prob_list))

    def __iter__(self):
        return iter(self.cells)

    def __getitem__(self, index):
        return self.cells[index]

    def __setitem__(self, index, value):
        self.cells[index] = value

    def __len__(self):
        return len(self.cells)

    def __repr__(self):
        return f"<LayerWrapper {self.layer_number} @ {self.session_id}>"

    def slice_cells(self, start : int = 0, end : int = -1):
        end = end if end > 0 else len(self.cells)
        return LayerWrapper(layer_number=self.layer_number, cells=self.cells[start:end])

    # TODO: maybe inefficient?
    def get_diff(self, other: Self):
        l = LayerWrapper(
            #layer_number=(self.layer_number + other.layer_number) / 2,
            layer_number=self.layer_number,
            session_id=self.session_id,
        )
        for cell1, cell2 in zip(self, other):
            c = CellWrapper(self.layer_number, cell1.token_number)
            for k in cell1.embeddings.keys() & cell2.embeddings.keys():
                c.add_embedding(cell1.embeddings[k] - cell2.embeddings[k], k)
            l.cells.append(c)
        return l
    
    # TODO: maybe inefficient?
    def get_kldiff(self, other: Self, emb_type: EmbeddingTypes):
        return [
            sum(kl_div(
                torch.nn.functional.softmax(cell1.get_embedding(emb_type), dim=-1).float().detach().cpu(),
                torch.nn.functional.softmax(cell2.get_embedding(emb_type), dim=-1).float().detach().cpu(),
            ))
            for cell1, cell2 in zip(self, other)
        ]

    def compute_probabilities(self, emb_type, decoder, decoding, return_0_on_error: bool = False):
        decoding_matrix = decoder.generate_decoding_matrix(decoding)[self.layer_number]
        return [{
            prob: getattr(cell, prob.value[0])(
                **prob.value[1],
                emb_type=emb_type,
                decoding_matrix=decoding_matrix,
                return_0_on_error=return_0_on_error
            )
            for prob in ProbabilityType
        } for cell in self.cells]

class Decoder:
    def __init__(self, model, tokenizer, model_config, max_rep=5):
        self.tokenizer = tokenizer
        self.output_embedding = model.lm_head
        self.input_embedding = model.get_input_embeddings()
        self.max_rep = max_rep
        self.model_config = model_config

    def _input_embedding_prediction(self, hidden_state, layer_n, return_logits=False, return_real_embedding=False):
        del layer_n  # Unused parameter
        output = torch.matmul(hidden_state.squeeze().to(
            self.input_embedding.weight.device), self.input_embedding.weight.T)
        token_id = output.argmax()
        probabilities = torch.nn.functional.softmax(output)
        if return_logits:
            return (token_id, probabilities, output)
        if return_real_embedding:
            real_embed = self.input_embedding.weight[token_id]
            return (token_id, probabilities, output, real_embed)
        return (token_id, probabilities)

    def _output_embedding_prediction(self, hidden_state, layer_n, return_logits=False, return_real_embedding=False):
        del layer_n  # Unused parameter
        hidden_state = torch.tensor(hidden_state).to(
            self.input_embedding.weight.device)
        logits = self.output_embedding(hidden_state.squeeze())
        logits = logits.float()
        probabilities = torch.nn.functional.softmax(logits)
        pred_id = torch.argmax(logits)
        if return_logits:
            return (pred_id, probabilities, logits)
        if return_real_embedding:
            true_embed = self.output_embedding.weight[pred_id]
            return (pred_id, probabilities, logits, true_embed)
        return (pred_id, probabilities)

    def _interpolated_embedding_prediction(
            self, hidden_state, layer_n, return_logits=False, return_real_embedding=False):
        n_layers = self.model_config.num_hidden_layers
        hidden_state = hidden_state.clone().detach().to(
            self.input_embedding.weight.device)
        # Input logits
        input_logits = torch.matmul(hidden_state.squeeze().to(
            self.input_embedding.weight.device), self.input_embedding.weight.T)

        # Output logits
        output_logits = self.output_embedding(hidden_state.squeeze())
        output_logits = output_logits.float()

        interpolated_embedding = (
            (n_layers - layer_n) * (input_logits) + layer_n * (output_logits)) / n_layers

        probabilities = torch.nn.functional.softmax(interpolated_embedding, dim=-1)
        pred_id = torch.argmax(probabilities)

        if return_logits:
            return (pred_id, probabilities, interpolated_embedding)
        if return_real_embedding:
            real_embed_out = self.input_embedding.weight[pred_id]
            true_embed_in = self.output_embedding.weight[pred_id]
            interpolated_real_embedding = (
                (n_layers - layer_n) * (true_embed_in) +
                layer_n * (real_embed_out)
            ) / n_layers
            return (pred_id, probabilities, interpolated_embedding, interpolated_real_embedding)
        return (pred_id, probabilities)

    def decode_hidden_state(self, target_hidden_state, decoding: str, layer: LayerWrapper):
        if decoding == 'input':
            pred_id, prob = self._input_embedding_prediction(
                layer.get_embedding(target_hidden_state), layer.layer_number)
        elif decoding == 'output':
            pred_id, prob = self._output_embedding_prediction(
                layer.get_embedding(target_hidden_state), layer.layer_number)
        elif decoding == 'interpolation':
            pred_id, prob = self._interpolated_embedding_prediction(
                layer.get_embedding(target_hidden_state), layer.layer_number
            )

        layer.add_probability(round(float(prob[pred_id]) * 100, 2), decoding)
        # Compute entropy
        entropy = Categorical(probs=prob).entropy()
        layer.add_probability(round(float(entropy) * 100, 2), "entropy")
        return self.tokenizer.convert_ids_to_tokens([pred_id])[0]

    def decode_secondary_tokens(self, decoding: str, layer: LayerWrapper, layer_n):
        if decoding == 'input':
            decoding_function = self._input_embedding_prediction
        elif decoding == 'output':
            decoding_function = self._output_embedding_prediction
        elif decoding == 'interpolation':
            decoding_function = self._interpolated_embedding_prediction

        hidden_state = layer
        secondary_tokens = []
        norms = []
        for _ in range(self.max_rep):
            token_id, _, current_hidden_state, real_embedding = decoding_function(
                hidden_state=hidden_state,
                layer_n=layer_n,
                return_real_embedding=True,
            )
            secondary_token = self.tokenizer.convert_ids_to_tokens([token_id])[
                0]
            norm = torch.norm(current_hidden_state)
            # Stop prematurely if norm is too small or if norm is bigger than previous one
            if norm <= 0.01 or (len(norms) > 0 and norm >= norms[-1]):
                break
            # Do not add repreated tokens
            if secondary_token not in secondary_tokens:
                secondary_tokens.append(secondary_token)
            norms.append(norm)
            # subtract the inverse to the current hidden state
            if hidden_state.device != real_embedding.device:
                hidden_state = hidden_state.clone().to(real_embedding.device)
            hidden_state = hidden_state - real_embedding
        secondary_tokens = [
            repr(chr(int(s[3:-1], 16)))[1:-
                                        1] if re.match(r'<0x\w\w>', s) else repr(s)[1:-1]
            for s in secondary_tokens
        ]
        # secondary_tokens = ["DEBUG" + s for s in secondary_tokens]
        return secondary_tokens

    def linear_interpolation(self, matrix_in, matrix_out):
        # TODO: n_layers + 1 because we assume that the embedding layer is also being decoded, maybe fix?
        n_layers = self.model_config.num_hidden_layers + 1
        return [
            ( (n_layers - layer_n) * matrix_in + layer_n * (matrix_out) ) / n_layers
            for layer_n in range(0, n_layers)
        ]

    # TODO: eventually make this an enum/dictionary?
    def generate_decoding_matrix(self, decoding):
        if decoding == 'input':
            return [self.input_embedding.weight] * self.model_config.num_hidden_layers
        elif decoding == 'output':
            return [self.output_embedding.weight] * self.model_config.num_hidden_layers
        elif decoding == 'interpolation':
            return self.linear_interpolation(self.input_embedding.weight, self.output_embedding.weight)
    
    def decode(self, layers: List[LayerWrapper], decoding):
        decoding_matrix = self.generate_decoding_matrix(decoding)
        return [self.decode_layer(layer, decoding_matrix) for layer in layers]

    def decode_layer(self, layer: LayerWrapper, decoding_matrix):
        decoding_matrix = decoding_matrix[layer.layer_number]
        return [self.decode_cell(cell, decoding_matrix) for cell in layer]

    def decode_cell(self, cell: CellWrapper, decoding_matrix):
        decoded_cell = {}
        for k, emb in cell.embeddings.items():
            secondary_tokens = []
            norms = []
            for _ in range(self.max_rep):
                logits = torch.matmul(emb, decoding_matrix.T)
                token_id = logits.argmax()
                real_embed = decoding_matrix[token_id]
                secondary_token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                norm = torch.norm(emb)
                # Stop prematurely if norm is too small or if norm is bigger than previous one
                if norm <= 0.01 or (len(norms) > 0 and norm >= norms[-1]):
                    break
                # Do not add repreated tokens
                if secondary_token not in secondary_tokens:
                    secondary_tokens.append(secondary_token)
                norms.append(norm)
                # subtract the inverse to the current hidden state
                emb = emb - real_embed
            secondary_tokens = [
                repr(chr(int(s[3:-1], 16)))[1:-1] if re.match(r'<0x\w\w>', s) else repr(s)[1:-1]
                for s in secondary_tokens
            ]
            decoded_cell[k] = secondary_tokens
        return decoded_cell

    def compute_probabilities(self, layers: List[LayerWrapper], decoding):
        decoding_matrix = self.generate_decoding_matrix(decoding)
        return [self.compute_layer(layer, decoding_matrix) for layer in layers]

    def compute_layer(self, layer: LayerWrapper, decoding_matrix):
        decoding_matrix = decoding_matrix[layer.layer_number]
        return [self.compute_cell(cell, decoding_matrix) for cell in layer]

    # TODO: Horrible
    def compute_cell(self, cell: CellWrapper, decoding_matrix):

        res = {}

        if EmbeddingTypes.BLOCK_INPUT not in cell.embeddings or EmbeddingTypes.POST_ATTENTION not in cell.embeddings:
            res |= {ProbabilityType.FFNN_RES_PERCENT: 0.0}
        else:
            initial_residual = cell.embeddings[EmbeddingTypes.BLOCK_INPUT]
            x_emb = cell.embeddings[EmbeddingTypes.POST_ATTENTION]
            res |= {ProbabilityType.ATT_RES_PERCENT: (
                    initial_residual.norm(2, dim=-1) / (initial_residual.norm(2, dim=-1) + x_emb.norm(2, dim=-1))
                ).squeeze().detach().float().cpu()
            }

        if EmbeddingTypes.POST_ATTENTION_RESIDUAL not in cell.embeddings or EmbeddingTypes.POST_FF not in cell.embeddings:
            res |= {ProbabilityType.FFNN_RES_PERCENT: 0.0}
        else:
            initial_residual = cell.embeddings[EmbeddingTypes.POST_ATTENTION_RESIDUAL]
            x_emb = cell.embeddings[EmbeddingTypes.POST_FF]
            res |= {ProbabilityType.FFNN_RES_PERCENT : (
                    initial_residual.norm(2, dim=-1) / (initial_residual.norm(2, dim=-1) + x_emb.norm(2, dim=-1))
                ).squeeze().detach().float().cpu()
            }

        res |= {ProbabilityType.ENTROPY: {
            k: Categorical(probs=cell.get_logits_info(k, decoding_matrix)[1]).entropy().detach().float().cpu()
            for k in cell.embeddings.keys()
        }}

        return res

