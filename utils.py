from enum import Enum

import re

import torch
from torch.distributions import Categorical


class EmbeddingTypes(str, Enum):
    BLOCK_OUTPUT = "block_output"
    BLOCK_INPUT = 'block_input'
    POST_ATTENTION = "post_attention"
    POST_FF = "post_ff"
    POST_ATTENTION_RESIDUAL = "post_attention_residual"


# class OldLayerWrapper:
#     def __init__(self, layer_number,):
#         self.layer_number = layer_number
#         self.probabilities = {}
#         self.embeddings = {}
#         # key: embedding type, value: list of secondary keywords decoded
#         self.seconary_tokens = {}

#     def add_embedding(self, tensor: torch.tensor, value: str):
#         self.embeddings[value] = tensor

#     def get_embedding(self, emb_type):
#         return self.embeddings[emb_type]

#     def add_probability(self, prob, decoding_strategy: str):
#         self.probabilities[decoding_strategy] = prob

#     def get_probability(self, decoding_strategy):
#         return self.probabilities[decoding_strategy]

#     def add_secondary_tokens(self, tokens: List[str], emb_type: EmbeddingTypes):
#         self.seconary_tokens[emb_type] = tokens

#     def __sub__(self, other):
#         l = LayerWrapper((self.layer_number + other.layer_number) / 2)
#         for k in self.embeddings.keys() & other.embeddings.keys():
#             # Why are some tensors on cpu and some on gpu?
#             l.add_embedding(
#                 self.embeddings[k].cpu() - other.embeddings[k].cpu(), k)
#         return l


class CellWrapper:
    def __init__(self, layer_number=-1, token_number=-1, emb_keys=(), emb_block=torch.Tensor(), probabilities=None):
        self.layer_number = layer_number
        self.token_number = token_number
        self.embeddings = {k: emb for k, emb in zip(emb_keys, emb_block)}
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

    def __sub__(self, other):
        l = CellWrapper((self.layer_number + other.layer_number) / 2)
        for k in self.embeddings.keys() & other.embeddings.keys():
            l.add_embedding(self.embeddings[k] - other.embeddings[k], k)
        return l


class LayerWrapper:
    def __init__(self, layer_number=-1, token_numbers=(), emb_keys=(), emb_block=torch.Tensor(), prob_list=()):
        self.layer_number = layer_number
        self.cells = [
            CellWrapper(layer_number, token_n, block_keys, emb, prob)
            for token_n, block_keys, emb, prob
            in zip(token_numbers, emb_keys, emb_block, prob_list)
        ]

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
        return (self.__class__, (self.layer_number, token_numbers, emb_keys, emb_block, prob_list))

    def __iter__(self):
        return iter(self.cells)

    def __getitem__(self, index):
        return self.cells[index]

    def __setitem__(self, index, value):
        self.cells[index] = value

    def __len__(self):
        return len(self.cells)


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
        hidden_state = torch.tensor(hidden_state).to(
            self.input_embedding.weight.device)
        # Input logits
        input_logits = torch.matmul(hidden_state.squeeze().to(
            self.input_embedding.weight.device), self.input_embedding.weight.T)

        # Output logits
        output_logits = self.output_embedding(hidden_state.squeeze())
        output_logits = output_logits.float()

        interpolated_embedding = (
            (n_layers - layer_n) * (input_logits) + layer_n * (output_logits)) / n_layers

        probabilities = torch.nn.functional.softmax(interpolated_embedding)
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

    def decode_secondary_tokens(self, target_hidden_state, decoding: str, layer: LayerWrapper):
        "Decode secondary tokens"
        if decoding == 'input':
            decoding_function = self._input_embedding_prediction
        elif decoding == 'output':
            decoding_function = self._output_embedding_prediction
        elif decoding == 'interpolation':
            decoding_function = self._interpolated_embedding_prediction

        hidden_state = layer.get_embedding(target_hidden_state)
        secondary_tokens = []
        norms = []
        for _ in range(self.max_rep):
            token_id, _, current_hidden_state, real_embedding = decoding_function(
                hidden_state=hidden_state,
                layer_n=layer.layer_number,
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
