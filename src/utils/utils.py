from typing import List, Self
from enum import Enum

import re

import torch


class EmbeddingsType(str, Enum):
    BLOCK_OUTPUT = "block_output"
    BLOCK_INPUT = "block_input"
    POST_ATTENTION = "post_attention"
    POST_FF = "post_ff"
    POST_ATTENTION_RESIDUAL = "post_attention_residual"

class ProbabilityType(str, Enum):
    ATT_RES_PERCENT = "att_res_percent"
    FFNN_RES_PERCENT = "ffnn_res_percent"
    ENTROPY = "entropy"
    ARGMAX = "argmax"

class DecodingType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    LINEAR = "linear_interpolation"
    QUADRATIC = "quadratic_interpolation"

class ResidualContribution(str, Enum):
    NORM = "norm"
    KL_DIV = "kl_divergence"

class SecondaryDecodingType(str, Enum):
    TOP_K = "top_k"
    ITERATIVE = "iterative"


def _res_contrib_norm(residual, x, embs, **kwargs): # pylint:disable=unused-argument
    return (
        embs[residual].norm(2, dim=-1) / (embs[residual].norm(2, dim=-1) + embs[x].norm(2, dim=-1))
    ).squeeze().detach().float().cpu()

def _res_contrib_kl_div(residual, x, ref, embs, norm, **kwargs): # pylint:disable=unused-argument
    emb_ref = norm(embs[ref])
    emb_res = norm(embs[residual])
    emb_x = norm(embs[x])
    kldiv_res = torch.nn.functional.kl_div(
        torch.log_softmax(emb_res, dim=-1),
        torch.log_softmax(emb_ref, dim=-1),
        log_target=True, reduction="sum",
    )
    kldiv_x = torch.nn.functional.kl_div(
        torch.log_softmax(emb_x, dim=-1),
        torch.log_softmax(emb_ref, dim=-1),
        log_target=True, reduction="sum",
    )
    return (kldiv_x / (kldiv_res + kldiv_x + 1e-10)).detach().float().cpu()

def _iterative_decoding(
    tokenizer, emb, decoding_matrix, normalization, max_rep: int = 5, **kwargs # pylint:disable=unused-argument
):
    secondary_tokens = []
    norms = []
    for rep in range(max_rep):
        logits = torch.matmul(normalization(emb), decoding_matrix.T)
        token_id = logits.argmax()
        real_embed = decoding_matrix[token_id]
        secondary_token = tokenizer.convert_ids_to_tokens([token_id])[0]
        norm = torch.norm(emb)
        # Stop prematurely if norm is too small or if norm is bigger than previous one
        if norm <= 0.01 or (len(norms) > 0 and norm >= norms[-1]) and rep != 0:
            break
        # Do not add repreated tokens
        if secondary_token not in secondary_tokens:
            secondary_tokens.append(secondary_token)
        norms.append(norm)
        # Subtract the inverse to the current hidden state
        emb = emb - real_embed
    return [clean_text(s) for s in secondary_tokens]

def _topk_decoding(tokenizer, emb, decoding_matrix, k: int = 5, **kwargs):  # pylint:disable=unused-argument
    return [
        clean_text(tokenizer.convert_ids_to_tokens([topk_id])[0])
        for topk_id in torch.topk(torch.matmul(emb, decoding_matrix.T), k=k, sorted=True).indices
    ]

# TODO: make more efficient/general
def clean_text(t):
    return repr( chr(int(t[3:-1], 16)) if re.match(r"<0x\w\w>", t) else t )[1:-1] \
        .replace("\u0120", "_") \
        .replace("\u010a", "\\n")


DEFAULT_SESSION_ID = "0"
RESIDUAL_CONTRIBUTION = {
    ResidualContribution.NORM: _res_contrib_norm,
    ResidualContribution.KL_DIV: _res_contrib_kl_div
}
SECONDARY_DECODING_STRATEGY = {
    SecondaryDecodingType.TOP_K: _topk_decoding,
    SecondaryDecodingType.ITERATIVE: _iterative_decoding
}


class CellWrapper:
    def __init__(
        self, layer_number = -1, token_number = -1, emb_keys=(), emb_block=torch.Tensor(), probabilities = None
    ):
        self.layer_number = layer_number
        self.token_number = token_number
        self.embeddings = dict(zip(emb_keys, emb_block))
        self.probabilities = probabilities if probabilities else {}

    def __reduce__(self):
        emb_keys = list(self.embeddings.keys())
        emb_block = torch.stack(list(self.embeddings.values()), dim=0)
        return (self.__class__, (self.layer_number, self.token_number, emb_keys, emb_block, self.probabilities))

    def add_embedding(self, tensor: torch.Tensor, emb_type: EmbeddingsType):
        self.embeddings[emb_type] = tensor

    def get_embedding(self, emb_type: EmbeddingsType, norm = None):
        return norm(self.embeddings[emb_type]) if norm else self.embeddings[emb_type]

    def get_embeddings(self, norm = None):
        return {k: self.get_embedding(k, norm) for k, v in self.embeddings.items()}

    def add_probability(self, prob, decoding_strategy: str):
        self.probabilities[decoding_strategy] = prob

    def get_probability(self, decoding_strategy):
        return self.probabilities[decoding_strategy]

    def get_logits_info(self, emb_type: EmbeddingsType, decoding_matrix, norm = None):
        logits = torch.matmul(self.get_embedding(emb_type, norm), decoding_matrix.T).float()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        token_id = logits.argmax()
        return logits, probs, token_id

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

    def slice_cells(self, start: int = 0, end: int = -1):
        end = end if end > 0 else len(self.cells)
        return LayerWrapper(layer_number=self.layer_number, cells=self.cells[start:end])

    # TODO: maybe inefficient?
    # l.cells = [
    #     CellWrapper(
    #         self.layer_number,
    #         cell1.token_number,
    #         emb_keys=keys,
    #         emb_block=torch.stack([cell1.embeddings[k] - cell2.embeddings[k] for k in keys], dim=0)
    #     )
    #     for cell1, cell2 in zip(self, other) if (keys := cell1.embeddings.keys() & cell2.embeddings.keys()) or True
    # ]
    def get_diff(self, other: Self):
        l = LayerWrapper(
            layer_number=self.layer_number,
            session_id=self.session_id,
        )
        for cell1, cell2 in zip(self, other):
            c = CellWrapper(self.layer_number, cell1.token_number)
            for k in cell1.embeddings.keys() & cell2.embeddings.keys():
                c.add_embedding(cell1.embeddings[k] - cell2.embeddings[k], k)
            l.cells.append(c)
        return l

    def get_kldiff(self, other: Self, emb_type: EmbeddingsType, other_emb_type: EmbeddingsType = None, norm = None):
        other_emb_type = emb_type if other_emb_type is None else other_emb_type
        return [
            torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(cell1.get_embedding(emb_type, norm), dim=-1),
                torch.nn.functional.log_softmax(cell2.get_embedding(other_emb_type, norm), dim=-1),
                log_target=True, reduction="sum",
            ).detach().float().cpu()
            for cell1, cell2 in zip(self, other)
        ]

    def compute_probabilities(self, emb_type, decoder, decoding, return_0_on_error: bool = False):
        decoding_matrix = decoder.decoding_matrix[decoding]()[self.layer_number]
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
    def __init__(self, model, tokenizer, model_config):
        self.tokenizer = tokenizer
        self.output_embedding = model.lm_head
        self.input_embedding = model.get_input_embeddings()
        self.model_config = model_config
        self.decoding_matrix = {
            DecodingType.INPUT:
                lambda: ( self.input_embedding.weight for _ in range(self.model_config.num_hidden_layers + 2) ),
            DecodingType.OUTPUT:
                lambda: ( self.output_embedding.weight for _ in  range(self.model_config.num_hidden_layers + 2) ),
            DecodingType.LINEAR:
                lambda: self.linear_interpolation(self.input_embedding.weight, self.output_embedding.weight),
            DecodingType.QUADRATIC:
                lambda: self.quadratic_interpolation(self.input_embedding.weight, self.output_embedding.weight),
        }

    def linear_interpolation(self, matrix_in, matrix_out):
        # TODO: n_layers + 1 + 1 because we assume that the embedding layer/final norm layer are also being decoded
        n_layers = self.model_config.num_hidden_layers + 1
        return (
            ( (n_layers - layer_n) * matrix_in + layer_n * (matrix_out) ) / n_layers
            for layer_n in range(0, n_layers + 1)
        )

    def quadratic_interpolation(self, matrix_in, matrix_out):
        # TODO: n_layers + 1 + 1 because we assume that the embedding layer/final norm layer are also being decoded
        n_layers = self.model_config.num_hidden_layers + 1
        return (
            ( (n_layers ** 2 - layer_n ** 2) * matrix_in + (layer_n ** 2) * (matrix_out) ) / (n_layers ** 2)
            for layer_n in range(0, n_layers + 1)
        )

    def decode(
        self, layers: List[LayerWrapper],
        decoding: DecodingType, secondary_decoding: SecondaryDecodingType,
        norm
    ):
        decoding_matrix = self.decoding_matrix[decoding]()
        return [
            [
                {
                    k: SECONDARY_DECODING_STRATEGY[secondary_decoding](
                        tokenizer=self.tokenizer, emb=emb, decoding_matrix=dm, normalization=norm,
                    )
                    for k, emb in cell.get_embeddings(norm).items()
                }
                for cell in layer
            ] for layer, dm in zip(layers, decoding_matrix)
        ]

    def compute_probabilities(
        self, layers: List[LayerWrapper], decoding, residual_contribution: ResidualContribution, norm
    ):
        decoding_matrix = self.decoding_matrix[decoding]()
        return [
            [
                self.compute_cell(cell, dm, residual_contribution, norm)
                for cell in layer
            ] for layer, dm in zip(layers, decoding_matrix)
        ]

    def compute_cell(
        self, cell: CellWrapper, decoding_matrix, residual_contribution: ResidualContribution, norm: bool
    ):
        res = {}

        entropy = {}
        argmax =  {}
        embs = {}
        for k in cell.embeddings.keys():
            _, probs, pred_id = cell.get_logits_info(k, decoding_matrix, norm)
            embs[k] = cell.get_embedding(k)
            entropy[k] = -torch.sum(probs * torch.log(probs)).detach().cpu()
            argmax[k] = probs[pred_id].detach().cpu()

        res[ProbabilityType.ENTROPY] = entropy
        res[ProbabilityType.ARGMAX] = argmax
        att_res_percent = torch.tensor(0.0) \
            if EmbeddingsType.BLOCK_INPUT not in cell.embeddings \
                or EmbeddingsType.POST_ATTENTION not in cell.embeddings \
            else RESIDUAL_CONTRIBUTION[residual_contribution](
                residual=EmbeddingsType.BLOCK_INPUT,
                x=EmbeddingsType.POST_ATTENTION,
                ref=EmbeddingsType.POST_ATTENTION_RESIDUAL,
                embs=embs,
                norm=norm,
            )
        res |= {ProbabilityType.ATT_RES_PERCENT: att_res_percent}

        ffnn_res_percent = torch.tensor(0.0) \
            if EmbeddingsType.POST_ATTENTION_RESIDUAL not in cell.embeddings \
                or EmbeddingsType.POST_FF not in cell.embeddings \
            else RESIDUAL_CONTRIBUTION[residual_contribution](
                residual=EmbeddingsType.POST_ATTENTION_RESIDUAL,
                x=EmbeddingsType.POST_FF,
                ref=EmbeddingsType.BLOCK_OUTPUT,
                embs=embs,
                norm=norm,
            )
        res |= {ProbabilityType.FFNN_RES_PERCENT: ffnn_res_percent}
        return res
