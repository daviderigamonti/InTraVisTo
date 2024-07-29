from enum import Enum
from typing import List, Self

import time
import re
import gc

from torch.cuda import OutOfMemoryError
from torch import bfloat16
from transformers import AutoTokenizer
from scipy.special import kl_div # (ufuncs in scipy.special are written in C) pylint:disable=E0611

import transformers
import torch

from transformer_wrappers.wrappers import InjectCausalLMWrapper, CausalLMWrapper# pylint:disable=E0401,E0611
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def _res_contrib_norm(residual, x, embs, **kwargs):
    return (
        embs[residual].norm(2, dim=-1) / (embs[residual].norm(2, dim=-1) + embs[x].norm(2, dim=-1))
    ).squeeze().detach().float().cpu()

def _res_contrib_kl_div(residual, x, ref, emb_probs, **kwargs):
    kldiv_res = kl_div(emb_probs[residual], emb_probs[ref]).sum()
    kldiv_x = kl_div(emb_probs[x], emb_probs[ref]).sum()
    return kldiv_x / (kldiv_res + kldiv_x)

DEFAULT_SESSION_ID = "0"
RESIDUAL_CONTRIBUTION = {
    ResidualContribution.NORM: _res_contrib_norm,
    ResidualContribution.KL_DIV: _res_contrib_kl_div
}


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

    def add_embedding(self, tensor: torch.Tensor, emb_type: EmbeddingsType):
        self.embeddings[emb_type] = tensor

    def get_embedding(self, emb_type: EmbeddingsType):
        return self.embeddings[emb_type]

    def add_probability(self, prob, decoding_strategy: str):
        self.probabilities[decoding_strategy] = prob

    def get_probability(self, decoding_strategy):
        return self.probabilities[decoding_strategy]

    def get_logits_info(self, emb_type: EmbeddingsType, decoding_matrix):
        logits = torch.matmul(self.embeddings[emb_type], decoding_matrix.T).float()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        token_id = logits.argmax()
        return logits, probs, token_id

    # TODO: remove
    def __sub__(self, other):
        l = CellWrapper((self.layer_number + other.layer_number) / 2)
        for k in self.embeddings.keys() & other.embeddings.keys():
            l.add_embedding(self.embeddings[k] - other.embeddings[k], k)
        return l

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
    def get_kldiff(self, other: Self, emb_type: EmbeddingsType, other_emb_type: EmbeddingsType = None):
        other_emb_type = emb_type if other_emb_type is None else other_emb_type
        return [
            kl_div(
                torch.nn.functional.softmax(cell1.get_embedding(emb_type).float().detach().cpu(), dim=-1),
                torch.nn.functional.softmax(cell2.get_embedding(other_emb_type).float().detach().cpu(), dim=-1),
            ).sum() / (
                (cell1.get_embedding(emb_type).norm() + cell2.get_embedding(other_emb_type).norm()) / 2
            ).float().detach().cpu()
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
    def __init__(self, model, tokenizer, model_config, max_rep=5):
        self.tokenizer = tokenizer
        self.output_embedding = model.lm_head
        self.input_embedding = model.get_input_embeddings()
        self.max_rep = max_rep
        self.model_config = model_config
        self.decoding_matrix = {
            DecodingType.INPUT:
                lambda: ( self.input_embedding.weight for _ in range(self.model_config.num_hidden_layers + 1) ),
            DecodingType.OUTPUT:
                lambda: ( self.output_embedding.weight for _ in  range(self.model_config.num_hidden_layers + 1) ),
            DecodingType.LINEAR:
                lambda: self.linear_interpolation(self.input_embedding.weight, self.output_embedding.weight),
            DecodingType.QUADRATIC:
                lambda: self.quadratic_interpolation(self.input_embedding.weight, self.output_embedding.weight),
        }

    def linear_interpolation(self, matrix_in, matrix_out):
        # TODO: n_layers + 1 because we assume that the embedding layer is also being decoded, maybe fix?
        n_layers = self.model_config.num_hidden_layers
        return (
            ( (n_layers - layer_n) * matrix_in + layer_n * (matrix_out) ) / n_layers
            for layer_n in range(0, n_layers + 1)
        )

    def quadratic_interpolation(self, matrix_in, matrix_out):
        # TODO: n_layers + 1 because we assume that the embedding layer is also being decoded, maybe fix?
        n_layers = self.model_config.num_hidden_layers
        return (
            ( (n_layers ** 2 - layer_n ** 2) * matrix_in + (layer_n ** 2) * (matrix_out) ) / (n_layers ** 2)
            for layer_n in range(0, n_layers + 1)
        )
    
    def decode(self, layers: List[LayerWrapper], decoding):
        decoding_matrix = self.decoding_matrix[decoding]()
        return [
            [
                self.decode_cell(cell, dm)
                for cell in layer
            ] for layer, dm in zip(layers, decoding_matrix)
        ]

    def decode_cell(self, cell: CellWrapper, decoding_matrix):
        decoded_cell = {}
        for k, emb in cell.embeddings.items():
            secondary_tokens = []
            norms = []
            for rep in range(self.max_rep):
                logits = torch.matmul(emb, decoding_matrix.T)
                token_id = logits.argmax()
                real_embed = decoding_matrix[token_id]
                secondary_token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                norm = torch.norm(emb)
                # Stop prematurely if norm is too small or if norm is bigger than previous one
                if norm <= 0.01 or (len(norms) > 0 and norm >= norms[-1]) and rep != 0:
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

    def compute_probabilities(self, layers: List[LayerWrapper], decoding, residual_contribution: ResidualContribution):
        decoding_matrix = self.decoding_matrix[decoding]()
        return [
            [
                self.compute_cell(cell, dm, residual_contribution)
                for cell in layer
            ] for layer, dm in zip(layers, decoding_matrix)
        ]

    # TODO: Horrible
    def compute_cell(self, cell: CellWrapper, decoding_matrix, residual_contribution: ResidualContribution):

        res = {}

        entropy = {}
        argmax =  {}
        emb_probs = {}
        for k in cell.embeddings.keys():
            _, probs, pred_id = cell.get_logits_info(k, decoding_matrix)
            emb_probs[k] = probs.float().detach().cpu()
            entropy[k] = -torch.sum(probs * torch.log(probs)).detach().float().cpu()
            argmax[k] = probs[pred_id].detach().float().cpu()

        res[ProbabilityType.ENTROPY] = entropy
        res[ProbabilityType.ARGMAX] = argmax
        att_res_percent = torch.tensor(0.0) \
            if EmbeddingsType.BLOCK_INPUT not in cell.embeddings or EmbeddingsType.POST_ATTENTION not in cell.embeddings \
            else RESIDUAL_CONTRIBUTION[residual_contribution](
                residual=EmbeddingsType.BLOCK_INPUT,
                x=EmbeddingsType.POST_ATTENTION,
                ref=EmbeddingsType.POST_ATTENTION_RESIDUAL,
                embs=cell.embeddings,
                emb_probs=emb_probs
            )
        res |= {ProbabilityType.ATT_RES_PERCENT: att_res_percent}

        ffnn_res_percent = torch.tensor(0.0) \
            if EmbeddingsType.POST_ATTENTION_RESIDUAL not in cell.embeddings or EmbeddingsType.POST_FF not in cell.embeddings \
            else RESIDUAL_CONTRIBUTION[residual_contribution](
                residual=EmbeddingsType.POST_ATTENTION_RESIDUAL,
                x=EmbeddingsType.POST_FF,
                ref=EmbeddingsType.BLOCK_OUTPUT,
                embs=cell.embeddings,
                emb_probs=emb_probs
            )
        res |= {ProbabilityType.FFNN_RES_PERCENT: ffnn_res_percent}
        return res

class ModelUtils:
    def __init__(self, model_id, device, quant=False, hf_token=None):
        self.id = model_id
        self.is_quantized = quant
        self.tokenizer, self.model, self.model_config, self.decoder, self.prefix_tokens = self._load_model(
            model_id, device, quant, hf_token
        )
        self.heartbeat_stamp = time.time()

    def _load_model(self, model_id, device, quant, hf_token, tries=4, try_timeout=5):
        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
        ) if quant else None
        model_config = transformers.AutoConfig.from_pretrained(
            model_id,
            output_attentions=True,
            output_hidden_states=True,
            output_scores=True,
            token=hf_token,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

        # TODO: find a solution for this
        # Compute prefix tokens (9 is a random number)
        prefix_tokens = tokenizer.encode("9", add_special_tokens=False, return_tensors="pt").to(device).flatten()
        prefix_tokens = prefix_tokens[0] if prefix_tokens.size(dim=0) > 1 else torch.tensor([]).to(device)

        MODEL_CONFIG = {
            "trust_remote_code": True,
            "device_map": device,
            "token": hf_token,
            "torch_dtype": bfloat16,
        }

        TOKENIZER_CONFIG = {
            "token": hf_token,
        }

        model = None
        while True:
            try:
                model = InjectCausalLMWrapper.from_pretrained(
                    model_id, model_kwargs=MODEL_CONFIG,
                    quantization_configs=quant_config,
                    tokenizer_name_or_path=model_id, tokenizer_kwargs=TOKENIZER_CONFIG,
                )
                model.enable_wrapper()
                break
            except OutOfMemoryError:

                del model
                model = None
                gc.collect()
                # TODO: add check for device
                torch.cuda.empty_cache()

                tries -= 1
                if tries <= 0:
                    print(f"Could not load {model_id}")
                    return None, None, None, None, None
                print(f"Out of Memory while loading {model_id}, {tries} attempt(s) left, next attempt in {try_timeout} seconds")
                
                time.sleep(try_timeout)

        decoder = Decoder(model=model, tokenizer=tokenizer, model_config=model_config)
        return tokenizer, model, model_config, decoder, prefix_tokens