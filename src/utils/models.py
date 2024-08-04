from dataclasses import dataclass, field
from typing import List
from enum import Enum

import inspect
import time
import gc

from torch.cuda import OutOfMemoryError
from torch import bfloat16
from transformers import AutoTokenizer

import transformers
import torch

from transformer_wrappers.wrappers import InjectCausalLMWrapper# pylint:disable=E0401,E0611

from utils.utils import Decoder


class NormalizationStep(str, Enum):
    NONE = "none"
    ONLY_NORM = "only_norm"
    NORM_SCALE = "norm_scale"

class ModelCompatibilityInfo(str, Enum):
    AVAILABLE_NORMS = "available_norm"
    NORM_PATH = "norm_path"
    NORM_DATA = "norm_data"


LAYERNORM_DATA = [{"name": "weight", "reset": torch.ones_like}, {"name": "bias", "reset": torch.zeros_like}]
RMSNORM_DATA = [{"name": "weight", "reset": torch.ones_like}]

COMPAT_GPT = {
    ModelCompatibilityInfo.AVAILABLE_NORMS: [
        NormalizationStep.NONE, NormalizationStep.ONLY_NORM, NormalizationStep.NORM_SCALE
    ],
    ModelCompatibilityInfo.NORM_PATH: ["base_model", "transformer", "base_model", "ln_f"],
    ModelCompatibilityInfo.NORM_DATA: LAYERNORM_DATA,
}
COMPAT_LLAMA = {
    ModelCompatibilityInfo.AVAILABLE_NORMS: [
        NormalizationStep.NONE, NormalizationStep.ONLY_NORM, NormalizationStep.NORM_SCALE
    ],
    ModelCompatibilityInfo.NORM_PATH: ["base_model", "model", "norm"],
    ModelCompatibilityInfo.NORM_DATA: RMSNORM_DATA,
}

MODEL_COMPATIBILITY_MAP = {
    "gpt2": COMPAT_GPT,
    "meta-llama/Llama-2-7b-hf": COMPAT_LLAMA,
    "mistralai/Mistral-7B-Instruct-v0.2": COMPAT_LLAMA,
    "google/gemma-2b": COMPAT_LLAMA,
    "google/gemma-7b": COMPAT_LLAMA,
    "google/gemma-2-2b": COMPAT_LLAMA,
    "meta-llama/Meta-Llama-3-8B": COMPAT_LLAMA,
}


class LayerNormalizationWrapper:
    def __init__(
        self,
        norm: torch.nn.Module,
        rescale_info: List[dict] = None, # List of dictionaries of the form {"name": attribute name, "reset": function to use in order to reset value}
        status: NormalizationStep = NormalizationStep.NORM_SCALE
    ):
        self.norm = norm
        self.rescale_info = rescale_info if rescale_info else []
        self.rescale_attr = [getattr(norm, info["name"]) for info in rescale_info]
        self.forward_f = None

        self.change_status(status)

    def __call__(self, hidden_states):
        return self.forward_f(hidden_states)

    def change_status(self, new_status):
        self.status = new_status
        if new_status == NormalizationStep.ONLY_NORM:
            for info, attr in zip(self.rescale_info, self.rescale_attr):
                setattr(self.norm, info["name"], torch.nn.Parameter(info["reset"](attr)))
            self.forward_f = self.norm.forward
        elif new_status == NormalizationStep.NORM_SCALE:
            for info, attr in zip(self.rescale_info, self.rescale_attr):
                setattr(self.norm, info["name"], attr)
            self.forward_f = self.norm.forward
        else: # NormalizationStep.NONE
            self.forward_f = lambda x: x


@dataclass(eq=True, frozen=True)
class ModelInfo:
    id: str = field(default="", compare=True)
    device: str = field(default="cpu", compare=True)
    quantized: bool = field(default=False, compare=True)

class ModelUtils:
    def __init__(self, info,  hf_token=None):
        self.info = info
        self.tokenizer, self.model, self.model_config, self.decoder, self.prefix_tokens = self._load_model(
            info.id, info.device, info.quantized, hf_token
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

        try:

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

        except RuntimeError as e:
            print(f"Could not load {model_id}")
            raise e

        return tokenizer, model, model_config, decoder, prefix_tokens
