import os

from dash import Dash
from transformers import AutoTokenizer
from torch import bfloat16, cuda

import transformers
import diskcache
import torch
import dash_bootstrap_components as dbc
import pandas as pd

from utils import Decoder
from app.constants import *  # pylint:disable=W0401,W0614
from app.callbacks import generate_callbacks
from app.layout import generate_layout

from transformer_wrappers.wrappers import InjectCausalLMWrapper # pylint:disable=E0401,E0611


ASSETS_PATH = "./app/assets/"
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.13.0/css/all.css"

pd.set_option("display.max_columns", None)

# ENVIRONMENT VARIABLES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "huggingface"
os.environ["TRANSFORMERS_CACHE"] = "huggingface"
hf_auth = os.environ["HF_TOKEN"]

# MODELS
# Llama model
# model_id = "meta-llama/Llama-2-7b-hf"
# Mistral model
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# DEVICES
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
# device = "cpu"

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    output_attentions=True,
    output_hidden_states=True,
    output_scores=True,
    token=hf_auth,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_auth)

# TODO: find a solution for this
# Compute prefix tokens (9 is a random number)
prefix_tokens = tokenizer.encode("9", add_special_tokens=False, return_tensors="pt").to(device).squeeze()
prefix_tokens = prefix_tokens[0] if prefix_tokens.size(dim=0) > 1 else torch.tensor([]).to(device)


MODEL_CONFIG = {
    "trust_remote_code": True,
    # "config": model_config,
    "device_map": device,
    "token": hf_auth,
    "torch_dtype": bfloat16,
}

TOKENIZER_CONFIG = {
    "token": hf_auth,
}


model = InjectCausalLMWrapper.from_pretrained(
    model_id, model_kwargs=MODEL_CONFIG,
    quantization_configs=bnb_config,
    tokenizer_name_or_path=model_id, tokenizer_kwargs=TOKENIZER_CONFIG,
)
model.enable_wrapper()

decoder = Decoder(model=model, tokenizer=tokenizer, model_config=model_config)

###################################

cache = diskcache.Cache("./app/cache")

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.MATERIA, FONT_AWESOME],
    assets_folder=ASSETS_PATH
)

app.title = "InTraVisTo"

# TODO: fix value passed to generate functions
app.layout = generate_layout(model_config)

generate_callbacks(app, cache, model, decoder, model_config, tokenizer, prefix_tokens, device)

if __name__ == '__main__':

    app.run(debug=False, host="0.0.0.0", port="8892")
