import threading
import os

from dash import Dash
from torch import cuda

import diskcache
import dash_bootstrap_components as dbc

from utils import ModelUtils
from app.constants import *  # pylint:disable=W0401,W0614
from app.callbacks import generate_callbacks
from app.layout import generate_layout

# CONSTANTS
TITLE = "InTraVisTo"
ASSETS_PATH = "./app/assets/"
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.13.0/css/all.css"

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

# TODO: eventually remove this initialization
models = {model_id: ModelUtils(model_id, device, quant=True, hf_token=hf_auth)}

models_lock = threading.Lock()
model_loading_lock = threading.Lock()

model_config = models[model_id].model_config

###################################

cache = diskcache.Cache("./app/cache")

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.MATERIA, FONT_AWESOME],
    assets_folder=ASSETS_PATH,
    title=TITLE,
    update_title=TITLE + "‎", # Avoid "Updating..." title while processing callbacks, invisible character is needed to avoid having equal title and update_title, leading to "undefined" title
    routes_pathname_prefix="/intravisto/"
)

# TODO: fix value passed to generate functions
app.layout = generate_layout(model_config)

generate_callbacks(app, cache, models, models_lock, model_loading_lock, device)

from dash import clientside_callback, ClientsideFunction, Input, Output, State
clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="detect_scroll_inject"
    ),
    Output("javascript-inject", "children"),
    Input("javascript-inject", "children")
)

clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="update_scroll"
    ),
    Output("table_scroll", "data"),
    Input("scrollable_table_js_store", "children"),
)

if __name__ == '__main__':

    app.run(debug=False, host="0.0.0.0", port="8892")
