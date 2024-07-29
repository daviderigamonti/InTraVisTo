import threading
import os

from dash import Dash

import diskcache
import dash_bootstrap_components as dbc

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

models = {}

models_lock = threading.Lock()
model_loading_lock = threading.Lock()

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
app.layout = generate_layout()

generate_callbacks(app, cache, models, models_lock, model_loading_lock)

if __name__ == '__main__':

    app.run(debug=False, host="0.0.0.0", port="8892")
