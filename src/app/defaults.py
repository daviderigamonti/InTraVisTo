import dataclasses

import plotly.graph_objects as go

from utils.utils import EmbeddingsType, DecodingType, ProbabilityType, ResidualContribution
from utils.models import ModelInfo, NormalizationStep
from utils.sankey import SankeyParameters, SizeAdapt, AttentionHighlight
from app.constants import *


DEFAULT_EMB_TYPE = EmbeddingsType.BLOCK_OUTPUT
DEFAULT_DECODING = DecodingType.LINEAR
DEFAULT_PROB_TYPE = ProbabilityType.ARGMAX
DEFAULT_RES_TYPE = ResidualContribution.NORM
DEFAULT_NORM = NormalizationStep.NORM_SCALE
DEFAULT_SECONDARY_DECODING = SecondaryDecodingType.TOP_K

DEFAULT_ATTENTION = AttentionHighlight.TOP_K
DEFAULT_ATT_HIGH_K = 1
DEFAULT_ATT_HIGH_W = 0.001

DEFAULT_QUESTION = "Q: What is the capital of Italy? A:"
DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_MODEL = ModelInfo(DEFAULT_MODEL_ID, CUDA_DEVICE, True)

DEFAULT_INITIAL_CALLS = ["update_model", "call_model_generate"]

DEFAULT_FIGURE = go.Figure(layout={
    "xaxis": {"visible": False},
    "yaxis": {"visible": False},
    "width": 1000, "height": 500,
    "plot_bgcolor": "white"
})

DEFAULT_FONT_SIZE = 12
DEFAULT_RUN_CONFIG = {"max_new_tok": 10, "injects": []}
DEFAULT_VIS_CONFIG = {
    "strategy": DEFAULT_DECODING,
    "res_contrib": DEFAULT_RES_TYPE,
    "norm": DEFAULT_NORM,
    "secondary_decoding": DEFAULT_SECONDARY_DECODING,
}
DEFAULT_SANKEY_VIS_CONFIG = {
    "hide_start": True,
    "reapport_start": True,
    "sankey_parameters": dataclasses.asdict(SankeyParameters(
        rowlimit=7,
        font_size=DEFAULT_FONT_SIZE,
        only_nodes_labels=True,
        size_adapt=SizeAdapt.FIXED,
    )),
}
DEFAULT_TABLE_VIS_CONFIG = {
    "hide_start": True,
    "font_size": DEFAULT_FONT_SIZE,
    "emb_type": DEFAULT_EMB_TYPE,
    "colour": DEFAULT_PROB_TYPE,
}

DEFAULT_SANKEY_SCALE = 1.0
