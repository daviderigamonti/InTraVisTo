import dataclasses

import plotly.graph_objects as go

from sankey import SankeyParameters
from utils import EmbeddingsType, DecodingType, ProbabilityType, ResidualContribution

DEFAULT_QUESTION = "Q: What is the capital of Italy? A:"
DEFAULT_EMB_TYPE = EmbeddingsType.BLOCK_OUTPUT
DEFAULT_DECODING = DecodingType.LINEAR
DEFAULT_PROB_TYPE = ProbabilityType.ARGMAX
DEFAULT_RES_TYPE = ResidualContribution.NORM
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_INITIAL_CALLS = ["update_output", "call_model_generate"]

DEFAULT_FIGURE = go.Figure(layout={
    "xaxis": {"visible": False},
    "yaxis": {"visible": False},
    "width": 1900, "height": 1000,
})

DEFAULT_FONT_SIZE = 12
DEFAULT_RUN_CONFIG = {"max_new_tok": 10, "injects": []}
DEFAULT_VIS_CONFIG = {"strategy": DEFAULT_DECODING, "res_contrib": DEFAULT_RES_TYPE}
DEFAULT_SANKEY_VIS_CONFIG = {
    "sankey_parameters": dataclasses.asdict(SankeyParameters(
        rowlimit=7,
        show_0=False,
        font_size=DEFAULT_FONT_SIZE,
        only_nodes_labels=True
    )),
}
DEFAULT_TABLE_VIS_CONFIG = {
    "hide_col": True,
    "font_size": DEFAULT_FONT_SIZE,
    "emb_type": DEFAULT_EMB_TYPE,
    "colour": DEFAULT_PROB_TYPE,
}
