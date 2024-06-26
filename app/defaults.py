import dataclasses

import plotly.graph_objects as go

from sankey import SankeyParameters
from utils import EmbeddingsType, DecodingType

DEFAULT_QUESTION = "Q: What is the capital of Italy? A:"
DEFAULT_EMB_TYPE = EmbeddingsType.BLOCK_OUTPUT
DEFAULT_DECODING = DecodingType.LINEAR

DEFAULT_FIGURE = go.Figure(layout={
    "xaxis": {"visible": False},
    "yaxis": {"visible": False},
    "width": 1900, "height": 1000,
})

DEFAULT_FONT_SIZE = 12
DEFAULT_RUN_CONFIG = {"max_new_tok": 10}
DEFAULT_VIS_CONFIG = {}
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
}
