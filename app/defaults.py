import dataclasses

import plotly.graph_objects as go

from sankey import SankeyParameters
from utils import EmbeddingTypes

DEFAULT_QUESTION = "Q: What is the capital of Italy? A:"
DEFAULT_EMB_TYPE = EmbeddingTypes.BLOCK_OUTPUT
DEFAULT_STRATEGY = "interpolation"

DEFAULT_FIGURE = go.Figure(layout={
    "xaxis": {"visible": False},
    "yaxis": {"visible": False},
    "annotations": [{
        "text": "No data",
        "xref": "paper",
        "yref": "paper",
        "showarrow": False,
        "font": {
            "size": 28
        }
    }],
    "width": 1900, "height": 1000,
})

DEFAULT_FONT_SIZE = 14
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
