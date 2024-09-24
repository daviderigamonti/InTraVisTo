from dash import html

import dash_bootstrap_components as dbc

from utils.utils import EmbeddingsType
from app.constants import *
from app.defaults import *

# TODO: fix css
def generate_tooltip_children_layout(
    layer: int = -1, token: int = -1, 
    emb_type: EmbeddingsType = None, ablation_opt: bool = False
):
    return dbc.Col([
        dbc.Row([html.H2("Inject Embedding")]),
        dbc.Row([html.P(f"Layer: {layer}, Token: {token}")]),
        dbc.Row([
            dbc.Input(
                type="text",
                value="",
                id={"type": "custom_emb", "index": True},
                debounce=False,
                className="form-control-sm border border-white text-white",
            )
        ], className="px-2 mb-3"),
        dbc.Row([
            dbc.Select(
                options=EMB_TYPE_MAP,
                value=DEFAULT_EMB_TYPE,
                id={"type": "custom_emb_location", "index": True},
                className="form-select mx-4 my-2 py-1 borderpx-1 w-75 text-white tooltip-bg"
            ),
        ]) if emb_type is None else dbc.Select(
            options=[EMB_TYPE_MAP], value=emb_type,
            id={"type": "custom_emb_location", "index": True},
            disabled=True, style={"display": "none"}
        ),
        dbc.Row([
            dbc.Select(
                options=DECODING_TYPE_MAP,
                value=DEFAULT_DECODING,
                id={"type": "custom_decoding", "index": True},
                className="form-select mx-4 my-2 py-1 borderpx-1 w-75 text-white tooltip-bg"
            ),
        ]),
        dbc.Row([
            dbc.Select(
                options=INJ_NORM_MAP,
                value=DEFAULT_INJ_NORM,
                id={"type": "custom_norm", "index": True},
                className="form-select mx-4 my-2 py-1 borderpx-1 w-75 text-white tooltip-bg"
            ),
        ]),
        dbc.Row([
            html.Button(
                "Add Injection",
                id={"type": "add_inj_button", "index": True},
                className="btn btn-sm my-2 btn-primary float-end"
            ),
        ], className="px-2 mt-2"),
        dbc.Row([
            html.Button(
                "Remove Node",
                id={"type": "add_abl_button", "index": True},
                className="btn btn-sm my-1 btn-warning float-end"
            ),
        ], className="px-2") if ablation_opt else None,
    ])

def generate_inject_card(card_id, text, position, decoding, norm, token, layer):
    return dbc.Card([
        dbc.CardHeader([
            f"Injecting {text}",
            html.Button(
                html.I(className="fas fa-times"),
                className="btn btn-sm btn-danger float-end", id={"type": "mod_close_button", "index": card_id}
            ),
        ]),
        dbc.CardBody([
            html.P(f"Position: {get_label_type_map(EMB_TYPE_MAP, position)}"),
            html.P(f"Decoding: {get_label_type_map(DECODING_TYPE_MAP, decoding)}"),
            html.P(f"Normalisation: {get_label_type_map(INJ_NORM_MAP, norm)}"),
            html.P(f"Layer: {layer}, Token: {token}"),
        ])
    ], className="mb-2 me-2 mod-card text-white bg-primary", id={"type": "inject_card", "index": card_id},)

def generate_ablation_card(card_id, position, token, layer):
    return dbc.Card([
        dbc.CardHeader([
            f"Removing {get_label_type_map(EMB_TYPE_MAP, position)}",
            html.Button(
                html.I(className="fas fa-times"),
                className="btn btn-sm btn-danger float-end", id={"type": "mod_close_button", "index": card_id}
            ),
        ]),
        dbc.CardBody([
            html.P(f"Layer: {layer}"),
            html.P(f"Token: {token}"),
        ])
    ], className="mb-2 me-2 mod-card text-white bg-primary", id={"type": "abl_card", "index": card_id},)
