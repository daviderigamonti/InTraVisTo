from dash import html, dcc

import dash_bootstrap_components as dbc

from app.constants import * # pylint:disable=W0401,W0614
from app.defaults import * # pylint:disable=W0401,W0614

# TODO: fix css
def generate_tooltip_children_layout(layer: int = -1, token: int = -1):
    return dbc.Col([
        dbc.Row([html.H2("Inject Custom Embedding")]),
        dbc.Row([html.P(f"Layer: {layer}, Token: {token}")]),
        dbc.Row([
            dbc.Input(
                type="text",
                value="",
                id={"type": "custom_emb", "index": True},
                debounce=False,
                className="form-control-sm border border-primary text-white",
            )
        ], className="px-2 mb-3"),
        dbc.Row([
            dbc.RadioItems(
                options=EMB_TYPE_MAP, value=DEFAULT_EMB_TYPE,
                id={"type": "custom_emb_location", "index": True}
            ),
        ]),
        dbc.Row([
            html.Button(
                "Add Injection",
                id={"type": "add_inj_button", "index": True},
                className="btn btn-sm btn-primary float-end"
            ),
        ], className="px-2 mt-2")
    ])

def generate_inject_card(card_id, text, position, token, layer):
    return dbc.Card([
        dbc.CardHeader([
            f"Injecting {text}",
            html.Button(
                html.I(className="fas fa-times"),
                className="btn btn-sm btn-danger float-end", id={"type": "inject_close_button", "index": card_id}
            ),
        ]),
        dbc.CardBody([
            html.P(f"Position: {get_label_type_map(EMB_TYPE_MAP, position)}"),
            html.P(f"Layer: {layer}, Token: {token} "),
        ])
    ], className="mb-2 me-2 inject-card text-white bg-primary", id={"type": "inject_card", "index": card_id},)
    