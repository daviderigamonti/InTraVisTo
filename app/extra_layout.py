from dash import html, dcc

from app.constants import * # pylint:disable=W0401,W0614
from app.defaults import * # pylint:disable=W0401,W0614

# TODO: fix css
def generate_tooltip_children_layout(layer: int = -1, token: int = -1):
    return [
        html.Div([
            html.H2("Inject Custom Embedding"),
            html.Div([
                html.P(f"Layer: {layer}, Token: {token}"),
                dcc.Input(
                    placeholder="Embedding to change",
                    type="text",
                    value="",
                    id="custom_emb",
                    debounce=False
                ),
            ], style={"display": "inline-block", "vertical-align": "top;"}),
            html.Div([
                dcc.RadioItems(options=EMB_TYPE_MAP, value=DEFAULT_EMB_TYPE, id='custom_emb_location'),
            ], style={"display": "inline-block", "vertical-align": "top;"})
        ], style={'display': 'inline-block', 'text-align': 'justify', 'vertical-align': 'top'})
    ]