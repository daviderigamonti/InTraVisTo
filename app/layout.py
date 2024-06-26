from dash import html, dcc

import dash_bootstrap_components as dbc

from app.constants import *  # pylint:disable=W0401,W0614
from app.defaults import * # pylint:disable=W0401,W0614


def generate_layout(model_config):
    return html.Div([
        _navbar(),
        html.Div([
            dbc.Row([
                dbc.Col([
                    _generation(),
                    html.Hr(),
                    _modes(),
                ]),
                dbc.Col([
                    _settings(model_config),
                    html.Hr(),
                    _injects(),
                ])
            ]),
            html.Hr(),
            html.Div(children=[
                dbc.Spinner(
                    dcc.Graph(figure=DEFAULT_FIGURE, id="main_graph", className="spinner-visible-element"),
                    spinner_class_name="spinner-graph", color="primary"
                )
            ], id="tooltip-target"),
            html.Hr(),
            dbc.Spinner(
                dcc.Graph(figure=DEFAULT_FIGURE, id="sankey_graph", className="spinner-visible-element"), 
                spinner_class_name="spinner-graph", color="primary"
            ),
            dbc.Tooltip(
                id="graph-tooltip", target="tooltip-target", is_open=False,
                flip=False, placement="top", autohide=False
            ),
            *_stores(),
        ], className="container-fluid pt-2")
    ])

def _navbar():
    return dbc.Navbar(
        dbc.Container([
            dbc.Row(
                [dbc.Col(dbc.NavbarBrand("InTraVisTo", className="ms-1 text-white"))],
                align="center",
                className="g-0",
            ),
        ]),
        sticky="top",
        color="primary",
    )

def _generation():
    return dbc.Col([
        dbc.Row([
            dbc.Col([
                html.H4("Input Prompt:"),
            ], className="col-md-auto"),
            dbc.Col([
                dcc.Input(
                    placeholder=DEFAULT_QUESTION,
                    type='text',
                    value=DEFAULT_QUESTION,
                    id="text",
                    debounce=False,  # Needed otherwise textbox gets reset every time a callback resolves
                    className="form-control border border-secondary px-1 w-100",
                ),
            ]),
            dbc.Col([
                html.Button(children=[
                    dbc.Spinner(html.Div("Generate", id="generate_button_load"), color="white")
                ], id="generate_button", className="btn btn-primary"),
            ], className="col-md-auto"),
        ], className="d-flex align-items-center"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H4("Output Text:"),
            ], className="col-md-auto"),
            dbc.Col([
                dcc.Textarea(
                    id="output_text",
                    className="form-control border border-secondary px-1 w-100",
                )
            ]),
        ], className="d-flex align-items-center"),
    ])

def _modes():
    return dbc.Row([
        dbc.Col([
            html.H4("Decoder used"),
            dbc.RadioItems(options=DECODING_TYPE_MAP, value=DEFAULT_DECODING, id='choose_decoding')
        ]),
        dbc.Col([
            html.H4("Embedding shown"),
            dbc.RadioItems(options=EMB_TYPE_MAP, value=DEFAULT_EMB_TYPE, id='choose_embedding')
        ]),
        dbc.Col([
            html.H4("Colour"),
            dbc.RadioItems(options=['P(argmax term)', 'Entropy[p]', 'Att Contribution %',
                            'FF Contribution %'], value='P(argmax term)', id='choose_colour')
        ])
    ])

def _injects():
    return dbc.Container([
        dbc.Row(
            dbc.Col([], id="inject_container", className="inject-container")
        ),
    ],
    fluid=True,
)

def _settings(model_config):
    return dbc.Col([
        dbc.Row([
            html.H4("Settings")
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Checklist(
                        [{"label": "Hide starting token", "value": "hide"}],
                        id="hide_col",
                        value=["hide"] if DEFAULT_TABLE_VIS_CONFIG["hide_col"] else [],
                        labelStyle={"float": "left"},
                        switch=True,
                    ),
                ]),
                dbc.Row([
                    dbc.Checklist(
                        [{"label": "Hide starting token (Sankey)", "value": "hide"}],
                        id="hide_0",
                        value=["hide"] if not DEFAULT_SANKEY_VIS_CONFIG["sankey_parameters"]["show_0"] else [],
                        labelStyle={"float": "left"},
                        switch=True,
                    ),
                ]),
                dbc.Row([
                    dbc.Checklist(
                        [{"label": "Hide non-layer tokens (Sankey)", "value": "hide"}],
                        id="hide_labels",
                        value=["hide"] if DEFAULT_SANKEY_VIS_CONFIG["sankey_parameters"]["only_nodes_labels"] else [],
                        labelStyle={"float": "left"},
                        switch=True,
                    ),
                ]),
            ]),
            dbc.Col([
                dbc.Row([
                    dbc.Input(
                        id="max_new_tokens", type='number', value=DEFAULT_RUN_CONFIG["max_new_tok"], min=0, max=1024,
                        className="w-25",
                    ),
                    html.Label("N° of tokens generated", className="w-75"),
                ], className="d-flex align-items-center"),
                dbc.Row([
                    dbc.Input(
                        id="font_size", type='number', value=DEFAULT_FONT_SIZE, min=1, max=72,
                        className="w-25",
                    ),
                    html.Label("Font size", className="w-75"),
                ], className="d-flex align-items-center"),
                dbc.Row([
                    dbc.Input(
                        id="row_limit", type='number', value=DEFAULT_SANKEY_VIS_CONFIG["sankey_parameters"]["rowlimit"],
                        min=1, max=model_config.num_hidden_layers,
                        className="w-25",
                    ),
                    html.Label("Sankey depth", className="w-75"),
                ], className="d-flex align-items-center"),
            ])
        ])
    ])

def _stores():
    return (
        dcc.Store(id="run_config", data=DEFAULT_RUN_CONFIG),
        dcc.Store(id="current_run_config"),
        dcc.Store(id="vis_config", data=DEFAULT_VIS_CONFIG),
        dcc.Store(id="sankey_vis_config", data=DEFAULT_SANKEY_VIS_CONFIG),
        dcc.Store(id="table_vis_config", data=DEFAULT_TABLE_VIS_CONFIG),
        dcc.Store(id="generation_notify"),
        dcc.Store(id="session_id")
    )
