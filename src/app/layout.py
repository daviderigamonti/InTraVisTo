from dash import html, dcc

import dash_bootstrap_components as dbc

from app.constants import *
from app.defaults import *


def generate_layout():
    return html.Div([
        _navbar(),
        html.Div([
            dbc.Row([
                dbc.Col([
                    _generation(),
                ]),
                dbc.Col([
                    _injects(),
                ])
            ]),
            html.Hr(),
            dbc.Row([
                _settings(),
            ]),
            html.Hr(),
            dbc.Tabs([
                dbc.Tab(dbc.Card(dbc.CardBody([
                    html.Div(children=[
                        dbc.Spinner(
                            dcc.Graph(
                                figure=DEFAULT_FIGURE, id="main_graph",
                                className="spinner-visible-element", config={"displaylogo": False}
                            ),
                            spinner_class_name="spinner-graph", color="primary"
                        ),
                    ], id="scrollable_graph", className="scrollable-div"),
                    dbc.Tooltip(
                        id="graph_tooltip", target="tooltip_target", is_open=False,
                        flip=False, placement="top", autohide=False, className="dash-tooltip", trigger="legacy",
                    ),
                    html.Div([], id="tooltip_target"),
                ]), className="mt-3"), label="Heatmap"),
                dbc.Tab(dbc.Card(dbc.CardBody([
                    dbc.Row([
                        _settings_sankey(),
                    ]),
                    html.Hr(),
                    html.Div(children=[
                        dbc.Spinner(
                            dcc.Graph(
                                figure=DEFAULT_FIGURE, id="sankey_graph",
                                className="spinner-visible-element", config={"displaylogo": False}
                            ),
                            spinner_class_name="spinner-graph", color="primary"
                        ),
                    ], id="scrollable_sankey", className="biscrollable-div"),
                ]), className="mt-3"), label="Sankey")
            ]),
            *_stores(),
            dcc.Interval(id="model_heartbeat", interval=HEARTBEAT_INTERVAL * 1000),
        ], className="container-fluid pt-2"),
        html.Div([], id="overlay", className="overlay"),
        html.Div([], id="javascript_inject", style={"display": "none"}),
        html.Div(id="scrollable_table_js_store", children=0, style={"display": "none"}),
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
                    type="text",
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
            dbc.Alert(
                ["Error during generation"],
                id="model_generate_alert",
                color="danger", dismissable=True, fade=True, is_open=False,
            ),
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
    return dbc.Row([dbc.Col([
        dbc.Row([
            dbc.Col([
                html.H5("Decoder used"),
                dbc.RadioItems(options=DECODING_TYPE_MAP, value=DEFAULT_DECODING, id="choose_decoding")
            ]),
            dbc.Col([
                html.H5("Embedding shown"),
                dbc.RadioItems(options=EMB_TYPE_MAP, value=DEFAULT_EMB_TYPE, id="choose_embedding")
            ]),
            dbc.Col([
                html.H5("Colour"),
                dbc.RadioItems(options=PROB_TYPE_MAP, value=DEFAULT_PROB_TYPE, id="choose_colour")
            ]),
            dbc.Col([
                html.H5("Residual contribution"),
                dbc.RadioItems(options=RES_TYPE_MAP, value=DEFAULT_RES_TYPE, id="choose_res_type")
            ]),
        ])
    ])])

def _injects():
    return dbc.Container([
        dbc.Row(
            dbc.Col([], id="inject_container", className="inject-container")
        ),
    ],
    fluid=True,
)

def _settings():
    return dbc.Col([
        dbc.Row([
            html.H4("Settings")
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col(["Model:"], className="col-md-auto"),
                    dbc.Col([
                        dbc.Row([
                            dbc.Select(
                                id="model_select",
                                options=MODEL_MAP,
                                value=encode_dataclass(DEFAULT_MODEL),
                                className="form-select borderpx-1 w-100"
                            ),
                            dbc.Alert(
                                ["Error while loading model"],
                                id="model_select_alert",
                                color="danger", dismissable=True, fade=True, is_open=False,
                            ),
                        ], className="gy-2")
                    ], className="me-5"),
                ], className="my-1 d-flex align-items-center"),
                dbc.Row([
                    dbc.Input(
                        id="max_new_tokens", type="number", value=DEFAULT_RUN_CONFIG["max_new_tok"], min=0, max=1024,
                        className="w-25",
                    ),
                    html.Label("N° of generated tokens ", className="w-75"),
                ], className="mx-2 my-1 d-flex align-items-center"),
                dbc.Row([
                    dbc.Input(
                        id="font_size", type="number", value=DEFAULT_FONT_SIZE, min=1, max=72,
                        className="w-25",
                    ),
                    html.Label("Font size", className="w-75"),
                ], className="mx-2 my-1 d-flex align-items-center"),
            ], className="col-2"),
            dbc.Col([
                dbc.Row([
                    dbc.Col(["Embedding normalisation:"], className="col-md-auto"),
                    dbc.Col([
                        dbc.Select(
                            id="norm_emb",
                            options=NORM_MAP,
                            value=DEFAULT_VIS_CONFIG["norm"],
                            className="form-select borderpx-1 w-100"
                        ),
                    ]),
                ], className="my-2 mx-1 my-1 d-flex align-items-center"),
                dbc.Row([
                    dbc.Col(["Secondary token decoding:"], className="col-md-auto"),
                    dbc.Col([
                        dbc.Select(
                            id="secondary_decoding",
                            options=SECONDARY_DECODING_MAP,
                            value=DEFAULT_VIS_CONFIG["secondary_decoding"],
                            className="form-select borderpx-1 w-100"
                        ),
                    ]),
                ], className="my-2 mx-1 my-1 d-flex align-items-center"),
                dbc.Row([
                    dbc.Checklist(
                        [{"label": "Hide <start> token", "value": "hide"}],
                        id="hide_start_table",
                        value=["hide"] if DEFAULT_TABLE_VIS_CONFIG["hide_start"] else [],
                        labelStyle={"float": "left"},
                        switch=True,
                    ),
                ], className="my-1 mx-1"),
            ], className="col-4"),
            dbc.Col([
                _modes(),
            ], className="col-6"),
        ])
    ])

def _settings_sankey():
    return dbc.Col([
        dbc.Row([
            html.H5("Sankey")
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Checklist(
                    [{"label": "Hide <start> token", "value": "hide"}],
                    id="hide_start_sankey",
                    value=["hide"] if DEFAULT_SANKEY_VIS_CONFIG["hide_start"] else [],
                    labelStyle={"float": "left"},
                    switch=True,
                ),
                dbc.Row([
                    html.Div(["⤷"], className="w-5 ms-3 mb-2 pe-1"),
                    dbc.Checklist(
                        [{"label": "Reapport weights to remaining nodes", "value": "reapport"}],
                        id="reapport_start",
                        value=["reapport"] if DEFAULT_SANKEY_VIS_CONFIG["reapport_start"] else [],
                        labelStyle={"float": "left"},
                        switch=True,
                        className="w-80 ps-0"
                    ),
                ], className="align-items-center fade-slide", id="reapport_start_div"),
                dbc.Checklist(
                    [{"label": "Hide labels for intermediate nodes", "value": "hide"}],
                    id="hide_labels",
                    value=["hide"] if DEFAULT_SANKEY_VIS_CONFIG["sankey_parameters"]["only_nodes_labels"] else [],
                    labelStyle={"float": "left"},
                    switch=True,
                ),
            ]),
            dbc.Col([
                dbc.Row([
                    dbc.Col(["Attention Highlight:"], className="col-md-auto"),
                    dbc.Col([
                        dbc.Select(
                            id="attention_select",
                            options=ATTENTION_MAP,
                            value=DEFAULT_ATTENTION,
                            className="form-select borderpx-1 w-100"
                        ),
                    ]),
                ], className="ms-1 me-2 mt-2 align-items-center"),
                dbc.Row([
                    dbc.Row([
                        html.Div(["⤷"], className="align-items-center w-10 ms-4"),
                        dbc.Input(
                            value=DEFAULT_ATT_HIGH_K,
                            type="number", min=0, max=25, id="att_high_k", className="w-10",
                        ),
                        html.Label("Top K attention traces", className="w-50"),
                    ], className="align-items-center fade-slide", id="att_high_k_div"),
                    dbc.Row([
                        html.Div(["⤷"], className="offset-arrow w-10 ms-4"),
                        dbc.Input(
                            value=DEFAULT_ATT_HIGH_W,
                            type="number", min=0, max=1, step=0.001, id="att_high_w", className="w-10",
                        ),
                        html.Label("Minimum attention weight", className="w-50"),
                    ], className="align-items-center fade-slide", id="att_high_w_div"),
                ], className="ms-1 me-2 my-1"),
            ]),
            dbc.Col([
                dbc.Row([
                    dbc.Input(
                        id="row_limit", type="number", value=DEFAULT_SANKEY_VIS_CONFIG["sankey_parameters"]["rowlimit"],
                        min=1, max=1,
                        className="w-20",
                    ),
                    html.Label("N° of output layers to show", className="w-80"),
                ], className="mx-2 d-flex align-items-center"),
                dbc.Row([
                    html.Label("Sankey scaling options:", className="w-30 px-0"),
                    dcc.Slider(
                        0, 1, value=DEFAULT_SANKEY_SCALE,
                        step=0.01, marks={0: "0%", 1: "100%"}, id="sankey_scale", className="w-30 py-0"
                    ),
                    dbc.Select(
                        id="sankey_size_adapt",
                        options=SANKEY_SIZE_MAP,
                        value=DEFAULT_SANKEY_VIS_CONFIG["sankey_parameters"]["size_adapt"],
                        className="form-select borderpx-1 w-40"
                    ),
                ], className="mx-2 my-2 d-flex align-items-center"),
            ]),
        ], className="d-flex align-items-center"),
    ])

def _stores():
    return (
        dcc.Store(id="table_scroll", data=0),
        dcc.Store(id="run_config", data=DEFAULT_RUN_CONFIG),
        dcc.Store(id="current_run_config"),
        dcc.Store(id="click_data_store", data={}),
        dcc.Store(id="vis_config", data=DEFAULT_VIS_CONFIG),
        dcc.Store(id="sankey_vis_config", data=DEFAULT_SANKEY_VIS_CONFIG),
        dcc.Store(id="table_vis_config", data=DEFAULT_TABLE_VIS_CONFIG),
        dcc.Store(id="generation_notify"),
        dcc.Store(id="new_model_notify"),
        dcc.Store(id="injection_card_id", data=0),
        dcc.Store(id="model_id", data=DEFAULT_MODEL_ID),
        dcc.Store(id="model_info", data=dataclasses.asdict(DEFAULT_MODEL)),
        dcc.Store(id="initial_callbacks", data=DEFAULT_INITIAL_CALLS),
        dcc.Store(id="session_id")
    )
