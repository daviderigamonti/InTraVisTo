from typing import List

import dataclasses
import pickle
import uuid
import time
import os

from dash import Dash, html, dcc, callback, Output, Input, State, ctx
from transformers import AutoTokenizer, GenerationConfig
from torch import bfloat16, cuda
from scipy.special import kl_div  # (ufuncs in scipy.special are written in C) pylint:disable=E0611

import transformers
import diskcache
import torch
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

from inject import INJECTS_PARAMETER, InjectCausalLMWrapper, InjectInfo, InjectPosition
from sankey import SankeyParameters, generate_complete_sankey, generate_sankey, format_sankey
from utils import EmbeddingTypes, CellWrapper, LayerWrapper, Decoder
from app.constants import *  # pylint:disable=W0401,W0614
from app.defaults import * # pylint:disable=W0401,W0614


def generate_layout(model_config):
    return html.Div([
        html.H3('InTraVisTo', style={'display': 'inline-block', 'margin-right': 10}),
        html.Hr(),
        html.Div([
            html.Div([
                html.H4('Input text', style={'display': 'inline-block', 'margin-right': 10}),
                dcc.Input(
                    placeholder=DEFAULT_QUESTION,
                    type='text',
                    value=DEFAULT_QUESTION,
                    style={"width": "500px"},
                    id="text",
                    debounce=False  # Needed otherwise textbox gets reset every time a callback resolves
                ),
                html.Button("Generate", id="generate_button"),
            ]),

            html.Hr(),
            html.Div([
                html.H4('Output text', style={'display': 'inline-block', 'margin-right': 10}),
                dcc.Textarea(
                    style={"width": "500px"},
                    id="output_text"
                )
            ]),
            html.Hr(),
            html.Div(children=[
                html.H4("Decoder used"),
                dcc.RadioItems(options=STRATEGY_MAP, value=DEFAULT_STRATEGY, id='choose_decoding')
            ], style={'marginTop': '5px', 'marginLeft': '40px', 'marginRight': '40px', 'display': 'inline-block', 'text-align': 'justify', 'vertical-align': 'top'}),
            html.Div(children=[
                html.H4("Embedding shown"),
                dcc.RadioItems(options=EMB_TYPE_MAP, value=DEFAULT_EMB_TYPE, id='choose_embedding')
            ], style={'marginTop': '5px', 'marginLeft': '40px', 'marginRight': '40px', 'display': 'inline-block', 'text-align': 'justify', 'vertical-align': 'top'}),
            html.Div(children=[
                html.H4("Colour"),
                dcc.RadioItems(options=['P(argmax term)', 'Entropy[p]', 'Att Contribution %',
                            'FF Contribution %'], value='P(argmax term)', id='choose_colour')
            ], style={'marginTop': '5px', 'marginLeft': '40px', 'marginRight': '40px', 'display': 'inline-block', 'text-align': 'justify', 'vertical-align': 'top'}),
            html.Div(children=[
                html.H4("Settings"),
                dcc.Checklist(
                    [{"label": "hide starting token", "value": "hide"}],
                    id="hide_col", value=["hide"] if DEFAULT_TABLE_VIS_CONFIG["hide_col"] else [], labelStyle={"float": "left"}
                ),
                dcc.Checklist(
                    [{"label": "hide starting token (Sankey)", "value": "hide"}],
                    id="hide_0", value=["hide"] if not DEFAULT_SANKEY_VIS_CONFIG["sankey_parameters"]["show_0"] else [], labelStyle={"float": "left"}
                ),
                dcc.Checklist(
                    [{"label": "hide non-layer tokens (Sankey)", "value": "hide"}],
                    id="hide_labels", value=["hide"] if DEFAULT_SANKEY_VIS_CONFIG["sankey_parameters"]["only_nodes_labels"] else [], labelStyle={"float": "left"}
                ),
                html.Div(children=[
                    dcc.Input(id="max_new_tokens", type='number',
                            value=DEFAULT_RUN_CONFIG["max_new_tok"], min=0, max=1024, style={"width": "60px"}),
                    html.Label("#tokens generated"),
                ]),
                html.Div(children=[
                    dcc.Input(id="font_size", type='number', value=DEFAULT_FONT_SIZE,
                            min=1, max=72, style={"width": "60px"}),
                    html.Label("font size"),
                ]),
                html.Div(children=[
                    dcc.Input(id="row_limit", type='number',
                            value=DEFAULT_SANKEY_VIS_CONFIG["sankey_parameters"]["rowlimit"], min=1, max=model_config.num_hidden_layers, style={"width": "60px"}),
                    html.Label("Sankey depth"),
                ]),
            ], style={'marginTop': '5px', 'marginLeft': '40px', 'marginRight': '40px', 'display': 'inline-block', 'text-align': 'justify', 'vertical-align': 'top'}),
        ], style={'marginTop': '5px', 'marginLeft': '40px', 'display': 'inline-block', 'text-align': 'justify', 'vertical-align': 'top'}),
        html.Hr(),
        html.Div(children=[
            dcc.Loading(id="loading-2", children=[dcc.Graph(figure=DEFAULT_FIGURE,
                        id='main_graph')], type="circle", overlay_style={"visibility": "visible"}),
        ], id="tooltip-target"),
        html.Hr(),
        dcc.Loading(id="loading-3", children=[dcc.Graph(figure=DEFAULT_FIGURE, id='sankey_graph')], type="circle"),
        dcc.Store(id="run_config", data=DEFAULT_RUN_CONFIG),
        dcc.Store(id="current_run_config"),
        dcc.Store(id="vis_config", data=DEFAULT_VIS_CONFIG),
        dcc.Store(id="sankey_vis_config", data=DEFAULT_SANKEY_VIS_CONFIG),
        dcc.Store(id="table_vis_config", data=DEFAULT_TABLE_VIS_CONFIG),
        dcc.Store(id="generation_notify"),
        dcc.Store(id="session_id"),
        dbc.Tooltip(id="graph-tooltip", target="tooltip-target", is_open=False,
                    flip=False, placement="top", autohide=False),
    ])