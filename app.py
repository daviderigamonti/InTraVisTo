# Import packages

from dataclasses import dataclass, field
from itertools import cycle
from typing import List

import dataclasses
import pickle
import math
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
import plotly.express as px
import pandas as pd
import numpy as np

from inject import INJECTS_PARAMETER, InjectCausalLMWrapper, InjectInfo, InjectPosition
from utils import EmbeddingTypes, CellWrapper, LayerWrapper, Decoder


def extract_diff_kl(wrappers, emb_type):

    def diff_kl(x):
        vals = [sum(kl_div(x[n], x[n+1])) for n in range(x.shape[0]-1)]
        return vals

    df = pd.DataFrame(wrappers).apply(lambda x: [
        torch.nn.functional.softmax(
            xx.get_embedding(emb_type),
            dim=-1
        ).float().detach().cpu()
        for xx in x
    ])

    return df.apply(diff_kl, axis=0)


def extract_layer_n_secondary(x, emb_type, strategy):
    return [decoder.decode_secondary_tokens(t, s, e) if type(e) == LayerWrapper else [" "] for e, t, s in zip(x, emb_type, strategy)]


def extract_diff_text(wrappers, emb_type, strategy):
    return pd.DataFrame(wrappers).diff().apply(lambda x: extract_layer_n_secondary(x, [emb_type]*len(x), [strategy]*len(x))).sort_index(ascending=False)

# Most likely tokens


def extract_layer_n(x, emb_type, strategy):
    return [decoder.decode_hidden_state(t, s, e) for e, t, s in zip(x, emb_type, strategy)]


def extract_text(wrappers, emb_type, strategy):
    return pd.DataFrame(wrappers).apply(lambda x: extract_layer_n(x, [emb_type]*len(x), [strategy]*len(x)))

# Probabilities


def extract_layer_prob(x, decoding_strategy):
    return [e.get_probability(d) for e, d in zip(x, decoding_strategy)]


def extract_probabilities(wrappers, strategy):
    return pd.DataFrame(wrappers).apply(lambda x: extract_layer_prob(x, [strategy]*len(x)))

# Secondary tokens


def extract_secondary_tokens(layers: List[List[LayerWrapper]], emb_type: EmbeddingTypes, strategy: str):
    # Compute secondary tokens
    secondary_tokens = []
    for row in layers:
        row_secondary = []
        for single_layer in row:
            representations = decoder.decode_secondary_tokens(target_hidden_state=emb_type,
                                                              decoding=strategy,
                                                              layer=single_layer)
            # single_layer.add_secondary_tokens(representations, emb_type=emb_type)
            row_secondary.append(representations)
        secondary_tokens.append(row_secondary)
    return secondary_tokens


def pad_masked_attentions(attentions, max_len):
    new_attentions = []
    for att in attentions:
        padding_length = max_len - att.size(0)
        if padding_length > 0:
            padded_att = torch.cat([att, torch.zeros(padding_length, dtype=att.dtype, device=att.device)])
            new_attentions.append(padded_att)
        else:
            new_attentions.append(att)
    return torch.stack(new_attentions)


def compute_batch_complete_padded_attentions(generated_output, heads):
    multi_layer_head_attentions = []
    for head in heads:
        multi_layer_attentions = []
        for layer in range(0, len(generated_output["attentions"][0])):
            # Prompt tokens
            prompt_att = [
                torch.squeeze(single_head)
                for single_head in torch.squeeze(torch.select(generated_output["attentions"][0][layer], 1, head))
            ]
            # Response tokens
            response_att = [
                torch.squeeze(torch.select(single_layer[layer], 1, head))
                for single_layer in generated_output["attentions"][1:]
            ]
            # Pad and merge attentions
            multi_layer_attentions.append(pad_masked_attentions(
                [att_token for att_token in prompt_att + response_att],
                len(response_att[-1])
            ))
        multi_layer_head_attentions.append(multi_layer_attentions)
    return multi_layer_head_attentions


@dataclass
class SankeyParameters:
    # DATA
    # Row of starting token (where 0 corresponds to the top row, and n_layers - 1 corresponds to the bottom row)
    row_index: int = 0
    token_index: int = 9  # Position index of starting token (where 0 is first token of the input sequence)
    rowlimit: int = 5  # Limit number of layers to visualize
    multirep: bool = True  # Accomodate for each token having multiple labels
    show_0: bool = True
    # COLORS
    colormap: list[str, ...] = field(default_factory=lambda: ["#206DE8"])  # Colors
    # colormap: list[str, ...] = field( default_factory = lambda : px.colors.qualitative.Plotly )
    color_change_count_threshold: int = 3  # Number of virtual rows that should have the same color associated to them
    color_brightness_range: tuple[float, float] = (-0.5, 0.2)  # Brightness range for tokens color gradient
    node_opacity: float = 0.7  # Opacity of nodes
    link_opacity: float = 0.4  # Opacity of links
    non_residual_link_color: tuple[int, int, int] = (100, 100, 100)  # Default color for non-resiudal links
    default_node_color: tuple[int, int, int] = (220, 220, 220)  # Default color for nodes
    color_nodes: bool = False  # If set to true, color nodes based on the colormap, otherwise all nodes will have their default color
    extra_brightness_map: dict[str, float] = field(
        default_factory=lambda: {"Node": -0.5, "FFNN": 0.15, "Attention": -0.15, "Intermediate": -0.3})
    # LAYOUT
    print_indexes: bool = False
    only_nodes_labels: bool = False
    rescale_factor: int = 3
    fixed_offsets: dict[str, float] = field(
        default_factory=lambda: {"Node": 0, "FFNN": 0.02, "Attention": 0.02, "Intermediate": 0})
    column_pad: float = None
    # Correction to avoid feeding nodes with a coordinate value of 0, which causes problems with Plotly Sankey Diagrams
    sankey_zero: float = 0.000000000000001
    font_size: float = 14  # Text font size
    size: int = 1800  # Size of square canvas


def cumulative_sankey_traces(
    dfs, linkinfo,            # Dataframes and link info to access labels and node hidden information
    row, indexes, el_indexes,  # Dataframe is indexed by index and row, while el_index references the index for sankey visualization
    bases,                    # Base attention value of parents
    labels,                   # Current set of labels for sankey visualization
    # Reference for duplicate nodes as a dictionary indexed with (row, index) and containing a dictionary composed of
    elmap,
                              #  an id and a base
    rowlimit,                 # Depth limit
    firstcall=True,           # Identify if it's the original function call
):
    new_labels = []
    new_indexes = []
    new_elmap = elmap.copy()  # TODO: copy necessary?

    under = []
    over = []
    val = []
    types = []
    # Calculate current value of node by weighting its attention value for the parent's weight
    for index, el_index, base in zip(indexes, el_indexes, bases):
        res_w = linkinfo["attn_res_percent"][-(row + 1)][index]  # .item()
        res_w += 0.0000000001 if res_w == 0.0 else (-0.0000000001 if res_w == 1.0 else 0)  # Prevent 0
        attn_w = 1 - res_w
        resattn_w = linkinfo["ffnn_res_percent"][-(row + 1)][index]  # .item()
        resattn_w += 0.0000000001 if resattn_w == 0.0 else (-0.0000000001 if resattn_w == 1.0 else 0)  # Prevent 0
        mlp_w = 1 - resattn_w
        # Create MLP / Attention / Intermediate nodes
        mlp_index = len(new_elmap.keys())
        new_labels.append(dfs["ffnn"].iloc[row+1][index] if dfs["ffnn"] is not None else ["FFNN"])
        new_elmap[(round(row + 1 - 0.8, 2), round(index - 0.5, 2))
                  ] = {"id": mlp_index, "base": base * mlp_w, "type": "FFNN"}
        attn_index = len(new_elmap.keys())
        new_labels.append(dfs["attention"].iloc[row+1][index] if dfs["attention"] is not None else ["Attention"])
        new_elmap[(round(row + 1 - 0.45, 2), round(index - 0.5, 2))
                  ] = {"id": attn_index, "base": base * attn_w, "type": "Attention"}
        hid_index = len(new_elmap.keys())
        new_labels.append(dfs["intermediate"].iloc[row+1][index] if dfs["intermediate"] is not None else ["-"])
        new_elmap[(round(row + 1 - 0.65, 2), index)] = {"id": hid_index, "base": base, "type": "Intermediate"}
        # Iterate over all elements of the next row
        for i, label in enumerate(dfs["states"].iloc[row+1].tolist()):
            v = base * attn_w * linkinfo["attentions"][row][index][i].item()
            if v > 0:
                over.append(attn_index)
                # If node is already present store its information
                if (row+1, i) in new_elmap:
                    under.append(new_elmap[(row+1, i)]["id"])
                    new_elmap[(row+1, i)]["base"] += v
                # If the node is new create a new entry in the element map with a new sankey index
                else:
                    new_index = len(new_elmap.keys())
                    new_labels.append(label)
                    new_indexes.append(i)
                    under.append(new_index)
                    new_elmap[(row+1, i)] = {"id": new_index, "base": v, "type": "Node"}
                val.append(v)
                types.append("att_in")
        # MLP State
        over.append(el_index)
        under.append(mlp_index)
        val.append(base * mlp_w)
        types.append("mlp_out")
        over.append(mlp_index)
        under.append(hid_index)
        val.append(base * mlp_w)
        types.append("mlp_in")
        # Attention State
        over.append(hid_index)
        under.append(attn_index)
        val.append(base * attn_w)
        types.append("att_out")
        # Residuals
        over.append(hid_index)
        under.append(new_elmap[(row+1, index)]["id"])
        val.append(base * res_w)
        types.append("residual")
        new_elmap[(row+1, index)]["base"] += base * res_w
        over.append(el_index)
        under.append(hid_index)
        val.append(base * resattn_w)
        types.append("residual")

    # If depth limit is reached, stop recurring
    if row + 1 < rowlimit:
        # Call itself on all the new nodes
        nex_under, nex_over, nex_val, nex_types, nex_labels, new_elmap = cumulative_sankey_traces(
            dfs, linkinfo,
            row+1, new_indexes, [new_elmap[(row+1, i)]["id"] for i in new_indexes],
            [new_elmap[(row+1, i)]["base"] for i in new_indexes],
            new_labels,
            new_elmap,
            rowlimit,
            firstcall=False,
        )
        # Update elements map, sankey trace lists and sankey labels list with children's results
        new_labels += nex_labels
        under += nex_under
        over += nex_over
        val += nex_val
        types += nex_types
    # Only executed at topmost level
    if firstcall:
        # Complete sankey labels list with starting label
        new_labels = labels + new_labels
    return under, over, val, types, new_labels, new_elmap


def generate_complete_sankey(dfs, linkinfo, sankey_parameters: SankeyParameters, gen_length):

    offset_0 = 0 if sankey_parameters.show_0 else 1
    last_token = sankey_parameters.token_index

    row_index = sankey_parameters.row_index
    token_indexes = range(last_token - gen_length + 1, last_token + 1)
    token_el_indexes = range(0, gen_length)
    token_labels = [dfs["states"].iloc[row_index].iloc[token_index] for token_index in token_indexes]
    token_base_val = 1.0 / gen_length
    elmap = {
        (row_index, tidx): {"id": telidx, "base": token_base_val, "type": "Node"}
        for tidx, telidx in zip(token_indexes, token_el_indexes)
    }

    if not sankey_parameters.show_0:
        linkinfo["attentions"] = [[[
            torch.tensor([0], device="cpu") if i == 0 or j == 0 else e2 for j, e2 in enumerate(e1)
        ] for i, e1 in enumerate(row)
        ] for row in linkinfo["attentions"]
        ]

    # Generate diagram data
    under, over, values, types, labels, elmap = cumulative_sankey_traces(
        dfs, linkinfo,
        row_index, token_indexes, token_el_indexes,
        [token_base_val] * gen_length,
        token_labels,
        elmap,
        sankey_parameters.rowlimit
    )

    return (under, over, values, types, labels, elmap)


def generate_sankey(dfs, linkinfo, sankey_parameters: SankeyParameters):

    row_index = sankey_parameters.row_index
    token_index = sankey_parameters.token_index
    token_label = dfs["states"].iloc[row_index].iloc[token_index]

    if not sankey_parameters.show_0:
        linkinfo["attentions"] = [[[
            torch.tensor([0], device="cpu") if i == 0 or j == 0 else e2 for j, e2 in enumerate(e1)
        ] for i, e1 in enumerate(row)
        ] for row in linkinfo["attentions"]
        ]

    # Generate diagram data
    under, over, values, types, labels, elmap = cumulative_sankey_traces(
        dfs, linkinfo,
        row_index, [token_index], [0],
        [1.0],
        [token_label],
        {(row_index, token_index): {"id": 0, "base": 1.0, "type": "Node"}},
        sankey_parameters.rowlimit
    )
    return (under, over, values, types, labels, elmap)

# Rescales values of a list inside a given range, if invert is set to True, the range is flipped


def rescale_list(l, range_min=0, range_max=1, old_min=None, old_max=None, invert=False):
    if old_max == None:
        old_max = max([i for i in l if i is not None])
    if old_min == None:
        old_min = min([i for i in l if i is not None])
    old_range = old_max - old_min
    new_range = range_max - range_min

    # Avoid division by 0
    if old_range == 0:
        old_range = 1
        range_min = new_range / 2

    invert_k = 0
    invert_a = 1
    if invert:
        invert_k = old_max
        invert_a = -1

    return [el if el == None else range_min + (((invert_k + (invert_a * (el - old_min))) * new_range) / old_range) for el in l]

# Given a list and a list of indexes that have been previously sorted, restore the original order of the list


def restore_list_order(l, indexes):
    return [l[indexes.index(i)] for i in range(0, len(indexes))]

# Return a list of RGBA color strings given a list of RGBA colors tuples


def build_rgba_from_tuples(l, opacity=1.0):
    return [f"rgba{tuple(el) + (opacity,)}" if len(el) == 3 else f"rgba{el}" for el in l]


def change_color_brightness(rgb_color, brightness):
    delta_color = tuple([int((channel) * brightness) for channel in rgb_color])
    return tuple([sum(channel) for channel in zip(rgb_color, delta_color)])


def format_sankey(un, ov, vl, types, lab, elmap, linkinfo, sankey_parameters: SankeyParameters):
    # Handle multiple labels for tokens with multiple representations
    typemap = [next(v["type"] for k, v in elmap.items() if v["id"] == i) for i in range(len(elmap.keys()))]
    nodes_extra = [
        {"text": l[0], "diff": "Diff from previous layer:" + " ".join(l[1])} if t in ["Node"]
        else {"text": l, "diff": ""}
        for l, t in zip(lab, typemap)
    ]
    if sankey_parameters.multirep:
        lab = [l[0][0] if t in ["Node"] else l[0] for l, t in zip(lab, typemap)]
    else:
        lab = [np.squeeze(l[0]) if t in ["Node"] else np.squeeze(l) for l, t in zip(lab, typemap)]
        # lab = [np.squeeze(l).item() for l in lab]

    # Generate numbered labels
    lab = [f"{k[1]} {lab[el['id']]}" if sankey_parameters.print_indexes and el["type"]
           in ["Node"] else lab[el['id']] for k, el in elmap.items()]

    # Remove extra labels not belonging to nodes
    if sankey_parameters.only_nodes_labels:
        lab = [l if t in ["Node"] else "" for l, t in zip(lab, typemap)]

    # Add non-rescaled info to links and nodes extra information
    for k, el in elmap.items():
        nodes_extra[el["id"]] = nodes_extra[el["id"]] | {"v": el["base"]}
    links_extra = [{"v": v, "type": t} for v, t in zip(vl, types)]

    # Rescale node and link values by a rescale factor to fit into graph
    rescale_factor = sankey_parameters.rescale_factor
    rescaled_elmap = {k: el | {"base": el["base"] / rescale_factor} for k, el in elmap.items()}
    rescaled_vl = [el / rescale_factor for el in vl]

    # rescaled_vl = [el.cpu() for el in rescaled_vl]
    # Create reverse mapping obtaining lists indexed by the node id and containing virtual coordinates and node values
    revmap = [next(k for k, v in rescaled_elmap.items() if v["id"] == i) for i in range(len(rescaled_elmap.keys()))]
    revmap_values = [next(v for k, v in rescaled_elmap.items() if v["id"] == i)
                     for i in range(len(rescaled_elmap.keys()))]
    revmap_x = [key[0] for key in revmap]
    revmap_y = [key[1] for key in revmap]
    # Sort reverse-mapped lists to perform transformations on them with more ease, while keeping an index list to reverse the sorting
    revmap_indexes = [i for i in range(0, len(revmap))]
    revmap_x_sort, revmap_y_sort, revmap_values_sort, revmap_indexes = zip(
        *sorted(zip(revmap_x, revmap_y, revmap_values, revmap_indexes), key=lambda x: x[0]))

    # Add kl divergence values to residual links between consecutive layers
    # TODO: clamping infinite values to max value
    max_kl = linkinfo["kl_diff"].replace([np.inf, -np.inf], np.nan).max(skipna=True).max()
    def checkinf(x): return x if not np.isinf(x) else max_kl
    kl_values = [
        checkinf(linkinfo["kl_diff"][math.ceil(revmap_y[el])]
                 [linkinfo["kl_diff"].shape[0] - math.ceil(revmap_x[el])]).item()
        if typ in ["residual"] and math.ceil(revmap_y[el]) > 0 else None
        for typ, el in zip(types, un)
    ]
    def format_kl(x): return "KL: {:.0f}m nats".format(x) if x >= 10 else "KL: {:.0f}μ nats".format(x * 1000)
    links_extra = [l | {"kl_diff": format_kl(kl * 1000) if kl != None else ""} for l, kl in zip(links_extra, kl_values)]

    # Build colors
    node_colors = []
    node_colors_ref = []
    link_colors = []
    colormap = cycle(sankey_parameters.colormap)
    current_color = next(colormap)
    old_x = -1
    change_count = sankey_parameters.color_change_count_threshold
    color_brightness_range = sankey_parameters.color_brightness_range
    # Node colors
    # for x, y, v in zip(revmap_x_sort, rescale_list(revmap_y_sort, range_min=color_brightness_range[0], range_max=color_brightness_range[1]), revmap_values_sort):
    for x, y, v in zip(revmap_y_sort, rescale_list(revmap_x_sort, range_min=color_brightness_range[0], range_max=color_brightness_range[1], invert=True), revmap_values_sort):
        # Color switching
        if x != old_x:
            if change_count > sankey_parameters.color_change_count_threshold:
                current_color = next(colormap)
                change_count = 0
            change_count += 1
        color_ref = change_color_brightness(px.colors.hex_to_rgb(current_color), y)
        node_colors_ref.append(color_ref)
        actual_color = sankey_parameters.default_node_color
        if sankey_parameters.color_nodes:
            actual_color = px.colors.hex_to_rgb(current_color)
        color = change_color_brightness(actual_color, y + sankey_parameters.extra_brightness_map[v["type"]])
        node_colors.append(color)
        old_x = x
    node_colors = restore_list_order(node_colors, revmap_indexes)
    node_colors_ref = restore_list_order(node_colors_ref, revmap_indexes)
    # Link colors
    link_colors = [node_colors_ref[el] if typ in ["residual", "att_in", "mlp_in"]
                   else sankey_parameters.non_residual_link_color for typ, el in zip(types, un)]
    link_colors = [
        # TODO hardcoded color
        # TODO kl values for color visualization are a bit off due to their range
        change_color_brightness((255, 99, 71), kl) if kl is not None else color
        for color, kl in zip(link_colors, rescale_list(kl_values, invert=True))
    ]
    # Convert colors and add opacities
    node_colors = build_rgba_from_tuples(node_colors, sankey_parameters.node_opacity)
    link_colors = build_rgba_from_tuples(link_colors, sankey_parameters.link_opacity)

    # Generate columns based on maximum node width for each column to fit nodes into
    zero_offset = 0 if sankey_parameters.show_0 else 1
    col_pad = sankey_parameters.column_pad
    columns_width = [max([v["base"] if y == y_index else 0 for (y, v) in zip(revmap_y_sort, revmap_values_sort)])
                     for y_index in range(zero_offset, max(revmap_y) + 1)]
    s = sum(columns_width)
    # Infer optimal column padding if not specified
    if col_pad == None:
        r = 1 - s
        col_pad = r / (len(columns_width) - 1) if r > 0 and len(columns_width) > 1 else 0
    s += col_pad * len(columns_width)
    columns_width = [w/s + col_pad for w in columns_width]
    columns_ys = []
    tot_w = 0
    for w in columns_width:
        columns_ys.append(tot_w)
        tot_w += w

    # Adjust coordinates
    revmap_x = rescale_list(revmap_x, range_min=sankey_parameters.sankey_zero, range_max=1, invert=False)
    revmap_y = [columns_ys[math.ceil(y) - zero_offset] + v["base"] / 2 -
                sankey_parameters.fixed_offsets[v["type"]] for y, v in zip(revmap_y, revmap_values)]

    fig = go.Figure(go.Sankey(
        orientation="v",
        arrangement="fixed",
        valueformat=".5r",
        node=dict(
            customdata=nodes_extra,
            hovertemplate="%{customdata.text}<br>%{customdata.diff}<extra>%{customdata.v:.1%}</extra>",
            align="left",
            label=lab,
            color=node_colors,
            x=revmap_x,
            y=revmap_y,
            pad=10000,
        ),
        link=dict(
            customdata=links_extra,
            hovertemplate="%{customdata.type} from %{target.label} to %{source.label}<br>%{customdata.kl_diff}<extra>%{customdata.v:.1%}</extra>",
            source=ov,
            target=un,
            value=rescaled_vl,
            color=link_colors
        )
    ))
    fig.update_layout(
        font_size=sankey_parameters.font_size, font_family="Verdana", font_color="black",
        width=sankey_parameters.size, height=sankey_parameters.size,
    )
    return fig

############################################### DASH APP ########################################


#################################################################################################
# Initialize the app
# external JavaScript files
external_scripts = [
    'https://www.google-analytics.com/analytics.js',
    {'src': 'https://cdn.polyfill.io/v2/polyfill.min.js'},
    {
        'src': 'https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.10/lodash.core.js',
        'integrity': 'sha256-Qqd/EfdABZUcAxjOkMi8eGEivtdTkh3b65xCZL4qAQA=',
        'crossorigin': 'anonymous'
    }
]

# external CSS stylesheets
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }
]

# app = Dash(__name__, external_scripts=external_scripts,
#                external_stylesheets=external_stylesheets)

cache = diskcache.Cache("./cache")
app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])


@app.callback(
    Output('run_config', 'data'),
    [
        Input('max_new_tokens', 'value'),
        Input("custom_emb", "value"),
        Input("custom_emb_location", "value"),
    ],
    State("vis_config", "data"),
    prevent_initial_call=True,
)
def update_run_config(max_new_tok, custom_emb, custom_emb_location, vis_config):
    return {
        "max_new_tok": max_new_tok,
        "inject_info": {
            "text": custom_emb,
            "location": custom_emb_location,
            "target_layer": vis_config["y"] - 1 if "y" in vis_config and vis_config["y"] != None else None,
            "target_token": vis_config["x"] if "x" in vis_config else None,
        }
    }


@app.callback(
    Output('vis_config', 'data'),
    [
        Input("generation_notify", "data"),
        Input("main_graph", "clickData"),
    ],
    [
        State('hide_col', 'value'),
        State('vis_config', 'data'),
    ],
    prevent_initial_call=True,
)
def update_vis_config(gen_notify, clickData, hide_col, vis_config):
    # If table visualization is hiding the first column, then offset all x-axis click data by 1
    col_0_offset = 1 if len(hide_col) > 0 else 0
    vis_config |= {"x": clickData["points"][0]["x"] + col_0_offset} if clickData else {"x": None}
    vis_config |= {"y": clickData["points"][0]["y"]} if clickData else {"y": None}
    if ctx.triggered_id == "generation_notify":
        vis_config |= {"x": None, "y": None}
    return vis_config


@app.callback(
    Output('sankey_vis_config', 'data'),
    [
        Input("generation_notify", "data"),
        Input("vis_config", "data"),
        Input('row_limit', 'value'),
        Input('hide_0', 'value'),
        Input("hide_labels", "value"),
        Input('font_size', 'value'),
    ],
    [
        State('hide_col', 'value'),
        State('sankey_vis_config', 'data'),
    ],
    prevent_initial_call=True,
)
def update_sankey_vis_config(gen_notify, vis_config, row_limit, hide_0, hide_labels, font_size, hide_col, sankey_vis_config):
    token = vis_config["x"] if "x" in vis_config and vis_config["x"] != None else 0
    layer = vis_config["y"] if "y" in vis_config and vis_config["y"] != None else 0
    row_limit = row_limit if layer - row_limit - 1 >= 0 or layer == 0 else layer
    hide_0 = len(hide_0) > 0
    hide_labels = len(hide_labels) > 0
    sankey_vis_config |= {
        "sankey_parameters": dataclasses.asdict(SankeyParameters(
            row_index=0,
            token_index=token,
            rowlimit=row_limit,
            show_0=not hide_0,
            font_size=font_size,
            only_nodes_labels=hide_labels,
        ))
    }
    return sankey_vis_config


@app.callback(
    Output('table_vis_config', 'data'),
    [
        Input('hide_col', 'value'),
        Input('font_size', 'value'),
    ],
    [
        State('table_vis_config', 'data'),
    ],
    prevent_initial_call=True,
)
def update_table_vis_config(hide_col, font_size, vis_config):
    vis_config |= {"font_size": font_size}
    vis_config |= {"hide_col": len(hide_col) > 0}
    return vis_config


@cache.memoize()
def model_generate(prompt, run_config, session):
    # model.model.init_debug_vectors()
    # input_residual_embedding = model.model.input_residual_embedding
    # attention_plus_residual_embedding = model.model.attention_plus_residual_embedding
    # post_attention_embedding = model.model.post_attention_embedding
    # post_FF_embedding = model.model.post_FF_embedding

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate
    # output = model(inputs.input_ids, return_dict=True, output_hidden_states=True, output_attentions=True)
    # print(f"Input len: {inputs.input_ids.shape}")
    # print(inputs.char_to_token)
    input_len = len(inputs.input_ids.squeeze().tolist())

    inject_info = dict(run_config["inject_info"]) if "inject_info" in run_config else None
    if (
        inject_info and
        "text" in inject_info and inject_info["text"] != None and inject_info["text"] != "" and
        "target_layer" in inject_info and inject_info["target_layer"] != None and
        "target_token" in inject_info and inject_info["target_token"] != None
    ):
        inj_token = tokenizer.encode(
            inject_info["text"],
            add_special_tokens=False,
            return_tensors="pt"
        ).to(device).squeeze()

        # Remove eventual prefix tokens
        mask = ~torch.isin(inj_token, prefix_tokens)
        inj_token = inj_token[mask]

        input_emb = model.get_input_embeddings()(inj_token)
        output_emb = model.lm_head.weight[inj_token]
        if inj_token.size()[-1] > 1:
            # TODO: multitoken embeddings are averaged by default
            print("Averaged multi-tokens for embedding injection")
            input_emb = input_emb.mean(dim=-2)
            output_emb = output_emb.mean(dim=-2)
        # TODO: injected embeddings are interpolated by default
        print("Interpolated embeddings for injection")
        n_layers, layer_n = model_config.num_hidden_layers, inject_info["target_layer"] + 1
        inject_info["embedding"] = ((n_layers - layer_n) * input_emb + layer_n * output_emb) / n_layers

        # TODO: inject info creation
        inject_translate = {
            EmbeddingTypes.BLOCK_OUTPUT : InjectPosition.OUTPUT,
            EmbeddingTypes.POST_ATTENTION : InjectPosition.ATTENTION,
            EmbeddingTypes.POST_FF : InjectPosition.FFNN,
            EmbeddingTypes.POST_ATTENTION_RESIDUAL : InjectPosition.INTERMEDIATE
        }
        inject_info = [
            InjectInfo(
                layer=inject_info["target_layer"],
                token=inject_info["target_token"],
                position=inject_translate[inject_info["location"]],
                embedding=inject_info["embedding"]
            )
        ]

    else:
        inject_info = None

    

    gen_config = GenerationConfig(
        pad_token_id=model.config.eos_token_id,
        max_new_tokens=run_config["max_new_tok"],
        return_attention_output=True,
        return_feed_forward_output=True,
        return_intermediate_hidden_states=True,
    )
    generation_result = model.generate(
        inputs.input_ids,
        generation_config=gen_config,
        return_inner_states=True,
        inject_info=inject_info
    )

    def standardize_wrapped_tensors(t):
        s = torch.stack(t, dim=0).squeeze().detach()
        return s

    output_len = generation_result["sequence_length"] - input_len
    generation_output = {
        "sequences": generation_result["output_ids"].squeeze(),
        "attentions": standardize_wrapped_tensors(generation_result["attention_weights"]).mean(dim=1)
    }

    # Create a list of LayerWrapper
    layers = []
    emb_type = EmbeddingTypes.BLOCK_OUTPUT

    hidden_states = standardize_wrapped_tensors(generation_result["hidden_states"])
    attention_outputs = standardize_wrapped_tensors(generation_result["attention_outputs"])
    feed_forward_outputs = standardize_wrapped_tensors(generation_result["feed_forward_outputs"])
    intermediate_hidden_state = standardize_wrapped_tensors(generation_result["intermediate_hidden_states"])

    # 1- Prepare matrix of input tokens hidden_state:  N_TOKENS x N_LAYER
    # input_hidden_states = generation_result["hidden_states"][0]

    per_token_layers = LayerWrapper(0)
    for tok_hs in hidden_states[0][:-1]:
        layer = CellWrapper()
        layer.add_embedding(tok_hs, EmbeddingTypes.BLOCK_OUTPUT)
        layer.add_embedding(tok_hs, EmbeddingTypes.POST_ATTENTION)
        layer.add_embedding(tok_hs, EmbeddingTypes.POST_FF)
        layer.add_embedding(tok_hs, EmbeddingTypes.POST_ATTENTION_RESIDUAL)
        per_token_layers.cells.append(layer)
    layers.append(per_token_layers)

    # Iterate over layers
    for layer_id, (layer_hs, layer_att, layer_ffnn, layer_inter) in enumerate(zip(hidden_states[1:], attention_outputs, feed_forward_outputs, intermediate_hidden_state)):
        # Iterate over tokens
        per_token_layers = LayerWrapper(layer_id + 1)

        for tok_hs, tok_att, tok_ffnn, tok_inter in zip(layer_hs[:-1], layer_att[:-1], layer_ffnn[:-1], layer_inter[:-1]):
            layer = CellWrapper()
            layer.add_embedding(tok_hs, EmbeddingTypes.BLOCK_OUTPUT)
            layer.add_embedding(tok_att, EmbeddingTypes.POST_ATTENTION)
            layer.add_embedding(tok_ffnn, EmbeddingTypes.POST_FF)
            layer.add_embedding(tok_inter, EmbeddingTypes.POST_ATTENTION_RESIDUAL)
            per_token_layers.cells.append(layer)
        layers.append(per_token_layers)

    for layer_hs, layer in zip(hidden_states, layers[1:]):
        for tok_hs, layer_token in zip(layer_hs, layer):
            layer_token.add_embedding(tok_hs, EmbeddingTypes.BLOCK_INPUT)

    # compute residuals contributions percentages
    for layer in layers[0]:
        layer.add_probability(0.0, "att_res_perc")
        layer.add_probability(0.0, "ff_res_perc")

    for row in layers[1:]:
        for single_layer in row:
            initial_residual = single_layer.get_embedding(EmbeddingTypes.BLOCK_INPUT)
            att_emb = single_layer.get_embedding(EmbeddingTypes.POST_ATTENTION)
            contribution = initial_residual.norm(
                2, dim=-1) / (initial_residual.norm(2, dim=-1) + att_emb.norm(2, dim=-1))
            final_contribition = round(contribution.squeeze().tolist(), 2)
            single_layer.add_probability(final_contribition, "att_res_perc")

    for row in layers[1:]:
        for single_layer in row:
            final_residual = single_layer.get_embedding(EmbeddingTypes.POST_FF)
            att_res_emb = single_layer.get_embedding(EmbeddingTypes.POST_ATTENTION_RESIDUAL)
            contribution = att_res_emb.norm(2, dim=-1) / (att_res_emb.norm(2, dim=-1) + final_residual.norm(2, dim=-1))
            final_contribition = round(contribution.squeeze().tolist(), 2)
            single_layer.add_probability(final_contribition, "ff_res_perc")

    a = time.time()
    l = pickle.dumps(layers)
    b = time.time()
    print(f"model_generate pickling time: {b - a}")
    a = time.time()
    l = pickle.loads(l)
    b = time.time()
    print(f"model_generate unpickling time: {b - a}")

    return generation_output, layers, input_len, output_len


@callback(
    [
        Output('session_id', 'data'),
        Output('current_run_config', 'data'),
        Output("generation_notify", "data"),
    ],
    Input("generate_button", "n_clicks"),
    [
        State('text', 'value'),
        State('run_config', 'data'),
    ],
)
def call_model_generate(button, text, run_config):
    session_id = str(uuid.uuid4())
    _, _, _, _ = model_generate(text, run_config, session_id)
    return session_id, run_config, True


@callback(
    [
        Output('main_graph', 'figure'),
        Output('output_text', 'value'),
    ],
    [
        Input('generation_notify', 'data'),
        Input('choose_decoding', 'value'),
        Input('choose_embedding', 'value'),
        Input("choose_colour", "value"),
        Input('table_vis_config', 'data'),
        Input("vis_config", "data")
    ],
    [
        State('session_id', 'data'),
        State('current_run_config', 'data'),
        State('text', 'value'),
    ],
    prevent_initial_call=True,
)
def update_graph(notify, strategy, emb_type, choose_colour, tab_vis_config, vis_config, session_id, run_config, text):

    # Retrieve model outputs
    generated_output, layers, input_len, output_len = model_generate(text, run_config, session_id)

    if choose_colour == "P(argmax term)":
        colour = strategy
    elif choose_colour == "Entropy[p]":
        colour = "entropy"
    elif choose_colour == "Att Contribution %":
        colour = "att_res_perc"
    elif choose_colour == "FF Contribution %":
        colour = "ff_res_perc"

    # Remove first column from visualization
    if tab_vis_config["hide_col"]:
        layers = [layer[1:] for layer in layers]

    # Compute secondary tokens
    secondary_tokens = extract_secondary_tokens(layers, emb_type=emb_type, strategy=strategy)

    text = extract_text(layers, emb_type, strategy).values.tolist()

    p = extract_probabilities(layers, colour).values.tolist()

    fig = go.Figure(data=go.Heatmap(
                    z=p,
                    text=pd.DataFrame(secondary_tokens),
                    xgap=2,
                    ygap=2,
                    x=[i - 0.5 for i in range(0, input_len + output_len)],
                    y=list(range(0, n_col)),
                    hovertemplate='<i>Probability</i>: %{z:.2f}%' +
                    '<br><b>Layer</b>: %{y}<br>' +
                    '<br><b>Number of token</b>: %{x}<br>' +
                    '<br><b>Secondary representations</b>: %{text}<br>' +
                    '<extra></extra>',
                    texttemplate="%{text[0]}",
                    textfont={"size": tab_vis_config["font_size"]},
                    colorscale="blues"))
    fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=5),
        height=1000,
        yaxis=dict(
            title_text="Transformer Layers",
            tickmode="linear",
            titlefont=dict(size=20),
        ),
        xaxis=dict(
            title_text="Token Position",
            tickmode="linear",
            titlefont=dict(size=20),
        ),
        template="plotly")
    fig.add_vline(x=input_len - 1 - 1 - 0.5, line_width=8, line_color='white')
    fig.add_vline(x=input_len - 1 - 1 - 0.5, line_width=2, line_color='darkblue')
    fig.add_hline(y=0.5, line_width=8, line_color='white')
    fig.add_hline(y=0.5, line_width=2, line_color='darkblue')
    if vis_config["x"] != None and vis_config["y"] != None:
        offset = 0 if tab_vis_config["hide_col"] else 1
        fig.add_shape(
            x0=vis_config["x"] + offset - 0.5, x1=vis_config["x"] + offset - 1.5,
            y0=vis_config["y"] - 0.5, y1=vis_config["y"] + 0.5,
            line_width=2, line_color="red"
        )
    # fig.add_hline(y=32, line_width=30, line_color='white')
    return fig, tokenizer.decode(generated_output["sequences"].squeeze()[input_len:])


@cache.memoize()
def generate_sankey_info(token, layer, session_id, run_config, text, strategy):
    generated_output, layers, input_len, output_len = model_generate(text, run_config, session_id)

    layers = layers[1:]

    if token == None or layer == None:
        layer = model_config.num_hidden_layers
        token = input_len + output_len - 2

    stop_layer = 0  # layer - sankey_param.rowlimit - 1
    interest_layers = [layer[0: token + 1] for layer in layers[stop_layer: layer + 1]]

    def make_df(emb_type):
        tokens = extract_secondary_tokens(layers=interest_layers, emb_type=emb_type, strategy=strategy)
        df = pd.DataFrame(tokens)
        df = df.sort_index(ascending=False)
        return df

    dfs = {
        "states": make_df(EmbeddingTypes.BLOCK_OUTPUT),
        "intermediate": make_df(EmbeddingTypes.POST_ATTENTION_RESIDUAL),
        "attention": make_df(EmbeddingTypes.POST_ATTENTION),
        "ffnn": make_df(EmbeddingTypes.POST_FF),
    }

    # Add labels for differences between consecutive layers
    diffs = extract_diff_text(interest_layers, EmbeddingTypes.BLOCK_OUTPUT, strategy)
    dfs["states"] = pd.DataFrame({col: zip(dfs["states"].loc[col], diffs[col]) for col in diffs.columns})

    attn_res_percent = extract_probabilities(layers, "att_res_perc").values.tolist()
    attn_res_percent = [el[0: token + 1] for el in attn_res_percent[stop_layer: layer + 1]]
    ffnn_res_percent = extract_probabilities(layers, "ff_res_perc").values.tolist()
    ffnn_res_percent = [el[0: token + 1] for el in ffnn_res_percent[stop_layer: layer + 1]]

    # attentions = compute_batch_complete_padded_attentions(generated_output, range(0, model_config.num_attention_heads))[-1]
    attentions = generated_output["attentions"]
    attentions = [[[e2 for e2 in e1[0: token + 1]] for e1 in row[0: token + 1]]
                  for row in attentions[stop_layer: layer + 1]]

    kl_diffs = extract_diff_kl(interest_layers, EmbeddingTypes.BLOCK_OUTPUT).sort_index(ascending=True)

    linkinfo = {"attentions": attentions, "attn_res_percent": attn_res_percent,
                "ffnn_res_percent": ffnn_res_percent, "kl_diff": kl_diffs}

    return dfs, linkinfo, token, layer, output_len


@app.callback(
    Output('sankey_graph', 'figure'),
    [
        Input("vis_config", "data"),
        Input('sankey_vis_config', 'data'),
        Input('choose_decoding', 'value'),
    ],
    [
        State('session_id', 'data'),
        State('current_run_config', 'data'),
        State('text', 'value'),
    ],
)
def update_sankey(vis_config, sankey_vis_config, strategy, session_id, run_config, text):
    if (
        "x" not in vis_config or "y" not in vis_config or
        vis_config["x"] == None or vis_config["y"] == None or
        vis_config["x"] <= 0 or vis_config["y"] <= 0
    ):
        x, y = None, None
    else:
        x, y = vis_config["x"], vis_config["y"]

    sankey_param = SankeyParameters(**sankey_vis_config["sankey_parameters"])

    dfs, linkinfo, token, layer, gen_len = generate_sankey_info(
        x, y, session_id, run_config, text, strategy
    )

    if x == None and y == None:
        sankey_param.token_index = token
        sankey_info = generate_complete_sankey(dfs, linkinfo, sankey_param, gen_len)
    else:
        sankey_info = generate_sankey(dfs, linkinfo, sankey_param)

    fig = format_sankey(*sankey_info, linkinfo, sankey_param)

    return fig


@callback(
    [
        Output("graph-tooltip", "is_open", allow_duplicate=True),
        Output("graph-tooltip", "trigger"),
        Output("graph-tooltip", "children", allow_duplicate=True),
    ],
    [
        Input("generation_notify", "data"),
        Input("main_graph", "clickData"),
    ],
    State("graph-tooltip", "trigger"),
    State("graph-tooltip", "autohide"),
    prevent_initial_call=True,
)
def display_embedding_tooltip(gen_notify, clickData, gtt, autohide):
    if clickData is None or ctx.triggered_id == "generation_notify":
        return False, "", []

    pt = clickData["points"][0]

    children = [
        html.Div([
            html.H2("Inject Custom Embedding"),
            html.Div([
                html.P(f"Layer: {clickData['points'][0]['y']}, Token: {clickData['points'][0]['x']}"),
                dcc.Input(
                    placeholder="Embedding to change",
                    type="text",
                    value="",
                    id="custom_emb",
                    debounce=False
                ),
            ], style={"display": "inline-block", "vertical-align": "top;"}),
            html.Div([
                dcc.RadioItems(options=emb_type_map, value=default_emb_type, id='custom_emb_location'),
            ], style={"display": "inline-block", "vertical-align": "top;"})
        ], style={'display': 'inline-block', 'text-align': 'justify', 'vertical-align': 'top'})
    ]

    return True, "hover focus", children

# @callback(
#     [
#         Output("graph-tooltip", "show", allow_duplicate=True),
#         Output("graph-tooltip", "children", allow_duplicate=True),
#     ],
#     Input("graph-tooltip", "hoverData"),
#     prevent_initial_call=True,
# )
# def delete_embedding_tooltip(hoverData):
#     print(hoverData)
#     if hoverData is None:
#         return False, []
#     else:
#         return no_update, no_update


##########################################################

# PANDAS
pd.set_option("display.max_columns", None)

# ENVIRONMENT VARIABLES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_HOME'] = "huggingface"
os.environ['TRANSFORMERS_CACHE'] = "huggingface"
hf_auth = os.environ["HF_TOKEN"]

# MODELS
# Llama model
# model_id = "meta-llama/Llama-2-7b-hf"
# Mistral model
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# DEVICES
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
# device = "cpu"

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    output_attentions=True,
    output_hidden_states=True,
    output_scores=True,
    token=hf_auth,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_auth)

# Compute prefix tokens (9 is a random number)
prefix_tokens = tokenizer.encode("9", add_special_tokens=False, return_tensors="pt").to(device).squeeze()
prefix_tokens = prefix_tokens[0] if prefix_tokens.size(dim=0) > 1 else torch.tensor([]).to(device)


MODEL_CONFIG = {
    "trust_remote_code": True,
    # "config": model_config,
    "device_map": device,
    "token": hf_auth,
    "torch_dtype": bfloat16,
}

TOKENIZER_CONFIG = {
    "token": hf_auth,
}

model = InjectCausalLMWrapper.from_pretrained(
    model_id, model_kwargs=MODEL_CONFIG,
    quantization_configs=bnb_config,
    tokenizer_name_or_path=model_id, tokenizer_kwargs=TOKENIZER_CONFIG,
)
model.enable_wrapper()

# question2 = "Q: What is the capital of Italy? A:"
# generated_output, output_len, input_len = generate_experiment(question2, model, tokenizer, device)
# layers = preprare_debug(model, generated_output, output_len, input_len)


# Run the app
# Incorporate data
decoder = Decoder(model=model, tokenizer=tokenizer, model_config=model_config)
n_col = model_config.num_hidden_layers + 1

strategy_map = [
    {"label": 'output_decoder', "value": 'output'},
    {"label": 'interpolated', "value": 'interpolation'},
    {"label": 'input_encoder', "value": 'input'},
]
emb_type_map = [
    {"label": 'Residual + FFNN', "value": EmbeddingTypes.BLOCK_OUTPUT},
    {"label": 'FFNN', "value": EmbeddingTypes.POST_FF},
    {"label": 'Residual + Self Attention', "value": EmbeddingTypes.POST_ATTENTION_RESIDUAL},
    {"label": 'Self Attention', "value": EmbeddingTypes.POST_ATTENTION},
]

default_question = "Q: What is the capital of Italy? A:"
default_emb_type = EmbeddingTypes.BLOCK_OUTPUT
default_strategy = "interpolation"

default_figure = go.Figure(layout={
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

default_font_size = 14
default_run_config = {"max_new_tok": 10}
default_vis_config = {}
default_sankey_vis_config = {
    "sankey_parameters": dataclasses.asdict(SankeyParameters(
        rowlimit=7,
        show_0=False,
        font_size=default_font_size,
        only_nodes_labels=True
    )),
}
default_table_vis_config = {
    "hide_col": True,
    "font_size": default_font_size,
}


# App layout
app.layout = html.Div([
    html.H3('InTraVisTo', style={'display': 'inline-block', 'margin-right': 10}),
    html.Hr(),
    html.Div([
        html.Div([
            html.H4('Input text', style={'display': 'inline-block', 'margin-right': 10}),
            dcc.Input(
                placeholder=default_question,
                type='text',
                value=default_question,
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
            dcc.RadioItems(options=strategy_map, value=default_strategy, id='choose_decoding')
        ], style={'marginTop': '5px', 'marginLeft': '40px', 'marginRight': '40px', 'display': 'inline-block', 'text-align': 'justify', 'vertical-align': 'top'}),
        html.Div(children=[
            html.H4("Embedding shown"),
            dcc.RadioItems(options=emb_type_map, value=default_emb_type, id='choose_embedding')
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
                id="hide_col", value=["hide"] if default_table_vis_config["hide_col"] else [], labelStyle={"float": "left"}
            ),
            dcc.Checklist(
                [{"label": "hide starting token (Sankey)", "value": "hide"}],
                id="hide_0", value=["hide"] if not default_sankey_vis_config["sankey_parameters"]["show_0"] else [], labelStyle={"float": "left"}
            ),
            dcc.Checklist(
                [{"label": "hide non-layer tokens (Sankey)", "value": "hide"}],
                id="hide_labels", value=["hide"] if default_sankey_vis_config["sankey_parameters"]["only_nodes_labels"] else [], labelStyle={"float": "left"}
            ),
            html.Div(children=[
                dcc.Input(id="max_new_tokens", type='number',
                          value=default_run_config["max_new_tok"], min=0, max=1024, style={"width": "60px"}),
                html.Label("#tokens generated"),
            ]),
            html.Div(children=[
                dcc.Input(id="font_size", type='number', value=default_font_size,
                          min=1, max=72, style={"width": "60px"}),
                html.Label("font size"),
            ]),
            html.Div(children=[
                dcc.Input(id="row_limit", type='number',
                          value=default_sankey_vis_config["sankey_parameters"]["rowlimit"], min=1, max=n_col-1, style={"width": "60px"}),
                html.Label("Sankey depth"),
            ]),
        ], style={'marginTop': '5px', 'marginLeft': '40px', 'marginRight': '40px', 'display': 'inline-block', 'text-align': 'justify', 'vertical-align': 'top'}),
    ], style={'marginTop': '5px', 'marginLeft': '40px', 'display': 'inline-block', 'text-align': 'justify', 'vertical-align': 'top'}),
    html.Hr(),
    html.Div(children=[
        dcc.Loading(id="loading-2", children=[dcc.Graph(figure=default_figure,
                    id='main_graph')], type="circle", overlay_style={"visibility": "visible"}),
    ], id="tooltip-target"),
    html.Hr(),
    dcc.Loading(id="loading-3", children=[dcc.Graph(figure=default_figure, id='sankey_graph')], type="circle"),
    dcc.Store(id="run_config", data=default_run_config),
    dcc.Store(id="current_run_config"),
    dcc.Store(id="vis_config", data=default_vis_config),
    dcc.Store(id="sankey_vis_config", data=default_sankey_vis_config),
    dcc.Store(id="table_vis_config", data=default_table_vis_config),
    dcc.Store(id="generation_notify"),
    dcc.Store(id="session_id"),
    dbc.Tooltip(id="graph-tooltip", target="tooltip-target", is_open=False,
                flip=False, placement="top", autohide=False),
])


if __name__ == '__main__':

    app.run(debug=False, host="0.0.0.0", port="8892")
