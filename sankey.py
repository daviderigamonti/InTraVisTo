from dataclasses import dataclass, field
from itertools import cycle

import math

import torch
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


@dataclass
class SankeyParameters:
    # DATA
    # Row of starting token (where 0 corresponds to the top row, and n_layers - 1 corresponds to the bottom row)
    row_index: int = 0
    token_index: int = 0  # Position index of starting token (where 0 is first token of the input sequence)
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
        res_w = linkinfo["attn_res_percent"][row-1][index].item()
        res_w += 0.0000000001 if res_w == 0.0 else (-0.0000000001 if res_w == 1.0 else 0)  # Prevent 0
        attn_w = 1 - res_w
        resattn_w = linkinfo["ffnn_res_percent"][row-1][index].item()
        resattn_w += 0.0000000001 if resattn_w == 0.0 else (-0.0000000001 if resattn_w == 1.0 else 0)  # Prevent 0
        mlp_w = 1 - resattn_w
        # Create MLP / Attention / Intermediate nodes
        mlp_index = len(new_elmap.keys())
        new_labels.append(dfs["ffnn"][row][index])
        new_elmap[(round(row - 1 + 0.8, 2), round(index - 0.5, 2))] = {
            "id": mlp_index, "base": base * mlp_w, "type": "FFNN"}
        attn_index = len(new_elmap.keys())
        new_labels.append(dfs["attention"][row][index])
        new_elmap[(round(row - 1 + 0.45, 2), round(index - 0.5, 2))] = {
            "id": attn_index, "base": base * attn_w, "type": "Attention"}
        hid_index = len(new_elmap.keys())
        new_labels.append(dfs["intermediate"][row][index])
        new_elmap[(round(row - 1 + 0.65, 2), index)] = {"id": hid_index, "base": base, "type": "Intermediate"}
        # Iterate over all elements of the next row
        for i, label in enumerate(dfs["states"][row-1]):
            v = base * attn_w * linkinfo["attentions"][row-1][index][i].item()
            if v > 0:
                over.append(attn_index)
                # If node is already present store its information
                if (row-1, i) in new_elmap:
                    under.append(new_elmap[(row-1, i)]["id"])
                    new_elmap[(row-1, i)]["base"] += v
                # If the node is new create a new entry in the element map with a new sankey index
                else:
                    new_index = len(new_elmap.keys())
                    new_labels.append(label)
                    new_indexes.append(i)
                    under.append(new_index)
                    new_elmap[(row-1, i)] = {"id": new_index, "base": v, "type": "Node"}
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
        under.append(new_elmap[(row-1, index)]["id"])
        val.append(base * res_w)
        types.append("residual")
        new_elmap[(row-1, index)]["base"] += base * res_w
        over.append(el_index)
        under.append(hid_index)
        val.append(base * resattn_w)
        types.append("residual")

    # If depth limit is reached, stop recurring
    if row - 1 > rowlimit:
        # Call itself on all the new nodes
        nex_under, nex_over, nex_val, nex_types, nex_labels, new_elmap = cumulative_sankey_traces(
            dfs, linkinfo,
            row-1, new_indexes, [new_elmap[(row-1, i)]["id"] for i in new_indexes],
            [new_elmap[(row-1, i)]["base"] for i in new_indexes],
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

    last_token = sankey_parameters.token_index

    row_index = sankey_parameters.row_index
    token_indexes = range(last_token - gen_length + 1, last_token + 1)
    token_el_indexes = range(0, gen_length)
    token_labels = [dfs["states"][row_index][token_index] for token_index in token_indexes]
    token_base_val = 1.0 / gen_length
    elmap = {
        (row_index, tidx): {"id": telidx, "base": token_base_val, "type": "Node"}
        for tidx, telidx in zip(token_indexes, token_el_indexes)
    }

    # if not sankey_parameters.show_0:
    #     linkinfo["attentions"] = [[[
    #         torch.tensor([0], device="cpu") if i == 0 or j == 0 else e2 for j, e2 in enumerate(e1)
    #     ] for i, e1 in enumerate(row)
    #     ] for row in linkinfo["attentions"]
    #     ]

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
    token_label = dfs["states"][row_index][token_index]

    # if not sankey_parameters.show_0:
    #     linkinfo["attentions"] = [[[
    #         torch.tensor([0], device="cpu") if i == 0 or j == 0 else e2 for j, e2 in enumerate(e1)
    #     ] for i, e1 in enumerate(row)
    #     ] for row in linkinfo["attentions"]
    #     ]

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
        invert_k = old_range
        invert_a = -1

    return [el if el is None else range_min + (((invert_k + (invert_a * (el - old_min))) * new_range) / old_range) for el in l]

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
    # nodes_extra = [
    #     {"text": l[0], "diff": "Diff from previous layer:" + " ".join(l[1])} if t in ["Node"]
    #     else {"text": l, "diff": ""}
    #     for l, t in zip(lab, typemap)
    # ]
    nodes_extra = [
        {"text": l, "diff": ""} for l, t in zip(lab, typemap)
    ]
    if sankey_parameters.multirep:
        #lab = [l[0][0] if t in ["Node"] else l[0] for l, t in zip(lab, typemap)]
        lab = [l[0] for l, t in zip(lab, typemap)]
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
    #max_kl = linkinfo["kl_diff"].replace([np.inf, -np.inf], np.nan).max(skipna=True).max()
    linkinfo["kl_diff"] = torch.stack(linkinfo["kl_diff"], dim=0)
    max_kl = torch.max(linkinfo["kl_diff"][torch.isfinite(linkinfo["kl_diff"])])
    def checkinf(x): return x if not np.isinf(x) else max_kl
    kl_values = [
        checkinf(
            linkinfo["kl_diff"][math.ceil(revmap_x[el])-1][math.ceil(revmap_y[el])]
        ).item() if typ in ["residual"] and math.ceil(revmap_y[el]) > 0 else None
        for typ, el in zip(types, un)
    ]
    def format_kl(x): return "KL: {:.0f}m nats".format(x) if x >= 10 else "KL: {:.0f}μ nats".format(x * 1000)
    links_extra = [l | {"kl_diff": format_kl(kl * 1000) if kl is not None else ""} for l, kl in zip(links_extra, kl_values)]

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
    col_pad = sankey_parameters.column_pad
    columns_width = [max([v["base"] if y == y_index else 0 for (y, v) in zip(revmap_y_sort, revmap_values_sort)])
                     for y_index in range(0, max(revmap_y) + 1)]
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
    revmap_x = rescale_list(revmap_x, range_min=sankey_parameters.sankey_zero, range_max=1, invert=True)
    revmap_y = [columns_ys[math.ceil(y)] + v["base"] / 2 -
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
