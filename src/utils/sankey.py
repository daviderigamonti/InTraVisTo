from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import heapq
import math

import plotly.graph_objects as go
import numpy as np


LINK_NAME_MAP = {
    "residual_att": "Residual", "residual_ff": "Residual", "residual_norm": "Normalization",
    "att_in": "Attention", "att_out": "Attention", 
    "ff_in": "Residual", "ff_out": "Feed Forward"
}
LINK_KL_MAP = {
    "residual_att": "kl_diff_in-int", "residual_ff": "kl_diff_int-out", "residual_norm": "kl_diff_out-out",
    "att_out": "kl_diff_att-int",
    "ff_in": "kl_diff_int-ff", "ff_out": "kl_diff_ff-out"
}

class AttentionHighlight(str, Enum):
    NONE = "none"
    ALL = "all"
    TOP_K = "top_k"
    TOP_P = "top_p"
    MIN_WEIGHT = "min_weight"

class SizeAdapt(str, Enum):
    FIXED = "fixed"
    TOKEN = "token"
    LAYER = "layer"


@dataclass
class SankeyParameters:
    # DATA
    # Row of starting token (where 0 corresponds to the top row, and n_layers - 1 corresponds to the bottom row)
    row_index: int = 0
    token_index: int = 0  # Position index of starting token (where 0 is first token of the input sequence)
    rowlimit: int = 5  # Limit number of layers to visualize
    multirep: bool = True  # Accomodate for each token having multiple labels
    # COLORS
    node_opacity: float = 0.7  # Opacity of nodes
    link_opacity: float = 0.5  # Opacity of links
    node_color_map: dict[str, tuple] = field(
        default_factory=lambda: {
            "Default": (220, 220, 220),
            "Node": (33, 150, 243),
            "Intermediate": (33, 150, 243),
            "FFNN": (224, 31, 92),
            "Attention": (93, 224, 31),
        }
    )
    link_color_map: dict[str, tuple] = field(
        default_factory=lambda: {
            "Default": (255, 255, 255),
            "residual_norm": (31, 93, 224),
            "residual_att": (31, 93, 224),
            "residual_ff": (31, 93, 224),
            "att_in": (93, 224, 31),
            "att_out": (93, 224, 31),
            "ff_in": (31, 93, 224),
            "ff_out": (224, 31, 92),
        }
    )
    attention_opacity: float = 0.0
    # LAYOUT
    print_indexes: bool = False
    attention_select: AttentionHighlight = AttentionHighlight.TOP_K
    attention_highlight: object = 0
    only_nodes_labels: bool = False
    rescale_factor: int = 3
    # Correction to avoid feeding nodes with a coordinate value of 0, which causes problems with Plotly Sankey Diagrams
    sankey_zero: float = 1e-16
    font_size: float = 12  # Text font size
    size: int = 1800  # Size of square canvas
    margins: dict[str, tuple] = field(
        default_factory=lambda: {
            "top": 20,
            "left": 20,
        }
    )
    size_adapt: SizeAdapt = SizeAdapt.FIXED


def cumulative_sankey_traces(
    dfs, linkinfo,            # Dataframes and link info to access labels and node hidden information
    # Dataframe is indexed by index and row, while el_index references the index for sankey visualization
    row, indexes, el_indexes,
    bases,                    # Base attention value of parents
    labels,                   # Current set of labels for sankey visualization
    # Reference for duplicate nodes as a dictionary indexed with (row, index) and containing a dictionary composed of
    # an id and a base
    elmap,
    rowlimit,                 # Depth limit
    firstcall=True,           # Identify if it's the original function call
):
    new_labels = []
    new_indexes = []
    new_elmap = elmap.copy()

    under = []
    over = []
    val = []
    types = []
    norm_layer = False
    # Calculate current value of node by weighting its attention value for the parent's weight
    for index, el_index, base in zip(indexes, el_indexes, bases):
        res_w = linkinfo["attn_res_percent"][row-1][index].item()
        res_w += 1e-16 if res_w == 0.0 else (-1e-16 if res_w == 1.0 else 0)  # Prevent 0
        attn_w = 1 - res_w
        resattn_w = linkinfo["ffnn_res_percent"][row-1][index].item()
        resattn_w += 1e-16 if resattn_w == 0.0 else (-1e-16 if resattn_w == 1.0 else 0)  # Prevent 0
        ff_w = 1 - resattn_w
        # Create FFNN / Attention / Intermediate nodes
        if index < len(dfs["ffnn"][row]):
            ff_index = len(new_elmap.keys())
            new_labels.append(dfs["ffnn"][row][index])
            new_elmap[(round(row - 1 + 0.8, 2), round(index - 0.5, 2))] = {
                "id": ff_index, "base": base * ff_w, "type": "FFNN"}
            attn_index = len(new_elmap.keys())
            new_labels.append(dfs["attention"][row][index])
            new_elmap[(round(row - 1 + 0.45, 2), round(index - 0.5, 2))] = {
                "id": attn_index, "base": base * attn_w, "type": "Attention"}
            hid_index = len(new_elmap.keys())
            new_labels.append(dfs["intermediate"][row][index])
            new_elmap[(round(row - 1 + 0.65, 2), index)] = {"id": hid_index, "base": base, "type": "Intermediate"}
        else:
            norm_layer = True
        if not norm_layer:
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
            # FFNN State
            over.append(el_index)
            under.append(ff_index)
            val.append(base * ff_w)
            types.append("ff_out")
            over.append(ff_index)
            under.append(hid_index)
            val.append(base * ff_w)
            types.append("ff_in")
            # Attention State
            over.append(hid_index)
            under.append(attn_index)
            val.append(base * attn_w)
            types.append("att_out")
            # Residuals
            over.append(hid_index)
            under.append(new_elmap[(row-1, index)]["id"])
            val.append(base * res_w)
            types.append("residual_att")
            new_elmap[(row-1, index)]["base"] += base * res_w
            over.append(el_index)
            under.append(hid_index)
            val.append(base * resattn_w)
            types.append("residual_ff")
        else:
            # Skip-residual
            new_index = len(new_elmap.keys())
            new_labels.append(dfs["states"][row-1][index])
            new_indexes.append(index)
            new_elmap[(row-1, index)] = {"id": new_index, "base": base, "type": "Node"}
            over.append(el_index)
            under.append(new_index)
            val.append(base)
            types.append("residual_norm")

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
    if old_max is None:
        old_max = max([i for i in l if i is not None])
    if old_min is None:
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

    return [
        el if el is None else \
            range_min + (((invert_k + (invert_a * (el - old_min))) * new_range) / old_range)
        for el in l
    ]

# Given a list and a list of indexes that have been previously sorted, restore the original order of the list
def restore_list_order(l, indexes):
    return [l[indexes.index(i)] for i in range(0, len(indexes))]

# Return a list of RGBA color strings given a list of RGBA colors tuples
def build_rgba_from_tuples(l, opacity=1.0):
    return [f"rgba{tuple(el) + (opacity,)}" if len(el) == 3 else f"rgba{tuple(el)}" for el in l]

def change_color_brightness(rgb_color, brightness):
    delta_color = tuple([int((channel) * brightness) for channel in rgb_color])
    return tuple([sum(channel) for channel in zip(rgb_color, delta_color)])


def format_sankey(un, ov, vl, types, lab, elmap, linkinfo, sankey_parameters: SankeyParameters, hide_nodes=dict()):
    # Handle multiple labels for tokens with multiple representations
    typemap = [next(v["type"] for k, v in elmap.items() if v["id"] == i) for i in range(len(elmap.keys()))]
    nodes_extra = [{
            "text": ', '.join(l),
            "diff": f"Decoded difference from previous layer: {', '.join(linkinfo["diff"][k[0]-1][k[1]])}"
                if t in ["Node"] and k[0] > 0 else "",
            "y": k[0], "x": k[1],
            "type": t,
        } for l, t, k in zip(lab, typemap, elmap.keys())
    ]
    if sankey_parameters.multirep:
        lab = [l[0] for l, t in zip(lab, typemap)]
    else:
        lab = [np.squeeze(l[0]) if t in ["Node"] else np.squeeze(l) for l, t in zip(lab, typemap)]

    # Generate numbered labels
    lab = [f"{k[1]} {lab[el["id"]]}" if sankey_parameters.print_indexes and el["type"]
           in ["Node"] else lab[el["id"]] for k, el in elmap.items()]

    # Remove extra labels not belonging to nodes
    if sankey_parameters.only_nodes_labels:
        lab = [l if t in ["Node"] else "" for l, t in zip(lab, typemap)]

    # Add non-rescaled info to links and nodes extra information
    for k, el in elmap.items():
        nodes_extra[el["id"]] = nodes_extra[el["id"]] | {"v": el["base"]}
    links_extra = [{"v": v, "type": t, "name": LINK_NAME_MAP[t]} for v, t in zip(vl, types)]

    # Create a map containing attention links with largest values for each node
    max_att_list = defaultdict(list)
    if sankey_parameters.attention_select == AttentionHighlight.NONE:
        pass
    if sankey_parameters.attention_select == AttentionHighlight.ALL:
        for el_ov, el_un, typ, value in zip(ov, un, types, vl):
            if typ == "att_in":
                max_att_list[el_ov].append((value, el_un))
    elif sankey_parameters.attention_select == AttentionHighlight.MIN_WEIGHT:
        for el_ov, el_un, typ, value in zip(ov, un, types, vl):
            if typ == "att_in":
                _ = max_att_list[el_ov]
                if value >= sankey_parameters.attention_highlight:
                    max_att_list[el_ov].append((value, el_un))
    elif sankey_parameters.attention_select == AttentionHighlight.TOP_K:
        for el_ov, el_un, typ, value in zip(ov, un, types, vl):
            if typ == "att_in":
                heapq.heappush(max_att_list[el_ov], (value, el_un))
                if len(max_att_list[el_ov]) > sankey_parameters.attention_highlight:
                    heapq.heappop(max_att_list[el_ov])
    max_att_list = {
        el_ov: [tup[1] for tup in max_att_list[el_ov]]
        for el_ov, typ in zip(ov, types) if typ == "att_in"
    }

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
    # Sort reverse-mapped lists to perform transformations on them with more ease,
    # while keeping an index list to reverse the sorting
    revmap_indexes = list(range(0, len(revmap)))
    _, revmap_y_sort, revmap_values_sort, revmap_indexes = zip(
        *sorted(zip(revmap_x, revmap_y, revmap_values, revmap_indexes), key=lambda x: x[0]))

    # Add kl divergence values to residual links between consecutive layers
    kl_values = [
        linkinfo[LINK_KL_MAP[typ]][math.floor(revmap_x[el])][math.ceil(revmap_y[el])]
        if typ not in ["att_in"] else None
        for typ, el in zip(types, un)
    ]

    def format_kl(x):
        i = 0
        udm = ["", "m", "μ", "n", "p"]
        while x < 0.1 and i < len(udm) - 1:
            x *= 1000
            i += 1
        return f"KL: {x:.0f}{udm[i]} nats"

    links_extra = [
        l | {"kl_diff": format_kl(kl) if kl is not None else ""}
        for l, kl in zip(links_extra, kl_values)
    ]

    # Node colors
    node_colors = [sankey_parameters.node_color_map[v["type"]] for v in revmap_values]

    # Link colors
    link_colors = []
    np_kl_values = np.array(kl_values)
    kl_mask = np_kl_values == None # pylint:disable=singleton-comparison
    sort_kl_values = np.argsort(np_kl_values[~kl_mask]).tolist()
    i = -1
    std_kl_values = [
        sort_kl_values.index(i) if not m else None
        for m in kl_mask
        if m or (i := i + 1) or True
    ]
    for typ, el, el_ov, kl in zip(
        types, un, ov, rescale_list(std_kl_values, invert=True, range_min=-0.3, range_max=1.1)
    ):
        color = sankey_parameters.link_color_map[typ].copy()
        # Color residuals according to difference of kl divergence
        if kl is not None:
            color = change_color_brightness(color, kl)
        elif typ in ["att_in"]:
            # Color attention following max attention values
            if el not in max_att_list[el_ov]:
                color = sankey_parameters.link_color_map["Default"].copy() + [sankey_parameters.attention_opacity,]
        link_colors.append(color)

    # Remove colors from hidden links and nodes
    hide_ids = [
        next(iter([
            i for i, (xy, typ) in enumerate(zip(revmap, typemap))
            if tuple(map(math.ceil, xy)) == hide_xy and typ == hide_typ
        ]), -1) for hide_xy, hide_typ in hide_nodes
    ]
    node_colors = [
        color if i not in hide_ids else sankey_parameters.node_color_map["Default"].copy() + [0.0,]
        for i, color in enumerate(node_colors)
    ]
    link_colors = [
        color if i not in hide_ids and j not in hide_ids else sankey_parameters.link_color_map["Default"].copy() + [0.0,]
        for i, j, color in zip(un, ov, link_colors)
    ]

    # Convert colors and add opacities
    node_colors = build_rgba_from_tuples(node_colors, sankey_parameters.node_opacity)
    link_colors = build_rgba_from_tuples(link_colors, sankey_parameters.link_opacity)

    # Generate columns based on maximum node width for each column to fit nodes into
    columns_width = {
        y_index: max((
            v["base"] if y == y_index else 0
            for (y, v) in zip(revmap_y_sort, revmap_values_sort)
        ))
        # Iterate also considering attention/feedforward blocks as columns, hence starting from -0.5 with a step of 0.5
        for y_index in np.arange(-0.5, max(revmap_y) + 1, 0.5)
    }

    # Halve columns that belong to attention/feedforward
    columns_width = {k: c if math.ceil(k) == k else c/2 for k,c in columns_width.items()}

    # Infer optimal column padding
    s = sum(columns_width.values())
    r = 1 - s
    l = len(columns_width)
    if r > 0:
        for _ in range(2*l):
            k = min(columns_width, key=columns_width.get)
            columns_width[k] += r / (2*l)
    else:
        columns_width = {key: w/s for key, w in columns_width.items()}
    columns_ys = {}
    tot_w = 0
    for key, w in sorted(columns_width.items()):
        columns_ys[key] = tot_w
        tot_w += w

    # Adjust coordinates
    revmap_x = rescale_list(revmap_x, range_min=sankey_parameters.sankey_zero, range_max=1, invert=True)
    revmap_y = [
        # Shift attention/ffnn nodes closer to their reference nodes
        columns_ys[y] + columns_width[y] / 2 for y, v in zip(revmap_y, revmap_values)
    ]

    if sankey_parameters.size_adapt == SizeAdapt.LAYER:
        size = (sankey_parameters.size / 2) + \
            (sankey_parameters.size / 10) * (sankey_parameters.row_index - sankey_parameters.rowlimit)
    elif sankey_parameters.size_adapt == SizeAdapt.TOKEN:
        size = (sankey_parameters.size / 2) + (sankey_parameters.size / 22) * l
    else:
        size = sankey_parameters.size

    hovertemplates_nodes = [
        f"{ne['text']}<br>{ne['diff']}<extra>{ne['v']:.1%}</extra>"
        if i not in hide_ids
        else "<extra></extra>"
        for i, ne in enumerate(nodes_extra)
    ]
    hovertemplates_links = [
        f"{le['name']} from {lab[un_el]} to {lab[ov_el]}<br>{le['kl_diff']}<extra>{le['v']:.1%}</extra>"
        if (le["type"] not in ["att_in"] or un_el in max_att_list[ov_el]) and un_el not in hide_ids and ov_el not in hide_ids 
        else "<extra></extra>"
        for le, un_el, ov_el in zip(links_extra, un, ov)
    ]

    fig = go.Figure(
        go.Sankey(
        orientation="v",
        arrangement="fixed",
        valueformat=".5r",
        customdata=nodes_extra, # Replicated customdata to handle node click events
        node={
            "customdata": hovertemplates_nodes,
            "hovertemplate": "%{customdata}",
            "align": "left",
            "label": lab,
            "color": node_colors,
            "x": revmap_x,
            "y": revmap_y,
            "pad": 10000,
            "line": {"color": "rgba(255,255,255,0)", "width": 0},
        },
        link={
            "customdata": hovertemplates_links,
            "hovertemplate": "%{customdata}",
            "source": ov,
            "target": un,
            "value": rescaled_vl,
            "color": link_colors,
            "line": {"color": "rgba(255,255,255,0)"},
        }
    ))
    fig.update_layout(
        font_size=sankey_parameters.font_size, font_family="Verdana", font_color="black",
        width=size, height=size, 
        margin={"t": sankey_parameters.margins["top"], "l": sankey_parameters.margins["left"]},
        modebar_remove=["select", "lasso"]
    )
    return fig
