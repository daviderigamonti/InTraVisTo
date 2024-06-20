from typing import List

import dataclasses
import pickle
import uuid
import time

from dash import callback, Output, Input, State, ctx
from transformers import GenerationConfig
from scipy.special import kl_div  # (ufuncs in scipy.special are written in C) pylint:disable=E0611

import torch
import plotly.graph_objects as go
import pandas as pd

from inject import INJECTS_PARAMETER, InjectInfo, InjectPosition
from sankey import SankeyParameters, generate_complete_sankey, generate_sankey, format_sankey
from utils import EmbeddingTypes, CellWrapper, LayerWrapper
from app import extra_layout

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


def extract_layer_n_secondary(x, emb_type, strategy, decoder):
    return [decoder.decode_secondary_tokens(t, s, e) if type(e) == LayerWrapper else [" "] for e, t, s in zip(x, emb_type, strategy)]


def extract_diff_text(wrappers, emb_type, strategy, decoder):
    return pd.DataFrame(wrappers).diff().apply(lambda x: extract_layer_n_secondary(x, [emb_type]*len(x), [strategy]*len(x), decoder)).sort_index(ascending=False)

# Most likely tokens


def extract_layer_n(x, emb_type, strategy, decoder):
    return [decoder.decode_hidden_state(t, s, e) for e, t, s in zip(x, emb_type, strategy)]


def extract_text(wrappers, emb_type, strategy, decoder):
    return pd.DataFrame(wrappers).apply(lambda x: extract_layer_n(x, [emb_type]*len(x), [strategy]*len(x), decoder))

# Probabilities


def extract_layer_prob(x, decoding_strategy):
    return [e.get_probability(d) for e, d in zip(x, decoding_strategy)]


def extract_probabilities(wrappers, strategy):
    return pd.DataFrame(wrappers).apply(lambda x: extract_layer_prob(x, [strategy]*len(x)))

# Secondary tokens


def extract_secondary_tokens(layers: List[List[LayerWrapper]], emb_type: EmbeddingTypes, strategy: str, decoder):
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

def generate_callbacks(app, cache, model, decoder, model_config, tokenizer, prefix_tokens, device):
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
        secondary_tokens = extract_secondary_tokens(layers, emb_type=emb_type, strategy=strategy, decoder=decoder)

        text = extract_text(layers, emb_type, strategy, decoder).values.tolist()

        p = extract_probabilities(layers, colour).values.tolist()

        fig = go.Figure(data=go.Heatmap(
                        z=p,
                        text=pd.DataFrame(secondary_tokens),
                        xgap=2,
                        ygap=2,
                        x=[i - 0.5 for i in range(0, input_len + output_len)],
                        y=list(range(0, model_config.num_hidden_layers + 1)),
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
            tokens = extract_secondary_tokens(layers=interest_layers, emb_type=emb_type, strategy=strategy, decoder=decoder)
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
        diffs = extract_diff_text(interest_layers, EmbeddingTypes.BLOCK_OUTPUT, strategy, decoder=decoder)
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

        children = extra_layout.generate_tooltip_children_layout(
            layer=clickData["points"][0]["y"],
            token=clickData["points"][0]["x"],
        )

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
