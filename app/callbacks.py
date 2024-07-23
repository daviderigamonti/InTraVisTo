from typing import List, Any

import dataclasses
import inspect
import uuid
import time
import gc
import os

from dash import Output, Input, State, ctx, no_update, ALL
from dash.exceptions import PreventUpdate
from transformers import GenerationConfig

import torch
import plotly.graph_objects as go
import pandas as pd

# TODO: put these values inside classes
from sankey import SankeyParameters, generate_complete_sankey, generate_sankey, format_sankey
from utils import ModelUtils, EmbeddingsType, ProbabilityType, ResidualContribution, CellWrapper, LayerWrapper, Decoder
from app import extra_layout
from app.constants import * # pylint:disable=W0401,W0614
from app.defaults import * # pylint:disable=W0401,W0614

from transformer_wrappers.wrappers import InjectInfo, InjectPosition # pylint:disable=E0401,E0611


def generate_callbacks(app, cache, models, models_lock, device):

    def extract_key_from_processed_layers(decoded_layers: List[List[object]], key: Any):
        return [[
            cell[key] for cell in layer if key in cell
        ] for layer in decoded_layers]

    # TODO: eventually put strategy as enum
    # Note: Every argument should be called as a key-value argument, otherwise it bypasses the "ignore"
    #       argument of cache.memoize
    @cache.memoize(ignore={"layers", "decoder"})
    def decode_layers(
        *args, layers, strategy: str, decoder: Decoder, _session_id: str
    ):
        if args:
            raise TypeError(f"Found positional argument(s) in decode_layers function {args}")
        return decoder.decode(layers, decoding=strategy)

    # Note: Every argument should be called as a key-value argument, otherwise it bypasses the "ignore"
    #       argument of cache.memoize
    @cache.memoize(ignore={"layers", "decoder"})
    def compute_probabilities(
        *args, layers, strategy: str, residual_contribution: ResidualContribution, decoder: Decoder, _session_id: str
    ):
        if args:
            raise TypeError(f"Found positional argument(s) in compute_probabilities function {args}")
        return decoder.compute_probabilities(layers, decoding=strategy, residual_contribution=residual_contribution)

    @cache.memoize()
    def model_generate(prompt, model_id, run_config, session):
        with models_lock:
            model = models[model_id]
        inputs = model.tokenizer(prompt, return_tensors="pt").to(device)
        # Generate
        # output = model(inputs.input_ids, return_dict=True, output_hidden_states=True, output_attentions=True)
        # print(f"Input len: {inputs.input_ids.shape}")
        # print(inputs.char_to_token)
        input_len = len(inputs.input_ids.squeeze().tolist())

        injects = []
        for inject in run_config["injects"]:
            inj_token = model.tokenizer.encode(
                inject["text"],
                add_special_tokens=False,
                return_tensors="pt"
            ).to(device).squeeze()

            # Remove eventual prefix tokens
            mask = ~torch.isin(inj_token, model.prefix_tokens)
            inj_token = inj_token[mask]

            # TODO: multitoken embeddings are averaged by default
            # TODO: injected embeddings are interpolated by default
            encoding = model.decoder.decoding_matrix[DecodingType.LINEAR]()
            # Skip first values of iterator to reach the wanted layer (is_cuda call just to get a boolean value)
            _ = [_ for _ in range(inject["target_layer"] - 1) if next(encoding).is_cuda and False]
            encoding = next(encoding)
            print("Averaged multi-tokens for embedding injection")
            print("Interpolated embeddings for injection")

            # TODO: inject info creation
            inject_translate = {
                EmbeddingsType.BLOCK_OUTPUT : InjectPosition.OUTPUT,
                EmbeddingsType.POST_ATTENTION : InjectPosition.ATTENTION,
                EmbeddingsType.POST_FF : InjectPosition.FFNN,
                EmbeddingsType.POST_ATTENTION_RESIDUAL : InjectPosition.INTERMEDIATE
            }
            injects.append(
                InjectInfo(
                    layer=inject["target_layer"],
                    token=inject["target_token"],
                    position=inject_translate[inject["location"]],
                    embedding=torch.stack([encoding[tok] for tok in inj_token], dim=0).mean(dim=0)
                )
            )

        gen_config = GenerationConfig(
            pad_token_id=model.model_config.eos_token_id,
            max_new_tokens=run_config["max_new_tok"],
            return_attention_output=True,
            return_feed_forward_output=True,
            return_intermediate_hidden_states=True,
        )
        wrapper_gen_config = {
            "generation_config": gen_config,
            "return_inner_states": True,
            model.model.INJECTS_PARAMETER: injects
        }

        generation_result = model.model.generate(inputs.input_ids, **wrapper_gen_config)
        # TODO: test wrong last token
        #gen_config = GenerationConfig(
        #    output_attentions=True,
        #    output_hidden_states=True,
        #    #output_scores=True,
        #    return_dict_in_generate=True,
        #    pad_token_id=model.model_config.eos_token_id,
        #    max_new_tokens=run_config["max_new_tok"],
        #)
        #generation_result = model.model.generate(inputs.input_ids, generation_config=gen_config)

        def standardize_wrapped_tensors(t):
            s = torch.stack(t, dim=0).squeeze().detach()
            return s

        output_len = generation_result["sequence_length"] - input_len
        generation_output = {
            "sequences": generation_result["output_ids"].squeeze(),
            "attentions": standardize_wrapped_tensors(generation_result["attention_weights"]).mean(dim=1)[:,:-1,:-1]
        }

        # Create a list of LayerWrapper
        layers = []

        hidden_states = standardize_wrapped_tensors(generation_result["hidden_states"])
        attention_outputs = standardize_wrapped_tensors(generation_result["attention_outputs"])
        feed_forward_outputs = standardize_wrapped_tensors(generation_result["feed_forward_outputs"])
        intermediate_hidden_states = standardize_wrapped_tensors(generation_result["intermediate_hidden_states"])

        # 1- Prepare matrix of input tokens hidden_state:  N_TOKENS x N_LAYER
        # input_hidden_states = generation_result["hidden_states"][0]

        per_token_layers = LayerWrapper(0, session_id=session)
        for tok_hs in hidden_states[0][:-1]:
            layer = CellWrapper()
            layer.add_embedding(tok_hs, EmbeddingsType.BLOCK_OUTPUT)
            per_token_layers.cells.append(layer)
        layers.append(per_token_layers)

        # TODO: fix variable names
        # Iterate over layers
        for layer_id, (layer_hs, layer_att, layer_ffnn, layer_inter) in enumerate(zip(hidden_states[1:], attention_outputs, feed_forward_outputs, intermediate_hidden_states)):
            # Iterate over tokens
            per_token_layers = LayerWrapper(layer_id + 1, session_id=session)

            for tok_hs, tok_att, tok_ffnn, tok_inter in zip(layer_hs[:-1], layer_att[:-1], layer_ffnn[:-1], layer_inter[:-1]):
                layer = CellWrapper()
                layer.add_embedding(tok_hs, EmbeddingsType.BLOCK_OUTPUT)
                layer.add_embedding(tok_att, EmbeddingsType.POST_ATTENTION)
                layer.add_embedding(tok_ffnn, EmbeddingsType.POST_FF)
                layer.add_embedding(tok_inter, EmbeddingsType.POST_ATTENTION_RESIDUAL)
                per_token_layers.cells.append(layer)
            layers.append(per_token_layers)

        for layer_hs, layer in zip(hidden_states, layers[1:]):
            for tok_hs, layer_token in zip(layer_hs, layer):
                layer_token.add_embedding(tok_hs, EmbeddingsType.BLOCK_INPUT)

        return generation_output, layers, input_len, output_len, session

    
    @cache.memoize()
    def generate_sankey_info(text, model_id, run_config, session_id, strategy, residual_contribution):
        with models_lock:
            model = models[model_id]
        generated_output, layers, input_len, output_len, session_id = model_generate(text, model_id, run_config, session_id)

        text = decode_layers(layers=layers, strategy=strategy, decoder=model.decoder, _session_id=session_id)
        p = compute_probabilities(
            layers=layers,
            strategy=strategy,
            residual_contribution=residual_contribution,
            decoder=model.decoder,
            _session_id=session_id,
        )

        dfs = {
            "states": extract_key_from_processed_layers(text, EmbeddingsType.BLOCK_OUTPUT),
            "intermediate": extract_key_from_processed_layers(text, EmbeddingsType.POST_ATTENTION_RESIDUAL),
            "attention": extract_key_from_processed_layers(text, EmbeddingsType.POST_ATTENTION),
            "ffnn": extract_key_from_processed_layers(text, EmbeddingsType.POST_FF),
        }

        # Add labels for differences between consecutive layers
        diffs = [layers[i].get_diff(layers[i-1]) for i in range(1, len(layers))]
        # TODO: Standardize differentiating value for session id
        token_diffs = decode_layers(layers=diffs, strategy=strategy, decoder=model.decoder, _session_id=session_id + "TOKEN_DIFFS")
        token_diffs = extract_key_from_processed_layers(token_diffs, EmbeddingsType.BLOCK_OUTPUT)

        attn_res_percent = extract_key_from_processed_layers(p, ProbabilityType.ATT_RES_PERCENT)
        ffnn_res_percent = extract_key_from_processed_layers(p, ProbabilityType.FFNN_RES_PERCENT)

        # attentions = compute_batch_complete_padded_attentions(generated_output, range(0, model_config.num_attention_heads))[-1]
        attentions = generated_output["attentions"]

        kl_diffs = [
            torch.stack(layers[i].get_kldiff(layers[i-1], EmbeddingsType.BLOCK_OUTPUT), dim=0)
            for i in range(1, len(layers))
        ]

        # TODO: choose how to pass values (as torch tensors or python lists)
        #       right now: attentions, kl_diffs -> torch tensor and ffnn_res_percent, attn_res_percent -> python list
        linkinfo = {
            "attentions": attentions, "attn_res_percent": attn_res_percent,
            "ffnn_res_percent": ffnn_res_percent, "kl_diff": kl_diffs, "diff": token_diffs
        }

        return dfs, linkinfo, input_len, output_len


    @app.callback(
        Output("initial_callbacks", "data"),
        Input("initial_callbacks", "empty"),
        State("initial_callbacks", "data")
    )
    def initial_call(_, initial_callbacks):
        return initial_callbacks

    def process_initial_call(initial_callbacks):
        if initial_callbacks[0] == inspect.stack()[1].function:
            return initial_callbacks[1:]
        raise PreventUpdate

    # TODO: take a look at matching callbacks ids
    @app.callback(
        [
            Output('run_config', 'data'),
            Output("injection_card_id", "data"),
        ],
        [
            Input('max_new_tokens', 'value'),
            Input({"type": "add_inj_button", "index": ALL}, "n_clicks"),
            Input({"type": "inject_close_button", "index": ALL}, "n_clicks"),
        ],
        [
            State("injection_card_id", "data"),
            State({"type": "custom_emb", "index": ALL}, "value"),
            State({"type": "custom_emb_location", "index": ALL}, "value"),
            State("run_config", "data"),
            State("vis_config", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_run_config(max_new_tok, inj_button, inj_close_button, card_id, custom_emb, custom_emb_location, run_config, vis_config):
        if not ctx.triggered_id:
            raise PreventUpdate
        run_config["max_new_tok"] = max_new_tok
        
        if "type" in ctx.triggered_id:
            if ctx.triggered_id["type"] == "inject_close_button" and not all(v is None for v in inj_close_button):
                close_button_id = ctx.triggered_id["index"]
                run_config["injects"] = [inj for inj in run_config["injects"] if inj["id"] != close_button_id]
            if ctx.triggered_id["type"] == "add_inj_button" and not all(v is None for v in inj_button) and custom_emb and custom_emb_location:
                run_config["injects"] = run_config["injects"] + [{
                    "id": card_id,
                    "text": custom_emb[0],
                    "location": custom_emb_location[0],
                    "target_layer": vis_config["y"] - 1 if "y" in vis_config and vis_config["y"] is not None else None,
                    "target_token": vis_config["x"] if "x" in vis_config else None,
                }]
                card_id += 1
        return run_config, card_id

    @app.callback(
        Output('vis_config', 'data'),
        [
            Input("generation_notify", "data"),
            Input("main_graph", "clickData"),
            Input("choose_decoding", "value"),
            Input("choose_res_type", "value"),
        ],
        [
            State('hide_col', 'value'),
            State('vis_config', 'data'),
        ],
        prevent_initial_call=True,
    )
    def update_vis_config(gen_notify, click_data, strategy, res_contrib, hide_col, vis_config):
        # If table visualization is hiding the first column, then offset all x-axis click data by 1
        col_0_offset = 1 if len(hide_col) > 0 else 0
        vis_config |= {"x": click_data["points"][0]["x"] + col_0_offset} if click_data else {"x": None}
        vis_config |= {"y": click_data["points"][0]["y"]} if click_data else {"y": None}
        vis_config |= {"strategy": strategy}
        vis_config |= {"res_contrib": res_contrib}
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
        token = vis_config["x"] if "x" in vis_config and vis_config["x"] is not None else 0
        layer = vis_config["y"] if "y" in vis_config and vis_config["y"] is not None else 0
        row_limit = row_limit if layer - row_limit - 1 >= 0 or layer == 0 else layer
        hide_0 = len(hide_0) > 0
        hide_labels = len(hide_labels) > 0
        sankey_vis_config |= {
            "sankey_parameters": dataclasses.asdict(SankeyParameters(
                row_index=layer,
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
            Input("choose_embedding", "value"),
            Input("choose_colour", "value"),
        ],
        [
            State('table_vis_config', 'data'),
        ],
        prevent_initial_call=True,
    )
    def update_table_vis_config(hide_col, font_size, emb_type, colour, vis_config):
        vis_config |= {"hide_col": len(hide_col) > 0}
        vis_config |= {"font_size": font_size}
        vis_config |= {"emb_type": emb_type}
        vis_config |= {"colour": colour}
        return vis_config

    @app.callback(
        [
            Output("initial_callbacks", "data", allow_duplicate=True),
            Output('session_id', 'data'),
            Output('current_run_config', 'data'),
            Output("generation_notify", "data"),
            Output("generate_button_load", "notify"),
        ],
        [
            Input("initial_callbacks", "data"),
            Input("generate_button", "n_clicks"),
        ],
        [
            State("text", "value"),
            State("model_id", "data"),
            State("run_config", "data"),
            State("vis_config", "data"),
        ],
        prevent_initial_call=True,
    )
    def call_model_generate(initial_callbacks, button, text, model_id, run_config, vis_config):
        initial_callbacks = process_initial_call(initial_callbacks) if ctx.triggered_id and ctx.triggered_id == "initial_callbacks" else no_update
        with models_lock:
            model = models[model_id]
        session_id = str(uuid.uuid4())
        _, layers, _, _, _ = model_generate(text, model_id, run_config, session_id)
        # Caching values
        _ = decode_layers(layers=layers, strategy=vis_config["strategy"], decoder=model.decoder, _session_id=session_id)
        _ = compute_probabilities(
            layers=layers,
            strategy=vis_config["strategy"],
            residual_contribution=vis_config["res_contrib"],
            decoder=model.decoder,
            _session_id=session_id,
        )
        # TODO: maybe add? # torch.cuda.empty_cache()
        return initial_callbacks, session_id, run_config, True, True


    @app.callback(
        [
            Output('main_graph', 'figure'),
            Output('output_text', 'value'),
        ],
        [
            Input('generation_notify', 'data'),
            Input("table_vis_config", "data"),
            Input("vis_config", "data")
        ],
        [
            State("text", "value"),
            State("model_id", "data"),
            State("current_run_config", "data"),
            State("session_id", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_graph(
        notify, tab_vis_config, vis_config, text, model_id, run_config, session_id
    ):
        with models_lock:
            model = models[model_id]
        # Retrieve model outputs
        generated_output, layers, input_len, output_len, session_id = model_generate(text, model_id, run_config, session_id)

        # Compute secondary tokens
        text = decode_layers(layers=layers, strategy=vis_config["strategy"], decoder=model.decoder, _session_id=session_id)
        text = extract_key_from_processed_layers(text, tab_vis_config["emb_type"])

        # Compute probabilities
        p = compute_probabilities(
            layers=layers,
            strategy=vis_config["strategy"],
            residual_contribution=vis_config["res_contrib"],
            decoder=model.decoder,
            _session_id=session_id,
        )
        p = extract_key_from_processed_layers(p, tab_vis_config["colour"])
        p = extract_key_from_processed_layers(p, tab_vis_config["emb_type"]) if tab_vis_config["colour"] in [ProbabilityType.ARGMAX, ProbabilityType.ENTROPY] else p

        # Remove first column from visualization
        if tab_vis_config["hide_col"]:
            text = [layer[1:] for layer in text]
            p = [layer[1:] for layer in p]

        fig = go.Figure(
            data=go.Heatmap(
                z=p,
                text=pd.DataFrame(text),
                xgap=2,
                ygap=2,
                x=[i - 0.5 for i in range(0, input_len + output_len)],
                y=list(range(0, model.model_config.num_hidden_layers + 1)),
                hovertemplate='<i>Probability</i>: %{z:.2f}%' +
                '<br><b>Layer</b>: %{y}' +
                '<br><b>Token position</b>: %{x}' +
                '<br><b>Secondary representations</b>: %{text}' +
                '<extra></extra>',
                texttemplate="%{text[0]}",
                textfont={"size": tab_vis_config["font_size"]},
                colorscale="blues"
            )
        )
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
        if vis_config["x"] is not None and vis_config["y"] is not None:
            offset = 0 if tab_vis_config["hide_col"] else 1
            fig.add_shape(
                x0=vis_config["x"] + offset - 0.5, x1=vis_config["x"] + offset - 1.5,
                y0=vis_config["y"] - 0.5, y1=vis_config["y"] + 0.5,
                line_width=2, line_color="red"
            )
        # fig.add_hline(y=32, line_width=30, line_color='white')
        return fig, model.tokenizer.decode(generated_output["sequences"].squeeze()[input_len:])

    @app.callback(
        Output('sankey_graph', 'figure'),
        [
            Input("vis_config", "data"),
            Input('sankey_vis_config', 'data'),
        ],
        [
            State("text", "value"),
            State("model_id", "data"),
            State("current_run_config", "data"),
            State("session_id", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_sankey(vis_config, sankey_vis_config, text, model_id, run_config, session_id):
        with models_lock:
            model = models[model_id]
        if (
            "x" not in vis_config or "y" not in vis_config or
            vis_config["x"] is None or vis_config["y"] is None or
            vis_config["x"] <= 0 or vis_config["y"] <= 0
        ):
            x, y = None, None
        else:
            x, y = vis_config["x"], vis_config["y"]

        sankey_param = SankeyParameters(**sankey_vis_config["sankey_parameters"])
        dfs, linkinfo, input_len, output_len = generate_sankey_info(
            text, model_id, run_config, session_id, vis_config["strategy"], vis_config["res_contrib"]
        )

        if not sankey_param.show_0:
            dfs = {key: [layer[1:] for layer in df] for key, df in dfs.items()}
            linkinfo = {key: [layer[1:] for layer in link] for key, link in linkinfo.items()}
            linkinfo["attentions"] = [[layer2[1:] for layer2 in layer1] for layer1 in linkinfo["attentions"]]

        token_offset = 1 if not sankey_param.show_0 else 0
        if x is None and y is None:
            sankey_param.row_index = model.model_config.num_hidden_layers
            sankey_param.token_index = input_len + output_len - token_offset - 2
            sankey_param.rowlimit = sankey_param.row_index - sankey_param.rowlimit
            sankey_info = generate_complete_sankey(dfs, linkinfo, sankey_param, output_len)
        else:
            sankey_param.token_index -= token_offset
            sankey_param.rowlimit = sankey_param.row_index - sankey_param.rowlimit
            sankey_info = generate_sankey(dfs, linkinfo, sankey_param)
        fig = format_sankey(*sankey_info, linkinfo, sankey_param)
        return fig

    @app.callback(
        [
            Output("initial_callbacks", "data", allow_duplicate=True),
            Output("model_id", "data"),
        ],
        [
            Input("initial_callbacks", "data"),
            Input("model_select", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_output(initial_callbacks, model_id):
        nonlocal models
        initial_callbacks = process_initial_call(initial_callbacks) if ctx.triggered_id and ctx.triggered_id == "initial_callbacks" else no_update
        with models_lock:
            if model_id not in models:
                models |= {model_id: ModelUtils(model_id, device, quant=True, hf_token=os.environ["HF_TOKEN"])}
        return initial_callbacks, model_id

    @app.callback(
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
    def display_embedding_tooltip(gen_notify, click_data, gtt, autohide):
        if click_data is None or ctx.triggered_id == "generation_notify":
            return False, "", []

        children = [extra_layout.generate_tooltip_children_layout(
            layer=click_data["points"][0]["y"],
            token=click_data["points"][0]["x"],
        )]

        return True, "hover focus", children

    
    @app.callback(
            Output("inject_container", "children", allow_duplicate=True),
        [
            Input("add_inj_button", "n_clicks")
        ],
        [
            State("vis_config", "data"),
            State("inject_container", "children"),
            State("custom_emb", "value"),
            State("custom_emb_location", "value"),
        ],
        prevent_initial_call=True,
    )
    def add_injection(button, vis_config, inject_container, custom_emb, custom_emb_location):
        if button is None:
            raise PreventUpdate
        # TODO: where are target layer/token coming from?
        target_layer = vis_config["y"] - 1 if "y" in vis_config and vis_config["y"] is not None else None
        target_token = vis_config["x"] if "x" in vis_config else None
        return inject_container + [extra_layout.generate_inject_card(
            button, custom_emb, custom_emb_location, target_layer, target_token
        )]

    @app.callback(
        Output("inject_container", "children"),
        [
            Input("run_config", "data"),
        ],
        [
            State("inject_container", "children"),
        ],
        prevent_initial_call=True,
    )
    def manage_inject_cards(run_config, inject_container):
        if len(run_config["injects"]) == len(inject_container):
            raise PreventUpdate

        # TODO: better to re-create each card every time or look for removed/new cards and remove/add them?
        return [
            extra_layout.generate_inject_card(
                card_id=inj["id"],
                text=inj["text"],
                position=inj["location"],
                token=inj["target_token"],
                layer=inj["target_layer"]
            )
            for inj in run_config["injects"]
        ]

    @app.callback(
        Input("model_heartbeat", "n_intervals"),
        State("model_id", "data"),
        prevent_initial_call=True,
    )
    def update_active_models(_, model_id):
        with models_lock:
            # Check for models with dead sessions and remove them
            cur = time.time()
            # TODO: Consider if it's necessary to separate the check-dead callback from the heartbeat callback
            # Delete models with expired timestamp while avoid deleting model of current session
            dead = [mid for mid, model in models.items() if cur - model.heartbeat_stamp > HEARTBEAT_TIMEOUT and mid != model_id]
            if dead:
                for mid in dead:
                    del models[mid]
                gc.collect()
                # TODO: add check for device
                torch.cuda.empty_cache()
            # Revive current model
            cur = time.time()
            models[model_id].heartbeat_stamp = cur
