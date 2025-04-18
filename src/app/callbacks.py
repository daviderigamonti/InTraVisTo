from typing import List, Any

import dataclasses
import inspect
import uuid
import time
import copy
import math
import gc
import os

from dash import Output, Input, State, ctx, no_update, ALL, clientside_callback, ClientsideFunction
from dash.exceptions import PreventUpdate
from transformers import GenerationConfig
from torch.cuda import OutOfMemoryError

import torch
import plotly.graph_objects as go

from transformer_wrappers.wrappers import InjectInfo, InjectionStrategy, AblationInfo

from utils.sankey import SankeyParameters, generate_complete_sankey, generate_sankey, format_sankey
from utils.utils import (
    EmbeddingsType, ProbabilityType, ResidualContribution, CellWrapper, LayerWrapper, Decoder,
    clean_text
)
from utils.models import MODEL_COMPATIBILITY_MAP, ModelUtils, ModelCompatibilityInfo, LayerNormalizationWrapper
from app import extra_layout
from app.constants import *
from app.defaults import *


def generate_callbacks(app, cache, models, models_lock, model_loading_lock):

    def extract_key_from_processed_layers(decoded_layers: List[List[object]], key: Any):
        return [[
            cell[key] for cell in layer if key in cell
        ] for layer in decoded_layers]

    # Note: Every argument should be called as a key-value argument, otherwise it bypasses the "ignore"
    #       argument of cache.memoize
    @cache.memoize(ignore={"layers", "decoder", "norm"})
    def decode_layers(
        *args, layers, strategy: str, norm, norm_id: str, secondary_decoding: str, # pylint:disable=unused-argument
        decoder: Decoder, _session_id: str
    ):
        if args:
            raise TypeError(f"Found positional argument(s) in decode_layers function {args}")
        return decoder.decode(layers, decoding=strategy, norm=norm, secondary_decoding=secondary_decoding)

    # Note: Every argument should be called as a key-value argument, otherwise it bypasses the "ignore"
    #       argument of cache.memoize
    @cache.memoize(ignore={"layers", "decoder", "norm"})
    def compute_probabilities(
        *args, layers, strategy: str, residual_contribution: ResidualContribution,
        norm, norm_id: str, decoder: Decoder, _session_id: str  # pylint:disable=unused-argument
    ):
        if args:
            raise TypeError(f"Found positional argument(s) in compute_probabilities function {args}")
        return decoder.compute_probabilities(
            layers, decoding=strategy, residual_contribution=residual_contribution, norm=norm
        )

    @cache.memoize()
    def model_generate(prompt, model_id, run_config, session):
        with models_lock:
            if model_id not in models:
                raise PreventUpdate
            model = models[model_id]
        inputs = model.tokenizer(prompt, return_tensors="pt").to(model.info.device)

        input_len = inputs.input_ids.squeeze()
        # 1-dimensional tensors get squeezed into 0-dimensional ones
        input_len = 1 if input_len.dim() == 0 else len(input_len)

        injects = []
        for inject in run_config["injects"]:
            inj_token = model.tokenizer.encode(
                inject["text"],
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.info.device).squeeze()

            # Remove eventual prefix tokens
            mask = ~torch.isin(inj_token, model.prefix_tokens)
            inj_token = inj_token[mask]

            if inj_token.size()[0] > 0:
                encoding = model.decoder.decoding_matrix[inject["decoding"]]()
                # Skip first values of iterator to reach the wanted layer (is_cuda call just to get a boolean value)
                _ = [_ for _ in range(inject["target_layer"] - 1) if next(encoding).is_cuda and False]
                encoding = next(encoding)
                # Obtain embedding
                emb = torch.stack([encoding[tok] for tok in inj_token], dim=0).mean(dim=0)
                # TODO: multitoken embeddings are averaged by default
                print("Averaged multi-tokens for embedding injection")
                # Normalize embedding
                norm = get_normalization(model_id, inject["norm"])
                
                injects.append(
                    InjectInfo(
                        layer=inject["target_layer"],
                        token=inject["target_token"],
                        position=INJECT_TRANSLATE[inject["location"]],
                        embedding=emb,
                        strategy=inject["type"],
                        decoding_matrix=encoding,
                        decoding_norm=norm
                    )
                )
        ablations = [
            AblationInfo(
                layers=(ablation["target_layer"],ablation["target_layer"]),
                token=ablation["target_token"],
                position=ABLATION_TRANSLATE[ablation["location"]],
            )
            for ablation in run_config["ablations"]
        ]

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
            model.model.INJECTS_PARAMETER: injects,
            model.model.ABLATIONS_PARAMETER: ablations,
        }

        generation_result = model.model.generate(inputs.input_ids, **wrapper_gen_config)

        output_tokens = model.tokenizer.convert_ids_to_tokens(generation_result["output_ids"].squeeze())

        def standardize_wrapped_tensors(t):
            s = torch.stack(t, dim=0).squeeze().detach()
            return s

        generation_output = {
            "sequences": generation_result["output_ids"].squeeze(),
            "attentions": standardize_wrapped_tensors(generation_result["attention_weights"]).mean(dim=1)[:,:-1,:-1]
        }

        # Create a list of LayerWrapper
        layers = []

        hidden_states = standardize_wrapped_tensors(generation_result["hidden_states"])
        attention_outputs = standardize_wrapped_tensors(generation_result["attention_outputs"])[:,:-1,:]
        feed_forward_outputs = standardize_wrapped_tensors(generation_result["feed_forward_outputs"])[:,:-1,:]
        intermediate_hidden_states = standardize_wrapped_tensors(
            generation_result["intermediate_hidden_states"]
        )[:,:-1,:]

        # Append normalized output states to hidden states tensor
        hidden_states = torch.cat(
            (hidden_states, generation_result["output_hidden_state"].detach()), dim=0
        )[:,:-1,:]

        # Handle embedding layer tokens
        emb_layer = LayerWrapper(0, session_id=session)
        for i, tok_hs in enumerate(hidden_states[0]):
            cell = CellWrapper(layer_number=0, token_number=i)
            cell.add_embedding(tok_hs, EmbeddingsType.BLOCK_OUTPUT)
            emb_layer.cells.append(cell)
        layers.append(emb_layer)

        # Iterate over layers
        for layer_id, (layer_hs, layer_att, layer_ffnn, layer_inter) in enumerate(
            zip(hidden_states[1:], attention_outputs, feed_forward_outputs, intermediate_hidden_states)
        ):
            # Iterate over tokens
            layer = LayerWrapper(layer_id + 1, session_id=session)

            for token_id, (tok_hs, tok_att, tok_ffnn, tok_inter) in enumerate(
                zip(layer_hs, layer_att, layer_ffnn, layer_inter)
            ):
                cell = CellWrapper(layer_number=layer_id + 1, token_number=token_id)
                cell.add_embedding(tok_hs, EmbeddingsType.BLOCK_OUTPUT)
                cell.add_embedding(tok_att, EmbeddingsType.POST_ATTENTION)
                cell.add_embedding(tok_ffnn, EmbeddingsType.POST_FF)
                cell.add_embedding(tok_inter, EmbeddingsType.POST_ATTENTION_RESIDUAL)
                layer.cells.append(cell)
            layers.append(layer)

        for layer_hs, layer in zip(hidden_states, layers[1:]):
            for tok_hs, layer_token in zip(layer_hs, layer):
                layer_token.add_embedding(tok_hs, EmbeddingsType.BLOCK_INPUT)

        # Handle normalized layer tokens
        last = len(layers)
        norm_layer = LayerWrapper(last, session_id=session)
        for i, tok_hs in enumerate(hidden_states[last]):
            cell = CellWrapper(layer_number=last, token_number=i)
            cell.add_embedding(tok_hs, EmbeddingsType.BLOCK_OUTPUT)
            norm_layer.cells.append(cell)
        layers.append(norm_layer)

        return generation_output, \
            layers, \
            [clean_text(t) for t in output_tokens[:input_len]], \
            [clean_text(t) for t in output_tokens[input_len:]], \
            session \

    @cache.memoize()
    def generate_sankey_info(
        text, model_id, run_config, session_id, strategy, residual_contribution, norm_id, secondary_decoding
    ):
        with models_lock:
            if model_id not in models:
                raise PreventUpdate
            model = models[model_id]
        generated_output, layers, input_tok, output_tok, session_id = model_generate(
            text, model_id, run_config, session_id
        )

        norm = get_normalization(model_id, norm_id)

        text = decode_layers(
            layers=layers, strategy=strategy, norm=norm, norm_id=norm_id,
            secondary_decoding=secondary_decoding, decoder=model.decoder, _session_id=session_id
        )
        p = compute_probabilities(
            layers=layers, strategy=strategy, residual_contribution=residual_contribution,
            norm=norm, norm_id=norm_id, decoder=model.decoder, _session_id=session_id,
        )

        dfs = {
            "states": extract_key_from_processed_layers(text, EmbeddingsType.BLOCK_OUTPUT),
            "intermediate": extract_key_from_processed_layers(text, EmbeddingsType.POST_ATTENTION_RESIDUAL),
            "attention": extract_key_from_processed_layers(text, EmbeddingsType.POST_ATTENTION),
            "ffnn": extract_key_from_processed_layers(text, EmbeddingsType.POST_FF),
        }

        linkinfo = {}

        # Add labels for differences between consecutive layers
        diffs = [layers[i].get_diff(layers[i-1]) for i in range(1, len(layers))]
        token_diffs = decode_layers(
            layers=diffs, strategy=strategy, norm=norm, norm_id=norm_id, secondary_decoding=secondary_decoding,
            decoder=model.decoder, _session_id=session_id + DASH_SESSION_DIFF_GEN
        )
        linkinfo["diff"] = extract_key_from_processed_layers(token_diffs, EmbeddingsType.BLOCK_OUTPUT)

        linkinfo["attn_res_percent"] = extract_key_from_processed_layers(p, ProbabilityType.ATT_RES_PERCENT)[1:]
        linkinfo["ffnn_res_percent"] = extract_key_from_processed_layers(p, ProbabilityType.FFNN_RES_PERCENT)[1:]

        for abl in run_config["ablations"]:
            if abl["location"] == EmbeddingsType.POST_ATTENTION:
                linkinfo["attn_res_percent"][abl["target_layer"]][abl["target_token"]] = torch.tensor(1.0)
            elif abl["location"] == EmbeddingsType.POST_FF:
                linkinfo["ffnn_res_percent"][abl["target_layer"]][abl["target_token"]] = torch.tensor(1.0)

        linkinfo["attentions"] = generated_output["attentions"]

        linkinfo |= {
            "kl_diff_in-int": [
                torch.stack(layers[i-1].get_kldiff(
                    layers[i],
                    EmbeddingsType.BLOCK_OUTPUT, EmbeddingsType.POST_ATTENTION_RESIDUAL,
                    norm,
                ), dim=0)
                for i in range(1, len(layers)-1)
            ],
            "kl_diff_att-int": [
                torch.stack(layer.get_kldiff(
                    layer,
                    EmbeddingsType.POST_ATTENTION, EmbeddingsType.POST_ATTENTION_RESIDUAL,
                    norm,
                ), dim=0)
                for layer in layers[1:-1]
            ],
            "kl_diff_int-out": [
                torch.stack(layer.get_kldiff(
                    layer,
                    EmbeddingsType.POST_ATTENTION_RESIDUAL, EmbeddingsType.BLOCK_OUTPUT,
                    norm,
                ), dim=0)
                for layer in layers[1:-1]
            ],
            "kl_diff_int-ff": [
                torch.stack(layer.get_kldiff(
                    layer,
                    EmbeddingsType.POST_ATTENTION_RESIDUAL, EmbeddingsType.POST_FF,
                    norm,
                ), dim=0)
                for layer in layers[1:-1]
            ],
            "kl_diff_ff-out": [
                torch.stack(layer.get_kldiff(
                    layer,
                    EmbeddingsType.POST_FF, EmbeddingsType.BLOCK_OUTPUT,
                    norm,
                ), dim=0)
                for layer in layers[1:-1]
            ],
            "kl_diff_out-out": [None] * (len(layers) - 2) + [
                torch.stack(layers[-2].get_kldiff(
                    layers[-1],
                    EmbeddingsType.BLOCK_OUTPUT, EmbeddingsType.BLOCK_OUTPUT,
                    norm,
                ), dim=0)
            ],
        }
        
        # TODO: choose how to pass values (as torch tensors or python lists)
        #       right now: attentions, kl_diffs -> torch tensor and ffnn_res_percent, attn_res_percent -> python list
        return dfs, linkinfo, len(input_tok), len(output_tok)

    # TODO: should be memoized but some normalization modules don't like it
    def get_normalization(
        model_id: str, norm: NormalizationStep
    ):
        with models_lock:
            if model_id not in models:
                raise PreventUpdate
            model = models[model_id]
        compat = MODEL_COMPATIBILITY_MAP[model_id]
        m = model.model
        for step in compat[ModelCompatibilityInfo.NORM_PATH]:
            m = getattr(m, step)
        return LayerNormalizationWrapper(
            norm=copy.deepcopy(m), rescale_info=compat[ModelCompatibilityInfo.NORM_DATA], status=norm
        )


    # Server callbacks

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
            Output("run_config", "data", allow_duplicate=True),
            Output("mod_card_id", "data"),
            Output("start_generation_notify", "data"),
        ],
        [
            Input("max_new_tokens", "value"),
            Input({"type": "add_abl_button", "index": ALL}, "n_clicks"),
            Input({"type": "add_inj_button", "index": ALL}, "n_clicks"),
            Input({"type": "mod_close_button", "index": ALL}, "n_clicks"),
        ],
        [
            State("mod_card_id", "data"),
            State({"type": "custom_emb", "index": ALL}, "value"),
            State({"type": "custom_inj_type", "index": ALL}, "value"),
            State({"type": "custom_emb_location", "index": ALL}, "value"),
            State({"type": "custom_decoding", "index": ALL}, "value"),
            State({"type": "custom_norm", "index": ALL}, "value"),
            State("run_config", "data"),
            State("vis_config", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_run_config(
        max_new_tok,
        abl_button, inj_button, mod_close_button,
        card_id,
        custom_emb, custom_inj_type, custom_emb_location, custom_decoding, custom_norm,
        run_config, vis_config
    ):
        if not ctx.triggered_id:
            raise PreventUpdate
        run_config["max_new_tok"] = max_new_tok

        generate = no_update

        if "type" in ctx.triggered_id:
            if ctx.triggered_id["type"] == "mod_close_button" and not all(v is None for v in mod_close_button):
                close_button_id = ctx.triggered_id["index"]
                run_config["injects"] = [inj for inj in run_config["injects"] if inj["id"] != close_button_id]
                run_config["ablations"] = [abl for abl in run_config["ablations"] if abl["id"] != close_button_id]
            if ctx.triggered_id["type"] == "add_inj_button" and \
                    not all(v is None for v in inj_button) and \
                    custom_emb and custom_inj_type and custom_emb_location and custom_decoding and custom_norm:
                run_config["injects"] = run_config["injects"] + [{
                    "id": card_id,
                    "text": custom_emb[0],
                    "type": custom_inj_type[0],
                    "location": custom_emb_location[0],
                    "decoding": custom_decoding[0],
                    "norm": custom_norm[0],
                    "target_layer": vis_config["y"] - 1 if "y" in vis_config and vis_config["y"] is not None else None,
                    "target_token": vis_config["x"] if "x" in vis_config else None,
                }]
                card_id += 1
                # Perform generation right after injection
                generate = True
            if ctx.triggered_id["type"] == "add_abl_button" and \
                    not all(v is None for v in abl_button) and \
                    custom_emb_location:
                run_config["ablations"] = run_config["ablations"] + [{
                    "id": card_id,
                    "location": custom_emb_location[0],
                    "target_layer": vis_config["y"] - 1 if "y" in vis_config and vis_config["y"] is not None else None,
                    "target_token": vis_config["x"] if "x" in vis_config else None,
                }]
                card_id += 1
                # Perform generation right after injection
                generate = True
        return run_config, card_id, generate

    @app.callback(
        Output("vis_config", "data"),
        [
            Input("generation_notify", "data"),
            Input("click_data_store", "data"),
            Input("choose_decoding", "value"),
            Input("choose_res_type", "value"),
            Input("norm_emb", "value"),
            Input("secondary_decoding", "value")
        ],
        [
            State("hide_start_table", "value"),
            State("hide_start_sankey", "value"),
            State("vis_config", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_vis_config(
        _, click_data, strategy, res_contrib, norm, secondary_decoding, 
        hide_start_table, hide_start_sankey, vis_config
    ):
        # If table visualization is hiding the first column, then offset all x-axis click data by 1
        col_0_offset = 1 if len(hide_start_table) > 0 else 0
        sankey_0_offset = 1 if len(hide_start_sankey) > 0 else 0
        vis_config |= {"source": "click" if ctx.triggered_id == "click_data_store" else None}
        vis_config |= {"click": click_data["click"]} if click_data else {"click": None}
        vis_config |= {
            "x": click_data["x"] + (col_0_offset if click_data["click"] == "table" else sankey_0_offset)
        } if click_data else {"x": None}
        vis_config |= {"y": click_data["y"]} if click_data else {"y": None}
        vis_config |= {"strategy": strategy}
        vis_config |= {"res_contrib": res_contrib}
        vis_config |= {"norm": norm}
        vis_config |= {"secondary_decoding": secondary_decoding}
        if ctx.triggered_id == "generation_notify":
            vis_config |= {"click": None, "x": None, "y": None}
        return vis_config

    @app.callback(
        Output("sankey_vis_config", "data"),
        [
            Input("generation_notify", "data"),
            Input("vis_config", "data"),
            Input("row_limit", "value"),
            Input("reapport_start", "value"),
            Input("hide_start_sankey", "value"),
            Input("att_high_k", "value"),
            Input("att_high_w", "value"),
            Input("attention_select", "value"),
            Input("hide_labels", "value"),
            Input("font_size", "value"),
            Input("sankey_size_adapt", "value"),
        ],
        [
            State("sankey_vis_config", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_sankey_vis_config(
        _, vis_config, row_limit, reapport_start, hide_start,
        att_high_k, att_high_w, att_select, hide_labels, # pylint:disable=unused-argument
        font_size, size_adapt, sankey_vis_config
    ):
        token = vis_config["x"] if "x" in vis_config and vis_config["x"] is not None else 0
        layer = vis_config["y"] if "y" in vis_config and vis_config["y"] is not None else 0
        hide_labels = len(hide_labels) > 0
        att_high = [locals()[v] if v else "" for k,v in ATTENTION_ID_MAP.items() if k == att_select][0]
        sankey_vis_config |= {"hide_start": len(hide_start) > 0}
        sankey_vis_config |= {"reapport_start": len(reapport_start) > 0}
        sankey_vis_config |= {
            "sankey_parameters": dataclasses.asdict(SankeyParameters(
                row_index=layer,
                token_index=token,
                rowlimit=row_limit,
                attention_select=att_select,
                attention_highlight=att_high,
                only_nodes_labels=hide_labels,
                font_size=font_size,
                size_adapt=size_adapt,
            ))
        }
        return sankey_vis_config


    @app.callback(
        Output("table_vis_config", "data"),
        [
            Input("hide_start_table", "value"),
            Input("font_size", "value"),
            Input("choose_embedding", "value"),
            Input("choose_colour", "value"),
        ],
        [
            State("table_vis_config", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_table_vis_config(hide_start, font_size, emb_type, colour, vis_config):
        vis_config |= {"hide_start": len(hide_start) > 0}
        vis_config |= {"font_size": font_size}
        vis_config |= {"emb_type": emb_type}
        vis_config |= {"colour": colour}
        return vis_config

    @app.callback(
        [
            Output("initial_callbacks", "data", allow_duplicate=True),
            Output("session_id", "data"),
            Output("current_run_config", "data"),
            Output("generation_notify", "data"),
            Output("generate_button_load", "notify"),
            Output("model_generate_alert", "children"),
        ],
        [
            Input("initial_callbacks", "data"),
            Input("generate_button", "n_clicks"),
            Input("start_generation_notify", "data")
        ],
        [
            State("text", "value"),
            State("model_id", "data"),
            State("run_config", "data"),
            State("vis_config", "data"),
        ],
        prevent_initial_call=True,
    )
    def call_model_generate(initial_callbacks, _gb, _gn, text, model_id, run_config, vis_config):
        initial_callbacks = process_initial_call(initial_callbacks) \
            if ctx.triggered_id and ctx.triggered_id == "initial_callbacks" else no_update
        session_id = str(uuid.uuid4())

        try:

            if not model_id:
                raise PreventUpdate

            with models_lock:
                if model_id not in models:
                    raise PreventUpdate
                model = models[model_id]

            _, layers, _, _, _ = model_generate(text, model_id, run_config, session_id)
            # Caching values
            norm = get_normalization(model_id=model_id, norm=vis_config["norm"])
            _ = decode_layers(
                layers=layers, strategy=vis_config["strategy"], norm=norm, norm_id=vis_config["norm"],
                secondary_decoding=vis_config["secondary_decoding"], decoder=model.decoder, _session_id=session_id
            )
            _ = compute_probabilities(
                layers=layers,
                strategy=vis_config["strategy"],
                residual_contribution=vis_config["res_contrib"],
                norm=norm, norm_id=vis_config["norm"],
                decoder=model.decoder,
                _session_id=session_id,
            )
        # Avoid locking input fields
        except (OutOfMemoryError, RuntimeError) as _:
            return initial_callbacks, session_id, no_update, False, True, "Out of memory"
        except PreventUpdate as _:
            return initial_callbacks, no_update, no_update, False, True, "Model not loaded"
        except Exception as _:
            return initial_callbacks, session_id, no_update, False, True, "Error during generation"

        # TODO: maybe add? # torch.cuda.empty_cache()
        return initial_callbacks, session_id, run_config, True, True, no_update


    @app.callback(
        [
            Output("table_graph", "figure"),
            Output("output_text", "value"),
            Output("display_alert", "children", allow_duplicate=True),
            Output("display_alert", "is_open", allow_duplicate=True),
        ],
        [
            Input("generation_notify", "data"),
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
    def update_table(
        _, tab_vis_config, vis_config, text, model_id, run_config, session_id
    ):
        try:
            if vis_config["source"] == "click" and vis_config["click"] == "sankey":
                return no_update, no_update, no_update, no_update

            with models_lock:
                if model_id not in models:
                    raise PreventUpdate
                model = models[model_id]

            norm = get_normalization(model_id, vis_config["norm"])

            # Retrieve model outputs
            generated_output, layers, input_tok, output_tok, session_id = model_generate(
                text, model_id, run_config, session_id
            )

            # Compute secondary tokens
            text = decode_layers(
                layers=layers, strategy=vis_config["strategy"], norm=norm, norm_id=vis_config["norm"],
                secondary_decoding=vis_config["secondary_decoding"], decoder=model.decoder, _session_id=session_id
            )
            text = extract_key_from_processed_layers(text, tab_vis_config["emb_type"])

            # Compute probabilities
            p = compute_probabilities(
                layers=layers, strategy=vis_config["strategy"], residual_contribution=vis_config["res_contrib"],
                norm=norm, norm_id=vis_config["norm"], decoder=model.decoder, _session_id=session_id,
            )

            # Extract information for max probability decoders
            p_max = extract_key_from_processed_layers(p, ProbabilityType.DECODER_MAX)
            p_max = extract_key_from_processed_layers(p_max, tab_vis_config["emb_type"])

            # Extract current probability
            p = extract_key_from_processed_layers(p, tab_vis_config["colour"])
            p = extract_key_from_processed_layers(p, tab_vis_config["emb_type"]) \
                if tab_vis_config["colour"] in [ProbabilityType.ARGMAX, ProbabilityType.ENTROPY] else p

            # Remove first column from visualization
            if tab_vis_config["hide_start"]:
                text = [layer[1:] for layer in text]
                p = [layer[1:] for layer in p]

            offset = 0 if tab_vis_config["hide_start"] else 1
            input_len = len(input_tok)
            output_len = len(output_tok)

            # TODO: find nicer workaround
            # Avoid empty tokens at start and end
            if p[0] == [] and text[0] == []:
                text[0] = [[""]] * (len(text[1]))
                p[0] = [None] * (len(p[1]))
            if p[-1] == [] and text[-1] == []:
                text[-1] = [[""]] * (len(text[1]))
                p[-1] = [None] * (len(p[1]))

            fig = go.Figure(data=go.Heatmap(
                z=p,
                text=text,
                xgap=2,
                ygap=2,
                x=[i - 0.5 for i in range(0, input_len + output_len)],
                y=list(range(0, model.model_config.num_hidden_layers + 2)),
                hovertemplate=TABLE_Z_FORMAT[tab_vis_config["colour"]] +
                "<br><b>Layer</b>: %{y}" +
                "<br><b>Token position</b>: %{x}" +
                "<br><b>" + SECONDARY_DECODING_TEXT[vis_config["secondary_decoding"]] + "</b>: %{text}" +
                "<extra></extra>",
                texttemplate="%{text[0]}",
                textfont={"size": tab_vis_config["font_size"]},
                colorscale="blues",
            ))
            fig.update_layout(
                margin={"l": 80, "r": 10, "t": 40, "b": 40},
                height=(model.model_config.num_hidden_layers + 2) * TABLE_HEIGHT_INCREMENT,
                width=(input_len + output_len) * TABLE_WIDTH_INCREMENT,
                xaxis={
                    "title_text": "", "tickmode": "linear", "titlefont": {"size": 20}, 
                    "showgrid": False, "zeroline": False, "range": [-0.5, input_len + output_len - 2 + offset - 0.5]
                },
                yaxis={
                    "title_text": "Transformer Layers", "tickmode": "linear", "titlefont": {"size": 20},
                    "showgrid": False, "zeroline": False, "range": [-0.5, model.model_config.num_hidden_layers + 1.5]
                },
                plot_bgcolor="white",
                template="plotly",
                modebar_remove=["zoom", "pan", "zoomIn", "zoomOut", "autoScale"],
                dragmode=False,
            )

            color_blues_max = "#08306B"
            color_blues_min = "#F7FBFF"
            color_blues_mid = "#6AAED6"

            fig.add_shape(
                x0=input_len+offset-1.5, x1=input_len+offset-1.5, y0=-1, y1=0.5,
                line_width=8, line_color="white"
            )
            fig.add_shape(
                x0=input_len+offset-1.5, x1=input_len+offset-1.5, y0=-1, y1=0.5,
                line_width=2, line_color=color_blues_max
            )
            fig.add_shape(
                x0=input_len+offset-2.5, x1=input_len+offset-2.5, y0=0.5, y1=model.model_config.num_hidden_layers + 1.5,
                line_width=8, line_color="white"
            )
            fig.add_shape(
                x0=input_len+offset-2.5, x1=input_len+offset-2.5, y0=0.5, y1=model.model_config.num_hidden_layers + 1.5,
                line_width=2, line_color=color_blues_max
            )

            fig.add_hline(y=0.5, line_width=8, line_color="white")
            fig.add_hline(y=0.5, line_width=2, line_color=color_blues_max)
            fig.add_hline(y=model.model_config.num_hidden_layers + 0.5, line_width=8, line_color="white")
            fig.add_hline(y=model.model_config.num_hidden_layers + 0.5, line_width=2, line_color=color_blues_max)

            # Input/Output annotations
            for i, tok in enumerate(input_tok[1 - offset:]):
                fig.add_annotation(
                    x=i, y=model.model_config.num_hidden_layers + 1.5, yshift=10, xshift=-TABLE_WIDTH_INCREMENT,
                    xref="x", yref="y", yanchor="bottom",
                    text=f"{tok}", hovertext=f"{tok}",
                    bgcolor="#94CCF9", bordercolor="black", opacity=0.7,
                    showarrow=False,
                )
                fig.add_annotation(
                    x=i, y=-0.5, yshift=-20,
                    xref="x", yref="y", yanchor="top",
                    text=f"{tok}", hovertext=f"{tok}",
                    bgcolor="#94CCF9", bordercolor="black", opacity=0.7,
                    showarrow=False,
                )
            for i, tok in enumerate(output_tok):
                shift = 0
                if i >= output_len - 1:
                    index = i
                    i = output_len - 2
                    shift = (index + 2 - output_len) * TABLE_WIDTH_INCREMENT
                fig.add_annotation(
                    x=input_len + i - 1 + offset, y=-0.5, yshift=-20, xshift=shift,
                    xref="x", yref="y", yanchor="top",
                    text=f"{tok}", hovertext=f"{tok}",
                    bgcolor="#FEE69A", bordercolor="black", opacity=0.7,
                    showarrow=False,
                )

            # Max decoding bookmarks
            if any(any(l) for l in p_max):
                fig.update_layout(shapes=fig.layout.shapes + tuple([
                    dict(
                        type="path",
                        path=
                            f"M {j + 0.482} {i + 0.20} L {j + 0.482} {i + 0.45} L {j + 0.38} {i + 0.45} Z" if c == DecodingType.OUTPUT else 
                            f"M {j + 0.482} {i - 0.20} L {j + 0.482} {i - 0.45} L {j + 0.38} {i - 0.45} Z",
                        line_width=1, line_color=color_blues_mid, fillcolor=color_blues_mid,
                    ) for i, l in enumerate(p_max) for j, c in enumerate(l[1 - offset:])
                ]))

            # Ablation reminders
            for abl in run_config["ablations"]:
                if abl["location"] == tab_vis_config["emb_type"]:
                    fig.add_shape(
                        x0=abl["target_token"] + offset - 0.5, x1=abl["target_token"] + offset - 1.5,
                        y0=abl["target_layer"] - 0.5 + 1, y1=abl["target_layer"] + 0.5 + 1,
                        line_width=2, line_color="red"
                    )
            # Injects reminders
            for inj in run_config["injects"]:
                if inj["location"] == tab_vis_config["emb_type"]:
                    fig.add_shape(
                        x0=inj["target_token"] + offset - 0.5, x1=inj["target_token"] + offset - 1.5,
                        y0=inj["target_layer"] - 0.5 + 1, y1=inj["target_layer"] + 0.5 + 1,
                        line_width=2, line_color="green"
                    )
            # Cell selector
            if vis_config["x"] is not None and vis_config["y"] is not None:
                fig.add_shape(
                    x0=vis_config["x"] + offset - 0.5, x1=vis_config["x"] + offset - 1.5,
                    y0=vis_config["y"] - 0.5, y1=vis_config["y"] + 0.5,
                    line_width=2, line_color="orange"
                )

            return fig, model.tokenizer.decode(generated_output["sequences"].squeeze()[input_len:]), [], False
        
        except Exception as ex:
            return no_update, no_update, "Layout error:" + str(ex), True

    @app.callback(
        [
            Output("sankey_graph", "figure"),
            Output("display_alert", "children", allow_duplicate=True),
            Output("display_alert", "is_open", allow_duplicate=True),
        ],
        [
            Input("vis_config", "data"),
            Input("sankey_vis_config", "data"),
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
        try:
            if vis_config["source"] == "click" and vis_config["click"] == "sankey":
                return no_update, no_update, no_update

            with models_lock:
                if model_id not in models:
                    raise PreventUpdate
                model = models[model_id]
            if (
                "x" not in vis_config or "y" not in vis_config or
                vis_config["x"] is None or vis_config["y"] is None or
                # Set threhsold to 1 when not visualizing token 0, to avoid visualization glitches
                vis_config["x"] <= (1 if sankey_vis_config["hide_start"] else 0) or
                vis_config["y"] <= (1 if sankey_vis_config["hide_start"] else 0)
            ):
                x, y = None, None
            else:
                x, y = vis_config["x"], vis_config["y"]

            sankey_param = SankeyParameters(**sankey_vis_config["sankey_parameters"])

            dfs, linkinfo, input_len, output_len = generate_sankey_info(
                text, model_id, run_config, session_id,
                vis_config["strategy"], vis_config["res_contrib"], vis_config["norm"], vis_config["secondary_decoding"],
            )

            row_limit = sankey_param.rowlimit
            sankey_param.rowlimit = row_limit \
                if sankey_param.row_index - row_limit - 1 >= 0 or sankey_param.row_index == 0 \
                else sankey_param.row_index

            if sankey_vis_config["hide_start"]:
                dfs = {key: [layer[1:] for layer in df] for key, df in dfs.items()}
                linkinfo = {
                    key: [layer[1:] if layer is not None else None for layer in link]
                    for key, link in linkinfo.items()
                }
                linkinfo["attentions"] = [[
                    (layer2[1:] / layer2[1:].sum()) if sankey_vis_config["reapport_start"] else layer2[1:]
                    for layer2 in layer1
                ] for layer1 in linkinfo["attentions"]]

            token_offset = 1 if sankey_vis_config["hide_start"] else 0
            if x is None and y is None:
                sankey_param.row_index = model.model_config.num_hidden_layers + 1
                sankey_param.token_index = input_len + output_len - token_offset - 2
                sankey_param.rowlimit = sankey_param.row_index - row_limit
                sankey_info = generate_complete_sankey(dfs, linkinfo, sankey_param, output_len)
            else:
                sankey_param.token_index -= token_offset
                sankey_param.rowlimit = sankey_param.row_index - sankey_param.rowlimit
                sankey_info = generate_sankey(dfs, linkinfo, sankey_param)
            hide_nodes = [
                (
                    (abl["target_layer"] + 1, abl["target_token"] - token_offset),
                    get_label_type_map(EMB_TYPE_SANKEY_NODE_MAP, abl["location"])
                ) for abl in run_config["ablations"]
            ]
            fig = format_sankey(*sankey_info, linkinfo, sankey_param, hide_nodes)
            return fig, [], False

        except Exception as ex:
            return no_update, "Layout error:" + str(ex), True

    @app.callback(
        [
            Output("model_id", "data", allow_duplicate=True),
            Output("model_info", "data", allow_duplicate=True),
        ],
        Input("model_select", "value"),
        prevent_initial_call=True,
    )
    def update_model_id(model_info):
        model_info = decode_dataclass(model_info, ModelInfo)
        return model_info.id, dataclasses.asdict(model_info)

    @app.callback(
        [
            Output("initial_callbacks", "data", allow_duplicate=True),
            Output("model_id", "data", allow_duplicate=True),
            Output("model_info", "data", allow_duplicate=True),
            Output("model_select_alert", "children"),
            Output("model_select_alert", "is_open"),
            Output("new_model_notify", "data"),
        ],
        [
            Input("initial_callbacks", "data"),
            Input("model_id", "data"),
        ],
        [
            State("model_info", "data"),
        ],
        running=[(Output("overlay", "class"), "overlay show", "overlay")],
        prevent_initial_call=True,
    )
    def update_model(initial_callbacks, model_id, model_info):
        nonlocal models
        initial_callbacks = process_initial_call(initial_callbacks) \
            if ctx.triggered_id and ctx.triggered_id == "initial_callbacks" else no_update

        if not model_id:
            return initial_callbacks, no_update, no_update, no_update, False, no_update

        # A dedicated lock is used since we are interested in model loading
        # being exclusive only with othere instances of itself
        with model_loading_lock:
            if model_id not in models:
                try:
                    model = ModelUtils(ModelInfo(**model_info), hf_token=os.environ["HF_TOKEN"])
                    if model.model is None:
                        return initial_callbacks, "", {}, \
                            "Out of memory while loading model", True, no_update
                    models |= {model_id: model}
                except RuntimeError as _:
                    return initial_callbacks, "", {}, "No NVIDIA Driver found", True, no_update
                except Exception as _:
                    return initial_callbacks, "", {}, "Error while loading model", True, no_update
        return initial_callbacks, model_id, model_info, no_update, False, True

    @app.callback(
        [
            Output("click_data_store", "data"),
            Output("table_graph", "clickData"),
            Output("sankey_graph", "clickData")
        ],
        [
            Input("table_graph", "clickData"),
            Input("sankey_graph", "clickData"),
        ],
        prevent_initial_call=True,
    )
    def click_data_handler(table_click_data, sankey_click_data):
        click_data = {"notify": str(uuid.uuid4())}
        if table_click_data:
            click_data |= {
                "x": table_click_data["points"][0]["x"], "y": table_click_data["points"][0]["y"],
                "bb_x": table_click_data["points"][0]["bbox"]["x0"] + TABLE_WIDTH_INCREMENT / 2,
                "bb_y": table_click_data["points"][0]["bbox"]["y0"],
                "click": "table"
            }
        elif sankey_click_data:
            click_data |= {
                "x": math.ceil(sankey_click_data["points"][0]["customdata"]["x"]),
                "y": math.ceil(sankey_click_data["points"][0]["customdata"]["y"]),
                "bb_x": sankey_click_data["points"][0]["y0"] + sankey_click_data["points"][0]["dy"] / 2 + SANKEY_LEFT_MARGIN,
                "bb_y": sankey_click_data["points"][0]["x0"] + SANKEY_TOP_MARGIN,
                "type": get_value_type_map(
                    EMB_TYPE_SANKEY_NODE_MAP, sankey_click_data["points"][0]["customdata"]["type"]
                ),
                "click": "sankey"
            }
        else:
            click_data = no_update, None, None
        return click_data, None, None

    @app.callback(
        [
            Output("sankey_tooltip", "is_open", allow_duplicate=True),
            Output("sankey_tooltip_target", "style"),
            Output("sankey_tooltip", "children", allow_duplicate=True),
        ],
        [
            Input("generation_notify", "data"),
            # Click data handler (only one, others are updated through click_data_store)
            Input("click_data_store", "data"),
        ],
        [
            State("sankey_scroll", "data"),
        ],
        prevent_initial_call=True,
    )
    def display_sankey_embedding_tooltip(_, click_data, sankey_scroll):
        if ctx.triggered_id == "generation_notify" or click_data["y"] <= 0 or click_data["click"] != "sankey":
            return False, no_update, []

        children = [extra_layout.generate_tooltip_children_layout(
            layer=click_data["y"],
            token=click_data["x"],
            emb_type=click_data["type"],
            ablation_opt=True if click_data["type"] in [EmbeddingsType.POST_FF, EmbeddingsType.POST_ATTENTION] else False
        )]

        x_tooltip = click_data["bb_x"] - sankey_scroll
        y_tooltip = click_data["bb_y"]

        tooltip_style = {
            "transform": f"translate({x_tooltip}px, {y_tooltip}px)"
        }

        return True, tooltip_style, children

    @app.callback(
        [
            Output("table_tooltip", "is_open", allow_duplicate=True),
            Output("table_tooltip_target", "style"),
            Output("table_tooltip", "children", allow_duplicate=True),
        ],
        [
            Input("generation_notify", "data"),
            # Click data handler (only one, others are updated through click_data_store)
            Input("click_data_store", "data"),
        ],
        [
            State("table_scroll", "data"),
        ],
        prevent_initial_call=True,
    )
    def display_table_embedding_tooltip(_, click_data, table_scroll):
        if ctx.triggered_id == "generation_notify" or \
                click_data["y"] <= 0 or \
                click_data["click"] != "table":
            return False, no_update, []

        children = [extra_layout.generate_tooltip_children_layout(
            layer=click_data["y"],
            token=click_data["x"],
        )]

        x_tooltip = click_data["bb_x"] - table_scroll
        y_tooltip = click_data["bb_y"]

        tooltip_style = {
            "transform": f"translate({x_tooltip}px, {y_tooltip}px)"
        }

        return True, tooltip_style, children

    @app.callback(
        Output("mod_container", "children"),
        [
            Input("run_config", "data"),
            Input("table_vis_config", "data"),
        ],
        [
            State("mod_container", "children"),
        ],
        prevent_initial_call=True,
    )
    def manage_inject_cards(run_config, tab_vis_config, mod_container):
        mods = run_config["injects"] + run_config["ablations"]
        if len(mods) == len(mod_container) and ctx.triggered_id != "table_vis_config":
            raise PreventUpdate
        
        # TODO: better to re-create each card every time or look for removed/new cards and remove/add them?
        mods = sorted(mods, key=lambda m: m["id"])
        return [
            extra_layout.generate_inject_card(
                card_id=mod["id"],
                text=mod["text"],
                inj_type=mod["type"],
                position=mod["location"],
                decoding=mod["decoding"],
                norm=mod["norm"],
                token=mod["target_token"] - (1 if tab_vis_config["hide_start"] else 0),
                layer=mod["target_layer"] + 1
            ) if mod in run_config["injects"] else
            extra_layout.generate_ablation_card(
                card_id=mod["id"],
                position=mod["location"],
                token=mod["target_token"] - (1 if tab_vis_config["hide_start"] else 0),
                layer=mod["target_layer"] + 1
            )
            for mod in mods
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
            dead = [
                mid
                for mid, model in models.items()
                if cur - model.heartbeat_stamp > HEARTBEAT_TIMEOUT and mid != model_id
            ]
            if dead:
                for mid in dead:
                    del models[mid]
                gc.collect()
                # TODO: add check for device
                torch.cuda.empty_cache()
            # Old sessions might be present and still send heartbeat signals
            if model_id in models:
                # Revive current model
                cur = time.time()
                models[model_id].heartbeat_stamp = cur

    @app.callback(
        [
            Output("row_limit", "max"),
            Output("run_config", "data", allow_duplicate=True),
        ],
        Input("new_model_notify", "data"),
        [
            State("model_id", "data"),
            State("run_config", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_components_new_model(_, model_id, run_config):
        with models_lock:
            if model_id not in models:
                raise PreventUpdate
            model = models[model_id]
        run_config |= {"injects": [], "ablations": []}
        return model.model_config.num_hidden_layers, run_config


    # Layout callbacks

    @app.callback(
        [
            Output("text", "disabled", allow_duplicate=True),
            Output("model_select", "disabled", allow_duplicate=True),
        ],
        [
            Input("initial_callbacks", "data"),
            Input("generate_button", "n_clicks"),
            Input("start_generation_notify", "data"),
        ],
        prevent_initial_call=True,
    )
    def generate_start_hooks(*_):
        return True, True

    @app.callback(
        [
            Output("text", "disabled", allow_duplicate=True),
            Output("model_select", "disabled", allow_duplicate=True),
            Output("model_generate_alert", "is_open"),
        ],
        [
            Input("generation_notify", "data"),
        ],
        prevent_initial_call=True,
    )
    def generate_end_hooks(notify):
        return False, False, not notify

    @app.callback(
        [
            Output("att_high_k_div", "style"),
            Output("att_high_w_div", "style"),
        ],
        [
            Input("attention_select", "value"),
        ],
    )
    def update_attention_layout(attention_select):
        _parameter_order = ["att_high_k", "att_high_w"]
        styles = [
            {"display": "flex"} if attention_select == p_key else {"display": "none"}
            for p in _parameter_order
            if (p_key := next(k for k, v in ATTENTION_ID_MAP.items() if v == p)) and True  # pylint:disable=simplifiable-condition
        ]
        return styles

    @app.callback(
        Output("reapport_start_div", "style"),
        [
            Input("hide_start_sankey", "value"),
        ],
    )
    def update_sankey_hide_buttons(hide_start_sankey):
        style = {"display": "flex"} if len(hide_start_sankey) > 0 else {"display": "none"}
        return style

    @app.callback(
        Output("sankey_graph", "style"),
        [
            Input("sankey_scale", "value"),
        ],
    )
    def update_sankey_scale(scale):
        return {"transform": f"scale({scale})", "transform-origin": "top left"}


    # Client-side callbacks

    clientside_callback(
        ClientsideFunction(
            namespace="clientside",
            function_name="detect_scroll_inject"
        ),
        Output("javascript_inject", "children"),
        Input("javascript_inject", "children")
    )

    clientside_callback(
        ClientsideFunction(
            namespace="clientside",
            function_name="update_table_scroll"
        ),
        Output("table_scroll", "data"),
        Input("scrollable_table_js_store", "children"),
    )

    clientside_callback(
        ClientsideFunction(
            namespace="clientside",
            function_name="update_sankey_scroll"
        ),
        Output("sankey_scroll", "data"),
        Input("scrollable_sankey_js_store", "children"),
    )
