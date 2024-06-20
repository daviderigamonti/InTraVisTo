from inspect import signature, Parameter
from typing import List, Set, Type, Optional
from enum import Enum
from dataclasses import dataclass
from copy import copy

import inspect

import torch

from transformer_wrappers.wrappers import TransformerWrapper, CausalLMWrapper
from transformer_wrappers.wrappers.base import LayerWrapper, LayersWrapper, ModuleWrapper
from transformer_wrappers.wrappers.constants import * # pylint:disable=W0401,W0614


INJECTS_PARAMETER = "inject_info"

def _inject(hidden_states: torch.FloatTensor, inject_embedding: torch.FloatTensor, inject_position: int):
    hidden_states[0, inject_position, :] = inject_embedding
    return hidden_states


class InjectPosition(Enum):
    ATTENTION = "inject_attention"
    INTERMEDIATE = "inject_intermediate"
    FFNN = "inject_ffnn"
    OUTPUT = "inject_output"

@dataclass
class InjectInfo:
    layer: int
    token: int
    position: InjectPosition
    embedding: torch.FloatTensor
    index: int = -1


class ParameterCausalLMWrapper(CausalLMWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gen_parameters: Set[str] = set()
        self._original_parameters: List[Parameter] = list(signature(self.base_model.forward).parameters.values())

    @property
    def gen_parameters(self):
        return copy(self._gen_parameters)

    def add_parameters(self, parameters: List[str] = None):
        if parameters:
            self._gen_parameters = self._gen_parameters.union(set(parameters))
            new_parameters = self._original_parameters + [
                Parameter(param, Parameter.KEYWORD_ONLY, default=None) for param in self._gen_parameters]
            self.base_model.forward.__signature__ = signature(
                self.base_model.forward).replace(
                parameters=new_parameters)

    def remove_parameters(self, parameters: List[str] = None):
        if parameters:
            self._gen_parameters = self._gen_parameters.difference(set(parameters))
            new_parameters = self._original_parameters + [
                Parameter(param, Parameter.KEYWORD_ONLY, default=None) for param in self._gen_parameters]
            self.base_model.forward.__signature__ = signature(
                self.base_model.forward).replace(
                parameters=new_parameters)

    def prepare_inputs_for_generation(self, *args, base_model_output: bool = True, **kwargs):
        #
        inputs = self._model.prepare_inputs_for_generation(*args, **kwargs)
        if self.is_wrapping:
            inputs |= {'base_model_output': base_model_output}
        # Add custom generation parameters
        inputs |= {param: kwargs[param] for param in self._gen_parameters if param in kwargs}

        return inputs

    def generate(self, *args, return_inner_states: bool = False, **kwargs):
        if not self.is_wrapping:
            return self.base_model.generate(*args, **kwargs)

        # Bypass CausalLMWrapper's generate implementation
        generate_output = super(CausalLMWrapper, self).generate(*args, **kwargs) # pylint:disable=E1101

        # Re-run through layers to collect all data  # TODO find better solution
        if return_inner_states or not self.is_benchmarking:
            #
            return self.forward(
                input_ids=generate_output,
                **{
                    k: kwargs.get(k)
                    for k in set(inspect.signature(self.prepare_inputs_for_generation).parameters.keys())
                    if k not in {'args', 'kwargs', 'self', 'base_model_output'}
                } | {param: kwargs[param] for param in self._gen_parameters if param in kwargs},
                return_dict=True,
                output_attentions=True,
                use_cache=True,
                output_hidden_states=True,
                return_attention_output=True,  # Self-attention layer output
                return_feed_forward_output=True,
                return_intermediate_hidden_states=True
            ) | {'output_ids': generate_output}
        return generate_output


class InjectLayerWrapper(LayerWrapper):
    INJECT_CANDIDATES = "inj_cand"

    def _attn_forward_inject(
            self, current_hidden_state: Optional[torch.FloatTensor],
            add_attn_residual: bool = True, inj_cand: InjectInfo = None, **kwargs):
        if current_hidden_state is None:
            raise ValueError()  # TODO add message
        #
        residual = current_hidden_state
        # Initial Normalisation
        current_hidden_state = self.initial_norm.forward(current_hidden_state)
        # Self attention
        attention_output = self.attention_wrapper.forward(
            current_hidden_state=current_hidden_state, **kwargs
        ).pop(self.attention_wrapper.module_output)

        # Injection ATTENTION
        attention_output |= {
            self.attention_wrapper.module_output: _inject(
                attention_output[self.attention_wrapper.module_output], inj_info.embedding, inj_info.index
            )
            for inj_info in inj_cand if inj_info.position == InjectPosition.ATTENTION
        }

        if add_attn_residual:
            current_hidden_state = attention_output[self.attention_wrapper.module_output] + residual
        else:
            current_hidden_state = attention_output[self.attention_wrapper.module_output]

        # Injection INTERMEDIATE
        current_hidden_state = [
            _inject(current_hidden_state, inj_info.embedding, inj_info.index)
            if inj_info.position == InjectPosition.INTERMEDIATE else current_hidden_state
            for inj_info in inj_cand
        ][0]

        #
        output = kwargs | {
            CURR_HIDDEN_STATE: current_hidden_state,
            self.intermediate_module_output: current_hidden_state,
            ADD_ATTN_RESIDUAL: add_attn_residual,
            self.attention_wrapper.module_output: attention_output,
            self.INJECT_CANDIDATES: inj_cand
        }

        return output

    def _ffnn_forward_inject(
            self, current_hidden_state: Optional[torch.FloatTensor],
            add_ffnn_residual: bool = True, inj_cand: InjectInfo = None, **kwargs):
        if current_hidden_state is None:
            raise ValueError()  # TODO add message
        #
        residual = current_hidden_state
        # Intermediate Normalisation
        current_hidden_state = self.intermediate_norm.forward(current_hidden_state)
        # Feed-Forward
        ffnn_output = self.feed_forward_wrapper.forward(
            current_hidden_state=current_hidden_state, **kwargs
        ).pop(self.feed_forward_wrapper.module_output)

        # Injection FFNN
        ffnn_output |= {
            self.feed_forward_wrapper.module_output: _inject(
                ffnn_output[self.feed_forward_wrapper.module_output], inj_info.embedding, inj_info.index
            )
            for inj_info in inj_cand if inj_info.position == InjectPosition.FFNN
        }

        if add_ffnn_residual:
            current_hidden_state = ffnn_output[self.feed_forward_wrapper.module_output] + residual  # TODO verify this
        else:
            current_hidden_state = ffnn_output[self.feed_forward_wrapper.module_output]

        # Injection OUTPUT
        current_hidden_state = [
            _inject(current_hidden_state, inj_info.embedding, inj_info.index)
            if inj_info.position == InjectPosition.OUTPUT else current_hidden_state
            for inj_info in inj_cand
        ][0]

        # Extend input with module output
        output = kwargs | {
            CURR_HIDDEN_STATE: current_hidden_state,
            ADD_FFNN_RESIDUAL: add_ffnn_residual,
            self.feed_forward_wrapper.module_output: ffnn_output
        }

        return output

    def _wrapped_forward_inject(
        self,
        skip_attention: bool = False,
        skip_ffnn: bool = False,
        **kwargs
    ):
        output = kwargs
        output = self._attn_forward_inject(**output)
        output = self._ffnn_forward_inject(**output)
        #
        output |= {self.module_output: output[CURR_HIDDEN_STATE]}

        return output

    def _wrapped_forward(self, **kwargs): # pylint:disable=W0221
        return self._wrapped_forward_inject(
            **kwargs) if self.INJECT_CANDIDATES in kwargs else super()._wrapped_forward(
            self, **kwargs)


class InjectLayersWrapper(LayersWrapper):
    _layer_dtype: Type[ModuleWrapper] = InjectLayerWrapper

    def _wrapped_forward(self, **kwargs):
        injections = kwargs[INJECTS_PARAMETER]
        position_ids = kwargs[POSITION_IDS].squeeze()
        output = self._init_state(**kwargs)
        # Iterate over layers
        for layer_idx, layer_wrapper in enumerate(self.get_layer_wrappers_iterator()):
            # Apply layer transformation
            if injections and (
                    inj_candidates := [i for i in injections if i.layer == layer_idx and i.token in position_ids]):
                for inj in inj_candidates:
                    inj.index = (position_ids == inj.token).nonzero(as_tuple=True)[0]
                output = layer_wrapper.forward(layer_idx=layer_idx, inj_cand=inj_candidates, **output)
            else:
                output = layer_wrapper.forward(layer_idx=layer_idx, **output)
            # Update model state
            output = self._update_state(**output)

        return output


class InjectTransformerWrapper(TransformerWrapper):
    _layers_dtype: Type[TransformerWrapper] = InjectLayersWrapper


class InjectCausalLMWrapper(ParameterCausalLMWrapper):
    _transformer_dtype: Type[TransformerWrapper] = InjectTransformerWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_parameters([INJECTS_PARAMETER])

    def generate(self, *args, return_inner_states: bool = False, **kwargs):
        return ParameterCausalLMWrapper.generate(self, *args, **kwargs)
