from utils import EmbeddingTypes

STRATEGY_MAP = [
    {"label": "output_decoder", "value": "output"},
    {"label": "interpolated", "value": "interpolation"},
    {"label": "input_encoder", "value": "input"},
]
EMB_TYPE_MAP = [
    {"label": "Residual + FFNN", "value": EmbeddingTypes.BLOCK_OUTPUT},
    {"label": "FFNN", "value": EmbeddingTypes.POST_FF},
    {"label": "Residual + Self Attention", "value": EmbeddingTypes.POST_ATTENTION_RESIDUAL},
    {"label": "Self Attention", "value": EmbeddingTypes.POST_ATTENTION},
]
