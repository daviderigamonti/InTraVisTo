import dataclasses
import json

from utils.utils import EmbeddingsType, DecodingType, ProbabilityType, ResidualContribution, SecondaryDecodingType
from utils.models import ModelInfo, NormalizationStep
from utils.sankey import AttentionHighlight, SizeAdapt


def get_label_type_map(type_map, value):
    return [i["label"] for i in type_map if i["value"] == value][0]

def encode_dataclass(c):
    return json.dumps(dataclasses.asdict(c))

def decode_dataclass(c, class_name):
    return class_name(**json.loads(c))


TITLE = "InTraVisTo"
ASSETS_PATH = "./app/assets/"
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.13.0/css/all.css"
CUDA_DEVICE = "cuda"

DASH_SESSION_DIFF_GEN = "TOKEN_DIFFS"

DECODING_TYPE_MAP = [
    {"label": "Output Decoder", "value": DecodingType.OUTPUT},
    {"label": "Linear Interpolation", "value": DecodingType.LINEAR},
    {"label": "Quadratic Interpolation", "value": DecodingType.QUADRATIC},
    {"label": "Input Encoder", "value": DecodingType.INPUT},
]

EMB_TYPE_MAP = [
    {"label": "Residual + FFNN", "value": EmbeddingsType.BLOCK_OUTPUT},
    {"label": "FFNN", "value": EmbeddingsType.POST_FF},
    {"label": "Residual + Self Attention", "value": EmbeddingsType.POST_ATTENTION_RESIDUAL},
    {"label": "Self Attention", "value": EmbeddingsType.POST_ATTENTION},
]

PROB_TYPE_MAP = [
    {"label": "P(argmax term)", "value": ProbabilityType.ARGMAX},
    {"label": "Entropy[p]", "value": ProbabilityType.ENTROPY},
    {"label": "Attention Contribution %", "value": ProbabilityType.ATT_RES_PERCENT},
    {"label": "FFNN Contribution %", "value": ProbabilityType.FFNN_RES_PERCENT},
]

RES_TYPE_MAP = [
    {"label": "Norm proportion", "value": ResidualContribution.NORM},
    {"label": "KL Divergence proportion", "value": ResidualContribution.KL_DIV},
]

ATTENTION_ID_MAP = {
    AttentionHighlight.NONE: "",
    AttentionHighlight.ALL: "",
    AttentionHighlight.TOP_K: "att_high_k",
    AttentionHighlight.MIN_WEIGHT: "att_high_w",
}

ATTENTION_MAP = [
    {"label": "None", "value": AttentionHighlight.NONE},
    {"label": "All", "value": AttentionHighlight.ALL},
    {"label": "Top K attention weights", "value": AttentionHighlight.TOP_K},
    {"label": "Exclude low-value weights", "value": AttentionHighlight.MIN_WEIGHT},
    #{"label": "", "value": AttentionHighlight.TOP_P},
]

NORM_MAP = [
    {"label": "No normalisation", "value": NormalizationStep.NONE},
    {"label": "Normalisation only", "value": NormalizationStep.ONLY_NORM},
    {"label": "Normalisation with rescaling parameters", "value": NormalizationStep.NORM_SCALE},
]

SECONDARY_DECODING_MAP = [
    {"label": "Top-5 probability tokens", "value": SecondaryDecodingType.TOP_K},
    {"label": "Iterative subtraction decoding", "value": SecondaryDecodingType.ITERATIVE},
]
SECONDARY_DECODING_TEXT = {
    SecondaryDecodingType.TOP_K: "Top-5 tokens",
    SecondaryDecodingType.ITERATIVE: "Secondary tokens",
}

SANKEY_SIZE_MAP = [
    {"label": "Fixed scale", "value": SizeAdapt.FIXED},
    {"label": "Adapt based on tokens", "value": SizeAdapt.TOKEN},
    {"label": "Adapt based on layers", "value": SizeAdapt.LAYER},
]

MODEL_MAP = [
    {"label": "No Model", "value": encode_dataclass(ModelInfo())},
    {"label": "GPT-2", "value": encode_dataclass(ModelInfo("gpt2", CUDA_DEVICE, False))},
    {"label": "GPT-2 (CPU)", "value": encode_dataclass(ModelInfo("gpt2", "cpu", False))},
    {"label": "Llama 2 7B (4bit)", "value": encode_dataclass(ModelInfo("meta-llama/Llama-2-7b-hf", CUDA_DEVICE, True))},
    {"label": "Llama 2 7B", "value": encode_dataclass(ModelInfo("meta-llama/Llama-2-7b-hf", CUDA_DEVICE, False))},
    {"label": "Llama 3 8B (4bit)", \
        "value": encode_dataclass(ModelInfo("meta-llama/Meta-Llama-3-8B", CUDA_DEVICE, True))},
    {"label": "Mistral Instruct 0.2 (4bit)", \
        "value": encode_dataclass(ModelInfo("mistralai/Mistral-7B-Instruct-v0.2", CUDA_DEVICE, True))},
    {"label": "Mistral Instruct 0.2", \
        "value": encode_dataclass(ModelInfo("mistralai/Mistral-7B-Instruct-v0.2", CUDA_DEVICE, False))},
    {"label": "Gemma 2B", "value": encode_dataclass(ModelInfo("google/gemma-2b", CUDA_DEVICE, False))},
    {"label": "Gemma 7B (4bit)", "value": encode_dataclass(ModelInfo("google/gemma-7b", CUDA_DEVICE, True))},
    # {"label": "Gemma 2 2B", "value": encode_dataclass(ModelInfo("google/gemma-2-2b", CUDA_DEVICE, False))},
    # {"label": "Gemma 2 2B (CPU)", "value": encode_dataclass(ModelInfo("google/gemma-2-2b", "cpu", False))},
]

TABLE_Z_FORMAT = {
    ProbabilityType.ARGMAX: "<i>Probability</i>: %{z:1.3%}",
    ProbabilityType.ENTROPY: "<i>Entropy</i>: %{z:.3f} nats",
    ProbabilityType.ATT_RES_PERCENT: "<i>Contribution</i>: %{z:1.2%}",
    ProbabilityType.FFNN_RES_PERCENT: "<i>Contribution</i>: %{z:1.2%}",
}

TABLE_HEIGHT_INCREMENT = 32
TABLE_WIDTH_INCREMENT = 85
GRAPH_TOOLTIP_OFFSET_X = 11.7
GRAPH_TOOLTIP_OFFSET_Y = 628.6

HEARTBEAT_INTERVAL = 10 # Seconds
HEARTBEAT_TIMEOUT = 15 # Seconds
