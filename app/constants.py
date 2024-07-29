import dataclasses
import json

from utils import EmbeddingsType, DecodingType, ProbabilityType, ResidualContribution, ModelInfo


def get_label_type_map(type_map, value):
    return [i["label"] for i in type_map if i["value"] == value][0]

def encode_dataclass(c):
    return json.dumps(dataclasses.asdict(c))

def decode_dataclass(c, class_name):
    return class_name(**json.loads(c))


CUDA_DEVICE = "cuda"

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

MODEL_MAP = [
    {"label": "No Model", "value": encode_dataclass(ModelInfo())},
    {"label": "GPT-2", "value": encode_dataclass(ModelInfo("gpt2", CUDA_DEVICE, False))},
    {"label": "GPT-2 (CPU)", "value": encode_dataclass(ModelInfo("gpt2", "cpu", False))},
    {"label": "Llama 2 7B (CPU)", "value": encode_dataclass(ModelInfo("meta-llama/Llama-2-7b-hf", "cpu", False))},
    {"label": "Llama 2 7B (4bit)", "value": encode_dataclass(ModelInfo("meta-llama/Llama-2-7b-hf", CUDA_DEVICE, True))},
    {"label": "Llama 2 7B", "value": encode_dataclass(ModelInfo("meta-llama/Llama-2-7b-hf", CUDA_DEVICE, False))},
    {"label": "Mistral Instruct 0.2 (4bit)", "value": encode_dataclass(ModelInfo("mistralai/Mistral-7B-Instruct-v0.2", CUDA_DEVICE, True))},
    {"label": "Mistral Instruct 0.2", "value": encode_dataclass(ModelInfo("mistralai/Mistral-7B-Instruct-v0.2", CUDA_DEVICE, False))},
]

TABLE_Z_FORMAT = {
    ProbabilityType.ARGMAX: "<i>Probability</i>: %{z:1.3%}",
    ProbabilityType.ENTROPY: "<i>Entropy</i>: %{z:.3f} nats",
    ProbabilityType.ATT_RES_PERCENT: "<i>Contribution</i>: %{z:1.2%}",
    ProbabilityType.FFNN_RES_PERCENT: "<i>Contribution</i>: %{z:1.2%}",
}

TABLE_HEIGHT_INCREMENT = 32
TABLE_WIDTH_INCREMENT = 85

HEARTBEAT_INTERVAL = 10 # Seconds
HEARTBEAT_TIMEOUT = 15 # Seconds