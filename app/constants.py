from utils import EmbeddingsType, DecodingType, ProbabilityType

DECODING_TYPE_MAP = [
    {"label": "Output Decoder", "value": DecodingType.OUTPUT},
    {"label": "Linear Interpolation", "value": DecodingType.LINEAR},
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

MODEL_MAP = [
    {"label": "Llama 2 ", "value": "meta-llama/Llama-2-7b-hf"},
    {"label": "Mistral Instruct", "value": "mistralai/Mistral-7B-Instruct-v0.2"},
]

HEARTBEAT_INTERVAL = 10 # Seconds
HEARTBEAT_TIMEOUT = 20 # Seconds

def get_label_type_map(type_map, value):
    return [i["label"] for i in type_map if i["value"] == value][0]
