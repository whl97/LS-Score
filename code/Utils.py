import jsonlines
from nltk import sent_tokenize
import torch
import random
import numpy as np
import math


def is_subtoken(x):
    return x.startswith("##")


def get_features_from_tokens(tokenizer, tokens, max_seq_length):
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]

    input_tokens = []
    segment_ids = []
    input_tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens:
        input_tokens.append(token)
        segment_ids.append(0)
    input_tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return input_ids, segment_ids, input_mask


class GELU(torch.nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
