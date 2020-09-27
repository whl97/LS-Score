import jsonlines
from nltk import sent_tokenize
from scipy.stats import pearsonr, spearmanr
import torch
import random
import numpy as np
import math
from CnndmLoader import *
from NewsroomLoader import *



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


def pearson_and_spearman(preds, labels):
	pearson_corr = pearsonr(preds, labels)[0]
	spearman_corr = spearmanr(preds, labels)[0]
	return pearson_corr,spearman_corr


def convert_text_to_inputs_for_masked_LM(text, tokenizer, block_size):
	input_ids = []
	tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
	for i in range(0, len(tokenized_text)-block_size+3, block_size-2): 
		input_ids_temp = tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+block_size-2])
		input_ids.append(input_ids_temp)
	return input_ids


def convert_artical_to_inputs_for_masked_LM(article, tokenizer, block_size):
	input_ids = []
	tokenized_article = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(article))
	for i in range(0, len(tokenized_article)-block_size+3, block_size-2):
		input_ids_temp = tokenizer.build_inputs_with_special_tokens(tokenized_article[i:i+block_size-2])
		input_ids.append(input_ids_temp)
	return input_ids


def get_sentence_pairs_from_summary(summary, tokenizer, isNeighbor, max_seq_length, non_neighbor_interval=2):
	sentences_in_summary = sent_tokenize(summary)
	num_sentences = len(sentences_in_summary)
	sentence_pairs = []

	if isNeighbor:
		sent_interval = 1
	else:
		if num_sentences<=non_neighbor_interval:
			return []
		sent_interval = non_neighbor_interval

	for i in range(num_sentences-sent_interval):
		tokens_a = tokenizer.tokenize(sentences_in_summary[i])
		tokens_b = tokenizer.tokenize(sentences_in_summary[i+sent_interval])
		input_ids, input_mask, segment_ids = get_features_for_sentence_pair(tokenizer, tokens_a, tokens_b, max_seq_length)
		sentence_pairs.append([input_ids, segment_ids, input_mask])

	return sentence_pairs


def get_features_for_sentence_pair(tokenizer, tokens_a, tokens_b, max_seq_length):
	truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
	tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
	segment_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b) + 1)
	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	input_mask = [1] * len(input_ids)
	padding = [0] * (max_seq_length - len(input_ids))
	input_ids += padding
	input_mask += padding
	segment_ids += padding
	return input_ids, input_mask, segment_ids



def truncate_seq_pair(tokens_a, tokens_b, max_length):
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()


