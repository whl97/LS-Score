from transformers.modeling_bert import BertForPreTraining, BertPreTrainedModel, BertLayerNorm, BertPreTrainingHeads, BertLMPredictionHead
from transformers import BertModel, BertTokenizer, BertConfig
import torch
from utils import *
from Models import *
import config as MY_CONFIG
import torch.nn.functional as F
import os
from Datasets import read_dataset, Features

class Features(object):
	def __init__(self, input_ids=None, segment_ids=None, input_mask=None, mlm_labels=None, is_next_label=None):
		
		self.input_ids = self.toTensor(input_ids)
		self.segment_ids = self.toTensor(segment_ids)
		self.input_mask = self.toTensor(input_mask)
		self.mlm_labels = self.toTensor(mlm_labels)
		self.is_next_label = self.toTensor(is_next_label)

	def toTensor(self, feat):
		if feat!=None:
			return torch.tensor(feat)
		return None


class Analysis:
	def __init__(self, metric_type, dataset_name, pretrained_model_dir, load_model_name=None, is_cuda=True):
		self.dataset_name = dataset_name
		self.load_model_name = load_model_name  
		self.pretrained_model_dir = pretrained_model_dir
		self.metric_type = metric_type

		self.config = BertConfig.from_pretrained(pretrained_model_dir)
		self.vocab_size = self.config.vocab_size
		self.device = torch.device("cuda" if is_cuda else "cpu")
		self.max_seq_length = 512
		self.is_cuda = is_cuda
		self.model_save_dir = MY_CONFIG.output_dir + "/model_save_{}".format(self.dataset_name)
		self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_dir, do_lower_case=True)

		self.bert_model = self.load_model(self.model_save_dir)
		self.bert_model.to(self.device)
		self.bert_model.eval()


	def load_model(self, dir_path):
		raise NotImplementedError()


class LS_ScoreAnalysis(Analysis):
	def load_model(self, dir_path):
		model = BertForLS_Score(self.config, self.is_cuda)
		assert self.load_model_name!=None
		checkpoint_dir = dir_path+"/"+self.load_model_name
		checkpoint = torch.load(checkpoint_dir)
		model.load_state_dict(checkpoint["model_state_dict"], strict=False)
		torch.cuda.empty_cache()
		return model

	def get_features(self, tokenizer, text):
		tokens = self.tokenizer.tokenize(text)
		input_ids, segment_ids, input_mask = get_features_from_tokens(tokenizer, tokens, max_seq_length=512)
		features = {"input_ids":input_ids, "segment_ids":segment_ids, "input_mask":input_mask}	
		return features

	def get_art_sum_pair(self, article, summary):
		art_features = self.get_features(self.tokenizer, article)
		sum_features = self.get_features(self.tokenizer, summary)
		art_sum_pair = {"article":art_features, "summary":sum_features}
		return art_sum_pair

	def __call__(self, article, summary):

		art_sum_pair = self.get_art_sum_pair(article, summary)
		score, _ = self.bert_model([art_sum_pair])
		return {"ls_score":score[0].item()}







