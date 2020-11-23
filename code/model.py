from transformers.modeling_bert import BertForPreTraining, BertPreTrainedModel, BertPreTrainingHeads, BertForMaskedLM
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import GELU


class BertForLS_Score(BertPreTrainedModel):
    def __init__(self, metric_type, config, is_cuda=True):
        super(BertForLS_Score, self).__init__(config)
        self.metric_type = metric_type
        self.config = config
        self.is_cuda = is_cuda
        self.device = torch.device("cuda" if self.is_cuda else "cpu")
        self.bert = BertModel(self.config)

        self.decoder_l = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            GELU(),
            nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        )

        self.max_seq_length = 512
        self.ones = torch.eye(config.hidden_size)
        self.init_weights()

    def L_Score(self, input_ids, input_mask, sum_seq_output):
        score = self.decoder_l(sum_seq_output).unsqueeze(0)
        score = F.log_softmax(score, dim=2)

        temp = torch.zeros(self.max_seq_length, self.config.vocab_size)
        if self.cuda:
            temp = temp.cuda()
        one_hot_input_ids = temp.scatter_(1, input_ids.view(-1, 1), 1).float()
        score = torch.sum(score.mul(one_hot_input_ids)) / \
            (torch.sum(input_mask).float())
        score = (score+200)/100
        return score

    def S_Score(self, art_seq_output, sum_seq_output):
        art_embeddings = art_seq_output[0]
        sum_embeddings = sum_seq_output[0]
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        score_for_summary = cos(art_embeddings, sum_embeddings)
        return score_for_summary

    def forward(self, art_sum_pairs):
        num_samples = len(art_sum_pairs)
        summaries = [pair["summary"] for pair in art_sum_pairs]
        article = art_sum_pairs[0]["article"]

        article_input_ids = article["input_ids"]
        article_segment_ids = article["segment_ids"]
        article_input_mask = article["input_mask"]

        summaries_input_ids = [s["input_ids"] for s in summaries]
        summaries_segment_ids = [s["segment_ids"] for s in summaries]
        summaries_input_mask = [s["input_mask"] for s in summaries]

        art_sum_input_ids = torch.tensor(
            [article_input_ids]+summaries_input_ids).to(self.device)
        art_sum_segment_ids = torch.tensor(
            [article_segment_ids]+summaries_segment_ids).to(self.device)
        art_sum_input_mask = torch.tensor(
            [article_input_mask]+summaries_input_mask).to(self.device)

        art_sum_sequence_output, _ = self.bert(input_ids=art_sum_input_ids,
                                               token_type_ids=art_sum_segment_ids,
                                               attention_mask=art_sum_input_mask)
        art_seq_output = art_sum_sequence_output[0]
        sums_seq_output = art_sum_sequence_output[1:]

        l_scores = torch.zeros(num_samples).to(self.device)
        s_scores = torch.zeros(num_samples).to(self.device)

        for i in range(num_samples):
            sum_i_input_ids = art_sum_input_ids[i+1]
            sum_i_segment_ids = art_sum_segment_ids[i+1]
            sum_i_input_mask = art_sum_input_mask[i+1]
            sum_seq_output = art_sum_sequence_output[i+1]
            l_scores[i] = self.L_Score(
                sum_i_input_ids, sum_i_input_mask, sum_seq_output)
            s_scores[i] = self.S_Score(art_seq_output, sum_seq_output)

        if num_samples > 1:
            m_minus_d = 1-(l_scores[0]-l_scores[[1, 2]])
            l_loss = torch.sum(m_minus_d[m_minus_d > 0])
            m_minus_d = 1-(s_scores[0]-s_scores[[1, 3]])
            s_loss = torch.sum(m_minus_d[m_minus_d > 0])
        else:
            l_loss = 0
            s_loss = 0

        loss = l_loss + s_loss
        scores = l_scores + s_scores

        return scores, loss





class BertForLS_Score(BertPreTrainedModel):
    def __init__(self, metric_type, config, is_cuda=True):
        super(BertForLS_Score, self).__init__(config)
        self.metric_type = metric_type
        self.config = config
        self.is_cuda = is_cuda
        self.device = torch.device("cuda" if self.is_cuda else "cpu")
        self.bert = BertModel(self.config)

        self.decoder_l = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            GELU(),
            nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        )

        self.max_seq_length = 512
        self.ones = torch.eye(config.hidden_size)
        self.init_weights()

    def L_Score(self, input_ids, input_mask, sum_seq_output):
        score = self.decoder_l(sum_seq_output).unsqueeze(0)
        score = F.log_softmax(score, dim=2)

        temp = torch.zeros(self.max_seq_length, self.config.vocab_size)
        if self.cuda:
            temp = temp.cuda()
        one_hot_input_ids = temp.scatter_(1, input_ids.view(-1, 1), 1).float()
        score = torch.sum(score.mul(one_hot_input_ids)) / \
            (torch.sum(input_mask).float())
        score = (score+200)/100
        return score

    def S_Score(self, art_seq_output, sum_seq_output):
        art_embeddings = art_seq_output[0]
        sum_embeddings = sum_seq_output[0]
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        score_for_summary = cos(art_embeddings, sum_embeddings)
        return score_for_summary

    def forward(self, art_sum_pairs):
        num_samples = len(art_sum_pairs)
        summaries = [pair["summary"] for pair in art_sum_pairs]
        article = art_sum_pairs[0]["article"]

        article_input_ids = article["input_ids"]
        article_segment_ids = article["segment_ids"]
        article_input_mask = article["input_mask"]

        summaries_input_ids = [s["input_ids"] for s in summaries]
        summaries_segment_ids = [s["segment_ids"] for s in summaries]
        summaries_input_mask = [s["input_mask"] for s in summaries]

        art_sum_input_ids = torch.tensor(
            [article_input_ids]+summaries_input_ids).to(self.device)
        art_sum_segment_ids = torch.tensor(
            [article_segment_ids]+summaries_segment_ids).to(self.device)
        art_sum_input_mask = torch.tensor(
            [article_input_mask]+summaries_input_mask).to(self.device)

        art_sum_sequence_output, _ = self.bert(input_ids=art_sum_input_ids,
                                               token_type_ids=art_sum_segment_ids,
                                               attention_mask=art_sum_input_mask)
        art_seq_output = art_sum_sequence_output[0]
        sums_seq_output = art_sum_sequence_output[1:]

        l_scores = torch.zeros(num_samples).to(self.device)
        s_scores = torch.zeros(num_samples).to(self.device)

        for i in range(num_samples):
            sum_i_input_ids = art_sum_input_ids[i+1]
            sum_i_segment_ids = art_sum_segment_ids[i+1]
            sum_i_input_mask = art_sum_input_mask[i+1]
            sum_seq_output = art_sum_sequence_output[i+1]
            l_scores[i] = self.L_Score(
                sum_i_input_ids, sum_i_input_mask, sum_seq_output)
            s_scores[i] = self.S_Score(art_seq_output, sum_seq_output)

        if num_samples > 1:
            m_minus_d = 1-(l_scores[0]-l_scores[[1, 2]])
            l_loss = torch.sum(m_minus_d[m_minus_d > 0])
            m_minus_d = 1-(s_scores[0]-s_scores[[1, 3]])
            s_loss = torch.sum(m_minus_d[m_minus_d > 0])
        else:
            l_loss = 0
            s_loss = 0

        loss = l_loss + s_loss
        scores = l_scores + s_scores

        return scores, loss
