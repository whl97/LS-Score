
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize
import utils
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import difflib
from dataset_reader import CNNDM_Reader, Newsroom_Reader


def read_dataset(dataset_name, dataset_dir):
    def filter_test_data(human_score, articles, references):
        def func(x):
            return {k: v for k, v in x.items() if k not in human_score}
        if articles != None:
            articles = func(articles)
        if references != None:
            references = func(references)
        return articles, references
    if dataset_name == "cnndm":
        data = CNNDM_Reader.read_data()
    elif dataset_name == "newsroom":
        data = Newsroom_Reader.read_data()
    human_scores, articles, references = data["human_scores"], data["articles"], data["references"]
    articles, references = filter_test_data(human_scores, articles, references)
    return articles, references


class RankingLossDataset(Dataset):
    def __init__(self, dataset_name, dataset_dir, tokenizer, train_or_test):
        self.tokenizer = tokenizer
        self.articles, self.references = read_dataset(
            dataset_name, dataset_dir)

        self.neg_func_list = [self.delete_words,
                              self.disorder_words, self.add_sentences]

        directory = os.path.dirname(os.path.abspath(__file__))
        self.cached_file = os.path.join(
            directory, 'cache/{}/cached_ranking_loss_negative_sampling'.format(dataset_name))
        self.sample_rate = 1
        self.train_proportion = 0.90
        self.train_or_test = train_or_test

        self.ART_SUM_PAIR_LIST = self.get_feature_lists(
            self.cached_file, self.train_or_test, self.train_proportion)

    def get_feature_lists(self, cached_file, train_or_test, train_proportion=0.95):
        if os.path.exists(self.cached_file):
            with open(self.cached_file, 'rb') as handle:
                art_sum_pairs_list = pickle.load(handle)
        else:
            art_sum_pairs_list = self._process_dataset_and_save(
                self.articles, self.references, self.tokenizer, cached_file)

        num_samples = int(len(art_sum_pairs_list)*self.sample_rate)
        sample_index = random.sample(
            list(range(len(art_sum_pairs_list))), num_samples)
        random.shuffle(sample_index)
        art_sum_pairs_list = [art_sum_pairs_list[i] for i in sample_index]

        art_sum_pairs_list = self._split_train_and_test(
            art_sum_pairs_list, train_or_test, train_proportion)

        return art_sum_pairs_list

    def _split_train_and_test(self, feature_list, train_or_test, train_proportion):
        split = int(len(feature_list)*train_proportion)
        return feature_list[:split] if train_or_test == "train" else feature_list[split:]

    def _process_dataset_and_save(self, articles, references, tokenizer, cached_file):

        art_sum_pairs_list = []
        num_articles = len(articles)

        for idx, art_id in enumerate(articles):
            if art_id in references:
                article = articles[art_id]
                reference = references[art_id]
                if not len(sent_tokenize(article)) > len(sent_tokenize(reference)):
                    continue

                art_tokens = tokenizer.tokenize(article)
                art_input_ids, art_segment_ids, art_input_mask = self.get_features_from_tokens(
                    tokenizer, art_tokens, max_seq_length=512)
                art_features = {"input_ids": art_input_ids,
                                "segment_ids": art_segment_ids, "input_mask": art_input_mask}

                reference_tokens = tokenizer.tokenize(reference)
                reference_and_negative_sampling_tokens = [reference_tokens]
                reference_and_negative_sampling_tokens += [
                    func(article, reference, tokenizer) for func in self.neg_func_list]

                n_art_sum_pairs = []

                for sum_tokens in reference_and_negative_sampling_tokens:
                    sum_input_ids, sum_segment_ids, sum_input_mask = self.get_features_from_tokens(
                        tokenizer, sum_tokens, max_seq_length=512)
                    sum_features = {"input_ids": sum_input_ids,
                                    "segment_ids": sum_segment_ids, "input_mask": sum_input_mask}
                    art_sum_pair = {"article": art_features,
                                    "summary": sum_features}
                    n_art_sum_pairs.append(art_sum_pair)

                art_sum_pairs_list.append(n_art_sum_pairs)

        with open(cached_file, 'wb') as handle:
            pickle.dump(art_sum_pairs_list, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        return art_sum_pairs_list

    def get_features_from_tokens(self, tokenizer, tokens, max_seq_length):

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

    def __getitem__(self, index):
        return self.ART_SUM_PAIR_LIST[index]

    def __len__(self):
        return len(self.ART_SUM_PAIR_LIST)

    def delete_words(self, article, summary, tokenizer, probability=0.2):
        positive_tokens = tokenizer.tokenize(summary)
        num_words_to_delete = int(len(positive_tokens)*probability)
        index_list = list(range(0, len(positive_tokens)))
        sample_index_list = random.sample(index_list, num_words_to_delete)
        delete_index_list = []
        for i in range(num_words_to_delete):
            index = sample_index_list[i]
            delete_index_list.append(index)
            if utils.is_subtoken(positive_tokens[index]):
                delete_index_list.append(index-1)

        index_left = list(set(index_list)-set(delete_index_list))

        negative_tokens = []

        for index in index_left:
            negative_tokens.append(positive_tokens[index])

        return negative_tokens

    def add_sentences(self, article, summary, tokenizer, add_sent_num=1):

        def get_match_rate(str1, str2):
            return difflib.SequenceMatcher(None, str1, str2).quick_ratio()

        def find_most_similar_sentence_in_article(all_sentences_in_article, sent):
            most_similar_idx = 0
            max_match_rate = -1
            for idx, _sent in enumerate(all_sentences_in_article):
                match_rate = get_match_rate(sent, _sent)
                if match_rate > max_match_rate:
                    max_match_rate = match_rate
                    most_similar_idx = idx

            return all_sentences_in_article[most_similar_idx]

        sentences_in_article = sent_tokenize(article)
        sentences_in_summary = sent_tokenize(summary)

        for sent in sentences_in_summary:
            most_similar_sent = find_most_similar_sentence_in_article(
                sentences_in_article, sent)
            sentences_in_article.remove(most_similar_sent)
        for i in range(add_sent_num):
            insert_index = random.choice(range(len(sentences_in_summary)+1))
            insert_sent = random.choice(sentences_in_article)
            sentences_in_summary.insert(insert_index, insert_sent)

        return tokenizer.tokenize(''.join(sentences_in_summary))

    def disorder_words(self, article, summary, tokenizer, probability=0.2):
        all_word_list = []
        sentences = sent_tokenize(summary)
        for sent in sentences:
            word_list = self.disorder_one_sent(sent, probability)
            all_word_list += word_list

        tokens = tokenizer.tokenize(' '.join(all_word_list))
        return tokens

    def disorder_one_sent(self, sent, probability):
        punc = string.punctuation
        temp_sent = sent
        while(1):

            flag = True
            temp_sent = ""
            for i in range(len(sent)):
                if sent[i] in punc:

                    if i > 0 and sent[i-1] != ' ':
                        flag = False
                        temp_sent = sent[:i] + ' ' + sent[i]
                    else:
                        temp_sent = sent[:i+1]
                    if i < len(sent)-1 and sent[i+1] != ' ':
                        flag = False
                        temp_sent += ' ' + sent[i+1:]
                    elif i < len(sent)-1:
                        temp_sent += sent[i+1:]
                    break
            sent = temp_sent
            if flag:
                break

        word_list = sent.split(' ')
        num_words_to_disorder = int(len(word_list)*probability)

        def get_random_list(word_list, num_words):
            index_list = list(range(0, len(word_list)))

            while(1):
                a_index_list = random.sample(index_list, num_words)
                if any([word_list[a] not in punc for a in a_index_list]):
                    break
            while(1):
                b_index_list = random.sample(index_list, num_words)
                if any([word_list[b] not in punc for b in b_index_list]):
                    break

            return a_index_list, b_index_list

        a_index_list, b_index_list = get_random_list(
            word_list, num_words_to_disorder)

        for i in range(num_words_to_disorder):
            idx_a = a_index_list[i]
            idx_b = b_index_list[i]

            temp = word_list[idx_a]
            word_list[idx_a] = word_list[idx_b]
            word_list[idx_b] = temp

        return word_list
