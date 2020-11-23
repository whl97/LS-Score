import jsonlines
import csv
import os
import pickle
from nltk.translate.bleu_score import corpus_bleu
import time


class Reader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def read_human_score(self, human_score_file):
        raise NotImplementedError()

    def read_articles(self, articles_file, human_scores):
        raise NotImplementedError()

    def read_data(self):
        raise NotImplementedError()

    def read_summaries_and_reference(self, *argw):
        raise NotImplementedError()


class CNNDM_Reader(Reader):

    def read_human_score(self, human_score_file):

        _human_scores = {}
        system_types = ['reference', 'ml', 'ml+rl', 'seq2seq', 'pointer']
        with open(human_score_file, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                if item['id'] not in _human_scores:
                    _human_scores[item['id']] = [None]*5

                system_idx = system_types.index(item['system'])
                _human_scores[item['id']][system_idx] = [item['prompts']['hter']['gold'],
                                                         item['prompts']['overall']['gold'],
                                                         item['prompts']['grammar']['gold'],
                                                         item['prompts']['redundancy']['gold']]

        prompt_list = ['hter', 'overall', 'grammer', 'redundancy']

        human_scores = {}
        for id_key in _human_scores:
            if None not in _human_scores[id_key]:
                human_scores[id_key] = _human_scores[id_key]

        num_summaries_per_article = len(list(human_scores.values())[0])
        human_scores_different_prompt = {}
        for prompt in prompt_list:
            prompt_index = prompt_list.index(prompt)

            human_scores_different_prompt[prompt] = {}

            for id_key in human_scores:
                temp = [scores_for_one_summary[prompt_index]
                        for scores_for_one_summary in human_scores[id_key]]
                temp.pop(0)
                human_scores_different_prompt[prompt][id_key] = temp

        return human_scores_different_prompt

    def read_articles(self, articles_file, human_scores):
        _articles = {}
        with open(articles_file, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                _articles[item['id']] = item['text']

        a_prompt = list(human_scores.keys())[0]
        human_scores = human_scores[a_prompt]

        articles = {id_key: _articles[id_key]
                    for id_key in _articles if id_key in human_scores}
        return articles

    def read_summaries_and_reference(self, summaries_file, human_scores):
        system_types = ['reference', 'ml', 'ml+rl', 'seq2seq', 'pointer']

        _summaries = {}
        with open(summaries_file, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                if item['id'] not in _summaries:
                    _summaries[item['id']] = [None]*5
                system_idx = system_types.index(item['system'])
                _summaries[item['id']][system_idx] = item['text']
        _summaries = dict((key, value)
                          for key, value in _summaries.items() if all(value) == True)

        a_prompt = list(human_scores.keys())[0]
        human_scores = human_scores[a_prompt]

        summaries = {}
        references = {}

        for id_key in _summaries:
            if id_key in human_scores:
                summaries[id_key] = _summaries[id_key][1:]
                references[id_key] = _summaries[id_key][0]
        return summaries, references

    def read_data(self):

        summaries_file = self.dataset_dir + 'summary.jsonl'
        human_score_file = self.dataset_dir + 'human_score.jsonl'
        articles_file = self.dataset_dir + 'articles.jsonl'

        human_scores = self._read_human_score(human_score_file)
        summaries, references = self._read_summaries_and_reference(
            summaries_file, human_scores)
        articles = self._read_articles(articles_file, human_scores)

        data = {'human_scores': human_scores,
                'summaries': summaries,
                'references': references,
                'articles': articles}

        return data


class Newsroom_Reader(Reader):

    def _get_human_scores_prompt(self, human_scores, prompt='coherence'):
        prompt_list = ['coherence', 'fluency', 'informativeness', 'relevent']
        prompt_index = prompt_list.index(prompt)
        human_scores_prompt = {}
        for id_key in human_scores:
            temp = [scores_for_one_summary[prompt_index]
                    for scores_for_one_summary in human_scores[id_key]]
            human_scores_prompt[id_key] = temp
        return human_scores_prompt

    def read_human_score(self, human_score_file):
        human_scores, _, _ = read_newsroom_human_score(human_score_file)
        prompt_list = ['coherence', 'fluency', 'informativeness', 'relevent']

        human_scores_different_prompt = {}
        for prompt in prompt_list:
            human_scores_different_prompt[prompt] = self.__get_human_scores_prompt(
                human_scores, prompt=prompt)
        return human_scores_different_prompt

    def read_articles(self, articles_file, human_scores):
        if os.path.exists(articles_file):
            with open(articles_file, 'rb') as handle:
                _articles = pickle.load(handle)

        a_prompt = list(human_scores.keys())[0]
        human_scores = human_scores[a_prompt]
        articles = {key: art for key, art in _articles.items()
                    if key in human_scores}
        return articles

    def read_summaries_and_reference(self, references_file, human_score_file, human_scores):

        with open(references_file, 'rb') as handle:
            references = pickle.load(handle)

        a_prompt = list(human_scores.keys())[0]
        human_scores = human_scores[a_prompt]
        _, _summaries, _ = read_newsroom_human_score(human_score_file)
        summaries = {}
        for id_key in _summaries:
            if id_key in human_scores:
                summaries[id_key] = _summaries[id_key]
        return summaries, references

    def read_data(self):
        human_score_file = self.dataset_dir + 'human-eval.csv'
        references_file = self.dataset_dir + "references_of_human_scores"
        articles_file = self.dataset_dir + 'articles_with_id'

        human_scores = self._read_human_score(human_score_file)
        summaries, references = self._read_summaries_and_reference(
            references_file, human_score_file, human_scores)
        articles = self._read_articles(articles_file, human_scores)

        data = {'human_scores': human_scores,
                'summaries': summaries,
                'references': references,
                'articles': articles}
        return data
