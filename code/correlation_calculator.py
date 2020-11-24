from inference_model import *
import config as MY_CONFIG
import torch
from utils import *
import pickle
import argparse
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from dataset_reader import CNNDM_Reader, Newsroom_Reader


class CorrelationCalculator:

    def get_correlation(self, score1, score2):
        def delete_None_item(score):
            scores = {k: v for k, v in score.items() if None not in v}
            return scores

        score1 = delete_None_item(score1)
        score2 = delete_None_item(score2)
        commom_keys = list(set(score1.keys()) & set(score2.keys()))
        if len(commom_keys) < 10:
            return None
        else:
            score1 = self._get_scores_vector(score1, key_list=commom_keys)
            score2 = self._get_scores_vector(score2, key_list=commom_keys)
            return self._pearson_and_spearman(score1, score2)

    def _get_scores_vector(self, scores, key_list=None):
        if key_list != None:
            scores = self._get_scores_in_list(scores, key_list)
        scores = self._scores_to_vector(scores)
        return scores

    def _scores_to_vector(self, scores):
        scores_sorted = sorted(scores.items(), key=lambda x: x[0])
        score_values = [item[1] for item in scores_sorted]
        score_values = np.array(score_values).flatten()
        score_values.dtype = 'float64'
        return score_values

    def _get_scores_in_list(self, scores, id_list):
        scores = {k: v for k, v in scores.items() if k in id_list}
        return scores

    def _pearson_and_spearman(self, preds, labels):

        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return pearson_corr, spearman_corr


class ModelEvaluator:
    def __init__(self, dataset_name, model_save_dir, load_model_name, pretrained_model_dir, model_type=None, is_cuda=True):

        self.dataset_name = dataset_name
        self.load_model_name = load_model_name
        self.model_type = model_type
        self.pretrained_model_dir = pretrained_model_dir

        self.is_cuda = is_cuda
        self.model_save_dir = model_save_dir + \
            "/model_save_{}".format(self.dataset_name)
        self.score_save_dir = self.model_save_dir+"/model_scores"
        self.confirm_dir(self.score_save_dir)

        _model_type = self.load_model_name.split('.')[0]
        self.model_type = _model_type

        self.dataset_dir = MY_CONFIG.dataset_dir_dict[self.dataset_name]

        if self.dataset_name == "cnndm":
            dataset_reader = CNNDM_Reader(self.dataset_dir)
            self.data = dataset_reader.read_data()

        elif self.dataset_name == "newsroom":
            dataset_reader = Newsroom_Reader(self.dataset_dir)
            self.data = dataset_reader.read_data()

        self.model = self.init_model()

        self.corr_cal = CorrelationCalculator()

    def confirm_dir(self, dir):
        if os.path.exists(self.model_save_dir):
            if not os.path.exists(dir):
                os.mkdir(dir)

    def model_exits(self):
        model_full_path = self.model_save_dir + "/" + self.load_model_name
        if os.path.exists(self.model_save_dir) and os.path.exists(model_full_path):
            return True
        else:
            return False

    def init_model(self):
        if not self.model_exits():
            return None

        if self.model_type == "LS_Score":
            model = LS_ScoreAnalysis(dataset_name=self.dataset_name,
                                     pretrained_model_dir=self.pretrained_model_dir,
                                     load_model_name=self.load_model_name,
                                     is_cuda=self.is_cuda)

        return model

    def _cal_moverscore(self, references, summaries, n_gram):

        score_filename = 'other_metrics/moverscore/{}/ngram_{}_results'.format(
            self.dataset_name, n_gram)
        if os.path.exists(score_filename):
            with open(score_filename, 'rb') as handle:
                scores = pickle.load(handle)

        else:

            num_systems = len(list(summaries.values())[0])
            scores = []

            ref_list = []
            sum_list = []
            article_id_list = list(references.keys())
            for article_id in article_id_list:
                for i in range(num_systems):
                    ref_list.append(references[article_id])
                sum_list += summaries[article_id]

            total_ref_list = ref_list
            total_sum_list = sum_list
            num_samples = len(ref_list)
            num_times = 5
            temp = int(num_samples/num_times)

            for i in range(num_times-1):
                ref_list = total_ref_list[i*temp:(i+1)*temp]
                sum_list = total_sum_list[i*temp:(i+1)*temp]

                idf_dict_hyp = defaultdict(lambda: 1.)
                idf_dict_ref = defaultdict(lambda: 1.)
                torch.cuda.empty_cache()

                with torch.no_grad():
                    scores += word_mover_score(ref_list, sum_list, idf_dict_ref, idf_dict_hyp,
                                               stop_words=[], n_gram=n_gram, remove_subwords=True)

                torch.cuda.empty_cache()

            ref_list = total_ref_list[(num_times-1)*temp:]
            sum_list = total_sum_list[(num_times-1)*temp:]

            idf_dict_hyp = defaultdict(lambda: 1.)
            idf_dict_ref = defaultdict(lambda: 1.)
            torch.cuda.empty_cache()

            with torch.no_grad():
                scores += word_mover_score(ref_list, sum_list, idf_dict_ref, idf_dict_hyp,
                                           stop_words=[], n_gram=n_gram, remove_subwords=True)

            torch.cuda.empty_cache()

            scores = {article_id_list[i]: scores[i*num_systems:(
                i+1)*num_systems] for i in range(len(article_id_list))}
            with open(score_filename, 'wb') as handle:
                pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return scores

    def _cal_sms(self, references, summaries, dataset_name, embedding_type, metric_type):
        # embedding_type: glove or elmo
        # metric_type: sms wms s+wms
        num_systems = len(list(summaries.values())[0])  # 7

        scores = {}

        score_filename = "other_metrics/sms/{}_ref_hyp_{}_{}.out".format(
            dataset_name, embedding_type, metric_type)  # todo

        if os.path.exists(score_filename):
            with open(score_filename, 'rb') as handle:
                score_list = pickle.load(handle)

            id_list = list(references.keys())
            id_list.sort()

            scores = {
                id_list[i]: score_list[i*num_systems:((i+1)*num_systems)] for i in range(len(id_list))}

        else:
            return None
        return scores

    def _cal_rouge_rpf(self, references, summaries, metric_name="ROUGE-1", rpf_flag='r'):
        num_systems = len(list(summaries.values())[0])
        scores = {}

        ref_list = []
        sum_list = []
        article_id_list = list(references.keys())
        for article_id in article_id_list:
            for i in range(num_systems):
                ref_list.append(references[article_id])
            sum_list += summaries[article_id]

        if "rouge" in metric_name.lower():
            rouge = Rouge()
            scores = rouge.get_scores(sum_list, ref_list)
            scores = [score[metric_name.lower()][rpf_flag] for score in scores]
            scores = {article_id_list[i]: scores[i*num_systems:(
                i+1)*num_systems] for i in range(len(article_id_list))}

        return scores

    def _cal_bleu_and_meteor(self, summaries, references, metric_name):
        num_systems = len(list(summaries.values())[0])

        scores = []

        ref_list = []
        sum_list = []
        article_id_list = list(references.keys())
        for article_id in article_id_list:
            for i in range(num_systems):
                ref_list.append(references[article_id])
            sum_list += summaries[article_id]

        if "bleu" in metric_name:
            n = int(metric_name[-1])
            for hyp, ref in zip(sum_list, ref_list):
                scores.append(acteval.bleu(hyp, [ref], n=n, smooth=True))

        elif metric_name == 'meteor':
            meteor_file = self.dataset_dir+"meteor_results.pkl"
            if os.path.exists(meteor_file) and os.path.getsize(meteor_file):
                with open(meteor_file, 'rb') as handle:
                    scores = pickle.load(handle)
            else:
                for hyp, ref in zip(sum_list, ref_list):
                    score = acteval.meteor(hyp, ref)
                    scores.append(score)

                with open(meteor_file, 'wb') as handle:
                    pickle.dump(scores, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

        num_of_articles = len(article_id_list)
        scores = {article_id_list[i]: scores[i*num_systems: (
            i+1)*num_systems] for i in range(num_of_articles)}

        return scores

    def _get_bertscore_Single_metric(self, rpf_flag='r'):
        rpf_index = "rpf".index(rpf_flag)
        filename = self.dataset_dir+'bertscore_results'
        if os.path.exists(filename) and os.path.getsize(filename):
            with open(filename, 'rb') as handle:
                bertscore_results = pickle.load(handle)

        single_bertscore_results = {}
        for id_key in bertscore_results:
            temp = bertscore_results[id_key]
            temp = [item[rpf_index] for item in temp]
            single_bertscore_results[id_key] = temp
        return single_bertscore_results

    def cal_our_scores(self, summaries, articles, references, load_from_file=True, save_to_file=True):
        filename = self.score_save_dir + \
            "/our_scores_{}.pkl".format(self.load_model_name)

        if load_from_file and os.path.exists(filename):
            with open(filename, 'rb') as handle:
                our_scores = pickle.load(handle)

        else:

            our_scores = {"ls_score": {}}
            for art_id, n_summary in summaries.items():
                article = articles[art_id]
                reference = references[art_id]
                for summary in n_summary:
                    scores = self.model(article, summary)
                    for _score_name, _score in scores.items():
                        if art_id not in our_scores[_score_name]:
                            our_scores[_score_name][art_id] = [_score]
                        else:
                            our_scores[_score_name][art_id].append(_score)
                        print(_score_name, _score)
            if save_to_file:
                with open(filename, 'wb') as handle:
                    pickle.dump(our_scores, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

        return our_scores

    def _write_to_txt(self, str):
        filename = "results.txt"
        with open(filename, 'a') as f:
            f.write(str+'\n')

    def get_results(self):
        human_scores = self.data["human_scores"]
        summaries = self.data["summaries"]
        articles = self.data["articles"]
        references = self.data["references"]
        our_scores_correlations = {}
        three_our_scores = self.cal_our_scores(summaries, articles, references)
        for score_name, scores in three_our_scores.items():
            our_scores_correlations[score_name] = {}
            for prompt in human_scores:
                if prompt == "hter":
                    continue

                corr = self.corr_cal.get_correlation(
                    scores, human_scores[prompt])
                our_scores_correlations[score_name][prompt] = corr
        return our_scores_correlations

    def display_results(self, display_all_metrics=True, write_to_txt=True):
        my_print = print if write_to_txt == False else self._write_to_txt
        my_print("%"*100)
        my_print("file path ：{}".format(MY_CONFIG.output_dir))
        my_print("dataset   ：{}".format(self.dataset_name))
        my_print("model type：{}".format(self.model_type))
        my_print("model name：{}".format(self.load_model_name))
        my_print("-"*100)

        human_scores = self.data["human_scores"]
        references = self.data["references"]
        summaries = self.data["summaries"]
        articles = self.data["articles"]

        metric_list = ["moverscore"]

        n_our_scores = self.cal_our_scores(summaries, articles, references)

        for prompt in human_scores:
            if prompt == "hter":
                continue
            my_print('-'*30 + prompt + '-'*30)
            for score_name, scores in n_our_scores.items():
                corr = self.corr_cal.get_correlation(
                    scores, human_scores[prompt])
                if corr != None:
                    my_print("Our_{}:  {}".format(score_name, corr))

            if display_all_metrics:
                metric_list = ["rouge-1", "rouge-2", "rouge-l",
                               "bleu-1", "bleu-2", "bleu-3", "bleu-4",
                               "meteor", "bert-score", "sms", "moverscore"]
                for metric_name in metric_list:
                    if "rouge" in metric_name:
                        for rpf_flag in "rpf":
                            scores = self._cal_rouge_rpf(references, summaries,
                                                         metric_name=metric_name, rpf_flag=rpf_flag)
                            corr = self.corr_cal.get_correlation(
                                scores, human_scores[prompt])
                            my_print(
                                "{}-{}:  {}".format(metric_name, rpf_flag, corr))

                    elif "bleu" in metric_name or "meteor" == metric_name:
                        scores = self._cal_bleu_and_meteor(
                            summaries, references, metric_name)
                        corr = self.corr_cal.get_correlation(
                            scores, human_scores[prompt])
                        my_print("{}:  {}".format(metric_name, corr))

                    elif metric_name == "bert-score":
                        for rpf_flag in "rpf":
                            scores = self._get_bertscore_Single_metric(
                                rpf_flag=rpf_flag)
                            corr = self.corr_cal.get_correlation(
                                scores, human_scores[prompt])
                            my_print(
                                "{}-{}:  {}".format(metric_name, rpf_flag, corr))

                    elif metric_name == "sms":
                        for embedding_type in ['glove', 'elmo']:
                            for metric_type in ['sms', 'wms', 's+wms']:
                                scores = self._cal_sms(references, summaries,
                                                       self.dataset_name, embedding_type, metric_type)
                                if scores != None:
                                    corr = self.corr_cal.get_correlation(
                                        scores, human_scores[prompt])
                                    my_print(
                                        "{}-{}:  {}".format(metric_type, embedding_type, corr))

                    elif metric_name == "moverscore":
                        for n_gram in [1, 2]:
                            scores = self._cal_moverscore(
                                references, summaries, n_gram)
                            if scores != None:
                                corr = self.corr_cal.get_correlation(
                                    scores, human_scores[prompt])
                                my_print(
                                    "{}-{}:  {}".format("moverscore", n_gram, corr))

        my_print("%"*100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device_num",
        default='1',
        type=str,
        required=False,
        help="which gpu to use",
    )

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    def list_all_model(dataset_name):
        model_save_dir = MY_CONFIG.output_dir + \
            "/model_save_{}".format(dataset_name)
        model_list = []
        dic_lis = [i for i in os.listdir(model_save_dir)]
        if len(dic_lis) == 0:
            raise FileNotFoundError(
                "can not find any state dict in {}!".format(model_save_dir))
        dic_lis = [i for i in dic_lis if "model.epoch" in i]
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))

        return dic_lis

    dataset_list = ["cnndm", "newsroom"]

    for dataset_name in dataset_list:
        model_list = list_all_model(dataset_name)

        for model_name in model_list:
            print('-'*100)
            print("dataset_name:{}			 model_name:{}".format(
                dataset_name, model_name))
            evaluator = ModelEvaluator(dataset_name=dataset_name,
                                       model_save_dir=MY_CONFIG.output_dir,
                                       load_model_name=model_name,
                                       pretrained_model_dir=MY_CONFIG.pretrained_model_dir,
                                       is_cuda=True)

            if evaluator.model_exits():
                evaluator.display_results(
                    display_all_metrics=False, write_to_txt=False)

            else:
                print("This model doesn't exists.")
            print('-'*100)
