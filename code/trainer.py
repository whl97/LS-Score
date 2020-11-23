import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm, trange
import io
import os
import math
import numpy as np
import jsonlines
import argparse
import random
import shutil
import pickle
import config as MY_CONFIG
import pandas as pd
from models import *
from utils import *
from datasets import RankingLossDataset as DATASET


class Trainer:
    def __init__(self, args, pretrained_model_dir, dataset_dir, is_cuda=True):

        self.dataset_dir = dataset_dir
        self.dataset_name = args.dataset
        self.lr = args.lr
        self.pretrained_model_dir = pretrained_model_dir

        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_dir, do_lower_case=True)
        self.config = BertConfig.from_pretrained(pretrained_model_dir)
        self.max_seq_length = 512
        self.config.hidden_dropout_prob = 0.4
        self.bert_model = BertForLS_Score(config=self.config,
                                          is_cuda=True)

        self.model_save_name = "LS_Score"

        self.device = torch.device("cuda" if is_cuda else "cpu")
        self.bert_model.to(self.device)

        self.model_save_dir = args.output_dir + \
            "/model_save_{}".format(self.dataset_name)
        self.log_file = self.model_save_dir + "/df_log.pkl"

        self.train_dataloader, self.test_dataloader = self.get_dataLoader(
            args, self.dataset_dir, self.tokenizer)
        self.init_optimizer(lr=args.lr)

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

    def init_optimizer(self, lr):
        optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(
            optim_parameters, lr=lr, weight_decay=1e-3)

    def get_dataLoader(self, args, dataset_dir, tokenizer):
        train_dataloader, _ = self._get_train_or_test_dataLoader(
            args, dataset_dir, tokenizer, "train")
        test_dataloader, _ = self._get_train_or_test_dataLoader(
            args, dataset_dir, tokenizer, "test")
        return train_dataloader, test_dataloader

    def _get_train_or_test_dataLoader(self, args, dataset_dir, tokenizer, train_or_test):
        dataset = DATASET(args.dataset, dataset_dir, tokenizer, train_or_test)
        data_sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=1)
        return dataloader, dataset.__len__()

    def load_model(self, dir_path,  load_pretrained_model, load_base_bert):
        if load_base_bert:
            self.bert_model = MODEL.from_pretrained(self.pretrained_model_dir)
        else:
            checkpoint_dir = self.find_most_recent_state_dict(
                dir_path, load_pretrained_model)
            if checkpoint_dir == None:
                exit()
            else:
                checkpoint = torch.load(checkpoint_dir)
                self.bert_model.load_state_dict(
                    checkpoint["model_state_dict"], strict=False)
        torch.cuda.empty_cache()
        self.bert_model.to(self.device)

    def train(self, epoch):
        torch.cuda.empty_cache()
        self.bert_model.train()
        self.iteration(epoch, self.train_dataloader, train=True)

    def test(self, epoch):
        torch.cuda.empty_cache()
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, data_loader, train=True):
        df_path = self.log_file
        if not os.path.isfile(df_path):
            df = pd.DataFrame(columns=["epoch", "train_loss", "train_ref_score_max_rate",
                                       "test_loss", "test_ref_score_max_rate"
                                       ])
            df.to_pickle(df_path)

        str_code = "train" if train else "test"
        data_iter = tqdm(enumerate(data_loader),
                         desc="EP_%s:%d" % (str_code, epoch),
                         total=len(data_loader),
                         bar_format="{l_bar}{r_bar}")

        total_loss = 0

        count_reference_highest_score = 0
        reference_compare_to_max_score = 0
        if train:
            self.bert_model.train()

        loss_save_file = self.model_save_dir + "/loss.pkl"
        if os.path.exists(loss_save_file) and not(epoch == 0 and train):
            with open(loss_save_file, 'rb') as handle:
                all_loss_record = pickle.load(handle)

        else:

            all_loss_record = {"train_loss": [], "test_loss": []}

        train_loss_record = all_loss_record["train_loss"]
        test_loss_record = all_loss_record["test_loss"]

        for i, batch in data_iter:

            score, loss = self.bert_model.forward(batch)
            score = score.detach().cpu().numpy().reshape(-1).tolist()

            mmax = max(score)
            index = score.index(mmax)
            if index == 0:
                count_reference_highest_score += 1

            if i == 0:
                ref_max_rate = count_reference_highest_score
            else:
                ref_max_rate = count_reference_highest_score/i

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            if train:
                train_loss_record.append(loss.item())
            else:
                test_loss_record.append(loss.item())

            if train:
                log_dic = {
                    "epoch": epoch,
                    "train_loss": total_loss/(i+1),
                    "train_ref_score_max_rate": ref_max_rate,
                    "test_loss": 0,
                    "test_ref_score_max_rate": 0
                }

            else:
                log_dic = {
                    "epoch": epoch,
                    "train_loss": 0,
                    "train_ref_score_max_rate": 0,
                    "test_loss": total_loss/(i+1),
                    "test_ref_score_max_rate": ref_max_rate
                }

            if i % 10 == 0:
                data_iter.write(
                    str({k: v for k, v in log_dic.items() if v != 0}))

            if i % 10 == 0:
                all_loss_record = {
                    "train_loss": train_loss_record,
                    "test_loss": test_loss_record
                }
                with open(loss_save_file, 'wb') as handle:
                    pickle.dump(all_loss_record, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

        if train:
            df = pd.read_pickle(df_path)
            df = df.append([log_dic])
            df.reset_index(inplace=True, drop=True)
            df.to_pickle(df_path)
        else:
            log_dic = {k: v for k, v in log_dic.items() if v !=
                       0 and k != "epoch"}
            df = pd.read_pickle(df_path)
            df.reset_index(inplace=True, drop=True)
            for k, v in log_dic.items():
                df.at[epoch, k] = v
            df.to_pickle(df_path)
            return ref_max_rate

        all_loss_record = {"train_loss": train_loss_record,
                           "test_loss": test_loss_record}
        with open(loss_save_file, 'wb') as handle:
            pickle.dump(all_loss_record, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def find_most_recent_state_dict(self, dir_path, load_pretrained_model):
        dic_lis = [i for i in os.listdir(dir_path)]
        dic_lis = [i for i in dic_lis if "model" in i]
        if len(dic_lis) == 0:
            return None

        if load_pretrained_model:
            dic_lis = [i for i in dic_lis if "bert" in i]
        else:
            dic_lis = [i for i in dic_lis if self.model_save_name in i]

        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]

    def save_state_dict(self, epoch, state_dict_dir, file_path=self.model_save_name+".model"):
        if not os.path.exists(state_dict_dir):
            os.mkdir(state_dict_dir)
        save_path = state_dict_dir + "/" + \
            file_path + ".epoch.{}".format(str(epoch))
        self.bert_model.to("cpu")
        torch.save(
            {"model_state_dict": self.bert_model.state_dict()}, save_path)
        self.bert_model.to(self.device)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",   # cnndm or newsroom
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int
    )

    parser.add_argument(
        "--lr",
        default=1e-7,
        type=float
    )

    parser.add_argument(
        "--device_num",
        default='-1',
        type=str,
        required=False

    )

    parser.add_argument(
        "--num_train_epochs",
        default=5,
        type=int
    )

    parser.add_argument(
        "--seed",
        default=40,
        type=int
    )

    args = parser.parse_args()
    utils.set_seed(args)

    model_save_name = "LS_Score"

    if args.device_num != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    def init_Trainer():
        trainer = Trainer(args, args.pretrained_model_dir,
                          MY_CONFIG.dataset_dir_dict, is_cuda=True)
        dynamic_lr = args.lr
        return trainer, dynamic_lr

    start_epoch = 0  # from scratch
    train_epoches = args.num_train_epochs

    Trainer, dynamic_lr = init_Trainer()

    threshold = 999
    patient = 3
    all_ref_max_rate = []
    for epoch in range(start_epoch, start_epoch + train_epoches):

        if epoch == start_epoch and epoch == 0:
            Trainer.load_model(dir_path=Trainer.model_save_dir,
                               load_pretrained_model=False,
                               load_base_bert=True)
            Trainer.save_state_dict(epoch=-1,
                                    state_dict_dir=Trainer.model_save_dir,
                                    file_path="{}.model".format(model_save_name))

            Trainer.init_optimizer(args.lr)

        elif epoch == start_epoch:
            Trainer.load_model(dir_path=Trainer.model_save_dir,
                               load_pretrained_model=False,
                               load_base_bert=False)
            Trainer.init_optimizer(args.lr) 

        Trainer.train(epoch)
        Trainer.save_state_dict(epoch,
                                state_dict_dir=Trainer.model_save_dir,
                                file_path="{}.model".format(model_save_name))

        ref_max_rate = Trainer.test(epoch)

        all_ref_max_rate.append(ref_max_rate)
        best_ref_max_rate = max(all_ref_max_rate)

        if all_ref_max_rate[-1] < best_ref_max_rate:
            dynamic_lr *= 0.1
            Trainer.init_optimizer(lr=dynamic_lr)
            threshold += 1
        else:
            threshold = 0

        max_epoch = start_epoch + np.argmax(np.array(all_ref_max_rate))
        if threshold >= patient:
            break


if __name__ == "__main__":
    main()
