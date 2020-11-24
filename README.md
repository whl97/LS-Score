

This project is the implementation of paper *Unsupervised Reference-Free Summary Quality Evaluation via Contrastive Learning*. ([Paper Link](https://arxiv.org/pdf/2010.01781.pdf))



# Datasets

###  Newsroom

the complete dataset is available [here](https://github.com/lil-lab/newsroom).

The processed dataset is available [here](
https://www.dropbox.com/s/aw3zhmh2jby351c/Newsroom%202.zip?dl=0).

### CNN/Daily Mail

the complete dataset is also available [here](https://github.com/becxer/cnn-dailymail/).


The processed dataset is available [here](https://www.dropbox.com/s/mrhlj4lqyr7rice/CNNDM.zip?dl=0).



# Fine-tuning

1. Download the pretrained model and modified `pretrained_model_dir`. We use [`bert-base-uncased`](https://github.com/huggingface/transformers) as our base model to finetune.

2. Download the datasets and modified `dataset_dir_dict`.

3. To fine-tune the model:

   ```shell
   python3 code/trainer.py --dataset_name=cnndm
   python3 code/trainer.py --dataset_name=newsroom
   ```