#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utils import set_seed
import torch
import os
import logging
from trainer import SimcseTrainer
from utils import PreprocessorFactory
from transformers import (
    AdamW,
    AutoModel,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
)
from logger import Experi_Logger

from config.nli_config import nli_parser_model_args
from dataset import DATASET_MAPPING_DICT
from model import MODEL_MAPPING_DICT


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.output_dir = f'/app/data/semantic-search-lib/model/{args.pretrained_model}'
    args.log_dir = f'/app/data/semantic-search-lib/log/{args.pretrained_model}'
    args.experiments_path = f'/app/data/semantic-search-lib/experiment/{args.pretrained_model}'
    args.model_max_len = 50
    args.valid_first = False

    # initialize seed
    set_seed(args.seed)

    # initialize logger
    Experi_Logger(args.log_dir)

    # check and make dir
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.experiments_path, exist_ok=True)
    args.experiments_path = f'{args.experiments_path}/train_simcse_{args.pooling_option}_{args.metric}.csv'
    model = MODEL_MAPPING_DICT[args.model_type].from_pretrained(
        args.pretrained_model, **vars(args), 
    )

    # load tokenizer
    logging.info("load pretrained checkpoint from [{}]".format(args.pretrained_model))
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    # load train data
    logging.info("Build train data")
    # mnli_train = os.path.join(args.train_file, 'multinli.train.ko.tsv')
    # snli_train = os.path.join(args.train_file, 'snli_1.0_train.ko.tsv')
    # mno_train = os.path.join(args.train_file, 'mno_train.tsv')
    # data_path = [mnli_train, snli_train, mno_train]
    preprocessor = PreprocessorFactory(data_type=args.data_type)
    train_features = preprocessor.preprocess(
                              data_path=args.train_file,
                              tokenizer=tokenizer, 
                              save_path=None)

    # print train features head
    for i in range(len(train_features)):
        if i < 10:
            print(train_features[i].sentence_c)
        else:
            break

    # load valid data
    logging.info("Build valid data")
    preprocessor = PreprocessorFactory(data_type=args.data_type)
    valid_features = preprocessor.preprocess(
                              data_path=args.valid_file,
                              tokenizer=tokenizer,
                              save_path=None
    )

    # define dataset
    logging.info("load dataset")
    train_dataset = DATASET_MAPPING_DICT['Sup_simcse']
    train_dataset = train_dataset(args=args, features=train_features, max_length=args.model_max_len, tokenizer=tokenizer)
    val_dataset = DATASET_MAPPING_DICT['Sup_simcse']
    val_dataset = val_dataset(args=args, features=valid_features, max_length=args.model_max_len, tokenizer=tokenizer)

    # train
    trainer = SimcseTrainer(args=args, logger=logging)
    trainer.train(model=model, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset, model_type=args.model_type)

    
if __name__ == "__main__":
    args = nli_parser_model_args()
    main(args=args)
