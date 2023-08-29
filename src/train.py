#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utils import set_seed
import torch
import os
import logging
import argparse
from trainer import SimcseTrainer
from utils import PreprocessorFactory 
from utils import get_model_argparse
from logger import Experi_Logger
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
from model import CONFIG_MAPPING_DICT

def main(args):
    
    args = nli_parser_model_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.output_dir = f'/app/data/model/{args.pretrained_model}'
    args.log_dir = f'/app/data/log/{args.pretrained_model}'
    args.experiments_path = f'/app/data/experiment/{args.pretrained_model}'
    args.model_max_len = 50
    args.valid_first = False

    """ initialize seed """
    set_seed(args.seed)

    """ initialize logger """
    Experi_Logger(args.log_dir)

    ## check and make dir
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.experiments_path, exist_ok=True)

    args.experiments_path = f'{args.experiments_path}/train_simcse_{args.pooling_option}_{args.metric}.csv'
    
    model = MODEL_MAPPING_DICT[args.model_type].from_pretrained(
        args.pretrained_model, **vars(args), 
    )

    """ load tokenizer """
    logging.info("load pretrained checkpoint from [{}]".format(args.pretrained_model))
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    logging.info("load features")

    mnli_train = os.path.join(args.train_file, 'multinli.train.ko.tsv')
    snli_train = os.path.join(args.train_file, 'snli_1.0_train.ko.tsv')
    data_path = [mnli_train, snli_train]

    preprocessor = PreprocessorFactory(data_type=args.train_data_type)
    train_features = preprocessor.preprocess(
                              data_path=data_path,
                              tokenizer=tokenizer, 
                              save_path=None)

    for i in  range(len(train_features)):
        if i < 10:
            print(train_features[i].sentence_c)
        else:
            break
    
    logging.info("Build valid data")
    
    data_path = '/app/data/open_data/KorSTS/sts-train.tsv'
    preprocessor = PreprocessorFactory(data_type=args.val_data_type)
    valid_features = preprocessor.preprocess(
                              data_path=data_path,
                              tokenizer=tokenizer, 
                              save_path=None
    )

    logging.info("load dataset")

    train_dataset = DATASET_MAPPING_DICT['Unsup_simcse']
    train_dataset = train_dataset(args=args, features=train_features, max_length=args.model_max_len, tokenizer=tokenizer)

    val_dataset = DATASET_MAPPING_DICT['StsDataset']
    val_dataset = val_dataset(args=args, features=valid_features, max_length=args.model_max_len, tokenizer=tokenizer)

    # if args.valid_first:
    #     """ validation check without learning """
    #     trainer.model_setting(model_type=args.model_type, train_dataset=train_dataset, model=model, tokenizer=tokenizer)
    #     eval_result = trainer.validate(test_dataset=test_dataset, epoch=0)

    trainer = SimcseTrainer(args=args, logger=logging)
    trainer.train(model=model, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset, model_type=args.model_type)

    
if __name__ == "__main__":
    args = nli_parser_model_args()
    main(args=args)
   