#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utils import set_seed
import torch
import os
import logging
import argparse
from trainer import SimcseTrainer
from dataset import SimcseDataset, StsDataset
from input import SimcseInput
from utils import PreprocessorFactory 
from utils import get_model_argparse
from model import MODEL_MAPPING_DICT
from model import CONFIG_MAPPING_DICT
from logger import Experi_Logger
from transformers import (
    AdamW,
    AutoModel,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
)
from logger import Experi_Logger

def main(args):

    args.no_cuda = False
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.model_type = 'sent_roberta'
    #args.pretrained_model = 'klue/bert-base'
    args.pretrained_model = 'klue/roberta-large'
    args.output_dir = f'/app/data/model/{args.pretrained_model}'
    args.log_dir = f'/app/data/log/{args.pretrained_model}'
    args.metric='spearman'
    args.seed = 777
    args.model_max_len = 50
    args.num_train_epochs = 10
    args.warmup_percent = 0.01
    args.gradient_accumulation_steps = 4
    args.temperature = 0.05
    args.train_batch_size = 256
    args.eval_batch_size = 64
    args.patience_limit = 3
    args.experiments_path = f'/app/data/experiment/{args.pretrained_model}'
    args.weight_decay = 0.01
    args.learning_rate = 3e-5
    args.max_grad_norm = 1.0
    args.amp = True
    args.is_preprocessed = True
    args.valid_first = False

    print(args)

    """ initialize seed """
    set_seed(args.seed)

    """ initialize logger """
    Experi_Logger(args.log_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.experiments_path, exist_ok=True)

    args.experiments_path = f'{args.experiments_path}/train_simcse_{args.pooling_option}_{args.metric}.csv'
    
    model_type = args.model_type
    model = MODEL_MAPPING_DICT[model_type].from_pretrained(
        args.pretrained_model, **vars(args), 
    )

    """ tokenizer """
    logging.info("load pretrained checkpoint from [{}]".format(args.pretrained_model))
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    logging.info("load features")

    if args.is_preprocessed:
        data_path = None
    else:
        mnli_train = os.path.join(args.train_file, 'multinli.train.ko.tsv')
        snli_train = os.path.join(args.train_file, 'snli_1.0_train.ko.tsv')
        data_path = [mnli_train, snli_train]

    preprocessor = PreprocessorFactory(model_type='simcse')
    save_path = '/app/data/open_data/preprocess/KorNLI/kornli.tsv'
    train_features = preprocessor.build(data_path=data_path, save_path=save_path, tokenizer=tokenizer, is_preprocessed=args.is_preprocessed)

    for i in  range(len(train_features)):
        if i < 10:
            print(train_features[i].sentence_c)
        else:
            break
    
    logging.info("Build valid data")
    
    preprocessor = PreprocessorFactory(model_type='sts')
    save_path = '/app/data/open_data/preprocess/KorSTS/korsts.tsv'
    valid_features = preprocessor.build(data_path=data_path, save_path=save_path, tokenizer=tokenizer, is_preprocessed=args.is_preprocessed)

    logging.info("load dataset")

    train_dataset = SimcseDataset(args=args, features=train_features, max_length=args.model_max_len, tokenizer=tokenizer)
    test_dataset = StsDataset(args=args, features=valid_features, max_length=args.model_max_len, tokenizer=tokenizer)

    if args.valid_first:
        """ validation check without learning """
        trainer.model_setting(model_type=model_type, train_dataset=train_dataset, model=model, tokenizer=tokenizer)
        eval_result = trainer.validate(test_dataset=test_dataset, epoch=0)

    trainer= SimcseTrainer(args=args, logger=logging)
    trainer.train(model=model, tokenizer=tokenizer, train_dataset=train_dataset, test_dataset=test_dataset, model_type=model_type)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Required parameters
    parser.add_argument(
        "--train_file",
        default='/app/data/open_data/preprocess/KorNLI',
        #required=True,
        type=str,
        help="The input training data file.",
    )
    parser.add_argument(
        "--valid_file",
        default='/app/data/open_data/preprocess/KorSTS',
        #required=True,
        type=str,
        help="The input training data file.",
    )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     required=True,
    #     help="The output directory where the model predictions and checkpoints will be written.",
    # )
    parser.add_argument(
        "--experiments_path",
        type=str,
        default='experiments/experiment.csv',
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=64,
        type=int,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=64,
        type=int,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--model_max_len", default=512, type=int, help="Maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_percent",
        default=0.1,
        type=float,
        help="Percentage of linear warmup over warmup_steps.",
    )
    parser.add_argument(
        "--logging_steps",
        default=1000,
        type=int,
        help="Generate log during training at each logging step.",
    )
    parser.add_argument(
        "--steps_per_evaluate",
        default=100,
        type=int,
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--model_type",
        default='sent_bert',
        type=str,
        help="must select model type in [{}]".format(", ".join(MODEL_MAPPING_DICT)),
    )
    parser.add_argument(
        "--pretrained_model",
        default=None,
        type=str,
        help="If there is pretrained Model",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.05, help="For SimCSE temperature"
    )

    parser = get_model_argparse(parser)
    args = parser.parse_args()

    main(args)