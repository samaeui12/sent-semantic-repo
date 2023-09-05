#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random
import torch
from datetime import datetime
import pandas as pd
import logging
import os
import sys
from torch import Tensor

def get_model_argparse(parser):
    parser.add_argument(
        "--num_labels",  default=10, type=int, help="num of labels to modeling bert"
    )
    parser.add_argument(
        "--pooling_option",  default='mean', type=str, help="Options to controll models pooling layer. If you select 'mean', model use mean pooling layers to construct sentence vector. You can also use 'first' to use [CLS] token of the model"
    )
    parser.add_argument(
        "--eps",  default=1e-2, type=float,
    )
    parser.add_argument(
        "--alpha",  default=0.000125, type=float, 
    )
    parser.add_argument(
        "--steps",  default=3, type=int, 
    )
    return parser


def set_seed(seed):
    """
    실험 재현을 위한 시드 고정
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

class SummaryWriter:
    def __init__(self, directory):

        self.dir = directory
        self.hparams = None
        self.load()
        self.writer = dict()

    def update(self, args, **results):
        now = datetime.now()
        date = "%s-%s %s:%s" % (now.month, now.day, now.hour, now.minute)
        self.writer.update({"date": date})
        self.writer.update(results)
        self.writer.update(vars(args))

        if self.hparams is None:
            self.hparams = pd.DataFrame(self.writer, index=[0])
        else:
            self.hparams = pd.concat([self.hparams, pd.DataFrame([self.writer])], ignore_index=True)
        self.save()
        logging.info("save experiment info [{}]".format(self.dir))

    def save(self):
        assert self.hparams is not None
        self.hparams.to_csv(self.dir, index=False)

    def load(self):
        path = os.path.split(self.dir)[0]
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            self.hparams = None
        elif os.path.exists(self.dir):
            self.hparams = pd.read_csv(self.dir)
        else:
            self.hparams = None

def print_grad(model, layer_name, ignore_none_grad:bool=True):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if layer_name =='all':
                if param.grad is None:
                    print('####')
                    print(name)
                else:
                    print(f"Gradient of {name}: {param.grad}")

            elif layer_name =='stats':
                if ignore_none_grad:
                    if param.grad is not None:
                        mean = param.grad.mean().item()
                        norm = param.grad.norm().item()
                        print(f"Gradient stats of {name}: mean = {mean}, norm = {norm}")
            else:
                if name == layer_name:
                    print(name, param.grad)    

def print_weight(model):
    for name, child in model.named_children():
        child_name = name
        print(child)
        before_weight = child.weight
        print(f"weight: {child.weight}")

def is_equal(weight1, weight2):
    return torch.ne(weight1, weight2)