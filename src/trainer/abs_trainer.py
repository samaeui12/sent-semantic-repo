#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn.functional as F
from abc import *
import numpy as np
class AbstractTrainer(metaclass=ABCMeta):
    """
    abstract trainer class
    """
    def __init__(self, args, model=None, loader = None):
        self.args = args
        self.args.n_gpu = torch.cuda.device_count()
        self.model = model
        self.loader = loader
        self.best_train_loss = np.inf
        self.patience_limit = args.patience_limit
        self.patience = 0

    @abstractmethod
    def train(self, epoch, tau=None):
        """ abstract class for train """

    @abstractmethod
    def train_one_epoch(self, epoch, accum_iter, tau):
        """ abstrac class for training one epoch """

    @abstractmethod
    def validate(self, epoch, tau):
        """ abstrac class for validate """

    @abstractmethod
    def _create_state_dict(self, epoch):
        """ create dictionary for model save"""