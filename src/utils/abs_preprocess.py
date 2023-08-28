#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from transformers.tokenization_utils import PreTrainedTokenizer
from input import TokenizerInput
import dataclasses

from abc import *
class AbsPreprocessor(metaclass=ABCMeta):
    @classmethod         
    def tokenizing(cls, input, tokenizer:PreTrainedTokenizer, tokenizer_input:TokenizerInput=None):
        if tokenizer_input is None:
            return tokenizer.encode_plus(input)
        else:
            kwargs = dataclasses.asdict(tokenizer_input)
            return tokenizer.encode_plus(input, **kwargs)

    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        """preprocessing raw data"""
    @abstractmethod
    def build(self):
        """ build model input object"""