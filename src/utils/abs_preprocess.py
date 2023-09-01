#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import dataclasses

from abc import *
class AbsPreprocessor(metaclass=ABCMeta):
    @classmethod         
    def tokenizing(cls, input, tokenizer, tokenizer_input=None):
        if tokenizer_input is None:
            return tokenizer.encode_plus(input)
        else:
            kwargs = dataclasses.asdict(tokenizer_input)
            return tokenizer.encode_plus(input, **kwargs)

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """reading raw data"""
    @abstractmethod
    def prerocess(self):
        """ preprocess model input object"""