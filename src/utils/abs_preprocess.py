#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from abc import *
class AbsPreprocessor(metaclass=ABCMeta):
    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        """preprocessing raw data"""
    @abstractmethod
    def build(self):
        """ build model input object"""