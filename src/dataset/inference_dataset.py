#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.input import NLIInput

import torch
from torch.utils.data import (
    DataLoader, Dataset
)
from src.input import InferenceDataset

class InferenceDataset(Dataset):
    def __init__(
            self,
            args,
            features:List[SingleSentenceInput],
            max_length,
            tokenizer,
            **kwargs
    ):
        super(EmbeddingDataset, self).__init__()
        self.args = args
        self.features = features
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id else tokenizer.eos_token_id

    def __getitem__(self, index) -> Dict[str, Any]:
        feature = self.features[index]
        return {
            'a_sentence': feature.sentence_a,
            'a_input_ids': torch.tensor(feature.a_input_ids, dtype=torch.long),
            'a_attention_mask': torch.tensor(feature.a_attention_mask, dtype=torch.long)
        }
    def __len__(self):
        return len(self.features)
    
    def loader(self, shuffle:bool=True, batch_size:int=64):
        return DataLoader(self, shuffle=shuffle, batch_size=batch_size, collate_fn=self.collater)

    def collater(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:

        a_sentence = [data['a_sentence'] for data in batch]
        a_input_ids = [data['a_input_ids'] for data in batch]
        a_attention_mask = [data['a_attention_mask'] for data in batch]
        ##  token level encoding
        batch_size = len(batch)
        sizes = [len(s) for s in a_input_ids]
        target_size = min(max(sizes), self.max_length)
        """ torch.full -> creates a tensor of a given shape and fills it with a scalar value self.pad_token_id here"""
        a_collated_ids = torch.full((batch_size, target_size), self.pad_token_id, dtype=torch.long)
        a_collated_attention_masks = torch.zeros((batch_size, target_size), dtype=torch.long)

        """ cut data if size > target_size else: fill by self.pad_token_id """
        for i, (input_id, attention_m, size) in enumerate(
                zip(a_input_ids, a_attention_mask, sizes)):
            diff = target_size - size
            if diff < 0:
                a_collated_ids[i, :target_size] = input_id[:target_size]
                a_collated_ids[i, -1] = self.sep_token_id
                a_collated_attention_masks[i, :target_size] = attention_m[:target_size]

            else:
                a_collated_ids[i, :size] = input_id
                a_collated_attention_masks[i, :size] = attention_m

        return {
            'a_sentence': a_sentence,
            'a_input_ids': a_collated_ids,
            'a_attention_mask': a_collated_attention_masks
        }