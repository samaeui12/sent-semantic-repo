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

class Unsup_simcse(Dataset):
    def __init__(
            self,
            args,
            features:List[NLIInput],
            max_length,
            tokenizer,
            **kwargs
    ):
        super(Unsup_simcse, self).__init__()
        self.args = args
        self.features = features
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id else tokenizer.eos_token_id

    def __getitem__(self, index) -> Dict[str, Any]:
        feature = self.features[index]
        return {
            'a_input_ids': torch.tensor(feature.a_input_ids, dtype=torch.long),
            'a_attention_mask': torch.tensor(feature.a_attention_mask, dtype=torch.long),
            'b_input_ids': torch.tensor(feature.b_input_ids, dtype=torch.long),
            'b_attention_mask': torch.tensor(feature.b_attention_mask, dtype=torch.long),
            'c_input_ids': torch.tensor(feature.c_input_ids, dtype=torch.long),
            'c_attention_mask': torch.tensor(feature.c_attention_mask, dtype=torch.long)
        }
    def __len__(self):
        return len(self.features)
    
    def loader(self, shuffle:bool=True, batch_size:int=64):
        return DataLoader(self, shuffle=shuffle, batch_size=batch_size, collate_fn=self.collater)

    def collater(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:

        a_input_ids = [data['a_input_ids'] for data in batch]
        a_attention_mask = [data['a_attention_mask'] for data in batch]

        b_input_ids = [data['b_input_ids'] for data in batch]
        b_attention_mask = [data['b_attention_mask'] for data in batch]

        c_input_ids = [data['c_input_ids'] for data in batch]
        c_attention_mask = [data['c_attention_mask'] for data in batch]

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

        sizes = [len(s) for s in b_input_ids]
        target_size = min(max(sizes), self.max_length)

        b_collated_ids = torch.full((batch_size, target_size), self.pad_token_id, dtype=torch.long)
        b_collated_attention_masks = torch.zeros((batch_size, target_size), dtype=torch.long)

        for i, (input_id, attention_m, size) in enumerate(
                zip(b_input_ids, b_attention_mask, sizes)):
            diff = target_size - size
            if diff < 0:
                b_collated_ids[i, :target_size] = input_id[:target_size]
                b_collated_ids[i, -1] = self.sep_token_id
                b_collated_attention_masks[i, :target_size] = attention_m[:target_size]

            else:
                b_collated_ids[i, :size] = input_id
                b_collated_attention_masks[i, :size] = attention_m

        sizes = [len(s) for s in c_input_ids]
        target_size = min(max(sizes), self.max_length)

        c_collated_ids = torch.full((batch_size, target_size), self.pad_token_id, dtype=torch.long)
        c_collated_attention_masks = torch.zeros((batch_size, target_size), dtype=torch.long)
        for i, (input_id, attention_m, size) in enumerate(
                zip(c_input_ids, c_attention_mask, sizes)):
            diff = target_size - size
            if diff < 0:
                c_collated_ids[i, :target_size] = input_id[:target_size]
                c_collated_ids[i, -1] = self.sep_token_id
                c_collated_attention_masks[i, :target_size] = attention_m[:target_size]
            else:
                c_collated_ids[i, :size] = input_id
                c_collated_attention_masks[i, :size] = attention_m
                
        collated_labels = torch.arange(c_collated_ids.size(0), c_collated_ids.size(0)+c_collated_ids.size(0), dtype=torch.long)

        return {
            'a_input_ids': a_collated_ids,
            'a_attention_mask': a_collated_attention_masks,

            'b_input_ids': b_collated_ids,
            'b_attention_mask': b_collated_attention_masks,

            'c_input_ids': c_collated_ids,
            'c_attention_mask': c_collated_attention_masks,

            'labels': collated_labels,
        }
    
class Sup_simcse(Dataset):
    def __init__(
            self,
            args,
            features:List[NLIInput],
            max_length,
            tokenizer,
            **kwargs
    ):
        super(Sup_simcse, self).__init__()
        self.args = args
        self.features = features
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id else tokenizer.eos_token_id

    def __getitem__(self, index) -> Dict[str, Any]:
        feature = self.features[index]
        return {
            'a_input_ids': torch.tensor(feature.a_input_ids, dtype=torch.long),
            'a_attention_mask': torch.tensor(feature.a_attention_mask, dtype=torch.long),
            'b_input_ids': torch.tensor(feature.b_input_ids, dtype=torch.long),
            'b_attention_mask': torch.tensor(feature.b_attention_mask, dtype=torch.long),
            'c_input_ids': torch.tensor(feature.c_input_ids, dtype=torch.long),
            'c_attention_mask': torch.tensor(feature.c_attention_mask, dtype=torch.long)
        }
    def __len__(self):
        return len(self.features)
    
    def loader(self, shuffle:bool=True, batch_size:int=64):
        return DataLoader(self, shuffle=shuffle, batch_size=batch_size, collate_fn=self.collater)

    def collater(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:

        a_input_ids = [data['a_input_ids'] for data in batch]
        a_attention_mask = [data['a_attention_mask'] for data in batch]

        b_input_ids = [data['b_input_ids'] for data in batch]
        b_attention_mask = [data['b_attention_mask'] for data in batch]

        c_input_ids = [data['c_input_ids'] for data in batch]
        c_attention_mask = [data['c_attention_mask'] for data in batch]

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

        sizes = [len(s) for s in b_input_ids]
        target_size = min(max(sizes), self.max_length)

        b_collated_ids = torch.full((batch_size, target_size), self.pad_token_id, dtype=torch.long)
        b_collated_attention_masks = torch.zeros((batch_size, target_size), dtype=torch.long)

        for i, (input_id, attention_m, size) in enumerate(
                zip(b_input_ids, b_attention_mask, sizes)):
            diff = target_size - size
            if diff < 0:
                b_collated_ids[i, :target_size] = input_id[:target_size]
                b_collated_ids[i, -1] = self.sep_token_id
                b_collated_attention_masks[i, :target_size] = attention_m[:target_size]

            else:
                b_collated_ids[i, :size] = input_id
                b_collated_attention_masks[i, :size] = attention_m

        sizes = [len(s) for s in c_input_ids]
        target_size = min(max(sizes), self.max_length)

        c_collated_ids = torch.full((batch_size, target_size), self.pad_token_id, dtype=torch.long)
        c_collated_attention_masks = torch.zeros((batch_size, target_size), dtype=torch.long)
        for i, (input_id, attention_m, size) in enumerate(
                zip(c_input_ids, c_attention_mask, sizes)):
            diff = target_size - size
            if diff < 0:
                c_collated_ids[i, :target_size] = input_id[:target_size]
                c_collated_ids[i, -1] = self.sep_token_id
                c_collated_attention_masks[i, :target_size] = attention_m[:target_size]
            else:
                c_collated_ids[i, :size] = input_id
                c_collated_attention_masks[i, :size] = attention_m
                
        collated_labels = torch.arange(c_collated_ids.size(0), c_collated_ids.size(0)+c_collated_ids.size(0), dtype=torch.long)

        return {
            'a_input_ids': a_collated_ids,
            'a_attention_mask': a_collated_attention_masks,

            'b_input_ids': b_collated_ids,
            'b_attention_mask': b_collated_attention_masks,

            'c_input_ids': c_collated_ids,
            'c_attention_mask': c_collated_attention_masks,

            'labels': collated_labels,
        }