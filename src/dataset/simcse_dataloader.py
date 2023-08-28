#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader, Dataset
from abc import *
from torch.utils.data.distributed import DistributedSampler
from .abs_dataloader import AbstractDataloader
from dataset import SimcseDataset
from typing import List, Dict, Any, Union
import torch
from src.input import NLIInput, TokenizerInput

class SimcseDataloader(AbstractDataloader):
    def __init__(
            self,
            args,
            features:List[NLIInput],
            tokenizer_input: TokenizerInput,
            tokenizer,
            **kwargs
    ):
        super(SimcseDataloader, self).__init__()
        self.args = args
        self.features = features
        self.tokenizer_input = tokenizer_input
        self.max_length = tokenizer_input.max_length if tokenizer_input.max_length else 512
        self.pad_token_id = tokenizer.pad_token_id
        self.sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id else tokenizer.eos_token_id

    def code(cls):
        return 'nli'

    def _get_train_dataset(self):
        return SimcseDataset(args=self.args, features=self.features, max_length=self.max_length)
    
    def _get_val_dataset(self):
        return SimcseDataset(args=self.args, features=self.features, max_length=self.max_length)

    def _get_test_dataset(self):
        return SimcseDataset(args=self.args, features=self.features, max_length=self.max_length)

    def collater(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:

        a_input_ids = [input['a_input_ids'] for input in batch]
        a_attention_mask = [input['a_attention_mask'] for input in batch]
        b_input_ids = [input['b_input_ids'] for input in batch]
        b_attention_mask = [input['b_attention_mask'] for input in batch]
        c_input_ids = [input['c_input_ids'] for input in batch]
        c_attention_mask = [input['c_attention_mask'] for input in batch]

        ## token level encoding, variable length -> fixed lenghth
        batch_size = len(batch)
        a_sizes = [len(s) for s in a_input_ids]
        target_size = min(max(a_sizes), self.max_length)

        a_collated_ids = torch.full((batch_size, target_size), self.pad_token_id, dtype=torch.long)
        a_collated_attention_masks = torch.zeros((batch_size, target_size), dtype=torch.long)

        for i, (input_id, attention_m, size) in enumerate(zip(a_input_ids, a_attention_mask, a_sizes)):
            diff = target_size - size
            if diff < 0:
                """ when sentence len > max_length """
                a_collated_ids[i, :target_size] = input_id[:target_size]
                a_collated_ids[i, -1] = self.sep_token_id
                a_collated_attention_masks[i, :target_size] = attention_m[:target_size]
            else:
                """ when sentence len < max_length """
                a_collated_ids[i, :size] = input_id
                a_collated_attention_masks[i, :size] = attention_m

        b_sizes = [len(s) for s in b_input_ids]
        b_target_size = min(max(b_sizes), self.max_length)

        b_collated_ids = torch.full((batch_size, b_target_size), self.pad_token_id, dtype=torch.long)
        b_collated_attention_masks = torch.zeros((batch_size, b_target_size), dtype=torch.long)

        for i, (input_id, attention_m, size) in enumerate(zip(b_input_ids, b_attention_mask, b_sizes)):
            diff = b_target_size - size
            if diff < 0:
                """ when sentence len > max_length """
                b_collated_ids[i, :b_target_size] = input_id[:b_target_size]
                b_collated_ids[i, -1] = self.sep_token_id
                b_collated_attention_masks[i, :b_target_size] = attention_m[:b_target_size]

            else:
                """ when sentence len < max_length """
                b_collated_ids[i, :size] = input_id
                b_collated_attention_masks[i, :size] = attention_m


        c_sizes = [len(s) for s in c_input_ids]
        c_target_size = min(max(c_sizes), self.max_length)

        c_collated_ids = torch.full((batch_size, c_target_size), self.pad_token_id, dtype=torch.long)
        c_collated_attention_masks = torch.zeros((batch_size, c_target_size), dtype=torch.long)

        for i, (input_id, attention_m, size) in enumerate(zip(c_input_ids, c_attention_mask, c_sizes)):
            diff = c_target_size - size
            if diff < 0:
                """ when sentence len > max_length """
                c_collated_ids[i, :c_target_size] = input_id[:c_target_size]
                c_collated_ids[i, -1] = self.sep_token_id
                c_collated_attention_masks[i, :c_target_size] = attention_m[:c_target_size]

            else:
                """ when sentence len < max_length """
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

            'labels': collated_labels
        }



