#!/usr/bin/env python
# -*- coding: utf-8 -*-
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from dataclasses import _MISSING_TYPE, dataclass, field, is_dataclass
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.models.bert.configuration_bert import BertConfig
import logging


class SentBertModelConfig(BertConfig):
    model_type = "sent_bert"
    
    def __init__(  
            self,
            pooling_option:str = 'mean',
            **kwargs
            ):
        super().__init__(**kwargs)
        self.pooling_option = pooling_option


class SentBertModel(BertPreTrainedModel):
    """
        add AuToModel
        https://huggingface.co/docs/transformers/model_doc/auto
    """
    config_class = SentBertModelConfig
    POOLING_OPTIONS = ['mean', 'first']

    def __init__(
            self,
            config,
            **kwargs
    ) -> None:
        super().__init__(config)
        self.config = config
        self.pooling_option = config.pooling_option
        assert self.pooling_option in self.POOLING_OPTIONS, 'check the pooling options [{}]'.format(", ".join(self.POOLING_OPTIONS))

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.post_init()

    def mean_pooling(self, last_hidden_state, attention_mask=None):
        ## use all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            **kwargs
    ) -> Dict[str, Any]:
        
        model_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if isinstance(model_outputs, tuple):
            """ past version of transformers return tuple"""
            last_hidden_state, _ = model_outputs
        elif is_dataclass(model_outputs):
            last_hidden_state = model_outputs.last_hidden_state
        else:
            raise NotImplementedError(f'can not support model output type: {type(model_outputs)}')
            
        if self.pooling_option == 'mean':
            ## use mean pooling
            logits = self.mean_pooling(last_hidden_state, attention_mask)
        else:
            ## use [CLS] token output
            logits = model_outputs[1]

        return logits
