#!/usr/bin/env python
# -*- coding: utf-8 -*-
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel
import torch.nn as nn
import torch
from typing import Any, Dict

class SentRobertaModelConfig(RobertaConfig):
    model_type = "sent_roberta"
    
    def __init__(self, pooling_option: str = 'mean', **kwargs):
        super().__init__(**kwargs)
        self.pooling_option = pooling_option


class SentRobertaModel(RobertaPreTrainedModel):
    config_class = SentRobertaModelConfig
    POOLING_OPTIONS = ['mean', 'first']

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.pooling_option = config.pooling_option
        assert self.pooling_option in self.POOLING_OPTIONS, f'Check the pooling options [{", ".join(self.POOLING_OPTIONS)}]'

        self.roberta = RobertaModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.post_init()

    def mean_pooling(self, last_hidden_state, attention_mask=None):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None, **kwargs):
        # print('input model', input_ids.size())
        model_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        if isinstance(model_outputs, tuple):
            last_hidden_state, _ = model_outputs
        elif hasattr(model_outputs, 'last_hidden_state'):
            last_hidden_state = model_outputs.last_hidden_state
        else:
            raise NotImplementedError(f'Cannot support model output type: {type(model_outputs)}')

        if self.pooling_option == 'mean':
            logits = self.mean_pooling(last_hidden_state, attention_mask)
        else:
            logits = model_outputs[1]

        return logits
