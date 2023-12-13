from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from dataclasses import _MISSING_TYPE, dataclass, field
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.models.bert.configuration_bert import BertConfig
import logging


class SentPgdBertModelConfig(BertConfig):
    model_type = "sent_pgd_bert"

    def __init__(
            self,
            pooling_option:str = 'mean',
            eps: float = 1e-2,
            alpha: float = 0.000125,
            steps: int = 3,
            **kwargs):
        super().__init__(**kwargs)
        self.pooling_option = pooling_option
        self.eps= eps
        self.alpha=alpha
        self.steps=steps


class SentPgdBertModel(BertPreTrainedModel):
    """
        add AuToModel
        https://huggingface.co/docs/transformers/model_doc/auto
    """
    config_class = SentPgdBertModelConfig
    POOLING_OPTIONS = ['mean', 'first']


    def __init__(
            self,
            config,
            **kwargs
    ) -> None:
        super().__init__(config)
        self.eps= config.eps
        self.alpha=config.alpha
        self.steps=config.steps
        self._targeted = -1
        self.config = config
        self.pooling_option = config.pooling_option
        assert self.pooling_option in self.POOLING_OPTIONS, 'check the pooling options [{}]'.format(", ".join(self.POOLING_OPTIONS))

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        self.loss_fct = nn.MSELoss()

        self.post_init()
        
    def mean_pooling(self, model_outputs, attention_mask=None):
        ## use all token embeddings
        last_hiddens = model_outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hiddens.size()).float()
        return torch.sum(last_hiddens * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            token_type_ids: torch.Tensor = None,
            tgt_logits : torch.Tensor = None,
            labels: torch.Tensor = None,
            **kwargs
    ) -> torch.Tensor:
        
        adv_embedding = self.bert.embeddings.word_embeddings(input_ids)
        
        if self.training and labels is not None and tgt_logits is not None:
            tgt_logits = tgt_logits.detach()
            adv_embedding = self.attack(adv_embedding, attention_mask, token_type_ids, tgt_logits, labels)
            
        logits = self.process(adv_embedding, attention_mask, token_type_ids)
        
        return logits
    
    def attack(
            self, 
            embeddings: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            token_type_ids: torch.Tensor = None,
            tgt_logits : torch.Tensor = None,
            labels: torch.Tensor = None,
            **kwargs
    ) -> torch.Tensor:

        labels = labels.clone().detach().to(self.device)
        adv_embeddings = embeddings.clone().detach()

        for i in range(self.steps):

            adv_embeddings.requires_grad = True

            adv_logits = self.process(
                inputs_embeds = adv_embeddings, 
                attention_mask = attention_mask,
                token_type_ids=token_type_ids,
            )
            
            cost = self._targeted * self.loss_fct(self.get_score(adv_logits, tgt_logits), labels.view(-1))

            grad = torch.autograd.grad(cost, adv_embeddings,
                                       retain_graph=False, create_graph=False)[0]

            adv_embeddings = adv_embeddings.detach() - self.alpha*grad.sign()
            delta = torch.clamp(adv_embeddings - embeddings, min=-self.eps, max=self.eps)
            #adv_embeddings = torch.clamp(embeddings + delta, min=0, max=1).detach()
            adv_embeddings = (embeddings + delta).detach()
        
        return adv_embeddings

    def process(
        self, 
        inputs_embeds: torch.Tensor = None, 
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        
        model_outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        if self.pooling_option == 'mean':
            ## use mean pooling
            logits = self.mean_pooling(model_outputs, attention_mask)
        else:
            ## use [CLS] token output
            logits = model_outputs[1]

        return logits
    
    def get_score(
        self,
        src_logits, 
        tgt_logits
    ) -> torch.Tensor :
        return torch.cosine_similarity(src_logits, tgt_logits)
    