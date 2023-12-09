#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Any, Union, Dict


@dataclass
class TokenizerInput:
    add_special_tokens:bool = True
    max_length:int = None
    pad_to_max_length:bool = False
    return_attention_mask:bool = True
    return_token_type_ids:bool = False


@dataclass
class NLIInput:
    sentence_a: str = None
    sentence_b: str = None
    sentence_c: str = None
    a_input_ids: List[int] = None
    a_attention_mask: List[int] = None
    b_input_ids: List[int] = None
    b_attention_mask: List[int] = None
    c_input_ids: List[int] = None
    c_attention_mask: List[int] = None
    label: Union[str, int] = None


@dataclass
class StsInput:
    sentence_a: str = None
    sentence_b: str = None
    sentence_c: str = None
    a_input_ids: List[int] = None
    a_attention_mask: List[int] = None
    b_input_ids: List[int] = None
    b_attention_mask: List[int] = None
    label: float = None

@dataclass
class SingleSentenceInput:
    sentence_a: str = None
    a_input_ids: List[int] = None
    a_attention_mask: List[int] = None

