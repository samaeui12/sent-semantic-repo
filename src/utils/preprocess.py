#!/usr/bin/env python
# -*- coding: utf-8 -*-

## python library import
import os
import re
import sys
import logging
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from typing import List, Dict, Union
import sys
import pandas as pd
import csv
from common import YamlParser
from io import StringIO
from input import NLIInput, TokenizerInput, StsInput
import dataclasses
from itertools import product
from numpy import random
from tqdm import tqdm

## transformers library import
from transformers.tokenization_utils import PreTrainedTokenizer
from .abs_preprocess import AbsPreprocessor

class NliPreprocessor(AbsPreprocessor):
    """ file open -> tokenizing -> dataclasses"""
    @staticmethod
    def load_data(data_path:str, save_path:str=None, header:bool=True) -> List:
        """  Object: [sentence1 | sentence2 | goldlabel] -> [sentence1 | positive sentence2 | hard negative sentence3]  
             data description: multinli.train.ko.tsv: senetence1 기준 groubpy 통계: 125323개 (3), 1975: (6), 511(9), 14(12) etc(20)   
             Question Mark: 
                if  sentence1:sentence2: P
                    sentence1:sentence3: N
                then sentence2:sentence3 -> N??  
        """     
        dataset = []
        dataset.append(['sentence_a', 'sentence_b', 'sentence_c'])

        sentences = dict()       
        if isinstance(data_path, str):      
            data_path = [data_path]

        for train_file in data_path:
            with open(train_file, 'r') as file:
                for i, row in enumerate(file):
                    if header and i==0:
                        continue
                    
                    line = row.strip().split('\t')
                    if len(line) < 3:
                        continue

                    sentence_a = line[0]
                    sentence_b = line[1]
                    gold_label = line[2]
                    
                    if sentence_a not in sentences:
                        """ key:sentence_a -> {'entailment': [sent1, sent2, ...,] , 'contradiction':[]}"""
                        sentences[sentence_a] = {}
                        sentences[sentence_a]['entailment'] = []
                        sentences[sentence_a]['contradiction'] = []
                        sentences[sentence_a]['neutral'] = []
                    
                    sentences[sentence_a][gold_label] += [sentence_b]

        """ make dataset for every combination of entail and contra """
        for key, val in sentences.items():
            entails = val.get('entailment', [])
            contradiction = val.get('contradiction', [])
            if entails and contradiction:
                for item in list(product(entails, contradiction)):
                    dataset.append((key, item[0], item[1]))                

        if save_path is not None:
            if os.path.exists(save_path):
                logging.info(f'{save_path}: exists -> removing')
                os.remove(save_path)

            with open(save_path, 'a') as f:
                for data in dataset:
                    f.write('\t'.join(data) + '\n')
            
        return dataset
            
    @classmethod
    def preprocess(cls, data_path, tokenizer:PreTrainedTokenizer, save_path, tokenizer_input: TokenizerInput=None, header:bool=True) -> None:
        """ try read tsv file using pandas first if [memory or parse] error catched use other reading method  """
        
        feature_list = list()
        skipped_line = 0

        datasets = cls.load_data(data_path, save_path=save_path, header=header)
        for i, line in enumerate(datasets):
            try:
                if (len(line) < 3) or (i==0):
                    ## skip incomplete data && header
                    skipped_line += 1
                    continue

                a_sentence = line[0]
                b_sentence = line[1]
                c_sentence = line[2]
                a_encoded_sentence = cls.tokenizing(input=a_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)
                b_encoded_sentence = cls.tokenizing(input=b_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)
                c_encoded_sentence = cls.tokenizing(input=c_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)

                feature_list.append(
                    NLIInput(
                        sentence_a = line[0],
                        sentence_b = line[1],
                        sentence_c = line[2], 

                        a_input_ids = a_encoded_sentence.input_ids,
                        a_attention_mask=a_encoded_sentence.attention_mask,

                        b_input_ids=b_encoded_sentence.input_ids,
                        b_attention_mask=b_encoded_sentence.attention_mask,

                        c_input_ids = c_encoded_sentence.input_ids,
                        c_attention_mask=c_encoded_sentence.attention_mask
                    )
                )
            except Exception as e:
                print(f'Error occurs in {i} lines in preprocessing')
                print(line)
                print(e)

        return feature_list

    
class Stsprocessor(AbsPreprocessor):
    """ file open -> tokenizing -> dataclasses"""

    @classmethod
    def load_data(cls, data_path:Union[str, List], save_path:str=None, header:bool=True) -> List:
        dataset = []
        dataset.append(['sentence_a', 'sentence_b', 'label'])

        if isinstance(data_path, str):
            data_path = [data_path]

        for train_file in data_path:
            with open(train_file, 'r') as file:
                for i, row in enumerate(file):
                    if header and i==0:
                        continue

                    line = row.strip().split('\t')
                    if len(line) < 3:
                        continue

                    label = float(line[-3])
                    a_sentence = line[-2]
                    b_sentence = line[-1]
                    dataset.append([a_sentence, b_sentence, float(label)])
            
            if save_path is not None:            
                if os.path.exists(save_path):
                    logging.info(f'{save_path}: exists -> removing')
                    os.remove(save_path)

                with open(save_path, 'a') as f:
                    f.write('\t'.join(['sentence_a', 'sentence_b', 'label'])+'\n')
                    for data in dataset:
                        f.write('\t'.join(data) + '\n')

            return dataset

    @classmethod
    def preprocess(cls, data_path, save_path, tokenizer: PreTrainedTokenizer, tokenizer_input:TokenizerInput=None, header=True):

        datasets = cls.load_data(data_path, save_path=save_path, header=header)
        feature_list = list()
        
        for i, line in enumerate(datasets):
            if header and i==0:
                continue
            
            """ Header:  genre  |  filename  |	year  |   id	|  score  |   sentence1   |   sentence2  """
            sentence_a = line[0]
            sentence_b = line[1]
            label = line[2]
            
            a_encoded_output = cls.tokenizing(input=sentence_a, tokenizer=tokenizer, tokenizer_input=tokenizer_input)
            b_encoded_output = cls.tokenizing(input=sentence_b, tokenizer=tokenizer, tokenizer_input=tokenizer_input)

            feature = StsInput(
                sentence_a=sentence_a,
                sentence_b=sentence_b,
                a_input_ids=a_encoded_output.input_ids,
                a_attention_mask=a_encoded_output.attention_mask,
                b_input_ids=b_encoded_output.input_ids,
                b_attention_mask=b_encoded_output.attention_mask,
                label=float(label),
            )
            feature_list.append(feature)

        return feature_list
    

class Tripleprocessor(AbsPreprocessor):
    """ file open -> tokenizing -> dataclasses"""
    @staticmethod
    def load_data(data_path:str, save_path:str=None, header:bool=True) -> List:
        """  Object: [sentence1 | sentence2 | sentence3] 
             sentence1 <----> sentence2 should be Positive
             sentence1 <----> sentence3 should be Negative
        """     
        dataset = []
        dataset.append(['sentence_a', 'sentence_b', 'sentence_c'])

        sentences = dict()       
        if isinstance(data_path, str):      
            data_path = [data_path]

        for train_file in data_path:
            with open(train_file, 'r') as file:
                for i, row in enumerate(file):
                    if header and i==0:
                        continue

                    line = row.strip().split('\t')
                    if len(line) < 3:
                        continue

                    sentence_a = line[0]
                    sentence_b = line[1]
                    sentence_c = line[2]
                    dataset.append([sentence_a, sentence_b, sentence_c])
             
        if save_path is not None:
            if os.path.exists(save_path):
                logging.info(f'{save_path}: exists -> removing')
                os.remove(save_path)

            with open(save_path, 'a') as f:
                for data in dataset:
                    f.write('\t'.join(data) + '\n')
            
        return dataset
            
    @classmethod
    def preprocess(cls, data_path, tokenizer:PreTrainedTokenizer, save_path, tokenizer_input: TokenizerInput=None, header:bool=True) -> None:
        """ try read tsv file using pandas first if [memory or parse] error catched use other reading method  """
    
        feature_list = list()
        skipped_line = 0

        datasets = cls.load_data(data_path, save_path=save_path, header=header)
        for i, line in enumerate(datasets):
            try:
                if (len(line) < 3) or (i==0):
                    ## skip incomplete data && header
                    skipped_line += 1
                    continue

                a_sentence = line[0]
                b_sentence = line[1]
                c_sentence = line[2]
                a_encoded_sentence = cls.tokenizing(input=a_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)
                b_encoded_sentence = cls.tokenizing(input=b_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)
                c_encoded_sentence = cls.tokenizing(input=c_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)

                feature_list.append(
                    NLIInput(
                        sentence_a = line[0],
                        sentence_b = line[1],
                        sentence_c = line[2], 

                        a_input_ids = a_encoded_sentence.input_ids,
                        a_attention_mask=a_encoded_sentence.attention_mask,

                        b_input_ids=b_encoded_sentence.input_ids,
                        b_attention_mask=b_encoded_sentence.attention_mask,

                        c_input_ids = c_encoded_sentence.input_ids,
                        c_attention_mask=c_encoded_sentence.attention_mask
                    )
                )
            except Exception as e:
                print(f'Error occurs in {i} lines in preprocessing')
                print(line)
                print(e)

        return feature_list


class Stsprocessor(AbsPreprocessor):
    """ file open -> tokenizing -> dataclasses"""

    @classmethod
    def load_data(cls, data_path:Union[str, List], save_path:str=None, header:bool=True) -> List:
        dataset = []
        dataset.append(['sentence_a', 'sentence_b', 'label'])

        if isinstance(data_path, str):
            data_path = [data_path]

        for train_file in data_path:
            with open(train_file, 'r') as file:
                for i, row in enumerate(file):
                    if header and i==0:
                        continue

                    line = row.strip().split('\t')
                    if len(line) < 3:
                        continue

                    label = float(line[-3])
                    a_sentence = line[-2]
                    b_sentence = line[-1]
                    dataset.append([a_sentence, b_sentence, float(label)])
            
            if save_path is not None:            
                if os.path.exists(save_path):
                    logging.info(f'{save_path}: exists -> removing')
                    os.remove(save_path)

                with open(save_path, 'a') as f:
                    f.write('\t'.join(['sentence_a', 'sentence_b', 'label'])+'\n')
                    for data in dataset:
                        f.write('\t'.join(data) + '\n')

            return dataset

    @classmethod
    def preprocess(cls, data_path, save_path, tokenizer: PreTrainedTokenizer, tokenizer_input:TokenizerInput=None, header=True):

        datasets = cls.load_data(data_path, save_path=save_path, header=header)
        feature_list = list()
        
        for i, line in enumerate(datasets):
            if header and i==0:
                continue
            
            """ Header:  genre  |  filename  |	year  |   id	|  score  |   sentence1   |   sentence2  """
            sentence_a = line[0]
            sentence_b = line[1]
            label = line[2]
            
            a_encoded_output = cls.tokenizing(input=sentence_a, tokenizer=tokenizer, tokenizer_input=tokenizer_input)
            b_encoded_output = cls.tokenizing(input=sentence_b, tokenizer=tokenizer, tokenizer_input=tokenizer_input)

            feature = StsInput(
                sentence_a=sentence_a,
                sentence_b=sentence_b,
                a_input_ids=a_encoded_output.input_ids,
                a_attention_mask=a_encoded_output.attention_mask,
                b_input_ids=b_encoded_output.input_ids,
                b_attention_mask=b_encoded_output.attention_mask,
                label=float(label),
            )
            feature_list.append(feature)

        return feature_list
    

class Faqprocessor(AbsPreprocessor):
    """ file open -> tokenizing -> dataclasses"""

    @classmethod
    def negative_sampling(cls, row, label_list:list, label2query:dict, sample_size:int):
        result = []
        random_list = random.choice(
            label_list, size=sample_size, replace=False)
        
        query_text = row[0]
        answer = row[1]
        label = int(row[2])
        sample_inds = [i for i in random_list if int(i) != label]
        
        for sample_ind in sample_inds:
            result.append([query_text, answer, label2query[int(sample_ind)]])
            
        return result


    @classmethod
    def load_data(cls, data_path:str, label_list, label2query, sample_size:int, header:bool=True, is_negativeSample:bool=False) -> List:
        """  Object: [sentence1 | sentence2 | sentence3] 
             sentence1 <----> sentence2 should be Positive
             sentence1 <----> sentence3 should be Negative
        """     
        dataset = []
        dataset.append(['query', 'answer' ,'query_label', 'answer_label'])

        if isinstance(data_path, str):      
            data_path = [data_path]

        for train_file in data_path:
            with open(train_file, 'r') as file:
                for i, row in enumerate(file):
                    if header and i==0:
                        continue

                    row = row.strip().split('\t')
                    if len(row) < 3:
                        continue

                    if is_negativeSample:
                        sampled_data = cls.negative_sampling(row=row, label_list=label_list, label2query=label2query, sample_size=sample_size)
                        dataset.extend(sampled_data)
                    else:
                        pass
                        

        return dataset
    

    @classmethod
    def load_data_v2(cls, data_path:str, negative_dict:dict, label_list:list, label2query, sample_size:int, header:bool=True) -> List:
        """  Object: [sentence1 | intent_nm | intent_idx] -> sent1 | sent2 | sent3 by negative sampling"""     
        dataset = []
        dataset.append(['query', 'intent', 'intentidx'])

        if isinstance(data_path, str):      
            data_path = [data_path]

        for train_file in data_path:
            with open(train_file, 'r') as file:
                for i, row in enumerate(file):
                    if header and i==0:
                        continue

                    row = row.strip().split('\t')
                    if len(row) < 3:
                        continue
                    
                    negative_intent_list = negative_dict.get(row[1], [])
                    if negative_intent_list:
                        for negative_sent in negative_intent_list:
                            dataset.append([row[0], row[1], negative_sent])
                    
                    if sample_size > len(negative_intent_list):
                        sampled_data = cls.negative_sampling(row=row, label_list=label_list, label2query=label2query, sample_size=(sample_size - len(negative_intent_list)))
                    
                    dataset += sampled_data

        return dataset
            
    @classmethod
    def preprocess(cls, data_path, label2query, label_list, tokenizer:PreTrainedTokenizer, tokenizer_input: TokenizerInput=None, header:bool=True, sample_size:int=30, negative_dict:dict=None) -> None:
        """ try read tsv file using pandas first if [memory or parse] error catched use other reading method  """
    
        feature_list = list()
        skipped_line = 0

        datasets = cls.load_data_v2(data_path, header=header, label2query=label2query, label_list=label_list, sample_size=sample_size, negative_dict=negative_dict)
        print(f'preprocessing: {len(datasets)}')
        for i, line in tqdm(enumerate(datasets)):
            try:
                if (len(line) < 3) or (i==0):
                    ## skip incomplete data && header
                    skipped_line += 1
                    continue

                a_sentence = line[0]
                b_sentence = line[1]
                c_sentence = line[2]
                a_encoded_sentence = cls.tokenizing(input=a_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)
                b_encoded_sentence = cls.tokenizing(input=b_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)
                c_encoded_sentence = cls.tokenizing(input=c_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)

                feature_list.append(
                    NLIInput(
                        sentence_a = line[0],
                        sentence_b = line[1],
                        sentence_c = line[2], 

                        a_input_ids = a_encoded_sentence.input_ids,
                        a_attention_mask=a_encoded_sentence.attention_mask,

                        b_input_ids=b_encoded_sentence.input_ids,
                        b_attention_mask=b_encoded_sentence.attention_mask,

                        c_input_ids = c_encoded_sentence.input_ids,
                        c_attention_mask=c_encoded_sentence.attention_mask
                    )
                )
            except Exception as e:
                print(f'Error occurs in {i} lines in preprocessing')
                print(line)
                print(e)

        return feature_list

class SingleSentenceProcessor(AbsPreprocessor):

    @classmethod
    def preprocess(cls, tokenizer,  input_list:List) -> None:
        """ try read tsv file using pandas first if [memory or parse] error catched use other reading method  """
    
        feature_list = list()
        skipped_line = 0

        for i, line in enumerate(input_list):
            try:
                a_encoded_sentence = cls.tokenizing(input=line, tokenizer=tokenizer, tokenizer_input=None)
                feature_list.append(
                    SingleSentenceInput(
                        sentence_a = line,
                        a_input_ids = a_encoded_sentence.input_ids,
                        a_attention_mask=a_encoded_sentence.attention_mask,
                    )
                )
            except Exception as e:
                print(f'Error occurs in {i} lines in preprocessing')
                print(line)
                print(e)
                break

        return feature_list


class PreprocessorFactory:
    def __new__(cls, data_type: str) -> AbsPreprocessor:
        if data_type.lower() == 'nli':
            return NliPreprocessor
        elif data_type.lower() == 'sts':
            return Stsprocessor
        elif data_type.lower() == 'triple':
            return Tripleprocessor
        elif data_type.lower() == 'faq':
            return Faqprocessor
        elif data_type.lower() == 'test':
            return SingleSentenceProcessor
        else:
            raise ValueError(f"Invalid model type: {data_type}")


