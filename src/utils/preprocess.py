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
from input import SimcseInput, TokenizerInput, StsInput
import dataclasses
from itertools import product

## transformers library import
from transformers.tokenization_utils import PreTrainedTokenizer
from utils import AbsPreprocessor

class DiffcsePreprocessor(AbsPreprocessor):
    @classmethod
    def input_restrict(cls, line):
        if (line == '') or (line is None):
            return None
        else:
            splitted_line = line.split(' ')
            if len(splitted_line) <= 2:
                return None
            else:
                """ 중복 처리를 위한 정규화 로직 실행 """
                # Standardize each sentence and remove spaces and punctuation
                standardized_line = re.sub(r'\s+|[.,]', '', line.lower())
                return standardized_line
   
    @classmethod
    def preprocess(cls, data_path:Union[str, List]=None, save_path:str=None, header:bool=True) -> List:
        """ ideation 
            1. answer, question을 Entailment로 엮을 것인가?
            2. unsup 방식으로 바라 볼 것인가.
        
        """
        """ Use Kornli (Supervised) + own dataset (Unsup) """
        
        if data_path is None:
            data_path = ['/app/data/skt_data/answer/answer.json', 
                        '/app/data/skt_data/question/question.json', 
                        '/app/data/skt_data/search_log/query.json', 
                        '/app/data/open_data/KorNLI/multinli.train.ko.tsv', 
                        '/app/data/open_data/KorNLI/snli_1.0_train.ko.tsv'
                        ]

        else:
            if isinstance(data_path, str):      
                data_path = [data_path]
        
        sentences = dict()       
        dataset = []

        for train_file in data_path:
            if 'answer' in train_file:
                continue
            
            if train_file.endswith('json'):
                import json
                if train_file =='/app/data/skt_data/question/question.json':
                    key = 'question'
                elif train_file =='/app/data/skt_data/answer/answer.json':
                    key = 'answer'
                elif train_file =='/app/data/skt_data/search_log/query.json':
                    key = 'text'
                else:
                    print(f'data_path: {train_file}')
                    raise NotImplementedError
                
                data_set = set()
                with open(train_file, 'r') as file:
                    for i, row in enumerate(file):
                        line = json.loads(row)
                        id = line['id']
                        data = line[key]
                        st_data = cls.input_restrict(data)
                        if st_data is None:
                            continue
                        data_set.add(st_data)
                
                for line in data_set:
                    dataset.append((line, line, ''))
                
            elif train_file.endswith('tsv'):
                with open(train_file, 'r') as file:
                    for i, row in enumerate(file):
                        if header and i==0:
                            header = row.strip().split('\t')
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
                    else:
                        continue
            else:
                raise NotImplementedError('only tsv && json supported')
            
        if save_path is None:
            return dataset
        
        else:
            if os.path.exists(save_path):
                logging.info(f'{save_path}: exists -> removing')
                os.remove(save_path)

            with open(save_path, 'a') as f:
                f.write('\t'.join(['sentence_a', 'sentence_b', 'sentence_c'])+'\n')
                for data in dataset:
                    f.write('\t'.join(data) + '\n')
            return dataset
        
    @classmethod
    def build(cls, data_path, tokenizer:PreTrainedTokenizer, save_path, tokenizer_input: TokenizerInput=None, header:bool=True, is_preprocessed:bool=False) -> None:
        """ try read tsv file using pandas first if [memory or parse] error catched use other reading method  """
        feature_list = list()
        skipped_line = 0
        if not is_preprocessed:
            datasets = cls.preprocess(data_path, save_path=save_path, header=header)
            header = False
        else:
            datasets = open(save_path, 'r')
            header = True

        for i, line in enumerate(datasets):
            try:
                line = line.strip().split('\t')
                if len(line) == 3:
                    a_sentence = line[0]
                    b_sentence = line[1]
                    c_sentence = line[2]

                elif len(line) ==2:
                    a_sentence = line[0]
                    b_sentence = line[1]
                    c_sentence = None

                else:
                    continue
                
                
                a_encoded_sentence = cls.tokenizing(input=a_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)
                b_encoded_sentence = cls.tokenizing(input=b_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)
                c_encoded_sentence = cls.tokenizing(input=c_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)

                if c_encoded_sentence is not None:
                    feature_list.append(
                        DiffcsePreprocessor(
                            sentence_a = a_sentence,
                            sentence_b = b_sentence,
                            sentence_c = c_sentence, 

                            a_input_ids = a_encoded_sentence.input_ids,
                            a_attention_mask=a_encoded_sentence.attention_mask,

                            b_input_ids=b_encoded_sentence.input_ids,
                            b_attention_mask=b_encoded_sentence.attention_mask,

                            c_input_ids = c_encoded_sentence.input_ids,
                            c_attention_mask=c_encoded_sentence.attention_mask
                        )
                    )
                else:
                    feature_list.append(
                        DiffcsePreprocessor(
                            sentence_a = a_sentence,
                            sentence_b = b_sentence,
                            sentence_c = c_sentence, 

                            a_input_ids = a_encoded_sentence.input_ids,
                            a_attention_mask=a_encoded_sentence.attention_mask,

                            b_input_ids=b_encoded_sentence.input_ids,
                            b_attention_mask=b_encoded_sentence.attention_mask,
                        )
                    )

            except Exception as e:
                print(f'Error occurs in {i} lines in preprocessing')
                print(line)
                print(e)

        if is_preprocessed:
            datasets.close()
        return feature_list

    @classmethod         
    def tokenizing(cls, input, tokenizer:PreTrainedTokenizer, tokenizer_input:TokenizerInput=None):
        if input is None:
            return None
        else:
            if tokenizer_input is None:
                return tokenizer.encode_plus(input)
            else:
                kwargs = dataclasses.asdict(tokenizer_input)
                return tokenizer.encode_plus(input, **kwargs)

class SimcsePreprocessor(AbsPreprocessor):
    """ file open -> tokenizing -> dataclasses"""
    @staticmethod
    def preprocess(data_path:str, save_path:str=None, header:bool=True) -> List:
        """  Object: [sentence1 | sentence2 | goldlabel] -> [sentence1 | positive sentence2 | hard negative sentence3]  
             data description: multinli.train.ko.tsv: senetence1 기준 groubpy 통계: 125323개 (3), 1975: (6), 511(9), 14(12) etc(20)   
             Question Mark: 
                if  sentence1:sentence2:P
                    sentence1:sentence3:N
                then sentence2:sentence3 -> N??  
        """     
        sentences = dict()       
        if isinstance(data_path, str):      
            data_path = [data_path]

        for train_file in data_path:
            with open(train_file, 'r') as file:
                for i, row in enumerate(file):
                    if header and i==0:
                        header = row.strip().split('\t')
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

        dataset = []
        """ make dataset for every combination of entail and contra """
        for key, val in sentences.items():
            entails = val.get('entailment', [])
            contradiction = val.get('contradiction', [])
            if entails and contradiction:
                for item in list(product(entails, contradiction)):
                    dataset.append((key, item[0], item[1]))                
            else:
                continue

        if save_path is None:
            return dataset
        
        else:
            if os.path.exists(save_path):
                logging.info(f'{save_path}: exists -> removing')
                os.remove(save_path)

            with open(save_path, 'a') as f:
                f.write('\t'.join(['sentence_a', 'sentence_b', 'sentence_c'])+'\n')
                for data in dataset:
                    f.write('\t'.join(data) + '\n')
            return dataset
            
    @classmethod
    def build(cls, data_path, tokenizer:PreTrainedTokenizer, save_path, tokenizer_input: TokenizerInput=None, header:bool=True, is_preprocessed:bool=False) -> None:
        """ try read tsv file using pandas first if [memory or parse] error catched use other reading method  """
        feature_list = list()
        skipped_line = 0
        if not is_preprocessed:
            datasets = cls.preprocess(data_path, save_path=save_path, header=header)
            header = False
        else:
            datasets = open(save_path, 'r')
            header = True

        for i, line in enumerate(datasets):
            try:
                if is_preprocessed:
                    line = line.strip().split('\t')

                if len(line) < 3:
                    skipped_line += 1
                    continue
                a_sentence = line[0]
                b_sentence = line[1]
                c_sentence = line[2]
                a_encoded_sentence = cls.tokenizing(input=a_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)
                b_encoded_sentence = cls.tokenizing(input=b_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)
                c_encoded_sentence = cls.tokenizing(input=c_sentence, tokenizer=tokenizer, tokenizer_input=tokenizer_input)

                feature_list.append(
                    SimcseInput(
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

        if is_preprocessed:
            datasets.close()
        return feature_list

    @classmethod         
    def tokenizing(cls, input, tokenizer:PreTrainedTokenizer, tokenizer_input:TokenizerInput=None):
        if tokenizer_input is None:
            return tokenizer.encode_plus(input)
        else:
            kwargs = dataclasses.asdict(tokenizer_input)
            return tokenizer.encode_plus(input, **kwargs)
    
class Stsprocessor(AbsPreprocessor):
    """ file open -> tokenizing -> dataclasses"""

    @classmethod
    def preprocess(cls, data_path:Union[str, List], save_path:str=None, header:bool=True) -> List:
        if isinstance(data_path, str):
            data_path = [data_path]

        dataset = []

        for train_file in data_path:
            with open(train_file, 'r') as file:
                for i, row in enumerate(file):
                    if header and i==0:
                        header = row.strip().split('\t')
                        continue
                    
                    line = row.strip().split('\t')

                    label = float(line[-3])
                    a_sentence = line[-2]
                    b_sentence = line[-1]
                    
                    dataset.append([a_sentence, b_sentence, str(label)])
            
            if save_path is None:
                return dataset
            
            else:
                if os.path.exists(save_path):
                    logging.info(f'{save_path}: exists -> removing')
                    os.remove(save_path)

                with open(save_path, 'a') as f:
                    f.write('\t'.join(['sentence_a', 'sentence_b', 'label'])+'\n')
                    for data in dataset:
                        f.write('\t'.join(data) + '\n')
                return dataset


    @classmethod
    def build(cls, data_path, save_path, tokenizer: PreTrainedTokenizer, tokenizer_input:TokenizerInput=None, header=True, is_preprocessed:bool=False):

        if not is_preprocessed:
            datasets = cls.preprocess(data_path, save_path=save_path, header=header)
            header = False
        else:
            datasets = open(save_path, 'r')
            header = True

        feature_list = list()
        for i, line in enumerate(datasets):
            if header and i==0:
                header = line.strip().split('\t')
                continue
            
            if is_preprocessed:
                line = line.strip().split('\t')

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

        if is_preprocessed:
            datasets.close()

        return feature_list
    
    @classmethod         
    def tokenizing(cls, input, tokenizer:PreTrainedTokenizer, tokenizer_input:TokenizerInput=None):
        if tokenizer_input is None:
            return tokenizer.encode_plus(input)
        else:
            kwargs = dataclasses.asdict(tokenizer_input)
            return tokenizer.encode_plus(input, **kwargs)


class PreprocessorFactory:
    def __new__(cls, model_type: str) -> AbsPreprocessor:
        if model_type.lower() == 'simcse':
            return SimcsePreprocessor
        elif model_type.lower() =='diffcse':
            return DiffcsePreprocessor
        elif model_type.lower() == 'sts':
            return Stsprocessor
        else:
            raise ValueError(f"Invalid model type: {model_type}")


if __name__ == '__main__':

    # yaml_file = '/app/code/config/config.yml'
    # parser = YamlParser(yaml_file)
    # config_dict = parser.parse_recursive()

    # Kornli_path = config_dict['data']['kornli']['path']
    # Korsts_path = config_dict['data']['korsts']['path']

    # xnli_dev = os.path.join(Kornli_path, 'xnli.dev.ko.tsv')
    # mnli_train = os.path.join(Kornli_path, 'multinli.train.ko.tsv')
    # snli_train = os.path.join(Kornli_path, 'snli_1.0_train.ko.tsv')

    # data_path = [mnli_train, snli_train]
    # preprocessor = PreprocessorFactory(model_type='simcse')

    # is_preprocessed = False
    # save_path = '/app/data/open_data/preprocess/KorNLI/kornli.tsv'
    # dataset = preprocessor.preprocess(data_path=data_path, save_path=save_path)
    # print(dataset[0])

    from transformers import (
        AdamW,
        AutoModel,
        get_linear_schedule_with_warmup,
        AutoTokenizer,
    )
    from model import MODEL_MAPPING_DICT
    
    model_type = 'klue/bert-base'
    """ initialize seed """
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    # data = preprocessor.build(data_path=data_path, save_path=save_path, tokenizer=tokenizer)
    # print(data[:3])

    preprocessor = PreprocessorFactory(model_type='diffcse')
    print(preprocessor)
    #Korsts_path = '/app/data/open_data/KorSTS'
    #data_path = [os.path.join(Korsts_path, 'sts-dev.tsv'),  os.path.join(Korsts_path, 'sts-test.tsv'), os.path.join(Korsts_path, 'sts-train.tsv')]
    #save_path = '/app/data/open_data/preprocess/KorSTS/korsts.tsv'
    data_path = None
    save_path = '/app/data/open_data/preprocess/merged/merge.tsv'
    print(data_path)
    dataset = preprocessor.preprocess(data_path=data_path, save_path=save_path)
    print(dataset[0])
    #data = preprocessor.build(data_path=data_path, save_path=save_path, tokenizer=tokenizer)
    #print(data[:3])