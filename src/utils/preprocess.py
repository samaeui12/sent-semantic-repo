#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import logging
import ast
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from typing import List, Union
from input import NLIInput, TokenizerInput, StsInput

from transformers.tokenization_utils import PreTrainedTokenizer
from utils import AbsPreprocessor

class NliPreprocessor(AbsPreprocessor):
    """ file open -> tokenizing -> dataclasses"""
    # @staticmethod
    # def load_data(data_path:str, save_path:str=None, header:bool=True) -> List:
    #     """  Object: [sentence1 | sentence2 | goldlabel] -> [sentence1 | positive sentence2 | hard negative sentence3]
    #          data description: multinli.train.ko.tsv: senetence1 기준 groubpy 통계: 125323개 (3), 1975: (6), 511(9), 14(12) etc(20)
    #          Question Mark:
    #             if  sentence1:sentence2: P
    #                 sentence1:sentence3: N
    #             then sentence2:sentence3 -> N??
    #     """
    #     dataset = []
    #     dataset.append(['sentence_a', 'sentence_b', 'sentence_c'])
    #
    #     sentences = dict()
    #     if isinstance(data_path, str):
    #         data_path = [data_path]
    #
    #     for train_file in data_path:
    #         with open(train_file, 'r') as file:
    #             for i, row in enumerate(file):
    #                 if header and i==0:
    #                     continue
    #
    #                 line = row.strip().split('\t')
    #                 if len(line) < 3:
    #                     continue
    #
    #                 sentence_a = line[0]
    #                 sentence_b = line[1]
    #                 gold_label = line[2]
    #
    #                 if sentence_a not in sentences:
    #                     """ key:sentence_a -> {'entailment': [sent1, sent2, ...,] , 'contradiction':[]}"""
    #                     sentences[sentence_a] = {}
    #                     sentences[sentence_a]['entailment'] = []
    #                     sentences[sentence_a]['contradiction'] = []
    #                     sentences[sentence_a]['neutral'] = []
    #
    #                 sentences[sentence_a][gold_label] += [sentence_b]
    #
    #     """ make dataset for every combination of entail and contra """
    #     for key, val in sentences.items():
    #         entails = val.get('entailment', [])
    #         contradiction = val.get('contradiction', [])
    #         if entails and contradiction:
    #             for item in list(product(entails, contradiction)):
    #                 dataset.append((key, item[0], item[1]))
    #
    #     if save_path is not None:
    #         if os.path.exists(save_path):
    #             logging.info(f'{save_path}: exists -> removing')
    #             os.remove(save_path)
    #
    #         with open(save_path, 'a') as f:
    #             for data in dataset:
    #                 f.write('\t'.join(data) + '\n')
    #
    #     return dataset

    @staticmethod
    def load_data(data_path: str, save_path: str = None, header: bool = True) -> List:
        dataset = []
        dataset.append(['sentence_a', 'sentence_b', 'sentence_c'])

        # make dataset for mno dataset
        if 'train' in data_path:
            file_num = 19
        else:
            file_num = 5
        for i in range(file_num):
            file_name = 'mno_faq_split_{0:02d}.json'.format(i)
            data_file_path = os.path.join(data_path, file_name)
            with open(data_file_path, 'r') as file:
                for idx, row in enumerate(file):
                    row = ast.literal_eval(row)
                    dataset.append((row[0], row[1], row[2]))

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
        """ try read tsv file using pandas first if [memory or parse] error catched use other reading method """
        feature_list = list()
        skipped_line = 0
        datasets = cls.load_data(data_path, save_path=save_path, header=header)
        for i, line in enumerate(datasets):
            try:
                if (len(line) < 3) or (i == 0):
                    # skip incomplete data & header
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
                        sentence_a=line[0],
                        sentence_b=line[1],
                        sentence_c=line[2],
                        a_input_ids=a_encoded_sentence.input_ids,
                        a_attention_mask=a_encoded_sentence.attention_mask,
                        b_input_ids=b_encoded_sentence.input_ids,
                        b_attention_mask=b_encoded_sentence.attention_mask,
                        c_input_ids=c_encoded_sentence.input_ids,
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


class PreprocessorFactory:
    def __new__(cls, data_type: str) -> AbsPreprocessor:
        if data_type.lower() == 'nli':
            return NliPreprocessor
        elif data_type.lower() == 'sts':
            return Stsprocessor
        elif data_type.lower() == 'triple':
            return Tripleprocessor
        else:
            raise ValueError(f"Invalid model type: {data_type}")


# if __name__ == '__main__':
#
#     # yaml_file = '/app/code/config/config.yml'
#     # parser = YamlParser(yaml_file)
#     # config_dict = parser.parse_recursive()
#
#     # Kornli_path = config_dict['data']['kornli']['path']
#     # Korsts_path = config_dict['data']['korsts']['path']
#
#     # xnli_dev = os.path.join(Kornli_path, 'xnli.dev.ko.tsv')
#     # mnli_train = os.path.join(Kornli_path, 'multinli.train.ko.tsv')
#     # snli_train = os.path.join(Kornli_path, 'snli_1.0_train.ko.tsv')
#
#     # data_path = [mnli_train, snli_train]
#     # preprocessor = PreprocessorFactory(model_type='simcse')
#
#     # is_preprocessed = False
#     # save_path = '/app/data/open_data/preprocess/KorNLI/kornli.tsv'
#     # dataset = preprocessor.preprocess(data_path=data_path, save_path=save_path)
#     # print(dataset[0])
#
#     from transformers import (
#         AdamW,
#         AutoModel,
#         get_linear_schedule_with_warmup,
#         AutoTokenizer,
#     )
#     from model import MODEL_MAPPING_DICT
#
#     model_type = 'klue/bert-base'
#     """ initialize seed """
#     tokenizer = AutoTokenizer.from_pretrained(model_type)
#     # data = preprocessor.build(data_path=data_path, save_path=save_path, tokenizer=tokenizer)
#     # print(data[:3])
#
#     preprocessor = PreprocessorFactory(model_type='diffcse')
#     print(preprocessor)
#     #Korsts_path = '/app/data/open_data/KorSTS'
#     #data_path = [os.path.join(Korsts_path, 'sts-dev.tsv'),  os.path.join(Korsts_path, 'sts-test.tsv'), os.path.join(Korsts_path, 'sts-train.tsv')]
#     #save_path = '/app/data/open_data/preprocess/KorSTS/korsts.tsv'
#     data_path = None
#     save_path = '/app/data/open_data/preprocess/merged/merge.tsv'
#     print(data_path)
#     dataset = preprocessor.preprocess(data_path=data_path, save_path=save_path)
#     print(dataset[0])
#     #data = preprocessor.build(data_path=data_path, save_path=save_path, tokenizer=tokenizer)
#     #print(data[:3])