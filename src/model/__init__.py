from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.models.bert.configuration_bert import BertConfig
from transformers import BertForSequenceClassification

from .sentence_bert import SentBertModel, SentBertModelConfig
from .pgd_sentence_bert import SentPgdBertModelConfig, SentPgdBertModel
from .sentence_roberta import SentRobertaModel, SentRobertaModelConfig
 
MODEL_MAPPING_DICT = {
    'sent_bert': SentBertModel,
    'sent_pgd_bert': SentPgdBertModel,
    'sent_roberta': SentRobertaModel
}

CONFIG_MAPPING_DICT = {
    'sent_bert' : SentBertModelConfig,
    'sent_pgd_bert' : SentPgdBertModelConfig,
    'sent_roberta' : SentRobertaModelConfig
}

## register custom model to automodel and config
for key_word in MODEL_MAPPING_DICT.keys():
    if key_word in MODEL_MAPPING_DICT and key_word in CONFIG_MAPPING_DICT:
        AutoConfig.register(key_word, CONFIG_MAPPING_DICT[key_word])
        AutoModel.register(CONFIG_MAPPING_DICT[key_word], MODEL_MAPPING_DICT[key_word])