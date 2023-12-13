from .simcse_dataset import Unsup_simcse, Sup_simcse
from .inference_dataset import InferenceDataset
from .sts_dataset import StsDataset
from .inference_dataset import InferenceDataset
from .sampler import RandomClassSampler, BalancedClassSampler, RandomClassBatchSampler

 
DATASET_MAPPING_DICT = {
    'Sup_simcse': Sup_simcse,
    'Unsup_simcse': Unsup_simcse,
    'StsDataset': StsDataset,
    'InferenceDataset': InferenceDataset 
}