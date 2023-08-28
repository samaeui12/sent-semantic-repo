from src.dataset.simcse_dataset import Unsup_simcse, Sup_simcse
from src.dataset.sts_dataset import StsDataset

 
DATASET_MAPPING_DICT = {
    'Sup_simcse': Sup_simcse,
    'Unsup_simcse': Unsup_simcse,
    'StsDataset': StsDataset
}