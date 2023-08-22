import logging
import os
from datetime import datetime
import torch
import numpy as np
import random
import sys

def set_seed(seed):
    """
        Freeze all random seed to train model with consistency
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def reset_logging(output_dir='temp'):
    """
        logging settings
    """
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)
    root.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'execute.log')
    if os.path.exists(log_path):
        os.remove(log_path)
        
    logging.basicConfig(filename=log_path, level=logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)