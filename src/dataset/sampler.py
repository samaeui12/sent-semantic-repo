import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import random

class RandomClassSampler(Sampler):
    def __init__(self, dataset, num_classes, batch_size):
        self.dataset = dataset
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.class_indices = self._get_class_indices()

    def _get_class_indices(self):
        # Organize indices by class
        class_indices = {i: [] for i in range(self.num_classes)}
        for idx in range(len(self.dataset)):
            _, class_label = self.dataset[idx]
            class_indices[class_label].append(idx)
        return class_indices

    def __iter__(self):
        batch = []
        classes = list(self.class_indices.keys())
        while len(batch) < self.batch_size:
            selected_class = random.choice(classes)
            classes.remove(selected_class)
            selected_index = random.choice(self.class_indices[selected_class])
            batch.append(selected_index)
            if len(classes) == 0:
                classes = list(self.class_indices.keys())
        random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return len(self.dataset)