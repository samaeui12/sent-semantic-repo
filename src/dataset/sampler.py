import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler, BatchSampler
import numpy as np
import random
from collections import defaultdict

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
            class_label = self.dataset[idx]['label']
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
    
class BalancedClassSampler(Sampler):
    """Samples elements such that only one sample from each class is chosen, and classes are randomly selected if batch_size < number of classes."""
    def __init__(self, data_source, class_to_idx, batch_size):
        self.data_source = data_source
        self.class_to_idx = class_to_idx
        self.batch_size = batch_size
        self.indices_per_class = defaultdict(list)
        for idx, data in enumerate(data_source):
            label = data['label']
            self.indices_per_class[label].append(idx)
        self.classes = list(self.indices_per_class.keys())

    def __iter__(self):
        while True:
            chosen_classes = random.sample(self.classes, min(len(self.classes), self.batch_size))
            for class_idx in chosen_classes:
                yield random.choice(self.indices_per_class[class_idx])

    def __len__(self):
        return len(self.data_source)

class RandomClassBatchSampler(BatchSampler):
    """Batch Sampler that creates batches with one sample per class."""
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch