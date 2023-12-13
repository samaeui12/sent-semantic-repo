#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import *
from torch.utils.data import (
    DataLoader, Dataset
)
from torch.utils.data.distributed import DistributedSampler


class AbstractDataloader(metaclass=ABCMeta):
    @abstractclassmethod
    def code(cls):
        """dataset code"""

    def get_pytorch_dataloaders(self, data, status =['train', 'val', 'test']):
        loader = dict()
        for stat in status:
            if stat =='train':
                loader[stat] = self._get_train_loader(data['train'])
            elif stat =='val':
                loader[stat] = self._get_val_loader(data['val'])
            elif stat =='test':
                loader[stat] = self._get_test_loader(data['test'])
        return loader

    def collater(self):
        """ customized collate_fn function for dataloader"""
        return None
    
    @abstractmethod
    def _get_train_dataset(self):
        """ logic for return train dataset """
    
    @abstractmethod
    def _get_val_dataset(self):
        """ logic for return val dataset """
    
    @abstractmethod
    def _get_test_dataset(self):
        """ logic for return test dataset """


    def _get_train_loader(self):
        """ get train loader """
        collater_fn = self.collater()
        dataset = self._get_train_dataset()

        if self.args.acclerator =='ddp': 
            train_sampler = DistributedSampler(dataset, shuffle=True)
            train_sampler.set_epoch(self.args.epoch)
            dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, collate_fn=collater_fn,
                                            shuffle=False, pin_memory=True, sampler=train_sampler, num_workers=self.args.num_workers)
        else:
            dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, collate_fn=collater_fn, 
                                            shuffle=True, pin_memory=True)
        return dataloader

    @abstractmethod
    def _get_loader(self, mode:str):
        if mode.lower() == 'train':
            return self._get_train_loader()
        elif mode.lower() == 'val':
            return self._get_eval_loader()
        elif mode.lower() == 'test':
            return self._get_test_loader()
        else:
            raise NotImplementedError('[train, val, test] occurs')

    def _get_eval_loader(self, mode:str):
        """ get eval / test loader """
        collater_fn = self.collater()
        if mode.lower() == 'val':
            dataset = self._get_val_dataset()
        elif mode.lower() == 'test':
            dataset = self._get_test_dataset()
        else:
            raise NotImplementedError('[train, val, test] occurs')

        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size

        if self.args.acclerator =='ddp':
            val_sampler = DistributedSampler(dataset, shuffle=False)
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=False,
                                            shuffle=False, pin_memory=True, sampler=val_sampler, num_workers=self.args.num_workers, collate_fn=collater_fn)
        else:
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=False,
                                            shuffle=False, pin_memory=True, collate_fn=collater_fn)
        return dataloader