import numpy as np
from typing import Dict
import os
from logger import Experi_Logger
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.cuda.amp as amp
import sys
import copy
from loss import Loss_MAPPING_DICT


from transformers import (
    AdamW,
    AutoModel,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
)
from tqdm import tqdm
from model import MODEL_MAPPING_DICT
from model import CONFIG_MAPPING_DICT
from trainer.abs_trainer import AbstractTrainer
from scipy.stats import pearsonr, spearmanr
from utils import SummaryWriter, print_grad


class SimcseTrainer(AbstractTrainer):
    def __init__(self, args, model=None, loader=None, tokenizer=None, logger=None):
        super(SimcseTrainer, self).__init__(args, model, loader)
        self.tokenizer = tokenizer
        if logger is None:
            self.logging = Experi_Logger(self.args.log_dir)
        else:
            self.logging = logger
        
        self.writer = SummaryWriter(args.experiments_path)
         
    def _create_state_dict(self, epoch):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'train_loss': self.best_train_loss,
            'patience': self.patience,
            'metric': self.metrics
        }  
        
    def initialize_metrics(self, metrics):
        assert (isinstance(metrics, list) or metrics is None), 'Argument metrics is expected to list type'
        if metrics is None:
            return None
        metric_pocket = dict()
        for metric in metrics:
            if 'loss' not in metric:
                metric_pocket[metric] = -np.inf
            else:
                print(metric)
                metric_pocket[metric] = np.inf

        return metric_pocket

    def early_stop(self, current_metric:dict, criterias:list, epoch:int, key:str='spearman'):
        is_early_stop = False

        for criteria in criterias:
            if criteria not in self.metrics.keys():
                raise KeyError(f'invalid {criteria} in {self.metrics.keys()}')
        
        best_metric = self.metrics[key]
        now_metric = current_metric[key]

        if 'loss' in key:
            pass

        else:
            best_metric = -1 * float(best_metric)
            now_metric = -1 * float(now_metric)

        if best_metric <= now_metric:
            self.patience += 1
            self.logging.info(f'patience: {self.patience}')
            if self.patience >= self.patience_limit:
                is_early_stop = True
        
        else:
            """ 초기화 """
            self.patience = 0
            self.logging.info(f'patience initialized to 0')
            self.metrics = copy.deepcopy(current_metric)
            model_save_path = os.path.abspath(self.args.output_dir)
            self.writer.update(self.args, **current_metric)
            model_to_save = (
               self.model.module if hasattr(self.model, "module") else self.model
            ) 
            model_to_save.save_pretrained(model_save_path)
            self.tokenizer.save_pretrained(model_save_path)
            #state_dict = self._create_state_dict(epoch=epoch)

            self.logging.info("save model to [{}]".format(model_save_path))
            for criteria in criterias:
                self.logging.info(f"***** best_{criteria} epoch: {epoch} && value: {current_metric[criteria]}")
            
        return is_early_stop
                    
    def model_setting(self, model_type:str, train_dataset:Dataset, model=None, tokenizer=None):
        if model is None and tokenizer is None:
            self.model = MODEL_MAPPING_DICT[model_type].from_pretrained(
                self.args.pretrained_model, **vars(self.args), 
            )
            """ tokenizer """
            self.logging.info("load pretrained checkpoint from [{}]".format(self.args.pretrained_model))
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)
        else:
            self.model = model
            self.tokenizer = tokenizer

        """ nn.distributed setting Not implemented yet """
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.args.device)

        """ setting decay """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        self.train_dataloader = train_dataset.loader(
            shuffle=True, batch_size=self.args.train_batch_size
        )
        total_step = len(self.train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        self.args.warmup_steps = int(self.args.warmup_percent * total_step)
        
        self.logging.info(f'warm_up_steps: {self.args.warmup_steps}')

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps= total_step
        )   
        loss_fct = Loss_MAPPING_DICT[self.args.loss]
        self.loss_fct = loss_fct(margin=self.args.margin, temperature=self.args.temperature)
        self.metrics = self.initialize_metrics(metrics=['spearman', 'pearson', 'train_loss', 'val_loss'])
        
    def cal_loss(self, batch):

        batch = {key: (item.to(self.args.device) if type(item) == torch.Tensor else item) for key, item in batch.items()}
        a_embedding = self.model(batch['a_input_ids'], batch['a_attention_mask'])
        b_embedding = self.model(batch['b_input_ids'], batch['b_attention_mask'])
        c_embedding = self.model(batch['c_input_ids'], batch['c_attention_mask'])
        a_norm = a_embedding / a_embedding.norm(dim=1)[:, None]
        b_norm = b_embedding / b_embedding.norm(dim=1)[:, None]
        c_norm = c_embedding / c_embedding.norm(dim=1)[:, None]
        final_loss = self.loss_fct(a_norm, b_norm, c_norm, label=batch['labels'])
        
        return final_loss

    def train(self, model_type, train_dataset, val_dataset, model=None, tokenizer=None) -> Dict[str, float]:
        global_step = 0
        self.model_setting(model_type=model_type, train_dataset=train_dataset, model=model, tokenizer=tokenizer)

        """ print requires grad in model for debuging """
        self.requires_grad_list = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.requires_grad_list.append(name)
        
        if self.args.amp_use:
            self.scaler = amp.GradScaler()
               
        for i in tqdm(range(self.args.num_train_epochs)):
            if i !=0 and global_step > self.args.warmup_steps:
                self.scheduler.step()

            global_step, train_loss = self.train_one_epoch(epoch=i, global_step=global_step)

            self.logging.info(f'epoch: {i}, train_loss: {train_loss}')
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss 

            eval_result = self.validate(val_dataset, epoch=i)
            is_early_stop = self.early_stop(eval_result, criterias=['spearman', 'val_loss'], epoch=i, key=self.args.metric)
            if is_early_stop:
                break

    def train_one_epoch(self, epoch, global_step):
        
        self.model.train()
        self.optimizer.zero_grad()

        train_losses = []
        accumulation_steps = 0

        for batch_idx, batch in enumerate(tqdm(self.train_dataloader)): 

            batch = {key: (item.to(self.args.device) if type(item) == torch.Tensor else item) for key, item in batch.items()}            
            if self.args.amp_use:
                with amp.autocast():
                    final_loss = self.cal_loss(batch=batch)

                if self.args.n_gpu > 1:
                    final_loss = final_loss.mean()
                    print(f'final_loss: {final_loss}')
                self.scaler.scale(final_loss).backward()
                # Update the gradient accumulation counter
                accumulation_steps += 1
                # Only perform optimizer step, gradient clipping, and zero gradients after the specified number of accumulation steps
                if accumulation_steps % self.args.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                
                final_loss = self.cal_loss(batch=batch)
                if self.args.n_gpu > 1:
                    final_loss = final_loss.mean()
                    print(f'final_loss: {final_loss}')
                            
                final_loss.backward()
                # Update the gradient accumulation counter
                accumulation_steps += 1
                # Only perform optimizer step, gradient clipping, and zero gradients after the specified number of accumulation steps
                if accumulation_steps % self.args.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()  # Move this line here
        
            train_losses.append(final_loss.detach().cpu().item())
            
            global_step += 1   
            ## logging
            if (self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0 and global_step > 0):
                logging_str =  "***** epoch [{}]".format(epoch)
                logging_str += " global_step [{}]".format(global_step)
                logging_str += " {} [{:.4}]".format('loss', final_loss.detach().cpu().item())
                self.logging.info(logging_str)
            
        final_loss = np.mean(train_losses)

        return global_step, final_loss
        
    def validate(self, val_dataset, epoch) -> Dict[str, float]:
        """ evaluate using STS dataset """
        val_dataloader = val_dataset.loader(
            shuffle=False, batch_size=self.args.eval_batch_size
        )
        loss_fct = nn.MSELoss()
        
        self.logging.info("***** Running evaluation [{}] *****".format(epoch))
        
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = []
        labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating", leave=False):
                batch = {key: (item.to(self.args.device) if type(item) == torch.Tensor else item) for key, item in batch.items()}
                with torch.no_grad():
                    ## a sentence
                    a_embedding = self.model(batch['a_input_ids'], batch['a_attention_mask'])
                    b_embedding = self.model(batch['b_input_ids'], batch['b_attention_mask'])

                    similarity = torch.cosine_similarity(a_embedding, b_embedding)

                    ## regression(MSE)
                    loss = loss_fct(similarity, batch['labels'].view(-1))
                    eval_loss += loss.detach().cpu().item()

                    preds.append(similarity.cpu().numpy().reshape(-1))
                    labels.append(batch['labels'].cpu().numpy().reshape(-1))

                nb_eval_steps += 1

            eval_loss = eval_loss/nb_eval_steps

            preds = np.concatenate(preds)
            labels = np.concatenate(labels)

            pearson_corr, _ = pearsonr(preds, labels)
            spearman_corr, _ = spearmanr(preds, labels)

            results = {
                'val_loss': eval_loss,
                'pearson': pearson_corr,
                'spearman': spearman_corr,
                'epoch': epoch
            }
            
            for key, value in results.items():
                self.logging.info("  %s = %s", key, str(value))

            return results
