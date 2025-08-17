import sys
import json
import os
import time
from datetime import datetime
import math
import sys
import hydra

# Set CUDA device - use GPU with most available memory to avoid multi-GPU conflicts
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 1 has most free memory (81089 MiB)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import colorama
from colorama import Back, Fore, Style
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import wandb
import torch.multiprocessing as mp

from utils.core_utils import (
    get_data_collator,
    get_dataset,
    load_model,
    set_seed,
    set_worker_seed,
    get_optimizer,
    get_scheduler,
    BinaryClassificationMetric,
    EarlyStopping
)



# Initialize log_path globally (will be updated in main())
log_path = None


class Trainer():
    def __init__(self,
                 cfg: DictConfig):
        self.cfg = cfg
        
        self.device = 'cuda'
        self.evaluator = BinaryClassificationMetric(self.device)
        self.type = cfg.type
        self.model_name = cfg.model
        self.dataset_name = cfg.dataset
        self.batch_size = cfg.batch_size
        self.num_epoch = cfg.num_epoch
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.save_path = log_path
        self.global_step = 0  # Add global step counter that persists across epochs
        
        if cfg.type == '5-fold':
            self.dataset_range = [1, 2, 3, 4, 5]
        elif cfg.type == 'temporal':
            self.dataset_range = ['temporal']
        else:
            raise ValueError('experiment type not supported')
        
        self.collator = get_data_collator(cfg.model, cfg.dataset, **cfg.data)
    
    def _reset(self, cfg, fold, type):
        drop_last = True if cfg.model == 'CAFE' else False
        train_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='train', **cfg.data)
        test_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='test', **cfg.data)
        test_collator = self.collator
        self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=True, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed), drop_last=drop_last)
        self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=test_collator, num_workers=min(32, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed), drop_last=drop_last)
        if type == 'temporal':
            valid_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='valid', **cfg.data)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed), drop_last=drop_last)

        # Print label distribution right before training starts
        logger.info(f"{Fore.CYAN}=== FINAL DATASET LABEL DISTRIBUTION (before training) ===")
        logger.info(f"{Fore.CYAN}Configuration: include_piyao={cfg.data.get('include_piyao', False)}, ablation_no_cot={cfg.data.get('ablation_no_cot', False)}")
        
        if hasattr(train_dataset, 'data'):
            train_labels = train_dataset.data['label'].value_counts().sort_index()
            logger.info(f"{Fore.CYAN}TRAIN set - Label distribution: {dict(train_labels)} (Total: {len(train_dataset.data)})")
        
        if hasattr(test_dataset, 'data'):
            test_labels = test_dataset.data['label'].value_counts().sort_index()
            logger.info(f"{Fore.CYAN}TEST set - Label distribution: {dict(test_labels)} (Total: {len(test_dataset.data)})")
        
        if type == 'temporal' and hasattr(valid_dataset, 'data'):
            valid_labels = valid_dataset.data['label'].value_counts().sort_index()
            logger.info(f"{Fore.CYAN}VALID set - Label distribution: {dict(valid_labels)} (Total: {len(valid_dataset.data)})")
        
        logger.info(f"{Fore.CYAN}Label mapping: 0=Real/真/辟谣, 1=Fake/假")
        logger.info(f"{Fore.CYAN}==========================================================")

        steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
        # Sync ablation parameters from data config to model parameters
        model_params = dict(cfg.para)
        if 'ablation_no_cot' in cfg.data:
            model_params['ablation_no_cot'] = cfg.data.ablation_no_cot
        if 'ablation_no_vision' in cfg.data:
            model_params['ablation_no_vision'] = cfg.data.ablation_no_vision
        
        # Handle cross-modal configuration
        if 'cross_modal' in model_params:
            # Always remove individual cross_modal params from model_params
            cross_modal_params = ['cross_modal_layers', 'cross_modal_heads', 'cross_modal_dim_ff', 'cross_modal_dropout']
            
            if model_params['cross_modal']:
                # If cross_modal is True, create config dict
                cross_modal_config = {
                    'n_layers': model_params.get('cross_modal_layers', 4),
                    'n_heads': model_params.get('cross_modal_heads', 8),
                    'd_ff': model_params.get('cross_modal_dim_ff', 1024),
                    'dropout': model_params.get('cross_modal_dropout', 0.1)
                }
                model_params['cross_modal_config'] = cross_modal_config
            
            # Remove individual params regardless of cross_modal value
            for key in cross_modal_params:
                model_params.pop(key, None)
            
        self.model = load_model(cfg.model, **model_params)
        self.model.to(self.device)
        
        # Print full model parameters after initialization (will show after first forward pass)
        from model.ExMRD import print_full_model_params
        print_full_model_params(self.model)
        
        self.optimizer = get_optimizer(self.model, **dict(cfg.opt))
        num_epoch = cfg.num_epoch
        self.scheduler = get_scheduler(self.optimizer, steps_per_epoch=steps_per_epoch, num_epoch=num_epoch, **dict(cfg.sche))
        self.earlystopping = EarlyStopping(patience=cfg.patience, path=self.save_path/'best_model.pth')
        self._first_forward_done = False
        
    def run(self):
        acc_list, f1_list, prec_list, rec_list = [], [], [], []
        f1_real_list, f1_fake_list = [], []
        prec_real_list, prec_fake_list = [], []
        rec_real_list, rec_fake_list = [], []
        
        for fold in self.dataset_range:
            self._reset(self.cfg, fold, self.type)
            logger.info(f'Current fold: {fold}')
            for epoch in range(self.num_epoch):
                logger.info(f'Current Epoch: {epoch}')
                self._train()
                if self.type == 'temporal':
                    self._valid(split='valid', use_earlystop=True)
                elif self.type == '5-fold':
                    self._valid(split='test', use_earlystop=True)
                if self.earlystopping.early_stop:
                    logger.info(f"{Fore.GREEN}Early stopping at epoch {epoch}")
                    break
                if self.type == 'temporal':
                    self._valid(split='test')
            logger.info(f'{Fore.RED}Best of Acc in fold {fold}:')
            self.model.load_state_dict(torch.load(self.save_path/'best_model.pth', weights_only=False))
            best_metrics = self._valid(split='test', final=True)
            
            # Collect macro metrics
            acc_list.append(best_metrics['acc'])
            f1_list.append(best_metrics['f1'])
            prec_list.append(best_metrics['prec'])
            rec_list.append(best_metrics['rec'])
            
            # Collect per-class metrics
            f1_real_list.append(best_metrics['f1_real'])
            f1_fake_list.append(best_metrics['f1_fake'])
            prec_real_list.append(best_metrics['prec_real'])
            prec_fake_list.append(best_metrics['prec_fake'])
            rec_real_list.append(best_metrics['rec_real'])
            rec_fake_list.append(best_metrics['rec_fake'])
            
        logger.info(f"{Fore.MAGENTA}=== FINAL RESULTS SUMMARY ===")
        logger.info(f"{Fore.MAGENTA}Macro Metrics: Acc: {np.mean(acc_list):.4f}, F1: {np.mean(f1_list):.4f}, Precision: {np.mean(prec_list):.4f}, Recall: {np.mean(rec_list):.4f}")
        logger.info(f"{Fore.MAGENTA}Real(0) Metrics: F1: {np.mean(f1_real_list):.4f}, P: {np.mean(prec_real_list):.4f}, R: {np.mean(rec_real_list):.4f}")
        logger.info(f"{Fore.MAGENTA}Fake(1) Metrics: F1: {np.mean(f1_fake_list):.4f}, P: {np.mean(prec_fake_list):.4f}, R: {np.mean(rec_fake_list):.4f}")
        logger.info(f"{Fore.MAGENTA}===============================")
        
    def _train(self):
        loss_list =  []
        self.model.train()
        pbar = tqdm(self.train_dataloader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}")
        for batch in pbar:
            _ = batch.pop('vids')
            
            # Separate tensor and non-tensor inputs
            inputs = {}
            for key, value in batch.items():
                if hasattr(value, 'to'):  # Check if it's a tensor
                    inputs[key] = value.to(self.device)
                else:  # Keep non-tensor values as is (like boolean flags)
                    inputs[key] = value
                    
            labels = inputs.pop('labels')
            
            if hasattr(self.model, 'name') and self.model.name == 'CAFE':
                output = self.model(**inputs, task='similarity')
                loss = self.model.calculate_loss(*output)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            output = self.model(**inputs)
            
            # Print full model parameters after first forward pass initializes lazy layers
            if not self._first_forward_done:
                try:
                    from model.ExMRD import print_full_model_params
                    print_full_model_params(self.model)
                except ImportError:
                    logger.info("ExMRD print_full_model_params not available, skipping parameter count")
                self._first_forward_done = True
            
            # Handle evidential model outputs
            if hasattr(self.model, 'name') and self.model.name == 'ExMRD_Evidential':
                # Check if using evidential or traditional classifier
                if hasattr(self.model, 'use_evidential') and self.model.use_evidential:
                    # Evidential classifier: output is a dict with alpha values
                    losses_dict = self.model.compute_losses(output, labels, self.global_step)
                    loss = losses_dict['loss_total']
                    
                    # Use fused predictions for evaluation
                    pred = output['p_fused']  # Predictive probabilities from fused evidence
                    
                    # Log evidential losses
                    if self.global_step % 100 == 0:
                        logger.debug(f"Evidential losses - Total: {loss.item():.4f}, Fused: {losses_dict['loss_fused'].item():.4f}, "
                                   f"Text: {losses_dict['loss_text'].item():.4f}, Audio: {losses_dict['loss_audio'].item():.4f}, "
                                   f"Image: {losses_dict['loss_image'].item():.4f}, KL_weight: {losses_dict['kl_weight'].item():.3f}")
                else:
                    # Traditional classifier: output is logits
                    losses_dict = self.model.compute_losses(output, labels, self.global_step)
                    loss = losses_dict['loss_total']
                    
                    # Convert logits to probabilities for evaluation
                    pred = F.softmax(output, dim=1)
                    
                    # Log traditional losses
                    if self.global_step % 100 == 0:
                        logger.debug(f"Traditional loss - Total: {loss.item():.4f}")
            else:
                # Regular model handling
                if type(output) is dict:
                    pred = output['pred']
                else:
                    pred = output
                
                loss = None
                if hasattr(self.model, 'name'):
                    match self.model.name:
                        case _:
                            loss = F.cross_entropy(pred, labels)
                else:
                    loss = F.cross_entropy(pred, labels)

            _, preds = torch.max(pred, 1)
            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1
            
        metrics = self.evaluator.compute()
        # print
        logger.info(f"{Fore.BLUE}Train: Loss: {np.mean(loss_list)}")
        logger.info(f'{Fore.BLUE}Train: Acc: {metrics["acc"]:.4f}, Macro F1: {metrics["f1"]:.4f}, Macro Precision: {metrics["prec"]:.4f}, Macro Recall: {metrics["rec"]:.4f}')
        logger.info(f'{Fore.BLUE}Train Per-class: Real(0) F1: {metrics["f1_real"]:.4f}, P: {metrics["prec_real"]:.4f}, R: {metrics["rec_real"]:.4f} | Fake(1) F1: {metrics["f1_fake"]:.4f}, P: {metrics["prec_fake"]:.4f}, R: {metrics["rec_fake"]:.4f}')
    
    def _valid(self, split, use_earlystop=False, final=False):
        loss_list = []
        self.model.eval()
        if split == 'valid' and final:
            raise ValueError('print_wrong only support test split')
        if split == 'valid':
            dataloader = self.valid_dataloader
            split_name = 'Valid'
            fcolor = Fore.YELLOW
        elif split == 'test':
            dataloader = self.test_dataloader
            split_name = 'Test'
            fcolor = Fore.RED
        else:
            raise ValueError('split not supported')
        for batch in tqdm(dataloader, bar_format=f"{fcolor}{{l_bar}}{{bar}}{{r_bar}}"):
            vids = batch.pop('vids')
            
            # Separate tensor and non-tensor inputs
            inputs = {}
            for key, value in batch.items():
                if hasattr(value, 'to'):  # Check if it's a tensor
                    inputs[key] = value.to(self.device)
                else:  # Keep non-tensor values as is (like boolean flags)
                    inputs[key] = value
                    
            labels = inputs.pop('labels')
        
            with torch.no_grad():
                output = self.model(**inputs)
                
                # Handle evidential model outputs
                if hasattr(self.model, 'name') and self.model.name == 'ExMRD_Evidential':
                    # Check if using evidential or traditional classifier
                    if hasattr(self.model, 'use_evidential') and self.model.use_evidential:
                        # Evidential classifier: use fused predictions
                        pred = output['p_fused']  # Predictive probabilities from fused evidence
                        losses_dict = self.model.compute_losses(output, labels)
                        loss = losses_dict['loss_total']
                    else:
                        # Traditional classifier: output is logits
                        losses_dict = self.model.compute_losses(output, labels)
                        loss = losses_dict['loss_total']
                        pred = F.softmax(output, dim=1)
                else:
                    # Regular model handling
                    if type(output) is dict:
                        pred = output['pred']
                    else:
                        pred = output
                    loss = F.cross_entropy(pred, labels)
            
            _, preds = torch.max(pred, 1)

            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())
        metrics = self.evaluator.compute()
        
        logger.info(f"{fcolor}{split_name}: Loss: {np.mean(loss_list)}")
        logger.info(f"{fcolor}{split_name}: Acc: {metrics['acc']:.4f}, Macro F1: {metrics['f1']:.4f}, Macro Precision: {metrics['prec']:.4f}, Macro Recall: {metrics['rec']:.4f}")
        logger.info(f"{fcolor}{split_name} Per-class: Real(0) F1: {metrics['f1_real']:.4f}, P: {metrics['prec_real']:.4f}, R: {metrics['rec_real']:.4f} | Fake(1) F1: {metrics['f1_fake']:.4f}, P: {metrics['prec_fake']:.4f}, R: {metrics['rec_fake']:.4f}")
        
        if use_earlystop:
            self.earlystopping(metrics['acc'], self.model)
        return metrics

@hydra.main(version_base=None, config_path="config", config_name="ExMRD_FakeSV")
def main(cfg: DictConfig):
    # Create dataset-specific log path with include_piyao and ablation indicators
    global log_path
    dataset_name = cfg.dataset if 'dataset' in cfg else 'unknown'
    piyao_suffix = '_with_piyao' if cfg.data.get('include_piyao', False) else ''
    ablation_suffix = '_no_cot' if cfg.data.get('ablation_no_cot', False) else ''
    log_path = Path(f'log/{dataset_name}{piyao_suffix}{ablation_suffix}_{datetime.now().strftime("%m%d-%H%M%S")}')
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger.remove()
    logger.add(log_path / 'log.log', retention="10 days", level="DEBUG")
    logger.add(sys.stdout, level="INFO")
    logger.info(OmegaConf.to_yaml(cfg))
    pd.set_option('future.no_silent_downcasting', True)
    colorama.init()
    set_seed(cfg.seed)
    # mp.set_sharing_strategy('file_system')
    
    trainer = Trainer(cfg)
    trainer.run()

if  __name__ == '__main__':
    main()