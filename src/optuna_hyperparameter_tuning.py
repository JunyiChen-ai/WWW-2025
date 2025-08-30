#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for ExMRD_Retrieval on FakeTT dataset
Optimizes for Test Set Accuracy
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import Trainer
from src.utils.core_utils import (
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


class OptunaTuner:
    def __init__(self, study_name: str = "fakett_exmrd_retrieval", n_trials: int = 1000):
        self.study_name = study_name
        self.n_trials = n_trials
        self.best_params = None
        self.best_accuracy = 0.0
        
        # Create results directory
        self.results_dir = Path("optuna_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / f"{study_name}_tuning.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Fixed parameters (user specified - do not change)
        self.fixed_params = {
            'include_piyao': False,
            'description': True,
            'temp_evolution': True,
            'use_text': True,
            'use_image': True,
            'use_audio': True,
            'filter_k': None,
            'retrieval_path': "text_similarity_results/full_dataset_retrieval_LongCLIP-GmP-ViT-L-14.json"
        }
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for current trial"""
        
        # Model architecture parameters
        hid_dim = trial.suggest_categorical('hid_dim', [128, 256, 512, 768])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        num_frozen_layers = trial.suggest_int('num_frozen_layers', 2, 8)
        
        # Evidential model parameters
        evidential_hidden = trial.suggest_categorical('evidential_hidden', [128, 256, 512])
        evidential_dropout = trial.suggest_float('evidential_dropout', 0.0, 0.3, step=0.05)
        anneal_steps = trial.suggest_int('anneal_steps', 500, 2000, step=100)
        
        # Loss weights
        loss_fused = trial.suggest_float('loss_fused', 0.1, 2.0, step=0.1)
        loss_text = trial.suggest_float('loss_text', 0.1, 2.0, step=0.1)
        loss_audio = trial.suggest_float('loss_audio', 0.1, 2.0, step=0.1)
        loss_image = trial.suggest_float('loss_image', 0.1, 2.0, step=0.1)
        
        # Multi-modal transformer parameters
        transformer_layers = trial.suggest_int('transformer_layers', 1, 4)
        transformer_heads = trial.suggest_int('transformer_heads', 1, 8)
        
        # Retrieval parameters
        retrieval_alpha = trial.suggest_float('retrieval_alpha', 0.1, 0.7, step=0.05)
        
        # Optimizer parameters
        lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        
        # Training parameters
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        num_epoch = trial.suggest_int('num_epoch', 10, 30)
        
        # Gradient clipping
        gradient_clip_norm = trial.suggest_float('gradient_clip_norm', 0.0, 2.0, step=0.1)
        
        # Optional: scheduler parameters
        scheduler_name = trial.suggest_categorical('scheduler_name', ['DummyLR', 'ReduceLROnPlateau'])
        
        # Build configuration dictionary
        config = {
            'data': {
                'tokenizer_name': "zer0int/LongCLIP-GmP-ViT-L-14",
                **self.fixed_params
            },
            'para': {
                'hid_dim': hid_dim,
                'dropout': dropout,
                'text_encoder': "zer0int/LongCLIP-GmP-ViT-L-14",
                'num_frozen_layers': num_frozen_layers,
                'evidential_hidden': evidential_hidden,
                'evidential_dropout': evidential_dropout,
                'anneal_steps': anneal_steps,
                'loss_weights': {
                    'fused': loss_fused,
                    'text': loss_text,
                    'audio': loss_audio,
                    'image': loss_image
                },
                'transformer_layers': transformer_layers,
                'transformer_heads': transformer_heads,
                'retrieval_alpha': retrieval_alpha,
                'gradient_clip_norm': gradient_clip_norm
            },
            'opt': {
                'name': 'AdamW',
                'lr': lr,
                'weight_decay': weight_decay
            },
            'sche': {
                'name': scheduler_name
            },
            'num_epoch': num_epoch,
            'batch_size': batch_size,
            'text_encoder': "zer0int/LongCLIP-GmP-ViT-L-14",
            'seed': 2024,
            'model': 'ExMRD_Retrieval',
            'dataset': 'FakeTT',
            'type': 'temporal',
            'patience': num_epoch,  # Set patience = num_epoch as requested
            'eval_only': False
        }
        
        return config
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function to maximize test accuracy"""
        
        try:
            # Get hyperparameters for this trial
            config = self.suggest_hyperparameters(trial)
            
            # Convert to OmegaConf
            cfg = OmegaConf.create(config)
            
            self.logger.info(f"Trial {trial.number}: Testing hyperparameters")
            self.logger.info(f"Trial {trial.number} config: {OmegaConf.to_yaml(cfg)}")
            
            # Set seed for reproducibility
            set_seed(cfg.seed)
            
            # Create a minimal trainer for this trial
            trainer = OptunaTrial(cfg, trial, self.logger)
            test_accuracy = trainer.run()
            
            self.logger.info(f"Trial {trial.number} completed with test accuracy: {test_accuracy:.4f}")
            
            # Update best results
            if test_accuracy > self.best_accuracy:
                self.best_accuracy = test_accuracy
                self.best_params = config
                self.save_best_results()
                self.logger.info(f"New best accuracy: {test_accuracy:.4f}")
            
            return test_accuracy
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed with error: {str(e)}")
            # Return a very low score for failed trials
            return 0.0
    
    def save_best_results(self):
        """Save the best hyperparameters and performance"""
        results = {
            'best_test_accuracy': self.best_accuracy,
            'best_hyperparameters': self.best_params,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.results_dir / 'best_hyperparameters.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Best results saved: Accuracy = {self.best_accuracy:.4f}")
    
    def run_optimization(self):
        """Run the hyperparameter optimization"""
        
        self.logger.info(f"Starting hyperparameter optimization for {self.study_name}")
        self.logger.info(f"Target: Maximize Test Accuracy")
        self.logger.info(f"Number of trials: {self.n_trials}")
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',
            sampler=TPESampler(seed=2024),
            pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=5),
            storage=f'sqlite:///{self.results_dir}/{self.study_name}.db',
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Print results
        self.logger.info("Optimization completed!")
        self.logger.info(f"Best trial: {study.best_trial.number}")
        self.logger.info(f"Best test accuracy: {study.best_value:.4f}")
        self.logger.info(f"Best parameters: {study.best_params}")
        
        # Save final results
        if study.best_value > self.best_accuracy:
            self.best_accuracy = study.best_value
            # Reconstruct the full config from best params
            self.best_params = self.suggest_hyperparameters(study.best_trial)
            self.save_best_results()


class OptunaTrial:
    """Lightweight trainer for single Optuna trial"""
    
    def __init__(self, cfg: DictConfig, trial: optuna.Trial, logger):
        self.cfg = cfg
        self.trial = trial
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def run(self) -> float:
        """Run training and return test accuracy"""
        
        # Create datasets
        train_dataset = get_dataset(
            self.cfg.model, self.cfg.dataset, 
            fold='temporal', split='train', 
            **self.cfg.data
        )
        
        valid_dataset = get_dataset(
            self.cfg.model, self.cfg.dataset, 
            fold='temporal', split='valid', 
            **self.cfg.data
        )
        
        test_dataset = get_dataset(
            self.cfg.model, self.cfg.dataset, 
            fold='temporal', split='test', 
            **self.cfg.data
        )
        
        # Create data loaders
        collator = get_data_collator(self.cfg.model, self.cfg.dataset, **self.cfg.data)
        generator = torch.Generator().manual_seed(self.cfg.seed)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.cfg.batch_size,
            collate_fn=collator, 
            num_workers=min(8, self.cfg.batch_size//2),
            shuffle=True, 
            generator=generator,
            worker_init_fn=lambda worker_id: set_worker_seed(worker_id, self.cfg.seed)
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=collator,
            num_workers=min(8, self.cfg.batch_size//2),
            shuffle=False,
            generator=generator,
            worker_init_fn=lambda worker_id: set_worker_seed(worker_id, self.cfg.seed)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=collator,
            num_workers=min(8, self.cfg.batch_size//2),
            shuffle=False,
            generator=generator,
            worker_init_fn=lambda worker_id: set_worker_seed(worker_id, self.cfg.seed)
        )
        
        # Create model
        steps_per_epoch = len(train_loader)
        model = load_model(self.cfg.model, **self.cfg.para)
        model = model.to(self.device)
        
        # Create optimizer and scheduler
        optimizer = get_optimizer(model, **self.cfg.opt)
        scheduler = get_scheduler(optimizer, steps_per_epoch=steps_per_epoch, num_epoch=self.cfg.num_epoch, **self.cfg.sche)
        
        # Training loop
        best_valid_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.cfg.num_epoch):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                outputs = model(batch)
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping if specified
                if self.cfg.para.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.para.gradient_clip_norm)
                
                optimizer.step()
                
                if hasattr(scheduler, 'step') and self.cfg.sche.name != 'ReduceLROnPlateau':
                    scheduler.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            valid_correct = 0
            valid_total = 0
            
            with torch.no_grad():
                for batch in valid_loader:
                    if isinstance(batch, dict):
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    else:
                        batch = batch.to(self.device)
                    
                    outputs = model(batch)
                    predictions = outputs['predictions']
                    labels = batch['labels']
                    
                    valid_correct += (predictions == labels).sum().item()
                    valid_total += labels.size(0)
            
            valid_acc = valid_correct / valid_total
            
            # Update scheduler if ReduceLROnPlateau
            if hasattr(scheduler, 'step') and self.cfg.sche.name == 'ReduceLROnPlateau':
                scheduler.step(1 - valid_acc)  # Use validation loss proxy
            
            # Early stopping logic (patience = num_epoch means no early stopping)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Report intermediate result for pruning
            self.trial.report(valid_acc, epoch)
            
            # Pruning: stop unpromising trials early
            if self.trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping check (only if patience < num_epoch)
            if patience_counter >= self.cfg.patience and self.cfg.patience < self.cfg.num_epoch:
                break
        
        # Test phase - final evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                outputs = model(batch)
                predictions = outputs['predictions']
                labels = batch['labels']
                
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)
        
        test_accuracy = test_correct / test_total
        
        return test_accuracy


def main():
    """Main function to run hyperparameter tuning"""
    
    # Configuration
    STUDY_NAME = "fakett_exmrd_retrieval_accuracy"
    N_TRIALS = 2000  # Large number of trials for comprehensive search
    
    print(f"🚀 Starting Optuna Hyperparameter Tuning for ExMRD_Retrieval on FakeTT")
    print(f"📊 Objective: Maximize Test Set Accuracy")
    print(f"🔬 Number of trials: {N_TRIALS}")
    print(f"📁 Study name: {STUDY_NAME}")
    
    # Create and run tuner
    tuner = OptunaTuner(study_name=STUDY_NAME, n_trials=N_TRIALS)
    tuner.run_optimization()
    
    print(f"✅ Optimization completed!")
    print(f"🏆 Best test accuracy: {tuner.best_accuracy:.4f}")
    print(f"💾 Results saved to: optuna_results/best_hyperparameters.json")


if __name__ == "__main__":
    main()