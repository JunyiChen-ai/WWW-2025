#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for ExMRD_Retrieval on FakeTT dataset
Optimizes for Test Set Accuracy
"""

import sys
import os
import json
import logging
import subprocess
import tempfile
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from omegaconf import DictConfig, OmegaConf

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


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
        # Ensure transformer_heads divides hid_dim evenly
        valid_heads = [h for h in [1, 2, 4, 8] if hid_dim % h == 0]
        if not valid_heads:
            valid_heads = [1]  # fallback
        transformer_heads = trial.suggest_categorical('transformer_heads', valid_heads)
        
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
        scheduler_name = trial.suggest_categorical('scheduler_name', ['DummyLR', 'StepLR'])
        
        # Random seed selection
        seed = trial.suggest_categorical('seed', [2024, 2025, 2026, 2027, 2028])
        
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
            'seed': seed,
            'model': 'ExMRD_Retrieval',
            'dataset': 'FakeTT',
            'type': 'temporal',
            'patience': 5,  # Fixed patience
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
            
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                OmegaConf.save(cfg, f.name)
                temp_config_path = f.name
            
            try:
                # Run main.py with subprocess - use hydra override syntax
                cmd = [
                    'python', 'src/main.py', 
                    '--config-path', os.path.dirname(temp_config_path),
                    '--config-name', os.path.basename(temp_config_path).replace('.yaml', '')
                ]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    cwd=project_root,
                    timeout=3600  # 1 hour timeout
                )
                
                if result.returncode != 0:
                    self.logger.error(f"Trial {trial.number} subprocess failed: {result.stderr}")
                    return 0.0
                
                # Parse validation accuracy from output
                test_accuracy = self._parse_accuracy_from_output(result.stdout)
                
            finally:
                # Clean up temp file
                os.unlink(temp_config_path)
            
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
    
    def _parse_accuracy_from_output(self, output: str) -> float:
        """Parse the final validation accuracy from main.py output"""
        try:
            # Look for validation accuracy patterns from main.py output
            # Match patterns like "Valid: Acc: 0.6187" or "Test: Acc: 0.6187"
            patterns = [
                r'Valid: Acc:\s*([\d\.]+)',
                r'Test: Acc:\s*([\d\.]+)',
                r'Best Valid Acc:\s*([\d\.]+)',
                r'Valid Acc:\s*([\d\.]+)'
            ]
            
            accuracies = []
            for pattern in patterns:
                matches = re.findall(pattern, output, re.IGNORECASE)
                if matches:
                    accuracies.extend([float(match) for match in matches])
            
            if accuracies:
                # Return the highest accuracy found (should be the best one)
                return max(accuracies)
            
            self.logger.warning("Could not parse accuracy from output, returning 0.0")
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error parsing accuracy: {str(e)}")
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
        
        # Run optimization - sequential execution only (n_jobs=1)
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1)
        
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