#!/usr/bin/env python3
"""
Unimodal retrieval analysis for FakeSV dataset.
Analyzes performance of different text combinations and visual/audio aggregation methods.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import sys
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnimodalAnalysis:
    def __init__(self, 
                 model_name: str = "OFA-Sys/chinese-clip-vit-large-patch14",
                 data_dir: str = "data/FakeSV", 
                 output_dir: str = "analysis/FakeSV",
                 batch_size: int = 32):
        """
        Initialize unimodal analysis system
        
        Args:
            model_name: Hugging Face model name for text embedding
            data_dir: Directory containing FakeSV data
            output_dir: Directory to save analysis results
            batch_size: Batch size for text encoding
        """
        self.model_name = model_name
        self.model_short_name = model_name.split("/")[-1]
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        
        # Data paths
        self.entity_dir = self.data_dir / "entity_claims"
        self.feature_dir = self.data_dir / "fea"
        self.vid_dir = self.data_dir / "vids"
        
        # Text combination modes
        self.text_modes = {
            'title_only': self._create_title_text,
            'title_keywords': self._create_title_keywords_text,
            'full_text': self._create_full_text
        }
        
        # Visual/audio aggregation modes  
        self.visual_audio_modes = ['frame_max', 'pooled']
        
        # Results storage
        self.sample_results = []
        self.memory_bank_data = {'true': [], 'fake': []}
        self.test_data = []
        
        logger.info(f"Initialized UnimodalAnalysis with model: {model_name}")
        logger.info(f"Output directory: {self.output_dir}")
        
    def _create_title_text(self, item: Dict) -> str:
        """Create text from title only"""
        return item.get('title', '')
        
    def _create_title_keywords_text(self, item: Dict) -> str:
        """Create text from title + keywords"""
        parts = []
        if 'title' in item and item['title']:
            parts.append(item['title'])
        if 'keywords' in item and item['keywords']:
            parts.append(item['keywords'])
        return ' '.join(parts)
        
    def _create_full_text(self, item: Dict) -> str:
        """Create text from all available text fields"""
        parts = []
        for field in ['title', 'keywords', 'description', 'temporal_evolution']:
            if field in item and item[field]:
                parts.append(str(item[field]))
        
        # Add entity claims if available
        if 'entity_claims' in item and item['entity_claims']:
            claims_text = []
            for entity, claims in item['entity_claims'].items():
                for claim in claims:
                    claims_text.append(f"{entity}: {claim}")
            if claims_text:
                parts.append(" ".join(claims_text))
        
        return " ".join(parts)
        
    def load_model(self):
        """Load the text embedding model"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
        
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the loaded model"""
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding texts"):
                batch_texts = texts[i:i+self.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                if 'chinese-clip' in self.model_name.lower():
                    outputs = self.model.text_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                else:
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
        
    def load_data(self):
        """Load entity claims data and splits"""
        logger.info("Loading data...")
        
        # Load true video extractions
        true_file = self.entity_dir / "video_extractions.jsonl"
        true_data = []
        with open(true_file, 'r', encoding='utf-8') as f:
            for line in f:
                true_data.append(json.loads(line))
        
        # Load fake video descriptions
        fake_file = self.entity_dir / "fake_video_descriptions.jsonl"
        fake_data = []
        with open(fake_file, 'r', encoding='utf-8') as f:
            for line in f:
                fake_data.append(json.loads(line))
        
        # Create lookup
        all_entity_data = {}
        for item in true_data + fake_data:
            all_entity_data[item['video_id']] = item
            
        # Load splits
        splits = {}
        for split in ['train', 'valid', 'test']:
            split_file = self.vid_dir / f"vid_time3_{split}.txt"
            with open(split_file, 'r') as f:
                splits[split] = set(line.strip() for line in f)
        
        # Prepare memory bank (train + valid)
        memory_bank_ids = splits['train'] | splits['valid']
        for video_id in memory_bank_ids:
            if video_id in all_entity_data:
                item = all_entity_data[video_id]
                if item.get('annotation') == '真':
                    self.memory_bank_data['true'].append(item)
                elif item.get('annotation') == '假':
                    self.memory_bank_data['fake'].append(item)
        
        # Prepare test data  
        for video_id in splits['test']:
            if video_id in all_entity_data:
                self.test_data.append(all_entity_data[video_id])
        
        logger.info(f"Memory bank - True: {len(self.memory_bank_data['true'])}, Fake: {len(self.memory_bank_data['fake'])}")
        logger.info(f"Test samples: {len(self.test_data)}")
        
    def load_multimodal_features(self):
        """Load visual and audio features"""
        logger.info("Loading multimodal features...")
        
        # Load visual features
        vit_file = self.feature_dir / "vit_tensor.pt"
        if vit_file.exists():
            self.visual_features = torch.load(vit_file, map_location='cpu')
            # Convert to numpy
            for video_id in self.visual_features:
                self.visual_features[video_id] = self.visual_features[video_id].numpy().astype(np.float32)
        else:
            self.visual_features = {}
            logger.warning(f"Visual features file not found: {vit_file}")
        
        # Load audio features
        audio_file = self.feature_dir / "audio_features_frames.pt"
        if audio_file.exists():
            self.audio_features = torch.load(audio_file, map_location='cpu')
            # Convert to numpy
            for video_id in self.audio_features:
                self.audio_features[video_id] = self.audio_features[video_id].numpy().astype(np.float32)
        else:
            self.audio_features = {}
            logger.warning(f"Audio features file not found: {audio_file}")
            
        logger.info(f"Loaded visual features for {len(self.visual_features)} videos")
        logger.info(f"Loaded audio features for {len(self.audio_features)} videos")
        
    def analyze_text_modality(self, text_mode: str = 'full_text') -> Dict[str, Dict]:
        """
        Analyze text modality with specified text combination mode
        
        Args:
            text_mode: One of 'title_only', 'title_keywords', 'full_text'
            
        Returns:
            Dictionary with predictions and similarities for each test sample
        """
        logger.info(f"Analyzing text modality with mode: {text_mode}")
        
        if text_mode not in self.text_modes:
            raise ValueError(f"Invalid text_mode: {text_mode}")
            
        text_creator = self.text_modes[text_mode]
        
        # Create texts for memory bank
        true_texts = [text_creator(item) for item in self.memory_bank_data['true']]
        fake_texts = [text_creator(item) for item in self.memory_bank_data['fake']]
        
        # Create texts for test samples
        test_texts = [text_creator(item) for item in self.test_data]
        
        # Encode all texts
        logger.info("Encoding memory bank texts...")
        true_embeddings = self.encode_texts(true_texts) if true_texts else np.array([])
        fake_embeddings = self.encode_texts(fake_texts) if fake_texts else np.array([])
        
        logger.info("Encoding test texts...")  
        test_embeddings = self.encode_texts(test_texts)
        
        # Compute predictions
        results = {}
        for i, test_item in enumerate(tqdm(self.test_data, desc=f"Computing {text_mode} predictions")):
            video_id = test_item['video_id']
            test_emb = test_embeddings[i:i+1]  # [1, D]
            
            # Find best true match
            best_true_sim = -1
            best_true_idx = -1
            if len(true_embeddings) > 0:
                true_sims = cosine_similarity(test_emb, true_embeddings)[0]
                best_true_idx = np.argmax(true_sims)
                best_true_sim = true_sims[best_true_idx]
            
            # Find best fake match  
            best_fake_sim = -1
            best_fake_idx = -1
            if len(fake_embeddings) > 0:
                fake_sims = cosine_similarity(test_emb, fake_embeddings)[0]
                best_fake_idx = np.argmax(fake_sims)
                best_fake_sim = fake_sims[best_fake_idx]
            
            # Make prediction
            prediction = '真' if best_true_sim > best_fake_sim else '假'
            
            results[video_id] = {
                'prediction': prediction,
                'similarity_true': float(best_true_sim),
                'similarity_fake': float(best_fake_sim),
                'best_true_id': self.memory_bank_data['true'][best_true_idx]['video_id'] if best_true_idx >= 0 else None,
                'best_fake_id': self.memory_bank_data['fake'][best_fake_idx]['video_id'] if best_fake_idx >= 0 else None,
                'best_true_info': self._extract_sample_info(self.memory_bank_data['true'][best_true_idx]) if best_true_idx >= 0 else None,
                'best_fake_info': self._extract_sample_info(self.memory_bank_data['fake'][best_fake_idx]) if best_fake_idx >= 0 else None
            }
        
        return results
        
    def _extract_sample_info(self, item: Dict) -> Dict:
        """Extract key information from a sample"""
        return {
            'video_id': item['video_id'],
            'annotation': item['annotation'],
            'title': item.get('title', ''),
            'keywords': item.get('keywords', ''),
            'description': item.get('description', '')
        }
        
    def _l2_normalize(self, x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
        """L2 normalization"""
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + eps)
        
    def _compute_frame_similarity_max(self, query_frames: np.ndarray, candidate_frames_list: List[np.ndarray]) -> np.ndarray:
        """
        Compute frame-level similarity using max strategy
        
        Args:
            query_frames: [T_q, D] frames for query video
            candidate_frames_list: List of [T_c, D] frames for candidate videos
            
        Returns:
            Array of similarity scores for each candidate
        """
        query_norm = self._l2_normalize(query_frames)
        similarities = []
        
        for candidate_frames in candidate_frames_list:
            if candidate_frames.shape[0] == 0:
                similarities.append(0.0)
                continue
                
            candidate_norm = self._l2_normalize(candidate_frames)
            
            # Compute similarity matrix [T_q, T_c]
            sim_matrix = np.dot(query_norm, candidate_norm.T)
            
            # For each query frame, find max similarity across candidate frames
            max_sims = np.max(sim_matrix, axis=1)
            
            # Average across query frames
            similarities.append(float(np.mean(max_sims)))
            
        return np.array(similarities)
        
    def _compute_pooled_similarity(self, query_frames: np.ndarray, candidate_frames_list: List[np.ndarray]) -> np.ndarray:
        """
        Compute pooled similarity using average pooling
        
        Args:
            query_frames: [T_q, D] frames for query video
            candidate_frames_list: List of [T_c, D] frames for candidate videos
            
        Returns:
            Array of similarity scores for each candidate
        """
        # Pool query frames
        query_pooled = np.mean(query_frames, axis=0)
        query_pooled = self._l2_normalize(query_pooled)
        
        similarities = []
        for candidate_frames in candidate_frames_list:
            if candidate_frames.shape[0] == 0:
                similarities.append(0.0)
                continue
                
            # Pool candidate frames
            candidate_pooled = np.mean(candidate_frames, axis=0)
            candidate_pooled = self._l2_normalize(candidate_pooled)
            
            # Compute cosine similarity
            similarity = float(np.dot(query_pooled, candidate_pooled))
            similarities.append(similarity)
            
        return np.array(similarities)
        
    def analyze_visual_audio_modality(self, modality: str = 'visual', aggregate_mode: str = 'frame_max') -> Dict[str, Dict]:
        """
        Analyze visual or audio modality with specified aggregation mode
        
        Args:
            modality: 'visual' or 'audio'
            aggregate_mode: 'frame_max' or 'pooled'
            
        Returns:
            Dictionary with predictions and similarities for each test sample
        """
        logger.info(f"Analyzing {modality} modality with mode: {aggregate_mode}")
        
        if modality not in ['visual', 'audio']:
            raise ValueError(f"Invalid modality: {modality}")
        if aggregate_mode not in self.visual_audio_modes:
            raise ValueError(f"Invalid aggregate_mode: {aggregate_mode}")
            
        # Get feature dictionary
        features_dict = self.visual_features if modality == 'visual' else self.audio_features
        
        # Collect memory bank features
        true_features = []
        true_video_ids = []
        for item in self.memory_bank_data['true']:
            video_id = item['video_id']
            if video_id in features_dict:
                true_features.append(features_dict[video_id])
                true_video_ids.append(video_id)
        
        fake_features = []
        fake_video_ids = []
        for item in self.memory_bank_data['fake']:
            video_id = item['video_id']
            if video_id in features_dict:
                fake_features.append(features_dict[video_id])
                fake_video_ids.append(video_id)
        
        # Compute predictions
        results = {}
        for test_item in tqdm(self.test_data, desc=f"Computing {modality}_{aggregate_mode} predictions"):
            video_id = test_item['video_id']
            
            if video_id not in features_dict:
                logger.warning(f"Missing {modality} features for {video_id}, skipping")
                continue
                
            test_features = features_dict[video_id]
            
            # Compute similarities
            if aggregate_mode == 'frame_max':
                true_sims = self._compute_frame_similarity_max(test_features, true_features) if true_features else np.array([])
                fake_sims = self._compute_frame_similarity_max(test_features, fake_features) if fake_features else np.array([])
            else:  # pooled
                true_sims = self._compute_pooled_similarity(test_features, true_features) if true_features else np.array([])
                fake_sims = self._compute_pooled_similarity(test_features, fake_features) if fake_features else np.array([])
            
            # Find best matches
            best_true_sim = -1
            best_true_idx = -1
            if len(true_sims) > 0:
                best_true_idx = np.argmax(true_sims)
                best_true_sim = true_sims[best_true_idx]
            
            best_fake_sim = -1
            best_fake_idx = -1
            if len(fake_sims) > 0:
                best_fake_idx = np.argmax(fake_sims)
                best_fake_sim = fake_sims[best_fake_idx]
            
            # Make prediction
            prediction = '真' if best_true_sim > best_fake_sim else '假'
            
            results[video_id] = {
                'prediction': prediction,
                'similarity_true': float(best_true_sim),
                'similarity_fake': float(best_fake_sim),
                'best_true_id': true_video_ids[best_true_idx] if best_true_idx >= 0 else None,
                'best_fake_id': fake_video_ids[best_fake_idx] if best_fake_idx >= 0 else None,
                'best_true_info': self._get_memory_bank_item_info('true', true_video_ids[best_true_idx]) if best_true_idx >= 0 else None,
                'best_fake_info': self._get_memory_bank_item_info('fake', fake_video_ids[best_fake_idx]) if best_fake_idx >= 0 else None
            }
        
        return results
        
    def _get_memory_bank_item_info(self, label: str, video_id: str) -> Dict:
        """Get memory bank item info by video_id"""
        for item in self.memory_bank_data[label]:
            if item['video_id'] == video_id:
                return self._extract_sample_info(item)
        return None
        
    def run_full_analysis(self, 
                         text_modes: List[str] = None,
                         visual_modes: List[str] = None, 
                         audio_modes: List[str] = None) -> None:
        """
        Run complete unimodal analysis
        
        Args:
            text_modes: List of text modes to analyze
            visual_modes: List of visual aggregation modes to analyze  
            audio_modes: List of audio aggregation modes to analyze
        """
        start_time = time.time()
        logger.info("Starting full unimodal analysis...")
        
        # Set defaults
        if text_modes is None:
            text_modes = ['title_only', 'title_keywords', 'full_text']
        if visual_modes is None:
            visual_modes = ['frame_max', 'pooled']
        if audio_modes is None:
            audio_modes = ['frame_max', 'pooled']
        
        # Load data and model
        self.load_data()
        self.load_multimodal_features()
        self.load_model()
        
        # Store all modality results
        all_results = {}
        
        # Analyze text modalities
        for text_mode in text_modes:
            try:
                results = self.analyze_text_modality(text_mode)
                all_results[f'text_{text_mode}'] = results
                logger.info(f"Completed text_{text_mode} analysis")
            except Exception as e:
                logger.error(f"Error in text_{text_mode} analysis: {e}")
        
        # Analyze visual modalities
        for visual_mode in visual_modes:
            try:
                results = self.analyze_visual_audio_modality('visual', visual_mode)
                all_results[f'visual_{visual_mode}'] = results
                logger.info(f"Completed visual_{visual_mode} analysis")
            except Exception as e:
                logger.error(f"Error in visual_{visual_mode} analysis: {e}")
        
        # Analyze audio modalities
        for audio_mode in audio_modes:
            try:
                results = self.analyze_visual_audio_modality('audio', audio_mode)
                all_results[f'audio_{audio_mode}'] = results
                logger.info(f"Completed audio_{audio_mode} analysis")
            except Exception as e:
                logger.error(f"Error in audio_{audio_mode} analysis: {e}")
        
        # Analyze multimodal combinations
        logger.info("Starting multimodal combination analysis...")
        combination_results = self.analyze_multimodal_combinations(all_results, text_modes, visual_modes, audio_modes)
        
        # Process and save results
        self._process_and_save_results(all_results, combination_results)
        
        total_time = time.time() - start_time
        logger.info(f"Full analysis completed in {total_time:.2f} seconds")
        
    def analyze_multimodal_combinations(self, all_results: Dict[str, Dict], text_modes: List[str], 
                                       visual_modes: List[str], audio_modes: List[str]) -> Dict[str, Dict]:
        """Analyze all possible multimodal combinations"""
        logger.info("Analyzing multimodal combinations...")
        
        combination_results = {}
        
        # Generate all combinations
        for text_mode in text_modes:
            for visual_mode in visual_modes:
                for audio_mode in audio_modes:
                    combination_name = f"{text_mode}+visual_{visual_mode}+audio_{audio_mode}"
                    logger.info(f"Analyzing combination: {combination_name}")
                    
                    # Get individual modality results
                    text_key = f"text_{text_mode}"
                    visual_key = f"visual_{visual_mode}"
                    audio_key = f"audio_{audio_mode}"
                    
                    if all(key in all_results for key in [text_key, visual_key, audio_key]):
                        combination_result = self._compute_combination_prediction(
                            all_results[text_key], 
                            all_results[visual_key], 
                            all_results[audio_key],
                            combination_name
                        )
                        combination_results[combination_name] = combination_result
                    else:
                        logger.warning(f"Missing modality data for combination: {combination_name}")
        
        logger.info(f"Completed analysis of {len(combination_results)} combinations")
        return combination_results
    
    def _compute_combination_prediction(self, text_results: Dict, visual_results: Dict, 
                                       audio_results: Dict, combination_name: str) -> Dict:
        """Compute predictions for a specific multimodal combination using multiple fusion strategies"""
        results = {}
        
        # Get all video IDs that have predictions in all three modalities
        common_video_ids = set(text_results.keys()) & set(visual_results.keys()) & set(audio_results.keys())
        
        for video_id in common_video_ids:
            text_pred = text_results[video_id]
            visual_pred = visual_results[video_id]  
            audio_pred = audio_results[video_id]
            
            # Extract predictions and similarities
            text_label = text_pred.get('prediction', text_pred) if isinstance(text_pred, dict) else text_pred
            visual_label = visual_pred.get('prediction', visual_pred) if isinstance(visual_pred, dict) else visual_pred
            audio_label = audio_pred.get('prediction', audio_pred) if isinstance(audio_pred, dict) else audio_pred
            
            # Get similarity scores for confidence-based fusion
            text_sim = text_pred.get('max_similarity', 0.5) if isinstance(text_pred, dict) else 0.5
            visual_sim = visual_pred.get('max_similarity', 0.5) if isinstance(visual_pred, dict) else 0.5
            audio_sim = audio_pred.get('max_similarity', 0.5) if isinstance(audio_pred, dict) else 0.5
            
            # Multiple fusion strategies
            fusion_results = self._apply_fusion_strategies(
                [(text_label, text_sim), (visual_label, visual_sim), (audio_label, audio_sim)]
            )
            
            # Store comprehensive information
            results[video_id] = {
                'prediction': fusion_results['majority_vote'],
                'fusion_strategies': fusion_results,
                'component_predictions': {
                    'text': text_label,
                    'visual': visual_label,
                    'audio': audio_label
                },
                'component_similarities': {
                    'text': text_sim,
                    'visual': visual_sim,
                    'audio': audio_sim
                },
                'consensus': fusion_results['consensus'],
                'confidence': fusion_results['confidence']
            }
        
        logger.info(f"Combination {combination_name}: {len(results)} predictions computed")
        return results
    
    def _apply_fusion_strategies(self, predictions_sims: List[tuple]) -> Dict:
        """Apply multiple fusion strategies to combine predictions"""
        predictions = [pred for pred, sim in predictions_sims]
        similarities = [sim for pred, sim in predictions_sims]
        
        fake_count = predictions.count('假')
        true_count = predictions.count('真')
        
        # 1. Majority voting
        majority_vote = '假' if fake_count > true_count else '真'
        
        # 2. Confidence-based (highest similarity wins)
        max_sim_idx = np.argmax(similarities)
        confidence_based = predictions[max_sim_idx]
        
        # 3. Weighted voting (weight by similarity)
        fake_weight = sum(sim for pred, sim in predictions_sims if pred == '假')
        true_weight = sum(sim for pred, sim in predictions_sims if pred == '真')
        weighted_vote = '假' if fake_weight > true_weight else '真'
        
        # 4. Conservative (require at least 2/3 agreement, otherwise default to '真')
        conservative = majority_vote if max(fake_count, true_count) >= 2 else '真'
        
        return {
            'majority_vote': majority_vote,
            'confidence_based': confidence_based,
            'weighted_vote': weighted_vote, 
            'conservative': conservative,
            'consensus': fake_count == 3 or true_count == 3,
            'confidence': max(fake_count, true_count) / 3.0,
            'vote_distribution': {'fake': fake_count, 'true': true_count},
            'similarity_stats': {
                'mean': np.mean(similarities),
                'max': np.max(similarities),
                'min': np.min(similarities),
                'std': np.std(similarities)
            }
        }
    
    def _analyze_combination_patterns(self, combination_sample_results: List[Dict]) -> Dict:
        """Analyze patterns and agreements between multimodal combinations"""
        logger.info("Analyzing combination patterns and agreements...")
        
        if not combination_sample_results:
            return {}
        
        # Get all combination names
        sample_with_combos = next((s for s in combination_sample_results if s['combination_predictions']), None)
        if not sample_with_combos:
            return {}
        
        combo_names = list(sample_with_combos['combination_predictions'].keys())
        n_combos = len(combo_names)
        
        # Initialize agreement matrix
        agreement_matrix = np.zeros((n_combos, n_combos))
        
        # Calculate pairwise agreements
        for i, combo1 in enumerate(combo_names):
            for j, combo2 in enumerate(combo_names):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    agreements = 0
                    total_comparisons = 0
                    
                    for sample in combination_sample_results:
                        pred1_info = sample['combination_predictions'].get(combo1)
                        pred2_info = sample['combination_predictions'].get(combo2)
                        
                        if pred1_info and pred2_info:
                            pred1 = pred1_info['prediction']
                            pred2 = pred2_info['prediction']
                            if pred1 == pred2:
                                agreements += 1
                            total_comparisons += 1
                    
                    agreement_matrix[i, j] = agreements / total_comparisons if total_comparisons > 0 else 0
        
        # Find best and worst performing combinations
        combo_accuracies = {}
        for combo_name in combo_names:
            correct = 0
            total = 0
            for sample in combination_sample_results:
                if combo_name in sample['combination_predictions']:
                    pred_info = sample['combination_predictions'][combo_name]
                    if pred_info['prediction'] == sample['ground_truth']:
                        correct += 1
                    total += 1
            
            combo_accuracies[combo_name] = correct / total if total > 0 else 0
        
        # Analyze fusion strategy performance
        fusion_strategy_performance = self._analyze_fusion_strategies(combination_sample_results)
        
        # Analyze consensus patterns
        consensus_analysis = self._analyze_consensus_patterns(combination_sample_results)
        
        # Complementarity analysis
        complementarity_analysis = self._analyze_complementarity(combination_sample_results, combo_names)
        
        return {
            'agreement_matrix': {
                'data': agreement_matrix.tolist(),
                'combination_names': combo_names,
                'summary_stats': {
                    'mean_agreement': float(np.mean(agreement_matrix[np.triu_indices(n_combos, k=1)])),
                    'max_agreement': float(np.max(agreement_matrix[np.triu_indices(n_combos, k=1)])),
                    'min_agreement': float(np.min(agreement_matrix[np.triu_indices(n_combos, k=1)])),
                    'std_agreement': float(np.std(agreement_matrix[np.triu_indices(n_combos, k=1)]))
                }
            },
            'combination_performance': {
                'accuracies': combo_accuracies,
                'best_combination': max(combo_accuracies.items(), key=lambda x: x[1]),
                'worst_combination': min(combo_accuracies.items(), key=lambda x: x[1]),
                'performance_ranking': sorted(combo_accuracies.items(), key=lambda x: x[1], reverse=True)
            },
            'fusion_strategy_analysis': fusion_strategy_performance,
            'consensus_patterns': consensus_analysis,
            'complementarity_analysis': complementarity_analysis
        }
    
    def _analyze_fusion_strategies(self, combination_sample_results: List[Dict]) -> Dict:
        """Analyze performance of different fusion strategies"""
        strategies = ['majority_vote', 'confidence_based', 'weighted_vote', 'conservative']
        strategy_performance = {strategy: {'correct': 0, 'total': 0} for strategy in strategies}
        
        for sample in combination_sample_results:
            ground_truth = sample['ground_truth']
            
            for combo_name, pred_info in sample['combination_predictions'].items():
                fusion_results = pred_info.get('fusion_strategies', {})
                
                for strategy in strategies:
                    if strategy in fusion_results:
                        strategy_pred = fusion_results[strategy]
                        if strategy_pred == ground_truth:
                            strategy_performance[strategy]['correct'] += 1
                        strategy_performance[strategy]['total'] += 1
        
        # Calculate accuracies
        for strategy in strategies:
            total = strategy_performance[strategy]['total']
            if total > 0:
                strategy_performance[strategy]['accuracy'] = strategy_performance[strategy]['correct'] / total
            else:
                strategy_performance[strategy]['accuracy'] = 0
        
        return strategy_performance
    
    def _analyze_consensus_patterns(self, combination_sample_results: List[Dict]) -> Dict:
        """Analyze consensus patterns across combinations"""
        consensus_stats = {
            'full_consensus': 0,  # All combinations agree
            'majority_consensus': 0,  # >50% agree
            'no_consensus': 0,  # Highly split
            'total_samples': len(combination_sample_results)
        }
        
        consensus_details = []
        
        for sample in combination_sample_results:
            predictions = []
            for combo_name, pred_info in sample['combination_predictions'].items():
                predictions.append(pred_info['prediction'])
            
            if predictions:
                fake_count = predictions.count('假')
                true_count = predictions.count('真')
                total_preds = len(predictions)
                
                # Classify consensus level
                if fake_count == total_preds or true_count == total_preds:
                    consensus_level = 'full_consensus'
                    consensus_stats['full_consensus'] += 1
                elif max(fake_count, true_count) > total_preds / 2:
                    consensus_level = 'majority_consensus' 
                    consensus_stats['majority_consensus'] += 1
                else:
                    consensus_level = 'no_consensus'
                    consensus_stats['no_consensus'] += 1
                
                consensus_details.append({
                    'video_id': sample['video_id'],
                    'ground_truth': sample['ground_truth'],
                    'consensus_level': consensus_level,
                    'vote_distribution': {'fake': fake_count, 'true': true_count},
                    'agreement_ratio': max(fake_count, true_count) / total_preds
                })
        
        return {
            'summary': consensus_stats,
            'details': consensus_details[:50]  # Limit for file size
        }
    
    def _analyze_complementarity(self, combination_sample_results: List[Dict], combo_names: List[str]) -> Dict:
        """Analyze which combinations are complementary (succeed where others fail)"""
        complementarity_matrix = {}
        
        for combo1 in combo_names:
            complementarity_matrix[combo1] = {}
            
            for combo2 in combo_names:
                if combo1 != combo2:
                    # Count cases where combo1 is correct and combo2 is wrong
                    combo1_correct_combo2_wrong = 0
                    both_evaluated = 0
                    
                    for sample in combination_sample_results:
                        pred1_info = sample['combination_predictions'].get(combo1)
                        pred2_info = sample['combination_predictions'].get(combo2)
                        
                        if pred1_info and pred2_info:
                            pred1_correct = pred1_info['prediction'] == sample['ground_truth']
                            pred2_correct = pred2_info['prediction'] == sample['ground_truth']
                            
                            if pred1_correct and not pred2_correct:
                                combo1_correct_combo2_wrong += 1
                            both_evaluated += 1
                    
                    complementarity_score = combo1_correct_combo2_wrong / both_evaluated if both_evaluated > 0 else 0
                    complementarity_matrix[combo1][combo2] = complementarity_score
        
        # Find most complementary pairs
        complementary_pairs = []
        for combo1 in combo_names:
            for combo2 in combo_names:
                if combo1 != combo2 and combo1 < combo2:  # Avoid duplicates
                    score1 = complementarity_matrix[combo1][combo2]
                    score2 = complementarity_matrix[combo2][combo1]
                    avg_complementarity = (score1 + score2) / 2
                    
                    complementary_pairs.append({
                        'combination1': combo1,
                        'combination2': combo2,
                        'complementarity_score': avg_complementarity,
                        'combo1_helps_combo2': score1,
                        'combo2_helps_combo1': score2
                    })
        
        complementary_pairs.sort(key=lambda x: x['complementarity_score'], reverse=True)
        
        return {
            'matrix': complementarity_matrix,
            'top_complementary_pairs': complementary_pairs[:10],
            'summary': {
                'most_complementary': complementary_pairs[0] if complementary_pairs else None,
                'least_complementary': complementary_pairs[-1] if complementary_pairs else None
            }
        }
    
    def _process_and_save_results(self, all_results: Dict[str, Dict], combination_results: Dict[str, Dict] = None) -> None:
        """Process all results and save comprehensive analysis"""
        logger.info("Processing and saving results...")
        
        # Create sample-level results for individual modalities
        individual_sample_results = []
        for test_item in self.test_data:
            video_id = test_item['video_id']
            ground_truth = test_item['annotation']
            
            sample_result = {
                'video_id': video_id,
                'ground_truth': ground_truth,
                'predictions': {},
                'correct_modalities': [],
                'num_correct': 0
            }
            
            # Collect predictions from all individual modalities
            for modality_name, modality_results in all_results.items():
                if video_id in modality_results:
                    pred_info = modality_results[video_id]
                    sample_result['predictions'][modality_name] = pred_info
                    
                    # Check if prediction is correct
                    pred_label = pred_info.get('prediction', pred_info) if isinstance(pred_info, dict) else pred_info
                    if pred_label == ground_truth:
                        sample_result['correct_modalities'].append(modality_name)
            
            sample_result['num_correct'] = len(sample_result['correct_modalities'])
            individual_sample_results.append(sample_result)
        
        # Create sample-level results for combinations
        combination_sample_results = []
        if combination_results:
            for test_item in self.test_data:
                video_id = test_item['video_id']
                ground_truth = test_item['annotation']
                
                combination_sample = {
                    'video_id': video_id,
                    'ground_truth': ground_truth,
                    'combination_predictions': {},
                    'correct_combinations': [],
                    'num_correct_combinations': 0
                }
                
                # Collect predictions from all combinations
                for combo_name, combo_results in combination_results.items():
                    if video_id in combo_results:
                        pred_info = combo_results[video_id]
                        combination_sample['combination_predictions'][combo_name] = pred_info
                        
                        # Check if combination prediction is correct
                        if pred_info['prediction'] == ground_truth:
                            combination_sample['correct_combinations'].append(combo_name)
                
                combination_sample['num_correct_combinations'] = len(combination_sample['correct_combinations'])
                combination_sample_results.append(combination_sample)
        
        # Analyze individual modality combinations
        individual_modality_analysis = self._analyze_modality_combinations(individual_sample_results)
        
        # Analyze multimodal combinations
        combination_analysis = self._analyze_combination_patterns(combination_sample_results) if combination_results else {}
        
        # Calculate metrics for individual modalities
        individual_metrics = self._calculate_metrics(all_results)
        
        # Calculate metrics for combinations  
        combination_metrics = self._calculate_metrics(combination_results) if combination_results else {}
        
        # Remove old final_results creation - now handled in comprehensive analysis
        
        # Save comprehensive analysis files
        self._save_comprehensive_analysis_files(
            individual_sample_results, combination_sample_results,
            individual_modality_analysis, combination_analysis, 
            individual_metrics, combination_metrics,
            all_results, combination_results
        )
    
    def _save_comprehensive_analysis_files(self, individual_sample_results: List[Dict], 
                                          combination_sample_results: List[Dict],
                                          individual_modality_analysis: Dict, combination_analysis: Dict,
                                          individual_metrics: Dict, combination_metrics: Dict,
                                          all_results: Dict, combination_results: Dict) -> None:
        """Save comprehensive analysis results to multiple organized files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"multimodal_analysis_{self.model_short_name}_{timestamp}"
        
        # 1. Executive summary with key insights
        summary_file = self.output_dir / f"{base_name}_executive_summary.json"
        summary_data = self._create_executive_summary(
            individual_metrics, combination_metrics, 
            individual_modality_analysis, combination_analysis,
            individual_sample_results, combination_sample_results
        )
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        # 2. Individual modality analysis
        individual_file = self.output_dir / f"{base_name}_individual_modalities.json"
        individual_data = {
            'metrics': individual_metrics,
            'modality_analysis': individual_modality_analysis,
            'sample_results': individual_sample_results[:100]  # Limit for file size
        }
        with open(individual_file, 'w', encoding='utf-8') as f:
            json.dump(individual_data, f, ensure_ascii=False, indent=2)
        
        # 3. Multimodal combination analysis
        combinations_file = self.output_dir / f"{base_name}_multimodal_combinations.json"
        combination_data = {
            'metrics': combination_metrics,
            'combination_analysis': combination_analysis,
            'sample_results': combination_sample_results[:100]  # Limit for file size
        }
        with open(combinations_file, 'w', encoding='utf-8') as f:
            json.dump(combination_data, f, ensure_ascii=False, indent=2)
        
        # 4. Agreement matrix and consensus analysis
        agreement_file = self.output_dir / f"{base_name}_agreement_analysis.json"
        agreement_data = {
            'pairwise_agreements': combination_analysis.get('agreement_matrix', {}),
            'consensus_patterns': combination_analysis.get('consensus_patterns', {}),
            'complementarity_analysis': combination_analysis.get('complementarity_analysis', {})
        }
        with open(agreement_file, 'w', encoding='utf-8') as f:
            json.dump(agreement_data, f, ensure_ascii=False, indent=2)
        
        # 5. Error analysis and failure patterns
        error_file = self.output_dir / f"{base_name}_error_analysis.json"
        error_data = self._create_error_analysis(
            individual_sample_results, combination_sample_results,
            individual_metrics, combination_metrics
        )
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        
        # 6. Complete sample predictions (all modalities and combinations)
        complete_predictions_file = self.output_dir / f"{base_name}_complete_predictions.json"
        complete_data = {
            'individual_predictions': all_results,
            'combination_predictions': combination_results or {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(individual_sample_results),
                'individual_modalities': len(all_results),
                'combinations': len(combination_results) if combination_results else 0
            }
        }
        with open(complete_predictions_file, 'w', encoding='utf-8') as f:
            json.dump(complete_data, f, ensure_ascii=False, indent=2)
        
        # 7. Human-readable comprehensive report
        text_report_file = self.output_dir / f"{base_name}_comprehensive_report.txt"
        self._save_comprehensive_text_report(
            text_report_file, summary_data, individual_metrics, 
            combination_metrics, combination_analysis
        )
        
        # 8. Interactive HTML report
        html_report_file = self.output_dir / f"{base_name}_interactive_report.html"
        self._generate_html_report(
            html_report_file, summary_data, individual_metrics,
            combination_metrics, combination_analysis, agreement_data
        )
        
        all_files = [summary_file, individual_file, combinations_file, agreement_file, 
                    error_file, complete_predictions_file, text_report_file, html_report_file]
        
        logger.info(f"Comprehensive analysis saved to {len(all_files)} files:")
        logger.info(f"  - Executive Summary: {summary_file}")
        logger.info(f"  - Individual Modalities: {individual_file}")
        logger.info(f"  - Multimodal Combinations: {combinations_file}")
        logger.info(f"  - Agreement Analysis: {agreement_file}")
        logger.info(f"  - Error Analysis: {error_file}")
        logger.info(f"  - Complete Predictions: {complete_predictions_file}")
        logger.info(f"  - Comprehensive Report: {text_report_file}")
        logger.info(f"  - Interactive HTML Report: {html_report_file}")
    
    def _create_executive_summary(self, individual_metrics: Dict, combination_metrics: Dict,
                                individual_modality_analysis: Dict, combination_analysis: Dict,
                                individual_sample_results: List[Dict], combination_sample_results: List[Dict]) -> Dict:
        """Create executive summary with key insights"""
        
        # Best performing individual modality
        best_individual = max(individual_metrics.items(), key=lambda x: x[1]['accuracy']) if individual_metrics else (None, {'accuracy': 0})
        
        # Best performing combination
        best_combination = max(combination_metrics.items(), key=lambda x: x[1]['accuracy']) if combination_metrics else (None, {'accuracy': 0})
        
        # Performance improvement analysis
        individual_best_acc = best_individual[1]['accuracy']
        combination_best_acc = best_combination[1]['accuracy']
        improvement = combination_best_acc - individual_best_acc
        improvement_pct = (improvement / individual_best_acc * 100) if individual_best_acc > 0 else 0
        
        # Key insights
        insights = []
        
        if combination_metrics:
            # Fusion strategy analysis
            fusion_analysis = combination_analysis.get('fusion_strategy_analysis', {})
            if fusion_analysis:
                best_strategy = max(fusion_analysis.items(), key=lambda x: x[1].get('accuracy', 0))
                insights.append(f"Best fusion strategy: {best_strategy[0]} ({best_strategy[1].get('accuracy', 0):.3f} accuracy)")
            
            # Consensus analysis
            consensus_patterns = combination_analysis.get('consensus_patterns', {})
            if consensus_patterns and 'summary' in consensus_patterns:
                summary = consensus_patterns['summary']
                total = summary.get('total_samples', 1)
                full_consensus_pct = summary.get('full_consensus', 0) / total * 100
                insights.append(f"Full consensus achieved in {full_consensus_pct:.1f}% of samples")
            
            # Complementarity insights
            complementarity = combination_analysis.get('complementarity_analysis', {})
            if 'summary' in complementarity and complementarity['summary'].get('most_complementary'):
                most_comp = complementarity['summary']['most_complementary']
                insights.append(f"Most complementary pair: {most_comp['combination1']} + {most_comp['combination2']}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'data_overview': {
                'test_samples': len(individual_sample_results),
                'memory_bank_samples': len(self.memory_bank_data['true']) + len(self.memory_bank_data['fake']),
                'individual_modalities_tested': len(individual_metrics),
                'multimodal_combinations_tested': len(combination_metrics) if combination_metrics else 0
            },
            'performance_highlights': {
                'best_individual_modality': {
                    'name': best_individual[0],
                    'accuracy': round(best_individual[1]['accuracy'], 4),
                    'f1_macro': round(best_individual[1].get('f1_macro', 0), 4)
                },
                'best_multimodal_combination': {
                    'name': best_combination[0],
                    'accuracy': round(best_combination[1]['accuracy'], 4),
                    'f1_macro': round(best_combination[1].get('f1_macro', 0), 4)
                } if combination_metrics else None,
                'multimodal_improvement': {
                    'absolute_improvement': round(improvement, 4),
                    'relative_improvement_percent': round(improvement_pct, 2)
                } if combination_metrics else None
            },
            'top_performing_combinations': sorted(
                [(k, v['accuracy']) for k, v in combination_metrics.items()],
                key=lambda x: x[1], reverse=True
            )[:5] if combination_metrics else [],
            'key_insights': insights,
            'analysis_completeness': {
                'individual_modality_analysis': bool(individual_modality_analysis),
                'combination_agreement_analysis': bool(combination_analysis.get('agreement_matrix')),
                'consensus_pattern_analysis': bool(combination_analysis.get('consensus_patterns')),
                'complementarity_analysis': bool(combination_analysis.get('complementarity_analysis')),
                'error_pattern_analysis': True
            }
        }
    
    def _create_error_analysis(self, individual_sample_results: List[Dict], combination_sample_results: List[Dict],
                              individual_metrics: Dict, combination_metrics: Dict) -> Dict:
        """Create comprehensive error analysis"""
        
        error_analysis = {
            'timestamp': datetime.now().isoformat(),
            'individual_modality_errors': {},
            'combination_errors': {},
            'systematic_failures': [],
            'improvement_opportunities': []
        }
        
        # Individual modality error patterns
        for modality, metrics in individual_metrics.items():
            error_samples = []
            for sample in individual_sample_results:
                if modality in sample['predictions']:
                    pred_info = sample['predictions'][modality]
                    pred_label = pred_info.get('prediction', pred_info) if isinstance(pred_info, dict) else pred_info
                    if pred_label != sample['ground_truth']:
                        error_samples.append({
                            'video_id': sample['video_id'],
                            'ground_truth': sample['ground_truth'],
                            'prediction': pred_label,
                            'similarity': pred_info.get('max_similarity', 0) if isinstance(pred_info, dict) else 0
                        })
            
            error_analysis['individual_modality_errors'][modality] = {
                'total_errors': len(error_samples),
                'error_rate': len(error_samples) / metrics.get('total_samples', 1),
                'error_samples': error_samples[:20]  # Limit for file size
            }
        
        # Combination error patterns
        if combination_sample_results:
            for combo_name, metrics in combination_metrics.items():
                error_samples = []
                for sample in combination_sample_results:
                    if combo_name in sample['combination_predictions']:
                        pred_info = sample['combination_predictions'][combo_name]
                        if pred_info['prediction'] != sample['ground_truth']:
                            error_samples.append({
                                'video_id': sample['video_id'],
                                'ground_truth': sample['ground_truth'],
                                'prediction': pred_info['prediction'],
                                'component_predictions': pred_info['component_predictions'],
                                'consensus': pred_info['consensus'],
                                'confidence': pred_info['confidence']
                            })
                
                error_analysis['combination_errors'][combo_name] = {
                    'total_errors': len(error_samples),
                    'error_rate': len(error_samples) / metrics.get('total_samples', 1),
                    'error_samples': error_samples[:10]  # Limit for file size
                }
        
        # Find systematic failures (samples that most modalities/combinations get wrong)
        if individual_sample_results:
            for sample in individual_sample_results:
                error_count = len(sample['predictions']) - sample['num_correct']
                if error_count >= len(sample['predictions']) * 0.7:  # 70% or more errors
                    error_analysis['systematic_failures'].append({
                        'video_id': sample['video_id'],
                        'ground_truth': sample['ground_truth'],
                        'error_modalities': len(sample['predictions']) - sample['num_correct'],
                        'total_modalities': len(sample['predictions']),
                        'error_rate': error_count / len(sample['predictions'])
                    })
        
        # Limit systematic failures for file size
        error_analysis['systematic_failures'] = error_analysis['systematic_failures'][:20]
        
        return error_analysis
    
    def _save_comprehensive_text_report(self, file_path: Path, summary_data: Dict, 
                                       individual_metrics: Dict, combination_metrics: Dict,
                                       combination_analysis: Dict) -> None:
        """Save comprehensive human-readable text report"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE MULTIMODAL ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(f"Model: {summary_data['model']}\n")
            f.write(f"Generated: {summary_data['timestamp']}\n")
            f.write(f"Test samples: {summary_data['data_overview']['test_samples']}\n")
            f.write(f"Individual modalities: {summary_data['data_overview']['individual_modalities_tested']}\n")
            f.write(f"Multimodal combinations: {summary_data['data_overview']['multimodal_combinations_tested']}\n\n")
            
            # Performance highlights
            perf = summary_data['performance_highlights']
            f.write("PERFORMANCE HIGHLIGHTS\n")
            f.write("-"*40 + "\n")
            f.write(f"Best individual modality: {perf['best_individual_modality']['name']} ")
            f.write(f"(Accuracy: {perf['best_individual_modality']['accuracy']:.4f})\n")
            
            if perf['best_multimodal_combination']:
                f.write(f"Best multimodal combination: {perf['best_multimodal_combination']['name']} ")
                f.write(f"(Accuracy: {perf['best_multimodal_combination']['accuracy']:.4f})\n")
                
                if perf['multimodal_improvement']:
                    imp = perf['multimodal_improvement']
                    f.write(f"Multimodal improvement: +{imp['absolute_improvement']:.4f} ")
                    f.write(f"({imp['relative_improvement_percent']:.2f}%)\n")
            
            # Top combinations
            if summary_data['top_performing_combinations']:
                f.write("\nTOP 5 MULTIMODAL COMBINATIONS\n")
                f.write("-"*40 + "\n")
                for i, (name, acc) in enumerate(summary_data['top_performing_combinations'], 1):
                    f.write(f"{i}. {name}: {acc:.4f}\n")
            
            # Individual modality performance
            f.write("\nINDIVIDUAL MODALITY PERFORMANCE\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Modality':<30} {'Accuracy':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}\n")
            f.write("-"*72 + "\n")
            
            sorted_individual = sorted(individual_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for modality, metrics in sorted_individual:
                f.write(f"{modality:<30} {metrics['accuracy']:<10.4f} {metrics.get('f1_macro', 0):<10.4f} ")
                f.write(f"{metrics.get('precision_macro', 0):<12.4f} {metrics.get('recall_macro', 0):<10.4f}\n")
            
            # Agreement analysis
            if 'agreement_matrix' in combination_analysis:
                agreement_stats = combination_analysis['agreement_matrix'].get('summary_stats', {})
                f.write("\nCOMBINATION AGREEMENT ANALYSIS\n")
                f.write("-"*40 + "\n")
                f.write(f"Mean agreement: {agreement_stats.get('mean_agreement', 0):.3f}\n")
                f.write(f"Max agreement: {agreement_stats.get('max_agreement', 0):.3f}\n")
                f.write(f"Min agreement: {agreement_stats.get('min_agreement', 0):.3f}\n")
            
            # Key insights
            if summary_data['key_insights']:
                f.write("\nKEY INSIGHTS\n")
                f.write("-"*40 + "\n")
                for insight in summary_data['key_insights']:
                    f.write(f"• {insight}\n")
            
            # Complementarity analysis
            if 'complementarity_analysis' in combination_analysis:
                comp_summary = combination_analysis['complementarity_analysis'].get('summary', {})
                if comp_summary.get('most_complementary'):
                    f.write("\nCOMPLEMENTARITY INSIGHTS\n")
                    f.write("-"*40 + "\n")
                    most_comp = comp_summary['most_complementary']
                    f.write(f"Most complementary pair: {most_comp['combination1']} + {most_comp['combination2']}\n")
                    f.write(f"Complementarity score: {most_comp['complementarity_score']:.3f}\n")
    
    def _generate_html_report(self, file_path: Path, summary_data: Dict, individual_metrics: Dict,
                             combination_metrics: Dict, combination_analysis: Dict, agreement_data: Dict) -> None:
        """Generate interactive HTML report with visualizations"""
        
        # Prepare data for JavaScript visualization
        individual_data = []
        for name, metrics in individual_metrics.items():
            individual_data.append({
                'name': name,
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics.get('f1_macro', 0),
                'precision_macro': metrics.get('precision_macro', 0),
                'recall_macro': metrics.get('recall_macro', 0)
            })
        
        combination_data = []
        if combination_metrics:
            for name, metrics in combination_metrics.items():
                combination_data.append({
                    'name': name,
                    'accuracy': metrics['accuracy'],
                    'f1_macro': metrics.get('f1_macro', 0),
                    'precision_macro': metrics.get('precision_macro', 0),
                    'recall_macro': metrics.get('recall_macro', 0)
                })
        
        # Agreement matrix data
        agreement_matrix_data = agreement_data.get('pairwise_agreements', {}).get('data', [])
        combination_names = agreement_data.get('pairwise_agreements', {}).get('combination_names', [])
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal Analysis Report - {summary_data['model']}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist@latest"></script>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 2px solid #333;
            padding-bottom: 20px;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 8px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2196F3;
        }}
        .chart-container {{
            width: 100%;
            height: 400px;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .highlight {{
            background-color: #e3f2fd;
        }}
        .tabs {{
            display: flex;
            margin-bottom: 20px;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            background: #e0e0e0;
            border: 1px solid #ccc;
            border-bottom: none;
        }}
        .tab.active {{
            background: white;
            border-bottom: 1px solid white;
        }}
        .tab-content {{
            display: none;
            border: 1px solid #ccc;
            padding: 20px;
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Comprehensive Multimodal Analysis Report</h1>
            <p><strong>Model:</strong> {summary_data['model']}</p>
            <p><strong>Generated:</strong> {summary_data['timestamp']}</p>
            <p><strong>Test Samples:</strong> {summary_data['data_overview']['test_samples']}</p>
        </div>

        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{summary_data['data_overview']['individual_modalities_tested']}</div>
                    <div>Individual Modalities</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary_data['data_overview']['multimodal_combinations_tested']}</div>
                    <div>Multimodal Combinations</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary_data['performance_highlights']['best_individual_modality']['accuracy']:.3f}</div>
                    <div>Best Individual Accuracy</div>
                </div>
                {'<div class="metric-card"><div class="metric-value">' + str(round(summary_data['performance_highlights']['best_multimodal_combination']['accuracy'], 3)) + '</div><div>Best Combination Accuracy</div></div>' if summary_data['performance_highlights']['best_multimodal_combination'] else ''}
            </div>
        </div>

        <div class="section">
            <div class="tabs">
                <div class="tab active" onclick="showTab('individual-tab', 'individual')">Individual Modalities</div>
                {'<div class="tab" onclick="showTab(\'combination-tab\', \'combination\')">Multimodal Combinations</div>' if combination_data else ''}
                {'<div class="tab" onclick="showTab(\'agreement-tab\', \'agreement\')">Agreement Analysis</div>' if agreement_matrix_data else ''}
            </div>

            <div id="individual" class="tab-content active">
                <h3>Individual Modality Performance</h3>
                <canvas id="individualChart" class="chart-container"></canvas>
                <table>
                    <tr>
                        <th>Modality</th>
                        <th>Accuracy</th>
                        <th>F1 Score</th>
                        <th>Precision</th>
                        <th>Recall</th>
                    </tr>
                    {''.join([f'<tr class="{"highlight" if data["name"] == summary_data["performance_highlights"]["best_individual_modality"]["name"] else ""}"><td>{data["name"]}</td><td>{data["accuracy"]:.4f}</td><td>{data["f1_macro"]:.4f}</td><td>{data["precision_macro"]:.4f}</td><td>{data["recall_macro"]:.4f}</td></tr>' for data in individual_data])}
                </table>
            </div>

            {'<div id="combination" class="tab-content"><h3>Multimodal Combination Performance</h3><canvas id="combinationChart" class="chart-container"></canvas><table><tr><th>Combination</th><th>Accuracy</th><th>F1 Score</th><th>Precision</th><th>Recall</th></tr>' + ''.join([f'<tr class="{"highlight" if combination_metrics and data["name"] == summary_data["performance_highlights"]["best_multimodal_combination"]["name"] else ""}"><td>{data["name"]}</td><td>{data["accuracy"]:.4f}</td><td>{data["f1_macro"]:.4f}</td><td>{data["precision_macro"]:.4f}</td><td>{data["recall_macro"]:.4f}</td></tr>' for data in combination_data]) + '</table></div>' if combination_data else ''}

            {'<div id="agreement" class="tab-content"><h3>Combination Agreement Matrix</h3><div id="agreementHeatmap" class="chart-container"></div></div>' if agreement_matrix_data else ''}
        </div>

        {'<div class="section"><h2>Key Insights</h2><ul>' + ''.join([f'<li>{insight}</li>' for insight in summary_data['key_insights']]) + '</ul></div>' if summary_data['key_insights'] else ''}
    </div>

    <script>
        // Individual modality chart
        const individualCtx = document.getElementById('individualChart').getContext('2d');
        const individualData = {json.dumps(individual_data)};
        
        new Chart(individualCtx, {{
            type: 'bar',
            data: {{
                labels: individualData.map(d => d.name),
                datasets: [{{
                    label: 'Accuracy',
                    data: individualData.map(d => d.accuracy),
                    backgroundColor: 'rgba(33, 150, 243, 0.8)',
                    borderColor: 'rgba(33, 150, 243, 1)',
                    borderWidth: 1
                }}, {{
                    label: 'F1 Score',
                    data: individualData.map(d => d.f1_macro),
                    backgroundColor: 'rgba(76, 175, 80, 0.8)',
                    borderColor: 'rgba(76, 175, 80, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1
                    }}
                }}
            }}
        }});

        // Combination chart (if data exists)
        {'const combinationCtx = document.getElementById("combinationChart");' if combination_data else ''}
        {'if (combinationCtx) {' if combination_data else ''}
            {'const combinationData = ' + json.dumps(combination_data) + ';' if combination_data else ''}
            {'new Chart(combinationCtx, {' if combination_data else ''}
                {'type: "bar",' if combination_data else ''}
                {'data: {' if combination_data else ''}
                    {'labels: combinationData.map(d => d.name.replace(/\\+/g, "\\n")),' if combination_data else ''}
                    {'datasets: [{' if combination_data else ''}
                        {'label: "Accuracy",' if combination_data else ''}
                        {'data: combinationData.map(d => d.accuracy),' if combination_data else ''}
                        {'backgroundColor: "rgba(255, 152, 0, 0.8)",' if combination_data else ''}
                        {'borderColor: "rgba(255, 152, 0, 1)",' if combination_data else ''}
                        {'borderWidth: 1' if combination_data else ''}
                    {'}]' if combination_data else ''}
                {'},' if combination_data else ''}
                {'options: {' if combination_data else ''}
                    {'responsive: true,' if combination_data else ''}
                    {'maintainAspectRatio: false,' if combination_data else ''}
                    {'scales: {' if combination_data else ''}
                        {'x: { ticks: { maxRotation: 45 } },' if combination_data else ''}
                        {'y: { beginAtZero: true, max: 1 }' if combination_data else ''}
                    {'}' if combination_data else ''}
                {'}' if combination_data else ''}
            {'});' if combination_data else ''}
        {'}' if combination_data else ''}

        // Agreement heatmap (if data exists)
        {'const agreementMatrix = ' + json.dumps(agreement_matrix_data) + ';' if agreement_matrix_data else ''}
        {'const combinationNames = ' + json.dumps(combination_names) + ';' if agreement_matrix_data else ''}
        {'if (agreementMatrix.length > 0) {' if agreement_matrix_data else ''}
            {'Plotly.newPlot("agreementHeatmap", [{' if agreement_matrix_data else ''}
                {'z: agreementMatrix,' if agreement_matrix_data else ''}
                {'x: combinationNames,' if agreement_matrix_data else ''}
                {'y: combinationNames,' if agreement_matrix_data else ''}
                {'type: "heatmap",' if agreement_matrix_data else ''}
                {'colorscale: "Viridis"' if agreement_matrix_data else ''}
            {'}], {' if agreement_matrix_data else ''}
                {'title: "Combination Agreement Matrix",' if agreement_matrix_data else ''}
                {'xaxis: { title: "Combinations" },' if agreement_matrix_data else ''}
                {'yaxis: { title: "Combinations" }' if agreement_matrix_data else ''}
            {'});' if agreement_matrix_data else ''}
        {'}' if agreement_matrix_data else ''}

        // Tab functionality
        function showTab(tabId, contentId) {{
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(contentId).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
        """
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Interactive HTML report generated: {file_path}")
    
    def _save_text_summary(self, file_path: Path, summary_data: Dict, metrics: Dict, modality_analysis: Dict) -> None:
        """Save human-readable text summary"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Unimodal Analysis Summary - {summary_data['model']}\n")
            f.write(f"Generated: {summary_data['timestamp']}\n")
            f.write("=" * 80 + "\n\n")
            
            # Data overview
            f.write("DATA OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Test samples: {summary_data['data_overview']['test_samples']}\n")
            f.write(f"Memory bank (true): {summary_data['data_overview']['memory_bank_true']}\n")
            f.write(f"Memory bank (fake): {summary_data['data_overview']['memory_bank_fake']}\n\n")
            
            # Top performing modalities
            f.write("TOP PERFORMING MODALITIES\n")
            f.write("-" * 40 + "\n")
            f.write("By Accuracy:\n")
            for i, (modality, score) in enumerate(summary_data['best_modalities']['by_accuracy'], 1):
                f.write(f"  {i}. {modality}: {score:.4f}\n")
            
            f.write("\nBy F1 Score:\n")
            for i, (modality, score) in enumerate(summary_data['best_modalities']['by_f1'], 1):
                f.write(f"  {i}. {modality}: {score:.4f}\n")
            
            # Modality agreement analysis
            f.write("\nMODALITY AGREEMENT ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"All modalities correct: {summary_data['modality_agreement']['all_correct_count']} samples ({summary_data['modality_agreement']['all_correct_percentage']}%)\n")
            f.write(f"Only 1 modality correct: {summary_data['modality_agreement']['single_correct_count']} samples ({summary_data['modality_agreement']['single_correct_percentage']}%)\n")
            
            # Detailed performance breakdown
            f.write("\nDETAILED PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Modality':<25} {'Accuracy':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}\n")
            f.write("-" * 67 + "\n")
            for modality, perf in summary_data['modality_performance'].items():
                f.write(f"{modality:<25} {perf['accuracy']:<10.4f} {perf['f1_macro']:<10.4f} {perf['precision_macro']:<12.4f} {perf['recall_macro']:<10.4f}\n")
            
            # Modality combinations insights
            if 'single_correct' in modality_analysis and modality_analysis['single_correct']['count'] > 0:
                f.write("\nSINGLE MODALITY WINNERS\n")
                f.write("-" * 40 + "\n")
                breakdown = modality_analysis['single_correct']['modality_breakdown']
                for modality, samples in breakdown.items():
                    f.write(f"{modality}: {len(samples)} samples\n")
            
            if 'two_correct' in modality_analysis and modality_analysis['two_correct']['count'] > 0:
                f.write("\nTOP MODALITY PAIRS\n")
                f.write("-" * 40 + "\n")
                breakdown = modality_analysis['two_correct']['combination_breakdown']
                sorted_pairs = sorted(breakdown.items(), key=lambda x: len(x[1]), reverse=True)[:5]
                for pair, samples in sorted_pairs:
                    f.write(f"{pair}: {len(samples)} samples\n")
        
    def _analyze_modality_combinations(self, sample_results: List[Dict]) -> Dict:
        """Analyze patterns of modality combinations"""
        # Count samples by number of correct modalities
        by_num_correct = {i: [] for i in range(8)}  # 0-7 modalities
        
        for sample in sample_results:
            num_correct = sample['num_correct']
            if num_correct <= 7:
                by_num_correct[num_correct].append(sample)
        
        # Detailed analysis
        analysis = {}
        
        # Samples with only 1 modality correct
        single_correct = by_num_correct[1]
        analysis['single_correct'] = {
            'count': len(single_correct),
            'samples': [s['video_id'] for s in single_correct],
            'modality_breakdown': {}
        }
        
        for sample in single_correct:
            modality = sample['correct_modalities'][0]
            if modality not in analysis['single_correct']['modality_breakdown']:
                analysis['single_correct']['modality_breakdown'][modality] = []
            analysis['single_correct']['modality_breakdown'][modality].append(sample['video_id'])
        
        # Samples with only 1 modality wrong (6 correct out of 7)
        total_modalities = len(sample_results[0]['predictions']) if sample_results else 0
        if total_modalities > 1:
            single_wrong = by_num_correct[total_modalities - 1]
            analysis['single_wrong'] = {
                'count': len(single_wrong),
                'samples': [s['video_id'] for s in single_wrong],
                'wrong_modality_breakdown': {}
            }
            
            for sample in single_wrong:
                all_modalities = set(sample['predictions'].keys())
                correct_modalities = set(sample['correct_modalities'])
                wrong_modality = list(all_modalities - correct_modalities)[0]
                
                if wrong_modality not in analysis['single_wrong']['wrong_modality_breakdown']:
                    analysis['single_wrong']['wrong_modality_breakdown'][wrong_modality] = []
                analysis['single_wrong']['wrong_modality_breakdown'][wrong_modality].append(sample['video_id'])
        
        # Samples with 2 modalities correct
        two_correct = by_num_correct[2]
        analysis['two_correct'] = {
            'count': len(two_correct),
            'samples': [s['video_id'] for s in two_correct],
            'combination_breakdown': {}
        }
        
        for sample in two_correct:
            combination = '_'.join(sorted(sample['correct_modalities']))
            if combination not in analysis['two_correct']['combination_breakdown']:
                analysis['two_correct']['combination_breakdown'][combination] = []
            analysis['two_correct']['combination_breakdown'][combination].append(sample['video_id'])
        
        # Samples with all modalities correct
        all_correct = by_num_correct[total_modalities] if total_modalities > 0 else []
        analysis['all_correct'] = {
            'count': len(all_correct),
            'samples': [s['video_id'] for s in all_correct]
        }
        
        # Add summary by number of correct modalities
        analysis['summary_by_count'] = {}
        for num_correct, samples in by_num_correct.items():
            if samples:
                analysis['summary_by_count'][num_correct] = {
                    'count': len(samples),
                    'percentage': len(samples) / len(sample_results) * 100
                }
        
        return analysis
        
    def _calculate_metrics(self, all_results: Dict[str, Dict]) -> Dict:
        """Calculate performance metrics for each modality"""
        metrics = {}
        
        for modality_name, modality_results in all_results.items():
            # Collect ground truth and predictions
            y_true = []
            y_pred = []
            
            for test_item in self.test_data:
                video_id = test_item['video_id']
                if video_id in modality_results:
                    y_true.append(test_item['annotation'])
                    y_pred.append(modality_results[video_id]['prediction'])
            
            if len(y_true) > 0:
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='macro', zero_division=0
                )
                
                # Per-class metrics
                labels = ['真', '假']
                precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                    y_true, y_pred, labels=labels, zero_division=0
                )
                
                metrics[modality_name] = {
                    'accuracy': float(accuracy),
                    'precision_macro': float(precision),
                    'recall_macro': float(recall),
                    'f1_macro': float(f1),
                    'per_class': {
                        labels[i]: {
                            'precision': float(precision_per_class[i]),
                            'recall': float(recall_per_class[i]),
                            'f1': float(f1_per_class[i]),
                            'support': int(support_per_class[i])
                        }
                        for i in range(len(labels))
                    },
                    'total_samples': len(y_true)
                }
        
        return metrics
        


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Unimodal retrieval analysis for FakeSV dataset')
    parser.add_argument('--model', type=str,
                       default='OFA-Sys/chinese-clip-vit-large-patch14',
                       help='Hugging Face model name for text embedding')
    parser.add_argument('--data-dir', type=str, default='data/FakeSV',
                       help='Directory containing FakeSV data')
    parser.add_argument('--output-dir', type=str, default='analysis/FakeSV',
                       help='Directory to save analysis results')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for text encoding')
    parser.add_argument('--text-modes', nargs='+', 
                       choices=['title_only', 'title_keywords', 'full_text'],
                       default=['title_only', 'title_keywords', 'full_text'],
                       help='Text combination modes to analyze')
    parser.add_argument('--visual-modes', nargs='+',
                       choices=['frame_max', 'pooled'],
                       default=['frame_max', 'pooled'], 
                       help='Visual aggregation modes to analyze')
    parser.add_argument('--audio-modes', nargs='+',
                       choices=['frame_max', 'pooled'],
                       default=['frame_max', 'pooled'],
                       help='Audio aggregation modes to analyze')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = UnimodalAnalysis(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    # Run analysis
    analyzer.run_full_analysis(
        text_modes=args.text_modes,
        visual_modes=args.visual_modes,
        audio_modes=args.audio_modes
    )