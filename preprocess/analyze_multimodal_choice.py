#!/usr/bin/env python3
"""
Multimodal Choice Analysis for Fusion Method Selection

This script performs comprehensive analysis to determine the optimal fusion method
(Sinkhorn, DSL, RRF, or entropy-gated selection) for multimodal retrieval.

Based on the conversation with the agent, this implements the minimal TODO for 
offline decision making with Event-Hit@1(pair) evaluation criteria.
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
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, entropy
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultimodalChoiceAnalyzer:
    def __init__(self, dataset: str = "FakeSV", 
                 audio_model: str = "CAiRE-SER-wav2vec2-large-xlsr-53-eng-zho-all-age",
                 text_model: str = None,
                 output_dir: str = None):
        """
        Initialize multimodal choice analyzer
        
        Args:
            dataset: Dataset name (e.g., 'FakeSV', 'FakeTT')
            audio_model: Audio model name suffix for loading features
            text_model: Text model for embeddings (auto-selected if None)
            output_dir: Output directory (auto-created if None)
        """
        self.dataset = dataset
        self.audio_model = audio_model
        
        # Auto-select text model if not provided
        if text_model is None:
            text_model = self.get_default_text_model(dataset)
        self.text_model = text_model
        self.text_model_short = text_model.split("/")[-1]
        
        # Setup paths
        self.data_dir = Path(f"data/{dataset}")
        if output_dir is None:
            output_dir = f"analysis/{dataset}/multimodal_choice"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized MultimodalChoiceAnalyzer for {dataset}")
        logger.info(f"Audio model: {audio_model}")
        logger.info(f"Text model: {text_model}")
        logger.info(f"Output dir: {output_dir}")
        
        # Data storage
        self.video_metadata = {}
        self.audio_features = {}
        self.visual_features = {}
        self.text_embeddings = {}
        self.splits = {}
        
        # Analysis results
        self.probs = {}  # T, I, A probability matrices
        self.candidate_meta = {}
        self.query_events = []
        
    def get_default_text_model(self, dataset: str) -> str:
        """Get default text model based on dataset"""
        if dataset.lower() == 'fakesv':
            return 'OFA-Sys/chinese-clip-vit-large-patch14'
        else:
            return 'zer0int/LongCLIP-GmP-ViT-L-14'
            
    def load_data(self):
        """Load all required data"""
        logger.info("Loading video metadata...")
        self._load_video_metadata()
        
        logger.info("Loading temporal splits...")
        self._load_temporal_splits()
        
        logger.info("Loading multimodal features...")
        self._load_multimodal_features()
        
        logger.info("Loading text model and computing embeddings...")
        self._load_text_model()
        self._compute_text_embeddings()
        
        logger.info("Data loading complete")
    
    def _load_video_metadata(self):
        """Load video descriptions and metadata"""
        llm_file = self.data_dir / "llm_video_descriptions.jsonl"
        if not llm_file.exists():
            raise FileNotFoundError(f"Video descriptions file not found: {llm_file}")
            
        with open(llm_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    video_id = item['video_id']
                    
                    # Extract keywords/event for evaluation
                    keywords_or_event = item.get('keywords', '') or item.get('event', '')
                    
                    self.video_metadata[video_id] = {
                        'annotation': item.get('annotation', ''),
                        'title': item.get('title', ''),
                        'keywords': keywords_or_event,
                        'description': item.get('description', ''),
                        'temporal_evolution': item.get('temporal_evolution', ''),
                        'entity_claims': item.get('entity_claims', {}),
                        'is_real': item.get('annotation', '').lower() in ['real', '真'],
                        'is_fake': item.get('annotation', '').lower() in ['fake', '假']
                    }
        
        logger.info(f"Loaded metadata for {len(self.video_metadata)} videos")
    
    def _load_temporal_splits(self):
        """Load train/valid/test splits"""
        vid_dir = self.data_dir / "vids"
        
        for split in ['train', 'valid', 'test']:
            split_file = vid_dir / f"vid_time3_{split}.txt"
            if split_file.exists():
                with open(split_file, 'r') as f:
                    self.splits[split] = set(line.strip() for line in f)
                logger.info(f"{split.capitalize()} split: {len(self.splits[split])} videos")
            else:
                self.splits[split] = set()
                logger.warning(f"Split file not found: {split_file}")
    
    def _load_multimodal_features(self):
        """Load audio and visual features"""
        feature_dir = self.data_dir / "fea"
        
        # Load visual features
        vit_file = feature_dir / "vit_tensor.pt"
        if vit_file.exists():
            raw_visual = torch.load(vit_file, map_location='cpu')
            # Mean pool frame-level features
            for video_id, frames in raw_visual.items():
                if frames.shape[0] > 0:  # Ensure not empty
                    pooled = torch.mean(frames, dim=0)  # [1024]
                    self.visual_features[video_id] = pooled.numpy()
            logger.info(f"Loaded and pooled visual features for {len(self.visual_features)} videos")
        else:
            logger.error(f"Visual features file not found: {vit_file}")
            
        # Load audio features with model suffix
        audio_file = feature_dir / f"audio_features_frames_{self.audio_model}.pt"
        if audio_file.exists():
            raw_audio = torch.load(audio_file, map_location='cpu')
            # Mean pool frame-level features
            for video_id, frames in raw_audio.items():
                if frames.shape[0] > 0:  # Ensure not empty
                    pooled = torch.mean(frames, dim=0)  # [1024]
                    self.audio_features[video_id] = pooled.numpy()
            logger.info(f"Loaded and pooled audio features for {len(self.audio_features)} videos")
        else:
            logger.error(f"Audio features file not found: {audio_file}")
    
    def _load_text_model(self):
        """Load text embedding model"""
        logger.info(f"Loading text model: {self.text_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model)
        self.model = AutoModel.from_pretrained(self.text_model)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Text model loaded on {self.device}")
    
    def _compute_text_embeddings(self):
        """Compute text embeddings for all videos"""
        video_ids = []
        texts = []
        
        # Create text representations (title only by default)
        for video_id, metadata in self.video_metadata.items():
            # Use title only as specified in requirements
            text_repr = metadata.get('title', '') or 'unknown'
            video_ids.append(video_id)
            texts.append(text_repr)
        
        # Encode in batches
        batch_size = 16
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing text embeddings"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                if 'chinese-clip' in self.text_model.lower():
                    max_len = 512
                elif 'longclip' in self.text_model.lower():
                    max_len = 248
                else:
                    max_len = 77
                    
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                if 'chinese-clip' in self.text_model.lower():
                    outputs = self.model.text_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                elif 'longclip' in self.text_model.lower():
                    outputs = self.model.text_model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                else:
                    outputs = self.model.text_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Combine all embeddings
        all_embeddings = np.vstack(all_embeddings)
        
        # Store embeddings
        for video_id, embedding in zip(video_ids, all_embeddings):
            self.text_embeddings[video_id] = embedding
        
        logger.info(f"Computed text embeddings for {len(self.text_embeddings)} videos")
    
    def prepare_analysis_data(self):
        """Prepare data matrices for analysis"""
        logger.info("Preparing analysis data matrices...")
        
        # Show data availability breakdown
        total_in_splits = sum(len(ids) for ids in self.splits.values())
        logger.info(f"Data availability breakdown:")
        logger.info(f"  Videos in splits: {total_in_splits}")
        logger.info(f"  Videos with metadata: {len(self.video_metadata)}")
        logger.info(f"  Videos with text embeddings: {len(self.text_embeddings)}")
        logger.info(f"  Videos with visual features: {len(self.visual_features)}")
        logger.info(f"  Videos with audio features: {len(self.audio_features)}")
        
        # Get all video IDs that have all three modalities
        common_video_ids = set(self.video_metadata.keys()) & \
                          set(self.text_embeddings.keys()) & \
                          set(self.visual_features.keys()) & \
                          set(self.audio_features.keys())
        
        # Filter by available splits
        split_video_ids = set()
        for split_videos in self.splits.values():
            split_video_ids.update(split_videos)
        
        valid_video_ids = list(common_video_ids & split_video_ids)
        logger.info(f"Videos with all modalities AND in splits: {len(valid_video_ids)}")
        
        # Build candidate database (memory bank: train + valid)
        memory_bank_ids = self.splits['train'] | self.splits['valid']
        candidate_ids = list(memory_bank_ids & set(valid_video_ids))
        logger.info(f"Memory bank size: {len(candidate_ids)} candidates")
        
        # Build query set (all splits)
        query_ids = valid_video_ids
        logger.info(f"Query set size: {len(query_ids)} queries")
        
        # Extract features into matrices
        N = len(candidate_ids)  # Number of candidates
        Q = len(query_ids)      # Number of queries
        
        # Text features
        text_candidates = np.stack([self.text_embeddings[vid] for vid in candidate_ids])
        text_queries = np.stack([self.text_embeddings[vid] for vid in query_ids])
        
        # Visual features  
        visual_candidates = np.stack([self.visual_features[vid] for vid in candidate_ids])
        visual_queries = np.stack([self.visual_features[vid] for vid in query_ids])
        
        # Audio features
        audio_candidates = np.stack([self.audio_features[vid] for vid in candidate_ids])
        audio_queries = np.stack([self.audio_features[vid] for vid in query_ids])
        
        # L2 normalize features
        text_candidates = text_candidates / (np.linalg.norm(text_candidates, axis=1, keepdims=True) + 1e-12)
        text_queries = text_queries / (np.linalg.norm(text_queries, axis=1, keepdims=True) + 1e-12)
        visual_candidates = visual_candidates / (np.linalg.norm(visual_candidates, axis=1, keepdims=True) + 1e-12)
        visual_queries = visual_queries / (np.linalg.norm(visual_queries, axis=1, keepdims=True) + 1e-12)
        audio_candidates = audio_candidates / (np.linalg.norm(audio_candidates, axis=1, keepdims=True) + 1e-12)
        audio_queries = audio_queries / (np.linalg.norm(audio_queries, axis=1, keepdims=True) + 1e-12)
        
        # Compute similarity matrices 
        text_sim = np.dot(text_queries, text_candidates.T)  # [Q, N]
        visual_sim = np.dot(visual_queries, visual_candidates.T)  # [Q, N]
        audio_sim = np.dot(audio_queries, audio_candidates.T)  # [Q, N]
        
        # Store similarity scores (for DBNorm)
        self.scores = {}
        self.scores['T'] = text_sim
        self.scores['I'] = visual_sim  
        self.scores['A'] = audio_sim
        
        # Convert to probabilities
        temp = 1.0  # Temperature for softmax
        self.probs = {}
        self.probs['T'] = self._softmax(text_sim / temp, axis=1)
        self.probs['I'] = self._softmax(visual_sim / temp, axis=1)
        self.probs['A'] = self._softmax(audio_sim / temp, axis=1)
        
        # Build candidate metadata
        candidate_events = []
        candidate_labels = []
        for vid in candidate_ids:
            metadata = self.video_metadata[vid]
            candidate_events.append(metadata['keywords'])
            candidate_labels.append(1 if metadata['is_real'] else 0)
        
        self.candidate_meta = {
            'candidate_event': np.array(candidate_events),
            'candidate_label': np.array(candidate_labels),
            'candidate_ids': np.array(candidate_ids)
        }
        
        # Build query events
        self.query_events = []
        self.query_ids = query_ids
        for vid in query_ids:
            self.query_events.append(self.video_metadata[vid]['keywords'])
        self.query_events = np.array(self.query_events)
        
        # Handle self-exclusion for train/valid queries
        self._apply_self_exclusion()
        
        logger.info(f"Analysis data prepared: {Q} queries × {N} candidates")
        logger.info(f"Probability matrices shape: {self.probs['T'].shape}")
    
    def _softmax(self, x, axis=1):
        """Stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _apply_self_exclusion(self):
        """Apply self-exclusion for train/valid queries"""
        candidate_ids_set = set(self.candidate_meta['candidate_ids'])
        
        for i, query_id in enumerate(self.query_ids):
            query_split = None
            for split, ids in self.splits.items():
                if query_id in ids:
                    query_split = split
                    break
            
            # Apply self-exclusion for train/valid
            if query_split in ['train', 'valid'] and query_id in candidate_ids_set:
                candidate_idx = list(self.candidate_meta['candidate_ids']).index(query_id)
                
                # Set probability to 0 and renormalize
                for modality in self.probs:
                    self.probs[modality][i, candidate_idx] = 0
                    # Renormalize
                    prob_sum = np.sum(self.probs[modality][i])
                    if prob_sum > 0:
                        self.probs[modality][i] /= prob_sum
        
        logger.info("Applied self-exclusion for train/valid queries")
    
    def _apply_self_exclusion_to(self, probs_dict):
        """Apply self-exclusion to any probability dictionary"""
        # Create mapping for fast lookup
        id2idx = {vid: i for i, vid in enumerate(self.candidate_meta['candidate_ids'])}
        
        for qi, qid in enumerate(self.query_ids):
            # Find query split
            query_split = None
            for split, ids in self.splits.items():
                if qid in ids:
                    query_split = split
                    break
            
            # Apply exclusion for train/valid
            if query_split in ['train', 'valid'] and qid in id2idx:
                ci = id2idx[qid]
                for m in probs_dict:
                    row = probs_dict[m][qi].copy()  # Work on copy
                    row[ci] = 0.0
                    s = row.sum()
                    if s > 0:
                        probs_dict[m][qi] = row / s
        
        return probs_dict
    
    def compute_single_modality_diagnostics(self) -> pd.DataFrame:
        """Compute single modality diagnostics"""
        logger.info("Computing single modality diagnostics...")
        
        results = []
        
        for modality in ['T', 'I', 'A']:
            modality_name = {'T': 'Text', 'I': 'Visual', 'A': 'Audio'}[modality]
            probs_m = self.probs[modality]  # [Q, N]
            
            # 1. Event-Hit@1(pair)
            hit_scores = []
            p_event_masses = []
            entropies = []
            margins = []
            
            for q in range(len(self.query_ids)):
                query_event = self.query_events[q]
                query_probs = probs_m[q]  # [N]
                
                # Find positive and negative candidates
                pos_mask = self.candidate_meta['candidate_label'] == 1
                neg_mask = self.candidate_meta['candidate_label'] == 0
                
                if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
                    # Top-1 positive and negative
                    pos_probs = query_probs[pos_mask]
                    neg_probs = query_probs[neg_mask]
                    
                    top1_pos_idx = np.argmax(pos_probs)
                    top1_neg_idx = np.argmax(neg_probs)
                    
                    # Get corresponding events
                    pos_events = self.candidate_meta['candidate_event'][pos_mask]
                    neg_events = self.candidate_meta['candidate_event'][neg_mask]
                    
                    top1_pos_event = pos_events[top1_pos_idx]
                    top1_neg_event = neg_events[top1_neg_idx]
                    
                    # Check hit
                    hit = 1 if (top1_pos_event == query_event or top1_neg_event == query_event) else 0
                    hit_scores.append(hit)
                else:
                    # Ensure consistent denominator
                    hit_scores.append(0)
                    
                # 2. p_event_mass
                event_mask = self.candidate_meta['candidate_event'] == query_event
                p_event_mass = np.sum(query_probs[event_mask])
                p_event_masses.append(p_event_mass)
                
                # 3. Uncertainty metrics
                H = entropy(query_probs + 1e-12)  # Entropy
                sorted_probs = np.sort(query_probs)[::-1]
                margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
                
                entropies.append(H)
                margins.append(margin)
            
            # Aggregate results
            results.append({
                'Modality': modality_name,
                'Event-Hit@1(pair)': np.mean(hit_scores) if hit_scores else 0.0,
                'Mean_p_event_mass': np.mean(p_event_masses),
                'Mean_entropy': np.mean(entropies),
                'Mean_top2_margin': np.mean(margins)
            })
        
        df = pd.DataFrame(results)
        output_file = self.output_dir / "diagnostics_single.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved single modality diagnostics to {output_file}")
        
        return df
    
    def compute_modality_pair_analysis(self) -> pd.DataFrame:
        """Compute modality pair complementarity analysis"""
        logger.info("Computing modality pair analysis...")
        
        results = []
        modality_pairs = [('T', 'I'), ('T', 'A'), ('I', 'A')]
        modality_names = {'T': 'Text', 'I': 'Visual', 'A': 'Audio'}
        
        for m1, m2 in modality_pairs:
            pair_name = f"{modality_names[m1]}-{modality_names[m2]}"
            
            # Get probability matrices
            probs_m1 = self.probs[m1]  # [Q, N]
            probs_m2 = self.probs[m2]  # [Q, N]
            
            # Compute metrics across all queries
            correlations = []
            jsds = []
            m1_wins = 0
            m2_wins = 0
            ties = 0
            
            for q in range(len(self.query_ids)):
                p1 = probs_m1[q]
                p2 = probs_m2[q]
                
                # 1. Spearman correlation on rankings
                ranks1 = (-p1).argsort().argsort()  # Convert to ranks (higher prob = lower rank)
                ranks2 = (-p2).argsort().argsort()
                corr, _ = spearmanr(ranks1, ranks2)
                correlations.append(corr if not np.isnan(corr) else 0.0)
                
                # 2. Jensen-Shannon Divergence
                jsd = jensenshannon(p1, p2) ** 2  # Squared to get proper JSD
                jsds.append(jsd if not np.isnan(jsd) else 0.0)
                
                # 3. Event-Hit@1 comparison for this query
                query_event = self.query_events[q]
                
                def compute_event_hit(probs):
                    pos_mask = self.candidate_meta['candidate_label'] == 1
                    neg_mask = self.candidate_meta['candidate_label'] == 0
                    
                    if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
                        pos_probs = probs[pos_mask]
                        neg_probs = probs[neg_mask]
                        
                        top1_pos_idx = np.argmax(pos_probs)
                        top1_neg_idx = np.argmax(neg_probs)
                        
                        pos_events = self.candidate_meta['candidate_event'][pos_mask]
                        neg_events = self.candidate_meta['candidate_event'][neg_mask]
                        
                        top1_pos_event = pos_events[top1_pos_idx]
                        top1_neg_event = neg_events[top1_neg_idx]
                        
                        return 1 if (top1_pos_event == query_event or top1_neg_event == query_event) else 0
                    return 0
                
                hit1 = compute_event_hit(p1)
                hit2 = compute_event_hit(p2)
                
                if hit1 > hit2:
                    m1_wins += 1
                elif hit2 > hit1:
                    m2_wins += 1
                else:
                    ties += 1
            
            # Aggregate results
            total_queries = len(self.query_ids)
            results.append({
                'Modality_Pair': pair_name,
                'Mean_Spearman': np.mean(correlations),
                'Mean_JSD': np.mean(jsds),
                'M1_Win_Ratio': m1_wins / total_queries,
                'M2_Win_Ratio': m2_wins / total_queries,
                'Tie_Ratio': ties / total_queries
            })
        
        df = pd.DataFrame(results)
        output_file = self.output_dir / "diagnostics_pairs.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved modality pair analysis to {output_file}")
        
        return df
    
    def compute_hubness_analysis(self) -> Tuple[pd.DataFrame, Dict]:
        """Compute hubness analysis"""
        logger.info("Computing hubness analysis...")
        
        # Count top-10 appearances for each candidate across all modalities
        candidate_top10_counts = np.zeros(len(self.candidate_meta['candidate_ids']))
        
        for modality in ['T', 'I', 'A']:
            probs_m = self.probs[modality]  # [Q, N]
            
            # For each query, find top-10 candidates
            top10_indices = np.argpartition(probs_m, -10, axis=1)[:, -10:]  # [Q, 10]
            
            # Count appearances
            for q_idx in range(len(self.query_ids)):
                for candidate_idx in top10_indices[q_idx]:
                    candidate_top10_counts[candidate_idx] += 1
        
        # Create hubness dataframe
        hubness_df = pd.DataFrame({
            'candidate_id': self.candidate_meta['candidate_ids'],
            'freq_top10_all_modalities': candidate_top10_counts
        })
        
        # Sort by frequency (descending)
        hubness_df = hubness_df.sort_values('freq_top10_all_modalities', ascending=False)
        
        # Save hubness data
        hubness_file = self.output_dir / "hubness.csv"
        hubness_df.to_csv(hubness_file, index=False)
        logger.info(f"Saved hubness data to {hubness_file}")
        
        # Calculate Gini coefficient
        def gini_coefficient(x):
            """Calculate Gini coefficient"""
            sorted_x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(sorted_x)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        gini = gini_coefficient(candidate_top10_counts)
        
        # Calculate additional statistics
        sorted_counts = np.sort(candidate_top10_counts)[::-1]  # Sort descending
        top_20_freq = float(sorted_counts[:20].sum())  # Top 20 by frequency
        total_top10_positions = float(len(self.query_ids) * 10 * 3)  # Q*10*3 not Q*3
        
        hubness_summary = {
            'gini_coefficient': gini,
            'top_20_candidates_freq': int(top_20_freq),
            'total_possible_top10': int(total_top10_positions),  # Renamed for clarity
            'top_20_dominance_ratio': top_20_freq / total_top10_positions if total_top10_positions > 0 else 0.0,
            'max_frequency': int(np.max(candidate_top10_counts)),
            'mean_frequency': float(np.mean(candidate_top10_counts)),
            'median_frequency': float(np.median(candidate_top10_counts))
        }
        
        # Save summary
        summary_file = self.output_dir / "hubness_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Hubness Analysis Summary\n")
            f.write("=======================\n\n")
            for key, value in hubness_summary.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Saved hubness summary to {summary_file}")
        logger.info(f"Gini coefficient: {gini:.3f}")
        
        return hubness_df, hubness_summary
    
    def compute_fusion_methods_comparison(self, hubness_summary: Dict) -> pd.DataFrame:
        """Compare different fusion methods"""
        logger.info("Computing fusion methods comparison...")
        
        methods = ['DSL', 'Sinkhorn', 'RRF', 'Entropy-Gated', 'UncertaintyWeightedLogPool']
        
        # Add DBNorm methods if hubness is high
        use_dbnorm = hubness_summary['gini_coefficient'] >= 0.30
        if use_dbnorm:
            methods.extend(['DBNorm+DSL', 'DBNorm+Sinkhorn'])
            logger.info("High hubness detected, adding DBNorm methods")
        
        results = []
        
        for method in methods:
            logger.info(f"Evaluating {method}...")
            
            # Get fused probabilities
            if method == 'DSL':
                fused_probs = self._dsl_fusion()
            elif method == 'Sinkhorn':
                fused_probs = self._sinkhorn_fusion()
            elif method == 'RRF':
                fused_probs = self._rrf_fusion()
            elif method == 'Entropy-Gated':
                fused_probs = self._entropy_gated_fusion()
            elif method == 'UncertaintyWeightedLogPool':
                fused_probs = self._uncertainty_weighted_log_pool()
            elif method == 'DBNorm+DSL':
                dbnorm_probs = self._apply_dbnorm_scores()
                dbnorm_probs = self._apply_self_exclusion_to(dbnorm_probs)
                fused_probs = self._dsl_fusion(dbnorm_probs)
            elif method == 'DBNorm+Sinkhorn':
                dbnorm_probs = self._apply_dbnorm_scores()
                dbnorm_probs = self._apply_self_exclusion_to(dbnorm_probs)
                fused_probs = self._sinkhorn_fusion(dbnorm_probs)
            
            # Evaluate Event-Hit@1(pair) and p_event_mass
            event_hits = []
            p_event_masses = []
            
            for q in range(len(self.query_ids)):
                query_event = self.query_events[q]
                query_probs = fused_probs[q]
                
                # Event-Hit@1(pair)
                pos_mask = self.candidate_meta['candidate_label'] == 1
                neg_mask = self.candidate_meta['candidate_label'] == 0
                
                if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
                    pos_probs = query_probs[pos_mask]
                    neg_probs = query_probs[neg_mask]
                    
                    top1_pos_idx = np.argmax(pos_probs)
                    top1_neg_idx = np.argmax(neg_probs)
                    
                    pos_events = self.candidate_meta['candidate_event'][pos_mask]
                    neg_events = self.candidate_meta['candidate_event'][neg_mask]
                    
                    top1_pos_event = pos_events[top1_pos_idx]
                    top1_neg_event = neg_events[top1_neg_idx]
                    
                    hit = 1 if (top1_pos_event == query_event or top1_neg_event == query_event) else 0
                    event_hits.append(hit)
                else:
                    # Ensure consistent denominator
                    event_hits.append(0)
                
                # p_event_mass
                event_mask = self.candidate_meta['candidate_event'] == query_event
                p_event_mass = np.sum(query_probs[event_mask])
                p_event_masses.append(p_event_mass)
            
            # Bootstrap confidence intervals
            n_bootstrap = 1000
            event_hit_boots = []
            p_event_mass_boots = []
            
            for _ in range(n_bootstrap):
                boot_indices = np.random.choice(len(event_hits), size=len(event_hits), replace=True)
                event_hit_boots.append(np.mean([event_hits[i] for i in boot_indices]))
                p_event_mass_boots.append(np.mean([p_event_masses[i] for i in boot_indices]))
            
            event_hit_ci = np.percentile(event_hit_boots, [2.5, 97.5])
            p_event_mass_ci = np.percentile(p_event_mass_boots, [2.5, 97.5])
            
            results.append({
                'Method': method,
                'Event-Hit@1(pair)': np.mean(event_hits) if event_hits else 0.0,
                'Event-Hit_CI_low': event_hit_ci[0],
                'Event-Hit_CI_high': event_hit_ci[1],
                'Mean_p_event_mass': np.mean(p_event_masses),
                'p_event_mass_CI_low': p_event_mass_ci[0],
                'p_event_mass_CI_high': p_event_mass_ci[1]
            })
        
        df = pd.DataFrame(results)
        output_file = self.output_dir / "method_compare.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved method comparison to {output_file}")
        
        return df
    
    def _dsl_fusion(self, probs_dict=None):
        """Direct Softmax Late fusion"""
        if probs_dict is None:
            probs_dict = self.probs
            
        # Element-wise product of probabilities (already softmaxed)
        fused = probs_dict['T'] * probs_dict['I'] * probs_dict['A']
        
        # Renormalize
        fused = fused / (np.sum(fused, axis=1, keepdims=True) + 1e-12)
        
        return fused
    
    def _sinkhorn_fusion(self, probs_dict=None, n_iters=10):
        """Sinkhorn fusion with column marginal target"""
        if probs_dict is None:
            probs_dict = self.probs
            
        # Set column marginal target as average of three modalities
        col_target = (probs_dict['T'] + probs_dict['I'] + probs_dict['A']) / 3.0
        log_col_target = np.log(col_target + 1e-12)  # [Q, N]
        
        # Convert to log space
        log_probs = {}
        for m in ['T', 'I', 'A']:
            log_probs[m] = np.log(probs_dict[m] + 1e-12)
        
        # Stack into 3D array: [Q, 3, N]
        Q, N = log_probs['T'].shape
        log_prob_stack = np.stack([log_probs['T'], log_probs['I'], log_probs['A']], axis=1)
        
        # Sinkhorn iterations with target marginal
        for _ in range(n_iters):
            # Row normalization (across modalities)
            log_prob_stack = log_prob_stack - logsumexp(log_prob_stack, axis=1, keepdims=True)
            
            # Column normalization to target marginal
            current_col_marginal = logsumexp(log_prob_stack, axis=1)  # [Q, N]
            adjustment = log_col_target - current_col_marginal  # [Q, N]
            log_prob_stack = log_prob_stack + adjustment[:, None, :]  # Broadcast to [Q, 3, N]
        
        # Get final column marginals
        fused_log_probs = logsumexp(log_prob_stack, axis=1)  # [Q, N]
        
        # Convert back to probabilities
        fused = np.exp(fused_log_probs)
        fused = fused / (np.sum(fused, axis=1, keepdims=True) + 1e-12)
        
        return fused
    
    def _rrf_fusion(self, k=60):
        """Reciprocal Rank Fusion"""
        Q, N = self.probs['T'].shape
        fused_scores = np.zeros((Q, N))
        
        for modality in ['T', 'I', 'A']:
            probs_m = self.probs[modality]
            
            # Convert probabilities to ranks (higher prob = lower rank number)
            ranks = (-probs_m).argsort(axis=1).argsort(axis=1) + 1  # +1 for 1-based ranking
            
            # RRF score: 1 / (k + rank)
            rrf_scores = 1.0 / (k + ranks)
            fused_scores += rrf_scores
        
        # Convert scores to probabilities
        fused = self._softmax(fused_scores, axis=1)
        
        return fused
    
    def _entropy_gated_fusion(self):
        """Entropy-gated selection (choose modality with lowest entropy per query)"""
        Q, N = self.probs['T'].shape
        fused = np.zeros((Q, N))
        
        for q in range(Q):
            # Calculate entropy for each modality for this query
            entropies = {}
            for modality in ['T', 'I', 'A']:
                p = self.probs[modality][q]
                entropies[modality] = entropy(p + 1e-12)
            
            # Choose modality with minimum entropy
            best_modality = min(entropies, key=entropies.get)
            fused[q] = self.probs[best_modality][q]
        
        return fused
    
    def _uncertainty_weighted_log_pool(self, eps=1e-12):
        """Uncertainty Weighted Log Pool fusion"""
        # 获取每个模态的概率矩阵
        P = {'T': self.probs['T'], 'I': self.probs['I'], 'A': self.probs['A']}  # [Q,N]
        Q, N = P['T'].shape
        
        # 1) 计算确定性分数 c_m(q) - 使用熵的倒数
        C = {}
        for m in P:
            # 默认：熵的倒数
            ent = np.apply_along_axis(lambda r: entropy(r + eps), 1, P[m])  # [Q]
            C[m] = 1.0 / (ent + eps)
        
        # 2) 归一化权重 w_m(q)
        W = {m: C[m] / (C['T'] + C['I'] + C['A']) for m in P}  # 每个都是 [Q]
        
        # 3) 对数线性池
        fused_log = np.zeros_like(P['T'])
        for m in P:
            fused_log += W[m][:, None] * np.log(P[m] + eps)
        
        fused = np.exp(fused_log)
        fused /= fused.sum(axis=1, keepdims=True) + eps
        
        return fused
    
    def _apply_dbnorm_scores(self, scores_dict=None, zscore=False):
        """Apply DBNorm to similarity scores (not probabilities)"""
        if scores_dict is None:
            scores_dict = self.scores
            
        dbnorm_probs = {}
        for modality, S in scores_dict.items():  # S: [Q, N] similarity scores
            # Column-wise mean (per candidate across queries)
            col_mean = S.mean(axis=0, keepdims=True)
            S_adjusted = S - col_mean
            
            if zscore:
                col_std = S.std(axis=0, keepdims=True) + 1e-6
                S_adjusted = S_adjusted / col_std
                
            # Convert to probabilities with softmax
            dbnorm_probs[modality] = self._softmax(S_adjusted, axis=1)
        
        return dbnorm_probs
    
    def _sinkhorn_fusion_pair(self, pair_probs, n_iters=10):
        """Sinkhorn fusion for two modalities only"""
        # Get modality keys
        keys = list(pair_probs.keys())
        
        # Set target as average of two
        col_target = (pair_probs[keys[0]] + pair_probs[keys[1]]) / 2.0
        log_col_target = np.log(col_target + 1e-12)
        
        # Convert to log space
        log_probs = {}
        for m in keys:
            log_probs[m] = np.log(pair_probs[m] + 1e-12)
        
        # Stack into 3D array: [Q, 2, N]
        Q, N = log_probs[keys[0]].shape
        log_prob_stack = np.stack([log_probs[keys[0]], log_probs[keys[1]]], axis=1)
        
        # Sinkhorn iterations
        for _ in range(n_iters):
            # Row normalization
            log_prob_stack = log_prob_stack - logsumexp(log_prob_stack, axis=1, keepdims=True)
            
            # Column normalization to target
            current_col_marginal = logsumexp(log_prob_stack, axis=1)
            adjustment = log_col_target - current_col_marginal
            log_prob_stack = log_prob_stack + adjustment[:, None, :]
        
        # Get final marginals
        fused_log_probs = logsumexp(log_prob_stack, axis=1)
        fused = np.exp(fused_log_probs)
        fused = fused / (np.sum(fused, axis=1, keepdims=True) + 1e-12)
        
        return fused
    
    def _evaluate_event_hit(self, probs):
        """Evaluate Event-Hit@1(pair) for given probability matrix"""
        hits = []
        for q in range(len(self.query_ids)):
            query_event = self.query_events[q]
            query_probs = probs[q]
            
            pos_mask = self.candidate_meta['candidate_label'] == 1
            neg_mask = self.candidate_meta['candidate_label'] == 0
            
            if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
                pos_probs = query_probs[pos_mask]
                neg_probs = query_probs[neg_mask]
                
                top1_pos_idx = np.argmax(pos_probs)
                top1_neg_idx = np.argmax(neg_probs)
                
                pos_events = self.candidate_meta['candidate_event'][pos_mask]
                neg_events = self.candidate_meta['candidate_event'][neg_mask]
                
                top1_pos_event = pos_events[top1_pos_idx]
                top1_neg_event = neg_events[top1_neg_idx]
                
                hit = 1 if (top1_pos_event == query_event or top1_neg_event == query_event) else 0
                hits.append(hit)
            else:
                hits.append(0)  # Consistent denominator
        
        return np.mean(hits)
    
    def create_visualizations(self, single_df: pd.DataFrame, pairs_df: pd.DataFrame, 
                            hubness_df: pd.DataFrame, methods_df: pd.DataFrame, hubness_summary: Dict):
        """Create the three required visualizations"""
        logger.info("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Figure A: Entropy distributions
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Collect entropy data for each modality
        entropy_data = []
        modality_labels = []
        
        for modality in ['T', 'I', 'A']:
            probs_m = self.probs[modality]
            for q in range(len(self.query_ids)):
                H = entropy(probs_m[q] + 1e-12)
                entropy_data.append(H)
                modality_labels.append({'T': 'Text', 'I': 'Visual', 'A': 'Audio'}[modality])
        
        entropy_df = pd.DataFrame({
            'Entropy': entropy_data,
            'Modality': modality_labels
        })
        
        sns.boxplot(data=entropy_df, x='Modality', y='Entropy', ax=ax)
        ax.set_title('Entropy Distributions Across Modalities', fontsize=14)
        ax.set_xlabel('Modality', fontsize=12)
        ax.set_ylabel('Entropy', fontsize=12)
        
        plt.tight_layout()
        entropy_fig_path = self.output_dir / "entropy_distribution.png"
        plt.savefig(entropy_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure A2: Entropy distributions with density curves
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot density curves for each modality
        for modality, label in [('T', 'Text'), ('I', 'Visual'), ('A', 'Audio')]:
            modality_data = entropy_df[entropy_df['Modality'] == label]['Entropy']
            modality_data.plot(kind='density', ax=ax, label=label, linewidth=2)
        
        ax.set_title('Entropy Distributions Across Modalities (Density)', fontsize=14)
        ax.set_xlabel('Entropy', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        entropy_density_path = self.output_dir / "entropy_distribution_density.png"
        plt.savefig(entropy_density_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure A3: Entropy distributions with histograms
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        
        for idx, (modality, label) in enumerate([('T', 'Text'), ('I', 'Visual'), ('A', 'Audio')]):
            ax = axes[idx]
            modality_data = entropy_df[entropy_df['Modality'] == label]['Entropy']
            ax.hist(modality_data, bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'{label} Entropy Distribution', fontsize=12)
            ax.set_xlabel('Entropy', fontsize=10)
            if idx == 0:
                ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = modality_data.mean()
            std_val = modality_data.std()
            ax.text(0.7, 0.95, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('Entropy Distributions Across Modalities (Histograms)', fontsize=14)
        plt.tight_layout()
        entropy_hist_path = self.output_dir / "entropy_distribution_histogram.png"
        plt.savefig(entropy_hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure B: JSD vs Pairwise Sinkhorn improvement
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Get JSD values from pairs analysis
        jsds = pairs_df['Mean_JSD'].values
        pair_names = pairs_df['Modality_Pair'].values
        
        # Compute pairwise fusion improvements
        pairwise_improvements = []
        modality_map = {'Text-Visual': ('T', 'I'), 'Text-Audio': ('T', 'A'), 'Visual-Audio': ('I', 'A')}
        
        for pair_name in pair_names:
            m1, m2 = modality_map[pair_name]
            
            # Create pairwise probability dict
            pair_probs = {m1: self.probs[m1], m2: self.probs[m2]}
            
            # Compute DSL for this pair
            dsl_pair = self.probs[m1] * self.probs[m2]
            dsl_pair = dsl_pair / (np.sum(dsl_pair, axis=1, keepdims=True) + 1e-12)
            
            # Compute Sinkhorn for this pair
            sinkhorn_pair = self._sinkhorn_fusion_pair(pair_probs)
            
            # Evaluate both methods on Event-Hit@1(pair)
            dsl_hit = self._evaluate_event_hit(dsl_pair)
            sinkhorn_hit = self._evaluate_event_hit(sinkhorn_pair)
            
            improvement = sinkhorn_hit - dsl_hit
            pairwise_improvements.append(improvement)
        
        # Create scatter plot with actual pairwise improvements
        colors = ['red', 'green', 'blue']
        for i, (jsd, improvement, pair_name) in enumerate(zip(jsds, pairwise_improvements, pair_names)):
            ax.scatter(jsd, improvement, color=colors[i], s=100, label=pair_name, alpha=0.7)
        
        ax.set_xlabel('Jensen-Shannon Divergence', fontsize=12)
        ax.set_ylabel('Pairwise Sinkhorn vs DSL Event-Hit@1 Improvement', fontsize=12)
        ax.set_title('Modality Divergence vs Pairwise Fusion Performance', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add meaningful trend line
        if len(set(pairwise_improvements)) > 1:  # Only if improvements vary
            z = np.polyfit(jsds, pairwise_improvements, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(jsds), max(jsds), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, 
                   label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
            ax.legend()
        
        plt.tight_layout()
        jsd_fig_path = self.output_dir / "jsd_vs_improvement.png"
        plt.savefig(jsd_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure C: Hubness curve
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        frequencies = hubness_df['freq_top10_all_modalities'].values
        candidate_ranks = np.arange(1, len(frequencies) + 1)
        
        ax.loglog(candidate_ranks, frequencies, 'b-', linewidth=2)
        ax.set_xlabel('Candidate Rank', fontsize=12)
        ax.set_ylabel('Top-10 Frequency', fontsize=12)
        ax.set_title('Hubness Distribution: Candidate Top-10 Frequencies', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add annotation for top candidates
        top_5_freq = frequencies[:5]
        ax.scatter(range(1, 6), top_5_freq, color='red', s=50, zorder=5)
        
        # Add Gini coefficient annotation
        gini = hubness_summary['gini_coefficient']
        ax.text(0.7, 0.9, f'Gini = {gini:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=12)
        
        plt.tight_layout()
        hubness_fig_path = self.output_dir / "hubness_curve.png"
        plt.savefig(hubness_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualizations:")
        logger.info(f"  - Entropy distribution: {entropy_fig_path}")
        logger.info(f"  - JSD vs improvement: {jsd_fig_path}")
        logger.info(f"  - Hubness curve: {hubness_fig_path}")
    
    def generate_decision_summary(self, single_df: pd.DataFrame, pairs_df: pd.DataFrame,
                                hubness_summary: Dict, methods_df: pd.DataFrame):
        """Generate decision summary with recommendations"""
        logger.info("Generating decision summary...")
        
        summary = []
        summary.append("="*60)
        summary.append("MULTIMODAL FUSION METHOD DECISION SUMMARY")
        summary.append("="*60)
        summary.append("")
        
        # Key metrics
        summary.append("KEY METRICS:")
        summary.append("-" * 20)
        
        # Modality performance
        for _, row in single_df.iterrows():
            summary.append(f"{row['Modality']:>8}: Event-Hit@1={row['Event-Hit@1(pair)']:.3f}, "
                          f"p_event_mass={row['Mean_p_event_mass']:.3f}, "
                          f"entropy={row['Mean_entropy']:.3f}")
        
        summary.append("")
        
        # Modality complementarity
        summary.append("MODALITY COMPLEMENTARITY:")
        summary.append("-" * 30)
        for _, row in pairs_df.iterrows():
            summary.append(f"{row['Modality_Pair']:>12}: Spearman={row['Mean_Spearman']:.3f}, "
                          f"JSD={row['Mean_JSD']:.3f}")
        
        summary.append("")
        
        # Hubness
        summary.append("HUBNESS ANALYSIS:")
        summary.append("-" * 20)
        summary.append(f"Gini coefficient: {hubness_summary['gini_coefficient']:.3f}")
        summary.append(f"Top-20 dominance: {hubness_summary['top_20_dominance_ratio']:.3f}")
        
        summary.append("")
        
        # Method comparison
        summary.append("FUSION METHOD PERFORMANCE:")
        summary.append("-" * 30)
        for _, row in methods_df.iterrows():
            summary.append(f"{row['Method']:>15}: Event-Hit@1={row['Event-Hit@1(pair)']:.3f} "
                          f"[{row['Event-Hit_CI_low']:.3f}-{row['Event-Hit_CI_high']:.3f}], "
                          f"p_event_mass={row['Mean_p_event_mass']:.3f}")
        
        summary.append("")
        summary.append("DECISION CRITERIA:")
        summary.append("-" * 20)
        
        # Apply decision rules
        recommendations = []
        
        # Check for Sinkhorn conditions
        max_jsd = pairs_df['Mean_JSD'].max()
        min_spearman = pairs_df['Mean_Spearman'].min()
        
        dsl_performance = methods_df[methods_df['Method'] == 'DSL']['Event-Hit@1(pair)'].iloc[0]
        sinkhorn_performance = methods_df[methods_df['Method'] == 'Sinkhorn']['Event-Hit@1(pair)'].iloc[0]
        best_single = single_df['Event-Hit@1(pair)'].max()
        
        gated_performance = methods_df[methods_df['Method'] == 'Entropy-Gated']['Event-Hit@1(pair)'].iloc[0]
        
        use_sinkhorn = False
        use_dbnorm = False
        use_gated = False
        
        # Sinkhorn criteria
        if max_jsd > 0.2 or min_spearman < 0.5:
            use_sinkhorn = True
            recommendations.append("✓ High modality divergence detected → Sinkhorn recommended")
        
        if (dsl_performance - best_single) < 0.01 and (sinkhorn_performance - dsl_performance) > 0.005:
            use_sinkhorn = True
            recommendations.append("✓ DSL shows minimal improvement, Sinkhorn shows stable gains → Sinkhorn recommended")
        
        # DBNorm criteria
        if hubness_summary['gini_coefficient'] >= 0.30:
            use_dbnorm = True
            recommendations.append("✓ High hubness detected → DBNorm preprocessing recommended")
        
        # Gated selection criteria
        if abs(gated_performance - max(methods_df['Event-Hit@1(pair)'])) <= 0.02:
            use_gated = True
            recommendations.append("✓ Entropy-gated selection performs competitively → Simple gating viable")
        
        # DSL criteria (fallback)
        if not use_sinkhorn and max_jsd <= 0.1 and min_spearman >= 0.6:
            recommendations.append("✓ Low modality divergence, high correlation → DSL (baseline) sufficient")
        
        # Final recommendation
        summary.append("")
        if recommendations:
            for rec in recommendations:
                summary.append(rec)
        else:
            summary.append("○ No clear winner - consider DSL as baseline")
        
        summary.append("")
        
        # Final recommendation
        final_method = "DSL"  # Default
        
        if use_gated and not use_sinkhorn:
            final_method = "Entropy-Gated"
        elif use_sinkhorn and use_dbnorm:
            final_method = "DBNorm+Sinkhorn"
        elif use_sinkhorn:
            final_method = "Sinkhorn"
        elif use_dbnorm:
            final_method = "DBNorm+DSL"
        
        summary.append("FINAL RECOMMENDATION:")
        summary.append("=" * 25)
        summary.append(f"🎯 {final_method}")
        summary.append("")
        
        performance = methods_df[methods_df['Method'] == final_method]['Event-Hit@1(pair)'].iloc[0]
        summary.append(f"Expected Event-Hit@1(pair) performance: {performance:.3f}")
        
        summary.append("")
        summary.append("="*60)
        
        # Save summary
        summary_text = "\n".join(summary)
        summary_file = self.output_dir / "decision_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        
        logger.info(f"Saved decision summary to {summary_file}")
        logger.info(f"RECOMMENDED METHOD: {final_method}")
        
        return summary_text
    
    def run_full_analysis(self):
        """Run complete multimodal choice analysis"""
        logger.info("Starting multimodal choice analysis...")
        start_time = datetime.now()
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Prepare analysis matrices
        self.prepare_analysis_data()
        
        # Step 3: Single modality diagnostics
        single_df = self.compute_single_modality_diagnostics()
        
        # Step 4: Modality pair analysis
        pairs_df = self.compute_modality_pair_analysis()
        
        # Step 5: Hubness analysis
        hubness_df, hubness_summary = self.compute_hubness_analysis()
        
        # Step 6: Fusion methods comparison
        methods_df = self.compute_fusion_methods_comparison(hubness_summary)
        
        # Step 7: Create visualizations
        self.create_visualizations(single_df, pairs_df, hubness_df, methods_df, hubness_summary)
        
        # Step 8: Generate decision summary
        decision_summary = self.generate_decision_summary(single_df, pairs_df, hubness_summary, methods_df)
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"Total duration: {duration:.1f} seconds")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Files generated:")
        for file_path in self.output_dir.glob("*"):
            logger.info(f"  - {file_path.name}")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Multimodal Choice Analysis for Fusion Method Selection')
    parser.add_argument('--dataset', type=str, default='FakeSV',
                       help='Dataset name (default: FakeSV)')
    parser.add_argument('--audio-model', type=str, 
                       default='CAiRE-SER-wav2vec2-large-xlsr-53-eng-zho-all-age',
                       help='Audio model name suffix for loading features')
    parser.add_argument('--text-model', type=str, default=None,
                       help='Text model for embeddings (auto-selected if None)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: analysis/{dataset}/multimodal_choice)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MultimodalChoiceAnalyzer(
        dataset=args.dataset,
        audio_model=args.audio_model,
        text_model=args.text_model,
        output_dir=args.output_dir
    )
    
    # Run analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()