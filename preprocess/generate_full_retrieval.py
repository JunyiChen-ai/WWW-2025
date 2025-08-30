#!/usr/bin/env python3
"""
Generate full dataset retrieval results for gating mechanism.
All samples retrieve from train+valid memory bank, with self-exclusion for train/valid samples.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Set
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import logging
import sys
from datetime import datetime
import time

# Add src to path
sys.path.append('src')
from data.baseline_data import FakeSVDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FullDatasetRetrieval:
    def __init__(self, model_name: str = "OFA-Sys/chinese-clip-vit-large-patch14", filter_k: int = None, use_pool: bool = False):
        """
        Initialize full dataset retrieval system
        
        Args:
            model_name: Hugging Face model name for text embedding
            filter_k: Number of most similar training samples to remove per test sample (None for no filtering)
            use_pool: Use pooled features instead of frame-level similarity calculation
        """
        self.model_name = model_name
        self.model_short_name = model_name.split("/")[-1]
        self.filter_k = filter_k
        self.use_pool = use_pool
        
        # Data paths
        self.entity_dir = Path("data/FakeSV/entity_claims")
        self.output_dir = Path("text_similarity_results")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initializing full dataset retrieval with model: {model_name}")
        if filter_k is not None:
            logger.info(f"Will filter top-{filter_k} similar training samples per test sample")
        if use_pool:
            logger.info("Using pooled features for visual/audio similarity (fast mode)")
        else:
            logger.info("Using frame-level similarity for visual/audio (detailed mode)")
        
    def load_model(self):
        """Load the embedding model"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
        
    def encode_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Encode texts using the loaded model
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings
        """
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding with {self.model_short_name}"):
                batch_texts = texts[i:i+batch_size]
                
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
                    # For Chinese-CLIP, use text_model
                    outputs = self.model.text_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                else:
                    # For other models
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        logger.info(f"Encoded {len(texts)} texts into shape {embeddings.shape}")
        return embeddings
    
    def load_entity_data(self):
        """Load entity claims data"""
        # Load true video extractions
        true_file = self.entity_dir / "video_extractions.jsonl"
        self.true_data = []
        with open(true_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.true_data.append(json.loads(line))
        logger.info(f"Loaded {len(self.true_data)} true video extractions")
        
        # Load fake video descriptions
        fake_file = self.entity_dir / "fake_video_descriptions.jsonl"
        self.fake_data = []
        with open(fake_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.fake_data.append(json.loads(line))
        logger.info(f"Loaded {len(self.fake_data)} fake video descriptions")
        
        # Load original data for timestamps
        self.load_original_timestamps()
    
    def load_original_timestamps(self):
        """Load original data to get publish timestamps"""
        from datetime import datetime
        
        # Load original data
        orig_file = Path("data/FakeSV/data_complete.jsonl")
        self.timestamp_lookup = {}
        
        with open(orig_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                video_id = item.get('video_id')
                publish_time = item.get('publish_time_norm')
                
                if video_id and publish_time:
                    # Convert timestamp to readable format
                    if len(str(publish_time)) > 10:  # Milliseconds
                        timestamp = datetime.fromtimestamp(publish_time / 1000.0)
                    else:  # Seconds
                        timestamp = datetime.fromtimestamp(publish_time)
                    
                    self.timestamp_lookup[video_id] = {
                        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'raw_timestamp': publish_time
                    }
        
        logger.info(f"Loaded timestamps for {len(self.timestamp_lookup)} videos")
    
    def load_multimodal_features(self):
        """Load visual and audio features with optimization"""
        feature_dir = Path("data/FakeSV/fea")
        
        # Load visual features
        vit_file = feature_dir / "vit_tensor.pt"
        if vit_file.exists():
            self.visual_features = torch.load(vit_file, map_location='cpu')
            logger.info(f"Loaded visual features for {len(self.visual_features)} videos")
        else:
            self.visual_features = {}
            logger.warning(f"Visual features file not found: {vit_file}")
        
        # Load audio features  
        audio_file = feature_dir / "audio_features_frames.pt"
        if audio_file.exists():
            self.audio_features = torch.load(audio_file, map_location='cpu')
            logger.info(f"Loaded audio features for {len(self.audio_features)} videos")
        else:
            self.audio_features = {}
            logger.warning(f"Audio features file not found: {audio_file}")
        
        # Convert to numpy and ensure float32 for efficiency
        logger.info("Converting and optimizing features...")
        for video_id in self.visual_features:
            features = self.visual_features[video_id].numpy().astype(np.float32)
            # Ensure contiguous memory layout for faster access
            self.visual_features[video_id] = np.ascontiguousarray(features)
        
        for video_id in self.audio_features:
            features = self.audio_features[video_id].numpy().astype(np.float32)
            # Ensure contiguous memory layout for faster access
            self.audio_features[video_id] = np.ascontiguousarray(features)
        
        # Initialize caches for normalized features
        self.visual_features_normalized = {}
        self.audio_features_normalized = {}
        
        logger.info("Multimodal features loaded, converted and optimized")
    
    def get_normalized_features(self, video_id: str, modality: str) -> np.ndarray:
        """Get normalized features with caching"""
        if modality == 'V':
            if video_id not in self.visual_features_normalized:
                if video_id in self.visual_features:
                    normalized = self.l2_normalize(self.visual_features[video_id], axis=-1)
                    self.visual_features_normalized[video_id] = normalized
                else:
                    return None
            return self.visual_features_normalized[video_id]
        
        elif modality == 'A':
            if video_id not in self.audio_features_normalized:
                if video_id in self.audio_features:
                    normalized = self.l2_normalize(self.audio_features[video_id], axis=-1)
                    self.audio_features_normalized[video_id] = normalized
                else:
                    return None
            return self.audio_features_normalized[video_id]
        
        return None
    
    def precompute_bank_features(self, memory_true_data: List[Dict], memory_fake_data: List[Dict], 
                                true_embeddings: np.ndarray, fake_embeddings: np.ndarray):
        """Pre-compute and cache normalized memory bank features"""
        logger.info("Pre-computing normalized memory bank features...")
        
        # Pre-compute true bank features
        self.true_bank_features = {'T': [], 'V': [], 'A': []}
        self.true_available_indices = []
        
        for i, memory_item in enumerate(memory_true_data):
            video_id = memory_item['video_id']
            visual_features = self.get_normalized_features(video_id, 'V')
            audio_features = self.get_normalized_features(video_id, 'A')
            
            if visual_features is not None and audio_features is not None:
                self.true_bank_features['V'].append(visual_features)
                self.true_bank_features['A'].append(audio_features)
                self.true_available_indices.append(i)
        
        if len(self.true_bank_features['V']) > 0:
            # Stack into arrays (already normalized from cache)
            self.true_bank_features['T'] = true_embeddings[self.true_available_indices]
            self.true_bank_features['V'] = np.stack(self.true_bank_features['V'], axis=0)
            self.true_bank_features['A'] = np.stack(self.true_bank_features['A'], axis=0)  
            
            logger.info(f"Pre-computed true bank features: {self.true_bank_features['V'].shape}")
        
        # Pre-compute fake bank features
        self.fake_bank_features = {'T': [], 'V': [], 'A': []}
        self.fake_available_indices = []
        
        for i, memory_item in enumerate(memory_fake_data):
            video_id = memory_item['video_id']
            visual_features = self.get_normalized_features(video_id, 'V')
            audio_features = self.get_normalized_features(video_id, 'A')
            
            if visual_features is not None and audio_features is not None:
                self.fake_bank_features['V'].append(visual_features)
                self.fake_bank_features['A'].append(audio_features)
                self.fake_available_indices.append(i)
        
        if len(self.fake_bank_features['V']) > 0:
            # Stack into arrays (already normalized from cache)
            self.fake_bank_features['T'] = fake_embeddings[self.fake_available_indices]
            self.fake_bank_features['V'] = np.stack(self.fake_bank_features['V'], axis=0)
            self.fake_bank_features['A'] = np.stack(self.fake_bank_features['A'], axis=0)
            
            logger.info(f"Pre-computed fake bank features: {self.fake_bank_features['V'].shape}")
        
        logger.info("Bank features pre-computation complete")
        
    def get_temporal_splits(self):
        """Get temporal split video IDs"""
        vid_dir = Path("data/FakeSV/vids")
        
        splits = {}
        for split in ['train', 'valid', 'test']:
            split_file = vid_dir / f"vid_time3_{split}.txt"
            with open(split_file, 'r') as f:
                splits[split] = set(line.strip() for line in f)
            logger.info(f"{split.capitalize()} split: {len(splits[split])} videos")
        
        return splits
    
    def l2_normalize(self, x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
        """L2 normalization for features"""
        n = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (n + eps)
    
    def compute_frame_similarity(self, query_frames: np.ndarray, candidate_frames: np.ndarray) -> float:
        """
        Compute frame-level similarity between query and candidate videos
        
        Args:
            query_frames: [T_q, D] frames for query video
            candidate_frames: [T_c, D] frames for candidate video
            
        Returns:
            Scalar similarity score
        """
        # L2 normalize frames
        query_frames = self.l2_normalize(query_frames)
        candidate_frames = self.l2_normalize(candidate_frames)
        
        # Compute similarity matrix [T_q, T_c]
        similarity_matrix = np.dot(query_frames, candidate_frames.T)
        
        # For each query frame, find max similarity across candidate frames
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # Average across all query frames
        return float(np.mean(max_similarities))
    
    def compute_batch_frame_similarity(self, query_frames: np.ndarray, candidate_frames_batch: np.ndarray) -> np.ndarray:
        """
        Vectorized batch computation of frame-level similarity
        
        Args:
            query_frames: [T_q, D] frames for query video (already normalized)
            candidate_frames_batch: [N, T_c, D] frames for N candidate videos (already normalized)
            
        Returns:
            Array of N similarity scores
        """
        # Handle edge case: empty candidate batch
        if candidate_frames_batch.shape[0] == 0:
            return np.array([], dtype=np.float32)
        
        # Ensure all arrays are float32 for better performance
        query_frames = query_frames.astype(np.float32)
        candidate_frames_batch = candidate_frames_batch.astype(np.float32)
        
        N, T_c, D = candidate_frames_batch.shape
        T_q = query_frames.shape[0]
        
        # Handle edge case: query has no frames
        if T_q == 0:
            return np.zeros(N, dtype=np.float32)
        
        # Reshape for batch matrix multiplication
        # query_frames: [T_q, D] -> [1, T_q, D] -> [N, T_q, D]
        query_batch = np.broadcast_to(query_frames[None, :, :], (N, T_q, D))
        
        # Batch matrix multiplication: [N, T_q, D] @ [N, D, T_c] -> [N, T_q, T_c]
        similarity_matrices = np.einsum('nqd,ndc->nqc', query_batch, candidate_frames_batch.transpose(0, 2, 1))
        
        # For each candidate, find max similarity for each query frame, then average
        max_similarities = np.max(similarity_matrices, axis=2)  # [N, T_q]
        similarity_scores = np.mean(max_similarities, axis=1)   # [N]
        
        return similarity_scores.astype(np.float32)
    
    def fused_scores_sample_gating(self, q: Dict[str, np.ndarray], bank: Dict[str, np.ndarray], 
                                  eps: float = 1e-12) -> np.ndarray:
        """
        Multimodal fusion using sample-adaptive gating (optimized version)
        
        Args:
            q: Query features {'T': text_emb, 'V': visual_frames, 'A': audio_frames}
            bank: Bank features {'T': text_bank, 'V': visual_bank, 'A': audio_bank}
            
        Returns:
            Fused scores for all samples in bank
        """
        # Step 1: L2 normalize features (assume they come from cache and are already normalized)
        q_norm = {}
        bank_norm = {}
        
        for m in q:
            # For query features, normalize text but visual/audio should already be normalized from cache
            if m == 'T':
                q_norm[m] = self.l2_normalize(q[m])
            else:
                q_norm[m] = q[m]  # Already normalized from cache
            
            # For bank features, they should already be normalized from precompute_bank_features
            if m == 'T':
                bank_norm[m] = self.l2_normalize(bank[m], axis=1)
            else:
                bank_norm[m] = bank[m]  # Already normalized from cache
        
        # Step 2: Compute similarities and Z-normalize per modality
        Z = {}
        for m in bank_norm:
            if m == 'T':
                # Text: simple dot product (already normalized)
                s = np.dot(bank_norm[m], q_norm[m])
            else:
                # Visual/Audio: choose between pooled and frame-level similarity
                if self.use_pool:
                    # Pooled mode: average pool features, re-normalize, then compute cosine similarity
                    q_pooled = np.mean(q_norm[m], axis=0)  # [D]
                    q_pooled = q_pooled / (np.linalg.norm(q_pooled) + eps)  # Re-normalize after pooling
                    
                    bank_pooled = np.mean(bank_norm[m], axis=1)  # [N, D]
                    bank_pooled = bank_pooled / (np.linalg.norm(bank_pooled, axis=1, keepdims=True) + eps)  # Re-normalize
                    
                    s = np.dot(bank_pooled, q_pooled)  # Now it's cosine similarity
                else:
                    # Frame-level mode: vectorized frame-level similarity
                    s = self.compute_batch_frame_similarity(q_norm[m], bank_norm[m])
            
            # Z-standardization across samples (handle case where std is 0)
            std_val = s.std()
            if std_val < eps:
                # If all similarities are the same, just center them
                Z[m] = s - s.mean()
            else:
                Z[m] = (s - s.mean()) / (std_val + eps)
        
        # Step 3: Sample-adaptive gating (vectorized)
        pos = [np.maximum(Z[m], 0.0) for m in Z]  # [·]₊
        
        # Vectorized computation
        pos_array = np.stack(pos, axis=0)  # [num_modalities, num_samples]
        num = np.sum(pos_array * pos_array, axis=0)  # Element-wise square and sum
        den = np.sum(pos_array, axis=0)  # Sum across modalities
            
        S = num / (den + eps)  # Final fused scores
        return S
        
    def prepare_data(self, splits: Dict[str, Set[str]], filtered_train_ids: Set[str] = None):
        """Prepare data for retrieval"""
        # Create lookup for entity data
        entity_lookup = {}
        for item in self.true_data + self.fake_data:
            entity_lookup[item['video_id']] = item
        
        # Use filtered train IDs if provided, otherwise use original
        if filtered_train_ids is None:
            filtered_train_ids = splits['train']
        
        # Memory bank consists of (filtered) train + valid samples
        memory_bank_video_ids = filtered_train_ids | splits['valid']
        logger.info(f"Memory bank size: {len(memory_bank_video_ids)} videos "
                   f"(filtered train: {len(filtered_train_ids)}, valid: {len(splits['valid'])})")
        
        # Prepare memory bank data (train + valid)
        self.memory_true_data = []
        self.memory_fake_data = []
        
        for video_id in memory_bank_video_ids:
            if video_id in entity_lookup:
                item = entity_lookup[video_id]
                if item.get('annotation') == '真':
                    self.memory_true_data.append(item)
                elif item.get('annotation') == '假':
                    self.memory_fake_data.append(item)
        
        # Prepare all query samples (filtered train + valid + test)
        all_video_ids = filtered_train_ids | splits['valid'] | splits['test']
        self.query_data = []
        self.query_splits = {}  # Track which split each sample belongs to
        
        for video_id in all_video_ids:
            if video_id in entity_lookup:
                item = entity_lookup[video_id]
                self.query_data.append(item)
                
                # Determine which split this video belongs to
                if video_id in filtered_train_ids:
                    self.query_splits[video_id] = 'train'
                elif video_id in splits['valid']:
                    self.query_splits[video_id] = 'valid'
                elif video_id in splits['test']:
                    self.query_splits[video_id] = 'test'
                else:
                    self.query_splits[video_id] = 'unknown'
        
        logger.info(f"Memory bank: {len(self.memory_true_data)} true, {len(self.memory_fake_data)} fake")
        logger.info(f"Query set: {len(self.query_data)} videos")
        
    def create_text_representation(self, item: Dict) -> str:
        """Create text representation for embedding"""
        # Use title, keywords, description, and temporal evolution
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
    
    def filter_training_set(self, splits: Dict[str, Set[str]]) -> Set[str]:
        """
        Filter training set by removing top-k most similar samples to test set
        
        Args:
            splits: Dictionary containing train/valid/test splits
            
        Returns:
            Set of filtered training video IDs
        """
        if self.filter_k is None:
            return splits['train']
        
        logger.info(f"Filtering training set: removing top-{self.filter_k} similar samples per test sample...")
        
        # Create entity lookup for all samples
        entity_lookup = {}
        for item in self.true_data + self.fake_data:
            entity_lookup[item['video_id']] = item
        
        # Only work with video IDs that exist in entity data to ensure consistency
        available_video_ids = set(entity_lookup.keys())
        filtered_train_ids = splits['train'] & available_video_ids
        filtered_test_ids = splits['test'] & available_video_ids
        
        logger.info(f"Available train IDs: {len(filtered_train_ids)} (original: {len(splits['train'])})")
        logger.info(f"Available test IDs: {len(filtered_test_ids)} (original: {len(splits['test'])})")
        
        # Prepare training data for temporary memory bank (only available IDs)
        train_data = []
        for video_id in filtered_train_ids:
            train_data.append(entity_lookup[video_id])
        
        # Prepare test data (only available IDs)
        test_data = []
        for video_id in filtered_test_ids:
            test_data.append(entity_lookup[video_id])
        
        logger.info(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        # Create text representations and encode
        train_texts = [self.create_text_representation(item) for item in train_data]
        test_texts = [self.create_text_representation(item) for item in test_data]
        
        logger.info("Encoding training samples for filtering...")
        train_embeddings = self.encode_texts(train_texts)
        
        logger.info("Encoding test samples for filtering...")
        test_embeddings = self.encode_texts(test_texts)
        
        # Compute similarities between test and training samples
        logger.info("Computing test-train similarities for filtering...")
        similarities = cosine_similarity(test_embeddings, train_embeddings)
        
        # Find top-k most similar training samples for each test sample
        training_samples_to_remove = set()
        for i, test_item in enumerate(test_data):
            test_similarities = similarities[i]
            
            # Get top-k most similar training sample indices
            top_k_indices = np.argsort(test_similarities)[-self.filter_k:]
            
            for idx in top_k_indices:
                similar_train_id = train_data[idx]['video_id']
                training_samples_to_remove.add(similar_train_id)
                
                logger.debug(f"Test {test_item['video_id']} -> Remove train {similar_train_id} "
                           f"(similarity: {test_similarities[idx]:.4f})")
        
        # Create filtered training set from the available train IDs
        final_filtered_train_ids = filtered_train_ids - training_samples_to_remove
        
        logger.info(f"Filtering complete: removed {len(training_samples_to_remove)} training samples")
        logger.info(f"Filtered training set: {len(final_filtered_train_ids)} samples "
                   f"(original: {len(splits['train'])})")
        
        # Save filtered training IDs to file
        vid_dir = Path("data/FakeSV/vids")
        filtered_file = vid_dir / f"vid_time3_train_k{self.filter_k}.txt"
        with open(filtered_file, 'w') as f:
            for vid_id in sorted(final_filtered_train_ids):
                f.write(f"{vid_id}\n")
        
        logger.info(f"Saved filtered training IDs to {filtered_file}")
        
        return final_filtered_train_ids
    
    def find_similar_videos(self, query_embeddings: np.ndarray, 
                           true_embeddings: np.ndarray, fake_embeddings: np.ndarray,
                           query_data: List[Dict], 
                           memory_true_data: List[Dict], memory_fake_data: List[Dict],
                           top_k: int = 1, batch_size: int = 64) -> List[Dict]:
        """Find most similar videos using optimized multimodal fusion"""
        logger.info(f"Computing multimodal similarities for {len(query_data)} query videos (batch_size={batch_size})...")
        
        # Pre-compute and cache normalized memory bank features
        self.precompute_bank_features(memory_true_data, memory_fake_data, true_embeddings, fake_embeddings)
        
        results = []
        
        # Process queries in batches for better performance  
        for batch_start in tqdm(range(0, len(query_data), batch_size), desc="Processing query batches"):
            batch_end = min(batch_start + batch_size, len(query_data))
            batch_queries = query_data[batch_start:batch_end]
            batch_embeddings = query_embeddings[batch_start:batch_end]
            
            batch_results = self._process_query_batch(
                batch_queries, batch_embeddings, memory_true_data, memory_fake_data
            )
            results.extend(batch_results)
            
        return results
    
    def _process_query_batch(self, batch_queries: List[Dict], batch_embeddings: np.ndarray,
                            memory_true_data: List[Dict], memory_fake_data: List[Dict]) -> List[Dict]:
        """Process a batch of queries efficiently"""
        batch_results = []
        
        # Process each query in the batch
        for i, query_item in enumerate(batch_queries):
            query_video_id = query_item['video_id']
            query_split = self.query_splits.get(query_video_id, 'unknown')
            
            # Skip if multimodal features not available for query
            visual_features = self.get_normalized_features(query_video_id, 'V')
            audio_features = self.get_normalized_features(query_video_id, 'A')
            
            if visual_features is None or audio_features is None:
                logger.warning(f"Missing multimodal features for query {query_video_id}, skipping")
                continue
                
            # Prepare query features (already normalized from cache)
            query_features = {
                'T': batch_embeddings[i],  # Text embedding from batch
                'V': visual_features,       # Normalized visual features
                'A': audio_features        # Normalized audio features  
            }
            
            # Process true and fake candidates in parallel using optimized method
            true_candidate, fake_candidate = self._find_best_candidates_parallel(
                query_features, query_split, query_video_id, memory_true_data, memory_fake_data
            )
            
            result = {
                'query_video': {
                    'video_id': query_video_id,
                    'annotation': query_item.get('annotation', ''),
                    'split': query_split,
                    'title': query_item.get('title', ''),
                    'keywords': query_item.get('keywords', ''),
                    'description': query_item.get('description', ''),
                    'temporal_evolution': query_item.get('temporal_evolution', ''),
                    'entity_claims': query_item.get('entity_claims', {}),
                    'publish_time': self.timestamp_lookup.get(query_video_id, {}).get('timestamp', 'Unknown')
                },
                'similar_true': true_candidate,
                'similar_fake': fake_candidate,
                'excluded_self': query_split in ['train', 'valid']
            }
            
            batch_results.append(result)
        
        return batch_results
    
    def _find_best_candidates_parallel(self, query_features: Dict[str, np.ndarray], query_split: str, 
                                     query_video_id: str, memory_true_data: List[Dict], memory_fake_data: List[Dict]):
        """Find best true and fake candidates using parallel processing"""
        import concurrent.futures
        
        # Use ThreadPoolExecutor for I/O-bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both true and fake candidate searches in parallel
            future_true = executor.submit(
                self._find_best_candidate, query_features, query_split, query_video_id,
                self.true_bank_features, self.true_available_indices, memory_true_data, 'true'
            )
            future_fake = executor.submit(
                self._find_best_candidate, query_features, query_split, query_video_id,
                self.fake_bank_features, self.fake_available_indices, memory_fake_data, 'fake'  
            )
            
            # Get results
            true_candidate = future_true.result()
            fake_candidate = future_fake.result()
            
        return true_candidate, fake_candidate
    
    def _find_best_candidate(self, query_features: Dict[str, np.ndarray], query_split: str,
                           query_video_id: str, bank_features: Dict[str, np.ndarray], available_indices: List[int],
                           memory_data: List[Dict], candidate_type: str):
        """Find best candidate from a specific bank (true or fake)"""
        if 'V' not in bank_features or bank_features['V'] is None or len(bank_features['V']) == 0:
            return None
        
        # Handle self-exclusion for train/valid
        exclude_indices = set()
        if query_split in ['train', 'valid'] and query_video_id:
            for j, idx in enumerate(available_indices):
                if memory_data[idx]['video_id'] == query_video_id:
                    exclude_indices.add(j)
        
        # Create filtered bank features
        if exclude_indices:
            keep_indices = [j for j in range(len(available_indices)) if j not in exclude_indices]
            if not keep_indices:
                return None
            filtered_bank = {
                'T': bank_features['T'][keep_indices],
                'V': bank_features['V'][keep_indices],
                'A': bank_features['A'][keep_indices]
            }
            filtered_indices = [available_indices[j] for j in keep_indices]
        else:
            filtered_bank = bank_features
            filtered_indices = available_indices
            
        # Compute multimodal similarities
        fused_scores = self.fused_scores_sample_gating(query_features, filtered_bank)
        best_idx = np.argmax(fused_scores)
        memory_idx = filtered_indices[best_idx]
        
        # Compute individual modality similarities for output
        text_sim = float(np.dot(self.l2_normalize(filtered_bank['T'][best_idx]), 
                               self.l2_normalize(query_features['T'])))
        visual_sim = self.compute_frame_similarity(query_features['V'], filtered_bank['V'][best_idx])
        audio_sim = self.compute_frame_similarity(query_features['A'], filtered_bank['A'][best_idx])
        
        best_item = memory_data[memory_idx]
        candidate = {
            'video_id': best_item['video_id'],
            'similarity_score': float(fused_scores[best_idx]),  # Fused score
            'text_similarity': text_sim,
            'visual_similarity': visual_sim,
            'audio_similarity': audio_sim,
            'annotation': best_item['annotation'],
            'title': best_item.get('title', ''),
            'keywords': best_item.get('keywords', ''),
            'description': best_item.get('description', ''),
            'temporal_evolution': best_item.get('temporal_evolution', ''),
            'entity_claims': best_item.get('entity_claims', {}),
            'publish_time': self.timestamp_lookup.get(best_item['video_id'], {}).get('timestamp', 'Unknown')
        }
        
        return candidate
    
    def run_retrieval(self):
        """Run the full retrieval process with performance monitoring"""
        start_time = time.time()
        logger.info("Starting optimized full dataset retrieval...")
        
        # Load data
        load_start = time.time()
        self.load_entity_data()
        splits = self.get_temporal_splits()
        logger.info(f"Data loading took {time.time() - load_start:.2f} seconds")
        
        # Load multimodal features
        features_start = time.time()
        self.load_multimodal_features()
        logger.info(f"Multimodal features loading took {time.time() - features_start:.2f} seconds")
        
        # Load model first (required for filtering)
        model_start = time.time()
        self.load_model()
        logger.info(f"Model loading took {time.time() - model_start:.2f} seconds")
        
        # Apply filtering to training set if specified (needs model loaded)
        filter_start = time.time()
        filtered_train_ids = self.filter_training_set(splits)
        logger.info(f"Training set filtering took {time.time() - filter_start:.2f} seconds")
        
        # Prepare data with filtered training set
        prepare_start = time.time()
        self.prepare_data(splits, filtered_train_ids)
        logger.info(f"Data preparation took {time.time() - prepare_start:.2f} seconds")
        
        # Prepare texts for encoding
        text_prep_start = time.time()
        logger.info("Preparing texts for encoding...")
        
        # Separate memory bank texts
        true_texts = [self.create_text_representation(item) for item in self.memory_true_data]
        fake_texts = [self.create_text_representation(item) for item in self.memory_fake_data]
        
        # Query texts (all data)
        query_texts = [self.create_text_representation(item) for item in self.query_data]
        logger.info(f"Text preparation took {time.time() - text_prep_start:.2f} seconds")
        
        # Encode texts separately
        encoding_start = time.time()
        logger.info("Encoding queries...")
        query_embeddings = self.encode_texts(query_texts)
        
        logger.info("Encoding true memory bank...")
        true_embeddings = self.encode_texts(true_texts) if true_texts else np.array([])
        
        logger.info("Encoding fake memory bank...")
        fake_embeddings = self.encode_texts(fake_texts) if fake_texts else np.array([])
        logger.info(f"Text encoding took {time.time() - encoding_start:.2f} seconds")
        
        # Find similar videos with optimized batch processing
        retrieval_start = time.time()
        results = self.find_similar_videos(
            query_embeddings, true_embeddings, fake_embeddings,
            self.query_data, 
            self.memory_true_data, self.memory_fake_data,
            batch_size=64  # Use larger batch size for better performance
        )
        logger.info(f"Similarity computation took {time.time() - retrieval_start:.2f} seconds")
        
        # Build output filename with appropriate suffixes
        filename_parts = [f"full_dataset_retrieval_{self.model_short_name}"]
        if self.filter_k is not None:
            filename_parts.append(f"k{self.filter_k}")
        if self.use_pool:
            filename_parts.append("pool")
        
        output_file = self.output_dir / f"{'_'.join(filename_parts)}.json"
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved retrieval results to {output_file}")
        
        # Generate summary statistics
        summary_start = time.time()
        self.generate_summary(results, output_file)
        logger.info(f"Summary generation took {time.time() - summary_start:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Processed {len(results)} queries at {len(results)/total_time:.2f} queries/sec")
        logger.info("="*60)
        
        return results
    
    def generate_summary(self, results: List[Dict], output_file: Path):
        """Generate summary statistics"""
        stats = {
            'total_queries': len(results),
            'memory_bank_size': len(self.memory_true_data) + len(self.memory_fake_data),
            'memory_true_count': len(self.memory_true_data),
            'memory_fake_count': len(self.memory_fake_data),
            'excluded_self_count': sum(1 for r in results if r['excluded_self']),
            'splits': {}
        }
        
        # Count by splits
        for result in results:
            split = result['query_video']['split']
            if split not in stats['splits']:
                stats['splits'][split] = 0
            stats['splits'][split] += 1
        
        # Calculate average similarities
        true_similarities = []
        fake_similarities = []
        test_to_memory_true_similarities = []
        test_to_memory_fake_similarities = []
        test_to_memory_train_true_similarities = []
        test_to_memory_train_fake_similarities = []
        
        # Create lookup for memory bank splits (train vs valid)
        memory_bank_splits = {}
        for item in self.memory_true_data + self.memory_fake_data:
            video_id = item['video_id']
            # Determine if this memory bank item is from train or valid
            if hasattr(self, 'query_splits') and video_id in self.query_splits:
                memory_bank_splits[video_id] = self.query_splits[video_id]
        
        for result in results:
            if result['similar_true']:
                true_similarities.append(result['similar_true']['similarity_score'])
                # Track test set to memory bank similarities separately
                if result['query_video']['split'] == 'test':
                    test_to_memory_true_similarities.append(result['similar_true']['similarity_score'])
                    # Check if the similar true sample is from train split in memory bank
                    similar_video_id = result['similar_true']['video_id']
                    if similar_video_id in memory_bank_splits and memory_bank_splits[similar_video_id] == 'train':
                        test_to_memory_train_true_similarities.append(result['similar_true']['similarity_score'])
                    
            if result['similar_fake']:
                fake_similarities.append(result['similar_fake']['similarity_score'])
                # Track test set to memory bank similarities separately  
                if result['query_video']['split'] == 'test':
                    test_to_memory_fake_similarities.append(result['similar_fake']['similarity_score'])
                    # Check if the similar fake sample is from train split in memory bank
                    similar_video_id = result['similar_fake']['video_id']
                    if similar_video_id in memory_bank_splits and memory_bank_splits[similar_video_id] == 'train':
                        test_to_memory_train_fake_similarities.append(result['similar_fake']['similarity_score'])
        
        stats['avg_true_similarity'] = float(np.mean(true_similarities)) if true_similarities else 0.0
        stats['avg_fake_similarity'] = float(np.mean(fake_similarities)) if fake_similarities else 0.0
        stats['test_to_memory_avg_true_similarity'] = float(np.mean(test_to_memory_true_similarities)) if test_to_memory_true_similarities else 0.0
        stats['test_to_memory_avg_fake_similarity'] = float(np.mean(test_to_memory_fake_similarities)) if test_to_memory_fake_similarities else 0.0
        stats['test_to_memory_train_avg_true_similarity'] = float(np.mean(test_to_memory_train_true_similarities)) if test_to_memory_train_true_similarities else 0.0
        stats['test_to_memory_train_avg_fake_similarity'] = float(np.mean(test_to_memory_train_fake_similarities)) if test_to_memory_train_fake_similarities else 0.0
        
        # Build stats filename with same suffixes
        stats_parts = [f"full_dataset_stats_{self.model_short_name}"]
        if self.filter_k is not None:
            stats_parts.append(f"k{self.filter_k}")
        if self.use_pool:
            stats_parts.append("pool")
        
        stats_file = output_file.parent / f"{'_'.join(stats_parts)}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # Print summary
        logger.info("="*60)
        logger.info("FULL DATASET RETRIEVAL SUMMARY")
        logger.info("="*60)
        logger.info(f"Total queries: {stats['total_queries']}")
        logger.info(f"Memory bank size: {stats['memory_bank_size']} (True: {stats['memory_true_count']}, Fake: {stats['memory_fake_count']})")
        logger.info(f"Self-exclusions: {stats['excluded_self_count']}")
        logger.info(f"Split distribution: {stats['splits']}")
        logger.info(f"Average similarities - True: {stats['avg_true_similarity']:.4f}, Fake: {stats['avg_fake_similarity']:.4f}")
        logger.info(f"Test-to-Memory average similarities - True: {stats['test_to_memory_avg_true_similarity']:.4f}, Fake: {stats['test_to_memory_avg_fake_similarity']:.4f}")
        logger.info(f"Test-to-Memory-Train average similarities - True: {stats['test_to_memory_train_avg_true_similarity']:.4f}, Fake: {stats['test_to_memory_train_avg_fake_similarity']:.4f}")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Statistics saved to: {stats_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate full dataset retrieval results')
    parser.add_argument('--model', type=str, 
                       default='OFA-Sys/chinese-clip-vit-large-patch14',
                       help='Hugging Face model name for text embedding')
    parser.add_argument('--filter-k', type=int, default=None,
                       help='Number of most similar training samples to remove per test sample')
    parser.add_argument('--use-pool', action='store_true',
                       help='Use pooled features instead of frame-level similarity (faster)')
    
    args = parser.parse_args()
    
    retriever = FullDatasetRetrieval(model_name=args.model, filter_k=args.filter_k, use_pool=args.use_pool)
    retriever.run_retrieval()