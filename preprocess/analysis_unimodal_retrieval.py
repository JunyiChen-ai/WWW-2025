#!/usr/bin/env python3
"""
Analyze unimodal retrieval performance for different modalities and configurations.
Generates separate retrieval results for text, visual, and audio modalities with various settings.
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
import logging
import sys
from datetime import datetime
import time

# Add src to path
sys.path.append('src')
from data.baseline_data import FakeSVDataset, FakeTTDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnimodalRetrievalAnalyzer:
    def __init__(self, dataset: str = "FakeSV", filter_k: int = None, audio_feature_file: str = None):
        """
        Initialize unimodal retrieval analyzer
        
        Args:
            dataset: Dataset name (e.g., 'FakeSV', 'FakeTT')
            filter_k: Number of most similar training samples to remove per test sample (None for no filtering)
            audio_feature_file: Specific audio feature file to use (None for auto-detection)
        """
        self.dataset = dataset
        self.filter_k = filter_k
        self.audio_feature_file = audio_feature_file
        
        # Dataset-specific paths
        self.data_dir = Path(f"data/{dataset}")
        self.entity_dir = self.data_dir
        self.output_dir = Path(f"analysis/{dataset}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get default text model based on dataset
        self.model_name = self.get_default_model(dataset)
        self.model_short_name = self.model_name.split("/")[-1]
        
        logger.info(f"Initializing {dataset} unimodal retrieval analysis")
        if filter_k is not None:
            logger.info(f"Will filter top-{filter_k} similar training samples per test sample")
    
    def get_default_model(self, dataset: str) -> str:
        """Get default text encoding model based on dataset"""
        if dataset.lower() == 'fakesv':
            return 'OFA-Sys/chinese-clip-vit-large-patch14'  # Chinese model
        else:
            return 'zer0int/LongCLIP-GmP-ViT-L-14'  # English model for FakeTT and others
    
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
    
    def encode_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Encode texts using the loaded model
        """
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding texts"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                if 'longclip' in self.model_name.lower():
                    max_len = 248
                elif 'chinese-clip' in self.model_name.lower():
                    max_len = 512  # Chinese-CLIP supports longer sequences
                elif 'clip' in self.model_name.lower():
                    max_len = 77   # Standard CLIP token limit
                else:
                    max_len = 512
                    
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=max_len,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                if 'chinese-clip' in self.model_name.lower():
                    outputs = self.model.text_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0]
                elif 'longclip' in self.model_name.lower():
                    outputs = self.model.text_model(**inputs)
                    last_hidden_state = outputs.last_hidden_state
                    embeddings = last_hidden_state.mean(dim=1)
                elif 'openai/clip' in self.model_name.lower() or 'clip-vit' in self.model_name.lower():
                    outputs = self.model.text_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0]
                
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        return embeddings
    
    def load_entity_data(self):
        """Load entity/video data"""
        llm_file = self.entity_dir / "llm_video_descriptions.jsonl"
        
        if llm_file.exists():
            all_data = []
            with open(llm_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_data.append(json.loads(line))
            logger.info(f"Loaded {len(all_data)} video descriptions")
            
            # Separate by annotation
            self.true_data = [item for item in all_data if item.get('annotation') in ['real', '真']]
            self.fake_data = [item for item in all_data if item.get('annotation') in ['fake', '假']]
            logger.info(f"Separated into {len(self.true_data)} true and {len(self.fake_data)} fake videos")
        else:
            logger.error(f"Data file not found: {llm_file}")
            self.true_data = []
            self.fake_data = []
        
        # Load timestamps
        self.load_original_timestamps()
    
    def load_original_timestamps(self):
        """Load original data to get publish timestamps"""
        from datetime import datetime
        
        if self.dataset == "FakeSV":
            orig_file = self.data_dir / "data_complete.jsonl"
            publish_time_field = 'publish_time_norm'
        else:
            orig_file = self.data_dir / "data.json"
            publish_time_field = 'publish_time'
        
        self.timestamp_lookup = {}
        
        if orig_file.exists():
            with open(orig_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        video_id = item.get('video_id')
                        publish_time = item.get(publish_time_field)
                        
                        if video_id and publish_time:
                            try:
                                if len(str(publish_time)) > 10:
                                    timestamp = datetime.fromtimestamp(publish_time / 1000.0)
                                else:
                                    timestamp = datetime.fromtimestamp(publish_time)
                                
                                self.timestamp_lookup[video_id] = {
                                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                    'raw_timestamp': publish_time
                                }
                            except (ValueError, OSError):
                                self.timestamp_lookup[video_id] = {
                                    'timestamp': 'Unknown',
                                    'raw_timestamp': publish_time
                                }
            
            logger.info(f"Loaded timestamps for {len(self.timestamp_lookup)} videos")
    
    def load_multimodal_features(self):
        """Load visual and audio features"""
        feature_dir = self.data_dir / "fea"
        
        # Load visual features
        vit_file = feature_dir / "vit_tensor.pt"
        if vit_file.exists():
            self.visual_features = torch.load(vit_file, map_location='cpu')
            # Convert to numpy
            for video_id in self.visual_features:
                self.visual_features[video_id] = self.visual_features[video_id].numpy().astype(np.float32)
            logger.info(f"Loaded visual features for {len(self.visual_features)} videos")
        else:
            self.visual_features = {}
            logger.warning(f"Visual features file not found: {vit_file}")
        
        # Load audio features
        audio_file = self._select_audio_feature_file(feature_dir)
        if audio_file and audio_file.exists():
            self.audio_features = torch.load(audio_file, map_location='cpu')
            # Convert to numpy
            for video_id in self.audio_features:
                self.audio_features[video_id] = self.audio_features[video_id].numpy().astype(np.float32)
            logger.info(f"Loaded audio features for {len(self.audio_features)} videos from {audio_file.name}")
        else:
            self.audio_features = {}
            logger.warning(f"Audio features file not found: {audio_file}")
    
    def _select_audio_feature_file(self, feature_dir: Path) -> Path:
        """
        Select audio feature file, prioritizing files with model marks
        
        Args:
            feature_dir: Directory containing feature files
        
        Returns:
            Path to selected audio feature file
        """
        if self.audio_feature_file:
            # If specific file is provided, use it
            return feature_dir / self.audio_feature_file
        
        # Look for files with pattern audio_features_frames*.pt
        audio_pattern = "audio_features_frames*.pt"
        candidates = list(feature_dir.glob(audio_pattern))
        
        if not candidates:
            # Fallback to default if no files found
            return feature_dir / "audio_features_frames.pt"
        
        # Filter out backup files and plain version
        filtered_candidates = []
        for candidate in candidates:
            name = candidate.name
            # Skip backup files and the plain version
            if name.endswith('.tensor_backup'):
                continue
            if name == 'audio_features_frames.pt':
                # Keep plain version as fallback, but with lowest priority
                continue
            filtered_candidates.append(candidate)
        
        if filtered_candidates:
            # Prioritize files with model names (CAiRE, wav2vec2, etc.)
            model_marked_files = []
            for candidate in filtered_candidates:
                name = candidate.name.lower()
                # Check for common model markers
                if any(marker in name for marker in ['caire', 'wav2vec2', 'bert', 'roberta', 'xlsr']):
                    model_marked_files.append(candidate)
            
            if model_marked_files:
                # Use the first (or most recent by default filesystem ordering) model-marked file
                selected = sorted(model_marked_files)[0]
                logger.info(f"Auto-selected audio feature file with model mark: {selected.name}")
                return selected
        
        # Fallback to any available file or default
        if candidates:
            selected = candidates[0]
            logger.info(f"Using available audio feature file: {selected.name}")
            return selected
        else:
            logger.info("Using default audio feature file: audio_features_frames.pt")
            return feature_dir / "audio_features_frames.pt"
    
    def get_temporal_splits(self):
        """Get temporal split video IDs"""
        if self.dataset == "FakeSV":
            vid_dir = self.data_dir / "vids"
            splits = {}
            
            for split in ['train', 'valid', 'test']:
                # Use filtered training file if available
                if split == 'train' and self.filter_k is not None:
                    split_file = vid_dir / f"vid_time3_train_k{self.filter_k}.txt"
                else:
                    split_file = vid_dir / f"vid_time3_{split}.txt"
                    
                if split_file.exists():
                    with open(split_file, 'r') as f:
                        splits[split] = set(line.strip() for line in f)
                    logger.info(f"{split.capitalize()} split: {len(splits[split])} videos")
                else:
                    splits[split] = set()
                    logger.warning(f"Split file not found: {split_file}")
        else:
            # For other datasets, create simple splits
            all_video_ids = set()
            for item in self.true_data + self.fake_data:
                all_video_ids.add(item['video_id'])
            
            video_list = sorted(list(all_video_ids))
            n_total = len(video_list)
            n_train = int(0.7 * n_total)
            n_valid = int(0.15 * n_total)
            
            splits = {
                'train': set(video_list[:n_train]),
                'valid': set(video_list[n_train:n_train + n_valid]),
                'test': set(video_list[n_train + n_valid:])
            }
        
        return splits
    
    def prepare_data(self, splits: Dict[str, Set[str]]):
        """Prepare data for retrieval"""
        # Create lookup for entity data
        entity_lookup = {}
        for item in self.true_data + self.fake_data:
            entity_lookup[item['video_id']] = item
        
        # Memory bank consists of train + valid samples
        memory_bank_video_ids = splits['train'] | splits['valid']
        
        # Prepare memory bank data
        self.memory_true_data = []
        self.memory_fake_data = []
        
        for video_id in memory_bank_video_ids:
            if video_id in entity_lookup:
                item = entity_lookup[video_id]
                annotation = item.get('annotation', '').lower()
                if annotation in ['真', 'real']:
                    self.memory_true_data.append(item)
                elif annotation in ['假', 'fake']:
                    self.memory_fake_data.append(item)
        
        # Prepare all query samples (train + valid + test)
        all_video_ids = splits['train'] | splits['valid'] | splits['test']
        self.query_data = []
        self.query_splits = {}
        
        for video_id in all_video_ids:
            if video_id in entity_lookup:
                item = entity_lookup[video_id]
                self.query_data.append(item)
                
                # Determine split
                if video_id in splits['train']:
                    self.query_splits[video_id] = 'train'
                elif video_id in splits['valid']:
                    self.query_splits[video_id] = 'valid'
                elif video_id in splits['test']:
                    self.query_splits[video_id] = 'test'
                else:
                    self.query_splits[video_id] = 'unknown'
        
        # Sort query data by video_id for consistent ordering across all outputs
        self.query_data = sorted(self.query_data, key=lambda x: x['video_id'])
        
        logger.info(f"Memory bank: {len(self.memory_true_data)} true, {len(self.memory_fake_data)} fake")
        logger.info(f"Query set: {len(self.query_data)} videos (sorted by video_id)")
    
    def l2_normalize(self, x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
        """L2 normalization"""
        n = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (n + eps)
    
    def create_text_representation(self, item: Dict, variant: str = "full") -> str:
        """
        Create text representation for embedding
        
        Args:
            item: Video item dictionary
            variant: Text variant type ('title_only', 'title_keywords', 'full')
        """
        parts = []
        
        # Get title/description
        title = item.get('title', '') or item.get('description', '')
        
        if variant == "title_only":
            # Only title
            if title:
                parts.append(str(title))
        
        elif variant == "title_keywords":
            # Title + keywords
            if title:
                parts.append(str(title))
            keywords_or_event = item.get('keywords', '') or item.get('event', '')
            if keywords_or_event:
                parts.append(str(keywords_or_event))
        
        elif variant == "full":
            # Full text: title + keywords + description + temporal_evolution
            if title:
                parts.append(str(title))
            
            keywords_or_event = item.get('keywords', '') or item.get('event', '')
            if keywords_or_event:
                parts.append(str(keywords_or_event))
            
            # Add description and temporal_evolution
            for field in ['description', 'temporal_evolution']:
                if field in item and item[field] and field != 'title':
                    parts.append(str(item[field]))
            
            # Add entity_claims if available
            if 'entity_claims' in item and item['entity_claims']:
                claims_text = []
                for entity, claims in item['entity_claims'].items():
                    for claim in claims:
                        claims_text.append(f"{entity}: {claim}")
                if claims_text:
                    parts.append(" ".join(claims_text))
        
        text_repr = " ".join(parts)
        if not text_repr.strip():
            # Fallback
            text_repr = item.get('video_id', 'unknown')
        
        return text_repr
    
    def compute_text_similarity(self, query_emb: np.ndarray, bank_emb: np.ndarray) -> np.ndarray:
        """Compute text similarity (cosine similarity)"""
        query_norm = self.l2_normalize(query_emb)
        bank_norm = self.l2_normalize(bank_emb, axis=1)
        return np.dot(bank_norm, query_norm)
    
    def compute_visual_audio_similarity(self, query_features: np.ndarray, bank_features: np.ndarray, 
                                       pool_type: str = "mean") -> np.ndarray:
        """
        Compute visual/audio similarity with different pooling strategies
        
        Args:
            query_features: [T_q, D] query video features
            bank_features: [N, T_c, D] bank video features
            pool_type: "mean" or "max" pooling
        """
        # Normalize features
        query_norm = self.l2_normalize(query_features, axis=-1)
        bank_norm = np.array([self.l2_normalize(f, axis=-1) for f in bank_features])
        
        if pool_type == "mean":
            # Mean pooling
            query_pooled = np.mean(query_norm, axis=0)
            query_pooled = self.l2_normalize(query_pooled)
            
            bank_pooled = np.mean(bank_norm, axis=1)
            bank_pooled = self.l2_normalize(bank_pooled, axis=1)
            
            similarities = np.dot(bank_pooled, query_pooled)
        
        elif pool_type == "max":
            # Max pooling (frame-level matching then max)
            N = bank_norm.shape[0]
            similarities = np.zeros(N)
            
            for i in range(N):
                # Compute frame-level similarity matrix
                sim_matrix = np.dot(query_norm, bank_norm[i].T)
                # For each query frame, find max similarity
                max_sims = np.max(sim_matrix, axis=1)
                # Average across query frames
                similarities[i] = np.mean(max_sims)
        
        return similarities
    
    def find_similar_videos_text(self, variant: str) -> List[Dict]:
        """Find similar videos using text-only retrieval"""
        logger.info(f"Running text retrieval with variant: {variant}")
        
        # Create text representations
        memory_true_texts = [self.create_text_representation(item, variant) for item in self.memory_true_data]
        memory_fake_texts = [self.create_text_representation(item, variant) for item in self.memory_fake_data]
        query_texts = [self.create_text_representation(item, variant) for item in self.query_data]
        
        # Encode texts
        logger.info("Encoding memory bank...")
        true_embeddings = self.encode_texts(memory_true_texts) if memory_true_texts else np.array([])
        fake_embeddings = self.encode_texts(memory_fake_texts) if memory_fake_texts else np.array([])
        
        logger.info("Encoding queries...")
        query_embeddings = self.encode_texts(query_texts)
        
        results = []
        
        for i, query_item in enumerate(tqdm(self.query_data, desc="Processing queries")):
            query_video_id = query_item['video_id']
            query_split = self.query_splits.get(query_video_id, 'unknown')
            query_emb = query_embeddings[i]
            
            # Find best true candidate
            true_candidate = self._find_best_text_candidate(
                query_emb, true_embeddings, self.memory_true_data, 
                query_video_id, query_split, 'true'
            )
            
            # Find best fake candidate
            fake_candidate = self._find_best_text_candidate(
                query_emb, fake_embeddings, self.memory_fake_data,
                query_video_id, query_split, 'fake'
            )
            
            result = {
                'query_video': {
                    'video_id': query_video_id,
                    'annotation': query_item.get('annotation', ''),
                    'split': query_split,
                    'title': query_item.get('title', ''),
                    'keywords': query_item.get('keywords', '') or query_item.get('event', ''),
                    'description': query_item.get('description', ''),
                    'temporal_evolution': query_item.get('temporal_evolution', ''),
                    'entity_claims': query_item.get('entity_claims', {}),
                    'publish_time': self.timestamp_lookup.get(query_video_id, {}).get('timestamp', 'Unknown')
                },
                'similar_true': true_candidate,
                'similar_fake': fake_candidate,
                'excluded_self': query_split in ['train', 'valid']
            }
            
            results.append(result)
        
        return results
    
    def _find_best_text_candidate(self, query_emb: np.ndarray, bank_embeddings: np.ndarray,
                                  memory_data: List[Dict], query_video_id: str, 
                                  query_split: str, candidate_type: str):
        """Find best text candidate from memory bank"""
        if len(bank_embeddings) == 0:
            return None
        
        # Compute similarities
        similarities = self.compute_text_similarity(query_emb, bank_embeddings)
        
        # Handle self-exclusion
        if query_split in ['train', 'valid']:
            for j, item in enumerate(memory_data):
                if item['video_id'] == query_video_id:
                    similarities[j] = -np.inf
        
        # Find best match
        best_idx = np.argmax(similarities)
        if similarities[best_idx] == -np.inf:
            return None
        
        best_item = memory_data[best_idx]
        
        return {
            'video_id': best_item['video_id'],
            'similarity_score': float(similarities[best_idx]),
            'text_similarity': float(similarities[best_idx]),
            'visual_similarity': 0.0,
            'audio_similarity': 0.0,
            'annotation': best_item['annotation'],
            'title': best_item.get('title', ''),
            'keywords': best_item.get('keywords', '') or best_item.get('event', ''),
            'description': best_item.get('description', ''),
            'temporal_evolution': best_item.get('temporal_evolution', ''),
            'entity_claims': best_item.get('entity_claims', {}),
            'publish_time': self.timestamp_lookup.get(best_item['video_id'], {}).get('timestamp', 'Unknown')
        }
    
    def find_similar_videos_visual(self, pool_type: str = "mean") -> List[Dict]:
        """Find similar videos using visual-only retrieval"""
        logger.info(f"Running visual retrieval with pool_type: {pool_type}")
        
        # Prepare visual features for memory bank
        memory_true_visual = []
        memory_true_indices = []
        for i, item in enumerate(self.memory_true_data):
            if item['video_id'] in self.visual_features:
                memory_true_visual.append(self.visual_features[item['video_id']])
                memory_true_indices.append(i)
        
        memory_fake_visual = []
        memory_fake_indices = []
        for i, item in enumerate(self.memory_fake_data):
            if item['video_id'] in self.visual_features:
                memory_fake_visual.append(self.visual_features[item['video_id']])
                memory_fake_indices.append(i)
        
        results = []
        
        for query_item in tqdm(self.query_data, desc="Processing queries"):
            query_video_id = query_item['video_id']
            query_split = self.query_splits.get(query_video_id, 'unknown')
            
            # Skip if no visual features
            if query_video_id not in self.visual_features:
                logger.warning(f"No visual features for query {query_video_id}")
                continue
            
            query_visual = self.visual_features[query_video_id]
            
            # Find best true candidate
            true_candidate = self._find_best_visual_candidate(
                query_visual, memory_true_visual, memory_true_indices,
                self.memory_true_data, query_video_id, query_split, pool_type
            )
            
            # Find best fake candidate  
            fake_candidate = self._find_best_visual_candidate(
                query_visual, memory_fake_visual, memory_fake_indices,
                self.memory_fake_data, query_video_id, query_split, pool_type
            )
            
            result = {
                'query_video': {
                    'video_id': query_video_id,
                    'annotation': query_item.get('annotation', ''),
                    'split': query_split,
                    'title': query_item.get('title', ''),
                    'keywords': query_item.get('keywords', '') or query_item.get('event', ''),
                    'description': query_item.get('description', ''),
                    'temporal_evolution': query_item.get('temporal_evolution', ''),
                    'entity_claims': query_item.get('entity_claims', {}),
                    'publish_time': self.timestamp_lookup.get(query_video_id, {}).get('timestamp', 'Unknown')
                },
                'similar_true': true_candidate,
                'similar_fake': fake_candidate,
                'excluded_self': query_split in ['train', 'valid']
            }
            
            results.append(result)
        
        return results
    
    def _find_best_visual_candidate(self, query_visual: np.ndarray, bank_visual: List[np.ndarray],
                                   bank_indices: List[int], memory_data: List[Dict],
                                   query_video_id: str, query_split: str, pool_type: str):
        """Find best visual candidate from memory bank"""
        if len(bank_visual) == 0:
            return None
        
        # Compute similarities
        bank_visual_array = np.array(bank_visual)
        similarities = self.compute_visual_audio_similarity(query_visual, bank_visual_array, pool_type)
        
        # Handle self-exclusion
        if query_split in ['train', 'valid']:
            for j, idx in enumerate(bank_indices):
                if memory_data[idx]['video_id'] == query_video_id:
                    similarities[j] = -np.inf
        
        # Find best match
        best_idx = np.argmax(similarities)
        if similarities[best_idx] == -np.inf:
            return None
        
        memory_idx = bank_indices[best_idx]
        best_item = memory_data[memory_idx]
        
        return {
            'video_id': best_item['video_id'],
            'similarity_score': float(similarities[best_idx]),
            'text_similarity': 0.0,
            'visual_similarity': float(similarities[best_idx]),
            'audio_similarity': 0.0,
            'annotation': best_item['annotation'],
            'title': best_item.get('title', ''),
            'keywords': best_item.get('keywords', '') or best_item.get('event', ''),
            'description': best_item.get('description', ''),
            'temporal_evolution': best_item.get('temporal_evolution', ''),
            'entity_claims': best_item.get('entity_claims', {}),
            'publish_time': self.timestamp_lookup.get(best_item['video_id'], {}).get('timestamp', 'Unknown')
        }
    
    def find_similar_videos_audio(self, pool_type: str = "mean") -> List[Dict]:
        """Find similar videos using audio-only retrieval"""
        logger.info(f"Running audio retrieval with pool_type: {pool_type}")
        
        # Prepare audio features for memory bank
        memory_true_audio = []
        memory_true_indices = []
        for i, item in enumerate(self.memory_true_data):
            if item['video_id'] in self.audio_features:
                memory_true_audio.append(self.audio_features[item['video_id']])
                memory_true_indices.append(i)
        
        memory_fake_audio = []
        memory_fake_indices = []
        for i, item in enumerate(self.memory_fake_data):
            if item['video_id'] in self.audio_features:
                memory_fake_audio.append(self.audio_features[item['video_id']])
                memory_fake_indices.append(i)
        
        results = []
        
        for query_item in tqdm(self.query_data, desc="Processing queries"):
            query_video_id = query_item['video_id']
            query_split = self.query_splits.get(query_video_id, 'unknown')
            
            # Skip if no audio features
            if query_video_id not in self.audio_features:
                logger.warning(f"No audio features for query {query_video_id}")
                continue
            
            query_audio = self.audio_features[query_video_id]
            
            # Find best true candidate
            true_candidate = self._find_best_audio_candidate(
                query_audio, memory_true_audio, memory_true_indices,
                self.memory_true_data, query_video_id, query_split, pool_type
            )
            
            # Find best fake candidate
            fake_candidate = self._find_best_audio_candidate(
                query_audio, memory_fake_audio, memory_fake_indices,
                self.memory_fake_data, query_video_id, query_split, pool_type
            )
            
            result = {
                'query_video': {
                    'video_id': query_video_id,
                    'annotation': query_item.get('annotation', ''),
                    'split': query_split,
                    'title': query_item.get('title', ''),
                    'keywords': query_item.get('keywords', '') or query_item.get('event', ''),
                    'description': query_item.get('description', ''),
                    'temporal_evolution': query_item.get('temporal_evolution', ''),
                    'entity_claims': query_item.get('entity_claims', {}),
                    'publish_time': self.timestamp_lookup.get(query_video_id, {}).get('timestamp', 'Unknown')
                },
                'similar_true': true_candidate,
                'similar_fake': fake_candidate,
                'excluded_self': query_split in ['train', 'valid']
            }
            
            results.append(result)
        
        return results
    
    def _find_best_audio_candidate(self, query_audio: np.ndarray, bank_audio: List[np.ndarray],
                                  bank_indices: List[int], memory_data: List[Dict],
                                  query_video_id: str, query_split: str, pool_type: str):
        """Find best audio candidate from memory bank"""
        if len(bank_audio) == 0:
            return None
        
        # Compute similarities
        bank_audio_array = np.array(bank_audio)
        similarities = self.compute_visual_audio_similarity(query_audio, bank_audio_array, pool_type)
        
        # Handle self-exclusion
        if query_split in ['train', 'valid']:
            for j, idx in enumerate(bank_indices):
                if memory_data[idx]['video_id'] == query_video_id:
                    similarities[j] = -np.inf
        
        # Find best match
        best_idx = np.argmax(similarities)
        if similarities[best_idx] == -np.inf:
            return None
        
        memory_idx = bank_indices[best_idx]
        best_item = memory_data[memory_idx]
        
        return {
            'video_id': best_item['video_id'],
            'similarity_score': float(similarities[best_idx]),
            'text_similarity': 0.0,
            'visual_similarity': 0.0,
            'audio_similarity': float(similarities[best_idx]),
            'annotation': best_item['annotation'],
            'title': best_item.get('title', ''),
            'keywords': best_item.get('keywords', '') or best_item.get('event', ''),
            'description': best_item.get('description', ''),
            'temporal_evolution': best_item.get('temporal_evolution', ''),
            'entity_claims': best_item.get('entity_claims', {}),
            'publish_time': self.timestamp_lookup.get(best_item['video_id'], {}).get('timestamp', 'Unknown')
        }
    
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
        
        for result in results:
            if result['similar_true']:
                true_similarities.append(result['similar_true']['similarity_score'])
            if result['similar_fake']:
                fake_similarities.append(result['similar_fake']['similarity_score'])
        
        stats['avg_true_similarity'] = float(np.mean(true_similarities)) if true_similarities else 0.0
        stats['avg_fake_similarity'] = float(np.mean(fake_similarities)) if fake_similarities else 0.0
        
        # Save stats
        stats_file = output_file.with_suffix('.stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Summary: {stats['total_queries']} queries, "
                   f"Avg similarities - True: {stats['avg_true_similarity']:.4f}, "
                   f"Fake: {stats['avg_fake_similarity']:.4f}")
    
    def run_all_analyses(self):
        """Run all unimodal retrieval analyses"""
        logger.info("="*60)
        logger.info("Starting Unimodal Retrieval Analysis")
        logger.info("="*60)
        
        # Load data
        self.load_entity_data()
        splits = self.get_temporal_splits()
        self.prepare_data(splits)
        
        # Load model for text retrieval
        self.load_model()
        
        # Text retrieval variants
        text_variants = [
            ("title_only", "Text retrieval with title only"),
            ("title_keywords", "Text retrieval with title + keywords"),
            ("full", "Text retrieval with full text (title + keywords + description + temporal)")
        ]
        
        for variant, description in text_variants:
            logger.info(f"\n{description}")
            results = self.find_similar_videos_text(variant)
            
            output_file = self.output_dir / f"unimodal_text_{variant}_retrieval.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.generate_summary(results, output_file)
            logger.info(f"Saved results to {output_file}")
        
        # Load multimodal features for visual/audio retrieval
        self.load_multimodal_features()
        
        # Visual retrieval variants
        visual_variants = [
            ("mean", "Visual retrieval with mean pooling"),
            ("max", "Visual retrieval with max pooling")
        ]
        
        for pool_type, description in visual_variants:
            logger.info(f"\n{description}")
            results = self.find_similar_videos_visual(pool_type)
            
            output_file = self.output_dir / f"unimodal_visual_{pool_type}_retrieval.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.generate_summary(results, output_file)
            logger.info(f"Saved results to {output_file}")
        
        # Audio retrieval variants
        audio_variants = [
            ("mean", "Audio retrieval with mean pooling"),
            ("max", "Audio retrieval with max pooling")
        ]
        
        for pool_type, description in audio_variants:
            logger.info(f"\n{description}")
            results = self.find_similar_videos_audio(pool_type)
            
            output_file = self.output_dir / f"unimodal_audio_{pool_type}_retrieval.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.generate_summary(results, output_file)
            logger.info(f"Saved results to {output_file}")
        
        logger.info("\n" + "="*60)
        logger.info("Unimodal Retrieval Analysis Complete")
        logger.info(f"All results saved to: {self.output_dir}")
        logger.info("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze unimodal retrieval performance')
    parser.add_argument('--dataset', type=str, default='FakeSV',
                       help='Dataset name (default: FakeSV). Examples: FakeSV, FakeTT')
    parser.add_argument('--filter-k', type=int, default=None,
                       help='Number of most similar training samples to remove per test sample')
    parser.add_argument('--audio-feature-file', type=str, default=None,
                       help='Specific audio feature file to use (default: auto-detect with model mark preference)')
    
    args = parser.parse_args()
    
    analyzer = UnimodalRetrievalAnalyzer(
        dataset=args.dataset,
        filter_k=args.filter_k,
        audio_feature_file=args.audio_feature_file
    )
    
    analyzer.run_all_analyses()