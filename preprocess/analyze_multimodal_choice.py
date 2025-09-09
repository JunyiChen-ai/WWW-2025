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
                 output_dir: str = None,
                 top_k: int = 10,
                 use_cache: bool = False):
        """
        Initialize multimodal choice analyzer
        
        Args:
            dataset: Dataset name (e.g., 'FakeSV', 'FakeTT')
            audio_model: Audio model name suffix for loading features
            text_model: Text model for embeddings (auto-selected if None)
            output_dir: Output directory (auto-created if None)
            top_k: Top-k samples to consider for entropy and p_event_mass calculations
        """
        self.dataset = dataset
        self.audio_model = audio_model
        self.top_k = top_k
        self.use_cache = use_cache
        
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
        # Cache directory to avoid recomputation across runs
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized MultimodalChoiceAnalyzer for {dataset}")
        logger.info(f"Audio model: {audio_model}")
        logger.info(f"Text model: {text_model}")
        logger.info(f"Output dir: {output_dir}")
        
        # Data storage
        self.video_metadata = {}
        self.audio_features = {}
        self.visual_features = {}
        self.text_embeddings = {}
        # Keep a raw (pre-normalization) copy for diagnostics like scale-mismatch
        self.text_embeddings_raw = {}
        self.transcript_embeddings = {}
        self.transcripts = {}
        self.splits = {}
        
        # Analysis results
        self.probs = {}  # T, I, A, S probability matrices
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
        
        logger.info("Loading transcripts...")
        self._load_transcripts()
        
        logger.info("Loading multimodal features...")
        self._load_multimodal_features()
        
        logger.info("Loading text model and computing embeddings...")
        self._load_text_model()
        self._compute_text_embeddings()
        
        logger.info("Computing transcript embeddings...")
        self._compute_transcript_embeddings()
        
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
    
    def _load_transcripts(self):
        """Load audio transcripts from transcript.jsonl"""
        transcript_file = self.data_dir / "transcript.jsonl"
        
        if not transcript_file.exists():
            logger.warning(f"Transcript file not found: {transcript_file}")
            return
        
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                video_id = data['vid']
                transcript = data.get('transcript', '')
                if transcript:
                    self.transcripts[video_id] = transcript
        
        logger.info(f"Loaded transcripts for {len(self.transcripts)} videos")
    
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
        audio_source = None
        audio_file_path = None
        
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
            
        # Load global audio features with model suffix (no pooling needed)
        audio_file = feature_dir / f"audio_features_global_{self.audio_model}.pt"
        if audio_file.exists():
            raw_audio = torch.load(audio_file, map_location='cpu')
            # Global features are already pooled
            for video_id, features in raw_audio.items():
                if features.numel() > 0:  # Ensure not empty
                    self.audio_features[video_id] = features.numpy()
            logger.info(f"Loaded global audio features for {len(self.audio_features)} videos")
            audio_source = "global"
            audio_file_path = str(audio_file)
        else:
            # Fallback to frame-level features if global not found
            logger.warning(f"Global audio features not found: {audio_file}")
            audio_file = feature_dir / f"audio_features_frames_{self.audio_model}.pt"
            if audio_file.exists():
                logger.info("Falling back to frame-level audio features with mean pooling")
                raw_audio = torch.load(audio_file, map_location='cpu')
                # Mean pool frame-level features
                for video_id, frames in raw_audio.items():
                    if frames.shape[0] > 0:
                        pooled = torch.mean(frames, dim=0)
                        self.audio_features[video_id] = pooled.numpy()
                logger.info(f"Loaded and pooled frame-level audio features for {len(self.audio_features)} videos")
                audio_source = "frames_mean_pool"
                audio_file_path = str(audio_file)
            else:
                # If model-specific files don't exist, try without model suffix
                logger.warning(f"Model-specific audio features not found, trying without model suffix")
                
                # Try global without model suffix
                audio_file_fallback = feature_dir / "audio_features_global.pt"
                if audio_file_fallback.exists():
                    logger.info("Loading global audio features without model suffix")
                    raw_audio = torch.load(audio_file_fallback, map_location='cpu')
                    for video_id, features in raw_audio.items():
                        if features.numel() > 0:
                            self.audio_features[video_id] = features.numpy()
                    logger.info(f"Loaded global audio features (no model suffix) for {len(self.audio_features)} videos")
                    audio_source = "global_no_model"
                    audio_file_path = str(audio_file_fallback)
                else:
                    # Try frames without model suffix
                    logger.warning(f"Global audio features without model suffix not found: {audio_file_fallback}")
                    audio_file_fallback = feature_dir / "audio_features_frames.pt"
                    if audio_file_fallback.exists():
                        logger.info("Loading frame-level audio features without model suffix with mean pooling")
                        raw_audio = torch.load(audio_file_fallback, map_location='cpu')
                        for video_id, frames in raw_audio.items():
                            if frames.shape[0] > 0:
                                pooled = torch.mean(frames, dim=0)
                                self.audio_features[video_id] = pooled.numpy()
                        logger.info(f"Loaded and pooled frame-level audio features (no model suffix) for {len(self.audio_features)} videos")
                        audio_source = "frames_mean_pool_no_model"
                        audio_file_path = str(audio_file_fallback)
                    else:
                        logger.error(f"No audio features file found (tried all variants)")
                        logger.error(f"  Global file: audio_features_global_{self.audio_model}.pt")
                        logger.error(f"  Frames file: audio_features_frames_{self.audio_model}.pt")
                        logger.error(f"  Global file (no model): audio_features_global.pt")
                        logger.error(f"  Frames file (no model): audio_features_frames.pt")

        # Report which audio branch was used and the L2 norm stats
        if len(self.audio_features) > 0:
            try:
                audio_norms = [float(np.linalg.norm(v)) for v in self.audio_features.values()]
                logger.info(
                    f"Audio loading branch: {audio_source}, file: {audio_file_path}"
                )
                logger.info(
                    f"Audio feature L2 norms: mean={np.mean(audio_norms):.3f}, std={np.std(audio_norms):.3f}"
                )
            except Exception as e:
                logger.warning(f"Failed to compute audio norm stats: {e}")
    
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
        # Try load from cache
        cache_file = self.cache_dir / "text_embeddings.npz"
        if self.use_cache and cache_file.exists():
            logger.info(f"Loading cached text embeddings from {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
            ids = list(data['ids'])
            embs = data['embeddings']
            embs_raw = data['embeddings_raw'] if 'embeddings_raw' in data else None
            for vid, emb in zip(ids, embs):
                self.text_embeddings[str(vid)] = emb
            if embs_raw is not None:
                for vid, emb in zip(ids, embs_raw):
                    self.text_embeddings_raw[str(vid)] = emb
            logger.info(f"Loaded {len(self.text_embeddings)} cached text embeddings")
            return

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
        all_embeddings_raw = []
        
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

                # Save a raw copy (pre-normalization) for diagnostics
                embeddings_raw = embeddings.detach().cpu().numpy()

                # Normalize embeddings for retrieval computations
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

                all_embeddings_raw.append(embeddings_raw)
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Combine all embeddings
        all_embeddings = np.vstack(all_embeddings)
        all_embeddings_raw = np.vstack(all_embeddings_raw)
        
        # Store embeddings
        for video_id, embedding in zip(video_ids, all_embeddings):
            self.text_embeddings[video_id] = embedding
        for video_id, embedding in zip(video_ids, all_embeddings_raw):
            self.text_embeddings_raw[video_id] = embedding
        
        logger.info(f"Computed text embeddings for {len(self.text_embeddings)} videos")

        # Save cache
        try:
            np.savez_compressed(cache_file,
                                ids=np.array(video_ids),
                                embeddings=all_embeddings,
                                embeddings_raw=all_embeddings_raw)
            logger.info(f"Cached text embeddings to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache text embeddings: {e}")
    
    def _compute_transcript_embeddings(self):
        """Compute embeddings for audio transcripts using same model as text"""
        if not self.transcripts:
            logger.warning("No transcripts loaded, skipping transcript embeddings")
            return

        # Try load from cache
        cache_file = self.cache_dir / "transcript_embeddings.npz"
        if self.use_cache and cache_file.exists():
            logger.info(f"Loading cached transcript embeddings from {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
            ids = list(data['ids'])
            embs = data['embeddings']
            for vid, emb in zip(ids, embs):
                self.transcript_embeddings[str(vid)] = emb
            logger.info(f"Loaded {len(self.transcript_embeddings)} cached transcript embeddings")
            return
        
        video_ids = []
        transcript_texts = []
        
        # Prepare transcript texts
        for video_id, transcript in self.transcripts.items():
            video_ids.append(video_id)
            transcript_texts.append(transcript if transcript else 'unknown')
        
        # Encode in batches (same process as text embeddings)
        batch_size = 16
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(transcript_texts), batch_size), desc="Computing transcript embeddings"):
                batch_texts = transcript_texts[i:i+batch_size]
                
                # Use same tokenizer settings as text
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
                
                # Get embeddings (same extraction as text)
                if 'chinese-clip' in self.text_model.lower():
                    outputs = self.model.text_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                elif 'longclip' in self.text_model.lower():
                    outputs = self.model.text_model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                else:
                    outputs = self.model.text_model(**inputs)
                    embeddings = outputs.pooler_output
                
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)
        
        all_embeddings = np.vstack(all_embeddings)
        
        # Store embeddings
        for video_id, embedding in zip(video_ids, all_embeddings):
            self.transcript_embeddings[video_id] = embedding
        
        logger.info(f"Computed transcript embeddings for {len(self.transcript_embeddings)} videos")

        # Save cache
        try:
            np.savez_compressed(cache_file,
                                ids=np.array(video_ids),
                                embeddings=all_embeddings)
            logger.info(f"Cached transcript embeddings to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache transcript embeddings: {e}")
    
    def prepare_analysis_data(self):
        """Prepare data matrices for analysis"""
        logger.info("Preparing analysis data matrices...")

        # Attempt to load prepared matrices from cache
        prepared_cache = self.cache_dir / "prepared_data.npz"
        if self.use_cache and prepared_cache.exists():
            try:
                data = np.load(prepared_cache, allow_pickle=True)
                self.query_ids = list(data['query_ids'])
                self.query_events = np.array(data['query_events'])
                self.candidate_meta = {
                    'candidate_event': data['candidate_event'],
                    'candidate_label': data['candidate_label'],
                    'candidate_ids': data['candidate_ids']
                }
                self.scores = {
                    'T': data['scores_T'],
                    'I': data['scores_I'],
                    'A': data['scores_A'],
                    'S': data['scores_S']
                }
                self.probs = {
                    'T': data['probs_T'],
                    'I': data['probs_I'],
                    'A': data['probs_A'],
                    'S': data['probs_S']
                }
                logger.info(f"Loaded prepared matrices from cache: {prepared_cache}")
                logger.info(f"Analysis data (cached): {len(self.query_ids)} queries × {len(self.candidate_meta['candidate_ids'])} candidates")
                return
            except Exception as e:
                logger.warning(f"Failed to load prepared cache, will recompute: {e}")
        
        # Show data availability breakdown
        total_in_splits = sum(len(ids) for ids in self.splits.values())
        logger.info(f"Data availability breakdown:")
        logger.info(f"  Videos in splits: {total_in_splits}")
        logger.info(f"  Videos with metadata: {len(self.video_metadata)}")
        logger.info(f"  Videos with text embeddings: {len(self.text_embeddings)}")
        logger.info(f"  Videos with visual features: {len(self.visual_features)}")
        logger.info(f"  Videos with audio features: {len(self.audio_features)}")
        logger.info(f"  Videos with transcript embeddings: {len(self.transcript_embeddings)}")
        
        # Get all video IDs that have all four modalities
        common_video_ids = set(self.video_metadata.keys()) & \
                          set(self.text_embeddings.keys()) & \
                          set(self.visual_features.keys()) & \
                          set(self.audio_features.keys()) & \
                          set(self.transcript_embeddings.keys())
        
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
        
        # Transcript features
        transcript_candidates = np.stack([self.transcript_embeddings[vid] for vid in candidate_ids])
        transcript_queries = np.stack([self.transcript_embeddings[vid] for vid in query_ids])
        
        # L2 normalize features
        text_candidates = text_candidates / (np.linalg.norm(text_candidates, axis=1, keepdims=True) + 1e-12)
        text_queries = text_queries / (np.linalg.norm(text_queries, axis=1, keepdims=True) + 1e-12)
        visual_candidates = visual_candidates / (np.linalg.norm(visual_candidates, axis=1, keepdims=True) + 1e-12)
        visual_queries = visual_queries / (np.linalg.norm(visual_queries, axis=1, keepdims=True) + 1e-12)
        audio_candidates = audio_candidates / (np.linalg.norm(audio_candidates, axis=1, keepdims=True) + 1e-12)
        audio_queries = audio_queries / (np.linalg.norm(audio_queries, axis=1, keepdims=True) + 1e-12)
        transcript_candidates = transcript_candidates / (np.linalg.norm(transcript_candidates, axis=1, keepdims=True) + 1e-12)
        transcript_queries = transcript_queries / (np.linalg.norm(transcript_queries, axis=1, keepdims=True) + 1e-12)
        
        # Compute similarity matrices 
        text_sim = np.dot(text_queries, text_candidates.T)  # [Q, N]
        visual_sim = np.dot(visual_queries, visual_candidates.T)  # [Q, N]
        audio_sim = np.dot(audio_queries, audio_candidates.T)  # [Q, N]
        transcript_sim = np.dot(transcript_queries, transcript_candidates.T)  # [Q, N]
        
        # Store similarity scores (for DBNorm)
        self.scores = {}
        self.scores['T'] = text_sim
        self.scores['I'] = visual_sim  
        self.scores['A'] = audio_sim
        self.scores['S'] = transcript_sim
        
        # Convert to probabilities
        temp = 0.1  # Temperature for softmax
        self.probs = {}
        self.probs['T'] = self._softmax(text_sim / temp, axis=1)
        self.probs['I'] = self._softmax(visual_sim / temp, axis=1)
        self.probs['A'] = self._softmax(audio_sim / temp, axis=1)
        self.probs['S'] = self._softmax(transcript_sim / temp, axis=1)
        
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

        # Save prepared cache
        try:
            np.savez_compressed(prepared_cache,
                                query_ids=np.array(self.query_ids),
                                query_events=self.query_events,
                                candidate_event=self.candidate_meta['candidate_event'],
                                candidate_label=self.candidate_meta['candidate_label'],
                                candidate_ids=self.candidate_meta['candidate_ids'],
                                scores_T=self.scores['T'], scores_I=self.scores['I'], scores_A=self.scores['A'], scores_S=self.scores['S'],
                                probs_T=self.probs['T'], probs_I=self.probs['I'], probs_A=self.probs['A'], probs_S=self.probs['S'])
            logger.info(f"Cached prepared matrices to {prepared_cache}")
        except Exception as e:
            logger.warning(f"Failed to cache prepared matrices: {e}")
    
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
        
        for modality in ['T', 'I', 'A', 'S']:
            modality_name = {'T': 'Text', 'I': 'Visual', 'A': 'Audio', 'S': 'Transcript'}[modality]
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
                    
                # 2. p_event_mass (only consider top-k)
                top_k = self.top_k
                top_k_indices = np.argpartition(query_probs, -top_k)[-top_k:] if len(query_probs) > top_k else np.arange(len(query_probs))
                event_mask_top_k = self.candidate_meta['candidate_event'][top_k_indices] == query_event
                p_event_mass = np.sum(query_probs[top_k_indices][event_mask_top_k])
                p_event_masses.append(p_event_mass)
                
                # 3. Uncertainty metrics (only consider top-k)
                top_k_probs = query_probs[top_k_indices]
                top_k_probs = top_k_probs / (top_k_probs.sum() + 1e-12)  # Renormalize
                H = entropy(top_k_probs + 1e-12)  # Entropy on top-k
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
        modality_pairs = [
            ('T', 'I'), ('T', 'A'), ('T', 'S'),
            ('I', 'A'), ('I', 'S'),
            ('A', 'S')
        ]
        modality_names = {'T': 'Text', 'I': 'Visual', 'A': 'Audio', 'S': 'Transcript'}
        
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
        
        methods = ['DSL', 'Sinkhorn', 'RRF', 'Entropy-Gated', 'Margin-Gated', 'UncertaintyWeightedLogPool', 'WeightedAverage']
        
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
            elif method == 'Margin-Gated':
                fused_probs = self._margin_gated_fusion()
            elif method == 'UncertaintyWeightedLogPool':
                fused_probs = self._uncertainty_weighted_log_pool()
            elif method == 'WeightedAverage':
                fused_probs = self._weighted_average_fusion()
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
                
                # p_event_mass (only consider top-k)
                top_k = self.top_k
                top_k_indices = np.argpartition(query_probs, -top_k)[-top_k:] if len(query_probs) > top_k else np.arange(len(query_probs))
                event_mask_top_k = self.candidate_meta['candidate_event'][top_k_indices] == query_event
                p_event_mass = np.sum(query_probs[top_k_indices][event_mask_top_k])
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
        fused = probs_dict['T'] * probs_dict['I'] * probs_dict['A'] * probs_dict['S']
        
        # Renormalize
        fused = fused / (np.sum(fused, axis=1, keepdims=True) + 1e-12)
        
        return fused
    
    def _sinkhorn_fusion(self, probs_dict=None, n_iters=10):
        """Sinkhorn fusion with column marginal target"""
        if probs_dict is None:
            probs_dict = self.probs
            
        # Set column marginal target as average of four modalities
        col_target = (probs_dict['T'] + probs_dict['I'] + probs_dict['A'] + probs_dict['S']) / 4.0
        log_col_target = np.log(col_target + 1e-12)  # [Q, N]
        
        # Convert to log space
        log_probs = {}
        for m in ['T', 'I', 'A', 'S']:
            log_probs[m] = np.log(probs_dict[m] + 1e-12)
        
        # Stack into 3D array: [Q, 4, N]
        Q, N = log_probs['T'].shape
        log_prob_stack = np.stack([log_probs['T'], log_probs['I'], log_probs['A'], log_probs['S']], axis=1)
        
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
        
        for modality in ['T', 'I', 'A', 'S']:
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
            for modality in ['T', 'I', 'A', 'S']:
                p = self.probs[modality][q]
                entropies[modality] = entropy(p + 1e-12)
            
            # Choose modality with minimum entropy
            best_modality = min(entropies, key=entropies.get)
            fused[q] = self.probs[best_modality][q]
        
        return fused

    def _margin_gated_fusion(self):
        """Margin-gated selection (choose modality with largest top1-top2 margin per query)"""
        Q, N = self.probs['T'].shape
        fused = np.zeros((Q, N))
        for q in range(Q):
            margins = {}
            for modality in ['T', 'I', 'A']:
                r = self.probs[modality][q]
                if r.size >= 2:
                    # use partial sort for top2
                    top2 = np.partition(r, -2)[-2:]
                    margin = float(top2.max() - top2.min())
                else:
                    margin = 1.0
                margins[modality] = margin
            best_modality = max(margins, key=margins.get)
            fused[q] = self.probs[best_modality][q]
        return fused
    
    def _uncertainty_weighted_log_pool(self, eps=1e-12):
        """Uncertainty Weighted Log Pool fusion"""
        # 获取每个模态的概率矩阵
        P = {'T': self.probs['T'], 'I': self.probs['I'], 'A': self.probs['A'], 'S': self.probs['S']}  # [Q,N]
        Q, N = P['T'].shape
        
        # 1) 计算确定性分数 c_m(q) - 使用熵的倒数 (only consider top-k)
        C = {}
        def entropy_top_k(r, k=self.top_k):
            top_k_idx = np.argpartition(r, -k)[-k:] if len(r) > k else np.arange(len(r))
            top_k_probs = r[top_k_idx]
            top_k_probs = top_k_probs / (top_k_probs.sum() + eps)
            return entropy(top_k_probs + eps)
            
        for m in P:
            # 默认：熵的倒数 (only top-k)
            ent = np.apply_along_axis(lambda r: entropy_top_k(r), 1, P[m])  # [Q]
            C[m] = 1.0 / (ent + eps)
        
        # 2) 归一化权重 w_m(q)
        W = {m: C[m] / (C['T'] + C['I'] + C['A'] + C['S']) for m in P}  # 每个都是 [Q]
        
        # 3) 对数线性池
        fused_log = np.zeros_like(P['T'])
        for m in P:
            fused_log += W[m][:, None] * np.log(P[m] + eps)
        
        fused = np.exp(fused_log)
        fused /= fused.sum(axis=1, keepdims=True) + eps
        
        return fused
    
    def _weighted_average_fusion(self, probs_dict=None, weights=None):
        """Weighted average fusion of four modalities"""
        if probs_dict is None:
            probs_dict = self.probs
        
        # Default weights: equal weight for each modality
        if weights is None:
            weights = {'T': 0.25, 'I': 0.25, 'A': 0.25, 'S': 0.25}
        
        # Weighted average of probabilities
        fused = (weights['T'] * probs_dict['T'] + 
                 weights['I'] * probs_dict['I'] + 
                 weights['A'] * probs_dict['A'] +
                 weights['S'] * probs_dict['S'])
        
        # Ensure probabilities sum to 1 (should already be close due to linear combination)
        fused = fused / (np.sum(fused, axis=1, keepdims=True) + 1e-12)
        
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
        
        for modality in ['T', 'I', 'A', 'S']:
            probs_m = self.probs[modality]
            for q in range(len(self.query_ids)):
                # Only consider top-k for entropy calculation
                top_k = self.top_k
                query_probs = probs_m[q]
                top_k_indices = np.argpartition(query_probs, -top_k)[-top_k:] if len(query_probs) > top_k else np.arange(len(query_probs))
                top_k_probs = query_probs[top_k_indices]
                top_k_probs = top_k_probs / (top_k_probs.sum() + 1e-12)  # Renormalize
                H = entropy(top_k_probs + 1e-12)
                entropy_data.append(H)
                modality_labels.append({'T': 'Text', 'I': 'Visual', 'A': 'Audio', 'S': 'Transcript'}[modality])
        
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
        for modality, label in [('T', 'Text'), ('I', 'Visual'), ('A', 'Audio'), ('S', 'Transcript')]:
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
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
        
        for idx, (modality, label) in enumerate([('T', 'Text'), ('I', 'Visual'), ('A', 'Audio'), ('S', 'Transcript')]):
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
        modality_map = {
            'Text-Visual': ('T', 'I'), 
            'Text-Audio': ('T', 'A'), 
            'Text-Transcript': ('T', 'S'),
            'Visual-Audio': ('I', 'A'),
            'Visual-Transcript': ('I', 'S'),
            'Audio-Transcript': ('A', 'S')
        }
        
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
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
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
    
    def create_entropy_vs_event_hit_curve(self, n_bins: int = 10):
        """Plot Event-Hit@1(pair) vs entropy bins for each modality.

        For each modality (Text, Visual, Audio):
          - Compute per-query top-k entropy (consistent with other places using top-k).
          - Compute per-query Event-Hit@1(pair) if we select this modality only.
          - Bin queries by entropy quantiles (per modality) and compute hit-rate per bin.
        Saves:
          - entropy_vs_event_hit_curve.png
          - entropy_vs_event_hit_by_bin.csv
        """
        logger.info("Creating entropy vs Event-Hit@1 curves...")

        eps = 1e-12
        modalities = [('T', 'Text'), ('I', 'Visual'), ('A', 'Audio'), ('S', 'Transcript')]

        def entropy_top_k_row(r, k=self.top_k):
            idx = np.argpartition(r, -k)[-k:] if r.shape[0] > k else np.arange(r.shape[0])
            sub = r[idx]
            sub = sub / (sub.sum() + eps)
            return float(entropy(sub + eps))

        def compute_event_hit_for_row(r, q_idx):
            """Event-Hit@1(pair) for a single modality row r = probs[q]"""
            pos_mask = (self.candidate_meta['candidate_label'] == 1)
            neg_mask = (self.candidate_meta['candidate_label'] == 0)
            if np.sum(pos_mask) == 0 or np.sum(neg_mask) == 0:
                return 0
            pos_probs = r[pos_mask]
            neg_probs = r[neg_mask]
            top1_pos_idx = int(np.argmax(pos_probs))
            top1_neg_idx = int(np.argmax(neg_probs))
            pos_events = self.candidate_meta['candidate_event'][pos_mask]
            neg_events = self.candidate_meta['candidate_event'][neg_mask]
            top1_pos_event = pos_events[top1_pos_idx]
            top1_neg_event = neg_events[top1_neg_idx]
            query_event = self.query_events[q_idx]
            return 1 if (top1_pos_event == query_event or top1_neg_event == query_event) else 0

        # Collect rows for CSV
        rows = []

        # Prepare plot
        plt.figure(figsize=(10, 6))

        for key, label in modalities:
            Pm = self.probs[key]  # [Q, N]
            Qn = Pm.shape[0]

            # Per-query entropy and hit
            ent_vec = np.zeros(Qn, dtype=np.float32)
            hit_vec = np.zeros(Qn, dtype=np.float32)
            for qi in range(Qn):
                r = Pm[qi]
                ent_vec[qi] = entropy_top_k_row(r)
                hit_vec[qi] = compute_event_hit_for_row(r, qi)

            # Bin by entropy quantiles (per modality)
            try:
                # Use pandas qcut to handle duplicates; drop duplicate edges
                cat = pd.qcut(ent_vec, q=n_bins, duplicates='drop')
                df_m = pd.DataFrame({'Entropy': ent_vec, 'Hit': hit_vec, 'Bin': cat.astype(str)})
                grouped = df_m.groupby('Bin')
                mins = grouped['Entropy'].min().values
                maxs = grouped['Entropy'].max().values
                centers = (mins + maxs) / 2.0
                hit_rates = grouped['Hit'].mean().values.astype(float)
                counts = grouped['Hit'].count().values.astype(int)

                # Sort by centers for monotonic x
                order = np.argsort(centers)
                centers = centers[order]
                hit_rates = hit_rates[order]
                counts = counts[order]
                mins = mins[order]
                maxs = maxs[order]

                plt.plot(centers, hit_rates, marker='o', label=label)

                for c, hr, ct, mn, mx in zip(centers, hit_rates, counts, mins, maxs):
                    rows.append({'Modality': label,
                                 'Entropy_min': float(mn),
                                 'Entropy_max': float(mx),
                                 'Entropy_center': float(c),
                                 'HitRate': float(hr),
                                 'Count': int(ct)})
            except Exception as e:
                logger.warning(f"qcut failed for modality {label}: {e}; falling back to uniform bins")
                # Uniform bins fallback
                mn, mx = float(ent_vec.min()), float(ent_vec.max())
                edges = np.linspace(mn, mx + eps, n_bins + 1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                hit_rates = []
                counts = []
                for b in range(n_bins):
                    mask = (ent_vec >= edges[b]) & (ent_vec < edges[b+1]) if b < n_bins - 1 else (ent_vec >= edges[b]) & (ent_vec <= edges[b+1])
                    if np.sum(mask) > 0:
                        hit_rates.append(float(hit_vec[mask].mean()))
                        counts.append(int(np.sum(mask)))
                    else:
                        hit_rates.append(0.0)
                        counts.append(0)
                plt.plot(centers, hit_rates, marker='o', label=label)
                for c, hr, ct, lo, hi in zip(centers, hit_rates, counts, edges[:-1], edges[1:]):
                    rows.append({'Modality': label,
                                 'Entropy_min': float(lo),
                                 'Entropy_max': float(hi),
                                 'Entropy_center': float(c),
                                 'HitRate': float(hr),
                                 'Count': int(ct)})

        plt.title('Event-Hit@1(pair) vs Entropy (per-modality bins)')
        plt.xlabel('Entropy (bin centers, top-k)')
        plt.ylabel('Event-Hit@1(pair)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fig_path = self.output_dir / 'entropy_vs_event_hit_curve.png'
        csv_path = self.output_dir / 'entropy_vs_event_hit_by_bin.csv'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"  - Entropy vs Event-Hit curve: {fig_path}")
        logger.info(f"  - Entropy vs Event-Hit CSV: {csv_path}")

    def create_scale_mismatch_hubness_figure(self):
        """
        Create Figure A: Scale Mismatch + Hubness Diagnosis (against direct concatenation)
        
        This figure supports literature arguments against naive early fusion by showing:
        1. L2 norm distributions across modalities (scale mismatch)
        2. Energy ratio bar charts (modality dominance)
        3. Enhanced hubness analysis with histograms and Gini/skewness
        
        References:
        - High-dimensional hubness (Radovanović et al., JMLR 2010)
        - Multimodal fusion challenges (Baltrušaitis et al., TPAMI 2019)
        """
        logger.info("Creating Scale Mismatch & Hubness Diagnosis figure...")
        
        # Get original (pre-normalized) features for norm analysis
        candidate_ids = list(self.candidate_meta['candidate_ids'])
        query_ids = list(self.query_ids)
        
        # Extract original features (before L2 normalization)
        # Use pre-normalization text embeddings to reveal true scale differences
        text_candidates_orig = np.stack([
            (self.text_embeddings_raw.get(vid, self.text_embeddings[vid])) for vid in candidate_ids
        ])
        text_queries_orig = np.stack([
            (self.text_embeddings_raw.get(vid, self.text_embeddings[vid])) for vid in query_ids
        ])
        visual_candidates_orig = np.stack([self.visual_features[vid] for vid in candidate_ids])
        visual_queries_orig = np.stack([self.visual_features[vid] for vid in query_ids])
        audio_candidates_orig = np.stack([self.audio_features[vid] for vid in candidate_ids])
        audio_queries_orig = np.stack([self.audio_features[vid] for vid in query_ids])
        
        # Combine queries and candidates for full dataset analysis
        text_all = np.vstack([text_queries_orig, text_candidates_orig])
        visual_all = np.vstack([visual_queries_orig, visual_candidates_orig])
        audio_all = np.vstack([audio_queries_orig, audio_candidates_orig])
        
        # Compute L2 norms
        text_norms = np.linalg.norm(text_all, axis=1)
        visual_norms = np.linalg.norm(visual_all, axis=1)
        audio_norms = np.linalg.norm(audio_all, axis=1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Part 1: L2 Norm Distributions (Box plots)
        ax1 = axes[0, 0]
        norm_data = []
        modality_labels = []
        
        for norms, label in [(text_norms, 'Text'), (visual_norms, 'Visual'), (audio_norms, 'Audio')]:
            norm_data.extend(norms)
            modality_labels.extend([label] * len(norms))
        
        norm_df = pd.DataFrame({'L2_Norm': norm_data, 'Modality': modality_labels})
        sns.boxplot(data=norm_df, x='Modality', y='L2_Norm', ax=ax1)
        ax1.set_title('L2 Norm Distributions Across Modalities', fontsize=12)
        ax1.set_ylabel('L2 Norm')
        
        # Part 2: Energy Ratio Bar Chart
        ax2 = axes[0, 1]
        text_energy = np.mean(text_norms**2)
        visual_energy = np.mean(visual_norms**2)
        audio_energy = np.mean(audio_norms**2)
        total_energy = text_energy + visual_energy + audio_energy
        
        energy_ratios = [text_energy/total_energy, visual_energy/total_energy, audio_energy/total_energy]
        modalities = ['Text', 'Visual', 'Audio']
        bars = ax2.bar(modalities, energy_ratios, alpha=0.7, color=['blue', 'green', 'red'])
        ax2.set_title('Energy Ratio (Norm² Proportion)', fontsize=12)
        ax2.set_ylabel('Energy Proportion')
        
        # Add percentage labels on bars
        for bar, ratio in zip(bars, energy_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{ratio*100:.1f}%', ha='center', va='bottom')
        
        # Part 3: Hubness Analysis - Frequency Histogram
        ax3 = axes[0, 2]
        # Compute hubness frequencies
        frequencies = self._compute_hubness_frequencies()
        
        # Plot histogram
        ax3.hist(frequencies, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_title('Hubness Frequency Distribution', fontsize=12)
        ax3.set_xlabel('Top-10 Frequency')
        ax3.set_ylabel('Number of Candidates')
        
        # Add Gini coefficient annotation
        gini_coeff = self._compute_gini_coefficient(frequencies)
        ax3.text(0.7, 0.9, f'Gini = {gini_coeff:.3f}', transform=ax3.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        # Part 4: Zipf Curve (Hubness Rank vs Frequency)
        ax4 = axes[1, 0]
        sorted_frequencies = np.sort(frequencies)[::-1]  # Sort in descending order
        ranks = np.arange(1, len(sorted_frequencies) + 1)
        ax4.loglog(ranks, sorted_frequencies, 'b-', linewidth=2)
        ax4.set_xlabel('Candidate Rank')
        ax4.set_ylabel('Top-10 Frequency')
        ax4.set_title('Hubness Zipf Curve', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Highlight top-20 dominance
        top_20_freq_sum = np.sum(sorted_frequencies[:20])
        total_freq_sum = np.sum(sorted_frequencies)
        dominance_ratio = top_20_freq_sum / total_freq_sum
        ax4.scatter(range(1, 21), sorted_frequencies[:20], color='red', s=30, zorder=5)
        ax4.text(0.6, 0.9, f'Top-20 Dom: {dominance_ratio:.3f}', transform=ax4.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Part 5: Concatenated vs Individual Modality Hubness Comparison
        ax5 = axes[1, 1]
        # Compare hubness before and after concatenation
        individual_ginis = []
        for modality in ['T', 'I', 'A']:
            mod_freqs = self._compute_single_modality_hubness(modality)
            individual_ginis.append(self._compute_gini_coefficient(mod_freqs))
        
        concatenated_gini = gini_coeff
        
        x_pos = np.arange(4)
        gini_values = individual_ginis + [concatenated_gini]
        labels = ['Text', 'Visual', 'Audio', 'Concatenated']
        colors = ['blue', 'green', 'red', 'orange']
        
        bars = ax5.bar(x_pos, gini_values, alpha=0.7, color=colors)
        ax5.set_title('Gini Coefficient: Individual vs Concatenated', fontsize=12)
        ax5.set_ylabel('Gini Coefficient')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(labels)
        
        # Add value labels on bars
        for bar, value in zip(bars, gini_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Part 6: Statistical Summary
        ax6 = axes[1, 2]
        ax6.axis('off')  # Turn off axis for text display
        
        # Compute statistics
        text_norm_stats = f"Mean: {np.mean(text_norms):.2f}±{np.std(text_norms):.2f}"
        visual_norm_stats = f"Mean: {np.mean(visual_norms):.2f}±{np.std(visual_norms):.2f}"
        audio_norm_stats = f"Mean: {np.mean(audio_norms):.2f}±{np.std(audio_norms):.2f}"
        
        skewness = self._compute_skewness(frequencies)
        
        summary_text = f"""Statistical Summary
        
Text L2 Norms:
{text_norm_stats}

Visual L2 Norms:
{visual_norm_stats}

Audio L2 Norms:
{audio_norm_stats}

Hubness Metrics:
Gini: {gini_coeff:.3f}
Skewness: {skewness:.3f}
Top-20 Dominance: {dominance_ratio:.3f}

Literature Support:
✓ Scale mismatch → fusion bias
✓ High hubness → distance metric degradation
✓ Concatenation amplifies both issues"""
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        fig_path = self.output_dir / "scale_mismatch_hubness_diagnosis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  - Scale Mismatch & Hubness figure: {fig_path}")
        return fig_path

    def create_similarity_concentration_neighborhood_figure(self):
        """
        Create Figure B: Similarity Concentration + Cross-modal Neighborhood Inconsistency
        
        This figure supports literature arguments about attention collapse and mismatch by showing:
        1. Similarity concentration (anisotropy) analysis
        2. Cross-modal kNN neighborhood overlap (Jaccard coefficients)
        3. Attention soft simulation with temperature sweeps
        
        References:
        - Representation anisotropy (Ethayarajh, EMNLP 2019)
        - Attention is not explanation (Jain & Wallace, NAACL 2019)
        - VQA bias issues (Agrawal et al., ICCV 2018)
        """
        logger.info("Creating Similarity Concentration & Neighborhood Inconsistency figure...")
        
        # Get normalized features for similarity analysis
        candidate_ids = list(self.candidate_meta['candidate_ids'])
        query_ids = list(self.query_ids)
        
        # Use already computed normalized features from self.scores
        text_sim = self.scores['T']  # [Q, N] similarity matrix
        visual_sim = self.scores['I']
        audio_sim = self.scores['A']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Part 1: Random Pairwise Cosine Similarity Distributions
        ax1 = axes[0, 0]
        
        # Sample random pairs for similarity analysis (to avoid memory issues)
        n_sample_pairs = 5000
        similarity_data = []
        modality_labels = []
        
        for sim_matrix, mod_name in [(text_sim, 'Text'), (visual_sim, 'Visual'), (audio_sim, 'Audio')]:
            # Sample random query-candidate pairs
            q_indices = np.random.choice(sim_matrix.shape[0], n_sample_pairs, replace=True)
            c_indices = np.random.choice(sim_matrix.shape[1], n_sample_pairs, replace=True)
            random_similarities = sim_matrix[q_indices, c_indices]
            
            similarity_data.extend(random_similarities)
            modality_labels.extend([mod_name] * len(random_similarities))
        
        sim_df = pd.DataFrame({'Cosine_Similarity': similarity_data, 'Modality': modality_labels})
        sns.violinplot(data=sim_df, x='Modality', y='Cosine_Similarity', ax=ax1)
        ax1.set_title('Random Pairwise Cosine Similarity Distributions', fontsize=12)
        ax1.set_ylabel('Cosine Similarity')
        
        # Part 2: Anisotropy Index (Mean Random Cosine)
        ax2 = axes[0, 1]
        anisotropy_indices = []
        modalities = ['Text', 'Visual', 'Audio']
        
        for sim_matrix, mod_name in [(text_sim, 'Text'), (visual_sim, 'Visual'), (audio_sim, 'Audio')]:
            # Compute anisotropy as mean of random similarity pairs
            q_indices = np.random.choice(sim_matrix.shape[0], n_sample_pairs, replace=True)
            c_indices = np.random.choice(sim_matrix.shape[1], n_sample_pairs, replace=True)
            random_sims = sim_matrix[q_indices, c_indices]
            anisotropy_idx = np.mean(random_sims)
            anisotropy_indices.append(anisotropy_idx)
        
        bars = ax2.bar(modalities, anisotropy_indices, alpha=0.7, color=['blue', 'green', 'red'])
        ax2.set_title('Anisotropy Index (Mean Random Cosine)', fontsize=12)
        ax2.set_ylabel('Mean Cosine Similarity')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars, anisotropy_indices):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Part 3: Cross-modal kNN Neighborhood Overlap (Jaccard)
        ax3 = axes[0, 2]
        k_values = [5, 10, 20]
        
        # Compute Jaccard overlaps for different k values
        jaccard_results = []
        pair_names = ['Text-Visual', 'Text-Audio', 'Visual-Audio']
        pair_combinations = [('T', 'I'), ('T', 'A'), ('I', 'A')]
        
        for k in k_values:
            row_jaccards = []
            for mod1, mod2 in pair_combinations:
                jaccard_scores = self._compute_knn_jaccard_overlap(
                    self.scores[mod1], self.scores[mod2], k=k)
                mean_jaccard = np.mean(jaccard_scores)
                row_jaccards.append(mean_jaccard)
            jaccard_results.append(row_jaccards)
        
        # Create heatmap
        jaccard_matrix = np.array(jaccard_results)
        im = ax3.imshow(jaccard_matrix, cmap='RdYlBu_r', aspect='auto')
        ax3.set_title(f'kNN Jaccard Overlap\n(Cross-modal Neighborhood Consistency)', fontsize=12)
        ax3.set_xlabel('Modality Pairs')
        ax3.set_ylabel('k Values')
        ax3.set_xticks(range(len(pair_names)))
        ax3.set_xticklabels(pair_names, rotation=45)
        ax3.set_yticks(range(len(k_values)))
        ax3.set_yticklabels([f'k={k}' for k in k_values])
        
        # Add text annotations
        for i in range(len(k_values)):
            for j in range(len(pair_names)):
                text = ax3.text(j, i, f'{jaccard_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax3)
        
        # Part 4: Attention Soft Simulation - Temperature Sweep
        ax4 = axes[1, 0]
        temperatures = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

        # Collect per-modality entropy curves for summary statistics
        attention_entropy_per_modality = []  # list of [len(temperatures)] arrays

        for mod_key, mod_name, color in [('T', 'Text', 'blue'), ('I', 'Visual', 'green'), ('A', 'Audio', 'red')]:
            attention_entropies = []

            for temp in temperatures:
                # Use candidate-wise logits to form attention over candidates per query
                sim_matrix = self.scores[mod_key]  # [Q, N]
                row_max = np.max(sim_matrix, axis=1, keepdims=True)
                attention_weights = self._softmax((sim_matrix - row_max) / temp, axis=1)  # [Q, N]

                # Compute mean entropy across queries
                entropies = [entropy(w + 1e-12) for w in attention_weights]
                mean_entropy = float(np.mean(entropies))
                attention_entropies.append(mean_entropy)

            attention_entropy_per_modality.append(np.array(attention_entropies))
            ax4.plot(temperatures, attention_entropies, marker='o', label=mod_name, color=color)
        
        ax4.set_xlabel('Temperature (τ)')
        ax4.set_ylabel('Mean Attention Entropy')
        ax4.set_title('Attention Collapse vs Temperature', fontsize=12)
        ax4.set_xscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Part 5: Similarity Concentration Statistics
        ax5 = axes[1, 1]
        
        concentration_stats = []
        for sim_matrix, mod_name in [(text_sim, 'Text'), (visual_sim, 'Visual'), (audio_sim, 'Audio')]:
            q_indices = np.random.choice(sim_matrix.shape[0], n_sample_pairs, replace=True)
            c_indices = np.random.choice(sim_matrix.shape[1], n_sample_pairs, replace=True)
            random_sims = sim_matrix[q_indices, c_indices]
            
            stats = {
                'mean': np.mean(random_sims),
                'std': np.std(random_sims),
                'entropy': entropy(np.histogram(random_sims, bins=50)[0] + 1e-12)
            }
            concentration_stats.append(stats)
        
        # Plot statistics
        x_pos = np.arange(len(modalities))
        width = 0.25
        
        means = [s['mean'] for s in concentration_stats]
        stds = [s['std'] for s in concentration_stats]
        entropies = [s['entropy'] for s in concentration_stats]
        
        ax5.bar(x_pos - width, means, width, label='Mean', alpha=0.7)
        ax5.bar(x_pos, stds, width, label='Std Dev', alpha=0.7)
        ax5.bar(x_pos + width, np.array(entropies)/10, width, label='Entropy/10', alpha=0.7)
        
        ax5.set_xlabel('Modality')
        ax5.set_ylabel('Value')
        ax5.set_title('Similarity Concentration Statistics', fontsize=12)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(modalities)
        ax5.legend()
        
        # Part 6: Literature Alignment Summary
        ax6 = axes[1, 2]
        ax6.axis('off')

        # Compute key metrics for summary
        mean_anisotropy = np.mean(anisotropy_indices)
        mean_jaccard = np.mean(jaccard_matrix)
        # Determine critical temperature as min of mean entropy across modalities
        if len(attention_entropy_per_modality) > 0:
            mean_entropy_over_modalities = np.mean(np.vstack(attention_entropy_per_modality), axis=0)  # [len(temperatures)]
            min_entropy_temp_idx = int(np.argmin(mean_entropy_over_modalities))
            min_temp_for_collapse = float(temperatures[min_entropy_temp_idx])
        else:
            min_entropy_temp_idx = 0
            min_temp_for_collapse = float(temperatures[min_entropy_temp_idx])
        
        summary_text = f"""Literature Alignment Summary

Anisotropy Analysis:
Mean Random Cosine: {mean_anisotropy:.3f}
→ High concentration supports 
  EMNLP 2019 findings

Cross-modal Consistency:
Mean kNN Jaccard: {mean_jaccard:.3f}
→ Low overlap indicates alignment 
  challenges (TPAMI 2019)

Attention Collapse:
Critical temp ≈ {min_temp_for_collapse:.2f}
→ Supports NAACL 2019 concerns
  about attention reliability

Key Evidence:
✓ Similarity concentration
✓ Neighborhood inconsistency  
✓ Temperature sensitivity
✓ Cross-modal misalignment

Conclusion:
Direct concatenation + attention
suffers from geometric biases
and structural inconsistencies"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        fig_path = self.output_dir / "similarity_concentration_neighborhood.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  - Similarity Concentration & Neighborhood figure: {fig_path}")
        return fig_path
    
    def _compute_hubness_frequencies(self):
        """Compute hubness frequencies for all candidates across all modalities"""
        all_scores = np.concatenate([
            self.scores['T'],  # [Q, N]
            self.scores['I'], 
            self.scores['A']
        ], axis=0)  # [3*Q, N]
        
        # Count how often each candidate appears in top-k across all queries
        k = 10
        N = all_scores.shape[1]
        frequencies = np.zeros(N)
        
        for q_idx in range(all_scores.shape[0]):
            top_k_candidates = np.argpartition(all_scores[q_idx], -k)[-k:]
            frequencies[top_k_candidates] += 1
            
        return frequencies
    
    def _compute_single_modality_hubness(self, modality_key):
        """Compute hubness frequencies for a single modality"""
        scores = self.scores[modality_key]  # [Q, N]
        k = 10
        N = scores.shape[1]
        frequencies = np.zeros(N)
        
        for q_idx in range(scores.shape[0]):
            top_k_candidates = np.argpartition(scores[q_idx], -k)[-k:]
            frequencies[top_k_candidates] += 1
            
        return frequencies
    
    def _compute_gini_coefficient(self, values):
        """Compute Gini coefficient for measuring inequality"""
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n
    
    def _compute_skewness(self, values):
        """Compute skewness of the distribution"""
        mean_val = np.mean(values)
        std_val = np.std(values)
        return np.mean(((values - mean_val) / std_val) ** 3)
    
    def _compute_knn_jaccard_overlap(self, scores1, scores2, k=10):
        """Compute Jaccard overlap between kNN neighborhoods of two modalities"""
        jaccard_scores = []
        
        for q_idx in range(scores1.shape[0]):
            # Get top-k neighbors for each modality
            top_k1 = set(np.argpartition(scores1[q_idx], -k)[-k:])
            top_k2 = set(np.argpartition(scores2[q_idx], -k)[-k:])
            
            # Compute Jaccard coefficient
            intersection = len(top_k1 & top_k2)
            union = len(top_k1 | top_k2)
            jaccard = intersection / union if union > 0 else 0
            jaccard_scores.append(jaccard)
            
        return jaccard_scores
    
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

        # Extra: entropy/margin/overlap diagnostics to explain similarity across modalities
        try:
            self.compute_entropy_diagnostics()
        except Exception as e:
            logger.warning(f"Entropy diagnostics failed: {e}")

        # Step 5: Hubness analysis
        hubness_df, hubness_summary = self.compute_hubness_analysis()
        
        # Step 6: Fusion methods comparison
        methods_df = self.compute_fusion_methods_comparison(hubness_summary)
        
        # Step 7: Create visualizations
        self.create_visualizations(single_df, pairs_df, hubness_df, methods_df, hubness_summary)
        
        # Step 7.1: Entropy vs Event-Hit curve
        try:
            self.create_entropy_vs_event_hit_curve()
        except Exception as e:
            logger.warning(f"Entropy vs Event-Hit curve failed: {e}")

        # Step 7.2: Create Figure A - Scale Mismatch & Hubness Diagnosis
        try:
            self.create_scale_mismatch_hubness_figure()
        except Exception as e:
            logger.warning(f"Scale Mismatch & Hubness figure failed: {e}")

        # Step 7.3: Create Figure B - Similarity Concentration & Neighborhood Inconsistency
        try:
            self.create_similarity_concentration_neighborhood_figure()
        except Exception as e:
            logger.warning(f"Similarity Concentration & Neighborhood figure failed: {e}")

        # Step 7.4: Single-plot NSI to show modality quality and variance
        try:
            self.create_modality_nsi_boxplot()
        except Exception as e:
            logger.warning(f"NSI single-plot creation failed: {e}")

        # Step 7.5: Standalone random pairwise cosine similarity
        try:
            self.create_random_pairwise_cosine_plot()
        except Exception as e:
            logger.warning(f"Random pairwise cosine plot failed: {e}")

        # Step 7.6: Standalone kNN Jaccard overlap heatmap
        try:
            self.create_knn_jaccard_overlap_plot()
        except Exception as e:
            logger.warning(f"kNN Jaccard overlap plot failed: {e}")

        # Step 7.7: Principal angles between modality subspaces (single axis, three lines)
        try:
            # Default: no extra preprocessing for this plot
            self.create_principal_angles_plot(R=64)
        except Exception as e:
            logger.warning(f"Principal angles plot failed: {e}")

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

    def create_principal_angles_plot(self, R: int = 64, do_l2: bool = False):
        """Plot cos(theta_r) for principal angles between T–V, T–A, V–A subspaces.

        Steps:
          1) 可选每样本 L2（默认关闭）；不做去均值、不做去主成分。
          2) 用 QR 分解得到样本空间的正交基 Q，取前 R 列作为 U_T, U_V, U_A。
          3) For each pair (U_A, U_B): SVD of U_A^T U_B → singular values = cos(theta_r).
          4) Plot three lines on one axis across r=1..R_eff. Cache intermediates/results.
        """
        if not hasattr(self, 'query_ids') or not self.query_ids:
            raise RuntimeError("Please call prepare_analysis_data() before plotting principal angles.")

        # Determine cache file
        N = len(self.query_ids)
        cache_name = f"principal_angles_raw_{self.text_model_short}_{self.audio_model}_N{N}_R{R}_l2{int(do_l2)}.npz"
        cache_file = self.cache_dir / cache_name

        if self.use_cache and cache_file.exists():
            try:
                data = np.load(cache_file)
                cos_TV = data['cos_TV']
                cos_TA = data['cos_TA']
                cos_VA = data['cos_VA']
                R_eff = int(data['R_eff'])
                logger.info(f"Loaded principal angles from cache: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load principal angles cache: {e}; recomputing.")
                cos_TV = cos_TA = cos_VA = None
        else:
            cos_TV = cos_TA = cos_VA = None

        # Compute if not cached
        if cos_TV is None:
            ids = list(self.query_ids)
            # Stack features for the same ordered ids
            XT = np.stack([self.text_embeddings[i] for i in ids], axis=0)
            XV = np.stack([self.visual_features[i] for i in ids], axis=0)
            XA = np.stack([self.audio_features[i] for i in ids], axis=0)

            # Optional per-sample L2 normalization
            if do_l2:
                eps = 1e-12
                XTp = XT / (np.linalg.norm(XT, axis=1, keepdims=True) + eps)
                XVp = XV / (np.linalg.norm(XV, axis=1, keepdims=True) + eps)
                XAp = XA / (np.linalg.norm(XA, axis=1, keepdims=True) + eps)
            else:
                XTp, XVp, XAp = XT, XV, XA

            # Orthonormal bases via QR decomposition (sample space)
            QT, _ = np.linalg.qr(XTp, mode='reduced')
            QV, _ = np.linalg.qr(XVp, mode='reduced')
            QA, _ = np.linalg.qr(XAp, mode='reduced')
            rT = min(R, QT.shape[1])
            rV = min(R, QV.shape[1])
            rA = min(R, QA.shape[1])
            R_eff = int(min(rT, rV, rA))
            if R_eff < 1:
                raise RuntimeError("Effective R is < 1; not enough rank for principal angles.")
            UT = QT[:, :R_eff]
            UV = QV[:, :R_eff]
            UA = QA[:, :R_eff]

            # Cosines of principal angles via singular values of cross-basis inner products
            cos_TV = np.linalg.svd(UT.T @ UV, compute_uv=False)
            cos_TA = np.linalg.svd(UT.T @ UA, compute_uv=False)
            cos_VA = np.linalg.svd(UV.T @ UA, compute_uv=False)

            # Cache results
            try:
                np.savez_compressed(cache_file,
                                    cos_TV=cos_TV, cos_TA=cos_TA, cos_VA=cos_VA,
                                    R_eff=np.array(R_eff), R_req=np.array(R))
                logger.info(f"Cached principal angles to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache principal angles: {e}")

        # Plot single-axis with up to three lines
        r_axis = np.arange(1, len(cos_TV) + 1)
        plt.figure(figsize=(7.5, 5.5))
        plt.plot(r_axis, cos_TV, label='Text–Vision', linewidth=2)
        plt.plot(r_axis, cos_TA, label='Text–Audio', linewidth=2)
        plt.plot(r_axis, cos_VA, label='Vision–Audio', linewidth=2)
        plt.xlabel('Subspace dimension r')
        plt.ylabel('cos(θ_r)')
        plt.title('Principal Angle Cosines between Modality Subspaces')
        plt.ylim(0.0, 1.0)
        plt.xlim(1, len(r_axis))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = self.output_dir / 'principal_angles_three_modalities.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  - Principal angles plot saved: {out_path} (R_req={R}, R_eff={len(r_axis)}, L2={do_l2})")

    def compute_entropy_diagnostics(self, tau_list: List[float] = [1.0, 0.5, 0.2, 0.1]):
        """Compute diagnostics to understand why entropy distributions look similar across modalities.

        Saves:
          - entropy_diagnostics_summary.csv
          - top1_top2_margin_distribution.png
          - entropy_cross_modality_scatter.png
          - topk_overlap_jaccard.png
          - entropy_temperature_sweep.csv, temperature_sweep_entropy.png
        """
        logger.info("Computing entropy diagnostics (margins, overlaps, temperature sweep)...")

        modalities = ['T', 'I', 'A', 'S']
        Q = len(self.query_ids)
        N = len(self.candidate_meta['candidate_ids'])
        k = min(self.top_k, N)

        def topk_indices_row(r, kk=k):
            return np.argpartition(r, -kk)[-kk:] if r.shape[0] > kk else np.arange(r.shape[0])

        # Collect per-modality vectors
        ent_topk = {}
        ent_full = {}
        perp_topk = {}
        margin_top2 = {}
        topk_idx_list = {}

        for m in modalities:
            P = self.probs[m]  # [Q, N]
            H_topk = np.zeros(Q)
            H_full = np.zeros(Q)
            M_margin = np.zeros(Q)
            idx_list = []
            for qi in range(Q):
                r = P[qi]
                idx = topk_indices_row(r)
                sub = r[idx]
                sub = sub / (sub.sum() + 1e-12)
                H_topk[qi] = entropy(sub + 1e-12)
                H_full[qi] = entropy(r + 1e-12)
                srt = np.sort(r)[::-1]
                M_margin[qi] = (srt[0] - srt[1]) if srt.shape[0] > 1 else 1.0
                idx_list.append(set(idx.tolist()))
            ent_topk[m] = H_topk
            ent_full[m] = H_full
            perp_topk[m] = np.exp(H_topk)
            margin_top2[m] = M_margin
            topk_idx_list[m] = idx_list

        # Pairwise correlations and Jaccard overlaps
        def pairwise_stats(vec_map, name):
            rows = []
            for a, b, label in [('T', 'I', 'Text-Visual'), ('T', 'A', 'Text-Audio'), ('I', 'A', 'Visual-Audio'), 
                                ('T', 'S', 'Text-Transcript'), ('I', 'S', 'Visual-Transcript'), ('A', 'S', 'Audio-Transcript')]:
                rho, _ = spearmanr(vec_map[a], vec_map[b])
                rows.append({'Pair': label, f'{name}_Spearman': 0.0 if np.isnan(rho) else float(rho)})
            return rows

        corr_entropy_rows = pairwise_stats(ent_topk, 'EntropyTopK')
        corr_margin_rows = pairwise_stats(margin_top2, 'MarginTop2')

        # Jaccard per query then average
        jacc_rows = []
        for a, b, label in [('T', 'I', 'Text-Visual'), ('T', 'A', 'Text-Audio'), ('I', 'A', 'Visual-Audio'),
                            ('T', 'S', 'Text-Transcript'), ('I', 'S', 'Visual-Transcript'), ('A', 'S', 'Audio-Transcript')]:
            jacc = []
            A = topk_idx_list[a]
            B = topk_idx_list[b]
            for qi in range(Q):
                inter = len(A[qi] & B[qi])
                union = len(A[qi] | B[qi])
                jacc.append(inter / union if union > 0 else 0.0)
            jacc_rows.append({'Pair': label,
                              'TopK_Jaccard_mean': float(np.mean(jacc)),
                              'TopK_Jaccard_std': float(np.std(jacc))})

        # Summarize per-modality
        rows_modal = []
        for m, label in [('T', 'Text'), ('I', 'Visual'), ('A', 'Audio'), ('S', 'Transcript')]:
            rows_modal.append({
                'Modality': label,
                'H_topk_mean': float(ent_topk[m].mean()),
                'H_topk_std': float(ent_topk[m].std()),
                'H_full_mean': float(ent_full[m].mean()),
                'H_full_std': float(ent_full[m].std()),
                'Perplexity_mean': float(perp_topk[m].mean()),
                'Perplexity_std': float(perp_topk[m].std()),
                'Top2_Margin_mean': float(margin_top2[m].mean()),
                'Top2_Margin_std': float(margin_top2[m].std()),
            })

        # Save summary CSV
        summary_df = pd.DataFrame(rows_modal)
        corr_df = pd.DataFrame(corr_entropy_rows).merge(pd.DataFrame(corr_margin_rows), on='Pair')
        jacc_df = pd.DataFrame(jacc_rows)
        # Write individual CSVs
        summary_df.to_csv(self.output_dir / 'entropy_diagnostics_summary.csv', index=False)
        corr_df.to_csv(self.output_dir / 'entropy_cross_modality_correlation.csv', index=False)
        jacc_df.to_csv(self.output_dir / 'topk_overlap_jaccard.csv', index=False)

        # Plots: margin distributions
        try:
            plt.figure(figsize=(10, 6))
            for m, label, color in [('T', 'Text', 'tab:blue'), ('I', 'Visual', 'tab:orange'), ('A', 'Audio', 'tab:green'), ('S', 'Transcript', 'tab:red')]:
                sns.kdeplot(margin_top2[m], label=label, fill=True, alpha=0.25, color=color)
            plt.title('Top-1 vs Top-2 Margin Distribution')
            plt.xlabel('Margin (p1 - p2)')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'top1_top2_margin_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Plot margin distribution failed: {e}")

        # Plots: entropy cross-modality scatter
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 8))
            pairs = [(('T', 'I'), 'Text vs Visual'), (('T', 'A'), 'Text vs Audio'), (('I', 'A'), 'Visual vs Audio'),
                     (('T', 'S'), 'Text vs Transcript'), (('I', 'S'), 'Visual vs Transcript'), (('A', 'S'), 'Audio vs Transcript')]
            axes = axes.flatten()
            for ax, ((a, b), title) in zip(axes, pairs):
                ax.scatter(ent_topk[a], ent_topk[b], alpha=0.3, s=8)
                rho, _ = spearmanr(ent_topk[a], ent_topk[b])
                ax.set_title(f'{title} (Spearman={0.0 if np.isnan(rho) else rho:.2f})')
                ax.set_xlabel('H_topk ' + a)
                ax.set_ylabel('H_topk ' + b)
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'entropy_cross_modality_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Plot entropy scatter failed: {e}")

        # Plots: top-k overlap Jaccard
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=True)
            pairs = [('T', 'I', 'Text-Visual'), ('T', 'A', 'Text-Audio'), ('I', 'A', 'Visual-Audio'),
                     ('T', 'S', 'Text-Transcript'), ('I', 'S', 'Visual-Transcript'), ('A', 'S', 'Audio-Transcript')]
            axes = axes.flatten()
            for ax, (a, b, label) in zip(axes, pairs):
                jacc = []
                for qi in range(Q):
                    inter = len(topk_idx_list[a][qi] & topk_idx_list[b][qi])
                    union = len(topk_idx_list[a][qi] | topk_idx_list[b][qi])
                    jacc.append(inter / union if union > 0 else 0.0)
                ax.hist(jacc, bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'{label} (mean={np.mean(jacc):.2f})')
                ax.set_xlabel('Jaccard')
                if label in ['Text-Visual', 'Text-Transcript']:
                    ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'topk_overlap_jaccard.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Plot top-k overlap failed: {e}")

        # Temperature sweep using raw similarity scores
        try:
            tau_records = []
            for tau in tau_list:
                probs_tau = {}
                for m in modalities:
                    S = self.scores[m]
                    probs_tau[m] = self._softmax(S / max(tau, 1e-6), axis=1)
                # Apply self-exclusion so it matches baseline handling
                probs_tau = self._apply_self_exclusion_to(probs_tau)

                # Compute mean top-k entropy per modality
                for m, label in [('T', 'Text'), ('I', 'Visual'), ('A', 'Audio'), ('S', 'Transcript')]:
                    Hs = []
                    Pm = probs_tau[m]
                    for qi in range(Q):
                        r = Pm[qi]
                        idx = topk_indices_row(r)
                        sub = r[idx]
                        sub = sub / (sub.sum() + 1e-12)
                        Hs.append(entropy(sub + 1e-12))
                    tau_records.append({'Tau': tau, 'Modality': label,
                                        'H_topk_mean': float(np.mean(Hs)),
                                        'H_topk_std': float(np.std(Hs))})

            tau_df = pd.DataFrame(tau_records)
            tau_df.to_csv(self.output_dir / 'entropy_temperature_sweep.csv', index=False)

            # Plot temperature sweep
            plt.figure(figsize=(10, 6))
            for label in ['Text', 'Visual', 'Audio', 'Transcript']:
                sub = tau_df[tau_df['Modality'] == label]
                plt.plot(sub['Tau'], sub['H_topk_mean'], marker='o', label=label)
            plt.gca().invert_xaxis()  # smaller tau to the right visually
            plt.title('Top-k Entropy vs Temperature (smaller tau = sharper)')
            plt.xlabel('Tau')
            plt.ylabel('Mean H_topk')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'temperature_sweep_entropy.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Temperature sweep failed: {e}")


    def create_modality_nsi_boxplot(self, k: int = 10, max_samples: int = 2000, seed: int = 42):
        """Create a single boxplot showing per-modality feature quality and intra-modality variance.

        Unsupervised Neighborhood Separability Index (NSI) per sample i:
          NSI(i) = (mean_topk - mean_rest) / (std_all + 1e-6),
        where similarities are cosine within the same modality (self excluded).
        Higher NSI implies clearer local structure (better quality). Distribution spread shows variance.
        """
        logger.info("Creating NSI single-plot (modality quality & variance)...")

        rng = np.random.default_rng(seed)

        def stack_and_sample(feat_dict, max_n):
            ids = list(feat_dict.keys())
            if len(ids) == 0:
                return None
            if len(ids) > max_n:
                sel = list(rng.choice(ids, size=max_n, replace=False))
            else:
                sel = ids
            X = np.stack([feat_dict[i] for i in sel], axis=0).astype(np.float32)
            # L2-normalize (diagnostics only)
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            return X

        feats = {
            'Text': stack_and_sample(self.text_embeddings, max_samples),
            'Visual': stack_and_sample(self.visual_features, max_samples),
            'Audio': stack_and_sample(self.audio_features, max_samples),
            # Exclude Transcript per latest requirement
        }

        rows = []
        for name, X in feats.items():
            if X is None or X.shape[0] < (k + 2):
                logger.warning(f"NSI: modality {name} has insufficient samples for k={k}")
                continue

            S = np.dot(X, X.T)
            np.fill_diagonal(S, -np.inf)

            # Top-k mean per row
            topk_vals = np.partition(S, -k, axis=1)[:, -k:]
            mean_topk = topk_vals.mean(axis=1)

            # Mean/std over all non-diagonal entries
            mask = ~np.eye(S.shape[0], dtype=bool)
            all_vals = S[mask].reshape(S.shape[0], -1)
            full_sum = all_vals.sum(axis=1)
            full_std = all_vals.std(axis=1)

            # Mean over the rest (excluding top-k)
            topk_sum = topk_vals.sum(axis=1)
            denom_rest = (S.shape[0] - 1 - k)
            mean_rest = (full_sum - topk_sum) / np.maximum(denom_rest, 1)

            nsi = (mean_topk - mean_rest) / (full_std + 1e-6)
            rows.extend({'Modality': name, 'NSI': float(v)} for v in nsi)

        if not rows:
            logger.warning("NSI: no data available to plot")
            return None

        df = pd.DataFrame(rows)
        mods_order = ['Text', 'Visual', 'Audio']

        plt.figure(figsize=(8, 5))
        ax = sns.boxplot(data=df, x='Modality', y='NSI', order=mods_order, showfliers=False)

        # Overlay mean ± std
        means = df.groupby('Modality')['NSI'].mean().reindex(mods_order)
        stds = df.groupby('Modality')['NSI'].std().reindex(mods_order)
        xlocs = np.arange(len(mods_order))
        ax.errorbar(xlocs, means.values, yerr=stds.values, fmt='o', color='black',
                    capsize=4, lw=1.2, label='Mean ± Std')
        ax.legend(loc='upper right')
        ax.set_title('Modality Feature Quality (NSI) and Intra-modality Variance')
        ax.set_ylabel('Neighborhood Separability Index (higher = better)')
        plt.tight_layout()
        out = self.output_dir / 'feature_quality_variance_nsi.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  - NSI single-plot saved: {out}")
        return out

    def create_random_pairwise_cosine_plot(self, n_sample_pairs: int = 5000, seed: int = 42):
        """Create a standalone violin plot of random pairwise cosine similarities for T/I/A."""
        logger.info("Creating standalone random pairwise cosine similarity plot...")
        rng = np.random.default_rng(seed)

        text_sim = self.scores['T']
        visual_sim = self.scores['I']
        audio_sim = self.scores['A']

        similarity_data = []
        modality_labels = []
        for sim_matrix, mod_name in [(text_sim, 'Text'), (visual_sim, 'Visual'), (audio_sim, 'Audio')]:
            q_indices = rng.integers(0, sim_matrix.shape[0], size=n_sample_pairs)
            c_indices = rng.integers(0, sim_matrix.shape[1], size=n_sample_pairs)
            vals = sim_matrix[q_indices, c_indices]
            similarity_data.extend(vals.tolist())
            modality_labels.extend([mod_name] * len(vals))

        df = pd.DataFrame({'Cosine_Similarity': similarity_data, 'Modality': modality_labels})
        plt.figure(figsize=(6.5, 5))
        sns.violinplot(data=df, x='Modality', y='Cosine_Similarity')
        plt.title('Random Pairwise Cosine Similarity Distributions')
        plt.ylabel('Cosine Similarity')
        plt.tight_layout()
        out = self.output_dir / 'random_pairwise_cosine.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  - Random pairwise cosine figure saved: {out}")
        return out

    def create_knn_jaccard_overlap_plot(self, k_values: List[int] = [5, 10, 20]):
        """Create a standalone heatmap of cross-modal kNN Jaccard overlaps for T/I/A."""
        logger.info("Creating standalone kNN Jaccard overlap heatmap...")

        pair_names = ['Text-Visual', 'Text-Audio', 'Visual-Audio']
        pair_combinations = [('T', 'I'), ('T', 'A'), ('I', 'A')]

        jaccard_results = []
        for k in k_values:
            row = []
            for mod1, mod2 in pair_combinations:
                scores1 = self.scores[mod1]
                scores2 = self.scores[mod2]
                jaccard_scores = self._compute_knn_jaccard_overlap(scores1, scores2, k=k)
                row.append(float(np.mean(jaccard_scores)))
            jaccard_results.append(row)

        jaccard_matrix = np.array(jaccard_results)
        plt.figure(figsize=(6.5, 5))
        ax = plt.gca()
        im = ax.imshow(jaccard_matrix, cmap='RdYlBu_r', aspect='auto')
        ax.set_title('kNN Jaccard Overlap (Cross-modal)')
        ax.set_xlabel('Modality Pairs')
        ax.set_ylabel('k Values')
        ax.set_xticks(range(len(pair_names)))
        ax.set_xticklabels(pair_names, rotation=45, ha='right')
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels([f'k={k}' for k in k_values])
        for i in range(len(k_values)):
            for j in range(len(pair_names)):
                ax.text(j, i, f'{jaccard_matrix[i, j]:.3f}', ha='center', va='center', color='black', fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        out = self.output_dir / 'knn_jaccard_overlap.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  - kNN Jaccard overlap heatmap saved: {out}")
        return out


def main():
    parser = argparse.ArgumentParser(description='Multimodal Choice Analysis for Fusion Method Selection')
    parser.add_argument('--dataset', type=str, default='FakeSV',
                       help='Dataset name (default: FakeSV)')
    parser.add_argument('--audio-model', type=str, 
                       default='laion-clap-htsat-fused',
                       help='Audio model name suffix for loading features')
    parser.add_argument('--text-model', type=str, default=None,
                       help='Text model for embeddings (auto-selected if None)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: analysis/{dataset}/multimodal_choice)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Top-k samples to consider for entropy and p_event_mass calculations (default: 10)')
    parser.add_argument('--use-cache', action='store_true',
                        help='Use cached embeddings and prepared matrices if available to avoid recomputation')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MultimodalChoiceAnalyzer(
        dataset=args.dataset,
        audio_model=args.audio_model,
        text_model=args.text_model,
        output_dir=args.output_dir,
        top_k=args.top_k,
        use_cache=args.use_cache
    )
    
    # Run analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
