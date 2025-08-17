#!/usr/bin/env python3
"""
Text similarity retrieval for ExMRD: For each test video, find most similar true/fake news from training set
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import logging
import sys

# Add src to path
sys.path.append('src')
from data.baseline_data import FakeSVDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextSimilarityRetrieval:
    def __init__(self, model_name: str = "OFA-Sys/chinese-clip-vit-large-patch14"):
        """
        Initialize text similarity retrieval system
        
        Args:
            model_name: Hugging Face model name for text embedding
        """
        self.model_name = model_name
        self.model_short_name = model_name.split("/")[-1]
        
        # Data paths
        self.entity_dir = Path("data/FakeSV/entity_claims")
        self.output_dir = Path("text_similarity_results")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initializing text similarity retrieval with model: {model_name}")
        
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
        
    def get_temporal_split_data(self):
        """Get temporal split data for train/test"""
        dataset = FakeSVDataset()
        
        # Get train and test video IDs from temporal split
        train_data = dataset._get_data('temporal', 'train')
        test_data = dataset._get_data('temporal', 'test')
        
        train_video_ids = set(train_data['video_id'].tolist())
        test_video_ids = set(test_data['video_id'].tolist())
        
        logger.info(f"Train split: {len(train_video_ids)} videos")
        logger.info(f"Test split: {len(test_video_ids)} videos")
        
        return train_video_ids, test_video_ids
    
    def prepare_data(self, train_video_ids: set, test_video_ids: set):
        """Prepare training and test data with entity claims"""
        # Create lookup for entity data
        entity_lookup = {}
        for item in self.true_data + self.fake_data:
            entity_lookup[item['video_id']] = item
        
        # Prepare training data (both true and fake)
        self.train_true_data = []
        self.train_fake_data = []
        
        for video_id in train_video_ids:
            if video_id in entity_lookup:
                item = entity_lookup[video_id]
                if item.get('annotation') == '真':
                    self.train_true_data.append(item)
                elif item.get('annotation') == '假':
                    self.train_fake_data.append(item)
        
        # Prepare test data
        self.test_data = []
        for video_id in test_video_ids:
            if video_id in entity_lookup:
                self.test_data.append(entity_lookup[video_id])
        
        logger.info(f"Training set: {len(self.train_true_data)} true, {len(self.train_fake_data)} fake")
        logger.info(f"Test set: {len(self.test_data)} videos")
        
    def concatenate_text_features(self, data_list: List[Dict]) -> List[str]:
        """Concatenate title + keywords + description + temporal_evolution"""
        texts = []
        for item in data_list:
            text_parts = []
            
            # Add each field if it exists
            for field in ['title', 'keywords', 'description', 'temporal_evolution']:
                value = item.get(field, '')
                if value and str(value).strip():
                    text_parts.append(str(value).strip())
            
            # Join with spaces and clean up
            combined_text = ' '.join(text_parts)
            combined_text = ' '.join(combined_text.split())  # Remove extra whitespace
            texts.append(combined_text)
            
        return texts
    
    def find_most_similar(self, test_embeddings: np.ndarray, train_embeddings: np.ndarray,
                         train_data: List[Dict]) -> List[Dict]:
        """Find most similar training video for each test video"""
        # Calculate cosine similarity
        similarities = cosine_similarity(test_embeddings, train_embeddings)
        
        results = []
        for i, sim_scores in enumerate(similarities):
            # Find index of most similar training video
            best_idx = np.argmax(sim_scores)
            best_score = sim_scores[best_idx]
            best_match = train_data[best_idx]
            
            # Get timestamp info
            video_id = best_match['video_id']
            timestamp_info = self.timestamp_lookup.get(video_id, {})
            
            results.append({
                'video_id': video_id,
                'similarity_score': float(best_score),
                'title': best_match.get('title', ''),
                'keywords': best_match.get('keywords', ''),
                'description': best_match.get('description', ''),
                'temporal_evolution': best_match.get('temporal_evolution', ''),
                'annotation': best_match.get('annotation', ''),
                'publish_time': timestamp_info.get('timestamp', 'N/A'),
                'raw_timestamp': timestamp_info.get('raw_timestamp', 0),
                'combined_text': self.concatenate_text_features([best_match])[0]
            })
            
        return results
    
    def run_retrieval(self):
        """Run the complete retrieval process"""
        logger.info("Starting text similarity retrieval...")
        
        # Load model
        self.load_model()
        
        # Load entity data
        self.load_entity_data()
        
        # Get temporal split
        train_video_ids, test_video_ids = self.get_temporal_split_data()
        
        # Prepare data
        self.prepare_data(train_video_ids, test_video_ids)
        
        # Concatenate text features
        logger.info("Preparing text features...")
        test_texts = self.concatenate_text_features(self.test_data)
        train_true_texts = self.concatenate_text_features(self.train_true_data)
        train_fake_texts = self.concatenate_text_features(self.train_fake_data)
        
        # Encode texts
        logger.info("Encoding test texts...")
        test_embeddings = self.encode_texts(test_texts)
        
        logger.info("Encoding training true texts...")
        train_true_embeddings = self.encode_texts(train_true_texts)
        
        logger.info("Encoding training fake texts...")
        train_fake_embeddings = self.encode_texts(train_fake_texts)
        
        # Find most similar true and fake news for each test video
        logger.info("Finding most similar true news...")
        most_similar_true = self.find_most_similar(test_embeddings, train_true_embeddings, self.train_true_data)
        
        logger.info("Finding most similar fake news...")
        most_similar_fake = self.find_most_similar(test_embeddings, train_fake_embeddings, self.train_fake_data)
        
        # Combine results
        results = []
        for i, test_item in enumerate(self.test_data):
            # Get timestamp for test video
            test_video_id = test_item['video_id']
            test_timestamp_info = self.timestamp_lookup.get(test_video_id, {})
            
            result = {
                'test_video': {
                    'video_id': test_video_id,
                    'annotation': test_item.get('annotation', ''),
                    'title': test_item.get('title', ''),
                    'keywords': test_item.get('keywords', ''),
                    'description': test_item.get('description', ''),
                    'temporal_evolution': test_item.get('temporal_evolution', ''),
                    'publish_time': test_timestamp_info.get('timestamp', 'N/A'),
                    'raw_timestamp': test_timestamp_info.get('raw_timestamp', 0),
                    'combined_text': test_texts[i]
                },
                'most_similar_true_news': most_similar_true[i],
                'most_similar_fake_news': most_similar_fake[i]
            }
            results.append(result)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: List[Dict]):
        """Save retrieval results"""
        # Save detailed results
        output_file = self.output_dir / f"text_similarity_retrieval_{self.model_short_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved detailed results to {output_file}")
        
        # Calculate and save summary statistics
        true_similarities = [r['most_similar_true_news']['similarity_score'] for r in results]
        fake_similarities = [r['most_similar_fake_news']['similarity_score'] for r in results]
        
        # Count by test video annotation
        true_test_count = sum(1 for r in results if r['test_video']['annotation'] == '真')
        fake_test_count = sum(1 for r in results if r['test_video']['annotation'] == '假')
        
        stats = {
            'model': self.model_name,
            'test_set_size': len(results),
            'test_true_count': true_test_count,
            'test_fake_count': fake_test_count,
            'train_true_count': len(self.train_true_data),
            'train_fake_count': len(self.train_fake_data),
            'similarity_stats': {
                'true_news_similarity': {
                    'mean': float(np.mean(true_similarities)),
                    'std': float(np.std(true_similarities)),
                    'min': float(np.min(true_similarities)),
                    'max': float(np.max(true_similarities))
                },
                'fake_news_similarity': {
                    'mean': float(np.mean(fake_similarities)),
                    'std': float(np.std(fake_similarities)),
                    'min': float(np.min(fake_similarities)),
                    'max': float(np.max(fake_similarities))
                }
            }
        }
        
        stats_file = self.output_dir / f"similarity_stats_{self.model_short_name}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved statistics to {stats_file}")
        
        # Create readable summary
        summary_file = self.output_dir / f"retrieval_summary_{self.model_short_name}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Text Similarity Retrieval Summary\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"=" * 50 + "\n\n")
            
            f.write(f"Dataset Statistics:\n")
            f.write(f"  Test set: {len(results)} videos ({true_test_count} true, {fake_test_count} fake)\n")
            f.write(f"  Training set: {len(self.train_true_data)} true, {len(self.train_fake_data)} fake\n\n")
            
            f.write(f"Similarity Statistics:\n")
            f.write(f"  Most similar true news:\n")
            f.write(f"    Mean: {np.mean(true_similarities):.4f}\n")
            f.write(f"    Std:  {np.std(true_similarities):.4f}\n")
            f.write(f"    Min:  {np.min(true_similarities):.4f}\n")
            f.write(f"    Max:  {np.max(true_similarities):.4f}\n\n")
            
            f.write(f"  Most similar fake news:\n")
            f.write(f"    Mean: {np.mean(fake_similarities):.4f}\n")
            f.write(f"    Std:  {np.std(fake_similarities):.4f}\n")
            f.write(f"    Min:  {np.min(fake_similarities):.4f}\n")
            f.write(f"    Max:  {np.max(fake_similarities):.4f}\n\n")
            
            f.write(f"Sample Results (first 5):\n")
            f.write(f"-" * 30 + "\n")
            for i, result in enumerate(results[:5]):
                f.write(f"\nTest Video {i+1} (ID: {result['test_video']['video_id']}, Label: {result['test_video']['annotation']}):\n")
                f.write(f"  Title: {result['test_video']['title'][:100]}...\n")
                f.write(f"  Publish Time: {result['test_video']['publish_time']}\n")
                f.write(f"  Most similar TRUE:  {result['most_similar_true_news']['similarity_score']:.4f} (ID: {result['most_similar_true_news']['video_id']}, Time: {result['most_similar_true_news']['publish_time']})\n")
                f.write(f"  Most similar FAKE:  {result['most_similar_fake_news']['similarity_score']:.4f} (ID: {result['most_similar_fake_news']['video_id']}, Time: {result['most_similar_fake_news']['publish_time']})\n")
        
        logger.info(f"Saved summary to {summary_file}")
        
        # Print summary to console
        logger.info(f"\n=== Retrieval Summary ===")
        logger.info(f"Test set: {len(results)} videos")
        logger.info(f"Avg similarity to true news: {np.mean(true_similarities):.4f}")
        logger.info(f"Avg similarity to fake news: {np.mean(fake_similarities):.4f}")


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="Text similarity retrieval for ExMRD")
    parser.add_argument("--model", type=str, default="OFA-Sys/chinese-clip-vit-large-patch14",
                       help="Hugging Face model name for embeddings")
    args = parser.parse_args()
    
    # Create retrieval system
    retrieval = TextSimilarityRetrieval(model_name=args.model)
    
    # Run retrieval
    results = retrieval.run_retrieval()
    
    logger.info("Text similarity retrieval completed!")


if __name__ == "__main__":
    main()