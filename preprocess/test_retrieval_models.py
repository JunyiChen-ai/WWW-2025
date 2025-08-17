#!/usr/bin/env python3
"""
Test different embedding models for retrieving similar true news for fake videos
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RetrievalTester:
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        """
        Initialize retrieval tester with specified model
        
        Args:
            model_name: Hugging Face model name for text embedding
        """
        self.model_name = model_name
        self.model_short_name = model_name.split("/")[-1]  # e.g., "bge-large-zh-v1.5"
        
        # Data paths
        self.data_dir = Path("data/FakeSV/entity_claims")
        self.output_dir = Path("data/FakeSV/entity_claims/retrieval_tests")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initializing retrieval tester with model: {model_name}")
        
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
                outputs = self.model(**inputs)
                
                # Pool the outputs (mean pooling)
                # For BGE models, we use [CLS] token representation
                if "bge" in self.model_name.lower():
                    embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                else:
                    # Mean pooling for other models
                    attention_mask = inputs['attention_mask']
                    embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
                    embeddings = embeddings / attention_mask.sum(dim=1, keepdim=True)
                
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        logger.info(f"Encoded {len(texts)} texts into shape {embeddings.shape}")
        return embeddings
    
    def load_data(self):
        """Load fake video descriptions and true video extractions"""
        # Load fake video descriptions
        fake_desc_file = self.data_dir / "fake_video_descriptions.jsonl"
        self.fake_descriptions = []
        with open(fake_desc_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.fake_descriptions.append(json.loads(line))
        logger.info(f"Loaded {len(self.fake_descriptions)} fake video descriptions")
        
        # Load true video extractions
        true_ext_file = self.data_dir / "video_extractions.jsonl"
        self.true_extractions = []
        with open(true_ext_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.true_extractions.append(json.loads(line))
        logger.info(f"Loaded {len(self.true_extractions)} true video extractions")
        
    def prepare_texts(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Prepare texts for encoding"""
        fake_texts = []
        fake_video_ids = []
        
        for desc in self.fake_descriptions:
            # Concatenate all text fields
            text = f"{desc.get('title', '')} {desc.get('keywords', '')} {desc.get('description', '')} {desc.get('temporal_evolution', '')}"
            fake_texts.append(text.strip())
            fake_video_ids.append(desc['video_id'])
        
        true_texts = []
        true_video_ids = []
        
        for ext in self.true_extractions:
            # Concatenate all relevant text fields
            text = f"{ext.get('title', '')} {ext.get('keywords', '')} {ext.get('description', '')} {ext.get('temporal_evolution', '')}"
            true_texts.append(text.strip())
            true_video_ids.append(ext['video_id'])
        
        return fake_texts, fake_video_ids, true_texts, true_video_ids
    
    def find_similar_videos(self, fake_embeddings: np.ndarray, true_embeddings: np.ndarray,
                           fake_video_ids: List[str], true_video_ids: List[str], 
                           top_k: int = 5) -> Dict:
        """Find top-k most similar true videos for each fake video"""
        logger.info(f"Computing similarities for {len(fake_video_ids)} fake videos...")
        
        # Calculate cosine similarity
        similarities = cosine_similarity(fake_embeddings, true_embeddings)
        
        results = {}
        for i, fake_vid in enumerate(fake_video_ids):
            # Get similarity scores for this fake video
            sim_scores = similarities[i]
            
            # Get indices of top-k most similar
            top_indices = np.argsort(sim_scores)[-top_k:][::-1]
            
            # Store results with similarity scores
            results[fake_vid] = [
                {
                    "video_id": true_video_ids[idx],
                    "similarity": float(sim_scores[idx])
                }
                for idx in top_indices
            ]
        
        return results
    
    def create_detailed_output(self, similarity_results: Dict) -> List[Dict]:
        """Create detailed output with full text information"""
        detailed_results = []
        
        # Create lookups
        fake_lookup = {d['video_id']: d for d in self.fake_descriptions}
        true_lookup = {e['video_id']: e for e in self.true_extractions}
        
        for fake_vid, similar_videos in similarity_results.items():
            fake_data = fake_lookup[fake_vid]
            
            result_entry = {
                "fake_video": {
                    "video_id": fake_vid,
                    "title": fake_data.get('title', ''),
                    "keywords": fake_data.get('keywords', ''),
                    "description": fake_data.get('description', ''),
                    "temporal_evolution": fake_data.get('temporal_evolution', ''),
                    "combined_text": f"{fake_data.get('title', '')} {fake_data.get('keywords', '')} {fake_data.get('description', '')} {fake_data.get('temporal_evolution', '')}"
                },
                "similar_true_videos": []
            }
            
            for sim_video in similar_videos:
                true_vid = sim_video['video_id']
                if true_vid in true_lookup:
                    true_data = true_lookup[true_vid]
                    result_entry["similar_true_videos"].append({
                        "video_id": true_vid,
                        "similarity_score": sim_video['similarity'],
                        "title": true_data.get('title', ''),
                        "keywords": true_data.get('keywords', ''),
                        "description": true_data.get('description', ''),
                        "temporal_evolution": true_data.get('temporal_evolution', ''),
                        "combined_text": f"{true_data.get('title', '')} {true_data.get('keywords', '')} {true_data.get('description', '')} {true_data.get('temporal_evolution', '')}",
                        "entities": true_data.get('entities', []),
                        "claims": true_data.get('claims', [])
                    })
            
            detailed_results.append(result_entry)
        
        return detailed_results
    
    def save_results(self, similarity_results: Dict, detailed_results: List[Dict]):
        """Save results with model name in filename"""
        # Save similarity scores
        similarity_file = self.output_dir / f"similarity_{self.model_short_name}.json"
        with open(similarity_file, 'w', encoding='utf-8') as f:
            json.dump(similarity_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved similarity results to {similarity_file}")
        
        # Save detailed results
        detailed_file = self.output_dir / f"detailed_retrieval_{self.model_short_name}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved detailed results to {detailed_file}")
        
        # Save summary statistics
        stats = {
            "model": self.model_name,
            "num_fake_videos": len(similarity_results),
            "num_true_videos": len(self.true_extractions),
            "avg_top1_similarity": np.mean([sims[0]['similarity'] for sims in similarity_results.values()]),
            "avg_top5_similarity": np.mean([np.mean([s['similarity'] for s in sims[:5]]) for sims in similarity_results.values()]),
            "min_top1_similarity": min([sims[0]['similarity'] for sims in similarity_results.values()]),
            "max_top1_similarity": max([sims[0]['similarity'] for sims in similarity_results.values()])
        }
        
        stats_file = self.output_dir / f"stats_{self.model_short_name}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved statistics to {stats_file}")
        
        # Print summary
        logger.info(f"\n=== Retrieval Results Summary for {self.model_name} ===")
        logger.info(f"Average Top-1 Similarity: {stats['avg_top1_similarity']:.4f}")
        logger.info(f"Average Top-5 Similarity: {stats['avg_top5_similarity']:.4f}")
        logger.info(f"Min Top-1 Similarity: {stats['min_top1_similarity']:.4f}")
        logger.info(f"Max Top-1 Similarity: {stats['max_top1_similarity']:.4f}")
    
    def run(self, top_k: int = 5):
        """Run the complete retrieval test"""
        logger.info(f"\n=== Starting Retrieval Test with {self.model_name} ===")
        
        # Load model
        self.load_model()
        
        # Load data
        self.load_data()
        
        # Prepare texts
        fake_texts, fake_video_ids, true_texts, true_video_ids = self.prepare_texts()
        
        # Encode texts
        logger.info("Encoding fake video texts...")
        fake_embeddings = self.encode_texts(fake_texts)
        
        logger.info("Encoding true video texts...")
        true_embeddings = self.encode_texts(true_texts)
        
        # Find similar videos
        similarity_results = self.find_similar_videos(
            fake_embeddings, true_embeddings, 
            fake_video_ids, true_video_ids, 
            top_k=top_k
        )
        
        # Create detailed output
        detailed_results = self.create_detailed_output(similarity_results)
        
        # Save results
        self.save_results(similarity_results, detailed_results)
        
        logger.info(f"\n=== Retrieval Test Complete ===")
        logger.info(f"Results saved in: {self.output_dir}")


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="Test retrieval models for fake news")
    parser.add_argument("--model", type=str, default="BAAI/bge-large-zh-v1.5",
                       help="Hugging Face model name for embeddings")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of similar videos to retrieve")
    args = parser.parse_args()
    
    # Create tester
    tester = RetrievalTester(model_name=args.model)
    
    # Run test
    tester.run(top_k=args.top_k)


if __name__ == "__main__":
    main()