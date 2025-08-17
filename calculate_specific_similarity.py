#!/usr/bin/env python3
"""
Calculate similarity between two specific videos using different models
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, ChineseCLIPProcessor, ChineseCLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_video_data(video_id, annotation_filter=None):
    """Load data for a specific video"""
    # Try to find in true video extractions first
    true_file = 'data/FakeSV/entity_claims/video_extractions.jsonl'
    with open(true_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item['video_id'] == video_id:
                logger.info(f"Found {video_id} in true video extractions")
                return item
    
    # Try fake video descriptions
    fake_file = 'data/FakeSV/entity_claims/fake_video_descriptions.jsonl'
    try:
        with open(fake_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if item['video_id'] == video_id:
                    logger.info(f"Found {video_id} in fake video descriptions")
                    return item
    except FileNotFoundError:
        pass
    
    # Try original data
    orig_file = 'data/FakeSV/data_complete_orig.jsonl'
    with open(orig_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get('video_id') == video_id:
                logger.info(f"Found {video_id} in original data (annotation: {item.get('annotation')})")
                return item
    
    return None

def calculate_similarity_bge(text1, text2, model_name="BAAI/bge-large-zh-v1.5"):
    """Calculate similarity using BGE model"""
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    texts = [text1, text2]
    
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        
        # Use CLS token for BGE models
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        # Calculate cosine similarity
        sim = cosine_similarity(embeddings[0:1].cpu().numpy(), embeddings[1:2].cpu().numpy())[0][0]
    
    return float(sim)

def calculate_similarity_clip(text1, text2, model_name="OFA-Sys/chinese-clip-vit-large-patch14"):
    """Calculate similarity using Chinese-CLIP model"""
    logger.info(f"Loading {model_name}...")
    processor = ChineseCLIPProcessor.from_pretrained(model_name)
    model = ChineseCLIPModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    texts = [text1, text2]
    
    with torch.no_grad():
        inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate cosine similarity
        sim = cosine_similarity(text_features[0:1].cpu().numpy(), text_features[1:2].cpu().numpy())[0][0]
    
    return float(sim)

def main():
    # Target video IDs
    true_video_id = "6671891732524829965"
    fake_video_id = "6893044994064600332"
    
    logger.info(f"Calculating similarity between:")
    logger.info(f"  True video: {true_video_id}")
    logger.info(f"  Fake video: {fake_video_id}")
    
    # Load video data
    true_data = load_video_data(true_video_id)
    fake_data = load_video_data(fake_video_id)
    
    if not true_data:
        logger.error(f"Could not find data for true video {true_video_id}")
        return
    
    if not fake_data:
        logger.error(f"Could not find data for fake video {fake_video_id}")
        return
    
    # Print video information
    print("\n" + "="*60)
    print("VIDEO INFORMATION")
    print("="*60)
    
    print(f"\nTrue video ({true_video_id}):")
    print(f"  Title: {true_data.get('title', 'N/A')}")
    print(f"  Keywords: {true_data.get('keywords', 'N/A')}")
    print(f"  Description: {true_data.get('description', 'N/A')[:100]}...")
    print(f"  Annotation: {true_data.get('annotation', 'N/A')}")
    
    print(f"\nFake video ({fake_video_id}):")
    print(f"  Title: {fake_data.get('title', 'N/A')}")
    print(f"  Keywords: {fake_data.get('keywords', 'N/A')}")
    print(f"  Description: {fake_data.get('description', 'N/A')[:100]}...")
    print(f"  Annotation: {fake_data.get('annotation', 'N/A')}")
    
    # Prepare texts for similarity calculation
    true_text = f"{true_data.get('title', '')} {true_data.get('keywords', '')} {true_data.get('description', '')} {true_data.get('temporal_evolution', '')}"
    fake_text = f"{fake_data.get('title', '')} {fake_data.get('keywords', '')} {fake_data.get('description', '')} {fake_data.get('temporal_evolution', '')}"
    
    # Remove extra spaces
    true_text = ' '.join(true_text.split())
    fake_text = ' '.join(fake_text.split())
    
    print("\n" + "="*60)
    print("SIMILARITY CALCULATIONS")
    print("="*60)
    
    # Calculate similarity using BGE model
    print("\n1. Using BAAI/bge-large-zh-v1.5:")
    bge_sim = calculate_similarity_bge(true_text, fake_text)
    print(f"   Similarity score: {bge_sim:.4f}")
    
    # Calculate similarity using Chinese-CLIP
    print("\n2. Using OFA-Sys/chinese-clip-vit-large-patch14:")
    clip_sim = calculate_similarity_clip(true_text, fake_text)
    print(f"   Similarity score: {clip_sim:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"BGE similarity: {bge_sim:.4f}")
    print(f"CLIP similarity: {clip_sim:.4f}")
    print(f"Average: {(bge_sim + clip_sim) / 2:.4f}")
    
    # Check if this pair exists in our retrieval results
    retrieval_file = 'data/FakeSV/entity_claims/retrieval_tests/detailed_retrieval_bge-large-zh-v1.5.json'
    try:
        with open(retrieval_file, 'r') as f:
            retrieval_data = json.load(f)
            for item in retrieval_data:
                if item['fake_video']['video_id'] == fake_video_id:
                    print(f"\nThis fake video's top matches in our retrieval test:")
                    for i, match in enumerate(item['similar_true_videos'][:5], 1):
                        is_target = " <-- TARGET" if match['video_id'] == true_video_id else ""
                        print(f"  {i}. {match['video_id']}: {match['similarity_score']:.4f}{is_target}")
                    break
    except:
        pass

if __name__ == "__main__":
    main()