#!/usr/bin/env python3
"""
Convert audio_features_frames.pt from tensor format to dictionary format
"""

import torch
import os

def convert_audio_frames_features():
    dataset_name = "FakeSV"
    frames_features_path = f"data/{dataset_name}/fea/audio_features_frames.pt"
    video_ids_path = f"data/{dataset_name}/fea/audio_video_ids.txt"
    
    print(f"Loading existing audio frame features from {frames_features_path}...")
    frames_features_tensor = torch.load(frames_features_path)
    
    # Check if it's already a dictionary
    if isinstance(frames_features_tensor, dict):
        print("File is already in dictionary format!")
        return
    
    print(f"Current format: tensor with shape {frames_features_tensor.shape}")
    print(f"  Shape breakdown: [num_videos={frames_features_tensor.shape[0]}, num_frames={frames_features_tensor.shape[1]}, feature_dim={frames_features_tensor.shape[2]}]")
    
    # Load video IDs
    print(f"\nLoading video IDs from {video_ids_path}...")
    with open(video_ids_path, 'r') as f:
        video_ids = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(video_ids)} video IDs")
    
    # Verify dimensions match
    if len(video_ids) != frames_features_tensor.shape[0]:
        print(f"ERROR: Number of video IDs ({len(video_ids)}) doesn't match tensor shape ({frames_features_tensor.shape[0]})")
        return
    
    # Create dictionary
    print("\nConverting to dictionary format...")
    frames_features_dict = {}
    for i, video_id in enumerate(video_ids):
        frames_features_dict[video_id] = frames_features_tensor[i]  # Shape: [16, 768]
    
    # Backup original file
    backup_path = f"{frames_features_path}.tensor_backup"
    print(f"Creating backup at {backup_path}...")
    torch.save(frames_features_tensor, backup_path)
    
    # Save as dictionary
    print(f"Saving dictionary format to {frames_features_path}...")
    torch.save(frames_features_dict, frames_features_path)
    
    print(f"\n✓ Successfully converted to dictionary format with {len(frames_features_dict)} entries")
    
    # Verify the conversion
    print("\nVerifying conversion...")
    loaded_dict = torch.load(frames_features_path)
    
    if isinstance(loaded_dict, dict):
        sample_keys = list(loaded_dict.keys())[:5]
        print(f"✓ File is now a dictionary")
        print(f"Sample keys: {sample_keys}")
        
        # Test with a known video ID
        test_id = "6946922230706097408"
        if test_id in loaded_dict:
            features = loaded_dict[test_id]
            print(f"\n✓ Test lookup successful:")
            print(f"  Video ID: {test_id}")
            print(f"  Feature shape: {features.shape} (16 frames × 768 dimensions)")
            print(f"  Feature dtype: {features.dtype}")
        else:
            # Try with first available ID
            first_id = sample_keys[0]
            features = loaded_dict[first_id]
            print(f"\n✓ Sample lookup:")
            print(f"  Video ID: {first_id}")
            print(f"  Feature shape: {features.shape} (16 frames × 768 dimensions)")
            print(f"  Feature dtype: {features.dtype}")
    else:
        print("ERROR: Conversion failed!")

if __name__ == "__main__":
    convert_audio_frames_features()