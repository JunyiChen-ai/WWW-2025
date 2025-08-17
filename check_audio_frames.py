#!/usr/bin/env python3
"""
Check the structure of audio_features_frames.pt
"""

import torch

def check_audio_frames():
    file_path = "/data/jehc223/ExMRD_ours/data/FakeSV/fea/audio_features_frames.pt"
    
    print(f"Loading {file_path}...")
    data = torch.load(file_path, map_location='cpu')
    
    print(f"\nData type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"✓ File is a dictionary with {len(data)} entries")
        
        # Show sample keys
        sample_keys = list(data.keys())[:5]
        print(f"\nFirst 5 keys:")
        for key in sample_keys:
            value = data[key]
            if torch.is_tensor(value):
                print(f"  {key}: tensor shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"  {key}: type {type(value)}")
                
    elif torch.is_tensor(data):
        print(f"✗ File is a tensor with shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"\nThis is NOT a dictionary format - it's a stacked tensor")
        print(f"  Likely format: [num_videos, num_frames, feature_dim]")
        
        # Check if there's a video ID mapping file
        video_ids_path = "/data/jehc223/ExMRD_ours/data/FakeSV/fea/audio_video_ids.txt"
        try:
            with open(video_ids_path, 'r') as f:
                video_ids = [line.strip() for line in f.readlines()]
            print(f"\n  Found video ID mapping file with {len(video_ids)} IDs")
            print(f"  Tensor has {data.shape[0]} videos")
            
            if len(video_ids) == data.shape[0]:
                print(f"  ✓ Video count matches - can be converted to dictionary")
            else:
                print(f"  ✗ Mismatch in counts!")
                
        except Exception as e:
            print(f"\n  No video ID mapping file found: {e}")
    else:
        print(f"Unexpected data type: {type(data)}")

if __name__ == "__main__":
    check_audio_frames()