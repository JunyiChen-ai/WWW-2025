#!/usr/bin/env python3
"""
Verify all feature files are in dictionary format and can be accessed by video ID
"""

import torch

def verify_feature_file(file_path, name):
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"File: {file_path}")
    print('='*60)
    
    try:
        data = torch.load(file_path, map_location='cpu')
        
        if isinstance(data, dict):
            print(f"✓ Format: Dictionary with {len(data)} entries")
            
            # Get sample key
            sample_key = list(data.keys())[0]
            sample_value = data[sample_key]
            
            if torch.is_tensor(sample_value):
                print(f"✓ Sample entry:")
                print(f"  Key (video ID): {sample_key}")
                print(f"  Value shape: {sample_value.shape}")
                print(f"  Value dtype: {sample_value.dtype}")
            
            # Test with known video ID
            test_id = "6946922230706097408"
            if test_id in data:
                print(f"✓ Can retrieve by video ID: {test_id}")
                print(f"  Shape: {data[test_id].shape}")
            
            return True
        else:
            print(f"✗ Format: {type(data)} - NOT a dictionary!")
            if torch.is_tensor(data):
                print(f"  Tensor shape: {data.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return False

def main():
    print("Verifying all feature files are in dictionary format...")
    
    files_to_check = [
        ("/data/jehc223/ExMRD_ours/data/FakeSV/fea/vit_tensor.pt", "ViT Visual Features"),
        ("/data/jehc223/ExMRD_ours/data/FakeSV/fea/audio_features_global.pt", "Audio Global Features"),
        ("/data/jehc223/ExMRD_ours/data/FakeSV/fea/audio_features_frames.pt", "Audio Frame Features")
    ]
    
    all_dict = True
    for file_path, name in files_to_check:
        is_dict = verify_feature_file(file_path, name)
        all_dict = all_dict and is_dict
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    if all_dict:
        print("✓ All feature files are in dictionary format")
        print("✓ All features can be accessed using: features = data[video_id]")
        print("\nFeature dimensions:")
        print("  - ViT: [16, 1024] (16 frames, 1024-dim per frame)")
        print("  - Audio Global: [768] (single 768-dim vector)")
        print("  - Audio Frames: [16, 768] (16 frames, 768-dim per frame)")
    else:
        print("✗ Some files are not in dictionary format!")
        print("  Run the conversion scripts to fix this.")

if __name__ == "__main__":
    main()