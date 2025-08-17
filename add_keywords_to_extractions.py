#!/usr/bin/env python3
"""
Add keywords field to video_extractions.jsonl from original data
"""

import json
from tqdm import tqdm

def main():
    # Read original data and create mapping of video_id to keywords
    video_keywords = {}
    
    print("Loading keywords from original data...")
    with open('data/FakeSV/data_complete_orig.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            video_id = item.get('video_id')
            keywords = item.get('keywords', '')
            if video_id:
                video_keywords[video_id] = keywords
    
    print(f"Loaded keywords for {len(video_keywords)} videos")
    
    # Read video extractions and add keywords
    updated_items = []
    missing_keywords = []
    
    print("\nProcessing video_extractions.jsonl...")
    with open('data/FakeSV/entity_claims/video_extractions.jsonl', 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            item = json.loads(line)
            video_id = item.get('video_id')
            
            # Add keywords if found in original data
            if video_id in video_keywords:
                item['keywords'] = video_keywords[video_id]
            else:
                missing_keywords.append(video_id)
                item['keywords'] = ""  # Add empty string if not found
            
            updated_items.append(item)
    
    # Write updated data back to file
    output_file = 'data/FakeSV/entity_claims/video_extractions_with_keywords.jsonl'
    print(f"\nWriting updated data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in updated_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total videos processed: {len(updated_items)}")
    print(f"Videos with keywords added: {len(updated_items) - len(missing_keywords)}")
    print(f"Videos without keywords in original data: {len(missing_keywords)}")
    
    if missing_keywords:
        print(f"\nFirst 10 videos without keywords: {missing_keywords[:10]}")
    
    # Verify the update
    print("\nVerifying the update...")
    with open(output_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        first_item = json.loads(first_line)
        print(f"\nFirst item with keywords:")
        print(f"  video_id: {first_item.get('video_id')}")
        print(f"  title: {first_item.get('title')}")
        print(f"  keywords: {first_item.get('keywords')}")
        print(f"  annotation: {first_item.get('annotation')}")
    
    print(f"\nUpdated file saved to: {output_file}")
    print("\nTo replace the original file, run:")
    print(f"  mv {output_file} data/FakeSV/entity_claims/video_extractions.jsonl")

if __name__ == "__main__":
    main()