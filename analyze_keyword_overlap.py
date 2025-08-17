#!/usr/bin/env python3
"""
Analyze keyword overlap between training and test sets for temporal split
"""

import json
import pandas as pd
from pathlib import Path
from typing import Set, List, Dict
from collections import Counter

def load_video_ids(file_path: str) -> List[str]:
    """Load video IDs from a text file"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def analyze_keyword_overlap(dataset_name: str = 'FakeSV'):
    """
    Analyze keyword overlap between train and test sets
    
    Args:
        dataset_name: Name of dataset ('FakeSV', 'FakeTT', or 'FVC')
    """
    print(f"\n{'='*60}")
    print(f"Analyzing keyword overlap for {dataset_name} dataset")
    print('='*60)
    
    # Determine paths based on dataset
    if dataset_name == 'FakeSV':
        data_file = 'data/FakeSV/data_complete_orig.jsonl'
        train_ids_file = 'data/FakeSV/vids/vid_time3_train.txt'
        test_ids_file = 'data/FakeSV/vids/vid_time3_test.txt'
        valid_ids_file = 'data/FakeSV/vids/vid_time3_valid.txt'
        id_field = 'video_id'
        keyword_field = 'keywords'
    elif dataset_name == 'FakeTT':
        data_file = 'data/FakeTT/data.jsonl'
        train_ids_file = 'data/FakeTT/vids/vid_time3_train.txt'
        test_ids_file = 'data/FakeTT/vids/vid_time3_test.txt'
        valid_ids_file = 'data/FakeTT/vids/vid_time3_valid.txt'
        id_field = 'video_id'
        keyword_field = 'keywords'
    elif dataset_name == 'FVC':
        data_file = 'data/FVC/data.jsonl'
        train_ids_file = 'data/FVC/vids/vid_time3_train.txt'
        test_ids_file = 'data/FVC/vids/vid_time3_test.txt'
        valid_ids_file = 'data/FVC/vids/vid_time3_valid.txt'
        id_field = 'vid'
        keyword_field = 'keywords'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Check if files exist
    if not Path(data_file).exists():
        print(f"Data file not found: {data_file}")
        return
    if not Path(train_ids_file).exists():
        print(f"Train IDs file not found: {train_ids_file}")
        return
    if not Path(test_ids_file).exists():
        print(f"Test IDs file not found: {test_ids_file}")
        return
    
    # Load video IDs for each split
    train_ids = load_video_ids(train_ids_file)
    test_ids = load_video_ids(test_ids_file)
    valid_ids = load_video_ids(valid_ids_file) if Path(valid_ids_file).exists() else []
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_ids)} videos")
    print(f"  Valid: {len(valid_ids)} videos")
    print(f"  Test: {len(test_ids)} videos")
    
    # Load data
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Skip samples with annotation='辟谣'
            if item.get('annotation') != '辟谣':
                data.append(item)
    
    print(f"\nTotal videos in dataset (excluding '辟谣'): {len(data)}")
    
    # Create ID to keywords mapping
    id_to_keywords = {}
    keyword_stats = {
        'total_videos': 0,
        'videos_with_keywords': 0,
        'empty_keywords': 0
    }
    
    for item in data:
        vid = item.get(id_field, '')
        keywords = item.get(keyword_field, '')
        
        if vid:
            keyword_stats['total_videos'] += 1
            if keywords and keywords.strip():
                id_to_keywords[vid] = keywords.strip()
                keyword_stats['videos_with_keywords'] += 1
            else:
                id_to_keywords[vid] = ''
                keyword_stats['empty_keywords'] += 1
    
    print(f"\nKeyword statistics:")
    print(f"  Total videos: {keyword_stats['total_videos']}")
    print(f"  Videos with keywords: {keyword_stats['videos_with_keywords']}")
    print(f"  Videos without keywords: {keyword_stats['empty_keywords']}")
    
    # Get keywords for each split
    train_keywords = []
    test_keywords = []
    valid_keywords = []
    
    train_missing = []
    test_missing = []
    valid_missing = []
    
    for vid in train_ids:
        if vid in id_to_keywords:
            if id_to_keywords[vid]:
                train_keywords.append(id_to_keywords[vid])
        else:
            train_missing.append(vid)
    
    for vid in test_ids:
        if vid in id_to_keywords:
            if id_to_keywords[vid]:
                test_keywords.append(id_to_keywords[vid])
        else:
            test_missing.append(vid)
    
    for vid in valid_ids:
        if vid in id_to_keywords:
            if id_to_keywords[vid]:
                valid_keywords.append(id_to_keywords[vid])
        else:
            valid_missing.append(vid)
    
    if train_missing:
        print(f"\nWarning: {len(train_missing)} train videos not found in data")
    if test_missing:
        print(f"Warning: {len(test_missing)} test videos not found in data")
    if valid_missing:
        print(f"Warning: {len(valid_missing)} valid videos not found in data")
    
    print(f"\nKeywords found:")
    print(f"  Train: {len(train_keywords)} videos with keywords")
    print(f"  Valid: {len(valid_keywords)} videos with keywords")
    print(f"  Test: {len(test_keywords)} videos with keywords")
    
    # Analyze exact keyword matching
    train_keyword_set = set(train_keywords)
    test_keyword_set = set(test_keywords)
    valid_keyword_set = set(valid_keywords)
    
    # Count how many test keywords exactly match train keywords
    test_matches = 0
    test_match_details = []
    
    for test_kw in test_keywords:
        if test_kw in train_keyword_set:
            test_matches += 1
            test_match_details.append(test_kw)
    
    print(f"\n{'='*60}")
    print("EXACT KEYWORD MATCHING RESULTS")
    print('='*60)
    print(f"\nTest set keyword overlap with train set:")
    print(f"  {test_matches}/{len(test_keywords)} test samples have exact keyword match in train")
    print(f"  Percentage: {test_matches/len(test_keywords)*100:.2f}%")
    
    # Analyze partial keyword matching (word-level)
    print(f"\n{'='*60}")
    print("WORD-LEVEL KEYWORD MATCHING RESULTS")
    print('='*60)
    
    # Split keywords into words
    def get_words(keywords_list):
        words = set()
        for kw in keywords_list:
            # Split by common delimiters
            for word in kw.replace(',', ' ').replace('，', ' ').replace('、', ' ').split():
                word = word.strip()
                if word:
                    words.add(word)
        return words
    
    train_words = get_words(train_keywords)
    test_words = get_words(test_keywords)
    valid_words = get_words(valid_keywords)
    
    print(f"\nUnique words in keywords:")
    print(f"  Train: {len(train_words)} unique words")
    print(f"  Valid: {len(valid_words)} unique words")
    print(f"  Test: {len(test_words)} unique words")
    
    # Word overlap
    test_train_word_overlap = test_words.intersection(train_words)
    print(f"\nWord-level overlap:")
    print(f"  {len(test_train_word_overlap)}/{len(test_words)} test words appear in train")
    print(f"  Percentage: {len(test_train_word_overlap)/len(test_words)*100:.2f}%")
    
    # Count how many test samples have at least one word match
    test_partial_matches = 0
    for test_kw in test_keywords:
        test_kw_words = set(test_kw.replace(',', ' ').replace('，', ' ').replace('、', ' ').split())
        if test_kw_words.intersection(train_words):
            test_partial_matches += 1
    
    print(f"\nTest samples with at least one word match in train:")
    print(f"  {test_partial_matches}/{len(test_keywords)} samples")
    print(f"  Percentage: {test_partial_matches/len(test_keywords)*100:.2f}%")
    
    # Show most common keywords
    print(f"\n{'='*60}")
    print("MOST COMMON KEYWORDS")
    print('='*60)
    
    train_counter = Counter(train_keywords)
    test_counter = Counter(test_keywords)
    
    print("\nTop 10 most common keywords in train:")
    for kw, count in train_counter.most_common(10):
        print(f"  {count:3d} times: {kw[:50]}{'...' if len(kw) > 50 else ''}")
    
    print("\nTop 10 most common keywords in test:")
    for kw, count in test_counter.most_common(10):
        in_train = " [IN TRAIN]" if kw in train_keyword_set else " [NOT IN TRAIN]"
        print(f"  {count:3d} times: {kw[:50]}{'...' if len(kw) > 50 else ''}{in_train}")
    
    # Save detailed results
    output_file = f'keyword_overlap_analysis_{dataset_name}.json'
    results = {
        'dataset': dataset_name,
        'split_sizes': {
            'train': len(train_ids),
            'valid': len(valid_ids),
            'test': len(test_ids)
        },
        'keyword_counts': {
            'train': len(train_keywords),
            'valid': len(valid_keywords),
            'test': len(test_keywords)
        },
        'exact_matching': {
            'test_matches': test_matches,
            'test_total': len(test_keywords),
            'percentage': test_matches/len(test_keywords)*100 if test_keywords else 0
        },
        'word_level_matching': {
            'unique_train_words': len(train_words),
            'unique_test_words': len(test_words),
            'overlapping_words': len(test_train_word_overlap),
            'test_samples_with_word_match': test_partial_matches,
            'percentage': test_partial_matches/len(test_keywords)*100 if test_keywords else 0
        },
        'test_keywords_in_train': test_match_details[:20]  # Save first 20 examples
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return results


def main():
    """Analyze all datasets"""
    import argparse
    parser = argparse.ArgumentParser(description="Analyze keyword overlap between train and test sets")
    parser.add_argument('--dataset', type=str, default='all', 
                       choices=['FakeSV', 'FakeTT', 'FVC', 'all'],
                       help='Dataset to analyze')
    args = parser.parse_args()
    
    if args.dataset == 'all':
        datasets = ['FakeSV', 'FakeTT', 'FVC']
    else:
        datasets = [args.dataset]
    
    all_results = {}
    for dataset in datasets:
        try:
            results = analyze_keyword_overlap(dataset)
            all_results[dataset] = results
        except Exception as e:
            print(f"Error analyzing {dataset}: {e}")
    
    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY OF ALL DATASETS")
        print('='*60)
        for dataset, results in all_results.items():
            if results:
                exact_pct = results['exact_matching']['percentage']
                word_pct = results['word_level_matching']['percentage']
                print(f"\n{dataset}:")
                print(f"  Exact keyword match: {exact_pct:.2f}%")
                print(f"  Word-level match: {word_pct:.2f}%")


if __name__ == "__main__":
    main()