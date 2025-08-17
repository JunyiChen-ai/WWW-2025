#!/usr/bin/env python3
"""
Analyze temporal distribution of train/valid/test splits
"""

import json
from datetime import datetime
from collections import Counter, defaultdict

def load_video_ids(file_path):
    """Load video IDs from split file"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def analyze_temporal_distribution():
    """Analyze temporal distribution of train/valid/test splits"""
    
    # Load video IDs for each split
    train_ids = set(load_video_ids('data/FakeSV/vids/vid_time3_train.txt'))
    valid_ids = set(load_video_ids('data/FakeSV/vids/vid_time3_valid.txt'))
    test_ids = set(load_video_ids('data/FakeSV/vids/vid_time3_test.txt'))
    
    print(f"Split sizes (from files):")
    print(f"  Train: {len(train_ids)} videos")
    print(f"  Valid: {len(valid_ids)} videos")
    print(f"  Test: {len(test_ids)} videos")
    print()
    
    # Load data with timestamps
    split_data = {
        'train': {'timestamps': [], 'dates': [], 'annotations': []},
        'valid': {'timestamps': [], 'dates': [], 'annotations': []},
        'test': {'timestamps': [], 'dates': [], 'annotations': []}
    }
    
    # Read all data
    with open('data/FakeSV/data_complete_orig.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            video_id = item.get('video_id')
            timestamp = item.get('publish_time_norm')
            annotation = item.get('annotation')
            
            if not timestamp:
                continue
                
            date = datetime.fromtimestamp(timestamp / 1000)
            
            if video_id in train_ids:
                split_data['train']['timestamps'].append(timestamp)
                split_data['train']['dates'].append(date)
                split_data['train']['annotations'].append(annotation)
            elif video_id in valid_ids:
                split_data['valid']['timestamps'].append(timestamp)
                split_data['valid']['dates'].append(date)
                split_data['valid']['annotations'].append(annotation)
            elif video_id in test_ids:
                split_data['test']['timestamps'].append(timestamp)
                split_data['test']['dates'].append(date)
                split_data['test']['annotations'].append(annotation)
    
    print("="*60)
    print("TEMPORAL DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Analyze each split
    for split_name in ['train', 'valid', 'test']:
        dates = split_data[split_name]['dates']
        annotations = split_data[split_name]['annotations']
        
        if not dates:
            print(f"\n{split_name.upper()}: No data found")
            continue
            
        print(f"\n{split_name.upper()} SET:")
        print(f"  Total samples: {len(dates)}")
        
        # Time range
        min_date = min(dates)
        max_date = max(dates)
        print(f"  Time range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        print(f"  Span: {(max_date - min_date).days} days")
        
        # Year distribution
        year_counts = Counter(d.year for d in dates)
        print(f"  Year distribution:")
        for year in sorted(year_counts.keys()):
            print(f"    {year}: {year_counts[year]:4d} samples ({year_counts[year]/len(dates)*100:.1f}%)")
        
        # Month distribution for key years
        print(f"  Monthly distribution (2020-2021):")
        for year in [2020, 2021]:
            year_dates = [d for d in dates if d.year == year]
            if year_dates:
                month_counts = Counter(d.month for d in year_dates)
                months_str = ', '.join([f"{m}月:{month_counts.get(m, 0)}" for m in range(1, 13)])
                print(f"    {year}: {months_str}")
        
        # Label distribution
        label_counts = Counter(annotations)
        print(f"  Label distribution:")
        for label, count in label_counts.most_common():
            print(f"    {label}: {count:4d} ({count/len(annotations)*100:.1f}%)")
    
    # Check temporal overlap
    print("\n" + "="*60)
    print("TEMPORAL OVERLAP ANALYSIS")
    print("="*60)
    
    train_dates = split_data['train']['dates']
    test_dates = split_data['test']['dates']
    valid_dates = split_data['valid']['dates']
    
    if train_dates and test_dates:
        train_min, train_max = min(train_dates), max(train_dates)
        test_min, test_max = min(test_dates), max(test_dates)
        valid_min, valid_max = (min(valid_dates), max(valid_dates)) if valid_dates else (None, None)
        
        print(f"\nTrain: {train_min.strftime('%Y-%m-%d')} to {train_max.strftime('%Y-%m-%d')}")
        print(f"Valid: {valid_min.strftime('%Y-%m-%d') if valid_min else 'N/A'} to {valid_max.strftime('%Y-%m-%d') if valid_max else 'N/A'}")
        print(f"Test:  {test_min.strftime('%Y-%m-%d')} to {test_max.strftime('%Y-%m-%d')}")
        
        # Check for temporal overlap
        if train_max >= test_min:
            overlap_days = (train_max - test_min).days
            print(f"\n⚠️  WARNING: Temporal overlap detected!")
            print(f"   Train set extends {overlap_days} days into test period")
            print(f"   Latest train: {train_max.strftime('%Y-%m-%d')}")
            print(f"   Earliest test: {test_min.strftime('%Y-%m-%d')}")
        else:
            gap_days = (test_min - train_max).days
            print(f"\n✓ No temporal overlap")
            print(f"  Gap between train and test: {gap_days} days")
    
    # Monthly timeline visualization
    print("\n" + "="*60)
    print("MONTHLY TIMELINE VISUALIZATION")
    print("="*60)
    
    # Create monthly bins
    all_dates = train_dates + valid_dates + test_dates
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        # Generate all months between min and max
        current = datetime(min_date.year, min_date.month, 1)
        end = datetime(max_date.year, max_date.month, 1)
        
        monthly_counts = defaultdict(lambda: {'train': 0, 'valid': 0, 'test': 0})
        
        for split_name, dates in [('train', train_dates), ('valid', valid_dates), ('test', test_dates)]:
            for date in dates:
                month_key = f"{date.year}-{date.month:02d}"
                monthly_counts[month_key][split_name] += 1
        
        # Print timeline
        print("\nMonth     | Train | Valid | Test  | Total")
        print("-" * 45)
        
        months = sorted(monthly_counts.keys())
        for month in months:
            counts = monthly_counts[month]
            total = counts['train'] + counts['valid'] + counts['test']
            
            # Visual bar for proportion
            train_bar = '█' * (counts['train'] // 20)
            valid_bar = '▓' * (counts['valid'] // 20)
            test_bar = '░' * (counts['test'] // 20)
            
            print(f"{month} | {counts['train']:5d} | {counts['valid']:5d} | {counts['test']:5d} | {total:5d} {train_bar}{valid_bar}{test_bar}")
    
    # Save detailed results
    results = {
        'split_sizes': {
            'train': len(train_dates),
            'valid': len(valid_dates),
            'test': len(test_dates)
        },
        'time_ranges': {
            'train': {
                'min': train_min.isoformat() if train_dates else None,
                'max': train_max.isoformat() if train_dates else None
            },
            'valid': {
                'min': valid_min.isoformat() if valid_dates else None,
                'max': valid_max.isoformat() if valid_dates else None
            },
            'test': {
                'min': test_min.isoformat() if test_dates else None,
                'max': test_max.isoformat() if test_dates else None
            }
        },
        'monthly_distribution': dict(monthly_counts)
    }
    
    with open('temporal_split_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: temporal_split_analysis.json")

if __name__ == "__main__":
    analyze_temporal_distribution()