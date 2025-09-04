#!/usr/bin/env python3
"""
Analyze exclusive modality winners in unimodal retrieval results.
Identifies queries where only one modality retrieves correct samples (matching keywords)
while the other two modalities fail.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModalityExclusiveWinnerAnalyzer:
    def __init__(self, dataset: str = "FakeSV"):
        """
        Initialize exclusive winner analyzer
        
        Args:
            dataset: Dataset name (e.g., 'FakeSV', 'FakeTT')
        """
        self.dataset = dataset
        
        # Dataset-specific paths
        self.analysis_dir = Path(f"analysis/{dataset}")
        
        if not self.analysis_dir.exists():
            raise ValueError(f"Analysis directory not found: {self.analysis_dir}")
        
        logger.info(f"Initializing exclusive winner analysis for {dataset}")
        
        # Define input files
        self.input_files = {
            'text_title_only': self.analysis_dir / "unimodal_text_title_only_retrieval.json",
            'text_title_keywords': self.analysis_dir / "unimodal_text_title_keywords_retrieval.json",
            'text_full': self.analysis_dir / "unimodal_text_full_retrieval.json",
            'visual_mean': self.analysis_dir / "unimodal_visual_mean_retrieval.json",
            'visual_max': self.analysis_dir / "unimodal_visual_max_retrieval.json", 
            'audio_mean': self.analysis_dir / "unimodal_audio_mean_retrieval.json",
            'audio_max': self.analysis_dir / "unimodal_audio_max_retrieval.json"
        }
        
        # Check that all required files exist
        missing_files = []
        for name, file_path in self.input_files.items():
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        logger.info("All required input files found")
    
    def load_retrieval_results(self):
        """Load all retrieval results"""
        self.results = {}
        
        for modality, file_path in self.input_files.items():
            logger.info(f"Loading {modality} results from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                self.results[modality] = json.load(f)
            logger.info(f"Loaded {len(self.results[modality])} results for {modality}")
        
        # Verify all modalities have the same number of queries and same video IDs
        video_ids_sets = {}
        for modality, results in self.results.items():
            video_ids_sets[modality] = set([r['query_video']['video_id'] for r in results])
        
        # Check consistency
        reference_modality = 'text_title_only'
        reference_video_ids = video_ids_sets[reference_modality]
        
        for modality, video_ids in video_ids_sets.items():
            if video_ids != reference_video_ids:
                logger.warning(f"Modality {modality} has different video IDs than {reference_modality}")
                logger.warning(f"  {reference_modality}: {len(reference_video_ids)} videos")
                logger.warning(f"  {modality}: {len(video_ids)} videos")
        
        # Use intersection of all video IDs for analysis
        self.common_video_ids = reference_video_ids
        for video_ids in video_ids_sets.values():
            self.common_video_ids = self.common_video_ids.intersection(video_ids)
        
        logger.info(f"Analyzing {len(self.common_video_ids)} common video IDs across all modalities")
    
    def check_keywords_match(self, query_keywords: str, similar_item: Optional[Dict]) -> bool:
        """
        Check if retrieved item's keywords match query keywords
        
        Args:
            query_keywords: Keywords from query video
            similar_item: Retrieved similar item (can be None)
            
        Returns:
            True if keywords match, False otherwise
        """
        if similar_item is None:
            return False
        
        retrieved_keywords = similar_item.get('keywords', '')
        return query_keywords == retrieved_keywords
    
    def check_modality_match(self, query_video_id: str, modality: str) -> bool:
        """
        Check if a modality has any matching retrievals for given query
        
        Args:
            query_video_id: Video ID of query
            modality: Modality to check ('text_title_only', 'text_title_keywords', 'text_full', 'visual_mean', 'visual_max', 'audio_mean', 'audio_max')
            
        Returns:
            True if modality has at least one matching retrieval
        """
        # Map modality to results key
        if modality in ['text_title_only', 'text_title_keywords', 'text_full', 'visual_mean', 'visual_max', 'audio_mean', 'audio_max']:
            results_key = modality
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        results = self.results[results_key]
        query_result = next((r for r in results if r['query_video']['video_id'] == query_video_id), None)
        
        if query_result is None:
            return False
        
        query_keywords = query_result['query_video']['keywords']
        
        # Check both true and fake retrievals
        match_true = self.check_keywords_match(query_keywords, query_result.get('similar_true'))
        match_fake = self.check_keywords_match(query_keywords, query_result.get('similar_fake'))
        
        return match_true or match_fake
    
    def find_exclusive_winners(self):
        """Find queries for all possible 3-modality combinations"""
        logger.info("Finding winners for all possible 3-modality combinations...")
        
        # Define all possible combinations: text x visual x audio
        text_variants = ['text_title_only', 'text_title_keywords', 'text_full']
        visual_variants = ['visual_mean', 'visual_max']
        audio_variants = ['audio_mean', 'audio_max']
        
        # Generate all possible combinations (3 x 2 x 2 = 12 combinations)
        self.combinations = []
        for text in text_variants:
            for visual in visual_variants:
                for audio in audio_variants:
                    combo_name = f"{text}__{visual}__{audio}"
                    self.combinations.append({
                        'name': combo_name,
                        'text': text,
                        'visual': visual, 
                        'audio': audio
                    })
        
        logger.info(f"Generated {len(self.combinations)} possible combinations")
        
        # Initialize storage for exclusive winners of each combination
        self.exclusive_winners = {}
        for combo in self.combinations:
            self.exclusive_winners[combo['name']] = []
        
        # Track detailed statistics
        self.match_stats = {
            'total_queries': 0,
            'text_title_only_matches': 0,
            'text_title_keywords_matches': 0,
            'text_full_matches': 0,
            'visual_mean_matches': 0,
            'visual_max_matches': 0,
            'audio_mean_matches': 0,
            'audio_max_matches': 0,
            'combination_stats': {}
        }
        
        # Initialize combination stats
        for combo in self.combinations:
            self.match_stats['combination_stats'][combo['name']] = {
                'exclusive_winners': 0,
                'percentage': 0.0
            }
        
        for video_id in self.common_video_ids:
            self.match_stats['total_queries'] += 1
            
            # Check each modality variant
            matches = {}
            for modality in ['text_title_only', 'text_title_keywords', 'text_full', 'visual_mean', 'visual_max', 'audio_mean', 'audio_max']:
                matches[modality] = self.check_modality_match(video_id, modality)
                if matches[modality]:
                    self.match_stats[f'{modality}_matches'] += 1
            
            # For each combination, find exclusive winners for each modality
            for combo in self.combinations:
                text_match = matches[combo['text']]
                visual_match = matches[combo['visual']]
                audio_match = matches[combo['audio']]
                
                combo_name = combo['name']
                
                # Text exclusive in this combination: text wins, visual and audio don't
                if text_match and not visual_match and not audio_match:
                    key = f"{combo_name}_text_win"
                    if key not in self.exclusive_winners:
                        self.exclusive_winners[key] = []
                    
                    query_result = next((r for r in self.results[combo['text']] 
                                       if r['query_video']['video_id'] == video_id), None)
                    if query_result:
                        self.exclusive_winners[key].append(query_result)
                
                # Visual exclusive in this combination: visual wins, text and audio don't  
                if visual_match and not text_match and not audio_match:
                    key = f"{combo_name}_visual_win"
                    if key not in self.exclusive_winners:
                        self.exclusive_winners[key] = []
                    
                    query_result = next((r for r in self.results[combo['visual']] 
                                       if r['query_video']['video_id'] == video_id), None)
                    if query_result:
                        self.exclusive_winners[key].append(query_result)
                
                # Audio exclusive in this combination: audio wins, text and visual don't
                if audio_match and not text_match and not visual_match:
                    key = f"{combo_name}_audio_win"
                    if key not in self.exclusive_winners:
                        self.exclusive_winners[key] = []
                    
                    query_result = next((r for r in self.results[combo['audio']] 
                                       if r['query_video']['video_id'] == video_id), None)
                    if query_result:
                        self.exclusive_winners[key].append(query_result)
        
        logger.info("Exclusive modality winner analysis complete")
        
        # Log results for each combination and modality
        for combo in self.combinations:
            combo_name = combo['name']
            text_key = f"{combo_name}_text_win"
            visual_key = f"{combo_name}_visual_win" 
            audio_key = f"{combo_name}_audio_win"
            
            text_count = len(self.exclusive_winners.get(text_key, []))
            visual_count = len(self.exclusive_winners.get(visual_key, []))
            audio_count = len(self.exclusive_winners.get(audio_key, []))
            
            logger.info(f"Combination {combo_name}:")
            logger.info(f"  Text exclusive: {text_count}")
            logger.info(f"  Visual exclusive: {visual_count}")
            logger.info(f"  Audio exclusive: {audio_count}")
    
    def generate_summary_statistics(self, winners: List[Dict], modality: str) -> Dict:
        """Generate summary statistics for exclusive winners"""
        stats = {
            'modality': modality,
            'total_exclusive_winners': len(winners),
            'total_queries_analyzed': self.match_stats['total_queries'],
            'exclusive_win_rate': len(winners) / self.match_stats['total_queries'] if self.match_stats['total_queries'] > 0 else 0.0,
            'splits': {},
            'match_types': {'true_match': 0, 'fake_match': 0, 'both_match': 0},
            'overall_match_stats': self.match_stats
        }
        
        # Count by splits and match types
        for winner in winners:
            split = winner['query_video']['split']
            if split not in stats['splits']:
                stats['splits'][split] = 0
            stats['splits'][split] += 1
            
            # Analyze match types
            query_keywords = winner['query_video']['keywords']
            true_match = self.check_keywords_match(query_keywords, winner.get('similar_true'))
            fake_match = self.check_keywords_match(query_keywords, winner.get('similar_fake'))
            
            if true_match and fake_match:
                stats['match_types']['both_match'] += 1
            elif true_match:
                stats['match_types']['true_match'] += 1
            elif fake_match:
                stats['match_types']['fake_match'] += 1
        
        return stats
    
    def generate_combination_statistics(self, winners: List[Dict], combo: Dict) -> Dict:
        """Generate statistics for a specific 3-modality combination"""
        stats = {
            'combination_name': combo['name'],
            'text_variant': combo['text'],
            'visual_variant': combo['visual'],
            'audio_variant': combo['audio'],
            'total_winners': len(winners),
            'total_queries_analyzed': self.match_stats['total_queries'],
            'win_rate': len(winners) / self.match_stats['total_queries'] if self.match_stats['total_queries'] > 0 else 0.0,
            'splits': {},
            'match_types': {'true_match': 0, 'fake_match': 0, 'both_match': 0},
            'overall_match_stats': self.match_stats
        }
        
        # Count by splits and match types
        for winner in winners:
            split = winner['query_video']['split']
            if split not in stats['splits']:
                stats['splits'][split] = 0
            stats['splits'][split] += 1
            
            # Analyze match types
            query_keywords = winner['query_video']['keywords']
            true_match = self.check_keywords_match(query_keywords, winner.get('similar_true'))
            fake_match = self.check_keywords_match(query_keywords, winner.get('similar_fake'))
            
            if true_match and fake_match:
                stats['match_types']['both_match'] += 1
            elif true_match:
                stats['match_types']['true_match'] += 1
            elif fake_match:
                stats['match_types']['fake_match'] += 1
        
        return stats
    
    def save_results(self):
        """Save exclusive modality winners organized by combinations"""
        logger.info("Saving exclusive modality winner results...")
        
        # Create base directory for combinations
        base_combo_dir = self.analysis_dir / "modality_exclusive_combinations"
        base_combo_dir.mkdir(exist_ok=True)
        
        # For each combination, create a subdirectory and save modality-specific winners
        for combo in self.combinations:
            combo_name = combo['name']
            combo_dir = base_combo_dir / combo_name
            combo_dir.mkdir(exist_ok=True)
            
            # Save text, visual, audio exclusive winners for this combination
            modalities = ['text', 'visual', 'audio']
            for modality in modalities:
                key = f"{combo_name}_{modality}_win"
                winners = self.exclusive_winners.get(key, [])
                
                # Save winners
                output_file = combo_dir / f"{modality}_exclusive.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(winners, f, ensure_ascii=False, indent=2)
                
                # Generate and save statistics
                stats = {
                    'combination': combo_name,
                    'text_variant': combo['text'],
                    'visual_variant': combo['visual'], 
                    'audio_variant': combo['audio'],
                    'winning_modality': modality,
                    'total_winners': len(winners),
                    'total_queries': self.match_stats['total_queries'],
                    'win_rate': len(winners) / self.match_stats['total_queries'] if self.match_stats['total_queries'] > 0 else 0.0,
                    'splits': {},
                    'match_types': {'true_match': 0, 'fake_match': 0, 'both_match': 0}
                }
                
                # Calculate detailed stats
                for winner in winners:
                    split = winner['query_video']['split']
                    if split not in stats['splits']:
                        stats['splits'][split] = 0
                    stats['splits'][split] += 1
                    
                    query_keywords = winner['query_video']['keywords']
                    true_match = self.check_keywords_match(query_keywords, winner.get('similar_true'))
                    fake_match = self.check_keywords_match(query_keywords, winner.get('similar_fake'))
                    
                    if true_match and fake_match:
                        stats['match_types']['both_match'] += 1
                    elif true_match:
                        stats['match_types']['true_match'] += 1
                    elif fake_match:
                        stats['match_types']['fake_match'] += 1
                
                stats_file = combo_dir / f"{modality}_exclusive.stats.json"
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved {len(winners)} {modality} exclusive winners for {combo_name}")
        
        logger.info(f"All results saved to {base_combo_dir}")
    
    def print_summary(self):
        """Print analysis summary for exclusive modality winners by combination"""
        logger.info("="*100)
        logger.info("EXCLUSIVE MODALITY WINNERS BY COMBINATION ANALYSIS SUMMARY")
        logger.info("="*100)
        logger.info(f"Total queries analyzed: {self.match_stats['total_queries']}")
        logger.info(f"Total combinations analyzed: {len(self.combinations)}")
        logger.info("-"*80)
        
        # Calculate totals for summary
        total_text_exclusive = 0
        total_visual_exclusive = 0
        total_audio_exclusive = 0
        
        for combo in self.combinations:
            combo_name = combo['name']
            text_count = len(self.exclusive_winners.get(f"{combo_name}_text_win", []))
            visual_count = len(self.exclusive_winners.get(f"{combo_name}_visual_win", []))
            audio_count = len(self.exclusive_winners.get(f"{combo_name}_audio_win", []))
            
            total_text_exclusive += text_count
            total_visual_exclusive += visual_count
            total_audio_exclusive += audio_count
            
            logger.info(f"{combo_name}:")
            logger.info(f"  Text exclusive: {text_count} ({text_count/self.match_stats['total_queries']*100:.2f}%)")
            logger.info(f"  Visual exclusive: {visual_count} ({visual_count/self.match_stats['total_queries']*100:.2f}%)")
            logger.info(f"  Audio exclusive: {audio_count} ({audio_count/self.match_stats['total_queries']*100:.2f}%)")
            logger.info("-"*50)
        
        logger.info("OVERALL TOTALS:")
        logger.info(f"Total text exclusive winners (across all combinations): {total_text_exclusive}")
        logger.info(f"Total visual exclusive winners (across all combinations): {total_visual_exclusive}")
        logger.info(f"Total audio exclusive winners (across all combinations): {total_audio_exclusive}")
        logger.info("="*100)
    
    def run_analysis(self):
        """Run the complete exclusive winner analysis"""
        logger.info("Starting exclusive modality winner analysis...")
        
        # Load all results
        self.load_retrieval_results()
        
        # Find exclusive winners
        self.find_exclusive_winners()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
        
        logger.info(f"Analysis complete! Results saved to: {self.analysis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze exclusive modality winners in retrieval results')
    parser.add_argument('--dataset', type=str, default='FakeSV',
                       help='Dataset name (default: FakeSV). Examples: FakeSV, FakeTT')
    
    args = parser.parse_args()
    
    analyzer = ModalityExclusiveWinnerAnalyzer(dataset=args.dataset)
    analyzer.run_analysis()