#!/usr/bin/env python3
"""
Exclusive Hit Rate (per-modality) bar chart

Definition (exclusive wrt T, V, A, and T+V+A):
  For a query q, a source S ∈ {T, V, A, T+V+A} gets an exclusive hit if
    - S achieves Event-Hit@1(pair) for q, and
    - none of the other sources achieve Event-Hit@1(pair) for q.

Event-Hit@1(pair) follows the same criterion used in preprocess/analyze_multimodal_choice.py
for consistency with prior analyses.

Usage example:
  python draw/exclusive_hit_rate.py \
    --dataset FakeSV \
    --audio-model laion-clap-htsat-fused \
    --text-model OFA-Sys/chinese-clip-vit-large-patch14 \
    --top-k 10

Outputs (default):
  analysis/<dataset>/draw/exclusive_hit_rate.png
  analysis/<dataset>/draw/exclusive_hit_rate.csv
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Make project root importable when running as a standalone script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse data loading and preparation from the analyzer
from preprocess.analyze_multimodal_choice import MultimodalChoiceAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_hit_vector(analyzer: MultimodalChoiceAnalyzer, probs: np.ndarray) -> np.ndarray:
    """Compute per-query Event-Hit@1(pair) as booleans, consistent with analyzer's logic.

    Returns an array of shape [Q] with values in {0,1}.
    """
    Q = len(analyzer.query_ids)
    hits = np.zeros(Q, dtype=np.int32)

    pos_mask = analyzer.candidate_meta['candidate_label'] == 1
    neg_mask = analyzer.candidate_meta['candidate_label'] == 0

    for q in range(Q):
        query_event = analyzer.query_events[q]
        row = probs[q]

        if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
            pos_probs = row[pos_mask]
            neg_probs = row[neg_mask]

            top1_pos_idx = int(np.argmax(pos_probs))
            top1_neg_idx = int(np.argmax(neg_probs))

            pos_events = analyzer.candidate_meta['candidate_event'][pos_mask]
            neg_events = analyzer.candidate_meta['candidate_event'][neg_mask]

            top1_pos_event = pos_events[top1_pos_idx]
            top1_neg_event = neg_events[top1_neg_idx]

            hit = int((top1_pos_event == query_event) or (top1_neg_event == query_event))
            hits[q] = hit
        else:
            hits[q] = 0

    return hits


def compute_exclusive_rates(analyzer: MultimodalChoiceAnalyzer, denominator: str = 'any_hit') -> pd.DataFrame:
    """Compute exclusive hit rates for T, V, A.

    - Exclusive: source hits while the other two do not.
    - Denominator for T/V: 'any_hit' = queries hit by at least one (T or V or A), 'all' = all queries.
    - Denominator for A: always number of queries hit by Audio (A_hit==1), per request.
    """
    # Prepare per-source probability matrices
    PT = analyzer.probs['T']
    PI = analyzer.probs['I']
    PA = analyzer.probs['A']

    # Per-query hits
    T_hit = compute_hit_vector(analyzer, PT)
    V_hit = compute_hit_vector(analyzer, PI)
    A_hit = compute_hit_vector(analyzer, PA)

    # Exclusive counts: hit for one source and not for others
    excl_T = (T_hit == 1) & (V_hit == 0) & (A_hit == 0)
    excl_V = (V_hit == 1) & (T_hit == 0) & (A_hit == 0)
    excl_A = (A_hit == 1) & (T_hit == 0) & (V_hit == 0)

    Q = len(analyzer.query_ids)
    any_hit_mask = (T_hit | V_hit | A_hit).astype(bool)
    denom_TV = int(any_hit_mask.sum()) if denominator == 'any_hit' else Q
    denom_A = int(A_hit.sum())  # Audio-specific denominator

    rates = {
        'Source': ['T', 'V', 'A'],
        'Exclusive_Hit_Rate': [
            float(excl_T.sum()) / denom_TV if denom_TV > 0 else 0.0,
            float(excl_V.sum()) / denom_TV if denom_TV > 0 else 0.0,
            float(excl_A.sum()) / denom_A if denom_A > 0 else 0.0,
        ],
        'Exclusive_Hit_Count': [int(excl_T.sum()), int(excl_V.sum()), int(excl_A.sum())],
        'Total_Queries': [Q, Q, Q],
        'Denominator_Mode': [denominator, denominator, 'audio_hit'],
        'Denominator_Value': [denom_TV, denom_TV, denom_A],
        'AnyHit_Count': [int(any_hit_mask.sum())]*3,
        'AudioHit_Count': [int(A_hit.sum())]*3,
    }
    return pd.DataFrame(rates)


def plot_bar(df: pd.DataFrame, out_path: Path):
    labels = df['Source'].tolist()
    values = df['Exclusive_Hit_Rate'].tolist()

    plt.figure(figsize=(6.5, 5))
    color_map = {'T': 'tab:blue', 'V': 'tab:orange', 'A': 'tab:red'}
    bars = plt.bar(labels, values, color=[color_map.get(s, 'gray') for s in labels], alpha=0.8)
    plt.ylabel('Exclusive Hit Rate')
    plt.ylim(0, max(0.001, max(values) * 1.2))
    # Title without denominator info per request
    plt.title('Exclusive Hit Rate per Source (Event-Hit@1(pair))')
    plt.grid(axis='y', alpha=0.3)

    # Annotate percentages above bars
    for bar, v in zip(bars, values):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + (0.01 if h > 0 else 0.002), f'{v*100:.1f}%',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Exclusive Hit Rate per Modality (and fused T+V+A)')
    parser.add_argument('--dataset', type=str, default='FakeSV', help='Dataset name')
    parser.add_argument('--audio-model', type=str, default='CAiRE-SER-wav2vec2-large-xlsr-53-eng-zho-all-age',
                        help='Audio model name suffix used to locate features')
    parser.add_argument('--text-model', type=str, default=None, help='Text model (auto if None)')
    # Note: top-k is not used in exclusive-hit computation; kept for parity with analyzer
    parser.add_argument('--top-k', type=int, default=10, help='Unused here; kept for parity with analyzer settings')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output dir (default: analysis/{dataset}/draw)')
    parser.add_argument('--denominator', type=str, default='any_hit', choices=['any_hit','all'],
                        help='Denominator for rates: any_hit (default) or all queries')

    args = parser.parse_args()

    # Output directory
    if args.output_dir is None:
        out_dir = Path(f'analysis/{args.dataset}/draw')
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare analyzer (reuses the same data and settings as the main analysis)
    analyzer = MultimodalChoiceAnalyzer(dataset=args.dataset,
                                        audio_model=args.audio_model,
                                        text_model=args.text_model,
                                        output_dir=str(out_dir),
                                        top_k=args.top_k)
    analyzer.load_data()
    analyzer.prepare_analysis_data()

    df = compute_exclusive_rates(analyzer, denominator=args.denominator)
    csv_path = out_dir / 'exclusive_hit_rate.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f'Saved CSV: {csv_path}')

    fig_path = out_dir / 'exclusive_hit_rate.png'
    plot_bar(df, fig_path)
    logger.info(f'Saved figure: {fig_path}')


if __name__ == '__main__':
    main()
