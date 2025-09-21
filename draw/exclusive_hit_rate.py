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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import logging

# Make project root importable when running as a standalone script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse data loading and preparation from the analyzer
from preprocess.analyze_multimodal_choice import MultimodalChoiceAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================
# Global plot parameters
# =====================
# Figure size kept very small while fonts are large
FIGSIZE = (2.0, 1.4)  # width, height in inches (narrow and small)

# Font sizes (use large values given tiny figure)
FONT_SIZE = 6
AXES_LABEL_SIZE = 10
# Separate x/y tick sizes so y-tick can be a bit smaller
XTICK_LABEL_SIZE = 9
YTICK_LABEL_SIZE = 8

# Bar appearance
BAR_EDGE_WIDTH = 0.6
BAR_ALPHA = 0.9
# Macaron (pastel) color scheme per source
BAR_COLORS = {
    'T': '#A8D8EA',  # pastel blue
    'V': '#FFD8A8',  # pastel orange
    'A': '#CDEAC0',  # pastel green
}
BAR_EDGE_COLOR = '#444444'

# Grid appearance
GRID_ALPHA = 0.25

# PDF vector font type (42 embeds TrueType, better compatibility)
PDF_FONTTYPE = 42

# Axis labels
Y_LABEL = 'Hit@1'

# Spacing controls
# Move plot area closer to the left edge
LEFT_MARGIN = 0.06  # fraction of figure width (0..1)
# Bring y-label closer to tick labels
YLABEL_PAD = 2.0
# Reduce gap between y-tick labels and the axis
YTICK_PAD = 1.0


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
    # Apply global styles
    mpl.rcParams.update({
        'pdf.fonttype': PDF_FONTTYPE,
        'font.size': FONT_SIZE,
        'axes.labelsize': AXES_LABEL_SIZE,
        'xtick.labelsize': XTICK_LABEL_SIZE,
        'ytick.labelsize': YTICK_LABEL_SIZE,
    })

    labels = df['Source'].tolist()
    values = df['Exclusive_Hit_Rate'].tolist()

    plt.figure(figsize=FIGSIZE)
    bars = plt.bar(
        labels,
        values,
        color=[BAR_COLORS.get(s, '#E0E0E0') for s in labels],
        alpha=BAR_ALPHA,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )
    ax = plt.gca()
    ax.set_ylabel(Y_LABEL, labelpad=YLABEL_PAD)
    # Tight headroom, keep visible margin
    ymax = max(0.001, max(values) * 1.1)
    ax.set_ylim(0, ymax)
    # Force exactly 4 y-ticks with two decimal places
    ax.yaxis.set_major_locator(LinearLocator(4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # No title per request
    # Grid only on Y for readability
    ax.grid(axis='y', alpha=GRID_ALPHA)
    # Make y-tick labels slightly smaller and closer to axis
    ax.tick_params(axis='y', labelsize=YTICK_LABEL_SIZE, pad=YTICK_PAD)
    ax.tick_params(axis='x', labelsize=XTICK_LABEL_SIZE)

    # No numeric annotations on bars per request

    plt.tight_layout()
    # Nudge plot toward the left by shrinking left margin
    plt.gcf().subplots_adjust(left=LEFT_MARGIN)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
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
    # Cache control: default to using cache; allow explicit disable via --no-cache
    parser.add_argument('--use-cache', dest='use_cache', action='store_true', default=True,
                        help='Use cached prepared data if available (default: True)')
    parser.add_argument('--no-cache', dest='use_cache', action='store_false',
                        help='Disable cache and recompute all intermediates')

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
                                        top_k=args.top_k,
                                        use_cache=args.use_cache)

    loaded_from_cache = False
    if args.use_cache:
        try:
            # Attempt to load prepared matrices directly from cache without heavy loading
            analyzer.prepare_analysis_data()
            # Heuristic check: cache should populate probs for T/I/A
            if isinstance(analyzer.probs, dict) and all(k in analyzer.probs for k in ['T', 'I', 'A']):
                loaded_from_cache = True
                logger.info('Prepared analysis data loaded from cache successfully.')
        except Exception as e:
            logger.warning(f'Failed to load prepared data from cache: {e}')

    if not loaded_from_cache:
        # Fall back to full data load and preparation
        logger.info('Cache disabled or missing; loading data and recomputing prepared matrices...')
        analyzer.load_data()
        analyzer.prepare_analysis_data()

    df = compute_exclusive_rates(analyzer, denominator=args.denominator)
    csv_path = out_dir / 'exclusive_hit_rate.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f'Saved CSV: {csv_path}')

    fig_path = out_dir / 'exclusive_hit_rate.pdf'
    plot_bar(df, fig_path)
    logger.info(f'Saved figure: {fig_path}')


if __name__ == '__main__':
    main()
