#!/usr/bin/env python3
"""
Text–Visual entropy diagnostics (from cache) in one figure.

Left:  Scatter of per-query top-k entropy H_topk(Text) vs H_topk(Visual),
       with Spearman correlation in the title.
Right: Grouped histogram (bar chart) of Event-Hit@1(pair) vs entropy bins
       for Text and Visual only (shared bin edges).

Data source: cached prepared matrices produced by
  preprocess/analyze_multimodal_choice.py --use-cache
which stores probabilities and metadata in:
  analysis/<dataset>/multimodal_choice/cache/prepared_data.npz

This script reuses those cached probabilities and metadata only; it does not
recompute embeddings or similarities.

Usage examples:
  python draw/entropy_text_visual_combo.py --dataset FakeSV
  python draw/entropy_text_visual_combo.py \
      --cache analysis/FakeSV/multimodal_choice/cache/prepared_data.npz \
      --top-k 10 --n-bins 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr
except Exception:  # Fallback if scipy is unavailable
    spearmanr = None  # type: ignore


def topk_entropy_rows(P: np.ndarray, k: int, eps: float = 1e-12) -> np.ndarray:
    """Compute per-row entropy over the top-k probabilities.

    P: [Q, N] probabilities for a modality.
    Return: [Q] vector of entropies.
    """
    Q, N = P.shape
    k_eff = min(k, N)
    # Indices of top-k per row (unordered within the top-k slice)
    idx = np.argpartition(P, -k_eff, axis=1)[:, -k_eff:]
    # Gather the top-k probs per row
    rows = np.arange(Q)[:, None]
    sub = P[rows, idx]
    sub = sub / (sub.sum(axis=1, keepdims=True) + eps)
    # Entropy = -sum p log p (natural log)
    H = -(sub * (np.log(sub + eps))).sum(axis=1)
    return H.astype(np.float32)


def compute_event_hit_vector(P: np.ndarray, candidate_event: np.ndarray, candidate_label: np.ndarray,
                             query_events: np.ndarray) -> np.ndarray:
    """Event-Hit@1(pair) per query for a single modality probability matrix P.

    Matches the analyzer's logic:
      - Among positives (label==1) and negatives (label==0), find top-1 per side.
      - A hit if either top-1 positive or top-1 negative event matches the query event.
    Return: [Q] int32 vector {0,1}.
    """
    Q, N = P.shape
    hits = np.zeros(Q, dtype=np.int32)
    pos_mask = (candidate_label == 1)
    neg_mask = (candidate_label == 0)
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return hits

    pos_events = candidate_event[pos_mask]
    neg_events = candidate_event[neg_mask]

    # Pre-slice probabilities to avoid repeated masking inside the loop
    P_pos = P[:, pos_mask]
    P_neg = P[:, neg_mask]

    # Indices of top-1 within each slice
    top1_pos_idx = np.argmax(P_pos, axis=1)
    top1_neg_idx = np.argmax(P_neg, axis=1)

    for qi in range(Q):
        top1_pos_event = pos_events[top1_pos_idx[qi]]
        top1_neg_event = neg_events[top1_neg_idx[qi]]
        hits[qi] = int((top1_pos_event == query_events[qi]) or (top1_neg_event == query_events[qi]))
    return hits


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if spearmanr is None:
        # Simple rank correlation fallback without ties handling sophistication
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        rx = (rx - rx.mean()) / (rx.std() + 1e-12)
        ry = (ry - ry.mean()) / (ry.std() + 1e-12)
        return float(np.mean(rx * ry))
    else:
        rho, _ = spearmanr(x, y)
        if isinstance(rho, float):
            return float(0.0 if np.isnan(rho) else rho)
        return 0.0


def main():
    ap = argparse.ArgumentParser(description="Compose Text–Visual entropy diagnostics from cached prepared data")
    ap.add_argument('--dataset', type=str, default='FakeSV', help='Dataset name')
    ap.add_argument('--cache', type=str, default=None,
                    help='Path to prepared_data.npz (default: analysis/<dataset>/multimodal_choice/cache/prepared_data.npz)')
    ap.add_argument('--top-k', type=int, default=10, help='Top-k for entropy (must match analysis)')
    ap.add_argument('--n-bins', type=int, default=10, help='Number of entropy bins for histogram')
    ap.add_argument('--output', type=str, default=None,
                    help='Output PNG (default: analysis/<dataset>/draw/text_visual_entropy_combo.png)')
    args = ap.parse_args()

    if args.cache is None:
        args.cache = f"analysis/{args.dataset}/multimodal_choice/cache/prepared_data.npz"
    if args.output is None:
        args.output = f"analysis/{args.dataset}/draw/text_visual_entropy_combo.png"

    cache_path = Path(args.cache)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        print(f"ERROR: cache file not found: {cache_path}")
        sys.exit(1)

    # Load cached prepared matrices
    data = np.load(cache_path, allow_pickle=True)
    probs_T = data['probs_T']  # [Q, N]
    probs_I = data['probs_I']  # [Q, N]
    candidate_event = data['candidate_event']
    candidate_label = data['candidate_label']
    query_events = data['query_events']

    # Compute per-query top-k entropies
    H_T = topk_entropy_rows(probs_T, args.top_k)
    H_I = topk_entropy_rows(probs_I, args.top_k)

    # Left figure stats
    rho_TI = safe_spearman(H_T, H_I)

    # Compute Event-Hit@1 vectors
    hit_T = compute_event_hit_vector(probs_T, candidate_event, candidate_label, query_events)
    hit_I = compute_event_hit_vector(probs_I, candidate_event, candidate_label, query_events)

    # Build shared entropy bins across both modalities (uniform over combined range)
    H_min = float(min(H_T.min(), H_I.min()))
    H_max = float(max(H_T.max(), H_I.max()))
    # Add tiny epsilon to include max in last bin
    edges = np.linspace(H_min, H_max + 1e-8, args.n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def binned_hit_rates(H: np.ndarray, hit: np.ndarray) -> np.ndarray:
        rates = np.zeros(args.n_bins, dtype=float)
        for b in range(args.n_bins):
            if b < args.n_bins - 1:
                mask = (H >= edges[b]) & (H < edges[b + 1])
            else:
                mask = (H >= edges[b]) & (H <= edges[b + 1])
            if mask.any():
                rates[b] = float(hit[mask].mean())
            else:
                rates[b] = 0.0
        return rates

    rates_T = binned_hit_rates(H_T, hit_T)
    rates_I = binned_hit_rates(H_I, hit_I)

    # Compose figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter H_T vs H_I
    axes[0].scatter(H_T, H_I, alpha=0.25, s=8)
    axes[0].set_xlabel('H_topk T (Text)')
    axes[0].set_ylabel('H_topk I (Visual)')
    axes[0].set_title(f'Text vs Visual (Spearman={rho_TI:.2f})')
    axes[0].grid(True, alpha=0.3)

    # Right: grouped histogram (bar chart) of hit-rate vs entropy bins
    width = (centers[1] - centers[0]) * 0.4 if len(centers) > 1 else 0.05
    axes[1].bar(centers - width/2, rates_T, width=width, label='Text', color='tab:blue', alpha=0.8)
    axes[1].bar(centers + width/2, rates_I, width=width, label='Visual', color='tab:orange', alpha=0.8)
    axes[1].set_xlabel('Entropy (top-k bins, shared)')
    axes[1].set_ylabel('Event-Hit@1(pair)')
    axes[1].set_title('Event-Hit@1 vs Entropy (Histogram)')
    axes[1].set_xlim(edges[0], edges[-1])
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_path}")


if __name__ == '__main__':
    main()

