#!/usr/bin/env python3
"""
Compose a two-panel PDF from cached prepared data:
  - Left: 3D scatter of per-query top-k entropy H_k(Text), H_k(Visual), H_k(Audio)
  - Right: Grouped histogram of Event-Hit@1(pair) vs entropy bins for Text/Visual/Audio

Data source: analysis/<dataset>/multimodal_choice/cache/prepared_data.npz
  expected arrays: probs_T, probs_I, probs_A, candidate_event, candidate_label, query_events

Usage:
  python draw/compose_triax_entropy_and_hit_from_cache.py --dataset FakeSV
  python draw/compose_triax_entropy_and_hit_from_cache.py \
      --cache analysis/FakeSV/multimodal_choice/cache/prepared_data.npz \
      --top-k 10 --n-bins 10 \
      --output analysis/FakeSV/draw/entropy_triax_and_hit.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt


def topk_entropy_rows(P: np.ndarray, k: int, eps: float = 1e-12) -> np.ndarray:
    Q, N = P.shape
    k_eff = min(k, N)
    idx = np.argpartition(P, -k_eff, axis=1)[:, -k_eff:]
    rows = np.arange(Q)[:, None]
    sub = P[rows, idx]
    sub = sub / (sub.sum(axis=1, keepdims=True) + eps)
    H = -(sub * (np.log(sub + eps))).sum(axis=1)
    return H.astype(np.float32)


def compute_event_hit_vector(P: np.ndarray, candidate_event: np.ndarray, candidate_label: np.ndarray,
                             query_events: np.ndarray) -> np.ndarray:
    Q, N = P.shape
    hits = np.zeros(Q, dtype=np.int32)
    pos_mask = (candidate_label == 1)
    neg_mask = (candidate_label == 0)
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return hits
    pos_events = candidate_event[pos_mask]
    neg_events = candidate_event[neg_mask]
    P_pos = P[:, pos_mask]
    P_neg = P[:, neg_mask]
    top1_pos_idx = np.argmax(P_pos, axis=1)
    top1_neg_idx = np.argmax(P_neg, axis=1)
    for qi in range(Q):
        hits[qi] = int((pos_events[top1_pos_idx[qi]] == query_events[qi]) or
                        (neg_events[top1_neg_idx[qi]] == query_events[qi]))
    return hits


def main():
    ap = argparse.ArgumentParser(description='Compose 3D entropy scatter and entropy–hit histogram from cache')
    ap.add_argument('--dataset', type=str, default='FakeSV', help='Dataset name')
    ap.add_argument('--cache', type=str, default=None,
                    help='Path to prepared_data.npz (default: analysis/<dataset>/multimodal_choice/cache/prepared_data.npz)')
    ap.add_argument('--top-k', type=int, default=10, help='Top-k for entropy')
    ap.add_argument('--n-bins', type=int, default=10, help='Number of entropy bins for histogram')
    ap.add_argument('--output', type=str, default=None,
                    help='Output PDF (default: analysis/<dataset>/draw/entropy_triax_and_hit.pdf)')
    args = ap.parse_args()

    if args.cache is None:
        args.cache = f"analysis/{args.dataset}/multimodal_choice/cache/prepared_data.npz"
    if args.output is None:
        args.output = f"analysis/{args.dataset}/draw/entropy_triax_and_hit.pdf"

    cache_path = Path(args.cache)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        print(f"ERROR: cache file not found: {cache_path}")
        sys.exit(1)

    data = np.load(cache_path, allow_pickle=True)
    # Required arrays
    probs_T = data['probs_T']
    probs_I = data['probs_I']
    probs_A = data['probs_A']
    candidate_event = data['candidate_event']
    candidate_label = data['candidate_label']
    query_events = data['query_events']

    # Entropies per query
    H_T = topk_entropy_rows(probs_T, args.top_k)
    H_I = topk_entropy_rows(probs_I, args.top_k)
    H_A = topk_entropy_rows(probs_A, args.top_k)

    # Hit vectors per modality
    hit_T = compute_event_hit_vector(probs_T, candidate_event, candidate_label, query_events)
    hit_I = compute_event_hit_vector(probs_I, candidate_event, candidate_label, query_events)
    hit_A = compute_event_hit_vector(probs_A, candidate_event, candidate_label, query_events)

    # Shared entropy bin edges across T/I/A
    H_min = float(min(H_T.min(), H_I.min(), H_A.min()))
    H_max = float(max(H_T.max(), H_I.max(), H_A.max()))
    edges = np.linspace(H_min, H_max + 1e-8, args.n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def binned_hit_rates(H: np.ndarray, hit: np.ndarray) -> np.ndarray:
        rates = np.zeros(args.n_bins, dtype=float)
        for b in range(args.n_bins):
            mask = (H >= edges[b]) & (H < edges[b + 1]) if b < args.n_bins - 1 else (H >= edges[b]) & (H <= edges[b + 1])
            rates[b] = float(hit[mask].mean()) if mask.any() else 0.0
        return rates

    rates_T = binned_hit_rates(H_T, hit_T)
    rates_I = binned_hit_rates(H_I, hit_I)
    rates_A = binned_hit_rates(H_A, hit_A)

    # Build figure (PDF)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(11, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])

    # Left: 3D scatter H_k(T), H_k(V), H_k(A)
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax3d.scatter(H_T, H_I, H_A, s=5, alpha=0.28, c='#1f77b4')
    ax3d.set_xlabel('Text', labelpad=6)
    ax3d.set_ylabel('Visual', labelpad=6)
    ax3d.set_zlabel('Audio', labelpad=10)
    # Tight ranges
    vmin = 0.0
    vmax = float(max(H_T.max(), H_I.max(), H_A.max()))
    ax3d.set_xlim(vmin, vmax)
    ax3d.set_ylim(vmin, vmax)
    ax3d.set_zlim(vmin, vmax)
    try:
        ax3d.set_box_aspect((1, 1, 0.7))
        ax3d.zaxis.set_rotate_label(False)
        ax3d.zaxis.set_label_coords(0.5, 1.02)
    except Exception:
        pass
    ax3d.tick_params(labelsize=8, pad=1)
    ax3d.view_init(elev=18, azim=30)
    ax3d.grid(True)

    # Right: grouped histogram of hit rate vs entropy bins (T/I/A)
    axh = fig.add_subplot(gs[0, 1])
    width = (centers[1] - centers[0]) * 0.28 if len(centers) > 1 else 0.05
    axh.bar(centers - width, rates_T, width=width, label='Text', color='tab:blue', alpha=0.85)
    axh.bar(centers,        rates_I, width=width, label='Visual', color='tab:orange', alpha=0.85)
    axh.bar(centers + width, rates_A, width=width, label='Audio', color='tab:green', alpha=0.85)
    axh.set_xlabel('Entropy (top-k bins, shared)')
    axh.set_ylabel('Event-Hit@1(pair)')
    axh.set_xlim(edges[0], edges[-1])
    axh.grid(True, axis='y', alpha=0.3)
    axh.legend()

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.92, wspace=0.26)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved PDF: {out_path}")


if __name__ == '__main__':
    main()

