#!/usr/bin/env python3
"""
Compose a two-panel figure by re-drawing from cached matrices (no PNG reuse):
  - Left: Random Pairwise Cosine Similarity Distributions (T/I/A)
  - Right: kNN Jaccard Overlap heatmap across modalities (T-Visual, T-Audio, Visual-Audio)

Inputs (defaults derived from dataset):
  prepared_data.npz produced by preprocess/analyze_multimodal_choice.py --use-cache
    - expects keys: scores_T, scores_I, scores_A

Outputs:
  analysis/{dataset}/draw/randomcos_and_knnjaccard_combined.png

Example:
  python draw/compose_randomcos_knnjaccard_from_cache.py --dataset FakeSV
  python draw/compose_randomcos_knnjaccard_from_cache.py \
      --cache analysis/FakeSV/multimodal_choice/cache/prepared_data.npz \
      --k-values 5,10,20 --n-samples 5000
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_random_pairwise_samples(scores: np.ndarray, n_samples: int, rng: np.random.Generator):
    Q, N = scores.shape
    q_idx = rng.integers(0, Q, size=n_samples)
    c_idx = rng.integers(0, N, size=n_samples)
    return scores[q_idx, c_idx]


def mean_knn_jaccard(scores1: np.ndarray, scores2: np.ndarray, k: int) -> float:
    # Top-k indices per row using argpartition
    top1 = np.argpartition(scores1, -k, axis=1)[:, -k:]
    top2 = np.argpartition(scores2, -k, axis=1)[:, -k:]
    # Compute per-row jaccard
    jacc = []
    for a, b in zip(top1, top2):
        A, B = set(a.tolist()), set(b.tolist())
        inter = len(A & B)
        union = len(A | B)
        jacc.append(inter / union if union > 0 else 0.0)
    return float(np.mean(jacc))


def main():
    parser = argparse.ArgumentParser(description='Compose RandomCos and kNN-Jaccard from cache (no PNG reuse)')
    parser.add_argument('--dataset', type=str, default='FakeSV', help='Dataset name')
    parser.add_argument('--cache', type=str, default=None,
                        help='Path to prepared_data.npz (default: analysis/{dataset}/multimodal_choice/cache/prepared_data.npz)')
    parser.add_argument('--n-samples', type=int, default=5000, help='Random pairs sampled per modality')
    parser.add_argument('--k-values', type=str, default='5,10,20', help='Comma-separated k list for Jaccard')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG (default: analysis/{dataset}/draw/randomcos_and_knnjaccard_combined.png)')
    args = parser.parse_args()

    if args.cache is None:
        args.cache = f"analysis/{args.dataset}/multimodal_choice/cache/prepared_data.npz"
    if args.output is None:
        args.output = f"analysis/{args.dataset}/draw/randomcos_and_knnjaccard_combined.png"

    cache_path = Path(args.cache)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        print(f"ERROR: cache file not found: {cache_path}")
        sys.exit(1)

    data = np.load(cache_path, allow_pickle=True)
    # Required scores
    S_T = data['scores_T']
    S_I = data['scores_I']
    S_A = data['scores_A']

    rng = np.random.default_rng(args.seed)
    sns.set_palette('husl')

    # Prepare left panel data (random pairwise cosine)
    rp_text = compute_random_pairwise_samples(S_T, args.n_samples, rng)
    rp_visual = compute_random_pairwise_samples(S_I, args.n_samples, rng)
    rp_audio = compute_random_pairwise_samples(S_A, args.n_samples, rng)

    # Prepare right panel data (kNN Jaccard overlap means)
    k_list = [int(x) for x in args.k_values.split(',') if x.strip()]
    pair_names = ['Text-Visual', 'Text-Audio', 'Visual-Audio']
    jacc_means = np.zeros((len(k_list), len(pair_names)), dtype=float)
    for i, k in enumerate(k_list):
        jacc_means[i, 0] = mean_knn_jaccard(S_T, S_I, k)
        jacc_means[i, 1] = mean_knn_jaccard(S_T, S_A, k)
        jacc_means[i, 2] = mean_knn_jaccard(S_I, S_A, k)

    # Compose figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: violin of random pairwise cosine
    import pandas as pd
    df_left = pd.DataFrame({
        'Cosine_Similarity': np.concatenate([rp_text, rp_visual, rp_audio], axis=0),
        'Modality': (['Text'] * len(rp_text)) + (['Visual'] * len(rp_visual)) + (['Audio'] * len(rp_audio))
    })
    sns.violinplot(data=df_left, x='Modality', y='Cosine_Similarity', ax=axes[0])
    axes[0].set_title('Random Pairwise Cosine Similarity Distributions')
    axes[0].set_ylabel('Cosine Similarity')

    # Right: heatmap of kNN Jaccard
    im = axes[1].imshow(jacc_means, cmap='RdYlBu_r', aspect='auto')
    axes[1].set_title('kNN Jaccard Overlap (Cross-modal)')
    axes[1].set_xlabel('Modality Pairs')
    axes[1].set_ylabel('k Values')
    axes[1].set_xticks(range(len(pair_names)))
    axes[1].set_xticklabels(pair_names, rotation=45, ha='right')
    axes[1].set_yticks(range(len(k_list)))
    axes[1].set_yticklabels([f'k={k}' for k in k_list])
    for i in range(len(k_list)):
        for j in range(len(pair_names)):
            axes[1].text(j, i, f'{jacc_means[i, j]:.3f}', ha='center', va='center', color='black', fontweight='bold')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined figure to {out_path}")


if __name__ == '__main__':
    main()

