#!/usr/bin/env python3
"""
Compose a two-panel figure using cached computations (no PNG reuse):
  - Left: Principal Angle Cosines between modality subspaces (Text–Vision, Text–Audio, Vision–Audio)
  - Right: kNN Jaccard Overlap heatmap across modalities (Text-Visual, Text-Audio, Visual-Audio)

Inputs (defaults):
  - principal angles cache: analysis/{dataset}/multimodal_choice/cache/principal_angles_*.npz
    expected keys: cos_TV, cos_TA, cos_VA, R_eff
  - prepared_data cache:   analysis/{dataset}/multimodal_choice/cache/prepared_data.npz
    expected keys: scores_T, scores_I, scores_A

Output:
  analysis/{dataset}/draw/angles_and_knnjaccard_combined.pdf

Examples:
  python draw/compose_angles_knnjaccard_from_cache.py --dataset FakeSV
  python draw/compose_angles_knnjaccard_from_cache.py \
      --angles-cache analysis/FakeSV/multimodal_choice/cache/principal_angles_raw_OFA-Sys...npz \
      --prepared-cache analysis/FakeSV/multimodal_choice/cache/prepared_data.npz \
      --k-values 5,10,20 --output analysis/FakeSV/draw/angles_and_knnjaccard_combined.pdf
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

# =====================
# Global plot parameters (small but legible; macaron palette)
# =====================
FIGSIZE = (5.2, 2.0)  # compact but with clear separation
FONT_SIZE = 9
AXES_LABEL_SIZE = 10
TICK_LABEL_SIZE = 9
LEGEND_SIZE = 9
GRID_ALPHA = 0.25
LINE_WIDTH = 1.8

# Pastel (macaron) colors for the three lines
LINE_COLORS = ['#A8D8EA', '#FFD8A8', '#CDEAC0']  # TV, TA, VA

# Heatmap colormap (pastel)
HEATMAP_CMAP = 'Pastel1'

# PDF font embedding
PDF_FONTTYPE = 42  # embed TrueType for crisp text

# Layout tweaks
LEFT_MARGIN = 0.10
WSPACE = 0.3  # inter-panel spacing
HSPACE = 0.02

# Whether to show a colorbar for the heatmap
SHOW_HEATMAP_COLORBAR = False


def find_latest_angles_cache(cache_dir: Path) -> Path | None:
    cand = sorted(cache_dir.glob('principal_angles_*.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0] if cand else None


def mean_knn_jaccard(scores1: np.ndarray, scores2: np.ndarray, k: int) -> float:
    top1 = np.argpartition(scores1, -k, axis=1)[:, -k:]
    top2 = np.argpartition(scores2, -k, axis=1)[:, -k:]
    jacc = []
    for a, b in zip(top1, top2):
        A, B = set(a.tolist()), set(b.tolist())
        inter = len(A & B)
        union = len(A | B)
        jacc.append(inter / union if union > 0 else 0.0)
    return float(np.mean(jacc))


def main():
    parser = argparse.ArgumentParser(description='Compose principal-angles and kNN-Jaccard from cache')
    parser.add_argument('--dataset', type=str, default='FakeSV', help='Dataset name')
    parser.add_argument('--angles-cache', type=str, default=None, help='Path to principal_angles_*.npz')
    parser.add_argument('--prepared-cache', type=str, default=None, help='Path to prepared_data.npz')
    parser.add_argument('--k-values', type=str, default='5,10,20', help='Comma-separated k list for Jaccard')
    parser.add_argument('--output', type=str, default=None, help='Output PDF path')
    parser.add_argument('--r-start', type=int, default=10, help='Start plotting from subspace dim r (1-based)')
    args = parser.parse_args()

    base_dir = Path(f"analysis/{args.dataset}/multimodal_choice")
    cache_dir = base_dir / 'cache'
    draw_out = Path(f"analysis/{args.dataset}/draw")
    draw_out.mkdir(parents=True, exist_ok=True)

    if args.angles_cache is None:
        # pick the most recent principal_angles cache
        angles_path = find_latest_angles_cache(cache_dir)
        if angles_path is None:
            print(f"ERROR: No principal_angles_*.npz found in {cache_dir}")
            sys.exit(1)
    else:
        angles_path = Path(args.angles_cache)

    if args.prepared_cache is None:
        prepared_path = cache_dir / 'prepared_data.npz'
    else:
        prepared_path = Path(args.prepared_cache)

    if args.output is None:
        out_path = draw_out / 'angles_and_knnjaccard_combined.pdf'
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if not angles_path.exists():
        print(f"ERROR: angles cache not found: {angles_path}")
        sys.exit(1)
    if not prepared_path.exists():
        print(f"ERROR: prepared cache not found: {prepared_path}")
        sys.exit(1)

    # Load caches
    ang = np.load(angles_path)
    cos_TV = ang['cos_TV']
    cos_TA = ang['cos_TA']
    cos_VA = ang['cos_VA']
    R_eff = int(ang.get('R_eff', len(cos_TV)))
    r_max = min(R_eff, len(cos_TV))
    r_start = max(1, int(args.r_start))
    if r_start > r_max:
        print(f"WARNING: r-start ({r_start}) > available R ({r_max}); using r-start=1")
        r_start = 1
    start_idx = r_start - 1  # convert to 0-based index
    r_axis = np.arange(r_start, r_max + 1)

    data = np.load(prepared_path, allow_pickle=True)
    S_T = data['scores_T']
    S_I = data['scores_I']
    S_A = data['scores_A']

    # kNN jaccard means
    k_list = [int(x) for x in args.k_values.split(',') if x.strip()]
    pair_names = ['Text-Visual', 'Text-Audio', 'Visual-Audio']
    jacc_means = np.zeros((len(k_list), len(pair_names)), dtype=float)
    for i, k in enumerate(k_list):
        jacc_means[i, 0] = mean_knn_jaccard(S_T, S_I, k)
        jacc_means[i, 1] = mean_knn_jaccard(S_T, S_A, k)
        jacc_means[i, 2] = mean_knn_jaccard(S_I, S_A, k)

    # Compose figure
    # Global styling
    mpl.rcParams.update({
        'pdf.fonttype': PDF_FONTTYPE,
        'font.size': FONT_SIZE,
        'axes.labelsize': AXES_LABEL_SIZE,
        'xtick.labelsize': TICK_LABEL_SIZE,
        'ytick.labelsize': TICK_LABEL_SIZE,
        'legend.fontsize': LEGEND_SIZE,
    })

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    # Left: principal angle cosines (dynamic y-limit close to max)
    cos_TV_sel = cos_TV[start_idx:start_idx+len(r_axis)]
    cos_TA_sel = cos_TA[start_idx:start_idx+len(r_axis)]
    cos_VA_sel = cos_VA[start_idx:start_idx+len(r_axis)]
    axes[0].plot(r_axis, cos_TV_sel, label='T–V', linewidth=LINE_WIDTH, color=LINE_COLORS[0])
    axes[0].plot(r_axis, cos_TA_sel, label='T–A', linewidth=LINE_WIDTH, color=LINE_COLORS[1])
    axes[0].plot(r_axis, cos_VA_sel, label='V–A', linewidth=LINE_WIDTH, color=LINE_COLORS[2])
    axes[0].set_xlabel('Subspace dimension r')
    axes[0].set_ylabel('cos(θ_r)')
    y_max = float(max(cos_TV_sel.max(initial=0.0), cos_TA_sel.max(initial=0.0), cos_VA_sel.max(initial=0.0)))
    axes[0].set_ylim(0.0, min(1.0, y_max + 0.02))
    axes[0].grid(True, alpha=GRID_ALPHA)
    # Compact legend in upper-right of the left subplot
    axes[0].legend(frameon=False, loc='upper right')

    # Right: heatmap of kNN Jaccard
    im = axes[1].imshow(jacc_means, cmap=HEATMAP_CMAP, aspect='auto')
    # X label restored per request; align with x-ticks
    axes[1].set_xlabel('Modality Pairs')
    axes[1].set_ylabel('')
    axes[1].set_xticks(range(len(pair_names)))
    axes[1].set_xticklabels(['T–V', 'T–A', 'V–A'], rotation=0, ha='center')
    axes[1].set_yticks(range(len(k_list)))
    axes[1].set_yticklabels([f'k={k}' for k in k_list])
    # Per-cell numeric annotations for readability
    for i in range(len(k_list)):
        for j in range(len(pair_names)):
            val = jacc_means[i, j]
            axes[1].text(j, i, f'{val:.3f}', ha='center', va='center',
                         color='black', fontsize=7, fontweight='bold')
    # Optional compact colorbar
    if SHOW_HEATMAP_COLORBAR:
        fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.02)

    # Remove titles per request
    for ax in axes:
        ax.set_title("")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined figure to {out_path}\n  angles_cache={angles_path}\n  prepared_cache={prepared_path}")


if __name__ == '__main__':
    main()
