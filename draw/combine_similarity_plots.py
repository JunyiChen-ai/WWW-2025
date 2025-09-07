#!/usr/bin/env python3
"""
Combine two existing similarity figures into a single side-by-side panel.

Defaults (no hard-coding to absolute paths):
  - input1: analysis/{dataset}/multimodal_choice/random_pairwise_cosine.png
  - input2: analysis/{dataset}/multimodal_choice/knn_jaccard_overlap.png
  - output: analysis/{dataset}/draw/similarity_combo.png

Usage examples:
  python draw/combine_similarity_plots.py --dataset FakeSV
  python draw/combine_similarity_plots.py \
    --input1 analysis/FakeSV/multimodal_choice/random_pairwise_cosine.png \
    --input2 analysis/FakeSV/multimodal_choice/knn_jaccard_overlap.png \
    --output analysis/FakeSV/draw/similarity_combo.png
"""

import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Combine two similarity figures into one panel')
    parser.add_argument('--dataset', type=str, default='FakeSV', help='Dataset name')
    parser.add_argument('--input1', type=str, default=None,
                        help='Path to random_pairwise_cosine.png')
    parser.add_argument('--input2', type=str, default=None,
                        help='Path to knn_jaccard_overlap.png')
    parser.add_argument('--output', type=str, default=None, help='Output path for combined figure')
    parser.add_argument('--title1', type=str, default='Random Pairwise Cosine Similarity')
    parser.add_argument('--title2', type=str, default='kNN Jaccard Overlap (Cross-modal)')
    args = parser.parse_args()

    # Resolve defaults relative to dataset
    if args.input1 is None:
        args.input1 = f"analysis/{args.dataset}/multimodal_choice/random_pairwise_cosine.png"
    if args.input2 is None:
        args.input2 = f"analysis/{args.dataset}/multimodal_choice/knn_jaccard_overlap.png"
    if args.output is None:
        args.output = f"analysis/{args.dataset}/draw/similarity_combo.png"

    p1 = Path(args.input1)
    p2 = Path(args.input2)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not p1.exists():
        print(f"ERROR: input1 not found: {p1}")
        sys.exit(1)
    if not p2.exists():
        print(f"ERROR: input2 not found: {p2}")
        sys.exit(1)

    img1 = plt.imread(str(p1))
    img2 = plt.imread(str(p2))

    # Create side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title(args.title1)
    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title(args.title2)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined figure to {out}")


if __name__ == '__main__':
    main()

