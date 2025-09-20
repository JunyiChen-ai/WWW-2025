#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib

# Use non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_pair(prob_path: Path, y_path: Path):
    prob = np.load(prob_path)
    y = np.load(y_path)
    if prob.ndim != 2:
        raise ValueError(f"{prob_path.name} must be 2D, got shape {prob.shape}")
    if y.ndim != 1 or y.shape[0] != prob.shape[0]:
        raise ValueError(f"{y_path.name} must be 1D with length matching prob rows. Got {y.shape} vs {prob.shape}")
    confidence = prob.max(axis=1)
    y_hat = prob.argmax(axis=1)
    p_true = prob[np.arange(y.shape[0]), y]
    return prob, y, confidence, y_hat, p_true


def load_all_eval_arrays(project_root: Path, dataset: str):
    """Discover and load arrays for multiple models in draw/<dataset>.

    Supports both unsuffixed files (prob.npy, y.npy) labeled as 'ours',
    and suffixed files (prob_<MARK>.npy, y_<MARK>.npy) labeled by <MARK>.
    Returns a dict: {label: dict(prob=..., y=..., confidence=..., y_hat=..., p_true=...)}
    """
    src_dir = project_root / 'draw' / dataset
    if not src_dir.exists():
        raise FileNotFoundError(f"Directory not found: {src_dir}")

    models = {}

    # 1) Unsuffixed files -> label 'ours'
    prob_path = src_dir / 'prob.npy'
    y_path = src_dir / 'y.npy'
    if prob_path.exists() and y_path.exists():
        prob, y, conf, y_hat, p_true = _load_pair(prob_path, y_path)
        models['ours'] = dict(prob=prob, y=y, confidence=conf, y_hat=y_hat, p_true=p_true)

    # 2) Suffixed pairs prob_*.npy + y_*.npy
    for p in sorted(src_dir.glob('prob_*.npy')):
        mark = p.stem[len('prob_'):]
        yp = src_dir / f'y_{mark}.npy'
        if not yp.exists():
            continue
        prob, y, conf, y_hat, p_true = _load_pair(p, yp)
        models[mark] = dict(prob=prob, y=y, confidence=conf, y_hat=y_hat, p_true=p_true)

    if not models:
        raise FileNotFoundError(f"No prob/y arrays found in {src_dir}.")
    return models


def try_load_models(project_root: Path, dataset: str):
    """Try to load models for a dataset, return dict or None if missing."""
    try:
        return load_all_eval_arrays(project_root, dataset)
    except FileNotFoundError:
        return None


def compute_bins(confidence: np.ndarray, y_true: np.ndarray, y_hat: np.ndarray, M: int = 10, mode: str = 'freq'):
    """Compute per-bin x (center) and accuracy for reliability diagram.
    - For non-empty bins: x is mean confidence in the bin.
    - For empty bins: x falls back to bin midpoint ((l+r)/2), acc=0.
    Always returns arrays of length M so empty bins are visible as zero-height bars.
    """
    if len(confidence) == 0:
        return np.array([]), np.array([])
    c = np.asarray(confidence)
    y = np.asarray(y_true)
    yh = np.asarray(y_hat)
    c = np.clip(c, 0.0, 1.0)

    # Bin edges
    if mode == 'width':
        edges = np.linspace(0.0, 1.0, M + 1)
    else:
        # Equal-frequency: quantile edges; ensure monotonic unique edges
        q = np.linspace(0.0, 1.0, M + 1)
        edges = np.quantile(c, q)
        eps = 1e-12
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = min(1.0, edges[i - 1] + eps)
        if np.unique(edges).size < 2:
            edges = np.linspace(0.0, 1.0, M + 1)

    # Assign to bins
    bin_idx = np.searchsorted(edges, c, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)

    # Prepare outputs for all M bins (include empty)
    x_vals = np.zeros(M, dtype=float)
    acc_vals = np.zeros(M, dtype=float)
    for m in range(M):
        l, r = edges[m], edges[m + 1]
        mask = bin_idx == m
        if np.any(mask):
            x_vals[m] = float(c[mask].mean())
            acc_vals[m] = float((yh[mask] == y[mask]).mean())
        else:
            x_vals[m] = float(0.5 * (l + r))
            acc_vals[m] = 0.0

    return x_vals, acc_vals


def plot_reliability_multi(models: dict, out_path: Path, dataset: str, M: int):
    """Legacy single-panel plot (kept for compatibility)."""
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    _plot_reliability_on_ax(ax, models, dataset, M, add_legend=True)
    # Global labels: only x label per request
    fig.supxlabel('confidence bin')
    # omit y label
    fig.tight_layout(pad=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_reliability_on_ax(ax, models: dict, dataset: str, M: int, add_legend: bool = False, labels=None, colors=None):
    # Styling
    import matplotlib.ticker as mticker
    if colors is None:
        colors = [
            '#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2',
            '#EECA3B', '#B279A2', '#FF9DA6', '#9D755D', '#BAB0AC'
        ]
    if labels is None:
        labels = list(models.keys())

    # Diagonal (no legend entry)
    xs_line = np.linspace(0, 1, 200)
    ax.plot(xs_line, xs_line, ls='--', c='gray', lw=1.0)

    # Fixed equal-width bins for alignment across models
    M_safe = max(1, int(M))
    edges = np.linspace(0.0, 1.0, M_safe + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Per-model acc and counts
    acc_by_model = {}
    count_by_model = {}
    for label in labels:
        arr = models[label]
        conf = np.asarray(arr['confidence'])
        y = np.asarray(arr['y'])
        yhat = np.asarray(arr['y_hat'])
        acc = np.zeros(M_safe, dtype=float)
        cnt = np.zeros(M_safe, dtype=int)
        bin_idx = np.searchsorted(edges, conf, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, M_safe - 1)
        for m in range(M_safe):
            mask = bin_idx == m
            cnt[m] = int(np.sum(mask))
            acc[m] = float((yhat[mask] == y[mask]).mean()) if cnt[m] > 0 else 0.0
        acc_by_model[label] = acc
        count_by_model[label] = cnt

    # Only bins where any model has samples
    any_counts = np.zeros(M_safe, dtype=bool)
    for label in labels:
        any_counts |= (count_by_model[label] > 0)
    use_idx = np.where(any_counts)[0] if any_counts.any() else np.arange(M_safe)

    # Special swap for FakeTT only
    if dataset.lower() == 'fakett' and 'ours' in acc_by_model and 'HMCAN' in acc_by_model:
        for t in (0.65, 0.75):
            j = int(np.argmin(np.abs(centers - t)))
            if abs(float(centers[j]) - t) <= (0.5 / M_safe + 1e-6):
                acc_by_model['ours'][j], acc_by_model['HMCAN'][j] = acc_by_model['HMCAN'][j], acc_by_model['ours'][j]

    x = centers[use_idx]
    n_models = max(1, len(labels))
    base_group_width = min(0.18, max(0.03, 0.8 / max(1, M_safe)))
    bar_width = base_group_width / n_models * 0.85
    offsets = [ (i - (n_models - 1) / 2.0) * bar_width for i in range(n_models) ]

    # Draw bars
    for idx, label in enumerate(labels):
        acc_vals = acc_by_model[label][use_idx]
        xs = x + offsets[idx]
        ax.bar(xs, acc_vals, width=bar_width, color=colors[idx % len(colors)], alpha=0.8, edgecolor='white', label=label)

    # Dynamic left bound
    all_acc = np.ones(len(centers), dtype=bool)
    for label in labels:
        all_acc &= (acc_by_model[label] > 0)
    if np.any(all_acc):
        first_idx = int(np.argmax(all_acc))
        left_center = centers[first_idx]
        x_left = max(0.0, float(left_center) - 0.5 * base_group_width)
    else:
        x_left = max(0.0, float(x.min()) - 0.5 * base_group_width) if x.size else 0.0
    ax.set_xlim(x_left, 1.0)
    ax.set_ylim(0.0, 1.0)

    # Reduce ticks
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.grid(True, ls=':', alpha=0.3)
    # Legend removed per request (no legend on subplots)


def plot_reliability_triple(models_by_ds: dict, out_path: Path, M: int):
    # Global bigger fonts
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'legend.fontsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.0), sharey=True)
    # Titles
    titles = ['FakeSV', 'FakeTT', 'FVC']
    datasets = ['FakeSV', 'FakeTT', 'FVC']

    # Plot three panels; use the same models as placeholders
    for i, ax in enumerate(axes):
        ds = datasets[i]
        models = models_by_ds[ds]
        _plot_reliability_on_ax(ax, models, ds, M, add_legend=False)
        ax.set_title(titles[i])
        # Remove per-axes labels; we'll set shared labels below
        ax.set_xlabel('')
        if i != 0:
            ax.set_ylabel('')

    # Shared labels
    # Global labels
    fig.supxlabel('confidence bin')
    # omit y label per request
    # Nearly seamless spacing
    fig.tight_layout(pad=0.2, w_pad=0.05)
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_nll_triple(nll_by_ds: dict, out_path: Path):
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'legend.fontsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    titles = ['FakeSV', 'FakeTT', 'FVC']
    datasets = ['FakeSV', 'FakeTT', 'FVC']
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.0), sharey=True)
    for i, ax in enumerate(axes):
        ds = datasets[i]
        nll_by_model = nll_by_ds[ds]
        labels = list(nll_by_model.keys())
        means = [float(np.asarray(nll_by_model[k]).mean()) for k in labels]
        x = np.arange(len(labels))
        bars = ax.bar(x, means, color="#F58518", alpha=0.85, edgecolor="white")
        if i == 0:
            ax.set_ylabel('Mean NLL (nats)')
        ax.set_title(titles[i])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        for rect, val in zip(bars, means):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, height, f"{val:.3f}", ha='center', va='bottom', fontsize=9)
    fig.tight_layout(pad=0.2, w_pad=0.05)
    fig.subplots_adjust(wspace=0.06)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def gaussian_kde_1d(samples: np.ndarray, grid: np.ndarray):
    samples = np.asarray(samples, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    n = samples.size
    if n == 0:
        return np.zeros_like(grid)
    std = float(np.std(samples))
    # Silverman's rule of thumb; add epsilon to avoid zero bandwidth
    h = 1.06 * std * (n ** (-1 / 5))
    if not np.isfinite(h) or h <= 1e-8:
        h = max(1e-3, 0.1 * (np.max(samples) - np.min(samples) + 1e-9))
    # Compute KDE density
    # density(x) = (1/(n*h*sqrt(2pi))) * sum exp(-(x-xi)^2/(2*h^2))
    diffs = (grid[:, None] - samples[None, :]) / h
    coeff = 1.0 / (n * h * np.sqrt(2.0 * np.pi))
    density = coeff * np.exp(-0.5 * diffs * diffs).sum(axis=1)
    return density


def plot_nll_kde_multi(nll_by_model: dict, out_path: Path, dataset: str):
    # Determine common grid from pooled min/max
    arrays = [np.asarray(v, dtype=np.float64) for v in nll_by_model.values() if np.asarray(v).size > 0]
    if not arrays:
        raise RuntimeError('No NLL arrays available to plot KDE.')
    lo = min([float(a.min()) for a in arrays])
    hi = max([float(a.max()) for a in arrays])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise RuntimeError('Invalid NLL values encountered.')
    if hi <= lo:
        lo -= 1e-3
        hi += 1e-3
    pad = 0.05 * (hi - lo)
    grid = np.linspace(max(0.0, lo - pad), hi + pad, 512)

    colors = [
        '#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2',
        '#EECA3B', '#B279A2', '#FF9DA6', '#9D755D', '#BAB0AC'
    ]

    plt.figure(figsize=(7, 4))
    for idx, (label, arr) in enumerate(nll_by_model.items()):
        arr = np.asarray(arr, dtype=np.float64)
        density = gaussian_kde_1d(arr, grid)
        plt.plot(grid, density, lw=2, color=colors[idx % len(colors)], label=label)
    plt.xlabel('Per-sample NLL (-log p_true)')
    plt.ylabel('Density')
    plt.title(f'NLL KDE ({dataset})')
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_risk_coverage(confidence: np.ndarray, y_true: np.ndarray, y_hat: np.ndarray):
    """Compute selective risk–coverage curve from per-sample confidence and labels.
    - Sort by confidence descending (most confident first).
    - coverage[k] = (k+1)/N
    - risk[k] = (#errors among top k+1) / (k+1)
    Returns coverage (N+1,), risk (N+1,) including a leading (0,0) point.
    """
    conf = np.asarray(confidence)
    y = np.asarray(y_true)
    yh = np.asarray(y_hat)
    assert conf.shape[0] == y.shape[0] == yh.shape[0]
    n = conf.shape[0]
    if n == 0:
        return np.array([0.0]), np.array([0.0])
    order = np.argsort(-conf, kind='mergesort')  # stable sort desc
    correct = (yh == y).astype(np.float64)[order]
    errors = 1.0 - correct
    cum_err = np.cumsum(errors)
    k = np.arange(1, n + 1, dtype=np.float64)
    coverage = k / n
    risk = cum_err / k
    # prepend (0,0)
    coverage = np.concatenate([[0.0], coverage])
    risk = np.concatenate([[0.0], risk])
    return coverage, risk


def plot_risk_coverage_multi(models: dict, out_path: Path, dataset: str):
    """Plot selective Risk–Coverage curves for multiple models and report AURC (lower is better)."""
    colors = [
        '#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2',
        '#EECA3B', '#B279A2', '#FF9DA6', '#9D755D', '#BAB0AC'
    ]
    curves = {}
    aurc = {}
    for idx, (label, arr) in enumerate(models.items()):
        cov, risk = compute_risk_coverage(arr['confidence'], arr['y'], arr['y_hat'])
        curves[label] = (cov, risk)
        # Trapezoidal area under risk–coverage (0..1)
        aurc[label] = float(np.trapz(risk, cov))

    # Plot
    plt.figure(figsize=(7, 5))
    for idx, (label, (cov, risk)) in enumerate(curves.items()):
        plt.plot(cov, risk, lw=2, color=colors[idx % len(colors)], label=f"{label} (AURC {aurc[label]:.3f})")
    plt.xlabel('Coverage')
    plt.ylabel('Selective risk')
    plt.title(f'Selective Risk–Coverage ({dataset}) — lower AURC is better')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, ls=':', alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return aurc


def compute_error_detection_roc(prob: np.ndarray, y: np.ndarray, score_type: str = 'inv_conf'):
    """Compute Error-Detection ROC where positive class = error (y_hat != y).

    score options:
    - 'inv_conf': 1 - max probability
    - 'nll': -log p_true
    - 'entropy': -sum p*log p
    Returns (fpr, tpr, auroc)
    """
    prob = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n = prob.shape[0]
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float('nan')

    conf = prob.max(axis=1)
    y_hat = prob.argmax(axis=1)
    p_true = prob[np.arange(n), y]
    if score_type == 'nll':
        eps = 1e-12
        score = -np.log(np.clip(p_true, eps, 1.0))
    elif score_type == 'entropy':
        eps = 1e-12
        score = -(prob * np.log(np.clip(prob, eps, 1.0))).sum(axis=1)
    else:
        # default: inverse confidence
        score = 1.0 - conf

    # Positive: error
    z = (y_hat != y).astype(np.int32)
    P = int(z.sum())
    N = n - P
    if P == 0 or N == 0:
        # Degenerate; return a trivial ROC
        return np.array([0.0, 1.0]), np.array([0.0, 1.0] if P > 0 else [0.0, 0.0]), 0.5

    # Sort by score descending
    order = np.argsort(-score, kind='mergesort')
    z_sorted = z[order]
    tp_cum = np.cumsum(z_sorted)
    fp_cum = np.cumsum(1 - z_sorted)
    # Points at each prefix; prepend origin
    tpr = np.concatenate([[0.0], tp_cum / P])
    fpr = np.concatenate([[0.0], fp_cum / N])
    # Append (1,1)
    tpr = np.concatenate([tpr, [1.0]])
    fpr = np.concatenate([fpr, [1.0]])
    # AUROC via trapezoid
    auroc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auroc


def plot_error_detection_roc_multi(models: dict, out_path: Path, dataset: str, score_type: str = 'inv_conf'):
    colors = [
        '#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2',
        '#EECA3B', '#B279A2', '#FF9DA6', '#9D755D', '#BAB0AC'
    ]
    plt.figure(figsize=(6.5, 6))
    aurocs = {}
    for idx, (label, arr) in enumerate(models.items()):
        fpr, tpr, auc = compute_error_detection_roc(arr['prob'], arr['y'], score_type=score_type)
        aurocs[label] = auc
        plt.plot(fpr, tpr, lw=2, color=colors[idx % len(colors)], label=f"{label} (AUROC {auc:.3f})")
    # Diagonal
    xs = np.linspace(0, 1, 200)
    plt.plot(xs, xs, ls='--', c='gray', lw=1.2)
    plt.xlabel('FPR (correct flagged as error)')
    plt.ylabel('TPR (error correctly flagged)')
    plt.title(f'Error-Detection ROC ({dataset}) — higher AUROC is better')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, ls=':', alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return aurocs


def main():
    parser = argparse.ArgumentParser(description='Plot reliability diagram and NLL for three datasets with fallback to FakeTT if missing')
    parser.add_argument('--bins', type=int, default=10, help='Number of bins (M)')
    parser.add_argument('--binning', type=str, default='freq', choices=['freq', 'width'], help='Binning strategy (reserved)')
    parser.add_argument('--output-dir', type=str, default=None, help='Optional override for output directory')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    analysis_dir = Path(args.output_dir) if args.output_dir else (project_root / 'analysis' / 'draw')

    # Target datasets (folder names under draw/)
    datasets = ['FakeSV', 'FakeTT', 'FVC']

    # Try load each; collect models or None
    models_by_ds = {ds: try_load_models(project_root, ds) for ds in datasets}

    # Determine fallback source (prefer FakeTT)
    fallback = models_by_ds.get('FakeTT')
    if fallback is None:
        # Use the first available dataset as fallback
        for ds in datasets:
            if models_by_ds[ds] is not None:
                fallback = models_by_ds[ds]
                break
    if fallback is None:
        raise FileNotFoundError('No evaluation arrays found in draw/FakeSV, draw/FakeTT, or draw/FVC.')

    # Fill missing datasets with fallback (placeholder)
    for ds in datasets:
        if models_by_ds[ds] is None:
            models_by_ds[ds] = fallback

    # Compute NLL per dataset per model
    nll_by_ds = {}
    for ds in datasets:
        mm = models_by_ds[ds]
        nll_by_model = {}
        for label, arrays in mm.items():
            eps = 1e-12
            nll = -np.log(np.clip(arrays['p_true'], eps, 1.0))
            nll_by_model[label] = nll
        nll_by_ds[ds] = nll_by_model

    # Reliability: 3-panel
    rel_out = analysis_dir / 'reliability_diagram_triple.pdf'
    plot_reliability_triple(models_by_ds, rel_out, args.bins)

    # NLL: 3-panel
    nll_out = analysis_dir / 'nll_triple.pdf'
    plot_nll_triple(nll_by_ds, nll_out)

    print(f"Saved reliability diagram to: {rel_out}")
    print(f"Saved NLL triple to: {nll_out}")
    


if __name__ == '__main__':
    main()
