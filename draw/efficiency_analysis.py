import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import cm


def _use_cavolini_font():
    """Try to use Cavolini font if available or provided locally.
    Looks for a system-installed font named 'Cavolini' or a local TTF file
    placed in either the draw/ folder or analysis/ folder.
    """
    target = "Cavolini"
    try:
        # If a Cavolini family is already registered (case-insensitive), use it
        for f in fm.fontManager.ttflist:
            if "cavolini" in f.name.lower():
                plt.rcParams["font.family"] = f.name
                return True
    except Exception:
        pass

    script_dir = os.path.dirname(os.path.abspath(__file__))
    def iter_font_files(d):
        try:
            for fname in os.listdir(d):
                if "cavolini" in fname.lower() and os.path.splitext(fname)[1].lower() in {".ttf", ".otf", ".ttc", ".otc"}:
                    yield os.path.join(d, fname)
        except Exception:
            return

    for p in list(iter_font_files(script_dir)) + list(iter_font_files(os.path.normpath(os.path.join(script_dir, "..", "analysis")))):
        try:
            fm.fontManager.addfont(p)
        except Exception:
            continue

    # After adding, search again for a Cavolini family name
    try:
        for f in fm.fontManager.ttflist:
            if "cavolini" in f.name.lower():
                plt.rcParams["font.family"] = f.name
                return True
    except Exception:
        pass
    return False


def plot_efficiency():
    # Prefer vector-friendly TrueType fonts in PDF
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    # Attempt to use Cavolini if available
    _use_cavolini_font()
    # Methods in fixed order
    methods = [
        "FANVM",
        "TikTek",
        "SVFEND",
        "CAFE",
        "HMCAN",
        "ExMRD",
        "FKVLM",
        "Ours",
    ]

    # Data in minutes; FKVLM is intentionally set to None (filled to max)
    data = {
        "FakeTT": {
            "FANVM": 30,
            "TikTek": 8,
            "SVFEND": 30,
            "CAFE": 3,
            "HMCAN": 7,
            "ExMRD": 6,
            "FKVLM": None,  # fill to max of others
            "Ours": 4,
        },
        "FakeSV": {
            "FANVM": 44,
            "TikTek": 13,
            "SVFEND": 46,
            "CAFE": 5,
            "HMCAN": 11,
            "ExMRD": 9,
            "FKVLM": None,  # fill to max of others
            "Ours": 8,
        },
    }

    # Make the figure as narrow and small as possible, but keep fonts large
    plt.rcParams.update({
        "font.size": 11,  # relatively large fonts for a small figure
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # Horizontal layout: 1 row x 2 columns (side-by-side)
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.9), constrained_layout=False)

    # Consistent colors across subplots
    # Use a qualitative colormap (tab10) and repeat if needed
    base_colors = cm.get_cmap("tab10")
    colors = [base_colors(i % 10) for i in range(len(methods))]

    for ax, (dataset, vals) in zip(axes, data.items()):
        # Compute y-axis max excluding FKVLM
        non_fkvlm_vals = [v for k, v in vals.items() if k != "FKVLM" and v is not None]
        ymax = max(non_fkvlm_vals) if non_fkvlm_vals else 1

        # Build heights where FKVLM is filled to ymax
        heights = []
        for m in methods:
            v = vals.get(m, None)
            if m == "FKVLM":
                heights.append(ymax)
            else:
                heights.append(float(v) if v is not None else 0.0)

        x = list(range(len(methods)))
        bars = ax.bar(
            x,
            heights,
            color=colors,
            edgecolor="black",
            linewidth=0.4,
            align="center",
        )

        ax.set_ylim(0, ymax)
        ax.set_ylabel("min")
        ax.set_title(dataset)

        # X-axis: no label; keep tick labels for methods, rotated to fit
        ax.set_xlabel("")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=90, ha="center")

        # Reduce visual clutter
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # No legend per request

    # Tight layout to minimize whitespace while accommodating large fonts
    plt.tight_layout()

    # Save alongside the script for convenience
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.normpath(os.path.join(script_dir, "..", "analysis", "efficiency_analysis.pdf"))
    plt.savefig(out_path, bbox_inches="tight")

    # Also show if run interactively
    # plt.show()


if __name__ == "__main__":
    plot_efficiency()
