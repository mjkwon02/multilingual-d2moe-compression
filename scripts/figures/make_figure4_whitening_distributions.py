"""Figure 5.3-distribution: between-language vs within-language distribution
comparison for the two §5.3 headline metrics (mean principal angle, k=64;
covariance relative Frobenius distance, k=64), as violin plots.
max_angle_deg_k64 intentionally excluded (saturates near 90 deg for both
within and between -- see the pre-lock review, not a usable discriminator).

Layout brief (2026-08-04): single column (3.0 in), so the two metrics are
stacked as rows -- side by side they would be 1.5 in each.  The pair counts
(n_within / n_between) and the permutation p-value are NOT drawn on the
panels; they belong to the LaTeX caption, and are echoed on stdout so the
caption can be kept in sync with the data.

Source: whitening_geometry_pairs.csv (raw, for the distributions) +
whitening_geometry_analysis.json (for the n / p-value stdout echo, so the
caption's numbers are exactly what's already been locked, not recomputed).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path[:0] = [str(Path(__file__).resolve().parent)] + [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()] + [str(Path(__file__).resolve().parents[2] / "src" / "multilingual_d2moe")]
from figure_style import apply_singlecol_style, save_singlecol  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "source_data"

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"
WITHIN_COLOR = "#0072B2"
BETWEEN_COLOR = "#D55E00"

# Canvas is oversized by exactly what bbox_inches="tight" trims, so the
# emitted PDF is 3.0 in wide -- i.e. \columnwidth at 1:1, no rescaling.
FIG_W_IN = 3.21
FIG_H_IN = 3.75
FS_LABEL = 8.0
FS_TICK = 7.0
FS_PANEL_LETTER = 9.0

# (column, panel letter, y-axis label).  Axis labels use the paper's metric
# names, not the code's variable names; "k=64" lives in the caption.
METRICS = [
    ("mean_angle_deg_k64", "(A)", "Principal angle (deg)"),
    ("cov_reldist_k64", "(B)", "Covariance rel. Frobenius dist."),
]
XTICK_LABELS = ["Within", "Between"]   # n counts moved to the caption


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=FS_TICK)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def main():
    apply_singlecol_style()
    df = pd.read_csv(RESULTS_DIR / "whitening_geometry_pairs.csv")
    df["within"] = df["cond_a"] == df["cond_b"]
    analysis = json.loads((RESULTS_DIR / "whitening_geometry_analysis.json").read_text())

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W_IN, FIG_H_IN), dpi=400,
                             sharex=True)
    fig.subplots_adjust(left=0.235, right=0.98, top=0.955, bottom=0.085,
                        hspace=0.30)

    for ax, (metric, letter, label) in zip(axes, METRICS):
        w = df.loc[df["within"], metric].to_numpy()
        b = df.loc[~df["within"], metric].to_numpy()
        parts = ax.violinplot([w, b], positions=[0, 1], showmeans=True,
                              showextrema=True, widths=0.70)
        for pc, color in zip(parts["bodies"], (WITHIN_COLOR, BETWEEN_COLOR)):
            pc.set_facecolor(color)
            pc.set_alpha(0.35)
            pc.set_edgecolor(color)
            pc.set_linewidth(0.8)
        for key in ("cmeans", "cmins", "cmaxes", "cbars"):
            parts[key].set_color(MUTED)
            parts[key].set_linewidth(0.8)
        # Boxes inside the violins: the two groups differ in spread as well as
        # in location, and the quartiles make that readable in grayscale.
        bp = ax.boxplot([w, b], positions=[0, 1], widths=0.12, showfliers=False,
                        patch_artist=True, zorder=3)
        for box in bp["boxes"]:
            box.set(facecolor="white", edgecolor=INK, linewidth=0.6)
        for key in ("whiskers", "caps", "medians"):
            for artist in bp[key]:
                artist.set(color=INK, linewidth=0.6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(XTICK_LABELS, fontsize=FS_TICK)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel(label, fontsize=FS_LABEL, color=INK)
        style_axis(ax)
        ax.text(-0.235, 1.10, letter, transform=ax.transAxes,
                fontsize=FS_PANEL_LETTER, fontweight="bold", color=INK,
                ha="left", va="top")

        m = analysis["metrics"][metric]["all_layers"]
        n_w, n_b = m["n_within"], m["n_between"]
        p = m["permutation"]["p_value_between_ge_within"]
        p_str = "p < 0.0001" if p == 0.0 else f"p = {p:.4f}"

        # n / ratio / p-value are NOT drawn on the panels -- they belong to the
        # caption in 00_docs/08_figure_정리.md; echo them so the caption can be
        # kept in sync with the data.
        print(f"[figure4] {metric}: n_within={n_w} n_between={n_b} "
              f"ratio={m['ratio_between_over_within']:.4f} {p_str}")

    save_singlecol(fig, "figure4_whitening_distributions")
    plt.close(fig)


if __name__ == "__main__":
    main()
