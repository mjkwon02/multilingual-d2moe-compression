"""Whitening-geometry paper figures (09_부족한_실험_정리.md item 3 / RQ-W1-W2):
consumes phase1_whitening_geometry_analysis.py's outputs (layer-curve JSON +
heatmap NPZ), never touches the raw svd_scale_processed.pt files or GPU --
pure post-processing/plotting, matches make_figure_expert_language_heatmap.py's
INK/MUTED/GRID palette convention so every whitening-mechanism figure in the
paper reads as one figure family.

Figure A (layer-wise): between-language vs within-language (same condition,
different calibration seed -- the sampling-noise floor) principal angle per
MoE layer, shaded 95% band from language-pair-level percentiles already
computed by the analysis script (design note §13 Figure A).

Figure B (heatmap): 6x6 condition-pair matrix (design note §13 Figure B) --
off-diagonal = mean between-condition distance, diagonal = mean WITHIN-
condition cross-seed distance for that condition (annotated, not left as
zero -- the design note's explicit recommendation, since a zero diagonal
would visually imply "same language calibration is perfectly reproducible
across seeds", which is not what's measured or true). The matrix is
symmetric by construction, so `--mask-upper` blanks the redundant upper
triangle when the appendix is under page pressure (no information lost).

Usage:
    conda run -n d2moe_env python make_figure_whitening_geometry.py \
        [--analysis-stem whitening_geometry_analysis] [--layer-subset all|late] \
        [--width full|column] [--mask-upper]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
from phase1_whitening_geometry_analysis import LATE_LAYERS  # noqa: E402

from figure_style import apply_singlecol_style, save_singlecol  # noqa: E402

RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "source_data"

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"
WITHIN_COLOR = "#0072B2"
BETWEEN_COLOR = "#D55E00"
CMAP = "viridis"

# Paper notation (00_docs/08_figure_정리.md §표기 통일): uppercase ISO codes in
# resource order, and the 5-language calibration is "Balanced" -- "mixed5" is a
# retired name and must not appear in any figure.
COND_LABEL = {
    "english_only": "EN", "chinese_only": "ZH", "korean_only": "KO",
    "swahili_only": "SW", "bengali_only": "BN", "mixed_5lang": "Balanced",
}

# Fixed color range: the data spans ~27-78 deg, so vmin=0 wasted the bottom
# half of the colormap and flattened the within/between contrast.
HEATMAP_VMIN, HEATMAP_VMAX = 25.0, 80.0

# Metric names as the paper writes them, not as the code names them
# (00_docs/08_figure_정리.md, 표기 통일 규약 §5).
METRIC_LABEL = {
    "mean_angle_deg_k64": "Mean principal angle (deg), k=64",
    "cov_reldist_k64": "Covariance rel. Frobenius distance, k=64",
}

# LATE_LAYERS is the paper's late band (19-27); the analysis restricts the
# figure-G1 shading to its last 7 layers, which is NOT one of the three named
# bands (early 1-9 / middle 10-18 / late 19-27), so it is labelled by range
# only -- "late layers (21-27)" would contradict the convention.
def _band_label(layers):
    return f"final layers ({min(layers)}–{max(layers)})"   # en-dash


# Print geometry: figsize is chosen FIRST and font sizes are absolute, so the
# PDF goes into LaTeX at 1:1 (never \includegraphics-scaled) and no text shrinks.
FIG_GEOMETRY = {
    "full":   dict(figsize=(6.3, 5.0), label=9.0, tick=9.0, cell=8.5, cbar=8.5, xrot=0),
    # aspect-equal imshow: the emitted width is driven by the HEIGHT (the
    # heatmap is square), so the height is what tunes the PDF to 3.0 in wide.
    "column": dict(figsize=(3.10, 2.53), label=8.0, tick=7.0, cell=7.0, cbar=7.0, xrot=0),
}


def plot_layer_curve(curve, metric, out_stem):
    layers = [r["layer"] for r in curve]
    within_mean = np.array([r["within_mean"] for r in curve])
    within_lo = np.array([r["within_lo"] for r in curve])
    within_hi = np.array([r["within_hi"] for r in curve])
    between_mean = np.array([r["between_mean"] for r in curve])
    between_lo = np.array([r["between_lo"] for r in curve])
    between_hi = np.array([r["between_hi"] for r in curve])

    # Oversized by what bbox_inches="tight" trims, so the emitted PDF is
    # 3.0 x ~2.0 in -- \columnwidth at 1:1.
    fig, ax = plt.subplots(figsize=(3.35, 2.09), dpi=400)
    # alpha 0.15: at column size the old 0.18 bands were dense enough to bury
    # the mean curves where the two ribbons overlap.
    ax.fill_between(layers, within_lo, within_hi, color=WITHIN_COLOR, alpha=0.15, linewidth=0, zorder=2)
    ax.fill_between(layers, between_lo, between_hi, color=BETWEEN_COLOR, alpha=0.15, linewidth=0, zorder=2)
    # Distinct marker AND line style per series, so the two stay separable in
    # grayscale print.
    ax.plot(layers, within_mean, color=WITHIN_COLOR, linewidth=1.2, marker="o", markersize=2.6,
            linestyle="-", label="Within-language (cross-seed floor)", zorder=3)
    ax.plot(layers, between_mean, color=BETWEEN_COLOR, linewidth=1.2, marker="^", markersize=2.8,
            linestyle="--", label="Between-language", zorder=3)
    ax.axvspan(min(LATE_LAYERS), max(LATE_LAYERS), color=GRID, alpha=0.5, zorder=1,
               label=_band_label(LATE_LAYERS))

    ax.set_xlabel("MoE layer", fontsize=8, color=INK)
    ax.set_ylabel(METRIC_LABEL.get(metric, metric.replace("_", " ")), fontsize=8, color=INK)
    # Layers are 1-27; without this the axis started at 0, i.e. at a layer that
    # does not exist.
    ax.set_xlim(0.5, 27.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=7)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    # Centre right: the between curve runs high and the within curve low, so
    # the mid-height right-hand corner is the only region free of both.
    ax.legend(fontsize=6.5, frameon=False, loc="center right", handlelength=1.6,
              handletextpad=0.5, labelspacing=0.25, borderaxespad=0.2)

    # Title/caption intentionally omitted -- both live in the LaTeX \caption
    # (text of record: 00_docs/08_figure_정리.md, 그림 절).
    save_singlecol(fig, out_stem)
    plt.close(fig)


def plot_heatmap(conditions, matrix, metric, layer_note, out_stem,
                 width="column", mask_upper=False):
    n = len(conditions)
    labels = [COND_LABEL.get(c, c) for c in conditions]
    geom = FIG_GEOMETRY[width]

    shown = np.array(matrix, dtype=float)
    if mask_upper:
        # Symmetric by construction, so the upper triangle is redundant.
        shown[np.triu_indices(n, k=1)] = np.nan

    fig, ax = plt.subplots(figsize=geom["figsize"], dpi=400)
    # Fixed vmin/vmax rather than data-driven: the values span ~27-78 deg, so
    # anchoring the ramp at 0 spent half the colormap on empty range and
    # flattened the within-vs-between contrast this figure exists to show.
    norm = plt.Normalize(vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)
    im = ax.imshow(shown, cmap=CMAP, norm=norm)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=geom["tick"], color=INK,
                       rotation=geom["xrot"], ha="right" if geom["xrot"] else "center")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=geom["tick"], color=INK)
    # Both axes carry the same condition set (the matrix is symmetric), so an
    # a/b distinction in the axis titles would be meaningless.
    ax.set_xlabel("Calibration condition", fontsize=geom["label"], color=INK)
    ax.set_ylabel("Calibration condition", fontsize=geom["label"], color=INK)
    ax.tick_params(length=0, pad=1.5)
    for i in range(n):
        for j in range(n):
            val = shown[i, j]
            if np.isnan(val):
                continue
            # Contrast against the CELL, not against the data max: viridis is
            # light at the top of the range, where white text is unreadable.
            txt_color = "white" if norm(val) < 0.5 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=geom["cell"], color=txt_color, fontweight=weight)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(METRIC_LABEL.get(metric, metric.replace("_", " ")),
                       fontsize=geom["cbar"], color=INK)
    cbar.ax.tick_params(labelsize=geom["cbar"], length=1.5, width=0.5, pad=1.5)
    cbar.outline.set_visible(False)

    # The layer subset is a caption matter, not an axis label.
    print(f"[figureG2] {out_stem}: metric={metric} layers={layer_note} "
          f"range={np.nanmin(matrix):.1f}-{np.nanmax(matrix):.1f} deg")

    # Title/caption intentionally omitted -- see plot_layer_curve.
    save_singlecol(fig, out_stem)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-stem", default="whitening_geometry_analysis")
    parser.add_argument("--layer-subset", choices=["all", "late"], default="all",
                         help="which heatmap layer-subset to plot (both are always in the NPZ)")
    parser.add_argument("--width", choices=["full", "column"], default="column",
                         help="print width the heatmap is emitted at (default: one column)")
    parser.add_argument("--mask-upper", action="store_true",
                         help="blank the redundant upper triangle of the symmetric matrix")
    args = parser.parse_args()

    apply_singlecol_style()

    curve_path = RESULTS_DIR / f"{args.analysis_stem}_layer_curve.json"
    heatmap_path = RESULTS_DIR / f"{args.analysis_stem}_heatmap.npz"
    if not curve_path.exists() or not heatmap_path.exists():
        raise FileNotFoundError(
            f"{curve_path} / {heatmap_path} missing -- run phase1_whitening_geometry_analysis.py first"
        )

    curve_data = json.loads(curve_path.read_text())
    plot_layer_curve(curve_data["layers"], curve_data["metric"], "figureG1_whitening_layer_curve_raw")

    npz = np.load(heatmap_path, allow_pickle=True)
    conditions = [str(c) for c in npz["conditions"]]
    metric = str(npz["metric"])
    matrix = npz["all_layers"] if args.layer_subset == "all" else npz["late_layers"]
    layer_note = "all MoE layers" if args.layer_subset == "all" else _band_label(LATE_LAYERS)
    heatmap_stem = "figureG2_whitening_heatmap" if args.layer_subset == "all" else f"figureG2_whitening_heatmap_{args.layer_subset}"
    plot_heatmap(conditions, matrix, metric, layer_note, heatmap_stem,
                 width=args.width, mask_upper=args.mask_upper)


if __name__ == "__main__":
    main()
