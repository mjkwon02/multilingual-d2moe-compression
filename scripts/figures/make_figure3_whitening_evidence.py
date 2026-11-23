"""Figure 3 (fig3_layerwise): §6.1 + §6.2 + §6.3 stacked on a shared MoE-layer
x-axis (1-27), so the "mid-layer gap/trough" alignment across all three
mechanisms is visible in one figure. Pure post-processing of already-computed
result JSONs / the whitening-geometry pairs CSV -- no GPU, no recompute of
the underlying statistics.

Panel A (§6.1) "Fisher rank correlation": mean cross-language Spearman rho
  (10 language pairs, shaded min-max range) vs mean placebo rho (3 pairs,
  noise floor) per layer.
  Source: phase1_51_fisher_rank_correlation_result_seed0.json
Panel B (§6.2) "Coverage deficiencies": stacked bar of deficient
  (layer, expert) pair counts per layer, colored by language (Swahili/Bengali
  -- English/Korean/Chinese are always 0). Source:
  phase1_52b_deficiency_pairs_list_result.json (raw flagged list, aggregated
  here by layer x lang -- NOT the same file
  make_figure_deficiency_layer_histogram.py uses, per this figure's brief).
Panel C (§6.3) "Between/within ratios": between/within ratio per layer for
  mean_angle_deg_k64 and cov_reldist_k64 on ONE ratio axis (both quantities
  are dimensionless ratios), ratio=1 dashed "no effect" reference line.
  Source: whitening_geometry_pairs.csv (raw pairs, ratio computed here per
  layer -- the saved layer_curve.json only has mean_angle_deg_k64).

Layout brief (2026-08-02): built at final print size -- single column,
~7.6 cm wide -- so nothing is rescaled at \\includegraphics time and every
tick/legend label stays >= 8 pt on the printed page. Bold A/B/C panel letters
+ short English noun-phrase subtitles; no "Figure 3" string and no caption
inside the image (caption of record: 00_docs/08_figure_정리.md).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path[:0] = [str(Path(__file__).resolve().parent)] + [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()] + [str(Path(__file__).resolve().parents[2] / "src" / "multilingual_d2moe")]
from figure_style import (  # noqa: E402
    SINGLE_COL_IN, apply_singlecol_style, save_singlecol,
)

RESULTS_DIR = Path("results/deepseek")
SOURCE_DATA = Path(__file__).resolve().parents[2] / "results" / "source_data"

OUT_STEM = "fig3_layerwise"

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"
LANG_COLOR = "#0072B2"
PLACEBO_COLOR = "#D55E00"
LANG_COLORS_52 = {"swh_Latn": "#D55E00", "ben_Beng": "#0072B2"}
# Paper notation: uppercase ISO codes, never the FLORES tag or a spelled-out
# language name (00_docs/08_figure_정리.md, 표기 통일 규약 §1).
LANG_LABELS_52 = {"swh_Latn": "SW", "ben_Beng": "BN"}
ANGLE_COLOR = "#0072B2"
COV_COLOR = "#D55E00"
MID_SHADE = "#f0f0f0"
# Shared shading band, identical in all three panels: the "middle" layer band
# of the paper's early (1-9) / middle (10-18) / late (19-27) split.  The span
# is drawn from layer 10 to layer 18 exactly, so its edges land on those two
# layer positions rather than half a layer outside them.
MID_LAYERS = (10, 18)
MID_LABEL = "middle layers (10–18)"   # en-dash per the notation convention

LAYERS = list(range(1, 28))
XTICK_LAYERS = [1, 5, 10, 15, 20, 25]

# Single-column print geometry: 7.7 cm = 3.03 in, emitted at 3.0 in. Font sizes
# below are what lands on paper, so do NOT scale this figure in LaTeX (use
# width=\columnwidth with the figure already at column width).
FIG_W_IN = SINGLE_COL_IN
FIG_H_IN = 5.2
FS_TICK = 7.0
FS_LABEL = 8.0
FS_TITLE = 8.0
FS_LEGEND = 6.5
FS_PANEL_LETTER = 9.0


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=FS_TICK, length=2.5, width=0.7, pad=1.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def shade_mid(ax):
    """The middle band, drawn from layer 10 to layer 18 exactly (all panels)."""
    lo, hi = MID_LAYERS
    ax.axvspan(lo, hi, color=MID_SHADE, zorder=0)


def panel_header(ax, letter, subtitle):
    """Bold A/B/C outside the panel's top-left + a short noun-phrase subtitle."""
    ax.set_title(subtitle, fontsize=FS_TITLE, color=INK, pad=3.5)
    ax.text(-0.235, 1.16, letter, transform=ax.transAxes, fontsize=FS_PANEL_LETTER,
            fontweight="bold", color=INK, ha="left", va="top")


def load_panel_a(seed=0):
    result = json.loads((RESULTS_DIR / f"phase1_51_fisher_rank_correlation_result_seed{seed}.json").read_text())
    per_layer = result["per_layer"]
    mean_lang = np.array([per_layer[str(l)]["mean_lang_rho"] for l in LAYERS])
    mean_placebo = np.array([per_layer[str(l)]["mean_placebo_rho"] for l in LAYERS])
    lang_lo = np.array([min(v["rho"] for v in per_layer[str(l)]["lang_pairs"].values()) for l in LAYERS])
    lang_hi = np.array([max(v["rho"] for v in per_layer[str(l)]["lang_pairs"].values()) for l in LAYERS])
    return mean_lang, mean_placebo, lang_lo, lang_hi


def load_panel_b():
    result = json.loads((RESULTS_DIR / "phase1_52b_deficiency_pairs_list_result.json").read_text())
    counts = {lang: defaultdict(int) for lang in LANG_COLORS_52}
    for row in result["flagged"]:
        lang = row["lang"]
        if lang in counts:
            counts[lang][row["layer"]] += 1
    return counts, result["total_flagged"], result["total_pairs_scanned"]


def load_panel_c():
    df = pd.read_csv(SOURCE_DATA / "whitening_geometry_pairs.csv")
    df["within"] = df["cond_a"] == df["cond_b"]
    angle_ratio, cov_ratio = [], []
    for l in LAYERS:
        sub = df[df["layer"] == l]
        w_a = sub.loc[sub["within"], "mean_angle_deg_k64"].mean()
        b_a = sub.loc[~sub["within"], "mean_angle_deg_k64"].mean()
        w_c = sub.loc[sub["within"], "cov_reldist_k64"].mean()
        b_c = sub.loc[~sub["within"], "cov_reldist_k64"].mean()
        angle_ratio.append(b_a / w_a)
        cov_ratio.append(b_c / w_c)
    return np.array(angle_ratio), np.array(cov_ratio)


def main(seed_51=0):
    apply_singlecol_style()
    x = np.array(LAYERS)   # plot against real layer numbers, not 0-based index

    mean_lang, mean_placebo, lang_lo, lang_hi = load_panel_a(seed_51)
    counts_b, total_flagged, total_pairs = load_panel_b()
    angle_ratio, cov_ratio = load_panel_c()

    fig, (axA, axB, axC) = plt.subplots(
        3, 1, figsize=(FIG_W_IN, FIG_H_IN), dpi=400, sharex=True,
    )
    fig.subplots_adjust(left=0.185, right=0.985, top=0.845, bottom=0.075,
                        hspace=0.35)

    # --- Panel A: Fisher rank correlation ---
    shade_mid(axA)
    axA.fill_between(x, lang_lo, lang_hi, color=LANG_COLOR, alpha=0.15, zorder=1,
                     label="Language-pair range")
    axA.plot(x, mean_lang, color=LANG_COLOR, marker="o", markersize=2.2, linewidth=1.2, zorder=3,
             label="Language pairs (mean, 10)")
    axA.plot(x, mean_placebo, color=PLACEBO_COLOR, marker="s", markersize=2.2, linewidth=1.2,
             linestyle="--", zorder=3, label="Placebo (mean, 3)")
    axA.set_ylabel("Spearman " + r"$\rho$", fontsize=FS_LABEL, color=INK)
    # Headroom above the placebo curve (rho ~= 0.95); the legend no longer
    # needs room inside the axes, so the band is not squeezed either.
    axA.set_ylim(min(-0.45, lang_lo.min() - 0.05), 1.06)
    axA.set_yticks([-0.4, 0.0, 0.4, 0.8])
    style_axis(axA)
    panel_header(axA, "A", "Fisher rank correlation")
    # Legend lifted clear of the axes: inside the panel it covered the
    # language-pair band.  The shading is explained here once, for all three
    # panels, since the band is identical in each.
    handles, labels = axA.get_legend_handles_labels()
    handles.append(Patch(facecolor=MID_SHADE, edgecolor="none"))
    labels.append(MID_LABEL)
    axA.legend(handles, labels, frameon=False, fontsize=FS_LEGEND,
               loc="lower center", bbox_to_anchor=(0.5, 1.28), ncol=2,
               handlelength=1.4, handletextpad=0.5, labelspacing=0.25,
               columnspacing=1.0, borderpad=0.0, borderaxespad=0.0)

    # --- Panel B: Coverage deficiencies ---
    shade_mid(axB)
    bottom = np.zeros(len(LAYERS))
    for lang in ("swh_Latn", "ben_Beng"):
        vals = np.array([counts_b[lang].get(l, 0) for l in LAYERS])
        axB.bar(x, vals, bottom=bottom, color=LANG_COLORS_52[lang], label=LANG_LABELS_52[lang],
                zorder=3, edgecolor="white", linewidth=0.3)
        bottom += vals
    axB.set_ylabel("Flagged count", fontsize=FS_LABEL, color=INK)
    axB.set_ylim(0, 10.4)
    axB.set_yticks([0, 2, 4, 6, 8])
    style_axis(axB)
    panel_header(axB, "B", "Coverage deficiencies")
    # Upper right: the flagged counts are concentrated in the left half, so
    # this corner is empty (SW/BN order follows the paper's resource order).
    axB.legend(frameon=False, fontsize=FS_LEGEND, loc="upper right", ncol=2, handlelength=1.0,
               handletextpad=0.4, columnspacing=0.8, borderpad=0.1, borderaxespad=0.2)

    # --- Panel C: Between/within ratios ---
    shade_mid(axC)
    axC.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":", zorder=2)
    axC.text(27.4, 1.0, "no effect", fontsize=FS_LEGEND, color=MUTED,
             ha="right", va="bottom", zorder=4)
    axC.plot(x, angle_ratio, color=ANGLE_COLOR, marker="o", markersize=2.2, linewidth=1.2, zorder=3,
             label="Principal angle")
    axC.plot(x, cov_ratio, color=COV_COLOR, marker="^", markersize=2.4, linewidth=1.2,
             linestyle="--", zorder=3, label="Covariance Frobenius")
    axC.set_ylabel("Between/within ratio", fontsize=FS_LABEL, color=INK)
    axC.set_ylim(0.6, 8.6)
    axC.set_yticks([1, 3, 5, 7])
    style_axis(axC)
    panel_header(axC, "C", "Between/within ratios")
    axC.legend(frameon=False, fontsize=FS_LEGEND, loc="upper center", ncol=1, handlelength=1.4,
               handletextpad=0.5, labelspacing=0.2, borderpad=0.1, borderaxespad=0.2)

    # x axis is shared: ticks/label belong to the bottom panel only.
    axC.set_xticks(XTICK_LAYERS)
    axC.set_xticklabels([str(l) for l in XTICK_LAYERS], fontsize=FS_TICK)
    axC.set_xlabel("MoE layer", fontsize=FS_LABEL, color=INK)
    axC.set_xlim(0.4, 27.6)

    # No title/caption on the image -- caption text lives in 00_docs/08_figure_정리.md.
    save_singlecol(fig, OUT_STEM)
    print(f"[figure3] panel B: {total_flagged} flagged of {total_pairs} pairs scanned")
    plt.close(fig)


if __name__ == "__main__":
    main()
