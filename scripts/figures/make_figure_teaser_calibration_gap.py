"""Teaser figure -- one calibration choice, five very different outcomes.

Bars: Delta-bpb% after compression when the compression statistics are
calibrated on **English only** (row "English" of Table 1).  Black diamonds: the
same five evaluation languages under **Balanced** calibration (row "Balanced"
of Table 1).  The gap between the two is the damage that a different
calibration mix would have avoided -- the paper's headline.

Languages are in the paper-wide resource order EN, ZH, KO, SW, BN (see
00_docs/08_figure_정리.md, 표기 통일 규약 §1), matching Table 1 and every other
figure -- NOT sorted by damage, which put SW before KO and disagreed with the
rest of the paper.

Palette/typeface follow the rest of the paper's figures (figure_style +
Okabe-Ito): #0072B2 blue for the calibration language itself (EN), #D55E00
vermillion for every other evaluation language.  Drawn at single-column print
size (7.9 cm) with 8 pt+ text, like Figure 3 -- do NOT rescale in LaTeX.

Writes figures/fig_teaser_calibration_gap.{png,pdf}.

SOURCE
------
00_docs/08_figure_정리.md, Table 1 (3-seed mean Delta-bpb%), rows
"English" and "Balanced".  Pure post-processing, no GPU:
  conda run -n d2moe_env python scripts/make_figure_teaser_calibration_gap.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path[:0] = [str(Path(__file__).resolve().parent)] + [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()] + [str(Path(__file__).resolve().parents[2] / "src" / "multilingual_d2moe")]
from figure_style import apply_singlecol_style, save_singlecol  # noqa: E402

# --- shared paper palette (Okabe-Ito, as in Figure 1/3/4/G1) ----------------
INK = "#1a1a1a"
MUTED = "#6b6b6b"
BLUE = "#0072B2"        # evaluation language == calibration language (EN)
VERMILLION = "#D55E00"  # every other evaluation language

FS_LABEL = 8.0
FS_TICK = 7.0
FS_VALUE = 7.0
FS_LEGEND = 6.5
FS_NOTE = 6.5

# --- Table 1, rows "English" and "Balanced", in resource order --------------
# The two arrays are indexed by LANGS: reordering the labels without moving
# these together would silently mislabel every bar.
LANGS = ["EN", "ZH", "KO", "SW", "BN"]
ENGLISH_ONLY = np.array([7.28, 13.80, 88.76, 33.09, 89.02])
BALANCED = np.array([9.29, 7.10, 5.36, 2.83, 2.90])
IS_CALIB_LANG = np.array([True, False, False, False, False])

YMAX = 100.0
BAR_W = 0.62
# The "avoidable damage" bracket now sits INSIDE the BN bar (arrow left of
# centre, rotated label right of it) instead of in a reserved strip to the
# right of the plot: that strip cost ~12% of the figure width for two glyphs.
ARROW_X = 4.0 - 0.15
LABEL_X = 4.0 + 0.13


def main():
    apply_singlecol_style()
    # Oversized by what bbox_inches="tight" trims, so the emitted PDF is
    # exactly one column wide and goes into LaTeX at 1:1.
    fig, ax = plt.subplots(figsize=(3.15, 2.30), dpi=400)

    xs = np.arange(len(LANGS))
    colors = [BLUE if own else VERMILLION for own in IS_CALIB_LANG]

    ax.bar(xs, ENGLISH_ONLY, width=BAR_W, color=colors, zorder=3)
    for x, v, c in zip(xs, ENGLISH_ONLY, colors):
        ax.text(x, v + 2.5, f"{v:.1f}", ha="center", va="bottom",
                fontsize=FS_VALUE, color=c, fontweight="bold", zorder=5)

    # balanced calibration: the achievable floor, same five languages
    ax.plot(xs, BALANCED, ls=":", lw=0.8, color=MUTED, zorder=4)
    ax.plot(xs, BALANCED, ls="none", marker="D", ms=4.2, color=INK,
            markeredgecolor="white", markeredgewidth=0.5, zorder=6)

    # the gap the paper is about, measured on the worst language (BN), drawn
    # in white on top of that bar so it needs no plot area of its own
    ax.annotate("", xy=(ARROW_X, ENGLISH_ONLY[-1]), xytext=(ARROW_X, BALANCED[-1]),
                arrowprops=dict(arrowstyle="<->", color="white", lw=0.7,
                                shrinkA=0, shrinkB=0), zorder=7)
    ax.text(LABEL_X, (ENGLISH_ONLY[-1] + BALANCED[-1]) / 2,
            "avoidable damage", rotation=90, ha="center", va="center",
            fontsize=FS_NOTE, color="white", zorder=7)

    # Legend above the axes: three entries inside the plot covered the top-left
    # quadrant, which is where the ZH/SW bars' headroom is.
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, label="EN calib., own language"),
            Patch(facecolor=VERMILLION, label="EN calib., other languages"),
            Line2D([], [], ls="none", marker="D", ms=3.4, color=INK,
                   markeredgecolor="white", markeredgewidth=0.5,
                   label="Balanced calib."),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
        fontsize=FS_LEGEND, handlelength=1.0, handleheight=0.9,
        handletextpad=0.5, labelspacing=0.25, columnspacing=1.0,
        borderpad=0.0, borderaxespad=0.0)

    ax.set_xticks(xs, LANGS, fontsize=FS_TICK)
    for tick, own in zip(ax.get_xticklabels(), IS_CALIB_LANG):
        tick.set_color(BLUE if own else INK)
    ax.set_xlim(-0.62, 4.62)
    ax.set_ylim(0, YMAX)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("$\\Delta$bpb% after compression", fontsize=FS_LABEL,
                  color=INK, labelpad=2.0)
    ax.set_xlabel("Evaluation language", fontsize=FS_LABEL, color=INK,
                  labelpad=1.5)

    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=2.5, colors=INK, labelsize=FS_TICK)
    ax.tick_params(axis="y", length=1.5, width=0.5, pad=1.5, colors=INK,
                   labelsize=FS_TICK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_color(MUTED)

    fig.subplots_adjust(left=0.155, right=0.995, top=0.855, bottom=0.175)
    save_singlecol(fig, "fig_teaser_calibration_gap")
    plt.close(fig)


if __name__ == "__main__":
    main()
