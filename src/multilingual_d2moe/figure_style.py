"""Shared paper-figure style + save helper for the four surviving figures
(Figure 3, Figure 4, Figure G1, Figure G2 -- see 00_docs/08_figure_정리.md).

Two conventions this module enforces, both decided 2026-08-02:

1. **Times-family typeface.** No literal "Times New Roman" on this box, so the
   preference list falls through to Nimbus Roman (URW Times clone) / Liberation
   Serif (metric-compatible with Times New Roman); mathtext uses STIX, which is
   the Times-metric math set. Add "Times New Roman" first anyway so the same
   scripts render identically on a machine that has the real thing.
2. **No captions baked into the image.** Figure titles/caption blocks live in
   00_docs/08_figure_정리.md and go into the LaTeX \\caption, not the PNG --
   so `save_figure` only writes the plot itself, and every script here keeps
   axis labels/legends (which are part of the plot) but no set_title/fig.text.

Every figure is written twice: PNG (draft/preview) + PDF (vector, submission).
"""
from pathlib import Path

import matplotlib

SERIF_STACK = [
    "Times New Roman",   # real Times, if the host has it
    "Nimbus Roman",      # URW Times clone (this box)
    "Liberation Serif",  # metric-compatible fallback
    "STIXGeneral",
    "DejaVu Serif",
]

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"

# ACL A4 two-column: one column is 7.7 cm = 3.03 in.  Every paper figure is
# drawn at this width so \includegraphics[width=\columnwidth] never rescales it
# -- rescaling would push the 7 pt tick labels below the 6 pt legibility floor.
SINGLE_COL_IN = 3.0


def apply_paper_style():
    """Times-family text + vector-friendly PDF output. Call before plotting."""
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF_STACK,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,   # embed TrueType, keep text selectable/searchable
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def apply_singlecol_style():
    """apply_paper_style() + the single-column type scale (2026-08-04 brief).

    Absolute point sizes, because the figure is emitted at final print size:
    body 8 pt, ticks/legend 7 pt.  Scripts may still pass explicit `fontsize=`
    where a specific element needs to differ, but nothing may go below 6 pt.
    """
    apply_paper_style()
    matplotlib.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
    })


def save_singlecol(fig, stem):
    """save_figure with the submission's tight/pad settings (PNG + vector PDF)."""
    return save_figure(fig, stem, bbox_inches="tight", pad_inches=0.02)


def save_figure(fig, stem, **savefig_kwargs):
    """Write <stem>.png and <stem>.pdf into figures/. Returns both paths."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(path, facecolor="white", **savefig_kwargs)
        print(f"[figure] wrote {path}")
        out.append(path)
    return out
