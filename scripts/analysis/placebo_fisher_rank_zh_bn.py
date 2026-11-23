"""POST-HOC SUPPLEMENT to frozen Table F1 (§5.1 Fisher rank correlation).

Table F1 reports 10 language pairs + 3 placebo pairs (EN/KO/SW). ZH and BN were
excluded because, at the time F1 was computed (2026-07-30), chinese_only_placebo was
not in phase1_calib_data.PLACEBO_CONDITIONS at all and bengali_only_placebo's real
Fisher had never been run (empty directory) -- see
phase1_51_fisher_rank_correlation.py's _placebo_of() docstring.

Both placebo arms now exist on disk with real fisher_processed.pt for all three
seeds, so the two missing placebo pairs can be filled in. This is a one-shot
extraction performed BEFORE those large binaries are deleted for disk recovery.

F1's BODY IS NOT MODIFIED and this is NOT folded into it. The numbers here are
reported separately as a post-hoc supplement.

Method is F1's, not a reimplementation: reduce_condition() is imported from
phase1_51_fisher_rank_correlation and does the identical reduction (per-expert
F_e = sum over that expert's 3 projections of the sample-averaged squared
gradients, mmap'd read, cached to fisher_expert_scalar_cache/). Per-layer
Spearman over the 64 experts, then mean/std over the 27 MoE layers with
np.std's default ddof=0 -- F1's convention (00_docs/08_figure_정리.md, "표 F1
std 컨벤션: ddof=0(population)").

Usage: python placebo_fisher_rank_zh_bn.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
# Importing this module is what guarantees "same method as F1" -- the reduction
# and its cache both come from there, unchanged.
from phase1_51_fisher_rank_correlation import (  # noqa: E402
    ALL_MOE_LAYERS, RESULTS_ROOT, reduce_condition,
)

PAIRS = [("chinese_only", "chinese_only_placebo"), ("bengali_only", "bengali_only_placebo")]
SEEDS = [0, 1, 2]
OUT_DIR = SCRIPT_DIR.parents[1] / "results" / "source_data"
OUT_JSON = OUT_DIR / "placebo_fisher_rank_zh_bn.json"
OUT_MD = OUT_DIR / "placebo_fisher_rank_zh_bn.md"

LABEL = ("POST-HOC SUPPLEMENT to frozen Table F1. Table F1's body is NOT modified "
         "and these two placebo pairs are NOT retro-fitted into it; they are reported "
         "separately. Extracted 2026-08-02, immediately before the chinese_only_placebo / "
         "bengali_only_placebo large binaries were deleted for disk recovery.")


def main():
    results = {"label": LABEL, "method": ("F1's reduce_condition() imported unchanged from "
                                          "phase1_51_fisher_rank_correlation.py; per-layer Spearman over "
                                          "64 experts; mean/std over 27 MoE layers, ddof=0 (F1 convention)"),
               "layers": ALL_MOE_LAYERS, "pairs": {}}

    for arm_a, arm_b in PAIRS:
        results["pairs"][f"{arm_a}|{arm_b}"] = {}
        for seed in SEEDS:
            a = reduce_condition(arm_a, seed, ALL_MOE_LAYERS)
            b = reduce_condition(arm_b, seed, ALL_MOE_LAYERS)
            per_layer = {}
            for layer in ALL_MOE_LAYERS:
                rho, p = spearmanr(a[layer], b[layer])
                per_layer[layer] = {"rho": float(rho), "p": float(p)}
            rhos = np.array([per_layer[l]["rho"] for l in ALL_MOE_LAYERS])
            results["pairs"][f"{arm_a}|{arm_b}"][f"seed{seed}"] = {
                "mean_rho": float(rhos.mean()),
                "std_rho": float(rhos.std()),          # ddof=0, F1 convention
                "min_rho": float(rhos.min()), "max_rho": float(rhos.max()),
                "n_layers": len(ALL_MOE_LAYERS),
                "per_layer": {str(k): v for k, v in per_layer.items()},
            }
            print(f"[zh-bn] {arm_a} vs {arm_b} seed{seed}: "
                  f"mean rho = {rhos.mean():.4f} +/- {rhos.std():.4f} "
                  f"(min {rhos.min():.4f}, max {rhos.max():.4f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"[zh-bn] wrote {OUT_JSON}")

    lines = [
        "# Placebo Fisher rank correlation — ZH / BN (post-hoc supplement to Table F1)",
        "",
        f"**{LABEL}**",
        "",
        "Method identical to Table F1: per-expert Fisher scalar F_e (sum of the sample-averaged",
        "squared gradients over that expert's 3 projections), per-layer Spearman over 64 experts,",
        "mean ± std over the 27 MoE layers (ddof=0, F1's convention). The reduction function is",
        "imported from `phase1_51_fisher_rank_correlation.py`, not reimplemented.",
        "",
        "## Placebo ρ (same language, disjoint calibration text)",
        "",
        "| pair | seed 0 | seed 1 | seed 2 | 3-seed mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm_a, arm_b in PAIRS:
        key = f"{arm_a}|{arm_b}"
        cells, means = [], []
        for seed in SEEDS:
            r = results["pairs"][key][f"seed{seed}"]
            cells.append(f"{r['mean_rho']:.3f} ± {r['std_rho']:.3f}")
            means.append(r["mean_rho"])
        lines.append(f"| {arm_a} vs {arm_b} | " + " | ".join(cells) + f" | **{np.mean(means):.3f}** |")
    lines += [
        "",
        "Each cell is the 27-layer mean ± std of the per-layer Spearman ρ for that seed.",
        "",
        "## Why these two pairs were missing from F1",
        "",
        "At F1's computation time `chinese_only_placebo` was absent from",
        "`phase1_calib_data.PLACEBO_CONDITIONS`, and `bengali_only_placebo`'s real Fisher had never",
        "been run (the directory existed but was empty). Both placebo arms have real",
        "`fisher_processed.pt` for all 3 seeds now, so the pairs are computable. They are",
        "reported here only — F1 keeps its original 3 placebo pairs (EN/KO/SW).",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"[zh-bn] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
