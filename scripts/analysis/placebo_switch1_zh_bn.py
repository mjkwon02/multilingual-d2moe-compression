"""POST-HOC SUPPLEMENT to frozen §6.1 / Appendix F — SWITCH-1 equivalence for
the ZH and BN placebo arms.

Frozen text, Table F1 and Table F3 are NOT modified and these numbers are NOT
retro-fitted into them; they are reported only as a separate supplementary
table.

Two outputs, both under the merge-weight definition SWITCH-1 uses:

 (a) DEFINITION AGREEMENT — per (condition, seed, layer), Spearman between the
     raw F_e ranking and the merge-weighted ranking, per projection
     (gate/up/down kept separate) plus a "combined" line. Compared against the
     existing EN/KO/SW reference (overall mean 0.939, minimum 0.831).
     GATE: agreement >= 0.83 -> PASS. Below that is FLAGged (and the number is
     withheld from the paper) but does NOT block the deletion step.

 (b) PLACEBO rho UNDER THE MERGE-WEIGHT DEFINITION — the same arm-A vs arm-B
     correlation part 2-A computes on raw F_e, recomputed on the merge-weighted
     combined scalar, i.e. F3's "does the within-language ceiling survive a
     change of definition?" checked on ZH/BN.

Method is SWITCH-1's, not a reimplementation: reduce_condition_by_proj(),
load_expert_freq(), merge_weighted() and part_a_definition_agreement() are all
imported unchanged from phase1_51b_merge_weight_rank_correlation.py, which is
itself left untouched. F_e_merge[proj][e] = F_e_raw[proj][e] * expert_freq[e]
(exact factoring, see that module's docstring).

RSS is managed by the imported reduction's mmap=True read (file-backed,
evictable page cache) and by reducing one condition/seed at a time, caching the
tiny per-projection arrays and dropping the mapping before the next. `free -h`
is logged around every reduction.

Usage: python placebo_switch1_zh_bn.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
from phase1_51b_merge_weight_rank_correlation import (  # noqa: E402
    ALL_MOE_LAYERS, PROJECTIONS, load_expert_freq, merge_weighted,
    part_a_definition_agreement, reduce_condition_by_proj,
)

PAIRS = [("chinese_only", "chinese_only_placebo"), ("bengali_only", "bengali_only_placebo")]
SEEDS = [0, 1, 2]
OUT_DIR = SCRIPT_DIR.parents[1] / "results" / "source_data"
OUT_JSON = OUT_DIR / "placebo_switch1_zh_bn.json"

# Existing EN/KO/SW SWITCH-1 reference, quoted for comparison only.
REF_OVERALL_MEAN = 0.939
REF_MIN = 0.831
GATE_THRESHOLD = 0.83

LABEL = ("POST-HOC SUPPLEMENT to frozen §6.1 / Appendix F. The frozen text and Tables F1 "
         "and F3 are NOT modified; these ZH/BN placebo numbers are reported only as a "
         "separate supplementary table and are not retro-fitted into F1/F3. Extracted "
         "2026-08-02, immediately before the chinese_only_placebo / bengali_only_placebo large "
         "binaries were deleted for disk recovery.")


def mem():
    try:
        return subprocess.run(["free", "-h"], capture_output=True, text=True).stdout.splitlines()[1]
    except Exception:
        return "(free unavailable)"


def main():
    raw_all, merge_all, conditions = {}, {}, []

    for arm_a, arm_b in PAIRS:
        for cond in (arm_a, arm_b):
            for seed in SEEDS:
                key = f"{cond}#seed{seed}"
                print(f"[switch1-zhbn] mem before {key}: {mem()}", flush=True)
                raw = reduce_condition_by_proj(cond, seed, ALL_MOE_LAYERS)
                freq = load_expert_freq(cond, seed, ALL_MOE_LAYERS)
                raw_all[key] = raw
                merge_all[key] = merge_weighted(raw, freq, ALL_MOE_LAYERS)
                conditions.append(key)
                print(f"[switch1-zhbn] mem after  {key}: {mem()}", flush=True)

    # ---- (a) definition agreement, SWITCH-1's own function, unchanged ----
    rows, summary = part_a_definition_agreement(raw_all, merge_all, conditions, ALL_MOE_LAYERS)

    per_cond = {}
    for key in conditions:
        krows = [r for r in rows if r["condition"] == key]
        per_cond[key] = {
            k: {"mean_rho": float(np.mean([r[k]["rho"] for r in krows])),
                "min_rho": float(np.min([r[k]["rho"] for r in krows]))}
            for k in PROJECTIONS + ("combined",)
        }

    all_rhos = [r[k]["rho"] for r in rows for k in PROJECTIONS + ("combined",)]
    observed_min = float(np.min(all_rhos))
    observed_mean = float(np.mean(all_rhos))
    gate_pass = observed_min >= GATE_THRESHOLD
    print(f"\n[switch1-zhbn] definition agreement: mean={observed_mean:.4f} min={observed_min:.4f} "
          f"(EN/KO/SW reference mean {REF_OVERALL_MEAN}, min {REF_MIN}) -> "
          f"{'PASS' if gate_pass else 'FLAG (below 0.83; withhold from paper, deletion still proceeds)'}",
          flush=True)

    # ---- (b) placebo rho under the merge-weight definition ----
    placebo_merge = {}
    for arm_a, arm_b in PAIRS:
        placebo_merge[f"{arm_a}|{arm_b}"] = {}
        for seed in SEEDS:
            ka, kb = f"{arm_a}#seed{seed}", f"{arm_b}#seed{seed}"
            per_layer = {}
            for l in ALL_MOE_LAYERS:
                ca = sum(merge_all[ka][l][p] for p in PROJECTIONS)
                cb = sum(merge_all[kb][l][p] for p in PROJECTIONS)
                rho, p = spearmanr(ca, cb)
                per_layer[l] = {"rho": float(rho), "p": float(p)}
            rhos = np.array([per_layer[l]["rho"] for l in ALL_MOE_LAYERS])
            placebo_merge[f"{arm_a}|{arm_b}"][f"seed{seed}"] = {
                "mean_rho": float(rhos.mean()), "std_rho": float(rhos.std()),  # ddof=0, F1 convention
                "min_rho": float(rhos.min()), "max_rho": float(rhos.max()),
                "per_layer": {str(k): v for k, v in per_layer.items()},
            }
            print(f"[switch1-zhbn] placebo(merge-weighted) {arm_a} vs {arm_b} seed{seed}: "
                  f"mean rho = {rhos.mean():.4f} +/- {rhos.std():.4f}", flush=True)

    out = {
        "label": LABEL,
        "method": ("SWITCH-1's own functions imported unchanged from "
                   "phase1_51b_merge_weight_rank_correlation.py: reduce_condition_by_proj(), "
                   "load_expert_freq(), merge_weighted(), part_a_definition_agreement(). "
                   "F_e_merge[proj][e] = F_e_raw[proj][e] * expert_freq[e]."),
        "layers": ALL_MOE_LAYERS,
        "definition_agreement": {
            "reference_en_ko_sw": {"overall_mean_rho": REF_OVERALL_MEAN, "min_rho": REF_MIN},
            "gate_threshold": GATE_THRESHOLD,
            "observed_mean_rho": observed_mean,
            "observed_min_rho": observed_min,
            "verdict": "PASS" if gate_pass else "FLAG_BELOW_REFERENCE_MIN",
            "note": ("FLAG does not block deletion (memory recovery has priority); it withholds "
                     "these numbers from the paper until reviewed."),
            "summary_all_conditions": summary,
            "per_condition": per_cond,
        },
        "placebo_rho_merge_weighted": placebo_merge,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[switch1-zhbn] wrote {OUT_JSON}")
    print(f"[switch1-zhbn] mem at end: {mem()}")


if __name__ == "__main__":
    main()
