"""§5.4 STEP 0 (0-1/0-2): select the intervention targets for the causal-chain
verification, from already-computed scan_disagreement_experts.py output. CPU
only, seconds, no GPU, no model load -- pure post-processing of scan_results.json
plus the §5.2 flagged list (phase1_52b_deficiency_pairs_list_result.json / Table
E1).

Reuses disagreement_common.normalize_layer + rank_by_disagreement DIRECTLY (the
same per-layer column-minmax winsorize + row-wise cross-language variance that
Stage 1 used to pick each layer's `disagreement_experts`), so "cross-language
proxy-importance dispersion" here is the identical statistic the rest of the
pipeline uses -- it cannot drift from §5.1/§5.2.

Selection (matches the §5.4 instruction's design):
  treatment (12): E1-flagged (layer, expert) high-dispersion experts --
    8 from the late band (22-27), forced to cover the Swahili layers 24/25/27;
    4 from the early band (1-5), Bengali-flagged included.
  control (12): for each treatment, a SAME-LAYER expert with LOW cross-language
    dispersion (bottom of the layer) that is NOT in that layer's flagged
    disagreement set and has sufficient routing (total hit_count over the 5
    language conditions >= the layer's median) -- 1:1 matched, distinct experts.

Records, per selected expert, every covariate the P1/P2/P3 tests need:
  disagreement variance, per-language normalized proxy, per-language raw proxy,
  per-language hit_count (= the routing frequency `freq` that P2 correlates
  restoration gain against), role.

Usage:
    conda run -n d2moe_env python phase1_sec54_select_targets.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
from disagreement_common import (  # noqa: E402
    LANG_CONDITIONS, normalize_layer, rank_by_disagreement,
)

RESULTS_ROOT = Path("results")
SCAN_PATH = RESULTS_ROOT / "deepseek" / "aux" / "scan_results.json"
E1_PATH = RESULTS_ROOT / "deepseek" / "phase1_52b_deficiency_pairs_list_result.json"
OUT_DIR = RESULTS_ROOT / "deepseek" / "sec54"

N_LATE = 8          # late-band (22-27) treatment count
N_EARLY = 4         # early-band (1-5) treatment count
LATE_BAND = range(22, 28)
EARLY_BAND = range(1, 6)
MUST_COVER_LATE = {24, 25, 27}   # E1 Swahili-concentrated layers


def per_expert_variance(layer_entry):
    """(64,) cross-language disagreement variance for one layer, via the exact
    Stage-1 statistic: column-minmax-winsorize proxy per language, then row-wise
    variance across the 5 language conditions."""
    proxy_by_cond = {c: np.asarray(layer_entry[c]["proxy"], dtype=float) for c in LANG_CONDITIONS}
    norm = normalize_layer(proxy_by_cond, LANG_CONDITIONS, method="minmax", clip_percentile=2.0)
    _order, variance = rank_by_disagreement(norm)
    return variance, norm  # norm[expert] = per-language normalized proxy row


def expert_record(scan, layer, expert):
    entry = scan[str(layer)]
    variance, norm = per_expert_variance(entry)
    hit_by_lang = {c: int(entry[c]["hit_count"][expert]) for c in LANG_CONDITIONS}
    proxy_by_lang = {c: float(entry[c]["proxy"][expert]) for c in LANG_CONDITIONS}
    norm_by_lang = {c: float(norm[expert][i]) for i, c in enumerate(LANG_CONDITIONS)}
    return {
        "layer": int(layer),
        "expert": int(expert),
        "role": entry.get("role"),
        "disagreement_var": float(variance[expert]),
        "total_hits": int(sum(hit_by_lang.values())),
        "hit_count_by_lang": hit_by_lang,
        "proxy_by_lang": proxy_by_lang,
        "norm_proxy_by_lang": norm_by_lang,
        "in_flagged_set": expert in (entry.get("disagreement_experts") or []),
    }


def pick_treatment(scan, flagged):
    """flagged: list of dicts {layer, expert, lang, hits} from Table E1."""
    def var_of(layer, expert):
        v, _ = per_expert_variance(scan[str(layer)])
        return float(v[expert])

    late = [f for f in flagged if f["layer"] in LATE_BAND]
    early = [f for f in flagged if f["layer"] in EARLY_BAND]
    for f in late + early:
        f["disagreement_var"] = var_of(f["layer"], f["expert"])

    # late: top-N_LATE by dispersion, but guarantee 24/25/27 each contribute
    late_sorted = sorted(late, key=lambda f: -f["disagreement_var"])
    chosen_late, covered = [], set()
    for f in late_sorted:
        if len(chosen_late) < N_LATE:
            chosen_late.append(f); covered.add(f["layer"])
    missing = MUST_COVER_LATE - covered
    for layer in sorted(missing):
        best = max((f for f in late if f["layer"] == layer), key=lambda f: f["disagreement_var"], default=None)
        if best is None:
            continue
        # swap out the lowest-dispersion currently-chosen from an OVER-covered layer
        droppable = sorted(
            (f for f in chosen_late if (covered_count(chosen_late, f["layer"]) > 1)),
            key=lambda f: f["disagreement_var"])
        if droppable:
            chosen_late.remove(droppable[0])
            chosen_late.append(best); covered.add(layer)

    early_sorted = sorted(early, key=lambda f: -f["disagreement_var"])
    chosen_early = early_sorted[:N_EARLY]
    return chosen_late, chosen_early


def covered_count(chosen, layer):
    return sum(1 for f in chosen if f["layer"] == layer)


def pick_controls(scan, treatments):
    """1:1 same-layer match. For each layer, assign the k lowest-dispersion
    eligible experts to the k treatments in that layer."""
    by_layer = {}
    for t in treatments:
        by_layer.setdefault(t["layer"], []).append(t)

    controls = {}  # (layer, t_expert) -> control record
    for layer, ts in by_layer.items():
        entry = scan[str(layer)]
        variance, _ = per_expert_variance(entry)
        n_experts = len(variance)
        total_hits = np.array([
            sum(int(entry[c]["hit_count"][e]) for c in LANG_CONDITIONS) for e in range(n_experts)])
        median_hits = float(np.median(total_hits))
        flagged_set = set(entry.get("disagreement_experts") or [])
        t_experts = {t["expert"] for t in ts}

        eligible = [
            e for e in range(n_experts)
            if e not in flagged_set and e not in t_experts and total_hits[e] >= median_hits
        ]
        eligible.sort(key=lambda e: variance[e])  # lowest dispersion first
        if len(eligible) < len(ts):
            raise RuntimeError(f"layer {layer}: only {len(eligible)} eligible controls for {len(ts)} treatments")
        for t, e in zip(sorted(ts, key=lambda t: t["expert"]), eligible):
            controls[(layer, t["expert"])] = expert_record(scan, layer, e)
    return controls


def main():
    scan = json.loads(SCAN_PATH.read_text())
    scan.pop("_meta", None)
    e1 = json.loads(E1_PATH.read_text())
    flagged = e1["flagged"]

    chosen_late, chosen_early = pick_treatment(scan, flagged)
    treatment_pairs = [{"layer": f["layer"], "expert": f["expert"], "e1_lang": f["lang"],
                        "band": "late" if f["layer"] in LATE_BAND else "early"}
                       for f in (chosen_late + chosen_early)]
    treatments = []
    for tp in treatment_pairs:
        rec = expert_record(scan, tp["layer"], tp["expert"])
        rec.update({"e1_lang": tp["e1_lang"], "band": tp["band"], "arm": "treatment"})
        treatments.append(rec)

    controls_map = pick_controls(scan, treatments)
    controls = []
    for t in treatments:
        c = controls_map[(t["layer"], t["expert"])]
        c.update({"arm": "control", "band": t["band"],
                  "matched_treatment_expert": t["expert"], "e1_lang": None})
        controls.append(c)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [{"layer": t["layer"], "treatment_expert": t["expert"],
              "control_expert": c["expert"], "band": t["band"], "e1_lang": t["e1_lang"],
              "t_disagreement_var": t["disagreement_var"], "c_disagreement_var": c["disagreement_var"]}
             for t, c in zip(treatments, controls)]

    out = {
        "design": {
            "compressed_model": "english_only seed0 (max cross-damage condition)",
            "eval": "FLORES devtest 60 sentences x 5 langs, delta-bpb%",
            "dispersion_metric": "disagreement_common.normalize_layer(minmax,clip=2)+rank_by_disagreement (row-wise cross-language variance)",
            "n_treatment": len(treatments), "n_control": len(controls),
            "late_band": [22, 23, 24, 25, 26, 27], "early_band": [1, 2, 3, 4, 5],
        },
        "lang_conditions": LANG_CONDITIONS,
        "pairs": pairs,
        "treatment": treatments,
        "control": controls,
    }
    (OUT_DIR / "targets.json").write_text(json.dumps(out, indent=2))

    # flat CSV: one row per selected expert
    fields = ["arm", "band", "layer", "expert", "role", "disagreement_var", "total_hits",
              "e1_lang", "matched_treatment_expert"] + \
             [f"hit_{c}" for c in LANG_CONDITIONS] + [f"proxy_{c}" for c in LANG_CONDITIONS] + \
             [f"nproxy_{c}" for c in LANG_CONDITIONS]
    with (OUT_DIR / "targets.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for rec in treatments + controls:
            row = {k: rec.get(k) for k in ["arm", "band", "layer", "expert", "role",
                                            "disagreement_var", "total_hits", "e1_lang",
                                            "matched_treatment_expert"]}
            for c in LANG_CONDITIONS:
                row[f"hit_{c}"] = rec["hit_count_by_lang"][c]
                row[f"proxy_{c}"] = rec["proxy_by_lang"][c]
                row[f"nproxy_{c}"] = rec["norm_proxy_by_lang"][c]
            w.writerow(row)

    # human-readable summary to stdout
    print(f"[sec54-select] wrote {OUT_DIR/'targets.json'} and targets.csv")
    print(f"[sec54-select] {len(treatments)} treatment / {len(controls)} control pairs\n")
    print(f"{'PAIR':>4}  {'layer':>5}  {'band':>5}  {'T_exp':>5} {'T_var':>7}  "
          f"{'C_exp':>5} {'C_var':>7}  {'e1_lang':>8}")
    for i, p in enumerate(pairs):
        print(f"{i:>4}  {p['layer']:>5}  {p['band']:>5}  {p['treatment_expert']:>5} "
              f"{p['t_disagreement_var']:>7.4f}  {p['control_expert']:>5} "
              f"{p['c_disagreement_var']:>7.4f}  {str(p['e1_lang']):>8}")


if __name__ == "__main__":
    main()
