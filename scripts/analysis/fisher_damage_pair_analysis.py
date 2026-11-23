"""Exploratory CPU analysis for reviewer G1rJ's language-similarity question:
across all 10 unordered pairs of {EN, ZH, KO, SW, BN}, how does the similarity
of the calibration languages' Fisher expert-importance rankings relate to the
performance damage one language's calibration inflicts on the other?

STRICTLY A RE-READ OF STORED RESULTS. No calibration, no Fisher, no merge, no
GPU. Every input already exists on disk; nothing is regenerated.

DEFINITIONS (fixed before any correlation is computed; not chosen after):

  F(A,B)   Fisher rank similarity. Table 17's definition: for each of the 27 MoE
           layers, the Spearman rho between calibration languages A and B's
           per-expert real-gradient Fisher importance, averaged over the 27
           layers. Read from phase1_51_fisher_rank_correlation_result_seed<s>.json
           (per_layer[*].lang_pairs[*].rho) -- the stored per-layer values, not
           the paper's rounded table.

  D(A->B)  Cross damage. Model calibrated on A, evaluated on B, as a relative
           bits-per-byte increase over the uncompressed baseline:
               D(A->B) = 100 * (bpb[A,s,B] / bpb[baseline,B] - 1)
           bpb[A,s,B] from <A>/seed<s>/scale_<A>_seed<s>/eval_ppl.json and
           bpb[baseline,B] from baseline/eval_ppl.json. This is exactly
           phase1_41_headline_gate.bpb_incr / diag_path (verified unchanged in
           the working tree), i.e. the whitening-ON, own-scale, delta_ratio 0.8,
           pruning-OFF configuration behind Table 1 / Table 13.

  Dsym(A,B) = [D(A->B) + D(B->A)] / 2
  Esym(A,B) = [(D(A->B) - D(B->B)) + (D(B->A) - D(A->A))] / 2
           i.e. mean damage in excess of the evaluated language's own-language
           damage.

  PRIMARY:    Spearman(F, Esym) over the 10 pairs.
  COMPARISON: Spearman(F, Dsym).  Both are always reported; neither is dropped.

Seed 0 is the primary analysis. The identical computation is repeated for seeds
1 and 2 ONLY because the same-definition inputs already exist for them. Fisher
and damage are matched within a seed; seeds are never mixed, and Table 1's
3-seed mean is never paired with seed-0 Fisher.

What is deliberately NOT computed: p-values. The 10 pairs share languages (each
language appears in 4 of them), so they are not independent, and multiplying
by 27 layers would not create 270 independent observations either. The output
is descriptive statistics plus an exploratory rank correlation. Placebo rows
are not used.

All arithmetic is on the stored full-precision values; rounding happens only
when a value is displayed.
"""
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RESULTS_ROOT = Path("results/deepseek")
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "source_data" / "fisher_damage_pairs"

# paper notation order EN/ZH/KO/SW/BN (00_docs notation conventions)
LANGS = [("EN", "english_only", "eng_Latn"),
         ("ZH", "chinese_only", "zho_Hans"),
         ("KO", "korean_only", "kor_Hang"),
         ("SW", "swahili_only", "swh_Latn"),
         ("BN", "bengali_only", "ben_Beng")]
COND = {k: c for k, c, _ in LANGS}
FLORES = {k: f for k, _, f in LANGS}
SEEDS = [0, 1, 2]
PRIMARY_SEED = 0

# Table 1 draft (00_docs/06_논문_구성.md, 3-seed mean, whitening-ON) -- used ONLY
# as a definition check that this script's D reproduces the documented formula.
# It is never an input to the correlation.
TABLE1_DRAFT = {("EN", "EN"): 7.28, ("EN", "KO"): 88.76, ("EN", "ZH"): 13.80, ("EN", "SW"): 33.09,
                ("KO", "EN"): 12.68, ("KO", "KO"): 3.86, ("KO", "ZH"): 12.68, ("KO", "SW"): 38.52,
                ("ZH", "EN"): 10.29, ("ZH", "KO"): 79.28, ("ZH", "ZH"): 4.24, ("ZH", "SW"): 35.10,
                ("ZH", "BN"): 87.41, ("SW", "EN"): 12.83, ("SW", "KO"): 66.65, ("SW", "ZH"): 15.20,
                ("SW", "SW"): -0.05, ("SW", "BN"): 75.06}


def fisher_pairs(seed):
    d = json.loads((RESULTS_ROOT / f"phase1_51_fisher_rank_correlation_result_seed{seed}.json").read_text())
    layers = sorted(d["per_layer"], key=int)
    per = {}
    nan_layers = {}
    for lay in layers:
        for key, v in d["per_layer"][lay]["lang_pairs"].items():
            a, b = key.split("|")
            fs = frozenset((a, b))
            r = v["rho"]
            per.setdefault(fs, []).append(r)
            if r is None or (isinstance(r, float) and math.isnan(r)):
                nan_layers.setdefault(fs, []).append(int(lay))
    out = {}
    for fs, vals in per.items():
        arr = np.array([np.nan if x is None else x for x in vals], dtype=float)
        out[fs] = {"F": float(np.nanmean(arr)), "n_layers": int(np.sum(~np.isnan(arr))),
                   "layers_nan": nan_layers.get(fs, [])}
    # consistency: mean over pairs of per-pair layer means must equal the stored
    # overall mean_lang_rho when there are no NaNs (mean of means commutes)
    stored_overall = d["summary"]["overall"]["mean_lang_rho"]
    recomputed = float(np.mean([v["F"] for v in out.values()]))
    return out, len(layers), {"stored_overall_mean_lang_rho": stored_overall,
                              "recomputed_mean_of_pair_means": recomputed,
                              "abs_diff": abs(stored_overall - recomputed)}


def damage(seed):
    base = json.loads((RESULTS_ROOT / "baseline" / "eval_ppl.json").read_text())
    D, paths = {}, {}
    for a, _, _ in LANGS:
        p = RESULTS_ROOT / COND[a] / f"seed{seed}" / f"scale_{COND[a]}_seed{seed}" / "eval_ppl.json"
        paths[a] = str(p)
        ev = json.loads(p.read_text())
        for b, _, _ in LANGS:
            D[(a, b)] = 100.0 * (ev[FLORES[b]]["bits_per_byte"] / base[FLORES[b]]["bits_per_byte"] - 1.0)
    return D, paths


def ranks_desc(values):
    """rank 1 = largest; ties get the average rank (same convention as Spearman)."""
    from scipy.stats import rankdata
    return rankdata([-v for v in values], method="average")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    order = [k for k, _, _ in LANGS]
    pairs = list(itertools.combinations(order, 2))   # 10 unordered pairs
    rows, per_seed, checks = [], {}, {}

    # definition check of D against the Table 1 draft (3-seed mean)
    Dall = {s: damage(s)[0] for s in SEEDS}
    t1 = []
    for (a, b), pub in TABLE1_DRAFT.items():
        m = float(np.mean([Dall[s][(a, b)] for s in SEEDS]))
        t1.append({"cell": f"{a}->{b}", "table1_draft": pub, "recomputed_3seed_mean": m,
                   "abs_diff_vs_2dp": abs(round(m, 2) - pub)})
    checks["damage_formula_vs_table1_draft"] = {
        "cells_compared": len(t1), "max_abs_diff_after_rounding_to_2dp": max(r["abs_diff_vs_2dp"] for r in t1),
        "n_exact_at_2dp": sum(1 for r in t1 if r["abs_diff_vs_2dp"] < 1e-9), "detail": t1,
        "note": ("the draft table predates later backfills and is a 2-dp display value, so agreement is "
                 "checked at 2 dp; it validates the formula and file choice, it is not an input"),
    }

    for s in SEEDS:
        Fp, nlay, fcheck = fisher_pairs(s)
        D, dpaths = damage(s)
        checks[f"fisher_seed{s}"] = {**fcheck, "n_layers": nlay}
        recs = []
        for a, b in pairs:
            fs = frozenset((COND[a], COND[b]))
            F = Fp[fs]["F"]
            dab, dba, daa, dbb = D[(a, b)], D[(b, a)], D[(a, a)], D[(b, b)]
            dsym = (dab + dba) / 2.0
            esym = ((dab - dbb) + (dba - daa)) / 2.0
            recs.append({"pair": f"{a}-{b}", "A": a, "B": b, "seed": s, "F": F,
                         "D_A_to_B": dab, "D_B_to_A": dba, "D_A_to_A": daa, "D_B_to_B": dbb,
                         "Dsym": dsym, "Esym": esym,
                         "fisher_layers_used": Fp[fs]["n_layers"], "fisher_layers_nan": Fp[fs]["layers_nan"]})
        Fv = [r["F"] for r in recs]
        Ev = [r["Esym"] for r in recs]
        Dv = [r["Dsym"] for r in recs]
        rho_E = float(spearmanr(Fv, Ev).correlation)
        rho_D = float(spearmanr(Fv, Dv).correlation)
        rF, rE, rD = ranks_desc(Fv), ranks_desc(Ev), ranks_desc(Dv)
        medF, medE, medD = float(np.median(Fv)), float(np.median(Ev)), float(np.median(Dv))
        for r, a1, a2, a3 in zip(recs, rF, rE, rD):
            r["rank_F_desc"] = float(a1); r["rank_Esym_desc"] = float(a2); r["rank_Dsym_desc"] = float(a3)
            # Pre-declared quadrant rule (median split, fixed before looking):
            # "above-median Fisher similarity AND above-median excess damage" is
            # the case a simple "similar Fisher => less damage" story does not fit.
            r["high_F_high_Esym"] = bool(r["F"] > medF and r["Esym"] > medE)
            r["low_F_low_Esym"] = bool(r["F"] < medF and r["Esym"] < medE)
            r["high_F_high_Dsym"] = bool(r["F"] > medF and r["Dsym"] > medD)
        ko_bn = next(r for r in recs if {r["A"], r["B"]} == {"KO", "BN"})
        per_seed[s] = {
            "spearman_F_Esym": rho_E, "spearman_F_Dsym": rho_D,
            "median_F": medF, "median_Esym": medE, "median_Dsym": medD,
            "KO_BN": {"F": ko_bn["F"], "rank_F_desc": ko_bn["rank_F_desc"],
                      "Esym": ko_bn["Esym"], "rank_Esym_desc": ko_bn["rank_Esym_desc"],
                      "Dsym": ko_bn["Dsym"], "rank_Dsym_desc": ko_bn["rank_Dsym_desc"]},
            "high_F_high_Esym_pairs": [r["pair"] for r in recs if r["high_F_high_Esym"]],
            "high_F_high_Dsym_pairs": [r["pair"] for r in recs if r["high_F_high_Dsym"]],
            "damage_files": dpaths,
            "records": recs,
        }
        rows.extend(recs)

    fields = ["pair", "A", "B", "seed", "F", "D_A_to_B", "D_B_to_A", "D_A_to_A", "D_B_to_B", "Dsym", "Esym",
              "rank_F_desc", "rank_Esym_desc", "rank_Dsym_desc", "high_F_high_Esym", "low_F_low_Esym",
              "high_F_high_Dsym", "fisher_layers_used"]
    with open(OUT_DIR / "fisher_damage_pairs_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)          # full precision; rounding only in the report

    summary = {
        "question": "Fisher expert-ranking similarity vs cross-language compression damage, 10 language pairs",
        "primary_seed": PRIMARY_SEED, "seeds_analysed": SEEDS,
        "definitions": {
            "F": "27-layer mean of per-layer Spearman rho between the two languages' per-expert Fisher (Table 17)",
            "D(A->B)": "100*(bpb[A,seed,B]/bpb[baseline,B]-1); whitening-ON own-scale, dr 0.8, pruning OFF",
            "Dsym": "[D(A->B)+D(B->A)]/2",
            "Esym": "[(D(A->B)-D(B->B)) + (D(B->A)-D(A->A))]/2",
            "primary": "Spearman(F, Esym)", "comparison": "Spearman(F, Dsym)",
            "quadrant_rule": "median split on the 10 pairs, declared before computing",
        },
        "not_computed": ["p-values (pairs share languages; not independent)",
                         "any 270-observation layer-level test", "placebo rows"],
        "per_seed": {str(s): {k: v for k, v in per_seed[s].items() if k != "records"} for s in SEEDS},
        "consistency_checks": checks,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for s in SEEDS:
        p = per_seed[s]
        print(f"[fisher-pairs] seed{s}: rho(F,Esym)={p['spearman_F_Esym']:+.4f}  "
              f"rho(F,Dsym)={p['spearman_F_Dsym']:+.4f}  "
              f"KO-BN rank F={p['KO_BN']['rank_F_desc']:.1f} Esym={p['KO_BN']['rank_Esym_desc']:.1f} "
              f"Dsym={p['KO_BN']['rank_Dsym_desc']:.1f}")
    c = checks["damage_formula_vs_table1_draft"]
    print(f"[fisher-pairs] Table1-draft check: {c['n_exact_at_2dp']}/{c['cells_compared']} exact at 2dp, "
          f"max diff {c['max_abs_diff_after_rounding_to_2dp']:.2f}")
    for s in SEEDS:
        fc = checks[f"fisher_seed{s}"]
        print(f"[fisher-pairs] Fisher seed{s}: stored overall={fc['stored_overall_mean_lang_rho']:.6f} "
              f"recomputed={fc['recomputed_mean_of_pair_means']:.6f} diff={fc['abs_diff']:.2e}")
    print(f"[fisher-pairs] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
