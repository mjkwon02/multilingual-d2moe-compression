"""Aggregate the revised MASSIVE run: metrics.csv, paired_contrasts.csv,
completion_grid.csv, summary.json.

Primary contrast is Balanced-character MINUS English at the same seed.
Balanced MINUS own-language is NOT produced -- the ZH/KO/SW/BN own-language
models are not evaluated in this run, so that contrast has no data and is not
reported.

BOOTSTRAP (protocol.json::uncertainty). 2,000 replicates, seed 42. Each
replicate resamples the 200 source IDs ONCE and applies that same resampled ID
set to all 5 languages and all 3 calibration seeds, then averages the seed
contrasts. This preserves both the translation correspondence across languages
(the same source utterance is in or out of a replicate in every language) and
the pairing between the two models being compared. The interval is
test-sampling uncertainty CONDITIONAL on the three observed calibration seeds;
across-seed variability is reported separately as sample std (ddof=1). The two
are never merged, and the three seeds are never treated as 3x the test sample.

A cell that was not run is reported as missing in completion_grid.csv and is
excluded from every aggregate -- never imputed, never averaged over, never
filled from another seed. Contrasts use only IDs completed on BOTH sides and
state the resulting n.
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import massive_data as M  # noqa: E402

NEW_DIR = SCRIPT_DIR.parents[1] / "results" / "downstream" / "massive_revised"
CONDITIONS = ["english_only", "mixed_5lang"]
COND_LABEL = {"english_only": "English", "mixed_5lang": "Balanced-character", "baseline": "baseline"}
SEEDS = [0, 1, 2]
N_BOOT = 2000
BOOT_SEED = 42


def macro_f1(gold, pred, labels):
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        f1s.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s) if f1s else float("nan")


def entropy(counts, n):
    import math
    h = 0.0
    for c in counts.values():
        if c:
            p = c / n
            h -= p * math.log(p)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    protocol = json.loads((NEW_DIR / "protocol.json").read_text())
    vp = json.loads((NEW_DIR / "validation_and_pilot.json").read_text())
    sig = vp["inference_signature"]
    test_ids = json.loads((NEW_DIR / ("test_ids.json" if args.split == "test" else "dev_ids.json")).read_text())["ids"]
    langs = list(M.LOCALES)

    # ------------------------------------------------------- load rows -----
    rows = collections.defaultdict(dict)   # (model, lang) -> {id: (gold, pred)}
    pred_path = NEW_DIR / "predictions.jsonl"
    if pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("inference_signature") != sig or r.get("split") != args.split:
                    continue
                rows[(r["model"], r["lang"])][r["id"]] = (r["gold"], r["pred"])

    models = ["baseline"] + [f"{c}_seed{s}" for c in CONDITIONS for s in SEEDS]

    # --------------------------------------------------- completion grid ---
    grid = []
    for m in models:
        for l in langs:
            got = rows.get((m, l), {})
            grid.append({"model": m, "condition": m.split("_seed")[0] if "_seed" in m else "baseline",
                         "seed": (m.split("_seed")[1] if "_seed" in m else ""),
                         "lang": l, "n_expected": len(test_ids), "n_completed": len(got),
                         "status": ("COMPLETE" if len(got) == len(test_ids)
                                    else ("NOT_RUN" if not got else "PARTIAL"))})
    with open(NEW_DIR / "completion_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0].keys()))
        w.writeheader(); w.writerows(grid)

    # --------------------------------------------------------- metrics -----
    acc, mrows = {}, []
    for m in models:
        for l in langs:
            got = rows.get((m, l), {})
            if not got:
                continue
            ids = [i for i in test_ids if i in got]
            gold = [got[i][0] for i in ids]
            pred = [got[i][1] for i in ids]
            n = len(ids)
            a = sum(g == p for g, p in zip(gold, pred)) / n
            acc[(m, l)] = a
            sup = sorted(set(gold))
            cnt = collections.Counter(pred)
            base = acc.get(("baseline", l))
            mrows.append({
                "model": m, "condition": COND_LABEL.get(m.split("_seed")[0] if "_seed" in m else "baseline"),
                "seed": (m.split("_seed")[1] if "_seed" in m else ""),
                "lang": l, "locale": M.LOCALES[l], "n": n, "n_expected": len(test_ids),
                "is_sample_not_full_test_split": True,
                "acc_pct": round(100 * a, 4),
                "uniform_chance_pct": round(100 / 60, 4),
                "macro_f1_support": round(macro_f1(gold, pred, sup), 6),
                "n_labels_with_support": len(sup),
                "macro_f1_fixed60": round(macro_f1(gold, pred, M.PROMPT_LABEL_ORDER), 6),
                "drop_pp_vs_baseline": "" if base is None else round(100 * (base - a), 4),
                "retention_vs_baseline": "" if not base else round(a / base, 6),
                "top_label_share": round(cnt.most_common(1)[0][1] / n, 6),
                "pred_entropy_nats": round(entropy({k: cnt.get(k, 0) for k in M.PROMPT_LABEL_ORDER}, n), 6),
                "n_distinct_predicted": len(cnt),
            })
    with open(NEW_DIR / "metrics.csv", "w", newline="") as f:
        if mrows:
            w = csv.DictWriter(f, fieldnames=list(mrows[0].keys()))
            w.writeheader(); w.writerows(mrows)

    # confusion matrices (fixed 60x60) stored separately, not in the csv
    conf = {}
    for m in models:
        for l in langs:
            got = rows.get((m, l), {})
            if not got:
                continue
            cm = [[0] * 60 for _ in range(60)]
            for i, (g, p) in got.items():
                cm[M.LABEL_TO_IDX[g]][M.LABEL_TO_IDX[p]] += 1
            conf[f"{m}/{l}"] = cm
    (NEW_DIR / "confusion_matrices.json").write_text(json.dumps(
        {"label_order": M.PROMPT_LABEL_ORDER, "matrices": conf}))

    # ----------------------------------------------- paired bootstrap ------
    # correct[(cond, seed, lang)] aligned to the ids completed on BOTH arms in
    # every seed and language that is fully paired.
    contrasts = []
    rng = np.random.default_rng(BOOT_SEED)
    paired_seeds = [s for s in SEEDS
                    if all(rows.get((f"{c}_seed{s}", l)) for c in CONDITIONS for l in langs)]
    if paired_seeds:
        common = set(test_ids)
        for s in paired_seeds:
            for c in CONDITIONS:
                for l in langs:
                    common &= set(rows[(f"{c}_seed{s}", l)])
        common_ids = [i for i in test_ids if i in common]
        n = len(common_ids)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        corr = {}
        for s in paired_seeds:
            for c in CONDITIONS:
                for l in langs:
                    g = rows[(f"{c}_seed{s}", l)]
                    corr[(c, s, l)] = np.array([float(g[i][0] == g[i][1]) for i in common_ids])

        # per-language contrast
        for l in langs:
            reps = np.zeros(N_BOOT)
            per_seed = {}
            for s in paired_seeds:
                b = corr[("mixed_5lang", s, l)]
                e = corr[("english_only", s, l)]
                per_seed[s] = 100 * float(b.mean() - e.mean())
                reps += 100 * (b[idx].mean(axis=1) - e[idx].mean(axis=1))
            reps /= len(paired_seeds)
            pt = float(np.mean([per_seed[s] for s in paired_seeds]))
            lo, hi = np.percentile(reps, [2.5, 97.5])
            sd = float(np.std([per_seed[s] for s in paired_seeds], ddof=1)) if len(paired_seeds) > 1 else float("nan")
            contrasts.append({
                "contrast": "Balanced_minus_English", "scope": l, "locale": M.LOCALES[l],
                "n_paired_ids": n, "n_seeds": len(paired_seeds),
                "seeds_used": ",".join(map(str, paired_seeds)),
                "seed_mean_contrast_pp": round(pt, 4),
                "ci95_lo_pp": round(float(lo), 4), "ci95_hi_pp": round(float(hi), 4),
                "across_seed_sample_std_pp_ddof1": round(sd, 4),
                "per_seed_contrast_pp": json.dumps({str(s): round(per_seed[s], 4) for s in paired_seeds}),
                "n_seeds_favouring_balanced": sum(1 for s in paired_seeds if per_seed[s] > 0),
                "all_seeds_same_direction": len({per_seed[s] > 0 for s in paired_seeds}) == 1,
            })
        # language-macro contrast, same resampled IDs
        reps = np.zeros(N_BOOT)
        per_seed = {}
        for s in paired_seeds:
            bs = np.mean([corr[("mixed_5lang", s, l)].mean() for l in langs])
            es = np.mean([corr[("english_only", s, l)].mean() for l in langs])
            per_seed[s] = 100 * float(bs - es)
            reps += 100 * (np.mean([corr[("mixed_5lang", s, l)][idx].mean(axis=1) for l in langs], axis=0)
                           - np.mean([corr[("english_only", s, l)][idx].mean(axis=1) for l in langs], axis=0))
        reps /= len(paired_seeds)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        sd = float(np.std([per_seed[s] for s in paired_seeds], ddof=1)) if len(paired_seeds) > 1 else float("nan")
        contrasts.append({
            "contrast": "Balanced_minus_English", "scope": "LANGUAGE_MACRO", "locale": "-",
            "n_paired_ids": n, "n_seeds": len(paired_seeds), "seeds_used": ",".join(map(str, paired_seeds)),
            "seed_mean_contrast_pp": round(float(np.mean([per_seed[s] for s in paired_seeds])), 4),
            "ci95_lo_pp": round(float(lo), 4), "ci95_hi_pp": round(float(hi), 4),
            "across_seed_sample_std_pp_ddof1": round(sd, 4),
            "per_seed_contrast_pp": json.dumps({str(s): round(per_seed[s], 4) for s in paired_seeds}),
            "n_seeds_favouring_balanced": sum(1 for s in paired_seeds if per_seed[s] > 0),
            "all_seeds_same_direction": len({per_seed[s] > 0 for s in paired_seeds}) == 1,
        })
    if contrasts:
        with open(NEW_DIR / "paired_contrasts.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(contrasts[0].keys()))
            w.writeheader(); w.writerows(contrasts)

    # --------------------------------------------------------- summary -----
    done = sum(1 for g in grid if g["status"] == "COMPLETE")
    per_cond = {}
    for c in CONDITIONS:
        got = {l: [acc[(f"{c}_seed{s}", l)] for s in SEEDS if (f"{c}_seed{s}", l) in acc] for l in langs}
        if not all(got.values()):
            per_cond[c] = {"incomplete": True,
                           "languages_with_any_seed": [l for l, v in got.items() if v]}
            continue
        mean = {l: float(np.mean(got[l])) for l in langs}
        worst = min(mean, key=mean.get)
        per_cond[c] = {
            "per_language_mean_acc_pct": {l: round(100 * mean[l], 3) for l in langs},
            "per_language_std_pct_ddof1": {l: (round(100 * float(np.std(got[l], ddof=1)), 3)
                                               if len(got[l]) > 1 else None) for l in langs},
            "per_language_per_seed_acc_pct": {l: {s: round(100 * acc[(f"{c}_seed{s}", l)], 3)
                                                  for s in SEEDS if (f"{c}_seed{s}", l) in acc} for l in langs},
            "n_seeds_per_language": {l: len(got[l]) for l in langs},
            "language_macro_acc_pct": round(100 * float(np.mean([mean[l] for l in langs])), 3),
            "worst_language": worst, "worst_language_acc_pct": round(100 * mean[worst], 3),
        }
    summary = {
        "split": args.split, "n_items_per_language": len(test_ids),
        "is_sample_not_full_test_split": True,
        "prompt": vp["prompt_selection"]["selected"],
        "scoring_mode": vp["microbatch_probe"]["scoring_mode"],
        "microbatch": vp["microbatch_probe"]["selected_microbatch"],
        "inference_signature": sig, "protocol_id": protocol["protocol_id"],
        "cells_complete": done, "cells_expected": len(grid),
        "baseline_acc_pct": {l: (round(100 * acc[("baseline", l)], 3) if ("baseline", l) in acc else None)
                             for l in langs},
        "baseline_language_macro_acc_pct": (
            round(100 * float(np.mean([acc[("baseline", l)] for l in langs])), 3)
            if all(("baseline", l) in acc for l in langs) else None),
        "per_condition": per_cond,
        "paired_seeds_used_for_contrast": paired_seeds if paired_seeds else [],
        "missing_cells": [f"{g['model']}/{g['lang']}" for g in grid if g["status"] != "COMPLETE"],
    }
    (NEW_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[agg] cells {done}/{len(grid)} complete; paired seeds={paired_seeds}")
    print(f"[agg] wrote metrics.csv ({len(mrows)}), paired_contrasts.csv ({len(contrasts)}), "
          "completion_grid.csv, confusion_matrices.json, summary.json")


if __name__ == "__main__":
    main()
