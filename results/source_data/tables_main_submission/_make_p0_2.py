#!/usr/bin/env python3
"""P0-2  Table 4 final -- 5 pruning conditions x 5 eval languages.

PROVENANCE
  reference json : results/table4_pruning_lang.json  (4 conditions only -- NO chinese_only)
  raw eval files : results/deepseek/{cond}/seed{s}/eval_ppl.json            (pruning OFF)
                   results/deepseek/{cond}/seed{s}/pp_ratio_0.2/eval_ppl.json (pruning ON)
                   results/deepseek/baseline/eval_ppl.json                   (baseline)
  seeds          : 0, 1, 2 for every condition
  metric (unchanged, from scripts/phase1_pruning_gate.py bpb_increase_pct / delta,
          restated in 00_docs/TABLE4_PRUNING_LANG_SOURCE.md §2):
      ddbpb%(cond,lang) = 100*(bpb_ON/bpb_base - 1) - 100*(bpb_OFF/bpb_base - 1)
                        = 100*(bpb_ON - bpb_OFF)/bpb_base
  The chinese_only row is NOT in table4_pruning_lang.json; it is recomputed here from
  the same raw files with the same formula. No re-run, no new definition.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
R = Path("results/deepseek")

LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
SHORT = {"eng_Latn": "EN", "kor_Hang": "KO", "zho_Hans": "ZH", "swh_Latn": "SW", "ben_Beng": "BN"}
VULN = ["kor_Hang", "swh_Latn", "ben_Beng"]
CONDS = ["english_only", "chinese_only", "korean_only", "swahili_only", "bengali_only"]
SEEDS = [0, 1, 2]

base = {l: json.loads((R / "baseline/eval_ppl.json").read_text())[l]["bits_per_byte"] for l in LANGS}


def bpb(cond, seed, pruned):
    p = R / cond / f"seed{seed}" / ("pp_ratio_0.2/eval_ppl.json" if pruned else "eval_ppl.json")
    return json.loads(p.read_text()), p


res = {}
missing = []
for c in CONDS:
    res[c] = {"per_lang": {}, "vulnerable_mean": {}, "source_files": {}}
    per_seed_vuln = []
    cells = {l: [] for l in LANGS}
    for s in SEEDS:
        try:
            on, pon = bpb(c, s, True)
            off, poff = bpb(c, s, False)
        except FileNotFoundError as e:
            missing.append(f"{c}/seed{s}: {e}")
            continue
        res[c]["source_files"][f"seed{s}"] = {"ON": str(pon), "OFF": str(poff)}
        vals = {}
        for l in LANGS:
            d = 100.0 * (on[l]["bits_per_byte"] - off[l]["bits_per_byte"]) / base[l]
            cells[l].append(d)
            vals[l] = d
        per_seed_vuln.append(sum(vals[l] for l in VULN) / 3)
    for l in LANGS:
        v = cells[l]
        res[c]["per_lang"][l] = {
            "per_seed": v,
            "mean": sum(v) / len(v),
            "sign_consistent": len({1 if x > 0 else (-1 if x < 0 else 0) for x in v}) == 1,
            "n_seeds": len(v),
        }
    res[c]["vulnerable_mean"] = {"per_seed": per_seed_vuln, "mean": sum(per_seed_vuln) / len(per_seed_vuln)}

# ---- cross-check against the existing 4-condition json ----
ref = json.loads((ROOT / "results/source_data/table4_pruning_lang.json").read_text())
xcheck = {"json_conditions": sorted(ref["conditions"].keys()),
          "n_conditions_in_json": len(ref["conditions"]),
          "conditions_recomputed_here": CONDS,
          "chinese_only_in_json": "chinese_only" in ref["conditions"],
          "per_cell": {}}
maxdiff = 0.0
for c, cd in ref["conditions"].items():
    for l in LANGS:
        a = cd["per_lang"][l]["mean"]
        b = res[c]["per_lang"][l]["mean"]
        xcheck["per_cell"][f"{c}|{l}"] = {"json_mean": a, "recomputed_mean": b, "abs_diff": abs(a - b)}
        maxdiff = max(maxdiff, abs(a - b))
xcheck["max_abs_diff_vs_json"] = maxdiff
xcheck["recompute_reproduces_json"] = maxdiff < 1e-9

# ---- cross-check against the official per-condition pruning-gate outputs ----
gate = {"note": ("phase1_pruning_gate.py official outputs, one json per condition. These are the "
                 "authoritative per-cell numbers; chinese_only HAS one (2026-08-02 05:04) even though "
                 "it is missing from table4_pruning_lang.json."), "per_condition": {}}
gmax = 0.0
for c in CONDS:
    p = R / f"phase1_pruning_gate_{c}_result.json"
    if not p.exists():
        gate["per_condition"][c] = {"status": "MISSING", "path": str(p)}
        continue
    g = json.loads(p.read_text())
    cells = {}
    for l in LANGS:
        if l not in g["per_language"]:
            cells[SHORT[l]] = {"status": "not_evaluated_in_gate_json"}
            continue
        a = g["per_language"][l]["mean_delta_pp"]
        b = res[c]["per_lang"][l]["mean"]
        cells[SHORT[l]] = {"gate_mean": a, "recomputed_mean": b, "abs_diff": abs(a - b),
                           "gate_sign_consistent": g["per_language"][l]["sign_consistent_across_seeds"],
                           "recomputed_sign_consistent": res[c]["per_lang"][l]["sign_consistent"],
                           "direction": g["per_language"][l].get("direction")}
        gmax = max(gmax, abs(a - b))
    gate["per_condition"][c] = {"status": "present", "path": str(p), "pp_ratio": g.get("pp_ratio"),
                                "langs_in_gate_json": sorted(g["per_language"].keys()), "cells": cells}
gate["max_abs_diff_vs_gate_jsons"] = gmax
xcheck["official_gate_jsons"] = gate

# ---- diff vs the numbers already written in the body ----
BODY = {  # 00_docs/08_figure_정리.md Table 4 + the chinese_only row quoted in the data request
    "english_only": {"EN": 0.29, "KO": 0.71, "ZH": 0.31, "SW": 0.61, "BN": 1.17, "vuln": 0.83},
    "chinese_only": {"EN": 0.67, "KO": 1.20, "ZH": 0.38, "SW": 1.17, "BN": 2.11, "vuln": None},
    "korean_only": {"EN": -0.53, "KO": 2.12, "ZH": 0.91, "SW": 0.76, "BN": 4.41, "vuln": 2.43},
    "swahili_only": {"EN": 1.97, "KO": 3.51, "ZH": 1.56, "SW": 1.82, "BN": 5.70, "vuln": 3.68},
    "bengali_only": {"EN": 2.37, "KO": 5.30, "ZH": 1.91, "SW": 2.03, "BN": 3.92, "vuln": 3.75},
}
bodydiff = {}
for c in CONDS:
    for l in LANGS:
        want = BODY[c][SHORT[l]]
        got = round(res[c]["per_lang"][l]["mean"], 2)
        bodydiff[f"{c}|{SHORT[l]}"] = {"body": want, "final_json_2dp": got,
                                       "computed_raw": res[c]["per_lang"][l]["mean"],
                                       "match": want == got}
    if BODY[c]["vuln"] is not None:
        want = BODY[c]["vuln"]
        got = round(res[c]["vulnerable_mean"]["mean"], 2)
        bodydiff[f"{c}|vulnerable_mean"] = {"body": want, "final_json_2dp": got,
                                            "computed_raw": res[c]["vulnerable_mean"]["mean"],
                                            "match": want == got}

order = sorted(CONDS, key=lambda c: res[c]["vulnerable_mean"]["mean"])
body_order = ["english_only", "korean_only", "swahili_only", "bengali_only"]  # 0.83 < 2.43 < 3.68 < 3.75
body_order_5 = None

out = {
    "provenance": {
        "generated_utc": "2026-08-02",
        "reference_json": "results/source_data/table4_pruning_lang.json",
        "raw_root": str(R),
        "baseline_file": str(R / "baseline/eval_ppl.json"),
        "seeds": SEEDS,
        "pp_ratio": 0.2,
        "metric": "delta_delta_bpb_pct_pp (ON - OFF) = 100*(bpb_ON - bpb_OFF)/bpb_baseline",
        "note": ("chinese_only is absent from table4_pruning_lang.json and was recomputed here from the "
                 "same raw eval_ppl.json files with the identical formula; the other 4 conditions were "
                 "recomputed too and reproduce the json exactly (see cross_check)."),
    },
    "baseline_bpb": base,
    "conditions": res,
    "cross_check_vs_table4_json": xcheck,
    "diff_vs_body_text": bodydiff,
    "missing": missing,
    "vulnerable_mean_ordering": {
        "computed_ascending": [(c, res[c]["vulnerable_mean"]["mean"]) for c in order],
        "body_claim_4cond": "0.83 < 1.49 < 2.43 < 3.68 < 3.75 (5 values quoted in the request)",
        "body_claim_4cond_matches": None,
    },
}
(OUT / "p0_2_table4_final.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

# ---------------- csv ----------------
with (OUT / "p0_2_table4_final.csv").open("w") as f:
    f.write("# Table 4 final -- structured pruning extra damage ddbpb%% (ON - OFF), pp_ratio=0.2, 5 conditions x 5 eval langs\n")
    f.write("# metric: 100*(bpb_ON - bpb_OFF)/bpb_baseline  (phase1_pruning_gate.py definition, unchanged)\n")
    f.write("# provenance: {cond}/seed{0,1,2}/[pp_ratio_0.2/]eval_ppl.json + baseline/eval_ppl.json under %s\n" % R)
    f.write("# chinese_only recomputed here (absent from results/table4_pruning_lang.json); other 4 rows reproduce that json to <1e-9\n")
    f.write("condition,eval_lang,seed0,seed1,seed2,mean_raw,mean_2dp,sign_consistent\n")
    for c in CONDS:
        for l in LANGS:
            d = res[c]["per_lang"][l]
            f.write("%s,%s,%.6f,%.6f,%.6f,%.6f,%.2f,%s\n" % (
                c, SHORT[l], d["per_seed"][0], d["per_seed"][1], d["per_seed"][2],
                d["mean"], d["mean"], d["sign_consistent"]))
        v = res[c]["vulnerable_mean"]
        f.write("%s,VULN_MEAN,%.6f,%.6f,%.6f,%.6f,%.2f,\n" % (
            c, v["per_seed"][0], v["per_seed"][1], v["per_seed"][2], v["mean"], v["mean"]))

print(json.dumps({
    "n_conditions_in_json": xcheck["n_conditions_in_json"],
    "json_conditions": xcheck["json_conditions"],
    "recompute_reproduces_json": xcheck["recompute_reproduces_json"],
    "max_abs_diff_vs_json": maxdiff,
    "chinese_only_row_2dp": {SHORT[l]: round(res["chinese_only"]["per_lang"][l]["mean"], 2) for l in LANGS},
    "chinese_only_vuln": res["chinese_only"]["vulnerable_mean"]["mean"],
    "chinese_only_sign_consistency": {SHORT[l]: res["chinese_only"]["per_lang"][l]["sign_consistent"] for l in LANGS},
    "body_mismatches": {k: v for k, v in bodydiff.items() if not v["match"]},
    "gate_json_status": {c: xcheck["official_gate_jsons"]["per_condition"][c]["status"] for c in CONDS},
    "gate_json_langs": {c: xcheck["official_gate_jsons"]["per_condition"][c].get("langs_in_gate_json") for c in CONDS},
    "max_abs_diff_vs_gate_jsons": xcheck["official_gate_jsons"]["max_abs_diff_vs_gate_jsons"],
    "vuln_ordering": [(c, round(res[c]["vulnerable_mean"]["mean"], 2)) for c in order],
    "korean_EN_per_seed": res["korean_only"]["per_lang"]["eng_Latn"]["per_seed"],
    "missing": missing,
}, indent=1))
