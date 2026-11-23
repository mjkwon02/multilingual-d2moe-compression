#!/usr/bin/env python3
"""P0-3  Table 6 source -- 4 candidate stability axes for Balanced (§7.2).

PROVENANCE
  axis 1/2 (bpb regret, worst-language)
      results/deepseek/phase1_41_headline_gate_result.json  ("grid" field, 6 conds x 5 langs x 3 seeds)
      = the official Table 1 source (scripts/phase1_41_headline_gate.py)
  axis 3 (SIB-200 collapse cells)
      results/deepseek/sib200_extended/SIB_EXTENDED_REPORT.md  (per-seed acc + top-label)
      raw runs: sib200_extended/*.json and {korean,swahili}_only/seed*/scale_*/eval_sib200.json
  axis 4 (Belebele macro retention)
      results/deepseek/phase1_belebele_gate_{mixed_5lang,english_only,chinese_only}_offonly_result.json
      results/deepseek/phase1_belebele_gate_korean_only_result.json
  seeds : 0, 1, 2 everywhere
  DEFINITIONS -- reused, not invented:
      regret(cond,lang) = dbpb%(cond,lang) - min over the 6 conditions of dbpb%(.,lang)
        (verified against the body's Balanced row EN 2.01 / KO 1.50 / ZH 2.86 / SW 2.88 / BN 1.57)
      collapse (SIB-200) = top-label fraction > 0.9 OR acc < 0.25   (SIB_EXTENDED_REPORT.md header)
      Belebele macro retention = mean retention over NON-flagged languages
        (phase1_belebele_gate.py; flagged = at-chance by phase1_belebele_floor_check.py)
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
R = Path("results/deepseek")

LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
SHORT = {"eng_Latn": "EN", "kor_Hang": "KO", "zho_Hans": "ZH", "swh_Latn": "SW", "ben_Beng": "BN"}
CONDS = ["english_only", "korean_only", "chinese_only", "swahili_only", "bengali_only", "mixed_5lang"]
LABEL = {"mixed_5lang": "Balanced (mixed_5lang)"}
SEEDS = [0, 1, 2]

out = {"provenance": {
    "generated_utc": "2026-08-02",
    "seeds": SEEDS,
    "sources": {
        "bpb_grid": str(R / "phase1_41_headline_gate_result.json"),
        "sib200": str(R / "sib200_extended/SIB_EXTENDED_REPORT.md"),
        "belebele": [str(R / f"phase1_belebele_gate_{c}_offonly_result.json") for c in
                     ["mixed_5lang", "english_only", "chinese_only"]] +
                    [str(R / "phase1_belebele_gate_korean_only_result.json")],
    },
    "definitions": {
        "regret": "dbpb%(cond,lang) - min_over_6_conditions dbpb%(.,lang); units %p",
        "collapse": "top-label fraction > 0.9 OR acc < 0.25  (SIB_EXTENDED_REPORT.md)",
        "belebele_macro": "mean retention over non-flagged languages (phase1_belebele_gate.py)",
    }}}

# ---------------- axis 1 & 2 : bpb regret / worst language ----------------
grid = json.loads((R / "phase1_41_headline_gate_result.json").read_text())["grid"]
mean = {c: {l: grid[f"{c}|{l}"]["mean"] for l in LANGS} for c in CONDS}
per_seed = {c: {l: grid[f"{c}|{l}"]["per_seed"] for l in LANGS} for c in CONDS}
colmin = {l: min(mean[c][l] for c in CONDS) for l in LANGS}
colmin_cond = {l: min(CONDS, key=lambda c: mean[c][l]) for l in LANGS}

regret = {c: {l: mean[c][l] - colmin[l] for l in LANGS} for c in CONDS}
out["axis1_bpb_regret"] = {
    "column_min_dbpb_pct": colmin,
    "column_min_condition": colmin_cond,
    "per_condition": {c: {
        "dbpb_pct_mean": mean[c],
        "regret_pp": regret[c],
        "regret_pp_2dp": {SHORT[l]: round(regret[c][l], 2) for l in LANGS},
        "max_regret_pp": max(regret[c].values()),
        "max_regret_lang": SHORT[max(LANGS, key=lambda l: regret[c][l])],
        "mean_regret_pp": sum(regret[c].values()) / 5,
    } for c in CONDS},
}
BODY_BAL = {"EN": 2.01, "KO": 1.50, "ZH": 2.86, "SW": 2.88, "BN": 1.57}
out["axis1_bpb_regret"]["balanced_row_check_vs_body"] = {
    SHORT[l]: {"body": BODY_BAL[SHORT[l]], "computed_2dp": round(regret["mixed_5lang"][l], 2),
               "computed_raw": regret["mixed_5lang"][l],
               "match": BODY_BAL[SHORT[l]] == round(regret["mixed_5lang"][l], 2)} for l in LANGS}
out["axis1_bpb_regret"]["balanced_row_all_match"] = all(
    v["match"] for v in out["axis1_bpb_regret"]["balanced_row_check_vs_body"].values())

out["axis2_worst_language"] = {c: {
    "worst_lang": SHORT[max(LANGS, key=lambda l: mean[c][l])],
    "worst_dbpb_pct": max(mean[c].values()),
    "worst_dbpb_pct_2dp": round(max(mean[c].values()), 2),
    "per_seed_worst_lang": [SHORT[max(LANGS, key=lambda l: per_seed[c][l][s])] for s in SEEDS],
    "worst_lang_seed_consistent": len({max(LANGS, key=lambda l: per_seed[c][l][s]) for s in SEEDS}) == 1,
    "per_seed_worst_value": [max(per_seed[c][l][s] for l in LANGS) for s in SEEDS],
} for c in CONDS}

# ---------------- axis 3 : SIB-200 collapse cells ----------------
report = (R / "sib200_extended/SIB_EXTENDED_REPORT.md").read_text()
sec = report.split("Per-seed raw accuracies")[1].split("(**C** = collapsed")[0]
sib = {}
for line in sec.strip().splitlines():
    m = re.match(r"\|\s*(\w+)/seed(\d)\s*\|(.+)\|\s*$", line)
    if not m:
        continue
    cond, seed, rest = m.group(1), int(m.group(2)), m.group(3)
    cells = [x.strip() for x in rest.split("|")]
    assert len(cells) == 5, (cond, seed, cells)
    sib.setdefault(cond, {})[seed] = {}
    for l, cell in zip(LANGS, cells):
        mm = re.match(r"([\d.]+)\s*\(([\d.]+)\)(\s*\*\*C\*\*)?", cell)
        acc, top, flag = float(mm.group(1)), float(mm.group(2)), bool(mm.group(3))
        derived = (top > 0.9) or (acc < 0.25)
        sib[cond][seed][l] = {"acc": acc, "top_label_frac": top,
                              "collapsed_in_report": flag, "collapsed_by_rule": derived,
                              "rule_agrees": flag == derived}

axis3 = {}
for c in CONDS:
    key = c
    percond = sib[key]
    lang_cells = {}
    full = partial = 0
    for l in LANGS:
        n = sum(1 for s in SEEDS if percond[s][l]["collapsed_in_report"])
        lang_cells[SHORT[l]] = {"n_collapsed_seeds": n, "of": 3, "label": f"{n}/3",
                                "per_seed_acc": [percond[s][l]["acc"] for s in SEEDS],
                                "per_seed_top_label_frac": [percond[s][l]["top_label_frac"] for s in SEEDS],
                                "cell_class": "full" if n == 3 else ("partial" if n > 0 else "clean")}
        if n == 3:
            full += 1
        elif n > 0:
            partial += 1
    axis3[c] = {"per_lang": lang_cells, "n_full_collapse_cells": full, "n_partial_collapse_cells": partial,
                "n_clean_cells": 5 - full - partial,
                "n_collapsed_seed_cells_of_15": sum(v["n_collapsed_seeds"] for v in lang_cells.values())}
out["axis3_sib200_collapse"] = {
    "note": ("18 runs = 6 conditions x 3 seeds; 30 condition-language cells; 90 condition-language-seed "
             "cells. The request's '18 cells' figure corresponds to the 18 RUNS, not to cells -- the cell "
             "grid is 30 (or 90 counting seeds). Both counts are given below so the author can pick."),
    "totals": {"runs": 18, "condition_language_cells": 30, "condition_language_seed_cells": 90},
    "rule": "top-label fraction > 0.9 OR acc < 0.25",
    "rule_selfcheck_all_agree": all(sib[c][s][l]["rule_agrees"] for c in CONDS for s in SEEDS for l in LANGS),
    "per_condition": axis3,
}

# ---------------- axis 4 : Belebele macro retention ----------------
bele = {}
for c, fn in [("mixed_5lang", "phase1_belebele_gate_mixed_5lang_offonly_result.json"),
              ("english_only", "phase1_belebele_gate_english_only_offonly_result.json"),
              ("chinese_only", "phase1_belebele_gate_chinese_only_offonly_result.json"),
              ("korean_only", "phase1_belebele_gate_korean_only_result.json")]:
    p = R / fn
    if not p.exists():
        bele[c] = {"status": "MISSING", "path": str(p)}
        continue
    d = json.loads(p.read_text())
    langs_present = list(d["per_language"].keys())
    flagged = [l for l, v in d["per_language"].items() if v.get("flagged_at_chance")]
    bele[c] = {
        "status": "present", "path": str(p), "pp_ratio": d.get("pp_ratio"),
        "off_only": d.get("off_only", False),
        "langs_evaluated": langs_present,
        "langs_flagged_at_chance_excluded": flagged,
        "langs_in_macro": [l for l in langs_present if l not in flagged],
        "per_language_retention_off": {l: v["retention_off"] for l, v in d["per_language"].items()},
        "per_language_retention_on": {l: v["retention_on"] for l, v in d["per_language"].items()},
        "macro_off": d["macro_average_retention"]["off"],
        "macro_on": d["macro_average_retention"]["on"],
        "macro_off_3dp": round(d["macro_average_retention"]["off"], 3),
        "worst_off": d["worst_language_retention"]["off"],
        "worst_on": d["worst_language_retention"]["on"],
        "n_seeds": {l: v["n_seeds"] for l, v in d["per_language"].items()},
    }
for c in ["swahili_only", "bengali_only"]:
    bele[c] = {"status": "MISSING", "note": "no Belebele gate result for this condition; Belebele was never run for it"}

present = {c: v for c, v in bele.items() if v.get("status") == "present"}
best = max(present, key=lambda c: present[c]["macro_off"])
out["axis4_belebele_macro_retention"] = {
    "note": ("Belebele coverage is INCOMPLETE: only 4 of 6 conditions exist, and 3 of those 4 are OFF-only "
             "(no pruning arm) and evaluated on EN/KO/ZH only. SW/BN are excluded for two different reasons "
             "-- in korean_only they were run and flagged at chance by phase1_belebele_floor_check.py; in the "
             "three off-only runs they were never evaluated at all."),
    "exclusion_rule": ("macro = mean over non-flagged languages. korean_only: swh_Latn and ben_Beng flagged "
                       "at chance (retention 0.906 / 0.977 against a 25% chance floor) and dropped. "
                       "mixed_5lang / english_only / chinese_only: only eng_Latn, kor_Hang, zho_Hans present."),
    "comparability_warning": ("korean_only's macro is over 3 non-flagged of 5 evaluated languages; the other "
                              "three are over 3 of 3 evaluated. The macro values are on the same EN/KO/ZH "
                              "language set, so they ARE comparable -- but korean_only is the only row with a "
                              "pruning-ON column."),
    "per_condition": bele,
    "best_condition_by_macro_off": best,
    "balanced_is_best": best == "mixed_5lang",
    "balanced_macro_off": present.get("mixed_5lang", {}).get("macro_off"),
}

(OUT / "p0_3_table6_source.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

# ---------------- csv ----------------
with (OUT / "p0_3_table6_source.csv").open("w") as f:
    f.write("# P0-3 Table 6 source -- 4 candidate stability axes. Long format.\n")
    f.write("# provenance: phase1_41_headline_gate_result.json / sib200_extended/SIB_EXTENDED_REPORT.md / phase1_belebele_gate_*_result.json\n")
    f.write("# seeds 0,1,2. regret = dbpb%(cond,lang) - column min over the 6 conditions.\n")
    f.write("axis,condition,eval_lang,value_raw,value_rounded,extra\n")
    for c in CONDS:
        for l in LANGS:
            f.write("bpb_regret_pp,%s,%s,%.6f,%.2f,dbpb_mean=%.6f\n" % (c, SHORT[l], regret[c][l], regret[c][l], mean[c][l]))
        a = out["axis1_bpb_regret"]["per_condition"][c]
        f.write("bpb_regret_pp,%s,MAX,%.6f,%.2f,lang=%s\n" % (c, a["max_regret_pp"], a["max_regret_pp"], a["max_regret_lang"]))
        f.write("bpb_regret_pp,%s,MEAN,%.6f,%.2f,\n" % (c, a["mean_regret_pp"], a["mean_regret_pp"]))
    for c in CONDS:
        w = out["axis2_worst_language"][c]
        f.write("worst_language_dbpb_pct,%s,%s,%.6f,%.2f,seed_consistent=%s\n" % (
            c, w["worst_lang"], w["worst_dbpb_pct"], w["worst_dbpb_pct"], w["worst_lang_seed_consistent"]))
    for c in CONDS:
        a = axis3[c]
        for l in LANGS:
            v = a["per_lang"][SHORT[l]]
            f.write("sib200_collapse,%s,%s,%d,%s,class=%s;acc=%s;top=%s\n" % (
                c, SHORT[l], v["n_collapsed_seeds"], v["label"], v["cell_class"],
                "/".join("%.3f" % x for x in v["per_seed_acc"]),
                "/".join("%.2f" % x for x in v["per_seed_top_label_frac"])))
        f.write("sib200_collapse,%s,FULL_CELLS,%d,%d,\n" % (c, a["n_full_collapse_cells"], a["n_full_collapse_cells"]))
        f.write("sib200_collapse,%s,PARTIAL_CELLS,%d,%d,\n" % (c, a["n_partial_collapse_cells"], a["n_partial_collapse_cells"]))
    for c in CONDS:
        b = bele[c]
        if b.get("status") != "present":
            f.write("belebele_macro_retention,%s,MACRO_OFF,,MISSING,%s\n" % (c, b.get("note", "")))
            continue
        f.write("belebele_macro_retention,%s,MACRO_OFF,%.6f,%.3f,langs=%s;excluded=%s\n" % (
            c, b["macro_off"], b["macro_off"], "+".join(SHORT[l] for l in b["langs_in_macro"]),
            "+".join(SHORT[l] for l in b["langs_flagged_at_chance_excluded"]) or "none_evaluated_SW_BN"))
        f.write("belebele_macro_retention,%s,WORST_OFF,%.6f,%.3f,\n" % (c, b["worst_off"], b["worst_off"]))

print(json.dumps({
    "balanced_regret_row_matches_body": out["axis1_bpb_regret"]["balanced_row_all_match"],
    "balanced_regret": {SHORT[l]: round(regret["mixed_5lang"][l], 2) for l in LANGS},
    "max_regret_by_cond": {c: round(out["axis1_bpb_regret"]["per_condition"][c]["max_regret_pp"], 2) for c in CONDS},
    "worst_lang": {c: (out["axis2_worst_language"][c]["worst_lang"], round(out["axis2_worst_language"][c]["worst_dbpb_pct"], 2)) for c in CONDS},
    "sib_rule_selfcheck": out["axis3_sib200_collapse"]["rule_selfcheck_all_agree"],
    "sib_counts": {c: (axis3[c]["n_full_collapse_cells"], axis3[c]["n_partial_collapse_cells"]) for c in CONDS},
    "belebele_macro": {c: (bele[c].get("macro_off") and round(bele[c]["macro_off"], 3)) or bele[c].get("status") for c in CONDS},
    "balanced_is_best_belebele": out["axis4_belebele_macro_retention"]["balanced_is_best"],
}, indent=1, ensure_ascii=False))
