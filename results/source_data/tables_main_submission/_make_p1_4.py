#!/usr/bin/env python3
"""P1-4  stage2x2 EN/SW -- main effects + verdict. The only outstanding calculation.

PROVENANCE
  off-diagonal cells (run 2026-08-02, driver scripts/stage2x2_sw_0802_driver.sh,
                      log logs/stage2x2_sw_0802.log):
      results/deepseek/english_only/seed{s}/scale_swahili_only_seed{s}/eval_ppl.json
      results/deepseek/swahili_only/seed{s}/scale_english_only_seed{s}/eval_ppl.json
  diagonal cells (reused, NEVER recomputed -- driver STEP note):
      results/deepseek/english_only/seed{s}/scale_english_only_seed{s}/eval_ppl.json
      results/deepseek/swahili_only/seed{s}/scale_swahili_only_seed{s}/eval_ppl.json
  baseline:  results/deepseek/baseline/eval_ppl.json
  reproducibility artefact:
      english_only/seed0/scale_swahili_only_seed0/eval_ppl_prior_20260730.json  (2026-07-30 file, preserved)
  seeds: 0, 1, 2
  DEFINITIONS AND VERDICT RULE: taken verbatim from the mini-preregistration fixed
  2026-08-02T07:29:39Z (before the first merge) and recorded in logs/stage2x2_sw_0802.log.
  Nothing here is a new definition.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
R = Path("results/deepseek")
SEEDS = [0, 1, 2]
LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]

PREREG = """PRIMARY METRICS (evaluated on the swh_Latn column):
  scale-language effect  = mean over fisher in {EN,SW} of
                           [ dbpb%(scale=EN) - dbpb%(scale=SW) ]
  fisher-language effect = mean over scale  in {EN,SW} of
                           [ dbpb%(fisher=EN) - dbpb%(fisher=SW) ]
  Both are main effects of the 2x2; positive = the mismatched (EN)
  calibration language is worse for Swahili.
  PREDICTION (same direction as the original EN/KO 2x2): scale effect >> fisher effect.

DECISION RULE:
  NO NOISE FLOOR IS AVAILABLE. The original 2x2's floor came from placebo
  Fisher artefacts, which have since been deleted, so an equivalent floor
  cannot be recomputed for this pair. This is stated as a limitation rather
  than substituted for. In its place the report gives:
    (a) sign consistency of both effects across the 3 seeds,
    (b) the ratio scale-effect / fisher-effect,
    (c) each effect against the seed spread of the cells that produced it.
  VERDICT VOCABULARY:
    SCALE_DOMINANT  = scale effect > 3x fisher effect AND both effects
                      sign-consistent across all 3 seeds
    MIXED           = otherwise
    FISHER_DOMINANT = fisher effect exceeds scale effect
  The headline Swahili floor of 0.91%p may be quoted for orientation ONLY,
  and MUST carry the label 'noise floor from a DIFFERENT level of analysis
  (headline single-condition comparison), not a floor for this 2x2'.

SECONDARY: the identical computation on the eng_Latn column. Descriptive only."""

base = {l: json.loads((R / "baseline/eval_ppl.json").read_text())[l]["bits_per_byte"] for l in LANGS}


def cell(fisher, scale, seed):
    p = R / fisher / f"seed{seed}" / f"scale_{scale}_seed{seed}" / "eval_ppl.json"
    d = json.loads(p.read_text())
    return {l: 100.0 * (d[l]["bits_per_byte"] / base[l] - 1) for l in LANGS}, str(p), d


ARMS = [("english_only", "english_only"), ("english_only", "swahili_only"),
        ("swahili_only", "english_only"), ("swahili_only", "swahili_only")]
grid, paths, bpbraw = {}, {}, {}
for f_, s_ in ARMS:
    for s in SEEDS:
        g, p, raw = cell(f_, s_, s)
        grid[(f_, s_, s)] = g
        paths[f"fisher={f_}|scale={s_}|seed{s}"] = p
        bpbraw[f"fisher={f_}|scale={s_}|seed{s}"] = {l: raw[l]["bits_per_byte"] for l in LANGS}


def effects(lang):
    per_seed = {"scale": [], "fisher": []}
    detail = []
    for s in SEEDS:
        e_e = grid[("english_only", "english_only", s)][lang]
        e_s = grid[("english_only", "swahili_only", s)][lang]
        s_e = grid[("swahili_only", "english_only", s)][lang]
        s_s = grid[("swahili_only", "swahili_only", s)][lang]
        scale = ((e_e - e_s) + (s_e - s_s)) / 2
        fisher = ((e_e - s_e) + (e_s - s_s)) / 2
        per_seed["scale"].append(scale)
        per_seed["fisher"].append(fisher)
        detail.append({"seed": s,
                       "cells_dbpb_pct": {"fisher=EN,scale=EN": e_e, "fisher=EN,scale=SW": e_s,
                                          "fisher=SW,scale=EN": s_e, "fisher=SW,scale=SW": s_s},
                       "scale_effect_pp": scale, "fisher_effect_pp": fisher})
    sc, fi = per_seed["scale"], per_seed["fisher"]
    return {
        "per_seed": detail,
        "scale_effect_per_seed_pp": sc, "fisher_effect_per_seed_pp": fi,
        "scale_effect_mean_pp": sum(sc) / 3, "fisher_effect_mean_pp": sum(fi) / 3,
        "scale_sign_consistent": len({x > 0 for x in sc}) == 1,
        "fisher_sign_consistent": len({x > 0 for x in fi}) == 1,
        "scale_seed_spread_pp": max(sc) - min(sc), "fisher_seed_spread_pp": max(fi) - min(fi),
        "ratio_scale_over_fisher": (sum(sc) / 3) / (sum(fi) / 3) if sum(fi) else None,
    }


sw = effects("swh_Latn")
en = effects("eng_Latn")


def verdict(e):
    sc, fi = e["scale_effect_mean_pp"], e["fisher_effect_mean_pp"]
    if fi > sc:
        return "FISHER_DOMINANT", "fisher effect exceeds scale effect"
    if sc > 3 * fi and e["scale_sign_consistent"] and e["fisher_sign_consistent"]:
        return "SCALE_DOMINANT", ("scale effect (%.6f) > 3x fisher effect (%.6f); both effects "
                                  "sign-consistent across all 3 seeds" % (sc, 3 * fi))
    return "MIXED", "neither the FISHER_DOMINANT nor the SCALE_DOMINANT condition is met"


v_sw, why_sw = verdict(sw)
v_en, why_en = verdict(en)

# ---- reproducibility: 2026-07-30 prior file vs the 2026-08-02 re-run ----
prior_p = R / "english_only/seed0/scale_swahili_only_seed0/eval_ppl_prior_20260730.json"
cur_p = R / "english_only/seed0/scale_swahili_only_seed0/eval_ppl.json"
pj, cj = json.loads(prior_p.read_text()), json.loads(cur_p.read_text())
repro = {"prior_file": str(prior_p), "prior_mtime_note": "file dated 2026-07-30 04:53, copied aside 2026-08-02 06:02 before the re-run",
         "rerun_file": str(cur_p), "per_lang": {}}
for l in LANGS:
    a, b = pj[l]["bits_per_byte"], cj[l]["bits_per_byte"]
    repro["per_lang"][l] = {"prior_bpb": a, "rerun_bpb": b, "abs_diff": abs(a - b),
                            "identical_6dp": round(a, 6) == round(b, 6), "bitwise_identical": a == b}
repro["all_identical_6dp"] = all(v["identical_6dp"] for v in repro["per_lang"].values())
repro["all_bitwise_identical"] = all(v["bitwise_identical"] for v in repro["per_lang"].values())
repro["max_abs_diff"] = max(v["abs_diff"] for v in repro["per_lang"].values())

FLOOR_LABEL = ("noise floor from a DIFFERENT level of analysis (headline single-condition "
               "comparison), not a floor for this 2x2")
out = {
    "provenance": {
        "generated_utc": "2026-08-02",
        "driver": "scripts/stage2x2_sw_0802_driver.sh",
        "log": "logs/stage2x2_sw_0802.log",
        "cells_run_utc": "2026-08-02T05:5x -- 09:01:14Z (6 off-diagonal cells, 0 skipped)",
        "seeds": SEEDS,
        "baseline_file": str(R / "baseline/eval_ppl.json"),
        "cell_paths": paths,
        "note": "POST-HOC EXTENSION of the 2026-07-26 EN/KO 2x2; not covered by that registration.",
    },
    "prereg_quoted_verbatim": PREREG,
    "baseline_bpb": base,
    "cell_bpb_raw": bpbraw,
    "primary_swh_Latn": {**sw, "verdict": v_sw, "verdict_basis": why_sw},
    "secondary_eng_Latn": {
        **en, "verdict_if_rule_applied_mechanically": v_en, "verdict_basis": why_en,
        "status": "DESCRIPTIVE ONLY -- the prereg says the eng_Latn column carries no verdict",
        "do_not_report_as_verdict": True,
        "caveat": ("Both effects are NEGATIVE on the English column (scale %.6f, fisher %.6f), which is the "
                   "expected direction: English calibration helps English. The verdict vocabulary was written "
                   "for the swh_Latn column where 'positive = the mismatched (EN) calibration is worse'. "
                   "Applying 'FISHER_DOMINANT = fisher effect exceeds scale effect' to two negative numbers "
                   "fires on -1.72 > -3.83, which inverts the intended meaning -- the scale path is in fact "
                   "the LARGER effect on English too (|-3.83| > |-1.72|, ratio 2.23x on absolute values). "
                   "Do not quote FISHER_DOMINANT for the English column."
                   % (en["scale_effect_mean_pp"], en["fisher_effect_mean_pp"])),
        "abs_ratio_scale_over_fisher": abs(en["scale_effect_mean_pp"]) / abs(en["fisher_effect_mean_pp"]),
    },
    "noise_floor": {
        "available": False,
        "prereg_statement": ("NO NOISE FLOOR IS AVAILABLE. The original 2x2's floor came from placebo "
                             "Fisher artefacts, which have since been deleted, so an equivalent floor "
                             "cannot be recomputed for this pair."),
        "orientation_only_value_pp": 0.91,
        "mandatory_label": FLOOR_LABEL,
        "ratio_of_scale_effect_to_that_value": sw["scale_effect_mean_pp"] / 0.91,
        "ratio_of_fisher_effect_to_that_value": sw["fisher_effect_mean_pp"] / 0.91,
        "warning": ("These ratios are NOT the requested 'multiple of the placebo noise floor'. No placebo "
                    "floor exists for the EN/SW pair. They divide by the headline Swahili floor, which the "
                    "prereg permits only for orientation and only with the label above."),
        "substitute_evidence_per_prereg": {
            "a_sign_consistency": {"scale": sw["scale_sign_consistent"], "fisher": sw["fisher_sign_consistent"]},
            "b_ratio_scale_over_fisher": sw["ratio_scale_over_fisher"],
            "c_effect_vs_seed_spread": {
                "scale_effect_mean": sw["scale_effect_mean_pp"], "scale_seed_spread": sw["scale_seed_spread_pp"],
                "scale_effect_over_spread": sw["scale_effect_mean_pp"] / sw["scale_seed_spread_pp"],
                "fisher_effect_mean": sw["fisher_effect_mean_pp"], "fisher_seed_spread": sw["fisher_seed_spread_pp"],
                "fisher_effect_over_spread": sw["fisher_effect_mean_pp"] / sw["fisher_seed_spread_pp"],
            },
        },
    },
    "reproducibility_20260730": repro,
}
(OUT / "p1_4_stage2x2_ensw.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

print(json.dumps({
    "SW_scale_per_seed": sw["scale_effect_per_seed_pp"], "SW_scale_mean": sw["scale_effect_mean_pp"],
    "SW_fisher_per_seed": sw["fisher_effect_per_seed_pp"], "SW_fisher_mean": sw["fisher_effect_mean_pp"],
    "SW_ratio": sw["ratio_scale_over_fisher"], "SW_signs": [sw["scale_sign_consistent"], sw["fisher_sign_consistent"]],
    "SW_VERDICT": v_sw, "SW_why": why_sw,
    "EN_scale_mean": en["scale_effect_mean_pp"], "EN_fisher_mean": en["fisher_effect_mean_pp"],
    "EN_ratio": en["ratio_scale_over_fisher"], "EN_VERDICT": v_en,
    "repro_all_identical_6dp": repro["all_identical_6dp"],
    "repro_all_bitwise": repro["all_bitwise_identical"], "repro_max_abs_diff": repro["max_abs_diff"],
}, indent=1))
