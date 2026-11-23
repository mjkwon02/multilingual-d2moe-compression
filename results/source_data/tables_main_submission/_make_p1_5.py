#!/usr/bin/env python3
"""P1-5  ZH/BN placebo -- Table 2 noise floors + Appendix D2 extension to all 5 languages.

PROVENANCE
  own (matched) arms : results/deepseek/{cond}/seed{s}/scale_{cond}_seed{s}/eval_ppl.json
  placebo arms       : results/deepseek/{cond}_b/seed{s}/scale_{cond}_b_seed{s}/eval_ppl.json
  baseline           : results/deepseek/baseline/eval_ppl.json
  official verdicts  : results/deepseek/phase1_41_headline_gate_result.json ("own_gain")
  generated          : 2026-08-02, by tables_main_submission/_make_p1_5.py
  seeds              : 0, 1, 2
  DEFINITIONS -- verbatim from scripts/phase1_41_headline_gate.py, unchanged:
      dbpb%(cond,seed,lang) = 100*(bpb(cond,seed,lang)/bpb_baseline(lang) - 1)
      noise_floor(L)        = max over seed of |dbpb%(matched(L),seed,L) - dbpb%(placebo(L),seed,L)|
      verdict SUPPORTED     = sign_consistent AND mean(own_gain) > 2 * noise_floor
  NOTE: results/placebo_zh_bn_summary.md is about Fisher-rank rho (Appendix F), NOT bpb.
  It contains no dbpb% numbers, so the bpb placebo data below comes from the raw eval files that
  phase1_41_headline_gate.py itself reads.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
R = Path("results/deepseek")
SEEDS = [0, 1, 2]
LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
SHORT = {"eng_Latn": "EN", "kor_Hang": "KO", "zho_Hans": "ZH", "swh_Latn": "SW", "ben_Beng": "BN"}
MATCHED = {"eng_Latn": "english_only", "kor_Hang": "korean_only", "zho_Hans": "chinese_only",
           "swh_Latn": "swahili_only", "ben_Beng": "bengali_only"}
PLACEBO = {l: MATCHED[l] + "_placebo" for l in LANGS}

base = {l: json.loads((R / "baseline/eval_ppl.json").read_text())[l]["bits_per_byte"] for l in LANGS}


def diag(cond, seed):
    return R / cond / f"seed{seed}" / f"scale_{cond}_seed{seed}" / "eval_ppl.json"


def dbpb(cond, seed, lang):
    return 100.0 * (json.loads(diag(cond, seed).read_text())[lang]["bits_per_byte"] / base[lang] - 1)


official = json.loads((R / "phase1_41_headline_gate_result.json").read_text())["own_gain"]
BODY_FLOOR = {"eng_Latn": 4.05, "kor_Hang": 0.79, "zho_Hans": 4.24, "swh_Latn": 0.91, "ben_Beng": 0.34}
BODY_MULT = {"eng_Latn": 1.47, "kor_Hang": 76.98, "zho_Hans": 2.54, "swh_Latn": 39.20, "ben_Beng": 201.57}

res = {}
for l in LANGS:
    own = [dbpb(MATCHED[l], s, l) for s in SEEDS]
    pla = [dbpb(PLACEBO[l], s, l) for s in SEEDS]
    diffs = [abs(a - b) for a, b in zip(own, pla)]
    signed = [b - a for a, b in zip(own, pla)]  # placebo - own; positive = placebo is WORSE
    floor = max(diffs)
    off = official[l]
    mult = off["mean_gain"] / floor
    res[l] = {
        "matched_condition": MATCHED[l], "placebo_condition": PLACEBO[l],
        "own_files": [str(diag(MATCHED[l], s)) for s in SEEDS],
        "placebo_files": [str(diag(PLACEBO[l], s)) for s in SEEDS],
        "own_dbpb_pct_per_seed": own, "placebo_dbpb_pct_per_seed": pla,
        "abs_diff_per_seed": diffs, "signed_diff_placebo_minus_own_per_seed": signed,
        "noise_floor_pp": floor, "noise_floor_driving_seed": diffs.index(floor),
        "official_noise_floor_pp": off["noise_floor"],
        "floor_matches_official": abs(floor - off["noise_floor"]) < 1e-9,
        "body_floor_2dp": BODY_FLOOR[l], "floor_2dp": round(floor, 2),
        "floor_matches_body": round(floor, 2) == BODY_FLOOR[l],
        "own_gain_per_seed": off["per_seed_gain"], "own_gain_mean": off["mean_gain"],
        "sign_consistent": off["sign_consistent"],
        "threshold_2x_floor": 2 * floor,
        "multiple_of_floor": mult, "multiple_of_floor_2dp": round(mult, 2),
        "body_multiple": BODY_MULT[l], "multiple_matches_body": round(mult, 2) == BODY_MULT[l],
        "verdict_official": off["verdict"],
        "verdict_recomputed": ("SUPPORTED" if (off["sign_consistent"] and off["mean_gain"] > 2 * floor)
                               else ("INCONCLUSIVE" if off["sign_consistent"] else "NOT_SUPPORTED")),
        # conservativeness: prereg reading is that the placebo arm should be no BETTER than the
        # matched arm on its own language (placebo dbpb% >= own dbpb%), i.e. the matched arm is
        # not being flattered by the comparison.
        "conservative_per_seed": [b >= a for a, b in zip(own, pla)],
        "conservative_all_seeds": all(b >= a for a, b in zip(own, pla)),
        "conservative_on_mean": (sum(pla) / 3) >= (sum(own) / 3),
        "own_mean_dbpb_pct": sum(own) / 3, "placebo_mean_dbpb_pct": sum(pla) / 3,
    }
    res[l]["verdict_recomputed_matches_official"] = res[l]["verdict_recomputed"] == off["verdict"]

out = {
    "provenance": {
        "generated_utc": "2026-08-02", "seeds": SEEDS, "results_root": str(R),
        "baseline_file": str(R / "baseline/eval_ppl.json"),
        "official_gate": str(R / "phase1_41_headline_gate_result.json"),
        "definitions_source": "scripts/phase1_41_headline_gate.py (unchanged)",
        "source_note": ("results/placebo_zh_bn_summary.md and placebo_switch1_zh_bn.json contain "
                        "Fisher-rank Spearman rho for Appendix F, NOT dbpb%. They hold no bpb numbers, so "
                        "the ZH/BN bpb placebo values here were read from the same raw eval_ppl.json files "
                        "that phase1_41_headline_gate.py reads."),
    },
    "baseline_bpb": base,
    "per_language": res,
    "focus_zh_bn": {l: {"noise_floor_pp": res[l]["noise_floor_pp"],
                        "body_value": BODY_FLOOR[l],
                        "matches_body": res[l]["floor_matches_body"],
                        "multiple_of_floor": res[l]["multiple_of_floor"],
                        "body_multiple": BODY_MULT[l],
                        "multiple_matches_body": res[l]["multiple_matches_body"],
                        "conservative_all_seeds": res[l]["conservative_all_seeds"]}
                    for l in ["zho_Hans", "ben_Beng"]},
}
(OUT / "p1_5_placebo_zh_bn.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

HDR = ("lang,matched_condition,placebo_condition,seed,own_dbpb_pct,placebo_dbpb_pct,"
       "abs_diff_pp,signed_diff_placebo_minus_own_pp,placebo_not_better_than_own\n")


def write_csv(path, langs, title):
    with path.open("w") as f:
        f.write("# %s\n" % title)
        f.write("# provenance: {cond}/seed{s}/scale_{cond}_seed{s}/eval_ppl.json and {cond}_b/... under %s\n" % R)
        f.write("#             + baseline/eval_ppl.json; verdicts from phase1_41_headline_gate_result.json\n")
        f.write("# dbpb%% = 100*(bpb/bpb_baseline - 1); noise_floor = max over seed |own - placebo| (phase1_41_headline_gate.py)\n")
        f.write("# seeds 0,1,2. Values at 6 dp.\n")
        f.write(HDR)
        for l in langs:
            d = res[l]
            for i, s in enumerate(SEEDS):
                f.write("%s,%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%s\n" % (
                    SHORT[l], d["matched_condition"], d["placebo_condition"], s,
                    d["own_dbpb_pct_per_seed"][i], d["placebo_dbpb_pct_per_seed"][i],
                    d["abs_diff_per_seed"][i], d["signed_diff_placebo_minus_own_per_seed"][i],
                    d["conservative_per_seed"][i]))
            f.write("%s,%s,%s,SUMMARY,%.6f,%.6f,%.6f,,noise_floor=%.6f;own_gain_mean=%.6f;multiple=%.2f;verdict=%s\n" % (
                SHORT[l], d["matched_condition"], d["placebo_condition"],
                d["own_mean_dbpb_pct"], d["placebo_mean_dbpb_pct"], d["noise_floor_pp"],
                d["noise_floor_pp"], d["own_gain_mean"], d["multiple_of_floor"], d["verdict_official"]))


write_csv(OUT / "p1_5_placebo_zh_bn.csv", ["zho_Hans", "ben_Beng"],
          "P1-5 ZH/BN placebo -- own vs placebo self-language dbpb%, per seed")
write_csv(OUT / "tableD2_full.csv", LANGS,
          "Table D2 (full) -- placebo raw data, all 5 languages, same format as the EN/KO/SW version")

print(json.dumps({l: {"floor": round(res[l]["noise_floor_pp"], 4), "body": BODY_FLOOR[l],
                      "floor_ok": res[l]["floor_matches_body"],
                      "mult": round(res[l]["multiple_of_floor"], 2), "body_mult": BODY_MULT[l],
                      "mult_ok": res[l]["multiple_matches_body"],
                      "conservative_all_seeds": res[l]["conservative_all_seeds"],
                      "conservative_on_mean": res[l]["conservative_on_mean"],
                      "verdict": res[l]["verdict_official"],
                      "verdict_recompute_ok": res[l]["verdict_recomputed_matches_official"]}
                  for l in LANGS}, indent=1))
