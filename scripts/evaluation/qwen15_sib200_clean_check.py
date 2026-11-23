"""Content check for ONE Qwen SIB-200 clean result (used as the gate after the
baseline and again after every compressed model). Reads only saved files.

Checks: 5 languages present; full n=204, clean n=98; 1020 prediction rows,
204 per language, index_ids unique per language; every logprob finite and
strictly <= 0; argmax of the stored logprobs equals the stored pred; stored
correct equals (pred == gold); the clean index_id set equals the DeepSeek
clean set for the same language; label distribution reported.

Usage: python qwen15_sib200_clean_check.py baseline [english_only_seed0 ...]
"""
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import qwen15_config as QC  # noqa: E402
from phase1_sib200_eval import CATEGORY_ORDER  # noqa: E402

QWEN_DIR = QC.RESULTS_ROOT.parent / "sib200_clean"
DS_PRED = SCRIPT_DIR.parent / "analysis" / "clean_eval_0916" / "sib200" / "baseline_predictions.jsonl"
LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]


def ds_clean_sets():
    out = {}
    for line in open(DS_PRED):
        r = json.loads(line)
        if r["in_clean"]:
            out.setdefault(r["lang"], set()).add(r["index_id"])
    return out


def check(key, ds_sets):
    problems = []
    f, pf = QWEN_DIR / f"{key}.json", QWEN_DIR / f"{key}_predictions.jsonl"
    if not f.exists() or not pf.exists():
        return [f"missing {f.name if not f.exists() else pf.name}"], {}
    d = json.loads(f.read_text())
    res = d["results"]
    rows = [json.loads(x) for x in open(pf)]
    if len(rows) != 1020:
        problems.append(f"{len(rows)} rows != 1020")
    summary = {}
    for l in LANGS:
        if l not in res:
            problems.append(f"{l} absent"); continue
        rl = [r for r in rows if r["lang"] == l]
        ids = [r["index_id"] for r in rl]
        if len(rl) != 204 or len(set(ids)) != 204:
            problems.append(f"{l}: {len(rl)} rows, {len(set(ids))} unique ids")
        if res[l]["full"]["n"] != 204 or res[l]["clean"]["n"] != 98:
            problems.append(f"{l}: n full={res[l]['full']['n']} clean={res[l]['clean']['n']}")
        clean_ids = {r["index_id"] for r in rl if r["in_clean"]}
        if clean_ids != ds_sets.get(l, set()):
            problems.append(f"{l}: clean id set differs from DeepSeek clean set "
                            f"({len(clean_ids)} vs {len(ds_sets.get(l, set()))})")
        n_bad_lp = n_argmax = n_corr = 0
        for r in rl:
            lp = r["logprobs"]
            if len(lp) != 7 or any((not math.isfinite(x)) or x > 0 for x in lp):
                n_bad_lp += 1
            am = CATEGORY_ORDER[max(range(7), key=lambda k: lp[k])]
            if am != r["pred"]:
                n_argmax += 1
            if r["correct"] != int(r["pred"] == r["gold"]):
                n_corr += 1
        if n_bad_lp or n_argmax or n_corr:
            problems.append(f"{l}: bad_logprob_rows={n_bad_lp} argmax_mismatch={n_argmax} correct_mismatch={n_corr}")
        recomputed_clean = sum(r["correct"] for r in rl if r["in_clean"]) / 98
        if abs(recomputed_clean - res[l]["clean"]["acc"]) > 1e-9:
            problems.append(f"{l}: clean acc {res[l]['clean']['acc']} != recomputed {recomputed_clean}")
        cl = res[l]["clean"]
        summary[l] = {"full_acc": round(res[l]["full"]["acc"], 4), "clean_acc": round(cl["acc"], 4),
                      "top_label": cl["top_label"], "top_share": round(cl["top_label_share"], 3),
                      "n_distinct_pred": cl["n_distinct_pred"], "pred_dist": cl["pred_distribution"],
                      "prompt_tokens_max": res[l]["prompt_tokens"]["max"]}
    tok = d.get("qwen_input_processing", {})
    summary["_input"] = {k: tok.get(k) for k in ("adds_bos_by_default", "prompt_prefix_token", "pad_token",
                                                    "choice_token_ids", "answer_colon_then_candidate_is_two_tokens")}
    return problems, summary


def main():
    ds_sets = ds_clean_sets()
    rc = 0
    for key in sys.argv[1:]:
        problems, summary = check(key, ds_sets)
        print(f"== {key}: {'OK' if not problems else 'PROBLEMS'}")
        for p in problems:
            print("   !!", p); rc = 1
        for l, v in summary.items():
            print(f"   {l}: {json.dumps(v, ensure_ascii=False)}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
