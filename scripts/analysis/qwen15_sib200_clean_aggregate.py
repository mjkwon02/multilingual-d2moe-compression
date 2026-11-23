"""Aggregate the Qwen1.5-MoE-A2.7B SIB-200 test-item-clean results (98 items per
language) and, on the SAME 98 items, the DeepSeek clean results for comparison.

Completeness gate (fails loudly, never fills in): baseline + 18 models, 5
languages each, full n=204 / clean n=98, 1020 prediction rows per model, no
non-finite accuracy, and within each language an identical index_id set across
all Qwen models AND the DeepSeek clean models.

Table: absolute clean accuracy (%), baseline as a single value, each condition
as mean +- sample std (ddof=1) over seeds 0,1,2 with every seed included
(collapsed seeds are NOT dropped), plus [k/3] = number of seeds meeting the
collapse criterion  top_label_share > 0.9  OR  acc < 0.25  on the clean subset.

Outputs under <results>/sib200_clean/report/: table_qwen_clean98.{md,csv,json},
table_deepseek_clean98.{md,csv} and completeness.json.
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import qwen15_config as QC  # noqa: E402

QWEN_DIR = QC.RESULTS_ROOT.parent / "sib200_clean"
DS_DIR = SCRIPT_DIR.parents[1] / "results" / "downstream" / "exposure_filtered" / "deepseek_clean_eval" / "sib200"
REPORT = QWEN_DIR / "report"
CONDS = ["english_only", "chinese_only", "korean_only", "swahili_only", "bengali_only", "mixed_5lang"]
LABEL = {"english_only": "English", "chinese_only": "Chinese", "korean_only": "Korean",
         "swahili_only": "Swahili", "bengali_only": "Bengali", "mixed_5lang": "Balanced-character"}
LANGS = [("eng_Latn", "EN"), ("zho_Hans", "ZH"), ("kor_Hang", "KO"), ("swh_Latn", "SW"), ("ben_Beng", "BN")]
SEEDS = (0, 1, 2)
COLLAPSE_TOP, COLLAPSE_ACC = 0.9, 0.25


def load(d, key, problems):
    f, pf = d / f"{key}.json", d / f"{key}_predictions.jsonl"
    if not f.exists():
        problems.append(f"{d.name}/{key}: missing result json"); return None, None
    res = json.loads(f.read_text())["results"]
    for l, _ in LANGS:
        if l not in res:
            problems.append(f"{d.name}/{key}: {l} absent"); continue
        if res[l]["full"]["n"] != 204 or res[l]["clean"]["n"] != 98:
            problems.append(f"{d.name}/{key}/{l}: n full={res[l]['full']['n']} clean={res[l]['clean']['n']}")
        for nm in ("full", "clean"):
            a = res[l][nm]["acc"]
            if a is None or not math.isfinite(a):
                problems.append(f"{d.name}/{key}/{l}: {nm} acc non-finite")
    if not pf.exists():
        problems.append(f"{d.name}/{key}: missing predictions"); return res, None
    rows = [json.loads(x) for x in open(pf)]
    if len(rows) != 5 * 204:
        problems.append(f"{d.name}/{key}: {len(rows)} prediction rows != 1020")
    sets = {}
    for r in rows:
        if r["in_clean"]:
            sets.setdefault(r["lang"], set()).add(r["index_id"])
    return res, sets


def collapsed(cl):
    return cl["top_label_share"] > COLLAPSE_TOP or cl["acc"] < COLLAPSE_ACC


def build_table(d, problems, tag):
    base, base_sets = load(d, "baseline", problems)
    cells = {}
    clean_sets = {}
    if base_sets:
        for l, s in base_sets.items():
            clean_sets.setdefault(l, {})[f"{tag}:baseline"] = frozenset(s)
    for c in CONDS:
        for s in SEEDS:
            res, sets = load(d, f"{c}_seed{s}", problems)
            cells[(c, s)] = res
            if sets:
                for l, ss in sets.items():
                    clean_sets.setdefault(l, {})[f"{tag}:{c}_seed{s}"] = frozenset(ss)
    table = {"baseline": {}, "conditions": {}, "per_seed": []}
    for l, L in LANGS:
        if base and l in base:
            b = base[l]["clean"]
            table["baseline"][L] = {"acc_pct": 100 * b["acc"], "top_label": b["top_label"],
                                    "top_label_share": b["top_label_share"], "collapsed": collapsed(b),
                                    "full204_acc_pct": 100 * base[l]["full"]["acc"]}
    for c in CONDS:
        table["conditions"][c] = {}
        for l, L in LANGS:
            vals, k = [], 0
            for s in SEEDS:
                res = cells[(c, s)]
                if not res or l not in res:
                    continue
                cl = res[l]["clean"]
                vals.append(100 * cl["acc"]); k += int(collapsed(cl))
                table["per_seed"].append({"condition": c, "seed": s, "lang": L, "clean_acc_pct": 100 * cl["acc"],
                                          "full204_acc_pct": 100 * res[l]["full"]["acc"],
                                          "top_label": cl["top_label"], "top_label_share": cl["top_label_share"],
                                          "collapsed": collapsed(cl)})
            table["conditions"][c][L] = {
                "n_seeds": len(vals), "mean_acc_pct": (st.mean(vals) if vals else None),
                "std_ddof1": (st.stdev(vals) if len(vals) > 1 else None), "seeds": vals, "n_collapsed": k}
    return table, clean_sets


def fmt_md(table, title, note):
    lines = [f"# {title}", "", "| Calibration | " + " | ".join(L for _, L in LANGS) + " |",
             "|:---|" + ":---:|" * len(LANGS)]
    b = table["baseline"]
    lines.append("| Uncompressed baseline | " + " | ".join(
        f"{b[L]['acc_pct']:.1f}" + (" (collapse)" if b[L]["collapsed"] else "") if L in b else "—" for _, L in LANGS) + " |")
    for c in CONDS:
        row = []
        for _, L in LANGS:
            v = table["conditions"][c][L]
            if v["mean_acc_pct"] is None:
                row.append("—"); continue
            s = f"{v['mean_acc_pct']:.1f}" + (f" ± {v['std_ddof1']:.1f}" if v["std_ddof1"] is not None else "")
            s += f" [{v['n_collapsed']}/{v['n_seeds']}]"
            row.append(s)
        lines.append(f"| {LABEL[c]} | " + " | ".join(row) + " |")
    lines += ["", note]
    return "\n".join(lines) + "\n"


def fmt_csv(table):
    out = ["condition,lang,n_seeds,seed0,seed1,seed2,mean_acc_pct,std_ddof1,n_collapsed"]
    b = table["baseline"]
    for _, L in LANGS:
        if L in b:
            out.append(f"baseline,{L},1,{b[L]['acc_pct']:.4f},,,{b[L]['acc_pct']:.4f},,{int(b[L]['collapsed'])}")
    for c in CONDS:
        for _, L in LANGS:
            v = table["conditions"][c][L]
            seeds = [f"{x:.4f}" for x in v["seeds"]] + [""] * (3 - len(v["seeds"]))
            m = "" if v["mean_acc_pct"] is None else f"{v['mean_acc_pct']:.4f}"
            sd = "" if v["std_ddof1"] is None else f"{v['std_ddof1']:.4f}"
            out.append(f"{LABEL[c]},{L},{v['n_seeds']},{','.join(seeds)},{m},{sd},{v['n_collapsed']}")
    return "\n".join(out) + "\n"


def main():
    problems = []
    qwen, q_sets = build_table(QWEN_DIR, problems, "qwen")
    ds_problems = []
    ds, d_sets = build_table(DS_DIR, ds_problems, "deepseek")
    # within-language item-set identity: Qwen models among themselves, and vs DeepSeek clean
    item_check = {}
    for l, L in LANGS:
        qs = q_sets.get(l, {}); dss = d_sets.get(l, {})
        allsets = {**qs, **dss}
        distinct = {v for v in allsets.values()}
        item_check[L] = {"n_qwen_models": len(qs), "n_deepseek_models": len(dss),
                         "identical_within_qwen": len(set(qs.values())) <= 1,
                         "identical_qwen_vs_deepseek": len(distinct) <= 1,
                         "n_items": (len(next(iter(distinct))) if distinct else 0)}
        if len(distinct) > 1:
            problems.append(f"{L}: clean item sets differ across models (Qwen and/or DeepSeek)")
    expected_models = 1 + len(CONDS) * len(SEEDS)
    n_qwen = sum(1 for k in ["baseline"] + [f"{c}_seed{s}" for c in CONDS for s in SEEDS] if (QWEN_DIR / f"{k}.json").exists())
    completeness = {"qwen_models_present": n_qwen, "qwen_models_expected": expected_models,
                    "qwen_problems": problems, "deepseek_problems": ds_problems,
                    "within_language_item_sets": item_check,
                    "complete": n_qwen == expected_models and not problems}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "completeness.json").write_text(json.dumps(completeness, indent=2))

    note = ("Absolute accuracy (%) on the 98 test-item-clean SIB-200 items per language (same keep set as the DeepSeek "
            "clean run); 14 fixed English demonstrations unchanged (6 of 14 calibration-exposed per the audit), so "
            "this is TEST-ITEM clean, not calibration-exposure-free. Conditions: mean ± sample std (ddof=1) over "
            "seeds 0,1,2, all seeds included; [k/3] = seeds meeting the collapse criterion "
            f"(top-label share > {COLLAPSE_TOP} OR accuracy < {int(COLLAPSE_ACC*100)}%) on the clean subset. "
            "Majority-class floor of the clean subset ~26.5% (26/98 science/technology).")
    (REPORT / "table_qwen_clean98.md").write_text(fmt_md(
        qwen, "Qwen1.5-MoE-A2.7B — SIB-200 test-item-clean accuracy (%), 14-shot, 98 items/language", note))
    (REPORT / "table_qwen_clean98.csv").write_text(fmt_csv(qwen))
    (REPORT / "table_qwen_clean98.json").write_text(json.dumps(
        {"model": QC.MODEL_ID, "revision": QC.MODEL_REVISION, "collapse_criterion":
         {"top_label_share_gt": COLLAPSE_TOP, "acc_lt": COLLAPSE_ACC, "subset": "clean98"},
         "table": qwen, "completeness": completeness}, indent=2))
    (REPORT / "table_deepseek_clean98.md").write_text(fmt_md(
        ds, "DeepSeek-MoE-16B — SIB-200 test-item-clean accuracy (%), same 98 items (from analysis/clean_eval_0916)",
        note.replace("Qwen", "DeepSeek")))
    (REPORT / "table_deepseek_clean98.csv").write_text(fmt_csv(ds))

    print((REPORT / "table_qwen_clean98.md").read_text())
    print((REPORT / "table_deepseek_clean98.md").read_text())
    print(f"[aggregate] Qwen models present {n_qwen}/{expected_models}; problems: {problems or 'none'}; "
          f"DeepSeek problems: {ds_problems or 'none'}")
    for L, v in item_check.items():
        print(f"[aggregate] {L}: items={v['n_items']} qwen_models={v['n_qwen_models']} ds_models={v['n_deepseek_models']} "
              f"identical_within_qwen={v['identical_within_qwen']} identical_vs_deepseek={v['identical_qwen_vs_deepseek']}")
    print(f"[aggregate] COMPLETE = {completeness['complete']}")
    sys.exit(0 if completeness["complete"] else 1)


if __name__ == "__main__":
    main()
