"""Completion gate for the clean re-evaluation campaign.

Success is decided from the CONTENT of the saved result files, never from a
process having exited. A cell counts as COMPLETE only if:
  * the campaign status log records DONE for its unit, AND
  * its result json exists and parses, AND
  * every language the published table reported is present, AND
  * full / clean item counts equal the expected values, AND
  * no accuracy is NaN, AND
  * its per-item prediction file exists with one record per full item.

States reported: COMPLETE, FAILED (status log says FAILED), INCOMPLETE (log says
DONE but content check fails), NOT_RUN (no record), INTENTIONAL_HOLD (work the
user explicitly paused -- listed, never counted as complete or failed).

Also enforces the reporting rule that, WITHIN one language, every model is
scored on the identical item set (SIB: index_id; Belebele: link#question).
Across languages the Belebele item sets legitimately differ and are only noted.

Writes analysis/clean_eval_0916/completion_check.json and exits non-zero if
anything in scope is not COMPLETE.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "results" / "downstream" / "exposure_filtered" / "deepseek_clean_eval"
CONDS = ["english_only", "chinese_only", "korean_only", "swahili_only", "bengali_only", "mixed_5lang"]
L5 = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
L3 = ["eng_Latn", "kor_Hang", "zho_Hans"]
SIB_FULL, SIB_CLEAN = 204, 98
BEL_FULL = 200
BEL_CLEAN = {"eng_Latn": 99, "zho_Hans": 99, "swh_Latn": 99, "kor_Hang": 113, "ben_Beng": 125}


def status_log():
    p = ROOT / "campaign_status.tsv"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) == 3:
                out[parts[1]] = parts[2]          # last write wins
    return out


def expected():
    sib = [("baseline", "baseline", L5)] + [(f"{c}_seed{s}", f"{c}/seed{s}", L5) for c in CONDS for s in (0, 1, 2)]
    bel = [("baseline", "baseline/belebele", L5)]
    for c in ("english_only", "chinese_only", "mixed_5lang"):
        bel += [(f"{c}_seed{s}", f"{c}/seed{s}", L3) for s in (0, 1, 2)]
    bel += [(f"korean_only_seed{s}", f"korean_only/seed{s}", L5) for s in (0, 1, 2)]
    bel += [(f"korean_only_seed{s}_pp0.2", f"korean_only/seed{s}", L5) for s in (0, 1, 2)]
    return sib, bel


def check(task, key, unit, langs, log):
    d = ROOT / ("sib200" if task == "sib" else "belebele")
    f, pf = d / f"{key}.json", d / f"{key}_predictions.jsonl"
    problems = []
    # baseline SIB was run standalone before the campaign log existed
    logged = log.get(unit)
    if task == "sib" and key == "baseline":
        logged = "DONE" if f.exists() else None
    if logged == "FAILED":
        return {"state": "FAILED", "problems": ["status log: FAILED"]}
    if not f.exists():
        return {"state": "NOT_RUN" if logged is None else "INCOMPLETE",
                "problems": [f"missing {f.name}"]}
    try:
        res = json.loads(f.read_text())["results"]
    except Exception as e:
        return {"state": "INCOMPLETE", "problems": [f"unparsable json: {e}"]}
    for l in langs:
        if l not in res:
            problems.append(f"{l} absent"); continue
        full, cl = res[l]["full"], res[l]["clean"]
        want_full = SIB_FULL if task == "sib" else BEL_FULL
        want_clean = SIB_CLEAN if task == "sib" else BEL_CLEAN[l]
        if full["n"] != want_full:
            problems.append(f"{l} full n={full['n']} != {want_full}")
        if cl is None or cl["n"] != want_clean:
            problems.append(f"{l} clean n={None if cl is None else cl['n']} != {want_clean}")
        for nm, v in (("full", full), ("clean", cl)):
            if v and (v["acc"] is None or math.isnan(v["acc"])):
                problems.append(f"{l} {nm} acc NaN")
    if not pf.exists():
        problems.append("missing per-item predictions")
    else:
        n = sum(1 for _ in open(pf))
        want = len(langs) * (SIB_FULL if task == "sib" else BEL_FULL)
        if n != want:
            problems.append(f"predictions rows={n} != {want}")
    if logged != "DONE":
        problems.append(f"status log: {logged}")
    return {"state": "COMPLETE" if not problems else "INCOMPLETE", "problems": problems}


def item_sets(task, keys_langs):
    """Per language, the item-id set each model was scored on."""
    d = ROOT / ("sib200" if task == "sib" else "belebele")
    idk = "index_id" if task == "sib" else "qkey"
    sets = {}
    for key, langs in keys_langs:
        pf = d / f"{key}_predictions.jsonl"
        if not pf.exists():
            continue
        for line in open(pf):
            r = json.loads(line)
            sets.setdefault(r["lang"], {}).setdefault(key, set()).add(str(r[idk]))
    report = {}
    for lang, by_model in sets.items():
        distinct = {frozenset(v) for v in by_model.values()}
        report[lang] = {"n_models": len(by_model), "identical_across_models": len(distinct) == 1,
                        "n_items": (len(next(iter(by_model.values()))) if by_model else 0)}
    return report


def main():
    log = status_log()
    sib, bel = expected()
    out = {"sib200": {}, "belebele": {}}
    for key, unit, langs in sib:
        out["sib200"][key] = check("sib", key, unit, langs, log)
    for key, unit, langs in bel:
        out["belebele"][key] = check("bel", key, unit, langs, log)

    out["within_language_item_sets"] = {
        "sib200": item_sets("sib", [(k, l) for k, _, l in sib]),
        "belebele": item_sets("bel", [(k, l) for k, _, l in bel]),
        "note": ("within a language every model must share one item set; across languages Belebele item "
                 "sets differ by construction (per-language split order + limit 200) and that is expected"),
    }
    out["intentional_holds"] = {
        "MASSIVE revised 200-item run": ("paused by user instruction before any test cell; validation complete; "
                                         "resume steps in analysis/massive_0916_revised/RESUME.md"),
        "XL-Sum / E2 remaining 7 cells": "stopped by user 2026-09-16T04:29:07Z; hold maintained",
        "Qwen1.5-MoE": "awaiting delivery; hold maintained",
    }
    tally = {}
    for task in ("sib200", "belebele"):
        for v in out[task].values():
            tally.setdefault(task, {}).setdefault(v["state"], 0)
            tally[task][v["state"]] += 1
    out["tally"] = tally
    bad_sets = [f"{t}/{l}" for t in ("sib200", "belebele")
                for l, v in out["within_language_item_sets"][t].items() if not v["identical_across_models"]]
    out["within_language_mismatch"] = bad_sets
    all_ok = all(v["state"] == "COMPLETE" for t in ("sib200", "belebele") for v in out[t].values()) and not bad_sets
    out["downstream_complete"] = all_ok
    (ROOT / "completion_check.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"[verify] SIB-200:  {tally.get('sib200')}")
    print(f"[verify] Belebele: {tally.get('belebele')}")
    for t in ("sib200", "belebele"):
        for k, v in out[t].items():
            if v["state"] != "COMPLETE":
                print(f"[verify]   {t}/{k}: {v['state']} {v['problems']}")
    for t in ("sib200", "belebele"):
        for l, v in out["within_language_item_sets"][t].items():
            print(f"[verify] {t} {l}: {v['n_models']} models, identical item set={v['identical_across_models']} "
                  f"(n={v['n_items']})")
    print(f"[verify] intentional holds (not counted): {list(out['intentional_holds'])}")
    print(f"[verify] DOWNSTREAM COMPLETE = {all_ok}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
