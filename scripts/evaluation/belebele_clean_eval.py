"""Belebele clean re-evaluation: identical lm-eval protocol to
phase1_belebele_eval.py, but with log_samples=True so per-question results
exist and the audit's clean subset can be isolated.

WHY A RE-RUN IS UNAVOIDABLE. The stored eval_belebele.json files contain only
`acc`, `acc_norm` and `n_samples` -- no per-question record at all, because the
original runs passed log_samples=False. Nothing can be filtered on CPU.

PROTOCOL HELD FIXED to match Table 28: lm-eval 0.4.3 (the same vendored source
the original runs used), `--limit 200`, `--num-fewshot 5`, `--batch-size 1`,
the same per-condition language subsets, and the same merge/scale/pruning
configuration. Only log_samples changes.

lm_eval is reached by sys.path, NOT by installing anything: the editable
install in this env points at <repo-root>/d2moe-multilingual/
..., a path that no longer exists because the tree was renamed to . The
identical source is still vendored at third_party/D2MoE/lm-evaluation-harness, so the
package environment is left untouched.

CLEAN SUBSET -- the important subtlety. The audit's 549 clean questions are
computed over the FULL 900-question test split, but every reported cell was run
with `--limit 200`, i.e. the first 200 documents. The usable clean subset is
therefore the INTERSECTION, which is 99 questions over 52 passages, not 549.
Exposure is removed at PASSAGE granularity (every question attached to an
exposed passage is dropped), per the audit's passage_rule.

Usage:
    python belebele_clean_eval.py --baseline --langs eng_Latn      # speed probe
    python belebele_clean_eval.py --baseline
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
D2MOE_DIR = Path(os.environ.get("D2MOE_DIR", SCRIPT_DIR.parents[1] / "third_party" / "D2MoE"))
LM_EVAL_SRC = D2MOE_DIR / "lm-evaluation-harness"
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
sys.path.insert(0, str(D2MOE_DIR))
sys.path.insert(0, str(LM_EVAL_SRC))
import phase1_belebele_eval as B  # noqa: E402 -- language/task mapping reused unchanged
from phase1_merge_eval import RESULTS_ROOT  # noqa: E402

OUT_DIR = SCRIPT_DIR.parent / "analysis" / "clean_eval_0916" / "belebele"
AUDIT = SCRIPT_DIR.parents[1] / "results" / "source_data" / "data_exposure_audit" / "clean_eval_ids.json"
DEFAULT_LIMIT = 200
DEFAULT_FEWSHOT = 5
DEFAULT_BATCH = 1


def clean_keep():
    d = json.loads(AUDIT.read_text())["belebele_test"]["per_language"]
    keep = {l: set(map(str, d[l]["primary"]["keep_question_ids"])) for l in B.BELEBELE_LANGS}
    if len({frozenset(v) for v in keep.values()}) != 1:
        raise RuntimeError("clean keep sets differ across languages -- audit asserts they are identical")
    return keep


def qkey(doc):
    return f"{doc['link']}#q{doc['question_number']}"


def stored_acc(model_key, condition, seed, pp_ratio, num_fewshot):
    if model_key == "baseline":
        p = RESULTS_ROOT / "baseline"
    else:
        p = RESULTS_ROOT / condition / f"seed{seed}" / f"scale_{condition}_seed{seed}"
        if pp_ratio is not None:
            p = p / f"pp_ratio_{pp_ratio}"
    if num_fewshot:
        p = p / f"fewshot_{num_fewshot}"
    p = p / "eval_belebele.json"
    if not p.exists():
        return {}, str(p)
    d = json.loads(p.read_text())["results"]
    return {l: v["acc"] for l, v in d.items()}, str(p)


def run(model, tokenizer, langs, limit, num_fewshot, batch_size):
    """One lm-eval pass with per-sample logging. Returns
    {lang: {"per_item": [...], "raw_acc": float}}"""
    from lm_eval import evaluator, tasks
    lang_subset = {l: B.BELEBELE_LANGS[l] for l in langs}
    raw = evaluator.simple_evaluate(
        model=model, tokenizer=tokenizer, tasks=list(lang_subset.values()),
        batch_size=batch_size, device=next(model.parameters()).device,
        write_out=False, log_samples=True, verbosity="WARNING",
        num_fewshot=num_fewshot, task_manager=tasks.TaskManager(), limit=limit,
    )
    out = {}
    for lang, task in lang_subset.items():
        samples = raw.get("samples", {}).get(task)
        if samples is None:
            raise RuntimeError(f"no logged samples for {task} -- log_samples did not take effect")
        items = []
        for s in samples:
            doc = s["doc"]
            acc = s.get("acc")
            accn = s.get("acc_norm")
            if acc is None:
                raise RuntimeError(f"sample record for {task} has no per-doc 'acc'; keys={list(s.keys())}")
            items.append({"qkey": qkey(doc), "link": doc["link"],
                          "question_number": str(doc["question_number"]),
                          "gold": str(doc["correct_answer_num"]),
                          "acc": float(acc), "acc_norm": (None if accn is None else float(accn))})
        r = raw["results"].get(task, {})
        out[lang] = {"per_item": items, "reported_acc": B._extract_metric(r, "acc"),
                     "reported_acc_norm": B._extract_metric(r, "acc_norm")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pp-ratio", type=float, default=None)
    ap.add_argument("--langs", default=None)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--num-fewshot", type=int, default=DEFAULT_FEWSHOT)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()
    if not args.baseline and not args.condition:
        ap.error("either --condition or --baseline is required")

    langs = args.langs.split(",") if args.langs else list(B.BELEBELE_LANGS)
    keep = clean_keep()
    key = "baseline" if args.baseline else f"{args.condition}_seed{args.seed}"
    if args.pp_ratio is not None:
        key += f"_pp{args.pp_ratio}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # NOTE: this entrypoint is for the baseline / speed probe. The merge-reuse
    # driver imports run() directly so one merged model serves SIB and Belebele.
    from phase1_merge_eval import load_model, merge_condition
    import phase1_structured_prune
    t = time.time()
    model, tokenizer = load_model()
    load_s = time.time() - t
    if hasattr(model, "hf_device_map"):
        off = {k: v for k, v in model.hf_device_map.items() if v in ("disk", "cpu")}
        if off:
            raise RuntimeError(f"{len(off)} module(s) offloaded off-GPU")
    print(f"[bel-clean] {key}: loaded in {load_s:.0f}s")
    merge_s = 0.0
    if not args.baseline:
        import fcntl, gc
        t0 = time.time()
        with open("/tmp/phase1_merge_eval.lock", "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            model = merge_condition(model, args.condition, args.seed, args.condition, args.seed)
            gc.collect()
        merge_s = time.time() - t0
        torch.cuda.empty_cache()
        if args.pp_ratio is not None:
            phase1_structured_prune.apply_structured_pruning(
                model, args.condition, args.seed, args.pp_ratio, tokenizer)

    t0 = time.time()
    res = run(model, tokenizer, langs, args.limit, args.num_fewshot, args.batch_size)
    secs = time.time() - t0
    stored, stored_path = stored_acc(key.replace(f"_pp{args.pp_ratio}", ""), args.condition, args.seed,
                                    args.pp_ratio, args.num_fewshot)
    summarize_and_write(key, args, langs, keep, res, secs, load_s, merge_s, stored, stored_path)


def summarize_and_write(key, args, langs, keep, res, secs, load_s, merge_s, stored, stored_path):
    results = {}
    per_item_all = []
    for lang in langs:
        items = res[lang]["per_item"]
        for it in items:
            it2 = dict(it); it2["model"] = key; it2["lang"] = lang
            it2["in_clean"] = int(it["qkey"] in keep[lang])
            per_item_all.append(it2)
        full = [it["acc"] for it in items]
        cl = [it["acc"] for it in items if it["qkey"] in keep[lang]]
        npass_full = len({it["link"] for it in items})
        npass_cl = len({it["link"] for it in items if it["qkey"] in keep[lang]})
        gold_full = {}
        gold_cl = {}
        for it in items:
            gold_full[it["gold"]] = gold_full.get(it["gold"], 0) + 1
            if it["qkey"] in keep[lang]:
                gold_cl[it["gold"]] = gold_cl.get(it["gold"], 0) + 1
        repro = None
        if lang in stored:
            rr = sum(full) / len(full)
            repro = {"stored_acc": stored[lang], "rerun_full_acc": rr,
                     "abs_diff": abs(stored[lang] - rr), "reproduced": abs(stored[lang] - rr) < 1e-9}
        results[lang] = {
            "full": {"n": len(full), "acc": sum(full) / len(full), "n_passages": npass_full,
                     "gold_distribution": dict(sorted(gold_full.items()))},
            "clean": ({"n": len(cl), "acc": sum(cl) / len(cl), "n_passages": npass_cl,
                       "gold_distribution": dict(sorted(gold_cl.items()))} if cl else None),
            "reported_acc_from_harness": res[lang]["reported_acc"],
            "reproduction_check": repro,
        }
        c = results[lang]["clean"]
        print(f"[bel-clean] {key}/{lang}: full={results[lang]['full']['acc']:.4f} (n={len(full)}) "
              f"clean={c['acc']:.4f} (n={c['n']}, {c['n_passages']} passages) "
              + (f"repro {'OK' if repro['reproduced'] else 'Δ=%.4f' % repro['abs_diff']}" if repro else "no stored"))
    payload = {
        "task": "belebele", "model": key,
        "condition": ("baseline" if args.baseline else args.condition),
        "seed": (None if args.baseline else args.seed), "pp_ratio": args.pp_ratio,
        "limit": args.limit, "num_fewshot": args.num_fewshot, "batch_size": args.batch_size,
        "langs": langs,
        "lm_eval_source": str(LM_EVAL_SRC),
        "clean_subset_source": str(AUDIT.relative_to(SCRIPT_DIR.parent)),
        "clean_note": ("audit clean = 549/900 over the FULL split; this cell evaluated limit=200, so the usable "
                       "clean subset is the intersection (99 questions / 52 passages). Exposure removed at "
                       "passage granularity."),
        "stored_file_for_reproduction_check": stored_path,
        "seconds_eval": secs, "seconds_load": load_s, "seconds_merge": merge_s,
        "results": results,
    }
    (OUT_DIR / f"{key}{args.out_tag}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    with open(OUT_DIR / f"{key}{args.out_tag}_predictions.jsonl", "w") as f:
        for r in per_item_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[bel-clean] wrote {OUT_DIR / (key + args.out_tag + '.json')} [{secs:.0f}s eval]")


if __name__ == "__main__":
    main()
