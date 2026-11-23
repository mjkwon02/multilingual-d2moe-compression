"""Revised (2026-09-16) MASSIVE runner: 200 test items, English vs
Balanced-character, seeds 0/1/2, GPU 2-3 single stream, 12h wall-clock budget.

REUSES, does not reimplement: massive_eval.CachedScorer / score_reference /
build_candidate_tokens (the reference-validated scorer), massive_build_protocol's
prompt builders, and phase1_merge_eval.load_model / merge_condition. What is new
here is only the revision's I/O and bookkeeping contract:

  * per-item incremental persistence (append-only jsonl, flushed every item),
    so an interruption loses at most the item in flight;
  * resume keyed on (model identity, condition, seed, locale, source ID,
    inference_signature) -- completed items are skipped on restart;
  * inference_signature (what actually determines the numbers: model path,
    dtype, scorer mode, microbatch, prompt hash, label map, score definition)
    kept SEPARATE from protocol_hash (the plan: grid, budget, subset rules).
    A 600->200 subset change alters protocol_hash but not
    inference_signature, so already-scored items stay reusable, which is the
    whole point of separating them;
  * runtime accounting that distinguishes WAITING for the global merge lock
    from ACTIVE work, because the budget excludes the former.

The merge holds /tmp/phase1_merge_eval.lock for the merge only (~62GB CPU RAM
for fisher + svd_scale); it is released before scoring so other jobs are not
blocked for hours. No other process is killed and the lock is never forced.

Usage:
    python massive_revised_run.py --baseline
    python massive_revised_run.py --condition english_only --seed 0
"""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
D2MOE_DIR = Path(os.environ.get("D2MOE_DIR", SCRIPT_DIR.parents[1] / "third_party" / "D2MoE"))
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
sys.path.insert(0, str(D2MOE_DIR))
import massive_data as M  # noqa: E402
import massive_build_protocol as P  # noqa: E402
import massive_eval as E  # noqa: E402 -- validated scorer, reused unchanged
from phase1_merge_eval import load_model, merge_condition, RESULTS_ROOT, MODEL_PATH  # noqa: E402

NEW_DIR = SCRIPT_DIR.parents[1] / "results" / "downstream" / "massive_revised"
PRED_PATH = NEW_DIR / "predictions.jsonl"
LOCK_PATH = "/tmp/phase1_merge_eval.lock"
SCORER_CODE_VERSION = "massive_eval.CachedScorer@2026-09-16-cachefix"


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def inference_signature(prompt_id, prompt_sha, mode, microbatch):
    """Everything that can change a produced NUMBER. Deliberately excludes the
    grid/subset plan, so shrinking 600->200 does not invalidate scored items."""
    return sha({
        "model_path": MODEL_PATH, "dtype": "bfloat16",
        "scorer": SCORER_CODE_VERSION, "mode": mode, "microbatch": microbatch,
        "prompt_id": prompt_id, "prompt_sha256": prompt_sha,
        "label_surface_map": M.LABEL_SURFACE, "label_order": M.PROMPT_LABEL_ORDER,
        "score": "mean_token_logprob_of_label_completion",
        "eos_scored": False, "context_tokens_scored": False,
    })


def load_done(sig):
    """(model, lang, id) already scored under this exact inference signature."""
    done = set()
    if PRED_PATH.exists():
        with open(PRED_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn final line from a hard kill; re-scored below
                if r.get("inference_signature") == sig:
                    done.add((r["model"], r["lang"], r["id"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["english_only", "mixed_5lang"])
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--prompt", default=None, help="default: the prompt the revised pilot selected")
    ap.add_argument("--microbatch", type=int, default=None, help="default: the probe's chosen value")
    ap.add_argument("--mode", default=None, choices=["shared_prefix", "per_item"])
    ap.add_argument("--langs", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--deadline-utc", default=os.environ.get("MASSIVE_DEADLINE_UTC", ""),
                    help="ISO8601; stop at a safe item boundary instead of starting work past it")
    args = ap.parse_args()
    if not args.baseline and not args.condition:
        ap.error("either --condition or --baseline is required")

    protocol = json.loads((NEW_DIR / "protocol.json").read_text())
    vp_path = NEW_DIR / "validation_and_pilot.json"
    vp = json.loads(vp_path.read_text()) if vp_path.exists() else {}
    prompt_id = args.prompt or vp.get("prompt_selection", {}).get("selected")
    mode = args.mode or vp.get("microbatch_probe", {}).get("scoring_mode")
    mb = args.microbatch or vp.get("microbatch_probe", {}).get("selected_microbatch")
    if not prompt_id or not mode or not mb:
        ap.error("prompt/mode/microbatch unresolved -- run massive_revised_pilot.py first "
                 f"(got prompt={prompt_id} mode={mode} mb={mb})")

    ids_file = "test_ids.json" if args.split == "test" else "dev_ids.json"
    eval_ids = json.loads((NEW_DIR / ids_file).read_text())["ids"]
    if args.limit:
        eval_ids = eval_ids[:args.limit]
    langs = args.langs.split(",") if args.langs else list(M.LOCALES)

    data = M.load_all()
    demo_meta = json.loads((SCRIPT_DIR.parents[1] / "data" / "splits" / "massive_demo_ids.json").read_text())
    demos = [data[demo_meta["lang"]][i] for i in demo_meta["ids"]]
    shared_text = P.build_shared_prefix([] if prompt_id == "P0" else demos)
    prompt_sha = M.sha256_text(shared_text)
    if prompt_sha != protocol["prompts"][f"{prompt_id}_shared_prefix_sha256"]:
        raise RuntimeError(f"{prompt_id} prompt hash drifted from the frozen protocol -- refusing to run")
    surfaces = [M.LABEL_SURFACE[lab] for lab in M.PROMPT_LABEL_ORDER]

    sig = inference_signature(prompt_id, prompt_sha, mode, mb)
    key = "baseline" if args.baseline else f"{args.condition}_seed{args.seed}"
    done = load_done(sig)
    todo = {l: [i for i in eval_ids if (key, l, i) not in done] for l in langs}
    n_todo = sum(len(v) for v in todo.values())
    print(f"[run] model={key} split={args.split} prompt={prompt_id} mode={mode} mb={mb}")
    print(f"[run] inference_signature={sig[:16]} items to score={n_todo} "
          f"(already done={len(eval_ids)*len(langs)-n_todo})")
    if n_todo == 0:
        print("[run] nothing to do")
        return

    deadline = None
    if args.deadline_utc:
        import datetime as _dt
        deadline = _dt.datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")).timestamp()

    rt = {"model": key, "split": args.split, "prompt": prompt_id, "mode": mode, "microbatch": mb,
          "inference_signature": sig, "protocol_hash": sha(protocol),
          "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "seconds_wait_for_lock": 0.0, "seconds_merge": 0.0, "seconds_load": 0.0,
          "seconds_eval_per_lang": {}, "failures": []}

    t = time.time()
    model, tokenizer = load_model()
    rt["seconds_load"] = time.time() - t
    if hasattr(model, "hf_device_map"):
        off = {k: v for k, v in model.hf_device_map.items() if v in ("disk", "cpu")}
        if off:
            raise RuntimeError(f"{len(off)} module(s) offloaded off-GPU (e.g. {next(iter(off.items()))})")
    rt["device_map_unique"] = sorted(set(getattr(model, "hf_device_map", {}).values()))
    rt["visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"[run] loaded in {rt['seconds_load']:.0f}s device_map={rt['device_map_unique']} "
          f"CUDA_VISIBLE_DEVICES={rt['visible_devices']}")

    if not args.baseline:
        cond_dir = RESULTS_ROOT / args.condition / f"seed{args.seed}"
        globbed = sorted(cond_dir.glob("deepseek_wikitext_*_expert_frequencies.json"))
        by_glob = list(cond_dir.glob("deepseek_wikitext_*_expert_frequencies.json"))
        if not by_glob:
            raise RuntimeError(f"MISSING_ARTIFACT: no expert_freq json in {cond_dir}")
        for req in ("fisher_processed.pt", "svd_scale_processed.pt"):
            if not (cond_dir / req).exists():
                raise RuntimeError(f"MISSING_ARTIFACT: {cond_dir/req} absent -- this model is NOT substituted "
                                   "by another seed or condition")
        rt["expert_freq_file"] = {"available": [p.name for p in globbed], "used_by_glob_order": by_glob[0].name}
        rt["stats_sha_prefix"] = {
            req: hashlib.sha256(open(cond_dir / req, "rb").read(1 << 20)).hexdigest()[:16]
            for req in ("fisher_processed.pt", "svd_scale_processed.pt")}
        t_all = time.time()
        t_lock = time.time()
        with open(LOCK_PATH, "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            rt["seconds_wait_for_lock"] = time.time() - t_lock
            print(f"[run] merge lock acquired after {rt['seconds_wait_for_lock']:.0f}s wait "
                  "(wait time is excluded from the work budget)")
            model = merge_condition(model, args.condition, args.seed, args.condition, args.seed)
            gc.collect()
        rt["seconds_merge"] = time.time() - t_all - rt["seconds_wait_for_lock"]
        torch.cuda.empty_cache()
        print(f"[run] merged in {rt['seconds_merge']:.0f}s; lock released")

    stopped_early = None
    with open(PRED_PATH, "a") as pf:
        for l in langs:
            if not todo[l]:
                continue
            if deadline and time.time() > deadline:
                stopped_early = f"deadline reached before {l}"
                break
            t0 = time.time()
            scorer = E.CachedScorer(model, tokenizer, shared_text, surfaces, microbatch=mb,
                                    share_prefix=(mode == "shared_prefix"))
            n = 0
            for i in todo[l]:
                if deadline and time.time() > deadline:
                    stopped_early = f"deadline reached mid-{l} after {n} items"
                    break
                row = data[l][i]
                means, sums, ntoks = scorer.score(P.build_item_tail(row["utt"]))
                if not all(math.isfinite(x) for x in means):
                    raise RuntimeError(f"non-finite score for {l} id={i} -- aborting (gate)")
                k = max(range(60), key=lambda j: means[j])
                pred = M.PROMPT_LABEL_ORDER[k]
                pf.write(json.dumps({
                    "model": key, "condition": ("baseline" if args.baseline else args.condition),
                    "seed": (None if args.baseline else args.seed),
                    "lang": l, "locale": M.LOCALES[l], "split": args.split, "id": i,
                    "prompt": prompt_id, "inference_signature": sig, "protocol_hash": sha(protocol),
                    "gold": row["intent"], "pred": pred, "correct": int(pred == row["intent"]),
                    "mean_logprob": [round(x, 6) for x in means],
                    "sum_logprob": [round(x, 6) for x in sums],
                    "n_completion_tokens": ntoks,
                }, ensure_ascii=False) + "\n")
                pf.flush()          # per-item durability (§7)
                os.fsync(pf.fileno())
                n += 1
            scorer.verify_cache_intact()
            secs = time.time() - t0
            rt["seconds_eval_per_lang"][l] = {"seconds": secs, "n_scored": n,
                                              "items_per_second": (n / secs if secs else None)}
            print(f"[run] {l}: scored {n} items in {secs:.0f}s ({n/secs:.2f} it/s)" if secs else f"[run] {l}: 0")
            del scorer
            gc.collect()
            torch.cuda.empty_cache()
            if stopped_early:
                break

    rt["peak_gpu_bytes"] = {str(d): torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())}
    try:
        import resource
        rt["peak_host_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:
        pass
    rt["stopped_early"] = stopped_early
    rt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rt_dir = NEW_DIR / "runtime"
    rt_dir.mkdir(exist_ok=True)
    (rt_dir / f"{key}__{args.split}.json").write_text(json.dumps(rt, indent=2))
    print(f"[run] wrote {rt_dir / (key + '__' + args.split + '.json')}")
    if stopped_early:
        print(f"[run] STOPPED EARLY: {stopped_early}")


if __name__ == "__main__":
    main()
