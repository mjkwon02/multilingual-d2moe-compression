"""Merge-reuse driver for the clean re-evaluations: ONE merge per
(condition, seed), then SIB-200 clean and Belebele clean scored off that same
merged model, and for korean_only the pruning-ON Belebele cell derived in place
from it.

WHY. Both clean re-evaluations need the same compressed models, and a merge
costs ~23 min and ~62 GB of CPU RAM. Run as two separate passes the campaign
needs 30 merges; sharing each merge needs 18. The 12 removed merges are exactly
the (condition, seed) pairs whose model configuration is IDENTICAL across the
two tasks -- verified on every axis that can change the weights, not just on
the condition/seed label:

    base checkpoint revision   521d2bc4fb69a3f3ae565310fcc3b65f97af2580
    Fisher statistics          <cond>/seed<s>/fisher_processed.pt
    whitening (svd_scale)      <cond>/seed<s>/svd_scale_processed.pt
    rank                       delta_ratio 0.8 (no delta_ratio_* path component
                               in either task's stored results => the default)
    structured pruning         OFF (no pp_ratio_* path component)

korean_only's Belebele pruning-ON cell is a DIFFERENT configuration and is
treated as such -- but it is derived from this same merge by applying
phase1_structured_prune in place, which is how the original grid produced it, so
it needs no extra merge. Because pruning mutates the model, the ordering is
forced: SIB -> Belebele OFF -> prune -> Belebele ON. Nothing is scored after
pruning that is supposed to be unpruned.

Each task writes its results the moment it finishes, so a later failure cannot
lose an earlier task's work, and an already-finished task is skipped on restart.

Evaluation settings and clean subsets are NOT harmonised between the two tasks:
SIB keeps its 14-shot prompt / 204-item split / 98-item clean subset, Belebele
keeps its 5-shot lm-eval protocol / limit-200 / 99-item clean subset. Mixing
them would make neither comparable to its published table.

Usage:
    python clean_eval_combined.py --condition english_only --seed 0 \
        --belebele-langs eng_Latn,kor_Hang,zho_Hans
    python clean_eval_combined.py --condition korean_only --seed 0 --belebele-pp-ratio 0.2
    python clean_eval_combined.py --condition swahili_only --seed 0 --no-belebele
"""
import argparse
import fcntl
import gc
import json
import sys
import time
import types
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
D2MOE_DIR = Path(os.environ.get("D2MOE_DIR", SCRIPT_DIR.parents[1] / "third_party" / "D2MoE"))
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
sys.path.insert(0, str(D2MOE_DIR))
sys.path.insert(0, str(D2MOE_DIR / "lm-evaluation-harness"))
import sib200_clean_eval as SIB      # noqa: E402
import belebele_clean_eval as BEL    # noqa: E402
import phase1_structured_prune       # noqa: E402
from phase1_merge_eval import load_model, merge_condition, RESULTS_ROOT  # noqa: E402

LOCK_PATH = "/tmp/phase1_merge_eval.lock"
RUNTIME_DIR = SCRIPT_DIR.parent / "analysis" / "clean_eval_0916" / "runtime"


def verify_config(condition, seed):
    """Confirm the artifacts that define the model exist, and record their
    identity. A missing artifact is reported as MISSING_ARTIFACT and is never
    substituted by another seed or condition."""
    import hashlib
    d = RESULTS_ROOT / condition / f"seed{seed}"
    info = {"condition": condition, "seed": seed, "dir": str(d),
            "delta_ratio": 0.8, "structured_pruning": "OFF at merge time",
            "scale_condition": condition, "scale_seed": seed}
    for req in ("fisher_processed.pt", "svd_scale_processed.pt"):
        p = d / req
        if not p.exists():
            raise RuntimeError(f"MISSING_ARTIFACT: {p}")
        with open(p, "rb") as f:
            info[f"{req}_head_sha256"] = hashlib.sha256(f.read(1 << 20)).hexdigest()[:16]
        info[f"{req}_bytes"] = p.stat().st_size
    freq = sorted(d.glob("deepseek_wikitext_*_expert_frequencies.json"))
    if not freq:
        raise RuntimeError(f"MISSING_ARTIFACT: no expert_frequencies json in {d}")
    info["expert_freq_available"] = [p.name for p in freq]
    info["expert_freq_used_by_glob_order"] = list(d.glob("deepseek_wikitext_*_expert_frequencies.json"))[0].name
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sib-langs", default="eng_Latn,kor_Hang,zho_Hans,swh_Latn,ben_Beng")
    ap.add_argument("--belebele-langs", default="eng_Latn,kor_Hang,zho_Hans,swh_Latn,ben_Beng")
    ap.add_argument("--no-sib", action="store_true")
    ap.add_argument("--no-belebele", action="store_true")
    ap.add_argument("--belebele-pp-ratio", type=float, default=None,
                    help="also score a pruning-ON Belebele cell, derived in place from this same merge")
    ap.add_argument("--sib-batch-size", type=int, default=4)
    ap.add_argument("--belebele-batch-size", type=int, default=1)
    ap.add_argument("--belebele-limit", type=int, default=200)
    ap.add_argument("--belebele-fewshot", type=int, default=5)
    args = ap.parse_args()

    key = f"{args.condition}_seed{args.seed}"
    sib_done = (SIB.OUT_DIR / f"{key}.json").exists()
    bel_done = (BEL.OUT_DIR / f"{key}.json").exists()
    bel_on_key = f"{key}_pp{args.belebele_pp_ratio}" if args.belebele_pp_ratio is not None else None
    bel_on_done = bel_on_key is not None and (BEL.OUT_DIR / f"{bel_on_key}.json").exists()

    want_sib = (not args.no_sib) and not sib_done
    want_bel = (not args.no_belebele) and not bel_done
    want_bel_on = (args.belebele_pp_ratio is not None) and not bel_on_done
    if not (want_sib or want_bel or want_bel_on):
        print(f"[combined] {key}: everything already done (sib={sib_done} bel={bel_done} bel_on={bel_on_done})")
        return

    cfg = verify_config(args.condition, args.seed)
    rt = {"model": key, "config": cfg, "tasks_requested": {"sib": want_sib, "belebele": want_bel,
                                                           "belebele_pruned": want_bel_on},
          "already_done_skipped": {"sib": sib_done, "belebele": bel_done, "belebele_pruned": bel_on_done},
          "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(f"[combined] {key}: sib={want_sib} belebele={want_bel} belebele_pruned={want_bel_on}")

    t = time.time()
    model, tokenizer = load_model()
    rt["seconds_load"] = time.time() - t
    if hasattr(model, "hf_device_map"):
        off = {k: v for k, v in model.hf_device_map.items() if v in ("disk", "cpu")}
        if off:
            raise RuntimeError(f"{len(off)} module(s) offloaded off-GPU -- refusing to run (NaN risk)")
    rt["device_map_unique"] = sorted(set(getattr(model, "hf_device_map", {}).values()))
    print(f"[combined] loaded in {rt['seconds_load']:.0f}s device_map={rt['device_map_unique']}")

    # ---- ONE merge, under the global lock (62GB CPU RAM); released right after
    t_all = time.time(); t_lock = time.time()
    with open(LOCK_PATH, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        rt["seconds_wait_for_lock"] = time.time() - t_lock
        print(f"[combined] merge lock after {rt['seconds_wait_for_lock']:.0f}s wait (wait excluded from work time)")
        model = merge_condition(model, args.condition, args.seed, args.condition, args.seed)
        gc.collect()
    rt["seconds_merge"] = time.time() - t_all - rt["seconds_wait_for_lock"]
    torch.cuda.empty_cache()
    print(f"[combined] merged in {rt['seconds_merge']:.0f}s; lock released -- this merge now serves "
          f"{sum([want_sib, want_bel, want_bel_on])} evaluation(s)")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. SIB clean (unpruned) ----
    if want_sib:
        t = time.time()
        SIB.sib_evaluate_and_write(model, tokenizer, key, args.condition, args.seed,
                                   args.sib_langs.split(","), args.sib_batch_size, False,
                                   {"model": key, "reused_merge": True})
        rt["seconds_sib"] = time.time() - t
        print(f"[combined] SIB done in {rt['seconds_sib']:.0f}s -> saved")

    # ---- 2. Belebele clean (unpruned) ----
    if want_bel:
        t = time.time()
        langs = args.belebele_langs.split(",")
        res = BEL.run(model, tokenizer, langs, args.belebele_limit, args.belebele_fewshot,
                      args.belebele_batch_size)
        stored, stored_path = BEL.stored_acc(key, args.condition, args.seed, None, args.belebele_fewshot)
        ns = types.SimpleNamespace(baseline=False, condition=args.condition, seed=args.seed,
                                   pp_ratio=None, limit=args.belebele_limit,
                                   num_fewshot=args.belebele_fewshot,
                                   batch_size=args.belebele_batch_size, out_tag="")
        BEL.summarize_and_write(key, ns, langs, BEL.clean_keep(), res, time.time() - t,
                                rt["seconds_load"], rt["seconds_merge"], stored, stored_path)
        rt["seconds_belebele"] = time.time() - t
        print(f"[combined] Belebele (OFF) done in {rt['seconds_belebele']:.0f}s -> saved")

    # ---- 3. pruning applied IN PLACE, then the pruning-ON Belebele cell ----
    if want_bel_on:
        t = time.time()
        print(f"[combined] applying structured pruning pp_ratio={args.belebele_pp_ratio} "
              "(mutates the model; nothing unpruned is scored after this point)")
        phase1_structured_prune.apply_structured_pruning(
            model, args.condition, args.seed, args.belebele_pp_ratio, tokenizer)
        gc.collect(); torch.cuda.empty_cache()
        rt["seconds_prune"] = time.time() - t
        t = time.time()
        langs = args.belebele_langs.split(",")
        res = BEL.run(model, tokenizer, langs, args.belebele_limit, args.belebele_fewshot,
                      args.belebele_batch_size)
        stored, stored_path = BEL.stored_acc(key, args.condition, args.seed, args.belebele_pp_ratio,
                                             args.belebele_fewshot)
        ns = types.SimpleNamespace(baseline=False, condition=args.condition, seed=args.seed,
                                   pp_ratio=args.belebele_pp_ratio, limit=args.belebele_limit,
                                   num_fewshot=args.belebele_fewshot,
                                   batch_size=args.belebele_batch_size, out_tag="")
        BEL.summarize_and_write(bel_on_key, ns, langs, BEL.clean_keep(), res, time.time() - t,
                                rt["seconds_load"], rt["seconds_merge"], stored, stored_path)
        rt["seconds_belebele_pruned"] = time.time() - t
        print(f"[combined] Belebele (ON, pp={args.belebele_pp_ratio}) done in "
              f"{rt['seconds_belebele_pruned']:.0f}s -> saved")

    rt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rt["peak_gpu_bytes"] = {str(d): torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())}
    try:
        import resource
        rt["peak_host_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:
        pass
    (RUNTIME_DIR / f"{key}.json").write_text(json.dumps(rt, indent=2))
    print(f"[combined] {key}: wrote runtime -> {RUNTIME_DIR / (key + '.json')}")


if __name__ == "__main__":
    main()
