"""SIB-200 clean re-evaluation: same protocol as phase1_sib200_eval.py, but
persisting PER-ITEM predictions keyed by the dataset's own `index_id`.

WHY A RE-RUN IS UNAVOIDABLE. The original eval_sib200.json files store only
`acc`, `pred_distribution` and ten `_sample_items` -- and those ten carry no
sample id. There is therefore no way to filter the existing results down to the
audit's clean subset on CPU. Every cell has to be scored again. That is the
single reason this script exists; nothing about the task changes.

WHAT IS HELD FIXED (so the clean numbers are comparable to Table 2):
prompt template, the fixed English 14-shot demonstrations (demo_seed 42,
2 per category), the 7 single-token candidate words, score_batch, batch size,
dtype, and the merge/scale configuration. All of it is imported from
phase1_sib200_eval.py rather than restated, so it cannot drift.

WHAT IS ADDED: per-item records (index_id, gold, pred, 7 logprobs), and the
full-204 accuracy is recomputed alongside the clean-98 accuracy. The full-204
number is a REPRODUCTION CHECK against the stored Table 2 value -- if it does
not reproduce, that is itself a finding and is reported, not smoothed over.

Clean subset: analysis/data_audit_0916/clean_eval_ids.json, sib200_test,
`primary` (identical to `strict` here, and the keep set is identical across all
five languages -- verified, not assumed). 98 of 204 kept.

Demonstration exposure is a SEPARATE issue from test exposure: the audit
reports 6 of the 14 demonstrations are calibration-exposed. This run does NOT
change the demonstrations -- doing so would mix two evaluation protocols and
make the clean numbers incomparable to Table 2. It is reported alongside.

Originals are never overwritten; everything lands under
analysis/clean_eval_0916/sib200/.

Usage:
    python sib200_clean_eval.py --baseline
    python sib200_clean_eval.py --condition english_only --seed 0
"""
import argparse
import fcntl
import gc
import json
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
D2MOE_DIR = Path(os.environ.get("D2MOE_DIR", SCRIPT_DIR.parents[1] / "third_party" / "D2MoE"))
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
sys.path.insert(0, str(D2MOE_DIR))
import phase1_sib200_eval as S          # noqa: E402 -- prompt/demos/candidates reused unchanged
from phase1_xnli_eval import score_batch  # noqa: E402 -- reused unchanged
from phase1_merge_eval import load_model, merge_condition, RESULTS_ROOT  # noqa: E402

OUT_DIR = SCRIPT_DIR.parent / "analysis" / "clean_eval_0916" / "sib200"
AUDIT = SCRIPT_DIR.parents[1] / "results" / "source_data" / "data_exposure_audit" / "clean_eval_ids.json"
LOCK_PATH = "/tmp/phase1_merge_eval.lock"
LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]


def clean_keep():
    d = json.loads(AUDIT.read_text())["sib200_test"]["per_language"]
    keep = {l: set(map(str, d[l]["primary"]["keep_index_ids"])) for l in LANGS}
    if len({frozenset(v) for v in keep.values()}) != 1:
        raise RuntimeError("clean keep sets differ across languages -- the audit asserts they are identical")
    return keep


def original_acc(model_key, condition, seed):
    """The already-published full-204 accuracy for this cell, for the
    reproduction check. Returns {} if the original file is absent."""
    if model_key == "baseline":
        p = RESULTS_ROOT / "baseline" / "eval_sib200.json"
    else:
        p = RESULTS_ROOT / condition / f"seed{seed}" / f"scale_{condition}_seed{seed}" / "eval_sib200.json"
    if not p.exists():
        return {}, str(p)
    d = json.loads(p.read_text())["results"]
    return {l: d[l]["acc"] for l in LANGS if l in d}, str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)   # same as the original runs
    ap.add_argument("--langs", default=None)
    args = ap.parse_args()
    if not args.baseline and not args.condition:
        ap.error("either --condition or --baseline is required")

    from datasets import load_dataset
    langs = args.langs.split(",") if args.langs else LANGS
    keep = clean_keep()
    key = "baseline" if args.baseline else f"{args.condition}_seed{args.seed}"
    out_path = OUT_DIR / f"{key}.json"
    if out_path.exists():
        print(f"[sib-clean] {key} already done -> {out_path}")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rt = {"model": key, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "seconds_wait_for_lock": 0.0, "seconds_merge": 0.0}
    t = time.time()
    model, tokenizer = load_model()
    rt["seconds_load"] = time.time() - t
    if hasattr(model, "hf_device_map"):
        off = {k: v for k, v in model.hf_device_map.items() if v in ("disk", "cpu")}
        if off:
            raise RuntimeError(f"{len(off)} module(s) offloaded off-GPU -- refusing to run (NaN risk)")
    rt["device_map_unique"] = sorted(set(getattr(model, "hf_device_map", {}).values()))
    print(f"[sib-clean] {key}: loaded in {rt['seconds_load']:.0f}s device_map={rt['device_map_unique']}")

    if not args.baseline:
        cond_dir = RESULTS_ROOT / args.condition / f"seed{args.seed}"
        for req in ("fisher_processed.pt", "svd_scale_processed.pt"):
            if not (cond_dir / req).exists():
                raise RuntimeError(f"MISSING_ARTIFACT: {cond_dir/req} -- not substituted by another seed")
        t_all = time.time(); t_lock = time.time()
        with open(LOCK_PATH, "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            rt["seconds_wait_for_lock"] = time.time() - t_lock
            print(f"[sib-clean] merge lock after {rt['seconds_wait_for_lock']:.0f}s wait")
            model = merge_condition(model, args.condition, args.seed, args.condition, args.seed)
            gc.collect()
        rt["seconds_merge"] = time.time() - t_all - rt["seconds_wait_for_lock"]
        torch.cuda.empty_cache()
        print(f"[sib-clean] merged in {rt['seconds_merge']:.0f}s; lock released")

    sib_evaluate_and_write(model, tokenizer, key, args.condition, args.seed, langs,
                           args.batch_size, args.baseline, rt)


def sib_evaluate_and_write(model, tokenizer, key, condition, seed, langs, batch_size, is_baseline, rt):
    """SIB clean evaluation for an ALREADY-BUILT model (baseline or merged).

    Split out of main() so the merge-reuse driver can score SIB and Belebele
    from a single merged model instead of merging twice per condition/seed.
    Everything protocol-relevant -- demonstrations, prompt, candidates,
    score_batch, batch size -- comes from phase1_sib200_eval, unchanged."""
    from datasets import load_dataset
    keep = clean_keep()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{key}.json"

    demos = S.build_demonstrations(seed=S.DEMO_SEED_DEFAULT, per_label=S.DEMOS_PER_LABEL_DEFAULT)
    choice_ids = S._choice_token_ids(tokenizer)
    orig, orig_path = original_acc(key, condition, seed)

    results, per_item = {}, []
    for lang in langs:
        t0 = time.time()
        ds = load_dataset(S.SIB_DATASET, lang, split="test")
        pred, gold, idx = [], [], []
        for i in range(0, len(ds), batch_size):
            chunk = ds[i:i + batch_size]
            prompts = [S.build_prompt(demos, t) for t in chunk["text"]]
            lp = score_batch(model, tokenizer, prompts, choice_ids)
            if torch.isnan(lp).any():
                raise RuntimeError(f"NaN logprob in {lang} at item {i} -- refusing to continue")
            bp = lp.argmax(dim=-1).tolist()
            bg = [S.CATEGORY_TO_IDX[c] for c in chunk["category"]]
            for j in range(len(bp)):
                per_item.append({
                    "model": key, "lang": lang, "index_id": str(chunk["index_id"][j]),
                    "gold": S.CATEGORY_ORDER[bg[j]], "pred": S.CATEGORY_ORDER[bp[j]],
                    "correct": int(bp[j] == bg[j]),
                    "in_clean": int(str(chunk["index_id"][j]) in keep[lang]),
                    "logprobs": [round(lp[j, k].item(), 6) for k in range(len(S.CATEGORY_ORDER))],
                })
            pred += bp; gold += bg; idx += [str(x) for x in chunk["index_id"]]
        secs = time.time() - t0

        def summarize(sel):
            n = len(sel)
            if not n:
                return None
            a = sum(pred[k] == gold[k] for k in sel) / n
            dist = {c: 0 for c in S.CATEGORY_ORDER}
            gd = {c: 0 for c in S.CATEGORY_ORDER}
            for k in sel:
                dist[S.CATEGORY_ORDER[pred[k]]] += 1
                gd[S.CATEGORY_ORDER[gold[k]]] += 1
            return {"n": n, "acc": a, "pred_distribution": dist, "gold_distribution": gd,
                    "top_label": max(dist, key=dist.get), "top_label_share": max(dist.values()) / n,
                    "n_distinct_pred": sum(1 for v in dist.values() if v)}

        allsel = list(range(len(gold)))
        cleansel = [k for k in allsel if idx[k] in keep[lang]]
        full, cl = summarize(allsel), summarize(cleansel)
        repro = None
        if lang in orig:
            repro = {"original_acc": orig[lang], "rerun_full_acc": full["acc"],
                     "abs_diff": abs(orig[lang] - full["acc"]),
                     "reproduced": abs(orig[lang] - full["acc"]) < 1e-9}
        results[lang] = {"full": full, "clean": cl, "reproduction_check": repro,
                         "seconds": secs, "items_per_second": len(gold) / secs}
        print(f"[sib-clean] {key}/{lang}: full={full['acc']:.4f} (n={full['n']})  "
              f"clean={cl['acc']:.4f} (n={cl['n']})  "
              f"{'repro OK' if repro and repro['reproduced'] else ('repro D=%.4f' % repro['abs_diff']) if repro else 'no original'}"
              f"  [{secs:.0f}s, {len(gold)/secs:.2f} it/s]")
        gc.collect(); torch.cuda.empty_cache()

    rt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rt["peak_gpu_bytes"] = {str(d): torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())}
    payload = {
        "task": "sib200", "model": key,
        "condition": ("baseline" if is_baseline else condition),
        "seed": (None if is_baseline else seed),
        "protocol": "sib200_natural_continuation_v1 (unchanged from phase1_sib200_eval.py)",
        "num_fewshot": len(S.CATEGORY_ORDER) * S.DEMOS_PER_LABEL_DEFAULT,
        "demo_seed": S.DEMO_SEED_DEFAULT, "batch_size": batch_size,
        "clean_subset_source": str(AUDIT.relative_to(SCRIPT_DIR.parent)),
        "clean_variant": "primary (identical to strict for SIB; keep set identical across all 5 languages)",
        "demonstration_exposure_note": ("the audit reports 6 of the 14 fixed English demonstrations are "
                                        "calibration-exposed. Demonstrations are deliberately NOT changed here: "
                                        "changing them would alter the evaluation protocol and make these "
                                        "numbers incomparable to Table 2. Reported separately from test exposure."),
        "original_file_for_reproduction_check": orig_path,
        "results": results, "runtime": rt,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    with open(OUT_DIR / f"{key}_predictions.jsonl", "w") as f:
        for r in per_item:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[sib-clean] wrote {out_path}")


if __name__ == "__main__":
    main()
