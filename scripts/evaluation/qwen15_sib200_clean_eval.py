"""Qwen1.5-MoE-A2.7B SIB-200 *test-item clean* evaluation (98 unexposed test
items per language), mirroring scripts/sib200_clean_eval.py for DeepSeek.

WHAT IS HELD FIXED (imported, never restated, so it cannot drift from the
DeepSeek clean run): the 14 fixed English demonstrations (demo_seed 42, 2 per
category, validation split), prompt template, the 7 single-token candidate
words, score_batch() (last-real-token logits, log_softmax, right padding),
batch size 4, the clean keep set (analysis/data_audit_0916/clean_eval_ids.json,
sib200_test primary, 98 of 204, identical across the 5 languages), and the
collapse criterion applied at aggregation (top-label share > 0.9 OR acc < 25%).

Demonstrations are NOT changed (the audit reports 6 of the 14 are
calibration-exposed), so this is a TEST-ITEM clean evaluation, not an
evaluation with calibration exposure fully removed. The output records this.

WHAT DIFFERS FOR QWEN (input processing only; recorded in every output):
  * tokenizer: slow Qwen2Tokenizer (use_fast=False, as every qwen15_* stage);
    it adds NO BOS token (bos_token is null), whereas the DeepSeek tokenizer
    prepends <｜begin▁of▁sentence｜>. The prompt is therefore scored without any
    prefix token -- the same choice as the Qwen FLORES primary bpb metric.
  * pad token = <|endoftext|> (eos), set by qwen15_config.load_tokenizer();
    score_batch() right-pads and reads each row at its own last real token, so
    padding never touches the scored position.
  * the 7 candidate words are single tokens for Qwen too (checked at runtime;
    ids recorded), so the scoring arithmetic is identical.
  * the model is the pinned base checkpoint with the SAVED compressed experts
    installed (qwen15_merge_eval.load_saved_compressed, per-layer digest check
    against the manifest) -- no Fisher/whitening/merge recomputation.

Outputs (never overwritten) under
results/qwen/sib200_clean/:
  <key>.json                per-language full-204 and clean-98 summaries
  <key>_predictions.jsonl   one record per (language, item): index_id, gold,
                            pred, correct, in_clean, 7 logprobs

Usage:
    python -u qwen15_sib200_clean_eval.py --baseline
    python -u qwen15_sib200_clean_eval.py --condition english_only --seed 0
"""
import argparse
import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
D2MOE_DIR = Path(os.environ.get("D2MOE_DIR", SCRIPT_DIR.parents[1] / "third_party" / "D2MoE"))
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
sys.path.insert(0, str(D2MOE_DIR))
import qwen15_config as QC  # noqa: E402
import qwen15_merge_eval as QM  # noqa: E402
import phase1_sib200_eval as S  # noqa: E402 -- demos / prompt / candidates, unchanged
from phase1_xnli_eval import score_batch  # noqa: E402 -- unchanged
from sib200_clean_eval import clean_keep, AUDIT, LANGS  # noqa: E402 -- same 98-item keep set

OUT_DIR = QC.RESULTS_ROOT.parent / "sib200_clean"
BATCH_SIZE = 4  # same as the DeepSeek clean run
DEEPSEEK_CLEAN_DIR = SCRIPT_DIR.parent / "analysis" / "clean_eval_0916" / "sib200"


def tokenizer_record(tokenizer, demos):
    """Everything about Qwen input processing that differs from DeepSeek, measured
    on the actual tokenizer rather than asserted."""
    choice_ids = S._choice_token_ids(tokenizer)  # raises if any candidate is not a single token
    probe = tokenizer("Answer:", add_special_tokens=True)["input_ids"]
    probe_ans = tokenizer("Answer: Science", add_special_tokens=True)["input_ids"]
    rec = {
        "tokenizer_class": type(tokenizer).__name__, "use_fast": False,
        "bos_token": tokenizer.bos_token, "eos_token": tokenizer.eos_token, "pad_token": tokenizer.pad_token,
        "adds_bos_by_default": bool(probe and probe[0] == tokenizer.bos_token_id and tokenizer.bos_token_id is not None),
        "prompt_prefix_token": None,
        "padding_side_used_by_score_batch": "right",
        "choice_words": S.CHOICE_WORDS, "choice_token_ids": choice_ids,
        "answer_colon_then_candidate_is_two_tokens": probe_ans[:-1] == probe and probe_ans[-1] == choice_ids[0],
        "note": ("Qwen2Tokenizer adds no BOS; DeepSeek's LlamaTokenizerFast adds <｜begin▁of▁sentence｜>. "
                 "No prefix token is inserted for Qwen (same as the Qwen FLORES primary bpb metric). "
                 "All other prompt text, demonstrations and candidates are byte-identical to the DeepSeek run."),
    }
    if not rec["answer_colon_then_candidate_is_two_tokens"]:
        raise RuntimeError(f"'Answer: Science' does not tokenize as 'Answer:' + ' Science' for Qwen: {probe_ans} vs {probe}")
    demo_block = S.build_prompt(demos, "X")
    rec["demo_prompt_tokens_with_placeholder_text"] = len(tokenizer(demo_block)["input_ids"])
    return rec


def evaluate(model, tokenizer, key, condition, seed, is_baseline, rt, source):
    keep = clean_keep()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{key}.json"
    pred_path = OUT_DIR / f"{key}_predictions.jsonl"
    if out_path.exists() or pred_path.exists():
        raise FileExistsError(f"{out_path} or {pred_path} exists -- refusing to overwrite")

    from datasets import load_dataset
    demos = S.build_demonstrations(seed=S.DEMO_SEED_DEFAULT, per_label=S.DEMOS_PER_LABEL_DEFAULT)
    tok_rec = tokenizer_record(tokenizer, demos)
    choice_ids = tok_rec["choice_token_ids"]
    max_pos = int(getattr(model.config, "max_position_embeddings", 0) or 0)

    results, per_item = {}, []
    for lang in LANGS:
        t0 = time.time()
        ds = load_dataset(S.SIB_DATASET, lang, split="test")
        if len(ds) != 204:
            raise RuntimeError(f"{lang}: test split has {len(ds)} items, expected 204")
        pred, gold, idx, plen = [], [], [], []
        for i in range(0, len(ds), BATCH_SIZE):
            chunk = ds[i:i + BATCH_SIZE]
            prompts = [S.build_prompt(demos, t) for t in chunk["text"]]
            plen += [len(tokenizer(p)["input_ids"]) for p in prompts]
            lp = score_batch(model, tokenizer, prompts, choice_ids)
            if not torch.isfinite(lp).all():
                raise RuntimeError(f"non-finite logprob in {lang} at item {i} -- refusing to continue")
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
        if max_pos and max(plen) > max_pos:
            raise RuntimeError(f"{lang}: longest prompt {max(plen)} tokens exceeds max_position_embeddings {max_pos}")

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
        if len(cleansel) != 98:
            raise RuntimeError(f"{lang}: clean subset has {len(cleansel)} items, expected 98")
        full, cl = summarize(allsel), summarize(cleansel)
        results[lang] = {"full": full, "clean": cl, "reproduction_check": None,
                         "prompt_tokens": {"min": min(plen), "max": max(plen), "mean": sum(plen) / len(plen)},
                         "seconds": secs, "items_per_second": len(gold) / secs}
        print(f"[qwen-sib-clean] {key}/{lang}: full={full['acc']:.4f} (n={full['n']})  clean={cl['acc']:.4f} "
              f"(n={cl['n']}) top={cl['top_label']} share={cl['top_label_share']:.3f}  "
              f"prompt_tokens max={max(plen)}  [{secs:.0f}s, {len(gold)/secs:.2f} it/s]", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    rt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rt["peak_gpu_bytes"] = {str(d): torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())}
    payload = {
        "task": "sib200", "model": key, "model_id": QC.MODEL_ID, "model_revision": QC.MODEL_REVISION,
        "condition": ("baseline" if is_baseline else condition), "seed": (None if is_baseline else seed),
        "model_source": source,
        "protocol": "sib200_natural_continuation_v1 (unchanged from phase1_sib200_eval.py)",
        "num_fewshot": len(S.CATEGORY_ORDER) * S.DEMOS_PER_LABEL_DEFAULT,
        "demo_seed": S.DEMO_SEED_DEFAULT, "batch_size": BATCH_SIZE,
        "demonstrations": [{"text": ex["text"], "category": ex["category"]} for ex in demos],
        "clean_subset_source": str(AUDIT.relative_to(SCRIPT_DIR.parent)),
        "clean_variant": "primary (identical to strict for SIB; keep set identical across all 5 languages)",
        "evaluation_scope": ("TEST-ITEM clean: the 98 test items per language that the audit marks unexposed. "
                             "The 14 fixed English demonstrations are unchanged (6 of them are "
                             "calibration-exposed per the audit), so calibration exposure is NOT fully removed."),
        "qwen_input_processing": tok_rec,
        "comparison_note": ("Comparable to DeepSeek only on the same 98-item clean subset "
                            f"({DEEPSEEK_CLEAN_DIR.relative_to(SCRIPT_DIR.parent)}); Qwen has no earlier "
                            "full-204 SIB result, so reproduction_check is null."),
        "results": results, "runtime": rt, "env": QC.environment_record(),
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    with open(pred_path, "w") as f:
        for r in per_item:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(out_path)
    print(f"[qwen-sib-clean] wrote {out_path} and {pred_path} ({len(per_item)} items)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--condition", choices=QC.MAIN_CONDITIONS)
    ap.add_argument("--seed", type=int, choices=list(QC.SEEDS))
    args = ap.parse_args()
    if not args.baseline and (args.condition is None or args.seed is None):
        ap.error("--baseline, or --condition and --seed")
    QC.check_env()
    key = "baseline" if args.baseline else f"{args.condition}_seed{args.seed}"
    if (OUT_DIR / f"{key}.json").exists():
        raise FileExistsError(f"{OUT_DIR / (key + '.json')} exists -- refusing to overwrite")

    rt = {"model": key, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t = time.time()
    tokenizer = QC.load_tokenizer()
    model = QC.load_model()  # 2 GPUs, no offload (asserted), config + param count asserted
    rt["seconds_load"] = time.time() - t
    rt["device_map_unique"] = sorted({str(v) for v in (getattr(model, "hf_device_map", {}) or {}).values()})
    if args.baseline:
        source = {"baseline": True, "params": QC.count_params(model)}
    else:
        od = QM.out_dir_for(args.condition, args.seed, args.condition, args.seed, False)
        t = time.time()
        manifest = QM.load_saved_compressed(model, od)  # digest + param-count checks inside
        rt["seconds_reload_compressed"] = time.time() - t
        source = {"baseline": False, "source": "saved", "compressed_model_dir": str(od / "compressed_model"),
                  "manifest_created_utc": manifest.get("created_utc"), "params": manifest["params"],
                  "scale_condition": manifest["scale_condition"], "scale_seed": manifest["scale_seed"]}
        model.eval()
        gc.collect(); torch.cuda.empty_cache()
    print(f"[qwen-sib-clean] {key}: model ready ({source})", flush=True)
    evaluate(model, tokenizer, key, args.condition, args.seed, args.baseline, rt, source)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
