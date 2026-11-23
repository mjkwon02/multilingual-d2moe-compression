"""Qwen1.5-MoE-A2.7B: D2-MoE compression (Fisher-weighted base merge +
activation-aware whitened delta SVD, structured pruning OFF) and FLORES-200
bits-per-byte, plus save / reload of the compressed model -- the Qwen
counterpart of phase1_merge_eval.py.

Modes
  --baseline                              uncompressed model, FLORES bpb
  --condition C --seed S                  main experiment: Fisher(C,S) +
                                          whitening(C,S) (= --scale-condition C)
  --condition C --seed S --scale-condition X [--scale-seed Y]
                                          Fisher x whitening 2x2 cell
  --condition C --seed S --plain-svd      whitening OFF (not the main experiment)
  --save-compressed                       also write compressed_model/ (per-layer
                                          safetensors of Wmean + delta factors +
                                          manifest); non-expert weights are the
                                          pinned original checkpoint
  --reload-eval                           rebuild from compressed_model/ in a fresh
                                          process, verify tensor digests, re-run
                                          FLORES and compare with eval_ppl.json

FLORES evaluation = phase1_merge_eval's set (first 60 devtest sentences, 5
languages) through data_utils.evaluate_sentences at batch 8, unchanged. Qwen's
tokenizer adds no BOS, so each sentence's first token is context only while its
bytes stay in the denominator (DeepSeek's tokenizer adds BOS, so there every
sentence token is scored). The primary number keeps the unchanged code path;
a second number with <|endoftext|> (eos id == config.bos_token_id) prepended --
the lm-eval rolling-loglikelihood convention -- is stored as a sensitivity
column, never substituted silently.

No OOM fallback to a smaller batch: an OOM stops the run.

Output never overwrites an existing result file.
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
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import qwen15_config as QC  # noqa: E402
import qwen15_moe_adapter as QA  # noqa: E402
import qwen15_freq  # noqa: E402
import data_utils  # noqa: E402
from phase1_merge_eval import EVAL_LANGS, EVAL_SOURCE, EVAL_SPLIT, EVAL_PREFIX, EVAL_N_SENTENCES  # noqa: E402


# ----------------------------------------------------------------- evaluation
@torch.no_grad()
def _batch_nll_prefixed(model, tokenizer, sentences, prefix_id):
    """data_utils.batch_sentence_nll with an explicit prefix token: same right
    padding, same shifted CE, same pad masking."""
    device = next(model.parameters()).device
    ids = [[prefix_id] + tokenizer(s)["input_ids"] for s in sentences]
    L = max(len(x) for x in ids)
    pad = tokenizer.pad_token_id
    input_ids = torch.tensor([x + [pad] * (L - len(x)) for x in ids], device=device)
    attention_mask = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids], device=device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    label_mask = attention_mask[:, 1:].bool()
    token_nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                reduction="none").reshape(labels.shape)
    token_nll = token_nll.masked_fill(~label_mask, 0.0)
    return token_nll.sum(dim=1).tolist(), label_mask.sum(dim=1).tolist()


def _evaluate_prefixed(model, tokenizer, sentences, batch_size, prefix_id):
    nll, ntok = [], []
    for i in range(0, len(sentences), batch_size):
        a, b = _batch_nll_prefixed(model, tokenizer, sentences[i:i + batch_size], prefix_id)
        nll.extend(a)
        ntok.extend(b)
    return nll, ntok


def eval_flores_bpb(model, tokenizer, batch_size=QC.EVAL_BATCH_SIZE):
    prefix_id = tokenizer.eos_token_id
    if prefix_id != model.config.bos_token_id:
        raise RuntimeError(f"eos id {prefix_id} != config.bos_token_id {model.config.bos_token_id}")
    model.eval()
    results = {}
    for lang_code, lang_name in EVAL_LANGS.items():
        sentences = data_utils.load_flores_sentences(EVAL_SOURCE, EVAL_SPLIT, EVAL_PREFIX, lang_code, EVAL_N_SENTENCES)
        total_bytes = sum(len(s.encode("utf-8")) for s in sentences)
        nll, ntok = data_utils.evaluate_sentences(model, tokenizer, sentences, batch_size=batch_size)
        bpb = (sum(nll) / math.log(2)) / total_bytes
        nll_b, ntok_b = _evaluate_prefixed(model, tokenizer, sentences, batch_size, prefix_id)
        bpb_b = (sum(nll_b) / math.log(2)) / total_bytes
        results[lang_code] = {
            "name": lang_name, "ppl": data_utils.corpus_ppl(nll, ntok), "n_sentences": len(sentences),
            "n_tokens": sum(ntok), "n_bytes": total_bytes, "bits_per_byte": bpb,
            "sensitivity_eos_prefixed": {"bits_per_byte": bpb_b, "n_tokens": sum(ntok_b), "prefix_id": prefix_id},
        }
        print(f"[qwen15-eval] {lang_name}: bpb={bpb:.6f} (eos-prefixed {bpb_b:.6f}) over {total_bytes} bytes, "
              f"{sum(ntok)} scored tokens", flush=True)
    bad = [k for k, r in results.items()
           if not (math.isfinite(r["bits_per_byte"]) and math.isfinite(r["sensitivity_eos_prefixed"]["bits_per_byte"]))]
    if bad:
        raise RuntimeError(f"non-finite bpb for {bad} -- refusing to write a result")
    return results


def eval_meta(tokenizer):
    return {"eval_source": EVAL_SOURCE, "eval_split": EVAL_SPLIT, "eval_n_sentences": EVAL_N_SENTENCES,
            "batch_size": QC.EVAL_BATCH_SIZE, "tokenizer_bos": tokenizer.bos_token,
            "primary": "data_utils.evaluate_sentences unchanged (tokenizer default special tokens: none for Qwen)",
            "sensitivity": "same, with eos (<|endoftext|>) prepended to every sentence"}


# ------------------------------------------------------------ model building
def out_dir_for(condition, seed, scale_condition, scale_seed, plain_svd):
    base = QC.condition_dir(condition, seed)
    return base / "plain_svd" if plain_svd else base / f"scale_{scale_condition}_seed{scale_seed}"


def check_inputs(condition, seed, scale_condition, scale_seed, plain_svd):
    freq = qwen15_freq.freq_path(condition, seed)
    missing = [] if freq.exists() else [str(freq)]
    fdir = QC.condition_dir(condition, seed) / "fisher_layers"
    missing += [str(fdir / f"layer_{i}.pt") for i in QC.MOE_LAYERS if not (fdir / f"layer_{i}.pt").exists()]
    if not plain_svd:
        sdir = QC.condition_dir(scale_condition, scale_seed) / "svd_scale_layers"
        missing += [str(sdir / f"layer_{i}.pt") for i in QC.MOE_LAYERS if not (sdir / f"layer_{i}.pt").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} input(s) missing, e.g. {missing[:3]} -- order is freq -> fisher -> "
                                "svd_scale -> merge (0802 prompt §3.20-13)")


def compress_model(model, condition, seed, scale_condition, scale_seed, plain_svd, rec=None):
    expert_freq = json.loads(qwen15_freq.freq_path(condition, seed).read_text())
    fdir = QC.condition_dir(condition, seed) / "fisher_layers"
    sdir = None if plain_svd else QC.condition_dir(scale_condition, scale_seed) / "svd_scale_layers"
    per_layer = {}
    for i in QC.MOE_LAYERS:
        t0 = time.time()
        hessian = QA.load_layer_stats(fdir, i)
        svd_scale = None if sdir is None else QA.load_layer_stats(sdir, i)
        block = QA.get_block(model, i)
        stats = QA.compress_block(block, model.config, hessian, expert_freq[str(i)], svd_scale, QC.DELTA_RATIO)
        del hessian, svd_scale
        gc.collect()
        torch.cuda.empty_cache()
        stats["seconds"] = round(time.time() - t0, 2)
        stats["device"] = str(block.gate.weight.device)
        per_layer[str(i)] = stats
        print(f"[qwen15-merge] layer {i}: merged in {stats['seconds']}s rank={stats['delta_rank']} "
              f"routed params {stats['routed_expert_params_before']} -> {stats['routed_expert_params_after']}",
              flush=True)
    return per_layer


def param_accounting(model, total_before, per_layer):
    total_after = QC.count_params(model)
    analytic = QA.analytic_param_accounting(model.config, len(QC.MOE_LAYERS), QC.DELTA_RATIO)
    routed_before = sum(v["routed_expert_params_before"] for v in per_layer.values())
    routed_after = sum(v["routed_expert_params_after"] for v in per_layer.values())
    if (routed_before, routed_after) != (analytic["routed_before"], analytic["routed_after"]):
        raise RuntimeError(f"measured routed params ({routed_before}, {routed_after}) != analytic "
                           f"({analytic['routed_before']}, {analytic['routed_after']})")
    return {"total_params_before": total_before, "total_params_after": total_after,
            "total_reduction": total_before - total_after,
            "total_reduction_pct": 100.0 * (total_before - total_after) / total_before,
            "routed_expert_params_before": routed_before, "routed_expert_params_after": routed_after,
            "routed_expert_reduction_pct": analytic["routed_reduction_pct"],
            "delta_ratio": QC.DELTA_RATIO, "delta_rank": analytic["rank"],
            "note": "delta_ratio sets the per-expert low-rank budget; the model-wide reduction is total_reduction_pct"}


def save_compressed(model, out_dir, manifest):
    cdir = out_dir / "compressed_model"
    if (cdir / "manifest.json").exists():
        raise FileExistsError(f"{cdir}/manifest.json exists -- refusing to overwrite a saved model")
    cdir.mkdir(parents=True, exist_ok=True)
    files = {}
    for i in QC.MOE_LAYERS:
        files[str(i)] = QA.save_compressed_block(QA.get_block(model, i), cdir / f"layer_{i}.safetensors",
                                                 {"layer": i, "model_revision": QC.MODEL_REVISION})
    manifest = {**manifest, "layers": files,
                "total_bytes": sum(v["bytes"] for v in files.values()),
                "load_recipe": "qwen15_merge_eval.load_saved_compressed(): original pinned checkpoint + per-layer "
                               "Wmean/delta tensors installed as block.experts (meanW_deltaUV)"}
    QC.write_json_new(cdir / "manifest.json", manifest)
    return manifest


def load_saved_compressed(model, out_dir):
    cdir = out_dir / "compressed_model"
    manifest = json.loads((cdir / "manifest.json").read_text())
    if manifest["model_revision"] != QC.MODEL_REVISION:
        raise RuntimeError("saved model was built from a different revision")
    mism = {}
    for i in QC.MOE_LAYERS:
        digest = QA.load_compressed_block(QA.get_block(model, i), model.config, cdir / f"layer_{i}.safetensors",
                                          manifest["delta_ratio"])
        if digest != manifest["layers"][str(i)]["sha256_tensors"]:
            mism[str(i)] = digest
    if mism:
        raise RuntimeError(f"tensor digests differ from manifest for layers {sorted(mism)}")
    n = QC.count_params(model)
    if n != manifest["params"]["total_params_after"]:
        raise RuntimeError(f"reloaded total params {n} != manifest {manifest['params']['total_params_after']}")
    return manifest


def build_model_for_eval(args, rec=None):
    """Shared by qwen15_downstream_eval.py: baseline, in-process merge, or reload."""
    tokenizer = QC.load_tokenizer()
    model = QC.load_model()
    if args.baseline:
        return model, tokenizer, {"baseline": True}
    od = out_dir_for(args.condition, args.seed, args.scale_condition, args.scale_seed, args.plain_svd)
    if getattr(args, "from_saved", False):
        load_saved_compressed(model, od)
        return model, tokenizer, {"baseline": False, "source": "saved", "manifest": str(od / "compressed_model")}
    check_inputs(args.condition, args.seed, args.scale_condition, args.scale_seed, args.plain_svd)
    compress_model(model, args.condition, args.seed, args.scale_condition, args.scale_seed, args.plain_svd)
    return model, tokenizer, {"baseline": False, "source": "merged_in_process"}


def resolve_args(ap, args):
    if not args.baseline and not args.condition:
        ap.error("--baseline or --condition is required")
    if args.baseline:
        return
    if args.seed is None:
        ap.error("--seed is required with --condition")
    if args.plain_svd and args.scale_condition:
        ap.error("--plain-svd and --scale-condition are exclusive")
    if not args.plain_svd:
        args.scale_condition = args.scale_condition or args.condition
        args.scale_seed = args.seed if args.scale_seed is None else args.scale_seed


def add_model_args(ap):
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--condition", choices=QC.CONDITIONS)
    ap.add_argument("--seed", type=int, choices=list(QC.SEEDS))
    ap.add_argument("--scale-condition", choices=QC.CONDITIONS, default=None)
    ap.add_argument("--scale-seed", type=int, choices=list(QC.SEEDS), default=None)
    ap.add_argument("--plain-svd", action="store_true")


def _baseline_bpb():
    p = QC.RESULTS_ROOT / "baseline" / "eval_ppl.json"
    return json.loads(p.read_text()) if p.exists() else None


def _delta_pct(results, base):
    if base is None:
        return None
    return {k: {"delta_bpb_pct": 100.0 * (r["bits_per_byte"] / base[k]["bits_per_byte"] - 1),
                "delta_bpb_pct_eos_prefixed": 100.0 * (r["sensitivity_eos_prefixed"]["bits_per_byte"]
                                                       / base[k]["sensitivity_eos_prefixed"]["bits_per_byte"] - 1)}
            for k, r in results.items()}


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    ap.add_argument("--save-compressed", action="store_true")
    ap.add_argument("--reload-eval", action="store_true")
    args = ap.parse_args()
    resolve_args(ap, args)
    QC.check_env()

    if args.baseline:
        out_dir = QC.RESULTS_ROOT / "baseline"
        out_name = "eval_ppl.json"
    else:
        out_dir = out_dir_for(args.condition, args.seed, args.scale_condition, args.scale_seed, args.plain_svd)
        out_name = "eval_ppl_reloaded.json" if args.reload_eval else "eval_ppl.json"
    if (out_dir / out_name).exists():
        raise FileExistsError(f"{out_dir / out_name} exists -- refusing to overwrite")
    if not args.baseline and not args.reload_eval:
        check_inputs(args.condition, args.seed, args.scale_condition, args.scale_seed, args.plain_svd)

    rec = QC.ResourceRecorder()
    run = {"mode": "baseline" if args.baseline else ("reload_eval" if args.reload_eval else "merge_eval"),
           "condition": args.condition, "seed": args.seed, "scale_condition": args.scale_condition,
           "scale_seed": args.scale_seed, "plain_svd": args.plain_svd, "delta_ratio": QC.DELTA_RATIO,
           "share_ratio": QC.SHARE_RATIO, "merge_method": QC.MERGE_METHOD, "structured_pruning": "off",
           "env": QC.environment_record(), "status": "running"}
    try:
        st = rec.begin("load")
        tokenizer = QC.load_tokenizer()
        model = QC.load_model()
        run["device_map"] = QC.device_map_summary(model)
        total_before = QC.count_params(model)
        rec.end(st)

        if args.reload_eval:
            st = rec.begin("reload_compressed")
            manifest = load_saved_compressed(model, out_dir)
            rec.end(st)
            run["params"] = manifest["params"]
        elif not args.baseline:
            st = rec.begin("merge")
            per_layer = compress_model(model, args.condition, args.seed, args.scale_condition, args.scale_seed,
                                       args.plain_svd)
            gc.collect()
            torch.cuda.empty_cache()
            rec.end(st)
            run["per_layer"] = per_layer
            run["params"] = param_accounting(model, total_before, per_layer)
            print(f"[qwen15-merge] params {run['params']['total_params_before']} -> "
                  f"{run['params']['total_params_after']} ({run['params']['total_reduction_pct']:.3f}% of the model; "
                  f"{run['params']['routed_expert_reduction_pct']:.3f}% of routed-expert params)", flush=True)
        else:
            run["params"] = {"total_params_before": total_before}

        st = rec.begin("eval_flores")
        results = eval_flores_bpb(model, tokenizer)
        rec.end(st)

        payload = {"_meta": {**eval_meta(tokenizer), "run": {k: run[k] for k in run if k not in ("per_layer", "env")},
                             "model_id": QC.MODEL_ID, "model_revision": QC.MODEL_REVISION}, **results}
        base = None if args.baseline else _baseline_bpb()
        if base is not None:
            payload["_delta_vs_baseline"] = _delta_pct(results, base)
        QC.write_json_new(out_dir / out_name, payload)
        print(f"[qwen15-eval] wrote {out_dir / out_name}", flush=True)

        if args.reload_eval:
            ref = json.loads((out_dir / "eval_ppl.json").read_text())
            cmp = {k: {"in_process": ref[k]["bits_per_byte"], "reloaded": results[k]["bits_per_byte"],
                       "identical": ref[k]["bits_per_byte"] == results[k]["bits_per_byte"],
                       "abs_diff": abs(ref[k]["bits_per_byte"] - results[k]["bits_per_byte"])} for k in results}
            run["reload_vs_in_process"] = cmp
            QC.write_json_versioned(out_dir / "reload_check.json", cmp)
            print(f"[qwen15-eval] reload identical for all languages: {all(v['identical'] for v in cmp.values())}",
                  flush=True)

        if args.save_compressed and not args.baseline and not args.reload_eval:
            st = rec.begin("save_compressed")
            man = {"model_id": QC.MODEL_ID, "model_revision": QC.MODEL_REVISION, "condition": args.condition,
                   "seed": args.seed, "scale_condition": args.scale_condition, "scale_seed": args.scale_seed,
                   "plain_svd": args.plain_svd, "delta_ratio": QC.DELTA_RATIO, "params": run["params"],
                   "dtype": "bfloat16", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            saved = save_compressed(model, out_dir, man)
            rec.end(st, bytes_written=saved["total_bytes"])
            print(f"[qwen15-merge] saved compressed model: {saved['total_bytes']/1e9:.2f} GB", flush=True)
        run["status"] = "ok"
    except BaseException as exc:
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        run["traceback"] = traceback.format_exc()
        raise
    finally:
        rec.close()
        run["resources"] = rec.summary()
        out_dir.mkdir(parents=True, exist_ok=True)
        p = QC.write_json_versioned(out_dir / f"{run['mode']}_run_meta.json", run)
        print(f"[qwen15-merge] wrote {p} (status={run['status']})", flush=True)


if __name__ == "__main__":
    main()
