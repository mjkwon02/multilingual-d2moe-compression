"""Qwen1.5-MoE-A2.7B: element-wise gradient Fisher for every MoE layer, one
calibration condition/seed at a time -- the Qwen counterpart of phase1_fisher.py.

Kept identical to phase1_fisher.py:
  - calibration text: phase1_fisher.build_samples(condition, 64, 512, seed),
    imported unchanged (same windows, same calib_text_hash as the DeepSeek logs);
  - tokenisation call: tokenizer(text, truncation=True, max_length=512), batch 1;
  - one layer at a time, only that layer's ROUTED expert weights require grad
    (shared expert, shared_expert_gate, router, attention all frozen);
  - model.train(), loss = the model's own LM loss with labels=input_ids;
  - F = mean over samples of grad**2, accumulated in fp32 on CPU, saved bf16,
    keys "mlp.experts.<e>.<proj>";
  - zero-routed experts -> F = 0 ("no evidence"), 0 < hits < 5 flagged, kept.

Qwen-specific differences (all recorded in the run metadata):
  - hits are counted from routed rows (qwen15_moe_adapter.RoutingCounter), not
    from `grad is not None`: Qwen's block calls every expert, so unrouted
    experts get an all-zero gradient rather than None. The Fisher VALUE is the
    same either way (accumulated zeros == the F=0 fallback); only the flag
    bookkeeping needed the change. A dead expert with a non-zero accumulated
    Fisher is treated as a counter bug and stops the run.
  - loss: Qwen2MoeForCausalLM adds the router load-balancing loss only when
    output_router_logits=True; the checkpoint config has it False, so the loss
    is plain cross-entropy. DeepSeek's remote code in train mode adds its aux
    loss (AddAuxiliaryLoss) automatically. Left as each model's default; the
    size of that effect on expert-weight gradients is not measured.
  - use_cache=False (memory only; no numerical effect).
  - per-layer files only; no assembled fisher_processed.pt (qwen15_merge_eval.py
    reads the per-layer files), which halves disk use and avoids loading all
    layers into RAM at once.

Stops (non-zero exit, nothing written for the failing layer) on: non-finite
loss, non-finite accumulated Fisher, offloaded expert params, counter
inconsistency. Completed layers are kept and skipped on a re-run.

Usage:
    python -u qwen15_fisher.py --condition english_only --seed 0
"""
import argparse
import hashlib
import sys
import traceback
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import qwen15_config as QC  # noqa: E402
import qwen15_moe_adapter as QA  # noqa: E402
import phase1_fisher  # noqa: E402 -- build_samples() reused unchanged


def freeze_all_but_layer(model, layer_idx):
    for p in model.parameters():
        p.requires_grad_(False)
    block = QA.get_block(model, layer_idx)
    for p in block.experts.parameters():
        p.requires_grad_(True)
    return block


def elementwise_fisher_for_layer(model, block, tokenizer, texts, seqlen, embed_device, layer_idx):
    accum = {}
    counter = QA.RoutingCounter(block)
    n_used, losses, realized = 0, [], []
    model.train()
    try:
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seqlen)
            input_ids = enc["input_ids"].to(embed_device)
            if input_ids.shape[1] < 2:
                continue
            model.zero_grad(set_to_none=True)
            counter.start_sample()
            out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            counter.end_sample()
            if not bool(torch.isfinite(out.loss).item()):
                raise RuntimeError(f"layer {layer_idx}: non-finite loss {out.loss.item()} at sample {n_used}")
            out.loss.backward()
            for e_idx, expert in enumerate(block.experts):
                for proj_name in QA.PROJ_NAMES:
                    w = getattr(expert, proj_name).weight
                    if w.grad is None:
                        continue
                    key = QA.fisher_key(e_idx, proj_name)
                    g2 = w.grad.detach().float().pow(2).cpu()
                    if key in accum:
                        accum[key] += g2
                    else:
                        accum[key] = g2
            model.zero_grad(set_to_none=True)
            losses.append(float(out.loss.item()))
            realized.append(int(input_ids.shape[1]))
            n_used += 1
    finally:
        counter.remove()
    if n_used == 0:
        raise RuntimeError("no usable samples (all shorter than 2 tokens)")

    flags = []
    dead = [e for e, h in enumerate(counter.hit_samples) if h == 0]
    if dead:
        offloaded = sorted({e for e in dead for n in QA.PROJ_NAMES
                            if getattr(block.experts[e], n).weight.device.type != "cuda"})
        if offloaded:
            raise RuntimeError(f"dead experts {offloaded[:10]} are not on a CUDA device -- offload, not routing sparsity.")
        for e in dead:
            for proj_name in QA.PROJ_NAMES:
                key = QA.fisher_key(e, proj_name)
                if key in accum and float(accum[key].abs().max().item()) != 0.0:
                    raise RuntimeError(f"expert {e} received 0 routed tokens but has non-zero Fisher in {key} "
                                       "-- routing counter and gradients disagree; stopping.")
                w = getattr(block.experts[e], proj_name).weight
                accum[key] = torch.zeros_like(w, dtype=torch.float32, device="cpu")
            flags.append({"layer": layer_idx, "expert": e, "hit_count": 0, "kind": "dead_fallback_F0", "source": "fisher"})
            print(f"[qwen15-fisher] layer {layer_idx} expert {e}: 0/{n_used} samples routed, F=0", flush=True)
    for e, h in enumerate(counter.hit_samples):
        if 0 < h < QC.LOW_HIT_THRESHOLD:
            flags.append({"layer": layer_idx, "expert": e, "hit_count": h, "kind": "low_hit_flagged", "source": "fisher"})

    missing = [QA.fisher_key(e, p) for e in range(len(block.experts)) for p in QA.PROJ_NAMES
               if QA.fisher_key(e, p) not in accum]
    if missing:
        raise RuntimeError(f"{len(missing)} expert weights never produced a gradient, e.g. {missing[:5]}")
    nonfinite = [k for k, v in accum.items() if not bool(torch.isfinite(v).all().item())]
    if nonfinite:
        raise RuntimeError(f"non-finite accumulated Fisher in {len(nonfinite)} tensors, e.g. {nonfinite[:5]}")

    out = {key: val.div(n_used).to(torch.bfloat16) for key, val in accum.items()}
    del accum
    n_zero = sum(int((v == 0).sum().item()) for v in out.values())
    n_all = sum(v.numel() for v in out.values())
    stats = {
        "n_used": n_used, "losses": losses, "realized_tokens": realized,
        "hit_samples": counter.hit_samples, "routed_tokens": counter.tokens,
        "dead_experts": dead, "fisher_zero_fraction_bf16": n_zero / n_all,
        "fisher_max": max(float(v.max().item()) for v in out.values()),
    }
    return out, flags, stats


def _merge_flags(flags_path, layer_idx, flags):
    import json
    existing = json.loads(flags_path.read_text()) if flags_path.exists() else {}
    existing[str(layer_idx)] = [f for f in existing.get(str(layer_idx), []) if f.get("source") != "fisher"] + flags
    flags_path.write_text(json.dumps(existing, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=QC.CONDITIONS)
    ap.add_argument("--seed", type=int, required=True, choices=list(QC.SEEDS))
    args = ap.parse_args()
    QC.check_env()

    out_dir = QC.condition_dir(args.condition, args.seed)
    layer_dir = out_dir / "fisher_layers"
    layer_dir.mkdir(parents=True, exist_ok=True)
    flags_path = out_dir / "routing_coverage_flags.json"
    rec = QC.ResourceRecorder()
    meta = {"stage": "fisher", "condition": args.condition, "seed": args.seed, "n_samples": QC.N_SAMPLES,
            "seqlen": QC.SEQLEN, "env": QC.environment_record(), "layers": {}, "status": "running"}
    try:
        st = rec.begin("load")
        tokenizer = QC.load_tokenizer()
        model = QC.load_model()
        embed_device = model.get_input_embeddings().weight.device
        meta["device_map"] = QC.device_map_summary(model)
        rec.end(st)

        texts = phase1_fisher.build_samples(args.condition, QC.N_SAMPLES, QC.SEQLEN, args.seed)
        meta["calib_text_hash"] = hashlib.md5("".join(texts).encode()).hexdigest()[:12]
        meta["n_texts"] = len(texts)
        meta["window_chars"] = [len(t) for t in texts]
        print(f"[qwen15-fisher] condition={args.condition} seed={args.seed} {len(texts)} windows "
              f"calib_text_hash={meta['calib_text_hash']} layers={QC.MOE_LAYERS}", flush=True)

        for layer_idx in QC.MOE_LAYERS:
            layer_path = layer_dir / f"layer_{layer_idx}.pt"
            if layer_path.exists():
                print(f"[qwen15-fisher] layer {layer_idx}: already done, skipping", flush=True)
                continue
            st = rec.begin(f"fisher_layer_{layer_idx}")
            block = freeze_all_but_layer(model, layer_idx)
            layer_fisher, flags, stats = elementwise_fisher_for_layer(
                model, block, tokenizer, texts, QC.SEQLEN, embed_device, layer_idx)
            tmp = layer_path.with_suffix(".tmp")
            torch.save(layer_fisher, tmp)
            tmp.rename(layer_path)
            if flags:
                _merge_flags(flags_path, layer_idx, flags)
            del layer_fisher
            torch.cuda.empty_cache()
            s = rec.end(st, n_flags=len(flags))
            meta["layers"][str(layer_idx)] = {**stats, "resources": s}
            print(f"[qwen15-fisher] layer {layer_idx}: done in {s['seconds']}s dead={stats['dead_experts']} "
                  f"loss_mean={sum(stats['losses'])/len(stats['losses']):.4f}", flush=True)
        for p in model.parameters():
            p.requires_grad_(False)
        meta["status"] = "ok"
    except BaseException as exc:
        meta["status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["traceback"] = traceback.format_exc()
        raise
    finally:
        rec.close()
        meta["resources"] = rec.summary()
        path = QC.write_json_versioned(out_dir / "fisher_run_meta.json", meta)
        print(f"[qwen15-fisher] wrote {path} (status={meta['status']})", flush=True)


if __name__ == "__main__":
    main()
