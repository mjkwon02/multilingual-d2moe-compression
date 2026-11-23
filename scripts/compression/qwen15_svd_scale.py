"""Qwen1.5-MoE-A2.7B: activation-aware whitening statistics (D2-MoE / SVD-LLM
"svd_scale") for every MoE layer -- the Qwen counterpart of phase1_svd_scale.py.

Kept identical to phase1_svd_scale.py:
  - same calibration windows (phase1_fisher.build_samples) and tokenisation;
  - eval-mode no-grad standard forward model(input_ids=...), batch 1;
  - per routed expert and projection, S = sum over routed tokens of x x^T of
    that Linear's INPUT (gate/up: the 2048-d MoE input; down: the 1408-d
    silu(gate)*up activation), accumulated in fp64 on CPU;
  - processing: symmetrise, eigh, if min eigenvalue <= 0 clamp ALL eigenvalues
    to >= 1e-5 and rebuild, Cholesky (fp64, on GPU), saved bf16, keys
    "mlp.experts.<e>.<proj>";
  - expert never routed -> identity matrix (= plain SVD for that expert);
  - low-hit flag rule: the DeepSeek script increments its hit counter once per
    PROJECTION hook call (3 per routed sample) and flags counter < 5; the same
    counter is reproduced here so the flag means the same thing, and the plain
    "samples routed" count is recorded next to it.

Qwen-specific:
  - Qwen's block calls every expert, including with 0 rows; the hook ignores
    empty inputs (adding a zero matrix would change nothing numerically, but
    would make every expert look "hit" and suppress the identity fallback that
    DeepSeek's eval-mode moe_infer gives unrouted experts).
  - the eigen-processing is instrumented (min/max eigenvalue, whether the clamp
    fired, how many eigenvalues it raised) with statements identical to
    phase1_svd_scale.process_scaling_matrix(); stats land in
    svd_scale_layers/layer_<i>.json.

Usage:
    python -u qwen15_svd_scale.py --condition english_only --seed 0
"""
import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import qwen15_config as QC  # noqa: E402
import qwen15_moe_adapter as QA  # noqa: E402
import phase1_fisher  # noqa: E402 -- build_samples() reused unchanged

CLAMP_MIN = 1e-5  # phase1_svd_scale.process_scaling_matrix / get_scale.py


def process_scaling_matrix_instrumented(raw_matrix, device):
    """Same statements as phase1_svd_scale.process_scaling_matrix (lines
    66-72); only the .item() reads for the record are added."""
    raw_matrix = raw_matrix.to(device)
    matrix = (raw_matrix + raw_matrix.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    min_eig = eigenvalues.min().item()
    stats = {"min_eig": min_eig, "max_eig": eigenvalues.max().item(), "clamped": False, "n_eig_raised": 0}
    if min_eig <= 0:
        stats["clamped"] = True
        stats["n_eig_raised"] = int((eigenvalues < CLAMP_MIN).sum().item())
        eigenvalues = torch.clamp(eigenvalues, min=CLAMP_MIN)
        matrix = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    L = torch.linalg.cholesky(matrix)
    if not bool(torch.isfinite(L).all().item()):
        raise RuntimeError("non-finite Cholesky factor")
    return L, stats


def elementwise_scale_for_layer(model, block, tokenizer, texts, seqlen, embed_device, layer_idx):
    accum, rows = {}, {}
    hook_calls = [0] * len(block.experts)

    def make_hook(e_idx, key):
        def hook(module, inp, output):
            x = inp[0].detach().float()
            x = x.reshape(-1, x.size(-1))
            if x.shape[0] == 0:
                return
            adds = torch.matmul(x.transpose(0, 1), x).double().cpu()
            if key in accum:
                accum[key] += adds
            else:
                accum[key] = adds
            rows[key] = rows.get(key, 0) + int(x.shape[0])
            hook_calls[e_idx] += 1
        return hook

    handles = []
    for e_idx, expert in enumerate(block.experts):
        for proj_name in QA.PROJ_NAMES:
            handles.append(getattr(expert, proj_name).register_forward_hook(make_hook(e_idx, QA.fisher_key(e_idx, proj_name))))
    counter = QA.RoutingCounter(block)

    model.eval()
    n_used = 0
    try:
        with torch.no_grad():
            for text in texts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seqlen)
                input_ids = enc["input_ids"].to(embed_device)
                if input_ids.shape[1] < 2:
                    continue
                counter.start_sample()
                model(input_ids=input_ids, use_cache=False)
                counter.end_sample()
                n_used += 1
    finally:
        for h in handles:
            h.remove()
        counter.remove()
    if n_used == 0:
        raise RuntimeError("no usable samples (all shorter than 2 tokens)")

    flags = []
    dead = [e for e, c in enumerate(hook_calls) if c == 0]
    for e in dead:
        for proj_name in QA.PROJ_NAMES:
            in_features = getattr(block.experts[e], proj_name).weight.shape[1]
            accum[QA.fisher_key(e, proj_name)] = torch.eye(in_features, dtype=torch.float64)
        flags.append({"layer": layer_idx, "expert": e, "hit_count": 0, "kind": "dead_fallback_identity", "source": "svd_scale"})
        print(f"[qwen15-svd-scale] layer {layer_idx} expert {e}: never routed -> identity (no whitening)", flush=True)
    for e, c in enumerate(hook_calls):
        if 0 < c < QC.LOW_HIT_THRESHOLD:
            flags.append({"layer": layer_idx, "expert": e, "hit_count": c, "kind": "low_hit_flagged",
                          "source": "svd_scale", "samples_routed": counter.hit_samples[e]})
    if [e for e, c in enumerate(hook_calls) if (c == 0) != (counter.hit_samples[e] == 0)]:
        raise RuntimeError("projection hooks and routing counter disagree about which experts were routed")

    nonfinite = [k for k, v in accum.items() if not bool(torch.isfinite(v).all().item())]
    if nonfinite:
        raise RuntimeError(f"non-finite X^T X in {len(nonfinite)} tensors, e.g. {nonfinite[:5]}")

    out, per_key = {}, {}
    for key in sorted(accum, key=lambda k: (int(k.split(".")[2]), k)):
        L, s = process_scaling_matrix_instrumented(accum[key], embed_device)
        out[key] = L.to(torch.bfloat16).cpu()
        s["rows"] = rows.get(key, 0)
        s["trace_raw"] = float(torch.trace(accum[key]).item())
        per_key[key] = s
        del L
    accum.clear()  # free the fp64 accumulators (~5 GB/layer) before the next layer
    stats = {
        "n_used": n_used, "dead_experts": dead, "hook_calls": hook_calls,
        "samples_routed": counter.hit_samples, "routed_tokens": counter.tokens,
        "n_matrices": len(per_key), "n_clamped": sum(1 for s in per_key.values() if s["clamped"]),
        "min_eig_overall": min(s["min_eig"] for s in per_key.values()),
        "max_n_eig_raised": max(s["n_eig_raised"] for s in per_key.values()),
    }
    return out, flags, stats, per_key


def _merge_flags(flags_path, layer_idx, flags):
    existing = json.loads(flags_path.read_text()) if flags_path.exists() else {}
    existing[str(layer_idx)] = [f for f in existing.get(str(layer_idx), []) if f.get("source") != "svd_scale"] + flags
    flags_path.write_text(json.dumps(existing, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=QC.CONDITIONS)
    ap.add_argument("--seed", type=int, required=True, choices=list(QC.SEEDS))
    args = ap.parse_args()
    QC.check_env()

    out_dir = QC.condition_dir(args.condition, args.seed)
    layer_dir = out_dir / "svd_scale_layers"
    layer_dir.mkdir(parents=True, exist_ok=True)
    flags_path = out_dir / "routing_coverage_flags.json"
    rec = QC.ResourceRecorder()
    meta = {"stage": "svd_scale", "condition": args.condition, "seed": args.seed, "n_samples": QC.N_SAMPLES,
            "seqlen": QC.SEQLEN, "clamp_min": CLAMP_MIN, "env": QC.environment_record(), "layers": {},
            "status": "running"}
    try:
        st = rec.begin("load")
        tokenizer = QC.load_tokenizer()
        model = QC.load_model()
        embed_device = model.get_input_embeddings().weight.device
        meta["device_map"] = QC.device_map_summary(model)
        rec.end(st)

        texts = phase1_fisher.build_samples(args.condition, QC.N_SAMPLES, QC.SEQLEN, args.seed)
        meta["calib_text_hash"] = hashlib.md5("".join(texts).encode()).hexdigest()[:12]
        print(f"[qwen15-svd-scale] condition={args.condition} seed={args.seed} {len(texts)} windows "
              f"calib_text_hash={meta['calib_text_hash']}", flush=True)

        for layer_idx in QC.MOE_LAYERS:
            layer_path = layer_dir / f"layer_{layer_idx}.pt"
            if layer_path.exists():
                print(f"[qwen15-svd-scale] layer {layer_idx}: already done, skipping", flush=True)
                continue
            st = rec.begin(f"svd_scale_layer_{layer_idx}")
            block = QA.get_block(model, layer_idx)
            layer_scale, flags, stats, per_key = elementwise_scale_for_layer(
                model, block, tokenizer, texts, QC.SEQLEN, embed_device, layer_idx)
            tmp = layer_path.with_suffix(".tmp")
            torch.save(layer_scale, tmp)
            tmp.rename(layer_path)
            QC.write_json_versioned(layer_dir / f"layer_{layer_idx}.json", per_key)
            if flags:
                _merge_flags(flags_path, layer_idx, flags)
            del layer_scale
            torch.cuda.empty_cache()
            s = rec.end(st, n_flags=len(flags))
            meta["layers"][str(layer_idx)] = {**stats, "resources": s}
            print(f"[qwen15-svd-scale] layer {layer_idx}: done in {s['seconds']}s dead={stats['dead_experts']} "
                  f"clamped={stats['n_clamped']}/{stats['n_matrices']} min_eig={stats['min_eig_overall']:.3e}",
                  flush=True)
        meta["status"] = "ok"
    except BaseException as exc:
        meta["status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["traceback"] = traceback.format_exc()
        raise
    finally:
        rec.close()
        meta["resources"] = rec.summary()
        path = QC.write_json_versioned(out_dir / "svd_scale_run_meta.json", meta)
        print(f"[qwen15-svd-scale] wrote {path} (status={meta['status']})", flush=True)


if __name__ == "__main__":
    main()
