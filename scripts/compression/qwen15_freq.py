"""Qwen1.5-MoE-A2.7B: expert routing frequency -- runs the vendored
D2MoE/preprocess/get_expert_freq.py UNMODIFIED with --model_type=qwen, fed the
project's multilingual calibration sentences through the same load_dataset
monkeypatch phase1_run_freq_and_scale.py uses (its helpers are imported, not
copied). Same budget as the DeepSeek grid: max_samples=2000, batch 4,
truncation at 2048, padding=True.

What the vendored script does for Qwen (read, not assumed):
  - hooks every layer's Qwen2MoeSparseMoeBlock, takes output[1] (router
    logits), softmax in fp32, top-k with k = module.top_k (4), and counts every
    selected (token, expert) pair -- padding positions included, exactly as it
    does for DeepSeek's MoEGate hook;
  - loads with device_map="auto", max_memory={i: "22GiB"} (the project memory
    patch) and default attention (SDPA in transformers 4.42 for Qwen2Moe);
  - tokenizer AutoTokenizer(use_fast=False): Qwen adds no BOS.
The revision is pinned by passing the local snapshot directory as
--base_model_path. The frequency list per layer is the f_j that multiplies the
element-wise Fisher in the merge.

Usage:
    python -u qwen15_freq.py --condition english_only --seed 0
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import qwen15_config as QC  # noqa: E402
import phase1_run_freq_and_scale as P1RF  # noqa: E402 -- dataset patch + script runner, reused unchanged


def freq_path(condition, seed):
    return QC.condition_dir(condition, seed) / f"qwen_wikitext_{QC.FREQ_MAX_SAMPLES}_expert_frequencies.json"


def validate(path, cfg):
    d = json.loads(Path(path).read_text())
    keys = sorted(d, key=int)
    if keys != [str(i) for i in QC.MOE_LAYERS]:
        raise RuntimeError(f"{path}: layer keys {keys[:3]}.. != 0..{QC.MOE_LAYERS[-1]}")
    sums = {k: float(sum(d[k])) for k in keys}
    for k in keys:
        if len(d[k]) != cfg["num_experts"]:
            raise RuntimeError(f"{path}: layer {k} has {len(d[k])} counts, expected {cfg['num_experts']}")
    # Every layer sees the same token positions and picks top-4 at each -> equal sums.
    if len(set(sums.values())) != 1:
        raise RuntimeError(f"{path}: per-layer selection totals differ: {sorted(set(sums.values()))[:5]}")
    total = next(iter(sums.values()))
    return {"per_layer_selection_total": total, "token_positions_counted": total / cfg["num_experts_per_tok"],
            "zero_freq_experts": {k: [e for e, c in enumerate(d[k]) if c == 0] for k in keys if 0 in d[k]},
            "min_freq": min(min(v) for v in d.values()), "max_freq": max(max(v) for v in d.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=QC.CONDITIONS)
    ap.add_argument("--seed", type=int, required=True, choices=list(QC.SEEDS))
    args = ap.parse_args()
    QC.check_env()
    QC.gpu_max_memory()  # asserts exactly 2 visible GPUs; the vendored loader caps each at 22GiB itself

    out_dir = QC.condition_dir(args.condition, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = freq_path(args.condition, args.seed)
    if out_json.exists():
        print(f"[qwen15-freq] {out_json} already exists -- stage done, not touching it", flush=True)
        return

    snapshot = QC.model_snapshot_dir()
    argv = [f"--base_model_path={snapshot}", f"--save_path={out_dir}", "--model_type=qwen",
            "--dataset_name=wikitext", "--split=train", f"--seed={args.seed}",
            f"--max_samples={QC.FREQ_MAX_SAMPLES}", f"--batch_size={QC.FREQ_BATCH_SIZE}"]
    rec = QC.ResourceRecorder()
    meta = {"stage": "freq", "condition": args.condition, "seed": args.seed, "argv": argv,
            "env": QC.environment_record(), "status": "running"}
    try:
        st = rec.begin("get_expert_freq")
        P1RF.install_dataset_patch(args.condition, args.seed)
        try:
            mod = P1RF.run_script_with_argv(QC.D2MOE_DIR / "preprocess" / "get_expert_freq.py", argv,
                                            "qwen15_get_expert_freq")
            m = mod.model
            meta["attn_implementation_used"] = m.config._attn_implementation
            meta["model_class"] = type(m).__name__
            meta["device_map"] = QC.device_map_summary(m)
            QC.assert_no_offload(m)
            P1RF.free_module_model(mod)
        finally:
            P1RF.restore_dataset_patch()
        rec.end(st)
        meta["validation"] = validate(out_json, QC.EXPECTED_CONFIG)
        meta["status"] = "ok"
        print(f"[qwen15-freq] wrote {out_json}: {meta['validation']['token_positions_counted']:.0f} token positions "
              f"counted per layer, zero-freq experts: {meta['validation']['zero_freq_experts']}", flush=True)
    except BaseException as exc:
        meta["status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["traceback"] = traceback.format_exc()
        raise
    finally:
        rec.close()
        meta["resources"] = rec.summary()
        QC.write_json_versioned(out_dir / "freq_run_meta.json", meta)


if __name__ == "__main__":
    main()
