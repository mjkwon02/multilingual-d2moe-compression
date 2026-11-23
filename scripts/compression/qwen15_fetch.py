"""Download the pinned Qwen/Qwen1.5-MoE-A2.7B snapshot (base checkpoint only)
into HF_HOME, after checking free disk. The only network-touching step of the
Qwen pipeline; every compute stage opens the snapshot with
local_files_only=True.

Checks: HF_HOME, free space on the HF_HOME filesystem >= download size +
margin, then after download: every file present with the Hub-reported size and
the safetensors headers (read locally, no tensor data) summing to the expected
parameter count.

Usage:
    python -u qwen15_fetch.py [--min-free-margin-gb 50]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import qwen15_config as QC  # noqa: E402

ALLOW = ["*.json", "*.safetensors", "vocab.json", "merges.txt", "LICENSE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-margin-gb", type=float, default=50.0)
    args = ap.parse_args()
    QC.check_env()
    from huggingface_hub import HfApi, snapshot_download
    from safetensors import safe_open

    info = HfApi().model_info(QC.MODEL_ID, revision=QC.MODEL_REVISION, files_metadata=True)
    if info.sha != QC.MODEL_REVISION:
        raise RuntimeError(f"Hub returned sha {info.sha}, expected {QC.MODEL_REVISION}")
    sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
    need = sum(v for k, v in sizes.items() if k.endswith(".safetensors"))
    # snapshot_download(local_files_only=True) succeeds as soon as ANY file of
    # the snapshot is cached (e.g. config/tokenizer), so completeness is judged
    # by the weight files themselves.
    try:
        snap = Path(QC.model_snapshot_dir(local_only=True))
        have = snap if all((snap / k).is_file() and (snap / k).stat().st_size == v
                           for k, v in sizes.items() if k.endswith(".safetensors")) else None
    except Exception:
        have = None
    free = shutil.disk_usage(QC.HF_HOME_EXPECTED).free
    print(f"[qwen15-fetch] weights {need/1e9:.2f} GB, free {free/1e9:.1f} GB on {QC.HF_HOME_EXPECTED}, "
          f"already complete: {have is not None}", flush=True)
    if have is None and free < need + args.min_free_margin_gb * 1e9:
        raise RuntimeError(f"not enough free disk: need {need/1e9:.1f}+{args.min_free_margin_gb} GB, have {free/1e9:.1f} GB")

    path = Path(snapshot_download(QC.MODEL_ID, revision=QC.MODEL_REVISION, allow_patterns=ALLOW))
    bad = {k: (sizes[k], (path / k).stat().st_size) for k in sizes
           if k.endswith(".safetensors") and (path / k).stat().st_size != sizes[k]}
    if bad:
        raise RuntimeError(f"size mismatch after download: {bad}")
    total = 0
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                n = 1
                for d in sf.get_slice(k).get_shape():
                    n *= d
                total += n
    if total != QC.EXPECTED_TOTAL_PARAMS:
        raise RuntimeError(f"parameter count from headers {total} != {QC.EXPECTED_TOTAL_PARAMS}")
    out = {"snapshot": str(path), "revision": QC.MODEL_REVISION, "weights_bytes": need, "total_params": total,
           "files": {k: v for k, v in sizes.items()}}
    QC.write_json_versioned(QC.RESULTS_ROOT / "_fetch" / "fetch_record.json", out)
    print(f"[qwen15-fetch] ok: {path} ({total} params)", flush=True)
    print(json.dumps({k: out[k] for k in ("snapshot", "total_params")}), flush=True)


if __name__ == "__main__":
    main()
