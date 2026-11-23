"""Qwen1.5-MoE-A2.7B port of the Phase 1 pipeline -- shared constants, loaders,
guards and run bookkeeping for the second model of the calibration-language
study.

Every phase1_*.py script hard-codes DeepSeek (MODEL_PATH, RESULTS_ROOT, MoE
layers 1..27, n_routed_experts=64). Rather than parametrising those scripts
(several carry uncommitted user edits and every published DeepSeek number was
produced by them as they are), the Qwen path lives in separate qwen15_*.py
files that IMPORT the model-independent pieces unchanged:

  - calibration text: phase1_fisher.build_samples() / phase1_calib_data
    (same pools, same eval-sentence exclusion, same placebo arm-B split, same
    seed rules) -- Qwen sees byte-identical windows, only the tokeniser differs;
  - FLORES eval set + evaluator: phase1_merge_eval.EVAL_* and
    data_utils.evaluate_sentences();
  - delta SVD: model.merge_deepseek.Merge_deepseekMoE.svd_delta (staticmethod)
    and the expert module meanW_deltaUV from the vendored D2MoE repo.

Nothing here writes into the DeepSeek results tree: RESULTS_ROOT carries the
model tag, and write_json_new() refuses to overwrite an existing result.
"""
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
D2MOE_DIR = Path(os.environ.get("D2MOE_DIR", SCRIPT_DIR.parents[1] / "third_party" / "D2MoE"))
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]

# Exact target: the BASE checkpoint. Not -Chat, not GPTQ/AWQ variants.
MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
# Pinned 2026-09-15 from HfApi().model_info(MODEL_ID).sha, so every stage (and
# the vendored get_expert_freq.py, which only accepts a path) reads the same
# weights even if the Hub repo is updated later.
MODEL_REVISION = "1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
MODEL_TAG = "qwen15_moe_a2.7b"
HF_HOME_EXPECTED = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
RESULTS_ROOT = Path("results/qwen/phase1")
DEEPSEEK_RESULTS_ROOT = Path("results/deepseek")  # read-only reference, never written

# Read from the pinned config.json and the safetensors headers on 2026-09-15.
# assert_model_config() re-checks them after loading instead of trusting them.
EXPECTED_CONFIG = {
    "model_type": "qwen2_moe",
    "num_hidden_layers": 24,
    "num_experts": 60,
    "num_experts_per_tok": 4,
    "norm_topk_prob": False,
    "hidden_size": 2048,
    "moe_intermediate_size": 1408,
    "shared_expert_intermediate_size": 5632,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "hidden_act": "silu",
    "output_router_logits": False,
    "tie_word_embeddings": False,
    "vocab_size": 151936,
}
EXPECTED_ARCHITECTURE = "Qwen2MoeForCausalLM"
EXPECTED_TOTAL_PARAMS = 14_315_784_192  # sum over safetensors headers

NUM_HIDDEN_LAYERS = EXPECTED_CONFIG["num_hidden_layers"]
# decoder_sparse_step=1 and mlp_only_layers=[] -> every layer is a sparse MoE
# block (unlike DeepSeek, whose layer 0 is dense and MoE layers are 1..27).
MOE_LAYERS = list(range(NUM_HIDDEN_LAYERS))

# Main-experiment settings, taken from what the DeepSeek grid actually ran, not
# from script defaults: run_phase1_seeds.sh passes --n-samples 64 to
# phase1_fisher.py (whose own default is 128); phase1_svd_scale.py defaults to
# 64; phase1_run_freq_and_scale.py runs get_expert_freq.py with 2000 samples at
# batch 4; phase1_merge_eval.py uses DELTA_RATIO=0.8 / share 1.0 / "fisher".
N_SAMPLES = 64
SEQLEN = 512
SEEDS = (0, 1, 2)
# Per-expert low-rank budget of the delta factors: rank r = int(m*n*0.8/(m+n)).
# NOT a model-wide 80% compression -- see compressed_param_accounting().
DELTA_RATIO = 0.8
SHARE_RATIO = 1.0
MERGE_METHOD = "fisher"
FREQ_MAX_SAMPLES = 2000
FREQ_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8  # bpb is not bit-reproducible across batch shapes (0802 prompt §3.20-10)
LOW_HIT_THRESHOLD = 5  # same project-wide value as phase1_fisher.py / phase1_svd_scale.py

# Paper conditions. mixed_5lang is the paper's "Balanced" (equal CHARACTER
# budget, phase1_calib_data.CHAR_BALANCED_MIXES) -- NOT the older 3-language
# "balanced". Placebo = arm B of the same language (second half of the
# post-eval-exclusion FLORES pool, phase1_calib_data._pool_for(lang, "B")).
MAIN_CONDITIONS = ["english_only", "chinese_only", "korean_only", "swahili_only", "bengali_only", "mixed_5lang"]
PLACEBO_CONDITIONS = ["english_only_placebo", "chinese_only_placebo", "korean_only_placebo", "swahili_only_placebo", "bengali_only_placebo"]
CONDITIONS = MAIN_CONDITIONS + PLACEBO_CONDITIONS
# Deliberately NOT ported: balanced_tok_mono/balanced_tok_mix (their token
# quotas were solved with the DeepSeek tokeniser), robust_mix / targeted /
# vulnerability / interference conditions (§6 methods, closed).
CONDITION_LABEL = {
    "english_only": "English", "chinese_only": "Chinese", "korean_only": "Korean",
    "swahili_only": "Swahili", "bengali_only": "Bengali", "mixed_5lang": "Balanced",
}

# Attention kernel for every stage. The DeepSeek runs on this host used SDPA
# (flash_attn not installed; modeling_deepseek.py hand-patched to fall back,
# 0802 prompt §3.20-8), and SDPA is also what transformers 4.42 picks for
# Qwen2Moe by default -- which is what the vendored get_expert_freq.py gets,
# since it cannot be passed an attn_implementation. Recorded in every output.
ATTN_IMPLEMENTATION = "sdpa"

# Placement: exactly two visible GPUs (project policy, safe_gpus.sh MAX_GPUS=2;
# one GPU forces accelerate offload, which produced silent NaNs / missing
# gradients before). "sequential" with a smaller cap on the first GPU because
# accelerate's root hook (io_same_device=True) copies the fp32 logits back to
# the INPUT device, so the eval cross-entropy chain lands on GPU 0. Placement
# changes where tensors live, not the arithmetic.
DEVICE_MAP_STRATEGY = "sequential"
GPU_MEM_CAPS_GIB = (13, 22)


def model_snapshot_dir(local_only=True):
    """Local snapshot directory of the pinned revision. local_only=True (the
    default for every compute stage) refuses to touch the network, so a stage
    can never silently pull a different revision; qwen15_fetch.py is the only
    place that downloads."""
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=local_only)


def check_env():
    hf_home = os.environ.get("HF_HOME")
    if hf_home != HF_HOME_EXPECTED:
        raise RuntimeError(f"HF_HOME={hf_home!r}, expected {HF_HOME_EXPECTED!r} (0802 prompt §3.20-9).")
    if not sys.prefix.endswith("d2moe_env"):
        raise RuntimeError(f"python prefix {sys.prefix} is not d2moe_env (base env breaks tokenisers, §3.20-5).")


def load_tokenizer():
    """use_fast=False, same call as every phase1_*.py and the vendored
    get_expert_freq.py. For Qwen this is the slow Qwen2Tokenizer (vocab.json +
    merges.txt); qwen15_calib_report.py checks that it produces the same ids as
    the fast tokenizer on every calibration window. Qwen's tokenizer adds NO
    BOS (bos_token is null) while DeepSeek's adds one -- see README/doc."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_snapshot_dir(), use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def gpu_max_memory():
    n_gpu = torch.cuda.device_count()
    if n_gpu != len(GPU_MEM_CAPS_GIB):
        raise RuntimeError(
            f"{n_gpu} visible GPU(s); this pipeline is budgeted for exactly {len(GPU_MEM_CAPS_GIB)} "
            "(source safe_gpus.sh and export CUDA_VISIBLE_DEVICES=$SAFE_GPUS).")
    return {i: f"{cap}GiB" for i, cap in enumerate(GPU_MEM_CAPS_GIB)}


def load_model():
    """bf16, pinned revision, two GPUs, no CPU/disk offload (asserted)."""
    from transformers import AutoModelForCausalLM
    max_memory = gpu_max_memory()
    model = AutoModelForCausalLM.from_pretrained(
        model_snapshot_dir(), device_map=DEVICE_MAP_STRATEGY, max_memory=max_memory,
        torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION,
    )
    assert_no_offload(model)
    assert_model_config(model)
    model.eval()
    return model


def assert_no_offload(model):
    dm = getattr(model, "hf_device_map", {}) or {}
    off = {k: v for k, v in dm.items() if v in ("cpu", "disk")}
    if off:
        raise RuntimeError(f"{len(off)} module(s) offloaded off-GPU (e.g. {next(iter(off.items()))}); refusing to run "
                           "(offloaded backward silently drops gradients, offloaded eval produced NaN).")


def assert_model_config(model):
    cfg = model.config
    bad = {k: (getattr(cfg, k, None), v) for k, v in EXPECTED_CONFIG.items() if getattr(cfg, k, None) != v}
    if bad:
        raise RuntimeError(f"config mismatch (got, expected): {bad}")
    if type(model).__name__ != EXPECTED_ARCHITECTURE:
        raise RuntimeError(f"model class {type(model).__name__} != {EXPECTED_ARCHITECTURE}")
    if cfg._attn_implementation != ATTN_IMPLEMENTATION:
        raise RuntimeError(f"attn_implementation {cfg._attn_implementation} != {ATTN_IMPLEMENTATION}")
    for i in MOE_LAYERS:
        name = type(model.model.layers[i].mlp).__name__
        if name != "Qwen2MoeSparseMoeBlock":
            raise RuntimeError(f"layer {i} mlp is {name}, expected Qwen2MoeSparseMoeBlock")
    n = count_params(model)
    if n != EXPECTED_TOTAL_PARAMS:
        raise RuntimeError(f"total params {n} != {EXPECTED_TOTAL_PARAMS}")


def count_params(module):
    # Module.parameters() already de-duplicates shared Parameters.
    return sum(p.numel() for p in module.parameters())


def condition_dir(condition, seed):
    return RESULTS_ROOT / condition / f"seed{seed}"


def write_json_new(path, obj):
    """Result files are never overwritten: a second run must pick a new path or
    the user must move the old file by hand."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists -- refusing to overwrite a result.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.rename(path)


def write_json_versioned(path, obj):
    """Bookkeeping files (run metadata): keep every attempt, never overwrite."""
    path = Path(path)
    if path.exists():
        path = path.with_name(f"{path.stem}.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}{path.suffix}")
    write_json_new(path, obj)
    return path


def _nvidia_smi_snapshot():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        return [dict(zip(("index", "uuid", "used_mib", "total_mib"), [x.strip() for x in line.split(",")]))
                for line in out.strip().splitlines()]
    except Exception as e:  # informational only
        return f"unavailable: {e}"


class ResourceRecorder:
    """Per-stage wall time, per-GPU peak allocated/reserved (torch caching
    allocator, excludes the CUDA context), process CPU RSS peak (1 s sampler)
    and an nvidia-smi snapshot at each stage end. Visible GPU i maps to the
    physical GPU in CUDA_VISIBLE_DEVICES[i]."""

    def __init__(self):
        import psutil
        self._proc = psutil.Process()
        self._rss_peak = 0
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._sample, daemon=True)
        self._thr.start()
        self.stages = []
        self.t0 = time.time()

    def _sample(self):
        while not self._stop.is_set():
            try:
                self._rss_peak = max(self._rss_peak, self._proc.memory_info().rss)
            except Exception:
                pass
            self._stop.wait(1.0)

    def begin(self, name):
        self._rss_peak = self._proc.memory_info().rss
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)
        return {"name": name, "t_start": time.time()}

    def end(self, st, **extra):
        gib = 1024 ** 3
        st["seconds"] = round(time.time() - st.pop("t_start"), 2)
        st["gpu"] = {str(d): {
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(d) / gib, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(d) / gib, 3),
            "now_allocated_gib": round(torch.cuda.memory_allocated(d) / gib, 3),
        } for d in range(torch.cuda.device_count())}
        st["cpu_rss_peak_gib_sampled"] = round(self._rss_peak / gib, 3)
        st["cpu_rss_peak_gib_lifetime"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / gib, 3)
        st["nvidia_smi_after"] = _nvidia_smi_snapshot()
        st.update(extra)
        self.stages.append(st)
        g = " ".join(f"gpu{d}:{v['peak_allocated_gib']}/{v['peak_reserved_gib']}GiB" for d, v in st["gpu"].items())
        print(f"[qwen15-res] {st['name']}: {st['seconds']}s peak(alloc/reserved) {g} "
              f"cpu_rss_peak={st['cpu_rss_peak_gib_sampled']}GiB", flush=True)
        return st

    def close(self):
        self._stop.set()

    def summary(self):
        return {"stages": self.stages, "total_seconds": round(time.time() - self.t0, 1)}


def environment_record():
    import transformers
    import accelerate
    rec = {
        "host": socket.gethostname(), "python": platform.python_version(), "prefix": sys.prefix,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "accelerate": accelerate.__version__,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "HF_HOME": os.environ.get("HF_HOME"),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "device_map_strategy": DEVICE_MAP_STRATEGY, "gpu_mem_caps_gib": list(GPU_MEM_CAPS_GIB),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return rec


def device_map_summary(model):
    dm = dict(getattr(model, "hf_device_map", {}) or {})
    layers_on = {}
    for k, v in dm.items():
        layers_on.setdefault(str(v), []).append(k)
    return {"hf_device_map": dm, "modules_per_device": {k: len(v) for k, v in layers_on.items()}}
