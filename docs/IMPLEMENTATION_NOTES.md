# Implementation notes: official D²-MoE vs. this project's code

Summary of what is upstream, what was modified, and what was reimplemented
(paper Appendix B.1). Authoritative per-file details:
`docs/provenance/d2moe_patch_manifest.md` (D²-MoE changes) and
`docs/provenance/deepseek_patch.md` (DeepSeek modeling patch).

## Upstream

- **Repository**: https://github.com/lliai/D2MoE
- **Commit**: `b3092f0` ("Update get_scale.py"), no local commits.
- **License**: check the upstream repository before redistributing any
  derived code. This public tree deliberately ships **no upstream D²-MoE
  files** — only a change manifest and clone instructions (`D2MOE_DIR`
  environment variable points scripts at the clone). The full 126-file
  working-tree diff is preserved in the private provenance archive.

## What the paper pipeline uses from upstream (unmodified semantics)

- **Fisher-weighted expert merging** (`model/merge_deepseek.py`) — used as
  published, with one trivial relative-import fix.
- **Activation-aware whitening + low-rank SVD decomposition** — the official
  algorithm; invoked with delta_ratio 0.8.
- Expert-frequency collection (`preprocess/get_expert_freq.py`) — used with
  a 22 GiB/GPU `max_memory` cap added for 24 GB consumer GPUs.

## What this project reimplemented (in `scripts/compression/`)

- **Layer-wise Fisher computation** — `phase1_fisher.py` supersedes
  upstream `preprocess/get_fisher.py`: CPU accumulation, eager gradient
  freeing, bf16 on-disk layout, explicit seqlen. The upstream file was also
  patched during development (memory caps) but is **not used** by the paper
  pipeline.
- **Whitening/scale statistics** — `phase1_svd_scale.py` supersedes
  upstream `preprocess/get_scale.py` (which had a device-shadowing bug and
  two memory leaks, fixed in the patched copy but likewise unused by the
  paper pipeline).
- **Structured pruning** — `phase1_structured_prune.py` (pp_ratio 0.2
  ablation).
- Calibration-text construction feeds upstream's own sampling logic via a
  `load_dataset` monkeypatch (`phase1_run_freq_and_scale.py`), so upstream
  sampling code runs unmodified on our multilingual text.

## Model-file patch (DeepSeek)

`patches/deepseek_sdpa_fallback.patch` wraps an unconditional `flash_attn`
import in try/except so transformers falls back to SDPA. No numerical or
architectural change. Verified (2026-09-28) to apply cleanly to the pinned
revision and reproduce byte-for-byte the file used in all runs.

## Qwen1.5-MoE-A2.7B provenance (separate from DeepSeek)

The Qwen pipeline (`scripts/compression/qwen15_*.py`) is this project's own
adaptation, written for the cross-model validation.
`qwen15_moe_adapter.py` applies the D²-MoE decomposition to the **60 routed
experts only**, exactly as the DeepSeek runs do (DeepSeek's 2 shared experts
are likewise untouched); Qwen's always-on shared expert, its sigmoid gate,
and the router keep the library's own forward code. It deliberately does
**not** use the vendored `D2MoE/model/merge_qwen.py` (which targets
Qwen2-57B-A14B, retries failed SVDs with random noise, and silently skips
layers on error — behaviours absent from the DeepSeek path); instead it
calls the DeepSeek path's own `svd_delta` and a statement-for-statement copy
of its `merge_method="fisher"` branch, so the method is identical across
models. The adapter's docstring documents the routing-coverage counting
subtlety (Qwen calls every expert each forward, so hits are counted by a
forward pre-hook on routed rows). `qwen15_config.py` pins revision, bf16,
and a strict 2-GPU no-offload loading contract. The Qwen model file is
**unmodified** (no SDPA patch needed). The Qwen experiments validate the
calibration-language *effect*; the mechanism analysis remains
DeepSeek-only.

## Belebele harness note

The vendored lm-evaluation-harness Belebele task configs used for the
*legacy* runs drew 5-shot demonstrations from the test split
(`legacy_exposed`); the corrected protocol re-evaluates from per-item
predictions (`scripts/evaluation/belebele_clean_eval.py`). Do not apply the
legacy task-config edits for corrected-protocol runs. See
`docs/data_exposure_audit.md`.

## Attribution requirements

If you redistribute a modified D²-MoE clone or the patched DeepSeek model
file, preserve the original copyright/license headers and mark your
modifications, per the upstream licenses (D²-MoE repository license;
DeepSeek model license / Apache-2.0 code header). This tree avoids the
question by shipping descriptions and a minimal diff rather than the files.
