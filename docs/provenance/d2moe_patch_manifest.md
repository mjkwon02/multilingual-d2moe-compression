# D²-MoE local-change manifest

Upstream: https://github.com/lliai/D2MoE, commit
`b3092f0` ("Update get_scale.py"), cloned without local commits. The full
126-file diff of the working tree used for the experiments is preserved in
the private provenance archive (`legacy_git/d2moe_local_changes.patch`);
this manifest records what each change was, why, and whether it is required
for reproduction.

| File(s) | Change | Why | Needed for reproduction? |
|---|---|---|---|
| `model/merge_deepseek.py` | `import hf.utils` → `from .hf.utils import ...` (relative import fix) | import worked only when run from a specific CWD | **Yes** (trivial) |
| `preprocess/get_expert_freq.py` | add `max_memory` cap of 22 GiB/GPU to `device_map="auto"` loading | 24 GB consumer GPUs; leave headroom | **Yes** on ≤24 GB GPUs |
| `preprocess/get_fisher.py` | Fisher accumulation moved to CPU, gradients freed eagerly, 10.5 GiB cap, explicit `seqlen=1024` | OOM avoidance | Not used by the paper pipeline (superseded by `scripts/compression/phase1_fisher.py`), kept for completeness |
| `preprocess/get_scale.py` | memory cap, device-shadowing bug fix, two memory-leak fixes | OOM/correctness | Not used by the paper pipeline (superseded by `scripts/compression/phase1_svd_scale.py`) |
| `lm-evaluation-harness/lm_eval/tasks/belebele/belebele_*.yaml` (122 files) | `test_split: "test"`, `fewshot_split: "test"`, per-language `dataset_name` | make the vendored harness run Belebele per language | **`legacy_exposed` only.** With the template's `first_n` sampler this draws 5-shot demonstrations from the test split itself — see `docs/data_exposure_audit.md`. Do NOT apply for corrected-protocol runs. |
| `model/modeling_deepseek.py` | unmodified | — | — |

Debugging-only edits not listed above (whitespace, prints) are excluded from
the public code; the private diff is the authoritative record.
