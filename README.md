# Tracing Calibration-Language Effects in D²-MoE Compression

Code and experimental results for
**"Tracing Calibration-Language Effects in D²-MoE Compression".**

Calibration data determine which information a model retains during compression. This project investigates how the **language of calibration data** affects multilingual performance when compressing Mixture-of-Experts (MoE) models with [D²-MoE](https://github.com/lliai/D2MoE).

This repository provides the experimental configurations, evaluation scripts, and saved results behind the paper's tables and figures, together with instructions for reproducing the experiments.

---

## Key findings

**01 — Calibration language changes what survives compression.**  
On FLORES-200, calibration in the evaluation language gives the smallest increase in bits-per-byte (Δbpb%) across all five languages. Character-balanced multilingual calibration ranks second in every language. This ordering holds on **DeepSeek-MoE-16B** and **Qwen1.5-MoE-A2.7B**.

**02 — Whitening is particularly language-sensitive.**  
In DeepSeek stage-separation experiments, changing the whitening calibration language has a larger effect than changing the Fisher-merging language for both tested language pairs. The effect ratios remain below the preregistered threshold for a dominance claim.

**03 — Restoring a few experts is not a reliable fix.**  
Under the tested intervention budgets, restoring high-disagreement experts does not consistently outperform control selections.

> **Scope:** Stage-wise and expert-level analyses were conducted **only on DeepSeek-MoE-16B**. Qwen validates the FLORES-200 performance ordering and provides an additional SIB-200 evaluation; the internal mechanism was not tested on Qwen.

---

## Experimental scope

| Model | Experiments included |
| :--- | :--- |
| **DeepSeek-MoE-16B (base)** | Full calibration grid, stage separation, expert analyses and restoration, downstream evaluations |
| **Qwen1.5-MoE-A2.7B** | Six-condition FLORES-200 grid (3 seeds), exposure-filtered SIB-200 |

**Calibration conditions:** English · Chinese · Korean · Swahili · Bengali · **Balanced**

Balanced (`mixed_5lang`) gives each language an equal **character** budget using mixed-language sequences. It is different from the token-balanced ablation (`balanced_tok_mono`). Each condition uses **64 character windows per seed**, with 2,048 characters per window and a 512-token cap.

**Primary metric:** FLORES-200 Δbpb% (**lower is better**). Supporting evaluations include SIB-200, Belebele, MASSIVE, XL-Sum, and four languages absent from the calibration set. Evaluation coverage varies by experiment; see [`docs/EXPERIMENT_MAPPING.md`](docs/EXPERIMENT_MAPPING.md).

---

## Repository map

```text
configs/                  Experiment configurations
requirements/             Environment specifications
src/multilingual_d2moe/   Shared utilities
scripts/
  calibration/             Calibration sampling and token statistics
  compression/             D²-MoE compression stages
  evaluation/              Benchmark evaluation
  analysis/                Aggregation and analysis
  figures/                 Figure generation
  infra/                   Execution utilities
results/
  deepseek/                Per-condition, per-seed DeepSeek metrics
  qwen/                    Qwen evaluation results
  downstream/              Downstream and exposure-filtered results
  source_data/             Table and figure source data
figures/                  Rendered paper figures
data/                     Evaluation ID lists and sampling manifests
docs/                     Reproduction guides, audits, and provenance
patches/                  Compatibility patch
manifests/                Verification records and checksums
tests/                    Reproducibility checks
```

---

## Getting started

Create the reference DeepSeek environment from the repository root:

```bash
conda env create -f requirements/deepseek_conda_env.yml -n d2moe_env
conda activate d2moe_env
```

DeepSeek and Qwen have model-specific dependencies. Before running an experiment, consult [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and [`requirements/`](requirements/).

**Saved-result regeneration does not require a GPU.** Full compression or inference does require model weights, the relevant datasets, and sufficient GPU memory. Compressed checkpoints and large intermediate tensors are not included. The compression pipeline also requires a separate checkout of the [upstream D²-MoE implementation](https://github.com/lliai/D2MoE); see the [patch manifest](docs/provenance/d2moe_patch_manifest.md).

---

## Reproducing results

### Run the reproducibility checks

```bash
bash tests/smoke_test.sh
```

### Regenerate selected results

```bash
# FLORES-200
python scripts/analysis/phase1_41_headline_gate.py
python scripts/analysis/qwen15_table1_aggregate.py --seeds 0 1 2

# SIB-200: test-item-filtered evaluation
python results/downstream/exposure_filtered/sib200_accuracy_tables/_make_table2_accuracy.py
python scripts/analysis/qwen15_sib200_clean_aggregate.py

# Additional downstream evaluations
python results/downstream/exposure_filtered/belebele_tables/_make_belebele_clean.py
python scripts/evaluation/massive_revised_aggregate.py
```

For the **complete table/figure-to-script mapping**, including stored result files and verification status, see [`docs/EXPERIMENT_MAPPING.md`](docs/EXPERIMENT_MAPPING.md).

---

## Evaluation integrity

The FLORES-200 evaluation uses held-out sentences separate from calibration. A subsequent audit identified overlap between calibration material and portions of the original SIB-200 and Belebele evaluations. We therefore preserve the **original results** alongside the **exposure-filtered re-evaluations**.

| Evaluation | What is reported | Important qualification |
| :--- | :--- | :--- |
| **SIB-200** | 98 calibration-unexposed test items per language | The original 14-shot demonstrations remain; **6 of 14** overlap with calibration material. This is **test-item-filtered**, not fully exposure-free. |
| **Belebele** | Passage-filtered evaluation | Retained item counts differ by language; original and filtered results are kept separately. |
| **MASSIVE** | Zero-shot, 60-way intent classification | Compares **English-only vs. Balanced**, not the complete six-condition grid. No string-level overlap was detected with the evaluated calibration pools. |

See [`docs/data_exposure_audit.md`](docs/data_exposure_audit.md) for overlap checks, filtering rules, and retained item IDs. **Do not compare original and filtered results as though they used the same evaluation subset.**

---

## Reproducibility and limitations

The repository retains per-seed metrics, analysis scripts, and summary statistics needed to regenerate the reported results without rerunning GPU experiments. Some analyses cover fewer conditions than the main FLORES-200 grid, including MASSIVE, XL-Sum, and expert-level interventions. See [`docs/limitations.md`](docs/limitations.md) for their precise scope.

**Model weights and dataset text are not redistributed.** Sources, required subsets, and setup steps are documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md); verification records are available in [`manifests/verification_report.md`](manifests/verification_report.md).

---

## Citation

If you use this repository or its results, please cite the accompanying paper. Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

---

## License and attribution

- **Original code:** [Apache License 2.0](LICENSE)
- **Original results and figures:** CC BY 4.0
- **Third-party materials:** Governed by their original licenses; see [`docs/LICENSES.md`](docs/LICENSES.md)

This project builds on the [official D²-MoE implementation](https://github.com/lliai/D2MoE), which is **not redistributed** here. Implementation provenance and compatibility changes are documented in [`docs/provenance/d2moe_patch_manifest.md`](docs/provenance/d2moe_patch_manifest.md) and [`docs/provenance/deepseek_patch.md`](docs/provenance/deepseek_patch.md).
