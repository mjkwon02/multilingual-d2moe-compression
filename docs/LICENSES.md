# License scope

This repository mixes first-party code, first-party result data, and
pointers to third-party material. The applicable terms differ by subtree;
nothing below applies a license to material we do not own.

## First-party code — Apache License 2.0

All code written for this project is licensed under the **Apache License,
Version 2.0** (root `LICENSE` file): everything under `scripts/`, `src/`,
`tests/`, `configs/`, and the documentation under `docs/`.

- Copyright (c) 2026 Minjeong Kwon, Jinhee Jang, JungMin Yun, YoungBin Kim.
- Individual source files do not carry per-file SPDX headers; the root
  `LICENSE` governs. No third-party source code is redistributed in this
  tree (verified — see below), so no foreign copyright notices are present
  to preserve inside `scripts/`/`src/`.

## First-party result data — CC BY 4.0

The experiment outputs authored by this project — the JSON/CSV/NPZ metric
and analysis files under `results/` and the rendered figures under
`figures/` — are made available under **Creative Commons Attribution 4.0
International (CC BY 4.0)**. Attribution: cite the paper (*Tracing
Calibration-Language Effects in D²-MoE Compression*; see `CITATION.cff`).

Exception within `results/`: fields quoting evaluation-item *identifiers*
(FLORES row numbers, SIB-200/Belebele/MASSIVE item IDs and URLs) are facts
about third-party datasets, included for reproducibility; they carry no
claim over the underlying datasets.

## Third-party material — NOT covered by the licenses above

| Material | Where it appears here | Terms |
|---|---|---|
| **D²-MoE** (https://github.com/lliai/D2MoE, commit `b3092f0`) | Not redistributed. `docs/provenance/d2moe_patch_manifest.md` describes local changes; scripts expect your own clone (`D2MOE_DIR`). | Upstream is **Apache-2.0** (its `LICENSE` file; the appendix copyright line is unfilled upstream). If you redistribute a modified clone, keep its license and mark your changes. |
| **DeepSeek-MoE-16B** model + remote code | `patches/deepseek_sdpa_fallback.patch` contains a few diff-context lines of `modeling_deepseek.py`. | DeepSeek model license + the model code's own terms. The patch is provided for interoperability; see `docs/provenance/deepseek_patch.md`. |
| **Qwen1.5-MoE-A2.7B** | Not redistributed (downloaded from Hugging Face). | Qwen license (Tongyi Qianwen); see the model card. |
| **FLORES-200, SIB-200, Belebele, wikitext, XNLI** | Not redistributed. Only item **ID lists** (`data/splits/`) are included. | Original dataset licenses. |
| **XL-Sum (BBC)** | **Text not redistributed** (CC BY-NC-SA 4.0 — non-commercial share-alike). `data/ood_xlsum/manifest.json` keeps the sampling rule, seed, document IDs, byte counts, and SHA-256 hashes; `scripts/calibration/e2_build_ood_corpus.py` rebuilds the identical corpus from the official source (hash-verified — see `docs/REPRODUCIBILITY.md`). | CC BY-NC-SA 4.0 (upstream). |
| **MASSIVE 1.1** | Dataset not redistributed; downloaded from the official Amazon Science release. Exception: `data/splits/massive_demo_ids.json` quotes the 8 fixed English demonstration utterances (with IDs and intent labels) used for prompt construction, redistributed under CC BY 4.0 with attribution to the MASSIVE dataset. | CC BY 4.0 (as recorded in `scripts/evaluation/massive_data.py`). |

## Verified statements

- No upstream D²-MoE, lm-evaluation-harness, or model source files are
  present in this tree (the patch above is the sole third-party-derived
  content).
- `data/` contains no third-party text, with one disclosed exception:
  `data/ood_xlsum/` holds only the manifest (IDs/hashes), and
  `data/splits/` holds ID lists plus the 8 MASSIVE demonstration
  utterances noted above (CC BY 4.0).
