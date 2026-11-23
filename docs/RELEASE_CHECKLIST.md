# Release checklist (pre-publication audit record)

Status as of **2026-11-23, final release pass**. Earlier pass records are
preserved in `manifests/verification_report.md`.

## Security

- [x] Credential patterns (`hf_…`, `sk-…`, AWS keys, private keys): **0**
  occurrences (re-scanned after all changes).
- [x] Personal information (usernames, emails): **0** occurrences. The
  author-facing Qwen camera-ready evidence files (which referenced private
  absolute paths) were moved **out of the public tree** to
  `../qwen_camera_ready_materials/`.
- [x] Internal hostnames: none (host redacted in
  `requirements/environment.md`; host-specific run drivers excluded,
  archived privately).
- [x] Internal project codenames: **0** (sanitized earlier this pass;
  originals archived privately). Two sanitization-induced corruptions from
  the 2026-09-28 migration were found and repaired:
  `data/splits/clean_eval_ids.json` (20 Belebele IDs) and
  `data/ood_xlsum/eng_Latn.txt` (2 occurrences; the corpus text has since
  been removed from the tree entirely — see below).
- [x] Absolute user paths: none (`/mnt/HDD` free-space notes remain in
  preserved Qwen run logs `results/qwen/followup_0917/logs/`; no usernames
  or hostnames; kept as runtime-cost provenance).
- [x] No `.git` directory in this tree (legacy git state archived
  privately), so no history to leak.

## Licenses and third-party material

- [x] **Code license: Apache License 2.0** (`LICENSE`, official full text).
  Copyright: the four paper authors (see `CITATION.cff`).
- [x] **First-party result data: CC BY 4.0** — documented in
  `docs/LICENSES.md` (scope map for every subtree).
- [x] **XL-Sum text removed from the tree** (CC BY-NC-SA 4.0, incompatible
  with redistribution here). `data/ood_xlsum/manifest.json` retains the
  sampling rule, seed, document IDs, byte counts, and SHA-256 hashes;
  `scripts/calibration/e2_build_ood_corpus.py` rebuilds the corpus from the
  official source — **verified 2026-11-23 byte-identical** to the original
  files for all five languages. `.gitignore` blocks re-committing the
  rebuilt text and the raw tarball cache.
- [x] Upstream D²-MoE is **Apache-2.0** (checked at
  https://github.com/lliai/D2MoE on 2026-11-23); no upstream code is
  redistributed (manifest + clone instructions only), attribution kept in
  `docs/provenance/` and `docs/LICENSES.md`.
- [x] `patches/deepseek_sdpa_fallback.patch` contains only a few diff-context
  lines of the DeepSeek model file (`docs/provenance/deepseek_patch.md`).
- [x] No per-file SPDX headers existed or were required: no third-party
  source code in `scripts/`/`src/`; root `LICENSE` governs (stated in
  `docs/LICENSES.md`).
- [x] Dataset sources, versions, and licenses listed in README §7 and
  `docs/REPRODUCIBILITY.md`.

## Content

- [x] CITATION.cff: new title, four authors in OpenReview order; no
  venue/DOI/URL invented (added when confirmed).
- [x] Largest tracked file ≈ 25 MB (`agg_covariance_spectra.npz`); no
  checkpoints, tensors, or dataset text.
- [x] Legacy vs. corrected downstream protocols separated and never mixed;
  README commands default to the exposure-filtered protocol.
- [x] GPU-pipeline status stated explicitly (code included, **not re-run**
  in the release pass); preserved-result aggregation is the verified path
  (`docs/EXPERIMENT_MAPPING.md`).
- [x] `tests/smoke_test.sh` passes after all changes (2026-11-23, final
  pass); all Python files compile.
- [x] **Camera-ready figure/table consistency audit (2026-11-23, after the
  final PDF became available)**: all 5 figures and 37 tables of the final
  PDF mapped to repository scripts/results in
  `docs/PAPER_FIGURE_TABLE_MAPPING.md`; README/REPRODUCIBILITY/
  EXPERIMENT_MAPPING/experiment_registry renumbered to the camera-ready;
  the submission-only `fig1_overview` (+ its script) removed from the
  tree at the author's request and archived privately; figure files not
  renamed (the .tex includes them by their current paths); regenerated
  figures visually matched the paper; zero result/script files changed
  (verified via checksum diff).
- [x] **Public repository cleanup (2026-11-23)**: 92 LEGACY /
  SUPPLEMENTARY / TEMPORARY files moved (not deleted) to
  `ARR_8/private_release_archive/` with before/after SHA-256 verification
  (see `ARR_8/PUBLIC_REPOSITORY_CLEANUP_REPORT.md` and
  `manifests/verification_report.md` Addendum 4). Smoke test, full
  compile, and bit-identical regeneration re-verified after the move;
  negative-finding evidence (App. J artifacts, XNLI collapse, legacy
  204-item SIB / n=200 Belebele) deliberately retained.
- [x] Checksum manifest regenerated and self-verifying.

## Remaining (non-blocking) items for the authors

1. ~~Verify table numbers against the camera-ready .tex~~ — **done
   2026-11-23 against the final camera-ready PDF**; see
   `docs/PAPER_FIGURE_TABLE_MAPPING.md`. Two paper-side observations
   remain for the authors (same doc, §4): Table 29 prints the legacy
   n=200 Belebele protocol while main-text §4 says passage-filtered, and
   the printed 61.44 pp is the 2-dp-intermediate value (full precision
   61.43).
2. Add venue/DOI to `CITATION.cff` once assigned.
3. Final co-author read-through of README and `docs/`.
