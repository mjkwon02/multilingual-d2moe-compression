# Verification report

## Addendum 4: public repository cleanup (2026-11-23)

Slimmed the public tree to CORE + reproducibility-support files; 92 files
(89 + the 3 snapshot files emitted by the post-cleanup verification run)
moved — not deleted — to `ARR_8/private_release_archive/` (a sibling of
this tree, never pushed), preserving original relative paths, with
SHA-256 recorded and verified before/after the move
(`ARCHIVE_MANIFEST.tsv` there; full report:
`ARR_8/PUBLIC_REPOSITORY_CLEANUP_REPORT.md`). Moved classes:

- LEGACY: `phase1_bengali_gate.py`/`phase1_swahili_gate.py` (layout-stale;
  quantities verified via the headline gate), pre-revision MASSIVE
  protocol scripts (`massive_aggregate/report/baseline_pilot.py` — no
  legacy MASSIVE results exist in the tree), partial-seed and
  intermediate Qwen reports, submission-era draft/inventory records in
  `tables_main_submission/` (its MANIFEST.md annotated).
- SUPPLEMENTARY (unreported): Fisher Pilot A analysis
  (`analyze_fisher_pilot_a.py`, `aux/pilot_a_*.json`) and the Qwen
  MASSIVE exploration (`qwen15_massive.py`, record JSON).
- TEMPORARY: smoke-mode / deprecated / backup result files
  (`eval_ppl_smoke`, `eval_xnli_smoke_deprecated`, `eval_xnli_v2_smoke`,
  `eval_sib200_pre_2gpu_rerun_backup`), 0-byte nohup logs, driver stdout
  dumps, `massive_throughput_probe.py`, and all timestamped
  `table1_qwen_seeds012.2026*` snapshot triplets (content-identical to
  the canonical files; `.gitignore` now blocks recommitting them).

Deliberately **kept** despite not being figure/table sources:
`fisher_pilot_a.py` (its `LAYER_ROLE`/capture helpers are imported by
four in-tree analysis scripts, incl. the smoke-tested
`phase1_51_fisher_rank_correlation.py`); `massive_data/_build_protocol/
_eval.py` (imported by the revised MASSIVE pipeline); the XNLI v2
preliminary results + script (back the Limitations collapse claim); all
App.-J negative-result artifacts (budget reallocation, robust-mix,
vulnerability-targeted, synthetic surrogate); the legacy 204-item
SIB-200 and legacy n=200 Belebele results (paper Tables 23–28/29 and the
exposure narrative); MASSIVE scorer ablation/diagnostic (scorer
validation provenance); per-cell run logs (runtime-cost provenance cited
in `docs/REPRODUCIBILITY.md`). Two empty directories removed
(`data/metadata/`, `results/qwen/massive/` after its file moved).

**Verification after the cleanup**: `tests/smoke_test.sh` **PASSED**;
109/109 Python files compile; checksum diff against the pre-cleanup
manifest showed exactly the 89 moved files missing and only the edited
docs/.gitignore + the 5 timestamp-unstable figure PDFs changed — every
remaining result file regenerated **bit-identical** (incl. the canonical
Qwen tables and all gate JSONs). Security re-scan clean. Manifest
regenerated (1,495 files, self-verifies).

## Addendum 3: camera-ready figure/table consistency audit (2026-11-23)

Performed against the **final camera-ready PDF** (now available; earlier
passes could not verify against it):

- **Full inventory extracted from the PDF**: 5 figures, 37 tables. Every
  one maps to a repository script + result source — recorded in the new
  `docs/PAPER_FIGURE_TABLE_MAPPING.md` (authoritative correspondence,
  incl. classifications PAPER_MAIN / PAPER_APPENDIX / SUPPLEMENTARY /
  LEGACY).
- **Figure filename mismatches confirmed and documented, not renamed**:
  `fig3_layerwise` = paper Fig 2, `figure4_whitening_distributions` =
  Fig 3, `figureG1_*`/`figureG2_*` = Figs 4/5. The camera-ready .tex
  includes figures by these exact paths (visible in the PDF's draft-mode
  placeholders), so renaming would break the paper build.
- **`fig1_overview` classified LEGACY, then removed**: it was the
  submission's Figure 1 (3-panel overview); the camera-ready replaced it
  with the teaser (`fig_teaser_calibration_gap` = Fig 1). Initially kept
  as LEGACY, then — at the author's request later the same day — the
  figure (pdf+png) and its generator `make_figure1_overview.py` were
  removed from the tree and archived verbatim in the private provenance
  archive (`release_prep_20261123/fig1_overview_legacy/`, with a WHY.md).
  The smoke test no longer runs the script. Its Panel-B provenance
  (`results/source_data/agg_covariance_spectra.npz`) is unique,
  non-regenerable derived data and stays in the tree.
- **Docs renumbered to camera-ready**: `README.md` §12 command comments
  (+ §8 Appendix C→K), `docs/REPRODUCIBILITY.md` Level-1 table (+ new
  rows for Tables 6/7/11 and 31/33/34), `docs/EXPERIMENT_MAPPING.md`
  section cells + §G rewritten as RESOLVED, `docs/experiment_registry.md`
  Paper-items column, `docs/limitations.md` (F4 → F.4). Key renumberings:
  SIB-200 full 204 = Table 23 (was "Table 3"), pruning = Table 8 (was
  "Table 4"), stability = Table 3 (was "Table 6"), XL-Sum OOD = Table 33
  (was rebuttal "Table 31"), Qwen = Tables 36/37, geometry figures =
  Figures 2–5.
- **Author-side flags recorded** (`PAPER_FIGURE_TABLE_MAPPING.md` §4):
  camera-ready Table 29 prints the *legacy* n=200 5-shot Belebele
  evaluation while §4 of the main text says passage-level filtering (the
  filtered tables exist here as rebuttal evidence); the camera-ready
  prints 61.44 pp (2-dp-intermediate convention; full precision 61.43).
- **Verification after the edits**: `tests/smoke_test.sh` PASSED (30
  scripts, incl. all 6 figure scripts); all 5 regenerated paper figures
  visually matched their camera-ready captions/content; `sha256sum -c`
  against the previous manifest showed **only** the 5 edited docs and the
  6 timestamp-unstable figure PDFs changed — every result file, PNG, and
  script regenerated bit-identical (the Qwen aggregator's new timestamped
  report snapshot is content-identical to the canonical file). Security
  re-scan of changed/new files clean. Manifest regenerated (1,584 files,
  self-verifies).

## Addendum 2: final release pass (2026-11-23, later the same day)

- **License finalized**: Apache-2.0 (`LICENSE`, official full text) for
  first-party code; CC BY 4.0 documented for first-party result data; scope
  map in `docs/LICENSES.md`. Upstream D²-MoE confirmed Apache-2.0 (checked
  on GitHub). `LICENSE_PENDING.md` removed (archived privately).
- **CITATION.cff**: four authors (Kwon, Jang, Yun, Kim) in OpenReview
  order; exact new title; no venue/DOI invented. Parses as valid YAML.
- **XL-Sum redistribution stopped**: `data/ood_xlsum/*.txt` removed from
  the tree (CC BY-NC-SA 4.0). Before removal, the corpus was **rebuilt from
  the official `csebuetnlp/xlsum` v2.0 release with
  `e2_build_ood_corpus.py --bytes-per-lang 51200 --seed 0` and verified
  byte-identical to the original files for all five languages** (original
  recovered from the private snapshot archive; its hashes match the
  manifest). This also replaced the public tree's `eng_Latn.txt`, which the
  2026-09-28 sanitization had corrupted (2 mangled occurrences, +14 bytes,
  manifest hash mismatch — same root cause as the `clean_eval_ids.json`
  corruption in Addendum 1). The manifest now also records the sampled
  XL-Sum document IDs per language. Two migration path bugs in
  `e2_build_ood_corpus.py` (output/cache dirs anchored one level too low)
  were fixed; sampling logic untouched. `.gitignore` blocks re-committing
  corpus text and the tarball cache.
- **61.44 pp investigated** (Korean 2×2 whitening effect): full-precision
  recomputation from raw per-seed cells gives 61.4306 (→ 61.43) under both
  aggregation orders; Fisher effect 23.4606 (→ 23.46) exact. The submitted
  61.44 is reproducible only via 2-decimal intermediate rounding. See
  `docs/EXPERIMENT_MAPPING.md` §H.
- **Author-facing Qwen camera-ready materials** (4 `QWEN_*` root files,
  added by a parallel session) moved out of the public tree to
  `../qwen_camera_ready_materials/` — their evidence report quotes private
  absolute paths.
- **Final checks after all changes**: `tests/smoke_test.sh` PASSED; 119/119
  Python files compile; security re-scan clean (credentials 0, PII 0
  outside intentional author names, codenames 0, user paths 0); checksum
  manifest regenerated (1,580 files) and self-verifies.

## Addendum 1: release-preparation pass (2026-11-23)

Performed while finalizing the camera-ready release (paper title updated to
*Tracing Calibration-Language Effects in D²-MoE Compression*):

- **`tests/smoke_test.sh` re-run: PASSED** (CPU-only, base Python
  environment). The regenerated `phase1_41_headline_gate_result.json` was
  additionally compared value-by-value against the legacy raw copy in the
  private provenance archive: identical to 1e-9 (modulo the documented
  `_only_b` → `_only_placebo` key rename).
- **Rebuttal headline numbers re-verified against preserved results**:
  DeepSeek and Qwen Own/Balanced FLORES Δbpb% rows, Qwen SIB-200 clean
  (Balanced 0/15 collapsed cells), Belebele filtered per-language n
  (99–125), MASSIVE (KO baseline 33.5 / Balanced 17.8 / English 4.8), and
  the 2×2 stage effects (KO 61.44/23.46 pp; SW 23.85/9.29 pp) all match.
- **Calibration determinism verified from a fresh FLORES download**
  (same seed → identical text; different seed → different text).
- **Clean-ID manifest re-derived from fresh SIB-200/Belebele/FLORES
  downloads** (`make_clean_eval_ids_0916.py`): reproduces SIB-200 98/204
  and Belebele 549/900 per language. This surfaced a 2026-09-28
  sanitization corruption in `data/splits/clean_eval_ids.json` (a codename
  replacement had mangled the substring "plotter" inside 20 Belebele
  question IDs); the file was repaired and is now byte-identical to the
  freshly derived version. The corrupted copy is archived privately. The
  evaluations themselves predate the corruption and used correct IDs (the
  preserved per-item prediction files contain the correct URLs).
- **Sanitization completed**: internal codenames removed from
  `manifests/migration_map.tsv`, three `src/multilingual_d2moe/` module
  comments, and one `configs/languages.yaml` comment (comment-level only;
  all 119 Python files still compile; originals archived privately).
- **Removed**: stale `results/qwen15_moe_a2.7b/` (a broken aggregate report
  generated with 0 inputs present; the authoritative copies live in
  `results/qwen/`), two empty config directories. Archived privately before
  removal.
- `docs/reproduction.md` renamed to `docs/REPRODUCIBILITY.md`;
  `docs/EXPERIMENT_MAPPING.md`, `docs/IMPLEMENTATION_NOTES.md`,
  `docs/RELEASE_CHECKLIST.md` added; README/CITATION retitled.
- `manifests/public_tree_checksums.sha256` regenerated for the release
  tree (figures and derived JSONs regenerated by the smoke test carry new
  timestamps; the 2026-09-28 migration-time manifest is preserved in the
  private provenance archive).
- Security re-scan: credentials 0, personal info 0, hostnames 0,
  codenames 0.

# Original verification report (2026-09-28)

All commands were run from the repository root in the pinned conda
environment, using **only the small result files preserved in this tree**
(no GPU, no tensors, no model weights). "identical" means every numeric
value matches the legacy result file after rounding to 1e-9 (timestamps and
path-metadata fields excluded).

## Numeric comparisons against legacy results

| Item | Outcome |
|---|---|
| Table 1/2 — `phase1_41_headline_gate_result.json` | identical (all values; placebo keys renamed `_only_b` → `_only_placebo`) |
| Table 3 — `SIB_EXTENDED_REPORT.md` | accuracy tables identical; source-path column now repo-relative |
| Table 3 revision — `table2_raw_per_seed.csv` | identical except source-path column |
| Table 4 — `table4_pruning_lang.json` | identical (179 values) |
| Table 5 — `table5_sec54_bundle.json` | identical (72 values) |
| Table 6 — `p0_3_table6_source.json` | identical (471 values) |
| H1/H2 — `tableH1_sec54_recovery.json` | identical (188 values) |
| B.6 — `table_b6_2x2_ensw_effects.json` | identical (50 values) |
| D2 — `p1_5_placebo_zh_bn.json` | identical (151 values) |
| B6 stage — `p1_4_stage2x2_ensw.json` | identical (159 values) |
| 26 deepseek gate/result JSONs (bulk) | identical, with two explained exceptions below |
| Belebele gates (incl. korean_only full ON+OFF) | identical |
| Belebele filtered — `belebele_raw_per_seed.csv` | identical |
| Whitening geometry — analysis JSON ×2 + heatmap NPZ | identical |
| MASSIVE revised — 8 aggregate JSONs incl. confusion matrices (126,000 values) | identical |
| Qwen Table 1 — `table1_qwen_seeds012.json` | identical (548 values) |
| Qwen SIB filtered — `table_qwen_clean98.json` | identical (604 values) |
| Figures 1, 3, 4, G1, G2, teaser | regenerated successfully; Fig 1 Panel-B constants independently re-derived from `agg_covariance_spectra.npz` (EN 1.000/0.556/0.453; SW 1.000/0.429/0.254 — exact match) |

## Explained exceptions (not regressions)

1. `phase1_pruning_gate_korean_only_result.json`: the current script reports
   two additional languages (swh, ben) that the stored legacy file predates;
   all shared cells are identical.
2. `p0_2_table4_final.json`: the regenerated file adds the recomputed
   `chinese_only` rows that were missing from the legacy reference JSON (the
   script's own cross-check reproduces the other 4 conditions to <1e-9).
3. `phase1_51_fisher_rank_correlation_result_seed*.json`: full-condition
   correlations regenerate identically from the scalar caches, but the
   placebo-comparison section requires the placebo Fisher tensors deleted on
   2026-08-02; the preserved files (which contain that section) are kept as
   authoritative and the script's summary print crashes when the tensors are
   absent (legacy behavior).
4. `phase1_bengali_gate.py` / `phase1_swahili_gate.py`: legacy scripts
   expecting a pre-final placebo directory layout; their quantities are
   produced (and verified) by `phase1_41_headline_gate.py`.

## Clean-tree checks

- Absolute user paths in the tree: **0** (path metadata inside result files
  was rewritten to repo-relative/`$HF_HOME` forms; originals preserved in
  the private provenance archive and on the legacy filesystem).
- Internal codenames / usernames / hostnames: **0** occurrences outside
  `docs/provenance/legacy_repository_manifest.md` (which describes them
  without naming the codenames).
- Credential patterns (`hf_…`, `sk-…`, API keys, private keys): **0**.
- `tests/smoke_test.sh`: **PASSED** end-to-end after all sanitization.

## Patch verification

`patches/deepseek_sdpa_fallback.patch` applies cleanly to upstream
`modeling_deepseek.py` at revision `521d2bc…` (base SHA-256
`c21ae68b…816158`) and reproduces byte-for-byte the file used in all
DeepSeek runs (both on-disk copies were identical).
