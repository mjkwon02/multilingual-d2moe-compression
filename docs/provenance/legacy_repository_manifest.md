# Legacy repository manifest

This public tree was assembled on 2026-09-28 from a private working
repository. The legacy repository itself is not published; this manifest
records its state so results can be traced back to it. Full artifacts
(status, tracked diff, untracked file list, per-file SHA-256, D²-MoE diff)
are preserved in a private provenance archive alongside this repository.

- **HEAD commit**: `a51215d82179687f41693a2a0f95cb3900a28aef`
  (2026-07-31, "Preserve Idea-1 diagnosis raw data (unified_sweep Phase A,
  27 layers x 5 langs)")
- **Branch**: `main` (tracking a private GitHub remote)
- **Working-tree state at migration**: 14 modified tracked files (3,326-line
  diff preserved), 3 deleted figure files, 378 untracked files — most
  paper-critical scripts and results were untracked; all were copied into
  this tree and checksummed.
- **Code checksums**: SHA-256 of every migrated source file at migration
  time is in the private archive (`legacy_git/FILE_CHECKSUMS.tsv`, 813
  entries) and, for the files in this tree, in
  `manifests/public_tree_checksums.sha256`.

## Naming changes from the legacy tree

| Legacy identifier | Public identifier | Note |
|---|---|---|
| internal project codenames | none / "this project" | Two internal animal codenames (a repo-level one and a project-level one) were removed from all code, paths and docs; they carried no technical meaning. |
| `<cond>_only_b` | `<cond>_only_placebo` | **`_b` was the actual on-disk identifier of the shuffled-text placebo arms** in every legacy run and inside preserved result-file *contents* (JSON keys, logs). Directory/file names and code were renamed for clarity; historical JSON contents were left untouched, so `_only_b` still appears inside some preserved result files. This is intentional provenance, not an inconsistency. |
| `paper_fill/<date>_*` | `results/source_data/tables_main_submission/`, `results/downstream/exposure_filtered/*_tables/` | table generators + final CSV/TeX |
| `analysis/clean_eval_0916` | `results/downstream/exposure_filtered/deepseek_clean_eval` | |
| `analysis/massive_0916_revised` | `results/downstream/massive_revised` | |
| absolute paths `/mnt/.../d2moe_results/{phase1,qwen15_moe_a2.7b}` | `results/{deepseek,qwen}` (repo-relative) | scripts now run from the repo root; `HF_HOME` and `D2MOE_DIR` are environment variables |
| host/GPU-queue names in drivers | removed | run drivers with host-specific queueing live in the private archive |

## Result files: what regeneration was verified against

Every table/figure regeneration listed in `docs/REPRODUCIBILITY.md` was run in
this tree on 2026-09-28 and compared numerically against the legacy result
files; `manifests/verification_report.md` records the outcomes. Preserved
result files whose generating tensors were deleted before migration
(placebo Fisher/whitening arms, deleted 2026-08-02) are flagged in
`docs/provenance/placebo_unique_results.md`.
