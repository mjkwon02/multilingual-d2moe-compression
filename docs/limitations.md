# Known limitations of this reproduction package

1. **Large intermediates are not preserved.** Fisher and whitening tensors
   (~62 GB per condition × seed, DeepSeek; ~75 GB, Qwen) and compressed model
   weights were deleted after their derived statistics were extracted. Every
   paper number regenerates from the preserved small files; recomputing the
   tensors themselves requires the Level-2 GPU pipeline.
2. **Placebo tensors are gone for good.** The placebo arms' Fisher/whitening
   tensors were deleted on 2026-08-02; Appendix F.4 (Table 20) and the B.6 noise floor
   rest on unique preserved derived files (see
   `docs/provenance/placebo_unique_results.md`).
3. **Legacy Belebele few-shot exposure.** The legacy Belebele runs drew
   5-shot demonstrations from the test split (vendored harness config); the
   exposure-filtered protocol is the default. See
   `docs/data_exposure_audit.md`.
4. **SIB-200 demonstrations kept.** The exposure-filtered SIB-200 test
   removes exposed *test items* only; the 14 fixed demonstrations (6 of them
   calibration-pool) were kept, which is why the subset is called
   "exposure-filtered", not "clean".
5. **Two legacy gate scripts were archived.**
   `phase1_bengali_gate.py`/`phase1_swahili_gate.py` expected a pre-final
   placebo layout and could no longer run against this tree; they were
   moved to the private archive in the 2026-11-23 cleanup. Their
   quantities are computed (and were verified) by
   `phase1_41_headline_gate.py`.
6. **Pre-training contamination unverifiable.** Whether the base models'
   pre-training corpora contain FLORES-200/SIB-200/Belebele/XL-Sum cannot be
   checked from here.
7. **Environment pinning is faithful, not minimal.** `requirements/` records
   the exact environment used (including packages irrelevant to any given
   script); no attempt was made to compute a minimal dependency set.
8. **Run drivers are archived privately.** Host-specific GPU queue drivers
   (with machine names and queue logic) were excluded from the public tree;
   the pipeline entry points (`scripts/compression/run_phase1_*.py`) are
   included.
