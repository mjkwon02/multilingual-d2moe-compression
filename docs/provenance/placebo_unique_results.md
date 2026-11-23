# Unique-copy placebo results (source tensors no longer exist)

The shuffled-text placebo arms' Fisher/whitening tensors were deleted on
2026-08-02 (recorded in the legacy run ledger, `item09_placebo_0802` log)
after their derived statistics were computed. The files below are therefore
the **only remaining copies** of those derived results and cannot be
recomputed without re-running the placebo compressions on GPU.

| File (this repo) | Legacy path | SHA-256 | Generating experiment | Paper use |
|---|---|---|---|---|
| `results/source_data/placebo_fisher_rank_zh_bn.json` | `results/placebo_fisher_rank_zh_bn.json` | `3ad39d8b652109d2a034ff46ba48affe27707360e3f9a02b685b71e81a0c952c` | `placebo_fisher_rank_zh_bn.py` on the ZH/BN placebo Fisher tensors | Appendix F4 (placebo rank ρ) |
| `results/source_data/placebo_fisher_rank_zh_bn.md` | same dir | `de856864a4b923fbbd3db1a0d1ad82cf53a4841ab1c182938536d8ec6ad1d5aa` | same | F4 narrative |
| `results/source_data/placebo_switch1_zh_bn.json` | same dir | `a551b5331875ba184d12ffe17120ec03f87072a1b136d7ed917f4e7d5f596411` | `placebo_switch1_zh_bn.py` | F4 |
| `results/source_data/placebo_zh_bn_summary.md` | same dir | `27d8b0a4737a3e6710d98a54523214ac610307faa944c052b77fb82fc4726d40` | summary of the above | F4 |
| `results/source_data/table_b6_2x2_ensw_effects.json` | `results/table_b6_2x2_ensw_effects.json` | `5e7b66a1911853c10c87d6f627cad978001f95ce2dae79d7bcf5620eee2de71c` | `phase1_2x2_ensw_effects.py` (noise-floor cells use placebo evals) | Table B.6 |

A tarball of the placebo evaluation JSONs backed up at deletion time
(`placebo_eval_backup_0802.tar.gz`,
`47b028426ed507a0c90ce8b51dc7f0dc2dcb204efd5bc47d9ff56b731b626665`) is kept
in the private provenance archive. The per-seed placebo *evaluation* files
(bpb JSONs) still exist under `results/deepseek/*_only_placebo/` and were
verified to regenerate the Table 2 noise floors and Table D2.
