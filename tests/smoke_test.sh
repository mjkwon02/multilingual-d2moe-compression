#!/usr/bin/env bash
# Smoke test: regenerate the key paper tables/figures from the preserved
# small result files. No GPU, no network. Run from the repository root:
#   bash tests/smoke_test.sh
# Every command here was verified on 2026-09-28 to reproduce the preserved
# numbers exactly (manifests/verification_report.md).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
fail=0
run(){ echo "== $*"; if ! $PY "$@" >/dev/null 2>&1; then echo "   FAIL: $*"; fail=1; fi; }

run scripts/analysis/phase1_41_headline_gate.py
run scripts/analysis/sib200_extended_report.py
run scripts/analysis/make_table4_pruning_lang.py
run scripts/analysis/make_table5_sec54_bundle.py
run scripts/analysis/make_tableH1_sec54_recovery.py
run scripts/analysis/phase1_2x2_gate.py
run scripts/analysis/phase1_2x2_verify_gate.py
run scripts/analysis/phase1_2x2_ensw_effects.py
run scripts/analysis/phase1_placebo_gate.py
run scripts/analysis/phase1_seed_gate.py
run scripts/analysis/phase1_52b_deficiency_pairs_list.py
run scripts/analysis/phase1_sec54_select_targets.py
run scripts/analysis/phase1_belebele_gate.py --condition korean_only --scale-condition korean_only --pp-ratio 0.2 --num-fewshot 5
run scripts/analysis/phase1_whitening_geometry_analysis.py
run scripts/analysis/make_external_bpb_grid_0916.py
run scripts/analysis/fisher_damage_pair_analysis.py
run scripts/analysis/qwen15_table1_aggregate.py --seeds 0 1 2
run scripts/analysis/qwen15_sib200_clean_aggregate.py
run scripts/evaluation/massive_revised_aggregate.py
run results/downstream/exposure_filtered/sib200_accuracy_tables/_make_table2_accuracy.py
run results/downstream/exposure_filtered/belebele_tables/_make_belebele_clean.py
run results/source_data/tables_main_submission/_make_p0_2.py
run results/source_data/tables_main_submission/_make_p0_3.py
run results/source_data/tables_main_submission/_make_p1_4.py
run results/source_data/tables_main_submission/_make_p1_5.py
run scripts/figures/make_figure3_whitening_evidence.py
run scripts/figures/make_figure4_whitening_distributions.py
run scripts/figures/make_figure_whitening_geometry.py
run scripts/figures/make_figure_teaser_calibration_gap.py

# headline spot-check: Table 1 own-language gains must match the paper
$PY - <<'EOF'
import json
r = json.load(open("results/deepseek/phase1_41_headline_gate_result.json"))
expected = {"eng_Latn": 5.963, "kor_Hang": 60.784, "zho_Hans": 10.759,
            "swh_Latn": 35.551, "ben_Beng": 67.850}
for lang, want in expected.items():
    got = round(r["own_gain"][lang]["mean_gain"], 3)
    assert abs(got - want) < 0.001, f"{lang}: {got} != {want}"
print("headline own-gain spot-check: OK")
EOF

[ "$fail" -eq 0 ] && echo "SMOKE TEST PASSED" || { echo "SMOKE TEST FAILED"; exit 1; }
