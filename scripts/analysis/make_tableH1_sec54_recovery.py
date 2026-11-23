#!/usr/bin/env python3
"""
표 H1 (부록 H.5 — §6.4 복원 범위별 회복률) 재생성 스크립트.

metric = 회복률(원판 압축 손상 대비 %)
       = 100 * (Δbpb%_원판 − Δbpb%_복원후) / Δbpb%_원판
       = 100 * gain_pp / compressed_delta_bpb_pct

gain_pp는 개입 런이 이미 저장해 둔 값이고(`halves_treatment`, `exploratory_probes`),
분모는 같은 런의 `compressed_delta_bpb_pct`다. 이 스크립트는 새 정의를 만들지 않고
그 두 필드로 25칸을 재계산한 뒤, 런이 저장한 `recovery_rate_pct_of_damage`와
1e-12 이내로 일치하는지 대조한다.

범위(행) 5종 — 세는 단위는 expert이지 (T,C) 쌍이 아니다:
  T_late8        treatment 12 중 late 8 expert (L22–25, 27)
  T_early4       treatment 12 중 early 4 expert (L1×2, L3, L5)
  explor_TC24    선정 24 expert 전부 (treatment 12 + control 12 = 12쌍)
  explor_L25_all layer 25 전 expert (64)
  explor_L11_all layer 11 전 expert (64)

조건: 압축 조건 english_only/seed0 (whitening ON, SEC54_RESULT.md AMENDMENT 1).
3~5행은 원 런에서 exploratory로 라벨돼 있으며 어떤 판정에도 입력되지 않는다.

원자료:
  phase1/sec54/step2_full_full.json   (AMENDMENT 2 개별 12쌍 + 반쪽 + 탐색 런)

사용:
  python3 scripts/make_tableH1_sec54_recovery.py
  python3 scripts/make_tableH1_sec54_recovery.py --json results/tableH1_sec54_recovery.json
"""
import argparse
import json
from pathlib import Path

SRC = Path("results/deepseek/sec54/step2_full_full.json")
EVAL_LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
SHORT = {"eng_Latn": "EN", "kor_Hang": "KO", "zho_Hans": "ZH",
         "swh_Latn": "SW", "ben_Beng": "BN"}
TOL = 1e-12

# (행 키, 표에 인쇄할 라벨, 원 런에서 gain_pp를 담고 있는 필드, exploratory 여부)
ROWS = [
    ("T_late8", "treatment 중 late 8 expert (L22–25, 27)", "halves_treatment", False),
    ("T_early4", "treatment 중 early 4 expert (L1×2, L3, L5)", "halves_treatment", False),
    ("explor_TC24", "선정 24 expert 전부 (T 12 + C 12)", "exploratory_probes", True),
    ("explor_L25_all", "layer 25 전 expert (64)", "exploratory_probes", True),
    ("explor_L11_all", "layer 11 전 expert (64)", "exploratory_probes", True),
]


def build():
    run = json.loads(SRC.read_text())
    damage = run["compressed_delta_bpb_pct"]
    stored = run["recovery_rate_pct_of_damage"]

    rows, worst = [], 0.0
    for key, label, field, exploratory in ROWS:
        gain = run[field][key]
        rec = {lang: 100 * gain[lang] / damage[lang] for lang in EVAL_LANGS}
        diffs = {lang: abs(rec[lang] - stored[key][lang]) for lang in EVAL_LANGS}
        worst = max([worst] + list(diffs.values()))
        rows.append({
            "key": key, "label": label, "exploratory": exploratory,
            "gain_pp": {l: gain[l] for l in EVAL_LANGS},
            "recovery_pct_of_damage": rec,
            "stored_max_abs_diff": max(diffs.values()),
        })

    # 개별 24 expert(본문 인용은 안 하지만 "SW 신호는 L1 expert 2개" 서술의 근거)
    per_expert = {k: v for k, v in stored.items()
                  if k not in {r[0] for r in ROWS}}

    return {
        "table": "H1 (부록 H.5) — §6.4 복원 범위별 회복률",
        "metric": "recovery_pct_of_damage = 100 * gain_pp / compressed_delta_bpb_pct",
        "unit": "손상 대비 % (pp 아님)",
        "compression_condition": run["scale_condition"],
        "counting_unit": "expert (not (T,C) pairs): T 12 expert = late 8 + early 4; "
                         "'24 expert' = T 12 + C 12 = 12 pairs",
        "eval_langs": EVAL_LANGS,
        "denominator_compressed_delta_bpb_pct": {l: damage[l] for l in EVAL_LANGS},
        "b3_reference_check_bpb_diff": run["reference_check_bpb_diff"],
        "rows": rows,
        "individual_expert_recovery_pct": per_expert,
        "source": str(SRC),
        "verification": {
            "against": "step2_full_full.json:recovery_rate_pct_of_damage",
            "n_cells": len(ROWS) * len(EVAL_LANGS),
            "tolerance": TOL,
            "max_abs_diff": worst,
            "all_within_tolerance": worst <= TOL,
        },
        "constraints": [
            "rows 3-5 are labelled exploratory in the source run and feed no verdict",
            "every row is english_only/seed0 only — single condition, single seed",
            "unit is % of damage, not pp; Table 5 (§6.4) is in pp",
        ],
    }


def render(out):
    lines = ["| 복원 범위 | " + " | ".join(SHORT[l] for l in EVAL_LANGS) + " |",
             "|---|" + "---:|" * len(EVAL_LANGS)]
    for r in out["rows"]:
        cells = [f"{r['recovery_pct_of_damage'][l]:+.2f}" for l in EVAL_LANGS]
        lines.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None, help="원자료 JSON 저장 경로")
    args = ap.parse_args()

    out = build()
    print(f"압축 조건: {out['compression_condition']}   단위: {out['unit']}")
    print("분모(원판 손상 Δbpb%): " + " / ".join(
        f"{SHORT[l]} {out['denominator_compressed_delta_bpb_pct'][l]:.2f}" for l in EVAL_LANGS))
    print()
    print(render(out))

    print("\n=== 이득 원값 gain(pp) ===")
    for r in out["rows"]:
        tag = " [exploratory]" if r["exploratory"] else ""
        print(f"  {r['key']:15s} " + " / ".join(
            f"{SHORT[l]} {r['gain_pp'][l]:+.4f}" for l in EVAL_LANGS) + tag)

    v = out["verification"]
    print(f"\n=== 원 런 저장값 대조 ({v['n_cells']}칸, 허용오차 {v['tolerance']:g}) ===")
    for r in out["rows"]:
        print(f"  {r['key']:15s} max|diff|={r['stored_max_abs_diff']:.3e}")
    print(f"  전체 max|diff| = {v['max_abs_diff']:.3e}  → "
          f"{'ALL WITHIN TOLERANCE' if v['all_within_tolerance'] else 'MISMATCH'}")
    print("  B-3 reference_check_bpb_diff: " + " / ".join(
        f"{SHORT[l]} {out['b3_reference_check_bpb_diff'][l]}" for l in EVAL_LANGS))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
