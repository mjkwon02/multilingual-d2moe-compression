#!/usr/bin/env python3
"""
Table 4 (structured pruning의 추가 손상, calibration × eval) 재생성 스크립트.

metric = ΔΔbpb% (ON − OFF), 단위 %p
       = [100*(bpb_ON  /bpb_baseline - 1)]
       - [100*(bpb_OFF /bpb_baseline - 1)]
       = 100 * (bpb_ON - bpb_OFF) / bpb_baseline

phase1_pruning_gate.py의 bpb_increase_pct()/delta 정의와 동일하며, 이 스크립트는
그 정의를 5개 calibration 조건 × 3 seed × 5 eval 언어로 확장해 표 전체를 만든다.

원자료:
  {cond}/seed{s}/eval_ppl.json               (OFF arm, pruning 미적용)
  {cond}/seed{s}/pp_ratio_0.2/eval_ppl.json  (ON  arm, pp_ratio=0.2)
  baseline/eval_ppl.json                     (증가율 분모)

공식 게이트 대조: 각 조건의 `phase1_pruning_gate_{cond}_result.json`
(`per_language.{lang}.mean_delta_pp` / `per_seed_delta`)과 1e-12 이내 일치를 확인한다.
korean_only 게이트 json만 EN/KO/ZH 3언어여서 SW/BN은 대조 대상이 없다(원자료 재계산만).

사용:
  python3 scripts/make_table4_pruning_lang.py                       # 표 + 게이트 대조 출력
  python3 scripts/make_table4_pruning_lang.py --json results/table4_pruning_lang.json
"""
import argparse
import json
from pathlib import Path
from statistics import mean

RESULTS_ROOT = Path("results/deepseek")
SEEDS = [0, 1, 2]
PP_RATIO = 0.2
# 표 인쇄 순서 = 취약열 평균 오름차순(본문 Table 4와 동일).
# chinese_only는 사전등록 4조건에 없던 post-hoc 추가 조건이다(PRUNING_LANG_PREREG.md §6 참고).
CONDITIONS = ["english_only", "chinese_only", "korean_only", "swahili_only", "bengali_only"]
PREREG_CONDITIONS = ["english_only", "korean_only", "swahili_only", "bengali_only"]
GATE_TOL = 1e-12
EVAL_LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
SHORT = {"eng_Latn": "EN", "kor_Hang": "KO", "zho_Hans": "ZH",
         "swh_Latn": "SW", "ben_Beng": "BN"}
# PRUNING_LANG_PREREG.md §3 primary metric과 동일한 취약 언어 집합(§5.4와 공유)
VULNERABLE = ["kor_Hang", "swh_Latn", "ben_Beng"]


def bpb_increase_pct(bpb, baseline_bpb):
    return 100 * (bpb / baseline_bpb - 1)


def gate_crosscheck(out):
    """원자료 재계산값 ↔ 공식 phase1_pruning_gate_{cond}_result.json 대조(허용오차 1e-12)."""
    report = {"tolerance": GATE_TOL, "conditions": {}, "max_abs_diff": 0.0,
              "all_within_tolerance": True}
    for cond in CONDITIONS:
        gate_path = RESULTS_ROOT / f"phase1_pruning_gate_{cond}_result.json"
        entry = {"gate_json": str(gate_path)}
        if not gate_path.exists():
            entry.update(status="gate_json_absent", compared=0)
            report["conditions"][cond] = entry
            report["all_within_tolerance"] = False
            continue
        gate = json.loads(gate_path.read_text())["per_language"]
        diffs, missing = {}, []
        for lang in EVAL_LANGS:
            if lang not in gate:          # korean_only 게이트는 EN/KO/ZH만 담고 있음
                missing.append(lang)
                continue
            mine = out["conditions"][cond]["per_lang"][lang]
            d = abs(mine["mean"] - gate[lang]["mean_delta_pp"])
            for a, b in zip(mine["per_seed"], gate[lang].get("per_seed_delta", [])):
                d = max(d, abs(a - b))
            diffs[lang] = d
        worst = max(diffs.values()) if diffs else 0.0
        entry.update(status="ok" if worst <= GATE_TOL else "MISMATCH",
                     compared=len(diffs), not_in_gate_json=missing,
                     per_lang_max_abs_diff=diffs, max_abs_diff=worst,
                     within_tolerance=worst <= GATE_TOL)
        report["conditions"][cond] = entry
        report["max_abs_diff"] = max(report["max_abs_diff"], worst)
        report["all_within_tolerance"] &= worst <= GATE_TOL
    return report


def build():
    baseline = json.loads((RESULTS_ROOT / "baseline" / "eval_ppl.json").read_text())
    out = {"metric": "delta_delta_bpb_pct_pp (ON - OFF)", "pp_ratio": PP_RATIO,
           "seeds": SEEDS, "eval_langs": EVAL_LANGS, "vulnerable": VULNERABLE,
           "results_root": str(RESULTS_ROOT), "condition_order": CONDITIONS,
           "prereg_conditions": PREREG_CONDITIONS,
           "post_hoc_conditions": [c for c in CONDITIONS if c not in PREREG_CONDITIONS],
           "conditions": {}}

    for cond in CONDITIONS:
        per_lang = {lang: [] for lang in EVAL_LANGS}
        vuln_per_seed, seeds_used = [], []
        for seed in SEEDS:
            off_path = RESULTS_ROOT / cond / f"seed{seed}" / "eval_ppl.json"
            on_path = RESULTS_ROOT / cond / f"seed{seed}" / f"pp_ratio_{PP_RATIO}" / "eval_ppl.json"
            if not (off_path.exists() and on_path.exists()):
                print(f"[skip] {cond} seed{seed}: "
                      f"{'OFF' if not off_path.exists() else 'ON'} arm 없음")
                continue
            off, on = json.loads(off_path.read_text()), json.loads(on_path.read_text())
            deltas = {}
            for lang in EVAL_LANGS:
                bb = baseline[lang]["bits_per_byte"]
                d = (bpb_increase_pct(on[lang]["bits_per_byte"], bb)
                     - bpb_increase_pct(off[lang]["bits_per_byte"], bb))
                per_lang[lang].append(d)
                deltas[lang] = d
            vuln_per_seed.append(mean(deltas[l] for l in VULNERABLE))
            seeds_used.append(seed)

        out["conditions"][cond] = {
            "n_seeds": len(seeds_used),
            "seeds_used": seeds_used,
            "per_lang": {
                lang: {
                    "per_seed": per_lang[lang],
                    "mean": mean(per_lang[lang]),
                    # seed 간 부호 불일치 = 방향 결론 불가 (표에서 †로 표시)
                    "sign_consistent": len({1 if d > 0 else -1 if d < 0 else 0
                                            for d in per_lang[lang]}) == 1,
                } for lang in EVAL_LANGS
            },
            "vulnerable_mean": {"per_seed": vuln_per_seed, "mean": mean(vuln_per_seed)},
        }
    out["gate_crosscheck"] = gate_crosscheck(out)
    return out


def render(out):
    lines = []
    header = "| calibration \\ eval | " + " | ".join(SHORT[l] for l in EVAL_LANGS) \
             + " | 취약열(KO/SW/BN) 평균 |"
    lines.append(header)
    lines.append("|---|" + "---:|" * (len(EVAL_LANGS) + 1))
    for cond in CONDITIONS:
        c = out["conditions"][cond]
        cells = []
        for lang in EVAL_LANGS:
            v = c["per_lang"][lang]
            cells.append(f"{v['mean']:+.2f}" + ("" if v["sign_consistent"] else "†"))
        lines.append(f"| {cond} | " + " | ".join(cells)
                     + f" | {c['vulnerable_mean']['mean']:.2f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None, help="원자료 JSON 저장 경로")
    args = ap.parse_args()

    out = build()
    print(render(out))
    print("\n=== per-seed 상세 ===")
    for cond in CONDITIONS:
        c = out["conditions"][cond]
        vp = c["vulnerable_mean"]
        print(f"\n[{cond}]  n={c['n_seeds']}  "
              f"취약열 평균 per seed: " + " / ".join(f"{v:.3f}" for v in vp["per_seed"])
              + f"  → {vp['mean']:.4f}")
        for lang in EVAL_LANGS:
            v = c["per_lang"][lang]
            flag = "" if v["sign_consistent"] else "   † 부호 불일치"
            print(f"   {SHORT[lang]}  " + " / ".join(f"{d:+.2f}" for d in v["per_seed"])
                  + f"   mean={v['mean']:+.4f}{flag}")

    gc = out["gate_crosscheck"]
    print(f"\n=== 공식 게이트 json 대조 (허용오차 {gc['tolerance']:g}) ===")
    for cond in CONDITIONS:
        e = gc["conditions"][cond]
        note = (f"  (게이트 json에 없어 미대조: {','.join(SHORT[l] for l in e['not_in_gate_json'])})"
                if e.get("not_in_gate_json") else "")
        print(f"  {cond:14s} {e['status']:8s} 대조 {e.get('compared', 0)}언어  "
              f"max|diff|={e.get('max_abs_diff', float('nan')):.3e}{note}")
    print(f"  전체 max|diff| = {gc['max_abs_diff']:.3e}  → "
          f"{'ALL WITHIN TOLERANCE' if gc['all_within_tolerance'] else 'MISMATCH'}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
