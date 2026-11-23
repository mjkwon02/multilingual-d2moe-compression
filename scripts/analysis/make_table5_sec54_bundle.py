#!/usr/bin/env python3
"""
Table 5 (§5.4 묶음 복원 이득) 재생성 스크립트.

metric = gain(pp) = dpct_compressed − dpct_intervened
       = 복원 후 Δbpb%(baseline 대비) 감소량. **클수록 회복**, 음수면 오히려 악화.

treatment 12쌍 = §5.1/§5.2가 고른 고분산 flagged expert 12개를 **동시** 복원.
control   12쌍 = 같은 레이어의 대조 expert 12개를 동시 복원.
"복원" = 병합된 meanW_deltaUV expert를 원본 W_j로 재바인딩 + delta_u 0
(prereg §2). 매 eval 후 undo하여 다음 arm에 누수 없음(런타임 torch.equal 검증).

취약 언어 평균 = mean(KO, SW, BN) — prereg §3의 P1 지표.

원자료: phase1/sec54/{,seed1_repro/,korean_only_repro/}step2_bundle.json

사용:
  python3 scripts/make_table5_sec54_bundle.py
  python3 scripts/make_table5_sec54_bundle.py --json results/table5_sec54_bundle.json
"""
import argparse
import json
from pathlib import Path
from statistics import mean

SEC54_ROOT = Path("results/deepseek/sec54")
LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
SHORT = {"eng_Latn": "EN", "kor_Hang": "KO", "zho_Hans": "ZH",
         "swh_Latn": "SW", "ben_Beng": "BN"}
VULNERABLE = ["kor_Hang", "swh_Latn", "ben_Beng"]

# (표시 라벨, step2_bundle.json 경로, 본문 표 포함 여부)
#   2026-08-02 개정: 본문 Table 5는 3개 압축 조건 6행이다. korean_only에서 P1 방향이
#   뒤집히므로(T +0.526 > C +0.092) english_only만 싣는 것은 selective reporting이 된다.
#   대상 12쌍은 세 런에서 동일하고 바뀌는 것은 압축 조건뿐 — 그래서 같은 표에 실을 수 있다.
RUNS = [
    ("english_only seed0", SEC54_ROOT / "step2_bundle.json", True),
    ("english_only seed1", SEC54_ROOT / "seed1_repro" / "step2_bundle.json", True),
    ("korean_only seed0", SEC54_ROOT / "korean_only_repro" / "step2_bundle.json", True),
]


def build():
    out = {"metric": "restoration gain (pp) = dpct_compressed - dpct_intervened",
           "vulnerable": VULNERABLE, "sec54_root": str(SEC54_ROOT), "runs": {}}
    for label, path, in_table in RUNS:
        if not path.exists():
            print(f"[skip] {label}: {path} 없음")
            continue
        d = json.loads(path.read_text())
        rec = {"in_main_table": in_table,
               "scale_condition": d.get("scale_condition"),
               "path": str(path),
               # B-3: 원판 압축 모델 bpb가 공식 grid 셀을 6자리로 재현하는지
               "reference_check_bpb_diff": d["reference_check_bpb_diff"],
               "compressed_delta_bpb_pct": d["compressed_delta_bpb_pct"],
               "arms": {}}
        for arm in ("treatment", "control"):
            g = d["gains_pp"][arm]
            rec["arms"][arm] = {"gain_pp": g,
                                "vulnerable_mean_pp": mean(g[l] for l in VULNERABLE)}
        p1 = d["P1_vulnerable_bundle"]
        rec["P1"] = {"difference_pp": p1["difference_pp"],
                     "direction_holds": p1["direction_holds"]}
        rec["P3_english_control_diff_pp"] = d["P3_english_control"]["difference_pp"]
        out["runs"][label] = rec
    return out


def render(out, in_table_only=True):
    lines = ["| 복원 대상 | " + " | ".join(SHORT[l] for l in LANGS) + " | 취약언어 평균 |",
             "|---|" + "---:|" * (len(LANGS) + 1)]
    for label, _, in_table in RUNS:
        if label not in out["runs"] or (in_table_only and not in_table):
            continue
        rec = out["runs"][label]
        cond, seed = label.split()
        run_tag = f"{cond} {seed.replace('seed', 'seed ')}"
        for arm in ("treatment", "control"):
            a = rec["arms"][arm]
            cells = " | ".join(f"{a['gain_pp'][l]:+.3f}" for l in LANGS)
            lines.append(f"| {arm} 12쌍 ({run_tag}) | {cells} "
                         f"| {a['vulnerable_mean_pp']:+.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    out = build()
    print("=== Table 5 본문 (english_only seed0/1 + korean_only seed0, whitening ON) ===\n")
    print(render(out))

    if any(not in_table for _, _, in_table in RUNS):
        print("\n=== 참고: 본문 표 밖의 압축 조건 ===\n")
        print(render(out, in_table_only=False))

    print("\n=== P1 판정 (취약언어 평균 이득 T − C, 예측은 T > C) ===")
    for label in out["runs"]:
        rec = out["runs"][label]
        t = rec["arms"]["treatment"]["vulnerable_mean_pp"]
        c = rec["arms"]["control"]["vulnerable_mean_pp"]
        holds = "HOLDS" if rec["P1"]["direction_holds"] else "FAILS"
        print(f"  {label:<20} T={t:+.3f}  C={c:+.3f}  diff={rec['P1']['difference_pp']:+.3f}pp"
              f"  -> direction {holds}")

    print("\n=== B-3 무결성 (원판 bpb가 공식 grid 셀을 재현하는지) ===")
    for label in out["runs"]:
        diffs = out["runs"][label]["reference_check_bpb_diff"]
        worst = max(abs(v) for v in diffs.values())
        print(f"  {label:<20} max|diff| = {worst:.6f}  "
              f"{'PASS' if worst < 0.001 else 'FAIL'}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
