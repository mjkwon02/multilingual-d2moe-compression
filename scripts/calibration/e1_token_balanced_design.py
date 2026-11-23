"""E1 STEP 0-d: token-balanced Balanced (`Balanced_tok_mono` / `Balanced_tok_mix`)의
calibration set 설계·검증 — CPU 전용(토크나이저만, 모델 forward 없음).

무엇을 푸는가
-------------
현행 Balanced(`mixed_5lang`)는 **문자** 예산 균등(언어당 20%)이다. 부록 A.2 실측
(`analysis/balanced_token_share_summary.md`)에 따르면 그 결과 **실현 토큰** share는
EN 4.7 / SW 9.9 / ZH 13.3 / KO 35.9 / BN 36.2 % (언어귀속 토큰 재정규화, 3seed 평균)로
KO+BN이 72%를 점유한다. E1은 이 편중을 제거한 두 조건을 만든다.

- `Balanced_tok_mix`: 현행과 **같은 메커니즘**(언어 혼합 풀 → 무작위 2048자 윈도우,
  512토큰 절단). 바꾸는 것은 언어별 **문자 가중치 w**뿐이며, 실현 토큰 share가 각 20%가
  되도록 w를 푼다. → `solve_mix()`.
- `Balanced_tok_mono`: sequence당 단일 언어. 이것은 가중치로 만들 수 없다 — 현행
  `phase1_fisher.build_samples()`가 **혼합 풀 전체**를 이어붙인 문자열 위에서 윈도우를
  뽑기 때문에 어떤 w를 줘도 윈도우 안은 섞인다. 별도 샘플러가 필요하며, 그 설계와
  예산 배분을 `plan_mono()`가 계산한다(코드 반영은 승인 후).

왜 "가중치 ∝ fertility"로 끝나지 않는가
---------------------------------------
윈도우는 2048자를 뽑은 뒤 **512토큰에서 잘린다**(Balanced는 cap 도달 100%). 잘린 뒤
남는 것은 각 윈도우 앞쪽 ~1/4 구간뿐이고, 그 구간의 언어 구성은 seed별 셔플이 정한다.
따라서 실현 share는 해석적 환산(문자예산 ÷ fertility)과 원리적으로 다르며, 부록 A.2가
실측한 seed별 요동도 크다(KO 30.3~37.7%). 그래서 이 스크립트는 환산하지 않고
**실제 윈도우를 토크나이즈해 측정하고, 그 측정값으로 고정점 반복**을 돈다.

사용법:
    HF_HOME=${HF_HOME:-~/.cache/huggingface} conda run -n d2moe_env python \\
        e1_token_balanced_design.py --mode all [--seeds 0 1 2] [--tol-pct 2.0]
"""
import argparse
import bisect
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import phase1_calib_data  # noqa: E402
import phase1_fisher  # noqa: E402

# 부록 A.2 스크립트와 같은 규약을 쓴다 -- 두 산출물이 같은 정의 위에 서야
# "현행 Balanced의 72% 대비" 비교가 성립한다.
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import phase1_a2_balanced_token_share as a2  # noqa: E402

MODEL_PATH = "deepseek-ai/deepseek-moe-16b-base"
LANG_ORDER = a2.LANG_ORDER  # 표 표시 순서: ["eng","swh","zho","kor","ben"]

# 가중치 dict를 만들 때 쓰는 **정본 순서**. 표시 순서와 절대 섞으면 안 된다:
# build_weighted_sentences()는 weights.items()를 그 순서대로 돌며 문장을 append하고,
# 그 뒤 random.Random(seed).shuffle()을 건다. shuffle의 순열은 리스트 *길이*에만
# 의존하므로, 언어 순서가 다르면 같은 순열이 **다른 문장들**에 적용되어 완전히 다른
# calibration 텍스트가 나온다(2026-08-03에 이 함정을 실제로 밟았다: 표시 순서로
# 가중치를 만들어 현행 Balanced의 KO+BN share를 69.81%로 측정했으나, 정본 순서로는
# 72.05%로 부록 A.2 실측과 일치한다). 신규 조건도 이 순서로 정의한다.
WEIGHT_ORDER = list(phase1_calib_data.MIXED_LANGS["mixed_5lang"])  # eng,kor,zho,swh,ben


def _w(values):
    """{lang: weight}를 **정본 순서**로 만든다. dict 리터럴을 직접 쓰지 말 것."""
    return {lang: values[lang] for lang in WEIGHT_ORDER}
SEP = a2.SEP
OUT_DIR = SCRIPT_DIR.parent / "analysis"
N_SAMPLES = 64
SEQLEN = 512
TARGET_SHARE = 100.0 / len(LANG_ORDER)  # 20%


# --------------------------------------------------------------- 공통 측정부
def _pairs_for_weights(weights, seed):
    """`build_weighted_sentences(weights, seed)`와 동일한 문장 리스트를 언어 라벨과
    함께 재현한다. a2.build_labeled_sentences()와 같은 논리지만 가중치를 받는다.

    동치성은 주장하지 않고 검사한다: 반환 직전 원본 build_weighted_sentences()를
    실제로 호출해 문장 순서까지 완전 일치하는지 assert한다(a2와 동일한 안전장치).
    """
    total_weight = sum(weights.values())
    all_langs = set(phase1_calib_data.LANG_OF_SIMPLE.values())
    pools = {lang: phase1_calib_data._pool_for(lang, "A") for lang in all_langs}
    total_chars = sum(sum(len(s) for s in p) for p in pools.values())

    pairs = []
    for lang, w in weights.items():
        if w <= 0:
            continue
        pool = pools[lang]
        target_chars = total_chars * w / total_weight
        acc_chars, idx = 0, 0
        while acc_chars < target_chars:
            s = pool[idx % len(pool)]
            pairs.append((s, lang))
            acc_chars += len(s)
            idx += 1
    random.Random(seed).shuffle(pairs)

    expected = phase1_calib_data.build_weighted_sentences(weights, seed)
    got = [s for s, _ in pairs]
    if got != expected:
        raise AssertionError(
            "labeled rebuild diverged from phase1_calib_data.build_weighted_sentences() -- "
            "라벨링 재구현이 원본과 갈라졌으므로 manifest를 믿을 수 없다. 중단.")
    return pairs


def measure_mix(weights, seed, tokenizer, n_samples=N_SAMPLES, seqlen=SEQLEN):
    """주어진 문자 가중치로 만든 혼합 풀에서 build_samples()와 **동일한 방식**으로
    윈도우를 뽑아, 실현 토큰의 언어별 share를 실측한다.

    build_samples()를 직접 부르지 않는 이유: 그 함수는 condition 이름으로만 호출되고
    임의 가중치를 받지 않는다. 대신 그 안의 윈도우 추출 로직(rng seed+1, 2048자,
    max_start)을 그대로 재현하며, 재현 동치성은 verify_matches_build_samples()가
    현행 Balanced 가중치에서 검사한다.
    """
    pairs = _pairs_for_weights(weights, seed)
    tot_text, starts, labels = a2.build_char_manifest(pairs)
    offsets = a2.window_offsets(tot_text, n_samples, seqlen, seed)

    counts = {lang: 0 for lang in LANG_ORDER}
    boundary = sep_only = special = total = 0
    for off in offsets:
        text = tot_text[off:off + seqlen * 4]
        enc = tokenizer(text, truncation=True, max_length=seqlen, return_offsets_mapping=True)
        for (a, b) in enc["offset_mapping"]:
            total += 1
            if b <= a:
                special += 1
                continue
            first = a2.label_at(starts, labels, off + a)
            last = a2.label_at(starts, labels, off + b - 1)
            if first != last:
                boundary += 1
            elif first == "sep":
                sep_only += 1
            else:
                counts[first] += 1

    lang_total = sum(counts.values())
    manifest_chars = {lang: 0 for lang in LANG_ORDER}
    for sent, lang in pairs:
        manifest_chars[lang] += len(sent)
    mtot = sum(manifest_chars.values())
    return {
        "seed": seed,
        "total_tokens": total,
        "language_attributed_tokens": lang_total,
        "tokens": dict(counts),
        # E1의 "언어별 토큰 수 균등"은 **언어에 귀속된** 토큰끼리의 균등으로 읽는다.
        # 구분자/특수 토큰은 어느 언어의 것도 아니므로 분모에서 뺀다(부록 A.2의
        # "언어귀속 토큰 재정규화" 열과 같은 기준 -- 72% 수치가 나온 그 기준).
        "share_pct": {l: 100.0 * counts[l] / lang_total if lang_total else 0.0 for l in LANG_ORDER},
        "share_pct_of_all": {l: 100.0 * counts[l] / total if total else 0.0 for l in LANG_ORDER},
        "manifest_char_share_pct": {l: 100.0 * manifest_chars[l] / mtot for l in LANG_ORDER},
        "boundary_tokens": boundary,
        "separator_only_tokens": sep_only,
        "special_tokens": special,
    }


def verify_matches_build_samples(tokenizer, seed=0):
    """현행 Balanced 가중치(모두 1.0)에서 이 스크립트의 윈도우 재현이
    phase1_fisher.build_samples()와 **문자 단위로 동일**한지 확인한다. 이게 깨지면
    아래 모든 측정이 압축이 실제로 먹는 텍스트와 무관해지므로 즉시 중단한다."""
    weights = _w({l: 1.0 for l in WEIGHT_ORDER})
    pairs = _pairs_for_weights(weights, seed)
    tot_text, _, _ = a2.build_char_manifest(pairs)
    offsets = a2.window_offsets(tot_text, N_SAMPLES, SEQLEN, seed)
    mine = [tot_text[o:o + SEQLEN * 4] for o in offsets]
    theirs = phase1_fisher.build_samples("mixed_5lang", N_SAMPLES, SEQLEN, seed)
    if mine != theirs:
        raise AssertionError(
            "윈도우 재현이 phase1_fisher.build_samples('mixed_5lang')와 다르다 -- 중단.")
    return {"verified": True, "n_windows": len(mine), "condition": "mixed_5lang", "seed": seed}


# ------------------------------------------------------- Balanced_tok_mix 해
def solve_mix(tokenizer, seeds, max_iter=12, tol_pct=2.0, damping=1.0):
    """실현 토큰 share가 언어별 20%가 되는 문자 가중치 w를 고정점 반복으로 찾는다.

        w <- w * (target_share / realized_share)   (기하평균 정규화)

    한 w를 전 seed에 공통으로 쓴다 -- 조건은 seed별로 텍스트만 다시 섞을 뿐 정의가
    seed에 따라 달라지면 안 되기 때문이다(다른 모든 조건과 같은 규약). 따라서 수렴
    판정도 seed **평균** share로 하고, seed별 편차는 따로 보고한다.
    """
    w = _w({l: 1.0 for l in WEIGHT_ORDER})
    history = []
    for it in range(max_iter):
        per_seed = [measure_mix(w, s, tokenizer) for s in seeds]
        mean_share = {l: sum(m["share_pct"][l] for m in per_seed) / len(per_seed) for l in LANG_ORDER}
        worst = max(abs(mean_share[l] - TARGET_SHARE) for l in LANG_ORDER)
        worst_seedwise = max(abs(m["share_pct"][l] - TARGET_SHARE)
                             for m in per_seed for l in LANG_ORDER)
        history.append({
            "iter": it,
            "weights": dict(w),
            "mean_share_pct": mean_share,
            "per_seed_share_pct": {m["seed"]: m["share_pct"] for m in per_seed},
            "worst_abs_dev_mean_pp": worst,
            "worst_abs_dev_any_seed_pp": worst_seedwise,
        })
        print(f"[solve_mix] iter {it}: mean share " +
              "  ".join(f"{l.split('_')[0]}={mean_share[l]:5.2f}" for l in LANG_ORDER) +
              f"  | worst dev mean={worst:.2f}pp  any-seed={worst_seedwise:.2f}pp")
        if worst <= tol_pct:
            break
        new = _w({l: w[l] * (TARGET_SHARE / mean_share[l]) ** damping if mean_share[l] > 0 else w[l] * 2
                 for l in WEIGHT_ORDER})
        g = 1.0
        for l in WEIGHT_ORDER:
            g *= new[l]
        g **= 1.0 / len(WEIGHT_ORDER)
        w = _w({l: new[l] / g for l in WEIGHT_ORDER})
    return {"weights": w, "history": history, "converged_mean": history[-1]["worst_abs_dev_mean_pp"] <= tol_pct}


# ------------------------------------------------------ Balanced_tok_mono 설계
def plan_mono(tokenizer, seeds, n_samples=N_SAMPLES, seqlen=SEQLEN):
    """sequence당 단일 언어 조건의 예산 배분을 계산한다.

    메커니즘(현행과 다른 부분만): 언어 l마다 **그 언어 풀만** 이어붙인 문자열을 만들고
    거기서 2048자 윈도우를 뽑는다 -> 윈도우 안에 다른 언어가 섞일 경로가 없다.

    예산: 현행 Balanced와 **같은 총 토큰 예산** 64x512 = 32,768 토큰을 유지하고,
    언어당 32768/5 = 6,553.6 토큰을 목표로 한다(총량을 바꾸면 E1이 재배분 효과가
    아니라 예산 크기 효과를 재게 된다).

    문제: 언어별 윈도우당 실현 토큰 r_l이 크게 다르다(EN은 2048자가 512토큰에 못
    미쳐 절단이 안 걸리고, KO/BN은 훨씬 넘겨 512에서 잘린다). 따라서 윈도우 **개수**를
    언어마다 다르게 줘야 하고, 개수만으로는 정수 반올림 잔차가 남는다. 여기서는
    (i) n_l = floor(목표/r_l) 윈도우를 온전히 쓰고 (ii) 마지막 한 윈도우의 텍스트를
    잘라 남은 토큰 수를 정확히 채우는 방식의 잔차를 계산해 보고한다.
    """
    out = {}
    for lang in LANG_ORDER:
        pool = phase1_calib_data._pool_for(lang, "A")
        per_seed = []
        for seed in seeds:
            sents = list(pool)
            random.Random(seed).shuffle(sents)
            tot_text = SEP.join(sents)
            rng = random.Random(seed + 1)
            max_start = max(1, len(tot_text) - seqlen * 4 - 1)
            realized = []
            for _ in range(n_samples):
                i = rng.randint(0, max_start)
                enc = tokenizer(tot_text[i:i + seqlen * 4], truncation=True, max_length=seqlen)
                realized.append(len(enc["input_ids"]))
            per_seed.append({
                "seed": seed,
                "pool_chars": len(tot_text),
                "mean_realized_tokens_per_window": sum(realized) / len(realized),
                "cap_hit_fraction": sum(1 for r in realized if r >= seqlen) / len(realized),
            })
        r = sum(p["mean_realized_tokens_per_window"] for p in per_seed) / len(per_seed)
        target = n_samples * seqlen / len(LANG_ORDER)
        n_full = int(target // r)
        out[lang] = {
            "per_seed": per_seed,
            "mean_realized_tokens_per_window": r,
            "target_tokens": target,
            "n_full_windows": n_full,
            "tokens_from_full_windows": n_full * r,
            "residual_tokens_needing_a_trimmed_window": target - n_full * r,
            "n_windows_if_rounded": round(target / r),
            "share_error_pct_if_rounded": 100.0 * (round(target / r) * r - target) / target,
        }
    total_windows_rounded = sum(v["n_windows_if_rounded"] for v in out.values())
    return {
        "languages": out,
        "total_windows_if_rounded": total_windows_rounded,
        "total_windows_current_balanced": n_samples,
        "worst_share_error_pct_if_rounded": max(abs(v["share_error_pct_if_rounded"]) for v in out.values()),
    }


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "solve-mix", "plan-mono", "baseline"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tol-pct", type=float, default=2.0,
                    help="E1 등록문의 '토큰 수 균등 +/-X%%' 허용 오차 (기본안 2.0, [확정 필요])")
    ap.add_argument("--max-iter", type=int, default=12)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    # phase1_fisher.py:217 / phase1_a2_balanced_token_share.py:282와 **동일한 인자**로
    # 로드한다. 압축이 실제로 쓰는 토크나이저가 기준이며, trust_remote_code=True로
    # 로드하면 리포의 원격 토크나이저 클래스가 잡혀 share가 달라진다(2026-08-03 확인:
    # 그 경로에서 ZH 15.01/BN 34.05 vs 이 경로 ZH 13.30/BN 36.15 — 부록 A.2 실측과
    # 어긋난다). 캐시에 tokenizer.model이 없어 use_fast=False여도 fast 구현으로
    # 폴백되며, offsets_mapping은 그 폴백 덕에 쓸 수 있다(A.2 주석과 동일 상황).
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    print(f"[e1] tokenizer={MODEL_PATH} fast={tokenizer.is_fast} seeds={args.seeds} tol=+/-{args.tol_pct}%")

    result = {
        "_purpose": "E1 STEP 0-d: token-balanced Balanced 조건 설계·검증 (CPU 전용)",
        "tolerance_pct": args.tol_pct,
        "target_share_pct": TARGET_SHARE,
        "lang_order": LANG_ORDER,
        "n_samples": N_SAMPLES,
        "seqlen": SEQLEN,
        "window_chars": SEQLEN * 4,
        "seeds": args.seeds,
    }

    result["build_samples_equivalence_check"] = verify_matches_build_samples(tokenizer)
    print(f"[e1] build_samples() 재현 동치 확인: {result['build_samples_equivalence_check']}")

    if args.mode in ("all", "baseline", "solve-mix"):
        base_w = _w({l: 1.0 for l in WEIGHT_ORDER})
        base = [measure_mix(base_w, s, tokenizer) for s in args.seeds]
        result["current_balanced"] = {
            "weights": base_w,
            "per_seed": base,
            "mean_share_pct": {l: sum(m["share_pct"][l] for m in base) / len(base) for l in LANG_ORDER},
        }
        ms = result["current_balanced"]["mean_share_pct"]
        print("[e1] 현행 Balanced 실현 토큰 share(언어귀속 재정규화): " +
              "  ".join(f"{l.split('_')[0]}={ms[l]:5.2f}%" for l in LANG_ORDER))
        print(f"[e1]   -> KO+BN = {ms['kor_Hang'] + ms['ben_Beng']:.2f}%")

    if args.mode in ("all", "solve-mix"):
        result["balanced_tok_mix"] = solve_mix(tokenizer, args.seeds,
                                               max_iter=args.max_iter, tol_pct=args.tol_pct)

    if args.mode in ("all", "plan-mono"):
        result["balanced_tok_mono_plan"] = plan_mono(tokenizer, args.seeds)
        p = result["balanced_tok_mono_plan"]
        print("[e1] mono 계획: 언어별 윈도우당 실현 토큰 / 필요 윈도우 수")
        for l in LANG_ORDER:
            v = p["languages"][l]
            print(f"[e1]   {l:10} r={v['mean_realized_tokens_per_window']:6.1f}  "
                  f"n(rounded)={v['n_windows_if_rounded']:3d}  "
                  f"share_err={v['share_error_pct_if_rounded']:+.2f}%  "
                  f"cap_hit={v['per_seed'][0]['cap_hit_fraction']:.2f}")
        print(f"[e1]   총 윈도우 {p['total_windows_if_rounded']} (현행 {p['total_windows_current_balanced']}), "
              f"최악 share 오차 {p['worst_share_error_pct_if_rounded']:.2f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "e1_token_balanced_design.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[e1] wrote {out}")


if __name__ == "__main__":
    main()
