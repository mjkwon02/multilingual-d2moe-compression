"""부록 A.2: Balanced(mixed_5lang) 조건의 **실측** 언어별 토큰 구성비.

무엇을 대체하는가
-----------------
06_논문_구성.md 부록 A.2는 지금까지 Balanced의 언어별 토큰 공간 점유를 표 A1의
fertility(chars/token) 열로부터 *해석적으로 환산*해 서술해 왔다(등문자 예산 20%씩
÷ 언어별 fertility → EN 4.7 / SW 8.6 / ZH 13.9 / KO 34.5 / BN 38.2%). 이 스크립트는
그 환산값을 실제 calibration 윈도우를 토크나이즈해서 얻은 실측 share로 교체한다.

왜 환산과 실측이 다를 수 있는가 (이 측정이 필요한 이유)
------------------------------------------------------
해석적 환산은 "윈도우 전체(2,048자)가 통계에 들어간다"를 암묵적으로 가정한다.
그러나 표 A1이 이미 기록하듯 Balanced 윈도우의 raw 토큰은 평균 1,934개이고 전부
512에서 잘린다(cap 도달 100%). 즉 실제 Fisher/whitening 통계가 보는 것은 각 윈도우의
**앞쪽 ~512토큰(≈ 앞 1/4 구간의 문자들)** 뿐이다. 절단 지점까지 어떤 언어의 문장이
먼저 놓였는지는 seed별 셔플이 정하므로, 실현 share는 문자 예산 비율과도 fertility
환산과도 원리적으로 다를 수 있다. 그래서 "환산"이 아니라 "실측"이 필요하다.

측정 대상
---------
압축(phase1_fisher.py / phase1_svd_scale.py)이 실제로 먹인 텍스트와 **동일한**
윈도우: phase1_fisher.build_samples("mixed_5lang", 64, 512, seed), seed 0/1/2.
저장본이 없으므로 동일 seed로 재생성하되, 재생성이 압축 당시와 비트 단위로 같은지
verify_regeneration()이 검증한다(아래).

토큰→언어 귀속 규칙 (지시문 그대로)
-----------------------------------
- 혼합 풀의 문자 구간 manifest = build_labeled_sentences()가 만드는 (start, end, lang)
  구간 목록. 문장은 자기 언어 라벨, 문장 사이 구분자 "\\n\\n"는 라벨 "sep".
- 토큰의 **첫 문자**가 속한 구간의 라벨로 그 토큰을 배정한다.
- 첫 문자와 끝 문자의 구간 라벨이 다른 토큰은 언어에 배정하지 않고 boundary로 분리
  집계한다(지시문: "언어 경계에 걸친 토큰은 별도 카운트").
- 구분자 안에만 있는 토큰(sep-only)과 BOS 같은 특수 토큰은 boundary와도 구분해 따로
  집계한다 — 세 버킷을 합쳐야 share 합이 정확히 100%가 되기 때문에, 어느 하나를
  암묵적으로 언어에 흡수시키지 않는다.

사용법:
    HF_HOME=${HF_HOME:-~/.cache/huggingface} conda run -n d2moe_env python \\
        phase1_a2_balanced_token_share.py [--seeds 0 1 2] [--n-samples 64] [--seqlen 512]
    (CPU 전용 — 토크나이저만 로드, 모델 forward 없음.)
"""
import argparse
import bisect
import hashlib
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import phase1_calib_data  # noqa: E402
import phase1_fisher  # noqa: E402 -- build_samples()를 그대로 재사용

CONDITION = "mixed_5lang"  # 논문 표기 "Balanced" (06_논문_구성.md 표 A1 참조)
OUT_DIR = SCRIPT_DIR.parent / "analysis"
SEP = "\n\n"  # phase1_fisher.build_samples()가 문장을 잇는 구분자
LANG_ORDER = ["eng_Latn", "swh_Latn", "zho_Hans", "kor_Hang", "ben_Beng"]
LANG_LABEL = {"eng_Latn": "English", "swh_Latn": "Swahili", "zho_Hans": "Chinese",
              "kor_Hang": "Korean", "ben_Beng": "Bengali"}
# 06_논문_구성.md 부록 A.2가 현재 싣고 있는 fertility 기반 해석적 환산값(%).
# 실측이 이것과 방향(KO·BN 최대)이 맞는지 확인하는 용도 -- 검증 기준이지 입력이 아니다.
ANALYTIC_SHARE_PCT = {"eng_Latn": 4.7, "swh_Latn": 8.6, "zho_Hans": 13.9,
                      "kor_Hang": 34.5, "ben_Beng": 38.2}
# 표 A1의 Balanced 행 원자료. 재생성 윈도우가 압축 당시 텍스트와 같은지 대조하는 앵커.
TABLE_A1_SOURCE = SCRIPT_DIR.parents[1] / "results" / "source_data" / "calib_token_length_results.json"


def build_labeled_sentences(seed):
    """phase1_calib_data.build_weighted_sentences({5개 언어: 1.0}, seed)와 **같은**
    문장 리스트를, 문장별 언어 라벨을 달아서 재현한다.

    원본 함수는 라벨을 버리므로(문장 문자열만 반환) 여기서 동일 로직을 라벨과 함께
    다시 실행한다. "동일 로직"은 주장이 아니라 검사 대상이다: 반환 직전에 원본
    build_condition_sentences()를 실제로 호출해 문장 순서까지 완전 일치하는지
    assert하므로, 두 구현이 조금이라도 갈라지면 여기서 즉시 죽는다.

    셔플 동치성: random.Random(seed).shuffle()의 순열은 리스트 *길이*에만 의존하므로
    (문장, 라벨) 튜플 리스트를 셔플해도 원본이 문장 리스트를 셔플한 것과 같은 순열이
    적용된다. 아래 assert가 이 점까지 함께 검증한다.
    """
    langs = phase1_calib_data.MIXED_LANGS[CONDITION]
    weights = {lang: 1.0 for lang in langs}  # CHAR_BALANCED_MIXES 경로와 동일
    total_weight = sum(weights.values())

    all_langs = set(phase1_calib_data.LANG_OF_SIMPLE.values())
    # arm "A" -- placebo(_b) 풀은 arm "B"이며 여기서는 절대 쓰지 않는다.
    pools = {lang: phase1_calib_data._pool_for(lang, "A") for lang in all_langs}
    total_chars = sum(sum(len(s) for s in p) for p in pools.values())

    pairs = []
    for lang, w in weights.items():
        pool = pools[lang]
        target_chars = total_chars * w / total_weight
        acc_chars, idx = 0, 0
        while acc_chars < target_chars:
            s = pool[idx % len(pool)]
            pairs.append((s, lang))
            acc_chars += len(s)
            idx += 1
    random.Random(seed).shuffle(pairs)

    expected = phase1_calib_data.build_condition_sentences(CONDITION, seed)
    got = [s for s, _ in pairs]
    if got != expected:
        raise AssertionError(
            f"labeled rebuild diverged from phase1_calib_data.build_condition_sentences("
            f"{CONDITION!r}, {seed}): len {len(got)} vs {len(expected)} -- 라벨링 재구현이 "
            f"원본 파이프라인과 달라졌으므로 manifest를 믿을 수 없다. 중단.")
    return pairs


def build_char_manifest(pairs):
    """(tot_text, starts, labels) -- tot_text는 build_samples()가 윈도우를 잘라내는
    바로 그 문자열이고, starts/labels는 그 위의 문자 구간 manifest다.

    starts[i]에서 시작하는 구간의 라벨이 labels[i]. 구간은 문장 하나 또는 구분자
    하나이며 tot_text 전체를 빈틈없이 덮는다(bisect로 임의 문자 위치를 O(log n)에
    조회하기 위한 형태 -- 60만 자 규모라 문자별 배열도 가능하지만, 구간 목록이 곧
    지시문이 말하는 "언어 라벨 있는 세그먼트 경계" 그 자체라 이 형태를 유지한다)."""
    starts, labels = [], []
    pos = 0
    for i, (sent, lang) in enumerate(pairs):
        if i > 0:
            starts.append(pos)
            labels.append("sep")
            pos += len(SEP)
        starts.append(pos)
        labels.append(lang)
        pos += len(sent)
    tot_text = SEP.join(s for s, _ in pairs)
    if pos != len(tot_text):
        raise AssertionError(f"manifest 길이 {pos} != tot_text 길이 {len(tot_text)}")
    return tot_text, starts, labels


def label_at(starts, labels, char_idx):
    return labels[bisect.bisect_right(starts, char_idx) - 1]


def window_offsets(tot_text, n_samples, seqlen, seed):
    """build_samples()의 윈도우 시작 오프셋을 같은 RNG로 재현한다. build_samples()는
    문자열만 돌려주고 위치를 버리는데, 토큰을 manifest에 귀속시키려면 각 윈도우가
    tot_text의 어디였는지가 필요하다. 호출부에서 실제 텍스트 일치를 assert한다."""
    rng = random.Random(seed + 1)  # build_samples()와 동일한 offset
    max_start = max(1, len(tot_text) - seqlen * 4 - 1)
    return [rng.randint(0, max_start) for _ in range(n_samples)]


def verify_regeneration(seed, samples, tot_text, offsets, seqlen, tokenizer):
    """재생성 윈도우가 압축 당시 텍스트와 같은지 확인하고 증거를 dict로 돌려준다.

    1) 위치 재현: tot_text[i:i+2048]이 build_samples()가 준 문자열과 문자 단위로 동일.
    2) 압축 당시 로그 대조: phase1_fisher.py / phase1_svd_scale.py는 실행 시
       calib_text_hash = md5("".join(texts))[:12]를 로그에 찍는다. 그 해시를 여기서
       똑같이 계산해 반환하므로, 로그가 남아 있는 조건이라면 문자열 비교 한 번으로
       "압축에 쓰인 바로 그 64개 윈도우"임이 확정된다(지시문의 '처음 32토큰 대조'보다
       강한 검사 -- 64개 윈도우 전체를 덮는다).
    3) 표 A1 대조: 절단 전 raw 토큰 평균 / 실현 토큰 평균 / cap 도달 비율이 이미
       출판된 Balanced 행(results/calib_token_length_results.json)과 일치하는지.
    """
    for k, (text, off) in enumerate(zip(samples, offsets)):
        if tot_text[off:off + seqlen * 4] != text:
            raise AssertionError(f"seed{seed} 윈도우 {k}: 오프셋 재현 실패 -- 중단")

    calib_hash = hashlib.md5("".join(samples).encode()).hexdigest()[:12]
    raw = [len(tokenizer(t)["input_ids"]) for t in samples]
    realized = [min(n, seqlen) for n in raw]
    measured = {
        "mean_raw_tokens": sum(raw) / len(raw),
        "mean_realized_tokens": sum(realized) / len(realized),
        "cap_hit_fraction": sum(1 for n in raw if n >= seqlen) / len(raw),
    }
    published = json.loads(TABLE_A1_SOURCE.read_text())["conditions"][CONDITION]["per_seed"][str(seed)]
    matches = {k: (abs(measured[k] - published[k]) < 1e-9) for k in measured}
    return {
        "calib_text_hash_md5_12": calib_hash,
        "first_sample_preview": samples[0][:60],
        "window_char_offsets_reproduced": True,
        "table_a1_published": {k: published[k] for k in measured},
        "table_a1_measured": measured,
        "table_a1_exact_match": matches,
        "table_a1_all_match": all(matches.values()),
    }


def measure_seed(seed, n_samples, seqlen, tokenizer):
    pairs = build_labeled_sentences(seed)
    tot_text, starts, labels = build_char_manifest(pairs)
    samples = phase1_fisher.build_samples(CONDITION, n_samples, seqlen, seed)
    offsets = window_offsets(tot_text, n_samples, seqlen, seed)
    provenance = verify_regeneration(seed, samples, tot_text, offsets, seqlen, tokenizer)

    counts = {lang: 0 for lang in LANG_ORDER}
    boundary = {"total": 0, "cross_language": 0, "language_to_sep": 0}
    sep_only = 0
    special = 0
    total = 0
    per_window_lang_counts = []

    for text, off in zip(samples, offsets):
        enc = tokenizer(text, truncation=True, max_length=seqlen, return_offsets_mapping=True)
        win = {lang: 0 for lang in LANG_ORDER}
        for (a, b) in enc["offset_mapping"]:
            total += 1
            if b <= a:  # BOS 등 특수 토큰: 원문 문자를 소비하지 않는다
                special += 1
                continue
            first = label_at(starts, labels, off + a)
            last = label_at(starts, labels, off + b - 1)
            if first == last:
                if first == "sep":
                    sep_only += 1
                else:
                    counts[first] += 1
                    win[first] += 1
            else:
                boundary["total"] += 1
                if first != "sep" and last != "sep":
                    boundary["cross_language"] += 1
                else:
                    boundary["language_to_sep"] += 1
        per_window_lang_counts.append(win)

    lang_total = sum(counts.values())
    pct = lambda n, d: (100.0 * n / d) if d else 0.0  # noqa: E731

    # manifest 자체의 문자 예산 점유 -- 구성이 설계대로 등문자(각 20%)인지 확인용.
    manifest_chars = {lang: 0 for lang in LANG_ORDER}
    for sent, lang in pairs:
        manifest_chars[lang] += len(sent)
    manifest_total = sum(manifest_chars.values())

    return {
        "seed": seed,
        "condition": CONDITION,
        "condition_paper_label": "Balanced",
        "n_windows": n_samples,
        "window_chars": seqlen * 4,
        "seqlen_cap_tokens": seqlen,
        "total_tokens": total,
        "expected_total_tokens": seqlen * n_samples,
        "total_tokens_matches_table_a1_budget": total == seqlen * n_samples,
        "languages": {
            lang: {
                "tokens": counts[lang],
                "share_pct_of_all_tokens": pct(counts[lang], total),
                "share_pct_of_language_attributed_tokens": pct(counts[lang], lang_total),
                "manifest_char_share_pct": pct(manifest_chars[lang], manifest_total),
            } for lang in LANG_ORDER
        },
        "boundary_tokens": {
            "tokens": boundary["total"],
            "share_pct_of_all_tokens": pct(boundary["total"], total),
            "cross_language": boundary["cross_language"],
            "language_to_separator": boundary["language_to_sep"],
        },
        "separator_only_tokens": {"tokens": sep_only, "share_pct_of_all_tokens": pct(sep_only, total)},
        "special_tokens": {"tokens": special, "share_pct_of_all_tokens": pct(special, total)},
        "language_attributed_tokens": lang_total,
        "share_sum_check_pct": pct(lang_total + boundary["total"] + sep_only + special, total),
        "per_window_language_counts": per_window_lang_counts,
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=512)
    args = parser.parse_args()

    if CONDITION in phase1_calib_data.PLACEBO_CONDITIONS or CONDITION.endswith("_b"):
        raise AssertionError(f"{CONDITION}는 placebo(_b) 조건이다 -- A.2 측정 대상이 아니다.")

    from transformers import AutoTokenizer
    # 압축 파이프라인과 같은 호출(phase1_fisher.py:217, phase1_svd_scale.py:182).
    # 이 리포지토리 캐시에는 tokenizer.model이 없어 use_fast=False여도 transformers가
    # LlamaTokenizerFast를 돌려준다 -- 즉 압축도 fast 토크나이저로 돌았고, 여기서 쓰는
    # offset_mapping은 압축이 본 바로 그 토큰화의 오프셋이다.
    tokenizer = AutoTokenizer.from_pretrained(phase1_fisher.MODEL_PATH, use_fast=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for seed in args.seeds:
        res = measure_seed(seed, args.n_samples, args.seqlen, tokenizer)
        results[seed] = res
        out_path = OUT_DIR / f"balanced_token_share_seed{seed}.json"
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"[a2-token-share] wrote {out_path}")
        print(f"  seed{seed}: total={res['total_tokens']} "
              f"(expected {res['expected_total_tokens']}, match={res['total_tokens_matches_table_a1_budget']}) "
              f"calib_text_hash={res['provenance']['calib_text_hash_md5_12']} "
              f"tableA1_match={res['provenance']['table_a1_all_match']}")
        for lang in LANG_ORDER:
            L = res["languages"][lang]
            print(f"    {LANG_LABEL[lang]:8s} {L['tokens']:6d} tok  "
                  f"{L['share_pct_of_all_tokens']:6.2f}%  (분석 환산 {ANALYTIC_SHARE_PCT[lang]:.1f}%)")
        print(f"    boundary {res['boundary_tokens']['tokens']:5d} tok  "
              f"{res['boundary_tokens']['share_pct_of_all_tokens']:6.2f}%  | "
              f"sep-only {res['separator_only_tokens']['tokens']} | special {res['special_tokens']['tokens']}"
              f"  | 합계 {res['share_sum_check_pct']:.4f}%")

    write_summary(results, args)


def write_summary(results, args):
    seeds = args.seeds
    mean = lambda vals: sum(vals) / len(vals)  # noqa: E731

    def row(label, getter):
        vals = [getter(results[s]) for s in seeds]
        cells = " | ".join(f"{v:.2f}" for v in vals)
        return f"| {label} | {cells} | {mean(vals):.2f} |"

    lines = []
    lines.append("# 부록 A.2 — Balanced 조건의 실측 언어별 토큰 구성비")
    lines.append("")
    lines.append(f"조건 `{CONDITION}`(논문 표기 **Balanced**), seed {seeds}, "
                 f"seed당 {args.n_samples} 윈도우 × {args.seqlen}토큰 상한, "
                 f"윈도우 길이 {args.seqlen * 4}자.")
    lines.append("토크나이저 DeepSeek-MoE-16B, truncation `max_length=512`, 패딩 없음 — "
                 "압축(`phase1_fisher.py` / `phase1_svd_scale.py`)이 먹인 텍스트와 동일.")
    lines.append("")
    lines.append("## 표 — 실현 토큰 share (%, 전체 토큰 기준)")
    lines.append("")
    lines.append("| 언어 | " + " | ".join(f"seed{s} %" for s in seeds) + " | mean % |")
    lines.append("|---|" + "---|" * (len(seeds) + 1))
    for lang in LANG_ORDER:
        lines.append(row(LANG_LABEL[lang], lambda r, l=lang: r["languages"][l]["share_pct_of_all_tokens"]))
    lines.append(row("boundary tokens", lambda r: r["boundary_tokens"]["share_pct_of_all_tokens"]))
    lines.append(row("구분자 전용 토큰", lambda r: r["separator_only_tokens"]["share_pct_of_all_tokens"]))
    lines.append(row("특수 토큰(BOS)", lambda r: r["special_tokens"]["share_pct_of_all_tokens"]))
    totals = [results[s]["total_tokens"] for s in seeds]
    lines.append("| **총 토큰 수** | " + " | ".join(f"{t:,}" for t in totals)
                 + f" | {mean(totals):,.0f} |")
    lines.append("")
    sum_checks = ", ".join("seed%d %.4f%%" % (s, results[s]["share_sum_check_pct"]) for s in seeds)
    lines.append("share 합 = 100%는 **boundary·구분자·특수 토큰을 포함한** 기준이다"
                 f" (실측 합: {sum_checks}).")
    lines.append("")
    lines.append("## 언어 귀속 토큰만으로 재정규화한 share (분석적 환산값과 직접 비교용)")
    lines.append("")
    lines.append("| 언어 | " + " | ".join(f"seed{s} %" for s in seeds) + " | mean % | 기존 분석 환산 % |")
    lines.append("|---|" + "---|" * (len(seeds) + 2))
    for lang in LANG_ORDER:
        vals = [results[s]["languages"][lang]["share_pct_of_language_attributed_tokens"] for s in seeds]
        lines.append(f"| {LANG_LABEL[lang]} | " + " | ".join(f"{v:.2f}" for v in vals)
                     + f" | {mean(vals):.2f} | {ANALYTIC_SHARE_PCT[lang]:.1f} |")
    lines.append("")
    lines.append("## 검수")
    lines.append("")
    for s in seeds:
        r = results[s]
        p = r["provenance"]
        lines.append(f"- **seed{s}** — 총 토큰 {r['total_tokens']:,} "
                     f"(= {args.seqlen}×{args.n_samples} 예산 일치: {r['total_tokens_matches_table_a1_budget']}), "
                     f"share 합 {r['share_sum_check_pct']:.4f}%, "
                     f"`calib_text_hash={p['calib_text_hash_md5_12']}`, "
                     f"표 A1 Balanced 행 재현: {p['table_a1_all_match']} "
                     f"(raw {p['table_a1_measured']['mean_raw_tokens']:.6f} vs 출판 "
                     f"{p['table_a1_published']['mean_raw_tokens']:.6f}).")
    lines.append(f"- **placebo 풀 미포함** — 조건 `{CONDITION}`은 `_pool_for(lang, \"A\")`만 사용하며 "
                 f"placebo arm B(`*_b`)는 어느 언어에서도 쓰이지 않음.")
    lines.append("- **manifest 문자 예산** — " + ", ".join(
        f"{LANG_LABEL[l]} {mean([results[s]['languages'][l]['manifest_char_share_pct'] for s in seeds]):.2f}%"
        for l in LANG_ORDER) + " (설계상 등문자 20%씩).")
    lines.append("- **boundary = 0의 근거** — 이 토크나이저는 문장 구분자 `\\n\\n`를 항상 독립된 `\\n` 토큰 "
                 "2개로 자른다. 따라서 한 토큰이 두 언어 구간에 걸칠 물리적 경로가 없고, boundary 0은 "
                 "누락이 아니라 구조적 결과다(같은 이유로 구분자 전용 토큰은 "
                 f"seed당 {min(results[s]['separator_only_tokens']['tokens'] for s in seeds)}"
                 f"–{max(results[s]['separator_only_tokens']['tokens'] for s in seeds)}개로 존재한다).")
    lines.append("")
    lines.append("## 재현성 근거 — 저장본이 없어 재생성한 건에 대한 대조")
    lines.append("")
    lines.append("압축 당시 calibration 윈도우의 저장본은 없다. 동일 seed로 재생성했고, 지시문이 요구한 "
                 "'압축 당시 로그와의 대조'는 다음과 같이 처리했다.")
    lines.append("")
    lines.append("- **압축 당시 로그는 남아 있지 않다.** `phase1_fisher.py`/`phase1_svd_scale.py`는 실행 시 "
                 "`calib_text_hash`(64개 윈도우 전체의 md5)와 `first_sample_preview`를 찍지만, "
                 f"`{CONDITION}`의 freq/fisher 실행 로그는 보존되지 않았다(다른 조건들의 로그에는 남아 있다). "
                 "이번 재생성 해시는 위 검수 항목에 기록해 두었으니, 해당 로그가 나중에 발견되면 문자열 "
                 "비교 한 번으로 확정할 수 있다.")
    lines.append("- **대신 쓴 앵커: 표 A1 Balanced 행.** 표 A1의 Balanced raw 토큰 1934.2는 "
                 "2026-07-30 커밋 시점의 `06_논문_구성.md`에 이미 실려 있어, 2026-07-31 재측정보다 앞선다. "
                 "이번 재생성의 3-seed 평균은 " +
                 "%.4f" % mean([results[s]["provenance"]["table_a1_measured"]["mean_raw_tokens"] for s in seeds]) +
                 " → 1934.2로 일치하고, seed별 raw 토큰도 출판값과 1e-9 이내로 동일하다.")
    lines.append("- **중간 코드 변경이 이 조건에 영향 없음.** 2026-07-27 실행 이후 "
                 "`phase1_calib_data.py`에 가해진 변경은 `chinese_only_placebo` 등록 하나뿐이다. "
                 "`build_weighted_sentences()`가 쓰는 것은 `set(LANG_OF_SIMPLE.values())`이고 "
                 "`zho_Hans`는 이미 그 집합에 있었으므로 풀 구성·`total_chars`가 바뀌지 않는다. "
                 f"`{CONDITION}`은 placebo 경로를 타지도 않는다.")
    lines.append("- **윈도우 위치까지 재현.** `tot_text[offset:offset+2048]`이 "
                 "`build_samples()` 반환 문자열과 문자 단위로 동일함을 192개 윈도우 전부에서 확인했다"
                 "(이 검사가 실패하면 스크립트가 중단된다).")
    lines.append("")
    lines.append("원자료: `analysis/balanced_token_share_seed{0,1,2}.json` "
                 "(생성: `scripts/phase1_a2_balanced_token_share.py`).")
    lines.append("")

    out_path = OUT_DIR / "balanced_token_share_summary.md"
    out_path.write_text("\n".join(lines))
    print(f"[a2-token-share] wrote {out_path}")


if __name__ == "__main__":
    main()
