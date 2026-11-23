# 부록 A.2 — Balanced 조건의 실측 언어별 토큰 구성비

조건 `mixed_5lang`(논문 표기 **Balanced**), seed [0, 1, 2], seed당 64 윈도우 × 512토큰 상한, 윈도우 길이 2048자.
토크나이저 DeepSeek-MoE-16B, truncation `max_length=512`, 패딩 없음 — 압축(`phase1_fisher.py` / `phase1_svd_scale.py`)이 먹인 텍스트와 동일.

## 표 — 실현 토큰 share (%, 전체 토큰 기준)

| 언어 | seed0 % | seed1 % | seed2 % | mean % |
|---|---|---|---|---|
| English | 5.42 | 3.91 | 4.37 | 4.57 |
| Swahili | 10.89 | 8.01 | 10.13 | 9.67 |
| Chinese | 13.52 | 11.79 | 13.50 | 12.94 |
| Korean | 30.28 | 37.67 | 36.80 | 34.92 |
| Bengali | 37.07 | 35.92 | 32.46 | 35.15 |
| boundary tokens | 0.00 | 0.00 | 0.00 | 0.00 |
| 구분자 전용 토큰 | 2.63 | 2.50 | 2.55 | 2.56 |
| 특수 토큰(BOS) | 0.20 | 0.20 | 0.20 | 0.20 |
| **총 토큰 수** | 32,768 | 32,768 | 32,768 | 32,768 |

share 합 = 100%는 **boundary·구분자·특수 토큰을 포함한** 기준이다 (실측 합: seed0 100.0000%, seed1 100.0000%, seed2 100.0000%).

## 언어 귀속 토큰만으로 재정규화한 share (분석적 환산값과 직접 비교용)

| 언어 | seed0 % | seed1 % | seed2 % | mean % | 기존 분석 환산 % |
|---|---|---|---|---|---|
| English | 5.58 | 4.02 | 4.49 | 4.70 | 4.7 |
| Swahili | 11.20 | 8.23 | 10.41 | 9.95 | 8.6 |
| Chinese | 13.91 | 12.12 | 13.88 | 13.30 | 13.9 |
| Korean | 31.16 | 38.71 | 37.84 | 35.90 | 34.5 |
| Bengali | 38.15 | 36.92 | 33.38 | 36.15 | 38.2 |

## 검수

- **seed0** — 총 토큰 32,768 (= 512×64 예산 일치: True), share 합 100.0000%, `calib_text_hash=ebd1bc442250`, 표 A1 Balanced 행 재현: True (raw 1922.109375 vs 출판 1922.109375).
- **seed1** — 총 토큰 32,768 (= 512×64 예산 일치: True), share 합 100.0000%, `calib_text_hash=034b0d770f48`, 표 A1 Balanced 행 재현: True (raw 1914.453125 vs 출판 1914.453125).
- **seed2** — 총 토큰 32,768 (= 512×64 예산 일치: True), share 합 100.0000%, `calib_text_hash=ac67f789d87c`, 표 A1 Balanced 행 재현: True (raw 1965.921875 vs 출판 1965.921875).
- **placebo 풀 미포함** — 조건 `mixed_5lang`은 `_pool_for(lang, "A")`만 사용하며 placebo arm B(`*_b`)는 어느 언어에서도 쓰이지 않음.
- **manifest 문자 예산** — English 20.01%, Swahili 20.01%, Chinese 20.00%, Korean 20.00%, Bengali 19.99% (설계상 등문자 20%씩).
- **boundary = 0의 근거** — 이 토크나이저는 문장 구분자 `\n\n`를 항상 독립된 `\n` 토큰 2개로 자른다. 따라서 한 토큰이 두 언어 구간에 걸칠 물리적 경로가 없고, boundary 0은 누락이 아니라 구조적 결과다(같은 이유로 구분자 전용 토큰은 seed당 819–863개로 존재한다).

## 재현성 근거 — 저장본이 없어 재생성한 건에 대한 대조

압축 당시 calibration 윈도우의 저장본은 없다. 동일 seed로 재생성했고, 지시문이 요구한 '압축 당시 로그와의 대조'는 다음과 같이 처리했다.

- **압축 당시 로그는 남아 있지 않다.** `phase1_fisher.py`/`phase1_svd_scale.py`는 실행 시 `calib_text_hash`(64개 윈도우 전체의 md5)와 `first_sample_preview`를 찍지만, `mixed_5lang`의 freq/fisher 실행 로그는 보존되지 않았다(다른 조건들의 로그에는 남아 있다). 이번 재생성 해시는 위 검수 항목에 기록해 두었으니, 해당 로그가 나중에 발견되면 문자열 비교 한 번으로 확정할 수 있다.
- **대신 쓴 앵커: 표 A1 Balanced 행.** 표 A1의 Balanced raw 토큰 1934.2는 2026-07-30 커밋 시점의 `06_논문_구성.md`에 이미 실려 있어, 2026-07-31 재측정보다 앞선다. 이번 재생성의 3-seed 평균은 1934.1615 → 1934.2로 일치하고, seed별 raw 토큰도 출판값과 1e-9 이내로 동일하다.
- **중간 코드 변경이 이 조건에 영향 없음.** 2026-07-27 실행 이후 `phase1_calib_data.py`에 가해진 변경은 `chinese_only_b` 등록 하나뿐이다. `build_weighted_sentences()`가 쓰는 것은 `set(LANG_OF_SIMPLE.values())`이고 `zho_Hans`는 이미 그 집합에 있었으므로 풀 구성·`total_chars`가 바뀌지 않는다. `mixed_5lang`은 placebo 경로를 타지도 않는다.
- **윈도우 위치까지 재현.** `tot_text[offset:offset+2048]`이 `build_samples()` 반환 문자열과 문자 단위로 동일함을 192개 윈도우 전부에서 확인했다(이 검사가 실패하면 스크립트가 중단된다).

원자료: `analysis/balanced_token_share_seed{0,1,2}.json` (생성: `scripts/phase1_a2_balanced_token_share.py`).
