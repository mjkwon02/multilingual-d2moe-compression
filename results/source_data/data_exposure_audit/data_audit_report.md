# Calibration–평가 데이터 감사 보고 (2026-09-16)

대상 논문: 「Tracing Calibration-Language Effects Through Multilingual MoE Compression」
실행 지시문: 2026-09-16 "Calibration 데이터 분리 검증과 독립 코퍼스 평가"
저장소: `<repo>`, git HEAD `a51215d`, branch `main`

> **읽는 법.** 이 문서는 **논문이 주장하는 것**과 **저장소에서 코드를 실행해 독립 확인한 것**을
> 항상 분리해 적는다. 확인하지 못한 것은 0이 아니라 **UNKNOWN**으로 적는다.
> 모든 수치는 `analysis/data_audit_0916/`의 기계 산출물에서 나왔고, 재현 명령은
> `artifact_manifest.json::run_commands`에 있다.

---

## 0. 한 문단 요약

**주 지표인 FLORES bpb 평가는 calibration과 실제로 분리돼 있다** — 문장 수준 중복 0,
실제 소비 윈도우 중복 0이며, 이는 인덱스 규약(`pool = column[60:]`, 평가 `column[:60]`)에서
구조적으로 따라 나온다. **그러나 downstream 두 과제는 분리돼 있지 않다.** SIB-200 test의
**204문항 중 106문항(52.0%)**, Belebele의 **900문항 중 351문항(39.0%, passage 488개 중 202개)**이
calibration 풀에 들어간 FLORES-200 devtest 문장에서 나왔고, 그 문장들은 expert-frequency
단계에서 **전량 실제로 소비**됐다. 외부 코퍼스 XL-Sum은 모든 검사 축에서 중복 0이다.
논문 §5.5/Appendix K의 XL-Sum 평가(Table 31)는 **원시 산출물로부터 완전히 재현**됐다
(Δbpb% 20개 값 전부, baseline bpb 9개 전부, Spearman ρ 평균 0.9867 ≈ 문서값 0.987).

---

## 1. 확인된 사실 — calibration 구성 (저장소 독립 확인)

### 1.1 출처와 hold-out 규약

| 항목 | 실측값 | 근거 |
|---|---|---|
| calibration 소스 | `israel/flores-parallel`, split `test` | `scripts/phase1_calib_data.py:SOURCE/SPLIT` |
| 실체 | FLORES-200 **devtest** 1012행 × 204언어, 컬럼 `sentence_<lang>` | 직접 로드 확인 (1012행, `id` 1..1012 유일, `URL` 281종) |
| hold-out | `EVAL_N_SENTENCES = 60` | `phase1_calib_data.py` |
| calibration 풀 | `column[60:]` = devtest 행 **60..1011**, 언어당 952문장 | `_pool_for()` |
| FLORES bpb 평가 | `column[:60]` = 행 **0..59** | `phase1_merge_eval.eval_flores_ppl` + `data_utils.load_flores_sentences(offset=0)` |
| placebo arm B | `pool[len(pool)//2:]` = 행 **536..1011**, 476문장 | `_pool_for(arm="B")` |

논문 §4(p.3, lines 165–205)의 "held-out 문장 분리" 주장은 **FLORES bpb 평가에 대해서는
코드와 일치**한다.

### 1.2 calibration 텍스트가 실제로 소비되는 세 경로

압축 1셀은 calibration을 **세 경로**로 먹는다. 셋 다 `phase1_merge_eval.merge_condition()`이
읽는 산출물을 만들므로 **셋 다 주 결과에 반영**된다(`expert_freq` / `fisher_processed.pt` /
`svd_scale_processed.pt`).

| 경로 | 스크립트 | 표집 | 재현성 |
|---|---|---|---|
| expert-frequency | `D2MoE/preprocess/get_expert_freq.py` | `dataset.shuffle(seed).select(range(min(2000, N)))`, **문장 단위 전량**, truncation 2048 tok | 재현 가능 |
| Fisher | `scripts/phase1_fisher.py::build_samples` | `random.Random(seed+1)`로 **64개 × 2048자** 윈도우, 512토큰 절단 | 재현 가능 |
| whitening | `scripts/phase1_svd_scale.py` | `phase1_fisher.build_samples`를 **무수정 import**, 같은 인자·seed | 재현 가능, **Fisher와 윈도우 동일** |

> **중요한 구조적 결과.** 단일 언어 조건의 풀은 952문장으로 `max_samples=2000`보다 작다.
> 따라서 expert-frequency 단계가 **풀 전체를 소비**한다. 지시문이 요구한 "pool overlap(가능성)
> vs consumed window overlap(실제 노출)"의 구별은 **단일 언어 조건에서는 성립하지 않는다** —
> 풀에 있었다는 것이 곧 실제로 먹혔다는 뜻이다. 이 감사는 그 붕괴를 감추지 않고 경로별로
> 따로 적었다(`overlap_summary.csv`의 `n_pool_overlap` / `n_consumed_freq_overlap` /
> `n_consumed_window_overlap` 3열). `mixed_5lang`에서는 붕괴하지 않는다(풀 4,079행 중
> expert-freq 소비 1,764행).

**512-token 절단의 언어 비대칭** (seed 0, `calibration_reconstruction.json`):

| 조건 | joined 문자수 | 윈도우 평균 실소비 문자 | 64개 중 절단된 것 | 소비된 FLORES 행 |
|---|---:|---:|---:|---:|
| english_only | 125,215 | 2,048 | 1/64 | 633 / 952 |
| chinese_only | 42,184 | 776 | 64/64 | 690 / 952 |
| korean_only | 63,529 | 312 | 64/64 | 292 / 952 |
| swahili_only | 130,922 | 1,242 | 64/64 | 460 / 952 |
| bengali_only | 122,151 | 281 | 64/64 | 163 / 952 |
| mixed_5lang | 486,583 | 590 | 64/64 | 443 / 4,079 |

2,048자 윈도우 중 English는 거의 전부를, Bengali·Korean은 **14~15%만** 실제로 먹는다
(DeepSeek tokenizer의 byte-fallback 때문). 이 비대칭은 Fisher/whitening 단계에만 해당하며
expert-frequency 단계에는 해당하지 않는다.

### 1.3 주 실험 설정 (실측)

`delta_ratio = 0.8`, structured pruning **OFF**(`cfg['prune_ratio']=0.0`), `merge_method=fisher`,
평가 batch **8**(부록 B.5), seed 0/1/2, 64 windows × 512 tokens.
Balanced(`mixed_5lang`)는 **문자 예산 균등**이며 토큰 균등이 아니다(부록 A.2와 일치).
병합된 압축 모델은 **디스크에 저장되지 않는다** — `phase1_merge_eval.py`에 `torch.save`/
`save_pretrained`가 없고, 매 평가마다 압축 통계에서 RAM에 재구성된다.

---

## 2. 확인된 사실 — 중복 검사 결과

검사 방법: (a) 출처 ID 복원, (b) 원문 exact match(UTF-8 해시), (c) 정규화 exact match
(NFC + 공백 축약만), (d) 실제 소비 윈도우의 char span 대조, (e) 문서 수준(URL) 검사,
(f) 교차 언어(번역 대응 source ID) 검사. 조건 21개(주 6 + 보조 1 + placebo 5, × seed 0/1/2)
= **63 재구성**, 평가 항목 6종에 대해 전수 검사했다(`overlap_summary.csv` 900행).

### 2.1 출처 ID 매핑 — 가정하지 않고 실측 확정

| 평가셋 | source ID | 복원 방법 | 상태 |
|---|---|---|---|
| FLORES | devtest 행 번호 (= `id` − 1) | 직접 | VERIFIED |
| SIB-200 | `index_id` | **`index_id − devtest_row = 997`, 유일값**으로 실측 확정. `index_id ≥ 997` ⇒ devtest 행 `index_id−997`; `< 997` ⇒ FLORES **dev**(이 프로젝트가 한 번도 쓰지 않는 split) | VERIFIED |
| Belebele | `link`(원문 기사 URL) + `flores_passage` | passage 안에 부분 문자열로 들어 있는 devtest 문장을 언어별로 탐색 | VERIFIED |
| XL-Sum | 문서 index + `tar_member` | `manifest.json` + 원본 tarball 재파싱 | VERIFIED |

감사 스크립트는 997 오프셋을 **하드코딩하지 않고 매 실행마다 재확인**하며, 오프셋 집합이
`{997}`이 아니면 중단한다(가정으로 덮지 않기 위해).

### 2.2 결과 — 평가셋별 (seed 0 대표값; 3 seed 전체는 CSV)

| 평가셋 | 검사 문항 | 풀 중복 | 실소비(freq) | 실소비(window) | 정확일치 | 번역대응 | 동일문서 | 미확인 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **FLORES bpb** (언어당) | 60 | **0** | **0** | **0** | 0 | 0 | **1** | 0 |
| **FLORES unseen**(E3, 언어당) | 60 | **0** | **0** | **0** | 0 | 0 | 1 | 0 |
| **XL-Sum**(언어당 5–21 문서) | 75 | **0** | **0** | **0** | 0 | 0 | **0** | 0 |
| **SIB-200 test**(언어당) | 204 | **106** | 106¹ | 15–76² | 106³ | 106³ | 106 | 0 |
| **SIB-200 demonstrations** | 14 | **6** | 6 | 6 | 6 | 0 | 6 | 0 |
| **Belebele passages**(언어당) | 488 | **202** | 202¹ | 89–201² | 202³ | 202³ | 3–37 | 0 |
| **Belebele 문항 환산** | 900 | **351** | 351 | 154–349² | — | — | — | 0 |

¹ 단일 언어 조건. `mixed_5lang`은 95(SIB)/200–201(Belebele).
² 조건·언어·seed에 따라 변동(§1.2의 절단 비대칭 때문). 예: SIB eng_Latn 실소비 윈도우 중복은
  english_only 70/67/69, korean_only 27/34/28, bengali_only 15/18/17 (seed 0/1/2).
³ **자기 언어 조건에서는 정확 일치**, 나머지 조건에서는 같은 source ID의 **번역 대응**.
  FLORES-204언어가 행 단위 병렬이므로 행 번호가 곧 semantic/source ID다.

**FLORES bpb의 "동일문서 1"의 정체**: devtest는 기사 단위로 정렬돼 있고, 행 60 경계를
가로지르는 기사는 **정확히 1개**다 — `en.wikinews.org/wiki/Funeral_of_Pope_John_Paul_II_takes_place`
(행 59, 60, 61, 62). 행 59는 평가에, 60–62는 calibration 풀에 들어간다. 행 0..59 안에
완전히 들어가는 기사는 18개다. 즉 **평가 60문장 중 59문장은 문서 수준에서도 완전히 분리**되고
1문장만 기사를 공유한다. 이것을 exact 문장 노출과 같이 세지 않는다.

### 2.3 placebo arm 대 원 arm

| 비교 수준 | arm A | arm B | 교집합 | arm B 대비 |
|---|---:|---:|---:|---:|
| 풀 | 952행 | 476행 | **476** | **100.0%** |
| expert-freq 실소비 | 952행 | 476행 | **476** | **100.0%** |
| Fisher/whitening 윈도우 | 163–690행 | 158–445행 | 25–309 | 15.8%–72.2% |

(5개 언어 × 3 seed 전부 동일 패턴, `placebo_arm_overlap.csv`)

### 2.4 평가셋 내부 중복

`eval_internal_duplication.json`: FLORES 각 언어 1012문장은 **전부 유일**하고, 평가 60문장과
calibration 풀 952문장 사이에 **정규화 문자열 충돌도 0**이다(즉 인덱스 분리가 문자열 수준에서도
성립한다). Belebele는 **488 passage에 900문항**이 달려 있어 한 passage가 평균 1.84문항을
공급한다 — 그래서 clean subset은 passage 단위로 제거한다.

---

## 3. 확인된 사실 — 기존 XL-Sum 평가(§5.5, Appendix K, Table 31)의 재검증

**이 평가는 이미 존재했고, 이 작업은 그것을 새 실험으로 표현하지 않는다.**

### 3.1 provenance

| 항목 | 실측 |
|---|---|
| 코퍼스 | `csebuetnlp/xlsum` v2.0, split `test`, 본문(`text`)만, title/summary 미사용 |
| 구축 | `scripts/e2_build_ood_corpus.py` → `data/ood_xlsum/{lang}.txt` + `manifest.json` |
| 예산 | `50 * 1024 = **51,200 bytes**`(=50 KiB). 캡션의 "약 50 KB"는 50 **KiB**를 가리킨다 |
| 실현 바이트 | EN 50,990 / KO 51,004 / ZH 51,032 / SW 50,853 / BN 50,751 (예산의 99.12–99.67%) |
| 문서 수 | EN 20 / KO 11 / ZH 18 / SW 21 / **BN 5** |
| 무결성 | 5개 언어 **sha256 전부 manifest와 일치** — 디스크 코퍼스는 2026-08-03 생성본과 비트 동일 |
| 평가 | `scripts/e2e3_ood_unseen_eval.py`, batch 8, 세그먼트 상한 288자, tokenizer truncation 미사용 |
| 원시 산출물 | `phase1/{cond}/seed{s}/scale_*/eval_ood_unseen.json` (`e2_ood` + `e3_unseen` 한 파일) |

### 3.2 Table 31 재현 — 유효 캐시 재사용, GPU 재실행 없음

기존 산출물이 완전해서 **산식만 재계산**했다(전체 GPU 재실행 중복 수행하지 않음).

| 조건 | EN | ZH | KO | SW | BN |
|---|---:|---:|---:|---:|---:|
| english_only | +6.93 | +16.63 | +106.70 | +35.88 | +93.19 |
| korean_only | +14.75 | +14.53 | +4.62 | +41.37 | +27.52 |
| mixed_5lang (Balanced-char) | +10.90 | +10.87 | +7.69 | +4.23 | +3.38 |
| balanced_tok_mono (Balanced-token) | +8.56 | +9.94 | +10.19 | +2.14 | +5.56 |

**20개 값 전부 소수점 2자리까지 문서값과 일치**. baseline bpb 9개(XL-Sum 5 + unseen 4)도 전부 일치.

- Spearman ρ(FLORES ↔ XL-Sum, 언어 안에서 4조건 순위): 15셀 평균 **0.9867**, 범위 0.80–1.00,
  **EN seed0만 0.80** — 문서의 "평균 +0.987, 범위 0.80–1.00, EN seed0만 0.80"과 일치.
  ties는 평균 순위, 코퍼스 간 Δbpb% **크기**는 비교하지 않았다.
- 부호 일관성: Table 31 범위 4조건 × 5언어 × 3seed = **60/60 셀**에서 FLORES와 XL-Sum의
  Δbpb% 부호가 같다(둘 다 양수 = 압축이 손해).
  지시문이 언급한 "6/6 sign consistency"는 이 60/60과 다른 집계 단위이며, 그 정의를 논문에서
  확정하기 전에는 **재확인되었다고 쓰지 않는다**.

### 3.3 Table 31의 범위 — 4조건이지 전 조건이 아니다

Table 31은 **English / Korean / Balanced-character / Balanced-token** 네 조건이다.
Chinese·Swahili·Bengali 단일 언어 arm은 **외부 코퍼스에서 평가된 적이 없었다**.
이번 작업이 그 9셀(3조건 × 3seed)을 채운다 — §5 참조.

---

## 4. 원논문 대비 정정 사항

1. **§4의 분리 주장은 FLORES bpb에만 해당한다.**
   논문이 "held-out 문장" 분리를 말할 때 그 규약은 FLORES 행 0..59 대 60..1011이다.
   **SIB-200과 Belebele는 이 규약의 보호를 전혀 받지 않는다** — 둘 다 FLORES-200에서 파생된
   데이터셋이고, 자기 split을 FLORES의 60-행 컷과 맞춘 적이 없다.
   실측: SIB-200 test 204문항 중 **106문항(52.0%)**, Belebele 900문항 중 **351문항(39.0%)**이
   calibration 풀 문장에서 나왔다. 이 문장들은 expert-frequency 단계에서 **전량 소비**됐다.
   → downstream 수치를 인용할 때 이 사실을 함께 적어야 한다.

2. **placebo(Appendix B.2)의 "비중복 half" 서술은 arm B 쪽만 참이다.**
   arm B는 풀의 뒤쪽 절반만 쓰는 것이 맞지만, **arm A는 풀 전체(952문장)를 쓴다**.
   expert-frequency 경로가 풀을 전량 소비하므로 **arm B의 텍스트는 100%가 arm A에도 소비됐다**.
   저장소 주석(`phase1_calib_data.build_condition_sentences`)은 이미 "arm A의 랜덤 표집이
   건드린 윈도우와의 중복까지 보장하지는 않는다"고 적고 있으나, 실제 상황은 그보다 강하다 —
   윈도우 수준 중복도 15.8–72.2%이고, 문장 수준 중복은 **결정론적으로 100%**다.
   → placebo를 "독립 재표집 noise floor"로 부를 때 이 한계를 명시해야 한다.

3. **Table 31 캡션의 "약 50 KB"는 50 KiB(51,200 bytes) 예산이고, 실현값은 그보다 작다**
   (50,751–51,032, 예산의 99.1–99.7%). 언어 간 실현 바이트 편차는 예산의 0.55%로 작다.

4. **"whole documents"는 참이지만 표집기는 예산 초과 문서를 건너뛰고 끝까지 훑는다.**
   `e2_build_ood_corpus.sample()`은 첫 초과에서 멈추지 않고 셔플된 전체 목록을 스캔하며
   남은 예산에 들어가는 문서를 계속 담는다(EN은 11,535건을 스캔해 11,515건을 건너뛰고 20건 채택).
   결과적으로 **짧은 문서 쪽으로 편향**된다. 코드 docstring("예산을 넘기는 마지막 문서는
   버린다")은 "첫 초과에서 중단"으로 읽히므로 **문서와 코드가 어긋난다**.
   실질 영향: 문서 수가 매우 적다(특히 **Bengali 5건**) — 문서 수준 paired bootstrap을
   이 코퍼스에서 하면 Bengali의 유효 표본은 5다.

---

## 5. 미확인 사항 (UNKNOWN — 0으로 출력하지 않았다)

| 항목 | 상태 | 이유 / 비용 |
|---|---|---|
| 모델 **사전학습** 오염 | **UNKNOWN** | 이 감사는 calibration↔평가 분리만 본다. DeepSeek-MoE-16B의 사전학습 코퍼스에 FLORES/SIB/Belebele/XL-Sum이 들어갔는지는 확인 수단이 없다. **"완전 분리"로 확대 해석 금지.** |
| SIB-200 / Belebele **clean subset 재평가** | **미실행 (승인자 판단으로 보류, 2026-09-16)** | 기존 `eval_sib200.json`은 문항별 예측을 저장하지 않는다(`acc` + `pred_distribution` + 표본 10개뿐). `eval_belebele.json`도 집계값만 저장하며 기존 실행은 `limit=200`·3언어라 커버리지가 이미 부분적이다. 따라서 CPU 재계산은 **원리적으로 불가**하고 GPU 재평가가 필요하다(SIB만 19셀 × ~28분 ≈ 9시간, Belebele는 그 이상). **이번 실행에서는 돌리지 않기로 확정**했다. 따라서 이 보고는 원본 정확도를 그대로 보존하고 clean subset의 **정의·제거 개수·label 분포 변화까지만** 낸다 — clean 정확도 수치는 존재하지 않으며 추정하지 않았다. |
| placebo arm의 **외부(XL-Sum) 평가** | **불가** | placebo 5조건의 `fisher_processed.pt`/`svd_scale_processed.pt`가 2026-08-02 디스크 회수로 **삭제**됐다(조건당 잔존 ~48–72KB, eval json + expert_freq만). 재평가하려면 **재압축**이 필요하다(조건·seed당 ~62GB, freq+fisher+svd ≈ 4시간). 지시문이 재압축을 보류시켰으므로 실행하지 않았다. |
| `balanced_tok_mix` | **없음** | 결과 트리에 디렉터리 자체가 없다. 유효한 기존 모델이 없으므로 보조 조건에서 제외했다(이를 위한 신규 압축 없음). |
| **문서별 raw NLL** (2026-09-16 패치 이전 셀) | **UNAVAILABLE (백필 보류, 2026-09-16)** | 기존 `eval_ood_unseen.json`은 코퍼스 합계만 저장했다. 오늘 평가기를 **가산적으로** 패치해(§6) 패치 이후 실행된 셀은 문서별 NLL을 남긴다. `chinese_only/seed0`은 패치 직전에 기동돼 포함되지 않는다. 기존 12셀 + 그 1셀 백필(~27분 × 13 ≈ **6시간**)은 **돌리지 않기로 확정**했다. 덧붙여 이 코퍼스는 언어당 문서가 5~21개뿐이라(Bengali 5) document-level paired bootstrap의 유효 표본이 애초에 작다 — §4의 정정 4 참조. 미산출 셀은 0이나 추정으로 채우지 않았다. |
| 벤더 `get_scale.py` 경로 | **해당 없음** | 이 스크립트는 `random`을 시드하지 않아 윈도우 복원이 원리적으로 불가하지만, **공식 격자의 `scale_*` 셀은 이 스크립트가 만들지 않았다**(`phase1_svd_scale.py`가 만들었고 그쪽은 재현 가능). 다른 파이프라인이 벤더 경로를 쓴다면 그때는 UNKNOWN이 된다. |

---

## 6. 이번 작업에서 변경한 코드 (전부 가산적)

| 파일 | 변경 | 기존 수치 영향 |
|---|---|---|
| `scripts/e2e3_ood_unseen_eval.py` | `bpb_over(..., per_segment=False)` 인자 추가 + `eval_ood()`가 세그먼트를 문서에 귀속시켜 `per_document` 기록 | **없음(비트 동일)**. 세그먼트별 NLL은 `evaluate_sentences()`가 이미 반환하던 값을 버리지 않고 함께 반환할 뿐이고, 집계 산식·배치·세그먼트화·기존 반환 키는 하나도 바뀌지 않았다. 문서 귀속이 가능함(문서별 `segment()` 결과를 이어붙이면 전체 `segment()` 결과와 문자열 단위로 동일)은 5개 언어 전부 실측 확인했다. |
| `scripts/e2_grid_completion_0916.sh` | **신규** 드라이버 (누락 9셀) | 기존 드라이버 미변경 |
| `scripts/audit_calib_eval_overlap.py` | **신규** | 읽기 전용 |
| `scripts/make_clean_eval_ids_0916.py` | **신규** | 읽기 전용 |
| `scripts/make_audit_manifest_0916.py` | **신규** | 읽기 전용 |
| `scripts/make_external_bpb_grid_0916.py` | **신규** | 읽기 전용 |

기존 파일·체크포인트·placebo 출력은 **하나도 삭제하지 않았다**. 논문에 쓴 데이터를 새 표본으로
바꾸지 않았다(XL-Sum 코퍼스는 sha256으로 동일성 확인, 재샘플링 없음).
신규 드라이버는 원 드라이버 대비 **두 곳만 다르며 둘 다 헤더에 명시**했다: (1) 디스크 문턱
500G→40G(이 잡은 디스크에 ~4KB JSON만 쓴다. 현재 여유 325G), (2) 대상이 누락 9셀이고
baseline은 재계산하지 않는다(코퍼스·설정당 1회 원칙).

---

## 7. 결과 해석 — 세 질문을 분리해서

### Q1. 평가 문장을 calibration에서 직접 또는 번역 대응 형태로 봤는가?

| 평가 | 답 |
|---|---|
| **FLORES bpb (주 지표)** | **아니오.** 문장 수준·윈도우 수준 중복 0. 60문장 중 59문장은 문서 수준에서도 분리. |
| **FLORES unseen (E3)** | **아니오.** 동일. |
| **XL-Sum (외부)** | **아니오.** 모든 축에서 0. |
| **SIB-200 test** | **예.** 204 중 106(52.0%). 자기 언어 조건은 정확 일치, 나머지는 번역 대응. |
| **SIB-200 demonstrations** | **예.** 14 중 6. (test 누출과 구별해 별도 보고) |
| **Belebele** | **예.** 900문항 중 351(39.0%), passage 488 중 202. |

### Q2. FLORES calibration으로 만든 모델의 조건 간 순위가 외부 평가 코퍼스에서도 유지되는가?

**기존 4조건 범위에서는 유지된다** — Spearman ρ 평균 0.9867(15셀), 최소 0.80, 부호 60/60 일치.
이 값은 원시 산출물과 집계 정의를 확인한 뒤 재계산한 것이므로 "재확인되었다"고 쓸 수 있다.
**6조건 범위의 답은 이 보고 시점에 아직 없다** — 9셀 평가가 실행 중이며, 결과는
`external_bpb_grid.csv` / `ranking_and_regret.csv`에 4조건 비교와 **분리된 열**로 들어간다.
두 비교를 섞어 쓰지 말 것.

### Q3. 다른 calibration 코퍼스에서도 같은 효과가 생기는가?

**이번 작업은 Q3에 답하지 못한다.** 바꾼 것은 **평가** 코퍼스뿐이고 calibration은 여전히
FLORES-200 devtest다. calibration 코퍼스 자체를 바꿔 다시 압축하는 것은 별개 작업이며
이번 실행 범위가 아니다(지시문 §3.3 말미와 동일).

---

## 8. 산출물 경로

모두 `<repo>/analysis/data_audit_0916/` 아래:

| 파일 | 내용 |
|---|---|
| `artifact_manifest.json` | 모든 입력·모델·revision·protocol·실행 명령·git 상태·조건별 산출물 실사 |
| `overlap_summary.csv` | 900행. 조건×seed×평가셋×언어별 분모(`n_items`,`n_questions`)와 exact/번역대응/부분노출/동일문서/미확인 열 |
| `overlap_evidence.jsonl` | 91,605행. 지시문이 요구한 전 필드. overlap 신호가 있거나 상태가 VERIFIED가 아닌 항목만 기록(분모는 summary가 보유) |
| `clean_eval_ids.json` | SIB/Belebele clean subset(primary·strict) + 생성 규칙 + label 분포 |
| `placebo_arm_overlap.csv` | arm A↔B 중복(풀/윈도우/freq 수준) |
| `eval_internal_duplication.json` | 평가셋 내부 중복, FLORES 평가↔풀 문자열 충돌 검사 |
| `calibration_reconstruction.json` | 63개 조건/seed 재구성: 윈도우 char span 64개, 절단 위치, 소비 행 수 |
| `external_bpb_grid.csv` | 조건×seed×언어 bpb / Δbpb% (XL-Sum + FLORES) |
| `external_bpb_per_document.jsonl` | 문서별 raw NLL / bytes / tokens |
| `external_bpb_missing_cells.json` | 미산출 셀 명시 |
| `ranking_and_regret.csv` | 순위·own-language contrast·max regret·worst-language damage·Spearman(비교 조건 집합 동봉) |
| `data_audit_report.md` | 이 문서 |
| `rebuttal_data_paragraphs.md` | 실제 해당 분기에 맞는 영어 문단 |

로그: `logs/audit_overlap_0916.log`, `logs/e2_grid_completion_0916.log`
