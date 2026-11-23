"""Calibration <-> evaluation overlap audit (2026-09-16 지시문 §3). CPU 전용.

무엇을 하는가
-------------
논문 §4(p.3)의 "calibration과 evaluation은 분리돼 있다"는 주장을, **문서 요약이 아니라
저장소 구현**을 실행해 검증한다. 세 가지를 구별해서 낸다:

  pool_overlap    -- 그 텍스트가 calibration **풀**에 들어 있었는가 (가능성)
  consumed_*      -- 실제 calibration 단계가 그 텍스트를 **먹었는가** (실제 노출 근거)
  parent_overlap  -- 같은 원문 기사(FLORES URL)를 공유하는가 (문서 수준, 문장 노출과 별개)

실제 소비 경로 (코드에서 확인, 2026-09-16)
------------------------------------------
압축 1셀은 calibration 텍스트를 **세 경로**로 먹는다. 세 경로 전부 phase1_merge_eval.
merge_condition()이 읽는 산출물을 만든다(expert_freq / fisher_processed.pt /
svd_scale_processed.pt) -- 즉 셋 다 주 결과에 실제로 반영된다.

  (1) expert-frequency : D2MoE/preprocess/get_expert_freq.py
      dataset.shuffle(seed=seed).select(range(min(2000, N))) -- **문장 단위**, 전량.
      단일 언어 조건의 풀은 952문장이라 2000 미만 -> **풀 전체가 소비된다**.
  (2) Fisher           : scripts/phase1_fisher.py build_samples()
      random.Random(seed+1)로 joined text 위에서 64개 x 2048자 윈도우, 512토큰 절단.
  (3) whitening        : scripts/phase1_svd_scale.py
      **(2)와 같은 build_samples를 무수정 import**, 같은 n_samples=64/seqlen=512/seed.
      -> 윈도우 집합이 Fisher와 **동일**하다. (벤더 D2MoE/preprocess/get_scale.py는
         random을 시드하지 않아 재현 불가하지만, 공식 격자의 scale_* 셀은 그 스크립트가
         아니라 phase1_svd_scale.py가 만든 것이다 -- phase1_merge_eval.py 주석 및
         local_gpu_driver.sh:203 참조.)

따라서 (1) 때문에 **단일 언어 조건에서는 pool == consumed** 가 성립한다. 이 감사는 그
붕괴를 감추지 않고 경로별로 따로 적는다.

출처 ID 복원
------------
  FLORES  : israel/flores-parallel test = FLORES-200 **devtest** 1012행. 행 r의 출처 ID는
            devtest row r (데이터셋 'id' 컬럼 = r+1), parent = 'URL' 컬럼(281개 기사).
            204개 언어가 같은 행 = 같은 원문의 번역 -> 행 번호가 곧 semantic/source ID다.
  SIB-200 : 'index_id'가 FLORES 통합 색인이다. **실측 확인**: index_id - devtest_row == 997
            (유일값)이므로 index_id >= 997 => devtest row (index_id-997),
            index_id < 997 => FLORES **dev** (이 프로젝트가 한 번도 건드리지 않는 split).
  Belebele: 'link'가 원문 기사 URL, 'flores_passage'가 FLORES 문장들로 구성된 문단이다.
            passage를 구성 문장으로 되돌리기 위해 그 언어의 devtest 1012문장 중
            passage 안에 **부분 문자열로 들어 있는 것**을 찾는다(정확 일치 기준).
  XL-Sum  : data/ood_xlsum/manifest.json의 문서 id/url.

정규화 규칙
-----------
raw = UTF-8 원문 그대로. norm = Unicode NFC + 연속 공백 1칸 축약 + strip. 그 이상은 하지
않는다(단어/문자 제거로 다른 문장을 같게 만들지 않는다). 두 해시를 모두 낸다.

evidence 파일 방침
------------------
overlap_evidence.jsonl은 **overlap 신호가 하나라도 있거나 verification_status가 VERIFIED가
아닌 항목만** 적는다(전 조건 x 전 항목을 적으면 11만 행이 되고 대부분이 같은 내용의
반복이다). 분모(검사한 전체 문항 수)는 overlap_summary.csv가 전부 들고 있으므로,
"검사하지 못한 것을 0으로 출력"하는 일은 생기지 않는다 -- 미확인은 UNKNOWN으로 남는다.

사용법:
    python audit_calib_eval_overlap.py [--out-dir ../analysis/data_audit_0916]
"""
import argparse
import csv
import hashlib
import json
import random
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import phase1_calib_data as PCD  # noqa: E402

FLORES_SOURCE, FLORES_SPLIT, FLORES_PREFIX = PCD.SOURCE, PCD.SPLIT, PCD.COL_PREFIX
EVAL_LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
UNSEEN_LANGS = ["arb_Arab", "hin_Deva", "tur_Latn", "rus_Cyrl"]
EVAL_N = PCD.EVAL_N_SENTENCES                      # 60, held-out for FLORES bpb
N_SAMPLES, SEQLEN = 64, 512                        # local_gpu_driver.sh:192/203
WINDOW_CHARS = SEQLEN * 4                          # build_samples(): seqlen*4 = 2048
FREQ_MAX_SAMPLES = 2000                            # phase1_run_freq_and_scale.py default
SIB_INDEX_OFFSET = 997                             # 실측 확정, 아래 verify_sib_offset()가 재확인

MAIN_CONDITIONS = ["english_only", "chinese_only", "korean_only", "swahili_only",
                   "bengali_only", "mixed_5lang"]  # 논문 주 6조건 (EN/ZH/KO/SW/BN + Balanced-char)
AUX_CONDITIONS = ["balanced_tok_mono"]             # Balanced-token, 보조
PLACEBO_CONDITIONS = PCD.PLACEBO_CONDITIONS
SEEDS = [0, 1, 2]


def norm(s):
    return " ".join(unicodedata.normalize("NFC", s).split())


def h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- FLORES ----
def load_flores():
    from datasets import load_dataset
    ds = load_dataset(FLORES_SOURCE, split=FLORES_SPLIT)
    meta = {"n_rows": len(ds), "id_col": list(ds["id"]), "url": list(ds["URL"]),
            "domain": list(ds["domain"]), "topic": list(ds["topic"])}
    sents = {lang: [s for s in ds[f"{FLORES_PREFIX}{lang}"]] for lang in EVAL_LANGS + UNSEEN_LANGS}
    return meta, sents


def build_row_index(sents):
    """raw/normalized 문자열 -> [(lang, row)]. 중복 문자열이 있으면 리스트가 길어진다."""
    raw_idx, norm_idx = defaultdict(list), defaultdict(list)
    for lang, col in sents.items():
        for r, s in enumerate(col):
            raw_idx[s].append((lang, r))
            norm_idx[norm(s)].append((lang, r))
    return raw_idx, norm_idx


# -------------------------------------------------- calibration 재구성 ----
def consumed_tokens_prefix(tokenizer, text, seqlen):
    """512토큰 절단이 실제로 소비한 **문자 prefix 길이**를 돌려준다.
    slow tokenizer에는 offset_mapping이 없으므로 이진탐색으로 정확히 찾는다."""
    n_tok = len(tokenizer(text)["input_ids"])
    if n_tok <= seqlen:
        return len(text), n_tok, False
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer(text[:mid])["input_ids"]) <= seqlen:
            lo = mid
        else:
            hi = mid - 1
    return lo, len(tokenizer(text[:lo])["input_ids"]), True


def reconstruct_condition(condition, seed, raw_idx, tokenizer):
    """조건/seed 하나의 (a) 풀 (b) Fisher+whitening 윈도우가 실제 소비한 문장
    (c) expert-freq가 소비한 문장을, 전부 (lang,row) 수준으로 되돌린다."""
    sentences = PCD.build_condition_sentences(condition, seed)
    joined = "\n\n".join(sentences)

    # 문장별 char span (join이 "\n\n" 2자를 넣는다)
    spans, pos = [], 0
    for s in sentences:
        spans.append((pos, pos + len(s)))
        pos += len(s) + 2

    prov = []  # 각 문장 원소의 (lang,row) 후보
    for s in sentences:
        prov.append(raw_idx.get(s, []))

    pool = set()
    for cands in prov:
        pool.update(cands)

    # --- (b) Fisher/whitening 윈도우: build_samples()와 **같은 산술** ---
    rng = random.Random(seed + 1)
    max_start = max(1, len(joined) - WINDOW_CHARS - 1)
    consumed_win, win_records = set(), []
    for w in range(N_SAMPLES):
        i = rng.randint(0, max_start)
        j = i + WINDOW_CHARS
        window = joined[i:j]
        keep_chars, n_tok, truncated = consumed_tokens_prefix(tokenizer, window, SEQLEN)
        cs, ce = i, i + keep_chars          # 512토큰 절단 후 실제 소비 span
        touched = set()
        # span overlap: 이진탐색 대신 선형(문장 수가 수천 수준이라 충분)
        for k, (a, b) in enumerate(spans):
            if b <= cs:
                continue
            if a >= ce:
                break
            for c in prov[k]:
                touched.add(c)
        consumed_win.update(touched)
        win_records.append({"window": w, "char_start": i, "char_end_raw": j,
                            "char_end_after_512tok": ce, "n_tokens_raw_window": n_tok,
                            "truncated": truncated, "n_sentences_touched": len(touched)})

    # --- (c) expert-frequency: get_expert_freq.py의 shuffle+select 재현 ---
    from datasets import Dataset
    ds = Dataset.from_dict({"text": sentences})
    sel = ds.shuffle(seed=seed).select(range(min(FREQ_MAX_SAMPLES, len(ds))))
    consumed_freq = set()
    for s in sel["text"]:
        consumed_freq.update(raw_idx.get(s, []))

    return {"n_sentences": len(sentences), "joined_chars": len(joined),
            "pool": pool, "consumed_window": consumed_win, "consumed_freq": consumed_freq,
            "windows": win_records,
            "calib_text_md5_12": hashlib.md5("".join(
                joined[r["char_start"]:r["char_start"] + WINDOW_CHARS] for r in win_records
            ).encode()).hexdigest()[:12]}


# ------------------------------------------------------------ eval sets ----
def verify_sib_offset(sents):
    """SIB index_id -> devtest row 오프셋을 **가정하지 않고 실측 재확인**한다."""
    from datasets import load_dataset
    pos = {s: r for r, s in enumerate(sents["eng_Latn"])}
    offs = set()
    sib = load_dataset("Davlan/sib200", "eng_Latn")
    for sp in ["train", "validation", "test"]:
        for row in sib[sp]:
            t = row["text"].strip()
            if t in pos:
                offs.add(row["index_id"] - pos[t])
    return sorted(offs)


def load_eval_sets(meta, sents, raw_idx, norm_idx):
    """모든 평가 항목을 {dataset: {lang: [item]}} 로. item은 FLORES devtest 행 집합을 들고 있다."""
    from datasets import load_dataset
    out = {}

    # FLORES bpb 주 평가 (rows 0..59)
    out["flores_bpb"] = {lang: [{"eval_id": f"{lang}/row{r}", "text": sents[lang][r],
                                 "flores_rows": {r}, "parent_id": meta["url"][r],
                                 "status": "VERIFIED"}
                                for r in range(EVAL_N)] for lang in EVAL_LANGS}
    # FLORES unseen (E3)
    out["flores_unseen"] = {lang: [{"eval_id": f"{lang}/row{r}", "text": sents[lang][r],
                                    "flores_rows": {r}, "parent_id": meta["url"][r],
                                    "status": "VERIFIED"}
                                   for r in range(EVAL_N)] for lang in UNSEEN_LANGS}

    # SIB-200 test
    sib_out, sib_demo = {}, None
    for lang in EVAL_LANGS:
        ds = load_dataset("Davlan/sib200", lang)
        items = []
        for row in ds["test"]:
            iid = row["index_id"]
            if iid >= SIB_INDEX_OFFSET:
                r = iid - SIB_INDEX_OFFSET
                rows, status, parent = {r}, "VERIFIED", meta["url"][r]
            else:
                rows, status, parent = set(), "VERIFIED_FLORES_DEV", f"flores_dev/{iid}"
            items.append({"eval_id": f"{lang}/idx{iid}", "text": row["text"],
                          "flores_rows": rows, "parent_id": parent, "status": status,
                          "source_id": f"sib_index_id/{iid}"})
        sib_out[lang] = items
        if lang == "eng_Latn":
            sib_demo = reconstruct_sib_demos(ds, meta)
    out["sib200_test"] = sib_out
    out["sib200_demos"] = {"eng_Latn": sib_demo}

    # Belebele
    bel_out = {}
    for lang in EVAL_LANGS:
        ds = load_dataset("facebook/belebele", lang)["test"]
        col = sents[lang]
        by_passage = defaultdict(list)
        for row in ds:
            by_passage[(row["link"], row["flores_passage"])].append(row["question_number"])
        items = []
        for (link, passage), qnums in by_passage.items():
            rows = {r for r, s in enumerate(col) if s and s in passage}
            items.append({"eval_id": f"{lang}/{h(link + passage)}", "text": passage,
                          "flores_rows": rows, "parent_id": link,
                          "status": "VERIFIED" if rows else "VERIFIED_NO_FLORES_MATCH",
                          "n_questions": len(qnums), "source_id": f"belebele_link/{link}"})
        bel_out[lang] = items
    out["belebele_passages"] = bel_out

    # XL-Sum (외부 코퍼스)
    man = json.loads((SCRIPT_DIR.parent / "data" / "ood_xlsum" / "manifest.json").read_text())
    xl = {}
    for lang, info in man["languages"].items():
        text = (SCRIPT_DIR.parent / "data" / "ood_xlsum" / f"{lang}.txt").read_text(encoding="utf-8")
        docs = text.split("\n\n")
        xl[lang] = [{"eval_id": f"{lang}/doc{i}", "text": d, "flores_rows": set(),
                     "parent_id": f"xlsum/{info['xlsum_config']}/{info['tar_member']}",
                     "status": "VERIFIED", "source_id": f"xlsum_doc/{i}"}
                    for i, d in enumerate(docs)]
    out["xlsum_docs"] = xl
    return out


def reconstruct_sib_demos(ds, meta):
    """demonstration 선택은 **재구현하지 않고** phase1_sib200_eval.build_demonstrations()를
    그대로 호출한다(순수 CPU/결정론적, 모델 불필요). 재구현하면 셔플 순서가 어긋나
    실제로 쓰인 것과 다른 문장을 감사하게 된다."""
    import phase1_sib200_eval as SIB
    demos = []
    for ex in SIB.build_demonstrations():
        iid = ex["index_id"]
        r = iid - SIB_INDEX_OFFSET if iid >= SIB_INDEX_OFFSET else None
        demos.append({"eval_id": f"demo/idx{iid}", "text": ex["text"],
                      "flores_rows": {r} if r is not None else set(),
                      "parent_id": meta["url"][r] if r is not None else f"flores_dev/{iid}",
                      "status": "VERIFIED" if r is not None else "VERIFIED_FLORES_DEV",
                      "source_id": f"sib_index_id/{iid}"})
    return demos

# ------------------------------------------------------------ comparison ----
def classify(item, eval_lang, recon, calib_langs, url_of_row):
    """한 평가 항목 x 한 (조건,seed)의 관계를 판정한다.

    match_type 우선순위 (강한 것부터):
      exact_same_language   -- 같은 언어 같은 FLORES 행 (문자열 동일). 진짜 test 노출.
      translation_parallel  -- 같은 FLORES 행이 calibration에 **다른 언어로** 들어갔다.
                               FLORES는 병렬 코퍼스라 행 번호가 곧 원문 ID다.
      parent_document       -- 문장은 다르지만 같은 원문 기사(URL)를 공유한다.
      none
    """
    rows = item["flores_rows"]
    pool_pairs_exact, pool_pairs_trans = set(), set()
    win_exact, win_trans = set(), set()
    freq_exact, freq_trans = set(), set()
    for r in rows:
        for lang in calib_langs:
            key = (lang, r)
            in_pool = key in recon["pool"]
            in_win = key in recon["consumed_window"]
            in_freq = key in recon["consumed_freq"]
            same = (lang == eval_lang)
            if in_pool:
                (pool_pairs_exact if same else pool_pairs_trans).add(key)
            if in_win:
                (win_exact if same else win_trans).add(key)
            if in_freq:
                (freq_exact if same else freq_trans).add(key)

    # parent-document: 평가 항목의 원문 URL을 calibration 풀의 어떤 행이 공유하는가
    parent_rows = set()
    if item["parent_id"] and not item["parent_id"].startswith(("xlsum/", "flores_dev/")):
        cand = url_of_row.get(item["parent_id"], set())
        for r in cand - rows:
            if any((lang, r) in recon["pool"] for lang in calib_langs):
                parent_rows.add(r)

    if pool_pairs_exact:
        mt = "exact_same_language"
    elif pool_pairs_trans:
        mt = "translation_parallel"
    elif parent_rows:
        mt = "parent_document"
    else:
        mt = "none"
    return {
        "match_type": mt,
        "pool_overlap": bool(pool_pairs_exact or pool_pairs_trans),
        "consumed_window_overlap": bool(win_exact or win_trans),
        "consumed_freq_overlap": bool(freq_exact or freq_trans),
        "exact_same_language": bool(pool_pairs_exact),
        "translation_parallel": bool(pool_pairs_trans),
        "parent_document": bool(parent_rows),
        "matched_rows": sorted({r for _, r in (pool_pairs_exact | pool_pairs_trans)}),
        "matched_langs": sorted({l for l, _ in (pool_pairs_exact | pool_pairs_trans)}),
        "parent_rows": sorted(parent_rows),
    }


def condition_langs(condition):
    if condition in PCD.MIXED_LANGS:
        return list(PCD.MIXED_LANGS[condition])
    if condition in PCD.CHAR_BALANCED_MIXES:
        return list(PCD.MIXED_LANGS[condition])
    if condition in PCD.E1_TOK_CONDITIONS:
        return list(PCD.MIXED_LANGS["mixed_5lang"])
    return [PCD.LANG_OF_SIMPLE[condition]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR.parents[1] / "results" / "source_data" / "data_exposure_audit"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    print("[audit] loading FLORES + tokenizer ...")
    meta, sents = load_flores()
    raw_idx, norm_idx = build_row_index(sents)
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base", use_fast=False)

    url_of_row = defaultdict(set)
    for r, u in enumerate(meta["url"]):
        url_of_row[u].add(r)

    offs = verify_sib_offset(sents)
    print(f"[audit] SIB index_id - devtest_row offsets observed: {offs}")
    if offs != [SIB_INDEX_OFFSET]:
        raise SystemExit(f"[audit] SIB 오프셋 가정 위반: {offs} -- 중단(가정으로 덮지 않는다)")

    # 평가셋 내부 중복 / 동일 원문 파생 문항
    evals = load_eval_sets(meta, sents, raw_idx, norm_idx)
    dupes = {}
    for dsname, per_lang in evals.items():
        for lang, items in per_lang.items():
            texts = [norm(i["text"]) for i in items]
            n_dup = len(texts) - len(set(texts))
            parents = [i["parent_id"] for i in items]
            dupes[f"{dsname}/{lang}"] = {
                "n_items": len(items),
                "n_duplicate_texts_normalized": n_dup,
                "n_unique_parents": len(set(parents)),
                "n_items_sharing_a_parent": len(items) - len(set(parents)),
                "n_questions_total": sum(i.get("n_questions", 1) for i in items),
            }

    # FLORES 자체 무결성: eval(0..59)과 calib(60..1011)이 문자열 수준에서도 분리되는가
    flores_selfcheck = {}
    for lang in EVAL_LANGS:
        col = sents[lang]
        ev = {norm(s) for s in col[:EVAL_N]}
        ca = {norm(s) for s in col[EVAL_N:]}
        flores_selfcheck[lang] = {
            "n_eval": EVAL_N, "n_calib_pool": len(col) - EVAL_N,
            "normalized_string_collisions_eval_vs_calibpool": len(ev & ca),
            "duplicate_normalized_strings_in_full_column": len(col) - len({norm(s) for s in col}),
        }

    # ---- 조건/seed 재구성 ----
    all_conditions = MAIN_CONDITIONS + AUX_CONDITIONS + PLACEBO_CONDITIONS
    recons = {}
    for cond in all_conditions:
        for seed in SEEDS:
            print(f"[audit] reconstructing {cond}/seed{seed} ...", flush=True)
            recons[(cond, seed)] = reconstruct_condition(cond, seed, raw_idx, tokenizer)

    # ---- arm A vs placebo arm B ----
    placebo_rows = []
    for pb in PLACEBO_CONDITIONS:
        arm_a = pb[:-2]
        for seed in SEEDS:
            A, B = recons[(arm_a, seed)], recons[(pb, seed)]
            for field in ["pool", "consumed_window", "consumed_freq"]:
                inter = A[field] & B[field]
                placebo_rows.append({
                    "arm_a": arm_a, "arm_b": pb, "seed": seed, "level": field,
                    "n_arm_a": len(A[field]), "n_arm_b": len(B[field]),
                    "n_intersection": len(inter),
                    "pct_of_arm_b": round(100.0 * len(inter) / max(1, len(B[field])), 3),
                })

    # ---- 본 비교 ----
    summary_rows, evidence = [], []
    for cond in all_conditions:
        clangs = condition_langs(cond)
        for seed in SEEDS:
            recon = recons[(cond, seed)]
            for dsname, per_lang in evals.items():
                for lang, items in per_lang.items():
                    c = defaultdict(int)
                    c["n_items"] = len(items)
                    c["n_questions"] = sum(i.get("n_questions", 1) for i in items)
                    for item in items:
                        if item["status"] not in ("VERIFIED", "VERIFIED_FLORES_DEV"):
                            c["n_unknown_provenance"] += 1
                        res = classify(item, lang, recon, clangs, url_of_row)
                        c[f"mt_{res['match_type']}"] += 1
                        c["n_pool_overlap"] += res["pool_overlap"]
                        c["n_consumed_window_overlap"] += res["consumed_window_overlap"]
                        c["n_consumed_freq_overlap"] += res["consumed_freq_overlap"]
                        c["n_exact_same_language"] += res["exact_same_language"]
                        c["n_translation_parallel"] += res["translation_parallel"]
                        c["n_parent_document"] += res["parent_document"]
                        if res["match_type"] != "none" or item["status"] not in ("VERIFIED", "VERIFIED_FLORES_DEV"):
                            evidence.append({
                                "model_id": "deepseek-ai/deepseek-moe-16b-base",
                                "calibration_condition": cond, "calibration_seed": seed,
                                "placebo": cond in PLACEBO_CONDITIONS,
                                "eval_dataset": dsname, "eval_language": lang,
                                "eval_id": item["eval_id"],
                                "source_id": item.get("source_id",
                                                      f"flores_devtest_row/{sorted(item['flores_rows'])}"),
                                "parent_id": item["parent_id"],
                                "match_type": res["match_type"],
                                "pool_overlap": res["pool_overlap"],
                                "consumed_window_overlap": res["consumed_window_overlap"],
                                "consumed_freq_overlap": res["consumed_freq_overlap"],
                                "matched_span": {"flores_rows": res["matched_rows"],
                                                 "calib_langs": res["matched_langs"],
                                                 "parent_rows": res["parent_rows"]},
                                "n_questions_affected": item.get("n_questions", 1),
                                "evidence_path": "analysis/data_audit_0916/artifact_manifest.json",
                                "verification_status": item["status"],
                            })
                    row = {"model_id": "deepseek-ai/deepseek-moe-16b-base",
                           "calibration_condition": cond, "calibration_seed": seed,
                           "placebo": cond in PLACEBO_CONDITIONS,
                           "eval_dataset": dsname, "eval_language": lang}
                    row.update({k: c[k] for k in sorted(c)})
                    summary_rows.append(row)

    # ---- 쓰기 ----
    (out_dir / "overlap_evidence.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in evidence), encoding="utf-8")

    fields = sorted({k for r in summary_rows for k in r})
    lead = ["calibration_condition", "calibration_seed", "placebo", "eval_dataset", "eval_language",
            "n_items", "n_questions"]
    fields = lead + [f for f in fields if f not in lead]
    with (out_dir / "overlap_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, 0) for k in fields})

    with (out_dir / "placebo_arm_overlap.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(placebo_rows[0]))
        w.writeheader(); w.writerows(placebo_rows)

    (out_dir / "eval_internal_duplication.json").write_text(
        json.dumps({"eval_set_internal": dupes, "flores_eval_vs_calibpool": flores_selfcheck},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    recon_report = {f"{c}/seed{s}": {
        "n_sentences": r["n_sentences"], "joined_chars": r["joined_chars"],
        "n_pool_flores_rows": len(r["pool"]),
        "n_consumed_window_flores_rows": len(r["consumed_window"]),
        "n_consumed_freq_flores_rows": len(r["consumed_freq"]),
        "window_char_spans": r["windows"],
    } for (c, s), r in recons.items()}
    (out_dir / "calibration_reconstruction.json").write_text(
        json.dumps(recon_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[audit] wrote {out_dir}/overlap_summary.csv ({len(summary_rows)} rows)")
    print(f"[audit] wrote {out_dir}/overlap_evidence.jsonl ({len(evidence)} rows)")
    print(f"[audit] wrote placebo_arm_overlap.csv, eval_internal_duplication.json, calibration_reconstruction.json")


if __name__ == "__main__":
    main()
