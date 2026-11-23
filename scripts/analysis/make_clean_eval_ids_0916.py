"""clean_eval_ids.json 생성 (2026-09-16 지시문 §3.3). CPU 전용.

규칙 (지시문 그대로, 하나도 완화하지 않는다)
--------------------------------------------
1. 제거 대상은 **모든 calibration 조건 x 모든 seed에 대해** overlap이 있는 평가 ID의
   **합집합**이다. 조건마다 다른 test를 쓰지 않기 위해서다.
2. FLORES는 병렬 코퍼스이므로 devtest 행 번호가 곧 source ID다. 한 언어에서 제거되는
   행은 **5개 평가 언어 전부에서** 같이 제거한다(번역 대응 제거 원칙).
3. 한 passage에 여러 문항이 걸린 경우(Belebele) **passage 전체**를 제거한다.
4. 두 가지 subset을 낸다:
     primary  -- 문장 수준 노출(exact + 번역대응)만으로 제거. parent-document 공유는
                 "exact test-sentence 노출과 같은 것으로 세지 않는다"는 지시문에 따라
                 여기서는 제거하지 않는다.
     strict   -- 위에 더해 calibration 풀과 **원문 기사(URL)를 공유**하는 항목도 제거.
5. label 분포 변화와 잔여 표본 크기를 함께 낸다. 특정 label이 사라지면 그 사실을 적는다.

주의: 이 스크립트는 **어떤 모델도 재평가하지 않는다**. 재평가는 GPU 작업이며 별도다.
"""
import collections
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import phase1_calib_data as PCD  # noqa: E402

OUT = SCRIPT_DIR.parents[1] / "results" / "source_data" / "data_exposure_audit"
EVAL_LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
SIB_OFFSET = 997
CALIB_POOL_FIRST_ROW = PCD.EVAL_N_SENTENCES   # 60 -- 이 행 이상이 calibration 풀


def main():
    from datasets import load_dataset
    fl = load_dataset(PCD.SOURCE, split=PCD.SPLIT)
    urls = list(fl["URL"])
    url_rows = collections.defaultdict(set)
    for r, u in enumerate(urls):
        url_rows[u].add(r)
    # calibration 풀에 실제로 들어간 devtest 행 집합 (모든 조건/seed 공통: 60..1011)
    pool_rows = set(range(CALIB_POOL_FIRST_ROW, len(fl)))
    # 풀 행과 기사를 공유하는 URL
    tainted_urls = {u for u, rs in url_rows.items() if rs & pool_rows}

    out = {
        "_purpose": "calibration 풀과 겹치는 평가 항목을 제거한 공통 clean subset (2026-09-16 감사)",
        "_removal_rule": {
            "basis": "FLORES-200 devtest 행 번호 = source ID. calibration 풀 = 행 60..1011 "
                     "(phase1_calib_data._pool_for: column[EVAL_N_SENTENCES:]). 이 집합은 "
                     "모든 조건·seed에서 동일하므로 합집합도 같다.",
            "translation_rule": "한 언어에서 제거되는 source ID는 5개 평가 언어 전부에서 제거한다.",
            "passage_rule": "Belebele는 passage 단위로 제거한다(한 passage에 여러 문항이 달려 있다).",
            "primary_vs_strict": "primary=문장 수준 노출만; strict=추가로 원문 기사(URL) 공유 항목까지 제거.",
            "NOT_removed": "parent-document 공유만 있는 항목은 primary에서 제거하지 않는다 "
                           "(지시문: 문서 공유를 exact 노출과 같이 세지 않는다).",
        },
        "calibration_pool_rows": {"first": CALIB_POOL_FIRST_ROW, "last": len(fl) - 1, "n": len(pool_rows)},
    }

    # ---------------- SIB-200 ----------------
    sib = {}
    for lang in EVAL_LANGS:
        ds = load_dataset("Davlan/sib200", lang)["test"]
        keep_p, keep_s, drop_p, drop_s = [], [], [], []
        cats_all, cats_p, cats_s = [], [], []
        for row in ds:
            iid, cat = row["index_id"], row["category"]
            cats_all.append(cat)
            r = iid - SIB_OFFSET if iid >= SIB_OFFSET else None
            in_pool = r is not None and r in pool_rows
            shares_url = r is not None and urls[r] in tainted_urls
            if in_pool:
                drop_p.append(iid); drop_s.append(iid)
            else:
                keep_p.append(iid); cats_p.append(cat)
                if shares_url:
                    drop_s.append(iid)
                else:
                    keep_s.append(iid); cats_s.append(cat)
        sib[lang] = {
            "n_total": len(ds),
            "primary": {"keep_index_ids": sorted(keep_p), "n_keep": len(keep_p), "n_dropped": len(drop_p),
                        "label_distribution": dict(sorted(collections.Counter(cats_p).items()))},
            "strict": {"keep_index_ids": sorted(keep_s), "n_keep": len(keep_s),
                       "n_dropped": len(ds) - len(keep_s),
                       "label_distribution": dict(sorted(collections.Counter(cats_s).items()))},
            "label_distribution_full": dict(sorted(collections.Counter(cats_all).items())),
        }
    # 번역 대응 원칙 검증: 5언어의 keep 집합이 동일해야 한다
    same_primary = len({tuple(sib[l]["primary"]["keep_index_ids"]) for l in EVAL_LANGS}) == 1
    same_strict = len({tuple(sib[l]["strict"]["keep_index_ids"]) for l in EVAL_LANGS}) == 1
    out["sib200_test"] = {"per_language": sib,
                          "keep_set_identical_across_languages": {"primary": same_primary, "strict": same_strict}}

    # ---------------- Belebele ----------------
    bel = {}
    for lang in EVAL_LANGS:
        ds = load_dataset("facebook/belebele", lang)["test"]
        col = list(fl[f"{PCD.COL_PREFIX}{lang}"])
        # passage -> 구성 FLORES 행
        passages = {}
        for row in ds:
            passages.setdefault((row["link"], row["flores_passage"]), []).append(row)
        keep_q_p, keep_q_s, drop_q_p, drop_q_s = [], [], [], []
        ans_p, ans_s, ans_all = [], [], []
        n_pass_keep_p = n_pass_keep_s = 0
        for (link, passage), qrows in passages.items():
            rows = {r for r, s in enumerate(col) if s and s in passage}
            in_pool = bool(rows & pool_rows)
            shares_url = link in tainted_urls
            qids = [f"{link}#q{q['question_number']}" for q in qrows]
            ans_all += [q["correct_answer_num"] for q in qrows]
            if in_pool:
                drop_q_p += qids; drop_q_s += qids
            else:
                keep_q_p += qids; ans_p += [q["correct_answer_num"] for q in qrows]; n_pass_keep_p += 1
                if shares_url:
                    drop_q_s += qids
                else:
                    keep_q_s += qids; ans_s += [q["correct_answer_num"] for q in qrows]; n_pass_keep_s += 1
        bel[lang] = {
            "n_questions_total": len(ds), "n_passages_total": len(passages),
            "primary": {"n_keep_questions": len(keep_q_p), "n_dropped_questions": len(drop_q_p),
                        "n_keep_passages": n_pass_keep_p,
                        "label_distribution": dict(sorted(collections.Counter(ans_p).items())),
                        "keep_question_ids": sorted(keep_q_p)},
            "strict": {"n_keep_questions": len(keep_q_s), "n_dropped_questions": len(drop_q_s),
                       "n_keep_passages": n_pass_keep_s,
                       "label_distribution": dict(sorted(collections.Counter(ans_s).items())),
                       "keep_question_ids": sorted(keep_q_s)},
            "label_distribution_full": dict(sorted(collections.Counter(ans_all).items())),
        }
    out["belebele_test"] = {"per_language": bel}

    # ---------------- FLORES bpb / XL-Sum ----------------
    out["flores_bpb"] = {
        "_finding": "행 0..59만 쓰고 calibration 풀은 60..1011이므로 제거 대상이 없다.",
        "n_removed": 0, "keep": "rows 0..59, 전 언어",
    }
    out["xlsum_ood"] = {
        "_finding": "FLORES와 출처가 겹치지 않는다(감사 결과 overlap 0). 제거 대상 없음.",
        "n_removed": 0,
    }

    (OUT / "clean_eval_ids.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT/'clean_eval_ids.json'}")
    for lang in EVAL_LANGS:
        s, b = out["sib200_test"]["per_language"][lang], out["belebele_test"]["per_language"][lang]
        print(f"  {lang}: SIB keep {s['primary']['n_keep']}/{s['n_total']} (strict {s['strict']['n_keep']}), "
              f"Belebele keep {b['primary']['n_keep_questions']}/{b['n_questions_total']} q "
              f"(strict {b['strict']['n_keep_questions']})")
    print("  SIB keep-set identical across languages:", out["sib200_test"]["keep_set_identical_across_languages"])


if __name__ == "__main__":
    main()
