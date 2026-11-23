"""외부(XL-Sum) bpb 격자 + 순위/regret 산출 (2026-09-16 지시문 §5). CPU 전용, 읽기만 한다.

입력: 각 셀의 eval_ood_unseen.json (XL-Sum OOD bpb + FLORES unseen bpb)와 eval_ppl.json
      (FLORES 5언어 bpb). baseline은 같은 평가 데이터·설정당 하나뿐이며 재계산하지 않는다.

산출:
  external_bpb_grid.csv          -- 조건 x seed x 언어 x 코퍼스의 bpb / Δbpb%
  external_bpb_per_document.jsonl-- 문서별 raw NLL / scored bytes / tokens
                                    (2026-09-16 패치 이후 실행된 셀만. 이전 셀은
                                     per_document가 저장되지 않았으므로 UNAVAILABLE로
                                     기록한다 -- 0이나 추정으로 채우지 않는다.)
  ranking_and_regret.csv         -- 조건 순위, own-language contrast, max regret,
                                    worst-language damage, FLORES<->XL-Sum Spearman rho

정의 (지시문 §5 그대로)
-----------------------
  bpb          = sum(scored-token NLL nats) / (ln2 * scored UTF-8 bytes).
                 문서별 bpb의 단순 평균이 **아니다**.
  Δbpb%        = 100*(compressed_bpb - base_bpb)/base_bpb, base는 **같은 코퍼스**의 원본값.
  own contrast = 그 언어의 own arm Δbpb% - 나머지 4개 **단일 언어** arm의 평균 Δbpb%.
                 Balanced와 placebo는 제외한다. 음수가 클수록 own-language 보호가 크다.
  max regret   = max_l [ Δbpb%(c,l) - min_{c' in COMPARISON_SET} Δbpb%(c',l) ].
                 비교 조건 집합을 결과에 같이 적는다(집합이 바뀌면 값이 바뀐다).
  worst damage = max_l Δbpb%(c,l).

Spearman은 **코퍼스 간 Δbpb% 크기를 직접 비교하지 않는다** -- 각 언어 안에서 조건
순위만 비교한다. ties는 평균 순위(scipy 기본)로 처리하고, 4조건(Table 31 범위)과
6조건 비교를 **따로** 낸다.
"""
import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = Path("results/deepseek")
OUT = ROOT / "analysis" / "data_audit_0916"

LANGS = ["eng_Latn", "zho_Hans", "kor_Hang", "swh_Latn", "ben_Beng"]   # 표기 통일: EN/ZH/KO/SW/BN
OWN_LANG = {"english_only": "eng_Latn", "chinese_only": "zho_Hans", "korean_only": "kor_Hang",
            "swahili_only": "swh_Latn", "bengali_only": "ben_Beng"}
SINGLE = ["english_only", "chinese_only", "korean_only", "swahili_only", "bengali_only"]
MAIN6 = SINGLE + ["mixed_5lang"]
AUX = ["balanced_tok_mono"]
TABLE31_SET = ["english_only", "korean_only", "mixed_5lang", "balanced_tok_mono"]  # 논문 Table 31 4조건
SEEDS = [0, 1, 2]


def cell_dir(cond, seed):
    return RESULTS_ROOT / cond / f"seed{seed}" / f"scale_{cond}_seed{seed}"


def spearman(a, b):
    """scipy 없이. ties는 평균 순위."""
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da and db else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=MAIN6 + AUX)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- baseline (코퍼스당 1회) ----
    base_ood = json.loads((RESULTS_ROOT / "baseline" / "eval_ood_unseen.json").read_text())
    base_flores = json.loads((RESULTS_ROOT / "baseline" / "eval_ppl.json").read_text())
    base = {
        "xlsum": {l: base_ood["e2_ood"]["results"][l]["bits_per_byte"] for l in LANGS},
        "flores": {l: base_flores[l]["bits_per_byte"] for l in LANGS},
        "flores_unseen": {l: v["bits_per_byte"] for l, v in base_ood["e3_unseen"]["results"].items()},
    }

    grid, perdoc, missing = [], [], []
    for cond in args.conditions:
        for seed in SEEDS:
            d = cell_dir(cond, seed)
            f_ood, f_ppl = d / "eval_ood_unseen.json", d / "eval_ppl.json"
            ood = json.loads(f_ood.read_text()) if f_ood.exists() else None
            ppl = json.loads(f_ppl.read_text()) if f_ppl.exists() else None
            if ood is None:
                missing.append({"condition": cond, "seed": seed, "missing": str(f_ood)})
            for lang in LANGS:
                row = {"condition": cond, "seed": seed, "language": lang,
                       "delta_ratio": 0.8, "pruning": "OFF", "eval_batch_size": 8}
                # XL-Sum
                if ood:
                    r = ood["e2_ood"]["results"][lang]
                    row.update({
                        "xlsum_bpb": r["bits_per_byte"],
                        "xlsum_base_bpb": base["xlsum"][lang],
                        "xlsum_delta_bpb_pct": 100.0 * (r["bits_per_byte"] - base["xlsum"][lang]) / base["xlsum"][lang],
                        "xlsum_n_bytes_scored": r["n_bytes_scored"], "xlsum_n_tokens_scored": r["n_tokens_scored"],
                        "xlsum_n_segments": r["n_segments"], "xlsum_status": "OK",
                        "xlsum_per_document": "AVAILABLE" if "per_document" in r else "UNAVAILABLE_not_recorded_by_run",
                    })
                    if "per_document" in r:
                        for doc in r["per_document"]:
                            perdoc.append({
                                "condition": cond, "seed": seed, "corpus": "xlsum", "language": lang,
                                "doc_index": doc["doc_index"], "nll_nats": doc["nll_nats"],
                                "n_tokens": doc["n_tokens"], "n_bytes": doc["n_bytes"],
                                "n_segments": doc["n_segments"],
                                "bpb_doc": (doc["nll_nats"] / math.log(2)) / doc["n_bytes"] if doc["n_bytes"] else None,
                                "source": str(f_ood),
                            })
                else:
                    row.update({"xlsum_bpb": None, "xlsum_delta_bpb_pct": None,
                                "xlsum_status": "MISSING", "xlsum_per_document": "MISSING"})
                # FLORES
                if ppl:
                    row.update({
                        "flores_bpb": ppl[lang]["bits_per_byte"],
                        "flores_base_bpb": base["flores"][lang],
                        "flores_delta_bpb_pct": 100.0 * (ppl[lang]["bits_per_byte"] - base["flores"][lang]) / base["flores"][lang],
                        "flores_status": "OK",
                    })
                else:
                    row.update({"flores_bpb": None, "flores_delta_bpb_pct": None, "flores_status": "MISSING"})
                grid.append(row)

    with (OUT / "external_bpb_grid.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0]))
        w.writeheader(); w.writerows(grid)
    (OUT / "external_bpb_per_document.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in perdoc), encoding="utf-8")

    # ---------------- ranking / regret ----------------
    def delta(corpus, cond, seed, lang):
        for r in grid:
            if r["condition"] == cond and r["seed"] == seed and r["language"] == lang:
                return r[f"{corpus}_delta_bpb_pct"]
        return None

    out_rows = []
    for corpus in ["xlsum", "flores"]:
        for cset_name, cset in [("main6", MAIN6), ("table31_4cond", TABLE31_SET), ("single5", SINGLE)]:
            for seed in SEEDS:
                avail = [c for c in cset if delta(corpus, c, seed, LANGS[0]) is not None]
                if len(avail) < 2:
                    continue
                for lang in LANGS:
                    vals = {c: delta(corpus, c, seed, lang) for c in avail}
                    best = min(vals.values())
                    order = sorted(avail, key=lambda c: vals[c])
                    for c in avail:
                        out_rows.append({
                            "metric": "per_language_rank", "corpus": corpus, "comparison_set": cset_name,
                            "conditions_in_set": "|".join(avail), "seed": seed, "language": lang,
                            "condition": c, "delta_bpb_pct": round(vals[c], 4),
                            "rank_best_is_1": order.index(c) + 1,
                            "regret_pct_vs_best_in_set": round(vals[c] - best, 4),
                            "value": "", "note": "",
                        })
            # 조건 수준 요약: max regret / worst-language damage
            for seed in SEEDS:
                avail = [c for c in cset if delta(corpus, c, seed, LANGS[0]) is not None]
                if len(avail) < 2:
                    continue
                for c in avail:
                    regrets, dmg = [], []
                    for lang in LANGS:
                        vals = [delta(corpus, c2, seed, lang) for c2 in avail]
                        v = delta(corpus, c, seed, lang)
                        regrets.append(v - min(vals)); dmg.append(v)
                    out_rows.append({"metric": "max_regret", "corpus": corpus, "comparison_set": cset_name,
                                     "conditions_in_set": "|".join(avail), "seed": seed, "language": "ALL",
                                     "condition": c, "delta_bpb_pct": "", "rank_best_is_1": "",
                                     "regret_pct_vs_best_in_set": "",
                                     "value": round(max(regrets), 4),
                                     "note": f"argmax_lang={LANGS[regrets.index(max(regrets))]}"})
                    out_rows.append({"metric": "worst_language_damage", "corpus": corpus, "comparison_set": cset_name,
                                     "conditions_in_set": "|".join(avail), "seed": seed, "language": "ALL",
                                     "condition": c, "delta_bpb_pct": "", "rank_best_is_1": "",
                                     "regret_pct_vs_best_in_set": "",
                                     "value": round(max(dmg), 4),
                                     "note": f"argmax_lang={LANGS[dmg.index(max(dmg))]}"})

        # own-language contrast: 5개 단일 언어 arm만
        for seed in SEEDS:
            avail = [c for c in SINGLE if delta(corpus, c, seed, LANGS[0]) is not None]
            complete = (len(avail) == 5)
            for cond in avail:
                lang = OWN_LANG[cond]
                own = delta(corpus, cond, seed, lang)
                others = [delta(corpus, c2, seed, lang) for c2 in avail if c2 != cond]
                out_rows.append({
                    "metric": "own_language_contrast", "corpus": corpus, "comparison_set": "single5",
                    "conditions_in_set": "|".join(avail), "seed": seed, "language": lang,
                    "condition": cond, "delta_bpb_pct": round(own, 4), "rank_best_is_1": "",
                    "regret_pct_vs_best_in_set": "",
                    "value": round(own - sum(others) / len(others), 4),
                    "note": ("own minus mean of other 4 single-language arms; Balanced/placebo excluded"
                             if complete else
                             f"INCOMPLETE: only {len(avail)}/5 single-language arms available -- "
                             "완전한 own-gain으로 보고하지 말 것"),
                })

    # FLORES <-> XL-Sum 순위 일치 (언어 안에서, seed별)
    for cset_name, cset in [("main6", MAIN6), ("table31_4cond", TABLE31_SET)]:
        for seed in SEEDS:
            avail = [c for c in cset
                     if delta("xlsum", c, seed, LANGS[0]) is not None and delta("flores", c, seed, LANGS[0]) is not None]
            if len(avail) < 3:
                continue
            for lang in LANGS:
                x = [delta("xlsum", c, seed, lang) for c in avail]
                f = [delta("flores", c, seed, lang) for c in avail]
                rho = spearman(f, x)
                sign_same = all((xi > 0) == (fi > 0) for xi, fi in zip(x, f))
                out_rows.append({
                    "metric": "spearman_flores_vs_xlsum", "corpus": "flores_vs_xlsum",
                    "comparison_set": cset_name, "conditions_in_set": "|".join(avail),
                    "seed": seed, "language": lang, "condition": "",
                    "delta_bpb_pct": "", "rank_best_is_1": "", "regret_pct_vs_best_in_set": "",
                    "value": round(rho, 4),
                    "note": (f"n_conditions={len(avail)}; ties=average rank; "
                             f"sign_consistent={sign_same}; "
                             f"flores_order={'>'.join(sorted(avail, key=lambda c: delta('flores', c, seed, lang)))}; "
                             f"xlsum_order={'>'.join(sorted(avail, key=lambda c: delta('xlsum', c, seed, lang)))}"),
                })

    with (OUT / "ranking_and_regret.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)

    (OUT / "external_bpb_missing_cells.json").write_text(
        json.dumps({"missing_xlsum_cells": missing,
                    "note": "미산출 셀은 0이나 추정으로 채우지 않는다."},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote external_bpb_grid.csv ({len(grid)} rows), "
          f"external_bpb_per_document.jsonl ({len(perdoc)} rows), "
          f"ranking_and_regret.csv ({len(out_rows)} rows)")
    print(f"missing XL-Sum cells: {len(missing)}")
    for m in missing:
        print("   ", m["condition"], "seed", m["seed"])


if __name__ == "__main__":
    main()
