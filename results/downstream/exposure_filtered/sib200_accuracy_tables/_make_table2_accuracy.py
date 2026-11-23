"""Table 2 (submitted paper; = Table 3 in 00_docs/08_figure_정리.md) rebuilt in
absolute accuracy (%), answering the reviewer comment that the baseline row was
accuracy while the compressed rows were retention.

CPU only. Reads stored JSON; starts no evaluation. Rerunnable: the clean table
fills in as analysis/clean_eval_0916/sib200/{condition}_seed{N}.json appear.

  A. full  : submitted-paper per-seed results, n=204 (SIB_EXTENDED_REPORT provenance).
             Never mixed with the 0916 re-run's full-204 numbers.
  B. clean : analysis/clean_eval_0916/sib200/*.json, `clean` block, n=98, against
             the clean baseline measured in the same campaign.

Rules: mean/std from unrounded per-seed accuracy (correct/n), std ddof=1,
shown in percentage points to 1 decimal. Collapsed seeds are kept in the mean;
collapse = top-label share > 0.9 or acc < 0.25 (unchanged project rule).
Bold = higher mean of {own-language, Balanced-character} within a column; in the
clean table only when both conditions have all 3 seeds.

Usage: python _make_table2_accuracy.py
"""
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
R = Path("results/deepseek")
CLEAN_DIR = Path("results/downstream/exposure_filtered/deepseek_clean_eval/sib200")
CLEAN_STATUS = Path("results/downstream/exposure_filtered/deepseek_clean_eval/campaign_status.tsv")
DOC = Path("docs/provenance/figure_registry_internal.md")  # optional; skipped when absent

LANGS = [("EN", "eng_Latn"), ("ZH", "zho_Hans"), ("KO", "kor_Hang"), ("SW", "swh_Latn"), ("BN", "ben_Beng")]
CONDS = [  # (condition id, row label, own-language column)
    ("english_only", "English calibration", "EN"),
    ("chinese_only", "Chinese calibration", "ZH"),
    ("korean_only", "Korean calibration", "KO"),
    ("swahili_only", "Swahili calibration", "SW"),
    ("bengali_only", "Bengali calibration", "BN"),
    ("mixed_5lang", "Balanced-character calibration", None),
]
BAL = "mixed_5lang"
SEEDS = [0, 1, 2]
TOP_SHARE_MAX, ACC_MIN = 0.9, 0.25


def full_source(cond, seed):
    # exactly the provenance table of phase1/sib200_extended/SIB_EXTENDED_REPORT.md
    if cond in ("korean_only", "swahili_only"):
        return R / cond / f"seed{seed}" / f"scale_{cond}_seed{seed}" / "eval_sib200.json"
    return R / "sib200_extended" / f"{cond}_seed{seed}.json"


def rec_from_full(path):
    res = json.loads(path.read_text())["results"]
    out = {}
    for ab, code in LANGS:
        x = res[code]
        n = x["n_samples"]
        assert sum(x["pred_distribution"].values()) == n
        k = round(x["acc"] * n)
        assert abs(k / n - x["acc"]) < 1e-9, (path, code)
        out[ab] = dict(n=n, correct=k, acc=k / n, top=max(x["pred_distribution"].values()) / n)
    return out


def rec_from_clean(path):
    d = json.loads(path.read_text())
    out = {}
    for ab, code in LANGS:
        c = d["results"][code]["clean"]
        k = round(c["acc"] * c["n"])
        assert abs(k / c["n"] - c["acc"]) < 1e-9
        out[ab] = dict(n=c["n"], correct=k, acc=k / c["n"], top=c["top_label_share"],
                       rerun_full_acc=d["results"][code]["full"]["acc"])
    return out


def clean_ids(key):
    ids = {}
    with open(CLEAN_DIR / f"{key}_predictions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["in_clean"]:
                ids.setdefault(r["lang"], set()).add(r["index_id"])
    return ids


def collapsed(r):
    return r["top"] > TOP_SHARE_MAX or r["acc"] < ACC_MIN


def load():
    data = {"full": {"baseline": rec_from_full(R / "baseline" / "eval_sib200.json")},
            "clean": {"baseline": rec_from_clean(CLEAN_DIR / "baseline.json")}}
    sources = {"full": [str(R / "baseline" / "eval_sib200.json")],
               "clean": [str(CLEAN_DIR / "baseline.json")]}
    ref_ids = clean_ids("baseline")
    assert all(len(v) == 98 for v in ref_ids.values()) and len({frozenset(v) for v in ref_ids.values()}) == 1
    for cond, _, _ in CONDS:
        for s in SEEDS:
            p = full_source(cond, s)
            data["full"][(cond, s)] = rec_from_full(p)
            sources["full"].append(str(p))
            q = CLEAN_DIR / f"{cond}_seed{s}.json"
            if q.exists():
                assert clean_ids(f"{cond}_seed{s}") == ref_ids, f"clean item set differs: {q}"
                data["clean"][(cond, s)] = rec_from_clean(q)
                sources["clean"].append(str(q))
    return data, sources


def cell_stats(data, subset, cond, lang):
    runs = [data[subset][(cond, s)][lang] for s in SEEDS if (cond, s) in data[subset]]
    if len(runs) < len(SEEDS):
        return None
    accs = [r["acc"] for r in runs]
    base = data[subset]["baseline"][lang]["acc"]
    rets = [a / base for a in accs]
    return dict(mean=statistics.mean(accs), sd=statistics.stdev(accs),
                k=sum(collapsed(r) for r in runs),
                rmean=statistics.mean(rets), rsd=statistics.stdev(rets))


def bold_set(data, subset):
    """{(cond, lang)} cells to bold: own-language vs Balanced-character, higher mean."""
    out = set()
    for cond, _, own in CONDS:
        if own is None:
            continue
        a, b = cell_stats(data, subset, cond, own), cell_stats(data, subset, BAL, own)
        if a is None or b is None or a["mean"] == b["mean"]:
            continue
        out.add((cond, own) if a["mean"] > b["mean"] else (BAL, own))
    return out


def mark(k, tex):
    return "" if k == 0 else (f"\\,[{k}/3]" if tex else f" [{k}/3]")


def acc_rows(data, subset, tex):
    pm = " $\\pm$ " if tex else " ± "
    bolds = bold_set(data, subset)
    rows = [["Uncompressed"] + [f"{100 * data[subset]['baseline'][l]['acc']:.1f}" for l, _ in LANGS]]
    for cond, label, _ in CONDS:
        row = [label]
        for l, _ in LANGS:
            st = cell_stats(data, subset, cond, l)
            if st is None:
                row.append("Pending")
                continue
            txt = f"{100 * st['mean']:.1f}{pm}{100 * st['sd']:.1f}"
            if (cond, l) in bolds:
                txt = f"\\textbf{{{txt}}}" if tex else f"**{txt}**"
            row.append(txt + mark(st["k"], tex))
        rows.append(row)
    return rows


def ret_rows(data, subset, tex):
    pm = " $\\pm$ " if tex else " ± "
    rows = []
    for cond, label, _ in CONDS:
        row = [label]
        for l, _ in LANGS:
            st = cell_stats(data, subset, cond, l)
            row.append("Pending" if st is None else f"{st['rmean']:.3f}{pm}{st['rsd']:.3f}{mark(st['k'], tex)}")
        rows.append(row)
    return rows


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join([":---"] + [":---:"] * (len(header) - 1)) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def tex_table(header, rows, caption, label, baseline_rule=True):
    lines = ["\\begin{table*}[t]", "\\centering", "\\small",
             "\\begin{tabular}{l" + "c" * (len(header) - 1) + "}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        lines.append(" & ".join(r) + " \\\\")
        if i == 0 and baseline_rule:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table*}"]
    return "\n".join(lines)


def status_failed():
    failed = set()
    if CLEAN_STATUS.exists():
        for line in CLEAN_STATUS.read_text().splitlines()[1:]:
            _, unit, st = line.split("\t")
            if st == "FAILED":
                failed.add(unit)
            elif unit in failed:
                failed.discard(unit)
    return failed


# ------------------------------------------------------------------ captions
CAP_COMMON = (
    "Cells for compressed models are mean $\\pm$ sample standard deviation (ddof = 1) over 3 calibration seeds, "
    "in percentage points; the Uncompressed row is a single evaluation and carries no standard deviation. "
    "All seeds, including collapsed ones, enter the mean and standard deviation. "
    "[$k$/3] marks the number of seeds whose predictions collapsed (top-label share $>$ 0.9 or accuracy $<$ 25\\%); "
    "cells without a bracket have no collapsed seed. The 25\\% threshold is the project's pre-existing collapse "
    "criterion and is not the 7-way uniform-chance rate (1/7 $\\approx$ 14.3\\%). "
    "Unlike the submitted version, which replaced fully collapsed cells with a placeholder, raw accuracy is now "
    "reported for every cell and collapse is marked separately. "
    "Bold marks, within each evaluation-language column, the higher mean of the own-language calibration and "
    "Balanced-character calibration; it is a descriptive comparison of means, not a significance test."
)
CAP_FULL = (
    "SIB-200 topic-classification accuracy (\\%) after compression, full test set "
    "($n$ = 204 per language; 14-shot, fixed English demonstrations). " + CAP_COMMON +
    " Values are the originally reported runs; retention relative to the uncompressed model is given separately in "
    "Table~\\ref{tab:sib-retention}."
)
CAP_CLEAN = (
    "SIB-200 topic-classification accuracy (\\%) on the clean subset ($n$ = 98 per language; the same 98 items in all "
    "five languages and for all models; same prompt and 14-shot demonstrations as the full evaluation). The "
    "Uncompressed row is the uncompressed model evaluated on this same subset. " + CAP_COMMON +
    " Pending: fewer than 3 seeds completed; no mean or bold is shown for such cells, and bold is shown only where "
    "both compared conditions have all 3 seeds. Clean and full accuracies are measured on different item sets, so "
    "their difference is descriptive only."
)
CAP_RET = (
    "SIB-200 retention (ratio, $\\times$; not a percentage), defined as compressed-model accuracy divided by "
    "uncompressed accuracy on the same evaluation subset: the full-test baseline ($n$ = 204) for the full panel and "
    "the clean-subset baseline ($n$ = 98) for the clean panel. Mean $\\pm$ sample standard deviation (ddof = 1) of "
    "per-seed retention over 3 seeds, collapsed seeds included; [$k$/3] as in the accuracy tables. A high "
    "retention in a cell with collapsed seeds, or where the uncompressed accuracy is itself low (e.g.\\ SW), "
    "should not be read as preserved task ability. Pending: fewer than 3 seeds completed."
)


def plain(cap):
    for a, b in [("$\\pm$", "±"), ("$>$", ">"), ("$<$", "<"), ("$\\approx$", "≈"), ("$\\times$", "×"),
                 ("$n$", "n"), ("[$k$/3]", "[k/3]"), ("\\%", "%"), ("e.g.\\ ", "e.g. "),
                 ("Table~\\ref{tab:sib-retention}", "the retention table")]:
        cap = cap.replace(a, b)
    return cap


def main():
    data, sources = load()
    H = ["Calibration"] + [l for l, _ in LANGS]
    done = {c: [s for s in SEEDS if (c, s) in data["clean"]] for c, _, _ in CONDS}
    incomplete = [c for c, _, _ in CONDS if len(done[c]) < len(SEEDS)]
    failed = status_failed()

    # ---------------- raw CSV
    with open(HERE / "table2_raw_per_seed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "condition", "seed", "lang", "n", "n_correct", "acc", "top_label_share",
                    "collapsed", "baseline_acc_same_subset", "retention",
                    "rerun_full204_acc_REPRO_CHECK_ONLY", "source_json"])
        for subset in ("full", "clean"):
            b = data[subset]["baseline"]
            src_base = sources[subset][0]
            for l, _ in LANGS:
                r = b[l]
                w.writerow([subset, "uncompressed", "", l, r["n"], r["correct"], f"{r['acc']:.6f}",
                            f"{r['top']:.6f}", int(collapsed(r)), f"{r['acc']:.6f}", "1.000000",
                            f"{r['rerun_full_acc']:.6f}" if subset == "clean" else "", src_base])
            for cond, _, _ in CONDS:
                for s in SEEDS:
                    if (cond, s) not in data[subset]:
                        continue
                    src = str(full_source(cond, s)) if subset == "full" else str(CLEAN_DIR / f"{cond}_seed{s}.json")
                    for l, _ in LANGS:
                        r = data[subset][(cond, s)][l]
                        w.writerow([subset, cond, s, l, r["n"], r["correct"], f"{r['acc']:.6f}", f"{r['top']:.6f}",
                                    int(collapsed(r)), f"{b[l]['acc']:.6f}", f"{r['acc'] / b[l]['acc']:.6f}",
                                    f"{r['rerun_full_acc']:.6f}" if subset == "clean" else "", src])

    # ---------------- progress table (incomplete clean conditions only)
    prog_rows = []
    for c in incomplete:
        for s in SEEDS:
            if (c, s) in data["clean"]:
                rr = data["clean"][(c, s)]
                prog_rows.append([c, f"seed{s}"] + [f"{100 * rr[l]['acc']:.1f}" + (" [C]" if collapsed(rr[l]) else "")
                                                    for l, _ in LANGS])
            else:
                st = "FAILED (not substituted)" if f"{c}/seed{s}" in failed else "not completed"
                prog_rows.append([c, f"seed{s}"] + [st] * len(LANGS))
    prog_md = (md_table(["condition", "seed"] + [l for l, _ in LANGS], prog_rows) if prog_rows
               else "_All clean conditions have 3/3 seeds; no partial results._")

    # ---------------- markdown files
    full_md = md_table(H, acc_rows(data, "full", False))
    clean_md = md_table(H, acc_rows(data, "clean", False))
    ret_full_md = md_table(H, ret_rows(data, "full", False))
    ret_clean_md = md_table(H, ret_rows(data, "clean", False))

    (HERE / "table2_full_accuracy.md").write_text(
        "# Table 2 (revised) — SIB-200 accuracy (%), full test set, n=204\n\n" + full_md +
        "\n\n**Caption.** " + plain(CAP_FULL) + "\n")
    (HERE / "table2_clean_accuracy.md").write_text(
        "# Table 2-clean — SIB-200 accuracy (%), clean subset, n=98\n\n" + clean_md +
        "\n\n**Caption.** " + plain(CAP_CLEAN) +
        "\n\n## Progress — completed seeds of conditions not yet at 3/3 (accuracy %, clean n=98; [C] = collapsed seed)\n\n"
        "Not for citation as final values.\n\n" + prog_md + "\n")
    (HERE / "table2_retention_supplement.md").write_text(
        "# Supplementary — SIB-200 retention (×, ratio to uncompressed accuracy on the same subset)\n\n"
        "## (a) Full test set, n=204 (denominator: full uncompressed accuracy)\n\n" + ret_full_md +
        "\n\n## (b) Clean subset, n=98 (denominator: clean uncompressed accuracy)\n\n" + ret_clean_md +
        "\n\n**Caption.** " + plain(CAP_RET) + "\n")

    # ---------------- LaTeX
    (HERE / "table2_full_accuracy.tex").write_text(
        tex_table(H, acc_rows(data, "full", True), CAP_FULL, "tab:sib-acc-full") + "\n")
    (HERE / "table2_clean_accuracy.tex").write_text(
        tex_table(H, acc_rows(data, "clean", True), CAP_CLEAN, "tab:sib-acc-clean") + "\n")
    rf, rc = ret_rows(data, "full", True), ret_rows(data, "clean", True)
    ncol = len(H)
    ret_tex = ["\\begin{table*}[t]", "\\centering", "\\small", "\\begin{tabular}{l" + "c" * (ncol - 1) + "}",
               "\\toprule", " & ".join(H) + " \\\\", "\\midrule",
               f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{(a) Full test set, $n$ = 204}}}} \\\\"]
    ret_tex += [" & ".join(r) + " \\\\" for r in rf]
    ret_tex += ["\\midrule", f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{(b) Clean subset, $n$ = 98}}}} \\\\"]
    ret_tex += [" & ".join(r) + " \\\\" for r in rc]
    ret_tex += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{CAP_RET}}}", "\\label{tab:sib-retention}",
                "\\end{table*}"]
    (HERE / "table2_retention_supplement.tex").write_text("\n".join(ret_tex) + "\n")

    (HERE / "captions_en.md").write_text(
        "# English captions (plain text; LaTeX versions are inside the .tex files)\n\n"
        "## Table 2 (revised, full)\n\n" + plain(CAP_FULL) +
        "\n\n## Table 2-clean\n\n" + plain(CAP_CLEAN) +
        "\n\n## Supplementary retention table\n\n" + plain(CAP_RET) + "\n")

    status_line = ("clean: ALL 6 conditions × 3 seeds complete" if not incomplete else
                   "clean: incomplete — " + ", ".join(f"{c} {len(done[c])}/3" for c in incomplete))
    (HERE / "sources_used.txt").write_text(
        "# full (n=204, submitted-paper runs)\n" + "\n".join(sources["full"]) +
        "\n\n# clean (n=98, analysis/clean_eval_0916)\n" + "\n".join(sources["clean"]) +
        f"\n\n# status\n{status_line}\n")

    # ---------------- block at top of 08_figure_정리.md
    b0, b1 = "<!-- TABLE2_ACC_REVISION:BEGIN -->", "<!-- TABLE2_ACC_REVISION:END -->"
    rel = "downstream/exposure_filtered/table2_accuracy"
    block = f"""{b0}
## ⚠︎ 수정본 — Table 2 (SIB-200) accuracy/retention 혼용 해소 (2026-09-16, 자동 생성 블록)

리뷰어 지적("first row is acc but the following rows are retention") 대응. **제출 논문 Table 2 = 본 문서 아래 「Table 3 — §5.2 SIB-200」**.
이 블록은 `{rel}/_make_table2_accuracy.py`가 생성한다(수동 편집 금지, 재실행 시 덮어씀). CPU 전용, 추가 GPU 평가 없음.

- 주 표는 전부 **absolute accuracy (%)**, 압축 모델 = 3 seed 평균 ± 표본표준편차(ddof=1, %p), Uncompressed = 단일값(± 없음).
- **붕괴 seed도 평균에 포함**, `[k/3]` = 붕괴 seed 수(top-label share > 0.9 또는 acc < 0.25; 1/7 균등확률과 동일시 금지). k=0이면 표기 생략.
  기존 [COLLAPSED] 대체 표기 → **raw accuracy 공개 + 붕괴 표시**로 변경.
- 굵게 = 각 열에서 own-language vs Balanced-character 중 평균이 높은 쪽(유의성 아님). clean은 두 조건 모두 3/3일 때만.
- Token-balanced는 원 Table 2의 6조건이 아니므로 **미포함**.
- full(A)은 제출 원 결과만 사용 — 0916 재실행의 full-204 값으로 교체하지 않음(재현 차이는 CSV의 `rerun_full204_acc_REPRO_CHECK_ONLY` 열에만 별도 기록).
- 상태: **{status_line}**

### A. Table 2 수정본 — full test, n=204

{full_md}

### B. Clean 평가표 — clean subset, n=98 (동일 98문항, 동일 clean baseline)

{clean_md}

진행 표(미완료 조건의 완료 seed만, accuracy %, [C]=붕괴 seed — 최종값으로 인용 금지):

{prog_md}

### 보조 — retention (×, 동일 subset의 uncompressed accuracy 대비 비율)

(a) full, n=204

{ret_full_md}

(b) clean, n=98

{ret_clean_md}

높은 retention이라도 붕괴 seed가 있거나 baseline 자체가 낮은 열(SW)에서는 능력 보존으로 해석하지 않는다.

**캡션(영문)·LaTeX·원시 CSV**: `{rel}/` — `table2_full_accuracy.{{md,tex}}`, `table2_clean_accuracy.{{md,tex}}`,
`table2_retention_supplement.{{md,tex}}`, `table2_raw_per_seed.csv`, `captions_en.md`, `sources_used.txt`.

---
{b1}
"""
    if DOC.exists():
        text = DOC.read_text()
        if b0 in text:
            pre, rest = text.split(b0, 1)
            text = pre + block.rstrip("\n") + rest.split(b1, 1)[1]
        else:
            text = block + "\n" + text
        DOC.write_text(text)
    else:
        print(f"[note] {DOC} absent -- registry block not written (tables above are complete)")
    print(status_line)


if __name__ == "__main__":
    main()
