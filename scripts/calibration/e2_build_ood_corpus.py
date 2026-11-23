"""E2 OOD 코퍼스 구축 — XL-Sum (BBC News) v2.0, 5개 평가 언어. CPU 전용.

무엇을 만드는가
---------------
캠페인 등록문(B-6 확정)에 따라 언어별 **동일 바이트 수 50KB**의 held-out 텍스트를
XL-Sum test split에서 뽑아 `data/ood_xlsum/` 아래에 고정한다. E2 평가 스크립트는 이
산출물만 읽고 재샘플링하지 않는다(평가 때마다 표본이 달라지면 조건 간 비교가 깨진다).

왜 Wikipedia가 아니라 XL-Sum인가 (등록 근거)
--------------------------------------------
FLORES-200 문장 자체가 Wikimedia 계열(Wikinews/Wikijunior/Wikivoyage)에서 나온 **번역문**
이다. 따라서 "Wikipedia held-out"은 발행처 계열도 문체도 겹치고 원어작성 축 하나만 바뀐다.
XL-Sum은 발행처(BBC)·문체(뉴스 기사)·원어작성 세 축이 모두 바뀌므로 H-E2가 시험하려는
domain shift를 실제로 만든다.

`datasets` 3.2.0은 스크립트 로더(xlsum.py)를 지원하지 않으므로 HF 저장소의 언어별
tarball을 직접 받아 파싱한다.

샘플링 규칙 (등록문 그대로)
---------------------------
- 언어별 **동일 바이트 수**(기본 50KB, UTF-8 기준). 문서 수를 맞추지 않는다.
- **문서 단위로만** 뽑는다. 문서를 중간에서 자르지 않는다 -- 앞부분만 취하면 기사
  리드문에 편향되기 때문. 예산을 넘기는 마지막 문서는 **버린다**(포함하지 않는다).
  따라서 실현 바이트는 예산 이하이며, 실현값을 결과에 기록한다.
- 문서 순서는 test split의 원래 순서를 고정 seed로 셔플한 것(언어 간 동일 규칙).
- 본문(`text`)만 쓴다. title/summary는 쓰지 않는다 -- 요약문은 기사 본문과 분포가
  다르고, bpb는 자연 텍스트 위에서 재야 하기 때문.

사용법:
    python e2_build_ood_corpus.py [--bytes-per-lang 51200] [--seed 0]
"""
import argparse
import hashlib
import json
import random
import sys
import tarfile
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUT_DIR = REPO_ROOT / "data" / "ood_xlsum"
CACHE_DIR = REPO_ROOT / "data" / "xlsum_raw"

DATASET = "csebuetnlp/xlsum"
VERSION = "v2.0"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/main/data"

# XL-Sum config 이름 -> 이 프로젝트의 FLORES 언어 코드. 평가 언어 5개와 1:1.
LANGS = {
    "english": "eng_Latn",
    "korean": "kor_Hang",
    "chinese_simplified": "zho_Hans",
    "swahili": "swh_Latn",
    "bengali": "ben_Beng",
}
SPLIT = "test"  # held-out. train은 쓰지 않는다.


def fetch(config):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{config}_XLSum_{VERSION}.tar.bz2"
    dest = CACHE_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[ood] {name}: cached ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    url = f"{BASE}/{name}"
    print(f"[ood] {name}: downloading ...")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"[ood] {name}: done ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def read_split(tar_path, config):
    """tarball 안에서 {config}_{SPLIT}.jsonl 을 찾아 문서 리스트를 돌려준다.

    파일명 규약을 가정하지 않고 tar 목록을 실제로 훑어 SPLIT이 들어간 항목을 고른다 --
    가정이 틀리면 조용히 train을 읽는 사고가 나므로, 후보가 정확히 1개가 아니면 죽인다.
    """
    with tarfile.open(tar_path, "r:bz2") as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and SPLIT in Path(m.name).stem.lower()]
        if len(members) != 1:
            raise SystemExit(f"[ood] {config}: {SPLIT} 후보가 {len(members)}개 "
                             f"({[m.name for m in members]}) -- 1개가 아니면 중단")
        raw = tf.extractfile(members[0]).read().decode("utf-8")
        member_name = members[0].name
    docs = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        t = (d.get("text") or "").strip()
        if t:
            docs.append({"id": d.get("id", ""), "url": d.get("url", ""), "text": t})
    return docs, member_name


def sample(docs, budget_bytes, seed):
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    picked, total = [], 0
    for i in order:
        b = len(docs[i]["text"].encode("utf-8"))
        if total + b > budget_bytes:
            continue  # 문서를 자르지 않는다 -- 예산 초과 문서는 건너뛴다
        picked.append(docs[i])
        total += b
        if total == budget_bytes:
            break
    return picked, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes-per-lang", type=int, default=50 * 1024,
                    help="언어별 동일 바이트 예산 (등록 확정값 50KB = 51200)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "_purpose": "E2 OOD bpb 평가용 코퍼스 (캠페인 등록 B-6 확정값)",
        "corpus_id": f"{DATASET}:{VERSION}:{SPLIT}",
        "dataset": DATASET, "version": VERSION, "split": SPLIT,
        "sampling_rule": ("equal BYTES per language; whole documents only (never truncated); "
                          "documents that would exceed the budget are skipped, not cut; "
                          "body text field only (no title/summary); fixed shuffle seed"),
        "bytes_budget_per_lang": args.bytes_per_lang,
        "shuffle_seed": args.seed,
        "languages": {},
    }

    for config, lang in LANGS.items():
        tar_path = fetch(config)
        docs, member = read_split(tar_path, config)
        picked, total = sample(docs, args.bytes_per_lang, args.seed)
        text = "\n\n".join(d["text"] for d in picked)

        out = OUT_DIR / f"{lang}.txt"
        out.write_text(text, encoding="utf-8")
        manifest["languages"][lang] = {
            "xlsum_config": config,
            "tar_member": member,
            "docs_available_in_split": len(docs),
            "docs_sampled": len(picked),
            "bytes_realized": total,
            "bytes_on_disk_with_separators": out.stat().st_size,
            "pct_of_budget": 100.0 * total / args.bytes_per_lang,
            "sha256_of_text": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "file": str(out.relative_to(REPO_ROOT)),
            # XL-Sum document ids of the sampled documents, in corpus order
            # (license-safe provenance: ids only, no text).
            "doc_ids": [d["id"] for d in picked],
        }
        print(f"[ood] {lang:10} {config:20} docs={len(picked):4d}/{len(docs):6d}  "
              f"bytes={total:6d} ({100.0*total/args.bytes_per_lang:5.1f}% of budget)")

    mpath = OUT_DIR / "manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[ood] wrote {mpath}")

    realized = [v["bytes_realized"] for v in manifest["languages"].values()]
    spread = 100.0 * (max(realized) - min(realized)) / args.bytes_per_lang
    print(f"[ood] 언어 간 실현 바이트 편차: {spread:.2f}% of budget "
          f"(min {min(realized)}, max {max(realized)})")
    if spread > 2.0:
        print("[ood] WARNING: 언어 간 바이트 편차가 2%를 넘는다 -- 결과 문서에 명시할 것")


if __name__ == "__main__":
    main()
