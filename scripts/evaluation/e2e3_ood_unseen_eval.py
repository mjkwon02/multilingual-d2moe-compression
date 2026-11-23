"""E2 + E3 통합 평가 — 한 모델당 병합 1회로 OOD bpb(5언어) + unseen bpb(4언어)를 모두 잰다.

왜 한 스크립트인가
------------------
실측상 병합·평가 1셀 ~26.5분 중 **실제 평가는 72.5초**고 나머지는 모델 로드 + 병합이다
(logs/dr_sweep_0802.log, logs/sec54_korean_repro_0802.log). 두 평가를 따로 돌리면 25분짜리
병합을 두 번 낸다. 지시문 2-b가 "단일 스크립트로 묶어 실행"을 요구하는 이유이며, 15개
모델 × 2평가에서 이 차이가 6시간 이상이다.

병합·로드는 phase1_merge_eval의 load_model()/merge_condition()을 **무수정 import**한다
(phase1_sib200_eval.py / phase1_belebele_eval.py와 같은 방식). 평가 배치는 8 고정
(부록 B.5 규약, 캠페인 §0.1).

E2 — OOD bpb
------------
코퍼스는 `data/ood_xlsum/`에 **이미 고정된 산출물**을 읽기만 한다(e2_build_ood_corpus.py가
생성). 평가 때마다 다시 뽑으면 모델 간 비교가 깨지므로 이 스크립트는 재샘플링하지 않는다.

세그먼트화(중요): 문서를 통째로 먹이면 벵골어 문서 하나가 수천 토큰이라 batch=8에서
logits 텐서가 수십 GB가 된다. 그래서 문단(개행) 단위로 자르고, 그래도 SEG_MAX_CHARS를
넘는 문단은 더 쪼갠다. 두 가지를 지킨다:

  1. **잘라낸 바이트를 잃지 않는다.** tokenizer truncation을 쓰지 않고(batch_sentence_nll은
     padding만 하고 truncation을 안 한다) 문자 단위로 미리 쪼개므로, 채점된 텍스트와
     바이트 분모가 정확히 일치한다. truncation을 켰다면 잘린 꼬리 바이트가 분모에만
     남아 bpb가 조용히 틀렸을 것이다.
  2. **세그먼트화 규칙이 전 모델 공통**이다. 코퍼스 파일도 규칙도 고정이므로 15개 모델의
     bpb는 서로 정확히 비교 가능하다. FLORES 절대값과의 비교는 목적이 아니다
     (H-E2: "크기 재현은 요구하지 않음", 부호와 순서만).

각 세그먼트의 첫 토큰은 shifted causal loss에서 예측 대상이 아니다 — FLORES 문장 평가와
같은 관례이며, 세그먼트화가 전 모델 공통이라 비교에는 영향이 없다. 결과에
`n_segments`를 남겨 이 관례의 크기를 사후 확인할 수 있게 한다.

E3 — unseen-language bpb
------------------------
캠페인 등록 B-3 확정 4개 언어(arb_Arab, hin_Deva, tur_Latn, rus_Cyrl)를 FLORES-200에서
**기존 프로토콜 그대로** 60문장씩 평가한다(phase1_merge_eval.eval_flores_ppl과 동일한
소스·split·문장 수·batch). 이 4개 언어는 어떤 calibration에도 쓰이지 않았으므로 held-out
오프셋을 따로 둘 필요가 없다.

사용법:
    python e2e3_ood_unseen_eval.py --condition mixed_5lang --seed 0 \
        --scale-condition mixed_5lang --scale-seed 0
    python e2e3_ood_unseen_eval.py --baseline
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import data_utils  # noqa: E402
import phase1_calib_data  # noqa: E402
from phase1_merge_eval import (  # noqa: E402 -- reused unchanged
    EVAL_PREFIX, EVAL_SOURCE, EVAL_SPLIT, RESULTS_ROOT, load_model, merge_condition,
)

OOD_DIR = SCRIPT_DIR.parent / "data" / "ood_xlsum"
OOD_LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]

# 캠페인 등록 B-3 확정값. 어떤 calibration 조건에도 쓰이지 않은 4개 언어.
UNSEEN_LANGS = {"arb_Arab": "Arabic", "hin_Deva": "Hindi",
                "tur_Latn": "Turkish", "rus_Cyrl": "Russian"}
UNSEEN_N_SENTENCES = 60  # phase1_merge_eval.EVAL_N_SENTENCES와 동일

EVAL_BATCH_SIZE = 8   # 부록 B.5 규약. 낮추지 말 것 -- bpb가 배치에 따라 재현되지 않는다.

# 세그먼트 상한(문자). 배치당 peak 토큰이 **이미 검증된 FLORES 평가**를 넘지 않도록
# 실측으로 정했다. batch=8에서 FLORES ben_Beng의 peak는 8 x 539 = 4,312토큰이고 그
# 구성이 현재 공식 격자를 만들어낸 설정이다(ben_Beng은 이 파이프라인에서 유일하게
# OOM 이력이 있는 언어다). 384자로 자르면 ben 세그먼트가 최대 713토큰 -> peak 5,704로
# 검증된 값보다 32% 높아 OOM 위험이 실재한다. 288자면 ben 최대 ~535토큰 -> peak ~4,280
# 으로 검증 범위 안에 들어온다. batch를 낮추는 대신 세그먼트를 줄이는 이유는 batch=8이
# 등록된 규약이고 bpb가 배치에 따라 재현되지 않기 때문이다.
SEG_MAX_CHARS = 288


def segment(text, max_chars=SEG_MAX_CHARS):
    """문단 단위로 자르고, 상한을 넘는 문단은 공백 경계에서 더 쪼갠다.
    공백이 없으면(중국어 등) 문자 수로 자른다 -- 어떤 경우에도 텍스트를 **버리지 않는다**.
    """
    segs = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        while len(para) > max_chars:
            cut = para.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars  # 공백 없는 스크립트: 문자 경계에서 자른다
            segs.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            segs.append(para)
    return [s for s in segs if s]


def bpb_over(model, tokenizer, segments, batch_size=EVAL_BATCH_SIZE, per_segment=False):
    """per_segment (2026-09-16, 데이터감사 지시문 §5 "source document별 raw NLL·scored
    bytes·token 수를 저장한다"): evaluate_sentences()는 **이미** 세그먼트별 (nll_sum,
    n_tokens)를 돌려주는데 기존 코드는 그것을 합계로만 쓰고 버렸다. 이 인자는 그 값을
    함께 반환할 뿐이며 **집계 산식·배치·세그먼트화·반환되는 기존 키를 하나도 바꾸지
    않는다** -- 따라서 이 변경 전후의 bits_per_byte는 비트 단위로 동일하다.
    문서 귀속은 eval_ood()가 한다(세그먼트화가 문서 단위로 분해 가능함을 실측 확인)."""
    nll_sums, n_tokens = data_utils.evaluate_sentences(model, tokenizer, segments, batch_size=batch_size)
    total_nll_nats = sum(nll_sums)
    total_bytes = sum(len(s.encode("utf-8")) for s in segments)
    out = {
        "bits_per_byte": (total_nll_nats / math.log(2)) / total_bytes,
        "ppl": data_utils.corpus_ppl(nll_sums, n_tokens),
        "n_segments": len(segments),
        "n_tokens_scored": sum(n_tokens),
        "n_bytes_scored": total_bytes,
    }
    if per_segment:
        out["per_segment"] = [
            {"seg": k, "nll_nats": float(nll_sums[k]), "n_tokens": int(n_tokens[k]),
             "n_bytes": len(segments[k].encode("utf-8"))}
            for k in range(len(segments))
        ]
    return out


def eval_ood(model, tokenizer):
    """2026-09-16 추가: 세그먼트를 **문서에 귀속**시켜 문서별 raw NLL/bytes/tokens를 남긴다.

    코퍼스 파일은 e2_build_ood_corpus.py가 "\n\n".join(문서본문)으로 쓴 것이고
    segment()는 "\n"으로 쪼갠 뒤 빈 문단을 버리므로, **문서를 따로 segment()해서 이어
    붙인 결과가 전체를 한 번에 segment()한 결과와 문자열 단위로 동일하다**(5개 언어
    전부 실측 확인, 2026-09-16). 따라서 segs는 그대로 두고(= 기존 수치 불변) 각
    세그먼트가 몇 번째 문서에서 왔는지만 CPU로 계산해 붙인다."""
    manifest = json.loads((OOD_DIR / "manifest.json").read_text())
    out = {}
    for lang in OOD_LANGS:
        text = (OOD_DIR / f"{lang}.txt").read_text(encoding="utf-8")
        segs = segment(text)
        doc_of_seg, docs = [], text.split("\n\n")
        for di, d in enumerate(docs):
            doc_of_seg.extend([di] * len(segment(d)))
        if len(doc_of_seg) != len(segs):
            raise SystemExit(f"[e2] {lang}: 문서 귀속 실패 ({len(doc_of_seg)} != {len(segs)}) -- "
                             "귀속을 추정으로 때우지 않는다")
        r = bpb_over(model, tokenizer, segs, per_segment=True)
        for k, rec in enumerate(r["per_segment"]):
            rec["doc_index"] = doc_of_seg[k]
        r["per_document"] = [
            {"doc_index": di,
             "nll_nats": sum(x["nll_nats"] for x in r["per_segment"] if x["doc_index"] == di),
             "n_tokens": sum(x["n_tokens"] for x in r["per_segment"] if x["doc_index"] == di),
             "n_bytes": sum(x["n_bytes"] for x in r["per_segment"] if x["doc_index"] == di),
             "n_segments": sum(1 for x in r["per_segment"] if x["doc_index"] == di)}
            for di in range(len(docs))
        ]
        r["name"] = lang
        r["docs_sampled"] = manifest["languages"][lang]["docs_sampled"]
        r["corpus_bytes_on_disk"] = manifest["languages"][lang]["bytes_realized"]
        out[lang] = r
        print(f"[e2] OOD {lang}: bpb={r['bits_per_byte']:.6f} over {r['n_bytes_scored']} bytes "
              f"({r['n_segments']} segs, {r['docs_sampled']} docs)")
    return out, manifest


def eval_unseen(model, tokenizer):
    out = {}
    for lang, name in UNSEEN_LANGS.items():
        sentences = data_utils.load_flores_sentences(
            EVAL_SOURCE, EVAL_SPLIT, EVAL_PREFIX, lang, UNSEEN_N_SENTENCES)
        r = bpb_over(model, tokenizer, sentences)
        r["name"] = name
        r["n_sentences"] = len(sentences)
        out[lang] = r
        print(f"[e3] UNSEEN {lang} ({name}): bpb={r['bits_per_byte']:.6f} "
              f"over {r['n_bytes_scored']} bytes ({len(sentences)} sentences)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=phase1_calib_data.CONDITIONS)
    ap.add_argument("--baseline", action="store_true", help="uncompressed model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale-condition", default=None)
    ap.add_argument("--scale-seed", type=int, default=None)
    ap.add_argument("--delta-ratio", type=float, default=None)
    ap.add_argument("--out-name", default="eval_ood_unseen.json")
    args = ap.parse_args()

    if not args.baseline and not args.condition:
        raise SystemExit("--condition 또는 --baseline 중 하나는 필요하다")
    if not (OOD_DIR / "manifest.json").exists():
        raise SystemExit(f"{OOD_DIR}/manifest.json 없음 -- e2_build_ood_corpus.py를 먼저 돌릴 것")

    t0 = time.time()
    model, tokenizer = load_model()
    print(f"[e2e3] model loaded in {time.time()-t0:.0f}s")

    if args.baseline:
        out_dir = RESULTS_ROOT / "baseline"
        tag = {"baseline": True}
    else:
        scale_cond = args.scale_condition
        scale_seed = args.scale_seed if args.scale_seed is not None else args.seed
        t1 = time.time()
        merge_condition(model, args.condition, args.seed,
                        scale_condition=scale_cond, scale_seed=scale_seed,
                        delta_ratio=args.delta_ratio)
        print(f"[e2e3] merged in {time.time()-t1:.0f}s")
        sub = RESULTS_ROOT / args.condition / f"seed{args.seed}"
        out_dir = sub / f"scale_{scale_cond}_seed{scale_seed}" if scale_cond else sub / "plain_svd"
        if args.delta_ratio is not None:
            out_dir = out_dir / f"delta_ratio_{args.delta_ratio}"
        tag = {"baseline": False, "condition": args.condition, "seed": args.seed,
               "scale_condition": scale_cond, "scale_seed": scale_seed,
               "delta_ratio": args.delta_ratio if args.delta_ratio is not None else 0.8}

    model.eval()
    with torch.no_grad():
        ood, manifest = eval_ood(model, tokenizer)
        unseen = eval_unseen(model, tokenizer)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    payload = {
        "_purpose": "E2 (OOD bpb) + E3 (unseen-language bpb), campaign E1-E5",
        "_registration": "logs/E1E5_PREREG.log (2026-08-03T04:45Z; confirmed values 05:35Z)",
        "run": tag,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "segmentation": {"max_chars": SEG_MAX_CHARS,
                         "rule": "split on newline into paragraphs; oversized paragraphs split at the "
                                 "last space before max_chars (character boundary if no space). "
                                 "No tokenizer truncation -- scored bytes == counted bytes."},
        "e2_ood": {
            "corpus_id": manifest["corpus_id"],
            "dataset": manifest["dataset"], "version": manifest["version"], "split": manifest["split"],
            "bytes_budget_per_lang": manifest["bytes_budget_per_lang"],
            "sampling_rule": manifest["sampling_rule"],
            "results": ood,
        },
        "e3_unseen": {
            "source": EVAL_SOURCE, "split": EVAL_SPLIT, "n_sentences": UNSEEN_N_SENTENCES,
            "languages": list(UNSEEN_LANGS),
            "results": unseen,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[e2e3] wrote {out_path}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
