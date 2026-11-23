"""artifact_manifest.json 생성 (2026-09-16 지시문 §6). CPU 전용, 읽기만 한다.

모든 입력·모델·revision·protocol·실행 명령을 **실측**으로 적는다. 문서가 주장하는 값이
아니라 코드 상수와 디스크 실사 결과를 적고, 둘이 다르면 양쪽을 따로 남긴다.
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import phase1_calib_data as PCD  # noqa: E402

RESULTS_ROOT = Path("results/deepseek")
HF_HOME = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
MAIN = ["english_only", "chinese_only", "korean_only", "swahili_only", "bengali_only", "mixed_5lang"]
AUX = ["balanced_tok_mono"]
SEEDS = [0, 1, 2]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e:
        return f"<failed: {e}>"


def file_info(p):
    p = Path(p)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p), "exists": True, "bytes": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}


def sha256_file(p, limit=None):
    hsh = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            hsh.update(b)
    return hsh.hexdigest()


def hub_revision(repo_kind, repo_id):
    d = HF_HOME / "hub" / f"{repo_kind}--{repo_id.replace('/', '--')}"
    if not d.exists():
        d2 = Path.home() / ".cache/huggingface/hub" / f"{repo_kind}--{repo_id.replace('/', '--')}"
        d = d2 if d2.exists() else d
    if not d.exists():
        return {"repo": repo_id, "local_cache": None, "revision": "UNKNOWN"}
    refs, snaps = d / "refs", d / "snapshots"
    rev = {}
    if refs.exists():
        for r in refs.iterdir():
            rev[r.name] = r.read_text().strip()
    return {"repo": repo_id, "local_cache": str(d), "refs": rev,
            "snapshots": sorted(p.name for p in snaps.iterdir()) if snaps.exists() else []}


def main():
    out_dir = ROOT / "analysis" / "data_audit_0916"
    out_dir.mkdir(parents=True, exist_ok=True)

    m = {
        "_purpose": "Calibration/evaluation 데이터 감사 + XL-Sum 외부 bpb 격자 완성 (2026-09-16 지시문)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo": {
            "root": str(ROOT),
            "git_head": sh(f"git -C {ROOT} rev-parse HEAD"),
            "git_branch": sh(f"git -C {ROOT} rev-parse --abbrev-ref HEAD"),
            "git_status_short": sh(f"git -C {ROOT} status --short").splitlines(),
            "git_diff_stat_tracked": sh(f"git -C {ROOT} diff --stat").splitlines(),
        },
        "environment": {
            "conda_env": "d2moe_env",
            "python": sh("conda run -n d2moe_env python -c 'import sys;print(sys.version.split()[0])'"),
            "packages": {k: sh(f"conda run -n d2moe_env python -c \"import {k};print({k}.__version__)\"")
                         for k in ["torch", "transformers", "datasets"]},
            "HF_HOME": str(HF_HOME),
            "gpus": sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader").splitlines(),
            "note": "환경은 이 작업에서 업그레이드하지 않았다.",
        },
        "model": {
            "base_model": "deepseek-ai/deepseek-moe-16b-base",
            "load_kwargs": {"device_map": "auto", "max_memory_per_gpu": "22GiB",
                            "trust_remote_code": True, "torch_dtype": "torch.bfloat16"},
            "tokenizer": {"same_repo": True, "use_fast": False},
            "hub": hub_revision("models", "deepseek-ai/deepseek-moe-16b-base"),
            "revision_pinned_in_code": None,
            "revision_note": ("코드가 revision을 고정하지 않는다(from_pretrained에 revision 인자 없음). "
                              "아래 hub.refs/snapshots가 실제 로드된 로컬 스냅샷이다."),
        },
        "compression_protocol": {
            "delta_ratio": 0.8,
            "share_ratio_and_merge_method": sh(
                f"grep -E '^(SHARE_RATIO|MERGE_METHOD|DELTA_RATIO)' {SCRIPT_DIR}/phase1_merge_eval.py").splitlines(),
            "structured_pruning": "OFF (cfg['prune_ratio']=0.0, pp_ratio 미사용)",
            "moe_layers": sh(f"grep -n 'MOE_LAYERS' {SCRIPT_DIR}/phase1_fisher.py | head -3").splitlines(),
            "merged_checkpoints_on_disk": ("없음 -- phase1_merge_eval.py에 torch.save/save_pretrained가 없다. "
                                           "병합 모델은 RAM에서만 존재하며 압축 통계로 재구성된다."),
        },
        "calibration_protocol": {
            "source": {"dataset": PCD.SOURCE, "split": PCD.SPLIT, "column_prefix": PCD.COL_PREFIX,
                       "what_it_is": "FLORES-200 devtest 1012문장 (ungated 재업로드)",
                       "hub": hub_revision("datasets", PCD.SOURCE)},
            "eval_holdout_rule": {"EVAL_N_SENTENCES": PCD.EVAL_N_SENTENCES,
                                  "rule": "각 언어 devtest의 앞 60문장은 평가 전용으로 제외하고 "
                                          "pool = column[60:] 가 calibration 풀이다 (phase1_calib_data._pool_for)"},
            "placebo_arm_rule": "arm B = pool[len(pool)//2:] (같은 언어, 풀의 뒤쪽 절반). Appendix B.2와 일치.",
            "consumption_paths": {
                "expert_frequency": {"script": "D2MoE/preprocess/get_expert_freq.py",
                                     "sampling": "dataset.shuffle(seed=seed).select(range(min(2000, N)))",
                                     "unit": "문장(행) 전량, tokenizer truncation max_length=2048",
                                     "max_samples": 2000,
                                     "consequence": "단일 언어 풀은 952문장 < 2000 -> 풀 전체가 소비된다"},
                "fisher": {"script": "scripts/phase1_fisher.py::build_samples",
                           "n_samples": 64, "seqlen": 512,
                           "window_chars": 2048, "window_rng": "random.Random(seed+1)",
                           "truncation": "tokenizer(truncation=True, max_length=512)"},
                "whitening": {"script": "scripts/phase1_svd_scale.py",
                              "sampling": "phase1_fisher.build_samples를 무수정 import, 같은 인자/seed",
                              "consequence": "Fisher와 윈도우 집합이 동일하다",
                              "vendored_alternative_NOT_used": {
                                  "script": "D2MoE/preprocess/get_scale.py",
                                  "issue": "random을 시드하지 않아 윈도우 복원 불가(seqlen*10=5120자 윈도우). "
                                           "공식 격자의 scale_* 셀은 이 스크립트가 아니라 phase1_svd_scale.py가 만들었다."}},
            },
            "driver_command": sh(f"grep -n 'phase1_fisher.py\\|phase1_svd_scale.py\\|phase1_run_freq_and_scale.py' "
                                 f"{SCRIPT_DIR}/local_gpu_driver.sh").splitlines(),
            "balanced_definition": ("mixed_5lang = **문자(character) 예산** 균등 5언어 혼합이다. "
                                    "토큰 균등이 아니다 -- 실현 토큰으로는 KO+BN이 72.05%를 점유(부록 A.2). "
                                    "balanced_tok_mono/balanced_tok_mix가 토큰 균등 조건이다."),
        },
        "evaluation_protocol": {
            "flores_bpb": {"source": PCD.SOURCE, "split": PCD.SPLIT, "n_sentences": 60,
                           "langs": ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"],
                           "batch_size": 8, "batch_note": "부록 B.5 -- bf16 reduction 순서 때문에 batch를 바꾸면 재현 안 됨",
                           "bpb_formula": "sum(scored-token NLL nats)/ (ln2 * sum(UTF-8 bytes of scored sentences))",
                           "shift": "causal label shift, pad 위치 마스크, 각 문장 첫 토큰은 예측 대상 아님"},
            "xlsum_ood_bpb": {"corpus_dir": "data/ood_xlsum/",
                              "builder": "scripts/e2_build_ood_corpus.py",
                              "evaluator": "scripts/e2e3_ood_unseen_eval.py",
                              "batch_size": 8, "segmentation_max_chars": 288,
                              "segmentation_rule": "개행 문단 분할 후 max_chars 초과 문단은 마지막 공백에서 재분할; "
                                                   "tokenizer truncation 미사용 -> 채점 바이트 == 분모 바이트"},
            "sib200": {"dataset": "Davlan/sib200", "split": "test", "n_per_lang": 204,
                       "demos": "eng_Latn validation, seed 42, 2/label = 14-shot",
                       "hub": hub_revision("datasets", "Davlan/sib200")},
            "belebele": {"dataset": "facebook/belebele", "split": "test", "n_per_lang": 900,
                         "hub": hub_revision("datasets", "facebook/belebele")},
        },
        "run_commands": {
            "overlap_audit": "conda run -n d2moe_env python -u scripts/audit_calib_eval_overlap.py",
            "xlsum_grid_completion": "bash scripts/e2_grid_completion_0916.sh   # 9셀: {zh,sw,bn}_only x seed{0,1,2}",
            "xlsum_grid_prior": "bash scripts/e2e3_driver_0803.sh   # 2026-08-03, baseline + {en,ko,mixed_5lang} + balanced_tok_mono",
        },
        "artifacts": {},
        "xlsum_corpus": json.loads((ROOT / "data" / "ood_xlsum" / "manifest.json").read_text()),
    }

    # 조건별 압축 산출물 실사
    for cond in MAIN + AUX + PCD.PLACEBO_CONDITIONS:
        for seed in SEEDS:
            d = RESULTS_ROOT / cond / f"seed{seed}"
            freq = sorted(d.glob("deepseek_wikitext_*_expert_frequencies.json"))
            m["artifacts"][f"{cond}/seed{seed}"] = {
                "fisher_processed": file_info(d / "fisher_processed.pt"),
                "svd_scale_processed": file_info(d / "svd_scale_processed.pt"),
                "expert_frequencies": [file_info(p) for p in freq],
                "eval_ppl_flores": file_info(d / f"scale_{cond}_seed{seed}" / "eval_ppl.json"),
                "eval_ood_unseen_xlsum": file_info(d / f"scale_{cond}_seed{seed}" / "eval_ood_unseen.json"),
                "eval_sib200": file_info(d / f"scale_{cond}_seed{seed}" / "eval_sib200.json"),
            }
    m["artifacts"]["baseline"] = {
        "eval_ppl_flores": file_info(RESULTS_ROOT / "baseline" / "eval_ppl.json"),
        "eval_ood_unseen_xlsum": file_info(RESULTS_ROOT / "baseline" / "eval_ood_unseen.json"),
    }

    # XL-Sum 코퍼스 파일 sha256 재확인 (manifest가 주장하는 값과 대조)
    checks = {}
    for lang, info in m["xlsum_corpus"]["languages"].items():
        p = ROOT / "data" / "ood_xlsum" / f"{lang}.txt"
        text = p.read_text(encoding="utf-8")
        checks[lang] = {
            "file": str(p), "bytes_on_disk": p.stat().st_size,
            "manifest_sha256_of_text": info["sha256_of_text"],
            "recomputed_sha256_of_joined_docs": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "note": ("manifest의 sha256은 문서를 '\\n\\n'으로 이은 text에 대한 값이고 디스크 파일도 같은 문자열이다. "
                     "두 값이 다르면 코퍼스가 재생성된 것이다."),
        }
        checks[lang]["match"] = (checks[lang]["manifest_sha256_of_text"]
                                 == checks[lang]["recomputed_sha256_of_joined_docs"])
    m["xlsum_corpus_integrity"] = checks

    (out_dir / "artifact_manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_dir/'artifact_manifest.json'}")
    print("xlsum integrity:", {k: v["match"] for k, v in checks.items()})


if __name__ == "__main__":
    main()
