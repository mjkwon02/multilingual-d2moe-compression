"""Qwen1.5-MoE-A2.7B downstream evaluation, reusing the DeepSeek evaluators
unchanged (they take any (model, tokenizer)):

  sib200   phase1_sib200_eval.eval_sib200  -- main downstream task (14-shot,
           English demos, natural-continuation scoring). Its single-token check
           on the 7 answer words raises if the Qwen tokenizer splits any.
  belebele phase1_belebele_eval.eval_belebele (lm-eval harness), supporting only
           (near-chance on DeepSeek); --num-fewshot 5 matches the DeepSeek grid.
  xlsum    e2e3_ood_unseen_eval.eval_ood (XL-Sum OOD bpb, fixed corpus in
           data/ood_xlsum/) + eval_unseen (4 unseen FLORES languages).

The model is the baseline, the compressed model rebuilt from compressed_model/
(--from-saved, preferred: no Fisher/whitening recomputation), or a fresh
in-process merge. Accuracy is written raw; retention (acc / baseline acc) is
computed by qwen15_report.py -- nothing here drops a language, seed or task.

Not run in the single-condition trial.

Usage:
    python -u qwen15_downstream_eval.py --baseline --tasks sib200
    python -u qwen15_downstream_eval.py --condition english_only --seed 0 --from-saved --tasks sib200 xlsum
"""
import argparse
import gc
import math
import sys
import traceback
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import qwen15_config as QC  # noqa: E402
import qwen15_merge_eval as QM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    QM.add_model_args(ap)
    ap.add_argument("--from-saved", action="store_true")
    ap.add_argument("--tasks", nargs="+", required=True, choices=["sib200", "belebele", "xlsum"])
    ap.add_argument("--sib-batch-size", type=int, default=4)
    ap.add_argument("--belebele-batch-size", type=int, default=4)
    ap.add_argument("--num-fewshot", type=int, default=5, help="Belebele shots (DeepSeek grid: fewshot_5)")
    args = ap.parse_args()
    QM.resolve_args(ap, args)
    QC.check_env()

    out_dir = QC.RESULTS_ROOT / "baseline" if args.baseline else QM.out_dir_for(
        args.condition, args.seed, args.scale_condition, args.scale_seed, args.plain_svd)
    targets = {"sib200": out_dir / "eval_sib200.json",
               "belebele": out_dir / f"fewshot_{args.num_fewshot}" / "eval_belebele.json",
               "xlsum": out_dir / "eval_ood_unseen.json"}
    for t in args.tasks:
        if targets[t].exists():
            raise FileExistsError(f"{targets[t]} exists -- refusing to overwrite")

    rec = QC.ResourceRecorder()
    run = {"tasks": args.tasks, "env": QC.environment_record(), "status": "running"}
    try:
        st = rec.begin("build_model")
        model, tokenizer, src = QM.build_model_for_eval(args)
        gc.collect()
        torch.cuda.empty_cache()
        run["model_source"] = src
        rec.end(st)
        tag = {"model_id": QC.MODEL_ID, "model_revision": QC.MODEL_REVISION, **src,
               "condition": args.condition, "seed": args.seed, "scale_condition": args.scale_condition,
               "scale_seed": args.scale_seed, "plain_svd": args.plain_svd}

        if "sib200" in args.tasks:
            import phase1_sib200_eval as SIB
            st = rec.begin("sib200")
            res = SIB.eval_sib200(model, tokenizer, batch_size=args.sib_batch_size, limit=None)
            nan = [l for l in SIB.SIB_LANGS if not math.isfinite(res[l]["acc"])]
            if nan:
                raise RuntimeError(f"NaN SIB accuracy for {nan}")
            QC.write_json_new(targets["sib200"], {"protocol": "sib200_natural_continuation_v1",
                                                  "num_fewshot": len(SIB.CATEGORY_ORDER) * SIB.DEMOS_PER_LABEL_DEFAULT,
                                                  "demo_seed": SIB.DEMO_SEED_DEFAULT, "run": tag, "results": res})
            rec.end(st)
        if "belebele" in args.tasks:
            import phase1_belebele_eval as BB
            st = rec.begin("belebele")
            res = BB.eval_belebele(model, tokenizer, batch_size=args.belebele_batch_size, num_fewshot=args.num_fewshot)
            QC.write_json_new(targets["belebele"], {"num_fewshot": args.num_fewshot, "run": tag, "results": res})
            rec.end(st)
        if "xlsum" in args.tasks:
            import e2e3_ood_unseen_eval as E2
            st = rec.begin("xlsum_ood_unseen")
            model.eval()
            with torch.no_grad():
                ood, manifest = E2.eval_ood(model, tokenizer)
                unseen = E2.eval_unseen(model, tokenizer)
            QC.write_json_new(targets["xlsum"], {
                "run": tag, "eval_batch_size": E2.EVAL_BATCH_SIZE,
                "segmentation": {"max_chars": E2.SEG_MAX_CHARS},
                "e2_ood": {"corpus_id": manifest["corpus_id"], "results": ood},
                "e3_unseen": {"languages": list(E2.UNSEEN_LANGS), "results": unseen}})
            rec.end(st)
        run["status"] = "ok"
    except BaseException as exc:
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        run["traceback"] = traceback.format_exc()
        raise
    finally:
        rec.close()
        run["resources"] = rec.summary()
        out_dir.mkdir(parents=True, exist_ok=True)
        QC.write_json_versioned(out_dir / "downstream_run_meta.json", run)


if __name__ == "__main__":
    main()
