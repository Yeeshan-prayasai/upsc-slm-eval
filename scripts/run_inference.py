"""Stage 4 — run one condition over the eval set.

Usage:
    python scripts/run_inference.py --condition C1a
    python scripts/run_inference.py --condition C1b
    python scripts/run_inference.py --condition C2
    python scripts/run_inference.py --condition C3

Writes per-condition prediction rows to results/predictions.parquet keyed by
(run_id, condition, question_id). Resumable: a re-run reads the existing
parquet and skips already-completed rows.

For C2/C3 the runner estimates total Gemini spend before connecting and
refuses to start above BUDGET_USD unless --confirm-cost is passed.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runners import (  # noqa: E402
    EVAL_TASK_TO_INFERENCE_TASKS,
    EvalItem,
    GeminiFewShotRunner,
    GeminiZeroShotRunner,
    MLXLoRARunner,
    estimate_gemini_cost,
)

REPO = Path(__file__).resolve().parent.parent
EVAL_SET = REPO / "data" / "eval_set.parquet"
FT_CORPUS = REPO / "data" / "ft_corpus.parquet"
OUT_PARQUET = REPO / "results" / "predictions.parquet"

# C1a/C1b use the MERGED + MLX-converted models (Stage 3.5 → mlx_lm convert)
# rather than the raw PEFT/HF adapters. The QLoRA adapter is in HF/PEFT format
# and cannot be loaded as an `adapter_path` by mlx_lm. The conversion path is:
#   EC2:  scripts/merge_adapter.py --base <hf-base> --adapter <peft-adapter>
#                                  --merged-out <merged-hf>
#   M5:   python -m mlx_lm convert --hf-path <merged-hf> --mlx-path <mlx-out>
#                                  -q --q-bits 4 --q-group-size 64
# The resulting mlx-out is loaded directly (no adapter_path).
CONDITIONS = {
    "C1a": ("gemma-FT",  REPO / "adapters/gemma4-e4b-upsc-v1-mlx",  None),
    "C1b": ("qwen-FT",   REPO / "adapters/qwen35-4b-upsc-v1-mlx",   None),
    "C2":  ("gemini-zs", "gemini-3-flash", None),
    "C3":  ("gemini-fs", "gemini-3-flash", None),
}

BUDGET_USD_PER_CONDITION = 25.0


def _build_runner(condition: str):
    short, model_or_path, _adapter = CONDITIONS[condition]
    if condition in ("C1a", "C1b"):
        merged_path = model_or_path
        if not merged_path.exists():
            raise FileNotFoundError(
                f"{condition} expects merged MLX model at {merged_path} (does not exist). "
                f"Run scripts/merge_adapter.py on EC2 then `mlx_lm convert` on the M5 "
                f"to produce it (see Makefile / merge_adapter.py docstring)."
            )
        return MLXLoRARunner(base=str(merged_path), adapter=None), f"{short}@{merged_path.name}"
    if condition == "C2":
        return GeminiZeroShotRunner(model=model_or_path), f"{short}@{model_or_path}"
    if condition == "C3":
        return GeminiFewShotRunner(ft_corpus_path=FT_CORPUS, model=model_or_path), f"{short}@{model_or_path}"
    raise ValueError(condition)


def _shard_path(condition: str, run_id: str) -> Path:
    """Per-(run_id, condition) shard. Each run writes its own file so the
    main predictions.parquet stays a clean union of completed shards and
    no shard rewrites the whole file at every checkpoint."""
    return OUT_PARQUET.parent / "shards" / f"predictions_{run_id}_{condition}.parquet"


def _load_done(condition: str, run_id: str) -> set[tuple[str, str]]:
    """Set of (question_id, inference_task) tuples already completed for this
    (run_id, condition). Reads ONLY the relevant shard, not the merged file,
    so resume is O(shard) instead of O(total_predictions)."""
    shard = _shard_path(condition, run_id)
    if not shard.exists():
        return set()
    df = pd.read_parquet(shard, columns=["question_id", "inference_task"])
    return set(zip(df["question_id"].tolist(), df["inference_task"].tolist()))


def _append_rows(new_rows: list[dict], shard: Path) -> None:
    """Append rows to a per-condition shard via pyarrow's row-group append.
    Avoids the O(n²) read-concat-rewrite pattern; each checkpoint write is
    O(checkpoint_size) regardless of how much data is already in the shard.
    """
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    shard.parent.mkdir(parents=True, exist_ok=True)
    if shard.exists():
        # Read existing shard, append new rows, rewrite. Per-shard scope
        # bounds the rewrite size to one condition's worth of predictions
        # (~3200 rows max) — manageable. A true append-only writer would
        # use pyarrow's ParquetWriter persistently; this is the pragmatic
        # middle ground.
        existing = pd.read_parquet(shard)
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_parquet(shard, index=False, compression="snappy")


def _merge_shards() -> int:
    """Combine all per-(run_id, condition) shards into the main
    predictions.parquet. Idempotent — safe to call repeatedly. Returns the
    total row count of the merged output."""
    shard_dir = OUT_PARQUET.parent / "shards"
    if not shard_dir.exists():
        return 0
    shards = sorted(shard_dir.glob("predictions_*.parquet"))
    if not shards:
        return 0
    frames = [pd.read_parquet(s) for s in shards]
    out = pd.concat(frames, ignore_index=True)
    # Dedup by (run_id, condition, question_id, inference_task) — a row's
    # logical identity. Keep latest occurrence (last shard write wins).
    out = out.drop_duplicates(
        subset=["run_id", "condition", "question_id", "inference_task"],
        keep="last",
    )
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False, compression="snappy")
    return len(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    # Default run_id includes time-of-day so two runs on the same day don't
    # accidentally resume from each other's state. A bug-fix re-run on the
    # same date previously inherited prior shards and silently skipped rows
    # that should have been re-scored.
    ap.add_argument("--run-id", default=dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--limit", type=int, default=None, help="cap items processed (smoke runs)")
    ap.add_argument("--confirm-cost", action="store_true",
                    help="proceed past the budget gate for Gemini conditions")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    # Safety net: abort the run if we see N consecutive errors. Catches API
    # auth-key issues, model not-loaded, etc., before we burn the entire eval.
    ap.add_argument("--max-consecutive-errors", type=int, default=20,
                    help="abort the run after this many consecutive errors")
    args = ap.parse_args()

    if not EVAL_SET.exists():
        print(f"[FAIL] {EVAL_SET} not found; run `make freeze` first")
        return 1

    if args.condition == "C3" and not FT_CORPUS.exists():
        print(f"[FAIL] C3 (few-shot) requires {FT_CORPUS}; run `make build-ft-corpus` first")
        return 1

    if args.condition in ("C2", "C3"):
        est = estimate_gemini_cost(EVAL_SET, few_shot=(args.condition == "C3"))
        print(f"Estimated Gemini cost for {args.condition}: ${est:.2f} (budget ${BUDGET_USD_PER_CONDITION:.2f}/condition)")
        if est > BUDGET_USD_PER_CONDITION and not args.confirm_cost:
            print(f"[REFUSE] estimate > budget. Re-run with --confirm-cost to proceed.")
            return 1

    eval_df = pd.read_parquet(EVAL_SET)
    done = _load_done(args.condition, args.run_id)

    # Expand eval rows × inference_tasks. One eval row may produce N>=1
    # prediction rows (Task A → [A, F]; Task B → [B, G]).
    work: list[tuple[dict, str]] = []
    for row in eval_df.to_dict("records"):
        for inf_task in EVAL_TASK_TO_INFERENCE_TASKS.get(row["task"], [row["task"]]):
            if (row["question_id"], inf_task) in done:
                continue
            work.append((row, inf_task))
    if args.limit:
        work = work[: args.limit]

    if not work:
        print(f"[OK] {args.condition} run_id={args.run_id}: nothing to do "
              f"({len(done)} rows already in {OUT_PARQUET})")
        return 0

    print(f"Condition: {args.condition}  run_id: {args.run_id}")
    print(f"Pending:   {len(work):,} prediction rows  (skipped {len(done):,} already done)\n")

    runner, model_version = _build_runner(args.condition)
    shard = _shard_path(args.condition, args.run_id)
    print(f"Shard:     {shard}")

    buffer: list[dict] = []
    n_ok = n_err = 0
    consecutive_errors = 0
    aborted = False
    for i, (row, inf_task) in enumerate(work, 1):
        item = EvalItem.from_row(row)
        try:
            pred = runner.predict(item, inference_task=inf_task)
            extras: dict = {}
            # Task A: separate verbal-confidence call (Pass-2). Tasks F/G are
            # capability tests with provided/different gold and don't need it.
            # `confidence()` returns 0-100 int or None on parse failure —
            # recorded verbatim. None ⇒ score_task_A leaves brier_loss as NaN
            # rather than averaging in a silent 0.5 default.
            if inf_task == "A" and pred.parsed.get("answer"):
                extras = {"confidence": runner.confidence(item, pred.parsed["answer"])}
            buffer.append({
                "run_id": args.run_id,
                "condition": args.condition,
                "model_version": model_version,
                "task": inf_task,                         # inference-task (A/B/C/E/F/G)
                "eval_task": item.task,                   # source eval-task (A/B/C/E)
                "inference_task": inf_task,
                "question_id": item.question_id,
                "language": item.language,
                "paper": item.paper,
                "subject": item.subject,
                "stratum_key": item.stratum_key,
                "input_text": "",
                "gold_payload": row["gold_payload"],
                "prediction": json.dumps({**pred.parsed, **extras}, ensure_ascii=False),
                "raw_output": pred.raw,
                "latency_ms": pred.latency_ms,
                "ttft_ms": pred.ttft_ms,
                "input_tokens": pred.input_tokens,
                "output_tokens": pred.output_tokens,
                "created_at": dt.datetime.now(dt.UTC).isoformat(),
            })
            n_ok += 1
            consecutive_errors = 0
        except KeyboardInterrupt:
            print("\n[INTERRUPT] flushing buffer before exit ...")
            _append_rows(buffer, shard)
            _merge_shards()
            raise
        except Exception as e:
            n_err += 1
            consecutive_errors += 1
            print(f"  [{i}/{len(work)}] ERR {item.question_id} [{inf_task}]: {type(e).__name__}: {e}")
            if consecutive_errors >= args.max_consecutive_errors:
                print(f"\n[ABORT] {consecutive_errors} consecutive errors — aborting "
                      f"run to avoid burning the full eval on a likely systemic failure "
                      f"(API auth, model not loaded, etc.).")
                _append_rows(buffer, shard)
                aborted = True
                break

        if i % args.checkpoint_every == 0:
            _append_rows(buffer, shard); buffer = []
            print(f"  [{i:>5d}/{len(work)}] checkpoint ok={n_ok} err={n_err}")

    _append_rows(buffer, shard)
    total = _merge_shards()
    print(f"\n[{'ABORTED' if aborted else 'OK'}] {args.condition} run_id={args.run_id}: "
          f"ok={n_ok} err={n_err} shard={shard} merged_total={total}")
    if aborted:
        return 3
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
