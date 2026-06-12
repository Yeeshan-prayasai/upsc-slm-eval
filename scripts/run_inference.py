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
    HFTransformersRunner,
    MLXLoRARunner,
    estimate_gemini_cost,
)

REPO = Path(__file__).resolve().parent.parent
EVAL_SET = REPO / "data" / "eval_set.parquet"
FT_CORPUS = REPO / "data" / "ft_corpus.parquet"
OUT_PARQUET = REPO / "results" / "predictions.parquet"

# C1a/C1b are FT-SLM conditions. Backend is dispatched by INFERENCE_BACKEND
# env var (or --backend CLI flag):
#   "mlx" (default) → load MLX 4-bit dir on Apple Silicon (M5).
#   "hf"            → load merged HF dir in bf16 on NVIDIA GPU (EC2 / cloud).
# The merged HF dir is the source for both — `mlx_lm convert` consumes it
# to produce the MLX dir for M5, while HF/transformers loads the same dir
# directly on the EC2 L40S.
#
# The §6.4 universal-metric implication is documented in eval-design §3.1:
# latency/TTFT/tokens-per-sec reflect whichever device actually runs the
# eval, so reports should call out the inference backend explicitly.
CONDITIONS = {
    # condition: (short, mlx_path, hf_path, model_or_api)
    "C1a": ("gemma-FT",
            REPO / "adapters/gemma4-e4b-upsc-v1-mlx",
            REPO / "adapters/gemma4-e4b-upsc-v1-merged"),
    "C1b": ("qwen-FT",
            REPO / "adapters/qwen35-4b-upsc-v1-mlx",
            REPO / "adapters/qwen35-4b-upsc-v1-merged"),
    "C2":  ("gemini-zs", None, None),
    "C3":  ("gemini-fs", None, None),
}
# Gemini model name — env-configurable so we can pin a different snapshot
# (e.g. gemini-3.5-flash vs gemini-3-flash-preview) without code changes.
# Default tracks the pre-registered name; runtime override via $GEMINI_MODEL.
import os as _os_for_gemini
GEMINI_MODEL = _os_for_gemini.getenv("GEMINI_MODEL", "gemini-3.5-flash")

BUDGET_USD_PER_CONDITION = 25.0


def _build_runner(condition: str, backend: str):
    """Dispatch the right runner for `condition` + `backend`.

    backend ∈ {"mlx", "hf"} controls C1a/C1b loading:
      mlx → MLXLoRARunner against the MLX-converted dir (M5 / Apple Silicon)
      hf  → HFTransformersRunner against the merged HF dir (NVIDIA GPU)
    C2/C3 are unaffected — Gemini API calls work the same regardless of where
    this script runs.
    """
    spec = CONDITIONS[condition]
    short = spec[0]
    if condition in ("C1a", "C1b"):
        _, mlx_path, hf_path = spec
        if backend == "mlx":
            if not mlx_path.exists():
                raise FileNotFoundError(
                    f"{condition} (backend=mlx) expects MLX dir at {mlx_path}. "
                    f"Run `python -m mlx_lm convert --hf-path {hf_path} --mlx-path {mlx_path} "
                    f"-q --q-bits 4 --q-group-size 64` to produce it."
                )
            return MLXLoRARunner(base=str(mlx_path), adapter=None), f"{short}@mlx:{mlx_path.name}"
        if backend == "hf":
            if not hf_path.exists():
                raise FileNotFoundError(
                    f"{condition} (backend=hf) expects merged HF dir at {hf_path}. "
                    f"Run scripts/merge_adapter.py on a CUDA host to produce it."
                )
            return HFTransformersRunner(hf_path=str(hf_path)), f"{short}@hf:{hf_path.name}"
        raise ValueError(f"unknown backend: {backend!r} (expected 'mlx' or 'hf')")
    if condition == "C2":
        return GeminiZeroShotRunner(model=GEMINI_MODEL), f"{short}@{GEMINI_MODEL}"
    if condition == "C3":
        return GeminiFewShotRunner(ft_corpus_path=FT_CORPUS, model=GEMINI_MODEL), f"{short}@{GEMINI_MODEL}"
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
    # FT-SLM backend dispatch — see _build_runner / CONDITIONS docstring.
    # Default falls back to the INFERENCE_BACKEND env var so Makefile targets
    # don't need to know which device they're on.
    import os as _os
    ap.add_argument("--backend", choices=("mlx", "hf"),
                    default=_os.getenv("INFERENCE_BACKEND", "mlx"),
                    help="C1a/C1b loader: mlx (M5/Apple) or hf (NVIDIA GPU). "
                         "Default from $INFERENCE_BACKEND or 'mlx'.")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="HF runner: group N predictions into one model.generate() "
                         "call. >1 trades per-row TTFT precision for ~5-10× "
                         "throughput on GPU. No-op for MLX/Gemini backends.")
    # v2 adapter override: point C1a/C1b at a different merged-HF / MLX dir
    # (a v2 cell adapter) without touching the hardcoded v1 CONDITIONS.
    ap.add_argument("--adapter-dir", type=Path, default=None,
                    help="Override the C1a/C1b model dir (merged HF dir for "
                         "--backend hf, MLX dir for --backend mlx). Used by "
                         "the v2/ablation eval targets; default = v1 paths.")
    args = ap.parse_args()

    if args.adapter_dir is not None and args.condition in ("C1a", "C1b"):
        short, _mlx, _hf = CONDITIONS[args.condition]
        CONDITIONS[args.condition] = (short, args.adapter_dir, args.adapter_dir)
        print(f"[run] {args.condition} adapter override → {args.adapter_dir}")

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

    runner, model_version = _build_runner(args.condition, args.backend)
    shard = _shard_path(args.condition, args.run_id)
    print(f"Backend:   {args.backend}")
    print(f"Batch:     {args.batch_size}")
    print(f"Shard:     {shard}")

    # Batched path: only if the runner supports it AND batch_size > 1. The
    # batched path groups predictions by inference_task (so each batch has a
    # uniform max_new_tokens) and processes them together.
    use_batched = (
        args.batch_size > 1
        and hasattr(runner, "predict_batch")
        and args.condition in ("C1a", "C1b")
    )

    def _row_to_dict(item: EvalItem, inf_task: str, row: dict,
                     pred, extras: dict) -> dict:
        return {
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
        }

    buffer: list[dict] = []
    n_ok = n_err = 0
    consecutive_errors = 0
    aborted = False

    if use_batched:
        # Group work items by inference_task so each batch has uniform shape +
        # uniform max_new_tokens (otherwise the largest cap dominates).
        from collections import defaultdict
        groups: dict[str, list[tuple[int, dict, EvalItem]]] = defaultdict(list)
        for i, (row, inf_task) in enumerate(work, 1):
            groups[inf_task].append((i, row, EvalItem.from_row(row)))
        processed = 0
        for inf_task, group in groups.items():
            for start in range(0, len(group), args.batch_size):
                chunk = group[start:start + args.batch_size]
                items = [it for _, _, it in chunk]
                tasks = [inf_task] * len(items)
                try:
                    preds = runner.predict_batch(items, tasks)
                    # Pass-2 verbal confidence — batched for Task A.
                    conf_results: list = []
                    if inf_task == "A":
                        valid_idx = [k for k, p in enumerate(preds) if p.parsed.get("answer")]
                        if valid_idx and hasattr(runner, "confidence_batch"):
                            conf_letters = [preds[k].parsed["answer"] for k in valid_idx]
                            conf_items = [items[k] for k in valid_idx]
                            conf_out = runner.confidence_batch(conf_items, conf_letters)
                            conf_map = dict(zip(valid_idx, conf_out))
                            conf_results = [conf_map.get(k) for k in range(len(items))]
                        else:
                            conf_results = [None] * len(items)
                    for (orig_i, row, item), pred, conf in zip(
                        chunk, preds,
                        conf_results if conf_results else [None] * len(items),
                    ):
                        extras = {"confidence": conf} if (inf_task == "A" and pred.parsed.get("answer")) else {}
                        buffer.append(_row_to_dict(item, inf_task, row, pred, extras))
                    n_ok += len(items)
                    consecutive_errors = 0
                    processed += len(items)
                except KeyboardInterrupt:
                    print("\n[INTERRUPT] flushing buffer before exit ...")
                    _append_rows(buffer, shard); _merge_shards()
                    raise
                except Exception as e:
                    n_err += len(items)
                    consecutive_errors += len(items)
                    qids = ",".join(it.question_id for it in items[:3])
                    print(f"  [{processed+1}-{processed+len(items)}/{len(work)}] "
                          f"BATCH ERR [{inf_task}] ({qids}...): {type(e).__name__}: {e}")
                    processed += len(items)
                    if consecutive_errors >= args.max_consecutive_errors:
                        print(f"\n[ABORT] {consecutive_errors} consecutive errors — aborting.")
                        _append_rows(buffer, shard)
                        aborted = True
                        break

                if len(buffer) >= args.checkpoint_every:
                    _append_rows(buffer, shard); buffer = []
                    print(f"  [{processed:>5d}/{len(work)}] checkpoint ok={n_ok} err={n_err}")
            if aborted:
                break
    else:
        # Per-row (single-prediction) path — used by MLX, Gemini, or HF with
        # batch_size=1.
        for i, (row, inf_task) in enumerate(work, 1):
            item = EvalItem.from_row(row)
            try:
                pred = runner.predict(item, inference_task=inf_task)
                extras: dict = {}
                # Task A: separate verbal-confidence call (Pass-2). Tasks F/G
                # are capability tests with provided/different gold and don't
                # need it. `confidence()` returns 0-100 int or None on parse
                # failure — recorded verbatim. None ⇒ score_task_A leaves
                # brier_loss as NaN rather than averaging in a silent 0.5.
                if inf_task == "A" and pred.parsed.get("answer"):
                    extras = {"confidence": runner.confidence(item, pred.parsed["answer"])}
                buffer.append(_row_to_dict(item, inf_task, row, pred, extras))
                n_ok += 1
                consecutive_errors = 0
            except KeyboardInterrupt:
                print("\n[INTERRUPT] flushing buffer before exit ...")
                _append_rows(buffer, shard); _merge_shards()
                raise
            except Exception as e:
                n_err += 1
                consecutive_errors += 1
                print(f"  [{i}/{len(work)}] ERR {item.question_id} [{inf_task}]: {type(e).__name__}: {e}")
                if consecutive_errors >= args.max_consecutive_errors:
                    print(f"\n[ABORT] {consecutive_errors} consecutive errors — aborting.")
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
