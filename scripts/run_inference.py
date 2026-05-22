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

CONDITIONS = {
    "C1a": ("gemma-FT",  "mlx-community/gemma-4-e4b-it-4bit",  REPO / "adapters/gemma4-e4b-upsc-v1"),
    "C1b": ("qwen-FT",   "mlx-community/Qwen3.5-4B-MLX-4bit",  REPO / "adapters/qwen35-4b-upsc-v1"),
    "C2":  ("gemini-zs", "gemini-3-flash", None),
    "C3":  ("gemini-fs", "gemini-3-flash", None),
}

BUDGET_USD_PER_CONDITION = 25.0


def _build_runner(condition: str):
    short, model, adapter = CONDITIONS[condition]
    if condition in ("C1a", "C1b"):
        return MLXLoRARunner(base=model, adapter=str(adapter)), f"{short}@{adapter.name}"
    if condition == "C2":
        return GeminiZeroShotRunner(model=model), f"{short}@{model}"
    if condition == "C3":
        return GeminiFewShotRunner(ft_corpus_path=FT_CORPUS, model=model), f"{short}@{model}"
    raise ValueError(condition)


def _load_done(condition: str, run_id: str) -> set[str]:
    if not OUT_PARQUET.exists():
        return set()
    df = pd.read_parquet(OUT_PARQUET)
    mask = (df["run_id"] == run_id) & (df["condition"] == condition)
    return set(df.loc[mask, "question_id"].tolist())


def _append_rows(new_rows: list[dict]) -> None:
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    if OUT_PARQUET.exists():
        existing = pd.read_parquet(OUT_PARQUET)
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False, compression="snappy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    ap.add_argument("--run-id", default=dt.datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--limit", type=int, default=None, help="cap items processed (smoke runs)")
    ap.add_argument("--confirm-cost", action="store_true",
                    help="proceed past the budget gate for Gemini conditions")
    ap.add_argument("--checkpoint-every", type=int, default=50)
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
    pending = eval_df[~eval_df["question_id"].isin(done)]
    if args.limit:
        pending = pending.head(args.limit)
    if pending.empty:
        print(f"[OK] {args.condition} run_id={args.run_id}: nothing to do "
              f"({len(done)} rows already in {OUT_PARQUET})")
        return 0

    print(f"Condition: {args.condition}  run_id: {args.run_id}")
    print(f"Pending:   {len(pending):,} of {len(eval_df):,}  (skipped {len(done):,} already done)\n")

    runner, model_version = _build_runner(args.condition)

    buffer: list[dict] = []
    n_ok = n_err = 0
    for i, row in enumerate(pending.to_dict("records"), 1):
        item = EvalItem.from_row(row)
        try:
            pred = runner.predict(item)
            extras: dict = {}
            if item.task == "A" and pred.parsed.get("answer"):
                # Pass-1 JSON already carries the explanation; only confidence
                # needs a separate call (verbal-confidence elicitation).
                extras = {"confidence": runner.confidence(item, pred.parsed["answer"])}
            buffer.append({
                "run_id": args.run_id,
                "condition": args.condition,
                "model_version": model_version,
                "task": item.task,
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
                "created_at": dt.datetime.utcnow().isoformat(),
            })
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"  [{i}/{len(pending)}] ERR {item.question_id}: {type(e).__name__}: {e}")

        if i % args.checkpoint_every == 0:
            _append_rows(buffer); buffer = []
            print(f"  [{i:>4d}/{len(pending)}] checkpoint ok={n_ok} err={n_err}")

    _append_rows(buffer)
    print(f"\n[OK] {args.condition} run_id={args.run_id}: ok={n_ok} err={n_err} → {OUT_PARQUET}")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
