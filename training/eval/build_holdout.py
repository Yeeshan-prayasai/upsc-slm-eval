"""Build `data/eval_set_holdout.parquet` — the in-training task-pulse probe.

200 single-select English MCQs from the local DB snapshot's `mcqs`
table that appear in NEITHER the locked eval set NOR the v1 ft_corpus
(and therefore not in the v2 SFT corpus, which derives from it).
The pulse callback (`training/eval/pulse.py`) reads this file every
`task_every_steps` to track Task-A accuracy during CPT/SFT.

Output schema (what `PulseEvalCallback._items_to_mcq` consumes):
    question_id, task="A", language="en",
    question, options (dict {"A": text, ...}), correct_option_letter

Deterministic: fixed seed, sorted candidate order before sampling.

CLI:
    python -m training.eval.build_holdout
    python -m training.eval.build_holdout --n 200
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from ..data.acquire._base import RepoPaths

REPO = RepoPaths.root()
DB_PATH = REPO / "data" / "prayas_local.sqlite"
EVAL_SET = REPO / "data" / "eval_set.parquet"
FT_CORPUS = REPO / "data" / "ft_corpus.parquet"
OUT_PATH = REPO / "data" / "eval_set_holdout.parquet"
DEFAULT_N = 200
SEED = 20260611


def _excluded_ids() -> set[str]:
    """Question ids already used by the locked eval set or the FT corpus."""
    excluded: set[str] = set()
    if EVAL_SET.exists():
        excluded |= set(pd.read_parquet(EVAL_SET)["question_id"].astype(str))
    if FT_CORPUS.exists():
        for pair_id in pd.read_parquet(FT_CORPUS, columns=["pair_id"])["pair_id"]:
            parts = str(pair_id).split(":")
            if len(parts) >= 2:
                excluded.add(parts[1])
    return excluded


def build(n: int = DEFAULT_N, out_path: Path = OUT_PATH, seed: int = SEED) -> int:
    if not DB_PATH.exists():
        print(f"ERROR: local DB snapshot not found at {DB_PATH}", file=sys.stderr)
        return 1
    excluded = _excluded_ids()
    print(f"Excluded ids (eval set + ft_corpus): {len(excluded):,}")

    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(
            "SELECT id, questionText, options, correctOptionIds, isMultiSelect "
            "FROM mcqs WHERE isMultiSelect = 0",
            con,
        )

    rows: list[dict] = []
    for _, r in df.sort_values("id").iterrows():
        qid = str(r["id"])
        if qid in excluded:
            continue
        try:
            opts_raw = json.loads(r["options"] or "[]")
            correct = json.loads(r["correctOptionIds"] or "[]")
        except json.JSONDecodeError:
            continue
        # Options come as [{"id": "a", "text": ...}] with lowercase ids;
        # correctOptionIds as ["D"] uppercase. Normalize to {A..D} + letter.
        opts = {str(o.get("id", "")).upper(): str(o.get("text", "")).strip()
                for o in opts_raw if isinstance(o, dict)}
        if set(opts) != {"A", "B", "C", "D"} or not all(opts.values()):
            continue
        if len(correct) != 1 or str(correct[0]).upper() not in "ABCD":
            continue
        q = str(r["questionText"] or "").strip()
        if not q:
            continue
        rows.append({
            "question_id": qid,
            "task": "A",
            "language": "en",
            "question": q,
            "options": json.dumps(opts, ensure_ascii=False),
            "correct_option_letter": str(correct[0]).upper(),
        })

    if len(rows) < n:
        print(f"ERROR: only {len(rows)} eligible held-out MCQs (< {n}).",
              file=sys.stderr)
        return 1

    rng = random.Random(seed)
    rng.shuffle(rows)
    out = pd.DataFrame(rows[:n])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Holdout: {len(out)} MCQs (from {len(rows):,} eligible) → "
          f"{out_path.relative_to(REPO)}")
    return 0


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(description="Build the task-pulse holdout probe.")
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)
    return build(n=args.n, out_path=args.out, seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
