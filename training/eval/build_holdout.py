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
DB_PATH = RepoPaths.db_snapshot()
EVAL_SET = REPO / "data" / "eval_set.parquet"
FT_CORPUS = REPO / "data" / "ft_corpus.parquet"
OUT_PATH = REPO / "data" / "eval_set_holdout.parquet"
# Only ~124 mcqs rows exist outside eval+FT-corpus (the eval set drew
# heavily from every question table); 120 is the honest maximum and is
# ample for the 50-question in-training pulse probe.
DEFAULT_N = 120
SEED = 20260611


def _excluded_ids() -> set[str]:
    """Question ids already used by the locked eval set or the FT corpus.

    Eval ids are namespaced `<ns>:<uuid>:<lang>` while DB primary keys
    are bare uuids — comparison happens at BASE-UUID level (comparing
    the namespaced string against a bare uuid never matches, which
    previously let 77/200 probe questions be locked-eval siblings)."""
    excluded: set[str] = set()
    if EVAL_SET.exists():
        for qid in pd.read_parquet(EVAL_SET)["question_id"].astype(str):
            parts = qid.split(":")
            excluded.add(parts[1] if len(parts) >= 2 else qid)
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
        df_ai = pd.read_sql_query(
            "SELECT id, question_english, options_english, answer "
            "FROM upsc_prelims_ai_generated_que "
            "WHERE quality_pass_flag = 1 AND question_english IS NOT NULL",
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

    # The strict sibling exclusion leaves <200 eligible mcqs rows (the
    # eval set drew heavily from that table) — top up from the
    # AI-generated question bank under the same exclusion rules.
    # Schema: options_english is a dict {"a": text, ...}; answer is "A".."D".
    for _, r in df_ai.sort_values("id").iterrows():
        qid = str(r["id"])
        if qid in excluded:
            continue
        try:
            opts_raw = json.loads(r["options_english"] or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(opts_raw, dict):
            continue
        opts = {str(k).upper(): str(v).strip() for k, v in opts_raw.items()}
        gold = str(r["answer"] or "").strip().upper()
        if set(opts) != {"A", "B", "C", "D"} or not all(opts.values()):
            continue
        if gold not in "ABCD" or not gold:
            continue
        q = str(r["question_english"] or "").strip()
        if not q:
            continue
        rows.append({
            "question_id": qid,
            "task": "A",
            "language": "en",
            "question": q,
            "options": json.dumps(opts, ensure_ascii=False),
            "correct_option_letter": gold,
        })

    if len(rows) < n:
        print(f"ERROR: only {len(rows)} eligible held-out MCQs (< {n}).",
              file=sys.stderr)
        return 1

    rng = random.Random(seed)
    rng.shuffle(rows)
    out = pd.DataFrame(rows[:n])
    # Hard assertion: no probe question may be a locked-eval sibling.
    overlap = set(out["question_id"]) & excluded
    if overlap:
        raise RuntimeError(
            f"holdout sampled {len(overlap)} locked-eval siblings — "
            f"exclusion logic regressed: {sorted(overlap)[:3]}"
        )
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
