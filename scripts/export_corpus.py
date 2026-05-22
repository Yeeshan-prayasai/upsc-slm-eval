"""Export a Parquet corpus to JSONL for human inspection.

One JSON object per line: `head`, `grep`, `jq`, `less` all work normally.
The JSONL is deterministically rebuildable from the parquet (and gitignored).

Usage:
    python scripts/export_corpus.py                              # ft_corpus.parquet → ft_corpus.jsonl
    python scripts/export_corpus.py --in data/eval_set.parquet --out data/eval_set.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, default=Path("data/ft_corpus.parquet"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if not args.src.exists():
        print(f"[FAIL] {args.src} not found")
        return 1

    out = args.out or args.src.with_suffix(".jsonl")
    df = pd.read_parquet(args.src)
    with out.open("w", encoding="utf-8") as f:
        for r in df.to_dict("records"):
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")

    print(f"[OK] {len(df):,} rows → {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    if "task" in df.columns:
        print(f"     by task: {df.groupby('task').size().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
