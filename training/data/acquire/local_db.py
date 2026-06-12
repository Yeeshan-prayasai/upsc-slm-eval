"""Extract English-only text from the local DB snapshot.

Reads the local SQLite snapshot produced by `scripts/snapshot_to_local.py`
(the only script authorized to touch the remote prod DB) and emits one
plain-text file per source table under `data/cpt_raw/local_db/`.

LEAKAGE GUARD: the per-row primary key (question_id / id) is recorded
in the manifest. The corpus-build leakage gate (`training.data.leakage`)
cross-references these against `data/eval_set.parquet` and refuses to
start training if any overlap survives. Rows whose question_id appears
in the eval set are SKIPPED at extract time, not at the leakage gate —
earlier is cheaper.

CLI:
    python -m training.data.acquire.local_db
    python -m training.data.acquire.local_db --only mcqs pyqs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ._base import Manifest, ManifestEntry, RepoPaths, now_iso

REPO = RepoPaths.root()
SNAPSHOT = REPO / "data" / "prayas_local.sqlite"
EVAL_SET = REPO / "data" / "eval_set.parquet"


@dataclass(frozen=True)
class Extract:
    """One extraction job — picks columns from a table and joins them into
    one text blob per row.

    `text_columns` — concatenated with `\\n\\n` between columns.
    `id_column` — primary key recorded in the manifest for leakage check.
    `filters` — optional SQL WHERE clauses, AND'd together.
    """
    table: str
    text_columns: list[str]
    id_column: str
    label_columns: list[str]
    filters: list[str]
    eval_id_column: str | None = None  # which eval_set row.question_id maps to this table


# Order matters only for logging; rows from each table go to their own file.
EXTRACTS: list[Extract] = [
    Extract(
        table="prelims_pyq_questions",
        text_columns=["question", "explanation"],
        id_column="question_id",
        label_columns=["year", "paper", "subject"],
        filters=["is_dropped IS NULL OR is_dropped = 0"],
        eval_id_column="question_id",
    ),
    Extract(
        table="pyqs",
        text_columns=["question", "model_answer", "hints"],
        id_column="question_id",
        label_columns=["paper", "subject", "year", "section", "word_count"],
        filters=[],
        eval_id_column="question_id",
    ),
    Extract(
        table="mcqs",
        text_columns=["questionText", "explanation"],
        id_column="id",
        label_columns=["paper", "question_pattern"],
        filters=[],
        eval_id_column=None,   # mcqs is a separate pool from eval-locked pyq sets
    ),
    Extract(
        table="evaluation_questions",
        text_columns=["question_text", "answer_text", "strengths",
                      "improvements", "model_answer"],
        id_column="question_id",
        label_columns=["subject", "score", "max_score", "word_count"],
        filters=[],
        eval_id_column="question_id",
    ),
    Extract(
        table="news_articles",
        text_columns=["title", "text", "prelimsInfo", "mainsInfo"],
        id_column="id",
        label_columns=["date", "source"],
        filters=[],
        eval_id_column=None,
    ),
    Extract(
        table="current_affairs",
        text_columns=["heading", "description", "pointed_analysis",
                      "mains_analysis", "prelims_info"],
        id_column="id",
        label_columns=["date", "source", "mains_subject", "prelims_subject"],
        filters=[],
        eval_id_column=None,
    ),
    Extract(
        table="upsc_prelims_ai_generated_que",
        # English-only columns; the Hindi variants stay deferred to v2-hindi-strategy.
        text_columns=["question_english", "options_english", "explanation"],
        id_column="id",
        label_columns=["subject", "topic", "difficulty", "month", "year"],
        filters=["quality_pass_flag = 1",
                 "question_english IS NOT NULL AND TRIM(question_english) <> ''"],
        eval_id_column=None,
    ),
    Extract(
        table="article_generated_questions",
        text_columns=["question", "options", "model_answer", "explanation", "hints"],
        id_column="question_id",
        label_columns=["paper", "subject", "year", "difficulty", "type"],
        filters=["question IS NOT NULL AND TRIM(question) <> ''"],
        eval_id_column="question_id",
    ),
    Extract(
        table="glossary",
        text_columns=["keyword", "definition"],
        id_column="id",
        label_columns=[],
        filters=["definition IS NOT NULL AND TRIM(definition) <> ''"],
        eval_id_column=None,
    ),
]


def _load_eval_ids() -> set[str]:
    """Frozen eval-set question_ids — any row whose id is here is
    skipped at extract time to prevent corpus leakage."""
    if not EVAL_SET.exists():
        return set()
    df = pd.read_parquet(EVAL_SET, columns=["question_id"])
    return set(df["question_id"].astype(str).tolist())


def _row_to_text(row: dict, ext: Extract) -> str:
    parts = []
    for col in ext.text_columns:
        v = row.get(col)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            parts.append(s)
    return "\n\n".join(parts)


def extract_table(
    conn: sqlite3.Connection,
    ext: Extract,
    eval_ids: set[str],
    manifest: Manifest,
) -> dict:
    """Pull one table → one .txt file per ~5K-row chunk + one .index.jsonl
    (per-row id, source columns, byte offsets in the .txt file). Idempotent:
    if the same (table, row_range) chunk already exists in the manifest,
    skip it.
    """
    cols = ext.text_columns + ext.label_columns + [ext.id_column]
    where = " AND ".join(f"({c})" for c in ext.filters) if ext.filters else "1=1"
    sql = f"SELECT {', '.join(cols)} FROM {ext.table} WHERE {where} ORDER BY {ext.id_column}"

    out_dir = RepoPaths.cpt_raw("local_db") / ext.table
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_rows = 5_000
    txt_path = out_dir / "rows.txt"
    idx_path = out_dir / "rows.index.jsonl"

    n_total = 0
    n_skipped_leakage = 0
    n_skipped_empty = 0
    total_bytes = 0
    h = hashlib.sha256()

    # Truncate + restart — fully deterministic; ordering is by id so the same
    # snapshot produces identical files.
    with txt_path.open("w", encoding="utf-8") as txt_fp, \
         idx_path.open("w", encoding="utf-8") as idx_fp:
        offset = 0
        for row in conn.execute(sql):
            d = {name: row[i] for i, name in enumerate(cols)}
            rid = str(d[ext.id_column])
            if ext.eval_id_column and rid in eval_ids:
                n_skipped_leakage += 1
                continue
            text = _row_to_text(d, ext)
            if not text:
                n_skipped_empty += 1
                continue
            # One blank-line-separated record per row, with a clear delimiter.
            block = text + "\n\n<<<END-RECORD>>>\n\n"
            payload = block.encode("utf-8")
            txt_fp.write(block)
            h.update(payload)
            entry = {"id": rid, "offset": offset, "length": len(payload),
                     "labels": {k: d.get(k) for k in ext.label_columns}}
            idx_fp.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            offset += len(payload)
            total_bytes += len(payload)
            n_total += 1

    if n_total == 0:
        # Nothing produced — drop the empty files to avoid manifest noise.
        txt_path.unlink(missing_ok=True)
        idx_path.unlink(missing_ok=True)
        return {"table": ext.table, "rows": 0, "bytes": 0,
                "skipped_leakage": n_skipped_leakage, "skipped_empty": n_skipped_empty}

    sha = h.hexdigest()
    manifest.add(ManifestEntry(
        url=f"sqlite:///{SNAPSHOT.name}#{ext.table}",
        local_path=str(txt_path.relative_to(REPO)),
        sha256=sha,
        bytes=total_bytes,
        title=f"local DB extract: {ext.table}",
        fetched_at=now_iso(),
        extra={"table": ext.table, "rows": n_total,
               "skipped_leakage": n_skipped_leakage,
               "skipped_empty": n_skipped_empty,
               "index_file": str(idx_path.relative_to(REPO))},
    ))
    return {"table": ext.table, "rows": n_total, "bytes": total_bytes,
            "skipped_leakage": n_skipped_leakage, "skipped_empty": n_skipped_empty}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract local DB English text for CPT corpus.")
    p.add_argument("--only", nargs="+", help="Limit to these table names")
    p.add_argument("--snapshot", default=str(SNAPSHOT),
                   help="Path to local SQLite snapshot")
    args = p.parse_args(argv)

    snap_path = Path(args.snapshot)
    if not snap_path.exists():
        print(f"ERROR: snapshot not found at {snap_path}. "
              f"Run `make snapshot` first.", file=sys.stderr)
        return 1

    eval_ids = _load_eval_ids()
    print(f"Eval-set leakage guard: {len(eval_ids)} question_ids loaded "
          f"({'enabled' if eval_ids else 'DISABLED — eval_set.parquet missing'})")

    selected = [e for e in EXTRACTS if (not args.only) or e.table in args.only]
    if args.only:
        unknown = set(args.only) - {e.table for e in EXTRACTS}
        if unknown:
            print(f"WARNING: unknown tables (skipped): {sorted(unknown)}",
                  file=sys.stderr)
    if not selected:
        print("No tables selected.", file=sys.stderr)
        return 1

    manifest = Manifest("local_db")
    conn = sqlite3.connect(f"file:{snap_path}?mode=ro", uri=True)
    conn.row_factory = None  # tuple rows; we map to dict via column names

    print(f"\nlocal DB extraction — {len(selected)} tables")
    totals = {"rows": 0, "bytes": 0, "skipped_leakage": 0, "skipped_empty": 0}
    for ext in selected:
        print(f"  [{ext.table}] extracting columns: {ext.text_columns}")
        result = extract_table(conn, ext, eval_ids, manifest)
        for k in totals:
            totals[k] += result.get(k, 0)
        print(f"     ↳ rows={result['rows']}  bytes={result['bytes']:,}  "
              f"skip_leak={result['skipped_leakage']}  skip_empty={result['skipped_empty']}")

    print(f"\nTotal: {totals}")
    print(f"Manifest: {manifest.summary()}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
