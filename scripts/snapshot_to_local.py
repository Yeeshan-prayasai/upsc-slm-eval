"""Snapshot the subset of prod-DB tables we need into local SQLite.

THIS IS THE ONLY SCRIPT IN THIS REPO THAT CONNECTS TO REMOTE POSTGRES.

Safety:
  - Every connection uses `conn.set_session(readonly=True, isolation_level="REPEATABLE READ")`.
  - The server will reject any non-SELECT statement issued under that session.
  - This script contains no INSERT/UPDATE/DELETE/DDL of any kind against remote.

Output:
  - data/prayas_local.sqlite   — single SQLite file containing all snapshotted tables
  - data/prayas_local.sha256   — SHA-256 of the SQLite file for reproducibility
"""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path

import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_creds import dsn
from local_db import LOCAL_DB_PATH, write_table

# (target_db, table_name, projection_sql, [json_columns to encode as TEXT])
# Each entry: (target_db, local_table_label, full_sql, json_columns).
# The full SQL is self-contained — includes FROM/JOIN/ORDER BY — so JOINs are
# expressible (e.g. mcqs ⨝ learning_items to attach the paper tag).
SNAPSHOTS: list[tuple[str, str, str, list[str]]] = [
    ("upscdev", "prelims_pyq_questions",
     "SELECT question_id, year, paper, subject, correct_option, question, options, "
     "explanation, silly_mistake_prone, question_pattern, content_type, is_dropped "
     "FROM public.prelims_pyq_questions ORDER BY question_id",
     ["question", "options", "explanation"]),
    ("upscdev", "upsc_prelims_ai_generated_que",
     "SELECT id::text AS id, subject, topic, answer, difficulty, "
     "question_english, options_english, question_hindi, options_hindi, "
     "explanation, prone_to_silly_mistakes, quality_pass_flag, month, year "
     "FROM public.upsc_prelims_ai_generated_que ORDER BY id",
     ["options_english", "options_hindi"]),
    ("upscdev", "pyqs",
     "SELECT question_id::text AS question_id, paper, subject, year, max_score, "
     "word_count, section, question, model_answer, hints "
     "FROM public.pyqs ORDER BY question_id",
     ["question", "model_answer", "hints"]),
    ("upscdev", "evaluation_questions",
     "SELECT question_id::text AS question_id, evaluation_id::text AS evaluation_id, "
     "answer_text, score, max_score, word_count, strengths, improvements, "
     "question_text, subject, model_answer "
     "FROM public.evaluation_questions ORDER BY question_id, evaluation_id",
     ["strengths", "improvements", "model_answer"]),
    ("prod", "mcqs",
     'SELECT m.id::text AS id, m."questionText", m.options, m."correctOptionIds", '
     'm."isMultiSelect", m.explanation, m.silly_mistake_prone, m.question_pattern, '
     'li.paper, li.tags AS learning_item_tags '
     'FROM public.mcqs m '
     'LEFT JOIN public.learning_items li ON li.id = m."learningItemId" '
     'ORDER BY m.id',
     ["options", "correctOptionIds", "explanation", "learning_item_tags"]),
    ("prod", "news_articles",
     'SELECT id::text AS id, date, title, text, "prelimsInfo", "mainsInfo", '
     '"newsThemeId"::text AS "newsThemeId", source '
     'FROM public.news_articles ORDER BY id',
     []),
    ("prod", "current_affairs",
     "SELECT id, date, description, heading, source, published_at, pointed_analysis, "
     "mains_analysis, prelims_info, mains_subject, prelims_subject, "
     "mains_topics, prelims_topics, theme_id::text AS theme_id "
     "FROM public.current_affairs ORDER BY id",
     ["mains_topics", "prelims_topics"]),
    ("prod", "glossary",
     "SELECT id::text AS id, keyword, definition FROM public.glossary ORDER BY id",
     []),
]


def main() -> int:
    print(f"Snapshot destination: {LOCAL_DB_PATH}")
    print(f"Tables to snapshot:   {len(SNAPSHOTS)}")
    print(f"Mode: read-only SELECTs only. No writes to any prod DB.\n")

    if LOCAL_DB_PATH.exists():
        LOCAL_DB_PATH.unlink()

    total = 0
    for target, table, sql, json_cols in SNAPSHOTS:
        print(f"  [{target:>7s}] {table:<35s}", end=" ", flush=True)
        with psycopg2.connect(**dsn(target), connect_timeout=20) as conn:
            conn.set_session(readonly=True, isolation_level="REPEATABLE READ")
            df = pd.read_sql_query(sql, conn)
        write_table(table, df, json_cols)
        print(f"{len(df):>7,} rows")
        total += len(df)

    sha = hashlib.sha256(LOCAL_DB_PATH.read_bytes()).hexdigest()
    LOCAL_DB_PATH.with_suffix(".sha256").write_text(sha + "\n")

    size_mb = LOCAL_DB_PATH.stat().st_size / 1024 / 1024
    print(f"\n[OK] snapshot complete")
    print(f"     total rows: {total:,}")
    print(f"     file size:  {size_mb:.1f} MB")
    print(f"     SHA-256:    {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
