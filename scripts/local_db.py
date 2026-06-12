"""Local SQLite snapshot of the prod-DB subset we need.

This module is the ONLY entry point downstream scripts use for data access.
The remote prod DBs are touched exclusively by snapshot_to_local.py, and only
via read-only SELECTs.

Schema convention: JSONB columns and Postgres array columns are stored as TEXT
(JSON-encoded). A sidecar `_snapshot_meta` table records which columns to
decode on read.
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path

import pandas as pd

LOCAL_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db_snapshot.sqlite"
META_TABLE = "_snapshot_meta"


def connect() -> sqlite3.Connection:
    LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LOCAL_DB_PATH)
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {META_TABLE} "
        f"(table_name TEXT PRIMARY KEY, json_columns TEXT NOT NULL)"
    )
    return conn


def write_table(name: str, df: pd.DataFrame, json_columns: list[str]) -> None:
    df = df.copy()
    for col in json_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x, default=str, ensure_ascii=False) if x is not None else None
            )
    conn = connect()
    df.to_sql(name, conn, if_exists="replace", index=False)
    conn.execute(
        f"INSERT OR REPLACE INTO {META_TABLE} (table_name, json_columns) VALUES (?, ?)",
        (name, json.dumps(json_columns)),
    )
    conn.commit()
    conn.close()


def read_table(name: str) -> pd.DataFrame:
    if not LOCAL_DB_PATH.exists():
        raise FileNotFoundError(
            f"Local snapshot not found at {LOCAL_DB_PATH}. Run `make snapshot` first."
        )
    conn = connect()
    cur = conn.execute(f"SELECT json_columns FROM {META_TABLE} WHERE table_name = ?", (name,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise RuntimeError(
            f"Table '{name}' not in snapshot. Add it to SNAPSHOTS in snapshot_to_local.py and re-run `make snapshot`."
        )
    df = pd.read_sql_query(f"SELECT * FROM {name}", conn)
    conn.close()
    for col in json.loads(row[0]):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x else None
            )
    return df
