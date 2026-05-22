"""Precondition gate. Run before any other script. Exits non-zero on failure.

Offline checks only — no network. Remote-DB reachability is verified by
snapshot_to_local.py at the moment it pulls data.
"""
from __future__ import annotations
import os
import sys
import importlib
from pathlib import Path

REQUIRED = [
    "psycopg2", "pandas", "pyarrow", "numpy", "sklearn", "scipy", "statsmodels",
    "jsonschema", "textstat", "tenacity", "pytest",
]
MIN_PY = (3, 12)


def main() -> int:
    fails: list[str] = []

    if sys.version_info < MIN_PY:
        fails.append(f"Python {MIN_PY[0]}.{MIN_PY[1]}+ required; have {sys.version.split()[0]}")

    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except ImportError:
            fails.append(f"missing package: {name}")

    creds_path = Path(__file__).resolve().parent.parent / "db-creds.txt"
    if not creds_path.exists():
        fails.append(f"db-creds.txt not found at {creds_path}")
    else:
        from db_creds import dsn
        for target in ("upscdev", "prod"):
            try:
                dsn(target)
            except KeyError as e:
                fails.append(f"db-creds.txt missing keys for {target}: {e}")

    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        fails.append("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        fails.append("ANTHROPIC_API_KEY not set in env")

    if fails:
        print("[FAIL] verify_env:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("[OK] env verified")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
