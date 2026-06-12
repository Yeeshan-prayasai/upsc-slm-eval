"""CI guard: no org/personal identifiers may appear in shipped code.

Hard project rule — `prayas`/`prayas.ai`/`irshad` must never be embedded
in User-Agents, docstrings, comments, configs, or any code path. This
test fails loudly if the literal regresses anywhere under training/ or
scripts/.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FORBIDDEN = re.compile(r"prayas|irshad", re.IGNORECASE)
SCAN_DIRS = ("training", "scripts")
SCAN_EXTS = (".py", ".yaml", ".yml")


def _scan_files():
    for d in SCAN_DIRS:
        for ext in SCAN_EXTS:
            yield from (REPO / d).rglob(f"*{ext}")


def test_no_org_identifiers_in_code():
    hits = []
    for f in _scan_files():
        if "__pycache__" in str(f) or f.name == "test_no_identifiers.py":
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if FORBIDDEN.search(line):
                hits.append(f"{f.relative_to(REPO)}:{i}: {line.strip()[:80]}")
    assert not hits, "org identifier leaked into code:\n" + "\n".join(hits)
