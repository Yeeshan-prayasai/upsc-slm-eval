"""Re-running freeze_eval_set.py with the same seed produces a byte-identical Parquet.

Requires Postgres reachability — the script connects to upscdev + prod.
"""
from __future__ import annotations
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_eval_set_freeze_is_deterministic(tmp_path):
    out1 = tmp_path / "eval_set_a.parquet"
    out2 = tmp_path / "eval_set_b.parquet"

    cmd = [sys.executable, str(REPO / "scripts" / "freeze_eval_set.py"),
           "--seed", "20260514", "--out", str(out1)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    cmd[-1] = str(out2)
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    h1 = hashlib.sha256(out1.read_bytes()).hexdigest()
    h2 = hashlib.sha256(out2.read_bytes()).hexdigest()
    assert h1 == h2, f"non-deterministic: {h1} != {h2}"
