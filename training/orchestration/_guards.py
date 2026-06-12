"""Shared run-lifecycle guards for the CPT/SFT entrypoints.

Two concerns both entrypoints need identically:

1. **HARD_STOP marker lifecycle.** A pulse hard-stop writes
   `<output_dir>/HARD_STOP` so the orchestrator exits non-zero instead
   of looking like a clean run. But the marker must be cleared at the
   START of a fresh launch — otherwise a stale marker from a previous
   stopped run makes a clean re-run burn its whole budget and then exit
   3, discarding the adapter. We snapshot whether a marker existed
   before training and treat only a marker written DURING this process
   as authoritative.

2. **Schedule-shape resume guard.** Custom LR schedules (WSD for CPT,
   cosine-with-min-lr for SFT) are rebuilt from config at resume; the
   checkpoint only restores the step counter. Resuming with a changed
   schedule config silently reshapes the curve. We persist the resolved
   schedule config on first launch and hard-fail on mismatch at resume.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HARD_STOP = "HARD_STOP"


def clear_stale_hard_stop(output_dir: Path) -> float:
    """Remove any pre-existing HARD_STOP marker before training and
    return the launch timestamp. Only a marker with mtime >= this
    timestamp counts as a fresh stop from the current run."""
    marker = output_dir / HARD_STOP
    if marker.exists():
        print(f"[guard] clearing stale {marker.name} from a previous run")
        marker.unlink()
    return time.time()


def hard_stopped_this_run(output_dir: Path, launch_ts: float) -> str | None:
    """Return the hard-stop reason if a marker was written during this
    run (mtime >= launch), else None."""
    marker = output_dir / HARD_STOP
    if marker.exists() and marker.stat().st_mtime >= launch_ts:
        return marker.read_text(encoding="utf-8").strip()
    return None


def schedule_resume_guard(output_dir: Path, resolved: dict, kind: str) -> bool:
    """Persist `resolved` schedule config on first launch; on resume,
    hard-fail if it differs. Returns False on mismatch (caller should
    exit non-zero), True otherwise."""
    marker = output_dir / f"{kind}_schedule_config.json"
    if marker.exists():
        prev = json.loads(marker.read_text(encoding="utf-8"))
        if prev != resolved:
            print(f"\n{kind.upper()} SCHEDULE MISMATCH on resume:\n"
                  f"  run started with {prev}\n  now resolved to  {resolved}\n"
                  f"Resuming would silently reshape the LR schedule. Restore "
                  f"the original values or start a fresh output_dir.",
                  file=sys.stderr)
            return False
    else:
        marker.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    return True
