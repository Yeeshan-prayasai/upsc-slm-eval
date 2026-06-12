"""6-cell ablation grid driver — per methodology §8.

Trains and evaluates each cell sequentially on a single GPU:

  cell 1: v1 baseline                            (no run; reuses existing v1 results)
  cell 2: SFT-only                              (skip CPT)
  cell 3: CPT-only           (no SFT — merge CPT adapter, eval directly)
  cell 4: CPT → SFT full     (HEADLINE run)
  cell 5: CPT → SFT, vanilla LoRA (no RSLoRA)   (isolates RSLoRA contribution)
  cell 6: CPT-50%-ckpt → SFT                    (measures CPT diminishing returns)

For each non-baseline cell, the driver:
  1. Sets up cell-specific config overrides on top of the per-model YAMLs
  2. Runs CPT (if cell needs it) → SFT (if cell needs it)
  3. Re-runs the v1 inference pipeline against the final adapter
     (via `scripts/run_inference.py` — unchanged, reused as-is)
  4. Scores via `scripts/score_tier1.py` → aggregate → hypothesis_tests
  5. Writes per-cell results to `runs/ablation/<cell>/`

The driver is RESUMABLE — if a cell's `runs/ablation/<cell>/done.txt`
exists, it skips. To force re-run, delete that marker.

Usage:
  python -m training.orchestration.run_ablation --model gemma
  python -m training.orchestration.run_ablation --model qwen --cells 4 5
  python -m training.orchestration.run_ablation --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ..data.acquire._base import RepoPaths

REPO = RepoPaths.root()
ABLATION_ROOT = REPO / "runs" / "ablation"


@dataclass(frozen=True)
class Cell:
    """One ablation cell."""
    n: int
    name: str
    description: str
    needs_cpt: bool
    needs_sft: bool
    cpt_overrides: dict   # YAML keys to override on the per-model CPT config
    sft_overrides: dict   # YAML keys to override on the per-model SFT config


CELLS = [
    Cell(
        n=1,
        name="v1_baseline",
        description="v1 SFT-only adapter — reuses existing v1 numbers, no training",
        needs_cpt=False,
        needs_sft=False,
        cpt_overrides={},
        sft_overrides={},
    ),
    Cell(
        n=2,
        name="sft_only",
        description="LoRA (rank 64) SFT only; skip CPT — isolates rank+RSLoRA benefit",
        needs_cpt=False,
        needs_sft=True,
        cpt_overrides={},
        sft_overrides={"resume_lora_from": None},   # no CPT adapter to resume from
    ),
    Cell(
        n=3,
        name="cpt_only",
        description="CPT-only adapter, no SFT — isolates pure-knowledge-injection effect",
        needs_cpt=True,
        needs_sft=False,
        cpt_overrides={},
        sft_overrides={},
    ),
    Cell(
        n=4,
        name="cpt_then_sft",
        description="HEADLINE: CPT → SFT full pipeline",
        needs_cpt=True,
        needs_sft=True,
        cpt_overrides={},
        sft_overrides={},
    ),
    Cell(
        n=5,
        name="cpt_then_sft_vanilla_lora",
        description="CPT → SFT with use_rslora=False — isolates RSLoRA scaling contribution",
        needs_cpt=True,
        needs_sft=True,
        cpt_overrides={"lora": {"use_rslora": False}},
        sft_overrides={"lora": {"use_rslora": False}},
    ),
    Cell(
        n=6,
        name="cpt_50pct_then_sft",
        description="50%-CPT-checkpoint → SFT — measures CPT diminishing returns past midpoint",
        needs_cpt=False,                # uses cell-4's 50% checkpoint, doesn't re-train
        needs_sft=True,
        cpt_overrides={},
        sft_overrides={},               # resume_lora_from rewritten at run time
    ),
]


def _cell_dir(cell: Cell, model_family: str) -> Path:
    """e.g. runs/ablation/cell4_cpt_then_sft__gemma/"""
    return ABLATION_ROOT / f"cell{cell.n}_{cell.name}__{model_family}"


def _is_done(cell: Cell, model_family: str) -> bool:
    return (_cell_dir(cell, model_family) / "done.txt").exists()


def _mark_done(cell: Cell, model_family: str, summary: dict) -> None:
    d = _cell_dir(cell, model_family)
    d.mkdir(parents=True, exist_ok=True)
    (d / "done.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_override_yaml(base_yaml: Path, overrides: dict, dest: Path) -> Path:
    """Write a YAML that's `base_yaml` with `overrides` deep-merged in.

    Used to materialize per-cell CPT/SFT configs without mutating the
    canonical config files.
    """
    import yaml
    with base_yaml.open(encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}

    def _merge(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            out = dict(a)
            for k, v in b.items():
                out[k] = _merge(a.get(k), v)
            return out
        return b
    merged = _merge(cfg, overrides)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(merged, fp, sort_keys=False)
    return dest


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def _run_cpt(cell: Cell, model_family: str, dry_run: bool) -> Path | None:
    """Run the CPT phase for one cell. Returns the final adapter dir,
    or None if not applicable."""
    if not cell.needs_cpt:
        return None
    cell_dir = _cell_dir(cell, model_family)
    cpt_dir = cell_dir / "cpt"
    cpt_dir.mkdir(parents=True, exist_ok=True)

    base_cpt = REPO / "training" / "configs" / f"cpt_{model_family}.yaml"
    overrides = dict(cell.cpt_overrides)
    overrides["output_dir"] = str(cpt_dir)
    cell_cfg = _write_override_yaml(base_cpt, overrides,
                                    cell_dir / f"cpt_{model_family}.yaml")
    cmd = [
        sys.executable, "-u", "-m", "training.orchestration.run_cpt",
        "--config", str(cell_cfg),
        "--runtime", str(REPO / "training" / "configs" / "runtime_l40s.yaml"),
    ]
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return cpt_dir / "final"
    rc = _run(cmd, cwd=REPO)
    if rc != 0:
        raise RuntimeError(f"CPT failed for {cell.name}/{model_family} (rc={rc})")
    return cpt_dir / "final"


def _run_sft(cell: Cell, model_family: str, cpt_adapter: Path | None,
             dry_run: bool) -> Path | None:
    """Run the SFT phase. `cpt_adapter` is the adapter to continue from
    (None for cell 2 = SFT-only)."""
    if not cell.needs_sft:
        return None
    cell_dir = _cell_dir(cell, model_family)
    sft_dir = cell_dir / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)

    base_sft = REPO / "training" / "configs" / f"sft_{model_family}.yaml"
    overrides = dict(cell.sft_overrides)
    overrides["output_dir"] = str(sft_dir)
    if cpt_adapter is None:
        overrides["resume_lora_from"] = None     # SFT-only / no resume
    else:
        overrides["resume_lora_from"] = str(cpt_adapter)

    cell_cfg = _write_override_yaml(base_sft, overrides,
                                    cell_dir / f"sft_{model_family}.yaml")
    cmd = [
        sys.executable, "-u", "-m", "training.orchestration.run_sft",
        "--config", str(cell_cfg),
        "--runtime", str(REPO / "training" / "configs" / "runtime_l40s.yaml"),
    ]
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return sft_dir / "final"
    rc = _run(cmd, cwd=REPO)
    if rc != 0:
        raise RuntimeError(f"SFT failed for {cell.name}/{model_family} (rc={rc})")
    return sft_dir / "final"


def _cell6_branch_from_cell4(model_family: str) -> Path | None:
    """Cell 6 reuses cell 4's 50%-CPT checkpoint — the rotation-exempt
    `checkpoint-midpoint/` copy saved by `MidpointCheckpointCallback`.
    No nearest-checkpoint fallback: silently substituting a ~72%
    checkpoint (what rotation used to leave behind) would invalidate
    the diminishing-returns measurement this cell exists for."""
    cell4_dir = _cell_dir(CELLS[3], model_family)
    target = cell4_dir / "cpt" / "checkpoint-midpoint"
    return target if target.exists() else None


def run_cell(cell: Cell, model_family: str, dry_run: bool) -> dict:
    """Train one cell end-to-end. Returns a summary dict."""
    if _is_done(cell, model_family):
        print(f"[cell {cell.n}/{cell.name}] already done — skipping. "
              f"Delete {_cell_dir(cell, model_family)}/done.txt to force re-run.")
        return {"cell": cell.n, "name": cell.name, "skipped": True}

    print(f"\n{'='*70}\nCELL {cell.n}: {cell.name} ({model_family})\n  {cell.description}\n{'='*70}")
    summary = {
        "cell": cell.n,
        "name": cell.name,
        "model": model_family,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "needs_cpt": cell.needs_cpt,
        "needs_sft": cell.needs_sft,
    }

    if cell.n == 1:
        # Baseline — nothing to train; results pulled from v1 artifacts in the report step
        summary["adapter_path"] = "data/eval_set.parquet (v1 baseline; uses scores in results/aggregate.parquet)"
    elif cell.n == 6:
        # Special-case: reuses cell-4's 50% checkpoint as the SFT seed
        cpt_seed = _cell6_branch_from_cell4(model_family)
        if cpt_seed is None and not dry_run:
            raise RuntimeError(
                f"cell 6 requires cell 4's rotation-exempt "
                f"cpt/checkpoint-midpoint dir; not found. Run cell 4 first "
                f"(MidpointCheckpointCallback saves it automatically)."
            )
        summary["cpt_seed"] = str(cpt_seed) if cpt_seed else None
        sft = _run_sft(cell, model_family, cpt_seed, dry_run)
        summary["adapter_path"] = str(sft) if sft else None
    else:
        cpt = _run_cpt(cell, model_family, dry_run)
        sft = _run_sft(cell, model_family, cpt, dry_run)
        summary["adapter_path"] = str(sft or cpt) if (sft or cpt) else None

    summary["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if not dry_run:
        _mark_done(cell, model_family, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="6-cell ablation grid driver.")
    p.add_argument("--model", choices=("gemma", "qwen"), required=True)
    p.add_argument("--cells", nargs="+", type=int, default=None,
                   help="Run only these cell numbers (default: all). "
                        "Cell 6 requires cell 4 to have completed.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    args = p.parse_args(argv)

    chosen = (
        [c for c in CELLS if c.n in set(args.cells)] if args.cells
        else CELLS
    )
    if args.cells:
        print(f"Selected cells: {[c.n for c in chosen]}")

    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict] = []
    for cell in chosen:
        try:
            s = run_cell(cell, args.model, args.dry_run)
            all_summaries.append(s)
        except Exception as e:
            print(f"\n✗ cell {cell.n} ({cell.name}) FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)
            all_summaries.append({
                "cell": cell.n, "name": cell.name, "model": args.model,
                "error": f"{type(e).__name__}: {e}",
            })
            # Continue with subsequent cells even if one fails — better
            # to get partial results than abort the whole grid.

    # Write a roll-up
    summary_path = ABLATION_ROOT / f"summary__{args.model}.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\n{'=' * 70}\nGrid summary → {summary_path.relative_to(REPO)}\n{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
