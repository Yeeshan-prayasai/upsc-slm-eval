"""Stage 3 — fine-tune a base SLM with LoRA on the UPSC multi-task corpus.

Reads data/ft_corpus.parquet, materializes a deterministic 95/5 train/valid
JSONL split (stratified by task) under data/ft_split/, then invokes
`python -m mlx_lm.lora` via subprocess. Same recipe for both base models;
only --base and --adapter-out differ.
"""
from __future__ import annotations
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "ft_split"
SEED = 20260514
VALID_FRACTION = 0.05


def _row_to_chat(r: dict) -> dict:
    return {"messages": [
        {"role": "user", "content": f"{r['instruction']}\n\n{r['input']}"},
        {"role": "assistant", "content": r["output"]},
    ]}


def _stratified_split(df: pd.DataFrame, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    train: list[dict] = []
    valid: list[dict] = []
    for _, g in df.groupby("task"):
        rows = g.to_dict("records")
        rng.shuffle(rows)
        cut = max(1, int(len(rows) * VALID_FRACTION))
        valid.extend(rows[:cut])
        train.extend(rows[cut:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def materialize_jsonl(corpus_path: Path) -> tuple[Path, int, int]:
    df = pd.read_parquet(corpus_path)
    train, valid = _stratified_split(df, SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train.jsonl"
    valid_path = DATA_DIR / "valid.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(_row_to_chat(r), ensure_ascii=False) + "\n")
    with valid_path.open("w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(_row_to_chat(r), ensure_ascii=False) + "\n")
    return DATA_DIR, len(train), len(valid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="HF id of an MLX base model (required unless --materialize-only)")
    ap.add_argument("--adapter-out", type=Path,
                    help="(required unless --materialize-only)")
    ap.add_argument("--corpus", type=Path, default=REPO / "data/ft_corpus.parquet")
    ap.add_argument("--config", type=Path, default=REPO / "configs/lora.yaml")
    ap.add_argument("--materialize-only", action="store_true",
                    help="just produce data/ft_split/{train,valid}.jsonl and exit "
                         "(used to prep data for the AWS path without invoking MLX)")
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"[FAIL] {args.corpus} not found; run `make build-ft-corpus` first")
        return 1

    print(f"Materializing JSONL split (seed={SEED}, valid_frac={VALID_FRACTION}) …")
    data_dir, n_train, n_valid = materialize_jsonl(args.corpus)
    print(f"  train: {n_train:,} → {data_dir / 'train.jsonl'}")
    print(f"  valid: {n_valid:,} → {data_dir / 'valid.jsonl'}")

    if args.materialize_only:
        print(f"\n[OK] materialize-only complete; skipping MLX LoRA training")
        return 0

    if not args.base or not args.adapter_out:
        print("[FAIL] --base and --adapter-out are required unless --materialize-only is set")
        return 1
    if not args.config.exists():
        print(f"[FAIL] {args.config} not found")
        return 1

    # MLX memory safety. The default Metal cap on a 16 GB box is ~12 GB and on
    # a 24 GB box is ~18 GB. The training peak for a 4B LoRA at seq=2048 batch=1
    # exceeds the 16 GB default by ~4-6 GB at the val→train transition (open
    # issues mlx-lm#828 + #1185 — val cache held when first train step runs).
    # Pre-flight: cap the in-process wired buffer + cache, force a clear before
    # invoking mlx_lm.lora. Requires `sudo sysctl iogpu.wired_limit_mb=21504`
    # already applied at the OS level for the 24 GB recipe to fit.
    import mlx.core as mx
    mx.set_wired_limit(20 * 1024 ** 3)     # 20 GiB in-process cap
    mx.set_cache_limit(512 * 1024 ** 2)    # 512 MiB activation cache cap
    mx.clear_cache()
    print(f"[mlx] wired_limit=20 GiB, cache_limit=512 MiB, cache cleared "
          f"(current active={mx.get_active_memory() / 1024**3:.2f} GiB)")

    args.adapter_out.mkdir(parents=True, exist_ok=True)
    log_path = args.adapter_out / "training.log"

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", args.base,
        "--train",
        "-c", str(args.config),
        "--data", str(data_dir),
        "--adapter-path", str(args.adapter_out),
    ]
    print(f"\n$ {' '.join(cmd)}\n")

    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
            logf.flush()
        proc.wait()

    if proc.returncode != 0:
        print(f"\n[FAIL] mlx_lm.lora exited {proc.returncode}")
        return proc.returncode

    print(f"\n[OK] adapter → {args.adapter_out}")
    print(f"     log     → {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
