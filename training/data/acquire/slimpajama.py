"""Sample English text from FineWeb-Edu for the CPT replay buffer.

The CPT phase mixes ~20 % general-distribution English (this script's
output) with ~80 % UPSC-domain text to mitigate catastrophic forgetting
on general English capability — see `v2-methodology.md §4.5` and
[Continual Learning of LLMs Survey, ACM CSUR 2025](https://dl.acm.org/doi/10.1145/3735633).

Source: **HuggingFaceFW/fineweb-edu** (Penedo et al. 2024, FineWeb-Edu
educational-content subset of FineWeb, arXiv 2406.17557). Selected
over the older SlimPajama-627B (which was de-listed from HF Hub):
- FineWeb-Edu is the cleaned, deduplicated, quality-filtered web corpus
  that backs Llama 3 / Gemma 3 educational tier
- ~1.3 T tokens total across the full dataset; we stream a sample
- "sample-10BT" config = 10 B-token shuffled sample, ideal for replay

Module name kept as `slimpajama` for backwards-compat with the
methodology doc; the underlying dataset is now FineWeb-Edu.

Target: ~0.7 B tokens (≈ 3-4 GB on disk). Streaming avoids downloading
the full 1.3 T-token dump.

CLI:
    python -m training.data.acquire.slimpajama --target-tokens 700_000_000
    python -m training.data.acquire.slimpajama --target-tokens 1_000_000   # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import sys

from ._base import Manifest, ManifestEntry, RepoPaths, now_iso

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"           # 10 B-token shuffled subset
DATASET_SPLIT = "train"
TOKENS_PER_WORD = 1.3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stream-sample FineWeb-Edu for the CPT replay buffer.")
    p.add_argument("--target-tokens", type=int, default=700_000_000,
                   help="Approx token budget (default 700 M).")
    p.add_argument("--max-doc-tokens", type=int, default=4096,
                   help="Drop docs longer than this token estimate (saves disk on outliers).")
    p.add_argument("--min-doc-tokens", type=int, default=64,
                   help="Drop docs shorter than this (low-value).")
    p.add_argument("--seed", type=int, default=20260514,
                   help="Shuffle seed for the streamed dataset.")
    p.add_argument("--shuffle-buffer", type=int, default=10_000,
                   help="HF streaming shuffle-buffer size.")
    args = p.parse_args(argv)

    from datasets import load_dataset

    out_dir = RepoPaths.cpt_raw("slimpajama")
    out_path = out_dir / "sample.jsonl"
    manifest = Manifest("slimpajama")

    print(f"Streaming {DATASET_NAME} ({DATASET_SPLIT}) — target {args.target_tokens:,} tokens")
    print(f"Doc filter: [{args.min_doc_tokens}, {args.max_doc_tokens}] tokens estimated")

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    tokens_so_far = 0
    docs_written = 0
    docs_dropped_short = 0
    docs_dropped_long = 0
    bytes_written = 0
    h = hashlib.sha256()

    import json

    with out_path.open("w", encoding="utf-8") as f:
        for example in ds:
            text = (example.get("text") or "").strip()
            if not text:
                continue
            est_tokens = int(text.count(" ") * TOKENS_PER_WORD) + 1
            if est_tokens < args.min_doc_tokens:
                docs_dropped_short += 1
                continue
            if est_tokens > args.max_doc_tokens:
                docs_dropped_long += 1
                continue
            line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
            payload = line.encode("utf-8")
            f.write(line)
            h.update(payload)
            bytes_written += len(payload)
            tokens_so_far += est_tokens
            docs_written += 1
            if docs_written % 10_000 == 0:
                print(f"  ... {docs_written:,} docs, "
                      f"{tokens_so_far / 1e6:.1f} M tokens written")
            if tokens_so_far >= args.target_tokens:
                break

    manifest.add(ManifestEntry(
        url=f"hf://{DATASET_NAME}#streaming-sample",
        local_path=str(out_path.relative_to(RepoPaths.root())),
        sha256=h.hexdigest(),
        bytes=bytes_written,
        title="SlimPajama-627B streaming replay sample (English)",
        fetched_at=now_iso(),
        extra={
            "dataset": DATASET_NAME,
            "split": DATASET_SPLIT,
            "shuffle_seed": args.seed,
            "estimated_tokens": tokens_so_far,
            "docs_written": docs_written,
            "docs_dropped_short": docs_dropped_short,
            "docs_dropped_long": docs_dropped_long,
            "tokens_per_word_estimate": TOKENS_PER_WORD,
        },
    ))

    print(f"\nDone. {docs_written:,} docs, ~{tokens_so_far / 1e9:.3f} B tokens, "
          f"{bytes_written / 1e9:.2f} GB on disk")
    print(f"Dropped: {docs_dropped_short:,} short / {docs_dropped_long:,} long")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
