"""CPT corpus builder — runs OCR → clean+dedupe → leakage gate → tokenize+pack.

Each stage is also a standalone CLI under `training.data.*` for
re-running individual stages. This script just wires them in order
with fail-loud at each boundary.

CLI:
    python -m training.data.build_cpt_corpus
    python -m training.data.build_cpt_corpus --skip-ocr           # if .txt already exists
    python -m training.data.build_cpt_corpus --skip-tokenize      # stop after dedupe
    python -m training.data.build_cpt_corpus --source ncert       # one source only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .acquire._base import RepoPaths
from . import build_instruct_cpt as instruct_mod
from . import clean as clean_mod
from . import leakage as leakage_mod
from . import ocr as ocr_mod
from . import tokenize_pack as tokenize_mod

REPO = RepoPaths.root()
CPT_RAW = REPO / "data" / "cpt_raw"
CPT_TEXT = REPO / "data" / "cpt_text"
CPT_CLEAN = REPO / "data" / "cpt_clean"
CPT_CLEAN_DEDUP = REPO / "data" / "cpt_clean_dedup"


def _stage_header(name: str) -> None:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")


def stage_ocr(source: str | None, workers: int) -> int:
    _stage_header(f"Stage 1: OCR / extract  ({source or 'all sources'})")
    argv = ["--workers", str(workers)]
    if source:
        argv += ["--source", source]
    else:
        # OCR doesn't yet have an "all sources" mode — run each source dir.
        # Only dirs that actually contain PDFs: text-only sources
        # (mea/orf/dte/... markdown, slimpajama jsonl) have nothing to
        # extract, and ocr.py's "no PDFs" exit code 1 — correct for a
        # typo'd --source on the CLI — previously aborted the whole
        # build at the first markdown-only source.
        all_dirs = sorted(p.name for p in CPT_RAW.iterdir() if p.is_dir())
        if not all_dirs:
            print("No source dirs under data/cpt_raw. Run the acquirers first.", file=sys.stderr)
            return 1
        sources = [s for s in all_dirs
                   if any((CPT_RAW / s).rglob("*.pdf"))]
        skipped = [s for s in all_dirs if s not in sources]
        if skipped:
            print(f"(skipping {len(skipped)} text-only sources, no PDFs: "
                  f"{', '.join(skipped)})")
        for src in sources:
            print(f"\n--- OCR for {src} ---")
            rc = ocr_mod.main(["--source", src, "--workers", str(workers)])
            if rc != 0:
                return rc
        return 0
    return ocr_mod.main(argv)


def stage_clean(source: str | None) -> int:
    _stage_header(f"Stage 2: Clean + dedupe  ({source or 'all sources'})")
    argv = ["--in", str(CPT_TEXT), "--out", str(CPT_CLEAN)]
    if source:
        argv += ["--source", source]
    return clean_mod.main(argv)


def stage_instruct() -> int:
    _stage_header("Stage 2b: Stage instruction slice (anti-forgetting for -it chat)")
    rc = instruct_mod.build()
    if rc != 0:
        print("Instruction slice unavailable (run `make build-sft-corpus` "
              "first) — continuing WITHOUT it; the mix's `instruct` source "
              "will contribute 0 tokens.", file=sys.stderr)
    return 0   # non-fatal: CPT can run without the instruct slice


def stage_leakage(source: str | None) -> int:
    _stage_header(f"Stage 3: Leakage gate  ({source or 'all sources'})")
    # Run against the deduped output (cpt_clean_dedup), since that's
    # what tokenize_pack will consume.
    argv = ["--cpt-raw", str(CPT_CLEAN_DEDUP)]
    if source:
        argv += ["--source", source]
    return leakage_mod.main(argv)


def stage_tokenize(tokenizer: str) -> int:
    _stage_header(f"Stage 4: Tokenize + pack  (tokenizer={tokenizer})")
    return tokenize_mod.main(["--tokenizer", tokenizer,
                              "--corpus-root", str(CPT_CLEAN_DEDUP)])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build the CPT corpus end-to-end "
                    "(OCR → clean → dedupe → leakage → tokenize+pack).",
    )
    p.add_argument("--source", help="Limit to one source dir (e.g. ncert)")
    p.add_argument("--skip-ocr", action="store_true",
                   help="Skip OCR/extraction (use existing data/cpt_text/)")
    p.add_argument("--skip-clean", action="store_true",
                   help="Skip cleaning + dedupe (use existing data/cpt_clean_dedup/)")
    p.add_argument("--skip-tokenize", action="store_true",
                   help="Stop after the leakage gate")
    p.add_argument("--tokenizer", choices=("gemma", "qwen", "both"), default="both",
                   help="Which tokenizer(s) to use in stage 4")
    p.add_argument("--workers", type=int, default=1,
                   help="OCR worker count (default 1 — Tesseract OCR on "
                        "scanned PDFs spikes RAM; M5 has crashed from "
                        "workers=4 in past runs. Bump to 4+ only on EC2 / "
                        "machines with >32 GB RAM)")
    args = p.parse_args(argv)

    if not args.skip_ocr:
        rc = stage_ocr(args.source, args.workers)
        if rc != 0:
            return rc

    if not args.skip_clean:
        rc = stage_clean(args.source)
        if rc != 0:
            return rc

    if not args.source:
        stage_instruct()
        # Purge eval-overlapping records from the table-dump sources
        # (local_db question tables ARE where the eval set was drawn
        # from). The gate below re-verifies, so this can't mask a leak.
        _stage_header("Stage 2c: Purge eval-overlapping records")
        from . import purge_eval_records as purge_mod
        purge_mod.purge()

    # Leakage gate is non-optional — it is the safety check that
    # prevents training results from being invalidated by eval
    # contamination.
    rc = stage_leakage(args.source)
    if rc != 0:
        print(
            "\nLEAKAGE GATE FAILED — refusing to proceed to tokenization. "
            "Resolve the listed overlaps before re-running.",
            file=sys.stderr,
        )
        return rc

    if not args.skip_tokenize:
        rc = stage_tokenize(args.tokenizer)
        if rc != 0:
            return rc

    print("\n✓ CPT corpus build complete — ready for training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
