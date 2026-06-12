"""Purge eval-overlapping records from the cleaned CPT corpus.

The local-DB extracts dump whole question tables (`mcqs`,
`upsc_prelims_ai_generated_que`, `evaluation_questions`) — the same
tables the locked eval set was drawn from, so the eval questions
themselves land verbatim in the CPT corpus. This stage walks every
`<<<END-RECORD>>>`-delimited `.txt` under `cpt_clean_dedup`, applies
the leakage gate's own checks (exact-hash + 50-token contiguous) to
each record, and rewrites the files with the flagged records removed —
keeping the thousands of non-eval questions, which are valuable
QA-format CPT data.

Runs between the clean and leakage stages of `build_cpt_corpus`; the
gate then re-verifies the result, so a purge bug cannot slip
contamination through.

CLI:
    python -m training.data.purge_eval_records
    python -m training.data.purge_eval_records --root data/cpt_clean_dedup
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .acquire._base import RepoPaths
from .clean import END_RECORD_DELIM
from .leakage import (
    EVAL_SET,
    _gram_hash,
    build_eval_index,
    question_hash,
    tokenize_loose,
)

REPO = RepoPaths.root()
CPT_CLEAN_DEDUP = REPO / "data" / "cpt_clean_dedup"
# The in-training pulse probe must be as protected as the locked eval
# set — if the probe questions are in the corpus, the mid-training
# Task-A trend measures memorization, not learning.
HOLDOUT = REPO / "data" / "eval_set_holdout.parquet"


def _record_leaks(text: str, hash_to_qid: dict, gram_to_qids: dict,
                  gram_lengths) -> bool:
    if question_hash(text) in hash_to_qid:
        return True
    toks = tokenize_loose(text)
    for n in gram_lengths:
        for i in range(len(toks) - n + 1):
            if _gram_hash(tuple(toks[i:i + n])) in gram_to_qids:
                return True
    return False


def purge(root: Path = CPT_CLEAN_DEDUP) -> int:
    """Two passes:
    1. END-RECORD .txt files (local-DB table dumps) — drop leaked records.
    2. ALL .md files — drop leaked paragraphs. Past-year UPSC questions
       are reprinted verbatim in training text (Mrunal's past-paper
       pages, Laxmikanth's PYQ appendix) and the eval set contains
       PYQ-derived items; paragraph-scoped removal keeps the rest of
       the document (dropping all of Laxmikanth over an appendix
       question would be absurd)."""
    import re
    _, hash_to_qid, gram_to_qids, gram_lengths = build_eval_index([EVAL_SET, HOLDOUT])
    print(f"[purge] index covers locked eval set + pulse holdout "
          f"({'present' if HOLDOUT.exists() else 'ABSENT — probe unprotected!'})")
    total_in = total_dropped = 0

    txts = [f for f in sorted(root.rglob("*.txt"))
            if END_RECORD_DELIM in f.read_text(encoding="utf-8", errors="replace")]
    print(f"[purge] {len(txts)} record-delimited files under {root.relative_to(REPO)}")
    for f in txts:
        text = f.read_text(encoding="utf-8", errors="replace")
        kept, dropped = [], 0
        for rec in text.split(END_RECORD_DELIM):
            if not rec.strip():
                continue
            total_in += 1
            if _record_leaks(rec, hash_to_qid, gram_to_qids, gram_lengths):
                dropped += 1
                continue
            kept.append(rec.strip())
        if dropped:
            body = ("\n\n" + END_RECORD_DELIM + "\n\n").join(kept)
            f.write_text(body + "\n\n" + END_RECORD_DELIM + "\n" if kept else "",
                         encoding="utf-8")
            print(f"  {f.relative_to(root)}: dropped {dropped}/{dropped + len(kept)} records")
            total_dropped += dropped

    mds = sorted(root.rglob("*.md"))
    print(f"[purge] scanning {len(mds)} .md files at paragraph level")
    for f in mds:
        text = f.read_text(encoding="utf-8", errors="replace")
        paras = re.split(r"\n\n+", text)
        kept, dropped = [], 0
        for para in paras:
            if not para.strip():
                continue
            total_in += 1
            if _record_leaks(para, hash_to_qid, gram_to_qids, gram_lengths):
                dropped += 1
                continue
            kept.append(para)
        if dropped:
            f.write_text("\n\n".join(kept) + "\n", encoding="utf-8")
            print(f"  {f.relative_to(root)}: dropped {dropped} paragraphs")
            total_dropped += dropped

    print(f"[purge] {total_dropped} records/paragraphs purged of {total_in:,} scanned")
    return 0


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(description="Purge eval-overlapping records from the corpus.")
    p.add_argument("--root", type=Path, default=CPT_CLEAN_DEDUP)
    args = p.parse_args(argv)
    return purge(args.root)


if __name__ == "__main__":
    sys.exit(main())
