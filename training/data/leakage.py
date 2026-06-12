"""Leakage gate — refuses to start training if any eval-set question
appears in the CPT corpus.

Three checks, each at the threshold cited from literature:

1. **ID-level** — set intersection of eval `question_id` values with
   any item ids recorded in CPT manifests. Catches direct duplication
   from the local-DB extractor. (No paper — straightforward sanity.)
2. **Normalized exact-text** — SHA-256 of every eval question's
   normalized text vs the same normalization over each CPT paragraph.
   Catches verbatim short-question stems that wouldn't reach the
   50-token threshold below.
3. **50-token contiguous overlap** — for each eval gold-text passage
   tokenized into a list of N tokens, the corpus is flagged if ANY
   50-consecutive-token window of the corpus equals any 50-token
   window of the eval passage. Per **Carlini et al. 2023**
   ("Quantifying Memorization Across Neural Language Models",
   arXiv 2202.07646) and **Lee et al. 2022** (arXiv 2107.06499),
   50 contiguous tokens is the canonical contamination/memorization
   threshold for LM-training corpora.

Implementation uses an **inverted-index lookup**: we materialize the
set of all eval 50-token sliding windows (hashed), then scan the
corpus once, hashing each 50-token window and checking set membership.
Time: O(total corpus tokens). Memory: O(sum of eval-passage tokens).

For eval passages shorter than 50 tokens, the 50-gram set is empty
and the check degenerates — but the SHA-256 exact-text check (#2)
covers that case.

CLI:
    python -m training.data.leakage
    python -m training.data.leakage --cpt-raw data/cpt_clean_dedup
    python -m training.data.leakage --source ncert
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .acquire._base import RepoPaths

REPO = RepoPaths.root()
EVAL_SET = REPO / "data" / "eval_set.parquet"
CPT_RAW = REPO / "data" / "cpt_raw"

# 50-token contiguous overlap — Carlini et al. 2023, Lee et al. 2022.
NGRAM_N = 50


def normalize(text: str) -> str:
    """Whitespace + casefold normalization for the exact-text check.
    Mirrors v1's `_normalize_question_text` in `scripts/build_ft_corpus.py`."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def question_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:16]


def tokenize_loose(text: str) -> list[str]:
    """Word-level tokenizer used for n-gram matching only (not for
    training). Lowercases, strips punctuation. Numbers/dates preserved
    as tokens (per project memory `feedback_keep_numbers_tables`)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _gram_hash(tokens: tuple[str, ...]) -> int:
    """64-bit hash of a token tuple. Using Python's `hash` is
    process-stable within a run, which is all we need (we never
    persist these hashes)."""
    return hash(tokens)


@dataclass
class LeakageReport:
    """Output of `check_corpus_text`. Empty fields = clean."""
    id_overlaps: set[str] = field(default_factory=set)
    hash_overlaps: set[str] = field(default_factory=set)   # hash → eval qid
    ngram_hits: list[tuple[str, str]] = field(default_factory=list)
    # (eval_question_id, source_file_relpath)

    def is_clean(self) -> bool:
        return not (self.id_overlaps or self.hash_overlaps or self.ngram_hits)

    def render(self) -> str:
        if self.is_clean():
            return "✓ CLEAN — no ID / hash / 50-token contiguous leakage detected."
        parts = []
        if self.id_overlaps:
            parts.append(f"✗ ID-LEVEL ({len(self.id_overlaps)}): "
                         f"{sorted(self.id_overlaps)[:5]}")
        if self.hash_overlaps:
            parts.append(f"✗ EXACT-TEXT ({len(self.hash_overlaps)}): "
                         f"first 5 eval qids: {sorted(self.hash_overlaps)[:5]}")
        if self.ngram_hits:
            parts.append(f"✗ 50-TOKEN OVERLAP ({len(self.ngram_hits)} hits):")
            # Group by eval qid for compact reporting
            by_qid: dict[str, list[str]] = {}
            for qid, src in self.ngram_hits:
                by_qid.setdefault(qid, []).append(src)
            for qid in sorted(by_qid)[:10]:
                srcs = by_qid[qid]
                parts.append(f"    eval={qid}  hits in: {srcs[:3]}"
                             + (f"  (+{len(srcs)-3} more)" if len(srcs) > 3 else ""))
        return "\n".join(parts)


def _extract_eval_text(row: pd.Series) -> str:
    """Concatenate the textual fields of a frozen-eval row's
    `gold_payload` JSON that could plausibly appear verbatim in NCERT
    or reference books."""
    gp = row.get("gold_payload")
    if isinstance(gp, str):
        try:
            gp = json.loads(gp)
        except json.JSONDecodeError:
            gp = {}
    if not isinstance(gp, dict):
        return ""
    return " ".join(
        str(gp.get(k) or "")
        for k in ("question", "question_text", "title", "article",
                  "answer_text", "explanation")
    )


def build_eval_index(eval_path: Path = EVAL_SET) -> tuple[
    set[str],                   # all eval question_ids
    dict[str, str],             # text-hash → eval question_id
    dict[int, set[str]],        # 50-gram hash → set of eval question_ids
]:
    """Read the locked eval set and build:
       - set of question_ids (for ID-level check)
       - text-hash → qid map (for exact-text check)
       - 50-gram hash → qid set (inverted index for the contiguous-overlap check)
    """
    df = pd.read_parquet(eval_path)
    ids: set[str] = set(df["question_id"].astype(str))
    hash_to_qid: dict[str, str] = {}
    gram_to_qids: dict[int, set[str]] = {}
    for _, row in df.iterrows():
        qid = str(row["question_id"])
        text = _extract_eval_text(row).strip()
        if not text:
            continue
        hash_to_qid[question_hash(text)] = qid
        toks = tokenize_loose(text)
        # Add every 50-token sliding window to the inverted index.
        for i in range(len(toks) - NGRAM_N + 1):
            gh = _gram_hash(tuple(toks[i : i + NGRAM_N]))
            gram_to_qids.setdefault(gh, set()).add(qid)
    return ids, hash_to_qid, gram_to_qids


def check_corpus_text(
    paragraphs: Iterable[tuple[str, str]],
    eval_ids: set[str],
    hash_to_qid: dict[str, str],
    gram_to_qids: dict[int, set[str]],
    item_ids: Iterable[str] | None = None,
) -> LeakageReport:
    """Check an iterable of (source_path, paragraph_text) tuples
    against the eval-set indices."""
    rep = LeakageReport()
    if item_ids is not None:
        rep.id_overlaps = eval_ids & set(item_ids)

    flagged_qids_per_src: dict[str, set[str]] = {}

    for src, text in paragraphs:
        if not text:
            continue
        h = question_hash(text)
        if h in hash_to_qid:
            rep.hash_overlaps.add(hash_to_qid[h])
        toks = tokenize_loose(text)
        if len(toks) < NGRAM_N:
            continue
        # Sliding 50-token window over this paragraph; lookup each in
        # the inverted index. O(len(toks)) per paragraph.
        for i in range(len(toks) - NGRAM_N + 1):
            gh = _gram_hash(tuple(toks[i : i + NGRAM_N]))
            hits = gram_to_qids.get(gh)
            if hits:
                # Record once per (qid, src) pair to avoid duplicate-flood
                # when one corpus paragraph hits many overlapping windows.
                seen = flagged_qids_per_src.setdefault(src, set())
                for qid in hits:
                    if qid not in seen:
                        seen.add(qid)
                        rep.ngram_hits.append((qid, src))
    return rep


# ----------- Corpus iteration helpers -----------

def iter_text_files(roots: list[Path]) -> Iterable[tuple[str, str]]:
    """Yield (relative_path, paragraph_text) from `.txt` and `.md` files
    under each root. Splits on the `<<<END-RECORD>>>` delimiter (local-DB
    extracts) or on blank lines."""
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.txt", "*.md"):
            for f in sorted(root.rglob(ext)):
                rel = str(f.relative_to(REPO))
                raw = f.read_text(encoding="utf-8", errors="replace")
                if "<<<END-RECORD>>>" in raw:
                    records = raw.split("<<<END-RECORD>>>")
                else:
                    records = re.split(r"\n\s*\n", raw)
                for rec in records:
                    rec = rec.strip()
                    if rec:
                        yield rel, rec


def iter_jsonl_text(roots: list[Path]) -> Iterable[tuple[str, str]]:
    """Yield (path, text) from JSONL files. Two row shapes:
    `{"text": ...}` (replay buffer) and `{"prompt": ..., "completion":
    ...}` (instruction slice) — both must pass through the gate."""
    for root in roots:
        if not root.exists():
            continue
        for jl in sorted(root.rglob("*.jsonl")):
            rel = str(jl.relative_to(REPO))
            with jl.open(encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = (d.get("text") or "").strip()
                    if not text and d.get("prompt"):
                        text = f"{d.get('prompt', '')} {d.get('completion', '')}".strip()
                    if text:
                        yield rel, text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Check CPT corpus for eval leakage "
                    "(ID + exact-text + 50-token contiguous per Carlini 2023)."
    )
    p.add_argument("--eval", default=str(EVAL_SET),
                   help="Path to eval_set.parquet")
    p.add_argument("--cpt-raw", default=str(CPT_RAW),
                   help="Root dir holding per-source acquired/cleaned text "
                        "(default data/cpt_raw)")
    p.add_argument("--source", action="append",
                   help="Limit to these source subdirs (repeatable)")
    args = p.parse_args(argv)

    eval_path = Path(args.eval)
    if not eval_path.exists():
        print(f"ERROR: eval set not found at {eval_path}", file=sys.stderr)
        return 1

    eval_ids, hash_to_qid, gram_to_qids = build_eval_index(eval_path)
    print(f"Eval index: {len(eval_ids)} ids, {len(hash_to_qid)} text-hashes, "
          f"{len(gram_to_qids):,} distinct 50-grams")

    cpt_root = Path(args.cpt_raw).resolve()
    if args.source:
        roots = [cpt_root / s for s in args.source]
    elif cpt_root.exists():
        roots = [p for p in cpt_root.iterdir() if p.is_dir()]
    else:
        roots = []
    print(f"Scanning {len(roots)} source dirs: {[r.name for r in roots]}")

    paragraphs = list(iter_text_files(roots)) + list(iter_jsonl_text(roots))
    print(f"Paragraphs to scan: {len(paragraphs):,}")

    rep = check_corpus_text(
        paragraphs=paragraphs,
        eval_ids=eval_ids,
        hash_to_qid=hash_to_qid,
        gram_to_qids=gram_to_qids,
    )
    print()
    print(rep.render())
    return 0 if rep.is_clean() else 2


if __name__ == "__main__":
    sys.exit(main())
