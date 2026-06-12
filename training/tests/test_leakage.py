"""Leakage-gate correctness tests.

Three failure modes must be caught:
1. ID-level: eval `question_id` appears in a per-source row index
2. Exact-text: SHA-256 of normalized eval question text appears verbatim
3. 50-token contiguous overlap: any 50-gram from eval passage in corpus

Plus a clean-corpus must report `is_clean() == True`.
"""
from __future__ import annotations

import re

import pytest

from training.data.leakage import (
    LeakageReport,
    NGRAM_N,
    check_corpus_text,
    normalize,
    question_hash,
    tokenize_loose,
)


def _build_indices(eval_passages: dict[str, str]):
    """Synthetic mini eval index from {qid: text} dict.
    Mimics `build_eval_index` but doesn't read parquet."""
    eval_ids = set(eval_passages)
    hash_to_qid: dict[str, str] = {}
    gram_to_qids: dict[int, set[str]] = {}
    for qid, text in eval_passages.items():
        if not text.strip():
            continue
        hash_to_qid[question_hash(text)] = qid
        toks = tokenize_loose(text)
        for i in range(len(toks) - NGRAM_N + 1):
            gh = hash(tuple(toks[i:i + NGRAM_N]))
            gram_to_qids.setdefault(gh, set()).add(qid)
    return eval_ids, hash_to_qid, gram_to_qids


def _filler(words: int) -> str:
    """Generic non-overlapping filler text — UPSC-themed prose
    that doesn't contain any of the test eval-passage tokens."""
    base = ("The Indian subcontinent has a rich historical tradition. "
            "Various civilizations flourished here over millennia. "
            "Trade networks connected distant cultural centers. ")
    return " ".join((base * 50).split()[:words])


def test_clean_corpus_passes():
    """No overlap → report.is_clean() True."""
    eval_passages = {"q1": "The Constitution of India was adopted on twenty sixth November "
                            "nineteen forty nine in a session of the constituent assembly "
                            "with representatives from across the newly formed nation."}
    ids, h2q, g2q = _build_indices(eval_passages)
    rep = check_corpus_text(
        paragraphs=[("/data/unrelated.txt", _filler(500))],
        eval_ids=ids, hash_to_qid=h2q, gram_to_qids=g2q,
    )
    assert rep.is_clean(), rep.render()


def test_id_level_leak_caught():
    """Eval qid in `item_ids` → ID overlap flagged."""
    eval_passages = {"q123": "stub"}
    ids, h2q, g2q = _build_indices(eval_passages)
    rep = check_corpus_text(
        paragraphs=[("/data/clean.txt", _filler(500))],
        eval_ids=ids, hash_to_qid=h2q, gram_to_qids=g2q,
        item_ids=["q123", "q999"],   # q123 is in eval set
    )
    assert not rep.is_clean()
    assert rep.id_overlaps == {"q123"}


def test_exact_text_leak_caught():
    """Verbatim eval question stem in a corpus paragraph → hash overlap."""
    eval_text = ("Article twenty one of the Indian Constitution guarantees the right to life "
                 "and personal liberty as a fundamental right that cannot be suspended.")
    ids, h2q, g2q = _build_indices({"qHASH": eval_text})
    rep = check_corpus_text(
        paragraphs=[("/data/ncert/chap.md", eval_text)],
        eval_ids=ids, hash_to_qid=h2q, gram_to_qids=g2q,
    )
    assert not rep.is_clean()
    assert "qHASH" in rep.hash_overlaps


def test_50gram_contiguous_overlap_caught():
    """50 consecutive eval tokens embedded inside a longer corpus paragraph
    → 50-gram hit recorded against the eval qid."""
    # Build an eval passage with > 50 tokens
    eval_text = " ".join([f"word{i}" for i in range(80)])
    ids, h2q, g2q = _build_indices({"qNGRAM": eval_text})
    # Embed eval_text inside a longer corpus para
    contaminated = _filler(200) + " " + eval_text + " " + _filler(200)
    rep = check_corpus_text(
        paragraphs=[("/data/ncert/contaminated.md", contaminated)],
        eval_ids=ids, hash_to_qid=h2q, gram_to_qids=g2q,
    )
    assert not rep.is_clean()
    assert any(qid == "qNGRAM" for qid, _ in rep.ngram_hits)


def test_short_passage_below_ngram_threshold_no_false_alarm():
    """Eval passages shorter than NGRAM_N=50 tokens don't create any
    n-gram entries → no contiguous-overlap false positives possible."""
    short_eval = "Article 21"     # 2 tokens after tokenize_loose
    ids, h2q, g2q = _build_indices({"qSHORT": short_eval})
    # Build a paragraph containing the exact short text
    para = _filler(50) + " article 21 " + _filler(50)
    rep = check_corpus_text(
        paragraphs=[("/data/x.md", para)],
        eval_ids=ids, hash_to_qid=h2q, gram_to_qids=g2q,
    )
    # No 50-gram could have been built from "Article 21" — so no ngram_hits.
    # (The exact-text check ALSO won't trip because hashes are over the
    # full normalized passage, which differs from the contaminated para.)
    assert not rep.ngram_hits


def test_normalize_is_stable():
    """`normalize()` must produce the same hash for whitespace + case
    variants of the same text."""
    a = "The   Constitution OF India was Adopted"
    b = "the constitution of india was adopted"
    c = "The\n\tConstitution of India\t  was adopted"
    assert question_hash(a) == question_hash(b) == question_hash(c)


def test_tokenize_loose_preserves_numbers():
    """Per `feedback_keep_numbers_tables` memory: numbers must survive
    tokenization for both leakage detection AND CPT signal."""
    toks = tokenize_loose("Article 21 of 1950, the gdp grew by 7.5 percent")
    assert "21" in toks
    assert "1950" in toks
    assert "7" in toks and "5" in toks   # 7.5 splits on the period
    assert "gdp" in toks
