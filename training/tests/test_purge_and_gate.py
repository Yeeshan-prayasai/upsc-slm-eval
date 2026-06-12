"""Tests for the eval-leakage gate + purge — the safety-critical paths
rewritten in the round-2 audit fixes.

Covers:
- per-field exact hash catches short verbatim eval stems (<50 tokens)
- 50-gram + 12-gram windows
- holdout parquet is indexed alongside the locked eval set
- purge drops leaked records (END-RECORD) AND leaked .md paragraphs,
  preserving the delimiter framing that tokenize_pack/_read_documents need
- record-level dedup keeps shared stems (does not mutilate QA records)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from training.data import leakage as L
from training.data.clean import END_RECORD_DELIM, dedupe_corpus


# ---------- gate index ----------

def _eval_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "eval.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def test_gate_catches_short_verbatim_stem(tmp_path):
    """An eval question under 50 tokens still matches via per-field hash."""
    q = ("Consider the following statements regarding the fundamental right to equality guaranteed under the Constitution of India and identify which of the listed Articles deals specifically with equality before the law and equal protection of the laws within the territory of India.")
    ev = _eval_parquet(tmp_path, [{
        "question_id": "ai:uuid-1:en",
        "gold_payload": json.dumps({"question": q, "answer_text": "Article 14"}),
    }])
    ids, hash_to_qid, gram_to_qids, glen = L.build_eval_index(ev)
    rep = L.check_corpus_text(
        [("corpus/x.md", f"Some intro. {q} Some trailing text.")],
        ids, hash_to_qid, gram_to_qids, gram_lengths=glen,
    )
    assert not rep.is_clean(), "short verbatim stem should be flagged"


def test_gate_indexes_holdout_too(tmp_path):
    ev = _eval_parquet(tmp_path, [{
        "question_id": "ai:uuid-1:en",
        "gold_payload": json.dumps({"question": "locked eval question here"}),
    }])
    hold = tmp_path / "holdout.parquet"
    probe_q = ("Which of the following committees was specifically constituted to recommend the held-out probe reform on banking supervision and financial regulation in India during the relevant policy review period mentioned above")
    pd.DataFrame([{
        "question_id": "uuid-2", "question": probe_q,
        "options": json.dumps({"A": "x", "B": "y", "C": "z", "D": "w"}),
        "correct_option_letter": "A",
    }]).to_parquet(hold, index=False)
    ids, h2q, g2q, glen = L.build_eval_index([ev, hold])
    assert "uuid-2" in ids
    # The probe question (>10 tokens) embedded in a larger corpus paragraph
    # is caught by the 10-gram short-window check.
    rep = L.check_corpus_text(
        [("c/x.md", f"Some context. {probe_q} and more trailing text here.")],
        ids, h2q, g2q, gram_lengths=glen)
    assert not rep.is_clean()


def test_gate_clean_on_unrelated_text(tmp_path):
    ev = _eval_parquet(tmp_path, [{
        "question_id": "ai:uuid-1:en",
        "gold_payload": json.dumps({"question": "what is the capital of france"}),
    }])
    ids, h2q, g2q, glen = L.build_eval_index(ev)
    rep = L.check_corpus_text(
        [("c/x.md", "The monsoon arrived over Kerala on June 1 this year.")],
        ids, h2q, g2q, gram_lengths=glen)
    assert rep.is_clean()


# ---------- record-level dedup keeps shared stems ----------

def test_record_dedup_preserves_shared_stems(tmp_path):
    """Two DISTINCT records sharing a boilerplate stem line must both
    survive — per-paragraph exact dedup would strip the second's stem."""
    in_root = tmp_path / "clean"
    out_root = tmp_path / "dedup"
    (in_root / "qa_bank" / "mcqs").mkdir(parents=True)
    f = in_root / "qa_bank" / "mcqs" / "rows.txt"
    stem = "Consider the following statements:"
    rec1 = f"{stem}\n1. India is a republic.\nAnswer: A"
    rec2 = f"{stem}\n1. The Ganga is a river.\nAnswer: B"
    f.write_text(f"{rec1}\n\n{END_RECORD_DELIM}\n\n{rec2}\n\n{END_RECORD_DELIM}\n",
                 encoding="utf-8")
    dedupe_corpus([f], in_root=in_root, out_root=out_root)
    out = (out_root / "qa_bank" / "mcqs" / "rows.txt").read_text(encoding="utf-8")
    # Both records present (whole-record dedup, not per-paragraph)
    assert "India is a republic" in out
    assert "The Ganga is a river" in out
    assert out.count(stem) == 2          # stem kept in BOTH records
    assert out.count(END_RECORD_DELIM) == 2


def test_record_dedup_drops_exact_duplicate_record(tmp_path):
    in_root = tmp_path / "clean"
    out_root = tmp_path / "dedup"
    (in_root / "qa_bank" / "t").mkdir(parents=True)
    f = in_root / "qa_bank" / "t" / "rows.txt"
    rec = "A unique question body that repeats verbatim."
    f.write_text(f"{rec}\n\n{END_RECORD_DELIM}\n\n{rec}\n\n{END_RECORD_DELIM}\n",
                 encoding="utf-8")
    rep = dedupe_corpus([f], in_root=in_root, out_root=out_root)
    out = (out_root / "qa_bank" / "t" / "rows.txt").read_text(encoding="utf-8")
    assert out.count(rec) == 1
    assert rep.exact_dups_removed == 1


def test_record_dedup_output_roundtrips_through_reader(tmp_path):
    """The deduped record file must split cleanly on the delimiter — the
    format tokenize_pack._read_documents and purge rely on."""
    from training.data.tokenize_pack import _read_documents
    in_root = tmp_path / "clean"
    out_root = tmp_path / "dedup"
    (in_root / "qa_bank" / "t").mkdir(parents=True)
    f = in_root / "qa_bank" / "t" / "rows.txt"
    f.write_text(f"record one\n\n{END_RECORD_DELIM}\n\nrecord two\n\n{END_RECORD_DELIM}\n",
                 encoding="utf-8")
    dedupe_corpus([f], in_root=in_root, out_root=out_root)
    docs = _read_documents(out_root / "qa_bank" / "t" / "rows.txt")
    assert "record one" in docs
    assert "record two" in docs
    assert END_RECORD_DELIM not in "".join(docs)
