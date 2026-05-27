"""The production leakage assertion in scripts/build_ft_corpus.py must trip on
two leakage modes:
  1. ID-level — an eval pair_id appears in the FT corpus.
  2. Content-level — an eval question's normalized text hash appears in an FT
     input. Catches the case where the same UPSC question lives in multiple
     source tables with different pair_ids (e.g., prelims_pyq_questions +
     prod.mcqs), which the ID check would miss.

Validated against the real function — no shadow re-implementation.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_ft_corpus import assert_no_leakage, _question_hash


# --- ID-level checks (existing behavior) -----------------------------------

def test_clean_separation_passes():
    assert_no_leakage({"a:1:en", "a:2:hi", "b:3:en"}, {"a:9:en", "b:10:en"})


def test_single_id_overlap_fails():
    with pytest.raises(AssertionError, match="ID-LEVEL LEAKAGE: 1"):
        assert_no_leakage({"a:1:en"}, {"a:1:en", "a:2:hi"})


def test_full_id_overlap_fails():
    with pytest.raises(AssertionError, match=r"ID-LEVEL LEAKAGE: 3"):
        assert_no_leakage({"a:1:en", "a:2:hi", "b:3:en"},
                          {"a:1:en", "a:2:hi", "b:3:en"})


def test_empty_sets_pass():
    assert_no_leakage(set(), set())


# --- Content-level checks (new) --------------------------------------------

_QUESTION_X = "Which Article of the Constitution guarantees the right to life?"
_QUESTION_Y = "What is the term of the Lok Sabha?"


def test_content_clean_passes():
    """No shared questions between eval and FT — both forms of check pass."""
    assert_no_leakage(
        eval_ids={"pyq:1:en"},
        ft_ids={"ai:5:en"},
        eval_question_hashes={_question_hash(_QUESTION_X)},
        ft_question_hashes={_question_hash(_QUESTION_Y)},
    )


def test_content_overlap_fails():
    """Same question text on both sides (different pair_ids) must trip."""
    with pytest.raises(AssertionError, match=r"CONTENT-LEVEL LEAKAGE: 1"):
        assert_no_leakage(
            eval_ids={"pyq:1:en"},               # eval has pyq:1
            ft_ids={"prod_mcq:abc:en"},          # ft has different pair_id
            eval_question_hashes={_question_hash(_QUESTION_X)},
            ft_question_hashes={_question_hash(_QUESTION_X)},   # but same text
        )


def test_content_whitespace_normalization():
    """Normalization makes 'X' and 'X\\n\\n  X' hash the same (whitespace-equivalent)."""
    assert _question_hash("Article 21 guarantees right to life.") == \
        _question_hash("  Article  21\n\nguarantees   right to life.  ")


def test_content_check_skipped_when_hashes_not_passed():
    """Old call sites that only pass IDs still work — content check is opt-in."""
    assert_no_leakage({"a:1"}, {"a:2"})       # no hash args — only ID check runs


def test_empty_string_hashes_excluded():
    """Empty-string question (defensive) shouldn't trigger false positives."""
    assert_no_leakage(
        eval_ids={"a:1"}, ft_ids={"a:2"},
        eval_question_hashes={_question_hash(""), _question_hash(_QUESTION_X)},
        ft_question_hashes={_question_hash(""), _question_hash(_QUESTION_Y)},
    )
