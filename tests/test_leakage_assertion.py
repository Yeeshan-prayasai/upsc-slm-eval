"""The production leakage assertion in scripts/build_ft_corpus.py must trip when
an eval-set ID is present in the FT corpus, and pass when they are disjoint.
Validated against the real function — no shadow re-implementation.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_ft_corpus import assert_no_leakage


def test_clean_separation_passes():
    assert_no_leakage({"a:1:en", "a:2:hi", "b:3:en"}, {"a:9:en", "b:10:en"})


def test_single_overlap_fails():
    with pytest.raises(AssertionError, match="LEAKAGE: 1"):
        assert_no_leakage({"a:1:en"}, {"a:1:en", "a:2:hi"})


def test_full_overlap_fails():
    with pytest.raises(AssertionError, match=r"LEAKAGE: 3"):
        assert_no_leakage({"a:1:en", "a:2:hi", "b:3:en"},
                          {"a:1:en", "a:2:hi", "b:3:en"})


def test_empty_sets_pass():
    assert_no_leakage(set(), set())
