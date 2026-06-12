"""Tests for the pure-Python parts of `training.eval.mcq_inference`
and `PulseEvalCallback._items_to_mcq` (schema-normalization helper).

The actual `mcq_accuracy()` function exercises a live model and is
covered by the smoke test (`test_cpt_smoke.py`) — not here.

Covered:
- format_mcq_prompt: stable formatting (matches MMLU/BB style)
- extract_letter: extracts the first A-D character from a raw decode
  (handles whitespace, mixed case, dots, lowercase variants)
- _items_to_mcq: maps MMLU rows AND prod.mcqs rows to a shared schema
"""
from __future__ import annotations

import pytest

from training.eval.mcq_inference import extract_letter, format_mcq_prompt


def test_format_mcq_prompt_shape():
    """Prompt format: question, options labeled A-D, then 'Answer:'."""
    p = format_mcq_prompt(
        "Which Article guarantees the right to life?",
        {"A": "14", "B": "19", "C": "21", "D": "25"},
    )
    lines = p.splitlines()
    assert lines[0] == "Which Article guarantees the right to life?"
    assert lines[1] == "A. 14"
    assert lines[2] == "B. 19"
    assert lines[3] == "C. 21"
    assert lines[4] == "D. 25"
    assert lines[5] == "Answer:"


def test_format_mcq_prompt_preserves_option_order():
    """Option ordering follows the dict's insertion order (Py3.7+).
    Production caller must always pass A→D in order, but we sanity-check
    the format respects whatever order it gets."""
    p = format_mcq_prompt("Q?", {"A": "x", "B": "y", "C": "z", "D": "w"})
    # Each option appears on its own line in alphabetical/dict order
    for letter in "ABCD":
        assert f"\n{letter}. " in p


@pytest.mark.parametrize("raw,expected", [
    (" C", "C"),
    ("C", "C"),
    ("C.", "C"),
    ("Option B is best", "B"),
    ("The answer is A.", "A"),
    ("d.", "D"),                       # lowercase → uppercase
    ("Answer: D", "D"),
    ("...A...", "A"),
    ("\n\nB\n", "B"),
])
def test_extract_letter_recovers_choice(raw, expected):
    """Extract the first A-D letter from a noisy generation tail."""
    assert extract_letter(raw) == expected


@pytest.mark.parametrize("raw", [
    "",                # empty
    "no letter here",  # purely lowercase letters that aren't A-D
    "EFGH",            # letters outside A-D
    "1 2 3",           # digits only
    "...",             # punctuation only
])
def test_extract_letter_returns_none_when_no_choice(raw):
    """No A/B/C/D in the decode → return None (caller treats as wrong)."""
    assert extract_letter(raw) is None


def test_extract_letter_picks_first_match():
    """Multiple letters in the decode → first one wins (greedy decode
    should rarely produce more than one in practice)."""
    assert extract_letter("Could be C or A actually") == "C"


# ----- _items_to_mcq schema normalization -----

@pytest.fixture
def callback():
    """Construct a PulseEvalCallback for its `_items_to_mcq` helper —
    avoid loading any HF model/dataset by using a tmp output dir."""
    import tempfile
    from training.eval.pulse import PulseConfig, PulseEvalCallback
    out = tempfile.mkdtemp(prefix="pulse_test_")
    return PulseEvalCallback(PulseConfig(), out)


def test_items_to_mcq_handles_mmlu_schema(callback):
    """MMLU rows: choices=[4 strings], answer=int 0-3 → A-D letter."""
    items = [
        {"question": "Q?", "choices": ["w", "x", "y", "z"], "answer": 2},
        {"question": "Q2?", "choices": ["p", "q", "r", "s"], "answer": 0},
    ]
    out = callback._items_to_mcq(items)
    assert len(out) == 2
    assert out[0]["gold_letter"] == "C"
    assert out[0]["options"] == {"A": "w", "B": "x", "C": "y", "D": "z"}
    assert out[1]["gold_letter"] == "A"


def test_items_to_mcq_handles_native_schema(callback):
    """prod.mcqs rows: options=dict, correct_option_letter='A'-'D'."""
    items = [
        {"question": "Q?",
         "options": {"A": "w", "B": "x", "C": "y", "D": "z"},
         "correct_option_letter": "C"},
    ]
    out = callback._items_to_mcq(items)
    assert len(out) == 1
    assert out[0]["gold_letter"] == "C"
    assert out[0]["options"]["A"] == "w"


def test_items_to_mcq_parses_json_options_string(callback):
    """If `options` is a JSON string (raw DB row), parse it before
    accepting."""
    items = [
        {"question": "Q?",
         "options": '{"A": "w", "B": "x", "C": "y", "D": "z"}',
         "correct_option_letter": "B"},
    ]
    out = callback._items_to_mcq(items)
    assert len(out) == 1
    assert out[0]["gold_letter"] == "B"


def test_items_to_mcq_skips_malformed(callback):
    """Rows missing required fields or with bad shape are silently
    skipped (don't crash the pulse)."""
    items = [
        {"question": "Q?", "choices": ["a", "b"], "answer": 0},        # only 2 choices
        {"question": "Q2?", "choices": ["a", "b", "c", "d"], "answer": "C"},  # str not int
        {"question": "Q3?", "options": "not-json", "correct_option_letter": "A"},
        {"question": "Q4?", "options": {"A": "x"}, "correct_option_letter": "Z"},  # wrong letter
        {"question": "Q5?",                                              # no opts/gold at all
         "choices": ["a", "b", "c", "d"], "answer": 1},                  # — this one is valid
    ]
    out = callback._items_to_mcq(items)
    # Only the last item is valid (4 choices, int answer 1→'B')
    assert len(out) == 1
    assert out[0]["gold_letter"] == "B"


def test_items_to_mcq_empty(callback):
    """Empty input → empty output (no crash)."""
    assert callback._items_to_mcq([]) == []
