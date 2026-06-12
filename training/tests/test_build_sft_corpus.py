"""Tests for the pure helpers in `training.data.build_sft_corpus`.

The full builder (`build()`) requires the v1 ft_corpus parquet + the
local SQLite DB, both of which exist on the workstation but aren't
checked into git — so the end-to-end build is exercised manually via
the CLI. Here we test only the deterministic pure helpers that have
correctness implications for the SFT length-penalty signal:

- `_qid_from_pair_id`: only the `mains:` namespace returns a question_id
- `_build_prompt`: user-turn content matches the inference prompt
  (instruction + input), plus the length hint when a target is known
- `_estimate_tokens`: over-length filter arithmetic
- `_split_stratified`: every task present in input is present in
  both train and valid splits (no minority-task disappearance)
"""
from __future__ import annotations

import pandas as pd
import pytest

from training.data.build_sft_corpus import (
    MAX_TRAIN_TOKENS,
    _build_prompt,
    _estimate_tokens,
    _qid_from_pair_id,
    _split_stratified,
)


@pytest.mark.parametrize("pair_id, expected", [
    ("mains:abc-def-123:en", "abc-def-123"),
    ("mains:abc-def-123:hi", "abc-def-123"),
    ("mains:abc:en", "abc"),
    # Other namespaces — no qid: target_word_count stays None.
    ("prod_mcq:x:en", None),
    ("pyq:y:en", None),
    ("news:z", None),
    ("eval:abc:def", None),
    # Malformed
    ("mains", None),
    ("", None),
    ("nocolon", None),
])
def test_qid_from_pair_id(pair_id, expected):
    """Only `mains:<qid>:...` parses to a qid; everything else None."""
    assert _qid_from_pair_id(pair_id) == expected


def test_build_prompt_matches_inference_format():
    """instruction → input, joined by a blank line — byte-identical to
    the prompt runners.py wraps in the chat template at inference."""
    row = pd.Series({
        "instruction": "[TASK=A] Answer this.",
        "input": "Q: What is X?",
    })
    assert _build_prompt(row, None) == "[TASK=A] Answer this.\n\nQ: What is X?"


def test_build_prompt_handles_empty_input():
    """If `input` is empty/None, just the instruction."""
    row = pd.Series({
        "instruction": "[TASK=B] Write an essay.",
        "input": "",
    })
    assert _build_prompt(row, None) == "[TASK=B] Write an essay."


def test_build_prompt_appends_length_hint():
    """Rows with a known target carry the length instruction — this IS
    the length-control mechanism (data-side, learned by plain CE)."""
    row = pd.Series({
        "instruction": "[TASK=B] Write an answer.",
        "input": "Q: Discuss X.",
    })
    out = _build_prompt(row, 150)
    assert out.endswith("Answer in approximately 150 words.")
    assert _build_prompt(row, None) == "[TASK=B] Write an answer.\n\nQ: Discuss X."


def test_estimate_tokens_flags_overlength():
    """A ~4000-word article + answer estimates above the 4096 budget."""
    long_prompt = "word " * 3500
    assert _estimate_tokens(long_prompt, "short answer") > MAX_TRAIN_TOKENS
    assert _estimate_tokens("short question", "short answer") < MAX_TRAIN_TOKENS


def test_split_stratified_preserves_all_tasks():
    """Every task in input must appear in BOTH train and valid splits."""
    df = pd.DataFrame({
        "task": (["A"] * 100) + (["B"] * 20) + (["C"] * 30) + (["E"] * 10),
        "text": [f"t{i}" for i in range(160)],
        "target_word_count": [None] * 160,
        "pair_id": [f"id{i}" for i in range(160)],
    })
    train, valid = _split_stratified(df, valid_frac=0.05, seed=42)
    assert set(train["task"]) == {"A", "B", "C", "E"}
    assert set(valid["task"]) == {"A", "B", "C", "E"}
    # Minority task E (10 rows, 5% = 0.5 → ceil to 1) should still appear.
    assert (valid["task"] == "E").sum() >= 1
    assert (train["task"] == "E").sum() >= 1


def test_split_stratified_no_row_loss():
    """train + valid row count must exactly equal input row count."""
    df = pd.DataFrame({
        "task": (["A"] * 50) + (["B"] * 20),
        "text": [f"t{i}" for i in range(70)],
        "target_word_count": [None] * 70,
        "pair_id": [f"id{i}" for i in range(70)],
    })
    train, valid = _split_stratified(df, valid_frac=0.10, seed=42)
    assert len(train) + len(valid) == len(df)
    # No id appears in both splits
    assert set(train["pair_id"]).isdisjoint(set(valid["pair_id"]))


def test_split_stratified_deterministic():
    """Same seed → identical split (reproducibility for the trainer)."""
    df = pd.DataFrame({
        "task": (["A"] * 100) + (["B"] * 20),
        "text": [f"t{i}" for i in range(120)],
        "target_word_count": [None] * 120,
        "pair_id": [f"id{i}" for i in range(120)],
    })
    t1, v1 = _split_stratified(df, valid_frac=0.05, seed=2026)
    t2, v2 = _split_stratified(df, valid_frac=0.05, seed=2026)
    assert list(t1["pair_id"]) == list(t2["pair_id"])
    assert list(v1["pair_id"]) == list(v2["pair_id"])


def test_split_stratified_min_one_per_task():
    """A task with only 1-2 rows still gets at least one valid row."""
    df = pd.DataFrame({
        "task": (["A"] * 100) + (["B"] * 2),
        "text": [f"t{i}" for i in range(102)],
        "target_word_count": [None] * 102,
        "pair_id": [f"id{i}" for i in range(102)],
    })
    train, valid = _split_stratified(df, valid_frac=0.05, seed=42)
    assert (valid["task"] == "B").sum() >= 1
    assert (train["task"] == "B").sum() >= 1
