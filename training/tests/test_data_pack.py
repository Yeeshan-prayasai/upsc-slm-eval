"""Sequence-packing invariants for the CPT corpus.

Tests `pack_token_streams()` — the pure pack-loop that turns N
pre-tokenized documents into K fixed-length sequences with an EOS
separator between docs. Trailing partial chunk is dropped (standard
CPT recipe — Llama 3, Gemma 3, Qwen 3).

Invariants under test:
1. Empty input → empty output, no sequences
2. One doc that exactly fills `seq_len` (with EOS) → exactly one sequence
3. EOS appears between concatenated docs, never elided
4. Multi-doc concatenation: total emitted tokens = sum(doc_lens) + n_docs
   for fully-packed sequences (each doc contributes len(ids)+1 to the
   stream, sequences are seq_len-long)
5. Trailing tail is dropped: `n_trailing_dropped < seq_len` always
6. Empty docs are skipped (no EOS inserted for empty list)
7. BOS prepended per doc when `bos_id` given (Gemma); never doubled
8. Mix config: fractional repeats are deterministic; cap+repeat>1 rejected
"""
from __future__ import annotations

import pytest

from training.data.tokenize_pack import (
    SourceMix,
    _frac_keep,
    load_mix_config,
    pack_token_streams,
)


def test_empty_input_emits_nothing():
    """No documents → no sequences."""
    seqs, trailing = pack_token_streams([], eos_id=0, seq_len=16)
    assert seqs == []
    assert trailing == 0


def test_single_doc_exactly_fills_seq():
    """A doc of `seq_len - 1` tokens + 1 EOS = exactly seq_len.
    → one sequence emitted, zero trailing."""
    seq_len = 16
    eos = 99
    doc = list(range(1, seq_len))   # 15 tokens
    seqs, trailing = pack_token_streams([doc], eos_id=eos, seq_len=seq_len)
    assert len(seqs) == 1
    assert seqs[0] == doc + [eos]
    assert trailing == 0


def test_eos_inserted_between_docs():
    """Two docs → concatenated stream is [doc1, EOS, doc2, EOS]."""
    seq_len = 32
    eos = 99
    doc1 = [1, 2, 3]
    doc2 = [4, 5, 6, 7]
    seqs, trailing = pack_token_streams([doc1, doc2], eos_id=eos, seq_len=seq_len)
    # No full sequence — total stream is 3+1+4+1 = 9 tokens, < seq_len=32
    assert seqs == []
    assert trailing == 9


def test_trailing_tail_under_seq_len():
    """For any pack output: tail < seq_len."""
    seq_len = 8
    eos = 0
    # 5 docs of 7 tokens each → each contributes 7+1 = 8 tokens
    # 5 × 8 = 40 → exactly 5 sequences, trailing = 0
    docs = [[i] * 7 for i in range(1, 6)]
    seqs, trailing = pack_token_streams(docs, eos_id=eos, seq_len=seq_len)
    assert len(seqs) == 5
    assert trailing == 0
    # Each sequence: 7 doc tokens + 1 EOS
    for i, seq in enumerate(seqs, start=1):
        assert seq == [i] * 7 + [eos]


def test_partial_trailing_is_dropped():
    """A stream of N tokens where N % seq_len != 0 → trailing tokens
    dropped (not padded, not emitted as a short sequence)."""
    seq_len = 10
    eos = 0
    # Two docs: 6 + 6 tokens → with EOS = 7 + 7 = 14 tokens
    # That's 1 full sequence of 10 + 4 trailing dropped.
    docs = [[1] * 6, [2] * 6]
    seqs, trailing = pack_token_streams(docs, eos_id=eos, seq_len=seq_len)
    assert len(seqs) == 1
    assert trailing == 4
    # The full sequence is the first 10 of the stream:
    # [1,1,1,1,1,1, 0, 2,2,2]
    assert seqs[0] == [1] * 6 + [eos] + [2] * 3


def test_empty_docs_skipped():
    """Empty docs are skipped (no EOS emitted for them)."""
    seq_len = 8
    eos = 99
    docs = [[1, 2, 3], [], [4, 5, 6]]   # middle doc is empty
    seqs, trailing = pack_token_streams(docs, eos_id=eos, seq_len=seq_len)
    # Stream: [1,2,3, 99, 4,5,6, 99] → 8 tokens → exactly one sequence
    assert len(seqs) == 1
    assert seqs[0] == [1, 2, 3, eos, 4, 5, 6, eos]
    assert trailing == 0


def test_eos_count_equals_doc_count():
    """For a fully-consumed input (no trailing), the count of EOS tokens
    across all output sequences equals the number of non-empty docs."""
    seq_len = 12
    eos = 7
    docs = [[1] * 5, [2] * 5, [3] * 5, [4] * 5]  # 4 × (5+1) = 24 tokens
    seqs, trailing = pack_token_streams(docs, eos_id=eos, seq_len=seq_len)
    assert trailing == 0
    total_eos = sum(seq.count(eos) for seq in seqs)
    assert total_eos == len(docs)


def test_doc_boundaries_preserved_via_eos():
    """The EOS at doc boundaries must always be present in the stream,
    even when it lands at a sequence boundary (last token of seq N or
    first token of seq N+1)."""
    seq_len = 4
    eos = 0
    # Doc1 = 3 tokens → with EOS = 4 (exactly fills seq 1)
    # Doc2 = 3 tokens → with EOS = 4 (exactly fills seq 2)
    docs = [[1, 1, 1], [2, 2, 2]]
    seqs, trailing = pack_token_streams(docs, eos_id=eos, seq_len=seq_len)
    assert len(seqs) == 2
    assert seqs[0] == [1, 1, 1, eos]   # EOS is last token of seq 1
    assert seqs[1] == [2, 2, 2, eos]
    assert trailing == 0


# ----- BOS handling (Gemma is BOS-sensitive; Qwen has no BOS) -----

def test_bos_prepended_per_doc():
    """With `bos_id`, every doc becomes [BOS] + ids + [EOS]."""
    seq_len = 5
    eos, bos = 0, 9
    docs = [[1, 2, 3], [4, 5, 6]]
    seqs, trailing = pack_token_streams(docs, eos_id=eos, seq_len=seq_len, bos_id=bos)
    assert len(seqs) == 2
    assert seqs[0] == [bos, 1, 2, 3, eos]
    assert seqs[1] == [bos, 4, 5, 6, eos]
    assert trailing == 0


def test_bos_not_doubled_when_already_present():
    """Docs already starting with BOS (pre-templated instruction rows)
    must not get a second BOS."""
    seq_len = 5
    eos, bos = 0, 9
    docs = [[bos, 1, 2, 3]]
    seqs, trailing = pack_token_streams(docs, eos_id=eos, seq_len=seq_len, bos_id=bos)
    assert seqs == [[bos, 1, 2, 3, eos]]
    assert trailing == 0


def test_no_bos_when_bos_id_none():
    """Default (Qwen path): no BOS anywhere."""
    seq_len = 4
    eos = 0
    docs = [[1, 2, 3]]
    seqs, _ = pack_token_streams(docs, eos_id=eos, seq_len=seq_len, bos_id=None)
    assert seqs == [[1, 2, 3, eos]]


# ----- Mix config -----

def test_frac_keep_is_deterministic(tmp_path):
    """Fractional-repeat inclusion is a pure function of (path, rep_idx)."""
    p = tmp_path / "doc.md"
    first = _frac_keep(p, 0, 0.5)
    assert all(_frac_keep(p, 0, 0.5) == first for _ in range(10))


def test_frac_keep_rate_approximates_fraction(tmp_path):
    """Over many paths, the keep rate approximates the fraction."""
    keeps = sum(
        _frac_keep(tmp_path / f"doc{i}.md", 0, 0.5) for i in range(1000)
    )
    assert 400 < keeps < 600


def test_mix_config_rejects_cap_with_repeat(tmp_path):
    """cap_tokens + repeat>1 is a config error (cap-skipped docs would
    slip back in on later repeats)."""
    cfg = tmp_path / "mix.yaml"
    cfg.write_text("sources:\n  orf: {repeat: 2, cap_tokens: 1000}\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_mix_config(cfg)


def test_mix_config_missing_file_fails_loud(tmp_path):
    """An absent mix config must hard-fail — the unweighted corpus is
    the silent failure this stage exists to prevent."""
    with pytest.raises(FileNotFoundError):
        load_mix_config(tmp_path / "nope.yaml")


def test_mix_config_parses_repeat_and_cap(tmp_path):
    cfg = tmp_path / "mix.yaml"
    cfg.write_text(
        "sources:\n"
        "  ncert: {repeat: 4}\n"
        "  orf: {cap_tokens: 30000000}\n"
        "  pib: {}\n",
        encoding="utf-8",
    )
    mix = load_mix_config(cfg)
    assert mix["ncert"] == SourceMix(repeat=4.0, cap_tokens=None)
    assert mix["orf"] == SourceMix(repeat=1.0, cap_tokens=30_000_000)
    assert mix["pib"] == SourceMix(repeat=1.0, cap_tokens=None)


def test_shipped_mix_config_is_valid():
    """The repo's data_mix_cpt.yaml parses, is domain-dominant, and
    keeps replay capped."""
    from training.data.tokenize_pack import DEFAULT_MIX_CONFIG
    mix = load_mix_config(DEFAULT_MIX_CONFIG)
    assert mix["ncert"].repeat >= 4          # keystone repeated hardest
    assert mix["orf"].cap_tokens is not None  # commentary capped
    assert mix["slimpajama"].cap_tokens is not None
    assert mix["wikipedia"].cap_tokens is not None
    replay_cap = mix["slimpajama"].cap_tokens + mix["wikipedia"].cap_tokens
    assert replay_cap <= 150_000_000          # replay stays a minority share
