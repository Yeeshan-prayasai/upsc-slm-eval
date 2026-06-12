"""Tests for the mix-weighting cap accounting and the OCR FFFD rescue —
both rewritten in the round-2 audit fixes."""
from __future__ import annotations

from training.data.tokenize_pack import SourceMix, _frac_keep, load_mix_config


def test_cap_and_repeat_mutually_exclusive(tmp_path):
    import pytest
    cfg = tmp_path / "m.yaml"
    cfg.write_text("sources:\n  x: {repeat: 2, cap_tokens: 100}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_mix_config(cfg)


def test_shipped_mix_caps_are_present():
    """Replay + commentary caps must be set (they're the band-enforcers)."""
    from training.data.tokenize_pack import DEFAULT_MIX_CONFIG
    mix = load_mix_config(DEFAULT_MIX_CONFIG)
    for s in ("slimpajama", "wikipedia", "orf", "mea"):
        assert mix[s].cap_tokens is not None, f"{s} should be capped"
    for s in ("ncert", "reference_books"):
        assert mix[s].repeat >= 4, f"{s} should repeat hard"
    # qa_bank must be a real, weighted source now (it has a dir).
    assert mix["qa_bank"].repeat >= 3


def test_frac_keep_deterministic(tmp_path):
    p = tmp_path / "d.md"
    first = _frac_keep(p, 0, 0.5)
    assert all(_frac_keep(p, 0, 0.5) == first for _ in range(20))


def test_fffd_rescue_triggers_on_dense_fffd(tmp_path, monkeypatch):
    """When pymupdf4llm output is FFFD-dense but plain get_text is clean,
    extract_pdf must fall back to the plain text."""
    import training.data.ocr as ocr_mod

    # Fake a 5-page doc: pymupdf4llm returns ligature-FFFD; raw is clean.
    dense = ("Na�onal Disaster Management. " * 50)   # ~10 FFFD/Kchar
    clean = ("National Disaster Management. " * 50)

    class _FakePage:
        def get_text(self, *a, **k): return clean

    class _FakeDoc:
        page_count = 5
        def __getitem__(self, i): return _FakePage()
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def close(self): pass

    import sys
    monkeypatch.setitem(sys.modules, "pymupdf", type("P", (), {
        "open": staticmethod(lambda *a, **k: _FakeDoc())})())
    monkeypatch.setitem(sys.modules, "pymupdf4llm", type("M", (), {
        "to_markdown": staticmethod(lambda *a, **k: dense)})())
    # extract_pdf mirrors REPO/data/cpt_raw → REPO/data/cpt_text.
    monkeypatch.setattr(ocr_mod, "REPO", tmp_path)
    monkeypatch.setattr(ocr_mod, "CPT_RAW", tmp_path / "data" / "cpt_raw")
    monkeypatch.setattr(ocr_mod, "CPT_TEXT", tmp_path / "data" / "cpt_text")
    pdf2 = tmp_path / "data" / "cpt_raw" / "dm_core" / "doc.pdf"
    pdf2.parent.mkdir(parents=True)
    pdf2.write_bytes(b"%PDF-1.4 fake")

    ocr_mod.extract_pdf(pdf2, force=True)
    out = (tmp_path / "data" / "cpt_text" / "dm_core" / "doc.md").read_text(encoding="utf-8")
    assert "�" not in out, "FFFD rescue should have replaced with clean text"
    assert "National Disaster" in out
