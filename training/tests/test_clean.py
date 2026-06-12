"""Cleaning-pass correctness for the CPT corpus.

Tests the per-document normalization + document floor filter, plus
the project-memory rule [[feedback_keep_numbers_tables]]: number-dense
and tabular content MUST survive the cleaning pass — that's UPSC
factual signal v1 was missing.

Invariants:
1. NFKC + targeted punctuation replacements applied, but numbers
   and units survive verbatim
2. Whitespace collapse preserves paragraph breaks (≥2 newlines → 2)
3. Word floor (<50 words, Penedo/Gopher) drops only tiny docs —
   single-paragraph full-length articles survive (the old line floor
   wrongly killed those)
4. Devanagari-majority docs dropped (English-only corpus)
5. Number-dense / tabular passages pass the floor (no special filter)
6. Paragraph splitter respects double-newline boundaries
7. Shingle generator preserves numeric tokens (UPSC factual signal)
8. Dedup is document-level near-dup (verified Jaccard ≥ 0.8) +
   paragraph-level exact; END-RECORD delimiters never deduped;
   replay .jsonl passes through untouched
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.data.clean import (
    END_RECORD_DELIM,
    LSH_THRESHOLD,
    MIN_WORDS_PER_DOC,
    SHINGLE_SIZE,
    _paragraphs,
    _shingles,
    clean_file,
    clean_jsonl_file,
    collapse_whitespace,
    dedupe_corpus,
    fix_idrop_typos,
    is_mrunal_cruft,
    normalize_unicode,
    source_priority_key,
    strip_dropcap_and_strikethrough,
)


def test_normalize_unicode_replacements():
    """Curly quotes, em-dashes, NBSP → ASCII equivalents."""
    raw = "It’s the “Constitution” — article 21 states..."
    out = normalize_unicode(raw)
    assert out == "It's the \"Constitution\" - article 21 states..."


def test_normalize_preserves_numbers_and_units():
    """Numbers, percentages, currencies, units must survive normalization
    (per feedback_keep_numbers_tables — UPSC factual content)."""
    raw = "GDP grew 7.5% to ₹23,84,000 cr (USD 2.8 T) in 2024-25."
    out = normalize_unicode(raw)
    # Every numeric/unit token survives
    for tok in ["7.5", "%", "23,84,000", "cr", "2.8", "T", "2024-25"]:
        assert tok in out, f"normalize stripped {tok!r}"


def test_collapse_whitespace_preserves_paragraph_breaks():
    """Double-newlines (paragraph boundary) → kept as exactly two newlines.
    Three+ newlines → collapsed to two."""
    raw = "Para one.\n\n\nPara two.\n\n\n\n\nPara three."
    out = collapse_whitespace(raw)
    assert out.count("\n\n") == 2
    assert "\n\n\n" not in out


def test_collapse_whitespace_collapses_intra_line_runs():
    """Multiple spaces/tabs within a line → single space."""
    raw = "Article    21\tof\t\tthe   Constitution"
    out = collapse_whitespace(raw)
    assert out == "Article 21 of the Constitution"


@pytest.fixture
def clean_tmp(tmp_path, monkeypatch):
    """`clean_file` computes paths relative to `REPO`; pin REPO to the
    pytest tmp dir for the duration of the test so tmp paths are valid."""
    import training.data.clean as clean_mod
    monkeypatch.setattr(clean_mod, "REPO", tmp_path)
    return tmp_path


def _write(tmp_path: Path, text: str) -> tuple[Path, Path]:
    src = tmp_path / "in.md"
    dst = tmp_path / "out.md"
    src.write_text(text, encoding="utf-8")
    return src, dst


# ~60-word filler so fixture docs clear the 50-word floor while the
# content under test stays small and readable.
_FILLER = (
    "This supporting paragraph exists to carry the document over the "
    "fifty word floor used by the cleaning pass. It mirrors the kind of "
    "explanatory prose that surrounds tables and statistics in the real "
    "corpus, where a figure is always introduced, presented, and then "
    "interpreted for the reader in plain language across several sentences."
)


def test_clean_file_drops_short_doc(clean_tmp):
    """Docs below the 50-word floor → dropped."""
    src, dst = _write(clean_tmp, "Too short.")
    res = clean_file(src, dst)
    assert res.dropped
    assert "words=" in res.drop_reason
    assert res.dst_path is None
    assert not dst.exists()


def test_clean_file_keeps_single_paragraph_article(clean_tmp):
    """A complete single-paragraph news article (1 line, 50+ words) must
    survive — the old 3-line floor wrongly dropped ~5.5% of the Hindu
    scrape this way."""
    text = ("The southwest monsoon arrived over Kerala on June 1, the "
            "India Meteorological Department said, marking a normal onset "
            "after two years of delays. Rainfall is expected to be above "
            "average in the core monsoon zone, supporting kharif sowing of "
            "rice, pulses and oilseeds across Punjab, Haryana, Uttar "
            "Pradesh, Bihar and Madhya Pradesh, the agency added in its "
            "first-stage forecast for the season.")
    assert "\n" not in text   # genuinely one line
    src, dst = _write(clean_tmp, text)
    res = clean_file(src, dst)
    assert not res.dropped, f"single-paragraph article dropped: {res.drop_reason}"


def test_clean_file_drops_devanagari_doc(clean_tmp):
    """A majority-Hindi document is dropped (English-only corpus)."""
    hindi_sentence = "भारत का संविधान देश का सर्वोच्च कानून है और यह शासन की रूपरेखा स्थापित करता है। "
    src, dst = _write(clean_tmp, hindi_sentence * 10)
    res = clean_file(src, dst)
    assert res.dropped
    assert "devanagari" in res.drop_reason


def test_clean_file_keeps_english_doc_quoting_hindi(clean_tmp):
    """English doc quoting a few Hindi terms must survive (threshold
    is 30% of alphabetic chars)."""
    text = _FILLER + ' The motto "सत्यमेव जयते" (Satyameva Jayate) appears on the State Emblem.'
    src, dst = _write(clean_tmp, text)
    res = clean_file(src, dst)
    assert not res.dropped, f"English doc with Hindi quote dropped: {res.drop_reason}"


def test_clean_file_keeps_normal_doc(clean_tmp):
    """A doc above the word floor passes through and is written."""
    text = ("The Constitution of India is the supreme law.\n"
            "It establishes the framework of governance.\n"
            "Article 21 guarantees the right to life.\n\n" + _FILLER)
    src, dst = _write(clean_tmp, text)
    res = clean_file(src, dst)
    assert not res.dropped
    assert dst.exists()
    assert dst.read_text(encoding="utf-8").startswith("The Constitution")


def test_clean_file_keeps_number_dense_doc(clean_tmp):
    """A number-dense doc (stats, percentages, years) must NOT be filtered
    out — feedback_keep_numbers_tables memory: UPSC factual signal."""
    text = ("India's GDP grew 7.5% in FY 2024-25 to ₹ 23,84,000 cr.\n"
            "Manufacturing share rose from 16.7% to 17.2% over 2020-25.\n"
            "Services contribute 54.3% of GDP as of 2025.\n"
            "Inflation moderated to 4.8% by Q4 FY25 from 6.4% in FY24.\n\n"
            + _FILLER)
    src, dst = _write(clean_tmp, text)
    res = clean_file(src, dst)
    assert not res.dropped, f"number-dense doc was dropped: {res.drop_reason}"
    cleaned = dst.read_text(encoding="utf-8")
    for fact in ["7.5%", "23,84,000", "16.7%", "17.2%", "54.3%", "4.8%", "6.4%"]:
        assert fact in cleaned, f"clean stripped fact {fact!r}"


def test_clean_file_keeps_tabular_doc(clean_tmp):
    """A markdown-table doc must NOT be filtered out — UPSC mains tables
    (population, fiscal, environmental data) are factual signal."""
    text = ("| Year | Pop (cr) | Lit (%) |\n"
            "|------|----------|---------|\n"
            "| 2001 | 102.9    | 64.8    |\n"
            "| 2011 | 121.0    | 74.0    |\n"
            "| 2021 | 138.0    | 77.7    |\n"
            "Source: Census of India.\n\n" + _FILLER)
    src, dst = _write(clean_tmp, text)
    res = clean_file(src, dst)
    assert not res.dropped, f"tabular doc was dropped: {res.drop_reason}"
    cleaned = dst.read_text(encoding="utf-8")
    for row in ["102.9", "121.0", "138.0", "64.8", "74.0", "77.7"]:
        assert row in cleaned


def test_paragraphs_splits_on_double_newline():
    """`_paragraphs` splits on blank-line boundaries, drops empties."""
    text = "First para.\n\n\nSecond para.\n\nThird.\n\n\n\n"
    paras = _paragraphs(text)
    assert paras == ["First para.", "Second para.", "Third."]


def test_shingles_preserve_numbers():
    """Shingles for MinHash must INCLUDE numeric tokens — required for
    correct near-dup detection on number-dense (UPSC factual) text."""
    text = "GDP grew 7 5 percent in FY 2024 25"
    grams = _shingles(text, n=SHINGLE_SIZE)
    assert any("2024" in g for g in grams), "MinHash dropped 2024"
    assert any("7" in g.split() for g in grams), "MinHash dropped numeric tokens"


def test_min_thresholds_match_literature():
    """Word floor per Penedo 2024 / Gopher rules; near-dup threshold
    per Lee et al. 2022 (document-level 0.8)."""
    assert MIN_WORDS_PER_DOC == 50
    assert LSH_THRESHOLD == 0.80
    assert SHINGLE_SIZE == 5


# ----- Corpus-level dedup -----

_DOC_A = (
    "The Indian National Congress was founded in 1885 by Allan Octavian "
    "Hume, a retired British civil servant, together with seventy-two "
    "delegates who met in Bombay. The early Congress followed a moderate "
    "programme of petitions and constitutional agitation, seeking greater "
    "Indian representation in legislative councils and the civil services "
    "while professing loyalty to the British Crown throughout its first "
    "two decades of existence as an organisation."
)


def test_dedupe_drops_near_duplicate_doc(tmp_path):
    """Two near-identical documents (one word changed) → second dropped,
    first (higher-priority order) kept."""
    in_root = tmp_path / "clean"
    out_root = tmp_path / "dedup"
    (in_root / "ncert").mkdir(parents=True)
    (in_root / "orf").mkdir(parents=True)
    keep = in_root / "ncert" / "a.md"
    drop = in_root / "orf" / "b.md"
    keep.write_text(_DOC_A, encoding="utf-8")
    drop.write_text(_DOC_A.replace("Bombay", "Mumbai"), encoding="utf-8")
    rep = dedupe_corpus([keep, drop], in_root=in_root, out_root=out_root)
    assert rep.near_dup_docs_removed == 1
    assert (out_root / "ncert" / "a.md").exists()
    assert not (out_root / "orf" / "b.md").exists()


def test_dedupe_keeps_distinct_docs_on_same_topic(tmp_path):
    """Topically-related but textually distinct docs both survive —
    the whole point of moving from paragraph-0.70 to doc-0.80-verified."""
    in_root = tmp_path / "clean"
    out_root = tmp_path / "dedup"
    (in_root / "src").mkdir(parents=True)
    a = in_root / "src" / "a.md"
    b = in_root / "src" / "b.md"
    a.write_text(_DOC_A, encoding="utf-8")
    b.write_text(
        "Article 21 of the Constitution guarantees that no person shall be "
        "deprived of life or personal liberty except according to procedure "
        "established by law. The Supreme Court in Maneka Gandhi v Union of "
        "India read this guarantee expansively, holding that the procedure "
        "must be fair, just and reasonable, and over the following decades "
        "derived rights to privacy, livelihood, shelter and a clean "
        "environment from the same clause.",
        encoding="utf-8",
    )
    rep = dedupe_corpus([a, b], in_root=in_root, out_root=out_root)
    assert rep.near_dup_docs_removed == 0
    assert (out_root / "src" / "a.md").exists()
    assert (out_root / "src" / "b.md").exists()


def test_dedupe_preserves_end_record_delimiter(tmp_path):
    """END-RECORD delimiters are structural — repeated occurrences must
    NOT be exact-deduped away (tokenize_pack splits records on them)."""
    in_root = tmp_path / "clean"
    out_root = tmp_path / "dedup"
    (in_root / "local_db").mkdir(parents=True)
    f = in_root / "local_db" / "rows.txt"
    f.write_text(
        f"Record one body text.\n\n{END_RECORD_DELIM}\n\n"
        f"Record two body text.\n\n{END_RECORD_DELIM}\n\n"
        f"Record three body text.\n\n{END_RECORD_DELIM}\n",
        encoding="utf-8",
    )
    dedupe_corpus([f], in_root=in_root, out_root=out_root)
    out = (out_root / "local_db" / "rows.txt").read_text(encoding="utf-8")
    assert out.count(END_RECORD_DELIM) == 3


def test_dedupe_jsonl_passthrough(tmp_path):
    """Replay .jsonl files copy through the dedup stage untouched."""
    in_root = tmp_path / "clean"
    out_root = tmp_path / "dedup"
    (in_root / "slimpajama").mkdir(parents=True)
    f = in_root / "slimpajama" / "sample.jsonl"
    rows = [json.dumps({"text": f"replay doc {i} " * 30}) for i in range(3)]
    f.write_text("\n".join(rows) + "\n", encoding="utf-8")
    rep = dedupe_corpus([f], in_root=in_root, out_root=out_root)
    assert rep.jsonl_passthrough == 1
    assert (out_root / "slimpajama" / "sample.jsonl").read_text(
        encoding="utf-8") == f.read_text(encoding="utf-8")


def test_clean_jsonl_file_floors_and_normalizes(tmp_path, monkeypatch):
    """Replay rows: short rows dropped, surviving rows normalized."""
    import training.data.clean as clean_mod
    monkeypatch.setattr(clean_mod, "REPO", tmp_path)
    src = tmp_path / "sample.jsonl"
    dst = tmp_path / "out.jsonl"
    long_text = "The quick brown fox jumps over the lazy dog near the river bank. " * 10
    src.write_text(
        json.dumps({"text": "too short"}) + "\n"
        + json.dumps({"text": long_text}) + "\n",
        encoding="utf-8",
    )
    res = clean_jsonl_file(src, dst)
    assert not res.dropped
    kept = [json.loads(l) for l in dst.read_text(encoding="utf-8").splitlines()]
    assert len(kept) == 1
    assert kept[0]["text"].startswith("The quick brown fox")


def test_source_priority_orders_ncert_before_orf():
    """NCERT (keystone) must dedup-process before ORF (commentary) so
    NCERT's copy survives any near-dup tie."""
    assert source_priority_key(Path("ncert/x.md")) < source_priority_key(
        Path("orf/y.md"))
    # Unlisted sources sort after all listed ones
    assert source_priority_key(Path("zzz_new_source/a.md")) > source_priority_key(
        Path("slimpajama/b.jsonl"))


# ----- Mrunal cruft filter -----

@pytest.mark.parametrize("body,expected", [
    ("# [Loot Lo] Disha Ebooks for only Rs. 5\nCategories: "
     "[Advertizement](https://mrunal.org/category/advertizement)|",
     True),
    ("# 5 Reasons You Need The X Theme\nCategories: "
     "[Top 10](https://mrunal.org/category/top-10)|Tags: ...",
     True),
    ("# [Download] Skholar Free Ebook\nCategories: "
     "[download](https://mrunal.org/category/download)|",
     True),
    ("# ACIO result announced\nCategories: "
     "[Notice Board](https://mrunal.org/category/noticeboard)|",
     True),
    ("# 4 Days Current: 25-28 March - Polity, Economy, IR\n"
     "Categories: [Current Affairs](https://mrunal.org/category/current-affairs)|",
     False),
    ("# Indian Economy Budget Analysis FY25-26\n"
     "Categories: [Economy](https://mrunal.org/category/economy)|", False),
    ("Some random text without categories header at all", False),
])
def test_is_mrunal_cruft(body, expected):
    """Mrunal's own category tags identify non-UPSC cruft cleanly."""
    assert is_mrunal_cruft(body) == expected


def test_clean_file_drops_mrunal_cruft(clean_tmp, monkeypatch, tmp_path):
    """Mrunal cruft category → dropped before normalization (path-keyed)."""
    # Simulate a file living under cpt_raw/mrunal/...
    mrunal_dir = tmp_path / "cpt_raw" / "mrunal"
    mrunal_dir.mkdir(parents=True)
    src = mrunal_dir / "5-reasons-you-need-the-x-theme.md"
    src.write_text(
        "# 5 Reasons You Need The X Theme\n"
        "Categories: [Top 10](https://mrunal.org/category/top-10)|\n"
        "Body text long enough to clear the FineWeb floor by a wide margin.\n"
        + ("Filler paragraph. " * 50),
        encoding="utf-8",
    )
    dst = clean_tmp / "out.md"
    res = clean_file(src, dst)
    assert res.dropped
    assert "mrunal cruft" in res.drop_reason
    assert not dst.exists()


def test_clean_file_keeps_mrunal_real_content(clean_tmp, tmp_path):
    """A real Mrunal current-affairs post (NOT in a cruft category)
    must survive the filter — we only drop the SEO/ebook/notice cruft."""
    mrunal_dir = tmp_path / "cpt_raw" / "mrunal"
    mrunal_dir.mkdir(parents=True)
    src = mrunal_dir / "4-days-current-from-21-22-23-24-march.md"
    src.write_text(
        "# 4 Days Current: 21-24 March - Polity, Economy, IR\n"
        "Categories: [Current Affairs](https://mrunal.org/category/current-affairs)|\n"
        "Inflation moderated to 4.8% by Q4 FY25 from 6.4% in FY24.\n"
        "Manufacturing share rose from 16.7% to 17.2% over 2020-25.\n\n"
        + _FILLER,
        encoding="utf-8",
    )
    dst = clean_tmp / "out.md"
    res = clean_file(src, dst)
    assert not res.dropped, f"real mrunal post was dropped: {res.drop_reason}"
    assert dst.exists()


# ----- Reference-book extraction-artifact cleanup -----

def test_strip_dropcap_strikethrough():
    """`~~T~~ he Mughal` → `The Mughal` (pymupdf4llm drop-cap artifact)."""
    raw = "> 1 ~~T~~ he Mughal Emperor granted Diwani in 1765."
    out = strip_dropcap_and_strikethrough(raw)
    assert out == "> 1 The Mughal Emperor granted Diwani in 1765."


def test_strip_multichar_strikethrough_keeps_content():
    """`~~SOURCES AND APPROACHES~~` (font-style false-positive) → keep
    the text content, drop the markup tokens."""
    raw = "~~**UNIT 1**~~\n\n~~The system of Budget was introduced in 1860.~~"
    out = strip_dropcap_and_strikethrough(raw)
    assert "UNIT 1" in out
    assert "system of Budget was introduced in 1860" in out
    assert "~~" not in out


def test_normalize_repairs_fffd_in_section_numbers():
    """U+FFFD decode-failure on '.' between digits → repaired."""
    assert normalize_unicode("Section 3�3�1 covers") == "Section 3.3.1 covers"
    assert normalize_unicode("4�10� Habitat loss") == "4.10. Habitat loss"


def test_normalize_repairs_fffd_after_letter():
    """U+FFFD after a single letter (list prefix) → period."""
    assert normalize_unicode("S� No. of species") == "S. No. of species"


def test_normalize_drops_isolated_fffd():
    """Isolated fffd (not between digits, not after letter) → dropped."""
    assert normalize_unicode("Plain text � extra") == "Plain text  extra"


def test_strip_dropcap_preserves_numbers_and_dates():
    """Drop-cap stripping must NOT touch numbers, years, or units —
    UPSC factual signal stays intact."""
    raw = "~~I~~ n 1947, India gained independence. GDP grew 7.5% to ₹2.4 lakh cr."
    out = strip_dropcap_and_strikethrough(raw)
    assert "In 1947, India gained independence" in out
    assert "7.5%" in out and "₹2.4 lakh cr" in out


@pytest.mark.parametrize("bad,good", [
    # Explicit dictionary fixes (irregular drops)
    ("Artcles 14 and 21",           "Articles 14 and 21"),
    ("the fnal decision",           "the final decision"),
    ("Defniton of state",           "Definition of state"),
    # Broad regex fixes — *aton suffix
    ("the Formaton of new states",  "the Formation of new states"),
    ("alteraton of boundaries",     "alteration of boundaries"),
    ("reservaton policy",           "reservation policy"),
    ("jurisdicton of the court",    "jurisdiction of the court"),
    ("educaton minister",           "education minister"),
    ("derogaton of fundamental rights", "derogation of fundamental rights"),
    ("recommendaton of the panel",  "recommendation of the panel"),
    # Broad regex fixes — *iton suffix
    ("Prohibiton of discrimination", "Prohibition of discrimination"),
    ("Aboliton of titles",          "Abolition of titles"),
    ("the conditon was met",        "the condition was met"),
    # Broad regex fixes — *cton suffix (jurisdiction, election, protection)
    ("jurisdicton of the court",    "jurisdiction of the court"),
    ("electon results announced",   "election results announced"),
    ("protecton of minorities",     "protection of minorities"),
    # Broad regex fixes — *tng suffix
    ("existng laws",                "existing laws"),
    ("relatng to commerce",         "relating to commerce"),
    ("actng President",             "acting President"),
])
def test_fix_idrop_typos(bad, good):
    """'fi'-ligature drop fix: dictionary + broad regex restore 'i' in
    observed PDF extraction errors."""
    assert fix_idrop_typos(bad) == good


def test_fix_idrop_does_not_touch_legitimate_words():
    """Words NOT in the typo pattern stay untouched — no false positives
    on real vocabulary like 'canton'/'proton'/'photon'/'newton'/'plankton'
    (which end in n/oton, not 'aton'), 'cation' (chemistry — ends in
    'tion' already), or 'automaton' (excluded explicitly)."""
    text = ("The canton of Geneva. Newton's third law. Photon emission. "
            "Proton mass. Plankton blooms. Automaton theory. "
            "Cation exchange. Triton is a moon.")
    assert fix_idrop_typos(text) == text


def test_fix_idrop_preserves_case():
    """Capitalised forms (sentence-initial, headings) are preserved."""
    assert fix_idrop_typos("Formaton of new states") == "Formation of new states"
    assert fix_idrop_typos("Reservaton policy")     == "Reservation policy"
    assert fix_idrop_typos("Prohibiton of titles")  == "Prohibition of titles"


def test_clean_file_strips_strikethrough(clean_tmp):
    """End-to-end: drop-cap + strikethrough artifacts removed by clean_file."""
    text = ("> 1 ~~T~~ he Mughal Emperor, Shah Alam, granted Diwani.\n"
            "~~**Chapter Heading**~~\n"
            "Body content survives.\n\n" + _FILLER)
    src, dst = _write(clean_tmp, text)
    res = clean_file(src, dst)
    assert not res.dropped
    out = dst.read_text(encoding="utf-8")
    assert "~~" not in out
    assert "The Mughal Emperor" in out
    assert "Chapter Heading" in out
    assert "Body content survives" in out


def test_bilingual_record_file_keeps_english_drops_hindi(clean_tmp):
    """Record-delimited DB extracts are bilingual; line-level filtering
    must keep EN content and drop HI lines instead of dropping the whole
    file (the bug that silently lost both PYQ tables)."""
    en = "Which Article of the Constitution deals with the Right to Equality before law?"
    hi = "संविधान का कौन सा अनुच्छेद कानून के समक्ष समानता के अधिकार से संबंधित है?"
    rec = f"{en}\n{hi}\n\n{END_RECORD_DELIM}\n\n"
    src, dst = _write(clean_tmp, (rec * 20))
    res = clean_file(src, dst)
    assert not res.dropped, f"bilingual record file dropped: {res.drop_reason}"
    out = dst.read_text(encoding="utf-8")
    assert "Right to Equality" in out
    assert "संविधान" not in out
    assert END_RECORD_DELIM in out
