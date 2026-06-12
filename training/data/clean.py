"""Text normalization + deduplication for the CPT corpus.

Per-document pipeline:
- NFKC unicode normalization + a minimal punctuation-replacement table
- Collapse whitespace runs to one space; preserve paragraph breaks
- **Language filter**: drop documents whose alphabetic content is
  majority-Devanagari (the corpus is English-only per methodology §4.5;
  scraped sources like The Hindu include Hindi-language articles)
- **Document floor** (Penedo et al. 2024 / Gopher rules): drop documents
  with fewer than **50 words**. (Word-based, not line-based — a
  line-count floor wrongly drops complete single-paragraph news
  articles, measured at ~5.5% of the Hindu scrape.)

Corpus-level pipeline:
- Exact-duplicate removal via SHA-256 of normalized paragraph text
  (kills repeated headers/footers/boilerplate across pages)
- Near-duplicate removal at **document level** via MinHash LSH at
  Jaccard threshold **0.80** with **128 permutations** and **5-gram
  word shingles**, with candidate verification against the stored
  MinHash (LSH banding alone admits false positives below the
  threshold). Threshold + granularity per Lee et al. 2022
  (arXiv 2107.06499); FineWeb (Penedo et al. 2024 §3.3) used ~0.75,
  also at document level. Paragraph-level near-dup at 0.70 (the
  previous configuration) was strictly more aggressive than either
  and risked deleting topically-similar-but-distinct UPSC coverage.
- Files are processed in **source-priority order** (NCERT → reference
  books → government primary → ecosystem scrapes) so that when two
  sources carry near-identical content, the higher-quality source's
  copy is the one kept.
- Replay-buffer `.jsonl` sources (FineWeb-Edu sample, Wikipedia) are
  cleaned + floored per row but skip the near-dup pass — both corpora
  are already deduplicated upstream by their curators.

What this module deliberately does NOT do:
- No header/footer detection (FineWeb doesn't; dedup handles it)
- No hyphenated-line-break reconnection (not in standard pipelines;
  tokenizer handles split tokens via BPE)
- No number-density / tabular filtering (per project memory
  `feedback_keep_numbers_tables` — UPSC factual signal)

CLI:
    python -m training.data.clean --in data/cpt_text --out data/cpt_clean
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .acquire._base import RepoPaths

REPO = RepoPaths.root()
CPT_RAW = REPO / "data" / "cpt_raw"
CPT_TEXT = REPO / "data" / "cpt_text"
CPT_CLEAN = REPO / "data" / "cpt_clean"
CPT_CLEAN_DEDUP = REPO / "data" / "cpt_clean_dedup"

# Document-level floor (Penedo et al. 2024 / Gopher rules — word-based).
MIN_WORDS_PER_DOC = 50

# English-only corpus (methodology §4.5): drop documents whose alphabetic
# content is majority-Devanagari. Threshold is deliberately high (30%) so
# English docs quoting Hindi terms/names survive.
MAX_DEVANAGARI_RATIO = 0.30
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

# Record delimiter used by the local-DB extracts (one .txt per table,
# thousands of independent records per file). Preserved through cleaning
# and dedup so tokenize_pack can split records into separate documents.
END_RECORD_DELIM = "<<<END-RECORD>>>"

# Process order for the corpus-level dedup pass: when two sources carry
# near-identical content, the FIRST-processed source's copy is kept —
# so order = source quality for UPSC factual yield. Unlisted sources
# sort after listed ones, alphabetically.
SOURCE_PRIORITY = [
    "ncert", "reference_books", "constitution", "dm_core", "tn_history",
    "economic_survey", "arc2", "niti", "mha_annual_reports",
    "moef", "ndma", "ipcc", "budget",
    "rbi", "nfhs_census",
    "qa_bank", "rc_qa", "pib", "isro", "drdo", "prs",
    "pmf_ias", "mrunal", "mea", "newspapers", "dte", "orf", "idsa",
    "local_db", "instruct", "wikipedia", "slimpajama",
]


def source_priority_key(rel_path: Path) -> tuple[int, str]:
    """Sort key: (priority index of top-level source dir, path)."""
    source = rel_path.parts[0] if rel_path.parts else ""
    try:
        idx = SOURCE_PRIORITY.index(source)
    except ValueError:
        idx = len(SOURCE_PRIORITY)
    return (idx, str(rel_path))

# Source-specific cruft filters. Right now only Mrunal needs this — the
# scrape pulled in ~46 non-UPSC posts (blog SEO, ebook download promos,
# WordPress theme reviews, notice-board exam announcements). We drop
# by URL category tag, which Mrunal embeds in the body as
# `Categories: [advertizement]`, `[download]`, `[noticeboard]`, `[top-10]`.
# These categories are Mrunal's own non-UPSC labels — by his own
# taxonomy, this content isn't subject-matter material.
MRUNAL_CRUFT_CATEGORIES = (
    "advertizement",   # Mrunal's spelling (sic)
    "advertisement",
    "download",
    "noticeboard",
    "notice-board",
    "top-10",
)
_MRUNAL_CAT_RE = re.compile(
    r"Categories:\s*(?:\[[^\]]+\]\([^)]+/category/("
    + "|".join(MRUNAL_CRUFT_CATEGORIES)
    + r")\)|"
    + r"\[("
    + "|".join(MRUNAL_CRUFT_CATEGORIES)
    + r")\])",
    re.I,
)

# MinHash LSH near-dup detection — DOCUMENT level, candidates verified
# against the stored MinHash. Threshold per Lee et al. 2022 (0.8);
# FineWeb used ~0.75 doc-level.
LSH_THRESHOLD = 0.80
LSH_NUM_PERM = 128
SHINGLE_SIZE = 5


# ----------- Unicode normalization -----------

UNICODE_REPLACE = {
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "-",
    "…": "...",
    " ": " ",   # non-breaking space → regular space
    "​": "",     # zero-width space
    "﻿": "",     # BOM
}


_FFFD_DIGIT_RE  = re.compile(r"(?<=\d)�")
_FFFD_LETTER_RE = re.compile(r"(?<=[A-Za-z])�")


def normalize_unicode(text: str) -> str:
    """NFKC + targeted punctuation replacements. Preserves all
    alphanumeric content including numbers and units.

    Also repairs the U+FFFD (Unicode replacement) artifact pattern
    observed in reference-book PDFs (e.g. Shankar IAS Environment):
    decode failure on the period character inside section numbers
    ("3.3.1" → "3�3�1") and list prefixes ("S. No" → "S� No").
    Targeted rules; any remaining isolated fffd gets dropped."""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in UNICODE_REPLACE.items():
        text = text.replace(src, dst)
    # Repair fffd patterns. Lookbehind/lookahead so consecutive fffd
    # (e.g. "3�3�1") all match in one sub call (non-overlapping `(\d)�(\d)`
    # would consume the middle digit and miss the second fffd).
    text = _FFFD_DIGIT_RE.sub(".", text)   # 3�3 → 3.3 and 10� → 10.
    text = _FFFD_LETTER_RE.sub(".", text)  # S� → S.
    text = text.replace("�", "")       # drop stragglers
    return text


# pymupdf4llm misinterprets paragraph-leading drop-capital letters as
# strikethrough — emitting `~~T~~ he Mughal Emperor...` where the source
# rendered a large drop-cap "T" followed by "he ...". It also wraps some
# bold/italic table-cell content and font-style runs in `~~...~~`. None
# of these books contain intentional strikethrough, so we strip the
# markers and keep the inner content.
_DROPCAP_RE = re.compile(r"~~([A-Z])~~\s+")
_STRIKETHRU_RE = re.compile(r"~~([^~\n]+)~~")


def strip_dropcap_and_strikethrough(text: str) -> str:
    """Remove pymupdf4llm-spurious `~~...~~` markup while keeping the
    actual text content. Applied to all sources, not just reference
    books — universal regex, no false positives on intentional
    Markdown strikethrough (none of our corpus uses it).

    Also strips unpaired `~~` tokens which can appear as OCR garbage
    from Tesseract on scanned pages (G.C. Leong Geography in our
    corpus has these in figure-caption regions)."""
    # Drop-caps: `~~T~~ he Mughal` → `The Mughal` (collapse the space too)
    text = _DROPCAP_RE.sub(r"\1", text)
    # Any other strikethrough span: keep the content, lose the markers
    text = _STRIKETHRU_RE.sub(r"\1", text)
    # Unpaired stragglers (Tesseract OCR noise on scanned books) — just delete
    text = text.replace("~~", "")
    return text


# 'fi'-ligature / italic-i font-decoding bug observed in Laxmikanth +
# Spectrum reference-book PDFs. The bug drops 'i' between consonants
# (mostly in *tion / *ting / *fi* patterns), producing systematic typos:
#   election → electon       prohibition → prohibiton
#   creating → creatng        final       → fnal
#   Articles → Artcles        definition  → defniton
#
# We fix in two passes:
#   (1) Explicit dictionary for irregular drops we can't generalize
#       (Artcles, fnal, Defniton, afrmaton, etc.)
#   (2) Broad regex for *aton/*iton/*tng suffixes with a tight
#       exclusion list — these patterns have essentially no legitimate
#       English vocabulary collisions at {3+}-char prefix length.
#
# Exclusions: `automaton` (real word ending in 'aton'). `canton` and
# `proton`/`photon`/`triton` already don't match because they end in
# 'nton'/'oton' not 'aton'.
_IDROP_EXPLICIT = {
    "Artcles":   "Articles",   "artcles":   "articles",
    "fnal":      "final",      "Fnal":      "Final",
    "Defniton":  "Definition", "defniton":  "definition",
    "afrmaton":  "affirmation", "Afrmaton":  "Affirmation",
}
# Real-word collisions caught by the broad regex — exclude case-insensitively.
_IDROP_EXCLUDE = {"automaton", "triton", "briton"}
_IDROP_EXPLICIT_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _IDROP_EXPLICIT) + r")\b"
)
# Broad pattern: any word ending in aton/iton/cton/tng with ≥2-char prefix.
# Captures the prefix so we can rebuild as prefix+'ation'/'ition'/'ction'/'ting'.
# Real-word collision check (manually verified at {2,} prefix):
#   aton: baton/Eaton/Caton/Aton — all have <2 chars before suffix → safe.
#         automaton — 5-char prefix, CAUGHT → explicitly excluded below.
#   iton: triton/Britton — <2 chars or capitalised proper noun → safe.
#   cton: no common English words end in 'cton'.
#   tng:  no common English words end in 'tng'.
_IDROP_BROAD_RE = re.compile(r"\b([A-Za-z]{2,})(aton|iton|cton|tng)\b")
_IDROP_SUFFIX = {"aton": "ation", "iton": "ition", "cton": "ction", "tng": "ting"}


def _idrop_broad_sub(m: "re.Match[str]") -> str:
    full = m.group(0)
    if full.lower() in _IDROP_EXCLUDE:
        return full
    prefix = m.group(1)
    return prefix + _IDROP_SUFFIX[m.group(2)]


def fix_idrop_typos(text: str) -> str:
    """Restore dropped 'i' from the PDF font-decoding bug observed in
    reference-book extractions. Two-pass: explicit dictionary for
    irregular forms, then broad regex on *aton/*iton/*tng suffixes
    with `automaton` excluded."""
    text = _IDROP_EXPLICIT_RE.sub(lambda m: _IDROP_EXPLICIT[m.group(1)], text)
    text = _IDROP_BROAD_RE.sub(_idrop_broad_sub, text)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse intra-line whitespace runs to one space; preserve
    paragraph breaks (≥ 2 newlines → exactly 2)."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ----------- File-level cleaning -----------

@dataclass
class CleanResult:
    src_path: str
    dst_path: str | None  # None if document was dropped
    in_chars: int
    out_chars: int
    dropped: bool = False
    drop_reason: str = ""


def _drop_reason(text: str) -> str:
    """Document-level drop decision after normalization. Returns the
    reason string, or '' to keep. Word floor per Penedo 2024 / Gopher;
    Devanagari-majority docs excluded per the English-only corpus rule."""
    word_count = len(text.split())
    if word_count < MIN_WORDS_PER_DOC:
        return f"words={word_count} below floor ({MIN_WORDS_PER_DOC})"
    alpha = [c for c in text if c.isalpha()]
    if alpha:
        deva = len(_DEVANAGARI_RE.findall(text))
        ratio = deva / len(alpha)
        if ratio > MAX_DEVANAGARI_RATIO:
            return f"devanagari ratio {ratio:.2f} > {MAX_DEVANAGARI_RATIO} (non-English doc)"
    return ""


def clean_jsonl_file(src: Path, dst: Path) -> CleanResult:
    """Clean a replay-buffer .jsonl file (FineWeb-Edu sample, Wikipedia
    subset): normalize + floor each row's `text` field, drop rows that
    fail, write the surviving rows as cleaned JSONL. Manifest .jsonl
    files (acquirer provenance, no `text` field) are skipped entirely."""
    import json as _json

    rows_in = rows_kept = 0
    in_chars = out_chars = 0
    kept_lines: list[str] = []
    with src.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                d = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            t = (d.get("text") or "").strip()
            if not t:
                continue   # provenance manifests etc. — not corpus rows
            rows_in += 1
            in_chars += len(t)
            t = collapse_whitespace(normalize_unicode(t))
            if _drop_reason(t):
                continue
            rows_kept += 1
            out_chars += len(t)
            kept_lines.append(_json.dumps({"text": t}, ensure_ascii=False))

    if not kept_lines:
        return CleanResult(
            src_path=str(src.relative_to(REPO)), dst_path=None,
            in_chars=in_chars, out_chars=0,
            dropped=True, drop_reason=f"no rows survived ({rows_in} in)",
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    return CleanResult(
        src_path=str(src.relative_to(REPO)),
        dst_path=str(dst.relative_to(REPO)),
        in_chars=in_chars, out_chars=out_chars,
    )


def is_mrunal_cruft(raw: str) -> bool:
    """Detect a Mrunal blog-cruft post by its self-applied category tag.

    Mrunal's WordPress export labels each post with category tags like
    `Categories: [advertizement]`, `[download]`, `[noticeboard]`,
    `[top-10]`. These categories are non-UPSC by Mrunal's own taxonomy
    (ebook download promos, WordPress theme reviews, exam-result
    notices, etc.) — keep only the genuinely UPSC-focused category
    posts that constitute the bulk of the corpus."""
    return bool(_MRUNAL_CAT_RE.search(raw))


def clean_file(src: Path, dst: Path) -> CleanResult:
    """Apply unicode + whitespace normalization, then the FineWeb
    document-level filter. Returns a CleanResult; `dst_path=None` if
    the document was dropped (in which case `dst` is not written)."""
    raw = src.read_text(encoding="utf-8", errors="replace")
    in_chars = len(raw)

    # Source-specific filter — drop Mrunal blog cruft BEFORE normalization
    # so we don't waste cycles on docs we'll throw away anyway.
    if "/mrunal/" in str(src) and is_mrunal_cruft(raw):
        return CleanResult(
            src_path=str(src.relative_to(REPO)),
            dst_path=None,
            in_chars=in_chars,
            out_chars=0,
            dropped=True,
            drop_reason="mrunal cruft category (advertizement/download/noticeboard/top-10)",
        )

    text = normalize_unicode(raw)
    text = strip_dropcap_and_strikethrough(text)
    text = fix_idrop_typos(text)
    text = collapse_whitespace(text)

    # Record-delimited DB extracts (one .txt per table) can be BILINGUAL —
    # question tables store EN+HI variants in the same blob, which put the
    # whole file at ~56% Devanagari and made the doc-level language filter
    # silently drop 40M chars of PYQ data. For these files, filter at LINE
    # level instead: drop Devanagari-majority lines (the HI duplicates),
    # keep the English content, and skip the doc-level check.
    if END_RECORD_DELIM in text:
        # EN and HI variants share lines in these blobs, so strip
        # Devanagari RUNS (plus the danda/space padding around them)
        # character-wise instead of dropping lines.
        text = re.sub(r"[\u0900-\u097F][\u0900-\u097F\s\u0964\u0965]*", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        word_count = len(text.split())
        reason = (f"words={word_count} below floor ({MIN_WORDS_PER_DOC})"
                  if word_count < MIN_WORDS_PER_DOC else "")
    else:
        reason = _drop_reason(text)
    if reason:
        return CleanResult(
            src_path=str(src.relative_to(REPO)),
            dst_path=None,
            in_chars=in_chars,
            out_chars=0,
            dropped=True,
            drop_reason=reason,
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text + "\n", encoding="utf-8")
    return CleanResult(
        src_path=str(src.relative_to(REPO)),
        dst_path=str(dst.relative_to(REPO)),
        in_chars=in_chars,
        out_chars=len(text),
    )


# ----------- Corpus-level deduplication -----------

@dataclass
class DedupeReport:
    files_seen: int = 0
    exact_dups_removed: int = 0       # paragraphs (headers/footers etc.)
    near_dup_docs_removed: int = 0    # whole documents
    paragraphs_in: int = 0
    paragraphs_out: int = 0
    jsonl_passthrough: int = 0        # replay files copied unmodified


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]


def _shingles(text: str, n: int = SHINGLE_SIZE) -> set[str]:
    """Word n-gram shingles for MinHash; lowercased + punctuation-stripped."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def dedupe_corpus(
    files: list[Path],
    in_root: Path,
    out_root: Path,
    threshold: float = LSH_THRESHOLD,
    num_perm: int = LSH_NUM_PERM,
) -> DedupeReport:
    """Two-level dedup over cleaned files, writing survivors to
    `out_root` (structure preserved relative to `in_root`):

    1. **Paragraph-level exact** dedup (SHA-256 of normalized text) —
       removes repeated headers/footers/boilerplate across pages.
       `<<<END-RECORD>>>` delimiter paragraphs are structural, never
       deduped (tokenize_pack splits records on them).
    2. **Document-level near-dup** via MinHash LSH at `threshold`, with
       every LSH candidate VERIFIED against its stored MinHash —
       banding alone admits below-threshold false positives (the
       estimated-Jaccard check makes the threshold real). A near-dup
       document is dropped whole; the first-processed copy wins, so
       callers should order `files` by source priority.

    `.jsonl` replay files are copied through untouched (FineWeb-Edu /
    Wikipedia are deduplicated upstream by their curators; row-level
    cleaning already happened in the clean pass).

    Files MUST live under `in_root` for relative-path computation.
    """
    from datasketch import MinHash, MinHashLSH

    rep = DedupeReport()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    stored: dict[str, "MinHash"] = {}
    seen_hashes: set[str] = set()

    for f in files:
        rep.files_seen += 1
        rel = f.relative_to(in_root)
        dst = out_root / rel

        if f.suffix == ".jsonl":
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(f.read_bytes())
            rep.jsonl_passthrough += 1
            continue

        text = f.read_text(encoding="utf-8", errors="replace")
        paragraphs = _paragraphs(text)
        rep.paragraphs_in += len(paragraphs)

        kept: list[str] = []
        for para in paragraphs:
            if para == END_RECORD_DELIM:
                kept.append(para)   # structural, never deduped
                continue
            h = hashlib.sha256(re.sub(r"\s+", " ", para).lower().encode()).hexdigest()
            if h in seen_hashes:
                rep.exact_dups_removed += 1
                continue
            seen_hashes.add(h)
            kept.append(para)

        if not kept:
            continue

        doc_text = "\n\n".join(kept)
        shingles = _shingles(doc_text)
        if len(shingles) >= 5:
            m = MinHash(num_perm=num_perm)
            for s in shingles:
                m.update(s.encode())
            # Verify candidates: LSH banding has false positives; only a
            # confirmed estimated-Jaccard >= threshold counts as a dup.
            is_dup = any(
                m.jaccard(stored[c]) >= threshold for c in lsh.query(m)
            )
            if is_dup:
                rep.near_dup_docs_removed += 1
                continue
            key = str(rel)
            lsh.insert(key, m)
            stored[key] = m

        rep.paragraphs_out += len(kept)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(doc_text + "\n", encoding="utf-8")

    return rep


# ----------- CLI -----------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Clean + dedupe extracted text for the CPT corpus "
                    "(per FineWeb / Lee 2022)."
    )
    p.add_argument("--in", dest="src", default=str(CPT_TEXT),
                   help="Input dir (default data/cpt_text)")
    p.add_argument("--out", dest="dst", default=str(CPT_CLEAN),
                   help="Cleaned-output dir (default data/cpt_clean)")
    p.add_argument("--dedup-out", default=str(CPT_CLEAN_DEDUP),
                   help="Deduplicated-output dir (default data/cpt_clean_dedup)")
    p.add_argument("--source", help="Limit to one source dir (e.g. ncert)")
    p.add_argument("--no-dedupe", action="store_true",
                   help="Skip the cross-file dedupe pass (clean only)")
    args = p.parse_args(argv)

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dedup_root = Path(args.dedup_out)
    if args.source:
        # For a single source, fall through to cpt_raw/<source> if
        # cpt_text/<source> doesn't exist — that's the case for
        # already-text sources (mrunal/orf/prs/newspapers/pmf_ias scrape)
        # which skip OCR and live directly under cpt_raw/.
        candidate_text = src_root / args.source
        candidate_raw = CPT_RAW / args.source
        if not candidate_text.exists() and candidate_raw.exists():
            print(f"  (cpt_text/{args.source}/ missing — using cpt_raw/{args.source}/)")
            src_root = candidate_raw
        else:
            src_root = candidate_text
        dst_root = dst_root / args.source
        dedup_root = dedup_root / args.source

    # Accept .txt (local DB extracts), .md (pymupdf4llm output), AND
    # .jsonl (replay buffer: FineWeb-Edu sample + Wikipedia subset —
    # previously these never entered the cleaned corpus at all, which
    # silently zeroed the replay share).
    def _gather(root: Path) -> list[Path]:
        out = (list(root.rglob("*.txt")) + list(root.rglob("*.md"))
               + [p for p in root.rglob("*.jsonl")
                  if p.name != "manifest.jsonl"])   # acquirer provenance, not corpus
        return sorted(out)

    txts = _gather(src_root)

    # Text-format sources (mrunal/orf/prs/newspapers/pmf_ias scrape,
    # slimpajama/wikipedia jsonl) live directly under data/cpt_raw/<source>/
    # and never went through OCR (since they're already text). Pull them
    # into the same cleaning pass so they don't silently bypass the
    # document floor + language + Mrunal cruft filters.
    if str(src_root).startswith(str(CPT_TEXT)) and not args.source:
        for src_dir in sorted(p for p in CPT_RAW.iterdir() if p.is_dir()):
            if (CPT_TEXT / src_dir.name).exists():
                continue  # PDF-extracted source — already covered above
            extra = _gather(src_dir)
            if extra:
                txts.extend(extra)
                print(f"  (+ {len(extra)} text files from cpt_raw/{src_dir.name}/)")
    if not txts:
        print(f"No .txt/.md/.jsonl files under {src_root}", file=sys.stderr)
        return 1
    print(f"Cleaning {len(txts)} files: {src_root} → {dst_root}")

    cleaned_paths: list[Path] = []
    total_in = total_out = 0
    docs_dropped = 0
    for src in txts:
        # `src` may live under cpt_text/ (PDF-extracted) OR cpt_raw/
        # (text-only source). Pick whichever root contains it so the
        # output mirrors the source's relative layout.
        if src.is_relative_to(src_root):
            rel = src.relative_to(src_root)
        elif src.is_relative_to(CPT_RAW):
            rel = src.relative_to(CPT_RAW)
        else:
            rel = Path(src.name)
        dst = dst_root / rel
        if src.suffix == ".jsonl":
            result = clean_jsonl_file(src, dst)
        else:
            result = clean_file(src, dst)
        total_in += result.in_chars
        total_out += result.out_chars
        if result.dropped:
            docs_dropped += 1
        else:
            cleaned_paths.append(dst)
    print(f"Clean pass: {total_in/1e6:.1f} M chars in → {total_out/1e6:.1f} M out  "
          f"(docs: {len(txts)} in, {len(cleaned_paths)} kept, {docs_dropped} dropped "
          f"by floor/language filters)")

    if not args.no_dedupe and cleaned_paths:
        # Source-priority order: when near-identical content exists in two
        # sources, the higher-priority source's copy survives.
        cleaned_paths.sort(key=lambda p: source_priority_key(p.relative_to(dst_root)))
        rep = dedupe_corpus(cleaned_paths, in_root=dst_root, out_root=dedup_root)
        print(f"Dedupe (doc-level LSH t={LSH_THRESHOLD}, verified, perms={LSH_NUM_PERM}, "
              f"shingles={SHINGLE_SIZE}-gram): "
              f"{rep.paragraphs_in:,} paragraphs in → {rep.paragraphs_out:,} out  "
              f"(exact paras={rep.exact_dups_removed:,}, "
              f"near-dup docs={rep.near_dup_docs_removed:,}, "
              f"jsonl passthrough={rep.jsonl_passthrough})")
        print(f"  written to: {dedup_root.relative_to(REPO)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
