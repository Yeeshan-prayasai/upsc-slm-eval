"""PDF → Markdown extractor.

Uses **pymupdf4llm** as the single extraction backend. Per the May 2026
PDF-extractor benchmarks (themenonlab.blog, pdfmux.com, unstract.com),
pymupdf4llm is the fastest native-PDF extractor in the open-source
field and the recommended LLM-corpus default. It:

- Detects multi-column layouts and emits text in proper reading order
  (the single biggest win over `pdfplumber` / `pdfminer.six` for
  NCERT-style two-column books)
- Emits Markdown — headers as `## ...`, bullets as `- ...`, italics
  preserved — which gives the downstream tokenizer a clean signal for
  document structure
- Auto-runs Tesseract on pages where embedded text isn't sufficient
  (figures, stylized chapter headings) without us having to manage a
  fallback path
- Preserves tables and numbers (per project memory
  `feedback_keep_numbers_tables`)

We preserve numbers and tables and apply the FineWeb document-level
floor at clean.py (`chars < 50 OR lines < 3` drop). This module's only
content filter is per-page alpha-char floor for OCR-fallback efficiency.

Outputs (one per input PDF):
- `data/cpt_text/<source>/<book>/foo.md`         (Markdown body)
- `data/cpt_text/<source>/<book>/foo.meta.json`  (per-doc stats)

CLI:
    python -m training.data.ocr --source ncert
    python -m training.data.ocr --source ncert --book keec1
    python -m training.data.ocr --pdf data/cpt_raw/ncert/keec1/keec101.pdf

References:
- pymupdf4llm: Artifex (PyMuPDF team), pypi.org/project/pymupdf4llm
- 2026 benchmark: pdfmux.com/blog/best-pdf-extraction-library-python
- FineWeb document filter: Penedo et al. 2024 (arXiv 2406.17557 §3.2)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .acquire._base import RepoPaths

REPO = RepoPaths.root()
CPT_RAW = REPO / "data" / "cpt_raw"
CPT_TEXT = REPO / "data" / "cpt_text"


@dataclass
class FileResult:
    pdf: str             # relative to repo
    md: str              # relative to repo
    in_bytes: int
    out_chars: int
    n_pages: int
    cached: bool


def extract_pdf(pdf_path: Path, force: bool = False) -> FileResult:
    """Extract one PDF to a `.md` + `.meta.json` pair.

    Idempotent: skips if `.md` is newer than `.pdf` (unless --force).
    """
    rel = pdf_path.relative_to(REPO)
    out_dir = CPT_TEXT / rel.with_suffix("").relative_to("data/cpt_raw").parent
    out_md = out_dir / (pdf_path.stem + ".md")
    out_meta = out_dir / (pdf_path.stem + ".meta.json")

    if (not force and out_md.exists() and out_meta.exists()
            and out_md.stat().st_mtime >= pdf_path.stat().st_mtime):
        meta = json.loads(out_meta.read_text(encoding="utf-8"))
        return FileResult(
            pdf=str(rel), md=str(out_md.relative_to(REPO)),
            in_bytes=pdf_path.stat().st_size,
            out_chars=int(meta.get("out_chars", 0)),
            n_pages=int(meta.get("n_pages", 0)),
            cached=True,
        )

    import pymupdf4llm
    import pymupdf

    out_dir.mkdir(parents=True, exist_ok=True)

    # pymupdf4llm.to_markdown is chatty; redirect its prints so the
    # caller's progress output stays parseable. Errors still surface
    # via exceptions.
    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        md_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            show_progress=False,
            # `image_size_limit=0.0` keeps the OCR-on-pictures behavior
            # we saw improve title-page extraction.
        )

    with pymupdf.open(str(pdf_path)) as doc:
        n_pages = doc.page_count

    # FFFD-ligature rescue. For some embedded fonts, pymupdf4llm's markdown
    # layer emits U+FFFD for ligature glyphs it can't map (e.g. `ti` →
    # "Na�onal", `ft` → "Shi�"), while pymupdf's plain `get_text` decodes
    # the same font cleanly. When the markdown is FFFD-dense, the structure
    # isn't worth the corrupted text — fall back to plain text extraction
    # (verified to produce 0 FFFD on the affected docs). The threshold is
    # high enough that the occasional genuine replacement char (decode of a
    # truly-missing glyph) doesn't trigger a needless fallback.
    extractor = "pymupdf4llm"
    fffd = md_text.count("�")
    fffd_per_kchar = fffd / max(1, len(md_text)) * 1000
    if fffd_per_kchar > 1.0:
        with pymupdf.open(str(pdf_path)) as doc:
            plain = "\n\n".join(doc[i].get_text("text") for i in range(doc.page_count))
        plain_fffd = plain.count("�")
        if plain_fffd < fffd:
            print(f"    [fffd-rescue] {pdf_path.name}: markdown had {fffd} U+FFFD "
                  f"({fffd_per_kchar:.1f}/Kchar) → plain-text re-extract has "
                  f"{plain_fffd}; using plain text")
            md_text = plain
            extractor = "pymupdf-plaintext (fffd-ligature rescue)"

    out_md.write_text(md_text, encoding="utf-8")
    meta = {
        "pdf": str(rel),
        "md": str(out_md.relative_to(REPO)),
        "n_pages": n_pages,
        "out_chars": len(md_text),
        "extractor": extractor,
        "extractor_messages": stdout_capture.getvalue(),
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return FileResult(
        pdf=str(rel), md=str(out_md.relative_to(REPO)),
        in_bytes=pdf_path.stat().st_size,
        out_chars=len(md_text),
        n_pages=n_pages,
        cached=False,
    )


def _gather_pdfs(source: str | None, book: str | None, pdf: str | None) -> list[Path]:
    if pdf:
        p = Path(pdf)
        if not p.is_absolute():
            p = REPO / p
        return [p]
    if not source:
        return []
    root = CPT_RAW / source
    if book:
        root = root / book
    return sorted(root.rglob("*.pdf"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract PDFs to Markdown via pymupdf4llm.")
    p.add_argument("--source", help="Source dir under data/cpt_raw (e.g. ncert)")
    p.add_argument("--book", help="Limit to one book code (e.g. keec1)")
    p.add_argument("--pdf", help="Process a single PDF (relative to repo)")
    p.add_argument("--workers", type=int, default=4,
                   help="Process pool size (default 4)")
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if .md is newer than .pdf")
    args = p.parse_args(argv)

    pdfs = _gather_pdfs(args.source, args.book, args.pdf)
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        return 1
    print(f"Extracting {len(pdfs)} PDFs via pymupdf4llm "
          f"({args.workers} workers)...")

    n_done = 0
    n_cached = 0
    n_chars = 0
    n_pages = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_pdf, pdf, args.force): pdf for pdf in pdfs}
        for fut in as_completed(futures):
            pdf = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"  ERROR {pdf.name}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            n_done += 1
            n_chars += r.out_chars
            n_pages += r.n_pages
            if r.cached:
                n_cached += 1
            tag = "[CACHED]" if r.cached else "        "
            print(f"  [{n_done:3d}/{len(pdfs)}] {tag} {pdf.name:30s}  "
                  f"pages={r.n_pages:3d}  chars={r.out_chars:>8,}")

    print(f"\nTotal: {n_done} PDFs ({n_cached} cached), "
          f"{n_pages} pages, {n_chars/1e6:.2f} M chars extracted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
