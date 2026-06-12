"""NCERT textbook PDF acquirer.

Targets the books explicitly named in `UPSC_Source_List.docx` (GS1-GS3).
NCERT's URL pattern is `https://ncert.nic.in/textbook/pdf/<code><suffix>.pdf`
where `<code>` is the 4-5 char book ID (e.g. `keec1`, `lehs2`) and
`<suffix>` is `ps` (preliminary), `01..NN` (chapters), or `an`
(appendix). We don't hit the listing pages — chapter PDFs are
sequentially numbered, and probing past the last existing chapter is
cheap (HEAD 404 per book ends the scan).

Book codes were validated against NCERT's live site 2026-06-05 via HEAD
probes. The codes here all returned 200 on `<code>ps.pdf`.

CLI:
    python -m training.data.acquire.ncert            # all books
    python -m training.data.acquire.ncert --only kegy1 kegy2
    python -m training.data.acquire.ncert --dry-run  # print URLs, no fetch
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import requests

from ._base import HttpClient, Manifest, ManifestEntry, RepoPaths, now_iso

NCERT_BASE = "https://ncert.nic.in/textbook/pdf"


@dataclass(frozen=True)
class Book:
    """One NCERT textbook to acquire.

    `code` is the 4-5 character book identifier in the PDF URL
    (e.g. `kegy1dd` for Class 11 Fundamentals of Physical Geography).
    `max_chapters` is an upper bound — we stop probing on the first
    404. Set conservatively per book based on known structure; 24 is
    safe (no NCERT textbook exceeds 22 chapters).
    """
    code: str
    title: str
    class_level: str
    subject: str
    max_chapters: int = 24
    has_preliminary: bool = True       # "ps" preliminary file (TOC, preface)
    has_appendix: bool = False         # "an" appendix file


# Books selected from UPSC_Source_List.docx. Codes verified 2026-06-05
# via HEAD-probes against https://ncert.nic.in/textbook/pdf/<code>ps.pdf.
# First letter = class (f=6, g=7, h=8, i=9, j=10, k=11, l=12). Mid chars
# = subject (gy=Geography, ps=Pol. Sci., ec=Economy, sy=Sociology, etc.).
NCERT_BOOKS: list[Book] = [
    # ----- GS1 History -----
    # Class 12: Themes in Indian History (3 parts) — note 'lehs' not 'lhss'
    Book("lehs1", "Themes in Indian History Part 1", "12", "History", 16),
    Book("lehs2", "Themes in Indian History Part 2", "12", "History", 16),
    Book("lehs3", "Themes in Indian History Part 3", "12", "History", 16),
    # Class 11: Themes in World History — 'kehs1' (single book, not Pt3)
    Book("kehs1", "Themes in World History", "11", "History", 14),
    # Class 8 History (supplementary for medieval/modern coverage)
    Book("hess2", "Our Pasts III (Class 8 History)", "8", "History", 12),

    # ----- GS1 Art & Culture -----
    # Class 11: An Introduction to Indian Art
    Book("kefa1", "An Introduction to Indian Art", "11", "Art & Culture", 10),
    Book("kefa2", "An Introduction to Indian Art Part 2", "11", "Art & Culture", 10),

    # ----- GS1 Geography -----
    # Class 11
    Book("kegy1", "Fundamentals of Physical Geography", "11", "Geography", 14),
    Book("kegy2", "India: Physical Environment", "11", "Geography", 10),
    Book("kegy3", "Practical Work in Geography Part 1", "11", "Geography", 10),
    # Class 12
    Book("legy1", "Fundamentals of Human Geography", "12", "Geography", 12),
    Book("legy2", "India: People and Economy", "12", "Geography", 12),
    Book("legy3", "Practical Work in Geography Part 2", "12", "Geography", 10),

    # ----- GS1 Society / Sociology -----
    # Class 11
    Book("kesy1", "Introducing Sociology", "11", "Sociology", 10),
    Book("kesy2", "Understanding Society", "11", "Sociology", 10),
    # Class 12
    Book("lesy1", "Indian Society", "12", "Sociology", 8),
    Book("lesy2", "Social Change and Development in India", "12", "Sociology", 8),

    # ----- GS2 Polity / Political Science -----
    # Class 11
    Book("keps1", "Indian Constitution at Work", "11", "Polity", 12),
    Book("keps2", "Political Theory", "11", "Polity", 10),
    # Class 12
    Book("leps1", "Contemporary World Politics", "12", "Polity", 10),
    Book("leps2", "Politics in India Since Independence", "12", "Polity", 10),

    # ----- GS3 Economy -----
    # Class 11
    Book("keec1", "Indian Economic Development", "11", "Economy", 12),
    Book("kest1", "Statistics for Economics", "11", "Economy", 10),
    # Class 12
    Book("leec1", "Introductory Macroeconomics", "12", "Economy", 8),
    Book("leec2", "Introductory Microeconomics", "12", "Economy", 8),

    # ----- GS3 Environment / Biology (Ecology + biotech/health chapters) -----
    Book("kebo1", "Biology (Class 11)", "11", "Biology", 22),
    Book("lebo1", "Biology (Class 12)", "12", "Biology", 16),

    # ----- Class 6-10 Science (foundational for S&T) -----
    Book("fesc1", "Science", "6", "Science", 16),
    Book("gesc1", "Science", "7", "Science", 18),
    Book("hesc1", "Science", "8", "Science", 18),
    Book("iesc1", "Science", "9", "Science", 14),
    Book("jesc1", "Science", "10", "Science", 16),

    # ----- Class 11 Psychology (Ethics — selective use) -----
    Book("kepy1", "Introduction to Psychology", "11", "Psychology", 10),
]


def chapter_url(code: str, suffix: str) -> str:
    """Build the chapter PDF URL. `suffix` is e.g. 'ps', '01', '02', 'an'."""
    return f"{NCERT_BASE}/{code}{suffix}.pdf"


def _probe_or_download(
    client: HttpClient,
    url: str,
    dest_dir,
    manifest: Manifest,
    book: Book,
    suffix: str,
    dry_run: bool,
) -> tuple[bool, str]:
    """Returns (was_downloaded, status_string). `was_downloaded=False`
    can mean either 'already cached' (status='cached') or '404' (status='missing')."""
    if manifest.has(url):
        return False, "cached"
    if dry_run:
        return False, "dry-run"
    try:
        # HEAD first — saves bandwidth on 404s, and gives us Content-Length.
        head = client.session.head(url, timeout=client.timeout, allow_redirects=True)
        if head.status_code == 404:
            return False, "missing"
        head.raise_for_status()
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return False, "missing"
        raise
    # Real download
    local = dest_dir / f"{book.code}{suffix}.pdf"
    sha, n = client.download(url, local)
    manifest.add(ManifestEntry(
        url=url,
        local_path=str(local.relative_to(RepoPaths.root())),
        sha256=sha,
        bytes=n,
        title=f"{book.title} — {suffix}",
        fetched_at=now_iso(),
        extra={"book_code": book.code, "class_level": book.class_level,
               "subject": book.subject, "chapter_suffix": suffix},
    ))
    return True, "downloaded"


def acquire_book(client: HttpClient, manifest: Manifest, book: Book, dry_run: bool) -> dict:
    """Download every chapter of one book. Returns per-book summary."""
    dest_dir = RepoPaths.cpt_raw("ncert") / book.code
    dest_dir.mkdir(parents=True, exist_ok=True)

    counts = {"downloaded": 0, "cached": 0, "missing": 0, "dry-run": 0}

    # 1) Preliminary (table of contents, preface)
    if book.has_preliminary:
        url = chapter_url(book.code, "ps")
        _, status = _probe_or_download(client, url, dest_dir, manifest, book, "ps", dry_run)
        counts[status] += 1
        if dry_run:
            print(f"  [dry-run] {url}")

    # 2) Numbered chapters 01..max
    consecutive_missing = 0
    for ch in range(1, book.max_chapters + 1):
        suffix = f"{ch:02d}"
        url = chapter_url(book.code, suffix)
        downloaded, status = _probe_or_download(client, url, dest_dir, manifest, book, suffix, dry_run)
        counts[status] += 1
        if dry_run:
            print(f"  [dry-run] {url}")
        # Two consecutive 404s = we've gone past the last chapter.
        if status == "missing":
            consecutive_missing += 1
            if consecutive_missing >= 2:
                break
        else:
            consecutive_missing = 0

    # 3) Optional appendix
    if book.has_appendix:
        url = chapter_url(book.code, "an")
        _, status = _probe_or_download(client, url, dest_dir, manifest, book, "an", dry_run)
        counts[status] += 1

    return {"code": book.code, "title": book.title, **counts}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download NCERT textbook PDFs for the CPT corpus.")
    p.add_argument("--only", nargs="+", help="Limit to these book codes")
    p.add_argument("--dry-run", action="store_true", help="Print URLs without downloading")
    p.add_argument("--rate", type=float, default=0.5,
                   help="Min seconds between requests to the same host (default 0.5)")
    args = p.parse_args(argv)

    client = HttpClient(rate_seconds=args.rate)
    manifest = Manifest("ncert")
    selected = [b for b in NCERT_BOOKS if (not args.only) or b.code in args.only]
    if args.only:
        unknown = set(args.only) - {b.code for b in NCERT_BOOKS}
        if unknown:
            print(f"WARNING: unknown book codes (skipped): {sorted(unknown)}", file=sys.stderr)
    if not selected:
        print("No books selected.", file=sys.stderr)
        return 1

    print(f"NCERT acquisition — {len(selected)} books, rate={args.rate}s/req, "
          f"dry_run={args.dry_run}")
    totals = {"downloaded": 0, "cached": 0, "missing": 0, "dry-run": 0}
    for b in selected:
        print(f"\n[{b.code}] {b.class_level} / {b.subject} — {b.title}")
        result = acquire_book(client, manifest, b, args.dry_run)
        for k in totals:
            totals[k] += result.get(k, 0)
        print(f"   ↳ downloaded={result['downloaded']}  cached={result['cached']}  "
              f"missing={result['missing']}  dry-run={result['dry-run']}")

    print(f"\nTotal: {totals}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
