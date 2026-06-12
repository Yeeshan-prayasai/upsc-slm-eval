"""DRDO press releases — sourced via DRDO's archive index, fetched from PIB.

DRDO's archive at `/drdo/en/documents/press-release/archive` lists ~60
press releases (6 pages × ~10 rows) as a table of titles + dates with
"View" links that route to `pib.gov.in/PressReleasePage.aspx?PRID=N`.

This is convenient: DRDO does the curation, but the actual content
lives on PIB (the unified GoI press distributor). We:
1. Walk the archive's 6 pages to harvest PIB PRID URLs + titles.
2. Fetch each PIB detail page directly using its plain GET endpoint
   (the PRID-based URL works without ASP.NET viewstate handling, even
   though PIB's listing search requires viewstate POSTs).
3. Extract the body from `div.innner-page-main-about-us-content-right-part`
   (PIB's typoed container class).

Content licence: Government of India / DRDO + PIB — public-domain
press content.

CLI:
    python -m training.data.acquire.drdo
    python -m training.data.acquire.drdo --limit 5
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

from ._base import HttpClient, Manifest, ManifestEntry, RepoPaths, now_iso

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
ARCHIVE = "https://www.drdo.gov.in/drdo/en/documents/press-release/archive"
PIB_BASE = "https://pib.gov.in/PressReleasePage.aspx"
PRID_RE = re.compile(r"PRID=(\d+)")
MIN_BODY_CHARS = 300


@dataclass(frozen=True)
class DRDODoc:
    prid: int
    pib_url: str
    title: str
    date: str


def _make_client(rate_seconds: float) -> HttpClient:
    client = HttpClient(rate_seconds=rate_seconds)
    client.session.headers.update({"User-Agent": BROWSER_UA})
    return client


def _list_archive_page(client: HttpClient, page: int) -> list[DRDODoc]:
    url = f"{ARCHIVE}?page={page}" if page > 0 else ARCHIVE
    r = client.fetch(url)
    soup = BeautifulSoup(r.text, "html.parser")
    docs: list[DRDODoc] = []
    for row in soup.select("table tbody tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        # Heuristic: first non-numeric cell is title; date is in another cell.
        text = row.get_text(" | ", strip=True)
        parts = [p.strip() for p in text.split("|")]
        # Find a link with PRID
        a = row.find("a", href=PRID_RE)
        if not a:
            continue
        m = PRID_RE.search(a.get("href", ""))
        if not m:
            continue
        prid = int(m.group(1))
        # Title = the longest non-date, non-numeric cell text
        title = ""
        date = ""
        for p in parts:
            if re.match(r"^\d+$", p):
                continue
            if re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", p):
                date = date or p
                continue
            if len(p) > len(title):
                title = p
        docs.append(DRDODoc(
            prid=prid,
            pib_url=f"{PIB_BASE}?PRID={prid}&reg=3&lang=1",
            title=title[:280],
            date=date,
        ))
    return docs


def _list_all(client: HttpClient, max_pages: int = 20) -> list[DRDODoc]:
    docs: list[DRDODoc] = []
    seen: set[int] = set()
    empty = 0
    for page in range(max_pages):
        items = _list_archive_page(client, page)
        if not items:
            empty += 1
            if empty >= 2:
                break
            continue
        empty = 0
        fresh = [d for d in items if d.prid not in seen]
        for d in fresh:
            seen.add(d.prid)
        docs.extend(fresh)
    return docs


def _extract_pib_body(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    # PIB's content container has a typo in the class name; keep both
    # spellings just in case they fix it later.
    container = (
        soup.select_one("div.innner-page-main-about-us-content-right-part")
        or soup.select_one("div.inner-page-main-about-us-content-right-part")
        or soup.select_one("div.container article")
        or soup.body
    )
    if not container:
        return None
    md = markdownify(
        str(container),
        heading_style="ATX",
        strip=["script", "style", "nav", "header", "footer", "aside",
               "img", "form", "iframe", "button"],
    )
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md if len(md) >= MIN_BODY_CHARS else None


def acquire(
    client: HttpClient,
    manifest: Manifest,
    doc: DRDODoc,
    dry_run: bool,
) -> tuple[str, int]:
    if manifest.has(doc.pib_url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0
    r = client.fetch(doc.pib_url)
    md = _extract_pib_body(r.text)
    if md is None:
        return "skipped-empty", 0
    dest_dir = RepoPaths.cpt_raw("drdo")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{doc.prid}.md"
    payload = md.encode("utf-8")
    dest.write_bytes(payload)
    manifest.add(ManifestEntry(
        url=doc.pib_url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        title=doc.title,
        fetched_at=now_iso(),
        extra={"prid": doc.prid, "date": doc.date},
    ))
    return "downloaded", len(payload)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Acquire DRDO press releases via PIB.")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=0.6)
    args = p.parse_args(argv)

    client = _make_client(args.rate)
    docs = _list_all(client)
    print(f"Found {len(docs)} DRDO press releases in archive")
    if args.limit > 0:
        docs = docs[: args.limit]

    manifest = Manifest("drdo")
    totals = {"downloaded": 0, "cached": 0, "skipped-empty": 0,
              "dry-run": 0, "failed": 0}
    bytes_total = 0
    for i, doc in enumerate(docs, start=1):
        if i % 10 == 0 or i <= 3:
            print(f"  [{i:3d}/{len(docs)}] PRID={doc.prid} {doc.title[:60]!r}")
        try:
            status, n = acquire(client, manifest, doc, args.dry_run)
            totals[status] = totals.get(status, 0) + 1
            bytes_total += n
        except (requests.HTTPError, requests.ConnectionError,
                requests.Timeout, PermissionError) as e:
            print(f"     FAILED {doc.prid}: {type(e).__name__}: {e}", file=sys.stderr)
            totals["failed"] += 1
            continue

    print(f"\nTotal: {totals}")
    print(f"  bytes written: {bytes_total:,}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
