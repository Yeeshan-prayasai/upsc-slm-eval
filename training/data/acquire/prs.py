"""PRS India bill-summary acquirer.

PRS Legislative Research publishes neutral, plain-English summaries of
every bill introduced in Parliament since ~2005, including Highlights,
Key Issues, and analysis. The content lives on each bill's page at
`prsindia.org/billtrack/<slug>` as structured HTML inside an
`<article>` element — we extract that, convert to Markdown via
`markdownify`, and save one `.md` per bill.

CLI:
    python -m training.data.acquire.prs                        # all bills
    python -m training.data.acquire.prs --limit 50             # first N (smoke test)
    python -m training.data.acquire.prs --since-year 2020      # filter on URL slug year
    python -m training.data.acquire.prs --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from markdownify import markdownify

from ._base import HttpClient, Manifest, ManifestEntry, RepoPaths, now_iso

BASE = "https://prsindia.org"
LISTING_URL = f"{BASE}/billtrack"

# Match bill-page slugs only — skip /billtrack/category/* index pages.
BILL_HREF_RE = re.compile(r'href="(/billtrack/(?!category/)[^"#?]+)"')

# The actual content container on a PRS bill page.
ARTICLE_RE = re.compile(r"<article[^>]*>(.*?)</article>", re.S | re.I)

# Reasonable minimum body length — pages shorter than this are skeleton
# stubs (status-only entries with no text content yet).
MIN_BODY_CHARS = 500


@dataclass(frozen=True)
class BillRef:
    slug: str
    url: str


def fetch_bill_list(client: HttpClient) -> list[BillRef]:
    """Fetch the master listing page and enumerate every bill URL."""
    r = client.fetch(LISTING_URL)
    refs: dict[str, BillRef] = {}
    for m in BILL_HREF_RE.finditer(r.text):
        path = m.group(1)
        if path.startswith("/billtrack/category"):
            continue
        slug = path.rsplit("/", 1)[-1]
        if not slug or slug == "billtrack":
            continue
        refs[slug] = BillRef(slug=slug, url=f"{BASE}{path}")
    return sorted(refs.values(), key=lambda b: b.slug)


def extract_bill_markdown(html: str) -> str | None:
    """Pull the `<article>` body out of a bill page and convert to
    clean Markdown. Returns None if no article container is found
    or the body is too short to be meaningful."""
    m = ARTICLE_RE.search(html)
    if not m:
        return None
    body_html = m.group(1)
    md = markdownify(body_html, heading_style="ATX", strip=["script", "style"])
    # Collapse runs of blank lines.
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if len(md) < MIN_BODY_CHARS:
        return None
    return md


def acquire_bill(
    client: HttpClient, manifest: Manifest, bill: BillRef, dry_run: bool,
) -> tuple[str, int]:
    """Returns (status, bytes_written)."""
    if manifest.has(bill.url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0
    r = client.fetch(bill.url)
    md = extract_bill_markdown(r.text)
    if md is None:
        return "skipped-empty", 0
    dest_dir = RepoPaths.cpt_raw("prs")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{bill.slug}.md"
    payload = md.encode("utf-8")
    dest.write_bytes(payload)
    manifest.add(ManifestEntry(
        url=bill.url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        title=bill.slug.replace("-", " ").title(),
        fetched_at=now_iso(),
        extra={"slug": bill.slug, "extractor": "markdownify"},
    ))
    return "downloaded", len(payload)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Acquire PRS India bill summaries as Markdown."
    )
    p.add_argument("--limit", type=int, default=0,
                   help="Cap to first N bills (0 = all). Useful for smoke tests.")
    p.add_argument("--since-year", type=int, default=0,
                   help="Filter to bills whose slug starts with a year >= this (e.g. 2020)")
    p.add_argument("--dry-run", action="store_true", help="Print URLs without fetching")
    p.add_argument("--rate", type=float, default=0.5,
                   help="Min seconds between requests")
    args = p.parse_args(argv)

    client = HttpClient(rate_seconds=args.rate)
    print("Fetching PRS India bill listing ...")
    refs = fetch_bill_list(client)
    print(f"  found {len(refs)} bill pages")

    if args.since_year:
        refs = [r for r in refs if r.slug[:4].isdigit() and int(r.slug[:4]) >= args.since_year]
        print(f"  after year filter (>= {args.since_year}): {len(refs)}")
    if args.limit > 0:
        refs = refs[:args.limit]
        print(f"  after limit ({args.limit}): {len(refs)}")

    manifest = Manifest("prs")
    totals = {"downloaded": 0, "cached": 0, "skipped-empty": 0, "dry-run": 0, "failed": 0}
    bytes_total = 0

    for i, bill in enumerate(refs, start=1):
        if i % 20 == 0 or i <= 3:
            print(f"  [{i:4d}/{len(refs)}] {bill.slug[:60]}")
        try:
            status, n = acquire_bill(client, manifest, bill, args.dry_run)
            totals[status] = totals.get(status, 0) + 1
            bytes_total += n
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            print(f"      ↳ FAILED on {bill.slug}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            totals["failed"] += 1
            continue

    print(f"\nTotal: {totals}")
    print(f"  bytes written: {bytes_total:,}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
