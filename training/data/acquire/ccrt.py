"""CCRT (Centre for Cultural Resources and Training, ccrtindia.gov.in).

CCRT historically hosted rich Indian art-and-culture study material
(performing arts, visual arts, crafts, festivals, heritage) — classic
GS-I culture content. NOTE: the 2024+ WordPress redesign removed all of
it. As of 2026-06 the site's page sitemap exposes exactly 59 pages, all
administrative (training schedules, scholarships, facility booking,
notices); the legacy section URLs (performingart.php, visualarts.php,
…) return 404 and no mirror exists on the domain.

This module still implements the full discovery → extract → manifest
pipeline against the live sitemap, filtered to cultural-content paths
only (admin/training/booking pages are excluded by design). If the
cultural sections ever return, a re-run picks them up; until then it
prints a clear notice and exits 0.

Discovery: `https://ccrtindia.gov.in/wp-sitemap.xml` (WordPress core
sitemap index) → `wp-sitemap-posts-page-*.xml` page lists.

CLI:
    python -m training.data.acquire.ccrt --limit 5     # smoke
    python -m training.data.acquire.ccrt               # full run
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from markdownify import markdownify

from ._base import (
    HttpClient, Manifest, ManifestEntry, RepoPaths, now_iso, url_local_filename,
)

SITEMAP_INDEX = "https://ccrtindia.gov.in/wp-sitemap.xml"

# Cultural-material sections only. Everything else on the current site is
# administrative (training/booking/scholarship/notice) and excluded.
CULTURAL_PATH_RE = re.compile(
    r"(performing-?arts?|visual-?arts?|crafts?|dance|music|theatre|theater|"
    r"puppet|painting|sculpture|architecture|heritage|festival|literature|"
    r"museum|monument|cultural-heritage)",
    re.I,
)

SECTION_RE = re.compile(r"</?section[^>]*>", re.I)
MIN_BODY_CHARS = 500


@dataclass(frozen=True)
class CCRTPage:
    url: str
    slug: str
    section: str   # first path segment


def discover(client: HttpClient) -> list[CCRTPage]:
    """Page URLs from the WP sitemap whose path looks like cultural content."""
    r = client.fetch(SITEMAP_INDEX)
    sitemaps = [u for u in re.findall(r"<loc>([^<]+)</loc>", r.text)
                if "wp-sitemap-posts-page-" in u]
    out: list[CCRTPage] = []
    seen: set[str] = set()
    for sm in sitemaps:
        r = client.fetch(sm)
        for u in re.findall(r"<loc>([^<]+)</loc>", r.text):
            path = urlparse(u).path
            if not CULTURAL_PATH_RE.search(path) or u in seen:
                continue
            seen.add(u)
            parts = path.strip("/").split("/")
            out.append(CCRTPage(url=u, slug=parts[-1] or "root", section=parts[0]))
    return out


def extract_page(html: str) -> str | None:
    """CCRT pages have no <main>/<article>; the body sits in the first
    <section> block before the <footer>. Markdownify that slice."""
    m = re.search(r"<section[^>]*>", html, re.I)
    if not m:
        return None
    body_html = html[m.start():]
    fi = body_html.lower().find("<footer")
    if fi > 0:
        body_html = body_html[:fi]
    md = markdownify(
        body_html, heading_style="ATX",
        strip=["script", "style", "nav", "header", "footer", "aside",
               "img", "form", "iframe", "button"],
    )
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if len(md) < MIN_BODY_CHARS:
        return None
    return md


def acquire_page(
    client: HttpClient, manifest: Manifest, page: CCRTPage, dry_run: bool,
) -> tuple[str, int]:
    if manifest.has(page.url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0
    r = client.fetch(page.url)
    md = extract_page(r.text)
    if md is None:
        return "skipped-empty", 0
    dest_dir = RepoPaths.cpt_raw("ccrt") / page.section
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / url_local_filename(page.url, ".md")
    payload = md.encode("utf-8")
    dest.write_bytes(payload)
    manifest.add(ManifestEntry(
        url=page.url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        title=page.slug.replace("-", " "),
        fetched_at=now_iso(),
        extra={"slug": page.slug, "section": page.section},
    ))
    return "downloaded", len(payload)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Acquire CCRT cultural-material pages.")
    p.add_argument("--limit", type=int, default=0, help="Cap to first N pages (0 = all)")
    p.add_argument("--force", action="store_true", help="Re-fetch items already in the manifest")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=1.0, help="Min seconds between requests")
    args = p.parse_args(argv)

    client = HttpClient(rate_seconds=args.rate)
    print("Discovering CCRT pages from wp-sitemap ...")
    pages = discover(client)
    print(f"  found {len(pages)} cultural-content page URLs")
    if not pages:
        print("\nNothing to acquire: ccrtindia.gov.in currently exposes no "
              "cultural-material pages.\nThe 2024+ site redesign removed the "
              "performing-arts / visual-arts / crafts sections;\nonly "
              "administrative pages (training, scholarships, bookings, notices) "
              "remain in the sitemap.\nRe-run if the cultural sections are restored.")
        return 0
    if args.limit > 0:
        pages = pages[:args.limit]
        print(f"  after limit ({args.limit}): {len(pages)}")

    manifest = Manifest("ccrt")
    if args.force:
        manifest._seen.clear()
    totals = {"downloaded": 0, "cached": 0, "skipped-empty": 0, "dry-run": 0, "failed": 0}
    bytes_total = 0
    for i, page in enumerate(pages, start=1):
        if i % 25 == 0 or i <= 3:
            print(f"  [{i:4d}/{len(pages)}] {page.section}/{page.slug[:60]}")
        try:
            status, n = acquire_page(client, manifest, page, args.dry_run)
            totals[status] = totals.get(status, 0) + 1
            bytes_total += n
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            print(f"     ↳ FAILED {page.slug[:60]}: {type(e).__name__}: {e}", file=sys.stderr)
            totals["failed"] += 1
            continue

    print(f"\nTotal: {totals}")
    print(f"  bytes written: {bytes_total:,}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
