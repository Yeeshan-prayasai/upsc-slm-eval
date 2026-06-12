"""Indian Space Research Organisation (ISRO) — press release archive.

ISRO publishes press releases as static HTML pages at
`https://www.isro.gov.in/<slug>.html`. The Press.html landing page
lists ~99 unique press release pages with no pagination — that's the
entire visible archive.

Each detail page is a static server-rendered HTML doc (no JS needed)
with the release body in a clearly identifiable content container.

Content licence: Government of India / Department of Space — public
information for non-commercial research.

Closes the GS3 Science & Technology factual gap (per source-list docx,
ISRO/DRDO/DST are the named "core" S&T current-affairs feed).

CLI:
    python -m training.data.acquire.isro
    python -m training.data.acquire.isro --limit 5  # smoke
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

from ._base import HttpClient, Manifest, ManifestEntry, RepoPaths, now_iso

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
BASE = "https://www.isro.gov.in"
LISTING_URL = f"{BASE}/Press.html"
EXCLUDED_LINKS = {"index.html", "Press.html"}
MIN_BODY_CHARS = 400


@dataclass(frozen=True)
class ISRODoc:
    url: str
    slug: str
    title: str


def _make_client(rate_seconds: float) -> HttpClient:
    """ISRO requires browser-like Accept + Accept-Language headers to avoid
    an infinite redirect loop (their load balancer routes bot-like clients
    in a cycle). User-Agent alone is insufficient — observed empirically."""
    client = HttpClient(rate_seconds=rate_seconds)
    client.session.headers.update({
        "User-Agent": BROWSER_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return client


def _list_releases(client: HttpClient) -> list[ISRODoc]:
    r = client.fetch(LISTING_URL)
    soup = BeautifulSoup(r.text, "html.parser")
    docs: list[ISRODoc] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if not href.endswith(".html") or href in EXCLUDED_LINKS:
            continue
        if href.startswith(("#", "javascript", "http")):
            continue
        url = urljoin(BASE + "/", href)
        if url in seen:
            continue
        seen.add(url)
        slug = href.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        title = a.get_text(" ", strip=True)[:280]
        docs.append(ISRODoc(url=url, slug=slug, title=title))
    return docs


def _extract_body(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    # ISRO's detail pages put body content inside <main>. Fall back to
    # a class-name match if that breaks, then body as last resort.
    container = (
        soup.select_one("main")
        or soup.select_one("div[class*=content]")
        or soup.select_one(".contentArea")
        or soup.select_one(".main-content")
        or soup.select_one("article")
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
    if len(md) < MIN_BODY_CHARS:
        return None
    return md


def acquire(
    client: HttpClient,
    manifest: Manifest,
    doc: ISRODoc,
    dry_run: bool,
) -> tuple[str, int]:
    if manifest.has(doc.url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0
    r = client.fetch(doc.url)
    md = _extract_body(r.text)
    if md is None:
        return "skipped-empty", 0
    dest_dir = RepoPaths.cpt_raw("isro")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{doc.slug}.md"
    payload = md.encode("utf-8")
    dest.write_bytes(payload)
    manifest.add(ManifestEntry(
        url=doc.url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        title=doc.title,
        fetched_at=now_iso(),
        extra={"slug": doc.slug},
    ))
    return "downloaded", len(payload)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Acquire ISRO press releases.")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=0.5)
    args = p.parse_args(argv)

    client = _make_client(args.rate)
    docs = _list_releases(client)
    print(f"Found {len(docs)} ISRO press releases")
    if args.limit > 0:
        docs = docs[: args.limit]

    manifest = Manifest("isro")
    totals = {"downloaded": 0, "cached": 0, "skipped-empty": 0,
              "dry-run": 0, "failed": 0}
    bytes_total = 0
    for i, doc in enumerate(docs, start=1):
        if i % 20 == 0 or i <= 3:
            print(f"  [{i:3d}/{len(docs)}] {doc.slug[:70]}")
        try:
            status, n = acquire(client, manifest, doc, args.dry_run)
            totals[status] = totals.get(status, 0) + 1
            bytes_total += n
        except (requests.HTTPError, requests.ConnectionError,
                requests.Timeout, requests.TooManyRedirects,
                PermissionError) as e:
            # ISRO's load balancer intermittently routes some pages into
            # an infinite redirect loop; clear cookies + skip and continue.
            print(f"     FAILED {doc.slug[:60]}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            client.session.cookies.clear()
            totals["failed"] += 1
            continue

    print(f"\nTotal: {totals}")
    print(f"  bytes written: {bytes_total:,}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
