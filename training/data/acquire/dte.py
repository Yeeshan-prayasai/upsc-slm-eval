"""Down To Earth (downtoearth.org.in) — environment journalism.

CSE's Down To Earth is the standard UPSC environment-current-affairs
source (GS-III environment/ecology). The site is a Quintype platform;
robots.txt currently allows everything (`Allow: /`), and the
`HttpClient` enforces robots on every request, so if that ever changes
the run stops with a clear message rather than bypassing it.

Discovery: `https://www.downtoearth.org.in/sitemap.xml` is an index of
per-day sitemaps named `sitemap-daily-YYYY-MM-DD.xml` going back to
1991 (~5,600 files, ~10 article URLs each). The date in the filename
lets us window to `--since` (default: 5 years back) without fetching
old sitemaps. Articles are kept only for the focus sections —
environment, climate-change, wildlife-biodiversity, pollution — which
also excludes the Hindi edition (different URL prefixes).

Paywall: most news is free; premium stories carry
`"access":"subscription"` in the embedded story JSON and are skipped.
The free-article body is server-rendered inside
`arr--element-container` divs, ending at the `arr--story-tags` block.

CLI:
    python -m training.data.acquire.dte --limit 5      # smoke
    python -m training.data.acquire.dte                 # full run (last 5 yrs)
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlparse

import requests
from markdownify import markdownify

from ._base import (
    HttpClient, Manifest, ManifestEntry, RepoPaths, now_iso, url_local_filename,
)

SITEMAP_INDEX = "https://www.downtoearth.org.in/sitemap.xml"
SECTIONS = ("environment", "climate-change", "wildlife-biodiversity", "pollution")

DAILY_RE = re.compile(r"sitemap-daily-(\d{4}-\d{2}-\d{2})\.xml$")
TITLE_RE = re.compile(r'data-testid="story-headline"[^>]*>\s*(.*?)\s*</h1>', re.S)
BODY_OPEN_RE = re.compile(r'<div class="arr--element-container[^"]*"[^>]*>')
END_MARKER = "arr--story-tags"
PAYWALL_MARKER = '"access":"subscription"'
MIN_BODY_CHARS = 600


@dataclass(frozen=True)
class DTEArticle:
    url: str
    slug: str
    section: str


def discover(client: HttpClient, since: datetime, limit: int) -> list[DTEArticle]:
    """Daily sitemaps newest-first, windowed to `since`, filtered to the
    focus sections. Stops early once `limit` articles are collected."""
    r = client.fetch(SITEMAP_INDEX)
    # Quintype serves sitemaps as `text/xml` with no charset parameter, so
    # `r.text` decodes the UTF-8 body as ISO-8859-1 (requests' RFC 2616
    # default). Non-ASCII slugs (el-niño, Belém …) mojibake into Ã±/Ã© and
    # the article fetch then double-encodes them (%C3%83%C2%B1) → 404.
    # The XML prolog declares UTF-8; decode explicitly.
    dailies: list[tuple[datetime, str]] = []
    for u in re.findall(r"<loc>([^<]+)</loc>", r.content.decode("utf-8", "replace")):
        m = DAILY_RE.search(u)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y-%m-%d")
        if d >= since:
            dailies.append((d, u))
    dailies.sort(reverse=True)
    print(f"  {len(dailies)} daily sitemaps since {since:%Y-%m-%d}")

    out: list[DTEArticle] = []
    seen: set[str] = set()
    for i, (_, sm) in enumerate(dailies, start=1):
        r = client.fetch(sm)
        for u in re.findall(r"<loc>([^<]+)</loc>", r.content.decode("utf-8", "replace")):
            parts = urlparse(u).path.strip("/").split("/")
            if len(parts) != 2 or parts[0] not in SECTIONS or u in seen:
                continue
            seen.add(u)
            out.append(DTEArticle(url=u, slug=parts[1], section=parts[0]))
        if limit and len(out) >= limit:
            return out[:limit]
        if i % 100 == 0:
            print(f"  scanned {i}/{len(dailies)} daily sitemaps, {len(out)} article URLs")
    return out


def extract_article(html: str) -> tuple[str, str] | None:
    """Return (title, markdown body) or None if no usable free body."""
    tm = TITLE_RE.search(html)
    title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", tm.group(1))).strip() if tm else ""
    bm = BODY_OPEN_RE.search(html)
    if not bm:
        return None
    body_html = html[bm.start():]
    ei = body_html.find(END_MARKER)
    if ei > 0:
        body_html = body_html[:ei]
    md = markdownify(
        body_html, heading_style="ATX",
        strip=["script", "style", "nav", "header", "footer", "aside",
               "img", "form", "iframe", "button", "svg"],
    )
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if len(md) < MIN_BODY_CHARS:
        return None
    return title, md


def acquire_article(
    client: HttpClient, manifest: Manifest, art: DTEArticle, dry_run: bool,
) -> tuple[str, int]:
    if manifest.has(art.url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0
    r = client.fetch(art.url)
    if PAYWALL_MARKER in r.text:
        return "skipped-paywalled", 0
    extracted = extract_article(r.text)
    if extracted is None:
        return "skipped-empty", 0
    title, md = extracted
    # The embedded-JSON paywall flag doesn't fire on server-rendered teaser
    # pages; the rendered body carries a subscribe call-to-action instead.
    if "To Continue Reading Subscribe" in md or "downtoearth.org.in/subscription" in md:
        return "skipped-paywalled", 0
    header = f"# {title}\n\n" if title and not md.startswith("#") else ""
    payload = (header + md).encode("utf-8")
    dest_dir = RepoPaths.cpt_raw("dte") / art.section
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / url_local_filename(art.url, ".md")
    dest.write_bytes(payload)
    manifest.add(ManifestEntry(
        url=art.url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        title=title or art.slug.replace("-", " "),
        fetched_at=now_iso(),
        extra={"slug": art.slug, "section": art.section},
    ))
    return "downloaded", len(payload)


def main(argv: list[str] | None = None) -> int:
    default_since = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="Acquire Down To Earth free environment articles.")
    p.add_argument("--limit", type=int, default=0, help="Cap to first N articles (0 = all)")
    p.add_argument("--since", default=default_since,
                   help=f"Only daily sitemaps on/after this date (default {default_since})")
    p.add_argument("--force", action="store_true", help="Re-fetch items already in the manifest")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=1.0, help="Min seconds between requests")
    args = p.parse_args(argv)
    since = datetime.strptime(args.since, "%Y-%m-%d")

    client = HttpClient(rate_seconds=args.rate)
    print("Discovering Down To Earth daily sitemaps ...")
    try:
        articles = discover(client, since, args.limit)
    except PermissionError as e:
        print(f"\nrobots.txt now disallows crawling — stopping without bypass.\n  {e}")
        return 0
    print(f"  found {len(articles)} article URLs in sections {SECTIONS}")

    manifest = Manifest("dte")
    if args.force:
        manifest._seen.clear()
    totals = {"downloaded": 0, "cached": 0, "skipped-paywalled": 0,
              "skipped-empty": 0, "dry-run": 0, "failed": 0}
    bytes_total = 0
    for i, art in enumerate(articles, start=1):
        if i % 50 == 0 or i <= 3:
            print(f"  [{i:5d}/{len(articles)}] {art.section}/{art.slug[:60]}")
        try:
            status, n = acquire_article(client, manifest, art, args.dry_run)
            totals[status] = totals.get(status, 0) + 1
            bytes_total += n
        except PermissionError as e:
            print(f"\nrobots.txt now disallows crawling — stopping without bypass.\n  {e}")
            break
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            print(f"     ↳ FAILED {art.slug[:60]}: {type(e).__name__}: {e}", file=sys.stderr)
            totals["failed"] += 1
            continue

    print(f"\nTotal: {totals}")
    print(f"  bytes written: {bytes_total:,}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
