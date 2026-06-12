"""Observer Research Foundation (ORF) — research papers + expert columns.

ORF is one of India's most-cited think tanks; their English-language
research and expert-speak content is highly UPSC-relevant for GS-II IR,
GS-III Economy/Tech, and Mains essay material.

Two sitemaps exposed in their `robots.txt`:
- `https://www.orfonline.org/research-sitemap.xml` — ~12,900 research papers
- `https://www.orfonline.org/expert-speak-sitemap.xml` — ~9,900 expert columns

Each article page has an `<article>` element with the body. Total
English content: ~22,800 articles → est. 100-300 M tokens.

We skip the Hindi/Bangla/Marathi sitemaps per the English-only scope.

CLI:
    python -m training.data.acquire.orf              # both sitemaps (default)
    python -m training.data.acquire.orf --sitemap research
    python -m training.data.acquire.orf --sitemap expert-speak
    python -m training.data.acquire.orf --limit 100
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

from ._base import HttpClient, Manifest, ManifestEntry, RepoPaths, now_iso

# Use a Chrome UA — ORF returns generic 404 pages to the default UA.
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

SITEMAPS = {
    "research":     "https://www.orfonline.org/research-sitemap.xml",
    "expert-speak": "https://www.orfonline.org/expert-speak-sitemap.xml",
}

# ORF pages have no `<article>` or `<main>` wrapper — content is a flat
# sequence of <p>, <h2>, <h4>, <ul>, etc. directly under <body>. We anchor
# from the article's `<h1>` and stop at common footer/related-articles
# markers. This gets ~30 K Markdown chars per research paper after
# markdownify (the inline citations + figures + footnotes survive cleanly).
H1_OPEN_RE = re.compile(r"<h1[^>]*>", re.I)
END_MARKERS = [
    "</footer>", "<footer", "related-articles", "Related Articles",
    '<section class="container related',
    '<div class="footer', "navigation-section",
]
MIN_BODY_CHARS = 800


@dataclass(frozen=True)
class ORFArticle:
    url: str
    slug: str
    category: str   # "research" or "expert-speak"


def _make_client(rate_seconds: float) -> HttpClient:
    """ORF needs a browser-like UA. Override the default."""
    client = HttpClient(rate_seconds=rate_seconds)
    client.session.headers.update({"User-Agent": BROWSER_UA})
    return client


def fetch_sitemap(client: HttpClient, sitemap_url: str, category: str) -> list[ORFArticle]:
    r = client.fetch(sitemap_url)
    urls = re.findall(r"<loc>([^<]+)</loc>", r.text)
    out: list[ORFArticle] = []
    for u in urls:
        slug = urlparse(u).path.rstrip("/").split("/")[-1]
        if not slug:
            continue
        out.append(ORFArticle(url=u, slug=slug, category=category))
    return out


def extract_article(html: str) -> str | None:
    """Anchor from the article's <h1> and stop at first footer/related
    marker; markdownify the slice."""
    m = H1_OPEN_RE.search(html)
    if not m:
        return None
    body_html = html[m.start():]
    # Find earliest occurring end-marker
    end_idx = len(body_html)
    lower = body_html.lower()
    for marker in END_MARKERS:
        i = lower.find(marker.lower())
        if 0 < i < end_idx:
            end_idx = i
    body_html = body_html[:end_idx]
    md = markdownify(
        body_html, heading_style="ATX",
        strip=["script", "style", "nav", "header", "footer", "aside",
               "img", "form", "iframe", "button"],
    )
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if len(md) < MIN_BODY_CHARS:
        return None
    return md


def acquire_article(
    client: HttpClient, manifest: Manifest, article: ORFArticle, dry_run: bool,
) -> tuple[str, int]:
    if manifest.has(article.url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0
    r = client.fetch(article.url)
    md = extract_article(r.text)
    if md is None:
        return "skipped-empty", 0
    dest_dir = RepoPaths.cpt_raw("orf") / article.category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{article.slug}.md"
    payload = md.encode("utf-8")
    dest.write_bytes(payload)
    manifest.add(ManifestEntry(
        url=article.url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        title=article.slug.replace("-", " "),
        fetched_at=now_iso(),
        extra={"slug": article.slug, "category": article.category},
    ))
    return "downloaded", len(payload)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Acquire ORF research + expert-speak articles.")
    p.add_argument("--sitemap", choices=("research", "expert-speak", "both"), default="both")
    p.add_argument("--limit", type=int, default=0, help="Cap to first N per sitemap (0 = all)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=0.4)
    args = p.parse_args(argv)

    client = _make_client(args.rate)
    chosen = (["research", "expert-speak"] if args.sitemap == "both"
              else [args.sitemap])
    all_articles: list[ORFArticle] = []
    for cat in chosen:
        print(f"Fetching ORF {cat} sitemap ...")
        urls = fetch_sitemap(client, SITEMAPS[cat], cat)
        print(f"  found {len(urls)} {cat} URLs")
        if args.limit > 0:
            urls = urls[:args.limit]
            print(f"  after limit ({args.limit}): {len(urls)}")
        all_articles.extend(urls)

    manifest = Manifest("orf")
    totals = {"downloaded": 0, "cached": 0, "skipped-empty": 0, "dry-run": 0, "failed": 0}
    bytes_total = 0
    for i, art in enumerate(all_articles, start=1):
        if i % 100 == 0 or i <= 3:
            print(f"  [{i:5d}/{len(all_articles)}] {art.category}/{art.slug[:60]}")
        try:
            status, n = acquire_article(client, manifest, art, args.dry_run)
            totals[status] = totals.get(status, 0) + 1
            bytes_total += n
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            print(f"     ↳ FAILED {art.slug}: {type(e).__name__}: {e}", file=sys.stderr)
            totals["failed"] += 1
            continue

    print(f"\nTotal: {totals}")
    print(f"  bytes written: {bytes_total:,}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
