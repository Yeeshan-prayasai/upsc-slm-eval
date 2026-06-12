"""Hindu + Indian Express current-news article scraper.

Both newspapers expose only a recent-articles sitemap publicly
(archives are blocked by robots.txt / paywalled). Scope is therefore
"the last ~week of articles", which is fine for current-affairs
signal in the CPT corpus — re-run weekly to accumulate over time.

Sources:
- The Hindu — `https://www.thehindu.com/sitemap/update/all.xml` (~420 URLs).
  Article body lives in `<div itemprop="articleBody">`.
  Robots-allowed: /sitemap/update/*.xml endpoints; /sitemap/archive/* is disallowed.
- Indian Express — `https://indianexpress.com/news-sitemap.xml` (~450 URLs).
  Article body lives in `<div id="pcl-full-content">`.

Re-runs are idempotent — items already in the manifest are skipped.
The "last-week" content evolves naturally over time.

CLI:
    python -m training.data.acquire.newspapers              # both
    python -m training.data.acquire.newspapers --source hindu
    python -m training.data.acquire.newspapers --source indian-express
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

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


@dataclass(frozen=True)
class NewspaperSource:
    key: str
    sitemap_url: str
    body_pattern: re.Pattern[str]
    min_body_chars: int = 300       # newspapers tend to be shorter than research papers


SOURCES = {
    "hindu": NewspaperSource(
        key="hindu",
        sitemap_url="https://www.thehindu.com/sitemap/update/all.xml",
        body_pattern=re.compile(
            r'<div[^>]*itemprop="articleBody"[^>]*>(.*?)</div>(?=\s*<(?:div|footer|aside))',
            re.S | re.I,
        ),
    ),
    "indian-express": NewspaperSource(
        key="indian-express",
        sitemap_url="https://indianexpress.com/news-sitemap.xml",
        body_pattern=re.compile(
            r'<div[^>]*id="pcl-full-content"[^>]*>(.*?)</div>',
            re.S | re.I,
        ),
    ),
}


@dataclass(frozen=True)
class Article:
    url: str
    slug: str


def _make_client(rate_seconds: float) -> HttpClient:
    client = HttpClient(rate_seconds=rate_seconds)
    client.session.headers.update({"User-Agent": BROWSER_UA})
    return client


def fetch_sitemap(client: HttpClient, source: NewspaperSource) -> list[Article]:
    r = client.fetch(source.sitemap_url)
    out: list[Article] = []
    seen: set[str] = set()
    for url in re.findall(r"<loc>([^<]+)</loc>", r.text):
        if url in seen:
            continue
        seen.add(url)
        path = urlparse(url).path.rstrip("/")
        # Slug: try the last meaningful segment, falling back to article ID.
        parts = [p for p in path.split("/") if p]
        slug = parts[-1] if parts else hashlib.sha256(url.encode()).hexdigest()[:16]
        # Hindu article URLs end in .ece; strip
        slug = slug.replace(".ece", "")
        out.append(Article(url=url, slug=slug))
    return out


def extract_body(html: str, source: NewspaperSource) -> str | None:
    m = source.body_pattern.search(html)
    if not m:
        return None
    body = m.group(1)
    body = re.sub(r"<script[^>]*>.*?</script>", " ", body, flags=re.S | re.I)
    body = re.sub(r"<style[^>]*>.*?</style>", " ", body, flags=re.S | re.I)
    md = markdownify(body, heading_style="ATX",
                     strip=["script", "style", "nav", "header", "footer", "aside",
                            "img", "form", "iframe", "button"])
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if len(md) < source.min_body_chars:
        return None
    return md


def acquire_article(
    client: HttpClient, manifest: Manifest, article: Article, source: NewspaperSource,
    dry_run: bool,
) -> tuple[str, int]:
    if manifest.has(article.url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0
    r = client.fetch(article.url)
    md = extract_body(r.text, source)
    if md is None:
        return "skipped-empty", 0
    dest_dir = RepoPaths.cpt_raw("newspapers") / source.key
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Truncate over-long slugs (some Hindu URLs are absurd)
    fname = (article.slug[:80] or "article") + ".md"
    dest = dest_dir / fname
    payload = md.encode("utf-8")
    dest.write_bytes(payload)
    manifest.add(ManifestEntry(
        url=article.url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        title=article.slug.replace("-", " ")[:120],
        fetched_at=now_iso(),
        extra={"source": source.key, "slug": article.slug},
    ))
    return "downloaded", len(payload)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Hindu + Indian Express current-news scraper.")
    p.add_argument("--source", choices=("hindu", "indian-express", "both"), default="both")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=0.4)
    args = p.parse_args(argv)

    chosen = (["hindu", "indian-express"] if args.source == "both"
              else [args.source])

    client = _make_client(args.rate)
    manifest = Manifest("newspapers")

    grand_totals = {"downloaded": 0, "cached": 0, "skipped-empty": 0,
                    "dry-run": 0, "failed": 0}
    bytes_total = 0

    for src_key in chosen:
        source = SOURCES[src_key]
        print(f"\n=== {src_key} ===")
        print(f"Fetching sitemap: {source.sitemap_url}")
        articles = fetch_sitemap(client, source)
        print(f"  {len(articles)} articles in sitemap")
        if args.limit > 0:
            articles = articles[:args.limit]

        for i, art in enumerate(articles, start=1):
            if i % 50 == 0 or i <= 3:
                print(f"  [{i:4d}/{len(articles)}] {art.slug[:60]}")
            try:
                status, n = acquire_article(client, manifest, art, source, args.dry_run)
                grand_totals[status] = grand_totals.get(status, 0) + 1
                bytes_total += n
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                print(f"     ↳ FAILED {art.slug}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                grand_totals["failed"] += 1

    print(f"\nTotal: {grand_totals}")
    print(f"  bytes written: {bytes_total:,}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
