"""PMF IAS article acquirer.

PMF IAS publishes a free public library of UPSC-aimed topic notes
(Geography, Environment, History, Polity, S&T). The site is WordPress-
based with a post-sitemap.xml that enumerates every article. The
article body lives inside the page's `<main>` element.

CLI:
    python -m training.data.acquire.pmf_ias                # all ~1000 articles
    python -m training.data.acquire.pmf_ias --limit 30     # smoke test
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

SITEMAP_URL = "https://www.pmfias.com/post-sitemap.xml"

# PMF IAS content lives inside the <main> element on each post page.
MAIN_RE = re.compile(r"<main[^>]*>(.*?)</main>", re.S | re.I)

# Skip non-article pages (about / disclaimer / privacy / refunds etc.)
SKIP_PATH_RE = re.compile(
    r"^/(?:about|disclaimer|privacy|refunds?|categories-list|contact|recent-posts|"
    r"pmf-ias-books-for-upsc|books)/?$",
)

MIN_BODY_CHARS = 1000


@dataclass(frozen=True)
class PMFArticle:
    url: str
    slug: str


def fetch_sitemap(client: HttpClient) -> list[PMFArticle]:
    """Return all article URLs from the post sitemap, sorted + deduped."""
    r = client.fetch(SITEMAP_URL)
    urls = re.findall(r"<loc>([^<]+)</loc>", r.text)
    out: list[PMFArticle] = []
    seen: set[str] = set()
    for u in urls:
        # Drop trailing slash for path matching
        from urllib.parse import urlparse
        path = urlparse(u).path
        if SKIP_PATH_RE.match(path):
            continue
        slug = path.strip("/").split("/")[-1] or "root"
        if slug in seen:
            continue
        seen.add(slug)
        out.append(PMFArticle(url=u, slug=slug))
    return out


def extract_main_markdown(html: str) -> str | None:
    """Pull the <main> body out and Markdown-ify. Strips scripts + styles
    inside the main first (the page has inline CSS for the countdown timer
    that otherwise leaks into output)."""
    m = MAIN_RE.search(html)
    if not m:
        return None
    body = m.group(1)
    body = re.sub(r"<script[^>]*>.*?</script>", " ", body, flags=re.S | re.I)
    body = re.sub(r"<style[^>]*>.*?</style>", " ", body, flags=re.S | re.I)
    md = markdownify(body, heading_style="ATX", strip=["script", "style", "form", "iframe"])
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if len(md) < MIN_BODY_CHARS:
        return None
    return md


def acquire_article(
    client: HttpClient, manifest: Manifest, article: PMFArticle, dry_run: bool,
) -> tuple[str, int]:
    if manifest.has(article.url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0
    r = client.fetch(article.url)
    md = extract_main_markdown(r.text)
    if md is None:
        return "skipped-empty", 0
    dest_dir = RepoPaths.cpt_raw("pmf_ias")
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
        extra={"slug": article.slug, "extractor": "markdownify"},
    ))
    return "downloaded", len(payload)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Acquire PMF IAS free-tier articles.")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap to first N articles (0 = all). Smoke-test default 0 = all.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=0.5,
                   help="Min seconds between requests")
    args = p.parse_args(argv)

    client = HttpClient(rate_seconds=args.rate)
    print("Fetching PMF IAS sitemap ...")
    articles = fetch_sitemap(client)
    print(f"  found {len(articles)} article URLs")
    if args.limit > 0:
        articles = articles[:args.limit]
        print(f"  after limit ({args.limit}): {len(articles)}")

    manifest = Manifest("pmf_ias")
    totals = {"downloaded": 0, "cached": 0, "skipped-empty": 0, "dry-run": 0, "failed": 0}
    bytes_total = 0
    for i, art in enumerate(articles, start=1):
        if i % 50 == 0 or i <= 3:
            print(f"  [{i:4d}/{len(articles)}] {art.slug[:60]}")
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
