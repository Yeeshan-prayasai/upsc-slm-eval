"""Mrunal article acquirer.

Mrunal publishes ~3000 free UPSC-targeted articles spanning Economy,
Polity, and exam-prep guidance. WordPress-based with three numbered
post-sitemaps; the post body lives inside `<article>` on each page.

CLI:
    python -m training.data.acquire.mrunal              # all ~3000 posts
    python -m training.data.acquire.mrunal --limit 30   # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass

import requests
from markdownify import markdownify

from ._base import HttpClient, Manifest, ManifestEntry, RepoPaths, now_iso

SITEMAP_INDEX = "https://mrunal.org/sitemap.xml"
ARTICLE_RE = re.compile(r"<article[^>]*>(.*?)</article>", re.S | re.I)
MIN_BODY_CHARS = 800


@dataclass(frozen=True)
class MrunalPost:
    url: str
    slug: str


def fetch_all_post_urls(client: HttpClient) -> list[MrunalPost]:
    """Read the sitemap index, follow each `post-sitemap*.xml` sub-sitemap,
    return the union of article URLs."""
    idx = client.fetch(SITEMAP_INDEX)
    sub_sitemaps = [u for u in re.findall(r"<loc>([^<]+)</loc>", idx.text)
                    if "post-sitemap" in u]
    seen: set[str] = set()
    out: list[MrunalPost] = []
    for sub in sub_sitemaps:
        sm = client.fetch(sub)
        for u in re.findall(r"<loc>([^<]+)</loc>", sm.text):
            if u in seen:
                continue
            seen.add(u)
            # Slug: last non-empty path segment, strip .html extension
            from urllib.parse import urlparse
            slug = urlparse(u).path.rstrip("/").split("/")[-1]
            slug = slug.replace(".html", "")
            if not slug:
                continue
            out.append(MrunalPost(url=u, slug=slug))
    return out


def extract_post_markdown(html: str) -> str | None:
    matches = ARTICLE_RE.findall(html)
    if not matches:
        return None
    # If multiple article tags (e.g. on archive pages), take the longest.
    body = max(matches, key=len)
    body = re.sub(r"<script[^>]*>.*?</script>", " ", body, flags=re.S | re.I)
    body = re.sub(r"<style[^>]*>.*?</style>", " ", body, flags=re.S | re.I)
    md = markdownify(body, heading_style="ATX", strip=["script", "style", "form", "iframe"])
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if len(md) < MIN_BODY_CHARS:
        return None
    return md


def acquire_post(
    client: HttpClient, manifest: Manifest, post: MrunalPost, dry_run: bool,
) -> tuple[str, int]:
    if manifest.has(post.url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0
    r = client.fetch(post.url)
    md = extract_post_markdown(r.text)
    if md is None:
        return "skipped-empty", 0
    dest_dir = RepoPaths.cpt_raw("mrunal")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{post.slug}.md"
    payload = md.encode("utf-8")
    dest.write_bytes(payload)
    manifest.add(ManifestEntry(
        url=post.url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        title=post.slug.replace("-", " "),
        fetched_at=now_iso(),
        extra={"slug": post.slug, "extractor": "markdownify"},
    ))
    return "downloaded", len(payload)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Acquire Mrunal article corpus.")
    p.add_argument("--limit", type=int, default=0, help="Cap to first N posts (0 = all)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=0.5)
    args = p.parse_args(argv)

    client = HttpClient(rate_seconds=args.rate)
    print("Fetching Mrunal sitemaps ...")
    posts = fetch_all_post_urls(client)
    print(f"  found {len(posts)} post URLs across sub-sitemaps")
    if args.limit > 0:
        posts = posts[:args.limit]
        print(f"  after limit ({args.limit}): {len(posts)}")

    manifest = Manifest("mrunal")
    totals = {"downloaded": 0, "cached": 0, "skipped-empty": 0, "dry-run": 0, "failed": 0}
    bytes_total = 0
    for i, post in enumerate(posts, start=1):
        if i % 50 == 0 or i <= 3:
            print(f"  [{i:5d}/{len(posts)}] {post.slug[:60]}")
        try:
            status, n = acquire_post(client, manifest, post, args.dry_run)
            totals[status] = totals.get(status, 0) + 1
            bytes_total += n
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            print(f"     ↳ FAILED {post.slug}: {type(e).__name__}: {e}", file=sys.stderr)
            totals["failed"] += 1
            continue

    print(f"\nTotal: {totals}")
    print(f"  bytes written: {bytes_total:,}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
