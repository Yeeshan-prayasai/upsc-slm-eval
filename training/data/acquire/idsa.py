"""MP-IDSA (Manohar Parrikar Institute for Defence Studies and Analyses).

idsa.in publishes freely-available strategic-affairs analysis that maps
straight onto GS-II IR and GS-III internal security / defence:
- "IDSA Comments"  → /publisher/comments/<slug>      (~3,100 items)
- Issue Briefs     → /publisher/issuebrief/<slug>    (~780 items)
- Backgrounders    → /publisher/backgrounder/<slug>  (~55 items)

Discovery: the WordPress (Yoast) sitemap index at
`https://idsa.in/sitemap_index.xml` lists `publisher-sitemap*.xml`
files (~7,400 URLs total) whose paths carry the category as the second
segment. The sitemap `<lastmod>` values are bulk-migration noise
(almost everything says 2025), so the publication date is parsed from
the article page's `<span id="postdate">` instead and filtered against
`--since` (default: 5 years back).

Body extraction: article HTML lives in the first
`<div class="... inner-content-area ...">` after the `<h1 id="posttitle">`;
it ends at the keywords/related block.

CLI:
    python -m training.data.acquire.idsa --limit 3 --since 2000-01-01  # smoke
    python -m training.data.acquire.idsa                               # full run
"""
from __future__ import annotations

import argparse
import hashlib
import html as htmllib
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

SITEMAP_INDEX = "https://idsa.in/sitemap_index.xml"
CATEGORIES = ("comments", "issuebrief", "backgrounder")

TITLE_RE = re.compile(r'<h1 id="posttitle"[^>]*>\s*(.*?)\s*</h1>', re.S)
DATE_RE = re.compile(r'<span id="postdate">([^<]+)</span>')
BODY_OPEN_RE = re.compile(r'<div[^>]*class="[^"]*inner-content-area[^"]*"[^>]*>')
# Body ends at the keywords/related block appended after the article text.
# (HTML comments are stripped before this search runs.) `authorDetails`
# bounds migration-artifact footnote stubs (e.g. /issuebrief/25-…) that have
# no Keywords block — their remainder is the author card + site chrome, so
# they fall under MIN_BODY_CHARS and get skipped.
END_MARKERS = ["Keywords :", "inner-content-area RV_test", "authorDetails", "</footer>"]
MIN_BODY_CHARS = 800


@dataclass(frozen=True)
class IDSAArticle:
    url: str
    slug: str
    category: str


def discover(client: HttpClient) -> list[IDSAArticle]:
    """Publisher sitemaps from the index → URLs in our three categories."""
    r = client.fetch(SITEMAP_INDEX)
    sitemaps = [u for u in re.findall(r"<loc>([^<]+)</loc>", r.text)
                if re.search(r"publisher-sitemap\d*\.xml$", u)]
    out: list[IDSAArticle] = []
    seen: set[str] = set()
    for sm in sitemaps:
        r = client.fetch(sm)
        for u in re.findall(r"<loc>([^<]+)</loc>", r.text):
            parts = urlparse(u).path.strip("/").split("/")
            # Expect publisher/<category>/<slug>
            if len(parts) != 3 or parts[0] != "publisher" or parts[1] not in CATEGORIES:
                continue
            if u in seen:
                continue
            seen.add(u)
            out.append(IDSAArticle(url=u, slug=parts[2], category=parts[1]))
    return out


def parse_postdate(html: str) -> datetime | None:
    m = DATE_RE.search(html)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1).strip(), "%B %d, %Y")
    except ValueError:
        return None


def extract_article(html: str) -> tuple[str, str] | None:
    """Return (title, markdown body) or None if the page has no usable body."""
    tm = TITLE_RE.search(html)
    title = htmllib.unescape(
        re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", tm.group(1))).strip()) if tm else ""
    # Anchor the body search after the title so the <style> blocks earlier
    # in the page (which also mention inner-content-area) can't match.
    start_from = tm.end() if tm else 0
    bm = BODY_OPEN_RE.search(html, start_from)
    if not bm:
        return None
    body_html = html[bm.end():]
    # Remove script/style blocks first (their contents embed `<!--`/`-->`
    # pairs that would break tag pairing if comments were stripped first),
    # then CDATA + HTML comments — share-widget blobs otherwise leak through.
    body_html = re.sub(r"<script[^>]*>.*?</script>", " ", body_html, flags=re.S | re.I)
    body_html = re.sub(r"<style[^>]*>.*?</style>", " ", body_html, flags=re.S | re.I)
    body_html = re.sub(r"<!\[CDATA\[.*?\]\]>", " ", body_html, flags=re.S)
    body_html = re.sub(r"<!--.*?-->", " ", body_html, flags=re.S)
    body_html = body_html.replace("<![CDATA[", " ").replace("]]>", " ")
    end_idx = len(body_html)
    for marker in END_MARKERS:
        i = body_html.find(marker)
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
    return title, md


def acquire_article(
    client: HttpClient, manifest: Manifest, art: IDSAArticle,
    since: datetime, dry_run: bool,
) -> tuple[str, int]:
    if manifest.has(art.url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0
    r = client.fetch(art.url)
    pub = parse_postdate(r.text)
    if pub is not None and pub < since:
        return "skipped-old", 0
    extracted = extract_article(r.text)
    if extracted is None:
        return "skipped-empty", 0
    title, md = extracted
    header = f"# {title}\n\n" if title and not md.startswith("#") else ""
    payload = (header + md).encode("utf-8")
    dest_dir = RepoPaths.cpt_raw("idsa") / art.category
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
        extra={"slug": art.slug, "category": art.category,
               "published": pub.strftime("%Y-%m-%d") if pub else ""},
    ))
    return "downloaded", len(payload)


def main(argv: list[str] | None = None) -> int:
    default_since = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="Acquire MP-IDSA comments / issue briefs / backgrounders.")
    p.add_argument("--limit", type=int, default=0, help="Cap to first N per category (0 = all)")
    p.add_argument("--since", default=default_since,
                   help=f"Skip articles published before this date (default {default_since})")
    p.add_argument("--force", action="store_true", help="Re-fetch items already in the manifest")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=1.0, help="Min seconds between requests")
    args = p.parse_args(argv)
    since = datetime.strptime(args.since, "%Y-%m-%d")

    client = HttpClient(rate_seconds=args.rate)
    print("Discovering IDSA publisher sitemaps ...")
    articles = discover(client)
    by_cat: dict[str, list[IDSAArticle]] = {c: [] for c in CATEGORIES}
    for a in articles:
        by_cat[a.category].append(a)
    chosen: list[IDSAArticle] = []
    for cat in CATEGORIES:
        items = by_cat[cat]
        print(f"  {cat}: {len(items)} URLs")
        if args.limit > 0:
            items = items[:args.limit]
        chosen.extend(items)
    if args.limit > 0:
        print(f"  after limit ({args.limit}/category): {len(chosen)}")

    manifest = Manifest("idsa")
    if args.force:
        manifest._seen.clear()
    totals = {"downloaded": 0, "cached": 0, "skipped-old": 0,
              "skipped-empty": 0, "dry-run": 0, "failed": 0}
    bytes_total = 0
    for i, art in enumerate(chosen, start=1):
        if i % 50 == 0 or i <= 3:
            print(f"  [{i:5d}/{len(chosen)}] {art.category}/{art.slug[:60]}")
        try:
            status, n = acquire_article(client, manifest, art, since, args.dry_run)
            totals[status] = totals.get(status, 0) + 1
            bytes_total += n
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
