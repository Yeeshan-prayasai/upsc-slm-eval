"""Ministry of External Affairs (MEA) — press releases, speeches, and
bilateral documents.

MEA's modern site is JS-rendered but exposes two clean AJAX endpoints:
- `/FrontEnd/FetchPublicationListingData?publicationId=<id>&PLngId=1&page=N&PageSize=20`
  Returns an HTML fragment containing 20 title/date/link items per page.
- `/FrontEnd/FetchPublicationDetailData?pkid=<id>&languageId=1`
  Returns the full body of a single press release / speech as clean HTML.

Per the route map embedded in MEA's page JS, publication IDs are:
  49 = Media Briefings        50 = Speeches & Statements
  51 = Press Releases         52 = Interviews
  53 = Bilateral Documents    60 = Media Advisory
  61 = Lok Sabha              62 = Rajya Sabha
  69 = Response to Media Queries

We default to the four IR-heavy types (51 PR, 50 Speeches, 53 Bilateral
Documents, 69 Media Queries) — the ones that carry the substantive
policy content. Empty-page detection (no detail links returned) signals
end-of-archive.

Content licence: Government of India Open Data Policy.

CLI:
    python -m training.data.acquire.mea                       # default 4 types
    python -m training.data.acquire.mea --publication 51      # press releases only
    python -m training.data.acquire.mea --max-pages 50        # cap pages per type
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

BASE = "https://www.mea.gov.in"
LISTING = "/FrontEnd/FetchPublicationListingData"
DETAIL = "/FrontEnd/FetchPublicationDetailData"

PUBLICATION_TYPES = {
    51: "press-releases",
    50: "speeches-statements",
    53: "bilateral-documents",
    69: "response-to-queries",
}

DEFAULT_PUB_IDS = list(PUBLICATION_TYPES.keys())
MIN_BODY_CHARS = 200
PKID_RE = re.compile(r"dtl/(\d+)(?:/([^?#]*))?")


@dataclass(frozen=True)
class MEADoc:
    pkid: int
    title: str
    date: str
    slug: str
    publication_id: int
    detail_url: str  # canonical absolute URL for manifest


def _make_client(rate_seconds: float) -> HttpClient:
    client = HttpClient(rate_seconds=rate_seconds)
    client.session.headers.update({
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{BASE}/",
    })
    return client


def _list_page(
    client: HttpClient, publication_id: int, page: int
) -> list[MEADoc]:
    """Fetch one listing page; return parsed MEADoc items."""
    url = f"{BASE}{LISTING}"
    params = {
        "publicationId": publication_id,
        "PLngId": 1,
        "page": page,
        "PageSize": 20,
        "SortBy": "",
        "DateRange": "",
        "KeywordName": "",
        "IsInternalMEA": "false",
    }
    r = client.fetch(url, params=params)
    soup = BeautifulSoup(r.text, "html.parser")
    docs: list[MEADoc] = []
    for box in soup.select("div.pressRelesastBox, div[class*=PressRelesastBox]"):
        a = box.select_one("h3.pressTitle a") or box.select_one("a[href*='dtl/']")
        if not a:
            continue
        href = a.get("href", "")
        m = PKID_RE.search(href)
        if not m:
            continue
        pkid = int(m.group(1))
        slug = (m.group(2) or "").strip("/") or f"pr_{pkid}"
        date_el = box.select_one("span.date, .date")
        docs.append(MEADoc(
            pkid=pkid,
            title=a.get_text(strip=True),
            date=date_el.get_text(strip=True) if date_el else "",
            slug=slug,
            publication_id=publication_id,
            detail_url=urljoin(BASE, href),
        ))
    # Fallback if class names change — anchor on every dtl/ link.
    if not docs:
        for a in soup.select("a[href*='dtl/']"):
            m = PKID_RE.search(a.get("href", ""))
            if not m:
                continue
            pkid = int(m.group(1))
            slug = (m.group(2) or "").strip("/") or f"pr_{pkid}"
            docs.append(MEADoc(
                pkid=pkid,
                title=a.get_text(strip=True),
                date="",
                slug=slug,
                publication_id=publication_id,
                detail_url=urljoin(BASE, a.get("href", "")),
            ))
    return docs


def _fetch_detail(client: HttpClient, pkid: int) -> str | None:
    """Return Markdown body for one press release, or None if too short."""
    r = client.fetch(f"{BASE}{DETAIL}", params={"pkid": pkid, "languageId": 1})
    soup = BeautifulSoup(r.text, "html.parser")
    # Strip script/style/footer-ish junk before markdownification.
    md = markdownify(
        str(soup),
        heading_style="ATX",
        strip=["script", "style", "nav", "header", "footer", "aside",
               "img", "form", "iframe", "button"],
    )
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if len(md) < MIN_BODY_CHARS:
        return None
    return md


def _safe_slug(slug: str, pkid: int) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in slug)
    return f"{pkid}_{safe[:140]}" if safe else f"{pkid}"


def acquire(
    client: HttpClient,
    manifest: Manifest,
    doc: MEADoc,
    dry_run: bool,
) -> tuple[str, int]:
    if manifest.has(doc.detail_url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0
    md = _fetch_detail(client, doc.pkid)
    if md is None:
        return "skipped-empty", 0
    section = PUBLICATION_TYPES.get(doc.publication_id, str(doc.publication_id))
    dest_dir = RepoPaths.cpt_raw("mea") / section
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{_safe_slug(doc.slug, doc.pkid)}.md"
    payload = md.encode("utf-8")
    dest.write_bytes(payload)
    manifest.add(ManifestEntry(
        url=doc.detail_url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        title=doc.title,
        fetched_at=now_iso(),
        extra={
            "pkid": doc.pkid,
            "date": doc.date,
            "publication_id": doc.publication_id,
            "section": section,
        },
    ))
    return "downloaded", len(payload)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Acquire MEA press releases, speeches, etc.")
    p.add_argument("--publication", type=int, action="append", choices=DEFAULT_PUB_IDS,
                   help="MEA publication ID (repeatable). Default: 51,50,53,69")
    p.add_argument("--max-pages", type=int, default=300,
                   help="Cap pages per publication type (default 300 = ~6000 items)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=0.5)
    args = p.parse_args(argv)

    client = _make_client(args.rate)
    pub_ids = args.publication if args.publication else DEFAULT_PUB_IDS

    # Walk each publication type's pagination until empty.
    all_docs: list[MEADoc] = []
    seen_pkids: set[int] = set()
    for pub_id in pub_ids:
        section = PUBLICATION_TYPES.get(pub_id, str(pub_id))
        print(f"Listing MEA {section} (pub_id={pub_id}) ...")
        page = 1
        empty_rounds = 0
        while page <= args.max_pages:
            try:
                items = _list_page(client, pub_id, page)
            except (requests.HTTPError, requests.ConnectionError,
                    requests.Timeout) as e:
                print(f"  page {page} FAILED: {e}", file=sys.stderr)
                break
            if not items:
                empty_rounds += 1
                if empty_rounds >= 2:
                    break
            else:
                empty_rounds = 0
                fresh = [d for d in items if d.pkid not in seen_pkids]
                for d in fresh:
                    seen_pkids.add(d.pkid)
                all_docs.extend(fresh)
                if page % 20 == 0 or page <= 3:
                    print(f"  page {page}: +{len(fresh)} (total {len(all_docs)})")
            page += 1

    print(f"Total candidate docs: {len(all_docs)}")
    if args.limit > 0:
        all_docs = all_docs[: args.limit]
        print(f"After --limit: {len(all_docs)}")

    manifest = Manifest("mea")
    totals = {"downloaded": 0, "cached": 0, "skipped-empty": 0,
              "dry-run": 0, "failed": 0}
    bytes_total = 0
    for i, doc in enumerate(all_docs, start=1):
        if i % 50 == 0 or i <= 3:
            print(f"  [{i:5d}/{len(all_docs)}] {PUBLICATION_TYPES[doc.publication_id]}/{doc.pkid}")
        try:
            status, n = acquire(client, manifest, doc, args.dry_run)
            totals[status] = totals.get(status, 0) + 1
            bytes_total += n
        except (requests.HTTPError, requests.ConnectionError,
                requests.Timeout, PermissionError) as e:
            print(f"     FAILED {doc.pkid}: {type(e).__name__}: {e}", file=sys.stderr)
            totals["failed"] += 1
            continue

    print(f"\nTotal: {totals}")
    print(f"  bytes written: {bytes_total:,}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
