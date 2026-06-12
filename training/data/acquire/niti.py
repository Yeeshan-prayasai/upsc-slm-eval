"""NITI Aayog publications — division reports, research papers, working
papers, policy papers, and Arth-NITI quarterly insights.

NITI publishes ~100-150 PDFs across five categories at
`https://www.niti.gov.in/publications/<category>`. Each category
lists 10-11 PDFs per page with a `?page=N` query-string pager.
PDFs themselves are served from `/sites/default/files/<yyyy-mm>/...pdf`.

Content licence: Government of India Open Data Policy (NDSAP) —
public-domain for non-commercial research use; manifest records
the source URL per file.

CLI:
    python -m training.data.acquire.niti                # all 5 categories
    python -m training.data.acquire.niti --category research-paper
    python -m training.data.acquire.niti --limit 5      # smoke-test
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ._base import HttpClient, Manifest, ManifestEntry, RepoPaths, now_iso

BASE = "https://www.niti.gov.in"
CATEGORIES = [
    "division-reports",
    "research-paper",
    "working-papers",
    "policy-and-research/policy-paper",
    "arth-niti",
]


@dataclass(frozen=True)
class NITIDoc:
    pdf_url: str
    title: str
    category: str


def _slug_from_url(url: str) -> str:
    """Last path segment of the PDF URL, stripped to a safe filename."""
    name = url.rstrip("/").rsplit("/", 1)[-1]
    name = name.split("?", 1)[0]
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return safe or "untitled.pdf"


def _list_category(
    client: HttpClient, category: str, max_pages: int = 50
) -> list[NITIDoc]:
    """Walk a category's paginated listing, harvesting PDF URLs + titles."""
    docs: list[NITIDoc] = []
    seen_urls: set[str] = set()
    for page in range(max_pages):
        url = f"{BASE}/publications/{category}?page={page}"
        r = client.fetch(url)
        soup = BeautifulSoup(r.text, "html.parser")
        pdfs = soup.find_all("a", href=re.compile(r"\.pdf(?:$|\?)", re.I))
        page_new = 0
        for a in pdfs:
            href = a.get("href")
            if not href:
                continue
            pdf_url = urljoin(BASE, href)
            if pdf_url in seen_urls:
                continue
            seen_urls.add(pdf_url)

            # Walk up to find the title-bearing container.
            container = a
            for _ in range(5):
                container = container.parent
                if container is None:
                    break
                if container.name and len(container.get_text(strip=True)) > 20:
                    break
            title = (
                container.get_text(" ", strip=True)[:280]
                if container
                else _slug_from_url(pdf_url)
            )
            docs.append(NITIDoc(pdf_url=pdf_url, title=title, category=category))
            page_new += 1
        if page_new == 0:
            break
    return docs


def acquire(
    client: HttpClient,
    manifest: Manifest,
    doc: NITIDoc,
    dry_run: bool,
) -> tuple[str, int]:
    if manifest.has(doc.pdf_url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0

    dest_dir = RepoPaths.cpt_raw("niti") / doc.category.replace("/", "_")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / _slug_from_url(doc.pdf_url)
    if not dest.suffix:
        dest = dest.with_suffix(".pdf")

    sha, n = client.download(doc.pdf_url, dest)
    manifest.add(ManifestEntry(
        url=doc.pdf_url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=sha,
        bytes=n,
        title=doc.title,
        fetched_at=now_iso(),
        extra={"category": doc.category},
    ))
    return "downloaded", n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Acquire NITI Aayog publications (PDFs).")
    p.add_argument("--category", choices=CATEGORIES + ["all"], default="all")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap total downloads (0 = all)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=0.6,
                   help="Min seconds between requests per host")
    args = p.parse_args(argv)

    client = HttpClient(rate_seconds=args.rate)
    chosen = CATEGORIES if args.category == "all" else [args.category]

    all_docs: list[NITIDoc] = []
    for cat in chosen:
        print(f"Listing NITI {cat} ...")
        docs = _list_category(client, cat)
        print(f"  {len(docs)} PDFs")
        all_docs.extend(docs)
    print(f"Total candidate PDFs: {len(all_docs)}")
    if args.limit > 0:
        all_docs = all_docs[: args.limit]
        print(f"After --limit: {len(all_docs)}")

    manifest = Manifest("niti")
    totals = {"downloaded": 0, "cached": 0, "dry-run": 0, "failed": 0}
    bytes_total = 0
    for i, doc in enumerate(all_docs, start=1):
        print(f"  [{i:3d}/{len(all_docs)}] {doc.category}/{_slug_from_url(doc.pdf_url)[:60]}")
        try:
            status, n = acquire(client, manifest, doc, args.dry_run)
            totals[status] = totals.get(status, 0) + 1
            bytes_total += n
        except (requests.HTTPError, requests.ConnectionError,
                requests.Timeout, PermissionError) as e:
            print(f"     FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            totals["failed"] += 1
            continue

    print(f"\nTotal: {totals}")
    print(f"  bytes written: {bytes_total:,}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
