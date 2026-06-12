"""Union Budget documents from indiabudget.gov.in.

Per fiscal-year landing page (`/budget<FY>/index.php`) and the current
year's home page, each FY has 330-345 PDFs covering:
- Speech, Budget Highlights, Budget at a Glance
- Annual Financial Statement (AFS)
- Receipts/Expenditure breakdowns, FRBM statements
- Per-ministry detailed grant demands (table-heavy)

We download all PDFs and let `clean.py`'s FineWeb document floor drop
the table-heavy low-text pages downstream.

Indiabudget.gov.in requires a browser User-Agent — the default
`corpus-build/0.1` UA gets 503s. We patch the client session
accordingly.

CLI:
    python -m training.data.acquire.budget                # all configured FYs
    python -m training.data.acquire.budget --fy 2023-24
    python -m training.data.acquire.budget --limit 10
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

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
BASE = "https://www.indiabudget.gov.in"
# Current FY served at root; prior FYs at /budget<FY>/. 2020-21 and earlier
# are gone from the modern domain (HTTP 503 on /budget2020-21/).
FY_LIST_DEFAULT = ["2025-26", "2024-25", "2023-24", "2022-23", "2021-22"]


@dataclass(frozen=True)
class BudgetDoc:
    pdf_url: str
    fy: str
    title: str


def _make_client(rate_seconds: float) -> HttpClient:
    client = HttpClient(rate_seconds=rate_seconds)
    client.session.headers.update({
        "User-Agent": BROWSER_UA,
        "Accept-Language": "en-US,en;q=0.9",
    })
    return client


def _fy_landing_url(fy: str) -> str:
    # Current FY (2025-26) is served at root; older years under /budget<FY>/.
    if fy == FY_LIST_DEFAULT[0]:
        return f"{BASE}/"
    return f"{BASE}/budget{fy}/index.php"


def _slug(url: str, fy: str) -> str:
    name = url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return f"{fy}_{safe}" if safe else f"{fy}_untitled.pdf"


def _list_fy(client: HttpClient, fy: str) -> list[BudgetDoc]:
    """Crawl one fiscal year's landing page; collect every PDF link."""
    landing = _fy_landing_url(fy)
    r = client.fetch(landing)
    soup = BeautifulSoup(r.text, "html.parser")
    docs: list[BudgetDoc] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=re.compile(r"\.pdf(?:$|\?)", re.I)):
        href = a.get("href")
        if not href:
            continue
        pdf_url = urljoin(landing, href)
        if pdf_url in seen:
            continue
        seen.add(pdf_url)
        title = (a.get("title") or a.get_text(" ", strip=True) or "")[:280]
        docs.append(BudgetDoc(pdf_url=pdf_url, fy=fy, title=title))
    return docs


def acquire(
    client: HttpClient,
    manifest: Manifest,
    doc: BudgetDoc,
    dry_run: bool,
) -> tuple[str, int]:
    if manifest.has(doc.pdf_url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0
    dest_dir = RepoPaths.cpt_raw("budget") / doc.fy
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / _slug(doc.pdf_url, doc.fy)
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
        extra={"fy": doc.fy},
    ))
    return "downloaded", n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Acquire Union Budget PDFs (indiabudget.gov.in).")
    p.add_argument("--fy", action="append",
                   help="Fiscal year(s) e.g. 2024-25; repeat for multiple. Default: all configured.")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=1.0,
                   help="indiabudget.gov.in is slow; default 1 req/s")
    args = p.parse_args(argv)

    client = _make_client(args.rate)
    fys = args.fy if args.fy else FY_LIST_DEFAULT

    all_docs: list[BudgetDoc] = []
    for fy in fys:
        print(f"Listing FY {fy} ...")
        try:
            docs = _list_fy(client, fy)
            print(f"  {len(docs)} PDFs")
            all_docs.extend(docs)
        except (requests.HTTPError, requests.ConnectionError,
                requests.Timeout) as e:
            print(f"  FAILED listing FY {fy}: {e}", file=sys.stderr)
    print(f"Total candidate PDFs: {len(all_docs)}")
    if args.limit > 0:
        all_docs = all_docs[: args.limit]
        print(f"After --limit: {len(all_docs)}")

    manifest = Manifest("budget")
    totals = {"downloaded": 0, "cached": 0, "dry-run": 0, "failed": 0}
    bytes_total = 0
    for i, doc in enumerate(all_docs, start=1):
        if i % 50 == 0 or i <= 3:
            print(f"  [{i:4d}/{len(all_docs)}] {doc.fy}/{_slug(doc.pdf_url, doc.fy)[:60]}")
        try:
            status, n = acquire(client, manifest, doc, args.dry_run)
            totals[status] = totals.get(status, 0) + 1
            bytes_total += n
        except (requests.HTTPError, requests.ConnectionError,
                requests.Timeout, PermissionError) as e:
            print(f"     FAILED {doc.pdf_url}: {type(e).__name__}: {e}", file=sys.stderr)
            totals["failed"] += 1
            continue

    print(f"\nTotal: {totals}")
    print(f"  bytes written: {bytes_total:,}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
