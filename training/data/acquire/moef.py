"""Ministry of Environment, Forest and Climate Change (MoEFCC) — PDF docs.

MoEFCC publishes its annual reports, publications, state-of-environment
reports, acts, rules, and notifications across a set of hub pages at
`https://moef.gov.in/<slug>`. Each hub has a flat list of PDF links.
No pagination on the hubs we sampled.

Total: ~130 unique PDFs covering Environment, biodiversity, forestry,
climate, and notifications. Closes the largest GS3 Environment gap
(per docx, Shankar IAS is the named text we lack but can't acquire;
MoEFCC primary sources are the next-best authoritative layer).

Content licence: Government of India Open Data Policy.

CLI:
    python -m training.data.acquire.moef            # all hubs
    python -m training.data.acquire.moef --hub annual-reports
    python -m training.data.acquire.moef --limit 5  # smoke
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

BASE = "https://moef.gov.in"
HUBS = [
    "annual-reports",
    "publications",
    "publication",
    "publications-2",
    "publications-3",
    "publications-4",
    "state-of-environment-report",
    "various-reports",
    "acts-and-rules",
    "policy-law-pl",
    "wildlife-notification",
    "guidelinesdocuments",
    "guidelinesnotifications",
    "orders-and-notification-2",
    "national-reports-submitted-to-unccd",
]


@dataclass(frozen=True)
class MoEFDoc:
    pdf_url: str
    hub: str
    title: str


def _slug(url: str) -> str:
    name = url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return safe or "untitled.pdf"


def _list_hub(client: HttpClient, hub: str) -> list[MoEFDoc]:
    r = client.fetch(f"{BASE}/{hub}")
    soup = BeautifulSoup(r.text, "html.parser")
    docs: list[MoEFDoc] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=re.compile(r"\.pdf(?:$|\?)", re.I)):
        href = a.get("href")
        if not href:
            continue
        url = urljoin(BASE, href)
        if url in seen:
            continue
        seen.add(url)
        title = (a.get("title") or a.get_text(" ", strip=True) or "")[:280]
        docs.append(MoEFDoc(pdf_url=url, hub=hub, title=title))
    return docs


def acquire(
    client: HttpClient,
    manifest: Manifest,
    doc: MoEFDoc,
    dry_run: bool,
) -> tuple[str, int]:
    if manifest.has(doc.pdf_url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0
    dest_dir = RepoPaths.cpt_raw("moef") / doc.hub
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / _slug(doc.pdf_url)
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
        extra={"hub": doc.hub},
    ))
    return "downloaded", n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Acquire MoEFCC PDFs.")
    p.add_argument("--hub", action="append",
                   help="Hub slug (repeatable). Default: all configured.")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=0.7)
    args = p.parse_args(argv)

    client = HttpClient(rate_seconds=args.rate)
    hubs = args.hub if args.hub else HUBS

    all_docs: list[MoEFDoc] = []
    seen_urls: set[str] = set()
    for hub in hubs:
        print(f"Listing MoEFCC {hub} ...")
        try:
            docs = _list_hub(client, hub)
            print(f"  {len(docs)} PDFs")
            for d in docs:
                if d.pdf_url not in seen_urls:
                    seen_urls.add(d.pdf_url)
                    all_docs.append(d)
        except (requests.HTTPError, requests.ConnectionError,
                requests.Timeout) as e:
            print(f"  FAILED listing {hub}: {e}", file=sys.stderr)
    print(f"Total unique PDFs (dedup across hubs): {len(all_docs)}")
    if args.limit > 0:
        all_docs = all_docs[: args.limit]

    manifest = Manifest("moef")
    totals = {"downloaded": 0, "cached": 0, "dry-run": 0, "failed": 0}
    bytes_total = 0
    for i, doc in enumerate(all_docs, start=1):
        print(f"  [{i:3d}/{len(all_docs)}] {doc.hub}/{_slug(doc.pdf_url)[:60]}")
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
