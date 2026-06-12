"""NDMA (National Disaster Management Authority) PDF acquirer.

NDMA publishes substantial English-language content under ndma.gov.in,
spread across six listing pages we scrape:

- /annual-reports          → 24 PDFs (annual reports 2007-08 onwards)
- /ndma-guidelines         → 45 PDFs (sectoral DM guidelines)
- /policy-plan             → 3 PDFs (laws compendium, Sendai midterm review)
- /technical-documents     → 24 PDFs (HDM manuals, training docs)
- /reports-studies         → 48 PDFs (best practices, case studies)
- /national-dm-policy      → 10 PDFs (NDM Policy 2009 in Indian languages; we filter English)

District DM plans (523 PDFs from /nationalstate-dm-plan) are excluded
by default — too specific and overlap heavily; can be added with
`--include-district-plans` if needed.

This is the **only critical-path UPSC GS-III Disaster Management
source** since `v2-hindi-strategy.md` carries the Hindi-language work
separately and the international Sendai/UNDRR PDFs landed broken
(redirect to HTML pages, not PDFs).

CLI:
    python -m training.data.acquire.ndma
    python -m training.data.acquire.ndma --include-district-plans
    python -m training.data.acquire.ndma --section ndma-guidelines
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

import requests

from ._base import HttpClient, Manifest, ManifestEntry, RepoPaths, now_iso

BASE = "https://ndma.gov.in"
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# Listing pages to scrape. Each has English PDFs at predictable URLs.
SECTIONS = {
    "annual-reports":    "/annual-reports",
    "ndma-guidelines":   "/ndma-guidelines",
    "policy-plan":       "/policy-plan",
    "technical-documents": "/technical-documents",
    "reports-studies":   "/reports-studies",
}

# /national-dm-policy is mostly translations of the 2009 NDM Policy into
# Indian languages — we filter to English-only by URL pattern.
ENGLISH_ONLY_PATTERN = re.compile(
    r"(Hindi|Tamil|Telugu|Bengali|Marathi|Gujarati|Kannada|Malayalam|Punjabi|"
    r"Odia|Oriya|Urdu|Konkani|Kashmiri|Dogri|Sanskrit|Maithili|Sindhi|Manipuri|Nepali|"
    r"Bodo|Assamese|Gujurati)",
    re.I,
)

# Non-content tender / job-listing PDFs to exclude
EXCLUDE_PATTERN = re.compile(
    r"(Tender|Corrigendum|Recruitment|RFP|Vacancy|Job|Internship)",
    re.I,
)


@dataclass(frozen=True)
class NDMAItem:
    url: str
    section: str
    filename: str


def _make_client(rate_seconds: float) -> HttpClient:
    client = HttpClient(rate_seconds=rate_seconds)
    client.session.headers.update({"User-Agent": BROWSER_UA})
    return client


def discover_section(client: HttpClient, section: str, path: str) -> list[NDMAItem]:
    """Fetch one listing page and extract every PDF URL."""
    r = client.fetch(f"{BASE}{path}")
    out: list[NDMAItem] = []
    seen: set[str] = set()
    for href in re.findall(r'href="([^"]+\.pdf[^"]*)"', r.text):
        url = href if href.startswith("http") else f"{BASE}{href}"
        if url in seen:
            continue
        seen.add(url)
        # Filter non-English (per ENGLISH_ONLY_PATTERN) and tender/admin PDFs
        if ENGLISH_ONLY_PATTERN.search(url):
            continue
        if EXCLUDE_PATTERN.search(url):
            continue
        fname = url.rsplit("/", 1)[-1].replace("%20", "_").replace(" ", "_")
        # URL-decode common patterns
        fname = re.sub(r"%[0-9A-Fa-f]{2}", "_", fname)
        out.append(NDMAItem(url=url, section=section, filename=fname))
    return out


def discover_district_plans(client: HttpClient) -> list[NDMAItem]:
    """Optional 523-PDF district DM plans from /nationalstate-dm-plan."""
    r = client.fetch(f"{BASE}/nationalstate-dm-plan")
    out: list[NDMAItem] = []
    seen: set[str] = set()
    for href in re.findall(r'href="([^"]+\.pdf[^"]*)"', r.text):
        url = href if href.startswith("http") else f"{BASE}{href}"
        if url in seen:
            continue
        seen.add(url)
        if EXCLUDE_PATTERN.search(url):
            continue
        # Filename includes the state/district from the URL path
        path_parts = url.split("/PDF/DDMP/")
        if len(path_parts) == 2:
            fname = path_parts[1].replace("/", "__")
        else:
            fname = url.rsplit("/", 1)[-1]
        fname = fname.replace("%20", "_").replace(" ", "_")
        out.append(NDMAItem(url=url, section="district-plans", filename=fname))
    return out


def acquire_item(
    client: HttpClient, manifest: Manifest, item: NDMAItem, dry_run: bool,
) -> tuple[str, int]:
    if manifest.has(item.url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0
    dest_dir = RepoPaths.cpt_raw("ndma") / item.section
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / item.filename
    sha, n = client.download(item.url, dest)
    manifest.add(ManifestEntry(
        url=item.url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=sha,
        bytes=n,
        title=item.filename,
        fetched_at=now_iso(),
        extra={"section": item.section},
    ))
    return "downloaded", n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Acquire NDMA English-language PDFs.")
    p.add_argument("--section", help="Limit to one section (e.g. ndma-guidelines)")
    p.add_argument("--include-district-plans", action="store_true",
                   help="Also fetch all 523 district DM plans from /nationalstate-dm-plan")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=0.5)
    args = p.parse_args(argv)

    client = _make_client(args.rate)
    sections = ({args.section: SECTIONS[args.section]} if args.section
                else SECTIONS)
    if args.section and args.section not in SECTIONS:
        print(f"Unknown section {args.section}; choose from {sorted(SECTIONS)}",
              file=sys.stderr)
        return 1

    all_items: list[NDMAItem] = []
    for sect, path in sections.items():
        try:
            items = discover_section(client, sect, path)
            print(f"  /{sect}: {len(items)} English PDFs")
            all_items.extend(items)
        except (requests.HTTPError, requests.ConnectionError) as e:
            print(f"  /{sect}: discovery FAILED ({type(e).__name__})", file=sys.stderr)

    if args.include_district_plans:
        try:
            district = discover_district_plans(client)
            print(f"  /nationalstate-dm-plan: {len(district)} district DM plans")
            all_items.extend(district)
        except (requests.HTTPError, requests.ConnectionError) as e:
            print(f"  /nationalstate-dm-plan: FAILED ({type(e).__name__})", file=sys.stderr)

    print(f"\nNDMA acquisition — {len(all_items)} PDFs total, dry_run={args.dry_run}")
    manifest = Manifest("ndma")
    totals = {"downloaded": 0, "cached": 0, "dry-run": 0, "failed": 0}
    for i, item in enumerate(all_items, start=1):
        if i % 20 == 0 or i <= 3:
            print(f"  [{i:4d}/{len(all_items)}] {item.section}/{item.filename[:60]}")
        try:
            status, n = acquire_item(client, manifest, item, args.dry_run)
            totals[status] = totals.get(status, 0) + 1
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            print(f"     ↳ FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            totals["failed"] += 1

    print(f"\nTotal: {totals}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
