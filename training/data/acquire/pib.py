"""PIB (Press Information Bureau) — multi-ministry press releases via
direct PRID iteration.

PIB's filterable archive forms (`AllRelease.aspx`, `erelease.aspx`,
`AdvanceSearch.aspx`) all proved unscrapeable in 2026:
- The main WebForms `__doPostBack` POST silently ignores dropdown
  value changes (server-side viewstate validation accepts the request
  but doesn't re-render with the filter applied).
- AdvanceSearch requires a hard CAPTCHA.
- GET param variants (`MinId=N&Year=Y`) return the default top-11 only.

The one working path is **direct PRID iteration**: fetch each
`PressReleasePage.aspx?PRID=N&reg=3&lang=1` and read the ministry off
the rendered detail page. PRIDs are monotonically increasing integers,
so a numerical range maps roughly to a date window.

This acquirer:
1. Iterates PRIDs in a configurable range
2. Fetches each one, extracts ministry/date/title/body via BeautifulSoup
3. Keeps releases whose ministry is in TARGET_MINISTRIES (default: 20
   UPSC-relevant ministries we don't have direct sources for)
4. Records every visited PRID in the manifest (kept or skipped) so the
   acquirer is idempotent and resumable

Roughly 150 PRIDs/day are issued, so the PRID-range for an N-month
window is ~150 × 30 × N. We default to ~12 months (~71K PRIDs).
Retention rate is ~30% (20 target ministries / ~67 active ministries).

Content licence: Government of India Open Data Policy — public.

CLI:
    python -m training.data.acquire.pib                      # default 12mo
    python -m training.data.acquire.pib --prid-start 2240000 --prid-end 2271000
    python -m training.data.acquire.pib --limit 50           # smoke
    python -m training.data.acquire.pib --all-ministries     # don't filter
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

from ._base import HttpClient, Manifest, ManifestEntry, RepoPaths, now_iso

BASE = "https://pib.gov.in"
DETAIL_URL = f"{BASE}/PressReleasePage.aspx"

# PRID range that maps to roughly the last 12 months at acquirer-author
# time (Jun 2026). Adjust via CLI if you want a longer window.
# Empirically: PRID 2,271,000 ≈ 10 Jun 2026; PRID 2,200,000 ≈ Mar 2026;
# PRID 2,150,000 ≈ Nov 2025; PRID 2,100,000 ≈ Aug 2025.
DEFAULT_PRID_START = 2_200_000
DEFAULT_PRID_END = 2_271_000

# UPSC-relevant ministries we DON'T have direct sources for. We skip the
# ones we already have (MEA / MoEFCC / Space=ISRO / Defence=DRDO /
# NITI / ARC / NDMA covered by their dedicated acquirers). String match
# is on the rendered MinistryNameSubhead text.
TARGET_MINISTRIES = {
    "Prime Minister's Office",
    "Cabinet",
    "Cabinet Committee on Economic Affairs (CCEA)",
    "Cabinet Secretariat",
    "Ministry of Finance",
    "Ministry of Health and Family Welfare",
    "Ministry of Education",
    "Ministry of Agriculture & Farmers Welfare",
    "Ministry of Rural Development",
    "Ministry of Social Justice & Empowerment",
    "Ministry of Women and Child Development",
    "Ministry of Tribal Affairs",
    "Ministry of Minority Affairs",
    "Ministry of Statistics & Programme Implementation",
    "Ministry of Science & Technology",
    "Ministry of Earth Sciences",
    "Ministry of New and Renewable Energy",
    "Ministry of Electronics & IT",
    "Ministry of Information & Broadcasting",
    "Ministry of Commerce & Industry",
    "Ministry of Housing & Urban Affairs",
    "Ministry of Power",
    "Ministry of Coal",
    "Ministry of Petroleum & Natural Gas",
    "Ministry of Railways",
    "Ministry of Civil Aviation",
    "Ministry of Communications",
    "Ministry of Culture",
    "Ministry of Labour & Employment",
    "Ministry of Personnel, Public Grievances & Pensions",
    "Ministry of Law and Justice",
    "Ministry of Home Affairs",
    "Ministry of Jal Shakti",
    "Ministry of Cooperation",
    "Ministry of Fisheries, Animal Husbandry & Dairying",
    "Ministry of Food Processing Industries",
    "Ministry of Mines",
    "Ministry of Steel",
    "Ministry of Heavy Industries",
    "Ministry of Road Transport & Highways",
    "Ministry of Skill Development and Entrepreneurship",
    # PIB's exact ministry string (not "Ministry of MSME"); the abbreviated
    # form matched zero releases. "PM Speech" removed — covered by the
    # Prime Minister's Office entry; as a label it matched nothing.
    "Ministry of Micro,Small & Medium Enterprises",
}

MIN_BODY_CHARS = 200
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


@dataclass(frozen=True)
class PIBRelease:
    prid: int
    ministry: str
    date_str: str
    title: str


def _make_client(rate_seconds: float) -> HttpClient:
    client = HttpClient(rate_seconds=rate_seconds)
    client.session.headers.update({"User-Agent": BROWSER_UA})
    return client


def _parse_release(html: str) -> "tuple[PIBRelease | None, str | None]":
    """Extract (release_meta, body_markdown) from a PRID detail page.
    Returns (None, None) on a 'release not found' / unparseable page."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("div.innner-page-main-about-us-content-right-part")
    if container is None:
        return None, None
    ministry_el = soup.select_one("div.MinistryNameSubhead")
    date_el = soup.select_one("div.ReleaseDateSubHeaddateTime")
    title_el = soup.select_one("h2") or soup.select_one("h1")
    ministry = ministry_el.get_text(strip=True) if ministry_el else ""
    date_str = (
        date_el.get_text(" ", strip=True).replace("Posted On:", "").strip()
        if date_el else ""
    )
    title = title_el.get_text(strip=True) if title_el else ""
    # Body markdown — strip noise children
    md = markdownify(
        str(container), heading_style="ATX",
        strip=["script", "style", "nav", "header", "footer", "aside",
               "img", "form", "iframe", "button"],
    )
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if len(md) < MIN_BODY_CHARS:
        return None, None
    return PIBRelease(prid=0, ministry=ministry, date_str=date_str, title=title), md


def _ministry_slug(ministry: str) -> str:
    """Filesystem-safe slug for the ministry sub-dir."""
    s = ministry.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "unknown"


def acquire_prid(
    client: HttpClient,
    manifest: Manifest,
    prid: int,
    target_ministries: "set[str] | None",
    dry_run: bool,
) -> tuple[str, int]:
    url = f"{DETAIL_URL}?PRID={prid}&reg=3&lang=1"
    if manifest.has(url):
        return "cached", 0
    if dry_run:
        return "dry-run", 0

    try:
        r = client.fetch(url)
    except requests.HTTPError as e:
        # 404 / 4xx = PRID gap or invalid. Record in manifest so we
        # don't re-fetch on rerun.
        manifest.add(ManifestEntry(
            url=url, local_path="", sha256="", bytes=0, title="",
            fetched_at=now_iso(),
            extra={"prid": prid, "status": "http-error",
                   "code": getattr(e.response, "status_code", None)},
        ))
        return "http-error", 0

    parsed, md = _parse_release(r.text)
    if parsed is None:
        manifest.add(ManifestEntry(
            url=url, local_path="", sha256="", bytes=0, title="",
            fetched_at=now_iso(),
            extra={"prid": prid, "status": "unparseable"},
        ))
        return "unparseable", 0

    if target_ministries is not None and parsed.ministry not in target_ministries:
        manifest.add(ManifestEntry(
            url=url, local_path="", sha256="", bytes=0,
            title=parsed.title, fetched_at=now_iso(),
            extra={"prid": prid, "status": "skipped-ministry",
                   "ministry": parsed.ministry},
        ))
        return "skipped-ministry", 0

    ministry_dir = RepoPaths.cpt_raw("pib") / _ministry_slug(parsed.ministry)
    ministry_dir.mkdir(parents=True, exist_ok=True)
    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", parsed.title)[:120] or "untitled"
    dest = ministry_dir / f"{prid}_{safe_title}.md"
    payload = md.encode("utf-8")
    dest.write_bytes(payload)
    manifest.add(ManifestEntry(
        url=url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        title=parsed.title,
        fetched_at=now_iso(),
        extra={"prid": prid, "ministry": parsed.ministry,
               "date": parsed.date_str, "status": "downloaded"},
    ))
    return "downloaded", len(payload)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Acquire PIB press releases via PRID iteration.")
    p.add_argument("--prid-start", type=int, default=DEFAULT_PRID_START,
                   help=f"First PRID to fetch (default {DEFAULT_PRID_START})")
    p.add_argument("--prid-end", type=int, default=DEFAULT_PRID_END,
                   help=f"Last PRID (inclusive, default {DEFAULT_PRID_END})")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap total fetches (0 = all)")
    p.add_argument("--all-ministries", action="store_true",
                   help="Keep releases from ALL ministries (skip filter)")
    p.add_argument("--ministry", action="append",
                   help="Override target-ministry set; repeatable")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=float, default=0.4)
    args = p.parse_args(argv)

    client = _make_client(args.rate)
    manifest = Manifest("pib")

    if args.all_ministries:
        targets: "set[str] | None" = None
    elif args.ministry:
        targets = set(args.ministry)
    else:
        targets = TARGET_MINISTRIES

    prids = list(range(args.prid_start, args.prid_end + 1))
    if args.limit > 0:
        prids = prids[: args.limit]
    print(f"PIB acquire: PRID {args.prid_start}-{args.prid_end} = {len(prids)} fetches "
          f"(targets: {'ALL' if targets is None else len(targets)} ministries)")
    if not targets is None:
        print(f"  target ministries: {sorted(targets)[:5]} ...")

    totals = {"downloaded": 0, "cached": 0, "skipped-ministry": 0,
              "unparseable": 0, "http-error": 0, "dry-run": 0}
    bytes_total = 0
    for i, prid in enumerate(prids, start=1):
        if i % 200 == 0 or i <= 3:
            print(f"  [{i:6d}/{len(prids)}] PRID={prid}  "
                  f"downloaded={totals['downloaded']} "
                  f"skipped={totals['skipped-ministry']} "
                  f"4xx={totals['http-error']}")
        try:
            status, n = acquire_prid(client, manifest, prid, targets, args.dry_run)
            totals[status] = totals.get(status, 0) + 1
            bytes_total += n
        except (requests.ConnectionError, requests.Timeout,
                PermissionError) as e:
            print(f"     PRID={prid} ERR: {type(e).__name__}: {e}",
                  file=sys.stderr)
            totals.setdefault("transport-error", 0)
            totals["transport-error"] += 1
            continue

    print(f"\nTotal: {totals}")
    print(f"  bytes written: {bytes_total:,}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
