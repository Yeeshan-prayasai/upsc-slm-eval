"""2nd Administrative Reforms Commission (2nd ARC) report PDF acquirer.

The 2nd ARC was the second body in independent India tasked with
reviewing the public-administration framework. It produced 15 reports
between 2006-2009 that are the canonical reference for UPSC GS-II
(Governance / Public Administration) and GS-IV (Ethics — Report 4 is
"Ethics in Governance").

All 15 reports are hosted on darpg.gov.in (Department of Administrative
Reforms and Public Grievances). Filenames + URLs verified 2026-06-05
via HEAD probes against the live site.

CLI:
    python -m training.data.acquire.arc2                  # all 15 reports
    python -m training.data.acquire.arc2 --only ethics4 rti_masterkey1
    python -m training.data.acquire.arc2 --only 4 1       # numeric form
    python -m training.data.acquire.arc2 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from ._base import HttpClient, Manifest, ManifestEntry, RepoPaths, now_iso

BASE_URL = "https://darpg.gov.in/sites/default/files/"


@dataclass(frozen=True)
class ARCReport:
    number: int            # 1..15
    filename: str          # under BASE_URL/
    title: str             # official title


# Source: darpg.gov.in/arc-reports listing, scraped 2026-06-05.
# Titles per the 2nd ARC's own report numbering.
REPORTS: list[ARCReport] = [
    ARCReport(1,  "rti_masterkey1.pdf",            "Right to Information — Master Key to Good Governance"),
    ARCReport(2,  "human_capital2.pdf",            "Unlocking Human Capital — Entitlements and Governance"),
    ARCReport(3,  "crisis_management3.pdf",        "Crisis Management — From Despair to Hope"),
    ARCReport(4,  "ethics4.pdf",                   "Ethics in Governance"),
    ARCReport(5,  "public_order5.pdf",             "Public Order — Justice for Each... Peace for All"),
    ARCReport(6,  "local_governance6.pdf",         "Local Governance"),
    ARCReport(7,  "capacity_building7.pdf",        "Capacity Building for Conflict Resolution"),
    ARCReport(8,  "combating_terrorism8.pdf",      "Combating Terrorism — Protecting by Righteousness"),
    ARCReport(9,  "Social_Capital9.pdf",           "Social Capital — A Shared Destiny"),
    ARCReport(10, "personnel_administration10.pdf","Refurbishing of Personnel Administration"),
    ARCReport(11, "promoting_egov11.pdf",          "Promoting e-Governance — The SMART Way Forward"),
    ARCReport(12, "ccadmin12.pdf",                 "Citizen Centric Administration — The Heart of Governance"),
    ARCReport(13, "org_structure_gov13.pdf",       "Organisational Structure of the Government of India"),
    ARCReport(14, "financial_mgmt14.pdf",          "Strengthening Financial Management Systems"),
    ARCReport(15, "sdadmin15.pdf",                 "State and District Administration"),
]


def acquire_report(
    client: HttpClient, manifest: Manifest, report: ARCReport, dry_run: bool
) -> str:
    url = BASE_URL + report.filename
    if manifest.has(url):
        return "cached"
    if dry_run:
        return "dry-run"
    dest_dir = RepoPaths.cpt_raw("arc2")
    dest = dest_dir / f"{report.number:02d}_{report.filename}"
    sha, n = client.download(url, dest)
    manifest.add(ManifestEntry(
        url=url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=sha,
        bytes=n,
        title=f"2nd ARC Report {report.number}: {report.title}",
        fetched_at=now_iso(),
        extra={"report_number": report.number, "filename": report.filename},
    ))
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download 2nd ARC reports (15 PDFs).")
    p.add_argument("--only", nargs="+",
                   help="Limit to these filename stems (e.g. ethics4) "
                        "or report numbers (e.g. 4)")
    p.add_argument("--dry-run", action="store_true", help="Print URLs without downloading")
    p.add_argument("--rate", type=float, default=0.5,
                   help="Min seconds between requests (default 0.5)")
    args = p.parse_args(argv)

    if args.only:
        stems = set(args.only)
        numeric = {s for s in stems if s.isdigit()}
        selected = [
            r for r in REPORTS
            if r.filename.rsplit(".", 1)[0] in stems
            or str(r.number) in numeric
        ]
    else:
        selected = list(REPORTS)
    if not selected:
        print("No reports selected.", file=sys.stderr)
        return 1

    client = HttpClient(rate_seconds=args.rate)
    manifest = Manifest("arc2")
    print(f"2nd ARC acquisition — {len(selected)} reports, dry_run={args.dry_run}")
    totals = {"downloaded": 0, "cached": 0, "dry-run": 0}
    for r in selected:
        print(f"  [{r.number:02d}] {r.title}")
        status = acquire_report(client, manifest, r, args.dry_run)
        totals[status] += 1
        print(f"    ↳ {status}")

    print(f"\nTotal: {totals}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
