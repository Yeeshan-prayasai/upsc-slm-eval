"""IPCC Summary-for-Policymakers (SPM) PDF acquirer.

Downloads the seven canonical SPMs covering the IPCC Sixth Assessment
Report (AR6) cycle + the three Special Reports. SPMs are the
government-approved highest-tier summaries — short, citation-grade,
and the form coaching modules / newspapers / UPSC graders recognize.

URLs were verified live against `ipcc.ch` on 2026-06-05 via HEAD
probes. The SPM file names on ipcc.ch are stable across years (the
files themselves were re-published from their original ~2018-2023
release dates, but the URL paths haven't changed).

References:
- IPCC AR6 cycle (WG1 2021, WG2 + WG3 2022, SYR 2023)
- IPCC Special Reports: SR1.5 (2018), SRCCL (2019), SROCC (2019)

CLI:
    python -m training.data.acquire.ipcc                       # all 7 SPMs
    python -m training.data.acquire.ipcc --only AR6_WG1_SPM
    python -m training.data.acquire.ipcc --dry-run
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from ._base import HttpClient, Manifest, ManifestEntry, RepoPaths, now_iso


@dataclass(frozen=True)
class SPM:
    """One Summary for Policymakers to acquire."""
    key: str          # short identifier for CLI / manifest
    title: str        # human-readable
    url: str
    cycle: str        # "AR6" or "SR" (Special Report)


# All URLs HEAD-verified against ipcc.ch on 2026-06-05.
SPMS: list[SPM] = [
    # AR6 — Sixth Assessment Report (2021-2023)
    SPM(
        key="AR6_WG1_SPM",
        title="AR6 Working Group I — The Physical Science Basis (SPM)",
        url="https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SPM.pdf",
        cycle="AR6",
    ),
    SPM(
        key="AR6_WG2_SPM",
        title="AR6 Working Group II — Impacts, Adaptation and Vulnerability (SPM)",
        url="https://www.ipcc.ch/report/ar6/wg2/downloads/report/"
            "IPCC_AR6_WGII_SummaryForPolicymakers.pdf",
        cycle="AR6",
    ),
    SPM(
        key="AR6_WG3_SPM",
        title="AR6 Working Group III — Mitigation of Climate Change (SPM)",
        url="https://www.ipcc.ch/report/ar6/wg3/downloads/report/"
            "IPCC_AR6_WGIII_SummaryForPolicymakers.pdf",
        cycle="AR6",
    ),
    SPM(
        key="AR6_SYR_SPM",
        title="AR6 Synthesis Report (SPM)",
        url="https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_SPM.pdf",
        cycle="AR6",
    ),
    # Special Reports — released between assessment cycles
    SPM(
        key="SR15_SPM",
        title="Special Report: Global Warming of 1.5°C (SPM)",
        url="https://www.ipcc.ch/site/assets/uploads/sites/2/2022/06/SPM_version_report_LR.pdf",
        cycle="SR",
    ),
    SPM(
        key="SROCC_SPM",
        title="Special Report: Ocean and Cryosphere in a Changing Climate (SPM)",
        url="https://www.ipcc.ch/site/assets/uploads/sites/3/2022/03/01_SROCC_SPM_FINAL.pdf",
        cycle="SR",
    ),
    SPM(
        key="SRCCL_SPM",
        title="Special Report: Climate Change and Land (SPM)",
        url="https://www.ipcc.ch/site/assets/uploads/sites/4/2022/11/SRCCL_SPM.pdf",
        cycle="SR",
    ),
]


def acquire_spm(client: HttpClient, manifest: Manifest, spm: SPM, dry_run: bool) -> str:
    """Download one SPM, returning status string."""
    if manifest.has(spm.url):
        return "cached"
    if dry_run:
        return "dry-run"
    dest_dir = RepoPaths.cpt_raw("ipcc") / spm.cycle
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{spm.key}.pdf"
    sha, n = client.download(spm.url, dest)
    manifest.add(ManifestEntry(
        url=spm.url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=sha,
        bytes=n,
        title=spm.title,
        fetched_at=now_iso(),
        extra={"key": spm.key, "cycle": spm.cycle},
    ))
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download IPCC AR6 + Special Report SPMs.")
    p.add_argument("--only", nargs="+", help="Limit to these SPM keys")
    p.add_argument("--dry-run", action="store_true", help="Print URLs without downloading")
    p.add_argument("--rate", type=float, default=0.5,
                   help="Min seconds between requests (default 0.5)")
    args = p.parse_args(argv)

    client = HttpClient(rate_seconds=args.rate)
    manifest = Manifest("ipcc")
    selected = [s for s in SPMS if (not args.only) or s.key in args.only]
    if args.only:
        unknown = set(args.only) - {s.key for s in SPMS}
        if unknown:
            print(f"WARNING: unknown SPM keys (skipped): {sorted(unknown)}", file=sys.stderr)
    if not selected:
        print("No SPMs selected.", file=sys.stderr)
        return 1

    print(f"IPCC SPM acquisition — {len(selected)} reports, "
          f"dry_run={args.dry_run}")
    totals = {"downloaded": 0, "cached": 0, "dry-run": 0}
    for s in selected:
        print(f"  [{s.key}] {s.title}")
        status = acquire_spm(client, manifest, s, args.dry_run)
        totals[status] += 1
        print(f"    ↳ {status}")

    print(f"\nTotal: {totals}")
    print(f"Manifest: {manifest.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
