"""Curated-list PDF acquirer — one module, many sources.

For sources whose content lives at a fixed, well-known set of PDF URLs
(annual reports, single-instance docs, government PDFs), a per-source
YAML manifest is much simpler than a dedicated Python module. This
acquirer loads any YAML matching the schema below and downloads every
listed item with the standard manifest/cache/retry pipeline.

YAML schema (`training/data/acquire/curated/<source>.yaml`):

    source: economic_survey          # → data/cpt_raw/economic_survey/
    description: "Free-text description for the manifest."
    items:
      - url: "https://example.gov.in/economic-survey-2024-25-vol1.pdf"
        title: "Economic Survey 2024-25 Volume 1"
        # Optional fields:
        # filename: "es_2024_25_vol1.pdf"  # default: derive from URL
        # year: 2024
        # category: "annual"

CLI:
    python -m training.data.acquire.curated --yaml training/data/acquire/curated/economic_survey.yaml
    python -m training.data.acquire.curated --source economic_survey   # short form, looks up yaml
    python -m training.data.acquire.curated --list                     # list available curated sources
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from ._base import (
    HttpClient,
    Manifest,
    ManifestEntry,
    RepoPaths,
    now_iso,
    url_local_filename,
)

CURATED_DIR = Path(__file__).resolve().parent / "curated"


@dataclass(frozen=True)
class CuratedItem:
    url: str
    title: str
    filename: str
    extra: dict


@dataclass(frozen=True)
class CuratedSource:
    name: str
    description: str
    items: list[CuratedItem]

    @classmethod
    def from_yaml(cls, path: Path) -> "CuratedSource":
        with path.open(encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
        if not data or "source" not in data or "items" not in data:
            raise ValueError(f"{path} missing required keys 'source' or 'items'")
        items: list[CuratedItem] = []
        for raw in data["items"]:
            url = raw["url"]
            title = raw.get("title", "")
            filename = raw.get("filename") or url_local_filename(url, suffix=".pdf")
            extra = {k: v for k, v in raw.items() if k not in {"url", "title", "filename"}}
            items.append(CuratedItem(url=url, title=title, filename=filename, extra=extra))
        return cls(
            name=data["source"],
            description=data.get("description", ""),
            items=items,
        )


def _acquire_one(
    client: HttpClient, manifest: Manifest, item: CuratedItem,
    source: str, dry_run: bool,
) -> str:
    if manifest.has(item.url):
        return "cached"
    if dry_run:
        return "dry-run"
    dest_dir = RepoPaths.cpt_raw(source)
    dest = dest_dir / item.filename
    sha, n = client.download(item.url, dest)
    manifest.add(ManifestEntry(
        url=item.url,
        local_path=str(dest.relative_to(RepoPaths.root())),
        sha256=sha,
        bytes=n,
        title=item.title,
        fetched_at=now_iso(),
        extra={**item.extra, "filename": item.filename},
    ))
    return "downloaded"


def _resolve_yaml(args: argparse.Namespace) -> Path:
    if args.yaml:
        return Path(args.yaml)
    if args.source:
        candidate = CURATED_DIR / f"{args.source}.yaml"
        if not candidate.exists():
            raise FileNotFoundError(
                f"No curated YAML at {candidate}. Available sources: "
                f"{sorted(p.stem for p in CURATED_DIR.glob('*.yaml'))}"
            )
        return candidate
    raise ValueError("Provide --yaml or --source")


def _list_sources() -> int:
    if not CURATED_DIR.exists():
        print(f"No curated/ dir found at {CURATED_DIR}")
        return 1
    yamls = sorted(CURATED_DIR.glob("*.yaml"))
    if not yamls:
        print(f"No YAML files in {CURATED_DIR}")
        return 1
    print(f"Available curated sources ({len(yamls)}):")
    for y in yamls:
        try:
            src = CuratedSource.from_yaml(y)
            print(f"  {src.name:30s}  {len(src.items):>3d} items   {src.description[:60]}")
        except Exception as e:
            print(f"  {y.stem:30s}  ERROR: {e}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download curated PDF list (one YAML = one source).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--yaml", help="Path to a curated YAML file")
    g.add_argument("--source",
                   help="Short name of a YAML under training/data/acquire/curated/")
    g.add_argument("--list", action="store_true",
                   help="List available curated sources and exit")
    p.add_argument("--dry-run", action="store_true", help="Print URLs without downloading")
    p.add_argument("--rate", type=float, default=0.5,
                   help="Min seconds between requests (default 0.5)")
    args = p.parse_args(argv)

    if args.list:
        return _list_sources()
    if not (args.yaml or args.source):
        p.error("provide --yaml, --source, or --list")

    yaml_path = _resolve_yaml(args)
    src = CuratedSource.from_yaml(yaml_path)
    print(f"Curated source: {src.name} ({len(src.items)} items)")
    if src.description:
        print(f"  {src.description}")

    client = HttpClient(rate_seconds=args.rate)
    manifest = Manifest(src.name)
    totals = {"downloaded": 0, "cached": 0, "dry-run": 0, "failed": 0}
    for i, item in enumerate(src.items, start=1):
        title_short = (item.title[:50] + "…") if len(item.title) > 50 else item.title
        print(f"  [{i:3d}/{len(src.items)}] {title_short}")
        try:
            status = _acquire_one(client, manifest, item, src.name, args.dry_run)
            totals[status] += 1
        except Exception as e:
            print(f"      ↳ FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            totals["failed"] += 1
            continue
        print(f"      ↳ {status}")

    print(f"\nTotal: {totals}")
    print(f"Manifest: {manifest.summary()}")
    return 0 if totals["failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
