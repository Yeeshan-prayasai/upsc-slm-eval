"""Stage 1.3 sibling — validate and hash the static UPSC facts lookup.

The JSON itself is hand-curated and committed at `data/upsc_facts.json` (Articles,
Schedules, Acts, Five-Year Plans, schemes, office-holders, commissions). This
script verifies its schema and writes a SHA-256 alongside.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import jsonschema

SCHEMA = {
    "type": "object",
    "required": ["_meta", "articles", "schedules", "acts",
                 "five_year_plans", "schemes", "office_holders", "commissions"],
    "properties": {
        "_meta": {
            "type": "object",
            "required": ["version", "schema", "build_date", "sources", "scope"],
        },
        "articles": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["title", "topic_tokens"],
                "properties": {
                    "title": {"type": "string"},
                    "part":  {"type": "string"},
                    "topic_tokens": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "schedules":       {"type": "object"},
        "acts":            {"type": "object"},
        "five_year_plans": {"type": "object"},
        "schemes":         {"type": "object"},
        "office_holders":  {"type": "object"},
        "commissions":     {"type": "object"},
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path", type=Path, default=Path("data/upsc_facts.json"))
    args = ap.parse_args()

    data = json.loads(args.path.read_text())
    jsonschema.validate(instance=data, schema=SCHEMA)

    sha = hashlib.sha256(args.path.read_bytes()).hexdigest()
    args.path.with_suffix(".sha256").write_text(sha + "\n")

    print(f"[OK] {args.path}")
    print(f"     SHA-256: {sha}")
    print(f"     articles:  {len(data['articles'])}")
    print(f"     schedules: {len(data['schedules'])}")
    print(f"     acts:      {len(data['acts'])}")
    print(f"     plans:     {len(data['five_year_plans']) - 1}")
    print(f"     schemes:   {len(data['schemes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
