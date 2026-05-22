"""Parse db-creds.txt into named DSN dicts. Read-only; never writes."""
from __future__ import annotations
import re
from pathlib import Path

CREDS_PATH = Path(__file__).resolve().parent.parent / "db-creds.txt"
_LINE = re.compile(r'^\s*(\w+)\s*=\s*"?([^"\n]+)"?\s*$')


def _load() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in CREDS_PATH.read_text().splitlines():
        m = _LINE.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def dsn(target: str) -> dict[str, str | int]:
    """Return a psycopg2-ready dict for one of: 'upscdev', 'prod'."""
    c = _load()
    if target == "upscdev":
        return dict(host=c["web_host"], port=int(c["web_port"]),
                    dbname=c["web_database"], user=c["web_user"], password=c["web_password"])
    if target == "prod":
        return dict(host=c["app_prod_DB_HOST"], port=int(c["app_prod_DB_PORT"]),
                    dbname=c["app_prod_DB_NAME"], user=c["app_prod_DB_USERNAME"],
                    password=c["app_prod_DB_PASSWORD"])
    raise KeyError(f"Unknown DB target: {target}")
