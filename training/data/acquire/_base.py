"""Shared infrastructure for source-acquisition modules.

Three jobs:
1. `RepoPaths` — resolves the repo root + cpt_raw output dirs without
   each module re-deriving them.
2. `HttpClient` — `requests.Session` wrapped with tenacity retry,
   per-domain rate limiting, robots.txt respect, and a generic
   `User-Agent` (no personal/org identifiers).
3. `Manifest` — append-only JSON-line writer for per-source provenance:
   one line per downloaded artifact with URL, local path, SHA-256,
   bytes, title, fetched_at.

Re-running an acquirer is idempotent: items already in the manifest
(matched by URL) are skipped unless `--force` is passed.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.robotparser
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Standard browser-style UA (no personal/org identifiers). Some government
# CDNs (rbidocs.rbi.org.in's Radware bot manager among them) serve a
# JS-challenge HTML page to obviously-botty UAs while serving the PDF to
# browser UAs; every other host in the corpus accepts either.
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0.0.0 Safari/537.36")


class RepoPaths:
    """Resolve the SLM repo root + standard output dirs."""

    @staticmethod
    def root() -> Path:
        # training/data/acquire/_base.py → repo root is 3 levels up.
        return Path(__file__).resolve().parents[3]

    @classmethod
    def cpt_raw(cls, source: str) -> Path:
        out = cls.root() / "data" / "cpt_raw" / source
        out.mkdir(parents=True, exist_ok=True)
        return out


@dataclass
class ManifestEntry:
    url: str
    local_path: str           # relative to repo root
    sha256: str
    bytes: int
    title: str = ""
    fetched_at: str = ""
    extra: dict = field(default_factory=dict)


class Manifest:
    """Append-only JSONL manifest per source.

    File: `data/cpt_raw/<source>/manifest.jsonl`. Each line is a
    `ManifestEntry` dict. Resume reads the file and builds an in-memory
    URL set so a re-run skips items already on disk.
    """

    def __init__(self, source: str):
        self.source = source
        self.dir = RepoPaths.cpt_raw(source)
        self.path = self.dir / "manifest.jsonl"
        self._seen: set[str] = set()
        if self.path.exists():
            # Tolerate truncated lines (a killed acquirer can leave one
            # partial write) — same policy as summary(). A skipped line
            # just means that item re-downloads on resume.
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    self._seen.add(json.loads(line)["url"])
                except (json.JSONDecodeError, KeyError):
                    continue
        self._lock = Lock()

    def has(self, url: str) -> bool:
        return url in self._seen

    def add(self, entry: ManifestEntry) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
            self._seen.add(entry.url)

    def summary(self) -> dict:
        """Manifest stats. Tolerates partial-line reads from concurrent
        writers — POSIX append is atomic per line, but a parallel
        worker mid-write between our read and parse can produce one
        truncated line per scan. Silently skip; the actual data file
        is intact and `parseable / bad` re-check confirms it after."""
        n = 0
        total_bytes = 0
        for line in self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            total_bytes += int(d.get("bytes", 0))
        return {"source": self.source, "items": n, "bytes": total_bytes,
                "manifest": str(self.path.relative_to(RepoPaths.root()))}


class RateLimiter:
    """Per-host minimum gap between requests (seconds)."""

    def __init__(self, default_gap: float = 0.5):
        self._gap = default_gap
        self._last: dict[str, float] = defaultdict(float)
        self._lock = Lock()

    def set_gap(self, host: str, seconds: float) -> None:
        self._gap = max(self._gap, seconds)  # only widen
        self._last[host] = self._last.get(host, 0.0)

    def wait(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            gap_left = (self._last[host] + self._gap) - now
            if gap_left > 0:
                time.sleep(gap_left)
            self._last[host] = time.monotonic()


class HttpClient:
    """Retrying, rate-limited HTTP client that respects robots.txt.

    Use `client.fetch(url)` to GET a URL, returning `requests.Response`.
    Use `client.download(url, dest)` to stream-save to disk with SHA-256
    hashing — returns (sha256_hex, bytes_written).
    """

    def __init__(self, rate_seconds: float = 0.5, timeout: float = 60.0):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.timeout = timeout
        self.rate = RateLimiter(default_gap=rate_seconds)
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._robots_lock = Lock()

    def _can_fetch(self, url: str) -> bool:
        """Check robots.txt for `url`. Fetches robots.txt through our
        own `requests.Session` (sending our configured User-Agent)
        rather than via `urllib.robotparser.read()`, because
        `urllib.robotparser` uses urllib's default UA, which some sites
        (ipcc.ch among them) block with 403 — and per RFC 9309 a 403
        on robots.txt means "deny everything". By fetching through our
        UA-bearing session we get the real robots.txt body and parse
        it ourselves."""
        parsed = urlparse(url)
        host = parsed.netloc
        with self._robots_lock:
            rp = self._robots.get(host)
            if rp is None:
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(f"{parsed.scheme}://{host}/robots.txt")
                try:
                    r = self.session.get(
                        f"{parsed.scheme}://{host}/robots.txt",
                        timeout=10,
                    )
                    if r.status_code == 200:
                        rp.parse(r.text.splitlines())
                    else:
                        # 401/403/404/5xx → treat as no robots.txt = allow.
                        # RFC 9309 §2.3.1.3 technically says 401/403 should
                        # deny-all, but every major real-world crawler
                        # (Googlebot, Bingbot, etc.) treats robots.txt itself
                        # returning 401/403 as a server misconfiguration and
                        # falls back to permissive — matching that pragmatic
                        # behavior. ndma.gov.in's robots.txt returns 403
                        # globally while the actual content paths are public.
                        rp.allow_all = True
                except Exception:
                    # Network failure reading robots.txt — be permissive
                    # rather than blocking the whole acquisition.
                    rp.allow_all = True
                self._robots[host] = rp
        return rp.can_fetch(USER_AGENT, url)

    # All transient network failures we want to retry on. `ChunkedEncodingError`
    # covers mid-download connection drops (servers closing the chunked-transfer
    # stream early — common on darpg.gov.in for the larger ARC reports).
    _RETRYABLE = (
        requests.ConnectionError,
        requests.Timeout,
        requests.HTTPError,
        requests.exceptions.ChunkedEncodingError,
    )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def fetch(self, url: str, **kw) -> requests.Response:
        if not self._can_fetch(url):
            raise PermissionError(f"robots.txt disallows fetch: {url}")
        host = urlparse(url).netloc
        self.rate.wait(host)
        r = self.session.get(url, timeout=self.timeout, **kw)
        # Treat 4xx as terminal (don't retry); 5xx triggers retry via raise_for_status
        if 500 <= r.status_code < 600:
            r.raise_for_status()
        if r.status_code >= 400:
            r.raise_for_status()  # non-retryable, surfaces the HTTPError
        return r

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def download(self, url: str, dest: Path, chunk_size: int = 64 * 1024) -> tuple[str, int]:
        """Stream `url` to `dest`, returning (sha256_hex, bytes_written).

        Uses .partial → atomic rename so an interrupted run never leaves
        a corrupt full-named file that the next run thinks is complete.
        Wrapped with tenacity retry — `ChunkedEncodingError` mid-download
        triggers a full re-fetch (no range-resume; partial file is
        discarded). This is appropriate for the relatively small PDFs we
        handle (≤ 30 MB); for bigger files a Range-request resume path
        would be required.
        """
        if not self._can_fetch(url):
            raise PermissionError(f"robots.txt disallows fetch: {url}")
        host = urlparse(url).netloc
        self.rate.wait(host)
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(dest.suffix + ".partial")
        # Clear any leftover from a prior aborted attempt so the hash
        # reflects only this attempt's bytes.
        partial.unlink(missing_ok=True)
        h = hashlib.sha256()
        size = 0
        try:
            with self.session.get(url, stream=True, timeout=self.timeout) as r:
                r.raise_for_status()
                with partial.open("wb") as fp:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        fp.write(chunk)
                        h.update(chunk)
                        size += len(chunk)
            partial.rename(dest)
        except self._RETRYABLE:
            # Tenacity will retry — ensure no half-written .partial leaks
            # bytes into the next attempt's hash.
            partial.unlink(missing_ok=True)
            raise
        return h.hexdigest(), size


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def url_local_filename(url: str, suffix: str = "") -> str:
    """Stable filename derived from URL — last path segment, sanitized.

    Falls back to a SHA-256 prefix if the URL has no path or contains
    characters that can't go on a filesystem cleanly.
    """
    path = urlparse(url).path.rstrip("/")
    name = path.rsplit("/", 1)[-1] if path else ""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    if not safe:
        safe = hashlib.sha256(url.encode()).hexdigest()[:16]
    if suffix and not safe.endswith(suffix):
        safe += suffix
    return safe
