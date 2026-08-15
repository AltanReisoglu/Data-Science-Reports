"""Shared collector base: policy gate, rate limit, disk cache, retry.

There is **no LLM in this layer** (docs/04 §1), for three reasons: it can be
tested offline against fixtures, the measurement is not polluted by the model's
mood, and calling a model on 5,000 signals is a waste.

If one collector falls over the others keep running (docs/04 §10, "source rot")
— but the failure is **not silent**: it is carried on the ``CollectionResult``
and surfaces in the final memo as "this source could not be read".
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import config
import policy as policy_module
from schemas import Signal


@dataclass
class CollectionResult:
    """One collector run, with its successes and its failures."""

    source: str
    signals: list[Signal] = field(default_factory=list)
    error: str | None = None
    requests: int = 0
    cache_hits: int = 0

    @property
    def succeeded(self) -> bool:
        return self.error is None


class Collector:
    """Shared base. Subclasses implement `collect()` only."""

    name: str = "base"

    # Set only when a source publishes an API whose terms permit programmatic
    # access while its robots.txt is written for crawlers. The string is the
    # justification and it is written to the audit log on every request; the
    # blocklist still wins over it. Leave as None unless you can cite the terms.
    robots_exemption: str | None = None

    def __init__(
        self,
        *,
        source_policy: policy_module.SourcePolicy | None = None,
        fetcher: Callable[..., tuple[int, str]] | None = None,
        use_cache: bool = True,
    ) -> None:
        self.policy = source_policy or policy_module.DEFAULT
        self._fetcher = fetcher  # injected by tests to serve fixtures
        self.use_cache = use_cache
        self.requests = 0
        self.cache_hits = 0

    # ------------------------------------------------------------ interface

    def collect(self, *, query: str, days: int) -> CollectionResult:  # pragma: no cover
        raise NotImplementedError

    def run(self, *, query: str, days: int) -> CollectionResult:
        """Wraps `collect()`: failures are caught here and carried upward."""
        self.requests = 0
        self.cache_hits = 0
        try:
            result = self.collect(query=query, days=days)
        except Exception as e:
            result = CollectionResult(source=self.name, error=f"{type(e).__name__}: {e}")
        result.requests = self.requests
        result.cache_hits = self.cache_hits
        return result

    # ------------------------------------------------------------ HTTP

    def headers(self) -> dict[str, str]:
        return {"User-Agent": config.USER_AGENT, "Accept-Encoding": "gzip, deflate"}

    def fetch(self, url: str, params: dict[str, Any] | None = None) -> str:
        """A single GET: through the policy gate, cached, with retries."""
        full_url = _with_params(url, params)

        if not self.policy.is_allowed(
            full_url, source=self.name, exemption=self.robots_exemption
        ):
            raise PermissionError(f"policy refused: {full_url}")

        if self.use_cache:
            hit = self._read_cache(full_url)
            if hit is not None:
                self.cache_hits += 1
                self.policy.record_response(self.name, full_url, 200, len(hit), cached=True)
                return hit

        status, text = self._fetch_with_retry(full_url, params)
        self.requests += 1
        self.policy.record_response(self.name, full_url, status, len(text))
        if status != 200:
            raise RuntimeError(f"{self.name}: HTTP {status} — {full_url}")
        if self.use_cache:
            self._write_cache(full_url, text)
        return text

    def fetch_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return json.loads(self.fetch(url, params))

    def _fetch_with_retry(self, url: str, params: dict[str, Any] | None) -> tuple[int, str]:
        if self._fetcher is not None:
            return self._fetcher(url, params)

        import httpx

        last_error: Exception | None = None
        for attempt in range(3):
            self.policy.wait(self.name)
            try:
                response = httpx.get(
                    url,
                    headers=self.headers(),
                    timeout=config.REQUEST_TIMEOUT,
                    follow_redirects=True,
                )
                # 429/5xx are treated as transient; 4xx is permanent, no retry.
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return response.status_code, response.text
            except Exception as e:  # network failure -> back off and retry
                last_error = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"{self.name}: network failure — {last_error}")

    # ------------------------------------------------------------ cache

    def _cache_path(self, url: str) -> Path:
        directory = config.CACHE / self.name
        directory.mkdir(parents=True, exist_ok=True)
        return directory / (hashlib.sha1(url.encode()).hexdigest()[:20] + ".json")

    def _read_cache(self, url: str) -> str | None:
        path = self._cache_path(url)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if time.time() - entry.get("retrieved_at", 0) > config.CACHE_TTL_SECONDS:
            return None
        return entry.get("body")

    def _write_cache(self, url: str, body: str) -> None:
        try:
            self._cache_path(url).write_text(
                json.dumps({"retrieved_at": time.time(), "url": url, "body": body}),
                encoding="utf-8",
            )
        except OSError:
            pass


def _with_params(url: str, params: dict[str, Any] | None) -> str:
    if not params:
        return url
    from urllib.parse import urlencode

    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode(params)}"
