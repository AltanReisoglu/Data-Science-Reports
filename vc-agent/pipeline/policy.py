"""Source policy — the **single gate** to the outside world.

The code enforcement of docs/03 §11 and docs/04 §6. Collectors never call HTTP
directly; every request passes through here, which guarantees three things:

1. **Blocklist** — sites that require a login or whose ToS forbids scraping get
   an unconditional ``False``. robots.txt is not even consulted; this is a legal
   decision, not a technical check. A test guards it.
2. **robots.txt** — fetched and cached for every remaining host. If it cannot be
   retrieved, access is treated as allowed (RFC 9309 behaviour) but the fact is
   written to the audit log rather than passed over in silence.
3. **Rate limiting and auditing** — a minimum interval per source, and a JSONL
   record of every call, so "where did this score come from" always has an answer.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlparse

import config


@dataclass
class AuditRecord:
    timestamp: str
    source: str
    url: str
    allowed: bool
    reason: str
    status: int | None = None
    bytes: int | None = None
    cached: bool = False
    # Agent actions only: which arguments were passed and a digest of their
    # values, never the values. See `record_agent_action`.
    keys: list[str] | None = None
    digest: str | None = None
    agent: str | None = None
    session: str | None = None


class SourcePolicy:
    """The gate every outbound call passes through.

    ``robots_fetcher`` is injectable so tests can run without touching the network.
    """

    def __init__(
        self,
        *,
        audit_path=None,
        robots_fetcher: Callable[[str], str | None] | None = None,
        rate_limited: bool = True,
    ) -> None:
        self.audit_path = audit_path or config.AUDIT_LOG
        self._robots_fetcher = robots_fetcher or self._fetch_robots_over_network
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()
        self._rate_limited = rate_limited
        self.records: list[AuditRecord] = []

    # ------------------------------------------------------------ permission

    def is_blocked(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return any(host == b or host.endswith("." + b) for b in config.BLOCKLIST)

    def is_allowed(self, url: str, *, source: str = "?", exemption: str | None = None) -> bool:
        if self.is_blocked(url):
            self._record(AuditRecord(
                timestamp=_utc_now(), source=source, url=url,
                allowed=False, reason="blocklist",
            ))
            return False

        if exemption:
            # A documented API whose terms permit programmatic access, on a host
            # whose robots.txt addresses crawlers rather than API clients
            # (arXiv is the live case). This is never silent: the justification
            # is written into the audit log on every single request, so an
            # exemption can be reviewed later instead of being discovered later.
            self._record(AuditRecord(
                timestamp=_utc_now(), source=source, url=url,
                allowed=True, reason=f"api_exemption: {exemption}",
            ))
            return True

        rp = self._robots_for(url)
        if rp is None:
            self._record(AuditRecord(
                timestamp=_utc_now(), source=source, url=url,
                allowed=True, reason="robots_unavailable",
            ))
            return True

        allowed = rp.can_fetch(config.USER_AGENT, url)
        self._record(AuditRecord(
            timestamp=_utc_now(), source=source, url=url, allowed=allowed,
            reason="robots_allow" if allowed else "robots_disallow",
        ))
        return allowed

    # ------------------------------------------------------------ rate limit

    def wait(self, source: str) -> float:
        """Honour the per-source minimum interval. Returns seconds slept."""
        if not self._rate_limited:
            return 0.0
        interval = config.RATE_LIMITS.get(source, config.RATE_LIMITS["default"])
        with self._lock:
            last = self._last_request.get(source, 0.0)
            elapsed = time.monotonic() - last
            sleep_for = max(0.0, interval - elapsed)
            if sleep_for:
                time.sleep(sleep_for)
            self._last_request[source] = time.monotonic()
        return sleep_for

    # ------------------------------------------------------------ auditing

    def record_response(
        self, source: str, url: str, status: int | None, size: int | None, *, cached: bool = False
    ) -> None:
        self._record(AuditRecord(
            timestamp=_utc_now(), source=source, url=url, allowed=True,
            reason="response", status=status, bytes=size, cached=cached,
        ))

    def record_agent_action(
        self,
        *,
        tool: str,
        arguments: dict,
        result_size: int,
        outcome: str = "executed",
        agent: str | None = None,
        session: str | None = None,
    ) -> None:
        """Record a decision an agent made, as opposed to a request a collector sent.

        The HTTP records answer "what did we fetch". These answer "what did an
        agent choose to do", which is the half of docs/04 §6 that the collector
        layer cannot see. Fed by `observability.EventCapture`.

        **Metadata only — argument values never reach this file.** That is
        OpenClaw's rule for its audit ledger (docs/13 §10.2) and it is the right
        one: this log is long-lived, unscoped and survives session deletion, so
        anything written here outlives the conversation it came from. A tool
        argument can carry a company name, a URL, a person. The parameter *names*
        and a digest of the values answer "which call was this" and let two
        records be compared, without the ledger becoming a second transcript.

        The full arguments are not lost. They live in the session transcript,
        which is scoped to one conversation and is deleted with it.
        """
        keys = sorted(str(k) for k in arguments)
        blob = json.dumps(arguments, ensure_ascii=False, default=str, sort_keys=True)
        self._record(AuditRecord(
            timestamp=_utc_now(),
            source=f"agent:{tool}",
            url="",
            allowed=outcome not in ("dropped_by_approval_gate", "blocked"),
            reason=outcome,
            bytes=result_size,
            keys=keys,
            digest=hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
            agent=str(agent) if agent else None,
            session=session,
        ))

    def _record(self, record: AuditRecord) -> None:
        self.records.append(record)
        try:
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
        except OSError:
            # If the audit log cannot be written the run continues, but the
            # record is not lost: it stays in `self.records`.
            pass

    # ------------------------------------------------------------ robots

    def _robots_for(self, url: str):
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root in self._robots:
            return self._robots[root]

        text = self._robots_fetcher(root + "/robots.txt")
        if text is None:
            self._robots[root] = None
            return None

        rp = urllib.robotparser.RobotFileParser()
        rp.parse(text.splitlines())
        self._robots[root] = rp
        return rp

    @staticmethod
    def _fetch_robots_over_network(url: str) -> str | None:
        try:
            import httpx

            response = httpx.get(
                url,
                timeout=10.0,
                headers={"User-Agent": config.USER_AGENT},
                follow_redirects=True,
            )
            if response.status_code == 200:
                return response.text
        except Exception:
            return None
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Shared instance, so the rate limiter is actually global across collectors.
DEFAULT = SourcePolicy()
