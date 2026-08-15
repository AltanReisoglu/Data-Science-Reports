"""Live check for a company already in the pipeline: what is true *now*, and what changed.

`company_detail` answers from the scan — a frozen record of what was true when the
funnel ran. This module answers the other question: **what has happened since?**

That difference is the whole point of the fourth principle in `docs/03`: the funnel
flows downward but monitoring is a loop. A company scored `watch` in a scan two
weeks ago is not a closed question; it is a question waiting for a change. Until
now nothing in the system could see a change, because nothing looked twice.

What it does, per company, against live APIs:

* **GitHub** — current stars/forks/last push for the repositories the scan
  recorded, and the delta against the star count stored at scan time.
* **Hacker News** — mentions since the scan's newest signal for that company.
* **SEC Form D** — filings under the company's name in the last 90 days, and
  whether any of them is newer than what the scan saw.

Two rules carried over from the rest of the pipeline:

1. **Every source can fail on its own.** A dead GitHub call does not stop the HN
   lookup, and the failure is reported rather than swallowed — "could not check"
   and "nothing changed" are different answers and must never look alike.
2. **Everything goes through `policy`.** Same gate, same rate limits, same audit
   log as the collectors; the fact that a human asked for it changes nothing.

It is deliberately shared: `conversation.py` exposes it as an agent tool and
`answers.py` calls it directly, so the deterministic path can refresh a company
even with no model configured. One implementation, two surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from collectors.github import GitHub
from collectors.hackernews import HackerNews
from collectors.sec_edgar import SecFormD


@dataclass
class SourceCheck:
    """One source's answer. `ok=False` means we could not look, not that nothing changed."""

    source: str
    ok: bool
    detail: str = ""
    error: str | None = None
    items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LiveReport:
    company: str
    checked_at: datetime
    scan_seen_until: datetime | None
    checks: list[SourceCheck] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[str]:
        return [c.source for c in self.checks if not c.ok]

    def as_dict(self) -> dict[str, Any]:
        return {
            "company": self.company,
            "checked_at": self.checked_at.isoformat(timespec="seconds"),
            "scan_seen_until": self.scan_seen_until.isoformat(timespec="seconds")
            if self.scan_seen_until
            else None,
            "changes": self.changes,
            "could_not_check": self.failed,
            "checks": [
                {
                    "source": c.source,
                    "ok": c.ok,
                    "detail": c.detail,
                    "error": c.error,
                    "items": c.items[:8],
                }
                for c in self.checks
            ],
        }

    def as_text(self) -> str:
        lines = [f"{self.company} — live check at {self.checked_at:%Y-%m-%d %H:%M} UTC"]
        if self.scan_seen_until:
            lines.append(f"the scan had seen up to {self.scan_seen_until:%Y-%m-%d}")
        lines.append("")
        if self.changes:
            lines.append("CHANGED SINCE THE SCAN:")
            lines += [f"  - {c}" for c in self.changes]
        else:
            lines.append("No change found in the sources that answered.")
        lines.append("")
        for check in self.checks:
            mark = "ok " if check.ok else "FAIL"
            lines.append(f"[{mark}] {check.source}: {check.detail or check.error}")
        if self.failed:
            lines.append("")
            lines.append(
                "Note: "
                + ", ".join(self.failed)
                + " could not be checked. That is not the same as no change."
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- helpers


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _repos_in_scan(company: dict) -> dict[str, int | None]:
    """`owner/repo` -> star count the scan recorded (None if it did not record one)."""
    repos: dict[str, int | None] = {}
    for signal in company.get("signals", []):
        url = (signal.get("source") or {}).get("url", "")
        if "github.com/" not in url:
            continue
        path = url.split("github.com/", 1)[1].strip("/")
        parts = path.split("/")
        if len(parts) >= 2:
            repos[f"{parts[0]}/{parts[1]}"] = (signal.get("raw") or {}).get("stars")
    return repos


def _latest_signal(company: dict) -> datetime | None:
    dates = [_parse(s.get("date")) for s in company.get("signals", [])]
    real = [d for d in dates if d]
    return max(real) if real else None


# --------------------------------------------------------------------------- checks


def _check_github(company: dict, report: LiveReport) -> None:
    repos = _repos_in_scan(company)
    owner = company.get("github")
    if not repos and not owner:
        report.checks.append(
            SourceCheck("github", True, "no repository was recorded for this company")
        )
        return

    collector = GitHub()
    seen: list[dict[str, Any]] = []
    for full_name, stars_then in repos.items():
        try:
            data = collector.fetch_json(f"https://api.github.com/repos/{full_name}")
        except Exception as e:
            report.checks.append(
                SourceCheck("github", False, error=f"{full_name}: {type(e).__name__}: {e}")
            )
            return
        stars_now = data.get("stargazers_count")
        entry = {
            "repository": full_name,
            "stars_now": stars_now,
            "stars_at_scan": stars_then,
            "last_push": data.get("pushed_at"),
            "open_issues": data.get("open_issues_count"),
            "url": data.get("html_url"),
        }
        seen.append(entry)
        if isinstance(stars_now, int) and isinstance(stars_then, int):
            delta = stars_now - stars_then
            if delta:
                report.changes.append(
                    f"{full_name}: stars {stars_then:,} → {stars_now:,} ({delta:+,})"
                )
        pushed = _parse(data.get("pushed_at"))
        if pushed and report.scan_seen_until and pushed > report.scan_seen_until:
            report.changes.append(f"{full_name}: pushed {pushed:%Y-%m-%d}, after the scan")

    report.checks.append(
        SourceCheck("github", True, f"{len(seen)} repository/ies checked", items=seen)
    )


def _check_hn(company: dict, report: LiveReport) -> None:
    name = company.get("name") or ""
    if not name:
        report.checks.append(SourceCheck("hn", True, "company has no name to search"))
        return

    result = HackerNews().run(query=name, days=90)
    if not result.succeeded:
        report.checks.append(SourceCheck("hn", False, error=result.error))
        return

    # Algolia's search is fuzzy: querying a company name returns posts that merely
    # rank near it. Measured on the first run — searching "ros-claw" returned a
    # story about a tokenizer. A mention only counts if the name is actually in
    # the title; a loose match here would report invented movement.
    needle = name.lower()
    on_topic = [s for s in result.signals if needle in s.summary.lower()]

    cutoff = report.scan_seen_until
    fresh = [s for s in on_topic if not cutoff or s.date > cutoff]
    items = [
        {"title": s.summary, "date": s.date.date().isoformat(), "url": s.source.url}
        for s in fresh[:8]
    ]
    if fresh:
        report.changes.append(
            f"{len(fresh)} Hacker News mention(s) naming it since the scan — newest: "
            f"{fresh[0].summary[:80]}"
        )
    report.checks.append(
        SourceCheck(
            "hn", True,
            f"{len(result.signals)} results in 90d, {len(on_topic)} actually name it, "
            f"{len(fresh)} newer than the scan",
            items=items,
        )
    )


def _check_sec(company: dict, report: LiveReport) -> None:
    name = company.get("name") or ""
    if not name or len(name) < 3:
        report.checks.append(SourceCheck("sec_edgar", True, "no usable company name"))
        return

    result = SecFormD().run(query=name, days=90)
    if not result.succeeded:
        report.checks.append(SourceCheck("sec_edgar", False, error=result.error))
        return

    cutoff = report.scan_seen_until
    fresh = [s for s in result.signals if not cutoff or s.date > cutoff]
    items = [
        {"summary": s.summary, "date": s.date.date().isoformat(), "url": s.source.url}
        for s in result.signals[:8]
    ]
    if fresh:
        report.changes.append(
            f"{len(fresh)} new Form D filing(s) since the scan — a filing appears within "
            f"15 days of a raise"
        )
    report.checks.append(
        SourceCheck("sec_edgar", True,
                    f"{len(result.signals)} filing(s) in 90d, {len(fresh)} newer than the scan",
                    items=items)
    )


# --------------------------------------------------------------------------- entry


def refresh(company: dict) -> LiveReport:
    """Check one company from the scan against live sources."""
    report = LiveReport(
        company=company.get("name", "unknown"),
        checked_at=datetime.now(timezone.utc),
        scan_seen_until=_latest_signal(company),
    )
    for check in (_check_github, _check_hn, _check_sec):
        try:
            check(company, report)
        except Exception as e:
            # A check that throws must not take the other checks with it.
            report.checks.append(
                SourceCheck(check.__name__.replace("_check_", ""), False,
                            error=f"{type(e).__name__}: {e}")
            )
    return report


def find_company(data: dict, name: str) -> dict | None:
    """Locate a company in a scan payload by name, case-insensitively."""
    target = (name or "").strip().lower()
    if not target:
        return None
    for candidate in data.get("candidates", []):
        company = candidate.get("company", {})
        if (company.get("name") or "").lower() == target:
            return company
    # A prefix match, so "argon" finds "Argonix" — but only if it is unambiguous.
    matches = [
        c.get("company", {})
        for c in data.get("candidates", [])
        if (c.get("company", {}).get("name") or "").lower().startswith(target)
    ]
    return matches[0] if len(matches) == 1 else None


def company_names(data: dict) -> list[str]:
    return [c.get("company", {}).get("name", "?") for c in data.get("candidates", [])]
