"""Scheduled work, and the rule that makes it safe: a fresh session every run.

This is `docs/04 §8` phase 5 (the watchlist) meeting phase 7 (the alert on a
phone), and it only became small once the gateway existed. Both halves were
already built: `live.py` re-checks a company against live sources, and the
OpenClaw bridge can deliver a sentence to Telegram. What was missing is the thing
in between — something that decides *when* to look and *whether the result is
worth interrupting a person for*.

**Fresh session per run, from OpenClaw (docs/13 §4.1).** A scheduled job that
reuses one long-lived session accumulates every previous run in its context: it
gets more expensive each day, and eventually it starts answering from memory of
last week instead of from this morning's data. So each run gets its own session
key and its own transcript, and nothing carries over except what was written to
memory on purpose.

**A finding is not a notification.** `live.py` reports what changed; most changes
are noise. `Threshold` is where the judgement lives, and it is deliberately dumb
and legible — star deltas, new filings, named mentions — rather than a model
call, because a notifier that costs tokens to decide whether to cost tokens is a
bad trade and because "why did it wake me" should have an answer a person can
read.

**Push goes through the same gate as everything else.** A notification is an
outbound message. `cron.auto_push` exists so an operator can say "for the
watchlist specifically, stop asking" — off by default, and separate from the
global `ALLOW_OUTBOUND`, so switching on unattended alerts is not the same
decision as letting the chat agent message people.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import config

from . import sessions as sessions_module

JOBS_PATH_NAME = "cron.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    """One scheduled piece of work."""

    id: str
    kind: str                       # scan | live
    every_minutes: int
    query: str = ""                 # scan: the sector; live: unused
    companies: list[str] = field(default_factory=list)   # live: what to watch
    days: int = 7
    enabled: bool = True
    last_run_at: str = ""
    last_status: str = ""
    runs: int = 0
    notify: bool = True

    def due(self, now: datetime | None = None) -> bool:
        if not self.enabled:
            return False
        if not self.last_run_at:
            return True
        try:
            last = datetime.fromisoformat(self.last_run_at)
        except ValueError:
            return True
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (now or _now()) - last >= timedelta(minutes=self.every_minutes)


@dataclass
class Finding:
    """Something a run noticed, and whether it is worth a person's attention."""

    company: str
    headline: str
    detail: str = ""
    url: str = ""
    notable: bool = False

    def as_line(self) -> str:
        return f"{self.company}: {self.headline}" + (f" — {self.url}" if self.url else "")


# `live.py` writes star movement into a change line as `(+123)`. Reading it back
# out is uglier than a structured field would be, and it is what the existing
# report actually carries — inventing a parallel shape for the notifier would
# mean two definitions of "what changed" that drift apart.
_STARS = re.compile(r"stars\s[\d,]+\s→\s[\d,]+\s\(([+-][\d,]+)\)")


class Threshold:
    """What counts as worth interrupting someone for.

    Legible on purpose. Every rule here can be read out loud when answering "why
    did you message me", which a learned scorer cannot do.

    It reads `LiveReport` as `live.py` actually produces it: `changes` is a list
    of sentences, `checks` carries `ok=False` when a source could not be reached.
    Those two are different in a way that matters and that this project keeps
    insisting on — **"could not check" is not "nothing changed"** — so a failed
    source is reported and never notified. Waking someone because GitHub rate
    limited us is noise; hiding it is a lie of omission.
    """

    def __init__(self, *, star_delta: int = 25, ignore: tuple[str, ...] = ()) -> None:
        self.star_delta = star_delta
        self.ignore = ignore

    def judge(self, report: Any) -> list[Finding]:
        findings: list[Finding] = []
        name = str(getattr(report, "company", "") or "unknown")

        for source in getattr(report, "failed", []) or []:
            error = next(
                (
                    str(c.error or "")
                    for c in getattr(report, "checks", [])
                    if getattr(c, "source", "") == source and not getattr(c, "ok", True)
                ),
                "",
            )
            findings.append(
                Finding(name, f"{source} could not be checked", error, notable=False)
            )

        for line in getattr(report, "changes", []) or []:
            text = str(line)
            if any(word and word.lower() in text.lower() for word in self.ignore):
                continue
            stars = _STARS.search(text)
            if stars:
                moved = abs(int(stars.group(1).replace(",", "").replace("+", "")))
                # Small drift is not news. A repository gaining three stars in a
                # day is what repositories do.
                findings.append(Finding(name, text, notable=moved >= self.star_delta))
                continue
            findings.append(Finding(name, text, notable=True))

        return findings


class CronStore:
    """Jobs on disk, in the state directory beside everything else."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (config.STATE / "state" / JOBS_PATH_NAME)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Job]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        jobs = {}
        for entry in raw.get("jobs", []):
            try:
                jobs[entry["id"]] = Job(**entry)
            except TypeError:
                continue
        return jobs

    def save(self, jobs: dict[str, Job]) -> None:
        payload = {"version": 1, "updated_at": _now().isoformat(),
                   "jobs": [asdict(j) for j in jobs.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)


class Scheduler:
    """Runs due jobs. Deliberately not a daemon — `tick()` is called by the caller.

    A loop that owns its own thread is harder to test and harder to stop; the
    gateway already has an event loop and can call this on a timer.
    """

    def __init__(
        self,
        *,
        store: CronStore | None = None,
        sessions: sessions_module.SessionManager | None = None,
        threshold: Threshold | None = None,
        notifier: Callable[[str], Any] | None = None,
    ) -> None:
        self.store = store or CronStore()
        self.sessions = sessions or sessions_module.SessionManager()
        self.threshold = threshold or Threshold()
        self.notifier = notifier
        self.jobs = self.store.load()

    # ------------------------------------------------------------ jobs

    def add(self, job: Job) -> Job:
        self.jobs[job.id] = job
        self.store.save(self.jobs)
        return job

    def watch(self, companies: list[str], *, every_minutes: int = 720) -> Job:
        """The watchlist, as one job."""
        return self.add(
            Job(id="watchlist", kind="live", every_minutes=every_minutes, companies=companies)
        )

    def remove(self, job_id: str) -> bool:
        removed = self.jobs.pop(job_id, None) is not None
        if removed:
            self.store.save(self.jobs)
        return removed

    def due(self, now: datetime | None = None) -> list[Job]:
        return [j for j in self.jobs.values() if j.due(now)]

    # ------------------------------------------------------------ running

    async def tick(self, now: datetime | None = None) -> list[dict[str, Any]]:
        results = []
        for job in self.due(now):
            results.append(await self.run(job))
        return results

    async def run(self, job: Job) -> dict[str, Any]:
        """One run: fresh session, do the work, judge it, maybe notify."""
        run_id = uuid.uuid4().hex[:8]
        # Fresh every time — `SessionKey.ephemeral` guarantees it even if the same
        # key is asked for twice.
        record = self.sessions.route("cron", peer=f"{job.id}-{run_id}", kind="run")
        self.sessions.store.append(
            record.id, {"event": "cron_start", "job": job.id, "kind": job.kind}
        )

        try:
            findings = await self._work(job)
            status = "ok"
        except Exception as exc:  # noqa: BLE001 — one bad job must not stop the rest
            findings, status = [], f"error: {type(exc).__name__}: {exc}"

        notable = [f for f in findings if f.notable]
        delivered = False
        if notable and job.notify:
            delivered = await self._notify(job, notable)

        job.last_run_at = _now().isoformat()
        job.last_status = status
        job.runs += 1
        self.store.save(self.jobs)

        self.sessions.store.append(
            record.id,
            {
                "event": "cron_end", "status": status,
                "findings": len(findings), "notable": len(notable), "delivered": delivered,
            },
        )
        return {
            "job": job.id, "session": record.id, "status": status,
            "findings": [f.as_line() for f in findings],
            "notable": [f.as_line() for f in notable],
            "delivered": delivered,
        }

    async def _work(self, job: Job) -> list[Finding]:
        if job.kind == "live":
            return await self._live(job)
        if job.kind == "scan":
            return await self._scan(job)
        raise ValueError(f"unknown job kind {job.kind!r}")

    async def _live(self, job: Job) -> list[Finding]:
        import live

        from . import tools as tools_module

        data = tools_module.read_latest_scan()
        if not data:
            return []
        findings: list[Finding] = []
        for name in job.companies:
            company = live.find_company(data, name)
            if company is None:
                continue
            report = await asyncio.to_thread(live.refresh, company)
            findings.extend(self.threshold.judge(report))
        return findings

    async def _scan(self, job: Job) -> list[Finding]:
        from . import tools as tools_module

        starter = tools_module.gateway_scan_starter()
        await asyncio.to_thread(starter, job.query, job.days)
        return [Finding(company=job.query, headline="scan started", notable=False)]

    async def _notify(self, job: Job, findings: list[Finding]) -> bool:
        """Deliver, if there is somewhere to deliver to and permission to do it."""
        if self.notifier is None:
            return False
        body = "\n".join(f.as_line() for f in findings)
        message = f"[{job.id}] {len(findings)} update(s):\n{body}"
        try:
            result = self.notifier(message)
            if asyncio.iscoroutine(result):
                result = await result
            return bool(result)
        except Exception:  # noqa: BLE001 — a failed notification is not a failed run
            return False


__all__ = ["CronStore", "Finding", "Job", "Scheduler", "Threshold"]
