"""Runs: a turn you can start, watch, wait for and abort — from another caller.

The SSE endpoint answers the person who asked. That is not enough once a second
caller exists: an MCP tool call from OpenClaw cannot hold a stream open for a
minute, and a cron job has nobody to stream to at all.

OpenClaw's answer (docs/13 §2.3) is the one taken here. The `agent` RPC does not
wait for the model — it validates, resolves the session, and returns
`{runId, acceptedAt}` **immediately**. `agent.wait` then blocks on that id until
the run reaches a terminal state and reports `ok | error | timeout`. The stream is
one view of a run rather than the run itself.

Two details that are easy to get wrong and are load-bearing here:

**Timeout is not a status the run reaches — it is the waiter giving up.** The run
keeps going. Reporting `timeout` while marking the run failed would lose a result
that arrives two seconds later, so `wait` returns `timeout` and leaves the run
alone.

**Abort is cooperative and needs the token.** `CancellationToken` is the only
thing AutoGen offers to stop a turn in flight, and it has to be created before the
stream starts and handed to `run_stream`. A run that never got one can be marked
cancelled but the model call behind it will still complete and still be paid for —
so `abort` reports whether it actually had a lever.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

ACTIVE = ("accepted", "running")
TERMINAL = ("ok", "error", "cancelled")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Run:
    id: str
    session_id: str
    kind: str = "chat"
    status: str = "accepted"
    accepted_at: str = field(default_factory=_now)
    started_at: str = ""
    ended_at: str = ""
    error: str = ""
    result: Any = None
    events: int = 0
    _token: Any = None
    _done: asyncio.Event = field(default_factory=asyncio.Event)
    _task: asyncio.Task | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.id,
            "session_id": self.session_id,
            "kind": self.kind,
            "status": self.status,
            "accepted_at": self.accepted_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "error": self.error,
            "events": self.events,
        }


class RunRegistry:
    """Every turn in flight, plus the recent ones so `wait` can answer late."""

    def __init__(self, *, keep: int = 200) -> None:
        self._runs: dict[str, Run] = {}
        self._order: list[str] = []
        self._keep = keep

    # ------------------------------------------------------------ creation

    def accept(self, session_id: str, *, kind: str = "chat") -> Run:
        """Register a run and return it immediately. Nothing has executed yet."""
        run = Run(id=uuid.uuid4().hex[:16], session_id=session_id, kind=kind)
        self._runs[run.id] = run
        self._order.append(run.id)
        self._trim()
        return run

    def attach_token(self, run: Run, token: Any) -> None:
        """Give the run its cancellation lever, once the stream owns one."""
        run._token = token

    def start(self, run: Run, task: asyncio.Task | None = None) -> None:
        run.status = "running"
        run.started_at = _now()
        run._task = task

    # ------------------------------------------------------------ completion

    def finish(self, run: Run, *, result: Any = None) -> None:
        if run.status in TERMINAL:
            return
        run.status = "ok"
        run.result = result
        run.ended_at = _now()
        run._done.set()

    def fail(self, run: Run, error: BaseException | str) -> None:
        if run.status in TERMINAL:
            return
        run.status = "error"
        run.error = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        run.ended_at = _now()
        run._done.set()

    def cancelled(self, run: Run) -> None:
        if run.status in TERMINAL:
            return
        run.status = "cancelled"
        run.ended_at = _now()
        run._done.set()

    # ------------------------------------------------------------ queries

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def status(self, run_id: str) -> dict[str, Any]:
        run = self._runs.get(run_id)
        return run.as_dict() if run else {"run_id": run_id, "status": "unknown"}

    async def wait(self, run_id: str, timeout: float = 120.0) -> dict[str, Any]:
        """Block until the run ends, or report `timeout` and leave it running."""
        run = self._runs.get(run_id)
        if run is None:
            return {"run_id": run_id, "status": "unknown"}
        if run.status in TERMINAL:
            return run.as_dict()
        try:
            await asyncio.wait_for(run._done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Deliberately not a failure. The run is still going; the caller just
            # stopped waiting, and asking again later is expected to work.
            payload = run.as_dict()
            payload["status"] = "timeout"
            return payload
        return run.as_dict()

    def abort(self, run_id: str) -> dict[str, Any]:
        """Cancel a run. Reports honestly whether there was anything to cancel."""
        run = self._runs.get(run_id)
        if run is None:
            return {"run_id": run_id, "aborted": False, "reason": "unknown run"}
        if run.status in TERMINAL:
            return {"run_id": run_id, "aborted": False, "reason": f"already {run.status}"}
        if run._token is None:
            # No token means the model call cannot be stopped; saying "aborted"
            # here would be a lie the operator pays for.
            self.cancelled(run)
            return {"run_id": run_id, "aborted": False, "reason": "no cancellation token"}
        run._token.cancel()
        self.cancelled(run)
        return {"run_id": run_id, "aborted": True}

    def active(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self._runs.values() if r.status in ACTIVE]

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return [self._runs[i].as_dict() for i in reversed(self._order[-limit:]) if i in self._runs]

    def for_session(self, session_id: str) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self._runs.values() if r.session_id == session_id]

    # ------------------------------------------------------------ internals

    def _trim(self) -> None:
        while len(self._order) > self._keep:
            oldest = self._order.pop(0)
            run = self._runs.get(oldest)
            # Never drop a run somebody may still be waiting on.
            if run is not None and run.status in ACTIVE:
                self._order.append(oldest)
                break
            self._runs.pop(oldest, None)


async def guard(
    registry: RunRegistry,
    run: Run,
    coro: Callable[[], Awaitable[Any]],
) -> Any:
    """Run `coro`, and make sure the run reaches a terminal state either way."""
    registry.start(run)
    try:
        result = await coro()
    except asyncio.CancelledError:
        registry.cancelled(run)
        raise
    except Exception as exc:  # noqa: BLE001 — the registry records the type
        registry.fail(run, exc)
        raise
    registry.finish(run, result=result)
    return result


__all__ = ["ACTIVE", "TERMINAL", "Run", "RunRegistry", "guard"]
