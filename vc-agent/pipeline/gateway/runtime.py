"""The gateway's one runtime, and the three rules that make it safe to use.

`fanin.py` builds a `SingleThreadedAgentRuntime` per enrichment: start it, publish
three tasks, drain a queue, stop. That shape is right for a batch — and wrong for
a gateway, which has no end and must not lose a turn.

The difference matters because of something this project measured rather than
read (`06 §8`, `09 §4`): a crashing handler returns the `gather` inside
`_process_publish` early, `stop_when_idle()` then opens its barrier before the
siblings finish, and **completed work disappears with no exception and no
warning**. On an enrichment branch that costs an analysis. On the control plane
it costs somebody's message.

So the runtime here is built to avoid every part of that:

**1 · It never goes idle-stopped.** `stop_when_idle()` is the barrier that opens
early, and it is simply not called. The runtime starts when the server starts and
stops when the server stops. Messages are processed as they arrive; there is no
moment where "the queue looks empty" has to mean anything.

**2 · Handlers do not raise.** `ignore_unhandled_exceptions=True` keeps one bad
turn from taking the process down, but that is the second line, not the first —
`SessionAgent` catches its own failures and publishes them, exactly as
`fanin.BranchWorker` does: *"a failure is published, not raised."*

**3 · One runtime per process.** OpenClaw's "exactly one Gateway per host"
(docs/13 §1) has a technical counterpart here: two runtimes would mean two agents
answering for the same session key, writing the same transcript.

The intervention handler comes along for free, and is worth having: with the
control plane on the runtime, `observability.AuditingInterventionHandler` now sees
session traffic as well as enrichment traffic — one audit surface instead of two.
"""

from __future__ import annotations

import logging
from typing import Any

from autogen_core import SingleThreadedAgentRuntime

import observability

log = logging.getLogger("vcagent.runtime")


class GatewayRuntime:
    """A long-lived `SingleThreadedAgentRuntime`, owned by the process."""

    def __init__(self) -> None:
        self._runtime: SingleThreadedAgentRuntime | None = None
        self._handler: observability.AuditingInterventionHandler | None = None
        self._registered: set[str] = set()
        self.started = False

    # ------------------------------------------------------------ lifecycle

    def build(self) -> SingleThreadedAgentRuntime:
        if self._runtime is None:
            self._handler = observability.AuditingInterventionHandler()
            self._runtime = SingleThreadedAgentRuntime(
                intervention_handlers=[self._handler],
                # A handler that raises must not end the runtime. The first line
                # of defence is that handlers do not raise; this is the second.
                ignore_unhandled_exceptions=True,
            )
        return self._runtime

    @property
    def runtime(self) -> SingleThreadedAgentRuntime:
        return self.build()

    @property
    def handler(self) -> observability.AuditingInterventionHandler | None:
        return self._handler

    async def start(self) -> None:
        if self.started:
            return
        self.build().start()
        self.started = True
        log.info("gateway runtime started")

    async def stop(self) -> None:
        """Stop, without ever asking whether the queue looks empty."""
        if not self.started or self._runtime is None:
            return
        try:
            # `stop()` returns immediately and does not cancel in-progress
            # handling (`05:1069`) — which is what we want on shutdown: a turn
            # already being answered gets to finish writing its transcript.
            await self._runtime.stop()
        finally:
            self.started = False
            log.info("gateway runtime stopped")

    async def close(self) -> None:
        await self.stop()
        if self._runtime is not None:
            try:
                await self._runtime.close()
            except Exception:  # noqa: BLE001
                pass
        self._runtime = None
        self._registered.clear()

    # ------------------------------------------------------------ registration

    async def register_once(self, agent_type: str, register) -> bool:
        """Register an agent type, at most once per process.

        Registering the same type twice raises, and the gateway builds its agents
        lazily from several entry points (`/api/chat`, the relay webhook, cron),
        so "have we done this yet" has to be tracked rather than assumed.
        """
        if agent_type in self._registered:
            return False
        await register(self.runtime)
        self._registered.add(agent_type)
        return True

    def registered(self) -> list[str]:
        return sorted(self._registered)

    def report(self) -> dict[str, Any]:
        return {
            "running": self.started,
            "agent_types": self.registered(),
            "routed": len(self._handler.routed) if self._handler else 0,
            "dropped": len(self._handler.dropped) if self._handler else 0,
        }


GATEWAY = GatewayRuntime()


__all__ = ["GATEWAY", "GatewayRuntime"]
