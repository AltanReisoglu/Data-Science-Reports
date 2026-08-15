"""Hooks: the extension surface, with OpenClaw's decision rules.

`observability.AuditingInterventionHandler` already put a gate on the message
path, and it worked — but it was one function with one signature, so everything
that wanted to influence a turn had to be that function. OpenClaw's answer
(docs/13 §2.4) is a set of **named points** with **explicit decision rules**, and
that is what this module is.

The points are OpenClaw's, one for one. The rules that matter:

* **`before_tool_call` returning `{"block": True}` is terminal.** No later hook
  runs, the call does not happen. This is the point the approval gate hangs on.
* **`message_sending` returning `{"cancel": True}` is terminal.** Nothing leaves.
* **`before_agent_reply` may take over the turn** by returning `{"reply": ...}` —
  the model is never called.
* Everything else *contributes*: results are merged, later hooks see earlier
  updates, and no single hook can silently end the pipeline.

### Quarantine, and why it is not a nicety

OpenClaw quarantines a plugin context engine that crashes and falls back to the
built-in one, with the stated goal that *the agent does not go silent*. The same
rule applies here and for the same reason. A hook is operator code running inside
the turn; if a broken one could raise, then one bad line stops every conversation
in the gateway. So a hook that raises is recorded, disabled, and the pipeline
continues without it.

The asymmetry is deliberate and worth stating plainly: **a hook that crashes is
skipped, a hook that says "block" is obeyed.** Failing open on a crash and failing
closed on a decision are different questions, and answering both the same way gets
one of them wrong. A crashed approval hook therefore does *not* mean "approved" —
`approval.py` registers the gate so that its own failure mode is a block, not a
skip.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("vcagent.hooks")

# OpenClaw's hook points (docs/13 §2.4). Names kept identical so the two systems
# can be read side by side.
BEFORE_MODEL_RESOLVE = "before_model_resolve"
BEFORE_PROMPT_BUILD = "before_prompt_build"
BEFORE_AGENT_REPLY = "before_agent_reply"
AGENT_END = "agent_end"
BEFORE_COMPACTION = "before_compaction"
AFTER_COMPACTION = "after_compaction"
BEFORE_TOOL_CALL = "before_tool_call"
AFTER_TOOL_CALL = "after_tool_call"
TOOL_RESULT_PERSIST = "tool_result_persist"
MESSAGE_RECEIVED = "message_received"
MESSAGE_SENDING = "message_sending"
MESSAGE_SENT = "message_sent"
SESSION_START = "session_start"
SESSION_END = "session_end"
GATEWAY_START = "gateway_start"
GATEWAY_STOP = "gateway_stop"

POINTS = (
    BEFORE_MODEL_RESOLVE, BEFORE_PROMPT_BUILD, BEFORE_AGENT_REPLY, AGENT_END,
    BEFORE_COMPACTION, AFTER_COMPACTION,
    BEFORE_TOOL_CALL, AFTER_TOOL_CALL, TOOL_RESULT_PERSIST,
    MESSAGE_RECEIVED, MESSAGE_SENDING, MESSAGE_SENT,
    SESSION_START, SESSION_END, GATEWAY_START, GATEWAY_STOP,
)

# Which key ends the chain at which point. A hook returning any of these on the
# matching point is obeyed and nothing after it runs.
TERMINAL_KEYS = {
    BEFORE_TOOL_CALL: "block",
    MESSAGE_SENDING: "cancel",
    BEFORE_AGENT_REPLY: "reply",
}


@dataclass
class Outcome:
    """What the chain decided. `stopped` is the only field a caller must check."""

    point: str
    stopped: bool = False
    reason: str = ""
    by: str = ""
    updates: dict[str, Any] = field(default_factory=dict)
    ran: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.stopped and self.point in (BEFORE_TOOL_CALL, MESSAGE_SENDING)

    def get(self, key: str, default: Any = None) -> Any:
        return self.updates.get(key, default)


@dataclass
class Registration:
    point: str
    name: str
    fn: Callable[..., Any]
    order: int = 0
    failures: int = 0
    quarantined: bool = False


class HookRegistry:
    """Named extension points, ordered, quarantining, sync or async."""

    def __init__(self, *, failure_limit: int = 3) -> None:
        self._hooks: dict[str, list[Registration]] = {p: [] for p in POINTS}
        self.failure_limit = failure_limit

    # ------------------------------------------------------------ registration

    def register(
        self,
        point: str,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        order: int = 0,
    ) -> Registration:
        if point not in self._hooks:
            raise ValueError(f"unknown hook point {point!r}")
        entry = Registration(point=point, name=name or getattr(fn, "__name__", "hook"), fn=fn, order=order)
        self._hooks[point].append(entry)
        self._hooks[point].sort(key=lambda r: (r.order, r.name))
        return entry

    def on(self, point: str, *, name: str | None = None, order: int = 0):
        def decorator(fn):
            self.register(point, fn, name=name, order=order)
            return fn

        return decorator

    def unregister(self, point: str, name: str) -> bool:
        before = len(self._hooks.get(point, []))
        self._hooks[point] = [r for r in self._hooks.get(point, []) if r.name != name]
        return len(self._hooks[point]) != before

    def registered(self, point: str | None = None) -> list[Registration]:
        if point is not None:
            return list(self._hooks.get(point, []))
        return [r for entries in self._hooks.values() for r in entries]

    # ------------------------------------------------------------ execution

    async def run(self, point: str, payload: dict[str, Any] | None = None) -> Outcome:
        """Run every hook on `point` in order until one of them ends the chain."""
        outcome = Outcome(point=point)
        payload = dict(payload or {})
        terminal_key = TERMINAL_KEYS.get(point)

        for entry in list(self._hooks.get(point, [])):
            if entry.quarantined:
                continue
            try:
                result = entry.fn({**payload, **outcome.updates})
                if inspect.isawaitable(result):
                    result = await result
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — quarantine, do not propagate
                self._quarantine(entry, exc)
                outcome.failed.append(entry.name)
                continue

            outcome.ran.append(entry.name)
            if not isinstance(result, dict):
                continue
            outcome.updates.update(result)

            if terminal_key and result.get(terminal_key):
                outcome.stopped = True
                outcome.by = entry.name
                outcome.reason = str(result.get("reason") or f"{terminal_key} by {entry.name}")
                break

        return outcome

    def _quarantine(self, entry: Registration, exc: BaseException) -> None:
        entry.failures += 1
        log.warning("hook %s on %s failed: %s", entry.name, entry.point, exc)
        if entry.failures >= self.failure_limit:
            entry.quarantined = True
            log.warning(
                "hook %s quarantined after %d failures; pipeline continues without it",
                entry.name, entry.failures,
            )

    def revive(self, name: str) -> int:
        """Bring quarantined hooks back — after the operator has fixed one."""
        count = 0
        for entry in self.registered():
            if entry.name == name and entry.quarantined:
                entry.quarantined = False
                entry.failures = 0
                count += 1
        return count

    def quarantined(self) -> list[str]:
        return [r.name for r in self.registered() if r.quarantined]


# The gateway's registry. One per process, matching "one gateway per host".
REGISTRY = HookRegistry()


__all__ = [
    "AGENT_END", "AFTER_COMPACTION", "AFTER_TOOL_CALL", "BEFORE_AGENT_REPLY",
    "BEFORE_COMPACTION", "BEFORE_MODEL_RESOLVE", "BEFORE_PROMPT_BUILD",
    "BEFORE_TOOL_CALL", "GATEWAY_START", "GATEWAY_STOP", "MESSAGE_RECEIVED",
    "MESSAGE_SENDING", "MESSAGE_SENT", "POINTS", "REGISTRY", "SESSION_END",
    "SESSION_START", "TERMINAL_KEYS", "TOOL_RESULT_PERSIST",
    "HookRegistry", "Outcome", "Registration",
]
