"""Sessions: where a message lands, how long it lives, and what is kept.

Before this module there was one conversation. `conversation.STATE_PATH` was a
single file, so every question from every origin shared one context and one
history. That is correct for exactly one user on one surface, and it stops being
correct the moment a second surface exists — which is what the OpenClaw bridge
adds. Two people asking through Telegram would have been reading each other's
conversation.

The model is OpenClaw's (docs/13 §4), reproduced rather than invented:

**Routing by origin.** A session key is `(channel, kind, peer, account)`. Direct
messages share or split according to `dm_scope`; groups and rooms are isolated per
room; **cron runs get a fresh session every time**, because a scheduled job that
inherits yesterday's context slowly drifts; webhooks are isolated per hook.

**Three timestamps, not one.** `session_started_at` drives the daily reset,
`last_interaction_at` drives the idle reset, `updated_at` is bookkeeping. Merging
them looks harmless and breaks both resets: a long conversation never ages out,
and a quiet one resets at the wrong hour.

**A lane per session.** Turns in one session are serialised; turns in different
sessions are not. OpenClaw calls this a session lane and it exists to stop two
turns racing on the same transcript and context.

**Transcript on disk, index beside it.** `<sessionId>.jsonl` is the conversation;
`sessions.json` is the index. The index is rewritten atomically because a second
process (the MCP server) reads it while this one writes it.

### On the AutoGen side

A session id is an agent *key*. That is not a metaphor: `autogen_core` addresses
agents as `(type, key)` and derives the key from a topic's source, so "one isolated
agent instance per session" is the runtime's own mechanism rather than something
built on top of it (`05:670`, docs/14 §3.6). This module uses the addressing idea
with AgentChat agents held in a dict — the literal core-runtime version is the
upgrade path, and the key layout here is already what it would need.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

# Module scope, not inside a function. `@message_handler` infers its message type
# with `get_type_hints`, and under `from __future__ import annotations` every
# annotation is a string resolved against *module* globals — a function-local
# import is invisible to it and registration fails with a bare `NameError`
# (docs/06 §11).
from autogen_core import (  # noqa: E402
    MessageContext,
    RoutedAgent,
    TopicId,
    TypeSubscription,
    message_handler,
)

import config

log = logging.getLogger("vcagent.sessions")

_SAFE = re.compile(r"[^a-zA-Z0-9._@+-]+")

# Origins a session can come from. `cron` and `mcp` are the two that make the
# distinction load-bearing: one must never reuse context, the other arrives from
# another process entirely.
CHANNELS = ("web", "cli", "mcp", "cron", "webhook")
KINDS = ("dm", "group", "room", "run", "hook")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _slug(text: str) -> str:
    """Make one component of a session id safe to put in a filename.

    Peer identifiers arrive from outside — an OpenClaw peer, a webhook payload —
    and a session id becomes a filename. Separators are already gone, so
    traversal was never reachable, but a run of dots is still worth collapsing:
    `..` in a name is a trap for anything that later joins these paths, and a
    leading dot silently hides the transcript from the operator.
    """
    cleaned = _SAFE.sub("-", (text or "default").strip())
    cleaned = re.sub(r"\.{2,}", ".", cleaned).strip("-.")
    return (cleaned or "default")[:64]


# --------------------------------------------------------------------------- key


@dataclass(frozen=True)
class SessionKey:
    """Where a turn came from. Two turns with the same key share a conversation."""

    channel: str
    kind: str
    peer: str
    account: str = "default"

    def as_id(self) -> str:
        return ":".join(
            ["agent", config.AGENT_ID, _slug(self.channel), _slug(self.kind), _slug(self.peer)]
            + ([_slug(self.account)] if self.account != "default" else [])
        )

    @property
    def ephemeral(self) -> bool:
        """A cron run is a fresh session by definition; it is never reused."""
        return self.channel == "cron"


def resolve(
    channel: str,
    *,
    peer: str = "local",
    kind: str = "dm",
    account: str = "default",
    dm_scope: str | None = None,
) -> SessionKey:
    """Apply OpenClaw's routing table to one incoming turn.

    `dm_scope` decides how much of the origin survives into the key, and it is the
    only knob here with a security consequence: under `main` every direct message
    from everyone lands in one session.
    """
    scope = dm_scope or config.SESSION_POLICY.dm_scope
    if scope not in config.VALID_DM_SCOPES:
        scope = "per-channel-peer"

    if channel == "cron":
        # Fresh every run. The run id is part of the key, so nothing is inherited.
        return SessionKey(channel="cron", kind="run", peer=peer, account=account)
    if channel == "webhook":
        return SessionKey(channel="webhook", kind="hook", peer=peer, account=account)
    if kind in ("group", "room"):
        # Isolated per room regardless of scope — a group is a shared space and
        # splitting it per speaker would give each person half a conversation.
        return SessionKey(channel=channel, kind=kind, peer=peer, account=account)

    if scope == "main":
        return SessionKey(channel="main", kind="dm", peer="shared")
    if scope == "per-peer":
        return SessionKey(channel="main", kind="dm", peer=peer)
    if scope == "per-account-channel-peer":
        return SessionKey(channel=channel, kind="dm", peer=peer, account=account)
    return SessionKey(channel=channel, kind="dm", peer=peer)


# --------------------------------------------------------------------------- record


@dataclass
class SessionRecord:
    """The index entry. The conversation itself is the transcript beside it."""

    id: str
    channel: str
    kind: str
    peer: str
    account: str
    created_at: str
    session_started_at: str
    last_interaction_at: str
    updated_at: str
    turns: int = 0
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> SessionKey:
        return SessionKey(self.channel, self.kind, self.peer, self.account)

    def touch(self) -> None:
        stamp = _now().isoformat()
        self.last_interaction_at = stamp
        self.updated_at = stamp


# --------------------------------------------------------------------------- store


class SessionStore:
    """The index and the transcripts. Plain files, atomic index writes."""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = directory or config.SESSIONS
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "sessions.json"

    # ------------------------------------------------------------ index

    def load(self) -> dict[str, SessionRecord]:
        if not self.index_path.exists():
            return {}
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt index loses the *list*, not the transcripts. Rebuilding an
            # empty one is survivable; refusing to start is not.
            return {}
        records = {}
        for entry in raw.get("sessions", []):
            try:
                records[entry["id"]] = SessionRecord(**entry)
            except TypeError:
                continue
        return records

    def save(self, records: dict[str, SessionRecord]) -> None:
        payload = {
            "version": 1,
            "agent": config.AGENT_ID,
            "updated_at": _now().isoformat(),
            "sessions": [asdict(r) for r in records.values()],
        }
        # Written to a sibling and renamed: the MCP server reads this file from a
        # different process and must never see a half-written one.
        tmp = self.index_path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.index_path)

    # ------------------------------------------------------------ transcript

    def transcript_path(self, session_id: str) -> Path:
        return self.dir / f"{_slug(session_id)}.jsonl"

    def append(self, session_id: str, entry: dict[str, Any]) -> None:
        line = json.dumps({"ts": _now().isoformat(), **entry}, ensure_ascii=False, default=str)
        with self.transcript_path(session_id).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def read(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        path = self.transcript_path(session_id)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def state_path(self, session_id: str) -> Path:
        """Where the agent's serialised context lives, beside its transcript."""
        return self.dir / f"{_slug(session_id)}.state.json"

    def drop(self, session_id: str) -> None:
        for path in (
            self.transcript_path(session_id),
            self.state_path(session_id),
        ):
            path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- manager


class SessionManager:
    """Resolves origins to sessions, applies resets, and owns the per-session lane.

    The agent factory is injected rather than imported: this module has no opinion
    about what an agent is, which keeps it testable without a model client and
    keeps the import graph acyclic (`conversation` imports the gateway, not the
    other way round).
    """

    def __init__(
        self,
        agent_factory=None,
        *,
        store: SessionStore | None = None,
        policy: config.SessionPolicy | None = None,
    ) -> None:
        self.store = store or SessionStore()
        self.policy = policy or config.SESSION_POLICY
        self._factory = agent_factory
        self._records: dict[str, SessionRecord] = self.store.load()
        self._locks: dict[str, asyncio.Lock] = {}
        self._agents: dict[str, Any] = {}

    # ------------------------------------------------------------ resolution

    def open(self, key: SessionKey) -> SessionRecord:
        """Get or create the session for this key, applying resets first."""
        session_id = key.as_id()
        if key.ephemeral:
            session_id = f"{session_id}:{uuid.uuid4().hex[:8]}"

        record = self._records.get(session_id)
        if record is not None and self._expired(record):
            self.reset(session_id, reason="expired")
            record = None

        if record is None:
            stamp = _now().isoformat()
            record = SessionRecord(
                id=session_id,
                channel=key.channel,
                kind=key.kind,
                peer=key.peer,
                account=key.account,
                created_at=stamp,
                session_started_at=stamp,
                last_interaction_at=stamp,
                updated_at=stamp,
            )
            self._records[session_id] = record
            self.store.append(session_id, {"event": "session_start", "key": key.as_id()})
            self._persist()
        return record

    def route(self, channel: str, **kwargs: Any) -> SessionRecord:
        return self.open(resolve(channel, **kwargs))

    def _expired(self, record: SessionRecord) -> bool:
        """Daily reset and idle reset, checked separately because they differ.

        System traffic must not keep a session alive — OpenClaw makes the same
        point about heartbeats and cron. Here that falls out of only calling
        `touch()` on a real turn.
        """
        now = _now()
        if self.policy.idle_minutes:
            last = _parse(record.last_interaction_at)
            if last and now - last > timedelta(minutes=self.policy.idle_minutes):
                return True
        if self.policy.daily_reset_hour is not None:
            started = _parse(record.session_started_at)
            if started:
                boundary = now.replace(
                    hour=self.policy.daily_reset_hour, minute=0, second=0, microsecond=0
                )
                if boundary > now:
                    boundary -= timedelta(days=1)
                if started < boundary:
                    return True
        return False

    # ------------------------------------------------------------ lane

    def lock(self, session_id: str) -> asyncio.Lock:
        """One lane per session: turns in a session serialise, sessions do not."""
        return self._locks.setdefault(session_id, asyncio.Lock())

    async def agent(self, record: SessionRecord):
        """The agent for this session, built once and kept."""
        if self._factory is None:
            raise RuntimeError("SessionManager has no agent factory")
        existing = self._agents.get(record.id)
        if existing is None:
            existing = await _maybe_await(self._factory(record))
            self._agents[record.id] = existing
        return existing

    # ------------------------------------------------------------ dispatch

    async def dispatch(
        self,
        record: SessionRecord,
        text: str,
        *,
        runtime=None,
        reply_to: str = "",
        origin: str = "",
        run_id: str = "",
    ) -> None:
        """Send a turn through the runtime, addressed by the session id.

        The message-driven path, used by callers with nobody to stream to: the
        relay webhook, cron, an MCP request. The dashboard keeps calling the
        conversation directly, because SSE needs tokens as they arrive and a
        published message hands back a finished answer.

        Both paths write the same transcript and hold the same lane, so the split
        is about *delivery*, not about two kinds of session. Worth being plain
        about rather than claiming the whole gateway is message-driven when the
        most-used route is not.
        """
        from .runtime import GATEWAY

        target = runtime or GATEWAY.runtime
        await target.publish_message(
            Turn(text=text, origin=origin, peer=record.peer, run_id=run_id, reply_to=reply_to),
            topic_id=TopicId(TURN_TOPIC, record.id),
        )

    # ------------------------------------------------------------ lifecycle

    def record_turn(self, record: SessionRecord, role: str, content: str, **extra: Any) -> None:
        record.turns += 1
        record.touch()
        if not record.title and role == "user":
            record.title = content.strip().splitlines()[0][:80] if content.strip() else ""
        self.store.append(record.id, {"role": role, "content": content, **extra})
        self._persist()

    def reset(self, session_id: str, *, reason: str = "manual") -> None:
        """End a session: transcript and serialised context go, the index entry goes."""
        self.store.append(session_id, {"event": "session_end", "reason": reason})
        self.store.drop(session_id)
        self._records.pop(session_id, None)
        self._agents.pop(session_id, None)
        self._locks.pop(session_id, None)
        self._persist()

    def list(self) -> list[SessionRecord]:
        return sorted(self._records.values(), key=lambda r: r.last_interaction_at, reverse=True)

    def get(self, session_id: str) -> SessionRecord | None:
        return self._records.get(session_id)

    def prune(self) -> int:
        """Drop sessions past the retention window, then cap the count.

        OpenClaw's `session.maintenance`. Without it a long-running gateway grows
        an index forever, and the index is read on every turn.
        """
        cutoff = _now() - timedelta(days=self.policy.prune_after_days)
        removed = 0
        for record in list(self._records.values()):
            last = _parse(record.last_interaction_at)
            if last and last < cutoff:
                self.reset(record.id, reason="pruned")
                removed += 1
        if len(self._records) > self.policy.max_entries:
            for record in self.list()[self.policy.max_entries :]:
                self.reset(record.id, reason="over_capacity")
                removed += 1
        return removed

    def _persist(self) -> None:
        try:
            self.store.save(self._records)
        except OSError:
            # The index is a convenience; the transcripts are the record. Failing
            # to write it must not fail the turn that was already answered.
            pass

    def __iter__(self) -> Iterator[SessionRecord]:
        return iter(self.list())


# --------------------------------------------------------------------------- core
#
# Where the session id stops being a dictionary key and becomes an agent key.
#
# `TypeSubscription(topic_type=TURN, agent_type="session")` maps a topic's
# **source** onto an agent's **key** (`05:670`). Our session ids are already the
# right shape — `agent:main:web:dm:local`, which satisfies CloudEvents'
# `^[\w\-\.\:\=]+\Z` — so publishing to `TopicId("turn", session_id)` addresses
# `AgentId("session", session_id)`, and the runtime creates that agent the first
# time it is addressed.
#
# One isolated agent instance per conversation, with no registry to keep in sync:
# that is the mechanism docs/14 §3.6 called "free multi-tenancy", used here for
# the thing it was described for.

TURN_TOPIC = "turn"
REPLY_TOPIC = "reply"
SESSION_AGENT_TYPE = "session"


@dataclass
class Turn:
    """A message to be answered, addressed by the topic's source."""

    text: str
    origin: str = ""
    peer: str = ""
    run_id: str = ""
    reply_to: str = ""       # channel name, or empty for "caller collects it"


@dataclass
class Reply:
    """The answer, published back on `reply.<channel>`."""

    text: str
    session: str
    run_id: str = ""
    error: str = ""

    @property
    def failed(self) -> bool:
        return bool(self.error)


class SessionAgent(RoutedAgent):
    """One conversation. `self.id.key` *is* the session id — there is no lookup.

    The discipline is `fanin.BranchWorker`'s, for the same measured reason: a
    handler that raises can take completed sibling work with it. So this one
    catches everything and publishes a `Reply` carrying the error instead.

    The transcript is written **before** the responder is called. A crash then
    costs the answer and not the record of the question, which is the difference
    between a conversation with a gap and a conversation that quietly forgot.
    """

    def __init__(self, store: "SessionStore", responder) -> None:
        super().__init__("A gateway session")
        self._store = store
        self._responder = responder

    @message_handler
    async def handle_turn(self, message: Turn, ctx: MessageContext) -> None:
        session_id = self.id.key
        self._store.append(
            session_id,
            {"role": "user", "content": message.text, "origin": message.origin},
        )

        try:
            text = await self._responder(session_id, message)
            reply = Reply(text=text or "", session=session_id, run_id=message.run_id)
        except Exception as exc:  # noqa: BLE001 — published, never raised
            reply = Reply(
                text="", session=session_id, run_id=message.run_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            log.warning("session %s failed: %s", session_id, reply.error)

        self._store.append(
            session_id,
            {"role": "assistant", "content": reply.text, "error": reply.error},
        )
        if message.reply_to:
            await self.publish_message(
                reply, topic_id=TopicId(f"{REPLY_TOPIC}.{message.reply_to}", session_id)
            )


async def register_sessions(runtime, store: "SessionStore", responder) -> None:
    """Register the session agent type and subscribe it to the turn topic."""
    await SessionAgent.register(
        runtime, SESSION_AGENT_TYPE, lambda: SessionAgent(store, responder)
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=TURN_TOPIC, agent_type=SESSION_AGENT_TYPE)
    )


def _parse(stamp: str) -> datetime | None:
    try:
        value = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


async def _maybe_await(value):
    if asyncio.iscoroutine(value):
        return await value
    return value


__all__ = [
    "CHANNELS",
    "KINDS",
    "REPLY_TOPIC",
    "SESSION_AGENT_TYPE",
    "TURN_TOPIC",
    "Reply",
    "SessionAgent",
    "SessionKey",
    "SessionManager",
    "SessionRecord",
    "SessionStore",
    "Turn",
    "register_sessions",
    "resolve",
]
