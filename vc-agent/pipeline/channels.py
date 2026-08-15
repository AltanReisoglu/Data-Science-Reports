"""Channels: the surfaces a turn can arrive from, under one contract.

`gateway/sessions.py` has always known a turn's *origin* — it is the first field
of a session key. What did not exist is the thing on the other side of that
origin: something that can also **send**. Sessions could route an incoming
message; nothing could deliver an outgoing one anywhere except back down the HTTP
response that asked.

That gap is what stops the web UI from being a channel in OpenClaw's sense
(docs/13 §1). A channel there is a two-way adapter: parse inbound, enforce access,
format outbound. This module is that contract, and three implementations of it.

    web       the dashboard — inbound over HTTP, outbound over the event queue
    cli       a terminal — inbound as a call, outbound to stdout
    openclaw  the bridge — inbound over our MCP server, outbound over messages_send

**The architecture did not change.** docs/15 §5 says we do not write Telegram or
WhatsApp adapters, and we still do not: OpenClaw holds those, and `openclaw` here
is one adapter that speaks to all of them at once. What is new is that our own
surfaces now satisfy the same contract, which is what makes a relay between them
possible at all.

### Access control lives here

OpenClaw's channel adapters do four things, and the third is the one worth
copying: authentication, inbound parsing, **access control**, outbound
formatting. A channel that can send to a person needs to know *which* people it
is allowed to send to, and that is a property of the channel rather than of the
agent that asked. `Channel.may_send_to` is where a mistaken peer id gets stopped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

# Module scope for `@message_handler`'s type inference — see docs/06 §11.
from autogen_core import (  # noqa: E402
    MessageContext,
    RoutedAgent,
    TypeSubscription,
    message_handler,
)

import config
from gateway.sessions import REPLY_TOPIC, Reply

log = logging.getLogger("vcagent.channels")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Inbound:
    """A message arriving from somewhere. What the gateway routes to a session."""

    channel: str
    peer: str
    text: str
    kind: str = "dm"
    account: str = "default"
    # Set when this message is a relay of something that arrived elsewhere.
    # `relay.py` reads it to stop a message going round for ever.
    origin: str = ""
    message_id: str = ""
    at: str = field(default_factory=_now)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def relayed(self) -> bool:
        return bool(self.origin)


@dataclass
class Outbound:
    """A message leaving. What a channel is asked to deliver."""

    peer: str
    text: str
    kind: str = "dm"
    account: str = "default"
    origin: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Delivery:
    """What happened. Never an exception — a channel that fails must say so."""

    ok: bool
    channel: str
    detail: str = ""
    remote_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "channel": self.channel, "detail": self.detail,
                "remote_id": self.remote_id}


@runtime_checkable
class Channel(Protocol):
    """A surface that can receive and deliver. The contract `caps.Channel` names."""

    name: str

    def may_send_to(self, peer: str) -> bool: ...

    async def send(self, message: Outbound) -> Delivery: ...


# --------------------------------------------------------------------------- web


class WebChannel:
    """The dashboard. Outbound lands in a per-peer queue the UI drains.

    The web UI is request/response, so "the gateway sends you a message" has no
    natural place to arrive. A queue gives it one: `/api/inbox` returns whatever
    accumulated, and a relayed reply from OpenClaw shows up in the conversation
    the same way a locally generated one does.
    """

    name = "web"

    def __init__(self) -> None:
        self.queues: dict[str, list[dict[str, Any]]] = {}

    def may_send_to(self, peer: str) -> bool:
        # Anyone with the dashboard open is already the operator; the gate that
        # matters for this channel is on the HTTP server, not here.
        return True

    async def send(self, message: Outbound) -> Delivery:
        self.queues.setdefault(message.peer, []).append(
            {"text": message.text, "origin": message.origin, "at": _now(),
             "meta": message.meta}
        )
        return Delivery(True, self.name, detail="queued")

    def drain(self, peer: str) -> list[dict[str, Any]]:
        items = self.queues.get(peer, [])
        self.queues[peer] = []
        return items

    def waiting(self, peer: str) -> int:
        return len(self.queues.get(peer, []))


# --------------------------------------------------------------------------- cli


class CliChannel:
    """A terminal. Outbound is printed; there is nobody to authorise."""

    name = "cli"

    def __init__(self, sink=print) -> None:
        self._sink = sink

    def may_send_to(self, peer: str) -> bool:
        return True

    async def send(self, message: Outbound) -> Delivery:
        self._sink(f"[{message.origin or 'gateway'}] {message.text}")
        return Delivery(True, self.name, detail="printed")


# --------------------------------------------------------------------------- openclaw


class OpenClawChannel:
    """OpenClaw, as one adapter standing in front of every channel it holds.

    Inbound does not go through this class: OpenClaw reaches us by calling our
    MCP server (`mcp_server.py`), which is already a session-attributed path.
    This is the **outbound** half — `messages_send` over the workbench attached
    in `openclaw.py`.

    Every send goes through the same `GatedWorkbench` the agent's tool calls do,
    so it is subject to the approval gate. The relay's own permission is separate
    and is described in `gateway/relay.py`; both have to say yes.
    """

    name = "openclaw"

    def __init__(self, workbench=None, *, allowed: tuple[str, ...] = ()) -> None:
        self.workbench = workbench
        # Empty means "anywhere OpenClaw will accept", which is the useful default
        # for a single-operator tool and the wrong one the moment it is not.
        self.allowed = allowed
        self.sent: list[Outbound] = []

    def may_send_to(self, peer: str) -> bool:
        if not peer:
            return False
        if not self.allowed:
            return True
        return any(peer == a or peer.startswith(a) for a in self.allowed)

    async def send(self, message: Outbound) -> Delivery:
        if self.workbench is None:
            return Delivery(False, self.name, detail="openclaw is not attached")
        if not self.may_send_to(message.peer):
            return Delivery(False, self.name, detail=f"peer {message.peer!r} is not allowed")

        try:
            result = await self.workbench.call_tool(
                "messages_send",
                {"conversationId": message.peer, "text": message.text},
            )
        except Exception as exc:  # noqa: BLE001 — a dead bridge is a delivery failure
            return Delivery(False, self.name, detail=f"{type(exc).__name__}: {exc}")

        text = result.to_text()
        if result.is_error:
            # This is where an approval refusal surfaces, and it reads as one.
            return Delivery(False, self.name, detail=text[:300])
        self.sent.append(message)
        return Delivery(True, self.name, detail=text[:200])


# --------------------------------------------------------------------------- core
#
# A channel becomes an agent subscribed to `reply.<name>`, so delivery is a
# **subscription** rather than a dictionary lookup. Adding a surface stops being
# "register it in a dict the sender must consult" and becomes "subscribe it to a
# topic" — the sender does not learn about it, which is the point of pub/sub.


class ChannelAgent(RoutedAgent):
    """Delivers replies for one channel. Never raises; a channel is not the turn."""

    def __init__(self, channel: Channel) -> None:
        super().__init__(f"{channel.name} channel")
        self._channel = channel
        self.delivered: list[Delivery] = []

    @message_handler
    async def deliver(self, message: Reply, ctx: MessageContext) -> None:
        # The topic source carries the session; the agent key does too, but the
        # message is what the far side needs addressed.
        peer = self.id.key
        try:
            result = await self._channel.send(
                Outbound(peer=peer, text=message.text or message.error,
                         origin="gateway", meta={"session": message.session})
            )
        except Exception as exc:  # noqa: BLE001
            result = Delivery(False, self._channel.name, detail=f"{type(exc).__name__}: {exc}")
        if not result.ok:
            log.warning("channel %s failed to deliver: %s", self._channel.name, result.detail)
        self.delivered.append(result)


def agent_type_for(channel_name: str) -> str:
    return f"channel.{channel_name}"


async def register_channel(runtime, channel: Channel) -> str:
    """Register one channel as an agent and subscribe it to its reply topic."""
    agent_type = agent_type_for(channel.name)
    await ChannelAgent.register(runtime, agent_type, lambda c=channel: ChannelAgent(c))
    await runtime.add_subscription(
        TypeSubscription(topic_type=f"{REPLY_TOPIC}.{channel.name}", agent_type=agent_type)
    )
    return agent_type


# --------------------------------------------------------------------------- registry


class ChannelRegistry:
    """The channels this gateway has. Small on purpose."""

    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}

    def add(self, channel: Channel) -> Channel:
        self._channels[channel.name] = channel
        return channel

    def get(self, name: str) -> Channel | None:
        return self._channels.get(name)

    def names(self) -> list[str]:
        return sorted(self._channels)

    async def send(self, channel: str, message: Outbound) -> Delivery:
        target = self._channels.get(channel)
        if target is None:
            return Delivery(False, channel, detail=f"no channel named {channel!r}")
        return await target.send(message)

    def report(self) -> dict[str, Any]:
        return {
            name: {
                "type": type(channel).__name__,
                "restricted": bool(getattr(channel, "allowed", ())),
            }
            for name, channel in sorted(self._channels.items())
        }


REGISTRY = ChannelRegistry()


def install_defaults(registry: ChannelRegistry | None = None, *, workbench=None) -> ChannelRegistry:
    target = registry or REGISTRY
    target.add(WebChannel())
    target.add(CliChannel())
    target.add(OpenClawChannel(workbench))
    return target


__all__ = [
    "REGISTRY", "Channel", "ChannelRegistry", "CliChannel", "Delivery",
    "Inbound", "OpenClawChannel", "Outbound", "WebChannel", "install_defaults",
]
