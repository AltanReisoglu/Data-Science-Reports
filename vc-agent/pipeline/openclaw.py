"""The outbound half of the bridge: our agent reaching into OpenClaw.

`mcp_server.py` lets OpenClaw ask us things. This is the other direction — our
analyst getting at the conversations OpenClaw is holding, through
`openclaw mcp serve`, which is OpenClaw acting as an MCP server over stdio.

The mechanism is one already in the codebase: `conversation._mcp_workbench`
attaches DeepWiki exactly this way. A second `McpWorkbench` in the same list is
all the plumbing this needs, which is the point of a workbench being a tool
*source*.

### Why this direction is the one with a gate on it

Inbound is read-only by construction. Outbound is not. `sessions_send` puts a
message into somebody's Telegram, and there is no undo, no draft, no preview —
the person's phone buzzes. `sessions_spawn` starts an agent that will itself act.
So this module is the first place in the project where a tool the model can choose
has a real-world effect on a person who did not ask for it, and it is why
`gateway/approval.py` exists and is switched on by default.

It is also **off by default at the connection level** (`VC_MCP_OPENCLAW`). Two
gates, deliberately: attaching the workbench is an operator decision, and each
outbound call is a second one. A single switch would mean turning on channel
awareness also turns on the ability to message people.

### What it does not touch

OpenClaw's config holds a Telegram bot token and provider API keys. Nothing here
reads `~/.openclaw/openclaw.json`. We talk to the `openclaw` binary, it talks to
its own secrets, and none of them are ours to hold or to log.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Any

import config

# What `openclaw mcp serve` actually exposes, observed against openclaw
# 2026.7.1-2 on 2026-08-14 rather than taken from the docs:
#
#   attachments_fetch  conversation_get  conversations_list  events_poll
#   events_wait  messages_read  messages_send  permissions_list_open
#   permissions_respond
#
# The gate matches on substrings (`config.OUTBOUND_TOOLS`), not on this list —
# upstream renames and adds tools between releases, and an exact allowlist fails
# open when it does. This is documentation of what we saw, not the enforcement.
READ_TOOLS = (
    "conversations_list", "conversation_get", "messages_read",
    "events_poll", "events_wait", "attachments_fetch", "permissions_list_open",
)

# Two, and the second is the one worth stopping to look at.
#
# `messages_send` is the obvious one: it puts text on somebody's phone.
#
# `permissions_respond` answers OpenClaw's *own* pending permission prompts — the
# approvals a human is supposed to give before OpenClaw runs something. An agent
# that can call it can approve OpenClaw's requests on the operator's behalf,
# which turns two independent gates into one. It carries no obvious verb like
# "send", so the default markers were widened to include `respond` and `approve`
# after this was observed. Guessing the list would have missed it.
WRITE_TOOLS = ("messages_send", "permissions_respond")


@dataclass
class Attachment:
    """The result of trying to attach OpenClaw. Never raises; always explains."""

    workbench: Any = None
    status: str = "not attempted"
    tools: list[str] = field(default_factory=list)

    @property
    def attached(self) -> bool:
        return self.workbench is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "attached": self.attached,
            "status": self.status,
            "tools": self.tools,
            "outbound_gated": not config.ALLOW_OUTBOUND,
        }


def available() -> bool:
    """Is the `openclaw` binary on PATH at all?"""
    return shutil.which(config.OPENCLAW_BIN) is not None


async def attach(*, timeout: float = 30.0) -> Attachment:
    """Start `openclaw mcp serve` as a tool source, or explain why not.

    A dead or missing OpenClaw must not break the conversation — the same rule
    the DeepWiki attachment follows. The agent keeps every local tool and simply
    does not have channel awareness, which is a smaller loss than no chat at all.
    """
    if not config.MCP_OPENCLAW:
        return Attachment(status="disabled (VC_MCP_OPENCLAW=0)")
    if not available():
        return Attachment(status=f"unavailable ({config.OPENCLAW_BIN} not on PATH)")

    try:
        from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

        params = StdioServerParams(
            command=config.OPENCLAW_BIN,
            args=["mcp", "serve"],
            read_timeout_seconds=timeout,
        )
        workbench = McpWorkbench(server_params=params)
        await workbench.start()
        names = [str(tool.get("name")) for tool in await workbench.list_tools()]
    except Exception as exc:  # noqa: BLE001 — a dead bridge is a status, not a crash
        return Attachment(status=f"unavailable ({type(exc).__name__}: {exc})")

    return Attachment(workbench=workbench, status=f"connected ({len(names)} tools)", tools=names)


def guidance(attachment: Attachment) -> str:
    """The paragraph added to the system prompt when OpenClaw is attached.

    Written to make the gate legible to the model. An agent that does not know a
    call needs approval will keep retrying it and report failure; one that knows
    will ask the person instead, which is the behaviour worth having.
    """
    if not attachment.attached:
        return ""
    outbound = [t for t in attachment.tools if any(w in t for w in WRITE_TOOLS)]
    lines = [
        "",
        "You are connected to OpenClaw, which holds this operator's messaging "
        "channels. You can read what is happening there.",
        "`openclaw_call` reaches its control plane — sessions, channels, cron, "
        "nodes, health. `openclaw_methods` lists what is reachable.",
        # Written after the agent answered "I don't have permission" to a camera
        # request when the real answer was "no node is paired". Both refusals
        # sound the same to the person asking and only one of them is fixable by
        # them, so the difference is worth a sentence.
        "Device capabilities — camera, screen, location — come from **paired "
        "nodes**, not from OpenClaw itself. Before saying you cannot do something "
        "with a device, call `openclaw_call` with `node.list` and say what you "
        "found: no node paired is a different answer from a node that refused, "
        "and only the first one tells the operator what to go and fix.",
    ]
    if outbound and not config.ALLOW_OUTBOUND:
        lines.append(
            f"Sending is gated: {', '.join(sorted(outbound))} will be refused until the "
            "operator approves that specific call. If you are refused, say what you "
            "wanted to send and to whom, and let them decide — do not retry."
        )
    return "\n".join(lines)


__all__ = ["Attachment", "READ_TOOLS", "WRITE_TOOLS", "attach", "available", "guidance"]
