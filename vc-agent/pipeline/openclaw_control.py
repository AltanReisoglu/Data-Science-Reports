"""Full control of OpenClaw from this side — and the reason it is not one tool.

`openclaw mcp serve` exposes nine tools, all about channel conversations. That is
a slice. OpenClaw's actual control plane is the Gateway WebSocket protocol (v4),
and it carries `sessions.*`, `channels.*`, `agents.*`, `cron.*`, `config.*`,
`secrets.*`, `exec.approval.*`, `update.*` — the whole machine.

Reaching it does not need a WebSocket client. `openclaw gateway call <method>`
is a generic RPC escape hatch, and the CLI already holds the gateway token and
the paired device identity, so there is no handshake to reimplement and no
pairing to approve. Verified: `health`, `status`, `sessions.list`,
`channels.status` and `cron.list` all answer.

### Why this file is mostly a classification table

"100% control" and "the agent can do anything to your machine" are the same
sentence read twice. This surface includes:

* `config.set` — rewrite any setting, including auth mode
* `secrets.*` — the credential store
* `exec.approval.resolve` — **approve arbitrary command execution**
* `channels.logout` — drop the WhatsApp session, which costs a re-pair
* `update.run` — replace the running software
* `sessions.delete` — destroy conversation history

An agent with unrestricted access to those is not an assistant with tools; it is
a root shell with a language model in front of it. So methods are classified, and
the classification **fails closed**: anything not explicitly listed as read-only
needs approval, which means a method OpenClaw adds next release is gated by
default rather than silently available.

Three tiers:

| tier | behaviour | examples |
|---|---|---|
| `READ` | runs freely | `health`, `sessions.list`, `cron.list` |
| `WRITE` | approval gate, one call at a time | `sessions.reset`, `cron.create`, `chat.send` |
| `FORBIDDEN` | refused outright, no approval path | `config.set`, `secrets.*`, `exec.approval.*`, `update.run` |

`FORBIDDEN` is not "gated harder". There is no approval prompt, because a prompt
implies a decision worth making in the moment, and "should the agent be allowed
to approve arbitrary shell commands on your behalf" is not that. An operator who
genuinely wants it sets `VC_OPENCLAW_ALLOW_ADMIN=1` and takes it on knowingly.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Any

import config

# Read-only, verified against the running gateway or documented in docs/13 §3.5.
# Everything here answers a question and changes nothing.
READ = frozenset({
    "health", "status", "system-presence", "gateway.identity.get",
    "diagnostics.stability", "models.list", "usage.status", "usage.cost",
    "sessions.usage", "sessions.list", "chat.history", "channels.status",
    "agents.list", "agents.workspace.list", "agents.workspace.get",
    "audit.list", "tasks.list", "tasks.get", "cron.list", "commands.list",
    "skills.list", "tools.catalog", "tools.effective", "node.list",
    "node.describe", "config.get", "config.schema", "update.status",
    "plugin.approval.list", "exec.approval.list",
})

# Changes something, but nothing that cannot be undone by hand. Gated.
WRITE = frozenset({
    "sessions.create", "sessions.send", "sessions.steer", "sessions.abort",
    "sessions.patch", "sessions.reset", "sessions.compact",
    "chat.send", "chat.abort", "chat.inject",
    "cron.create", "cron.update", "cron.delete", "cron.run",
    "tasks.cancel", "wake", "node.invoke", "node.event",
    "agents.create", "agents.update",
})

# No approval path. See the module docstring.
FORBIDDEN_PREFIXES = (
    "config.set", "config.patch", "config.apply", "config.unset",
    "secrets.", "update.run", "exec.approval.resolve", "exec.approval.respond",
    "plugin.approval.resolve", "wizard.", "channels.logout",
    "sessions.delete", "agents.delete", "node.pair", "devices.",
)

ALLOW_ADMIN = os.getenv("VC_OPENCLAW_ALLOW_ADMIN", "") in ("1", "true", "yes")


def classify(method: str) -> str:
    """`read` · `write` · `forbidden`. Unknown methods are `write`, not `read`."""
    name = (method or "").strip()
    if not name:
        return "forbidden"
    if any(name.startswith(p) for p in FORBIDDEN_PREFIXES):
        return "forbidden"
    if name in READ:
        return "read"
    if name in WRITE:
        return "write"
    # Fail closed. A method added by a future OpenClaw release is gated until
    # somebody looks at it and decides it belongs in READ.
    return "write"


def available() -> bool:
    return shutil.which(config.OPENCLAW_BIN) is not None


async def call(
    method: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Invoke one Gateway method through the CLI. Never raises; always reports."""
    if not available():
        return {"ok": False, "error": f"{config.OPENCLAW_BIN} is not on PATH"}

    tier = classify(method)
    if tier == "forbidden" and not ALLOW_ADMIN:
        return {
            "ok": False,
            "tier": tier,
            "error": (
                f"{method} is not reachable from here. It can change credentials, "
                "config, or approve command execution, and there is no per-call "
                "approval for that. Set VC_OPENCLAW_ALLOW_ADMIN=1 to take it on."
            ),
        }

    argv = [
        config.OPENCLAW_BIN, "gateway", "call", method,
        "--params", json.dumps(params or {}, ensure_ascii=False),
        "--json", "--timeout", str(int(timeout * 1000)),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout + 10)
    except asyncio.TimeoutError:
        return {"ok": False, "tier": tier, "error": f"{method} timed out after {timeout}s"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "tier": tier, "error": f"{type(exc).__name__}: {exc}"}

    text = out.decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        detail = err.decode("utf-8", "replace").strip() or text
        return {"ok": False, "tier": tier, "error": detail[:600]}

    # The CLI prints config warnings before the JSON body; take the last object.
    start = text.find("{")
    body: Any = text
    if start >= 0:
        try:
            body = json.loads(text[start:])
        except json.JSONDecodeError:
            body = text[-2000:]
    return {"ok": True, "tier": tier, "method": method, "result": body}


# --------------------------------------------------------------------------- gate
#
# The approval gate matches on **tool names** (`config.OUTBOUND_TOOLS`), and that
# is the right shape for `messages_send` — the name says what it does.
#
# This tool breaks that assumption. `openclaw_call` is one name covering a
# hundred methods, and its danger lives in an *argument*: `openclaw_call` matches
# no outbound marker, so `openclaw_call("sessions.reset")` would have gone
# through ungated. A name-based gate cannot see inside a call.
#
# So the classification gets its own hook on the same point. Nothing about the
# existing gate changes; this one runs beside it and blocks on the tier of the
# method being asked for.


def make_gate_hook(gate=None):
    """Build the `before_tool_call` handler, bound to a specific approval gate.

    A factory rather than a module function because the gate is injectable — the
    tests use their own, and a hook that reached for the singleton would record
    requests somewhere the caller cannot see. Found by a test that approved a
    request and watched the next call get blocked anyway.
    """

    def hook(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if str(payload.get("tool") or "") != "openclaw_call":
                return {}

            arguments = payload.get("arguments") or {}
            method = str(arguments.get("method") or "")
            tier = classify(method)
            if tier == "read":
                return {}
            if tier == "forbidden" and not ALLOW_ADMIN:
                return {
                    "block": True,
                    "reason": (
                        f"{method} changes credentials, config, or command approvals. "
                        "There is no per-call approval for that."
                    ),
                }

            from gateway import approval as approval_module

            target = gate or approval_module.GATE
            # `require`, not `check`: the name `openclaw_call` matches no outbound
            # marker, so the name-based path would wave this straight through.
            return target.require(
                f"openclaw_call:{method}", arguments,
                session=str(payload.get("session") or ""),
                reason=f"{method} changes OpenClaw state and needs approval before it runs.",
            )
        except Exception as exc:  # noqa: BLE001
            # Same asymmetry as the main gate: a broken guard closes, never opens.
            return {"block": True, "reason": f"openclaw gate failed: {type(exc).__name__}: {exc}"}

    return hook


# Kept as a name for callers that want the default-gate behaviour directly.
gate_hook = make_gate_hook()


def install_gate(registry=None, gate=None) -> None:
    from gateway import hooks as hooks_module

    target = registry or hooks_module.REGISTRY
    target.unregister(hooks_module.BEFORE_TOOL_CALL, "openclaw_method_gate")
    target.register(
        hooks_module.BEFORE_TOOL_CALL,
        make_gate_hook(gate),
        name="openclaw_method_gate",
        order=-90,
    )


def methods() -> dict[str, list[str]]:
    return {
        "read": sorted(READ),
        "write": sorted(WRITE),
        "forbidden_prefixes": sorted(FORBIDDEN_PREFIXES),
        "admin_enabled": ALLOW_ADMIN,
    }


__all__ = [
    "ALLOW_ADMIN", "FORBIDDEN_PREFIXES", "READ", "WRITE",
    "available", "call", "classify", "gate_hook", "install_gate", "methods",
]
