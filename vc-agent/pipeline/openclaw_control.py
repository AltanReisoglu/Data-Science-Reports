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
| `WRITE` | approval gate, one call at a time | `sessions.reset`, `cron.add`, `chat.send` |
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
import time
import uuid
from typing import Any

import config

# Read-only, verified against the running gateway or documented in docs/13 §3.5.
# Everything here answers a question and changes nothing.
READ = frozenset({
    "health", "status", "system-presence", "gateway.identity.get",
    "diagnostics.stability", "models.list", "usage.status", "usage.cost",
    "sessions.usage", "sessions.list", "chat.history", "channels.status",
    "agents.list", "agents.workspace.list", "agents.workspace.get",
    "audit.list", "tasks.list", "tasks.get", "cron.list", "cron.get",
    "cron.status", "cron.runs", "commands.list",
    "skills.list", "tools.catalog", "tools.effective", "node.list",
    "node.describe", "config.get", "config.schema", "update.status",
    "plugin.approval.list", "exec.approval.list",
})

# Changes something, but nothing that cannot be undone by hand. Gated.
WRITE = frozenset({
    "sessions.create", "sessions.send", "sessions.steer", "sessions.abort",
    "sessions.patch", "sessions.reset", "sessions.compact",
    "chat.send", "chat.abort", "chat.inject",
    # Measured against the running Gateway, not guessed from the docs: it is
    # `cron.add`/`cron.remove`, and `cron.create` answers "unknown method".
    "cron.add", "cron.update", "cron.remove", "cron.run", "cron.trigger",
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


# --------------------------------------------------------------------------- typed line
#
# The operator's escape hatch. `/openclaw sessions.list` in the chat box goes
# straight to the Gateway — no model, no tool schema, no paraphrase of what
# happened. OpenClaw has the same thing in `openclaw gateway call`, and it exists
# for the same reason: when you are debugging, a layer that summarises the answer
# for you is the layer you are fighting.
#
# **This path is not gated, and that is the decision, not an oversight.** The
# approval gate exists because a *model* chose to act. Here a human typed the
# method and its arguments, and asking them to approve what they just typed is
# theatre that trains people to click yes. `FORBIDDEN` still refuses — those
# change credentials or approve command execution, and no amount of intent makes
# a config rewrite a good thing to do from a chat box.
#
# What keeps it honest: the typed line and the tier land in the transcript before
# the call runs, so the record shows what was asked even if the answer never came.

PREFIX = "/openclaw"


def parse_line(line: str) -> tuple[str, dict[str, Any], str]:
    """`cron.run {"id": "x"}` -> (method, params, error). Never raises.

    Returns the error as a string rather than raising because the caller is a
    chat box: a traceback is not an answer, and every failure here is a typo
    somebody can fix in the next line.
    """
    text = (line or "").strip()
    if text.startswith(PREFIX):
        text = text[len(PREFIX):].strip()
    if not text:
        return "", {}, f"Usage: {PREFIX} <method> [json]   ·   {PREFIX} methods"

    method, _, rest = text.partition(" ")
    method, rest = method.strip(), rest.strip()
    if not rest:
        return method, {}, ""

    try:
        params = json.loads(rest)
    except json.JSONDecodeError as exc:
        return method, {}, f"params is not JSON: {exc.msg} at position {exc.pos}"
    if not isinstance(params, dict):
        return method, {}, f"params must be a JSON object, got {type(params).__name__}"
    return method, params, ""


# Talking to OpenClaw's own agent, as opposed to calling its control plane.
#
# `chat.send` starts a run and returns `{"runId": ..., "status": "started"}` —
# the answer is not in the response. `--expect-final` does not change that for
# this method, measured against 2026.7.1-2. So the reply is collected by polling
# `chat.history`, which is a read.
#
# The session key is ours and nobody else's: OpenClaw's real conversations live
# under `agent:main:dashboard:<uuid>` and the channel sessions elsewhere, so a
# question typed here can never land in the middle of a Telegram thread.
#
# `deliver` is left out on purpose. In OpenClaw the flag is opt-in
# (`params.request.deliver === true`, `agent-delivery-phase.ts:84`), so omitting
# it means the answer comes back to us and is not delivered to any channel.
# Typing a question here must not make somebody's phone buzz.
CHAT_SESSION_PREFIX = "agent:main:vcagent:"


def _reply_text(message: dict[str, Any]) -> str:
    """`content` is a list of typed parts; take the text ones."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p).strip()
    return ""


async def ask(
    text: str, *, peer: str = "local", timeout: float = 180.0, poll: float = 2.0
) -> dict[str, Any]:
    """Put a question to OpenClaw's agent and wait for its answer."""
    session_key = f"{CHAT_SESSION_PREFIX}{peer or 'local'}"

    before = await call("chat.history", {"sessionKey": session_key}, timeout=30.0)
    seen = len(before.get("result", {}).get("messages", []) or []) if before.get("ok") else 0

    started = await call(
        "chat.send",
        {
            "sessionKey": session_key,
            "message": text,
            # Required by the schema; a retried transport failure must not ask
            # the agent the same question twice.
            "idempotencyKey": uuid.uuid4().hex,
        },
        timeout=60.0,
    )
    if not started.get("ok"):
        return {**started, "method": "chat.send", "asked": text}

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(poll)
        history = await call("chat.history", {"sessionKey": session_key}, timeout=30.0)
        if not history.get("ok"):
            continue
        messages = history.get("result", {}).get("messages", []) or []
        for message in messages[seen:]:
            # An assistant row with no text shows up while the run is still
            # going. Treating it as the answer would print a blank bubble and
            # call it done.
            if message.get("role") == "assistant" and _reply_text(message):
                return {
                    "ok": True, "tier": "chat", "method": "chat.send",
                    "asked": text, "session_key": session_key,
                    "result": _reply_text(message),
                }

    return {
        "ok": False, "tier": "chat", "method": "chat.send", "asked": text,
        "session_key": session_key,
        "error": (
            f"asked, but no answer within {timeout:.0f}s. The run may still be going — "
            f"`/openclaw chat.history {{\"sessionKey\": \"{session_key}\"}}` shows where it got to."
        ),
    }


def looks_like_a_method(token: str) -> bool:
    """Is this a Gateway method name, or the first word of a sentence?

    The rule is the dot, and it is deliberately dumb: every Gateway method is
    `noun.verb` and no sentence starts that way. A misspelt `sessions.lst` stays
    a method and fails with "unknown method", which is the error the person needs;
    silently turning their typo into a chat message would hide it.
    """
    return "." in token.strip()


def plan_line(line: str) -> dict[str, Any]:
    """What a typed line *would* do, without doing it.

    Split out of `run_line` so a caller can decide about a line before it runs.
    The route needs that: a sentence goes to OpenClaw's own agent, and that agent
    has shell access this gate cannot see, so the decision has to happen before
    the bytes leave. Classifying inside `run_line` would have meant asking the
    question after answering it.

    `mode` is `local` (answered here, no Gateway), `method` (a Gateway call) or
    `sentence` (a turn for OpenClaw's agent). `tier` is only meaningful for
    `method` and comes from `classify`.
    """
    text = (line or "").strip()
    if text.startswith(PREFIX):
        text = text[len(PREFIX):].strip()

    first = text.split(" ", 1)[0] if text else ""
    if first == "methods":
        return {"mode": "local", "method": "methods", "params": {}, "text": text,
                "tier": "local", "error": ""}
    # `schedule` is a *deterministic* subcommand, and the reason it exists is a
    # measured failure: asking OpenClaw's agent in prose to "create a daily task"
    # produced a clarifying question instead of a job, and the answer to that
    # question went to a different agent because it carried no prefix. A sentence
    # is the right shape for asking; it is the wrong shape for a thing that must
    # either exist afterwards or say why not.
    if first == "schedule":
        return {"mode": "schedule", "method": "cron.add", "params": {},
                "text": text[len(first):].strip(), "tier": "write", "error": ""}
    # `foto` is the same decision as `schedule`, for the same reason. Asking in
    # prose works — OpenClaw's agent has a shell and would take the picture — but
    # *it* would choose where the file lands, and a path chosen by a model is one
    # we can only recover by parsing prose or scanning a directory. So we name the
    # file and write the command; the agent's only job is to run it.
    if first in ("foto", "photo"):
        return {"mode": "foto", "method": "exec", "params": {},
                "text": text[len(first):].strip(), "tier": "write", "error": ""}
    if text and not looks_like_a_method(first):
        return {"mode": "sentence", "method": "", "params": {}, "text": text,
                "tier": "", "error": ""}

    method, params, error = parse_line(line)
    return {"mode": "method", "method": method, "params": params, "text": text,
            "tier": classify(method) if method else "", "error": error}


async def run_line(line: str, *, peer: str = "local", timeout: float = 30.0) -> dict[str, Any]:
    """Parse a typed line and run it. The whole `/openclaw` feature, server side."""
    plan = plan_line(line)

    # `methods` is not a Gateway method; it is the local question "what can I even
    # type here", which is the first thing anybody wants and the one answer that
    # never needs the binary to be running.
    if plan["mode"] == "local":
        return {"ok": True, "method": "methods", "tier": "local", "result": methods()}
    if plan["mode"] == "sentence":
        return await ask(plan["text"], peer=peer)
    if plan["error"]:
        return {"ok": False, "method": plan["method"], "error": plan["error"], "tier": ""}
    return await call(plan["method"], plan["params"], timeout=timeout)


__all__ = [
    "ALLOW_ADMIN", "CHAT_SESSION_PREFIX", "FORBIDDEN_PREFIXES", "PREFIX", "READ", "WRITE",
    "ask", "available", "call", "classify", "gate_hook", "install_gate",
    "looks_like_a_method", "methods", "parse_line", "plan_line", "run_line",
]
