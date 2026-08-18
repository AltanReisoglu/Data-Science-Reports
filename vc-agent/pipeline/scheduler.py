"""Scheduled work, delegated to OpenClaw's scheduler.

The pipeline has no scheduler of its own (`docs/18 §11`), and this module does
not add one. It is a **translator**: our vocabulary for "when" and "what" into
the shape `cron.add` wants, and back again for listing.

### Why delegate rather than build

`docs/17` argues the control plane should end up ours, and it should. But timing
is the one piece OpenClaw already does well and already runs: its scheduler lives
in the Gateway process, persists jobs in SQLite, survives restarts, and reschedules
overdue jobs instead of replaying them at boot. Rebuilding that to run a nightly
scan would be a week spent re-deriving decisions we already measured.

The wake path needed no new plumbing either: our `vc-agent` MCP server is already
registered with OpenClaw, so a job whose payload is a turn for OpenClaw's agent
can simply ask that agent to call one of our tools.

So this is deliberately thin, and thin is the point — when the scheduler does
become ours, this file is what gets replaced, and nothing else has to move.

### The honest limit

Scheduling only runs while OpenClaw's Gateway runs. On this host it is a systemd
*user* service with `Linger=no`, which means it stops when the session ends. A
job that quietly stopped firing is the worst failure a scheduler has, so `jobs()`
reports the Gateway being unreachable as its own state rather than an empty list.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

import openclaw_control

# Only the three schedule kinds that mean something for scheduled *work*.
# `on-exit` and `stream` are event sources that supervise a command, need
# `cron.triggers.enabled`, and carry a different trust class; they are out of
# scope here rather than half-supported.
KINDS = ("cron", "every", "at")

TZ = "Europe/Istanbul"

_EVERY = re.compile(r"^\s*(\d+)\s*(dk|dakika|sa|saat|gun|gün|m|h|d)\s*$", re.I)
_AFTER = re.compile(r"^\s*(\d+)\s*(dk|dakika|sa|saat|m|h)\s+sonra\s*$", re.I)
_DAILY = re.compile(r"^\s*her\s*g[uü]n\s+(\d{1,2}):(\d{2})\s*$", re.I)

_UNIT_MS = {"dk": 60_000, "dakika": 60_000, "m": 60_000,
            "sa": 3_600_000, "saat": 3_600_000, "h": 3_600_000,
            "gun": 86_400_000, "gün": 86_400_000, "d": 86_400_000}


class WhenError(ValueError):
    """The `when` string is not one this module accepts."""


def parse_when(when: str) -> dict[str, Any]:
    """`"her gün 09:00"` / `"30dk"` / `"20dk sonra"` → an OpenClaw schedule.

    Three forms, not five, and the refusal names the three. A scheduler that
    guesses at an unparsed string is a scheduler that fires at a time nobody
    chose.
    """
    text = (when or "").strip()

    daily = _DAILY.match(text)
    if daily:
        hour, minute = int(daily.group(1)), int(daily.group(2))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise WhenError(f"{text!r}: saat 00:00–23:59 aralığında olmalı")
        return {"kind": "cron", "expr": f"{minute} {hour} * * *", "tz": TZ}

    after = _AFTER.match(text)
    if after:
        ms = int(after.group(1)) * _UNIT_MS[after.group(2).lower()]
        at = datetime.now(timezone.utc) + timedelta(milliseconds=ms)
        return {"kind": "at", "at": at.isoformat().replace("+00:00", "Z")}

    every = _EVERY.match(text)
    if every:
        ms = int(every.group(1)) * _UNIT_MS[every.group(2).lower()]
        if ms < 60_000:
            raise WhenError(f"{text!r}: en kısa aralık 1 dakika")
        return {"kind": "every", "everyMs": ms}

    raise WhenError(
        f"{when!r} anlaşılmadı. Üç biçim var: 'her gün 09:00' · '30dk' · '20dk sonra'"
    )


def build_job(name: str, when: str, ask: str, *, session: str = "isolated",
              to: str = "") -> dict[str, Any]:
    """The full `cron.add` params for one job.

    The payload is always `agentTurn`. `command` and `script` payloads exist and
    are not used: both are shell, and the decision about shell belongs to the
    approval gate on `/api/openclaw`, not to a job definition that runs unattended
    at 3am.

    `sessionTarget` is `isolated` by default for the reason OpenClaw gives — a
    scheduled run should not inherit, or pollute, the conversation someone is
    having.

    `to` is an optional `channel:address` delivery target. It is never defaulted:
    a job with no target still runs and its result lands in the task log, whereas
    guessing an address to send to is the kind of helpful that mails a stranger.
    """
    if not (name or "").strip():
        raise WhenError("işin bir adı olmalı")
    if not (ask or "").strip():
        raise WhenError("işin ne yapacağı yazılmalı")

    job: dict[str, Any] = {
        "name": name.strip(),
        "schedule": parse_when(when),
        "sessionTarget": session,
        "wakeMode": "now",
        "payload": {"kind": "agentTurn", "message": ask.strip()},
    }
    if to:
        channel, _, address = to.partition(":")
        if not address.strip():
            raise WhenError(f"teslimat hedefi 'kanal:adres' olmalı — {to!r} değil")
        job["delivery"] = {"mode": "announce", "channel": channel.strip(),
                           "to": address.strip()}
    return job


def parse_command(text: str) -> dict[str, Any]:
    """One typed `/openclaw schedule ...` line into an action.

    Four shapes, and the separator is a bar because it is the one character that
    survives being typed in Turkish on any keyboard and never appears in a time
    expression:

        (empty)                          → list
        sil <id>                         → remove
        <when> | <what>                  → create
        <when> | <what> > <channel:to>   → create, delivered somewhere

    Delivery is optional and *not* defaulted. A job with no target still runs and
    its result lands in the task log; guessing a chat id to send to would be the
    kind of helpful that mails a stranger.
    """
    body = (text or "").strip()
    if not body:
        return {"action": "list"}

    if body.split(" ", 1)[0].lower() in ("sil", "kaldır", "kaldir", "remove", "rm"):
        job_id = body.split(" ", 1)[1].strip() if " " in body else ""
        if not job_id:
            raise WhenError("silinecek işin id'si gerekiyor: schedule sil <id>")
        return {"action": "remove", "id": job_id}

    if "|" not in body:
        raise WhenError(
            "Biçim: schedule <ne zaman> | <ne yapsın>   ·   örnek: "
            "schedule her gün 05:00 | bana merhaba de"
        )

    when, _, rest = body.partition("|")
    what, sep, target = rest.partition(">")
    job = {"action": "create", "when": when.strip(), "ask": what.strip(),
           "to": target.strip() if sep else ""}
    if not job["ask"]:
        raise WhenError("işin ne yapacağı yazılmalı — bardan sonrası boş")
    return job


def describe(schedule: dict[str, Any]) -> str:
    """An OpenClaw schedule back in our words, for listing."""
    kind = (schedule or {}).get("kind", "")
    if kind == "cron":
        expr = schedule.get("expr", "")
        parts = expr.split()
        if len(parts) == 5 and parts[2:] == ["*", "*", "*"]:
            return f"her gün {int(parts[1]):02d}:{int(parts[0]):02d}"
        return f"cron: {expr}"
    if kind == "every":
        ms = int(schedule.get("everyMs", 0))
        if ms % 86_400_000 == 0:
            return f"{ms // 86_400_000} günde bir"
        if ms % 3_600_000 == 0:
            return f"{ms // 3_600_000} saatte bir"
        return f"{ms // 60_000} dakikada bir"
    if kind == "at":
        return f"bir kez · {schedule.get('at', '')}"
    return kind or "?"


async def create(name: str, when: str, ask: str, *, session: str = "isolated",
                 to: str = "") -> dict[str, Any]:
    return await openclaw_control.call(
        "cron.add", build_job(name, when, ask, session=session, to=to)
    )


async def jobs() -> dict[str, Any]:
    """Scheduled jobs, or why we cannot say.

    An unreachable Gateway returns `reachable: False` rather than an empty list,
    because "no jobs" and "cannot ask" are different answers and only one of them
    means nothing is scheduled.
    """
    outcome = await openclaw_control.call("cron.list", {})
    if not outcome.get("ok"):
        return {"reachable": False, "jobs": [],
                "note": outcome.get("error", "OpenClaw Gateway'e ulaşılamadı"),
                "linger_warning": LINGER_NOTE}

    raw = outcome.get("result") or {}
    items = raw.get("jobs") if isinstance(raw, dict) else raw
    out = []
    for job in items or []:
        out.append({
            "id": job.get("id", ""),
            "name": job.get("name", ""),
            "when": describe(job.get("schedule") or {}),
            "enabled": job.get("enabled", True),
            "last": (job.get("lastRun") or {}).get("status", ""),
        })
    return {"reachable": True, "jobs": out, "note": "", "linger_warning": LINGER_NOTE}


async def remove(job_id: str) -> dict[str, Any]:
    return await openclaw_control.call("cron.remove", {"id": job_id})


LINGER_NOTE = (
    "Zamanlama yalnız OpenClaw Gateway ayaktayken çalışır. Bu makinede servis "
    "systemd kullanıcı servisi ve Linger=no — oturum kapanınca zamanlama da durur."
)

__all__ = ["KINDS", "LINGER_NOTE", "TZ", "WhenError", "build_job", "create",
           "describe", "jobs", "parse_command", "parse_when", "remove"]
