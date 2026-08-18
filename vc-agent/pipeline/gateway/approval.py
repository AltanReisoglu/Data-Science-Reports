"""The approval gate, finally load-bearing.

`observability` has carried this sentence since the intervention handler was
written: *"it is wired and tested so that the first mutating tool has a gate
waiting for it rather than needing one retrofitted."* The OpenClaw bridge is that
tool. `sessions_send` puts a message into a real person's Telegram, and there is
no undo — so it is exactly the case the gate was built for, and switching it on
is not a precaution, it is the point.

**Default deny, by name.** A tool whose name contains any of `config.OUTBOUND_TOOLS`
is blocked unless an operator approved that specific call. Matching on substrings
rather than an exact list is deliberate: the tool names come from a *remote*
server that can rename or add tools between releases, so an allowlist of exact
names fails open the moment upstream changes. A substring rule fails closed — a
new `sessions_send_media` is blocked without anyone updating a constant.

**The failure mode is a block.** `hooks.py` quarantines a hook that raises and
carries on, which is right for a hook that decorates a turn and wrong for one that
guards it. So this gate does its own work inside a `try` and returns
`{"block": True}` on error: if the gate itself is broken, nothing outbound goes
out. The asymmetry is the whole design.

**Approvals are one call, once.** A request records the tool and a digest of its
arguments; approving it clears exactly that call. The next identical call needs a
new approval — otherwise "yes" to one message becomes "yes" to every message.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import config

from . import hooks as hooks_module


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(tool: str, arguments: dict[str, Any] | None) -> str:
    blob = json.dumps(arguments or {}, ensure_ascii=False, default=str, sort_keys=True)
    return hashlib.sha256(f"{tool}\0{blob}".encode("utf-8")).hexdigest()[:16]


def is_outbound(tool: str) -> bool:
    """Does this tool reach outside in a way that cannot be taken back?"""
    name = (tool or "").lower()
    return any(marker and marker in name for marker in config.OUTBOUND_TOOLS)


@dataclass
class Request:
    id: str
    tool: str
    digest: str
    keys: list[str]
    session: str
    requested_at: str
    status: str = "pending"          # pending | approved | denied | expired
    decided_at: str = ""
    note: str = ""
    # Kept in memory only, so the operator can see what they are approving. It is
    # never written to the audit ledger — that stores metadata (`policy.py`).
    preview: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "digest": self.digest,
            "keys": self.keys,
            "session": self.session,
            "status": self.status,
            "requested_at": self.requested_at,
            "decided_at": self.decided_at,
            "note": self.note,
            "preview": self.preview,
        }


class ApprovalGate:
    """Pending outbound calls, and the decisions made about them."""

    def __init__(self, *, allow_all: bool | None = None) -> None:
        # `allow_all` is the operator saying "I accept the blast radius". There is
        # no partial setting that is honest about what it does.
        self.allow_all = config.ALLOW_OUTBOUND if allow_all is None else allow_all
        self._requests: dict[str, Request] = {}
        self._granted: set[str] = set()

    # ------------------------------------------------------------ decisions

    def check(self, tool: str, arguments: dict[str, Any] | None, *, session: str = "") -> dict[str, Any]:
        """Decide one call by **tool name**. Returns a `before_tool_call` result.

        The name-based path: a tool whose name matches an outbound marker is
        gated, everything else passes. Right for `messages_send`.
        """
        if not is_outbound(tool):
            return {}
        return self.require(tool, arguments, session=session)

    def require(
        self, tool: str, arguments: dict[str, Any] | None, *, session: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        """Gate this call regardless of its name.

        For callers that decided on grounds the name cannot carry — `openclaw_call`
        is one tool covering a hundred Gateway methods, and its blast radius is in
        an argument. `check` would wave it through because "openclaw_call" matches
        no marker; this is the entry point that does not consult the name.
        """
        if self.allow_all:
            return {}

        token = digest(tool, arguments)
        if token in self._granted:
            # Consumed: approval covers this call, not every call like it.
            self._granted.discard(token)
            return {"approved": True}

        request = self.request(tool, arguments, session=session)
        # The id is appended, never replaced by a caller's `reason`. It used to be
        # part of the default text only, so a caller that explained the block in
        # its own words silently dropped the one thing the operator needed: the
        # UI reads the id back out of this sentence to decide whether to draw an
        # Approve button, and with no id it drew "this has no approval path"
        # instead — over a request that was sitting in the queue, approvable.
        head = reason or f"{tool} reaches outside and needs approval."
        return {
            "block": True,
            "reason": f"{head} Approve request {request.id} to let it through.",
            "approval_id": request.id,
        }

    def request(
        self, tool: str, arguments: dict[str, Any] | None, *, session: str = ""
    ) -> Request:
        token = digest(tool, arguments)
        for existing in self._requests.values():
            if existing.digest == token and existing.status == "pending":
                return existing

        args = arguments or {}
        request = Request(
            id=uuid.uuid4().hex[:12],
            tool=tool,
            digest=token,
            keys=sorted(str(k) for k in args),
            session=session,
            requested_at=_now(),
            preview={k: _clip(v) for k, v in args.items()},
        )
        self._requests[request.id] = request
        return request

    def approve(self, request_id: str, *, note: str = "") -> dict[str, Any]:
        request = self._requests.get(request_id)
        if request is None:
            return {"ok": False, "reason": "unknown request"}
        if request.status != "pending":
            return {"ok": False, "reason": f"already {request.status}"}
        request.status = "approved"
        request.decided_at = _now()
        request.note = note
        self._granted.add(request.digest)
        return {"ok": True, "request": request.as_dict()}

    def deny(self, request_id: str, *, note: str = "") -> dict[str, Any]:
        request = self._requests.get(request_id)
        if request is None:
            return {"ok": False, "reason": "unknown request"}
        request.status = "denied"
        request.decided_at = _now()
        request.note = note
        self._granted.discard(request.digest)
        return {"ok": True, "request": request.as_dict()}

    # ------------------------------------------------------------ queries

    def pending(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self._requests.values() if r.status == "pending"]

    def all(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self._requests.values()]

    def get(self, request_id: str) -> Request | None:
        return self._requests.get(request_id)

    # ------------------------------------------------------------ hook

    def hook(self, payload: dict[str, Any]) -> dict[str, Any]:
        """`before_tool_call` handler. Blocks on its own failure, never opens."""
        try:
            return self.check(
                str(payload.get("tool") or payload.get("tool_name") or ""),
                payload.get("arguments") or {},
                session=str(payload.get("session") or ""),
            )
        except Exception as exc:  # noqa: BLE001
            # A broken gate must not become an open door. The registry would
            # quarantine this hook and continue; that is right for a decorating
            # hook and wrong for a guarding one, so the block happens here.
            return {"block": True, "reason": f"approval gate failed: {type(exc).__name__}: {exc}"}


def _clip(value: Any, limit: int = 400) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


GATE = ApprovalGate()


def install(registry: hooks_module.HookRegistry | None = None, gate: ApprovalGate | None = None) -> None:
    """Put the gate on `before_tool_call`, ahead of any decorating hook."""
    target = registry or hooks_module.REGISTRY
    target.unregister(hooks_module.BEFORE_TOOL_CALL, "approval_gate")
    target.register(
        hooks_module.BEFORE_TOOL_CALL,
        (gate or GATE).hook,
        name="approval_gate",
        order=-100,
    )


__all__ = ["GATE", "ApprovalGate", "Request", "digest", "install", "is_outbound"]
