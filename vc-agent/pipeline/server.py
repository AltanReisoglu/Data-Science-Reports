"""Local backend for the VC pipeline: serves the chat UI and runs scans.

    .venv/bin/python pipeline/server.py          # http://127.0.0.1:8000

Single-user by design (docs/03): no auth, no tenants, bound to loopback. FastAPI
and uvicorn were already in the venv as transitive dependencies, so this adds no
new packages.

What the backend buys over the static page:

* **The chat can reach a model.** With `VC_LLM_*` set, a question goes to the
  configured LLM with the scan's facts as its ground truth; without one, the same
  deterministic answers as before. The UI always shows which path answered.
* **Scans start from the interface.** `POST /api/scan` runs `scan.py` as a
  subprocess and streams its output, so the funnel is watchable while it runs.
  A subprocess rather than an in-process call: a scan that hangs or dies must not
  take the server with it.
* **Past runs stay reachable.** Every scan JSON on disk can be loaded back.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import (  # noqa: E402
    HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse,
)
from pydantic import BaseModel  # noqa: E402

import answers as answers_module  # noqa: E402
import config  # noqa: E402
import conversation as conversation_module  # noqa: E402
import dashboard  # noqa: E402

WEB = Path(__file__).resolve().parent / "web"

# There is a backend here, so the per-company live check can be offered.
dashboard.LIVE_AVAILABLE = True
app = FastAPI(title="VC pipeline", docs_url=None, redoc_url=None)


# --------------------------------------------------------------------------- state


class ScanRun:
    """One scan subprocess and the log it is producing."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.lines: list[str] = []
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.args: dict = {}

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, query: str, days: int, limit: int) -> None:
        if self.running:
            raise RuntimeError("a scan is already running")
        self.lines = []
        self.started_at = time.time()
        self.finished_at = None
        self.args = {"query": query, "days": days, "limit": limit}
        command = [
            sys.executable, str(Path(__file__).resolve().parent / "scan.py"),
            "--query", query, "--days", str(days), "--limit", str(limit),
        ]
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            with self.lock:
                self.lines.append(line.rstrip("\n"))
                # A scan is bounded; the log does not need to be.
                if len(self.lines) > 4000:
                    del self.lines[:1000]
        self.process.wait()
        self.finished_at = time.time()

    def snapshot(self) -> dict:
        with self.lock:
            lines = list(self.lines)
        return {
            "running": self.running,
            "args": self.args,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.process.poll() if self.process else None,
            "lines": lines,
        }


RUN = ScanRun()


def _current_scan() -> dict | None:
    try:
        return load_scan()
    except HTTPException:
        return None


def _start_scan(query: str, days: int = 7) -> None:
    RUN.start(query, days, config.THRESHOLDS.max_candidates)


# One registry for the whole server, one conversation per session. The gateway
# owns which session a turn belongs to; this process owns the agents.
CHAT = conversation_module.ConversationRegistry(_current_scan, _start_scan)


def _session(peer: str = "local", channel: str = "web"):
    """Resolve the caller to a session. The web UI is one peer among several."""
    return CHAT.get(channel, peer=peer, kind="dm")


def _install_channels() -> None:
    """Register the surfaces this gateway can reach.

    The OpenClaw channel is given the *gated* workbench rather than the raw one,
    so a relayed message is subject to the same approval gate as a tool call the
    agent chose to make. The relay's link is consent to bridge; it is not a way
    around the gate.
    """
    import channels as channels_module

    channels_module.REGISTRY.add(channels_module.WebChannel())
    channels_module.REGISTRY.add(channels_module.CliChannel(sink=lambda _t: None))
    channels_module.REGISTRY.add(channels_module.OpenClawChannel(None))


_install_channels()


@app.on_event("startup")
async def _start_gateway() -> None:
    """Bring up the one runtime, and put sessions and channels on it.

    The responder is the per-session `Conversation`, so the message-driven path
    and the streaming path answer with the same agent and the same context.
    """
    import channels as channels_module
    from gateway import runtime as runtime_module
    from gateway import sessions as sessions_module

    async def responder(session_id: str, turn) -> str:
        record = CHAT.sessions.get(session_id)
        if record is None:
            return ""
        conversation = CHAT.for_session(record)
        final = ""
        async with CHAT.sessions.lock(session_id):
            async for event in conversation.stream(turn.text):
                if event.get("type") == "done":
                    final = event.get("text", "")
        return final

    gateway = runtime_module.GATEWAY
    await gateway.register_once(
        "session",
        lambda rt: sessions_module.register_sessions(rt, CHAT.sessions.store, responder),
    )
    for name in channels_module.REGISTRY.names():
        await gateway.register_once(
            channels_module.agent_type_for(name),
            lambda rt, n=name: channels_module.register_channel(
                rt, channels_module.REGISTRY.get(n)
            ),
        )
    await gateway.start()


@app.on_event("shutdown")
async def _stop_gateway() -> None:
    from gateway import runtime as runtime_module

    await runtime_module.GATEWAY.close()
    await CHAT.close()


async def _bind_openclaw_channel() -> str:
    """Attach the live OpenClaw workbench to the channel, once the agent has one."""
    import channels as channels_module
    from gateway import workbench as workbench_module

    _, conversation = _session()
    await conversation.ensure()
    channel = channels_module.REGISTRY.get("openclaw")
    if conversation.openclaw is not None and conversation.openclaw.attached:
        channel.workbench = workbench_module.GatedWorkbench(
            conversation.openclaw.workbench, session_id="relay"
        )
        return conversation.openclaw.status
    return conversation.openclaw.status if conversation.openclaw else "not attempted"


def scan_files() -> list[Path]:
    return sorted(config.OUTPUT.glob("scan-*.json"), reverse=True)


def load_scan(name: str | None = None) -> dict:
    files = scan_files()
    if not files:
        raise HTTPException(404, "No scan has been run yet. Start one from the composer.")
    target = files[0]
    if name:
        match = [f for f in files if f.name == name]
        if not match:
            raise HTTPException(404, f"No scan named {name}")
        target = match[0]
    data = json.loads(target.read_text(encoding="utf-8"))
    data["_source_name"] = target.name
    return data


# --------------------------------------------------------------------------- routes


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB / "index.html").read_text(encoding="utf-8")


@app.get("/style.css", response_class=PlainTextResponse)
def style() -> PlainTextResponse:
    # Served from `dashboard.STYLE` so the page and the static export cannot
    # drift apart: one stylesheet, two surfaces.
    extra = (WEB / "app.css").read_text(encoding="utf-8") if (WEB / "app.css").exists() else ""
    return PlainTextResponse(dashboard.STYLE + extra, media_type="text/css")


@app.get("/app.js", response_class=PlainTextResponse)
def script() -> PlainTextResponse:
    return PlainTextResponse((WEB / "app.js").read_text(encoding="utf-8"), media_type="text/javascript")


@app.get("/api/state")
def state(scan: str | None = None, peer: str = "local") -> JSONResponse:
    _, conversation = _session(peer)
    files = scan_files()
    if not files:
        return JSONResponse(
            {
                "has_scan": False,
                "live_llm": config.live_llm_available(),
                "missing_llm": config.missing_llm_settings(),
                "thesis_is_placeholder": config.THESIS.is_placeholder,
                "thesis": config.THESIS.as_prompt(),
                "scans": [],
            }
        )
    data = load_scan(scan)
    entries = answers_module.catalogue(data)
    opening_title, opening_html = entries["summary"]
    return JSONResponse(
        {
            "has_scan": True,
            "live_llm": config.live_llm_available(),
            "missing_llm": config.missing_llm_settings(),
            "thesis_is_placeholder": config.THESIS.is_placeholder,
            "thesis": config.THESIS.as_prompt(),
            "source": data["_source_name"],
            "query": data.get("query"),
            "days": data.get("days"),
            "mode": data.get("mode"),
            "funnel": data.get("funnel", {}),
            "banners": dashboard._banners(data),
            "opening": {"title": opening_title, "html": opening_html},
            "candidates": [
                dashboard._candidate_parts(c)["name"] for c in data.get("candidates", [])
            ],
            "scans": [f.name for f in files[:30]],
            # Per session now, not per process: `CHAT` is a registry of
            # conversations rather than one conversation. Missed in the refactor
            # because this route has no test — see `test_server_routes.py`.
            "mcp": conversation.mcp_status,
            "chat_cost": conversation.cost(),
        }
    )


class Question(BaseModel):
    question: str
    scan: str | None = None
    use_model: bool = True


@app.post("/api/ask")
async def ask(payload: Question) -> JSONResponse:
    data = load_scan(payload.scan)
    result = await answers_module.answer(
        payload.question, data, prefer_model=payload.use_model
    )
    return JSONResponse(result)


class ScanRequest(BaseModel):
    query: str = "ai infrastructure"
    days: int = 7
    limit: int = 5


@app.post("/api/scan")
def start_scan(payload: ScanRequest) -> JSONResponse:
    try:
        RUN.start(payload.query, payload.days, payload.limit)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return JSONResponse({"started": True, "args": RUN.args})


@app.get("/api/scan")
def scan_status() -> JSONResponse:
    return JSONResponse(RUN.snapshot())


class LiveRequest(BaseModel):
    company: str
    scan: str | None = None


@app.post("/api/live")
async def live_check(payload: LiveRequest) -> JSONResponse:
    """Check one pipeline company against live sources, right now.

    Deliberately not routed through the model even when one is configured. A
    button that says "Check now" promises an action, and an action must not
    depend on a model choosing to call a tool. The agent has the same capability
    for conversational questions; this is the direct path.
    """
    import asyncio

    import live as live_module

    data = load_scan(payload.scan)
    company = live_module.find_company(data, payload.company)
    if company is None:
        raise HTTPException(
            404,
            f"No candidate named {payload.company!r}. "
            f"Candidates: {', '.join(live_module.company_names(data)) or 'none'}",
        )
    report = await asyncio.to_thread(live_module.refresh, company)
    return JSONResponse(
        {
            "path": "live",
            "title": f"{report.company} — live",
            "text": None,
            "html": answers_module.render_live(report),
            "report": report.as_dict(),
        }
    )


class ChatTurn(BaseModel):
    question: str
    peer: str = "local"


@app.post("/api/chat")
async def chat(payload: ChatTurn):
    """Stream a live agent turn as Server-Sent Events.

    Only when an LLM is configured. Without one the client uses `/api/ask`,
    which answers from the scan data — a different guarantee, and the interface
    says which one it got.
    """
    if not config.live_llm_available():
        raise HTTPException(409, "No LLM configured; use /api/ask.")

    record, conversation = _session(payload.peer)
    lane = CHAT.sessions.lock(record.id)

    async def events():
        # One turn at a time *per session*. Two people on two channels no longer
        # queue behind each other, which is the point of the session lane.
        async with lane:
            CHAT.sessions.record_turn(record, "user", payload.question)
            final = ""
            async for event in conversation.stream(payload.question):
                if event.get("type") == "done":
                    final = event.get("text", "")
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if final:
                CHAT.sessions.record_turn(record, "assistant", final)

            # If this session is bridged, the answer goes out too. The relay's
            # own rules decide whether it actually crosses.
            if final:
                from gateway import relay as relay_module

                for delivery in await relay_module.RELAY.forward_out(record.id, final):
                    yield (
                        "data: "
                        + json.dumps({"type": "relay", **delivery.as_dict()}, ensure_ascii=False)
                        + "\n\n"
                    )
        yield "data: {\"type\": \"end\"}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/stop")
def chat_stop(peer: str = "local") -> JSONResponse:
    _, conversation = _session(peer)
    return JSONResponse({"cancelled": conversation.cancel()})


@app.post("/api/chat/reset")
async def chat_reset(peer: str = "local") -> JSONResponse:
    record, _ = _session(peer)
    await CHAT.reset(record.id)
    return JSONResponse({"reset": True, "session": record.id})


# --------------------------------------------------------------------------- gateway


@app.get("/api/sessions")
def sessions_list() -> JSONResponse:
    """Every conversation this gateway is holding, newest first."""
    return JSONResponse(
        {
            "agent": config.AGENT_ID,
            "state_dir": str(config.STATE),
            "dm_scope": config.SESSION_POLICY.dm_scope,
            "sessions": [
                {
                    "id": r.id, "channel": r.channel, "kind": r.kind, "peer": r.peer,
                    "turns": r.turns, "title": r.title,
                    "started": r.session_started_at, "last": r.last_interaction_at,
                }
                for r in CHAT.sessions.list()
            ],
        }
    )


@app.get("/api/sessions/{session_id}/transcript")
def session_transcript(session_id: str, limit: int = 200) -> JSONResponse:
    if CHAT.sessions.get(session_id) is None:
        raise HTTPException(404, f"No session {session_id}")
    return JSONResponse({"session": session_id, "entries": CHAT.sessions.store.read(session_id, limit)})


@app.delete("/api/sessions/{session_id}")
async def session_delete(session_id: str) -> JSONResponse:
    await CHAT.reset(session_id)
    return JSONResponse({"deleted": session_id})


# --------------------------------------------------------------------------- relay


@app.get("/api/channels")
def channels_list() -> JSONResponse:
    """Which surfaces this gateway can reach, and how far each may send."""
    import channels as channels_module

    return JSONResponse(channels_module.REGISTRY.report())


@app.get("/api/relays")
def relays_list() -> JSONResponse:
    from gateway import relay as relay_module

    return JSONResponse(relay_module.RELAY.report())


class LinkRequest(BaseModel):
    channel: str = "openclaw"
    peer: str
    direction: str = "both"
    max_hops: int = 4
    session_peer: str = "local"


@app.post("/api/relays")
async def relay_link_route(payload: LinkRequest) -> JSONResponse:
    if payload.channel == "openclaw":
        # Binding is lazy: the workbench only exists once the agent has been
        # built, and building it on demand keeps a dead bridge from delaying start.
        status = await _bind_openclaw_channel()
        import channels as channels_module

        if channels_module.REGISTRY.get("openclaw").workbench is None:
            raise HTTPException(409, f"openclaw is not attached: {status}")
    return relay_link(payload)


def relay_link(payload: LinkRequest) -> JSONResponse:
    """Bridge this session to a conversation elsewhere.

    **Creating the link is the consent.** Messages inside it are not approved one
    by one — that would be forty clicks nobody reads by the tenth. One deliberate
    act, scoped to one pair of addresses, listed here, revocable.
    """
    from gateway import relay as relay_module

    record, _ = _session(payload.session_peer)
    try:
        link = relay_module.RELAY.link(
            record.id, payload.channel, payload.peer,
            direction=payload.direction, max_hops=payload.max_hops,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(link.as_dict())


@app.delete("/api/relays/{link_id}")
def relay_unlink(link_id: str) -> JSONResponse:
    from gateway import relay as relay_module

    if not relay_module.RELAY.unlink(link_id):
        raise HTTPException(404, f"no relay {link_id}")
    return JSONResponse({"unlinked": link_id})


class RelayInbound(BaseModel):
    text: str
    sender: str = ""


@app.post("/api/relays/{link_id}/inbound")
async def relay_inbound(link_id: str, payload: RelayInbound) -> JSONResponse:
    """A message arriving from the far side, to be answered by our agent.

    The webhook OpenClaw would call. The relay decides whether it crosses; if it
    does, the text is framed as relayed before the agent ever sees it.
    """
    from gateway import relay as relay_module

    message = relay_module.RELAY.accept_in(link_id, payload.text, sender=payload.sender)
    if message is None:
        return JSONResponse({"accepted": False, "reason": "declined by relay"})

    link = relay_module.RELAY.get(link_id)
    record = CHAT.sessions.get(link.session_id)
    if record is None:
        raise HTTPException(404, f"session {link.session_id} is gone")

    if not config.live_llm_available():
        return JSONResponse({"accepted": True, "answered": False,
                             "reason": "no LLM configured"})

    conversation = CHAT.for_session(record)
    reply = ""
    async with CHAT.sessions.lock(record.id):
        CHAT.sessions.record_turn(record, "user", message.text, origin=message.origin)
        async for event in conversation.stream(message.text):
            if event.get("type") == "done":
                reply = event.get("text", "")

    # The answer goes back the way the question came, with the origin attached so
    # the loop check can recognise it.
    deliveries = await relay_module.RELAY.forward_out(
        record.id, reply, origin="", hops=1
    ) if reply else []
    return JSONResponse({
        "accepted": True, "answered": bool(reply), "reply": reply,
        "delivered": [d.as_dict() for d in deliveries],
    })


@app.get("/api/inbox")
def inbox(peer: str = "local") -> JSONResponse:
    """Messages the gateway has for the dashboard — relayed replies land here."""
    import channels as channels_module

    web = channels_module.REGISTRY.get("web")
    return JSONResponse({"peer": peer, "messages": web.drain(peer) if web else []})


@app.get("/api/approvals")
def approvals_list() -> JSONResponse:
    """Outbound tool calls waiting on a person. Empty is the normal state."""
    from gateway import approval as approval_module

    return JSONResponse(
        {
            "gate": "open" if config.ALLOW_OUTBOUND else "closed",
            "outbound_markers": list(config.OUTBOUND_TOOLS),
            "pending": approval_module.GATE.pending(),
        }
    )


class ApprovalDecision(BaseModel):
    note: str = ""


@app.post("/api/approvals/{request_id}/approve")
def approval_approve(request_id: str, payload: ApprovalDecision) -> JSONResponse:
    from gateway import approval as approval_module

    outcome = approval_module.GATE.approve(request_id, note=payload.note)
    if not outcome["ok"]:
        raise HTTPException(404, outcome["reason"])
    return JSONResponse(outcome)


@app.post("/api/approvals/{request_id}/deny")
def approval_deny(request_id: str, payload: ApprovalDecision) -> JSONResponse:
    from gateway import approval as approval_module

    outcome = approval_module.GATE.deny(request_id, note=payload.note)
    if not outcome["ok"]:
        raise HTTPException(404, outcome["reason"])
    return JSONResponse(outcome)


def _runtime_report() -> dict:
    from gateway import runtime as runtime_module

    return runtime_module.GATEWAY.report()


@app.get("/api/health")
def health() -> JSONResponse:
    from gateway import approval as approval_module
    from gateway import hooks as hooks_module

    record, conversation = _session()
    return JSONResponse(
        {
            "ok": True,
            "live_llm": config.live_llm_available(),
            "scans": len(scan_files()),
            "scan_running": RUN.running,
            "mcp": conversation.mcp_status,
            "openclaw": conversation.openclaw.as_dict() if conversation.openclaw else None,
            "state_dir": str(config.STATE),
            "sessions": len(CHAT.sessions.list()),
            "session": record.id,
            "context": conversation.context_report(),
            "approvals_pending": len(approval_module.GATE.pending()),
            "hooks_quarantined": hooks_module.REGISTRY.quarantined(),
            "runtime": _runtime_report(),
        }
    )


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="VC pipeline local server")
    parser.add_argument("--host", default="127.0.0.1", help="loopback by default; this is a single-user tool")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    mode = (
        "live agent (tools, memory, MCP)" if config.live_llm_available()
        else "rules only (no LLM configured)"
    )
    print(f"\n  VC pipeline · http://{args.host}:{args.port}")
    print(f"  answering: {mode}")
    if config.THESIS.is_placeholder:
        print("  thesis:    PLACEHOLDER — thesis-fit scores are uncalibrated")
    print(f"  scans:     {len(scan_files())} on disk\n")

    uvicorn.run("server:app" if args.reload else app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
