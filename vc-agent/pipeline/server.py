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
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import (  # noqa: E402
    HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse,
)
from pydantic import BaseModel  # noqa: E402

import answers as answers_module  # noqa: E402
import config  # noqa: E402
import conversation as conversation_module  # noqa: E402
import dashboard  # noqa: E402
import openclaw_control  # noqa: E402
import runlog  # noqa: E402
import stages  # noqa: E402

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
        # Core stages lifted out of the same stdout the log comes from. A scan is
        # a subprocess, so there is no in-process queue to share — see `stages.py`.
        self.stages: list[dict] = []
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
        self.stages = []
        self.started_at = time.time()
        self.finished_at = None
        self.args = {"query": query, "days": days, "limit": limit}
        # A scan is the one path that really assembles a team, so it is the one
        # the flow screen has the most to say about. Recorded like a chat turn.
        self.run = runlog.LOG.begin("scan", f"tarama · {query}")
        command = [
            sys.executable, str(Path(__file__).resolve().parent / "scan.py"),
            "--query", query, "--days", str(days), "--limit", str(limit),
        ]
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            # Ask the scan to narrate its core mechanisms. Only the server sets
            # this: `python scan.py` by hand stays readable.
            env={**os.environ, stages.STREAM_ENV: "1"},
        )
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            text = line.rstrip("\n")
            stage = stages.parse_line(text)
            if stage is not None:
                # Machine lines never reach the human log: the operator reading a
                # scan should not have to skip past JSON to find the funnel.
                with self.lock:
                    self.stages.append(stage)
                    if len(self.stages) > 500:
                        del self.stages[:200]
                self.run.event(stage)
                continue
            with self.lock:
                self.lines.append(text)
                # A scan is bounded; the log does not need to be.
                if len(self.lines) > 4000:
                    del self.lines[:1000]
        self.process.wait()
        self.finished_at = time.time()
        self.run.end("done" if self.process.poll() == 0 else "error")

    def snapshot(self, *, since: int = 0) -> dict:
        with self.lock:
            lines = list(self.lines)
            # `since` lets the panel take only what it has not drawn yet; the log
            # itself is still sent whole because the client renders it whole.
            new_stages = self.stages[since:]
            total = len(self.stages)
        return {
            "running": self.running,
            "args": self.args,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.process.poll() if self.process else None,
            "lines": lines,
            "stages": new_stages,
            "stage_count": total,
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

    # The code-exec container belongs to the process, not to a call: bringing it
    # up costs two or three seconds and the user would pay that on the first
    # question. Starting it here also means a broken Docker shows up at boot
    # rather than in the middle of a demo.
    import codeexec as codeexec_module

    if await codeexec_module.start():
        print(f"  code exec: açık · {config.CODE_EXEC_IMAGE}", flush=True)


@app.on_event("shutdown")
async def _stop_gateway() -> None:
    import codeexec as codeexec_module
    from gateway import runtime as runtime_module

    await codeexec_module.stop()
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


@app.get("/rough.js", response_class=PlainTextResponse)
def rough() -> PlainTextResponse:
    """Hand-drawn SVG primitives — the browser half of `docs/diagrams/rough.py`,
    so the screen and the PDF draw in the same hand.

    Nothing loads this at the moment: its only consumer was the mechanism panel,
    which is no longer part of the chat. The route and the file stay because the
    interface that replaces the panel will want the same hand, and re-deriving
    these primitives from the Python side is the expensive way to get them back.
    """
    return PlainTextResponse((WEB / "rough.js").read_text(encoding="utf-8"), media_type="text/javascript")


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
def scan_status(since: int = 0) -> JSONResponse:
    """The scan log, plus whichever core stages the caller has not seen yet."""
    return JSONResponse(RUN.snapshot(since=since))


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
            # The turn is recorded as it streams. The id goes out first so the
            # interface can offer the flow screen for *this* question rather than
            # for whatever ran last — two tabs asking at once would otherwise
            # each open the other's turn.
            run = runlog.LOG.begin("chat", payload.question, record.id)
            yield f"data: {json.dumps({'type': 'run', 'id': run.id})}\n\n"
            final = ""
            status = "done"
            async for event in conversation.stream(payload.question):
                if event.get("type") == "done":
                    final = event.get("text", "")
                elif event.get("type") in ("cancelled", "error"):
                    status = event["type"]
                run.event(event)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            run.end(status)

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


@app.get("/api/mechanisms")
def mechanisms() -> JSONResponse:
    """The mechanism catalogue, plus the core runtime's own counters.

    The panel's labels come from here rather than from JavaScript so there is one
    place to correct when the code moves. The counters come with them because the
    core lane makes a claim — "this turn did not touch core" — and a claim like
    that should be a reading, not a sentence somebody typed into a template.
    """
    from gateway.runtime import GATEWAY

    return JSONResponse({
        "mechanisms": stages.catalogue(),
        "runs": stages.RUNS,
        "core_idle_note": stages.CORE_IDLE_NOTE,
        "runtime": GATEWAY.report(),
    })


@app.get("/akis", response_class=HTMLResponse)
def akis() -> str:
    """The flow screen: one turn, drawn.

    A separate page rather than a panel in the chat. The panel had to explain the
    machine in the margin of a conversation and did both jobs badly; here the
    drawing gets the whole viewport, and the chat gets its width back.
    """
    return (WEB / "akis.html").read_text(encoding="utf-8")


@app.get("/akis.js", response_class=PlainTextResponse)
def akis_script() -> PlainTextResponse:
    return PlainTextResponse((WEB / "akis.js").read_text(encoding="utf-8"),
                             media_type="text/javascript")


@app.get("/figures.js", response_class=PlainTextResponse)
def figures_script() -> PlainTextResponse:
    """The mechanism drawings — the browser half of the hap deck's diagrams.

    Recovered from the panel that used to sit in the chat (`3af7313`). The panel
    went; these did not, because they are the part that actually explains a turn,
    and they are the same drawings as `docs/pdf/hap-autogen.pdf` in the same hand.
    """
    return PlainTextResponse((WEB / "figures.js").read_text(encoding="utf-8"),
                             media_type="text/javascript")


@app.get("/patterns.js", response_class=PlainTextResponse)
def patterns_script() -> PlainTextResponse:
    """The eight design-pattern drawings, ported from `docs/diagrams/figures.py`.

    Same coordinates, same palette, same caption lines as the deck. Two separate
    drawings of one claim start telling it two different ways within a release.
    """
    return PlainTextResponse((WEB / "patterns.js").read_text(encoding="utf-8"),
                             media_type="text/javascript")


class TeamTurn(BaseModel):
    kind: str
    question: str
    max_messages: int = 6


@app.post("/api/team")
async def team_run(payload: TeamTurn):
    """Run one question through a real AutoGen team, streamed as SSE.

    This is the one path where the flow screen has an actual team to draw. The
    chat path deliberately has none — a single `AssistantAgent` — and saying so
    is the honest answer, but it left the five team types as something we could
    only describe. Here they run.
    """
    import teams as teams_module

    if payload.kind not in teams_module.KINDS:
        raise HTTPException(400, f"unknown team: {payload.kind}")
    if not teams_module.available():
        raise HTTPException(409, "No LLM configured; a team needs a live model.")

    async def events():
        run = runlog.LOG.begin("team", payload.question)
        run.variant = payload.kind
        yield f"data: {json.dumps({'type': 'run', 'id': run.id})}\n\n"
        bus = stages.StageBus()
        status = "done"
        try:
            async for event in teams_module.run(payload.kind, payload.question,
                                                bus=bus, spans=run.spans,
                                                max_messages=payload.max_messages):
                for stage in bus.drain():
                    run.event(stage)
                    yield f"data: {json.dumps(stage, ensure_ascii=False)}\n\n"
                run.event(event)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001 — the stream reports its own failure
            status = "error"
            yield ("data: " + json.dumps({"type": "error",
                                          "message": f"{type(e).__name__}: {e}"}) + "\n\n")
        for stage in bus.drain():
            run.event(stage)
            yield f"data: {json.dumps(stage, ensure_ascii=False)}\n\n"
        run.end(status)
        yield "data: {\"type\": \"end\"}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


class MafTurn(BaseModel):
    question: str
    approval: str = "never_require"


@app.post("/api/maf")
async def maf_run(payload: MafTurn):
    """Run one question through Microsoft Agent Framework instead of AutoGen.

    A comparison surface, not a production path: the pipeline's tools, gate,
    memory and scan all live on the AutoGen side. What this shows is the same
    question in the successor framework, and where its defaults differ.
    """
    import maf as maf_module

    if not maf_module.available():
        raise HTTPException(409, "MAF modu hazır değil: " + maf_module.report()["why"])

    async def events():
        run = runlog.LOG.begin("maf", payload.question)
        yield f"data: {json.dumps({'type': 'run', 'id': run.id})}\n\n"
        bus = stages.StageBus()
        status = "done"
        async for event in maf_module.run(payload.question,
                                          approval=payload.approval, bus=bus):
            for stage in bus.drain():
                run.event(stage)
                yield f"data: {json.dumps(stage, ensure_ascii=False)}\n\n"
            if event.get("type") == "error":
                status = "error"
            run.event(event)
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        for stage in bus.drain():
            run.event(stage)
            yield f"data: {json.dumps(stage, ensure_ascii=False)}\n\n"
        run.end(status)
        yield "data: {\"type\": \"end\"}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


# Sunumda gösterilecek desteler. Beyaz liste, çünkü ad bir yol parçası:
# `docs/pdf` altındaki her şeyi açmak, istemcinin seçtiği bir dizeyi dosya
# yoluna çevirmek demek olurdu. `shots.py`'deki aynı disiplin.
# Etiketler KISA: deste sütunu 32 rem ve dört sekme + kapat düğmesi tek satıra
# sığmalı. Uzun adlarda "Kapat ×" alt satıra düşüyor ve bant iki kat oluyor.
DECKS = {
    "autogen": ("hap-autogen.pdf", "AutoGen"),
    "openclaw": ("hap-openclaw.pdf", "OpenClaw"),
    "openclaw-nis": ("hap-openclaw-nis.pdf", "niş"),
    # Uzun rehber: hap desteler "ne" diye sorana, bu "hangisi, neden" diye
    # sorana cevap veriyor. Sunumda soru gelince açılacak yer burası.
    "rehber": ("rehber-cerceveler.pdf", "Rehber"),
    # Sunum kâğıtları. Deste izleyiciye bakar; bunlar KONUŞANA bakar ve
    # sunum sırasında açılır — o yüzden aynı panelde, en sonda.
    "dort": ("sunum-dort-sayfa.pdf", "4 sayfa"),
    "kart-autogen": ("kart-autogen.pdf", "kart·AG"),
    "kart-openclaw": ("kart-openclaw.pdf", "kart·OC"),
}


@app.get("/api/decks")
def decks() -> JSONResponse:
    """Hangi desteler gösterilebilir. Diskte olmayanı listelemiyoruz."""
    base = Path(__file__).resolve().parent.parent / "docs" / "pdf"
    out = []
    for key, (filename, label) in DECKS.items():
        path = base / filename
        if path.exists():
            out.append({"id": key, "label": label,
                        "pages": _page_count(path),
                        "size_kb": round(path.stat().st_size / 1024)})
    return JSONResponse({"decks": out, "default": "autogen"})


def _page_count(path: Path) -> int:
    """Sayfa sayısı, PDF kütüphanesi olmadan.

    İki yol birden okunuyor ve büyüğü alınıyor: `/Type /Page` girdileri sıkıştırılmış
    nesne akışlarının içinde kalabiliyor, `/Count` ise sayfa ağacının kökünde duruyor
    ama iç düğümlerde de geçiyor. Üç destede ikisi de aynı sayıyı verdi (43 · 19 · 17);
    ayrışsalardı gezinme yanlış yerde durur, hata vermezdi.
    """
    try:
        data = path.read_bytes()
        pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
        counts = [int(m) for m in re.findall(rb"/Count\s+(\d+)", data)]
        return max([pages] + counts) or 1
    except Exception:  # noqa: BLE001 — sayfa sayısı bir gösterge, koşuyu düşüremez
        return 1


@app.get("/deck/{deck_id}")
def deck(deck_id: str) -> Response:
    """One slide deck as a PDF, by id — never by path.

    The id is looked up in a whitelist and never joined onto a path from
    client input. Same rule as the camera frames: the request names a key,
    the server owns the filename.
    """
    entry = DECKS.get(deck_id)
    if entry is None:
        raise HTTPException(404, "no such deck")
    path = Path(__file__).resolve().parent.parent / "docs" / "pdf" / entry[0]
    if not path.exists():
        raise HTTPException(404, "deck not built")
    # Deste bir kez yükleniyor ve gezinmeyi tarayıcının görüntüleyicisi
    # yapıyor; önbellek başlığı sekmeler arası geçişi bedavaya getiriyor.
    return Response(path.read_bytes(), media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="{entry[0]}"',
                             "Cache-Control": "public, max-age=3600"})


@app.get("/api/maf")
def maf_status() -> JSONResponse:
    """Whether MAF mode can be offered, and why not when it cannot."""
    import maf as maf_module

    return JSONResponse(maf_module.report())


@app.get("/api/teams")
def team_kinds() -> JSONResponse:
    """Which team types can be run, and what picks the speaker in each."""
    import teams as teams_module

    return JSONResponse({
        "kinds": [{"id": k, "picker": teams_module.PICKER[k]} for k in teams_module.KINDS],
        "roster": [a["name"] for a in teams_module.ROSTER],
        "available": teams_module.available(),
    })


@app.get("/api/runs")
def runs() -> JSONResponse:
    """Recent turns, newest first — the flow screen's own picker."""
    return JSONResponse({"runs": runlog.LOG.listing()})


@app.get("/api/run/{run_id}")
def run_report(run_id: str) -> JSONResponse:
    """Everything the flow screen draws for one turn, in one request.

    `latest` is accepted as an id so the screen can be opened cold — from a
    bookmark, or on a second monitor — without the chat having handed it one.
    """
    from gateway.runtime import GATEWAY

    run = runlog.LOG.latest() if run_id == "latest" else runlog.LOG.get(run_id)
    if run is None:
        raise HTTPException(404, "no such run")
    return JSONResponse({**run.report(), "runtime": GATEWAY.report()})


class OpenClawLine(BaseModel):
    line: str
    peer: str = "local"


def _openclaw_gate(plan: dict, session: str) -> dict | None:
    """Hold a `/openclaw` line for approval, or `None` to let it run.

    Reuses `approval_module.GATE` rather than a second gate: one queue, one set
    of decisions, one place the UI reads pending work from. `require` is the
    entry point that ignores the tool name, which is what this needs — the line
    is not a tool call and its blast radius is in the text, not the name.
    """
    from gateway import approval as approval_module

    mode, tier = plan["mode"], plan["tier"]
    if plan["error"]:
        # A line that did not parse has nothing to approve. Holding it would ask
        # the operator to sign off on a typo, and hide the usage text that tells
        # them what to type instead.
        return None
    if mode == "local" or (mode == "method" and tier in ("read", "forbidden")):
        # `forbidden` goes through so `call` can refuse it in its own words. An
        # approval prompt for something that has no approval path would be a lie.
        return None

    if mode == "sentence":
        tool, args = "openclaw_ask", {"text": plan["text"]}
        reason = (
            "Bu satır OpenClaw'ın kendi ajanına gidiyor. O ajanın kabuk erişimi "
            "var ve şu an onay sormadan çalıştırıyor (exec: mode=full, ask=off); "
            "bizim kapımız içeride ne yapacağını görmez."
        )
    else:
        tool, args = "openclaw_call", {"method": plan["method"], **(plan["params"] or {})}
        reason = f"{plan['method']} OpenClaw'da bir şey değiştirir."

    decision = approval_module.GATE.require(tool, args, session=session, reason=reason)
    if decision.get("block"):
        return {"ok": False, "held": True, "tier": tier or "chat",
                "method": plan["method"] or "chat.send",
                "reason": decision["reason"], "approval_id": decision["approval_id"]}
    return None


async def _openclaw_schedule(plan: dict, record) -> dict:
    """`/openclaw schedule …` — the deterministic path to a job that exists.

    Asking OpenClaw's agent in prose to "create a daily task" is a reasonable
    thing to want and a bad way to get it: measured, it produced a clarifying
    question rather than a job, and the answer to that question went to a
    different agent because the follow-up line carried no prefix. Prose is the
    right shape for asking and the wrong shape for something that must either
    exist afterwards or say why not.

    So this branch never asks anything. It parses, and either creates, lists,
    removes, or refuses with the syntax spelled out.

    ### Neden burada da bir kayıt açılıyor

    Zamanlama, akış ekranında görünmeyen tek yoldu: sohbet, takım, tarama ve MAF
    kendi turlarını kaydederken `/openclaw schedule` hiçbir iz bırakmıyordu.
    Ekranın "bu sistemde ne olduğunu gösteriyorum" iddiası varsa, kaydetmediği
    bir yol o iddianın deliği olur. Üç aşama az ama doğru üç aşama: cümlenin
    okunması, kapı, ve devir.
    """
    import scheduler as scheduler_module

    run = runlog.LOG.begin("cron", plan["text"][:160], record.id)
    bus = stages.StageBus()

    def _flush() -> None:
        runlog.LOG.record(run, bus.drain())

    try:
        command = scheduler_module.parse_command(plan["text"])
        bus.emit("cron_parse", text=plan["text"][:120],
                 action=command["action"])
        _flush()
    except scheduler_module.WhenError as exc:
        # Ayrıştırma hatası da kaydediliyor: reddedilen bir zamanlama, hiç
        # denenmemiş bir zamanlamadan farklı bir şey.
        bus.emit("cron_parse", text=plan["text"][:120], error=str(exc)[:120])
        _flush()
        run.end("error")
        return {"ok": False, "method": "schedule", "tier": "local", "error": str(exc)}

    if command["action"] == "list":
        listing = await scheduler_module.jobs()
        run.end()
        return {"ok": True, "method": "schedule", "tier": "local", "result": listing}

    if command["action"] == "remove":
        held = _openclaw_gate(
            {"mode": "method", "method": "cron.remove", "tier": "write",
             "params": {"id": command["id"]}, "text": "", "error": ""},
            record.id,
        )
        bus.emit("cron_gate", method="cron.remove", held=held is not None)
        _flush()
        if held is not None:
            run.end("blocked")
            return held
        out = await scheduler_module.remove(command["id"])
        bus.emit("cron_done", action="remove", id=command["id"])
        _flush()
        run.end()
        return out

    try:
        params = scheduler_module.build_job(
            command["ask"][:60], command["when"], command["ask"], to=command["to"]
        )
    except scheduler_module.WhenError as exc:
        bus.emit("cron_parse", text=plan["text"][:120], error=str(exc)[:120])
        _flush()
        run.end("error")
        return {"ok": False, "method": "schedule", "tier": "local", "error": str(exc)}

    # Signed on what was typed, not on the resolved schedule: `"20dk sonra"` is a
    # different timestamp every time it is parsed, and a digest over the result
    # could never match the grant it had just been given.
    held = _openclaw_gate(
        {"mode": "method", "method": "cron.add", "tier": "write",
         "params": {k: command[k] for k in ("when", "ask", "to")},
         "text": "", "error": ""},
        record.id,
    )
    bus.emit("cron_gate", method="cron.add", held=held is not None,
             when=scheduler_module.describe(params["schedule"])[:80])
    _flush()
    if held is not None:
        run.end("blocked")
        return held

    outcome = await openclaw_control.call("cron.add", params)
    outcome["when"] = scheduler_module.describe(params["schedule"])
    bus.emit("cron_done", action="add", ok=bool(outcome.get("ok")),
             when=outcome["when"][:80], to=command["to"] or "—")
    _flush()
    run.end("done" if outcome.get("ok") else "error")
    if not command["to"]:
        outcome["note"] = (
            "Teslimat hedefi verilmedi: iş koşacak ama sonucu yalnız task kaydına "
            "düşecek. Bir yere gitmesi için: … | … > telegram:<sohbet-id>"
        )
    return outcome


async def _openclaw_foto(plan: dict, record) -> dict:
    """`/openclaw foto` — OpenClaw çeker, dosyayı biz adlandırırız.

    Cümle yolu (“fotoğrafımı çek”) bugün de çalışıyor, ama dosyanın nereye
    düştüğüne model karar veriyor. Burada kimliği ve yolu biz üretip komutu tam
    metin olarak veriyoruz; ajana kalan tek iş onu çalıştırmak. Onay kartında
    görünen metin, çalışacak komutun kendisi — `docs/16 §2.2`'deki “onaylanan şey
    donmuş bir plandır” ilkesinin küçük hâli.

    Gateway ulaşılamazsa kare yine de alınıyor: `shots.local_capture` aynı yola
    aynı adla yazıyor, ve sunma tarafı ikisini ayırt etmiyor.
    """
    import shots as shots_module

    body = (plan["text"] or "").strip().lower()
    if body in ("sil", "temizle", "clear"):
        return {"ok": True, "method": "foto", "tier": "local",
                "result": {"cleared": shots_module.clear()}}
    if body in ("liste", "list", "ls"):
        return {"ok": True, "method": "foto", "tier": "local",
                "result": {"shots": shots_module.recent()}}

    device, size = shots_module.DEFAULT_DEVICE, shots_module.DEFAULT_SIZE

    # Signed on the *stable* part of the request, not on the generated sentence.
    # Every call mints a fresh frame id, so a digest over the sentence would be
    # different every time and the grant could never be consumed — the same bug
    # `schedule` had with `"5dk sonra"`, found the same way. The operator still
    # sees the exact command, because it goes in the reason.
    held = _openclaw_gate(
        {"mode": "method", "method": "camera.capture", "tier": "write",
         "params": {"device": device, "size": size}, "text": "", "error": ""},
        record.id,
    )
    if held is not None:
        held["reason"] = (
            f"Kamerandan tek kare alınacak ({device}, {size}) ve panele basılacak. "
            "Komutu OpenClaw'ın ajanı çalıştıracak; o ajanın kabuğu var ve onay "
            "sormuyor (exec: mode=full, ask=off). " + held["reason"]
        )
        return held

    shot_id = shots_module.new_id()
    sentence = shots_module.sentence(shot_id, device=device, size=size)
    outcome = await openclaw_control.ask(sentence, peer="local")
    if not shots_module.exists(shot_id):
        # OpenClaw cevap vermiş olabilir ama dosya yoksa çekim olmamıştır. Kota,
        # kapalı gateway, izin — hangisi olursa olsun cevabı beklemek yerine
        # kareyi burada alıyoruz, ve bunu gizlemiyoruz.
        local = shots_module.local_capture(shot_id, device=device, size=size)
        if not local["ok"]:
            return {"ok": False, "method": "foto", "tier": "write",
                    "error": local["error"],
                    "note": "OpenClaw kare üretmedi ve yerel çekim de başarısız."}
        return {"ok": True, "method": "foto", "tier": "write",
                "result": {"id": shot_id, "url": f"/api/shot/{shot_id}", "by": "local"},
                "note": "OpenClaw kare üretmedi; kare yerel ffmpeg ile alındı."}

    shots_module.prune()
    return {"ok": True, "method": "foto", "tier": "write",
            "result": {"id": shot_id, "url": f"/api/shot/{shot_id}", "by": "openclaw",
                       "said": outcome.get("result") if isinstance(outcome, dict) else ""}}


@app.get("/api/shot/{shot_id}")
def shot(shot_id: str) -> Response:
    """Bir kareyi **kimlikle** ver. İstekten gelen metin hiç yol olmuyor.

    `shots.path_for` önce biçimi doğruluyor, sonra birleştiriyor, sonra çözülmüş
    yolun dizinin içinde kaldığını kontrol ediyor. Bir web sunucusunda dosya
    sunan kodun tek ciddi arızası yol kaçışı, ve o kontrol tek yerde duruyor.
    """
    import shots as shots_module

    try:
        data = shots_module.read(shot_id)
    except shots_module.ShotError as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/shots")
def shots_list() -> JSONResponse:
    import shots as shots_module

    return JSONResponse({"shots": shots_module.recent(), "keep": shots_module.KEEP})


@app.post("/api/openclaw")
async def openclaw_line(payload: OpenClawLine) -> JSONResponse:
    """One typed line to OpenClaw — a Gateway method, or a question for its agent.

    The escape hatch, deliberately thin: no model of ours, no tools, no summary.
    What OpenClaw returns is what the screen shows, because the reason to type
    this instead of asking our agent is that you want to see the actual bytes.

    Which of the two it is comes from the first word: a dot makes it a method
    (`sessions.list`), anything else is a sentence and goes to OpenClaw's own
    agent (`adın ne`). The route records which way it went, so a transcript never
    leaves you guessing whether a line reached the control plane or the model.

    It still belongs to a session. The line goes into the transcript *before* the
    call runs and the answer after it, so a call that hangs or dies still leaves
    evidence that it was made — the same rule the chat path follows.

    **What the gate covers.** Read-class methods run straight through: the point
    of this hatch is seeing the actual bytes, and asking for approval before every
    `sessions.list` would retire it. A write-class method, and any sentence, is
    held for approval first. The sentence matters most and is the least obvious:
    it goes to OpenClaw's *own* agent, which on this host runs shell without
    prompting (`tools.exec` measured at `mode=full, ask=off`), and our gate cannot
    see what that agent does once the line arrives. So the approval text says so.
    """
    record, _ = _session(payload.peer)

    line = payload.line.strip()
    plan = openclaw_control.plan_line(payload.line)
    to_agent = plan["mode"] == "sentence"

    CHAT.sessions.record_turn(
        record, "user", line, channel="openclaw-direct",
        method="chat.send" if to_agent else plan["method"],
        tier="chat" if to_agent else plan["tier"],
    )

    # `schedule` is answered here rather than by `run_line`, because it is the one
    # subcommand that is not a Gateway call: it is our own translation plus a
    # `cron.add`. Listing and refusals never reach the Gateway at all.
    if plan["mode"] in ("schedule", "foto"):
        outcome = await (
            _openclaw_schedule(plan, record) if plan["mode"] == "schedule"
            else _openclaw_foto(plan, record)
        )
        if outcome.get("held"):
            CHAT.sessions.record_turn(
                record, "assistant", outcome["reason"], channel="openclaw-direct", ok=False,
            )
            return JSONResponse({**outcome, "session": record.id}, status_code=202)
        CHAT.sessions.record_turn(
            record, "assistant", json.dumps(outcome, ensure_ascii=False)[:4000],
            channel="openclaw-direct", ok=bool(outcome.get("ok")),
        )
        return JSONResponse({**outcome, "session": record.id})

    held = _openclaw_gate(plan, record.id)
    if held is not None:
        CHAT.sessions.record_turn(
            record, "assistant", held["reason"], channel="openclaw-direct", ok=False,
        )
        return JSONResponse({**held, "session": record.id}, status_code=202)

    outcome = await openclaw_control.run_line(payload.line, peer=payload.peer)
    answer = outcome.get("result", outcome.get("error", ""))
    CHAT.sessions.record_turn(
        record, "assistant",
        (answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False))[:4000],
        channel="openclaw-direct", ok=bool(outcome.get("ok")),
    )
    if not outcome.get("ok") and not to_agent and "Usage" in str(outcome.get("error", "")):
        outcome.setdefault("usage", f"{openclaw_control.PREFIX} <method> [json]  ·  or just ask")
    return JSONResponse({**outcome, "session": record.id})


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
async def approval_approve(request_id: str, payload: ApprovalDecision) -> JSONResponse:
    """Onayla — ve onaylanan şey kodsa, **onu** koştur.

    Kapının reddi turu bitiriyor: ajan reddedildiğini söyleyip devam ediyor. Onay
    o turu geri getiremiyor, ve modelden kodu yeniden yazmasını beklemek işe
    yaramıyor — ölçüldü, aynı soru iki farklı program üretti, dolayısıyla iki
    farklı imza. Onaylananla çalışanın aynı olmasının tek yolu, çalıştırılacak
    olanın `Request.payload`'da **saklanan metin** olması.
    """
    import codeexec as codeexec_module
    from gateway import approval as approval_module

    request = approval_module.GATE.get(request_id)
    outcome = approval_module.GATE.approve(request_id, note=payload.note)
    if not outcome["ok"]:
        raise HTTPException(404, outcome["reason"])

    if request is not None and request.tool == codeexec_module.TOOL_NAME:
        code = str((request.payload or {}).get("code", ""))
        run = await codeexec_module.run_approved(code)
        # Grant tüketiliyor: onay bu koşuyu kapsıyor, sonraki her benzerini değil.
        approval_module.GATE._granted.discard(request.digest)  # noqa: SLF001
        outcome["ran"] = {"code": code, **run}
    return JSONResponse(outcome)


@app.post("/api/approvals/{request_id}/deny")
def approval_deny(request_id: str, payload: ApprovalDecision) -> JSONResponse:
    from gateway import approval as approval_module

    outcome = approval_module.GATE.deny(request_id, note=payload.note)
    if not outcome["ok"]:
        raise HTTPException(404, outcome["reason"])
    return JSONResponse(outcome)


class ScheduleJob(BaseModel):
    name: str
    when: str
    ask: str
    session: str = "isolated"


@app.get("/api/schedule")
async def schedule_list() -> JSONResponse:
    """Scheduled jobs, as OpenClaw holds them.

    We do not keep a second copy. A local list would drift the moment somebody
    used `openclaw automations` directly, and a scheduler you cannot trust the
    listing of is worse than none.
    """
    import scheduler as scheduler_module

    return JSONResponse(await scheduler_module.jobs())


@app.post("/api/schedule")
async def schedule_create(payload: ScheduleJob) -> JSONResponse:
    """Create a job. Goes through the same gate as any other write."""
    import scheduler as scheduler_module

    record, _ = _session("local")
    try:
        params = scheduler_module.build_job(
            payload.name, payload.when, payload.ask, session=payload.session
        )
    except scheduler_module.WhenError as exc:
        raise HTTPException(400, str(exc)) from exc

    # The gate signs what the *person* asked for, not the resolved schedule.
    # `"5dk sonra"` becomes a different timestamp every time it is parsed, so a
    # digest over `params` never matched the grant it had just been given and the
    # approval could not be consumed — approve, retry, get asked again, forever.
    # The three fields below are exactly what the approval card shows, which is
    # the other half of the argument: you should be signing what you read.
    held = _openclaw_gate(
        {"mode": "method", "method": "cron.add", "tier": "write",
         "params": {"name": payload.name, "when": payload.when, "ask": payload.ask},
         "text": "", "error": ""},
        record.id,
    )
    if held is not None:
        return JSONResponse({**held, "session": record.id}, status_code=202)

    outcome = await openclaw_control.call("cron.add", params)
    return JSONResponse({**outcome, "when": scheduler_module.describe(params["schedule"])})


@app.delete("/api/schedule/{job_id}")
async def schedule_delete(job_id: str) -> JSONResponse:
    import scheduler as scheduler_module

    record, _ = _session("local")
    held = _openclaw_gate(
        {"mode": "method", "method": "cron.remove", "params": {"id": job_id},
         "tier": "write", "text": "", "error": ""},
        record.id,
    )
    if held is not None:
        return JSONResponse({**held, "session": record.id}, status_code=202)
    return JSONResponse(await scheduler_module.remove(job_id))


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
