"""Faz 4: Web arayüzü — FastAPI, tek WebSocket ile çift yönlü canlı akış.

contracts/websocket_protocol.md'deki mesaj şemasını uygular (specs/
003-web-ui-live-trace/). Statik dosyalar (index.html/app.js/style.css) build
aracı olmadan doğrudan sunulur (Principle V) — frontend'de framework yok.

.env okuma (load_dotenv), agent/graph.py'nin CLI'daki kullanımıyla aynı ilkeyle,
uygulamanın giriş noktasında (bu modülün import edilmesi sırasında) yapılır.
"""

from __future__ import annotations

import asyncio
import queue
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from grounded_assistant.agent import graph

# T009: CLI ile aynı mantık, tek kaynak (Principle V) — ayrı bir "web DTO" icat edilmiyor.
from grounded_assistant.cli import _build_answer
from grounded_assistant.ptc.sandbox_runner import run_sandbox
from grounded_assistant.trace import Trace

load_dotenv()

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_QUEUE_SENTINEL = object()

# Altan'ın kararı (2026-08-30): sunumda "engelleme" senaryosunu göstermek için
# LLM'in o an ne yazacağına güvenmiyoruz — model, kendi güvenlik hizalaması
# nedeniyle "ağa bağlanmaya çalışan kod yaz" isteklerini tutarsız biçimde
# reddedebiliyor (bulundu, 2026-08-30: aynı "evil.com" isteği bir turda 16
# denied_action üretirken, başka bir turda LLM doğrudan reddetti). Bu yüzden
# demo_escape mesajı, agent'ı/LLM'i HİÇ devreye sokmadan doğrudan run_sandbox'ı
# sabit bir kodla çalıştırır — sahnede her zaman aynı, garanti sonucu üretir.
_DEMO_ESCAPE_CODE = """import socket
try:
    socket.create_connection(("evil.com", 443), timeout=5)
    set_result("BAGLANTI KURULDU (beklenmeyen!)")
except Exception as e:
    set_result(f"Engellendi: {e}")
"""

app = FastAPI()
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


async def _drain_ptc_events(websocket: WebSocket, event_queue: queue.Queue) -> None:
    """T015: `run_sandbox`'ın (bir arka-plan thread'inde çalışan) `on_event`
    callback'inin thread-safe yazdığı olayları kuyruktan alıp WebSocket'e
    `ptc_event` olarak gönderir (research.md §5). `_QUEUE_SENTINEL` görülünce
    döner — bu, ana coroutine'in `agent.invoke` bitince kuyruğa koyduğu, tüm
    gerçek olayların işlendiğinden emin olmamızı sağlayan işaret."""
    loop = asyncio.get_running_loop()
    while True:
        event = await loop.run_in_executor(None, event_queue.get)
        if event is _QUEUE_SENTINEL:
            return
        await websocket.send_json({"type": "ptc_event", **event})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """T008: bağlantı başına bir oturum (session_id + Trace + agent) kurulur,
    bağlantı ömrü boyunca yaşar. T009: `question` mesajlarını işler — agent
    çağrısı bloklayıcı olduğu için bir arka-plan thread'inde (`asyncio.to_thread`,
    research.md §5) çalıştırılır; bu sürede PTC olayları `_drain_ptc_events` ile
    eş zamanlı akar (contracts/websocket_protocol.md)."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    trace = Trace()
    event_queue: queue.Queue = queue.Queue()
    # `asyncio.to_thread`: build_agent -> live_systems.get_live_system_tools()
    # kendi içinde `asyncio.run()` çağırıyor (Faz 1) — bu, zaten çalışan bir
    # event loop'un (bu async handler'ın) İÇİNDEN çağrılamaz ("asyncio.run()
    # cannot be called from a running event loop"). Ayrı bir thread'de
    # çalıştırmak, o thread'in kendi (boş) loop durumuna sahip olmasını sağlar.
    agent = await asyncio.to_thread(graph.build_agent, trace, event_queue.put)

    try:
        while True:
            message = await websocket.receive_json()

            if message.get("type") == "demo_escape":
                # Agent/LLM'i atlayıp doğrudan sandbox'ı çalıştırır (yukarıdaki
                # not) — sonucu answer/chat balonu değil, ayrı bir demo_result
                # mesajıyla bildirir (bu bir soru-cevap turu değil).
                drain_task = asyncio.create_task(_drain_ptc_events(websocket, event_queue))
                run = await asyncio.to_thread(
                    run_sandbox, _DEMO_ESCAPE_CODE, on_event=event_queue.put
                )
                event_queue.put(_QUEUE_SENTINEL)
                await drain_task
                await websocket.send_json(
                    {
                        "type": "demo_result",
                        "status": run.status.value,
                        "denied_count": len(run.denied_actions),
                    }
                )
                continue

            if message.get("type") != "question":
                continue

            turn_mark = trace.mark()
            drain_task = asyncio.create_task(_drain_ptc_events(websocket, event_queue))
            try:
                # invoke_and_resolve artık native async (graph.py'de düzeltilen
                # gerçek hata) — burada ayrıca asyncio.to_thread'e SARMAYA gerek
                # yok; LangChain'in kendisi senkron tool'ları (run_ptc_code)
                # zaten otomatik bir thread'de çalıştırıyor (bkz. graph.py notu).
                result = await graph.invoke_and_resolve(agent, message["text"], session_id)
            except Exception as exc:  # noqa: BLE001 - beklenmeyen hata, bağlantıyı çökertmemeli
                event_queue.put(_QUEUE_SENTINEL)
                await drain_task
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue
            event_queue.put(_QUEUE_SENTINEL)
            await drain_task

            raw_text = result["messages"][-1].content
            # trace.since(turn_mark): agent/checkpointer (konuşma hafızası için)
            # tüm oturum boyunca AYNI kalıyor, ama grounding/kaynak hesaplaması
            # yalnızca BU turda eklenen kayıtları görmeli (bkz. trace.py'deki not).
            answer = _build_answer(trace.since(turn_mark), raw_text)
            await websocket.send_json(
                {
                    "type": "answer",
                    "text": answer.text,
                    "grounded": answer.grounded,
                    "source_refs": answer.source_refs,
                    "partial_failure_notes": answer.partial_failure_notes,
                    "access_paths_used": [p.value for p in answer.access_paths_used],
                }
            )
    except WebSocketDisconnect:
        pass
