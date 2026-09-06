"""Faz 4: Web arayüzü — FastAPI, tek WebSocket ile çift yönlü canlı akış.

contracts/websocket_protocol.md'deki mesaj şemasını uygular (specs/
003-web-ui-live-trace/). Statik dosyalar (index.html/app.js/style.css) build
aracı olmadan doğrudan sunulur (Principle V) — frontend'de framework yok.

.env okuma (load_dotenv), agent/graph.py'nin CLI'daki kullanımıyla aynı ilkeyle,
uygulamanın giriş noktasında (bu modülün import edilmesi sırasında) yapılır.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from grounded_assistant.agent import graph

# T009: CLI ile aynı mantık, tek kaynak (Principle V) — ayrı bir "web DTO" icat edilmiyor.
from grounded_assistant.cli import _build_answer
from grounded_assistant.ptc.sandbox_runner import run_sandbox
from grounded_assistant.session import oturum_kimligi
from grounded_assistant.web import durum as durum_modulu
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


@app.get("/durum")
async def durum_sayfasi() -> FileResponse:
    """Gözlemlenebilirlik paneli — pod'lar, izinli akışlar, artifact'ler.

    Ayrı bir sayfa çünkü hedef kitlesi farklı: sohbet ekranı son kullanıcı
    için, burası sistemi kuran/denetleyen kişi için.
    """
    return FileResponse(_STATIC_DIR / "durum.html")


@app.get("/api/durum")
async def durum_verisi(session: str | None = None) -> dict:
    """Panelin tek veri kaynağı. Üç toplayıcı BAĞIMSIZ çalışıyor: biri
    başarısız olursa (ör. cluster kapalı) diğerleri yine dönüyor ve panelin o
    bölümü hata mesajı gösteriyor — tamamen kararmasındansa."""
    from grounded_assistant.agent.graph import _kapsam_jetonu  # noqa: PLC0415

    return {
        "podlar": await asyncio.to_thread(durum_modulu.podlar),
        "akislar": await asyncio.to_thread(durum_modulu.akislar),
        "artifactler": await asyncio.to_thread(
            durum_modulu.artifactler, session, _kapsam_jetonu
        ),
    }


@app.get("/api/artifact/{artifact_id}")
async def artifact_onizleme(artifact_id: str, session: str | None = None) -> dict:
    """Bir artifact'in İÇERİĞİ — panelde önizleme için.

    Künye zaten `/api/durum`'da geliyordu; bu uç nokta "deponun içini görelim"
    isteğini karşılıyor. Baytlar sunucuda çözülüp sadeleştirilmiş bir
    tablo/metne çevriliyor: Parquet'i tarayıcıda açmak pyarrow'un tarayıcı
    sürümünü gerektirirdi ve gereksiz veri taşırdı.

    Kapsam yine JETONDAN: panel başka bir workflow'un artifact'ini isterse
    servis 404 döner, buradan da "bulunamadı" olarak geçer.
    """
    from grounded_assistant.agent.graph import _kapsam_jetonu  # noqa: PLC0415

    if not session:
        return {"hata": "Oturum yok"}
    return await asyncio.to_thread(
        durum_modulu.artifact_icerigi, session, artifact_id, _kapsam_jetonu
    )


@app.get("/api/artifact/{artifact_id}/soy")
async def artifact_soy(artifact_id: str, session: str | None = None) -> dict:
    """Bu artifact'in ATALARI ve ÜRÜNLERİ — panelin soy grafiği.

    `parents` kayıt defterinde baştan beri vardı ama hiçbir yerde
    OKUNMUYORDU. Sandbox artık soyu otomatik dolduruyor (okunan girdiler →
    üretilen çıktının ebeveyni); bu uç nokta onu görünür kılıyor.
    """
    from grounded_assistant.agent.graph import _kapsam_jetonu  # noqa: PLC0415

    if not session:
        return {"hata": "Oturum yok"}
    return await asyncio.to_thread(
        durum_modulu.soy_agaci, session, artifact_id, _kapsam_jetonu
    )


@app.get("/api/akis")
async def canli_akis() -> StreamingResponse:
    """Hubble'dan canlı ağ akışı — Server-Sent Events.

    ## Neden SSE, WebSocket değil

    Akış TEK YÖNLÜ: sunucu gönderiyor, tarayıcı yalnızca dinliyor. SSE bunun
    için yeterli ve tarayıcı tarafında yeniden bağlanmayı KENDİSİ yapıyor —
    `EventSource` bağlantı koptuğunda otomatik deniyor. WebSocket'te bunu elle
    yazmak gerekirdi.

    ## Neden alt süreç

    `hubble observe --follow` uzun ömürlü bir akış; Python'da bir Hubble gRPC
    istemcisi yazmak yerine resmî CLI'ı kullanıyoruz. Çıktı satır satır JSON
    (`-o json`), doğrudan ayrıştırılabiliyor.

    İstemci sekmeyi kapatınca `asyncio.CancelledError` geliyor ve alt süreç
    öldürülüyor — yoksa her sekme bir `hubble` süreci bırakırdı.
    """
    async def uret():
        try:
            surec = await asyncio.create_subprocess_exec(
                "hubble", "observe", "--server", durum_modulu.HUBBLE_SUNUCU,
                "--namespace", os.environ.get("PTC_NAMESPACE", "default"),
                # `--last`: bağlanır bağlanmaz son akışları da gönder.
                #
                # NEDEN: sistem sessizken panel bomboş kalıyordu ve kullanıcı
                # "bozuk mu, sakin mi?" diye ayırt edemiyordu. Hubble UI'ın
                # kendisi de böyle çalışıyor — önce yakın geçmiş, sonra canlı.
                "--last", "25",
                "--follow", "-o", "json",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError:
            yield _sse({"hata": "hubble CLI bulunamadı"})
            return

        try:
            while True:
                satir = await surec.stdout.readline()
                if not satir:
                    break
                try:
                    ham = json.loads(satir)
                except ValueError:
                    continue
                sade = durum_modulu._akisi_sadelestir(ham.get("flow") or {})
                if sade and not durum_modulu.akis_tekrari_mi(sade):
                    yield _sse(sade)
        except asyncio.CancelledError:
            raise
        finally:
            # Sekme kapandığında alt süreci de kapat.
            if surec.returncode is None:
                surec.kill()
                await surec.wait()

    return StreamingResponse(
        uret(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(veri: dict) -> str:
    return f"data: {json.dumps(veri)}\n\n"


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
    session_id = oturum_kimligi(websocket.query_params.get("session"))
    trace = Trace()
    event_queue: queue.Queue = queue.Queue()
    # `asyncio.to_thread`: build_agent -> live_systems.get_live_system_tools()
    # kendi içinde `asyncio.run()` çağırıyor (Faz 1) — bu, zaten çalışan bir
    # event loop'un (bu async handler'ın) İÇİNDEN çağrılamaz ("asyncio.run()
    # cannot be called from a running event loop"). Ayrı bir thread'de
    # çalıştırmak, o thread'in kendi (boş) loop durumuna sahip olmasını sağlar.
    # Checkpointer BURADA kuruluyor, `build_agent`'ın içinde değil: async
    # saver'lar `asyncio.get_running_loop()` istiyor ve `build_agent`
    # loop'suz bir thread'de çalışmak ZORUNDA (aşağıdaki nota bakın).
    checkpointer = graph.build_checkpointer()
    agent = await asyncio.to_thread(
        graph.build_agent, trace, event_queue.put, session_id, checkpointer
    )

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
    finally:
        # aiosqlite bağlantısını KAPAT (2026-09-04). Her WebSocket bağlantısı
        # kendi checkpointer'ını kuruyor; aiosqlite her bağlantı için ayrı bir
        # thread açıyor. Kapatmazsak her sekme açılıp kapandığında bir thread
        # sızıyor ve uzun ömürlü sunucuda birikiyordu.
        kapat = getattr(checkpointer, "conn", None)
        if kapat is not None:
            try:
                await kapat.close()
            except Exception:  # noqa: BLE001 — kapanış hatası isteği etkilemesin
                pass
