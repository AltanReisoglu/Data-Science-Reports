"""What is running right now, named honestly.

The chat interface streams an answer but hides the machine that produced it. This
module is the other half: a catalogue of the mechanisms a turn actually passes
through, and a bus the stream drains so the interface can light them up as they
happen.

### The catalogue is the single source of the teaching text

Every label the panel shows — the mechanism's name, the real AutoGen class behind
it, the line in the user guide, the one-sentence explanation — lives here and is
served over `/api/mechanisms`. The browser holds none of it. A panel that teaches
from strings hardcoded in JavaScript drifts away from the code it describes within
a release or two, and nothing fails when it does.

### Three lanes, and why the third one exists

``agentchat`` and ``core`` are AutoGen's own layers. ``ours`` is the third, and
keeping it separate is the point: the approval gate and the compacting context are
**not** AutoGen features. Drawing them in the same colour as `ToolCallRequestEvent`
would teach somebody that AutoGen ships an approval gate, and it does not — the
cookbook shows you how to build one (`05:6638`).

### What this module refuses to do

It reports what happened; it never asserts what *should* have. The core lane is
drawn from the runtime's own counters, so when the panel says core did not run,
that is a measurement (`routed == 0`) and not a sentence somebody typed. A panel
that lights a box because the author expected it to light is worse than no panel.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

# Lanes. The order is the order they are drawn in.
AGENTCHAT = "agentchat"
CORE = "core"
OURS = "ours"


@dataclass(frozen=True)
class Mechanism:
    """One step of a turn, and everything the interface needs to explain it."""

    id: str
    lane: str
    title: str
    klass: str          # the real class or event, never a paraphrase
    ref: str            # `08:2298` — the line in the guide, so it can be checked
    note: str           # one sentence, the thing worth knowing
    module: str = ""    # the file it lives in, so "what ran" is answerable

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# The ten steps a chat turn passes through, in order, read off `conversation.py`
# rather than off the documentation — several of these differ from the defaults
# the guide describes, and the difference is the interesting part.
CATALOGUE: dict[str, Mechanism] = {
    m.id: m for m in [
        Mechanism(
            "context", OURS, "Bağlam kuruluyor",
            "CompactingChatCompletionContext.get_messages()", "05:2341",
            "Modele gidecek mesaj kümesi bütçeye göre seçiliyor. AutoGen'in "
            "BufferedChatCompletionContext'i mesaj sayar; bu bizim token sayan hâlimiz.",
            "pipeline/context_engine.py",
        ),
        Mechanism(
            "compaction", OURS, "Bağlam sıkıştırılıyor",
            "CompactionEvent · context_engine.py", "05:2341",
            "Bütçe dolduğu için eski turlar özetleniyor. Sınır asla bir tool "
            "çağrısını sonucundan ayırmayacak yerden geçiyor.",
            "pipeline/context_engine.py",
        ),
        Mechanism(
            "model", CORE, "Model çağrısı",
            "model_client.create_stream()", "05:1980",
            "create() değil create_stream(): bu yüzden LLMCallEvent değil "
            "LLMStreamEndEvent yayılıyor — maliyeti yalnız ilkini dinleyerek sayan 0 görür.",
            "pipeline/engine.py",
        ),
        Mechanism(
            "stream", AGENTCHAT, "Token akışı",
            "ModelClientStreamingChunkEvent", "08:2236",
            "Cevap token token geliyor. model_client_stream=True olmasaydı cevap "
            "tek parça hâlinde, model bitirdikten sonra gelirdi.",
            "pipeline/conversation.py",
        ),
        Mechanism(
            "tool_request", AGENTCHAT, "Model bir tool istedi",
            "ToolCallRequestEvent", "08:2298",
            "Şema imzadan ve docstring'den üretildi; yani docstring dokümantasyon "
            "değil, arayüz.",
            "pipeline/conversation.py",
        ),
        Mechanism(
            "gate", OURS, "Kapı",
            "before_tool_call → GatedWorkbench", "05:6638",
            "AutoGen'de böyle bir şey yok; cookbook nasıl yazılacağını gösteriyor. "
            "Kapı ajanın uyum göstermeyi seçmesine değil, hattın kendisine dayanıyor.",
            "pipeline/gateway/workbench.py",
        ),
        Mechanism(
            "tool_exec", CORE, "Tool koşuyor",
            "workbench.call_tool()", "05:2841",
            "Workbench bir tool kaynağı: yerel fonksiyonlar StaticWorkbench'te, "
            "OpenClaw ve DeepWiki McpWorkbench'te — ajan için hepsi aynı arayüz.",
            "pipeline/gateway/tools.py",
        ),
        Mechanism(
            "tool_result", AGENTCHAT, "Sonuç döndü",
            "ToolCallExecutionEvent", "08:2298",
            "Sonuç bağlama giriyor ve döngü modele geri dönüyor.",
            "pipeline/conversation.py",
        ),
        Mechanism(
            "loop", AGENTCHAT, "Döngü devam ediyor",
            "max_tool_iterations=6", "08:2298",
            "Varsayılan 1'dir: ajan bir tool çağırır, sonucu görür ve susar. "
            "Zincirleme davranış için bu değer elle yükseltildi.",
            "pipeline/conversation.py",
        ),
        Mechanism(
            "done", AGENTCHAT, "Tur bitti",
            "TaskResult.stop_reason", "08:2813",
            "TaskResult iki şey taşıyor: bütün konuşma ve neden durduğu.",
            "pipeline/conversation.py",
        ),

        # ---- the scan, which is GraphFlow and not core -------------------------
        #
        # Worth being exact about, because it is easy to get backwards: `scan.py`
        # calls `graph.enrich`, which is **GraphFlow** — AgentChat. `fanin.py` is
        # the core pub/sub alternative and only `compare_fanin.py` runs it. The
        # panel draws what the scan does, not what the more interesting module
        # would have done.
        Mechanism(
            "graph_build", AGENTCHAT, "Graf kuruldu",
            "DiGraphBuilder → GraphFlow(participants, graph)", "08:5398",
            "Akış önceden çizildi: üç analist paralel, sonra risk denetçisi, sonra "
            "skorlayıcı. MaxMessageTermination sigortası ve custom_message_types "
            "beyanı da burada — takım tanımadığı mesaj tipini yönlendirmez.",
            "pipeline/graph.py",
        ),
        Mechanism(
            "graph_run", AGENTCHAT, "Akış koşuyor",
            "flow.run_stream(task=...)", "08:5398",
            "run() değil run_stream(): bir dal fırlatırsa run() gelmiş olan her şeyi "
            "de atıyor. Akıştan okumak, ulaşanı elde tutuyor.",
            "pipeline/graph.py",
        ),
        Mechanism(
            "analysts", AGENTCHAT, "Dallar paralel",
            "AssistantAgent × 3 · teknik · pazar · ekip", "08:2298",
            "Çok ajanlı olmalarının sebebi zekâ değil ayrıştırma: üç ayrı kaynağa "
            "bakıyorlar ve aynı anda koşabiliyorlar.",
            "pipeline/agents/analysts.py",
        ),
        Mechanism(
            "count", OURS, "Beklenen dal sayıldı",
            "Score.missing_data", "08:5398",
            "Bariyere sorulmuyor, beklenen dal SAYILIYOR. Gelmeyen her dal "
            "missing_data'ya yazılıyor: sessiz eksik sonuç, beyan edilmiş bir "
            "bilgi yokluğuna çevriliyor.",
            "pipeline/graph.py",
        ),

        # ---- fanin.py: the core engine, measured but not on the scan path ------
        Mechanism(
            "runtime_start", CORE, "Runtime açıldı",
            "SingleThreadedAgentRuntime.start()", "05:490",
            "Mesaj işleme döngüsü başladı. Bu runtime kısa ömürlü — şirket başına "
            "kuruluyor ve kapatılıyor; gateway'inki ise sürekli koşuyor.",
            "pipeline/fanin.py",
        ),
        Mechanism(
            "subscribe", CORE, "Abonelikler kuruldu",
            "TypeSubscription(topic_type, agent_type)", "05:670",
            "Üç dal da AYNI topic tipine abone. Paralellik buradan geliyor: "
            "yayınlayan taraf kaç abone olduğunu bilmiyor.",
            "pipeline/fanin.py",
        ),
        Mechanism(
            "publish", CORE, "Görev yayınlandı",
            "runtime.publish_message(BranchTask, TopicId(...))", "05:1108",
            "TEK yayın, üç ajan koşuyor. Tool değil, model kararı değil — düz bir "
            "Python satırı ve bir abonelik tablosu araması.",
            "pipeline/fanin.py",
        ),
        Mechanism(
            "branch", CORE, "Dal çalışıyor",
            "BranchWorker(RoutedAgent).handle", "05:894",
            "Handler'ı tip anotasyonu seçti. Bu ajan asla fırlatmıyor: hata da bir "
            "sonuç olarak yayınlanıyor, yoksa kardeşlerinin işini götürürdü.",
            "pipeline/fanin.py",
        ),
        Mechanism(
            "collect", CORE, "Sonuç kuyruğa düştü",
            "ClosureAgent → asyncio.Queue", "05:6825",
            "Sonuç üretildiği anda yayınlandı ve kuyruk onu çoktan tuttu. "
            "Güvenilmeyecek bariyer yok, çünkü bariyer yok.",
            "pipeline/fanin.py",
        ),
        Mechanism(
            "intervention", CORE, "Müdahale kapısı",
            "AuditingInterventionHandler.on_publish", "05:6534",
            "Runtime'a takılan tek kapı: her mesaj buradan geçiyor ve denetim "
            "kaydına yazılıyor. Takmanın bedeli, runtime'ı kendin kurmak.",
            "pipeline/observability.py",
        ),
        Mechanism(
            "join", AGENTCHAT, "Birleşme",
            'GraphFlow · activation_condition="all"', "08:5398",
            "Beklenen dal SAYISI sayılıyor, runtime'ın 'boşta' demesi beklenmiyor — "
            "ölçtük, o bariyer bir dal çökünce tamamlanmış kardeşleri de kaybettiriyor.",
            "pipeline/graph.py",
        ),
        Mechanism(
            "runtime_stop", CORE, "Runtime kapandı",
            "stop_when_idle() → close()", "05:490",
            "Toplama bittikten SONRA, süre sınırıyla. Bariyer burada kapatma aracı; "
            "sonuç toplamada kullanılmıyor.",
            "pipeline/graph.py",
        ),
    ]
}

# What kind of machine is running, stated per run rather than per mechanism.
#
# The panel used to draw a fixed pipeline with boxes lighting up, which answered
# "how far along is it" and nothing else. These answer the questions actually
# worth asking about an AutoGen run: which team was assembled, which of the nine
# patterns it is, and which message types are in flight. The chat answer is the
# surprising one — there is **no team**.
RUNS: dict[str, dict[str, Any]] = {
    "chat": {
        "team": "yok — tek AssistantAgent",
        "team_note": "Beş takım tipinden hiçbiri: agent.run_stream() doğrudan çağrılıyor.",
        "pattern": "tool döngüsü",
        "pattern_note": "Dokuz desenden biri değil. max_tool_iterations=6 (varsayılan 1).",
        "messages": [
            "TextMessage", "ModelClientStreamingChunkEvent", "ToolCallRequestEvent",
            "ToolCallExecutionEvent", "TaskResult",
        ],
    },
    "scan": {
        "team": "GraphFlow · 5 katılımcı",
        "team_note": "teknik · pazar · ekip → risk denetçisi → skorlayıcı",
        "pattern": "eşzamanlı dal + join(all) → sıralı",
        "pattern_note": "Dokuz desenden ikisinin bileşimi: Concurrent Agents + Sequential.",
        "messages": [
            "TextMessage", "StructuredMessage[Score]", "ToolCallRequestEvent", "TaskResult",
        ],
        "declared": "custom_message_types=[StructuredMessage[Score]]",
    },
}

# Which diagram the panel should draw. A chat turn and a scan are different
# machines, and drawing one with the other's boxes greyed out taught less than
# just switching.
CHAT_FLOW = ("context", "model", "stream", "tool_request", "gate", "tool_exec", "done")
# The scan's real path: GraphFlow, with our own branch count standing in for the
# barrier. `fanin.py`'s core mechanisms are catalogued above but are not here,
# because `scan.py` does not call them.
SCAN_FLOW = ("graph_build", "intervention", "graph_run", "analysts", "join",
             "count", "runtime_stop")

# Drawn dim, always. See the module docstring: this lane is a measurement.
# The last sentence used to say the scan is where core really runs. It is not —
# `scan.py` calls `graph.enrich`, which is GraphFlow, and the comment forty lines
# up says so. The note contradicted its own file while being the one text the
# panel puts in front of a reader, which is the worst place for a stale claim.
CORE_IDLE_NOTE = (
    "Sohbet turu core'a hiç uğramıyor: /api/chat doğrudan conversation.stream'i "
    "çağırıyor. Runtime ayakta ve abonelikler kurulu, ama yönlendirilen mesaj yok. "
    "Tarama da core'a uğramıyor: scan.py, graph.enrich'i çağırıyor ve o GraphFlow "
    "— yani AgentChat. Buradaki core mekanizmaları fanin.py'de gerçek ve kurulu, "
    "ama onu yalnız compare_fanin.py koşturuyor."
)


def catalogue() -> list[dict[str, Any]]:
    """The whole catalogue, in draw order."""
    return [m.as_dict() for m in CATALOGUE.values()]


# --------------------------------------------------------------- across processes
#
# The scan runs as a subprocess (`server.ScanRun`), so its stages cannot ride an
# in-process queue. They go out on stdout as one tagged JSON line each, and the
# server lifts them back out of the log it is already draining.
#
# Off unless the server asks for it: a person running `python scan.py` by hand is
# reading that output, and machine lines in the middle of it are noise.
LINE_TAG = "##STAGE "
STREAM_ENV = "VC_STAGE_STREAM"


def line_streaming() -> bool:
    return os.getenv(STREAM_ENV, "") in ("1", "true", "yes")


def emit_line(stage_id: str, **meta: Any) -> None:
    """Report a stage from another process. Never raises; never blocks."""
    if not line_streaming() or stage_id not in CATALOGUE:
        return
    try:
        print(LINE_TAG + json.dumps({"id": stage_id, "meta": meta}, ensure_ascii=False),
              flush=True)
    except Exception:  # noqa: BLE001 — a scan must not die reporting on itself
        pass


def parse_line(line: str) -> dict[str, Any] | None:
    """Turn a tagged log line back into a stage event, or None if it is just log."""
    if not line.startswith(LINE_TAG):
        return None
    try:
        raw = json.loads(line[len(LINE_TAG):])
    except (json.JSONDecodeError, ValueError):
        return None
    mechanism = CATALOGUE.get(str(raw.get("id")))
    if mechanism is None:
        return None
    event = {"type": "stage", **mechanism.as_dict()}
    if raw.get("meta"):
        event["meta"] = raw["meta"]
    return event


class StageBus:
    """Stage events on their way to the stream, from wherever they happen.

    A queue rather than a plain list because the gate does not run inside the
    `run_stream` loop — it runs inside the hook chain, underneath a tool call the
    loop is still awaiting. Whoever emits must not have to know who is draining.

    Bounded and lossy on purpose. If the interface is slow or gone, a full queue
    drops the newest stage rather than applying backpressure to the turn: the
    panel is a window onto the work, and a window must never hold up the work.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)

    def emit(self, stage_id: str, **meta: Any) -> None:
        """Record that `stage_id` is happening. Never raises, never blocks."""
        mechanism = CATALOGUE.get(stage_id)
        if mechanism is None:
            # An unknown id is a bug in the caller, but a panel is not worth
            # breaking a turn over. Drop it; `test_stages` catches the drift.
            return
        event = {"type": "stage", **mechanism.as_dict()}
        if meta:
            event["meta"] = meta
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    def drain(self) -> list[dict[str, Any]]:
        """Everything emitted since the last drain, oldest first."""
        out: list[dict[str, Any]] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                return out


# ---------------------------------------------------------------- why not a hook
#
# The obvious way to watch the gate is to register a hook on `before_tool_call`
# at a positive order, so it runs after the gate and reads the decision out of
# the payload. It does not work, and the reason is worth writing down.
#
# `HookRegistry.run` stops the chain the moment a hook returns the point's
# terminal key (`gateway/hooks.py:180` — `block` for `before_tool_call`). So on
# exactly the runs worth watching — the blocked ones — a probe registered after
# the gate never executes. It would report every allowed call and go silent on
# every refusal: a monitor that works until it matters.
#
# So `GatedWorkbench` emits its own stages instead. It is the only place that
# sees all three outcomes — filtered by name, blocked by a hook, allowed through.

__all__ = [
    "AGENTCHAT", "CATALOGUE", "CHAT_FLOW", "CORE", "CORE_IDLE_NOTE", "LINE_TAG",
    "Mechanism", "OURS", "SCAN_FLOW", "STREAM_ENV", "RUNS", "StageBus", "catalogue",
    "emit_line", "line_streaming", "parse_line",
]
