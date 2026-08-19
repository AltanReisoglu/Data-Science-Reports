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

The chat no longer draws any of this. The panel that used to sit above the
composer was removed so the explaining could get an interface of its own; the
catalogue, the bus and `/api/mechanisms` are untouched and still emit on every
turn, which is what that interface will read. The one consumer left in the chat
is the terminal, which listens for `code_request` and `code_result` and ignores
the rest.

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
import time
from dataclasses import asdict, dataclass
from typing import Any

# Lanes. The order is the order they are drawn in.
AGENTCHAT = "agentchat"
MAF = "maf"
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
        # ---- kod yürütme: resmî sekiz desenin sonuncusu -----------------------
        #
        # Bu ikisi `tool_request`/`tool_result`'ın özel hâli, ve ayrı olmalarının
        # sebebi terminal: panel kodu ve çıktısını gösterebilmek için ham metne
        # ihtiyaç duyuyor, ve o metin normal tool meta'sına sığmıyor.
        Mechanism(
            "code_request", OURS, "Kod yazıldı",
            "CodeExecutor(code=…) → onay kapısı", "05:3054",
            "Model, uygun bir tool bulamadığı için kod yazdı. Çalışmadan önce "
            "kapıya takılıyor: adı hiçbir outbound markerına uymadığı için "
            "kanca onu adına göre değil, ne olduğuna göre yakalıyor.",
            "pipeline/codeexec.py",
        ),
        Mechanism(
            "code_result", CORE, "Konteyner cevapladı",
            "DockerCommandLineCodeExecutor.execute_code_blocks()", "05:3054",
            "Her blok bir dosyaya yazılıp ayrı süreçte koştu. Çıktı koşu bitince "
            "tek parça dönüyor — akış hâlinde değil, ve panel öyleymiş gibi "
            "göstermiyor.",
            "pipeline/codeexec.py",
        ),
        Mechanism(
            "loop", AGENTCHAT, "Döngü devam ediyor",
            "max_tool_iterations=6", "08:2298",
            # Ölçüldü (2026-08-18): varsayılanda ikinci tool hiç çağrılmıyor ve
            # kullanıcıya ham tool çıktısı gidiyor (ToolCallSummaryMessage) —
            # model sonucu okuyup cevabı yazmıyor. Zincirlemeyi açan
            # `max_tool_iterations`, modelin cevabı yazmasını açan ayrı bir
            # anahtar: `reflect_on_tool_use`.
            "Varsayılan 1'dir: zincir bir adım sonra durur ve kullanıcıya ham "
            "tool çıktısı gider. Bu değer zincirleme için elle yükseltildi.",
            "pipeline/conversation.py",
        ),
        Mechanism(
            "done", AGENTCHAT, "Tur bitti",
            "TaskResult.stop_reason", "08:2813",
            "TaskResult iki şey taşıyor: bütün konuşma ve neden durduğu.",
            "pipeline/conversation.py",
        ),

        # ---- MAF: halef çerçeve, ayrı sanal ortamda --------------------------
        Mechanism(
            "maf_build", MAF, "MAF kuruldu",
            "agent_framework 1.14.0 · OpenAIChatClient", "maf:Agent",
            "AutoGen'in resmî halefi. Ayrı bir sanal ortamda koşuyor: iki çerçeve "
            "aynı bağımlılık ağacını paylaşamıyor.",
            "pipeline/maf_runner.py",
        ),
        Mechanism(
            "maf_tool", MAF, "Tool tanımlandı",
            "FunctionTool(approval_mode, max_invocations)", "maf:FunctionTool",
            "Onay ve çağrı tavanı tool'un KENDİ alanları. AutoGen'de ikisi de yok: "
            "kapıyı sarmalayıcı olarak, tavanı ajan ayarı olarak biz kuruyoruz.",
            "pipeline/maf_runner.py",
        ),
        Mechanism(
            "maf_gate", MAF, "Kapı çerçevede",
            "ToolApprovalMiddleware", "maf:ToolApprovalMiddleware",
            "Onay ara katmanı hazır geliyor. Bizim GatedWorkbench'imizin karşılığı, "
            "ama yazılmış hâlde.",
            "pipeline/maf_runner.py",
        ),
        Mechanism(
            "maf_agent", MAF, "Ajan kuruldu",
            "agent_framework.Agent", "maf:Agent",
            "Ayrı bir run_stream() yok: akış run(stream=True) parametresi. "
            "AutoGen'de bunlar iki ayrı yüzey.",
            "pipeline/maf_runner.py",
        ),
        Mechanism(
            "maf_session", MAF, "Oturum açıldı",
            "AgentSession", "maf:AgentSession",
            "Onay ara katmanı oturumsuz koşuya takılamıyor — ölçüldü, "
            "RuntimeError ile düşüyor.",
            "pipeline/maf_runner.py",
        ),
        Mechanism(
            "maf_run", MAF, "Ajan koşuyor",
            "Agent.run(messages, session=...)", "maf:Agent",
            "Tek çağrı. Tool döngüsü çerçevenin içinde dönüyor.",
            "pipeline/maf_runner.py",
        ),
        Mechanism(
            "maf_approval", MAF, "Onay istendi",
            "AgentResponse.user_input_requests", "maf:AgentResponse",
            "Onay cevabın BİRİNCİ SINIF alanı: tur duruyor, finish_reason "
            "'tool_calls' oluyor ve bir function_approval_request dönüyor.",
            "pipeline/maf_runner.py",
        ),
        Mechanism(
            "maf_done", MAF, "MAF turu bitti",
            "AgentResponse", "maf:AgentResponse",
            "Tool çağrıldığında response.text BOŞ kalıyor ve cevap mesajların "
            "içinde — AutoGen'in reflect_on_tool_use varsayılanıyla aynı sonuç.",
            "pipeline/maf_runner.py",
        ),

        # ---- takımlar: beş tipin gerçekten koştuğu yol -------------------------
        Mechanism(
            "team_build", AGENTCHAT, "Takım kuruldu",
            "RoundRobin / Selector / Swarm / MagenticOne / GraphFlow", "08:1789",
            "Beş tipin tek farkı sırayı kimin belirlediği: sabit döngü, model "
            "seçimi, ajanın devri, planlayıcı, ya da önceden çizilmiş graf.",
            "pipeline/teams.py",
        ),
        Mechanism(
            "speaker", AGENTCHAT, "Sıra bir ajanda",
            "BaseGroupChatManager.select_speaker", "08:1908",
            "Konuşan değişti. Kimin konuşacağına karar veren mekanizma takım "
            "tipine göre değişiyor; ajanlar aynı kalıyor.",
            "pipeline/teams.py",
        ),
        Mechanism(
            "handoff", AGENTCHAT, "Devir",
            "Handoff(target=...) → tool çağrısı", "08:2093",
            "Swarm'da sırayı ajanın kendisi devrediyor ve devir bir TOOL "
            "çağrısı olarak gerçekleşiyor — ölçülen en pahalı desen bu yüzden.",
            "pipeline/teams.py",
        ),
        # Zamanlayıcı. Bizim yazdığımız hat: AutoGen'de zamanlama diye bir
        # kavram yok, ve olmaması bir eksiklik değil — bir kütüphane saat
        # tutmaz. Bu üç aşama `scheduler.py`'nin OpenClaw'ın cron'una devrettiği
        # işin ekrandaki karşılığı.
        Mechanism(
            "cron_parse", OURS, "Zamanlama okundu",
            "scheduler.parse_command()", "18:63",
            "\"her sabah 9'da\" gibi bir cümle cron ifadesine çevriliyor. "
            "Çeviremezse SORMUYOR — sözdizimini yazıp reddediyor.",
            "pipeline/scheduler.py",
        ),
        Mechanism(
            "cron_gate", OURS, "Zamanlama kapıda",
            "GATE.require() → cron.add", "18:150",
            "İş yaratmak dışarı yazan bir çağrı. Onay imzası KULLANICININ "
            "YAZDIĞI cümlenin üstünde: \"20dk sonra\" her ayrıştırmada başka "
            "bir zaman damgası veriyor ve sonucun üstündeki imza hiç tutmazdı.",
            "pipeline/server.py",
        ),
        Mechanism(
            "cron_done", OURS, "Zamanlayıcıya devredildi",
            "openclaw cron.add", "18:502",
            "Zamanlayıcı bizde değil. OpenClaw'ın Gateway sürecinde yaşıyor, "
            "işleri SQLite'ta tutuyor ve yeniden başlatmayı atlatıyor. "
            "gateway/cron.py yerli karşılığı — yazıldı, bağlanmadı.",
            "pipeline/scheduler.py",
        ),
        Mechanism(
            "team_tool", CORE, "Ajan tool çağırdı",
            "workbench.call_tool()", "05:2473",
            "Kadro sabit ama iş bölümü koşuya ait: hangi ajanın hangi tool'a "
            "uzandığı, takım tipinin gerçekten farklı davrandığının kanıtı.",
            "pipeline/teams.py",
        ),
        Mechanism(
            "team_done", AGENTCHAT, "Takım bitti",
            "TerminationCondition → TaskResult", "08:2813",
            "Sonlandırma koşulu tetiklendi. Koşulu olmayan takım sonsuza kadar "
            "konuşuyor ve fatura gerçek.",
            "pipeline/teams.py",
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
# Bir takım koşusunun yolu. Sohbetten farkı: burada gerçekten bir takım var,
# ve ekranın "takım yok" demediği tek yol bu.
TEAM_FLOW = ("team_build", "speaker", "team_tool", "handoff", "team_done",
             "runtime_stop")
# Zamanlama turu: sohbetin içinden çıkıyor, ayrı bir koşu değil. Şeritte
# göründüğü yer de bu yüzden sohbet akışının içi.
CRON_FLOW = ("cron_parse", "cron_gate", "cron_done")
# MAF turu. Ayrı bir şerit rengi var çünkü ayrı bir çerçeve — AutoGen'in
# katmanlarıyla aynı renge boyamak, ekranın anlattığı ayrımı silerdi.
MAF_FLOW = ("maf_build", "maf_tool", "maf_gate", "maf_agent", "maf_session",
            "maf_run", "maf_approval", "maf_done")

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
        print(LINE_TAG + json.dumps({"id": stage_id, "t": time.time(), "meta": meta},
                                    ensure_ascii=False),
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
    # The scan is a subprocess, so this timestamp was taken over there. Same
    # machine, same clock — and it is still closer to the truth than the moment
    # the server happened to read the line off stdout.
    event = {"type": "stage", "t": raw.get("t") or time.time(), **mechanism.as_dict()}
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
        # Emitted-at, not drained-at. The queue is drained only when the run
        # loop next yields, so a gate decision taken underneath an awaited tool
        # call reaches the reader seconds after it happened. Stamping here is
        # what keeps a timeline from collapsing six stages onto one instant —
        # measured: a real turn showed context, model, gate and tool_exec all at
        # +8.23s because the first drain was the first timestamp anyone took.
        event = {"type": "stage", "t": time.time(), **mechanism.as_dict()}
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


# --------------------------------------------------------------- uzun anlatım
#
# `Mechanism.note` bir cümle: şeridin altında, geçerken okunacak kadar. Bu tablo
# onun uzun hâli — bir aşamanın üstüne basıldığında baştan sona anlatılması için.
#
# Dört soru, hep aynı sırada, çünkü bir mekanizmayı anlamak hep aynı sırada
# oluyor: **ne** olduğu, **nasıl** çalıştığı, **neden** böyle kurulduğu, ve
# **nerede ısırdığı**. Dördüncüsü en değerlisi ve çoğu belgede hiç yazmıyor.
#
# Katalogla aynı dosyada duruyor, çünkü ikisi birlikte değişiyor: bir mekanizma
# adı değiştiğinde anlatımı da değişmeli, ve iki dosyaya bakmak birini unutmanın
# en kısa yolu.
DETAILS: dict[str, dict[str, str]] = {
    # ---------------------------------------------------------------- sohbet
    "context": {
        "what": "Modele gönderilecek mesaj kümesi burada seçiliyor. Ajanın "
                "belleği bu nesnede yaşıyor; `model_context` verilmezse ajanın "
                "belleği hiç olmuyor ve bu durum hata da vermiyor.",
        "how": "`CompactingChatCompletionContext.get_messages()` çağrılıyor ve "
               "geçmiş, token bütçesine göre kırpılıyor. AutoGen'in kendi "
               "`BufferedChatCompletionContext`'i **mesaj** sayar; bu bizim "
               "yazdığımız hâli **token** sayıyor.",
        "why": "Mesaj saymak, uzunlukları çok farklı mesajlarda bütçeyi tutmaz: "
               "on kısa mesaj ile on uzun mesaj aynı sayılır ama aynı maliyette "
               "değildir. Bütçe token cinsinden konuşulduğu için sayaç da token "
               "saymalı.",
        "trap": "Her tur sistem prompt'u, tool şemaları ve workbench tarifleri "
                "yeniden gönderiliyor. \"merhaba\" bile tam tarifeyi ödüyor — "
                "bağlamın tabanı sabit bir maliyet, ve bütçeyi asıl o belirliyor.",
    },
    "compaction": {
        "what": "Bağlam bütçeyi aştı ve eski mesajlar özete indirildi.",
        "how": "Model çağrısından hemen önce, tur ortasında oluyor. Kırpılan "
               "mesajlar atılmıyor; yerlerine bir özet konuyor.",
        "why": "Bütçeyi aşan istek reddedilir. Sıkıştırma, uzun bir konuşmanın "
               "turu düşürmeden sürmesini sağlıyor.",
        "trap": "Özet bir kayıptır. Sıkıştırma sonrası ajan, konuşmanın erken "
                "kısmındaki bir ayrıntıyı artık bilmiyor olabilir — ve bunu "
                "bilmediğini de bilmiyor.",
    },
    "model": {
        "what": "Model çağrısı. Ajan, bağlamı ve tool şemalarını modele "
                "gönderiyor ve bir karar bekliyor: cevap mı yazsın, tool mu "
                "çağırsın.",
        "how": "`create()` değil `create_stream()` çağrılıyor. İkisi farklı "
               "olaylar yayıyor: birincisi `LLMCallEvent`, ikincisi "
               "`LLMStreamEndEvent`.",
        "why": "Akış olmadan cevap tek parça geliyor ve arayüz, model bitirene "
               "kadar donmuş görünüyor. `model_client_stream=True` bunun için.",
        "trap": "Maliyeti yalnız `LLMCallEvent` dinleyerek sayan bir gözlemci, "
                "akışlı çağrılarda **sıfır** görür. Ölçüldü: iki sayaç birden "
                "gerekiyor, yoksa maliyet ya iki modda da yanlış ya hiç yok.",
    },
    "stream": {
        "what": "Cevap token token geliyor.",
        "how": "Her parça bir `ModelClientStreamingChunkEvent`. Tur bitmeden "
               "ekranda metin oluşmaya başlıyor.",
        "why": "Bekleme süresi değişmiyor, ama algılanan süre değişiyor: ilk "
               "token'ın ekrana düştüğü an, sistemin çalıştığının kanıtı.",
        "trap": "Akış açıkken nihai metin iki yerden gelebiliyor — parçalardan "
                "ve `TaskResult`'tan. İkisini birden yazan bir arayüz cevabı "
                "iki kez gösterir.",
    },
    "tool_request": {
        "what": "Model bir tool çağırmaya karar verdi. **Henüz hiçbir şey "
                "koşmadı** — bu yalnız bir istek.",
        "how": "`ToolCallRequestEvent`, tool adı ve modelin seçtiği "
               "argümanlarla geliyor. Şema, fonksiyon imzasından ve "
               "docstring'den üretilmiş.",
        "why": "İstek ile yürütmenin ayrı olaylar olması bir tasarım kararı: "
               "aradaki boşluk, kapının oturduğu yer. Tek olay olsaydı "
               "araya girecek yer olmazdı.",
        "trap": "Docstring dokümantasyon değil, **arayüz**. Yanlış yazılmış bir "
                "docstring, yanlış çağrılan bir tool demek — ve bu hata "
                "çalışma anında değil, cevabın içeriğinde görünür.",
    },
    "gate": {
        "what": "Kapı. Her tool çağrısının geçtiği tek nokta, ve bizim "
                "kodumuz — AutoGen'de böyle bir şey yok.",
        "how": "`GatedWorkbench`, herhangi bir `Workbench`'i sarmalıyor ve "
               "`before_tool_call` kancasını çalıştırıyor. Politika reddederse "
               "çağrı hiç yapılmıyor; geriye hata işaretli bir `ToolResult` "
               "dönüyor.",
        "why": "Red bir **istisna değil**, bir sonuç. İstisna turu düşürürdü; "
               "sonuç, ajanın gerekçeyi okuyup kullanıcıya söylemesine izin "
               "veriyor. Kapı, ajanın uyum göstermeyi seçmesine değil, hattın "
               "kendisine dayanıyor.",
        "trap": "İmza `(tool, argümanlar)` üstünde ve bir kez tüketiliyor. "
                "Model aynı soruya iki kez aynı programı yazmıyor — ölçüldü — "
                "yani onaylanan şeyin çalışması için **onaylanan metnin** "
                "saklanması gerekiyor, yeniden üretilenin değil.",
    },
    "tool_exec": {
        "what": "Tool gerçekten koşuyor.",
        "how": "`workbench.call_tool()` çağrılıyor. Workbench bir tool "
               "*kaynağı*: yerel fonksiyonlar `StaticWorkbench`'te, OpenClaw ve "
               "DeepWiki `McpWorkbench`'te — ajan için hepsi aynı arayüz.",
        "why": "Ajana `tools=` ile düz bir liste vermek yerine bir kaynak "
               "vermek, kaynağı sarmalanabilir yapıyor. Kapı tam olarak bunu "
               "kullanıyor; liste verilseydi araya girecek yer olmazdı.",
        "trap": "`tools=` ve `workbench=` aynı ajana birlikte verilemiyor: "
                "`ValueError: Tools cannot be used with a workbench.`",
    },
    "tool_result": {
        "what": "Sonuç ajana döndü ve bağlama girdi.",
        "how": "`ToolCallExecutionEvent`. Reddedilen çağrı da bu tiple dönüyor "
               "— hata işaretiyle.",
        "why": "İstek ve sonucun ayrı olaylar olması, aradaki kapının "
               "reddedebilmesinden geliyor: reddedilirse sonuç hiç olmaz.",
        "trap": "Varsayılanda ajan sonucu okuyup cevabı yazmıyor; ham tool "
                "çıktısı doğrudan kullanıcıya gidiyor "
                "(`ToolCallSummaryMessage`). Bunu açan ayrı bir anahtar var: "
                "`reflect_on_tool_use`.",
    },
    "loop": {
        "what": "Döngü devam ediyor: ajan sonucu gördü ve modeli tekrar çağırıyor.",
        "how": "`max_tool_iterations=6`. Bu değer elle yükseltildi.",
        "why": "Zincirleme davranış — bir tool'un çıktısını başka bir tool'a "
               "vermek — ancak birden fazla tur mümkünse oluyor.",
        "trap": "Varsayılan **1**. Yani varsayılanda ajan bir tool çağırır, "
                "sonucu görür ve **durur**. Hata vermez; yalnız zincirleme "
                "sessizce imkânsızdır.",
    },
    "code_request": {
        "what": "Model, mevcut tool'ların karşılamadığı bir iş için Python yazdı.",
        "how": "Kod normal bir tool çağrısı olarak gidiyor: aynı döngü, aynı "
               "kapı, aynı onay. Onay kartında çalışacak metnin kendisi görünüyor.",
        "why": "Kaçış kapağı, orkestrasyon dili değil. Model önce mevcut "
               "tool'lara bakıyor; uyan bir tool yoksa kod yazıyor.",
        "trap": "Onay, çalıştırılacak **metne** bağlanmalı. Model aynı soruya "
                "her seferinde farklı bir program yazıyor; onayı yeniden "
                "üretilen koda bağlarsan onay hiç tüketilemiyor.",
    },
    "code_result": {
        "what": "Program izole bir konteynerde koştu ve çıktısı döndü.",
        "how": "`PythonCodeExecutionTool` + `DockerCommandLineCodeExecutor`. "
               "Konteyner sürece ait, çağrıya değil: her çağrıda konteyner "
               "ayağa kaldırmak iki üç saniye ve demoyu öldürüyor.",
        "why": "Model kodu doğrudan makinede koşturmuyor; yıkım yarıçapı "
               "konteynerle sınırlanıyor.",
        "trap": "Konteynerin **ağ erişimi var**. `DockerCommandLineCodeExecutor` "
                "içinde `network_mode` diye bir parametre yok — ölçüldü. "
                "Varsayılan kapalı olması ve onay kartının bunu yazması, "
                "\"sandbox güvenli\" demenin yerine geçmiyor.",
    },
    "done": {
        "what": "Tur bitti.",
        "how": "`TaskResult` iki şey taşıyor: bütün konuşma (`messages`) ve "
               "neden durduğu (`stop_reason`).",
        "why": "Turun nasıl bittiği, ne ürettiği kadar önemli: süre sınırı mı "
               "doldu, sonlandırma koşulu mu tetiklendi, model mi bitirdi.",
        "trap": "`stop_reason` boşsa takım değil **tek ajan** koştu — "
                "sonlandırma koşulu yalnız takımlarda var. Ayrıca bu aşamanın "
                "`tool_calls` sayacı yalnız **koşan** çağrıları sayıyor: kapı "
                "bir çağrıyı tuttuğunda sıfır kalıyor, ve sıfıra bakıp \"model "
                "tool çağırmıyor\" demek yanlış bir teşhis oluyor.",
    },
    # ---------------------------------------------------------------- takımlar
    "team_build": {
        "what": "Beş takım tipinden biri kuruldu ve aynı üç ajanla koşuyor. "
                "`MagenticOneGroupChat` bir istisna: kadroya ek olarak kendi "
                "yöneticisini de yaratıyor.",
        "how": "Kadro sabit: Planner · Researcher · Critic. Değişen tek şey "
               "sırayı belirleyen mekanizma.",
        "why": "Kadro da değişseydi ölçülen token farkının takım tipinden mi "
               "kadrodan mı geldiği belirsiz kalırdı.",
        "trap": "Sonlandırma koşulu olmayan takım sonsuza kadar konuşuyor. "
                "Buradaki `MaxMessageTermination` bir maliyet tavanı, üslup "
                "tercihi değil. Ayrı bir uyarı: `MagenticOne` preset'i "
                "(WebSurfer · FileSurfer · Coder · ComputerTerminal) tarayıcı "
                "ve kabuk kullanıyor; resmî kılavuz onu konteynerde ve insan "
                "gözetiminde koşturmayı, logları izlemeyi ve web sayfalarından "
                "gelen prompt injection'a dikkat etmeyi şart koşuyor. Biz o "
                "preset'i koşturmuyoruz — yalnız yöneticiyi kendi ajanlarımızın "
                "üstünde kullanıyoruz.",
    },
    "speaker": {
        "what": "Sıra bir ajana geçti.",
        "how": "RoundRobin sırayla dolaşıyor; Selector her turdan önce bir model "
               "çağrısıyla soruyor; Swarm'da ajanın kendisi devrediyor; "
               "MagenticOne'da bir planlayıcı dağıtıyor; GraphFlow'da sıra "
               "önceden çizilmiş graftan geliyor.",
        "why": "Bir takımı diğerinden ayıran şey ajanları değil, sıranın "
               "nereden geldiği.",
        "trap": "`SelectorGroupChat`, ajanların `description` alanına bakarak "
                "seçiyor. Boşsa seçim kör yapılıyor ve hata da vermiyor.",
    },
    "cron_parse": {
        "what": "Türkçe bir zamanlama cümlesi cron ifadesine çevrildi.",
        "how": "`parse_command` metni ayrıştırıyor; anlamadığında `WhenError` "
               "fırlatıp kabul edilen sözdizimini yazıyor.",
        "why": "Zamanlamada tahmin etmek pahalı: yanlış okunan bir cümle her "
               "gün yanlış saatte koşan bir iş demek, ve kimse fark etmiyor.",
        "trap": "Cron'un gün alanları **OR**'lanıyor: `0 9 * * 1` ile "
                "`0 9 1 * *` birlikte yazılırsa hem pazartesileri hem ayın "
                "birinde koşuyor. OpenClaw bunu ayrı ayrı iş açarak çözüyor.",
    },
    "cron_gate": {
        "what": "Zamanlanmış iş yaratma çağrısı kapıya geldi.",
        "how": "`GATE.require()` — imza **kullanıcının yazdığı cümlenin** "
               "üstünde, çözülmüş zaman damgasının değil.",
        "why": "\"20dk sonra\" her ayrıştırmada başka bir zaman veriyor. "
               "Sonucun üstündeki bir imza hiçbir zaman tutmazdı ve onay "
               "sistemi sessizce işe yaramaz hâle gelirdi.",
        "trap": "Zamanlanmış iş, onayı **gelecekteki** bir koşuya taşıyor: "
                "onaylayan kişi o an odada olmayacak. Bu yüzden her koşu "
                "kendi taze oturumunu alıyor ve bağlam taşımıyor.",
    },
    "cron_done": {
        "what": "İş OpenClaw'ın zamanlayıcısına yazıldı.",
        "how": "`cron.add` — iş OpenClaw'ın Gateway sürecinde, SQLite'ta "
               "yaşıyor; süreç yeniden başlarsa geçmiş işleri tekrar "
               "oynatmıyor, yeniden zamanlıyor.",
        "why": "Zamanlayıcı bizim değil, ve bilinçli: OpenClaw'ın zaten "
               "koşan, kalıcı ve yeniden başlatmayı atlatan bir zamanlayıcısı "
               "var. Gecelik bir tarama için onu yeniden yazmak, ölçtüğümüz "
               "kararları yeniden türetmek olurdu.",
        "trap": "Bu devir bir bağımlılık: OpenClaw ayakta değilse zamanlanmış "
                "hiçbir iş koşmuyor ve bunu söyleyen bir alarm yok. "
                "`gateway/cron.py` yerli karşılığı — yazıldı ve testli, ama "
                "hiçbir yerden çağrılmıyor.",
    },
    "team_tool": {
        "what": "Takımdaki bir ajan bir tool çağırdı.",
        "how": "`ToolCallRequestEvent`in `content`inde çağrının adı var; olay "
               "hangi ajandan geldiyse `source` orada yazıyor. Devir tool'ları "
               "(`transfer_to_*`) dışarıda tutuluyor — onlar zaten `handoff` "
               "olarak çiziliyor ve iki kez göstermek Swarm'ı kalabalık yapar.",
        "why": "Konuşma sırası **kimin** konuştuğunu söylüyor, bu **ne yaptığını**. "
               "İkisi olmadan beş takım tipi ekranda birbirinin aynı üç kutu "
               "olarak duruyor, ve grafın anlatması gereken tek fark iş bölümü.",
        "trap": "Takımdaki ajanlara tool vermezsen bu aşama hiç çıkmaz ve graf "
                "yalnız konuşma sırasını gösterir — eksik değil, ama takımın "
                "gerçekten iş yaptığını da göstermez.",
    },
    "handoff": {
        "what": "Bir ajan işi başka bir ajana devretti.",
        "how": "Devir bir **tool çağrısı**: `Handoff(target=...)` bir "
               "`transfer_to_<ajan>` tool'u üretiyor ve model onu çağırıyor.",
        "why": "Yönlendirme kararı dışarıdaki bir yöneticide değil, ajanın "
               "kendisinde. Agents SDK'nın tek modeli bu.",
        "trap": "Tool adı küçük harfe düşüyor; elle yazınca eşleşmiyor. "
                "Ve ölçüldü: her devir bir tur demek — 334 token ile en pahalı "
                "desen.",
    },
    "team_done": {
        "what": "Sonlandırma koşulu tetiklendi ve takım durdu.",
        "how": "`TaskResult` bütün konuşmayı ve `stop_reason`'ı taşıyor.",
        "why": "Turun nasıl bittiği ne ürettiği kadar önemli: koşul mu doldu, "
               "mesaj tavanı mı, yoksa ajanlar mı bitirdi.",
        "trap": "`stop_reason` boşsa takım değil tek ajan koşmuştur — "
                "sonlandırma koşulu yalnız takımlarda var.",
    },
    # ---------------------------------------------------------------- tarama
    "graph_build": {
        "what": "Graf kuruluyor: beş katılımcı ve aralarındaki kenarlar.",
        "how": "`DiGraphBuilder` ile düğümler ekleniyor, üç analist dalından "
               "risk denetçisine giden kenarlara `activation_condition=\"all\"` "
               "veriliyor, sonra `GraphFlow` üretiliyor.",
        "why": "Zenginleştirme sabit ve paralel. GraphFlow, AutoGen'de gerçek "
               "eşzamanlılığı olan tek takım — diğer dördü sırayla konuşuyor.",
        "trap": "Grafın kenarları **veri taşımıyor**. Bütün katılımcılar tek bir "
                "paylaşılan `group_topic_type`'a abone; mesaj zaten herkese "
                "gidiyor ve kenarlar yalnız sıranın kimde olduğunu belirliyor.",
    },
    "graph_run": {
        "what": "Graf koşmaya başladı.",
        "how": "Görev tek bir yayınla giriyor ve üç dal aynı anda çalışmaya "
               "başlıyor.",
        "why": "Paralellik bir model kararı değil, bir abonelik tablosu "
               "sonucu: yayınlayan taraf kaç dalın dinlediğini bilmiyor.",
        "trap": "Bir dal çökerse `gather` erken dönüyor ve tamamlanmış "
                "kardeşlerin sonuçları sessizce kaybolabiliyor — ölçüldü, "
                "deterministik bile değil.",
    },
    "analysts": {
        "what": "Üç analist dalı paralel koşuyor: teknik, pazar, ekip.",
        "how": "Üçü de aynı görevi farklı gözle okuyor ve kimse kimseyi "
               "beklemiyor.",
        "why": "Kılavuzun *Concurrent Agents* deseni: tek yayın → çok dal → "
               "toplayıcı.",
        "trap": "Resmî desenler bu konuda birbiriyle çelişiyor: *Concurrent "
                "Agents* sonucu kuyrukla topluyor, *Mixture of Agents* "
                "`asyncio.gather` ile — ve sessiz kaybın kaynağı ikincisi.",
    },
    "join": {
        "what": "Birleşme. Risk denetçisi, üç dal da gelmeden başlamıyor.",
        "how": "Beklenen dal **sayısı** sayılıyor; runtime'ın \"boşta\" demesi "
               "beklenmiyor.",
        "why": "`stop_when_idle()` bariyeri bir dal çöktüğünde erken açılıyor. "
               "Sayarak beklemek, o bariyere güvenmemek demek.",
        "trap": "Gelmeyen her dal `missing_data`'ya yazılıyor: sessiz bir "
                "eksik, beyan edilmiş bir bilgi yokluğuna çevriliyor. Eksik "
                "veriyle düşük skor aynı şey değil.",
    },
    "count": {
        "what": "Skorlayıcı şemaya bağlı çıktı üretiyor.",
        "how": "`StructuredMessage[Score]` — alanları önceden tanımlı bir mesaj.",
        "why": "Serbest metin bir skoru taşıyabilir ama doğrulayamaz. Şema, "
               "eksik alanı bir hata hâline getiriyor.",
        "trap": "Takıma `custom_message_types=[...]` ile beyan edilmezse runtime "
                "`Message type StructuredMessage[X] is not registered` diyerek "
                "düşüyor.",
    },
    "intervention": {
        "what": "Müdahale kapısı: runtime'a takılan tek kapı.",
        "how": "`AuditingInterventionHandler.on_publish` — her mesaj buradan "
               "geçiyor ve denetim kaydına yazılıyor. `DropMessage` dönerse "
               "teslimat hiç olmuyor.",
        "why": "AgentChat'in mesaj listesi konuşmayı gösteriyor; bu, "
               "yönlendirmeyi gösteriyor. Kapı ajanın davranışına değil, "
               "runtime'a bağlı.",
        "trap": "Takmanın bedeli runtime'ı kendin kurmak. Ve kendi runtime'ında "
                "çöken bir ajan fırlatmıyor, **asılıyor**.",
    },
    "runtime_start": {
        "what": "Aktör runtime'ı açıldı; mesaj işleme döngüsü başladı.",
        "how": "`SingleThreadedAgentRuntime.start()`.",
        "why": "Ajanlar gerçekten aktör: kendi mailbox'ı olan, mesajı **tipe "
               "göre** yönlendiren birimler.",
        "trap": "Bu runtime kısa ömürlü — şirket başına kurulup kapatılıyor. "
                "Gateway'inki ise sürekli koşuyor; ikisi ayrı ömür ve "
                "karıştırılırsa oturum durumu beklenmedik yerde kayboluyor.",
    },
    "subscribe": {
        "what": "Abonelikler kuruldu.",
        "how": "`TypeSubscription(topic_type, agent_type)`. Üç dal da **aynı** "
               "topic tipine abone.",
        "why": "Topic kaynağı, ajan anahtarına dönüşüyor: `TopicId(\"turn\", "
               "\"oturum-42\")`'ye yayın yapmak `AgentId(\"session\", "
               "\"oturum-42\")` ajanını yaratıyor — oturum başına izole örnek, "
               "elle sözlük tutmadan.",
        "trap": "Yayınlayan taraf kaç abonesi olduğunu bilmiyor. Bu paralelliğin "
                "kaynağı, ama aynı zamanda \"kaç sonuç bekleyeceğim\" sorusunun "
                "cevabının runtime'da olmamasının da sebebi.",
    },
    "publish": {
        "what": "Görev yayınlandı.",
        "how": "`runtime.publish_message(BranchTask, TopicId(...))` — tek satır.",
        "why": "Tek yayın, üç ajan koşuyor. Tool değil, model kararı değil: düz "
               "bir Python satırı ve bir abonelik tablosu araması.",
        "trap": "Yayının dönüş değeri **yok**. Bir sonucu bekleyeceksen doğrudan "
                "`send_message`, bir olayı duyuracaksan yayın. Handler çökerse "
                "yayında loglanıyor, doğrudan çağrıda çağırana fırlatılıyor.",
    },
    "branch": {
        "what": "Bir dal çalışıyor.",
        "how": "`BranchWorker(RoutedAgent).handle` — handler'ı **tip anotasyonu** "
               "seçti, bir if/else değil.",
        "why": "Tipe göre yönlendirme, yeni bir mesaj tipi eklendiğinde "
               "yönlendirme kodunu değiştirmeyi gereksiz kılıyor.",
        "trap": "Bu ajan asla fırlatmıyor: hata da bir sonuç olarak "
                "yayınlanıyor. Fırlatsaydı kardeşlerinin işini de götürürdü.",
    },
    "collect": {
        "what": "Sonuç kuyruğa düştü.",
        "how": "`ClosureAgent` → `asyncio.Queue`. Sonuç üretildiği anda "
               "yayınlandı ve kuyruk onu çoktan tuttu.",
        "why": "Güvenilmeyecek bariyer yok, çünkü bariyer yok. Ölçüldü: bu yol "
               "arıza altında iki sonucu ~3 ms'de topluyor, GraphFlow ise sıfır "
               "ya da bir sonuçla süre sınırına giriyor.",
        "trap": "Kuyruğun sınırı ve tüketicisi olmalı. Tüketici yoksa sonuçlar "
                "birikiyor ve kimse fark etmiyor.",
    },
    "runtime_stop": {
        "what": "Runtime kapandı.",
        "how": "`stop_when_idle()` süre sınırıyla, sonra `close()`.",
        "why": "Toplama bittikten **sonra** çağrılıyor: bariyer burada bir "
               "kapatma aracı, sonuç toplama aracı değil.",
        "trap": "Aynı çağrıyı sonuç toplamak için kullanmak, bu projede ölçülen "
                "sessiz veri kaybının tam kaynağı.",
    },
}


def detail(stage_id: str) -> dict[str, str] | None:
    """Bir aşamanın uzun anlatımı; yoksa None."""
    return DETAILS.get(stage_id)


__all__ = [
    "AGENTCHAT", "CATALOGUE", "CHAT_FLOW", "CORE", "CORE_IDLE_NOTE", "DETAILS",
    "LINE_TAG", "Mechanism", "OURS", "SCAN_FLOW", "STREAM_ENV", "RUNS", "StageBus",
    "TEAM_FLOW", "MAF", "MAF_FLOW",
    "catalogue", "detail", "emit_line", "line_streaming", "parse_line",
]
