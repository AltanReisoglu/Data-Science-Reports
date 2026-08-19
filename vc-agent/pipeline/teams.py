"""Beş takım tipini gerçekten kuran ve koşturan katman.

Bu dosyaya kadar akış ekranı her sohbet turunda **"takım yok"** diyordu, ve doğru
söylüyordu: boru hattı tek bir `AssistantAgent` koşturuyor. Ama bu, ekranın en
çok anlatması gereken şeyi — beş takım tipinin farkını — yalnız *anlatılabilir*
bırakıyordu, gösterilebilir değil.

Burada beşi de kuruluyor ve aynı soruya koşturulabiliyor. Ekran artık gerçek bir
takımı çiziyor: kim konuştu, sırayı kim seçti, kaç token gitti.

### Neden aynı kadro

Beş takım aynı üç ajanı paylaşıyor. Fark yalnız **sırayı kimin belirlediği**
olsun diye: kadro değişseydi ölçülen token farkı takım tipinden mi kadrodan mı
geldiği belirsiz kalırdı. `poc/kiyas.py` de aynı disiplinle ölçmüştü —
Selector 204 · GraphFlow 270 · RoundRobin 274 · Swarm 334.

### Ölçülmüş tuzaklar, hepsi burada karşılandı

* `Swarm`'ın devri bir tool çağrısı: `Handoff(target=...)`. Adı **küçük harfe**
  düşüyor, elle yazınca eşleşmiyor — `Handoff(...).name` ile üretiliyor.
* `SelectorGroupChat` seçimi bir model çağrısıyla yapıyor; ajanların
  `description`'ı boşsa seçim kör oluyor. Üçünde de dolu.
* Sonlandırma koşulu olmayan takım sonsuza kadar konuşuyor. Hepsinde
  `MaxMessageTermination` var ve bu bir maliyet tavanı, bir üslup tercihi değil.
* `MagenticOneGroupChat` bir yönetici modeli istiyor ve en pahalısı; varsayılan
  tur sayısı bilerek düşük tutuldu.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

import config
import engine
import observability
import stages as stages_module

# Takım tipleri, ekranda göründükleri sırayla. `picker` tek ayırt edici soruyu
# cevaplıyor: sırayı kim belirliyor.
KINDS = ("roundrobin", "selector", "swarm", "magenticone", "graphflow")

# Kadro. Üç ajan, üç ayrı iş — ve `description` alanları dolu, çünkü Selector
# tam olarak o metne bakarak sıradaki konuşmacıyı seçiyor.
ROSTER = [
    {
        "name": "Planner",
        "description": "Görevi alt adımlara bölen ve kimin ne yapacağını söyleyen ajan.",
        "system": "Sen bir planlayıcısın. Görevi en fazla üç adıma böl, her adımın "
                  "yanına hangi ajanın yapacağını yaz. Kendin araştırma yapma.",
        # Bilerek tool'suz: planlayıcının işi aramak değil, dağıtmak. Ve grafta
        # tool'u olmayan bir kutunun durması, iş bölümünün gerçek olduğunu
        # tool'u olanlardan daha iyi anlatıyor.
        "tools": (),
    },
    {
        "name": "Researcher",
        "description": "Soruyu kendi bilgisiyle cevaplayan, olguları toplayan ajan.",
        "system": "Sen bir araştırmacısın. **Tek bir** tool çağrısı yap: belge "
                  "sorusu için `search_docs`, tarama sorusu için `scan_facts`. "
                  "Aynı aramayı tekrar etme. Sonra bulduğunu en fazla beş madde "
                  "hâlinde yaz. Emin olmadığın yeri 'emin değilim' diye işaretle.",
        "tools": ("search_docs", "scan_facts"),
    },
    {
        "name": "Critic",
        "description": "Üretilen cevabı eksik ve çelişki için denetleyen ajan.",
        "system": "Sen bir eleştirmensin. Önceki cevabı oku, eksik ve çelişkili "
                  "yerleri say. **En fazla bir** `search_docs` çağrısıyla tek bir "
                  "iddiayı kontrol edebilirsin; fazlası yasak. Sorun kalmadıysa "
                  "yalnızca 'ONAY' yaz.",
        "tools": ("search_docs",),
    },
]


def _tool_catalogue() -> dict[str, Any]:
    """Takım ajanlarına verilecek tool'lar — sohbetin kullandığı fonksiyonların
    aynısı, aynı `gateway/tools.py`'den.

    Neden tool veriliyor: kadro baştan tool'suzdu ve beş takım tipi ekranda
    birbirinin aynı üç konuşan kutu olarak duruyordu. Konuşan bir takım desen
    farkını göstermiyor; **iş bölümü** gösteriyor, ve iş bölümü ancak ajanların
    yapacak ayrı işleri varsa görünür.

    Kapı burada YOK, ve bu bilinçli: takım koşusu bir kıyas yüzeyi, üretim yolu
    değil. Verilen iki tool da salt okunur (`search_docs`, `scan_facts`) —
    dışarı yazan bir tool'u kapısız vermek başka bir karar olurdu.
    """
    from autogen_core.tools import FunctionTool

    import gateway.tools as tools_module

    out: dict[str, Any] = {}
    # `scan_getter` en son taramayı diskten okuyor; sohbetin kullandığı yolun
    # aynısı. `scan_starter` yok — takım tarama BAŞLATAMAZ, ve o bir yetenek
    # kısıtı değil bir kapsam kararı: kıyas turu dışarıya iş açmamalı.
    sources = tools_module.Sources(
        scan_getter=tools_module.read_latest_scan, session_id="team")
    for fn in tools_module.build(sources):
        name = getattr(fn, "__name__", "")
        if name in ("search_docs", "scan_facts"):
            out[name] = FunctionTool(
                fn, description=(fn.__doc__ or "").strip().split("\n")[0])
    return out


def available() -> bool:
    """Takım koşturmak için canlı bir model gerekiyor; kuru modda desen yok."""
    return config.live_llm_available()


# --------------------------------------------------------------------- kurulum
def _agents(ledger: "engine.Ledger", *, handoffs: bool = False) -> list[Any]:
    """Kadroyu kur. `handoffs=True` iken Swarm'ın devir tool'ları ekleniyor."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.base import Handoff

    names = [a["name"] for a in ROSTER]
    catalogue = _tool_catalogue()
    built = []
    for spec in ROSTER:
        extra: dict[str, Any] = {}
        picked = [catalogue[t] for t in spec.get("tools", ()) if t in catalogue]
        if picked:
            # `max_tool_iterations` ELLE veriliyor, ve iki ucu da ölçüldü.
            # Varsayılan **1**: ajan tool'u çağırır, sonucu görür ve DURUR —
            # cevabı hiç yazmaz, tur sessizce boş geçer.
            # **4** ise ters uca düştü: ilk canlı koşuda üç ajan 41 tool çağrısı
            # yaptı, 74.612 token ve 124 saniye. Sunumda izlenemez.
            # **2** ikisinin arasında: bir arama, bir de cevabı yazacak tur.
            # Sistem prompt'ları da "tek çağrı" diyor — ikisi birlikte tutuyor.
            extra["tools"] = picked
            extra["max_tool_iterations"] = 2
        if handoffs:
            # Devir bir TOOL çağrısı. Adı küçük harfe düşüyor, o yüzden elle
            # yazılmıyor — `Handoff(...)` nesnesi üretiyor.
            targets = [n for n in names if n != spec["name"]]
            extra["handoffs"] = [Handoff(target=t) for t in targets]
        built.append(AssistantAgent(
            spec["name"],
            description=spec["description"],
            system_message=spec["system"] + (
                "\nİşin bittiğinde uygun ajana devret." if handoffs else ""),
            model_client=ledger.raw_client("mid"),
            model_client_stream=False,
            **extra,
        ))
    return built


def build(kind: str, ledger: "engine.Ledger", *, max_messages: int = 6,
          runtime: Any = None) -> Any:
    """İstenen takımı kur.

    `max_messages` bir maliyet tavanı. Sonlandırma koşulu olmayan bir takım
    sonsuza kadar konuşuyor ve fatura gerçek — bu yüzden varsayılanı var ve
    düşük.
    """
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import (
        DiGraphBuilder, GraphFlow, MagenticOneGroupChat, RoundRobinGroupChat,
        SelectorGroupChat, Swarm,
    )

    stop = MaxMessageTermination(max_messages)
    common: dict[str, Any] = {"termination_condition": stop}
    if runtime is not None:
        common["runtime"] = runtime

    if kind == "roundrobin":
        return RoundRobinGroupChat(_agents(ledger), **common)

    if kind == "selector":
        # Seçim başına bir model çağrısı daha. Ölçülen en ucuz desen bu, çünkü
        # gereksiz konuşmacıyı hiç çağırmıyor.
        # Katmanlar `cheap · mid · strong`; "small" diye bir katman yok ve
        # `KeyError: 'small'` ile takım daha ilk anda düşüyordu. Seçim kısa bir
        # karar, o yüzden en ucuzu.
        return SelectorGroupChat(_agents(ledger),
                                 model_client=ledger.raw_client("cheap"), **common)

    if kind == "swarm":
        # Swarm ilk katılımcıdan başlıyor ve sırayı ajanlar devrediyor.
        return Swarm(_agents(ledger, handoffs=True), **common)

    if kind == "magenticone":
        return MagenticOneGroupChat(_agents(ledger),
                                    model_client=ledger.raw_client("mid"), **common)

    if kind == "graphflow":
        agents = _agents(ledger)
        builder = DiGraphBuilder()
        for agent in agents:
            builder.add_node(agent)
        # Planner → (Researcher, Critic): iki dal paralel, sonra graf duruyor.
        builder.add_edge(agents[0], agents[1])
        builder.add_edge(agents[0], agents[2])
        builder.set_entry_point(agents[0])
        return GraphFlow(agents, graph=builder.build(), **common)

    raise ValueError(f"bilinmeyen takım tipi: {kind}")


# ---------------------------------------------------------------------- koşu
async def run(kind: str, task: str, *, bus: Any = None, spans: list | None = None,
              max_messages: int = 6) -> AsyncIterator[dict[str, Any]]:
    """Takımı koştur ve olaylarını akıt.

    Aşamalar `bus`'a yayınlanıyor — sohbet turunun kullandığı `StageBus`'ın
    aynısı, yani akış ekranı ikisini de aynı şekilde çiziyor.
    """
    from autogen_core import SingleThreadedAgentRuntime

    import telemetry as telemetry_module

    if kind not in KINDS:
        raise ValueError(f"bilinmeyen takım tipi: {kind}")

    ledger = engine.Ledger()
    provider, collector = telemetry_module.provider()
    runtime = SingleThreadedAgentRuntime(tracer_provider=provider)
    runtime.start()
    team = build(kind, ledger, max_messages=max_messages, runtime=runtime)

    if bus is not None:
        bus.emit("team_build", kind=kind,
                 participants=[a["name"] for a in ROSTER],
                 picker=PICKER[kind], termination=f"MaxMessageTermination({max_messages})")

    # Maliyet, sohbet turunun saydığı yerden sayılıyor: `autogen_core`'un olay
    # akışı. Ölçüldü: `Ledger`'ın `create_calls` sayacı yalnız replay
    # istemcisinde dolu, canlıda sıfır — ve ekran "16 saniye koştu, 0 LLM
    # çağrısı" diyordu. Üç ajan konuşmuşken sıfır imkânsız bir sayı.
    capture = observability.EventCapture()
    started = time.time()
    speakers: list[str] = []
    try:
      with capture:
        async for event in team.run_stream(task=task):
            name = type(event).__name__
            source = getattr(event, "source", "") or ""
            if name == "TaskResult":
                if bus is not None:
                    bus.emit("team_done", speakers=speakers,
                             stop_reason=getattr(event, "stop_reason", "") or "",
                             messages=len(getattr(event, "messages", []) or []),
                             llm_calls=capture.totals.llm_calls,
                             tokens=capture.totals.total_tokens,
                             tool_calls=len(capture.totals.tool_calls),
                             seconds=round(time.time() - started, 2))
                yield {"type": "done", "text": _final(event),
                       "stop_reason": getattr(event, "stop_reason", "") or ""}
                continue
            if source and source not in ("user",):
                speakers.append(source)
                if bus is not None:
                    bus.emit("speaker", who=source, turn=len(speakers), event=name)
            # Devir, Swarm'da bir tool çağrısı olarak görünüyor.
            if name == "ToolCallRequestEvent" and kind == "swarm" and bus is not None:
                bus.emit("handoff", who=source, to=_handoff_target(event) or "?")
            # Hangi ajan hangi tool'u çağırdı. Konuşma sırası "kim konuştu"yu
            # söylüyordu ama "ne yaptı"yı söylemiyordu, ve ekranda üç ajan
            # birbirinin aynı üç kutu olarak duruyordu. Devir tool'ları
            # (`transfer_to_*`) dışarıda: onlar zaten `handoff` olarak
            # çiziliyor, ve iki kez göstermek Swarm'ı gereksiz kalabalık yapar.
            if name == "ToolCallRequestEvent" and bus is not None:
                for call in getattr(event, "content", None) or []:
                    tool = getattr(call, "name", "") or ""
                    if tool and not tool.startswith("transfer_to"):
                        bus.emit("team_tool", who=source, tool=tool)
            yield {"type": "message", "source": source, "kind": name,
                   "text": _text(event)}
    finally:
        await runtime.stop()
        await ledger.close()
        telemetry_module.flush(provider)
        # Span'ler kapanışta toplanıyor: koşarken okumak yarım span'ler verir ve
        # şelalede bitiş çizgisi olmayan çubuklar çıkardı.
        report = collector.report()
        if spans is not None:
            spans.extend(report)
        if bus is not None:
            bus.emit("runtime_stop", spans=len(report))


PICKER = {
    "roundrobin": "sırayla", "selector": "model seçer", "swarm": "handoff",
    "magenticone": "planlayıcı", "graphflow": "DAG",
}


def _text(event: Any) -> str:
    content = getattr(event, "content", "")
    return content if isinstance(content, str) else str(content)[:400]


def _final(result: Any) -> str:
    messages = getattr(result, "messages", None) or []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _handoff_target(event: Any) -> str | None:
    """Devrin hedefi tool adının içinde: `transfer_to_<ajan>`.

    Tool adı **küçük harfe düşüyor**, ve ham hâliyle döndürünce ekranda var
    olmayan bir şeride (`researcher` ≠ `Researcher`) ok çiziliyordu. Kadroyla
    büyük/küçük harf duyarsız eşleştirip gerçek adı döndürüyoruz.
    """
    try:
        for call in getattr(event, "content", []) or []:
            name = str(getattr(call, "name", ""))
            if not name.startswith("transfer_to_"):
                continue
            raw = name[len("transfer_to_"):]
            for spec in ROSTER:
                if spec["name"].lower() == raw.lower():
                    return spec["name"]
            return raw
    except Exception:  # noqa: BLE001
        pass
    return None
