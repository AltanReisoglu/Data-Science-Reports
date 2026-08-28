"""LangGraph ajan grafiği — erişim yolu tool'larını `trace`'e bağlı olarak kurar.

.env okuma (load_dotenv) burada değil, cli.py'de (uygulama giriş noktası) yapılır;
bu modül os.environ'un zaten dolu olduğunu varsayar.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from grounded_assistant.access_paths import knowledge_base, live_systems
from grounded_assistant.agent import tool_policy
from grounded_assistant.models import SandboxRunStatus
from grounded_assistant.ptc.sandbox_runner import run_sandbox
from grounded_assistant.trace import Trace


def _build_model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL_NAME"],
    )


def _make_ptc_tool(trace: Trace, on_ptc_event: Callable[[dict], None] | None = None):
    """T014 — Faz 2. Kod, ayrı ve Cilium ile ağ-kısıtlı bir Kubernetes Job'unda
    çalışır (sandbox_runner.run_sandbox); FR-011: başarısız bir çalıştırmadan
    tahmini değer üretilmez, modele açıkça 'üretme' talimatı döner. tool_calls
    T015 gereği Trace'e beslenir (aynı LiveToolCall/record_tool_call, Faz 1'deki
    canlı-sistem yoluyla birebir aynı mekanizma).

    `on_ptc_event` (Faz 4, T006): verilirse `run_sandbox`'a olduğu gibi iletilir
    — web arayüzünün sol-alt canlı paneli bunu kullanır (contracts/
    websocket_protocol.md). CLI hiç geçmez (`None`), davranışı değişmez."""

    @tool
    def run_ptc_code(code: str) -> str:
        """LLM'in ürettiği Python kodunu, Tool Gateway dışında hiçbir yere
        çıkamayan (Cilium/eBPF ile ağ seviyesinde kısıtlı) ayrı bir Kubernetes
        pod'unda çalıştırır. Kod, search_knowledge_base(query)/
        get_ticket_status(ticket_id)/list_open_tickets() tool proxy'lerini normal
        Python fonksiyonu gibi çağırıp set_result(deger) ile nihai sonucu
        bildirmelidir. Birden fazla tool çağrısını tek turda, programatik olarak
        (döngü/koşul ile) sıralamak istediğinde bunu kullan."""
        run = run_sandbox(code, on_event=on_ptc_event)
        trace.record_sandbox_run(run)  # SC-003: çalıştırmanın kendisi (T017)
        for call in run.tool_calls:
            trace.record_tool_call(call)  # T015: çalıştırma içindeki her tool çağrısı
        for action in run.denied_actions:
            trace.record_denied_action(action)  # T021, SC-002: engellenen erişim girişimi
        if run.status is SandboxRunStatus.SUCCESS:
            return run.result_text
        if run.status is SandboxRunStatus.TIMEOUT:
            return "Sandbox çalıştırması zaman aşımına uğradı. Tahmini bir değer üretme."
        if run.status is SandboxRunStatus.DENIED_ACTION:
            return (
                "Sandbox, onaylı Tool Gateway dışında bir hedefe erişmeye çalıştı; "
                "bu ağ seviyesinde (Cilium) engellendi. Tahmini bir değer üretme."
            )
        return "Sandbox çalıştırması başarısız oldu. Tahmini bir değer üretme."

    return run_ptc_code


def _build_tools(trace: Trace, on_ptc_event: Callable[[dict], None] | None = None) -> list:
    """T016 (knowledge_base) + T022 (live_systems) + T014 (PTC sandbox) erişim
    yollarının tool'ları."""
    return [
        knowledge_base.make_kb_tool(trace),
        *live_systems.get_live_system_tools(),
        _make_ptc_tool(trace, on_ptc_event),
    ]


def build_agent(trace: Trace, on_ptc_event: Callable[[dict], None] | None = None):
    """`on_ptc_event` (Faz 4, T006): PTC sandbox'ının canlı olaylarını almak
    isteyen çağıranlar (web arayüzü) için — CLI bunu hiç vermez."""
    tools = _build_tools(trace, on_ptc_event)
    tool_policy.assert_known_tools(tools)  # Principle II savunma hattı (bkz. tool_policy.py)
    return create_agent(
        model=_build_model(),
        tools=tools,
        middleware=[
            tool_policy.build_middleware(),
            live_systems.LiveSystemTraceMiddleware(trace),
        ],
        checkpointer=InMemorySaver(),
    )


async def invoke_and_resolve(agent, message: str, thread_id: str) -> dict:
    """Ajanı çalıştırır; bir interrupt dönerse tool_policy.auto_resolve ile
    (insan olmadan) anında karar verip devam ettirir.

    `ainvoke` (senkron `invoke` DEĞİL) kullanılıyor — bulunan gerçek bir hata
    (2026-08-28, Faz 4'ün web testleriyle ortaya çıktı, Faz 1'den beri
    varmış): `live_systems`'in MCP-adapted tool'ları (langchain_mcp_adapters)
    yalnızca async çağrılabiliyor; `agent.invoke()`'un senkron tool-çalıştırma
    yolu bunlarda `NotImplementedError: StructuredTool does not support sync
    invocation` fırlatıyordu. `ainvoke`, `BaseTool._arun`'un varsayılanı
    sayesinde (`langchain_core/tools/base.py:932` —
    `run_in_executor(None, self._run, ...)`) senkron tool'ları (`run_ptc_code`,
    `search_knowledge_base`) da otomatik bir thread'de çalıştırarak doğru
    ele alıyor — yani her iki tool türü için de doğru olan tek yol bu."""
    config = {"configurable": {"thread_id": thread_id}}
    messages = {"messages": [{"role": "user", "content": message}]}
    result = await agent.ainvoke(messages, config=config)

    while "__interrupt__" in result:
        hitl_request = result["__interrupt__"][0].value
        command = tool_policy.auto_resolve(hitl_request)
        result = await agent.ainvoke(command, config=config)

    return result
