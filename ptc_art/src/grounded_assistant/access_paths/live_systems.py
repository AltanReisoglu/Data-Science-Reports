"""MCP client bağlantısı: mock canlı sisteme (T020/T021).

`langchain-mcp-adapters` ile `mock_services/mock_live_system/server.py`'ye (stdio
transport) bağlanır. Onaylı-kanal (Principle II): yalnızca
`tool_policy.ALLOWED_TOOLS`'taki tool'lar expose edilir (Faz 4'te 2 yeni tool
eklendi: `create_support_ticket`, `search_employee_directory`) — closed-world,
savunma amaçlı çift kontrol (bkz. research.md #4).

Not: `MultiServerMCPClient.get_tools()` async'tir; burada `asyncio.run` ile
senkron ajan kurulumuna köprüleniyor (`get_live_system_tools`, çağıranın zaten
bir event loop içindeyse `asyncio.to_thread` ile sarması gerekir — bkz.
web/app.py). Gerçek LLM ile uçtan uca doğrulandı (2026-08-28, Faz 4) — bu,
MCP tool'larının `agent.ainvoke()` ile (senkron `invoke()` değil) çağrılması
gerektiğini ortaya çıkardı (bkz. graph.py, `awrap_tool_call` aşağıda).
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from grounded_assistant.agent.tool_policy import ALLOWED_TOOLS
from grounded_assistant.models import LiveToolCall, ToolCallStatus
from grounded_assistant.trace import Trace

_SERVER_CONFIG = {
    "mock_live_system": {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "mock_services.mock_live_system.server"],
    }
}


async def _get_tools() -> list:
    client = MultiServerMCPClient(_SERVER_CONFIG)
    tools = await client.get_tools()
    return [t for t in tools if t.name in ALLOWED_TOOLS]


def get_live_system_tools() -> list:
    """Senkron sarmalayıcı — agent kurulumu sırasında bir kere çağrılır."""
    return asyncio.run(_get_tools())


def _record_failure(
    trace: Trace,
    tool_name: str,
    arguments: dict,
    timestamp: datetime,
    tool_call_id: str,
    status: ToolCallStatus,
    message: str,
) -> ToolMessage:
    trace.record_tool_call(
        LiveToolCall(tool_name=tool_name, arguments=arguments, timestamp=timestamp, status=status)
    )
    return ToolMessage(content=message, name=tool_name, tool_call_id=tool_call_id, status="error")


def _record_success(trace: Trace, tool_name: str, arguments: dict, timestamp: datetime, result):
    failed = getattr(result, "status", "success") == "error"
    status = ToolCallStatus.ERROR if failed else ToolCallStatus.SUCCESS
    trace.record_tool_call(
        LiveToolCall(
            tool_name=tool_name,
            arguments=arguments,
            timestamp=timestamp,
            status=status,
            result=None if failed else str(getattr(result, "content", result)),
        )
    )
    return result


class LiveSystemTraceMiddleware(AgentMiddleware):
    """Her canlı sistem tool çağrısını Trace'e (FR-009) kaydeder ve
    zaman aşımı/hata durumunda (FR-011) tahmini değer üretilmeden, açık bir
    hata mesajıyla modele geri döner. Onay/red kararı burada değil,
    `tool_policy.py`'de (mekanizma/politika ayrımı, DSH ilhamlı).

    Hem `wrap_tool_call` (senkron) HEM `awrap_tool_call` (async) uygulanıyor —
    bulunan gerçek bir hata (Faz 4, 2026-08-28): `agent.ainvoke()` kullanılınca
    (graph.py'deki düzeltme — MCP tool'ları için gerekli) LangGraph, middleware'in
    ASYNC karşılığını arıyor; sadece sync'i uygulamak açık bir hata fırlatıyordu
    ("Asynchronous implementation of awrap_tool_call is not available")."""

    def __init__(self, trace: Trace) -> None:
        super().__init__()
        self._trace = trace

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in ALLOWED_TOOLS:
            return handler(request)

        timestamp = datetime.now(UTC)
        tool_name = request.tool_call["name"]
        arguments = request.tool_call["args"]
        tool_call_id = request.tool_call["id"]

        try:
            result = handler(request)
        except TimeoutError:
            return _record_failure(
                self._trace, tool_name, arguments, timestamp, tool_call_id,
                ToolCallStatus.TIMEOUT,
                f"'{tool_name}' canlı sistemine erişilemedi (zaman aşımı). "
                "Tahmini bir değer üretme.",
            )
        except Exception:
            return _record_failure(
                self._trace, tool_name, arguments, timestamp, tool_call_id,
                ToolCallStatus.ERROR,
                f"'{tool_name}' canlı sistemine erişilemedi. Tahmini bir değer üretme.",
            )
        return _record_success(self._trace, tool_name, arguments, timestamp, result)

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in ALLOWED_TOOLS:
            return await handler(request)

        timestamp = datetime.now(UTC)
        tool_name = request.tool_call["name"]
        arguments = request.tool_call["args"]
        tool_call_id = request.tool_call["id"]

        try:
            result = await handler(request)
        except TimeoutError:
            return _record_failure(
                self._trace, tool_name, arguments, timestamp, tool_call_id,
                ToolCallStatus.TIMEOUT,
                f"'{tool_name}' canlı sistemine erişilemedi (zaman aşımı). "
                "Tahmini bir değer üretme.",
            )
        except Exception:
            return _record_failure(
                self._trace, tool_name, arguments, timestamp, tool_call_id,
                ToolCallStatus.ERROR,
                f"'{tool_name}' canlı sistemine erişilemedi. Tahmini bir değer üretme.",
            )
        return _record_success(self._trace, tool_name, arguments, timestamp, result)
