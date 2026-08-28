"""Sandbox entrypoint — LLM'in ürettiği kodu çalıştırır (Faz 2).

Bilerek Python-seviyesinde bir kısıtlama (RestrictedPython, builtins filtresi
vb.) YOK — research.md §4.3'teki karar: enforcement Cilium'da (network
seviyesinde), burada değil. Kod istediği kütüphaneyi import edebilir, ama
Tool Gateway dışında hiçbir yere çıkamaz (Cilium bunu kernel'de engeller).

Kontrat: contracts/sandbox_job_contract.md
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

from fastmcp import Client

TOOL_GATEWAY_ENDPOINT = os.environ["TOOL_GATEWAY_ENDPOINT"]
CODE_PATH = "/sandbox/code.py"

# Faz 1'in tool_policy.ALLOWED_TOOLS + LOCAL_TOOLS ile birebir aynı olmalı
# (CapabilityGrant.allowed_tools, data-model.md). Faz 4'te 2 yeni tool eklendi.
ALLOWED_TOOLS = (
    "search_knowledge_base",
    "get_ticket_status",
    "list_open_tickets",
    "create_support_ticket",
    "search_employee_directory",
)

# LLM'in ürettiği kod, tool'ları normal bir Python fonksiyonu gibi pozisyonel
# çağırabilir (ör. `search_knowledge_base("vpn erisim")`) — MCP'nin kendisi
# yalnızca adlandırılmış argüman kabul ettiği için, pozisyonel argümanları
# isimlere çevirmek amacıyla bu sabit eşleme gerekiyor (contracts/tool_gateway_mcp.md'deki
# tool imzalarıyla birebir).
_ARG_NAMES: dict[str, tuple[str, ...]] = {
    "search_knowledge_base": ("query",),
    "get_ticket_status": ("ticket_id",),
    "list_open_tickets": (),
    "create_support_ticket": ("title", "description"),
    "search_employee_directory": ("query",),
}


def _make_sync_tool(tool_name: str):
    """Sandbox kodunun senkron çağırabileceği bir tool-proxy fonksiyonu üretir.
    Gerçek iş fastmcp.Client ile Tool Gateway'e (Cilium'un izin verdiği TEK
    hedef) yapılan bir HTTP çağrısıdır.

    Her çağrı, nihai sonuç satırından ÖNCE ayrı bir JSON satırı olarak stdout'a
    da yazılır (`"type": "tool_call"`) — ana asistan (sandbox_runner.py, T015)
    bunu `Trace.record_tool_call`'a besler (FR-008). Kontratın orijinal nihai
    satırında (`sandbox_job_contract.md`) `type` alanı YOK — bu, iki satır türünü
    ayırt etmenin yolu."""

    def _call(*args, **kwargs):
        named_from_args = dict(zip(_ARG_NAMES.get(tool_name, ()), args))
        kwargs = {**named_from_args, **kwargs}

        async def _do():
            async with Client(TOOL_GATEWAY_ENDPOINT) as client:
                result = await client.call_tool(tool_name, kwargs)
                return result.data if hasattr(result, "data") else str(result)

        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            value = asyncio.run(_do())
        except Exception:
            print(
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool": tool_name,
                        "args": kwargs,
                        "status": "error",
                        "timestamp": timestamp,
                    }
                )
            )
            raise
        print(
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": tool_name,
                    "args": kwargs,
                    "status": "success",
                    "timestamp": timestamp,
                }
            )
        )
        return value

    return _call


def main() -> None:
    with open(CODE_PATH, encoding="utf-8") as f:
        code = f.read()

    result_holder: dict = {}

    def set_result(value) -> None:
        """Sandbox kodu, nihai sonucunu bununla bildirir (research.md kontratı)."""
        result_holder["value"] = value

    sandbox_globals: dict = {
        "set_result": set_result,
        **{name: _make_sync_tool(name) for name in ALLOWED_TOOLS},
    }

    try:
        exec(compile(code, CODE_PATH, "exec"), sandbox_globals)  # noqa: S102
    except Exception as exc:  # noqa: BLE001 - sandbox kodunun hatası, çökmeden bildirilmeli
        print(json.dumps({"status": "error", "message": str(exc)}))
        sys.exit(0)

    if "value" in result_holder:
        print(json.dumps({"status": "success", "result": result_holder["value"]}))
    else:
        print(json.dumps({"status": "error", "message": "kod set_result() çağırmadı"}))


if __name__ == "__main__":
    main()
