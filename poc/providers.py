"""
providers.py — tool sağlayıcıları (ajan hangi tool setini kullanacak).

Aynı arayüz: SCHEMAS, SYSTEM, tool_type(name), resource_of(name,args), run(name,args).
Böylece agent.py / demo_server / langgraph_agent tek satırla generic ↔ product geçer;
13 stratejinin hiçbiri değişmez.

  generic  → 6 jenerik mock tool (terminal/read/web/snapshot/grep/write) — offline, tip-çeşitli
  product  → GERÇEK 119 ürün tool'u (Jira/NETA/LDAP/Confluence/doküman/analiz) — canlı LLM ile
"""
from __future__ import annotations

import tool_schemas as TS
import tools_product as TP
from harness import MockTools

GENERIC_SYSTEM = (
    "Sen tool kullanan bir yazılım asistanısın. Kullanıcının isteğini yerine getirmek "
    "için gereken tool'ları çağır (run_terminal, read_file, web_extract, take_snapshot, "
    "grep, write_file). Gereksiz tekrar çağırma. Yeterince bilgi toplayınca tool çağırmayı "
    "bırak ve kısa, net bir Türkçe yanıt yaz."
)


class GenericProvider:
    name = "generic"
    SYSTEM = GENERIC_SYSTEM

    def __init__(self) -> None:
        self.SCHEMAS = TS.SCHEMAS
        self._tools = MockTools()

    def tool_type(self, n: str) -> str:
        return TS.tool_type(n)

    def resource_of(self, n: str, a: dict) -> str:
        return TS.resource_of(n, a)

    def run(self, n: str, a: dict) -> str:
        return TS.run(self._tools, n, a)


class ProductProvider:
    name = "product"
    SYSTEM = TP.PRODUCT_SYSTEM

    def __init__(self) -> None:
        self.SCHEMAS = TP.SCHEMAS

    def tool_type(self, n: str) -> str:
        return TP.tool_type(n)

    def resource_of(self, n: str, a: dict) -> str:
        return TP.resource_of(n, a)

    def run(self, n: str, a: dict) -> str:
        return TP.run(n, a)


def get_provider(name: str):
    return ProductProvider() if name == "product" else GenericProvider()
