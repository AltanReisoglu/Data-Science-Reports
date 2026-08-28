"""Tool Gateway: HumanInTheLoopMiddleware policy (Principle II).

Fail-closed güvence middleware'in kendisinden DEĞİL, yalnızca bu iki tool'un
`tools=[...]`'a kayıtlı olmasından (closed-world) geliyor — middleware, kendi
`interrupt_on` sözlüğünde olmayan bir tool'u otomatik onaylar (bkz.
specs/001-ptc-grounded-assistant/research.md #4 — langchain-ai/langchain kaynak
kod doğrulaması, 2026-08-27).
"""

from __future__ import annotations

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langgraph.types import Command

# contracts/mock_live_system_mcp.md — canlı sistem tool'ları (Faz 4'te 2 yeni
# tool eklendi: create_support_ticket — durum DEĞİŞTİRDİĞİ için özellikle
# onay-gerektiren bir aday; search_employee_directory — salt-okunur).
ALLOWED_TOOLS = (
    "get_ticket_status",
    "list_open_tickets",
    "create_support_ticket",
    "search_employee_directory",
)

# Onaylı-kanal/gateway gerektirmeyen, yerel (dış sisteme çıkmayan) tool'lar.
# `run_ptc_code` (Faz 2, T014) burada — kendisi Tool Gateway'e çıkmıyor, bunun
# yerine ayrı bir Kubernetes pod'unu tetikliyor; asıl kısıtlama bu middleware'de
# değil, o pod'un CiliumNetworkPolicy'sinde (research.md §4.1).
LOCAL_TOOLS = ("search_knowledge_base", "run_ptc_code")

# Agent'a eklenmesine izin verilen TÜM tool'lar — assert_known_tools bununla
# karşılaştırır (bkz. graph.py). Yeni bir tool buraya (ya ALLOWED_TOOLS ya
# LOCAL_TOOLS'a) eklenmeden agent'a bağlanırsa build_agent hata verir; middleware'in
# fail-open varsayılanına (research.md #4) sessizce güvenmek yerine kendi
# fail-closed kontrolümüzü uyguluyoruz.
KNOWN_TOOLS = ALLOWED_TOOLS + LOCAL_TOOLS


def assert_known_tools(tools: list) -> None:
    """Principle II savunma hattı: her tool ya ALLOWED_TOOLS'ta (onaylı-kanal
    politikalı) ya da LOCAL_TOOLS'ta (yerel, gateway gerektirmeyen) açıkça
    tanımlı olmalı."""
    unknown = [t.name for t in tools if t.name not in KNOWN_TOOLS]
    if unknown:
        msg = (
            f"Bilinmeyen tool(lar) agent'a eklenmeye çalışıldı: {unknown}. "
            "tool_policy.ALLOWED_TOOLS veya tool_policy.LOCAL_TOOLS'a ekleyin "
            "(middleware'in fail-open varsayılanına sessizce güvenilmiyor)."
        )
        raise RuntimeError(msg)


def build_middleware() -> HumanInTheLoopMiddleware:
    """İki bilinen tool için: her çağrıda dur (when her zaman True), yalnızca
    approve/reject kararlarını kabul et (edit/respond gerekmiyor, bu iki tool
    salt-okunur sahte veri döndürüyor)."""
    interrupt_on: dict[str, InterruptOnConfig] = {
        tool_name: InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            when=lambda _request: True,
        )
        for tool_name in ALLOWED_TOOLS
    }
    return HumanInTheLoopMiddleware(interrupt_on=interrupt_on)


def auto_resolve(hitl_request: dict) -> Command:
    """İnsan yok — otomatik karar vericimiz. `hitl_request["action_requests"]`
    içindeki her bekleyen eylem için: ALLOWED_TOOLS'taysa approve, değilse
    (teorik olarak closed-world nedeniyle asla olmamalı) reject (savunma amaçlı
    fail-closed)."""
    decisions = [
        {"type": "approve"}
        if action["name"] in ALLOWED_TOOLS
        else {"type": "reject", "message": "tool onaylı değil"}
        for action in hitl_request["action_requests"]
    ]
    return Command(resume={"decisions": decisions})
