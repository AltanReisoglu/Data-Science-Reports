"""Tool Gateway — FastMCP HTTP transport (Faz 2).

Sandbox pod'unun (Cilium ile) erişebildiği TEK ağ hedefi. Faz 1'in
knowledge_base.py ve mock_live_system/data.py mantığını in-process sarar —
ayrı bir "mock MCP sunucusu" pod'una ihtiyaç yok (research.md #3).

Çalıştırma: python -m mock_services.tool_gateway.server
"""

from __future__ import annotations

from fastmcp import FastMCP

from grounded_assistant.access_paths.knowledge_base import query_knowledge_base
from mock_services.mock_live_system.data import (
    count_open_tickets,
    create_ticket,
    get_ticket,
    search_employees,
)

mcp = FastMCP("tool-gateway")


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """Kurumsal bilgi bankasını (politika, kurumsal wiki, destek talebi arşivi,
    teknik dokümantasyon) sorgula. Kullanıcının sorusuyla ilgili kurumsal bilgi
    gerektiğinde bunu çağır."""
    sources = query_knowledge_base(query)
    lines = []
    for source in sources:
        if source.status.value == "ok":
            snippets = "; ".join(hit.snippet for hit in source.hits[:3])
            lines.append(f"[{source.source_id.value}] {snippets}")
        elif source.status.value == "empty":
            lines.append(f"[{source.source_id.value}] sonuç yok")
        else:
            lines.append(f"[{source.source_id.value}] hata: kaynağa erişilemedi")
    return "\n".join(lines) if lines else "Hiçbir kaynaktan sonuç bulunamadı."


@mcp.tool()
def get_ticket_status(ticket_id: str) -> dict:
    """Sahte bir destek-talebi/ticket sisteminin anlık durumunu döndürür."""
    ticket = get_ticket(ticket_id)
    if ticket is None:
        return {"error": "not_found"}
    return ticket


@mcp.tool()
def list_open_tickets() -> dict:
    """Açık ticket'ların (sahte) sayısını döndürür."""
    return count_open_tickets()


@mcp.tool()
def create_support_ticket(title: str, description: str) -> dict:
    """Yeni bir sahte destek talebi oluşturur, oluşturulan ticket'ı döndürür."""
    return create_ticket(title, description)


@mcp.tool()
def search_employee_directory(query: str) -> list[dict]:
    """Sahte personel dizininde isim veya departmana göre arama yapar."""
    return search_employees(query)


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8443)
