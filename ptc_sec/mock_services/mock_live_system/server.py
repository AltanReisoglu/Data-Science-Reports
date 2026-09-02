"""Mock canlı sistem — sahte destek-talebi MCP sunucusu (contracts/mock_live_system_mcp.md).

Çalıştırma: python -m mock_services.mock_live_system.server
"""

from __future__ import annotations

from fastmcp import FastMCP

from mock_services.mock_live_system.calculator import calculate
from mock_services.mock_live_system.data import (
    count_open_tickets as count_open_tickets_impl,
    create_ticket,
    get_ticket,
    search_employees,
)
from mock_services.mock_live_system.web_search import search_web

mcp = FastMCP("mock-live-system")


@mcp.tool()
def get_ticket_status(ticket_id: str) -> dict:
    """Sahte bir destek-talebi/ticket sisteminin anlık durumunu döndürür."""
    ticket = get_ticket(ticket_id)
    if ticket is None:
        return {"error": "not_found"}
    return ticket


@mcp.tool()
def count_open_tickets() -> dict:
    """Açık ticket'ların (sahte) SAYISINI döndürür — {"open_count": int, "as_of": str}
    şeklinde bir sözlük, ticket LİSTESİ DEĞİL. `len()` ile çağırma; asıl sayı için
    `["open_count"]` alanını kullan."""
    return count_open_tickets_impl()


@mcp.tool()
def create_support_ticket(title: str, description: str) -> dict:
    """Yeni bir sahte destek talebi oluşturur, oluşturulan ticket'ı döndürür."""
    return create_ticket(title, description)


@mcp.tool()
def search_employee_directory(query: str) -> list[dict]:
    """Sahte personel dizininde isim veya departmana göre arama yapar."""
    return search_employees(query)


@mcp.tool()
def web_search(query: str) -> list[dict]:
    """Genel internette gerçek bir web araması yapar (DuckDuckGo), başlık/link/
    özet döndürür. Kurumsal olmayan, güncel/genel bilgi sorularında kullan."""
    return search_web(query)


@mcp.tool()
def calculator(expression: str) -> dict:
    """Bir aritmetik ifadeyi (+ - * / ** % //, parantez) güvenle hesaplar."""
    return calculate(expression)


if __name__ == "__main__":
    mcp.run(transport="stdio")
