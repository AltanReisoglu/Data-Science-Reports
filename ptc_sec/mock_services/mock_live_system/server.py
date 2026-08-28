"""Mock canlı sistem — sahte destek-talebi MCP sunucusu (contracts/mock_live_system_mcp.md).

Çalıştırma: python -m mock_services.mock_live_system.server
"""

from __future__ import annotations

from fastmcp import FastMCP

from mock_services.mock_live_system.data import (
    count_open_tickets,
    create_ticket,
    get_ticket,
    search_employees,
)

mcp = FastMCP("mock-live-system")


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
    mcp.run(transport="stdio")
