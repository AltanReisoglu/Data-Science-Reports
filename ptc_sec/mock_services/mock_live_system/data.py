"""Sahte destek-talebi verisi (mock canlı sistem sunucusu için)."""

from __future__ import annotations

from datetime import datetime, timezone

FAKE_TICKETS: dict[str, dict] = {
    "TCK-1001": {"status": "open", "last_updated": "2026-08-20T09:00:00Z"},
    "TCK-1002": {"status": "closed", "last_updated": "2026-08-15T14:30:00Z"},
    "TCK-1042": {"status": "open", "last_updated": "2026-08-27T10:15:00Z"},
    "TCK-1077": {"status": "in_progress", "last_updated": "2026-08-26T16:45:00Z"},
}

# Faz 4 (tool sayısını artırma isteği, 2026-08-28) — sahte bir personel dizini.
FAKE_EMPLOYEES: list[dict] = [
    {"name": "Ayşe Yılmaz", "department": "İnsan Kaynakları", "email": "ayse.yilmaz@kurum.example"},
    {"name": "Mehmet Demir", "department": "Bilgi Teknolojileri", "email": "mehmet.demir@kurum.example"},
    {"name": "Zeynep Kaya", "department": "Finans", "email": "zeynep.kaya@kurum.example"},
    {"name": "Ali Şahin", "department": "Bilgi Teknolojileri", "email": "ali.sahin@kurum.example"},
]

_next_ticket_number = 1078  # FAKE_TICKETS'teki en yüksek numaradan (1077) sonrası


def get_ticket(ticket_id: str) -> dict | None:
    ticket = FAKE_TICKETS.get(ticket_id)
    if ticket is None:
        return None
    return {"ticket_id": ticket_id, **ticket}


def count_open_tickets() -> dict:
    open_count = sum(1 for ticket in FAKE_TICKETS.values() if ticket["status"] == "open")
    return {"open_count": open_count, "as_of": datetime.now(timezone.utc).isoformat()}


def create_ticket(title: str, description: str) -> dict:
    """Yeni bir sahte destek talebi oluşturur, durumu her zaman 'open' başlar."""
    global _next_ticket_number
    ticket_id = f"TCK-{_next_ticket_number}"
    _next_ticket_number += 1
    FAKE_TICKETS[ticket_id] = {
        "status": "open",
        "title": title,
        "description": description,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    return {"ticket_id": ticket_id, **FAKE_TICKETS[ticket_id]}


def search_employees(query: str) -> list[dict]:
    """Sahte personel dizininde isim/departmana göre arama yapar."""
    query_lower = query.lower()
    return [
        emp
        for emp in FAKE_EMPLOYEES
        if query_lower in emp["name"].lower() or query_lower in emp["department"].lower()
    ]
