# Contract: Mock Canlı Sistem — MCP Sunucusu

`mock_services/mock_live_system/server.py`, `fastmcp` ile yazılmış, verisi tamamen
sahte olan yerel bir MCP sunucusudur. Ajan buna `langchain-mcp-adapters` üzerinden
(stdio transport) bağlanır. Bu sunucu, spec.md US2'deki "canlı sistemler" senaryosunu
karşılar.

## Tool: `get_ticket_status`

Sahte bir destek-talebi/ticket sisteminin anlık durumunu döndürür.

**Input schema**:

```json
{
  "type": "object",
  "properties": {
    "ticket_id": { "type": "string", "description": "Sorgulanacak ticket kimliği" }
  },
  "required": ["ticket_id"]
}
```

**Output** (string, JSON serileştirilmiş):

```json
{
  "ticket_id": "TCK-1042",
  "status": "open",
  "last_updated": "2026-08-27T10:15:00Z"
}
```

**Hata durumu**: `ticket_id` sahte veri setinde yoksa, tool `{"error": "not_found"}`
döner — ajan bunu FR-011 gereği "erişilemedi/bulunamadı" olarak yansıtmalı, tahmini bir
değer üretmemeli.

## Tool: `list_open_tickets`

Açık ticket'ların (sahte) sayısını/listesini döndürür — spec.md US2 örneğindeki
"şu an açık kritik ticket sayısı kaç?" sorusunu karşılamak için.

**Input schema**: parametre almaz (`{}`).

**Output** (string, JSON serileştirilmiş):

```json
{ "open_count": 7, "as_of": "2026-08-27T10:15:00Z" }
```

## Onaylı-kanal notu

Ajan tarafı (`access_paths/live_systems.py`), bu iki tool dışında **hiçbir** MCP
tool'unu çağırmamalı; hangi tool'ların çağrılabileceği kod içinde sabit bir allowlist
olarak tutulur (Principle II — vendor metadata'sına değil, kendi kodumuza güveniyoruz).
