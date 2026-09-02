"""Tool Gateway — FastMCP HTTP transport (Faz 2).

Sandbox pod'unun (Cilium ile) erişebildiği TEK ağ hedefi. Faz 1'in
knowledge_base.py ve mock_live_system/data.py mantığını in-process sarar —
ayrı bir "mock MCP sunucusu" pod'una ihtiyaç yok (research.md #3).

Çalıştırma: python -m mock_services.tool_gateway.server
"""

from __future__ import annotations

import requests
from fastmcp import FastMCP

from grounded_assistant.access_paths.knowledge_base import query_knowledge_base
from mock_services.mock_live_system.calculator import calculate
from mock_services.mock_live_system.data import (
    count_open_tickets as count_open_tickets_impl,
    create_ticket,
    get_ticket,
    search_employees,
)
from mock_services.mock_live_system.web_search import search_web

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


@mcp.tool()
def resolve_dns(hostname: str) -> dict:
    """Verilen hostname'i DNS ile çözer — TCP BAĞLANTISI DENEMEZ, yalnızca isim
    çözümlemesi yapar. Tool Gateway'in DNS'i serbesttir (herhangi bir isim
    sorgulanabilir); bu tool'un başarılı olması o hedefe BAĞLANILABİLECEĞİ
    anlamına GELMEZ — yalnızca ismin çözülebildiğini gösterir (DNS ile TCP
    ayrı kısıtlanmıştır, bkz. tool-gateway-egress policy'si)."""
    import socket

    try:
        ip = socket.gethostbyname(hostname)
        return {"resolved": True, "ip": ip}
    except socket.gaierror as e:
        return {"resolved": False, "error": str(e)}


@mcp.tool()
def check_connectivity(host: str, port: int) -> dict:
    """Verilen host:port'a HAM bir TCP bağlantısı kurmayı dener (HTTP DEĞİL,
    yalnızca soket seviyesi el sıkışması) — önce DNS çözer, sonra bağlanmayı
    dener, hangi aşamada durduğunu açıkça bildirir. Onaylı olmayan bir hedefe
    bağlantı ağ seviyesinde (Cilium) engellenir."""
    import socket

    try:
        ip = socket.gethostbyname(host)
    except socket.gaierror as e:
        return {"stage": "dns", "success": False, "detail": f"DNS çözümü başarısız: {e}"}
    try:
        sock = socket.create_connection((host, port), timeout=5)
        sock.close()
        return {"stage": "tcp", "success": True, "resolved_ip": ip}
    except OSError as e:
        return {"stage": "tcp", "success": False, "resolved_ip": ip, "detail": str(e)}


@mcp.tool()
def fetch_url(url: str, method: str = "GET") -> dict:
    """Verilen URL'ye Tool Gateway'in KENDİSİNDEN bir HTTP isteği atar — hedefi
    çağıranın belirlediği ham bir ağ çağrısıdır. AŞAMA AŞAMA teşhis döner:
    DNS çözülemezse "dns" aşamasında, TCP bağlantısı reddedilirse "tcp_connect"
    aşamasında, zaman aşımına uğrarsa "timeout" aşamasında durur — hangi
    seviyede engellendiği (isim çözme mi, bağlantı mı) AÇIKÇA görünür. Onaylı
    olmayan (tool-gateway-egress policy'sindeki 3 FQDN dışında) bir hedefe
    gidilirse bu istek ağ seviyesinde (Cilium/eBPF) engellenir — bu tool'un
    varlığı PoC'nin egress-policy sınırını canlı göstermek içindir (Altan'ın
    kararı, 2026-08-31)."""
    from urllib.parse import urlparse

    hostname = urlparse(url).hostname
    if hostname is None:
        return {"stage": "input", "success": False, "detail": "Geçersiz URL"}

    import socket

    try:
        resolved_ip = socket.gethostbyname(hostname)
    except socket.gaierror as e:
        return {"stage": "dns", "success": False, "detail": f"DNS çözümü başarısız: {e}"}

    try:
        response = requests.request(method, url, timeout=5)
        return {
            "stage": "complete",
            "success": True,
            "resolved_ip": resolved_ip,
            "status_code": response.status_code,
            "content_preview": response.text[:200],
        }
    except requests.exceptions.Timeout:
        return {
            "stage": "timeout",
            "success": False,
            "resolved_ip": resolved_ip,
            "detail": "Bağlantı zaman aşımına uğradı (muhtemelen ağ seviyesinde engellendi)",
        }
    except requests.exceptions.ConnectionError as e:
        return {
            "stage": "tcp_connect",
            "success": False,
            "resolved_ip": resolved_ip,
            "detail": f"DNS çözüldü ({resolved_ip}) ama TCP bağlantısı kurulamadı: {e}",
        }
    except Exception as e:  # noqa: BLE001 - beklenmeyen hata, çökmeden bildirilmeli
        return {"stage": "unknown", "success": False, "detail": str(e)}


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8443)
