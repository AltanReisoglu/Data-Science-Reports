# Contract: Tool Gateway — MCP over HTTP

Faz 1'deki `contracts/mock_live_system_mcp.md`'nin (stdio) yerini alan, cluster-içi
**HTTP transport'lu** hâli. Aynı 3 tool, artık aynı pod'dan (in-process) sunuluyor:

| Tool | Kaynak (Faz 1'den yeniden kullanılan) |
|---|---|
| `search_knowledge_base` | `knowledge_base.py` (4 kaynak + Hybrid Search + RRF) |
| `get_ticket_status` | `mock_services/mock_live_system/data.py` (in-process, artık ayrı pod değil) |
| `list_open_tickets` | aynı |

**Endpoint**: `http://tool-gateway.default.svc.cluster.local:8443/mcp`
(FastMCP'nin HTTP transport'u — `fastmcp.FastMCP(...).run(transport="http", port=8443)`)
— bu, servisin genel/DNS adıdır. **Sandbox'a özel not**: `sandbox_runner.py`,
`sandbox-egress` policy'sinin (research.md §4.1) DNS gerektirmemesi için bu
adı DEĞİL, Service'in ClusterIP'sini (`read_namespaced_service` ile okunup)
doğrudan enjekte eder — sandbox hiçbir zaman DNS sorgusu yapmaz.

**Onaylı-kanal notu**: Tool Gateway, `tool_policy.KNOWN_TOOLS` dışında hiçbir
tool expose etmez — bu, Faz 1'deki `assert_known_tools` kontrolünün servis
tarafındaki karşılığıdır.

**Kimlik doğrulama**: Bu PoC'de yok (sadece Cilium'un pod-identity bazlı ağ
kısıtlaması güveniliyor) — üretim bir sistemde ayrıca mTLS/token gerekirdi,
ama Altan'ın "asıl odak Cilium" önceliğine göre bu faz için kapsam dışı
bırakıldı (Principle V).
