# Data Model: Web Arayüzü + Canlı PTC İzleme Paneli (Faz 4)

Spec.md'nin Key Entities bölümünde belirtildiği gibi: **bu özellik yeni bir kalıcı veri
modeli TANIMLAMIYOR**. Faz 1/2'nin `Answer`, `Trace`, `SandboxRun`, `LiveToolCall`,
`DeniedAction` dataclass'ları (`src/grounded_assistant/models.py`, `trace.py`) olduğu
gibi kullanılıyor — bu belge, bunların WebSocket üzerinden nasıl SERİLEŞTİRİLDİĞİNİ
(kalıcı depolama değil, bir oturumun ömrü boyunca akan mesajlar) özetliyor. Tam mesaj
şeması için: [contracts/websocket_protocol.md](contracts/websocket_protocol.md).

## Oturum (yeni bir dataclass değil, bir kavram)

Bir WebSocket bağlantısı = bir oturum. Sunucu tarafında, bağlantı süresince şunlar
bellekte tutulur (kalıcı değil, bağlantı kapanınca kaybolur):

| Alan | Kaynak | Açıklama |
|---|---|---|
| `session_id` | yeni üretilir (uuid4) | Faz 1'in `thread_id`'siyle aynı rolde — LangGraph checkpointer'ına geçirilir |
| `trace` | `Trace()` | Faz 1/2'deki gibi, bu oturumun `agent`'ına bağlı |
| `agent` | `graph.build_agent(trace)` | Bağlantı açılınca bir kere kurulur |

## PTC olay akışı → var olan modellerin izdüşümü

| WS mesajı (`ptc_event`, `stage=...`) | Kaynağı olan mevcut model/alan |
|---|---|
| `configmap_created` | (yeni — sandbox_runner'ın kendi adımı, bir model alanı değil) |
| `job_created` (kod içerir) | `SandboxRun.code` |
| `tool_call` | `LiveToolCall` (tool_name, arguments, status, timestamp) |
| `denied_action` | `DeniedAction` (attempted_destination, verdict, observed_at) |
| `final` | `SandboxRun.status`, `SandboxRun.result_text` |

## Nihai yanıt → var olan modelin izdüşümü

| WS mesajı (`answer`) alanı | Kaynağı olan mevcut model/alan |
|---|---|
| `text` | `Answer.text` |
| `grounded` | `Answer.grounded` |
| `source_refs` | `Answer.source_refs` |
| `partial_failure_notes` | `Answer.partial_failure_notes` |
| `access_paths_used` | `Answer.access_paths_used` |
