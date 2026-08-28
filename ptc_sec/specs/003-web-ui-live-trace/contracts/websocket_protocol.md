# Contract: WebSocket Protokolü (`/ws`)

Tek, çift yönlü WebSocket bağlantısı (research.md §3). Bağlantı açılınca sunucu bir
`session_id` üretir; bağlantı kapanana kadar bu oturumun `Trace`'i/agent'ı yaşar.

## İstemci → Sunucu

Kullanıcı bir soru gönderdiğinde, TEK mesaj tipi:

```json
{"type": "question", "text": "Uzaktan çalışma politikamız nedir?"}
```

## Sunucu → İstemci

Üç mesaj tipi, aynı bağlantıdan, sırayla akar:

### 1. `ptc_event` — sol-alt panel için (sıfır ya da daha fazla, bir soru PTC tetiklemezse hiç gelmez)

```json
{"type": "ptc_event", "run_id": "4ccb3e75801d", "stage": "configmap_created", "timestamp": "..."}
{"type": "ptc_event", "run_id": "4ccb3e75801d", "stage": "job_created", "code": "tickets = list_open_tickets()\nset_result(...)", "timestamp": "..."}
{"type": "ptc_event", "run_id": "4ccb3e75801d", "stage": "tool_call", "tool_name": "list_open_tickets", "arguments": {}, "status": "success", "timestamp": "..."}
{"type": "ptc_event", "run_id": "4ccb3e75801d", "stage": "denied_action", "attempted_destination": "8.8.8.8:443", "verdict": "DROPPED", "timestamp": "..."}
{"type": "ptc_event", "run_id": "4ccb3e75801d", "stage": "final", "status": "success", "result_text": "...", "timestamp": "..."}
```

`stage` sırası daima: `configmap_created` → `job_created` → (0'dan fazla) `tool_call` →
(0'dan fazla) `denied_action` → `final`. `denied_action` mesajları, `final`'dan HEMEN
ÖNCE, toplu gelir (research.md §4 — Hubble sorgusu Job bittikten sonra çalışıyor,
bilinçli bir sadeleştirme; diğer tüm `stage`'ler gerçek zamanlı).

### 2. `answer` — ana yanıt alanı için (her soru için tam olarak bir kere, PTC kullanılsın kullanılmasın)

```json
{
  "type": "answer",
  "text": "...",
  "grounded": true,
  "source_refs": ["list_open_tickets"],
  "partial_failure_notes": [],
  "access_paths_used": ["ptc_sandbox", "live_system"]
}
```

Soru PTC kullanmadan yanıtlanırsa: hiç `ptc_event` gelmez, sadece bir `answer` mesajı
gelir — istemci bunu FR-006 gereği "bu soru için sandbox kullanılmadı" olarak
göstermelidir (`ptc_event` hiç alınmamışsa).

### 3. `error` — beklenmeyen bir sunucu hatası olursa

```json
{"type": "error", "message": "..."}
```

## Sözleşme notu

Bu mesajlar, `data-model.md`'nin belirttiği gibi, mevcut `SandboxRun`/`LiveToolCall`/
`DeniedAction`/`Answer` dataclass'larının doğrudan alan-alan yansımasıdır — sunucu
tarafında ayrı bir "web DTO" katmanı icat edilmiyor (Principle V).
