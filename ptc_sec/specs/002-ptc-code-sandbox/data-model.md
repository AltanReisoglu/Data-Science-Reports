# Data Model: PTC Kod Sandbox'ı (Faz 2)

Faz 1'in `models.py`'sindeki `LiveToolCall`, `ToolCallStatus` yeniden kullanılır
(bir sandbox'ın yaptığı her tool-proxy çağrısı zaten bir `LiveToolCall`'dır).
Aşağıdakiler Faz 2'ye özgü yeni varlıklardır.

## SandboxRun

Bir PTC çalıştırmasının (bir Kubernetes Job'unun) kaydı.

| Alan | Tip | Açıklama |
|---|---|---|
| `run_id` | str | Benzersiz kimlik; K8s Job adı bundan türetilir (`ptc-sandbox-{run_id}`) |
| `code` | str | LLM'in ürettiği Python kodu (ConfigMap'e yazılan) |
| `started_at` | datetime | Job oluşturma anı |
| `finished_at` | datetime \| None | Job tamamlanma anı (hâlâ çalışıyorsa None) |
| `status` | enum | `running` \| `success` \| `error` \| `timeout` \| `denied_action` |
| `tool_calls` | list[LiveToolCall] | Bu çalıştırma sırasında Tool Gateway'e yapılan çağrılar (Faz 1'deki model) |
| `result_text` | str \| None | Pod log'undan okunan nihai sonuç (başarılıysa) |

**Validation**: `status in {error, timeout, denied_action}` iken `result_text`
`None` olmalı (FR-011 — başarısız bir çalıştırmadan olgusal iddia üretilmez).

## CapabilityGrant

Bir `SandboxRun`'a hangi erişimin tanındığını kaydeder — Faz 1'deki
`tool_policy.KNOWN_TOOLS`'un bu çalıştırmaya özgü izdüşümü.

| Alan | Tip | Açıklama |
|---|---|---|
| `run_id` | str | İlgili SandboxRun |
| `tool_gateway_endpoint` | str | Job'a enjekte edilen Tool Gateway Service ClusterIP:port (ör. `10.96.12.4:8443`) |
| `allowed_tools` | tuple[str, ...] | `search_knowledge_base`, `get_ticket_status`, `list_open_tickets` — Faz 1'in `tool_policy.KNOWN_TOOLS` ile birebir aynı olmalı |

**Validation**: `allowed_tools`, `tool_policy.KNOWN_TOOLS`'tan farklı bir küme
OLAMAZ — farklıysa bu bir konfigürasyon hatasıdır (Principle II ihlali).

## DeniedAction

Sandbox'ın onaylanmamış bir hedefe erişmeye çalıştığı, **Cilium/Hubble'ın ağ
seviyesinde** engellediği bir girişimin kaydı. `LiveToolCall`'dan farkı: bu,
uygulama seviyesinde hiç gerçekleşmedi — sandbox kodu bir `socket`/`connect()`
denedi, paket kernel'de düştü; sandbox kendi tarafında sadece bir bağlantı
hatası (timeout/connection refused) görür. Asıl "reddedildi" kaydı **Hubble'ın
flow log'undan** okunur.

| Alan | Tip | Açıklama |
|---|---|---|
| `run_id` | str | İlgili SandboxRun |
| `attempted_destination` | str | Hedef IP:port veya FQDN (Hubble flow'undan) |
| `verdict` | str | Hubble'ın kendi terimi — `DENIED`/`DROPPED` |
| `observed_at` | datetime | Hubble'ın flow'u gözlemlediği an |

**Not**: Bu tablo, Faz 1'deki `KnowledgeBaseSource`/`LiveToolCall`'ın aksine
uygulama kodumuzun DOĞRUDAN ürettiği bir şey değil — Hubble'ın kendi gözlem
verisinin bizim `Trace`'imize aktarılmış hâlidir (`hubble observe --verdict
DENIED --pod ptc-sandbox-*` ile sorgulanabilir, bkz. quickstart.md).

## State / akış özeti

```text
SandboxRun oluşturulur (running)
  │
  ├─▶ Tool Gateway'e izinli çağrı(lar) ─▶ LiveToolCall (success/error/timeout)
  │
  ├─▶ Onaysız bir hedefe deneme        ─▶ DeniedAction (Hubble'dan)
  │                                        └─▶ status = denied_action
  │
  ├─▶ Zaman aşımı (activeDeadlineSeconds) ─▶ status = timeout
  │
  └─▶ Kod hatasız bitti                 ─▶ status = success, result_text dolu
```
