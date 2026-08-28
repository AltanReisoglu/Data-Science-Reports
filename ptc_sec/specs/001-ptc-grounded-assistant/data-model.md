# Data Model: Kurumsal Zemine-Dayalı Asistan (Faz 1)

Spec.md → Key Entities bölümünün Faz 1 kapsamındaki somutlaştırması. Hafıza (Memory
Record) bu fazda implemente edilmediği için buraya dahil edilmemiştir.

## Query

Kullanıcının CLI üzerinden sorduğu soru.

| Alan | Tip | Açıklama |
|---|---|---|
| `text` | str | Sorunun metni |
| `session_id` | str | Tek CLI çalıştırması/oturumu kimliği |
| `timestamp` | datetime | Sorunun sorulma anı |

## KnowledgeBaseSource

Kurumsal bilgi bankasını oluşturan 4 sabit kaynaktan her biri (spec.md FR-002/003).

| Alan | Tip | Açıklama |
|---|---|---|
| `source_id` | enum | `policy` \| `wiki` \| `support_tickets` \| `technical_docs` |
| `status` | enum | `ok` \| `empty` \| `error` — bu sorgu için erişim/sonuç durumu |
| `hits` | list[RetrievalHit] | BM25+dense+RRF sonrası bu kaynaktan gelen sıralı sonuçlar |

### RetrievalHit

| Alan | Tip | Açıklama |
|---|---|---|
| `doc_id` | str | Kaynak doküman kimliği |
| `snippet` | str | İlgili metin parçası |
| `bm25_rank` | int \| None | BM25 sıralamasındaki yeri (yoksa None) |
| `dense_rank` | int \| None | Dense embedding sıralamasındaki yeri (yoksa None) |
| `rrf_score` | float | RRF ile birleştirilmiş nihai skor |

**Validation**: `status == "error"` olan bir kaynağın `hits` listesi boş olmalı (FR-010 —
kısmi hata durumunda yanıt kalan kaynaklarla üretilir, hata açıkça belirtilir).

## LiveToolCall

Mock canlı sisteme (MCP üzerinden) yapılan bir çağrının kaydı.

| Alan | Tip | Açıklama |
|---|---|---|
| `tool_name` | str | Çağrılan MCP tool'unun adı |
| `arguments` | dict | Çağrıya giden parametreler |
| `timestamp` | datetime | Çağrı anı |
| `status` | enum | `success` \| `timeout` \| `error` \| `unavailable` (DSH ilhamlı — cevaplayıcı/politika hiç yanıt vermediğinde) |
| `result` | str \| None | Tool'dan dönen ham sonuç (başarılıysa) |

**Validation**: `status != "success"` ise `result` `None` olmalı; yanıt üretiminde bu
durum FR-011 gereği açıkça kullanıcıya yansıtılmalı (erişilemedi, tahmini değer yok).

## Answer

Kullanıcıya sunulan nihai yanıt (spec.md Key Entities → Yanıt).

| Alan | Tip | Açıklama |
|---|---|---|
| `text` | str | Kullanıcıya gösterilen metin |
| `grounded` | bool | En az bir erişim yolundan gerçek veri kullanıldı mı |
| `access_paths_used` | list[enum] | `knowledge_base` \| `live_system` (bu fazda `memory` yok) |
| `source_refs` | list[str] | Katkı sağlayan `KnowledgeBaseSource.source_id` ve/veya `LiveToolCall.tool_name` referansları (FR-009 izlenebilirlik) |
| `partial_failure_notes` | list[str] | Erişilemeyen/boş dönen kaynaklara dair açık notlar (FR-010) |

**Validation**:
- `grounded == False` ⇒ `text`, hiçbir olgusal iddia içermemeli, yalnızca "veri
  bulunamadı" açıklamasını taşımalı (FR-007, SC-002).
- `grounded == True` ⇒ `source_refs` boş olamaz (SC-001 — her iddia bir kaynağa
  izlenebilir olmalı).

## State / akış özeti

```text
Query
  ├─▶ KnowledgeBaseSource (x4, paralel) ─┐
  ├─▶ LiveToolCall (MCP, gerekirse)      ├─▶ RRF/birleştirme ─▶ Answer
  └─▶ (Memory — Faz 1'de yok)           ─┘
```

Hiçbir dal veri döndürmezse: `Answer.grounded = False`, `text` açık bir "bulunamadı"
mesajı (FR-007).
