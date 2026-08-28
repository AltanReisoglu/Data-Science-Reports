# Quickstart: Kurumsal Zemine-Dayalı Asistan (Faz 1)

Bu doküman, Faz 1'in uçtan uca çalıştığını doğrulamak için gereken adımları anlatır.
Uygulama kodu değil, çalıştırma/doğrulama rehberidir.

## Ön koşullar

- Python 3.11+
- `sample_docs/` altına Altan'ın sağlayacağı örnek dokümanlar yerleştirilmiş olmalı
  (bkz. research.md #7 — bu implementasyon başlamadan önce gereken bir bağımlılık)
- Bağımlılıklar kurulu: `langgraph`, `deepagents`, `langchain-mcp-adapters`, `fastmcp`,
  `rank_bm25`, `sentence-transformers`, `typer`

## Kurulum

```bash
pip install -e .
```

## Mock canlı sistemi başlatma

```bash
python -m mock_services.mock_live_system.server
```

Bu, `contracts/mock_live_system_mcp.md`'de tanımlanan `get_ticket_status` ve
`list_open_tickets` tool'larını sunan yerel bir MCP sunucusunu (stdio transport)
ayağa kaldırır.

## Doğrulama senaryoları

### Senaryo 1 — Bilgi bankasından zemine dayalı yanıt (US1, P1)

```bash
assistant ask "Uzaktan çalışma politikamız nedir?"
```

**Beklenen**: Yanıt, `sample_docs/policy/` içindeki ilgili dokümana atıfla gelir;
`Kaynaklar:` satırında en az `kurumsal bilgi bankası` görünür.

### Senaryo 2 — Canlı sistemden güncel veriyle yanıt (US2, P2)

```bash
assistant ask "Şu an açık kritik ticket sayısı kaç?"
```

**Beklenen**: Yanıt, mock MCP sunucusundaki `list_open_tickets` çağrısının sonucunu
zaman damgasıyla birlikte yansıtır.

### Senaryo 3 — Veri bulunamadı (Edge case)

```bash
assistant ask "Marstaki ofisimizin açılış saati nedir?"
```

**Beklenen**: Yanıt açıkça "bu konuda veri bulunamadı" der, hiçbir tahmini bilgi
üretmez (FR-007, SC-002).

### Senaryo 4 — Kısmi kaynak hatası (US1, senaryo 2)

Bir kaynağı geçici olarak erişilemez hale getirip (ör. `sample_docs/wiki/`'i geçici
olarak yeniden adlandırarak) aynı sorguyu tekrar çalıştır.

**Beklenen**: Yanıt, kalan kaynaklardan üretilir; `Kaynaklar:` satırında erişilemeyen
kaynak açıkça belirtilir (FR-010).

## İzlenebilirlik kontrolü (SC-005)

```bash
assistant ask "Uzaktan çalışma politikamız nedir?" --trace
```

**Beklenen**: `--trace` çıktısındaki JSON, yanıttaki her iddianın hangi
`KnowledgeBaseSource`/`LiveToolCall` kaydına karşılık geldiğini gösterir.
