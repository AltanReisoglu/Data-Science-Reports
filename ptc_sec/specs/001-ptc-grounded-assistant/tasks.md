---

description: "Task list for Kurumsal Zemine-Dayalı Asistan — Faz 1"
---

# Tasks: Kurumsal Zemine-Dayalı Asistan (Faz 1)

**Input**: Design documents from `/specs/001-ptc-grounded-assistant/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Spec.md testleri açıkça istemiyor; bu doküman test-yazma görevleri içermez.
Doğrulama, `quickstart.md` senaryolarının uçtan uca çalıştırılmasıyla yapılır.

**Organization**: Görevler spec.md'deki kullanıcı hikayelerine göre gruplanmıştır.
**US3 (Hafıza) bu fazda kapsam dışıdır** — Altan'ın açık kararıyla ertelendi, aşağıda
hiçbir görev üretilmedi.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Farklı dosyalarda, bağımsız çalıştırılabilir
- **[Story]**: US1 veya US2 (US3 bu fazda yok)

## Path Conventions

Tek proje (CLI) — `src/grounded_assistant/`, `mock_services/`, `sample_docs/`, `tests/`
(bkz. plan.md → Project Structure).

---

## Phase 1: Setup

**Purpose**: Proje iskeleti ve bağımlılıklar

- [x] T001 `src/grounded_assistant/`, `mock_services/mock_live_system/`, `sample_docs/{policy,wiki,support_tickets,technical_docs}/`, `tests/{unit,integration}/` dizin iskeletini plan.md'deki yapıya göre oluştur
- [x] T002 `pyproject.toml` ile Python 3.11+ projesini başlat; bağımlılıklar: `langgraph`, `langchain`, `langchain-openai`, `langchain-mcp-adapters`, `fastmcp`, `rank_bm25`, `numpy`, `typer`, `pytest`
- [x] T003 [P] Lint/format aracı yapılandır (ör. `ruff`) — `pyproject.toml` içinde
- [x] T004 [P] `sample_docs/` altına her 4 kaynak için bir `README.md` yerleştir; **⚠️ bağımlılık**: gerçek örnek dokümanları Altan sağlayacak (research.md #7) — bu görevler gelene kadar T018 sentetik/placeholder veriyle ilerleyebilir ama tam doğrulama bekler

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: US1 ve US2'nin ikisinin de üzerine kurulduğu ortak altyapı

**⚠️ CRITICAL**: Bu faz bitmeden hiçbir user story görevine başlanmaz

- [x] T005 [P] `Query`, `KnowledgeBaseSource`, `RetrievalHit`, `LiveToolCall`, `Answer` veri sınıflarını data-model.md'ye göre `src/grounded_assistant/models.py` içinde uygula
- [x] T006 [P] Reciprocal Rank Fusion (RRF, k=60) fonksiyonunu `src/grounded_assistant/retrieval/fusion.py` içinde uygula (research.md #2)
- [x] T007 [P] İzlenebilirlik kaydediciyi (`Answer.source_refs`, kaynak/erişim-yolu logu — FR-009) `src/grounded_assistant/trace.py` içinde uygula
- [x] T008 LangChain `HumanInTheLoopMiddleware`'i (`interrupt_on` — yalnızca 2 bilinen tool için açık `InterruptOnConfig`, `when` her ikisi için de her zaman `True`; middleware'in kendisi tanımsız bir tool'u otomatik onaylar, fail-closed güvencesi closed-world tool kaydından gelir) `src/grounded_assistant/agent/tool_policy.py` içinde yapılandır; middleware'in ürettiği interrupt'ı bizim otomatik karar-verici kodumuzun anında approve/reject ile resume etmesini sağla (Principle II)
- [x] T009 Temel LangGraph ajan grafiğinin iskeletini (henüz hiçbir erişim yolu bağlanmamış, `InMemorySaver` checkpointer ile — `interrupt()`/resume akışı bunu gerektiriyor) `src/grounded_assistant/agent/graph.py` içinde uygula
- [x] T010 CLI iskeletini (`ask` komutu, `--trace` bayrağı, ajan grafiğini çağırıp `Answer`'ı contracts/cli_interface.md formatında yazdıran; `rich` ile terminal biçimlendirme) `src/grounded_assistant/cli.py` içinde uygula

**Checkpoint**: Temel hazır — US1 ve US2 artık bağımsız şekilde geliştirilebilir

---

## Phase 3: User Story 1 - Kurumsal bilgi bankasından zemine dayalı yanıt (Priority: P1) 🎯 MVP

**Goal**: 4 paralel kaynağı (politika, wiki, destek talebi, teknik dok) sorgulayıp
Hybrid Search + RRF ile birleştirilmiş, kaynak atıflı bir yanıt üretmek; hiçbir kaynak
veri döndürmediğinde bunu açıkça belirtmek.

**Independent Test**: `quickstart.md` → Senaryo 1, 3, 4 (bilgi bankası yanıtı, veri
bulunamadı, kısmi kaynak hatası).

### Implementation for User Story 1

- [x] T011 [P] [US1] BM25 lexical index oluşturma/sorgulama işlevini `src/grounded_assistant/retrieval/bm25_index.py` içinde uygula (`rank_bm25`)
- [x] T012 [P] [US1] Dense embedding index oluşturma/sorgulama işlevini (`.env` gateway'i üzerinden `Qwen3-Embedding-8B`, `langchain-openai` `OpenAIEmbeddings` ile `base_url` override) `src/grounded_assistant/retrieval/dense_index.py` içinde uygula
- [x] T013 [US1] 4 kaynağı (policy/wiki/support_tickets/technical_docs) paralel sorgulayan, BM25+dense sonuçlarını T006'daki RRF ile birleştiren orkestrasyonu `src/grounded_assistant/access_paths/knowledge_base.py` içinde uygula (depends on T011, T012, T006)
- [x] T014 [US1] Kısmi kaynak hatası/boş sonuç durumunu (FR-010) `src/grounded_assistant/access_paths/knowledge_base.py` içinde `KnowledgeBaseSource.status` ile işle
- [x] T015 [US1] Hiçbir kaynak veri döndürmediğinde `Answer.grounded=False` ve açık "bulunamadı" metnini (FR-007, SC-002) üreten yol — T010'da `cli.py`'nin `_build_answer`'ına zaten yazılmıştı, `trace.py`'nin OK/SUCCESS olmayan durumları `source_refs`'e eklememesiyle tutarlı; T016'da gerçek KB sonuçlarına bağlanınca uçtan uca doğrulanacak
- [x] T016 [US1] `knowledge_base` erişim yolunu bir LangGraph node'u olarak `src/grounded_assistant/agent/graph.py` içine bağla (depends on T009, T013)
- [ ] T017 [US1] `src/grounded_assistant/cli.py` üzerinden US1 akışını uçtan uca çalıştırıp `specs/001-ptc-grounded-assistant/quickstart.md` Senaryo 1, 3, 4'ü doğrula
- [ ] T018 [US1] Altan'ın sağlayacağı örnek dokümanları `sample_docs/` altına yükleyip index'lere besleyen bir ingestion scripti yaz (⚠️ dokümanlar gelene kadar bu görev sentetik/placeholder veriyle test edilir)

**Checkpoint**: User Story 1 bağımsız olarak tam çalışır ve test edilebilir durumda

---

## Phase 4: User Story 2 - Canlı sistemlerden güncel veriyle yanıt (Priority: P2)

**Goal**: Skill/tool mekanizması (MCP) üzerinden mock bir canlı sisteme erişip güncel
veriyi zaman damgasıyla sunmak; erişilemezse tahmini değer üretmemek.

**Independent Test**: `quickstart.md` → Senaryo 2.

### Implementation for User Story 2

- [x] T019 [P] [US2] `fastmcp` ile mock canlı sistem MCP sunucusunu (`get_ticket_status`, `list_open_tickets` — contracts/mock_live_system_mcp.md) `mock_services/mock_live_system/server.py` + `mock_services/mock_live_system/data.py` içinde uygula
- [x] T020 [US2] `langchain-mcp-adapters` (`MultiServerMCPClient`, stdio transport) ile mock MCP sunucusuna bağlanan, T008'deki `tool_policy.py` middleware'i üzerinden yalnızca 2 allowlist'li tool'u expose eden `src/grounded_assistant/access_paths/live_systems.py` dosyasını uygula (depends on T008, T019)
- [x] T021 [US2] Zaman aşımı/hata durumunda "erişilemedi, tahmini değer yok" fallback'ini (FR-011) `src/grounded_assistant/access_paths/live_systems.py` içinde uygula (LiveSystemTraceMiddleware — exception'ı yutup açık bir hata ToolMessage'ı döner, model tahmin üretmez)
- [x] T022 [US2] `live_systems` erişim yolunu bir LangGraph node'u olarak `src/grounded_assistant/agent/graph.py` içine bağla (depends on T009, T020)
- [ ] T023 [US2] `src/grounded_assistant/cli.py` üzerinden US2 akışını uçtan uca çalıştırıp `specs/001-ptc-grounded-assistant/quickstart.md` Senaryo 2'yi doğrula

**Checkpoint**: User Story 1 VE User Story 2 birlikte bağımsız çalışır durumda

---

**Not — US3 (Hafıza)**: Spec.md'de tanımlı ama Altan'ın 2026-08-27 tarihli açık kararıyla
Faz 1 kapsamı dışında bırakıldı. Bu fazda hiçbir görev üretilmedi; ileride ayrı bir
`/speckit-specify` / spec güncellemesiyle ele alınacak.

---

## Phase 5: Polish & Cross-Cutting Concerns

- [ ] T024 [P] `specs/001-ptc-grounded-assistant/quickstart.md`'deki tüm senaryoları (1-4 + `--trace` kontrolü) uçtan uca çalıştırıp doğrula
- [x] T025 [P] Kurulum ve çalıştırma talimatlarını `README.md`'ye ekle
- [x] T026 Implementasyon sonrası plan.md → Constitution Check'i tekrar gözden geçir; özellikle Principle II (onaylı-kanal) ve Principle III (izlenebilirlik) fiilen çalıştığını doğrula — 1 bakım riski bulundu ve giderildi (`tool_policy.assert_known_tools`), 1 bilinen sınırlama (faithfulness) kasıtlı olarak açık bırakıldı

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Bağımlılık yok, hemen başlar
- **Foundational (Phase 2)**: Setup'a bağlı — US1 ve US2'yi bloklar
- **US1 (Phase 3)**: Foundational bitince başlar, US2'ye bağımlı değil
- **US2 (Phase 4)**: Foundational bitince başlar, US1'e bağımlı değil (paralel geliştirilebilir)
- **Polish (Phase 5)**: İstenen tüm user story'ler bitince başlar

### Parallel Opportunities

- T003, T004 (Setup) paralel
- T005, T006, T007 (Foundational) paralel
- Foundational bitince: T011+T012 (US1) ve T019 (US2) paralel başlayabilir
- US1 ve US2, Foundational bittikten sonra birbirinden bağımsız olarak paralel ilerleyebilir

---

## Parallel Example: Foundational sonrası

```bash
# US1 ve US2 aynı anda başlayabilir:
Task: "BM25 lexical index — src/grounded_assistant/retrieval/bm25_index.py"
Task: "Dense embedding index — src/grounded_assistant/retrieval/dense_index.py"
Task: "Mock MCP sunucusu — mock_services/mock_live_system/server.py"
```

---

## Implementation Strategy

### MVP First (User Story 1)

1. Phase 1: Setup
2. Phase 2: Foundational (kritik — tüm story'leri bloklar)
3. Phase 3: User Story 1
4. **DUR ve DOĞRULA**: US1'i bağımsız test et (`quickstart.md` Senaryo 1, 3, 4)

### Incremental Delivery

1. Setup + Foundational → temel hazır
2. US1 eklenir → bağımsız test edilir → MVP demo edilebilir
3. US2 eklenir → bağımsız test edilir → tam Faz 1 demo edilebilir
4. (US3/Hafıza, PTC, egress policy, UI → sonraki fazlar, bu tasks.md kapsamı dışında)

---

## Notes

- [P] = farklı dosyalar, bağımsız
- T004 ve T018'deki "Altan'ın örnek dokümanları sağlaması" bağımlılığı, implementasyonun
  gerçek veriyle tam doğrulanabilmesi için gereklidir — bu gelene kadar sentetik/placeholder
  veriyle ilerlemek mümkündür
- Her görevden sonra commit at, checkpoint'lerde story'yi bağımsız doğrula
- US3 (Hafıza) kasıtlı olarak burada yok — bkz. yukarıdaki not
