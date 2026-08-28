# Implementation Plan: Kurumsal Zemine-Dayalı Asistan (Faz 1)

**Branch**: `001-ptc-grounded-assistant` | **Date**: 2026-08-27 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-ptc-grounded-assistant/spec.md`

**Kapsam notu**: Bu plan yalnızca 4 fazlı yol haritasının **Faz 1**'ini (kurumsal asistan; PTC, egress policy ve UI olmadan) kapsar. Hafıza erişim yolu bu fazda kasıtlı olarak dışarıda bırakılmıştır (Altan'ın kararı — bkz. Assumptions güncellemesi aşağıda).

## Summary

Asistan, kullanıcı sorularını yalnızca fiilen çağrılan araçlardan gelen veriye dayanarak yanıtlar; hiçbir erişim yolu veri döndürmediğinde bunu açıkça belirtir. Faz 1 teknik yaklaşımı: **LangGraph** üzerinde çalışan bir Python CLI ajanı; kurumsal bilgi bankası **Hybrid Search (BM25 + yerel dense embedding) + RRF** ile tamamen yerel/hafif olarak sorgulanır; canlı sistemler erişim yolu, gerçek bir protokol (MCP) üzerinden konuşan ama verisi tamamen sahte olan yerel bir mock servis üzerinden test edilir.

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**: `langgraph`, `langchain` (agent middleware — HumanInTheLoopMiddleware için), `langchain-openai` (`.env`'deki OpenAI-uyumlu gateway üzerinden chat model `gemma-4-31B-it` ve embedding model `Qwen3-Embedding-8B` için), `langchain-mcp-adapters`, `fastmcp` (mock canlı sistem sunucusu için), `rank_bm25` (lexical arama), `numpy` (RRF birleştirme), `typer` (CLI)

**Storage**: Yerel dosya sistemi — Altan'ın sağlayacağı örnek dokümanlar (4 kaynak: politika/prosedür, kurumsal wiki, destek talebi arşivi, teknik dokümantasyon) + yerelde oluşturulan basit bir embedding index (dış vector DB yok, disk üzerinde basit bir index dosyası)

**Testing**: pytest

**Target Platform**: Yerel geliştirme ortamı (Linux), tek-process CLI uygulaması

**Project Type**: Single project (CLI)

**Performance Goals**: PoC seviyesi — sabit bir SLA hedeflenmiyor; spec'teki SC-003 gereği tipik bir sorgu için makul (birkaç saniye) yanıt süresi

**Constraints**:
- Hafıza erişim yolu bu fazda **yok** (deferred — Altan'ın açık kararı)
- Ayrı bir dış vector DB **yok** (index yerel diskte tutulur); embedding hesaplaması, yeni bir credential/servis eklenmeden, zaten `.env`'de onaylı OpenAI-uyumlu gateway üzerinden (`Qwen3-Embedding-8B`) yapılır — Altan'ın 2026-08-27 kararı
- Canlı sistemler erişim yolu **mock ama gerçek bir protokol üzerinden** çalışmalı — trivial bir in-process Python fonksiyon stub'ı OLMAMALI (Altan'ın açık kararı)
- Kurumsal bilgi bankası örnek dokümanları **Altan tarafından sağlanacak** — bunlar henüz elde değil; bu bir dış bağımlılık/varsayımdır (bkz. Complexity/Assumptions)
- Onaylı-kanal (Principle II) denetimi `langchain.agents.middleware.HumanInTheLoopMiddleware` (`interrupt_on` + `when` predicate) ile yapılandırılıyor; hiçbir vendor'ın `allowed_callers` benzeri metadata'sına güvenilmiyor. **Düzeltme**: middleware'in kendisi tanımsız bir tool'u otomatik onaylar (fail-open) — fail-closed güvencesi middleware'den değil, `tools=[...]`'a yalnızca 2 bilinen tool'un kayıtlı olmasından (closed-world) geliyor (bkz. research.md #4). Ayrıca bir checkpointer (`InMemorySaver`) gerekiyor.

**Scale/Scope**: Tek demo kullanıcı/oturum, sınırlı örnek doküman seti, 4 paralel bilgi bankası kaynağı + 1 mock canlı sistem

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| İlke | Durum | Not |
|---|---|---|
| I. Zemine Dayalılık — Uydurma Yok | **PASS** | Tasarım, her yanıtın en az bir erişim yolundan gelen veriye dayanmasını zorunlu kılıyor; hiçbir kaynak veri döndürmediğinde açık "bulunamadı" yolu var (data-model.md → `Answer.grounded`). |
| II. Yalnızca Onaylı Kanal | **PASS** | Canlı sistem erişimi yalnızca MCP protokolü üzerinden; her çağrı `HumanInTheLoopMiddleware` policy'sinden geçiyor. Fail-closed güvencesi `tools=[...]`'a yalnızca 2 bilinen tool'un kayıtlı olmasından (closed-world) geliyor, middleware'in kendi varsayılanından değil (bkz. research.md #4). Vendor `allowed_callers` metadata'sına güvenilmiyor. |
| III. İzlenebilirlik ve Denetlenebilirlik | **PASS** | Her yanıt, katkı sağlayan erişim yolu/kaynak(lar)ı kaydedecek şekilde tasarlandı (`trace.py`, data-model.md → `Answer.source_refs`). |
| IV. Teknoloji Kararları Kullanıcıya Aittir | **PASS** | Framework, RAG tekniği, vector store yaklaşımı, mock canlı sistem niteliği ve arayüz Altan ile doğrudan teyit edildi. **Tek istisna**: mock canlı sistemin somut protokolünün (MCP) kendisi ayrı bir soru olarak sorulmadı — bu, research.md'de gerekçesiyle işaretlendi ve Altan'ın onayına/vetosuna açık bırakıldı. |
| V. Basitlik — PoC Kapsam Disiplini | **PASS** | Hafıza erişim yolu, hosted vector DB, çoklu kullanıcı/yetkilendirme gibi bu faz için gereksiz karmaşıklıklar kapsam dışı bırakıldı. |

Post-Phase-1 re-check: Aşağıdaki tasarım (data-model.md, contracts/) yukarıdaki gate'leri ihlal etmiyor; Complexity Tracking'e girecek bir sapma yok.

### Post-implementasyon Constitution Check (T026, 2026-08-27)

Kod yazıldıktan sonra 5 ilke tekrar, gerçek dosyalara bakılarak gözden geçirildi:

| İlke | Durum | Bulgu |
|---|---|---|
| I. Zemine Dayalılık | **PASS** *(1 bilinen sınırlama)* | `cli.py._build_answer`, hiçbir tool başarılı olmadığında modelin ürettiği metni tamamen görmezden gelip açık "bulunamadı" mesajıyla değiştiriyor — model tool çağırmadan doğrudan yanıt vermeye kalksa bile uydurma sızmıyor. **Bilinen sınırlama** (kasıtlı, Faz 1 kapsamı dışı): en az bir tool başarılı olduğunda, modelin metninin tool sonuçlarının ötesine geçip geçmediğini (faithfulness) doğrulayan bir mekanizma yok. |
| II. Yalnızca Onaylı Kanal | **PASS — düzeltildi** | Başlangıçta yalnızca "closed-world'e güveniyoruz" deniyordu; bakım riski bulunup (middleware'in fail-open varsayılanı, bir tool yeni eklenip politikaya işlenmezse sessizce onaylanabilirdi) **giderildi**: `tool_policy.KNOWN_TOOLS` + `assert_known_tools()`, `graph.py`'nin `build_agent`'ında çağrılıyor — politika listesine (ALLOWED_TOOLS/LOCAL_TOOLS) işlenmemiş bir tool agent'a eklenmeye çalışılırsa kurulum anında hata verir (sessiz fail-open yerine gürültülü fail-closed). |
| III. İzlenebilirlik | **PASS** | Her KB kaynağı ve her canlı tool çağrısı `Trace`'e status'üyle kaydediliyor; `--trace` bayrağı ham kaydı JSON olarak veriyor. |
| IV. Teknoloji Kararları Kullanıcıya Aittir | **PASS** | Framework, RAG tekniği, embedding modeli, Tool Gateway mekanizması, `rich` kütüphanesi — hepsi Altan ile teyit edildi. Tek istisna (MCP protokol seçimi) baştan itibaren şeffaf işaretlendi. |
| V. Basitlik | **PASS** | Hafıza yok, hosted vector DB yok, auth/çoklu-kullanıcı yok. Ingestion scripti (T018) Altan'ın açık talebiyle eklendi. |

**Sonuç**: Complexity Tracking'e girecek bir gate ihlali yok. Principle I'deki tek bilinen sınırlama, Faz 2/3'ün (PTC, egress policy) doğal genişleme noktası olarak kayıtlı.

## Project Structure

### Documentation (this feature)

```text
specs/001-ptc-grounded-assistant/
├── plan.md              # Bu dosya
├── research.md          # Faz 0 çıktısı
├── data-model.md         # Faz 1 çıktısı
├── quickstart.md         # Faz 1 çıktısı
├── contracts/            # Faz 1 çıktısı
│   ├── mock_live_system_mcp.md
│   └── cli_interface.md
└── tasks.md              # Faz 2 çıktısı (/speckit-tasks ile, bu komutla oluşturulmaz)
```

### Source Code (repository root)

```text
src/
└── grounded_assistant/
    ├── __init__.py
    ├── cli.py                    # CLI giriş noktası (typer)
    ├── agent/
    │   ├── __init__.py
    │   ├── graph.py              # LangGraph graf tanımı
    │   └── tool_policy.py        # Tool Gateway: HumanInTheLoopMiddleware policy (fail-closed varsayılan)
    ├── access_paths/
    │   ├── __init__.py
    │   ├── knowledge_base.py      # 4 paralel kaynağı sorgulayan orkestrasyon (FR-002/003)
    │   └── live_systems.py        # MCP client bağlantısı (mock canlı sisteme)
    ├── retrieval/
    │   ├── __init__.py
    │   ├── bm25_index.py          # Lexical (BM25) index
    │   ├── dense_index.py         # Yerel embedding tabanlı dense index
    │   └── fusion.py              # Reciprocal Rank Fusion (RRF)
    └── trace.py                   # Her yanıt için kaynak/erişim-yolu izlenebilirlik kaydı (FR-009)

mock_services/
└── mock_live_system/
    ├── server.py                  # FastMCP tabanlı sahte MCP sunucusu
    └── data.py                    # Sahte veri (ör. destek talebi durumu)

sample_docs/                       # Altan'ın sağlayacağı 4-kaynak örnek dokümanları buraya konacak
├── policy/
├── wiki/
├── support_tickets/
└── technical_docs/

tests/
├── unit/
└── integration/
```

**Structure Decision**: Single project (CLI). "Canlı sistemler" ve "kurumsal bilgi bankası" ayrı `access_paths` modülleri olarak tutuluyor ki ileride (Faz 2/3) her biri kendi PTC/egress davranışını bağımsız kazanabilsin. Mock canlı sistem, gerçek bir üründe olacağı gibi ayrı bir process/sunucu (`mock_services/`) olarak çalışıyor — ana ajan kodunun içine gömülü bir stub değil.

## Complexity Tracking

> Gate ihlali yok, bu bölüm boş.
