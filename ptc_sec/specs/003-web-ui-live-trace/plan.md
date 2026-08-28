# Implementation Plan: Web Arayüzü + Canlı PTC İzleme Paneli (Faz 4)

**Branch**: `003-web-ui-live-trace` | **Date**: 2026-08-28 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/003-web-ui-live-trace/spec.md`

## Summary

Faz 1/2'nin kurumsal asistanına ve PTC yeteneğine bir web arayüzü ekleniyor. Kullanıcı
tarayıcıda soru sorar (mevcut `assistant` CLI'sinin web karşılığı — grounded/source_refs/
partial_failure_notes gösterimiyle); soru bir PTC (sandbox) çalıştırması tetiklerse,
ekranın sol-alt köşesindeki bir panel, o çalıştırmanın TÜM adımlarını (ConfigMap/Job
oluşturma, kodun kendisi, her tool-proxy çağrısı, engellenen eylemler, nihai sonuç)
GERÇEK ZAMANLI, terminal-benzeri bir akışta gösterir. Teknik yaklaşım: FastAPI (tek
WebSocket, çift yönlü) + düz HTML/JS/CSS; `sandbox_runner.run_sandbox`'a canlı olay
yayma (`on_event` callback + pod log'unu `follow=True` ile stream okuma) eklenir —
CLI'nin mevcut, doğrulanmış davranışı (`on_event=None`) hiç değişmez.

## Technical Context

**Language/Version**: Python 3.11 (mevcut `pyproject.toml` ile aynı)

**Primary Dependencies**: `fastapi`, `uvicorn[standard]` (WebSocket desteği için) —
mevcut bağımlılıklara (langgraph, langchain-openai, kubernetes, fastmcp, vb.) ek olarak.
Frontend'de HİÇBİR bağımlılık yok (düz HTML/JS/CSS, build aracı yok).

**Storage**: N/A — Faz 1/2 gibi durumsuz; `Trace`/`SandboxRun` bir oturum (WebSocket
bağlantısı) ömrü boyunca bellekte tutulur, kalıcı depolama yok (Principle V).

**Testing**: `pytest` (mevcut dev bağımlılığı) — WebSocket mesaj sözleşmesi için birim
testleri; gerçek uçtan-uca doğrulama `quickstart.md`'deki senaryolarla (Faz 1/2'deki gibi
elle/tarayıcıyla).

**Target Platform**: Linux, yerel makine (`localhost`) — Faz 1/2'nin geri kalanıyla aynı
yerel/araştırma ortamı; uzaktan erişim kapsam dışı (spec.md Assumptions).

**Project Type**: web-service (backend gömülü, `src/grounded_assistant/` paketinin bir
alt-modülü — ayrı bir mikroservis DEĞİL) + statik frontend (framework'süz).

**Performance Goals**: Tek kullanıcı/küçük sayıda eş zamanlı sekme (PoC ölçeği) — SC-001
(arayüz kaynaklı gözle görülür gecikme yok), SC-002 (ilk PTC adımı, çalıştırma bitmeden
önce görünür).

**Constraints**: Kimlik doğrulama yok, yalnızca `localhost` (spec.md Assumptions);
mevcut `sandbox_runner.py`/`graph.py`'nin senkron/bloklayan doğası DEĞİŞTİRİLMEDEN
(async'e yeniden yazılmadan) bir thread sınırının arkasına alınır (research.md §5).

**Scale/Scope**: Birkaç eş zamanlı tarayıcı sekmesi (US3'ün test senaryosu — 2 sekme);
üretim ölçeği/yük hedefi yok (PoC).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| İlke | Durum | Gerekçe |
|---|---|---|
| I. Zemine Dayalılık | **PASS** | UI, mevcut `Answer`/`Trace`'i olduğu gibi yansıtıyor; FR-002 grounded=False'un açıkça gösterilmesini zorunlu kılıyor — yeni bir fabrikasyon riski yok. |
| II. Yalnızca Onaylı Kanal | **PASS / kapsam dışı not** | Bu özellik agent'ın DIŞ ağ erişimini (Cilium/Tool Gateway) hiç değiştirmiyor — sadece yerel bir web ön/arka yüzü ekliyor. Yeni bir ingress yüzeyi (web sunucusu) açılıyor olması ayrı bir konu; spec.md Assumptions'ta "yalnızca localhost, kimlik doğrulama kapsam dışı" olarak açıkça sınırlandı (Faz 1'in kimliksiz Tool Gateway kararıyla tutarlı). |
| III. İzlenebilirlik | **PASS (güçlendiriyor)** | Bu özelliğin asıl amacı, mevcut izlenebilirliği (Trace/SandboxRun) canlı/görünür kılmak — Principle III'ü zayıflatmıyor, aktif olarak destekliyor. |
| IV. Teknoloji Kararları Kullanıcıya Aittir | **PASS** | FastAPI + düz HTML/JS/CSS, Altan'a açıkça soruldu ve onaylandı (2026-08-28). |
| V. Basitlik | **PASS** | Frontend framework'süz, backend tek process/tek WebSocket, kalıcı depolama yok, DeniedAction canlı akışı bilinçli olarak toplu bırakıldı (research.md §4) — spekülatif hiçbir soyutlama eklenmedi. |

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/grounded_assistant/
├── web/
│   ├── __init__.py
│   ├── app.py                  # FastAPI app: statik dosya sunumu + tek WebSocket endpoint'i (/ws)
│   └── static/
│       ├── index.html          # Tek sayfa: soru kutusu + yanıt alanı + sol-alt PTC paneli
│       ├── app.js              # WebSocket bağlantısı, mesaj tiplerini (ptc_event/answer) işleme
│       └── style.css
├── ptc/
│   └── sandbox_runner.py       # DEĞİŞECEK: run_sandbox'a opsiyonel on_event callback + pod log'unu
│                                # follow=True ile stream okuma eklenir (research.md §4); CLI'nin
│                                # kendi çağrısı (on_event=None) davranışsal olarak AYNI kalır.
├── agent/graph.py              # _make_ptc_tool, on_event'i run_sandbox'a geçirecek şekilde genişler
└── trace.py, models.py, cli.py # değişmez (yeniden kullanılır)
```

**Structure Decision**: Ayrı bir `backend/`/`frontend/` üst-düzey dizini AÇILMIYOR —
backend, mevcut `src/grounded_assistant/` paketinin yeni bir `web/` alt-modülü (Faz 2'nin
`ptc/` alt-modülüyle aynı desen); frontend, o alt-modülün `static/` klasöründe düz
dosyalar olarak duruyor (build/paketleme adımı yok, FastAPI `StaticFiles` ile doğrudan
sunuyor). Bu, "backend+frontend ayrı üst-düzey proje" seçeneğini KULLANMIYOR çünkü
frontend'in kendi bağımsız bir derleme/dağıtım hattı yok — Principle V.

## Complexity Tracking

> Gate ihlali yok, bu bölüm boş.

## Post-Tasarım Constitution Check (Phase 1 sonrası tekrar kontrol)

`research.md`/`data-model.md`/`contracts/`/`quickstart.md` yazıldıktan sonra 5 ilke
tekrar gözden geçirildi — yukarıdaki tablo geçerliliğini koruyor, tek ek not:
`sandbox_runner.py`'ye yapılacak `on_event` eklentisi (research.md §4), Faz 2'nin
GERÇEK cluster'a karşı doğrulanmış davranışını (`on_event=None` yolu) değiştirmediği
için Principle V'in "gereksiz karmaşıklık ekleme" şartını ihlal etmiyor — CLI'nin kendi
çağrı yolu, bu fazdan hiç etkilenmeden aynı kalıyor.

## Post-İmplementasyon Constitution Check (T024)

Faz 4'ün 24 görevi tamamlandıktan, gerçek cluster + gerçek LLM'e karşı (WebSocket
Python istemcileriyle — tarayıcı DOM doğrulaması hariç, bkz. tasks.md'deki dürüst
notlar) test edildikten sonra:

| İlke | Durum | Kanıt |
|---|---|---|
| I. Zemine Dayalılık | **PASS** | `grounded=True`/`grounded=False` her ikisi de gerçek LLM ile uçtan uca doğrulandı; UI hiçbir yeni fabrikasyon riski eklemedi (mevcut `_build_answer` doğrudan yeniden kullanıldı). |
| II. Yalnızca Onaylı Kanal | **PASS, kapsam notu geçerli** | Bu faz agent'ın egress'ini hiç değiştirmedi; yeni ingress yüzeyi (web sunucusu) kimlik doğrulamasız/yalnızca localhost olarak kalmaya devam ediyor (plan.md'nin ilk Constitution Check'inde zaten işaretliydi). |
| III. İzlenebilirlik | **PASS (güçlendi)** | PTC'nin iç işleyişi artık canlı görünür — `configmap_created`→`job_created`→`tool_call`→`final` zinciri gerçek clusterda, `answer`'dan önce, sırayla akarken kanıtlandı. |
| IV. Teknoloji Kararları Kullanıcıya Aittir | **PASS** | FastAPI + düz HTML/JS, Altan'a açıkça soruldu; implementasyon sırasında YENİ bir teknoloji kararı gerekmedi. |
| V. Basitlik | **PASS** | `on_event`, CLI'nin davranışını değiştirmedi (doğrulandı — CLI gerçek LLM ile tekrar test edildi, aynı şekilde çalıştı); `_wait_and_stream`'in "her turda tüm log'u tekrar oku" sadeleştirmesi gerçek stream yönetiminden kaçındı; `_build_answer`'ın web'de yeniden kullanılması ayrı bir "web DTO" katmanı icat etmedi. |

**İmplementasyon sırasında bulunup düzeltilen gerçek hatalar** (Faz 2'nin
T030'undaki gibi — hepsi ancak gerçek LLM/cluster'a karşı çalıştırarak ortaya çıktı):

1. **En önemlisi**: Faz 1'in doğrudan-tool-calling yolu (`list_open_tickets`/
   `get_ticket_status`), bu depoda İLK KEZ gerçek bir LLM'e karşı uçtan uca
   çalıştırıldı (Faz 1'in kendi T023/T024'ü hep ertelenmişti) — ve MCP-adapted
   tool'ların yalnızca async çağrılabildiği, `agent.invoke()`'un bunlarda hata
   verdiği ortaya çıktı. `graph.invoke_and_resolve` `agent.ainvoke()`'a çevrildi.
2. Bunun zincirleme sonucu: `LiveSystemTraceMiddleware`'in yalnızca senkron
   `wrap_tool_call`'ı vardı — `awrap_tool_call` eklendi.
3. `web/app.py`'de `build_agent` çağrısı, kendi içinde `asyncio.run()` çağıran
   `get_live_system_tools()`'u zaten çalışan bir event loop'un içinden tetikliyordu
   → `asyncio.to_thread` ile sarıldı.
4. README'nin Faz 2 bölümündeki `docker build` komutu yanlış context kullanıyordu
   (repo kökü yerine alt dizin) — düzeltildi.
5. **Bulunan bir sınır (hata değil)**: Hubble'ın post-hoc sorgusu (research.md §4,
   Faz 2'de "bilinçli sadeleştirme" diye işaretlenmişti), saatlerce süren/yoğun bir
   oturumda ring-buffer rotasyonu yüzünden bazen kısa süreli sandbox akışlarını
   kaçırabiliyor — mekanizmanın kendisi daha önce (ve bu oturumda tazeyken) defalarca
   kanıtlandı, bu bir tasarım kusuru değil, ortam/zamanlama sınırı.

Ayrıca (Faz 4'ün kapsamı dışında ama test sırasında yeniden karşılaşılan, zaten
bilinen bir sınır): `sample_docs/wiki/`'deki KKB dokümanları parçalanmamış —
`search_knowledge_base`'in doğrudan (PTC dışı) çağrısı bu yüzden bazen 30sn+
sürebiliyor, hiçbir zaman aşımı korumasız. Bu, Faz 4'ün başında da bilinen,
Altan'ın "şimdilik bırakalım" dediği bir konu.
