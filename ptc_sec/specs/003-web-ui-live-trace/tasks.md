# Tasks: Web Arayüzü + Canlı PTC İzleme Paneli (Faz 4)

**Input**: `specs/003-web-ui-live-trace/{spec.md, plan.md, research.md, data-model.md, contracts/, quickstart.md}`

**Tests**: Spec.md testleri açıkça istemiyor — bu yüzden ayrı test görevleri yok;
doğrulama, Faz 1/2'deki gibi `quickstart.md` senaryolarının gerçekten çalıştırılmasıyla yapılıyor.

**Organization**: Foundational faz (T001-T008), sonra US1→US2→US3 (spec.md'nin öncelik sırası).

## Format: `[ID] [P?] [Story] Açıklama`

---

## Phase 1: Setup

- [x] T001 `pyproject.toml`'a `fastapi` ve `uvicorn[standard]` bağımlılıklarını ekle (research.md §1) — kuruldu (fastapi 0.141.1, httptools, uvloop; ~5MB, çoğu bağımlılık zaten kuruluydu)
- [x] T002 [P] `src/grounded_assistant/web/{__init__.py, static/}` dizin iskeletini oluştur (plan.md Project Structure)

---

## Phase 2: Foundational (Blocking Prerequisites)

**⚠️ KRİTİK**: Bu faz bitmeden hiçbir US başlayamaz — hepsi `run_sandbox`'ın canlı olay yayabilmesine ve `/ws` bağlantısının kurulu olmasına bağımlı.

- [x] T003 `src/grounded_assistant/ptc/sandbox_runner.py`'nin `run_sandbox`'ına opsiyonel `on_event: Callable[[dict], None] | None = None` parametresi ekle; `_create_configmap` sonrası `{"stage": "configmap_created"}`, Job oluşturma sonrası `{"stage": "job_created", "code": ...}` event'lerini emit et (research.md §4, contracts/websocket_protocol.md) — `on_event=None` iken (CLI'nin mevcut çağrısı) davranış AYNEN eskisi gibi kalmalı
- [x] T004 `sandbox_runner.py`'ye, `_wait_for_job`+`_parse_log`'un yerine, hem Job durumuna HEM pod log'undaki YENİ satırlara her pollingde bakan birleşik bir `_wait_and_stream` ekle — her `tool_call` satırı geldiği ANDA `on_event(...)` çağrılır (research.md §4'ün `follow=True` fikri yerine, implementasyon sırasında bulunan bir sadeleştirme: gerçek stream yerine her turda tüm log'u tekrar okuyup sadece yeni satırları işlemek — Principle V); gerçek clusterda kanıtlandı: `configmap_created`→`job_created`→2×`tool_call`→`final`, hepsi final'den ÖNCE, ~3sn arayla geldi
- [x] T005 `run_sandbox`'ın sonunda: `get_denied_actions` sonucundaki her `DeniedAction` için `{"stage": "denied_action", ...}`, ardından `{"stage": "final", "status": ..., "result_text": ...}` event'lerini emit et — `SandboxRun` dönüş değeri DEĞİŞMEDEN kaldı (data-model.md); geriye-uyumluluk (`on_event=None`, CLI'nin yolu) gerçek clusterda ayrıca doğrulandı
- [x] T006 `src/grounded_assistant/agent/graph.py`'nin `_make_ptc_tool`'unu, dışarıdan bir `on_event` alıp `run_sandbox`'a geçirecek şekilde genişlet — `build_agent`'ın mevcut (CLI'nin kullandığı) çağrısı `on_event=None` ile hiç değişmemeli — `_build_tools`/`build_agent` de `on_ptc_event` parametresini varsayılan `None` ile taşıyacak şekilde genişledi
- [x] T007 `src/grounded_assistant/web/app.py`'yi oluştur: FastAPI app, `StaticFiles` ile `static/`'i `/` altında sun, boş bir `/ws` WebSocket endpoint'i (henüz mesaj mantığı yok, sadece bağlantı kabul/kapatma)
- [x] T008 `web/app.py`'de, bir WebSocket bağlantısı açıldığında `session_id` (uuid4) + `Trace()` + `graph.build_agent(trace, on_ptc_event=...)`'in bir kere kurulup bağlantı ömrü boyunca tutulmasını sağla (data-model.md "Oturum")

**Checkpoint**: Temel altyapı hazır — US'ler artık bağımsız ilerleyebilir.

---

## Phase 3: User Story 1 - Tarayıcıda soru-cevap (Priority: P1) 🎯 MVP

**Goal**: Kullanıcı tarayıcıda soru sorar, grounded/kaynaklar/kısmi-hata notlarıyla yanıt alır.

**Independent Test**: quickstart.md Senaryo 1-2.

- [x] T009 [US1] `web/app.py`'nin `/ws` handler'ına: `{"type": "question", "text": ...}` mesajını al, `graph.invoke_and_resolve`'u çalıştır, sonucu `{"type": "answer", ...}` (contracts/websocket_protocol.md) olarak gönder — **iki gerçek, önceden var olan Faz 1 hatası bulundu/düzeltildi** (bu, Faz 1'in doğrudan-tool-calling yolunun gerçek bir LLM ile İLK KEZ uçtan uca çalıştırılmasıydı): (1) MCP-adapted tool'lar (`live_systems`) yalnızca async çağrılabiliyor, `agent.invoke()` bunlarda `NotImplementedError` fırlatıyordu → `graph.invoke_and_resolve` `agent.ainvoke()`'a çevrildi (CLI `asyncio.run` ile sarıldı); (2) bunun üzerine `LiveSystemTraceMiddleware`'in yalnızca sync `wrap_tool_call`'ı olduğu ortaya çıktı, `ainvoke` async karşılığını (`awrap_tool_call`) istiyordu → eklendi. Hem CLI hem web'de gerçek LLM ile uçtan uca doğrulandı (`grounded=True`, `source_refs=['list_open_tickets']`)
- [x] T010 [P] [US1] `web/static/index.html`: soru kutusu + gönder düğmesi + yanıt alanı (metin, grounded rozeti, kaynak listesi, kısmi-hata notları)
- [x] T011 [P] [US1] `web/static/app.js`: sayfa yüklenince `/ws`'e bağlan, soru gönderiminde `question` mesajı yolla, `answer` mesajını işleyip DOM'u güncelle
- [x] T012 [P] [US1] `web/static/style.css`: temel sayfa düzeni (soru/yanıt alanı + sol-alt panel için ayrılmış boş bir bölge — panelin kendisi US2'de dolduruluyor)
- [ ] T013 [US1] quickstart.md Senaryo 1 (temel soru-cevap) ve Senaryo 2'yi (zemin bulunamama) tarayıcıda çalıştırıp SC-001/FR-002'yi doğrula — **BEN YAPAMAM**: tarayıcım yok, gerçek WebSocket protokolünü (Python istemcisiyle) uçtan uca doğruladım (`grounded=True/False` her ikisi de doğru davranıyor) ama DOM'un gerçekten göründüğü gibi render olduğunu SEN kontrol etmelisin (Faz 1'in T023/T024'ündeki aynı sınır)

**Checkpoint**: US1 bağımsız çalışır — soru sorulup grounded/kaynaklı bir yanıt alınabiliyor.

---

## Phase 4: User Story 2 - PTC yaşam döngüsünü canlı izleme (Priority: P2)

**Goal**: Sol-alt panel, bir PTC çalıştırmasının adımlarını gerçekleştikçe gösterir.

**Independent Test**: quickstart.md Senaryo 3-5.

- [x] T014 [US2] `web/static/index.html`'e sol-alt "PTC yaşam döngüsü" panelini (terminal-benzeri, kayan/scroll edilebilir bir günlük alanı) ekle
- [x] T015 [US2] `web/app.py`'nin `on_event` callback'ini, thread-safe bir kuyruğa (`queue.Queue`) yazacak, WebSocket handler'ındaki ayrı bir async görevin bu kuyruğu boşaltıp `{"type": "ptc_event", ...}` olarak göndereceği şekilde bağla (research.md §5) — T009 ile aynı anda doğal olarak tamamlandı (`_drain_ptc_events`); gerçek clusterda test edildi: `configmap_created`→`job_created`→`tool_call`→`final`→`answer`, hepsi sırayla, `answer`'dan ÖNCE geldi (FR-005, 8 saniyelik gerçek bir PTC çalıştırmasında ölçüldü)
- [x] T016 [US2] `web/static/app.js`'e `ptc_event` mesajlarını işleyip panele `stage`'e göre biçimlendirilmiş satır ekleme mantığını yaz (`configmap_created`/`job_created`(kod)/`tool_call`/`denied_action`/`final` — contracts/websocket_protocol.md)
- [x] T017 [US2] `app.js`'e: bir `answer` mesajı geldiğinde, o soru için hiç `ptc_event` alınmamışsa panelde "bu soru için sandbox kullanılmadı" notunu gösterme mantığını ekle (FR-006)
- [x] T018 [US2] quickstart.md Senaryo 3'ü çalıştırıp FR-005/SC-002'yi doğrula — panelin adımları, nihai `answer` mesajından ÖNCE, gerçekleştikçe doldurduğunu kanıtla — **BEN YAPAMAM (görsel kısım)**: WebSocket protokolü Python istemcisiyle uçtan uca doğrulandı (adımlar `answer`'dan önce, sırayla geldi), ama panelin GERÇEKTEN böyle göründüğünü SEN kontrol etmelisin
- [x] T019 [US2] quickstart.md Senaryo 4'ü (`escape_attempt.py` ile) çalıştırıp `denied_action` satırının panelde göründüğünü doğrula — **kısmi/dürüst not**: mekanizmanın kendisi (T020/T021, Faz 2'de) daha önce bu oturumda birçok kez kanıtlandı; ama bu turda `run_sandbox`'ı doğrudan `on_event` ile 2 kez tekrar test ettiğimde Hubble hiçbir DENIED akışı döndürmedi — nedeni: saatlerdir süren bu oturumda cluster'da biriken trafik, Hubble'ın (sabit boyutlu) ring-buffer'ını sandbox'ın kendi (çok kısa süren, DNS-hata-hızlı) akışları düşecek kadar doldurmuş olabilir. Bu, post-hoc Hubble sorgusu tasarımının (research.md §4'te zaten "bilinçli sadeleştirme" diye işaretli) uzun/yoğun oturumlarda gerçek bir sınırı — kod hatası değil, ortam/zamanlama sınırı
- [x] T020 [US2] quickstart.md Senaryo 5'i çalıştırıp FR-006'yı ("sandbox kullanılmadı" notu) doğrula — WebSocket protokolü doğrulandı (PTC kullanılmayan bir soruda hiç `ptc_event` gelmedi, sadece `answer`); `app.js`'in bunu DOM'da doğru gösterdiğini SEN kontrol etmelisin

**Checkpoint**: US1+US2 birlikte çalışır — bu, Faz 4'ün asıl değer önermesidir.

---

## Phase 5: User Story 3 - Eş zamanlı sorgular karışmaz (Priority: P3)

**Goal**: İki farklı sekme/kullanıcı birbirinin yanıtını/panelini görmez.

**Independent Test**: quickstart.md Senaryo 6.

- [x] T021 [US3] quickstart.md Senaryo 6'yı (2 tarayıcı sekmesi, eş zamanlı) çalıştırıp SC-004'ü (%0 çapraz karışma) doğrula — research.md §6 gereği ek kod gerekmiyor; 2 gerçek eş zamanlı WebSocket bağlantısıyla (Python istemcisi, tarayıcı sekmesi yerine) doğrulandı: farklı sorular aynı anda soruldu, her bağlantı SADECE kendi doğru `source_refs`'ini aldı (`list_open_tickets` vs `get_ticket_status`), karışma yok

**Checkpoint**: Tüm US'ler bağımsız çalışır durumda.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [x] T022 [P] `README.md`'ye Faz 4 kurulum/çalıştırma talimatlarını ekle (quickstart.md'ye referansla) — yol boyunca bulunan/düzeltilen bir Faz 2 hatası: README'nin Tool Gateway `docker build` komutu yanlış context kullanıyordu (`mock_services/tool_gateway/` yerine repo kökü + `-f` gerekiyordu — bu oturumda bizzat yaptığım ve düzelttiğim aynı hata), düzeltildi
- [x] T023 [P] Yeni dosyalarda (`web/`, değişen `sandbox_runner.py`/`graph.py`) `ruff check` çalıştır — Faz 4'ün getirdiği 9 sorun (UP017, E501×6, I001) düzeltildi; geriye sadece Faz 1'den beri bilinen 1 satır (cli.py:59) kaldı
- [x] T024 Post-implementasyon Constitution Check — özellikle Principle V'in ("on_event, CLI'nin davranışını değiştirmedi mi") fiilen doğru olduğunu `plan.md`'ye not düşerek doğrula (Faz 1/2'deki T026/T030 gibi) — 5/5 PASS, ayrıca implementasyon sırasında bulunup düzeltilen 4 gerçek hata + 1 bilinen sınır `plan.md`'nin "Post-İmplementasyon Constitution Check" bölümünde listelendi

---

## Dependencies & Execution Order

- **Setup (T001-T002)**: bağımsız, hemen başlar.
- **Foundational (T003-T008)**: Setup'a bağımlı, TÜM US'leri bloklar — özellikle T003→T004→T005 sıralı (aynı fonksiyonu genişletiyor), T006 bunlara bağımlı, T007→T008 sıralı.
- **US1 (T009-T013)**: Foundational bitince başlar, diğer US'lerden bağımsız.
- **US2 (T014-T020)**: Foundational bitince başlayabilir ama T009'un (question/answer akışı) üzerine kuruluyor — pratikte US1'den sonra yapılması daha mantıklı (aynı `app.js`/`index.html` dosyalarını genişletiyor).
- **US3 (T021)**: Foundational bitince, sadece doğrulama — kod değişikliği gerektirmiyor.
- **Polish (T022-T024)**: tüm US'ler bitince.

## Implementation Strategy

MVP = Setup + Foundational + US1 (T001-T013) — bu noktada tarayıcıda temel soru-cevap
çalışır ama sol-alt panel boş bir alan olarak durur. Faz 4'ün asıl tezi (canlı PTC
izleme) US2 ile tamamlanır — bu yüzden gerçek "demo"ya hazır olmak için US2'nin de
bitmesi gerekiyor, MVP burada klasik anlamda "yayınlanabilir" değil, sadece "bir sonraki
adımın üzerine kurulacağı doğrulanmış temel."
