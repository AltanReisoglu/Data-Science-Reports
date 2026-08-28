---

description: "Task list for PTC Kod Sandbox'ı — Faz 2"
---

# Tasks: PTC Kod Sandbox'ı (Faz 2)

**Input**: Design documents from `/specs/002-ptc-code-sandbox/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Spec.md testleri açıkça istemiyor; doğrulama `quickstart.md` senaryolarıyla yapılır.

**Organization**: Görevler spec.md'deki kullanıcı hikayelerine göre gruplanmıştır.
Altan'ın önceliği: iskelet (K8s client, Job, Tool Gateway) hızlı geçilir,
**Cilium policy görevlerine (T009, T016, T020-T023) orantısız özen gösterilir.**

## Format: `[ID] [P?] [Story] Description`

## Path Conventions

Hibrit yapı — `src/grounded_assistant/ptc/` (Python), `k8s/` (manifest'ler),
`sandbox_image/`, `mock_services/tool_gateway/` (container image'ları). Bkz. plan.md.

---

## Phase 1: Setup

- [x] T001 `k8s/kind-config.yaml` oluştur (1 control-plane, `disableDefaultCNI: true` — research.md, `Installation Using Kind` referansı)
- [x] T002 `pyproject.toml`'a `kubernetes` bağımlılığını ekle — kuruldu (v36.0.3, ~3MB)
- [x] T003 [P] `sandbox_image/`, `mock_services/tool_gateway/`, `k8s/{tool-gateway,sandbox,policies}/`, `src/grounded_assistant/ptc/` dizin iskeletini oluştur
- [x] T004 [P] `src/grounded_assistant/ptc/__init__.py` (boş) oluştur

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: US1/US2/US3'ün hepsinin üzerine kurulduğu cluster + Tool Gateway + sandbox altyapısı

**⚠️ CRITICAL**: Bu faz bitmeden hiçbir user story çalıştırılamaz (gerçek bir cluster gerekiyor)

- [x] T005 `kind create cluster --config=k8s/kind-config.yaml` ile cluster'ı, ardından Cilium'u Helm ile kur (`--set hubble.enabled=true --set hubble.relay.enabled=true --set hubble.ui.enabled=false --set operator.replicas=1` — tek node'da 2. operator replikası zamanlanamadığı için düzeltildi); `cilium status` ile doğrulandı (Cilium/Operator/Envoy/Hubble Relay hepsi OK)
- [x] T006 `src/grounded_assistant/models.py`'ye `data-model.md`'ye göre `SandboxRun`, `CapabilityGrant`, `DeniedAction` dataclass/enum'larını ekle
- [x] T007 `mock_services/tool_gateway/server.py`'yi `fastmcp` HTTP transport ile yaz (`search_knowledge_base` → Faz 1'in `knowledge_base.py`'sini, `get_ticket_status`/`list_open_tickets` → Faz 1'in `mock_services/mock_live_system/data.py`'sini in-process saracak şekilde) — contracts/tool_gateway_mcp.md
- [x] T008 `mock_services/tool_gateway/Dockerfile`'ı yaz, image'ı build edip `kind load docker-image` ile cluster'a yükle (360MB — pyproject.toml'un tamamı değil, sadece gerekli paketler kurularak indirme kısıtlandı)
- [x] T009 `k8s/policies/tool-gateway-egress.ciliumnetworkpolicy.yaml`'ı research.md §4.2'deki YAML'a göre yaz — Tool Gateway'in TEK dış hedefi embedding gateway FQDN'i (`toFQDNs` + DNS), başka her şey deny
- [x] T010 `k8s/tool-gateway/deployment.yaml` + `service.yaml`'ı yaz; `.env`'den `kubectl create secret generic tool-gateway-env --from-env-file=.env` ile (YAML'a gömmeden) Secret oluşturuldu; deployment/service/policy uygulandı, pod `Running`, FastMCP `0.0.0.0:8443`'te dinliyor (log ile doğrulandı)
- [x] T011 `sandbox_image/entrypoint.py`'yi yaz — `/sandbox/code.py`'yi okur, `TOOL_GATEWAY_ENDPOINT` ortam değişkenini kullanarak MCP client kurar, kodu çalıştırır, sonucu contracts/sandbox_job_contract.md'deki JSON formatında stdout'a yazar — bilerek Python-seviyesinde kısıtlama yok (research.md §4.3), enforcement Cilium'da
- [x] T012 `sandbox_image/Dockerfile`'ı (`python:3.11-slim` tabanlı, sadece `fastmcp`) yaz, build edip `kind load docker-image` ile yükle — image 222MB
- [x] T013 `src/grounded_assistant/ptc/sandbox_runner.py`'yi `kubernetes` client ile yaz: `run_sandbox(code: str) -> SandboxRun` — ConfigMap oluştur, `k8s/sandbox/job-template.yaml`'a göre Job oluştur (activeDeadlineSeconds=30, `TOOL_GATEWAY_ENDPOINT` enjekte edilmiş), tamamlanmasını bekle, pod log'unu oku, ConfigMap+Job'u temizle (`ttlSecondsAfterFinished` zaten var ama açıkça da silinir) — gerçek cluster'a karşı uçtan-uca test edildi (`list_open_tickets()` çağrısı, `status=success`); yol boyunca bulundu/düzeltildi: `kubernetes` client'ın `read_namespaced_pod_log`'u JSON'a benzeyen log'u sessizce parse edip `str(dict)` ile geri yazıyor (tırnakları bozuyor) — `_preload_content=False` ile ham yanıt okunarak atlatıldı

**Checkpoint**: Cluster + Tool Gateway ayakta, `sandbox_runner.run_sandbox("print(1)")` çağrısı çalışıp temiz bir sonuç dönüyor olmalı — US1/US2/US3 artık bağımsız ilerleyebilir.

---

## Phase 3: User Story 1 - Kod yazarak çoklu-adımlı görev orkestre etme (Priority: P1) 🎯 MVP

**Goal**: Model, birden fazla tool çağrısını tek bir sandbox çalıştırmasında orkestre edebilsin.

**Independent Test**: quickstart.md Senaryo 1 — 2+ tool çağrısı gerektiren bir görev, tek sandbox çalıştırmasıyla, daha az model turuyla tamamlanır.

### Implementation for User Story 1

- [x] T014 [US1] `src/grounded_assistant/agent/graph.py`'ye, modelin "kod yaz ve sandbox'ta çalıştır" seçeneğini kullanabileceği bir `run_ptc_code` tool'u ekle (bu tool, çağrıldığında `sandbox_runner.run_sandbox`'ı tetikler) — Faz 1'deki `search_knowledge_base`/canlı-sistem tool'ları ile birlikte, ayrı bir mod olarak; `tool_policy.LOCAL_TOOLS`'a eklendi (kendisi Tool Gateway'e çıkmıyor, kısıtlama sandbox pod'unun CiliumNetworkPolicy'sinde)
- [x] T015 [US1] `sandbox_runner.py`'nin döndürdüğü `SandboxRun.tool_calls`'ı `trace.py`'nin `Trace.record_tool_call`'ına besleyen entegrasyonu yaz (FR-008) — bunun için `entrypoint.py`'nin kontratı genişletildi (her tool çağrısı `"type": "tool_call"` satırı olarak da loglanıyor, contracts/sandbox_job_contract.md güncellendi); yol boyunca bulunan/düzeltilen 2. hata: tool proxy'ler yalnızca keyword-argüman kabul ediyordu, LLM'in doğal pozisyonel çağrısı (`search_knowledge_base("sorgu")`) patlıyordu — `_ARG_NAMES` eşlemesiyle düzeltildi; gerçek clusterda 2 tool çağrısıyla (list_open_tickets + search_knowledge_base) uçtan uca doğrulandı
- [x] T016 [US1] `k8s/policies/sandbox-egress.ciliumnetworkpolicy.yaml`'ı research.md §4.1'deki YAML'a göre yaz (tek kural: sandbox → Tool Gateway, `toEndpoints`, DNS yok) — apply edildi (`kubectl get ciliumnetworkpolicies` → VALID); yol boyunca bulundu/düzeltildi: `sandbox_runner.py` Tool Gateway'i DNS adıyla enjekte ediyordu (bu policy'nin DNS'siz tasarımını kıracaktı) — ClusterIP'yi `read_namespaced_service` ile okuyup doğrudan enjekte edecek şekilde düzeltildi; Hubble ile ÇİFT YÖNLÜ doğrulandı: `8.8.8.8:443`'e deneme → `Policy denied DROPPED`, Tool Gateway'e gerçek çağrı → `FORWARDED` (Hubble CLI de kuruldu, ~21MB, v1.19.4)
- [x] T017 [US1] `cli.py`'nin `_build_answer`'ını, sandbox sonucundan (varsa) `result_text`'i de kullanacak şekilde güncelle — SC-003'ün ("her sandbox çalıştırması trace'de görünür") T015'in tool_calls-only kaydıyla tam karşılanmadığı görüldü (hiç tool çağrısı yapmadan timeout olan bir run hiç görünmezdi); `AccessPath.PTC_SANDBOX` + `Trace.record_sandbox_run` eklendi, `graph.py`'de her çalıştırma sonrası çağrılıyor; `cli.py`'nin "hiçbir kaynak yok" mesajı Faz 2'yi de anacak şekilde güncellendi
- [x] T018 [US1] quickstart.md Senaryo 1'i çalıştırıp SC-001'i (model turu azalması) doğrula — gerçek LLM (gemma, .env) ile çalıştırıldı: model, 4 ayrı `search_knowledge_base` turu yerine TEK `run_ptc_code` çağrısıyla (sandbox içinde 4 çağrıyı döngüyle yaparak) sonucu üretti; `--trace` çıktısı bunu 1 `ptc_sandbox` + 4 iç `live_system` girdisi olarak doğru gösterdi

**Checkpoint**: US1 bağımsız çalışır ve test edilebilir durumda.

---

## Phase 4: User Story 2 - Onaylı-kanal dışına çıkışın engellenmesi (Priority: P2)

**Goal**: Sandbox, Tool Gateway dışında hiçbir hedefe erişemesin; bu, gözlemlenebilir ve kanıtlanabilir olsun.

**Independent Test**: quickstart.md Senaryo 2 — kaçış denemesi yapan bir kod, ağ seviyesinde engellenir, `hubble observe` ile görülür.

### Implementation for User Story 2

- [x] T019 [P] [US2] `sandbox_test_fixtures/escape_attempt.py` oluştur (quickstart.md Senaryo 2'deki kaçış denemesi) — `requests` yerine stdlib `urllib.request` (sandbox image bilerek minimal, research.md §4.3 — yeni paket indirmeden aynı ağ-seviyesi test)
- [x] T020 [US2] `sandbox_runner.py`'ye, bir `SandboxRun` bittikten sonra `hubble observe --pod <job-adı> --verdict DENIED` çıktısını parse edip `DeniedAction` kayıtlarına dönüştüren bir yardımcı fonksiyon ekle (data-model.md → DeniedAction) — `get_denied_actions()`, hubble CLI kuruldu (~21MB, v1.19.4), `-o json` ile parse ediliyor, ulaşılamazsa best-effort []; yol boyunca bulunan eksik: `SandboxRun`'da `denied_actions` alanı hiç yoktu, eklendi (models.py); `status=denied_action` olduğunda `result_text=None` zorunluluğu (FR-011) `run_sandbox`'a işlendi
- [x] T021 [US2] `trace.py`'ye `record_denied_action(action: DeniedAction)` metodunu ekle, `to_json()`'a dahil et — `graph.py`'nin `_make_ptc_tool`'undan çağrılıyor
- [x] T022 [US2] `escape_attempt.py`'yi `run_sandbox` ile çalıştırıp SC-002'yi (0 başarılı kaçış) doğrula — gerçek clusterda çalıştırıldı: `status=DENIED_ACTION`, `result_text=None`, hem Hubble ham çıktısında (`coredns:53 ... Policy denied DROPPED`) hem üretilen `--trace` JSON'unda (`denied:coredns-...:53`) göründü
- [x] T023 [US2] Kontrol testi: aynı sandbox'ın Tool Gateway'e yaptığı izinli çağrının `hubble observe --verdict FORWARDED`'da göründüğünü doğrula (yanlış pozitif/negatif olmadığını kanıtlamak için) — izinli bir çalıştırma (`list_open_tickets()`) `status=SUCCESS`, `denied_actions=[]` üretti; ayrıca Hubble ham çıktısında `ptc-sandbox → tool-gateway:8443 ... FORWARDED` doğrulandı

**Checkpoint**: US1 + US2 birlikte çalışır; bu fazın çekirdek iddiası (Principle II) kanıtlanmış durumda.

---

## Phase 5: User Story 3 - Zaman aşımı ve hatanın zarifçe ele alınması (Priority: P3)

**Goal**: Sandbox çöker/asılı kalırsa asistan çökmesin, tahmini değer üretmesin.

**Independent Test**: quickstart.md Senaryo 3 — sonsuz döngü, `activeDeadlineSeconds` ile sonlandırılır, açık "tamamlanamadı" yanıtı döner.

### Implementation for User Story 3

- [x] T024 [P] [US3] `sandbox_test_fixtures/infinite_loop.py` oluştur (`while True: pass`)
- [x] T025 [US3] `sandbox_runner.py`'ye Job'un `status.conditions[].reason == "DeadlineExceeded"` durumunu algılayıp `SandboxRun.status = timeout` döndüren mantığı ekle (FR-007) — T013'te önceden yazılmıştı; T024 ile test edilirken gerçek bir hata bulundu/düzeltildi: bu Kubernetes sürümünde ilk gelen koşulun `type`'ı `"Failed"` değil `"FailureTarget"` (reason yine `DeadlineExceeded`) — sadece `type`'a bakmak timeout'u kaçırıp `error`'a düşürüyordu; kontrol `reason`'a göre yapılacak şekilde düzeltildi
- [x] T026 [US3] `cli.py`'de `status == timeout`/`error` durumunda "tamamlanamadı, tahmini değer yok" yanıtının üretildiğini doğrula (grounded=False yolu, Faz 1'deki gibi) — sentetik bir TIMEOUT `SandboxRun` ile `_build_answer` doğrudan test edildi: `grounded=False`, jenerik "veri bulunamadı" metni, `partial_failure_notes=['...: timeout']` — fabrikasyon yok
- [x] T027 [US3] `infinite_loop.py`'yi çalıştırıp SC-004'ü (öngörülebilir sürede sonlanma) doğrula — gerçek clusterda çalıştırıldı: `status=TIMEOUT`, 30.2s'de sonuçlandı (activeDeadlineSeconds=30 + küçük tampon); pod'un kendisinin SIGTERM'e yanıt vermeyip grace-period sonunda SIGKILL'lenmesi ~30s daha sürebiliyor ama bu, `run_sandbox`'ın DÖNÜŞ süresini etkilemiyor (Job silme çağrısı bloklamıyor)

**Checkpoint**: Tüm user story'ler bağımsız çalışır durumda.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [x] T028 [P] `README.md`'ye Faz 2 kurulum/çalıştırma talimatlarını ekle (quickstart.md'ye referansla) — yol boyunca bulunan/düzeltilen bir Faz 1 hatası: README'nin `assistant ask "..."` örnekleri yanlıştı (gerçek CLI, typer'ın tek-komut düzleştirmesiyle `assistant "..."` bekliyor — `--help` ile doğrulandı), düzeltildi
- [x] T029 [P] `.gitignore`'a yeni riskleri kontrol et — `k8s/` altında hiçbir Secret manifesti yok (yalnızca `secretRef` referansı; `.env` içeriği `kubectl create secret --from-env-file` ile imperative oluşturuldu, hiçbir YAML'a gömülmedi), `.env` zaten `.gitignore`'da — ek risk bulunmadı
- [x] T030 Post-implementasyon Constitution Check — özellikle Principle II'nin (iki CiliumNetworkPolicy) ve Principle III'ün (üç seviyeli izlenebilirlik: pod log, Tool Gateway log, Hubble flow) fiilen çalıştığını `plan.md`'ye not düşerek doğrula (Faz 1'deki T026 gibi) — 5/5 PASS, ayrıca implementasyon sırasında bulunup düzeltilen 5 gerçek hata `plan.md`'nin yeni "Post-İmplementasyon Constitution Check" bölümünde listelendi

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Bağımlılık yok
- **Foundational (Phase 2)**: Setup'a bağlı — **gerçek bir cluster gerektirdiği için** US1/US2/US3'ün hepsini bloklar (Faz 1'den farklı olarak burada "foundational" bir altyapı kurulumu, sadece kod değil)
- **US1 (Phase 3)**: Foundational bitince başlar
- **US2 (Phase 4)**: Foundational bitince başlar, US1'e bağımlı değil (ama T019-T023 pratikte T014'teki `run_ptc_code` tool'unu kullanır)
- **US3 (Phase 5)**: Foundational bitince başlar, US1/US2'ye bağımlı değil
- **Polish (Phase 6)**: Tüm story'ler bitince

### Parallel Opportunities

- T003, T004 (Setup) paralel
- Foundational bitince: T019/T024 (test fixture'ları) hemen yazılabilir, diğer US2/US3 görevlerini beklemez
- T028, T029 (Polish) paralel

---

## Implementation Strategy

### MVP First (User Story 1)

1. Phase 1: Setup
2. Phase 2: Foundational (kritik — gerçek `kind`+Cilium cluster'ı burada kurulur)
3. Phase 3: User Story 1
4. **DUR ve DOĞRULA**: quickstart.md Senaryo 1

### Incremental Delivery

1. Setup + Foundational → cluster + Tool Gateway hazır
2. US1 → çoklu-tool orkestrasyonu çalışıyor (MVP)
3. US2 → **bu fazın asıl iddiası kanıtlanıyor** (Cilium gerçekten engelliyor)
4. US3 → sağlamlık katmanı

---

## Notes

- Bu fazın "foundational" katmanı, Faz 1'den çok daha ağır — gerçek bir Kubernetes
  cluster'ı kurmayı içeriyor. T005 bitmeden hiçbir şey test edilemez.
- T009 ve T016 (iki CiliumNetworkPolicy) bu tasks.md'nin en kritik iki görevi —
  Altan'ın "asıl amaç bu" dediği kısım.
- US3 (Hafıza) Faz 1'de olduğu gibi burada da yok.
