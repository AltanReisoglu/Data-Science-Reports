# Kurumsal Zemine-Dayalı Asistan — Faz 1

PTC (Programmatic Tool Calling) egress güvenliği araştırmasının bir parçası olan
4 fazlı yol haritasının ilk fazı: kullanıcı sorularını yalnızca fiilen çağrılan
araçlardan gelen veriye dayanarak yanıtlayan bir kurumsal asistan PoC'si.

Detaylı spesifikasyon, plan ve tasarım kararları: [`specs/001-ptc-grounded-assistant/`](specs/001-ptc-grounded-assistant/)
(`spec.md`, `plan.md`, `research.md`, `data-model.md`, `contracts/`, `tasks.md`).

## Mimari (Faz 1)

- **Framework**: LangGraph (`langchain.agents.create_agent`)
- **LLM & Embedding**: `.env`'deki OpenAI-uyumlu gateway üzerinden — chat modeli
  `gemma-4-31B-it`, embedding modeli `Qwen3-Embedding-8B`
- **Kurumsal bilgi bankası**: 4 paralel kaynak (politika, wiki, destek talebi,
  teknik dok) — Hybrid Search (BM25 + dense embedding) + Reciprocal Rank Fusion
- **Canlı sistemler**: mock bir MCP sunucusu (gerçek protokol, sahte veri)
- **Tool Gateway**: `HumanInTheLoopMiddleware` (onay/red kararı) + `assert_known_tools`
  (kayıt-zamanı fail-closed kontrolü)
- **İzlenebilirlik**: her yanıt, katkı sağlayan kaynak/tool'ları `--trace` ile gösterir

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## `.env`

Depo kökünde bir `.env` dosyası olmalı:

```
LLM_BASE_URL=https://.../v1
LLM_API_KEY=...
LLM_MODEL_NAME=gemma-4-31B-it
```

## Örnek dokümanları yerleştirme

`sample_docs/{policy,wiki,support_tickets,technical_docs}/` altına ilgili
kurumsal örnek dokümanları (`.md` dosyaları) koyun. Ardından embedding'leri
önceden hesaplamak için (opsiyonel ama önerilir — her sorguda gateway'e tekrar
embedding isteği atılmasını önler):

```bash
python scripts/ingest_sample_docs.py
```

Bu script çalıştırılmazsa sistem yine çalışır, embedding'ler her sorguda canlı
hesaplanır.

## Mock canlı sistemi başlatma

```bash
python -m mock_services.mock_live_system.server
```

Bu, `get_ticket_status` ve `list_open_tickets` tool'larını sunan yerel bir MCP
sunucusu (stdio) başlatır. CLI çalışırken bu sunucuya otomatik bağlanır; ayrıca
elle başlatmanız gerekmez, `live_systems.py` `MultiServerMCPClient` ile kendi
alt-process'ini yönetir.

## Çalıştırma

```bash
assistant "Uzaktan çalışma politikamız nedir?"
assistant "Şu an açık kritik ticket sayısı kaç?"
assistant "Uzaktan çalışma politikamız nedir?" --trace
```

(`ask` alt-komutu YOK — typer, tek komutlu bir uygulamayı otomatik olarak
düzleştiriyor; `assistant ask "..."` yazarsanız "ask" kelimesi de soru
metninin bir parçası sayılıp hata verir.)

Daha fazla doğrulama senaryosu için: [`specs/001-ptc-grounded-assistant/quickstart.md`](specs/001-ptc-grounded-assistant/quickstart.md).

## Faz 2 — PTC Kod Sandbox'ı (Cilium/eBPF)

Bu fazda, model artık soruyu `search_knowledge_base`/`get_ticket_status`/
`list_open_tickets`'ı tek tek çağırarak DEĞİL, bunları programatik olarak
(döngü/koşul ile) çağıran bir Python kodu yazıp bu kodu ayrı, ağ seviyesinde
kısıtlı bir Kubernetes pod'unda (`run_ptc_code` tool'u) çalıştırarak da
yanıtlayabilir — bu PoC'nin asıl konusu: **PTC'nin egress'i eBPF/Cilium ile
yalnızca onaylı tool kanallarına (Tool Gateway) sınırlanması**
(bkz. [`docs/topic_is_this.md`](docs/topic_is_this.md)).

Detaylı spesifikasyon/tasarım: [`specs/002-ptc-code-sandbox/`](specs/002-ptc-code-sandbox/)
(`spec.md`, `plan.md`, `research.md` — özellikle §4: CiliumNetworkPolicy tasarımı
— `data-model.md`, `contracts/`, `tasks.md`). Kurulum ve tüm doğrulama senaryoları
(çoklu-adımlı orkestrasyon, kaçış denemesi, zaman aşımı) için:
[`specs/002-ptc-code-sandbox/quickstart.md`](specs/002-ptc-code-sandbox/quickstart.md).

Kısaca:

```bash
pip install -e ".[dev]"   # kubernetes client'ı da içerir

# Yerel cluster + Cilium (bkz. quickstart.md tam komutlar için)
kind create cluster --config=k8s/kind-config.yaml
helm install cilium cilium/cilium --version 1.20.0 --namespace kube-system \
  --set hubble.enabled=true --set hubble.relay.enabled=true --set operator.replicas=1

# Tool Gateway + sandbox image'larını yükle, Secret'ı .env'den oluştur, policy'leri uygula
# (context repo kökü OLMALI — Dockerfile, sample_docs/ ve src/'i de kopyalıyor)
docker build -t tool-gateway:local -f mock_services/tool_gateway/Dockerfile . && kind load docker-image tool-gateway:local --name <cluster-adı>
docker build -t ptc-sandbox:local -f sandbox_image/Dockerfile . && kind load docker-image ptc-sandbox:local --name <cluster-adı>
kubectl create secret generic tool-gateway-env --from-env-file=.env
kubectl apply -f k8s/tool-gateway/ -f k8s/policies/

# Hubble gözlemi için (DeniedAction kaydı ve `hubble observe` doğrulaması bunu gerektirir)
cilium hubble port-forward &

assistant "4 kaynaktaki tüm dokümanları tara, VPN konusunu geçen kaç tanesi var?" --trace
```

## Faz 4 — Web Arayüzü + Canlı PTC İzleme Paneli

Faz 1/2'nin CLI'sinin web karşılığı: tarayıcıda soru sorulur, yanıt (grounded/
kaynaklar/kısmi hata notlarıyla) gösterilir. Ekranın sol-alt köşesinde, bir PTC
(sandbox) çalıştırması tetiklendiğinde, o çalıştırmanın TÜM adımları (ConfigMap/
Job oluşturma, çalıştırılan kod, her tool-proxy çağrısı, varsa engellenen bir
erişim girişimi, nihai sonuç) **gerçek zamanlı**, terminal-benzeri bir günlükte
akar — tek bir WebSocket üzerinden.

Detaylı spesifikasyon: [`specs/003-web-ui-live-trace/`](specs/003-web-ui-live-trace/)
(`spec.md`, `plan.md`, `research.md`, `contracts/websocket_protocol.md`,
`quickstart.md`). Backend: FastAPI (mevcut `src/grounded_assistant/` paketinin
`web/` alt-modülü, ayrı bir mikroservis değil). Frontend: düz HTML/JS/CSS,
build aracı yok.

```bash
pip install -e ".[dev]"   # fastapi + uvicorn[standard] da içerir

# Faz 1/2'nin kurulumu (kind cluster + Cilium + Tool Gateway + sandbox image)
# zaten yapılmış olmalı — yukarıdaki Faz 2 bölümüne bakın.

uvicorn grounded_assistant.web.app:app --reload
```

Tarayıcıda `http://localhost:8000` açılır. Doğrulama senaryoları için:
[`specs/003-web-ui-live-trace/quickstart.md`](specs/003-web-ui-live-trace/quickstart.md).

**Not (Faz 4'te bulunan, önceden var olan gerçek Faz 1 hataları — düzeltildi)**:
Faz 1'in doğrudan-tool-calling yolu (`list_open_tickets`/`get_ticket_status`),
Faz 4'ün web testleriyle İLK KEZ gerçek bir LLM'e karşı uçtan uca çalıştırıldı
ve iki gerçek hata ortaya çıktı: (1) MCP-adapted tool'lar yalnızca async
çağrılabiliyordu, `agent.invoke()` bunlarda hata veriyordu → `agent.ainvoke()`'a
geçildi (`graph.invoke_and_resolve` artık async, CLI `asyncio.run` ile sarıyor);
(2) `LiveSystemTraceMiddleware`'in yalnızca senkron `wrap_tool_call`'ı vardı,
`ainvoke` async karşılığını (`awrap_tool_call`) istiyordu → eklendi. Bu, CLI'yi
de düzeltti (Faz 1'den beri var olan, hiç fark edilmemiş bir hataydı).

## Geliştirme

```bash
ruff check .
pytest
```
