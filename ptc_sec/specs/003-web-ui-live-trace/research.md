# Research: Web Arayüzü + Canlı PTC İzleme Paneli (Faz 4)

## 1. Backend framework: FastAPI

- **Decision**: FastAPI + `uvicorn[standard]` (ASGI sunucu, WebSocket desteği `standard` extra'sıyla gelir).
- **Rationale**: Altan'ın kararı. Python-native, WebSocket'i ekstra bir uzantı gerektirmeden destekler, mevcut `src/grounded_assistant/` paketine ayrı bir mikroservis açmadan (aynı process, aynı import alanı) sarılabilir.
- **Alternatives considered**: Flask+Flask-SocketIO (ek bağımlılık), çıplak `websockets` kütüphanesi (yönlendirme/istek ayrıştırmayı elle yazmak gerekirdi) — ikisi de reddedildi.
- **Kaynak**: Altan'ın kararı (2026-08-28).

## 2. Frontend: düz HTML/JS/CSS

- **Decision**: Build aracı yok, framework yok — tarayıcının native `WebSocket` API'si + `fetch` (gerekirse) ile doğrudan konuşan tek bir statik sayfa.
- **Rationale**: Altan'ın kararı; Principle V'e (PoC kapsam disiplini) en sadık seçenek — npm/build zinciri, bu depoya şu ana kadar hiç girmemiş bir karmaşıklık katmanı olurdu.
- **Kaynak**: Altan'ın kararı (2026-08-28).

## 3. Tek WebSocket, çift yönlü, çok-tipli mesaj — ayrı HTTP endpoint'i yok

- **Decision**: Tarayıcı, sayfa yüklenince TEK bir WebSocket bağlantısı açar (`/ws`). Kullanıcı soru gönderdiğinde bu AYNI bağlantı üzerinden gönderilir; sunucu da hem PTC yaşam-döngüsü olaylarını hem nihai yanıtı AYNI bağlantı üzerinden, tipli mesajlar (`{"type": "ptc_event", ...}` / `{"type": "answer", ...}`) olarak geri gönderir.
- **Rationale**: FR-004/FR-005'in "canlı akış" gereksinimi zaten bir WebSocket gerektiriyor; soru gönderimi için AYRICA bir HTTP POST endpoint'i açmak (ki bu da yaygın bir desendir) burada gereksiz bir ikinci kanal olurdu — Principle V.
- **Alternatives considered**: HTTP POST /ask (yanıt için) + ayrı WebSocket /ptc-stream (panel için) — iki bağlantıyı senkronize tutmak (hangi PTC olayının hangi soruya ait olduğu) gereksiz karmaşıklık katardı, reddedildi.

## 4. `sandbox_runner.py`'nin canlı olay yayma ihtiyacı — GERÇEK bir kod değişikliği

Bu, en önemli teknik karar — çünkü **zaten yazılıp gerçek cluster'a karşı doğrulanmış Faz 2 kodunu** etkiliyor.

- **Mevcut durum (Faz 2)**: `run_sandbox(code) -> SandboxRun` tamamen senkron/toplu (batch) çalışıyor — Job bitene kadar `_wait_for_job` ile bekliyor, ANCAK bittikten SONRA `_read_pod_log` ile TÜM log'u tek seferde okuyup `_parse_log` ile ayrıştırıyor. Bu, FR-005'in ("adımlar GERÇEKLEŞTİKÇE görünmeli, sona toplanıp basılmamalı") gerektirdiği şeyle DOĞRUDAN çelişiyor — mevcut haliyle kullanıcı hiçbir şeyi Job bitene kadar göremez.
- **Decision**: `run_sandbox`'a opsiyonel bir `on_event: Callable[[dict], None] | None = None` parametresi eklenir. Bu callback, şu anlarda çağrılır:
  1. ConfigMap yazıldıktan hemen sonra (`{"stage": "configmap_created", ...}`)
  2. Job oluşturulduktan hemen sonra, kodun kendisiyle birlikte (`{"stage": "job_created", "code": ...}`)
  3. Pod log'u `read_namespaced_pod_log(..., follow=True)` ile (Job bitmeden, SATIR SATIR akan bir stream olarak) okunur — her `tool_call` satırı ayrıştırıldığı ANDA callback'e verilir (`{"stage": "tool_call", ...}`)
  4. Job bittiğinde nihai durum (`{"stage": "final", "status": ..., "result_text": ...}`)
  5. `get_denied_actions` (Hubble sorgusu) hâlâ Job bittikten SONRA, TOPLU çalışır — bu bilinçli bir sadeleştirme (bkz. aşağıdaki not).
- **Rationale**: `follow=True`, kubernetes client'ın zaten desteklediği bir parametre (pod'un log'unu, bitmeden, canlı bir stream olarak döner) — yeni bir bağımlılık gerektirmiyor, sadece `_read_pod_log`'un TOPLU okuma şeklini STREAM okumaya çeviriyor. Mevcut `_parse_log`'un satır-satır ayrıştırma mantığı zaten bu kullanıma uygun (küçük bir uyarlamayla, satır geldikçe ayrıştırılacak şekilde).
- **Bilinçli sadeleştirme — DeniedAction hâlâ toplu**: Hubble flow'larını GERÇEKTEN canlı izlemek (`hubble observe --follow`'u sürekli açık tutup ilgili pod'a ait olayları filtrelemek) ayrı bir arka-plan sürecinin sürekli çalışmasını gerektirirdi — Principle V gereği bu fazın kapsamına alınmadı. Denied action'lar, Job bittikten hemen sonra (nihai durumdan HEMEN ÖNCE) toplu olarak gösterilir — kullanıcı bunları "olay anında" değil "çalıştırma biter bitmez" görür. Bu, FR-004'ün ruhuna uyar (adım hâlâ gösteriliyor) ama FR-005'in "olduğu anda" vurgusunu bu TEK adım için gevşetir; bu doküman bunu açıkça not düşüyor.
- **Geriye dönük uyumluluk**: `on_event=None` olduğunda (ör. CLI'nin kendi çağrısında) davranış AYNEN eskisi gibi kalır — CLI hiçbir şekilde etkilenmez, Faz 2'nin gerçek cluster testleri geçerliliğini korur.
- **Kaynak**: Mevcut `src/grounded_assistant/ptc/sandbox_runner.py` kodu + `kubernetes` client'ın `read_namespaced_pod_log(follow=True)` parametresi (resmi client kütüphanesinin standart özelliği).

## 5. Eş zamanlılık: FastAPI'nin event loop'unu bloklamamak

- **Decision**: Her WebSocket bağlantısı için, `agent.invoke(...)` (ve onun tetiklediği `run_sandbox`) bir arka-plan thread'inde (`asyncio.to_thread`) çalıştırılır. `on_event` callback'i bu thread'den, thread-safe bir kuyruğa (`queue.Queue`) olay yazar; WebSocket handler'ındaki ayrı bir async görev bu kuyruğu boşaltıp `websocket.send_json(...)` çağırır.
- **Rationale**: `sandbox_runner.py`/`graph.py`'nin altındaki `kubernetes` client'ı ve LangGraph'ın senkron `invoke()`'u zaten senkron/bloklayıcı — bunları async'e YENİDEN YAZMAK (Principle V'e aykırı, gereksiz risk, zaten doğrulanmış kodu bozma riski) yerine, bloklayan kısmı bir thread sınırının arkasına almak yeterli ve çok daha güvenli.
- **Kaynak**: FastAPI'nin kendi resmi dokümantasyonunun önerdiği desen (senkron/bloklayan iş için `run_in_threadpool`/`asyncio.to_thread`).

## 6. US3 (eş zamanlı sorgular karışmaz): oturum kimliği

- **Decision**: Her WebSocket bağlantısı kendi `session_id`'sini (rastgele üretilmiş) alır — Faz 1'in CLI'sindeki `thread_id` kavramının birebir aynısı. Bir bağlantının olayları/yanıtı SADECE o bağlantıya gönderilir (FastAPI'de her WebSocket zaten kendi bağımsız handler coroutine'inde çalışır) — çapraz karışma mimari olarak mümkün değil, ekstra bir kilitleme/routing mekanizması gerekmiyor.
- **Rationale**: FastAPI'nin WebSocket modeli zaten bağlantı-başına izole; SC-004 ("%0 çapraz karışma") bunun doğal bir sonucu, ekstra bir şey inşa etmeye gerek yok.
