# PTC Çalıştırma Ortamı — SOTA Araştırması

**Soru:** Her PTC çalıştırması için sıfırdan bir Kubernetes Job ayağa kaldırmak
(ölçülen ~7.21 sn) gerekli mi? Kod çalıştıran üretim sistemleri hangi izolasyon
primitifini seçiyor, adımlar arası kalıcılığı nasıl çözüyor, ve "her run yeni pod"
modeli artifact persistence ile gerçekten çelişiyor mu?

**Tarih:** 2026-09-03 · **Tür:** araştırma dokümanı (karar dokümanı değil) ·
**Kod değişmedi.**

---

## 0. Bu doküman neyi kapsıyor, neyi kapsamıyor

Üç doküman üst üste biniyor; sınırı baştan çizmek gerekiyor:

| Doküman | Kapsamı | Bu dokümanla ilişkisi |
|---|---|---|
| `PTC_Error_Recovery_ve_Artifact_Persistence_Arastirmasi.md` | Mevcut kodun tespiti: traceback yok, yazılabilir disk yok, `json.dumps` + `str()` yolu | Girdi — problem tanımı buradan geliyor |
| `PTC_Artifact_Persistence_Arastirmasi.md` (2026-09-02) | **Depo katmanı**: orkestratör desenleri, Parquet/Arrow, pickle/CWE-502, GC/TTL, covert storage channel, presigned URL, capability handle | **Tekrarlanmıyor.** Oradaki sonuçlar veri kabul edildi |
| `archive/egress-policy/PTC_Daha_Hafif_Alternatifler_Arastirmasi.md` | Kubernetes-içi hızlandırma: `SandboxWarmPool`, tek-pod-çok-container, k3s/kind, Firecracker snapshot, WASM | Kısmen örtüşüyor; §5'te **doğrulanıyor ve bir alıntısı düzeltiliyor** |
| **Bu doküman** | **Çalıştırma ortamı (runtime) katmanı**: SOTA sistemlerin izolasyon primitifi, soğuk başlatma rakamları, kalıcılık modeli | Yeni |

Yeni olan asıl parça: **Cloudflare Code Mode** (dünkü notta hiç yok) ve
**runtime ömrü ↔ artifact kalıcılığı** ilişkisinin sistem sistem çıkarılması.

---

## 1. Çerçeve: iki ayrı katman, tek bir soru sanılıyor

Araştırmanın en belirleyici bulgusu bir mimari ayrım:

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│  ÇALIŞTIRMA ORTAMI          │        │  DEPO / DURUM                │
│  (runtime, izolasyon)       │        │  (artifact, state)           │
│  - VM / microVM / container │  ≠     │  - object store / volume     │
│  - gVisor / V8 isolate      │        │  - snapshot / checkpoint     │
│  - WASM / in-process        │        │  - DB (SQLite, DO storage)   │
│  Ömrü: ms … saatler         │        │  Ömrü: dakikalar … 30 gün +  │
└─────────────────────────────┘        └──────────────────────────────┘
```

Hiçbir SOTA sistem bu ikisini aynı şey olarak ele almıyor. "Sandbox ne kadar
yaşamalı" ve "veri nerede yaşamalı" **ayrı ayrı** cevaplanmış sorular. PoC'de
bunlar tek soruya yapışmış durumda ("pod ölünce her şey gidiyor") — bu, bir
zorunluluk değil, bir tasarım artığı.

---

## 2. S1 — SOTA hangi izolasyon primitifini kullanıyor?

### 2.1 Toplu tablo

Rakamlar **yalnızca kaynağın kendi verdiği** rakamlardır; kaynak rakam vermiyorsa
"kaynakta rakam yok" yazıyor. Kıyaslama uyarısı için §5.1'e bakın — bu sütunlar
aynı şeyi ölçmüyor.

| Sistem | İzolasyon tipi | K8s? | Soğuk başlatma (kaynağın kendi rakamı) | Kalıcılık modeli | Ağ |
|---|---|---|---|---|---|
| **Anthropic code execution / PTC** | Yönetilen container (detay açıklanmamış) | Bilinmiyor (açıklanmamış) | Kaynakta rakam yok; "container startup" bir *overhead* olarak anılıyor | `container.id` geri gönderilerek yeniden kullanım; ~5 dk boştan sonra **checkpoint**, **30 gün** içinde geri yüklenir; dosyalar + **REPL durumu** korunur | **Tamamen kapalı** ("Internet access: Completely disabled") |
| **OpenAI Code Interpreter (Responses)** | Yönetilen container | Bilinmiyor | Kaynakta rakam yok | Container **20 dk kullanılmazsa** sona erer; sona erince "python nesneleri dâhil bellekteki durum kaybolur"; `auto` modda bağlamdaki container yeniden kullanılır | Dokümanda bu sayfada net ifade yok |
| **E2B** | Linux VM (Firecracker tabanlı — resmî dokümanda "isolated Linux VM") | Hayır (kendi bulut altyapısı) | Docs'ta başlatma rakamı **yok**; **pause ≈ 4 sn / 1 GiB RAM**, **resume ≈ 1 sn** | `pause()`/`connect()`: **dosya sistemi + bellek**; duraklatılmış sandbox **süresiz** saklanır; ayrıca snapshot, fork, filesystem-only snapshot, Volumes (private beta) | Varsayılan açık; `allowInternetAccess:false` ≡ `denyOut:['0.0.0.0/0']`; allow/deny listeleri |
| **Modal (Sandboxes)** | **gVisor** ("containerized and virtualized using gVisor") | Hayır | "Containers boot in about one second"; memory snapshot ile azaltılabilir | Sandbox varsayılan ömrü **5 dk**, `timeout` ile **24 saate** kadar; **Filesystem Snapshot** (varsayılan TTL 30 gün), **Memory Snapshot** (7 gün, uzatılamaz); **Volumes** ayrı katman | Doküman sayfasında ağ kısıtı ayrı konu |
| **Vercel Sandbox** | **Firecracker microVM**, kendi kernel'i | Hayır | "Startup time: **Milliseconds** (Firecracker optimized for fast boot)" (tabloda; mutlak sayı yok) | **Varsayılan kalıcı**: durunca dosya sistemi otomatik snapshot'lanır, sonraki oturum aynı yerden devam eder; Snapshots, Drives (beta) | Varsayılan çıkış açık; **sandbox firewall** ile politika; per-sandbox proxy CA |
| **Daytona** | Varsayılan **Linux container**; ayrıca Linux VM / Windows VM / GPU sınıfları | Bilinmiyor (açıklanmamış) | Container sınıfı: **<90 ms** | **Varsayılan kalıcı**: durdurmak silmez, dosya sistemi kalır. Container sınıfı **pause/resume desteklemez** (bellek her stop'ta silinir); VM sınıfları fork + pause/resume + memory snapshot; Volumes; S3/R2/GCS FUSE mount | Network limits + preview auth |
| **Fly.io Machines** | Hızlı açılan VM (REST API) | Hayır | İlk oluşturma "maybe low double digit seconds"; **var olan makineyi başlatma "well under a second"** | Kök dosya sistemi **efemer** ("blank slate on every startup"); kalıcılık için Volume | — |
| **Cloudflare Code Mode / Dynamic Workers** | **V8 isolate** (dosya sistemi yok, env yok) | Hayır | Blog: isolate'lar "a handful of milliseconds" içinde başlıyor, "a few megabytes"; kesin rakam yok | **Sandbox'ta kalıcılık YOK.** "The executor and connector instances remain transient." Durum sandbox dışında: Durable Object **SQLite** (execution kayıtları, connector logları, snippet'ler) | **Varsayılan kapalı**: `globalOutbound: null`; `fetch()`/`connect()` hata fırlatır. Tool'lara erişim **Workers RPC binding**'leri üzerinden |
| **Cloudflare Sandbox SDK** (ayrı ürün) | Containers (tam Linux) + Durable Objects koordinasyon | Hayır | Kaynakta rakam yok | "Persistent state between executions" | — |
| **Letta `run_code`** | Sunucu tarafı sandbox; Docker kurulumlarında **E2B** (`E2B_API_KEY`) | Hayır | Kaynakta rakam yok | **"Fresh environment per execution (no state persistence)"** | "Full network access for API calls" |
| **Pyodide / WASM** | Dil/bytecode seviyesi (CPython → WebAssembly) | — | Kaynakta rakam yok | Yok (MEMFS bellek içi) | Host'un verdiği yeteneklerle sınırlı |
| **Deno permissions** | **Süreç seviyesi**, in-process değil | — | — | — | Varsayılan-yasak (`--allow-net=example.com` gibi kapsamlanabilir). **Uyarı:** `--allow-run` ve `--allow-ffi` sandbox'ı fiilen devre dışı bırakır; docs "katmanlı savunma" öneriyor |
| **Bizim PoC** | `runc` container, Kubernetes Job, Cilium/eBPF | **Evet** | **~7.21 sn toplam** (~3.08 sn Job → ilk `tool_call`) | Yok (salt-okunur ConfigMap; iş bitince Job+ConfigMap silinir) | Yalnız Tool Gateway (identity-based `toEndpoints`) |

### 2.2 Dört aile ve seçim mantığı

Tablodaki 12 sistem dört aileye ayrılıyor; aileyi belirleyen şey **hangi tehdide
karşı savunulduğu**:

| Aile | Örnek | Ne garanti ediyor | Bedeli |
|---|---|---|---|
| **microVM** (kendi kernel'i) | Vercel Sandbox, E2B, Fly | Kernel-seviyesi kaçış yüzeyi yok; ayrıcalıklı iş yükleri (Docker-in-sandbox, FUSE, VPN) mümkün | En ağır imaj/bellek; Firecracker'ın kendi iddiası **<125 ms boot, <5 MiB overhead, 150 microVM/sn/host** |
| **Kullanıcı-uzayı kernel** | Modal (gVisor), Agent Sandbox (gVisor/Kata) | Syscall'ları userspace'te yakalar; VM'in sabit maliyeti yok | "higher per-system call overhead", syscall-yoğun iş yüklerinde yavaş |
| **V8 isolate** | Cloudflare Code Mode | Milisaniyelik başlatma, MB'lık bellek; **dosya sistemi hiç yok** | Yalnızca JS/TS; pandas/pyarrow gibi bir veri yığını yok |
| **In-process / dil seviyesi** | Pyodide, Deno permissions | Kernel hiç devreye girmiyor | Deno'nun kendi dokümanı bunu güvenilmeyen kod için **yeterli görmüyor** |

**PoC'nin yeri:** düz `runc` + ağ politikası. Yani izolasyon gücü bakımından
tablonun en zayıf ucunda (paylaşılan kernel, syscall filtresi yok), buna karşılık
soğuk başlatma bakımından da en yavaş ucunda. Bu kombinasyon SOTA'da bir örneği
olmayan bir noktada duruyor: *hız için ödenen bedel izolasyonla geri alınmıyor.*
Sebebi mimari değil — bedelin kaynağı izolasyon değil, **Kubernetes'in pod
kurulum yolu** (§5.2).

### 2.3 Cloudflare Code Mode — neden ayrı bir başlık hak ediyor

Code Mode, PTC'nin *aynı problemine* verilmiş, bizimkinden radikal biçimde farklı
bir cevap. Fikir şu: MCP tool'larını modele tool olarak sunmak yerine **TypeScript
API'sine çevir**, modele tek bir `codemode({ code })` tool'u ver, kodu izole bir
Worker'da çalıştır.

Sayısal iddia (Cloudflare kendi API'si üzerinden):

| Ölçüm | Değer |
|---|---|
| Cloudflare API endpoint sayısı | 2.500+ |
| Klasik MCP ile tool tanımlarının bağlam maliyeti | ~1,17 milyon token |
| Code Mode ile | ~1.000 token (iki tool: `search()`, `execute()`) |
| Azalma | %99,9 |

Mekanizma, PoC'nin `entrypoint.py`'sindeki tool-proxy desenine şaşırtıcı derecede
yakın, ama üç noktada daha katı:

```
Model ──codemode({code})──►  Dynamic Worker Loader
                             └─ V8 isolate (dosya sistemi YOK, env değişkeni YOK)
                                globalOutbound: null   ← fetch()/connect() hata fırlatır
                                    │
                                    │ Workers RPC (binding)
                                    ▼
                             Connector (MCP / OpenAPI)  ← runtime her çağrıyı ARAYA GİRİP denetler
                                    │
                                    ▼
                             Durable Object SQLite   ← execution kaydı, connector log'u,
                                                       pending approval, snippet
```

Birincil ifadeler:

- *"DynamicWorkerExecutor uses a Dynamic Worker Loader to create an isolated Worker
  for each execution pass."* — **her çalıştırma için yeni izolat**. Yani "her run
  sıfırdan" ilkesi burada da var; sadece maliyeti milisaniye.
- *"External `fetch()` and `connect()` calls are blocked by default.
  DynamicWorkerExecutor configures `globalOutbound: null` unless you provide
  another value."* — sıfır-egress varsayılanı, ağ politikası ile değil **runtime
  konfigürasyonuyla**.
- *"Connector calls cross the sandbox boundary through Workers remote procedure
  calls (RPC). The runtime intercepts each call before the connector executes it."*
  — bizim Tool Gateway'in yaptığı işin dil-seviyesi karşılığı; ama **araya girme
  noktası ağ değil, RPC**.
- *"The executor and connector instances remain transient."* + *"The runtime stores
  execution records, connector-call logs, pending approvals, and snippets in
  isolated SQLite storage. This state survives request completion and Durable
  Object hibernation."* — **çalıştırma ortamı efemer, durum dışarıda.**
- Limit: *"Each value stored for durable replay has a serialized character limit of
  1,000,000."* Ayrıca varsayılan 50 terminal execution kaydı, duraklamış
  execution'lar için 24 saat.
- `snippets`: bir çalıştırmadaki kod, `saveSnippet(name, …)` ile kaydedilip sonraki
  çalıştırmalarda `search()`/`describe()` ile bulunabiliyor — **kod düzeyinde
  yeniden kullanım**, veri düzeyinde değil.

**PTC için önemi:** bu, "sandbox efemer kalsın, kalıcılık başka katmanda olsun"
tezinin en saf uygulaması. Ve dikkat çekici olan: prompt injection savunması olarak
*dosya sisteminin ve env değişkenlerinin hiç var olmaması* sayılıyor
("no file system, no environment variables to leak through prompt injection").
Dünkü dokümanın §6.2'de açık bıraktığı "artifact'e gömülü talimat" sorusuna
Cloudflare'ın verdiği cevap: sandbox'ta okunabilir kalıcı bir yüzey bırakmamak.

Karşılaştırma için Anthropic'in aynı problemi çözme biçimi tam tersi yönde:
dosya sistemi **var** ve merkezî — ara sonuçlar dosyaya yazılıyor, `$OUTPUT_DIR`
sözleşmesiyle dışarı veriliyor, tekrar kullanılan kod "skills" klasöründe
saklanıyor. İki tasarım da tutarlı; ortak nokta, **kalıcılığın açık bir sözleşmeye
bağlanması**.

### 2.4 Peki Kubernetes gerekli mi?

Doğrudan cevap veren bir birincil kaynak var: Kubernetes SIG Apps'in
**`agent-sandbox`** projesi ve GKE'nin ürünleştirmesi.

- kubernetes.io blogu, sorunu bizim ölçtüğümüzle aynı yerde tanımlıyor:
  *"Starting a new pod adds about a second of overhead. That's perfectly fine when
  deploying a new version of a microservice, but when an agent is invoked after
  being idle, a one-second cold start breaks the continuity of the interaction."*
- Çözüm: `SandboxWarmPool` — *"maintaining a pool of pre-provisioned Sandbox pods,
  effectively eliminating cold starts"*; `SandboxClaim` ile havuzdan alınıyor.
- Sandbox CRD'nin sunduğu şey aslında **uzun ömürlü, durumlu tekil pod**:
  kalıcı kimlik, kalıcı disk ("secure scratchpad"), **suspend/resume**, sıfıra
  ölçekleme, ve runtime olarak **gVisor / Kata Containers**.
- GKE'nin rakamı (Google Cloud blogu): *"The Agent Sandbox API's integrated warm
  pool enables GKE to allocate **300 sandboxes per second, per cluster, at sub
  second latency, with 90% of allocations completing in 200 milliseconds**."*
  Ayrıca boştaki ajanlar için **Pod Snapshots** ile askıya alma, ve ucuz
  yenilenme için "cold pool of suspended sandboxes".

Ama aynı blog, Kubernetes'in sınırını da açıkça söylüyor: **Agent Substrate**,
*"takes the core secure runtime and snapshotting capabilities of Agent Sandbox and
pairs them with a minimal control plane designed to bypass some of the limitations
of Kubernetes, without reinventing the rest of it."*

**Sonuç (S1):** Kubernetes *gerekli değil* — SOTA kod-çalıştırma sağlayıcılarının
çoğu (E2B, Modal, Vercel, Fly, Cloudflare) K8s kullandığını söylemiyor bile; Modal
ve Cloudflare kendi kontrol düzlemlerini yazmış. Ama Kubernetes *yeterli* — koşul,
"her istek için yeni pod" desenini bırakmak. Ekosistemin kendi cevabı bu:
**havuzdan claim + suspend/resume**, sıfırdan Job değil.

---

## 3. S2 — Adımlar arası kalıcılık nasıl çözülüyor?

Üç yaklaşım, üçü de sahada:

### 3.1 (a) Uzun ömürlü sandbox + pause / resume / snapshot

| Sistem | Ne saklanıyor | Süre | Rakamlar |
|---|---|---|---|
| **Anthropic container reuse** | Dosyalar **+ Python yorumlayıcı durumu** (`code_execution_20260120`+) | ~5 dk boştan sonra checkpoint; **30 gün** içinde ID ile geri yükleme; sonrasında geri getirilemez | 5 GiB RAM, 5 GiB disk, 1 CPU; PTC'de hücre başına 90 sn duvar-saati; bekleyen PTC çağrısı ~4 dk sonra `TimeoutError` |
| **E2B pause/resume** | Dosya sistemi **+ bellek** (süreçler, yüklü değişkenler) | Duraklatılmış sandbox **süresiz** | pause ≈ **4 sn / 1 GiB RAM**, resume ≈ **1 sn**; sürekli çalışma 24 sa (Pro) / 1 sa (Base), pause sayaç sıfırlar |
| **E2B filesystem-only** | Yalnız disk (`keepMemory:false`) | — | "Faster"; resume'da **yeniden boot**, süreçler ölür |
| **Modal snapshots** | FS snapshot: yalnız disk / Memory snapshot: bellek + disk | FS: varsayılan **30 gün TTL**; Memory: **7 gün, uzatılamaz** | Memory snapshot alırken sandbox sonlanıyor; açık TCP bağlantıları kapanıyor; aynı instance tipinde geri yüklenmeli |
| **Vercel persistent sandbox** | Dosya sistemi (stop'ta otomatik snapshot) | Oturum varsayılan **5 dk** timeout; sandbox birden çok oturuma yayılır | "Resuming from a snapshot is even faster than starting a fresh sandbox" |
| **Daytona** | Container sınıfı: stop/start arası **dosya sistemi**; VM sınıfları ayrıca bellek | Silinene kadar (ephemeral/auto-delete hariç) | Container **<90 ms**; container sınıfı pause/resume **desteklemiyor** |
| **GKE Agent Sandbox** | Pod Snapshots ile askıya alınmış pod | — | warm pool: %90 tahsis **200 ms** içinde |

Ortak desen: **kalıcılık opt-in ve bir kimliğe bağlı** (container ID, sandbox ID,
snapshot ID). Hiçbiri "her şey her zaman kalıcı" demiyor.

### 3.2 (b) Harici nesne deposu / volume

| Sistem | Mekanizma | Kritik ayrıntı |
|---|---|---|
| **Modal Volumes** | Dağıtık dosya sistemi, birden çok container mount edebilir | **`commit()`/`reload()` gerekiyor**; "last write wins ... any data the last writer didn't have when committing changes will be lost"; aynı dosyaya eşzamanlı yazım önerilmiyor; v1'de <50.000 dosya önerisi |
| **E2B Volumes** | Sandbox ömründen bağımsız kalıcı depo, birden çok sandbox'a mount | private beta |
| **E2B / Vercel / Daytona → object storage** | S3/R2/GCS'i FUSE ile mount, ya da "Mount remote storage" | Depo, sandbox'ın *dışında* bir ürün |
| **Vercel Drives** | Sandbox'lara takılan kalıcı disk | beta |
| **Anthropic Files API** | `$OUTPUT_DIR`'e kopyalanan dosyalar `file_id` olarak döner, Files API'den indirilir | Container verisi 30 gün; Files API dosyaları **silinene kadar** kalır |

Modal'ın uyarısı burada bizim için doğrudan geçerli: paylaşılan bir volume,
**eşzamanlılık semantiği** getiriyor. Dünkü dokümanın "Tool Gateway arkasında
nesne deposu + opaque handle" önerisi, tam da bu semantiği tek bir yazıcıya
(gateway) toplayıp içeriği değişmez (immutable, content-hash'li) yaptığı için
Modal'ın "last write wins" tuzağına düşmüyor.

### 3.3 (c) REPL durumu (yorumlayıcı canlı)

Yalnızca iki sistemde açıkça var:

- **Anthropic**: *"the Python interpreter state (such as variable bindings) also
  persists across requests that reuse the container"* — ve bu özellik **PTC'ye
  bağlı** (`code_execution_20260120`+; Haiku 4.5'te yok).
- **E2B**: pause/resume bellek dâhil olduğu için yüklü değişkenler ve çalışan
  süreçler resume'da yerinde.

Buna karşılık **Letta `run_code`** tam tersini seçiyor: *"Fresh environment per
execution (no state persistence)"*. **OpenAI** ise ikisinin arasında: container
yaşadığı sürece python nesneleri duruyor, 20 dk sonra "any state in the old
container's memory (like python objects) will be lost".

### 3.4 Üç yaklaşımın karşılaştırması

| Kriter | (a) Uzun ömürlü sandbox | (b) Harici depo | (c) REPL durumu |
|---|---|---|---|
| Dataframe'i adımlar arası taşır mı | Evet (diskte) | Evet (serileştirilmiş) | Evet (bellekte, serileştirmesiz) |
| Serileştirme kararı gerekir mi | Hayır | **Evet** (Parquet/Arrow — dünkü doküman §4) | Hayır |
| Çalıştırmalar arası izolasyon | **Zayıflar** (aynı ortam) | Depo tarafında kurulabilir | Zayıflar |
| Çok-oturum / çok-kullanıcı ölçeği | Sandbox başına kaynak tutar | İyi | Sandbox başına kaynak tutar |
| Denetlenebilirlik (lineage, TTL) | Snapshot metadata'sı kadar | **En iyi** (handle, hash, TTL, provenance) | Yok |
| Ağ politikasının göremediği kanal açar mı | Evet (aynı disk) | Evet — ama gateway'de denetlenebilir | Evet |
| PoC'ye uygulanabilirlik | Orta (pod ömrü değişir) | **Yüksek** (Cilium politikası değişmez) | Düşük (pod her run ölüyor) |

**Sonuç (S2):** SOTA'nın çoğunluğu (a)+(b) kombinasyonunu kullanıyor: ortam bir
süre yaşıyor, ama *kalıcı olması gereken* veri ortamın dışına yazılıyor. (c) bir
konfor katmanı — Anthropic'te bile opsiyonel ve container reuse'a bağlı.

---

## 4. S3 — "Her run yeni pod" ile artifact persistence gerçekten çelişiyor mu?

### 4.1 Sistemleri iki eksende yerleştirince çelişki dağılıyor

| Sistem | Çalıştırma ortamı ömrü | Artifact / durum nerede yaşıyor |
|---|---|---|
| Cloudflare Code Mode | **Her execution için yeni izolat** | Dışarıda: Durable Object SQLite (kayıtlar, snippet'ler) |
| Letta `run_code` | **Her çağrıda taze ortam** | Ortamda hiç — sonuç modele döner |
| Anthropic code execution | Varsayılan yeni container; **opt-in** yeniden kullanım | Container diski (30 gün) + Files API (kalıcı) |
| OpenAI Code Interpreter | `auto` modda bağlamdaki container yeniden kullanılır | Container (20 dk) + üretilen dosyalar |
| E2B / Vercel / Daytona | **Uzun ömürlü, kalıcı varsayılan** | Sandbox diski + Volumes / Drives / object storage |
| Modal Sandbox | 5 dk varsayılan, 24 sa'e kadar | Volumes + snapshot'lar |
| GKE Agent Sandbox | Uzun ömürlü, warm pool + snapshot ile askıya alınır | PersistentVolume ("scratchpad") |
| **Bizim PoC** | **Her run için yeni pod** | **Hiçbir yerde** ← eksik olan tek hücre |

Tablo okununca şu görülüyor: **efemer runtime seçen sistemler, kalıcılığı runtime'ın
dışına koyarak çözüyor.** Cloudflare bunu en açık söylüyor ("executor ... remain
transient" + "state ... survives request completion"). Yani:

> "Her run yeni pod" ile artifact persistence **çelişmiyor**; çelişen şey
> "her run yeni pod" ile **"hiçbir yerde kalıcı hiçbir şey yok"**.

Dünkü dokümanın §8.4'te vardığı sonuç bu araştırmayla bağımsız olarak doğrulanıyor:
"iz bırakmaz" iddiası, *çalıştırma ortamı* hakkında bir iddia olarak korunabilir;
üretilen verinin onaylı bir depoda saklanması bu iddiayı bozmaz.

### 4.2 Ama bir taviz var ve adı konmalı

Efemer runtime + harici depo ailesi üç şeyi kaybediyor:

1. **REPL durumu yok.** Bir adımda kurulan pahalı nesne (eğitilmiş model, açık
   bağlantı, ısınmış cache) bir sonraki adımda yeniden kurulmak zorunda.
2. **Serileştirme zorunlu hale geliyor.** Deponun içinden geçen her şeyin bir
   formatı olmak zorunda (dünkü doküman §4: Parquet/Arrow, pickle yasak).
3. **Self-repair pahalı.** Error recovery dokümanının §3'te işaret ettiği kesişim:
   4. adımda hata alınca 1-3. adımların çıktısı da yeniden üretiliyor. Kalıcılık
   *bu maliyeti düşürüyor* ama pod başlatma maliyetini düşürmüyor.

SOTA bu tavizi nasıl karşılıyor? İki yolla: (i) runtime'ı o kadar ucuz yapmak ki
tekrar üretmek acıtmasın (Cloudflare: milisaniyelik izolat), (ii) content-hash /
snippet ile tekrar üretmemek (Anthropic'in "skills" klasörü, Cloudflare'ın
`saveSnippet`'i). **Bizde ikisi de yok**: runtime 7 saniye ve tekrar-kullanım
mekanizması sıfır. Asıl gerilim burada.

---

## 5. S4 — Bizim PoC için pratik sonuç

### 5.1 Önce dürüst kıyas uyarısı

Aşağıdaki rakamları yan yana koymak *ancak* şu uyarıyla anlamlı:

| Rakam | Aslında neyi ölçüyor |
|---|---|
| Firecracker **<125 ms** | Yalnızca microVM boot |
| Daytona **<90 ms** | Sandbox tahsisi (muhtemelen sıcak altyapı üzerinden) |
| GKE **%90 < 200 ms** | **Önceden ısıtılmış havuzdan claim** — pod oluşturma değil |
| kubernetes.io **"~1 sn"** | Yeni pod'un ek yükü (imaj hazırken) |
| Modal **"~1 sn"** | Container boot |
| E2B resume **~1 sn** | Duraklatılmış VM'in geri yüklenmesi |
| **Bizim 7.21 sn** | **Uçtan uca `run_sandbox()`**: ConfigMap+Job yazımı → scheduling → CNI → Python başlangıcı → kod → **1 sn'lik polling kuantizasyonu** → **2 adet `hubble observe` subprocess'i** → senkron silme |

Yani 7.21 sn ile 125 ms'i doğrudan kıyaslamak yanlış olur — biz uçtan uca bir
orkestrasyonu, onlar tek bir primitifin açılışını ölçüyor. Doğru kıyas noktası
kubernetes.io'nun "~1 sn" ifadesi ve arşiv dokümanındaki **3.08 sn** (Job → ilk
`tool_call`).

### 5.2 7.21 saniyenin anatomisi — kodun söylediği

`sandbox_runner.py:run_sandbox` okunduğunda, ölçülen sürenin **kritik yoluna**
izolasyonla hiç ilgisi olmayan üç kalem giriyor:

```python
# 1) Polling kuantizasyonu — iş bitse bile ortalama ~0.5 sn, en kötü 1 sn beklenir
_POLL_INTERVAL_SECONDS = 1.0
...
        time.sleep(_POLL_INTERVAL_SECONDS)

# 2) Hubble: İKİ ayrı subprocess, her biri 10 sn timeout'lu — sonucu beklenerek
denied_actions = get_denied_actions(run_id, job_name, started_at)   # ← _cleanup'tan ÖNCE

# 3) Senkron temizlik — ttlSecondsAfterFinished zaten varken bir de elle silinir
_cleanup(core_v1, batch_v1, run_id, job_name)
_emit(on_event, {"stage": "final", ...})
...
    finished_at=datetime.now(UTC),     # ← ölçüm burada bitiyor
```

Ayrıca `_read_pod_log` **her turda tüm log'u yeniden okuyor** (bilinçli bir
sadeleştirme, yorumda gerekçesi var) — küçük log'da ucuz, ama her tur bir ekstra
API çağrısı.

**Pratik sonuç:** ~7.21 sn'nin tamamı "Kubernetes pod maliyeti" değil. Arşiv
dokümanının kendi ölçümüne göre Job → ilk `tool_call` **3.08 sn**; kalan ~4 sn'nin
kayda değer bir kısmı polling + Hubble + senkron cleanup. Bunların hiçbiri
izolasyon primitifi değiştirmeyi gerektirmiyor.

### 5.3 Seçenekler

| # | Seçenek | Beklenen kazanç | Mühendislik yükü | Güvenlik modeline etkisi | Feda edilen iddia |
|---|---|---|---|---|---|
| **0** | **Ölçüm hijyeni**: poll aralığını 1 sn → ~0.1-0.2 sn, Hubble sorgusunu cleanup sonrasına/arka plana al, `_cleanup`'ı `ttlSecondsAfterFinished`'a bırak | Muhtemelen **1-3 sn** (ölçülmeli) | **Çok düşük** — birkaç satır | Yok (Hubble zaten best-effort; engelleme Cilium'da) | Yok |
| **1** | **Warm pool** (`kubernetes-sigs/agent-sandbox` veya elle: N adet hazır Job/Pod) | GKE rakamı: %90 tahsis **<200 ms** | Orta-yüksek (CRD + controller ya da kendi havuz yöneticimiz) | Havuzdaki pod ağı **zaten kurulu**; Cilium identity aynı etiketle geliyor, politika değişmez | "Her run *sıfırdan yaratılmış* pod" — ama "her run temiz ortam" korunabilir (claim'de reset) |
| **2** | **Uzun ömürlü sandbox + opt-in yeniden kullanım** (Anthropic container reuse'un birebir karşılığı): `run_id`/oturum → pod eşlemesi, boşta kalınca sonlandır | Aynı oturumda ikinci ve sonraki çağrılarda **~0** pod maliyeti; ayrıca REPL durumu bedava gelir | Orta (yaşam döngüsü + kota + reaper) | **Çalıştırmalar arası izolasyon zayıflar**; dünkü §6.1'deki "Cilium'un göremediği kanal" bu sefer *pod içinde* açılır | "Her run yeni pod" tümden düşer; "iz bırakmaz" nitelenmek zorunda |
| **3** | **K8s'i bırak, hafif çalıştırıcı** (Firecracker snapshot/restore, gVisor, ya da izolat) | Teorik olarak **10-30 ms** (arşiv §7) | **Çok yüksek** — kendi orkestrasyonumuz | Cilium'u Kubernetes dışında (CLI/API ile) kurmak gerekir; PoC'nin gösterdiği şey değişir | PoC'nin *anlatısı*: "Kubernetes + Cilium ile onaylı kanal" |
| **4** | **Code Mode deseni**: runtime'ı olabildiğince ucuz + durumu tamamen dışarıda (Tool Gateway arkasındaki depo) | Kalıcılık sorununu çözer, **hız sorununu çözmez** | Düşük-orta (dünkü §8.3'teki kapsam) | En tutarlısı — Cilium politikası hiç değişmez, yetkilendirme gateway'de | Hiçbiri (mevcut tezle uyumlu) |

Seçenekler dışlayıcı değil: **0 + 4** birbirini tamamlıyor ve hiçbir iddiayı feda
etmiyor; **1** hız problemini kalıcı olarak çözüyor; **2** yalnızca REPL durumu
gerçekten gerekliyse anlamlı.

### 5.4 Cluster'ı korumak mı, hafif çalıştırıcıya geçmek mi?

Araştırmanın verdiği cevap **cluster'ı korumak** yönünde, üç gerekçeyle:

1. **Darboğaz Kubernetes'in kendisi değil, "her istek için yeni pod" deseni.**
   kubernetes.io bunu açıkça yazıyor ("about a second of overhead") ve çözümü de
   Kubernetes içinde veriyor (warm pool + suspend/resume). Ekosistem bu problemi
   terk ederek değil, yeni bir primitif ekleyerek çözmüş.
2. **Ölçülen 7.21 sn'nin önemli bir kısmı izolasyondan gelmiyor** (§5.2). Runtime
   değiştirmeden kazanılabilecek saniyeler duruyorken, mimariyi değiştirmek
   yanlış katmana müdahale olur.
3. **Egress politikası artık değiştirilebilir olsa bile**, PoC'nin gösterdiği
   şey Cilium/eBPF ile kanal onayı. Hafif çalıştırıcıya geçmek (seçenek 3) bu
   gösterimin *taşıyıcısını* sökmek demek — teknik olarak mümkün, ama kazanç
   (saniyeler) ile kayıp (anlatı + büyük mühendislik yükü) orantısız.

Sıralama önerisi (karar değil):

```
Şimdi        →  Seçenek 0 (ölçüm hijyeni) + Seçenek 4 (Tool Gateway arkasında artifact deposu)
Sonra        →  Seçenek 1 (warm pool) — hız gerçekten sorun olmaya devam ederse
Yalnızca     →  Seçenek 2 (uzun ömürlü sandbox) — REPL durumu somut bir ihtiyaç haline gelirse,
gerekirse       ve "iz bırakmaz" iddiası açıkça nitelendikten sonra
Kapsam dışı  →  Seçenek 3 — üretim ölçeğinde bir sistem olsaydı ilk bakılacak yer
```

---

## 6. Bulunamayanlar, çelişkiler, düzeltmeler

Bunları saklamak yerine yazmak gerekiyor:

| Konu | Durum |
|---|---|
| **Anthropic container soğuk başlatma süresi** | Birincil dokümanda **rakam yok**. Yalnızca "container startup" bir overhead kalemi olarak anılıyor ("trades a small fixed overhead (container startup, script generation)") |
| **Anthropic'in altyapısı K8s mi** | Açıklanmamış. "secure, sandboxed container" deniyor, altındaki primitif belirtilmiyor |
| **E2B'nin Firecracker kullandığı** | Docs sayfalarında "isolated Linux VM" deniyor; **Firecracker adı ve ~150 ms iddiası birincil E2B dokümanında/README'sinde bulunamadı** (ikincil kaynaklarda ve üçüncü taraf repo açıklamalarında var). Firecracker'ın <125 ms rakamı ise Firecracker'ın **kendi** sitesinden |
| **Cloudflare izolat başlatma süresi** | Kesin rakam yok: "a handful of milliseconds", "a few megabytes". Workers dokümanı da mutlak sayı vermiyor, "around a hundred times faster than a Node process on a container or virtual machine" diyor |
| **Modal Sandbox'a özel cold start** | Yok; genel ifade "Containers boot in about one second" |
| **Fly Machines'in Firecracker olduğu** | Machines *overview* sayfasında Firecracker adı **geçmiyor** ("fast-launching VMs"); yalnızca "well under a second" (başlatma) ve "low double digit seconds" (ilk oluşturma) |
| **Letta `run_code_with_tools`** | İsim **doğrulandı**: Letta'nın kendi blog yazısında geçiyor ([Programmatic Tool Calling with Any LLM](https://www.letta.com/blog/programmatic-tool-calling-with-any-llm/), 1 Ara 2025) — *"Agents with the `run_code_with_tools` tool attached can write scripts that directly invoke other tools"*. Dokümantasyonda ise `run_code` var: *"fresh environment per execution (no state persistence)"*. İkisi farklı tool olabilir ya da isim değişmiş olabilir; **kalıcılık açısından kritik olan**, Letta'nın belgelenmiş kod çalıştırıcısının çalıştırmalar arası durum TAŞIMADIĞInı açıkça söylemesi |
| **Arşiv dokümanındaki alıntı hatası** | `PTC_Daha_Hafif_Alternatifler_Arastirmasi.md` §2, kubernetes.io blogundan *"sub-second startup latency ... an improvement of up to ninety percent"* ve *"cold starts take ~4s+"* alıntılıyor. **Bu ifadeler o sayfada yok.** Sayfada olan: *"Starting a new pod adds about a second of overhead"*. Sayısal iddia **Google Cloud blogunda** ve farklı: *"300 sandboxes per second, per cluster, at sub second latency, with 90% of allocations completing in 200 milliseconds"*. Arşiv dokümanı düzeltilmeli |
| **OpenAI container ağ erişimi** | Code Interpreter kılavuzunda net bir "internet kapalı/açık" ifadesi bu sayfada bulunamadı |
| **Bizim 7.21 sn'lik ölçümün dağılımı** | Tek bir oturumda, `count_open_tickets()` ile ölçülmüş; tekrar ölçüm yok, varyans bilinmiyor |

---

## 7. Kaynaklar

### Anthropic (birincil, offline kopya mevcut)
- [Code execution tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool) — container reuse, 30 gün, ~5 dk checkpoint, `$OUTPUT_DIR`, 5 GiB RAM/disk, ağ kapalı, fiyatlandırma
- [Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling) — `allowed_callers`, container yaşam döngüsü, 90 sn hücre limiti, ~4 dk PTC timeout, alternatif uygulama modelleri
- [Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) — MCP'yi kod API'si olarak sunma, ara sonuçları dosyaya yazma, "skills" klasörü
- [Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)

### Cloudflare (Code Mode / izolat)
- [Code Mode: the better way to use MCP](https://blog.cloudflare.com/code-mode/) — MCP → TypeScript API, V8 isolate, Dynamic Worker Loader
- [Code Mode: give agents an entire API in 1,000 tokens](https://blog.cloudflare.com/code-mode-mcp/) — 2.500 endpoint / 1,17M token → ~1.000 token
- [How Code Mode works](https://developers.cloudflare.com/agents/tools/codemode/how-it-works/) — `globalOutbound: null`, Workers RPC, DO SQLite, transient executor
- [Code Mode API reference](https://developers.cloudflare.com/agents/tools/codemode/api-reference/) — `search`/`describe`/`saveSnippet`, 1.000.000 karakter limiti, 50 execution, 24 sa
- [Dynamic Workers](https://developers.cloudflare.com/dynamic-workers/)
- [How Workers works](https://developers.cloudflare.com/workers/reference/how-workers-works/) — izolat vs container/VM
- [Cloudflare Sandbox SDK](https://developers.cloudflare.com/sandbox/) — container tabanlı, DO ile kalıcı durum

### OpenAI
- [Code Interpreter tool](https://developers.openai.com/api/docs/guides/tools-code-interpreter) — `auto`/explicit container, 20 dk boşta sona erme, bellek katmanları

### Sandbox sağlayıcıları
- [E2B — Sandbox lifecycle](https://docs.e2b.dev/sandbox) · [Persistence](https://docs.e2b.dev/sandbox/persistence) · [Snapshots](https://docs.e2b.dev/sandbox/snapshots) · [Filesystem-only snapshots](https://docs.e2b.dev/sandbox/filesystem-only-snapshots) · [Volumes](https://docs.e2b.dev/volumes) · [Internet access](https://docs.e2b.dev/network/internet-access)
- [Modal — Sandboxes](https://modal.com/docs/guide/sandbox) · [Sandbox snapshots](https://modal.com/docs/guide/sandbox-snapshots) · [Volumes](https://modal.com/docs/guide/volumes) · [Security (gVisor)](https://modal.com/docs/guide/security) · [Cold start performance](https://modal.com/docs/guide/cold-start)
- [Vercel Sandbox](https://vercel.com/docs/sandbox) · [Understanding Sandboxes (Firecracker, persistence, firewall)](https://vercel.com/docs/sandbox/concepts)
- [Daytona — Sandboxes (<90 ms)](https://www.daytona.io/docs/en/sandboxes/) · [Isolation](https://www.daytona.io/docs/en/isolation) · [Persistence](https://www.daytona.io/docs/en/persistence)
- [Fly.io Machines overview](https://fly.io/docs/machines/overview/)
- [Letta — Built-in server tools (`run_code`)](https://docs.letta.com/guides/core-concepts/tools/builtin-tools)

### İzolasyon primitifleri
- [Firecracker](https://firecracker-microvm.github.io/) — <125 ms boot, <5 MiB overhead, 150 microVM/sn/host
- [gVisor documentation](https://gvisor.dev/docs/) — kullanıcı-uzayı kernel, syscall overhead
- [Deno — Security and permissions](https://docs.deno.com/runtime/fundamentals/security/) — varsayılan-yasak, `--allow-run`/`--allow-ffi` uyarısı
- [Pyodide](https://pyodide.org/en/stable/) — CPython → WebAssembly

### Kubernetes tarafı
- [Running Agents on Kubernetes with Agent Sandbox — kubernetes.io](https://kubernetes.io/blog/2026/03/20/running-agents-on-kubernetes-with-agent-sandbox/) — "about a second of overhead", `SandboxWarmPool`, gVisor/Kata
- [kubernetes-sigs/agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) — Sandbox CRD, kalıcı depo, pause/resume, warm pool
- [Bringing you Agent Sandbox on GKE and Agent Substrate — Google Cloud](https://cloud.google.com/blog/products/containers-kubernetes/bringing-you-agent-sandbox-on-gke-and-agent-substrate) — 300 sandbox/sn/cluster, %90 < 200 ms, Pod Snapshots, "bypass some of the limitations of Kubernetes"

### Proje içi
- `PTC_Artifact_Persistence_Arastirmasi.md` — depo katmanı (serileştirme, GC/TTL, güvenlik)
- `PTC_Error_Recovery_ve_Artifact_Persistence_Arastirmasi.md` — mevcut kodun tespiti
- `archive/egress-policy/PTC_Daha_Hafif_Alternatifler_Arastirmasi.md` — warm pool, Firecracker snapshot, WASM (§6'daki düzeltme notuyla birlikte okunmalı)
- `src/grounded_assistant/ptc/sandbox_runner.py`, `sandbox_image/entrypoint.py`, `k8s/sandbox/job-template.yaml`
