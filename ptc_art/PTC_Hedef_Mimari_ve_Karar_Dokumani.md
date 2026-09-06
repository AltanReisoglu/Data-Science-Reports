# PTC Artifact Persistence — Hedef Mimari ve Karar Dokümanı

**Tarih:** 2026-09-03 · **Durum:** karar taslağı, iki açık soru ekibi bekliyor

Bu doküman bir araştırma notu **değil**. Üç araştırma dokümanının ve bu oturumdaki
tartışmanın sentezi; ne inşa edeceğimizi ve neden onu seçtiğimizi söylüyor.

**Case:**
> PTC: artifact persistence (files, dataframes) for multi-step workflows —
> çok adımlı workflow'larda üretilen dosya, dataframe ve ara çıktıların kalıcı
> olarak saklanması; sonraki adımların aynı artifact'leri yeniden üretmeden
> kullanabilmesi.

**Dayandığı dokümanlar:**
- [PTC_Artifact_Persistence_Arastirmasi.md](PTC_Artifact_Persistence_Arastirmasi.md) — depo desenleri, serileştirme, GC, güvenlik
- [PTC_Calistirma_Ortami_ve_SOTA_Arastirmasi.md](PTC_Calistirma_Ortami_ve_SOTA_Arastirmasi.md) — izolasyon primitifleri, SOTA kalıcılık modelleri
- [PTC_Error_Recovery_ve_Artifact_Persistence_Arastirmasi.md](PTC_Error_Recovery_ve_Artifact_Persistence_Arastirmasi.md) — mevcut kod tespiti
- [archive/egress-policy/PTC_OpenShift_Uyumluluk_Arastirmasi.md](archive/egress-policy/PTC_OpenShift_Uyumluluk_Arastirmasi.md) — CNI göçü riski

§8'de bu dokümanların **hangi önerilerinin geçersizleştiği** yazıyor. Oraya bakmadan
eski dokümanlardaki tavsiyeleri uygulamayın.

---

## 1. Yeni kısıt: OpenShift

Ekip OpenShift kullanacak. Bu üç şeyi birden belirliyor.

**a) Cilium düşüyor — ve bu kazanç.** OpenShift kendi CNI'ı (OVN-Kubernetes) ile
kurulu geliyor. Çalışan bir cluster'ı Cilium'a geçirmek: 7 adım, **tüm node'ların
reboot'u**, "significant cluster downtime", ve geri dönüş yolu *"likely involves
cluster reinstallation"*; göç rehberinin kendisi işlemi *"officially unsupported"*
diyor. Egress artık case olmadığına göre bu riski almanın gerekçesi yok.
Yerine OVN-Kubernetes `NetworkPolicy` / `AdminNetworkPolicy` / `EgressFirewall`.

**b) İzolasyon bedavaya güçleniyor.** SOTA tablosunda PoC'nin yeri kötüydü: düz
`runc` — en zayıf izolasyon ailesi, üstelik en yavaş başlatma. OpenShift'in
**sandboxed containers**'ı (Kata Containers'ın Red Hat destekli sürümü, container
başına hafif VM) pod spec'inde `runtimeClassName` ile devreye giriyor. Güvenilmeyen
kod çalıştırmak bu ürünün ilan edilmiş amacı.

**c) Yeni iş: kodumuz Cilium'a bağımlı.** `get_denied_actions()` `hubble observe`
subprocess'i çağırıyor — OVN'de Hubble yok. Bu yüzden §7'deki 1. adım bir
optimizasyon değil, **taşınabilirlik şartı**.

Ayrıca `restricted-v2` SCC **rastgele UID** atıyor ve root'u engelliyor; imajda
sabit UID varsayımı olamaz, yazılabilir dizinler grup-yazılabilir olmalı.

---

## 2. Tasarımı belirleyen bulgu

Egress serbest kalınca "doğrudan sandbox → nesne deposu" mantıklı görünüyordu.
OpenShift'in kendi mekanizması bu kapıyı kapatıyor:

> **ObjectBucketClaim, bucket'a kilitli bir hesap üretir — oturuma kilitli değil.**

OBC bir bucket ve ona ait bir uygulama hesabı yaratıyor; hesap "yalnızca tek bir
bucket'a erişebiliyor" ve bağlantı bilgileri bir ConfigMap + Secret olarak
düşüyor. Ama bizim izolasyon sınırımız **oturum** olmak zorunda — çünkü özelliğin
bütün amacı çalıştırmalar arası paylaşım (§3). Aynı OBC secret'ını her sandbox'a
verirsek, Y oturumunun çalıştırması X oturumunun artifact'lerini okur.

Doğrudan erişimle bunu engellemenin yolları:
| Yol | Neden olmaz |
|---|---|
| Oturum başına OBC | Her oturumda bir Kubernetes objesi — yaratma gecikmesi, GC yükü |
| STS benzeri kapsamlı token | NooBaa tarafında zemin zayıf |

**Sonuç: erişim Tool Gateway üzerinden.** Dikkat — bu, dünkü dokümanın vardığı
sonucun aynısı ama **gerekçesi farklı**. Eski gerekçe "Cilium politikası değişmesin"
idi; o gerekçe öldü. Yeni gerekçe: **oturum kapsamını ucuza uygulayabileceğimiz
tek yer orası.**

### 2.1 Depo durable — bu, kapsam anahtarını bir soruna çeviriyor

**Karar (2026-09-03): artifact store durable bir depodur; PTC çalıştırmaları
artifact'leri oraya yazar.** Yani veri, çalıştırmadan da oturumdan da uzun yaşar.

Bu karar mevcut kodda bir uyumsuzluk açığa çıkarıyor. `session_id` şu an
`str(uuid.uuid4())` ile üretiliyor ve ömrü kısa:

| Yer | Ne zaman yeni uuid üretiliyor |
|---|---|
| `web/app.py:81` | **Her WebSocket bağlantısında** ("bağlantı başına bir oturum") |
| `cli.py:66` | Her CLI çağrısında |

Yani anahtar, aynı bağlantıdaki turlar arasında sabit (C okuması çalışır) ama
sayfa yenilenince değişiyor. Durable bir depoda bunun sonucu:
**artifact bucket'ta yaşamaya devam eder, ama erişilemez hale gelir** — okuyucusu
olmayan çöp birikir.

**Durable depo, kendisinden uzun yaşayan bir anahtar ister.**

| Seçenek | Değerlendirme |
|---|---|
| `session_id`'yi olduğu gibi bırak | Depo fiilen "uzun TTL'li cache" olur; "durable" adı yanıltıcı kalır |
| Üstte kalıcı bir kimlik (kullanıcı / workspace / proje) | Doğru çözüm, ama PoC'de böyle bir kavram (ve auth) yok |
| **İstemci kimliği geri gönderir** — Anthropic'in `container.id` modeli | **Önerilen.** Web UI `session_id`'yi `localStorage`'da tutar, yeniden bağlanınca gönderir; göndermezse temiz başlar |

Üçüncüsü hem küçük bir değişiklik, hem de dokümanın her yerinde tekrarlanan
ilkeyi birebir koruyor: **kalıcılık opt-in ve bir kimliğe bağlı.**

---

## 3. "Çok adımlı workflow" tam olarak ne — üç okuma

Bu ayrım daha önce hiçbir dokümanda yoktu ve tasarımı doğrudan belirliyor. Bizim
mimaride "adım" diye bir soyutlama yok; olan şey `run_ptc_code(code)` çağrısı —
yani bir Kubernetes Job, yani bir Python süreci.

| Okuma | Ne | Kalıcılık gerekiyor mu | Çözümü |
|---|---|---|---|
| **A** — tek script içindeki aşamalar | LLM tek script yazar; döngü, koşul, sıralı tool çağrıları | **Aktarım için hayır** (aynı bellekteler). Ama script patlarsa hepsi baştan koşar | İçerik-adresli **cache** |
| **B** — aynı turda ardıl çalıştırmalar | `MAX_SANDBOX_RUNS_PER_TURN = 2`; pratikte hata → düzeltme → tekrar | **Evet** — süreç sınırı gerçek | İsimle **handle** |
| **C** — turlar/oturumlar arası | Takip sorusu; yeni tur, yeni pod | **Evet** | İsimle **handle**, oturum kapsamında |

**Ve bir gerilim var:** PTC'nin kendi mantığı her şeyi tek script'e itiyor (token
ve gecikme kazancı oradan geliyor). Yani PTC iyi çalıştıkça B'deki sınır azalıyor,
"run 1'den run 2'ye dataframe geçirme" ihtiyacı sönüyor. Ama aynı hamle script'i
büyütüyor ve **tek bir hatanın imha ettiği iş büyüyor** — üstelik düzeltme için
tek atış hakkımız var (`MAX = 2`).

**Sonuç: pratikte en çok A işe yarayacak.** Bu yüzden depo iki arayüzle kurulur,
biri sonradan eklenen bir konfor değil.

---

## 4. Hedef mimari

```
┌── Sandbox Pod (efemer, her run yeni) ─────────────────────────┐
│  SCC restricted-v2 · rastgele UID · readOnlyRootFilesystem    │
│  runtimeClassName: kata            ← kullanılabilirse         │
│  /scratch  (emptyDir, sizeLimit, grup 0 yazılabilir)          │
│                                                                │
│  df.to_parquet("/output/tickets.parquet")   ← B/C: isimle      │
│  pd.read_parquet("/output/tickets.parquet")                    │
│  os.path.exists("/output/tickets.parquet") ← A: retry atlar    │
└───────────────────────────┬───────────────────────────────────┘
                            │ MCP — NetworkPolicy'nin izin verdiği tek hedef
                ┌───────────▼────────────┐
                │     Tool Gateway       │  oturum kapsamı BURADA
                │  handle ↔ URI          │  provenance · TTL · content_hash
                │  eşik: küçük = inline  │  format doğrulama (pickle RED)
                │        büyük = kısa    │
                │        ömürlü imzalı URL│
                └───────────┬────────────┘
                            │ S3 — OBC secret'ı YALNIZCA gateway'de
                    ┌───────▼────────┐
                    │  OBC / NooBaa  │  artifacts/{session}/{run}/...
                    │  lifecycle TTL │
                    └────────────────┘
```

### 4.1 Çalıştırma katmanı

Efemer pod **korunuyor** — ve bu bir taviz değil. Cloudflare Code Mode, Letta
`run_code` ve varsayılan haliyle Anthropic aynısını yapıyor. Kalıcılık runtime'ın
dışında olduğu için "her run temiz ortam" iddiası bozulmuyor.

| Ayar | Değer | Gerekçe |
|---|---|---|
| Ağ | OVN `NetworkPolicy`, tek izinli hedef Tool Gateway | Cilium'un yerine, göç riski olmadan |
| Runtime | `runtimeClassName: kata` (varsa) | `runc` → microVM ailesi; tek alan |
| Kök FS | `readOnlyRootFilesystem: true` | SCC ile uyumlu yerleşik desen |
| Scratch | `emptyDir` + `sizeLimit` | Aşılırsa pod tahliye edilir — kota bedava |
| UID | Sabitlenmez, SCC atar | `restricted-v2` rastgele UID veriyor |

**Uyarı:** Kata izolasyonu güçlendirirken **başlatmayı yavaşlatır** (VM boot >
container boot). Yani Kata alınırsa §7'deki 1. adım daha da önem kazanır, ve hız
acıtırsa warm pool gündeme gelir.

### 4.2 Depo katmanı

Argo'nun "artifact repository" deseni, OpenShift'te OBC olarak. Tek bucket,
oturum prefix'i. Secret yalnızca gateway'de; **sandbox depoyu hiç görmez**, elinde
sadece opak `art_7f3a…` vardır (capability / unforgeable reference deseni).

### 4.3 Taşıma — Airflow'un eşik kuralı, transport'a uygulanmış

Airflow'un `xcom_objectstorage_threshold` fikri: küçükse kontrol düzleminde,
büyükse depoda. Bizde:

- **Eşik altı**: artifact MCP üzerinden inline geçer.
- **Eşik üstü**: gateway **kısa ömürlü, tek kullanımlık, oturuma bağlı** imzalı
  URL üretir.

Böylece §6.3'teki presigned URL sakıncası (süresiz, sahibi kim olursa olsun
çalışan bearer token) ortadan kalkar; ama megabaytları MCP'den geçirmek zorunda
da kalmayız.

### 4.4 İki arayüz, tek depo

```python
df.to_parquet("/output/acik_ticketlar.parquet")   # B/C — süreç sınırını aşar
df = pd.read_parquet("/output/acik_ticketlar.parquet")

if not os.path.exists("/output/acik_ticketlar.parquet"):          # A — retry atlar
    pahali_tool_dongusu().to_parquet("/output/acik_ticketlar.parquet")
```

`cached`, Metaflow'un content-addressing'ini Anthropic'in efemer-runtime
kısıtlarına taşımaktan ibaret. Prefect'in cache key'i, Flyte'ın dataframe
içeriğini hash'lemesi, Nextflow'un task hash'i aynı mekanizma.

### 4.5 Format

| Format | Kullanım |
|---|---|
| **Parquet** | Varsayılan — en küçük, şemalı, arşiv |
| **Arrow IPC** | Hızlı devir gerekiyorsa — tüm Arrow tiplerini korur |
| **pickle** | **Gateway tarafından reddedilir** |

pickle'ın deserialization'ı tasarım gereği kod çalıştırır (CWE-502). Artifact'i
yazan taraf **LLM'in ürettiği kod** olduğu için bu teorik bir risk değil.
Dokümanda yasak yazmak yetmez — kodda uygulanır.

### 4.6 Metadata ve lineage

Her artifact'in yanında: `artifact_id`, `uri`, `session_id`, üreten `run_id`,
`content_hash`, şema/dtype, boyut, `created_at`, `ttl`. `Trace`'e işlenir —
Kubeflow'un MLMD'de yaptığının küçük hâli.

### 4.7 Yaşam döngüsü

Depo durable olduğu için **varsayılan saklamaktır**; TTL istisnadır. Ama GC
ortadan kalkmıyor — durable depo da yönetilmezse sınırsız büyür.

Argo'nun artifact-**düzeyi** GC stratejisini örnek alıyoruz (`OnWorkflowCompletion`
/ `OnWorkflowDeletion` / `Never`): her artifact kendi saklama politikasını taşır.

| Artifact türü | Politika |
|---|---|
| Nihai çıktı (kullanıcıya dönen) | Kalıcı — silinene kadar |
| Adımlar arası ara sonuç | Kalıcı ama sahibi (kapsam anahtarı) ile birlikte yaşar |
| `cached()` girdileri | TTL'li — yeniden üretilebilir oldukları için ucuz kayıp |

Bucket'ta expiration kuralı ve `AbortIncompleteMultipartUpload` (yarım kalan
parçalar listelemede görünmez ama ücreti işler) her koşulda açık olmalı.

Kıyas: Anthropic'te container 30 gün ömürlü ama **Files API dosyaları silinene
kadar** kalıyor — ortam ömrü ile veri ömrünün ayrı sayılar olmasının tam örneği.

### 4.8 Güvenlik

| Risk | Nerede karşılanıyor |
|---|---|
| Oturumlar arası sızıntı (ağ politikasının göremediği kanal) | Gateway'de oturum kapsamı |
| Artifact'e gömülü prompt injection | İçerik LLM bağlamına **veri olarak** etiketli ve sınırlı girer. **Durable depo bu riski büyütür**: enjeksiyonun yaşadığı pencere artık oturum değil, saklama süresi (§6.2'deki cross-session stored injection sınıfı) |
| Deserialization ile kod çalıştırma | Format doğrulaması, pickle reddi |
| Eşzamanlı yazım / "last write wins" | **Oluşmuyor** — tek yazıcı (gateway) + değişmez, içerik-adresli nesneler |
| Depo şişmesi | TTL + lifecycle + `sizeLimit` |

Modal'ın Volumes için verdiği "last write wins ... any data the last writer didn't
have when committing will be lost" uyarısı bizde yapısal olarak ortaya çıkmıyor.

### 4.9 İddianın nitelenmesi

Eski hâli: *"sandbox hiçbir zaman kalıcı iz bırakmaz."*

Yeni hâli: ***çalıştırma ortamı*** *iz bırakmaz; artifact'ler onaylı kanaldan
geçer, bir kimliğe bağlıdır ve TTL'lidir.* Anthropic'in kendi formülasyonu bu —
kalıcılık opt-in ve `container.id`'ye bağlı, gönderilmezse her istek temiz başlar.

---

## 5. Kararlar

| Karar | Durum | Gerekçe |
|---|---|---|
| Efemer pod korunacak | **Verildi** | SOTA çoğunluğu; iddia bozulmuyor |
| Cilium bırakılacak | **Verildi** | Göç riski belgelenmiş; gerekçe ölmüş |
| ~~Erişim gateway aracılı~~ → **ayrı Artifact Service** | **Revize** (2026-09-04) | Aracılık korunuyor, aracı DEĞİŞTİ — bkz. §5.1 |
| Artifact taşıması akışlı HTTP | **Verildi** (2026-09-04) | base64+MCP %33 şişme + iki uçta tam tampon |
| Prefetch → manifest + tembel okuma | **Verildi** (2026-09-04) | O(hepsi) maliyeti 512Mi `/output`'u patlatıyordu |
| Workflow state kalıcı | **Verildi** (2026-09-04) | SQLite checkpointer; Postgres'e geçiş ortam değişkeni — bkz. §5.3 |
| TTL reaper (CronJob) | **Verildi** (2026-09-04) | Şema/silme kodu vardı, çalıştıran yoktu — bkz. §5.2 |
| **Depo durable** — PTC'ler artifact'leri oraya yazar | **Verildi** (2026-09-03) | Kullanıcı kararı |
| Kapsam = oturum, run değil | **Verildi** | Özelliğin amacı çalıştırmalar arası paylaşım |
| **Kapsam anahtarı nasıl kalıcılaşacak** | **Verildi** (2026-09-04) | İstemci kimliği geri gönderir (Anthropic `container.id` modeli) — bkz. §5.3 |
| Parquet/Arrow; pickle yasak | **Verildi** | CWE-502; yazan taraf LLM |
| İçerik-adresli anahtarlama | **Verildi** | A okuması pratikte baskın (§3) |
| Eşik kuralı (inline / imzalı URL) | **Verildi** | Airflow deseni; §6.3'ü etkisizleştirir |
| Kata (`runtimeClassName`) | **Ekibe soru** | Destek durumu OCP sürümüne göre değişiyor |
| ODF/OBC kurulu mu | **Ekibe soru** | Yoksa MinIO + `anyuid` SCC'si gerekir |
| Warm pool | **Ertelendi** | Hız acıtırsa (GKE: %90 tahsis <200 ms) |
| Uzun ömürlü sandbox / REPL durumu | **Ertelendi** | Serileştirilemeyen bir ihtiyaç çıkarsa |
| K8s'i bırakıp microVM/izolat | **Kapsam dışı** | Ekip zaten OpenShift'te |

### 5.1 Neden gateway aracılığı "ayrı servis"e revize edildi (2026-09-04)

Önceki karar "erişim gateway aracılı" idi ve **aracılık ilkesi aynen duruyor** —
sandbox'ın hâlâ MinIO'ya rotası yok, kimlik bilgisi yok, presigned URL yok.
Değişen şey aracının kim olduğu.

Bir ara adımda "büyük dosyalar için presigned URL" düşünüldü ve **elendi**:
yanlış teşhisti. Darboğaz aracının varlığı değil, baytların MCP çağrısı içinde
base64 taşınmasıydı — %33 şişme ve iki uçta tam tampon. Akışlı HTTP bunu
zaten çözüyor, dolayısıyla depoya rota açmanın (ağ politikasının göremediği
ikinci bir çıkış) getirisi yok.

İkinci gerekçe yetki ayrımı. Tek pod üç işi birden yapıyordu: tool proxy'si,
kayıt defteri, MinIO kimlik bilgisi taşıyıcısı — araştırma dokümanının §26'sının
"önerilmeyen yaklaşım" dediği şey. Şimdi:

|  | Tool Gateway | Artifact Service |
|---|---|---|
| İnternet | ✓ (3 onaylı FQDN) | ✗ |
| Artifact deposu | ✗ | ✓ |
| Kapsam imza sırrı | ✗ | ✓ |

Tek bir workload'ın ele geçirilmesi ikisini birden vermiyor. OpenAI'nin Ağustos
2026 olayında Artifactory'nin oynadığı köprü rolüne karşı yapısal önlem.

**Doğrulandı** (canlı cluster, 2026-09-04): gateway pod'undan `minio:9000`
zaman aşımına düşüyor; ortamında `AWS_*`/`BUCKET_*` yok; sandbox MinIO'yu ne
adla ne IP ile çözebiliyor; pickle 415 ile reddediliyor; başka bir workflow'un
artifact'i 404 görünüyor.

**Bu geçişte bulunan hata** — politikaya DNS izni koymamıştım, gerekçe olarak da
"FQDN kuralı yok, DNS gerekmez" yazmıştım. Yanlış: `toEndpoints` ağ yolunu açar
ama minio istemcisi önce `BUCKET_HOST`'u çözmek zorunda. İlk istekte sessizce
asılı kaldı (default-deny düşürür, reddetmez), tek worker'lı uvicorn bloke
olunca readiness probe da düştü ve sorun iki katman yukarıda görünür oldu.
2026-09-03'te gateway→MinIO yolunda yaşananın aynısı.

### 5.2 TTL reaper — zamanlayıcı ayrı, yazıcı tek

`ttl_seconds` şemada, `expired()` metadata'da, `delete()` serviste vardı — ama
üçünü çağıran kimse yoktu. Yani TTL yazılıydı, çalışmıyordu.

Reaper'ı ayrı bir pod yapmak **mümkün değildi**: kayıt defteri SQLite ve PVC
`ReadWriteOnce`, yani tek yazıcı. İkinci bir pod DB'yi kendisi açsaydı ya
mount edemezdi ya da eşzamanlı yazıcı olup bozulma riski doğururdu. İş bölümü
bu yüzden şöyle:

- **CronJob** zamanlayıcı — yeniden deneme, geçmiş, gözlemlenebilirlik
- **Servis** tek yazıcı — `POST /admin/reap` işi kendi süreci içinde yapar

Postgres'e geçilince kısıt kalkar ve reaper işi kendisi yapabilir.

Yetki ayrı bir jetonda (`PTC_ADMIN_TOKEN`). Kapsam jetonu bir workflow'a bağlı,
reaper ise workflow'lar arası süpürüyor; ayrıca sandbox'ın jetonunun buraya
**yetmemesi** gerekiyor — LLM'in ürettiği kodun toplu silme tetikleyebilmesi,
emniyet ağı olarak kurduğumuz kalıcılığı tek çağrıda geri alabilirdi. Sır
tanımlı değilse uç nokta 503; "açık bırakılmış" bir hâli yok.

**Doğrulandı** (canlı, 2026-09-04): TTL'li ve TTL'siz iki artifact üretildi,
reaper `{"aday": 1, "silinen": 1}` döndü, TTL'siz olan kaldı. Sandbox kendi
kapsam jetonuyla `/admin/reap`'i çağırdığında **401** aldı.

Bu testte dedup davranışı da doğrulanmış oldu: iki artifact'in içeriği aynı
olduğu için ikincisi birincinin baytını gösteriyordu. Reaper birinciyi silerken
paylaşılan baytı **korudu** ve ikincisi hâlâ okunabildi. (İlk bakışta MinIO'da
yetim bayt gibi görünüyordu — değildi; anahtar ilk yükleyenin adını taşıyor.)

### 5.3 Oturum kalıcılığı — iki eksik, tek iş

§2.1 bir sorunu tespit etmişti ve açık bırakmıştı. Kapatıldı.

`session_id` TEK bir uuid'ydi ve İKİ yere birden gidiyordu:

    session_id ──> thread_id    (konuşma hafızası → InMemorySaver)
               └─> workflow_id  (artifact kapsamı → MinIO + kayıt defteri)

Her bağlantıda/CLI çağrısında yeniden üretiliyordu. Konuşma hafızasının gitmesi
beklenen bir şeydi; **asıl sorun ikincisiydi**: artifact'ler MinIO'da ve kayıt
defterinde sağ kalıyor ama onları gösteren `workflow_id` bir daha asla
üretilmediği için erişilemez hale geliyorlardı. Kalıcı bir depoya yazıp okuma
anahtarını çöpe atmak.

Bu yüzden iki düzeltme tek iş:

**(a) Kalıcı anahtar.** Web'de kimlik `localStorage`'da, WebSocket'e
`?session=` ile geliyor; CLI'de `--session` bayrağı. Biçim UUID'ye kilitli —
bu değer S3 anahtarının parçası olduğu için serbest metin kabul edilemez.
Vermeyene temiz oturum açılıyor: **kalıcılık opt-in**.

**(b) Kalıcı state.** `InMemorySaver` yerine arka ucu config'den seçilen bir
checkpointer (`graph.build_checkpointer`):

    PTC_CHECKPOINT_DSN → AsyncPostgresSaver (üretim)
    PTC_CHECKPOINT_DB  → AsyncSqliteSaver   (varsayılan, metadata ile aynı duruş)
    ikisi de yoksa     → InMemorySaver      (test / tek atış)

**Saver'lar ASYNC olmak zorunda** — 2026-09-04'te üretimde bulundu. Hem web hem
CLI `agent.ainvoke()` yoluna giriyor; senkron `SqliteSaver` orada açık hata
veriyor (*"The SqliteSaver does not support async methods"*). İlk testler bunu
kaçırmıştı çünkü saver'ı doğrudan senkron `put`/`get_tuple` ile sınıyorlardı —
uygulamanın hiç kullanmadığı yol.

Async saver'lar kurulurken çalışan bir event loop istiyor, `build_agent` ise
bilerek loop'suz bir thread'de çalışıyor (içinde `asyncio.run()` var). Bu yüzden
checkpointer çağıran tarafta (loop içinde) kurulup `build_agent`'a geçiriliyor.
Bağlantı, WebSocket kapanınca kapatılıyor — aksi hâlde her sekme bir aiosqlite
thread'i sızdırıyordu.

Metadata için verdiğimiz kararın aynısı: SQLite, çok replikaya çıkana kadar.
Postgres'e geçiş kod değil ortam değişkeni.

**Doğrulandı** (2026-09-04): iki AYRI süreçte, aynı `--session` ile — birincisi
`oturum.raporu`'nu sakladı, ikincisi (süreç öldükten sonra) onu okudu ve
`{"bulundu": True, "toplam": 350}` döndü. Kontrol: kimlik verilmeyince yeni bir
workflow açıldı ve `{"bulundu": False}` geldi — kapsam izolasyonu bozulmuyor.

**Sınır:** bu kimlik doğrulama DEĞİL. Uuid'yi bilen herkes o oturumun
artifact'lerini okuyabilir. Tahmin edilemez olduğu için PoC'de yeterli, üretimde
gerçek auth'a bağlanmalı.

---

## 6. Ekibe iki soru

1. **OpenShift Data Foundation kurulu mu?** Kuruluysa artifact deposu bir
   `ObjectBucketClaim`. Değilse MinIO'yu iş yükü olarak koşarız — ama imaj sabit
   UID istediği için `anyuid` SCC'si ya da uyumlu bir imaj gerekir.
2. **Sandboxed containers operator'ı kullanılabilir mi, hangi OCP sürümü?**
   Kullanılabilirse `runtimeClassName` ile izolasyon ailesi atlıyoruz.

Her iki cevap da §7'nin 1. adımını **etkilemiyor** — o adım her koşulda doğru.

---

## 7. İnşa sırası

**1. Ölçüm hijyeni** — her koşulda doğru, bugün yapılabilir, iki sorudan bağımsız.
Kodun kritik yolunda üç kalem var:

```
_POLL_INTERVAL_SECONDS = 1.0     → ortalama ~0.5 sn, en kötü 1 sn boşuna beklenir
get_denied_actions()             → İKİ hubble subprocess'i, her biri timeout=10,
                                   ve _cleanup'tan ÖNCE, sonucu beklenerek
_cleanup(...)                    → ttlSecondsAfterFinished zaten varken senkron silme
```

Ölçülen ~7.21 sn'nin (Job → ilk `tool_call` 3.08 sn) kalanının kayda değer kısmı
bunlar. Beklenen kazanç 1-3 sn, ve Hubble bağımlılığının sökülmesi OpenShift'e
taşınmanın önkoşulu.

**2. Yazılabilir zemin** — `emptyDir` scratch, `sizeLimit`, SCC uyumu (rastgele
UID, grup 0). *Not: 2026-09-02'de hazırlanan `ptc-sandbox:artifacts` /
`tool-gateway:artifacts` imajları bu açıdan test edilmedi.*

**3. Depo ve iki arayüz** — gateway'de `put_artifact` / `get_artifact` / `cached`;
entrypoint'te proxy fonksiyonlar (`_make_sync_tool` deseni aynen kullanılabilir);
`ALLOWED_TOOLS` hem `entrypoint.py`'de hem `agent/tool_policy.py`'de senkron.

**4. TTL, lineage, `Trace` kaydı.**

---

## 8. Eski dokümanların hangi kısımları geçersiz

Üç doküman farklı kısıtlar altında yazıldı; çelişkileri burada çözülüyor.

| Doküman / bölüm | Durum |
|---|---|
| `PTC_Artifact_Persistence_Arastirmasi.md` §8.1 — *"Cilium politikası hiç değişmez, o yüzden gateway aracılı"* | **Sonuç doğru, gerekçe ölü.** Yeni gerekçe §2 |
| Aynı doküman §6.3 — presigned URL | **Yarısı geçerli.** "Egress'i deler" kısmı düştü; "süresiz bearer token" kısmı duruyor — §4.3 bunu kısa ömürlü + tek kullanımlık yaparak karşılıyor |
| Aynı doküman §8.3 — sadece `put/get` öneriyor | **Eksik.** `cached` eklendi; teşhis A'ydı, reçete B'ydi (§3) |
| Aynı doküman §6.5 — MinIO + STS AssumeRole + prefix | **OpenShift'te gereksiz** — OBC bunu hazır veriyor |
| `PTC_Calistirma_Ortami_ve_SOTA_Arastirmasi.md` §5.4 — "cluster'ı koru" | **Doğrulandı ve konusuz kaldı** — ekip zaten OpenShift'te |
| Aynı doküman §5.3 Seçenek 3 — K8s'i bırak | **Kapsam dışı** |
| `archive/egress-policy/PTC_Daha_Hafif_Alternatifler_Arastirmasi.md` §2 | **Hatalı alıntı içeriyor.** kubernetes.io'ya atfedilen *"cold starts take ~4s+"* ve *"sub-second ... ninety percent"* ifadeleri o sayfada **yok**; sayfadaki tek rakam *"Starting a new pod adds about a second of overhead"*. Ham sayfa üzerinden doğrulandı |
