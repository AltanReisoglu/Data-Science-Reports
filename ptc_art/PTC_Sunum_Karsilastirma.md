# PTC Artifact Persistence — Karşılaştırmalı Sunum

**Her sayfa bir özellik. Her sayfada bir tablo. Sonunda "biz neredeyiz".**

Tarih: 2026-09-07 · Detaylar: [PTC_Piyasa_Mentaliteleri.md](PTC_Piyasa_Mentaliteleri.md)

---

## Sayfa 1 — Problem

> Sandbox'ın **ölmesi** güvenlik için gerekli.
> Ürettiğinin **kalması** iş için gerekli.

| | Ne olur | Sonuç |
|---|---|---|
| Sandbox yaşarsa | Bir çalıştırma diğerine bulaşır | İzolasyon yok |
| Artifact ölürse | 40 sn'lik iş her turda tekrarlanır | Kullanılamaz |

**Kural:** Sandbox'ın yaşam süresi, ürettiğinin yaşam süresini belirlememeli.

**Karıştırılmaması gereken üçlü:**

| Ne | Örnek | Nerede durmalı |
|---|---|---|
| Geçici dosya | Ara hesap, cache | Sandbox diski — **ölmesi istenir** |
| Artifact | 40 sn'de üretilen tablo | Kalıcı depo |
| State | "Konuşmada nerede kaldım" | Ayrı, ama o da kalıcı |

---

## Sayfa 2 — İzolasyon: kod nerede çalışıyor

| Yöntem | Ne demek | Kim kullanıyor | Güç |
|---|---|---|---|
| Container (runc) | Kernel paylaşılır | **BİZ** | En zayıf |
| V8 isolate | JS motoru içinde bölge | Cloudflare Code Mode | Dar ama sıkı |
| gVisor | Araya sahte kernel | Google GKE Agent Sandbox | Orta |
| Kata / microVM | Pod başına kendi kernel'i | Red Hat önerisi, E2B, Vercel | Güçlü |
| Hyper-V | Donanım sanallaştırma | Microsoft | Güçlü |

**Sonuç:** Bu listede en zayıf olan biziz. Red Hat AI-üretimi kod için açıkça
Kata öneriyor. Kod değişikliği gerektirmez — node seviyesinde önkoşul.

---

## Sayfa 3 — Sandbox ömrü

| Model | Kim | Süre |
|---|---|---|
| Her çağrıda yeni, saklanmaz | **BİZ** | 3.14 sn |
| Efemer + cooldown'da yok edilir | Microsoft | Havuzdan ms |
| Varsayılan yeni, id ile canlanır | Anthropic | **30 gün** (5 dk'da checkpoint) |
| Adlandırılmış, uzun ömürlü | Google Agent Engine | **14 gün** |
| Süre sınırlı oturum | AWS | 15 dk – 8 saat |
| Kalıcı workspace + snapshot | OpenAI | — |

**Ayrım:** Anthropic *container'ı* saklıyor (değişkenler, dosyalar, hatta
yorumlayıcı durumu). Biz *artifact'i* saklıyoruz. Farklı problem, farklı çözüm.

---

## Sayfa 4 — Depoya erişim: mount mu, API mi

| Yaklaşım | Yazınca ne olur | Kim |
|---|---|---|
| **Mount** | Yol = bucket. Yazdığın an gider | E2B, Modal, Daytona, Vercel, Cloudflare, Fly.io |
| **API/SDK** | Sıradan klasör; bir bileşen taşır | **BİZ**, Anthropic, OpenAI, Red Hat |

**Red Hat ne diyor:**
> *"…you must create a local client … by using an AWS SDK such as Boto3."*

FUSE mount ve CSI sürücüsü dokümanda **hiç geçmiyor**.

**Mount'un bedeli:**

| | Mount | Bizimki |
|---|---|---|
| Çok büyük dosya (>512Mi) | ✓ | ✗ |
| Kısmi/akan yazma | ✓ | ✗ |
| **Kayıt defteri** | ✗ | ✓ |
| **İçerik doğrulaması** | ✗ | ✓ |
| Kapsam granülaritesi | Bucket/prefix | **Çalıştırma başına** |

---

## Sayfa 5 — Anahtar kimde

| Konum | Kim | Risk |
|---|---|---|
| **Sandbox'ın içinde** | E2B (`/root/.passwd-s3fs`), Vercel (düz), Daytona (external) | Denetlenmemiş kod anahtarı görüyor |
| Sandbox'ta ama IAM ile dar | AWS (execution role) | Rol kadar |
| **Dışarıda, aracıda** | Cloudflare (binding/proxy), Vercel (proxy) | Mount var, anahtar yok |
| **Hiç yok** | **BİZ**, Anthropic, OpenAI, Fly.io | — |

**Bizde:** sandbox'ta S3 anahtarı yok, MinIO'ya rota yok, DNS yok.
**Doğrulandı:** `gaierror` + `ConnectionRefusedError`.

---

## Sayfa 6 — Kayıt defteri var mı

| Var | Yok |
|---|---|
| **BİZ** (`artifact_id`, tip, soy, TTL, hash) | E2B, Modal, Daytona, Vercel |
| Anthropic (Files API) | Cloudflare, Fly.io, Microsoft |
| Red Hat / KFP (MLMD) | AWS (denetim CloudTrail'de) |

**Araştırmanın en keskin bulgusu:**
> Mount eden sağlayıcıların **hiçbirinde** artifact registry yok. Yazılan
> dosyanın `artifact_id`'si, soyu, TTL'i yok — sadece bir S3 anahtarı var.
> Dokümanlar "persistent data access" diyor; **hiçbiri "artifact" demiyor.**

**Kural:** Kayıt defteri, yalnızca yazma yolu **bir bileşenden geçtiğinde**
ayakta kalıyor. Doğrudan mount = "sadece bucket".

---

## Sayfa 7 — Ağ duruşu

| Duruş | Kim |
|---|---|
| Tamamen kapalı | Anthropic (*"Completely disabled for security"*) |
| **Varsayılan reddet + allowlist** | **BİZ**, Red Hat/OpenShell, Google GKE |
| Açık, yapılandırılabilir | AWS ("network modes") |
| Opsiyonel kontroller | Microsoft |

**Red Hat'in ifadesi:**
> *"The default posture is deny-all. In practice, you write a policy that
> allowlists exactly the endpoints your agent needs."*

**Bizde iki servis ayrı:** Tool Gateway internete çıkar, depoya çıkamaz.
Artifact Service depoya çıkar, internete çıkamaz. Tek workload'ın ele
geçirilmesi ikisini birden vermiyor.

---

## Sayfa 8 — Yazmayı kim tetikliyor

| Mekanizma | Kim | LLM bilmek zorunda mı |
|---|---|---|
| `$OUTPUT_DIR` süpürme | Anthropic | **Hayır** |
| `/mnt/data` | OpenAI | **Hayır** |
| Workspace dizini | OpenHands | **Hayır** |
| `.path` → launcher kopyalar | KFP / OpenShift AI | **Hayır** |
| `/output` süpürme | **BİZ** | **Hayır** |
| Açık RPC zorunlu | Cloudflare Code Mode *(dosya sistemi yok)* | Evet |

**Dokuz sistemin dokuzu da bir DOSYA YOLU.** Hiçbiri sandbox'taki koda
artifact fonksiyonu sunmuyor.

**2026-09-06: bizdeki açık API kaldırıldı.** `put_artifact`/`get_artifact`/
`cached` piyasada emsalsizdi ve canlı hataların hepsi o yüzeydeydi.

| | Önce | Şimdi |
|---|---|---|
| Yazma | `put_artifact(df, name=...)` | `df.to_parquet("/output/x.parquet")` |
| Okuma | `get_artifact("x")` | `pd.read_parquet("/output/x.parquet")` |
| Keşif | `list_artifacts()` | `os.listdir("/output")` |
| Retry atlama | `cached(...)` | `if os.path.exists(...)` |

**Süpürme = bucket'a kopyalama:**
`/output/x.parquet` → HTTP akış → Artifact Service → S3 PUT → bucket

---

## Sayfa 9 — KEŞİF: ajan çekeceğini nasıl anlıyor

**Bu, oturum boyunca en çok kafa karıştıran soruydu.**

| # | Desen | Nasıl | Kim |
|---|---|---|---|
| 1 | Sadece tool tarifi | Model çağırmayı *seçmek* zorunda | *(eskiden biz)* |
| 2 | Referans otomatik context'te | `file_id` tool sonucunda döner | Anthropic |
| 3 | Dosya sistemi + `ls` | Sandbox yaşıyorsa model bakar | Anthropic, Google, OpenHands |
| 4 | **İsimler prompt'a enjekte** | İsimler talimatlarda, içerik talep üzerine | **Google ADK**, **BİZ** |
| 3+4 | **İkisi birden** | Manifest promptta **ve** `os.listdir("/output")` çalışıyor | **BİZ** (2026-09-06) |
| 5 | Semantik arama | Vektör deposunda `file_search` | Llama Stack (RAG) |
| — | **Keşif YOK** | DAG statik, girdi bağlanmış | KFP, Argo, Airflow, Tekton |

**Desen 4'ün üç kuralı:**
1. İsimler her zaman context'te — ucuz, model unutamaz
2. İçerik talep üzerine — pahalı olan sadece istendiğinde
3. İçerik geçmişe kalıcı yazılmaz — context şişmez

---

## Sayfa 10 — Klasik pipeline vs ajan: neden farklı

| | Klasik pipeline | Ajan dünyası |
|---|---|---|
| DAG | İnsan önceden yazar | LLM o an icat eder |
| 5. adımın girdisi | **Bağlanmış** | **Keşfedilmeli** |
| Kim çözer | Driver, adım başlamadan | Modelin kendisi |
| Örnek | `adim5(girdi=adim1.ciktilar["features"])` | `os.listdir("/output")` |
| Kim | Airflow, Argo, KFP, Tekton | Anthropic, Google ADK, biz |

**Tek cümle:** Klasik pipeline'da 5. adım bir şey *anlamaz* — kendisine söylenir.
Ajan dünyasında sormak zorundadır, çünkü kendisi de o an icat edilmiştir.

---

## Sayfa 11 — OpenShift ne öneriyor

| Konu | OpenShift'in cevabı |
|---|---|
| Nesne deposu | **Çekirdekte YOK** — depolama dokümanı baştan sona PV/PVC/CSI |
| Ama AI ürünü | **S3 zorunlu** — *"You have an existing S3-compatible object storage bucket"* |
| Erişim | **SDK (boto3)**, mount değil |
| Kimlik bilgisi | Secret → pod ortamı (`AWS_S3_ENDPOINT`, `AWS_S3_BUCKET`…) |
| Artifact taşıma | **Launcher** `.path` ↔ `.uri` kopyalıyor |
| Depo kökü | `pipeline_root` — **3 düzeyde yapılandırılabilir** |
| Artifact tipleri | MLMD şema başlıkları (`system.Dataset`…) |
| Kapsam | **Namespace** düzeyinde (`kfp-launcher` ConfigMap) |
| Tekton farklı | Workspace → **PVC**, nesne deposu değil |
| Güvenilmeyen kod | **Kata** öneriliyor |
| **Keşif** | **Yerleşik cevabı YOK** ← |

**Neden keşif cevabı yok:**

| | Artifact kavramı | Ajan kavramı |
|---|---|---|
| KFP / DSP | ✓ | ✗ |
| Llama Stack | ✗ (RAG belgeleri) | ✓ |

Bizim durumumuz tam bu boşlukta. Depolama için OpenShift'i kopyaladık,
keşif için Google ADK'yı.

---

## Sayfa 12 — Hız

| Sistem | Açılış | Nasıl |
|---|---|---|
| Cloudflare | milisaniyeler | Isolate hafif, havuza gerek yok |
| Microsoft | milisaniyeler | **Warm pool** |
| Google GKE | < 1 sn | Warm pool + snapshot = "instant-on" |
| **BİZ** | **3.14 sn** | Havuz yok |

**Bizim 3.14 sn'nin dağılımı:** 1.62 sn pod başlatma + 1.49 sn süreç açılışı.

**Yani warm pool'un tavanı 1.6 saniye.** Ölçtük, karmaşıklığa değmedi.

---

## Sayfa 13 — Biz neredeyiz: özet

| Boyut | Kimle aynı |
|---|---|
| PTC tezi (kod yaz, tool çağırma) | **Cloudflare** |
| Ağ (kapalı + allowlist) | **Anthropic**, Red Hat, Google GKE |
| Depoya erişim (SDK, mount yok) | **Red Hat**, Anthropic, OpenAI |
| Çıktı yakalama (`/output`) | **Anthropic**, OpenAI |
| Launcher deseni | **Red Hat / KFP** |
| Kayıt defteri + tipler + kök | **Red Hat / KFP (MLMD)** |
| Keşif (isimler prompt'ta) | **Google ADK** |
| Sandbox ömrü (efemer) | Microsoft |
| **LLM yüzeyi (dosya, API yok)** | **hepsi** — KFP, Anthropic, OpenAI, MS |
| **Kapsam (tenant)** | **Red Hat / KFP** — `pipeline_root` paylaşımlı |
| **İzolasyon** | **Kimse — bizimki daha zayıf** |
| **Girdi yerleştirme (kod başlamadan)** | **Argo `init` / KFP `launcher`** |
| Çapraz-çalıştırma erişimi (açık çağrı) | **KFP** (açık URI) · **Google ADK** (`load_artifact`) |

**Üç cümle:**
1. Omurga tartışmasız SOTA — tezi Cloudflare'den, veri modelini Anthropic'ten,
   platform desenini Red Hat'ten aldık.
2. İzolasyonda herkesin gerisindeyiz (düz container). Warm pool yok ama
   kazancını ölçtük: ≤1.6 sn.
3. Emsalsiz olan **üç şeyi de** bıraktık: LLM'e artifact API'si ve
   çalıştırma başına kapsam (2026-09-06), sonra şeffaf tembel okuma
   (2026-09-07) — yerine Argo/KFP'nin "girdiyi kod başlamadan yerleştir"
   deseni geçti. Hataların hepsi bizim icat ettiğimiz yerlerde çıkıyordu.

---

## Sayfa 14 — Ne değişti (2026-09-04 → 09-06)

| Değişiklik | Öncesi | Sonrası |
|---|---|---|
| Artifact servisi ayrıldı | Tool Gateway'de, base64+MCP | Kendi pod'unda, akışlı HTTP |
| Prefetch kapsamı daraltıldı | Tenant'ın tamamı iniyordu, O(hepsi) | Çalıştırmaya kapsanmış yerleştirme, azami 35 KiB |
| TTL reaper | Şema vardı, çalıştıran yoktu | Saat başı CronJob |
| Oturum kimliği | Her bağlantıda yeni → artifact erişilemez | `localStorage` / `--session` |
| Workflow state | `InMemorySaver` | `AsyncSqliteSaver` (Postgres'e hazır) |
| Depo sözleşmesi | Yalnızca OBC | OBC **+** OpenShift AI connection |
| Artifact tipleri | Yok | MLMD şema başlıkları + `.metadata` |
| Depo kökü | Sabit kod | `PTC_ARTIFACT_ROOT` |
| **Keşif** | Yumuşak garanti | **İsimler prompt'ta** (ADK deseni) |

**2026-09-06 — büyük sadeleşme:**

| Değişiklik | Öncesi | Sonrası |
|---|---|---|
| **LLM artifact API'si** | 5 fonksiyon | **YOK** — düz Python + `/output` |
| Keşif | `list_artifacts()` | Manifest promptta + `/output` yerleştirilmiş |
| Kapsam | Çalıştırma başına mühürlü | **Tenant** (KFP gibi) |
| Dizin çıktısı | Sessizce kayboluyordu | Tek tar, açılmış hâlde yerleşiyor |
| Soy ağacı | Kaydediliyor, okunmuyordu | Otomatik + panelde mermaid grafiği |
| PDF/PNG | `.bin` olarak duruyordu | Gerçek tip + panelde önizleme |
| **Çalıştırma izolasyonu** | `/output` düz — başkasınınki sızıyordu | **İki kök:** `/output` + `/artifacts/<wf>/` |

**Test sayısı:** 179 · **Canlı doğrulanan:** hepsi

---

## Sayfa 15 — Girdi yerleştirme: icadı bırakmak

**Emsalsiz olan son desenimiz 2026-09-07'de kaldırıldı.**

| | Öncesi (icat) | Sonrası (Argo/KFP) |
|---|---|---|
| `/output` içeriği | manifestten uydurulmuş isimler | **gerçek dosyalar** |
| İndirme anı | okuma çağrısının **ortasında** | pod açılışında, sidecar'da |
| `os.scandir("/output")` | `[]` — yamayı deliyordu | `['x.parquet']` |
| Yama satırı | ~120 | **0** |
| `entrypoint.py` | 651 satır | **357 satır** |
| Okuyan çalıştırma | 4,11 sn | **3,13 sn** |

Yaptığı iş FUSE'un işiydi (Databricks, E2B, Vercel onu kullanıyor); OpenShift
`restricted-v2` `/dev/fuse` vermediği için kullanıcı alanında taklit etmiştik.

**Prefetch'i öldüren kapsamdı, prefetch değil.** Eski prefetch tenant'ın
tamamını indiriyordu. Ölçüm: workflow başına medyan 3 dosya / 13,7 KiB,
azami **35,3 KiB** — 512Mi'nin **on binde yedisi**.

---

## Sayfa 16 — Beyan · Süzgeç · Alias

**Üç açık, üç kanonik cevap. Üçü de kopya.**

| Açık | Kaynak | Bizdeki hâli |
|---|---|---|
| Hangi girdi okundu (soy şişiyordu) | **MLMD** `Event.DECLARED_INPUT` | `run_ptc_code(kod, inputs=[...])` |
| 62 addan 39'u görünüyor, arama yok | **MLMD** `ListOptions(filter_query=…)` | `?name= ?type= ?workflow= ?q=` |
| Aynı ad 17 kez, hep en yeni geliyor | **MLflow** `models:/<ad>@<alias>` | `by-name/rapor.pdf@onaylanmis` |

**Beyanın iki etkisi birden:**

| | Beyansız | `inputs=["a.txt"]` |
|---|---|---|
| `/output`'a yerleşen | çalıştırmanın **hepsi** | yalnızca `a.txt` |
| `turev.txt`'nin ebeveyni | `a.txt, b.txt, c.txt` | **`a.txt`** |

**Ara denemeydi, atıldı:** soyu `atime` ile ölçmeyi denedik — çalışıyordu ama
sahada emsali yok. Piyasa soyu **gözlemlemiyor, beyan ediyor**; olay tipinin
adı zaten `DECLARED_INPUT`.

---

## Sayfa 17 — Baytı HTTP ile göndermek: kimin varsayılanı

**OpenShift'in varsayılanı değil — ama icat da değil.**

| | Baytlar | S3 anahtarı nerede | Kayıt defteri |
|---|---|---|---|
| **KFP / OpenShift AI** | launcher → S3 **doğrudan** | **kullanıcı container'ında** | MLMD (ayrı kanal) |
| **Argo Workflows** | wait sidecar → S3 doğrudan | ayrı container | **yok** |
| **MLflow (proxied)** | client → **HTTP** → server → depo | **server'da** | tracking DB |
| **BİZ** | sidecar → **HTTP** → servis → MinIO | **serviste** | SQLite |

> *"The tracking server works as a **proxy** for accessing remote artifacts.
> The MLflow clients make **HTTP request to the server** for fetching artifacts."*
> — `--serve-artifacts`, MLflow'da **varsayılan açık**

**KFP'yi neden alamadık:** launcher kullanıcı kodunun container'ını *sarmalıyor*
→ S3 anahtarı LLM'in `os.environ`'unda olurdu.
**Argo'yu neden alamadık:** yerleşimi doğru ama **kayıt defteri yok**.

**Aldığımız:** yerleşim Argo'dan (ayrı container), kanal MLflow'dan (HTTP vekili).

---

## Sayfa 18 — OpenShift'e geçilmeyecek

**Soru soruldu, ölçüldü, cevap hayır — gerekçe teknik değil, getiri.**

| | Var (bu laptop) | CRC istiyor |
|---|---|---|
| Fiziksel çekirdek | 10 | 4 ✅ |
| RAM | **15,35 GB** (8 boş) | **10,5 GB boş** ⚠️ |
| OS | **Ubuntu 24.04** | *"Ubuntu and Debian: Not supported"* ⚠️ |

**Kazandıracağı tek şey doğrulama:** `restricted-v2` SCC'yi taklit yerine
gerçekte, `EgressFirewall`'ı canlı. §17 iş yükünün uyumluluğunu SCC taklidiyle
**zaten ölçtü** (rastgele UID 1000670000, `drop ALL`, `RuntimeDefault`).

**Kazandırmayacağı:** Hubble'ın karşılığı yok · OpenShift AI (KFP+MLMD+Model
Registry) o RAM'e sığmaz · kind'da çalıştırma **3,1 sn**, CRC açılışı dakikalar.

**Kalan tek açık:** Cilium → `NetworkPolicy` + `EgressFirewall` çevirisi
yazılmadı; yazılırsa kind'da test edilemez.

---

## Sayfa 19 — Canlı konsol: her şey gerçek

**`/konsol` — dört sekme, sahte veri yok.**

| Sekme | Kaynağı |
|---|---|
| **Sohbet** | mevcut ajan ekranı, `app.js` tek satır değişmeden |
| **Hatlar** | `/api/pipelines` — iki hat, adımları ve kodlarıyla |
| **Çalıştırma** | `/ws/pipeline` — **gerçek Kubernetes Job'ları** |
| **Depo** | `/api/depo` — kayıt defteri + süzgeçler |
| **Soy ağacı** | `/api/depo/<id>/soy` — gerçek `parents` kenarları |

**İki node türü, fark uydurma değil:**
`sandbox` gerçek PTC pod'u açar · `query` kayıt defterine sorgu atar, **pod açmaz**.
Sandbox'ın listeleme yolu hiç yok — keşif host tarafında olur.

**Çapraz workflow canlı doğrulandı:**

```
adım 1  GET /artifacts?name=processed-result.json → üreten wf e841efd4
adım 2  load_artifact("e841efd4-…", "processed-result.json")
        analysis-input.json  parents=['art_5605dd638cab']   ← sınır geçildi
```

Sol altta PTC terminali: pod adı, çalıştırılan kod, süpürülen artifact — canlı.

---

## Sayfa 20 — Açıklar (saklamıyoruz)

| Konu | Durum | Etki |
|---|---|---|
| **İzolasyon** | Düz container, Kata yok | Red Hat'in önerisine uymuyoruz |
| **Metadata DB** | SQLite | Tek replika sınırı |
| **Workflow state** | Postgres yolu **test edilmedi** | Cluster'da Postgres yok |
| **Auth** | Yok | Uuid'yi bilen okur |
| **Büyük dosya** | 100 MiB / 512Mi | 5 GB çalışmaz |
| **İsim çakışması** | "En yeni" kazanır, sessiz | Tenant genelinde daha olası |
| ~~**Şeffaf okuma**~~ | **KAPANDI (2026-09-07)** — yama yok, dosyalar gerçek | Emsalsiz olan son desenimizdi |
| **`user_metadata`** | Süpürme yolunda doldurulamıyor | `put_artifact` kalkınca kapandı |
| **Tip** | Yalnızca dosya uzantısından | Metrik/Dataset ayrımı kayboldu |
| **Soy imzasız** | Kayıt defterine yazabilen değiştirebilir | Tekton Chains bunu çözüyor |
| **Gerçek OBC/ODF** | Test edilmedi | ODF kapsam dışı |

---

## Sayfa 21 — Ekibe dört soru

| # | Soru | Neden önemli |
|---|---|---|
| 1 | **Kurumda S3-uyumlu depo var mı?** | ODF yok → S3'ü biz getireceğiz. Red Hat'in kendi ürünü de zorunlu tutuyor |
| 2 | **Hangi StorageClass'lar var, RWX destekleyen var mı?** | Tekton'un PVC deseni mümkün mü belirler |
| 3 | **OpenShift Sandboxed Containers (Kata) kurulu mu?** | En büyük açığımız; kod değişikliği gerektirmez |
| 4 | **PostgreSQL sağlanabilir mi?** | Hem artifact metadata hem workflow state |

**1. soru en kritik** — ODF kapsam dışına çıkınca "S3 nereden gelecek" cevapsız
kaldı. Varsa endpoint + bucket + anahtar yeter, **kod hazır** (iki sözleşmeyi
de okuyor).
