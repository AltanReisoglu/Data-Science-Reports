# PTC Artifact Persistence — Karşılaştırmalı Sunum

**Her sayfa bir özellik. Her sayfada bir tablo. Sonunda "biz neredeyiz".**

Tarih: 2026-09-04 · Detaylar: [PTC_Piyasa_Mentaliteleri.md](PTC_Piyasa_Mentaliteleri.md)

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
| `/output` süpürme + açık API | **BİZ** | **Hayır** (ikisi de var) |
| Açık RPC zorunlu | Cloudflare Code Mode *(dosya sistemi yok)* | Evet |

**Hiçbir SOTA sistem bu riski açık bir tool'a bağlı bırakmıyor.**

**Bizde iki tetikleyici, tek kapı:**

| | Açık çağrı | Süpürme |
|---|---|---|
| Nasıl | `put_artifact(df, name=...)` | `df.to_csv("/output/x.csv")` |
| Ne zaman | İstediği an | Çalışma sonunda, **hata olsa bile** |
| Tip korunumu | Evet (Parquet) | Uzantıdan tahmin |

---

## Sayfa 9 — KEŞİF: ajan çekeceğini nasıl anlıyor

**Bu, oturum boyunca en çok kafa karıştıran soruydu.**

| # | Desen | Nasıl | Kim |
|---|---|---|---|
| 1 | Sadece tool tarifi | Model çağırmayı *seçmek* zorunda | *(eskiden biz)* |
| 2 | Referans otomatik context'te | `file_id` tool sonucunda döner | Anthropic |
| 3 | Dosya sistemi + `ls` | Sandbox yaşıyorsa model bakar | Anthropic, Google, OpenHands |
| 4 | **İsimler prompt'a enjekte** | İsimler talimatlarda, içerik talep üzerine | **Google ADK**, **BİZ** |
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
| Örnek | `adim5(girdi=adim1.ciktilar["features"])` | `list_artifacts()` |
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
| **Kapsam granülaritesi** | **Kimse — bizimki daha dar** |
| **İzolasyon** | **Kimse — bizimki daha zayıf** |
| **Şeffaf okuma** | **Kimse — emsalsiz** |

**Üç cümle:**
1. Omurga tartışmasız SOTA — tezi Cloudflare'den, veri modelini Anthropic'ten,
   platform desenini Red Hat'ten aldık.
2. İzolasyonda herkesin gerisindeyiz (düz container). Warm pool yok ama
   kazancını ölçtük: ≤1.6 sn.
3. Kapsam granülaritesinde herkesin ilerisindeyiz (çalıştırma başına imzalı
   jeton; KFP namespace düzeyinde).

---

## Sayfa 14 — Bugün ne değişti (2026-09-04)

| Değişiklik | Öncesi | Sonrası |
|---|---|---|
| Artifact servisi ayrıldı | Tool Gateway'de, base64+MCP | Kendi pod'unda, akışlı HTTP |
| Prefetch kaldırıldı | Her artifact iniyordu, O(hepsi) | Manifest + tembel doldurma, O(kullanılan) |
| TTL reaper | Şema vardı, çalıştıran yoktu | Saat başı CronJob |
| Oturum kimliği | Her bağlantıda yeni → artifact erişilemez | `localStorage` / `--session` |
| Workflow state | `InMemorySaver` | `AsyncSqliteSaver` (Postgres'e hazır) |
| Depo sözleşmesi | Yalnızca OBC | OBC **+** OpenShift AI connection |
| Artifact tipleri | Yok | MLMD şema başlıkları + `.metadata` |
| Depo kökü | Sabit kod | `PTC_ARTIFACT_ROOT` |
| **Keşif** | Yumuşak garanti | **İsimler prompt'ta** (ADK deseni) |

**Test sayısı:** 108 · **Canlı doğrulanan:** hepsi

---

## Sayfa 15 — Açıklar (saklamıyoruz)

| Konu | Durum | Etki |
|---|---|---|
| **İzolasyon** | Düz container, Kata yok | Red Hat'in önerisine uymuyoruz |
| **Metadata DB** | SQLite | Tek replika sınırı |
| **Workflow state** | Postgres yolu **test edilmedi** | Cluster'da Postgres yok |
| **Auth** | Yok | Uuid'yi bilen okur |
| **Büyük dosya** | 100 MiB / 512Mi | 5 GB çalışmaz |
| **Soy ağacı** | Kaydediliyor, keşifte kullanılmıyor | Eksik yetenek |
| **İsim çakışması** | "En yeni" kazanır, sessiz | Veri kaybı riski |
| **Şeffaf okuma** | 5 pandas okuyucusu + `open` | `pyarrow`, `csv`, `PIL` yakalanmıyor |
| **Gerçek OBC/ODF** | Test edilmedi | ODF kapsam dışı |

---

## Sayfa 16 — Ekibe dört soru

| # | Soru | Neden önemli |
|---|---|---|
| 1 | **Kurumda S3-uyumlu depo var mı?** | ODF yok → S3'ü biz getireceğiz. Red Hat'in kendi ürünü de zorunlu tutuyor |
| 2 | **Hangi StorageClass'lar var, RWX destekleyen var mı?** | Tekton'un PVC deseni mümkün mü belirler |
| 3 | **OpenShift Sandboxed Containers (Kata) kurulu mu?** | En büyük açığımız; kod değişikliği gerektirmez |
| 4 | **PostgreSQL sağlanabilir mi?** | Hem artifact metadata hem workflow state |

**1. soru en kritik** — ODF kapsam dışına çıkınca "S3 nereden gelecek" cevapsız
kaldı. Varsa endpoint + bucket + anahtar yeter, **kod hazır** (iki sözleşmeyi
de okuyor).
