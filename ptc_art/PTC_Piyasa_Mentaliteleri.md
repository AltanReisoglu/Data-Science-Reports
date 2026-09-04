# Sandbox ve Artifact Mentaliteleri — Kim Nasıl Düşünüyor

**Tarih:** 2026-09-04

Bu doküman iki işe yarasın diye yazıldı:
1. **Anlatabilmek** — bir ekip arkadaşı sorduğunda adım adım açıklayabilmek.
2. **Karşılaştırabilmek** — "biz neredeyiz" sorusunu boyut boyut cevaplayabilmek.

Bu yüzden her şirket için önce **mekanizma adım adım** anlatılıyor, sonra
karşılaştırma tabloları geliyor, en sonda da **ekipten gelebilecek sorular ve
cevapları** var.

Terimleri bilmiyorsanız §0'daki sözlükten başlayın; doküman o sözlüğe göre
yazıldı.

## Yol haritası

| Bölüm | Ne var |
|---|---|
| **§0** | Sözlük — FUSE, isolate, Kata, warm pool… önce bunlar |
| **§1** | Herkesin cevaplamak zorunda olduğu **5 soru** — karşılaştırmanın iskeleti |
| **§2–§9** | Şirket şirket, adım adım mekanizma + o beş sorunun cevabı |
| **§10** | Karşılaştırma tabloları (izolasyon, ömür, erişim, kayıt defteri, ağ) |
| **§11** | **Bizim mimarimiz, en baştan en sona** |
| **§12** | Biz neredeyiz — boyut boyut kiminle örtüştüğümüz |
| **§13** | Ekipten gelecek 16 soru ve hazır cevapları |
| **§14** | Doğrulanamayanlar |

Her şirket bölümü aynı beş soruyla bitiyor, **§11 dahil**. İki mimariyi
karşılaştırmak, iki tabloyu yan yana koymak demek.

---

# §0 — Sözlük (önce bunlar)

Bu terimler her yerde geçiyor. Kısa ve somut tanımlar:

**Sandbox.** Güvenilmeyen kodun çalıştırıldığı izole ortam. "Güvenilmeyen" =
ne yapacağını önceden bilmediğin kod. Bizim durumumuzda kodu LLM yazıyor.

**Container (runc).** Normal Docker container'ı. Host'un çekirdeğini (kernel)
paylaşır, sadece namespace'lerle ayrılır. En hızlı, en zayıf izolasyon.
*Analoji:* aynı binada ayrı daireler — duvarlar var ama temel ortak.

**gVisor.** Container ile gerçek çekirdek arasına, kullanıcı alanında çalışan
ikinci bir "sahte çekirdek" koyar. Container sistem çağrısı yaptığında gerçek
çekirdeğe değil buna gider. *Analoji:* daire ile bina arasına bir güvenlik
katı eklemek. Orta izolasyon, orta hız.

**Kata Containers / microVM.** Her pod'a **kendi çekirdeği** olan hafif bir
sanal makine verir. *Analoji:* aynı binada değil, ayrı müstakil evler.
OpenShift'teki adı "OpenShift Sandboxed Containers".

**Hyper-V izolasyonu.** Microsoft'un aynı fikri — donanım sanallaştırmasıyla
her session ayrı VM'de.

**Isolate (V8 isolate).** Container bile değil — tek bir JavaScript motorunun
içinde ayrılmış bellek bölgesi. Milisaniyede açılır, birkaç megabayt yer kaplar.
Cloudflare'in yaklaşımı. *Sınırı:* sadece JS/WASM çalıştırır, tam Linux değil.

**FUSE (Filesystem in Userspace).** Uzaktaki bir depoyu (S3 gibi) yerel klasör
gibi göstermeye yarayan sürücü. `s3fs`, `mount-s3`, `gcsfuse`, `rclone` bunun
uygulamaları. *Sonuç:* `open("/mnt/bucket/x.csv")` yazınca arka planda S3
çağrısı olur.

**Mount etmek.** Bir depoyu dosya sistemine bağlamak. Mount edilmişse yol
**doğrudan** bucket'tır; yazdığın an gider.

**Presigned URL (imzalı URL).** Tek bir dosyaya, tek bir işleme (indir ya da
yükle), kısa süreliğine izin veren, içinde imza taşıyan URL. Elinde anahtar
olmadan o URL ile o dosyaya erişebilirsin.

**Warm pool (ön ısıtılmış havuz).** Önceden açılmış, boşta bekleyen sandbox'lar.
İstek gelince sıfırdan yaratmak yerine havuzdan biri verilir.

**Checkpoint / snapshot.** Çalışan bir ortamın o anki hâlinin (bellek + disk)
dondurulup saklanması. Sonra "geri yükle" denince aynı yerden devam eder.

**Artifact.** Üretilmesi pahalı, sonra tekrar kullanılacak çıktı: tablo, model,
rapor. Geçici dosyadan farkı — kaybolması işi baştan yaptırır.

**Artifact registry (kayıt defteri).** "Hangi artifact var, nerede duruyor,
kim üretti, ne zaman silinecek" bilgisini tutan veritabanı. Baytların
kendisinden **ayrı**.

**MLMD (ML Metadata).** Kubeflow'un kayıt defteri. Artifact'leri tipleriyle,
üretildikleri adımla ve soy ağacıyla saklar.

**OBC (ObjectBucketClaim).** OpenShift'te "bana bir bucket ver" demenin yolu.
ODF kuruluysa çalışır, bir ConfigMap + Secret üretir.

**ODF (OpenShift Data Foundation).** Red Hat'in depolama ürünü. **Bizde yok.**

**CSI (Container Storage Interface).** Kubernetes'in depolama sürücüsü standardı.

**PVC / RWO / RWX.** PersistentVolumeClaim = kalıcı disk talebi.
ReadWriteOnce = tek node yazabilir. ReadWriteMany = birden çok node yazabilir
(paylaşımlı disk için şart, ve her sürücüde yok).

---

# §1 — Herkesin cevaplamak zorunda olduğu 5 soru

Bütün bu şirketler aynı beş soruyla boğuşuyor. Karşılaştırmayı bu beş soru
üzerinden yapmak en temizi:

| # | Soru | Uçlar |
|---|---|---|
| **S1** | Kod **nerede** çalışıyor? | Isolate ↔ container ↔ gVisor ↔ microVM |
| **S2** | Sandbox **ne kadar yaşıyor**? | Her çağrıda yeni ↔ günlerce süren oturum |
| **S3** | Veri **nasıl girip çıkıyor**? | Mount ↔ SDK/API ↔ hiç (kapalı) |
| **S4** | **Anahtar** kimde? | Sandbox'ın içinde ↔ dışarıda bir aracıda ↔ hiç yok |
| **S5** | **Kayıt defteri** var mı? | Var (tip/soy/TTL) ↔ yok (sadece bucket) |

Bir mimariyi anlamak = bu beş sorunun cevabını bilmek. Aşağıda her şirket için
bu beşi ayrı ayrı veriyorum.

---

# §2 — Anthropic

## Tez tek cümlede
> Sandbox interneti **hiç görmesin**; veri kapıdan girsin, kapıdan çıksın.

## Adım adım ne oluyor

1. Kullanıcı bir CSV yükler → dosya **Files API**'ye gider, bir `file_id` alır.
2. Modele istek atılırken o `file_id` referans verilir.
3. Anthropic bir **container** açar. Bu container'ın interneti **yoktur**.
4. Model kod yazar, çalıştırır. Kütüphane indiremez — sadece önceden kurulu
   olanlar vardır.
5. Model bir çıktı üretirse onu `$OUTPUT_DIR` adlı dizine yazar.
6. Komut bitince **o dizindeki her dosya yakalanır**, her birine bir `file_id`
   verilir ve sonuçta döner.
7. Kullanıcı o `file_id` ile dosyayı Files API'den indirir.
8. Yanıtta ayrıca bir `container` nesnesi döner; içinde `id` ve `expires_at`.

## Süreklilik nasıl çalışıyor (en ilginç kısmı)

- **Varsayılan:** her istek **yeni** container'da çalışır.
- Önceki yanıttan gelen `container` id'sini geri gönderirsen **aynı container**
  kullanılır — dosyalar yerinde kalır.
- ~5 dakika hareketsizlikten sonra container **checkpoint**'lenir (dondurulur).
- 30 gün içinde o id ile istek gelirse **geri yüklenir**.
- Yeni tool sürümü + PTC ile **Python yorumlayıcı durumu da** korunur — yani
  `x = 5` dediysen, sonraki istekte `x` hâlâ 5'tir.

Yani "efemer mi kalıcı mı" sorusuna ikisi birden diyorlar: **varsayılan efemer,
talep edilirse kalıcı.**

## Beş soru

| | Cevap |
|---|---|
| **S1 Nerede** | Yönetilen container |
| **S2 Ömür** | Varsayılan tek istek; id ile 30 güne kadar (5 dk'da checkpoint) |
| **S3 Veri** | Files API (giriş) + `$OUTPUT_DIR` yakalama (çıkış). Mount **yok** |
| **S4 Anahtar** | Sandbox'ta depo anahtarı diye bir şey yok |
| **S5 Kayıt defteri** | **Var** — Files API (`id`, `filename`, `mime_type`, `size_bytes`, `created_at`, `expires_at`) |

## Sayılar
5 GiB çalışma alanı · container 30 gün · ~5 dk sonra checkpoint ·
internet tamamen kapalı.

## Alıntılar
> *"The container has no internet access, so Claude can't download packages at
> runtime: only the pre-installed libraries are available"*

> *"Internet access: Completely disabled for security"*

> *"Each request runs in a new container unless you pass an earlier response's
> container ID back"*

## Ekstra
Ürettiği medya dosyaları **C2PA Content Credentials** taşıyor — kriptografik
imzalı, "bu dosyayı Claude üretti" diyen bir manifest.

---

# §3 — OpenAI

## Tez tek cümlede
> "Çalışma alanı" ile "kalıcı durum" **ayrı kavramlardır**; sandbox'ı
> dondurup çözebilirsin.

## Adım adım ne oluyor

1. Dosyalar container'a yüklenir; sandbox içinde `/mnt/data` altında görünür.
2. Model kod yazar, `/mnt/data`'ya okur/yazar.
3. Ürettiği dosyalar container files uç noktalarından indirilir.
4. Sandbox Agents yaklaşımında sandbox'ın **lifecycle**'ı, **snapshot/resume**'u
   ve **uzak dosya sistemi** bağlantısı üç ayrı kavram olarak ele alınır.

`/mnt/data` konvansiyonu Anthropic'in `$OUTPUT_DIR`'ıyla aynı fikir: **model
sıradan kod yazar, platform bir dizini toplar.**

## Beş soru

| | Cevap |
|---|---|
| **S1 Nerede** | Yönetilen container |
| **S2 Ömür** | Kalıcı workspace + snapshot/resume |
| **S3 Veri** | `/mnt/data` + container files uçları. Mount **yok** |
| **S4 Anahtar** | Sandbox'ta depo anahtarı yok |
| **S5 Kayıt defteri** | Dosya kimlikleri var |

> **Dürüstlük notu:** Bu bölüm bu turda birincil kaynaktan yeniden
> doğrulanmadı; önceki araştırma taramamıza dayanıyor. Ekibe sunmadan önce
> OpenAI'ın güncel dokümanından teyit edilmeli. Diğer bölümler bu turda
> birincil kaynaktan çekildi.

---

# §4 — Cloudflare

Cloudflare'in **iki ayrı** tezi var ve ikisi de bizim için önemli.

## Tez 1 — "LLM'e kod yazdır, tool çağırtma"

Bu, bizim PTC'mizin varlık sebebinin aynısı.

**Argüman şu:** Klasik tool calling'de her tool çağrısının çıktısı modelin
sinir ağından geçip bir sonraki çağrının girdisine **kopyalanır**. Beş adımlık
bir iş, beş kez model turu demektir. Halbuki model bir kez kod yazsa, o kod
beş çağrıyı kendi içinde yapar ve sadece sonucu döndürür.

> *"With the traditional approach, the output of each tool call must feed into
> the LLM's neural network, just to be copied over to the inputs of the next
> call, wasting time, energy, and tokens."*

**İkinci argüman eğitim verisiyle ilgili** — ve daha ilginç olanı bu:

> *"LLMs are better at writing code to call MCP, than at calling MCP directly."*

> *"LLMs have seen a lot of code. They have not seen a lot of 'tool calls'."*

Yani modeller milyarlarca satır gerçek kod görmüş ama "tool call" denen şey
görece yeni ve sentetik. Doğal olarak kod yazmakta daha iyiler.

## Tez 2 — "Isolate, container'dan hafiftir"

> *"Isolates are far more lightweight than containers. An isolate can start in a
> handful of milliseconds using only a few megabytes of memory."*

Sonuç: her kod parçası için **tek kullanımlık** izolat açabiliyorlar. Havuza,
ön ısıtmaya gerek yok — çünkü açılış zaten milisaniye.

*Sınırı:* isolate tam Linux değil, JS/WASM çalıştırır. Tam Linux gerektiğinde
Cloudflare Sandbox (container tabanlı) devreye giriyor.

## Depolama — mount ediyorlar

Cloudflare Sandbox'ta R2 (kendi nesne depoları) **yerel dosya sistemi olarak
mount ediliyor**:

> *"standard file operations with data that persists across sandbox lifecycles"*

Kimlik bilgisi konusunda **üç mod** var:
1. **R2 binding** — sandbox'ta anahtar yok, platform bağlıyor
2. **`credentialProxy: true`** — sandbox anahtarsız mount eder, istekleri
   dışarıdaki bir proxy imzalar
3. **Düz `credentials`** — anahtar sandbox'a verilir

İkinci mod önemli: **mount'un kolaylığı + anahtarın dışarıda kalması.**

## Beş soru

| | Cevap |
|---|---|
| **S1 Nerede** | Code Mode'da V8 isolate; Sandbox ürününde container |
| **S2 Ömür** | Isolate tek kullanımlık; Sandbox'ta Durable Objects "stateful coordination layer" |
| **S3 Veri** | R2 **mount** — dosya sistemi gibi |
| **S4 Anahtar** | Üç mod: binding (yok) / proxy (yok) / düz (var) |
| **S5 Kayıt defteri** | Yok — mount edilen bucket "sadece bucket" |

> **Doğrulanamayan:** Sandbox'ın kendisinin mi kalıcı olduğu, yoksa kalıcılığın
> tamamen Durable Objects'ten mi geldiği dokümandan net çıkmadı.

---

# §5 — Google (iki ayrı cevap veriyor)

Google aynı problemi iki üründe **farklı** çözüyor. Karıştırmamak önemli.

## §5.1 — Vertex AI Agent Engine (yönetilen)

### Tez
> Sandbox **adlandırılmış ve uzun ömürlü bir kaynaktır**; süreklilik oradan
> gelir.

### Adım adım
1. Bir sandbox'a **isim** verirsin (`sandbox_name`).
2. Kod çalıştırırsın — sandbox durumu **otomatik korur**.
3. Sonraki çağrıda **aynı ismi** verirsen aynı ortama düşersin: değişkenler,
   import'lar, dosyalar yerinde.
4. Sandbox'lar durumu **14 güne kadar** tutar; bu TTL yapılandırılabilir.
5. ADK entegrasyonunda `AgentEngineSandboxCodeExecutor` bir ajan görevi boyunca
   **tek sandbox** tutar.

### Beş soru
| | Cevap |
|---|---|
| **S1 Nerede** | Yönetilen sandbox |
| **S2 Ömür** | **14 güne kadar** (yapılandırılabilir TTL) |
| **S3 Veri** | 100 MB'a kadar dosya yükleme; veri tekrar yüklenmeden çok analiz |
| **S4 Anahtar** | Platform yönetiyor |
| **S5 Kayıt defteri** | Ayrı bir artifact registry değil, sandbox'ın kendi durumu |

**Sayı:** kod çalıştırma **300 saniyede** zaman aşımı.

## §5.2 — GKE Agent Sandbox (Kubernetes-yerel)

Bu, OpenShift'e en yakın karşılaştırma — ikisi de Kubernetes.

### Tez
> "İzole, **stateful**, tek-replikalı iş yüklerini" Kubernetes nesnesi olarak
> yönet.

### Adım adım
1. Yönetici bir **`SandboxTemplate`** tanımlar (imaj, kaynak, politika).
2. Uygulama bir **`SandboxClaim`** yaratır — "bana bu şablondan bir sandbox ver".
3. Controller claim'i gerçek bir sandbox'a eşler.
4. Sandbox **gVisor** ile çekirdek düzeyinde izole; istenirse **Kata** eklenir.
5. Ağ **varsayılan reddet** — sandbox'taki kod iç ağa ve kontrol düzlemine
   erişemez.
6. Hız için **warm pool** + **pod snapshot** birlikte kullanılır.

### Beş soru
| | Cevap |
|---|---|
| **S1 Nerede** | **gVisor** birincil, Kata opsiyonel |
| **S2 Ömür** | Stateful; snapshot ile geri yükleme |
| **S3 Veri** | Claim modeli depolama karmaşıklığını soyutluyor |
| **S4 Anahtar** | Platform yönetiyor |
| **S5 Kayıt defteri** | Belirtilmemiş |

**Sayılar:** "sub-second provisioning" · warm pool'lar 1 saniyenin altında ·
warm pool + snapshot = **"instant-on"**.

**Not:** `SandboxClaim`/`SandboxTemplate` deseni, Kubernetes'in `PVC`/
`StorageClass` desenini sandbox'a uyarlamış hâli. Aynı fikir: kullanıcı "ne
istediğini" söyler, yönetici "nasıl sağlanacağını" tanımlar.

## §5.3 — Google ADK: keşif problemini çözen tek birinci-sınıf mekanizma

Agent Engine "sandbox'ı yaşat" diyerek keşfi atlıyordu. ADK (Agent Development
Kit) ise **artifact** kavramını ayrıca tanımlıyor ve "ajan neyin var olduğunu
nasıl öğrenir" sorusuna doğrudan cevap veriyor.

### Artifact servisi

`BaseArtifactService` şu metodları sunuyor:
`save_artifact()` · `load_artifact()` · `list_artifact_keys()` ·
`list_artifact_versions()` · `get_artifact_version()` · `delete_artifact()`

- **`save_artifact()`** yeni bir **sürüm** kaydeder, sürüm numarası döner (0'dan
  başlar) — yani ADK'da da artifact'ler değişmez, yeni yazım yeni sürümdür.
- **`load_artifact()`** sürüm verilmezse **en yeniyi** döndürür.
- **`list_artifact_keys()`** o oturumdaki tüm artifact adlarını döndürür.

Artifact içeriği `google.genai.types.Part` olarak taşınıyor; `inline_data`
içinde ham `data` (bayt) ve `mime_type` var.

### Kritik ayrım: bu metodlar LLM'e TOOL olarak sunulmuyor

> *"The primary way you interact with artifacts within your agent's logic
> (specifically within callbacks or tools) is through methods provided by the
> `CallbackContext` and `ToolContext` objects."*

Yani `list_artifact_keys()` bir **geliştirici API'si**. Model bunu çağıramaz.
Modelin haberdar olması için geliştiricinin iki seçeneği var:

**Seçenek A — elle enjeksiyon.** Bir callback içinde listeyi çek, formatla,
kullanıcı isteğinin başına ekle. Dokümanın örneği bunu *"Prepend this
information to the user's request for the model"* diye tarif ediyor.

**Seçenek B — `LoadArtifactsTool`.** Yerleşik mekanizma. İki iş birden yapıyor:

> *"LoadArtifactsTool **lists available artifacts in the model instructions**.
> When the model calls the load_artifacts tool, ADK **temporarily appends** the
> selected artifact contents to that request."*

### Tasarımın özü: isim ucuz, içerik pahalı

`LoadArtifactsTool`'un çözdüğü asıl problem context şişmesi. Üç kuralla
çözüyor:

1. **İsimler her zaman context'te** — model talimatlarında listeleniyor, ucuz,
   model unutamaz
2. **İçerik talep üzerine** — model `load_artifacts` çağırınca geliyor
3. **İçerik geçmişe KALICI yazılmıyor** — *"the model should call the tool again
   when it needs the same artifact in a later turn"*

Üçüncüsü ince ve önemli: bir kez yüklenen 50 MB'lık tablo, sonraki her turda
context'te taşınmıyor.

### Beş soru

| | Cevap |
|---|---|
| **S1 Nerede** | ADK bir framework; sandbox ayrı (Agent Engine / GKE) |
| **S2 Ömür** | Artifact'ler oturum kapsamında, sürümlü |
| **S3 Veri** | `save_artifact`/`load_artifact`, `Part`/`inline_data` |
| **S4 Anahtar** | Artifact service arkasında; ajan görmez |
| **S5 Kayıt defteri** | **Var** — isim + sürüm listesi |

---

# §6 — AWS (Bedrock AgentCore Code Interpreter)

## Tez tek cümlede
> Ajanın keyfi kod çalıştırması bir **veri sızıntısı riskidir**; bunu yönetilen
> bir sandbox'a devret ve her şeyi logla.

Gerekçeyi kendileri açıkça yazıyor:
> *"This is critical in Agentic AI applications where the agents may execute
> arbitrary code that can lead to data compromise or security risks."*

## Adım adım — ve burada **boyut** belirleyici

AWS veriyi iki ayrı yoldan taşıyor ve aradaki fark 50 kat:

**Küçük veri (≤100 MB):** doğrudan istekte "inline" yüklenir.

**Büyük veri (≤5 GB):** sandbox'a bir **execution role** (IAM rolü) verilir,
sandbox **terminal komutlarıyla** (AWS CLI gibi) S3'e kendisi konuşur.

> *"For inline upload, the file size can be up to 100 MB. And for uploading to
> Amazon S3 through terminal commands, the file size can be as large as 5 GB."*

Bu ikinci yol önemli: **anahtar sandbox'ın içinde** (IAM rolü olarak), ama o
rol IAM politikasıyla daraltılmış. Yani "anahtar var ama dar".

## Ağ — burada herkesten ayrılıyorlar

Anthropic interneti tamamen kapatırken, AWS **açık** bırakıyor ve
yapılandırılabilir hâle getiriyor:

> *"advanced features, including large file support and internet access, and
> CloudTrail logging capabilities"*

"Network modes" ile kurumsal gereksinime göre ayarlanıyor.

## Beş soru

| | Cevap |
|---|---|
| **S1 Nerede** | Container'laştırılmış ortam |
| **S2 Ömür** | Varsayılan **15 dakika**, **8 saate** kadar uzatılabilir |
| **S3 Veri** | İkiye ayrık: inline 100 MB / S3 üzerinden 5 GB |
| **S4 Anahtar** | **Sandbox'ta** — ama IAM execution role olarak daraltılmış |
| **S5 Kayıt defteri** | Ayrı registry değil; denetim **CloudTrail**'de |

## Ayırt edici özelliği
**Denetlenebilirlik.** CloudTrail loglaması ürünün ilan edilmiş özelliği —
"kim ne çalıştırdı" sorusunun kurumsal cevabı.

---

# §7 — Microsoft (Azure Container Apps dynamic sessions)

## Tez tek cümlede
> Güçlü izolasyon ile hız **aynı anda** elde edilebilir — cevap **ön ısıtılmış
> havuz**.

Bu, diğerlerinin çoğunun kaçındığı takası doğrudan hedef alıyor: genelde
"güçlü izolasyon = yavaş açılış" kabul edilir. Microsoft "havuzla çözülür"
diyor.

## Adım adım

1. Yönetici bir **session pool** (havuz) kurar.
2. Havuz, **önceden açılmış ama tahsis edilmemiş** session'lar tutar.
3. Uygulama bir istek atar; istekte bir **`identifier`** sorgu parametresi olur.
4. Havuz, o identifier'a ait **var olan** bir session'a yönlendirir; yoksa
   havuzdan yenisini tahsis eder — bu **milisaniyeler** sürer.
5. İstek yolunun geri kalanı session container'ına iletilir.
6. İstekler geldiği sürece session ayakta kalır.
7. **Cooldown** süresi boyunca istek gelmezse session **yok edilir**, kaynaklar
   temizlenir.

## İki havuz tipi

| Tip | Ne zaman |
|---|---|
| **Code interpreter pool** | Hazır çalışma ortamı; LLM üretimi kod için, kurulum yok |
| **Custom container pool** | Kendi imajın; özel bağımlılık gerekiyorsa |

## Beş soru

| | Cevap |
|---|---|
| **S1 Nerede** | **Hyper-V izolasyonu** — her session ayrı VM |
| **S2 Ömür** | Efemer; cooldown sonrası yok ediliyor. Süreklilik `identifier` ile |
| **S3 Veri** | Session container'ı üzerinden |
| **S4 Anahtar** | Platform yönetiyor |
| **S5 Kayıt defteri** | Yok |

## Alıntılar
> *"Prewarmed pools enable subsecond launch times… New sessions are allocated
> in **milliseconds** thanks to pools of ready but unallocated sessions."*

> *"Sessions are ephemeral and isolated, designed for short-lived tasks…
> After the task completes or the cooldown period expires, the session is
> destroyed and resources are cleaned up."*

---

# §8 — Red Hat / OpenShift

Burada dikkat: OpenShift **üç ayrı** cevap veriyor, çünkü üç ayrı ürün var.
Karıştırmak en sık yapılan hata.

## §8.1 — Temel platform (OpenShift Container Platform)

**En önemli gerçek:** OCP'nin depolama dokümanı baştan sona **PV / PVC / CSI**.
Ephemeral storage, persistent storage, CSI, dinamik provisioning, volume
genişletme. **S3 diye bir bölüm yok.**

Desteklenen CSI sürücüleri tamamen blok/dosya: AWS EBS, AWS EFS, Azure Disk,
Azure File, GCP PD, Google Filestore, IBM VPC Block, OpenStack Cinder,
OpenStack Manila, CIFS/SMB, vSphere.

> **Sonuç:** OpenShift'te nesne deposu ya **ODF** ile gelir, ya **harici bir
> üründen**. Platformun kendisi vermez.

## §8.2 — OpenShift AI (Data Science Pipelines)

Bu, KFP v2'nin Red Hat paketlenmiş hâli.

**S3 zorunlu.** Pipeline server kurmanın ön koşulu:
> *"You have an existing S3-compatible object storage bucket and you have
> configured write access to your S3 bucket on your storage account."*

Dikkat çekici ayrıntı: **veritabanı** için "cluster'daki varsayılan" diye bir
geliştirme seçeneği sunuluyor, ama **depolama için sunulmuyor**. Yani Red Hat'in
kendi ürünü bile "S3 bul, getir" diyor.

### Kullanıcı kodu depoya nasıl erişiyor — **mount değil, SDK**

> *"To interact with data stored in an S3-compatible object store from a
> workbench, you must create a local client to handle requests to the AWS S3
> service by using an AWS SDK such as Boto3."*

Kimlik bilgisi ortam değişkeniyle geliyor:
```python
key_id   = os.environ.get('AWS_ACCESS_KEY_ID')
secret   = os.environ.get('AWS_SECRET_ACCESS_KEY')
endpoint = os.environ.get('AWS_S3_ENDPOINT')
```

**FUSE mount ve CSI sürücüsü hiç geçmiyor.**

### Pipeline artifact'i nasıl taşınıyor — launcher deseni

1. Nesne deposu ayarları **namespace başına** bir ConfigMap'te:
   > *"To configure the object store utilized by the KFP Launcher, you will
   > need to edit the `kfp-launcher` Kubernetes ConfigMap."*
   > *"this configmap needs to be deployed in the same namespace where the
   > Pipelines will be created."*
2. Kimlik bilgileri Secret'lardan gelir; AWS'de IRSA ile ServiceAccount'a da
   bağlanabilir.
3. **Launcher** süreci adımın çıktısını depoya yükler, girdisini indirir.
4. Kullanıcının component container'ı S3 ile **doğrudan konuşmaz**.

### Artifact modeli — `.uri` / `.path` ayrımı

KFP'de her artifact iki adres taşır:
- **`.uri`** — nesne deposundaki gerçek yer
- **`.path`** — container içindeki yerel yol

Ve sistem ikisi arasındaki kopyalamayı **kendisi** yapar: yerel dosya sistemine
yazarsın, launcher `.path`'teki dosyayı `.uri`'ye taşır.

### Artifact tipleri (MLMD şema başlıkları)

| DSL sınıfı | Şema başlığı |
|---|---|
| `Artifact` | `system.Artifact` (taban tip) |
| `Dataset` | `system.Dataset` |
| `Model` | `system.Model` |
| `Metrics` | `system.Metrics` |
| `ClassificationMetrics` | `system.ClassificationMetrics` |
| `SlicedClassificationMetrics` | `system.SlicedClassificationMetrics` |
| `HTML` | `system.HTML` |
| `Markdown` | `system.Markdown` |

Her artifact dört alan taşır: **`name`** (kimlik), **`.uri`** (uzak yer),
**`.path`** (yerel yol), **`.metadata`** (serbest anahtar-değer).

### `pipeline_root` — "bucket'ın neresine" yapılandırmadır

> *"Pipeline root represents the path within an object store bucket where
> Kubeflow Pipelines stores a pipeline's artifacts."*

Üç düzeyde ayarlanır:
1. **Dağıtım geneli** — KFP Launcher ConfigMap'i
2. **Pipeline başına** — `kfp.dsl.pipeline` anotasyonu
3. **Çalıştırma başına** — SDK/UI'dan `pipeline_root` argümanı

Bu bir mentalite: **yol kodda sabit değil, yapılandırmada.**

## §8.3 — OpenShift Pipelines (Tekton) — bambaşka bir yol

Tekton'da adımlar arası paylaşım **workspace** ile yapılır, ve workspace bir
**PVC**'ye bağlanır (`volumeClaimTemplate` ile otomatik provisioning). Nesne
deposu değil, **paylaşımlı disk**.

*Bunun kısıtı:* birden çok pod'un aynı anda yazması **RWX** gerektirir; RWX ise
yalnızca dosya tabanlı StorageClass'larda var (NFS, Azure File, EFS, Filestore,
Manila, CIFS), blok tabanlılarda yok.

## §8.4 — Güvenilmeyen kod için Red Hat'in rehberi

**İzolasyon — katmanlı:**
> *"Running OpenShell inside an OpenShift sandboxed containers VM gives you
> both layers simultaneously."*

Yani AI-üretimi kod için **Kata**.

**Ağ — varsayılan reddet:**
> *"The default posture is deny-all. In practice, you write a policy that
> allowlists exactly the endpoints your agent needs."*

**Olgunluk uyarısı:** Red Hat bu çalışmalar için kendisi *"early validations,
not shipping product features yet"* diyor.

## §8.5 — Ve OpenShift'te KEŞİF: yerleşik cevabı yok

"Ajan, önceki adımın ürettiğini nasıl bulur?" sorusunun OpenShift'te **yerleşik
bir cevabı yok** — ve bunun sebebi yapısal.

**Pipeline dünyasında (KFP/DSP) soru hiç sorulmuyor.** Karar veren bir ajan yok;
DAG'ı insan önceden yazıyor. 5. adımın girdisi **bağlanmış** oluyor: driver
adım başlamadan `.uri`'yi çözüyor, launcher dosyayı `.path`'e indiriyor,
container doğduğunda dosya **zaten orada**. Keşif yerine **statik bağlama** var.

**Agentic dünyada (Llama Stack) artifact kavramı yok.** Red Hat'in OpenShift
AI'daki ajan bahsi Llama Stack; sekiz OpenAI-uyumlu API sunuyor ve içlerinde
Vector Stores'tan ayrı bir **Files API** (`/v1/openai/v1/files`) var. Ajan
dosyaları **`file_search`** tool'uyla buluyor:

> *"…particularly useful for retrieval-augmented generation (RAG) workflows that
> rely on the `file_search` tool to retrieve context from vector stores."*

Ama bu **RAG** — ingest edilmiş belgeler üzerinde semantik arama. Files API'nin
tarifi de bunu söylüyor: *"manages file uploads for use in embedding and
retrieval workflows."* Dokümanda "ajanın ürettiği ve sonra geri aldığı artifact"
diye bir kavram geçmiyor.

| | Artifact kavramı | Ajan kavramı |
|---|---|---|
| **KFP / DSP** | ✓ (tipli, soylu, MLMD'de) | ✗ |
| **Llama Stack** | ✗ (RAG belgeleri var, workflow artifact'i yok) | ✓ |

**Sonuç:** bizim durumumuz tam ikisinin arasındaki boşlukta. Artifact
**depolaması** için kopyalanacak OpenShift deseni var (ve kopyaladık);
artifact **keşfi** için yok. Orada kopyalanacak yer Google ADK'nın
`LoadArtifactsTool`'u.

*Not: Llama Stack, Red Hat'in kendi ifadesiyle **Technology Preview**.*

## Beş soru (OpenShift AI ekseninde)

| | Cevap |
|---|---|
| **S1 Nerede** | Normal pod; AI-üretimi kod için **Kata** öneriliyor |
| **S2 Ömür** | Pipeline adımı = pod; efemer |
| **S3 Veri** | S3, **SDK ile**. Mount önerilmiyor |
| **S4 Anahtar** | Secret → pod ortamı. Kapsam **namespace** düzeyinde |
| **S5 Kayıt defteri** | **Var** — MLMD, tipli artifact + soy ağacı |

---

# §9 — Uzman sandbox sağlayıcıları

Bunları ayrı tutuyorum çünkü **müşterileri farklı**, ve bu her şeyi değiştiriyor.

## Neden farklı düşünüyorlar

| | Uzman sağlayıcılar | Anthropic / OpenAI / biz |
|---|---|---|
| Sandbox'taki kodu kim yazıyor | Genellikle **kullanıcı** | **LLM** |
| Müşteri kim | Geliştirici, kendi kodunu koşturuyor | Son kullanıcı, kodu görmüyor bile |
| Satılan şey | "Bucket'ını mount et, esnek ol" | "Güvenli bir yetenek" |

Kullanıcının kendi kodunu koşturduğu bir üründe, kullanıcıya kendi bucket'ını
mount ettirmek **doğru** karardır. LLM'in kod yazdığı bir üründe aynı şey,
denetlenmemiş koda depo anahtarı vermek olur.

## Kim ne yapıyor

| Sağlayıcı | Mekanizma | Anahtar sandbox'ta mı |
|---|---|---|
| **E2B** | Sandbox içinde `sudo s3fs` / `gcsfuse` | **Evet** — `/root/.passwd-s3fs` dosyasına yazılıyor |
| **Modal** | `CloudBucketMount` | Belirsiz (secret mount'a veriliyor) |
| **Daytona — Volumes** | Platform mount ediyor, `subpath` ile daraltma | Hayır |
| **Daytona — External storage** | Snapshot'a `mount-s3`/`gcsfuse`/`rclone` gömülüyor | **Evet** — `envVars` ile |
| **Vercel — düz** | `sudo mount-s3` | **Evet** — dokümanı bunu açıkça uyarıyor |
| **Vercel — proxy'li** | `mount-s3 --no-sign-request` + imzalayan Function | Hayır |
| **Cloudflare Sandbox** | `sandbox.mountBucket()` (s3fs) | Hayır (binding/proxy) / Evet (düz) |
| **Fly.io Sprites** | Depo **blok cihaz** olarak kernel'e, üstünde ext4 | Yok — S3'ü hiç görmüyor |

Fly.io en uç nokta: FUSE bile değil, nesne deposunu ham disk gibi sunup üstüne
normal bir dosya sistemi kuruyorlar. "Bilgisayardaki disk gibi" tanımına en
yakın olan bu.

## FUSE ayrıcalık problemi ve üç çözümü

FUSE mount etmek ayrıcalık ister (`/dev/fuse`, `CAP_SYS_ADMIN`). Sahada üç
farklı yerde çözülüyor:

| Çözüm | Kim | Ayrıcalık nerede |
|---|---|---|
| microVM içinde `sudo` | E2B, Vercel, Daytona | Sandbox'ın **kendi** çekirdeğinde — host'a dokunmuyor |
| Ayrı sidecar/pod | GKE gcsfuse CSI, Mountpoint CSI v2 | Sandbox'ın **dışında** |
| Privileged node daemon | `k8s-csi-s3` | Node'da, privileged |

Birinci satır kritik: E2B/Vercel `sudo` verebiliyor **çünkü sandbox zaten bir
microVM** — kendi kernel'i var. Düz container'da aynı şey aynı anlama gelmez.

Kubernetes tarafında ise ayrıcalıklı süreci dışarı almak standart:
> *"The Cloud Storage FUSE CSI driver does not need privileged access. This
> minimizes the risks associated with privileged access and leads to a better
> security posture."*

## Kubernetes'te mount seçenekleri (ve neden hepsi kısıtlı)

| Seçenek | Kısıt |
|---|---|
| `mountpoint-s3-csi-driver` | Yalnızca AWS S3 — **MinIO desteklemiyor** |
| `gcsfuse csi` | Yalnızca GCS |
| `k8s-csi-s3` | **Privileged container gerekiyor** |
| COSI | Hâlâ alpha; `Key` modunda anahtar pod'a mount ediliyor |

Son satır önemli: COSI *provisioning*'i çözüyor ama **güvenilmeyen kod**
problemini çözmüyor.

## En keskin ortak bulgu

> Mount eden sağlayıcıların **hiçbirinde** artifact registry yok. Mount edilen
> bucket'a yazılan dosyanın `artifact_id`'si, lineage'i, TTL'i, content-hash'i
> yok — sadece bir S3 anahtarı var. Dokümanlar bunu "persistent data access",
> "share state across Sandboxes", "stream large datasets" diye pazarlıyor;
> **hiçbiri "artifact" demiyor.**

Tek istisna Anthropic — ve orada zaten mount yok, Files API var.

**Kural olarak:** kayıt defteri, yalnızca yazma yolu **bir bileşenden
geçtiğinde** ayakta kalıyor. Doğrudan mount = "sadece bucket".

---

# §10 — Karşılaştırma tabloları

## §10.1 — İzolasyon (S1)

Zayıftan güçlüye:

| Yöntem | Kim kullanıyor | Açılış hızı |
|---|---|---|
| Normal container (runc) | **Biz** | ~1.6 sn (ölçtük) |
| V8 isolate | Cloudflare Code Mode | milisaniyeler |
| gVisor | GKE Agent Sandbox (birincil) | saniye altı |
| Kata / microVM | Red Hat önerisi, GKE opsiyonel, E2B/Vercel | saniyeler |
| Hyper-V | Microsoft | havuzdan milisaniyeler |

## §10.2 — Sandbox ömrü (S2)

| Model | Kim |
|---|---|
| Her çağrıda yeni, saklanmaz | **Biz** |
| Efemer + cooldown'da yok edilir | Microsoft |
| Varsayılan yeni, id ile 30 gün | **Anthropic** |
| Adlandırılmış, 14 gün | Google Agent Engine |
| 15 dk – 8 saat | AWS |
| Kalıcı workspace + snapshot | OpenAI |
| Tek kullanımlık isolate | Cloudflare Code Mode |

## §10.3 — Depoya erişim (S3) ve anahtar (S4)

| Yaklaşım | Kim | Anahtar sandbox'ta |
|---|---|---|
| **Mount, anahtar içeride** | E2B, Daytona (external), Vercel (düz) | **Evet** |
| **Mount, anahtar dışarıda** | Cloudflare (binding/proxy), Vercel (proxy), Daytona (Volumes) | Hayır |
| **Blok cihaz** | Fly.io | Yok |
| **SDK, anahtar pod'da** | Red Hat OpenShift AI, AWS (IAM role) | Evet (dar) |
| **API, anahtar hiç yok** | Anthropic, OpenAI, **biz** | Hayır |

## §10.4 — Kayıt defteri (S5)

| Var | Yok |
|---|---|
| Anthropic (Files API) | E2B, Modal, Daytona, Vercel, Cloudflare, Fly.io |
| Red Hat / KFP (MLMD) | Microsoft, Google (sandbox durumu var ama registry değil) |
| **Biz** | AWS (denetim CloudTrail'de, registry değil) |

## §10.5 — Süreklilik anahtarı

Üçü de aynı deseni kullanıyor: **istemci bir kimlik saklar, geri gönderir.**

| Kim | Alan adı | Neyi canlandırıyor |
|---|---|---|
| Anthropic | `container` id | **Container'ın kendisi** (checkpoint'ten) |
| Google | `sandbox_name` | **Sandbox'ın kendisi** (14 gün) |
| Microsoft | `identifier` | **Session'a yönlendirme** |
| **Biz** | oturum uuid'si | **Sadece depo kapsamı** — container her seferinde yeni |

Bu tablo, bizimle diğerleri arasındaki en ince ama en önemli farkı gösteriyor:
aynı arayüz, farklı mekanizma.

## §10.6 — Keşif: ajan artifact'ten çekeceğini nasıl anlıyor

Beş ayrı desen var. Zayıftan güçlüye:

| # | Desen | Nasıl çalışıyor | Kim |
|---|---|---|---|
| 1 | **Sadece tool tarifi** | Tool listesinde fonksiyon durur; model çağırmayı *seçmek* zorunda | **Biz** |
| 2 | **Referans otomatik context'e düşer** | Üretilen dosyanın `file_id`'si tool sonucunda döner → konuşmaya girer | **Anthropic** |
| 3 | **Dosya sistemi + `ls`** | Sandbox yaşıyorsa model bash ile dizine bakar | Anthropic (container reuse), Google Agent Engine, OpenHands |
| 4 | **İsimler prompt'a otomatik enjekte** | İsimler model talimatlarında listelenir, içerik talep üzerine eklenir | **Google ADK `LoadArtifactsTool`** |
| 5 | **Semantik arama** | Belgeler vektör deposuna ingest edilir, `file_search` ile aranır | Llama Stack (OpenShift AI) — ama bu RAG, workflow-artifact değil |
| — | **Keşif YOK, statik bağlama** | DAG önceden yazılı; driver girdiyi çözer, container hazır doğar | KFP / Argo / Airflow / Tekton |

### Desen 4'ün üç kuralı (en olgun çözüm)

1. **İsimler her zaman context'te** — ucuz, model unutamaz
2. **İçerik talep üzerine** — pahalı olan sadece istendiğinde
3. **İçerik geçmişe kalıcı yazılmaz** — bir kez yüklenen 50 MB sonraki turlarda
   taşınmaz

### Bizim durumumuz

| Desen | Bizde |
|---|---|
| 1 — Tool tarifi | ✓ **tek dayanağımız** |
| 2 — Referans otomatik context'te | ✗ |
| 3 — Dosya sistemi `ls` | Kısmen (pod yeni doğduğu için `/output` boş; isim bilinirse tembel doldurma çalışır) |
| 4 — İsimler prompt'a enjekte | ✗ |
| 5 — Semantik arama | ✗ (N onlarca mertebesindeyken gerekmiyor) |

Yani beş desenden **yalnızca en zayıfı** bizde tam. Manifest'i sandbox içinde
çekiyoruz ama **LLM onu görmüyor.**

## §10.7 — Ağ duruşu

| Duruş | Kim |
|---|---|
| Tamamen kapalı | **Anthropic** |
| Varsayılan reddet + allowlist | Red Hat/OpenShell, GKE Agent Sandbox, **biz** |
| Açık, yapılandırılabilir | **AWS** |
| Opsiyonel kontroller | Microsoft |

---

# §11 — Bizim mimarimiz, en baştan en sona

Yukarıdaki şirketlerle aynı formatta anlatıyorum: önce tez, sonra bileşenler,
sonra **bir turun tam hikâyesi adım adım**, sonra §1'deki beş soru.

> **Bu bölüm 2026-09-04 akşamı itibarıyla günceldir.** O gün mimaride beş şey
> değişti: artifact işi Tool Gateway'den ayrı bir servise çıktı, prefetch
> kaldırılıp tembel doldurma geldi, TTL reaper eklendi, oturum kimliği kalıcı
> hâle geldi, ve artifact'ler KFP tipleriyle etiketlenmeye başladı.

## §11.0 — Tez tek cümlede

> Sandbox'ın **ölmesi** güvenlik için gerekli, ürettiğinin **kalması** iş için
> gerekli. O yüzden sandbox'ın ömrü, ürettiğinin ömrünü belirlememeli.

## §11.1 — Bileşenler: ne nerede çalışıyor

Cluster'da dört şey duruyor:

| Bileşen | İşi | İnternet | Depo |
|---|---|---|---|
| **Sandbox pod** | LLM'in kodunu çalıştırır. Her çalıştırmada YENİ, sonunda silinir | ✗ | ✗ |
| **Tool Gateway** | 10 tool'u sunar (`search_knowledge_base`, `web_search`…) | ✓ 3 onaylı FQDN | ✗ |
| **Artifact Service** | Artifact baytları + kayıt defteri | ✗ | ✓ |
| **MinIO** | Baytların durduğu S3-uyumlu depo | ✗ | — |

Artı: bir **CronJob** (TTL reaper) saat başı çalışıyor.

```
┌── Sandbox Pod (her çalıştırmada YENİ, ~3.14 sn) ──────────┐
│  /sandbox  configMap  → LLM'in kodu (salt okunur)         │
│  /scratch  emptyDir   → geçici (ölmesi İSTENİR)           │
│  /output   emptyDir   → süpürülür + tembel doldurulur     │
│  kalıcı disk YOK · S3 anahtarı YOK · DNS YOK              │
└──────────┬────────────────────────────┬───────────────────┘
           │ MCP/HTTP                   │ akışlı HTTP
           │ (yalnızca tool'lar)        │ (yalnızca artifact)
┌──────────▼──────────┐      ┌──────────▼──────────────┐
│    Tool Gateway     │      │   Artifact Service      │
│  internet ✓ depo ✗  │      │   internet ✗ depo ✓     │
└─────────────────────┘      └───┬──────────────────┬──┘
                                 │ S3               │ SQLite
                          ┌──────▼─────┐   ┌────────▼──────┐
                          │   MinIO    │   │ kayıt defteri │
                          │  BAYTLAR   │   │ (PVC)         │
                          └────────────┘   └───────────────┘
```

**Bu ayrımın sebebi:** eskiden tek pod üç işi birden yapıyordu — tool proxy'si,
kayıt defteri, MinIO anahtarı taşıyıcısı. Şimdi iki yetenek iki ayrı pod'da.
Tool Gateway ele geçirilse depoya ulaşılamıyor; Artifact Service ele geçirilse
internete çıkılamıyor.

**Doğrulandı:** gateway pod'undan `minio:9000` bağlantısı zaman aşımına düşüyor,
ve ortamında `AWS_*`/`BUCKET_*` değişkenlerinden hiçbiri yok — rota da sır da
alındı.

## §11.2 — Bir turun tam hikâyesi, adım adım

Örnek: kullanıcı *"satışları çıkar"* diyor, sonra *"toplamı da söyle"* diyor.

### Adım 1 — Oturum kimliği (her şeyin anahtarı)

Tarayıcı `localStorage`'dan bir uuid okur; yoksa üretir ve saklar. WebSocket'e
`?session=<uuid>` olarak gider. CLI'de karşılığı `--session` bayrağı.

Bu **tek** değer **iki** yere birden gidiyor:

```
oturum kimliği ──> thread_id     (konuşmanın hafızası)
               └─> workflow_id   (artifact'lerin kapsamı)
```

**Neden önemli:** bu değer daha önce her bağlantıda yeniden üretiliyordu. Sonuç,
sayfa yenilenince artifact'lerin depoda **sağ kalıp erişilemez** hâle
gelmesiydi — kalıcı bir depoya yazıp okuma anahtarını çöpe atmak. Kimlik
gönderilmezse temiz oturum açılır: **kalıcılık opt-in**.

Biçim UUID'ye kilitli, çünkü bu değer S3 anahtarının parçası oluyor.

### Adım 2 — Agent kod yazar

LangGraph ajanı `run_ptc_code(code)` çağırır. Bu, veriye erişmenin **tek**
yoludur. Bütçe: bir turda en fazla **2** çalıştırma (birincisi hata verirse
ikincisi düzeltme).

### Adım 3 — Kapsam jetonu imzalanır

`sandbox_runner` bir Kubernetes Secret'ından imza anahtarını okur, HMAC-SHA256
ile **15 dakika** ömürlü bir jeton üretir. İçinde `workflow_id`, `run_id`,
`owner`, `node_id` var.

**İncelik:** jeton laptop'ta — sandbox'ın erişemediği yerde — imzalanır.
Sandbox'ın eline yalnızca imzalanmış sonuç geçer, imza anahtarı hiç geçmez.
Bu yüzden sandbox jetonu okuyabilir ama **başka bir workflow için geçerli bir
jeton üretemez**.

### Adım 4 — Pod doğar

Bir Kubernetes Job yaratılır. Sırayla:
1. Kod bir ConfigMap'e yazılır (`ptc-code-{run_id}`)
2. Jeton imzalanır
3. Tool Gateway'in **ve** Artifact Service'in ClusterIP'leri çözülür (DNS adı
   değil — sandbox'ın DNS'e hiç ihtiyacı olmasın diye)
4. Job yaratılır
5. ConfigMap'e Job'u gösteren bir `ownerReference` konur — Job silinince
   Kubernetes ConfigMap'i de kaskad siler

Pod'un ağ çıkışı **iki** iç hedefle sınırlı. MinIO'ya rota yok, S3 anahtarı yok,
DNS yok.

### Adım 5 — Manifest çekilir (tek istek)

Pod açılır açılmaz Artifact Service'e **bir** istek gider: *"bu workflow'da
hangi isimler var?"* Sadece isimler döner, bayt dönmez.

**Neden sadece isimler:** önceki tasarımda pod doğarken depodaki **her**
artifact indiriliyordu. Maliyet var olan her şeyle ölçekleniyordu — 6 tane
100 MiB'lik artifact biriktiğinde, hiçbirine dokunmayan bir script bile
512Mi'lık `/output`'u patlatıyordu. Şimdi maliyet **kullanılan kadar**.

### Adım 6 — Kod çalışır

`main()` şu sırayla ilerliyor:

```python
istemci = ArtifactClient(...) if SCOPE_TOKEN and ENDPOINT else None
artifact_globals, artifact_internal = _artifact_api(istemci) if istemci else ({}, {})

inenler = {}
if istemci:
    tembel_globals = _tembel_okumayi_kur(istemci, _manifest(istemci), inenler)

try:
    exec(kod, sandbox_globals)
except Exception as exc:
    _ciktilari_supur(artifact_internal, inenler)   # ← hata olsa BİLE önce süpür
    print(hata); exit(0)

_ciktilari_supur(artifact_internal, inenler)       # ← başarıda da
print(sonuc)
```

Dikkat: süpürme **hata yolunda da** çalışıyor. Script son satırda patlasa bile
o ana kadar üretilen dosyalar kurtarılıyor — asıl değeri de burada.

### Adım 7 — Artifact Service dört kontrolü akış sırasında yapar

Baytlar akarken:

| Kontrol | Nasıl | Neden |
|---|---|---|
| **Kapsam** | `workflow_id` **jetondan** okunur, çağıranın iddiasından değil | Paylaşılan depo, ağ politikasının **göremediği** bir kanal açar |
| **Format** | pickle — hem etiketten hem **ilk iki bayttan** | Deserialization kod çalıştırır (CWE-502); baytları LLM yazdı |
| **Boyut** | Sayarak; sınır aşılınca okuma **orada** kesilir | Veriyi sandbox içinde süzmeye zorlar |
| **İsim** | `../`, `/`, boşluk reddedilir | `artifact_save("/etc/shadow")` sınıfı |

**"Akış sırasında" olması kritik:** pickle imzası ilk iki bayttadır, yani ilk
parçada karar verilir ve reddedilen yükleme depoya **tek bayt bile** yazmaz.
Gövde 8 MiB'ı aşınca diske taşar — süreç belleği yükleme boyutundan bağımsız.

### Adım 8 — İki ayrı depoya yazılır

```
baytlar → MinIO                → "veri burada"
kayıt   → SQLite kayıt defteri → "o verinin hikâyesi"
```

Kayıt defterinde: `artifact_id`, isim, **tip** (`system.Dataset` vb.),
**metadata** (serbest anahtar-değer), workflow, içerik hash'i, boyut, depo
adresi, soy (`parents`), TTL.

### Adım 9 — Pod ölür

Job biter, pod silinir, `/scratch` ve `/output` yok olur. ConfigMap'i
Kubernetes'in çöp toplayıcısı alır. Geriye MinIO'daki baytlar + kayıt satırı
kalır.

### Adım 10 — Sonraki tur

Kullanıcı *"toplamı da söyle"* der. **Aynı oturum kimliği → aynı
`workflow_id`.** Yeni pod açılır, manifest çeker, `satislar.csv`'yi görür.
LLM `pd.read_csv("/output/satislar.csv")` yazınca dosya arka planda iner.

**40 saniyelik iş tekrarlanmaz.**

## §11.3 — Yazma yolu: iki tetikleyici, tek kapı

| | **A — Açık çağrı** | **B — Süpürme** |
|---|---|---|
| Nasıl | `put_artifact(df, name="satislar")` | `df.to_csv("/output/satislar.csv")` |
| LLM bilmek zorunda mı | Evet | **Hayır** |
| Ne zaman | Kod içinde istediği an | Çalışma sonunda, başarı **ya da hata** |
| Tip korunumu | Evet (Parquet) | Uzantıdan tahmin |
| Denetim | Aynı dört kontrol | Aynı dört kontrol |

**B neden var:** A yolu LLM'in API'yi *bilmesini ve hatırlamasını* gerektirir.
Bilmezse ürettiği her şey pod'la birlikte kaybolurdu. Bu, Anthropic'in
`$OUTPUT_DIR`'ı ve OpenAI'ın `/mnt/data`'sıyla aynı konvansiyon.

## §11.4 — Okuma yolu: dört keşif yolu

5. adım 1. adımın ürettiğini **dört** ayrı yoldan bulabiliyor, ve hiçbiri tek
dayanak değil:

1. **Kayıt defterine sorma** — `list_artifacts()` → isimler döner
2. **İsimle çözme** — `get_artifact(name="satislar")` → o isimdeki en yeni sürüm
3. **Dosya sistemi yanılsaması** — `pd.read_csv("/output/satislar.csv")`
4. **Konuşma hafızası** — checkpointer kalıcı olduğu için LLM kendi verdiği ismi
   hatırlıyor

### Üçüncüsü nasıl çalışıyor (tembel doldurma)

`pd.read_csv` / `read_parquet` / `read_json` / `read_excel` / `read_feather`
sarmalanmış, ayrıca sandbox'ın globals'ına tembel bir `open` konmuş. Dosya
`/output`'ta yoksa **ve** adı manifestte geçiyorsa, o an indiriliyor; sonra
pandas normal şekilde okuyor.

`builtins` **değiştirilmiyor** — sadece LLM'in doğrudan çağrısı yakalanıyor,
kütüphanelerin iç dosya işlemleri hiç etkilenmiyor. Yama alanını dar tutmanın
en temiz yolu bu.

**İnce nokta:** indirilen dosyanın (değişim zamanı, boyut) çifti kaydediliyor;
süpürme dokunulmamış olanı **atlıyor**. Yoksa sadece *okuyan* bir çalıştırma
bile dosyayı "üretilmiş" sayıp geri yüklerdi. (Prefetch döneminde gerçekten
yaşanmış bir kusur.)

## §11.5 — İki değişmez kural

**1. Değişmezlik.** Aynı isme ikinci kez yazmak eskisini ezmez; **yeni bir
`artifact_id`** doğar, okuma en yeniyi çözer.

Bunun bir yan etkisi var ve lehimize: **S3 Versioning'e hiç bağımlı değiliz.**
Aynı object key'i asla üzerine yazmıyoruz. Yani depo ürünü değişse de kod aynı
çalışır.

**2. Silme sırası: önce bayt, sonra kayıt.** Ters sıra yetim blob bırakır
(görünmez maliyet); bu sıra en kötü ihtimalle sarkan referans bırakır — o
tespit edilebilir.

Artı **içerik-hash dedup**: aynı içerik ikinci kez yazılırsa bayt tekrar
yüklenmez, yeni kayıt var olan adresi gösterir. Silme bunu biliyor — paylaşılan
baytı silmiyor.

## §11.6 — Arka planda: TTL reaper

Saat başı bir CronJob, servisin `/admin/reap` ucunu çağırıyor.

**Neden CronJob işi kendisi yapmıyor:** kayıt defteri SQLite ve PVC
`ReadWriteOnce` — tek yazıcı olabilir. İş bölümü: **CronJob zamanlayıcı, servis
tek yazıcı.**

**Yetki ayrı bir jetonda** (`PTC_ADMIN_TOKEN`). Sandbox'ın kapsam jetonu buraya
**yetmiyor** — doğrulandı, 401 alıyor. LLM'in ürettiği kodun toplu silme
tetikleyebilmesi, emniyet ağı olarak kurduğumuz kalıcılığı tek çağrıda geri
alırdı.

TTL'i olmayan artifact'lere dokunulmuyor — varsayılan bu.

## §11.7 — Beş soru (§1'deki çerçeve)

| | Cevap |
|---|---|
| **S1 Nerede** | Kubernetes Job, normal container (runc). Kata hedef, henüz yok |
| **S2 Ömür** | **Her çalıştırmada yeni**, ~3.14 sn. Saklanmıyor, canlandırılmıyor |
| **S3 Veri** | Akışlı HTTP, ayrı bir servise. Mount **yok**, presigned URL **yok** |
| **S4 Anahtar** | Sandbox'ta **hiç yok**. Anahtar Artifact Service pod'unda |
| **S5 Kayıt defteri** | **Var** — `artifact_id`, KFP tipleri, `.metadata`, soy, TTL, hash |

## §11.8 — Sayılar

| | Değer |
|---|---|
| Çalıştırma süresi | **3.14 sn** (1.62'si pod başlatma) |
| Bir turda çalıştırma sınırı | 2 |
| Kapsam jetonu ömrü | 15 dakika |
| Artifact başına boyut | 100 MiB |
| `/output` ve `/scratch` | 512Mi (emptyDir) |
| Servis bellek eşiği | 8 MiB (aşınca diske taşar) |
| Reaper sıklığı | Saat başı |
| Test sayısı | 98 |

## §11.9 — Yapılandırma sözleşmeleri

Depo bağlantısı **iki** sözleşmeyi de okuyor:

| Sözleşme | Değişkenler | Ne zaman |
|---|---|---|
| **ObjectBucketClaim** | `BUCKET_NAME`, `BUCKET_HOST`, `BUCKET_PORT` | ODF varsa |
| **OpenShift AI connection** | `AWS_S3_ENDPOINT`, `AWS_S3_BUCKET`, `AWS_DEFAULT_REGION` | **ODF yoksa** |

İkisinde de `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`. İkisi birden varsa
OBC kazanıyor.

Diğer ayarlar:

| Değişken | İşi |
|---|---|
| `PTC_ARTIFACT_ROOT` | Bucket içindeki kök — KFP'nin `pipeline_root`'u |
| `PTC_METADATA_DB` | Kayıt defteri yolu |
| `PTC_CHECKPOINT_DSN` / `PTC_CHECKPOINT_DB` | Workflow state — Postgres / SQLite |
| `PTC_SCOPE_SIGNING_KEY` | Kapsam jetonu imza sırrı |
| `PTC_ADMIN_TOKEN` | Reaper yetkisi |

## §11.10 — Bilinen açıklar (saklamıyoruz)

| Konu | Durum |
|---|---|
| **İzolasyon** | Düz container. Red Hat Kata öneriyor, kurulu değil |
| **Metadata DB** | SQLite. `open_postgres()` kodda var, test edilmedi |
| **Workflow state** | `AsyncSqliteSaver` — çalışıyor, doğrulandı. Postgres yolu (`AsyncPostgresSaver`) **test edilmedi** |
| **Yüksek erişilebilirlik** | Tek replika (SQLite + RWO PVC) |
| **Auth** | Yok — uuid'yi bilen o oturumun artifact'lerini okur |
| **Büyük dosya** | 100 MiB / 512Mi sert sınırlar. 5 GB çalışmaz |
| **Manifest promptta değil** | LLM `list_artifacts()`'i kendisi çağırmalı — yumuşak garanti |
| **Soy ağacı** | Kaydediliyor, **keşifte kullanılmıyor** |
| **İsim çakışması** | "En yeni" kazanır, LLM farkında olmaz |
| **Şeffaf okuma** | Yalnızca 5 pandas okuyucusu + `open`. `pyarrow`, `csv`, `PIL` yakalanmıyor |
| **Warm pool** | Yok (kazancı ölçüldü: ≤1.6 sn) |
| **Gerçek OBC/ODF** | Test edilmedi (ODF kapsam dışı) |

---

# §12 — Biz neredeyiz: boyut boyut örtüşme

Tek bir şirketle örtüşmüyoruz; boyut boyut farklı şirketlerle örtüşüyoruz.

| Boyut | Bizde | Kimle aynı |
|---|---|---|
| PTC tezi (kod yaz, tool çağırma) | Tek erişim yolu `run_ptc_code` | **Cloudflare** |
| Ağ | İnternet yok, DNS yok, 2 iç hedef | **Anthropic**, GKE, Red Hat |
| Depoya erişim | SDK, servis içinde; mount yok | **Red Hat**, Anthropic, OpenAI |
| Çıktı yakalama | `/output` süpürme | **Anthropic** (`$OUTPUT_DIR`), OpenAI (`/mnt/data`) |
| `.uri` ↔ `.path` kopyalama | Süpürme + tembel doldurma | **KFP launcher** |
| Kayıt defteri | `artifact_id`, tip, soy, TTL, hash | **Anthropic**, **KFP/MLMD** |
| Artifact tipleri | `system.*` (aynı sözlük) | **KFP/MLMD** |
| Yapılandırılabilir kök | `PTC_ARTIFACT_ROOT` | **KFP `pipeline_root`** |
| Süreklilik anahtarı | İstemci uuid geri gönderiyor | Anthropic/Google/Microsoft (desen), mekanizma farklı |
| Sandbox ömrü | Her çalıştırmada yeni (~3.14 sn) | **Microsoft** (ruhen) |
| Kapsam granülaritesi | **Çalıştırma başına** imzalı jeton | **Kimse** — KFP namespace, diğerleri sandbox |
| İzolasyon | Normal container | **Kimse** — herkes daha güçlü |
| Warm pool | Yok | Microsoft/Google'da var |
| Şeffaf okuma (`read_csv` → tembel indirme) | Var | **Kimse** |

## Üç cümlelik özet

1. **Tezi Cloudflare'den, veri modelini Anthropic'ten, platform desenini
   Red Hat'ten alıyoruz.** Omurga tartışmasız SOTA.
2. **İki yerde kimsenin gerisindeyiz:** izolasyon (düz container) ve warm pool.
   Birincisi gerçek bir eksik, ikincisinin kazancını ölçtük — en fazla 1.6 sn.
3. **İki yerde kimsenin ilerisindeyiz:** kapsam granülaritesi (çalıştırma
   başına, diğerleri en iyi ihtimalle namespace) ve şeffaf okuma. İkincisi
   emsalsiz olduğu için **test edilmemiş** sayılmalı, üstünlük değil.

---

# §13 — Ekipten gelebilecek sorular ve cevapları

Bu bölüm sunumda ya da kod incelemesinde gelecek soruların hazır cevapları.
Her cevap kısa, sonra gerekirse detaya inilir.

### S: "Neden S3'ü mount etmiyoruz? Herkes ediyor."

**Kısa cevap:** Bizim işimize en çok benzeyen iki sistem (Anthropic, OpenAI) de
etmiyor. Ve mount eden hiçbir sistemde kayıt defteri kalmıyor.

**Detay:** Mount edenlerin müşterisi geliştirici — sandbox'ta **kullanıcının
kendi kodu** çalışıyor. Bizde LLM'in kodu çalışıyor. Ayrıca mount edince
"yazma anı" bir kernel çağrısı hâline geliyor; pickle reddi, boyut sınırı,
isim denetimi yapacak bir katman kalmıyor. Elimizde artifact store değil,
sadece bucket olur.

### S: "Peki mount edemez miydik? Teknik engel mi var?"

**Kısa cevap:** Edebilirdik, teknik engel yok — ama Kubernetes'teki dört
seçeneğin dördü de bize uymuyor.

**Detay:** `mountpoint-s3-csi-driver` yalnızca AWS S3 (MinIO desteklemiyor),
`gcsfuse csi` yalnızca GCS, `k8s-csi-s3` privileged container istiyor, COSI
hâlâ alpha ve `Key` modunda anahtarı pod'a mount ediyor. Ayrıca sandbox'a
`/dev/fuse` vermek zorunda değildik — endüstri deseni ayrıcalıklı FUSE sürecini
dışarı almak — ama yukarıdaki dört kısıt yine de kalıyordu.

### S: "Neden sandbox'ı her seferinde yeniden yaratıyoruz? Yavaş değil mi?"

**Kısa cevap:** 3.14 saniye, ve bunun sadece 1.62'si pod başlatma. Warm pool
kursak kazanacağımız en fazla 1.6 saniye.

**Detay:** Ölçtük. Karmaşıklığa değmiyor. Cloudflare de farklı gerekçeyle aynı
sonuca varıyor — isolate zaten hafif olduğu için havuza gerek duymuyorlar.
Microsoft ve Google havuz kuruyor ama onların ölçeği farklı (binlerce eşzamanlı
session).

### S: "Anthropic 30 gün container saklıyor, biz neden saklamıyoruz?"

**Kısa cevap:** Onlar **container'ı** saklıyor, biz **artifact'i** saklıyoruz.
İkisi farklı problem.

**Detay:** Anthropic'te `container` id'yi geri gönderince ortam checkpoint'ten
canlanıyor — değişkenler, dosyalar, hatta Python yorumlayıcı durumu. Bizde
oturum uuid'si hiçbir şeyi canlandırmıyor; sadece "artifact'leri hangi kapsamda
arayacağız" diyor. Container her seferinde sıfırdan doğuyor.

Bizim yaklaşımın avantajı: izolasyon her çalıştırmada tam. Dezavantajı:
yorumlayıcı durumu taşınmıyor, her şey artifact'e yazılmak zorunda.

### S: "Neden PostgreSQL değil SQLite?"

**Kısa cevap:** PoC. Ve tek replikayla sınırlı olmamızın sebebi bu.

**Detay:** KFP'nin MLMD'si gerçek bir DB'de duruyor, Red Hat de üretimde
PostgreSQL diyor. Bizde `open_postgres()` kodda hazır, sadece test edilmedi —
cluster'da Postgres yok. Metadata SQLite'ı `ReadWriteOnce` bir PVC'de olduğu
için servis şu an >1 replikaya çıkamıyor. Çok replika gerektiğinde geçilecek.

### S: "İzolasyon zayıf değil mi? Herkes gVisor/Kata kullanıyor."

**Kısa cevap:** Evet, bu listede en zayıf olan biziz. Gerekçesi yok, sadece
henüz yapılmadı.

**Detay:** Red Hat AI-üretimi kod için açıkça Kata öneriyor, Google gVisor
kullanıyor, Microsoft Hyper-V. Bizde düz container. OpenShift Sandboxed
Containers kuruluysa `runtimeClassName` ile açılır — kod değişikliği
gerektirmez, node seviyesinde bir önkoşul. Ekibe sorulacak dört sorudan biri bu.

### S: "İnterneti tamamen kapatmak fazla katı değil mi? AWS açık bırakıyor."

**Kısa cevap:** İki farklı tehdit modeli. AWS'de sandbox'ta kullanıcının kodu
olabiliyor; bizde her zaman LLM'in kodu var.

**Detay:** AWS "network modes" ile yapılandırılabilir bırakıyor ve denetimi
CloudTrail'e yıkıyor. Anthropic tam tersi — *"Internet access: Completely
disabled for security"*. Biz Anthropic tarafındayız, ama tam kapalı değiliz:
sandbox iki iç servise çıkabiliyor (tool gateway + artifact service), internete
çıkamıyor.

### S: "Artifact tipleri niye `system.Dataset` gibi garip isimler?"

**Kısa cevap:** Kubeflow'un MLMD şema başlıklarıyla **birebir aynı** olsun diye.

**Detay:** OpenShift AI'ın pipeline motoru KFP. Aynı isimleri kullanmak, ileride
bir KFP pipeline'ına ya da MLMD'ye bağlanmayı isim çevirisi gerektirmeyen bir
işe indiriyor. Kendi isimlerimizi uydursaydık her entegrasyonda çeviri tablosu
yazacaktık.

### S: "Tip nereden biliniyor? LLM yazmak zorunda mı?"

**Kısa cevap:** Hayır, otomatik çıkarılıyor. Açıkça verilirse o kazanıyor.

**Detay:** DataFrame/Parquet/CSV → `Dataset`, sayısal sözlük → `Metrics`,
`.html` → `HTML`. Bir incelik var: "sayısal sözlük = metrik" ancak **nesneye**
bakarak anlaşılır — serileştirilince o da sadece `application/json`. Bu yüzden
çıkarım **sandbox tarafında**, nesne hâlâ elde iken yapılıyor. Ortak kural
`serialize.py`'de, o dosya sandbox imajına aynen kopyalandığı için iki taraf
ayrışamıyor.

### S: "`pipeline_root` neden lazımdı?"

**Kısa cevap:** "Bucket'ın neresine yazılacağı" KFP'de **yapılandırma**, bizde
sabit koddu.

**Detay:** KFP bunu üç düzeyde ayarlatıyor — dağıtım varsayılanı (ConfigMap),
pipeline başına, çalıştırma başına. Bizde artık `PTC_ARTIFACT_ROOT` (dağıtım)
ve `X-Artifact-Root` (çağrı başına) var. Verilmezse eski düzen korunuyor.

### S: "5 GB'lık bir dosyayla çalışmam gerekirse?"

**Kısa cevap:** Şu an **çalışmaz**. İki sert sınır var.

**Detay:** Artifact başına 100 MiB ve `/output` emptyDir 512Mi. AWS aynı
problemi ikiye ayırarak çözüyor — inline 100 MB, S3 üzerinden 5 GB. Bizde
karşılığı, Vercel/Cloudflare'in "kimlik-bilgisiz mount + imzalayan proxy"
deseni olurdu: mount'un ergonomisi + aracının kontrolü. Yapılmadı, ama
gelecekte "büyük dosya için mount, küçük için servis" melezine gidilecek yol bu.

### S: "5. adım 1. adımın ürettiğini nasıl buluyor?"

**Kısa cevap:** Soruyor. Klasik pipeline'da söylenirdi, ajan dünyasında sormak
zorunda.

**Detay:** Airflow/Argo/KFP'de DAG önceden yazılıdır, 5. adımın girdisi
**bağlanmıştır** — driver çözer, adım hazır girdiyle doğar. Ajan dünyasında
5. adım *yazılmadan önce var değil*, o yüzden keşif zorunlu. Bizde dört yol
var: `list_artifacts()` (kayıt defterine sorma), `get_artifact(name=...)`
(isimle en yeni sürüm), `/output/x.csv` okuma (tembel doldurma), ve konuşma
hafızası. Hiçbiri tek dayanak değil.

### S: "Peki LLM `list_artifacts()` çağırmayı unutursa?"

**Kısa cevap:** Bu gerçek bir açık. Şu an **yumuşak garanti**, ve kopyalanacak
hazır bir desen var.

**Detay:** Sistem promptu "önceki bir sonuca atıf varsa ÖNCE bunu çağır" diyor
ama model unutabilir. Sert garanti, isimleri her turda prompt'a enjekte etmek
olurdu. Bu **Google ADK'da `LoadArtifactsTool` olarak yerleşik**: isimleri model
talimatlarına koyuyor, içeriği model istediğinde geçici olarak ekliyor, ve
içeriği geçmişe kalıcı yazmıyor. Bizde henüz yok. Bkz. §10.6.

### S: "OpenShift bu keşif işini nasıl yapıyor? Ondan kopyalayalım."

**Kısa cevap:** Kopyalanacak bir şey yok — OpenShift'te bu sorunun yerleşik
cevabı **yok**.

**Detay:** Sebebi yapısal. KFP/DSP'de artifact var ama **ajan yok** — DAG
statik, 5. adımın girdisi bağlanmış, driver çözüyor, container hazır doğuyor.
Llama Stack'te ajan var ama **workflow-artifact'i yok** — orada `file_search`
var, o da RAG (ingest edilmiş belgeler üzerinde semantik arama). Bizim durumumuz
tam bu ikisinin arasındaki boşlukta.

Artifact **depolaması** için OpenShift deseni var ve kopyaladık (S3, SDK,
launcher, MLMD tipleri, `pipeline_root`). Artifact **keşfi** için yok; oradaki
referans Google ADK. Bkz. §8.5.

### S: "Artifact sayısı çok artarsa isim listelemek yetmez, ne yaparız?"

**Kısa cevap:** Llama Stack'in yaptığını yaparız — listelemek yerine aramak.

**Detay:** Bizim N'imiz onlarca mertebesinde olduğu sürece isimleri listelemek
doğru ve ucuz. Yüzlere çıkarsa `file_search` benzeri semantik arama daha
mantıklı hale gelir: artifact adları ve `.metadata` alanları üzerinde arama.
Şimdilik gerekmiyor, tasarım seçeneği olarak duruyor.

### S: "Kayıt defterinde soy ağacı (lineage) var mı?"

**Kısa cevap:** Kaydediliyor ama **keşifte kullanılmıyor**.

**Detay:** Her artifact `parents` alanı taşıyor ve doldurulabiliyor. Ama
"1. adımın çıktısından türeyen ne var?" diye sorgulayan kod yok. KFP'de MLMD
bunu keşif için kullanır; bizde şimdilik sadece kayıt.

### S: "İki adım aynı ismi kullanırsa ne olur?"

**Kısa cevap:** "En yeni" kazanır, ve LLM bunun farkında olmaz.

**Detay:** Klasik pipeline'da bu imkânsız çünkü kenarlar açıkça bağlı. Bizde
sessiz bir kayıp riski. Bilinen sınır.

### S: "Bu uuid'yi bilen başkasının artifact'lerini okuyabilir mi?"

**Kısa cevap:** Evet. Bu kimlik doğrulama **değil**.

**Detay:** Uuid tahmin edilemez olduğu için PoC'de yeterli; üretimde gerçek
auth'a bağlanmalı. Bunu saklamıyoruz, karar dokümanında da yazılı.

---

# §14 — Doğrulanamayanlar

Ekibe sunmadan önce bakılması gerekenler:

1. **Red Hat "kimlik bilgileri sandbox'ta saklanmaz, ağ sınırında enjekte
   edilir"** cümlesi — arama özetinde çıktı, makalenin kendisinden birebir
   teyit edemedim.
2. **OpenAI bölümü** (§3) — bu turda birincil kaynaktan yeniden çekilmedi,
   önceki araştırma taramamıza dayanıyor.
3. **Cloudflare Sandbox'ın kalıcılık modeli** — sandbox'ın kendisi mi kalıcı,
   yoksa kalıcılık tamamen Durable Objects'ten mi geliyor, doküman açıkça
   söylemiyor.
4. **Modal'ın kimlik bilgisi konumu** — secret mount'a veriliyor ama sandbox
   içinden görünüp görünmediği dokümandan çıkmıyor.
5. **OCP erişim modları tablosu** (hangi CSI sürücüsü RWX veriyor) —
   docs.redhat.com sayfaları içerik yerine gezinme menüsü döndürdü. §8.3'teki
   RWX listesi genel Kubernetes bilgisine dayanıyor, Red Hat tablosundan alıntı
   **değil**. Kümede `kubectl get storageclass` ile bakılmalı.
6. **`AWS_S3_BUCKET`** değişken adı ikincil kaynaktan; `AWS_ACCESS_KEY_ID`,
   `AWS_SECRET_ACCESS_KEY`, `AWS_S3_ENDPOINT`, `AWS_DEFAULT_REGION` birincil
   dokümanda birebir geçiyor.

---

# Kaynaklar

**Anthropic**
- [Code execution tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)
- [Programmatic Tool Calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)

**Cloudflare**
- [Cloudflare Sandbox](https://developers.cloudflare.com/sandbox/)
- [Code Mode](https://blog.cloudflare.com/code-mode/)
- [Sandbox — Storage](https://developers.cloudflare.com/sandbox/api/storage/)

**Google**
- [Agent Engine Code Execution](https://cloud.google.com/agent-builder/agent-engine/code-execution/overview)
- [Code Execution troubleshooting](https://docs.cloud.google.com/agent-builder/agent-engine/troubleshooting/code-execution)
- [About GKE Agent Sandbox](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/machine-learning/agent-sandbox)

**AWS**
- [Bedrock AgentCore Code Interpreter](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html)

**Microsoft**
- [Dynamic sessions in Azure Container Apps](https://learn.microsoft.com/en-us/azure/container-apps/sessions)

**Red Hat / OpenShift**
- [OCP 4.17 Storage](https://docs.redhat.com/en/documentation/openshift_container_platform/4.17/html-single/storage/index)
- [OpenShift AI — Managing data science pipelines](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.25/html/working_with_data_science_pipelines/managing-data-science-pipelines_ds-pipelines)
- [OpenShift AI — Connect workbench to S3](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.25/html-single/working_with_data_in_an_s3-compatible_object_store/index)
- [OpenShift Pipelines 1.16](https://docs.redhat.com/en/documentation/red_hat_openshift_pipelines/1.16/html/about_openshift_pipelines/understanding-openshift-pipelines)
- [Layered sandboxing for AI agents](https://developers.redhat.com/articles/2026/07/16/layered-sandboxing-ai-agents-openshift-and-openshell)

**Kubeflow Pipelines**
- [Object Store Configuration](https://www.kubeflow.org/docs/components/pipelines/operator-guides/configure-object-store/)
- [Create, use, pass, and track ML artifacts](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/)
- [Pipeline Root](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/pipeline-root/)

**GKE FUSE CSI**
- [Cloud Storage FUSE CSI driver](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/cloud-storage-fuse-csi-driver)

**Uzman sağlayıcılar** — ayrıntılı tablo, alıntılar ve tek tek kaynak linkleri:
[PTC_Sandbox_Depo_Dogrudan_Erisim_Arastirmasi.md](PTC_Sandbox_Depo_Dogrudan_Erisim_Arastirmasi.md) §2.1
