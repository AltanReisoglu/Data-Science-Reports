# Sandbox ve Artifact Mentaliteleri — Kim Nasıl Düşünüyor

**Tarih:** 2026-09-06 · **Kod durumu:** 179 test + 38 canlı kabul kontrolü geçiyor

> **Bu turda değişen ana karar:** LLM'e sunulan artifact API'si kaldırıldı;
> yerine KFP launcher deseni geçti (düz Python + `/output`). Gerekçe ve bunu
> zorlayan hatalar §11.11'de; piyasadaki karşılığı §9.6'da.

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
| **§9.5** | **Artifact storage** — baytlar fiilen hangi üründe duruyor, ne kadar |
| **§9.6** | **Aracı nerede duruyor** — baytı kim taşıyor, sarmalayıcı mı sınır mı |
| **§10** | Karşılaştırma tabloları (izolasyon, ömür, erişim, kayıt defteri, ağ) |
| **§11** | **Bizim mimarimiz, en baştan en sona** (§11.10 açıklar, §11.11 ne değişti) |
| **§12** | Biz neredeyiz — boyut boyut kiminle örtüştüğümüz |
| **§13** | Ekipten gelecek soruların hazır cevapları |
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
| **S3 Veri** | Üçe ayrık: inline 100 MB / terminalden S3'e 5 GB / oturuma **mount** edilmiş S3 Files ya da EFS (bkz. §9.5.5) |
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

**Ama Tekton'ın bir "artifact" kavramı da var** — ve bu, workspace'ten
tamamen ayrı bir şey. Step, `$(step.artifacts.path)`'e bir JSON yazıyor:
`uri` + `digest` (+ isteğe bağlı `buildOutput`). Tekton Chains bunu görüp
**imzalı bir attestation** üretiyor.

Kritik nokta: burada saklanan **bayt değil, künye**. Doküman açık —
*"only metadata and attestations are stored, not artifact content."*
Yani Tekton'ın artifact'i bir **referans**, bizimki gibi bir **kap** değil.
Ayrıntılı karşılaştırma §8.6'da.

## §8.4 — Güvenilmeyen kod için Red Hat'in rehberi

**İzolasyon — katmanlı:**
> *"Running OpenShell inside an OpenShift sandboxed containers VM gives you
> both layers simultaneously."*

Yani AI-üretimi kod için **Kata**.

**Ağ — varsayılan reddet:**
> *"The default posture is deny-all. In practice, you write a policy that
> allowlists exactly the endpoints your agent needs."*

**Ürünleşen taraf — Red Hat build of Agent Sandbox (Technology Preview):**
Yukarıdaki makale "henüz ürün değil" derken, Red Hat aynı problemi ürünleştiren
bir şey de çıkardı: `kubernetes-sigs/agent-sandbox`'ın downstream'i.
Tanımı birebir bizim durumumuz:

> *"provides autonomous AI agents a more security-rich, virtual machine
> (VM)-isolated place to **execute untrusted code**"*

Verdiği: `Sandbox` / `SandboxTemplate` / `SandboxClaim` CRD'leri,
**`SandboxWarmPool`** (milisaniyede tahsis), Kata ile donanım izolasyonu,
PVC ile kalıcı depolama.

`§11.10`'daki iki açığımızı — izolasyon ve warm pool — aynı anda kapatıyor.
**Ama artifact'i çözmüyor:** verdiği kalıcılık bir PVC, yani disk. Bkz. §8.7.

**Olgunluk uyarısı:** OpenShell/katmanlı sandbox çalışmaları için Red Hat
kendisi *"early validations, not shipping product features yet"* diyor;
Agent Sandbox ise **Technology Preview** — GA değil.

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

## §8.6 — OpenShift'in **iki** artifact deseni var, ve farklı soruları cevaplıyorlar

Bu ayrımı kaçırmak kolay, çünkü ikisine de "artifact" deniyor. Ama biri
**veriyi saklıyor**, diğeri **verinin künyesini imzalıyor**.

| | OpenShift AI (KFP) | OpenShift Pipelines (Tekton) |
|---|---|---|
| Cevapladığı soru | *"Veriyi adımlar arası nasıl taşırım ve saklarım?"* | *"Bu çıktının neyden üretildiğini nasıl kanıtlarım?"* |
| Artifact **nedir** | Dosyanın kendisi | `uri` + `digest` — **referans**, bayt değil |
| Baytlar nerede | Nesne deposu (`pipeline_root`) | Başka yerde: container registry, PVC, git |
| Nasıl bildiriliyor | İmzada `Output[Dataset]` | Step, `$(step.artifacts.path)`'e JSON yazıyor |
| Kayıt defteri | **MLMD** (MySQL/MariaDB) | **Tekton Chains** → 7 arka uçtan biri (Grafeas, OCI, GCS, DocDB, Archivista…) |
| Ne saklanıyor | İçerik **ve** künye | **Yalnızca künye ve imza** |
| Amaç | Veri kalıcılığı | Tedarik zinciri güvenliği (SLSA/in-toto) |

Tekton'ın step'i şunu yazıyor:

```json
{"outputs": [{"name": "image",
              "values": [{"uri": "pkg:...", "digest": {"sha256": "..."},
                          "buildOutput": true}]}]}
```

Chains bunu görüp **imzalı bir attestation** üretiyor. Ama dokümanın kendisi
net: *"only metadata and attestations are stored, **not artifact content**."*

### Hangisi bizim sorumuz

Bizimki birincisi. Sandbox'ın ürettiği bir DataFrame'in "container registry'de
bir imaj" karşılığı yok; baytın kendisi saklanmak zorunda. Tekton deseni bizde
uygulanamaz — çünkü referans verecek bir yer yok, referansın işaret edeceği
şeyi de biz saklıyoruz.

O yüzden §11'deki mimari KFP'nin veri modelini alıyor, Tekton'ınkini değil.
Bu bir tercih değil, sorunun ne olduğuyla ilgili.

### Tekton'dan alınabilecek tek fikir: **imzalı** soy

Bizim soy ağacımız (`parents`) doğru ama **imzasız**. Kayıt defterine yazma
yetkisi olan biri soyu değiştirebilir ve bunu kimse fark etmez. Tekton
Chains'in çözdüğü problem tam olarak bu.

Bizde karşılığı yok — `§11.10`'a açık olarak eklendi.

---

## §8.7 — Peki OpenShift'in **güvenilmeyen kod** için artifact cevabı ne

Yok. Ve bu, dokümandaki en önemli boşluklardan biri.

Yukarıdaki iki desenin **ikisi de kodun güvenilir olduğunu varsayıyor**:

- KFP'de launcher, kullanıcı koduyla **aynı pod'da** çalışıyor ve S3 kimlik
  bilgisi ortam değişkeninde. Kod onu okuyabilir. Veri bilimcisinin gözden
  geçirilmiş, git'teki kodu için bu makul.
- Tekton'da step'in yazdığı `uri`/`digest` JSON'una **kimse bakmıyor** —
  step ne yazarsa Chains onu imzalıyor. Step doğruyu söyler varsayımı var.

Red Hat'in güvenilmeyen kod için ürünü **Red Hat build of Agent Sandbox**
(Technology Preview) — ama o **izolasyonu** çözüyor, artifact'i değil:
verdiği kalıcılık bir **PVC**, yani disk. `artifact_id` yok, soy yok, TTL yok,
içerik hash'i yok. `§9.5.10`'daki bulgunun aynısı: *disk = sadece bucket.*

**Sonuç:** OpenShift'te bizim sorumuzun tam bir birinci-parti cevabı yok.
Üç parçayı birleştirmek gerekiyor:

| Parça | Nereden | Bizde |
|---|---|---|
| Veri modeli + taşıma deseni | **KFP launcher** | Alındı (§11.3) |
| İzolasyon + warm pool | **Agent Sandbox** (TP) | **Alınmadı** — düz container |
| Güvenilmeyen koda kimlik vermeme | **Hiçbiri** | Kendi çözümümüz: kapsam jetonu |

Üçüncü satır, mimarinin gerçekten özgün olduğu tek yer — ve özgün olmak
burada bir övünme değil, **test edilmemiş** demek.

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

Bu bulgunun tablo hâli §9.5.10'da — orada "kayıt defteri" bir sütun, ve
hangi satırların boş kaldığı tek bakışta görünüyor.

---

# §9.5 — Artifact storage: baytlar **fiilen** nerede duruyor

Buraya kadarki bölümler **erişim mekanizmasını** anlattı: mount mu, SDK mi,
API mi. Bu bölüm bir adım altını soruyor — *dosya kapatıldıktan sonra baytlar
hangi ürünün içinde yatıyor, ne kadar duruyor, kim sahibi.*

İki soru bilerek ayrı tutuluyor, çünkü cevapları bağımsız: Anthropic ile
Microsoft'un ikisinde de "mount yok, API var" ama biri baytı 30 gün saklıyor,
diğeri oturum bitince siliyor.

> **Okuma uyarısı.** Aşağıda "belgelenmemiş" yazan yerler gerçekten
> belgelenmemiş demek — tahmin yazmadım. Ticari sağlayıcıların çoğu
> *arayüzünü* belgeliyor, *arka ucunu* değil; ve bir SaaS için bu makul bir
> tercih. Bizim açımızdan önemli olan da zaten arayüzün verdiği garanti.

---

## §9.5.1 — Anthropic: container diski + Files API

| | |
|---|---|
| **Depo ürünü** | Belgelenmemiş (Files API'nin arka ucu açıklanmıyor) |
| **Sandbox'taki yol** | Container'ın kendi diski — **5 GiB** çalışma alanı |
| **Dışarı çıkış** | `$OUTPUT_DIR`'ın üst düzeyi yakalanır → `file_id` → Files API |
| **Saklama** | Container **30 gün**; ~5 dk hareketsizlikten sonra checkpoint'lenir, 30 gün içinde kimliğiyle geri çağrılabilir |
| **Kayıt defteri** | **Var** — `file_id`, Files API üzerinden |

Yakalama kuralı bizim süpürmemizin birebir emsali:

> *"Each `bash_code_execution` call gets a new, empty directory, available to
> the command as `$OUTPUT_DIR`. When the command finishes, the files at the top
> level of that directory are captured and returned as the `file_id` entries in
> the result's `content` list. **Files written anywhere else stay in the
> container and aren't returned.**"*

"Yalnızca üst düzey" kısıtı da aynı — biz de alt dizinlere inmiyoruz.

Bir ilginç ekstra: ürettikleri görsel/video/ses dosyalarına **C2PA Content
Credentials** gömüyorlar — imzalı bir manifest, "bu dosyayı Claude üretti"
diyor. Kayıt defterini dosyanın *içine* koymanın bir biçimi; bizde karşılığı
yok, kayıt tamamen dışarıda.

---

## §9.5.2 — OpenAI: container diski, kalıcı depo **yok**

| | |
|---|---|
| **Depo ürünü** | Belgelenmemiş |
| **Sandbox'taki yol** | Container'ın kendi dosya sistemi |
| **Dışarı çıkış** | "get container file content" uç noktası |
| **Saklama** | Container **20 dakika kullanılmazsa** ölür, geri getirilemez |
| **Kayıt defteri** | Container dosya listesi var; kalıcı bir artifact registry değil |

Dokümanın kendisi kalıcılık iddiasını açıkça reddediyor:

> *"We highly recommend you treat containers as ephemeral and **store all data
> related to the use of this tool on your own systems**."*
>
> *"You can't move a container from an expired state to an active one."*

Bu, dokümandaki en dürüst cümlelerden biri ve bizim tezimizi doğruluyor:
**sandbox'ın diski bir depo değildir.** Biz de tam olarak bunu söyleyip ayrı
bir Artifact Service koyduk.

---

## §9.5.3 — Cloudflare: R2 — ama ürüne kilitli değil

| | |
|---|---|
| **Depo ürünü** | **R2** (yerel seçenek), ayrıca **S3 ve GCS** |
| **Mekanizma** | `sandbox.mountBucket()` / `unmountBucket()` — s3fs |
| **Saklama** | Bucket'ın kendi ömrü — sandbox'tan bağımsız |
| **Kayıt defteri** | **Yok** — mount edilen bucket'ta sadece anahtar var |

> *"Mount S3-compatible storage buckets (R2, S3, GCS) into the sandbox
> filesystem for persistent data access."*

Dikkat çeken nokta: kendi depo ürünleri (R2) olmasına rağmen **S3-uyumlu her
şeyi** kabul ediyorlar. Yani "depo bizim" değil, "protokol standart" diyorlar.
Bizim MinIO ↔ ODF ↔ AWS S3 arasında taşınabilir olmamızla aynı duruş.

---

## §9.5.4 — Google: ADK'nın **takılabilir** artifact servisi

Google, artifact deposunu bir ürün değil bir **arayüz** olarak tasarlamış —
`ArtifactService`, ve üç uygulaması var:

| Uygulama | Baytlar nerede | Ne için |
|---|---|---|
| `InMemoryArtifactService` | Süreç belleğinde (Python'da düz bir `dict`) | Geliştirme/test |
| `GcsArtifactService` | **Google Cloud Storage** | Üretim |
| `FileArtifactService` | Yerel dosya sistemi (taban dizin argümanı) | Kotlin; yerel kalıcılık |

Bu bizim `ObjectStore` soyutlamamızın birebir karşılığı ve doğrulaması: depo
ürünü **yapılandırma**, mimari değil.

Vertex AI Agent Engine tarafında oturum/hafıza saklanıyor ama **arka uç deposu
belgelenmemiş**; ayrı bir "artifact storage" ürünü olarak da sunulmuyor.

---

## §9.5.5 — AWS: **kendi dosya sistemini getir**

§6'da anlatılan iki yol (inline 100 MB / terminalden S3'e 5 GB) hâlâ geçerli,
ama üzerine **üçüncü bir yol** eklenmiş: oturuma doğrudan dosya sistemi mount
etmek.

| | |
|---|---|
| **Depo ürünü** | **Amazon S3 Files** (S3 bucket'ı destekli) veya **Amazon EFS** — *senin* AWS hesabında |
| **Mekanizma** | `filesystemConfigurations` → access point `/mnt/<ad>`'e mount edilir |
| **Taşıma** | S3 Files: **NFSv4.2 over TLS + IAM auth**, port 2049. EFS: NFSv4.1 over TLS |
| **Senkron** | *"Files at your mount path (for example, `/mnt/s3data`) **sync bidirectionally** with the backing S3 bucket."* |
| **Kayıt defteri** | Yok |

Üç şey dikkat çekici:

**1. Ayrıcalık problemini platform çözüyor.** §9'daki "FUSE ayrıcalık
problemi" tablosunda üçüncü bir satır açıyorlar — mount ne sandbox'ın içinde
ne ayrı bir sidecar'da, **platformun kendisinde**:

> *"You don't need custom mount code, privileged containers, or download
> orchestration — AgentCore performs all mount operations inside the session
> sandbox."*

**2. FUSE değil, NFS.** Herkes s3fs/mountpoint ile FUSE yaparken AWS,
S3'ün önüne NFS konuşan bir servis koymuş. Ayrıcalık ihtiyacı böyle kayboluyor.

**3. Yönetilen bir depo YOK — ve bunu açıkça yazıyorlar:**

> *"Unlike AgentCore Runtime, AgentCore Code Interpreter **does not offer a
> managed session-storage option**. Code Interpreter supports bring-your-own
> Amazon S3 Files and Amazon EFS access points only."*

Yani AWS bile, kod çalıştıran sandbox'a "kutudan çıkan bir artifact deposu"
vermiyor. Kalıcılığı isteyen kendi bucket'ını getiriyor.

**Sınırlar:** oturum başına en fazla 4 S3 Files + 4 EFS (toplam 8) mount;
mount yolu `/mnt/` altında tek seviye olmak zorunda (`/mnt/s3data` olur,
`/mnt/a/b` olmaz); VPC zorunlu; tek bir mount hatası oturumu hiç başlatmıyor.

---

## §9.5.6 — Microsoft: `/mnt/data`, ve oturumla birlikte ölüyor

| | |
|---|---|
| **Depo ürünü** | **Yok** — oturumun kendi dosya sistemi (Hyper-V sınırı içinde) |
| **Sandbox'taki yol** | `/mnt/data` |
| **Giriş/çıkış** | Session pool yönetim API'si: `files` uç noktası (upload / download / list / metadata) |
| **Yükleme sınırı** | **128 MB** — aşılırsa HTTP 413 |
| **Saklama** | Oturum yok edilince kaynaklar temizlenir; kalıcı depo yok |

> *"Uploaded files are stored in the session's file system in the `/mnt/data`
> directory."*

`files` uç noktası künye döndürüyor (`filename`, `size`, `lastModifiedTime`)
ama bu bir **dizin listesi**, kayıt defteri değil: `artifact_id` yok, soy yok,
içerik hash'i yok, oturumdan bağımsız bir ömrü yok.

OpenAI'ın `/mnt/data`'sıyla aynı yolu seçmiş olmaları tesadüf değil — Code
Interpreter'ın alışkanlığı sektörde bir kural hâline gelmiş.

---

## §9.5.7 — Red Hat / Kubeflow: depo **yapılandırma**, kayıt defteri **MLMD**

Burası bizim en çok hizalandığımız yer ve en açık belgelenen taraf: baytlar
nesne deposunda, künyeler ayrı bir metadata veritabanında.

| KFP launcher'ın desteklediği depo | Kimlik nasıl veriliyor |
|---|---|
| **AWS S3** | Statik kimlik (Secret) **veya IRSA** (`fromEnv: true`) |
| **S3-uyumlu** (MinIO dâhil) | Statik kimlik |
| **Google Cloud Storage** | Statik kimlik **veya** App Credentials |
| **SeaweedFS** | Statik kimlik; ayrıca AWS/GCP/Azure/MinIO S3'e gateway |

Nereye yazılacağını `pipeline_root` söylüyor — bizim `PTC_ARTIFACT_ROOT`'umuzun
kaynağı. Kayıt defteri **MLMD**; `system.Dataset` / `system.Model` /
`system.Metrics` tip sözlüğünü oradan aldık.

**IRSA satırı özellikle önemli:** pod'a uzun ömürlü anahtar koymadan, pod'un
service account'una bağlı bir rol veriliyor. §10.3'teki "SDK, anahtar pod'da"
satırının en olgun hâli bu — ve bizim kapsam jetonumuzun bulut karşılığı.

---

## §9.5.8 — Uzman sağlayıcılar: depo *senin*, ya da hiç yok

| Sağlayıcı | Depo ürünü | Not |
|---|---|---|
| **E2B** | Kendi bucket'ın (s3fs/gcsfuse) + **snapshot** | Snapshot dosya sistemi **ve belleği** birlikte donduruyor; duraklatılmış sandbox'lar öldürülene kadar süresiz duruyor |
| **Modal — Volumes** | Kendi dağıtık dosya sistemleri | *"backed by multiple underlying cloud providers to guarantee high availability"* — hangileri belirtilmiyor. v2'de dosya < 1 TiB, ~2,5 GB/s tavan |
| **Modal — CloudBucketMount** | **AWS S3, Cloudflare R2, GCS** | AWS Mountpoint üzerine kurulu, ve onun kısıtlarını devralıyor: **append yok**, `seek`+write yok, yazmak için `truncate` şart |
| **Daytona** | Kendi bucket'ın (mount-s3/gcsfuse/rclone) ya da platform Volume'ü | — |
| **Vercel Sandbox** | Kendi bucket'ın | Düz kurulumda anahtar sandbox'ta |
| **Fly.io Sprites** | **S3-uyumlu nesne deposu**, JuiceFS benzeri katman | 100 GB kalıcı kök dosya sistemi |

Fly.io'nun tasarımı ayrıca anlatmaya değer, çünkü "nesne deposunu disk gibi
göstermenin" en ileri örneği:

> *"the root of storage is S3-compatible object storage"*
>
> *"It works by splitting storage into data ('chunks') and metadata (a map of
> where the 'chunks' are). Data chunks live on object stores; metadata lives in
> fast local storage."*

Metadata SQLite'ta ve **Litestream** ile dayanıklı tutuluyor; üstünde NVMe'de
`dm-cache` benzeri bir önbellek var. Yani: baytlar S3'te, harita yerelde,
okuma önbellekli. Sprite uyuyabiliyor, makineler arasında taşınabiliyor ve
dosya sistemi bozulmadan geri geliyor.

**Ama** — bu bir *dosya sistemi* çözümü, artifact deposu değil. Chunk haritası
"hangi bayt nerede" der; "bu dosya neyden türedi, tipi ne, ne zaman
silinmeli" demez.

---

## §9.5.9 — Biz

| | |
|---|---|
| **Depo ürünü** | **MinIO** (yerel/kind) · **ODF** (ObjectBucketClaim) ya da harici S3 (OpenShift AI bağlantısı) — üçü de aynı kod yolundan |
| **Kök** | `PTC_ARTIFACT_ROOT` (KFP'nin `pipeline_root`'u) |
| **Sandbox'taki yol** | `/output` — üst düzeyi süpürülür |
| **Kayıt defteri** | **SQLite** — `artifact_id`, tip, boyut, içerik hash'i, soy, TTL |
| **Saklama** | TTL + saat başı çalışan reaper CronJob |
| **Anahtar** | Sandbox'ta **yok**; çalıştırma başına HMAC imzalı kapsam jetonu |

---

## §9.5.10 — Tek tabloda hepsi

| | Depo ürünü | Sandbox'taki yol | Kayıt defteri | Baytların ömrü |
|---|---|---|---|---|
| **Anthropic** | Belgelenmemiş | container diski (5 GiB) | `file_id` (Files API) | 30 gün |
| **OpenAI** | Belgelenmemiş | container diski | yok (dizin listesi) | 20 dk hareketsizlik |
| **Cloudflare** | **R2** / S3 / GCS | mount noktası | yok | bucket'ın ömrü |
| **Google (ADK)** | **GCS** / bellek / yerel disk | ArtifactService API | ADK artifact adı+sürüm | deponun ömrü |
| **AWS** | **S3 Files** / **EFS** (senin hesabın) | `/mnt/<ad>` | yok (CloudTrail denetim) | bucket/EFS'in ömrü |
| **Microsoft** | **yok** | `/mnt/data` | yok | oturumla ölür |
| **Red Hat / KFP** | S3 / GCS / MinIO / SeaweedFS | `.path` (launcher kopyalar) | **MLMD** | `pipeline_root`'un ömrü |
| **E2B / Daytona / Vercel** | senin bucket'ın | mount noktası | yok | bucket'ın ömrü |
| **Modal** | kendi FS'i / S3-R2-GCS | Volume ya da mount | yok | Volume'ün ömrü |
| **Fly.io** | S3-uyumlu (JuiceFS benzeri) | kök dosya sistemi (100 GB) | yok (chunk haritası) | Sprite'ın ömrü |
| **Biz** | **MinIO / ODF / harici S3** | `/output` | **SQLite** (id, tip, hash, soy, TTL) | TTL + reaper |

### Bu tablodan çıkan üç şey

**1. "Artifact storage" diye ayrı bir ürün kimsede yok.** Herkes ya bir nesne
deposunu (S3/GCS/R2) ya da container diskini kullanıyor. Fark, üstüne ne
konduğunda.

**2. Kayıt defteri sütunu üç yerde dolu:** Anthropic, Google ADK, Red Hat/KFP —
ve biz. Üçünün de ortak özelliği, **yazma yolunun bir bileşenden geçmesi.**
Mount edenlerin hiçbirinde kayıt defteri yok; §9'daki "en keskin ortak
bulgu" bu tabloda sütun olarak görünüyor.

**3. Ömür sütunu bizde tek başına.** Diğerlerinde ya sabit bir süre var
(30 gün, 20 dk) ya da "bucket'ın ömrü" (yani sonsuz, kimse temizlemiyor).
Artifact **başına** TTL taşıyan tek yer biz — ve bu, kayıt defterini tutmanın
doğal sonucu: neyin ne zaman öleceğini ancak künyesini tutuyorsan bilirsin.

---

# §9.6 — Baytı kim taşıyor, ve **aracı nerede duruyor**

`§9.5` "baytlar nerede duruyor" diye sordu. Bu bölüm bir adım öncesini soruyor:
**kod dosyayı yazdıktan sonra onu oraya kim götürüyor, ve o götüren şey kodun
neresinde duruyor?**

İki soru bilerek ayrı, çünkü cevapları bağımsız. KFP ile Anthropic'in ikisinde
de "aracı var" denir — ama biri kullanıcı koduyla **aynı container'da**, diğeri
kodun **hiç erişemeyeceği yerde**. Güvenlik açısından aradaki fark, olan ile
olmayan kadar büyük.

---

## §9.6.1 — Üç aile, süreç modeliyle

```
A) MOUNT — taşıyan yok, çekirdek yapıyor
   [ kullanıcı kodu ] --write()--> FUSE/NFS --> bucket
                                    ^^^^^^^^ araya girecek yer YOK

B) SARMALAYICI — taşıyan var, ama kodun YANINDA
   [ launcher + kullanıcı kodu ] --depo anahtarı--> bucket
     ^^^^^^^^^^^^^^^^^^^^^^^^^^ aynı container, aynı ortam

C) SINIR — taşıyan var, kodun ERİŞEMEYECEĞİ yerde
   [ kullanıcı kodu ] --> dizin --> (( platform / servis )) --> depo
                                    ^^^^^^^^^^^^^^^^^^^^^^ ayrı güven alanı
```

Fark **kimin ne yaptığı** değil, **nerede durduğu**. B'de de bir program
kopyalama yapıyor, C'de de. Ama B'deki program, denetlemesi beklenen kodun
kardeşi.

---

## §9.6.2 — B ailesi: KFP / OpenShift AI

Bir pipeline adımı = bir pod, içinde iki program:

```
Pod
├── init container:  kfp-driver     → girdileri çözer, cache'e bakar,
│                                     PodSpec'i yamalar
└── main container:  kfp-launcher   ← PID 1
       ├─ girdileri  .uri  →  .path  indirir
       ├─ KULLANICI KODUNU alt süreç olarak çalıştırır
       └─ çıktıları  .path →  .uri   yükler + MLMD'ye yazar
```

Birincil kaynak:
> *"V2 Driver (kfp-driver), an **init container** that resolves task inputs…
> V2 Launcher (kfp-launcher), the **main container wrapper** that manages
> artifact I/O, **invokes user code**, and publishes execution metadata."*

### Buradan çıkan sonuç

Kullanıcının kodu, launcher'ın **aynı container'daki alt süreci.** Aynı ortam
değişkenleri, aynı dosya sistemi, aynı ağ namespace'i. Launcher depoya ulaşmak
için hangi kimliği taşıyorsa, kullanıcı kodu onu **miras alır.**

> Not: KFP dokümanı kimlik bilgisinin nasıl sağlandığını ayrıntılamıyor
> (`gocloud.dev/blob` üzerinden konuştuğunu söylüyor, o kadar). Yukarıdaki
> "miras alır" cümlesi dokümandan bir alıntı değil, **süreç modelinin
> sonucu** — aynı container'da çalışan bir alt süreç ortamı devralır.
> Object Store yapılandırma kılavuzu kimlikleri Secret'tan env'e ya da IRSA
> ile veriyor; ikisi de pod düzeyinde.

Yani launcher bir **sarmalayıcı**, bir **sınır** değil:

```python
def benim_bilesenim(cikti: Output[Dataset]):
    df.to_parquet(cikti.path)                  # launcher'ın beklediği yol
    boto3.client("s3").delete_object(...)      # ...ama bu da mümkün
```

Denetim koyacak bir yer yok, çünkü kod launcher'ı atlayıp doğrudan depoya
konuşabilir. `§10.3`'te "SDK, anahtar pod'da" satırının anlamı bu.

**Ve KFP için bu doğru karardır.** Oradaki kodu bir insan yazdı, gözden geçirdi,
git'te duruyor. Korunacak bir şey yok; sarmalayıcının işi güvenlik değil
**kolaylık** — bileşen yazarının S3 kodu yazmaktan kurtulması.

---

## §9.6.3 — C ailesi: Anthropic (ve OpenAI, Microsoft)

Burada kodun yaptığı tek şey dosya yazmak. Hiçbir çağrı yok:

```
1. Model bash çalıştırır       → her çağrıya YENİ, boş bir $OUTPUT_DIR
2. Kod dosya yazar             → cp rapor.pdf "$OUTPUT_DIR/"
3. Komut biter                 → platform o dizinin ÜST DÜZEYİNİ yakalar
4. Sonuçta file_id döner       → Files API ile indirilir
```

> *"When the command finishes, the files at the top level of that directory
> **are captured and returned** as the `file_id` entries in the result's
> `content` list. **Files written anywhere else stay in the container and
> aren't returned.**"*

"are captured" — edilgen. Yakalayan **platform**, kod değil. Model bir fonksiyon
çağırmıyor; çağırabileceği bir fonksiyon da yok.

### "Belgelenmemiş" tam olarak neyi kastediyor

`§9.5`'teki kimlik sütununu. Anthropic şunları belgeliyor:

| Belgeli | Değer |
|---|---|
| Çalışma alanı diski | 5 GiB |
| Container ömrü | 30 gün; ~5 dk hareketsizlikte checkpoint |
| Yakalama kuralı | yalnızca `$OUTPUT_DIR` **üst düzeyi** |
| Erişim | `file_id` → Files API |
| Ekstra | üretilen medyaya **C2PA** imzası gömülüyor |

Belgelemediği: baytların fiilen hangi üründe durduğu, ve sandbox'ın herhangi
bir depo kimliği taşıyıp taşımadığı.

Bir SaaS için makul bir tercih — arayüz garantisi veriliyor, arka uç
verilmiyor. Tüketici açısından da fark etmiyor: **model hiçbir depo adı
görmüyor**, dolayısıyla arka ucun ne olduğu onun için gözlemlenemez.

OpenAI ve Microsoft aynı ailede, ama kalıcılık dereceleri farklı: OpenAI
container'ı 20 dk hareketsizlikte ölüyor ve dokümanı *"treat containers as
ephemeral and store all data … on your own systems"* diyor; Microsoft'ta
oturum bitince dosya da bitiyor — orada C ailesinin "sınır"ı var ama
arkasında **kalıcı depo yok.**

---

## §9.6.4 — A ailesi: mount edenler

Taşıyan bir program yok. `write()` çağrısı çekirdeğe gidiyor, FUSE ya da NFS
istemcisi onu bucket'a çeviriyor.

| Sağlayıcı | Mekanizma | Anahtar nerede |
|---|---|---|
| E2B, Daytona (external), Vercel (düz) | sandbox içinde `sudo s3fs` / `mount-s3` | **sandbox'ın içinde** |
| Modal `CloudBucketMount` | AWS Mountpoint | Secret ya da OIDC |
| AWS AgentCore | platform mount ediyor, **NFSv4 over TLS** | pod'un IAM rolü |
| Fly.io Sprites | JuiceFS benzeri; chunk'lar S3'te, harita SQLite'ta | platformda |

Araya girecek yer yok — ve zaten amaç bu. `§9`'un bulgusu buradan geliyor:
**mount eden hiçbir sağlayıcıda kayıt defteri yok.** `write()` bir kernel
çağrısı; ona `artifact_id` üretecek, tipini çıkaracak, TTL verecek bir kanca
takamazsınız.

### Melez: imzalayan proxy

Cloudflare (binding/proxy) ve Vercel (proxy) bir ara yol buluyor:
`mount-s3 --no-sign-request` ile mount ediliyor, imzayı **dışarıdaki bir
Worker/Function** atıyor. Mount'un rahatlığı duruyor, anahtar sandbox'tan
çıkıyor.

Ama denetim yine yok: proxy imzalıyor, **içeriğe bakmıyor**. Kayıt defteri de
doğmuyor.

---

## §9.6.5 — Biz: yapı B'den, sınır C'den

```
KFP:        [ launcher + kullanıcı kodu ]  ──S3 anahtarı──>  bucket
              ^^^^^^^^^^^^^^^^^^^^^^^^^ aynı container

Anthropic:  [ kullanıcı kodu ]  →  $OUTPUT_DIR  →  (( platform ))  →  depo
                                                    ^^^^^^^^^^^^ erişilemez

Biz:        [ entrypoint + LLM kodu ] ──kapsam jetonu──> [[ Artifact Service ]] ──S3──> bucket
              ^^^^^^^^^^^^^^^^^^^^^^ aynı container       ^^^^^^^^^^^^^^^^^^^^ AYRI POD
```

**2026-09-06'da bu şema değişti (§11.12).** Aktarım da sidecar'a taşındı —
yani artık her iki katmanda da C ailesindeyiz:

```
Biz (bugün):  [ LLM kodu ]  →  /output  →  (( sidecar ))  →  [[ Artifact Service ]]  →  bucket
                                            ^^^^^^^^^^^ ayrı container, jeton onda
```

Sidecar Argo'nun `wait` container'ının karşılığı; sandbox'ta ne depo anahtarı
ne kapsam jetonu var, ve proxy'de yazma uç noktası yok.

Böylece beş kontrol (kapsam · isim/yol · depo kökü · boyut · pickle) LLM'in kodunun
**erişemeyeceği** bir güven alanında çalışıyor — Anthropic'in platformunun
durduğu yerde.

### Dürüst incelik: jeton koda karşı sır DEĞİL

LLM'in kodu `os.environ["PTC_SCOPE_TOKEN"]`'i okuyabilir ve servise doğrudan
POST atabilir. Bu bilinen ve kabul edilmiş bir şey; kabul testimiz tam olarak
bunu yapıyor (depo kökü yol geçişi denemesi, `scripts/kabul_testi.py` §5).

Fark şurada:

| | S3 anahtarı | Kapsam jetonu |
|---|---|---|
| Neye yeter | **bucket'ın tamamı** | yalnızca servisin izin verdiği |
| Kapsam | bucket / prefix | tek workflow (imzalı) |
| Ömür | rotasyona kadar | **15 dakika** |
| Denetim | yok — S3 içeriğe bakmaz | beş kontrol, her yazmada |
| Üretilebilir mi | — | hayır, imza sırrı sandbox'ta yok |

Yani jetonu ele geçirmek "servisi kullanabilmek" demek; anahtarı ele geçirmek
"bucket'ın sahibi olmak" demek.

---

## §9.6.6 — Karar tablosu

Aracıyı nereye koyacağınız, sandbox'taki **kodu kimin yazdığına** bağlı:

| Kodu kim yazdı | Doğru aile | Neden |
|---|---|---|
| Geliştirici, review'dan geçmiş, git'te | **A ya da B** | Korunacak bir şey yok; mount ve sarmalayıcı en rahat yol |
| Son kullanıcı (SaaS'ta keyfi kod) | **C** | Kod düşman varsayılır; sınır kodun dışında olmalı |
| **LLM** | **C** | Aynısı, ve bir fazlası: kod her çalıştırmada değişiyor, denetlenecek sabit bir yüzey yok |

`§8.7`'nin sonucu buradan da görünüyor: OpenShift'in iki yerleşik deseni de
(KFP → B, Tekton → workspace/PVC) **A ve B ailesinde.** Güvenilmeyen kod için
C ailesinden birinci-parti bir cevap yok — Agent Sandbox izolasyonu çözüyor,
artifact'i değil. Bizim aracıyı yazmamızın sebebi bu.

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

Bu tablonun "neden" tarafı §9.6'da: taşıyanın **nerede durduğu** anahtarın
nerede olacağını belirliyor.

| Yaklaşım | Kim | Anahtar sandbox'ta | §9.6 ailesi |
|---|---|---|
| **Mount, anahtar içeride** | E2B, Daytona (external), Vercel (düz) | **Evet** | A |
| **Mount, anahtar dışarıda** | Cloudflare (binding/proxy), Vercel (proxy), Daytona (Volumes) | Hayır | A (melez) |
| **Blok cihaz** | Fly.io | Yok | A |
| **SDK, anahtar pod'da** | Red Hat OpenShift AI, AWS (IAM role) | Evet (dar) | **B — sarmalayıcı** |
| **API, anahtar hiç yok** | Anthropic, OpenAI, **biz** | Hayır | **C — sınır** |

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
| 1 | **Sadece tool tarifi** | Tool listesinde fonksiyon durur; model çağırmayı *seçmek* zorunda | *(2026-09-06'ya kadar biz)* |
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
| 1 — Tool tarifi | **✗ kaldırıldı** — çağrılacak bir fonksiyon yok (§11.11) |
| 2 — Referans otomatik context'te | ✗ — `file_id` gibi bir kavramımız yok |
| 3 — Dosya sistemi `ls` | **✓ tam** — `os.listdir` / `os.path.exists` / `glob` manifestle birleştirilmiş; `/output` bu koşu, `/artifacts/<wf>` başkaları |
| 4 — İsimler prompt'a enjekte | **✓ tam** — ADK deseni, iki grup hâlinde (bu oturum / başka çalıştırmalar) |
| 5 — Semantik arama | ✗ (N onlarca mertebesindeyken gerekmiyor) |

**En olgun iki desen (3 ve 4) birlikte var** — ve bu bilinçli: ikisi birbirinin
yedeği. Model promptu atlarsa dosya sistemi aynı gerçeği söylüyor; manifest
sessizce devre dışı kalırsa (`ARTIFACT_SERVICE_URL` tanımsız) `os.listdir`
yine çalışıyor.

Desen 1'i **kaldırdık**, çünkü zayıf olmasının ötesinde zararlıydı: bu haftanın
ciddi hatalarının hepsi o API yüzeyinde çıktı. Desen 2 de yok — ama gerekmiyor,
çünkü isim zaten promptta ve dosya sisteminde.

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
│  /sandbox    configMap → LLM'in kodu (salt okunur)        │
│  /scratch    emptyDir  → geçici (ölmesi İSTENİR)          │
│  /output     emptyDir  → BU koşu: süpürülür + tembel iner │
│  /artifacts  emptyDir  → BAŞKA koşular: <wf>/<ad>, salt   │
│                          okuma, süpürülmez                │
│  kalıcı disk YOK · S3 anahtarı YOK · DNS YOK · API YOK    │
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

Pod açılır açılmaz Artifact Service'e **bir** istek gider: *"bu tenant'ta
hangi isimler var?"* Sadece isimler döner, bayt dönmez.

Gelen liste **ikiye** ayrılıyor:

| Nereden | Nereye düşüyor |
|---|---|
| Bu çalıştırmanın kendi çıktıları | `/output/<ad>` |
| Başka çalıştırmaların çıktıları | `/artifacts/<workflow_id>/<ad>` |

**Neden sadece isimler:** önceki tasarımda pod doğarken depodaki **her**
artifact indiriliyordu. Maliyet var olan her şeyle ölçekleniyordu — 6 tane
100 MiB'lik artifact biriktiğinde, hiçbirine dokunmayan bir script bile
512Mi'lık `/output`'u patlatıyordu. Şimdi maliyet **kullanılan kadar**.

### Adım 6 — Kod çalışır

`main()` şu sırayla ilerliyor:

```python
istemci  = ArtifactClient(...) if SCOPE_TOKEN and ENDPOINT else None
launcher = _launcher_api(istemci, okunanlar) if istemci else {}
#          ^^^^^^^^^^^^ LLM'e GÖRÜNMEZ; içinde yalnızca `_put_file` var

inenler = {}
if istemci:
    tembel_globals = _tembel_okumayi_kur(istemci, _manifest(istemci),
                                         inenler, okunanlar)

sandbox_globals = {"set_result": set_result, **tool'lar, **tembel_globals}
#                  ^ artifact fonksiyonu YOK — 2026-09-06'da kaldırıldı (§11.11)

try:
    exec(kod, sandbox_globals)
except Exception as exc:
    _ciktilari_supur(launcher, inenler)   # ← hata olsa BİLE önce süpür
    print(hata); exit(0)

_ciktilari_supur(launcher, inenler)       # ← başarıda da
print(sonuc)
```

Dikkat: süpürme **hata yolunda da** çalışıyor. Script son satırda patlasa bile
o ana kadar üretilen dosyalar kurtarılıyor — asıl değeri de burada.

### Adım 7 — Artifact Service kontrolleri akış sırasında yapar

Baytlar akarken:

| Kontrol | Nasıl | Neden |
|---|---|---|
| **Kapsam** | `workflow_id` **jetondan** okunur, çağıranın iddiasından değil | Paylaşılan depo, ağ politikasının **göremediği** bir kanal açar |
| **Format** | pickle — hem etiketten hem **ilk iki bayttan** | Deserialization kod çalıştırır (CWE-502); baytları LLM yazdı |
| **Boyut** | Sayarak; sınır aşılınca okuma **orada** kesilir | Veriyi sandbox içinde süzmeye zorlar |
| **İsim** | `../`, `/`, boşluk reddedilir | Yol geçişi sınıfı |
| **Depo kökü** | `X-Artifact-Root` doğrulanır; `..` ve mutlak yol reddedilir | 2026-09-06'ya kadar doğrulanmıyordu — sandbox ham POST ile kökü kendisi seçebiliyordu |

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
LLM `pd.read_parquet("/output/satislar.parquet")` yazınca dosya arka planda
iner. Başka bir çalıştırmanın çıktısı gerekiyorsa yol kimliğini taşır:
`/artifacts/<workflow_id>/<ad>`.

**40 saniyelik iş tekrarlanmaz.**

## §11.3 — Yazma yolu: **tek** tetikleyici

LLM'in bilmesi gereken hiçbir şey yok — `/output` altına dosya bırakıyor,
çalışma bitince süpürme onu depoya kopyalıyor:

```
/output/satislar.parquet ──HTTP akış──> Artifact Service ──S3 PUT──> bucket
```

| | Süpürme |
|---|---|
| Nasıl tetiklenir | `df.to_parquet("/output/satislar.parquet")` — düz Python |
| LLM bilmek zorunda mı | **Hayır** |
| Ne zaman | Çalışma sonunda, başarı **ya da hata** fark etmeksizin |
| Tip nereden | Dosya uzantısından (`.parquet` → `system.Dataset`) |
| Denetim | Dört kontrol, Artifact Service'te |

Bu, KFP launcher'ının yükleme adımının aynısı — orada da bileşen `.path`'e
yazar, launcher `.uri`'ye kopyalar. Ve Anthropic'in `$OUTPUT_DIR`'ı, OpenAI'ın
`/mnt/data`'sıyla aynı konvansiyon.

> Eskiden ikinci bir yol vardı: LLM'in doğrudan çağırdığı `put_artifact`.
> 2026-09-06'da kaldırıldı — gerekçe ve bunu bulduran hatalar §11.11'de.

### Dizin çıktıları (2026-09-06)

Süpürme yalnızca üst düzey **dosyaları** alıyordu; `/output/model.v1/` gibi bir
DİZİN sessizce atlanıyordu. Canlı denendi: LLM üç dosyalık bir model dizini
yazdı, hiçbiri saklanmadı, **hiçbir yerde hata çıkmadı**.

KFP launcher'ı bu ayrımı yapıyor — *"the launcher determines artifact type
(file vs directory), then uploads from local path to object storage URI"* —
bizde eksikti. Artık dizinler de alınıyor, ama **tek bir tar** olarak:

| | KFP | Biz |
|---|---|---|
| Dizin nasıl saklanıyor | Nesne deposuna **özyinelemeli** (1 artifact = N nesne) | **Tek tar** (`model.v1` → `model.v1.tar`) |
| Sebebi | Depoda gezilebilir kalsın | Künyenin dört değişmezi: `content_hash`, dedup, `size_bytes`, akışlı tek-nesne put/get |
| Bedeli | — | Dizinden TEK dosya ayrı çekilemiyor |

Tar **tekrarlanabilir** üretiliyor (mtime/uid/gid sıfır, girdiler sıralı);
aksi hâlde aynı içerik iki kez paketlenince farklı baytlar çıkar ve
içerik-hash dedup'ı sessizce ölürdü.

Okuma tarafı simetrik: `/output/model.v1/weights.json` istendiğinde dizin
yoksa `model.v1.tar` inip açılıyor. Açarken `filter="data"` uygulanıyor —
arşiv bizim ürettiğimiz olsa da depoya süpürme yoluyla başka bir tar
girmiş olabilir (CVE-2007-4559).

## §11.4 — Okuma yolu: üç keşif yolu

5. adım 1. adımın ürettiğini üç ayrı yoldan bulabiliyor, ve hiçbiri tek dayanak
değil. **Üçü de düz Python** — çağrılacak bir API yok:

1. **Manifest promptta** — isimler model talimatlarına enjekte ediliyor, iki
   grup hâlinde: *bu oturumda üretilenler* ve *başka çalıştırmalardan*
   (ADK'nın `LoadArtifactsTool` deseni)
2. **Dosya sistemi** — `os.listdir("/output")`, `os.path.exists(...)`,
   `glob` manifestle birleştirilmiş; `/artifacts` başka çalıştırmaları listeler
3. **Konuşma hafızası** — checkpointer kalıcı olduğu için LLM kendi verdiği
   dosya adını hatırlıyor

### Nasıl çalışıyor (tembel doldurma)

`pd.read_csv` / `read_parquet` / `read_json` / `read_excel` / `read_feather`
sarmalanmış, ayrıca sandbox'ın globals'ına tembel bir `open` konmuş. Dosya
yoksa **ve** adı manifestte geçiyorsa, o an indiriliyor; sonra pandas normal
şekilde okuyor.

İki yol biçimi tanınıyor — `/output/<ad>` (bu çalıştırma, **katı**: tenant'a
düşmez) ve `/artifacts/<wf>/<ad>` (adı verilen çalıştırma). Bu ayrımın neden
zorunlu olduğu §11.11'de: düz bir isim uzayında ajan başkasının dosyasını
kendi işi sanıyordu.

`os.listdir` ve `os.path.exists` de yamalı — API kalkınca keşfin tek yolu
dosya sistemi kaldığı için `/output`'un boş görünmesi modeli doğrudan
yanıltıyordu. Launcher kendi işini yamalanmamış orijinallerle yapıyor.

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
| **S3 Veri** | `/output`'a düz dosya; launcher akışlı HTTP ile ayrı servise taşır. Mount **yok**, presigned URL **yok**, LLM'e API **yok** |
| **S4 Anahtar** | Sandbox'ta **hiç yok** — ne S3 anahtarı ne kapsam jetonu (2026-09-06, §11.12). Jeton sidecar'da, S3 anahtarı Artifact Service'te |
| **S5 Kayıt defteri** | **Var** — `artifact_id`, KFP tipleri, soy (otomatik), TTL, hash, `workflow_id` |

## §11.8 — Sayılar

| | Değer |
|---|---|
| Çalıştırma süresi | **~4,1 sn** (2026-09-06'da sidecar ile 3,14'ten çıktı, §11.12) |
| Bir turda çalıştırma sınırı | 2 |
| Kapsam jetonu ömrü | 15 dakika |
| Artifact başına boyut | 100 MiB |
| `/output`, `/scratch`, `/artifacts` | 512Mi (emptyDir, her biri) |
| Pod bellek limiti | 1 Gi (matplotlib için 256Mi'den yükseltildi) |
| Süre tavanı | 90 sn (`activeDeadlineSeconds`) |
| Servis bellek eşiği | 8 MiB (aşınca diske taşar) |
| Reaper sıklığı | Saat başı |
| Test sayısı | **179** birim/entegrasyon + **38** canlı kabul kontrolü |

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
| `ARTIFACT_SERVICE_URL` | **Host tarafı** — manifest enjeksiyonu buradan çekiyor. Tanımsızsa manifest sessizce devre dışı kalıyordu (2026-09-06'da yaşandı); artık bir kez uyarı basılıyor |
| `PTC_OUTPUT_DIR` / `PTC_ARTIFACTS_DIR` | Sandbox'taki iki kök (varsayılan `/output`, `/artifacts`) |

## §11.10 — Bilinen açıklar (saklamıyoruz)

| Konu | Durum |
|---|---|
| **İzolasyon** | Düz container. Red Hat Kata öneriyor, kurulu değil |
| **Metadata DB** | SQLite. `open_postgres()` kodda var, test edilmedi |
| **Workflow state** | `AsyncSqliteSaver` — çalışıyor, doğrulandı. Postgres yolu (`AsyncPostgresSaver`) **test edilmedi** |
| **Yüksek erişilebilirlik** | Tek replika (SQLite + RWO PVC) |
| **Auth** | Yok — jetonu üretebilen tenant'ın TÜMÜNÜ okur (2026-09-06'da kapsam workflow'dan tenant'a genişledi, §11.11) |
| **Büyük dosya** | 100 MiB servis sınırı, pod 1Gi (2026-09-05'te 256Mi'den yükseltildi). 5 GB çalışmaz |
| **Manifest promptta** | ADK deseniyle enjekte ediliyor, iki grup hâlinde (bu oturum / başka çalıştırmalar). `os.listdir("/output")` de aynı gerçeği söylüyor — iki bağımsız yol |
| **Soy ağacı** | Otomatik dolduruluyor + panelde grafik olarak okunuyor (2026-09-05). Kural: bir çalıştırmada OKUNAN artifact'ler, o çalıştırmada ÜRETİLENLERİN ebeveyni. Ajanın keşifte kullanması ("X'i üreten veriyi bul") **henüz yok** |
| **İsim çakışması** | "En yeni" kazanır; çağıranın kendi workflow'u öncelikli. Tenant genelinde aynı ad iki çalıştırmada varsa LLM farkında olmaz |
| **`user_metadata` doldurulamıyor** | Kayıt defterinde sütun ve `X-Artifact-Metadata` başlığı var, ama süpürme yolunda dolduran yok — `put_artifact` kalkınca (2026-09-06) sandbox'ın serbest anahtar-değer geçirme yolu kapandı. KFP'de `.metadata` bileşen imzasından gelir; bizde bildirim olmadığı için karşılığı yok |
| **Tip yalnızca uzantıdan** | Eskiden `put_artifact(df, ...)` nesneye bakıp `system.Metrics` çıkarabiliyordu; artık `.json` her zaman `system.Artifact`. Sayısal sözlük/metrik ayrımı kayboldu |
| **Manifest host değişkenine bağlı** | `ARTIFACT_SERVICE_URL` tanımsızsa manifest enjeksiyonu SESSİZCE devre dışıydı; 2026-09-06'da bir kez uyarı basılıyor ve `.env`'e eklendi. Ama hâlâ "çalışmazsa çalışmaz" bir bağımlılık |
| **Soy imzasız** | `parents` doğru ama kriptografik olarak korumasız — kayıt defterine yazabilen değiştirebilir. Tekton Chains'in çözdüğü problem (§8.6); bizde karşılığı yok |
| **Şeffaf okuma** | 5 pandas okuyucusu + `open` + `os.listdir`/`os.path.exists`/`glob`. `pyarrow`, `csv`, `PIL`, `matplotlib` doğrudan açarsa yakalanmıyor |
| **Warm pool** | Yok (kazancı ölçüldü: ≤1.6 sn) |
| **Gerçek OBC/ODF** | Test edilmedi (ODF kapsam dışı) |

---

## §11.11 — 2026-09-06: LLM'e sunulan artifact API'si KALDIRILDI

> **§11.3–§11.4 zaten bugünkü hâli anlatıyor.** Bu bölüm oraya *nasıl*
> gelindiğini anlatıyor — hangi hatalar bu tasarımı zorladı. Kararın kendisi
> kadar gerekçesi de sunumda soruluyor.

### Ne kalktı

Sandbox'ın globals'ında beş fonksiyon vardı:
`put_artifact`, `get_artifact`, `list_artifacts`, `artifact_metadata`, `cached`.
**Hepsi kaldırıldı.** Sandbox'a görünen tek artifact yüzeyi artık `/output`
dizini.

### Neden

İki gerekçe, ikisi de bu dokümanın kendi bulgularından:

**1. Emsali yoktu.** §12'nin tablosunda o satırlar "Kimse" diyordu. Piyasada
kod-yazan-LLM'e artifact API'si sunan bir sistem yok; KFP `.path`'e yazdırıyor,
Anthropic `$OUTPUT_DIR`'a, OpenAI `/mnt/data`'ya. Üçü de **dosya**.

**2. Hataların hepsi oradaydı.** Bir haftalık canlı kullanımda çıkan ciddi
kusurlar: pozisyonel `get_artifact` sessizce `None` dönüyordu; ad uzantı
taşımadığı için `read_csv` kriptik hata veriyordu; `os.listdir("/output")`
boş dönüyordu. Kopyaladığımız hiçbir parçadan hata çıkmadı.

### Yerine ne geçti — KFP launcher deseni, birebir

| KFP | Bizde |
|---|---|
| Bileşen `.path`'e (yerel yol) yazar | LLM `/output/<ad>`'a yazar |
| Launcher, çalışma sonrası `.uri`'ye kopyalar | Süpürme, çalışma sonrası depoya yükler |
| Launcher, çalışma öncesi `.uri`'den indirir | Tembel doldurma, okunduğunda indirir |
| Kod nesne deposunu HİÇ görmez | Aynı |

LLM'in yazdığı kod artık şu:

```python
df.to_parquet("/output/satislar.parquet")          # saklanır
df = pd.read_parquet("/output/satislar.parquet")   # geri okunur
os.listdir("/output")                              # depoda ne varsa
os.path.exists("/output/x.parquet")                # depodakini de sayar
```

`cached()`'in karşılığı düz Python:

```python
if os.path.exists("/output/tarama.parquet"):
    df = pd.read_parquet("/output/tarama.parquet")
else:
    df = pahali_tarama(); df.to_parquet("/output/tarama.parquet")
```

### Keşif artık dosya sisteminin kendisi

API gidince keşfin tek yolu `/output` kaldı — ama `/output` pod açılışında
fiziksel olarak **boş**. Bu, düz Python yazan modeli doğrudan yanıltıyordu:
`os.listdir("/output")` boş liste dönüyor, model "hiçbir şey yok" sonucuna
varıp veriyi yeniden üretiyordu.

Bu yüzden `os.listdir`, `os.path.exists` ve `glob` manifestle **birleştirildi**.
Dosya sistemi artık deponun görünümü: listede görünen bir ad okunmak
istendiğinde iniyor.

> Launcher'ın kendisi bu yamadan etkilenmiyor — `os.listdir`/`os.path.exists`
> orijinalleri import anında saklanıyor. Aksi hâlde süpürme, henüz inmemiş
> bir ismi "üretilmiş" sanardı.

### İKİ KÖK — ve bunu bulduran gerçek arıza

İlk hâlinde `/output` bütün tenant'ın çıktılarını düz bir liste olarak
gösteriyordu. Canlı denemede şu oldu:

```
1. tur  ajan analiz üretti          -> İK ortalaması 34,46
2. tur  "az önce ürettiğin analizde İK kaçtı?"
        ajan /output'ta gördüğü BAŞKA bir run'ın
        `departman.ozet.parquet`'ini okudu   -> "7,46" dedi
```

Cevap **sessizce yanlıştı** — hiçbir yerde hata yoktu, ajan başkasının
dosyasını kendi işi sandı.

Sebep yapısaldı: düz bir isim uzayında "benim" ile "başkasının" ayırt
edilemez. KFP'de bu sorun yok çünkü her çalıştırma
`pipeline_root/<run-id>/...` altına yazar; başka bir run'ın çıktısına ancak
**kimliğini içeren bir yolla** ulaşılır. Aynısını aldık:

| Yol | Ne |
|---|---|
| `/output/<ad>` | YALNIZCA bu çalıştırmanın çıktıları |
| `/artifacts` | hangi çalıştırmalar var (workflow kimlikleri) |
| `/artifacts/<wf>/<ad>` | adı verilen çalıştırmanın çıktısı |

`os.listdir`, `os.path.exists` ve `glob` üç kökü de biliyor; isim çözümü
`/output` için **katı** (o workflow'a bağlı), `/artifacts/<wf>` için o
workflow'a bağlı. Yani bir çalıştırma başkasının dosyasını kazara okuyamıyor —
okumak için kimliğini yazmak zorunda.

Doğrulama (aynı senaryo, düzeltmeden sonra):

```
kendi_output_bos : True        (B hiçbir şey üretmedi)
output_sizinti   : False       (A'nın dosyası B'nin /output'unda YOK)
kosu_gorunuyor   : True        (/artifacts A'yı listeliyor)
okundu           : [{'dep': 'İK', 'ort': 40.45}]   (/artifacts/<A>/ ile)
```

Ve ajan aynı soruyu ikinci kez sorulunca **34,46** dedi — kendi rakamı,
kendi artifact'inden.

### Kapsam workflow'dan TENANT'a taşındı

Eskiden bir çalıştırma yalnızca kendi workflow'unun artifact'lerini
görebiliyordu. Artık **aynı tenant'taki her çalıştırmanın çıktısı görünür ve
okunabilir**.

Gerekçe yine emsal: KFP'de bütün run'lar aynı `pipeline_root` altına yazar ve
izolasyon sınırı **namespace**'tir. Çalıştırma başına mühürlemek bizim
eklediğimiz bir şeydi — ve ürün "başka workflow'un artifact'ini gözlemleyip
gerekirse kullanabilsin" istiyor.

Kaybedilen: çalıştırmalar arası okuma sınırı — ama **kazara** değil, ancak
kimlik yazılarak (bkz. iki kök).
Korunan: **yerleştirme** hâlâ çalıştırma başına
(`{owner}/{workflow}/{node}/{run}/…`), yani "kim ne zaman üretti" bilgisi
duruyor; künye de `workflow_id` taşıyor. Ve tenant sınırı yerinde:
başka bir `owner` hiçbir şey göremiyor.

### Canlı doğrulama (iki AYRI workflow)

```
[A: URET  ] produced  ham.tickets.parquet        (düz to_parquet, API yok)
            produced  kunye.json
            produced  model.v1.tar               (dizin -> tek tar)

[B: KESIF ] kendi_output_bos : True     ← A'nın çıktısı B'nin /output'una SIZMIYOR
            kosu_gorunuyor   : True     ← /artifacts A'yı listeliyor
            baskanin_ozeti   : ['ham.tickets.parquet', 'kunye.json', 'model.v1.tar']

[B: KULLAN] consumed  ham.tickets.parquet        (/artifacts/<A>/ ile okundu)
            consumed  kunye.json
            produced  departman.ozet.parquet     parents=[her iki girdi]
```

B, A'yı hiç tanımıyor; `pd.read_parquet("/artifacts/<A>/ham.tickets.parquet")`
yazdı, 150 satır geldi. Soy ağacı kendiliğinden kuruldu. Ve `/output`'u boş
kaldı — başkasının çıktısını kendi işi sanması artık **yapısal olarak**
imkânsız.

Ajan tarafında da doğrulandı: aynı soru ("az önce ürettiğin özette İK kaçtı?")
düzeltmeden önce başka bir run'ın rakamını veriyordu (7,46), düzeltmeden sonra
kendi rakamını verdi (34,46).

---

## §11.12 — 2026-09-06: aktarım SIDECAR'a taşındı (Argo modeli)

§11.11 LLM'e sunulan API'yi kaldırmıştı. Geriye bir çelişki kalmıştı:

| Katman | Ailemiz (§9.6) |
|---|---|
| Denetim (beş kontrol) | **C** — ayrı pod, kodun erişemeyeceği yerde |
| **Aktarımı başlatan** | **B** — kodla aynı container |

`entrypoint.py` süpürmeyi yapıyordu, yani kapsam jetonu LLM'in kodunun
okuyabileceği bir ortam değişkenindeydi. Bu, KFP'yi eleştirdiğimiz konumun
aynısıydı.

### Argo'nun cevabı

Argo Workflows bu problemi dört farklı yerleşimle denemiş (`docker`,
`kubelet`, `k8sapi`, `pns`) ve v3.4'te **hepsini kaldırmış**. `docker` için
gerekçe açık: *"breaks security completely"* — host'un `docker.sock`'unu
mount etmek gerekiyordu, OPA/PSP reddediyordu.

Kalan model:

```
init container   → girdileri paylaşılan volume'e indirir
main container   → kullanıcı kodu; SADECE düz dosya yolu okur/yazar
wait  (sidecar)  → main bitince çıktıları toplayıp yükler
```

> *"**After the main container completes**, the wait container collects output
> artifacts from the main container's filesystem through volume mounts… and
> uploads it to the configured artifact repository."*

### Bizdeki karşılığı

```
Pod ptc-sandbox-{run_id}
├── initContainers:
│     artifact-sidecar   (restartPolicy: Always → yerleşik sidecar, k8s 1.29+)
│       · PTC_SCOPE_TOKEN YALNIZCA BURADA
│       · 127.0.0.1:8099'da okuma proxy'si
│       · SIGTERM'de /output'u süpürüp yükler
└── containers:
      sandbox
        · jeton YOK · PTC_ARTIFACT_PROXY=http://127.0.0.1:8099
        · yalnızca /output'a dosya yazar
```

Kubernetes'in yerleşik sidecar semantiği tam gereken şeyi veriyor:

> *"Upon Pod termination, the kubelet **postpones terminating sidecar
> containers until the main application container has fully stopped**."*

### Asıl kazanç: jeton taşımak DEĞİL, yazma yolunu kaldırmak

Yalnızca jetonu sidecar'a taşımak **yetmezdi** — LLM'in kodu localhost
proxy'ye de aynı çağrıyı atabilirdi, yetenek değişmezdi.

Kazanç, **yükleme kararının artık sandbox'ta verilmemesi.** Proxy'de yazma uç
noktası hiç yok; sidecar neyi yükleyeceğine `/output`'a bakarak kendi karar
veriyor. LLM'in etkileyebileceği tek şey dosya yazmak — yani zaten kastedilen
arayüz. Ad seçmek, TTL koymak, depo kökü belirlemek, süpürme kuralını atlamak
mümkün değil.

İki defter de sandbox'tan çıktı:

| Defter | Eskiden | Şimdi |
|---|---|---|
| "Bunu ben indirdim, üretilmiş sayma" | `inenler`, LLM'in sürecinde | Sidecar sunduğu baytın sha256'sını tutuyor |
| Soy ağacının ebeveynleri | `okunanlar`, LLM'in sürecinde | Sidecar sunduğu `artifact_id`'ler |

İkisi de artık **kurcalanamaz**.

### Bedeli: ~1 saniye

| | Öncesi | Sonrası |
|---|---|---|
| Çalıştırma süresi (medyan) | 3,14 sn | **4,1 sn** |

İki kaynağı var: sidecar'ın açılışı (init container, seri) ve süpürmenin ana
container bittikten SONRA çalışması. İkincisini sidecar'ın bitiş sinyaliyle
kısalttık — o olmadan 6,1 sn'ydi.

### Canlı doğrulama

```
[URET ] sizan_jeton: []            ← sandbox'ta PTC_SCOPE_TOKEN YOK
        produced ilk.parquet
        produced not.txt
[TURET] consumed ilk.parquet       ← tembel okuma proxy üzerinden
        produced turev.txt  parents=['art_6c99f21f0b6a']   ← sidecar kaydetti
```

Kabul testinde ayrıca: jetonsuz doğrudan yazma **401**, proxy'ye POST **501**
(yazma uç noktası yok), pickle süpürme yolundan da **reddediliyor**.

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
| Kapsam granülaritesi | Yerleştirme çalıştırma başına, **okuma tenant** (2026-09-06) | **KFP** — `pipeline_root` paylaşımlı, sınır namespace |
| İzolasyon | Normal container | **Kimse** — herkes daha güçlü |
| Warm pool | Yok | Microsoft/Google'da var |
| LLM yüzeyi | **Yok** — düz Python + `/output` (2026-09-06) | **KFP** (`.path`), Anthropic (`$OUTPUT_DIR`), OpenAI (`/mnt/data`) |
| Şeffaf okuma (`read_csv` → tembel indirme) | Var | **Kimse** — KFP girdileri peşin indirir |
| Dizin artifact'i | Tek tar, tembel açılıyor | **KFP** (o özyinelemeli yüklüyor) |
| Çalıştırma izolasyonu | `/output` bu koşu, `/artifacts/<wf>/` adı verilen koşu | **KFP** — `pipeline_root/<run-id>/` |
| Aracının yeri | Ayrı pod, kodun erişemeyeceği yerde | **Anthropic/OpenAI** (§9.6 C ailesi) |
| Aktarımı başlatan | **Sidecar** (`wait` deseni), jeton onda | **Argo Workflows** — init + wait |

## §12.5 — Doğrudan soru: "OpenShift kullanan şirketlerin yaptığı bu mu?"

Bu soruyu dürüst cevaplamak için önce ikiye bölmek gerekiyor, çünkü
"en optimal" ile "çoğunun yaptığı" aynı şey olmak zorunda değil.

### Önce bir epistemik uyarı

**Kaç şirketin ne yaptığını bilmiyorum** — anket verisi yok, olduğunu da
iddia etmiyorum. Bilinebilen şey, Red Hat'in **belgelediği ve desteklediği**
desenler. Aşağıdaki her satır bir dokümana dayanıyor, bir yaygınlık ölçümüne
değil. (Bkz. §14.)

### Veri akışı deseni: **evet, aynısı**

OpenShift'in veri saklama deseni KFP launcher'ı (§8.2, §8.6). Karşılıklar:

| KFP | Bizde |
|---|---|
| `.path` (yerel yol) | `/output/<ad>` |
| `.uri` (nesne deposu) | `storage_uri` |
| launcher, çalışma sonrası yükler | `_ciktilari_supur()` |
| launcher, çalışma öncesi indirir | `_tembel_oku()` — tembel |
| dizin artifact'i | tek tar (§11.3) |
| `pipeline_root` | `PTC_ARTIFACT_ROOT` |
| `system.Dataset/Model/Metrics` | aynı sözlük |

Kod nesne deposuna hiç dokunmuyor, aradaki bileşen taşıyor. Şekil birebir.

### Kimlik bilgisi: **hayır, ve bilerek**

KFP'nin launcher'ı kullanıcı koduyla **aynı pod'da** ve S3 anahtarı ortam
değişkeninde — kod onu okuyabilir. Gözden geçirilmiş kod için makul, LLM'in
yazdığı kod için doğrudan açık.

Bu, "çoğunun yaptığını yapmama" kararı ve mimarinin özü. Ama şunu da
söylemek lazım: **bunu kimse yapmıyor**, yani emsali yok, yani
kanıtlanmış da değil.

**Canlı kanıt (2026-09-06, pod'un kendi içinden ölçüldü):**

| Katman | Sonuç |
|---|---|
| Ortamda S3 kimlik bilgisi | `AWS_ACCESS_KEY_ID`/`SECRET` **yok** |
| S3 SDK | `boto3` yok, `minio` yok |
| DNS ile MinIO | `gaierror` — çözülemiyor |
| **DNS'i atlayıp ClusterIP ile** | **`TimeoutError`** — Cilium paketi düşürüyor |
| Aynı koddan izinli hedef | ulaşıldı |

Son iki satır birlikte önemli: Kubernetes her servis için `MINIO_PORT_9000_TCP_ADDR`
gibi ortam değişkenlerini otomatik enjekte ediyor, yani **adres sızıyor**. Ama
adres tek başına işe yaramıyor — ağ politikası IP'ye gitmeyi de kesiyor.
(Yine de gereksiz bir ifşa; pod spec'inde `enableServiceLinks: false` ile
tamamen kapatılabilir.)

### Kayıt defteri arka ucu: **hayır, ve burada geridiz**

| | OpenShift | Bizde |
|---|---|---|
| Kayıt defteri | MLMD | kendi şemamız |
| Arka uç | **MySQL/MariaDB** | **SQLite** |
| Replika | HA | tek |

Şema tarafında hizalıyız (tipli artifact, soy, `pipeline_root`). Arka uç
seçimi kanıtlanmamış olan taraf.

### İzolasyon: **hayır, ve Red Hat'in artık bir cevabı var**

Agent Sandbox (TP) — `SandboxWarmPool` + Kata. İki açığımızı birden
kapatıyor (§8.4). Kullanmıyoruz.

### Tek cümlelik cevap

> **Veri akışında evet, kimlik bilgisinde bilerek hayır, kayıt defteri
> arka ucunda ve izolasyonda ise "hayır ve düzeltilmeli".**

Ve altı çizilmesi gereken: OpenShift'in **güvenilmeyen kod için** tam bir
artifact cevabı yok (§8.7). KFP'nin veri modeli + Agent Sandbox'ın izolasyonu
+ kimliksiz erişim için kendi çözümümüz — üçünü birleştirmek zorundayız,
çünkü hazır bir birleşimi kimse sunmuyor.

---

## Üç cümlelik özet

1. **Tezi Cloudflare'den, veri modelini Anthropic'ten, platform desenini
   Red Hat'ten alıyoruz.** Omurga tartışmasız SOTA.
2. **İki yerde kimsenin gerisindeyiz:** izolasyon (düz container) ve warm pool.
   Birincisi gerçek bir eksik, ikincisinin kazancını ölçtük — en fazla 1.6 sn.
3. ~~İki yerde kimsenin ilerisindeyiz~~ — **2026-09-06'da bu iddia geri
   çekildi.** Çalıştırma başına kapsam kaldırıldı (KFP gibi tenant), LLM'e
   sunulan artifact API'si kaldırıldı (KFP gibi düz dosya). Geriye emsalsiz
   tek şey şeffaf tembel okuma kaldı — o da bir kolaylık, üstünlük değil.
   Gerekçe §11.11'de: emsalsiz olan her yüzeyden hata çıktı, kopyaladığımız
   hiçbir parçadan çıkmadı.
4. **OpenShift ekseninde:** veri akışı deseni birebir KFP'nin (§12.5); ayrıldığımız
   iki yerden biri kasıtlı (kimlik bilgisi vermemek), diğeri eksik (SQLite,
   düz container).

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

**Kısa cevap:** Hayır — **dosya uzantısından** çıkarılıyor.

**Detay:** `.parquet`/`.csv` → `system.Dataset`, `.md` → `system.Markdown`,
`.html` → `system.HTML`, `.pdf`/`.png` → `system.Artifact`. Sözlük tek yerde
(`serialize.py`) ve o dosya sandbox imajına aynen kopyalandığı için iki taraf
ayrışamıyor. Ters yön (content_type → uzantı) da aynı sözlükten **türetiliyor**;
elle tutulan iki harita 2026-09-06'da gerçekten ayrıştı ve PDF'ler depoda
`.bin` olarak duruyordu.

**Kaybedilen bir şey var:** eskiden `put_artifact(deger, ...)` **nesneye**
bakabiliyordu ve "sayısal sözlük = `system.Metrics`" çıkarımını yapabiliyordu.
API kalkınca o bilgi kayboldu — artık `.json` her zaman `system.Artifact`.
Metrik/Dataset ayrımının kaybı §11.10'da açık olarak yazılı.

**Bu yüzden dosya adı önemli:** uzantısız ad verilirse tip
`application/octet-stream`'e düşüyor. Sistem promptu bunu açıkça söylüyor.

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
5. adım *yazılmadan önce var değil*, o yüzden keşif zorunlu. Bizde üç yol var,
üçü de düz Python: isimler promptta (manifest), dosya sistemi
(`os.listdir` / `os.path.exists` / okuma), ve konuşma hafızası. Hiçbiri tek
dayanak değil.

### S: "Peki model manifesti okumazsa / promptu atlarsa?"

**Kısa cevap:** Dosya sistemi de aynı gerçeği söylüyor — iki bağımsız yol var.

**Detay:** Eskiden bu gerçek bir açıktı: keşif `list_artifacts()` çağırmaya
bağlıydı ve model unutabiliyordu. İki şey değiştirdi. **Bir:** isimler artık
her turda prompt'a enjekte ediliyor (ADK'nın `LoadArtifactsTool` deseni —
isimler talimatlarda, içerik talep üzerine, içerik geçmişe kalıcı yazılmıyor).
**İki:** `os.listdir("/output")` de manifestle birleştirildi, yani model
promptu hiç okumasa bile dosya sistemine bakarak aynı listeyi görüyor.

Kalan risk manifest yolunun **sessizce** devre dışı kalması: `ARTIFACT_SERVICE_URL`
tanımsızsa enjeksiyon olmuyor. 2026-09-06'da tam bunu yaşadık (bkz. §11.10) —
artık bir kez uyarı basılıyor.

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

**Ama bir kısmı zaten yaşandı:** keşif tenant'a genişleyince liste 80+ satıra
çıktı ve manifest de panel de okunmaz hâle geldi. Çözüm aramak değil
**ayırmak** oldu: manifest ikiye bölündü (bu oturum / başka çalıştırmalar) ve
`/output` ile `/artifacts/<wf>` ayrı köklere konuldu. N büyüdüğünde ilk
yapılacak şey arama değil, kapsamı daraltmak.

### S: "Kayıt defterinde soy ağacı (lineage) var mı? Kullanılıyor mu?"

**Kısa cevap:** Var, **otomatik doluyor**, ve panelde okunuyor.

**Detay:** Kural KFP/MLMD'nin yaptığının aynısı — adımın çıktısı adımın
girdilerinden türemiş sayılır. Fark: KFP bunu **bildirimden** alır
(`Input[Dataset]` imzada), bizde derlenmiş bir imza olmadığı için
**gözlemden** alınıyor: bir çalıştırmada baytı gerçekten okunan artifact'ler,
o çalıştırmada üretilenlerin ebeveyni oluyor. Küme çağrı anında okunduğu için
nedensellik doğru — bir yazmadan sonra okunan şey o yazmanın ebeveyni olmuyor.

Okuma tarafı da var: `GET /artifacts/{id}/lineage` atalarla ürünleri döndürüyor,
panel bunu mermaid grafiği olarak çiziyor.

**Kalan açık:** soy **imzasız**. Kayıt defterine yazabilen değiştirebilir ve
kimse fark etmez — Tekton Chains'in çözdüğü problem (§8.6).

### S: "İki çalıştırma aynı ismi kullanırsa ne olur?"

**Kısa cevap:** Kendi çalıştırmanınki kazanır; başkasınınkine ancak kimliğini
yazarak ulaşırsın.

**Detay:** Bu soru 2026-09-06'da **gerçek bir arıza** olarak yaşandı. Keşif
kapsamı tenant'a genişleyince `/output` bütün çalıştırmaların çıktılarını düz
bir liste olarak gösteriyordu; ajan 1. turda ürettiği analizi 2. turda sorunca
**başka bir run'ın** aynı adlı dosyasını okuyup yanlış sayı verdi. Hiçbir yerde
hata yoktu.

Çözüm KFP'nin düzeni: her çalıştırmanın kendi dizini var.
`/output/<ad>` katı biçimde bu çalıştırmayı çözüyor, başkasınınki
`/artifacts/<workflow_id>/<ad>`. Kazara okumak artık mümkün değil.

Aynı workflow içinde aynı ad iki kez kullanılırsa hâlâ "en yeni" kazanıyor ve
model bunun farkında olmuyor — bu sınır duruyor (§11.10).

### S: "Bu uuid'yi bilen başkasının artifact'lerini okuyabilir mi?"

**Kısa cevap:** Evet — ve kapsam 2026-09-06'da **genişledi**, o yüzden bedeli
de büyüdü.

**Detay:** Yetki sınırı artık workflow değil **tenant** (`owner`): jetonu
üretebilen, o tenant'taki her çalıştırmanın çıktısını okuyabilir. Bu bilinçli
bir hizalama — KFP'de de izolasyon namespace düzeyinde, ve ürün "başka
workflow'un artifact'ini kullanabilsin" istiyordu.

Ama bu kimlik doğrulama **değil**. Uuid tahmin edilemez olduğu için PoC'de
yeterli; üretimde gerçek auth'a bağlanmalı. Tenant sınırı yerinde duruyor:
başka bir `owner` hiçbir şey göremiyor (canlı doğrulandı, 404).

---

# §14 — Doğrulanamayanlar

Ekibe sunmadan önce bakılması gerekenler:

1. **Red Hat "kimlik bilgileri sandbox'ta saklanmaz, ağ sınırında enjekte
   edilir"** cümlesi — arama özetinde çıktı, makalenin kendisinden birebir
   teyit edemedim.
2. ~~**OpenAI bölümü** (§3) birincil kaynaktan çekilmedi~~ — **kapandı**
   (2026-09-06): `developers.openai.com` dokümanından doğrudan alındı, 20
   dakikalık ömür ve "treat containers as ephemeral" cümlesi §9.5.2'de
   birebir alıntı.
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

## "Çoğu şirket şunu yapıyor" diyemediğim yer

12. **§12.5'teki yaygınlık sorusu.** Hangi desenin kaç OpenShift müşterisinde
    kullanıldığına dair elimde **hiçbir veri yok** — anket, kullanım raporu,
    telemetri, hiçbiri. Dokümandaki her "OpenShift şöyle yapıyor" cümlesi
    Red Hat'in **belgelediği ve desteklediği** deseni anlatıyor; kaç kişinin
    öyle yaptığını değil. İkisi çoğu zaman örtüşür ama aynı şey değildir, ve
    sunumda "çoğu şirket böyle yapıyor" diye söylenirse savunulamaz.

## §9.5'te bilerek "belgelenmemiş" yazdığım yerler

Bunlar araştırma eksiği değil — sağlayıcı arayüzünü belgeliyor, arka ucunu
belgelemiyor. Tahmin yazmamak için boş bıraktım:

7. **Anthropic Files API'nin arka uç deposu.** Container diskinin 5 GiB
   olduğu, `$OUTPUT_DIR` yakalamasının nasıl çalıştığı ve 30 günlük ömür
   birincil dokümanda var; `file_id`'nin işaret ettiği baytların hangi
   üründe durduğu yok.
8. **OpenAI container'ının arka uç deposu** — aynı durum. Dokümanın kendisi
   zaten "kendi sisteminde sakla" diyerek bu soruyu kapatıyor.
9. **Modal Volumes'ün hangi bulut sağlayıcılarına dayandığı.** Doküman
   *"backed by multiple underlying cloud providers"* diyor, isim vermiyor.
10. **Vertex AI Agent Engine'in oturum/hafıza deposu.** Sessions'ın ne
    sakladığı belgeli, nerede sakladığı değil; ayrı bir artifact storage
    ürünü olarak da sunulmuyor (ADK'nın `ArtifactService`'i ondan bağımsız).
11. **Fly.io'nun kullandığı nesne deposu.** Blog *"S3-compatible object
    storage"* diyor ve tasarımı ayrıntılı anlatıyor, ama hangi ürün
    (Tigris mi, başkası mı) yazılmıyor — ve yazarı zaten *"because it's in
    flux, let's keep it simple"* diye uyarıyor.

---

# Kaynaklar

**Anthropic**
- [Code execution tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)
- [Programmatic Tool Calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)

**OpenAI**
- [Code interpreter tool](https://developers.openai.com/api/docs/guides/tools-code-interpreter) — 20 dk ömür, "treat containers as ephemeral" (§9.5.2)

**Cloudflare**
- [Cloudflare Sandbox](https://developers.cloudflare.com/sandbox/)
- [Code Mode](https://blog.cloudflare.com/code-mode/)
- [Sandbox — Storage](https://developers.cloudflare.com/sandbox/api/storage/)

**Google**
- [ADK — Artifacts](https://adk.dev/artifacts/) — `InMemory` / `Gcs` / `File` ArtifactService (§9.5.4)
- [Agent Engine Code Execution](https://cloud.google.com/agent-builder/agent-engine/code-execution/overview)
- [Code Execution troubleshooting](https://docs.cloud.google.com/agent-builder/agent-engine/troubleshooting/code-execution)
- [About GKE Agent Sandbox](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/machine-learning/agent-sandbox)

**AWS**
- [Bedrock AgentCore Code Interpreter](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html)
- [File system configurations for AgentCore Code Interpreter](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-filesystem-configurations.html) — S3 Files / EFS mount (§9.5.5)

**Microsoft**
- [Dynamic sessions in Azure Container Apps](https://learn.microsoft.com/en-us/azure/container-apps/sessions)
- [Serverless Code Interpreter Sessions](https://learn.microsoft.com/en-us/azure/container-apps/sessions-code-interpreter) — `/mnt/data`, 128 MB sınırı (§9.5.6)

**Red Hat / OpenShift**
- [OCP 4.17 Storage](https://docs.redhat.com/en/documentation/openshift_container_platform/4.17/html-single/storage/index)
- [OpenShift AI — Managing data science pipelines](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.25/html/working_with_data_science_pipelines/managing-data-science-pipelines_ds-pipelines)
- [OpenShift AI — Connect workbench to S3](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.25/html-single/working_with_data_in_an_s3-compatible_object_store/index)
- [OpenShift Pipelines 1.16](https://docs.redhat.com/en/documentation/red_hat_openshift_pipelines/1.16/html/about_openshift_pipelines/understanding-openshift-pipelines)
- [Tekton — Artifact Provenance Data](https://tekton.dev/docs/pipelines/artifacts/) — `$(step.artifacts.path)`, uri+digest (§8.3, §8.6)
- [Tekton — Artifact Storage in Chains](https://tekton.dev/blog/2026/03/02/artifact-storage-in-tekton-chains/) — 7 arka uç, *"not artifact content"* (§8.6)
- [kubernetes-sigs/agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) — Agent Sandbox upstream'i: `Sandbox` / `SandboxTemplate` / `SandboxClaim` (§8.4)
- [Layered sandboxing for AI agents](https://developers.redhat.com/articles/2026/07/16/layered-sandboxing-ai-agents-openshift-and-openshell)

**Kubeflow Pipelines**
- [V2 Execution System: Driver and Launcher](https://deepwiki.com/kubeflow/pipelines/6.2-v2-execution-system:-driver-and-launcher) — driver init container, launcher main container wrapper (§9.6.2)
- [Object Store Configuration](https://www.kubeflow.org/docs/components/pipelines/operator-guides/configure-object-store/)
- [Create, use, pass, and track ML artifacts](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/)
- [Pipeline Root](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/pipeline-root/)

**GKE FUSE CSI**
- [Cloud Storage FUSE CSI driver](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/cloud-storage-fuse-csi-driver)

**Uzman sağlayıcılar (§9.5.8 için birincil kaynaklar)**
- [E2B — Sandbox persistence](https://e2b.dev/docs/sandbox/persistence) · [Snapshots](https://e2b.dev/docs/sandbox/snapshots)
- [Modal — Volumes](https://modal.com/docs/guide/volumes) · [CloudBucketMount](https://modal.com/docs/guide/cloud-bucket-mounts)
- [Fly.io — The Design & Implementation of Sprites](https://fly.io/blog/design-and-implementation/)

**Uzman sağlayıcılar** — ayrıntılı tablo, alıntılar ve tek tek kaynak linkleri:
[PTC_Sandbox_Depo_Dogrudan_Erisim_Arastirmasi.md](PTC_Sandbox_Depo_Dogrudan_Erisim_Arastirmasi.md) §2.1
