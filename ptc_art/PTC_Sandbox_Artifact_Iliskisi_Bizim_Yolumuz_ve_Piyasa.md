# Sandbox ile Artifact Deposu Arasındaki İlişki

**Bizim yolumuz baştan sona, piyasadaki yöntemlerle yan yana, ve dürüst bir SOTA değerlendirmesi**

**Tarih:** 2026-09-04

Bu doküman üç soruyu sırayla cevaplıyor:

1. **Bizim yolumuz tam olarak nasıl işliyor?** — bir dosyanın doğuşundan sonraki
   turda okunmasına kadar, adım adım.
2. **Aynı işi piyasa nasıl yapıyor?** — her adımda alternatifler neler.
3. **Bizimki gerçekten herkesin kullandığı SOTA yöntem mi?** — bunun cevabı
   "evet" değil. Kısmen evet, kısmen bilinçli bir azınlık tercihi, ve bir
   yerde de kimseye benzemiyor. Üçünü de ayırıyorum.

Dayandığı dokümanlar:
[PTC_Mimari.md](PTC_Mimari.md) ·
[PTC_Artifact_Kaydetme_Sureci.md](PTC_Artifact_Kaydetme_Sureci.md) ·
[PTC_Sandbox_Depo_Dogrudan_Erisim_Arastirmasi.md](PTC_Sandbox_Depo_Dogrudan_Erisim_Arastirmasi.md) ·
[PTC_Hedef_Mimari_ve_Karar_Dokumani.md](PTC_Hedef_Mimari_ve_Karar_Dokumani.md) ·
[sandbox_artifact_persistence_architecture.md](sandbox_artifact_persistence_architecture.md)

---

# BÖLÜM 0 — Problem tam olarak ne

## 0.1 En baştan: neden böyle bir şeye ihtiyaç var

LLM'e veriyle iş yaptırmanın bir yolu, ona kod yazdırıp o kodu çalıştırmaktır
(PTC — Programmatic Tool Calling). Ama LLM'in yazdığı kod **güvenilmeyen
koddur**: ne yapacağını önceden bilemezsin. Bu yüzden kendi makinende değil,
izole bir yerde çalıştırırsın. O izole yere **sandbox** denir.

Sandbox'ın izole olması için en pratik yol, onu her çalıştırmada **sıfırdan
yaratıp sonunda yok etmektir**. Kalıcı bir şey kalmazsa, bir çalıştırmadan
diğerine bulaşacak bir şey de kalmaz.

Ama burada bir çelişki doğuyor:

> Sandbox'ın **yok olması** güvenlik için gerekli.
> Ürettiği şeyin **kalması** ise işin kendisi için gerekli.

Kullanıcı "satışları çıkar" der, 40 saniyelik bir işlem sonunda bir tablo
oluşur, sandbox ölür, tablo da onunla ölür. Sonra kullanıcı "peki bunu
departmana göre grupla" derse, her şey baştan hesaplanır.

**Artifact persistence** bu çelişkiyi çözme işidir: sandbox ölsün, ürettiği
şey kalsın.

## 0.2 Tek cümlelik kural

Bütün mimari şu cümleden türüyor:

> **Sandbox'ın yaşam süresi, ürettiği şeyin yaşam süresini belirlememeli.**

Bu cümle bize ait değil; bağımsız bir kaynak da aynı yere varıyor
([sandbox_artifact_persistence_architecture.md](sandbox_artifact_persistence_architecture.md)
§1). Yani bu tartışmalı bir tercih değil, alanın ortak kabulü.

## 0.3 Üç ayrı şeyi karıştırmamak

En sık yapılan hata bu üçünü aynı kutuya koymak:

| Ne | Örnek | Nerede durmalı |
|---|---|---|
| **Geçici dosya** | ara hesap, cache, yarım çıktı | Sandbox'ın kendi diski — **ölmesi istenir** |
| **Artifact** | 40 sn'de üretilen tablo, rapor, model | Kalıcı depo — yeniden üretmek pahalı |
| **State** | "konuşmada nerede kaldım", hangi artifact'i kullanacağım | Ayrı bir yerde, ama o da kalıcı |

Üçüncüsü en çok atlanan. Artifact'i kalıcı yapıp state'i unutursan, artifact
depoda sağ kalır ama **ona nasıl ulaşacağını** kaybedersin. Bizde de tam olarak
bu oldu, §1.1'de anlatıyorum.

---

# BÖLÜM 1 — Bizim yolumuz, adım adım

Tek bir örneği baştan sona takip edelim:

> **Tur 1:** kullanıcı "satışları çıkar" der → bir CSV üretilir
> **Tur 2:** kullanıcı "toplamı da söyle" der → o CSV yeniden üretilmeden okunur

## 1.1 Adım 0 — Oturum kimliği (her şeyin anahtarı)

Kullanıcı tarayıcıda sayfayı açtığında istemci `localStorage`'dan bir uuid
okur; yoksa üretir ve saklar. WebSocket'e `?session=<uuid>` olarak gider.

Bu tek değer **iki yere birden** gidiyor:

```
oturum kimliği ──> thread_id     (konuşmanın hafızası)
               └─> workflow_id   (artifact'lerin kapsamı)
```

**Neden önemli:** bu değer önceden her bağlantıda yeniden üretiliyordu. Sonuç
şuydu — sayfa yenilenince artifact'ler depoda **sağ kalıyor** ama onları
gösteren `workflow_id` bir daha asla üretilmediği için **erişilemez** hale
geliyorlardı. Kalıcı bir depoya yazıp okuma anahtarını çöpe atmak.

Kimlik gönderilmezse temiz bir oturum açılır: **kalıcılık opt-in**, kimse
istemeden başkasının oturumuna düşmez.

> **Sınır, açıkça:** bu kimlik doğrulama değil. Uuid'yi bilen o oturumun
> artifact'lerini okur. Tahmin edilemez olduğu için PoC'de yeterli; üretimde
> gerçek auth'a bağlanmalı.

## 1.2 Adım 1 — Agent kod yazar

LangGraph ajanı `run_ptc_code(code)` tool'unu çağırır. Bu, veriye erişmenin
**tek** yoludur; agent'a bağlı başka tool yoktur.

İlk kontrol bütçe: bir turda en fazla **2** sandbox çalıştırması. Birincisi
hata verirse ikincisi düzeltmedir. Bu sınır, artifact persistence'ın değerini
belirleyen şey — `cached()` olmadan o düzeltme pahalı işi de baştan yapar.

## 1.3 Adım 2 — Kapsam jetonu imzalanır

`sandbox_runner` bir Kubernetes Secret'ından imza anahtarını okur ve
HMAC-SHA256 ile 15 dakika ömürlü bir **kapsam jetonu** üretir. İçinde
`workflow_id`, `run_id`, `owner`, `node_id` var.

**İncelik:** jeton laptop'ta (sandbox'ın erişemediği yerde) imzalanır.
Sandbox'ın eline yalnızca imzalanmış sonuç geçer, imza anahtarı hiç geçmez. Bu
yüzden sandbox jetonu okuyabilir ama **başka bir workflow için geçerli bir
jeton üretemez**.

## 1.4 Adım 3 — Pod doğar

Bir Kubernetes Job yaratılır. Pod'un içinde şunlar mount'lu:

| Yer | Cinsi | Ömrü | Ne için |
|---|---|---|---|
| `/sandbox` | configMap | Salt okunur | LLM'in kodu |
| `/scratch` | emptyDir 512Mi | **Pod ile ölür — istenir** | Geçici dosyalar, cache |
| `/output` | emptyDir 512Mi | Süpürülür | Kalması istenen çıktılar |

`/scratch` ile `/output`'un ayrı olması kasıtlı: `/scratch` süpürülseydi her ara
dosya, her kütüphane cache'i artifact'e dönerdi.

Pod'un ağ çıkışı **iki hedefle** sınırlı: Tool Gateway (tool'lar için) ve
Artifact Service (veri için). Nesne deposuna (MinIO) **rotası yok**, S3 kimlik
bilgisi **yok**.

## 1.5 Adım 4 — Manifest çekilir (tek istek)

Pod açılır açılmaz Artifact Service'e **bir** istek gider:
*"bu workflow'da hangi isimler var?"* Sadece isimler döner, bayt dönmez.

**Neden sadece isimler:** önceki tasarımda pod doğarken depodaki **her**
artifact indiriliyordu. Maliyet var olan her şeyle ölçekleniyordu — workflow'da
6 tane 100 MiB'lik artifact biriktiğinde, hiçbirine dokunmayan bir script bile
512Mi'lık `/output`'u patlatıyordu. Şimdi maliyet **kullanılan kadar**.

## 1.6 Adım 5 — Kod çalışır: YAZMA yolu

LLM'in çıktıyı kalıcılaştırmasının **iki** yolu var ve ikisi de aynı kapıdan
geçiyor:

**A — Açık çağrı.** LLM `put_artifact(df, name="satislar")` yazar. DataFrame
Parquet'e çevrilir (tipler korunur), ham bayt olarak servise akar.

**B — Süpürme (emniyet ağı).** LLM sıradan kod yazar:
`df.to_csv("/output/satislar.csv")`. Script bitince — **başarı ya da hata fark
etmeksizin** — `/output`'un üst düzeyi taranır ve içindekiler artifact'e
çevrilir.

```python
try:
    exec(code)
except Exception:
    _ciktilari_supur(...)   # ← hata olsa bile ÖNCE süpür
    print(hata); exit
_ciktilari_supur(...)       # ← başarıda da
```

**B neden var:** A yolu, LLM'in o API'yi *bilmesini ve hatırlamasını*
gerektiriyor. Bilmezse ürettiği her şey pod'la birlikte kaybolurdu. B'nin asıl
değeri de hata durumunda: script son satırda patlasa bile o ana kadar üretilen
dosyalar kurtarılır.

## 1.7 Adım 6 — Kod çalışır: OKUMA yolu (tembel)

Tur 2'de LLM `pd.read_csv("/output/satislar.csv")` yazıyor. Ama bu **yepyeni
bir pod**; o dosya orada yok.

Devreye tembel doldurma giriyor: `pd.read_csv`/`read_parquet`/`read_json`/
`read_excel` sarmalanmış, ayrıca sandbox'ın globals'ına tembel bir `open`
konmuş. Dosya `/output`'ta yoksa **ve** adı manifestte geçiyorsa, o an
indirilir; sonra pandas normal şekilde okur.

`builtins` **değiştirilmiyor** — sadece LLM'in doğrudan çağrısı yakalanıyor,
kütüphanelerin iç dosya işlemleri hiç etkilenmiyor.

**Bir incelik:** indirdiğimiz dosyanın (değişim zamanı, boyut) çifti
kaydediliyor. Süpürme, dokunulmamış olanı atlıyor — yoksa sadece *okuyan* bir
çalıştırma bile dosyayı "üretilmiş" sayıp geri yüklerdi. (Bu, prefetch
döneminde gerçekten yaşanmış bir kusurdu.)

## 1.8 Adım 7 — Artifact Service dört kontrol yapar

Baytlar akarken, **akış sırasında**:

| Kontrol | Ne yapıyor | Neden |
|---|---|---|
| **Kapsam** | `workflow_id`'yi jetondan okur, çağıranın iddiasından değil | Paylaşılan depo, ağ politikasının **göremediği** bir kanal açar |
| **İsim** | `../`, `/`, boşluk reddedilir | `artifact_save("/etc/shadow")` sınıfı |
| **Boyut** | Sayarak; sınır aşılınca okuma **orada** kesilir | Veriyi sandbox içinde süzmeye zorlar |
| **Format** | pickle reddi — hem etiketten hem bayt imzasından | Deserialization kod çalıştırır (CWE-502); baytları LLM'in kodu yazdı |

**Akış sırasında olması önemli:** pickle imzası ilk iki bayttadır, yani ilk
parçada karar verilir ve reddedilen yükleme depoya **tek bayt bile** yazmaz.
Gövde 8 MiB'ı aşınca diske taşar — süreç belleği yükleme boyutundan bağımsız
kalır.

## 1.9 Adım 8 — İki ayrı depoya yazılır

```
baytlar  → MinIO (S3-uyumlu)     → "veri burada"
kayıt    → SQLite kayıt defteri  → "o verinin hikâyesi"
```

Kayıt defterinde: `artifact_id`, isim, workflow, içerik hash'i, boyut, depo
adresi, türediği artifact'ler (lineage), TTL.

**Bu ayrım olmadan elde artifact store değil, sadece bir bucket olur.**

İki kural:

- **Değişmezlik.** Aynı isme ikinci kez yazmak eskisini ezmez; **yeni bir
  `artifact_id`** doğar. Okuma en yeniyi çözer. Böylece rollback, reproducibility
  ve denetim mümkün kalır.
- **Silme sırası: önce bayt, sonra kayıt.** Ters sıra yetim blob bırakır
  (görünmez maliyet); bu sıra en kötü ihtimalle sarkan referans bırakır — o
  tespit edilebilir.

Ayrıca **içerik-hash dedup**: aynı içerik ikinci kez yazılırsa bayt tekrar
yüklenmez, yeni kayıt var olan adresi gösterir.

## 1.10 Adım 9 — Pod ölür

Job biter, pod silinir, `/scratch` ve `/output` yok olur. ConfigMap'i
Kubernetes'in kendi çöp toplayıcısı alır (`ownerReference`).

Geriye kalan: MinIO'daki baytlar + kayıt defterindeki satır.

## 1.11 Adım 10 — Sonraki tur

Kullanıcı "toplamı da söyle" der. Aynı oturum kimliği → aynı `workflow_id`.
Yeni pod, manifest çekiyor, `satislar.csv`'yi görüyor, LLM `read_csv` yazınca
dosya iniyor. **40 saniyelik iş tekrarlanmıyor.**

## 1.12 Arka planda: TTL toplayıcı

Artifact'ler sonsuza kadar birikmesin diye saat başı çalışan bir CronJob,
servisin `/admin/reap` ucunu çağırıyor. TTL'i olmayan artifact'lere
dokunulmuyor (varsayılan bu). Yetki ayrı bir jetonda — sandbox'ın kapsam jetonu
buraya **yetmiyor**, çünkü LLM'in ürettiği kodun toplu silme tetikleyebilmesi
kalıcılığı tek çağrıda geri alırdı.

## 1.13 Bütün resim

```
┌── Sandbox Pod (her çalıştırmada YENİ) ────────────────────┐
│  /sandbox (kod)  /scratch (geçici)  /output (süpürülür)   │
│                                                            │
│  put_artifact(df, name=...)      ← açık yol               │
│  df.to_csv("/output/x.csv")      ← emniyet ağı            │
│  pd.read_csv("/output/x.csv")    ← tembel doldurma        │
└──────────┬──────────────────────────────┬─────────────────┘
           │ tool'lar                     │ artifact baytları
           │ (MCP)                        │ (akışlı HTTP)
┌──────────▼──────────┐        ┌──────────▼─────────────┐
│    Tool Gateway     │        │   Artifact Service     │
│  internet ✓ depo ✗  │        │  internet ✗ depo ✓     │
└─────────────────────┘        └───┬────────────────┬───┘
                                   │                │
                            ┌──────▼─────┐   ┌──────▼──────┐
                            │   MinIO    │   │   SQLite    │
                            │  BAYTLAR   │   │ KAYIT DEFT. │
                            └────────────┘   └─────────────┘
```

---

# BÖLÜM 2 — Aynı işi piyasa nasıl yapıyor

Şimdi her kararı alternatifleriyle yan yana koyalım.

## 2.1 Sandbox nerede çalışır — izolasyon

| Yöntem | Ne demek | Güç | Hız |
|---|---|---|---|
| **Normal container (runc)** | Kernel paylaşılır, namespace ile ayrılır | En zayıf | En hızlı |
| **gVisor** | Araya kullanıcı-alanı kernel girer | Orta | Orta |
| **Kata / OpenShift Sandboxed Containers** | Container başına hafif VM | Güçlü | Daha yavaş açılır |
| **Yönetilen platformlar** (E2B, Modal, Daytona) | Kendi runtime'ları | Değişken | Optimize |

**Biz:** normal container. **Hedef:** Kata (OpenShift'in kendi ürünü, ilan
edilmiş amacı güvenilmeyen kod).

## 2.2 Sandbox depoya nasıl bağlanır — dört aile

Bu, dokümanın en önemli tablosu. Araştırmamızda birincil kaynaklardan
doğrulanan hâli:

| Sistem | Mekanizma | Kimlik bilgisi sandbox'ta mı |
|---|---|---|
| **E2B** | Sandbox içinde `sudo s3fs`/`gcsfuse` | **Evet** — dosyaya yazılıyor |
| **Modal** | `CloudBucketMount` | Belirsiz |
| **Daytona — Volumes** | Platform mount ediyor | **Hayır** |
| **Daytona — External storage** | Snapshot'a `mount-s3` gömülüyor | **Evet** — env ile |
| **Vercel — düz mount** | `sudo mount-s3` | **Evet** (doküman uyarıyor) |
| **Vercel — proxy'li mount** | `mount-s3 --no-sign-request` + imzalayan Function | **Hayır** |
| **Cloudflare Sandbox** | `sandbox.mountBucket()` | **Hayır** (binding/proxy) |
| **Fly.io Sprites** | Depo blok cihaz olarak sunuluyor | Yok — S3'ü hiç görmüyor |
| **Anthropic code execution** | **Mount yok.** İnternet kapalı, dosyalar Files API'den | — |
| **OpenAI Code Interpreter** | **Mount yok.** Container files uçları | — |

Dört aile:

1. **Sandbox içinde FUSE mount** — kimlik bilgisi sandbox'a giriyor
2. **Platform tarafından mount** — kimlik bilgisi sandbox'ta yok
3. **Kimlik-bilgisiz mount + imzalayan proxy** — Vercel/Cloudflare deseni
4. **Mount hiç yok, API üzerinden** — Anthropic, OpenAI

**Biz:** 4. aile. Sandbox'ın deposu görmesi diye bir şey yok; her şey servis
üzerinden.

## 2.3 Kayıt defteri korunuyor mu

Bu bulgu keskin ve doğrudan bizim tezimizi destekliyor:

> **Taranan sağlayıcıların hiçbirinin bucket-mount yolunda artifact registry'si
> yok.** Mount edilen bucket'a yazılan dosyanın `artifact_id`'si, lineage'i,
> TTL'i, content-hash'i yok — sadece bir S3 anahtarı var. Dokümanlar bunu
> "persistent data access", "share state across Sandboxes" diye pazarlıyor;
> hiçbiri "artifact" demiyor.
>
> Tek istisna **Anthropic**: orada zaten mount yok, Files API var ve o bir
> kayıt defteri (`id`, `filename`, `mime_type`, `size_bytes`, `created_at`,
> `expires_at`).

Yani: **kayıt defteri, yalnızca yazma yolu bir bileşenden geçtiğinde ayakta
kalıyor.** Doğrudan mount = "sadece bucket".

## 2.4 İçerik doğrulaması nerede yapılıyor

Araştırmanın bulgusu: **yazma yolunda hazır bir mekanizma yok.** Bucket
policy ile içerik-tipi/boyut kısıtlamanın sınırlı yolları var ama sahada
"sandbox yazarken içeriği denetle" diye bir standart desen bulunamadı.

**Biz:** yazma anında, akış sırasında, dört kontrol. Bu bizi **azınlıkta**
bırakıyor ama azınlıkta olmamızın sebebi başkalarının daha iyi bir yolu bulmuş
olması değil — çoğu sistem bu problemi hiç çözmüyor, çünkü onların
sandbox'ında çalışan kodu çoğu zaman *kullanıcı* yazıyor, bizimkini **LLM**
yazıyor.

## 2.5 Yazmayı kim tetikliyor

| Sistem | Mekanizma |
|---|---|
| **Anthropic** | `$OUTPUT_DIR` — komut bitince dizinin üst düzeyi yakalanır |
| **OpenAI Code Interpreter** | `/mnt/data` — aynı konvansiyon |
| **OpenHands** | Workspace dizini — dosyalar zaten kalıcı |
| **Cloudflare Code Mode** | *(istisna)* dosya sistemi yok, açık RPC zorunlu |

**Hiçbir SOTA sistem bu riski açık bir tool'a bağlı bırakmıyor.** Hepsi
"LLM sıradan kod yazar, bir dizin süpürülür" deseninde.

**Biz:** aynı desen (`/output`), artı açık API. İkisi rakip değil.

## 2.6 Okuma nasıl oluyor

Burada durum farklı. Mount eden sistemlerde okuma "zaten orada" — dosya sistemi
gibi görünüyor. Mount etmeyen sistemlerde (Anthropic, OpenAI) okuma **açık bir
çağrı**: dosyayı Files API'den ID ile istersin.

Paylaşılan mimari dokümanının §27'si de bu ikinciyi öneriyor:
`artifact.load("art_789")` — yani LLM'in artifact ID'sini bilmesi ve taşıması.

**Biz:** üçüncü bir yol. Mount yok ama LLM'e ID de taşıtmıyoruz — `read_csv`
şeffaf çalışıyor, dosya arka planda iniyor. Buna §3.4'te döneceğim, çünkü
**bu kısım özgün ve o yüzden risk taşıyor.**

## 2.7 Kapsam — bir sandbox başkasının verisini neden okuyamıyor

| Yaklaşım | Granülarite |
|---|---|
| Uzun ömürlü bucket kimlik bilgisi | Bucket — hiç kapsam yok |
| Prefix bazlı IAM policy | Prefix |
| Oturum başına ayrı bucket/OBC | Bucket, ama her oturumda K8s objesi |
| Kubernetes ServiceAccount + IRSA | **ServiceAccount** |
| **Bizim kapsam jetonumuz** | **Çalıştırma (run)** |

Araştırmanın tespiti: Kubernetes tarafında pod-başına kimlik mümkün ama
**granülarite ServiceAccount'ta takılıyor**. Her çalıştırma için ayrı
ServiceAccount pratik değil.

Bizim HMAC imzalı jetonumuz her çalıştırma için ayrı üretiliyor ve içinde
workflow/run/owner/node var. Bu **çoğu sağlayıcıdan dar**.

## 2.8 Sandbox ömrü

| Model | Kim kullanıyor |
|---|---|
| **A — Oturum boyunca yaşayan tek sandbox** | Google ADK (variables/imports/file state taşınır), OpenAI Sandbox Agents (snapshot/resume) |
| **B — Her çalıştırmada yeni sandbox** | Bizim yolumuz; paylaşım artifact store üzerinden |
| **Hibrit** | Bazı adımlar aynı sandbox'ta, önemli çıktılar artifact'e |

**Biz:** B. Ölçtük — bir çalıştırma 3.14 sn ve bunun sadece 1.62'si pod
başlatma. Yani warm pool kursak kazanacağımız en fazla 1.6 saniye; izolasyonu
bozmaya değmez.

## 2.9 Format

Ortak kabul: **Parquet gibi kolonlu formatlar** (sıkıştırma, şema, hızlı kolon
erişimi). Ve **pickle yasak** — deserialization kod çalıştırır.

**Biz:** Parquet varsayılan, pickle hem etiketten hem bayt imzasından reddedilir.
Bu noktada tam uyum.

---

# BÖLÜM 3 — Bizimki gerçekten SOTA mı?

Dürüst cevap: **kısmen.** Dört ayrı kategori var ve karıştırmamak lazım.

## 3.1 Çoğunlukla aynı olduğumuz yerler — burada güvendeyiz

Bunlar tartışmasız, birden fazla bağımsız kaynak aynı yere varıyor:

| Karar | Kim daha yapıyor |
|---|---|
| Efemer sandbox + kalıcı dış depo | SOTA çoğunluğu |
| Nesne deposu (bayt) + kayıt defteri (hikâye) ayrımı | Kubeflow, MLflow, Anthropic Files API |
| `/output` süpürme konvansiyonu | Anthropic `$OUTPUT_DIR`, OpenAI `/mnt/data`, OpenHands |
| Parquet varsayılan, pickle yasak | Genel kabul |
| Değişmezlik + içerik-hash | Metaflow'un content-addressed datastore'u |
| İçerideki hesaplama / dışarıdaki depo ayrımı | Daytona'nın control/compute plane ayrımı |

**Bu kategoride "biz farklı bir şey yapıyoruz" diye bir kaygı yok.**

## 3.2 Bilinçli azınlıkta olduğumuz yer — mount etmiyoruz

Ticari sağlayıcıların **çoğu mount ediyor** (E2B, Modal, Daytona, Vercel,
Cloudflare, Fly). Biz etmiyoruz. Yani sayı olarak azınlıktayız.

Ama sayı yanıltıcı. Şuna dikkat:

> **Bizim işimize en çok benzeyen iki sistem — Anthropic'in code execution'ı ve
> OpenAI'ın Code Interpreter'ı — mount ETMİYOR.**

Ve fark tesadüf değil, kullanım amacından geliyor:

| | Ticari sandbox sağlayıcıları | Anthropic / OpenAI / biz |
|---|---|---|
| Sandbox'taki kodu kim yazıyor | Genellikle **kullanıcı** | **LLM** |
| Müşteri kim | Geliştirici (kendi kodunu koşturuyor) | Son kullanıcı (kodu görmüyor bile) |
| Ne satılıyor | "Bucket'ını mount et, esnek ol" | "Güvenli bir yetenek" |

Kullanıcının kendi kodunu koşturduğu bir üründe, kullanıcıya kendi bucket'ını
mount ettirmek **doğru** karardır. LLM'in kod yazdığı bir üründe aynı şey,
denetlenmemiş koda depo anahtarı vermek olur.

Ayrıca mount etmenin somut bedeli var — araştırmanın kendi tespiti:
**mount eden hiçbir sistemde kayıt defteri kalmıyor.** Mount edersek elimizde
artifact store değil, "sadece bucket" kalır.

**Verdict: azınlıktayız ama doğru azınlıktayız.** Ve mount etmeyen iki sistem
tam olarak bizim yaptığımız işi yapıyor.

Bir de üçüncü yol var: **kimlik-bilgisiz mount + imzalayan proxy**
(Vercel/Cloudflare). Bu ilginç ve bizim modelimize en yakın "mount" varyantı —
mount var ama sandbox'ta anahtar yok. Değerlendirilebilir, ama kayıt defteri
sorununu yine çözmüyor.

## 3.3 Geride olduğumuz yerler — bunları saklamayalım

| Konu | Durum | SOTA ne diyor |
|---|---|---|
| **İzolasyon** | Normal container (runc) | Kata / gVisor. En zayıf ailedeyiz. |
| **Metadata DB** | SQLite | PostgreSQL. Tek replika sınırı buradan. |
| **Workflow state** | SQLite checkpointer (yeni) | PostgreSQL. Postgres yolu **kodda var ama test edilmedi.** |
| **Yüksek erişilebilirlik** | Tek replika | Çok replika |
| **Auth** | Yok — uuid bilen okur | Gerçek kimlik doğrulama |
| **Warm pool** | Yok | Bazı platformlarda var (ölçtük: kazancı ≤1.6 sn) |
| **Gerçek OBC/ODF** | Test edilmedi | OpenShift'te standart |
| **Kata** | Test edilmedi | — |

Bunların çoğu "PoC olduğu için" makul. Ama **ekibe sunarken "her şey hazır"
demek yanlış olur** — bu tablo aynen gösterilmeli.

## 3.4 Kimseye benzemediğimiz yer — ve bu bir risk

Bir şey var ki araştırmada **hiçbir sistemde bulamadım**: okuma tarafının
şeffaf olması.

Herkes yazma tarafını otomatikleştirmiş (`$OUTPUT_DIR` süpürme). Ama okuma
tarafında ya mount var (dosya zaten orada) ya da açık çağrı var
(`artifact.load("art_789")`). Paylaşılan mimari dokümanının §27'si de açık
çağrıyı öneriyor.

Bizde üçüncü bir şey var: mount yok, ama LLM `pd.read_csv("/output/x.csv")`
yazınca dosya arka planda iniyor. LLM artifact ID diye bir kavramla hiç
karşılaşmıyor.

**Bunu neden yaptık:** LLM'in ID taşıması, tam da unutabileceği türden bir yük.
Sıradan pandas kodu yazması hem daha doğal hem daha güvenilir.

**Ama dürüst olalım — bunun anlamı şu:** bu deseni sahada kimse zorlamamış.
Bilmediğimiz kenar durumları olabilir:

- pandas dışındaki okuyucular (`pyarrow.parquet.read_table`, `csv` modülü,
  `PIL.Image.open`) yakalanmıyor — sadece pandas ve doğrudan `open`
- isim çakışması: iki farklı workflow adımı aynı dosya adını kullanırsa
  "en yeni" kazanır, LLM bunun farkında olmayabilir
- büyük N'de manifest maliyeti ölçülmedi
- Türkçe/özel karakterli dosya adları süpürmede tireye çevriliyor, tembel
  okuma aynı dönüşümü uyguluyor ama bu eşleşme kırılgan

**Verdict: bu bizim özgün katkımız, ama "SOTA" değil — SOTA'nın ötesinde bir
deneme.** Ekibe böyle sunulmalı: *"herkesin yaptığı şeyi yaptık, artı okuma
tarafında bir iyileştirme denedik ve şu kenar durumları henüz bilmiyoruz."*

## 3.5 Tek cümlelik cevap

> **Omurga SOTA: efemer sandbox, kalıcı dış depo, kayıt defteri ayrımı,
> süpürme konvansiyonu, Parquet/pickle-yasak — hepsi çoğunluğun yaptığı şey.**
>
> **Erişim modelinde bilinçli azınlıktayız: mount etmiyoruz. Ama bizim işimize
> en çok benzeyen iki sistem (Anthropic, OpenAI) de etmiyor, ve mount edenlerin
> hiçbirinde kayıt defteri kalmıyor.**
>
> **Bazı yerlerde SOTA'nın gerisindeyiz (Kata yok, Postgres yok, tek replika,
> auth yok) — bunlar PoC olmanın bedeli, gizlenmemeli.**
>
> **Bir yerde de SOTA'nın ötesinde bir şey denedik (şeffaf okuma) — o yüzden
> orada emsal yok, kenar durumları da yok değil.**

---

# EK — Ekibe sunarken kullanılacak üç slaytlık özet

**Slayt 1 — Problem**
Sandbox'ın ölmesi güvenlik için gerekli, ürettiğinin kalması iş için gerekli.
Çözüm: sandbox'ın ömrü artifact'in ömrünü belirlemesin.

**Slayt 2 — Bizim yolumuz**
Efemer pod → akışlı HTTP → Artifact Service (4 kontrol) → MinIO + kayıt defteri.
Sandbox'ın deposu görmesi yok. Yazma iki yolla (açık API + `/output` süpürme),
okuma şeffaf (tembel doldurma). Kapsam her çalıştırma için ayrı imzalı jeton.

**Slayt 3 — Nerede duruyoruz**
Omurga SOTA ile aynı · Erişim modelinde Anthropic/OpenAI ile aynı safta ·
Kata/Postgres/HA/auth eksik (PoC) · Şeffaf okuma özgün ve emsalsiz.
