# Artifact Persistence — Dış Kaynak Araştırması

Çok adımlı workflow'larda üretilen **dosya, dataframe ve ara çıktıların** kalıcı
olarak saklanması; sonraki adımların aynı artifact'i yeniden üretmeden kullanabilmesi.

Bu doküman **dış kaynaklara** dayanır: iş akışı orkestratörleri (Airflow, Dagster,
Prefect, Argo, Flyte, Metaflow, Kubeflow), agent sandbox'ları (Claude code execution,
E2B, CodeAct), serileştirme ve depolama literatürü, ve güvenlik araştırmaları.
Kod tarafındaki mevcut durum tespiti ayrı bir dokümanda:
[PTC_Error_Recovery_ve_Artifact_Persistence_Arastirmasi.md](PTC_Error_Recovery_ve_Artifact_Persistence_Arastirmasi.md) §2.

Kaynak listesi §10'da.

---

## 1. Problem neden var

Çok adımlı bir workflow şuna benzer:

```
Adım 1  → tickets_df = list_open_tickets()      (~3 sn, 40 tool çağrısı)
Adım 2  → enriched   = join(tickets_df, dir_df)  (~1 sn)
Adım 3  → summary    = groupby(...)              (~0.2 sn)
Adım 4  → rapor üret                             ← burada NameError
```

Kalıcılık yoksa 4. adımdaki tek bir yazım hatası **1-3. adımların tamamını**
yeniden koşturmayı gerektirir. Maliyet üç yerde birden çıkar:

1. **Hesaplama**: pod ayağa kalkma + çalışma süresi baştan ödenir.
2. **Tool çağrısı**: kaynak sistemlere aynı sorgular tekrar gider.
3. **Bağlam**: ara sonucu LLM bağlamında taşımak tek gerçekçi alternatif — ama
   bir dataframe bağlama sığmaz.

Bu yüzden literatürde artifact persistence neredeyse her zaman **retry / caching /
self-repair** ile birlikte tartışılır: Prefect'te "caching depends on persisted
results — kalıcılığı kapatmak caching'i de kapatır" ifadesi bunun en net hâli
([Prefect Results](https://docs.prefect.io/v3/advanced/results)).

---

## 2. Endüstride üç mimari desen

Bütün sistemler üç şablondan birine (ya da karışımına) düşüyor.

### 2.1 Desen A — Değeri bağlam/kontrol düzleminde taşı (in-band)

Ara değer, orkestratörün kendi durum deposuna (metadata DB, agent state) yazılır.

- **Airflow XCom (varsayılan)**: değerler metadata veritabanında tutulur. Yerel
  geliştirme için yeterli, performansı sınırlı.
- **LangGraph state / checkpointer**: her düğüm sonrası tüm graph state'i
  checkpoint'lenir.

**Sınır neti**: LangGraph dokümanı açıkça uyarıyor — *"büyük dosyaları state
içinde saklamak önerilmez, çünkü her adımda yeni bir kopya kaydedilir; dosyayı
harici depoya koyup state'e yalnızca URL'i yazın"*
([LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)).

Bu desen **dataframe için elenir**. Ama küçük skalerler ve referanslar için
doğru yerdir.

### 2.2 Desen B — Referansı taşı, veriyi nesne deposunda tut (by-reference)

Baskın desen. Ara değer bir nesne deposuna (S3/MinIO/GCS) yazılır, kontrol
düzleminde yalnızca **URI + metadata** dolaşır.

| Sistem | Mekanizma | Ayırt edici detay |
|---|---|---|
| **Airflow** | `XComObjectStorageBackend` | `xcom_objectstorage_threshold` **eşiği**: bu bayttan küçük değerler DB'de, büyükler nesne deposunda; DB'de yalnızca referans kalır ([doc](https://airflow.apache.org/docs/apache-airflow-providers-common-io/stable/xcom_backend.html)) |
| **Dagster** | IO Manager | Okuma/yazma **step kodundan tamamen ayrılmış**; `FilesystemIOManager`, `S3PickleIOManager`, `BigQueryPandasIOManager`… aynı asset kodu farklı backend'lerle çalışır |
| **Prefect** | Result storage | `result_storage` bloğu; varsayılan `~/.prefect/storage`, tipik prod S3. **Varsayılan olarak kalıcı değil** — bilinçli açılır |
| **Argo Workflows** | Artifact repository | `outputs.artifacts` / `inputs.artifacts`; S3-uyumlu depo (MinIO dahil); varsayılan **tar+gzip** paketleme ([doc](https://argo-workflows.readthedocs.io/en/latest/walk-through/artifacts/)) |
| **Flyte** | Offloaded tipler | `FlyteFile`, `FlyteDirectory`, `StructuredDataset` — **tip sisteminin parçası**; veri `raw output prefix`'e yazılır, literal yalnızca işaretçi taşır |
| **Kubeflow Pipelines** | `dsl.Artifact` + MLMD | Artifact'in `.uri` alanı var; launcher nesne deposuna yükler ve URI'yi **ML Metadata**'ya kaydeder → lineage grafiği |
| **Metaflow** | Content-addressed datastore | Kod ve veri **içeriğinin hash'i** ile adreslenir (git gibi); aynı içerik otomatik dedup edilir ([datastore.md](https://github.com/Netflix/metaflow/blob/master/docs/datastore.md)) |

**Ortak yapı**: veri düzlemi (nesne deposu) ile kontrol düzlemi (DB/metadata)
ayrılmış; adımlar arasında geçen şey **veri değil, ada**.

### 2.3 Desen C — Çalışma ortamını canlı tut (stateful runtime)

Artifact'i dışarı yazmak yerine **süreci öldürmemek**.

- **CodeAct / Jupyter kernel modeli**: yorumlayıcı turlar arasında yeniden
  başlatılmaz; değişkenler bir sonraki hücrede hâlâ bellektedir. Ölçülen etki:
  çok turlu ek yükte %30'a varan azalma
  ([CodeAct, ICML 2024](https://github.com/xingyaoww/code-act)).
- **E2B sandbox persistence**: pause/resume ile **dosya sistemi + bellek** (çalışan
  süreçler, yüklü değişkenler dahil) snapshot'lanır. Pause ~4 sn/GiB RAM, resume
  ~1 sn. `keep_memory=False` ile yalnızca disk saklanır
  ([E2B](https://e2b.dev/docs/sandbox/persistence)).
- **Claude code execution tool** — bu PTC için en yakın referans, §7.2'de ayrı.

**Ana zaaf**: canlı ortam = kalıcı saldırı yüzeyi. Bir sandbox'ın ömrünü uzatmak,
onu izole tutmanın bütün maliyetini de uzatır.

### 2.4 Karşılaştırma

| | A: bağlamda taşı | B: nesne deposu + referans | C: canlı ortam |
|---|---|---|---|
| Dataframe uygun mu | Hayır | **Evet** | Evet |
| Adımlar arası yeniden kullanım | Sınırlı | Evet | Evet (aynı oturum) |
| Farklı oturum/turlar arası | Hayır | **Evet** | Kısmen (snapshot ile) |
| Yeni kalıcı depo gerekir mi | Hayır | Evet | Hayır (ama uzun ömürlü pod) |
| İzolasyon modeline etkisi | Yok | Orta (§6) | **Yüksek** |
| Denetlenebilirlik | Yüksek | Yüksek (URI + metadata) | Düşük (durum opak) |

---

## 3. Desen B'nin çalışan detayları

### 3.1 Eşik kuralı (threshold)

Airflow'un çözümü tek satırda özetlenebilir: **küçükse DB'de, büyükse depoda.**
`xcom_objectstorage_threshold` bayt cinsinden bir eşik; altındakiler metadata
veritabanında kalır, üstündekiler nesne deposuna gider ve DB'de yalnızca referans
saklanır.

Bunun pratik faydası: her küçük skaler için nesne deposuna round-trip yapılmaz.
Astronomer'ın önerisi de aynı yönde — *"XCom varsayılan değerlerini küçük tutun,
seçili büyük yükler için nesne deposu kullanın; DataFrame'in kendisini değil
referansını saklayın"*
([Astronomer](https://www.astronomer.io/docs/learn/custom-xcom-backend-strategies)).

### 3.2 Okuma/yazmayı iş kodundan ayır

Dagster'ın IO Manager'ı ve Flyte'ın offloaded tipleri aynı fikri paylaşıyor:
**adım kodu `to_parquet`/`read_parquet` çağırmaz.** Adım bir değer döndürür;
nereye, hangi formatta yazılacağına runtime karar verir.

Kazanç iki yönlü:
- Aynı kod yerelde dosya sistemine, prod'da S3'e yazar — kod değişmez.
- Depolama yolu **tek bir yerde** politika uygulanabilir hâle gelir (şifreleme,
  prefix, TTL, erişim kontrolü). Sandbox içindeki LLM kodu depolama yoluna hiç
  dokunmaz.

Bu ikinci madde güvenlik açısından belirleyici (§6.4).

### 3.3 Content addressing → adım atlama (caching)

Kalıcılığın asıl getirisi burada. Artifact'i **içeriğinin hash'i** ile adreslemek
iki şeyi birden verir: dedup ve memoization.

| Sistem | Cache anahtarı | Sonuç |
|---|---|---|
| **Metaflow** | İçerik hash'i (git benzeri) | Aynı bayt iki kez saklanmaz; `resume` hızlanır |
| **Snakemake** | Adımlar, parametreler, yazılım yığını ve girdilerden **Merkle ağacı** | Workflow'lar *arası* cache ([doc](https://snakemake.readthedocs.io/en/stable/executing/caching.html)) |
| **Nextflow** | Task hash → work dizini | `-resume`: hash tutuyorsa görev hiç çalışmaz |
| **Prefect** | Hesaplanan "cache key" → result storage'daki yol | Anahtar tutarsa task **çalışmaz**, `Cached` state'e geçer |
| **Flyte** | Data Catalog; `StructuredDataset` için **dataframe içeriğinin hash'i** (depolama konumu değil) | Aynı veri farklı konumda olsa da cache tutar |

Flyte'ın ayrımı ince ama önemli: konumu değil **içeriği** hash'lemek, aynı verinin
farklı yollardan gelmesi durumunda bile cache isabetini korur.

Bu mekanizma §1'deki senaryonun doğrudan cevabı: 4. adım düzeltilip yeniden
çalıştığında 1-3. adımların hash'i değişmediği için tekrar koşmaz.

### 3.4 Lineage — artifact'in yanında ne saklanır

Kubeflow'un ayrımı temiz: **artifact store** (veri) + **metadata store** (MLMD).
Launcher artifact'i nesne deposuna yükler, URI'yi MLMD'ye yazar; sonuç, hangi
execution'ın hangi artifact'i ürettiğini/tükettiğini gösteren bir **lineage
grafiği**dir. Aynı artifact birden çok çalıştırmada kullanılıyorsa bu grafikte
görünür.

Minimum metadata seti (kaynaklardan ortak çıkan):
`artifact_id`, `uri`, `üreten run_id`, `content_hash`, `şema/tip`, `boyut`,
`oluşturma zamanı`, `TTL/son kullanma`.

---

## 4. Serileştirme — dataframe'i nasıl yazmalı

### 4.1 Format karşılaştırması

| Format | Tip korunumu | Boyut | Hız | Güvenlik | Not |
|---|---|---|---|---|---|
| **Parquet** | Yüksek (şemalı, sütunlu) | **En küçük** | İyi | Güvenli | Gelişmiş sıkıştırma/kodlama; disk ve ağ yavaşsa kısa ömürlü cache için bile tercih edilir |
| **Arrow IPC / Feather V2** | **Tam** (bütün Arrow tipleri) | Orta | **En hızlı** | Güvenli | Feather V2 = Arrow IPC dosya formatının kendisi; zero-copy okuma |
| **CSV** | Yok | Büyük | Yavaş | Güvenli | Tip bilgisi kaybolur — okuma tarafında yeniden tahmin |
| **JSON** | Kısmi | Büyük | Yavaş | Güvenli | Dataframe için uygun değil; metadata için doğru |
| **pickle** | Tam | Orta | Hızlı | **Kullanılmamalı** | §4.2 |
| **safetensors** | Tensör | Küçük | Hızlı | Güvenli | Tensör için pickle'ın yerine geçen format; metadata JSON |

Pratik seçim: **arşiv/uzun ömür → Parquet**, **adımlar arası hızlı devir →
Arrow IPC**. İkisi de tip korur, ikisi de kod çalıştırmaz.

### 4.2 pickle neden eleniyor

pickle'ın deserialization'ı **tasarım gereği kod çalıştırır** — CWE-502
(Deserialization of Untrusted Data). Gerçek örnekler:

- PyTorch `torch.load()` içeride pickle kullanır; `weights_only=True` olmadan
  yüklenen dosya rastgele kod çalıştırabilir
  ([GHSA-9pf3-7rrr-x5jh](https://github.com/InternLM/lmdeploy/security/advisories/GHSA-9pf3-7rrr-x5jh)).
- LeRobot'un async inference hattı kimlik doğrulamasız gRPC üzerinden gelen veriyi
  `pickle.loads()` ile açıyordu → RCE
  ([issue #3047](https://github.com/huggingface/lerobot/issues/3047)).

**PTC için bu teorik değil.** Artifact'i yazan taraf **LLM'in ürettiği koddur**.
Bir sonraki adımın o artifact'i pickle ile geri yüklemesi, "LLM'in ürettiği veriyi
kod olarak çalıştır" demektir. Sandbox içinde bile bu, izolasyonun kendisini
anlamsızlaştırır — çünkü artifact sandbox dışına da çıkabilir.

**Dikkat çeken bir tuzak:** Dagster'ın *varsayılan* IO manager'ı
(`FilesystemIOManager`) ve `S3PickleIOManager` çıktıları **pickle** olarak yazar.
Yani "hazır bir IO manager al, kullan" yolu doğrudan bu riskin içine giriyor.
Bir artifact katmanı kurulacaksa format **açıkça** seçilmeli, varsayılana
bırakılmamalı.

### 4.3 Kayıp yalnızca formatta değil

Mevcut PTC yolunda değer `json.dumps` → `str(...)` zincirinden geçiyor; tip
bilgisi zaten kayboluyor. Artifact katmanı eklenirken **tip/şemanın metadata'da
taşınması** ayrı bir iş kalemi — Flyte'ın `StructuredDataset`'i tam olarak bunu
yapıyor (dataframe + şema + format birlikte).

---

## 5. Yaşam döngüsü: GC, TTL, kota

Kalıcılık eklendiği anda **silme politikası da eklenmek zorunda**. Kaynaklardaki
ortak pratikler:

**Argo Workflows — `artifactGC`** (v3.4+), üç strateji:
- `OnWorkflowCompletion` — workflow biter bitmez sil
- `OnWorkflowDeletion` — workflow kaynağı silinince sil
- `Never` — sakla

Strateji hem `spec.artifactGC.strategy` (workflow düzeyi) hem de
`artifacts[].artifactGC.strategy` (artifact düzeyi) ile verilebilir. Yani **geçici
ara çıktılar hemen, nihai çıktı kalıcı** olabilir. Ayrıca S3 anahtarını
`{{workflow.uid}}` ile parametrelemek eşzamanlı çalıştırmaların çakışmasını
önlüyor — bu, çok-çalıştırmalı bir sistemde isim alanı ayrımının temel kuralı.

**Nesne deposu tarafı — S3 lifecycle:**
- Expiration kuralları ile otomatik silme; nesne etiketleme (tagging) ile birleştirilir.
- AWS'in özellikle vurguladığı nokta: **`AbortIncompleteMultipartUpload` kuralını
  her bucket'ta açın** — yarım kalan parçalar normal listelemede *görünmez* ama
  depolama ücreti işler. Yaygın ve sinsi bir maliyet kaynağı.

**Yetim artifact problemi gerçek:** MLflow'da metadata silinip artifact'lerin
blob depoda kalması bilinen bir sorun
([mlflow#12917](https://github.com/mlflow/mlflow/issues/12917)); `mlflow gc`
sürüme göre yalnızca metadata'yı ya da ikisini birden temizliyor. Ders: **metadata
kaydı ile bayt'ların silinmesi tek bir işlemde ele alınmazsa ayrışır.**

**Dedup:** DVC content-addressable cache ile aynı içeriği iki kez saklamaz;
lakeFS varsayılanında aynı dosyanın iki yüklemesi iki fiziksel nesne üretir.
Content addressing (§3.3) burada ikinci kez kazandırıyor.

---

## 6. Güvenlik — PTC modeli açısından kritik bölüm

Bu bölüm, artifact persistence'ın PTC'nin ağ-merkezli güvenlik modeliyle nerede
çakıştığını dış kaynaklarla temellendiriyor.

### 6.1 Paylaşılan depo, ağ politikasının göremediği bir kanaldır

NIST tanımı doğrudan bu duruma oturuyor:

> **Covert storage channel**: bir sistem varlığının, ikinci bir varlığın sonradan
> okuyacağı bir depolama konumuna yazarak bilgi iletmesini sağlayan sistem özelliği.
> ([NIST CSRC](https://csrc.nist.gov/glossary/term/covert_storage_channel))

İki sandbox çalıştırması birbirine **paket göndermez**; depo üzerinden haberleşir.
Cilium ağ katmanında çalıştığı için bu akışı ne görür ne engeller.

Literatür bunun teorik olmadığını gösteriyor: **Sync+Sync**, aynı depolama
aygıtını paylaşan **iki ayrı container** arasında `fsync` zamanlamaları üzerinden
kurulan güvenilir bir covert channel — ağ izolasyonu tam olsa bile
([arXiv:2309.07657](https://arxiv.org/pdf/2309.07657)). Benzer şekilde `syncfs`
üzerinden yan kanal saldırıları
([arXiv:2411.10883](https://arxiv.org/pdf/2411.10883)).

**Sonuç:** artifact deposu eklendiği anda erişim kontrolü **ayrı bir katmanda**
(depo tarafında) kurulmak zorunda. Bu bir tercih değil, yapısal zorunluluk.

### 6.2 Artifact'ler kalıcı prompt injection taşıyıcısıdır

Bu, kalıcılığın en az konuşulan ama en ciddi sonucu.

Normal prompt injection oturum bitince ölür. **Depoya yazılan** enjeksiyon ölmez:

> Bir saldırı, düşmanca içerik oturumlar arası kalıcı bir bağlama yazıldığında
> ya da onu değiştirdiğinde **stored** hâle gelir ve gelecekteki etkileşimlerde
> yeniden bağlama girer.

- Palo Alto Unit 42, dolaylı prompt injection'ın uzun-vadeli belleği
  zehirleyebildiğini gösteriyor
  ([Unit 42](https://unit42.paloaltonetworks.com/indirect-prompt-injection-poisons-ai-longterm-memory/)).
- **MemoryGraft**: tetikleyicisiz ve dolaylı — dokümantasyon veya repo notu gibi
  masum görünen artifact'ler, agent'ın "başarılı deneyim" olarak kaydedip sonra
  taklit ettiği kötücül kayıtlar üretmesine yol açıyor
  ([arXiv:2512.16962](https://arxiv.org/html/2512.16962v1)).
- Paylaşılan bilgi tabanlarında zehirlenme **saatler içinde yayılabiliyor**: bir
  agent kötücül içeriği yazınca, o depoya okuma erişimi olan her agent normal
  çalışması sırasında onu geri alıyor.

**PTC'ye çevirisi:** Çalıştırma A'nın ürettiği bir CSV'nin bir hücresinde talimat
metni olabilir. Çalıştırma B o dosyayı okuyup özetini LLM'e verdiğinde, metin
**model bağlamına girer**. Artifact deposu, ilk kez, tur sınırlarını aşan bir
veri yolu açar.

Karşı önlem yönleri (kaynaklarda ortak): artifact içeriğini **her zaman veri
olarak** işaretlemek (asla talimat değil), provenance'ı (hangi run üretti)
metadata'da tutmak, ve okuma tarafında bağlama giren özeti sınırlandırmak.

### 6.3 Presigned URL bir egress kaçış yoludur

"Sandbox'a doğrudan S3 erişimi vermeyelim, presigned URL verelim" refleksi
PTC'nin modelini **kırar**:

- Presigned URL **public bir kaynaktır ve kullanıcı doğrulaması yapmaz**; geçerli
  URL'e sahip olan herkes erişir.
- URL'i imzalayan IAM kimliğinin yetkisiyle çalışır — sızma faaliyeti **imzalayan
  principal'dan geliyormuş gibi** görünür.
- Egress'i kısıtlamak için konan aracılar tipik olarak URI'leri loglar; imza
  query parametresi içinde taşındığı için presigned URL'ler **egress allowlist'lerini
  atlatabilir** ([WithSecure Labs](https://labs.withsecure.com/publications/pre-signed-at-your-service)).
- Loglama varsayılan olarak açık olmadığından tespit de zorlaşır.

PTC'nin kurucu iddiası "sandbox yalnızca Tool Gateway'e çıkabilir". Sandbox'a
presigned URL vermek, ona **Tool Gateway'in görmediği ikinci bir çıkış** vermektir.

### 6.4 Doğru desen: opaque handle (capability)

Güvenlik literatürünün önerdiği desen, §3.2'deki mühendislik önerisiyle birebir
örtüşüyor:

> Handle katmanı **capability** tarzı bir tasarımı izler: yetki, ham veriye maruz
> kalmakla değil, **taklit edilemez (unforgeable) bir referansla** taşınır. Okuma
> yolunda gateway **opak bir handle** ve gerektiğinde **sınırlı bir özet** döndürür.

Capability modelinin iki kuralı burada doğrudan işe yarıyor:
- Yetki yalnızca **referansı elde tutmakla** kazanılır.
- Yetki **zayıflatılabilir (attenuable)** olmalı: handle sahibi daha zayıf bir
  handle türetebilmeli, asla daha güçlüsünü değil.

PTC'ye uygulaması: sandbox koduna `s3://bucket/path` verilmez; `artifact_id`
(ör. `art_7f3a…`) verilir. Bu kimlik tek başına yetki taşımaz — Tool Gateway,
çağıran çalıştırmanın o artifact'i okumaya hakkı olup olmadığına kendisi karar
verir. Sandbox depolama topolojisini **hiç görmez**.

### 6.5 Çok-çalıştırma izolasyonu depo tarafında kurulur

Nesne deposu seçilirse standart mekanizma hazır:

- **MinIO STS `AssumeRole`**: uzun ömürlü anahtar dağıtmadan, **politikayla
  daraltılmış** geçici kimlik bilgisi üretir (`WithPolicy` ile scope-down).
- **Prefix izolasyonu**: `arn:aws:s3:::shared-bucket/tenant-alpha/*` gibi kaynak
  kalıpları ve `ListBucket` için prefix koşullarıyla, paylaşılan bucket içinde
  kiracılar ayrılır.
- **`AssumeRoleWithWebIdentity` + Kubernetes ServiceAccount JWT**: MinIO Operator,
  PolicyBinding CRD'si ile JWT'yi doğrulayıp politikayı bağlar — kimlik bilgisi
  hiç elle taşınmaz.

PTC'de doğal prefix `run_id` (ya da oturum/konuşma kimliği). Argo'nun
`{{workflow.uid}}` ile anahtar parametreleme pratiği aynı fikir.

### 6.6 Özet: hangi kontrol nerede

| Risk | Cilium engelleyebilir mi | Nerede karşılanır |
|---|---|---|
| Sandbox'ın onaysız hedefe çıkması | **Evet** (mevcut) | Egress policy |
| Çalıştırma A → B veri sızdırması (depo üzerinden) | Hayır (§6.1) | Gateway'de yetkilendirme + prefix politikası |
| Artifact'e gömülü prompt injection | Hayır | İçerik işleme politikası (§6.2) |
| Presigned URL ile egress atlatma | Kısmen/hayır (§6.3) | Sandbox'a URL hiç vermeyerek |
| Deserialization ile kod çalıştırma | Hayır | Format yasağı (§4.2) |
| Depo şişmesi / maliyet | Hayır | TTL + lifecycle + kota (§5) |

---

## 7. Sandbox tarafında yazılabilir alan

### 7.1 Kubernetes mekanikleri

Mevcut pod'da yazılabilir alan yok (tek volume salt-okunur ConfigMap; ConfigMap
başına **1 MiB** sınırı da var). Seçenekler:

| Mekanizma | Ömür | Not |
|---|---|---|
| **`emptyDir`** | Pod ömrü | En basit. `sizeLimit` konabilir; **aşılırsa pod tahliye edilir** (evict). Node ephemeral storage'dan ayrılır |
| **Generic ephemeral volume** | Pod ömrü | PVC arayüzü, pod ile birlikte silinir. `emptyDir` gibi world-writable mount edilmez |
| **PVC** | Kalıcı | Gerçek kalıcılık; ama GC, kota ve çok-çalıştırma erişim kontrolü tamamen elde kalır |

`readOnlyRootFilesystem: true` ile birlikte `emptyDir`'i yazılabilir scratch
olarak mount etmek yerleşik desen — kök dosya sistemi salt-okunur kalırken kodun
geçici alanı olur.

**Önemli ayrım:** `emptyDir` **kalıcılık sağlamaz**; yalnızca "kod bir dataframe'i
diske yazıp Parquet'e çevirebilsin" sorununu çözer. Kalıcılık ayrı katmandır.

### 7.2 Claude code execution tool — en yakın referans mimari

Bu, PTC'nin problemine mimari olarak en çok benzeyen üretim sistemi: **internet
erişimi tamamen kapalı** bir sandbox'ta artifact üretmek ve dışarı vermek.
([Code execution tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool))

Belirleyici tasarım kararları:

- **Ağ**: *"Internet access: Completely disabled for security. No outbound network
  requests permitted."* Container çalışma anında paket bile indiremez — yalnızca
  önyüklü kütüphaneler (pandas, numpy, pyarrow, matplotlib…) vardır.
- **Kaynaklar**: 5 GiB RAM, 5 GiB workspace disk, 1 CPU.
- **Kalıcılık container kimliğiyle**: yanıt gövdesinde `container.id` döner; bunu
  bir sonraki istekte geri göndermek **aynı container'ı** (ve dosyalarını)
  kullandırır. Container'lar **oluşturulduktan 30 gün sonra** sona erer; ~5 dakika
  hareketsizlikten sonra **checkpoint'lenir** ve 30 gün içinde ID ile geri yüklenir.
- **REPL durumu**: `code_execution_20260120` ve sonrasında (programmatic tool
  calling ile birlikte) **Python yorumlayıcı durumu da** (değişken bağlamaları)
  container yeniden kullanıldığında korunur — yani Desen C.
- **Artifact çıkışı `$OUTPUT_DIR` sözleşmesiyle**: her bash çağrısı boş bir dizin
  alır; komut bitince o dizinin **en üst düzeyindeki** dosyalar yakalanır ve
  sonuçta `file_id` olarak döner. **Başka yere yazılan dosyalar container'da kalır,
  dönmez.** Dosyalar sonra Files API üzerinden indirilir.
- **Kapsam**: container'lar isteğin workspace'ine scope'lanır.

**PTC için üç ders:**
1. Sıfır-egress sandbox ile kalıcı artifact **çelişmiyor** — artifact çıkışı ağ
   üzerinden değil, **runtime sözleşmesi** üzerinden yapılıyor.
2. "Neyin dışarı çıkacağı" **açık ve dar** bir sözleşmeyle belirleniyor
   (`$OUTPUT_DIR`'in tepesi). Kod her yere yazabilir; ama yalnızca bir yer sayılır.
3. Kalıcılık **opt-in ve kimliğe bağlı**: container ID geri gönderilmezse her
   istek temiz bir ortamda başlar. Varsayılan hâlâ "iz bırakma".

---

## 8. PTC'ye uyarlama

### 8.1 Seçeneklerin dış kaynaklarla yeniden değerlendirilmesi

Ön inceleme dokümanı (§2.4) dört yön saymıştı. Araştırma bunları şöyle sıralıyor:

| Yaklaşım | Endüstri karşılığı | Değerlendirme |
|---|---|---|
| **Tool Gateway arkasında nesne deposu + opaque handle** | Argo/KFP artifact repository + capability handle deseni (§6.4) | **En tutarlı.** Cilium politikası değişmez (sandbox yine yalnız gateway'e çıkar); erişim kontrolü zaten allowlist'in olduğu yerde; STS/prefix ile çok-çalıştırma izolasyonu hazır |
| **Pod'a PVC mount** | — | Basit ama §6.1'deki kanalı **denetimsiz** açar; GC/kota elde kalır; çalıştırmalar arası yetkilendirme yok |
| **Uzun ömürlü sandbox / snapshot** | E2B pause-resume, Claude container reuse | Güçlü ama izolasyon maliyetini süreklileştirir; "her run sıfırdan" iddiası tümden düşer |
| **Agent belleğinde taşıma** | LangGraph state | Dataframe için gerçekçi değil — LangGraph'ın kendi dokümanı bunu önermiyor |

### 8.2 Önerilen mimari (dış kaynaklardan türetilmiş)

```
┌────────────── Sandbox Pod (egress: yalnız Tool Gateway) ──────────────┐
│  LLM kodu                                                             │
│    df = ...                                                           │
│    h = put_artifact(df)        →  handle: "art_7f3a…"                 │
│    df2 = get_artifact("art_1c9…")                                     │
│  /scratch  (emptyDir, sizeLimit, readOnlyRootFilesystem yanında)      │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ MCP (mevcut tek izinli hedef)
                    ┌───────────▼───────────┐
                    │    Tool Gateway       │  ← yetkilendirme burada
                    │  put/get_artifact     │    (run_id → prefix)
                    │  handle ↔ URI eşlemesi│    provenance, TTL
                    └───────────┬───────────┘
                                │  S3 API (gateway'in kimliğiyle)
                        ┌───────▼────────┐
                        │ MinIO / S3     │  prefix: artifacts/{session}/{run_id}/
                        │ lifecycle: TTL │
                        └────────────────┘
```

Bu tasarımın her parçasının bir kaynağı var:

| Parça | Dayanak |
|---|---|
| Sandbox'ın depoya doğrudan erişmemesi | §6.3 (presigned URL egress atlatma), §6.4 (capability) |
| Opak `artifact_id`, URI değil | §6.4 unforgeable reference |
| Yetkilendirmenin gateway'de olması | §6.1 (Cilium bu akışı göremez) |
| `run_id`/oturum prefix'i | §6.5 MinIO prefix izolasyonu, Argo `{{workflow.uid}}` |
| Parquet/Arrow, pickle yasak | §4.1, §4.2 |
| `content_hash` metadata | §3.3 dedup + cache |
| TTL + lifecycle kuralı | §5 Argo `artifactGC`, S3 expiration |
| `emptyDir` scratch | §7.1 |
| Dar çıkış sözleşmesi | §7.2 `$OUTPUT_DIR` deseni |

### 8.3 Minimum uygulanabilir kapsam

Mevcut koda göre değişecek yerler:

1. **`sandbox_image/entrypoint.py`** — `set_result` yanına iki proxy fonksiyon:
   `put_artifact(value, name=None)` ve `get_artifact(handle)`. Mevcut
   `_make_sync_tool` deseni aynen kullanılabilir; `_ARG_NAMES`'e iki giriş eklenir.
2. **`mock_services/tool_gateway/server.py`** — iki yeni MCP tool. Serileştirme
   burada **değil**, sandbox tarafında yapılmalı (dataframe gateway'e ham gitmez);
   gateway bayt + metadata alır, prefix'e yazar, handle döner.
3. **`ALLOWED_TOOLS`** — hem entrypoint'te hem `agent/tool_policy.py`'de (ikisi
   birebir aynı olmak zorunda).
4. **`k8s/sandbox/job-template.yaml`** — `emptyDir` scratch volume + `sizeLimit`;
   `activeDeadlineSeconds: 30` artifact yazma süresini de kapsayacak şekilde
   gözden geçirilmeli.
5. **`models.py` / `trace.py`** — artifact üretimi/tüketimi iz kaydına girmeli
   (KFP'nin MLMD'de yaptığının küçük hâli); handle'lar `Trace`'te görünmeli.
6. **Lifecycle** — MinIO bucket'ında expiration kuralı + `AbortIncompleteMultipartUpload`.

### 8.4 Doküman tutarlılığı

Ön inceleme §2.2'deki gerilim dış kaynaklarla da doğrulanıyor: "hiçbir zaman
kalıcı iz bırakmaz" iddiası nitelenmek zorunda. Claude code execution tool'un
formülasyonu iyi bir şablon: **çalıştırma ortamı varsayılan olarak izsizdir;
kalıcılık opt-in'dir ve açık bir kimliğe (container ID / artifact handle) bağlıdır.**

---

## 9. Açık sorular — araştırmanın getirdiği cevap adayları

| Ön incelemedeki soru | Araştırmanın işaret ettiği yön |
|---|---|
| Depo hangi katmanda? | Tool Gateway arkasında nesne deposu (§8.1) — PVC değil |
| Erişim kontrolü kim yapacak? | Gateway; depo tarafında STS + prefix politikası ikinci katman (§6.5) |
| Artifact ömrü / GC? | Artifact düzeyinde strateji (Argo modeli): ara çıktı kısa TTL, nihai çıktı uzun; + S3 lifecycle (§5) |
| Serileştirme formatı? | Parquet (arşiv) / Arrow IPC (devir); pickle kesin dışarıda (§4) |
| "İz bırakmaz" iddiası? | Nitelenmeli: varsayılan izsiz, kalıcılık opt-in (§8.4) |
| **Yeni soru:** artifact'e gömülü talimat metni? | Kaynaklarda açık cevabı olmayan, PTC'ye özgü karar noktası (§6.2) |
| **Yeni soru:** cache/adım atlama bu fazda mı? | Content hash metadata'ya şimdi konursa sonra ücretsiz gelir (§3.3) |

---

## 10. Kaynaklar

**Orkestratörler**
- [Airflow — Object Storage XCom Backend](https://airflow.apache.org/docs/apache-airflow-providers-common-io/stable/xcom_backend.html)
- [Airflow — XComs](https://airflow.apache.org/docs/apache-airflow/stable/concepts/xcoms.html)
- [Astronomer — Strategies for custom XCom backends](https://www.astronomer.io/docs/learn/custom-xcom-backend-strategies)
- [Prefect — How to persist and retrieve workflow results](https://docs.prefect.io/v3/advanced/results)
- [Prefect — Configure task caching](https://docs-3.prefect.io/v3/develop/task-caching)
- [Dagster — IO managers rehberi](https://www.getorchestra.io/guides/dagster-tutorials-a-comprehensive-guide-to-io-managers)
- [Argo Workflows — Artifacts walkthrough](https://argo-workflows.readthedocs.io/en/latest/walk-through/artifacts/)
- [Argo Workflows — Configuring your artifact repository](https://argo-workflows.readthedocs.io/en/latest/configure-artifact-repository/)
- [Pipekit — MinIO artifact repository for Argo Workflows](https://pipekit.io/blog/how-to-set-up-a-minio-artifact-repository-for-argo-workflows)
- [Flyte — Understand How Flyte Handles Data](https://docs-legacy.flyte.org/en/v1.13.1/concepts/data_management.html)
- [Flyte — StructuredDataset](https://docs-legacy.flyte.org/en/v1.12.0/user_guide/data_types_and_io/structureddataset.html)
- [Flyte — Data Catalog / caching](https://docs-legacy.flyte.org/en/v1.12.0/concepts/catalog.html)
- [Metaflow — datastore.md](https://github.com/Netflix/metaflow/blob/master/docs/datastore.md)
- [Metaflow — Checkpointing progress](https://docs.metaflow.org/scaling/checkpoint/introduction)
- [Kubeflow — Create, use, pass, and track ML artifacts](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/)
- [Kubeflow — ML Metadata](https://www.kubeflow.org/docs/components/pipelines/concepts/metadata/)
- [Snakemake — Between workflow caching](https://snakemake.readthedocs.io/en/stable/executing/caching.html)
- [Nextflow — Task processing and execution (DeepWiki)](https://deepwiki.com/nextflow-io/nextflow/3.3-task-processing-and-execution)

**Agent sandbox / runtime**
- [Anthropic — Code execution tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)
- [Anthropic — Files API](https://platform.claude.com/docs/en/build-with-claude/files)
- [E2B — Sandbox persistence](https://e2b.dev/docs/sandbox/persistence)
- [CodeAct — Executable Code Actions Elicit Better LLM Agents (ICML 2024)](https://github.com/xingyaoww/code-act)
- [LangGraph — Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)

**Serileştirme**
- [Apache Arrow — FAQ (Feather V2 = Arrow IPC)](https://arrow.apache.org/faq/)
- [Saving Pandas DataFrames: Parquet vs Feather vs ORC vs CSV](https://towardsdatascience.com/saving-pandas-dataframes-efficiently-and-quickly-parquet-vs-feather-vs-orc-vs-csv-26051cc98f2e/)
- [Faster DataFrame Serialization](https://towardsdatascience.com/faster-dataframe-serialization-75205b6b7c69/)
- [LMDeploy — RCE via insecure deserialization in torch.load()](https://github.com/InternLM/lmdeploy/security/advisories/GHSA-9pf3-7rrr-x5jh)
- [LeRobot — Unsafe pickle deserialization (CWE-502)](https://github.com/huggingface/lerobot/issues/3047)
- [Invicti — Python pickle serialization](https://www.invicti.com/web-application-vulnerabilities/python-pickle-serialization)

**Depolama, yaşam döngüsü, güvenlik**
- [MinIO — AssumeRole (STS)](https://docs.min.io/aistor/developers/security-token-service/assumerole/)
- [MinIO — Multi-tenancy](https://docs.min.io/aistor/administration/multi-tenancy/)
- [MinIO — STS for MinIO Operator](https://minio.community/community/minio-object-store/developers/sts-for-operator.html)
- [WithSecure Labs — Pre-signed at your service](https://labs.withsecure.com/publications/pre-signed-at-your-service)
- [AWS — S3 presigned URL best practices (PDF)](https://docs.aws.amazon.com/pdfs/prescriptive-guidance/latest/presigned-url-best-practices/presigned-url-best-practices.pdf)
- [MLflow — Deleting orphaned artifacts (#12917)](https://github.com/mlflow/mlflow/issues/12917)
- [Kubernetes — Volumes](https://kubernetes.io/docs/concepts/storage/volumes/)
- [DVC vs Git-LFS vs lakeFS — data versioning](https://reintech.io/blog/dvc-vs-git-lfs-vs-lakefs-ml-data-versioning)

**Saldırı literatürü**
- [NIST CSRC — Covert storage channel](https://csrc.nist.gov/glossary/term/covert_storage_channel)
- [Sync+Sync: A Covert Channel Built on fsync with Storage (arXiv:2309.07657)](https://arxiv.org/pdf/2309.07657)
- [I Know What You Sync: Covert and Side Channel Attacks via syncfs (arXiv:2411.10883)](https://arxiv.org/pdf/2411.10883)
- [Unit 42 — Indirect prompt injection poisons AI long-term memory](https://unit42.paloaltonetworks.com/indirect-prompt-injection-poisons-ai-longterm-memory/)
- [MemoryGraft: Persistent Compromise of LLM Agents via Poisoned Experience Retrieval (arXiv:2512.16962)](https://arxiv.org/html/2512.16962v1)
- [Exploring Cross-Session Stored Prompt Injection in Agentic Systems (arXiv:2606.04425)](https://arxiv.org/pdf/2606.04425)
- [Forcepoint — Persistent memory poisoning in AI agents](https://www.forcepoint.com/blog/x-labs/persistent-memory-poisoning-ai-agents)
- [Tracking Capabilities for Safer Agents (arXiv:2603.00991)](https://arxiv.org/pdf/2603.00991)
- [Awesome Object Capabilities and Capability-based Security](https://github.com/dckc/awesome-ocap)
