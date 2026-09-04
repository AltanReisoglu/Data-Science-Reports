# PTC Artifact Persistence — PoC Planı

**Tarih:** 2026-09-03 · **Hedef ortam:** OpenShift

Birinci bölüm, bu araştırma boyunca gördüğümüz farklı "sandbox/artifact zihniyetlerinin"
kısa haritası. İkinci bölüm, bunlardan çıkardığımız PoC planı.

Kararların gerekçeleri: [PTC_Hedef_Mimari_ve_Karar_Dokumani.md](PTC_Hedef_Mimari_ve_Karar_Dokumani.md)

---

# BÖLÜM 1 — Gördüğümüz zihniyetler

## 1.1 Beş ayrı yaklaşım

| # | Zihniyet | "Veri nerede yaşar" | Örnekler |
|---|---|---|---|
| 1 | **Ortamı dondur** | Sandbox'ın kendi diski/belleği; ortam canlı tutulur | Anthropic container reuse (30 gün, `container.id`), E2B pause/resume (FS+bellek), Modal snapshots (FS 30 gün / bellek 7 gün), Vercel (stop'ta otomatik snapshot), Daytona |
| 2 | **Ortamın dışına yaz** | Ayrı bir depo; ortam ölse de veri kalır | Anthropic Files API (`$OUTPUT_DIR` → `file_id`), Modal Volumes, E2B Volumes, Vercel Drives, S3/R2 FUSE mount |
| 3 | **Durumu tamamen dışarı koy** | Sandbox'ta hiçbir kalıcılık yok, bilinçli olarak | Cloudflare Code Mode (durum Durable Object SQLite'ta; *"executor and connector instances remain transient"*), Letta `run_code` (*"fresh environment per execution"*) |
| 4 | **Orkestratör deseni** | Nesne deposu + kontrol düzleminde referans | Airflow (`XComObjectStorageBackend` + eşik), Dagster (IO Manager), Prefect (result storage + cache key), Argo (artifact repository + `artifactGC`), Flyte (`StructuredDataset`), KFP (`dsl.Artifact` + MLMD), Metaflow (**içerik hash'i**) |
| 5 | **Değeri bağlamda taşı** | Kontrol düzleminde, serileştirilmiş küçük değer | Airflow varsayılan XCom, LangGraph state — **dataframe için ölü** (LangGraph kendi dokümanında "büyük dosyayı state'e koyma, URL sakla" diyor) |

## 1.2 Hepsini ayıran tek eksen

SOTA'nın hiçbiri şu ikisini aynı soru saymıyor:

```
ÇALIŞTIRMA ORTAMI ömrü     ≠     VERİ ömrü
(ms … saatler)                   (dakikalar … silinene kadar)
```

En net kanıt Anthropic'in kendi sistemi: **container 30 gün**, ama **Files API
dosyaları silinene kadar**. Modal'da sandbox 5 dk–24 saat, Volumes bambaşka bir
katman. Bizim PoC'de bu ikisi yapışıktı ("pod ölünce her şey gidiyor") — bu bir
zorunluluk değil, tasarım artığıydı.

## 1.3 Artifact persistence, sandbox persistence'tan güçlüdür

```
Sandbox persistence:   PTC#1 → Sandbox → dosya durur → PTC#2
                       ⚠ sandbox yaşamak ZORUNDA

Artifact persistence:  PTC#1 → Sandbox → save() → Depo
                       Sandbox silinir ✗
                       PTC#2 → yeni Sandbox → load(art_id)
                       ✓ ortam ölse de veri duruyor
```

Bu yüzden **zihniyet 2/3/4 ailesini seçiyoruz**, 1'i değil.

## 1.4 "Çok adımlı workflow" üç farklı şey demek

Bizim mimaride "adım" soyutlaması yok; olan şey `run_ptc_code(code)` çağrısı —
bir Kubernetes Job, bir Python süreci.

| Okuma | Ne | Kalıcılık gerekli mi | Çözüm |
|---|---|---|---|
| **A** | Tek script içindeki aşamalar | Aktarım için **hayır** (aynı bellek). Ama script patlarsa hepsi baştan koşar | İçerik-adresli **cache** |
| **B** | Aynı turda ardıl çalıştırmalar (`MAX_SANDBOX_RUNS_PER_TURN = 2`; pratikte hata → düzeltme) | **Evet** | İsimle **handle** |
| **C** | Turlar/oturumlar arası | **Evet** | İsimle handle + **keşif** |

**Gerilim:** PTC her şeyi tek script'e itiyor (kazanç oradan geliyor). Yani B azalıyor,
ama script büyüdükçe tek hatanın imha ettiği iş büyüyor. **Pratikte en çok A işe yarar.**

## 1.5 Dört ayrı depo, dört ayrı iş

```
Sandbox        → "çalış burada"           (geçici: /tmp, build, cache)
Artifact Store → "önemliyi burada sakla"  (durable: parquet, rapor, çıktı)
State Store    → "workflow nerede kaldı?" (son artifact, tur konumu)
LLM Context    → "şu an bilmem gereken"   (özet, sayı, handle)
```

Karar kuralı:

```
küçük + semantik   → context        (42, "completed", {"mean": 14.2})
geçici dosya       → sandbox        (/tmp, ara build)
büyük + reusable   → artifact store (parquet, model, rapor)
workflow konumu    → state store
```

---

# BÖLÜM 2 — PoC Planı

## 2.0 Ne inşa ediyoruz

```
┌── Sandbox Pod (efemer) ────────────────────────────┐
│  put_artifact / get_artifact / list_artifacts      │
│  cached(key, fn)                                   │
│  /scratch (emptyDir)                               │
└──────────────────┬─────────────────────────────────┘
                   │ MCP — tek izinli hedef
        ┌──────────▼──────────┐
        │  Tool Gateway       │
        │  = Artifact Service │  kapsam · doğrulama · TTL · hash
        └──────────┬──────────┘
              ┌────┴────┐
              ▼         ▼
        Metadata DB   Object Store
        (SQLite)      (MinIO → OBC)
```

**Namespace:** `artifacts/{user}/{workflow}/{run}/{artifact_id}`
Kullanıcı/workflow'a çapalı — böylece durable depo, uçucu bir anahtara bağlı kalmıyor.

## 2.1 Faz 0 — Ölçüm hijyeni (her koşulda doğru, bugün yapılabilir)

Ölçülen ~7,21 sn'nin (Job → ilk `tool_call` 3,08 sn) kalanının kayda değer kısmı
izolasyondan değil, kendi kodumuzdan geliyor:

| Kalem | Dosya | Yapılacak |
|---|---|---|
| Polling kuantizasyonu | `sandbox_runner.py` — `_POLL_INTERVAL_SECONDS = 1.0` | 0,1–0,2 sn'ye indir |
| İki `hubble observe` subprocess'i (her biri `timeout=10`), `_cleanup`'tan **önce**, sonucu beklenerek | `sandbox_runner.py` — `get_denied_actions()` | Kritik yoldan çıkar (arka plan / opsiyonel bayrak) |
| Senkron temizlik | `sandbox_runner.py` — `_cleanup()` | `ttlSecondsAfterFinished`'a bırak |

**Bu faz aynı zamanda OpenShift'e taşınmanın önkoşulu**: `hubble` Cilium'a özgü,
OVN-Kubernetes'te yok.

**Çıktı:** öncesi/sonrası ölçüm tablosu.

### SONUÇ (2026-09-03, tamamlandı)

Ölçüm aracı: `scripts/measure_sandbox.py` — LLM'i hiç devreye sokmadan
`run_sandbox`'ı doğrudan çağırır, aşama zamanlarını `on_event`'ten toplar.
5 koşunun medyanı, kind + tek node.

| | Dokümante edilen baseline | Faz 0 sonrası |
|---|---|---|
| **Toplam** | ~7,21 sn | **3,14 sn** (min 2,95 / max 3,34) |

Yapılan üç değişiklik:

| # | Değişiklik | Kazanç |
|---|---|---|
| 1 | `_POLL_INTERVAL_SECONDS` 1,0 → 0,15 | polling kuantizasyonu |
| 2 | `_cleanup` arka plana (daemon thread); `ttlSecondsAfterFinished` güvenlik ağı | `return == final` oldu |
| 3 | **Terminal JSON satırı görülünce erken dönüş** — Job.status güncellemesini bekleme | **~2,7 sn** (tek başına en büyük kalem) |

Ayrıca `hubble observe` çağrıları (iki subprocess, her biri `timeout=10`)
kritik yoldan tamamen kaldırıldı → `archive/egress-policy/code/`.

### Kalan 3,14 saniyenin anatomisi

| Aşama | Medyan | Δ |
|---|---|---|
| `configmap_created` | 0,01 sn | 0,01 |
| `job_created` | 0,03 sn | 0,01 |
| `pod_running` | 1,64 sn | **1,62** ← K8s pod kurulumu |
| `tool_call` (ilk) | 3,14 sn | **1,49** ← Python + fastmcp açılışı + MCP round trip |
| `final` / `return` | 3,14 sn | 0,00 |

**Warm pool kararı için kritik:** kalan maliyet neredeyse ikiye bölünüyor.
Düz bir warm pool (önceden yaratılmış pod'lar) yalnızca **1,62 sn'yi** geri alır;
diğer **1,49 sn** container içindeki süreç açılışıdır ve ancak *süreç* de önceden
ayaktaysa kazanılır (Anthropic'in container reuse + REPL kalıcılığının verdiği şey).

kubernetes.io'nun *"Starting a new pod adds about a second of overhead"* ifadesi
ile ölçtüğümüz 1,62 sn aynı mertebede — bağımsız doğrulama.

## 2.2 Faz 1 — OpenShift uyumu

**Kural: kind bir *çalıştırma ortamı*, bir *tasarım kaynağı* değil.** Her manifest
ve her sabit, OpenShift'te doğru olacak şekilde yazılır; kind onu taklit eder.
Tersi değil.

### 2.2.1 Kodda kalan kind varsayımları

| Varsayım | Nerede | OpenShift'te | Yapılacak |
|---|---|---|---|
| `NAMESPACE = "default"` | `sandbox_runner.py:48` | İş yükü `default`'ta koşmaz | Env değişkeninden oku (`PTC_NAMESPACE`) |
| `imagePullPolicy: Never`, `image: ptc-sandbox:local` | `job-template.yaml:27-28` | **Kırılır** — `kind load` yok | Registry'den çek; imaj referansı parametrik |
| **RBAC manifesti yok** | `k8s/` | kind'da cluster-admin olduğu için gizli kalmış | ServiceAccount + Role + RoleBinding yaz |
| `serviceAccountName` yok | `job-template.yaml` | Varsayılan SA yetersiz kalır | Açıkça belirt |
| `securityContext` yok | `job-template.yaml` | `restricted-v2` rastgele UID atar | §2.2.2 |
| `kind-config.yaml` | `k8s/` | Konusuz | Kalsın, yalnız yerel |

**RBAC'in ihtiyacı olan minimum** (kendi namespace'inde):
`configmaps: create,delete` · `jobs: create,delete` · `pods: get,list` · `pods/log: get`

### 2.2.2 SCC uyumu

| İş | Dosya |
|---|---|
| `emptyDir` scratch + `sizeLimit` | `k8s/sandbox/job-template.yaml` |
| `readOnlyRootFilesystem: true`, `runAsNonRoot: true`, `fsGroup` | aynı |
| Namespace'e PodSecurity `restricted` etiketi (**OpenShift SCC'sini kind'da taklit**) | `k8s/` |
| Rastgele yüksek UID ile çalıştırıp doğrula | test |

**Not:** 2026-09-02'de hazırlanan `ptc-sandbox:artifacts` / `tool-gateway:artifacts`
imajları bu açıdan **test edilmedi** — kök varsayımı olabilir.

### 2.2.3 Kata alınırsa gözden geçirilecek süreler

`activeDeadlineSeconds: 30` ve `_WAIT_TIMEOUT_SECONDS = 45.0`, `runc` boot süresine
göre konmuş. Kata (VM boot) bunları zorlayabilir — Kata devreye girerse ikisi de
yeniden ölçülmeli.

## 2.3 Faz 2 — Artifact Service

**Depo:** MinIO (kind'da), bağlantı bilgileri **OBC'nin ürettiği şekle birebir uyan**
bir ConfigMap + Secret'tan okunur. OpenShift'te gerçek OBC aynı şekli ürettiği için
**göçte kod değişmez, sadece manifest değişir.**

**Metadata DB:** gateway içinde SQLite. Şema:

```
artifact_id · name · workflow_id · run_id · turn_id
content_hash · type · size · storage_uri
parents[] · owner · created_at · ttl
```

**API (başlangıç seti):**

```python
artifact.create()    artifact.get()      artifact.list()
artifact.metadata()  artifact.delete()
```

**Gateway'in uygulayacağı kontroller:**

| Kontrol | Neden |
|---|---|
| Kapsam: `artifact_id` çağıranın workflow'una ait mi | Oturumlar arası sızıntı (ağ politikası bunu göremez) |
| **Yol/mime/boyut doğrulaması** | `artifact_save("/etc/shadow")` sınıfı |
| **pickle reddi** | CWE-502 — yazan taraf LLM'in ürettiği kod |
| Immutability | Yeni sürüm = yeni `artifact_id`; "last write wins" hiç oluşmaz |

**Eşik kuralı** (Airflow deseni, transport'a uygulanmış): eşik altı MCP'den inline;
eşik üstü için gateway **kısa ömürlü, tek kullanımlık, workflow'a bağlı** imzalı URL.

## 2.4 Faz 3 — Sandbox arayüzü

`entrypoint.py`'de mevcut `_make_sync_tool` deseni aynen kullanılabilir:

```python
h  = put_artifact(df, name="acik_ticketlar")      # B/C
df = get_artifact(h)
ls = list_artifacts(workflow="wf_42")             # C — keşif
df = cached("acik_ticketlar_v1", lambda: ...)     # A — retry atlar
```

- Serileştirme **sandbox tarafında** (Parquet varsayılan, Arrow IPC opsiyon)
- `ALLOWED_TOOLS` **iki yerde senkron**: `entrypoint.py` + `agent/tool_policy.py`
- `_ARG_NAMES`'e yeni girdiler

## 2.5 Faz 4 — State store ve keşif

Turlar arası boşluk: **2. tur, 1. turun ne ürettiğini nereden bilecek?**

PoC için en ucuz çözüm: `list_artifacts(workflow_id)` + `Trace`'e artifact
üretim/tüketim kaydı. Ayrı bir state servisi kurmaya gerek yok — metadata DB
zaten "workflow'un ne ürettiği" sorusunu cevaplıyor.

## 2.6 Faz 5 — Demo senaryosu

Üç okumayı da tek akışta gösteren senaryo:

```
Tur 1  "Açık ticketları çek ve özetle"
       → PTC#1 → 40 tool çağrısı → art_001 (parquet)          [C'nin kurulumu]

Tur 2  "Departmana göre grupla"
       → PTC#2 → get_artifact(art_001)  ← 40 çağrı TEKRAR YAPILMIYOR
       → ilk deneme KeyError                                   [B: self-repair]
       → PTC#3 düzeltilmiş → art_001 hâlâ elde → art_002
       → MAX_SANDBOX_RUNS bütçesi korunmuş oldu

Tur 3  "Rapor üret"
       → PTC#4 → list_artifacts() → art_002 → rapor            [C: keşif]
```

`cached()` ise A için ayrı ölçülür: aynı script iki kez, ikincisinde pahalı blok atlanır.

**Ölçülecekler:**

| Metrik | Kalıcılıksız | Kalıcılıkla |
|---|---|---|
| Toplam tool çağrısı | | |
| Tur 2 süresi | | |
| Self-repair maliyeti (B) | | |
| Retry'da atlanan blok (A) | | |

### SONUÇ — A okuması ölçüldü (2026-09-03)

`scripts/demo_retry_maliyeti.py`. Senaryo, PTC'nin gerçek başarısızlık biçimi:
tek script pahalı işi (12 tool çağrısı) bitirir, sonra son satırda `NameError`
verir; düzeltilmiş sürüm baştan koşar.

| | 1. deneme (hata) | 2. deneme (düzeltme) | Toplam |
|---|---|---|---|
| **Kalıcılıksız** | 12 çağrı | **12 çağrı** | 24 |
| **`cached()` ile** | 14 çağrı¹ | **1 çağrı** | 15 |

¹ 12 pahalı çağrı + `get_artifact` (boş döner) + `put_artifact`.

**Düzeltme denemesinde 12 çağrı 1'e indi** — pahalı blok hiç çalışmadı,
sonucu artifact'ten geldi. `MAX_SANDBOX_RUNS_PER_TURN = 2` bütçesi altında bu
fark, self-repair'i pahalı olmaktan çıkarıyor.

**Süre sütunu bilerek tabloya konmadı.** Ölçümde toplam süre 8,36 → 8,06 sn
çıktı, yani neredeyse fark yok — çünkü mock tool'lar anında dönüyor ve süreyi
4 pod başlatması (~3,9 sn/adet) belirliyor. Anlamlı sinyal **çağrı sayısı**;
zaman kazancı gerçek kaynak sistemlerin gecikmesiyle orantılı büyür, pod
maliyetiyle değil. Bunu demoda da açıkça söylemek gerekiyor — aksi halde
tablo kendini olduğundan fazla satar.

### SONUÇ — B ve C okumaları (2026-09-03)

`scripts/demo_artifact_persistence.py`. Üç ayrı pod:

| PTC | Node | Tool'lar | Sonuç |
|---|---|---|---|
| #1 | extract | `count_open_tickets`, `put_artifact` | `art_614007ebe4c8` |
| #2 | transform | `get_artifact`, `put_artifact` | **`count_open_tickets` hiç çağrılmadı** |
| #3 | — | `list_artifacts` | Kendisinden önce üretilen ikisini de buldu |

Üç pod da silindi; artifact'ler duruyor.

## 2.6b Depo kararı ve dayanıklılık düzeltmesi (2026-09-03)

**Karar: MinIO kalıyor** (yerel), OpenShift'te OBC. Kod S3 protokolü konuştuğu
için bu bir kilitlenme değil — `BUCKET_HOST`/`BUCKET_PORT`/`AWS_*` değiştirilerek
ODF/NooBaa'ya ya da gerçek AWS S3'e geçilebilir. Kodda MinIO'ya özgü tek satır
yok (`from minio import Minio` bir S3 *istemcisidir*, boto3 gibi).

### Bulunan ve kapatılan açık

Depo topolojisi incelenirken ölçülmüş bir tutarsızlık çıktı:

| | Nerede | Pod ölürse |
|---|---|---|
| Baytlar | MinIO → PVC | Kalır |
| Metadata | gateway **`/tmp`** | **Giderdi** |

Sayım: bucket'ta 9 nesne, metadata'da 7 kayıt → **2 yetim blob**. Baytları
duruyor, onları gösteren kayıt yok. Bu, `metadata.py`'de "MLflow'un sorunu"
diye belgelediğimiz şeyin ters yönden gelmiş hâli: silme sırasını doğru
tasarlamıştık ama **dağıtım** yetim üretiyordu.

**Düzeltme:** gateway'e `tool-gateway-metadata` PVC'si (1Gi),
`PTC_METADATA_DB=/var/lib/ptc/artifacts.db`.

**Doğrulama:** artifact yazıldı → gateway pod'u silindi → **yeni pod'dan**
okundu, `[1, 2, 3]` geri geldi.

**Kalan sınır:** SQLite ReadWriteOnce bir PVC'de olduğu için gateway şu an
tek replika. Üretimde `open_postgres()` — Red Hat'in Model Registry için
çizdiği çizginin aynısı.

## 2.7 Kapsam dışı (bilerek)

| Konu | Neden |
|---|---|
| Kata / `runtimeClassName` | Ekibin cluster'ında test edilir; yerelde nested virt gerekir |
| Gerçek OBC | Aynı — şekli taklit ediyoruz, kod değişmeyecek |
| Warm pool | Hız Faz 0'dan sonra hâlâ acıtırsa |
| Uzun ömürlü sandbox / REPL durumu | Serileştirilemeyen bir ihtiyaç çıkarsa |
| Yerel OpenShift (CRC) | 10,5 GB RAM ister (makinede 15 GB toplam); üstelik OBC/Kata'yı zaten doğrulayamaz. **Kapsam dışı olması, kind'a göre tasarlamak anlamına gelmez** — §2.2 |
| Versiyonlama (v1/v2 ağacı) | Immutable ID yeterli; ağaç sonraki iş |

## 2.8 Ekibe iki soru

1. **ODF kurulu mu?** Kuruluysa depo bir `ObjectBucketClaim`. Değilse MinIO iş yükü
   olarak koşar — imaj sabit UID istediği için `anyuid` SCC'si gerekebilir.
2. **Sandboxed containers kullanılabilir mi, hangi OCP sürümü?** Kullanılabilirse
   `runtimeClassName` ile izolasyon ailesi atlanıyor.

**Hiçbiri Faz 0–4'ü bloke etmiyor.**

## 2.9 Sıra

```
Faz 0  ölçüm hijyeni        ← bağımsız, bugün
Faz 1  yazılabilir zemin + SCC
Faz 2  artifact service (MinIO + gateway + metadata)
Faz 3  sandbox arayüzü (4 fonksiyon)
Faz 4  keşif + Trace kaydı
Faz 5  demo senaryosu + ölçüm
```
