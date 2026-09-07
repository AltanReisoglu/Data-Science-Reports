# PTC — Bütün Karşılaştırma Tabloları

**Tarih:** 2026-09-07 · Anlatının tamamı:
[PTC_Piyasa_Mentaliteleri.md](PTC_Piyasa_Mentaliteleri.md)

Bu dosya **yalnızca tablolardan** oluşuyor. Oturum boyunca dağınık yerlerde
üretilen karşılaştırmaların tek yerde toplanmış hâli — sunumda ve kod
incelemesinde hızlı bakmak için. Her tablonun altında hangi bölümden geldiği
yazılı; gerekçeler ve alıntılar orada.

## İçindekiler

| # | Tablo | Soru |
|---|---|---|
| [1](#1--beş-soru) | Beş soru | Karşılaştırmanın iskeleti |
| [2](#2--izolasyon-s1) | İzolasyon | Kod nerede çalışıyor |
| [3](#3--sandbox-ömrü-s2) | Sandbox ömrü | Ne kadar yaşıyor |
| [4](#4--depoya-erişim-s3-ve-anahtar-s4) | Depoya erişim + anahtar | Nasıl ulaşıyor, kimlik kimde |
| [5](#5--kayıt-defteri-s5) | Kayıt defteri | Künye tutuluyor mu |
| [6](#6--baytlar-fiilen-nerede-duruyor) | Baytların yeri | Hangi üründe |
| [7](#7--sandboxtaki-kod-ne-yazıyor) | Kodun yüzeyi | Fonksiyon mu, yol mu |
| [8](#8--baytı-kim-taşıyor--aracının-yeri) | Aracının yeri | Sarmalayıcı mı, sınır mı |
| [9](#9--aktarım-yöntemleri--dokuz-seçenek) | Aktarım yöntemleri | Dokuz seçenek, neden 9 |
| [10](#10--keşif-ajan-çekeceğini-nasıl-anlıyor) | Keşif | Ajan nasıl buluyor |
| [11](#11--ağ-duruşu) | Ağ duruşu | Ne kadar kapalı |
| [12](#12--süreklilik-anahtarı) | Süreklilik | Ne canlanıyor |
| [13](#13--openshiftin-iki-artifact-deseni) | OpenShift'in iki deseni | KFP vs Tekton |
| [14](#14--cognition--cerebras--databricks) | Üç şirket daha | Devin, Cerebras, Databricks |
| [15](#15--biz-neredeyiz) | Biz neredeyiz | Boyut boyut |
| [16](#16--bilinen-açıklar) | Açıklar | Saklamıyoruz |
| [17](#17--openshift-uyumluluğu) | **OpenShift uyumluluğu** | **Bizimki orada çalışır mı** |

---

## 1 — Beş soru

Bütün karşılaştırmanın iskeleti. Her sistem bunları cevaplamak zorunda.

| | Soru |
|---|---|
| **S1** | Kod **nerede** çalışıyor? (izolasyon) |
| **S2** | Sandbox **ne kadar** yaşıyor? |
| **S3** | Depoya **nasıl** erişiliyor? |
| **S4** | **Anahtar** kimde? |
| **S5** | **Kayıt defteri** var mı? |

*(§1)*

---

## 2 — İzolasyon (S1)

| Yöntem | Ne demek | Kim | Güç |
|---|---|---|---|
| Container (runc) | Kernel paylaşılır | **BİZ** | En zayıf |
| V8 isolate | JS motoru içinde bölge | Cloudflare Code Mode | Dar ama sıkı |
| gVisor | Araya sahte kernel | Google GKE Agent Sandbox | Orta |
| Kata / microVM | Pod başına kendi kernel'i | Red Hat önerisi, E2B, Vercel, **Devin** | Güçlü |
| Hyper-V | Donanım sanallaştırma | Microsoft | Güçlü |
| Lakeguard | Spark Connect + container sandbox + egress izolasyonu | Databricks | Orta-güçlü |

**Bu listede en zayıf olan biziz.** Red Hat AI-üretimi kod için açıkça Kata
öneriyor; kod değişikliği gerektirmiyor, node seviyesinde önkoşul. *(§10.1)*

---

## 3 — Sandbox ömrü (S2)

| Model | Kim | Süre |
|---|---|---|
| Her çağrıda yeni, saklanmaz | **BİZ** | ~4,1 sn |
| Efemer + cooldown'da yok edilir | Microsoft | Havuzdan ms |
| Varsayılan yeni, id ile canlanır | Anthropic | **30 gün** (5 dk'da checkpoint) |
| Adlandırılmış, uzun ömürlü | Google Agent Engine | **14 gün** |
| Süre sınırlı oturum | AWS | 15 dk – 8 saat |
| Kalıcı workspace + snapshot | OpenAI | — |
| Hipervizör snapshot (RAM + süreç + disk) | **Devin** | Süresiz, uykuya geçiyor |
| SSH'lı ev dizini | Databricks Sandbox (beta) | 100 GB, oturumlar arası |

**Ayrım:** Anthropic *container'ı* saklıyor, biz *artifact'i*. Farklı problem.
*(§10.2)*

---

## 4 — Depoya erişim (S3) ve anahtar (S4)

| Yaklaşım | Kim | Anahtar sandbox'ta | Aile (§8) |
|---|---|---|---|
| **Mount, anahtar içeride** | E2B, Daytona (external), Vercel (düz) | **Evet** | A |
| **Mount, anahtar dışarıda** | Cloudflare (binding/proxy), Vercel (proxy), Daytona (Volumes) | Hayır | A (melez) |
| **Blok cihaz** | Fly.io | Yok | A |
| **Denetimli mount** | **Databricks** (`/Volumes`, Unity Catalog) | Hayır | **D** |
| **SDK, anahtar pod'da** | Red Hat OpenShift AI (KFP), AWS (IAM role) | Evet (dar) | **B** |
| **API, anahtar hiç yok** | Anthropic, OpenAI, **BİZ** | Hayır | **C** |

*(§10.3)*

---

## 5 — Kayıt defteri (S5)

| VAR | YOK |
|---|---|
| **BİZ** — `artifact_id`, tip, soy, TTL, hash | E2B, Modal, Daytona, Vercel |
| Anthropic — Files API | Cloudflare, Fly.io, Microsoft |
| Red Hat / KFP — MLMD | AWS (denetim CloudTrail'de, registry değil) |
| Google ADK — artifact adı + sürüm | **Devin** (çıktı git'te), Cerebras |
| Databricks — Unity Catalog + MLflow | |

> **Araştırmanın en keskin bulgusu:** Mount eden sağlayıcıların **hiçbirinde**
> artifact registry yok. Yazılan dosyanın `artifact_id`'si, soyu, TTL'i yok —
> sadece bir S3 anahtarı. Dokümanlar "persistent data access" diyor; hiçbiri
> "artifact" demiyor.

**Kural:** Kayıt defteri, yalnızca yazma yolu **bir bileşenden geçtiğinde**
ayakta kalıyor. *(§10.4, §9)*

---

## 6 — Baytlar fiilen nerede duruyor

| | Depo ürünü | Sandbox'taki yol | Kayıt defteri | Baytların ömrü |
|---|---|---|---|---|
| **Anthropic** | Belgelenmemiş | container diski (5 GiB) | `file_id` | 30 gün |
| **OpenAI** | Belgelenmemiş | container diski | yok | 20 dk hareketsizlik |
| **Cloudflare** | **R2** / S3 / GCS | mount noktası | yok | bucket'ın ömrü |
| **Google (ADK)** | **GCS** / bellek / yerel disk | ArtifactService API | ADK adı+sürüm | deponun ömrü |
| **AWS** | **S3 Files** / **EFS** (senin hesabın) | `/mnt/<ad>` | yok | bucket/EFS |
| **Microsoft** | **yok** | `/mnt/data` | yok | oturumla ölür |
| **Red Hat / KFP** | S3 / GCS / MinIO / SeaweedFS | `.path` (launcher kopyalar) | **MLMD** | `pipeline_root` |
| **Databricks** | **Unity Catalog Volumes** | `/Volumes/<cat>/<schema>/<vol>` | **UC + MLflow** | volume'ün ömrü |
| **Devin** | **yok** — git | makine snapshot'ı | git commit | süresiz |
| **E2B / Daytona / Vercel** | senin bucket'ın | mount noktası | yok | bucket'ın ömrü |
| **Modal** | kendi FS'i / S3-R2-GCS | Volume ya da mount | yok | Volume'ün ömrü |
| **Fly.io** | S3-uyumlu (JuiceFS benzeri) | kök FS (100 GB) | yok | Sprite'ın ömrü |
| **BİZ** | **MinIO / ODF / harici S3** | `/output` + `/artifacts/<wf>` | **SQLite** | TTL + reaper |

*(§9.5.10)*

---

## 7 — Sandbox'taki kod ne yazıyor

| Ürün | Kodun dokunduğu şey | Dışarı nasıl çıkıyor |
|---|---|---|
| **Anthropic** | `$OUTPUT_DIR/` altına **dosya** | Üst düzey yakalanır → `file_id` |
| **OpenAI** | `/mnt/data` altına **dosya** | container file content uç noktası |
| **Microsoft** | `/mnt/data` altına **dosya** | `files` yönetim API'si (128 MB sınır) |
| **KFP / OpenShift AI** | `.path` — **yerel yol** | launcher `.uri`'ye kopyalar |
| **AWS** | `/mnt/s3data` — **mount** | çift yönlü senkron |
| **Databricks** | `/Volumes/...` — **POSIX/FUSE** | UC yönetiminde |
| **Cloudflare / E2B / Daytona / Vercel** | mount edilmiş **yol** | dosya sistemi zaten bucket |
| **Fly.io** | normal kök **dosya sistemi** | chunk'lar S3'te |
| **BİZ** | `/output/` altına **dosya** | sidecar süpürür → depo |

**Dokuzunun dokuzu da bir DOSYA YOLU.** Hiçbiri sandbox'taki koda artifact
fonksiyonu sunmuyor. *(§9.6, §11.11)*

> Görünen tek istisna Google ADK: `save_artifact()` / `load_artifact()` var —
> ama doküman bunun **geliştirici API'si** olduğunu söylüyor; model çağıramıyor.
> Modelin gördüğü tek şey `LoadArtifactsTool`, o da sadece **isimleri**
> talimatlara koyuyor.

---

## 8 — Baytı kim taşıyor / aracının yeri

### Dört aile

| Aile | Model | Araya girecek yer |
|---|---|---|
| **A — Mount** | `write()` → FUSE/NFS → bucket | **Yok** |
| **B — Sarmalayıcı** | launcher, kullanıcı koduyla **aynı container** | Var ama kod onu atlayabilir |
| **C — Sınır** | kod dizine yazar → **ayrı güven alanı** taşır | Var, kod erişemez |
| **D — Denetimli mount** | mount, ama sürücü yetkilendirme uyguluyor | Sürücünün içinde |

### Yükleyici nerede çalışıyor

| Ürün | Yükleyici nerede | Kimlik kodun ulaşabileceği yerde mi | Aile |
|---|---|---|---|
| **Argo Workflows** | **`wait` sidecar** (ayrı container) | **Hayır** | C |
| **Tekton** | entrypoint binary + controller'ın enjekte ettiği sidecar | Hayır | C |
| **KFP / OpenShift AI** | `kfp-launcher`, **main container** | **Evet** | B |
| **Databricks** | FUSE sürücüsü (compute plane) | Hayır | D |
| **AWS AgentCore** | Yükleyici yok — platform NFS mount ediyor | Hayır (IAM dar) | A |
| **Anthropic / OpenAI / Microsoft** | Platform, dışarıdan hasat ediyor | Hayır | C |
| **E2B / Daytona / Vercel / Modal / Fly** | Yükleyici yok — kernel/FUSE | **Evet** (çoğunda) | A |
| **BİZ** | **`artifact-sidecar`** (ayrı container) | **Hayır** | **C** |

*(§9.6, §11.12)*

---

## 9 — Aktarım yöntemleri: dokuz seçenek

| # | Yöntem | Kim | Kimlik kodun elinde mi | Neden biz değil |
|---|---|---|---|---|
| 1 | FUSE mount (s3fs) | E2B, Daytona, Vercel | **Evet** | Denetim koyacak yer yok → kayıt defteri ölür |
| 2 | Mount + imzalayan proxy | Cloudflare, Vercel | Hayır | Proxy imzalar ama **içeriğe bakmaz** |
| 3 | Platform NFS mount | AWS | Hayır (IAM dar) | Yönetilen depo yok; kayıt defteri yok |
| 4 | Denetimli mount | **Databricks** | Hayır | En yakın rakip — tüm platformu gerektiriyor |
| 5 | Blok cihaz / makine snapshot | Fly.io, Devin | — | Devin'in artifact'i git |
| 6 | Main container'da launcher | **KFP** | **Evet** | Kodu güvenilir varsayıyor |
| 7 | Platform hasadı | Anthropic, OpenAI, MS | Hayır | Kapalı; taşınabilir karşılığı yok |
| 8 | Presigned URL | — | kısa ömürlü | **Denetimi kaldırır** |
| 9 | **init + wait sidecar** | **Argo**, Tekton | **Hayır** | ✅ **Bizim seçtiğimiz** |

**Argo dört yerleşimi denedi** (`docker`, `kubelet`, `k8sapi`, `pns`) ve v3.4'te
hepsini kaldırdı — `docker` için gerekçe *"breaks security completely"*.
*(§11.12)*

---

## 10 — Keşif: ajan çekeceğini nasıl anlıyor

| # | Desen | Nasıl | Kim |
|---|---|---|---|
| 1 | Sadece tool tarifi | Model çağırmayı *seçmek* zorunda | *(2026-09-06'ya kadar biz)* |
| 2 | Referans otomatik context'te | `file_id` tool sonucunda döner | Anthropic |
| 3 | Dosya sistemi + `ls` | Sandbox yaşıyorsa model bakar | Anthropic, Google, OpenHands |
| 4 | **İsimler prompt'a enjekte** | İsimler talimatlarda, içerik talep üzerine | **Google ADK** |
| 3+4 | **İkisi birden** | Manifest promptta **ve** `os.listdir` çalışıyor | **BİZ** |
| 5 | Semantik arama | Vektör deposunda `file_search` | Llama Stack (RAG) |
| — | **Keşif YOK** | DAG statik, girdi bağlanmış | KFP, Argo, Airflow, Tekton |

**Desen 4'ün üç kuralı:** isimler her zaman context'te · içerik talep üzerine ·
içerik geçmişe kalıcı yazılmaz. *(§10.6)*

---

## 11 — Ağ duruşu

| Duruş | Kim |
|---|---|
| Tamamen kapalı | Anthropic — *"Completely disabled for security"* |
| **Varsayılan reddet + allowlist** | **BİZ**, Red Hat / OpenShell, Google GKE |
| Açık, yapılandırılabilir | AWS ("network modes") |
| UDF egress izolasyonu | Databricks (Lakeguard) |
| Opsiyonel kontroller | Microsoft |

**Bizde iki servis bilerek ayrı:** Tool Gateway internete çıkar depoya çıkamaz;
Artifact Service depoya çıkar internete çıkamaz. *(§10.7)*

---

## 12 — Süreklilik anahtarı

| Kim | Alan adı | Neyi canlandırıyor |
|---|---|---|
| Anthropic | `container` id | **Container'ın kendisi** (checkpoint'ten) |
| Google | `sandbox_name` | **Sandbox'ın kendisi** (14 gün) |
| Microsoft | `identifier` | **Session'a yönlendirme** |
| Devin | machine snapshot | **Makinenin tamamı** (RAM dahil) |
| **BİZ** | oturum uuid'si | **Sadece depo kapsamı** — container her seferinde yeni |

*(§10.5)*

---

## 13 — OpenShift'in iki artifact deseni

| | OpenShift AI (KFP) | OpenShift Pipelines (Tekton) |
|---|---|---|
| Cevapladığı soru | *"Veriyi adımlar arası taşı ve sakla"* | *"Bu çıktının neyden üretildiğini kanıtla"* |
| Artifact **nedir** | Dosyanın kendisi | `uri` + `digest` — **referans** |
| Baytlar nerede | Nesne deposu (`pipeline_root`) | Başka yerde: registry, PVC, git |
| Nasıl bildiriliyor | İmzada `Output[Dataset]` | Step, `$(step.artifacts.path)`'e JSON yazıyor |
| Kayıt defteri | **MLMD** (MySQL/MariaDB) | **Tekton Chains** → 7 arka uçtan biri |
| Ne saklanıyor | İçerik **ve** künye | *"only metadata and attestations… **not artifact content**"* |
| Amaç | Veri kalıcılığı | Tedarik zinciri güvenliği (SLSA/in-toto) |

**Bizim sorumuz birincisi.** *(§8.6)*

---

## 14 — Cognition · Cerebras · Databricks

| | Sandbox izolasyonu | Artifact deposu | Aracı nerede | Kayıt defteri |
|---|---|---|---|---|
| **Devin** | microVM + hipervizör snapshot | **yok** — git | — | git (commit) |
| **Cerebras** | yok (geliştiricinin Docker'ı) | yok | — | yok |
| **Databricks** | Lakeguard | **Unity Catalog Volumes** | FUSE sürücüsü (D) | **var** (UC + MLflow) |
| **BİZ** | container + ağ politikası | MinIO/ODF + kayıt defteri | sidecar (C) | var |

**Devin'in artifact'i git** — çıktısı bir PR. **Cerebras bu işte değil** —
modeli veriyor, kalıcılık katmanını vermiyor. **Databricks en yakını** — ama
UC function dokümanı bile *"Customers are responsible for running only trusted
code"* diyor. *(§9.7)*

---

## 15 — Biz neredeyiz

| Boyut | Kimle aynı |
|---|---|
| PTC tezi (kod yaz, tool çağırma) | **Cloudflare** |
| Ağ (kapalı + allowlist) | **Anthropic**, Red Hat, Google GKE |
| Depoya erişim (mount yok) | **Red Hat**, Anthropic, OpenAI |
| Çıktı yakalama (`/output` süpürme) | **Anthropic**, OpenAI |
| Launcher deseni (`.path` → `.uri`) | **Red Hat / KFP** |
| Kayıt defteri + tipler + kök | **Red Hat / KFP (MLMD)** |
| Keşif (isimler prompt'ta) | **Google ADK** |
| LLM yüzeyi (dosya, API yok) | **hepsi** — KFP, Anthropic, OpenAI, MS |
| Kapsam (tenant) | **Red Hat / KFP** — `pipeline_root` paylaşımlı |
| Çalıştırma izolasyonu (`/output` + `/artifacts/<wf>`) | **KFP** — `pipeline_root/<run-id>/` |
| **Aktarımı başlatan (sidecar)** | **Argo Workflows** — init + wait |
| İzolasyon | **Kimse — bizimki daha zayıf** |
| Şeffaf okuma | **Kimse — emsalsiz** |

*(§12)*

---

## 16 — Bilinen açıklar

| Konu | Durum | Etki |
|---|---|---|
| **İzolasyon** | Düz container, Kata yok | Red Hat'in önerisine uymuyoruz |
| **Metadata DB** | SQLite | Tek replika sınırı |
| Workflow state | Postgres yolu test edilmedi | Cluster'da Postgres yok |
| **Auth** | Yok | Jetonu üretebilen tenant'ın tümünü okur |
| Büyük dosya | 100 MiB servis / 1 Gi pod | 5 GB çalışmaz |
| İsim çakışması | Aynı workflow'da "en yeni" kazanır, sessiz | Veri kaybı riski |
| Şeffaf okuma | 5 pandas okuyucusu + `open` + `listdir`/`exists`/`glob` | `pyarrow`, `PIL` doğrudan açarsa yakalanmıyor |
| `user_metadata` | Süpürme yolunda doldurulamıyor | API kalkınca kapandı |
| Tip | Yalnızca dosya uzantısından | Metrik/Dataset ayrımı kayboldu |
| Soy imzasız | Kayıt defterine yazabilen değiştirebilir | Tekton Chains bunu çözüyor |
| Manifest host değişkenine bağlı | `ARTIFACT_SERVICE_URL` yoksa sessizce kapanıyordu | Artık uyarı basıyor |
| Gerçek OBC/ODF | Test edilmedi | ODF kapsam dışı |

*(§11.10)*

---

## 17 — OpenShift uyumluluğu

**Soru: bizimki OpenShift'te çalışır mı, ve her şey OpenShift varsayılanı mı?**

### Test edilenler (kind üzerinde, OpenShift SCC'si taklit edilerek)

`restricted-v2` SCC'sinin koşulları uygulanarak tam akış çalıştırıldı:
`runAsNonRoot: true`, rastgele UID (1000670000), `capabilities: drop [ALL]`,
`seccompProfile: RuntimeDefault`.

| Kontrol | Sonuç |
|---|---|
| Sidecar + sandbox açılışı | ✅ |
| `/output`, `/scratch` yazılabilir | ✅ emptyDir **0777** geliyor, rastgele UID yazabiliyor |
| matplotlib (`MPLCONFIGDIR=/scratch/.mpl`) | ✅ `HOME=/` olmasına rağmen |
| Parquet + PNG üretimi | ✅ |
| Sidecar süpürmesi + yükleme | ✅ `produced scc.parquet`, `produced scc.png` |
| Sabit UID 1001 ile (kind) | ✅ üretim + tembel okuma + soy |
| Rastgele UID 1000670000 ile (SCC ezmesi) | ✅ aynı sonuç |

**Sonuç: iş yükü OpenShift uyumlu.** SCC'nin dayattığı hiçbir kısıt bizi
kırmıyor.

İki düzeltme yapıldı ki kind ile OpenShift **aynı** davransın (fark ancak
OpenShift'te görülürdü):

- Job şablonuna açık `securityContext` — `runAsNonRoot: true`,
  `seccompProfile: RuntimeDefault`, container'larda `capabilities: drop [ALL]`.
  `runAsUser` **bilerek yok**: OpenShift namespace'e özgü UID atıyor.
- Sandbox imajına `USER 1001` — `runAsNonRoot` bir root imajı görünce
  `CreateContainerConfigError` veriyor. Sayısal olmak zorunda; kontrol isim
  çözemiyor.

### Değişmesi gerekenler

| Bizde (kind) | OpenShift varsayılanı | Durum |
|---|---|---|
| **Cilium CNI** | **OVN-Kubernetes** | ⚠️ Değişmeli |
| `CiliumNetworkPolicy` — pod→pod (4 politika) | standart `NetworkPolicy` | ✅ Birebir çevrilebilir |
| `CiliumNetworkPolicy` — FQDN allowlist | **`EgressFirewall`** (`k8s.ovn.org/v1`, `dnsName`) | ✅ Çevrilebilir, uyarısıyla |
| **Hubble** (canlı akış paneli) | Karşılığı **yok** | ⚠️ Panelin o bölümü çalışmaz |
| `imagePullPolicy: Never` + `kind load` | Internal registry / ImageStream | ⚠️ Değişmeli |
| MinIO (kendi deploy'umuz) | ODF/OBC ya da harici S3 | ✅ **Zaten iki sözleşme de okunuyor** |
| Yerleşik sidecar (k8s 1.29+) | **OCP 4.16+** | ✅ Sürüm koşulu |
| `securityContext` yok | SCC enjekte ediyor | ✅ **Eklendi** — `runAsNonRoot` + `seccompProfile` + `drop: [ALL]` artık şablonda |
| İmaj root çalışıyordu | SCC rastgele UID atıyor | ✅ **Düzeltildi** — `USER 1001` (sayısal, `runAsNonRoot` isim çözemiyor) |

### `EgressFirewall`'un uyarısı

Red Hat dokümanı DNS adı kullanımını sınırlıyor:

> *"Because the egress firewall controller and pods asynchronously poll the same
> local name server, the pod might obtain the updated IP address before the
> egress controller does, which causes a race condition. Due to this current
> limitation, **domain name usage in EgressFirewall objects is only recommended
> for domains with infrequent IP address changes**."*

Sık değişen IP'ler ve joker adlar için `DNSNameResolver` var — ama o
**Technology Preview**.

### Özet

| | Değerlendirme |
|---|---|
| **İş yükü** (Job, sidecar, imajlar, SCC) | ✅ OpenShift varsayılanıyla çalışıyor — **test edildi** |
| **Depo sözleşmesi** | ✅ OBC ve OpenShift AI connection, ikisi de okunuyor |
| **Ağ politikaları** | ⚠️ Cilium'a bağımlı; OVN karşılıkları var, çeviri gerekiyor |
| **Gözlemlenebilirlik** (Hubble) | ⚠️ OpenShift'te karşılığı yok; panelin canlı akış bölümü kapanır |
| **İmaj dağıtımı** | ⚠️ Registry gerekiyor |

**Tek cümle:** Ürünün kendisi OpenShift varsayılanında çalışıyor; **kind'a
özgü olan altyapı katmanı** (Cilium, Hubble, `kind load`), ve bunların
OpenShift karşılıkları belli.

---

## Kaynaklar

Bütün alıntılar ve linkler
[PTC_Piyasa_Mentaliteleri.md](PTC_Piyasa_Mentaliteleri.md)'nin **Kaynaklar**
bölümünde. §17'nin kaynakları:

- [OpenShift — Egress Firewall](https://docs.redhat.com/en/documentation/openshift_container_platform/4.18/html/network_security/egress-firewall)
- [OpenShift — EgressFirewall API (`k8s.ovn.org/v1`)](https://docs.redhat.com/en/documentation/openshift_container_platform/4.17/html/network_apis/egressfirewall-k8s-ovn-org-v1)
- [OCP 4.16 release notes](https://docs.redhat.com/en/documentation/openshift_container_platform/4.16/html/release_notes/ocp-4-16-release-notes) — Kubernetes 1.29
- [Kubernetes — Sidecar Containers](https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/)
