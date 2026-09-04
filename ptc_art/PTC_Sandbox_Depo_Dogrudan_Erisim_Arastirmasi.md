# Sandbox ↔ Nesne Deposu Doğrudan Erişim — Araştırma

**Tarih:** 2026-09-04 · **Durum:** dış kaynak taraması, PoC yok

Bu doküman tek bir soruyu kovalıyor:

> PTC'de sandbox, Tool Gateway'i aradan çıkarıp nesne deposuyla **doğrudan**
> konuşsaydı ne olurdu? Sahada bunu kim, nasıl yapıyor?

---

## 0. Bu doküman neyi kapsıyor, neyi kapsamıyor

**Verili kabul edilenler** (tekrar araştırılmadı):

| Konu | Nerede |
|---|---|
| Mevcut mimarinin tamamı | [PTC_Mimari.md](PTC_Mimari.md) |
| İzolasyon primitifleri, kalıcılık modelleri, SOTA sandbox taraması | [PTC_Calistirma_Ortami_ve_SOTA_Arastirmasi.md](PTC_Calistirma_Ortami_ve_SOTA_Arastirmasi.md) |
| Orkestratör desenleri, pickle/CWE-502, presigned URL'in egress riski, opaque handle | [PTC_Artifact_Persistence_Arastirmasi.md](PTC_Artifact_Persistence_Arastirmasi.md) |

**Bu dokümanın kapsamı:** erişim *mekaniği* (mount mu, SDK mı, hangi ayrıcalıkla),
kimlik bilgisinin *nerede durduğu*, kapsamın *nasıl daraltıldığı*, doğrudan yazma
olunca *kayıt defterine ne olduğu*, ve *içerik denetiminin* nereye taşındığı.

**Kapsam dışı:** hangi izolasyon primitifi (gVisor/Kata/Firecracker) — zaten
tarandı. Presigned URL'in neden PTC modelini kırdığı — zaten yazıldı; burada
yalnızca *melez desenlerde* nerede kullanıldığına bakılıyor.

---

## 1. Çerçeve: "doğrudan erişim" tek bir şey değil

Sağlayıcı dokümanlarını yan yana koyunca, "sandbox nesne deposuna doğrudan
erişiyor" cümlesinin **dört ayrı ekseni** gizlediği görülüyor. Karar bu dört
ekseni ayırmadan verilemiyor.

| Eksen | Uçlar |
|---|---|
| **E1 — Mount'u kim yapıyor** | Sandbox'ın *içindeki* kod (`sudo s3fs …`) ↔ platform, sandbox'ın *dışında* |
| **E2 — Kimlik bilgisi nerede** | Sandbox'ın diskinde/ortamında ↔ sandbox'ın hiç göremediği bir bileşende |
| **E3 — Kapsam nerede daraltılıyor** | Hiç ↔ prefix/IAM ↔ ayrı bucket ↔ ayrı hesap |
| **E4 — İçerik denetimi nerede** | Hiç ↔ yazma yolunda araya giren bileşen ↔ okuma yolunda |

Kritik gözlem: **E1 ile E2 bağımsız.** Sandbox içinde FUSE mount olması,
kimlik bilgisinin sandbox'ta olmasını gerektirmiyor. Sahadaki en ilginç iki
tasarım (Vercel, Cloudflare) tam olarak bu ayrımın üstüne kurulu — ve PTC için
en anlamlı çıkış yolu da orada.

---

## 2. S1 — Erişim mekanizması: kim, neyi, nasıl bağlıyor

### 2.1 Toplu tablo

| Sistem | Mekanizma | FUSE sürücüsü | Kimlik bilgisi sandbox'ta mı | Kaynak |
|---|---|---|---|---|
| **E2B** | Sandbox içinde `sudo s3fs` / `gcsfuse`; özel template'e FUSE kurulur | s3fs, gcsfuse | **Evet** — `/root/.passwd-s3fs` dosyasına yazılıyor | [E2B Cloud buckets](https://e2b.dev/docs/sandbox/connect-bucket) |
| **Modal** | `CloudBucketMount`, `Sandbox.create(volumes=…)` ile | "AWS' mountpoint technology" | Belirsiz — `secret=` mount'a veriliyor (bkz. §7) | [Modal cloud bucket mounts](https://modal.com/docs/guide/cloud-bucket-mounts), [modal.Sandbox](https://modal.com/docs/reference/modal.Sandbox) |
| **Daytona — Volumes** | Platform tarafından mount edilen FUSE; `subpath` ile daraltma | belirtilmemiş | **Hayır** — kullanıcı hiç anahtar görmüyor | [Daytona Volumes](https://www.daytona.io/docs/en/volumes/) |
| **Daytona — External storage** | Snapshot'a `mount-s3`/`gcsfuse`/`blobfuse2`/`rclone` gömülür, sandbox içinde çalıştırılır | Mountpoint, gcsfuse, blobfuse2, rclone | **Evet** — `envVars` ile `AWS_ACCESS_KEY_ID` geçiliyor | [Daytona Mount External Storage](https://www.daytona.io/docs/en/mount-external-storage/) |
| **Vercel Sandbox — düz mount** | `sudo mount-s3` sandbox içinde | Mountpoint for S3 | **Evet** — doküman bunu açıkça uyarıyor | [Vercel — Mount remote storage](https://vercel.com/docs/sandbox/mount-remote-storage) |
| **Vercel Sandbox — proxy'li mount** | `mount-s3 --no-sign-request` + firewall `forwardURL` → imzalayan Function | Mountpoint for S3 | **Hayır** | aynı sayfa, "Mount an S3 bucket without exposing credentials" |
| **Cloudflare Sandbox** | `sandbox.mountBucket()` | s3fs (`s3fsOptions` alanı açıkta) | **Hayır** (R2 binding veya `credentialProxy: true`) / Evet (düz `credentials`) | [Cloudflare Sandbox — Storage](https://developers.cloudflare.com/sandbox/api/storage/) |
| **Fly.io Sprites** | Nesne deposu **blok cihaz** olarak kernel'e sunuluyor, üstünde ext4 | yok (FUSE değil) | Yok — sandbox S3'ü hiç görmüyor | [Fly Sprites](https://fly.io/sprites/) |
| **Anthropic code execution** | **Yok.** Container'ın interneti kapalı; dosyalar Files API'den | — | — | [Code execution tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool) |
| **OpenAI Code Interpreter** | **Yok.** Container files uçları üzerinden | — | — | [Code interpreter](https://developers.openai.com/api/docs/guides/tools-code-interpreter) |

### 2.2 Dört aile

Tablo dört mimariye ayrışıyor:

**Aile A — Sandbox içinde FUSE + kimlik bilgisi.** E2B, Daytona external,
Vercel düz mount. En yaygın ve en zayıf. Vercel'in kendi changelog kodundaki
yorum bunu tek cümlede itiraf ediyor:

> `// Pass aws credentials only to the mount-s3 command.`
> `// Note: this does expose the credentials permanently in the sandbox!`
> `// Use a restricted role only`
> — [Vercel changelog, 3 Temmuz 2026](https://vercel.com/changelog/vercel-sandbox-now-supports-fuse-based-filesystems)

Cloudflare aynı şeyi doküman düzyazısında söylüyor:

> "When you mount with explicit credentials, s3fs writes those credentials to a
> password file on the container's disk. A compromised container process can
> read and exfiltrate the credentials, or use them to access storage outside the
> intended bucket scope."
> — [Cloudflare, Mount buckets](https://developers.cloudflare.com/sandbox/guides/mount-buckets/)

**Aile B — Sandbox içinde FUSE, kimlik bilgisi dışarıda.** Vercel'in proxy'li
mount'u ve Cloudflare'ın `credentialProxy: true`/R2-binding modu. Sandbox
**imzasız** S3 isteği gönderiyor, ağ katmanında bir bileşen isteği yakalayıp
gerçek kimlikle yeniden imzalıyor. §3.1'de ayrıntılı — bu dokümanın ana bulgusu.

**Aile C — Mount'u platform yapıyor.** Daytona Volumes, Kubernetes CSI
sürücüleri. Sandbox bir dizin görüyor; anahtarları hiç görmüyor. İzolasyon
sınırı mount noktasında:

> "Isolation is enforced at the FUSE mount boundary. Each sandbox sees its
> assigned subpath as the volume root, so a sandbox mounted at `users/alice`
> cannot reach `users/bob` through relative paths such as `../bob`."
> — [Daytona Volumes](https://www.daytona.io/docs/en/volumes/)

**Aile D — Doğrudan erişim yok.** Anthropic ve OpenAI. Anthropic'inki PTC'nin
bugünkü modelinin birebir aynısı:

> "The container has no internet access, so Claude can't download packages at
> runtime" · "**Internet access:** Completely disabled for security" ·
> "**External connections:** No outbound network requests permitted" ·
> "**Workspace scoping:** Like the Files API, containers are scoped to the
> request's workspace"
> — [Code execution tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)

Sayılar: container 5 GiB RAM + 5 GiB disk; container verisi 30 güne kadar
saklanıyor. Files API tarafında dosya başına **500 MB**, kuruluş başına **1 TB**,
kuruluş başına en fazla **100 workspace**
([Files API](https://platform.claude.com/docs/en/build-with-claude/files)).
OpenAI tarafında container bellek seçenekleri `1g` (varsayılan) / `4g` / `16g` /
`64g`, 20 dakika kullanılmazsa container düşüyor, 100 RPM. **OpenAI dokümanında
container'ın internet erişimi olup olmadığına dair açık ifade bulunamadı.**

**Fly.io Sprites bir beşinci nokta:** nesne deposunu *sandbox'a göstermek* yerine
*altına koymak*. "It presents an object storage bucket to the kernel as a real
block device and runs ext4 on top" — sandbox POSIX görüyor, 100 GB, S3 API'si
hiç ortada yok. Bu, "doğrudan erişim" sorusunu tamamen ortadan kaldıran ama
karşılığında **artifact semantiğini de** ortadan kaldıran bir tasarım: elde
bucket değil, disk var.

### 2.3 FUSE ayrıcalığı sorunu ve üç çözümü

FUSE mount ayrıcalık ister. Sahada üç farklı yerde çözülüyor:

| Çözüm | Kim | Ayrıcalık nerede | Kanıt |
|---|---|---|---|
| **microVM içinde `sudo`** | E2B, Vercel, Daytona | Sandbox'ın *kendi* çekirdeğinde — host'a dokunmuyor | "Because FUSE runs as a system-privileged process, the install and mount commands run with `sudo`" ([Vercel](https://vercel.com/docs/sandbox/mount-remote-storage)) |
| **Ayrı sidecar/pod** | GKE gcsfuse CSI, Mountpoint CSI v2 | Sandbox'ın *dışında* | aşağıda |
| **Privileged node daemon** | `k8s-csi-s3` (GeeseFS) | Node'da, privileged | "Kubernetes has to allow privileged containers" ([README](https://github.com/yandex-cloud/k8s-csi-s3/blob/master/README.md)) |

Kubernetes tarafında en öğretici olan **Mountpoint for Amazon S3 CSI Driver
v2 mimarisi**: FUSE süreci artık iş yükü pod'unun içinde değil, aynı node'da
**ayrı bir "Mountpoint Pod"da** çalışıyor; iş yükü pod'una yalnızca bir `bind`
mount veriliyor. Kimlik bilgisi (service account token'ları) node bileşeni
tarafından **Mountpoint Pod'un** credential path'ine yazılıyor:

> "`NodePublishVolume` … This method also provides AWS credentials for Mountpoint
> instances." · "To support sharing Mountpoint Pods, we create `bind` mounts to
> target Mountpoint Pods from each workload in this method"
> — [ARCHITECTURE.md](https://github.com/awslabs/mountpoint-s3-csi-driver/blob/main/docs/ARCHITECTURE.md)

GKE'nin GCS FUSE CSI sürücüsü aynı sonuca sidecar ile varıyor ve bunu açıkça
güvenlik gerekçesiyle sunuyor:

> "The Cloud Storage FUSE CSI driver does not need privileged access. This
> minimizes the risks associated with privileged access and leads to a better
> security posture."
> — [GKE Cloud Storage FUSE CSI driver](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/cloud-storage-fuse-csi-driver)

**PTC için doğrudan sonuç:** Kubernetes'te sandbox pod'una `/dev/fuse` ya da
`CAP_SYS_ADMIN` vermek **gerekmiyor** — endüstri deseni, ayrıcalıklı FUSE
sürecini sandbox'ın dışına almak. Bu, güvenilmeyen kod çalıştıran bir pod için
tek kabul edilebilir yol; OpenShift'te varsayılan `restricted-v2` SCC zaten
privileged container'a izin vermiyor
([Red Hat — Managing SCC](https://docs.redhat.com/en/documentation/openshift_container_platform/4.17/html/authentication_and_authorization/managing-pod-security-policies)).

### 2.4 Kubernetes: dört seçenek

| Seçenek | Kimlik kapsamı | Sandbox anahtarı görür mü | Olgunluk | Not |
|---|---|---|---|---|
| `mountpoint-s3-csi-driver` (`authenticationSource: pod`) | Pod'un ServiceAccount'u → IRSA / EKS Pod Identity | Hayır | GA, AWS resmî | Yalnızca AWS S3; MinIO desteklenmiyor |
| `gcsfuse.csi.storage.gke.io` | KSA → Workload Identity | Hayır | GA, Google resmî | Yalnızca GCS; k8s `agent-sandbox` projesinin tek nesne-deposu rehberi bu ([agent-sandbox FUSE CSI](https://agent-sandbox.sigs.k8s.io/docs/volumes/gcsfuse-csi/)) |
| `k8s-csi-s3` (GeeseFS/s3fs/rclone) | StorageClass'a bağlı Secret | Hayır (mount node'da) | Topluluk | **Privileged container gerekiyor** |
| **COSI** (`objectstorage.k8s.io`) | `BucketAccess` → per-workload credential | **Evet** — Secret pod'a mount ediliyor | `v1alpha1`/`v1alpha2` | Aşağıda |

COSI'nin akışı: `BucketClaim` bucket'ı sağlar, `BucketAccess` erişimi verir,
sonuç bir Secret'ta `BucketInfo` anahtarı altında toplanır ve **uygulama pod'una
mount edilir** ([COSI docs](https://container-object-storage-interface.sigs.k8s.io/)).
`BucketAccessClass.authenticationType` `Key` olduğunda bu, sandbox'ın okuyabildiği
statik bir anahtar demek. Yani COSI *provisioning*'i çözüyor, *güvenilmeyen kod*
problemini çözmüyor — ve hâlâ alpha.

---

## 3. S2 — Kimlik bilgisi kapsamlaması (en kritik bölüm)

### 3.1 Ana bulgu: sıfır-kimlik-bilgisi mount + imzalayan egress proxy

İki bağımsız sağlayıcı, **birbirinden habersiz olamayacak kadar aynı** tasarıma
varmış. Desen şu:

```
Sandbox                        Ağ katmanı                     Nesne deposu
  mount-s3 --no-sign-request      yakala                         S3
  (ya da s3fs dummy creds)   →    gerçek kimlikle SigV4 imzala →
  imzasız S3 isteği               kapsamı burada DENETLE
```

**Vercel** — `defineSandboxProxy` + firewall `forwardURL` + OIDC federasyonu:

> "The sandbox sends unsigned S3 requests, the firewall forwards them to a Vercel
> Function you control, and the function authorizes each request and signs it
> with Signature Version 4 before it reaches S3. **AWS credentials exist only
> inside the function which decides what each sandbox can access.**"

Ve yetkilendirmenin nerede yapılacağını kod yorumunda gösteriyor:

> "// Authorize before signing. With path-style requests the path is
> `/<bucket>/<key>`, so you can restrict each sandbox to its own key prefix,
> for example one derived from `meta.sandboxId`."
> — [Vercel — Mount remote storage](https://vercel.com/docs/sandbox/mount-remote-storage)

Mount komutu: `mount-s3 <bucket> /mnt/s3 --allow-other --no-sign-request
--force-path-style --endpoint-url https://s3.amazonaws.com`. Ağ politikası
`s3.amazonaws.com` → `forwardURL` ile kilitleniyor; başka egress yok.

**Cloudflare** — `credentialProxy: true`:

> "Instead of passing real credentials into the container, the Durable Object
> intercepts all outbound S3 requests at the network layer, re-signs them with
> the real credentials, and forwards them upstream. **The container only ever
> holds dummy credentials that are useless outside the proxy.**"

Ayrıca R2 binding modu hiç kimlik bilgisi kullanmıyor: "Uses credential-less
egress interception for R2". Doküman bu seçeneği tavsiye ediyor ve gelecekte
varsayılan yapacağını söylüyor: "It is recommended to set `credentialProxy: true`
for all endpoint mounts. The option defaults to `false` for backwards
compatibility and will become the default in a future version."
— [Cloudflare — Storage](https://developers.cloudflare.com/sandbox/api/storage/)

**Bu desen neden PTC için önemli:** gateway *aradan çıkmıyor*, **katman
değiştiriyor**. MCP seviyesinde değil, S3 protokolü seviyesinde araya giriyor.
Sandbox POSIX dosya sistemi görüyor (LLM'in yazdığı sıradan kod çalışıyor), ama
her `PUT`/`GET` hâlâ bizim yazdığımız bir bileşenden geçiyor.

### 3.2 Klasik kapsamlama mekanizmaları ve gerçek limitleri

| Mekanizma | En dar kapsam | Süre | Sert limitler | Kaynak |
|---|---|---|---|---|
| **STS `AssumeRole` + session policy** | Politikanın yazabildiği kadar (prefix dâhil) | 900 sn – 43 200 sn (12 sa); rol zincirlemede **en fazla 1 sa** | Inline + managed session policy **düz metni 2 048 karakteri aşamaz**; en fazla **10** managed policy ARN'i; ayrıca ayrı bir "packed" limit (`PackedPolicySize` %100'ü aşarsa `PackedPolicyTooLarge`) | [AssumeRole API](https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html) |
| **S3 Access Grants** | Bucket / prefix / **tek nesne**, READ·WRITE·READWRITE | Varsayılan 1 sa, 15 dk – 12 sa | Instance başına bölge/hesap sınırı var (dokümanda ayrı sayfaya atıf) | [Access Grants concepts](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-grants-concepts.html) |
| **S3 Access Points** | Access point policy + VPC kısıtı; bucket policy ile birlikte çalışır | — | Yalnızca **nesne** işlemleri; bucket silme, replication gibi işlemler yapılamaz | [Access points](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-points.html) |
| **MinIO STS `AssumeRole`** | Inline `Policy` ile scope-down; sonuç, canned policy ∩ session policy | 900 sn – **31 536 000 sn (365 gün)**, varsayılan 3 600 | `Policy` uzunluğu **en fazla 2 048** | [minio/docs/sts/assume-role.md](https://github.com/minio/minio/blob/master/docs/sts/assume-role.md) |

Session policy semantiği her iki üründe de aynı ve PTC'nin capability modeliyle
birebir örtüşüyor:

> "The permissions for a session are the **intersection** of the identity-based
> policies for the IAM entity … and the session policies. … You cannot use
> session policies to grant more permissions than those allowed by the
> identity-based policy of the role that is being assumed."
> — [IAM — Session policies](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html#policies_session)

Yani: **yetki zayıflatılabilir, güçlendirilemez.** Bu, PTC'nin HMAC kapsam
jetonuyla aynı kural. STS session policy, kapsam jetonunun depo tarafındaki
karşılığı olarak okunabilir.

**Pratik uyarı:** 2 048 karakterlik sınır, "her çalıştırma için `workflow_id`
prefix'ine kilitli bir session policy üret" tasarımını mümkün kılıyor (böyle bir
politika 300–400 karakter), ama çok sayıda prefix'i tek politikaya sığdırmayı
engelliyor.

### 3.3 Çok kiracılılık: ne engelliyor komşu prefix'i okumayı

Sahada dört farklı sınır kullanılıyor. Sertlikleri farklı:

| Sınır | Kim kullanıyor | Zorlayan kim | Sertlik |
|---|---|---|---|
| **Mount noktası (subpath)** | Daytona Volumes | Platform'un FUSE katmanı | Sandbox içinden aşılamaz (`../bob` çalışmıyor) — ama platforma güven gerektirir |
| **Prefix + IAM/session policy** | S3 Access Grants, MinIO STS, Vercel proxy | Depo | Sert; politika doğru yazıldığı sürece |
| **Ayrı bucket / ayrı hesap** | mountpoint CSI cross-account örneği | IAM + trust policy | En sert, en pahalı |
| **Ayrı "workspace"** | Anthropic Files API | Platform | Anthropic bunu *tek* izolasyon sınırı ilan ediyor |

Anthropic'in uyarısı, PTC'nin opaque-handle tasarımıyla birebir aynı tuzağa
işaret ediyor ve alıntılamaya değer:

> "**Uploaded files are accessible to your entire workspace, not scoped to an end
> user, conversation, or session.** … **Never accept `file_id` values from end
> users or other untrusted sources**: a user-supplied file ID would let one user
> of your application read content that another user uploaded. … If you are
> building a multi-tenant application on the Files API, create a separate
> workspace for each tenant."
> — [Files API](https://platform.claude.com/docs/en/build-with-claude/files)

Bu, PTC'de `artifact_id`'nin **tek başına yetki taşımaması** kuralının dış
doğrulaması: handle opak olsa bile, kapsam kontrolünü *çağıranın beyanına* değil
imzalı jetona dayamak zorunludur.

### 3.4 Kubernetes'te pod-başına kimlik: mümkün, ama granülarite ServiceAccount

AWS'in `mountpoint-s3-csi-driver` dokümanı pod-level ve driver-level kimliği
açıkça ayırıyor:

> "By setting driver-level credentials, the whole cluster uses the same set of
> credentials."
>
> "You can configure Mountpoint CSI Driver to use the credentials associated with
> the pod's Service Account rather than the driver's own credentials. With this
> approach, **a multi-tenant architecture is possible** using EKS Pod Identity or
> IRSA."
> — [CONFIGURATION.md](https://github.com/awslabs/mountpoint-s3-csi-driver/blob/main/docs/CONFIGURATION.md)

Açılışı `authenticationSource: pod` (PV'nin `volumeAttributes`'ında) yapıyor.
İki sert kısıt var:

> "If you configure a driver-level credential source when using
> `authenticationSource: pod`, **it will be ignored**."
>
> "**Only EKS Pod Identity and IRSA are supported with Pod-Level credentials.**
> You cannot configure Kubernetes secrets or use instance profiles."

IRSA ile kullanırken sürücünün STS bölgesini bilmesi gerekiyor (`stsRegion`
volume attribute'u, ya da IMDS ile otomatik tespit — bunun için
`HttpPutResponseHopLimit ≥ 2`).

**Ama izolasyonun gerçek granülaritesi ServiceAccount, çalıştırma değil.**
Mountpoint Pod paylaşımı şu koşullarda devreye giriyor:

> "- Workloads are scheduled on the same node
> - Workloads use the same volume (same PV name and volume ID)
> - Workloads use the same mount options
> - Workloads use the same authentication source (`driver` or `pod`)
> - Workloads have the same FSGroup …
> - For pod-level identity, workloads must also have: **the same namespace, the
>   same service account name, the same IAM role ARN**"
> — [MOUNTPOINT_POD_SHARING.md](https://github.com/awslabs/mountpoint-s3-csi-driver/blob/main/docs/MOUNTPOINT_POD_SHARING.md)

Yani aynı SA ile koşan iki sandbox **aynı Mountpoint örneğini ve aynı kimliği**
paylaşır. PTC'de sandbox pod'ları efemer ve workflow başına farklı; her workflow
için ayrı ServiceAccount + ayrı IAM rolü üretmek gerekirdi. Bu, PTC'nin bugün
HMAC jetonuyla **çalıştırma anında** çözdüğü şeyi, **küme nesnesi yaratma**
seviyesine taşımak demek — 3,14 saniyelik bir pod ömrü için ağır.

### 3.5 Karşılaştırma: PTC'nin kapsam jetonu vs. alternatifler

| Mekanizma | Kapsam nesnesi | Kim üretiyor | Sandbox anahtarı görür mü | PTC'ye uyum |
|---|---|---|---|---|
| **Bugünkü: HMAC kapsam jetonu** | `workflow_id` | `sandbox_runner`, imza anahtarı Secret'ta | Jetonu görür, imza anahtarını görmez | Referans |
| Uzun ömürlü bucket anahtarı | Bucket | Elle / OBC | **Evet** | ✗ kabul edilemez |
| STS session policy (MinIO) | Prefix | Gateway/runner | Evet (geçici, dar) | ~ mümkün, §5'e bak |
| Vercel/Cloudflare tarzı imzalayan proxy | Prefix, istek başına | Gateway | **Hayır** | ✓ en iyi uyum |
| mountpoint CSI `authenticationSource: pod` | ServiceAccount | Küme | Hayır | ~ granülarite kaba, AWS'e bağlı |
| COSI `BucketAccess` | Bucket | COSI sürücüsü | **Evet** (Secret mount) | ✗ güvenilmeyen kod için uygun değil |
| Daytona tarzı platform mount + subpath | Prefix | Platform | Hayır | ✓ ama kendi FUSE katmanımızı yazmak demek |

---

## 4. S3 — Doğrudan yazınca kayıt defterine ne oluyor

### 4.1 Tespit: hiçbiri kayıt defteri tutmuyor

Taranan sağlayıcıların **hiçbirinin** bucket-mount yolunda bir artifact
registry'si yok. Mount edilen bucket'a yazılan dosyanın `artifact_id`'si,
lineage'i, TTL'i, content-hash'i yok — sadece bir S3 anahtarı var.
Dokümanlar bunu "persistent data access", "share state across Sandboxes",
"stream large datasets" diye pazarlıyor; hiçbiri "artifact" demiyor.

Tek istisna **Anthropic**: orada zaten mount yok, Files API var ve o bir kayıt
defteri (`id`, `filename`, `mime_type`, `size_bytes`, `created_at`,
`downloadable`, `expires_at`). Yani kayıt defteri, **yalnızca yazma yolu bir
bileşenden geçtiğinde** ayakta kalıyor.

Bu, PTC_Mimari §3'teki tezi dışarıdan doğruluyor: *"Bu ayrım olmadan elde
artifact store değil, sadece bir bucket olur."* Doğrudan mount = tam olarak
"sadece bucket".

### 4.2 Gerilim: kayıt defterini geri kazanmanın üç yolu ve üçünün de bedeli

| Yol | Nasıl | Bedel |
|---|---|---|
| **Olay bildirimi** | S3 Event Notifications → SQS/SNS/Lambda/EventBridge; MinIO'da webhook | **At-least-once**, sıra garantisi yok, gecikme belirsiz |
| **Periyodik tarama** | Bucket'ı listele, kayıt defteriyle karşılaştır | Maliyetli; "şu an tutarlı mı" sorusuna cevap vermiyor |
| **Hiç** | — | Artifact store değil, bucket |

Olay bildiriminin garantisi AWS'in kendi dokümanında net:

> "Amazon S3 event notifications are designed to be **delivered at least once**.
> Typically, event notifications are delivered in seconds but can sometimes take
> **a minute or longer**."
> — [S3 Event Notifications](https://docs.aws.amazon.com/AmazonS3/latest/userguide/EventNotifications.html)

Bunun PTC için anlamı somut: `cached("tarama", pahali_fn)` çağrısı, bir önceki
adımın yazdığı artifact'i **bulamayabilir**, çünkü kayıt defteri bildirim
gelmeden güncellenmemiş olur. Yani doğrudan yazma, PTC'nin en çok işe yarayan
özelliğini (Okuma A — içerik-adresli cache) doğrudan bozar. `list_artifacts`
de aynı şekilde eksik cevap verir.

MinIO tarafında karşılığı bucket notification + webhook
([minio/docs/bucket/notifications](https://github.com/minio/minio/tree/master/docs/bucket/notifications)),
aynı asenkron doğaya sahip.

**Periyodik tarama teorik değil — adı konmuş bir bakım işi.** Apache Iceberg'in
`remove_orphan_files` prosedürü tam olarak bu problemi çözmek için var:

> "Used to remove files which are not referenced in any metadata files of an
> Iceberg table and can thus be considered 'orphaned'."

`older_than` varsayılanı **3 gün önce** — yani "şu an yazılmakta olan dosyayı
yetim sanma" diye konmuş bir emniyet payı
([Iceberg spark-procedures](https://iceberg.apache.org/docs/latest/spark-procedures/)).
Kayıt defterinin dışından yazılan her sistemin sonunda böyle bir uzlaştırıcı işe
ihtiyacı oluyor; ve o iş, tutarsızlığı *gün* mertebesinde bir gecikmeyle kapatıyor.

### 4.3 Seçenekler

| Seçenek | Kayıt defteri | Cache doğruluğu | Karmaşıklık |
|---|---|---|---|
| Bugünkü: gateway yazar | Senkron, kesin | Kesin | Düşük |
| Mount + S3 event → gateway webhook | Asenkron, at-least-once | **Bozulur** (sn–dk pencere) | Orta |
| Mount + periyodik tarama | Gecikmeli | Bozulur (dk–saat) | Orta |
| Mount + imzalayan proxy (§3.1) | **Senkron** — proxy `PUT`'u görüyor | Kesin | Orta-yüksek |

Son satır, imzalayan proxy deseninin ikinci büyük faydası: **kimlik bilgisini
gizlemek için kurulan bileşen, kayıt defterini de senkron tutabiliyor**, çünkü
her yazma zaten oradan geçiyor.

---

## 5. S4 — İçerik doğrulaması gateway kalkınca nereye gidiyor

### 5.1 Bulgu: yazma yolunda hazır bir mekanizma **yok**

**S3 Object Lambda yazma yolunu kapsamıyor.** AWS'in kendi tanımı:

> "With Amazon S3 Object Lambda, you can add your own code to Amazon S3 `GET`,
> `LIST`, and `HEAD` requests to modify and process data as it is returned to an
> application." · "**All other requests are processed as normal.**"

Üstelik hizmet yeni müşterilere kapatılmış:

> "As of November 7th, 2025, S3 Object Lambda is available only to existing
> customers that are currently using the service as well as to select AWS
> Partner Network (APN) partners."
> — [Transforming objects with S3 Object Lambda](https://docs.aws.amazon.com/AmazonS3/latest/userguide/transforming-objects.html)

Yani "okuma yolunda araya girme" mekanizması (a) PUT'u yakalayamıyor, (b)
kapanıyor. PTC'nin pickle reddi bir **yazma** kontrolü; Object Lambda ile
yapılamazdı.

**Bucket policy ile boyut/tip kısıtlaması:** `content-length-range` koşulu
**yalnızca POST policy** içinde tanımlı — tarayıcı tabanlı form yüklemesi için.
Bir IAM/bucket policy condition key'i olarak nesne boyutunu sınırlayan bir
anahtar **bu araştırmada bulunamadı**; `s3:object-size` gibi bir anahtarın
varlığı doğrulanamadı (bkz. §7). Dolayısıyla PTC'nin "boyut sınırı" kontrolü de
düz bucket policy'ye devredilemiyor.

### 5.2 Sahada içerik denetimi nerede yapılıyor

| Sistem | Yazma anında | Okuma anında | Not |
|---|---|---|---|
| E2B / Daytona external / Vercel düz mount | **Hiç** | Hiç | s3fs/mount-s3 baytı geçirir |
| Cloudflare `credentialProxy` | Proxy isteği görür (doküman denetimden söz etmiyor, imzalamaya odaklı) | — | Kanca mevcut, kullanım örneği yok |
| Vercel imzalayan proxy | **Var** — "add your own authorization checks in the handler before signing" | — | Yetkilendirme için; içerik denetimi de aynı noktada mümkün |
| Anthropic | Files API + `$OUTPUT_DIR` yakalaması | — | Tek kapı |
| PTC (bugün) | Gateway: isim, boyut, pickle imzası | — | Referans |

Vercel'in cümlesi bu noktada kritik:

> "Since the proxy example forwards each request unchanged, **add your own
> authorization checks in the handler before signing**. For example, restrict
> each sandbox to its own key prefix."

Bu, "imzalamadan önce isteği incele" davetidir. `PUT /bucket/key` gövdesinin ilk
baytlarına bakıp pickle sihirli baytını reddetmek de tam olarak orada yapılır.
Yani PTC'nin dört kontrolünden üçü (isim, boyut, format) **S3-protokol
katmanında da uygulanabilir** — MCP katmanında olmaları zorunlu değil.

### 5.3 Denetim noktası karşılaştırması

| Denetim noktası | İsim/yol geçişi | Boyut | pickle reddi | Kapsam (workflow) | Kayıt defteri |
|---|---|---|---|---|---|
| MCP gateway (bugün) | ✓ | ✓ | ✓ | ✓ | ✓ senkron |
| S3-protokol imzalayan proxy | ✓ (anahtar denetimi) | ✓ (`Content-Length`) | ✓ (gövde ön-eki) | ✓ (prefix) | ✓ senkron |
| Bucket policy + STS session policy | ✓ (prefix) | ✗ | ✗ | ✓ | ✗ |
| S3 Object Lambda | ✗ | ✗ | yalnızca okuma yolunda | ✗ | ✗ |
| Ham mount + statik anahtar | ✗ | ✗ | ✗ | ✗ | ✗ |

---

## 6. S5 — PTC için pratik sonuç

### 6.1 "Doğrudan erişim"e geçersek ne kaybederiz

| PTC kontrolü | Ham mount + anahtar | Mount + STS session policy | Mount + imzalayan proxy |
|---|---|---|---|
| Kapsam (workflow izolasyonu) | **Kaybolur** | Korunur (prefix) | Korunur (prefix, istek başına) |
| İsim / yol geçişi | **Kaybolur** | Kısmen (prefix dışına yazamaz) | Korunur |
| Boyut sınırı | **Kaybolur** | **Kaybolur** | Korunur |
| pickle reddi | **Kaybolur** | **Kaybolur** | Korunur |
| Content-hash dedup | **Kaybolur** | **Kaybolur** | Korunur (proxy hash'leyebilir) |
| Kayıt defteri / lineage / TTL | **Kaybolur** | **Kaybolur** | Korunur |
| Tip korunumu (Parquet/Arrow) | Kaybolur | Kaybolur | Kaybolur (dosya sistemi semantiği) |
| Egress sınırı ("tek hedef gateway") | **Kaybolur** — depo ikinci çıkış olur | Kaybolur | Korunur (proxy tek hedef) |

En alttaki satır belirleyici: PTC'nin kurucu iddiası "sandbox yalnızca Tool
Gateway'e çıkabilir". Sandbox'a MinIO'ya rota vermek, [PTC_Artifact_Persistence
§6.3](PTC_Artifact_Persistence_Arastirmasi.md)'te presigned URL için yazılan
itirazın aynısını doğurur: **gateway'in görmediği ikinci bir çıkış**. İmzalayan
proxy deseni bu itirazı doğuran şeyi ortadan kaldırıyor, çünkü hedef hâlâ tek.

### 6.2 Melez desenler sahada var mı? **Evet — ve tek bir eşik kuralı değil, üç ayrı melez**

Soru "küçük dosyalar gateway'den, büyükler doğrudan" diye soruldu. Sahada bunun
karşılığı var, ama beklenenden farklı bir eksende bölünüyor:

| Melez | Kim | Bölünme ekseni | Kaynak |
|---|---|---|---|
| **SDK dosya API'si + bucket mount** | E2B, Daytona, Cloudflare | Küçük/kontrol dosyaları SDK'nın `files.write`/`readFile` yolundan; büyük veri kümeleri mount'tan | Cloudflare aynı sayfada iki yolu da sunuyor; Daytona hem `fs.upload_file` hem mount belgeliyor |
| **Platform volume + harici bucket mount** | Daytona (Volumes ↔ External storage), Vercel (Drives ↔ FUSE) | Verinin *kimin hesabında* durduğuna göre | "Use a FUSE mount when your data must live in an external provider such as S3" ([Vercel Drives](https://vercel.com/docs/sandbox/concepts/drives)) |
| **Kimlik bilgisiz mount + imzalayan proxy** | Vercel, Cloudflare | *Bayt* mount'tan, *yetki* proxy'den | §3.1 |

Üçüncüsü, "küçük/büyük" eşiğini tamamen gereksiz kılan tasarım: bayt akışı
dosya sistemi hızında kalıyor, karar hâlâ merkezde veriliyor.

Vercel'in Drives limitleri, ikinci melezin sınırlarını da gösteriyor: sandbox
başına en fazla **4 drive**, drive başına varsayılan **100 GiB** (1 TiB'a kadar),
tek bölgeye bağlı, **tek okuyucu-tek yazıcı**, ve "we recommend using drives for
caching and other non-critical use cases during the private beta period".

### 6.3 kind / OpenShift üzerinde gerçekçilik

| Seçenek | kind'de | OpenShift'te | Değerlendirme |
|---|---|---|---|
| Sandbox içinde `s3fs`/`mount-s3` + anahtar | `/dev/fuse` + privileged gerekir | `restricted-v2` SCC engeller; özel SCC gerekir | ✗ Güvenilmeyen kod için baştan elenir |
| `mountpoint-s3-csi-driver` | ✗ AWS S3'e özgü, MinIO yok | ✗ | ✗ |
| `k8s-csi-s3` (GeeseFS) | Çalışır, ama **privileged** node daemon | Özel SCC gerekir | ~ Kayıt defteri yine yok |
| COSI + ODF/NooBaa | Alpha; Secret pod'a iner | Aynı | ✗ Anahtar sandbox'ta |
| **Gateway'i S3-protokol proxy'sine dönüştür** | Çalışır — sandbox `--endpoint-url http://gateway:PORT` ile mount eder, ağ politikası değişmez | Aynı | **✓ Önerilen** |
| Bugünkü MCP gateway | Çalışıyor | Çalışacak | ✓ Referans |

**Önerilen yön (uygulanmadı, PoC yok):** mevcut Tool Gateway'e, MCP tool'larının
*yanına*, dar bir S3-uyumlu uç eklemek. Sandbox `mount-s3`/`s3fs` ile bu uca
imzasız bağlanır (Vercel'in `--no-sign-request --force-path-style --endpoint-url`
üçlüsü MinIO ile de çalışır, çünkü Mountpoint'in kendisi S3-uyumlu uçları
destekliyor); gateway isteği kapsam jetonundan okuduğu prefix'e göre yetkilendirir,
`Content-Length`'e bakar, `PUT` gövdesinin ilk baytlarını pickle imzasına karşı
denetler, kendi MinIO kimliğiyle imzalar, ve kayıt defterine **senkron** yazar.

Kazanç: LLM `df.to_parquet("/artifacts/x.parquet")` yazabilir — `put_artifact`
API'sini bilmesine gerek kalmaz (PTC_Mimari §7.1'deki `/output` süpürmesinin
gerçek zamanlı hâli). Kayıp: tip korunumu (Parquet baytları geçer ama "bu bir
DataFrame'dir" bilgisi dosya adından tahmin edilir) ve gateway'in `pandas`
içermeme garantisi korunur ama artık S3 protokolü ayrıştırmak zorundadır —
bu, yeni bir saldırı yüzeyi.

**Bunu bugün yapmamak için de sağlam bir gerekçe var:** PTC'de sandbox 3,14 sn
yaşıyor ve tek script çalıştırıyor. Mount'un çözdüğü problem ("POSIX bekleyen
araçları değiştirmeden çalıştırmak", "büyük veri kümesini indirmeden akıtmak")
PTC'de henüz yok. §6.1 tablosundaki kayıplar gerçek, kazanç ise şu an teorik.

---

## 7. Doğrulanamayanlar ve çelişkiler

| Konu | Durum |
|---|---|
| **Modal — secret container'a enjekte ediliyor mu** | `CloudBucketMount(secret=…)` imzası doğrulandı; secret'ın container ortamına da geçip geçmediği dokümanda **açıkça yazmıyor**. "Mount'a özel" olduğu çıkarım, garanti değil. |
| **Modal — R2 mount teknolojisi** | Doküman "built on top of AWS' mountpoint technology" diyor; S3 için Mountpoint kesin, R2 için aynı sürücünün kullanıldığı ifadesi tek bir cümleye dayanıyor, ayrıca doğrulanamadı. |
| **Modal — Sandbox + CloudBucketMount birlikte** | `Sandbox.create(..., volumes: dict[str \| os.PathLike, _Volume \| _CloudBucketMount])` imzasından doğrulandı; ancak rehber sayfası Sandbox'tan hiç söz etmiyor. İki sayfa arasında bir boşluk var. |
| **E2B — geçici kimlik bilgisi (`sessionToken`) desteği** | Bir üçüncü taraf doküman "yalnızca statik `accessKeyId`/`secretAccessKey`/`sessionToken` destekleniyor, yenileme dışarıda halledilmeli" diyor. **E2B'nin kendi dokümanında bu ifade yok**; E2B sayfası yalnızca `.passwd-s3fs` örneğini gösteriyor. Birincil kaynak bulunamadı, iddia kullanılmadı. |
| **OpenAI Code Interpreter — ağ erişimi** | Resmî doküman container'ın internet erişimi olup olmadığını **belirtmiyor**. Anthropic'in aksine açık bir ifade yok. Varsayım yapılmadı. |
| **`s3:object-size` benzeri bir IAM condition key** | AWS'in `service-authorization` referans sayfası JS ile render edildiği için ham olarak çekilemedi. `content-length-range`'in POST policy'ye özgü olduğu doğrulandı; nesne boyutunu sınırlayan bir bucket-policy anahtarının **yokluğu** doğrudan bir cümleyle kanıtlanamadı. "Bulunamadı" olarak yazıldı, "yok" olarak değil. |
| **Cloudflare — Sandbox security model sayfası** | Navigasyonda listeleniyor ama `/sandbox/concepts/security-model/` 404 dönüyor. İçerik alınamadı. |
| **S3 Access Grants / Access Points sayısal limitleri** | Her iki doküman da limitleri ayrı sayfaya havale ediyor; o sayfalar çekilmedi. **Kaynakta rakam alınmadı.** |
| **Fly.io Sprites — bucket mount primitifi** | Sprites dokümantasyonunda nesne deposu *mount eden* bir primitif bulunamadı. Bulunan tek şey "S3 Block Device" (early access): bucket, kernel'e blok cihaz olarak sunuluyor. Bunlar farklı şeyler; karıştırılmamalı. |
| **Daytona Volumes'ün altındaki FUSE sürücüsü** | Doküman "FUSE-based" diyor, hangi sürücü olduğunu söylemiyor. |

---

## 8. Kaynaklar

### Agent sandbox sağlayıcıları — nesne deposu erişimi
- [E2B — Cloud buckets (connect-bucket)](https://e2b.dev/docs/sandbox/connect-bucket)
- [E2B — Internet access (ağ politikası)](https://e2b.dev/docs/sandbox/internet-access)
- [Modal — Cloud bucket mounts](https://modal.com/docs/guide/cloud-bucket-mounts)
- [Modal — `modal.CloudBucketMount` referansı](https://modal.com/docs/reference/modal.CloudBucketMount)
- [Modal — `modal.Sandbox` referansı](https://modal.com/docs/reference/modal.Sandbox)
- [Daytona — Volumes](https://www.daytona.io/docs/en/volumes/)
- [Daytona — Mount External Storage](https://www.daytona.io/docs/en/mount-external-storage/)
- [Vercel — Mount remote storage](https://vercel.com/docs/sandbox/mount-remote-storage)
- [Vercel — Sandbox firewall (credentials brokering, requests proxying)](https://vercel.com/docs/sandbox/concepts/firewall)
- [Vercel — Drives (private beta)](https://vercel.com/docs/sandbox/concepts/drives)
- [Vercel — Understanding Sandboxes](https://vercel.com/docs/sandbox/concepts)
- [Vercel changelog — Sandbox now supports FUSE-based filesystems (3 Tem 2026)](https://vercel.com/changelog/vercel-sandbox-now-supports-fuse-based-filesystems)
- [Cloudflare Sandbox — Storage API (`mountBucket`, `credentialProxy`)](https://developers.cloudflare.com/sandbox/api/storage/)
- [Cloudflare Sandbox — Mount buckets rehberi](https://developers.cloudflare.com/sandbox/guides/mount-buckets/)
- [Fly.io — Sprites (tiered storage, S3 Block Device)](https://fly.io/sprites/)
- [Fly.io — Agent Sandboxes (izolasyon tablosu)](https://fly.io/learn/agent-sandbox/)

### Gateway-only modeller (doğrudan erişim yok)
- [Anthropic — Code execution tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)
- [Anthropic — Files API (limitler, workspace izolasyonu)](https://platform.claude.com/docs/en/build-with-claude/files)
- [OpenAI — Code interpreter](https://developers.openai.com/api/docs/guides/tools-code-interpreter)

### Kimlik bilgisi kapsamlaması
- [AWS STS — `AssumeRole` API referansı](https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html)
- [AWS IAM — Policies and permissions (Session policies)](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html#policies_session)
- [Amazon S3 — Access Grants concepts](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-grants-concepts.html)
- [Amazon S3 — Managing access with access points](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-points.html)
- [MinIO — `AssumeRole` (minio/docs/sts/assume-role.md)](https://github.com/minio/minio/blob/master/docs/sts/assume-role.md)

### Kubernetes: CSI, COSI, FUSE ayrıcalığı
- [awslabs/mountpoint-s3-csi-driver — CONFIGURATION.md (pod-level vs driver-level)](https://github.com/awslabs/mountpoint-s3-csi-driver/blob/main/docs/CONFIGURATION.md)
- [awslabs/mountpoint-s3-csi-driver — ARCHITECTURE.md (Mountpoint Pod)](https://github.com/awslabs/mountpoint-s3-csi-driver/blob/main/docs/ARCHITECTURE.md)
- [awslabs/mountpoint-s3-csi-driver — MOUNTPOINT_POD_SHARING.md](https://github.com/awslabs/mountpoint-s3-csi-driver/blob/main/docs/MOUNTPOINT_POD_SHARING.md)
- [GKE — Cloud Storage FUSE CSI driver ("does not need privileged access")](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/cloud-storage-fuse-csi-driver)
- [kubernetes-sigs/agent-sandbox — FUSE CSI (GCS + Workload Identity)](https://agent-sandbox.sigs.k8s.io/docs/volumes/gcsfuse-csi/)
- [kubernetes-sigs/agent-sandbox — volumeClaimTemplates](https://agent-sandbox.sigs.k8s.io/docs/volumes/volume-claim-template/)
- [yandex-cloud/k8s-csi-s3 — README ("Kubernetes has to allow privileged containers")](https://github.com/yandex-cloud/k8s-csi-s3/blob/master/README.md)
- [COSI — Container Object Storage Interface dokümantasyonu](https://container-object-storage-interface.sigs.k8s.io/)
- [Red Hat — Managing security context constraints (OCP 4.17)](https://docs.redhat.com/en/documentation/openshift_container_platform/4.17/html/authentication_and_authorization/managing-pod-security-policies)

### Kayıt defteri ve içerik doğrulaması
- [Amazon S3 — Event Notifications (at-least-once, gecikme)](https://docs.aws.amazon.com/AmazonS3/latest/userguide/EventNotifications.html)
- [MinIO — Bucket notifications](https://github.com/minio/minio/tree/master/docs/bucket/notifications)
- [Amazon S3 — Transforming objects with S3 Object Lambda (GET/LIST/HEAD; 7 Kas 2025 kapanışı)](https://docs.aws.amazon.com/AmazonS3/latest/userguide/transforming-objects.html)
- [Apache Iceberg — Spark procedures (`remove_orphan_files`)](https://iceberg.apache.org/docs/latest/spark-procedures/)

### Proje içi
- [PTC_Mimari.md](PTC_Mimari.md)
- [PTC_Calistirma_Ortami_ve_SOTA_Arastirmasi.md](PTC_Calistirma_Ortami_ve_SOTA_Arastirmasi.md)
- [PTC_Artifact_Persistence_Arastirmasi.md](PTC_Artifact_Persistence_Arastirmasi.md)
