# AI Agent Sandbox ve Artifact Persistence Mimarisi

> Araştırma tarihi: 4 Eylül 2026  
> Kapsam: PTC / ReAct agent'lar, OpenShift, sandboxed code execution ve multi-step / multi-turn artifact persistence.

## 1. Executive Summary

Agent'ın ürettiği kodu çalıştırmak ile o çalışmanın çıktısını kalıcı olarak saklamak iki farklı problemdir.

En temiz mimari ayrım:

```text
                         USER / LLM
                              |
                              v
                       Agent / ReAct
                              |
                              v
                       Orchestrator
                         /       \
                        /         \
                       v           v
                Workflow State   Sandbox Manager
                                     |
                                     v
                              OpenShift Sandbox
                              (container / Kata)
                                     |
                              Python / Shell / Tools
                                     |
                              /workspace
                                     |
                              artifact.save()
                                     |
                                     v
                              Artifact Service
                                /          \
                               v            v
                         Metadata DB    Object Storage
                                        S3-compatible
                                      (ODF/NooBaa,
                                       Ceph RGW,
                                       MinIO, vb.)
```

Temel roller:

| Katman | Sorumluluk |
|---|---|
| Agent / ReAct | Hangi işi yapacağına ve sıradaki adıma karar verir |
| Orchestrator | Workflow, PTC execution, retry, lifecycle ve state'i yönetir |
| Sandbox | AI-generated/untrusted kodun izole biçimde çalışmasını sağlar |
| Artifact Service | Dosya/dataframe/model gibi çıktıları isimlendirir, metadata tutar ve storage'a yazar/okur |
| Object Storage | Artifact'in dayanıklı ve uzun ömürlü byte'larını saklar |
| Metadata DB / State Store | Artifact metadata'sını ve workflow state/checkpoint'lerini tutar |

Önemli sonuç:

> **Sandbox'ın yaşam süresi artifact'in yaşam süresini belirlememeli.**

Sandbox silinebilir, yeniden yaratılabilir veya başka bir sandbox ile değiştirilebilir. Kalıcı olması gereken çıktılar object storage'a aktarılmalıdır.

---

# 2. “Workflow” ne demek?

Workflow, tek bir PTC içindeki step'lerden oluşmak zorunda değildir.

Aşağıdakilerin tamamı workflow olabilir:

### Aynı PTC içindeki adımlar

```text
PTC #1
 ├── load data
 ├── clean
 ├── feature engineering
 └── save artifact
```

### Farklı PTC execution'ları

```text
PTC #1
   |
   v
artifact A
   |
PTC #2
   |
   v
artifact B
   |
PTC #3
```

### Node-based workflow

```text
Node A -> Node B -> Node C -> Node D
           |          |
          PTC        PTC
```

### Farklı turn'lere yayılan workflow

```text
Turn 1 -> PTC #1 -> artifact A
Turn 2 -> PTC #2 -> artifact B
Turn 3 -> PTC #3 -> artifact C
```

Dolayısıyla artifact persistence, özellikle workflow birden fazla execution veya turn'e yayıldığında önem kazanır.

---

# 3. Artifact nedir?

Artifact, agent'ın çalışması sırasında üretilen ve daha sonra yeniden kullanılabilecek somut çalışma ürünüdür.

Örnek:

```text
cleaned.parquet
features.parquet
model.pkl
predictions.csv
report.pdf
plot.png
embeddings.parquet
database.sqlite
```

Bir dataframe RAM'de:

```python
df = load_data()
```

iken sadece runtime state'idir.

Kalıcı bir dosyaya dönüştürülmesi:

```python
df.to_parquet("features.parquet")
```

ile artifact oluşturur.

Ancak production mimarisinde önemli nokta, dosyanın sandbox filesystem'inde kalması değil, artifact service üzerinden durable storage'a alınmasıdır.

---

# 4. Sandbox nedir?

Sandbox, agent'ın ürettiği kodun çalıştırıldığı izole execution environment'tır.

Örneğin:

```text
LLM
 |
 | generated Python
 v
Sandbox
 |
 +-- Python
 +-- shell
 +-- packages
 +-- /workspace
 +-- network policy
 +-- CPU / RAM / disk limits
```

Sandbox'ın görevi:

- kod çalıştırmak,
- process'leri izole etmek,
- filesystem erişimini sınırlandırmak,
- network'ü kontrol etmek,
- CPU/RAM/disk kaynaklarını sınırlamak,
- gerekiyorsa untrusted code için daha güçlü isolation sağlamak.

Sandbox'ın görevi esas olarak **compute ve isolation**'dır.

---

# 5. Sandbox nerede çalışabilir?

Genel olarak birkaç yaygın model var.

## 5.1 Normal container

```text
OpenShift
   |
   v
Pod / Container
   |
   v
Python / Shell
```

Avantaj:

- hızlı,
- Kubernetes/OpenShift ile doğal uyum,
- kaynak limitleri ve lifecycle yönetimi kolay.

Dezavantaj:

- container kernel'i host kernel ile paylaşır,
- çok güçlü untrusted-code izolasyonu gereken senaryolarda ek katman gerekebilir.

Daytona gibi modern agent sandbox platformları container'ları varsayılan sandbox olarak kullanıyor ve VM tabanlı seçenekleri de ayrıca sunuyor. OpenAI, Google ve Anthropic'in agent execution yaklaşımlarında da sandbox/container tabanlı code execution örnekleri bulunuyor.

## 5.2 gVisor

```text
Pod
 |
 v
gVisor
 |
 v
Host Linux Kernel
```

gVisor, container ile host kernel arasına ek bir isolation katmanı koyar. Güvenlik kazanımı karşılığında workload'a göre performans ve uyumluluk maliyeti vardır.

## 5.3 Kata Containers / OpenShift Sandboxed Containers

```text
OpenShift
   |
   v
Pod
   |
   v
Kata runtime
   |
   v
Lightweight VM
   |
   v
AI-generated code
```

Bu model, untrusted veya daha hassas workload'lar için daha güçlü bir güvenlik sınırı sağlar.

Red Hat OpenShift Sandboxed Containers, Kata Containers'ı opsiyonel runtime olarak entegre eder ve container workload'larını lightweight VM'lerde çalıştırır.

## 5.4 Managed sandbox platformları

E2B, Daytona, Modal ve benzeri sistemler sandbox lifecycle, compute ve isolation katmanlarını servis olarak sunabilir.

Bunlar özellikle kendi sandbox orchestration katmanını yazmak istemeyen ekiplerde avantaj sağlar.

---

# 6. OpenShift kullanan bir ekip için sandbox önerisi

OpenShift zaten varsa ilk tercih genellikle ayrı bir sandbox sağlayıcısı kurmak yerine mevcut cluster'ı kullanmaktır.

Örnek:

```text
                 OpenShift Cluster
                        |
             +----------+----------+
             |                     |
             v                     v
       Orchestrator           Sandbox Pod
                                  |
                             runtimeClass:
                                  kata
                                  |
                                  v
                           Python / Shell
```

Burada:

- OpenShift = orchestration
- Kata = isolation
- Pod = execution instance

olur.

Her AI-generated code workload'u için `Kata` zorunlu değildir. Güvenilen/internal code ile gerçekten untrusted model-generated code arasında farklı runtime sınıfları kullanılabilir.

Örneğin:

```text
Trusted internal workload  -> normal container
Untrusted AI-generated code -> Kata
```

güvenlik-politikası olarak düşünülebilir.

---

# 7. Artifact persistence nedir?

Artifact persistence:

> Agent workflow'unda oluşan dosya, dataframe, model, grafik ve diğer ara çıktıların sandbox veya Python process'inden bağımsız olarak daha sonra tekrar erişilebilir halde tutulmasıdır.

Temel akış:

```text
PTC
 |
 v
Sandbox
 |
 | creates
 v
/workspace/features.parquet
 |
 | persist
 v
Artifact Service
 |
 v
Object Storage
```

Bundan sonra sandbox yok olsa bile:

```text
features.parquet
```

hala kullanılabilir.

---

# 8. Neden sandbox ile artifact store ayrılmalı?

Kötü model:

```text
Sandbox filesystem
    =
kalıcı artifact storage
```

Bu modelde sandbox ölürse çıktı da kaybolabilir.

Daha iyi:

```text
Sandbox filesystem
      |
      | save important output
      v
Artifact Service
      |
      v
Durable Object Storage
```

Böylece:

```text
Sandbox #1
   |
   v
artifact A
   |
   v
Object Storage
   |
   X Sandbox #1 deleted
   |
   v
Sandbox #2
   |
   v
artifact A yeniden yüklenir
```

Bu, multi-turn ve fault-tolerant workflow'lar için daha sağlıklı bir modeldir.

---

# 9. Artifact storage için S3-compatible object storage

Burada “S3” kelimesinin AWS S3 olmak zorunda olmadığını ayırmak gerekir.

S3 bir API / object storage ekosistemidir.

Kurum içi seçenekler:

```text
S3-compatible
 |
 +-- OpenShift Data Foundation / NooBaa (MCG)
 |
 +-- Ceph RADOS Gateway (RGW)
 |
 +-- MinIO
 |
 +-- diğer kurumsal object storage ürünleri
```

OpenShift Data Foundation'ın Multicloud Object Gateway'i (MCG/NooBaa), AWS S3 SDK/API kullanan uygulamalara S3-compatible object interface sağlayabilir.

Eğer kurum zaten ODF kullanıyorsa yeni bir MinIO deployment'ı kurmadan mevcut object storage katmanından yararlanmak daha doğal olabilir.

---

# 10. OpenShift Data Foundation ile örnek

```text
                     OpenShift
                         |
                         v
                     ODF Layer
                 /               \
                v                 v
           NooBaa / MCG        Ceph RGW
                |                 |
                +-------+---------+
                        |
                        v
                 Durable Storage
```

Agent tarafında:

```text
artifact.save()
       |
       v
S3-compatible endpoint
       |
       v
ODF backend
```

Böylece PTC'nin “hangi storage teknolojisinin altta olduğu” bilgisine sahip olması gerekmez.

---

# 11. MinIO nerede devreye girer?

MinIO, Amazon S3'ün kendisi değildir.

> MinIO, S3-compatible object storage sağlayan bir üründür.

Örneğin:

```text
OpenShift
   |
   +-- Agent
   |
   +-- Kata Sandbox
   |
   +-- MinIO
         |
         v
   Persistent Storage
```

MinIO Kubernetes/OpenShift üzerinde çalıştırılabilir.

Ancak kurumda zaten ODF/Ceph/başka bir S3-compatible storage varsa ikinci bir object store eklemek genellikle gereksiz operasyon yükü yaratır.

Bu nedenle seçim sırası:

```text
1. Mevcut kurumsal S3-compatible storage
2. ODF/NooBaa/Ceph zaten varsa onu kullan
3. Hiçbiri yoksa MinIO değerlendir
```

---

# 12. Artifact Service neden gerekli?

PTC'nin doğrudan:

```text
S3 PUT
```

yapması mümkün olsa da uzun vadede artifact API'si araya koymak daha temizdir.

Mimari:

```text
PTC
 |
 v
Artifact Service
 |
 +-- Metadata DB
 |
 +-- Object Storage
```

PTC:

```python
artifact.save("/workspace/features.parquet")
```

diyebilir.

Artifact Service:

```text
1. dosyayı doğrula
2. checksum hesapla
3. metadata oluştur
4. object storage'a yükle
5. artifact_id üret
6. metadata DB'ye yaz
7. artifact_id döndür
```

Örnek cevap:

```json
{
  "artifact_id": "art_82913",
  "type": "parquet",
  "size_bytes": 834829102,
  "name": "features"
}
```

LLM'e yüzlerce MB/GB dataframe'i geri göndermek gerekmez.

---

# 13. Artifact metadata

Sadece dosya URI'sini saklamak yeterli değildir.

Önerilen metadata:

```json
{
  "artifact_id": "art_82913",
  "name": "features",
  "type": "dataset",
  "format": "parquet",
  "size_bytes": 834829102,
  "checksum": "sha256:...",
  "workflow_id": "wf_123",
  "run_id": "run_456",
  "turn_id": "turn_7",
  "ptc_execution_id": "ptc_9",
  "parent_artifacts": ["art_001"],
  "created_at": "...",
  "version": 3,
  "storage_uri": "s3://.../art_82913"
}
```

Bu metadata:

- lineage,
- debugging,
- access control,
- reproducibility,
- versioning,
- cleanup / TTL

için kullanılır.

---

# 14. Artifact ve state aynı şey değildir

Bu ayrım özellikle önemlidir.

## Artifact

```text
features.parquet
model.pkl
report.pdf
```

## Workflow state

```json
{
  "run_id": "run_123",
  "current_step": "training",
  "last_successful_node": "feature_engineering",
  "input_artifact": "art_002"
}
```

State:

> Workflow şu anda nerede?

Artifact:

> Workflow ne üretti?

Bunlar ayrı katmanlar olarak ele alınmalıdır.

---

# 15. Multi-turn workflow

Örneğin:

### Turn 1

```text
User:
"CSV'yi temizle."

PTC #1
  |
  v
cleaned.parquet
  |
  v
artifact A
```

### Turn 2

```text
User:
"Şimdi feature engineering yap."

PTC #2
  |
  v
load artifact A
  |
  v
features.parquet
  |
  v
artifact B
```

### Turn 3

```text
User:
"Modeli eğit."

PTC #3
  |
  v
load artifact B
  |
  v
model.pkl
  |
  v
artifact C
```

Burada workflow:

```text
Turn 1 -> Turn 2 -> Turn 3
```

diye zaman içinde uzayabilir.

Artifact persistence, bu turn'leri birbirine bağlayan çalışma çıktısı katmanıdır.

---

# 16. PTC execution'ları aynı sandbox'ta mı olmalı?

İki temel model vardır.

## Model A — Session başına uzun yaşayan sandbox

```text
Session
 |
 +-- PTC #1
 |
 +-- PTC #2
 |
 +-- PTC #3
 |
 +-- PTC #4
```

Avantaj:

- filesystem state korunur,
- Python variables/imports korunabilir,
- package installation tekrar gerekmez,
- latency azalabilir.

Google Agent Runtime Code Execution dokümantasyonunda tek sandbox'ın bir workflow session boyunca tutulması ve variables, imports ve file state'in sonraki tool call'larına taşınması açıkça destekleniyor.

OpenAI Sandbox Agents de persistent workspace ve saved sandbox state yaklaşımını destekliyor.

## Model B — Her execution için yeni sandbox

```text
PTC #1 -> Sandbox A -> artifact A

PTC #2 -> Sandbox B -> artifact B

PTC #3 -> Sandbox C -> artifact C
```

Avantaj:

- daha güçlü isolation boundary,
- daha temiz lifecycle,
- daha kolay horizontal scaling.

Dezavantaj:

- startup maliyeti,
- her execution'da environment hydration gerekebilir.

Bu iki modelin birlikte kullanılması da mümkündür.

---

# 17. Hibrit yaklaşım

Production'da güçlü bir pattern:

```text
Workflow Session
       |
       +---- Sandbox A
       |       |
       |       +-- PTC #1
       |       +-- PTC #2
       |
       +---- Sandbox B
       |       |
       |       +-- PTC #3
       |
       +---- Sandbox C
               |
               +-- PTC #4
```

Sandbox'lar arasında paylaşım:

```text
Artifact Store
```

ile yapılır.

Bu durumda:

- aynı session içinde temporary state filesystem'de kalabilir,
- sandbox değiştiğinde önemli çıktılar artifact olarak devam eder.

---

# 18. Artifact persistence + failure recovery

Asıl güçlü kullanım alanlarından biri budur.

```text
PTC #1
  |
  v
artifact A ✓

PTC #2
  |
  v
artifact B ✓

PTC #3
  |
  X
  error
```

Workflow restart:

```text
State Store
 |
 +-- Step 1 completed
 +-- Step 2 completed
 +-- Step 3 failed

Artifact Store
 |
 +-- artifact A
 +-- artifact B
```

Agent:

```text
resume Step 3
```

diyebilir.

Böylece pahalı Step 1 ve Step 2 tekrar çalıştırılmaz.

Bu model checkpointing / durable execution sistemleriyle birlikte kullanıldığında fault tolerance sağlar.

---

# 19. Artifact lineage

Artifact'lerin hangi çıktılardan üretildiğini tutmak çok faydalıdır.

```text
raw.csv
   |
   v
artifact A
   |
   v
cleaned.parquet
   |
   v
artifact B
   |
   +------> statistics.json
   |
   v
features.parquet
   |
   v
artifact C
   |
   v
model.pkl
   |
   v
artifact D
```

Metadata:

```json
{
  "artifact_id": "art_C",
  "parent_artifacts": ["art_B"]
}
```

Böylece:

> model.pkl hangi dataset'ten üretildi?

sorusunun cevabı bulunabilir.

---

# 20. Artifact versioning

Artifact'leri immutable yapmak iyi bir production yaklaşımıdır.

Örneğin:

```text
art_001 -> features v1
art_002 -> features v2
art_003 -> features v3
```

Yeni içerik için yeni artifact ID üretilir.

Bu:

- reproducibility,
- rollback,
- debugging,
- audit

açısından faydalıdır.

S3 Versioning de aynı object key'in farklı sürümlerini koruyarak overwrite/delete durumlarında geri dönüş imkanı sağlar; kurum içi S3-compatible storage'larda benzer özelliklerin desteklenip desteklenmediği seçilen ürüne göre kontrol edilmelidir.

---

# 21. Büyük dataframe nasıl saklanmalı?

LLM context'ine:

```text
500 MB dataframe
```

göndermek yerine:

```text
features.parquet
```

artifact olarak saklamak daha mantıklıdır.

Parquet gibi columnar formatlar, büyük analitik datasetlerde:

- compression,
- schema,
- hızlı kolon erişimi

gibi avantajlar sağlayabilir.

Örneğin:

```python
df.to_parquet("/workspace/features.parquet")
```

ve daha sonra:

```python
df = pd.read_parquet(
    "/workspace/features.parquet",
    columns=["temperature", "consumption"]
)
```

kullanılabilir.

---

# 22. Küçük result ile artifact arasında karar

Her sonucu artifact yapmak gerekmez.

```text
42
"completed"
{"mean": 14.2}
```

gibi küçük sonuçlar:

```text
context / workflow state
```

içinde tutulabilir.

Büyük ve yeniden kullanılacak çıktılar:

```text
500 MB parquet
2 GB model
PDF
image
large JSON
database
```

artifact store'a gitmelidir.

Zihinsel model:

```text
small semantic result -> state/context

large reusable output -> artifact
```

---

# 23. Sandbox network ve artifact storage

Sandbox'ın internete tamamen açık olması gerekmeyebilir.

Örneğin:

```text
                     Sandbox
                        |
               +--------+---------+
               |        |         |
               v        v         v
            Artifact   Internal   Approved
             Store      APIs      Egress
               ✓          ✓          ?
```

Egress policy ile:

```text
Internet           -> deny
Artifact endpoint  -> allow
Internal API       -> allow
Approved domains   -> allow
```

gibi kurallar uygulanabilir.

Bu özellikle AI-generated code çalıştırılan production sandbox'larında önemlidir.

---

# 24. OpenShift için önerilen referans mimari

OpenShift kullanan bir ekip için:

```text
                           +-------------------+
                           |      LLM          |
                           +---------+---------+
                                     |
                                     v
                           +-------------------+
                           | ReAct / Agent     |
                           | Orchestrator      |
                           +---------+---------+
                                     |
                       +-------------+-------------+
                       |                           |
                       v                           v
                +-------------+             +-------------+
                | State Store |             |  Sandbox    |
                | PostgreSQL  |             |  Manager    |
                +-------------+             +------+------+
                                                  |
                                                  v
                                          +---------------+
                                          | OpenShift     |
                                          | Sandbox Pod   |
                                          +-------+-------+
                                                  |
                                          +-------v-------+
                                          | Kata runtime  |
                                          | (untrusted)   |
                                          +-------+-------+
                                                  |
                                             /workspace
                                                  |
                                          artifact.save()
                                                  |
                                                  v
                                      +-----------------------+
                                      | Artifact Service      |
                                      +-----------+-----------+
                                                  |
                              +-------------------+------------------+
                              |                                      |
                              v                                      v
                       +-------------+                         +-------------+
                       | Metadata DB |                         | Object Store|
                       | PostgreSQL  |                         | S3-compatible
                       +-------------+                         +-------------+
                                                                    |
                                                         +----------+---------+
                                                         |                    |
                                                      ODF/MCG             Ceph RGW
                                                         |
                                                      / or /
                                                         |
                                                       MinIO
```

Not:

> ODF/MCG, Ceph RGW ve MinIO'nun üçünü birden kullanmak zorunda değilsiniz. Bunlar object-storage tarafında alternatif/katmanlı seçeneklerdir.

---

# 25. Benim önerdiğim OpenShift stack

Kurum zaten OpenShift kullanıyorsa:

### Sandbox

```text
OpenShift
   +
Kata Containers
```

AI-generated/untrusted code için güçlü default.

### Artifact

```text
Existing enterprise S3-compatible storage
```

Varsa onu kullan.

Özellikle ODF zaten varsa:

```text
ODF
 |
 +-- NooBaa / MCG
 +-- Ceph RGW
```

seçenekleri değerlendirilmelidir.

### Storage yoksa

```text
MinIO
```

değerlendirilebilir.

### Metadata / workflow state

```text
PostgreSQL
```

### Temporary workspace

```text
Sandbox / ephemeral filesystem
```

---

# 26. Önerilmeyen yaklaşım

Her şeyi aynı katmana koymak:

```text
OpenShift Pod
 |
 +-- code execution
 +-- workflow state
 +-- persistent files
 +-- artifact registry
 +-- database
 +-- object storage
```

Bu kötü separation of concerns yaratır.

Daha iyi:

```text
Compute      -> Sandbox
Durable data -> Artifact Store
Workflow     -> State Store
Orchestration -> OpenShift / controller
```

---

# 27. Artifact Service'in API taslağı

Basit bir API:

```http
POST /artifacts
GET  /artifacts/{artifact_id}
GET  /artifacts/{artifact_id}/metadata
GET  /workflows/{workflow_id}/artifacts
DELETE /artifacts/{artifact_id}
```

Örneğin:

```json
POST /artifacts

{
  "name": "features",
  "type": "dataset",
  "format": "parquet",
  "workflow_id": "wf_123",
  "run_id": "run_456"
}
```

Response:

```json
{
  "artifact_id": "art_789",
  "status": "stored"
}
```

PTC:

```text
artifact.save()
       |
       v
artifact_id = art_789
```

Sonraki PTC:

```text
artifact.load("art_789")
```

---

# 28. Önemli production özellikleri

Artifact Service için düşünülmesi gerekenler:

```text
Authentication
Authorization
Tenant isolation
Checksum
Immutable versions
Lineage
TTL / retention
Quotas
Encryption
Audit logs
Virus/malware scanning where applicable
Content-type validation
Size limits
Lifecycle management
```

Object storage tarafında versioning/lifecycle gibi özellikler artifact lifecycle'ını yönetmeye yardımcı olur.

---

# 29. Sandbox lifecycle için öneri

Uzun ömürlü sandbox her zaman gerekli değildir.

### Kısa task

```text
create sandbox
   |
execute
   |
save artifacts
   |
destroy sandbox
```

### Uzun session

```text
create sandbox
   |
PTC #1
   |
PTC #2
   |
PTC #3
   |
snapshot/persist
   |
resume later
```

### Multi-turn

```text
Turn 1 -> sandbox A
          |
          +-> artifact A

Turn 2 -> sandbox B
          |
          +-> load artifact A
```

Bu son yaklaşım artifact persistence'ın en güçlü kullanım şekillerindendir.

---

# 30. Büyük resim

Bütün kavramları tek yerde toplarsak:

```text
                         AGENT
                           |
                    ReAct / Planning
                           |
                           v
                     ORCHESTRATOR
                     /           \
                    /             \
                   v               v
              STATE STORE      SANDBOX MANAGER
                   |                 |
                   |                 v
                   |          OPENSHIFT SANDBOX
                   |              /       \
                   |             /         \
                   |          Python      Shell
                   |             |
                   |             v
                   |         /workspace
                   |             |
                   |       artifact.save()
                   |             |
                   |             v
                   |       ARTIFACT SERVICE
                   |          /         \
                   |         /           \
                   v        v             v
               workflow   METADATA    OBJECT STORAGE
                 state       DB        S3-compatible
                                         |
                                +--------+--------+
                                |        |        |
                               ODF    Ceph RGW  MinIO
```

Buradaki en önemli dört cümle:

> **Sandbox = code execution ve isolation.**

> **Artifact Store = reusable/durable çalışma ürünleri.**

> **State Store = workflow'un nerede kaldığı ve hangi artifact'lerin kullanılacağı.**

> **OpenShift = bunların workload lifecycle/orchestration katmanı.**

---

# 31. Kaynaklardan çıkan ortak yön

### OpenAI

OpenAI Agents SDK'nin Sandbox Agents yaklaşımı model için gerçek filesystem üzerinde çalışan persistent workspace sağlıyor; sandbox lifecycle, snapshots/resume ve remote filesystem girişleri ayrı kavramlar olarak ele alınıyor. Bu, “execution workspace” ile “durable state/storage” ayrımını destekliyor.

Kaynak:
- OpenAI Agents SDK — Sandbox Agents: https://openai.github.io/openai-agents-python/sandbox_agents/
- OpenAI Agents SDK — Sandbox Concepts: https://openai.github.io/openai-agents-python/sandbox/guide/

### Anthropic

Anthropic PTC'yi code execution altyapısındaki sandbox container'lar üzerine kuruyor. PTC execution sonuçları / artifacts container altyapısında tutulabiliyor ve retention/lifecycle ayrı şekilde yönetiliyor.

Kaynak:
- Anthropic — Programmatic Tool Calling: https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling
- Anthropic — Code Execution: https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool

### Google ADK

Google Agent Runtime Code Execution aynı sandbox'ı bir workflow session boyunca koruyarak variables, imports ve file state'in sonraki execution'larda kullanılmasını destekliyor. Bu, session-level sandbox persistence'ın doğrudan bir örneğidir.

Kaynak:
- Google ADK — Agent Runtime Code Execution: https://google.github.io/adk-docs/tools/google-cloud/code-exec-agent-engine/

### OpenShift / Red Hat

Red Hat OpenShift Sandboxed Containers, Kata Containers runtime'ını OpenShift'e entegre ederek workload'ları lightweight VM'lerde çalıştırıyor ve untrusted/privileged workload'lar için daha güçlü isolation sağlıyor.

Kaynak:
- Red Hat — OpenShift Sandboxed Containers: https://docs.redhat.com/en/documentation/openshift_sandboxed_containers/1.13/html/deploying_openshift_sandboxed_containers_on_bare-metal_servers/osc-discover_metal-osc

### OpenShift Data Foundation

ODF Multicloud Object Gateway, AWS S3 API/SDK kullanan uygulamalar tarafından erişilebilen S3-compatible object service sağlıyor. Dolayısıyla OpenShift ortamında ayrı bir object storage servisi olarak değerlendirilebilir.

Kaynak:
- Red Hat — ODF Multicloud Object Gateway: https://docs.redhat.com/en/documentation/red_hat_openshift_data_foundation/4.20/html/managing_hybrid_and_multicloud_resources/accessing-the-multicloud-object-gateway-with-your-applications_rhodf

### gVisor

gVisor container ile host Linux kernel'i arasında ek bir isolation katmanı sağlar; production kullanımında security/performance trade-off'ları açıkça vurgulanıyor.

Kaynak:
- gVisor Production Guide: https://gvisor.dev/docs/user_guide/production/

### Daytona

Daytona mimarisinde control plane ile compute plane ayrılıyor; sandbox'lar izole compute birimleri olarak çalışıyor ve persistent volumes S3-compatible object storage ile desteklenebiliyor. Bu, sandbox ile durable storage'ın ayrıştırıldığı modern bir referans mimari örneğidir.

Kaynak:
- Daytona Architecture: https://www.daytona.io/docs/en/architecture/
- Daytona Sandboxes: https://www.daytona.io/docs/en/sandboxes/

### MinIO

MinIO, Kubernetes üzerinde çalışabilen ve AWS S3-compatible API sağlayan object storage çözümüdür.

Kaynak:
- MinIO Kubernetes Documentation: https://min.io/docs/minio/kubernetes/upstream/index.html

### LangGraph

LangGraph persistence/checkpointing, workflow state'in adım adım saklanmasını ve workflow'ların kaldığı noktadan devam etmesini sağlar. Bu, artifact storage'dan farklı olan workflow-state katmanına iyi bir örnektir.

Kaynak:
- LangGraph Persistence: https://docs.langchain.com/oss/python/langgraph/persistence

---

# 32. Sonuç / Tavsiye

OpenShift kullanan bir kurum için en sade ve sağlam başlangıç:

```text
                    ReAct / PTC Agent
                           |
                      Orchestrator
                       /         \
                      v           v
                 PostgreSQL     OpenShift
                 state DB          |
                                   v
                              Kata Sandbox
                                   |
                              /workspace
                                   |
                              artifact.save()
                                   |
                                   v
                            Artifact Service
                              /         \
                             v           v
                         PostgreSQL    S3-compatible
                         metadata       storage
                                        |
                              Existing ODF/MCG/
                              Ceph RGW if available
                              otherwise MinIO
```

### Öncelikli karar sırası

1. **Sandbox:** OpenShift içinde çalıştır.
2. **Untrusted AI-generated code:** Kata runtime'ı değerlendir.
3. **Temporary files:** sandbox filesystem.
4. **Reusable/large outputs:** artifact service üzerinden durable object storage.
5. **Object storage:** kurumda zaten ODF/MCG, Ceph RGW veya başka S3-compatible storage varsa onu kullan.
6. **Yoksa:** MinIO değerlendir.
7. **Workflow state:** PostgreSQL/checkpoint store.
8. **Artifact metadata:** PostgreSQL.
9. **Sandbox ile artifact storage'ı aynı lifecycle'a bağlama.**

## Kısa mimari prensip

```text
"Compute here"
      ↓
   Sandbox

"Remember this output"
      ↓
 Artifact Store

"Remember where I am"
      ↓
 State Store
```

Bu ayrım, tek bir PTC'den çok daha büyük olan **multi-PTC, multi-node ve multi-turn agent workflow'larında** özellikle önemlidir.
