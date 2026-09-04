Evet. Buradaki **PTC = Programmatic Tool Calling** ise, yazdığın iki kavramı biraz daha teknik araştırınca önemli bir ayrım ortaya çıkıyor: **“error recovery” PTC’nin kendisinin zorunlu bir özelliği değil**, PTC + code execution/sandbox altyapısı üzerinde kurulabilen bir **self-repair pattern’i**. Aynı şekilde **artifact persistence** de PTC’nin temel amacı değil; çok adımlı agent workflow’larında code execution/container state ile birlikte kullanılan bir persistence yaklaşımı. ([Claude Platform][1])

## 1. PTC nedir?

Normal tool calling:

```text
LLM
 ↓
tool_call()
 ↓
Tool sonucu
 ↓
LLM
 ↓
tool_call()
 ↓
Tool sonucu
 ↓
LLM
```

PTC'de ise model bir program üretir:

```text
LLM
 ↓
Python programı
 ↓
 ├── tool_1()
 ├── tool_2()
 ├── tool_3()
 ├── dataframe işlemleri
 └── sonuçları birleştir
 ↓
tek final sonuç
 ↓
LLM
```

Yani model her tool çağrısından sonra tekrar reasoning yapmak zorunda kalmaz. Program sandbox/container içerisinde tool'ları çağırabilir. Bu özellikle **çok adımlı workflow'larda latency ve token kullanımını azaltıyor.** Anthropic'in resmi dokümanında da PTC'nin temel amacı bu şekilde tanımlanıyor. ([Claude Platform][1])

OpenAI Agents SDK'da da aynı fikir var: model JavaScript üreterek birden fazla tool'u hosted execution environment içerisinde koordine edebiliyor. ([OpenAI GitHub][2])

---

# 2. PTC: Error Recovery

Senin yazdığın:

> stack trace → LLM → patched code

**Doğru bir pattern.** Ama bunu biraz daha doğru ifade etmek lazım.

Örneğin agent şöyle bir kod üretiyor:

```python
df = load_data()

result = df.groupby("city")["revenue"].mean()

print(result)
```

Ama execution sırasında:

```text
KeyError: 'revenue'
```

oluşuyor.

Normal sistem:

```text
execution failed
       ↓
error returned
       ↓
agent stops
```

Self-repair yapan PTC sisteminde:

```text
              ┌───────────────┐
              │  LLM generates│
              │     code      │
              └───────┬───────┘
                      ↓
                Code execution
                      ↓
                 ERROR ❌
                      ↓
              stack trace + code
                      ↓
              ┌───────────────┐
              │      LLM      │
              │ analyze error │
              └───────┬───────┘
                      ↓
                patched code
                      ↓
                execute again
                      ↓
                   SUCCESS
```

Örneğin LLM şunu fark eder:

```python
df.columns
# ['city', 'sales']
```

ve kodu:

```python
result = df.groupby("city")["sales"].mean()
```

olarak patch eder.

### Buradaki kritik nokta

**Stack trace tek başına yeterli değil.**

İyi bir error-recovery sisteminde LLM'e mümkün olduğunca:

```text
original code
+
stack trace
+
stdout
+
stderr
+
relevant variables/state
+
tool outputs
+
environment information
```

verilir.

Örneğin:

```json
{
  "code": "...",
  "error": {
    "type": "KeyError",
    "message": "'revenue'",
    "traceback": "..."
  },
  "stdout": "...",
  "stderr": "...",
  "state": {
    "df_columns": ["city", "sales"]
  }
}
```

Böylece LLM sadece:

> "Kod çalışmadı"

demiyor;

> "Kod neden çalışmadı ve nasıl değiştirmeliyim?"

sorusunu cevaplayabiliyor.

---

# 3. PTC'de çok önemli bir detay: tool error da programın içine dönebiliyor

Anthropic'in PTC dokümanında özellikle bu davranış var.

Programmatic tool çağrısı hata verirse, hata **code execution'ın aldığı sonuç olarak programa geri dönebiliyor** ve program bunu handle edebiliyor. Örneğin timeout durumunda execution ortamı `stderr` üzerinden hatayı görebiliyor. ([Claude Platform][1])

Dolayısıyla daha gelişmiş bir sistem:

```python
try:
    result = query_database(...)
except Exception as e:
    print(e)
```

gibi bir mekanizmayla hatayı yakalayabilir.

Sonra agent:

```text
error
 ↓
diagnosis
 ↓
repair
 ↓
retry
```

yapabilir.

Bu aslında **agentic execution loop** oluşturuyor.

---

# 4. Artifact Persistence

İkinci konu daha da önemli.

Senin tanımın:

> Çok adımlı workflow'larda üretilen dosya, dataframe ve ara çıktıların kalıcı olarak saklanmasıdır.

**Mantık doğru**, fakat "persistent" kelimesine dikkat etmek gerekiyor.

Burada iki farklı şey var:

### A. In-memory state

Örneğin:

```python
df = load_csv()
df = clean(df)
```

`df` sadece mevcut execution/container yaşamı boyunca durabilir.

Sonraki adımda aynı process/container devam ediyorsa:

```python
df
```

hala vardır.

---

### B. Persistent artifact

Bunu fiziksel bir artifact olarak kaydedersin:

```text
/data/
    raw.csv
    cleaned.parquet
    features.parquet
    model.pkl
    report.pdf
```

Sonraki agent step'i:

```python
df = read_parquet("/data/cleaned.parquet")
```

yapabilir.

Yani:

```text
Step 1
 ↓
raw.csv
 ↓
cleaned.parquet
 ↓
Step 2
 ↓
features.parquet
 ↓
Step 3
 ↓
model.pkl
 ↓
Step 4
 ↓
report.pdf
```

Burada her step'in çıktısı **artifact**.

---

# 5. Neden artifact persistence önemli?

Asıl avantajı şu:

### Persistence yoksa

Agent:

```text
Step 1
 ↓
10 GB dataset işle
 ↓
Step 2
 ↓
dataset'i tekrar yükle
 ↓
Step 3
 ↓
tekrar üret
```

yapabilir.

Bu çok pahalı.

Persistence varsa:

```text
Step 1
 ↓
10 GB dataset
 ↓
cleaned.parquet
        ↓
Step 2
        ↓
features.parquet
        ↓
Step 3
```

olur.

Yani **intermediate result tekrar hesaplanmaz.**

---

# 6. PTC + Artifact Persistence birleşince çok güçlü oluyor

Bence senin çalıştığın konu açısından asıl önemli mimari bu:

```text
                 USER TASK
                     │
                     ↓
                   LLM
                     │
                     ↓
            Generate program
                     │
                     ↓
          ┌──────────────────┐
          │ Sandbox / Runtime │
          └────────┬─────────┘
                   │
       ┌───────────┼───────────┐
       ↓           ↓           ↓
    Tool A      Tool B      Tool C
       │           │           │
       └───────────┼───────────┘
                   ↓
              Artifacts
                   │
          ┌────────┴────────┐
          ↓                 ↓
      dataframe          file
          ↓                 ↓
      parquet             csv
          ↓                 ↓
          └────────┬────────┘
                   ↓
              Next Step
```

Ve execution hata verirse:

```text
                    ERROR
                      │
                      ↓
              traceback + state
                      │
                      ↓
                     LLM
                      │
                 repair code
                      │
                      ↓
                  execute
                      │
               ┌──────┴──────┐
               │             │
             fail          success
               │             │
               └──→ retry    ↓
                         artifact
```

Bu noktada **PTC sadece tool calling mekanizması olmaktan çıkıp küçük bir execution engine gibi davranmaya başlıyor.**

---

# 7. Container persistence konusu özellikle önemli

Burada önemli bir nüans var.

Anthropic'in güncel PTC implementation'ında code execution container'ları yeniden kullanılabiliyor ve container state korunabiliyor; dokümantasyon container'ın belirli koşullarda tekrar kullanılabildiğini ve execution artifacts/output'larının belirli bir süre tutulduğunu belirtiyor. ([Claude Platform][1])

Fakat OpenAI Agents SDK'nın PTC implementation'ında generated program **fresh hosted V8 environment** içerisinde çalışıyor ve filesystem/network/persistent process erişimi bulunmuyor. ([OpenAI GitHub][2])

Dolayısıyla:

> **"PTC = artifact persistence"**

demek doğru değil.

Daha doğru:

> **PTC + execution runtime + artifact storage = persistent multi-step agent workflow**

---

# 8. Bence senin tabloda tanımlar şöyle olmalı

### PTC: Error Recovery

> **Kod execution sırasında oluşan hata, stack trace ve ilgili execution state ile birlikte LLM'e geri beslenerek hatalı kodun otomatik olarak analiz edilmesi, patch edilmesi ve yeniden çalıştırılmasıdır. Agent'ın execution failure'larından otomatik olarak recovery yapmasını sağlar.**

### PTC: Artifact Persistence

> **Çok adımlı agent workflow'larında oluşturulan dosya, dataframe, model, rapor ve diğer ara çıktılarının artifact olarak kalıcı biçimde saklanarak sonraki workflow adımlarında yeniden hesaplanmadan kullanılabilmesidir.**

Bunlar mevcut tanımlarından biraz daha teknik ve doğru.

---

## 9. Bir de üçüncü kavramı eklemeni öneririm: **Execution State Persistence**

Artifact persistence ile aynı şey değil.

Örneğin:

```text
Artifact:
cleaned_data.parquet
model.pkl
report.pdf
```

ama state:

```text
df = ...
selected_columns = [...]
current_step = 4
tool_results = {...}
variables = {...}
```

Yani:

**Artifact = üretilen veri/nesne**

**State = agent'ın workflow'daki mevcut durumu**

Bu ayrım agent altyapısı tasarlarken çok önemli.

---

### Kısaca

| Konu                     | Ne yapıyor?                                                           |
| ------------------------ | --------------------------------------------------------------------- |
| **PTC**                  | LLM'in kod yazarak birden fazla tool'u programatik çağırmasını sağlar |
| **Error Recovery**       | Hata → traceback/state → LLM → patch → retry                          |
| **Artifact Persistence** | CSV, Parquet, dataframe, model, PDF vb. çıktıları saklar              |
| **State Persistence**    | Agent'ın workflow durumunu saklar                                     |
| **Sandbox/Runtime**      | Kodun güvenli şekilde çalıştığı ortam                                 |
| **Retry Loop**           | Başarısız execution'ı tekrar dener                                    |

Ve güncel kaynaklara göre PTC'nin en güçlü kullanım alanlarından biri zaten **3+ bağımlı tool çağrısı, veri işleme, filtering/transformation ve intermediate sonuçların model context'ine gereksiz yere taşınmaması**. ([Claude Platform][1])

[Anthropic — Programmatic Tool Calling dokümantasyonu](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling?utm_source=chatgpt.com)
[OpenAI Agents SDK — Tools / Programmatic Tool Calling](https://openai.github.io/openai-agents-python/tools/?utm_source=chatgpt.com)

İstersen bir sonraki adımda **bu iki özelliğin gerçek bir agent framework'ünde nasıl implement edildiğini (sandbox + artifact store + retry/self-repair loop) Python koduyla mimari olarak** gösterebilirim.

[1]: https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling?cd=1966&utm_source=chatgpt.com "Programmatic tool calling - Claude Platform Docs"
[2]: https://openai.github.io/openai-agents-python/tools/?utm_source=chatgpt.com "Tools - OpenAI Agents SDK"


Evet. Bu konuyu baştan sona anlamanın en iyi yolu, **“artifact neden var?” sorusundan başlayıp gerçek bir agent workflow’unda dosya/dataframe’ın nasıl yaşadığını** görmek.

Önce çok önemli bir düzeltme:

> **Artifact persistence, PTC’nin kendisi değildir.**
> PTC, modelin ürettiği programın birden fazla tool çağrısını koordine etmesini sağlar. Artifact persistence ise bu program/workflow sırasında oluşan **dosya, dataset, dataframe çıktısı, grafik, model, rapor gibi ara ürünlerin saklanıp sonraki adımlarda tekrar kullanılmasını** sağlar. Modern agent runtime’ları bu ikisini birlikte kullanabilir. ([OpenAI GitHub][1])

---

# 1. Önce “artifact” nedir?

Artifact'i çok basit düşün:

> **Agent'ın çalışma sırasında ürettiği ve daha sonra tekrar kullanılabilecek somut çıktı.**

Örneğin agent'a şunu verdin:

> “Bu CSV'yi temizle, analiz et, model eğit ve sonuç raporu oluştur.”

Agent'ın workflow'u şöyle olabilir:

```text
raw.csv
   ↓
cleaned.parquet
   ↓
features.parquet
   ↓
model.pkl
   ↓
predictions.csv
   ↓
report.pdf
```

Bunların hepsi artifact olabilir.

Başka örnekler:

```text
dataset.csv
dataframe
image.png
chart.png
model.pkl
embeddings.parquet
sqlite.db
report.pdf
presentation.pptx
json
markdown
```

Ama burada bir ayrım var:

### Runtime memory

```python
df = pd.read_csv("data.csv")
```

`df` RAM'dedir.

### File artifact

```python
df.to_parquet("cleaned.parquet")
```

Artık diskte bir artifact oluşmuştur.

### Persistent artifact

Bu dosya yalnızca mevcut Python process'i değil, **sonraki workflow step'i veya sonraki agent invocation'ı tarafından da erişilebilir şekilde saklanıyorsa** persistence elde etmiş olursun.

---

# 2. Neden buna ihtiyacımız var?

Şöyle bir agent düşün:

```text
Step 1:
CSV'yi oku ve temizle.

Step 2:
Temizlenmiş veriden feature üret.

Step 3:
Model eğit.

Step 4:
Rapor oluştur.
```

Persistence olmasaydı her step'in birbirine veri aktarması gerekir.

Örneğin:

```text
Step 1
 ↓
df = 5 GB
 ↓
LLM context
 ↓
Step 2
```

Bu saçma olur.

Çünkü 5 GB dataframe'i LLM context'ine sokmak istemezsin.

Bunun yerine:

```text
Step 1
 ↓
cleaned.parquet
 ↓
filesystem/object storage
 ↓
Step 2
 ↓
read cleaned.parquet
```

dersin.

Yani:

> **Artifact persistence, büyük ara sonuçları model context'i yerine dışarıda tutup gerektiğinde yeniden yükleme mekanizmasıdır.**

Bu, agent sistemlerinin ölçeklenmesinde çok önemli.

---

# 3. PTC burada tam olarak nereye giriyor?

Şimdi PTC'yi ekleyelim.

Normal tool calling:

```text
LLM
 ↓
tool A
 ↓
LLM
 ↓
tool B
 ↓
LLM
 ↓
tool C
 ↓
LLM
```

Model her tool sonucunu tekrar görüyor.

PTC:

```text
LLM
 ↓
generate program
 ↓
program
 ├── tool A()
 ├── tool B()
 ├── tool C()
 └── computations
 ↓
final result
 ↓
LLM
```

OpenAI'nin güncel Agents SDK dokümantasyonunda PTC, modelin JavaScript üretip birden fazla eligible tool'u program içinde çağırabilmesi olarak tanımlanıyor. Anthropic'te de programmatic tool calling code-execution container'ı üzerinden gerçekleştiriliyor. ([OpenAI GitHub][1])

Şimdi artifact ekleyelim:

```text
LLM
 ↓
PTC program
 ↓
┌───────────────────────────┐
│ tool A                     │
│ dataframe transformation   │
│ tool B                     │
│ save artifact              │
└────────────┬──────────────┘
             ↓
      cleaned.parquet
             ↓
       artifact store
             ↓
        next step
```

İşte senin başlıktaki konsept bu.

---

# 4. “Persistence” ne demek?

Burada persistence kelimesini doğru anlamak çok önemli.

Persistence:

> **Bir çıktının mevcut execution'ın RAM'inden bağımsız olarak daha sonra tekrar erişilebilir durumda tutulması.**

Örneğin:

```python
df = pd.read_csv("data.csv")
```

sadece memory.

Ama:

```python
df.to_parquet("/workspace/cleaned.parquet")
```

artifact.

Sonra yeni bir step:

```python
df = pd.read_parquet("/workspace/cleaned.parquet")
```

derse artifact yeniden kullanılmış olur.

Daha da ileri gidip container kapanıp daha sonra başka bir runtime açıldığında aynı dosyaya ulaşabiliyorsan:

```text
Container A
   ↓
cleaned.parquet
   ↓
persistent storage
   ↓
Container B
   ↓
read cleaned.parquet
```

artık gerçek anlamda **runtime-independent persistence** elde etmiş olursun.

---

# 5. Üç farklı persistence seviyesini ayır

Bence bunu öğrenirken en önemli ayrım bu.

## Seviye 1 — Process persistence

```text
Python process
 ├── df
 ├── model
 └── variables
```

Process kapanınca hepsi gider.

---

## Seviye 2 — Container/workspace persistence

```text
Sandbox
 ├── cleaned.parquet
 ├── model.pkl
 └── report.pdf
```

Aynı container/workspace sonraki step'te tekrar kullanılırsa dosyalar kalır.

Claude'un güncel Code Execution/Programmatic Tool Calling sisteminde container tekrar kullanılabildiğinde dosyalar ve Python interpreter state'i sonraki request'lerde korunabiliyor. Container lifecycle ve retention ise platform tarafından sınırlandırılıyor. ([Claude Platform][2])

Google ADK'nın Agent Engine sandbox'ında da aynı sandbox task boyunca korunarak değişkenlerin, import'ların ve dosyaların sonraki tool call'larında kullanılabilmesi açıkça belirtiliyor. ([Google GitHub][3])

---

## Seviye 3 — External durable storage

En sağlam yöntem:

```text
Agent
 ↓
sandbox
 ↓
artifact store
 ├── S3
 ├── GCS
 ├── Azure Blob
 ├── database
 └── dedicated file service
```

Artık sandbox silinse bile artifact durur.

Bu production sistemlerde genellikle daha mantıklıdır.

---

# 6. Artifact ile state aynı şey değil

Bu çok önemli.

### Artifact

```text
cleaned.parquet
model.pkl
report.pdf
```

### Agent state

```json
{
  "current_step": 3,
  "dataset": "cleaned.parquet",
  "features": ["age", "income"],
  "model": "model.pkl"
}
```

Yani:

```text
Artifact = fiziksel/lojik çıktı

State = workflow'un nerede ve hangi bilgilerle kaldığı
```

LangGraph bu ikinci probleme checkpointing ile yaklaşır: graph state'i her step'te checkpoint'ler halinde saklanabilir; böylece workflow son başarılı noktadan devam edebilir. ([Docs by LangChain][4])

Bu yüzden production agent mimarisinde genellikle şunları ayrı düşünürsün:

```text
Artifact Store
+
State Store / Checkpointer
+
Execution Runtime
```

---

# 7. Gerçek bir örnek üzerinden gidelim

Kullanıcı:

> “2024 elektrik tüketim verisini analiz et. Eksik değerleri temizle, hava durumuyla birleştir, feature engineering yap, XGBoost modeli eğit ve PDF rapor üret.”

Agent planlıyor:

```text
Step 1 → load electricity data
Step 2 → clean data
Step 3 → join weather data
Step 4 → feature engineering
Step 5 → train model
Step 6 → generate report
```

Artifact'ler:

```text
raw_electricity.csv

cleaned_electricity.parquet

electricity_weather.parquet

features.parquet

xgb_model.pkl

predictions.parquet

feature_importance.png

report.pdf
```

Workflow:

```text
                    USER
                     │
                     ↓
                    LLM
                     │
                     ↓
               PTC Program
                     │
      ┌──────────────┼──────────────┐
      ↓              ↓              ↓
   load_data      weather_api    transform
      │              │              │
      └──────────────┼──────────────┘
                     ↓
            cleaned.parquet
                     │
              Artifact Store
                     │
                     ↓
               Next Step
                     │
            read cleaned.parquet
                     ↓
             weather join
                     ↓
          features.parquet
                     │
              Artifact Store
                     │
                     ↓
            model training
                     ↓
               model.pkl
                     │
                     ↓
                report.pdf
```

Burada **modelin context'ine 5 GB dataframe sokulmuyor.**

Model sadece:

```text
Artifact ID:
artifact_82f3...
Type:
parquet
Path:
...
Schema:
...
Rows:
12,431,921
```

gibi metadata görebilir.

Sonra gerektiğinde runtime:

```python
df = read_artifact("artifact_82f3...")
```

yapar.

---

# 8. Artifact metadata neden önemli?

Gerçek production sisteminde sadece:

```text
cleaned.parquet
```

saklamak yeterli değildir.

Bir artifact registry düşün:

```json
{
  "artifact_id": "art_9281",
  "name": "cleaned-electricity",
  "type": "parquet",
  "size_bytes": 834829102,
  "created_by_step": "step_2",
  "created_at": "...",
  "schema": {
    "timestamp": "datetime",
    "transformer_id": "string",
    "consumption": "float"
  },
  "parent_artifacts": [
    "art_001"
  ],
  "version": 3,
  "checksum": "sha256:..."
}
```

Böylece agent şunu bilir:

```text
Bu dosya nedir?
Kim oluşturdu?
Hangi step'te oluşturuldu?
Hangi input'tan üretildi?
Boyutu ne?
Schema'sı ne?
Son versiyonu hangisi?
```

Bu production-grade artifact management'in temelidir.

---

# 9. Artifact lineage

Buradan çok önemli başka bir kavram çıkıyor:

**Lineage**

Örneğin:

```text
raw.csv
   │
   ▼
cleaned.parquet
   │
   ├──────────────┐
   ▼              ▼
features.parquet  statistics.json
   │
   ▼
model.pkl
   │
   ▼
predictions.csv
   │
   ▼
report.pdf
```

Bunun anlamı:

```text
report.pdf
    ↓
hangi predictions'tan?
    ↓
hangi modelden?
    ↓
hangi feature datasetinden?
    ↓
hangi cleaned datasetten?
    ↓
hangi raw datadan?
```

Bu özellikle debugging için inanılmaz önemli.

Örneğin model yanlış sonuç verdi.

Sen:

```text
report.pdf
 ↓
predictions.csv
 ↓
model.pkl
 ↓
features.parquet
 ↓
cleaned.parquet
```

şeklinde geriye gidebilirsin.

---

# 10. Versioning neden gerekli?

Agent aynı artifact'i tekrar üretebilir.

Örneğin:

```text
features_v1.parquet
features_v2.parquet
features_v3.parquet
```

Ama daha iyisi immutable artifact yaklaşımı:

```text
artifact_id = sha256(content)
```

ve metadata:

```text
artifact:
a8f31...
```

şeklinde tutulabilir.

Böylece:

```text
Step 4
 ↓
artifact A

Step 5
 ↓
model trained using artifact A
```

sonradan artifact A değişmez.

Bu reproducibility açısından çok önemlidir.

---

# 11. “Sonraki step yeniden üretmeden kullanıyor” tam olarak ne demek?

Örneğin:

```text
Step 1:
10 milyon satır veriyi temizle
```

1 saat sürdü.

Output:

```text
cleaned.parquet
```

Step 2:

```text
feature engineering
```

çalışırken workflow çöktü.

Eğer persistence yoksa:

```text
workflow restart
 ↓
Step 1 tekrar
 ↓
1 saat
 ↓
Step 2
```

Ama persistence varsa:

```text
workflow restart
 ↓
artifact registry
 ↓
cleaned.parquet bulunuyor
 ↓
Step 2'ye devam
```

İşte gerçek fayda burada.

---

# 12. Bu aslında caching'e de benziyor

Evet, ama aynı şey değil.

### Cache

> Aynı hesaplamayı tekrar yapmamak.

### Artifact persistence

> Üretilen çalışma çıktısını daha sonra kullanmak.

Örneğin:

```text
raw.csv
 ↓
expensive transformation
 ↓
features.parquet
```

`features.parquet` hem artifact hem cache gibi davranabilir.

Ama artifact'in amacı yalnızca hız değildir.

Aynı zamanda:

```text
sharing
reproducibility
debugging
workflow continuation
versioning
auditability
```

sağlar.

---

# 13. PTC ile artifact persistence neden birbirine çok yakışıyor?

Çünkü PTC programının kendisi küçük bir workflow engine gibi davranabilir.

Mesela model:

```javascript
const data = await load_data();

const cleaned = await clean(data);

await save_artifact("cleaned.parquet", cleaned);

const features = await feature_engineering(
    load_artifact("cleaned.parquet")
);

await save_artifact("features.parquet", features);

const model = await train_model(
    load_artifact("features.parquet")
);

await save_artifact("model.pkl", model);
```

Böylece:

```text
LLM
 │
 └── PTC generated program
       │
       ├── tool
       ├── artifact write
       ├── artifact read
       ├── tool
       ├── artifact write
       └── artifact read
```

Bu oldukça güçlü bir pattern.

---

# 14. Ancak önemli bir ayrım: OpenAI PTC ile Claude PTC aynı runtime modeline sahip değil

Bu çok önemli.

OpenAI'nin güncel Agents SDK dokümantasyonunda PTC programı **fresh hosted V8 environment** üzerinde çalışıyor ve bu ortamın Node.js API'larına, filesystem'a veya network'e erişimi yok. Dolayısıyla OpenAI'nin PTC özelliğini tek başına “persistent filesystem” olarak düşünmemelisin. ([OpenAI GitHub][1])

Buna karşılık Claude'un PTC'si code execution container kullanıyor; container reuse edildiğinde state ve oluşturulan dosyalar korunabiliyor. ([Claude Platform][5])

OpenAI'nin ayrı **Sandbox Agents** yaklaşımı ise persistent workspace kavramını doğrudan destekliyor; model filesystem üzerinde çalışabiliyor, artifact üretebiliyor ve saved sandbox state üzerinden işe devam edebiliyor. ([OpenAI GitHub][6])

Yani:

```text
PTC
≠
Persistent filesystem
```

Fakat:

```text
PTC
+
runtime
+
artifact store
+
state persistence
```

çok güçlü bir architecture oluşturur.

---

# 15. Artifact Store nasıl tasarlanır?

Ben production'da kabaca şöyle düşünürdüm:

```text
                Agent
                  │
                  ▼
          ┌───────────────┐
          │ Artifact API  │
          └───────┬───────┘
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
      Metadata   Blob      Index
        DB       Store
        │         │
     Postgres    S3
```

Metadata DB:

```text
artifact_id
name
type
size
checksum
created_at
run_id
step_id
parent_artifacts
version
```

Blob storage:

```text
s3://agent-artifacts/
    run_123/
        raw.csv
        cleaned.parquet
        features.parquet
        model.pkl
        report.pdf
```

---

# 16. Agent'in artifact API'si nasıl olur?

Örneğin:

```python
artifact = store.put(
    file="features.parquet",
    metadata={
        "type": "dataset",
        "format": "parquet"
    }
)
```

Dönen:

```json
{
  "artifact_id": "art_123",
  "uri": "s3://agent-artifacts/run_42/features.parquet"
}
```

Sonraki step:

```python
artifact = store.get("art_123")

df = pd.read_parquet(artifact.uri)
```

Agent'ın context'inde ise tüm dosyanın kendisi değil:

```text
artifact_id=art_123
type=parquet
rows=12.4M
schema=...
```

bulunur.

Bu **context efficiency** açısından çok değerlidir.

---

# 17. Artifact persistence + failure recovery

Burada senin önceki konun olan error recovery ile birleşiyor.

Örneğin:

```text
Step 1
 ↓
cleaned.parquet ✅

Step 2
 ↓
features.parquet ✅

Step 3
 ↓
training
 ↓
ERROR ❌
```

Agent restart olduğunda:

```text
load state
 ↓
Step 1 already complete
 ↓
Step 2 already complete
 ↓
features.parquet exists
 ↓
resume Step 3
```

Böylece baştan başlamaz.

LangGraph'ın checkpoint persistence yaklaşımı tam olarak bu tür fault-tolerant resumption sağlar; başarısız bir node olduğunda başarılı işlemlerin yeniden çalıştırılmamasını sağlayan pending writes mekanizması da bulunuyor. ([Docs by LangChain][4])

---

# 18. State ve artifact birlikte nasıl çalışıyor?

Bunu şöyle düşün:

```text
                RUN
                 │
      ┌──────────┴──────────┐
      │                     │
      ▼                     ▼
 State Store           Artifact Store
      │                     │
      │                     │
 current_step=3        features.parquet
 run_id=42             model.pkl
 status=running        report.pdf
```

State:

```text
"Şu anda Step 3'teyim."
```

Artifact:

```text
"Step 2'nin çıktısı burada."
```

İkisini birlikte tutarsan agent crash sonrası devam edebilir.

---

# 19. Artifact persistence uzun süreli memory değildir

Bunu da karıştırmamak lazım.

Artifact:

```text
model.pkl
dataset.parquet
report.pdf
```

Long-term memory:

```text
"Bu kullanıcı PDF raporlarını tercih ediyor."
```

State:

```text
"Şu workflow Step 4'te."
```

Üç farklı kavram:

```text
Artifact
   ↓
data/output

State
   ↓
workflow execution

Memory
   ↓
information about user/world
```

LangChain/LangGraph tarafında da short-term memory/checkpointing ile long-term memory ayrı kavramlar olarak ele alınıyor. ([Docs by LangChain][7])

---

# 20. Gerçek sistemde dataframe nasıl persistence edilir?

Dataframe'i doğrudan JSON'a çevirme:

```python
df.to_json(...)
```

çok büyük datasetlerde kötü olabilir.

Genellikle:

```text
CSV
Parquet
Arrow
Feather
DuckDB
SQLite
```

gibi formatlar daha mantıklıdır.

Örneğin:

```python
df.to_parquet("features.parquet")
```

Sonra:

```python
df = pd.read_parquet("features.parquet")
```

Bunun avantajı:

```text
columnar storage
compression
schema
faster reads
partial column selection
```

gibi özelliklerdir.

Örneğin sadece iki kolon:

```python
df = pd.read_parquet(
    "features.parquet",
    columns=["temperature", "consumption"]
)
```

okuyabilirsin.

Büyük agent workflow'ları için bu ciddi fark yaratır.

---

# 21. Artifact persistence için hangi storage?

Küçük local prototype:

```text
/workspace/artifacts/
```

Gayet yeterli.

Orta seviye:

```text
Postgres metadata
+
S3-compatible object storage
```

Production:

```text
Artifact Registry
+
Object Storage
+
Metadata DB
+
Versioning
+
Checksum
+
ACL
+
Lifecycle policy
```

Örneğin:

```text
S3
 ↓
artifact binary

Postgres
 ↓
artifact metadata

Redis
 ↓
temporary coordination
```

---

# 22. Security boyutu çok önemli

Agent dosya oluşturabiliyorsa:

```text
malicious_agent_output
        ↓
/etc/passwd
        ↓
system damage
```

gibi riskler oluşabilir.

Bu yüzden artifact workspace:

```text
sandbox
filesystem isolation
path restrictions
quotas
permissions
```

ile korunmalı.

Dahası multi-tenant sistemde:

```text
User A
 ↓
artifacts_A/

User B
 ↓
artifacts_B/
```

olmalı.

Örneğin Anthropic Files API dokümantasyonunda uploaded files'ın workspace-wide erişime sahip olabildiği ve multi-tenant uygulamalarda user-supplied file IDs'nin doğrudan güvenilmemesi gerektiği özellikle belirtiliyor. ([Claude Platform][8])

Bu çok önemli bir production güvenlik detayıdır.

---

# 23. Artifact lifecycle

Artifact sonsuza kadar tutulmak zorunda değildir.

Örneğin:

```text
temporary artifacts
     ↓
7 days

important artifacts
     ↓
90 days

final outputs
     ↓
1 year
```

Metadata:

```text
retention_policy
expires_at
storage_class
```

tutulabilir.

Cloud ortamında lifecycle rule:

```text
hot storage
 ↓
cold storage
 ↓
delete
```

şeklinde olabilir.

---

# 24. PTC workflow açısından en doğru zihinsel model

Bunu ezberle:

```text
                 LLM
                  │
                  ▼
            PTC Program
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
     Tool       Code       Tool
       │          │          │
       └──────────┼──────────┘
                  ↓
             Artifact
                  │
                  ▼
          Artifact Storage
                  │
                  ↓
             Next Step
                  │
                  ▼
            Read Artifact
```

ve workflow state:

```text
                State Store
                    │
                    ▼
             "Step 4 completed"
```

Bunları birleştir:

```text
             ┌───────────────┐
             │      LLM      │
             └───────┬───────┘
                     ↓
             PTC / Agent Logic
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
   Execution Runtime       State Store
          │
          ↓
   Artifact Store
          │
          ↓
     Next Execution
```

İşte production-grade agent altyapısı buna yaklaşır.

---

# 25. Güncel framework'lerde nasıl karşılık buluyor?

Bugün farklı framework'lerde isimler değişse de fikir benzer.

### OpenAI

PTC ayrı bir mekanizma; persistent workspace ise Sandbox Agents tarafında daha doğrudan ele alınıyor. OpenAI'nin sandbox yaklaşımı dosya sistemi, shell, snapshot ve resume gibi yetenekleri bir persistent workspace etrafında topluyor. ([OpenAI GitHub][1])

### Anthropic

PTC + Code Execution + reusable containers kombinasyonu çok doğal bir artifact persistence modeli sağlıyor. Container yeniden kullanılırsa dosyalar ve Python state'i sonraki request'lere taşınabiliyor. ([Claude Platform][5])

Anthropic ayrıca Files API ile dosyayı bir kez yükleyip daha sonra `file_id` ile tekrar kullanma modelini destekliyor. ([Claude Platform][8])

### Google ADK

Agent Engine sandbox'u session boyunca sandbox state'ini koruyarak variables, imports ve file state'in sonraki execution'larda kullanılmasını sağlıyor. ([Google GitHub][3])

### LangGraph

Daha çok workflow state/checkpointing tarafında çok güçlü. Checkpoint'ler ile state her step'te saklanarak workflow'un fault-tolerant biçimde devam etmesi sağlanıyor. ([Docs by LangChain][4])

Deep Agents tarafında da filesystem backend'leri thread seviyesinde persistence veya store tabanlı thread'ler arası durable storage sağlayabiliyor. ([Docs by LangChain][9])

---

# 26. Senin yazdığın tanım aslında neyi anlatıyor?

Senin:

> **“Çok adımlı workflow'larda üretilen dosya, dataframe ve ara çıktıların kalıcı olarak saklanmasıdır. Sonraki adımların aynı artifact'leri yeniden üretmeden kullanabilmesini sağlar.”**

ifaden **temel olarak doğru**.

Ama teknik açıdan daha güçlü hale getirirsek:

> **PTC: Artifact Persistence**, programmatic tool-calling ile yürütülen çok adımlı agent workflow'larında oluşturulan dosya, dataframe, model, grafik ve diğer ara çıktıların execution state'inden bağımsız şekilde artifact olarak saklanmasını ve sonraki workflow adımlarının bu artifact'leri yeniden hesaplamadan veya LLM context'ine taşımadan referanslayıp kullanabilmesini sağlayan mekanizmadır.

Bu versiyonda üç önemli fikir var:

```text
1. store
2. reference
3. reuse
```

ve bunlar kavramın özünü oluşturuyor.

---

# 27. En basit kodla gösterirsek

Mesela kendi mini agent runtime'ını yazıyorsun:

```python
from pathlib import Path
import pandas as pd

ARTIFACT_DIR = Path("./artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


def save_artifact(name: str, obj):
    path = ARTIFACT_DIR / name

    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(path)
    else:
        raise TypeError("Unsupported artifact")

    return str(path)


def load_artifact(name: str):
    path = ARTIFACT_DIR / name
    return pd.read_parquet(path)
```

Step 1:

```python
df = pd.read_csv("raw.csv")

df = df.dropna()

artifact_id = save_artifact(
    "cleaned.parquet",
    df
)
```

Step 2:

```python
df = load_artifact("cleaned.parquet")

df["rolling_mean"] = (
    df["consumption"]
    .rolling(24)
    .mean()
)

save_artifact(
    "features.parquet",
    df
)
```

Burada:

```text
Step 1
   ↓
cleaned.parquet
   ↓
Step 2
   ↓
features.parquet
```

İşte **artifact persistence'in en basit hali** bu.

Production'da bunun üzerine:

```text
artifact IDs
metadata
hash
versioning
storage backend
ACL
lineage
TTL
checkpointing
```

eklersin.

---

# 28. Son olarak en önemli nokta

Bu konsepti bence şu cümleyle kafana oturt:

> **LLM'in context'i her şeyi taşımak için değil, artifact sistemi ise büyük çalışma ürünlerini taşımak için kullanılır.**

Yani:

```text
LLM Context
    ↓
"features.parquet dosyasını kullan"

Artifact Store
    ↓
500 MB / 5 GB gerçek veri
```

Agent:

```text
reason with metadata
        +
execute against artifact
```

şeklinde çalışır.

Bu mimari sayesinde agent:

```text
çok büyük dataset
      +
çok adımlı workflow
      +
uzun süren computation
      +
restart/resume
      +
parallel agents
```

gibi işlerde çok daha ölçeklenebilir hale gelir.

Özellikle senin agent/sandbox/egress-policy tarafında düşündüğün mimaride, **“sandbox filesystem” ile “durable artifact store”u birbirinden ayırmak** önemli olur: sandbox execution içindir; artifact store workflow çıktılarının yaşam döngüsü içindir. Claude, OpenAI ve Google'ın güncel agent/sandbox yaklaşımlarında bu ayrımın farklı biçimlerde uygulandığını görebiliyoruz. ([Claude Platform][2])

### Kaynaklar

Anthropic — Programmatic Tool Calling: ([Claude Platform][5])
Anthropic — Code Execution / Containers / File Persistence: ([Claude Platform][2])
Anthropic — Files API: ([Claude Platform][8])
OpenAI Agents SDK — Programmatic Tool Calling: ([OpenAI GitHub][1])
OpenAI Agents SDK — Persistent Sandbox Agents: ([OpenAI GitHub][6])
Google ADK — Persistent Agent Runtime Sandbox: ([Google GitHub][3])
LangGraph — Persistence / Checkpointing: ([Docs by LangChain][4])

[1]: https://openai.github.io/openai-agents-python/tools/?utm_source=chatgpt.com "Tools - OpenAI Agents SDK"
[2]: https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool?cmid=b38d5b4f-e65c-4994-b7ff-02472f785995&utm_source=chatgpt.com "Code execution tool - Claude Platform Docs"
[3]: https://google.github.io/adk-docs/tools/google-cloud/code-exec-agent-engine/?utm_source=chatgpt.com "Agent Runtime Code Execution tool for ADK - Agent Development Kit (ADK)"
[4]: https://docs.langchain.com/oss/python/langgraph/persistence?utm_source=chatgpt.com "Persistence - Docs by LangChain"
[5]: https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling?cd=1966&utm_source=chatgpt.com "Programmatic tool calling - Claude Platform Docs"
[6]: https://openai.github.io/openai-agents-python/sandbox_agents/?utm_source=chatgpt.com "Quickstart - OpenAI Agents SDK"
[7]: https://docs.langchain.com/oss/python/langchain/short-term-memory?utm_source=chatgpt.com "Short-term memory - Docs by LangChain"
[8]: https://platform.claude.com/docs/en/build-with-claude/files?utm_source=chatgpt.com "Files API - Claude Platform Docs"
[9]: https://docs.langchain.com/oss/python/deepagents/backends?utm_source=chatgpt.com "Backends - Docs by LangChain"




İkisi de mümkün kardeşim; **“workflow” kelimesi PTC içinde step-step kod yazılmasıyla sınırlı değil.** Buradaki kritik ayrım, **PTC’nin execution primitive olması**, workflow’un ise bunun üstündeki orkestrasyon yapısı olması.

Şöyle düşün:

### 1. Tek PTC içinde aşama aşama

Model tek bir program üretir:

```python
data = load_data()

cleaned = clean(data)
save_artifact("cleaned.parquet", cleaned)

features = make_features(cleaned)
save_artifact("features.parquet", features)

model = train(features)
save_artifact("model.pkl", model)

report = generate_report(model, features)
save_artifact("report.pdf", report)
```

Burada:

```text
1 PTC execution
    ├── Step 1
    ├── Step 2
    ├── Step 3
    └── Step 4
```

Bu da **workflow** sayılır.

Artifact persistence burada çok doğal çünkü step 2, step 3'ün kullanacağı artifact'i oluşturuyor.

---

### 2. Node-based workflow

Daha gelişmiş agent sisteminde workflow dışarıdan tanımlanabilir:

```text
          ┌───────────┐
          │  Node A   │
          │ Data Load │
          └─────┬─────┘
                ↓
          ┌───────────┐
          │  Node B   │
          │  Clean    │
          └─────┬─────┘
                ↓
          ┌───────────┐
          │  Node C   │
          │ Features  │
          └─────┬─────┘
                ↓
          ┌───────────┐
          │  Node D   │
          │ Training  │
          └───────────┘
```

Burada her node:

* ayrı bir LLM çağrısı olabilir,
* ayrı bir PTC execution olabilir,
* normal bir tool olabilir,
* deterministic Python function olabilir.

Yani:

```text
Workflow
   │
   ├── Node A → PTC #1
   ├── Node B → Python
   ├── Node C → PTC #2
   └── Node D → PTC #3
```

olabilir.

---

## 3. “Birden fazla PTC” de olabilir

Evet, tam olarak.

Örneğin:

```text
Workflow
│
├── PTC #1
│     └── data acquisition
│
├── PTC #2
│     └── cleaning + transformation
│
├── PTC #3
│     └── model training
│
└── PTC #4
      └── report generation
```

Artifact'ler aralarında köprü olur:

```text
PTC #1
   ↓
raw.parquet
   ↓
PTC #2
   ↓
cleaned.parquet
   ↓
PTC #3
   ↓
model.pkl
   ↓
PTC #4
   ↓
report.pdf
```

Burada artifact persistence **çok daha anlamlı** hale geliyor.

Çünkü PTC #2 bittiğinde PTC #3'ün aynı Python process'inde olması gerekmiyor.

---

# 4. Hatta node ≠ PTC

Burası önemli.

Bir workflow node'u şöyle olabilir:

```text
Node
├── LLM reasoning
├── PTC execution
├── normal tool call
├── human approval
├── database query
└── deterministic function
```

Dolayısıyla:

> **PTC bir node tipi olabilir.**

Ama:

> **Her node PTC değildir.**

---

# 5. En genel mimari

Production-grade bir agent workflow'u şöyle düşünebilirsin:

```text
                  WORKFLOW
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
      Node A       Node B       Node C
        │            │            │
       PTC          PTC        Function
        │            │            │
        ↓            ↓            ↓
      Artifact ───── Artifact ──── Artifact
```

ve state ayrı:

```text
Workflow
   │
   ├── State Store
   │
   └── Artifact Store
```

---

# 6. PTC'nin kendi içindeki step'ler ile workflow node'larını ayır

Bunu özellikle aklında tut:

```text
WORKFLOW LEVEL
──────────────────────────

Node 1
  ↓
Node 2
  ↓
Node 3

     Node 2:
     ┌──────────────────┐
     │ PTC execution    │
     │                  │
     │ step A           │
     │ step B           │
     │ step C           │
     └──────────────────┘
```

Yani hiyerarşi olabilir:

```text
Workflow
 ├── Node 1
 ├── Node 2
 │     └── PTC
 │          ├── tool call
 │          ├── computation
 │          └── artifact creation
 └── Node 3
```

Bu durumda **PTC execution ile workflow orchestration iki farklı abstraction layer**.

---

# 7. Senin cümlendeki “multi-step workflow” hangisini ifade ediyor?

İkisini de kapsayabilir.

Ama teknik bir dokümanda:

> **multi-step workflow**

dendiğinde çoğu zaman yalnızca:

```text
tek PTC
  ├── step1
  ├── step2
  └── step3
```

anlamına gelmez.

Daha genel anlamı:

```text
birden fazla execution aşamasının
birbirinin çıktısını kullanarak ilerlemesi
```

dir.

Bu aşamalar:

```text
tek PTC içindeki kod blokları

veya

farklı PTC execution'ları

veya

farklı workflow node'ları
```

olabilir.

---

# 8. Artifact persistence açısından en önemli senaryo

Asıl fark şu:

### Tek PTC

```text
PTC
 ├── step1
 ├── step2
 ├── step3
 └── step4
```

Burada artifact'i sadece memory'de tutmak bile mümkün:

```python
a = step1()
b = step2(a)
c = step3(b)
```

Persistence zorunlu olmayabilir.

---

### Birden fazla node/PTC

```text
PTC #1
   ↓
artifact
   ↓
PTC #2
   ↓
artifact
   ↓
PTC #3
```

Burada persistence **çok daha kritik**, çünkü PTC #2'nin PTC #1'in Python memory'sine erişmesi gerekmiyor.

---

# 9. En güzel örnek

Mesela bir data science agent:

```text
              USER
               │
               ▼
          ORCHESTRATOR
               │
      ┌────────┼─────────┐
      ↓        ↓         ↓
   Acquire   Analyze   Train
      │        │         │
     PTC      PTC       PTC
      │        │         │
      ▼        ▼         ▼
  raw.parq  feat.parq  model.pkl
      │        │         │
      └────────┴─────────┘
               ↓
             Report
               │
              PTC
```

Burada **artifact store node'lar arasında veri taşıyan ortak katman** haline geliyor.

Bence senin baktığın kavramı anlamak için en doğru zihinsel model bu:

> **PTC = “çalıştırma mekanizması”**
>
> **Node = “workflow'daki bir aşama”**
>
> **Workflow = “bu aşamaların orkestrasyonu”**
>
> **Artifact persistence = “aşamalar arasındaki kalıcı veri/çıktı taşıma katmanı”**

Dolayısıyla **“workflow illa tek PTC içindeki step'lerdir” diye düşünme.** Bir workflow, tek PTC de olabilir; **birden fazla PTC ve farklı node'lardan oluşan daha büyük bir graph da olabilir.**

İşin güzel tarafı, bu ayrımı oturttuğunda **LangGraph, OpenAI Agents, Claude Agent SDK, Temporal gibi sistemlerde node / execution / state / artifact kavramlarının neden ayrı tutulduğunu** çok daha rahat anlayacaksın.


Bence bunu kurarken **“sandbox” ile “artifact store”u iki ayrı ama konuşan katman** olarak tasarlamalısın. En temiz model şu:

```text
                    ORCHESTRATOR
                         │
                  ┌──────┴──────┐
                  │             │
              LLM / ReAct     State DB
                  │
                  ▼
             PTC execution
                  │
                  ▼
             ┌───────────┐
             │  Sandbox  │
             │           │
             │ /workspace│
             │           │
             │  a.py     │
             │  df.parquet
             │  plot.png │
             └─────┬─────┘
                   │
             artifact API
                   │
                   ▼
            ┌──────────────┐
            │ Artifact     │
            │ Store        │
            │              │
            │ S3 / GCS     │
            │ + metadata DB│
            └──────────────┘
```

Burada kritik fikir:

> **Sandbox = agent'ın çalıştığı geçici/çalışma ortamı**
> **Artifact Store = agent'ın ürettiği değerli çıktıların kalıcı deposu**

OpenAI'nin güncel Sandbox Agents yaklaşımı da persistent workspace, sandbox lifecycle, snapshots ve remote storage mount'larını ayrı kavramlar halinde ele alıyor; Anthropic PTC ise code execution'ı sandbox container içinde çalıştırıyor ve container çıktılarının retention'ını ayrıca yönetiyor. ([OpenAI GitHub][1])

## 1. En basit senaryo

Agent:

> “CSV'yi temizle.”

PTC:

```python
df = pd.read_csv("/workspace/input.csv")
df = clean(df)
df.to_parquet("/workspace/cleaned.parquet")
```

Sandbox'ta:

```text
/workspace/
    input.csv
    cleaned.parquet
```

Şimdi agent'ın işi bitmiş olabilir.

Ama `cleaned.parquet` önemli bir output ise:

```text
/workspace/cleaned.parquet
           │
           ▼
      artifact.put()
           │
           ▼
S3: runs/run_123/cleaned.parquet
```

Böylece sandbox silinse bile artifact kalır.

---

# 2. Neden her sandbox dosyasını artifact yapmamalısın?

Çünkü sandbox'ta çok fazla gereksiz şey olabilir:

```text
/workspace/
    tmp1.txt
    python_cache/
    debug.log
    test.py
    cleaned.parquet      ← önemli
    model.pkl            ← önemli
    report.pdf           ← önemli
```

Hepsini persistent storage'a atmak gereksiz.

O yüzden:

```text
Sandbox
   ├── temporary files
   ├── intermediate files
   └── artifacts
          ↓
      persist
```

diye düşün.

Bir artifact ancak **sonraki step/turn'de kullanılacaksa veya kullanıcıya dönecekse** persist edilebilir.

---

# 3. PTC ile konuşma nasıl olmalı?

Ben `PTC`'yi doğrudan S3 gibi storage'a erişen bir şey yapmazdım.

Şöyle:

```text
PTC
 │
 ├── filesystem
 ├── shell
 ├── python
 └── artifact tool
```

Mesela PTC'nin görebileceği tool'lar:

```python
artifact_save(...)
artifact_load(...)
artifact_info(...)
artifact_list(...)
```

Örnek:

```python
df = load_data()

df = clean(df)

artifact_id = artifact_save(
    path="/workspace/cleaned.parquet",
    name="cleaned_dataset"
)
```

Tool:

```text
artifact_save()
      ↓
hash file
      ↓
upload object storage
      ↓
write metadata DB
      ↓
return artifact_id
```

Sonuç:

```json
{
  "artifact_id": "art_82913",
  "name": "cleaned_dataset",
  "type": "parquet",
  "size": 834829102
}
```

LLM'e 800 MB dataframe gönderilmiyor.

Sadece:

```text
artifact_id=art_82913
```

ve metadata geliyor.

Bu PTC'nin en güzel taraflarından biri: Anthropic'in PTC modelinde intermediate tool sonuçları model context'ine geri yüklenmeden code execution içinde işlenebiliyor. ([Claude Platform][2])

---

# 4. Sonraki PTC nasıl kullanacak?

Diyelim ikinci PTC başka bir turn'de başladı.

```text
Turn 1
PTC #1
    ↓
artifact: art_82913
```

Turn 2:

```text
User:
"Şimdi feature engineering yap."
```

Orchestrator state'ten:

```json
{
  "last_artifact": "art_82913"
}
```

bulur.

PTC #2:

```python
df = artifact_load("art_82913")

features = make_features(df)

artifact_save(
    features,
    name="features"
)
```

Böylece:

```text
PTC #1
   ↓
Artifact Store
   ↓
PTC #2
```

olur.

---

# 5. Burada State DB neden ayrıca var?

Çünkü artifact store şunu bilir:

```text
art_82913 = cleaned.parquet
```

Ama workflow state şunu bilmelidir:

```text
run_123
current_step = feature_engineering
input_artifact = art_82913
```

Dolayısıyla:

```text
               Workflow
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
    State DB             Artifact Store
        │                    │
        │                    │
 current_step=2        art_82913
 run_id=123            art_73192
 status=running        art_99173
```

**State ≠ artifact.**

Bu ayrımı koruman çok önemli.

---

# 6. Ben workflow ID / run ID / turn ID / PTC execution ID'lerini de ayırırdım

Örneğin:

```text
workflow_id = wf_123
run_id      = run_456
turn_id     = turn_7
ptc_id      = ptc_9
```

Artifact metadata:

```json
{
  "artifact_id": "art_82913",
  "workflow_id": "wf_123",
  "run_id": "run_456",
  "turn_id": "turn_7",
  "ptc_id": "ptc_9",
  "name": "cleaned_dataset",
  "type": "parquet"
}
```

Böylece sonra:

> “Bu dosyayı kim oluşturdu?”

diye sorabilirsin.

---

# 7. Asıl sevdiğim architecture

Ben şöyle yapardım:

```text
                         USER
                          │
                          ▼
                     ORCHESTRATOR
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
              LLM                STATE DB
                │
                ▼
              PTC
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
     Python   Shell    Tools
        │
        ▼
     Sandbox
        │
        ├── /workspace/tmp
        ├── /workspace/output
        └── /workspace/artifacts
                     │
                     ▼
               Artifact API
                     │
                     ▼
              Artifact Store
               ┌─────┴─────┐
               ▼           ▼
              S3        Metadata DB
```

Bu bence çok temiz.

---

# 8. Sandbox'ı “artifact store” yapma

Mesela:

```text
docker container
    ↓
/workspace
```

sandbox olarak kullan.

Ama:

```text
docker container
    ↓
tek persistent storage
```

yapma.

Çünkü sandbox:

* crash olabilir
* silinebilir
* yeniden oluşturulabilir
* farklı runtime'a taşınabilir
* ölçeklenebilir

Artifact store ise:

* durable
* versioned
* addressable
* erişim kontrollü

olmalı.

OpenAI'nin güncel sandbox docs'unda da workspace state/snapshot ile mounted remote storage birbirinden ayrılıyor; mount edilen remote storage snapshot'ın kalıcı workspace içeriği olarak kopyalanmıyor. ([OpenAI GitHub][1])

---

# 9. Çok önemli: artifact ID kullan

Ben şöyle bir sistem **öneririm**:

```text
artifact://art_123
```

ve LLM:

```text
Use artifact art_123.
```

der.

Runtime bunu:

```text
artifact_id
   ↓
metadata DB
   ↓
storage URI
```

ile çözer.

Örneğin:

```json
{
  "artifact_id": "art_123",
  "uri": "s3://agent-artifacts/wf_42/art_123.parquet"
}
```

Böylece LLM'e:

```text
s3://...
```

gibi internal infrastructure detaylarını vermek zorunda kalmazsın.

---

# 10. Daha da iyi: artifact versioning

Mesela:

```text
cleaned_data
   ├── v1
   ├── v2
   └── v3
```

Ama agent'ın:

```text
artifact_id = art_123
```

demesi daha güvenli.

Artifact immutable olsun:

```text
art_123 → cleaned_v1
```

Yeni sürüm:

```text
art_456 → cleaned_v2
```

Eski artifact değişmesin.

Bu reproducibility için çok değerli.

---

# 11. Lineage de tut

Şunu yapmak çok güzel olur:

```text
raw.csv
   ↓
art_001
   ↓
clean
   ↓
art_002
   ↓
feature engineering
   ↓
art_003
   ↓
training
   ↓
art_004
```

Metadata:

```json
{
  "artifact_id": "art_003",
  "parents": ["art_002"],
  "created_by": "ptc_9"
}
```

Artık:

> “Bu model hangi veriden eğitildi?”

sorusunun cevabı var.

---

# 12. Farklı turn'ler için tam senaryo

İşte senin önceki soruna doğrudan bağlanan örnek:

### Turn 1

```text
User
"CSV'yi temizle"

     ↓

PTC #1
     ↓

cleaned.parquet
     ↓

Artifact Store
     ↓

art_001
```

### Turn 2

```text
User
"Feature engineering yap"

     ↓

State DB
     ↓

last artifact = art_001

     ↓

PTC #2
     ↓

read art_001
     ↓

features.parquet
     ↓

Artifact Store
     ↓

art_002
```

### Turn 3

```text
User
"Model eğit"

     ↓

PTC #3
     ↓

read art_002
     ↓

model.pkl
     ↓

Artifact Store
     ↓

art_003
```

Final:

```text
Turn 1       Turn 2       Turn 3
   │            │            │
 PTC #1       PTC #2       PTC #3
   │            │            │
   ▼            ▼            ▼
art_001  →   art_002   →   art_003
```

**İşte senin “workflow” dediğin şey burada birden fazla turn + birden fazla PTC olabilir.**

---

# 13. Ama küçük artifact'ler için optimizasyon

Her şeyi S3'e koyup geri indirmek de gereksiz olabilir.

Örneğin:

```text
42
"completed"
{"mean": 14.2}
```

gibi küçük sonuçları direkt state/context'te tutabilirsin.

Karar mekanizması:

```text
result
  │
  ├── küçük + semantik bilgi → state/context
  │
  └── büyük + reusable       → artifact
```

Örneğin:

```text
12 KB JSON
    → context

800 MB Parquet
    → artifact

2 GB model
    → artifact

PDF
    → artifact
```

---

# 14. Bence API'nin şu 5 operasyonu yeterli başlangıç için

```python
artifact.create()
artifact.get()
artifact.list()
artifact.delete()
artifact.metadata()
```

Biraz daha production:

```python
artifact.create()
artifact.get()
artifact.list()
artifact.delete()
artifact.metadata()

artifact.copy()
artifact.version()
artifact.lineage()
artifact.download()
artifact.upload()
```

---

# 15. En önemli güvenlik konusu

PTC sandbox'ta çalışıyorsa:

```text
PTC
 ↓
artifact_save()
 ↓
S3
```

modelin kafasına göre:

```python
artifact_save("/etc/shadow")
```

yapabilmesini istemezsin.

Artifact API:

```text
allowed workspace paths
allowed mime types
max size
tenant isolation
authentication
authorization
```

kontrolü yapmalı.

Ayrıca artifact ID ile erişim yapılırken:

```text
art_123
```

kullanıcının gerçekten o artifact'e erişip erişemediğini server tarafı kontrol etmeli.

---

# 16. Production için benim önerdiğim son architecture

```text
                         ┌───────────────┐
                         │      LLM      │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ Orchestrator  │
                         └───────┬───────┘
                                 │
                       ┌─────────┴─────────┐
                       ▼                   ▼
                  State Store         PTC Runtime
                  PostgreSQL          Sandbox/VM
                                           │
                              ┌────────────┼───────────┐
                              ▼            ▼           ▼
                           Python        Shell       Tools
                              │
                              ▼
                         /workspace
                              │
                       artifact.save()
                              │
                              ▼
                     ┌─────────────────┐
                     │ Artifact Service│
                     └───────┬─────────┘
                             │
                     ┌───────┴────────┐
                     ▼                ▼
                 Object Store      Metadata DB
                     │                │
                    S3          artifact_id
                                version
                                lineage
                                owner
                                run_id
```

Ve çalışma prensibi:

```text
Sandbox = "çalış burada"

Artifact Store = "önemli şeyleri burada sakla"

State Store = "workflow nerede kaldı?"

LLM Context = "şu anda bilmem gereken özet bilgi burada"
```

Bence **en doğru ayrım tam olarak bu**.

OpenAI'nin güncel sandbox mimarisinde de sandbox session/workspace ile runner'ın tuttuğu resume state ayrılmış durumda; sandbox session dosya değişikliklerini ve izolasyonu sahiplenirken outer runtime approval, tracing ve resume için gereken state'i yönetiyor. ([OpenAI GitHub][1])

Ve Claude PTC'de de programın sandbox container içinde çalışması, intermediate sonuçların context'e sokulmaması ve container execution artifact'lerinin ayrı retention mekanizmasına sahip olması bu ayrımın pratikte neden gerekli olduğunu gösteriyor. ([Claude Platform][2])

**Özetle benim tasarım kararım:**
`PTC → Sandbox → Artifact Service → Object Storage`, bunun yanında `Orchestrator → State DB`.

Bu mimaride **aynı workflow içindeki farklı PTC'ler, farklı turn'lerdeki PTC'ler, hatta workflow yeniden başladıktan sonraki PTC'ler** aynı artifact'leri kullanabilir.

[1]: https://openai.github.io/openai-agents-python/sandbox/guide/?utm_source=chatgpt.com "Concepts - OpenAI Agents SDK"
[2]: https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling?cd=1966&utm_source=chatgpt.com "Programmatic tool calling - Claude Platform Docs"


Tarama sonucu oldukça net bir ortak mimari çıkıyor. **Sandbox ile artifact storage aynı şey yapılmıyor**; modern sistemler çoğunlukla execution ile durable storage'ı ayırıyor. Ama sandbox'ın *ne kadar izole* olması gerektiği workload'un güven seviyesine göre değişiyor.

## 1. Genel kanı: Sandbox nerede çalışmalı?

Üç ana yaklaşım var:

| Yaklaşım                | İzolasyon  | Başlangıç hızı | Yönetim  | Kullanım                  |
| ----------------------- | ---------- | -------------: | -------- | ------------------------- |
| Normal container        | Orta       |      Çok hızlı | Kolay    | Güvenilir/controlled code |
| gVisor / Kata / benzeri | Yüksek     |          Hızlı | Orta     | Untrusted AI code         |
| microVM / VM            | Çok yüksek |     Daha yavaş | Daha zor | Gerçekten untrusted code  |

### A. Container

En yaygın temel yaklaşım:

```text
Agent
  ↓
Container
  ↓
Python / Node / Shell
```

Daytona örneğin Linux container'ları varsayılan sandbox olarak kullanıyor; her sandbox'a ayrı CPU, RAM, disk ve filesystem veriyor. ([daytona.io][1])

Modal da sandbox'ları izole container'lar olarak sunuyor ve çok yüksek concurrency hedefliyor. ([Modal][2])

Bu modelin büyük avantajı:

```text
container start
    ↓
çok hızlı
    ↓
code execute
```

Ama önemli sorun:

> Container'ın kernel'i host kernel'iyle paylaşılır.

Dolayısıyla gerçekten saldırgan/untrusted code çalıştırıyorsan, yalnızca `Docker container` demek güvenlik açısından yeterli olmayabilir.

---

## 2. Daha güvenli ortak yaklaşım: gVisor / Kata

gVisor container ile host kernel arasına ek bir isolation layer koyuyor. Kendi dokümantasyonu da amacını container'ları host Linux kernel'inden ve birbirlerinden izole etmek olarak açıklıyor. Bunun karşılığında performans overhead'i var. ([gvisor.dev][3])

Kubernetes tarafında örneğin:

```text
Kubernetes
    ↓
Pod
    ↓
gVisor runtime
    ↓
Sandbox
```

şeklinde kullanılabiliyor. ([gvisor.dev][4])

Bu nedenle AI-generated code için bence:

```text
normal Docker
      ↓
     gVisor
```

veya

```text
Kata Containers
```

gibi ara katmanlar mantıklı.

---

# 3. En güçlü izolasyon: microVM

Özellikle:

> “LLM ne kod üretirse üretsin çalıştıracağım.”

diyorsan microVM çok daha rahat bir güvenlik sınırı oluşturuyor.

Örneğin Firecracker yaklaşımı:

```text
Host
 ├── microVM 1
 │     └── Agent code
 │
 ├── microVM 2
 │     └── Agent code
 │
 └── microVM 3
       └── Agent code
```

Her VM kendi kernel'ine sahip.

2026'da agent sandbox sektöründe bu model oldukça görünür hale gelmiş durumda; örneğin Daytona artık gerçek microVM tabanlı sandbox seçeneği sunuyor ve kendi dokümanında bunu machine-level isolation olarak tanımlıyor. ([daytona.io][5])

E2B de Firecracker microVM tabanlı sandbox yaklaşımını kullanıyor; güncel pazar karşılaştırmalarında da E2B'nin temel izolasyon modeli Firecracker olarak gösteriliyor. ([beam.cloud][6])

---

# 4. Cloud tarafındaki genel pattern

Cloud provider'larda da aynı fikir var.

Örneğin Google'ın Cloud Run sandbox özelliği AI agent'lar gibi untrusted code/tool çalıştırmaya yönelik sandbox sağlıyor. ([Google Cloud Documentation][7])

Google ADK Agent Runtime'da ise:

```text
Agent
 ↓
Sandbox Environment
 ↓
Code Execution
```

ve sandbox task/session boyunca korunabiliyor:

```text
PTC #1
 ↓
sandbox
 ↓
PTC #2
 ↓
same sandbox
 ↓
PTC #3
```

Böylece variables, imports ve file state sonraki execution'lara taşınabiliyor. ([Google GitHub][8])

OpenAI'nin Sandbox Agents yaklaşımı da agent'a gerçek filesystem üzerinde çalışabileceği persistent workspace veriyor ve sandbox lifecycle/snapshot/resume gibi kavramları birinci sınıf hale getiriyor. ([OpenAI GitHub][9])

Anthropic ise PTC'de code execution'ı sandboxed container içinde çalıştırıyor; self-managed kullanımda özellikle network isolation gibi güvenlik kısıtlarını belirtip, managed execution seçeneğinde sandbox yönetimini kendisi üstleniyor. ([Claude Platform][10])

---

# 5. Buradan çıkan genel mimari

Ben internetteki güncel örnekleri bir araya getirdiğimde şu pattern'in oldukça güçlü olduğunu görüyorum:

```text
                    AGENT
                      │
                      ▼
               ORCHESTRATOR
                      │
                      ▼
               Sandbox Manager
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
      Sandbox #1              Sandbox #2
      container/VM             container/VM
          │                       │
      Python/shell             Python/shell
          │                       │
          └──────────┬────────────┘
                     ▼
               Artifact Store
```

Yani **sandbox compute**, artifact ise **durable data**.

---

# 6. Peki artifactler için genel görüş ne?

Burada da çok net bir ortak yön var:

> **Artifact'i sandbox filesystem'inin kalıcı hali olarak düşünmek yerine, durable object storage üzerinde tutmak daha doğru.**

Örneğin:

```text
Sandbox
   │
   │ save
   ▼
S3 / GCS / Azure Blob
```

S3, GCS ve Azure Blob zaten doğal olarak bu model için tasarlanmış durumda.

AWS S3:

* object versioning,
* lifecycle,
* replication,
* durability

gibi özellikler sağlıyor. S3 Standard için AWS, verinin birden fazla Availability Zone'a yayıldığını ve 99.999999999% durability hedeflediğini belirtiyor. ([AWS Belgeleri][11])

S3 Versioning ile eski artifact sürümlerini koruyabiliyorsun. ([AWS Belgeleri][12])

Lifecycle ile eski artifact'leri:

```text
hot storage
    ↓
cold/archive
    ↓
delete
```

yapabiliyorsun. ([AWS Belgeleri][13])

GCS ve Azure Blob da object versioning / retention / immutability gibi aynı familyadan özellikler sunuyor. ([Google Cloud Documentation][14])

Dolayısıyla:

```text
artifact
    ↓
object storage
```

çok güçlü bir default.

---

# 7. Ama artifact'in kendisi sadece S3 dosyası değildir

Bu bence çok önemli.

Şunu yapma:

```text
s3://bucket/report.pdf
```

ve bitti.

Bunun yanında metadata DB tut:

```text
artifact_id
workflow_id
run_id
turn_id
ptc_execution_id
name
type
size
hash
storage_uri
created_at
parent_artifacts
version
owner
ttl
```

Örneğin:

```json
{
  "artifact_id": "art_92f1",
  "type": "parquet",
  "size": 824923123,
  "hash": "sha256:abc...",
  "storage_uri": "s3://agent-artifacts/run_123/features.parquet",
  "parents": ["art_73e1"],
  "workflow_id": "wf_42",
  "run_id": "run_123"
}
```

Böylece artifact store artık sadece blob storage değil:

> **artifact registry + object storage**

oluyor.

---

# 8. Sandbox persistence ile artifact persistence arasında önemli fark

Burada senin önceki sorularının tamamı birleşiyor.

### Sandbox persistence

```text
PTC #1
 ↓
Sandbox
 ↓
file exists
 ↓
PTC #2
```

Ama sandbox yaşamaya devam etmek zorunda.

### Artifact persistence

```text
PTC #1
 ↓
Sandbox
 ↓
save()
 ↓
S3
```

sonra:

```text
Sandbox #1
    X
```

silinse bile:

```text
S3
 ↓
artifact
```

duruyor.

Sonraki PTC:

```text
Sandbox #2
 ↓
load(artifact_id)
```

yapıyor.

**Bu nedenle artifact persistence sandbox persistence'tan daha güçlü bir persistence türü.**

---

# 9. Çok kritik bir production yaklaşımı: ikisini birlikte kullan

Bence en iyi pattern:

```text
                 PTC #1
                   │
                   ▼
                Sandbox
                   │
          ┌────────┴────────┐
          │                 │
       temp files       important output
          │                 │
          X                 ↓
                     Artifact Store
                            │
                            ↓
                         PTC #2
                            │
                            ↓
                        Sandbox
```

Yani:

### Sandbox

Geçici/intermediate:

```text
/tmp
/build
/cache
python environment
compiled files
```

### Artifact Store

Kalıcı/reusable:

```text
dataset
model
report
image
parquet
csv
checkpoint
```

---

# 10. Peki sandbox nerede fiziksel olarak çalışmalı?

Burada benim önerim workload'a göre üç seviyeli.

### Geliştirme / internal agent

```text
Docker container
```

yeterli olabilir.

### Production + untrusted LLM code

```text
gVisor / Kata
```

daha mantıklı.

### Çok yüksek riskli arbitrary code

```text
microVM / dedicated VM
```

tercihim olur.

Bu ayrım gVisor'ın kendi security/performance trade-off dokümanıyla da uyumlu. ([gvisor.dev][3])

---

# 11. Kubernetes kullanmalı mısın?

Eğer:

```text
1000+ concurrent agents
multiple workers
multi-tenant
GPU
autoscaling
```

gibi ihtiyaçların varsa Kubernetes mantıklı.

Örneğin:

```text
Kubernetes
│
├── Agent Controller
│
├── Sandbox Pod
│     └── gVisor
│
├── Sandbox Pod
│     └── gVisor
│
└── Sandbox Pod
      └── gVisor
```

Ama küçük bir sistem için:

> **Kubernetes'i sırf agent sandbox yapmak için kullanmak gereksiz operasyon yükü olabilir.**

Bu yüzden Modal / Daytona / E2B gibi managed sandbox altyapıları özellikle küçük ekipler için çok cazip.

Güncel pazar karşılaştırmalarında E2B, Modal, Daytona, Beam ve CodeSandbox gibi çözümler tam olarak bu problem etrafında farklı isolation/latency/persistence trade-off'larıyla konumlanıyor. ([beam.cloud][6])

---

# 12. Ben olsam senin sistemini nasıl kurardım?

Senin PTC + ReAct + multi-turn modelini düşünerek:

```text
                         User
                           │
                           ▼
                     ReAct Agent
                           │
                     Orchestrator
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
            State DB             Sandbox Manager
                                      │
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
                     Sandbox 1    Sandbox 2    Sandbox 3
                         │            │            │
                         └──────┬─────┴────────────┘
                                ▼
                         Artifact Service
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                 Metadata DB             S3/GCS
```

Ve karar:

```text
Small result
    → state/context

Temporary file
    → sandbox

Reusable output
    → artifact store

Workflow position
    → state DB
```

Bu dört ayrım bence sisteminin omurgası olmalı.

---

# 13. Hatta sandbox'ı kullanıcı/workflow bazlı düşün

Örneğin:

```text
user_42
   │
   ├── workflow_1
   │      └── sandbox_abc
   │
   ├── workflow_2
   │      └── sandbox_def
   │
   └── workflow_3
          └── sandbox_xyz
```

Böylece iki farklı agent'ın filesystem'ı birbirine karışmaz.

Artifact namespace:

```text
tenant/user/workflow/run/artifact
```

örneğin:

```text
s3://artifacts/
    user_42/
       wf_123/
          run_456/
             art_001.parquet
             art_002.pkl
```

Bu multi-tenant isolation açısından da çok daha temiz.

---

# 14. Sonuç: internette gördüğüm ortak yön

Bugünkü yaklaşımları toplarsak kabaca şu fikirde birleşiyorlar:

### Sandbox tarafı

```text
LLM-generated code
        ↓
isolated execution
        ↓
container / gVisor / microVM / VM
```

İhtiyaç arttıkça:

```text
container
   ↓
sandboxed container
   ↓
microVM
```

şeklinde daha güçlü izolasyona geçiliyor. ([gvisor.dev][3])

### Artifact tarafı

```text
Sandbox
   ↓
Artifact API
   ↓
Object Storage
   +
Metadata DB
```

ve üzerine:

```text
versioning
lineage
TTL/lifecycle
access control
checksums
```

ekleniyor. S3/GCS/Azure Blob'un güncel versioning ve lifecycle özellikleri de bu kullanım modelini doğal biçimde destekliyor. ([AWS Belgeleri][12])

### Benim özellikle önerdiğim sınır

```text
              ┌──────────────────────────┐
              │        ORCHESTRATOR      │
              │                          │
              │ ReAct / state / retries  │
              └────────────┬─────────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │   SANDBOX    │
                   │              │
                   │ execute code │
                   │ temp files   │
                   └──────┬───────┘
                          │
                    artifact.save
                          │
                          ▼
              ┌────────────────────────┐
              │    ARTIFACT SERVICE    │
              └───────────┬────────────┘
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
             Metadata DB       Object Store
                              (S3/GCS/Blob)
```

**En önemli mimari karar:** Sandbox'ı **hesaplama ve geçici çalışma alanı**, artifact store'u ise **durable çalışma ürünü deposu** olarak konumlandır. Sandbox'ın persistent olması faydalı olabilir ama **artifact durability'yi sandbox'ın yaşam döngüsüne bağlama**. OpenAI, Google, Anthropic ve Daytona'nın güncel mimarilerinde farklı şekillerde gördüğümüz ortak yön tam olarak bu ayrım. ([OpenAI GitHub][9])

Bir sonraki seviyede bunu **“ReAct → PTC → sandbox → artifact → farklı turn → yeniden sandbox”** şeklinde uçtan uca bir sequence diagram ve örnek API tasarımıyla kurmak, bu kavramları tamamen yerine oturtur.

[1]: https://www.daytona.io/docs/en/sandboxes/?utm_source=chatgpt.com "Sandboxes | Daytona"
[2]: https://modal.com/docs/sdk/js/latest/Sandbox?utm_source=chatgpt.com "Sandbox | Modal Docs"
[3]: https://gvisor.dev/docs/user_guide/production/?utm_source=chatgpt.com "Production guide - gVisor"
[4]: https://gvisor.dev/docs/user_guide/quick_start/kubernetes/?utm_source=chatgpt.com "Kubernetes Quick Start - gVisor"
[5]: https://www.daytona.io/dotfiles/vms-pause-and-fork?utm_source=chatgpt.com "VMs (Pause & Fork)"
[6]: https://www.beam.cloud/blog/2026-sandbox-guide?utm_source=chatgpt.com "Best Code Execution Environments for AI Agents in 2026 | Beam"
[7]: https://docs.cloud.google.com/run/docs/configuring/services/sandboxes?utm_source=chatgpt.com "Configure sandboxes for services  |  Cloud Run  |  Google Cloud Documentation"
[8]: https://google.github.io/adk-docs/tools/google-cloud/code-exec-agent-engine/?utm_source=chatgpt.com "Agent Runtime Code Execution tool for ADK - Agent Development Kit (ADK)"
[9]: https://openai.github.io/openai-agents-python/sandbox_agents/?utm_source=chatgpt.com "Quickstart - OpenAI Agents SDK"
[10]: https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling?cd=1966&utm_source=chatgpt.com "Programmatic tool calling - Claude Platform Docs"
[11]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/DataDurability.html?utm_source=chatgpt.com "Data protection in Amazon S3 - Amazon Simple Storage Service"
[12]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html?utm_source=chatgpt.com "Retaining multiple versions of objects with S3 Versioning - Amazon Simple Storage Service"
[13]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html?utm_source=chatgpt.com "Managing the lifecycle of objects - Amazon Simple Storage Service"
[14]: https://docs.cloud.google.com/storage/docs/object-versioning?utm_source=chatgpt.com "Object Versioning  |  Cloud Storage  |  Google Cloud Documentation"



ekip openshift kullanıyor desem ne olur

O zaman iş çok daha kolay kardeşim. Ekibin zaten OpenShift kullanıyorsa, sandbox için ayrı bir E2B/Daytona/Modal altyapısı kurmak zorunda değilsiniz. OpenShift'i doğrudan sandbox orchestration layer olarak kullanabilirsiniz.

Ama bir nüans var: “OpenShift kullanıyoruz” ≠ “otomatik olarak güvenli sandbox'ımız var.” Normal OpenShift pod'u ile OpenShift Sandboxed Containers (Kata) arasında önemli fark var. Red Hat, OpenShift Sandboxed Containers'ı untrusted workload'lar için Kata Containers tabanlı ek bir VM izolasyon katmanı olarak sunuyor.

Senin mimaride nasıl olur?

Ben şöyle kurardım:

                    Agent / ReAct
                         │
                         ▼
                    Orchestrator
                         │
                         ▼
                 Sandbox Manager
                         │
                    OpenShift API
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        Sandbox Pod A           Sandbox Pod B
        runtime: kata          runtime: kata
             │                       │
       Python / Shell          Python / Shell
             │                       │
             └───────────┬───────────┘
                         ▼
                  Artifact Service
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          Object Store          Metadata DB
          S3 / MinIO            PostgreSQL

Burada OpenShift'in rolü:

“Bu agent kodunu izole bir execution environment içinde çalıştır.”

Artifact store'un rolü:

“Bu execution'ın değerli çıktılarını uzun süre sakla.”

1. Normal OpenShift pod mu, Kata mı?

Örneğin normal OpenShift:

spec:
  containers:
    - name: agent-sandbox
      image: my-python-image

Bu standart container isolation.

Ama AI agent'ın ürettiği arbitrary Python/Shell kodu çalıştırıyorsan ben daha ciddi izolasyon düşünürdüm.

OpenShift Sandboxed Containers ile:

spec:
  runtimeClassName: kata
  containers:
    - name: agent-sandbox
      image: my-python-image

diyebilirsin.

Red Hat'e göre kata runtime'ı pod'u lightweight VM içinde çalıştırıyor; dolayısıyla workload'lar arasında daha güçlü kernel/VM isolation elde ediliyor. runtimeClassName: kata ile belirli workload'u bu runtime'a yönlendirebiliyorsun.

Yani:

OpenShift
   │
   ├── normal pod
   │
   ├── normal pod
   │
   └── kata pod
          ↓
       lightweight VM

Bu senin kullanımında çok mantıklı.

2. Neden OpenShift sizin için güzel?

Çünkü zaten ekipte şu altyapı var:

Kubernetes orchestration
Scheduling
Resource limits
Namespaces
RBAC
Secrets
NetworkPolicy
Service accounts
Storage
Observability

Dolayısıyla ayrı:

E2B
Daytona
Modal

kurup bunları şirket altyapısına bağlamak zorunda değilsiniz.

OpenShift zaten workload lifecycle'ını yönetebilir.

3. Sandbox başına ne oluşturulur?

Ben agent başına kalıcı Pod oluşturmazdım.

Daha çok:

Task / Session
      ↓
sandbox pod
      ↓
execute
      ↓
destroy

veya uzun task ise:

Agent session
      ↓
sandbox pod
      ↓
PTC #1
      ↓
PTC #2
      ↓
PTC #3
      ↓
destroy

şeklinde düşünürdüm.

Ama kullanıcı çok uzun süre sonra devam edecekse:

sandbox
   X
artifact store
   ✓

yapardım.

Yani sandbox'a güvenip bütün state'i orada bırakmazdım.

4. PTC'ler arasında iki farklı persistence modeli

OpenShift size aslında iki seçenek verir.

Aynı sandbox
PTC #1
  ↓
Pod
  ↓
PTC #2
  ↓
Pod
  ↓
PTC #3

Dosya filesystem'de kalabilir.

Farklı sandbox
PTC #1
  ↓
Pod A
  ↓
artifact
  ↓
S3 / MinIO
  ↓
Pod B
  ↓
PTC #2

İkinci model production için çok daha güçlü.

Çünkü Pod A ölürse bile:

artifact

yaşamaya devam eder.

5. OpenShift Storage burada devreye giriyor

Artifact için örneğin:

OpenShift
    │
    ├── PVC
    │
    └── S3-compatible object storage

kullanabilirsiniz.

Ama ben:

Büyük durable artifact
S3 / MinIO / object storage
Küçük workflow state
PostgreSQL

kullanırdım.

Sandbox temporary files
Pod ephemeral filesystem

Bu ayrım çok temiz.

6. Örneğin şöyle

Agent:

"CSV'yi temizle"

OpenShift:

create sandbox pod
runtimeClassName: kata

PTC:

df = pd.read_csv("/workspace/raw.csv")

df = clean(df)

df.to_parquet("/workspace/cleaned.parquet")

Sonra artifact service:

/workspace/cleaned.parquet
        ↓
artifact.save()
        ↓
MinIO/S3

Metadata DB:

{
  "artifact_id": "art_123",
  "workflow_id": "wf_456",
  "run_id": "run_789",
  "type": "parquet",
  "uri": "s3://artifacts/art_123"
}

Pod:

destroy

Sonra kullanıcı başka turn'de:

"Şimdi feature engineering yap."

Orchestrator:

artifact_id = art_123

buluyor.

OpenShift'te yeni sandbox:

create Pod B
runtimeClassName: kata

PTC #2:

artifact_download("art_123", "/workspace/input.parquet")

df = pd.read_parquet("/workspace/input.parquet")

Bu bence sizin mevcut altyapınıza çok doğal oturur.

7. Hatta OpenShift NetworkPolicy + egress kontrolü eklenebilir

Sizin daha önce konuştuğunuz egress policy konusu burada doğrudan işe yarıyor.

Örneğin sandbox:

Agent sandbox
      │
      ├── Artifact Store ✓
      ├── Internal API ✓
      ├── Database ✓
      │
      └── Internet ✗

ve sadece izin verilen domain'lere:

pypi.org
github.com
api.example.com

erişim verebilirsiniz.

Bu noktada:

OpenShift
+
NetworkPolicy
+
Cilium/eBPF
+
Kata

çok güçlü bir sandboxing stack oluşturabilir.

Tabii burada Cilium kullanıp kullanmadığınızı ayrıca doğrulamak gerekir; OpenShift'in kendi networking stack'i ile Cilium deployment'ınızın nasıl yapıldığı ayrı bir mimari karardır.

8. Ben ekibe bunu nasıl önerirdim?

Kabaca:

                  AGENT PLATFORM
                       │
                  Orchestrator
                       │
                 Sandbox Manager
                       │
                OpenShift Cluster
                       │
              ┌────────┴────────┐
              │                 │
          Normal Pods        Kata Pods
                              ↑
                       untrusted code
                              │
                         PTC runtime
                              │
                        /workspace
                              │
                     artifact upload
                              ↓
                    S3 / MinIO storage
                              │
                     PostgreSQL metadata

Ve policy:

Trusted deterministic workloads
        → normal pod

LLM-generated / untrusted code
        → Kata sandbox

Large reusable outputs
        → object storage

Workflow state
        → PostgreSQL/checkpointer

Bu bence oldukça profesyonel bir ayrım.

9. OpenShift'teki en büyük avantajınız

Aslında senin önceki sorundaki:

“Sandbox nerede çalışmalı?”

sorusuna ekip OpenShift kullanıyorsa cevap büyük ölçüde:

Mevcut OpenShift cluster'ınızın içinde, agent execution için ayrılmış workload'lar olarak.

Ama untrusted LLM-generated code için mümkünse OpenShift Sandboxed Containers/Kata runtime'ı kullanarak.

Red Hat bunu özellikle untrusted/privileged workload'ların daha güçlü isolation ile çalıştırılması için konumlandırıyor.

10. Benim size önerdiğim MVP

İlk versiyonda fazla komplike yapmazdım:

OpenShift
   │
   ├── Agent Orchestrator
   │
   └── Sandbox Deployment
          runtimeClassName: kata
                 │
                 ▼
             /workspace
                 │
                 ▼
          Artifact Service
                 │
                 ▼
              MinIO
                 │
                 ▼
            PostgreSQL

Yani:

OpenShift = compute/orchestration

Kata = isolation

MinIO/S3 = artifact bytes

PostgreSQL = artifact metadata + workflow state

PTC = sandbox içinde execution

Bu architecture ile aynı turn'deki PTC, farklı turn'deki PTC, hatta farklı sandbox'taki PTC aynı artifact'i kullanabilir.

Bence sizin durumda en önemli sonraki teknik soru artık “sandbox nerede?” değil, “OpenShift üzerinde her PTC için yeni pod mu açacağız, yoksa session başına uzun yaşayan sandbox mı kullanacağız?” Bu karar latency, maliyet, isolation ve artifact tasarımını doğrudan etkiler.




