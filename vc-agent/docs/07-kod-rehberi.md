# 07 — Kod rehberi: her dosya, her kavram

*Bu belge iki soruyu birlikte cevaplıyor: **AutoGen'in başlıkları ne anlatıyor** ve
**bizim kodumuz ne yapıyor.** İkisi ayrı ayrı anlatılınca köprü kurulmuyor; burada
her kavramın yanında onu kullanan dosya, her dosyanın yanında dayandığı kavram var.*

**Diğer belgeler ne için:**

| Belge | Ne için |
|---|---|
| [01](01-autogen-kaynak-haritasi.md) | AutoGen araştırmasının tezi ve birincil kaynakları |
| [02](02-autogen-el-kitabi.md) | API el kitabı — kurulu v0.7.5'ten doğrulanmış |
| [03](03-vc-domain-plani.md) | VC sistemi: **ne ve neden** |
| [04](04-vc-agentic-akis.md) | VC sistemi: **nasıl** (tasarım) |
| [05](05-autogen-core-user-guide.md) | Core kılavuzun **tam metni**, 42 sayfa |
| [06](06-autogen-incelikleri.md) | Pratikte **ısıran** 13 incelik |
| **07 (bu)** | **Kod + kavram köprüsü** — okuma sırası burası |

---

## 0 — Sistem tek bakışta

Halka açık kaynakları tarayıp girişim keşfeden, skorlayan ve yatırım notu üreten
çok-ajanlı bir sistem. Tasarımın tamamı tek bir cümleden türüyor:

> **Çoğa az harca, aza çok harca.**

Her katman bir öncekinden pahalı olduğu için, pahalı katmana ulaşan kayıt sayısı
azalmak zorunda. Kodda gördüğün hemen her karar bunun sonucu.

```
  ~binlerce ham sinyal   ┌──────────────────────────────────────┐
                         │ KATMAN 1 · TOPLAYICILAR              │  LLM yok
     hn · sec_edgar      │ collectors/                          │  deterministik
     github              └───────────────┬──────────────────────┘  ağsız test edilebilir
                                         │ list[Signal]
                         ┌───────────────▼──────────────────────┐
     ~onlarca şirket     │ KATMAN 2 · NORMALİZASYON             │  LLM yok
                         │ normalize.py                         │
                         └───────────────┬──────────────────────┘
                                         │ list[Company]
                         ┌───────────────▼──────────────────────┐
     ~aday               │ KATMAN 3 · TRİYAJ                    │  kural → ucuz model
                         │ agents/triage.py                     │
                         └───────────────┬──────────────────────┘
                                         │ geçenler
                         ┌───────────────▼──────────────────────┐
     ~az sayıda          │ KATMAN 4 · ZENGİNLEŞTİRME            │  orta model
                         │ graph.py  (3 paralel dal)            │  × 3 dal
                         │   ├ TechnicalAnalyst                 │
                         │   ├ MarketAnalyst      ─→ join       │
                         │   └ TeamAnalyst                      │
                         │        ↓ RiskAuditor → Scorer        │
                         └───────────────┬──────────────────────┘
                                         │ Score
                         ┌───────────────▼──────────────────────┐
     ~çok az             │ KATMAN 5 · NOT                       │  en güçlü model
                         │ agents/memo.py                       │
                         └───────────────┬──────────────────────┘
                                         │ InvestmentMemo
                         ┌───────────────▼──────────────────────┐
                         │ KATMAN 6 · TESLİM                    │
                         │ scan.py → md/json/html                │
                         │ server.py → sohbet arayüzü            │
                         └──────────────────────────────────────┘
```

### Dosya haritası

| Dosya | Satır | Rol |
|---|---:|---|
| `config.py` | 235 | Tez, model kademeleri, eşikler, oran sınırları, kara liste, `.env` yükleyici |
| `schemas.py` | 217 | **Veri sözleşmesi** — sistemin tek ortak dili |
| `policy.py` | 204 | Dışarıya açılan **tek kapı**: kara liste, robots.txt, oran sınırı, denetim |
| `engine.py` | 204 | Model fabrikası (üç kademe), `ResilientClient`, ölçüm defteri |
| `observability.py` | 192 | `autogen_core` olay yakalama + müdahale (intervention) kapısı |
| `collectors/base.py` | 176 | Toplayıcı tabanı: önbellek, retry, politika kapısı |
| `collectors/hackernews.py` | 122 | HN Algolia — fon haberi, lansman |
| `collectors/sec_edgar.py` | 83 | SEC Form D — huninin en değerli kaynağı |
| `collectors/github.py` | 83 | Repo ivmesi + kurucu profili |
| `collectors/arxiv.py` | 67 | Akademik iz (keşif değil, **doğrulama** kaynağı) |
| `normalize.py` | 112 | Varlık çözümleme + tekilleştirme |
| `agents/triage.py` | 149 | Kural ön elemesi + ucuz LLM |
| `agents/analysts.py` | 196 | Üç dal + risk denetçisi + skorlayıcı + not yazarı fabrikaları |
| `agents/tools.py` | 125 | Analistlerin tool'ları |
| `agents/memo.py` | 135 | Yatırım notu + Markdown render |
| `graph.py` | 254 | **GraphFlow**: fan-out, join, dal sayımı, süre sınırı |
| `fanin.py` | 231 | Aynı fan-out'un `autogen_core` pub/sub karşılığı |
| `compare_fanin.py` | 136 | İki motoru aynı arıza altında ölçer |
| `scan.py` | 332 | CLI giriş noktası — huninin tamamı |
| `answers.py` | 208 | LLM'siz soru yönlendirme (deterministik yol) |
| `conversation.py` | 355 | **Canlı ajan**: bellek, tool, MCP, iptal, durum |
| `dashboard.py` | 1049 | Tek dosyalık HTML üretimi + bütün stil belirteçleri |
| `server.py` | 321 | FastAPI backend: arayüz, sorular, tarama tetikleme |
| `probe_llm.py` | 260 | Uç noktanın **ne desteklediğini ölçer** |
| `web/` | 570 | Arayüz: `index.html`, `app.js`, `app.css` |
| `tests/` | 1007 | 52 test, hepsi ağsız |

### Kavram → kod dizini

Gönderdiğin başlıkların kodda nereye düştüğü:

| AutoGen başlığı | Bizde | Nerede |
|---|---|---|
| Agent and Multi-Agent Applications | ✔ | tüm `agents/` |
| Agent Runtime Environments | ✔ | `graph.py` (kendi runtime'ımızı kuruyoruz) |
| Application Stack | ✔ | katman ayrımının kendisi |
| Agent Identity and Lifecycle | ✔ | `fanin.py` (`BranchWorker.register`) |
| Topic and Subscription | ✔ | `fanin.py` (`TypeSubscription`) |
| Agent and Agent Runtime | ✔ | `graph.py`, `fanin.py` |
| Message and Communication | ✔ | `fanin.py` (pub/sub), `graph.py` (`custom_message_types`) |
| Logging | ✔ | `observability.py` |
| Open Telemetry | ✗ | yazılmadı — runtime bizim, `tracer_provider` takılabilir |
| Distributed Agent Runtime | ✗ | tek süreç yetiyor |
| Component config | ✗ | `dump_component` kullanılmadı |
| Model Clients | ✔ | `engine.py` |
| Model Context | ✔ | `conversation.py` |
| Tools | ✔ | `agents/tools.py`, `conversation.py` |
| Workbench (and MCP) | ✔ | `conversation.py` (`StaticWorkbench` + `McpWorkbench`) |
| Command Line Code Executors | ✗ | burada kod yürütülmüyor |
| Concurrent Agents | ✔ | `fanin.py` |
| Sequential Workflow | ✔ | katman mimarisi |
| Group Chat | ✔ (AgentChat sürümü) | `graph.py` |
| Handoffs | ✗ | huni tek yönlü; POC'ta ölçüldü, en pahalı desen |
| Mixture of Agents | ✗ | toplaması `asyncio.gather` — bilerek alınmadı |
| Multi-Agent Debate | ✗ | rubrik istiyoruz, oy değil |
| Reflection | kısmen | `RiskAuditor` tek turluk reflection |
| Code Execution | ✗ | uygulanamaz |

---

# BÖLÜM A — AutoGen kavramları

*Gönderdiğin başlıkların her biri. Kendi cümlelerimle, ve "bizde nerede" notuyla.*

## A1 — Core Concepts

### Agent and Multi-Agent Applications

AutoGen'in ajan tanımı dar: **mesaj alan ve mesaja karşılık üreten bir birim.**
Zeki olması gerekmiyor; bir fonksiyon da ajan olabilir. Çok-ajanlı uygulama ise
bu birimlerin bir görevi aralarında bölüşmesi.

Kritik ayrım şu: çok-ajanlı yapmanın *sebebi* zeka değil, **ayrıştırma**. Üç
analistimiz üç ayrı ajan çünkü üç ayrı kaynağa bakıyorlar ve **paralel
koşabiliyorlar** — daha akıllı oldukları için değil.

**Bizde:** `agents/analysts.py`'de altı ajan var. Her biri tek bir işi yapıyor ve
her birinin kendi model istemcisi var (kademelendirme buradan geliyor).

### Agent Runtime Environments

Runtime, ajanlar arası mesajlaşmayı yürüten, kimlik ve yaşam döngüsünü yöneten
katman. İki tip: **standalone** (tek süreç, `SingleThreadedAgentRuntime`) ve
**distributed** (host + worker, gRPC). İkisi **aynı API'yi** sunuyor — ajan kodunu
değiştirmeden geçiş yapılabiliyor.

**Bizde:** Normalde AgentChat runtime'ı kendisi kuruyor. Biz `graph.py`'de kendimiz
kuruyoruz, çünkü `InterventionHandler` takmanın tek yolu bu:

```python
runtime = SingleThreadedAgentRuntime(
    intervention_handlers=[handler], ignore_unhandled_exceptions=False
)
flow = GraphFlow(..., runtime=runtime)
```

**Bedeli var ve ölçtük:** runtime'ı sen verirsen çöken bir ajan `run_stream`'i
**fırlatmıyor, asıyor** (gömülü runtime'da fırlatıyordu). Bu yüzden `graph.py`'de
duvar saati sınırı bir tedbir değil, doğruluk şartı. → [06 §7](06-autogen-incelikleri.md)

### Application Stack

AutoGen'in katman modeli: `core` (aktör runtime) → `agentchat` (hazır ajan/takım
soyutlamaları) → `ext` (model istemcileri, tool'lar, MCP). Aşağıya inebilirsin ama
inmek zorunda değilsin.

**Bizde:** Her üç katmanı da kullanıyoruz — `agentchat` günlük iş için, `core`
gözlemlenebilirlik ve alternatif fan-in için, `ext` model istemcisi ve MCP için.

### Agent Identity and Lifecycle

Bir ajanın kimliği `(type, key)` çiftinden oluşuyor. Runtime ajanı **ilk mesaj
geldiğinde** yaratıyor (lazy). `register` ile tipi ve fabrikasını tanıtıyorsun,
örnek kendiliğinden oluşuyor.

**Bizde:** `fanin.py`'de her dal kendi ajan tipi:

```python
await BranchWorker.register(runtime, branch, _factory(branch, agent))
```

Bir de pratik sonucu: `ToolCallEvent`'in taşıdığı `agent_id`, ajan runtime
yönetimindeki bir handler içindeyken doluyor — çıplak `agent.run()`'da `None`.
Canlı doğrulandı. → [06 §6](06-autogen-incelikleri.md)

### Topic and Subscription

Pub/sub'ın adresleme modeli. Bir topic'in **tipi** ve **kaynağı** var. Abonelik
(`TypeSubscription`) bir topic tipini bir ajan tipine bağlıyor, ve **topic kaynağı
ajan anahtarına dönüşüyor.**

Bu son cümle önemli: bir topic'e `("gorev", "sirket-42")` diye yayın yaparsan,
runtime `("analist", "sirket-42")` ajanını yaratıyor. Yani **şirket başına izole
ajan örneği bedava** — çok kiracılı yapının hazır mekanizması.

**Bizde:** `fanin.py` tek kaynak (`"default"`) kullanıyor, çünkü tek kullanıcılı bir
araç. Hacim sorusu cevaplanınca (günde 200 mü 5.000 mi) bu mekanizma şirket başına
izolasyonun hazır cevabı olacak.

---

## A2 — Framework Guide

### Agent and Agent Runtime

`RoutedAgent` + `@message_handler`: mesaj tipine göre yönlendirme. Handler'ın
**imzasından** tip çıkarılıyor.

```python
class BranchWorker(RoutedAgent):
    @message_handler
    async def handle(self, message: BranchTask, ctx: MessageContext) -> None: ...
```

**Tuzak:** Tip çıkarımı `get_type_hints()` ile yapılıyor ve `from __future__ import
annotations` varken anotasyonlar **modül genelinde** çözülüyor. `MessageContext`'i
fonksiyon içinde import edersen kayıt sırasında çıplak bir `NameError` alıyorsun.
Bu yüzden `fanin.py`'de bütün core importları modül düzeyinde. → [06 §11](06-autogen-incelikleri.md)

### Message and Communication

Üç iletişim biçimi: **yayın** (pub/sub, cevap beklemeyen), **doğrudan mesaj**
(`send_message`, RPC gibi), ve AgentChat'in **ortak thread**'i.

**Bizde:** `fanin.py` yayın kullanıyor (dallar sonucu topic'e basıyor),
`graph.py` AgentChat'in thread modelini kullanıyor.

**Tuzak:** Takım, tanımadığı bir mesaj tipini yönlendirmiyor. Yapısal çıktı üreten
bir ajan takımdaysa tipi beyan etmen gerekiyor:

```python
GraphFlow(..., custom_message_types=[StructuredMessage[Score]])
```

Tek başına `agent.run()`'da gerekmiyor — sorun yalnız yönlendirmede çıkıyor.

### Logging

`autogen_core` yapılandırılmış olayları `autogen_core.events` logger'ına basıyor.
`LLMCallEvent`, `LLMStreamEndEvent`, `ToolCallEvent`, `MessageEvent`,
`MessageDroppedEvent` ve dahası.

**Bizde:** `observability.py` bu akışı dinliyor ve iki iş yapıyor — token sayımı ve
tool çağrılarını denetim kaydına aynalama.

**İki tuzak birden:**
- `create()` → `LLMCallEvent`, `create_stream()` → **sadece** `LLMStreamEndEvent`.
  Akış kullanınca yalnız ilkini dinlersen maliyet 0 raporlanıyor.
- `LLMCallEvent` alanlarını öznitelik yapıyor, `ToolCallEvent` **`.kwargs`
  sözlüğünde** tutuyor. `event.tool_name` yazarsan `AttributeError` alıyorsun ve o
  hata `logging.Handler` içinde yutuluyor — olay hiç kaydedilmiyor, hata da
  görünmüyor.

### Open Telemetry

Runtime `tracer_provider` alıyor; ajan mesajları ve model çağrıları span'a dönüşüyor.

**Bizde: yok.** Ama yolu açık — runtime'ı zaten kendimiz kuruyoruz, tek satır.
Kalan işlerin en kolayı.

### Distributed Agent Runtime

gRPC host + worker'lar. Ajanlar makinelere dağılıyor, API değişmiyor.

**Bizde: yok.** Tek süreç yetiyor; hacim sorusu cevaplanmadan gereksiz karmaşa.
Ama "aynı API" vaadi şu anlama geliyor: gerekirse geçiş, ajan kodunu değiştirmeden.

### Component config

`dump_component()` / `load_component()` — ajanı JSON'a serileştirip dosyadan
yükleme. AutoGen'de "skill" diye bir soyutlama **yok**; en yakın karşılığı bu.

**Bizde: yok.** Ajanlar kodda tanımlı. Tez konfigürasyondan geliyor, ajan
topolojisi değil.

---

## A3 — Components Guide

### Model Clients

`ChatCompletionClient` arayüzü. `OpenAIChatCompletionClient` OpenAI ve uyumlu
endpoint'ler için; `ReplayChatCompletionClient` önceden yazılmış yanıtlar için.

**Bizde:** `engine.py`. İki mod:

```python
if config.live_llm_available():
    return OpenAIChatCompletionClient(model=..., base_url=..., api_key=...)
return ReplayChatCompletionClient(list(script or ["…"]), model_info=DRY_MODEL_INFO)
```

**En pahalı tuzak burada:** model adı bilinen bir OpenAI modeli değilse
`model_info` **zorunlu**. "OpenAI-uyumlu endpoint" kullanan herkes buraya
düşüyor ve ilk canlı istekte patlıyor. Kod artık önce `model_info`'suz deniyor,
`ValueError` gelirse `config.LIVE_MODEL_INFO` ile tekrar kuruyor.

Ve `model_info` bir **beyan**, ölçüm değil — kütüphane sözüne inanıyor. Bu yüzden
`probe_llm.py` var.

### Model Context

Ajanın belleği. `AssistantAgent`, `run()` çağrıları arasında **durumsuz**;
sohbet gibi davranması için bir bağlam nesnesi vermen gerekiyor.

| Sınıf | Ne yapıyor |
|---|---|
| `BufferedChatCompletionContext(buffer_size=N)` | Son N **mesajı** tutar |
| `HeadAndTailChatCompletionContext` | Baş + son, ortayı atar |
| `TokenLimitedChatCompletionContext` | **Token** sınırı — maliyeti asıl bu sınırlar |

**Bizde:** `conversation.py`, `BufferedChatCompletionContext(buffer_size=24)`.
Buffer'ın **mesaj** saydığını, token saymadığını not düştüm — uzun bir sohbette
maliyeti sınırlayan şey bu değil.

### Tools

`FunctionTool` bir Python fonksiyonunu tool'a çeviriyor. Şemayı **imzadan ve
docstring'den** üretiyor — yani docstring dokümantasyon değil, **arayüz**.

**Bizde:** `agents/tools.py` ve `conversation.py`. Her tool `Args:` bloklu docstring
taşıyor çünkü model parametreyi oradan öğreniyor.

**Tuzak:** `max_tool_iterations` varsayılanı **1**. Ajan bir tool çağırıyor, sonucu
görüyor, **duruyor**. Zincirleme davranış hata vermeden imkânsız.

### Workbench (and MCP)

Workbench bir tool **kaynağı** — tek tek tool yerine "bana hangi tool'ların
olduğunu söyle" diyebildiğin bir nesne. Uzak MCP sunucusu tam olarak bu.

**Bizde:** `conversation.py`, DeepWiki'yi `McpWorkbench` ile bağlıyor.

**Tuzak:** `tools=` ile `workbench=` **aynı ajana verilemiyor**
(`ValueError: Tools cannot be used with a workbench`). Çözüm — yerel fonksiyonları
`StaticWorkbench`'e sarıp **liste** vermek:

```python
workbenches = [StaticWorkbench(local_tools)]
if mcp is not None:
    workbenches.append(mcp)
AssistantAgent(..., workbench=workbenches)
```

### Command Line Code Executors

`DockerCommandLineCodeExecutor` ve yerel muadilleri — modelin ürettiği kodu
sandbox'ta çalıştırma. AutoGen'in kurucu özelliklerinden biri.

**Bizde: yok**, çünkü burada kod yürütülmüyor. Bir girişimi değerlendirmek için
kod çalıştırmak gerekmiyor.

---

## A4 — Multi-Agent Design Patterns

### Intro

Desenler, ajanların *nasıl dizildiğini* anlatıyor. Ana ayrım: akış **önceden mi
çizili** (graf, sıralı) yoksa **konuşmadan mı doğuyor** (group chat, handoff).

### Concurrent Agents

Üç varyant: tek mesaj–çok işleyici, çok mesaj–çok işleyici (fan-out/fan-in),
doğrudan mesajlaşma. **Sonuç toplama** için `ClosureAgent` + `asyncio.Queue`.

**Bizde:** `fanin.py` bunun birebir uygulaması ve **en değerli devşirme**. Sebebi
ölçüldü: sonuç üretildiği anda yayınlanıyor ve kuyruk onu tutuyor, yani çöken bir
dal tamamlanmış kardeşleri götüremiyor.

### Sequential Workflow

Her ajan bir öncekinin çıktısını alıyor, zincir hâlinde.

**Bizde:** Katman mimarisinin kendisi bu — ama ajanlarla değil, kodla kurulu.
Toplayıcı→normalize→triyaj zincirinde LLM yok, dolayısıyla ajan da yok.

### Group Chat

Ortak thread + bir konuşmacı-seçici. Core sürümü için dokümanın **kendi cümlesi**:
*"not meant to be used in real applications… a starting point."* Üretim sürümü
AgentChat'in `SelectorGroupChat`'i.

**Bizde:** `graph.py` AgentChat sürümünü kullanıyor. Core sürümünden alınacak bir
şey yok — doküman zaten öyle diyor.

### Handoffs

Ajan konuşmanın **sahipliğini** başka bir ajana devrediyor (`Handoff`, `Swarm`).

**Bizde: yok.** POC'ta ölçüldü: **en pahalı desen** (334 token; Selector 204).
Her devir bir tool çağrısı + iş üretmeyen bir LLM turu harcıyor. Huni tek yönlü
olduğu için devre gerek de yok.

**Tuzak (POC'tan):** `Handoff` tool adı küçük harfe düşüyor
(`transfer_to_veriuzmani`). Elle yazarsan eşleşmiyor.

### Mixture of Agents

Katmanlı işçi ajanlar; her katmanın çıktısı birleştirilip sonrakine veriliyor,
sonda bir orkestratör topluyor.

**Bizde: yok** — ve **almama sebebim bir bulgu**: orkestratör `asyncio.gather(...)`
ile topluyor, yani POC'ta sessiz kardeş kaybının kaynağı olan yapı. Resmî desenler
bu konuda birbiriyle çelişiyor: *Concurrent Agents* kuyrukla topluyor, *Mixture of
Agents* `gather` ile.

### Multi-Agent Debate

Seyrek topoloji, birkaç tur, sonda çoğunluk oyu.

**Bizde: yok.** Skorlamada **sabit rubrik** istiyoruz — "adil kıyas" ilkesi.
Oylama, aynı şirkete iki koşuda iki puan verme riskini geri getirir. Üstelik
N ajan × R tur maliyeti `agentic_tuffs.md` satır 317'nin ("burns tokens") örneği.

### Reflection

Üretici + eleştirmen döngüsü; eleştirmen onaylayana kadar dönüyor.

**Bizde: kısmen.** `RiskAuditor` tek turluk bir reflection — üç analizi çapraz
denetliyor. Döngüye çevirmedim: dokümanın kendisi **durma ölçütü önermiyor**,
yani faturayı sınırlayan tek şey kalmıyor.

### Code Execution

Group chat + kod yürütücü: model kod yazıyor, bir ajan çalıştırıyor, sonuç geri
dönüyor.

**Bizde: yok**, uygulanamaz.

---

# BÖLÜM B — Kod, dosya dosya

## B0 — Temel katman

### `config.py` — tek yapılandırma noktası

**Rol:** Tez, model kademeleri, eşikler, oran sınırları, kara liste, `.env` yükleme.

**Neden tez burada:** Rubriğin kalibrasyonu teze bağlı, yani tezi değiştirmek bütün
skorları değiştiriyor. Bu yüzden ortam değişkeni değil, **git'te izlenen bir kayıt**:

```python
THESIS = Thesis(
    sectors=["AI infrastructure", "developer tools", "data infrastructure"],
    stages=["pre-seed", "seed"],
    requirements=["technical founder", "public technical trace (repo/paper/demo)"],
    red_lines=["solo non-technical founder", "closed-source consulting business"],
    is_placeholder=True,   # ← senin tezin gelene kadar True
)
```

`is_placeholder=True` olduğu sürece her koşu uyarı basıyor ve `thesis_fit`
"kalibre edilmemiş" sayılıyor. **Sessizce yanlış skor üretmektense gürültülü
şekilde eksik olmak.**

**`.env` yükleyici:** Üç konuma bakıyor (`vc-agent/.env`, `pipeline/.env`, repo
kökü), ve **takma adları** kabul ediyor:

```python
_ALIASES = {
    "LLM_BASE_URL": ["VC_LLM_BASE_URL"],
    "LLM_API_KEY": ["VC_LLM_API_KEY"],
    "LLM_MODEL_NAME": ["VC_MODEL_CHEAP", "VC_MODEL_MID", "VC_MODEL_STRONG"],
}
```

Çalışan bir `.env`'i tek projenin ön ekine uydurmak için yeniden yazmak zorunda
kalmayasın diye. Ortamda zaten export edilmiş değerler kazanıyor
(`os.environ.setdefault`), yani komut satırından tek seferlik override mümkün.

`VC_SKIP_ENV_FILE` kaçış kapısı test paketi için — aşağıda.

**Model yetenek beyanı:**

```python
LIVE_MODEL_INFO = {
    "vision": _flag("VC_MODEL_VISION", False),
    "function_calling": _flag("VC_MODEL_FUNCTION_CALLING", True),
    "json_output": _flag("VC_MODEL_JSON_OUTPUT", True),
    "structured_output": _flag("VC_MODEL_STRUCTURED_OUTPUT", True),
    "family": os.getenv("VC_MODEL_FAMILY", "unknown"),
}
```

Her alan ayrı ayrı geçersiz kılınabiliyor, çünkü yanlış beyan **sessizce ve geç**
patlıyor.

**Zaman aşımı:** `LLM_TIMEOUT = 300`, `LLM_MAX_RETRIES = 2`. Varsayılan istemci
zaman aşımı sohbetin uzun istemine (yedi tool şeması) yetmiyordu.

---

### `schemas.py` — veri sözleşmesi

**Rol:** Ajanlar arasında serbest metin dolaşmıyor; her şey Pydantic.

İki alan **bilinçli olarak zorunlu** ve ikisi de birer ilkenin kodla zorlanması:

```python
class Source(BaseModel):
    name: str
    url: str          # ← ZORUNLU: kaynaksız iddia şema seviyesinde imkânsız
    ...
    @field_validator("url")
    def _url_is_resolvable(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Source.url must be a resolvable link: {v!r}")
        return v

class Score(BaseModel):
    thesis_fit: int = Field(ge=0, le=5)
    ...
    missing_data: list[str]   # ← ZORUNLU: "0 sonuç açıklama borçludur"
    decision: Decision
```

`missing_data`'nın varsayılanı **yok** — vermeyi unutamıyorsun. Düşük puan ile
bilgi yokluğu farklı şeyler ve sistem hangisi olduğunu söylemek zorunda.

Türetilmiş alanlar hesaplanıyor, saklanmıyor:

```python
@property
def reliability(self) -> str:
    n = len(self.missing_data)
    return "high" if n == 0 else ("medium" if n <= 2 else "low")
```

**Varlık anahtarı** — kanıt gücüne göre sıralı:

```python
@property
def key(self) -> str:
    if self.domain:  return f"domain:{self.domain.lower()}"
    if self.github:  return f"gh:{self.github.lower()}"
    return f"name:{normalize_name(self.name)}"
```

`BranchResult.succeeded=False` bir hata değil, **kaydedilmiş bilgi yokluğu** —
doğrudan `Score.missing_data`'ya akıyor.

---

### `policy.py` — dışarıya tek kapı

**Rol:** Toplayıcılar doğrudan HTTP çağırmıyor; hepsi buradan geçiyor.

Üç garanti:

**1. Kara liste — koşulsuz.** robots.txt'e bile bakılmıyor; bu teknik değil
**hukuki** bir karar:

```python
def is_allowed(self, url, *, source="?", exemption=None) -> bool:
    if self.is_blocked(url):          # linkedin, crunchbase, ...
        self._record(...reason="blocklist")
        return False
    if exemption:                     # ← muafiyet kara listeyi YENEMEZ
        ...
```

Bir testle korunuyor: `test_blocklist_beats_api_exemption`.

**2. robots.txt** — okunuyor, önbelleğe alınıyor. Ulaşılamıyorsa "izin var"
sayılıyor (RFC 9309) **ama denetime yazılıyor.**

**API muafiyeti:** arXiv'in `robots.txt`'i `Disallow: /` diyor, ama API Terms of
Use programatik erişime izin veriyor. Bunu sessizce geçmedim — gerekçe **her
istekte** denetim kaydına yazılıyor:

```python
robots_exemption = "arXiv API Terms of Use: https://info.arxiv.org/help/api/tou.html"
```

**3. Oran sınırı + denetim.** Kaynak başına asgari bekleme, ve her çağrının JSONL
kaydı. `record_agent_action()` ayrı bir kayıt tipi: toplayıcının *ne çektiği*
değil, ajanın *ne çağırmayı seçtiği*.

`robots_fetcher` enjekte edilebilir — testler ağa çıkmadan koşsun diye.

---

### `engine.py` — model fabrikası ve ölçüm

**Rol:** Üç kademe için istemci üretmek, ve maliyeti saymak.

**Ölçüm iskeleti yeniden yazılmadı** — `poc/motor.py`'deki `Olcum` doğrudan içe
aktarılıyor. POC ile pipeline aynı metrikleri aynı tanımla üretiyor.

**`ResilientClient`** — bir ölçümün sonucu, tercih değil:

```python
async def create(self, *args, **kwargs):
    try:
        return await self._inner.create(*args, **kwargs)
    except Exception as e:
        return CreateResult(
            finish_reason="stop",
            content=f"{BRANCH_FAILURE_MARKER} {type(e).__name__}: {e}",
            ...
        )
```

Hata **mesaja çevriliyor**. Sebebi: GraphFlow fan-out'unda bir dal exception
fırlatınca takım iptal ediliyor ve tamamlanmış kardeş dalların işi de gidiyor.
Hata mesaja dönüşürse join yine üç girdi alıyor.

**Ama her yerde doğru değil** — bu oturumun en ince dersi. Sohbette kardeş yok;
orada hatayı yutmak, uç nokta arızasını **analistin görüşü gibi** göstermek olur.
O yüzden ikinci bir fabrika var:

```python
def raw_client(self, tier, script=None):
    """A client whose failures are allowed to raise."""
```

`conversation.py` bunu kullanıyor. **Aynı mekanizmanın doğruluğu bağlama bağlı.**

**`Ledger.measurement()` delta döndürüyor**, kümülatif değil:

```python
m.prompt_token = totals["prompt"] - self._counted["prompt"]
self._counted = totals
```

Defter bir tarama boyunca yaşıyor ama ölçüm şirket başına alınıyor; kümülatif
okursa aynı token'lar her şirkette bir kez, sonra satırlar toplanınca bir kez daha
sayılır.

---

### `observability.py` — olaylar ve müdahale kapısı

**Rol:** `autogen_core`'un olay akışını dinlemek ve mesaj hattına oturmak.

**`EventCapture`** bir `logging.Handler`:

```python
if isinstance(event, (LLMCallEvent, LLMStreamEndEvent)):
    self.totals.prompt_tokens += event.prompt_tokens
    ...
elif isinstance(event, ToolCallEvent):
    self._record_tool_call(event)
```

İki tuzak burada yaşıyor ve ikisi de yorumla işaretli:
- İki olay, tek anlam — `create()` vs `create_stream()`
- `ToolCallEvent` alanları `.kwargs`'ta, içinde `agent_id` de var

Tool çağrısı denetim kaydına aynalanıyor. `docs/04` §6'nın *"bu puan nereden
geldi"* sorusunu gerçekten cevaplanabilir kılan şey bu — toplayıcının HTTP
kayıtlarının üstüne **hangi ajan neyi çağırmayı seçti** katmanı biniyor.

**`AuditingInterventionHandler`** mesaj hattına oturuyor, `DropMessage`
döndürebiliyor. Yani onay kapısı **runtime seviyesinde** — ajanın uymayı
seçmesine bağlı değil. Şu an gözlemci modda, çünkü buradaki tool'ların hepsi
salt-okunur; ilk mutasyon yapan tool geldiğinde kapı hazır olsun diye bağlı ve
testli.

---

## B1 — Toplayıcılar (`collectors/`)

**Bu katmanda LLM yok.** Üç sebep: fixture'la ağsız test edilebilsin, ölçüm modelin
keyfinden etkilenmesin, ve binlerce sinyale model çağırmanın israfı olmasın.

### `base.py`

Ortak taban: politika kapısı → önbellek → retry → kayıt.

```python
def fetch(self, url, params=None) -> str:
    full_url = _with_params(url, params)
    if not self.policy.is_allowed(full_url, source=self.name,
                                  exemption=self.robots_exemption):
        raise PermissionError(...)
    if self.use_cache:
        hit = self._read_cache(full_url)
        if hit is not None: ...
    status, text = self._fetch_with_retry(full_url, params)
```

Retry politikası ayrımlı: **429/5xx geçici** (tekrar dener), **4xx kalıcı**
(denemez). `fetcher` enjekte edilebilir — testler fixture veriyor.

**Arıza sessiz kalmıyor:**

```python
def run(self, *, query, days) -> CollectionResult:
    try:
        result = self.collect(query=query, days=days)
    except Exception as e:
        result = CollectionResult(source=self.name, error=f"{type(e).__name__}: {e}")
```

Bir toplayıcı düşerse diğerleri koşuyor, ama düşme `CollectionResult.error` olarak
yukarı taşınıyor ve nihai notta "şu kaynağa bakılamadı" diye görünüyor.

### `hackernews.py`

`search_by_date` kullanıyor — alaka değil **tazelik** sıralaması (domain §3.2).

İki varlık çözümleme kuralı burada, ikisi de canlı koşuda bulundu:

```python
candidate_domain=(
    _company_domain(external_url) if kind == "product_launch" else None
),
```

**Haber makalesinin alan adı yayıncıya ait**, konuya değil — `bbc.com` böyle aday
olmuştu. Yalnız Show HN'de gönderen kendi ürününü gösteriyor.

```python
_GENERIC_HOSTS = {"github.com", "gitlab.com", "medium.com", "vercel.app", ...}
```

**Platform hostu şirket değil** — GitHub'a link veren üç ayrı Show HN tek şirkette
birleşmişti.

Başlıktan isim çıkarma **muhafazakâr**: emin olamayınca `None` dönüyor. Yanlış
isim, isimsiz kayıttan pahalı.

### `sec_edgar.py`

Huninin en değerli kaynağı: ABD'de özel sermaye toplayan şirket **15 gün içinde**
Form D dolduruyor, yani tur basın açıklamasından *önce* görünüyor.

```python
def headers(self):
    # SEC's own access policy: a User-Agent without contact details gets 403.
    return {"User-Agent": config.SEC_USER_AGENT, ...}
```

Kaynak güveni `"official"` — şirketin düzenleyiciye kendi beyanı, ikincil habere
göre farklı bir sınıf.

**Sınır:** yalnız ABD. Coğrafya dışıysa sessizce boş dönüyor; bu bir hata değil
kapsam sınırı ve nota `missing_data` olarak yazılması gerekiyor.

### `github.py`

Repo ivmesi. Anahtarsız 60 istek/saat, `GITHUB_TOKEN` ile 5.000.
Organizasyon hesaplarına öncelik veriyor (kişisel hesap zayıf sinyal).

`public_profile()` ayrı bir fonksiyon — ekip analistinin tool'u. Yalnız halka açık
alanlar okunuyor (KVKK notu).

### `arxiv.py`

**Keşif listesinde değil, ve bu bir bulgu:**

```python
# ArXiv is deliberately NOT here. A paper carries no company name or domain, so
# every arXiv signal arrives unattachable... Measured on the first live run: 30 of
# 30 arXiv signals were unattached.
DISCOVERY = [HackerNews, SecFormD, GitHub]
```

Bir kat aşağıda, ekip analistinin `publication_trace` tool'u olarak işe yarıyor:
*adı bilinen* bir kurucu için yayın izi, "teknik kurucu" şartının kanıtı.

---

## B2 — `normalize.py`

Sinyaller şirketlere bağlanıyor. Sıra **kanıt gücüne göre**:

```
alan adı  >  GitHub org  >  normalize edilmiş isim (bulanık, eşik 0.92)
```

**Belirsizlikte birleştirmiyor.** Eşik 0.92 bilerek yüksek:

> Wrongly merging two records costs more than leaving them apart: once merged,
> the evidence package is contaminated and it is no longer recoverable which
> signal belonged to whom.

Sahibi çözülemeyen sinyal atılmıyor ama şirkete de bağlanmıyor — bağlamsız kanıt
yanıltıcı. Kaç tanesinin bağlanamadığı `scan.py` çıktısında **açıkça** raporlanıyor.

ChromaDB'li semantik dedup yazılmadı (paket kurulu değil); yerine deterministik
anahtar + `difflib`. Kurulunca `name_similarity()` yerine vektör araması geçebilir.

---

## B3 — Ajanlar (`agents/`)

### `triage.py` — huninin en ucuz katmanı

İki kademeli: **önce kural, sonra ucuz LLM.**

```python
def prefilter(company, thesis=None) -> TriageResult | None:
    # kırmızı çizgi → ele
    # sadece üçüncü-taraf haber, birinci-taraf kanıt yok → ele
    # tez sektörü + ≥2 sinyal → geçir
    return None   # ← kararsız: LLM'e devret
```

`None` dönmesi "bilmiyorum" demek ve **eleme değil**. İlke 1 burada kodlanıyor:

> Rule: absence of information is NOT grounds for rejection.

Sistem mesajı da aynı şeyi modele söylüyor, ve yapısal yanıt gelmezse bile
aday geçiyor:

```python
# No structured answer: still no rejection — uncertainty is not grounds.
return TriageResult(True, "triage returned no structured answer; candidate passed on", True)
```

Kanıta dayalı bir eleme kuralı var, o da ilkeyi çiğnemiyor:

```python
kinds = {s.kind for s in company.signals}
if kinds and kinds <= {"news"} and not company.github:
    return TriageResult(False, "only third-party mentions; no first-party evidence...", False)
```

Bu "bilgi yok" diye elemek değil; **elimizdeki kanıtın ne olduğuna** bakıp elemek.

### `analysts.py` — altı ajan fabrikası

Üç dal + risk denetçisi + skorlayıcı + not yazarı. Kademelendirme burada:

```
technical / market / team  → mid
risk auditor, scorer       → mid
memo writer (~5/gün)       → strong
```

Her dalın **kendi istemcisi** var — `poc/desen_4_graphflow.py`'deki karar: paralel
dallar tek replay istemcisini paylaşsaydı yanıtları hangi sırayla tüketecekleri
belirsiz olurdu, ve canlı modda dal başına farklı model kullanmanın doğal yolu bu.

Her ajanın anlamlı bir `description`'ı var — süs değil: boş açıklama
`SelectorGroupChat`'i kör seçim yaptırıyor.

Sistem istemlerinde ortak bir kural bloğu var:

```
Every claim must carry the source URL it came from. If you do not have a source
for something, do not write it: write instead which specific fact is missing.
"A zero result always owes an explanation" — state where you looked.
```

Kuru mod senaryolarında **tool çağrısı yok**, bilerek: replay edilen bir tool
çağrısı gerçek tool'u çalıştırıp ağa çıkardı ve kuru modun "deterministik ve
çevrimdışı" özelliğini bozardı.

### `tools.py` — analistlerin araçları

`inspect_repository`, `search_market_chatter`, `founder_profile`,
`publication_trace`. Hepsi politika kapısından geçiyor ve **kaynak linkiyle
birlikte** dönüyor:

> Tools return their **source URL alongside the data**. The reason is discipline
> rather than schema: if the text in front of an agent carries no link, the agent
> cannot put a link in the memo.

### `memo.py` — en pahalı adım

Notu, dalların **gerçekten raporladığından** kuruyor. Çöken dallar açıkça
soru olarak geçiyor:

```python
questions=(
    [f"The {b} branch returned nothing — what is the actual position there?" for b in missing]
    or [...]
),
```

Yani kanıttaki boşluk okuyucuya **sessizlik olarak değil, kurucuya sorulacak soru
olarak** görünüyor.

Kaynakça modele bırakılmıyor:

```python
if not memo.references:
    memo.references = candidate.company.sources[:10]
```

`render_markdown()` insanın okuduğu çıktıyı üretiyor — eksen tablosu, eksik veri
listesi, tekilleştirilmiş kaynaklar.

---

## B4 — Orkestrasyon

### `graph.py` — GraphFlow, ve bariyere güvenmemek

```
TechnicalAnalyst ─┐
MarketAnalyst    ─┼─► RiskAuditor ─► Scorer      (join: activation_condition="all")
TeamAnalyst      ─┘
```

Üç analistin gelen kenarı yok, yani **kaynak düğüm** — eşzamanlı başlıyorlar.

```python
for branch in (technical, market, team):
    builder.add_edge(branch, risk, activation_group="enrichment", activation_condition="all")
builder.add_edge(risk, scorer)
```

**Modülün varlık sebebi olan kural:**

```python
EXPECTED_BRANCHES: dict[str, str] = {
    "TechnicalAnalyst": "technical",
    "MarketAnalyst": "market",
    "TeamAnalyst": "team",
}
```

Framework'e "bütün dallar geldi mi" diye **sormuyoruz**; beklediğimiz dalları
sayıyoruz. Raporlamayan her dal `Score.missing_data`'ya giriyor — ve modelin bu
konuda son sözü yok:

```python
if score is not None:
    for branch in branches:
        if not branch.succeeded:
            note = f"{branch.branch} branch produced no result"
            if note not in score.missing_data:
                score.missing_data.append(note)
```

Üç savunma katmanı üst üste:
1. `ResilientClient` — hatayı mesaja çevirir
2. `run_stream` + `asyncio.wait_for` — süre sınırı (dış runtime asıyor)
3. Dal sayımı — bariyer ne derse desin

`stop_when_idle()` bile sınırlı bekleniyor: *"it is the barrier the POC found
unreliable, and it will not be trusted to return here either."*

### `fanin.py` — core seviyesinde toplama

Aynı fan-out, ama toplama framework'ten alınmış:

```python
@message_handler
async def handle(self, message: BranchTask, ctx: MessageContext) -> None:
    try:
        result = await self._agent.run(task=message.company_brief)
        ...
    except Exception as e:
        # The whole point: a failure is published, not raised.
        outcome = BranchOutcome(branch=self._branch, succeeded=False, error=...)
    await self.publish_message(outcome, topic_id=TopicId(RESULT_TOPIC, source="default"))
```

`ClosureAgent` sonuçları kuyruğa boşaltıyor, ve toplayıcı **beklediği sayıyı**
bekliyor, runtime'ın "boşaldım" demesini değil:

```python
async def drain() -> None:
    while len(collected) < len(branch_agents):
        outcome = await results.get()
        collected[outcome.branch] = outcome
await asyncio.wait_for(drain(), timeout=deadline)
```

> Güvenilmeyecek bariyer yok, çünkü bariyer yok.

### `compare_fanin.py` — iki motoru aynı arıza altında ölçmek

`poc/kiyas.py` geleneği: numaralı senaryolar, tek tablo, JSON çıktı.
Üç mod — temiz, sarmalayıcı arkasında hata, ham hata:

| motor | temiz | sarmalayıcı arkasında | ham hata |
|---|---:|---:|---:|
| `graph.py` | 3 | 2 | **0–1, süre sınırı dolar** |
| `fanin.py` | 3 | 2 | **2, ~3 ms** |

**Kaç dal kaybedildiği deterministik değil** — tekrarlı koşularda 0 ve 1.

---

## B5 — Giriş noktaları

### `scan.py` — huninin tamamı, CLI'dan

```bash
.venv/bin/python pipeline/scan.py --query "ai infrastructure" --days 7 --limit 5
```

`_preflight()` en başta koşulları **söylüyor**: kuru mod mu, tez placeholder mı.
Her aşama neyi düşürdüğünü basıyor:

```
  ✓ hn          17 signals (1 requests, 0 cached)
  ✗ arxiv      FAILED — ...
  58 signals -> 45 companies
  13 signals had no resolvable owner and were left unattached
```

Aşama başına maliyet raporu, huninin gerçekten "çoğa az" harcadığını
denetleyebilmek için:

```
  triage                    45 calls    7409 tokens
  enrichment:Argonix         5 calls     748 tokens
```

Eşiği geçen olmazsa bu bile açıklamalı:

> none reached the review threshold (17/25).
> **This is a threshold outcome, not an absence of candidates.**

Çıktı üç formatta: Markdown, JSON, ve HTML (JSON'dan üretiliyor, böylece sayfa
anlattığı koşudan sapamıyor).

### `answers.py` — LLM'siz yol

Soru önce **aday adlarına**, sonra anahtar kelime kümelerine karşı eşleşiyor:

```python
# A company name beats every general intent.
for index, candidate in enumerate(data.get("candidates", [])):
    name = ...
    if name and (name in q or q and name.startswith(q)):
        return f"company:{index}"
```

Eşleşme yoksa `None` — ve bu **dürüst dal**:

```python
if key is None:
    return {"path": "rules", "title": "Not something I hold",
            "text": "This scan does not hold an answer to that. Here is what it does hold.", ...}
```

`facts()` fonksiyonu modele verilecek **tek gerçek kaynağı** üretiyor: huni,
maliyet, adaylar, eksik veriler, kaynak URL'leri. Sistem istemi bunun üstüne tek
bir yasak koyuyor:

> Never state a fact that is not in the JSON.

Model yolu ile kural yolu birlikte çalışıyor: model düzyazı yazıyor, deterministik
blok **kanıt olarak altına** ekleniyor.

### `conversation.py` — canlı ajan

Bitmiş taramanın okuyucusu değil; konuşurken iş yapan bir ajan.

**Beş yerel tool:** `scan_facts`, `company_detail`, `search_github`,
`search_hacker_news`, **`start_scan`** — sonuncusu ajanı rapor edenden iş yapana
çeviriyor.

**Tool + MCP birlikte:**

```python
workbenches = [StaticWorkbench(local)]
self._mcp = await self._mcp_workbench()
if self._mcp is not None:
    workbenches.append(self._mcp)
```

MCP bağlanamazsa sohbet çalışmaya devam ediyor, durum `mcp_status`'ta yazılı.

**Bellek:**

```python
model_context=BufferedChatCompletionContext(buffer_size=self._buffer_size),
model_client_stream=True,
max_tool_iterations=6,  # the default is 1: no chained tool calls
```

**Ham istemci:**

```python
# ... as an event, not as an answer. See `Ledger.raw_client`.
model_client=self._ledger.raw_client("mid"),
```

**Akış olayları** arayüze çeviriliyor: `chunk`, `tool`, `tool_result`, `done`,
`cancelled`, `error`. Tool çağrıları arayüzde **olurken** görünüyor — denetim
kaydının kuralının arayüzdeki karşılığı.

**Durum:** her turdan sonra `save_state()` diske yazılıyor; sunucu yeniden
başlayınca `load_state()` geri yüklüyor. Bozuk dosya sessizce silinip sıfırdan
başlanıyor.

**Maliyet:** `EventCapture` ile sayılıyor, çünkü canlı token yalnız olay akışında
görünüyor.

### `server.py` — backend

FastAPI + uvicorn (ikisi de venv'de zaten vardı). Loopback, auth yok — tek
kullanıcılı araç kararının gereği.

| Uç nokta | İş |
|---|---|
| `GET /` | Arayüz |
| `GET /style.css` | **`dashboard.STYLE`'dan** — canlı uygulama ile statik dışa aktarım aynı belirteçleri kullanıyor |
| `GET /api/state` | Açılış raporu, huni, adaylar, geçmiş taramalar, MCP durumu |
| `POST /api/ask` | Deterministik cevap |
| `POST /api/chat` | Canlı ajan turu, **SSE** |
| `POST /api/chat/stop` · `/reset` | İptal · sohbeti sıfırla |
| `POST /api/scan` · `GET /api/scan` | Tarama başlat · canlı log |

**Tarama alt süreç olarak koşuyor:**

```python
self.process = subprocess.Popen(command, stdout=subprocess.PIPE, ...)
threading.Thread(target=self._drain, daemon=True).start()
```

> A subprocess rather than an in-process call: a scan that hangs or dies must not
> take the server with it.

**Tek seferde tek sohbet turu:**

```python
async with CHAT.lock:
    async for event in CHAT.stream(payload.question):
```

Ajanın tek bir bağlamı var; iki eşzamanlı tur onun içine karışırdı.

### `dashboard.py` ve `web/`

`dashboard.py` iki iş yapıyor: tek dosyalık HTML üretmek, ve **bütün stil
belirteçlerini** barındırmak (sunucu da oradan servis ediyor).

Tasarım iki kural setinin buluşması:

**Grafik katmanı** — bütün grafikler tek serili olduğu için tek renk (uzunluk
zaten değeri taşıyor). Huni sıralı rampa üstünde ve en sönük basamağı yüzeye karşı
2:1'i geçiyor. Durum renkleri (güvenilirlik, çöken dal, düşen kaynak) rezerve ve
**her zaman bir kelimeyle**. Her grafiğin altında veri tablosu.

**Arayüz katmanı** — sistem tipografisi, yarı saydam üst çubuk + scroll edge,
**bırakışta değil basışta** geri bildirim, kritik sönümlü geçişler (~0.35 s,
overshoot yok). `prefers-reduced-motion` ve `prefers-reduced-transparency`
karşılanıyor; açık ve koyu ayrı ayrı tasarlandı, ters çevrilmedi.

Karanlık modda huni rampası **ters** — yoğunluk huni boyunca artmalı, koyu zeminde
yoğunluk "daha açık" demek.

`web/app.js` SSE'yi okuyup akışı basıyor, `Stop` düğmesini yönetiyor, ve her
cevabın **hangi yoldan geldiğini** rozetle gösteriyor (`model` / `from scan data`)
— ikisinin garantisi farklı olduğu için.

---

## B6 — `probe_llm.py`

Uç noktanın **ne desteklediğini ölçüyor**, adından çıkarsamıyor. Altı kontrol:
erişim, sohbet, **tool çağrısı**, **yapısal çıktı**, akış, akışta kullanım.

```python
if not key:
    # An empty bearer is a malformed header, and httpx refuses it client-side —
    # which hides the endpoint's real answer behind a local error.
```

Sonunda `.env`'e yapıştırılacak satırları **ölçtüğü değerlerle** basıyor, ve
eksik yetenek varsa neyin bozulacağını söylüyor:

> ⚠ No structured output. The scorer and the memo writer use
> `output_content_type`; they will fail at the end of the funnel.

---

## B7 — Testler (`tests/`, 52 test)

Hepsi ağsız. En önemlisi `__init__.py`:

```python
# Must happen before `config` is imported anywhere in the suite.
for _name in ("VC_LLM_BASE_URL", "VC_LLM_API_KEY", ..., "LLM_MODEL_NAME"):
    os.environ[_name] = ""
os.environ["VC_SKIP_ENV_FILE"] = "1"
os.environ["VC_MCP_DEEPWIKI"] = "0"
```

Anahtar gelince test paketi gerçek modele gitmeye başlamıştı — 5 saniyelik koşu
dakikalara çıktı ve token harcıyordu. Bu dosya paketi kuru moda **zorluyor**.

| Dosya | Neyi kilitliyor |
|---|---|
| `test_policy.py` | Kara liste **her koşulda** reddediyor; muafiyet onu yenemiyor |
| `test_schemas.py` | `Source.url` ve `Score.missing_data` gerçekten zorunlu |
| `test_collectors.py` | Fixture'lı ayrıştırma; **belirsiz isimler birleştirilmiyor** |
| `test_graph.py` | Dal çökünce kardeşler yaşıyor **ve** eksik olan `missing_data`'ya yazılıyor |
| `test_fanin.py` | Ham hata bir dala mal oluyor, **süreye mal olmuyor** |
| `test_observability.py` | Olaylar denetime düşüyor; kapı `DropMessage` döndürüyor |
| `test_answers.py` | Şirket adı genel niyeti yeniyor; **bilinmeyen soru hiçbir yere yönlenmiyor** |
| `test_conversation.py` | Tool workbench üzerinden erişilebiliyor; bellek turlar arası taşınıyor; durum yeniden yükleniyor |

`test_graph.py`'de iki yönlü bir test var — sarmalayıcıyla ve sarmalayıcısız:

```python
self.assertLess(len(survivors), 3,
    "if this ever passes with 3 survivors, the framework fixed the abort "
    "semantics and ResilientClient can be revisited")
```

Yani bu test bir gün geçmeye başlarsa, o bizim için **haber**: framework düzeltmiş,
sarmalayıcı gözden geçirilebilir.

---

# BÖLÜM C — Uçtan uca iki izlek

## C1 — Bir taramanın yolculuğu

```
scan.py main()
  └ _preflight()                      kuru mod / placeholder tez uyarısı
  └ collect(query, days)
      └ HackerNews().run()            → base.fetch → policy.is_allowed
      │                                 → robots → oran sınırı → önbellek/HTTP
      │                                 → Signal listesi (kaynak URL zorunlu)
      └ SecFormD().run()  GitHub().run()
  └ resolve(signals, stats)           normalize.resolve → Company listesi
  └ run_triage(companies, ledger)
      └ triage.prefilter()            kural: kırmızı çizgi / birinci-taraf kanıt
      └ triage.decide()               kararsızsa ucuz model, yapısal karar
  └ run_enrichment(candidates)
      └ graph.enrich(company)
          ├ SingleThreadedAgentRuntime(intervention_handlers=[...])
          ├ EventCapture()            token + tool olayları
          ├ GraphFlow.run_stream()    3 dal paralel → RiskAuditor → Scorer
          ├ asyncio.wait_for(...)     süre sınırı
          └ dal sayımı                eksik dal → Score.missing_data
  └ write_memos(candidates)           eşiği geçenler, en güçlü kademe
  └ report(...)                       md + json + html
```

## C2 — Bir sorunun yolculuğu

**LLM yokken:**

```
tarayıcı → POST /api/ask
  └ answers.answer(question, data)
      └ route()      aday adı? → company:N ; değilse anahtar kelime
      └ catalogue()  dashboard._answers(data) → hazır HTML blokları
  → {path: "rules", title, html}
tarayıcı: "FROM SCAN DATA" rozeti + blok
```

**LLM varken:**

```
tarayıcı → POST /api/chat (SSE)
  └ CHAT.lock                         tek tur
  └ Conversation.stream(question)
      ├ ensure()                      ajan + bellek + workbench(+MCP)
      ├ EventCapture()                canlı token
      └ agent.run_stream(cancellation_token)
          ├ ToolCallRequestEvent   → {"type":"tool"}        arayüzde satır
          ├ ToolCallExecutionEvent → {"type":"tool_result"} tooltip
          ├ ModelClientStreamingChunkEvent → {"type":"chunk"} akan metin
          └ TaskResult             → {"type":"done"}
      └ save_state()                  sohbet diske
tarayıcı: "MODEL" rozeti, akan imleç, Stop düğmesi
```

---

# BÖLÜM D — Yazılmayanlar ve neden

| Eksik | Neden |
|---|---|
| **İzleme durum makinesi** (Faz 5) | Koşular arası hiçbir şey kalıcı değil → değişiklik tespiti yok. Huninin 4. ilkesi ("izleme döngüseldir") kodda karşılıksız. `demo-brain-agent/taskboard.py` bekliyor |
| **MCP sunucusu + telefon kanalı** (Faz 7) | Uyarı telefona gitmiyor |
| **İnsan onay kapısı** | Runtime mekanizması hazır ve testli; `UserProxyAgent` bağlı değil |
| **OpenTelemetry** (Faz 8) | Maliyet ölçülüyor, iz sürülmüyor. Runtime bizim olduğu için en kolay kalan iş |
| **Geri-test** | **Recall ölçülmüyor** — "kaçırdığımız iyi girişim" sayısı bilinmiyor, ve huninin en pahalı hatası tam olarak o |
| **ChromaDB semantik dedup** | Paket kurulu değil; deterministik anahtar + `difflib` yerine geçiyor |
| **Dağıtık runtime, component config, kod yürütücüler** | Gerek yok / uygulanamaz |

**Ve en önemlisi: tez hâlâ placeholder.** Bu yüzden kural ön elemesi neredeyse
hiçbir şeyi çözemiyor ve adayların tamamı modele gidiyor. Kendi kırmızı çizgilerin
yazılınca triyajın çoğu LLM'e hiç gitmeden elenecek — **huni asıl o zaman
ucuzluyor.**

---

# BÖLÜM E — Bu koddan çıkan üç genel ders

**1. Arızalar sessiz.** Bu projede bulunan hataların hiçbiri exception fırlatmadı:
sıfır döndü, boş kaldı, asılı kaldı, ya da hata metnini cevap diye sundu. Bir
çerçeveyi öğrenmenin yolu API'sini okumak değil, **arıza davranışını ölçmek**.

**2. Beyan ile yetenek aynı şey değil.** `model_info` bir sözleşme değil iddia;
kütüphane doğrulamıyor. Aynı mantık kodun kendisi için de geçerli — "ben şunu
yaparım" demesi yaptığı anlamına gelmiyor. `probe_llm.py` ve
`compare_fanin.py` bu yüzden var.

**3. Aynı mekanizma iki bağlamda iki farklı doğru.** `ResilientClient` paralel
dalda doğru (kardeşleri korur), sohbette yanlış (arızayı görüş gibi gösterir).
Bu yüzden `Ledger`'da iki fabrika var. Bir kalıbı "iyi" diye her yere taşımak,
onu bir yerde zarara çeviriyor.
