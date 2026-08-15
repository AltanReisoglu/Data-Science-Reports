# 11 — AutoGen Core: Türkçe rehber

*[05-autogen-core-user-guide.md](05-autogen-core-user-guide.md)'nin 42 sayfasının
tamamı, Türkçe. Her bölümün başında orijinaldeki satır numarası var: `05:2841` gibi.*

---

## Bu belge ne, ne değil

[10](10-agentchat-turkce.md) ile aynı kural: **anlatı Türkçe, kod ve API adları
İngilizce.** `RoutedAgent`'ı çevirirsem çalışmaz. Orijinal 41.114 kelime ve
büyük kısmı kod; kelimesi kelimesine çeviri hem devasa hem yanlış olurdu.

**Neden bu belge diğerinden önemli:** AgentChat (`08`, `10`) hazır ajanları ve
takımları anlatıyor — günlük iş orada. Core ise **altındaki makine**: aktör
modeli, mesajlaşma, runtime, olaylar. Bu projede AgentChat'in çözemediği
sorunları bir kat aşağı inip core'da çözdük, ve bulduğumuz 13 inceliğin çoğu
buradan çıktı.

**Okuma sırası önerisi:** Temel kavramlar (§2) → Framework rehberi (§3) →
Concurrent Agents (§5). Cookbook'u baştan sona okuma; ihtiyaç oldukça bak.

**İlgili:** [06](06-autogen-incelikleri.md) ısıran incelikler ·
[07](07-kod-rehberi.md) kavram↔kod köprüsü · [10](10-agentchat-turkce.md) AgentChat Türkçe

---

# BÖLÜM 1 — Başlangıç

## Core nedir · `05:73`

`autogen-core`, olay güdümlü bir **aktör** çerçevesi. Tanımı dar ve bilinçli:
bir ajan, **mesaj alan ve mesaja karşılık üreten** bir birimdir. Zeki olması
gerekmez; LLM içermek zorunda değildir.

Vaat ettiği şeyler: ölçeklenebilirlik, dağıtılabilirlik (aynı API ile başka
makinelerde), dil bağımsızlığı (Python + .NET), ve **gözlemlenebilirlik**.

## Kurulum · `05:179`

```bash
pip install "autogen-core"
```

Kod yürütücü için ayrıca `DockerCommandLineCodeExecutor` ve Docker gerekir.

## Hızlı başlangıç · `05:278`

Core'un "merhaba dünya"sı AgentChat'inkinden farklı — burada **her şeyi kendin
bağlarsın**:

```python
@dataclass
class MyMessageType:
    content: str

@default_subscription
class MyAgent(RoutedAgent):
    @message_handler
    async def handle(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"{self.id.type} received: {message.content}")

runtime = SingleThreadedAgentRuntime()
await MyAgent.register(runtime, "my_agent", lambda: MyAgent("demo"))
runtime.start()
await runtime.publish_message(MyMessageType("hello"), DefaultTopicId())
await runtime.stop_when_idle()
```

Dört adım: mesaj tipi tanımla → handler yaz → ajanı runtime'a kaydet → runtime'ı
başlat ve mesaj yayınla.

---

# BÖLÜM 2 — Temel kavramlar

## Ajan ve çok-ajanlı uygulamalar · `05:455`

Çok-ajanlı yapmanın sebebi **zekâ değil ayrıştırma**. Üç analistimiz üç ayrı
ajan çünkü üç ayrı kaynağa bakıyorlar ve **paralel koşabiliyorlar** — daha akıllı
oldukları için değil.

Kılavuz iki fayda sayıyor: modülerlik (her ajan tek işi yapar, ayrı test edilir)
ve uzmanlaşma (farklı rol, farklı model, farklı tool).

## Ajan çalışma zamanı ortamları · `05:490`

Runtime; mesajlaşmayı yürüten, kimlik ve yaşam döngüsünü yöneten katman. İki tip:

| Tip | Ne zaman |
|---|---|
| **Standalone** — `SingleThreadedAgentRuntime` | Tek süreç, tek dil |
| **Distributed** — host + worker'lar (gRPC) | Çok süreç, çok makine, çok dil |

Kritik vaat: **ikisi aynı API'yi sunuyor.** Ajan kodunu değiştirmeden geçiş
yapabiliyorsun.

> **Bizim ölçtüğümüz:** Runtime'ı **kendin verirsen** hata semantiği değişiyor.
> `InterventionHandler` takmanın tek yolu runtime'ı kendin kurmak; ama o zaman
> çöken bir ajan `run_stream`'i **fırlatmıyor, asıyor** (gömülü runtime'da
> fırlatıyordu). `MaxMessageTermination` kurtaramıyor çünkü yeni mesaj da
> gelmiyor. Tek çare duvar saati sınırı. → [06 §7](06-autogen-incelikleri.md)

## Uygulama yığını · `05:543`

Katmanlar: en altta mesajlaşma ve olay güdümlü ajanlar, üstünde geliştiricinin
tanımladığı **mesaj tipleri** ve **ajan davranışları**, en üstte uygulama.

Kılavuzun verdiği örnek bir kod yazma/çalıştırma döngüsü: `CodingResultMsg` →
`ExecutionResultMsg`. Ajanlar birbirini **tanımaz**, yalnız mesaj tiplerini
bilir — gevşek bağlılık buradan geliyor.

## Ajan kimliği ve yaşam döngüsü · `05:601`

Bir ajanın kimliği **`(type, key)`** çiftidir.

- `type` — hangi sınıf/rol ("analyst")
- `key` — hangi örnek ("company-42")

Runtime, ajanı **ilk mesaj geldiğinde yaratır** (lazy). `register` ile tipi ve
bir **fabrika** kaydedersin; örnek kendiliğinden oluşur.

```python
await MyAgent.register(runtime, "my_agent", lambda: MyAgent("demo"))
```

> **Pratik sonucu:** `ToolCallEvent`'in taşıdığı `agent_id`, ajan runtime
> yönetimindeki bir handler içindeyken dolu; çıplak `agent.run()`'da `None`.
> Canlı doğrulandı. → [06 §6](06-autogen-incelikleri.md)

## Topic ve abonelik · `05:670`

Pub/sub'ın adresleme modeli. **En çok atlanan ve en değerli sayfa.**

Bir topic'in iki parçası var:

- **tip** — mesaj kategorisi (`"github_issues"`)
- **kaynak** — o kategorideki benzersiz kimlik (`"repo/issue-42"`)

`TypeSubscription`, bir topic **tipini** bir ajan **tipine** bağlar. Ve asıl
incelik şu:

> **Topic kaynağı, ajan anahtarına dönüşür.**

Yani `("gorev", "sirket-42")` topic'ine yayın yaparsan, runtime
`("analist", "sirket-42")` ajanını yaratır. **Şirket başına izole ajan örneği
bedava.** Çok kiracılı yapının hazır mekanizması bu.

Kılavuzun önerdiği üç senaryo:

| Senaryo | Nasıl |
|---|---|
| Tek kiracı, tek topic | Hepsi aynı topic tipi, kaynak sabit `"default"` |
| Tek kiracı, çok topic | Ajan başına ayrı topic tipi, kaynak aynı |
| Çok kiracı | Kaynak veriye bağlı — kiracı/oturum/kayıt başına izolasyon |

Bizde `fanin.py` tek kaynak (`"default"`) kullanıyor, çünkü tek kullanıcılı bir
araç. Hacim sorusu cevaplanınca bu mekanizma hazır cevap.

---

# BÖLÜM 3 — Framework rehberi

## Ajan ve ajan çalışma zamanı · `05:894`

`RoutedAgent` + `@message_handler`: mesaj **tipine göre** yönlendirme.

```python
class MyAgent(RoutedAgent):
    @message_handler
    async def handle_text(self, message: TextMessage, ctx: MessageContext) -> None: ...

    @message_handler
    async def handle_image(self, message: ImageMessage, ctx: MessageContext) -> None: ...
```

Bir ajanın birden çok handler'ı olabilir; hangisinin çalışacağını **tip
anotasyonu** belirler. Union tipi de kullanılabilir.

> **Tuzak — bizi yakaladı:** Tip çıkarımı `get_type_hints()` ile yapılıyor ve
> `from __future__ import annotations` varken anotasyonlar **modül genelinde**
> çözülüyor. `MessageContext`'i fonksiyon içinde import edersen kayıt sırasında
> çıplak bir `NameError` alırsın — handler'ın kendisinde değil, **kayıtta**.
> Ayrıca parametre adları bağlayıcı: `message` ve `ctx` olmak zorunda.
> → [06 §11](06-autogen-incelikleri.md)

## Mesajlaşma ve iletişim · `05:1108`

Üç biçim:

| Biçim | Nasıl | Ne zaman |
|---|---|---|
| **Doğrudan mesaj** | `send_message(msg, AgentId(...))` | Cevap bekliyorsan (RPC gibi) |
| **Yayın** | `publish_message(msg, TopicId(...))` | Kim dinliyorsa duysun |
| **Cevap** | handler'dan `return` | Doğrudan mesajın karşılığı |

Yayının **cevabı yoktur** — bu bir kısıt değil, tasarım: yayınla haberleşen
ajanlar birbirini tanımak zorunda kalmaz.

Sayfa ayrıca `CancellationToken` ile çalışan bir işi iptal etmeyi anlatıyor.

## Günlükleme · `05:1534`

İki logger:

```python
logging.getLogger(EVENT_LOGGER_NAME)  # "autogen_core.events" — yapılandırılmış
logging.getLogger(TRACE_LOGGER_NAME)  # insan okunur iz
```

Olay akışında `LLMCallEvent`, `LLMStreamEndEvent`, `ToolCallEvent`,
`MessageEvent`, `MessageDroppedEvent`, `MessageHandlerExceptionEvent` var.

> **İki tuzak, ikisi de ölçüldü:**
> `create()` → `LLMCallEvent`, `create_stream()` → **yalnız** `LLMStreamEndEvent`.
> Akış kullanınca sadece ilkini dinlersen maliyet **0** raporlanır.
> Ve `LLMCallEvent` alanlarını öznitelik yapıyor ama `ToolCallEvent` her şeyi
> **`.kwargs` sözlüğünde** tutuyor — `event.tool_name` yazarsan `AttributeError`
> alırsın ve o hata `logging.Handler` içinde **yutulur**: olay hiç kaydedilmez,
> hata da görünmez. → [06 §4, §6](06-autogen-incelikleri.md)

## OpenTelemetry · `05:1647`

Runtime `tracer_provider` alır; ajan mesajları ve model çağrıları span'a dönüşür.

```python
runtime = SingleThreadedAgentRuntime(tracer_provider=tracer_provider)
```

Bizde henüz yok — ama runtime'ı zaten kendimiz kurduğumuz için tek satır. Kalan
işlerin en kolayı.

## Dağıtık çalışma zamanı · `05:1742`

`GrpcWorkerAgentRuntimeHost` (koordinatör) + `GrpcWorkerAgentRuntime`
(işçiler). Ajanlar makinelere dağılır, **ajan kodu değişmez**.

```python
host = GrpcWorkerAgentRuntimeHost(address="localhost:50051")
host.start()
worker = GrpcWorkerAgentRuntime(host_address="localhost:50051")
```

Sayfa ayrıca diller arası çalışmayı (Python ↔ .NET) ve gRPC seçeneklerini
anlatıyor.

Bizde yok: tek süreç yetiyor ve hacim sorusu cevaplanmadan gereksiz karmaşa.

## Bileşen yapılandırma · `05:1888`

```python
config = client.dump_component()          # → JSON
client = ChatCompletionClient.load_component(config)
```

Ajanı, model istemcisini, tool'u JSON'a çevirip geri yükleme. Kendi sınıfını
serileştirilebilir yapmak için `ComponentToConfig` / `ComponentFromConfig`.

AutoGen'de **"skill" diye bir soyutlama yok**; en yakın karşılığı bu.

---

# BÖLÜM 4 — Bileşenler

## Model istemcileri · `05:1980`

`ChatCompletionClient` arayüzü. `create()` ve `create_stream()`, `total_usage()`,
`count_tokens()`, `remaining_tokens()`.

```python
model_client = OpenAIChatCompletionClient(model="gpt-4o")
result = await model_client.create([UserMessage(content="Hello", source="user")])
```

Sayfa ayrıca **önbelleği** (`ChatCompletionCache`) ve model yeteneklerini
(`model_info`) anlatıyor.

> **En pahalı tuzak:** Model adı bilinen bir OpenAI modeli değilse `model_info`
> **zorunlu**. "OpenAI-uyumlu endpoint" kullanan herkes buraya düşer — bizde ilk
> canlı istekte patladı. Ve `model_info` bir **beyandır**, ölçüm değil:
> desteklenmeyen bir yeteneği iddia edersen hata başlangıçta değil, huninin en
> sonunda (skorlayıcıda) çıkar. Bu yüzden `pipeline/probe_llm.py` var.
> → [06 §3](06-autogen-incelikleri.md)

## Model bağlamı · `05:2341`

Ajanın belleği. Model istemcisine gönderilecek mesaj listesini yöneten nesne.

| Sınıf | Ne yapar |
|---|---|
| `UnboundedChatCompletionContext` | Her şeyi tutar |
| `BufferedChatCompletionContext(buffer_size=N)` | Son N **mesajı** |
| `HeadAndTailChatCompletionContext` | Baş + son |
| `TokenLimitedChatCompletionContext` | **Token** sınırı |

> **İnce nokta:** `buffer_size` **mesaj** sayar, token değil. Uzun bir sohbette
> maliyeti sınırlayan şey bu değildir — onun için `TokenLimited...` var.
> Ve `save_state`'in kaydettiği şey bağlamın tuttuğudur: bağlam yoksa
> kaydedilecek sohbet de yoktur.

## Tool'lar · `05:2473`

`FunctionTool` bir Python fonksiyonunu tool'a çevirir. Şemayı **imzadan ve
docstring'den** üretir — yani docstring dokümantasyon değil, **arayüz**.

```python
async def get_stock_price(ticker: str, date: Annotated[str, "YYYY/MM/DD"]) -> float:
    """Get the stock price."""
    return random.uniform(10, 200)

tool = FunctionTool(get_stock_price, description="Get the stock price.")
```

Sayfa ayrıca `PythonCodeExecutionTool`'u ve **tool-equipped agent** kurmayı
anlatıyor: model tool çağrısı döndürür, sen çalıştırıp sonucu geri verirsin.

## Workbench ve MCP · `05:2841`

Workbench bir **tool kaynağı** — tek tek tool yerine "bana hangi tool'ların
olduğunu söyle" diyebildiğin nesne. `list_tools()` ve `call_tool()`.

```python
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
workbench = McpWorkbench(server_params=params)
tools = await workbench.list_tools()
result = await workbench.call_tool(name, arguments)
```

Uzak bir MCP sunucusu tam olarak bu: ajan yazılırken var olmayan tool'ları
listeleyebilir.

> **Tuzak:** `tools=` ile `workbench=` **aynı ajana verilemez**
> (`ValueError: Tools cannot be used with a workbench`). Yerel fonksiyonları
> `StaticWorkbench`'e sarıp **liste** vererek ikisini birleştirirsin.
> → [06 §1](06-autogen-incelikleri.md)

## Komut satırı kod yürütücüleri · `05:3054`

```python
LocalCommandLineCodeExecutor(work_dir=...)     # yerel — dikkat
DockerCommandLineCodeExecutor(image=...)       # izole
```

Kılavuz yerel yürütücü için açık uyarı veriyor: modelin ürettiği kodu izolesiz
çalıştırmak risklidir. Docker önerilir.

Bizde yok — VC hattında kod yürütülmüyor.

---

# BÖLÜM 5 — Çok-ajan tasarım desenleri

## Giriş · `05:3209`

Desenler, ajanların **nasıl dizildiğini** anlatıyor. Ana ayrım: akış **önceden
mi çizili** (sıralı, graf) yoksa **konuşmadan mı doğuyor** (group chat, handoff).

## Eşzamanlı ajanlar · `05:3236`

**Bu projenin en çok işine yarayan desen.** Üç varyant:

1. **Tek mesaj, çok işleyici** — aynı topic'e abone ajanlar, hepsi aynı anda çalışır
2. **Çok mesaj, çok işleyici** — `@type_subscription` ile mesaj tipine göre yönlendirme
3. **Doğrudan mesajlaşma** — `AgentId` ile adresleme

Ve asıl kritik parça: **sonuç toplama**.

```python
queue = asyncio.Queue[TaskResponse]()

async def collect_result(_agent: ClosureContext, message: TaskResponse, ctx: MessageContext) -> None:
    await queue.put(message)

await ClosureAgent.register_closure(
    runtime, "collector", collect_result,
    subscriptions=lambda: [TypeSubscription(topic_type=RESULTS, agent_type="collector")],
)
```

İşçiler sonucu **yayınlar**, `ClosureAgent` kuyruğa boşaltır, sen kuyruktan
okursun.

> **Neden bu bizim için önemli:** `pipeline/fanin.py` bunun birebir uygulaması.
> Ölçtük — ham bir hata altında AgentChat'in `GraphFlow`'u **0–1 dal** kurtarıyor
> ve süre sınırını dolduruyor; pub/sub + kuyruk **2 dal** kurtarıyor ve ~3 ms'de
> dönüyor. Sebep basit: sonuç üretildiği anda yayınlanıyor ve kuyruk onu çoktan
> tutuyor. **Güvenilmeyecek bariyer yok, çünkü bariyer yok.**
> → [06 §8](06-autogen-incelikleri.md), `pipeline/compare_fanin.py`

## Sıralı iş akışı · `05:3504`

Her ajan bir öncekinin çıktısını alır. `@type_subscription` ile her adım kendi
topic tipini dinler:

```python
@type_subscription(topic_type="concept_extractor")
class ConceptExtractorAgent(RoutedAgent): ...
```

Zincir: kavram çıkar → yaz → biçimlendir. Bizim katman mimarimiz bu — ama
ajanlarla değil kodla kurulu, çünkü ilk iki katmanda LLM yok.

## Group Chat · `05:3772`

Ortak thread + bir **konuşmacı seçici**. Mesaj tipleri: `GroupChatMessage`
(içerik) ve `RequestToSpeak` (sıra sende).

> **Kılavuzun kendi cümlesi:** *"not meant to be used in real applications…
> a starting point."* Üretim sürümü AgentChat'in `SelectorGroupChat`'i.
> Yani buradan alınacak bir şey yok — mekanizmayı anlamak dışında.

## Devirler (Handoffs) · `05:4349`

OpenAI'ın "Swarm" desenine dayanıyor. Ajan, bir **devir tool'u** çağırarak
konuşmayı başkasına verir. Kılavuzun örneği bir müşteri hizmetleri akışı:
triyaj → sorun/onarım → insan.

Bizde yok: huni tek yönlü, ve POC'ta ölçtüğümüz kadarıyla **en pahalı desen**
(334 token; Selector 204) — her devir bir tool çağrısı artı iş üretmeyen bir LLM
turu harcıyor.

## Mixture of Agents · `05:4989`

Katmanlı işçi ajanlar: her katmanın çıktıları birleştirilip sonraki katmana
verilir, sonunda bir orkestratör toplar.

> **Almama sebebim bir bulgu:** Orkestratör `asyncio.gather(...)` ile topluyor —
> yani POC'ta (`poc/desen_5_core_aktor.py`) sessiz kardeş kaybının kaynağı olan
> yapı. **Resmî desenler bu konuda birbiriyle çelişiyor:** Concurrent Agents
> kuyrukla topluyor, Mixture of Agents `gather` ile. Tek bir kütüphane
> hatasından daha güçlü bir bulgu — kılavuzun kendisi iki farklı arıza davranışı
> öneriyor.

## Çok-ajan münazarası · `05:5358`

**Seyrek topoloji**: her çözücü ajan yalnız birkaç komşusuna bağlı (dört ajan bir
kare, her biri iki komşuya). Birkaç tur boyunca ajanlar komşularının cevaplarını
görüp kendi cevaplarını düzeltir; sonunda bir toplayıcı **çoğunluk oyu** alır.

Bizde yok: skorlamada **sabit rubrik** istiyoruz, oy değil — "adil kıyas" ilkesi.
Oylama, aynı şirkete iki koşuda iki puan verme riskini geri getirir. Üstelik
N ajan × R tur maliyeti ciddi.

## Yansıma (Reflection) · `05:5822`

Üretici + eleştirmen döngüsü: `CoderAgent` kod yazar, `ReviewerAgent`
`CodeReviewResult` döndürür, onaylanana kadar döner.

> Bizim `RiskAuditor`'ımız **tek turluk** bir reflection — üç analizi çapraz
> denetliyor. Döngüye çevirmedim: kılavuzun kendisi **durma ölçütü önermiyor**,
> yani faturayı sınırlayan tek şey kalmıyor.

## Kod yürütme · `05:6188`

Group chat + kod yürütücü: model kod yazar, bir ajan çalıştırır, sonuç geri
döner. AutoGen'in kurucu makalesinin konusu buydu.

Bizde uygulanamaz.

---

# BÖLÜM 6 — Cookbook

*Tarif koleksiyonu. Baştan sona okunacak bir bölüm değil; ihtiyaç oldukça bak.*

## Cookbook girişi · `05:6461`

## Azure OpenAI + AAD kimlik doğrulama · `05:6490`

API anahtarı yerine Azure Active Directory token'ı:
`AzureOpenAIChatCompletionClient` + `get_bearer_token_provider(DefaultAzureCredential())`.

## Müdahale ile sonlandırma · `05:6534`

`InterventionHandler`, mesaj hattına **teslimden önce** oturur.

```python
class TerminationHandler(DefaultInterventionHandler):
    async def on_publish(self, message, *, message_context):
        if isinstance(message, Termination):
            self._termination_value = message
        return message

runtime = SingleThreadedAgentRuntime(intervention_handlers=[TerminationHandler()])
```

> **Uyarı:** Handler `None` döndürürse runtime uyarı basar — mesajı ya da
> `DropMessage`'ı **açıkça** döndürmelisin. Bizde `on_publish` ve `on_response`
> bu yüzden açıkça geçiş fonksiyonu olarak yazılı.

## Tool yürütme için kullanıcı onayı · `05:6638`

Aynı mekanizmanın asıl kullanımı: `DropMessage` döndürerek bir tool çağrısını
**engellemek**. Onay kapısı ajanın uyum göstermeyi seçmesine değil **runtime'a**
dayanır.

> Bizde bağlı ve testli ama **gözlemci modda** — buradaki tool'ların hepsi
> salt-okunur, her çağrıda onay sormak tören olurdu. İlk mutasyon yapan tool
> geldiğinde kapı hazır.

## Ajanla sonuç çıkarma · `05:6825`

`ClosureAgent` ile yayınlanan sonuçları bir kuyruğa toplama — Concurrent
Agents'taki toplama mekanizmasının tek başına anlatımı.

## OpenAI Assistant ajanı · `05:6914`

OpenAI'ın Assistants API'sini bir AutoGen ajanına sarma. Thread yönetimi, dosya
arama, kod yorumlayıcı.

## LangGraph destekli ajan · `05:7482`

Bir LangGraph grafını `RoutedAgent` içine sarma. İki çerçeveyi birlikte
kullanmanın yolu: AutoGen mesajlaşmayı, LangGraph iç akışı yönetiyor.

## LlamaIndex destekli ajan · `05:7662`

Aynı fikir LlamaIndex için: RAG'i LlamaIndex yapıyor, ajan sarmalayıcı AutoGen.

## Yerel LLM'ler: LiteLLM ve Ollama · `05:8030`

Yerel model çalıştırma. `OpenAIChatCompletionClient` + `base_url` ile LiteLLM
proxy'sine ya da Ollama'ya bağlanma — ve **`model_info` vermek zorunda kalma**
(bkz. §Model istemcileri tuzağı).

## Kodunu yerelde izleme · `05:8192`

OpenTelemetry'yi yerelde kurup Jaeger gibi bir toplayıcıya bağlama.

## Topic abonelik senaryoları · `05:8233`

Temel kavramlardaki topic/abonelik anlatımının **çalışan örneklerle** hâli: tek
kiracı tek topic, tek kiracı çok topic, çok kiracı. Çok kiracılı yapı
kuracaksan okunacak sayfa bu.

## GPT-4o ile yapısal çıktı · `05:8745`

`response_format` ile Pydantic şeması dayatma. AgentChat'teki
`output_content_type` bunun üstüne kuruluyor.

## LLM kullanımını logger ile takip · `05:8840`

```python
class LLMUsageTracker(logging.Handler):
    def emit(self, record):
        if isinstance(record.msg, LLMCallEvent):
            self._prompt_tokens += record.msg.prompt_tokens
            self._completion_tokens += record.msg.completion_tokens

logging.getLogger(EVENT_LOGGER_NAME).handlers = [LLMUsageTracker()]
```

> Bizim `observability.py`'nin çekirdeği bu — ama `LLMStreamEndEvent` de
> dinlenecek şekilde genişletilmiş hâli. Kılavuzdaki bu tarif akış kullanınca
> **eksik kalıyor**.

---

# BÖLÜM 7 — Sık sorulanlar · `05:8914`

Kılavuzun kapattığı konular: bir ajanı nasıl uzaktan çağırırım, runtime'lar
arası fark, aynı ajanın birden çok örneği, mesajın kaybolması, hata yönetimi,
ve AgentChat ile core arasında nasıl seçim yapılır.

---

## Ek — Core'u okurken aklında tutulacaklar

**1. AgentChat'in bittiği yer core'un başladığı yerdir.** Bu projede paralel dal
kaybını AgentChat'te çözemedik; core'a inip `ClosureAgent` + kuyrukla çözdük.
Aşağı inmek zorunda değilsin ama **inebildiğini bilmek** bir güvence.

**2. Kılavuz "nasıl yapılır"ı anlatıyor, "ne zaman bozulur"u anlatmıyor.**
Bulduğumuz 13 incelik hiçbir sayfada yazmıyor. → [06](06-autogen-incelikleri.md)

**3. Arızalar sessiz.** Bu projede bulunan hataların hiçbiri exception
fırlatmadı: sıfır döndü, boş kaldı, asılı kaldı, ya da hata metnini cevap diye
sundu. Core'u öğrenmenin yolu API'sini okumak değil, **arıza davranışını
ölçmek**.

**4. Topic kaynağı → ajan anahtarı eşlemesini unutma.** Kılavuzun en az
konuşulan, en çok işe yarayacak cümlesi bu; ölçek gerektiğinde ilk bakılacak yer.

---

*Kaynak: [05-autogen-core-user-guide.md](05-autogen-core-user-guide.md)
(microsoft/autogen, MIT). Kod ve API adları bilinçli olarak İngilizce bırakıldı.*
