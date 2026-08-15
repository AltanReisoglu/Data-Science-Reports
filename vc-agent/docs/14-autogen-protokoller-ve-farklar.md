# 14 — AutoGen'in içi, konuştuğu protokoller ve komşuları

*Bu belge üç soruyu ayrı ayrı cevaplıyor: **içinde ne var**, **neyi nasıl yapıyor**,
**başkalarından nerede ayrılıp nerede benziyor**. Karşılaştırmanın kısa hâli
[09](09-framework-karsilastirma.md)'da; burası protokol katmanına iniyor.*

---

## 0 — Kural: her iddia etiketli, ve neyin ölçülebildiği belli

[09](09-framework-karsilastirma.md)'daki etiket düzeni aynen sürüyor:

| Etiket | Anlamı |
|---|---|
| **`[ölçüldü]`** | Bu makinede koşturuldu ya da kurulu pakete introspeksiyon yapıldı; çıktı belgede |
| **`[kaynak]`** | Birincil kaynakta yazıyor — resmî kılavuz, `.proto`, docstring; atıf verili |
| **`[teyitsiz]`** | Okudum, doğrulamadım. Çoğu zaman kurulu olmayan bir paket hakkında |

Ve dürüst olmak gereken yer burası: **karşılaştırdığım framework'lerin çoğu bu
ortamda kurulu değil.** `[ölçüldü]` ne kadar uzağa gidebilir, tablosu:

```
autogen-core           0.7.5      ← ölçülebilir
autogen-agentchat      0.7.5      ← ölçülebilir
autogen-ext            0.7.5      ← ölçülebilir
google-adk             2.6.3      ← ölçülebilir
mcp                    1.29.0
opentelemetry-api      1.42.1
protobuf               5.29.6
openai                 3.0.0
pydantic               2.13.4
grpcio                 —          ← YOK: dağıtık runtime import bile edilemiyor
langgraph              —          ← YOK: bu belgedeki her LangGraph satırı [teyitsiz]
crewai                 —          ← YOK
openai-agents          —          ← YOK
agno                   —          ← YOK
deepagents             —          ← YOK
a2a-sdk                —          ← YOK
```

**`[ölçüldü]`** — `importlib.metadata` ile, 2026-08-14.

Yani: AutoGen ve ADK hakkında söylediklerim pakete bakılarak yazıldı; LangGraph,
CrewAI, Agents SDK ve DeepAgents hakkındakiler **okuduğumdan**. İkisini
karıştırmıyorum.

**Bir ara yol var ve §3.9'da onu kullandım:** kurulu olmayan bir framework'ün
API'sini, o reponun **kendi test dosyalarındaki gerçek koddan** okumak. Bu
`[kaynak]` sayılır — çünkü `openai/openai-agents-python` içindeki bir test,
`handoffs=[...]`'ın nasıl çağrıldığının birincil kanıtı. Ama yalnız **API
yüzeyi** için geçerli: o kodun çalışırken nasıl davrandığı hâlâ `[teyitsiz]`.
Her alıntının repo ve dosya yolu yazılı, kontrol edebilirsin.

---

# BÖLÜM 1 — İçinde ne var

## 1.1 Üç paket, envanter

**`[ölçüldü]`** Kurulu 0.7.5'in alt paketleri:

| Paket | Alt paketler |
|---|---|
| `autogen_core` | `code_executor`, `memory`, `model_context`, `models`, `tool_agent`, `tools`, `utils` |
| `autogen_agentchat` | `agents`, `base`, `conditions`, `state`, `teams`, `tools`, `ui`, `utils` |
| `autogen_ext` | `agents`, `auth`, `cache_store`, `code_executors`, `experimental`, `memory`, `models`, `runtimes`, `teams`, `tools`, `ui` |

Dikkat çeken şey **`autogen_core`'un ne kadar küçük olduğu**: yedi alt paket, ve
içinde model istemcisi *uygulaması* yok — sadece arayüz. Somut her şey `ext`'te.
Bu tesadüf değil, katman ayrımının kendisi.

## 1.2 Hazır gelen sınıflar

**`[ölçüldü]`** `dir()` çıktıları, birebir:

**Takımlar (5):**
`RoundRobinGroupChat`, `SelectorGroupChat`, `Swarm`, `GraphFlow`, `MagenticOneGroupChat`
— artı graf kurucuları: `DiGraph`, `DiGraphBuilder`, `DiGraphEdge`, `DiGraphNode`

**Ajanlar (AgentChat, 8 sınıf + 3 onay tipi):**
`AssistantAgent`, `BaseChatAgent`, `CodeExecutorAgent`, `UserProxyAgent`,
`SocietyOfMindAgent`, `MessageFilterAgent` (+`MessageFilterConfig`, `PerSourceFilter`),
`ApprovalRequest`, `ApprovalResponse`, `ApprovalFuncType`

**Sonlandırma koşulları (11):**
`MaxMessageTermination`, `TextMentionTermination`, `TextMessageTermination`,
`TokenUsageTermination`, `TimeoutTermination`, `HandoffTermination`,
`SourceMatchTermination`, `ExternalTermination`, `StopMessageTermination`,
`FunctionCallTermination`, `FunctionalTermination`

**Model bağlamları (6):**
`UnboundedChatCompletionContext`, `BufferedChatCompletionContext`,
`HeadAndTailChatCompletionContext`, `TokenLimitedChatCompletionContext`,
`ChatCompletionContext`, `ChatCompletionContextState`

**Bellek:** `autogen_core.memory`'de yalnız protokol + `ListMemory`;
gerçek uygulamalar `autogen_ext.memory`'de: **`chromadb`, `mem0`, `redis`, `canvas`**

**Model sağlayıcıları** (`autogen_ext.models`): `openai`, `anthropic`, `azure`,
`ollama`, `llama_cpp`, `semantic_kernel`, `replay`, `cache`

**Tool aileleri** (`autogen_ext.tools`): `mcp`, `http`, `code_execution`,
`graphrag`, `azure`, **`langchain`**, **`semantic_kernel`**

**Hazır ajanlar** (`autogen_ext.agents`): `web_surfer`, `file_surfer`,
`video_surfer`, `magentic_one`, `openai`, `azure`

## 1.3 Bu envanterden çıkan üç okuma

**① Rakip ekosistemleri tool olarak içeri alıyor.** `tools/langchain` ve
`tools/semantic_kernel` adapter'ları var; `models/semantic_kernel` de öyle. Yani
AutoGen "LangChain'in alternatifi" diye konumlanmıyor — **LangChain tool'unu
çalıştırabiliyor.** Rekabet ettiği yer orkestrasyon, tool ekosistemi değil.

**② İnsan onayı kutunun içinden çıkıyor, ama yalnız kod için.**
**`[ölçüldü]`** `CodeExecutorAgent.__init__` bir `approval_func` parametresi
alıyor; `ApprovalRequest` alanları `(code, context)`, `ApprovalResponse` alanları
`(approved, reason)`. Yani hazır onay kapısı **kod yürütmeye** özel. Genel bir
"şu tool'u çağırmadan önce sor" mekanizması için hâlâ `InterventionHandler`
yazman gerekiyor (`05:6638`) — bizim planladığımız onay kapısı da o yolda olacak.

**③ `SocietyOfMindAgent` sessizce çok önemli.** Bir **takımı** tek ajan gibi
paketliyor. Yani takımlar iç içe geçebiliyor: alt takımın gürültüsü dışarı
sızmadan, sadece sonucu görünüyor. Hiyerarşi için ayrı bir soyutlama yok, çünkü
buna gerek yok.

---

# BÖLÜM 2 — Konuştuğu protokoller

Burası bu belgenin asıl sebebi. Çoğu karşılaştırma "AutoGen pub/sub kullanır"
deyip geçiyor; **hangi tel üstünde ne konuşulduğu** hiç yazmıyor.

## 2.1 CloudEvents v1 — topic modeli AutoGen'in icadı değil

**`[kaynak]`** `autogen_core/_topic.py` içindeki `TopicId` docstring'i, iki alanı
için de doğrudan CNCF spesifikasyonuna atıf veriyor:

```python
# TopicId.type
"""Must match the pattern: ^[\w\-\.\:\=]+\Z
Learn more here: https://github.com/cloudevents/spec/.../spec.md#type"""

# TopicId.source
"""Identifies the context in which an event happened. Adhere's to the cloud event spec.
Learn more here: https://github.com/cloudevents/spec/.../spec.md#source-1"""
```

**`[ölçüldü]`** Ve dağıtık runtime'da bu bir benzetme değil, birebir format:
`autogen_ext/runtimes/grpc/protos/cloudevent_pb2.py` derlenmiş hâlde geliyor ve
protobuf paketi **`io.cloudevents.v1`**, tek mesaj tipi **`CloudEvent`**.

**Neden önemli:** `12`'de "topic kaynağı ajan anahtarına dönüşür" dediğim
mekanizmanın altında **standart bir olay formatı** var. Yani AutoGen'in yayınları
prensipte Kafka/NATS/Knative gibi CloudEvents konuşan her şeyin diline çevrilebilir.
Bunu kimse yapmamış olabilir — ama format seçimi bunu düşünerek yapılmış.

## 2.2 gRPC + protobuf — dağıtık runtime'ın teli

**`[ölçüldü]`** `agent_worker.proto`'nun derlenmiş descriptor'ından okundu.
Protobuf paketi `agents`, tek servis:

```
servis AgentRpc:
   OpenChannel(Message) -> Message                      [çift yönlü akış]
   OpenControlChannel(ControlMessage) -> ControlMessage  [çift yönlü akış]
   RegisterAgent(RegisterAgentTypeRequest)   -> RegisterAgentTypeResponse
   AddSubscription(AddSubscriptionRequest)   -> AddSubscriptionResponse
   RemoveSubscription(RemoveSubscriptionRequest) -> RemoveSubscriptionResponse
   GetSubscriptions(GetSubscriptionsRequest) -> GetSubscriptionsResponse
```

Mesaj tipleri: `RpcRequest`, `RpcResponse`, `Payload`, `AgentId`, `Subscription`,
`TypeSubscription`, `TypePrefixSubscription`, `SaveStateRequest/Response`,
`LoadStateRequest/Response`, `ControlMessage`, `Message`.

Üç şey buradan okunuyor:

1. **Veri düzlemi ile kontrol düzlemi ayrı.** `OpenChannel` mesajları taşıyor,
   `OpenControlChannel` ayrı bir kanal. İkisi de **çift yönlü akış** — yani worker
   ile host arasında kalıcı bir bağlantı var, istek-cevap değil.
2. **Abonelik runtime'a değil, host'a kayıtlı.** `AddSubscription` bir RPC.
   Yönlendirme kararı merkezde veriliyor.
3. **Durum protokole dahil.** `SaveState`/`LoadState` RPC olarak var — yani
   kalıcılık sonradan eklenmiş bir yardımcı değil, telin parçası.

`Payload` mesajı `data_type` + `data_content_type` + `data` taşıyor: **dilden
bağımsız.** Python↔.NET vaadinin (`05:1742`) teknik dayanağı bu.

> **`[ölçüldü]` Ama bu ortamda çalışmıyor:**
> ```
> >>> from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime
> ModuleNotFoundError: No module named 'grpc'
> ```
> `grpcio` kurulu değil; `autogen-ext[grpc]` ekstrası gerekiyor. Yani dağıtık
> runtime'ı **hiç koşturmadım** — yukarıdaki her şey `.proto` descriptor'ından
> okundu, davranış değil **arayüz** bilgisi.

## 2.3 MCP — Model Context Protocol

**`[ölçüldü]`** `autogen_ext.tools.mcp` üç taşıyıcıyı da destekliyor:

| Taşıyıcı | Sınıf |
|---|---|
| stdio (yerel süreç) | `StdioServerParams` + `StdioMcpToolAdapter` |
| SSE | `SseServerParams` + `SseMcpToolAdapter` |
| Streamable HTTP | `StreamableHttpServerParams` + `StreamableHttpMcpToolAdapter` |

Artı `McpWorkbench` (tool *kaynağı* olarak bütün sunucu) ve `mcp_server_tools`
(tek tek tool'a çevirme).

**Bizde:** `conversation.py` DeepWiki'yi `McpWorkbench` ile bağlıyor. Ve
`06 §1`'deki tuzak tam burada ısırıyor — yerel fonksiyonlarla MCP'yi aynı ajanda
birleştirmek için yerel olanları `StaticWorkbench`'e sarmak gerekiyor.

**Yaşam döngüsü notu:** `06 §12`'deki bağımlılık kırılması da MCP'den geldi —
`autogen-ext 0.7.5` `mcp>=1.11.0` diyor, **üst sınır yok**. Kurulu sürüm şu an
`1.29.0`; SDK 2.0 çıktığında `requirements.txt`'teki `mcp>=1.24,<2` pini olmasa
proje kurulamayacaktı.

## 2.4 OpenTelemetry GenAI semantic conventions

**`[ölçüldü]`** `autogen_core/_telemetry/_genai.py`, OTel'in **GenAI semantic
convention**'larını birebir kullanıyor:

```python
GENAI_SYSTEM_AUTOGEN = "autogen"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_SYSTEM         = "gen_ai.system"
GEN_AI_AGENT_ID / _NAME / _DESCRIPTION
GEN_AI_TOOL_NAME / _DESCRIPTION / _CALL_ID
```

Üretilen operasyonlar: **`create_agent`**, **`invoke_agent`**, **`execute_tool`**.

**Neden değerli:** span'lar AutoGen'e özel isimlerle değil, **satıcı-bağımsız bir
sözlükle** çıkıyor. Jaeger, Arize, Langfuse, hangi backend olursa olsun ajan
adımlarını tanıyor — özel adapter yazmadan.

**Bizde yok** ve kalan işlerin en kolayı: runtime'ı zaten kendimiz kuruyoruz
(`graph.py`), `SingleThreadedAgentRuntime(tracer_provider=...)` tek satır.

## 2.5 OpenAI Chat Completions — fiilî tel protokolü

**`[kaynak]`** `OpenAIChatCompletionClient(base_url=...)`, OpenAI'ın HTTP
sözleşmesini konuşan **her** sunucuya bağlanıyor. Bizim kendi endpoint'imiz de
böyle bağlı.

> **`[ölçüldü]` Ve tam burada ısırıyor (`06 §3`):**
> ```
> ValueError: model_info is required when model name is not a valid OpenAI model
> ```
> Model adı OpenAI kataloğunda yoksa yetenek kaydını **sen beyan ediyorsun**. Bu
> bir *beyan*, ölçüm değil: desteklenmeyen bir yeteneği iddia edersen hata
> başlangıçta değil, huninin **en sonunda** çıkıyor. `probe_llm.py`'yi bu yüzden
> yazdım — beyanı ölçümle değiştirmek için.

## 2.6 Bileşen yapılandırma — serileştirme "protokolü"

**`[ölçüldü]`** `ComponentModel`'ın JSON şeması:

| Alan | İçerik |
|---|---|
| `provider` | Import yolu (`"autogen_ext.models.openai.OpenAIChatCompletionClient"`) |
| `component_type` | `model` · `agent` · `tool` · `termination` · `token_provider` · `workbench` |
| `version` / `component_version` | Şema ve bileşen sürümü ayrı |
| `config` | Bileşene özel yapılandırma |
| `description` / `label` | İnsan için |

**İki sürüm alanı olması** düşünülmüş bir şey: şema değişince eski JSON'ı
okuyabilmek için. Ve `component_type` **kapalı bir liste değil** (`anyOf` içinde
serbest string de var) — kendi tipini ekleyebiliyorsun.

**`[kaynak]`** AutoGen'de "skill" diye bir soyutlama yok (`05:1888`, `02 §15`);
bildirimsel ajan tanımının karşılığı bu.

## 2.7 Konuşmadığı protokol: A2A

**`[ölçüldü]`** `autogen_core`, `autogen_agentchat`, `autogen_ext` içinde `a2a`
diye bir modül **yok**; `a2a-sdk` de kurulu değil. Buna karşılık kurulu
`google-adk 2.6.3` içinde **var**:

```
google/adk/a2a/converters/   event_converter, part_converter, request_converter, …
google/adk/a2a/executor/     a2a_agent_executor, task_result_aggregator, interceptors
google/adk/agents/remote_a2a_agent.py
```

**Ve fark burada keskinleşiyor:**

| | AutoGen | ADK |
|---|---|---|
| Ajanlar arası dağıtım | **Kendi gRPC/CloudEvents katmanı** | **A2A** (ajanlar arası standart) |
| Kimin ajanı | Hepsi seninki | Başkasının ajanı da olabilir |
| Sınır | Süreç/makine | **Organizasyon** |

AutoGen'in dağıtık runtime'ı **tek bir sistemi** makinelere yayıyor. A2A ise
**farklı sahiplerin** ajanlarını konuşturmak için. İkisi rakip değil, farklı
problem — ama "hangisi daha dağıtık" diye sorulunca cevap bu ayrımda.

## 2.8 Protokol özeti

| Katman | Protokol | Kanıt |
|---|---|---|
| Olay formatı | **CloudEvents v1** | `_topic.py` docstring · `io.cloudevents.v1` proto |
| Dağıtık taşıma | **gRPC + protobuf**, çift yönlü akış | `AgentRpc` servisi |
| Tool federasyonu | **MCP** (stdio · SSE · streamable HTTP) | `ext.tools.mcp` |
| Gözlemlenebilirlik | **OTel GenAI conventions** | `_genai.py`, `gen_ai.system="autogen"` |
| Model erişimi | **OpenAI Chat Completions** (fiilî) | `OpenAIChatCompletionClient` |
| Serileştirme | Kendi `ComponentModel` şeması | JSON schema |
| Ajanlar arası federasyon | **yok** | `a2a` modülü mevcut değil |

**Bir cümlede:** AutoGen dışa açılan her yerde **standart** kullanıyor
(CloudEvents, gRPC, MCP, OTel, OpenAI), yalnız kendi bileşen serileştirmesinde
kendi şemasını yazmış — ve ajanlar arası federasyonu hiç kapsamına almamış.

---

# BÖLÜM 3 — Runtime'ın içinde mesaj nasıl yol alıyor

Protokoller "neyle" sorusunun cevabıysa, bu bölüm "nasıl"ın. Ve burası
AgentChat'te öğrenip durursan **hiç görmeyeceğin** katman — bütün ilginç
davranış burada.

## 3.1 Dört adım: core'un "merhaba dünya"sı

**`[kaynak]`** `05:278` · Türkçesi [11](11-core-guide-turkce.md)'de. AgentChat'in
aksine burada **her şeyi kendin bağlıyorsun**:

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

| adım | ne oluyor | neden böyle |
|---|---|---|
| 1. Mesaj tipi | Düz bir `dataclass` | Ajanlar birbirini değil, **tipleri** tanıyor |
| 2. Handler | `@message_handler` + tip anotasyonu | Yönlendirme `if` bloğunda değil, tip sisteminde |
| 3. Kayıt | `register(runtime, tip, **fabrika**)` | Sınıfı değil fabrikayı veriyorsun; örnek ilk mesajda doğuyor |
| 4. Başlat + yayınla | `start()` → `publish_message()` → `stop_when_idle()` | Kime gittiği yayının konusu değil, **abonelik** meselesi |

Bu beş satır, `pipeline/fanin.py`'nin tamamının iskeleti.

## 3.2 `SingleThreadedAgentRuntime` — ve onu kendin vermenin bedeli

**`[kaynak]`** `05:490`. İki runtime tipi var ve **ikisi aynı API'yi sunuyor**:

| tip | ne zaman |
|---|---|
| `SingleThreadedAgentRuntime` | Tek süreç, tek dil — bizim kullandığımız |
| `GrpcWorkerAgentRuntimeHost` + worker'lar | Çok süreç, çok makine, Python↔.NET |

Ajan kodu değişmeden geçiş yapabilmen bu yüzden mümkün. §2.2'deki `.proto`
servisi de bu vaadin altındaki tel.

Ama `SingleThreadedAgentRuntime`'ı **kendin kurmanın** bir bedeli var, ve bu
sayfada en pahalı öğrendiğimiz şey:

> **`[ölçüldü]`** `InterventionHandler` takmanın tek yolu runtime'ı kendin
> kurmak. Ama o anda hata semantiği değişiyor: gömülü runtime `run_stream`'den
> **exception fırlatıyor**, dıştan verilen **hiç dönmüyor**. Duvar saati sınırı
> burada tedbir değil, **doğruluk şartı**. → [06 §7](06-autogen-incelikleri.md)

Ek olarak: runtime'ı sen verdiysen `start()`/`stop()` de sana ait. AgentChat
yalnız **kendi kurduğu** runtime'ı başlatıp durduruyor.

## 3.3 İki iletişim biçimi — ve aralarındaki fark adresleme değil

Kılavuz bunu tek cümleyle ayırıyor **`[kaynak]`** (`05:1274`):

> *"There are two types of communication in AutoGen core: **Direct Messaging** —
> sends a direct message to another agent · **Broadcast** — publishes a message
> to a topic."*

Üçüncü bir şey yok; handler'dan dönen `return` ayrı bir biçim değil,
**doğrudan mesajın cevabı**. Yayında `return` yazarsan hiçbir yere gitmiyor.

| biçim | nasıl | cevap alır mı |
|---|---|---|
| **Doğrudan** | `send_message(msg, AgentId(tip, anahtar))` | **Evet** — handler'ın dönüş değeri |
| **Yayın** | `publish_message(msg, TopicId(tip, kaynak))` | **Hayır** — her zaman `None` |

**Ve asıl mesele adresleme değil: ikisinin arıza davranışı farklı.** Bu projede
en pahalı öğrendiğimiz şey buydu, o yüzden aşağıda dört başlıkta açıyorum.

### A · Doğrudan mesajlaşma — ajanlar arası fonksiyon çağrısı

İki giriş noktası var: handler'ın **içinden** `self.send_message(...)`, dışarıdan
`runtime.send_message(...)`. İkisi de alıcının handler'ının **dönüş değerini**
veriyor; handler `None` dönerse `None` alıyorsun. Kılavuzun kendi benzetmesi:
*"You can think of this as a function call between agents."*

Kılavuzun örneği, **eşlenik ajan** deyimini gösteriyor `[kaynak]` (`05:1310`):

```python
class InnerAgent(RoutedAgent):
    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> Message:
        return Message(content=f"Hello from inner, {message.content}")   # ← cevap

class OuterAgent(RoutedAgent):
    def __init__(self, description: str, inner_agent_type: str):
        super().__init__(description)
        self.inner_agent_id = AgentId(inner_agent_type, self.id.key)      # ← aynı key

    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> None:
        response = await self.send_message(Message(...), self.inner_agent_id)
```

Dikkat edilecek satır `AgentId(inner_agent_type, self.id.key)`: **aynı anahtar,
farklı tip**. Yani "her dış ajanın kendi iç ajanı" ilişkisi hiçbir kayıt defteri
tutmadan kuruluyor — `(type, key)` adreslemesi bunu bedava veriyor. §3.6'daki
topic→anahtar eşlemesinin kardeşi.

**Hata semantiği** `[kaynak]` (`05:1289`):

> *"If the invoked agent raises an exception while the sender is awaiting, the
> exception will be propagated back to the sender."*

**Kılavuz ne zaman kullanılacağını da söylüyor** `[kaynak]` (`05:1352`):

> *"direct messaging is appropriate for scenarios when the sender and recipient
> are **tightly coupled** — they are created together and the sender is linked to
> a specific instance of the recipient."*

Verdiği somut örnek çok öğretici: bir ajanın tool çağrılarını
`autogen_core.tool_agent.ToolAgent`'a **doğrudan mesajla** yollayıp cevaplarla
bir **eylem-gözlem döngüsü** kurması. Yani AutoGen'in kendi tool yürütme
mekanizması bile doğrudan mesajlaşmayla yazılmış.

### B · Yayın — tek yönlü, ve dört sessiz kural

Yayın yaparken de iki giriş noktası var: `self.publish_message(...)` ve
`runtime.publish_message(...)`. İkincisi önemli — **hiçbir ajan örneği olmadan**
sisteme mesaj sokabiliyorsun; bizim `fanin.py`'de işi başlatan şey bu.

```python
class BroadcastingAgent(RoutedAgent):
    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> None:
        await self.publish_message(
            Message("..."),
            topic_id=TopicId(type="default", source=self.id.key),
        )
```

Kılavuzun dört ayrı yerde düştüğü notlar **`[kaynak]`**:

| kural | kaynak | sonucu |
|---|---|---|
| *"it will always return `None`"* — ama yine de `await` edilmeli, runtime teslimi zamanlasın diye | `05:1404` | Yayın **hiçbir zaman** veri döndürmez; `await` bir senkronizasyon, bir cevap değil |
| *"If a response is given to a published message, it will be thrown away."* | `05:1367` | Yayın handler'ından `return` yazan biri hiçbir uyarı almaz |
| *"If an agent publishes a message type for which it is subscribed it will not receive the message it published."* | `05:1372` | Kendi yayınını duymuyorsun — **sonsuz döngüyü önlemek için**, açıkça |
| *"If an agent raises an exception while handling a published message, this will be **logged** but will not be propagated back to the publishing agent."* | `05:1407` | Çöküş **kaybolmuyor, sadece kontrol akışına girmiyor** |

**Dördüncüsü bu belgedeki en ince ayrım.** "Yayında hata kaybolur" demek
eksikti: hata **`MessageHandlerExceptionEvent` olarak olay akışına düşüyor**.
Yani gözlemlenebilir ama **eyleme dönüştürülemez** — yayınlayan taraf bir şey
öğrenmiyor, sadece bir dinleyici öğrenebiliyor.

> **`[ölçüldü]` Bizim için doğrudan sonucu:** `observability.EventCapture` zaten
> `EVENT_LOGGER_NAME`'i dinliyor. Yani fan-out dallarındaki çöküşleri
> **saymamız mümkün** — kaybı önleyemeyiz ama sessiz bırakmak zorunda değiliz.
> Şu an `ClosureAgent` kuyruğuyla beklenen sonuç sayısını sayarak çözüyoruz
> (§3.7); olay tarafını da saymak ikinci bir emniyet olurdu.

### C · Abonelik iki yoldan kaydediliyor

Kolayca kaçan bir ayrıntı `[kaynak]` (`05:1430`):

```python
# Yol 1 — dekoratör: kayıtta aboneliği kendisi ekliyor
@type_subscription(topic_type="default")
class ReceivingAgent(RoutedAgent): ...

# Yol 2 — elle: kayıt ayrı, abonelik ayrı
await BroadcastingAgent.register(runtime, "broadcasting_agent", lambda: ...)
await runtime.add_subscription(TypeSubscription(topic_type="default",
                                                agent_type="broadcasting_agent"))
```

İkinci yol **çalışma zamanında** abonelik eklemeye izin veriyor — ve §2.2'de
gördüğümüz gRPC servisinde `AddSubscription`/`RemoveSubscription`'ın RPC olarak
durmasının sebebi bu.

`DefaultTopicId` ise kısayol: topic tipi `"default"`, **kaynak yayınlayan ajanın
anahtarı**. `@default_subscription` ile birlikte tek kapsamlı senaryoyu üç satıra
indiriyor.

### D · Hangisini ne zaman

| | doğrudan mesaj | yayın |
|---|---|---|
| **bağ** | Sıkı — gönderen alıcının **örneğini** biliyor | Gevşek — kimse kimseyi tanımıyor |
| **cevap** | Handler'ın dönüş değeri | **Yok**, her zaman `None` |
| **hata** | **Çağırana fırlatılıyor** | Loglanıyor, yayınlayana gitmiyor |
| **kaç alıcı** | Tam bir tane | Kaç abone varsa |
| **kılavuzun örneği** | `ToolAgent` eylem-gözlem döngüsü | Fan-out, olay dağıtımı |
| **bizde** | Yok | `fanin.py`'nin tamamı |

**Bu tablo bir tasarım hatasını da açıklıyor.** Fan-out dallarını yayınla
başlatıp sonuçları toplamaya çalışırken hata görünmüyorsa, bu bir bug değil —
**yanlış iletişim biçimi seçilmiş olabilir.** Üç analisti doğrudan mesajla
çağırsaydık her çöküş elimize gelirdi; ama o zaman da paralellik ve gevşek bağ
giderdi. Biz üçüncü yolu seçtik: **yayın + kuyruk + beklenen sayıyı saymak.**

## 3.4 Yönlendirme: önce tip, sonra `match`

**`[kaynak]`** `05:1268`. Tip yönlendirmesini biliyoruz. Bilmediğimiz şey şuydu:
**aynı tipteki mesajları farklı handler'lara ayırmak** için `match` parametresi
var.

```python
class RoutedBySenderAgent(RoutedAgent):
    @message_handler(match=lambda msg, ctx: msg.source.startswith("user1"))
    async def on_user1_message(self, message: TextMessage, ctx: MessageContext) -> None: ...

    @message_handler(match=lambda msg, ctx: msg.source.startswith("user2"))
    async def on_user2_message(self, message: TextMessage, ctx: MessageContext) -> None: ...
```

Üç incelik, üçü de kılavuzda yazılı ama kolayca kaçıyor:

1. `match`, tip yönlendirmesine **ikincil** — önce tip eşleşiyor, sonra koşul.
2. Koşullar **handler adlarının alfabetik sırasıyla** deneniyor. Yani sıralamayı
   fonksiyon adı belirliyor; dosyadaki yazım sırası değil.
3. **Hiçbir `match` tutmazsa mesaj sessizce işlenmiyor.** Kılavuzun kendi
   örneğinde dört mesajdan biri hiç ele alınmıyor ve çıktıda buna dair tek satır
   yok — sadece beklediğin satır eksik.

Üçüncüsü bu belgedeki örüntünün bir örneği daha: **arıza gürültülü değil.**
`ctx.sender` de aynı işi ajan kimliğiyle yapabiliyor.

## 3.5 Runtime'ı durdurmanın üç yolu, üç farklı anlamı

**`[kaynak]`** `05:1069`. `SingleThreadedAgentRuntime.start()` bir **arka plan
görevi** başlatıyor; durdurmanın üç ayrı yolu var ve karıştırmak pahalı:

| çağrı | ne yapar |
|---|---|
| `stop()` | Hemen döner — **devam eden mesaj işlemeyi iptal etmez** |
| `stop_when_idle()` | İşlenmemiş mesaj kalmayana ve hiçbir ajan çalışmayana kadar **bloklar** |
| `close()` | Runtime'ı kapatır, kaynakları bırakır |

`start()` tekrar çağrılabiliyor — arka plan görevi kaldığı yerden devam ediyor.
Batch senaryosu (bizim taramalar) için doğru olan `stop_when_idle()`.

> **`[ölçüldü]` Ama ona da güvenilmiyor.** Bir handler çökerse bariyer **erken
> açılıyor** ve tamamlanmış kardeş sonuçlar kayboluyor. Kural: `stop_when_idle()`
> döndü diye iş bitti sayma — **beklenen sonuç sayısını say**.
> → [06 §13](06-autogen-incelikleri.md)

## 3.6 Topic ve abonelik — kılavuzun en atlanan sayfası

**`[kaynak]`** `05:670`. Bir topic'in iki parçası var: **tip** (kategori) ve
**kaynak** (o kategorideki benzersiz kimlik). `TypeSubscription` bir topic
*tipini* bir ajan *tipine* bağlıyor. Asıl incelik ise şu:

> **Topic kaynağı, ajan anahtarına dönüşür.**

Ajan kimliği zaten bir `(type, key)` çifti (`05:601`) ve örnekler **lazy**
doğuyor. İkisi birleşince: `("gorev", "sirket-42")` topic'ine yayın yapınca
runtime `("analist", "sirket-42")` ajanını yaratıyor.

**Şirket başına izole ajan örneği bedava.** Çok kiracılı yapının hazır
mekanizması bu. Kılavuzun saydığı üç senaryo:

| senaryo | nasıl |
|---|---|
| Tek kiracı, tek topic | Hepsi aynı topic tipi, kaynak sabit `"default"` |
| Tek kiracı, çok topic | Ajan başına ayrı topic tipi, kaynak aynı |
| **Çok kiracı** | Kaynak veriye bağlı — kiracı/oturum/kayıt başına izolasyon |

Bizde `fanin.py` tek kaynak (`"default"`) kullanıyor, çünkü tek kullanıcılı bir
araç. **Hacim sorusu ("günde 200 şirket mi 5.000 mi") cevaplanınca ilk bakılacak
yer burası** — kod değişikliği değil, kaynak seçimi.

> **`[ölçüldü]` Pratik yan etki:** `ToolCallEvent`'in taşıdığı `agent_id`, ajan
> runtime yönetimindeki bir handler içindeyken dolu; çıplak `agent.run()`'da
> `None`. Aynı ajan bir takım içindeyken `TechnicalAnalyst_<uuid>`.

## 3.7 Toplama: bariyer yerine kuyruk

**`[kaynak]`** `05:3236` (*Concurrent Agents*). İşçiler sonucu **yayınlıyor**,
bir `ClosureAgent` kuyruğa boşaltıyor, sen kuyruktan okuyorsun:

```python
queue = asyncio.Queue[TaskResponse]()

async def collect_result(_agent: ClosureContext, message: TaskResponse, ctx: MessageContext) -> None:
    await queue.put(message)

await ClosureAgent.register_closure(
    runtime, "collector", collect_result,
    subscriptions=lambda: [TypeSubscription(topic_type=RESULTS, agent_type="collector")],
)
```

> **`[ölçüldü]` Bu birkaç satır, §6.2'deki kaybı çözen şey.** Ham hata altında
> `GraphFlow` 0–1 dal kurtarıp süre sınırını doldururken, bu yapı 2 dal
> kurtarıp ~3 ms'de dönüyor. Sebep basit: sonuç **üretildiği anda** yayınlanıyor
> ve kuyruk onu çoktan tutuyor. **Güvenilmeyecek bariyer yok, çünkü bariyer
> yok.** → `pipeline/fanin.py`, `pipeline/compare_fanin.py`

## 3.8 Müdahale kapısı

**`[kaynak]`** `05:6534`, `05:6638`. `InterventionHandler` mesaj hattına
**teslimden önce** oturuyor:

```python
class TerminationHandler(DefaultInterventionHandler):
    async def on_publish(self, message, *, message_context):
        if isinstance(message, Termination):
            self._termination_value = message
        return message

runtime = SingleThreadedAgentRuntime(intervention_handlers=[TerminationHandler()])
```

Asıl kullanımı `DropMessage` döndürerek bir tool çağrısını **engellemek**. Ve
bunun neden değerli olduğu tek cümlede:

> **Onay kapısı ajanın uyum göstermeyi seçmesine değil, runtime'a dayanır.**
> Model ne kadar ikna edilirse edilsin, mesaj hattı onun altında.

**Bizde:** bağlı ve testli, ama **gözlemci modda** — buradaki tool'ların hepsi
salt-okunur, her çağrıda onay sormak tören olurdu. İlk mutasyon yapan tool
geldiğinde kapı hazır. → `pipeline/observability.py`

> **`[ölçüldü]` Tuzak:** Handler `None` döndürürse runtime uyarı basıyor —
> mesajı ya da `DropMessage`'ı **açıkça** döndürmek gerekiyor. Bizde `on_publish`
> ve `on_response` bu yüzden açıkça geçiş fonksiyonu olarak yazılı.

## 3.9 Aynı işi diğerleri nasıl yapıyor

**Bu bölüm asıl farkın durduğu yer.** Yukarıdaki mekanizmaların hepsi tek bir
soruya cevap veriyor: *bir ajan başka bir ajana nasıl iş verir?* Ve her
framework buna başka bir cevap vermiş.

Aşağıdaki satırların kaynağı: ADK **`[ölçüldü]`** (kurulu paket, `grep`),
diğerleri **`[kaynak]`** (GitHub'da gerçek kod — repo yolları verili).
Davranışa dair yorumlar **`[teyitsiz]`**, çünkü hiçbirini koşturmadım.

### AutoGen — mesaj otobüsü

```python
await self.send_message(msg, AgentId("analyst", "company-42"))   # adresli
await self.publish_message(msg, TopicId("task", "company-42"))   # yayın
```

Ajan ajanı **tanımıyor**; tipi ve topic'i tanıyor. Hata doğrudan mesajda geri
taşınıyor, yayında taşınmıyor (§3.3).

### LangGraph — paylaşılan state + `goto`

**`[kaynak]`** `langchain-ai/langgraph` · `libs/langgraph/tests/test_pregel_async.py`,
`langchain-ai/open_deep_research` · `src/legacy/multi_agent.py`:

```python
return Command(goto="tool_node", update={"messages": response})
return Command(goto=[Send("research_team", {"section": s}) for s in sections])
```

Düğüm bir **mesaj göndermiyor** — bir sonraki düğümün *adını* döndürüyor ve
paylaşılan state'i güncelliyor. `Send(...)` ile dinamik fan-out yapılabiliyor,
her dala ayrı yük verilerek.

**Fark:** AutoGen'de taşınan şey **mesaj**, LangGraph'ta **state güncellemesi**.
İletişim kanalı yok; ortak bir sözlük var ve sıradaki düğüm onu okuyor.
`[teyitsiz]` Bunun avantajı checkpoint alınabilir olması, bedeli her düğümün
aynı state şemasına bağlı kalması.

### OpenAI Agents SDK — devir, ve devir bir tool

**`[kaynak]`** `openai/openai-agents-python` · `tests/test_handoff_history_duplication.py`,
`tests/models/test_openai_responses_converter.py`:

```python
triage = Agent(name="triage", model=triage_model, handoffs=[delegate])
transfer_handoff = handoff(Agent(name="specialist"), tool_name_override="lookup_account")
Converter.convert_tools(tools, handoffs=[transfer_handoff])
```

Son satır ilginç olanı: **handoff'lar tool'larla aynı dönüştürücüden geçiyor.**
Yani modelin gözünde devir bir tool çağrısından ibaret — ayrı bir kanal değil.
AutoGen'in `Swarm`'ı ile aynı fikir, ve `poc/kiyas.py`'de en pahalı çıkmasının
sebebi de bu **`[ölçüldü]`**: her devir bir tool çağrısı artı iş üretmeyen bir
LLM turu.

Not: reponun test dosyalarından birinin adı `test_handoff_history_duplication` —
devirde konuşma geçmişinin ne kadarının taşınacağı orada da çözülmesi gereken
bir problem.

### CrewAI — task zinciri, ajan değil task birincil

**`[kaynak]`** `crewAIInc/crewAI` · `lib/crewai/tests/test_crew.py`,
`crewAIInc/crewAI-examples`:

```python
Task(description="...", expected_output="x", agent=researcher)
Task(description="async1", expected_output="x", agent=researcher, async_execution=True)
```

Birincil nesne **ajan değil task**. Ajan bir task'a *atanıyor*, ve sıralama
`Crew(agents=[...], tasks=[...])` içindeki task listesinden geliyor. Ajanlar
arası iletişim diye ayrı bir mekanizma yok — bir task'ın çıktısı bir sonrakinin
bağlamına giriyor.

**Fark:** AutoGen'de birim **ajan**, CrewAI'da **task**. Bu yüzden CrewAI'da
"kim konuşacak" sorusu hiç doğmuyor; sıra zaten yazılı.

### Google ADK — ajan ağacı + `transfer_to_agent` tool'u

**`[ölçüldü]`** Kurulu `google-adk 2.6.3`:

```
google/adk/agents/base_agent.py:146   sub_agents: list[BaseAgent] = Field(default_factory=list)
google/adk/flows/llm_flows/agent_transfer.py:29
    from ...tools.transfer_to_agent_tool import TransferToAgentTool
```

Ajanlar bir **ağaç** oluşturuyor (`sub_agents`), ve devir yine bir tool:
`transfer_to_agent`. Artı `sequential_agent`, `parallel_agent`, `loop_agent` gibi
**workflow ajanları** — akış kontrolü ajan tipine gömülü.

**Fark:** AutoGen'de hiyerarşi yok, topoloji abonelikten doğuyor. ADK'da
hiyerarşi **veri yapısının kendisi**. `SocietyOfMindAgent` (§1.3) bunun
AutoGen'deki en yakın karşılığı ama bir ağaç değil, bir sarmalayıcı.

### DeepAgents — alt ajan bir tool çağrısı, bağlam izole

**`[kaynak]`** `langchain-ai/deepagents` ·
`libs/deepagents/tests/unit_tests/test_subagents.py`:

```python
agent = create_deep_agent(
    model=...,
    subagents=[{"name": "remote-researcher", "description": "...", "system_prompt": "..."}],
)
assert "task" in agent_tools
assert "start_async_task" in agent_tools
assert "check_async_task" in agent_tools
```

Alt ajan çağırmanın yolu **`task` adında bir tool**. Senkron çağrının yanında
`start_async_task` / `check_async_task` çifti var — yani uzun süren alt işler
başlatılıp sonra yoklanıyor.

**Fark ve neden ilginç:** burada ne mesaj otobüsü var, ne graf, ne task listesi.
Ana ajan bir tool çağırıyor, geriye **bir metin** alıyor. `[teyitsiz]` Tasarımın
amacı bağlam izolasyonu görünüyor — alt ajanın bütün ara adımları ana ajanın
bağlamına girmiyor. Bu, AutoGen'in ortak-thread modelinin **tam tersi** tercih:
AutoGen'de herkes her şeyi görüyor ve `poc/kiyas.py`'de ölçtüğümüz bağlam şişmesi
oradan geliyor.

### Tek tabloda

| | birincil birim | ajan → ajan nasıl | akış kararı | hata nereye gider |
|---|---|---|---|---|
| **AutoGen** | ajan | `send_message` / `publish_message` | konuşmacı seçimi | doğrudan: çağırana · yayın: **hiçbir yere** |
| **LangGraph** | düğüm | `Command(goto=...)` + paylaşılan state | kenarlar / `goto` | `[teyitsiz]` graf yürütücüsüne |
| **Agents SDK** | ajan | `handoffs=[...]` → tool çağrısı | ajanın kendisi | `[teyitsiz]` çalıştıran döngüye |
| **CrewAI** | **task** | task çıktısı → sonraki task | task listesi | `[teyitsiz]` crew döngüsüne |
| **ADK** | ajan ağacı | `transfer_to_agent` tool'u | workflow ajanı / LLM | `[teyitsiz]` graf motoruna |
| **DeepAgents** | ajan | `task` tool'u (alt ajan) | ana ajanın kendisi | `[teyitsiz]` tool sonucuna |

**Tablodan çıkan asıl ders:** dördü (Agents SDK, ADK, DeepAgents ve AutoGen'in
`Swarm`'ı) ajanlar arası geçişi **tool çağrısına** indirgemiş. Yani modele
"başka bir ajana geç" demenin fiilî standardı bu oldu. AutoGen'i ayıran şey
tool'a *ek olarak* bir **mesaj otobüsü** taşıması — ve bu otobüsün altında
CloudEvents + gRPC olması (§2).

Bedeli de tabloda görünüyor: **AutoGen tek başına "hata nereye gider" sorusuna
iki farklı cevap veren framework.** Diğerlerinde tek bir yol var, o yüzden
sessiz kayıp da yok.

## 3.10 Beş mekanizma, özet

Yukarıdakileri tek tabloya indirirsek:

**① Adresleme — ajan bir nesne değil, bir anahtar.** `(type, key)`. Runtime ajanı
ilk mesajda yaratıyor. Ve topic kaynağı doğrudan ajan anahtarına dönüştüğü için
**şirket başına izole örnek bedava** (`05:670`).

**② Yönlendirme — tip anotasyonundan.** `@message_handler`, imzadaki tipi
`get_type_hints()` ile okuyor. Karar `if` bloklarında değil, tip sisteminde.
Bedeli `06 §11`'deki `NameError` tuzağı: anotasyonlar modül genelinde çözülüyor.

**③ Orkestrasyon — "kim konuşacak" bir strateji nesnesi.** RoundRobin'de sıra,
Selector'da model (ya da `selector_func` ile senin Python'un), Swarm'da ajanın
kendisi, GraphFlow'da önceden çizilmiş graf. **Aynı ajanlarla desen
değiştirilebiliyor** — ve `poc/kiyas.py`'de ölçtüğümüz gibi bu **%63,7 token
farkı** demek.

**④ Gözlem — olay akışı, dönüş değeri değil.** İki logger, altı olay tipi, ve
`InterventionHandler` ile her mesaj geçişinde durdurma hakkı. **Maliyet
ölçümümüzün tamamı buradan geliyor.** Bedeli `06 §4` ve `§6`: yanlış olayı
dinlersen sıfır görürsün, ve `logging.Handler` içindeki hata yutulur.

**⑤ Durum — `model_context`'in içeriği.** `save_state()` bağlamı kaydediyor;
bağlam vermediysen kaydedilecek bir şey de yok (`06 §2`). Kalıcılık ayrı bir
sistem değil, bellek nesnesinin serileşmesi.

---

# BÖLÜM 4 — Bileşenler: sayfa sayfa, ve diğerlerindeki karşılığı

*Core kılavuzunun **Components Guide** başlığı altındaki beş sayfa. Her biri:
kılavuz ne diyor · bizde nerede · diğerlerinde ne var.*

Bu bölümden itibaren karşılaştırma listesine **Agno**'yu da ekliyorum
(`agno-agi/agno`). Kurulu değil; API yüzeyi reponun kendi cookbook ve kaynak
dosyalarından okundu — `[kaynak]`, davranış `[teyitsiz]`.

## 4.1 Model istemcileri · `05:1980`

**Ne diyor:** `ChatCompletionClient` bir **arayüz**. `create()`,
`create_stream()`, `total_usage()`, `count_tokens()`, `remaining_tokens()`.
Somut uygulamalar `autogen_ext.models` altında: `openai`, `anthropic`, `azure`,
`ollama`, `llama_cpp`, `semantic_kernel`, artı `replay` (deterministik test) ve
`cache`.

**Bizde:** `engine.build_client` bu arayüzü sarıyor. `ReplayChatCompletionClient`
kuru modun tamamı — testler modelsiz koşuyor.

> **`[ölçüldü]` En pahalı tuzak:** OpenAI kataloğunda olmayan bir model adı için
> `model_info` **zorunlu** ve bir *beyandır*, ölçüm değil.
> → [06 §3](06-autogen-incelikleri.md), `pipeline/probe_llm.py`

**Diğerlerinde:**

| | model katmanı |
|---|---|
| **AutoGen** | Arayüz core'da, uygulama ext'te — **soyutlama önce** |
| **Agno** `[kaynak]` | `agno.models.openai.OpenAIChat` / `OpenAIResponses`, `anthropic.Claude`, `google.Gemini`, `ollama.Ollama` — sağlayıcı başına sınıf |
| **LangGraph** `[teyitsiz]` | Kendi katmanı yok; LangChain'in `BaseChatModel`'ini kullanıyor |
| **ADK** `[ölçüldü]` | `google/adk/models/` alt paketi |
| **CrewAI / Agents SDK** `[teyitsiz]` | Sağlayıcı ayarı ajanın parametresi |

**Agno'nun burada tek başına yaptığı bir şey var** `[kaynak]`
(`cookbook/03_teams/17_fallback_models/`):

```python
Team(
    model=OpenAIChat(id="gpt-4o"),
    fallback_config=FallbackConfig(
        on_rate_limit=[OpenAIChat(id="gpt-4o-mini"), Claude(id="claude-sonnet-4-20250514")],
        on_context_overflow=[Claude(id="claude-sonnet-4-20250514")],
        on_error=[Claude(id="claude-sonnet-4-20250514")],
    ),
    members=[researcher, writer],
)
```

**Hata tipine göre farklı yedek model.** AutoGen'de bunun karşılığı yok — bizde
`engine.ResilientClient` hatayı mesaja çeviriyor ama yedek modele düşmüyor.
Bu, alınabilecek bir fikir.

## 4.2 Model bağlamı · `05:2341`

**Ne diyor:** Ajanın belleği — modele gönderilecek mesaj listesini yöneten nesne.
Dört strateji: `Unbounded`, `Buffered(buffer_size=N)`, `HeadAndTail`,
`TokenLimited`.

**Bizde:** `conversation.py` → `BufferedChatCompletionContext(24)`.

> **İnce nokta:** `buffer_size` **mesaj** sayıyor, token değil. Maliyeti
> sınırlayan şey bu değil — onun için `TokenLimited...` var. Ve `save_state()`'in
> kaydettiği şey bağlamın tuttuğu: bağlam yoksa kaydedilecek sohbet de yok.

**Diğerlerinde:** Burası çoğu framework'ün **açıkta bıraktığı** yer.

| | bağlam yönetimi |
|---|---|
| **AutoGen** | Takılabilir **strateji nesnesi**, dört hazır sınıf |
| **DeepAgents** `[kaynak]` | Farklı bir cevap: alt ajan `task` tool'uyla çağrılıyor, ara adımlar ana bağlama **hiç girmiyor** — bağlamı kısaltmak yerine **bölüyor** |
| **Agno** `[kaynak]` | `add_datetime_to_context`, `FallbackConfig(on_context_overflow=...)` — taşmayı **model değiştirerek** karşılıyor |
| **LangGraph** `[teyitsiz]` | State'i sen tasarlıyorsun; kısaltma da senin işin |
| **CrewAI** `[teyitsiz]` | Katmanlı bellek (kısa/uzun/varlık) |

**Üç farklı felsefe:** AutoGen kısaltıyor, DeepAgents bölüyor, Agno taşınca model
değiştiriyor. Bizim huni için doğru olan ilk ikisinin karması — analist
bağlamları zaten izole, sohbet bağlamı kısalıyor.

## 4.3 Tool'lar · `05:2473`

**Ne diyor:** `FunctionTool` bir Python fonksiyonunu tool'a çeviriyor ve şemayı
**imzadan ve docstring'den** üretiyor.

```python
async def get_stock_price(ticker: str, date: Annotated[str, "YYYY/MM/DD"]) -> float:
    """Get the stock price."""
    return random.uniform(10, 200)
```

> **Bunun sonucu şu:** docstring dokümantasyon değil, **arayüz**. Modelin tool'u
> doğru çağırıp çağırmayacağını docstring belirliyor. Bizim `conversation.py`'deki
> yedi tool'un docstring'leri bu yüzden özenli yazılı.

**Diğerlerinde:** **Yakınsamanın en net olduğu yer burası.** Hepsi aynı şeyi
yapıyor: fonksiyon + tip ipuçları + docstring → JSON şema. `[teyitsiz]`

Fark tool *tanımında* değil, **tool kütüphanesinde**:

| | hazır tool'lar |
|---|---|
| **AutoGen** `[ölçüldü]` | `mcp`, `http`, `code_execution`, `graphrag`, `azure`, `langchain`, `semantic_kernel` — çoğu **adapter**, hazır entegrasyon az |
| **Agno** `[kaynak]` | `duckduckgo`, `hackernews`, `yfinance`, `serpapi`, `newspaper4k`, `websearch`… — **hazır entegrasyon çok** |
| **ADK** `[ölçüldü]` | `tools/` + `mcp_tool/` + `toolbox_toolset` |

**Bizim için somut sonucu:** Agno'da `HackerNewsTools` ve `YFinanceTools` kutudan
çıkıyor; biz `collectors/hackernews.py`'yi elle yazdık. Ama elle yazdığımız şey
**politika kapısından geçiyor** (`policy.py`) ve kaynak URL'si taşıyor — hazır
tool'da o garanti yok. Bu bir kayıp değil, bilinçli bir takas.

## 4.4 Workbench ve MCP · `05:2841`

**Ne diyor:** Workbench bir **tool kaynağı** — tek tek tool yerine "bana hangi
tool'ların olduğunu söyle" diyebildiğin nesne. `list_tools()` / `call_tool()`.
Uzak MCP sunucusu tam olarak bu: **ajan yazılırken var olmayan** tool'ları
listeleyebiliyor.

**Bizde:** `conversation.py` → `[StaticWorkbench(yerel)] + [McpWorkbench(deepwiki)]`.

> **`[ölçüldü]` Tuzak:** `tools=` ile `workbench=` aynı ajana verilemiyor.
> → [06 §1](06-autogen-incelikleri.md)

**Diğerlerinde:** MCP artık hepsinde var (§7). Ayıran şey **soyutlamanın
seviyesi**: AutoGen'in `Workbench`'i tool listesini *çalışma zamanında*
soruyor — yani sunucu tool eklerse ajan yeniden yazılmadan görüyor. `[teyitsiz]`
Diğerlerinde MCP tool'ları çoğunlukla başlangıçta bir listeye çevriliyor.

Küçük bir ayrım gibi duruyor ama uzun ömürlü bir sistemde fark yaratan yer bu.

## 4.5 Komut satırı kod yürütücüleri · `05:3054`

**Ne diyor:** `LocalCommandLineCodeExecutor` ve `DockerCommandLineCodeExecutor`.
Kılavuz yerel yürütücü için açık uyarı veriyor: modelin ürettiği kodu izolesiz
çalıştırmak risklidir, Docker önerilir.

**Bizde yok** — VC hattında kod yürütülmüyor.

**Diğerlerinde:** **AutoGen'in hâlâ açık ara önde olduğu tek eksen.** Kod
yürütme burada **birinci sınıf**: ayrı bir alt paket (`code_executors`), ayrı bir
ajan (`CodeExecutorAgent`), ve onay kapısı (`approval_func`, §1.3).

`[teyitsiz]` Diğerlerinde kod çalıştırmak **bir tool**: LangGraph'ta manuel,
CrewAI ve Agents SDK'da tool, ADK'da `code_executors` alt paketi var ama ona
adanmış bir ajan tipi yok **`[ölçüldü]`**. DeepAgents'ın bir uyarlamasında
`include_execute=True` biçiminde bir **yetenek bayrağı** görünüyor `[kaynak]`
(`vstorm-co/pydantic-deepagents`) — yani orada da bir anahtar, mimarinin merkezi
değil. Agno için bu ekseni doğrulamadım `[teyitsiz]`.

Sebebi tarihsel: AutoGen'in kurucu makalesinin konusu **kod yazma/çalıştırma
döngüsüydü**. Framework o problemin etrafında büyümüş.

---

# BÖLÜM 5 — Dokuz desen: sayfa sayfa, ve diğerlerindeki karşılığı

*Core kılavuzunun **Multi-Agent Design Patterns** başlığı altındaki dokuz sayfa.
Kılavuzun kendi ayrımı: akış **önceden mi çizili**, yoksa **konuşmadan mı
doğuyor**.*

## 5.0 Intro · `05:3209`

Kısa bir giriş sayfası, ama bir şeyi netleştiriyor: bunlar **kütüphane değil,
desen**. Yani `autogen_core`'la yeniden kurabileceğin şeyler; hazır sınıf olarak
gelmiyorlar.

**Bu, belgenin tezinin kaynağı:** desenler taşınabilir, bağımlılık taşınabilir
değil. AutoGen kapansa bile bu dokuz sayfa değerini koruyor.

## 5.1 Concurrent Agents · `05:3236`

**Ne diyor:** Üç varyant — tek mesaj/çok işleyici, çok mesaj/çok işleyici
(`@type_subscription`), doğrudan mesajlaşma. Ve asıl kritik parça **sonuç
toplama**: işçiler yayınlıyor, `ClosureAgent` kuyruğa boşaltıyor (§3.7).

**Bizde:** `pipeline/fanin.py` — birebir bu. **Bu projenin en çok işine yarayan
desen.**

> **`[ölçüldü]`** Ham hata altında `GraphFlow` 0–1 dal kurtarıp süre sınırını
> dolduruyor; bu yapı 2 dal kurtarıp ~3 ms'de dönüyor.

**Diğerlerinde:**

| | paralel dal | toplama |
|---|---|---|
| **AutoGen (core)** | Aynı topic'e abonelik | **Kuyruk** — bariyer yok |
| **AutoGen (`GraphFlow`)** | Graf fan-out | `activation_condition="all"` bariyeri |
| **LangGraph** `[kaynak]` | `Command(goto=[Send("node", payload), ...])` | Paylaşılan state'te birleşme |
| **Agno** `[kaynak]` | `agno.workflow.parallel.Parallel` — **ayrı bir birincil sınıf** | `merge_parallel_session_states` |
| **ADK** `[ölçüldü]` | `agents/parallel_agent.py` — **ajan tipi olarak** | Oturum state'i |
| **CrewAI** `[kaynak]` | `Task(..., async_execution=True)` | Task çıktısı zinciri |
| **Agents SDK / DeepAgents** `[teyitsiz]` | Yok / `start_async_task` + `check_async_task` | Tool sonucu |

**İki gözlem.** Birincisi: **paralellik üç farklı yere konmuş** — AutoGen'de
aboneliğe, Agno ve ADK'da bir *sınıfa/ajan tipine*, CrewAI'da task *bayrağına*.
İkincisi ve daha önemlisi: **hiçbiri "bir dal çökerse ne olur" sorusunu belgesinde
öne çıkarmıyor** — biz onu ölçerek bulduk, ve AutoGen'in kendi iki deseni bile
farklı cevap veriyor.

## 5.2 Sequential Workflow · `05:3504`

**Ne diyor:** Her ajan bir öncekinin çıktısını alıyor. `@type_subscription` ile
her adım kendi topic tipini dinliyor. Kılavuzun örneği: kavram çıkar → yaz →
biçimlendir.

**Bizde:** Katman mimarimiz bu — ama **ajanlarla değil kodla** kurulu, çünkü ilk
iki katmanda LLM yok. Huninin ucuz ucunda ajan çalıştırmak para yakmak olurdu.

**Diğerlerinde:** Herkeste var, ama **kimin birinci sınıf vatandaş olduğu**
değişiyor:

- **Agno** `[kaynak]`: `Workflow(steps=[Step(...), Steps(...)])` — sıralı akış
  **ana soyutlama**. Yanında `Condition`, `Loop`, `Router` ve **CEL ifadeleri**
  (`validate_cel_expression`) var.
- **ADK** `[ölçüldü]`: `agents/sequential_agent.py` — sıralılık bir **ajan tipi**.
- **CrewAI** `[kaynak]`: `Crew(tasks=[...])` zaten sıralı; varsayılan bu.
- **LangGraph** `[teyitsiz]`: kenar çizerek.
- **AutoGen**: ya `RoundRobinGroupChat` ya `GraphFlow` — **ikisi de bu iş için
  fazla ağır.** Sıralı akış AutoGen'in en zayıf temsil ettiği desen.

## 5.3 Group Chat · `05:3772`

**Ne diyor:** Ortak thread + bir **konuşmacı seçici**. Mesaj tipleri
`GroupChatMessage` (içerik) ve `RequestToSpeak` (sıra sende).

> **Kılavuzun kendi cümlesi:** *"not meant to be used in real applications… a
> starting point."* Üretim sürümü AgentChat'in `SelectorGroupChat`'i.

**Bizde yok** — ama `answers.py`'deki deterministik yönlendirme mantığı
`selector_func` fikrinin LLM'siz hâli.

**Diğerlerinde:** **Bu, AutoGen'in en ayırt edici deseni.** Ortak thread + takılıp
çıkarılabilir seçici kombinasyonu başka yerde nadir.

- **Agno** `[kaynak]`: en yakın karşılık — `Team(members=[...], mode=TeamMode.broadcast)`,
  artı `show_members_responses`. Yani takım lideri + üyeler modeli var.
- **CrewAI** `[teyitsiz]`: hiyerarşik modda bir manager delege ediyor — seçici
  sabit.
- **LangGraph** `[teyitsiz]`: supervisor deseni elle kuruluyor, kütüphane değil.
- **ADK / Agents SDK / DeepAgents**: karşılığı yok; devir ya da ağaç var.

**AutoGen'in farkı:** seçici bir **strateji nesnesi** — `selector_func` ile modeli
tamamen devre dışı bırakabiliyorsun. `[ölçüldü]` Ve bu en ucuz yol: 204 token,
Swarm'ın 334'üne karşı.

## 5.4 Handoffs · `05:4349`

**Ne diyor:** OpenAI'ın Swarm desenine dayanıyor. Ajan bir **devir tool'u**
çağırarak konuşmayı başkasına veriyor. Örnek: triyaj → sorun/onarım → insan.

**Bizde yok:** huni tek yönlü, ve `[ölçüldü]` **en pahalı desen** — 334 token.
Her devir bir tool çağrısı artı iş üretmeyen bir LLM turu.

**Diğerlerinde:** §3.9'da ayrıntısı var. Kısası: **devir fiilî standart oldu.**
Agents SDK'nın *tek* modeli bu, ADK'da `transfer_to_agent`, DeepAgents'ta `task`
tool'u.

> **Buradan çıkan cümle önemli:** Agents SDK'nın tek modeli olan handoff,
> AutoGen'in ölçülmüş **en pahalı** desenidir. Framework seçmek çoğu zaman
> "hangi yönlendirme maliyetini ödemeye razısın" sorusunu cevaplamak demek.

`[ölçüldü]` Küçük ama pahalı bir ayrıntı: `Handoff` tool adı küçük harfe düşüyor
(`transfer_to_veriuzmani`); elle yazarsan eşleşmiyor.

## 5.5 Mixture of Agents · `05:4989`

**Ne diyor:** Katmanlı işçi ajanlar — her katmanın çıktıları birleştirilip
sonrakine veriliyor, sonunda bir orkestratör topluyor.

**Bizde yok, ve almama sebebim bir bulgu:**

> **`[ölçüldü]`** Orkestratör `asyncio.gather(...)` ile topluyor — yani POC'ta
> sessiz kardeş kaybının kaynağı olan yapıyla. **Resmî desenler birbiriyle
> çelişiyor:** Concurrent Agents kuyrukla topluyor, Mixture of Agents `gather`
> ile. Tek bir kütüphane hatasından daha güçlü bir bulgu — kılavuzun kendisi iki
> farklı arıza davranışı öneriyor.

**Diğerlerinde:** Doğrudan karşılığı olan yok. En yakını Agno'nun `Parallel` +
`Steps` bileşimi `[kaynak]` ve LangGraph'ın `Send` fan-out'u `[kaynak]` — ikisi
de aynı fikri kuruyor ama "katman" diye bir soyutlama sunmuyorlar.

## 5.6 Multi-Agent Debate · `05:5358`

**Ne diyor:** **Seyrek topoloji** — her çözücü ajan yalnız birkaç komşusuna bağlı
(dört ajan bir kare, her biri iki komşuya). Birkaç tur boyunca ajanlar
komşularının cevaplarını görüp kendi cevaplarını düzeltiyor; sonunda bir
toplayıcı **çoğunluk oyu** alıyor.

**Bizde yok, ve gerekçesi ilkesel:** skorlamada **sabit rubrik** istiyoruz, oy
değil. Oylama, aynı şirkete iki koşuda iki puan verme riskini geri getirir —
"adil kıyas" ilkesine aykırı. Üstelik N ajan × R tur maliyeti ciddi.

**Diğerlerinde:** İlginç olan, bunun **Agno'da bir cookbook başlığı olması**
`[kaynak]`: `cookbook/03_teams/02_modes/broadcast/04_structured_debate.py` —
`TeamMode.broadcast` ile Proponent/Opponent, lider sentezliyor.

Fark: AutoGen'de münazara **seyrek topoloji + oy sayımı** (bir yapı);
Agno'da **yayın modu + lider sentezi** (bir takım ayarı). İkincisi daha ucuz,
birincisi daha ilkeli. `[teyitsiz]`

## 5.7 Reflection · `05:5822`

**Ne diyor:** Üretici + eleştirmen döngüsü. `CoderAgent` kod yazıyor,
`ReviewerAgent` bir `CodeReviewResult` döndürüyor, onaylanana kadar dönüyor.

**Bizde kısmen var:** `RiskAuditor` **tek turluk** bir reflection — üç analizi
çapraz denetliyor.

> **Döngüye çevirmedim, ve sebebi kılavuzun kendisinde:** durma ölçütü
> önerilmiyor. Döngü kurarsan faturayı sınırlayan tek şey kalmıyor. Bizde her
> takımda `MaxMessageTermination` sigortası var — reflection döngüsünde o sigorta
> "kaç tur kaliteli oldu" sorusunu cevaplamıyor, sadece kesiyor.

**Diğerlerinde:**

- **Agno** `[kaynak]`: `agno.workflow.loop.Loop` + CEL ile bitiş koşulu
  (`evaluate_cel_loop_end_condition`) — **durma ölçütü birinci sınıf**.
- **ADK** `[ölçüldü]`: `agents/loop_agent.py`.
- **LangGraph** `[teyitsiz]`: döngü kenarı + koşul.
- **CrewAI / Agents SDK** `[teyitsiz]`: elle.

**Burada AutoGen geride.** Reflection en yaygın ihtiyaçlardan biri ve AutoGen
onu bir *desen anlatısı* olarak bırakmış; Agno ve ADK **çalıştırılabilir bir
birim** yapmış. Bizim tek turda kalmamızın sebebi de kısmen bu.

## 5.8 Code Execution · `05:6188`

**Ne diyor:** Group chat + kod yürütücü — model kod yazıyor, bir ajan
çalıştırıyor, sonuç geri dönüyor. AutoGen'in kurucu makalesinin konusu buydu.

**Bizde uygulanamaz.**

**Diğerlerinde:** §4.5'te. Tek cümlede: **bu, AutoGen'in hâlâ en güçlü olduğu
yer** — desen + ajan tipi + yürütücü paketi + onay kapısı, dördü birden.

## 5.9 Dokuz desenin özeti

| desen | bizde | AutoGen'in gücü | en iyi alternatif |
|---|---|---|---|
| Concurrent Agents | **`fanin.py`** | Yüksek (core pub/sub) | Agno `Parallel` |
| Sequential Workflow | kodla | **Düşük** — fazla ağır | Agno `Workflow`, ADK `sequential_agent` |
| Group Chat | — | **En ayırt edici** | Agno `Team` |
| Handoffs | — | Orta, **en pahalı** | Agents SDK (tek modeli) |
| Mixture of Agents | — | Düşük (`gather` tuzağı) | — |
| Multi-Agent Debate | — | İlkeli ama pahalı | Agno `broadcast` |
| Reflection | **`RiskAuditor`** (tek tur) | **Düşük** — durma ölçütü yok | Agno `Loop` + CEL |
| Code Execution | — | **En yüksek** | — |

**Tablodan çıkan dürüst sonuç:** AutoGen dokuz desenin **üçünde** açık ara güçlü
(Concurrent, Group Chat, Code Execution), üçünde ortalama, **üçünde geride**
(Sequential, Reflection, Mixture). Ve geride kaldığı üçü, Agno'nun `Workflow`
ilkelleriyle birinci sınıf yaptığı üçü.

Bu bizim seçimimizi değiştirmiyor — kullandığımız iki desen (Concurrent ve
kısmen Reflection) tam da güçlü olduğu yerde. Ama "AutoGen her desende iyidir"
demek yanlış olurdu.

---

# BÖLÜM 6 — Farklı düşündüğü yer

## 6.1 Metafor: her framework'ün bir dünya görüşü var

**`[kaynak]`** AutoGen · **`[teyitsiz]`** diğerleri.

| Framework | Metafor | Temel primitif |
|---|---|---|
| **AutoGen** | **Konuşma** | Ortak mesaj thread'i; ajanlar "conversable" |
| LangGraph | Graf | Düğüm + kenar + state; checkpointer |
| CrewAI | İnsan ekibi | Agent · Task · Crew (rol/backstory) |
| OpenAI Agents SDK | Devir | Agent · Tool · Handoff · Guardrail |
| Google ADK | Graf (derlenen) | `Agent` + workflow ajanları |
| **Agno** | **İşletim sistemi** | `Agent` · `Team` · `Workflow` + `AgentOS` |
| MetaGPT | SOP / montaj hattı | Sabit roller (PM→Mimar→Mühendis→QA) |

**Agno'nun metaforu neden ayrı bir satır hak ediyor** `[kaynak]`: paket
`agno.os` diye bir alt paket taşıyor, içinde `AgentOS` ve arayüzler
(`agno/os/interfaces/agui/` — AG-UI protokolü). Yanında `agno.db.sqlite` /
`agno.db.postgres`, `agno.registry.Registry`, ve `RemoteAgent` / `RemoteTeam` /
`RemoteWorkflow`. Yani Agno kendini bir kütüphane değil, **ajanların çalıştığı
bir işletim ortamı** olarak konumluyor. AutoGen'in `SingleThreadedAgentRuntime`'ı
bir mesaj döngüsü; Agno'nun `AgentOS`'u bir sunucu.

Pratik sonucu: **AutoGen'de akış konuşmanın içinden doğuyor.** Ajanlar birbirine
mesaj *göndermiyor*, ortak bir thread'e yayın yapıyor ve herkes görüyor. "Kim
konuşacak" bir strateji nesnesi.

Bu yüzden AutoGen'de **tek kütüphane içinde beş desen** var. Diğerlerinde çoğu
zaman **desen = framework seçimi**: graf istiyorsan LangGraph, devir istiyorsan
Agents SDK.

Bedeli de var ve ölçtük: ortak thread bağlamı şişiriyor, konuşmacı seçimi
kırılganlık üretiyor.

## 6.2 Asıl fark: aktör modeli bir *eşzamanlılık modeli*

**`[kaynak]`** AutoGen, ajanları gerçekten **aktör** yapan neredeyse tek yaygın
framework: kendi mailbox'ı olan, mesajı tipe göre yönlendiren, makinelere
dağıtılabilen birimler.

**`[teyitsiz]`** Diğerlerinde bu katman yok: LangGraph'ın altında graf yürütücü +
checkpointer var — **durability** sağlıyor, eşzamanlılık modeli değil; CrewAI
düpedüz bir Python döngüsü; Agents SDK'da `Runner` tek bir tur döngüsü.

> "AutoGen mı LangGraph mı" çoğu zaman yanlış sorulmuş bir soru: LangGraph'ın
> grafı bir **yürütme planı**, `autogen_core` bir **eşzamanlılık modeli**.

### Ama vaat tam tutmuyor

**`[ölçüldü]`** `poc/desen_5_core_aktor.py` ve `pipeline/compare_fanin.py`, aynı
arıza enjeksiyonuyla:

| motor | temiz | sarmalayıcı arkasında | ham hata |
|---|---:|---:|---:|
| `GraphFlow` (AgentChat) | 3 dal | 2 dal | **0–1 dal, süre sınırı dolar** |
| pub/sub + `ClosureAgent` kuyruğu (core) | 3 dal | 2 dal | **2 dal, ~3 ms** |

Çöken bir handler `_process_publish` içindeki `gather`'ı erken döndürüyor,
`stop_when_idle()` bariyeri erken açılıyor, **tamamlanmış** kardeş sonuçlar
sessizce kayboluyor — ne exception, ne uyarı. Ve kaç tanesinin kaybolduğu
**deterministik değil**.

> **Aktör modeli runtime'ı koruyor, veriyi korumuyor.**

Üstelik **resmî desenler bu konuda birbiriyle çelişiyor**: *Concurrent Agents*
kuyrukla topluyor, *Mixture of Agents* `asyncio.gather` ile — yani kaybın
kaynağı olan yapıyla.

---

# BÖLÜM 7 — Nerede benziyor: 2026'nın yakınsaması

Karşılaştırmaların çoğu farkları sayıp benzerlikleri atlıyor. Oysa 2024'ten beri
bu alan **ciddi biçimde yakınsadı** ve artık dördü de aynı şeyleri yapıyor:

| Ortaklaşan şey | Durum |
|---|---|
| **Tool = fonksiyon + şema** (imza ve docstring'den üretilen) | Hepsinde aynı **`[teyitsiz]`** |
| **MCP desteği** | AutoGen'de var **`[ölçüldü]`**, ADK'da var **`[ölçüldü]`** (`tools/mcp_tool/`, `_remote_mcp_server.py`), diğerlerinde de **`[teyitsiz]`** |
| **OpenTelemetry** | AutoGen GenAI konvansiyonlarıyla **`[ölçüldü]`**, ADK'da `telemetry` alt paketi **`[ölçüldü]`** |
| **Devir/handoff fikri** | AutoGen `Swarm` **`[ölçüldü]`**, Agents SDK'nın merkezi **`[teyitsiz]`** |
| **Yapısal çıktı (Pydantic)** | Hepsinde **`[teyitsiz]`** |
| **Bellek eklentisi** | AutoGen: chromadb/mem0/redis **`[ölçüldü]`**; ADK: `memory` paketi **`[ölçüldü]`** |
| **Graf** | AutoGen `GraphFlow` ile sonradan ekledi **`[ölçüldü]`**; LangGraph ve ADK merkeze aldı |
| **İnsan döngüde** | LangGraph `interrupt()`, Agno `HumanReview` **`[kaynak]`**, AutoGen `UserProxyAgent`/`HandoffTermination` **`[ölçüldü]`** |
| **Uzak bileşen** | Agno `RemoteAgent`/`RemoteTeam`/`RemoteWorkflow` **`[kaynak]`**, ADK `remote_a2a_agent` **`[ölçüldü]`**, AutoGen gRPC worker'ları |

**Yani "hangi framework tool çağırabiliyor" gibi sorular artık ayırt edici değil.**
Kalan gerçek farklar üç tane:

1. **Runtime modeli** — aktör mü, graf yürütücü mü, düz döngü mü
2. **Akış kararının nerede verildiği** — konuşmadan mı doğuyor, önceden mi çizili
3. **Operasyonel yüzey** — eval, deploy, session yönetimi kutunun içinde mi

Ve ilginç olan: AutoGen birincide açık ara önde, üçüncüde açık ara geride.

---

# BÖLÜM 8 — ADK ile yan yana (ölçülebilen tek komşu)

`google-adk 2.6.3` bu ortamda kurulu, o yüzden burada **`[teyitsiz]`** yerine
gerçek dosya kanıtı verebiliyorum.

## 8.1 Kapsam farkı, alt paket sayısıyla

**`[ölçüldü]`** ADK'nın alt paketleri:

```
a2a, agents, apps, artifacts, auth, cli, code_executors, dependencies,
environment, errors, evaluation, events, examples, features, flows,
integrations, labs, memory, models, optimization, planners, platform,
plugins, sessions, skills, telemetry, tools, utils, workflow
```

**29 alt paket.** AutoGen'in üç paketi toplamda 26 alt paket ediyor ama içerikleri
tamamen farklı: AutoGen'de `evaluation` yok, `cli` yok, `sessions` yok,
`platform` yok, `plugins` yok, `skills` yok.

> **Okuma:** ADK'nın kapsamı **ürün-operasyon**; AutoGen'inki **eşzamanlılık**.
> ADK sana bir deployment hikâyesi satıyor, AutoGen bir mesajlaşma modeli.

## 8.2 Bildirimsel ajan: ADK'da var, AutoGen'de yarım

**`[ölçüldü]`** ADK'nın `agents/` dizininde her ajan tipinin bir de `_config`
eşi var:

```
llm_agent.py        + llm_agent_config.py
loop_agent.py       + loop_agent_config.py
parallel_agent.py   + parallel_agent_config.py
sequential_agent.py + sequential_agent_config.py
                    + config_schemas/   (dizin)
```

Yani ADK'da ajanı **YAML/JSON'dan tanımlamak** birinci sınıf bir yol.
AutoGen'de en yakın karşılık `ComponentModel` (§2.6) — çalışıyor ama ajan
*davranışını* değil, bileşen *kurulumunu* tarif ediyor.

**`[ölçüldü]`** ADK'da ayrıca bir **`skills`** paketi var
(`skill_registry.py`, `models.py`, `prompt.py`). AutoGen'de skill soyutlaması
**yok** — bu, `02 §15`'te not düştüğümüz farkın karşı tarafı.

## 8.3 ADK, LangGraph'ı ajan olarak sarıyor

**`[ölçüldü]`** `google/adk/agents/langgraph_agent.py` var. Yani ADK, LangGraph'ı
rakip değil **çalıştırılabilir bir bileşen** olarak görüyor — tıpkı AutoGen'in
`tools/langchain` adapter'ı gibi.

**Bu iki dosya birlikte önemli bir şey söylüyor:** framework'ler artık
birbirlerini dışlamıyor, sarmalıyor. "Hangisini seçmeliyim" sorusunun cevabı
gitgide "hangisi **dışta** olsun" oluyor.

## 8.4 Hata ne zaman görünür — gerçek ayrım noktası

**`[kaynak]`** ADK graf kurulurken doğruluyor (`validate_graph()`); AutoGen'de
graf hataları **çalışma zamanında** çıkıyor.

Bizim ölçtüğümüz `GraphFlow` fan-in kaybı tam da bu kategoride: kurulum sırasında
hiçbir uyarı yok, hata koşarken ve **sessizce** ortaya çıkıyor.

---

# BÖLÜM 9 — Eksen eksen matris

**`[kaynak]`** AutoGen, ADK ve Agno sütunları · **`[teyitsiz]`** diğerleri.

| eksen | AutoGen | LangGraph | CrewAI | Agents SDK | Google ADK | Agno |
|---|---|---|---|---|---|---|
| **metafor** | konuşma | graf | insan ekibi | devir | graf (derlenen) | **işletim sistemi** |
| **iletişim** | pub/sub ortak thread | paylaşılan state + `Command` | task-context zinciri | yalnızca handoff | graf kenarları | takım lideri → üye |
| **akış kararı** | konuşmacı seçimi (**değiştirilebilir**) | kenarlar (önceden çizili) | manager delegasyonu | ajanın kendisi | kenarlar (**derlenen**) | `Workflow` ilkelleri + `TeamMode` |
| **runtime** | **aktör modeli, dağıtılabilir** | graf yürütücü | Python döngüsü | tur döngüsü | graf motoru | **`AgentOS` sunucusu** |
| **dağıtım protokolü** | **gRPC + CloudEvents** | — | — | — | **A2A** | `Remote*` + **AG-UI** |
| **tool protokolü** | **MCP** (3 taşıyıcı) | MCP | MCP | MCP | **MCP** | MCP + geniş hazır toolkit |
| **izleme** | **OTel GenAI conventions** | LangSmith + OTel | ayrı | ayrı | `telemetry` paketi | mlflow autolog entegrasyonu |
| **bildirimsel tanım** | yarım (`ComponentModel`) | — | YAML | — | **tam (`*_config`, `config_schemas`)** | `Registry` + `*Factory` |
| **hata ne zaman görünür** | çalışma zamanında | çalışma zamanında | çalışma zamanında | çalışma zamanında | **graf kurulurken** | çalışma zamanında |
| **model arızası** | `ResilientClient` (bizim) | — | — | — | — | **`FallbackConfig` — hata tipine göre yedek** |
| **durability** | zayıf | **checkpointer + time-travel** | katmanlı bellek | session | `Session` + servis | **`db=` zorunlu vatandaş** |
| **insan onayı** | kod için hazır, geneli elde | `interrupt()` | — | — | — | **`HumanReview` her ilkelde** |
| **kod yürütme** | **birinci sınıf** | manuel | tool | tool | `code_executors`, ajan tipi yok | doğrulanmadı |
| **desen çeşitliliği** | **5 takım tipi** | supervisor/swarm | sequential/hierarchical | tek | workflow ajanları | `Step`/`Parallel`/`Loop`/`Condition`/`Router` |
| **eval** | ayrı araç (AutoGenBench) | ayrı | ayrı | ayrı | **`evaluation` paketi içinde** | ayrı |
| **deploy** | senin işin | LangGraph Platform | — | — | Vertex/Cloud Run | **`AgentOS`** |
| **yaşam döngüsü** | **bakım modu** | aktif | aktif | aktif | aktif | aktif |

### İki ince nokta

**1. Graf konusunda taraflar yer değiştirdi.** AutoGen grafı `GraphFlow` olarak
**beş takım tipinden biri** diye ekledi; ADK grafı **merkeze** aldı ve konuşma
modelini hiç benimsemedi.

**2. "ADK oturmuş" demek yanlış olur.** ADK 2.0 kırıcı bir sürümdü ve 1.x hattı
paralel sürüyor. Bir yılda temel API'sini değiştirdi.

---

# BÖLÜM 10 — Ne zaman hangisi

| Durum | Tercih | Gerekçe |
|---|---|---|
| Tek ajan + birkaç tool | **Agents SDK** | AutoGen'in runtime'ı boşa yük |
| Uzun süren iş akışı, checkpoint/geri sarma şart | **LangGraph** | AutoGen'de durability zayıf |
| Hazır deploy + eval + session | **ADK** | Kutunun içinde geliyor |
| Ekibin öğrenme bütçesi düşük | **CrewAI** | En düşük giriş engeli |
| Çok süreç/çok makine **tek sistem** | **AutoGen** | gRPC + CloudEvents katmanı |
| Başkasının ajanıyla konuşmak | **ADK (A2A)** | AutoGen bu problemi kapsamıyor |
| Desenleri **kıyaslamak** istiyorsan | **AutoGen** | Beşi aynı API'de; değişkeni izole edebiliyorsun |
| Kod yazma/çalıştırma döngüsü | **AutoGen** | Kod yürütme birinci sınıf |
| Sıralı/koşullu/döngülü akış + insan onayı | **Agno** | `Step`/`Condition`/`Loop`/`Router`, her birinde `HumanReview` |
| Hazır entegrasyon çok, kod az | **Agno** | Toolkit kütüphanesi geniş; `AgentOS` sunucuyu da veriyor |
| Sağlayıcı arızasına dayanıklılık | **Agno** | `FallbackConfig` hata tipine göre yedek modele düşüyor |

---

# BÖLÜM 11 — Bizim seçimimiz ve bedeli

Bu projede üç katmanı da kullandık: AgentChat günlük iş için (`graph.py`), core
gözlemlenebilirlik ve alternatif toplama için (`observability.py`, `fanin.py`),
ext model istemcisi ve MCP için (`engine.py`, `conversation.py`).

**Ve AutoGen bakım modunda.** Microsoft yeni işi Agent Framework'e (MAF) taşıdı.
Bunun **ölçülmüş** bedeli `06 §12`: aktif proje bağımlılığına üst sınır koyar,
bakım modundaki koymaz — ve düzeltecek kimse yoktur.

> **Tez cümlesi:** AutoGen'i ürün olarak değil, **desen kütüphanesi ve ölçüm
> zemini** olarak kullanıyoruz. Desenler taşınabilir; bağımlılık taşınabilir
> değil.

Bu belge o tezin gerekçesini bir kat daha aşağıda gösteriyor: taşınabilir olan
şey aslında **protokoller** — CloudEvents, MCP, OTel, OpenAI şeması. Bunların
hepsi AutoGen'in dışında yaşıyor ve AutoGen ölse de duruyorlar. Taşınamayan tek
şey `autogen_core`'un aktör API'si.

---

# Ek — Bu belgede ölçülmemiş olanlar

Dürüstlük listesi. Yukarıda **`[teyitsiz]`** geçen her şey ve fazlası:

- **LangGraph, CrewAI, OpenAI Agents SDK, Agno ve DeepAgents kurulu değil.**
  Bu beşi hakkındaki her satır ya okumaya ya da **reponun kendi koduna**
  dayanıyor (§0'daki ara yol); hiçbirini bu ortamda koşturmadım.
- **Agno'nun `[kaynak]` satırları API yüzeyidir, davranış değil.**
  `agno.workflow.__init__`'in `Loop`, `Condition`, `Router`, `HumanReview`
  ihraç ettiğini gördüm; bunların çalışırken nasıl davrandığını görmedim.
  "Agno'da durma ölçütü birinci sınıf" derken kastettiğim, API'de öyle
  görünmesi.
- **§5'teki "en iyi alternatif" sütunu bir yargıdır, ölçüm değil.** Dokuz desenin
  hiçbirini başka bir framework'te kurup kıyaslamadım. O sütun, API yüzeylerine
  bakarak verilmiş bir kanaat — yanılabilir.
- **Dağıtık runtime hiç çalıştırılmadı.** `grpcio` yok. §2.2'deki her şey
  `.proto` descriptor'ından okundu — **arayüz** bilgisi, davranış değil. Yani
  "Python↔.NET çalışıyor" demiyorum; "protokol dilden bağımsız tasarlanmış"
  diyorum.
- **A2A test edilmedi.** ADK'da dosyaların varlığını gördüm, protokolü
  koşturmadım.
- **OTel dışa aktarımı kurulmadı.** `gen_ai.*` sabitlerinin kodda olduğunu
  gördüm; gerçek bir backend'e span gittiğini görmedim.
- **ADK davranışı ölçülmedi.** Alt paket ve dosya varlığı `[ölçüldü]`; o
  dosyaların ne yaptığı `[teyitsiz]`.
- **Diğer framework'lerin fan-in davranışı ölçülmedi.** `compare_fanin.py` yalnız
  AutoGen'in iki motorunu kıyaslıyor. "LangGraph bu durumda daha iyi olurdu"
  diyemem — denemedim.

**Bu listeyi kısaltmanın yolu belli:** her framework'ü ayrı bir venv'e kurup aynı
görevi aynı arıza enjeksiyonuyla koşturmak. Yapılmadı, ve yapılmadığı için
yukarıdaki `[teyitsiz]` etiketleri duruyor.

---

**İlgili:** [09](09-framework-karsilastirma.md) kısa karşılaştırma ·
[12](12-autogen-bastan-sona.md) uçtan uca anlatım ·
[06](06-autogen-incelikleri.md) ölçülmüş tuzaklar ·
[02](02-autogen-el-kitabi.md) API el kitabı
