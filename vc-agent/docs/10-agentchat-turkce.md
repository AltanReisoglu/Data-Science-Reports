# 10 — AutoGen AgentChat: Türkçe rehber

*[08-autogen-agentchat-user-guide.md](08-autogen-agentchat-user-guide.md)'nin 24
sayfasının tamamı, Türkçe. Her bölümün başında orijinaldeki satır numarası var.*

---

## Bu belge ne, ne değil

**Ne:** AgentChat kılavuzunun bütün sayfalarını ve alt başlıklarını Türkçe
anlatan bir rehber. Hiçbir sayfa atlanmadı; sıra da orijinalle aynı.

**Ne değil:** Kelimesi kelimesine çeviri. Orijinal 49.666 kelime ve içinin
önemli bir kısmı **kod**. Kodu ve API adlarını çevirmek zaten yanlış olurdu —
`AssistantAgent` Türkçeye çevrilirse çalışmaz. O yüzden burada **anlatı Türkçe,
kod ve tanımlayıcılar İngilizce**. Bir cümlenin tam orijinalini görmek
istediğinde satır numarası duruyor: `08:2813` gibi.

**Nasıl kullanılır:** Bu dosyayı okuyup kavramı anla, sonra `08`'de o satıra
gidip kodu oradan al. Ürünün içinden de sorabilirsin — sohbet arayüzü bu
belgelerin hepsini arıyor (`docs_index.py`), ve cevabın hangi belgeden geldiğini
söylüyor.

**İlgili belgeler:**
[05](05-autogen-core-user-guide.md) core kılavuzun tam metni ·
[06](06-autogen-incelikleri.md) pratikte ısıranlar ·
[07](07-kod-rehberi.md) kavram↔kod köprüsü ·
[09](09-framework-karsilastirma.md) framework karşılaştırması

---

# BÖLÜM 1 — Başlangıç

## AgentChat nedir  · `08:53`

AgentChat, `autogen-core`'un üstüne kurulmuş **görev odaklı, üst seviye** bir
API. Core'da her şeyi kendin bağlarsın: aktörler, mesaj tipleri, abonelikler.
AgentChat'te hazır ajanlar ve hazır takımlar vardır; "iki ajan sırayla konuşsun"
demek üç satırdır.

Kılavuzun kendi konumlandırması şu: v0.2'nin yerini alan katman bu. Core'a inmek
zorunda değilsin — ama gerektiğinde inebilirsin, ve **inmen gereken yerler var**
(bkz. [06](06-autogen-incelikleri.md)).

Sunduğu şeyler:

- **Ajanlar** — `AssistantAgent`, `UserProxyAgent`, `CodeExecutorAgent`
- **Takımlar** — `RoundRobinGroupChat`, `SelectorGroupChat`, `Swarm`,
  `GraphFlow`, `MagenticOneGroupChat`
- **Sonlandırma koşulları**, **bellek**, **durum kaydı**, **bileşen
  serileştirme**

## Kurulum · `08:222`

```bash
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

`autogen-ext` **extra**'larla gelir: `[openai]`, `[azure]`, `[docker]`, `[mcp]`
gibi. Hangi özelliğin hangi extra'yı istediği [02](02-autogen-el-kitabi.md) §1'de
tablo hâlinde.

> **Bizim ölçtüğümüz tuzak:** `autogen-ext`'in `mcp` bağımlılığı üst sınırsız
> yazılmış (`mcp>=1.11.0`). MCP SDK 2.0 çıkınca `ImportError: RequestContext`
> alıyorsun. `requirements.txt`'e `mcp>=1.24,<2` pini şart. → [06 §12](06-autogen-incelikleri.md)

## Hızlı başlangıç · `08:321`

Tek ajan, tek tool, akışlı çıktı — kılavuzun "ilk beş dakika" örneği:

```python
agent = AssistantAgent(
    "assistant",
    model_client=model_client,
    tools=[get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,
)
await Console(agent.run_stream(task="What is the weather in New York?"))
```

Dikkat edilecek iki parametre:

- **`reflect_on_tool_use=True`** — ajan tool sonucunu alıp **üstüne bir cümle
  kurar**. `False` olursa tool'un ham çıktısı cevap olur.
- **`model_client_stream=True`** — token token akış. Bu olmadan cevap tek
  parça hâlinde, model bitirdikten sonra gelir. Sohbet arayüzünün "sohbet gibi"
  hissettirmesi buna bağlı.

## v0.2 → v0.4 göç kılavuzu · `08:403` (1382 satır)

Kılavuzun **en uzun sayfası** ve bu projenin tezi için en değerlisi. AutoGen
v0.4, v0.2'nin devamı değil — **sıfırdan yeniden yazımı**. Gerekçeyi kendi
cümlesiyle veriyor:

> …**from-the-ground-up rewrite adopting an asynchronous, event-driven
> architecture** to address issues such as **observability, flexibility,
> interactive control, and scale**.

Yani sayılan dört ihtiyaç: gözlemlenebilirlik, esneklik, etkileşimli kontrol,
ölçek. Dördü de **üretimin** istediği şeyler. ([09](09-framework-karsilastirma.md)
§11'de bunun neden yine de yetmediğini tartışıyorum.)

Pratik olarak neyin neye dönüştüğü:

| v0.2 | v0.4 |
|---|---|
| `ConversableAgent` | `AssistantAgent` |
| `initiate_chat(...)` | `team.run(task=...)` / `run_stream(...)` |
| `GroupChat` + `GroupChatManager` | `RoundRobinGroupChat`, `SelectorGroupChat` |
| `register_function` | `tools=[...]` |
| `llm_config={"config_list": [...]}` | `model_client=OpenAIChatCompletionClient(...)` |
| senkron | **async** (`await`, `async for`) |

> **Filtre olarak kullan:** Bir eğitim materyalinde `ConversableAgent` ya da
> `initiate_chat` görüyorsan, o kaynak **v0.2 ya da AG2 Classic** anlatıyor ve
> bu kılavuzla uyumsuz. Hazır kurs değerlendirirken ilk baktığım şey buydu.

---

# BÖLÜM 2 — Öğretici

## Giriş · `08:1786`

Öğreticinin izlediği sıra: model istemcisi → mesajlar → ajanlar → takımlar →
insan → sonlandırma → durum. Bu sıra keyfi değil; her adım öncekinin üstüne
biniyor.

## Model istemcileri · `08:1873`

Ajan modelle doğrudan konuşmaz; arada bir `ChatCompletionClient` vardır.

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient
model_client = OpenAIChatCompletionClient(model="gpt-4o")
```

Kılavuzun kapsadığı istemciler: OpenAI, Azure OpenAI (`AzureTokenProvider` ile
AAD kimlik doğrulama dahil), Azure AI, Anthropic, Ollama, ve **OpenAI-uyumlu**
her endpoint (`base_url` vererek).

Sayfanın anlattığı diğer şeyler: **önbellek** (`ChatCompletionCache` ile aynı
istemi iki kez ödememek), **kullanım sayacı** (`total_usage()`), ve
**yapılandırılmış çıktı**.

> **En pahalı tuzak burada ve kılavuzda yazmıyor:** Model adı bilinen bir OpenAI
> modeli değilse `OpenAIChatCompletionClient` **`model_info` istiyor**:
> `ValueError: model_info is required when model name is not a valid OpenAI
> model`. "OpenAI-uyumlu endpoint" kullanan **herkes** buraya düşer — bizde ilk
> canlı istekte patladı. Ve `model_info` bir **beyandır**, ölçüm değil:
> desteklenmeyen bir yeteneği iddia edersen hata başlangıçta değil, huninin en
> sonunda çıkar. → [06 §3](06-autogen-incelikleri.md), `pipeline/probe_llm.py`

## Mesajlar · `08:2236`

İki aile var ve ayrım önemli:

- **`BaseChatMessage`** — ajanlar arasında **iletişim**: `TextMessage`,
  `MultiModalMessage`, `HandoffMessage`, `StopMessage`,
  `ToolCallSummaryMessage`, `StructuredMessage[T]`
- **`BaseAgentEvent`** — ajanın **iç olayları**: `ToolCallRequestEvent`,
  `ToolCallExecutionEvent`, `ModelClientStreamingChunkEvent`,
  `MemoryQueryEvent`, `UserInputRequestedEvent`

Ayrım pratikte şu işe yarıyor: akışı dinlerken "bu bir cevap mı yoksa ajanın
yaptığı bir iş mi" sorusunu tip söylüyor. Bizim arayüzde tool çağrılarının
ayrı satır olarak görünmesi bu ayrım sayesinde.

> **Tuzak:** Takım, tanımadığı bir mesaj tipini **yönlendirmez**. Yapısal çıktı
> üreten bir ajan takımın içindeyse tipi beyan etmen gerekir:
> `GraphFlow(..., custom_message_types=[StructuredMessage[Score]])`. Tek başına
> `agent.run()`'da gerekmez — sorun yalnız yönlendirmede çıkar. → [06 §9](06-autogen-incelikleri.md)

## Ajanlar · `08:2298`

`AssistantAgent` bu kılavuzun ana kahramanı. Önemli parametreleri:

| Parametre | Ne yapar |
|---|---|
| `model_client` | Hangi model — **ajan başına ayrı** olabilir (kademelendirme buradan) |
| `tools` | Fonksiyonlar; şema **imzadan ve docstring'den** üretilir |
| `workbench` | Tool *kaynağı* (MCP dahil) — `tools` ile **birlikte verilemez** |
| `system_message` | Rolü |
| `description` | Takım içinde **yönlendirme** bunu okur |
| `model_context` | Bellek — yoksa ajan turlar arası **durumsuzdur** |
| `output_content_type` | Pydantic ile yapısal çıktı |
| `reflect_on_tool_use` | Tool sonucunu yorumlasın mı |
| `max_tool_iterations` | **Varsayılan 1** — zincirleme tool çağrısı yok |
| `handoffs` | Devir hedefleri (Swarm için) |

Çalıştırma iki biçimde:

```python
result = await agent.run(task="...")                 # tek seferde
async for event in agent.run_stream(task="..."):     # olay olay
    ...
```

> **İki tuzak:**
> `tools=` ile `workbench=` aynı ajana verilemez (`ValueError: Tools cannot be
> used with a workbench`); ikisini birlikte istiyorsan yerel fonksiyonları
> `StaticWorkbench`'e sarıp **liste** ver. Ve `max_tool_iterations` varsayılanı
> **1**: ajan bir tool çağırır, sonucu görür, **susar** — hata vermeden.
> → [06 §1, §10](06-autogen-incelikleri.md)

## Takımlar · `08:2813`

En basit takım `RoundRobinGroupChat`: ajanlar **sırayla** konuşur.

```python
team = RoundRobinGroupChat([agent_a, agent_b], termination_condition=termination)
await Console(team.run_stream(task="..."))
```

Sayfanın anlattıkları: takımı çalıştırmak, akışı izlemek, `reset()` ile
sıfırlamak, `max_turns` ile sınırlamak, ve çalışan bir takımı **durdurmak**
(`ExternalTermination`).

Dönen `TaskResult` iki şey taşır: `messages` (bütün konuşma) ve `stop_reason`
(neden durdu).

> **Bizim ölçtüğümüz:** Aynı görevde desen seçimi **%63,7 token farkı** yaratıyor
> — RoundRobin kimseyi atlayamadığı için ortada, Selector en ucuz (204), Swarm en
> pahalı (334). → `poc/kiyas.py`

## İnsan döngüde · `08:3327`

İki yol:

1. **`UserProxyAgent`** — takımın içinde bir ajan gibi durur, sırası gelince
   insana sorar. Basit ama **takımı bloklar**.
2. **Çalışmayı bitirip geri dönmek** — `HandoffTermination` ile ajan insana
   devreder, takım durur, sen cevabı alıp `run()`'ı tekrar çağırırsın.

Kılavuz ikincisini öneriyor, sebebi de net: uzun süren bir insan beklemesi
sırasında takımı ayakta tutmak kırılgan.

## Sonlandırma · `08:3670`

Sonlandırma koşulu, takımın **ne zaman duracağını** söyler. Kılavuzun listesi:

| Koşul | Ne zaman durur |
|---|---|
| `MaxMessageTermination` | N mesaj sonra |
| `TextMentionTermination` | Bir metin geçince ("TERMINATE") |
| `TokenUsageTermination` | Token sınırı |
| `TimeoutTermination` | Süre |
| `HandoffTermination` | Belirli bir hedefe devir olunca |
| `SourceMatchTermination` | Belirli bir ajan konuşunca |
| `ExternalTermination` | Dışarıdan `set()` çağrılınca |
| `StopMessageTermination` | `StopMessage` gelince |
| `FunctionCallTermination` | Belirli bir tool çağrılınca |
| `FunctionalTermination` | Kendi yazdığın fonksiyon |

**Birleştirilebilirler:** `MaxMessageTermination(10) | TextMentionTermination("TERMINATE")`
(veya `&`).

> **Bu opsiyonel değil.** Sonlandırma koşulu olmayan bir takım, sonsuz döngüye
> girdiğinde gerçek bir fatura üretir. Bizim her takımımızda `MaxMessageTermination`
> sigortası var.

## Durum yönetimi · `08:4007`

```python
state = await agent.save_state()      # veya team.save_state()
await agent.load_state(state)
```

Kaydedilen şey ajanın **bağlamıdır** — yani `model_context`'in tuttuğu. Bağlam
vermediysen kaydedilecek sohbet de yoktur. Bu yüzden `model_context` ve
`save_state` birlikte düşünülmeli.

Bizim sohbet arayüzünde sunucu yeniden başlayınca konuşmanın kalması bu.

---

# BÖLÜM 3 — İleri konular

## Selector Group Chat · `08:4231`

Konuşmacıyı **bir model seçer**. Her turda ajanların `description`'larını ve
konuşma geçmişini modele verip "sırada kim konuşmalı" diye sorar.

```python
team = SelectorGroupChat(
    [planner, web_search, data_analyst],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=...,       # istersen kendi istemin
    allow_repeated_speaker=True,
)
```

İki kaçış kapısı:

- **`selector_func`** — seçimi **Python'la** yaparsın, model hiç çağrılmaz.
  En ucuz yol; bizim triyajımızın mantığı bu.
- **`candidate_func`** — modele sunulacak aday listesini daraltırsın.

> **Tuzak:** `description` boş bırakılmış bir ajan, seçiciyi **kör** bırakır.
> Seçici elindeki tek bilgiyle karar veriyor. → [06 §13](06-autogen-incelikleri.md)

## Swarm · `08:4991`

Konuşmayı **ajanın kendisi devreder**. `handoffs` ile hedefler tanımlanır ve
ajan bir devir tool'u çağırarak sırayı verir.

```python
travel_agent = AssistantAgent(
    "travel_agent",
    handoffs=["flights_refunder", "user"],
    model_client=model_client,
    system_message="...",
)
team = Swarm([travel_agent, flights_refunder], termination_condition=termination)
```

`HandoffTermination(target="user")` ile insana devredince takım durur.

> **İki not:** Devir tool'unun adı **küçük harfe düşer**
> (`transfer_to_flights_refunder`); elle yazarsan eşleşmez, `Handoff(target=X).name`
> ile üret. Ve ölçtüğümüz kadarıyla Swarm **en pahalı desen** — her devir bir
> tool çağrısı artı iş üretmeyen bir LLM turu harcıyor.

## GraphFlow (iş akışları) · `08:5398`

Akışı **önceden çizersin**. AutoGen'in graf cevabı.

```python
builder = DiGraphBuilder()
builder.add_node(writer).add_node(reviewer)
builder.add_edge(writer, reviewer)
builder.set_entry_point(writer)          # kaynak düğüm yoksa şart
flow = GraphFlow(participants=[writer, reviewer], graph=builder.build())
```

Sayfanın kapsadıkları: sıralı akış, **paralel dal + birleşme**, koşullu kenarlar
(`condition=`), döngüler, `activation_group` / `activation_condition` ile
karmaşık bağımlılıklar, ve `MessageFilterAgent` ile bir düğüme giden mesajı
süzmek.

Birleşme (join) böyle kurulur:

```python
builder.add_edge(analyst_a, writer, activation_group="analysis", activation_condition="all")
builder.add_edge(analyst_b, writer, activation_group="analysis", activation_condition="all")
```

`"all"` → ikisini de bekler. `"any"` → ilk geleni alır.

> **Bizim en önemli ölçümümüz burada.** Bir dal exception fırlatınca takım iptal
> ediliyor ve **tamamlanmış kardeş dalların işi de gidiyor** — üç dallı koşu bir
> dalla döndü, ve kaç dal kaybedildiği **deterministik değil** (0 ve 1). Çözüm:
> hatayı mesaja çevirmek (`engine.ResilientClient`) ve **beklenen dal sayısını
> saymak**, bariyere güvenmemek. → [06 §8](06-autogen-incelikleri.md),
> `pipeline/compare_fanin.py`

## Magentic-One · `08:6031`

Genel amaçlı, hazır bir çok-ajan takımı: bir orkestratör artı web gezgini, dosya
okuyucu, kod yazıcı (`MagenticOneCoderAgent`) ve kod çalıştırıcı
(`CodeExecutorAgent`).

```python
team = MagenticOneGroupChat([surfer, file_surfer, coder, terminal], model_client=model_client)
```

Kılavuz **güvenlik uyarısıyla** birlikte veriyor: web'e giren ve kod çalıştıran
bir sistem; sandbox ve insan gözetimi öneriliyor.

Bizde kullanılmıyor — VC hattında kod yürütme yok.

## Bellek · `08:6220`

`Memory` protokolü: ajan her turda belleği sorgular ve dönen içeriği bağlamına
ekler.

| Uygulama | Ne için |
|---|---|
| `ListMemory` | En basit — sıralı bir liste |
| `ChromaDBVectorMemory` | Vektör araması (semantik hatırlama) |
| `Mem0Memory` | Harici bellek servisi |

```python
memory = ListMemory()
await memory.add(MemoryContent(content="The user prefers metric units",
                               mime_type=MemoryMimeType.TEXT))
agent = AssistantAgent("assistant", model_client=model_client, memory=[memory])
```

Sorgulama olayı akışta `MemoryQueryEvent` olarak görünür — yani "ajan neyi
hatırladı" izlenebilir.

**`model_context` ile farkı:** `model_context` o **konuşmanın** son N mesajını
tutar; `Memory` konuşmalar arası **kalıcı bilgi** içindir. İkisi ayrı iş yapar.

## Günlükleme · `08:6780`

İki logger var ve ayrım işe yarar:

```python
logging.getLogger(EVENT_LOGGER_NAME)  # yapılandırılmış olaylar
logging.getLogger(TRACE_LOGGER_NAME)  # insan okunur iz
```

> **Bizim kullandığımız yer:** `EVENT_LOGGER_NAME`'e düşen `LLMCallEvent`,
> `LLMStreamEndEvent` ve `ToolCallEvent`'i dinleyip maliyeti sayıyor ve tool
> çağrılarını denetim kaydına yazıyoruz. **Tuzak:** `create()` ilkini,
> `create_stream()` **yalnız** ikincisini yayar — akış kullanınca sadece
> `LLMCallEvent` dinlersen maliyet 0 görünür. → [06 §4](06-autogen-incelikleri.md)

## Bileşen serileştirme · `08:6812`

```python
config = agent.dump_component()      # → JSON
agent = AssistantAgent.load_component(config)
```

Ajanı, takımı, sonlandırma koşulunu, model istemcisini JSON'a çevirip geri
yükleyebilirsin. AutoGen'de **"skill" diye bir soyutlama yok**; en yakın
karşılığı bu ve AutoGen Studio'nun da kullandığı mekanizma.

Bizde kullanılmıyor — ajan topolojisi kodda tanımlı, konfigürasyondan gelen tek
şey tez.

## İzleme ve gözlemlenebilirlik · `08:6953`

OpenTelemetry. `TracerProvider` kurup runtime'a verirsin; ajan mesajları ve
model çağrıları span'a dönüşür.

```python
runtime = SingleThreadedAgentRuntime(tracer_provider=tracer_provider)
team = SelectorGroupChat([...], runtime=runtime)
```

Bizde **henüz yok** ama yolu açık: runtime'ı zaten kendimiz kuruyoruz, tek satır.
Kalan işlerin en kolayı.

## Özel ajanlar · `08:7226`

Hazır ajanlar yetmediğinde `BaseChatAgent`'tan türetirsin. Uygulaman gereken üç
şey:

```python
class MyAgent(BaseChatAgent):
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]: ...
    async def on_messages(self, messages, cancellation_token) -> Response: ...
    async def on_reset(self, cancellation_token) -> None: ...
```

Sayfanın örnekleri: aritmetik yapan basit bir ajan (`ArithmeticAgent`), Gemini
SDK'sını doğrudan saran bir ajan (`GeminiAssistantAgent`), ve o ajana
`dump_component`/`load_component` desteği eklemek
(`GeminiAssistantAgentConfig`).

Akışlı özel ajan için `on_messages_stream` yazılır.

---

# BÖLÜM 4 — Örnekler

## Örnekler · `08:7813`

Üç uçtan uca örnek. Hepsi aynı iskeleti kullanıyor: birkaç `AssistantAgent`,
`RoundRobinGroupChat`, ve bir `TextMentionTermination`.

## Seyahat planlama · `08:7877`

Dört rol — planlayıcı, yerel rehber, dil danışmanı, özetleyici — sırayla
konuşuyor ve sonuncusu birleştirip "TERMINATE" diyor. Tool yok; tamamen model
işbirliği. En sade örnek.

## Şirket araştırması · `08:8045`

Tool'lu örnek: arama, borsa verisi çekme, analiz. `FunctionTool` ile sarılmış
fonksiyonlar ve `max_turns` ile sınırlı bir `RoundRobinGroupChat`.

Bizim VC hattımıza en yakın örnek bu — ama bizde toplayıcılar **LLM'siz** ve
ajan katmanının altında; burada tool'ları ajanlar çağırıyor.

## Literatür taraması · `08:8357`

arXiv ve Google Scholar araması yapan tool'lar, sonra bir yazar ajanının
bulguları rapora çevirmesi.

> Bu örnek, bizim `arxiv.py` toplayıcımızın **neden keşif listesinde olmadığını**
> iyi gösteriyor: makale bir şirket adı taşımaz. Literatür taraması için mükemmel,
> girişim keşfi için işe yaramaz — ilk canlı koşuda 30 sinyalin 30'u bağlanamadan
> düştü.

---

## Ek — Bu kılavuzu okurken aklında tutulacaklar

**1. Kılavuz "nasıl yapılır"ı anlatıyor, "ne zaman bozulur"u anlatmıyor.**
Bu projede bulduğumuz 13 incelik hiçbir sayfada yazmıyor —
[06](06-autogen-incelikleri.md)'da hepsi, ölçümüyle birlikte.

**2. Kod örnekleri `gpt-4o` varsayıyor.** OpenAI-uyumlu başka bir endpoint
kullanıyorsan `model_info` vermen gerekecek, ve hangi yeteneklerin gerçekten
desteklendiğini **ölçmen** gerekir (`pipeline/probe_llm.py`).

**3. AgentChat tavan değil.** Kılavuzun bittiği yerde `autogen_core` başlıyor:
aktör modeli, pub/sub, olay akışı, müdahale kapısı. Bu projede AgentChat'in
çözemediği bir sorunu (paralel dal kaybı) bir kat aşağı inerek çözdük.
→ [05](05-autogen-core-user-guide.md), [07](07-kod-rehberi.md)

**4. AutoGen bakım modunda.** Bu kılavuz doğru ama donmuş; yeni özellik
gelmeyecek. Halefi Microsoft Agent Framework ve AutoGen'den bir göç kılavuzu
var. → [09](09-framework-karsilastirma.md)

---

*Kaynak: [08-autogen-agentchat-user-guide.md](08-autogen-agentchat-user-guide.md)
(microsoft/autogen, MIT). Kod ve API adları bilinçli olarak İngilizce bırakıldı.*
