# 12 — AutoGen baştan sona

*Kaynak `docs/` altındaki iki resmî kılavuzun tam metni — atıflar `05:satır`
(core) ve `08:satır` (agentchat) biçiminde, açıp bakabilirsin.*

---

# 1 — AutoGen nedir: bir yeniden yazım ve üç katman

Önce en çok yanlış anlaşılan şey: **v0.4, v0.2'nin devamı değil, sıfırdan yeniden yazımı.** Gerekçeyi resmî göç kılavuzunun kendisi söylüyor (`08:403`):

> a **from-the-ground-up rewrite adopting an asynchronous, event-driven architecture** to address issues such as observability, flexibility, interactive control, and scale.

Sayılan dört ihtiyaç — gözlemlenebilirlik, esneklik, etkileşimli kontrol, ölçek — tam olarak **üretimin** istediği şeyler. Bu cümle bütün tasarımı açıklıyor.

Üç paket, üç katman:

```
  autogen-ext        model istemcileri · tool'lar · MCP · kod yürütücüler
       ↑
  autogen-agentchat  AssistantAgent · takımlar · sonlandırma · yapısal çıktı
       ↑             ← tutorial'ların bittiği yer
  autogen-core       aktör modeli · event-driven runtime · pub/sub · gRPC
                     ← asıl mühendislik hikâyesi burada
```

**Çoğu karşılaştırmanın atladığı nokta:** AutoGen'i ayıran şey üst katman değil, alt katman. AgentChat'te gördüğün `AssistantAgent` her framework'te var. `autogen_core` karşılığı çoğunda **yok**.

Ben aşağıda **alttan yukarı** anlatacağım, çünkü üstten öğrenirsen ısıran şeylerin sebebini hiç göremiyorsun.

---

# 2 — Katman 1: `autogen_core` (aktör modeli)

## 2.1 Bir ajanın kimliği: `(type, key)`

`05:601`. Ajan bir nesne değil, bir **adres**.

- `type` → hangi rol ("analyst")
- `key` → hangi örnek ("company-42")

Sen sınıfı değil, bir **fabrikayı** kaydediyorsun; örneği runtime **ilk mesaj geldiğinde** yaratıyor (lazy):

```python
await MyAgent.register(runtime, "my_agent", lambda: MyAgent("demo"))
```

## 2.2 Runtime

`05:490`. Mesajlaşmayı yürüten, kimlik ve yaşam döngüsünü yöneten katman. İki tip:

| Tip | Ne zaman |
|---|---|
| `SingleThreadedAgentRuntime` | Tek süreç, tek dil |
| `GrpcWorkerAgentRuntimeHost` + worker'lar | Çok süreç, çok makine, Python↔.NET |

Kritik vaat: **ikisi aynı API'yi sunuyor**, ajan kodunu değiştirmeden geçebiliyorsun (`05:1742`).

## 2.3 Üç mesajlaşma biçimi

`05:1108`:

| Biçim | Nasıl | Ne zaman |
|---|---|---|
| **Doğrudan** | `send_message(msg, AgentId(...))` | Cevap bekliyorsan, RPC gibi |
| **Yayın** | `publish_message(msg, TopicId(...))` | Kim dinliyorsa duysun |
| **Cevap** | handler'dan `return` | Doğrudan mesajın karşılığı |

**Yayının cevabı yoktur.** Bu kısıt değil, tasarım: yayınla haberleşen ajanlar birbirini tanımak zorunda kalmıyor.

## 2.4 Topic ve abonelik — kılavuzun en değerli sayfası

`05:670`. En az konuşulan, en çok işe yarayacak sayfa. Bir topic'in iki parçası var: **tip** (kategori) ve **kaynak** (o kategorideki benzersiz kimlik). `TypeSubscription` bir topic *tipini* bir ajan *tipine* bağlıyor. Ve asıl incelik:

> **Topic kaynağı, ajan anahtarına dönüşür.**

Yani `("gorev", "sirket-42")` topic'ine yayın yapınca runtime `("analist", "sirket-42")` ajanını yaratıyor. **Şirket başına izole ajan örneği bedava.** Çok kiracılı yapının hazır mekanizması bu — "günde 200 şirket mi 5.000 mi" sorusu cevaplanınca ilk bakılacak yer burası.

## 2.5 `RoutedAgent` — tipe göre yönlendirme

`05:894`:

```python
class MyAgent(RoutedAgent):
    @message_handler
    async def handle_text(self, message: TextMessage, ctx: MessageContext) -> None: ...

    @message_handler
    async def handle_image(self, message: ImageMessage, ctx: MessageContext) -> None: ...
```

Hangi handler'ın çalışacağını **tip anotasyonu** belirliyor. Bir ajanın birden çok handler'ı olabilir.

## 2.6 Olaylar — gözlemlenebilirlik burada

`05:1534`. İki logger var:

```python
logging.getLogger(EVENT_LOGGER_NAME)  # "autogen_core.events" — yapılandırılmış
logging.getLogger(TRACE_LOGGER_NAME)  # insan okunur iz
```

Akıştaki olaylar: `LLMCallEvent`, `LLMStreamEndEvent`, `ToolCallEvent`, `MessageEvent`, `MessageDroppedEvent`, `MessageHandlerExceptionEvent`. **Bizim maliyet ölçümümüzün tamamı bu akıştan geliyor** — modele değil, olaya bakıyoruz.

Ayrıca `InterventionHandler`: her mesaj geçişini görebilen, isterse `DropMessage` döndürüp **durduran** bir kapı (`05:6534`, `05:6638`). Bizim denetim katmanımız bu.

`05:1888` bileşen yapılandırma: `dump_component()` / `load_component()` ile ajanı, istemciyi, tool'u JSON'a çevirip geri yükleme. **AutoGen'de "skill" diye bir soyutlama yok**; en yakın karşılığı bu.

---

# 3 — Katman 2: `autogen_agentchat` (günlük iş)

## 3.1 `AssistantAgent`'ın anatomisi

`08:2298`. Bütün kılavuzun ana kahramanı, ve parametreleri kararların çoğunu içeriyor:

| Parametre | Ne yapar |
|---|---|
| `model_client` | Hangi model — **ajan başına ayrı** olabilir (model kademelendirme buradan) |
| `tools` | Fonksiyonlar; şema **imzadan ve docstring'den** üretilir |
| `workbench` | Tool *kaynağı* (MCP dahil) — `tools` ile birlikte verilemez |
| `system_message` | Rolü |
| `description` | Takım içinde **yönlendirme bunu okur** |
| `model_context` | Bellek — yoksa ajan turlar arası **durumsuzdur** |
| `output_content_type` | Pydantic ile yapısal çıktı |
| `max_tool_iterations` | **Varsayılan 1** |
| `handoffs` | Devir hedefleri (Swarm için) |

## 3.2 İki mesaj ailesi

`08:2236`. Ayrım pratikte çok işe yarıyor:

- **`BaseChatMessage`** → ajanlar arası **iletişim**: `TextMessage`, `HandoffMessage`, `StopMessage`, `StructuredMessage[T]`
- **`BaseAgentEvent`** → ajanın **iç olayları**: `ToolCallRequestEvent`, `ToolCallExecutionEvent`, `ModelClientStreamingChunkEvent`, `MemoryQueryEvent`

Akışı dinlerken "bu bir cevap mı, yoksa ajanın yaptığı bir iş mi" sorusunu **tip** cevaplıyor. Bizim arayüzde tool çağrılarının ayrı satır olarak görünmesi bu sayede.

## 3.3 Beş takım tipi

Tek kütüphanede beş orkestrasyon deseni — diğer framework'lerin çoğunda **desen seçimi = framework seçimi**:

| Takım | Kim konuşacağına kim karar verir | Kaynak |
|---|---|---|
| `RoundRobinGroupChat` | Sıra | `08:2813` |
| `SelectorGroupChat` | **Bir model** (ya da `selector_func` ile Python) | `08:4231` |
| `Swarm` | **Ajanın kendisi**, devir tool'uyla | `08:4991` |
| `GraphFlow` | **Önceden çizilmiş graf** | `08:5398` |
| `MagenticOne` | Genel amaçlı orkestratör | `08:6031` |

`SelectorGroupChat`'in iki kaçış kapısı önemli: **`selector_func`** ile seçimi Python'la yaparsın, model hiç çağrılmaz (en ucuz yol); **`candidate_func`** ile aday listesini daraltırsın.

`GraphFlow` şöyle kuruluyor:

```python
builder = DiGraphBuilder()
builder.add_node(writer).add_node(reviewer)
builder.add_edge(writer, reviewer)
builder.set_entry_point(writer)
flow = GraphFlow(participants=[writer, reviewer], graph=builder.build())
```

Paralel dal, birleşme (`activation_condition="all"`), koşullu kenar ve döngü destekliyor.

## 3.4 Sonlandırma — opsiyonel değil

`08:3670`. `MaxMessageTermination`, `TextMentionTermination`, `TokenUsageTermination`, `TimeoutTermination`, `HandoffTermination`, `SourceMatchTermination`, `ExternalTermination`, `FunctionCallTermination`… ve birleştirilebiliyorlar:

```python
MaxMessageTermination(10) | TextMentionTermination("TERMINATE")
```

**Sonlandırma koşulu olmayan takım = sonsuz döngü = gerçek fatura.** Bizim her takımımızda sigorta var.

## 3.5 Durum, bellek, insan

- **Durum** (`08:4007`): `save_state()` / `load_state()`. Kaydedilen şey **`model_context`'in tuttuğudur** — bağlam vermediysen kaydedilecek sohbet de yok. İkisi birlikte düşünülmeli.
- **Bellek** (`08:6220`): `Memory` protokolü, ChromaDB/mem0 uygulamaları.
- **İnsan döngüde** (`08:3327`): iki yol var — `UserProxyAgent` (basit ama takımı bloklar) ve `HandoffTermination` ile **çalışmayı bitirip geri dönmek**. Kılavuz ikincisini öneriyor: uzun insan beklemesi sırasında takımı ayakta tutmak kırılgan.

---

# 4 — Katman 3: `autogen_ext` (dış dünya)

- **Model istemcileri** (`05:1980`): `ChatCompletionClient` arayüzü — `create()`, `create_stream()`, `total_usage()`, `count_tokens()`
- **Model bağlamı** (`05:2341`): `BufferedChatCompletionContext` (**mesaj** sayar, token değil), `TokenLimitedChatCompletionContext`
- **Tool'lar** (`05:2473`): `FunctionTool` — şema imzadan ve docstring'den
- **Workbench ve MCP** (`05:2841`): `StaticWorkbench`, `McpWorkbench`. Workbench bir tool **kaynağı**; ajan yazılırken var olmayan tool'ları listeleyebiliyor — uzak MCP sunucusunun yaptığı tam olarak bu
- **Kod yürütücüler** (`05:3054`): Docker / yerel komut satırı. AutoGen'in kurucu makalesinin konusu buydu, ve **kod yürütme hâlâ birinci sınıf** — çoğu rakipte sadece bir tool

---

# 5 — Dokuz çok-ajan deseni

`05:3209` ve sonrası. Ana ayrım: akış **önceden mi çizili**, yoksa **konuşmadan mı doğuyor**.

| Desen | Fikir | Bizde |
|---|---|---|
| **Concurrent Agents** `05:3236` | Aynı topic'e abone ajanlar paralel koşar, sonuçlar kuyruğa toplanır | **Evet** — `fanin.py` birebir bu |
| **Sequential Workflow** `05:3504` | Her ajan bir öncekinin çıktısını alır | Katman mimarimiz bu, ama ilk iki katmanda LLM yok — kodla kurulu |
| **Group Chat** `05:3772` | Ortak thread + konuşmacı seçici | Hayır. Kılavuzun kendi cümlesi: *"not meant to be used in real applications… a starting point"* |
| **Handoffs** `05:4349` | Ajan devir tool'uyla sırayı verir | Hayır — ölçtüğümüz **en pahalı desen** (334 token; Selector 204) |
| **Mixture of Agents** `05:4989` | Katmanlı işçiler, orkestratör toplar | Hayır — sebebi aşağıda |
| **Multi-Agent Debate** `05:5358` | Seyrek topoloji, komşular birbirini düzeltir, çoğunluk oyu | Hayır — skorlamada **sabit rubrik** istiyoruz, oy değil |
| **Reflection** `05:5822` | Üretici + eleştirmen döngüsü | **Kısmen** — `RiskAuditor` tek turluk bir reflection. Döngüye çevirmedim: kılavuz **durma ölçütü önermiyor** |
| **Code Execution** `05:6188` | Model yazar, ajan çalıştırır | Uygulanamaz |

Concurrent Agents'ın kritik parçası **sonuç toplama**:

```python
queue = asyncio.Queue[TaskResponse]()

async def collect_result(_agent: ClosureContext, message: TaskResponse, ctx: MessageContext) -> None:
    await queue.put(message)

await ClosureAgent.register_closure(
    runtime, "collector", collect_result,
    subscriptions=lambda: [TypeSubscription(topic_type=RESULTS, agent_type="collector")],
)
```

İşçiler sonucu **yayınlar**, `ClosureAgent` kuyruğa boşaltır, sen kuyruktan okursun.

---

# 6 — Pratikte ne ısırıyor

[06-autogen-incelikleri.md](06-autogen-incelikleri.md)'nin tamamı bu — 13 madde, hepsi **koşarken** bulundu. Ve hepsinde tekrar eden bir örüntü var:

> **AutoGen'de arızalar gürültülü değil.** Exception fırlatmıyor; sıfır dönüyor, boş kalıyor, asılı kalıyor.

O yüzden bunların çoğu "hata ayıklama" değil, **ölçüm** sonucu bulundu. En pahalı beşi:

**① Fan-in'de bir çöküş kardeşleri de götürüyor.** `compare_fanin.py`, iki toplama mekanizmasını aynı arıza enjeksiyonuyla ölçüyor:

| motor | temiz | sarmalayıcı arkasında | ham hata |
|---|---:|---:|---:|
| `GraphFlow` (AgentChat) | 3 dal | 2 dal | **0–1 dal, süre sınırı dolar** |
| pub/sub + `ClosureAgent` kuyruğu (core) | 3 dal | 2 dal | **2 dal, ~3 ms** |

Kendisiyle ilgisi olmayan bir dalın çökmesi **tamamlanmış** kardeşlerin işini yok ediyor, ve **kaç tanesini yok ettiği deterministik değil**. Bundan daha güçlü bir bulgu: **resmî desenler bu konuda birbiriyle çelişiyor** — Concurrent Agents kuyrukla topluyor, Mixture of Agents `asyncio.gather` ile, yani kaybın kaynağı olan yapıyla.

**② Dış runtime verirsen çöken ajan fırlatmıyor, asıyor.** `InterventionHandler` takmanın tek yolu runtime'ı kendin kurmak — ama o anda hata semantiği değişiyor. Gömülü runtime `run_stream`'den exception fırlatıyor, dıştan verilen **hiç dönmüyor**. `MaxMessageTermination` kurtaramıyor, çünkü yeni mesaj da gelmiyor. **Duvar saati sınırı burada tedbir değil, doğruluk şartı.**

**③ Akış kullanınca ölçüm olayı değişiyor.** `create()` → `LLMCallEvent`; `create_stream()` → **yalnız** `LLMStreamEndEvent`. `model_client_stream=True` koyduğumuz an maliyet **0** raporlamaya başladı; kod doğruydu, yanlış olayı dinliyordu.

**④ `ToolCallEvent` alanlarını öznitelikte tutmuyor** — hepsi `.kwargs` içinde. `event.tool_name` yazarsan `AttributeError` alırsın, ve o hata bir `logging.Handler` içinde oluştuğu için **yutulur**: olay hiç kaydedilmez, hata da görünmez.

**⑤ `max_tool_iterations` varsayılanı 1.** Ajan bir tool çağırır, sonucu görür, **susar**. "Önce ara, sonra bulduğunu incele" davranışı varsayılan ayarla imkânsız ve hata vermiyor.

Kalanlar da `06`'da: `tools`↔`workbench` çatışması, OpenAI-*uyumlu* endpoint'te `model_info` zorunluluğu, `@message_handler`'ın `get_type_hints()` tuzağı, takımın tanımadığı mesaj tipini yönlendirmemesi, ve bakım modundaki paketin bağımlılık üst sınırı koymaması (`mcp>=1.11` üst sınırsız → MCP SDK 2.0 çıkınca kurulum kırıldı).

---

# 7 — Bu projede hangi kavram nerede

Öğrenmenin en hızlı yolu bu tablo — her kavramın çalışan bir karşılığı var:

| AutoGen yüzeyi | Bizde |
|---|---|
| `AssistantAgent`, tool'lar, yapısal çıktı | [pipeline/agents/](../pipeline/agents/) |
| `GraphFlow` + `DiGraphBuilder`, join | [pipeline/graph.py](../pipeline/graph.py) |
| core pub/sub, `RoutedAgent`, `ClosureAgent` | [pipeline/fanin.py](../pipeline/fanin.py) |
| `InterventionHandler`, `DropMessage`, olay akışı | [pipeline/observability.py](../pipeline/observability.py) |
| `model_context`, `StaticWorkbench`, `McpWorkbench`, `save_state` | [pipeline/conversation.py](../pipeline/conversation.py) |
| `ReplayChatCompletionClient` (deterministik kuru mod) | [pipeline/engine.py](../pipeline/engine.py) |

**Henüz kullanmadıklarımız:** dağıtık runtime (gRPC), `dump_component`/`load_component`, kod yürütücüler, `Memory` protokolü, Magentic-One, OpenTelemetry, `Handoff`/`Swarm`.

---

# 8 — Neden AutoGen, ne zaman değil

**Ayıran şey aktör modeli.** AutoGen, ajanları gerçekten **aktör** yapan neredeyse tek yaygın framework: kendi mailbox'ı olan, mesajı tipe göre yönlendiren, makinelere dağıtılabilen birimler. LangGraph'ın altındaki graf yürütücü + checkpointer **durability** sağlıyor, eşzamanlılık modeli değil. Yani "AutoGen mı LangGraph mı" çoğu zaman yanlış sorulmuş bir soru — ikisi farklı katmanlarda duruyor.

**Ama vaat tam tutmuyor, ve ölçtük:** çöken bir handler `_process_publish` içindeki `gather`'ı erken döndürüyor, `stop_when_idle()` bariyeri erken açılıyor, tamamlanmış kardeş sonuçlar sessizce kayboluyor.

> **Aktör modeli runtime'ı koruyor, veriyi korumuyor.**

**Ne zaman kullanmazsın:** tek ajanlı basit bir tool döngüsü istiyorsan (Agents SDK daha ucuz), uzun süren iş akışında checkpoint/time-travel şartsa (LangGraph), hazır deploy istiyorsan (ADK/Vertex), ya da ekibin öğrenme bütçesi düşükse (CrewAI).

**Ve bilmen gereken:** AutoGen **bakım modunda**. Microsoft yeni işi Agent Framework'e (MAF) taşıdı. Bunun ölçülmüş bedeli yukarıdaki bağımlılık kırılması: aktif proje üst sınır koyar, bakım modundaki koymaz — ve düzeltecek kimse yoktur. Tez cümlemiz de zaten bu: **AutoGen'i ürün olarak değil, desen kütüphanesi ve ölçüm zemini olarak kullanıyoruz.**

---

# 9 — Nereden başlamalı

Sırayla oku, atlama:

1. **[11-core-guide-turkce.md](11-core-guide-turkce.md)** — Bölüm 2 (Temel kavramlar). 40 satır, ama `(type, key)` ve topic→anahtar eşlemesini anlamadan gerisi havada durur.
2. **[10-agentchat-turkce.md](10-agentchat-turkce.md)** — Ajanlar + Takımlar + Sonlandırma. Günlük iş burada.
3. **[06-autogen-incelikleri.md](06-autogen-incelikleri.md)** — hepsi. 271 satır ve bence en yüksek getirili belge, çünkü kılavuzların **söylemedikleri** burada.
4. **[07-kod-rehberi.md](07-kod-rehberi.md)** — kavramın kodda nerede yaşadığı; sonunda uçtan uca iki izlek var.
5. Tam metin lazım olduğunda [05](05-autogen-core-user-guide.md) ve [08](08-autogen-agentchat-user-guide.md), satır numarasıyla.

Ürünün içinden de arayabilirsin — hepsi indekste:

```bash
python -m pipeline.server --port 8777   # sonra "graphflow join nasıl kurulur" diye sor
```

**Bir not:** yukarıdaki `[ölçüldü]` etiketli sayıların hepsi bu repodaki koşulardan; `poc/kiyas.py` ve `pipeline/compare_fanin.py` tekrar çalıştırılabilir. Diğer framework'lerle ilgili satırların bir kısmı [09](09-framework-karsilastirma.md)'da **`[teyitsiz]`** işaretli — onları ben ölçmedim, okuduğumla yazdım.
