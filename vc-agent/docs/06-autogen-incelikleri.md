# 06 — AutoGen'in incelikleri: pratikte ısıranlar

*Hepsi bu projede **koşarken** bulundu, dokümandan derlenmedi. Her madde: API ne
diyor · gerçekte ne oluyor · bizde nerede patladı · ne yapmalı.*

Referans metinler ayrı: [02](02-autogen-el-kitabi.md) doğrulanmış API el kitabı,
[05](05-autogen-core-user-guide.md) resmî core kılavuzunun tam metni. Bu dosya
onların söylemediği şeyler için: **sessizce yanlış davranan** yerler.

Bir örüntü var ve her maddede tekrar ediyor: **AutoGen'de arızalar gürültülü
değil.** Exception fırlatmıyor, sıfır dönüyor, boş kalıyor, asılı kalıyor. O
yüzden aşağıdakilerin çoğu "hata ayıklama" değil, *ölçüm* sonucu bulundu.

---

## 1 — `tools` ile `workbench` aynı ajana verilemez

```python
AssistantAgent("A", model_client=c, tools=[f], workbench=[wb])
# ValueError: Tools cannot be used with a workbench.
```

**Neden önemli:** Yerel fonksiyonlarınız *ve* uzak bir MCP sunucunuz varsa ikisini
tek ajanda birleştirmenin yolu bu değil.

**Çözüm:** Yerel fonksiyonları `StaticWorkbench`'e sar, `workbench`'e **liste** ver:

```python
from autogen_core.tools import FunctionTool, StaticWorkbench
workbenches = [StaticWorkbench([FunctionTool(f, description="…")]), mcp_workbench]
AssistantAgent("A", model_client=c, workbench=workbenches)
```

**Kavram:** Workbench bir tool *kaynağı*; ajan yazılırken var olmayan tool'ları
listeleyebilir. Uzak sunucunun tam olarak yaptığı şey bu.
→ `pipeline/conversation.py`

---

## 2 — `model_context` vermezsen ajanın belleği yoktur

`AssistantAgent`, `run()` çağrıları arasında **durumsuzdur**. Sohbet gibi davranması
için bağlam nesnesi vermek gerekir:

```python
from autogen_core.model_context import BufferedChatCompletionContext
AssistantAgent(..., model_context=BufferedChatCompletionContext(buffer_size=24))
```

**İnce nokta:** `buffer_size` **mesaj** sayar, token değil. Uzun bir sohbette
maliyeti sınırlayan şey bu değildir — onun için
`TokenLimitedChatCompletionContext` var.

**İkinci ince nokta:** `save_state()`'in kaydettiği şey bağlamın tuttuğudur.
Bağlam yoksa kaydedilecek sohbet de yoktur.

---

## 3 — OpenAI-*uyumlu* endpoint'te `model_info` zorunlu

```
ValueError: model_info is required when model name is not a valid OpenAI model
```

**Nerede ısırdı:** Sunucuyu ilk kez canlı modda başlattığımızda, ilk istekte.
`VC_LLM_BASE_URL` vermenin bütün amacı OpenAI dışı bir sağlayıcı kullanmak; o
yüzden **herkes** buraya düşer.

**Çözüm:** Model adı tanınmıyorsa yetenek kaydını sen ver:

```python
OpenAIChatCompletionClient(model=..., base_url=..., api_key=...,
    model_info={"vision": False, "function_calling": True, "json_output": True,
                "structured_output": True, "family": "unknown"})
```

**Tehlike:** Bu bir *beyandır*, ölçüm değil. Modelin desteklemediği bir yeteneği
iddia edersen hata başlangıçta değil, **huninin en sonunda** çıkar — yapısal
çıktı üreten skorlayıcıda. `pipeline/config.py`'de her alan ayrı ortam
değişkeniyle geçersiz kılınabilir olmasının sebebi bu.
→ `pipeline/engine.py`

---

## 4 — Akış kullanınca kullanım olayı değişir

| Çağrı | Yayılan olay |
|---|---|
| `create()` | `LLMCallEvent` |
| `create_stream()` | `LLMStreamEndEvent` — ve **asla** `LLMCallEvent` |

**Nerede ısırdı:** Sohbete `model_client_stream=True` koyunca maliyet **0**
raporlamaya başladı. Kod doğruydu; yalnız yanlış olayı dinliyordu.

**Çözüm:** İkisini birden yakala. Ayrıca gerçek sağlayıcılar akışta kullanım
bilgisini ancak `stream_options={"include_usage": true}` istenirse gönderir —
0 token görüyorsan önce buna bak.
→ `pipeline/observability.py`

---

## 5 — `ReplayChatCompletionClient` olay yaymaz; `create_calls` yalnız onda vardır

İkisi birbirinin **aynası**:

- `create_calls` özniteliği → sadece replay istemcisinde
- `LLMCallEvent`/`LLMStreamEndEvent` → sadece gerçek istemcilerde

Yalnız birini sayarsan modlardan birinde ölçüm sıfır çıkar ve bunu fark etmezsin.
Bizde önce canlı mod kördü, sonra düzeltilince kuru mod. İkisini toplamak gerekiyor.

---

## 6 — `ToolCallEvent` alanlarını öznitelikte tutmaz

`LLMCallEvent.prompt_tokens` çalışır. `ToolCallEvent.tool_name` **çalışmaz** —
her şey `.kwargs` sözlüğündedir:

```python
fields = event.kwargs
fields["tool_name"], fields["arguments"], fields["result"], fields["agent_id"]
```

**Neden sinsi:** `AttributeError` bir `logging.Handler` içinde oluşur ve
`handleError` tarafından yutulur. Olay hiç kaydedilmez, hata da görünmez.

**Bonus:** `agent_id` yalnız runtime yönetimindeki bir handler içinde doludur.
Çıplak `agent.run()`'da `None`; aynı ajan bir takım içindeyken
`TechnicalAnalyst_<uuid>`. Canlı doğrulandı.

---

## 7 — Dış runtime verirsen çöken ajan **fırlatmaz, asar**

`InterventionHandler` takmanın tek yolu runtime'ı kendin vermek:

```python
runtime = SingleThreadedAgentRuntime(intervention_handlers=[h])
GraphFlow(..., runtime=runtime)
```

Ama bunu yapınca hata semantiği değişir:

| Runtime | Çöken ajan |
|---|---|
| Gömülü (AgentChat kendi kurar) | `run_stream` **exception fırlatır** |
| Dışarıdan verilen | `run_stream` **hiç dönmez** |

Gelmeyecek bir sonlanma mesajı beklenir. `MaxMessageTermination` kurtaramaz —
yeni mesaj da gelmiyor.

**Çözüm:** Duvar saati sınırı. Burada tedbir değil, **doğruluk şartı**. Ayrıca
runtime'ı sen verdiysen `start()`/`stop()` de sana ait; AgentChat yalnız kendi
kurduğu runtime'ı başlatıp durdurur.
→ `pipeline/graph.py`

---

## 8 — Paralel dalda bir çöküş kardeşleri de götürür

GraphFlow fan-out'unda bir dal exception fırlatınca takım iptal edilir ve
**tamamlanmış** kardeş dalların çıktısı da gider. Ölçüldü
(`pipeline/compare_fanin.py`, aynı arıza enjeksiyonu):

| motor | temiz | sarmalayıcı arkasında | ham hata |
|---|---:|---:|---:|
| GraphFlow | 3 | 2 | **0–1, süre sınırı dolar** |
| core pub/sub + `ClosureAgent` kuyruğu | 3 | 2 | **2, ~3 ms** |

**Kaç dal kaybedildiği deterministik değil** — tekrarlı koşularda 0 ve 1.

**İki çözüm, ikisi de kullanıldı:**
1. Hatayı **mesaja çevir** (`engine.ResilientClient`): model çağrısı patlarsa
   exception yerine metin döner, join yine üç girdi alır.
2. Toplamayı framework'ten al: sonuçlar üretildiği anda bir topic'e yayınlanır,
   `ClosureAgent` kuyruğa boşaltır. Güvenilmeyecek bariyer yoktur, çünkü bariyer
   yoktur.

**Not:** Resmî desenler bu konuda **birbiriyle çelişiyor**. *Concurrent Agents*
kuyrukla toplar; *Mixture of Agents* `asyncio.gather(...)` ile — POC'ta
(`poc/desen_5_core_aktor.py`) sessiz kardeş kaybının kaynağı olan yapı.

---

## 9 — Takım, tanımadığı mesaj tipini yönlendirmez

```
ValueError: Message type StructuredMessage[Score] is not registered.
```

`output_content_type` kullanan bir ajan takımın içindeyse tipi beyan etmelisin:

```python
GraphFlow(..., custom_message_types=[StructuredMessage[Score]])
```

Tek başına `agent.run()`'da gerekmez — sorun yönlendirmede çıkar.

---

## 10 — `max_tool_iterations` varsayılanı **1**

Ajan bir tool çağırır, sonucu görür, **durur**. "Önce ara, sonra bulduğunu incele"
gibi zincirleme davranış varsayılan ayarla imkânsızdır ve hata vermez — ajan
sadece erken susar.

```python
AssistantAgent(..., max_tool_iterations=6)
```

---

## 11 — `@message_handler` tipi imzadan çıkarır

Tip çıkarımı `get_type_hints()` ile yapılır ve `from __future__ import annotations`
varken anotasyonlar **modül genelinde** çözülür. Handler'ın kullandığı adı
fonksiyon içinde import edersen:

```
NameError: name 'MessageContext' is not defined
```

Kayıt sırasında patlar, handler'ın kendisinde değil. `MessageContext`,
`RoutedAgent`, `TopicId` modül düzeyinde import edilmeli.

**Ek:** Parametre adları da bağlayıcıdır — `message` ve `ctx` olmak zorunda
(POC'ta bulunmuştu; bu yüzden o iki isim Türkçeleştirilememişti).

---

## 12 — Bağımlılık üst sınırı koymayan proje kurulamaz hâle gelir

`autogen-ext 0.7.5` → `mcp>=1.11.0`, üst sınır yok. MCP SDK 2.0 çıkınca:

```
ImportError: cannot import name 'RequestContext' from 'mcp.shared.context'
```

Aynı gün `google-adk` aynı bağımlılığı `mcp>=1.24,<2` diye sınırlamış. Aktif
proje sınır koyar, bakım modundaki koymaz — ve düzeltecek kimse yoktur.
`requirements.txt`'teki `mcp>=1.24,<2` pini bu yüzden zorunlu.

---

## 13 — Küçük ama pahalı olanlar

| İncelik | Sonuç |
|---|---|
| `description` boş bırakılmış ajan | `SelectorGroupChat` kör seçim yapar |
| `Handoff` tool adı küçük harfe düşer (`transfer_to_veriuzmani`) | Elle yazarsan eşleşmez; `Handoff(target=X).name` ile üret |
| Sonlandırma koşulu yok | Sonsuz ajan döngüsü = gerçek fatura |
| `stop_when_idle()` | Bir handler çökerse bariyer erken açılır — güvenme, **beklenen sonucu say** |
| MCP `ask_question` (DeepWiki) | Şu an sunucu tarafında bozuk: şemaya uygun girdiye "Error processing question:" döner. `read_wiki_structure` / `read_wiki_contents` çalışıyor |

---

## Bu projede hangi yüzey kullanılıyor

| AutoGen yüzeyi | Nerede |
|---|---|
| `AssistantAgent`, tool'lar, yapısal çıktı | `pipeline/agents/` |
| `GraphFlow` + `DiGraphBuilder`, join `activation_condition="all"` | `pipeline/graph.py` |
| `autogen_core` pub/sub, `RoutedAgent`, `ClosureAgent`, `TypeSubscription` | `pipeline/fanin.py` |
| `SingleThreadedAgentRuntime`, `InterventionHandler`, `DropMessage` | `pipeline/graph.py`, `observability.py` |
| Olay akışı (`LLMCallEvent`, `LLMStreamEndEvent`, `ToolCallEvent`) | `pipeline/observability.py` |
| `model_context`, `StaticWorkbench`, `McpWorkbench`, `CancellationToken`, `save_state` | `pipeline/conversation.py` |
| `ReplayChatCompletionClient` (deterministik kuru mod) | `pipeline/engine.py`, `poc/` |

**Henüz kullanılmayan:** dağıtık runtime (gRPC), `dump_component`/`load_component`,
kod yürütücüler, `Memory` protokolü (ChromaDB/mem0), Magentic-One, OpenTelemetry,
`Handoff`/`Swarm` (POC'ta var, pipeline'da yok).
