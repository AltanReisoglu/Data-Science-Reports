# 02 — AutoGen El Kitabı: Baştan Sona

*Hedef: bu belgeyi okuyup üstüne tam donanımlı bir agent sistemi kurabilmek —
skill, MCP, sandbox, bellek, kalıcılık, dağıtık runtime dahil.*

**Sürüm:** AutoGen **v0.7.5** · Python 3.10+ · Belge tarihi 2026-08-13
**Doğrulama yöntemi:** Buradaki her sınıf/parametre adı, kurulu paketten
(`autogen/.venv`) introspection ile çıkarıldı — dokümantasyondan kopyalanmadı.
Bir isim burada yazıyorsa, o sürümde gerçekten var.

İlgili belgeler: [01 — Kaynak haritası](01-autogen-kaynak-haritasi.md) ·
[report/14 — Agentic mega-atlas](../../report/14-agentic-mega-atlas.md) · Çalışan POC: [../poc/](../poc/)

---

## §0 — Bunu okumadan başlama

AutoGen **bakım modunda** (2026 Nisan'dan beri). Yeni özellik gelmiyor, halefi
Microsoft Agent Framework. Buna rağmen bu el kitabı neden yazılıyor:

1. **Öğrenmek için hâlâ en iyi kaynak.** Aktör modeli, konuşmacı seçimi, handoff,
   graf akışı — hepsi tek kütüphanede ve okunabilir hâlde.
2. **Halefine geçiş kolay.** MAF, AutoGen'in kavramlarını taşıyor; burada
   öğrendiğin her şey oraya çevriliyor.
3. **Ölçüm için ideal.** Beş orkestrasyon desenini aynı API ile deneyip
   karşılaştırabildiğin başka bir kütüphane yok.

**Ama üretime kurmadan önce oku:** [§20 Tuzaklar](#20--tuzaklar-ölçülmüş) bölümünde
bu oturumda **ölçerek bulduğum** üç gerçek hata var; ikisi sessiz veri kaybı.

---

## §1 — Kurulum ve extras haritası

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "autogen-agentchat" "autogen-core" "autogen-ext[openai]"
```

`autogen-ext`'in **38 extra**'sı var. Neyi ne zaman kuracağın:

| İhtiyaç | Extra | Gerçek bağımlılık |
|---|---|---|
| OpenAI / OpenRouter / uyumlu API | `openai` | `openai>=1.93`, `tiktoken` |
| Anthropic | `anthropic` | `anthropic>=0.48` |
| Yerel model | `ollama` / `llama-cpp` | `ollama>=0.4.7` / `llama-cpp-python` |
| Azure | `azure` | `azure-ai-inference`, `azure-identity` … |
| **MCP** | `mcp` | `mcp>=1.11.0` ⚠ bkz. §20.2 |
| **Docker sandbox** | `docker` | `docker~=7.0` |
| Jupyter yürütücü | `jupyter-executor` | `ipykernel`, `nbclient` |
| Docker + Jupyter | `docker-jupyter-executor` | `docker`, `websockets`, `aiohttp` |
| Vektör bellek | `chromadb` / `mem0` / `redisvl` | `chromadb>=1.0` / `mem0ai` / `redisvl` |
| **Dağıtık runtime** | `grpc` | `grpcio~=1.70` |
| Magentic-One | `magentic-one` | `playwright`, `markitdown[all]`, `magika` |
| Web/dosya/video gezgini | `web-surfer` / `file-surfer` / `video-surfer` | `playwright` / `markitdown` / `opencv`, `whisper` |
| LLM yanıt önbelleği | `diskcache` / `redis` | `diskcache` / `redis>=5.2.1` |
| Renkli konsol | `rich` | `rich>=13.9.4` |

Tam kurulum (ağır — ~2 GB):
```bash
pip install "autogen-ext[openai,mcp,docker,chromadb,grpc,magentic-one,rich]" "mcp>=1.24,<2"
```

> ⚠ **`mcp` pini şart.** `autogen-ext` `mcp>=1.11.0` diyor, üst sınır yok. MCP SDK
> 2.0 çıkınca `ImportError: cannot import name 'RequestContext'` alırsın. Detay §20.2.

---

## §2 — Zihinsel model: üç katman

```
┌─────────────────────────────────────────────────────────┐
│ APPS      Magentic-One · AutoGen Studio · senin uygulaman │
├─────────────────────────────────────────────────────────┤
│ AgentChat  görev odaklı, yüksek seviye                   │
│            AssistantAgent · Team · Termination           │
├─────────────────────────────────────────────────────────┤
│ Core       aktör modeli, event-driven runtime            │
│            RoutedAgent · pub/sub · gRPC ile dağıtık      │
├─────────────────────────────────────────────────────────┤
│ Extensions model istemcileri · sandbox · bellek · MCP    │
└─────────────────────────────────────────────────────────┘
```

**Karar kuralı:** İşin "birkaç ajan konuşsun, iş bitsin" ise AgentChat yeter.
İşin "binlerce olay, süreçlere dağılmış aktörler, kısmi hata toleransı" ise Core.
Süper agent sisteminde **ikisi birden** olacak — bkz. [§21 Blueprint](#21--süper-agent-sistemi-blueprint).

---

## §3 — Ajanlar

`autogen_agentchat.agents` içindeki **tüm** ajanlar:

| Ajan | Ne yapar |
|---|---|
| `AssistantAgent` | Ana iş atı. Model + tool + handoff + bellek |
| `UserProxyAgent` | İnsanı döngüye sokar (`input_func`) |
| `CodeExecutorAgent` | Kod üretir **ve çalıştırır** (sandbox'la birlikte) |
| `SocietyOfMindAgent` | Bir **takımı** tek ajan gibi paketler |
| `MessageFilterAgent` | Ajana giden mesajları filtreler (`PerSourceFilter`) |
| `BaseChatAgent` | Kendi ajanını yazmak için taban sınıf |

### AssistantAgent — parametreler (v0.7.5, doğrulanmış)

```python
AssistantAgent(
    name, model_client, *,
    tools=None,                      # düz fonksiyon | BaseTool
    workbench=None,                  # Workbench | Sequence[Workbench]  (MCP buraya)
    handoffs=None,                   # ["Hedef"] | [Handoff(...)]
    model_context=None,              # bağlam penceresi stratejisi (§6)
    description="...",               # SelectorGroupChat bunu okur — boş bırakma
    system_message="...",
    model_client_stream=False,       # token token akış
    reflect_on_tool_use=None,        # tool sonucunu modele yorumlatır
    max_tool_iterations=1,           # tek turda kaç tool döngüsü
    tool_call_summary_format="{result}",
    output_content_type=None,        # Pydantic → yapılandırılmış çıktı (§7)
    memory=None,                     # Memory listesi (§13)
)
```

**En sık yapılan üç hata:**

1. **`description` boş bırakmak.** `SelectorGroupChat` konuşmacıyı seçerken
   `description` alanını okur. Yazmazsan seçici körleşir.
2. **`reflect_on_tool_use` anlamamak.** `False` ise tool sonucu ham döner
   (`ToolCallSummaryMessage`), model yorumlamaz — **bir LLM çağrısı tasarruf**.
   `True` ise model sonucu cümleye çevirir — daha okunur, bir çağrı daha pahalı.
3. **`max_tool_iterations=1` varsayılanını unutmak.** Ajan tek turda tek tool
   döngüsü yapar. Zincirleme tool kullanımı istiyorsan artır.

### Kendi ajanını yazmak

```python
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage, BaseChatMessage

class SayacAjani(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, "Mesajları sayar")
        self._n = 0

    @property
    def produced_message_types(self):
        return (TextMessage,)

    async def on_messages(self, messages, cancellation_token) -> Response:
        self._n += len(messages)
        return Response(chat_message=TextMessage(content=f"toplam {self._n}", source=self.name))

    async def on_reset(self, cancellation_token) -> None:
        self._n = 0
```

---

## §4 — Mesaj tipleri

İki aile var ve **farkı bilmek kritik**:

- **`BaseChatMessage`** → konuşmanın parçası, modele gider
- **`BaseAgentEvent`** → gözlem/telemetri, modele gitmez

| Chat mesajları | Olaylar (event) |
|---|---|
| `TextMessage` | `ToolCallRequestEvent` |
| `MultiModalMessage` (metin+`Image`) | `ToolCallExecutionEvent` |
| `StructuredMessage` (Pydantic) | `ModelClientStreamingChunkEvent` |
| `HandoffMessage` | `ThoughtEvent` (akıl yürütme) |
| `StopMessage` | `MemoryQueryEvent` |
| `ToolCallSummaryMessage` | `SelectSpeakerEvent` |
| | `CodeGenerationEvent`, `CodeExecutionEvent` |
| | `UserInputRequestedEvent` |

```python
result = await team.run(task="...")     # TaskResult: .messages, .stop_reason
async for m in team.run_stream(task="..."):   # canlı akış
    ...
await Console(team.run_stream(task="..."))    # hazır terminal arayüzü
```

---

## §5 — Model istemcileri

| İstemci | Import |
|---|---|
| OpenAI / OpenRouter / uyumlu | `autogen_ext.models.openai.OpenAIChatCompletionClient` |
| Azure OpenAI | `...openai.AzureOpenAIChatCompletionClient` |
| Anthropic | `autogen_ext.models.anthropic.AnthropicChatCompletionClient` |
| Ollama (yerel) | `autogen_ext.models.ollama.OllamaChatCompletionClient` |
| llama.cpp | `autogen_ext.models.llama_cpp` |
| **Önbellek sarmalayıcı** | `autogen_ext.models.cache.ChatCompletionCache` |
| **Test/replay** | `autogen_ext.models.replay.ReplayChatCompletionClient` |

```python
# OpenRouter (ücretsiz key ile de çalışır) — OpenAI uyumlu endpoint
client = OpenAIChatCompletionClient(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
```

**Katalogda olmayan modeller için `model_info` şart.** Aksi hâlde AutoGen tool
desteği olup olmadığını bilemez ve `ValueError: The model does not support function
calling.` verir:

```python
from autogen_core.models import ModelInfo
model_info = ModelInfo(vision=False, function_calling=True,
                       json_output=True, family="unknown", structured_output=True)
```

### Önbellek — geliştirirken parayı yakmamak için

```python
from autogen_ext.models.cache import ChatCompletionCache
from autogen_ext.cache_store.diskcache import DiskCacheStore
import diskcache

cached = ChatCompletionCache(client, DiskCacheStore(diskcache.Cache("./.llm-cache")))
```
Aynı prompt ikinci kez ücretsiz. Deneme-yanılma döngüsünde **kat kat** tasarruf.

### Replay — deterministik test

```python
from autogen_ext.models.replay import ReplayChatCompletionClient
client = ReplayChatCompletionClient(["birinci yanıt", "ikinci yanıt"], model_info=model_info)
```
POC'umuzun tamamı bunun üstünde koşuyor: anahtarsız, ağsız, tekrarlanabilir.
Tool çağrısı senaryolamak için listeye `CreateResult(finish_reason="function_calls", …)` koy.

---

## §6 — Bağlam yönetimi (`model_context`)

Uzun konuşmada bağlamı ne yapacağını **sen** seçersin:

| Sınıf | Davranış |
|---|---|
| `UnboundedChatCompletionContext` | Her şeyi tut (varsayılan) |
| `BufferedChatCompletionContext(buffer_size=n)` | Son n mesaj |
| `HeadAndTailChatCompletionContext(head, tail)` | Baş + son (ortayı at) |
| `TokenLimitedChatCompletionContext(token_limit=…)` | Token bütçesine göre kırp |

```python
from autogen_core.model_context import HeadAndTailChatCompletionContext
agent = AssistantAgent(..., model_context=HeadAndTailChatCompletionContext(head_size=2, tail_size=8))
```

> Bu, senin [08-baglam-basinci.md](../../report/08-baglam-basinci.md) ve
> [hybrid-compaction/](../../hybrid-compaction/) çalışmanın AutoGen'deki karşılığı.
> AutoGen burada **kırpma** sunuyor, **özetleme** sunmuyor — özetleyici bir
> `ChatCompletionContext` yazmak senin katkı alanın olabilir.

---

## §7 — Yapılandırılmış çıktı

```python
from pydantic import BaseModel

class Rapor(BaseModel):
    baslik: str
    bulgular: list[str]
    guven: float

agent = AssistantAgent("Analist", model_client=client, output_content_type=Rapor)
res = await agent.run(task="Analiz et")
rapor: Rapor = res.messages[-1].content        # StructuredMessage.content
```
Serbest metin ayrıştırmaktan **her zaman** daha iyi. Ajanlar arası veri
taşıyacaksan varsayılan yolun bu olsun.

---

## §8 — Tool'lar ve Workbench

### Üç kayıt yolu

```python
# 1) düz fonksiyon — docstring ve tip ipuçları şemaya çevrilir
def veri_getir(metrik: str) -> list[float]:
    """Metriğin serisini döndürür.
    Args:
        metrik: 'gelir' | 'hata_orani'
    """
    ...

# 2) açık FunctionTool
from autogen_core.tools import FunctionTool
arac = FunctionTool(veri_getir, description="Seri getirir")

# 3) Workbench — tool koleksiyonu (MCP bu yoldan gelir)
from autogen_core.tools import StaticWorkbench
wb = StaticWorkbench([arac])
agent = AssistantAgent("A", model_client=client, workbench=wb)
```

**Docstring şemadır.** `Args:` bölümünü yazmazsan model parametreleri tahmin eder.
Tool kalitesinin %80'i docstring kalitesi.

### Ajanı/takımı tool yapmak — kompozisyon

```python
from autogen_agentchat.tools import AgentTool, TeamTool

arastirma_araci = TeamTool(team=arastirma_takimi, name="arastir",
                           description="Çok-ajanlı derin araştırma yapar")
yonetici = AssistantAgent("Yonetici", model_client=client, tools=[arastirma_araci])
```

**Bu, süper agent sisteminin bel kemiği.** Bir takımı tool'a çevirince
hiyerarşi kurabilirsin: üst ajan alt takımı çağırır, alt takımın iç konuşması
üst bağlamı kirletmez. `SocietyOfMindAgent` de aynı işi ajan arayüzüyle yapar.

---

## §9 — Sonlandırma koşulları

**11 koşul var** ve `|` (VEYA) / `&` (VE) ile birleşiyorlar:

| Koşul | Ne zaman durur |
|---|---|
| `TextMentionTermination("BITTI")` | Metinde kelime geçince |
| `MaxMessageTermination(20)` | Mesaj sayısı |
| `TokenUsageTermination(...)` | Token bütçesi |
| `TimeoutTermination(60)` | Süre |
| `HandoffTermination(target="user")` | İnsana devredilince |
| `SourceMatchTermination(["Yazar"])` | Belirli ajan konuşunca |
| `StopMessageTermination()` | `StopMessage` gelince |
| `TextMessageTermination()` | Herhangi bir metin mesajı |
| `FunctionCallTermination("kaydet")` | Belirli tool çağrılınca |
| `FunctionalTermination(fn)` | Kendi fonksiyonun |
| `ExternalTermination()` | Dışarıdan `.set()` — UI'daki "Durdur" düğmesi |

```python
durma = TextMentionTermination("RAPOR_TAMAM") | MaxMessageTermination(20) | TimeoutTermination(120)
```

> **Kural:** Üretimde tek koşul yazma. Her zaman bir **sigorta** ekle
> (`MaxMessageTermination` veya `TimeoutTermination`). Sonsuz ajan döngüsü
> gerçek bir fatura kalemidir.

---

## §10 — Takımlar: beş desen

| Takım | Sırayı kim belirler | Ne zaman |
|---|---|---|
| `RoundRobinGroupChat` | Sabit döngü | Adımlar sabit ve hepsi gerekli |
| `SelectorGroupChat` | LLM **veya** `selector_func` | Dinamik ama merkezî kontrol |
| `Swarm` | Ajanın kendisi (`Handoff`) | Uzmanlık devri, triyaj |
| `GraphFlow` | Önceden çizilmiş graf | Deterministik akış, **paralellik** |
| `MagenticOneGroupChat` | Orchestrator + görev defteri | Açık uçlu, web/dosya görevleri |

### Ölçülmüş maliyet farkı

Aynı görev, aynı ajanlar (POC ölçümü, replay modunda):

| desen | mesaj | LLM | token |
|---|---:|---:|---:|
| SelectorGroupChat (`selector_func`) | 8 | 5 | **204** |
| GraphFlow | 11 | 7 | 270 |
| RoundRobinGroupChat | 9 | 6 | 274 |
| Swarm | 14 | 7 | **334** |

**%63.7 fark.** Ödediğin şey *yönlendirme özerkliği*. Selector'ın
`selector_func`'ı sıfır token harcar; Swarm'da her devir bir tool çağrısı + bir
LLM turu yakar ve o tur iş üretmez.

### SelectorGroupChat — harness'ın beyni

```python
def secici(mesajlar) -> str | None:
    if not mesajlar: return "Arastirmaci"
    son = mesajlar[-1]
    if son.source == "Analist":
        return "Elestirmen" if "aykırı" in str(son.content) else "Yazar"
    return None      # None → LLM seçiciye düş

takim = SelectorGroupChat(
    [a, b, c], model_client=client,
    selector_func=secici,          # deterministik kısayol
    selector_prompt="...",         # LLM'e düşünce kullanılacak şablon
    allow_repeated_speaker=False,
    max_selector_attempts=3,
    termination_condition=durma,
)
```
`selector_func` **`None` dönerse LLM devralır** — yani kısayol, tam ikame olmak
zorunda değil. Bildiğin durumları kodla, bilmediklerini modele bırak.

### Swarm — handoff

```python
from autogen_agentchat.base import Handoff
Handoff(target="VeriUzmani").name    # → 'transfer_to_veriuzmani'  (küçük harf!)

triyaj = AssistantAgent("Triyaj", model_client=client, handoffs=["VeriUzmani"])
takim = Swarm([triyaj, uzman], termination_condition=HandoffTermination(target="user") | durma)
```
Devir bir mesaj değil **tool çağrısıdır**: `Handoff` ajana `transfer_to_x`
adında bir tool takar. `HandoffTermination(target="user")` ile insana devir
= Swarm'ın human-in-the-loop hâli.

### GraphFlow — paralellik ve join

```python
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

b = DiGraphBuilder()
for a in (arastirmaci, analist1, analist2, yazar): b.add_node(a)
b.add_edge(arastirmaci, analist1)
b.add_edge(arastirmaci, analist2)                       # fan-out (paralel)
b.add_edge(analist1, yazar, activation_group="analiz", activation_condition="all")
b.add_edge(analist2, yazar, activation_group="analiz", activation_condition="all")  # join
b.set_entry_point(arastirmaci)

takim = GraphFlow(participants=[...], graph=b.build(), termination_condition=durma)
```
Diğer dört desende bir anda tek ajan konuşur. **GraphFlow gerçek eşzamanlılık
verir.** `activation_condition="all"` bir bariyerdir: Yazar iki analisti de bekler.

> Paralel dallarda **ajan başına ayrı model istemcisi** kullan. Tek replay
> istemcisi paylaşılırsa yanıtları hangi sırayla tüketecekleri belirsizleşir —
> ve gerçek hayatta da ucuz model → analiz / güçlü model → yazım ayrımı
> yapabilirsin.

### Magentic-One — genelci sistem

```python
from autogen_agentchat.teams import MagenticOneGroupChat
takim = MagenticOneGroupChat([surfer, coder, executor], model_client=client,
                             max_stalls=3, max_turns=20)
```
Orchestrator bir **görev defteri** (task ledger) tutar, plan yapar, ilerlemeyi
izler, takılırsa (`max_stalls`) planı yeniler. Hazır ajanları için
`autogen_ext.teams.magentic_one` (`pip install "autogen-ext[magentic-one]"`,
`playwright install`).

---

## §11 — İnsan döngüde

```python
# 1) Basit: UserProxyAgent
user = UserProxyAgent("insan", input_func=input)          # veya async input_func

# 2) Swarm'da: insana devret ve dur
takim = Swarm([a, b], termination_condition=HandoffTermination(target="user"))
res = await takim.run(task="...")
# insan cevabını al, aynı takımı devam ettir:
res = await takim.run(task=HandoffMessage(source="user", target="Triyaj", content=cevap))

# 3) Dışarıdan durdurma (UI düğmesi)
dur = ExternalTermination()
takim = RoundRobinGroupChat([...], termination_condition=dur)
# başka bir coroutine'den: dur.set()
```

`run()`'ı **task vermeden** çağırırsan takım kaldığı yerden devam eder.
Duraklat/devam et için `await team.pause()` / `await team.resume()`.

---

## §12 — Kod yürütme ve sandbox

**Dört yürütücü var.** Güvenlik sırasına göre:

| Yürütücü | İzolasyon | Import |
|---|---|---|
| `LocalCommandLineCodeExecutor` | **YOK** — kodu makinende çalıştırır | `autogen_ext.code_executors.local` |
| `JupyterCodeExecutor` | Süreç izolasyonu, durum korur | `...code_executors.jupyter` |
| **`DockerCommandLineCodeExecutor`** | Konteyner | `...code_executors.docker` |
| `DockerJupyterCodeExecutor` | Konteyner + durumlu kernel | `...code_executors.docker_jupyter` |
| `ACADynamicSessionsCodeExecutor` | Azure yönetilen sandbox | `...code_executors.azure` |

```python
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent

async with DockerCommandLineCodeExecutor(
    image="python:3.12-slim", work_dir="./calisma", timeout=60,
) as executor:
    kodcu = CodeExecutorAgent(
        "Kodcu", code_executor=executor, model_client=client,
        max_retries_on_error=2,           # hata alırsa kendi düzeltir
        supported_languages=["python", "sh"],
        approval_func=onay_fonksiyonu,    # ← her çalıştırmadan önce onay
    )
```

**`approval_func` üretimde zorunlu saymalısın.** İmzası `ApprovalRequest → ApprovalResponse`:

```python
from autogen_agentchat.agents import ApprovalRequest, ApprovalResponse

def onay_fonksiyonu(req: ApprovalRequest) -> ApprovalResponse:
    yasak = ("rm -rf", "curl", "os.system", "subprocess")
    if any(k in req.code for k in yasak):
        return ApprovalResponse(approved=False, reason="tehlikeli çağrı")
    return ApprovalResponse(approved=True, reason="ok")
```

> **Asla `LocalCommandLineCodeExecutor` ile üretime çıkma.** LLM'in ürettiği kodu
> izolasyonsuz çalıştırmak, prompt injection'ı doğrudan kabuk erişimine çevirir.
> Docker + `approval_func` + `timeout` üçlüsü asgari kurulumdur.

---

## §13 — Bellek

`Memory` bir **protokol**: `add`, `query`, `update_context`, `clear`.
Ajana verdiğinde her turdan önce `update_context` çalışır ve bağlama enjekte eder.

| Uygulama | Kullanım |
|---|---|
| `ListMemory` (core) | Basit, sıralı, kalıcı değil |
| `ChromaDBVectorMemory` | Vektör arama, yerel/kalıcı |
| `Mem0Memory` | Yönetilen bellek servisi |
| `RedisMemory` | Redis + RedisVL |
| `TextCanvasMemory` | **Paylaşılan yazılabilir tuval** — ajanlar aynı belgeyi düzenler |

```python
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType

bellek = ListMemory()
await bellek.add(MemoryContent(content="Kullanıcı metrik birimi olarak TL istiyor",
                               mime_type=MemoryMimeType.TEXT))
agent = AssistantAgent("A", model_client=client, memory=[bellek])
```

`TextCanvasMemory` özellikle ilginç: çok-ajanlı **ortak doküman** üretimi için
tasarlanmış, `unidiff` ile fark takibi yapıyor. Rapor yazan bir takım için
mesaj thread'inden daha doğru bir soyutlama.

---

## §14 — MCP: dış dünyayla asıl köprü

**Bu bölüm senin süper agent sistemin için en kritik parça.**

### AutoGen bir MCP istemcisi olarak

Üç taşıma katmanı: `StdioServerParams`, `SseServerParams`, `StreamableHttpServerParams`.

```python
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

params = StdioServerParams(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/veri"])
async with McpWorkbench(params) as wb:
    agent = AssistantAgent("DosyaAjani", model_client=client, workbench=wb)
    await Console(agent.run_stream(task="/veri altındaki csv'leri özetle"))
```

HTTP tabanlı sunucular için (bizim `.mcp.json`'daki DeepWiki gibi):

```python
from autogen_ext.tools.mcp import StreamableHttpServerParams
params = StreamableHttpServerParams(url="https://mcp.deepwiki.com/mcp")
```

Tek tek tool almak istersen: `mcp_server_tools(params)` → `[StdioMcpToolAdapter, …]`.
Bunları normal `tools=[...]` listesine koyabilirsin.

**`Workbench` vs `tools`:** Workbench oturumu yönetir (bağlantı, yeniden bağlanma,
tool listesinin dinamik değişmesi). MCP için **her zaman Workbench kullan**.

### AutoGen'i MCP sunucusu yapmak

Kütüphanede hazır bir "expose as MCP server" yok (ADK'da var — bkz.
[01 §ADK](01-autogen-kaynak-haritasi.md)). Kendin sararsın:

```python
# autogen_mcp_server.py — takımı tek tool olarak dışarı aç
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("autogen-analiz")

@mcp.tool()
async def analiz_yap(metrik: str) -> str:
    """Çok-ajanlı analiz takımını koşturur ve raporu döndürür."""
    takim = takim_kur()
    res = await takim.run(task=f"{metrik} serisini analiz et")
    return str(res.messages[-1].content)

if __name__ == "__main__":
    mcp.run()          # stdio
```

Bu sunucuyu OpenClaw'a `openclaw mcp add` ile tanıtırsan, WhatsApp'tan gelen bir
mesaj AutoGen takımını tetikler. Claude Code'a `claude mcp add` ile de bağlanır.
**Sistemini dış dünyaya açmanın en ucuz yolu bu.**

---

## §15 — Skill'ler: AutoGen'de karşılığı ne?

AutoGen'in **`skill` diye bir birinci sınıf soyutlaması yok.** Claude Code'daki
`SKILL.md` ya da OpenClaw'un `skills/` klasörünün doğrudan karşılığı bulunmuyor.
Karşılığını üç yoldan biriyle sen kurarsın:

| Skill nedir | AutoGen karşılığı |
|---|---|
| Yeniden kullanılabilir **talimat** | `system_message` + `description` (ajan profili) |
| Yeniden kullanılabilir **yetenek** | `FunctionTool` / `Workbench` (MCP) |
| Yeniden kullanılabilir **ekip** | `TeamTool` / `SocietyOfMindAgent` |
| **Bildirimsel** tanım (dosyadan yükle) | **Component config** — aşağıda |

### Component config — skill'e en yakın mekanizma

AutoGen'in her bileşeni (ajan, takım, tool, model istemcisi, termination)
`Component` protokolünü uygular: **JSON'a serileştirilip geri yüklenebilir.**

```python
cfg = agent.dump_component()          # → ComponentModel (JSON'lanabilir)
open("skills/analist.json","w").write(cfg.model_dump_json(indent=2))

# sonra, başka bir süreçte:
from autogen_agentchat.agents import AssistantAgent
agent = AssistantAgent.load_component(json.load(open("skills/analist.json")))
```

Bu, **dosyadan yüklenen ajan/takım** demek — yani senin "skill" dediğin şeyin
altyapısı. Bir `skills/` klasörü kurup her dosyayı bir ajan/takım tanımı yapabilir,
çalışma anında keşfedip yükleyebilirsin. AutoGen Studio da tam olarak bunu kullanıyor.

---

## §16 — Durum ve kalıcılık

```python
durum = await takim.save_state()        # dict
json.dump(durum, open("oturum.json","w"))

# sonra:
await takim.load_state(json.load(open("oturum.json")))
await takim.run()                        # task vermeden → kaldığı yerden
```

Ajan seviyesinde de var: `agent.save_state()` / `agent.load_state()`.
Ayrıca `await takim.reset()` (temiz başlangıç), `pause()`, `resume()`.

**Uyarı:** Bu bir *snapshot* mekanizması; LangGraph'ın checkpointer'ı gibi
her düğümde otomatik yazmıyor. Kalıcılığı **sen** tetiklersin. Uzun süren
işlerde her turdan sonra `save_state` çağırmak sana düşer.

---

## §17 — `autogen_core`: aktör modeli

Tutorial'ların bittiği yer burası; AutoGen'in v0.4'te baştan yazılma sebebi.

```python
from dataclasses import dataclass
from autogen_core import (RoutedAgent, message_handler, MessageContext,
                          SingleThreadedAgentRuntime, TopicId, AgentId, type_subscription)

@dataclass
class Is:
    metrik: str

@type_subscription(topic_type="analiz")          # birden fazla dekoratör = birden fazla abonelik
class Isci(RoutedAgent):
    def __init__(self): super().__init__("işçi")

    @message_handler                              # ← parametre adı ZORUNLU: message, ctx
    async def calis(self, message: Is, ctx: MessageContext) -> None:
        ...

runtime = SingleThreadedAgentRuntime()
await Isci.register(runtime, "isci", lambda: Isci())
runtime.start()
await runtime.publish_message(Is("gelir"), TopicId("analiz", "default"))   # pub/sub
sonuc = await runtime.send_message(Istek(), AgentId("rapor", "default"))   # RPC
await runtime.stop_when_idle()
await runtime.close()
```

**Anahtar kavramlar:**

| Kavram | Ne |
|---|---|
| `RoutedAgent` | Mesajı **tipine göre** handler'a yönlendiren taban |
| `@message_handler` / `@event` / `@rpc` | Handler kaydı (event = yanıtsız, rpc = yanıtlı) |
| `TypeSubscription` / `@type_subscription` | Topic aboneliği |
| `TypePrefixSubscription` | Önek eşleşmeli abonelik (çok kiracılı) |
| `AgentId(type, key)` | Aktör kimliği — `key` ile kiracı/oturum ayrımı |
| `ClosureAgent` | Sınıf yazmadan fonksiyonla aktör — sonuç toplamak için ideal |
| `InterventionHandler` | **Mesaj hattına müdahale** — denetim, filtreleme, `DropMessage` |
| `CancellationToken` | İptal yayılımı |

### Dağıtık: gRPC

```python
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime, GrpcWorkerAgentRuntimeHost

host = GrpcWorkerAgentRuntimeHost(address="localhost:50051"); host.start()
worker = GrpcWorkerAgentRuntime(host_address="localhost:50051"); await worker.start()
```
Aynı ajan kodu, farklı makinelerde. Resmi örnek:
`python/samples/core_distributed-group-chat` ve `core_grpc_worker_runtime`.
Diller arası bile çalışıyor (`core_xlang_hello_python_agent`).

### InterventionHandler — denetim katmanı

```python
from autogen_core import DefaultInterventionHandler, DropMessage

class Denetci(DefaultInterventionHandler):
    async def on_publish(self, message, *, message_context):
        if yasakli(message): return DropMessage()
        kaydet(message)
        return message

runtime = SingleThreadedAgentRuntime(intervention_handlers=[Denetci()])
```
Güvenlik, kota, audit log — hepsi buraya. AgentChat katmanında bunun karşılığı yok.

---

## §18 — Gözlemlenebilirlik

```python
import logging
from autogen_core import TRACE_LOGGER_NAME, EVENT_LOGGER_NAME, ROOT_LOGGER_NAME

logging.getLogger(TRACE_LOGGER_NAME).setLevel(logging.DEBUG)   # insan okuyacak
logging.getLogger(EVENT_LOGGER_NAME).setLevel(logging.INFO)    # yapılandırılmış olay
```

**OpenTelemetry** yerleşik: `trace_create_agent_span`, `trace_invoke_agent_span`,
`trace_tool_span`. `SingleThreadedAgentRuntime(tracer_provider=...)` ile bağlarsın.
Langfuse / AgentOps (ikisi de senin starlarında) bu yoldan takılıyor.

---

## §19 — AutoGen Studio & AutoGenBench

```bash
pip install autogenstudio && autogenstudio ui --port 8081
```
Kodsuz ajan/takım kurma arayüzü. Alt yapısı §15'teki component config —
Studio'da kurup JSON'unu dışa aktarabilir, koduna gömebilirsin.

`AutoGenBench` ise tekrarlı/izole ajan benchmark koşumu için (Magentic-One
makalesinde bunun üstünde ölçüldü).

---

## §20 — Tuzaklar (ölçülmüş)

Bunlar teorik değil; bu oturumda **koşarak** bulundu.

### 20.1 Aktör modeli veriyi korumuyor 🔴

`_process_publish` abonelerin handler'larını `asyncio.gather` ile bekliyor. Bir
handler exception fırlatınca gather **hemen** dönüyor ve arkasından `task_done()`
çağrılıyor → kuyruk, kardeş handler'lar hâlâ çalışırken "boşaldı" sayılıyor.
`stop_when_idle()` erken dönüyor; ardından `close()` çağrılırsa **yarım kalan
sağlam ajanların sonuçları kayboluyor**. Ne exception yükseliyor ne uyarı çıkıyor.

Üç koşuda birebir tekrarlandı — kanıt: [`../poc/desen_5_core_aktor.py`](../poc/desen_5_core_aktor.py).

**Korunma:** Kritik yayınlarda (a) her handler'ı kendi `try/except`'iyle sar,
(b) çöken ajanları ayrı topic'e taşı, (c) beklenen sonuç sayısını **say ve doğrula**.
Bariyerin tuttuğuna güvenme.

### 20.2 `mcp` bağımlılığı üst sınırsız 🟠

`autogen-ext` `mcp>=1.11.0` diyor. MCP SDK 2.0 çıkınca:
```
ImportError: cannot import name 'RequestContext' from 'mcp.shared.context'
```
`requirements.txt`'e **`mcp>=1.24,<2`** yaz. (Karşılaştırma: `google-adk` aynı
bağımlılığı `mcp>=1.24,<2` diye sınırlamış — aktif proje üst sınır koymuş,
bakım modundaki koymamış.)

### 20.3 Handoff tool adı küçük harfe düşüyor 🟡

`Handoff(target="VeriUzmani").name` → `transfer_to_veriuzmani`. Elle yazarsan
sessizce `tool not found` alırsın (hata mesajı tool sonucunda gömülü kalır).
Her zaman `Handoff(target=X).name` ile üret.

### 20.4 Diğerleri

- `reflect_on_tool_use=True` **her tool çağrısına bir LLM turu** ekler — token faturası.
- `description` boşsa `SelectorGroupChat` kör seçim yapar.
- `max_tool_iterations=1` varsayılan; zincirleme tool kullanımı sessizce kesilir.
- Paralel dallarda ajanlar tek model istemcisini paylaşırsa yanıt sırası belirsizleşir.

---

## §21 — Süper agent sistemi: blueprint

Yukarıdaki parçaları birleştiren hedef mimari:

```
                        ┌──────────────── DIŞ DÜNYA ────────────────┐
                        │ OpenClaw (WhatsApp/Telegram/Slack)        │
                        │ Claude Code · başka MCP istemcileri       │
                        └───────────────────┬───────────────────────┘
                                            │ MCP (stdio/http)
                        ┌───────────────────▼───────────────────────┐
                        │  autogen_mcp_server.py   (§14)            │
                        │  takımları tool olarak dışa açar          │
                        └───────────────────┬───────────────────────┘
                                            │
        ┌───────────────────────────────────▼────────────────────────────────┐
        │  ORKESTRASYON  ·  SelectorGroupChat (selector_func + LLM yedeği)   │
        │                                                                    │
        │   Triyaj ──► [AraştırmaTakımı]  ──► [AnalizGrafı]  ──► Yazar       │
        │              TeamTool (§8)         GraphFlow, paralel (§10)        │
        └───────┬────────────────┬───────────────────┬───────────────────────┘
                │                │                   │
        ┌───────▼──────┐ ┌───────▼────────┐ ┌────────▼─────────┐
        │ MCP Workbench│ │ Kod sandbox'ı  │ │ Bellek           │
        │ dosya, web,  │ │ Docker +       │ │ ChromaDB (uzun)  │
        │ DeepWiki…    │ │ approval_func  │ │ TextCanvas (ortak)│
        └──────────────┘ └────────────────┘ └──────────────────┘
                                │
        ┌───────────────────────▼────────────────────────────────────────────┐
        │  ÇEKİRDEK  ·  SingleThreadedAgentRuntime → GrpcWorkerAgentRuntime   │
        │  InterventionHandler (audit/kota) · OpenTelemetry · save_state      │
        └────────────────────────────────────────────────────────────────────┘
```

**Kurulum sırası — her adım öncekinin üstüne biner:**

| # | Adım | Bölüm | Neden bu sırada |
|---|---|---|---|
| 1 | Model istemcisi + **önbellek** + replay | §5 | Geliştirirken para ve determinizm |
| 2 | Tool'lar, docstring disiplini | §8 | Ajan ancak tool'u kadar iyi |
| 3 | Yapılandırılmış çıktı | §7 | Ajanlar arası veri sözleşmesi |
| 4 | Tek ajan → çalışan tek görev | §3 | Takım kurmadan önce ajanı doğrula |
| 5 | Takım + **iki katmanlı** termination | §9,10 | Sigortasız takım fatura üretir |
| 6 | `selector_func` ile yönlendirme | §10 | En ucuz desen, ölçüldü |
| 7 | Alt takımları `TeamTool` ile paketle | §8 | Bağlam kirlenmesini önler |
| 8 | MCP Workbench | §14 | Dış dünya buradan giriyor |
| 9 | Docker sandbox + `approval_func` | §12 | Kod yürütme olmadan "agent" eksik |
| 10 | Bellek | §13 | Oturumlar arası süreklilik |
| 11 | `save_state` + `load_state` | §16 | Çökmeden sonra devam |
| 12 | Component config ile "skill" klasörü | §15 | Ajanları koddan ayır |
| 13 | InterventionHandler + OTel | §17,18 | Denetim ve gözlem |
| 14 | gRPC ile dağıtım | §17 | Ancak ölçek gerekiyorsa |
| 15 | MCP sunucusu olarak dışa aç | §14 | OpenClaw/Claude Code'a bağlan |

**Bir uyarı:** 14. adıma çoğu proje hiç gelmez ve gelmemeli. Aktör modelinin
dağıtım yeteneği etkileyici ama §20.1'deki bulgu, o katmanın da denetim
gerektirdiğini gösteriyor. Önce 1-13 sağlam çalışsın.

---

## §22 — Kaynaklar

### Resmî — kod ve doküman

| Kaynak | Link |
|---|---|
| Repo (bakım modu uyarısı README'de) | https://github.com/microsoft/autogen |
| Stable doküman | https://microsoft.github.io/autogen/stable/ |
| **v0.2 → v0.4 migration guide** | https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html |
| AutoGen → MAF migration guide | https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/ |
| Microsoft Agent Framework (halef) | https://github.com/microsoft/agent-framework · https://learn.microsoft.com/en-us/agent-framework/overview/ |
| AG2 (v0.2 kolundan fork, aktif) | https://github.com/ag2ai/ag2 |
| AutoGen Studio | https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/ |
| Tasarım notları | `microsoft/autogen/docs/design/` |

> Migration guide'ı hafife alma: v0.2'den v0.4'e **neyin neden değiştiğini**
> anlatan tek belge o. Mimari kırılmayı en hızlı orada kavrarsın.

### Resmî örnekler — `microsoft/autogen/python/samples/`

Bunlar el kitabının kod karşılığı, sırayla okunacak liste:

| Örnek | Ne öğretir |
|---|---|
| `agentchat_fastapi` | Takımı HTTP servisine sarma |
| `agentchat_streamlit` / `agentchat_chainlit` | UI bağlama |
| `agentchat_chess_game` | İki ajan, sıkı kurallı oyun döngüsü |
| `agentchat_graphrag` | Graf tabanlı RAG entegrasyonu |
| `agentchat_dspy` | DSPy ile prompt optimizasyonu |
| `agentchat_azure_postgresql` | Kalıcı durum + bulut veritabanı |
| **`core_distributed-group-chat`** | Dağıtık grup sohbeti (gRPC) |
| **`core_grpc_worker_runtime`** | Worker runtime kurulumu |
| `core_semantic_router` | Çekirdek seviyede yönlendirme |
| `core_streaming_handoffs_fastapi` | Handoff + streaming + HTTP |
| `core_async_human_in_the_loop` | Asenkron insan onayı |
| `core_xlang_hello_python_agent` | **Diller arası** ajan iletişimi |
| `task_centric_memory` | Görev merkezli bellek deneyi |
| `gitty` | Gerçek bir yardımcı program |

### Makaleler

| Makale | Link | Neden |
|---|---|---|
| **AutoGen** (Wu, Bansal, Zhang, Wang et al., 2023) | [2308.08155](https://arxiv.org/abs/2308.08155) | Kurucu makale: konuşma-merkezli programlama modeli |
| **Magentic-One** (2024) | [2411.04468](https://arxiv.org/abs/2411.04468) | Orchestrator + görev defteri deseni |
| **AutoGen Studio** (EMNLP 2024) | [2408.15247](https://arxiv.org/abs/2408.15247) | Kodsuz araç ve component config |
| **Why Do Multi-Agent LLM Systems Fail?** (NeurIPS 2025) | [2503.13657](https://arxiv.org/abs/2503.13657) | **MAST**: 7 framework, 1600+ trace, 14 hata modu. §20 bulgularının teorik çerçevesi |

### Eğitim repoları

| Repo | Değerlendirme |
|---|---|
| [mayank953/Youtube — Autogen Crash Course](https://github.com/mayank953/Youtube/tree/main/Agentic%20AI/Autogen%20Crash%20Course) | **Taradım.** 12 modül, saf v0.4+ API (v0.2 mirası yok). AgentChat seviyesi için iyi rampa. **Eksikleri:** Swarm/Handoff, kod yürütme, Memory, MCP, `autogen_core`. Yani bu el kitabının §12-17'si orada yok. Lisans dosyası yok |
| [jkmaina/autogen_blueprint](https://github.com/jkmaina/autogen_blueprint) | Kitap eşlikçisi, geniş kapsam |
| [HumbertoFelipe/autogen-examples](https://github.com/HumbertoFelipe/autogen-examples) | Karışık projeler + tutorial'lar |
| [Poly186-AI-DAO/AutoGen-Example-Scripts](https://github.com/Poly186-AI-DAO/AutoGen-Example-Scripts) | Notebook'lardan sadeleştirilmiş script'ler |

> Üçüncü parti tutorial'larda **sürüm kontrolü yap**: `ConversableAgent` ve
> `initiate_chat` görüyorsan o kaynak v0.2 ya da AG2 anlatıyor, bu el kitabıyla
> uyumlu değil. Doğru işaret: `autogen_agentchat` / `autogen_core` import'ları.

### Bu repodaki ilgili çalışmalar

- [01 — AutoGen kaynak haritası](01-autogen-kaynak-haritasi.md) — bakım modu bulgusu, proje çerçevesi
- [report/14 — Agentic mega-atlas](../../report/14-agentic-mega-atlas.md) — AutoGen'in rakiplerine karşı konumu
- [../poc/](../poc/) — beş desenin çalışan POC'u ve ölçümleri
- [report/08 — Bağlam basıncı](../../report/08-baglam-basinci.md) — §6'daki `model_context` konusunun teorisi

---

*Bu belgedeki API isimleri AutoGen v0.7.5'ten introspection ile doğrulandı.
Yeni bir sürüme geçerken önce `dir()` ile kontrol et — bakım modundaki bir
kütüphanede doküman ile kod arasındaki fark zamanla büyür.*
