"""İki wiki'nin bölüm metinleri: AutoGen+MAF ve OpenClaw.

`make_wiki.py` motoru taşıyor (şema gömme, `.excalidraw` üretimi, doğrulama);
burası yalnız metin. Ayrı dosya, çünkü üç belgenin gövdesi tek dosyada bin
satırı geçiyordu ve motoru okumak isteyen kişi metinlerin içinde kayboluyordu.

`bind()` ile motorun `svg()` yardımcısı ve `figures` modülü enjekte ediliyor —
bu dosya `make_wiki`'yi içe aktarmıyor, yoksa döngüsel bağımlılık olurdu.
"""

from __future__ import annotations

from typing import Any, Callable

svg: Callable[[str, str], str]
figures: Any


def bind(svg_fn: Callable[[str, str], str], figures_mod: Any) -> None:
    global svg, figures
    svg, figures = svg_fn, figures_mod


# ══════════════════════════════════════════════ 26 — AutoGen + MAF
def autogen_maf() -> str:
    return f"""# AutoGen ve MAF — çerçeve wiki'si

> **Bu ne:** AutoGen'i kullanacak ya da MAF'a geçmeyi düşünen bir mühendis için.
> Tek dosya, arayarak okunmak için.
>
> **Sürümler:** `autogen-core` / `agentchat` / `ext` **0.7.5** · `agent-framework`
> **1.14.0** — ikisi de kurulu ve ölçüldü.
>
> **Etiketler:** **[ölçüldü]** koşturuldu · **[kaynak]** birincil metinden ·
> **[teyitsiz]** okundu, koşturulmadı.

---

## İçindekiler

1. [Dört isim, tek karmaşa](#s1)
2. [Üç katman](#s2)
3. [Aktör modeli](#s3)
4. [Kimlik: bir şey değil, iki şey](#s4)
5. [İki iletişim biçimi](#s5)
6. [Tool döngüsü ve tarif](#s6)
7. [Beş takım tipi ve faturaları](#s7)
8. [Durmayı öğretmek](#s8)
9. [Sekiz resmî desen](#s9)
10. [Built-in tool'lar — ve neden yok](#s10)
11. [Kod yürütücüler](#s11)
12. [Ölçülmüş tuzaklar](#s12)
13. [MAF: halef ne getirdi](#s13)
14. [MAF: ne kaybettirdi](#s14)
15. [Geçiş haritası](#s15)

---

## 1 · Dört isim

Karıştırılan dört ayrı şey var:

| İsim | Ne | Durum |
|---|---|---|
| **microsoft/autogen v0.4+** | `autogen-core` + `agentchat` + `ext` | Bakım modu, **0.7.5** |
| AutoGen v0.2 | `ConversableAgent`, `initiate_chat` | Terk edilmiş |
| **ag2ai/ag2** | v0.2 kolundan fork, ayrı ekip | Aktif — ama `pip install ag2` artık `import autogen` **sunmuyor** |
| **microsoft/agent-framework** | AutoGen + Semantic Kernel birleşimi | Resmî halef, **1.14.0** |

> **Filtre:** Bir kaynakta `ConversableAgent` ya da `initiate_chat` görüyorsan
> o kaynak v0.2 ya da AG2 anlatıyor — bu sürümle **uyumsuz**.

---

## 2 · Üç katman

{svg("f_layers", "AutoGen'in üç katmanı")}

Ayıran şey **`autogen_core`**: ajanlar gerçekten aktör — kendi mailbox'ı olan,
mesajı **tipe göre** yönlendiren, makinelere dağıtılabilen birimler.

LangGraph'ın graf yürütücü + checkpointer'ı **dayanıklılık** sağlıyor,
eşzamanlılık modeli değil. *"AutoGen mı LangGraph mı"* çoğu zaman yanlış
sorulmuş soru — farklı katmanlar.

---

## 3 · Aktör modeli

{svg("f_actor", "Ajan ajanı çağırmıyor")}

Bir ajan başka bir ajanın nesnesini tutmuyor; runtime'a mesaj veriyor.

**Bedeli:** *"kim kimi çağırdı"* yığın izinde görünmüyor.
**Karşılığı:** yeni ajan eklemek çağıran kodu değiştirmiyor · bütün mesajlar tek
noktadan geçtiği için müdahale ve ölçüm oraya takılıyor · aynı sınıftan istediğin
kadar örnek bedava.

---

## 4 · Kimlik

{svg("f_identity", "AgentId = tip + anahtar")}

`AgentId(type, key)` — **iki parçalı**. Ve en az konuşulan, en çok işe yarayan
mekanizma şu:

> **Topic kaynağı, ajan anahtarına dönüşüyor.**
> `TopicId("tur", "oturum-42")`'ye yayın yapmak `AgentId("session", "oturum-42")`
> ajanını **yaratıyor** — oturum başına izole örnek, elle sözlük tutmadan.

Bu projede gateway oturumları tam olarak böyle çalışıyor. Ölçek gerektiğinde
ilk bakılacak yer burası.

---

## 5 · İki iletişim biçimi

{svg("f_send_vs_publish", "Doğrudan ve yayın — asimetri hatada")}

| | `send_message` | `publish_message` |
|---|---|---|
| Alıcı | tek `AgentId` | topic'e abone olan herkes |
| Dönüş | **var** | **yok** |
| Handler çökerse | çağırana **fırlatır** | **loglanır, fırlatmaz** |

Son satır bir tasarım kararı: sonucu bekleyeceksen doğrudan, olay duyuracaksan
yayın.

### Ve buradan doğan en pahalı arıza

{svg("f_fanout", "Fan-out / fan-in — ve sessiz kayıp")}

Çöken bir handler `_process_publish` içindeki `gather`'ı erken döndürüyor,
`stop_when_idle()` bariyeri erken açılıyor, ve **tamamlanmış kardeş sonuçlar
sessizce kayboluyor.**

Aynı arıza enjeksiyonuyla ölçüldü **[ölçüldü]**:

| Motor | Temiz | Sarmalayıcı arkasında | Ham hata |
|---|---:|---:|---:|
| GraphFlow | 3 | 2 | **0–1**, süre sınırı dolar |
| core pub/sub + `ClosureAgent` kuyruğu | 3 | 2 | **2**, ~3 ms |

> Resmî desenler bu konuda **birbiriyle çelişiyor**: *Concurrent Agents* kuyrukla
> topluyor, *Mixture of Agents* `asyncio.gather(...)` ile — sessiz kaybın kaynağı
> olan yapı.

---

## 6 · Tool döngüsü

{svg("f_tool_loop", "Model ister · kapı · çalışır · sonucu görür · döngü")}

### Tarif = arayüz

Model fonksiyonu görmüyor; **adını, tarifini ve parametre şemasını** görüyor.
`description` prompt'a giren tek metin, ve modelin o tool'a *ne zaman*
uzanacağına karar verdiği şey o.

### Varsayılan tavan — altı çerçeve, altı cevap

Hepsi kurulu paketten okundu **[ölçüldü]**:

| Çerçeve | Alan | Varsayılan |
|---|---|---:|
| **AutoGen** | `max_tool_iterations` | **1** |
| OpenAI Agents SDK | `Runner.run(max_turns=)` | 10 |
| CrewAI | `Agent.max_iter` | 25 |
| **MAF** | `DEFAULT_MAX_ITERATIONS` | **40** |
| LangGraph | `recursion_limit` | 10007 |
| Google ADK | `LoopAgent.max_iterations` | sınırsız |

AutoGen'de **1**: ajan tool'u çağırır, sonucu görür, **durur** — ve hata vermez.
Microsoft bunu göç kılavuzunda kendisi yazıyor **[kaynak]**.

---

## 7 · Beş takım

{svg("f_teams", "Değişen tek şey: sırayı kim belirliyor")}

Aynı görev, aynı ajanlar **[ölçüldü]**:

| Desen | Sırayı kim belirliyor | Mesaj | LLM | Tool | Token |
|---|---|---:|---:|---:|---:|
| **SelectorGroupChat** | model her turda seçiyor | 8 | 5 | 2 | **204** |
| GraphFlow | önceden çizilmiş DAG | 11 | 7 | 3 | 270 |
| RoundRobinGroupChat | sırayla | 9 | 6 | 2 | 274 |
| **Swarm** | ajan devrediyor | 14 | 7 | 4 | **334** |

**%63,7 fark.** Ödenen zekâ değil **yönlendirme özerkliği**.

### GraphFlow — boruyu çizmek

{svg("f_graphflow", "DiGraphBuilder ile akış")}

Kenarlar **veri taşımıyor**, yalnız sırayı belirliyor. Join'de
`activation_condition="all"` demezsen ilk gelen dal akışı ilerletiyor.

---

## 8 · Durmayı öğretmek

{svg("f_termination", "On bir sonlandırma koşulu")}

Sonlandırma koşulu olmayan takım **sonsuza kadar** konuşuyor, ve fatura gerçek.
On bir koşul var; en çok kullanılan dördü:

* `MaxMessageTermination` — mesaj sayar
* `TokenUsageTermination` — **token sayar**, faturaya en yakın olan
* `TimeoutTermination` — süre
* `TextMentionTermination` — bir kelime geçince

> Koşullar `&` ve `|` ile birleşiyor. Yalnız mesaj sayan bir koşul, uzun
> cevaplarla dolu bir turu ucuz sanıyor.

---

## 9 · Sekiz desen

{svg("f_patterns", "Resmî sekiz orkestrasyon deseni")}

Eşzamanlı · Sıralı · Group Chat · Handoff · Mixture of Agents · Münazara ·
Yansıma · Kod yürütme.

Bunlar **kütüphane değil, tarif**. Hiçbiri `import` edilmiyor; kılavuzda kodla
anlatılan yapılar.

---

## 10 · Built-in tool'lar — ve neden yok

{svg("f_tools_component", "Tool: fonksiyon + şema")}

En sık yanılınan yer. **AutoGen hazır tool ile gelmiyor.** `autogen_ext.tools`
altında yedi modül var ve altısı **adaptör** — tool değil **[ölçüldü]**:

| Modül | Ne veriyor | Kurulu mu |
|---|---|---|
| `code_execution` | `PythonCodeExecutionTool` — **tek gerçek tool** | ✔ |
| `mcp` | `StdioMcpToolAdapter` · `SseMcpToolAdapter` · `McpWorkbench` | ✔ |
| `langchain` | `LangChainToolAdapter` — LangChain tool'unu sarmalıyor | ✔ |
| `azure` | Azure AI Search adaptörü | ekstra gerekiyor |
| `graphrag` | GraphRAG adaptörü | ekstra gerekiyor |
| `http` | HTTP çağrısı tool'u | ekstra gerekiyor |
| `semantic_kernel` | SK tool adaptörü | ekstra gerekiyor |

`autogen-ext`'in **38 ayrı ekstrası** var (`docker`, `grpc`, `http-tool`,
`file-surfer`, `jupyter-executor`…). Yetenekler paket içinde değil, **kurulum
seçeneklerinde**.

### Tool'u kim veriyor — üç sistem

| | Hazır tool | Nereden |
|---|---:|---|
| **AutoGen** | **~1** | Kendin yazıyorsun |
| **MAF** | 6 hosted sözleşme | `SupportsCodeInterpreterTool` · `SupportsWebSearchTool` · `SupportsFileSearchTool` · `SupportsImageGenerationTool` · `SupportsShellTool` · `SupportsMCPTool` **[ölçüldü]** |
| **OpenClaw** | **51** (44'ü canlı) | Çekirdekte, 11 grupta |

> **Sonuç:** AutoGen bir **motor**, bir asistan değil. Tool yazmak kullanan
> tarafın işi. Bu bir eksiklik değil, bir kapsam tercihi — ama kurulumdan sonra
> hazır yetenek bekleyen bir plan buna göre düzeltilmeli.

### Tool nasıl yazılıyor

Bir fonksiyon + docstring yetiyor; şema **imzadan** çıkarılıyor:

```python
def scan_facts(query: str) -> str:
    "Son taramanın özetini döndürür."      # docstring = tarif = arayüz
    ...

FunctionTool(scan_facts, description=scan_facts.__doc__)
```

Modele giden fonksiyon değil, **şeması**:

```json
{{"name": "scan_facts",
 "description": "Son taramanın özetini döndürür.",
 "parameters": {{"type": "object",
                "properties": {{"query": {{"type": "string"}}}},
                "required": ["query"]}}}}
```

**Üç kural:**

1. **Docstring arayüzdür** — modelin o tool'a *ne zaman* uzanacağına karar
   verdiği tek metin. Dokümantasyon değil.
2. **Şemalar her turda ödeniyor.** 17 tool = her istekte 17 şema; `docs/06`
   bunun canlı bir zaman aşımına yol açtığını kaydediyor.
3. **Tip ipucu zorunlu.** `query: str` yoksa şema üretilemiyor.

### Workbench — liste değil, **kaynak**

{svg("f_workbench_component", "Üç kaynak, tek arayüz")}

```python
AssistantAgent(tools=[a, b], workbench=wb)
# ValueError: Tools cannot be used with a workbench.
```

İkisi aynı anda olamıyor, çünkü aynı soruyu **farklı zamanda** cevaplıyorlar.
Liste ajan yazılırken donuyor; kaynak her turda sorulabiliyor — MCP sunucusu
tool listesini çalışma zamanında verdiği için tek doğru soyutlama bu.

Kapıyı oraya koymanın sebebi: workbench, yerel bir Python fonksiyonuyla uzak bir
MCP tool'unu **aynı gören tek yer**. Kural, ajan yazılırken **var olmayan**
tool'lar için de geçerli oluyor.

### Model istemcileri

{svg("f_model_clients", "İstemci ve model_info")}

**Tuzak:** OpenAI-*uyumlu* bir endpoint kullanıyorsan `model_info` **zorunlu**.
Verilmezse hata net: `model_info is required when model name is not a valid
OpenAI model`. Azure, vLLM, Ollama, OpenRouter — hepsi bu kapsamda.

---

## 11 · Kod yürütücüler

{svg("f_code_executors", "Yerel · Docker · Jupyter")}

Resmî sekiz desenin sonuncusu **Code Execution**, ve diğer yedisinden farkı şu:
onlar orkestrasyon deseni, bu bir **yetenek**. Modelin yazdığı Python'u
çalıştırıyor.

### Dört yürütücü

`autogen_ext.code_executors` altında **[ölçüldü]**:

| Yürütücü | İzolasyon | Not |
|---|---|---|
| `local` | **yok** | Kod doğrudan sunucu sürecinin yanında koşuyor |
| `docker` | konteyner | Kılavuzun önerdiği |
| `jupyter` | çekirdek | Ekstra gerekiyor · **durum taşıyor** |
| `docker_jupyter` | konteyner + çekirdek | İkisinin birleşimi |
| `azure` | uzak | Azure Container Apps |

Kılavuz yerel yürütücü için açık uyarı veriyor: **modelin ürettiği kodu izolesiz
çalıştırmak risklidir.**

### Docker yürütücünün parametreleri — ve orada olmayanlar

`DockerCommandLineCodeExecutor` **[ölçüldü]**:

```python
DockerCommandLineCodeExecutor(
    image="python:3-slim",      # varsayılan
    timeout=60,                  # saniye
    work_dir=None,               # host'ta bağlanan dizin
    auto_remove=True,            # konteyner çıkışta siliniyor
    stop_container=True,
    extra_volumes=None,
    device_requests=None,        # GPU
    init_command=None,           # konteyner açılışında koşacak komut
)
```

**Ve listede olmayanlar, listede olanlardan daha önemli:**

| Yok | Sonucu |
|---|---|
| `network_mode` | Konteyner varsayılan **bridge** ağında — **interneti var** |
| `user` | İçeride **root** |
| `read_only` | Kök dosya sistemi **yazılabilir** |
| `mem_limit` · `nano_cpus` · `pids_limit` | **Kaynak sınırı yok** |
| `cap_drop` | Hiçbir yetki düşürülmüyor |

Bu bir yapılandırma eksikliği değil, **API'de o parametreler yok** — kaynağında
ağ ile ilgili tek kelime geçmiyor.

> **Sonuç:** *"kod sandbox'ta koşuyor"* cümlesi bu yürütücüyle kurulamaz.
> Kurulabilecek cümle: *"kod izole bir konteynerde koşuyor, ve konteynerin ağ
> erişimi var."*

Sertleştirme mümkün ama bedava değil: `start()` override edilip
`containers.create(..., network_mode="none", user="1000", mem_limit="512m")`
geçilebilir — bu **yukarı akışın iç koduna bağımlılık** yaratıyor ve sürüm
değişince sessizce kırılıyor. Bakım modundaki bir projede risk daha yüksek.

### Konteynerin ömrü: çağrı başına mı, süreç başına mı

Konteyner ayağa kaldırmak **2–3 saniye**, ve bu süre kullanıcının beklediği
zamana ekleniyor. İki seçenek:

| | Çağrı başına | Süreç başına |
|---|---|---|
| Gecikme | her çağrıda 2–3 sn | bir kez, açılışta |
| Turlar arası durum | temiz | **taşınıyor** |
| İzolasyon | konteyner ↔ host, tur ↔ tur | yalnız konteyner ↔ host |

`start()` / `stop()` sunucunun yaşam döngüsüne bağlanırsa süreç başına tek
konteyner olur — hızlı, ama bir turun `/tmp`'ye yazdığını sonraki tur görüyor.

### Tool'a dönüşmesi — ve tarifin önemi

`PythonCodeExecutionTool(executor)` yürütücüyü normal bir tool'a çeviriyor, yani
**aynı döngüden, aynı workbench'ten, aynı kapıdan** geçiyor. Ayrı bir yol yok.

Ama varsayılan tarifi tek cümle: **`"Execute Python code blocks."`** Bu tarifle
model kodu bir *kaçış kapağı* değil, bir *genel çözüm* sanıyor ve her hesabı
yeniden icat ediyor — mevcut tool'lar boşta kalıyor.

Tarif, modelin bu tool'a **ne zaman** uzanacağına karar verdiği tek metin. Rolü
anlatan bir tarif şunu söylemeli: *"önce mevcut tool'lara bak; sorulanı
karşılayan yoksa kod yaz."*

### Kapı için özel bir kanca gerekiyor

Ad bazlı bir kapı **bu tool'u göremiyor.** `"CodeExecutor"` tipik dışarı-yazma
işaretlerinin (`send`, `post`, `write`, `delete`) hiçbirine uymuyor, yani ada
bakan bir filtre onu sessizce geçiriyor.

Çözüm: `before_tool_call` seviyesinde **ada değil türe** bakan bir kanca, ve
onayı `(tool, argümanlar)` imzasına bağlamak — böylece kod değişirse eski onay
tutmuyor.

> **Ve onay tüketildikten sonra:** aynı soruyu modele tekrar sormak **farklı bir
> program** üretiyor **[ölçüldü]**. Onaylananla çalışanın aynı olmasının tek
> yolu, çalıştırılacak olanın **onaylanan metin** olması — yeniden üretilen değil.

### MAF tarafı

MAF'ta karşılığı **hosted tool** olarak geliyor: `SupportsCodeInterpreterTool`
sözleşmesini karşılayan bir istemci, kodu **sağlayıcı tarafında** çalıştırıyor.
Ayrıca `MontyCodeActProvider` ile sandbox'lı, çapraz platform bir yorumlayıcı
seçeneği var **[teyitsiz]**.

Fark: AutoGen'de konteyner **senin makinende**, MAF'ın hosted yolunda
**sağlayıcıda**. İkisi farklı güven kararı — birinde altyapı senin, diğerinde
veri dışarı çıkıyor.

---

## 12 · Ölçülmüş tuzaklar

{svg("f_gotchas", "Hiçbiri istisna fırlatmıyor")}

| Tuzak | Sonuç |
|---|---|
| `tools=` ve `workbench=` birlikte | `ValueError` — tek net hata |
| `model_context` verilmemiş | Ajanın **belleği yok**, hata da vermiyor |
| OpenAI-*uyumlu* endpoint | `model_info` **zorunlu** |
| `max_tool_iterations` = 1 | Zincirleme davranış sessizce imkânsız |
| Dış runtime, ajan çöküyor | **Fırlatmaz, asar** |
| `description` boş ajan | `SelectorGroupChat` **kör** seçiyor |
| `Handoff` adı küçük harfe düşüyor | Elle yazınca eşleşmiyor |
| `stop_when_idle()` | Handler çökerse bariyer erken açılıyor |

> **Ortak nokta:** bulunan hataların **hiçbiri istisna fırlatmadı.** Sıfır
> döndü, boş kaldı, asılı kaldı, ya da hata metnini cevap diye sundu.
> Core'u öğrenmenin yolu API'sini okumak değil, **arıza davranışını ölçmek**.

---

## 13 · MAF ne getirdi

{svg("f_components", "MAF'ın eklediği katmanlar")}

AutoGen'de **karşılığı olmayan** beş şey:

| Yetenek | Ne yapıyor |
|---|---|
| **Middleware** | Ajan / sohbet / fonksiyon seviyelerinde ara katman; her biri turu **durdurabiliyor** |
| **Checkpoint** | Workflow durumu diske yazılıp geri yükleniyor |
| **İnsan döngüde** | `ctx.request_info()` + `@response_handler` — çerçevenin **içinde** |
| **Harness** | `create_harness_agent` — todo, plan/execute kipleri, dosya belleği, onay, OTel |
| **FIDES** | Bütünlük + gizlilik etiketleri; politika hassas tool çalışmadan **önce** zorlanıyor |

Kılavuzun kendi cümlesi **[kaynak]**:

> *"AutoGen's `Team` abstraction runs continuously once started and doesn't
> provide built-in mechanisms to pause execution for human input."*

### Mimari fark: kontrol akışından veri akışına

* **GraphFlow** — *control-flow*: kenarlar geçiş, mesajlar **herkese** yayınlanır
* **Workflow** — *data-flow*: mesajlar **belirli kenarlardan**, yürütücü girdisi
  hazır olunca tetikleniyor

Bu, §5'teki sessiz kardeş kaybının kökeni.

---

## 14 · MAF ne kaybettirdi

| Yetenek | AutoGen | MAF |
|---|---|---|
| Dağıtık runtime (gRPC) | var (deneysel) | **yok** — "planned" |
| Model yanıtı önbelleği | `ChatCompletionCache` | **yok** — "🚧 Planned" |
| Aktör modeli / topic | `autogen-core` | **yok** |

### Ve hızın faturası

* **1.0 GA'dan sonra iki ayda 15 kırıcı değişiklik** — Microsoft'un kendi
  🔴 işaretlemesiyle **[kaynak]**
* 36 paketin **8'i** kararlı; 22 `beta`, 6 `alpha`
* Harness, FIDES, beceriler → hepsi `experimental` ve gerçekten
  `ExperimentalWarning` fırlatıyor **[ölçüldü]**

### Sürüm hızı — ölçüldü

| Paket | Son sürüm | Kaç gün önce |
|---|---|---:|
| **autogen-agentchat** | 0.7.5 | **323** |
| agent-framework | 1.14.0 | 5 |
| langgraph | 1.2.11 | 8 |
| openai-agents | 0.22.0 | **0** |

---

## 15 · Geçiş haritası

Microsoft'un kendi göç kılavuzundan **[kaynak]**:

| AutoGen | MAF |
|---|---|
| `AssistantAgent(model_client=…)` | `Agent(client=…)` |
| `FunctionTool(fn)` | `@tool` — şemayı imzadan çıkarıyor |
| `RoundRobinGroupChat` | `SequentialBuilder` |
| `SelectorGroupChat` | `GroupChatBuilder(selection_func=…)` |
| `Swarm` | `HandoffBuilder` |
| `MagenticOneGroupChat` | `MagenticBuilder` |
| `GraphFlow` | `WorkflowBuilder` |
| `model_context` (ajana ait) | `AgentSession` (çağrıya ait) |

> **Dikkat:** kılavuz Swarm ve Selector'ı *"currently in development"* diyor,
> ama ikisi de 1.14.0'da **var ve `released`** **[ölçüldü]**. Kılavuz kendi
> paketinin gerisinde — paketi aç, `dir()` çek.

---

<sub>Üretim: `python docs/tools/make_wiki.py` · şemalar `docs/diagrams/figures.py`
· kaynak metinler `docs/05` · `docs/08` · `docs/20` · `docs/21` · Türkçe rehberler
`docs/11` · `docs/22`</sub>
"""


# ══════════════════════════════════════════════════════ 27 — OpenClaw
def openclaw() -> str:
    return f"""# OpenClaw — harness wiki'si

> **Bu ne:** OpenClaw'ın nasıl çalıştığı, ve bir kurumda **neyinin alınıp
> neyinin alınmayacağı**. Tek dosya, arayarak okunmak için.
>
> **Sürüm:** OpenClaw `@01cc7106` · 22 paket · 51 tool (44'ü canlı kurulumda)
>
> **Ayrım:** OpenClaw bir kütüphane değil, **çalışan bir sistem**. AutoGen'i
> kodunuza gömüyorsunuz; OpenClaw'ı çalıştırıyorsunuz.

---

## İçindekiler

1. [Harness ne demek](#s1)
2. [Mimari kuşbakışı](#s2)
3. [Üç kontrol ekseni](#s3)
4. [Yetki kapsamları](#s4)
5. [Onay: komuta değil, plana](#s5)
6. [Dış içerik veri, talimat değil](#s6)
7. [Bellek katmanları](#s7)
8. [Kademeli açığa çıkarma](#s8)
9. [Bağlam motoru](#s9)
10. [Zamanlama](#s10)
11. [Dayanıklılık](#s11)
12. [İki kayıt hattı](#s12)
13. [Niş yüzeyler](#s13)
14. [Ne alınır, ne alınmaz](#s14)

---

## 1 · Harness ne demek

Microsoft'un kendi tanımı — MAF kılavuzundan, ve kelime artık resmî
**[kaynak]**:

> *"An agent harness is the runtime scaffolding that turns a language model into
> an agent that can perform work. It drives model and tool calls, manages
> conversation state and context, applies approval policies, and can keep the
> agent progressing through a multi-step task."*

OpenClaw bunun bugün çalışan, olgun bir örneği.

---

## 2 · Mimari

{svg("f_oc_arch", "Gateway, ajan, kanallar, node'lar")}

### Paket haritası — kod nasıl bölünmüş

{svg("f_packages", "22 paket · her ilginç parça ayrı")}

Bir sistemin nasıl düşünüldüğünü öğrenmenin en hızlı yolu, kodun nasıl
bölündüğüne bakmaktır. En büyük üçü:

| Paket | Dosya | İşi |
|---|---:|---|
| `ai` | 118 | Model sağlayıcılarıyla konuşuyor |
| `gateway-protocol` | 108 | Kontrol düzleminin **tipli şeması** |
| `memory-host-sdk` | 83 | Bellek sağlayıcılarının uyması gereken **sözleşme** |

Üçünün ayrı olması bir tercih: model erişimi, kontrol düzlemi ve bellek
birbirinden bağımsız değiştirilebilsin diye. Kontrol düzleminin **kendi şema
paketi** olması, protokolün koddan önce geldiğini gösteriyor.

### Sessiz altyapı — görünmeyen yarı

Beş paket hiçbir özellik listesinde yer almıyor, ama biri eksik olduğunda hemen
fark ediliyor:

| Paket | Ne yapıyor |
|---|---|
| `normalization-core` | Farklı kanallardan gelen içeriği **tek biçime** indiriyor |
| `markdown-core` | Modelin yazdığını her kanalın kaldırabileceği biçime çeviriyor — WhatsApp'ın markdown'ı Slack'inki değil |
| `terminal-core` | TUI çıktısı |
| `retry` | Yeniden deneme politikası |
| `net-policy` | Ağ erişim kuralları |

**Tek Gateway süreci.** Oturumlar, onaylar, zamanlama, kanallar hepsi orada.
Kurulu sistemde ölçüldü:

| Ne | Sayı |
|---|---:|
| Sohbet komutu | **89** |
| Denetim olayı (canlı) | **100** + devam imleci |
| Tool | 51 kaynakta · **44** canlı kurulumda |
| Paket | 22 |

---

## 3 · Üç kontrol ekseni

{svg("f_three_axes", "Sandbox · tool policy · elevated")}

"İzin" tek kavram değil, **üç ayrı soru**:

| Eksen | Soru | Anahtar |
|---|---|---|
| **Sandbox** | Tool **nerede** koşuyor? | `agents.*.sandbox.mode` |
| **Tool policy** | **Hangi** tool çağrılabilir? | `tools.allow` / `tools.deny` |
| **Elevated** | Kutunun **dışına** çıkış var mı? | `tools.elevated.*` — yalnız `exec` |

**Kurallar:** `deny` her zaman kazanır · `allow` doluysa listede olmayan her şey
bloklu · tool policy sert duraktır, `/exec` bile reddedilmiş `exec`'i geri
getiremez.

### Ve OpenClaw'ın kendi belgesindeki uyarı

> *"Tool policy tool'u **adına göre** filtreler; `exec` içindeki yan etkileri
> incelemez. `exec` serbestse, `write`/`edit`/`apply_patch`'i reddetmek shell
> komutlarını salt-okunur yapmaz."*

**"Yazma tool'unu kapattık, artık read-only" cümlesi yanlıştır.** Üç ekseni
karıştırmak en yaygın yapılandırma hatası, ve sonucu güvenlik tiyatrosu.

---

## 4 · Yetki kapsamları

{svg("f_scopes", "Kapsam çağrının parametresinden türetiliyor")}

Sekiz yetki kapsamı var, ama asıl fikir tabloda değil:

> **Aynı metot, farklı parametreyle farklı yetki istiyor.**

`agent` metodu sıradan bir tur için yazma yetkisiyle geçiyor, ama `/reset` için
yönetici istiyor. Yani yetki **metot adına** değil, **çağrının içeriğine**
bakıyor.

### Taşınacak fikir: rol bir tool listesi değil, **grup adı**

13 tool grubu var (`group:fs`, `group:runtime`, `group:web`…). Kurumda bu
`group:musteri-verisi`, `group:kredi-sorgu`, `group:rapor`, `group:dis-erisim`
olur. **Yeni bir tool eklendiğinde 40 rol dosyası güncellenmiyor.**

---

## 5 · Onay

{svg("f_frozen_plan", "Onay plana bağlanıyor, komuta değil")}

**Donmuş plan:** onay bir komuta değil, **plana** bağlanıyor. Onaydan sonra
argümanlar yeniden doğrulanıyor; dosya değiştiyse koşu reddediliyor.

Bizim karşılığımız: imza `(tool, argümanlar)` üstünde ve **bir kez tüketiliyor**.

> Neden önemli: modelden aynı işi ikinci kez istediğinde farklı bir program
> yazıyor **[ölçüldü]** — imzalar `029f4d1f…` ve `107fdfd1…`. Onaylananla
> çalışanın aynı olmasının tek yolu, çalıştırılacak olanın **onaylanan metin**
> olması.

---

## 6 · Dış içerik

{svg("f_external_content", "Veri, talimat değil")}

Web'den, dosyadan, tool sonucundan gelen içerik **veri** olarak işaretleniyor —
talimat olarak değil. Prompt enjeksiyonuna karşı ilk savunma bu ayrım.

**Dürüst sınır:** bu heuristik bir savunma, deterministik değil. Deterministik
karşılığı MAF'ta var — **FIDES**, bütünlük ve gizlilik etiketleriyle — ve
deneysel.

---

## 7 · Bellek

{svg("f_memory_tiers", "Beş bellek katmanı")}

{svg("f_memory_write", "Güvenlik sınırı YAZMA yolunda")}

Belleğin güvenlik sınırı **okuma** tarafında değil, **yazma** tarafında. Bir
kere yanlış yazılan olgu orada yaşıyor ve her turda geri okunuyor.

Bizim karşılığımız: bellek düz **Markdown** dosyaları, gizli depo yok.
Okunabilen, düzeltilebilen, silinebilen, sürüm kontrolüne konabilen bellek.

---

## 8 · Kademeli açığa çıkarma

{svg("f_skill_disclosure", "Bir satır tarif · gövde seçilince")}

Prompt'a giren şey yalnız **frontmatter**: ad + bir satır açıklama. Skill'in
gövdesi girmiyor; ancak seçilince yükleniyor.

Elli skill'in tamamını prompt'a koymak elli skill'lik token demek. Bir satırla
elli tanesi taşınabiliyor.

### Skill arama üç katman **[kaynak]**

* **Yerel keşif** — diskte `SKILL.md` dosyaları, `agent-filter` ile bu ajana
  görünenler süzülüyor
* **Kademeli yükleme** — gövde ancak `/skill <ad>` ile geliyor
* **Uzak arama** — `/clawhub`, bir kayıt defteri; *npm'in skill'ler için hâli*

Ve döngüyü kapatan: **`/learn`** — az önce yaptığın düzeltmeyi yeniden
kullanılabilir bir skill'e çeviriyor.

### Aynı fikrin tool tarafı

{svg("f_tool_search", "Büyük katalog, küçük prompt")}

---

## 9 · Bağlam motoru

{svg("f_ctx_engine", "Dört yaşam döngüsü noktası")}

**Ingest** (mesaj eklendi) · **Assemble** (bütçeye sığan sıralı küme) ·
**Compact** (pencere doldu) · **After turn**.

### Sıkıştırmayı doğru yapan kural

> Bir tool çağrısı ve sonucu **tek birimdir.** Bölme noktası bir tool bloğunun
> içine düşerse sınır kaydırılıyor.

Aksi hâlde: cevabı görünen ama sorusu özetlenip silinmiş bir tool sonucu kalıyor,
ve sağlayıcılar bu diziyi doğrudan **reddediyor**.

Bizim `context_engine.py` bu kuralı AutoGen'in `ChatCompletionContext`'i üstünde
yeniden kuruyor — çünkü AutoGen'in `BufferedChatCompletionContext`'i **mesaj
sayıyor, token değil**.

---

## 10 · Zamanlama

{svg("f_task_stack", "Zamanlama yığını — altı mekanizma, altı ayrı soru")}

Bu bölüm wiki'nin en uzunu, çünkü **AutoGen'de karşılığı hiç yok** ve bir kurumda
en çok istenen şey bu. Kaynak: `packages/gateway-protocol/src/schema/cron.ts`.

### Beş tür — ve ikisi zamana hiç bakmıyor

Şema **kapalı bir birleşim** (`Type.Union`), yani altıncı bir tür uydurulamıyor
**[kaynak]**:

| Tür | Ne zaman tetikliyor | Alanlar |
|---|---|---|
| `at` | Bir kez, belirli anda | `at` |
| `every` | Her N milisaniyede | `everyMs` · `anchorMs` |
| `cron` | Cron ifadesine göre | `expr` · `tz` · `staggerMs` |
| **`on-exit`** | İzlenen komut **çıkınca** | `command` · `cwd` |
| **`stream`** | Komutun **çıktı satırından** | `command` · `mode` · `match` · `batchMs` |

Son ikisi zamanlayıcı değil, **olay kaynağı**. `on-exit` izlenen bir komut
bittiğinde bir kez tetikliyor; `stream` uzun ömürlü bir komutun stdout/stderr
satırlarından.

> **Ve `on-exit`'in kod yorumu tek başına bir tasarım dersi:** watcher, turun
> süreç ağacında değil **gateway'in `ProcessSupervisor`'ında** koşuyor. Yani turu
> bitirmek watcher'ı öldürmüyor. Sahiplik doğru yere konmuş.

### İki incelik — ikisi de ölçekte fark ediyor

**① Yük dağıtma (`staggerMs`).** Saat başına denk gelen tekrarlı işler
kendiliğinden **5 dakikaya kadar** kaydırılıyor — yüz iş aynı anda uyanıp yük
tepesi yapmasın diye. `--exact` ile kapatılıyor **[kaynak]**.

**② Cron'un OR tuzağı.** Ayın-günü ve haftanın-günü alanlarının ikisi de joker
değilse, `croner` **ya biri ya öteki** eşleştiğinde tetikliyor:

```bash
# Niyet: "ayın 15'i, ama yalnız Pazartesiyse"
0 9 15 * 1
# Gerçek: her ayın 15'inde 9'da, VE her Pazartesi 9'da
# → ayda 0–1 yerine 5–6 kez
```

Standart Vixie cron davranışı, yani hata değil. Ama bir kurumsal asistanda
**"ayda beş kez rapor gönderdi" bir arıza kaydıdır.** Çözüm: croner'ın `+`
değiştiricisi (`0 9 15 * +1`) ya da bir alanı işin içinde kontrol etmek.

### Koşul gözcüsü — zamana değil **duruma** bağlanmak

Zamanlamanın en az konuşulan türü. *"Her sabah 9'da bak"* değil, **"şu koşul
doğru olduğunda haber ver."**

Fark pratik: zamana bağlı bir iş, hiçbir şey değişmediğinde de koşuyor ve her
koşusu para. Duruma bağlı olan yalnız değiştiğinde uyanıyor.

Bizim `gateway/cron.py`'de karşılığı `Threshold` — ve bilerek **aptal**: yıldız
farkı, yeni başvuru, adı geçme. Model çağrısı değil.

> **Neden model değil:** token harcayıp token harcamaya karar veren bir bildirici
> kötü bir takas. Ve *"beni neden uyandırdı"* sorusunun bir insanın okuyabileceği
> cevabı olmalı.

### Task defteri — zamanlayıcı değil, **kayıt**

{svg("f_task_lifecycle", "Bir işin yaşam döngüsü")}

Defter **ne zaman koşacağına karar vermiyor**; ne koştuğunu yazıyor. İkisini
karıştırmak, yeniden başlatmada geçmiş işleri yeniden oynatmaya götürüyor.

Doğru davranış: süreç yeniden başladığında **geçmiş işleri tekrar oynatmıyor,
yeniden zamanlıyor.** Bir gecede kaçırılan üç koşu, sabah üç kez arka arkaya
koşmuyor.

### Üç eksen — ve tip düzeyinde ayrılmış olmaları

{svg("f_task_axes", "Ne zaman · nerede · nereye")}

| Eksen | Soru |
|---|---|
| **Zamanlama** | Ne zaman koşacak? |
| **Oturum hedefi** | Hangi bağlamda koşacak? |
| **Teslimat** | Sonuç nereye gidecek? |

Üçü **ayrı alanlar**, ve ayrı olmaları bir kaza değil: bir işi "her sabah koş"
diye kurmakla "sonucu Telegram'a at" demek iki farklı karar, ve ikincisi
**dışarı mesaj gönderiyor** — yani onay kapısının konusu.

### Taze oturum kuralı

**Zamanlanmış koşu kendi oturumunu alıyor.** Tek uzun ömürlü oturum kullanan bir
iş her önceki koşuyu bağlamında biriktiriyor:

* her gün **pahalılaşıyor**
* ve sonunda bu sabahın verisi yerine **geçen haftanın hafızasından** cevap
  veriyor

Hiçbiri hata vermiyor; iş "çalışıyor" görünüyor.

### Yoklama yanlış şekil

{svg("f_threads", "Yoklama yerine olay")}

*"Her 30 saniyede bak, değişti mi"* bir zamanlayıcı deseni değil, bir zamanlayıcı
**eksikliğinin** belirtisi. `on-exit` ve `stream` tam olarak bunun yerine var.

### Bizde bugün

| | Durum |
|---|---|
| Çevirmen (`scheduler.py`) | **bağlı** — Türkçe ifadeyi cron şekline çeviriyor, üç biçim kabul ediyor |
| Yerli zamanlayıcı (`gateway/cron.py`) | **yazıldı, 19 test, bağlanmadı** |
| Koşul gözcüsü (`Threshold`) | `cron.py` içinde, bağlı değil |

**Dürüst sınır:** zamanlama yalnız OpenClaw'ın Gateway'i koşarken çalışıyor. Bu
makinede systemd *kullanıcı* servisi, `Linger=no` → oturum bitince duruyor.
Sessizce ateşlemeyi bırakmış bir iş, bir zamanlayıcının **en kötü arızası** — o
yüzden liste, Gateway'e ulaşılamamasını *boş liste değil, kendi durumu* olarak
raporluyor.

---

## 11 · Dayanıklılık

{svg("f_durable", "Dayanıklı durum — ama durable execution değil")}

**Önemli ayrım:** OpenClaw dayanıklı **durum** tutuyor; dayanıklı **yürütme**
değil. Süreç ortada ölürse, yarım kalan tur kaldığı yerden devam etmiyor.

`Temporal`/`durable execution` bekleyen biri bunu bilmeli.

{svg("f_failover", "Model failover")}

{svg("f_loopguard", "Döngü kırıcı ve sıkıştırma sonrası nöbetçi")}

---

## 12 · İki kayıt hattı

{svg("f_two_ledgers", "Uyum kaydı ile hata ayıklama kaydı ayrı")}

| | Uyum kaydı | Hata ayıklama kaydı |
|---|---|---|
| Değişmez mi | **evet** | hayır |
| Saklama süresi | var | kısa |
| Sır taşır mı | **asla** | taşıyabilir |
| Kim okur | denetçi | mühendis |

Tek hatla ikisini birden yapmak **ikisini de bozar**: ya denetim kaydına sır
sızar, ya hata ayıklama kaydı ömür boyu saklanır.

{svg("f_secrets", "Sırlar ve telemetri")}

---

## 13 · Niş yüzeyler

Sorulursa açılacak, kendiliğinden anlatılmayacak konular.

{svg("f_repair", "Tool call repair — bozuk çağrıyı kurtarmak")}

Model bozuk JSON ürettiğinde turu çöpe atmak yerine onarmaya çalışıyor.

{svg("f_result_middleware", "tokenjuice — komuta değil SONUCA dokunuyor")}

Tool **sonucunu** küçültüyor. Tool'un ne yaptığı değişmiyor; **modelin ne kadarını
gördüğü** değişiyor.

{svg("f_trajectory", "Trajectory — oturumun uçuş kayıt cihazı")}

`/export-trajectory` redakte edilmiş bir destek paketi çıkarıyor. Kullanıcının
gerçekten rapor göndermesini sağlayan şey bu.

{svg("f_self_learning", "Düzeltmeyi skill'e çevirmek")}

### Built-in tool kataloğu — ve dağılımın anlattığı

{svg("f_tool_catalog", "51 tool · 11 grup")}

Bir ajanın ne yapabildiğini elindeki tool listesi belirliyor. Asıl bilgi sayıda
değil **dağılımda**:

| Grup | Tool | Ne anlama geliyor |
|---|---:|---|
| `sessions` | **15** | Alt-ajan başlatmak, iş devretmek, cevabını beklemek, aralarında mesajlaşmak |
| dosya işlemleri | 4 | — |
| komut çalıştırma | 3 | — |

Yatırımın büyük kısmı dosyaya ya da kabuğa değil, **ajanlar arası koordinasyona**
yapılmış. Bir harness'ın ne için tasarlandığını en iyi bu oran anlatıyor.

### "Kaç tool var?" — üç cevap, üçü de doğru

{svg("f_profiles", "51 · 44 · doküman tablosu")}

| Sayı | Ne sayıyor |
|---:|---|
| **51** | Kaynak kodda tanımlı |
| **44** | Canlı gateway'in gerçekten sunduğu **[ölçüldü]** |
| başka | Dokümandaki tablo — **eskimiş** |

Aradaki fark filtrelerde eriyor, ve daralma **üç aşamada**: profil bir taban
liste veriyor → `allow`/`deny` onu kesiyor → sandbox'tayken sandbox politikası
bir kez daha kesiyor.

> **Üçü de yalnız daraltıyor; hiçbiri listeye tool ekleyemiyor.** Bir yetkilendirme
> katmanının doğru yönü bu — genişleyebilen bir filtre, filtre değildir.

### Koşan bir tura müdahale etmenin dört yolu

{svg("f_session_tools", "/steer · /btw · /goal · /loop")}

Sıradan bir sohbet arayüzünde mesajı gönderdikten sonra yapabileceğin tek şey
beklemektir. Bu dört komut aynı soruya farklı cevaplar veriyor: **tur çoktan
başlamışken ne yapabilirsin?**

| Komut | Ne yapıyor |
|---|---|
| `/steer` | Koşan turu **yönlendiriyor**. Runtime müdahaleyi kabul etmezse mesajı çöpe atmıyor, sıradan bir prompt olarak gönderiyor |
| `/btw` | Araya **yan soru** sokuyor ve cevabı konuşma geçmişine eklemiyor — asıl işin bağlamını kirletmiyor |
| `/goal` | Oturuma **kalıcı bir hedef** bağlıyor; hem operatör hem model aynı hedefi görüyor |
| `/loop` | Konuşmaya bağlı, **kendini tekrarlayan** bir iş kuruyor |

Canlı ölçümde `commands.list` **89 komut** döndürüyor; dördü bu grupta
**[ölçüldü]**.

> **Karşılaştırma:** AutoGen'de koşan bir turun içine girmenin **hiçbir yolu
> yok**. Tur başladıktan sonra tek seçenek beklemek ya da iptal etmek.

### Doğrulanmamış olanlar

Aşağıdaki iki başlık **koşturulmadı.** Şema, katalog tarifinin ve resmî
belgelerin okunmasından çiziliyor — ölçümden değil. Ayrımın görünür kalması
için buraya konuldular, önceki bölümlere değil.

#### Lobster — tipli iş akışı runtime'ı · **[teyitsiz]**

{svg("f_lobster", "Tek çağrı · gömülü kapılar · devam token'ı — [teyitsiz]")}

Katalog kaydı doğrulandı **[kaynak]**: `@openclaw/lobster`, *"Lobster workflow
tool plugin (typed pipelines + resumable approvals)"*, `source: official`,
`minHostVersion >= 2026.4.25`. **Çekirdekte değil, ayrı bir eklenti**, ve bu
kurulumda yüklü değil.

Anlattığı fikir — katalog tarifinden okunuyor:

* Çok adımlı bir işi modele orkestra ettirirsen **her adım ayrı bir tur** olur;
  dört adım dört model çağrısı, ve her turda bütün bağlam yeniden gönderilir.
* Tipli boru hattı orkestrasyonu modelden alıp **runtime'a** veriyor: model bir
  kez konuşuyor, ara sonuçlar prompt'a hiç uğramıyor.
* Yan etkili bir adımda akış **duruyor** ve bir **devam token'ı** dönüyor —
  onaydan sonra baştan değil, kaldığı yerden.

> **Kapıyı runtime tutuyor, model değil.** Model *"bu sefer onay sormayayım"*
> diye karar veremiyor — kapı boru hattının parçası, erişebileceği bir yerde
> değil. Aynı ilke bizim workbench kapımızda da geçerli.

#### Code Mode / Swarm · **[teyitsiz]**

Katalog büyüdükçe bütün tool şemalarını prompt'a koymak imkânsızlaşıyor.
**Code Mode** bunu şöyle çözüyor: model tool şemalarını görmüyor, küçük bir
köprüye `search` · `describe` · `call` yazıyor. **Swarm** aynı köprüden
eşzamanlı alt-ajanlar başlatıyor.

---

## 14 · Ne alınır, ne alınmaz

{svg("f_atlas", "Üç ayrı ilişki")}

### Alınacaklar — **karar kuralları**, kod değil

* Üç kontrol ekseninin **ayrı** tutulması
* Rol = grup adı, tool listesi değil
* Onay plana bağlanır, komuta değil
* Dış içerik veri, talimat değil
* Bellek güvenlik sınırı **yazma** yolunda
* İki kayıt hattı
* Kademeli açığa çıkarma
* Zamanlanmış koşu **taze oturum** alır

### Alınmayacak — **güven modeli**

OpenClaw **tek bir güvenilen operatör** varsayıyor. Kurumda o varsayım geçerli
değil.

Ölçülen kanıt, sistemin kendi cümlesi — bir `/openclaw` satırı gönderdiğimizde
kapımız tuttu **[ölçüldü]**:

> *"O ajanın kabuk erişimi var ve şu an onay sormadan çalıştırıyor
> (**exec: mode=full, ask=off**); bizim kapımız içeride ne yapacağını görmez."*

### Sonuç: üç ayrı ilişki

**AutoGen'i gömüyoruz** (motor, ince arayüz arkasında) · **OpenClaw'ı
öğreniyoruz** (karar kuralları) · **OpenClaw'ı mühendislikte kullanmaya devam
ediyoruz**.

> Atlas olarak OpenClaw **kurmuyoruz**.

---

<sub>Üretim: `python docs/tools/make_wiki.py` · kaynak `docs/13` (mimari analiz) ·
`docs/16` (kurumsal okuma) · `docs/18` (zamanlama ve dayanıklılık) · canlı
ölçümler kurulu OpenClaw `@01cc7106`'dan</sub>
"""
