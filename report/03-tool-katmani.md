# Tool Katmanı: Modelin Dünyaya Açılan Arayüzü

> **Bölümün tezi:** Bir tool, modele verilen bir yetenek değil, bağlama yazılan bir sözleşme metnidir. Modelin tool kullanma *becerisi* ağırlıklarında, tool *envanteri* bağlamındadır. Bu ayrım, tool calling'in bütün mühendislik sonuçlarını — halüsinasyon davranışını, token maliyetini, ölçeklenme stratejisini ve framework seçiminin gerçekte neyi değiştirdiğini — belirler.

---

## 1. Problem: stateless bir fonksiyonu dünyaya bağlamak

Dil modeli, girdi token dizisini alıp çıktı token dizisi üreten saf bir fonksiyondur. Dosya okuyamaz, HTTP isteği atamaz, veritabanı sorgulayamaz. Tool calling bu boşluğu kapatmaz — **boşluğu protokolleştirir.**

Çözülmesi gereken üç alt problem var:

| Problem | Soru |
|---|---|
| **Bildirim** | Modele "elinde şu araçlar var" nasıl söylenir? |
| **İfade** | Model "şunu çağırmak istiyorum"u nasıl ifade eder? |
| **Geri besleme** | Sonuç modele nasıl döner ve ondan ayırt edilir? |

Tool calling'in tarihi, bu üç sorunun cevabının **metinden yapıya** göç etmesinin tarihidir.

---

## 2. Tarihsel yay: ReAct'ten native tool calling'e

### 2.1 ReAct (2022) — her şey metin

2022'de modeller tool kullanmak üzere eğitilmemişti. Ellerinde yalnızca metin devam ettirme yeteneği vardı. ReAct'in çözümü, tool kullanımını modelin yapabildiği tek şeye indirgemekti.

Tool bildirimi, prompt metnine `.format()` ile gömülen iki satırdı:

```
You have access to the following tools:

search: useful for when you need to answer questions about current events
calculator: useful for when you need to do math

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [search, calculator]
Action Input: the input to the action
Observation: the result of the action
... (repeat N times)
Thought: I now know the final answer
Final Answer: the final answer
```

Döngü:

```python
scratchpad = ""
while True:
    prompt = TEMPLATE.format(tools=tool_descs, question=q, scratchpad=scratchpad)
    out = llm(prompt, stop=["\nObservation:"])              # ← kritik satır

    if "Final Answer:" in out:
        return out.split("Final Answer:")[-1].strip()

    m = re.search(r"Action:\s*(.*?)\n\s*Action Input:\s*(.*)", out, re.DOTALL)
    if not m:
        raise OutputParserException("Could not parse LLM output")

    obs = tools[m.group(1).strip()].run(m.group(2).strip())
    scratchpad += f"{out}\nObservation: {obs}\nThought:"
```

**`stop=["\nObservation:"]` ReAct'in gerçek kalbidir.** Kesilmezse model gözlemi kendisi uydurur:

```
Thought: Nüfusu aramalıyım
Action: search
Action Input: Istanbul nüfusu
Observation: 15.9 milyon        ← MODEL UYDURDU; arama hiç çalışmadı
Thought: Şimdi 3'e böleyim
```

Yani ReAct'te döngüyü durduran şey modelin bir kontrol sinyali üretmesi değil, **harness'in kelime eşleşmesiyle üretimi kesmesidir.**

### 2.2 Native tool calling — yapı ayrı katmana taşınıyor

Bugün `tool_use` bloğu üretmek post-training ile modelin içine yazıldı. ReAct'in prompt mühendisliğiyle simüle ettiği her parça kendi katmanına ayrıldı:

| ReAct | → | Modern |
|---|---|---|
| `"Thought:"` metni | → | `thinking` bloğu (ayrı modalite) |
| `"Action:/Action Input:"` metni | → | `tool_use` bloğu (yapısal) |
| regex parse | → | parse yok, tipli alan |
| `stop=["\nObservation:"]` | → | `stop_reason` (model sinyali) |
| `"Observation:"` metni | → | `tool_result` bloğu (ayrı rol) |
| `"Error: ..."` metni | → | `is_error: true` |
| tool listesi prompt'ta | → | `tools` alanı + `defer_loading` |
| adım başına tek Action | → | paralel `tool_use` |

Aynı döngü:

```python
messages = [{"role": "user", "content": question}]
while True:
    r = client.messages.create(model="claude-opus-5", max_tokens=8000,
                               tools=TOOLS, messages=messages)
    #                          ↑ stop_sequences YOK

    if r.stop_reason == "end_turn":                        # "Final Answer:" yerine
        return "".join(b.text for b in r.content if b.type == "text")

    messages.append({"role": "assistant", "content": r.content})   # regex YOK

    results = []
    for b in r.content:
        if b.type == "tool_use":                           # re.search yerine
            try:
                out = TOOLS_IMPL[b.name](**b.input)        # b.input zaten dict
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": str(out)})
            except Exception as e:
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": str(e), "is_error": True})   # hata SİNYALİ

    messages.append({"role": "user", "content": results})  # hepsi TEK mesajda
```

ReAct döngüsünde olup burada **olmayan** üç satır: `stop=[...]`, `re.search(...)`, `OutputParserException`.

### 2.3 Neden bu fark önemli — yapısal sonuçlar

| ReAct'teki kısıt | Sebebi | Modern karşılığı |
|---|---|---|
| Paralel çağrı imkânsız | Format tek `Action:` satırı tanımlıyor | Tek yanıtta N adet `tool_use` bloğu |
| Parse hatası bir hata sınıfı | `Action input:` yazarsa döngü çöker | Parse adımı yok; `b.input` tipli |
| Argüman doğrulaması yok | `Action Input:` düz string | JSON Schema + `strict: true` |
| Muhakeme çıktıyla aynı kanalda | `Thought:` kullanıcı metniyle aynı akışta | `thinking` ayrı blok tipi |
| Hata sinyali yok | Hata da `Observation:` metni | `is_error: true` |
| Tool sonucu modelin metniyle aynı düzlemde | Rol ayrımı yok | Ayrı rol + ayrı blok tipi |
| Halüsinasyon riski | Kesilmezse gözlem uydurulur | **Yapısal olarak imkânsız** |

Son satır kritik: modern formatta model gözlemi uyduramaz, çünkü gözlem **farklı bir blok tipinde ve farklı bir roldedir.** Uydurmaya kalksa bir `text` bloğu üretir; harness yalnızca `tool_use` bloklarını çalıştırdığı için o metin hiçbir şeyi tetiklemez.

> **Bulgu 1.** ReAct ile native tool calling arasındaki fark bir "desen tercihi" değil, katman farkıdır. ReAct tool kullanımını **metne çevirir**; modern yaklaşım **metni tool kullanımından ayırır.** Halüsinasyon, paralellik, doğrulama, hata sinyali ve muhakeme görünürlüğü — beşi de doğrudan bu ayrımın sonucudur.

---

## 3. Tool tanımının anatomisi

Modelin bir tool hakkında bildiği **her şey** üç alandır:

```json
{
  "name": "get_weather",
  "description": "Bir şehrin güncel hava durumunu getirir. Kullanıcı güncel hava, sıcaklık veya yağış sorduğunda çağır.",
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "Şehir ve ülke kodu, ör. 'Istanbul, TR'"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Sıcaklık birimi."}
    },
    "required": ["location"]
  }
}
```

Fonksiyon gövdesi, dosya adı, implementasyon dili, kod kalitesi — hiçbiri görünmez.

| Alan | Rolü |
|---|---|
| `name` | Dispatch anahtarı ve ikincil bir semantik sinyal |
| `description` | **Kararın verildiği yer** |
| `input_schema` | Argüman üretiminin uyacağı yapı |

Opsiyonel alanlar: `strict` (şema uyum garantisi), `defer_loading` (§7), `cache_control`, `allowed_callers` (programmatic tool calling).

### 3.1 Description: en yüksek kaldıraçlı mühendislik kararı

İki kural:

**a) Ne yaptığını değil, ne zaman çağrılacağını yaz.**

```python
# ✗ model "ne zaman" sorusunu kendi tahmin etmek zorunda
"description": "Web'de arama yapar."

# ✓ tetikleyici koşullu
"description": ("Web'de arama yapar. Cevap konuşmada bulunmayan güncel bilgiye "
                "bağlıysa (son olaylar, güncel fiyatlar, sürüme özgü davranış) "
                "hafızadan cevaplama, önce bunu çağır.")
```

**b) Agresif dilden kaçın.** `CRITICAL: YOU MUST`, `Şüphedeyse bunu kullan` kalıpları eski modeller için yazılmıştı. Güncel modeller sistem prompt'unu çok daha literal takip ediyor; bu ifadeler artık **aşırı tetiklemeye** yol açıyor. Model gereksiz yere tool çağırıyorsa çözüm daha fazla guardrail değil, dili yumuşatmaktır.

### 3.2 Şema, dekoratörden nasıl üretiliyor

```python
@beta_tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Bir şehrin güncel hava durumunu getirir.

    Args:
        location: Şehir ve ülke kodu, ör. Istanbul, TR.
        unit: Sıcaklık birimi, "celsius" veya "fahrenheit".
    """
```

| Python'daki | JSON'daki karşılığı |
|---|---|
| fonksiyon adı | `name` |
| docstring ilk paragrafı | `description` |
| tip anotasyonu | `properties[...].type` |
| docstring `Args:` satırı | `properties[...].description` |
| varsayılansız parametre | `required` |

> **Bulgu 2.** Docstring artık bir yorum değil, **prompt'tur.** Boş bırakılan bir docstring modeli kör bırakır. `@tool` "otomatik anlıyor" demek değil, "otomatik şema üretiyor" demektir.

---

## 4. Model tool'ları nasıl "biliyor"

Bu, bölümün merkez sorusu. İki ayrı "bilme" var ve karıştırılınca her şey karışır:

| Ne | Nereden | Ne zaman |
|---|---|---|
| `tool_use` bloğu nasıl üretilir, şemaya nasıl uyulur | **Ağırlıklardan** (post-training) | Eğitimde bir kez |
| Senin `search` tool'unun var olduğu, ne yaptığı, hangi argümanları aldığı | **Bağlamdan** | **Her turda yeniden** |

Model tool kullanmayı *öğrenmiştir*; senin tool'larını *okur*.

### 4.1 Runtime: bağlama nasıl giriyor

```
tools: [{...}, {...}]  ──serialize──▶  prompt prefix (0. pozisyon)  ──tokenize──▶  model
```

Render sırası `tools → system → messages`. Tool tanımları system prompt'undan bile önce gelir. Bu sıra keyfi değil: prompt caching bir prefix eşleşmesi olduğu için sabit içerik önde olmak zorundadır (§7.1).

### 4.2 Doğrudan gözlemsel kanıt

Bu raporun yazıldığı Claude Code oturumunda bağlamda iki grup tool bulunuyordu:

**Şeması yüklü olanlar (11 adet):** `Read`, `Bash`, `Write`, `Edit`, `Skill`, `Agent`, `ToolSearch`, …

**Yalnızca adı bildirilenler (≈18 adet):** `WebFetch`, `TodoWrite`, `CronCreate`, `Monitor`, `EnterPlanMode`, …

İkinci grup için sistem prompt'unda şu ifade yer alıyordu:

> *"Until fetched, only the name is known — there is no parameter schema, so the tool cannot be invoked."*
> *"Deferred tools appear by name… calling them directly will fail with `InputValidationError`."*

Yani model, `WebFetch`'in **var olduğunu bilir ama kullanamaz** — şeması bağlamında değildir. Kullanabilmesi için `ToolSearch` çağırıp şemayı bağlama çekmesi gerekir; çektiği anda tool kullanılabilir hâle gelir.

> **Bulgu 3.** Aynı model, aynı ağırlıklar, aynı oturum — tek değişken şemanın bağlamda olup olmaması. Ağırlıklarda saklı bir tool envanteri bulunsaydı bu davranış mümkün olmazdı. **Bilme = bağlamda olma.**

### 4.3 Karar mekanizması

Model bir sonraki token'ı tahmin eder; tool kullanımı için eğitilmiş olduğundan üretim uzayı yapılandırılmış bir dala ayrılabilir:

```
kullanıcı niyeti  ─┐
                   ├─▶ [semantik eşleştirme] ─┬─▶ text bloğu
tool description ─┘                           └─▶ tool_use bloğu
                                                   ├─ name  = eşleşen tool
                                                   └─ input = şemaya uyan JSON
```

Kararı etkileyen faktörler, etki gücü sırasıyla:

| Faktör | Etkisi |
|---|---|
| `description` metni | **En güçlüsü.** Tetikleyici koşul → tetiklenir; salt tanım → kaçırılır |
| `tool_choice` | `auto` / `any` / belirli tool / `none` — sert kontrol |
| `effort` | Yüksek effort → belirgin şekilde daha fazla tool çağrısı |
| System prompt | Yönlendirici talimatlar |
| Konuşma geçmişi | Önceki başarılı/başarısız çağrılar sonrakini biçimlendirir |
| `input_schema` | `enum`, `required`, property açıklamaları → argüman kalitesi |

### 4.4 Pratik sonuçlar

1. **Tool eklemek eğitim değil, metin eklemektir.** Yeni tool bir sonraki istekten itibaren geçerlidir; deploy veya fine-tune gerekmez.
2. **Tool çalışmıyorsa önce description'a bakılır.** Kod doğru ama model çağırmıyorsa sorun neredeyse her zaman açıklama metnindedir.
3. **Bilgi bedava değildir.** Bağlamdaki her tool her turda token yakar — `defer_loading`'in varlık sebebi budur.
4. **Model tool'un gerçekten çalıştığını bilmez.** Yalnızca `tool_result` içeriğini görür. Tool silinmiş ama şema gönderilmeye devam ediyorsa model çağırmayı sürdürür.

---

## 5. Döngüyü kim çevirir — üç kategori

Literatürde en çok karıştırılan eksen budur.

| Kategori | Tool'u kim çalıştırır | Döngüyü kim çevirir | Örnek |
|---|---|---|---|
| **Client-side, manuel** | Sen | Sen (`while`) | §2.2'deki döngü |
| **Client-side, yönetilen** | Framework/SDK — **hâlâ senin makinende** | Framework | `tool_runner`, `AgentExecutor` |
| **Server-side** | **Sağlayıcının altyapısı** | Sağlayıcı | `web_search_20260209`, `code_execution_20260521`, tool search |

Üçüncüsü niteliksel olarak farklıdır: `tools` dizisine eklenir, sonuç aynı yanıtta content bloğu olarak döner, **hiçbir döngü kodu yazılmaz.**

### 5.1 Literatür eleştirisi: "embedded tool calling"

Popüler bir anlatım, döngüyü bir kütüphaneye devretmenin halüsinasyonu azalttığını öne sürer:

> *"Reduced Hallucination: Since the tool execution is handled externally, the LLM does not fabricate tool calls."*

**Bu iddia iki bağımsız mekanizmayı birbirine karıştırır.** Her iki yaklaşımda da tool execution zaten model dışındadır — traditional'da client, embedded'da kütüphane çalıştırır; ikisi de model dışıdır. Döngüyü kimin çevirdiğinin halüsinasyonla ilgisi yoktur.

Halüsinasyonu engelleyen şey **çıktı formatıdır**:

| Yaklaşım | Halüsinasyonu ne engeller |
|---|---|
| ReAct (metin tabanlı) | `stop=["\nObservation:"]` — harness üretimi keser |
| Native tool calling | `tool_use` ayrı blok tipi, sonuç ayrı rol → **yapısal olarak imkânsız** |
| Manuel döngü ↔ runner | **Fark yok** — ikisi de aynı `tool_use` bloğunu alır |

Bu anlatım, tarihsel olarak aynı anda gerçekleşen iki geçişi (ReAct → native, ve manuel → framework) tek sebebe bağlamaktadır.

> **Bulgu 4.** İki eksen bağımsızdır: *(a)* tool'u kim çalıştırır — client / framework / sunucu; *(b)* model tool çağrısını nasıl ifade eder — metin / yapısal blok. **Halüsinasyon (b)'nin, geliştirici ergonomisi (a)'nın fonksiyonudur.**

### 5.2 "Kontrol lazım" ≠ manuel döngü

Manuel döngüye inmenin yaygın gerekçesi "kontrol"dür, ancak yönetilen runner'lar turn hook'larıyla bunu zaten sağlar:

| İhtiyaç | Runner'da karşılığı |
|---|---|
| İnsan onayı / gating | Tool gövdesinde sor, reddedilirse "kullanıcı reddetti" döndür; ya da yield edilen mesajdaki `tool_use`'u inceleyip tool çalışmadan müdahale et |
| Hata yakalama | Sonucu modele gitmeden incele |
| Sonucu değiştirme | `cache_control` ekleme, çıktı dönüştürme |
| Retry / parametre değişimi | Turu yeniden çalıştır; `max_iterations` ile sınırla |

Gerçekten manuel döngü gerektiren durumlar: SDK'nın kuramadığı istek şekli, özel transport, beta bağımlılığı istememe, döngü ortasında alakasız iş yapma.

### 5.3 Konuşmalı döngünün gizli maliyeti: plan yeniden türetme

Döngü sahipliğinden bağımsız olarak, **konuşma tabanlı** tool kullanımının yapısal bir vergisi vardır:

> *"...a series of conversational tool calls where **the model re-derives its plan from chat history on every turn**."*
> — Glean Engineering, 2026

Her turda model tüm geçmişi yeniden okur ve "neredeyim, sırada ne var" sorusunu yeniden cevaplar. Beş adımlı bir iş, planın beş kez türetilmesi demektir. Maliyet iki katmanlıdır: token (geçmiş yeniden gönderilir) ve **dikkat** (muhakemenin bir kısmı yeniden yönelmeye harcanır).

Programmatic tool calling bu vergiyi ortadan kaldırır: plan bir kez kod olarak yazılır ve döngü kodun içinde çalışır (§07.8). Glean'in ölçümüyle **tek bir sandbox çalıştırmasında 20 tool çağrısı** gerçekleşebilir ve orchestrator yalnızca özeti görür.

---

## 6. Framework'ler ne yapıyor

### 6.1 Ortak gerçek

```
     Bildirim                        Framework                 Sağlayıcı API
──────────────────────         ──────────────────────      ─────────────────────
@tool def search(q)     ──┐
BaseTool alt sınıfı     ──┼──▶  şema derleyici  ──▶  "tools": [{name, description,
FunctionTool(fn)        ──┤     + döngü sürücüsü               input_schema}]
Pydantic args_schema    ──┘                                          │
                         ◀──  tool_use / tool_calls  ◀───────────────┘
```

**Modele hiçbir framework izi ulaşmaz.** CrewAI'ın `role`/`goal`/`backstory` üçlüsü system prompt metnine derlenir; LangGraph'ın grafiği hiç gönderilmez. Framework iki iş yapar: bildirimi sağlayıcı şemasına derlemek ve döngüyü çevirmek.

### 6.2 Karşılaştırma

| | Tool bildirimi | Şema kaynağı | Döngü sahibi | Durum nerede | Ayırt edici |
|---|---|---|---|---|---|
| **LangGraph** | `@tool` / Pydantic | Tip + docstring | **Sen** — grafik kenarları | `State` (TypedDict + reducer), checkpoint'li | Döngü açık; kalıcılık + interrupt |
| **CrewAI** | `BaseTool` / `@tool` | Pydantic `args_schema` | **Framework** — gizli | Task çıktıları arası | Rol tabanlı; **delegasyon bir tool** |
| **Google ADK** | `FunctionTool` / `AgentTool` | İmza + docstring | Runner — callback'lerle açılır | `session.state` (prefix'li) | `ToolContext` — tool akışa müdahale edebilir |
| **OpenAI Agents SDK** | `@function_tool` | İmza + docstring | Framework | `Session` | `handoff` — ajan devri de tool |
| **Pydantic AI** | `@agent.tool` | İmza + docstring | Framework | `RunContext[Deps]` | Bağımlılık enjeksiyonu, tipli çıktı |
| **Anthropic SDK** | `@beta_tool` | İmza + docstring | Runner (turn hook'lu) | `messages` | En ince katman |

### 6.3 Üç dikkat çekici tasarım kararı

**a) LangChain'in normalize ara temsili.** Sağlayıcı farkını siler:

```python
ai_msg.tool_calls
# [{"name": "search", "args": {...}, "id": "toolu_01A"}]
# ↑ Anthropic'in tool_use'u da OpenAI'ın tool_calls'ı da BUNA dönüşür
```

Sonuç `ToolMessage(content=..., tool_call_id=...)` olarak eklenir, sağlayıcıya giderken native formata geri çevrilir.

**b) Çok ajanlı koordinasyon = tool çağrısı.** CrewAI'da `allow_delegation=True` olduğunda framework ajana otomatik olarak *"Delegate work to coworker"* tool'unu enjekte eder. ADK'nın `AgentTool`'u ve OpenAI Agents SDK'nın `handoff`'u aynı fikri paylaşır: **bir ajanı başka bir ajana bağlamak için özel protokole gerek yoktur, tool arayüzü yeterlidir.**

**c) ADK'nın `ToolContext`'i.** Tool yalnızca hesap yapmaz; durum okuyup yazabilir, akışı yönlendirebilir. `tool_context` parametresi şemadan otomatik çıkarılır — modele gönderilen `input_schema`'da görünmez, framework onu çalışma anında enjekte eder. Tool'un saf fonksiyondan harness bileşenine dönüştüğü nokta.

### 6.4 Gerçek ayrım ekseni

```
Kontrol kimde
├── Tamamen framework'te ────────── CrewAI          (hızlı prototip, zor hata ayıklama)
├── Framework + kesme noktaları ─── ADK, OpenAI Agents SDK, Pydantic AI
├── Sende, yapılandırılmış ──────── LangGraph       (grafik açık; kalıcılık, interrupt)
└── Tamamen sende ───────────────── ham SDK
```

İkinci ve bağlam yönetimi açısından daha önemli eksen: **durum nerede yaşıyor.**

| Framework | Durum |
|---|---|
| Ham SDK | Yalnızca `messages` — **bağlamda** |
| CrewAI | Task çıktıları arası aktarım — kısmen bağlamda |
| LangGraph | `State` + checkpoint — **bağlam dışında, diskte** |
| ADK | `session.state` + Memory servisi — **bağlam dışında, servis arkasında** |

> **Bulgu 5.** Framework'ler tool calling'i icat etmez, sarmalar. Modelin gördüğü şey framework'ten bağımsızdır. Gerçek farkları üç eksende toplanır: **(1)** döngünün sahibi, **(2)** durumun bağlamda mı dışarıda mı yaşadığı, **(3)** araya girme noktalarının birinci sınıf olup olmadığı. Uzun koşan ajanlarda fark (2)'de açılır.

---

## 7. Ölçek: tool sayısı büyüdüğünde

### 7.1 Token vergisi ve cache

Tool tanımları prompt'un 0. pozisyonundadır ve **her istekte** input token olarak sayılır. Üç sonucu vardır:

**a) Maliyet kalıcıdır.** 40 tool varsa hiçbiri kullanılmasa da her turda bedeli ödenir.

> **Saha kanıtı.** Vercel, ajanının tool'larının **%80'ini kaldırdıktan sonra daha yüksek güvenilirlik ve 3,5× daha düşük gecikme** ölçtüğünü bildirmiştir. Yetenek eksiltmenin performansı artırması, bu bölümün iki iddiasını birden doğrular: tool tanımları sabit bir token vergisidir **ve** kalabalık tool seti karar belirsizliği yaratır (§3.3). ⚠️ *Rakam Glean'in yazısından ikincil aktarımdır; metodoloji birincil kaynaktan doğrulanmalıdır.*

**b) Cache'i en kolay bozan yer burasıdır.** Geçersiz kılma hiyerarşisi:

| Değişiklik | tools | system | messages |
|---|:---:|:---:|:---:|
| Tool tanımı ekle/çıkar/sırala | ✗ | ✗ | ✗ |
| Model değiştir | ✗ | ✗ | ✗ |
| System prompt içeriği | ✓ | ✗ | ✗ |
| `tool_choice`, thinking aç/kapa | ✓ | ✓ | ✗ |
| Mesaj ekle | ✓ | ✓ | ✗ |

Sessiz bozucular:

```python
# ✗ kullanıcıya göre değişen tool seti → kullanıcılar arası hiç paylaşım yok
tools = build_tools_for(user)

# ✗ sıralamasız serialize → baytlar her istekte farklı
tools = [schema(t) for t in tool_set]        # set → sıra garantisi yok

# ✓ deterministik ve sabit
tools = sorted(ALL_TOOLS, key=lambda t: t["name"])
```

Konuşma ortasında tool ekleyip çıkarmak da aynı sebeple tüm cache'i geçersiz kılar. "Mod değiştirme" için tool seti takas edilmemeli; mod mesaj içeriğiyle geçirilmelidir.

**c) Ölçek çözümü ertelemedir.**

### 7.2 `defer_loading` + tool search

```python
tools = [
    {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
    {"name": "get_weather", "description": "...", "input_schema": {...},
     "defer_loading": True},
    # ... 40 tool daha, hepsi defer_loading
]
```

Ertelenen tool'lar istekte **bildirilir** ama modelin bağlamına **yüklenmez**. Model arama tool'uyla ihtiyacı olanı bulur; şemalar bağlamın **sonuna** eklendiği için prefix bozulmaz ve cache hayatta kalır.

Kısıtlar: arama tool'unun kendisi `defer_loading` olamaz ve en az bir tool ertelenmemiş olmalıdır — aksi hâlde `400 All tools have defer_loading set`. BM25 varyantı da mevcuttur (`tool_search_tool_bm25_20251119`).

Bu desen, Claude Code'un kendi harness'inde üretimde kullanılmaktadır (§4.2).

### 7.3 Konuşma ortası tool değişimi

Uygulama tool setinin değiştiğine karar verdiğinde (mod geçişi, erişilebilir hâle gelen kaynak, iptal edilen yetki), Opus 5 ve sonrasında cache bozulmadan değişiklik mümkündür:

```python
# beta: mid-conversation-tool-changes-2026-07-01
messages = [
    {"role": "user", "content": "..."},
    {"role": "system", "content": [
        {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "get_forecast"}},
    ]},
]
```

Eklenecek tool `tools[]` içinde `defer_loading: True` ile önceden bildirilmiş olmalıdır.

**Ayrım:** tool search = *model kendi bulur* (keşif). Mid-conversation changes = *uygulama karar verir* (kontrol).

---

## 8. Tool yüzeyi tasarımı: bash mı, özel tool mu

Model tool çağrısı üretir; harness onları işler. Çağrının **şekli**, harness'in ne yapabileceğini belirler.

Bash tool geniş programatik güç verir ama harness'e yalnızca opak bir komut string'i bırakır — her eylem için aynı şekil. Bir eylemi özel tool'a terfi ettirmek, harness'e tipli argümanlarla eyleme özgü bir kanca verir.

| Terfi gerekçesi | Örnek |
|---|---|
| **Güvenlik sınırı** | Geri döndürülmesi zor eylemler onay arkasına alınabilir. `send_email` gate'lenebilir; `bash -c "curl -X POST ..."` gate'lenemez |
| **Bayatlık kontrolü** | Özel `edit` tool'u, dosya son okumadan sonra değiştiyse yazmayı reddedebilir. Bash bu invaryantı uygulayamaz |
| **Render** | Soru sorma tool'a terfi edilirse modal olarak gösterilebilir, seçenek sunulabilir, cevaba kadar döngü bloklanabilir |
| **Zamanlama** | `glob`/`grep` gibi salt-okunur tool'lar paralel-güvenli işaretlenebilir. Bash'ten geçince harness paralel-güvenli `grep` ile paralel-güvensiz `git push`'u ayırt edemez |

**Kural:** kapsam için bash ile başla; gate, render, denetle veya paralelleştir gerekiyorsa özel tool'a terfi ettir.

---

## 9. Güvenlik

Tool argümanları **güvenilmez model çıktısıdır** — kullanıcı girdisi gibi ele alınmalıdır.

| Tool | Risk | Önlem |
|---|---|---|
| Bash | Rastgele komut çalıştırma | İzole ortam (container/VM/kısıtlı kullanıcı), çalıştırılabilir **allowlist**'i, shell operatörlerinin (`&&`, `\|`, `;`, `` ` ``, `$()`) reddi, timeout ve kaynak limitleri. **Blocklist yetersizdir** |
| Text editor / dosya | Path traversal | Her `path`'i kanonik forma çöz, proje kökü içinde kaldığını doğrula; `..`, symlink, `%2e%2e%2f` reddet |
| Memory | Sır sızıntısı, PII | API anahtarı/token asla yazılmaz; çok kullanıcılı sistemde kullanıcı başına dizin + kimlik doğrulama |
| Yan etkili tool'lar | Geri döndürülemez eylem | Onay kapısı (tool gövdesinde veya turn hook'unda) |

Runner kullanılırken de bu geçerlidir: runner tool fonksiyonlarını model istediğinde **otomatik** çalıştırır. Yan etkili tool'lar için gating tool'un içinde veya turn hook'unda yapılmalıdır.

---

## 10. Ölçüm metodolojisi

Bölümün iddialarının hepsi ölçülebilir. Üç deney önerilir.

### Deney 1 — "Bilme bağlamdadır" (Bulgu 3'ün doğrulaması)

Tek değişken: `description` string'i. Model, ağırlıklar, soru sabit.

```python
BASE = {"name": "search",
        "input_schema": {"type": "object",
                         "properties": {"query": {"type": "string"}},
                         "required": ["query"]}}

VARYANTLAR = {
    "boş":         "",
    "tanım":       "Web'de arama yapar.",
    "tetikleyici": ("Web'de arama yapar. Cevap konuşmada bulunmayan güncel bilgiye "
                    "bağlıysa (son olaylar, güncel fiyatlar) hafızadan cevaplama, "
                    "önce bunu çağır."),
}

N = 20
for etiket, desc in VARYANTLAR.items():
    cagirdi = 0
    for _ in range(N):
        r = client.messages.create(
            model="claude-opus-5", max_tokens=1024,
            tools=[{**BASE, "description": desc}],
            messages=[{"role": "user", "content": "Bugün Bitcoin kaç dolar?"}],
        )
        cagirdi += any(b.type == "tool_use" for b in r.content)
    print(f"{etiket:<12} → tetiklenme: {cagirdi}/{N}")
```

**Beklenen:** aşağı doğru tetiklenme oranı artar. İkinci varyant olarak `name`'i `search` → `xq7` yapıp description'ı sabit tut; ad da bir sinyaldir, düşüş beklenir.

### Deney 2 — Cache disiplini

Aynı ajan döngüsünü üç varyantla çalıştır, `usage.cache_read_input_tokens` izle:

| Varyant | Değişiklik | Beklenen |
|---|---|---|
| A (kontrol) | Sabit `tools` + sabit `system` | 2. turdan itibaren `cache_read > 0` |
| B (cache düşmanı) | `system`'e `f"Şu an: {datetime.now()}"` ekle | **Her turda `cache_read = 0`** |
| C (tool seti değişken) | Her turda `tools` sırasını karıştır | **Her turda `cache_read = 0`** |

### Deney 3 — `defer_loading` etkisi

30 sahte tool tanımla, 25'ine `defer_loading: True` ver. Tur başına toplam prompt token'ını iki durumda karşılaştır.

**Ölçüm kodu (üç deney için ortak):**

```python
u = response.usage
total_prompt = (u.input_tokens
                + (u.cache_creation_input_tokens or 0)
                + (u.cache_read_input_tokens or 0))
print(f"prompt={total_prompt:>7} fresh={u.input_tokens:>6} "
      f"cache_read={u.cache_read_input_tokens or 0:>7} out={u.output_tokens}")
```

> **Metodolojik not:** `usage.input_tokens` yanıltıcıdır — yalnızca cache'lenmemiş kısmı gösterir. Toplam prompt = `input_tokens + cache_creation + cache_read`. Maliyet raporlanırken üçü toplanmalıdır.

---

## 11. Yaygın hatalar

| Hata | Sonuç |
|---|---|
| Description'ı boş veya jenerik bırakmak | Model tool'u ya hiç çağırmaz ya rastgele çağırır |
| `CRITICAL: YOU MUST` tarzı agresif dil | Aşırı tetikleme |
| `response.content[0].text` okumak | İlk blok `thinking` veya `tool_use` olabilir; `refusal`'da patlar |
| Asistan turunu eklemeyip yalnızca metni eklemek | `tool_use` blokları kaybolur → API 400 |
| Bir `tool_use`'u yanıtsız bırakmak | API isteği reddeder |
| Paralel sonuçları ayrı mesajlara bölmek | Model sessizce paralel çağırmayı bırakır |
| Hatalı sonucu düşürmek | Yanıtsız `tool_use` → 400. `is_error: True` ile açıklayıcı mesaj döndürülmeli |
| Tool listesini isteğe göre üretmek | Cache tamamen ölür |
| `tool_use.input`'u ham string olarak eşleştirmek | Unicode / `/` kaçış farklarında bozulur |
| `strict: true`'yu `tool_choice`'a koymak | Yanlış yer — tool tanımına gider |
| Tüm tool'lara `defer_loading` vermek | `400 All tools have defer_loading set` |
| Şemasız tool'a (`bash`, `text_editor`) `input_schema` vermek | Gömülü davranış kaybolur, farklı bir tool olur |

---

## 12. Sonuç

Bölümün tezi beş bulguyla desteklendi:

| # | Bulgu |
|---|---|
| **1** | ReAct ile native tool calling arasındaki fark bir desen tercihi değil, katman farkıdır. Halüsinasyon, paralellik, doğrulama, hata sinyali ve muhakeme görünürlüğü — beşi de bu ayrımın doğrudan sonucudur |
| **2** | Docstring bir yorum değil, prompt'tur. Şema üretimi otomatiktir; anlam üretimi değildir |
| **3** | Bilme = bağlamda olma. Şeması yüklenmemiş bir tool, adı bilinse dahi çağrılamaz — gözlemsel olarak doğrulanmıştır |
| **4** | "Döngüyü kim çevirir" ile "halüsinasyonu ne engeller" bağımsız eksenlerdir. Halüsinasyon çıktı formatının, ergonomi döngü sahipliğinin fonksiyonudur |
| **5** | Framework'ler tool calling'i sarmalar, icat etmez. Modelin gördüğü şey framework'ten bağımsızdır; fark döngü sahipliği, durumun konumu ve müdahale noktalarındadır |

Bunların ortak sonucu, raporun genel tezini destekler: **bir ajanın yeteneği modelinin değil, bağlamının fonksiyonudur.** Tool katmanı bunun en somut örneğidir — aynı model, farklı `description` metniyle farklı davranır; aynı tool, şeması yüklenmediğinde erişilemez hâle gelir.

---

## Ekler

- Uygulama düzeyinde referans, dil eşlemeleri ve kontrol listesi: [Ek A](ek-a-tool-referans.md)
- İstek gövdesinin tam anatomisi ve tur tur bağlam akışı: [`02-bir-akisin-hayati.md`](02-bir-akisin-hayati.md)
