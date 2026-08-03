# LLM'e Tool Tanıtma — Kapsamlı Referans

Claude API üzerinden bir modele tool (fonksiyon) tanıtmanın tam mekaniği: şemanın nasıl üretildiği, prompt'un neresine yerleştiği, modelin nasıl karar verdiği, döngünün kim tarafından çevrildiği ve ölçek büyüdüğünde neyin bozulduğu.

Örnekler Python ağırlıklı; diğer diller için §14'teki eşleme tablosu.

---

## 1. Zihinsel model: tool nedir, ne değildir

Bir tool **modele verilen bir yetenek değil, bir sözleşme metnidir.**

| Sanılan | Gerçek |
|---|---|
| Model fonksiyonu çağırır | Model sadece "şu tool'u şu argümanlarla çağırmak istiyorum" diye yapılandırılmış bir blok üretir |
| Model kodu görür/çalıştırır | Model fonksiyon gövdesini asla görmez; hiçbir şey otomatik çalışmaz |
| `@tool` dekoratörü modele bir şey öğretir | Dekoratör sadece JSON şema üretir; öğreten şey o şemanın içindeki metindir |
| Tool bir kez tanıtılır | API stateless — tool tanımları **her istekte** yeniden gönderilir |

Akış her zaman şu üç adım:

```
1. Sen        → tools listesini gönderirsin
2. Model      → tool_use bloğu üretir (niyet beyanı), stop_reason: "tool_use"
3. Sen        → fonksiyonu çalıştırır, tool_result olarak geri beslersin
```

Adım 2 ile 3 arasında ne olduğu tamamen senin kontrolünde. İşte "harness" dediğimiz şeyin çekirdeği bu boşluk.

---

## 2. Tool tanımının anatomisi

Üç zorunlu alan:

```python
{
    "name": "get_weather",
    "description": "Bir şehrin güncel hava durumunu getirir. Kullanıcı güncel hava, sıcaklık veya yağış sorduğunda çağır.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Şehir ve ülke kodu, ör. 'Istanbul, TR'",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Sıcaklık birimi. Varsayılan celsius.",
            },
        },
        "required": ["location"],
    },
}
```

| Alan | Rolü |
|---|---|
| `name` | Dispatch anahtarı. Modelin döndüğü `tool_use.name` ile senin fonksiyon eşlemen bunun üzerinden buluşur. Spesifik ol: `get_current_weather` > `weather`. |
| `description` | **Kararın verildiği yer.** Aşağıda ayrı başlık. |
| `input_schema` | JSON Schema. Modelin argüman üretirken uyacağı yapı. |

Opsiyonel alanlar:

| Alan | Etki |
|---|---|
| `strict: true` | Argümanların şemaya birebir uyduğunu garanti eder. `additionalProperties: false` + `required` zorunlu. |
| `defer_loading: true` | Şema istekte bildirilir ama modelin bağlamına yüklenmez (bkz. §11). |
| `cache_control` | Bu tool tanımına kadar olan prefix'i cache'ler. |
| `allowed_callers` | Programmatic tool calling için — tool'un kod içinden çağrılmasına izin verir. |

---

## 3. Description yazma — en yüksek kaldıraçlı iş

Modelin elindeki tek sinyal bu metin. İki kural:

**Ne yaptığını değil, ne zaman çağrılacağını yaz.**

```python
# ✗ zayıf — model "ne zaman" sorusunu kendi tahmin etmek zorunda
"description": "Web'de arama yapar."

# ✓ tetikleyici koşullu
"description": (
    "Web'de arama yapar. Cevap konuşmada bulunmayan güncel bilgiye bağlıysa "
    "(son olaylar, güncel fiyatlar, sürüme özgü davranış) önce bunu çağır; "
    "hafızadan cevaplama."
)
```

Yeni Opus modelleri tool'lara daha temkinli uzanıyor. Prescriptive ("şunu sorduğunda çağır") açıklamalar, salt tanımlayıcı açıklamalara göre ölçülebilir tetiklenme farkı yaratıyor.

**Agresif dilden kaçın.** `CRITICAL: YOU MUST`, `Şüphedeyse bunu kullan` gibi kalıplar eski modeller için yazılmıştı. Güncel modeller sistem prompt'unu çok daha literal takip ediyor; bu ifadeler artık **aşırı tetiklemeye** yol açar. Model gereksiz yere tool çağırıyorsa çözüm daha fazla guardrail eklemek değil, dili yumuşatmak.

Her property'ye de açıklama yaz — model argümanları oradan biçimlendiriyor (`"Istanbul, TR"` mi `"istanbul"` mu?).

---

## 4. Şema kuralları

**Desteklenen:** temel tipler (object, array, string, integer, number, boolean, null), `enum`, `const`, `anyOf`, `allOf`, `$ref`/`$defs`, string formatları (`date-time`, `date`, `email`, `uri`, `uuid`, `ipv4`, `ipv6`, …), `additionalProperties: false`.

**Desteklenmeyen (structured outputs / strict modda):** özyinelemeli şemalar, sayısal kısıtlar (`minimum`, `maximum`, `multipleOf`), string kısıtları (`minLength`, `maxLength`), karmaşık dizi kısıtları, `additionalProperties`'in `false` dışında bir değeri.

Python ve TypeScript SDK'ları desteklenmeyen kısıtları otomatik olarak şemadan çıkarıp istemci tarafında doğruluyor.

### Strict mode

```python
{
    "name": "book_flight",
    "description": "Belirtilen destinasyona uçuş rezervasyonu yapar.",
    "strict": True,                      # ← tool tanımının kendisinde, tool_choice'ta değil
    "input_schema": {
        "type": "object",
        "properties": {
            "destination": {"type": "string"},
            "date": {"type": "string", "format": "date"},
            "passengers": {"type": "integer", "enum": [1, 2, 3, 4, 5, 6, 7, 8]},
        },
        "required": ["destination", "date", "passengers"],
        "additionalProperties": False,   # ← strict için zorunlu
    },
}
```

Notlar: yeni bir şema ilk istekte bir kerelik derleme gecikmesi yaratır (sonraki 24 saat cache'lenir). Strict mode; programmatic tool calling, `disable_parallel_tool_use` ve zorlanmış `tool_choice` ile birlikte kullanılamaz.

---

## 5. `@beta_tool`: şema nereden geliyor

Dekoratör derleme zamanında fonksiyonu introspect eder:

| Python'daki | JSON'daki karşılığı |
|---|---|
| fonksiyon adı | `name` |
| docstring'in ilk paragrafı | `description` |
| tip anotasyonu (`location: str`) | `properties[...].type` |
| docstring `Args:` satırı | `properties[...].description` |
| varsayılanı olmayan parametre | `required` içinde |

```python
from anthropic import beta_tool

@beta_tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Bir şehrin güncel hava durumunu getirir.

    Args:
        location: Şehir ve ülke kodu, ör. Istanbul, TR.
        unit: Sıcaklık birimi, "celsius" veya "fahrenheit".
    """
    return f"{location}: 22°C, güneşli"
```

→ tel üzerinde §2'deki JSON'a dönüşür.

**Sonuç:** docstring'i boş bırakırsan model kör kalır. `@tool` "otomatik anlıyor" demek değil, "otomatik şema üretiyor" demek. Docstring artık yorum değil, **prompt**.

Async için `@beta_async_tool`. TypeScript'te Zod ile `betaZodTool`, Zod istemiyorsan ham JSON Schema kabul eden `betaTool`.

---

## 6. Tel üzerindeki format

### İstek

```json
{
  "model": "claude-opus-5",
  "max_tokens": 16000,
  "tools": [ /* §2'deki tanımlar */ ],
  "messages": [{"role": "user", "content": "İstanbul'da hava nasıl?"}]
}
```

### Yanıt — model tool istediğinde

```json
{
  "stop_reason": "tool_use",
  "content": [
    {"type": "text", "text": "Kontrol ediyorum."},
    {
      "type": "tool_use",
      "id": "toolu_01A09q90qw90lq917835lq9",
      "name": "get_weather",
      "input": {"location": "Istanbul, TR", "unit": "celsius"}
    }
  ]
}
```

### Senin cevabın

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
      "content": "22°C, güneşli"
    }
  ]
}
```

`tool_use_id` eşleşmesi zorunlu. Her `tool_use` bloğu için tam olarak bir `tool_result` dönmeli — eksik bırakırsan API isteği reddeder.

> **Uyarı:** `tool_use.input` alanını asla ham string olarak eşleştirme. Modeller Unicode veya `/` kaçışını farklı üretebiliyor. SDK zaten parse edilmiş dict/object veriyor; ham HTTP kullanıyorsan `json.loads()` / `JSON.parse()` kullan.

---

## 7. Döngü — manuel

```python
import anthropic

client = anthropic.Anthropic()
messages = [{"role": "user", "content": user_input}]

while True:
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=16000,
        tools=tools,
        messages=messages,
    )

    if response.stop_reason == "end_turn":
        break

    # Sunucu taraflı tool iterasyon limitine takıldı — devam ettir
    if response.stop_reason == "pause_turn":
        messages.append({"role": "assistant", "content": response.content})
        continue

    # 1) Asistan turunu OLDUĞU GİBİ ekle — tool_use bloklarını kaybetme
    messages.append({"role": "assistant", "content": response.content})

    # 2) Tüm tool çağrılarını çalıştır
    results = []
    for block in response.content:
        if block.type == "tool_use":
            try:
                out = execute_tool(block.name, block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(out),
                })
            except Exception as e:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Hata: {e}",
                    "is_error": True,
                })

    # 3) HEPSİ tek user mesajında
    messages.append({"role": "user", "content": results})

final_text = next(b.text for b in response.content if b.type == "text")
```

Üç kritik nokta yorumlarda işaretli. Üçüncüsü özellikle sinsi: paralel çağrıların sonuçlarını ayrı mesajlara bölersen modele sessizce "paralel çağırma" öğretmiş olursun.

---

## 8. Döngü — tool runner

Runner yukarıdaki döngünün paketlenmiş hâli. İçeriden yaptığı işin mantıksal karşılığı:

```python
tool_map = {t.name: t for t in tools}
messages = list(initial_messages)

for _ in range(max_iterations):
    response = client.beta.messages.create(
        model=model, max_tokens=max_tokens,
        tools=[t.to_dict() for t in tools],   # her turda yeniden — API stateless
        messages=messages,
    )
    yield response                            # ← `for m in runner` burayı görür
                                              #    (tool'lar HENÜZ çalışmadı)
    if response.stop_reason != "tool_use":
        return

    messages.append({"role": "assistant", "content": response.content})
    results = []
    for block in response.content:
        if block.type == "tool_use":
            fn = tool_map[block.name]
            ...                               # çalıştır, tool_result üret
    messages.append({"role": "user", "content": results})
```

Kullanımı:

```python
runner = client.beta.messages.tool_runner(
    model="claude-opus-5",
    max_tokens=16000,
    tools=[get_weather],
    messages=[{"role": "user", "content": "İstanbul'da hava nasıl?"}],
)

for message in runner:
    print(message)
```

Kazandırdıkları: `stop_reason` kontrolü, isim→fonksiyon dispatch'i, `tool_use_id` eşleştirmesi, sonuçların tek mesajda toplanması, `max_iterations` sınırı.

### "Kontrol lazım" ≠ manuel döngü

Yaygın yanılgı. Runner'ın turn hook'ları şunları zaten karşılıyor:

| İhtiyaç | Runner'da nasıl |
|---|---|
| İnsan onayı / gating | Tool'un gövdesinde sor, reddedilirse `"kullanıcı reddetti"` döndür. Ya da yield edilen mesajdaki `tool_use` bloğunu incele, `set_messages_params()` ile tool çalışmadan önce müdahale et |
| Hata yakalama | `generate_tool_call_response()` ile sonucu modele gitmeden incele |
| Sonucu değiştirme | Aynı yerden `cache_control` ekleme, çıktıyı dönüştürme |
| Retry / parametre değişimi | Turu yeniden çalıştır, `max_iterations` ile döngüyü sınırla |
| Streaming | `stream=True` |

**Gerçekten manuel döngü gerektiren durumlar:** SDK'nın kuramadığı bir request şekli, özel transport, beta bağımlılığı istememek, döngü ortasında alakasız iş yapmak.

---

## 9. Paralel tool kullanımı

Varsayılan açık — model tek yanıtta birden fazla `tool_use` bloğu üretebilir.

```python
tool_calls = [b for b in response.content if b.type == "tool_use"]
# Eşzamanlı çalıştır
results = await asyncio.gather(*[run(b) for b in tool_calls])
# TEK user mesajında geri dön
messages.append({"role": "user", "content": [to_result(b, r) for b, r in zip(tool_calls, results)]})
```

Kapatmak için herhangi bir `tool_choice` değerine `"disable_parallel_tool_use": true` ekle.

---

## 10. `tool_choice` — zorlama

| Değer | Davranış |
|---|---|
| `{"type": "auto"}` | Model karar verir (varsayılan) |
| `{"type": "any"}` | En az bir tool çağırmak zorunda |
| `{"type": "tool", "name": "get_weather"}` | Belirtilen tool'u çağırmak zorunda |
| `{"type": "none"}` | Tool kullanamaz |

Tetiklenme oranını etkileyen diğer kaldıraç: **`effort`**. Yüksek effort'ta model tool'lara belirgin biçimde daha çok uzanır; düşük effort'ta işi olduğu gibi kapsayıp erken bitirir.

---

## 11. Prompt'taki konum, cache ve token maliyeti

Render sırası sabit:

```
tools  →  system  →  messages
```

Tool tanımları prompt'un **0. pozisyonunda**, system prompt'undan bile önce. Üç sonucu var:

### a) Token maliyeti kalıcı

Tool tanımları her istekte input token olarak sayılır. 40 tool varsa hiçbiri kullanılmasa da her turda bedelini ödersin. Ölçmek için:

```python
client.messages.count_tokens(model="claude-opus-5", tools=tools, system=system, messages=messages)
```

### b) Cache'i en kolay bozan yer

Prompt caching bir **prefix eşleşmesi**; 0. pozisyondaki tek bayt değişimi arkasındaki her şeyi geçersiz kılar.

```python
# ✗ kullanıcıya göre değişen tool seti → kullanıcılar arası hiç paylaşım yok
tools = build_tools_for(user)

# ✗ set üzerinden dolaşma / sıralamasız serialize → baytlar her istekte farklı
tools = [to_schema(t) for t in tool_set]

# ✓ deterministik
tools = sorted(ALL_TOOLS, key=lambda t: t["name"])
```

Konuşma ortasında tool ekleme/çıkarma da aynı sebeple tüm cache'i uçurur. "Mod değiştirme" için tool setini takas etme — modu mesaj içeriğiyle geçir.

Doğrulama: `response.usage.cache_read_input_tokens`. Aynı prefix'le peş peşe isteklerde sıfır kalıyorsa yukarıdaki sessiz bozuculardan biri iş başında.

Geçersiz kılma hiyerarşisi — her değişiklik her şeyi bozmuyor:

| Değişiklik | tools cache | system cache | messages cache |
|---|:---:|:---:|:---:|
| Tool tanımları (ekle/çıkar/sırala) | ✗ | ✗ | ✗ |
| Model değişimi | ✗ | ✗ | ✗ |
| System prompt içeriği | ✓ | ✗ | ✗ |
| `tool_choice`, thinking aç/kapa | ✓ | ✓ | ✗ |
| Mesaj içeriği | ✓ | ✓ | ✗ |

Yani `tool_choice`'u istek başına değiştirmek tools+system cache'ini bozmaz — asıl dikkat edilecek olan tool tanımı ve model değişimi.

### c) Ölçek çözümü: `defer_loading` + tool search

```python
tools = [
    {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
    {"name": "get_weather", "description": "...", "input_schema": {...},
     "defer_loading": True},
    # ... 40 tool daha, hepsi defer_loading
]
```

Ertelenen tool'lar istekte *bildirilir* ama bağlama *yüklenmez*. Model arama tool'uyla ihtiyacı olanı bulup yükler; şemalar sona eklendiği için prefix bozulmaz, cache hayatta kalır.

Kısıtlar: arama tool'unun kendisi `defer_loading` olamaz, ve en az bir tool ertelenmemiş olmak zorunda — yoksa `400 All tools have defer_loading set`.

BM25 varyantı da var: `tool_search_tool_bm25_20251119`.

### d) Konuşma ortasında tool değişimi (Opus 5+, beta)

Uygulama tool setinin değiştiğine karar verdiğinde (mod geçişi, yeni erişilebilir kaynak, iptal edilen yetki):

```python
# beta header: mid-conversation-tool-changes-2026-07-01
messages = [
    {"role": "user", "content": "..."},
    {"role": "system", "content": [
        {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "get_forecast"}},
    ]},
]
```

Eklenecek tool `tools[]` içinde `defer_loading: True` ile önceden bildirilmiş olmalı. `tool_removal` bloğu ya bir assistant mesajının hemen öncesinde ya da `messages` sonunda olmalı.

**Ayrım:** tool search = *model kendi buluyor* (keşif). Mid-conversation changes = *uygulama karar veriyor* (kontrol).

---

## 12. Tool yüzeyi tasarımı — bash mı özel tool mu?

Model tool çağrısı üretir; harness onları işler. Çağrının **şekli**, harness'in ne yapabileceğini belirler.

**Bash tool** geniş programatik güç verir ama harness'e sadece opak bir komut string'i bırakır — her eylem için aynı şekil. Bir eylemi **özel tool'a terfi ettirmek**, harness'e tipli argümanlarla eyleme özgü bir kanca verir.

Terfi ettirme gerekçeleri:

| Gerekçe | Örnek |
|---|---|
| **Güvenlik sınırı** | Geri döndürülmesi zor eylemler (mail gönderme, veri silme) onay arkasına alınabilir. `send_email` tool'unu gate'lemek kolay; `bash -c "curl -X POST ..."` değil |
| **Bayatlık kontrolü** | Özel `edit` tool'u, dosya son okumadan sonra değiştiyse yazmayı reddedebilir. Bash bu invaryantı uygulayamaz |
| **Render** | Soru sormayı tool'a terfi ettirirsen modal olarak gösterip seçenek sunabilir, cevaba kadar döngüyü bloklayabilirsin |
| **Zamanlama** | `glob`/`grep` gibi salt-okunur tool'lar paralel-güvenli işaretlenebilir. Aynı işler bash'ten geçince harness paralel-güvenli `grep` ile paralel-güvensiz `git push`'u ayırt edemez, hepsini seri çalıştırmak zorunda kalır |

**Pratik kural:** kapsam için bash ile başla; gate, render, denetle veya paralelleştirmen gereken eylemi özel tool'a terfi ettir.

Tool sayısını da odaklı tut — çok fazla tool modeli şaşırtır (ölçek gerekiyorsa §11c).

---

## 13. Tool türleri: kim çalıştırıyor

| Tür | Çalıştıran | Örnek | Not |
|---|---|---|---|
| **Kullanıcı tanımlı** | Sen | `get_weather` | Bu dokümanın ana konusu |
| **Anthropic tanımlı, istemci taraflı** | Sen | `bash_20250124`, `text_editor_20250728`, `memory_20250818` | Şemasız — sadece `type`+`name` bildir, `input_schema` **verme** |
| **Sunucu taraflı** | Anthropic | `web_search_20260209`, `web_fetch_20260209`, `code_execution_20260521`, tool search | `tools`'a ekle, gerisi otomatik; sonuç aynı yanıtta content bloğu olarak gelir |
| **MCP** | Anthropic orchestration | Üçüncü parti sunucular | `mcp_servers` + `mcp_toolset` birlikte zorunlu |

### Şemasız tool'lar

```python
tools = [
    {"type": "bash_20250124", "name": "bash"},
    {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
]
```

`name`/`type` çifti eşleşmek zorunda — `text_editor_20250728` ile `str_replace_editor` (eski isim) karıştırılırsa 400. `"bash"` adında kendi şemanla custom tool tanımlarsan gömülü davranışı almazsın, bambaşka bir tool olur.

> **Güvenlik:** bash komutları ve editor `path` değerleri **güvenilmez model çıktısıdır**. Bash için izole ortam (container/VM/kısıtlı kullanıcı), izin verilen çalıştırılabilir **allowlist**'i, shell operatörlerinin (`&&`, `|`, `;`, `` ` ``, `$()`) reddi, timeout ve kaynak limitleri. Blocklist yeterli değil. Editor için her `path`'i kanonik forma çözüp proje kökü içinde kaldığını doğrula (`..`, symlink, `%2e%2e%2f`).

### MCP connector

İki parça birlikte zorunlu, biri eksikse validation hatası:

```python
client.beta.messages.create(
    model="claude-opus-5", max_tokens=1024,
    betas=["mcp-client-2025-11-20"],
    mcp_servers=[{"type": "url", "url": "https://example/sse", "name": "example-mcp"}],
    tools=[{"type": "mcp_toolset", "mcp_server_name": "example-mcp"}],
    messages=[...],
)
```

### Programmatic tool calling

Standart tool kullanımında her çağrı bir gidiş-dönüş: model çağırır → sonuç bağlama girer → model düşünür → sonrakini çağırır. Üç ardışık eylem üç tur demek; ara verilerin çoğu bir daha kullanılmaz.

PTC modelin bunları **bir script'e derlemesine** izin verir. Script code execution container'ında çalışır; tool çağırdığında container duraklar, çağrı çalıştırılır, sonuç **modelin bağlamına değil, çalışan koda** döner. Sadece script'in nihai çıktısı modele gider.

```python
tools=[
    {"type": "code_execution_20260120", "name": "code_execution"},
    {"name": "get_orders", "description": "...", "input_schema": {...},
     "allowed_callers": ["code_execution_20260120"]},
]
```

Ne zaman: çok sayıda ardışık çağrı, veya bağlama girmeden filtrelenmesi gereken büyük ara sonuçlar.

### `pause_turn`

Sunucu taraflı tool'lar sunucu tarafında bir sampling döngüsü çalıştırır. Varsayılan 10 iterasyon limitine ulaşırsa yanıt `stop_reason: "pause_turn"` ile döner. Devam ettirmek için user mesajını ve asistan yanıtını tekrar gönder — **ek bir "Continue." mesajı ekleme**, API trailing `server_tool_use` bloğunu görüp otomatik devam ediyor.

> Runner'lar `pause_turn`'ü otomatik devam ettirmiyor (Python `anthropic` 0.116.0 / TS 0.110.0 itibarıyla). Duraklamış tur sessizce final mesaj olarak dönüyor — hata yok, uyarı yok, sadece kesik cevap. Sunucu tool'larını runner'a karıştırıyorsan her iterasyonda `stop_reason` kontrol et.

---

## 14. Diller arası eşleme

| | Tool tanımı | Runner |
|---|---|---|
| **Python** | `@beta_tool` dekoratörü | `client.beta.messages.tool_runner(...)` → `runner.until_done()` |
| **TypeScript** | `betaZodTool({...})` (Zod) veya `betaTool` (ham şema) | `client.beta.messages.toolRunner(...)` → `await runner` |
| **Go** | `toolrunner.NewBetaToolFromJSONSchema(...)` + `jsonschema` struct tag'leri | `client.Beta.Messages.NewToolRunner(...)` → `.RunToCompletion(ctx)` |
| **Java** | `Supplier<String>` implement eden anotasyonlu sınıf (`@JsonClassDescription`, `@JsonPropertyDescription`) | `client.beta().messages().toolRunner(...)`, `.addBeta("structured-outputs-2025-11-13")` |
| **Ruby** | `Anthropic::BaseTool` alt sınıfı + `input_schema` | `client.beta.messages.tool_runner(...)` |
| **PHP** | `BetaRunnableTool(definition:, run:)` | `$client->beta->messages->toolRunner(...)` |
| **C#** | Ham JSON şema | `client.Beta.Messages.ToolRunner(...)` → `BetaToolRunner` |
| **cURL** | Ham JSON | Yok — döngüyü kendin yazarsın |

Statik tipli SDK'larda (Go/Java/C#) tip adlarını tahmin etme; derleyici hatası en hızlı rehber.

---

## 15. Yaygın hatalar

| Hata | Sonuç |
|---|---|
| Docstring/description'ı boş veya jenerik bırakmak | Model tool'u ya hiç çağırmaz ya rastgele çağırır |
| `CRITICAL: YOU MUST` tarzı agresif dil | Aşırı tetikleme; güncel modeller literal takip ediyor |
| Sadece `response.content[0].text` okumak | `tool_use` bloğu geldiğinde veya `stop_reason: "refusal"` durumunda patlar |
| Asistan turunu ekleme yerine sadece metni eklemek | `tool_use` blokları kaybolur → API 400 |
| `tool_use_id` eşleştirmemek / bir çağrıyı yanıtsız bırakmak | API isteği reddeder |
| Paralel sonuçları ayrı mesajlara bölmek | Model sessizce paralel çağırmayı bırakır |
| Hatalı tool sonucunu düşürmek | Yanıtsız `tool_use` → 400. Bunun yerine `is_error: True` ile açıklayıcı mesaj dön |
| Tool listesini kullanıcıya/istek anına göre üretmek | Cache tamamen ölür (0. pozisyon) |
| `tool_use.input`'u ham string olarak eşleştirmek | Unicode/`/` kaçış farklarında bozulur |
| Konuşma ortasında tool ekleyip çıkarmak | Tüm prefix cache'i geçersiz |
| `pause_turn`'ü ele almamak | Sessizce kesik cevap |
| Şemasız tool'a `input_schema` vermek | Gömülü davranış kaybolur, farklı bir tool olur |
| Tüm tool'lara `defer_loading` vermek | `400 All tools have defer_loading set` |
| `strict: true`'yu `tool_choice`'a koymak | Yanlış yer — tool tanımının kendisine gider |

---

## 16. Kontrol listesi

Yeni bir tool eklerken:

- [ ] `name` spesifik ve fiil içeriyor (`search_invoices`, `invoices` değil)
- [ ] `description` **ne zaman çağrılacağını** söylüyor, sadece ne yaptığını değil
- [ ] Her property'nin kendi `description`'ı var, format örneği içeriyor
- [ ] Sabit değer kümeleri `enum` ile
- [ ] Gerçekten zorunlu olanlar `required`'da; gerisi opsiyonel + varsayılanlı
- [ ] Argüman geçerliliği kritikse `strict: true` + `additionalProperties: false`
- [ ] Yan etkili tool ise onay kapısı var (tool gövdesinde veya turn hook'unda)
- [ ] Girdi doğrulaması tool'un içinde — model çıktısı güvenilmez
- [ ] Hata yolu `is_error: True` ile açıklayıcı mesaj dönüyor
- [ ] Tool listesi deterministik sıralı ve istekler arası sabit
- [ ] `count_tokens` ile tool tanımlarının token maliyeti ölçüldü
- [ ] 15+ tool varsa `defer_loading` + tool search değerlendirildi
- [ ] `response.usage.cache_read_input_tokens` sıfır değil

---

## Kaynaklar

- Tool use genel bakış: `platform.claude.com/docs/en/agents-and-tools/tool-use/overview`
- Tool search: `.../tool-use/tool-search-tool`
- Programmatic tool calling: `.../tool-use/programmatic-tool-calling`
- Bash / text editor / memory tool: `.../tool-use/{bash,text-editor,memory}-tool`
- MCP: `modelcontextprotocol.io`
- Prompt caching: `platform.claude.com/docs/en/build-with-claude/prompt-caching`
