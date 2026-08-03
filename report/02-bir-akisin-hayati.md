# Bir Akışın Hayatı: LLM'e Baştan Sona Tam Olarak Ne Veriliyor

> Bu bölüm tek bir soruyu cevaplıyor: *bir ajan oturumu boyunca modele hangi veri, hangi formatta, hangi sırayla, kaç kez gönderiliyor?*
>
> Örnekler uydurma değil — bu raporun yazıldığı Claude Code oturumunun kendi turları kullanılıyor.

---

## 0. Tek cümlelik cevap

**Modele her turda tek bir HTTP POST gövdesi gider ve o gövde bağlamın tamamıdır.** Konuşma "devam etmiyor"; her turda sıfırdan yeniden kuruluyor. 4. turdaki istek 1. turun her şeyini yeniden taşır.

```
POST https://api.anthropic.com/v1/messages
Content-Type: application/json
x-api-key: sk-ant-...
anthropic-version: 2023-06-01
anthropic-beta: <varsa>

{ ...bağlamın TAMAMI... }
```

Bu bölümün geri kalanı o `{ ... }` içinde ne olduğunu açıyor.

---

## 1. İstek gövdesinin anatomisi

### 1.1 Üst seviye alanlar

```jsonc
{
  // ── ZORUNLU ──────────────────────────────────────────────
  "model": "claude-opus-5",
  "max_tokens": 16000,            // thinking + metin TOPLAMI için sert tavan
  "messages": [ /* §4 */ ],

  // ── BAĞLAM KURULUMU ──────────────────────────────────────
  "system": "..." ,               // string VEYA content block dizisi
  "tools": [ /* §2 */ ],
  "tool_choice": {"type": "auto"},

  // ── MUHAKEME KONTROLÜ ────────────────────────────────────
  "thinking": {"type": "adaptive", "display": "summarized"},
  "output_config": {
    "effort": "high",             // low | medium | high | xhigh | max
    "format": { /* structured outputs */ },
    "task_budget": {"type": "tokens", "total": 64000}
  },

  // ── BAĞLAM YÖNETİMİ ──────────────────────────────────────
  "cache_control": {"type": "ephemeral"},
  "context_management": {
    "edits": [{"type": "compact_20260112"}]
  },

  // ── DİĞER ────────────────────────────────────────────────
  "stream": true,
  "metadata": {"user_id": "..."},
  "mcp_servers": [...],
  "container": {"skills": [...]},
  "betas": ["..."]
}
```

**Raporda vurgulanacak nokta:** burada "session_id", "conversation_id", "history" diye bir alan **yok**. Sürekliliğin taşıyıcısı sadece `messages` dizisi. Sunucu tarafında hiçbir konuşma durumu tutulmuyor.

### 1.2 Render sırası — bu sıra keyfi değil

Model prompt'u şu sırayla görür:

```
tools  →  system  →  messages
   ▲         ▲          ▲
 sabit     sabit     büyüyen
```

Prompt caching bir **prefix eşleşmesi** olduğu için sabit içerik önde olmak zorunda. Değişken bir şey (tarih, session id, kullanıcıya göre üretilen tool listesi) öne konursa arkasındaki her şeyin cache'i ölür. Sıra, cache mimarisinin dayattığı bir kısıt.

---

## 2. `tools` bloğu — 0. pozisyon

### 2.1 Format

```json
"tools": [
  {
    "name": "Read",
    "description": "Reads a file from the local filesystem.\n\n- `file_path` must be an absolute path.\n- Reads up to 2000 lines by default...",
    "input_schema": {
      "type": "object",
      "properties": {
        "file_path": {"type": "string", "description": "The absolute path to the file to read"},
        "offset":    {"type": "integer", "description": "The line number to start reading from"},
        "limit":     {"type": "integer", "description": "The number of lines to read"}
      },
      "required": ["file_path"]
    }
  }
]
```

Modelin bir tool hakkında bildiği **her şey** bu üç alandır. Fonksiyon gövdesi, dosya adı, implementasyon dili — hiçbiri görünmez. Description bir yorum değil, **prompt**.

### 2.2 Bu oturumda gerçekte ne var

Bağlamımdaki tool'lar iki gruba ayrılmış:

**Tam şemalı (11 adet):** `Agent`, `Artifact`, `AskUserQuestion`, `Bash`, `Edit`, `Read`, `ReportFindings`, `ScheduleWakeup`, `Skill`, `ToolSearch`, `Write`

**Sadece isim (≈18 adet):** `CronCreate`, `CronDelete`, `CronList`, `DesignSync`, `EnterPlanMode`, `EnterWorktree`, `ExitPlanMode`, `ExitWorktree`, `Monitor`, `NotebookEdit`, `PushNotification`, `RemoteTrigger`, `SendMessage`, `TaskOutput`, `TaskStop`, `TodoWrite`, `WebFetch`, `WebSearch`

İkinci grup şu notla geliyor:

> *Some tools are deferred and not listed above. When a deferred tool is surfaced later, its full schema appears as a `<function>{...}</function>` definition inside a `<functions>` block, and it is immediately callable.*

Bu **`defer_loading` + tool search** deseninin canlı hâli. API tarafındaki karşılığı:

```python
tools = [
    {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},

    {"name": "Read", "description": "...", "input_schema": {...}},          # yüklü

    {"name": "WebFetch", "description": "...", "input_schema": {...},
     "defer_loading": True},                                                 # ertelenmiş
    {"name": "CronCreate", "description": "...", "input_schema": {...},
     "defer_loading": True},
]
```

Ertelenmiş tool'lar istekte **bildirilir** ama modelin bağlamına **yüklenmez**. Model `ToolSearch` çağırırsa şemalar bağlamın **sonuna** eklenir — prefix bozulmadığı için cache hayatta kalır.

> **Kısıt:** arama tool'unun kendisi `defer_loading` olamaz ve en az bir tool ertelenmemiş olmalı. Aksi hâlde `400 All tools have defer_loading set`.

### 2.3 Maliyet

Tool tanımları **her istekte** input token olarak sayılır. Ölçüm:

```python
resp = client.messages.count_tokens(
    model="claude-opus-5",
    tools=tools,
    system=system,
    messages=messages,
)
print(resp.input_tokens)

# tools'suz aynı çağrıyı yapıp farkı al → tool'ların sabit vergisi
```

---

## 3. `system` bloğu

### 3.1 İki format

```jsonc
// Basit
"system": "You are a helpful coding assistant."

// Cache kontrollü — büyük prompt'lar için bunu kullan
"system": [
  {
    "type": "text",
    "text": "<uzun sabit sistem prompt'u>",
    "cache_control": {"type": "ephemeral"}   // buraya kadarki prefix cache'lenir
  },
  {
    "type": "text",
    "text": "<oturum başına değişen ama tur içinde sabit kısım>"
  }
]
```

`cache_control` işaretinden **önceki her şey** (tools dahil) tek bir cache girdisi olur.

### 3.2 Bu oturumun system prompt'unda ne var

Gözlemleyebildiğim bölümler:

| Bölüm | İçerik | Değişkenlik |
|---|---|---|
| Kimlik | "You are Claude Code, Anthropic's official CLI…" | Sabit |
| Harness kuralları | Çıktı markdown olarak render edilir, izin modları, hook'lar | Sabit |
| Davranış politikası | Onay gerektiren eylemler, dürüst raporlama, düzeltme disiplini | Sabit |
| Memory talimatı | Dizin yolu, dosya formatı, `MEMORY.md` index kuralı | Sabit |
| **Ortam** | cwd, git repo mu, branch, son commit'ler, platform, OS, model kimliği | **Oturum başında donmuş** |
| Scratchpad | `/tmp/claude-.../scratchpad` yolu | Oturum başına |
| Oturum yönergeleri | Slash komut davranışı, ultrareview açıklaması | Sabit |

**Kritik tasarım detayı:** git status bloğunda açıkça şu yazıyor —

> *"This is the git status at the start of the conversation. Note that this status is a snapshot in time, and will not update during the conversation."*

Canlı güncellenseydi system prompt her turda değişir, **tüm cache her turda ölürdü.** Bunun yerine anlık görüntü dondurulmuş ve modele bayat olduğu söylenmiş. Bu, cache ekonomisinin prompt tasarımını doğrudan şekillendirdiği en net örnek.

---

## 4. `messages` dizisi — büyüyen kısım

### 4.1 Roller

| Rol | Ne taşır |
|---|---|
| `user` | Kullanıcı mesajı **ve** tool sonuçları |
| `assistant` | Modelin ürettiği her şey (metin, thinking, tool_use) |
| `system` | Konuşma ortası operatör talimatı (model-gated: Opus 5, Opus 4.8, Fable 5, Mythos 5) |

`tool_result`'ların `user` rolünde olması ilk bakışta garip gelir ama mantıklı: modelin bakış açısından tool sonucu **dışarıdan gelen bir girdi**dir, kendi ürettiği bir şey değil.

### 4.2 Content block taksonomisi

**İstekte gönderebileceklerin:**

```jsonc
{"type": "text", "text": "..."}

{"type": "image",    "source": {"type": "base64", "media_type": "image/png", "data": "<b64>"}}
{"type": "image",    "source": {"type": "url",  "url": "https://..."}}
{"type": "image",    "source": {"type": "file", "file_id": "file_01..."}}

{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "<b64>"},
                     "title": "Q4 Report", "citations": {"enabled": true}}

{"type": "tool_use",    "id": "toolu_01...", "name": "Read", "input": {...}}
{"type": "tool_result", "tool_use_id": "toolu_01...", "content": "...", "is_error": false}

{"type": "thinking", "thinking": "...", "signature": "<opak>"}   // aynen geri gönder
{"type": "redacted_thinking", "data": "<opak>"}

{"type": "container_upload", "file_id": "file_01..."}            // code execution'a dosya
```

**Yanıtta alabileceklerin:**

| Blok | Ne zaman |
|---|---|
| `text` | Normal çıktı |
| `thinking` / `redacted_thinking` | Adaptive thinking açıkken |
| `tool_use` | Model tool çağırmak istediğinde |
| `server_tool_use` | Sunucu taraflı tool (web search vb.) çalıştığında |
| `web_search_tool_result`, `web_fetch_tool_result` | Sunucu tool sonuçları |
| `bash_code_execution_tool_result` | Code execution çıktısı (`.content.stdout`) |
| `text_editor_code_execution_tool_result` | Sandbox dosya işlemi |
| `tool_search_tool_result` | Tool search sonucu |
| `mcp_tool_use`, `mcp_tool_result` | MCP çağrıları |
| `compaction` | Sunucu bağlamı özetlediğinde — **aynen geri göndermelisin** |
| `fallback` | Refusal fallback devreye girdiğinde geçiş noktası |
| `container_upload` | Sandbox dosya ürettiğinde |

> **Sık yapılan hata:** `response.content[0].text` okumak. İlk blok `thinking` veya `tool_use` olabilir. Her zaman `type` üzerinden daralt.

---

## 5. Gerçek akış: bu oturumun turları

Aşağıda bu raporun yazıldığı oturumun `messages` dizisinin tur tur nasıl büyüdüğü var. Uzun içerikler `…` ile kısaltıldı, yapı gerçek.

### Tur 1 — kullanıcı sorusu

**İstek:**

```jsonc
{
  "model": "claude-opus-5",
  "max_tokens": 16000,
  "thinking": {"type": "adaptive"},
  "tools": [ /* 11 tam + 18 ertelenmiş */ ],
  "system": [ {"type": "text", "text": "You are Claude Code…", "cache_control": {"type": "ephemeral"}} ],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n# userEmail\nThe user's email address is altanscript@gmail.com.\n# currentDate\nToday's date is 2026-08-03.\n\n      IMPORTANT: this context may or may not be relevant to your tasks…\n</system-reminder>"},
        {"type": "text", "text": "<ide_opened_file>The user opened the file /home/altan/Desktop/adapted/lists/agents.md in the IDE. This may or may not be related to the current task.</ide_opened_file>"},
        {"type": "text", "text": "bir sorum olacak bu llme toolları nasıl tanıtabiliriz"}
      ]
    }
  ]
}
```

Burada üç şeye dikkat:

1. **`<system-reminder>` bir XML etiketi, ayrı bir rol değil.** Harness, operatör bilgisini `user` turunun içine metin olarak enjekte ediyor.
2. Enjeksiyon **kendini itibarsızlaştırıyor**: *"bu bağlam görevinle alakalı olabilir de olmayabilir de"*. Modelin bunu talimat sanmasını engelliyor.
3. IDE durumu (açık dosya) her turda yeniden enjekte ediliyor — değişken veri **sona** yazılıyor, system prompt'a değil. Cache disiplini.

**Yanıt:**

```jsonc
{
  "stop_reason": "tool_use",
  "content": [
    {"type": "thinking", "thinking": "Kullanıcı Türkçe soruyor… skill'in SKIP kuralı önce grep istiyor…"},
    {"type": "tool_use", "id": "toolu_01A…", "name": "Bash",
     "input": {"command": "grep -rEil 'openai|langchain_openai|…' .", "description": "Grep for LLM providers"}},
    {"type": "tool_use", "id": "toolu_01B…", "name": "Read",
     "input": {"file_path": "/home/altan/Desktop/adapted/lists/agents.md"}}
  ],
  "usage": {"input_tokens": 21430, "output_tokens": 312,
            "cache_creation_input_tokens": 19800, "cache_read_input_tokens": 0}
}
```

**İki `tool_use` bloğu tek yanıtta** — paralel tool kullanımı. ReAct'te yapısal olarak imkânsız olan şey.

### Tur 2 — tool sonuçları geri besleniyor

Harness her iki tool'u da çalıştırdı ve sonuçları **tek bir `user` mesajında** ekledi:

```jsonc
"messages": [
  { /* Tur 1 user mesajı — aynen duruyor */ },
  {
    "role": "assistant",
    "content": [ /* Tur 1 yanıtı — thinking + 2 tool_use, AYNEN */ ]
  },
  {
    "role": "user",
    "content": [
      {
        "type": "tool_result",
        "tool_use_id": "toolu_01A…",
        "content": "lists/agents.md\nlists/ai.md\nlists/reinforcement.md\n---\n.:\nlists\nRESEARCH.md\n…"
      },
      {
        "type": "tool_result",
        "tool_use_id": "toolu_01B…",
        "content": "     1\t## Task description\n     2\t\n     3\tProvide an overview…\n  …\n  1555\thttps://x.com/Vtrivedy10/status/2066366762111672348"
      },
      {
        "type": "text",
        "text": "<system-reminder>[Truncated: PARTIAL view — /home/altan/Desktop/adapted/lists/agents.md: showing lines 1-1555 of 2290 total (31289 tokens, cap 25000). Call Read with offset=1556 limit=1555 for the next page, or Grep to find a specific section. Do NOT answer from this page alone if the answer may be further in the file.]</system-reminder>"
      }
    ]
  }
]
```

Bu turun **üç ayrı ders**i var:

**a) Sonuçlar tek mesajda birleşti.** Ayrı mesajlara bölünseydi model sessizce paralel çağırmayı bırakırdı.

**b) Asistan turu bire bir korunuyor.** Sadece metni alıp `tool_use` bloklarını atarsan API 400 döner — çünkü `tool_result`'ların eşleşeceği `tool_use_id` kalmaz.

**c) Harness tool çıktısını kırptı ve bunu MODELE SÖYLEDİ.** ~25K token'lık sert bir tavan var; 31289 token'lık dosya kesildi. Kritik olan kesilmesi değil, **kesilmenin görünür olması** — sessiz kırpma olsaydı eksik veriyle tam güvenle cevap verirdim. Bu, harness tasarımındaki en önemli detaylardan biri.

Kod tarafındaki karşılığı:

```python
MAX_TOOL_RESULT_TOKENS = 25_000

def format_tool_result(tool_use_id: str, raw: str) -> dict:
    n = count_tokens(raw)
    if n <= MAX_TOOL_RESULT_TOKENS:
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": raw}

    head = truncate_to_tokens(raw, MAX_TOOL_RESULT_TOKENS)
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": (
            f"{head}\n\n"
            f"[Truncated: {n} tokens, cap {MAX_TOOL_RESULT_TOKENS}. "
            f"Devamı için offset kullan. Bu sayfa tek başına yeterli olmayabilir.]"
        ),
    }
```

### Tur 3 — skill enjeksiyonu

Model `Skill(claude-api)` çağırdı. Dönen `tool_result` devasa bir dokümandı:

```jsonc
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_01C…",
      "content": "Base directory for this skill: /tmp/…/claude-api\n\n# Building LLM-Powered Applications with Claude\n\n## Before You Start\n…\n<doc path=\"python/claude-api/tool-use.md\">\n# Tool Use — Python\n…\n</doc>\n<doc path=\"shared/tool-use-concepts.md\">\n…\n</doc>"
    }
  ]
}
```

**Skill mekanizmasının tamamı bu:** skill "modele öğretilen bilgi" değil, **bir tool sonucu olarak bağlama enjekte edilen prompt**. Özel bir kanal, ayrı bir rol, gizli bir mekanizma yok.

Üç katmanlı ekonomisi:

| Katman | Nerede | Bu oturumdaki boyut |
|---|---|---|
| 1. `name` + tek satır açıklama | **Her turda** system-reminder'da | ~8 skill × ~25 token = **~200 token sürekli** |
| 2. `SKILL.md` + bundle'lanmış `<doc>`'lar | Çağrılınca tool_result olarak | **~50.000 token, bir kez** |
| 3. Referans dosyalar / script'ler | Model ayrıca isterse | Değişken (çekilmedi) |

Hepsi baştan yüklü olsaydı oturum ~400K token ile başlardı. Katman 1'in sabit ~200 token'lık maliyeti, 50K'lık yükü *koşullu* hâle getirmenin bedeli.

**Yükleme kararı da kural tabanlıydı.** Skill açıklaması bir TRIGGER/SKIP bloğu taşıyordu:

> *TRIGGER — read BEFORE opening the target file… whenever the task is LLM-shaped with provider unstated (agent/MCP/tool-definition/…)*
> *SKIP only when another provider is being worked on… run this grep FIRST if no provider named*

Bu yüzden Tur 1'de önce grep çalıştı. **Skill'in kendi açıklaması, ne zaman yükleneceğini tarif eden bir yönlendirici.**

### Tur 4+ — birikim

```
Tur 1:  system + tools + user                          ≈  21K token
Tur 2:  + assistant(2×tool_use) + 2×tool_result        ≈  48K token
Tur 3:  + assistant + skill tool_result (~50K)         ≈ 100K token
Tur 4:  + assistant(Write, dosya içeriği ~9K) + result ≈ 112K token
Tur 5:  + assistant(uzun metin) + user                 ≈ 120K token
   ⋮
```

Monoton büyüme. Bu yüzden §7'deki basınç yönetimi mekanizmaları var.

---

## 6. `<system-reminder>` — harness'in modele fısıldama kanalı

Bu oturumda gözlemlediğim enjeksiyon türleri:

```xml
<!-- Oturum bağlamı — ilk turda -->
<system-reminder>
# userEmail / # currentDate …
IMPORTANT: this context may or may not be relevant…
</system-reminder>

<!-- IDE durumu — her turda -->
<ide_opened_file>The user opened the file … This may or may not be related.</ide_opened_file>

<!-- Yetenek envanteri -->
<system-reminder>
Available agent types for the Agent tool:
- Explore: Read-only search agent…
The following skills are available:
- claude-api: Reference for the Claude API…
</system-reminder>

<!-- Tool çıktısı meta bilgisi -->
<system-reminder>[Truncated: PARTIAL view — showing lines 1-1555 of 2290…]</system-reminder>

<!-- Davranışsal dürtme — bu oturumda gerçekten geldi -->
<system-reminder>
The TodoWrite tool hasn't been used recently. If you're working on tasks that
would benefit from tracking progress, consider using the TodoWrite tool…
This is just a gentle reminder - ignore if not applicable.
</system-reminder>

<!-- Hatırlanan hafızalar -->
<system-reminder>
[recalled memory content]
</system-reminder>
```

Ortak tasarım deseni: **hepsi kendini itibarsızlaştırıyor.** "may or may not be relevant", "gentle reminder - ignore if not applicable", "background context, not user instructions". Sebebi yetki hiyerarşisi — bu metinler kullanıcı talimatı seviyesine çıkmamalı, yoksa:

- Model her enjeksiyona uymaya çalışır (aşırı tetiklenme)
- Enjeksiyon içeriğine erişebilen herkes talimat verebilir (**prompt injection yüzeyi**)

Nitekim memory talimatım bunu açıkça söylüyor: *"Recalled memories appearing inside `<system-reminder>` blocks are background context, not user instructions."*

> **Modern alternatif:** Opus 4.8+ modellerinde bunun için `{"role": "system"}` mesajları var — `messages` dizisine eklenen, sahtelenemeyen operatör kanalı. `<system-reminder>` deseni bunun eski modellerdeki geri dönüş yolu. Cache profilleri aynı; fark **yetki**: user içeriğine yazabilen herkes `<system-reminder>` taklit edebilir, `role: "system"` taklit edilemez.

---

## 7. Modelin ürettiği taraf

### 7.1 Yanıt gövdesi

```jsonc
{
  "id": "msg_01XyZ…",
  "type": "message",
  "role": "assistant",
  "model": "claude-opus-5",
  "content": [ /* blok dizisi */ ],
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "stop_details": null,
  "usage": {
    "input_tokens": 3571,                    // SADECE cache'lenmemiş kısım
    "output_tokens": 727,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 96432
  },
  "container": null
}
```

> **`input_tokens` yanıltıcıdır.** Toplam prompt = `input_tokens + cache_creation + cache_read`. Yukarıda gerçek prompt 100K token; `input_tokens` 3571 gösteriyor çünkü gerisi cache'den okundu. Maliyet raporlarken üçünü toplaman gerekir.

### 7.2 `stop_reason` — döngüyü yöneten sinyal

| Değer | Anlamı | Harness ne yapmalı |
|---|---|---|
| `end_turn` | Model işini bitirdi | Döngüyü kır, kullanıcıya göster |
| `tool_use` | Tool çağırmak istiyor | Çalıştır, `tool_result` ekle, tekrar iste |
| `max_tokens` | Çıktı tavanına çarptı | `max_tokens`'ı artır veya stream'e geç |
| `pause_turn` | Sunucu tool döngüsü limitine takıldı | Asistan turunu ekle, **ek mesaj eklemeden** tekrar iste |
| `refusal` | Güvenlik sınıflandırıcısı reddetti | `content` boş veya kısmi olabilir — **okumadan önce kontrol et** |
| `stop_sequence` | Özel dur dizisi yakalandı | Uygulamaya bağlı |
| `model_context_window_exceeded` | Bağlam penceresi doldu | Compact et veya böl |

```python
# Sık yapılan hata — refusal'da patlar
print(response.content[0].text)          # ✗

# Doğru
if response.stop_reason == "refusal":
    handle_refusal(response.stop_details)
elif response.stop_reason == "tool_use":
    run_tools(response)
else:
    for b in response.content:
        if b.type == "text":
            print(b.text)
```

### 7.3 Streaming — aynı veri, parça parça

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_01…","usage":{"input_tokens":3571,…}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Kullanıcı…"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01…","name":"Read","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"file_"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"path\":\"/home/…\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":312}}

event: message_stop
data: {"type":"message_stop"}
```

Dikkat: `tool_use` girdisi **`input_json_delta` ile parça parça** gelir — birleştirilip parse edilmesi gerekir. SDK bunu senin için yapar (`stream.get_final_message()`).

---

## 8. Bağlam basıncı — üç savunma hattı

| Hat | Nerede çalışır | Yöntem | Kayıp |
|---|---|---|---|
| **Tool çıktı kırpma** | Harness, tool sonucu üretilirken | Sert token tavanı + görünür uyarı | Sayfalama turu gerekir |
| **Context editing** | Sunucu, istek işlenirken | Eski tool sonuçlarını / thinking'leri **siler** | Geri dönülemez |
| **Compaction** | Sunucu, pencere dolarken | Eski bağlamı **özetler** | Detay kaybı |
| **Subagent** | Harness, ayrı bağlam açarak | Gürültüyü başka pencerede tutar | Soğuk başlangıç |

### Context editing — budama

```python
client.beta.messages.create(
    model="claude-opus-5",
    max_tokens=16000,
    betas=["context-management-2025-06-27"],
    context_management={
        "edits": [
            {"type": "clear_tool_uses_20250919", "clear_tool_inputs": True},
            {"type": "clear_thinking_20251015"},
        ]
    },
    tools=tools,
    messages=messages,
)
```

### Compaction — özetleme

```python
response = client.beta.messages.create(
    model="claude-opus-5",
    max_tokens=16000,
    betas=["compact-2026-01-12"],
    context_management={"edits": [{"type": "compact_20260112"}]},
    messages=messages,
)

# ⚠️ TAM content'i geri ekle — sadece metni değil.
# compaction bloğu kaybolursa sunucu özeti bir sonraki turda kullanamaz.
messages.append({"role": "assistant", "content": response.content})
```

Bu, en sık yapılan hatalardan biri: `next(b.text for b in response.content)` ile metni çekip onu geçmişe eklemek. `compaction` bloğu düşer, sıkıştırma durumu sessizce kaybolur.

### Subagent — izolasyon

Bu oturumdaki `Agent` tool açıklaması kararın maliyetini açıkça yazıyor:

> *"Each spawn starts cold and re-derives context you already have — it's the expensive path."*
> *"The agent's final report is not shown to the user — relay what matters."*

Kazanç: 50 dosya taranacaksa o 50 dosya ana bağlama hiç girmez, sadece rapor girer.
Bedel: subagent senin bana anlattıklarının hiçbirini bilmez.

---

## 9. Cache: neden bu mimari

Tüm yerleşim kararlarının arkasındaki tek kural: **caching prefix eşleşmesidir; N. bayttaki değişiklik ≥N konumundaki her cache'i geçersiz kılar.**

### Geçersiz kılma hiyerarşisi

| Değişiklik | tools | system | messages |
|---|:---:|:---:|:---:|
| Tool tanımı ekle/çıkar/sırala | ✗ | ✗ | ✗ |
| Model değiştir | ✗ | ✗ | ✗ |
| System prompt içeriği | ✓ | ✗ | ✗ |
| `tool_choice`, thinking aç/kapa | ✓ | ✓ | ✗ |
| Mesaj ekle | ✓ | ✓ | ✗ |

Yani `tool_choice` değiştirmek tools+system cache'ini korur; tool listesi değiştirmek her şeyi öldürür.

### Sessiz bozucular

```python
# ✗ system prompt'ta zaman damgası
system = f"Bugün {datetime.now()}. Sen bir asistansın…"

# ✗ kullanıcıya göre tool listesi
tools = build_tools_for(user)

# ✗ sıralamasız serialize
tools = [schema(t) for t in tool_set]          # set → her seferinde farklı sıra

# ✓
system = "Sen bir asistansın…"                 # donmuş
tools  = sorted(ALL_TOOLS, key=lambda t: t["name"])
# değişken veri → messages'ın SONUNA
messages.append({"role": "user", "content": f"<context>Tarih: {today}</context>\n{user_msg}"})
```

Doğrulama:

```python
u = response.usage
print(f"cache_read={u.cache_read_input_tokens}  create={u.cache_creation_input_tokens}  fresh={u.input_tokens}")
# Aynı prefix'le peş peşe isteklerde cache_read sıfır kalıyorsa yukarıdakilerden biri iş başında
```

---

## 10. Kendi gözünle gör — çalışan kod

Teoriyi doğrulamanın en hızlı yolu, giden gövdeyi diske yazmak.

```python
"""
Ajan döngüsünün her turunda giden isteği ve gelen yanıtı JSON olarak kaydeder.
Rapor için: turn-01-request.json … dosyalarını açıp bağlamın nasıl büyüdüğünü göster.
"""
import json, pathlib, anthropic

OUT = pathlib.Path("wire-log"); OUT.mkdir(exist_ok=True)
client = anthropic.Anthropic()

TOOLS = [{
    "name": "read_file",
    "description": "Yerel bir dosyanın içeriğini döndürür. Kullanıcı bir dosyanın "
                   "içeriğini sorduğunda veya bir dosyayı incelemen gerektiğinde çağır.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Mutlak dosya yolu"}},
        "required": ["path"],
    },
}]

SYSTEM = [{"type": "text",
           "text": "Sen bir dosya inceleme asistanısın. Kısa ve net cevap ver.",
           "cache_control": {"type": "ephemeral"}}]

MAX_RESULT_CHARS = 4000

def execute(name: str, args: dict) -> tuple[str, bool]:
    if name != "read_file":
        return f"Bilinmeyen tool: {name}", True
    try:
        text = pathlib.Path(args["path"]).read_text()
    except Exception as e:
        return f"Hata: {e}", True
    if len(text) > MAX_RESULT_CHARS:          # kırpma + GÖRÜNÜR uyarı
        text = (text[:MAX_RESULT_CHARS] +
                f"\n\n[Kırpıldı: toplam {len(text)} karakter, tavan {MAX_RESULT_CHARS}. "
                f"Devamı için offset kullan; bu sayfa tek başına yeterli olmayabilir.]")
    return text, False

def run(user_input: str, max_turns: int = 10):
    messages = [{"role": "user", "content": user_input}]

    for turn in range(1, max_turns + 1):
        request = {
            "model": "claude-opus-5",
            "max_tokens": 8000,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "medium"},
            "system": SYSTEM,
            "tools": TOOLS,
            "messages": messages,
        }
        (OUT / f"turn-{turn:02d}-request.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2))

        response = client.messages.create(**request)
        (OUT / f"turn-{turn:02d}-response.json").write_text(
            response.to_json(indent=2))

        u = response.usage
        total_prompt = (u.input_tokens
                        + (u.cache_creation_input_tokens or 0)
                        + (u.cache_read_input_tokens or 0))
        print(f"tur {turn:>2} | stop={response.stop_reason:<9} "
              f"prompt={total_prompt:>7} (fresh={u.input_tokens:>6} "
              f"cache_read={u.cache_read_input_tokens or 0:>7}) "
              f"out={u.output_tokens}")

        if response.stop_reason == "refusal":
            print("reddedildi:", response.stop_details); return
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content}); continue
        if response.stop_reason != "tool_use":
            for b in response.content:
                if b.type == "text": print("\n" + b.text)
            return

        # asistan turunu AYNEN ekle — tool_use blokları korunmalı
        messages.append({"role": "assistant", "content": response.content})

        results = []
        for block in response.content:
            if block.type == "tool_use":
                out, is_error = execute(block.name, block.input)
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": out,
                                "is_error": is_error})
        # HEPSİ tek user mesajında
        messages.append({"role": "user", "content": results})

if __name__ == "__main__":
    run("README dosyasını oku ve ne işe yaradığını bir cümleyle özetle. Yol: ./README.md")
```

**Beklenen çıktı deseni:**

```
tur  1 | stop=tool_use  prompt=   1180 (fresh=  1180 cache_read=      0) out=95
tur  2 | stop=end_turn  prompt=   3402 (fresh=  2222 cache_read=   1180) out=48
```

İkinci turda `cache_read` sıfırdan büyük → prefix cache tuttu. Rapora bu iki satırı ve `wire-log/turn-01-request.json` ile `turn-02-request.json` arasındaki diff'i koyarsan, "bağlam her turda yeniden gönderiliyor" iddian **gözlemsel kanıt** hâline gelir.

### Ölçüm deneyi önerisi

Aynı script'i iki varyantla çalıştır:

| Varyant | Değişiklik | Beklenen |
|---|---|---|
| A (kontrol) | Yukarıdaki hâli | `cache_read` 2. turdan itibaren > 0 |
| B (cache düşmanı) | `SYSTEM`'e `f"Şu an: {datetime.now()}"` ekle | `cache_read` **her turda 0** |
| C (defer) | 30 sahte tool ekle, 25'ine `defer_loading: True` | Tur başına prompt token'ı belirgin düşer |

Üç varyantın `cache_read` / toplam prompt grafiği, raporun en güçlü tek görseli olur.

---

## 11. Özet — akış boyunca ne veriliyor

| Ne | Format | Ne sıklıkta | Nerede |
|---|---|---|---|
| Tool şemaları | JSON Schema | **Her turda** | `tools`, 0. pozisyon |
| Ertelenmiş tool isimleri | Metin listesi | Her turda | System prompt notu |
| Kimlik + kurallar | Metin | Her turda (cache'li) | `system` |
| Ortam (cwd, git, tarih) | Metin, **donmuş** | Her turda (cache'li) | `system` |
| Skill envanteri | Ad + 1 satır | Her turda | `<system-reminder>` |
| Skill gövdesi | Uzun metin | **Bir kez**, çağrılınca | `tool_result` |
| Memory index | Satır listesi | Oturum başında | `system` / reminder |
| Memory içeriği | Dosya metni | Gerektiğinde | `<system-reminder>` veya `Read` sonucu |
| IDE durumu | XML etiketi | Her turda | `user` turunun başında |
| Kullanıcı mesajı | Metin | Bir kez, sonra geçmişte | `user` |
| Model çıktısı | Blok dizisi | Bir kez, sonra geçmişte | `assistant` |
| Tool sonuçları | `tool_result` blokları | Bir kez, sonra geçmişte | `user` |
| Kırpma uyarısı | XML etiketi | Kırpma olunca | `tool_result` yanında |
| Davranışsal dürtmeler | XML etiketi | Koşullu | `user` turunda |

**Değişmez kural:** en sağdaki sütunda `system` veya `tools` yazan her şey **prefix'te**, yani her baytı cache'i etkiliyor. `messages`'ın sonuna yazan her şey ucuz.

Bir harness'in kalitesi, hangi bilgiyi hangi satıra koyduğuna bakılarak ölçülebilir.
