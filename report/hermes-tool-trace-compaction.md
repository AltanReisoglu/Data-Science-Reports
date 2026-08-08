# Hermes-Agent — Tool-Trace Compaction (Yalnızca Tool İzi): Her Detay

> **Kapsam:** Bu belge **sadece tool-trace compaction**'ı anlatır — yani tool **çağrıları ve sonuçlarının** context içindeki izini nasıl küçülttüğünü. Konuşma-özetleme (middle-turn LLM summarization), eşik hesabı, anti-thrash gibi *genel context compaction* konuları kapsam dışıdır; onlar ayrı bir mekanizmadır. Burada odak: **`_prune_old_tool_results`** ve **`prune_tool_results_only`** — Hermes'in *LLM'siz, deterministik, tool-sonucu* budama motoru.
>
> Kaynak: [../harnesses/hermes-agent/agent/context_compressor.py](../harnesses/hermes-agent/agent/context_compressor.py). Her satır koddan doğrulanmıştır; issue numaraları (#61932, #32106…) koddaki yorumlardan gelir.

---

## 0. Tool-trace compaction tam olarak nedir (ve ne DEĞİLDİR)

**Tool-trace** = context içindeki tool çağrılarının (`assistant` mesajındaki `tool_calls`) ve tool sonuçlarının (`role:"tool"` mesajları) toplamı. Bir ajan çalıştıkça bunlar birikir ve context'in en büyük şişme kaynağı olur.

**Tool-trace compaction** = bu tool mesajlarını, **konuşmanın anlamını LLM'e sormadan**, deterministik kurallarla küçültmek. Üç işlem yapar:
1. **Dedup** — aynı sonucu tekrar eden tool mesajlarını tekilleştir.
2. **Informative summary** — büyük tool *sonuçlarını* bilgi taşıyan tek satıra indir.
3. **Argument truncation** — büyük tool *çağrı argümanlarını* kısalt.

**Ne DEĞİL:** Bu bir LLM özetlemesi değildir. "Orta turn'leri auxiliary model'e verip özet yazdırma" tamamen ayrı bir faz (full compression'ın Adım 4'ü). Tool-trace budama **hiç model çağırmaz** — bu yüzden ucuz, hızlı ve "kalite-riski sıfır"dır.

**İki giriş noktası, aynı çekirdek:**
| Fonksiyon | Ne zaman | LLM özeti | Tetik |
|---|---|---|---|
| `_prune_old_tool_results` | Full compression'ın Adım 1'i olarak | — (sadece deterministik) | Eşik dolunca |
| `prune_tool_results_only` | Bağımsız, maliyet için | Yok (sadece bu 3 pass) | `proactive_prune_tokens` |

İkisi de **aynı 3 pass'i** çalıştırır; fark tetik ve tail-koruma stratejisidir.

---

## 1. Mesaj yapısı — neyi bozamayız (kritik ön bilgi)

Tool-trace, `messages[]` dizisinde şöyle görünür:
```python
{"role": "assistant", "tool_calls": [
    {"id": "call_7", "function": {"name": "read_file", "arguments": "{\"path\":\"a.py\"}"}}
]}
{"role": "tool", "tool_call_id": "call_7", "content": "<dosyanın 40.000 karakteri>"}
```

**Değişmez kural:** Her `tool` mesajı, `tool_call_id` ile bir `assistant.tool_calls[].id`'ye **eşleşmek zorundadır**. Bu çift bozulursa (ör. tool mesajını silersen ama çağrıyı bırakırsan) sağlayıcı **HTTP 400** döner. Bu yüzden budama:
- Tool sonucunun **içeriğini** değiştirir, **mesajın kendisini silmez** (`tool_call_id` korunur).
- Argümanları kısaltırken bile **geçerli JSON** bırakır (yoksa 400).

Budama en başta bir **indeks** kurar — her `tool_call_id`'yi tool adına + argümanlarına bağlar (böylece bir sonucu görünce onu üreten çağrıyı bilir):
```python
call_id_to_tool: Dict[str, tuple] = {}   # "call_7" -> ("read_file", "{\"path\":\"a.py\"}")
```
Bu indeks, "informatif özet"in nasıl `[read_file] read a.py …` üretebildiğinin sırrıdır: sonucun içeriği gitse de, onu *hangi çağrının* ürettiği bu indekste durur.

---

## 2. Prune boundary — nereye kadar budanır?

Budama, **son mesajları korur** (yakın bağlam dokunulmaz) ve sınırdan öncekini budar. Sınır iki yolla belirlenir:

### (a) Token bütçesiyle (öncelikli)
`protect_tail_tokens` verilmişse, sondan geriye doğru yürünür, token biriktirilir; bütçe aşılınca sınır çizilir.
```python
accumulated = 0
for i in range(len(result)-1, -1, -1):
    msg_tokens = _estimate_msg_budget_tokens(msg)
    if accumulated + msg_tokens > protect_tail_tokens and (len(result)-i) >= min_protect:
        boundary = i; break
    accumulated += msg_tokens
```

### (b) Mesaj sayısıyla (yedek)
`protect_tail_tokens` yoksa: `prune_boundary = len(result) - protect_tail_count`.

### Taban: `_MAX_TAIL_MESSAGE_FLOOR = 8`
Token bütçesi kullanılsa bile en az `min(protect_tail_count, len, 8)` mesaj korunur.
**Neden 8, 20 değil?** Koddaki yorum: *"varsayılan `protect_last_n=20`'yi sert taban yaparsak, bir sürü hacimli tool çıktısını budanamaz halde 'dondurur' — eski 'hiçbir şey sıkışamıyor' vakası geri gelir."* Yani 20 çok koruyucu; 8 hem yakın bağlamı korur hem büyük çıktıların budanmasına izin verir.

### İnce matematik uyarısı (koddan)
Bütçe-yürüyüşünü "korunan sayı"ya çevirip taban **count-space'te** uygulanır, sonra tekrar sınıra çevrilir:
```python
budget_protect_count = len(result) - boundary
protected_count = max(budget_protect_count, min_protect)   # count-space'te max doğru okunur
prune_boundary = len(result) - protected_count
```
**Neden?** Yorum: *"index-space'te `max` yönü ters çevirir (küçük index = DAHA çok korunan), o yüzden cömert bir bütçe sessizce `min_protect`'e kırpılırdı."* — pozisyon/sayı çevriminde işaret hatası tuzağı.

---

## 3. Pass 1 — Dedup (byte-identik tool sonuçlarını tekilleştir)

**Sorun:** Model aynı dosyayı 5 kez okuyabilir (`read_file a.py` × 5). Beş özdeş 40K blok = 200K boşa token.

**Çözüm:** Sondan geriye yürü, her tool sonucunun içeriğini **hash'le**. Aynı hash tekrar görülürse: **en yeni tam kopyayı tut**, eskileri bir **geri-referansla** değiştir.

```python
content_hashes: dict = {}  # hash -> (index, tool_call_id)
for i in range(len(result)-1, -1, -1):
    msg = result[i]
    if msg.get("role") != "tool": continue
    content = msg.get("content") or ""
    if isinstance(content, list): continue          # multimodal → atla
    if not isinstance(content, str): continue        # dict-envelope → hash'lenemez, atla
    if len(content) < 200: continue                  # dedup floor (küçükler değmez)
    # ... hash → görülmüşse geri-referansla değiştir
```

**Kritik özellikler:**
- **Tail-agnostik:** Dedup **korunan kuyruk dahil HER YERDE** çalışır. Neden güvenli? Çünkü **kayıpsız (lossless)** — en yeni tam kopya korunur, sadece *özdeş* eskiler referansa iner. Hiçbir benzersiz içerik kaybolmaz.
- **Dedup floor = 200 karakter:** 200'ün altını dedup etme (kazanç yok, gürültü).
- **Multimodal atlama:** Görsel içeren tool sonuçları (list veya `{_multimodal:True}` zarfı) metin olarak hash'lenemez → dokunulmaz.

**Örnek:**
```
ÖNCE:
[tool call_3] read a.py → <40K içerik>
[tool call_8] read a.py → <aynı 40K içerik>   ← özdeş
SONRA:
[tool call_3] read a.py → [Duplicate of call_8's result — see below]
[tool call_8] read a.py → <40K içerik>          ← en yeni tam kopya korunur
```

---

## 4. Pass 2 — Informative summary (büyük sonucu bilgi taşıyan satıra indir)

Sınırdan **önceki** her tool sonucu, `_demote_tool_result_at(idx)` ile budanır. Ama **her şey budanmaz** — önce elemeler:

```python
def _demote_tool_result_at(idx):
    content = result[idx].get("content") or ""
    if content.startswith("[") and " chars)" in content and len(content) < 400:
        return False   # zaten budanmış (kendi çıktımızı tekrar budama)
    if content.startswith("[screenshot removed"):
        return False   # zaten kaldırılmış
    if len(content) <= min_prune_chars:   # varsayılan 200
        return False   # küçük — budamaya değmez
    # skill_view koruması (#32106) — aşağıda
    tool_name, tool_args = call_id_to_tool.get(call_id, ("unknown",""))
    summary = _summarize_tool_result(tool_name, tool_args, content)
    result[idx] = {**msg, "content": summary}   # İÇERİĞİ değiştir, mesajı SİLME
    pruned += 1
```

**Dört eleme (neyi budaMA):**
1. **Zaten budanmış** — içerik `[...chars)` şeklinde ve <400 → kendi çıktımızı tekrar özetlersek her turn bozulur (yakınsamaz).
2. **Zaten kaldırılmış screenshot** — `[screenshot removed...`.
3. **Küçük** — `≤ min_prune_chars` (200). Bir özet, 200 karakterden kısa bir şeyi büyütebilir → asla küçültme.
4. **Korunan skill (#32106)** — aşağıda.

**İçerik yerine `_summarize_tool_result`** — informatif tek satır. Tüm tool dalları:

| Tool | Girdi → Üretilen satır |
|---|---|
| `terminal` | `[terminal] ran \`npm test\` -> exit 0, 47 lines output` (regex ile `exit_code` çekilir, komut 80 kar. kırpılır) |
| `read_file` | `[read_file] read config.py from line 1 (3,400 chars)` |
| `write_file` | `[write_file] wrote auth.py (N lines)` (`content`'teki `\n` sayısı) |
| `search_files` | `[search_files] content search for 'compress' in agent/ -> 12 matches` |
| `web_search` | `[web_search] query='langgraph subagent' (8,400 chars result)` |
| `web_extract` | `[web_extract] https://x.com (+2 more) (12,000 chars)` (dict→url unwrap, aşağıda) |
| `delegate_task` | `[delegate_task] 'auth'u denetle' (3,200 chars result)` (goal 60 kar.) |
| `execute_code` | `[execute_code] \`print(df.head())...\` (14 lines output)` (kod 60 kar.) |
| `browser_*` | `[browser_navigate] https://... (6,000 chars)` |
| `skill_view` | `[skill_view] name=X (5,200 chars) <pruned-marker>` (#32106, aşağıda) |
| bilinmeyen | `[<tool>] (N chars result)` (backstop) |

**Neden jenerik placeholder değil?** `[Old tool output cleared]` sıfır bilgi taşır → model "npm test geçti mi?" diye tekrar çalıştırır. `-> exit 0, 47 lines` kararı verdirir, tekrar çağrı gerekmez.

**Asla çökmez:** `_summarize_tool_result` her zaman try/except'le sarılı:
```python
try: return _summarize_tool_result_unguarded(...)
except Exception:
    return f"[{tool_name}] ({len(content):,} chars result)"   # backstop
```
**Neden?** Argümanlar *kalıcı history'den* gelir; model bazen bozuk (bool/int/None) üretir. Bozuk bir tarihsel çağrı budamayı çökertirse, budama **aynı history'de retry** eder → sonsuz **crash-loop**.

**Gerçek kenar-vaka (web_extract dict→str):** `web_search` sonucu `{"url":...}` dict'idir; model bunu doğrudan `web_extract`'a forward eder. Kod `urls[0]` dict ise unwrap eder (`.get("url") or .get("href")`) — yoksa aşağıdaki `+=` işlemi **`dict + str` TypeError** atıp tüm ön-budamayı çökertir.

**Ghost-skill savunması (#32106):** `skill_view` çıktısı `_SKILL_VIEW_PRUNE_MIN_CHARS = 5000`'i aşarsa budanır — ama düz metadata özeti **tehlikeli**:
```python
if content_len > 5000:
    return f"[skill_view] name={name} ({content_len:,} chars) " + _skill_pruned_marker(name)
```
**Neden?** Sadece "skill_view name=X (5000 chars)" yazarsan model skill'in **hâlâ yüklü** olduğunu sanır ("hayalet skill") ve talimatlarına göre davranır — ama içerik gitti! Kanonik marker modele: *(1) talimatlar YOK artık, (2) geri almak için `skill_view`'ı tekrar çağır.*
Ayrıca **aktif skill koruması:** `skill_view` sonucu, o skill "az önce yüklendi/aktif referanslı" ise (`protected_skills`) **budanmaz** (verbatim kalır) — Pass-4 basınç demotion'ı hariç.

---

## 5. Pass 3 — Tool_call argümanlarını kısalt (assistant tarafı)

Sadece tool *sonuçları* değil, tool *çağrılarının argümanları* da dev olabilir. Klasik örnek: `write_file` 50KB içerikle çağrılır — bu argüman `assistant.tool_calls[].function.arguments` içinde durur ve Pass 2'den (o tool sonucunu değil çağrıyı budar) etkilenmez.

```python
def _truncate_tool_call_args_at(idx):
    for tc in msg["tool_calls"]:
        args = tc["function"]["arguments"]
        if len(args) > 500:
            new_args = _truncate_tool_call_args_json(args)   # JSON İÇİNDE kısalt
```
`_truncate_tool_call_args_json` argümanları **ayrıştırılmış JSON yapısı içinde** kısaltır (500+ karakterlik alanları `[:200]+"...[truncated]"`) — **geçerli JSON kalır**.

**Neden geçerli JSON şart?** Koddaki yorum: *"aksi halde downstream sağlayıcılar, bozuk çağrı pencereden düşene kadar her turn 400 döner."* Yani ham string kesme = bozuk JSON = kalıcı 400.

**Kapsam:** Pass 3 yalnız sınırdan önceki (`prune_boundary`) assistant mesajlarında çalışır — korunan kuyruktaki çağrılara dokunmaz.

---

## 6. Pass 4 — Basınç demotion'ı (#61932)

Korunan bölgenin *kendisi* hâlâ yumuşak kuyruk bütçesini aşıyorsa (`protect_tail_tokens * 1.5`), bir **basınç pass'i** korunan bölgedeki büyük *tamamlanmış* tool/dosya çıktılarını bile demote eder — ama `_PRESSURE_KEEP_RECENT_MESSAGES = 3` son mesaj hep verbatim kalır (aktif kullanıcı isteği + en yeni tool çifti okunur olsun).

**Örnek:** Kuyrukta tek bir 50K `read_file` var, bütçe 20K. Normalde korunurdu; ama korunan bölge bütçe×1.5'i aştığı için basınç pass'i o çıktıyı informatif özete indirir — son 3 mesaj hariç.

---

## 7. Proactive yol — `prune_tool_results_only` (maliyet motoru)

Aynı 3 pass'i **LLM özet fazı olmadan**, full-compression eşiğinden **bağımsız** çalıştırır.

**Neden ayrı?** Büyük-pencereli modellerde `should_compress()` (≈%50) nadiren tetiklenir; o zamana kadar eski tool çıktıları history'de **her turn tekrar tekrar gönderilir**. Bu yol onları erken geri kazanır — **kalite-riskli LLM özeti olmadan**.

**Tail koruması burada COUNT ile** (`protect_last_n`), token bütçesiyle DEĞİL. Neden? Token bütçesi %50 eşikten türetilir (1M pencerede ≈100K) → tüm oturumu korur, hiçbir şey budanmaz.

**Üç kapı (hepsi fayda-freni):**
1. `proactive_prune_tokens <= 0` → kapalı, INPUT'u aynen döndür.
2. `current_tokens < proactive_prune_tokens` → henüz erken.
3. `before < _proactive_prune_rearm_tokens` → yeniden-kurulum bekliyor.

**Pass başına eşik:** Sadece Pass 2'nin tabanı `proactive_prune_min_result_chars = 8000`'e yükseltilir (proaktif yolda sadece gerçekten büyük sonuçlar hedeflenir). Pass 1 (dedup) ve Pass 3 kendi sabit tabanlarını korur. Dedup tail-agnostik (kayıpsız).

### PROMPT-CACHE SÖZLEŞMESİ (en kritik detay)
Budama, sağlayıcının **zaten gördüğü** mesaj gövdelerini yeniden yazar → en erken yeniden-yazılan mesajdan itibaren **cache'lenmiş prefix'i geçersiz kılar** (tıpkı bir compression boundary gibi).

Bu yüzden budama **ancak** `proactive_prune_min_reclaim_tokens = 4096` token geri kazanıyorsa **commit eder**; sonra history yeniden bir tetik-boyutu "runway" kadar büyüyene dek **disarm** olur (kendini kapatır). Bu iki kapının altında:
```python
return messages, 0   # INPUT nesnesi aynen döner (no-op)
```
Çağıran taraf sözleşmesi: `result is not input` ise bookkeeping yapılır (yani gerçekten değişti mi diye kimlik karşılaştırması).

**Neden bu kadar dikkatli?** Her budama cache'i bozar. Küçük bir budama için cache'i bozmak, kazandığından fazlasına mal olur. Bu yüzden "yeterince kazandırmıyorsan, dokunma" — cache-farkındalıklı fayda-freni.

**Kapasite kapısı:** Pahalı 3-pass taramasından **önce**, session store atomik kalıcılaştırma yapabiliyor mu bakılır (`archive_and_compact`). Yapamıyorsa (duck-typed/plugin store) budama kalıcı olamaz → her seferinde no-op olacağına, taramayı hiç yapma.

---

## 8. Uçtan uca örnek — tam bir trace, önce → sonra

**ÖNCE:**
```
[assistant] read_file("a.py")
[tool call_1] <40K a.py içeriği>                          ← eski
[assistant] search_files("login","auth/")
[tool call_2] 47 eşleşme, 300 satır (6K)                  ← eski
[assistant] read_file("a.py")            (tekrar!)
[tool call_3] <aynı 40K a.py içeriği>                     ← özdeş
[assistant] write_file("a.py", <50K yeni içerik>)         ← dev argüman
[tool call_4] "wrote a.py"
[assistant] terminal("pytest")
[tool call_5] <2000 satır test çıktısı> (15K)             ← YAKIN (korunur)
[assistant] Testler geçti.                                 ← YAKIN (korunur)
```

**SONRA (Pass 1→2→3, son 8/tail korunur):**
```
[assistant] read_file("a.py")
[tool call_1] [read_file] read a.py from line 1 (40,000 chars)   ← Pass2 informatif
[assistant] search_files("login","auth/")
[tool call_2] [search_files] content search for 'login' in auth/ -> 47 matches  ← Pass2
[assistant] read_file("a.py")
[tool call_3] [Duplicate of call_1's result]                     ← Pass1 dedup
[assistant] write_file("a.py", {"path":"a.py","content":"<...200 kar...>...[truncated]"})  ← Pass3 arg kısaltma
[tool call_4] "wrote a.py"
[assistant] terminal("pytest")
[tool call_5] <2000 satır test çıktısı> (15K)             ← korundu (yakın)
[assistant] Testler geçti.                                 ← korundu
```
**Sonuç:** ~110K → ~16K. Her `tool_call_id` korundu (0 tane 400 riski). Hiçbir benzersiz bilgi kaybolmadı — a.py'nin içeriği call_1'de özetlendi ama "40K okundu" bilinir; call_3 zaten kopyaydı; write argümanı kısaldı ama "a.py'ye yazıldı" bilinir; test çıktısı ve son mesaj korundu.

---

## 9. Guard/sabit tablosu (tek bakışta)

| Sabit / kapı | Değer | Rolü |
|---|---|---|
| `_MAX_TAIL_MESSAGE_FLOOR` | 8 | Korunan minimum son mesaj (20 dondururdu) |
| `_PRESSURE_KEEP_RECENT_MESSAGES` | 3 | Basınçta bile verbatim kalan son mesaj (#61932) |
| dedup floor | 200 kar. | Bunun altını dedup etme |
| `min_prune_chars` | 200 (varsayılan) | Pass 2: bunun altını özetleme (büyütür) |
| `_SKILL_VIEW_PRUNE_MIN_CHARS` | 5000 | skill_view bunun üstündeyse buda (+ghost-marker) |
| tool_call arg eşiği | 500 kar. | Pass 3: bunun üstünü kısalt |
| arg kısaltma head | 200 kar. | JSON içi alanı `[:200]+truncated` |
| `proactive_prune_min_result_chars` | 8000 | Proaktif Pass 2 tabanı |
| `proactive_prune_min_reclaim_tokens` | 4096 | Bu kadar kazanmıyorsa commit etme (cache) |
| `_PRUNED_TOOL_PLACEHOLDER` | metin | Son çare jenerik ("cleared to save context") |

---

## 10. POC eşlemesi — bizim tool-trace compaction'ımız

| Bizim POC (`poc/`) | Hermes tool-trace karşılığı |
|---|---|
| `_render_messages` — tool gövdesini fate ile yeniden yaz, `tool_call_id` koru | Pass 2 — `result[idx]={**msg,"content":summary}`, id korunur |
| `fate=ÖZET` (informatif not) | `_summarize_tool_result` (tool-tipine özel satır) |
| `fate=DEDUP` | Pass 1 — byte-identik dedup, en yeni tam kopya |
| Fayda-freni `est(note)<raw` | `min_prune_chars`/200-floor + `proactive_prune_min_reclaim_tokens` |
| `RECENT=2` pozisyon koruması | `_MAX_TAIL_MESSAGE_FLOOR=8` + token-tail + `_PRESSURE_KEEP_RECENT=3` |
| `tool_call_id` bütünlüğü (400 riski) | mesaj silinmez, içerik değişir; Pass 3 JSON geçerli kalır |
| — (bizde yok) | Pass 3 **argüman** kısaltma · ghost-skill marker (#32106) · **prompt-cache commit sözleşmesi** · basınç pass'i (#61932) |

**Ders:** Bizim POC'un tam kalbi (fate ile tool gövdesi yeniden yazma + informatif özet + dedup + id koruma + fayda-freni + son-N) Hermes'te **birebir** var. Eksiğimiz üç şey: (1) tool-*çağrı-argümanı* budama, (2) prompt-cache'i bozmadan commit disiplini, (3) ghost-skill gibi domain-özel işaretler.

---

## Kaynaklar
- [../harnesses/hermes-agent/agent/context_compressor.py](../harnesses/hermes-agent/agent/context_compressor.py):
  - `_prune_old_tool_results` (3 pass + boundary) · `prune_tool_results_only` (proaktif + cache sözleşmesi) · `_summarize_tool_result[_unguarded]` (tüm tool dalları) · `_truncate_tool_call_args_json` (Pass 3) · sabitler (`_MAX_TAIL_MESSAGE_FLOOR`, `_PRESSURE_KEEP_RECENT_MESSAGES`, `_SKILL_VIEW_PRUNE_MIN_CHARS`, `_PRUNED_TOOL_PLACEHOLDER`).
- Issue referansları (koddan): #61932 (basınç keep-recent), #32106 (ghost-skill).
- İlgili: [hermes-agent-harness.md](hermes-agent-harness.md) §6 · [poc/](../poc/) tool-trace compaction.
