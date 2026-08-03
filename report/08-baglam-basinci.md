# 8. Bağlam Basıncı ve Token Ekonomisi

> **Bölümün tezi:** Bağlam monoton büyür; hiçbir ajan bu büyümeyi kendiliğinden durduramaz. Dört savunma hattı vardır ve bunlar birbirinin alternatifi değil, **farklı katmanlarda çalışan tamamlayıcı mekanizmalardır.**

---

## 8.1 Problem: monoton büyüme

API durumsuzdur; her turda tüm geçmiş yeniden gönderilir (§02). Bu raporun yazıldığı oturumun ölçülen büyümesi:

```
Tur 1:  system + tools + user                              ≈  21K token
Tur 2:  + assistant(2×tool_use) + 2×tool_result            ≈  48K
Tur 3:  + assistant + skill tool_result (~50K)             ≈ 100K
Tur 4:  + assistant(Write, dosya içeriği) + result         ≈ 112K
Tur 5:  + assistant(uzun metin) + user                     ≈ 120K
   ⋮
```

Ve iki ayrı bozulma türü devreye girer (§01.4): **context rot** (uzunluk ekseni) ve **lost in the middle** (konum ekseni).

Yaygın bir yanılgıyı burada kapatmak gerekir:

> *"**Waiting for larger context windows might seem like an obvious tactic. But** it's likely that for the foreseeable future, context windows of all sizes will be subject to context pollution and information relevance concerns."*
> — Anthropic, Eyl 2025

Bu öngörü doğrulandı: pencereler 200K'dan 1M'e çıktı, teknikler hâlâ gerekli.

---

## 8.2 Hat 1 — Tool çıktısı kırpma (ingestion'da)

**En ucuz savunma, veriyi hiç içeri almamaktır.**

ML Mastery'nin tespiti:

> *"Tool responses — especially search and API results — are often the **largest cost**. **Filtering and trimming them at ingestion is more effective than compressing later**; only keep what's needed for the next step."*

### Doğrudan gözlem

Rapor oturumunda `lists/agents.md` okunduğunda dönen `tool_result`'ın sonunda şu vardı:

```
[Truncated: PARTIAL view — /home/altan/Desktop/adapted/lists/agents.md:
showing lines 1-1555 of 2290 total (31289 tokens, cap 25000).
Call Read with offset=1556 limit=1555 for the next page, or Grep to find a
specific section. Do NOT answer from this page alone if the answer may be
further in the file.]
```

Üç tasarım kararı bir arada:

| Karar | Neden |
|---|---|
| Sert tavan (~25.000 token) | Tek bir tool sonucu bağlamı domine edemesin |
| **Kırpma modele bildirilir** | Sessiz kırpma, eksik veriyle tam güvenle cevap üretir — en tehlikeli hata sınıfı |
| Devam yolu tarif edilir (`offset`) | Model sayfalamayı bilir, çaresiz kalmaz |

> **Bulgu 11.** Kritik olan kırpmanın kendisi değil, **kırpmanın görünürlüğüdür.** Modele "bu veri eksik ve devamı şöyle alınır" denmezse, ajan kısmi bilgiyle tam bir cevap üretir ve bu hata çıktıda hiçbir iz bırakmaz.

### Uygulama

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
            f"[Kırpıldı: toplam {n} token, tavan {MAX_TOOL_RESULT_TOKENS}. "
            f"Devamı için offset kullan. Bu sayfa tek başına yeterli olmayabilir.]"
        ),
    }
```

---

## 8.3 Hat 2 — Context editing (budama)

**Sunucu tarafında, eski tool sonuçlarını ve thinking bloklarını siler.**

```python
client.beta.messages.create(
    model="claude-opus-5", max_tokens=16000,
    betas=["context-management-2025-06-27"],
    context_management={"edits": [
        {"type": "clear_tool_uses_20250919", "clear_tool_inputs": True},
        {"type": "clear_thinking_20251015"},
    ]},
    tools=tools, messages=messages,
)
```

Gerekçesi Anthropic'in ifadesiyle:

> *"once a tool has been called deep in the message history, why would the agent need to see the raw result again?"*

Ve niteliği: *"one of the **safest, lightest touch** forms of compaction."*

Budama sıkıştırmadan daha güvenlidir çünkü **özet üretmez** — dolayısıyla özet hatası yapmaz. Kaybedilen bilgi bellidir ve seçimi deterministiktir.

---

## 8.4 Hat 3 — Compaction (özetleme)

**Pencere sınırına yaklaşan konuşmayı özetleyip yeni pencere başlatmak.**

```python
response = client.beta.messages.create(
    model="claude-opus-5", max_tokens=16000,
    betas=["compact-2026-01-12"],
    context_management={"edits": [{"type": "compact_20260112"}]},
    messages=messages,
)

# ⚠️ TAM content'i geri ekle — sadece metni değil.
messages.append({"role": "assistant", "content": response.content})
```

> **Sık yapılan hata:** `next(b.text for b in response.content)` ile metni çekip onu geçmişe eklemek. `compaction` bloğu düşer ve sıkıştırma durumu **sessizce** kaybolur.

### Claude Code'un uygulaması

| Korunan | Atılan |
|---|---|
| Mimari kararlar | Fazlalık tool çıktıları |
| Çözülmemiş buglar | Fazlalık mesajlar |
| Implementasyon detayları | |

Devam ederken: **sıkıştırılmış bağlam + en son erişilen 5 dosya.**

### "Yük taşıyan" durumun tanımı

Glean, üretimde sorgularının **~%5'inin** bağlam sınırına çarptığını bildiriyor ve korunması gereken durumu şöyle tanımlıyor:

```
✓ kullanıcı niyeti
✓ şimdiye kadar verilen kararlar
✓ BAŞARISIZ OLAN yaklaşımlar          ← sıkça atlanan
✓ planlanan sonraki adımlar
✓ son yüksek-sinyalli tool çıktıları
```

Üçüncü madde özellikle önemlidir: **başarısız denemeler korunmazsa ajan aynı çıkmazı tekrar dener.** Bu, §09.4'teki drift göstergelerinden birinin (*"zaten verdiği kararı yeniden ifade ediyor"*) doğrudan panzehiridir — ve compaction prompt'unu ayarlarken recall aşamasında (yukarıdaki reçete) özellikle aranması gereken içeriktir.

Hedef ifadesi: *"compressed but **semantically complete** representation of the task state."* Yani sıkıştırma, hacim hedefi değil **anlam bütünlüğü** hedefiyle ayarlanır.

### İki katman

| Katman | Ne yapar |
|---|---|
| **Conversation compaction** | Ham gidiş-gelişi yoğunlaştırır: ne istendi, ne denendi, ne işe yaradı, ne yaramadı, sırada ne var |
| **Tool-output compaction** | Büyük çıktılar sandbox dosyalarına **yazılır**, bağlamda **özet + dosya yolu** kalır |

İkincisi §8.2'deki kırpmanın bir üst versiyonudur: kırpma bilgiyi **atar**, dosyaya yazma bilgiyi **taşır**. Ajan gerekirse geri okuyabilir — bu, geri dönülemez kaybı geri dönülebilir bir erişim maliyetine çevirir.

### Ayarlama reçetesi

> *"Start by **maximizing recall** to ensure your compaction prompt captures every relevant piece of information from the trace, then iterate to improve **precision** by eliminating superfluous content."*

```
1. Karmaşık ajan izleri üzerinde çalış
2. Önce RECALL   → hiçbir ilgili parça kaçmasın
3. Sonra PRECISION → fazlalığı ele
```

Sıra kritiktir. Tersi yapılırsa kritik bilgi baştan kaybolur ve geri getirilemez.

Ve uyarı:

> *"overly aggressive compaction can result in the loss of subtle but critical context whose importance **only becomes apparent later**."*

---

## 8.5 İki bozulma türü: bloat vs poisoning

ML Mastery'nin en değerli ayrımı:

| | **Context bloat** | **Context poisoning** |
|---|---|---|
| Ne olur | Eski tool çıktıları, çözülmüş hatalar, geçersizleşmiş kararlar bağlamda kalır | Modelin **erken bir hatası korunur ve doğruymuş gibi ele alınır** |
| Etkisi | Token yakar, değer katmaz | Sonraki muhakeme onun üstüne kurulur |
| Karakteri | **Pahalı** | **Yanlış** |
| Zamanla | Sabit maliyet | **Birikimli hata** |

Poisoning çok daha tehlikelidir çünkü kendini besler: model 3. turda yanlış bir varsayım yapar, varsayım geçmişte kalır, 10. turda artık "bilinen gerçek" muamelesi görür.

**Compaction bloat'ı çözer, poisoning'i çözmez** — hatta özet, yanlış varsayımı "karar" olarak kaydederek kalıcılaştırabilir. Poisoning'e karşı savunma ölçümdedir (§09).

---

## 8.6 Geçmiş yönetiminde üç strateji

| Strateji | Nasıl | Artı | Eksi |
|---|---|---|---|
| **Recency truncation** | Son N turu tut | Ucuz, basit | **Uzun vadeli durum kaybolur** |
| **Rolling summarization** | Eski alışverişleri periyodik özetle | Anlamı korur | Özetin özeti → **drift** |
| **Anchored iterative summarization** | Sabit şemalı oturum-durumu belgesi, sürekli güncellenir | Drift'i önler | En çok mühendislik |

Üçüncüsünün şablonu §05.7'de. "Anchored" olmasının sebebi: serbest özet kademeli olarak kayar (5. özet, 4. özetin özetidir); sabit başlıklar her güncellemede aynı soruların yeniden cevaplanmasını zorlar.

---

## 8.7 Hat 4 — Subagent (izolasyon)

**Ayrı bağlam penceresi.** Keşif gürültüsü alt ajanda kalır, ana ajana yalnızca sonuç döner.

| | Token |
|---|---|
| Alt ajanın harcadığı | on binlerce+ |
| Ana ajana döndürdüğü | **1.000–2.000** |

10–50× sıkıştırma.

**Bedeli soğuk başlangıçtır.** Bu oturumdaki `Agent` tool'unun açıklaması maliyeti açıkça yazıyor:

> *"Each spawn **starts cold and re-derives context you already have** — it's the expensive path."*
> *"The agent's final report is not shown to the user — relay what matters."*

Yani izolasyon bedava değildir: alt ajan, kullanıcının ana ajana anlattığı hiçbir şeyi bilmez.

### PTC ile bileşim: programatik subagent

Subagent'ı bir tool çağrısı olarak açmak, sayıyı elle yazılabilir olanla sınırlar. PTC ile birlikte orchestrator **kod yazarak** subagent açabilir:

```python
customers = crm.search(segment="enterprise")        # 300 kayıt
results = spawn_parallel([                          # kod içinde döngü
    subagent(template=RESEARCH_PROMPT, task={"customer": c})
    for c in customers
])
summary = aggregate(results)
print(summary)                                       # bağlama SADECE bu girer
```

Bu bileşim, aynı analizi bağımsız öğelerden oluşan büyük bir koleksiyona uygulamayı mümkün kılar — yüzlerce müşteriyi araştırmak, binlerce destek kaydını analiz etmek. Her subagent kendi bağlam penceresi ve token bütçesiyle çalışır; orchestrator ortak talimatı bir prompt şablonu veya programatik görev tanımı olarak sağlar.

> *"the depth of analysis improves as each one can **focus its attention on the context and the job to be done without distraction**."*
> — Glean Engineering, 2026

Yapı, yüksek seviye ajanların işi ayrıştırıp sınırlı görevleri alt işçilere devrettiği **özyinelemeli dil modeli** desenini yansıtır.

---

## 8.8 İki eksen: uzaysal ve zamansal

Dört hattı katmana göre sıralamak (ingestion → sunucu → harness) uygulama sırasını verir; ancak daha aydınlatıcı bir sınıflandırma **neyi izole ettikleridir**:

> *"Sub-agents reduce how much irrelevant detail the orchestrator ever sees by keeping intermediate work inside isolated contexts. Compaction handles what remains in the orchestrator's own history, preserving task coherence across time. **One manages context spatially, the other temporally.**"*
> — Glean Engineering, 2026

```
UZAYSAL (spatial)                    ZAMANSAL (temporal)
────────────────────                 ────────────────────
Subagent, PTC sandbox'ı              Compaction, context editing

aynı anda, FARKLI pencerelerde       aynı pencerede, FARKLI zamanlarda

"bu detay buraya hiç girmesin"       "bu detay artık gerekmiyor"

→ orchestrator'ın gördüğünü azaltır  → geçmişte kalanı yönetir
```

İkisi birbirini **takviye eder**, ikame etmez:

- Yalnızca uzaysal izolasyon: orchestrator az şey görür ama uzun oturumda kendi geçmişi yine şişer
- Yalnızca zamansal sıkıştırma: her ara sonuç önce bağlama girer, sonra özetlenir — girmemesi mümkünken

Üçüncü bir eksen olarak **ingestion kırpma** (§8.2) ve **JIT getirme** (§06) sayılabilir: bunlar verinin *hiç üretilmemesini* değil, üretildiği hâlde *hiç girmemesini* sağlar. Bu açıdan uzaysal ailenin en ucuz üyesidirler.

---

## 8.9 Hangisi ne zaman

| Mekanizma | Katman | Ne yapar | En uygun |
|---|---|---|---|
| Çıktı kırpma | Harness, ingestion | Sert tavan + görünür uyarı | Her zaman — ilk savunma |
| Context editing | Sunucu, istek anı | Bayat tool sonuçlarını **siler** | Uzun tool-yoğun döngüler |
| Compaction | Sunucu, pencere dolarken | Geçmişi **özetler** | Yoğun gidiş-geliş gerektiren görevler |
| Note-taking / memory | Harness, dosya sistemi | Bağlam dışına yazar | Net kilometre taşları olan iteratif iş |
| Subagent | Harness, ayrı pencere | Gürültüyü izole eder | Paralel keşfin karşılık verdiği araştırma |

Anthropic'in kendi rehberi:

> *Compaction, yoğun gidiş-geliş gerektiren görevlerde konuşma akışını korur; note-taking, net kilometre taşları olan iteratif geliştirmede öne çıkar; multi-agent mimarileri, paralel keşfin karşılık verdiği karmaşık araştırma ve analizde uygundur.*

---

## 8.10 Cache ekonomisi

Yerleşim kararlarının arkasındaki tek kural: **caching bir prefix eşleşmesidir; N. bayttaki değişiklik ≥N konumundaki tüm cache'i geçersiz kılar.**

### Geçersiz kılma hiyerarşisi

| Değişiklik | tools | system | messages |
|---|:---:|:---:|:---:|
| Tool tanımı ekle/çıkar/sırala | ✗ | ✗ | ✗ |
| Model değiştir | ✗ | ✗ | ✗ |
| System prompt içeriği | ✓ | ✗ | ✗ |
| `tool_choice`, thinking aç/kapa | ✓ | ✓ | ✗ |
| Mesaj ekle | ✓ | ✓ | ✗ |

`tool_choice` değiştirmek tools+system cache'ini korur; **tool listesi değiştirmek her şeyi öldürür.**

### Sessiz bozucular

```python
# ✗ system prompt'ta zaman damgası
system = f"Bugün {datetime.now()}. Sen bir asistansın…"

# ✗ kullanıcıya göre tool listesi
tools = build_tools_for(user)

# ✗ sıralamasız serialize
tools = [schema(t) for t in tool_set]        # set → sıra garantisi yok

# ✓
system = "Sen bir asistansın…"                # donmuş
tools  = sorted(ALL_TOOLS, key=lambda t: t["name"])
messages.append({"role": "user",
                 "content": f"<context>Tarih: {today}</context>\n{user_msg}"})
```

### Gözlemlenen uygulama

Rapor oturumunun system prompt'undaki git durumu bloğunda açıkça şu yazıyor:

> *"This is the git status at the start of the conversation. Note that this status is a **snapshot in time, and will not update** during the conversation."*

Canlı güncellenseydi system prompt her turda değişir ve **tüm cache her turda ölürdü.** Bunun yerine anlık görüntü dondurulmuş ve modele bayat olduğu söylenmiştir.

> **Bulgu 12.** Cache ekonomisi prompt tasarımını doğrudan şekillendirir. "Değişken veriyi dondur ve modele bayat olduğunu söyle" kararı, doğruluk ile maliyet arasında bilinçli bir takastır — ve raporun genel tezinin somut bir örneğidir: mimari kararlar model kararlarını belirler.

### Cache'i bozmadan değiştirme

İki mekanizma, iki farklı sorun için:

| Değiştirilmek istenen | Cache-korur yöntem | Gereksinim |
|---|---|---|
| Tool seti | `{"role":"system"}` içinde `tool_addition` / `tool_removal` blokları | Opus 5+, beta `mid-conversation-tool-changes-2026-07-01`; hedef tool `defer_loading: True` ile önceden bildirilmiş olmalı |
| System talimatı | `messages` içine `{"role":"system", "content": "..."}` | Opus 5, Opus 4.8, Fable 5, Mythos 5 (Sonnet 5 **değil**); beta header gerekmez |

İkincisi ayrıca **prompt injection açısından da üstündür**: `<system-reminder>` metni user içeriğine yazabilen herkes tarafından taklit edilebilir; `role: "system"` edilemez.

---

## 8.11 Token bütçesi

### Birim: tek çağrı değil, tüm koşu

> *"tokens accumulate across turns, so budgeting must treat **the full run** as the cost unit."*

### Hedef doluluk

> *"Aim for roughly **60–80% context utilization** rather than maxing out capacity."*

⚠️ Bu bir **sezgisel kuraldır**, türetilmiş bir yasa değil. Gerekçesi örtüktür: dolulukta hem rot hem position bias devreye girer, ayrıca ani büyümeler için pay gerekir. Kendi ölçümünle doğrulanmalıdır (§09).

### Dinamik tahsis

Basit görevler minimal bağlam, karmaşık çok adımlı görevler daha fazla. Bugünkü API karşılıkları:

```python
output_config={
    "effort": "low",                                   # düşük: az tool çağrısı, kısa muhakeme
    "task_budget": {"type": "tokens", "total": 64000}, # ajan döngüsü için token tavanı
}
```

`task_budget` ile `max_tokens` farkı önemlidir:

| | Ne | Model bilir mi |
|---|---|---|
| `max_tokens` | Yanıt başına **sert** tavan | ❌ Hayır — habersiz kesilir |
| `task_budget` | Döngü için **bütçe önerisi** | ✅ Evet — geri sayımı görür, işini ona göre paylaştırır |

### Muhasebe uyarısı

```python
u = response.usage
total_prompt = (u.input_tokens
                + (u.cache_creation_input_tokens or 0)
                + (u.cache_read_input_tokens or 0))
```

`usage.input_tokens` **yalnızca cache'lenmemiş kısmı** gösterir. 100K token'lık gerçek bir prompt'ta bu alan 3.571 gösterebilir. Maliyet raporlanırken üçü toplanmalıdır.

---

## 8.12 Özet

Dört hat, aynı problemin farklı katmanlarındaki çözümleridir:

```
                    veri  ─────────────────────────────────►  bağlam
                            │           │          │
       ┌────────────────────┘           │          └──────────────────┐
       ▼                                ▼                             ▼
  HAT 1: hiç girmesin            HAT 2/3: girdiyse             HAT 4: başka
  (kırpma, grep hunisi,          budansın/özetlensin           pencerede kalsın
   artefakt işleme)              (editing, compaction)         (subagent)

                    ve girdiyse UCUZA girsin → prompt caching
```
