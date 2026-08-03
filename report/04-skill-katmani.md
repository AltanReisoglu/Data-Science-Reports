# 4. Skill Katmanı: Koşullu Prompt Enjeksiyonu

> **Bölümün tezi:** Skill, modele öğretilen bir bilgi değil, **koşullu olarak bağlama enjekte edilen bir prompt'tur.** Değeri içeriğinden değil, *ne zaman yüklenmediğinden* gelir.

---

## 4.1 Çözdüğü problem

Bir ajanın işe yaraması için alan bilgisi gerekir: kurum içi kodlama standartları, bir dosya formatının incelikleri, bir aracın doğru kullanımı, bir sürecin adımları.

Naif çözüm bunu system prompt'a yazmaktır. Üç şekilde çöker:

```
1. Alan sayısı arttıkça system prompt şişer      → her turda ödenen sabit maliyet
2. Alanların çoğu çoğu görevde alakasızdır       → gürültü, aşırı tetikleme
3. System prompt 0. pozisyondadır                → her düzenleme tüm cache'i öldürür
```

Skill bu üç sorunu tek bir fikirle çözer: **bilgiyi katmanlara böl, sadece gereken katmanı yükle.**

---

## 4.2 Üç katmanlı progressive disclosure

| Katman | İçerik | Bağlamda ne zaman | Maliyet |
|---|---|---|---|
| **1** | `name` + tek satırlık `description` | **Her turda** | ~20–30 token × skill sayısı |
| **2** | `SKILL.md` gövdesi | Skill çağrıldığında | Binlerce token |
| **3** | Referans dosyalar, script'ler (`references/*.md`, `scripts/*.py`) | Model gövdedeki yönlendirmeyi okuyup ayrıca isterse | Değişken |

Ekonomisi basit bir hesap:

```
8 skill × 25 token   =    200 token   → SÜREKLİ maliyet
1 skill gövdesi      = 50.000 token   → SADECE gerektiğinde
─────────────────────────────────────
Hepsi baştan yüklü olsaydı: ~400.000 token ile başlayan bir oturum
```

**200 token'lık sabit maliyet, 50K'lık yükü koşullu hâle getirmenin bedelidir.**

### Katman 0: skill indeksi

Üç katmanlı yapı bir sınır barındırır: **katman 1 de bağlam yer.** Skill sayısı arttıkça sürekli maliyet doğrusal büyür.

```
  8 skill × 25 token =    200 token   ← yönetilebilir
500 skill × 25 token = 12.500 token   ← her turda, çoğu alakasız
```

Glean bu sınırı kurumsal ölçekte doğrudan raporluyor:

> *"progressive disclosure alone **does not go far enough at enterprise scale.** It is not feasible for an agent to browse hundreds of tools or skills as **even one-line descriptions per item add up** and create noise the agent has to reason through on requests where most of those capabilities are irrelevant."*

Çözüm, katmanın **altına** bir katman eklemektir: skill'lerin üzerinde aranabilir bir indeks.

| Katman | Bağlamda ne var | Ne zaman | Kaç öğe |
|---|---|---|---|
| **0 — indeks** | **Hiçbir şey** — aranabilir dış yapı | Hep | 500 |
| **1 — kısa liste** | Ad + açıklama, **yalnızca aranmış olanlar** | Arama sonrası | 5–10 |
| **2 — gövde** | `SKILL.md` | Çalıştırma anı | 1 |
| **3 — referanslar** | Yardımcı dosyalar, script'ler | Gerekirse | 0–n |

Glean'in üç fazlı keşif akışı:

```
1. İndeksi ara       → model bir yetenek gerektiğini fark eder, sorgu atar
2. Kısa liste        → ad, açıklama, yürütme ipucu döner
                        ↑ model ilk kez HERHANGİ bir detay görür
3. Şemayı hidrat et  → yalnızca çalıştırmaya karar verilince tam şema girer
```

> *"the agent **never pays the context tax of browsing the full capability surface.**"*

**Sıralama sinyali salt semantik değildir.** Kurumsal ortamda aynı işi yapan onlarca benzer skill bulunabilir; Glean bunları **kurumsal graf sinyalleriyle** sıralar:

> *"we leverage enterprise graph signals to rank based on **creator, usage**, and more. The aim is to find the right skill even in scenarios where **multiple individuals create and share similar skills**."*

Yani "hangi skill" sorusunun cevabı metin benzerliğinde değil, **sosyal ve kullanım sinyalindedir**: kim yazmış, kaç kez kullanılmış, hangi ekipte yaygın.

Bu, §03.7.2'deki `defer_loading` + tool search deseninin skill'lere uygulanmış ve zenginleştirilmiş hâlidir. Aynı ilke, dördüncü kez: **haritayı ucuza al, bölgeyi pahalıya al** — ancak burada harita bile bağlamda tutulmaz, sorgulanır.

---

## 4.3 Doğrudan gözlem: bu oturumda ne oldu

Bu raporun yazıldığı Claude Code oturumunda mekanizmanın üç katmanı da gözlemlendi.

### Katman 1 — sürekli bağlamda

Oturum boyunca bağlamda bir `<system-reminder>` bloğu içinde şu liste vardı:

```
The following skills are available for use with the Skill tool:

- dataviz: Use this skill whenever you are about to create ANY chart, graph, plot...
- artifact-design: Design guidance and fundamentals for Artifacts.
- claude-api: Reference for the Claude API / Anthropic SDK — model ids, pricing,
  params, streaming, tool use, MCP, agents, caching, token counting, model migration.
  TRIGGER — read BEFORE opening the target file... whenever: the prompt names
  Claude/Anthropic in any form...; the user asks about an LLM...; OR the task is
  LLM-shaped with provider unstated (agent/MCP/tool-definition/...).
  SKIP only when another provider is being worked on (overrides all triggers):
  OpenAI/GPT/Gemini/... named in the query; OR `grep -rE 'openai|langchain_openai|
  google.generativeai|genai|mistralai|cohere|ollama'` over the project hits
  (run this grep FIRST if no provider named — don't Read the file).
- update-config: ...
- simplify: ...
```

Gövdeler yoktu. Sadece **ne olduğu** ve **ne zaman gerekeceği**.

### Tetikleme — kural tabanlı, keyfî değil

Kullanıcı *"bu LLM'e tool'ları nasıl tanıtabiliriz"* diye sordu. Bu, `claude-api` skill'inin TRIGGER koşuluna uyuyordu (*"tool-definition"*, sağlayıcı belirtilmemiş). Ancak SKIP kuralı önce bir kontrol istiyordu.

Model önce grep'i çalıştırdı:

```bash
grep -rEil 'openai|langchain_openai|google.generativeai|genai|mistralai|cohere|ollama' . \
  --exclude-dir=.git | head -20
```

Çıktı üç dosyaydı — hepsi araştırma notu listeleri, üzerinde çalışılan bir sağlayıcı projesi değil. SKIP koşulu sağlanmadı, skill yüklendi.

> **Gözlem:** Yükleme kararı modelin sezgisine bırakılmamış, **skill'in kendi açıklamasına gömülü bir karar prosedürüne** bağlanmıştı. Skill açıklaması yalnızca "ne olduğunu" değil, "nasıl karar verileceğini" de taşıyor.

### Katman 2 — enjeksiyon

`Skill(claude-api)` çağrısı bir `tool_use` bloğuydu. Dönen `tool_result` şu yapıdaydı:

```
Base directory for this skill: /tmp/.../claude-api

# Building LLM-Powered Applications with Claude
## Before You Start
...
<doc path="python/claude-api/tool-use.md">
# Tool Use — Python
...
</doc>
<doc path="shared/tool-use-concepts.md">
...
</doc>
<doc path="shared/prompt-caching.md">
...
</doc>
```

SKILL.md gövdesi + paketlenmiş referans dokümanları, tek bir `tool_result` içinde. Bağlam bir anda ~50.000 token büyüdü ve sonraki her turda bu yük taşındı.

### Katman 3 — yüklenmedi

Dokümanın içinde *"bu bilgiyi WebFetch ile şu adresten al"* diyen satırlar vardı. İhtiyaç doğmadığı için çekilmedi.

---

## 4.4 Mekanizmanın özeti: özel bir kanal yok

Bu bölümün en önemli teknik tespiti:

> **Skill, `tool_result` olarak bağlama giren düz metinden ibarettir.**

Özel bir rol, gizli bir enjeksiyon kanalı, ayrı bir API alanı yoktur. Model açısından bir skill ile büyük bir dosya okuması arasında **hiçbir yapısal fark yoktur** — ikisi de `tool_result` bloğudur.

Skill'i skill yapan şey içeriği değil, **etrafındaki koşullu yükleme protokolüdür**: sürekli bağlamda duran ucuz bir tanıtım satırı + tetikleme kuralı + talep üzerine yüklenen ağır gövde.

Bu, raporun genel tezinin bir örneğidir: mekanizma modelde değil, harness'tedir.

---

## 4.5 İki tür progressive disclosure

Terim iki farklı şeyi tanımlamak için kullanılıyor. Ayrım önemli:

| | **Tasarlanmış** | **Keşfedilen** | **Aranan** |
|---|---|---|---|
| Kim kurar | Skill yazarı, önceden | Ajan, çalışma anında | İndeks, sorgu üzerine |
| Yapı | Sabit üç katman (ad → gövde → referans) | Ortamın kendi yapısı (dizin → dosya → bölüm) | Sorgu → kısa liste → şema |
| Sinyal | `description` metni | Dosya adı, boyut, konum, zaman damgası | Semantik + kullanım/yaratıcı sinyalleri |
| Örnek | `claude-api` skill'i | `glob` → `grep -l` → `grep -C` → `Read offset` (§06) | Glean skill indeksi, tool search |
| Kontrol | Tasarımcıda | Modelde | Paylaşık — model sorar, indeks sıralar |
| Ölçek | ~10 skill | Ortam kadar | **Yüzlerce–binlerce** |

Üçü de aynı ilkenin uygulamasıdır: **haritayı ucuza al, bölgeyi pahalıya al.** Fark, haritanın kim tarafından çizildiği ve haritanın kendisinin bağlamda tutulup tutulmadığıdır. Üçüncüsünde harita bile bağlam dışıdır — yalnızca sorgunun cevabı içeri girer.

---

## 4.6 API tarafındaki varyant: Agent Skills

Claude Code'daki skill'ler dosya sistemi tabanlıdır. Messages API'de aynı fikrin farklı bir uygulaması vardır:

```python
response = client.beta.messages.create(
    model="claude-opus-5", max_tokens=16000,
    betas=["code-execution-2025-08-25", "skills-2025-10-02"],
    container={"skills": [{"type": "anthropic", "skill_id": "pptx", "version": "latest"}]},
    tools=[{"type": "code_execution_20260521", "name": "code_execution"}],
    messages=[{"role": "user", "content": "3 slaytlık bir sunum hazırla"}],
)
```

Farklar:

| | Claude Code skill | API Agent Skill |
|---|---|---|
| Nerede yaşar | Yerel dosya sistemi (`SKILL.md` klasörü) | Anthropic'in sunucusu |
| Nasıl açılır | `Skill` tool'u | `container.skills` parametresi |
| Çalışma ortamı | Senin makinen | Sandbox konteyneri |
| Ne yapar | Talimat enjekte eder | Talimat enjekte eder **+ kütüphaneleri hazır bulundurur** |
| Örnek | `claude-api`, `dataviz` | `pptx`, `xlsx`, `docx`, `pdf` |

İkincisinde kritik nokta: `pptx` skill'i modele "sunum üretmeyi öğretmez", modele **`python-pptx` kullanarak sunum üreten kodu nasıl yazacağını** anlatır ve o kütüphaneyi sandbox'ta hazır bulundurur (§07).

Özel skill'ler Skills API ile yönetilir (`POST /v1/skills`, sürümleme dahil); ajan başına en fazla 20 skill.

---

## 4.7 Tasarım ilkeleri

Gözlemlerden çıkan pratik kurallar:

**1. Açıklama, tetikleyici koşulu içermeli.**

```
✗ "Excel dosyalarıyla çalışmak için."
✓ "Excel dosyası okuma, düzenleme veya oluşturma gerektiğinde çağır.
   .xlsx/.xls girdisi verildiğinde veya çıktı istendiğinde tetiklenir."
```

**2. Negatif koşul da yazılmalı.** `claude-api` skill'inin SKIP kuralı, yanlış tetiklenmeyi grep ile önlüyordu. Ne zaman *yüklenmemesi* gerektiğini söylemek, ne zaman yükleneceğini söylemek kadar değerlidir.

**3. Gövde ile referans ayrılmalı.** Her göreve gereken temel akış gövdede; kenar durumlar, uzun kod parçaları, format tabloları referans dosyalarda. Aksi hâlde katman 3 anlamsızlaşır ve skill tek seferde tüm ağırlığını yükler.

**4. Gövde uzunluğu bir bütçe kararıdır.** Skill yüklendiği anda sonraki her turda taşınır. 50K token'lık bir skill, 10 turluk bir oturumda etkin olarak 10 kez ödenir (cache sayesinde ucuza, ama dikkat bütçesinden tam bedelle).

**5. Skill'ler bağlama tool gibi girmez.** Tool şemaları 0. pozisyondadır ve cache prefix'ini oluşturur; skill gövdesi `messages` içine, sona eklenir. Bu yüzden bir skill yüklemek mevcut cache'i **bozmaz** — yalnızca üstüne ekler. Tasarım açısından önemli bir avantajdır.

---

## 4.8 Sınırlar

| Sınır | Sonuç | Çözümü var mı |
|---|---|---|
| Skill yüklendikten sonra **boşaltılamaz** | Yanlış tetiklenme, oturumun geri kalanına maliyet olarak yansır | ❌ Yok — bu yüzden SKIP kuralları kritik |
| Katman 1 açıklamaları da bağlam yer | 500 skill × 25 token = 12.500 token sürekli | ✅ **Katman 0 — indeks** (§4.2) |
| Tetikleme modelin kararıdır | Kural yazılabilir, garanti edilemez; `tool_choice` benzeri zorlama yoktur | ⚠️ Kısmen — indeks araması kararı daraltır |
| Gövde statiktir | Çalışma anındaki duruma göre uyarlanmaz | ⚠️ Kısmen — PTC ile script'e dönüştürülebilir (§07) |

Birinci sınır en ciddi olanıdır ve raporun genel çerçevesinde tanıdıktır: **bağlama giren geri çıkmaz.** Bu, §08'deki tüm basınç mekanizmalarının varlık sebebiyle aynıdır — ve skill için henüz bir `clear_skill` mekanizması yoktur.

---

## 4.9 Bulgu

> **Bulgu 6.** Skill mekanizması özel bir model yeteneği değil, bir **yükleme protokolüdür**. Bağlama giren şey sıradan bir `tool_result` metnidir; değeri, sürekli bağlamda duran ucuz bir tanıtım satırı ile talep üzerine yüklenen ağır gövde arasındaki maliyet farkından gelir. Bu oturumda ölçülen oran: **~200 token sürekli maliyet karşılığında ~50.000 token'lık koşullu yük.**
