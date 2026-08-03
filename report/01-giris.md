# 1. Giriş: Bağlam Neden Mühendislik Konusu

## 1.1 Problem

Üretimde çalışan bir yapay zekâ ajanı bozulduğunda ilk refleks modeli suçlamaktır. Ancak alan pratiğinde tekrarlanan tespit şudur:

> *"When AI agents break down in production, **the problem is rarely the model.** More often, the context window is mismanaged: bloated with stale history, redundant retrieval results, and raw tool outputs that bury the signal the model actually needs."*
> — ML Mastery, Nis 2026

Bu rapor, o "yönetim" işinin ne olduğunu baştan sona inceliyor.

### Ölçülmüş kanıt

İddia yalnızca kavramsal değil. Sektörden üç veri noktası:

| Kim | Ne yaptı | Sonuç |
|---|---|---|
| **LangChain** | Yalnızca harness değişikliği — **model sabit tutuldu** | Terminal-Bench'te **+13,7 puan** |
| **Vercel** | Ajanın tool'larının **%80'ini kaldırdı** | Daha yüksek güvenilirlik, **3,5× daha düşük gecikme** |
| **Glean** | Talimatları kademeli yüklenen skill'lere taşıdı | System prompt **%45+ küçüldü** |

> *"In these cases, the performance delta came from **the harness, not the model**."*
> — Glean Engineering, 2026

Vercel'in bulgusu özellikle dikkat çekicidir: **yetenek eksiltmek performansı artırmıştır.** Bu, bağlamın sonlu bir kaynak olduğu tezinin en doğrudan ampirik sonucudur (§03.7, §01.4).

> ⚠️ **Kaynak notu:** LangChain ve Vercel rakamları Glean'in yazısından ikincil aktarımla alınmıştır; metodolojileri görülmeden nihai rapora konulmadan önce birincil kaynaklardan doğrulanmalıdır.

### Nedensellik: harness'ler neden değişiyor

Yaygın anlatı, harness'lerin modeller güçlendiği için değiştiğini söyler. Glean bu açıklamanın eksik olduğunu savunur:

> *"That is **not the only driver.** Harnesses are being rebuilt because **we are giving agents more work, which requires more context.** Every additional tool call, skill invocation, search result, and execution output adds to the context window."*

İki kuvvet aynı anda ve **ters yönde** çalışıyor:

```
Model güçleniyor      →  harness BASİTLEŞEBİLİR
                         (daha az kuralcı prompt, daha çok otonomi)

İşin hacmi büyüyor    →  harness KARMAŞIKLAŞMAK ZORUNDA
                         (her tool çağrısı, skill, arama sonucu bağlama ekleniyor)
```

Bu rapor ikinci kuvveti asıl itici güç olarak ele alıyor: ajanlara verilen işin karmaşıklığı, modellerin iyileşme hızından daha hızlı büyüdüğü sürece bağlam yönetimi merkezi mühendislik problemi olarak kalacaktır.

---

## 1.2 Temel ayrım: model ≠ harness

Raporun her bölümü bu ayrımın üstüne oturuyor. Karıştırılırsa geri kalan her şey karışır.

| | **Model** | **Harness** |
|---|---|---|
| Ne yapar | Token dizisi alır, token dizisi üretir. Sadece bu. | Bağlamı kurar, tool'ları çalıştırır, döngüyü çevirir, izin sorar, durum saklar |
| Hafızası | **Yok.** Durumsuz. | Var — dosya sistemi, transcript, oturum durumu |
| Bağlamı | Kendisine verilen prefix | Bu prefix'i **her turda yeniden inşa eden** taraf |
| Örnek | `claude-opus-5` | Claude Code, LangGraph, CrewAI, senin `while` döngün |

Model her turda sıfırdan doğar. "Hatırlıyor" görünmesinin tek sebebi harness'in geçmişi her istekte yeniden göndermesidir (§02.5).

**Sonuç:** context engineering tamamen harness tarafında yapılan bir iştir. Modelin içinde olan bir şey değildir.

Bu, raporun tezinin ilk yarısıdır: *bir ajanın zekâsının ne kadarı modelden, ne kadarı ona ne gösterildiğinden gelir?*

### Harness'in tanımı

Rapor boyunca kullanılan en keskin tanım Glean'e aittir:

> **"The harness is, at its core, a **distributed context management system**."**

Yani harness'in diğer işlevleri — tool çalıştırma, izin kapıları, durum saklama, döngü yönetimi — bağımsız sorumluluklar değil, **tek bir sorumluluğun türevleridir**: sonlu bir dikkat bütçesine neyin, ne zaman, hangi biçimde gireceğine karar vermek.

"Dağıtık" olmasının sebebi, bu kararın tek bir yerde verilmemesidir. §04–§08'de görüleceği gibi karar noktaları harness'in her katmanına yayılmıştır: tool şeması yüklenirken, skill aranırken, tool çıktısı kırpılırken, geçmiş sıkıştırılırken, subagent açılırken.

---

## 1.3 Prompt engineering → context engineering

| | Prompt engineering | Context engineering |
|---|---|---|
| Tanım | Talimatları yazma ve düzenleme yöntemleri | Çıkarım sırasında optimal token kümesini **derleme ve sürdürme** stratejileri |
| Karakter | **Ayrık** — bir kez yazılır | **Döngüsel** — her turda "ne geçireceğiz" kararı yeniden verilir |
| Kapsam | Ağırlıklı olarak system prompt | System + tools + MCP + dış veri + mesaj geçmişi + tool sonuçları |
| Kim yazar | İnsan | Kısmen insan, büyük ölçüde **çalışma anında oluşur** |

Kritik fark sonuncusu. Prompt'u yazan kişi tool sonuçlarını, getirilen belgeleri veya geçmişin kendisini önceden yazmaz — bunlar ajan çalıştıkça birikir. Yönetilmesi gereken şey metin değil, **durum**.

Anthropic'in ifadesiyle:

> *"An agent running in a loop generates more and more data that could be relevant for the next turn of inference, and this information must be **cyclically refined**."*

Dört fiil olarak operasyonelleştirilebilir:

```
GİR      → ne bağlama alınacak
SIKIŞTIR → ne özetlenecek
GETİR    → ne talep üzerine çekilecek
AT       → ne tamamen dışarıda bırakılacak
```

---

## 1.4 Bağlam neden sonlu bir kaynak

### İki maliyet

| | Ne | Nasıl görünür |
|---|---|---|
| **Finansal** | Milyon input token başına ücret; çok adımlı döngüde katlanır | Faturada, doğrudan |
| **Bilişsel** | Model tüm token'lara eşit davranmaz | Kalite düşüşü olarak, dolaylı |

Finansal maliyet ölçülebilir ve tahmin edilebilir. Asıl mesele ikincisidir.

### İki ayrı bozulma olgusu

Literatürde sık karıştırılan iki bağımsız olgu var. Rapor bunları ayrı tutuyor:

| | **Context rot** | **Lost in the middle** |
|---|---|---|
| Kaynak | Anthropic (Eyl 2025) | Liu et al. (2023) |
| İddia | Bağlam **uzadıkça** hatırlama doğruluğu düşer | Bağlamın **ortası**, uçlardan daha az etkili |
| Eksen | **Uzunluk** | **Konum** |
| Sonucu | *Ne kadar* bilginin işlendiğini etkiler | *Hangi* bilginin işlendiğini etkiler |

İkisi de gerçektir ve birbirini dışlamaz. 800K token'lık bir bağlamda hem genel hassasiyet düşer (rot) hem de ortadaki bilgi uçlardakinden daha az ağırlık alır (position bias).

### Context rot'un mimari gerekçesi

Anthropic üç mekanizma öneriyor:

**a) n² ilişki patlaması.** Transformer'da her token diğer her token'a dikkat edebilir; n token için n² ikili ilişki. Bağlam uzadıkça bu ilişkileri yakalama kapasitesi incelir — *"stretched thin"*. Bağlam boyutu ile dikkat odağı arasında yapısal bir gerilim doğar.

**b) Eğitim dağılımı yanlılığı.** Modeller dikkat kalıplarını eğitim verisinden öğrenir ve o veride kısa diziler uzun dizilerden çok daha yaygındır. Sonuç: bağlam-geneli bağımlılıklar için hem daha az deneyim hem daha az özelleşmiş parametre.

**c) Pozisyon kodlama interpolasyonu.** Uzun dizileri işleyebilmek için orijinal olarak eğitilen daha küçük bağlama uyarlama teknikleri kullanılır; bu, token pozisyonu anlayışında bir miktar bozulmaya mal olur.

**Sonuç bir uçurum değil, bir eğim:**

> *"These factors create a **performance gradient rather than a hard cliff**: models remain highly capable at longer contexts but may show reduced precision for information retrieval and long-range reasoning."*

Bu nüans korunmalıdır. İddia "bağlam dolunca model bozulur" değil, "bağlam doldukça hassasiyet kademeli düşer"dir.

> ⚠️ **Kaynak notu:** Bu mekanizma açıklaması bir satıcı yayınından gelmektedir ve hakemli bir çalışma değildir. Rapora dahil edilmeden önce NIAH literatürüyle çapraz doğrulanması önerilir.

### İki metafor

**Dikkat bütçesi (Anthropic).** İnsanın çalışma belleği gibi modelin de sonlu bir dikkat bütçesi vardır. Her yeni token bu bütçeden tüketir. **Pencere kapasitesi ≠ kullanılabilir dikkat** — "1M token bağlam" demek "1M token'ı eşit kalitede işler" demek değildir.

**RAM / disk (ML Mastery).** Daha operasyonel:

```
Bağlam penceresi   =  RAM    → hızlı, güçlü, SONLU, oturumlar arası SİLİNİR
Dosya sistemi      =  Disk   → ucuz, büyük, ama AÇIK BİR GETİRME gerektirir
Veritabanı, hafıza =  Disk
```

> *"Good context engineering decides **at each step** what belongs in RAM right now and what lives on disk until needed."*

Raporun ilerleyen bölümlerindeki her mekanizma (skill, memory, JIT retrieval, artefakt işleme, subagent) bu ayrımın bir uygulamasıdır: **veriyi diskte tut, RAM'e sadece gerekeni al.**

---

## 1.5 Bağlamı dolduran dört katman

Üretimdeki bir bağlam penceresi tipik olarak şunları içerir:

| Katman | İçerik | Karakter | Ana risk |
|---|---|---|---|
| **System instructions** | Rol, davranış kuralları, tool açıklamaları, çıktı formatı, few-shot örnekler | Büyük ölçüde **statik** → prefix cache adayı | Şişkinlik |
| **Conversation history** | Kullanıcı turları, ajan yanıtları, tool çağrıları, tool sonuçları | **En hızlı büyüyen** | En az yönetilen katman |
| **Retrieved knowledge** | Dış kaynaklardan çekilen belge, kayıt, hafıza öğesi | Talep üzerine | İlgili-ama-fazlalık içerik |
| **Working state** | Ara sonuçlar, scratchpad muhakemesi, görev ilerlemesi | Çok adımlı tutarlılık için gerekli | Ayrıntılı iz olarak saklanırsa pahalı |

**Amaç her katmanı küçültmek değil, katmanlar arası bütçe dağılımını bilinçli yapmaktır.**

### Çöken birinci nesil desen

Sektörde yaygınlaşan ve uzun ufuklu işte çöken kalıp şudur:

> *"The common pattern of a **single system prompt (often 20K+ tokens of conditional instructions)** with **20 to 40 tool schemas injected upfront** does not hold up."*
> — Glean Engineering, 2026

Sayısal olarak: 20K talimat + 30 şema ≈ **her turda ~25–30K token sabit maliyet**, işin çoğunda büyük kısmı alakasız. Bu, yukarıdaki iki başarısızlık modunun ilkidir (ilgisiz içeriği dahil etmek) ve prefix'te yer aldığı için en pahalı biçimidir (§08.10).

Raporun §03.7 (tool erteleme), §04 (skill katmanları) ve §08 (bağlam basıncı) bölümleri, bu deseni parçalayan mekanizmaları inceler.

İki başarısızlık modu vardır ve ikisi de mimari kararlardır:

```
mevcut adımla ilgisiz içeriği DAHİL ETMEK      → gürültü, maliyet, rot
önemli olan içeriği DIŞARIDA BIRAKMAK          → yanlış cevap, tekrar deneme
```

> *"Both are **architecture decisions, not model decisions.**"*

Bu cümle raporun tezinin ikinci yarısıdır.

---

## 1.6 Statik / dinamik ayrımı

En yüksek değerli tek yapısal karar, isteklerin arasında sabit kalan içerik ile her turda değişen içeriğin ayrılmasıdır.

```
┌─ STATİK (önde, cache'lenir) ───────────────────────┐
│  tool şemaları, system instructions,               │
│  ajan kimliği, sabit kurallar                      │
├─ DİNAMİK (arkada, minimal tutulur) ────────────────┤
│  güncel kullanıcı girdisi, son tool çıktıları,     │
│  getirilen belgeler, oturum durumu                 │
└─────────────────────────────────────────────────────┘
```

İki geçişli montaj:

```python
ctx  = system_prompt + tool_schemas + long_lived_summaries   # geçiş 1: statik
ctx += current_state + fresh_retrieval + recent_history      # geçiş 2: dinamik
```

Claude API'si bu ayrımı alan seviyesinde zaten dayatır: render sırası `tools → system → messages` şeklindedir (§02.1.2).

**İkinci faydası — teşhis.** Beklenmedik davranış ya statik yapılandırmaya (prompt mühendisliği sorunu) ya dinamik duruma (retrieval veya geçmiş yönetimi sorunu) izlenebilir. Ayrım, hata ayıklamayı ikiye böler.

---

## 1.7 Ajan tanımı

Rapor boyunca kullanılan tanım Anthropic'in sade formülasyonudur:

> **Ajan = tool'ları bir döngü içinde otonom olarak kullanan LLM.**

Bu tanımın önemli bir sonucu var: döngünün varlığı ayırt edici değildir. ReAct, LangGraph, CrewAI ve elle yazılmış yirmi satırlık bir `while` — hepsi bu tanıma girer. Ayırt edici olan **döngünün içinde neyin dolaştığı** (§03.2) ve **döngünün çevresinde ne olduğudur** (§04–§08).

---

## 1.8 Raporun iddiası

Modern ajan harness'lerinde karmaşıklık döngünün *içinde* değil, *çevresinde* toplanmıştır:

```
┌─ Bağlam mühendisliği katmanı ─────────────────────────┐
│  progressive disclosure, defer_loading, memory,       │
│  compaction, context editing, çıktı kırpma,           │
│  subagent izolasyonu, prompt caching                  │
│                                                        │
│   ┌─ Harness kontrol katmanı ───────────────────┐    │
│   │  izin kapıları, hook'lar, plan modu,         │    │
│   │  görev listesi, dosya sistemi durumu         │    │
│   │                                               │    │
│   │   ┌─ Ajan döngüsü ────────────────────┐     │    │
│   │   │  while stop_reason == "tool_use"  │     │    │
│   │   │  ← BURASI BASİT (≈20 satır)       │     │    │
│   │   └───────────────────────────────────┘     │    │
│   └───────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
```

ReAct döneminde zekâ prompt şablonundaydı: format talimatı, few-shot örnekler, stop sequence, ayrıştırma mantığı. Döngü karmaşık, çevresi boştu.

Bugün tam tersidir. Döngü `while` kadar sadedir; karmaşıklığın tamamı bağlam yönetimindedir.

**Sonraki bölümler bu çevre katmanını mekanizma mekanizma açıyor.**
