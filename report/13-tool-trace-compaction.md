# 13 — Tool Trace Compaction

**Ağustos 2026 · Bağımsız bölüm**

Bu bölüm kendi başına okunabilir. Bir ajanın tool etkileşim geçmişini — *trace*'ini — nasıl sıkıştıracağını baştan sona ele alır: trace nedir, output'tan farkı, yapısı, bir trace-özetinin ne içermesi gerektiği, hangi yöntemlerle sıkıştırıldığı, ve hangi biçimde temsil edileceği.

> **Doğruluk sınırı:** Buradaki dış kaynaklı iddialar (SALT, ACON, Beyond Compaction, LCM/LCC) Ağustos 2026'da web'den doğrulandı. Alan hızlı hareket ediyor; üretim öncesi birincil kaynaktan teyit edilmeli. ⚠️ işaretli sayılar ikincil aktarımdır.

**İçindekiler**
1. [Problem: neden trace ayrı bir konu](#s1)
2. [Trace nedir — output'tan farkı](#s2)
3. [Anatomi: atomik birimden diziye](#s3)
4. [Trace-özet şeması: ne içermeli](#s4)
5. [Sıkıştırma yöntemleri](#s5)
6. [Ne zaman çalışmalı: tetikleme ve eskime](#s6)
7. [Temsil biçimi: flat text mi, yapılandırılmış mı](#s7)
8. [Karar rehberi](#s8)
9. [Özet](#s9)

---

<a name="s1"></a>
## 1. Problem: neden trace ayrı bir konu

Bir ajan uzun bir görevde onlarca tool çağırır. Her çağrının sonucu bağlam penceresine eklenir ve birikir. Bu birikimi yönetmenin bilinen yolu **tool output compaction**: büyük bir sonucu (200 satırlık dosya, 500 satırlık log) küçültmek.

Ama tek bir sonucu küçültmek, bir şeyi kaçırır. Şu diziye bak:

```
Read(a.py)   → 200 satır
Grep("foo")  → 3 eşleşme
Read(a.py)   → 200 satır      ← AYNI dosya, ikinci kez
Bash("ls")   → 40 dosya
Read(a.py)   → 200 satır      ← ÜÇÜNCÜ kez
```

Output compaction her `Read` sonucunu **ayrı ayrı** küçültür. Ama asıl israfı göremez: **aynı dosya üç kez okundu.** Bu israf tek bir sonuçta değil, sonuçlar **arasındaki ilişkide.**

İşte bu yüzden trace ayrı bir konu. Output compaction "bu sonuç ne kadar büyük" sorusuna bakar; trace compaction "bu çağrılar birbiriyle nasıl ilişkili" sorusuna.

---

<a name="s2"></a>
## 2. Trace nedir — output'tan farkı

| | Tool **output** | Tool **trace** |
|---|---|---|
| Ne | Tek bir `tool_result`'ın gövdesi | Çağrı-sonuç birimlerinin **dizisi** + ilişkileri |
| Görebildiği | "Bu sonuç büyük" | "Bu çağrı 3 kez tekrarladı" |
| Sıkıştırma | Gövdeyi küçült | Yapıyı katla |
| Analiz gerekir mi | Hayır | **Evet** — önce ilişkiyi çıkar |

Tanım:

> **Tool trace** = bir yörünge boyunca çağrı-sonuç birimlerinin (`tool_use` ↔ `tool_result`, `id` ile bağlı) turlara gruplu, zaman sıralı dizisi — **artı** birimler arası ilişkiler (sıra, tekrar, nedensellik, hata zinciri, paralellik).

Pratik sonuç: bağlam basıncı yönetiminde "tool sonucu" katmanı aslında ikiye ayrılır:
- **Output (2a):** tekil sonuç gövdesini küçült
- **Trace (2b):** çağrı→argüman→sonuç→hata dizisini yapısal sıkıştır

Bu bölüm 2b'yi ele alır; 2a'ya yalnızca karşılaştırma için değinir.

> **Literatür doğrulaması (CoACT, arXiv 2607.02911).** Bu 2a/2b ayrımı ölçülmüş: gözlem (output) token'ları toplam tüketimin **%45,7'si** (SWE-bench Verified), Terminal-Bench'te **%67,8'i.** Ve iki katman **tamamlayıcı** — CoACT (output sıkıştırma) bir trajectory-sıkıştırma yöntemiyle (AgentDiet) birleştiğinde maliyet $45,65 → $25,88'e iniyor. Yani 2a ve 2b rakip değil, üst üste binen kazançlar. Bu, §5'teki "boru hattı" iddiasının kanıtı.

---

<a name="s3"></a>
## 3. Anatomi: atomik birimden diziye

### Atomik birim — üç parça

Bir tool etkileşimi her zaman: **çağrı → (yürütme) → sonuç.**

```json
// assistant turu — çağrı
{"type":"tool_use", "id":"toolu_01A", "name":"Read",
 "input":{"file_path":"/config.py"}}

// sonraki user turu — sonuç, id ile eşleşir
{"type":"tool_result", "tool_use_id":"toolu_01A",
 "content":"port=8080...", "is_error":false}
```

`tool_use_id` ↔ `id` eşleşmesi trace'in **omurgasıdır** — hangi sonucun hangi çağrıya ait olduğunu bağlayan tek şey. Yürütmenin kendisi (süre, harness işlemi) trace'te görünmez ama zamanlama meta'sı buradan gelir.

### Tam trace — birimlerin dizisi

Birimler `messages[]` içinde iç içe, turlara gruplu:

```
messages[]
├─ user:      "raporu bul"
├─ assistant: [tool_use: Grep("rapor")]        ┐ birim 1
├─ user:      [tool_result: 3 aday]            ┘
├─ assistant: [tool_use: Read("a.md")]         ┐ birim 2
├─ user:      [tool_result: 200 satır]         ┘
├─ assistant: [tool_use: Read("a.md")]         ┐ birim 3  ← TEKRAR
├─ user:      [tool_result: 200 satır]         ┘
├─ assistant: [tool_use: ls][tool_use: grep]   ┐ birim 4+5 ← PARALEL (tek tur)
├─ user:      [tool_result][tool_result]       ┘
└─ assistant: [text: "buldum..."]              end_turn
```

Trace düz bir liste değil, **turlara gruplu** bir dizidir — paralel çağrılar tek assistant turunda birden çok birim üretir.

### Trace'i "trace" yapan — ilişkiler

Tekil bir `tool_result`'a bakınca görünmeyen, ama dizi düzeyinde görünen boyutlar:

| Boyut | Örnek | Neden önemli |
|---|---|---|
| **Sıra** | Grep önce, Read sonra | Nedenselliğin temeli |
| **Tekrar** | birim 2 = birim 3 | En büyük israf kaynağı |
| **Nedensellik** | Grep→"a.md"→Read("a.md") | "Neden" bu çağrı |
| **Hata zinciri** | Read(hata)→Read(düzeltme) | Başarısızlık = ders |
| **Paralellik** | birim 4+5 tek turda | Gruplanabilir |

**Kritik:** bu ilişki alanları ham trace'te **yoktur.** `repeat_of`, `caused_by` gibi alanları compaction analizle üretir. Trace compaction'ı output'tan zor yapan da budur: önce yapıyı *çıkarman*, sonra sıkıştırman gerekir.

---

<a name="s4"></a>
## 4. Trace-özet şeması: ne içermeli

Bir trace birimini sıkıştırırken korunması gereken çekirdek üçlü:

```
ne için (niyet)  ·  ne ile (girdi)  ·  ne oldu (sonuç)
```

Bu üç alanın **kaynağı farklıdır**, ve bu belirleyici:

| Alan | Anlam | Nereden gelir | Ham trace'te var mı |
|---|---|---|---|
| **ne için** | niyet | — | ❌ **Yok — çıkarım gerekir** |
| **ne ile** | girdi | `tool_use.input` | ✅ Doğrudan var |
| **ne oldu** | sonuç | `tool_result.content` | ✅ Var (özetlenir) |

### Niyet neden en değerlisi

`Read(config.py)` çağrısı **neden** okunduğunu söylemez. O bilgi ya çevredeki `thinking` bloğunda gizlidir ya hiç yoktur. Yani niyet, ham trace'ten **geri kazanılması gereken** tek alandır — ve en değerlisidir:

```
Niyet olmadan:  "config.py okundu → 200 satır"
                → ajan sonra: "bunu neden okumuştum?" → belki tekrar okur

Niyet ile:      "port bulmak için config.py okundu → port=8080"
                → ajan port'a ihtiyaç duyunca cevap hazır → tekrar yok
```

Niyet, sonucu bir **soruya** bağlar ve aranabilir yapar. Bir sonraki adım o soruyu tekrar sorunca, cevap trace özetinde durur.

### İki alan daha

Çekirdek üçlüye iki alan eklemek işe yarar:

| Alan | Neden ayrı |
|---|---|
| **durum** (`is_error`) | Başarısızlıklar **ders**tir; özetleyici LLM'in doğal eğilimi onları atmaktır. "python komutu yoktu" bir sonuç değil, bir uyarıdır. Ayrı tutulmalı ki korunsun. |
| **etki** | Bu sonucun sonraki adımı nasıl değiştirdiği — nedensellik zincirini korur. "port=8080 → DB bağlantısı buna göre kuruldu." |

### Tam trace-özet birimi

```json
{
  "niyet":  "veritabanı portunu bulmak",     // çıkarım (prose)
  "girdi":  "config.py",                       // input'tan (yapısal)
  "sonuc":  "port=8080",                       // result özeti (kritik: verbatim)
  "durum":  "ok",                              // is_error'dan
  "etki":   "DB bağlantısı 8080'e kuruldu"     // nedensellik (prose)
}
```

~15 token. Ham birim ~250 token'dı (200 satır dosya) → **16× sıkışma**, ama niyet/girdi/sonuç korunuyor, ajan devam edebiliyor.

---

<a name="s5"></a>
## 5. Sıkıştırma yöntemleri

İki küme: **output** (tekil gövde) ve **trace** (dizi). İkincisi her zaman önce analiz gerektirir.

### Output yöntemleri (referans için)

| Yöntem | Mekanizma | Maliyet | Risk | Örnek |
|---|---|---|---|---|
| **Kırpma** | Eşik sonrası kes, kesildiğini söyle | 0 | Kör — sonrası kaybolur | (Read cap'i) |
| **Deterministik filtre** | Regex ile gürültü at (ANSI, ilerleme) | 0 | Görev-körü | RTK %60-90 ⚠️ |
| **Görev-koşullu LLM** | Ucuz model, "birebir koru" kuralıyla | 1 LLM | Parafraz (hata string) | Squeez |
| **Offload** | Dosyaya yaz, özet+yol bırak | Düşük | Düşük (kaynak durur) | Context Mode %98 ⚠️ |
| **Extractive** | Var olan cümlelerden seç | 0 | Sözcüksel, anlam kaçar | SALT (2607.17486) |

### Trace yöntemleri (asıl konu)

| Yöntem | Mekanizma | Risk | Örnek |
|---|---|---|---|
| **Dedup** | Aynı name+input tekrarını katla | Dosya arada değiştiyse yanlış | — |
| **Hata-zinciri katlama** | hata+düzeltmeyi tek derse indir | Hatanın kendisi ders olabilir | — |
| **Keşif katlama** | ls/grep dizisini bulguya indir | Negatif bilgi kaybolur | Search-as-Code %85 ⚠️ |
| **Yapısal atma** | Konumsal değil, **tip** bazlı at | Tip yanlış sınıflanırsa | Beyond Compaction (2606.11213) |
| **Yörünge özeti** | Tüm trace'i tek LLM özetine indir | En yüksek — yapı yoruma girer | ACON (2510.00615), ReSum, ACM (2607.23809) |
| **Ajan-kontrollü** | Ana model tool ile kendi trace'ini yönetir | Ajan yanlış karar verirse | — |
| **Önleme (PTC)** | Trace'i hiç oluşturma — sandbox'ta tut | (compaction değil, uzaysal) | Search-as-Code |

### Üç trace yönteminin somut örneği

Aynı 5 birimlik trace, üç yaklaşımla:

```
HAM (5 birim, ~800 token):
  Grep("rapor")→3 aday · Read(a.md)→200 satır · Read(a.md)→200 satır
  · Bash("ls")→40 dosya · Grep("x")→12 eşleşme

ÇIKTI compaction (~300 token):
  Grep→[özet] · Read→[özet] · Read→[özet] · Bash→[özet] · Grep→[özet]
  ↑ her gövde küçüldü, ama 5 birim ve TEKRAR hâlâ orada

TRACE compaction (~120 token):
  "a.md okundu (son hâli: X). ls+2 grep ile keşif, bulgu: Y."
  ↑ tekrar katlandı, keşif tek cümleye indi — YAPI korundu, gövde gitti
```

### Pratik boru hattı

Tek yöntem değil, katman. Ucuz/güvenli önce, pahalı/riskli sona:

```
Ham etkileşim
 ↓ deterministik filtre        (gürültü at)          ← output
 ↓ dedup + hata katlama        (yapısal tekrar/hata) ← trace
 ↓ eşik aşılırsa görev-koşullu (kalanı özetle)       ← output
 ↓ pencere dolarsa yörünge özet (holistik)           ← trace
 ↓ büyük kalıntı → offload      (dosya+yol)           ← ortak
```

**Sıra kritik:** dedup'ı LLM özetinden *önce* yap — LLM zaten tekilleşmiş trace'i özetler; daha az token, daha az kayıp.

**İki ilke:**
1. Output "ne kadar"a, trace "hangi ilişki"ye bakar.
2. Trace yöntemleri **her zaman önce analiz** (tekrar/hata/keşif tespiti), sonra sıkıştırma — bu onları hem daha güçlü hem daha riskli yapar.

> **Uyarı — "complexity trap" (Lindenbauer et al., arXiv 2508.21433).** Boru hattının pahalı ucunu (LLM özetleme) eklemeden önce ölç: bu çalışma, **basit observation masking'in LLM özetlemesiyle aynı çözüm oranını yaklaşık yarı maliyetle** verdiğini gösteriyor. Yani "deterministik önce" sadece sıralama değil — çoğu zaman deterministik **tek başına yeterli.** LLM sıkıştırmasının ek maliyeti hak edip etmediği ölçülmeden varsayılmamalı. Ledger de (2608.00808) sıfır LLM çağrısıyla +8 puan aldı — bu uyarıyı destekliyor.

### 5.1 Modern yöntem manzarası — güncel yaklaşımlar taksonomisi

alphaXiv taraması (Ağu 2026), trace/geçmiş yönetimini **geçmişi neye dönüştürdüğüne** göre altı sınıfa ayırıyor. ECHO'nun (2606.31650) taksonomisi + deterministik execution-state satırı:

| Sınıf | Geçmiş neye dönüşür | Kaynak izi | Örnek çalışmalar |
|---|---|---|---|
| **Append-only** | Hiçbir şeye — tam tutulur | Açık (ama bütçeyi aşar) | Vanilla ReAct |
| **Deletion / pruning** | Turlar atılır | Kısmi / kayıp | Sliding window, AgentDiet |
| **Edited memory** | Düzenlenen kompakt hafıza durumu | Aksiyon-izli | MemAct, Memory-R1 |
| **Collapsed folding** | Uzak geçmiş tek özete katlanır | Çökük (kaynak kaybı) | SUPO, **Context-Folding**, ReSum |
| **Selective turn memory** | Her tur kaynak-indeksli kayıt | **Kaynak-indeksli** | **ECHO** (2606.31650) |
| **Execution state (deterministik)** | Yapılandırılmış durum, LLM'siz | Yapısal, tam | **Ledger** (2608.00808), **CWL** (2606.11213) |

İki eksen bu tabloyu özetliyor:
- **Kim öğreniyor:** deterministik (Ledger, CWL) vs eğitilen (CompactionRL, SUPO, FoldGRPO — özetleyiciyi RL ile öğrenir)
- **İz korunuyor mu:** çökük özet (folding) kaynağı kaybeder; kaynak-indeksli (ECHO) ve execution-state (Ledger/CWL) korur

**Güncel yöntemler, kısa notlar:**

| Yöntem | Fikir | Sonuç |
|---|---|---|
| **CompactionRL** (2607.05378) | Özetlemeyi RL ile eğit — yürütme + özet aynı ödülle optimize | SWE-bench +7 puan; özet kalitesi kritik (49→55,5 sadece özetleyici değişince) |
| **ECHO** (2606.31650) | Kaynak-indeksli tur hafızası + provenance-güdümlü kredi ataması | BrowseComp %43,4 (GRPO %28,9, SUPO %36,1); daha az tur |
| **Context-Folding** (2510.11967) | `branch`/`return` ile alt-bağlam aç/kapa; KV-cache geri sarılır | Ana iz ~8K'da tutulur, toplam 100K+ işlenir |
| **Ledger** (2608.00808) | Deterministik execution ledger, change counter | +8 puan / −%32 maliyet, sıfır LLM |
| **CWL** (2606.11213) | Ajan trace'ini tiplediği DAG + LLM'siz kademeli eviction | 89 görev / 80M token tek oturum, %20-70 maliyet ↓ |

> **Ortak yön:** 2026'nın en güçlü sonuçları ya **deterministik** (Ledger, CWL) ya **kaynak-izini koruyan** (ECHO). Çökük özetleme (folding) hâlâ yaygın ama iki bilinen zaafı var: kaynak kaybı ve özetleyici kalitesine aşırı duyarlılık.

### 5.2 Beyond Compaction (CWL) — pratik mekanizma

**Context Window Lifecycle** (CWL, arXiv 2606.11213) §13'ün Format 3 + eskime tasarımının **ürünleşmiş** hâli — ve tam bir çalışan protokol veriyor. Üç bileşen:

**1. Annotation protokolü — tek tool: `delimiter`.** Ajan trace'ini *çalışırken* kendisi tipliyor (çıkarım sonradan değil):

```jsonc
// Keşif başlat
{"action":"start", "name":"config-arama", "type":"expl"}
// Eylem başlat — bağımlılık DEKLARE et
{"action":"start", "name":"port-yaz", "type":"act", "dependencies":["config-arama"]}
// Keşif bitir — description ZORUNLU (eviction sonrası kalan tek şey)
{"action":"end", "description":"port config.py:41'de, 8080"}
// Eylem bitir
{"action":"end"}
```

İki episode tipi — §13'ün expl/act ayrımının aynısı:

| Tip | İçerik | Eviction'da |
|---|---|---|
| **Exploratory (expl)** | Bilgi toplama (grep, read, ls) | Ham içerik atılır, **description kalır** |
| **Action (act)** | Yazma/düzenleme — etkisi ortama kalıcı | **İLK atılır** (etki zaten diskte) |

**2. Episode graph — tiplenmiş DAG.** Kenarlar yalnızca expl→act (act episode hangi keşfe dayandığını bildirir). Üç değişmez: asiklik, tiplenmiş kenarlar, **prologue koruması** (sistem promptu + tool tanımları + ilk user turu asla evict edilmez).

**3. Eviction policy — deterministik, kademeli, LLM'siz.** Bütçe aşılınca:

```
En eski ACT episode (varsa) → yoksa en eski EXPL episode
  ama: EXPL ancak ona bağlı TÜM act'ler evict edildiyse atılabilir  ← bağımlılık kısıtı
       │
       ▼ episode içinde artan agresiflikte:
  1. Reasoning trace strip   (CoT çıkar — sonucu zaten sonraki adımlarda)
  2. Bulk output strip        (grep/glob/listing tamamen)
  3. Intermediate artifact    (file read, bash çıktısı)
  4. Full episode removal     (son çare)
  → her seviyeden sonra bütçeyi yeniden ölç, karşılandıysa dur
```

**Altı tasarım ilkesi** (§13'ün prensipleriyle birebir):
1. Compaction bir kurtarma değil, **protokolün parçası** — ajan baştan buna göre çalışır
2. **Ajan yapının otoritesidir** — sınırları/bağımlılıkları o deklare eder
3. **User içeriği dokunulmaz** — bütçe karşılanamıyorsa durumu bildir, sessizce bozma
4. **Nedensel bağımlılık recency'yi yener** — eviction grafiği izler, zaman çizgisini değil
5. **Compaction modeli çağırmamalı** — her adım deterministik → halüsinasyon riski yapısal olarak yok
6. **Kademeli, felaket değil** — en küçük artımla başla

**Ölçülmüş sonuç:** tek oturumda **89 sıralı görev, 80M token**, izole-oturum taban çizgisiyle doğruluk paritesi; **%20-70 maliyet düşüşü.**

> **Kritik pratik uyarı — KV cache.** CWL yerinde eviction yapıyor → prefix'i değiştiriyor → cache'i bozuyor. Sürekli bütçe baskısı altında her turda evict olursa cache **net-negatif** olabilir (bizim §11 B.13 prefix argümanımızın ta kendisi). Çözüm: **bütçe tavanı τ'yi sabit tut** (pencerenin ~%30'u, ~80K). Yeni içerik kuyruğa girerken eşit hacim baştan evict olur → prefix'in *bulk*'ı sabit kalır → cache tekrar tutmaya başlar. τ üç boyutlu bir kadran: maliyet (düşük τ ucuz), look-back (düşük τ erken evict → yeniden keşif), model kalitesi (düşük τ dikkat/halüsinasyon rejiminden uzak). Birinci ve üçüncü boyut düşük τ'yi savunuyor; sadece ikincisi karşı çekiyor → **aşağıya doğru hata yapmak daha güvenli.** Pareto: τ ∈ [80K, 120K].

**§13 ile ilişki:** CWL, bizim tasarladığımız her şeyi doğruluyor — Format 3'ün tiplenmiş olayları = episode tipleri; `intent_ref` = dependency deklarasyonu; koruma penceresi = aktif episode; "eski = bağımlılık" = eviction grafiği. Fark: CWL'de tipleme **ajanın kendi tool'uyla, çalışırken** yapılıyor (sonradan çıkarım değil). Bu, POC için doğrudan uygulanabilir bir protokol.

---

<a name="s6"></a>
## 6. Ne zaman çalışmalı: tetikleme ve eskime

İki soru: trace compaction **ne zaman tetiklenmeli**, ve tetiklendiğinde **hangi birimler "eski"** sayılıp sıkıştırılmalı.

### 6.1 Tetikleme: output ingestion'da, trace dikişte

```
Output compaction  →  ingestion'da (çıktı gelir gelmez)  "bu sonuç 25K, hemen kırp"
Trace compaction   →  dikişte (bir iş birimi bitince)     "keşif bitti, ara adımlar gereksiz"
```

Sebep: trace'in sıkışması için önce **bir dizi birikmiş olmalı** — tek çağrıda sıkıştıracak ilişki yok. Output "büyük mü" diye bakar (anında cevaplanır); trace "bu birimler artık gerekli mi" diye bakar (ancak iş bitince cevaplanır). Bu yüzden trace compaction **erken çalışamaz.**

### 6.2 Beş tetikleyici (en iyiden en kötüye)

| Tetikleyici | Ne zaman | Değerlendirme |
|---|---|---|
| **Faz sınırı** | Alt-hedef bitince | **En iyi** — dikiş doğal, ara adımlar gereksizleşir, sonuç taze kalır |
| **Yapısal fırsat** | Tekrar/hata zinciri belirince | İsabetli — boyuta değil israfa tepki |
| **Adım sayısı** | Her N tool çağrısı | Basit ama kör — faz ortasında kesebilir |
| **Token eşiği** | `input_tokens > eşik` | Reaktif, geç — son savunma hattı |
| **Pencere dolması** | Sınıra yakın | Acil durum — buraya gelinmemeliydi |

En çok işi **faz sınırı** yapmalı (en az kayıplı); token eşiği sadece diğerleri yetmezse.

**Faz sınırı sinyalleri** (soyut "faz bitti"nin somut hâli):

| Sinyal | Faz bitti demek |
|---|---|
| `end_turn` | Soru-cevap döngüsü kapandı |
| Write/Edit yapıldı | Keşif→eylem geçişi; keşif katlanabilir |
| Test/komut geçti | Düzeltme fazı bitti |
| Ajan konu değiştirdi | Önceki konunun ara adımları gereksiz |
| TodoWrite'ta madde ✓ | Açık faz sınırı |

Son satır zarif: ajan bir plan artefaktı tutuyorsa, bir maddeyi tamamladığında o maddenin trace'i sıkıştırılabilir — plan durumu doğal dikişleri verir.

### 6.3 Katmanlı tetikleme

```
SÜREKLI:   yapısal tespit (dedup/hata fırsatı görülünce)
DİKİŞTE:   faz/alt-hedef bitince           ← ana tetikleyici
PERİYODİK: her N adımda hafif geçiş (güvenlik ağı)
EŞİKTE:    token sınırı → agresif yörünge özeti (son çare)
```

### 6.4 Eskime: "eski" bir saat değil, bağımlılıktır

Tetikleyici ateşleyince "hangi birim sıkıştırılacak" sorusu gelir. "Eski"yi ölçmenin beş yolu var:

| Ölçü | "Eski" = | Tür |
|---|---|---|
| Tur bazlı | N tur önce | Saat |
| Birim bazlı | Son K birim dışı | Saat |
| Soru-cevap bazlı | Önceki kullanıcı döngüsü | Saat |
| Faz bazlı | Tamamlanmış alt-hedefe ait | Anlam |
| **Bağımlılık bazlı** | Şu anki işin artık kullanmadığı | **Anlam — en doğru** |

**Soru-cevap sınırı yetmez** — iki yönde de yanılır:
- **Çok kaba:** tek soruda birden çok faz olur (`[keşif][yazma][doğrulama]` hepsi tek soru-cevap); keşif, Write olur olmaz eskir ama cevap daha gelmedi.
- **Çok geniş:** önceki cevabın bulgusu sonraki soruda hâlâ gerekli olabilir (Soru 1'de bulunan port, Soru 2'de DB bağlantısında kullanılır).

**Doğru kriter — bağımlılık:** bir birimin sonucu bir sonraki adıma **aktarıldıysa** (etki alanına gömüldüyse) ham birim eskir. Aktarılmadıysa, ne kadar eski olursa olsun tazedir.

### 6.5 Eskimeyi yaklaşıklama — üç sinyal

Gerçek bağımlılık grafiği pahalı; üç sinyalle yaklaşıkla:

```
Birim son 3-5 içinde mi?           → EVET → taze, dokunma (KORUMA PENCERESİ)
     │ hayır
Sonucu son birimlerde anılıyor mu? → EVET → taze (bağımlılık var)
     │ hayır
Ait olduğu faz bitti mi?           → EVET → ESKİ, sıkıştır
     │ belirsiz
Token eşiği aşıldı mı?             → EVET → ESKİ say (agresif, son çare)
                                    → HAYIR → şüphede bırak, dokunma
```

**Koruma penceresi** (son 3-5 birim) her durumda dokunulmaz — ajanın anlık çalışma seti. Glean'in `keep` parametresi budur: analiz yerine "son N garanti taze" emniyeti. Trace compaction **geçmişe** uygulanır, şimdiye değil.

### 6.5.1 Somut mekanizma — Ledger'ın change counter'ı

Yukarıdaki "referans kontrolü" soyut kaldı. **Ledger** (arXiv 2608.00808) bunun deterministik, LLM'siz uygulamasını veriyor — ve "eski = bağımlılık" tezimizi doğrudan kanıtlıyor. Açılış cümlesi tam bizim problemimiz:

> *"Trajectory'de hiçbir şey hangi gözlemin depoyu hâlâ doğru tanımladığını göstermiyor."*

Mekanizma — üç kayıt tipi + iki sayaç:

| Kayıt | Ne tutar |
|---|---|
| **Observation record** | Hangi dosyanın hangi satır aralığı **gerçekten döndü** (sadece istenen değil) + o andaki sayaç değerleri |
| **Modification state** | Hangi dosya/sembol değişti; her dosyada **local counter**, herhangi değişiklikte **global counter** artar |
| **Command record** | Normalize komut + kategori (okuma/arama/test/düzenleme) + konum → exact repeat, tekrarlı test, kısa döngü tespiti |

**Eskimeyi nasıl belirliyor:** bir gözlemin kaydettiği sayaç değeri ile şu anki değer karşılaştırılır.
- Sayaçlar aynı → gözlem **hâlâ geçerli** (reuse edilebilir)
- Local counter değişti → o dosya düzenlendi, gözlem **bayat**
- Belirsizse → **hiç tazelik iddiası yapma** (muhafazakâr)

Bu, bizim §6.5'teki üç sinyalin somut hâli: change counter = "referans kontrolü", command record = "dedup", modification state = "faz sınırı". Ve tamamı **model çağrısı olmadan** — bizim "önce ucuz deterministik" boru hattımızın kanıtı.

**Ölçülmüş sonuç:** SWE-bench Verified'da Pass@1 **+8 puan** (56,2→64,2), maliyet **−%31,8**, tekrar okuma **−%35.** İki müdahale noktası: `inform` (state view'ı context **sonuna** render eder → prefix/cache bozulmaz) + `govern` (redundant komutu yürütülmeden önce durdurur, eski sonucu referansla döner).

> Kilit cümle: *"Uzun-ufuk ajanların eksiği geçmişin daha kısa görünümü değil, kendi yürütme durumlarının açık kaydıdır."* — yani trace compaction sadece "sıkıştır" değil, **yapılandırılmış execution state tut.**

### 6.6 Somut örnek

```
Soru: "raporu düzelt ve doğrula"

birim 1-8:   keşif (grep/read)     ┐
birim 9:     Write(rapor.md)       ┴─ Write → 1-8 ESKİDİ (keşif Write'a gömüldü)
birim 10-12: test (3 çalıştırma)    ┐
birim 13:    test geçti             ┴─ geçti → 10-12 ESKİDİ (düzeltme bitti)
birim 14-15: son 2 birim            → KORUMA PENCERESİ, dokunma

Sıkıştırılabilir: 1-8 (→"keşif, bulgu X"), 10-12 (→"test geçti")
```

Soru-cevap daha bitmedi (cevap birim 15'ten sonra) ama **1-8 çoktan sıkıştırılabilir.** Faz sınırı bunu görür, soru-cevap sınırı göremezdi.

### 6.7 Uçtan uca pratik akış — bütün mekanizma tek diyagramda

Buraya kadarki her şey (Format 3, ledger, eskime, eviction, koruma penceresi) tek bir çalışan döngüde birleşir. Trace compaction, ajan döngüsünde **model çağrısı ile tool yürütmesi arasındaki boşluğa** girer — ek-a'daki "harness = 2. ve 3. adım arası" fikri.

```
┌──────────────────────────────────────────────────────────────────────┐
│  T0  KULLANICI İSTEĞİ                                                  │
│      "config'i bul, portu 9090 yap, testi geçir"                      │
└──────────────────────────────────────────────────────────────────────┘
                              │
        ╔═════════════════════▼══════════════════════╗
        ║          AJAN DÖNGÜSÜ (her tur)             ║
        ╚═════════════════════╤══════════════════════╝
                              │
        ┌─────────────────────▼──────────────────────┐
        │ 1. MODEL üretir                             │
        │    thinking + tool_use(Grep "port")         │
        │    stop_reason: "tool_use"                   │
        └─────────────────────┬──────────────────────┘
                              │
   ┌───────────────  T R A C E   K A T M A N I  ───────────────┐
   │  (model çağrısı ile yürütme ARASINDAKİ boşluk = harness)  │
   │                                                            │
   │  2a. OLAY KAYDET (Format 3, §7)                            │
   │      {seq, type:"tool", payload:{name,args}, intent_ref}   │
   │      reasoning → ayrı olay (niyet yakalanır, §4)           │
   │                                                            │
   │  2b. YÜRÜT + SONUCU KAYDET                                 │
   │      grep çalışır → output → payload'a (verbatim, §7)      │
   │                                                            │
   │  2c. LEDGER GÜNCELLE (deterministik, LLM'siz — Ledger)     │
   │      • observation record: hangi dosya/satır DÖNDÜ         │
   │      • command record: normalize komut + kategori          │
   │      • change counter: local/global ++                     │
   │              │                                             │
   │              ▼ ANALİZ (§5 trace yöntemleri)                │
   │      ┌───────────────────────────────────────┐            │
   │      │ Bu çağrı tekrar mı?  → dedup           │            │
   │      │ Hata sonrası düzeltme? → zincir katla  │            │
   │      │ Dosya değişti mi? → eski gözlem bayat  │            │
   │      └───────────────────────────────────────┘            │
   │                                                            │
   │  2d. TETİKLEME KONTROL (§6.2)                              │
   │      Faz bitti mi? (Write/test-geçti/konu-değişti)         │
   │      │         │ token eşiği aşıldı mı?                    │
   │      │ hayır   │ hayır → EVICTION YOK, devam               │
   │      ▼ evet    ▼ evet                                      │
   │  ┌──────────── EVICTION (§5.2 CWL, kademeli) ───────────┐  │
   │  │ ADAY: son 3-5 birim HARİÇ (koruma penceresi, §6.5)   │  │
   │  │       + faz bitmiş + bağımlısı kalmamış               │  │
   │  │   ▼                                                  │  │
   │  │ SEÇ: en eski ACT (etki diskte) → yoksa en eski EXPL  │  │
   │  │   ▼ kademeli strip:                                  │  │
   │  │   reasoning → bulk output → intermediate → full      │  │
   │  │   her adımdan sonra bütçeyi ölç, karşılandıysa dur    │  │
   │  │   ▼                                                  │  │
   │  │ SONUÇ: ham birim → 5-alan özet (§4)                  │  │
   │  │   {niyet, girdi, sonuç:verbatim, durum, etki}        │  │
   │  │   veya EXPL için → description (tek satır)            │  │
   │  └──────────────────────────────────────────────────────┘ │
   │                                                            │
   │  2e. [opsiyonel] DOĞRULA (§7 NAP)                          │
   │      Sıkıştırma sonrası "sonraki aksiyon" değişti mi?      │
   │      Değiştiyse → o alanı verbatim geri koy                │
   └────────────────────────────┬───────────────────────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │ 3. tool_result olarak messages[]'e ekle       │
        │    (sıkıştırılmış geçmiş + taze son birimler) │
        └──────────────────────┬──────────────────────┘
                               │
                  ┌────────────▼────────────┐
                  │  stop_reason == tool_use │──evet──┐
                  │  ?                        │        │ 1'e DÖN
                  └────────────┬────────────┘        │ (döngü)
                               │ hayır (end_turn)     └────────┘
                               ▼
                        NİHAİ YANIT
```

**Diyagramın okunuşu — üç ayrı zaman ölçeği:**

| Ne | Ne zaman | Maliyet |
|---|---|---|
| **Olay kaydı + ledger** (2a-2c) | **Her turda** | Deterministik, ~0 (LLM yok) |
| **Eviction** (2d) | **Faz sınırında** (her turda değil) | Deterministik strip; opsiyonel LLM özet |
| **NAP doğrulama** (2e) | Sadece agresif sıkıştırmada | 1 ucuz model çağrısı |

**Kritik tasarım noktaları:**
1. **Ledger her turda, eviction ara ara.** Kayıt sürekli (ucuz), sıkıştırma seyrek (faz sınırında). Bu §6.1'in "output ingestion'da, trace dikişte" ayrımı.
2. **Prefix bozulmuyor:** yeni içerik `messages[]` sonuna eklenir; eviction eski birimleri özete çevirir ama sistem promptu/tool tanımları (prologue) asla — cache korunur (§5.2 KV uyarısı).
3. **Koruma penceresi kutunun içinde:** son 3-5 birim eviction adayına hiç girmez — ajanın anlık çalışma seti.
4. **Hepsi deterministik olabilir:** 2a-2d LLM gerektirmez (Ledger + CWL kanıtı). LLM yalnızca opsiyonel özet (2d sonu) ve doğrulama (2e) için.

### 6.8 Tam mekanizma: tetikleme → eskime → evict eleği → çıktı

§6.7 döngüyü gösterdi; bu diyagram **sıkıştırma anının içini** açar — bir faz bitince ne tetiklenir, eskime nasıl belirlenir, evict eleği hangi sırayla çalışır, ve bir birim neye dönüşür. (POC'deki 6 fazın birebir karşılığı.)

```
╔═══════════════════════════════════════════════════════════════════════╗
║  A. TETİKLEME — compaction ne zaman çalışır                            ║
╚═══════════════════════════════════════════════════════════════════════╝

   Her tool sonrası ledger güncellenir (ucuz, her turda).
   Compaction ise ŞU İKİ sinyalden biriyle tetiklenir:

     ┌─ FAZ SINIRI ────────────┐     ┌─ TOKEN EŞİĞİ ──────────┐
     │ Write yapıldı            │     │ trace_tokens > bütçe    │
     │ test geçti               │ VEYA│                         │
     │ ajan konu değiştirdi     │     │ (son çare, geç)         │
     └──────────┬──────────────┘     └──────────┬─────────────┘
                └───────────────┬────────────────┘
                                ▼ tetiklendi

╔═══════════════════════════════════════════════════════════════════════╗
║  B. ESKİME — hangi birim "eski" (change counter mekanizması)           ║
╚═══════════════════════════════════════════════════════════════════════╝

   Her okuma, dosyanın O ANKİ sürüm sayacıyla damgalanır:

     seq5  read config.py   → damga: local_counter[config]=0
     seq9  read config.py   → damga: local_counter[config]=0
     seq17 WRITE config.py  → local_counter[config]: 0 → 1   ← SAYAÇ ARTAR
     seq19 read config.py   → damga: local_counter[config]=1

   is_stale(seq) = (damga < güncel sayaç) ?
     seq5:  0 < 1  → BAYAT   (yazmadan önce okundu, içerik artık yanlış)
     seq19: 1 = 1  → TAZE    (yazmadan sonra okundu)

   → "eski" bir SAAT değil, BAĞIMLILIK: sonucu bir yazmayla geçersizleşen birim.

╔═══════════════════════════════════════════════════════════════════════╗
║  C. EVICT ELEĞİ — 6 faz, güvenlik sırasında (kayıp artan sırada)       ║
╚═══════════════════════════════════════════════════════════════════════╝

   ÖNCE: koruma penceresi ayrılır (son 3-5 birim → DOKUNULMAZ)

   tool birimleri
      │
      ▼ FAZ 1  DEDUP          aynı çağrı tekrar mı?          [sıfır kayıp]
      │        └ kanıt: aynı name+args, değişmemiş koşul
      ▼ FAZ 2  STALENESS      B'deki bayat mı?               [sıfır kayıp]
      │        └ kanıt: damga < güncel sayaç
      ▼ FAZ 3  HATA-ZİNCİRİ   hata + sonra düzeltme?         [düşük kayıp]
      │        └ hatayı katla AMA mesajı verbatim koru (ders)
      ▼ FAZ 4  KEŞİF KATLAMA  ardışık ls/grep dizisi?        [bulgu korunur]
      │        └ diziyi tek bulguya indir, verbatim sonuç kalır
      ═══ buraya kadar hâlâ bütçe aşılıyorsa ═══
      ▼ FAZ 5  KATEGORİ       act önce (etki diskte), expl sonra  [son çare]
      │        └ her adımda ölç, bütçe karşılanınca DUR (açgözlü)
      ▼ FAZ 6  CWL EPISODE    ajan-tipli, bağımlılık-farkında
               └ expl ANCAK bağlı act evict edildiyse → description'a in

   Her faz bir öncekinin bıraktığına bakar; gerçekten gerekirse müdahale.

╔═══════════════════════════════════════════════════════════════════════╗
║  D. ÇIKTI — evict edilen birim neye dönüşür (5-alan şema §4)           ║
╚═══════════════════════════════════════════════════════════════════════╝

   ÖNCE (evicted=False):                    SONRA (evicted=True):
   ┌────────────────────────────┐          ┌──────────────────────────┐
   │ payload: {                 │          │ compacted: {             │
   │   name: "read_file",       │  ──────► │   niyet: "port bul"      │ ← intent_ref'ten
   │   args: {path: config.py}, │  evict   │   girdi: "path=config"   │ ← args'tan
   │   output: "<40 satır>"     │          │   sonuc: "PORT=8080"     │ ← verbatim ise birebir
   │ }                          │          │   durum: "ok",           │
   │ ~200 token                 │          │   etki:  "bayat"         │ ← neden atıldı
   └────────────────────────────┘          │ } ~15 token              │
                                           └──────────────────────────┘
   İçerik gitti; niyet + kritik sonuç + neden KALDI.   → 13× küçülme

   Silme DEĞİL — olay yerinde durur, id eşleşmesi bozulmaz,
   sadece gösterimi (payload → compacted) değişir.
```

**Dört aşamanın tek cümlesi:** *bir faz bitince* (A), *change counter bayat birimleri işaretler* (B), *evict eleği en güvenliden en agresife sıkıştırır* (C), *her birim niyet+sonucu koruyan 5-alan özete iner* (D) — koruma penceresi hariç, ve hiçbir aşama LLM gerektirmeden.

---

<a name="s7"></a>
## 7. Temsil biçimi: flat text mi, yapılandırılmış mı

Trace özeti *nasıl yazılmalı* — ayrı bir tasarım kararı. Flat text bir uçtaki (kayıplı) seçenek; bir spektrum var.

### Spektrum

| Biçim | Ne | Denetlenebilir | Kayıp | Örnek |
|---|---|---|---|---|
| **Verbatim** | Ham tut | ✅ Tam | Yok | — |
| **Flat prose** | LLM doğal dil özeti | ✅ İnsan okur | **Yüksek — parafraz** | LLM summarization |
| **Yapılandırılmış** | Bölümlü/şemalı, kritik string birebir | ✅ Programatik | Orta, kontrollü | Structured summaries |
| **Provenance graph** | DAG, tam köken, tam geri getirme | ⚠️ Karmaşık | Kayıpsız | LCM |
| **Latent** | Buffer token, LoRA compiler | ❌ Opak | Düşük ama okunamaz | LCC (16× ⚠️) |

Kodlama ajanları için belirleyici uyarı:

> *"Kodlama ajanları tam token'larla çalışır; hata mesajlarını parafraz eden özetleyiciler debugging bağlamını yok eder — compaction ya hata mesajını **birebir tutar** ya tamamen siler."*

Bu tek başına **flat prose'u kodlama ajanı için eler** — çünkü `SyntaxError: line 42`'yi "bir sözdizimi hatası" diye bozar.

### Üçlü karışık tiptir → hibrit gerekir

Beş alan aynı temsili istemiyor:

| Alan | En iyi temsil | Neden |
|---|---|---|
| ne için (niyet) | **Prose** | Doğal dil; yapı gerektirmez |
| ne ile (girdi) | **Yapılandırılmış** | Tam değer, parse edilebilir |
| ne oldu (sonuç) | **Hibrit** | Özet prose olabilir; hata/ID/sayı **verbatim** |

Cevap "flat mı structured mı" değil — **yapılandırılmış iskelet + verbatim ada.**

### Üç somut format

Sahada gözlenen üç temsil:

```jsonc
// Format 1 — record (tek-tipli)
{"turn":1, "name":"get_current_stock_price", "args":{"symbol":"NVDA"},
 "result":"206.6400", "status":"ok"}

// Format 2 — steps (1 ile AYNI, alan adları farklı)
{"tool_name":"get_current_stock_price", "tool_input":{"symbol":"NVDA"},
 "tool_output":"206.6400", "status":"ok"}

// Format 3 — events (tiplenmiş olay listesi)
[
 {"seq":0, "type":"reasoning",
  "payload":{"text":"NVDA fiyatını YFinance ile kontrol edeceğim."}, "status":"ok"},
 {"seq":1, "type":"tool",
  "payload":{"name":"get_current_stock_price","args":{"symbol":"NVDA"},"output":"206.6400"},
  "status":"ok"},
 {"seq":2, "type":"answer",
  "payload":{"text":"NVDA şu an $206.64."}, "status":"ok"}
]
```

**Değerlendirme:**

| | Format 1/2 | Format 3 |
|---|---|---|
| Yapı | **Aynı** (isim farkı kozmetik) | Farklı sınıf |
| ne için (niyet) | ❌ **Yok** (tek-tipli) | ✅ `reasoning` olayı |
| ne ile / ne oldu | ✅ | ✅ |
| Sıra | `turn` (kaba) | `seq` (her olay) |
| **Tip-duyarlı compaction** | ❌ Yapamaz | ✅ `type` ile: reasoning'i sıkıştır, tool'u verbatim tut |

**Asıl kazanç:** Format 3'ün `type` alanı iki şeyi birden verir — (1) reasoning'i ayrı olay yaparak **niyeti yakalar**, (2) aynı tipleme **yapısal atmayı** (§5) ve tip bazlı eskimeyi (§6) mümkün kılar. Format 1/2 bunu yapamaz çünkü her şey tek tip.

Üçü de **flat text değildir** — hepsi yapılandırılmış. Ayrım "tek-tipli (1/2) vs tiplenmiş olay akışı (3)."

### Öneri: Format 3 + iki ekleme

```jsonc
{"seq":1, "type":"tool",
 "payload":{"name":"...", "args":{...}, "output":"206.6400", "verbatim":true},
 "status":"ok", "intent_ref":0}
```
- **`verbatim:true`** → kritik string'i parafrazdan korur (hata/ID/sayı)
- **`intent_ref:0`** → tool'u tetikleyen reasoning'e bağlar; compaction reasoning'i sıkıştırsa bile "ne için" izlenebilir kalır

Bu ikisiyle Format 3, §4'teki beş-alanlı ideal trace-özet birimini tam karşılar.

### Sıkıştırma bir şey kaybetti mi — NAP ile ölçüm

Temsil biçimini seçtin, sıkıştırdın; peki kritik bir bilgi kayboldu mu? **CoACT'in NAP'ı (next-action preservation, arXiv 2607.02911)** bunun ucuz ve ölçülebilir cevabı:

> Sıkıştırılmış birim, ham birimle **aynı sonraki aksiyonu** ürettiriyorsa, karar için gereken bilgi korunmuştur.

Neden zekice: final görev başarısını (PASS@1) beklemek hem pahalı (tüm yörüngeyi çalıştırman gerekir) hem seyrek (yüzlerce sıkıştırma kararının ortak etkisi). NAP ise **adım düzeyinde, yoğun** sinyal — sıkıştırmadan sonra modelin bir sonraki aksiyonu değişti mi diye bakar. Değişmediyse o sıkıştırma güvenli.

Bu, §4'teki "verbatim koru" kuralının doğrulama karşılığı: hangi string'i koruyacağını sezgiyle değil, **NAP testiyle** belirlersin — bir alanı atınca sonraki aksiyon değişiyorsa o alan verbatim kalmalı. §9'daki probe tabanlı değerlendirmenin (recall/artifact/continuation) trace'e özgü hâli.

---

<a name="s8"></a>
## 8. Karar rehberi

```
Ajan kodlama/tool-ağırlıklı mı?
  Evet → hata/ID/yol korunmalı → YAPILANDIRILMIŞ + verbatim (Format 3)
  Hayır ↓
Trace'e sonradan programatik erişilecek mi?
  Evet → YAPILANDIRILMIŞ (parse edilebilir)
  Hayır ↓
Tam geri getirme / köken gerekli mi (denetim, tekrar oynatma)?
  Evet → PROVENANCE GRAPH (LCM)
  Hayır ↓
Maksimum sıkışma, denetlenebilirlik önemsiz mi?
  Evet → LATENT (LCC, 16×) — ama okunamaz, riskli
  Hayır → FLAT PROSE (basit, insan okur)
```

**Yöntem seçimi:**

```
Trace uzun ama çok tekrar var    → dedup önce
Çok hata-düzeltme döngüsü         → hata-zinciri katlama
Çok keşif adımı (ls/grep)         → keşif katlama
Pencere doluyor, holistik gerek   → yörünge özeti (dedup'tan SONRA)
Ajan en iyi bilir neyin gerektiğini → ajan-kontrollü
En iyisi: trace'i hiç oluşturma   → PTC / Search-as-Code (önleme)
```

Çoğu ciddi ajan için: **tiplenmiş yapılandırılmış temsil (Format 3) + boru hattı (filtre→dedup→görev-koşullu→gerekirse yörünge özeti).**

---

<a name="s9"></a>
## 9. Özet

1. **Trace ≠ output.** Output tekil sonuç gövdesi; trace çağrı-sonuç birimlerinin dizisi + ilişkileri. Output "ne kadar", trace "hangi ilişki."

2. **İlişkiler ham trace'te yoktur** — tekrar, nedensellik, hata zinciri compaction tarafından analizle üretilir. Bu, trace compaction'ı output'tan hem güçlü hem riskli yapar.

3. **İyi trace özeti üç şey yapar:**
   - **Niyeti geri kazanır** (ham trace'te yok, en değerli — sonucu aranabilir yapar)
   - **Kritik string'i verbatim korur** (hata/ID/sayı parafraz edilemez)
   - **Tiplenmiş olur** (reasoning/tool/answer → tip-duyarlı sıkışma mümkün)

4. **Beş-alan şema:** niyet · girdi · sonuç · durum · etki. Durum ayrıdır çünkü başarısızlıklar derstir; etki nedensellik zincirini korur.

5. **Temsil:** flat text kayıplı uçtur. Kodlama ajanında yapılandırılmış+verbatim kazanır. Somut form: **Format 3 (tiplenmiş olay) + `verbatim` + `intent_ref`.**

6. **Tetikleme dikiştedir, boyutta değil:** output ingestion'da çalışır, trace bir iş birimi bitince (faz sınırı) — çünkü sıkıştırılacak ilişki ancak faz tamamlanınca oluşur. Token eşiği son çaredir.

7. **"Eski" bir saat değil, bağımlılıktır:** bir birimin sonucu sonraki adıma aktarıldıysa ham birim eskir. Soru-cevap sınırı yetmez (hem kaba hem yanıltıcı); faz sınırı + referans kontrolü + koruma penceresi (son 3-5 birim dokunulmaz) ile yaklaşıklanır.

8. **En etkili "trace compaction" çoğu zaman trace'i hiç oluşturmamaktır** — PTC ile ara adımları sandbox'ta tutmak (uzaysal önleme).

9. **Deterministik çoğu zaman yeterlidir:** Ledger (2608.00808) sıfır LLM çağrısıyla +8 puan / −%32 maliyet aldı; Lindenbauer (2508.21433) basit masking'in LLM özetiyle eşit çözüm oranını yarı maliyetle verdiğini gösterdi. LLM sıkıştırmasının ek maliyeti ölçülmeden varsayılmamalı.

---

## Literatür dayanağı (alphaXiv, Ağustos 2026)

Bu bölümün tasarım iddialarının her biri güncel literatürde doğrulandı. Tarama alphaXiv üzerinden yapıldı; iki paper tam metin okundu (**Ledger**, **CoACT**), kalanların özeti tarandı.

| §13 iddiası | Doğrulayan çalışma | Nasıl |
|---|---|---|
| Eskime = bağımlılık, saat değil | **Ledger** (2608.00808) | Change counter ile deterministik bayatlık tespiti; +8 puan Pass@1 |
| Deterministik önce, çoğu zaman tek başına yeter | Ledger + **Lindenbauer complexity trap** (2508.21433) | Sıfır-LLM ledger +8 puan; masking ≈ özetleme yarı maliyetle |
| Trace ≠ output (2a/2b), tamamlayıcı | **CoACT** (2607.02911) | Output %45,7–67,8 pay; CoACT+AgentDiet $45→$25 |
| Kayıp ölçülebilir (verbatim doğrulama) | **CoACT NAP** | Sonraki-aksiyon korunumu = adım düzeyi kayıp sinyali |
| "Load-bearing" durum korunmalı | **Plans Don't Persist** (2606.22953) | Compaction ancak atılan bilgi artık gerekmiyorsa güvenli |
| Compaction sessiz kayıp riski taşır | **Governance Decay** (2606.22528) | Compaction güvenlik kısıtlarını sessizce siliyor |
| Yapısal (tip bazlı) atma + tiplenmiş DAG | **Beyond Compaction / CWL** (2606.11213) | Konumsal değil yapısal eviction; delimiter protokolü, 89 görev/80M token |
| Kaynak-izini koruma (folding'in aksine) | **ECHO** (2606.31650) | Kaynak-indeksli tur hafızası; BrowseComp %43,4 |
| Özetlemeyi RL ile eğitmek | **CompactionRL** (2607.05378) | Yürütme+özet ortak ödül; SWE-bench +7 puan |

**Tam metin okunan (5):** Ledger (2608.00808), CoACT (2607.02911), CWL/Beyond Compaction (2606.11213), ECHO (2606.31650), CompactionRL (2607.05378).

**Taranan ama tam okunmayan yakın küme** (POC öncesi okunmaya değer): Self-Compacting Agents (2606.23525), TokenPilot (2606.17016, cache-verimli), Memory as Execution State (2606.06090), ACM (2607.23809), SWE-MeM (2606.28434), Addressable Recall Compaction (2607.25066), Governance Decay (2606.22528), Plans Don't Persist (2606.22953).

> ⚠️ Beş paper tam metin okunup doğrulandı. Diğerleri yalnızca özet düzeyinde tarandı; iddiaları rapora girmeden tam metinden teyit edilmeli.

---

## Kaynaklar

**Trace/yörünge sıkıştırma — alphaXiv'de tam metin okundu**
- [Ledger: Turning Interaction History into Execution State](https://arxiv.org/abs/2608.00808) — deterministik execution ledger, change counter, +8 puan / −%32 maliyet. §6.5.1'in temeli
- [CoACT: Action-Preserving Observation Compression for Coding Agents](https://arxiv.org/abs/2607.02911) — NAP (next-action preservation), 2a/2b tamamlayıcılığı. §2 ve §7'nin temeli
- [Beyond Compaction: Structured Context Eviction (CWL)](https://arxiv.org/abs/2606.11213) — delimiter protokolü, tiplenmiş episode DAG, LLM'siz kademeli eviction. **§5.2'nin temeli** (pratik mekanizma)
- [ECHO: Prune To Act, Trace To Learn](https://arxiv.org/abs/2606.31650) — kaynak-indeksli tur hafızası, modern yöntem taksonomisi. §5.1'in temeli
- [CompactionRL: RL with Context Compaction](https://arxiv.org/abs/2607.05378) — özetlemeyi RL ile eğitme; özet kalitesi kritik

**Trace/yörünge sıkıştırma — özet tarandı ⚠️ *(tam metinden teyit edilmeli)***
- [Plans Don't Persist: Why Context Management Is Load Bearing](https://arxiv.org/abs/2606.22953) — atılan bilgi artık gerekmiyorsa güvenli
- [Governance Decay: How Compaction Silently Erases Safety Constraints](https://arxiv.org/abs/2606.22528) — sessiz kayıp riski
- [Self-Compacting LM Agents](https://arxiv.org/abs/2606.23525) · [TokenPilot](https://arxiv.org/abs/2606.17016) · [Memory as Execution State](https://arxiv.org/abs/2606.06090) · [ACM](https://arxiv.org/abs/2607.23809) · [SWE-MeM](https://arxiv.org/abs/2606.28434)
- [Context-Folding (branch/return)](https://arxiv.org/abs/2510.11967) — alt-bağlam KV-cache geri sarma
- [The Complexity Trap: Masking ≈ LLM Summarization (Lindenbauer)](https://arxiv.org/abs/2508.21433) — deterministik yeterlilik uyarısı
- [ACON](https://arxiv.org/pdf/2510.00615) · [Slipstream](https://arxiv.org/html/2605.08580v1) ⚠️

**Temsil biçimi**
- [Compaction vs Summarization — Morph](https://www.morphllm.com/compaction-vs-summarization) — verbatim vs LLM özet vs opak
- [Context Compaction: Delete Noise, Keep Signal — Morph](https://www.morphllm.com/context-compaction) — hata string'i verbatim koruma kuralı
- [Structured Outputs vs Tool Calling — DEV](https://dev.to/thedailyagent/structured-outputs-vs-tool-calling-when-your-agent-actually-needs-which-kgk) — yapılandırılmış vs freeform
- [ExpGraph: Graph-Structured Memory for LLM Agents](https://arxiv.org/pdf/2605.30712) — graf tabanlı trace/deneyim

**Output/extractive sıkıştırma**
- [SALT: Salience-Aware Lexical Trie for Long-Context Compression](https://arxiv.org/abs/2607.17486) — extractive, tema-korumalı seçim
- Kullanıcı listeleri — Search-as-Code %85, Context Mode %98, RTK %60-90 ⚠️ *depo iddiaları, bağımsız ölçülmedi*

---

**← Önceki:** [12 — MCP ve modern yöntemler](12-mcp-ve-modern-yontemler.md) · **İlgili:** [08 — Bağlam basıncı](08-baglam-basinci.md) · [Ek B — Derinleştirmeler](ek-b-derinlestirmeler.md)
