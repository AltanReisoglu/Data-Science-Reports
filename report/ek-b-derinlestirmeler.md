# Ek B — Derinleştirmeler ve Soru-Cevap Notları

**Ağustos 2026**

Bu ek, raporun ana bölümlerinin üstüne, tartışma sırasında ortaya çıkan derinleştirmeleri topluyor. Her başlık ilgili ana bölüme bağlanır; oraya henüz işlenmemiş ama işlenmeye aday malzemedir.

> **Doğruluk notu:** Bu ekteki dış kaynaklı iddialar (SALT, ACE rol ayrımı, ADK `ToolContext`) Ağustos 2026'da web'den / birincil dokümantasyondan doğrulandı. Yine de üretim öncesi teyit edilmeli.

**İçindekiler**
- [B.1 — Compaction'ın üç seviyesi ve dört biçimi (SALT dâhil)](#b1)
- [B.2 — Compaction'ı kim yapar: ajan değil, çağrı](#b2)
- [B.3 — ACE rol ayrımı: Generator / Reflector / Curator](#b3)
- [B.4 — Bölge × compaction: tool/skill/doc'un kaderi](#b4)
- [B.5 — Uzaysal vs zamansal, netleştirme](#b5)
- [B.6 — Framework'lerin harness enjeksiyonu (ADK ToolContext)](#b6)
- [B.7 — Tool trace anatomisi ve trace-özet şeması](#b7)
- [B.8 — Trace sıkıştırma yöntemleri (output vs trace)](#b8)
- [B.9 — Trace temsil biçimi: flat text mi, yapılandırılmış mı](#b9)

---

<a name="b1"></a>
## B.1 — Compaction'ın üç seviyesi ve dört biçimi

**İlgili:** §08 (bağlam basıncı), §11 K5

"Compaction" tek bir şey değil. İki eksende ayrışıyor: **hangi seviyede** çalıştığı ve **hangi biçimde** sıkıştırdığı.

### Üç seviye (nerede çalışır)

| Seviye | Ne | İçerik kaybı | Kim kontrol eder |
|---|---|---|---|
| **1 — Konuşma** | Eski turları yük taşıyan duruma indir | Var (kasıtlı) | Harness |
| **2 — Tool çıktısı** | Büyük çıktıyı dosyaya yaz, özet+yol bırak | Var (referans kalır) | Harness |
| **3 — Mimari/latent** | KV-cache / temsil sıkıştırma | Yok (hassasiyet kaybı olabilir) | Model/eğitim |

- **Glean** yalnızca Seviye 1-2'yi anlatıyor (harness katmanı).
- **Kullanıcı `ai.md` listesi** ek olarak Seviye 3'ü getiriyor: **LCLM** (Latent Context Language Models), **KV-CAT** (KV-Compression Aware Training), **RedKnot** (head-aware KV-cache), End-to-End Context Compression.
- **Rapor şu an yalnızca Seviye 1-2'yi** kapsıyor. Seviye 3 bilinçli kapsam dışı (harness değil model mimarisi) ama okuyucu compaction'ı sadece harness işi sanmasın diye §08'e bir "Seviye 3 vardır, kapsam dışı" notu düşülmeli.

### Dört biçim (nasıl sıkıştırır)

| Biçim | Yöntem | LLM gerekir mi | Örnek |
|---|---|---|---|
| **Abstractive + LLM** | Yeniden yaz | ✅ | Glean conversation compaction, `compact_20260112` |
| **Extractive + istatistik** | Var olan cümlelerden seç | ❌ | **SALT** |
| **Latent + eğitim** | Temsili küçült | ❌ (eğitimde) | LCLM, KV-CAT |
| **Deterministik filtre** | Kural bazlı at | ❌ | RTK (terminal çıktısı) |

### SALT — extractive satırı dolduran çalışma

**SALT: Salience-Aware Lexical Trie for Long-Context Compression** (arXiv 2607.17486, Tem 2026).

- **Çözdüğü problem — theme collapse:** Skor bazlı sıkıştırma her cümleye tek alaka skoru verip en yüksekleri tutar. Sıkı bütçede baskın tema tüm bütçeyi yer, seyrek ama görev-kritik temalar tamamen düşer.
- **Çözümü — iki faz:**
  1. *İndeksleme:* cümle düzeyi anahtar kelime istatistiğinden temaları tahmin et, belgeyi **salience-aware lexical trie** (belirginlik-duyarlı önek ağacı) olarak organize et.
  2. *Seçim:* hedef kelime bütçesi altında ağaçta gezinerek **her temadan pay ayıran** cümle alt kümesi seç.
- **Neden önemli:** LLM çağırmıyor (ucuz, hızlı), deterministik (kayıp denetlenebilir), theme collapse'a özel.
- **Sınırları:** yalnızca extractive (yeniden ifade edemez → konuşma compaction'ı için uygun değil); sözcüksel, anlamsal değil (parafraz/eşanlamlıyı kaçırabilir); seyrek tema gerçekten alakasızsa gürültü korur.
- **ACE'nin brevity bias'ından farkı:** ACE özetlemenin *ayrıntı* kaybını, SALT seçmenin *çeşitlilik* kaybını hedefler.

> Kullanıcının listelerinde ve şu anki raporda extractive satır yoktu; SALT bu boşluğu dolduruyor.

---

<a name="b2"></a>
## B.2 — Compaction'ı kim yapar: ajan değil, çağrı

**İlgili:** §08.7

Compaction'ı bir **LLM** yapıyor — ama ayrı/kalıcı bir "ajan" değil, konuşmaya atılan **tek atışlık özetleme çağrısı.**

| | Compaction çağrısı (gerçek) | Compaction ajanı (yok) |
|---|---|---|
| Kalıcı mı | Hayır — tek atış | Kendi döngüsü olan süreç |
| Tool'u var mı | Hayır | Olabilir |
| Karar verir mi | Hayır — korunacaklar prompt'ta sabit | Kendi stratejisini seçer |
| Adım | 1 (girdi→özet) | Çok adımlı |

**Üç yapılış biçimi:**
1. **İstemci tarafı** — harness ayrı `messages.create` atar (Claude Code `/compact`; bu oturumda gözlendi).
2. **Sunucu tarafı** — `compact_20260112`, API eşiği aşınca kendi içinde özetler.
3. **Ajanın kendisi** — model bir tool'la (`write_summary`) kendi durumunu dosyaya yazar.

**Kritik iki nokta:**
- Genelde **ucuz bir model** yeter (özetleme akıl yürütmeden kolay) — ama yanlış özetleme **sessiz kayıp** riski taşır. Ucuz model, pahalı modelin kritik bulduğu nüansı atabilir. Güvenlik isteniyorsa ana modelle yapılır.
- "Ne korunacağı" özetleyiciye **prompt'ta dikte edilir** — özellikle **başarısız yaklaşımlar**, çünkü LLM'in doğal eğilimi başarısızlıkları atmak; atılırsa ana model aynı hatayı tekrarlar. (Bu oturumun compaction özetinde "stale link after git mv", "section numbering collision" gibi başarısızlıklar korundu — bu sayede aynı hatalar tekrarlanmadı.)

**Fark tek cümlede:** compaction bir **dönüşüm**, ajan bir **süreç.** Compaction, konuşmayı okuyup kısaltan bir fonksiyondur; sadece o fonksiyonu bir dil modeli çalıştırır.

---

<a name="b3"></a>
## B.3 — ACE rol ayrımı: Generator / Reflector / Curator

**İlgili:** §11 K4, §12.7

ACE'de çıkarımı **görevi çözen ajan yapmıyor** — ayrı bir rol yapıyor. Bu, ACE'nin temel tasarım kararı.

| Rol | Ne yapar |
|---|---|
| **Generator** | Görevi çözer, yörünge üretir |
| **Reflector** | Yörüngeye dışarıdan bakar, ders çıkarır |
| **Curator** | Dersleri playbook'a **delta** olarak ekler |

**Kritik nüans — "ayrı rol" ≠ "ayrı ajan":**
- Ayrı **rol** = farklı iş, farklı prompt, farklı çağrı, temiz bağlam. ACE bunu gerektirir.
- Ayrı **ajan/model** = kalıcı bağımsız varlık. **Şart değil** — üç rol de aynı temel modelin farklı promptlarla çağrılması olabilir.

**Neden ayrılıyor (iki sebep):**
1. **Kendi hatasını göremez.** Generator yörüngeyi "doğru" varsayarak üretti; aynı bağlamda "nerede hata yaptım" diye sormak kendi varsayımlarının içinden bakmaktır. Reflector temiz bağlamla, yörüngeyi dış gözlem olarak alır.
2. **Bağlam kirlenmesi.** Generator'ın bağlamı görev gürültüsüyle dolu; Reflector yörüngeyi özet olarak alıp temiz değerlendirir.

Bu, §08.8'in subagent mantığıyla (ayrı rol = ayrı bağlam = odak) ve B.2'nin "compaction ayrı çağrıdır" tespitiyle aynı desen.

> §12.7'deki kodun neden `generation()`/`reflection()`/`curation()` diye **üç ayrı fonksiyon** olduğunun gerekçesi budur. Tek fonksiyon olsaydı ACE değil, "kendi kendine not alan ajan" olurdu.

---

<a name="b4"></a>
## B.4 — Bölge × compaction: her şeyin kaderi

**İlgili:** §02 (bölgeler), §04 (skill katmanları), §08 (compaction)

Bir şeyin compaction'da başına ne geleceği, **hangi bölgede olduğuna** bağlıdır.

```
tools[]   ← tool şemaları              } SABİT PREFIX — compaction DOKUNAMAZ
system    ← sistem promptu, CLAUDE.md, } (cache prefix'i; değişirse cache ölür)
            skill açıklamaları         }
messages[] ← doc/artefakt, tool_use,   } compaction BURAYI işler
             tool_result, skill gövdesi }
```

| Şey | Nasıl girer | Bölge | Compaction'da kaderi |
|---|---|---|---|
| Tool şeması | `tools[]` | Prefix | Dokunulmaz |
| Tool sonucu | `tool_result` | messages | Özet + dosya yolu |
| Skill açıklaması (K1) | system/tools | Prefix | Dokunulmaz |
| Skill gövdesi (K2) | `tool_result` | messages | **Kırpılır → adres kalır** |
| Doc türevi | `tool_result`/`document` | messages | Özet + dosya yolu |
| Sistem promptu | `system` | Prefix | Dokunulmaz |
| Kullanıcı mesajı | `user` | messages | Korunur (niyet) |
| Asistan çıktısı | `assistant` | messages | Kısmen korunur (omurga) |

**İki ilke:**
1. **Prefix'tekiler compaction'ın konusu değil.** Bir şeyi korumak istiyorsan prefix'e koy — ama prefix pahalı (her turda, silinemez). Takas: kalıcılık vs esneklik.
2. **"Adres bırakma" deseni evrensel.** Tool sonucu, skill gövdesi, doc türevi — üçü de aynı dönüşümü geçer: **büyük içerik → özet + geri getirme yolu.** Silmek güvenli çünkü kaynak diskte, yol biliniyor.

> **Bu oturumun canlı kanıtı:** `claude-api` skill gövdesi (~50K) compaction'da kırpıldı ve yerine `[... skill content truncated for compaction; use Read on the skill path ...]` kondu. Ama Katman 1 açıklaması prefix'te olduğu için tam kaldı — model skill'in *var olduğunu* biliyor, *içeriğini* gerekince yeniden Read ediyor. İlk kırpılan skill gövdesi olması tesadüf değil: eski + düşük-sinyalli + geri-getirilebilir olanın üçünü de karşılıyor.

---

<a name="b5"></a>
## B.5 — Uzaysal vs zamansal, netleştirme

**İlgili:** §08.8

Bağlamdan bir şeyi uzak tutmanın iki ekseni:

```
UZAYSAL   →  "NEREDE işlensin?"   →  bilgi ana bağlama HİÇ girmez
ZAMANSAL  →  "NE ZAMAN azalsın?"  →  bilgi girer ama sonra çıkar
```

| | Uzaysal | Zamansal |
|---|---|---|
| Ana bağlama girer mi | Hayır, hiç | Evet, sonra çıkar |
| Mekanizma | Subagent, PTC | Compaction, context editing |
| Karar ne zaman | Önceden (işi göndermeden) | Sonradan (eşik aşılınca) |
| Analoji | İşi taşerona ver, raporu al | Eski evrakı arşive kaldır |
| Yöntem | Önleme (baştan filtrele) | Tedavi (girmişle başa çık) |

**Neden ikisi de gerekli:** Subagent (uzaysal) orkestratörün irrelevant detayı *hiç görmemesini* sağlar; ama ana ajanın **kendi konuşması** yine birikir (subagent bunu azaltmaz). Compaction (zamansal) bu kalanı yönetir. İkisi birbirini tamamlar — biri kapıda çevirir, diğeri içeride eritir.

> **Sınıflandırma tuzağı:** Bazı araçlar "compaction" der ama uzaysaldır. Örn. Context Mode "tool çıktısını SQLite'a boşaltır" — bu offload (uzaysal), özetleme (zamansal) değil. Bir "context reduction" iddiasını değerlendirirken mekanizmaya bak: özetliyor mu (zamansal), taşıyor mu (uzaysal), yoksa sadece ölçüyor mu.

---

<a name="b6"></a>
## B.6 — Framework'lerin harness enjeksiyonu (ADK ToolContext)

**İlgili:** §03.6

Framework'ler tool'u modele **aynı** şekilde verir (imza→function declaration; §03.6'nın bulgusu). Ama harness durumunu tool'un içine **farklı** kanallardan enjekte eder.

**ADK örneği — iki kanal:**

```python
def save_note(text: str, tool_context: ToolContext) -> dict:
    #          ↑ MODEL kanalı    ↑ HARNESS kanalı
    tool_context.state["last_note"] = text     # session state
    tool_context.actions.escalate = True       # sonraki adımı etkile
    return {"saved": True}
```

- `text` → şemaya girer, model doldurur.
- `tool_context` → **şemadan otomatik dışlanır**, ADK runtime enjekte eder, model görmez.

`ToolContext` üç kol taşır: `state` (oku/yaz, kalıcı), `actions` (akış kontrolü), `function_call_id`.

**Değerlendirme — ne katıyor:**

| Soru | Cevap |
|---|---|
| Yeni **yetenek** mi | ❌ Hayır — her harness zaten "modele göstermeden tool'a durum geçirir" (Claude Code'un `Read`'i de harness state'e erişir) |
| Yeni **desen** mi | ⚠️ Kısmen — dependency injection'ı tool'lara uygulamak |
| Yeni **ergonomi** mi | ✅ Evet — test edilebilirlik (sahte context enjekte et), boilerplate silme (şema dışlama otomatik), tiplenmiş keşfedilebilirlik |

**Sonuç:** `ToolContext` mimari yenilik değil, framework konforu — DI deseninin tool tanımına taşınmış hâli. **§03.6'nın "framework farkları yüzeyseldir" tezini çürütmüyor, doğruluyor.** Bu yüzden §03.6'ya ayrı bir "harness enjeksiyon kanalları" tablosu eklemek, yüzeysel farkı önemli göstermek olurdu — en fazla mevcut cümleye tek satırlık dipnot: *"harness'e erişim de framework'e göre farklı paketlenir ama yetenek aynıdır."*

---

<a name="b7"></a>
## B.7 — Tool trace anatomisi ve trace-özet şeması

**İlgili:** §02 (wire formatı), §08 (bağlam basıncı)

### Trace ≠ output

Bu ayrım tüm bölümün temeli:

- **Tool output** = tek bir `tool_result`'ın gövdesi.
- **Tool trace** = bir yörünge boyunca **çağrı-sonuç birimlerinin** turlara gruplu, zaman sıralı dizisi — **artı** birimler arası ilişkiler.

Output compaction (B.1 Seviye 2) aslında ikiye ayrılmalı:
- **2a — output:** tekil sonuç gövdesini küçült
- **2b — trace:** çağrı→argüman→sonuç→hata *dizisini* yapısal sıkıştır

### Atomik birim — üç parça

Bir tool etkileşimi her zaman: çağrı → (yürütme) → sonuç.

```json
// assistant turu
{"type":"tool_use", "id":"toolu_01A", "name":"Read",
 "input":{"file_path":"/config.py"}}

// sonraki user turu — id ile eşleşir
{"type":"tool_result", "tool_use_id":"toolu_01A",
 "content":"port=8080...", "is_error":false}
```

`tool_use_id` ↔ `id` eşleşmesi trace'in **omurgası** — hangi sonucun hangi çağrıya ait olduğunu bağlayan tek şey.

### Tam trace — birimlerin dizisi

```
messages[]
├─ user:      "raporu bul"
├─ assistant: [tool_use: Grep("rapor")]        ┐ birim 1
├─ user:      [tool_result: 3 aday]            ┘
├─ assistant: [tool_use: Read("a.md")]         ┐ birim 2
├─ user:      [tool_result: 200 satır]         ┘
├─ assistant: [tool_use: Read("a.md")]         ┐ birim 3  ← TEKRAR
├─ user:      [tool_result: 200 satır]         ┘
├─ assistant: [tool_use: ls][tool_use: grep]   ┐ birim 4+5 ← PARALEL
├─ user:      [tool_result][tool_result]       ┘
└─ assistant: [text: "buldum..."]              end_turn
```

Trace düz liste değil, **turlara gruplu** dizi (paralel çağrılar tek turda çok birim).

### Trace'i "trace" yapan — ilişkiler

Tekil `tool_result`'ta olmayan, ama dizi düzeyinde görünen:

| Boyut | Örnek |
|---|---|
| **Sıra** | Grep önce, Read sonra |
| **Tekrar** | birim 2 = birim 3 (a.md 2 kez) |
| **Nedensellik** | Grep→"a.md"→Read("a.md") |
| **Hata zinciri** | Read(hata)→Read(düzeltme) |
| **Paralellik** | birim 4+5 tek turda |

Bu ilişki alanları ham trace'te **yok** — compaction analizle üretir. Trace compaction'ı zor yapan da bu: önce yapıyı çıkar, sonra sıkıştır.

### Trace-özet şeması — beş alan

Çekirdek üçlü (kullanıcı çerçevelemesi): **ne için / ne ile / ne oldu.** Alanların kaynağı farklı, ve bu belirleyici:

| Alan | Anlam | Nereden gelir | Ham trace'te var mı |
|---|---|---|---|
| **ne için** | niyet | — | ❌ **Yok — çıkarım gerekir** |
| **ne ile** | girdi | `tool_use.input` | ✅ Var |
| **ne oldu** | sonuç | `tool_result.content` | ✅ Var (özetlenir) |
| **durum** | başarı/hata | `is_error` | ✅ Var |
| **etki** | sonraki adıma bağ | — | ❌ Çıkarım |

**Niyet en değerlisi çünkü ham trace'te yoktur ve sonucu bir soruya bağlayıp aranabilir yapar.** "config.py okundu" değil, "**port bulmak için** config.py okundu → port=8080" — ajan sonra port'a ihtiyaç duyunca cevabı hazır, tekrar okumaya gerek yok.

**Durum ayrı tutulur** çünkü başarısızlıklar ders (§11 K5, Glean'in "başarısız yaklaşımlar"ı). "python komutu yoktu" bir sonuç değil, bir uyarıdır.

Tam birim:
```json
{
  "niyet":  "veritabanı portunu bulmak",     // çıkarım (prose)
  "girdi":  "config.py",                       // input'tan (yapısal)
  "sonuc":  "port=8080",                       // result özeti (verbatim kritik)
  "durum":  "ok",                              // is_error
  "etki":   "DB bağlantısı 8080'e kuruldu"     // nedensellik (prose)
}
```
~15 token; ham birim ~250 token'dı → **16× sıkışma**, niyet/girdi/sonuç korunuyor.

---

<a name="b8"></a>
## B.8 — Trace sıkıştırma yöntemleri

**İlgili:** §08, §11 K5

İki küme: **output** (gövde) ve **trace** (dizi). İkincisi her zaman önce analiz gerektirir.

### Output yöntemleri (tekil `content`)

| Yöntem | Mekanizma | Maliyet | Risk | Liste karşılığı |
|---|---|---|---|---|
| **Kırpma** | Eşik sonrası kes, kesildiğini söyle | 0 | Kör — sonrası kaybolur | (agents.md kırpması) |
| **Deterministik filtre** | Regex ile gürültü at (ANSI, ilerleme) | 0 | Görev-körü | RTK %60-90 |
| **Görev-koşullu LLM** | Ucuz model, "birebir koru" kuralıyla | 1 LLM | Parafraz (hata string) | Squeez |
| **Offload** | Dosyaya yaz, özet+yol bırak | Düşük | Düşük (kaynak durur) | Context Mode %98 |
| **Önbellek** | Tekrarlı çıktıyı sakla, referansla | Düşük | Düşük | token-optimizer %95 |
| **Extractive** | Var olan cümlelerden seç | 0 | Sözcüksel, anlam kaçar | SALT |

### Trace yöntemleri (çağrı-sonuç dizisi)

| Yöntem | Mekanizma | Risk | Liste karşılığı |
|---|---|---|---|
| **Dedup** | Aynı name+input tekrarını katla | Dosya arada değiştiyse yanlış | — |
| **Hata-zinciri katlama** | hata+düzeltmeyi tek derse indir | Hatanın kendisi ders olabilir | — |
| **Keşif katlama** | ls/grep dizisini bulguya indir | Negatif bilgi (neyin olmadığı) kaybolur | Search-as-Code %85 |
| **Yapısal atma** | Konumsal değil, tip bazlı at | Tip yanlış sınıflanırsa | Beyond Compaction (2606.11213) |
| **Yörünge özeti** | Tüm trace'i tek LLM özetine indir | En yüksek — yapı yoruma girer | ACON, ReSum, ACM (2607.23809) |
| **Ajan-kontrollü** | Ana model tool ile kendi trace'ini yönetir | Ajan yanlış karar verirse | — |
| **Önleme (PTC)** | Trace'i hiç oluşturma — sandbox'ta tut | (compaction değil, uzaysal) | Search-as-Code |

### Pratik boru hattı

Tek yöntem değil, katman. Ucuz/güvenli önce, pahalı/riskli sona:

```
Ham etkileşim
 ↓ deterministik filtre        (gürültü at)        ← output
 ↓ dedup + hata katlama        (yapısal tekrar/hata) ← trace
 ↓ eşik aşılırsa görev-koşullu (kalanı özetle)      ← output
 ↓ pencere dolarsa yörünge özet (holistik)          ← trace
 ↓ büyük kalıntı → offload      (dosya+yol)          ← ortak
```

**Sıra kritik:** dedup'ı LLM özetinden *önce* yap — LLM zaten tekilleşmiş trace'i özetler, daha az token, daha az kayıp.

**İki ilke:** output "ne kadar"a, trace "hangi ilişki"ye bakar. Trace yöntemleri her zaman **önce analiz** (tekrar/hata/keşif tespiti), sonra sıkıştırma — bu onları hem daha güçlü hem daha riskli yapar.

---

<a name="b9"></a>
## B.9 — Trace temsil biçimi: flat text mi, yapılandırılmış mı

**İlgili:** §08, §09 (ölçüm)

Trace özeti *nasıl* yazılmalı — ayrı bir tasarım kararı. Araştırma bir **spektrum** gösteriyor; flat text bir uçtaki (kayıplı) seçenek.

### Spektrum

| Biçim | Ne | Denetlenebilir | Kayıp | Kaynak |
|---|---|---|---|---|
| **Verbatim** | Ham tut | ✅ Tam | Yok | — |
| **Flat prose** | LLM doğal dil özeti | ✅ İnsan okur | **Yüksek — parafraz** | LLM summarization |
| **Yapılandırılmış** | Bölümlü/şemalı, kritik string birebir | ✅ Programatik | Orta, kontrollü | Structured summaries |
| **Provenance graph** | DAG, tam köken, tam geri getirme | ⚠️ Karmaşık | Kayıpsız | LCM |
| **Latent** | Buffer token, LoRA compiler | ❌ Opak | Düşük ama okunamaz | LCC (16×) |

Kritik uyarı (kodlama ajanları): *"hata mesajlarını parafraz eden özetleyiciler debugging bağlamını yok eder — compaction ya hata mesajını birebir tutar ya tamamen siler."* Bu tek başına flat prose'u kodlama ajanı için eler.

### Üçlü karışık tiptir → hibrit gerekir

Beş alan aynı temsili istemiyor:

| Alan | En iyi temsil | Neden |
|---|---|---|
| ne için (niyet) | **Prose** | Doğal dil; yapı gerektirmez |
| ne ile (girdi) | **Yapılandırılmış** | Tam değer, parse edilebilir |
| ne oldu (sonuç) | **Hibrit** | Özet prose olabilir, ama hata/ID/sayı **verbatim** |

Cevap "flat mı structured mı" değil — **yapılandırılmış iskelet + verbatim ada.**

### Üç somut format (uygulama)

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
 {"seq":0, "type":"reasoning", "payload":{"text":"NVDA fiyatını YFinance ile kontrol edeceğim."}, "status":"ok"},
 {"seq":1, "type":"tool", "payload":{"name":"get_current_stock_price","args":{"symbol":"NVDA"},"output":"206.6400"}, "status":"ok"},
 {"seq":2, "type":"answer", "payload":{"text":"NVDA şu an $206.64."}, "status":"ok"}
]
```

**Değerlendirme:**

| | Format 1/2 | Format 3 |
|---|---|---|
| Yapı | **Aynı** (isim farkı kozmetik) | Farklı sınıf |
| ne için (niyet) | ❌ **Yok** (tek-tipli) | ✅ `reasoning` olayı |
| ne ile / ne oldu | ✅ | ✅ |
| Sıra | `turn` (kaba) | `seq` (her olay) |
| **Tip-duyarlı compaction** | ❌ Yapamaz (tek tip) | ✅ `type` ile: reasoning'i sıkıştır, tool'u verbatim tut |

**Asıl kazanç:** Format 3'ün `type` alanı iki şeyi birden veriyor — (1) reasoning'i ayrı olay yaparak **niyeti yakalar**, (2) aynı tipleme **yapısal atmayı** (B.8) mümkün kılar. Format 1/2 bunu yapamaz çünkü her şey tek tip.

Üçü de **flat text değil** — hepsi yapılandırılmış. Ayrım "tek-tipli (1/2) vs tiplenmiş olay akışı (3)." Araştırmanın *"reasoning/tool/answer tag'leriyle yapılandırılmış özet üstündür"* bulgusu Format 3'ü işaret ediyor.

### Öneri

Format 3'ü temel al, iki ekleme yap:

```jsonc
{"seq":1, "type":"tool",
 "payload":{"name":"...", "args":{...}, "output":"206.6400", "verbatim":true},
 "status":"ok", "intent_ref":0}
```
- **`verbatim:true`** → kritik string'i parafrazdan korur (hata/ID/sayı)
- **`intent_ref:0`** → tool'u tetikleyen reasoning'e bağlar; compaction reasoning'i sıkıştırsa bile "ne için" izlenebilir kalır

Bu ikisiyle Format 3, B.7'nin beş-alanlı ideal trace-özet birimini tam karşılar.

### Karar rehberi

```
Kodlama/tool-ağırlıklı ajan?  → Evet: yapılandırılmış + verbatim (hata/ID korunur)
Trace'e programatik erişim?   → Evet: yapılandırılmış (parse)
Tam geri getirme/köken?       → Evet: provenance graph (LCM)
Maks sıkışma, denetim önemsiz? → Evet: latent (LCC, riskli)
Basit / insan okuyacak?        → flat prose yeter
```

Çoğu ciddi ajan: **tiplenmiş yapılandırılmış (Format 3) + verbatim kritik alanlar.**

---

## Ana rapora işleme durumu

Bu ekteki maddelerin hangisinin ana bölüme taşınacağı:

| Madde | Hedef bölüm | Öncelik | Durum |
|---|---|---|---|
| B.1 Seviye 3 notu + dört biçim tablosu | §08 | Orta | Bekliyor |
| B.1 SALT (extractive satır) | §08 / §11 K5 | Orta | Bekliyor |
| B.2 Compaction = çağrı | §08.7 | Düşük | Bekliyor |
| B.3 ACE rol ayrımı | §11 K4 | **Yüksek** | Bekliyor |
| B.4 Bölge × compaction tablosu | §08 | **Yüksek** | Bekliyor |
| B.5 Uzaysal/zamansal netleştirme | §08.8 | Düşük (zaten var) | Bekliyor |
| B.6 ADK ToolContext | §03.6 | Düşük (dipnot yeter) | Bekliyor |
| B.7 Trace anatomisi + 5-alan şema | §08 (yeni 2b) | **Yüksek** | Bekliyor |
| B.8 Trace sıkıştırma yöntemleri | §08 / §11 K5 | **Yüksek** | Bekliyor |
| B.9 Trace temsil biçimi (3 format) | §08 / §09 | Orta | Bekliyor |

---

**← Ana rapor:** [00-README](00-README.md) · **İlgili ekler:** [Ek A — Tool referansı](ek-a-tool-referans.md)
