# 13-Ek — Ajan Episode'u Ne Zaman Açar: Kıstas

**Ağustos 2026 · §13 / CWL derinleştirmesi · POC'ye dayalı**

Bu ek tek bir soruya odaklanır: **Bir ajan, çağırdığı tool'ları bir episode'a (CWL keşif/eylem grubuna) koyacağını nasıl bilir? Kıstas nedir?**

Kısa cevap önce: **Sistem ajanı zorlamaz. Episode'u ajanın kendisi açar; ne zaman açacağını da system prompt'a gömülü bir yönerge söyler.** Ledger otomatiktir (tool adından kategori çıkar); CWL ise ajan-bildirimlidir. Bu ek, o yönergeyi, altındaki kavramsal kıstası ve bildirilmediğinde ne olduğunu açar.

**İçindekiler**
1. [Ledger otomatik, CWL bildirimli — fark](#s1)
2. [Kıstas nerede yaşıyor: system prompt](#s2)
3. [Altındaki gerçek ayrım: yan etki (expl vs act)](#s3)
4. [dependencies: eylem hangi keşfi tüketti](#s4)
5. [Garanti değil, öneri: üç sonuç](#s5)
6. [Neden otomatik episode koymadık](#s6)
7. [Daha güçlü kıstas: açık eşikler](#s7)
8. [Özet](#s8)

---

<a name="s1"></a>
## 1. Ledger otomatik, CWL bildirimli — fark

İki defter, kıstasları taban tabana zıt:

| | Ledger | CWL episode grafiği |
|---|---|---|
| Kim doldurur | **sistem** (otomatik) | **ajan** (delimiter ile) |
| Kıstas nereden | tool adı → kategori haritası | ajanın yargısı + prompt yönergesi |
| Neyi bilir | "bu tool read'di, kaynağı XOM, sürüm 0" | "bu 8 tool tek bir 'veri-toplama' keşfi" |
| Yanlış olursa | olamaz (deterministik) | olabilir (graceful degradation ile korunur) |

Ledger, `get_stock_price`'ın read olduğunu **tool adından** çıkarır — ajana sormaz. Ama "bu 8 tool tek bir mantıksal iştir" bilgisi **niyet**tir; tool adından çıkmaz. Bu yüzden CWL'yi ajan bildirmek zorunda.

---

<a name="s2"></a>
## 2. Kıstas nerede yaşıyor: system prompt

Kıstas, ajana verilen system prompt'a gömülü (`chat.py:19-30`, `EQUITY_SYSTEM`):

```
- Çok veri toplayacaksan ÖNCE delimiter(action='start', type='expl',
  name='veri-toplama') çağır; toplama bitince delimiter(action='end',
  description='ne öğrendin') çağır.
- Grafik/rapor üretmeden ÖNCE delimiter(action='start', type='act',
  name='rapor', dependencies=['veri-toplama']) çağır, sonra visualize_data,
  sonra delimiter(action='end').
```

İki tetikleyici cümle:

| Tetikleyici (prompt) | Ne açar |
|---|---|
| **"Çok veri toplayacaksan"** | `expl` episode (keşif) |
| **"Grafik/rapor üretmeden önce"** | `act` episode (eylem) |

Gerisi LLM'in yargısı: "çok veri" ne kadar çok, "rapor" ne sayılır — bunları model kendi yorumlar. Yani kıstas, prompt'un verdiği sezgi + modelin o an ki kararıdır.

---

<a name="s3"></a>
## 3. Altındaki gerçek ayrım: yan etki (expl vs act)

Prompt bir sezgi verir, ama arkasındaki kavramsal kıstas (Beyond Compaction, arXiv 2606.11213) tek soruya iner: **bu grup yan etki üretiyor mu?**

| Kıstas | `expl` (keşif) | `act` (eylem) |
|---|---|---|
| Ne yapar | bilgi **toplar** | topladığı bilgiyle bir şey **yapar** |
| Yan etki | yok (okuma/arama) | var (yazma/üretim) |
| Ledger kategorisi | `read` / `search` | `write` |
| Örnek (equity) | `get_stock_price`, `get_income_statements` | `visualize_data` |
| Örnek (ürün) | `jira_get_issue`, `confluence_search` | `docx_create`, `pptx_add_slide` |
| Eviction sırası | en son atılır (gerekçe) | önce atılır (etkisi diskte) |

Kritik gözlem: bu ayrım **ledger kategorisiyle örtüşür**. Bir grup `read/search` tool = keşif; sonrasında gelen `write` tool = eylem. Yani "expl mi act mi" sorusunun cevabı çoğu zaman "kategori read/search mı, write mı" sorusunun cevabıyla aynıdır. Ajan bunu prompt'la teyit eder, sistem kategoriyle çapraz doğrular.

**Neden eylem önce, keşif sonra atılır?** Keşif, eylemin gerekçesidir. "Rapor" hâlâ bağlamda canlıyken onu besleyen "veri-toplama"yı atarsan model raporun neye dayandığını kaybeder — bağlam çöker. Eylemin etkisi zaten diskte (dosya/grafik üretildi); keşif ise sadece bağlamda yaşar, o yüzden en son gider.

---

<a name="s4"></a>
## 4. dependencies: eylem hangi keşfi tüketti

`act` açarken ajan `dependencies=['veri-toplama']` verir. Bu, "bu eylem hangi keşfin çıktısını kullandı" sorusunun cevabıdır ve eviction politikasının çekirdeğidir (`episode_graph.py:83-98`):

> Bir `expl` episode ANCAK ona bağlı TÜM `act`'ler zaten atıldıysa atılabilir.

```
[expl] veri-toplama  (8 okuma)
[act]  rapor  ← dependencies=[veri-toplama]  (1 üretim)

eviction:
  1. Faz 5: act 'rapor' atılır (write, etkisi diskte)
  2. Faz 6: 'rapor'un tüm olayları atıldı → 'veri-toplama' artık atılabilir
            → 8 okuma tek description'a iner
```

Bağımlılık bildirilmezse bu güvenli sıra kaybolur; keşif, gerekçesi hâlâ canlıyken atılabilir hale gelir. Bu yüzden prompt act'te dependency'yi zorunlu kılar.

---

<a name="s5"></a>
## 5. Garanti değil, öneri: üç sonuç

Episode bildirimi bir **garanti değil, öneridir**. Üç sonuç:

**1. Ajan doğru bildirirse** → Faz 6 (bağımlılık-farkında episode eviction) devreye girer, yapısal kazanç gelir. Keşif grupları tek cümleye iner.

**2. Ajan hiç bildirmezse** → sistem **çökmez**. Faz 6 atlanır; Faz 1-5 + 7 (ledger dedup/stale, kategori, boyut) yine çalışır. Ledger otomatik olduğu için ana kazancın çoğu zaten gelir — CWL onun üstüne yapısal bir katmandır, temeli değil. Buna **graceful degradation** denir: en kötü ihtimalle bir optimizasyon kaçar, doğruluk bozulmaz.

**3. Ajan yanlış bildirirse** (ör. act'i expl sanır, ya da yanlış dependency) → kötü grup oluşur, ama `evictable_expl`'in bağımlılık kısıtı yanlış eviction'ı büyük ölçüde engeller: dayanağı hâlâ canlı olan bir expl atılamaz. Yani yanlış bildirim genelde "daha az sıkıştırma"ya yol açar, "yanlış bilgi kaybı"na değil.

Bu üç sonuç, tasarımın **güvenli tarafa düştüğünü** gösterir: CWL katmanı kazanç ekler ama hata durumunda sistemi bozmaz.

---

<a name="s6"></a>
## 6. Neden otomatik episode koymadık

"Bu 8 tool tek bir mantıksal iştir" bilgisi tool adından çıkmaz. `get_stock_price` + `get_income_statements`: iki bağımsız okuma mı, yoksa tek bir "XOM değerleme keşfi" mi? İki durumun tool adları aynı görünür; farkı yalnızca **niyet** belirler.

Ledger **atomik gerçeği** (bu tool read'di, kaynağı XOM, sürüm 0) otomatik çıkarabilir çünkü bu tool adından deterministik. Ama **gruplama ve bağımlılık** niyet gerektirir; onu tahmin etmek kırılgan olurdu (yanlış grup = yanlış eviction). Beyond Compaction'ın ana tezi tam bu: **yapıyı tahmin etme, ajana bildirt.** Ajan görevi yapan taraf olduğu için hangi tool'ların tek işe ait olduğunu en iyi o bilir.

---

<a name="s7"></a>
## 7. Daha güçlü kıstas: açık eşikler

Prompt yönergesi ("çok veri", "rapor") modelin yorumuna açık. Üretimde daha kesin, daha az yoruma dayalı kurallara bağlanabilir:

```
- 3+ okuma/arama tool'unu art arda çağıracaksan ÖNCE
  delimiter(start, expl, <konu>) aç; bitince delimiter(end, description=...).
- HERHANGİ bir üretim/yazma tool'undan (visualize_data, docx_*, pdf_*,
  pptx_*, analysis_render_chart) ÖNCE delimiter(start, act, <isim>,
  dependencies=[<önceki expl>]) aç; tamamlayınca delimiter(end).
```

Bu, kıstası "modelin yargısı"ndan "açık sayısal/kategorik eşik"e kaydırır:
- **expl eşiği:** ardışık okuma/arama sayısı ≥ 3
- **act eşiği:** kategorisi `write` olan herhangi bir tool

Sistem bu eşikleri kategori bilgisiyle **çapraz doğrulayabilir** de: ajan bir grubu expl bildirdiği halde içinde write varsa uyarı verebilir. Böylece ajanın yargısına daha az, deterministik sinyale daha çok dayanılır — ama nihai bildirim yine ajanda kalır (niyet gerektiren gruplama için).

---

<a name="s8"></a>
## 8. Özet

- **Kıstas ajanda + prompt'ta**, sistemde zorunlu değil. Ledger otomatik; CWL ajan-bildirimli.
- **Tetikleyici (prompt):** "çok veri toplayacaksan" → expl; "üretim/rapor öncesi" → act.
- **Kavramsal ayrım:** yan etki var mı? Yok → expl (read/search); Var → act (write). Ledger kategorisiyle örtüşür.
- **dependencies:** eylemin tükettiği keşif; "act atılmadan expl atılamaz" kuralını besler.
- **Garanti değil:** bildirilmezse Faz 6 atlanır ama sistem çökmez (graceful degradation); yanlış bildirim genelde az sıkıştırmaya yol açar, bilgi kaybına değil.
- **Neden otomatik değil:** gruplama niyet gerektirir; tool adından çıkmaz. Yapıyı tahmin etme, ajana bildirt.
- **Güçlendirme:** "3+ okuma" ve "write öncesi" gibi açık eşiklerle yoruma daha az yer bırakılabilir; sistem kategoriyle çapraz doğrular.

---

*Bu ek `poc-trace-compaction/` (chat.py EQUITY_SYSTEM, episode_graph.py, compactor.py Faz 6) Ağustos 2026 durumuna dayanır.*
