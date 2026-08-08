# 13-Ek — Tool Trace Compaction: Sıfırdan, El Ele (Babysitter Modu)

**Ağustos 2026 · Hiçbir ön bilgi varsaymaz · Her terim açıklanır**

Bu doküman, tool trace compaction'ı **hiç bilmeyen birine** anlatır gibi yazıldı. Yavaş gider, her kelimeyi tanımlar, her adımı numaralar, analoji kullanır. Acele etme; sırayla oku. Kod referansları `poc-trace-compaction/` klasörüne göredir.

---

## 0. En baştan: birkaç kelime ne demek

Devam etmeden önce 4 kelimeyi oturtalım. Bunları bilmeden gerisi havada kalır.

- **LLM (model):** Metin üreten yapay zekâ (bizde Gemma). Ona bir soru + bağlam verirsin, o cevap üretir.
- **Tool (araç):** Modelin çağırabildiği bir fonksiyon. Örn. `jira_get_issue(key="ATLAS-101")` → bir Jira kaydının detayını döndürür. Model kendisi veri bilmez; **tool çağırıp** öğrenir.
- **Bağlam (context):** Modele her seferinde gönderdiğin metnin tamamı — sistem talimatı + kullanıcı mesajları + modelin düşünceleri + **tool çıktıları**. Modelin "hafızası" budur; ama sınırlı (belli bir token sayısına kadar).
- **Token:** Metnin ölçü birimi. Kabaca ~4 karakter = 1 token. "Bağlam doluyor" demek "token sayısı sınıra yaklaşıyor" demek.

Tuttuysan devam. Tutmadıysan tekrar oku — gerisi bunlara dayanıyor.

---

## 1. Problem: neden bir şey yapmamız gerekiyor

Bir örnek düşün. Kullanıcı "Atlas projesi hakkında rapor hazırla" diyor. Model şunları yapar:

```
1. jira_resolve_project("Atlas")   → hangi proje bu? → ATLAS
2. jira_get_issue("ATLAS-101")     → 200 satır detay döner
3. jira_get_issue("ATLAS-102")     → 200 satır detay döner
4. jira_aggregate(...)             → "toplam 47 iş" döner
5. docx_create(...)                → Word belgesi oluştur
... 15 tool daha ...
```

Her tool'un çıktısı **bağlama eklenir ve orada kalır**. 20 tool çağrılınca bağlam, 20 tool çıktısının hepsini taşır. Sonunda:

- Bağlam dolar → model yavaşlar, pahalılaşır, en sonunda **taşar** (sınırı aşınca çalışmaz).
- Çıktıların çoğu **artık gereksizdir** — ama orada durup yer kaplar.

**İşte çözmemiz gereken problem bu:** Biriken tool çıktılarını, **gerekeni kaybetmeden**, küçültmek. Buna **tool trace compaction** diyoruz.

> **"trace" ne demek?** Modelin çağırdığı tool'ların **sırayla dizilmiş geçmişi**. "Şunu çağırdım, şu döndü; sonra şunu çağırdım, şu döndü..." zinciri. Türkçesi "iz" — ajanın bıraktığı ayak izleri.

---

## 2. Kilit fikir: çıktıları ATMIYORUZ, üç kadere ayırıyoruz

Naif çözüm: "eski çıktıları sil." Ama bu **çalışmaz** ve nedenini bilmen önemli.

Model API'sinde her tool çağrısı ile çıktısı **birbirine bağlıdır** (`tool_call_id` denen bir kimlikle). Çıktıyı listeden silersen bu bağ kopar ve API **hata verir (400)**. Yani "sil" seçeneği yok.

Bunun yerine her tool çıktısını **üç kaderden birine** koyarız — ama üçünde de çıktı **yerinde kalır**, sadece içeriği değişir:

| Kader | Ne olur | Örnek |
|---|---|---|
| **TAM** | çıktı olduğu gibi durur | Şu an gereken taze veri |
| **ÖZET** | çıktı 5-satırlık bir karta iner | Eski ama izi lazım olan çıktı |
| **SİL** | çıktı tek satır nota iner | Aynısı başka yerde zaten duruyorsa |

Not: "SİL" dediğimiz bile gerçek silme değil — çıktının yerine "[silindi] şuraya bak" notu koyarız. Bağ (`tool_call_id`) hep korunur. **Silme yok; sadece küçültme var.**

---

## 3. Nasıl karar veriyoruz: üç defter tutuyoruz

Bir tool çalıştığında, çıktısını **üç ayrı deftere** yazarız. Her defterin işi farklı. Bunları bir muhasebeci gibi düşün — her hareketi kaydediyor ki sonra "bu hâlâ geçerli mi?" diye bakabilelim.

### Defter 1 — Trace (ham günlük)
Sadece "ne oldu"yu sırayla tutar: hangi tool, hangi argüman, ne döndü. Ham metin. (`trace.py`)

### Defter 2 — Ledger (akıllı muhasebe defteri)
Her okumanın bir **fişini** keser: "hangi kaynağı, hangi anda, hangi sürümde okudum." (`ledger.py`) Bu fiş sayesinde sonra şunu sorabiliriz:
- **Bu okuma bayat mı?** (kaynak sonradan değişti mi?)
- **Bu çağrı tekrar mı?** (aynı şeyi daha önce çektik mi?)

Muhasebeci analojisi: "XOM fiyatını 2. dakikada gördüm, sürüm 0. 6. dakikada biri güncelledi, artık sürüm 1. Elimdeki fiyat notu artık **eski**."

### Defter 3 — CWL Episode (ajanın iş defteri)
Modelin kendisinin doldurduğu defter: "şu 8 tool tek bir 'veri toplama' işiydi; sonraki 'rapor' işi buna dayanıyor." (`episode_graph.py`) Model bunu `delimiter` denen özel bir tool'la bildirir (bkz. §6).

**Neden üç defter?** Çünkü üç ayrı soruya cevap veriyorlar:
- Trace: "ne oldu?"
- Ledger: "bu tek okuma hâlâ geçerli mi?"
- CWL: "bu tool grubu güvenle atılabilir mi?"

---

## 4. Ledger'ın kalbi: kategori + kaynak + sürüm

Ledger'ın nasıl "bayat mı / tekrar mı" bildiğini yavaş açalım — çünkü sistemin özü burada.

### 4a. Her tool bir KATEGORİye ait
Ledger, tool'un ne yaptığına göre onu sınıflar:
- **read (okuma):** durum getirir — `jira_get_issue`. Bu bayatlayabilir.
- **search (arama):** sorguya bağlı getirir — `jira_search_issues`. Sorgu aynıysa tekrar.
- **write (yazma):** durumu değiştirir — `docx_add_chart`. Başkalarını bayatlatır.

### 4b. Her okumanın bir KAYNAĞI var
Kaynak = tool'un dokunduğu varlığın kimliği. `jira_get_issue`'nun kaynağı `key` (ATLAS-101); `docx_add_chart`'ın kaynağı `document_id`.

### 4c. Her kaynağın bir SÜRÜM sayacı var
Bir kaynağa **yazma** yapılınca, o kaynağın sürüm sayacı bir artar. Okuma fişi ise okuma anındaki sürümü **donduarak** saklar.

### Şimdi sihir: "bayat mı?" iki basit karşılaştırma
```
Okuma fişi diyor: "ATLAS-101'i sürüm 0'da okudum."
Şu anki sürüm:    1  (arada biri yazdı)
0 < 1  →  BAYAT.  Elimdeki not eski.
```
Ya da zaman geçmesiyle (bazı kaynaklar — fiyat, outline — zamanla bozulur, buna `ttl` denir):
```
Fişi: "adım 2'de okudum, ttl=1"
Şimdi: adım 10
10 - 2 = 8 > 1  →  BAYAT.  Çok bekledi.
```

Ve "tekrar mı?": *"Aynı kaynağı, aynı sürümde daha önce okumuş muydum?"* Evetse bu ikinci okuma gereksiz — tekrar.

**Kritik:** Bunların hiçbiri modele sormaz, metin karşılaştırmaz. Sadece **tamsayı karşılaştırması**. Deterministik, bedava, tekrarlanabilir. (§7'de bunun neden önemli olduğunu göreceğiz.)

---

## 5. Somut örnek: read → write → read

En sık kafa karıştıran senaryoyu adım adım yürüyelim. Bir Word belgesi düzenlediğini düşün:

```
Adım 1:  docx_get_outline(belge)   → "belgede 1 blok var" (sürüm 1'i okudu)   [FİŞ kesildi]
Adım 2:  docx_add_chart(belge)     → grafik ekledi (belgeye YAZDI → sürüm 2)  [SÜRÜM arttı]
Adım 3:  docx_get_outline(belge)   → "belgede 2 blok var" (sürüm 2'yi okudu)  [YENİ fiş]
```

Şimdi compaction "Adım 1'in outline'ı hâlâ geçerli mi?" diye sorar:
- Adım 1 sürüm 1'i gördü. Şu anki sürüm 2 (Adım 2 yazdı). `1 < 2` → **BAYAT**.
- Adım 3 güncel outline'ı canlı tutuyor (sürüm 2).
- Karar: Adım 1'in outline'ı → **SİL** (güncel kopyası zaten canlı, güvenli).

Gördün mü? Yazma (Adım 2), kendinden **önceki** okumayı otomatik geçersizleştirdi. Bu, sistemin en güzel çalıştığı yer — ve ürün tool'larında canlı çalıştığını kanıtladık:

```
KANIT (gerçek çıktı): docx_get_outline(v1) → add_chart YAZDI → is_stale=True (mutasyon)
```

---

## 6. delimiter: modelin "bu tool'lar bir grup" demesi

Ledger tek tek tool'ları izler. Ama bazen "şu 8 tool **birlikte** bir işti" bilgisi lazım. Bunu tool adından çıkaramayız (niyet gerektirir). O yüzden **modelin kendisi bildirir** — `delimiter` denen özel bir tool'la.

`delimiter` veri işlemez; sadece **sınır işareti** koyar:
```
delimiter(start, type="expl", name="veri-toplama")   ← "keşif başlıyor"
  jira_get_issue(...)   ┐
  jira_get_issue(...)   ├─ hepsi "veri-toplama" grubuna ait
  jira_aggregate(...)   ┘
delimiter(end, description="ATLAS verisi toplandı")   ← "bitti, öğrendiğim bu"
```

İki grup tipi:
- **expl (keşif):** bilgi toplayan tool'lar (okuma/arama). Yan etkisi yok.
- **act (eylem):** bir şey üreten tool'lar (yazma). "Hangi keşfe dayandığını" da bildirir.

**Neden lazım?** Compaction şunu bilir: "rapor (act) hâlâ bağlamda gerekliyse, onu besleyen veri-toplama'yı (expl) atma — yoksa model raporun neye dayandığını unutur." Yani **eylem önce atılır, keşif sonra** (ve ancak dayanağı kalmayınca). Bir keşif grubu atılınca da, ajanın kapanışta yazdığı **tek cümleye** iner (8 tool → 1 cümle).

Canlı testte gördük: Gemma `delimiter`'ı **kendisi çağırdı** (system prompt'ta "üretim öncesi delimiter çağır" yazdığı için) ve `[act] rapor-uretimi` grubunu kurdu.

---

## 7. Compaction ne zaman ve nasıl çalışır

### Ne zaman? — İki eşik
- **budget (bütçe):** Bu token sınırını aşınca compaction **başlar**. Altındaysa hiç dokunmaz (yer varken sıkıştırmak gereksiz).
- **target (hedef):** Compaction, buraya kadar **iner** sonra durur. (budget'ın ~yarısı.)

İki eşik olmasının sebebi: bir kez sıkıştırıp durunca, hemen tekrar tetiklenmesin (testere gibi inip çıkmasın). Buna histerezis denir.

### Nasıl? — 7 faz, ucuzdan pahalıya
Compaction başlayınca sırayla 7 aşama çalışır. Mantık: **en güvenli ve en ucuz olanı önce.**

| Faz | Ne yapar | Hangi defter |
|---|---|---|
| 1 | **Tekrarları** at (dedup) | Ledger |
| 2 | **Bayatları** temizle (staleness) | Ledger |
| 3 | Hata-düzeltme zincirini katla (hata mesajı korunur) | Trace |
| 4 | Ardışık **keşif dizisini** tek bulguya indir | Ledger kategori |
| 5 | Kademeli: önce **eylem**, sonra **keşif** at | Ledger kategori |
| 6 | **CWL episode**'ları bağımlılık sırasına göre at | CWL |
| 7 | Acil: en büyük çıktıyı önce at | Trace |

Faz 1-2 her zaman çalışır (bedava kazanç). Faz 4-7 sadece hâlâ hedefin üstündeysek.

### İki güvenlik freni
1. **Koruma penceresi:** Son N tool **asla** sıkıştırılmaz. Bu turda gereken taze veri hep tam kalır.
2. **Fayda freni:** Özet, ham'dan büyük olacaksa sıkıştırma **yapılmaz** (küçülteceğine büyütürdü — saçma olurdu).

---

## 8. Özet nasıl görünür — "silme yok"un somut hali

Bir tool ÖZET'e inince, çıktısı 5 alana iner. Her alan bir kaybı önler:

**KANIT (gerçek üretim, bayat outline):**
```
[özet] niyet: docx_get_outline çağrıldı · girdi: document_id=doc_fe96de ·
       sonuç: Outline doc_fe96de (v1, 1 blok):… · durum: ok · etki: bayat (eskidi)
ham: 141 token → özet: 40 token
```

- **niyet:** neden çağırdım (modelin o anki düşüncesinden geri alınır)
- **girdi:** hangi kaynağa baktım
- **sonuç:** ne döndü (kısaltılmış; ama sayı/bütçe gibi kritikse **birebir** — buna "verbatim" denir)
- **durum:** başarılı mıydı
- **etki:** neden sıkıştırıldı, izin nereye gitti

SİL ise daha da kısa: `[silindi] tekrar ≡ seq=3 (aynı içerik canlı)`.

**Bu özet hiç LLM kullanmaz.** Sadece argümanları alıp çıktının ilk satırını kesip ledger'ın sebebini ekleyen bir şablon. Neden? Çünkü özet için model çağırmak hem para hem gecikme; deterministik şablon bedava ve tekrarlanabilir.

---

## 9. Bu, LLM'e nasıl ulaşır — köprü

Buraya kadar "trace defterinde" karar verdik. Ama asıl önemli olan: **bu karar modele gerçekten gidiyor mu?**

Evet. `_render_messages` (`agent.py`) fonksiyonu, modele mesajları göndermeden önce her tool çıktısının içeriğini kaderine göre yeniden yazar:
- SİL olan → tek satır not
- ÖZET olan → 5-alan kart
- TAM olan → dokunulmaz

Ve kritik: `tool_call_id` **aynı kalır** → API 400 riski yok.

**KANIT (gerçek çıktı — köprünün somut hali):**
```
tool_call_id=call_002  (AYNI korunur → API 400 yok)
  ÖNCE (ham, 437 karakter): Issue ATLAS-101: ...
  SONRA (compact):          [özet] niyet: [jira-veri] keşif episode'u · girdi: ...
ham messages: 683 tok → render: 602 tok
```

Yani ölçtüğümüz kazanç, modelin **gerçekten gördüğü** bağlama yansıyor. Bu, sistemin en kritik parçasıydı ve tamamlandı.

---

## 10. Ürün tool'ları: 119 gerçek tool

Tüm bunları ürünün gerçek envanterine (`toolsmockproduct/*.yaml`) uyguladık. 9 toolkit, 119 tool:

| Toolkit | Ne | Örnek davranış |
|---|---|---|
| Jira | yazılım iş takibi | issue ttl=20 (gün içinde değişir) |
| NETA | proje bütçe/portföy | verbatim (sayı), ttl=None (stabil) |
| LDAP | kurumsal dizin | ttl=None (neredeyse hiç değişmez) |
| Confluence | wiki | ttl=25 (sayfa düzenlenir) |
| docx/pdf/pptx/xlsx | doküman üretimi | outline ttl=1 (her düzenlemede bayat) |
| analysis | veri/SQL | run_sql okuma, load yazma |

Her tool'a `{cat, resource, ttl, verbatim}` sözleşmesi yazdık. **Önemli dürüstlük:** Bu sözleşmeyi biz çıkardık (YAML'da yoktu) — tool adının fiilinden (`get→okuma`, `add→yazma`) + birkaç domain kuralıyla. Gerçek üründe doğrusu, tool yazarının bunu **beyan etmesi**.

---

## 11. Çalıştığının kanıtı

| Test | Ne kanıtlar | Sonuç |
|---|---|---|
| `test_deterministic.py` | çekirdek mekanizma (26 test) | **26/26** |
| `test_product.py` | köprü + dedup + staleness + CWL | **6/6** |
| `run_equity.py` | tekrarlı senaryoda kazanç | **%67** |
| `run_product.py` | 119 ürün tool'u uçtan uca | çalışıyor |
| canlı Gemma | gerçek model, gerçek sorular | keşif katlama + graceful + verbatim korundu |

**Canlıdan öğrendiğimiz dürüst gerçek:** Yetenekli bir model (Gemma) aynı çağrıyı **kendisi tekrarlamaz**, o yüzden dedup nadir tetiklenir — asıl iş gören **keşif katlama** + **staleness**. Dedup, zayıf modeller / gerçek yeniden-okumalar için bir **güvenlik ağı** (deterministik testte kanıtlı).

---

## 12. Kapsam: neyi yapıyoruz, neyi YAPMIYORUZ

Bu netlik önemli — sistem "her şeyi ezen" değil:

**Yapıyoruz (bizim işimiz):**
- Tool **çıktılarını** sıkıştırmak (dedup, staleness, keşif katlama, CWL episode).

**Yapmıyoruz (kapsam dışı):**
- **Reasoning (modelin düşünce metni) sıkıştırma** — bu tool trace değil, ayrı bir iş.
- **Verbatim veriyi ezme** — bütçe/issue gibi kritik sayılar bilerek **tam** tutulur.
- **Tool tanımlarını (şema) bağlama sığdırma** — bu "girdi ekseni", ürünün tool-manager'ı (retrieval) zaten çözüyor.

**Dürüst sayı:** Kazanç, verbatim-ağırlıklı kısa senaryoda %13-15, tekrarlı uzun oturumda %67. Az görünen sayı bir kusur değil — sistem gerekli veriyi koruyor. Kanıt: model, çıktılar özetlendikten sonra bile doğru sentez üretti (bilgi kaybı yok).

---

## 13. Nasıl çalıştırırsın

```bash
cd poc-trace-compaction

python run_product.py            # LLM'siz uçtan uca demo (119 ürün tool'u)
python test_product.py           # köprü testi (6/6)
python test_deterministic.py     # çekirdek testler (26/26)
python chat_server.py --product  # canlı tarayıcı chatbot (gerçek Gemma)
python live_product_test.py      # canlı çok-turlu test
```

Dosya haritası:
- `ledger.py` — Defter 2 (kaynak/sürüm/kategori/ttl)
- `episode_graph.py` — Defter 3 (CWL episode)
- `trace.py` — Defter 1 (ham günlük) + 5-alan özet
- `compactor.py` — 7 faz + güvenlik frenleri
- `agent.py` — ajan döngüsü + messages[] köprüsü
- `product_tools.py` — 119 ürün tool'u (SCHEMAS + DISPATCH + TOOL_META)

---

## 14. Tek nefeste özet (her şeyi tek paragrafta)

Bir ajan çok tool çağırınca çıktılar bağlamda birikir ve taşar; biz bu çıktıları, gerekeni kaybetmeden küçültürüz. Her tool çağrısını üç deftere yazarız — ham günlük (**trace**), akıllı muhasebe (**ledger**: kaynak/sürüm/ttl), ve ajanın bildirdiği gruplar (**CWL episode**). Bütçe aşılınca 7 fazlı bir hat çalışır: önce ucuz/güvenli (tekrarları at, bayatları temizle), sonra yapısal (keşif dizilerini katla, episode'ları bağımlılık sırasına göre at). Her tool ya **TAM** kalır (gerekli veri), ya 5-alan **ÖZET**'e iner, ya **SİL** notuna — ama hiçbiri listeden çıkmaz (`tool_call_id` korunur, API 400 yok). Kararların hepsi **tamsayı karşılaştırması, sıfır LLM**. Sıkışık hal `messages[]`'e gerçekten yazılır, yani modelin gördüğü bağlam gerçekten küçülür. Bunu ürünün **119 gerçek tool'una** uyguladık, deterministik testler (26/26 + 6/6) ve canlı Gemma ile doğruladık. Sistem "her şeyi ezen" değil, "gereksizi atıp gerekeni koruyan" — bu yüzden kazanç dürüst (kısa senaryoda %13, tekrarlı oturumda %67) ve hiçbir gerekli bilgi kaybolmaz.

---

*Bu babysitter dokümanı `poc-trace-compaction/` (Ağustos 2026) kodundan üretilen gerçek çıktılara dayanır. Daha teknik derinlik için: `13-ek-mekanizma-ledger-cwl-fazlar.md`, `13-ek-episode-kistasi.md`, `13-ek-urun-toollari-yasam-dongusu.md`.*
