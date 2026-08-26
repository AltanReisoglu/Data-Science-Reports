# Eklenen Kaynakların Sentezi

Okunan: 9 dosya (~307 KB), 2026-08-21. Hepsi baştan sona okundu.

Bu dosya üç şey yapıyor: (1) her kaynağın ne olduğu ve içinde **ne benzersiz** olduğu,
(2) mevcut envantere göre **yeni** olan kavramlar, (3) PoC'de **açık kalan** noktalar.

---

## 0. Dokuz kaynak, bir cümlede

| Dosya | Ne | Türü |
|---|---|---|
| `unutma_case_bu.md` | Görev tanımının kendisi | brief |
| `galileo_ai_video.md` | Galileo demo transkripti — sessiz retry döngüsü | gözlemlenebilirlik |
| `pi_anti_doom_loop.md` | `pi` için yayınlanmış npm eklentisi, 6 tespit sinyali | **çalışan ürün** |
| `arize_research_control_loop.md` | "Agent control loop nedir" — kavramsal çerçeve | kavram |
| `loop_budget_source.md` | 5 döngü koruması + çalışan `LoopGuard` kodu (TR) | kavram + kod |
| `loop_budget_medium2.md` | Modexa "Ajan Döngüsü Problemi" makalesi (TR) + aynı kod | kavram |
| `agentbudget_framework.md` | AgentBudget — süreç-içi dolar tavanı SDK'sı | **çalışan ürün** |
| `claude_platform_budget_control.md` | Claude Task Budgets resmî dokümanı (beta) | **resmî API** |
| `sde_offer_loop.md` | "Loop Engineering" — döngü tasarımı disiplini | kavram |

---

## 1. Envantere göre YENİ olan altı kavram

Bunlar `loop_budget.md` ve `harness_kontrolleri.md`'de hiç yoktu.

### 1.1 Model tarafında öz-düzenleme — dördüncü zorlama katmanı

`claude_platform_budget_control.md` · Claude Task Budgets, beta başlığı `task-budgets-2026-03-13`

Token Budgets makalesi üç katman tanımlıyordu: derleme zamanı / yazılım (runtime middleware) /
transport (HTTP 402). Bu **dördüncüsü**: modelin kendisi.

```json
"output_config": { "task_budget": {"type": "tokens", "total": 64000} }
```

Sunucu, konuşmaya modelin gördüğü bir **geri sayım işareti** enjekte ediyor. Model bunu
görüp kendini ayarlıyor, bütçe azaldıkça işi toparlıyor.

Kritik ayrıntılar:
- **Tavsiye niteliğinde, zorlayıcı değil.** Model, kesmenin bitirmekten daha bozucu olacağı
  bir eylemin ortasındaysa bütçeyi aşabiliyor. Sert tavan hâlâ `max_tokens`.
- **Geri sayım yalnızca modele görünüyor.** API cevabında kalan bütçe alanı YOK, SDK'da
  erişimci YOK. İstemci tarafında izlemek istiyorsan kendin toplayacaksın.
- Sayaç, **modelin o turda gördüğünü** düşüyor — istemcinin yeniden gönderdiği geçmişi değil.
  Dokümandaki örnek: 3 turda istemci ~20.820 token göndermiş, bütçeden 19.000 düşmüş.
- **Çok küçük bütçe reddetme benzeri davranış üretiyor.** Model işi hiç denemeyebiliyor,
  agresif biçimde daraltabiliyor ya da erken duruyor. Dokümanın kendi uyarısı: *"beklenmedik
  reddetme veya erken durma görüyorsan, başka parametreleri ayıklamadan önce bütçeyi yükselt."*
- Minimum `total` = 20.000 token; altı 400 dönüyor.
- **Prompt caching çakışması:** bütçe değeri render edilen prompt'a giriyor, `remaining`'i her
  turda düşürürsen cache önekini geçersiz kılıyorsun. Öneri: bütçeyi bir kez kur, modelin
  sunucu tarafı geri sayıma karşı kendini ayarlamasına izin ver.
- Compaction yapıyorsan `remaining` ile bütçeyi taşı, yoksa sunucu sıfırdan sayar.
- Claude Code ve Cowork'te **desteklenmiyor**; Messages API üzerinden kullanılıyor.
- `effort` derinliği, `task_budget` genişliği ayarlıyor — tamamlayıcı.

### 1.2 Süreç-içi dolar tavanı — gateway'e alternatif

`agentbudget_framework.md` · Python/Go/TS SDK, Apache 2.0

LiteLLM gateway katmanında duruyordu. Bu **süreç içinde**, istemciyi sararak çalışıyor:
`agentbudget.init("$5.00")` iki satır, kod değişikliği yok. Kendi tanımıyla *"Not an LLM proxy"*.

Bizde karşılığı olmayan üç mekanizma:

- **`finalization_reserve`** — bütçenin bir kısmını nihai cevap adımına ayırıyor. `max_spend="$1.00"`,
  `finalization_reserve=0.05` → sert limit 0,95 $'da tetikleniyor, son 0,05 $ serbest kalıyor.
  Ajan görevin ortasında kesilmiyor. Ayrıca `would_exceed(tahmini_maliyet)` ile son çağrıdan
  önce kontrol.
- **İç içe bütçeler** — `parent.child_session(max_spend=2.0)`, maliyetler yukarı toplanıyor.
  Çok-agent'ta bütçe bölüştürme probleminin somut çözümü.
- **Zaman pencereli döngü tespiti** — `max_repeated_calls=10` **`loop_window_seconds=60`** ile
  birlikte. Sayı değil, birim zamandaki sayı.

Üç kademeli devre kesici: soft limit (callback, uyarı) → hard limit (`BudgetExhausted`) →
loop detected. Rapor `terminated_by` alanı taşıyor: `null` / `"budget_exhausted"` / `"loop_detected"`.

Sorun tanımı da alıntılanabilir: *"Sıkışmış bir ajan 10 dakikada 200 LLM çağrısı yapar.
Kimse fark etmeden 50–200 $."* Ve ölçek: *"%5 hata oranıyla 1.000 eşzamanlı oturum = 50 kaçak ajan."*

### 1.3 Yakın-benzer metin tespiti — gömme kullanmadan

`pi_anti_doom_loop.md` · yayınlanmış npm eklentisi

PoC'nin bilinen sınırlarından biri "anlamsal denklik yakalanmıyor" idi ve çözümü gömme
sanıyordum. `pi` bunu **token örtüşme oranıyla** çözüyor: ardışık mesajlar **≥%55 token
paylaşıyorsa** model aynı adımı yeniden ifade ediyor demektir. Gömme yok, model çağrısı yok.

Altı sinyalin tamamı (varsayılan eşik hepsinde 3, pencere 10):
1. Aynı `(araç, argüman)` tekrarı
2. Aynı aracın ardışık hata vermesi
3. **Aynı asistan metninin birebir tekrarı**
4. **TEK bir mesajın içinde aynı cümlenin 3+ kez geçmesi** (büyüyen kendi-kendine-ekleme döngüsü)
5. **Yakın-benzer metin** (≥%55 token benzerliği, ardışık)
6. **Yakın-benzer metin çevrimi** — dönüşümlü yeniden ifade edilmiş komutlar; ne birebir aynı
   ne de ardışık, ama pencerede eşiğe ulaşıyor

Ayrıca bizde olmayan üç yapılandırma:
- `PI_ANTI_LOOP_TIME_WINDOW` — zaman penceresi; **uzun oturumdaki yavaş kronik döngüleri**
  yakalamak için eski girdileri düşürüyor
- `PI_ANTI_LOOP_FAIL_RATE` — bir aracın penceredeki hata **oranı** eşiği (sayı değil oran)
- `PI_ANTI_LOOP_TOOLS_EXCLUDE` — kaçış kapısı; belirli araçlar hiç izlenmiyor

**Üç kademeli müdahale** (bizde iki kademe var): steer (yönlendir, koşum sürsün) → abort +
**bir** taze devam direktifi kuyruğa al → gerçekten dur. Otomatik devam bütçesi tavanlı,
yani sıkışmış model sonsuza dönemiyor.

İki tasarım ayrıntısı doğrudan kopyalanabilir:
- **Boşa giden token sayımı** — dedektör tekrarlarda yakılan token'ı tahmin edip (~4 karakter/token)
  engelleme gerekçesinde raporluyor: *"~N tokens burned on repeats."* Döngünün durdurulmadan
  önce ne kadara mal olduğu görünür oluyor.
- **Eşikler minimum 2'ye kırpılıyor** ki bozuk bir yapılandırma ajanı kilitlemesin.

### 1.4 Doğrulama kapılı durma — PoC'nin en büyük açığı

`sde_offer_loop.md` · "Loop Engineering"

Dört durma sinyali sıralıyor ve aralarında bir hiyerarşi kuruyor:
1. Model "bitti" diyor — **gerekli ama yeterli değil**
2. **Doğrulama kapılı** — testler geçti, build başarılı, çıktı şemaya uyuyor. En güçlü sinyal,
   çünkü modelin görüşüne değil ortama dayanıyor
3. Hedef/çıkış kriteri — açık bir yüklem
4. Bütçe tükenmesi — hedef değil, son çare

Vurgusu: *"Modelin 'bitti' demesi durma **isteğidir**, tamamlanma **kanıtı** değildir."*
Ve: *"'Bitirdim dedi', ajan dünyasının 'benim makinemde derleniyor'udur."*

Ayrıca: *"Zayıf bir model, sıkı bir doğrula-ve-yeniden-dene döngüsüyle, doğrulaması olmayan
güçlü bir modeli çoğu zaman geçer."*

### 1.5 Sessiz retry döngüsü — sonunda başarılı olan hata

`galileo_ai_video.md`

PoC'deki senaryoların hepsi ya durmuyor ya başarısız oluyor. Bu farklı: **retry sonunda
başarılı oluyor**, kullanıcı doğru cevabı alıyor, dışarıdan hiçbir şey yanlış görünmüyor.

> *"Bu, hakkında hiç ticket açılmayan türden bir hata. Sessiz ama öldürücü."*

Demo: iki analist ajan bir hisseyi puanlıyor. Temiz koşumda sorun yok. İkinci koşumda araç
çağrısı tutarsız — ama analistler yine uzlaşıya varıyor, **aynı format, aynı güven**.
Konsolda bakınca: araç hata oranı sıçramış, ana span'de bir başarıdan önce iki başarısız
araç çağrısı var.

Teşhis: *"İki tekrar önemsiz görünebilir ama önemli olan desen. Bu retry mantığında ne
devre kesici ne zaman aşımı koruması var. Bugünkü geçici hata, yarın sürekli bir kesinti
olur ve her istek maksimum retry'ı yakarak tüm alt akışı tıkar."*

Ve suçun yeri: *"Ajanlar doğru davrandı. Zafiyet veri çekme katmanında."*

### 1.6 Dış iyileştirme döngüsü — eşikleri kim ayarlayacak

`agent_improvement_loop.md` · OpenAI cookbook

İç döngünün (koşum) dışındaki döngü: **izler → insan ve model geri bildirimi → otomatik
üretilen eval takımı → doğrulama kapısı → optimizasyon → harness değişikliği için Codex'e devir.**
Tek bir dosya işi taşıyor: `codex_handoff.md`.

Bizim için asıl değeri: eşiklerin nasıl ayarlanacağı sorusuna kurumsal bir cevap.
`EVAL_METADATA` bir harness sürümünü `version` / `status: promoted` / `promotion_gate` ile
işaretliyor — yani **döngü yapılandırması prompt'la birlikte sürümleniyor ve terfi kapısından
geçiyor.** Arize'ın "loop configuration gets versioned alongside the prompt" cümlesinin
çalışan hâli.

---

## 2. Birden fazla kaynağın bağımsız olarak söylediği yedi şey

Bunlar en güçlü sinyaller — farklı yazarlar, aynı sonuç.

**1 · Hareket etmek ilerlemek değildir.** Modexa: *"Ajan sürekli hareket halindedir ancak
sistem ilerlemiyordur."* Arize aynı fikri "plan thrash" olarak adlandırıyor ve teşhisi keskin:
*"bu, akıl yürütme kılığına girmiş bir bağlam kurma hatasıdır."* `loop_budget_source` üçüncü
döngü sebebi olarak sayıyor: *"Ajan, hareket etmeyi ilerleme ile karıştırır."*

**2 · Eşikleri ölçerek seç, tahmin ederek değil.** Üç kaynak, üç ayrı formülasyon:
- Arize: *"Başarılı görevlerin adım sayısı dağılımına bak, tavanı kuyruğunun üstüne koy.
  Meşru işi kesecek kadar dar bir adım limiti, modelin kötüleştiği gibi görünen sessiz bir
  kalite gerilemesine dönüşür."*
- Claude dokümanı: bütçesiz temsili bir örneklem koştur, dağılımı kaydet, **p99'dan başla**.
- `sde_offer_loop`: *"Döngüyü sezgiyle ayarlayamazsın"* — eval'lerle ölç, tek şeyi değiştir,
  yeniden ölç.

**3 · Modeli "bitti mi" sorusunun tek yargıcı yapma.** Arize: *"her döngüye modelin yargısına
bağlı olmayan sert bir durdurma gerekiyor, çünkü model, bitip bitmediği konusunda yanılması
en muhtemel bileşendir."* Modexa: *"durdurma modelin dışında zorunlu kılınmalıdır."*
`sde_offer_loop`: doğrulama kapısı.

**4 · Durma sebebini kaydet.** Arize: *"Her koşumda bir durma sebebi kaydedin. `completed`,
`max_steps`, `budget_exceeded` ve `error` farklı sonuçlardır; ortalamalarını almak neyin
yanlış gittiğini gizler."* AgentBudget'ta bu `terminated_by` alanı. PoC'deki `Status` enum'u
aynı fikrin karşılığı — bağımsız olarak doğrulanmış oldu.

**5 · Sert kesme değil, zarif iniş.** Dört kaynak:
- `sde_offer_loop`: *"Bütçeler sadece bir kill switch değil; döngünün üzerinde akıl
  yürütebileceği bir sinyaldir"* — adım bütçesi azalınca "2 adımın kaldı, bulduklarını özetle ve dur".
- Claude task budgets: geri sayımın varlık sebebi tam olarak bu.
- AgentBudget: `finalization_reserve`.
- `loop_budget_source`: **"Onurunla dur"** — durma cevabı dört şey içermeli: ne denedi,
  ne buldu, neden durdu (bütçe/ilerleme/izin), en iyi bir sonraki eylem.
  *"Durmak başarısızlık değildir. Durmak kontroldür."*

**6 · "Kullanıcıya sor" birinci sınıf bir sonuç olmalı.** `loop_budget_source` beşinci koruma
olarak sayıyor; Modexa geri dönüş merdiveninin dördüncü basamağı yapıyor ve nasıl yapılacağını
söylüyor: *tek soru sor (beş değil), neyi değiştireceğini açıkla, kullanıcı umursamazsa
varsayılan sun.* Gerekçe: *"40 adım boyunca yanlış tahmin yürütmekten ucuzdur."*

**7 · Parmak izine sonucu da kat.** İki bağımsız kod örneği aynı şeyi yapıyor:
`action_fingerprint(tool, args, outcome)` (`loop_budget_source`) ve
`sig = (action.name, action.args, result.error)` (`sde_offer_loop`). Yani eylem + argüman +
**sonuç** birlikte hash'leniyor.

---

## 3. Tekrarlayan yapısal reçeteler

**Döngü sebepleri taksonomisi** (`loop_budget_source` 3 + Modexa 5, örtüşüyor):
araç beklenen yanıtı vermedi · hedef yeterince belirtilmemiş · hareket/ilerleme karışması ·
açık bir "bitti" tanımı yok · güvenilmez araç + naif retry · belirsiz hedef · her adımda
bozulan bağlam · **ajanın "yanlış yapmamak" üzerine optimize edilmesi**.

Sonuncusu incelikli ve başka yerde geçmiyor: aşırı tedbir teşvik eden bir prompt, en güvenli
yolu "bir kez daha doğrula" yapıyor. *"Bir sorun çözücü inşa etmediniz. Bir riskten kaçınma
makinesi inşa ettiniz."*

**Geri dönüş merdiveni** (Modexa) — ajan kendi kendine tekrar icat etmemeli, merdiveni izlemeli:
1. Bekleme süresiyle bir kez tekrar dene → 2. Aracı/sağlayıcıyı değiştir → 3. Kapsamı daralt →
4. Kullanıcıya açıklayıcı soru sor → 5. Elindeki en iyi cevabı + sonraki adımları döndür.

**Durum makinesi** (Modexa) — serbest formlu döngüyü `ANLA → TOPLA → HAREKETE GEÇ → DOĞRULA →
YANITLA → DEVRET` gibi net durumlara indir, yalnızca belirli geçişlere izin ver.
*"Sadece bu bile çok büyük bir döngü sınıfını ortadan kaldırır."*

**Dört döngü topolojisi** (`sde_offer_loop`): tek döngü · refleksiyon (taslak→eleştiri→revizyon) ·
planla-sonra-uygula · orkestratör + alt-ajanlar. Uyarısı: *"Topolojiyi aşırı mühendislik etme.
Refleksiyon ve çok-ajan model çağrısı, gecikme ve maliyet ekler."*

**Örnek politika sayıları** (birbirine yakın çıkıyor):
`loop_budget_source` — adım ≤12, araç çağrısı ≤8, araç başına deneme ≤2, süre ≤60 sn.
Modexa — araç çağrısı ≤6, mantık adımı ≤10, süre ≤20 sn.
PoC'nin varsayılanı (adım 12) bağımsız olarak aynı aralıkta çıkmış.

---

## 4. PoC'de açık kalan yedi nokta

Sırayla, en önemliden:

1. **Doğrulama kapısı yok.** `Finish` olduğu gibi kabul ediliyor. `sde_offer_loop`'un en sert
   uyarısı tam buraya: model "bitti" dediğinde bu bir istek, kanıt değil. Bir `verify()`
   kancası ve doğrulama başarısız olduğunda gözleme dönüp devam etme yolu eklenebilir.
2. **Araç başına retry limiti yok.** Sadece genel `max_replans` var. İki kaynak da araç
   bazında sayıyor (`max_retries_per_tool=2`). Bir araç bozuksa genel sayacı tek başına yiyor.
3. **Nihai cevap bütçe dışında.** `_request_final_answer()` tavanı aştıktan sonra çağrılıyor.
   AgentBudget'ın `finalization_reserve`'ü daha dürüst: rezerv baştan ayrılıyor, sert limit
   erken tetikleniyor, tavan hiç aşılmıyor.
4. **"Kullanıcıya sor" bir sonuç değil.** `Status` enum'unda `OK/STUCK/BUDGET_EXHAUSTED/DEGRADED`
   var; `NEEDS_INPUT` yok. İki kaynak bunu birinci sınıf sonuç yapılmasını söylüyor.
5. **Zaman penceresi yok.** Tespit sayı tabanlı. `pi` ve AgentBudget ikisi de zaman penceresi
   tutuyor — uzun oturumdaki yavaş kronik döngüler sayı eşiğine hiç ulaşmadan bütçeyi yiyebilir.
6. **Durma cevabı yapılandırılmamış.** Tek bir metin dönüyor. "Onurunla dur" dört alan istiyor:
   ne denendi, ne bulundu, neden durdu, sonraki adım önerisi.
7. **Sessiz başarılı retry senaryosu yok.** Galileo'nun vakası: retry sonunda başarılı oluyor,
   `Status.OK` dönüyor, hiçbir guardrail tetiklenmiyor — ama iki araç çağrısı boşa gitti.
   Yakalamanın yolu hata **oranı** sinyali (`pi`'nin `FAIL_RATE`'i) ya da boşa giden token sayımı.

Ayrıca ucuz bir alternatif: `loop_budget_source`'un `detect_repeat_pattern`'i dönüşümlü
döngüyü **"son 6 eylem ≤2 benzersiz parmak izine düşüyorsa"** kuralıyla yakalıyor —
PoC'nin k=1..12 çevrim taramasından çok daha basit. Karşılaştırmalı olarak test edilebilir.

---

## 5. Sunuma girecek alıntılar

- *"Ajanlar sadece yanıldıkları için başarısız olmazlar. Israrcı oldukları için başarısız
  olurlar. Ve ısrar pahalıdır."* — `loop_budget_source`
- *"Durma koşulları olmayan otonomi, sadece bir yangındır."* — `loop_budget_source`
- *"Model, bitip bitmediği konusunda yanılması en muhtemel bileşendir."* — Arize
- *"Bu, hakkında hiç ticket açılmayan türden bir hata. Sessiz ama öldürücü."* — Galileo
- *"Sıkışmış bir ajan 10 dakikada 200 LLM çağrısı yapar. Kimse fark etmeden 50–200 $."* — AgentBudget
- *"Her ajan döngüsünün tasarlanmış bir çıkışı olmalıdır — başarı, geri dönüş, sorma veya
  devretme. 'Sonsuza kadar dene' değil. 'Umarım model çözer' değil."* — Modexa
- *"Durmak başarısızlık değildir. Durmak kontroldür."* — `loop_budget_source`
