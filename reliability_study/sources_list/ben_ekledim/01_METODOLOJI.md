# `ben_ekledim/` — Her Kaynağın Metodolojisi ve Brief'e Faydası

Hazırlanma: 2026-08-21 · 12 kaynak, hepsi baştan sona okundu.

İki soru soruluyor:
**(1) Bu kaynak iddiasını nereden biliyor?** — yöntem ve kanıt gücü.
**(2) `unutma_case_bu.md`'deki hangi maddeye ne katıyor?** — teslimata dönüşümü.

Brief'in üç teslimatı, aşağıda kısaltmalarıyla anılıyor:

- **T1** — Yöntemlerin çalışma prensibi, tespit mantıkları ve **sınırları**; referans kaynak derlemesi
- **T2** — Kontrollerin devreye girdiğini gösteren çalışan PoC/demo (döngüye giren / limit aşan senaryolar)
- **T3** — Kısa doküman + ekip sunumu: **avantajlar, dezavantajlar**, entegrasyon gereklilikleri

Brief'in bütçe kalemleri: **max steps / replans / tokens / süre**.

---

## Kanıt gücüne göre sıralama

Hepsi eşit değil. Sunumda bir iddiayı savunacaksan üstten başla.

| # | Kaynak | Metodoloji türü | Kanıt ölçeği | Sınıf |
|---|---|---|---|---|
| 1 | when Agents Do Not Stop (IAL-SCAN) | Statik program analizi + büyük ölçekli repo taraması + manuel doğrulama | 6.549 repo, 68 doğrulanmış bulgu, %91,9 kesinlik | **hakemli ölçüm** |
| 2 | real_time_Detection_and_repair | Kontrollü enjeksiyon + gerçek koşum korpusu + ablasyon + dış benchmark | 2.823 bölüm, 25 veri kümesi, 3 framework, 4 model | **hakemli ölçüm** |
| 3 | Inference-Time Budget Control | Sabit bütçe altında karşılaştırmalı benchmark + ablasyon + teorem | 4 veri kümesi × 4 bütçe × 3 omurga | **hakemli ölçüm** |
| 4 | claude_platform_budget_control | Resmî API spesifikasyonu | ürünün kendi davranış sözleşmesi | **birincil doküman** |
| 5 | agentbudget_framework | Açık kaynak ürün README'si | çalışan SDK, 3 dil, yayımlanmış paket | **çalışan ürün** |
| 6 | pi_anti_doom_loop | Açık kaynak ürün README'si + test takımı tanımı | yayımlanmış npm eklentisi, 5 katmanlı test | **çalışan ürün** |
| 7 | agent_improvement_loop | Çalıştırılabilir öğretici (notebook) | uçtan uca koşan pipeline | **çalışan örnek** |
| 8 | arize_research_control_loop | Kavramsal derleme (satıcı araştırma yazısı) | ölçüm yok, damıtılmış pratik | mühendislik yazısı |
| 9 | sde_offer_loop | Kavramsal derleme (eğitim içeriği) | ölçüm yok, tasarım disiplini | mühendislik yazısı |
| 10 | loop_budget_source | Kavramsal + çalışan referans kod | ölçüm yok, `LoopGuard` iskeleti | mühendislik yazısı |
| 11 | loop_budget_medium2 | Kavramsal (Medium makalesi) + aynı kod | ölçüm yok, 1 anlatısal vaka | mühendislik yazısı |
| 12 | galileo_ai_video | Ürün demosu transkripti | **tek vaka**, satıcı anlatısı | pazarlama demosu |

**Uyarı:** 8–12 arası hiçbir kaynak ölçüm sunmuyor. Damıtılmış pratik bilgi taşıyorlar
ve tasarım kararı için iyiler; ama sunumda "şu oran şudur" diye bir sayı gerekiyorsa
kaynak 1–3 olmalı. 12 numaralı kaynak satıcının kendi ürününü sattığı bir demo — anlatısı
doğru ama tek vaka ve bağımsız doğrulaması yok.

---

## 1 · when Agents Do Not Stop: Uncovering Infinite Agentic Loops

### Metodoloji

**Statik program analizi.** Kodu çalıştırmadan, kaynağı okuyarak döngü avlıyor. Sekiz aşama:

1. Kaynak kodu ve framework davranışını ortak bir **Agent IR**'ye çeviriyor — `llm.invoke()` →
   `LLM_CALL`, `messages.append()` → `STATE_APPEND`, `add_edge()` → workflow geçişi.
   Bu adım olmadan LangGraph'ta döngü **görünmüyor**, çünkü döngü kodun sözdiziminde değil
   framework'ün semantiğinde.
2. Agent IR'den **ALDG** adlı yönlü öznitelikli graf kuruluyor.
3. Grafta **SCC** (güçlü bağlı bileşen) hesaplanıyor — "yürütme buraya geri dönebilir mi".
4. Çevrimin içinde **maliyetli çağrı ya da durum büyümesi** var mı diye filtreleniyor.
   Sayaç artıran döngü elenir; LLM çağıran döngü kalır.
5. **Controller sınıflandırması** — döngüyü kim sürdürüyor: deterministic / model-controlled /
   tool-controlled / external-state-controlled / exception-controlled / mixed.
6. **Bound coverage denetimi** — sekiz kategori: `verified_bound`, `framework_default_bound`,
   `config_dependent_bound`, `missing_bound`, `weak_bound`, `disabled_bound`,
   `ineffective_bound`, `bypassed_bound`.
7. Belirsiz adaylar için **LLM negatif filtre** — LLM asla bulgu üretmiyor, sadece eliyor.
8. **Manuel doğrulama** — %91,9 kesinlik iddiası buna dayanıyor, değerlendiriciler arası uyum %94,6.

Kanıt: 6.549 Python repo, 246.748 dosya, 74 alarm, 68 doğrulanmış gerçek bulgu / 47 projede.

### Brief'e faydası

**T1 — çalışma prensibi.** Bu, alanın soru biçimini değiştiren kaynak: "döngü var mı" değil,
*"ajanik bir geri besleme yolu, maliyetli/durum büyüten bir işlemi etkili bir sınır olmadan
tekrar tekrar çalıştırabilir mi"*. Dokümanın giriş çerçevesi bu olmalı.

**Bound'un etkili sayılması için dört şart** — doğrudan alınabilir:
(1) ilgili controller'a uygulanmalı, (2) controller'ın runtime scope'unu kapsamalı,
(3) gerçek feedback path üzerinde baskın olmalı, (4) içteki bir işlemle sınırlı kalmayıp
dış döngüyü de durdurabilmeli.

Ve iç içe örnek, entegrasyon bölümünün en önemli uyarısı:
```
Supervisor loop
 └── Agent A  (max_turns=3)
      └── Tool loop
```
Agent A'nın sınırı var; supervisor Agent A'yı sürekli yeniden çağırıyorsa **dış yol hâlâ sınırsız.**

**T2 — PoC senaryoları.** Desen dağılımı senaryo seçimini kanıta bağlıyor:
sınırsız retry %25,0 · sınırsız tool-call iterasyonu %23,5 · tur sınırsız çok-agent sohbeti %20,6.
Gerçek vaka kodları da var (`2456868764/LiteRAG`, `NVIDIA-AI-Blueprints/ai-virtual-assistant`).

**T3 — dezavantajlar.** Dört sınırı kendisi kabul ediyor: statik analiz over-approximation
yapıyor · yalnızca Python ve 8 framework · projeye özgü sonlandırma mantığı çözülemiyor ·
LLM pruning kararsız (aynı adaylar farklı modellerde farklı eleniyor).

**T3 — entegrasyon gereklilikleri.** On bir katmanlı önlem listesi doğrudan gereklilik maddesi
olur: max turn · max workflow step · max tool-call · retry limiti · timeout · token/maliyet
bütçesi · context boyutu sınırı · **handoff/delegation derinliği sınırı** · yan etkili
araçlarda insan onayı · runtime telemetri ve alarm · döngü başına benzersiz ilerleme kontrolü.

**Kritik ayrım:** Bu araç **dağıtımdan önce** çalışıyor, bizim PoC **çalışma anında**.
CI ile üretim gibi; biri diğerinin yerine geçmiyor. Dokümanda bu ayrım baştan kurulmalı.

---

## 2 · Real-Time Detection and Repair of LLM Agent Failures

### Metodoloji

**Kontrollü enjeksiyon + gerçek korpus + ablasyon + dış doğrulama.** Dört katmanlı:

1. Her ajan adımından **telemetri vektörü** çıkarılıyor: semantik çıktı (karakter 3-gram
   özellik hash'i — **gömme modeli yok**), belirsizlik, eylem/davranış metadatası. v1'de 43,
   v4'te 60 boyut.
2. Sağlıklı koşumlardan **tek-sınıflı ESN + CUSUM** eğitiliyor. Sadece readout katmanı
   eğitiliyor (ridge regression) — eğitim 1,7 sn, karşılaştırmalı GRU 68 sn.
3. **Enjekte edilmiş hata** (looping, tool cascade, goal drift, content corruption) ve
   **organik hata** ayrı ayrı ölçülüyor.
4. Kendilerinin kurmadığı iki dış benchmark'ta (AFTraj-2K, ATBench) tekrar test ediliyor.

Ölçek: 2.823 bölüm, 25 veri kümesi, 3 framework (özel döngü, LangGraph, AutoGen), 4 model,
gerçek araçlarla 770 bölüm.

Ablasyon disiplini örnek alınmalı: token log-prob kanalını çıkardıklarında performans
neredeyse değişmemiş — yani "belirsizlik sinyali şart" varsayımını **kendi verileriyle
çürütmüşler**.

### Brief'e faydası

**T1 — tespit mantıkları ve sınırları.** Bu kaynağın asıl değeri burada: hangi hata sınıfının
telemetriyle **görülebileceğini**, hangisinin **yapısal olarak görünemeyeceğini** ölçmüş.
Gerçek korpusta sınıf bazında tespit:

| Hata sınıfı | Tespit aralığı |
|---|---|
| Döngü | 0,48 – 1,00 |
| Araç zinciri hatası | 0,17 – 1,00 |
| Hedef sapması | 0,66 – 0,86 |
| Yanlış belge içeriği | **0,00 – 0,20** |
| Bozuk JSON | 0,05 – 0,50 |

Kural: **davranış değişiyorsa görülür, yalnızca içerik değişiyorsa görülmez.**

**T1 + T3 — en güçlü tek bulgu.** Deterministik kontroller istatistiksel monitörü yeniyor:

| Sistem | Yakalanan | Yanlış pozitif |
|---|---:|---:|
| Deterministik kontroller | %60 | **0/63** |
| Kontroller + coverage | **%96** | **0/63** |
| Davranışsal monitör | %54 | 11/63 = %17 |

Üç kontrol: `total_consistency` (belirtilen toplam, görülen araç sonuçlarından yeniden
hesaplanıyor) · `required_coverage` (gerekli araç çağrıları gerçekten yapıldı mı) ·
`tool_contract` (sonuç, aracın üretebileceği biçimlerden biri mi). Bunlar sağlıklı null
dağılımı, eşik kalibrasyonu ve ikinci LLM çağrısı **gerektirmiyor**.

Bu bulgu bizim PoC'nin deterministik tercihini bağımsız olarak doğruluyor — ve sunumda
"neden LLM yargıcı kullanmadınız" sorusunun sayısal cevabı.

**T2 — PoC'ye doğrudan girecek iki şey.**

*Onarım merdiveni.* Alarmdan sonra son bilgi-toplama checkpoint'ine geri sarıp yeniden
çalıştırma. Kademelerin ölçülmüş kurtarma oranları:

| Onarım kademesi | Kurtarma |
|---|---:|
| Hiçbir şey yapmama | %0 |
| Basit yeniden örnekleme | %16 |
| **Hangi kontrolün başarısız olduğunu söyleme** | **%45** |
| Genel "yeniden kontrol et" | %36 |
| Spesifik bulguyu ve doğru değerleri verme | %36 |
| Hesap makinesi kullan talimatı | %28 |

Bonferroni sonrası yalnızca "hangi kontrolün başarısız olduğunu söyleme" anlamlı kalıyor
($p = 0{,}0005$). **Ajana doğru cevabı vermek, hangi koşulun ihlal edildiğini söylemekten
daha iyi değil.** Bizim `NUDGE` mesajı bu yüzden genel bir öğüt değil, **tetiklenen
dedektörün adını** taşımalı.

Net görev başarısı %52 → %73; 25 başarısız bölüm kurtarılmış, çalışan hiçbir bölüm bozulmamış.

*Döngü için onarım çalışmıyor.* Ölçülmüş: goal drift'te 5 vakanın 4'ü kurtarılıyor ama
**looping'de yeniden çalıştırma bazen aynı döngüyü üretiyor**, tool cascade'de araç bozuksa
aynı hatalı sonuç geliyor. Yani döngüde doğru hamle onarmak değil **durdurmak**. PoC'nin
`STUCK` → dur kararı bu bulguyla desteklenmiş oluyor.

**T3 — dezavantajlar bölümünün omurgası.** Beş ölçülmüş sınır:

1. **Deployment'lar arası aktarım yok.** Qwen2.5 7B'de eğitilen monitör Llama 3.1 8B'ye
   taşındığında AUROC **0,527**, sağlıklı yanlış alarm **0,75**. Yeniden kalibre edilince 0,885.
   Model, sıcaklık, araç seti veya gecikme rejimi değişince yeniden kalibrasyon şart.
2. **Ufuk bağımlılığı.** ESN'nin bellek avantajı hata sonrası kalan adım sayısına bağlı:
   ≤3 adım +0,09 · 4–8 adım +0,14 · ≥9 adım **+0,40**. Kısa bölümlerde belleksiz yöntem
   daha rekabetçi.
3. **Organik hatalarda şans düzeyi.** Enjekte edilmemiş, doğal hatalarda AUROC 0,31–0,42 —
   yani şans düzeyi veya altı. Sıcaklıkla eşleştirilmiş null'da **%36 yanlış alarm**.
   Enjekte edilmiş hata dağılımı doğal hata dağılımına benzemiyor.
4. **Makul görünen yanlış değer tespit edilemiyor.** Araç sonucu biçimsel olarak doğru
   ama içeriği yanlışsa, dış referans olmadan telemetri bunu bilemez.
5. **LLM yargıcı da kusurlu.** Ölçülen gerçek Gemini 2.5 Flash yargıcı: tespit **0,548**,
   yanlış alarm 0,052 — varsayılan 0,90/0,02 değil. Gerçek oranlar simülasyona konunca
   tespit kurtarma %82'den %43'e düşüyor. Yargıç goal drift'te mükemmel, context
   corruption'da zayıf (0,18); monitör tam tersi (1,00).

**T3 — avantajlar.** Maliyet tarafı çarpıcı: telemetri adaptörü ortanca **674 µs/adım**,
p95 1.045 µs; ESN monitörü ~200 µs/adım. Yani gerçek zamanlı izleme ölçülebilir biçimde ucuz.

**Bonus:** Sessiz abort'u çok basit bir completion check yakalıyor — **7/7, 0 yanlış pozitif**.
En ucuz kontrol en yüksek getiriyi verdiği bir vaka.

---

## 3 · Inference-Time Budget Control for LLM Search Agents

### Metodoloji

**Sabit bütçe altında karşılaştırmalı benchmark + ablasyon + teorem.** Ayırt edici tercihi:
tüm yöntemler **aynı sert bütçe** altında ölçülüyor ve sınırı aşan örnek başarısız sayılıyor.
Böylece bir yöntemin iyi görünmesi "daha çok kaynak harcadığı için" olamıyor.

Dört veri kümesi (HotpotQA, 2WikiMultihopQA, MuSiQue, Bamboogle) × dört bütçe seviyesi ×
üç omurga. Bütçe seviyeleri **çift eksenli**: araç çağrısı üst sınırı (1–3) ve çıktı token
üst sınırı (100–500) birlikte.

Yöntem eğitimsiz: her adımda üç eylemi (SEARCH / DECOMPOSE / ANSWER) **birim bütçe başına
görev değerine** göre puanlıyor, en yükseği seçiyor.

### Brief'e faydası

**T1 — bütçenin "kaç" değil "nereye" sorusu olduğu.** Brief bütçeyi bir tavan olarak
tanımlıyor; bu makale bir **tahsis** problemi olduğunu gösteriyor. Ana tezi:

> Daha fazla arama ya da daha çok token, otomatik olarak daha iyi yanıt demek değil;
> asıl önemli olan bütçenin hangi karar noktasına harcandığı.

**Çift bütçe kavramı** doğrudan brief'in kalemleriyle örtüşüyor: bir ajan araç çağrısı
bakımından ekonomik olup token bütçesini tüketebilir, ya da tersi. Tek sayaçlı tavan bunu
kaçırır. Bizim PoC'nin beş eksenli tasarımı bu gerekçeyle savunulabilir.

**Bütçe baskısı sinyali** — kalan bütçe, iki eksenden **en kritik olanına** göre hesaplanıyor:
$\rho_t = 1 - \min\{b_{\text{tool}}/B_{\text{tool}},\; b_{\text{tok}}/B_{\text{tok}}\}$.
Baskı arttıkça yeni arama cezalandırılıyor, cevap verme çekici hâle geliyor. PoC'de
`warn_at` eşiği bunun basit bir hâli; en kritik eksene göre hesaplama iyileştirme olabilir.

**T1 + T3 — ölçülmüş ablasyon bulgusu.** Bütçe-bağımlı ceza bileşeni çıkarıldığında her veri
setinde performans düşüyor; 2WikiMultihopQA'da F1 0,63 → 0,43. Yani ana teknik sonuç
"VOI skoru kullandık" değil, **kalan bütçe durumunu eylem seçimine açıkça katmanın kritik
olduğu**.

**T3 — avantajlar, sayaçlı.** Kontrolcü ek bir hesaplama katmanı olmasına rağmen ortalama
duvar-saati süresini 20,91 sn → 15,23 sn'ye indiriyor (**%27,2 düşüş**). Gereksiz retrieval
ve tekrarlı decomposition azaldığı için. "Kontrol koymak yavaşlatır" itirazının ölçülmüş cevabı.

**T3 — dezavantajlar.** Makale kendi sınırını net çiziyor: kazanç **bütçe kıtken** büyük,
bol kaynakta azalıyor; güçlü omurgada (Qwen3.5-122B) sonuçlar karışıyor, bazı yüksek bütçe
hücrelerinde rakipler öne geçiyor. Ayrıca "VOI" adı güçlü görünse de uygulama sabit
katsayılar ve el yapımı cezalar üzerine kurulu — öğrenilmiş bir değer fonksiyonu değil.
Ve teorem pratikte uygulanmıyor: gerçek kazanç/zarar miktarları bilinmediği için yayımlanan
sistem muhafazakâr deterministik bir kural kullanıyor. **Yazarlar bunu açıkça kabul ediyor.**

**T2 — koruyucu kural fikri.** Salt maliyet optimizasyonu tehlikeli: bütçe azalınca cevap
vermek ucuz olduğu için aşırı çekici hâle geliyor. Bu yüzden üstte deterministik guard'lar
var — zayıf kanıtla erken cevap engelleniyor, bileşimsel soruda en az bir arama zorunlu.
PoC'ye "bütçe bitti diye kalitesiz cevap verme" koruması eklenebilir.

---

## 4 · claude_platform_budget_control (Claude Task Budgets)

### Metodoloji

**Resmî API spesifikasyonu.** Ölçüm değil, ürünün davranış sözleşmesi. Kanıt gücü farklı bir
türde: burada yazan şey ürünün ne yapacağının taahhüdü, dolayısıyla entegrasyon kararı için
en güvenilir kaynak türü — ama "bu yaklaşım işe yarıyor mu" sorusuna cevap vermez.

### Brief'e faydası

**T1 — envantere dördüncü zorlama katmanı.** Token Budgets makalesi üç katman tanımlıyordu
(derleme zamanı / runtime middleware / transport). Bu dördüncüsü: **modelin kendisi**.
Sunucu, modelin gördüğü bir geri sayım işareti enjekte ediyor; model kendini ayarlıyor.

**T3 — en önemli tasarım dersi:** **tavsiye niteliğinde, zorlayıcı değil.** Model, kesmenin
bitirmekten daha bozucu olacağı bir eylemin ortasındaysa bütçeyi aşabiliyor. Sert tavan
hâlâ `max_tokens`. Yani model tarafı öz-düzenleme bir guardrail'in **yerine geçmez**,
üstüne gelir. Dokümanda bu ayrım net durmalı.

**T3 — beklenmedik başarısızlık modu.** Çok küçük bütçe **reddetme benzeri davranış**
üretiyor: model işi hiç denemeyebiliyor, agresif daraltabiliyor ya da erken durabiliyor.
Dokümanın kendi uyarısı: *"beklenmedik reddetme veya erken durma görüyorsan, başka
parametreleri ayıklamadan önce bütçeyi yükselt."* Bu, "limit koyalım" refleksinin
ölçülmemiş ama belgelenmiş bedeli.

**T3 — entegrasyon gereklilikleri.** Dört somut kısıt:
- Minimum `total` = 20.000 token; altı 400 hatası
- Geri sayım **yalnızca modele görünüyor** — API cevabında kalan bütçe alanı yok, kendi
  sayacını tutacaksın
- Sayaç modelin o turda **gördüğünü** düşüyor, istemcinin yeniden gönderdiği geçmişi değil
  (dokümandaki örnek: 3 turda ~20.820 token gönderilmiş, bütçeden 19.000 düşmüş)
- **Prompt caching çakışması:** bütçe değeri render edilen prompt'a giriyor; `remaining`'i
  her turda güncellersen cache önekini geçersiz kılıyorsun
- Claude Code ve Cowork'te desteklenmiyor

**T2 — PoC'ye taşınabilir fikir.** "Bütçe, modelin üzerinde akıl yürütebileceği bir sinyal"
tasarımı. Bizim `NUDGE`'a kalan bütçeyi eklemek bunun basit hâli.

---

## 5 · agentbudget_framework (AgentBudget)

### Metodoloji

**Açık kaynak ürün README'si.** Kanıtı "çalışıyor ve yayımlanmış olması"; bağımsız ölçüm yok.
Üç dilde SDK (Python/Go/TS), Apache 2.0, sürüm notlarıyla birlikte olgunluk gösteriyor.

### Brief'e faydası

**T1 — envantere üçüncü konuşlandırma noktası.** LiteLLM gateway'de, framework'ler koşum
döngüsünde; bu **süreç içinde**, istemciyi sararak. Kendi tanımıyla *"Not an LLM proxy."*
Atlas benzeri bir platformda "kontrol nereye konur" tartışmasının üçüncü seçeneği.

**T2 — PoC'nin bir açığını kapatan tasarım.** `finalization_reserve`: bütçenin bir dilimi
nihai cevap adımına ayrılıyor, sert limit erken tetikleniyor, tavan hiç aşılmıyor.
Bizim PoC nihai cevabı tavanı aştıktan **sonra** çağırıyor — bu daha az dürüst.
Ayrıca `would_exceed(tahmini_maliyet)` ile son çağrıdan önce kontrol.

**T2/T3 — çok-agent bütçe bölüştürme.** `parent.child_session(max_spend=2.0)`, maliyetler
yukarı toplanıyor. IAL-SCAN'in "bound'lar nested agent'lar arasında aktarılmalı" önerisinin
çalışan karşılığı.

**T3 — sorun tanımı için sayılar.** *"Sıkışmış bir ajan 10 dakikada 200 LLM çağrısı yapar.
Kimse fark etmeden 50–200 $."* Ve ölçek argümanı: *"%5 hata oranıyla 1.000 eşzamanlı oturum
= 50 kaçak ajan."* Sunumun açılış slaytı için — ama kaynağın satıcı olduğu belirtilmeli.

**T3 — üç kademeli devre kesici** ve rapordaki `terminated_by` alanı (`null` /
`budget_exhausted` / `loop_detected`) bizim `Status` enum'unu bağımsız doğruluyor.

---

## 6 · pi_anti_doom_loop

### Metodoloji

**Açık kaynak ürün README'si + test takımı tanımı.** Kanıt gücü açısından en ilginç yanı,
beş katmanlı test takımını açıkça listelemesi: unit (dedektör semantiği) · fixture
(**gerçek doom-loop transkriptleri yakalanıyor, sağlıklı oturumlar yakalanmıyor**) ·
fuzz (tohumlu rastgele akışlar: asla patlamıyor, yanlış pozitif yok, enjekte edilen döngü
her zaman engelleniyor) · integration · e2e.

Fixture ve fuzz katmanı bizim PoC'nin kontrol senaryolarıyla aynı felsefe — yakalamak kadar
yakalamamak da test ediliyor.

### Brief'e faydası

**T1 — altı tespit sinyali**, üçü bizde yok:
1. Aynı `(araç, argüman)` tekrarı — bizde var
2. Aynı aracın ardışık hata vermesi — bizde var
3. Aynı asistan metninin birebir tekrarı — **bizde kısmen** (monolog sayıyor, metin kimliğine bakmıyor)
4. **Tek bir mesajın içinde aynı cümlenin 3+ kez geçmesi** — kendi kendine ekleme döngüsü, **bizde yok**
5. **Yakın-benzer metin, ≥%55 token benzerliği** — **bizde yok**
6. **Yakın-benzer metin çevrimi** — dönüşümlü yeniden ifade edilmiş komutlar, **bizde yok**

5 ve 6 numara, PoC'nin "anlamsal denklik yakalanmıyor" sınırına **gömme kullanmadan** cevap
veriyor. Token örtüşme oranı ucuz ve deterministik.

**T2 — üç kademeli müdahale.** Bizde iki kademe var (nudge → dur). Burada üç: **steer**
(yönlendir, koşum sürsün) → **abort + bir taze devam direktifi** → gerçekten dur.
Otomatik devam bütçesi tavanlı.

**T2 — iki kopyalanabilir ayrıntı.** *Boşa giden token sayımı*: dedektör tekrarlarda yakılan
token'ı tahmin edip engelleme gerekçesinde raporluyor — döngünün durdurulmadan önce ne
kadara mal olduğu görünür oluyor. *Eşikler minimum 2'ye kırpılıyor* ki bozuk yapılandırma
ajanı kilitlemesin.

**T3 — entegrasyon gereklilikleri.** Üç yapılandırma bizde yok ve gereklilik listesine girer:
zaman penceresi (uzun oturumdaki **yavaş kronik döngüler** sayı eşiğine hiç ulaşmadan bütçeyi
yiyebilir) · hata **oranı** eşiği · araç dışlama listesi (kaçış kapısı).

Ve bir UX gerekliliği: `/loopcheck suspend` — kasıtlı tekrar için manuel askıya alma.
Kontrolün kullanıcı tarafından kapatılabilmesi, kontrolün kendisi kadar önemli.

---

## 7 · agent_improvement_loop (OpenAI cookbook)

### Metodoloji

**Çalıştırılabilir öğretici.** Sentetik bir şirket veri odası kuruyor, Agents SDK tabanlı bir
analist ajan tanımlıyor, beş izli koşum üretiyor, insan + LLM geri bildirimi topluyor,
otomatik eval takımı üretiyor, doğrulama kapısından geçiriyor ve `codex_handoff.md` adlı tek
bir dosyayla harness değişikliğini devrediyor.

Kanıt gücü: uçtan uca koşuyor, ama bir iddiayı ölçmüyor — bir **iş akışı** gösteriyor.

### Brief'e faydası

**T1/T3 — dış döngü.** Bizim çalıştığımız her şey iç döngü (bir koşumun kontrolü). Bu, dış
döngü: **izler → geri bildirim → eval → harness değişikliği**. Eşiklerin kim tarafından,
neye bakarak değiştirileceği sorusunun kurumsal cevabı.

**T3 — entegrasyon gerekliliği, doğrudan alınabilir.** `EVAL_METADATA` bir harness sürümünü
`version` / `status: promoted` / `promotion_gate: manual_review` ile işaretliyor. Yani
**döngü yapılandırması prompt'la birlikte sürümleniyor ve terfi kapısından geçiyor.**
Arize'ın "loop configuration gets versioned alongside the prompt" cümlesinin çalışan hâli.
Bir platformda eşik değiştirmek kod değiştirmekle aynı süreçten geçmeli.

**T2 — araç politikası deseni.** `TOOL_POLICY` içinde `allowed_data_root`, `writable_output_root`,
`required_artifacts`, `mutation_policy` var. Ve iki **runtime doğrulama aracı**:
`check_evidence_coverage.py` ve `validate_output_contract.py` — ajan nihai cevaptan önce
bunları çalıştırmakla yükümlü. Bu, `sde_offer_loop`'un "doğrulama kapılı durma" ilkesinin
somut uygulaması ve PoC'nin en büyük açığına doğrudan şablon.

Otomasyon derecesi de bir tasarım kararı olarak sunuluyor: önerilen değişiklik setini
geliştirici onaylıyor (başlangıç), eval kapısına güven arttıkça daha derin otomasyon.

---

## 8 · arize_research_control_loop

### Metodoloji

**Kavramsal derleme.** Satıcı araştırma yazısı; ölçüm yok, damıtılmış pratik var. Değeri
ölçümde değil, **kavramları doğru adlandırmasında**.

### Brief'e faydası

**T1 — dokümanın kavram iskeleti.** Dört faz: **observe → decide → act → update**, ve
"model yalnızca *decide* fazını yürütür". Bu tek cümle, brief'in *"agent kendi adımlarına
kendisi karar veriyor"* ifadesini teknik bir yere oturtuyor: model kararı veriyor, harness
diğer üç fazın tamamına sahip.

Beş durma koşulu: görev tamamlama · adım limiti · bütçe tavanı · deadline · hata politikası.
Ve sıralaması net: **"adım limiti tek başına en etkili guardrail, çünkü modelin söylediği
hiçbir şeye bağlı değil."**

**T3 — sunumun en güçlü tek cümlesi:**
> *"Her döngüye modelin yargısına bağlı olmayan sert bir durdurma gerekiyor, çünkü model,
> bitip bitmediği konusunda yanılması en muhtemel bileşendir."*

**T1 — üç kaçak döngü deseni** ve teşhisleri: *identical retries* (tool adı + normalize
argüman hash'i, tekrar edince önceki sonucu açıkça enjekte et veya çağrıyı engelle) ·
*oscillation* (ileri geri düzenleme; ilerleme dedektörü gerekli) · *plan thrash* (yürütmeden
yeniden planlama). Sonuncunun teşhisi keskin: **"akıl yürütme kılığına girmiş bir bağlam
kurma hatasıdır."** Plan churn bizim PoC'de yok.

**T2/T3 — eşik seçiminin cevabı.** *"Başarılı görevlerin adım sayısı dağılımına bak, tavanı
kuyruğunun üstüne koy. Meşru işi kesecek kadar dar bir adım limiti, modelin kötüleştiği gibi
görünen sessiz bir kalite gerilemesine dönüşür."* Ve izleme metriği: koşumların yüzde kaçı
limitte sonlanıyor — bu oran tırmanıyorsa görev zorluğu ya da araç güvenilirliği değişmiştir.

**T3 — entegrasyon gerekliliği.** *"Her koşumda bir durma sebebi kaydedin. `completed`,
`max_steps`, `budget_exceeded` ve `error` farklı sonuçlardır; ortalamalarını almak neyin
yanlış gittiğini gizler."* Ve telemetri: **iterasyon başına bir span**, iterasyon indeksi,
araç adı ve argüman hash'i ile etiketli — *"bütün koşum tek bir span ise, 4–19. adımların
aynı çağrı olduğunu göremezsin."*

---

## 9 · sde_offer_loop (Loop Engineering)

### Metodoloji

**Kavramsal derleme / eğitim içeriği.** Ölçüm yok; bir tasarım disiplini tanımlıyor.
"Harness engineering"in altındaki katman olarak konumlanıyor.

### Brief'e faydası

**T1 — dokümanın en iyi tek çerçevesi.** Beş satırlık döngü iskeletini yazıp her yorum
satırını bir tasarım kararına bağlıyor: `done()` ne zaman doğru · bütçeler ne kadar ·
`execute` başarısız olunca ne · `verify` neyi kontrol ediyor · state nasıl compact ediliyor ·
alttan düşünce ne oluyor.

**T1/T2 — PoC'nin en büyük açığının kaynağı.** Dört durma sinyali arasında hiyerarşi kuruyor
ve **doğrulama kapılı durmayı** en güçlü sinyal ilan ediyor, çünkü modelin görüşüne değil
ortama dayanıyor. Alıntılar sunuma girer:

> *"Modelin 'bitti' demesi durma isteğidir, tamamlanma kanıtı değildir."*
> *"'Bitirdim dedi', ajan dünyasının 'benim makinemde derleniyor'udur."*
> *"Zayıf bir model, sıkı bir doğrula-ve-yeniden-dene döngüsüyle, doğrulaması olmayan güçlü
> bir modeli çoğu zaman geçer."*

**T3 — "bütçe bir sinyaldir" tezi.** *"Bütçeler sadece bir kill switch değil; döngünün
üzerinde akıl yürütebileceği bir sinyaldir"* — adım bütçesi azalınca "2 adımın kaldı,
bulduklarını özetle ve dur". Claude'un task budget geri sayımı bunun ürünleşmiş hâli;
iki kaynak birbirini doğruluyor.

**T3 — altı tuzak**, dezavantajlar bölümünün çatısı: model-declared "done"a güvenmek ·
döngü tespiti olmaması (bütçeye tek başına güvenmek = tüm bütçeyi tek bir hataya harcamak) ·
zarif inişi olmayan bütçeler · **topolojiyi aşırı mühendislik etmek** · context rot ·
körlemesine ayar (trajectory analizi olmadan yanlış şeyi düzeltirsin).

**T1 — dört topoloji:** tek döngü · refleksiyon · planla-sonra-uygula · orkestratör+alt-ajan.
Entegrasyon bölümünde "hangi topoloji" bir karar maddesi olarak durabilir.

---

## 10 · loop_budget_source

### Metodoloji

**Kavramsal + çalışan referans kod.** Ölçüm yok ama tam bir `LoopGuard` sınıfı veriyor:
`check_budget()` · `record_progress()` · `record_tool_call()` · `should_retry()` ·
`detect_repeat_pattern()`. Türkçe ve satır satır yorumlu.

### Brief'e faydası

**T1 — beş koruma**, brief'in kalemleriyle birebir örtüşüyor: kesin sınırlar (adım + araç +
**araç başına retry** + token/maliyet) · üstel gecikme + jitter · ilerleme kontrolleri ·
döngü parmak izleri · **çekimser kalma yolu**.

**T2 — üç somut kod fikri.**
- `action_fingerprint(tool, args, outcome)` — parmak izine **sonucu da katıyor**.
  `sde_offer_loop` bağımsız olarak aynı şeyi yapıyor (`sig = (name, args, result.error)`).
  Bizim PoC eylem ve gözlemi ayrı hash'liyor; birleştirmek denenebilir.
- `record_progress()` — son 3 `state_hash` aynıysa döngüdesin. Dış durumun hash'i,
  bizim `progress` bayrağından daha nesnel bir sinyal.
- `detect_repeat_pattern()` — **son 6 eylem ≤2 benzersiz parmak izine düşüyorsa döngü.**
  Bizim k=1..12 çevrim taramasından çok daha basit ve A-B-A-B'yi yakalıyor.
  Karşılaştırmalı test etmeye değer.

**T2 — araç başına retry limiti.** `max_retries_per_tool=2`. Bizde sadece genel `max_replans`
var; bir araç bozuksa genel sayacı tek başına yiyor. PoC açığı.

**T3 — "Onurunla dur" ilkesi**, `Status` mesajının yapısını belirliyor. Durma cevabı dört şey
içermeli: ne denedi · ne buldu · neden durdu (bütçe/ilerleme/izin) · en iyi bir sonraki eylem.
*"Bu, sonsuz döngüyü faydalı kısmi sonuca dönüştürür."*

**T3 — sunum kapanışı için iki cümle:**
> *"Ajanlar sadece yanıldıkları için başarısız olmazlar. Israrcı oldukları için başarısız
> olurlar. Ve ısrar pahalıdır."*
> *"Durmak başarısızlık değildir. Durmak kontroldür."*

**T3 — örnek politika**, gereklilik listesine sayı verir: adım ≤12, araç çağrısı ≤8,
araç başına deneme ≤2, süre ≤60 sn.

---

## 11 · loop_budget_medium2 (Modexa, "Ajan Döngüsü Problemi")

### Metodoloji

**Kavramsal makale.** Ölçüm yok; bir anlatısal vaka var (destek ajanının "tekrar doğrula"
tuzağı). Dosya bir sohbet dışa aktarımı olduğu için aynı kod bloğu dört kez tekrarlanıyor —
özgün içerik 2286. satırdan sonra.

10 numaralı kaynakla önemli ölçüde örtüşüyor; ama üç şeyi tek başına getiriyor.

### Brief'e faydası

**T1 — beş döngü sebebi**, biri başka hiçbir kaynakta yok: *"ajanın 'yanlış yapmamak' üzerine
optimize edilmesi."* Aşırı tedbir teşvik eden bir prompt, en güvenli yolu "bir kez daha
doğrula" yapıyor. *"Bir sorun çözücü inşa etmediniz. Bir riskten kaçınma makinesi inşa
ettiniz."* Bu, prompt tasarımının döngü üretmesi — teknik bir kontrolle çözülmüyor.

**T2/T3 — geri dönüş merdiveni.** Ajan kendi kendine tekrar icat etmemeli, merdiveni izlemeli:
(1) backoff ile bir kez dene → (2) aracı/sağlayıcıyı değiştir → (3) kapsamı daralt →
(4) kullanıcıya açıklayıcı soru sor → (5) elindeki en iyi cevabı + sonraki adımları döndür.
PoC'de `NUDGE` sonrası davranış bu merdivene bağlanabilir.

**T3 — durum makinesi önerisi.** Serbest formlu döngüyü `ANLA → TOPLA → HAREKETE GEÇ →
DOĞRULA → YANITLA → DEVRET` gibi net durumlara indir, yalnızca belirli geçişlere izin ver.
*"Sadece bu bile çok büyük bir döngü sınıfını ortadan kaldırır."* Entegrasyon bölümünde
"agent'ları serbest bırakmak yerine durum makinesine oturtmak" bir seçenek olarak durabilir.

**T3 — "kullanıcıya sor"un nasıl yapılacağı.** Tek soru sor (beş değil) · neyi değiştireceğini
açıkla · kullanıcı umursamazsa varsayılan sun. Gerekçe: *"40 adım boyunca yanlış tahmin
yürütmekten ucuzdur."*

**T3 — geliştirici kuralı**, dokümanın kapanış cümlesi olabilir:
> *"Her ajan döngüsünün tasarlanmış bir çıkışı olmalıdır — başarı, geri dönüş, sorma veya
> devretme. 'Sonsuza kadar dene' değil. 'Umarım model çözer' değil."*

---

## 12 · galileo_ai_video

### Metodoloji

**Ürün demosu transkripti.** En zayıf kanıt: **tek vaka**, satıcının kendi ürününü gösterdiği
senaryo, bağımsız doğrulama yok. Ama anlattığı hata modu başka hiçbir kaynakta bu netlikte
yok.

### Brief'e faydası

**T2 — PoC'de eksik olan hata modu.** Bizim altı hata senaryosunun hepsi ya durmuyor ya
başarısız oluyor. Bu farklı: **retry sonunda başarılı oluyor.** Kullanıcı doğru cevabı alıyor,
aynı format, aynı güven; dışarıdan hiçbir şey yanlış görünmüyor.

> *"Bu, hakkında hiç ticket açılmayan türden bir hata. Sessiz ama öldürücü."*

Senaryo eklenebilir: iki başarısız araç çağrısı, sonra başarı. PoC `Status.OK` döner ve
hiçbir guardrail tetiklenmez — **bu bir açık**. Yakalamanın yolu hata *oranı* sinyali
(`pi`'nin `FAIL_RATE`'i) ya da boşa giden token sayımı.

**T3 — gözlemlenebilirlik gerekliliği.** Teşhis span düzeyinde yapılıyor: araç hata oranı
sıçraması, ana span'de bir başarıdan önce iki başarısız çağrı. Arize'ın "iterasyon başına
bir span" gerekliliğiyle örtüşüyor — iki bağımsız kaynak aynı telemetri şartını söylüyor.

**T3 — dezavantaj argümanı için.** *"İki tekrar önemsiz görünebilir ama önemli olan desen.
Bu retry mantığında ne devre kesici ne zaman aşımı koruması var. Bugünkü geçici hata, yarın
sürekli bir kesinti olur ve her istek maksimum retry'ı yakarak tüm alt akışı tıkar."*
Yani eşiğin altında kalan bir döngü bugün zararsız, yarın kesinti.

Ve suçun yerini doğru koyması önemli: *"Ajanlar doğru davrandı. Zafiyet veri çekme
katmanında."* Loop detection'ın her zaman ajanı suçlamaması gerektiğinin hatırlatması.

---

## Brief maddesi × kaynak matrisi

| Brief maddesi | Birincil kaynaklar | Destekleyici |
|---|---|---|
| **T1** Çalışma prensibi | when Agents Do Not Stop · arize · sde_offer | loop_budget_source |
| **T1** Tespit mantıkları | real_time_Detection · pi_anti_doom · loop_budget_source | when Agents · arize |
| **T1** **Sınırlar** | **real_time_Detection** · when Agents · Inference-Time | claude_platform |
| **T2** Döngü senaryoları | when Agents (desen dağılımı) · galileo (sessiz retry) | pi_anti_doom (fixture felsefesi) |
| **T2** Limit aşımı senaryoları | Inference-Time (çift bütçe) · agentbudget | claude_platform |
| **T2** Müdahale tasarımı | real_time_Detection (onarım merdiveni) · pi_anti_doom (3 kademe) | loop_budget_medium2 (fallback ladder) |
| **T3** Avantajlar | real_time_Detection (%96 / 0 FP) · Inference-Time (%27,2 hızlanma) | agentbudget |
| **T3** Dezavantajlar | real_time_Detection (5 sınır) · when Agents (4 sınır) · Inference-Time | claude_platform (reddetme davranışı) |
| **T3** Entegrasyon gereklilikleri | when Agents (11 önlem) · agent_improvement_loop (sürümleme) · arize (durma sebebi kaydı) | pi_anti_doom · agentbudget |
| **Eşik seçimi** | arize · claude_platform (p99) | sde_offer |

---

## Hiçbir kaynağın cevaplamadığı üç soru

Dürüst kayıt — brief'in ihtiyacı olup elimizde olmayanlar:

1. **Yanlış pozitifin bedeli ölçülmemiş.** Meşru bir uzun koşumu kesmenin maliyeti hiçbir
   yerde sayıyla yok. `real_time_Detection` yanlış alarm *oranını* veriyor (%17 davranışsal
   monitörde, %0 deterministik kontrollerde) ama kesilen koşumun iş değeri kaybı ölçülmemiş.
2. **Kontrolün kendi maliyeti kısmen ölçülmüş.** `real_time_Detection` mikrosaniye düzeyinde
   telemetri maliyeti veriyor, `Inference-Time` net hızlanma gösteriyor; ama guardrail'in
   mühendislik ve bakım maliyeti (eşik kalibrasyonu, deployment başına yeniden ayar)
   hiçbir yerde nicelleştirilmemiş — üstelik `real_time_Detection` bunun **her deployment
   için tekrarlanması gerektiğini** ölçmüş durumda.
3. **Türkçe/çok dilli ortamda tespit davranışı.** Metin benzerliği tabanlı sinyaller
   (`pi`'nin ≥%55 token örtüşmesi, karakter 3-gram hash'i) Türkçe'nin eklemeli yapısında
   nasıl davranır — hiçbir kaynak bakmamış. Atlas Türkçe içerik işliyorsa bu ölçülmeli.
