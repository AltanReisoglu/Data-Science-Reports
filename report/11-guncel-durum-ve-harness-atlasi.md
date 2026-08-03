# 11 — Güncel Durum ve Harness Atlası

**Ağustos 2026 · Sentez bölümü**

Bu bölüm iki yönde okunur:

- **Bölüm A — En sondan en başa.** Alanın bugünkü hâlinden başlayıp katman katman geriye, değişmeyen çekirdeğe iner. Amaç: *"2026'da context engineering nedir"* sorusuna, her katmanın **hangi problemi çözmek için** eklendiğini göstererek cevap vermek.
- **Bölüm B — En baştan en sona.** Her harness yapısını tek tek ele alır: nasıl tanımlanır, **modele tam olarak ne gider**, model ne üretir, harness ne yapar, ne kadara mal olur.
- **Bölüm C** ikisini tek bir turda birleştirir: gerçek bir uçtan uca iz.

Diğer bölümlerin sentezidir; mekanizmaların ayrıntısı için çapraz referanslar verilmiştir.

---
---

# BÖLÜM A — En sondan en başa

Aşağıdaki katmanlar **kronolojik olarak tersten** sıralanmıştır: en üstte 2026'nın en yeni katmanı, en altta 2022'nin çekirdeği. Her katman kendisinden bir alttakinin **ölçeklenme sınırına** cevap olarak doğdu.

```
K7  Harness mühendisliği bir disiplin          2026 Q2–Q3   ← şu an buradayız
K6  Bağlam grafiği / ontoloji / semantik katman 2026 Q1–Q2
K5  Öğrenilmiş bağlam sıkıştırma               2025 Q4–2026
K4  Bağlamın kendisi öğrenen artefakt (ACE)    2025 Q4
K3  Progressive disclosure: skill + PTC         2025 Q3–Q4
K2  Ajanik arama vektör RAG'i devirdi           2025 Q2
K1  "Context engineering" adını aldı            2025 Q3
K0  ReAct + prompt engineering + RAG v1         2022–2023
──  DEĞİŞMEYEN ÇEKİRDEK: model durumsuzdur
```

---

## K7 — Harness mühendisliği bir disiplin oldu (2026 Q2–Q3)

**Neye cevap:** K0–K6'daki tekniklerin hepsi *modelin dışında* yaşıyordu ama kimse bu dışarısını bir bütün olarak adlandırmamıştı.

Denklem netleşti:

> **Ajan = Model + Harness**
>
> Harness: modelin kendisi olmayan **her şey** — kod, konfigürasyon, yürütme mantığı. Bağlamı, araçları, durumu, kısıtları, izinleri, izlemeyi ve hatadan dönüşü yöneten sistem.

Üç iş birbirinden ayrıldı:

| Disiplin | Kapsam | Zaman ölçeği |
|---|---|---|
| **Prompt engineering** | Tek bir metnin ifadesi | Bir istek |
| **Context engineering** | Çok adımlı yürütme boyunca bağlamda ne olduğu | Bir oturum |
| **Harness engineering** | Prompt + bağlam + araç + izin + kurtarma etrafındaki **kapalı döngü** | Sistemin ömrü |

### Ölçülmüş sonuç

Aynı model havuzu ve aynı görevlerle çalışan **en iyi ve en kötü konfigüre edilebilir harness arasındaki fark 23,8 puan** ölçüldü. Bu bir model hikâyesi değil, harness hikâyesi. ⚠️ *(İkincil kaynak — MindStudio derlemesi; birincil benchmark'tan doğrulanmalı.)*

Bunun doğal sonucu şu iddia: **ajan yeteneği "model" düzeyinde değil, "model–harness konfigürasyonu" düzeyinde raporlanmalı.** SWE-bench'te "Model X %72 aldı" cümlesi eksik bilgidir; hangi harness ile alındığı sorulmalıdır.

### Model-bound vs harness-bound

| | Model-bound | Harness-bound |
|---|---|---|
| Örnek | Tek bir iş sınıfı için sıkı bağlı yığın (Codex tarzı) | Model takılıp çıkarılabilen genel harness |
| Güçlü yanı | O iş sınıfında tepe performans | Model değiştirilebilir, ucuz modelle iyi sonuç |
| Zayıf yanı | Taşınamaz | Konfigürasyon kalitesine aşırı duyarlı (yukarıdaki 23,8 puan) |

**Alandan gelen pratik özet** (kullanıcının kaynak listesinden):

> *"Model, etrafına sardığın şeyden daha az önemli. Sıkı bir harness'e sahip ucuz bir model — kurallar, anti-slop listesi, geri besleme döngüsü ve doğru bağlamla — körlemesine çalışan frontier modeli yener."*

### Bu katmanın kendi sınırı

Harness elle konfigüre ediliyor ve konfigürasyon uzayı büyük. Buna verilen yanıt **kendini geliştiren harness'ler**: gözlemlenebilirlik verisinden (hangi tool başarısız oldu, hangi adım tekrarlandı) harness'in kendisini otomatik evrimleştirme. Ağırlıkları, harness'i ve görev çözümlerini birlikte evrimleştiren çalışmalar 2026'da yayınlandı — ama üretimde yaygın değil. **Bu, alanın açık ucu.**

---

## K6 — Bağlam grafiği, ontoloji, semantik katman (2026 Q1–Q2)

**Neye cevap:** Ajanik arama (K2) bir kod deposunda mükemmel çalışıyor. **Kurumsal veride çalışmıyor** — çünkü orada aranan şey dosya değil *anlam*: "gelir" hangi tabloda, hangi tanımla, kim onayladı.

Ayrım 2026'da netleşti:

| Yapı | Ne yapar | Soru |
|---|---|---|
| **Semantik katman** | Arama/eşleme | "Bu metrik hangi kolon?" |
| **Ontoloji** | Bağlam ve akıl yürütme | "Bu şey nedir, neyle ilişkilidir, hangi eylem mümkün?" |
| **Bağlam grafiği** | Semantik anlam + bilgi ilişkileri + operasyonel sinyalleri **tek yapıda** birleştirir | "Şu an, bu kullanıcı için, bu doğru mu?" |

Gartner'ın Mart 2026 Data & Analytics Summit'te çizdiği ayrım: altta **Context layer** (ontoloji, şema, metrikler, policy-as-code), üstte **Intelligence layer** (akıl yürütme, model, ajanik iş akışı). ⚠️ *İkincil aktarım.*

522 kurumsal sorgu üzerinde yapılan bir ölçümde, birleşik çok boyutlu bağlama erişimi olan ajanlar yalnızca semantik tanımlara erişenlere göre **%38 daha yüksek doğruluk** verdi. ⚠️ *İkincil — birincil çalışmadan doğrulanmalı.*

### Kritik uyarı

> **Ajanları devreye almadan önce tam ontoloji kurmaya çalışmak, projeleri durduran şeyin ta kendisi.**

Pratik yol: elde olan metadata, SQL geçmişi ve dashboard'lardan bağlamı **bootstrap et**, sonra alan uzmanlarıyla döngü içinde iyileştir.

### Kod tarafındaki karşılığı

Aynı fikrin kod deposu versiyonu: **Tree-sitter ile yerel bilgi grafiği**. Ajan tüm dosyayı değil, sembol düzeyinde yalnızca gereken düğümleri okuyor. Büyük monorepo'larda 49× azalma iddia ediliyor (§A.8'deki tabloya bakınız). ⚠️ *Depo sahibinin iddiası.*

**Ne zaman gerekir:** kurumsal veri, çok kaynaklı, anlam belirsizliği yüksek. **Ne zaman gereksiz:** tek bir kod deposunda çalışıyorsan grep zaten yeterli (K2).

---

## K5 — Öğrenilmiş bağlam sıkıştırma (2025 Q4 – 2026)

**Neye cevap:** K1'in compaction'ı ve kırpması **kör** — "son N tool sonucunu sil", "ilk 25K token'ı tut". Uzun ufuklu ajanlarda kritik kanıt (tam hata string'i, tam dosya yolu) bu körlükte kayboluyor.

Terminal/kodlama ajanlarında problem özellikle keskin:

> Ayrıntılı loglar ve derleyici izleri, **düşük değerli gürültüyü seyrek ama tam olması gereken yürütme kanıtıyla iç içe geçiriyor.** Genel amaçlı budama bu kritik string'leri atabilir; özetleme ise onları başka kelimelerle ifade edip bozabilir.

Bu, §08'deki dört savunma hattının üstüne beşinci bir hat ekliyor: **görev-koşullu, öğrenilmiş sıkıştırma.**

| Yaklaşım | Fikir | Ölçülen |
|---|---|---|
| **ACON** (arXiv 2510.00615) | Hem çevre gözlemlerini hem etkileşim geçmişini sıkıştır; başarısızlıktan öğrenen, göreve duyarlı sıkıştırma kılavuzu optimizasyonu | **%26–54 tepe token azalması** |
| **Squeez** (arXiv 2604.04979) | Kodlama ajanları için **görev-koşullu tool çıktısı budama** | — |
| **Self-GC** (arXiv 2607.00692) | Uzun ufuklu ajanlar için kendi kendini yöneten bağlam | — |
| **Observational compression** (arXiv 2604.19572) | Terminal ajanları için kendini evrimleştiren gözlem sıkıştırma çerçevesi | — |

**Ortak fikir:** sıkıştırma kararı artık sabit bir kural değil, **görevin ne olduğuna bakan bir fonksiyon.** Aynı 500 satırlık test çıktısı, "testi geçir" görevinde 3 satıra inebilir; "flaky test'i teşhis et" görevinde inemez.

Mevcut tekniklerin envanteri: yörünge özetleme, hafıza getirme, bağlam kırpma, gözlem maskeleme, proaktif bağlam katlama, öğrenilmiş geçmiş sıkıştırma.

### Üretimdeki basit versiyonu

Bu araştırmanın pratikleşmiş hâli, tool çıktısını **bağlama girmeden önce** filtrelemek — kullanıcının kaynak listesindeki depoların çoğu tam olarak bunu yapıyor (§A.8).

---

## K4 — Bağlamın kendisi öğrenen bir artefakt: ACE (2025 Q4)

**Neye cevap:** K1–K3'ün hepsi bağlamı *yönetiyor* ama bağlam hâlâ **statik** — insan yazıyor, ajan tüketiyor. Ajan kendi deneyiminden öğrenemiyor.

**ACE — Agentic Context Engineering** (arXiv 2510.04618, Ekim 2025) bağlamı **evrilen bir oyun kitabı (playbook)** olarak ele alıyor.

### Teşhis ettiği iki hastalık

| Hastalık | Ne olur | Nerede görülür |
|---|---|---|
| **Brevity bias** (kısalık yanlılığı) | Alan bilgisi, "temiz özet" uğruna atılır | Her özetleme adımında |
| **Context collapse** (bağlam çöküşü) | Yinelemeli yeniden yazma ayrıntıyı zamanla aşındırır | Compaction'ın compaction'ı |

İkincisi doğrudan §08.7'nin uyarısıyla örtüşüyor: **özetin özetini almak bilgi kaybını üstel hâle getirir.**

### Çözümü: üç bileşenli döngü

```
Generation  →  görevi çöz, yörüngeyi üret
     ↓
Reflection  →  ne işe yaradı / ne yaramadı, çıkar
     ↓
Curation    →  playbook'a ARTIMLI DELTA olarak yaz  ──┐
     ↑                                                │
     └────────────────────────────────────────────────┘
```

Kritik tasarım kararı: **yeniden yazma değil, artımlı delta güncellemesi.** Playbook baştan yazılmadığı için çöküş olmuyor.

### Sonuçlar

- **+%10,6** ajan benchmark'larında
- **+%8,6** finans alanında
- **Etiketli denetim gerekmeden** — yalnızca doğal yürütme geri beslemesiyle çalışabiliyor

### Felsefi tersine dönüş

ACE, alanın geri kalanına ters bir şey söylüyor:

> Bağlamlar **kapsamlı, evrilen playbook'lar** olmalı — ayrıntılı, kapsayıcı, alan içgörüsü açısından zengin. LLM'ler uzun ve ayrıntılı bağlamla daha etkilidir ve **ilgiyi kendileri damıtabilirler.**

Bu, §01'in "dikkat bütçesi sonludur, az yaz" ilkesiyle **gerilim hâlinde.** Raporun tuttuğu pozisyon:

> İkisi çelişmiyor çünkü farklı şeylerden bahsediyorlar. §01 **her turda bağlama giren** şeyi kısıtlamayı savunuyor. ACE **diskte duran ve seçici olarak getirilen** bilgi tabanının zengin olmasını savunuyor. Playbook uzun olabilir; bağlama giren dilim kısa olmalı. Zaten ACE'nin kendisi de curation adımıyla seçim yapıyor.

**Bugünkü karşılığı:** §05'teki `MEMORY.md` + dosya tabanlı hafıza, ACE'nin elle işletilen, insan onaylı versiyonudur. Fark: ACE'de reflection/curation'ı model yapıyor, burada kullanıcı onaylıyor.

---

## K3 — Progressive disclosure: skill ve programatik tool çağrısı (2025 Q3 – Q4)

**Neye cevap:** K1 "bağlamı küçük tut" dedi ama **yetenek eklemek bağlamı büyütüyordu.** 30 tool şeması + 20K sistem promptu = daha başlamadan dolu bağlam.

Çözüm, bilgiyi **kademeli açığa çıkarmak**:

| Katman | İçerik | Ne zaman yüklenir |
|---|---|---|
| **0** | Aranabilir skill/tool indeksi | Hiç — sorgulanır |
| **1** | `name` + `description` | Her zaman (~50 token/skill) |
| **2** | `SKILL.md` gövdesi | Model çağırınca (~5K) |
| **3** | Referans dosyalar, script'ler | Model o dosyayı okuyunca |

Aynı fikrin tool tarafındaki karşılığı **`defer_loading`**: tool'un yalnızca *adı* bağlamda durur, şeması `ToolSearch` ile istendiğinde gelir. Bu oturumda tam olarak bu çalışıyor — §B.4.

**Programmatic tool calling (PTC)** ise farklı bir eksende çözüyor: tool çağrılarını modelin bağlamında değil, **sandbox'ta kod olarak** yürütüyor. 20 tool çağrısının ara çıktısı bağlama hiç girmiyor; yalnızca son sonuç giriyor.

> **Ayrım (Glean):** PTC ve subagent **uzaysal**; compaction ve context editing **zamansal** bağlam yönetimidir. §08.8.

### Bu katmanın kendi sınırı

Katman 1 açıklamaları da bağlam yiyor. 200 skill × 50 token = 10K token, hiçbiri kullanılmadan. Buna cevap Katman 0 (aranabilir indeks) — ama o da bir arama tool'u maliyeti getiriyor. **Progressive disclosure kendi ölçek sınırına ulaştı** (§10, Bulgu 15).

---

## K2 — Ajanik arama vektör RAG'i devirdi (2025 Q2)

**Neye cevap:** RAG v1 (K0) kod üzerinde kötü çalışıyordu — chunk sınırları fonksiyonları kesiyor, embedding "authentication" ile `def login()` arasındaki bağı yakalayamıyor, indeks dosya değişince bayatlıyor.

**Mayıs 2025:** Anthropic, Claude Code'dan vektör aramayı kaldırdı. Embedding boru hattı, yerel vektör veritabanı ve chunk'lama sezgiselleri **grep ile değiştirildi.**

> Claude Code'un yaratıcısı Boris Cherny: sonuç *"her şeyden daha iyi performans gösterdi. Hem de çok."*

**Sektör bunu takip etti:** Cursor bu kararın arkasındaki mühendisleri işe aldı; Windsurf, Cline, Devin ve Sourcegraph Amp vektörleri bırakıp tool tabanlı aramaya geçti. AAAI 2026'daki bir Amazon Science çalışması, ajanik anahtar kelime aramasını **sıfır vektör deposuyla RAG sadakatinin %94,5'i** olarak ölçtü.

### Neden çalışıyor

| | Vektör RAG | Ajanik arama |
|---|---|---|
| Ne zaman indekslenir | Önceden | Hiç |
| Bayatlar mı | Evet | Hayır — canlı dosya sistemi |
| Kesinlik | Yaklaşık (anlamsal) | Tam (string) |
| Ajan neyi görür | Chunk'lar | Yapı → daralt → oku |
| Yeniden kullanım | Sorgular bağımsız | **Her arama bir sonrakini bilgilendirir** |

Son satır belirleyici: ajan grep sonucuna bakıp **bir sonraki grep'i düzeltiyor.** Tek atışlık retrieval bunu yapamaz. §06.

### Ama sınırı var

> Grep, **doğrudan keşfedilebilecek kadar küçük** bir kod deposunda spesifik bir şey ararken çalışır. RAG, bilgi **dışsal, bağlama sığmayacak kadar büyük ve anlamla eşleşiyorsa** çalışır.

Yani K2, K6'yı geçersiz kılmıyor — ikisi farklı veri türleri için. Kod deposu → grep. Kurumsal bilgi → grafik/semantik katman. 10.000 PDF → hibrit (§07).

---

## K1 — "Context engineering" adını aldı (2025 Q3)

**Neye cevap:** Prompt engineering tek atışlık görevler için yeterliydi; çok turlu ajanlarda değildi. Sorun promptun ifadesi değil, **turlar boyunca bağlamda ne biriktiğiydi.**

Bu katmanın getirdiği kavramlar:

- **Dikkat bütçesi** — bağlam sonlu bir kaynak; her token diğerlerinin payını azaltır
- **Context rot** — bağlam uzadıkça, ilgili bilgi orada *olsa bile*, hatırlama bozulur
- **JIT retrieval** — önden yükleme değil, gerektiğinde getirme
- **Üçlü savunma** — compaction, note-taking, subagent
- **En küçük yeterli token kümesi** — hedef "çok bilgi" değil, "doğru bilgi"

§01 ve §08'in temeli budur.

### Tarihsel not

Bu katmanı tanımlayan Anthropic yazısı (Eylül 2025) bazı şeyleri **problem** olarak tanımladı — şişkin tool setleri, tool çıktısı birikimi. Bu problemlerin API seviyesindeki çözümleri (`defer_loading`, tool search, sunucu tarafı compaction) **yazıdan sonra** ürünleşti. Yani K1'in teşhisi K3 ve K5'i doğurdu.

---

## K0 — ReAct, prompt engineering, RAG v1 (2022–2023)

Başlangıç noktası:

- **ReAct** (Yao et al., 2022): `Thought → Action → Observation` döngüsü, **prompt içinde metin olarak** kurgulanmış. `stop=["\nObservation:"]` ile kesip regex ile parse et.
- **Prompt engineering:** tek bir metnin ifadesini optimize et.
- **RAG v1:** chunk → embed → sorgu embed → kosinüs benzerliği → en yakın k → prompta yapıştır.

Bugün bunların **fikirleri** yaşıyor, **mekanizmaları** yok:

| ReAct'ten kalan | ReAct'ten gitmiş |
|---|---|
| Akıl yürüt → eyle → gözlemle döngüsü | Metin formatı, stop sequence, regex parse |
| Gözlemin bir sonraki adımı beslemesi | `Thought:`/`Action:` etiketleri, tek tool/tur |

Yerini alan: **native tool calling** — `tool_use` içerik blokları, `stop_reason: "tool_use"`, yapısal JSON, paralel çağrı. §03.

---

## Değişmeyen çekirdek

Yedi katman değişti; şu değişmedi:

> **Model durumsuzdur. Her turda bağlam sıfırdan yeniden inşa edilir.**
>
> Modelin hafızası, araçları, becerileri, süreklilik hissi yoktur. Bunların hepsi harness'in her istekte yeniden ürettiği token dizisidir. Ajan tasarlamak, **sonlu bir dikkat bütçesine neyin gireceğine karar vermektir.**

K0'dan K7'ye kadar her katman bu tek cümlenin farklı bir sonucudur:

| Katman | Aynı cümlenin hangi sonucu |
|---|---|
| K0 | Döngüyü kur |
| K1 | Bütçenin sonlu olduğunu fark et |
| K2 | Bütçeyi önden değil, anında doldur |
| K3 | Yeteneği bütçe harcamadan ekle |
| K4 | Bütçe dışında öğrenen bir depo tut |
| K5 | Bütçeye giren şeyi göreve göre sıkıştır |
| K6 | Bütçeye girenin **anlamını** yapılandır |
| K7 | Bütün bunları bir sistem olarak tasarla ve ölç |

---

## A.8 — Alandaki pratik araçlar

Kullanıcının kaynak listesindeki token azaltma depoları, yukarıdaki katmanların **ürünleşmiş** hâlleridir. Hangi katmana ait olduklarıyla:

| Araç | Tekniği | Katman | İddia edilen azalma |
|---|---|---|---|
| **RTK** (Rust Token Killer) | Terminal çıktısını bağlama girmeden filtrele | K5 | %60–90 |
| **Context Mode** | Playwright/GitHub tool çıktısını SQLite'a boşalt, sohbete yalnızca özet geçir | K5 + PTC | %98 |
| **code-review-graph** | Tree-sitter ile yerel bilgi grafiği; yalnızca gereken düğüm okunur | K6 | 49× (monorepo) |
| **Token Savior** | Kodu dosya değil **sembol** düzeyinde referansla | K6 | %97 |
| **token-optimizer-mcp** | Tekrarlayan tool çıktılarını önbellekle ve sıkıştır | K5 | %95+ |
| **claude-context** (Zilliz) | BM25 + vektör hibrit arama | K2/K6 hibrit | ~%40 |
| **token-optimizer** | "Hayalet token" tespiti — sessizce bağlam yiyen şeyler | K1 ölçüm | — |
| **claude-token-efficient** | Yalnızca `CLAUDE.md` ile çıktıyı kısa tut | K1 | — |
| **Caveman Claude** | Çıktı üslubunu kısaltarak output token'ı kes | Çıktı tarafı | %65–75 |

> ⚠️ **Bu yüzdelerin hepsi depo sahiplerinin iddiasıdır**, bağımsız ölçüm değil. Yöntem şeffaf değil (hangi iş yükü, hangi taban çizgisi). §09'daki probe tabanlı değerlendirme ile doğrulanmadan üretimde kullanılmamalı — özellikle **agresif filtrelemenin kalite maliyeti** ölçülmeli: K5 araştırmasının uyardığı gibi, kritik hata string'lerini atmak token tasarrufundan pahalıya patlar.

**Desen olarak okunması gereken:** dokuz aracın yedisi aynı iki şeyi yapıyor — (1) tool çıktısını bağlama girmeden önce kes, (2) tam içerik yerine referans/sembol geçir. Bu, raporun ana tezinin pazar tarafından doğrulanması.

---
---

# BÖLÜM B — Harness Atlası: her yapı, uçtan uca

Her yapı aynı şablonla: **ne çözer → nasıl tanımlanır → modele tam olarak ne gider → model ne üretir → harness ne yapar → maliyet.**

Sıra, bir isteğin **wire üzerindeki render sırasıdır**: `tools → system → messages`.

---

## B.1 — Sistem promptu ve ortam bloğu

**Ne çözer:** Modelin kim olduğu, nerede çalıştığı, neye izinli olduğu. Süreklilik hissinin kaynağı.

**Nasıl tanımlanır:** Harness tarafından; kullanıcı `CLAUDE.md` ile katkı verir.

**Modele giden tam form** — `system` alanı, **her istekte**:

```json
"system": [
  {"type": "text",
   "text": "You are Claude Code, Anthropic's official CLI...",
   "cache_control": {"type": "ephemeral"}},
  {"type": "text",
   "text": "# Environment\nPrimary working directory: /home/altan/Desktop/adapted\nIs a git repository: true\nPlatform: linux\n...\ngitStatus: Current branch: main\nRecent commits:\nc29f2fd Initialize repo..."}
]
```

**Model ne üretir:** Doğrudan bir şey değil — davranışı şekillendirir. Ama **dolaylı olarak kritik**: `gitStatus` bloğu olmasa model hangi branch'te olduğunu sormak zorunda kalırdı. Bu blok, bir tool çağrısının yerini tutuyor.

**Maliyet:** ~2–5K token, **her turda**. Cache prefix'inin başında olduğu için cache hit'te ~%10 fiyatla okunur.

**Tasarım kuralı:** buraya konan her satır, oturumun her turunda ödenir. Test: *"model bunu 2 tool çağrısıyla kendisi bulabilir mi?"* Evet ise buraya koyma — §04'teki `CLAUDE.md` budama kuralı.

---

## B.2 — CLAUDE.md ve proje hafızası

**Ne çözer:** Depoya özgü, koddan türetilemeyen bilgi.

**Nasıl tanımlanır:** `CLAUDE.md` (kök, her zaman), `<altdizin>/CLAUDE.md` (o dizinde çalışılınca), `.claude/rules/*.md` (`paths` frontmatter'ı ile koşullu).

**Modele giden form:** Sistem promptuna veya ilk kullanıcı mesajına metin olarak enjekte edilir.

**Ne konur / konmaz:**

| Konur (türetilemez) | Konmaz (türetilebilir) |
|---|---|
| Tuzaklar: "X güvenli görünür ama Y yapar" | Dizin yapısı (`ls` söyler) |
| Tasarım gerekçesi | Bağımlılık listesi (manifest söyler) |
| **Varsayılandan sapan** konvansiyonlar | Standart build komutları |
| Güvenlik yasakları: "asla main'e push etme" | "Temiz kod yaz" türü genel öğütler |
| Tahmin edilemeyen komutlar/flag'ler | Lint/CI'ın zaten zorladığı kurallar |

**Maliyet:** Kök dosya her turda. ~40.000 karakteri aşınca harness uyarı veriyor.

---

## B.3 — Tool (yerleşik)

**Ne çözer:** Modelin dış dünyaya tek erişimi.

**Nasıl tanımlanır:** Üç eşdeğer yol — hepsi aynı JSON Schema'ya iner:

```python
# 1. Ham JSON
{"name": "get_weather",
 "description": "Bir şehrin güncel hava durumunu döndürür.",
 "input_schema": {"type":"object",
                  "properties":{"city":{"type":"string","description":"Şehir adı"}},
                  "required":["city"]}}

# 2. Dekoratör — imza + docstring'den şema üretilir
@beta_tool
def get_weather(city: str) -> str:
    """Bir şehrin güncel hava durumunu döndürür.

    Args:
        city: Şehir adı
    """
    return api.fetch(city)

# 3. Zod / Pydantic — tip nesnesinden şema üretilir
```

**Modele giden tam form** — `tools` dizisi, **sistem promptundan ÖNCE**, her istekte:

```json
"tools": [
  {"name":"Read","description":"Reads a file from the local filesystem...",
   "input_schema":{"type":"object","properties":{
      "file_path":{"type":"string","description":"The absolute path..."},
      "limit":{"type":"integer"},"offset":{"type":"integer"}},
    "required":["file_path"]}},
  {"name":"Bash", "...": "..."}
]
```

**Model ne üretir:**

```json
{"role":"assistant",
 "content":[
   {"type":"thinking","thinking":"Dosyayı okumam gerek..."},
   {"type":"text","text":"Dosyayı okuyorum."},
   {"type":"tool_use","id":"toolu_01A9...","name":"Read",
    "input":{"file_path":"/home/altan/Desktop/adapted/report/00-README.md"}}],
 "stop_reason":"tool_use"}
```

**Harness ne yapar:** `stop_reason == "tool_use"` görür → üretimi durdurur → `name`e karşılık gelen fonksiyonu `input` ile çağırır → sonucu **yeni bir user mesajı** olarak ekler:

```json
{"role":"user",
 "content":[{"type":"tool_result",
             "tool_use_id":"toolu_01A9...",
             "content":"     1\t# Bağlam Mühendisliği...\n     2\t..."}]}
```

Sonra **tüm mesaj dizisini baştan** tekrar gönderir. Model hiçbir zaman "dosyayı okumaz" — okunmuş hâlini bağlamında bulur.

**Hata da aynı kanaldan döner:**

```json
{"type":"tool_result","tool_use_id":"toolu_01A9...",
 "content":"ENOENT: no such file or directory","is_error":true}
```

Bu **kritik bir tasarım noktası**: hata mesajı modelin bir sonraki denemesini besleyen tek sinyaldir. `Error: failed` arama uzayını daraltmaz; `ENOENT ... Did you mean: readme.md?` daraltır.

**Maliyet:** Şema ~200–800 token/tool, **her turda**. Sonuçlar birikir — asıl maliyet burada.

**Kurallar:**
- Açıklama, aracı hiç görmemiş bir stajyere yeter mi? Test bu.
- 10–15 tool üstünde seçim doğruluğu düşer → `defer_loading` (B.4) veya subagent (B.10)
- Örtüşen tool'lar (`search_files` + `find_files`) yanlış seçim üretir

---

## B.4 — Ertelenmiş tool (`defer_loading` + ToolSearch)

**Ne çözer:** 50 tool'un şeması 40K token eder; hiçbiri kullanılmasa bile ödenir.

**Nasıl tanımlanır:**

```json
{"name":"WebFetch","description":"...","input_schema":{...},
 "defer_loading": true}
```

**Modele giden form — bu oturumdaki canlı kanıt.** Şema yerine yalnızca isim listesi bir `<system-reminder>` içinde geliyor:

```
The following deferred tools are now available via ToolSearch. Their schemas
are NOT loaded — calling them directly will fail with InputValidationError.
Use ToolSearch with query "select:<name>[,<name>...]" to load tool schemas:
CronCreate, CronDelete, CronList, DesignSync, EnterPlanMode, ...,
WebFetch, WebSearch
```

**Model ne üretir:**

```json
{"type":"tool_use","name":"ToolSearch",
 "input":{"query":"select:WebSearch,WebFetch,TodoWrite","max_results":3}}
```

**Harness ne yapar:** Eşleşen tool'ların tam tanımını bir `<functions>` bloğu içinde `tool_result` olarak döndürür. O andan itibaren tool normal şekilde çağrılabilir.

> **Bu bölümü yazarken tam olarak bu oldu:** `WebSearch`, `WebFetch` ve `TodoWrite` ertelenmişti; `ToolSearch` ile yüklendiler, sonra kullanıldılar.

**Maliyet:** İsim ~5 token/tool (şema ~500 yerine). Bedeli: ihtiyaç anında **bir ekstra tur**.

**Karar:** sık kullanılan tool'ları yerleşik bırak, uzun kuyruğu ertele.

---

## B.5 — Skill

**Ne çözer:** Uzun, göreve özgü talimatları her turda ödemeden kullanılabilir kılmak.

**Nasıl tanımlanır:** `.claude/skills/<ad>/SKILL.md`

```markdown
---
name: pdf-isleme
description: PDF'ten metin/tablo çıkarma, sayfa seçimi, taranmış PDF tespiti.
  Kullanıcı bir .pdf dosyasıyla çalışmak istediğinde kullan.
---

# PDF İşleme
[... 5.000 token talimat ...]

Ayrıntı için `reference/tablo-cikarma.md` dosyasına bak.
```

**Modele giden form — katman katman:**

*Katman 1* (her zaman, ~50 token):
```
### Skill: pdf-isleme
PDF'ten metin/tablo çıkarma, sayfa seçimi, taranmış PDF tespiti.
Kullanıcı bir .pdf dosyasıyla çalışmak istediğinde kullan.
```

*Katman 2* — model `{"type":"tool_use","name":"Skill","input":{"skill":"pdf-isleme"}}` üretince, `SKILL.md`'nin **tam gövdesi** bir `tool_result` olarak bağlama girer (~5K token).

*Katman 3* — model gövdedeki referansı okumaya karar verirse `Read("reference/tablo-cikarma.md")` çağırır.

> **Bu oturumdaki canlı kanıt:** `claude-api` skill'i tetiklendi ve ~50K token'lık gövdesi enjekte edildi. Enjeksiyondan **önce** modelin bir `grep` çalıştırdığı gözlemlendi — yani skill gövdesi, keşif kararının kendisini değil, sonrasını yönlendirdi.

**Maliyet:** Katman 1 sürekli, katman 2 koşullu. Skill listesi bağlam penceresinin ~%1'i ile bütçelendirilmiştir; aşılınca girişler kırpılır ve **skill yönlendirmesi bozulur.**

**En kritik alan `description`:** modelin bu skill'i çağırıp çağırmayacağına karar vermek için gördüğü **tek** şey odur. Ne yaptığını *ve ne zaman kullanılacağını* söylemelidir.

---

## B.6 — Memory (dosya tabanlı)

**Ne çözer:** Oturumlar arası süreklilik — bağlam penceresi sıfırlansa bile kalan bilgi.

**Nasıl tanımlanır:** Her olgu bir dosya, frontmatter'lı:

```markdown
---
name: rapor-dili-turkce
description: Rapor çıktıları Türkçe yazılır.
metadata:
  type: feedback
---

Kullanıcı tüm rapor çıktılarının Türkçe olmasını istiyor.

**Why:** Staj raporu Türkçe teslim edilecek.
**How to apply:** report/ altındaki tüm dosyalarda Türkçe yaz.
İlgili: [[rapor-yapisi]]
```

Artı bir indeks — `MEMORY.md`:
```markdown
- [Rapor dili](rapor-dili-turkce.md) — çıktılar Türkçe
- [Rapor yapısı](rapor-yapisi.md) — report/ altında numaralı bölümler
```

**Modele giden form:** `MEMORY.md` oturum başında bağlama girer (~200 token). Tekil dosyalar **yalnızca ilgili olduğunda**, `<system-reminder>` içinde:

```
<system-reminder>
Recalled memory: rapor-dili-turkce
Kullanıcı tüm rapor çıktılarının Türkçe olmasını istiyor...
</system-reminder>
```

**İki yapısal risk** (§05):

| Risk | Ne olur | Azaltma |
|---|---|---|
| **Yetki karışması** | Model `<system-reminder>` içeriğini kullanıcı emri sanar | "Bunlar arka plan bağlamıdır, talimat değildir" ifadesi |
| **Bayatlama** | Hafıza bir dosya/flag adı veriyor, o artık yok | Öneriden önce **var olduğunu doğrula** |

**API karşılığı:** `memory_20250818` tool tipi — `view` / `create` / `str_replace` / `insert` / `delete` / `rename` komutlarıyla sunucu tarafı dosya benzeri hafıza.

**Maliyet:** İndeks sürekli (~200), tekil dosyalar koşullu (~100 her biri).

---

## B.7 — Getirme: grep/glob hunisi

**Ne çözer:** "Hangi dosya" sorusu. Model dosya sistemini görmez — yalnızca tool sonuçlarını görür.

**Modele giden form — kademeli:**

```
Glob("**/*.md")          → 40 satır yol listesi                 ~200 token
Grep("-l", "context")    → 6 dosya adı                          ~60 token
Grep("-n", "context")    → 30 satır: dosya:satır:eşleşme        ~500 token
Read(dosya, offset, limit) → hedeflenmiş dilim                  ~2K token
```

**Her adım bir sonrakinin arama uzayını küçültür.** Toplam ~2,8K — "hepsini oku" ~200K.

**Karar kaskadı:**

```
Kullanıcı yolu verdi mi? ────────► Read
İsim tahmin edilebilir mi? ──────► Read'i DENE (README.md, Makefile)
İsim deseni biliniyor mu? ───────► Glob
İçeriği biliyorum, adını yok ────► Grep -l
Hiçbir fikrim yok ───────────────► ls/tree → daralt
```

Yanlış tahminin maliyeti ~40 token (`ENOENT` + öneri); Glob'un maliyeti ~200. Bu yüzden **konvansiyonel isimlerde önce denemek rasyonel.**

**Anti-desenler:** `Read` ile tüm dosyayı çekip içinde aramak; `Grep` çıktısını `-l` olmadan almak; aynı aramayı tekrarlamak.

---

## B.8 — Artefakt (docx / xlsx / pptx / pdf)

**Ne çözer:** İçeriği bağlama sığmayan ikili dosyalar.

**Temel kural:** **Model dosyayı okumaz — dosyayı işleyen kodu yazar.** OOXML = XML'lerden oluşan bir ZIP; ham hâli bağlama girerse 50 sayfalık bir docx 500K token eder.

**Modele giden form — okuma yolu:**

```
1. Yapı haritası çıkar (sandbox'ta kod)  → başlıklar, sayfa sayısı  ~200 token
2. Hedefli bölümü çıkar                   → yalnızca o bölüm        ~2K token
```

**Düzenleme yolu:** cerrahi (tek çalıştırma değiştir) veya yeniden üretim (baştan yaz). Docx'te `runs` tuzağı: tek bir cümle biçimlendirme yüzünden 5 XML çalıştırmasına bölünmüş olabilir; naif `text.replace()` sessizce başarısız olur.

**PDF üç yol:**

| Yol | Bağlama giren | Maliyet |
|---|---|---|
| **Native PDF** (`document` bloğu) | Her sayfa: **metin + sayfanın görseli** | ~1.500–3.000 token/sayfa |
| **Sandbox'ta metin çıkarma** | Yalnızca `stdout` | ~500–800/sayfa |
| **Sayfa → görsel** | Yalnızca seçilen sayfa | ~1.500/görsel |

Native yolun tam formu:
```json
{"type":"document",
 "source":{"type":"base64","media_type":"application/pdf","data":"JVBER..."},
 "citations":{"enabled":true}}
```
`citations` açıkken yanıt parçalara bölünür ve atıflı parçalar `page_location` taşır — halüsinasyon doğrulanabilir hâle gelir.

**Sınırlar:** 32 MB istek, 600 sayfa (200K bağlamlı modellerde 100).

---

## B.9 — Programatik tool çağrısı (PTC)

**Ne çözer:** Çok adımlı tool zincirlerinin **ara çıktılarının** bağlamı doldurması.

**Nasıl tanımlanır:**

```json
{"name":"get_page_count","description":"...","input_schema":{...},
 "allowed_callers":["code_execution_20250825"]}
```

**Modele giden form:** Model tool'u tek tek çağırmak yerine **kod yazar**:

```python
results = []
for path in glob.glob("belgeler/**/*.pdf"):
    n = get_page_count(path)          # tool, sandbox'tan çağrılıyor
    if n > 100:
        results.append((path, n))
print(sorted(results)[:5])
```

**Bağlama giren:** yalnızca `print` çıktısı — 5 satır. 200 tool çağrısının 200 sonucu **girmedi.**

**Ekstra fayda:** hata olursa sandbox **stack trace** döndürür; model hatayı kod düzeyinde düzeltip yeniden çalıştırır — her denemeye bir tur harcamadan.

**Ne zaman:** çıktısı büyük, adımı çok, ara sonucu ilgisiz zincirler. **Ne zaman değil:** tek çağrı, veya her ara sonucun modelin kararını değiştirdiği durumlar.

---

## B.10 — Subagent

**Ne çözer:** Bir alt görevin tüm keşif gürültüsünü ana bağlamdan uzak tutmak. **Uzaysal** bağlam yönetimi.

**Modele giden form:**

```json
{"type":"tool_use","name":"Agent",
 "input":{"subagent_type":"Explore",
          "description":"PDF işleme kodunu bul",
          "prompt":"Depoda PDF okuyan/yazan tüm kod yollarını bul, dosya:satır ver."}}
```

**Ne olur:** Subagent **kendi temiz bağlam penceresinde** çalışır — 40 tool çağrısı yapar, 60K token harcar.

**Ana bağlama dönen:**

```json
{"type":"tool_result","tool_use_id":"...",
 "content":"3 yol bulundu:\n- src/ingest/pdf.py:41 — pdfplumber ile metin\n- src/api/upload.py:88 — sayfa sayısı doğrulama\n- tests/test_pdf.py:12 — fixture"}
```

**~80 token. 60K girmedi.**

**Bedeli — ve bu bedel gerçek:** subagent **soğuk başlar.** Ana ajanın bildiği her şeyi yeniden türetmesi gerekir. Bu yüzden subagent, "birden fazla açıdan bakılması gereken" her iş için değil, **çıktısı özete indirgenebilen keşif işleri** için doğru araçtır.

---

## B.11 — Context editing (silme — zamansal)

**Ne çözer:** Biriken tool sonuçları. Silme, özetleme değil.

**Nasıl tanımlanır:**

```json
"context_management": {
  "edits": [
    {"type":"clear_tool_uses_20250919",
     "trigger":{"type":"input_tokens","value":100000},
     "keep":{"type":"tool_uses","value":3},
     "clear_at_least":{"type":"input_tokens","value":5000}},
    {"type":"clear_thinking_20251015"}
  ]}
```

**Modele giden form:** Eski `tool_result` blokları yerlerinde **yer tutucuya** dönüşür; mesaj dizisinin yapısı bozulmaz.

**Tehlike:** silinen sonuçta modelin hâlâ ihtiyacı olan bir olgu varsa **sessizce kaybolur.** Bu yüzden model, kritik bulguları silinmeyen bir yere yazmalı — dosyaya veya metin bloğuna. §08.

---

## B.12 — Compaction (özetleme — zamansal)

**Ne çözer:** Bağlam penceresinin dolması.

```json
"context_management": {"edits": [{"type":"compact_20260112"}]}
```

**Ne olur:** Konuşmanın tamamı veya bir kısmı özetle değiştirilir. Model yeni pencerede özeti + kalan ham bağlamı görür.

**Neyin taşınması gerektiği** (bu oturumun kendi compaction'ından çıkan ders):

| Taşınmalı | Neden |
|---|---|
| Kullanıcının **birebir** istekleri | Yeniden yorum kayma üretir |
| **Dosya yolları ve kod parçaları** | Yeniden bulmak tur maliyeti |
| **Başarısız olan** yaklaşımlar | Yoksa model aynı hatayı tekrarlar |
| Açık kalan sorular | Kaybolursa iş yarım kalır |
| Son eylem ve sıradaki adım | Sürekliliğin kendisi |

Son satırdan öncesi kritik: **başarısızlıkları taşımayan özet, ajanı döngüye sokar.**

**ACE'nin uyarısı burada geçerli** (K4): özetin özetini almak *context collapse* üretir. Mümkünse özet **yeniden yazılmamalı, artımlı güncellenmelidir.**

---

## B.13 — Prompt cache

**Ne çözer:** Her turda aynı 50K token'ın tam fiyatla yeniden okunması.

**Nasıl tanımlanır:** Kararlı prefix'in sonuna işaret koy:

```json
{"type":"text","text":"<sistem promptu>","cache_control":{"type":"ephemeral"}}
```

**Nasıl çalışır:** **Tam prefix eşleşmesi.** Prefix'in başında bir token değişirse sonrasının tamamı geçersizleşir.

```
tools → system → messages
  ↑        ↑         ↑
  └── burada bir değişiklik, sağındaki HER ŞEYİ geçersiz kılar
```

**Sessiz geçersizleştiriciler:** sistem promptuna zaman damgası koymak, tool sırasını değiştirmek, `MEMORY.md`'yi her turda yeniden üretmek.

**Kritik okuma tuzağı:** `usage.input_tokens` **yalnızca cache'lenmemiş kalanı** gösterir. Gerçek toplam:

```
toplam = input_tokens + cache_read_input_tokens + cache_creation_input_tokens
```

Bunu bilmeden "bağlamım küçük" sanmak yaygın bir hata.

**Yerleşim kuralı:** en kararlıdan en değişkene sırala — `tools` → `system` → `CLAUDE.md` → sabit örnekler → **cache işareti** → değişken konuşma.

---

## B.14 — Hook

**Ne çözer:** Harness'e model dışı, deterministik kontrol eklemek.

**Türleri:** `PreToolUse` (çağrı öncesi — engelleyebilir), `PostToolUse`, `UserPromptSubmit`, `SessionStart`, `Stop`.

**Modele giden form:** Hook çıktısı **kullanıcı geri bildirimi gibi** bağlama girer:

```
<system-reminder>
A session-scoped Stop hook is now active with condition: "..."
The hook will block stopping until the condition holds.
</system-reminder>
```

> **Bu oturumdaki canlı kanıt:** kullanıcı `/goal` çalıştırdı; bir `Stop` hook'u devreye girdi ve hedef karşılanana kadar oturumun durmasını engelliyor. Bu bölüm o hook'un koşulu altında yazıldı.

**Neden önemli:** hook, harness'in **modele güvenmediği** yerdir. Model bir kuralı unutabilir; hook unutamaz. Güvenlik kritik kısıtlar (asla `main`'e push etme, üretim veritabanına dokunma) prompta değil hook'a yazılır.

**Maliyet uyarısı:** `PreToolUse`/`PostToolUse` **her tool çağrısında** döngüyü bloklar. >2 sn tipik süre ajanı hissedilir yavaşlatır.

---

## B.15 — Özet tablo: her yapı ne zaman

| Yapı | Ekseni | Ne zaman kullan | Bağlam maliyeti |
|---|---|---|---|
| Sistem promptu | Statik | Her zaman | 2–5K sürekli |
| CLAUDE.md | Statik | Türetilemez depo bilgisi | Boyutu kadar sürekli |
| Tool | Statik + dinamik | Dış dünya erişimi | 200–800/tool sürekli |
| `defer_loading` | Statik | >15 tool | 5/tool sürekli + 1 tur |
| Skill | Koşullu | Uzun, göreve özgü talimat | 50 sürekli + 5K koşullu |
| Memory | Kalıcı | Oturumlar arası olgular | 200 sürekli + 100 koşullu |
| Grep hunisi | Dinamik | "Hangi dosya" | Her adım artımlı |
| Artefakt kodu | Dinamik | Bağlama sığmayan dosya | Yalnızca çıkarılan |
| **PTC** | **Uzaysal** | Çok adımlı, gürültülü zincir | Yalnızca son çıktı |
| **Subagent** | **Uzaysal** | Özete indirgenebilir keşif | Yalnızca özet (+soğuk başlangıç) |
| **Context editing** | **Zamansal** | Tool sonucu birikimi | Negatif (siler) |
| **Compaction** | **Zamansal** | Pencere dolması | Negatif (özetler) |
| Prompt cache | Ekonomik | Kararlı prefix | Maliyeti ~%10'a indirir |
| Hook | Kontrol | Modele güvenilemeyecek kısıt | Çıktısı kadar |

---
---

# BÖLÜM C — Uçtan uca tek tur

Yukarıdaki on beş yapı ayrı ayrı değil, **aynı anda** çalışır. Tek bir kullanıcı mesajının tam yaşam döngüsü:

```
┌─ T0: KULLANICI YAZAR ────────────────────────────────────────────┐
│  "raporu güncelle"                                               │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌─ T1: HARNESS BAĞLAMI İNŞA EDER (her turda SIFIRDAN) ─────────────┐
│  tools:    [yerleşik şemalar] + [ertelenmiş isim listesi]        │
│  system:   [kimlik] [ortam+git] [CLAUDE.md] [MEMORY.md]          │
│            [skill katman-1 listesi] [aktif hook bildirimi]       │
│  messages: [önceki turların tamamı] + [yeni kullanıcı mesajı]    │
│            ← cache_control işareti burada                        │
└──────────────────────────────────────────────────────────────────┘
                            ↓  POST /v1/messages
┌─ T2: MODEL ÜRETİR ───────────────────────────────────────────────┐
│  thinking → text → tool_use(Grep)                                │
│  stop_reason: "tool_use"                                         │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌─ T3: HARNESS YÜRÜTÜR ────────────────────────────────────────────┐
│  PreToolUse hook → izin kontrolü → grep çalıştır                 │
│  Çıktı 25K token'ı aşarsa KIRP + kırpma notu ekle                │
│  → tool_result olarak user mesajına ekle                         │
└──────────────────────────────────────────────────────────────────┘
                            ↓
                  ┌─────────────────────┐
                  │  T1'E GERİ DÖN      │  ← döngü burada
                  │  (tüm dizi yeniden) │
                  └─────────────────────┘
                            ↓
┌─ T4: BASINÇ EŞİĞİ AŞILIRSA ──────────────────────────────────────┐
│  input_tokens > 100K → clear_tool_uses (eskiler yer tutucuya)    │
│  pencere dolmaya yakın → compact (özetle, yeniden başla)         │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌─ T5: MODEL BİTİRİR ──────────────────────────────────────────────┐
│  stop_reason: "end_turn"                                         │
│  → Stop hook koşulu kontrol edilir                               │
│  → sağlanmadıysa: durma engellenir, T1'e dön                     │
└──────────────────────────────────────────────────────────────────┘
```

### Tek turda ödenen fatura (temsilî, 20. tur)

| Bileşen | Token | Cache? |
|---|---|---|
| Tool şemaları (yerleşik) | 6.000 | ✅ okunur |
| Ertelenmiş tool isimleri | 200 | ✅ |
| Sistem promptu + ortam | 3.500 | ✅ |
| CLAUDE.md + MEMORY.md | 900 | ✅ |
| Skill katman-1 listesi | 800 | ✅ |
| Enjekte edilmiş skill gövdesi | 12.000 | ✅ (girdikten sonra) |
| Konuşma geçmişi (19 tur) | 48.000 | kısmen |
| Yeni kullanıcı mesajı | 40 | ❌ |
| **Toplam girdi** | **~71.400** | |
| **Faturalanan** (cache hit) | **~7.500 eşdeğer** | |

> ⚠️ Bu tablo **temsilîdir**, ölçüm değil. §09'daki wire log script'i ile gerçek değerler çıkarılabilir.

**Okunacak ders:** 71K token'ın 70K'sı kullanıcının yazdığı 40 token'ı *anlamlandırmak için* orada. İşin tamamı bu 70K'nın kompozisyonundadır — modelde değil.

---

## C.2 — 2026 kontrol listesi

Bir ajan sistemi kurarken, katman katman:

**Temel (K1)**
- [ ] Sistem promptundaki her satır "model bunu kendisi bulabilir mi?" testinden geçti mi?
- [ ] `usage` üç alanı birden mi okunuyor (cache dâhil), yoksa yalnızca `input_tokens` mu?
- [ ] Tool sayısı 15'in altında mı? Değilse `defer_loading` var mı?

**Getirme (K2)**
- [ ] Kod deposu araması grep tabanlı mı, yoksa bayatlayan bir vektör indeksi mi?
- [ ] Getirme kademeli mi (Glob → Grep -l → Grep -n → Read offset), yoksa tek atış mı?

**Ölçeklenme (K3)**
- [ ] Uzun talimatlar skill'e taşındı mı? `description` alanları "ne zaman kullanılacağını" söylüyor mu?
- [ ] Çok adımlı gürültülü zincirler PTC veya subagent ile izole edildi mi?

**Öğrenme (K4)**
- [ ] Oturumlar arası taşınması gereken olgular için bir hafıza katmanı var mı?
- [ ] Hafıza **artımlı** mı güncelleniyor, yoksa her seferinde yeniden mi yazılıyor? (context collapse)

**Basınç (K5)**
- [ ] Compaction özeti **başarısız yaklaşımları** taşıyor mu?
- [ ] Tool çıktısı kırpması kritik string'leri (tam hata, tam yol) koruyor mu?

**Anlam (K6)**
- [ ] Kurumsal veri varsa: ontoloji/semantik katman gerekiyor mu, yoksa grep yetiyor mu?
- [ ] Tam ontoloji kurmayı beklemek yerine mevcut metadata'dan bootstrap ediliyor mu?

**Sistem (K7)**
- [ ] Güvenlik kritik kısıtlar prompta mı yazılı, hook'a mı? (Hook olmalı.)
- [ ] Performans "model X" olarak mı raporlanıyor, "model X + harness Y" olarak mı?
- [ ] Probe tabanlı değerlendirme var mı, yoksa yalnızca son çıktı mı ölçülüyor? (§09)

---

## C.3 — Üç cümlelik sonuç

1. **Model durumsuzdur; ajan yeteneği harness'in her turda kurduğu bağlamın fonksiyonudur.** Yedi katmanlık evrimin tamamı bu tek olgunun sonuçlarıdır.

2. **2026'nın ana hareketi, bağlam kararlarının elden makineye geçmesidir:** hangi dosyanın açılacağı (ajanik arama), neyin sıkıştırılacağı (öğrenilmiş sıkıştırma), neyin hatırlanacağı (ACE) artık kural değil, öğrenilen fonksiyon.

3. **Ama ölçme hâlâ elde.** Sıkıştırma yüzdeleri satıcı iddiası, harness farkları benchmark bağımlı, ontoloji projeleri kapsam altında ölüyor. §09'daki probe yaklaşımı olmadan bu bölümdeki hiçbir sayı üretim kararına dayanak olmamalıdır.

---

## Kaynaklar (bu bölüm)

**Birincil / akademik**
- [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618) — ACE, K4
- [ACON: Optimizing Context Compression for Long-horizon LLM Agents](https://arxiv.org/pdf/2510.00615) — K5
- [Squeez: Task-Conditioned Tool-Output Pruning for Coding Agents](https://arxiv.org/pdf/2604.04979) — K5
- [Self-GC: Self-Governing Context for Long-Horizon LLM Agents](https://arxiv.org/pdf/2607.00692) — K5
- [A Self-Evolving Framework for Efficient Terminal Agents via Observational Context Compression](https://arxiv.org/pdf/2604.19572) — K5
- [Harness-Aware Self-Evolving: Co-Evolving Model Weights, Harness, and Task Solutions](https://arxiv.org/pdf/2607.03935) — K7
- [Agent Harness Engineering: A Survey](https://picrew.github.io/LLM-Harness/) — K7
- [Awesome-Agent-Context-Compression](https://github.com/YerbaPage/Awesome-Agent-Context-Compression) — K5 envanteri

**İkincil ⚠️ *(birincilden doğrulanmalı)***
- [Agent Harness Engineering — Addy Osmani](https://addyosmani.com/blog/agent-harness-engineering/) — K7
- [Agent = Model + Harness — Cobus Greyling](https://cobusgreyling.medium.com/agent-model-harness-0d018f3d5014) — K7 tanımı
- [Agent Harnesses Beat Model Upgrades — MindStudio](https://www.mindstudio.ai/blog/agent-harnesses-beat-model-upgrades-5-benchmarks) — 23,8 puanlık harness farkı
- [RAG vs Agentic RAG: why search beats embeddings for code](https://explainx.ai/blog/rag-vs-agentic-rag-pageindex-2026) — K2, Claude Code'un Mayıs 2025 kararı
- [Code Retrieval: Grep, RAG, or Both?](https://medium.com/@jhanavibehl/code-retrieval-grep-rag-or-both-706cdefd0b70) — K2 sınırları
- [Ontologies, Context Graphs, and Semantic Layers: What AI Actually Needs in 2026](https://contextandchaos.substack.com/p/ontologies-context-graphs-and-semantic) — K6
- [What Is a Context Layer for AI Agents?](https://www.tellius.com/resources/blog/what-is-a-context-layer-for-ai-agents-the-definitive-guide-for-2026) — K6, Gartner ayrımı
- [Building a Context Pruning Pipeline for Long-Running Agents](https://machinelearningmastery.com/building-a-context-pruning-pipeline-for-long-running-agents/) — K5 pratiği
- [awesome-harness-engineering](https://github.com/ai-boost/awesome-harness-engineering) — K7 derlemesi

**Kullanıcı kaynak listeleri**
- `lists/context_eng_Alittlebitodlerthanothers.md` — token azaltma depoları (§A.8), context graph başlıkları
- `lists/agents.md` — harness/skill/memory literatür haritası

---

**← Önceki:** [10 — Sonuç](10-sonuc.md) · **Sonraki →** [12 — MCP ve modern yöntemler](12-mcp-ve-modern-yontemler.md) · **Ek:** [Ek A — Tool referansı](ek-a-tool-referans.md)
