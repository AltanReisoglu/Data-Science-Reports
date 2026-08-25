# Güvenilirlik Stratejileri — Ne Yaparlar, Nasıl Çalışırlar

Bir ajanın kontrolden çıkmasını önlemenin tek bir doğru yolu yok. Bu belgede on yedi
farklı yaklaşım var ve hepsi aynı soruya farklı cevaplar veriyor: **ajan ne zaman
durdurulmalı, ve bunu kim söylemeli?**

Neden hepsini birden uygulamıyoruz da ayrı ayrı seçilebilir yapıyoruz? Çünkü her
yaklaşımın yakaladığı hata farklı, kaçırdığı hata farklı, ve yanlış tetiklendiğinde
verdiği zarar farklı. Hangisinin sizin işinize uyduğunu ancak aynı görevde yan yana
koşturarak görebilirsiniz.

> **Numaralandırma uyarısı — iki belge, iki sıra.**
> Bu belge stratejileri **aileye göre** numaralıyor (A: kaynaklardan türeyen 1–11,
> B: harness'lardan türeyen 12–17). Bu sıra CLI kayıt defteriyle ve plan belgesiyle aynı.
> [`zihniyetler.md`](zihniyetler.md) ise aynı on yediyi **zorluğa göre** numaralıyor —
> en basitten en karmaşığa. Aynı stratejinin iki belgede farklı numarası var; `id` her
> ikisinde de aynı, referans verirken `id` kullanın.
>
> | id | bu belge | zihniyetler.md | seviye |
> |---|---:|---:|---|
> | `arize-control` | 2 | **1** | 1 · Sayaç |
> | `agentscope-grace` | 16 | **2** | 1 · Sayaç |
> | `hermes-no-pressure` | 13 | **3** | 1 · Sayaç |
> | `claude-advisory` | 6 | **4** | 1 · Sayaç |
> | `strands-entropy` | 15 | **5** | 2 · Pencere |
> | `openhands-stuck` | 12 | **6** | 2 · Pencere |
> | `openclaw-pingpong` | 14 | **7** | 2 · Pencere |
> | `loopguard-dignity` | 3 | **8** | 2 · Pencere |
> | `verify-gate` | 7 | **9** | 3 · Dünya |
> | `galileo-breaker` | 10 | **10** | 3 · Dünya |
> | `agentbudget-dollar` | 5 | **11** | 3 · Dünya |
> | `modexa-statemachine` | 4 | **12** | 4 · Şekil |
> | `autogen-static` | 17 | **13** | 4 · Şekil |
> | `telemetry-repair` | 8 | **14** | 5 · Kademe |
> | `pi-signature` | 1 | **15** | 5 · Kademe |
> | `voi-allocation` | 9 | **16** | 6 · Karar |
> | `improvement-loop` | 11 | **17** | 6 · Karar |
>
> Hangisini okumalı: **uygulama sırası** arıyorsanız zihniyetler.md, **hangi kaynaktan
> geldi** diye soruyorsanız bu belge.

---

## Önce: bir stratejiyi neyin farklı kıldığı

On yedi yaklaşımı dört soruyla ayırt edebilirsiniz:

**1 · Ne zaman araya giriyor?**
Kimi eylem yapılmadan önce durduruyor, kimi yapıldıktan sonra fark ediyor, kimi ancak
koşum bittiğinde "aslında bu iş olmamış" diyor.

**2 · Neye bakıyor?**
Kimi ajanın *ne yaptığına* bakıyor (aynı tıklamayı tekrar mı ediyor), kimi *ne olduğuna*
(ekran değişti mi), kimi *ne harcadığına* (kaç token gitti), kimi *ne söylediğine*
(aynı cümleyi mi kuruyor).

**3 · Nasıl karar veriyor?**
Kimi sabit bir sayı sayıyor ("üç kez tekrarladıysa dur"), kimi oran hesaplıyor ("bu
aracın çağrılarının yarısı hata veriyor"), kimi dış dünyaya soruyor ("testler geçti mi").

**4 · Tetiklendiğinde ne yapıyor?**
Kimi hemen kesiyor, kimi önce uyarıp bir şans daha veriyor, kimi geri sarıp tekrar
deniyor, kimi kullanıcıya soruyor.

Bu dört eksende farklı yerlerde duran her yaklaşım, ayrı bir strateji.

---

# A Ailesi — Kaynaklardan Türeyen Zihniyetler

## 1 · `pi-signature` — Çok sinyal, kademeli müdahale

**Kaynak:** `pi-anti-doom-loop` (yayımlanmış npm eklentisi)

**Zihniyet:** Tek bir akıllı dedektör aramak yerine, altı tane ucuz sinyali birden dinle.
Hiçbiri tek başına kesin değil ama birlikte güvenilirler.

**Nasıl çalışır.** Altı ayrı şeye bakıyor: aynı aracın aynı argümanlarla tekrarı, aynı
aracın üst üste hata vermesi, ajanın aynı metni kelimesi kelimesine yeniden yazması, tek
bir mesajın içinde aynı cümlenin defalarca geçmesi, ve en ilginci — **ardışık mesajların
%55'ten fazla ortak kelime taşıması.** Bu sonuncusu "aynı şeyi farklı kelimelerle
söylüyor" durumunu yakalıyor, üstelik hiçbir yapay zekâ modeli çağırmadan, sadece kelime
sayarak.

Tetiklendiğinde üç kademeli davranıyor: önce ajanı **yönlendiriyor** (koşum devam ediyor),
ısrar ederse **kesip bir kez daha başlatıyor**, yine ısrar ederse gerçekten durduruyor.
Ayrıca tekrarlarda boşa giden token'ı sayıp raporluyor — döngünün durdurulmadan önce ne
kadara mal olduğu görünür oluyor.

**Yakalar:** Aynı şeyi farklı kelimelerle tekrar eden ajanları — imza karşılaştırmasının
kör kaldığı yeri.
**Kaçırır:** Ajan gerçekten farklı şeyler deniyorsa ama hiçbiri işe yaramıyorsa.

---

## 2 · `arize-control` — Sert durdurma, modele sorulmadan

**Kaynak:** Arize'ın agent control loop yazısı

**Zihniyet:** Döngünün dört fazı var — gözlemle, karar ver, uygula, kaydet — ve model
bunlardan yalnızca birini yapıyor. Durma kararı modele bırakılamaz, çünkü **model, işin
bitip bitmediği konusunda yanılması en muhtemel bileşen.**

**Nasıl çalışır.** Beş durma koşulu tanımlıyor ve aralarında açık bir hiyerarşi kuruyor:
görev tamamlandı · adım limiti doldu · bütçe bitti · süre doldu · hata politikası devreye
girdi. Bunlardan **adım limiti birincil**, çünkü modelin söylediği hiçbir şeye bağlı değil.

Ayırt edici yanı: her koşumun sonunda **neden durduğunu kaydediyor.** "Tamamlandı",
"adım limiti", "bütçe aşıldı" ve "hata" farklı sonuçlardır; hepsini tek bir başarı oranına
karıştırmak neyin yanlış gittiğini gizler.

**Yakalar:** Her şeyi — çünkü hiçbir şeye bakmıyor, sadece sayıyor.
**Kaçırır:** Neyin yanlış gittiğini. Size "çok fazla adım attı" der, "aynı çağrıyı
tekrarlıyordu" demez. Teşhis değil, tavan.

---

## 3 · `loopguard-dignity` — Onurunla dur

**Kaynak:** "Ne Zaman Durması Gerektiğini Bilen Ajanlar"

**Zihniyet:** Bütçe sonradan eklenen bir güvenlik önlemi değil, **birinci sınıf bir ürün
kısıtı.** Ve durmak bir başarısızlık değil; kötü durmak başarısızlık.

**Nasıl çalışır.** Sert sınırlar koyuyor ama sınırları **araç bazında** tutuyor: bir aracın
bozuk olması genel deneme hakkını tek başına yiyemesin. İlerlemeyi durum özetinin
değişip değişmediğinden anlıyor. Ve elindeki bilgi yetmiyorsa **kullanıcıya soruyor** —
bunu bir yenilgi değil, meşru bir sonuç sayıyor.

En ayırt edici yanı durma şekli. Koşum bittiğinde dört şey söylüyor: ne denedi, ne buldu,
neden durdu, sırada ne var. Bu, sonsuz bir döngüyü **faydalı bir kısmi sonuca** çeviriyor.

**Yakalar:** Kaynak israfını, ve daha önemlisi kullanıcının elinde bir şey kalmasını sağlıyor.
**Kaçırır:** Hızlı ve ucuz döngüleri — sınırlar dolmadan önce çok tur dönebilir.

---

## 4 · `modexa-statemachine` — Döngünün şeklini kısıtla

**Kaynak:** "Ajan Döngüsü Problemi" (Modexa)

**Zihniyet:** Ajanlar belirsiz özgürlükleri sever, sistemlerin ise net durumlara ihtiyacı
var. Döngüyü tespit etmeye çalışmak yerine, **döngünün oluşamayacağı bir şekil ver.**

**Nasıl çalışır.** Serbest akışlı döngüyü bir durum makinesine indirgiyor: ANLA → TOPLA →
EYLEM → DOĞRULA → YANIT → DEVRET. Ve yalnızca belirli geçişlere izin veriyor. Ajan
"doğrula" durumundan "topla"ya geri dönemiyorsa, o döngü hiç oluşmuyor.

Bir de **geri dönüş merdiveni** var: ajan kendi kendine tekrar deneme icat etmiyor, sabit
bir merdiveni izliyor — bekleyip bir kez dene, aracı değiştir, kapsamı daralt, kullanıcıya
sor, elindekiyle cevap ver.

**Yakalar:** Bütün bir döngü sınıfını, oluşmadan önce.
**Kaçırır:** Durum makinesine sığmayan görevleri — esnekliği azaltıyor, ve bazı işler
gerçekten esneklik istiyor.

---

## 5 · `agentbudget-dollar` — Doların kendisini say

**Kaynak:** AgentBudget SDK

**Zihniyet:** Token bir vekil ölçü; asıl önemli olan dolar. Ve bütçe, koşum bittikten
sonra raporlanan bir şey değil, **koşum sırasında kesen bir tavan.**

**Nasıl çalışır.** Üç kademeli devre kesici: yumuşak limit uyarı veriyor, sert limit
kesiyor, döngü tespiti ayrı çalışıyor. Ama en özgün yanı **nihai cevap payı** — bütçenin
küçük bir dilimini baştan ayırıyor, sert limit erken tetikleniyor, böylece ajan işin
ortasında kesilmiyor, elindekini toparlayacak parası kalıyor.

Ayrıca alt görevlere **iç içe bütçe** dağıtabiliyor ve harcama üst bütçeye toplanıyor. Ve
tekrar tespitini zaman penceresiyle yapıyor: "on çağrı" değil, "bir dakikada on çağrı".

**Yakalar:** Pahalı olan her şeyi, döngü olsun olmasın.
**Kaçırır:** Ucuz döngüleri. Küçük bir model saatlerce dönebilir ve tavana hiç değmez.

---

## 6 · `claude-advisory` — Modele saatini göster

**Kaynak:** Claude Task Budgets (resmî API)

**Zihniyet:** Ajanı dışarıdan kesmek yerine, ne kadar kaldığını **ona söyle** ve kendini
ayarlamasına izin ver.

**Nasıl çalışır.** Konuşmaya bir geri sayım işareti giriyor; model bunu görüyor ve bütçe
azaldıkça işi toparlıyor. Kritik nokta: bu **tavsiye, zorlama değil.** Model, kesilmesi
bitirilmesinden daha zararlı olacak bir işin ortasındaysa bütçeyi aşabiliyor. Sert tavan
ayrı bir mekanizma olarak duruyor.

Bir de ince ayar var, pahalıya öğrenilmiş: **uyarı hangi eksende verildiği önemli.**
Adım sayısı azalıyor diye uyarmak modeli erken pes ettiriyor — Hermes bunu ölçüp ara
uyarıları tamamen kaldırmış. Süre azalıyor diye uyarmak ise işe yarıyor. Bu strateji
uyarıyı eksene göre veriyor.

**Yakalar:** Hiçbir şey — yakalamak için değil, **öngörülebilir bitiş** için var.
**Kaçırır:** Model dinlemezse her şeyi. Tek başına kullanılmamalı.

---

## 7 · `verify-gate` — "Bitirdim" demek kanıt değil

**Kaynak:** "Loop Engineering"

**Zihniyet:** Modelin bittiğini söylemesi bir **istek**tir, tamamlanma **kanıtı** değil.
Yazarın deyişiyle: *"'Bitirdim dedi', ajan dünyasının 'benim makinemde derleniyor'udur."*

**Nasıl çalışır.** Ajan işi bitirdiğini iddia ettiğinde bu doğrudan kabul edilmiyor; önce
bir doğrulama çalıştırılıyor — testler geçti mi, dosya yazıldı mı, ekranda beklenen şey
var mı. Doğrulama başarısızsa sonuç **reddedilmiyor**, ajanın gözlem akışına geri dönüyor
ve döngü devam ediyor. Ajan kendi hatasını görüp düzeltiyor.

**Yakalar:** Yarım bırakılmış işi "tamamlandı" diye teslim eden ajanları.
**Kaçırır:** Döngüyü. Ajan hiç "bitirdim" demezse bu kapı hiç açılmıyor. Mutlaka bir
bütçe stratejisiyle birlikte kullanılmalı.

---

## 8 · `telemetry-repair` — Tespit et, geri sar, düzelt

**Kaynak:** *Real-Time Detection and Repair of LLM Agent Failures*

**Zihniyet:** Hatayı yakalamak yetmez; **onarmak** da gerekir. Ve onarmanın en iyi yolu
ajana doğru cevabı vermek değil, **hangi kontrolün düştüğünü söylemek.**

**Nasıl çalışır.** Üç deterministik doğrulayıcı çalıştırıyor: ajanın söylediği toplam
gerçekten gördüğü verilerden çıkıyor mu, gereken bütün araç çağrıları yapıldı mı, aracın
döndürdüğü şey o aracın üretebileceği bir biçimde mi. Bunlar istatistik değil, kural —
eşik ayarı, kalibrasyon, ikinci bir model gerektirmiyorlar.

Bir kontrol düştüğünde ajan son sağlam noktaya geri sarılıyor ve **yalnızca hangi
kontrolün düştüğü söylenerek** yeniden çalıştırılıyor. Ölçülmüş sonuç: bu yaklaşım
kurtarma oranını %45'e çıkarıyor; ajana doğru cevabı doğrudan vermek ise %36'da kalıyor.

**Yakalar:** Sessizce yanlış iş yapan ajanları — döngüye girmeyen ama yanlış sonuç üreten.
**Kaçırır:** Onarımın işe yaramadığı durumları. Ölçülmüş: hedef sapmasında beş vakanın
dördü kurtarılıyor, ama **döngüde yeniden çalıştırma çoğu zaman aynı döngüyü üretiyor.**
Döngüde doğru hamle onarmak değil, durmak.

---

## 9 · `voi-allocation` — Bütçe bir tavan değil, bir bütçe

**Kaynak:** *Inference-Time Budget Control for LLM Search Agents*

**Zihniyet:** Diğer bütün stratejiler "ne zaman dur" diye soruyor. Bu strateji **"parayı
nereye harca"** diye soruyor. Bütçe bir kesme noktası değil, bir dağıtım problemi.

**Nasıl çalışır.** Her adımda ajanın seçenekleri puanlanıyor — ama ham faydaya göre değil,
**birim bütçe başına faydaya** göre. Pahalı bir eylem biraz daha faydalı olabilir; ucuz
bir eylem neredeyse bedava. Sistem oranı yüksek olanı seçiyor.

İki bütçeyi birden takip ediyor: araç çağrısı sayısı ve token. Biri kritik seviyeye
inince yeni keşif eylemleri pahalılaşıyor, bitirme çekici hale geliyor. Üstünde de
koruyucu kurallar var ki bütçe bitince ajan ucuz olduğu için aceleyle kötü bir cevaba
kaçmasın.

**Yakalar:** İsrafı — döngü olmayan ama gereksiz olan işi.
**Kaçırır:** Bütçe bolken hiçbir şey. Ölçülmüş: kazancı kaynak kıtken büyük, bol kaynakta
eriyor.

---

## 10 · `galileo-breaker` — Suç ajanda değil, araçta

**Kaynak:** Galileo demo vakası

**Zihniyet:** Bazı hatalar ajanın kusuru değil. Ajan doğru davranıyor, **araç bozuk.** Ve
bu tür hatalar en tehlikelileri, çünkü **başarıyla bitiyorlar.**

**Nasıl çalışır.** Tekrar saymıyor, **hata oranı** hesaplıyor: bir aracın son N çağrısının
yüzde kaçı hata verdi. Oran eşiği aşarsa o araç için devre kesici devreye giriyor.

Yakaladığı desen şu: ajan bir aracı çağırıyor, hata alıyor, tekrar deniyor, hata alıyor,
üçüncüde başarıyor. Kullanıcı doğru cevabı alıyor, koşum "başarılı" bitiyor, kimse ticket
açmıyor — ama iki çağrı boşa gitti. Bugün zararsız görünen bu desen, araç yarın tamamen
çökerse her isteğin bütün deneme hakkını yakmasına dönüşüyor.

**Yakalar:** Sessiz israfı, ve suçun yerini doğru gösteriyor.
**Kaçırır:** Araçların düzgün çalıştığı ama ajanın kafasının karıştığı durumları.

---

## 11 · `improvement-loop` — Bu koşumu kurtarma, sonrakini düzelt

**Kaynak:** OpenAI'ın agent improvement loop örneği

**Zihniyet:** Diğer on strateji koşumun içinde müdahale ediyor. Bu strateji **hiç müdahale
etmiyor** — sadece kaydediyor. Çünkü asıl soru şu: bu eşikleri kim, neye bakarak koydu?

**Nasıl çalışır.** Koşum boyunca her adımı ayrıntılı kaydediyor, sonunda dağılımdan eşik
öneriyor: başarılı koşumlar kaç adım sürmüş, kuyruğu nerede, tavan nereye konmalı.
Kaynakların üçü bağımsız olarak aynı şeyi söylüyor — **eşiği tahmin ederek değil ölçerek
seç.** Meşru işi kesecek kadar dar bir limit, modelin kötüleştiği gibi görünen sessiz bir
kalite gerilemesine dönüşüyor.

Bir de disiplin getiriyor: döngü ayarları prompt'la birlikte sürümleniyor ve bir terfi
kapısından geçiyor. Eşik değiştirmek, kod değiştirmekle aynı süreçten geçmeli.

**Yakalar:** Hiçbir şey — bu koşumda. Sonraki koşumların eşiklerini düzeltiyor.
**Kaçırır:** Şu anki koşumu. Tek başına bir koruma değil, bir ölçüm aracı.

---

# B Ailesi — Üretimdeki Harness'lardan Türeyen Zihniyetler

Bunlar makalelerden değil, gerçekten çalışan ajan sistemlerinin kaynak kodundan çıktı.

## 12 · `openhands-stuck` — Beş desen, ve "sıkışmak" ayrı bir sonuç ✅

**Kaynak:** OpenHands `stuck_detector.py`

**Zihniyet:** Döngü tek bir şey değil, **beş farklı desen.** Ve ajan sıkıştığında bu bir
hata değil, ayrı bir durum — öyle raporlanmalı.

**Nasıl çalışır.** Beş senaryoyu birden tarıyor: aynı eylem aynı sonucu veriyor · aynı
eylem sürekli hata veriyor · ajan kendi kendine konuşuyor · iki eylem arasında gidip
geliyor · bağlam penceresi hatası döngüsü. Eşikler senaryoya göre farklı.

Kritik bir ayrıntı var: iki olayı karşılaştırırken **her turda zaten değişen alanları
görmezden geliyor.** Bu yapılmazsa iki özdeş çağrı hiçbir zaman eşit çıkmaz ve dedektör
sessizce hiçbir şey bulmaz — hata türlerinin en kötüsü, çünkü çalışıyor gibi görünür.

Ve kademelendirme yok: sıkıştıysan sıkışmışsındır, doğrudan durur.

**Yakalar:** Tekrar eden her şeyi, dönüşümlü döngüler dahil.
**Kaçırır:** Ajan her turda farklı bir şey deniyorsa ama hiçbiri işe yaramıyorsa.

---

## 13 · `hermes-no-pressure` — Uyarma, sadece dur

**Kaynak:** Hermes `agent_init.py`

**Zihniyet:** Bütün diğer stratejiler "önce uyar, sonra dur" diyor. Hermes bunu denemiş
ve **geri almış.** Kod yorumu açık: *"Ara basınç uyarıları yok — modelleri karmaşık
görevlerde erken pes ettiriyordu."*

**Nasıl çalışır.** Adım bütçesi boyunca hiçbir uyarı vermiyor. Model bütçesinin ne durumda
olduğunu bilmiyor ve işine bakıyor. Bütçe gerçekten dolduğunda tek bir mesaj giriyor,
ajana **bir lütuf çağrısı** veriliyor, ve o çağrıda da metin üretmezse zorla özet isteniyor.

İlginç olan: aynı sistem **süre** bütçesinde %80'de bir toparlama uyarısı veriyor. Yani
eksene göre farklı karar vermişler. "Adımın azalıyor" mesajı modele "bu görev bana göre
değil" gibi geliyor; "süren azalıyor" ise "elindekiyle topla" gibi.

**Yakalar:** Uyarının kendisinin yarattığı zararı — erken pes etmeyi.
**Kaçırır:** Uyarının kurtarabileceği koşumları. Bu bir denge ve Hermes bir tarafı seçmiş.

---

## 14 · `openclaw-pingpong` — Adlandırılmış dedektörler ve sıkıştırma tuzağı

**Kaynak:** OpenClaw `tools.loopDetection`

**Zihniyet:** Dedektörleri isimlendir, ayrı ayrı aç-kapa, ve **bağlam sıkıştırmasının
kendisinin bir döngü kaynağı olduğunu** kabul et.

**Nasıl çalışır.** Üç adlandırılmış dedektör: aynı araç aynı parametrelerle · yoklama
araçlarında ilerleme yokluğu · **ping-pong** (iki eylem arasında gidip gelme). Üç kademeli
eşik: uyarı, kritik, ve küresel ilerleme-yok kesicisi.

En özgün parçası **sıkıştırma sonrası koruması.** Ajanın bağlamı sıkıştırıldıktan sonra
üç deneme boyunca ayrı bir koruma kurulu kalıyor. Sebebi somut: bağlam sıkıştırma
döngüsü belgelenmiş bir hata deseni — sıkıştır, bağlam yine dolsun, tekrar sıkıştır.

Parmak izini araç adı, argümanlar **ve sonuç** ile birlikte alıyor: aynı çağrı farklı
sonuç veriyorsa bu döngü sayılmıyor.

**Yakalar:** Dönüşümlü döngüleri ve sıkıştırma döngüsünü.
**Kaçırır:** Varsayılan olarak her şeyi — çünkü kutudan **kapalı** geliyor.

---

## 15 · `strands-entropy` — Tekrarı sayma, çeşitliliği say

**Kaynak:** Strands

**Zihniyet:** Bütün diğer dedektörler "aynı şey kaç kez tekrarlandı" diye soruyor. Bu
tersini soruyor: **"son N adımda kaç FARKLI şey oldu?"**

**Nasıl çalışır.** Son adımlara bakıp kaç ayrı eylem/düğüm çalıştığını sayıyor. Sayı
düşükse ajan dar bir alanda dönüyor demektir — ister aynı şeyi tekrarlasın, ister iki
şey arasında gidip gelsin, ister üç adımlık bir çevrimde dönsün. Tek bir kural bütün
çevrim desenlerini yakalıyor; k=2, k=3 diye ayrı ayrı taramaya gerek kalmıyor.

**Yakalar:** Her uzunlukta çevrimi, tek kuralla.
**Kaçırır:** Meşru olarak dar bir alanda çalışan işleri — bir dosyayı defalarca düzenleyip
test eden ajan da düşük çeşitlilik gösterir.

---

## 16 · `agentscope-grace` — Bitirmeye zorla

**Kaynak:** AgentScope

**Zihniyet:** Limit dolduğunda koşumu çöpe atmak israf. Ama modelden nazikçe bitirmesini
istemek de yetmiyor — **bitirmekten başka seçeneği kalmasın.**

**Nasıl çalışır.** Limit dolduğunda koşum bitmiyor; ajana beş turluk bir **lütuf bütçesi**
veriliyor. Ama bu turlarda araç seçimi kilitleniyor: ajan artık yeni araç çağıramıyor,
yalnızca cevap üretebiliyor. Nazik bir rica değil, mekanik bir kısıt.

**Yakalar:** Yarım kalmış işin çöpe gitmesini.
**Kaçırır:** Döngünün kendisini — bu bir tespit değil, iniş mekanizması.

---

## 17 · `autogen-static` — Koşmadan önce yakala

**Kaynak:** AutoGen `GraphFlow`

**Zihniyet:** En iyi döngü tespiti, çalışma zamanında hiç yapılmayan tespittir. Ajan
akışının **şeklini** baştan denetle.

**Nasıl çalışır.** Ajan grafiği kurulurken doğrulanıyor: bir çevrim var ve o çevrimin
çıkış koşulu yoksa, sistem **hiç başlamıyor** — hata veriyor. Çalışma zamanında hiçbir
maliyet yok, çünkü koşum hiç başlamadı.

**Yakalar:** Yapısal döngüleri, sıfır maliyetle ve sıfır yanlış pozitifle.
**Kaçırır:** Modelin kararından doğan döngüleri. Grafiğin şekli doğru olabilir ama model
yine de aynı düğümde takılabilir.

---

## Taban çizgisi · `none` — Hiçbir kontrol ✅

**Kaynak:** Anthropic'in referans computer-use döngüsü

Bu bir strateji değil, **ölçüm aracı.** Anthropic'in kendi referans uygulamasında ana
döngü şöyle: `while True`, tek çıkış modelin araç çağırmayı bırakması. Tur sayacı yok,
döngü tespiti yok, bütçe yok.

"Kontrol koymanın faydası ne" sorusunun cevabı, bu sütunla diğerleri arasındaki farktır.

---

# Toplu Bakış

| Strateji | Ne zaman | Neye bakar | Tetiklenince |
|---|---|---|---|
| `pi-signature` | eylemden sonra | eylem imzası + metin benzerliği | yönlendir → kes → dur |
| `arize-control` | eylemden önce | sayaçlar | dur, sebebi kaydet |
| `loopguard-dignity` | eylemden önce | araç bazlı sayaç + ilerleme | dur ve dört maddeyle raporla |
| `modexa-statemachine` | eylemden önce | durum geçişi | geçişi reddet, merdivene bin |
| `agentbudget-dollar` | eylemden önce | dolar | uyar → kes, pay ayrılmış |
| `claude-advisory` | istek hazırlanırken | kalan bütçe | modele söyle, kesme |
| `verify-gate` | bitirme iddiasında | dış doğrulama | reddet, döngüye geri ver |
| `telemetry-repair` | eylemden sonra | deterministik kontroller | geri sar, hangi kontrol düştü söyle |
| `voi-allocation` | eylemden önce | fayda/maliyet oranı | eylemi değiştir |
| `galileo-breaker` | eylemden sonra | araç hata oranı | aracı devre dışı bırak |
| `improvement-loop` | koşumdan sonra | dağılım | eşik öner |
| `openhands-stuck` ✅ | eylemden sonra | beş olay deseni | doğrudan dur |
| `hermes-no-pressure` | bütçe dolunca | sayaç | lütuf çağrısı + zorla özet |
| `openclaw-pingpong` | eylemden sonra | üç dedektör + sıkıştırma | uyar → kritik → küresel kes |
| `strands-entropy` | eylemden sonra | çeşitlilik | dur |
| `agentscope-grace` | limit dolunca | sayaç | araç seçimini kilitle |
| `autogen-static` | koşum başlamadan | graf şekli | başlatma |
| `none` ✅ | — | — | — |

✅ = kodda çalışıyor. Diğerleri tasarlandı, Faz 2–3'te yazılacak.

---

# Birlikte Kullanım

Kaynakların neredeyse tamamı aynı şeyi söylüyor: **tek katman yetmiyor.** Stratejiler
birleştirilebilir; ilk tetiklenen kazanır.

Mantıklı üçlü kombinasyonlar:

**Kaba ama sağlam:** `arize-control` + `openhands-stuck` + `verify-gate`
Bir tavan, bir desen dedektörü, bir çıkış kapısı. Kimse kimsenin işine karışmıyor.

**Maliyet odaklı:** `agentbudget-dollar` + `galileo-breaker` + `voi-allocation`
Doları say, bozuk aracı kes, kalanı akıllı harca.

**Az müdahaleci:** `claude-advisory` + `hermes-no-pressure` + `agentscope-grace`
Modele güven, ama bittiğinde bitir.

Bir uyarı: **iki strateji birbirini maskeleyebilir.** Bütçe sınırı çok darsa döngü
dedektörü hiç konuşamaz — koşum doğru durur ama *neden* durduğunu öğrenemezsiniz.
Bütçe bir şeyin yanlış gittiğini söyler; döngü tespiti neyin yanlış gittiğini.
