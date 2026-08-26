# Agent Reliability — Loop Detection ve Task Budget Kontrolleri

> **Konu:** Otonom ajanların sonlanmama problemi; nedenleri, tespit yöntemleri, yöntemlerin
> sınırları ve uygulama önerileri.
> **Hedef okuyucu:** Ajan veya ajan benzeri otomasyon geliştiren ekipler.

---

## 1. Özet

Bir dil modeli tek başına sonlanır: girdi alır, cevap üretir, biter. **Ajan = model + döngü.**
Ajanın yeteneği bu döngüden gelir; modelde bulunmayan bir soru da buradan doğar:
**döngüyü kim sonlandıracak?**

Yetkinin modele bırakılması yaygın ancak hatalı bir tercihtir. Sebep modelin yetersizliği
değil **görüş alanıdır**: model her turda yalnızca önündeki bağlama bakarak bir sonraki adımı
seçer; kaç kez aynı şeyi denediğini ve o ana kadar ne harcadığını görmez.

İki tamamlayıcı kontrol ailesi tanımlanır:

| | Sorduğu soru | Müdahale ölçütü |
|---|---|---|
| **Loop detection** | Sistem ilerliyor mu, yoksa aynı işi mi tekrarlıyor? | Tekrar deseni, durum değişmezliği |
| **Task budget** | Bu koşuma ne kadar kaynak ayrıldı, ne kadarı harcandı? | Adım, replan, token, süre, maliyet |

İkisi birbirinin yerine geçmez. Bütçe her hata modunu eninde sonunda keser ancak nedenini
açıklamaz; döngü tespiti nedeni raporlar ancak ilerlemenin gerçek olduğu senaryolarda
etkisizdir.

**Sonuç:** tek katmanlı koruma yeterli değildir. Katmanlar birbirinin kör noktasını kapatacak
biçimde seçilmelidir.

---

## 2. Problem tanımı

Otonom bir sistemin `karar ver → araç çağır → gözlemle → gözden geçir → tekrarla`
döngüsünü sürdürmesi, ancak bir sonuca ulaşmaması.

Belirleyici özellik: ajan **çökmez**. İstisna fırlatmaz, hata log'u üretmez, alarm
tetiklemez. Sağlıklı görünür; yalnızca ilerlemez. Bu nedenle durum tabanlı klasik izleme
problemi göremez.

### Gözlemlenen desenler

| Desen | Görünüm |
|---|---|
| Sonsuz tekrar | Aynı araç, aynı argüman; sonuç değişmiyor |
| Bitmeyen arama | Her turda "bir kaynak daha" |
| Özeleştiri sarmalı | Üretilen cevabın sürekli yeniden değerlendirilmesi |
| Araç ping-pong'u | A → B → A → B; hiçbiri sonuç üretmiyor |
| Plan çalkantısı | Her adımda planın baştan yazılması |

Ortak nokta: sistem sürekli hareket hâlindedir ancak durum ilerlemez.
**Hareket, ilerleme anlamına gelmez.**

---

## 3. Kök nedenler

Döngülerin çoğu model kalitesinden değil, **eksik kısıtlardan ve hatalı teşviklerden**
kaynaklanır.

| # | Neden | Açıklama |
|---|---|---|
| 1 | **Tanımlı bir "bitti" kriterinin olmaması** | İnsanlarda içsel bir yeterlilik algısı vardır; ajanda yoktur. *"Kapsamlı ol, doğrula, hiçbir şeyi kaçırma"* biçiminde bir yönerge, sınırsız arama talimatıdır. Ajan yönergeye uymaktadır. |
| 2 | **Güvenilmez araç + naif retry** | Araç tutarsız davrandığında (zaman aşımı, hız sınırı, kısmi yanıt) ajan bunu "tekrar dene, biraz farklı" olarak yorumlar. Üst sınır tanımlı değilse deneme sayısı sınırsızdır. |
| 3 | **Belirsiz hedef** | Başarı kriteri tanımsızsa ajan yorumlar arasında salınır. Hedef belirsizliği döngü baskısı üretir. |
| 4 | **Bağlamın kademeli bozulması** | Her adım bağlama yeni token ekler: loglar, kısmi planlar, araç çıktıları. Sonraki kararlar giderek daha gürültülü bir geçmiş üzerinden verilir. |
| 5 | **Hata yapmama yönünde optimizasyon** | Aşırı tedbiri teşvik eden bir yönergede en düşük riskli eylem "bir kontrol daha yapmak" olur. |

**Not:** 5. nedenin kaynağı kod değil yönergedir. Bu dokümanda anlatılan mekanizmaların
hiçbiri onu çözmez; 1–4 kod tarafında ele alınabilir, 5 yalnızca yönerge tasarımıyla.

### Kavramsal model

Ajan bir arama algoritması olarak düşünülebilir. Şu üç bileşenden yoksun bir arama
sonlanmaz: **durma koşulu · bütçe · ilerleme ölçütü**.

Bu çerçevede döngü bir davranış bozukluğu değil, **eksik kısıt**tır. Buradan çözümün biçimi
de çıkar: amaç ajana durmasını *söylemek* değil, sisteme durması için **mekanik nedenler**
tanımlamaktır. Yönergeye eklenen bir uyarı cümlesi kontrol sayılmaz.

---

## 4. Problemin ölçeği

| Kaynak | Bulgu | Değer |
|---|---|---|
| MAST — 1642 yürütme izi, 7 framework | Adım tekrarı (en sık hata modu) + durma koşulunu tanımama | **%28,1** |
| Token Budgets — 21 framework, 2023–2026 | Doğrulanmış üretim olayı | **63** |
| Yaygın referans computer-use döngüsü | `while True`; tur sayacı, döngü tespiti veya bütçe içermiyor | **0 kontrol** |
| 31 üretim harness'ı, kaynak kodu incelemesi | Mekanizma mevcut ancak varsayılanda kapalı | **12/24** |
| Aynı inceleme | Ardışık olmayan çevrimi (A-B-A-B) tespit edebilen | **2/22** |

Vaka kataloğunun sonucu: incelenen bütçe aşımlarının hiçbiri kullanıcı maliyeti oluşmadan
önce engellenmemiştir. Düzeltmeler hızlı gelmekte, ancak **olay gerçekleştikten sonra**.

### Değerlendirme kriteri

Bir framework için doğru soru *"loop detection var mı"* değildir; olgun sistemlerin
neredeyse tamamında bir mekanizma bulunur. Belirleyici olan iki soru:

1. **Varsayılanda etkin mi?**
2. **Doğru katmanda mı tetikleniyor?**

Gözlemlenen tipik uyumsuzluklar:

- Dokümantasyonda belirtilen varsayılan ile koddaki gerçek varsayılanın farklı olması
  (belgede sonlu bir değer, kodda pratikte sınırsız bir sabit).
- İterasyon sınırının varsayılan olarak tanımsız bırakılması.
- Token sınırlayıcının mevcut olması ancak döngü iterasyonu başına değil yalnızca istek
  başına tetiklenmesi — **sınır mevcut, yanlış katmanda**.

---

## 5. Kontrollerin döngüdeki yeri

![Ajan döngüsü ve kontrol noktaları](gorseller/01-ajan-dongusu.png)

Kontroller döngünün dışında değil, döngü içindeki beş kancada çalışır. Her kanca farklı bir
soruyu yanıtlar:

| Kanca | Soru | İlgili kategoriler |
|---|---|---|
| `before_step` | Adımı atmaya bütçe var mı? | SAYAÇ · KARAR |
| `on_action` | Bu çağrı daha önce yapıldı mı? | PENCERE · ŞEKİL |
| `on_observation` | Ortam gerçekten değişti mi? | PENCERE · DÜNYA |
| `on_finish_claim` | İddia gerçekle uyumlu mu? | DÜNYA |
| `on_stop` | Koşum neden sonlandı? | tümü |

`on_stop` sıklıkla atlanır ancak teşhis değeri en yüksek olan kancadır. *"Limit doldu"*
bilgisi yetersizdir; kaydedilmesi gereken dört alan: **hangi eksen doldu, ne denendi,
ne bulundu, önerilen sonraki adım.**

---

## 6. Kontrol kategorileri

![Beş kontrol kategorisi](gorseller/02-bes-kategori.png)

İncelenen on altı yaklaşım aynı soruyu yanıtlar — *"şimdi durmalı mı?"* — ancak beş farklı
bakış açısından. Sıralama uygulama önceliğini yansıtır.

Her kategori bir **mekanizmayı** paylaşır; alt yaklaşımlar aynı mekanizmayı farklı ayarlarla
kullanır. Aradaki fark akademik değildir: aynı kategori içindeki iki yaklaşım aynı koşumu
farklı terminal durumla sonlandırabilir.

---

### 6.1 · SAYAÇ — sayaç tut, eşiği aşınca sonlandır

Adım, replan, token, süre ve maliyet eksenlerinde sayaç tutar; modele danışmaz.
En düşük maliyetli ve en düşük yanlış pozitif riskli katmandır.

| Yaklaşım | Mekanizma | Limit dolduğunda | Kendi sınırı |
|---|---|---|---|
| **`arize-control`**<br><sub>Arize control loop</sub> | Beş eksende sayaç; adım limiti birincil. Her koşumda durma nedeni kaydedilir | Sert durdurma, lütuf turu yok | Neden durduğunu söylemez |
| **`budget-grace`**<br><sub>AgentScope `EXCEED_MAX_ITERS` + Hermes</sub> | Limit dolunca 1–5 ek tur; **araç seçimi kilitlenir** | Ajan yalnızca cevap üretebilir | Lütuf turları da maliyet yakar; uzun tutulursa tavan anlamsızlaşır |
| **`claude-advisory`**<br><sub>Anthropic `task_budget`</sub> | Yönergeye geri sayım enjekte edilir | Zorlama yok, model aşabilir | Model uyarıyı yok sayarsa hiçbir koruma sağlamaz |
| **`agentbudget-dollar`**<br><sub>agent budget framework</sub> | Dolar tavanı + **%15 nihai cevap payı**; zaman pencereli patlama tespiti | Ayrılan pay cevap üretimine kalır | Fiyat tablosu bakım gerektirir; model değişince yanlış sayar |

**Eksen seçimi önemlidir.** Uyarı davranışı eksene göre değişir: iterasyon ekseninde ara
uyarı vermenin modelleri erken pes ettirdiği gözlenmiştir; süre ekseninde ise %80 uyarısı
yararlıdır. Tek bir "uyar / uyarma" kuralı yoktur.

**Kategori sınırı:** durma nedenini açıklamaz, yalnızca eşiğin aşıldığını bildirir.
Teşhis için pencere tabanlı bir katman gerekir.

---

### 6.2 · PENCERE — son N olayı karşılaştır

Olay geçmişinde bir pencere tutar ve tekrar arar. Döngüyü tespit eder **ve nedenini
raporlar**: hangi çağrı, kaç kez.

| Yaklaşım | Mekanizma | Müdahale biçimi | Kendi sınırı |
|---|---|---|---|
| **`openhands-stuck`**<br><sub>OpenHands `stuck_detector`</sub> | Beş desen taraması: eylem-gözlem, eylem-hata, monolog, çevrim, bağlam penceresi. Eşikler 4/3/3/6 | **Doğrudan durdurur**, uyarı yok. Ayrı bir terminal durum üretir | İmza normalizasyonu hatalıysa sessizce hiçbir şey bulmaz ve testleri geçer |
| **`openclaw-pingpong`**<br><sub>`tools.loopDetection`</sub> | Adlandırılmış dedektörler: `genericRepeat`, `knownPollNoProgress`, `pingPong`. Parmak izi = araç + argüman + **sonuç** | Üç kademe: 10 uyarı → 20 kritik → 30 kesici. Sıkıştırma sonrası ek koruma | Varsayılanda kapalı olabilir; üçüncü kademeye kadar önemli maliyet oluşur |
| **`pi-signature`**<br><sub>anti-doom-loop</sub> | Altı ucuz sinyal; yakın-benzer metin için ≥%55 kelime örtüşmesi de sayılır | **Önce yönlendirir**, ısrar hâlinde keser | Altı sinyal × ayrı eşik × iki kademe; yanlış uygulanırsa sessizce etkisiz kalır |
| **`strands-entropy`**<br><sub>Strands SDK</sub> | Tekrarı değil **farklılığı** sayar: son N adımda kaç farklı eylem yürütüldü | Çeşitlilik eşiğin altına düşerse durdurur | Meşru olarak dar alanda çalışan iş de düşük çeşitlilik gösterir |
| **`loopguard-dignity`** | Ortam durumu hash'i değişmiyorsa ilerleme yok sayılır; eylem başına deneme hakkı ayrıca izlenir | **Çekimser kalır** — hata değil, girdi bekleyen ayrı bir sonuç. Dört alanlı rapor üretir | Hızlı ve ucuz döngüler sınırlar dolmadan çok tur dönebilir |

**Tek kural mı, çok sinyal mi?** `strands-entropy` tek bir kuralla bütün çevrim desenlerini
yakalar ve eşik taraması gerektirmez; `pi-signature` daha ayrıntılı teşhis verir ancak
kalibrasyon yükü yüksektir.

**Kategori sınırı:** içerik her adımda değişiyorsa etkisizdir. Sonsuz sayfalama gibi
senaryolarda ajan gerçekten ilerleme kaydeder; bu katman bir anomali göremez. Böyle
durumlarda tek etkili kontrol bütçedir.

---

### 6.3 · DÜNYA — modelin dışından kanıt topla

Kararı modelin beyanına değil ortamın gözlemlenebilir durumuna dayandırır.
Döngü içermeyen hata sınıfını yakalayan tek kategoridir.

| Yaklaşım | Mekanizma | Müdahale biçimi | Kendi sınırı |
|---|---|---|---|
| **`verify-gate`** | Bitirme iddiası geldiğinde testi / dosyayı / ortam durumunu sınar | Doğrulama geçmezse iddia **reddedilir**, koşum gözleme geri döner | Ajan bitirme iddiası üretmezse kapı hiç açılmaz |
| **`telemetry-repair`** | Üç deterministik kontrol: `total_consistency` (iddia görülen veriden çıkıyor mu), `required_coverage` (gereken araçlar çağrıldı mı), `tool_contract` (sonuç beklenen biçimde mi) | Checkpoint'e geri sarar ve **hangi kontrolün düştüğünü** bildirir | Checkpoint tutmak mimari yük getirir; döngü içindeyken onarım etkisizdir, durdurmak gerekir |
| **`galileo-breaker`** | Araç başına hata **oranı** izlenir (sayı değil); sessizce başarısız olan retry'lar da sayılır | Eşiği aşan araç için devre kesici açılır | Geçici ile kalıcı hatayı ayırması tamamen eşiğe bağlıdır |

`verify-gate` iddiayı sınar, `telemetry-repair` iddianın **hangi yönden** hatalı olduğunu
söyler, `galileo-breaker` ise sorunu ajanda değil **araç katmanında** arar. Üçü farklı soruyu
yanıtlar.

**Doğrulayıcı göreve özel olmalıdır.** Sabit bir başarı metni arayan genel bir kural, farklı
görev tiplerinde her iddiayı reddeder. Doğrulama ölçütü görevden türetilmelidir.

**Kategori sınırı:** ajan bitirme iddiası üretmezse hiçbir yaklaşım tetiklenmez.
Bütçe katmanıyla birlikte kullanılması zorunludur.

---

### 6.4 · ŞEKİL — döngünün biçimini kısıtla

Tekrarı saymak yerine, döngünün alabileceği biçimleri önceden sınırlar.

| Yaklaşım | Mekanizma | Müdahale biçimi | Kendi sınırı |
|---|---|---|---|
| **`modexa-statemachine`** | İzinli geçiş tablosu; doğrulama adımı bir kapıdır, atlanamaz | İhlalde merdivende bir basamak yükselir: geri çekil → alternatif yaklaşım → kapsamı daralt → **kullanıcıya sor** → sonlandır | Esnekliği azaltır; önceden modellenemeyen işler makineye sığmaz. Sonradan eklenmesi zordur |
| **`autogen-static`**<br><sub>AutoGen `GraphFlow`</sub> | Koşum başlamadan graf taranır; çıkış koşulu olmayan çevrim aranır (`Cycle detected without exit condition`) | Böyle bir çevrim varsa **sistem hiç başlatılmaz** | Yalnızca yapısal döngüyü görür; modelin aynı düğümde takılmasını göremez |

İkisi farklı zamanlarda çalışır: `modexa-statemachine` çalışma zamanında, `autogen-static`
koşumdan önce. Statik doğrulama yanlış pozitif üretmez — çünkü çalışma zamanı davranışı
hakkında hiçbir iddiada bulunmaz.

**Kategori sınırı:** serbest biçimli, önceden modellenemeyen görevlerde uygulanamaz.

---

### 6.5 · KARAR — bütçeyi tavan değil tahsis olarak ele al

Diğer dört kategori *"durmalı mı"* sorusunu yanıtlar; bu kategori *"kalan bütçe nereye
harcanmalı"* sorusunu yanıtlar.

| Yaklaşım | Mekanizma | Müdahale biçimi | Kendi sınırı |
|---|---|---|---|
| **`voi-allocation`**<br><sub>inference-time budget control</sub> | Eylemler birim bütçe başına beklenen faydaya göre puanlanır; bütçe baskısı arttıkça seçim daralır. Çift bütçe: araç + token | Yetersiz kanıtla verilen **erken cevabı engeller** | Koşumu sonlandırmaz. Bütçe bolken kazanç erir |
| **`improvement-loop`** | Koşum izleri toplanır; başarılı koşum dağılımından p99 eşik türetilir. Sürümlenmiş yapılandırma + terfi kapısı | **Müdahale yok.** Sonraki tur için eşik önerisi üretir | Mevcut koşumu kurtarmaz; veri birikmesini bekler |

**Kategori sınırı:** her iki yaklaşım da koşumu sonlandırmaz. Tek başlarına koruma
sağlamazlar; diğer kategorilerin üzerine eklenen bir optimizasyon katmanıdır.

---

### 6.6 · Katman seçimi

| Öncelik | Katman | Gerekçe |
|---|---|---|
| 1 | **SAYAÇ** | En ucuz, en düşük yanlış pozitif riski, doğrudan maliyet koruması |
| 2 | **DÜNYA** | Diğer katmanların göremediği hata sınıfını kapsar |
| 3 | **PENCERE** | Teşhis üretir: yalnızca "durdu" değil, "neden durdu" |
| 4 | **ŞEKİL** | Görev yapısı önceden modellenebiliyorsa |
| 5 | **KARAR** | İz verisi biriktikten sonra, kalibrasyon amacıyla |

Katmanlar birleştirilebilir. Birleştirmede **ilk tetiklenen sonucu belirler**; bu nedenle
sıralama, hangi teşhisin öne çıkacağını da belirler.

---

## 7. Harness kaynaklı döngüler

Döngülerin bir kısmı model davranışından değil, **çalıştırma katmanındaki bir hatadan**
kaynaklanır. En yaygın örnek koordinat uzayı uyumsuzluğudur.

Görsel ajanlarda ekran görüntüsü token maliyeti nedeniyle küçültülerek modele iletilir.
Model, küçültülmüş görüntünün piksel uzayında koordinat üretir. Bu koordinat gerçek ekrana
uygulanmadan önce ölçek çarpanıyla dönüştürülmelidir:

```
modele iletilen görüntü : 1280 x 720
gerçek ekran            : 1920 x 1080
ölçek çarpanı           : 1.5
```

Dönüşüm uygulanmazsa her etkileşim hedefin `1/1.5` katına, yani boş alana düşer. Ortam
değişmez, hata dönmez, ajan tekrar dener. **Model doğru davranmasına rağmen döngü oluşur.**

Aynı hata sınıfının diğer örnekleri:

- Bir sistem çağrısının bulunmadığı ortamda istisnanın yutulup boş değer döndürülmesi;
  ilgili kontrol çalışmadığı hâlde hata üretmez.
- Sayaçların yanlış noktada artırılması; engellenen eylemlerin gerçekleşmiş sayılması.
- Boş araç çıktısının gösterim amaçlı bir yer tutucuya dönüştürülmesi ve bu yer tutucunun
  aşağı akıştaki doğrulama tarafından gerçek kanıt olarak yorumlanması.

### Bu sınıfın teşhis açısından önemi

Böyle bir koşumda döngü tespit katmanlarının tamamı tetiklenir ve hepsi **doğru** rapor
verir: *"aynı çağrı tekrarlanıyor"*. Ancak hiçbiri kök nedene ulaşamaz.

> **Döngü tespiti bir teşhis aracı değil, bir frendir.**
> Kök neden yalnızca araç katmanının kendi telemetrisinden okunabilir.

Ortak risk: kontrol çalışmıyordur ancak **rapor çalıştığını bildirir**. Bu durum, kontrolün
hiç bulunmamasından daha risklidir.

---

## 8. Yanlış pozitif riski

Yalnızca yakalanan vakaları gösteren bir değerlendirme, tespit mekanizmasının yanlış pozitif
oranı hakkında bilgi vermez. Her yakalama senaryosunun karşısında bir **kontrol senaryosu**
bulunmalı ve beklenen sonuç, kontrolsüz koşumla birebir aynı olmalıdır.

Sık karşılaşılan yanlış pozitif kaynakları:

| Kaynak | Mekanizma |
|---|---|
| Şema ayıklanmadan uygulanan desen | URL içindeki yol parçasının dosya adı olarak yorumlanması |
| Bağlamsız fiil taraması | Görev metnindeki bir fiilin yanlış araç gereksinimi çıkarması |
| Kaynaktan kopyalanan eşikler | Farklı bir iş yükü için kalibre edilmiş eşiklerin meşru koşumları kesmesi |
| Ardışık sayaç sıfırlanması | A-B-A-B deseninde tekrarın hiç sayılmaması (yanlış negatif ikizi) |

**Uygulama önerisi:** doğal dilden çıkarım yapan her kural için en az bir yanlış pozitif
testi yazın. Eşikleri başka bir sistemin dokümantasyonundan almayın; kendi izlerinizden ölçün.

---

## 9. Karşılaştırma sonuçlarının yorumlanması

Bir karşılaştırmada tetiklenmemiş bir kontrolün "etkisiz" sayılması hatalıdır.
Tetiklenmeme dört farklı duruma karşılık gelir:

| Görünen | Gerçek durum |
|---|---|
| Tetiklenmedi | **Eşiğe ulaşılmadı** — koşum daha önce sonlandı |
| Tetiklenmedi | **Yeterli veri yok** — karar için gereken minimum gözlem toplanmadı |
| Tetiklenmedi | **Kör nokta** — bu hata modunu yapısal olarak göremez |
| Tetiklenmedi | **Tasarım gereği durdurmaz** — yalnızca ölçüm yapan katman |

Bu dört durum ayırt edilmeden hazırlanan bir karşılaştırma tablosu yanıltıcıdır.

Ayrıca kontroller üst üste bindirildiğinde **ilk tetiklenen sonucu belirler** ve koşum orada
sonlanır. Tüm katmanların aynı anda etkinleştirilmesi bir karşılaştırma yöntemi değildir;
en düşük eşikli kontrol diğerlerinin tetiklenmesine izin vermez.

---

## 10. Uygulama kontrol listesi

**Temel seviye**

- [ ] Adım, token ve süre tavanı tanımlayın. Varsayılanı tanımsız bırakmayın.
- [ ] Bütçe tükenmesini ayrı bir terminal durum olarak raporlayın; `OK` içine gizlemeyin.
- [ ] Her koşuma durma nedeni alanı (`terminated_by`) ekleyin.
- [ ] İterasyon başına span yazın. Tek span'lık iz, hangi adımların tekrarlandığını göstermez.

**İleri seviye**

- [ ] Tekrarı ardışık değil, pencere içinde sayın; ardışık sayaç A-B-A-B deseninde sıfırlanır.
- [ ] İmzaya sonucu da dâhil edin; aynı çağrı + aynı sonuç daha güçlü bir sinyaldir.
- [ ] Bitirme iddiasını ortam durumuna karşı doğrulayın. Doğrulayıcı göreve özel olmalıdır.
- [ ] Kademeli müdahale uygulayın: önce yönlendirme, ısrar hâlinde sonlandırma.
- [ ] Araç katmanına kendi telemetrisini ekleyin; kök neden yalnızca orada görünür.

**Kaçınılması gerekenler**

- Yönergeye eklenen uyarı cümlesini kontrol olarak saymak.
- Eşikleri başka bir sistemin dokümantasyonundan kopyalamak.
- Tüm katmanları aynı anda etkinleştirip karşılaştırma yaptığını varsaymak.
- Yalnızca yakalanan vakaları raporlayıp yanlış pozitif oranını ölçmemek.

---

## 11. Entegrasyon gereksinimleri

1. **İterasyon başına span.** Alanlar: iterasyon indeksi, araç adı, argüman özeti, sonuç
   özeti, ortam durumu özeti, süre, token, kontrol kararı.
2. **Durma nedeni alanı** her koşumda zorunlu.
3. **Eşikler yapılandırmadan okunmalı**, kodda sabit olmamalı ve sürümlenmelidir.
4. **Bütçe tükenmesi ayrı terminal durum** olarak modellenmelidir.
5. **Eşik kalibrasyonu ölçüme dayanmalıdır**; başarılı koşum dağılımından türetilir.

**Önerilen sıra:** tüm katmanların aynı anda devreye alınması gerekmez. İlk adım SAYAÇ
katmanıdır — en düşük maliyet, en düşük yanlış pozitif riski, doğrudan maliyet koruması.
İz verisi biriktikçe eşikler kalibre edilir; PENCERE ve DÜNYA katmanları sonraki aşamada
eklenir.

---

## 12. Sözlük

| Terim | Tanım |
|---|---|
| **Guardrail** | Ajanın kendisini koruyan kontrol: döngü, bütçe, tekrar |
| **Kanca (hook)** | Döngü içinde kontrolün karar verdiği nokta |
| **İmza** | Bir araç çağrısının kimliği: araç adı + argümanlar (+ opsiyonel sonuç) |
| **Çevrim (cycle)** | Ardışık olmayan tekrar deseni: A → B → A → B |
| **Terminal durum** | Koşumun sonlanma biçimi: `OK`, `STUCK`, bütçe tükenmesi vb. |
| **Karşı olgusal** | Gerçekten koşturulmamış, "müdahale etseydi" varsayımına dayanan sonuç |
| **Yanlış pozitif** | Meşru bir koşumun hatalı olarak sonlandırılması |
| **Taban çizgisi** | Hiçbir kontrol etkin değilken yapılan koşum; karşılaştırma referansı |
| **set-of-marks** | Ekran öğelerinin numaralandırılarak modele koordinat yerine numara ürettirme yöntemi |

---

## 13. Kaynaklar

| Doküman | İçerik |
|---|---|
| `sources_list/01_METODOLOJI.md` | Birincil kaynakların metodoloji sınıflandırması ve kanıt gücü sıralaması |
| `sources_list/harness_kontrolleri.md`<br>`sources_list/harness_kontrolleri_2.md` | 31 üretim harness'ının kaynak kodu düzeyinde kontrol mekanizması incelemesi |
| `sources_list/tespit_sinirlari.md` | Tespit yöntemlerinin kör noktaları |
| `cua_lab/docs/zihniyetler.md` | Yaklaşımların ayrıntılı karşılaştırması |
| `cua_lab/docs/computer_use_zihniyet.md` | Computer-use mimarileri: eylem uzayı, ekran temsili, döngü topolojisi |
| `Agent_Reliability_2_Sayfa.pdf` | Özet referans |

**Diyagramlar:** `gorseller/*.svg` (düzenlenebilir kaynak), `gorseller/*.png` (sayfa eki).
