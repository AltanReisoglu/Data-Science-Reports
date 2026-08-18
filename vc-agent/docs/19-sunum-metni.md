# Sunum metni — AutoGen, OpenClaw ve Atlas

**Tarih:** 19 Ağustos 2026
**Desteler:** `hap-autogen.pdf` (37) · `hap-openclaw.pdf` (19) · `hap-openclaw-nis.pdf` (17)
**Soru hazırlığı:** ayrı sayfada, 24 soru

---

## Nasıl kullanılır

**Kalın** yazılan cümleler ağızdan çıkacak olanlar. Normal yazılanlar sahne
yönergesi — okunmaz, yapılır. `[S12]` bir slayt numarası.

Üç sürüm var, çünkü toplantı süresi daralır ve daraldığında neyi atacağını o an
düşünmek istemezsin:

| sürüm | süre | slayt | kime |
|---|---|---|---|
| **Kısa** | 12 dk | 7 + 2 kapak | yalnız yönetim varsa |
| **Tam** | 30 dk | 19 + 2 kapak | karma izleyici — varsayılan |
| **Derin** | 50 dk | 45+ | mühendisler ağırlıktaysa |

Sayılar bölüm başlıklarındaki listelerle birebir: kısa sürüm Perde 1'den 3,
Perde 2'den 4 slayt alıyor; tam sürüm 10 ve 9.

Her bölümün başında hangi sürümde ne kalacağı yazılı.

**Tek kural:** süre daralırsa slayt atla, cümle kısaltma. Yarım anlatılmış bir
mekanizma, hiç anlatılmamış olandan daha zararlı — çünkü izleyici anladığını
sanır.

---

## Omurga

Sunumun tamamı tek bir cümleyi kurar ve o cümleyi kanıtlar:

> **AutoGen bize bir motor veriyor ama kontrol düzlemi vermiyor. OpenClaw
> kontrol düzlemini çözmüş ama güven modeli bizim kurumumuz için yanlış. Atlas'ın
> alması gereken şey: AutoGen'in motoru, OpenClaw'ın karar kuralları, ve bizim
> kendi güven modelimiz.**

Üç perde bunu sırayla kanıtlıyor. Bir slaytın sunumda kalıp kalmayacağının testi
şu: **bu cümlenin hangi parçasını taşıyor?** Taşımıyorsa uzun destede kalsın.

---

## Açılış — 2 dakika

*Her sürümde aynı. Slayt yok, ya da kapak açık.*

**"Üç aydır iki sistemi inceliyorum. Microsoft'un AutoGen'i ve OpenClaw adında
bir ajan harness'ı. Bugün size ne öğrendiğimi değil, neyi ölçtüğümü
anlatacağım."**

Duraklama.

**"Baştan söyleyeyim, çünkü sonunda söylersem geç olur. AutoGen bakım modunda —
yani yeni özellik almayacak. OpenClaw'ı da olduğu gibi kurmamızı önermiyorum.
Buna rağmen ikisini de anlatıyorum, ve sebebi şu: ikisi de bizim çözmemiz
gereken problemleri bizden önce çözmüş, ve nerede durduklarını yazmışlar."**

**"Bugünün çıktısı bir ürün önerisi değil. Bir değerlendirme çerçevesi. Hangi yolu
seçersek seçelim — kendimiz yazalım, hazır alalım, ya da bir tanesini
gömelim — soracağımız sorular aynı olacak."**

Yönerge: buraya kadar 90 saniye. Hızlı geç, ayrıntı verme. Amaç izleyicinin
"bu adam bir şey satmıyor" diye karar vermesi.

---

## Perde 1 — Motor: AutoGen ne veriyor

*Kısa sürüm: `[S2] [S9] [S18]` — 3 slayt, 3 dakika*
*Tam sürüm: `[S2] [S3] [S6] [S7] [S9] [S10] [S14] [S15] [S17] [S18]` — 10 slayt, 10 dakika*
*Derin sürüm: 2–37 arası, bileşen ve desen sayfaları dahil*

### [S2] Üç katman

**"AutoGen tek bir kütüphane değil, üst üste duran üç katman. En altta aktör
modeli var: ajanlar birbirini çağırmıyor, bir runtime'a mesaj veriyor. Ortada
günlük iş var: hazır ajan, beş takım tipi. En üstte dış dünya: model istemcileri
ve MCP."**

**"Pratik kuralı tek cümle: yukarıdan başla. Ortadaki katmanın çözdüğü bir
problemi en altta yeniden çözmek, aynı işi daha az testle yapmaktır."**

### [S3] Aktör modeli

*Tam ve derin sürümde.*

**"Bir ajan başka bir ajanın nesnesini tutmuyor, metodunu çağırmıyor. Runtime'a
mesaj veriyor, teslimatı runtime yapıyor."**

**"Bunun bir bedeli var: bir şey ters gittiğinde 'kim kimi çağırdı' sorusunun
cevabı yığın izinde görünmüyor, çünkü ortada çağrı zinciri yok."**

**"Karşılığında aldığımız şey bugün bizim için kritik: bütün mesajlar tek bir
noktadan geçtiği için, müdahale ve ölçüm oraya bir kez takılıyor. Onay kapısını
kurabilmemizin sebebi bu."**

### [S6] Topic — sinsi tuzak

*Tam ve derin sürümde. Bu slayt izleyicinin uyanık kaldığı yer, acele etme.*

**"Topic'in iki parçası var. Biri mesajın ne olduğunu söylüyor, diğeri hangi iş
için olduğunu. Ve ikincisi doğrudan ajanın kimliğine dönüşüyor."**

**"Şimdi tuzak. Bu ikinci parçayı her istekte değiştirirseniz, her istekte
yepyeni bir ajan doğuyor. Önceki ajanın belleği silinmiyor — sadece ulaşılamaz
hale geliyor."**

Duraklama.

**"Ve bu daha kötü. Çünkü hiçbir hata çıkmıyor. Sistem çalışıyor, ajanlar cevap
veriyor, sadece hiçbiri bir öncekini hatırlamıyor. Bunu üretimde bulmak,
testte bulmaktan çok daha pahalı."**

### [S7] Fan-out / fan-in

*Tam ve derin sürümde.*

**"Üç analist aynı duyuruya abone olduğu için tek bir yayın üçünü birden
uyandırıyor ve paralel koşuyorlar. Dikkat edin: ortada 'paralel çalıştır' diyen
bir çağrı yok. Paralellik, aynı duyuruyu birden fazla kişinin dinlemesinin
sonucu."**

**"Ama sonuçları toplamak ayrı bir iş, ve burada ölçtüğümüz bir arıza var: bir
dal sessizce ölürse toplayıcı sonsuza kadar bekliyor. Sebebi mekanik — yayın
fonksiyonu hata fırlatmıyor, yalnız logluyor."**

**"Çözümü tek satır: sayacı `finally` bloğunda artıracaksınız. Bu desenin fiyatı
o satır."**

### [S9] AssistantAgent ve tool döngüsü

*Her sürümde. Destenin en yüksek getirili slaytı.*

**"Bu destede tek bir şey hatırlayacaksanız bu olsun."**

**"Model bir tool çağırıyor, tool koşuyor, sonuç dönüyor. Ve varsayılan ayarda
tur tam orada bitiyor. Yani model, tool'un bulduğu sonucu hiç görmüyor."**

**"Kullanıcıya giden cevap, tool hiç çağrılmamış gibi yazılmış oluyor. Ve hiçbir
hata çıkmıyor. Loga bakarsınız: tool çağrılmış, sonuç dönmüş, her şey yolunda
görünüyor. Ama cevap yanlış."**

**"Eksik olan çağrı değil, ikinci model turu."**

Yönerge: burada bir soru gelirse iyi işaret — izleyici gerçekten dinliyor.

### [S10] Beş takım ve [S15] cache sınırı — maliyetin nereden geldiği

*Tam ve derin sürümde. İkisini birleştirerek anlat.*

**"Beş takım tipi var ve hepsi aynı işi yapıyor: birden çok ajanı sırayla
konuşturmak. Aralarındaki tek fark, sıraya kimin karar verdiği."**

**"Bu masum bir tercih değil. Ölçtük."**

`[S15]` göster, ama asıl sayıyı sen söyle:

**"Aynı görev, aynı ajanlar, aynı model. Yalnız orkestrasyon deseni değişiyor.
En ucuzu 204 token, en pahalısı 334. Aradaki fark yüzde 63,7."**

**"Ödediğiniz şey zekâ değil, yönlendirme özerkliği. Sırayı modelin seçmesi bir
lüks, ve fiyatı var."**

**"Aynı slaytta ikinci bir şey var: model hiçbir şey hatırlamıyor. Her turda
bağlamın tamamı yeniden gönderiliyor. Sağlayıcılar isteğin başındaki değişmeyen
kısmı ucuza faturalandırıyor — ama önbellek önekten çalışıyor, ilk farklı bayta
kadar. Yani prompt'un başına değişken bir şey koyarsanız, arkasındaki her şey
düşer. Tek bir tarih damgası bütün sistem prompt'unu yakabilir."**

### [S14] Kaç desen var

*Tam ve derin sürümde. Kısa geç.*

**"Resmî kılavuzda sekiz desen var. Bunu söylüyorum çünkü internette 'AutoGen'in
dokuz deseni' diye tablolar dolaşıyor ve o tasnifler kılavuzun değil, yazarın."**

**"Ben de bir kez dokuz yazdım. Kaynağa dönünce sekiz olduğunu gördüm, üçünü
uydurmuşum, ikisini atlamışım. Düzelttim."**

Yönerge: bu itiraf küçük ama işlevi büyük — geri kalan sayıların nasıl
denetlendiğini gösteriyor. Atlamak cazip gelecek, atlama.

### [S17] Dört sessiz varsayılan

*Tam ve derin sürümde.*

**"Bir çerçevede en pahalı şey, makul görünen ama sessizce yanlış olan
varsayılandır. AutoGen'de dört tane var ve dördü de sistemi çalıştırıp sonucu
bozuyor."**

**"Tool sonucu modelden saklanıyor. Bağlam verilmezse ajanın belleği hiç olmuyor.
Akış kapalıysa token yayını hiç çıkmıyor. Sonlandırma koşulu yoksa takım
tavansız koşuyor — yani faturayı modelin kararına bırakıyorsunuz."**

**"Dördünü de açıkça yazmak, bir sonraki sürümde varsayılan değişse bile korur."**

### [S18] Geriye kalan — **perde dönüşü**

*Her sürümde. Sunumun menteşesi burası, ağır söyle.*

**"Buraya kadar anlattığım her şey çalışıyor. AutoGen sağlam bir motor veriyor:
aktör modeli, takımlar, akış, sonlandırma, olay yayını."**

Duraklama.

**"Vermediği şey kontrol düzlemi. Üçünü tek tek söyleyeyim, çünkü 'hiç yok'
dersem haksızlık etmiş olurum — üçünün de yakını var, ama üçü de eksik."**

**"Kapı. Yani her tool çağrısının geçtiği, politikayla reddedilebilen tek nokta.
AutoGen'de buna en yakın şey mesaj katmanında duruyor ve bir mesajı
düşürebiliyor. İki sorunu var: gördüğü şey 'şu ajana bir mesaj gitti', 'bu ajan
şu komutu şu argümanlarla çalıştırmak istiyor' değil. Ve düşürülen mesaj ajana
gerekçe döndürmüyor, dolayısıyla ajan başka bir yol da deneyemiyor."**

**"Onay. Bu var — ama tek bir ajan sınıfında, deneysel etiketiyle, ve yalnız kod
çalıştırmayı kapsıyor. Verilmezse sadece bir uyarı yazıp kodu çalıştırıyor. Onay
isteğinin içinde donmuş bir plan yok, çalıştırma anında hiçbir şey yeniden
doğrulanmıyor."**

**"Ve bir ayrıntı var ki bize doğrudan dokunuyor: onay fonksiyonu olan bir ajan
yapılandırmaya yazılamıyor. Kod bunu açıkça reddediyor. Yani kapılı bir ajanı
yapılandırma olarak dağıtamıyorsunuz — ya kapıyı bırakırsınız ya da kodu."**

**"Denetim kaydı. Olay yayını var ve iyi bir yüzey. Ama o Python'ın kendi
logging'i: teslim garantisi yok, yazılamazsa koşu devam ediyor, kimlik yok,
kurcalama kanıtı yok. Ve içine modelin bütün mesajlarını koyuyor."**

Duraklama.

**"Bu sonuncusunu aklınızda tutun. Birazdan OpenClaw'ın denetim kaydını
anlatacağım ve onun sorunu tam tersi olacak: hiç içerik tutmuyor. İkisi de bizim
istediğimiz şey değil, ve ikisi farklı yönde yanlış."**

**"Bunlar eksiklik değil, kapsam kararı. Ama bizim doldurmamız gereken bir
boşluk. Ve bu boşluğu ters uçtan doldurmuş bir sistem var."**

Deste değiştir.

---

## Perde 2 — Kuşatma: OpenClaw nasıl çözmüş

*Kısa sürüm: `[S2] [S5] [S18] [S19]` — 4 slayt, 5 dakika*
*Tam sürüm: `[S2] [S3] [S5] [S6] [S10] [S12] [S16] [S18] [S19]` — 9 slayt, 12 dakika*
*Derin sürüm: tamamı + niş desteden 4–6 slayt*

### [S2] Kuşbakışı

**"Solda kanallar var: web, komut satırı, sohbet uygulamaları, cihazlar. Sağda
yetenekler: ajan runtime'ı, tool'lar, bellek. Ortada Gateway duruyor ve içinde
kim olduğunuz, neye yetkili olduğunuz, ne kaydedildiği var."**

**"Mimarinin özü tek cümle: ajan runtime'ı bunların hiçbirini bilmiyor. Kimlik,
yetki ve denetim ajan döngüsünün dışında kalıyor."**

**"Ve tam bu yüzden değiştirilebilir. Ajan motorunu söküp yerine başkasını
koyabilirsiniz, kontrol düzlemi yerinde kalır. Bizim mimarimizin de dayandığı
fikir bu."**

### [S3] Üç kontrol ekseni

*Tam ve derin sürümde.*

**"'İzin' tek bir kavram değil, üç ayrı soru: kod nerede koşuyor, hangi tool
çağrılabiliyor, ve kutunun dışına çıkmanın bir yolu var mı."**

**"Bunu neden anlatıyorum? Çünkü bir cümle var ve yanlış: 'yazma tool'unu
kapattık, artık salt-okunur.' Yanlış — çünkü tool politikası tool'u yalnız
adına göre filtreliyor, komutun içinde ne yapıldığına bakmıyor. Kabuk serbestse
yazmak zaten mümkün."**

**"Bu bizim de yapabileceğimiz bir hata, ve yapılırsa denetim raporunda 'salt
okunur' diye yazacak."**

### [S5] Onay komuta değil, plana bağlanır

*Her sürümde. Kurumsal izleyicinin en çok ilgileneceği slayt.*

**"Naif bir onay akışında bir boşluk var: kullanıcı gördüğü şeyi onaylıyor, ama
onayla çalıştırma arasında geçen sürede argümanlar değişmiş olabiliyor."**

**"OpenClaw onay isteğinin içine kanonik bir plan koyuyor — çalışma dizini, tam
argüman listesi, sabitlenmiş dosya yolu. Onaydan sonra saklanan planı
çalıştırıyor, çağıranın sonradan gönderdiğini değil."**

**"Ve bir ayrıntı: onay bir dosyaya bağlıysa ve dosya onaydan sonra değiştiyse,
koşuyu reddediyor."**

**"Yani onayladığınız şey bir cümle değil, donmuş bir plan. Bu, bir denetim
sorusuna cevap verebilmenin ön koşulu."**

### [S6] Dış içerik veri, talimat değil

*Tam ve derin sürümde.*

**"Modelin bağlamına giren her şey aynı görünüyor: metin. Sizin sistem talimatınız
da metin, müşterinin gönderdiği PDF de metin."**

**"O PDF'in içinde 'önceki talimatları yok say' yazıyorsa, model bunu neden
talimat saymasın? Ayırt edecek bir işaret yok."**

**"OpenClaw dış içeriği rastgele kimlikli bir sınırın içine koyuyor. Kimlik sabit
olsaydı içerik kendi kapanış etiketini yazıp kutudan çıkardı."**

**"Ve şüpheli desenleri yalnızca loglıyor, engellemiyor. Sebebini kendileri
yazmış: desen eşleştirmeyle injection engellenemez. Bunu açıkça söylemeleri
güvenilirlik işareti — çünkü aksini iddia eden ürünler var."**

### [S10] Belleğin güvenlik sınırı

*Tam ve derin sürümde.*

**"Belleğe zehirli bir bilgi girdikten sonra onu içerik taramasıyla yakalamak
güvenilir değil. 'Şu şirketin genel müdürü X' cümlesi doğru mu yanlış mı, metne
bakarak anlaşılmıyor."**

**"Bu yüzden savunma 'kötü belleği sonradan bul' değil, 'kötü bellek terfi
edemesin' üstüne kurulmuş."**

**"Her kaydın bir köken sınıfı var. Kapalı bir kümeden seçiliyor ve veritabanında
ayrı bir sütunda duruyor — yani modelin düzyazıyla yazamayacağı bir yerde. Ve
sınıflandırma muhafazakâr: kökeni belirlenemeyen dışsal bir içerik güvenilmez
sayılıyor, asla sahip varsayılmıyor."**

### [S12] Zamanlama yığını

*Tam ve derin sürümde.*

**"'Arka planda iş çalıştır' tek bir ihtiyaç değil. Beş tetikleyici türü var ve
son ikisi zamana hiç bakmıyor — onlar olay kaynağı."**

**"Karar da iki ayrı zamanlayıcıya bölünmüş, ve ayrımın sebebi pratik: 'her sabah
dokuzda rapor' izolasyon istiyor, 'ara sıra gelen kutusuna bak' bağlam istiyor.
Aynı mekanizma ikisini birden iyi yapamıyor."**

### [S16] Dayanıklı durum — ama durable execution değil

*Tam ve derin sürümde. Bu slayt tekniği ikna eder.*

**"Her şey veritabanında: konuşma geçmişi, yarıda kalan tur, zamanlanmış işler.
Süreç ölse de durum kaybolmuyor."**

**"Ama kurtarmanın ne olduğu önemli. Gateway ajana sentetik bir sistem mesajı
yazıyor: 'önceki turun kesildi, mevcut transkriptten devam et.'"**

Duraklama.

**"Yani devam eden şey bir fonksiyon değil, bir istem. Deterministik replay yok.
Tamamlanmış adımların hatırlanması yok."**

**"Somut sonucu şu: bir tur yan etkili bir tool'u çağırdıktan sonra çöktüyse —
mesela bir mesaj gönderdikten sonra — o tool'un ikinci kez çağrılmasını mekanik
olarak engelleyen hiçbir şey yok. Tek koruma, modelin transkripti okuyup fark
etmesi."**

**"Bunu söylüyorum çünkü 'dayanıklı' kelimesi bir ürün sayfasında görüldüğünde
insanlar bunu anlıyor sanıyor. Anlamıyor."**

### [S18] İki kayıt hattı — **KKB için en önemli slayt**

*Her sürümde. Buraya en çok zamanı ayır.*

**"OpenClaw'ın denetim kaydı içerik tutmuyor. Prompt yok, tool argümanı yok, URL
yok, komut çıktısı yok. Kimlikler kurulum-yerel bir anahtarla takma ada
çevriliyor."**

**"Ve kendi sınırını söylüyor, aynen alıntılıyorum: 'Bu korelasyondur,
anonimleştirme değildir.'"**

**"Dahası kayıt best-effort. Kuyruk dolarsa satır düşüyor, koşu devam ediyor."**

Duraklama.

**"Bir geliştirici aracı için bu doğru öncelik. Log uğruna işi durdurmak saçma
olurdu."**

**"Bizim için tersi gerekiyor. Uyum hattı kayıpsız, senkron ve fail-closed
olmalı — yazılamıyorsa koşu düşmeli."**

**"Yani OpenClaw'dan alacağımız şey bu mekanizma değil. Aldığımız şey ayrım:
operasyonel hat ile uyum hattı farklı garantiler ister, ve tek bir log ikisini
birden karşılayamaz. Bu ayrımı yapmayan her mimaride, denetime götürdüğünüz
kayıt aynı zamanda performans için budanmış olan kayıttır."**

### [S19] Geriye kalan tek cümle — **perde dönüşü**

*Her sürümde.*

**"Bu destedeki her mekanizma bir kurumsal asistana taşınabilir. Kapı, donmuş
plan, köken sınıfı, iki kayıt hattı — hepsi."**

**"Taşınamayan tek şey şu: OpenClaw tek bir güvenilen operatörün etrafında
tasarlanmış. Belgelerindeki bütün 'bu bir güvenlik sınırı değildir' cümleleri
buradan geliyor. O modelde zaten herkes güvenilir, dolayısıyla ayrımlar bir
kolaylıktan ibaret."**

**"Birbirine güvenmeyen departmanların olduğu bir kurumda aynı cümleler birer
açık haline geliyor."**

**"Yapılacak iş net: mekanizmaları al, güven modelini yeniden kur."**

---

## Perde 3 — Bizde ne var

*Kısa sürüm: yalnız sayı tablosu, slayt yok, 2 dakika*
*Tam sürüm: 5 dakika*
*Derin sürüm: niş destesinden `[S17]` "ne taşınır" slaytı + canlı demo*

Yönerge: buraya kadar iki dış sistem anlattın. İzleyici şimdi "peki sen ne
yaptın" diye soracak. Sormadan cevapla.

**"Bunlar okuduğum sistemler değil, ölçtüğüm sistemler. Ölçmek için bir boru
hattı yazdım ve bugün çalışıyor."**

**"On iki bin dokuz yüz satır. Dört bin beş yüzü AutoGen'e dokunuyor, sekiz bin
dört yüzü dokunmuyor."**

**"Bu ikinci sayı önemli. Çünkü kontrol düzlemini taşıyan altı modülde — onay,
kanca, zamanlayıcı, tool yüzeyi, koşu defteri, röle — AutoGen'den tek bir satır
yok. Motor değişirse o altı modül yerinde kalır."**

**"Üç yüz seksen bir test var ve hepsi geçiyor. Bugün koşturdum, on bir
saniyede."**

Demo yapılacaksa aşağıdaki üç perdeyi koş. Toplam **3 dakika**, ve üçü de
biraz önce sözle iddia ettiğin şeyleri ekranda kanıtlıyor.

### Demo — üç perde, 3 dakika

Yönerge: demo bir özellik turu değil. Üç iddiayı kanıtlıyor ve başka hiçbir şey
göstermiyor: **izleme gerçek**, **kapı gerçek**, **onay tüketiliyor**.
Ölçtüm — sayılar 18 Ağustos'ta bu makinede alındı.

**① İzleme şeridi · 30 sn**

Panele sıradan bir soru yaz. Aşağıda dört satır beliriyor:
`context → model → stream → done`. Yaklaşık bir saniye.

**"Bu şerit süs değil. Her satırda gerçek bir sınıf adı, gerçek bir dosya yolu
ve kılavuzda karşılık gelen satır numarası var. Şu an tek model çağrısı yapıldı
ve 3.379 token harcandı — ekranda yazıyor."**

**② Kapı ve ikinci tur · 60 sn — asıl perde bu**

Belgelerde arama gerektiren bir soru yaz. Şerit uzuyor:

```
context → model → tool_request → gate → tool_exec → tool_result → loop → model → stream → done
```

**"İşte biraz önce anlattığım iki şey aynı ekranda."**

`gate` satırını göster — altında kendi notu yazıyor:

**"Kapı burada. Ve altındaki not şunu söylüyor: AutoGen'de böyle bir şey yok.
Bu satır bizim yazdığımız bir workbench sarmalayıcısı, ve her tool çağrısı
buradan geçiyor."**

Sonra `loop` satırını göster:

**"Ve şu satır: modelin tool sonucunu görüp ikinci kez konuştuğu an.
`max_tool_iterations` varsayılanda kalsaydı bu satır hiç olmayacaktı, model
bulduğu bilgiyi hiç görmeyecekti, ve cevap yanlış olacaktı. İki model çağrısı,
bir tool çağrısı, 8.175 token."**

**③ Onay kapısı · 90 sn — KKB perdesi**

Yaz: `/openclaw schedule her gün 05:00 | bana merhaba de`

İş **çalışmıyor**, onay kartı çıkıyor. Kartta `method`, `when`, `ask` ve `to`
alanları görünüyor.

**"Bu kart bir tarif göstermiyor, birazdan çalışacak şeyin kendisini
gösteriyor. Onaylamazsam hiçbir şey olmuyor."**

Onayla, aynı satırı tekrar gönder. İş gerçekten kuruluyor —
`0 5 * * *`. Listeyi göster, sonra sil.

**"Ve onay tüketildi. Aynı satırı şimdi bir daha gönderirsem yeniden soracak,
çünkü onaylanan şey 'bu tool' değil, 'bu tool bu argümanlarla, bir kez'."**

### Demo çökerse

Yönerge: hiçbirini gizleme, üçünün de dürüst bir cümlesi var.

| ne çökerse | ne yap | ne söyle |
|---|---|---|
| Canlı model cevap vermiyor | Kuru moda geç | *"Kuru modda koşuyorum: model cevapları önceden yazılmış. Ama kontrol akışı gerçek, ve zaten göstermek istediğim o."* |
| OpenClaw Gateway kapalı | ③'ü atla | *"Zamanlama OpenClaw'a devredilmiş durumda ve Gateway şu an ayakta değil. Bu tam da az önce söylediğim risk: `Linger=no`."* |
| Panel hiç açılmıyor | Üçünü de atla | *"Ekran görüntülerine geçiyorum — ve dönünce canlı gösteririm."* Sunumu demoya bağlama. |

**Demo öncesi kontrol** — sunumdan 10 dakika önce, tek tek:

- [ ] `curl -s localhost:8000/api/health` → `"ok":true` ve `"live_llm":true`
- [ ] Bir soru sor, şeritte dört satır çıkıyor mu
- [ ] `openclaw gateway status` → servis ayakta
- [ ] `/openclaw schedule` → mevcut işleri listeliyor
- [ ] **Test artıklarını sil.** Önceki denemelerden kalan işler OpenClaw'ın
      deposunda kalıcı; listede yalnız gerçekten olması gerekenler dursun.

### Söylenecek üç dürüst sınır

Yönerge: bunları sen söylersen güç kazanırsın, biri bulursa kaybedersin.

**"Üç şeyi de söyleyeyim, çünkü sonra sorulursa zaten çıkacak."**

**"Bir: 'AutoGen'i ince bir arayüzün arkasına gömdük' derken tam doğru
söylemiyorum. Gateway katmanında bu tuttu — orada AutoGen üç import satırı. Ama
bütününde on beş modüle sızmış. Yarın değiştirilebilecek olan gateway, motor
katmanı değil."**

**"İki: kendi zamanlayıcımız yazılı ve test edilmiş, ama sisteme bağlı değil.
Şu an zamanlama OpenClaw'a devredilmiş durumda, ve bu kontrol düzlemi kararımızla
tutarsız. Düzeltilecekler listesinde."**

**"Üç: rakip çerçeveler hakkındaki karşılaştırmalarım okumaya dayanıyor. AutoGen
ve Google ADK'yı gerçekten koşturdum. LangGraph, CrewAI ve diğerlerini
koşturmadım — o cümleleri kesin diye sunmuyorum."**

---

## Kapanış ve istenen karar — 3 dakika

*Her sürümde. Slayt yok. Bu kısmı ezberle.*

**"Toparlayayım."**

**"AutoGen bize bir motor veriyor, kontrol düzlemi vermiyor. OpenClaw kontrol
düzlemini çözmüş ama tek bir güvenilen operatör varsayarak çözmüş, ve o varsayım
bizde geçerli değil."**

**"Önerim üç ayrı ilişki kurmak. AutoGen'i gömüyoruz — bir motor olarak, ince bir
arayüzün arkasına. OpenClaw'ı öğreniyoruz — karar kurallarını alıyoruz, kodunu
değil. Ve OpenClaw'ı mühendislik takımında araç olarak kullanmaya devam
ediyoruz."**

**"Atlas olarak OpenClaw kurmuyoruz."**

Duraklama.

**"İstediğim şey bugün bir ürün kararı değil. Doksan günlük planın birinci fazı
için onay istiyorum: onay kapısı, uyum kayıt hattı ve tek bir dar kullanım. Otuz
gün, tek kişi."**

**"Birinci faz bittiğinde elimizde ölçülmüş bir şey olacak, ve kalan iki fazın
süresini o zaman konuşabiliriz. Şimdi konuşursak tahmin etmiş oluruz."**

**"Sorular?"**

---

## Süre daralırsa

Yönerge: toplantının yarısı gittiyse panikleme, şu üçünü sırayla feda et.

1. **Önce Perde 1'in ortasını at** — `[S3] [S6] [S7] [S14] [S17]`. `[S9]` ve
   `[S18]` kalsın; biri en pahalı tuzağı, diğeri perde dönüşünü taşıyor.
2. **Sonra Perde 2'nin ortasını at** — `[S3] [S6] [S10] [S12]`. `[S5]`, `[S18]`,
   `[S19]` asla atılmaz: onay, denetim ve güven modeli.
3. **En son Perde 3'ü sayıya indir** — üç cümle: on iki bin dokuz yüz satır,
   altı modülde sıfır AutoGen, üç yüz seksen bir test geçiyor.

**Kapanış hiçbir koşulda kısaltılmaz.** İstenen karar söylenmezse sunum
bilgilendirme olur, ve bilgilendirmeden karar çıkmaz.

---

## Sunum öncesi kontrol

- [ ] Üç PDF açık ve doğru sırada
- [ ] Demo yapılacaksa gateway ayakta — `openclaw gateway status`
- [ ] Soru hazırlığı sayfası ikinci ekranda ya da telefonda
- [ ] Kapanıştaki "istediğim şey" cümlesi ezberde
- [ ] Dört düzeltmenin (`Z5`) hangi sırayla söyleneceği belli

**Son hatırlatma:** sunumun ikna gücü sayılardan değil, sayıların nasıl
denetlendiğinden geliyor. Bir şeyi bilmiyorsan "ölçmedik" de. Bu cevap zayıflık
değil, geri kalan her sayının teminatı.
