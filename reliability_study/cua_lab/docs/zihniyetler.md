# Ajan Güvenilirliğinde Zihniyetler

Bu belge mekanizmaları değil, **düşünme biçimlerini** anlatıyor.

Bir ajanı nasıl durduracağınıza karar vermeden önce, onun neden durmadığına dair bir
görüşünüz olması gerekiyor. Buradaki her yaklaşım aslında bir teşhis: "ajanlar şu yüzden
bozulur" diyor ve çözümünü o teşhise göre kuruyor.

İşin ilginç yanı şu — teşhislerin çoğu doğru. Ama farklı şeyler hakkında doğru. Bu yüzden
hiçbiri tek başına yetmiyor, ve bu yüzden hepsini ayrı ayrı uygulayıp yan yana koymak
mantıklı.

---

## Başlangıç noktası: ajan neden durmuyor?

Bir dil modeli tek başına durur. Sorarsınız, cevap verir, biter. Ajan dediğimiz şey o
modeli bir döngünün içine koymaktır — cevabı al, bir şey yap, sonucu geri ver, tekrar sor.
Durma sorunu tam olarak buradan çıkıyor: **döngüyü kim kesecek?**

En doğal cevap "model kendisi karar versin" olur. Ama pratikte en kötü cevap bu. Model
her turda yalnızca bir sonraki adıma bakıyor; işin genel gidişatını, kaç kez aynı şeyi
denediğini, ne kadar para harcadığını görmüyor. Görse bile o bilgiyi doğru yorumlayacağının
garantisi yok.

Arize'ın yazısındaki cümle bu durumu net koyuyor:

> *"Her döngüye modelin yargısına bağlı olmayan sert bir durdurma gerekiyor, çünkü model,
> işin bitip bitmediği konusunda yanılması en muhtemel bileşendir."*

Bu cümleyi kabul ettiğiniz anda soru değişiyor: model karar vermeyecekse, **kim verecek ve
neye bakarak?** Bundan sonraki her şey bu sorunun farklı cevapları.

---

## Bu belge neye göre sıralandı

On yedi zihniyet **en basitten en karmaşığa** dizildi. Sıralamanın ölçütü şu üç soru:

1. **Ne kadar durum tutuyor?** Bir sayaç mı, son yirmi adımın geçmişi mi, kontrol noktası mı?
2. **Kaç kavram gerekiyor?** Tek eşik mi, birbirini tetikleyen basamaklar mı?
3. **Sisteme nereden dokunuyor?** Sadece döngüyü mü kesiyor, akışın şeklini mi değiştiriyor,
   yoksa koşumun tamamen dışında mı duruyor?

Ortaya altı seviye çıkıyor. Her seviye bir öncekinin çözemediği bir şeyi çözüyor — yani sıra
aynı zamanda **bir problemin nasıl derinleştiğinin hikâyesi.**

| Seviye | Adı | Ne yapıyor | Zihniyetler |
|---|---|---|---|
| **1** | Sayaç | Bir sayı tut, aşınca kes. Durum yok. | 1 · 2 · 3 · 4 |
| **2** | Pencere | Son N adımı sakla, karşılaştır. | 5 · 6 · 7 · 8 |
| **3** | Dünya | Modelin dışından kanıt al: oran, para, ortam. | 9 · 10 · 11 |
| **4** | Şekil | Döngünün oluşmasına baştan izin verme. | 12 · 13 |
| **5** | Kademe | Çok sinyal + basamaklı müdahale + geri alma. | 14 · 15 |
| **6** | Karar | Ne zaman duracağı değil, ne yapacağı ve eşiğin ne olacağı. | 16 · 17 |

Numaralar bu belgenin iki bölümünde de aynı: aşağıdaki anlatım ve sondaki teknik ek, aynı
on yediyi aynı sırada ele alıyor.

Eski bir okuma biçimi de duruyor — zihniyetleri *inandıkları şeye* göre altı aileye ayırmak.
O tablo "Toparlarsak" bölümünde, çapraz referans olarak.

---
---

# SEVİYE 1 · Sayaç

> **Ortak fikir:** Ne yaptığını anlamaya çalışma. Kaç yaptığını say.

Bu seviyenin tamamı tek bir tamsayıyla çalışıyor. Geçmiş tutmuyorlar, karşılaştırma
yapmıyorlar, "aynı şey ne demek" sorusunu hiç sormuyorlar. Yazması bir öğleden sonra sürer.

Zayıflığı da gücüyle aynı yerden geliyor: size **ne** olduğunu söylemiyorlar, sadece **çok**
olduğunu söylüyorlar.

Dördü de aynı sayacı tutuyor. Aralarındaki tek fark **sayaç dolduğunda ne olduğu** — ve
şaşırtıcı biçimde asıl tartışma orada.

## 1 · Modele sorma, say

*(`arize-control` — Seviye 1)*

Bu bir güvensizlik beyanı, ve bilinçli olarak öyle.

Beş durma koşulu var — görev tamamlandı, adım limiti doldu, bütçe bitti, süre doldu, hata
politikası devreye girdi — ve aralarında bir hiyerarşi kuruluyor. **Adım limiti birincil,
çünkü modelin söylediği hiçbir şeye bağlı değil.**

Diğer bütün sinyaller yorum gerektiriyor. "Görev tamamlandı" modelin görüşü. "Hata
politikası" hatanın ne sayıldığına bağlı. Adım sayısı ise tartışmasız: on iki adım attıysa
on iki adım atmıştır.

Bu zihniyetin ikinci ve az fark edilen katkısı: **durma sebebini kaydetmek.** Arize'ın
ifadesi şu — "tamamlandı", "adım limiti", "bütçe aşıldı" ve "hata" farklı sonuçlardır;
hepsini tek bir başarı oranına karıştırmak neyin yanlış gittiğini gizler.

Bir sistemin %70 başarı oranı varsa, kalan %30'un ne olduğu her şeyi değiştirir. Hepsi
adım limitine takılıyorsa limitiniz dardır. Hepsi hata veriyorsa araçlarınız bozuktur.
Ortalama ikisini de gizler.

**Neden ilk sırada:** Bu belgedeki en ucuz ve en kandırılamaz mekanizma. Başka hiçbir şey
uygulamayacaksanız bunu uygulayın. Kalan on altısı bunun *üstüne* gelir, yerine değil.

## 2 · Nazikçe isteme, seçeneksiz bırak

*(`agentscope-grace` — Seviye 1)*

Bir öncekiyle aynı sayaç, ama sayaç dolduğunda koşum bitmiyor.

Ajana **beş turluk bir lütuf bütçesi** veriliyor. Ve o turlarda araç seçimi kilitleniyor —
ajan artık yeni araç çağıramıyor, yalnızca cevap üretebiliyor.

Nazik bir rica değil, mekanik bir kısıt. "Lütfen bitir" demek yerine bitirmekten başka
seçenek bırakmıyor.

Çözdüğü problem şu: sert bir tavan ajanı işin ortasında keser ve o ana kadar yaptığı iş
çöpe gider. Bu, aynı tavanı korurken çıkışa bir rampa ekliyor.

**Ek karmaşıklık:** iki sayaç yerine bir sayaç artı bir bayrak, ve modele giden istekte
`tool_choice` alanını değiştirme yeteneği. Hâlâ çok basit.

## 3 · Uyarma bile

*(`hermes-no-pressure` — Seviye 1)*

Yine aynı sayaç. Ama bu sefer soru şu: **sayaç dolmadan önce modele haber verilmeli mi?**

Sezgi "evet" der. Hermes de öyle demiş, denemiş, sonra kaldırmış. Kod yorumu şöyle:

> *"Ara basınç uyarıları yok — modelleri karmaşık görevlerde erken pes ettiriyordu."*

Yani "adımların azalıyor" mesajı modele "bu görev bana göre değil" gibi geliyor ve
gerçekten yapabileceği işi bırakıyor.

Onların çözümü: adım bütçesi boyunca **hiçbir uyarı yok.** Bütçe gerçekten dolduğunda tek
bir mesaj giriyor, ajana bir **lütuf çağrısı** veriliyor, ve o çağrıda da metin üretmezse
zorla özet isteniyor.

Ama aynı sistem **süre** bütçesinde %80'de uyarı veriyor. Yani eksene göre farklı karar
vermişler. Muhtemelen şundan: "sürenin azalıyor" mesajı "elindekiyle topla" diye okunuyor;
"adımın azalıyor" ise "yeterince iyi değilsin" diye.

**Ek karmaşıklık:** kod olarak sıfır — hatta *eksi*. Uyarıyı silmek bir satır. Zor olan
karar, kod değil.

## 4 · Modele saatini göster

*(`claude-advisory` — Seviye 1)*

Ve şimdi bir öncekinin tam tersi — bu belgedeki en öğretici karşıtlık.

Konuşmaya bir geri sayım işareti giriyor. Model ne kadar bütçesi kaldığını görüyor ve
azaldıkça işi toparlıyor.

Kritik nokta: bu **tavsiye, zorlama değil.** Model, kesilmesi bitirilmesinden daha zararlı
olacak bir işin ortasındaysa bütçeyi aşabiliyor. Sert tavan ayrı bir mekanizma olarak
duruyor.

Yani bu zihniyet bir guardrail'in *yerine* geçmiyor, üstüne geliyor. Amacı durdurmak değil,
**öngörülebilir bir iniş** sağlamak.

Bir de belgelenmiş bir yan etki var: bütçe görevi için açıkça yetersizse model işi hiç
denemiyor, agresif biçimde daraltıyor ya da erken duruyor. Yani "her ihtimale karşı düşük
tutayım" refleksi geri tepiyor.

**Üç ile dördü birlikte okuyun.** İki olgun sistem aynı soruya zıt cevap vermiş ve ikisi de
gerekçesini ölçmüş. Bizim aldığımız ders "hangisi haklı" değil, **eksene göre karar vermek**:
adım ekseninde uyarma, süre ekseninde uyar.

---

# SEVİYE 2 · Pencere

> **Ortak fikir:** Sıkışmış ajan kendini tekrar eder. Tekrarı görürsek sıkışmayı görürüz.

Birinci seviye "çok yaptı" diyebiliyordu ama "aynı şeyi yaptı" diyemiyordu. Bu seviye onu
söyleyebilmek için **son N adımın geçmişini** tutuyor.

Bedeli hemen ortaya çıkıyor: artık bir soruya cevap vermek zorundasınız — **"aynı şey" ne
demek?** Bu seviyenin bütün zorluğu o tanımda.

## 5 · Tekrarı tanımlama, çeşitliliği ölç

*(`strands-entropy` — Seviye 2)*

Bu seviyeye en kolay giriş, çünkü "aynı şey ne demek" sorusundan kaçıyor.

Tekrar saymak isteyen herkes bir tanım bataklığına giriyor. Ardışık tekrar mı sayacağız?
Peki A-B-A-B gidip gelirse? Peki A-B-C-A-B-C üçlü çevrimi olursa? Her desen için ayrı bir
tarama mı yazacağız?

Strands şunu soruyor yerine: **son N adımda kaç FARKLI şey oldu?**

Cevap düşükse ajan dar bir alanda dönüyordur. Aynı şeyi mi tekrarlıyor, iki şey arasında
mı gidip geliyor, üç adımlık bir çevrimde mi dönüyor — hiç fark etmez. Hepsi düşük
çeşitlilik üretir. Tek bir kural, bütün desenler.

Kodu da o kadar basit: pencereyi bir kümeye at, boyutuna bak.

Bunun bedeli de var: meşru olarak dar bir alanda çalışan işler de düşük çeşitlilik
gösterir. Bir dosyayı defalarca düzenleyip test eden ajan aslında ilerliyordur ama
çeşitlilik ölçüsü onu döngüde sanır.

## 6 · Aynı çağrıyı say

*(`openhands-stuck` — Seviye 2)*

Ve şimdi kaçılan soruyla yüzleşme. Ajanın yaptığı her eylemin bir parmak izini çıkar, son
N tanesine bak, aynısı tekrarlanıyorsa dur.

Buradaki asıl zorluk teknik ve sinsi. İki eylem aslında aynı ama sistem onları farklı
görüyor olabilir. Çünkü her araç çağrısının bir kimliği var, bir zaman damgası var, bir
istek numarası var — ve bunlar her seferinde değişiyor. Bu alanları da parmak izine
katarsanız, iki özdeş çağrı hiçbir zaman eşit çıkmaz. Dedektörünüz çalışır, hata vermez,
log basar — **ve hiçbir zaman hiçbir şey bulmaz.**

Bu, hata türlerinin en kötüsü. Çünkü sistem çalışıyor gibi görünür. Kimse fark etmez.

OpenHands bunu şöyle çözüyor: karşılaştırma yaparken kimlik alanlarını bilerek atıyor,
yalnızca içeriğe bakıyor. Ama bunun tersi de tuzak — atmayı abartırsanız bu sefer *gerçek*
argümanları da atmış olursunuz. O zaman "aynı aracı elli farklı dosyada çağırmak" da
döngü sayılır ve meşru bir toplu işlemi kesersiniz.

Yani doğru çizgi şurada: **her turda zaten değişen şeyleri at, işin anlamını taşıyan
şeyleri tut.**

Karşılığında çeşitlilikten fazlasını alıyorsunuz: beş ayrı desen ayırt ediliyor ve
"sıkışmak" ayrı bir terminal sonuç oluyor — başarı da değil, hata da değil.

## 7 · Çözümün kendisi sorun kaynağı olabilir

*(`openclaw-pingpong` — Seviye 2)*

Aynı pencere mantığı, ama üretimde yıllanmış hâli. Ve bir olgunluk göstergesi.

Uzun süren ajanların bağlamı dolar. Çözüm: sıkıştırma — eski konuşmayı özetleyip yerine
koymak. Ama bunun kendisi bir döngü kaynağı olabiliyor: sıkıştır, bağlam yine dolsun,
tekrar sıkıştır, tekrar dolsun.

Bu belgelenmiş bir hata deseni ve gerçek para yakmış. OpenClaw buna özel bir savunma
koymuş: bağlam sıkıştırıldıktan sonra üç deneme boyunca **ayrı bir koruma** kurulu kalıyor.

Yanına bir de üç kademeli eşik dizisi geliyor — uyarı, kritik, küresel devre kesici — ve
dedektörlerin adı var: `genericRepeat`, `knownPollNoProgress`, `pingPong`. İsim vermek
kozmetik değil: hangi dedektörün konuştuğunu bilmek, ne olduğunu bilmek demek.

Genel ders şu: bir güvenilirlik mekanizması eklerken, o mekanizmanın kendisinin yeni bir
hata kaynağı olup olmadığını sormak gerekiyor.

## 8 · Hareket ilerleme değildir

*(`loopguard-dignity` — Seviye 2, üçüncü seviyeye köprü)*

Bu seviyenin en derin fikri, ve tekrar saymanın kör noktasını kapatıyor.

Şöyle bir ajan düşünün: her turda gerçekten farklı bir şey deniyor. Farklı araç, farklı
argüman, farklı yaklaşım. Hiçbir tekrar dedektörü tetiklenmiyor, çünkü hiçbir şey
tekrarlanmıyor. Ama hiçbiri de işe yaramıyor. Ajan meşgul, sistem yerinde sayıyor.

Modexa'nın yazısındaki ifadeyle: *"Ajan sürekli hareket halindedir ancak sistem
ilerlemiyordur. Hareket etmek, ilerlemek demek değildir."*

Çözüm eyleme değil **sonuca** bakmak. Dünyanın durumunun bir özetini çıkar, her adımda
karşılaştır. Üç adımdır özet değişmiyorsa ajan ne yaparsa yapsın ilerlemiyordur.

Bunun computer-use'daki karşılığı doğrudan: eylem öncesi ekranın durumunu al, eylem
sonrası tekrar al. Değişmediyse o tıklama hiçbir şey yapmamış.

Burada da bir tuzak var: bazı eylemler **meşru olarak** hiçbir şey değiştirmez. `wait`
komutu ekranı değiştirmez ve değiştirmemesi normaldir. Ekran görüntüsü almak da öyle. Bu
ayrımı yapmazsanız modelin meşru beklemesini döngü sanarsınız.

Ve bir de durmanın *biçimini* önemsiyor: dört parçalı bir durma raporu ve "cevap
veremiyorum" diyebilme yolu. Durmak başarısızlık değil; kötü durmak başarısızlık.

**Bu madde köprü:** artık ajanın eylemlerine değil, **dünyanın hâline** bakıyoruz. Bir
sonraki seviyenin tamamı bunun üstüne kuruluyor.

---

# SEVİYE 3 · Dünya

> **Ortak fikir:** Ajanın en tehlikeli hatası döngüye girmek değil, **yanlış işi doğru
> sanmak.** Ona sorma, dışarıya bak.

İki seviyedir ajanın davranışına bakıyorduk — kaç adım attı, ne tekrarladı. Bu seviye
davranışı bırakıp **kanıta** geçiyor: test geçti mi, aracın hata oranı ne, fatura kaç dolar.

Ek karmaşıklık gerçek: artık ajanın dışında bir bilgi kaynağınız olmak zorunda.

## 9 · "Bitirdim" bir istektir, kanıt değil

*(`verify-gate` — Seviye 3)*

Ajan işini bitirdiğini söylediğinde bu bir bilgi değil, bir taleptir: "durmak istiyorum."
Bunu doğrudan kabul etmek, öğrencinin sınav kâğıdını kendisinin okuması gibi.

Loop Engineering yazısındaki benzetme yerinde: *"'Bitirdim dedi', ajan dünyasının 'benim
makinemde derleniyor'udur."*

Çözüm, bitirme iddiasını bir kapıya bağlamak. Testler geçti mi? Dosya gerçekten yazıldı
mı? Ekranda beklenen şey var mı? Kapı açılmazsa iddia reddediliyor — ama koşum
bitirilmiyor. Doğrulama sonucu ajanın gözlem akışına geri veriliyor ve döngü devam ediyor.
Ajan kendi hatasını görüp düzeltiyor.

Bu zihniyetin güçlü tarafı, doğrulamanın **ortama** dayanması. Modelin görüşü değil,
dünyanın hâli.

Ve kör noktası aynı yerden geliyor: ajan hiç "bitirdim" demezse bu kapı hiç açılmaz.
Sonsuza kadar dönen bir ajanı yakalamaz. Mutlaka bir bütçe stratejisiyle birlikte
kullanılmalı.

**Ek karmaşıklık:** göreve özel bir `verify()` yazmak zorundasınız. Bu seviyenin
"dünyaya bağlanma" bedeli.

## 10 · Ajan doğru davranıyor olabilir, araç bozuktur

*(`galileo-breaker` — Seviye 3)*

Bu zihniyet suçun yerini değiştiriyor.

Anlattığı vaka şu: ajan bir aracı çağırıyor, hata alıyor, tekrar deniyor, hata alıyor,
üçüncüde başarıyor. Kullanıcı doğru cevabı alıyor. Koşum "başarılı" bitiyor. Hiç kimse
ticket açmıyor.

Ama iki çağrı boşa gitti. Ve daha kötüsü — bugün geçici olan bu hata yarın kalıcı olursa,
her istek bütün deneme hakkını yakar ve alt akışı tıkar.

> *"Bu, hakkında hiç ticket açılmayan türden bir hata. Sessiz ama öldürücü."*

Bu deseni yakalamak için tekrar saymak işe yaramıyor, çünkü koşum başarıyla bitiyor.
Bakılması gereken şey **hata oranı**: bir aracın son çağrılarının yüzde kaçı hata verdi.
Oran eşiği aşarsa o araç için devre kesici devreye giriyor.

Ve bir teşhis dersi: *"Ajanlar doğru davrandı. Zafiyet veri çekme katmanında."* Bir
guardrail her zaman ajanı suçlamamalı.

**Ek karmaşıklık:** sayı değil **oran** tutmak, ve durumu ajan başına değil **araç başına**
saklamak. İlk kez sayaç birden fazla oluyor.

## 11 · Tokeni değil, doları say

*(`agentbudget-dollar` — Seviye 3)*

Token bir vekil ölçüdür. İki model arasında on beş kat fiyat farkı olabilir; aynı token
sayısı çok farklı faturalar üretir. Bu zihniyet doğrudan asıl birimi sayıyor — sistemin
dışındaki gerçek birimi.

Ve bir tasarım inceliği getiriyor: **nihai cevap payı.**

Sorun şu — sert bir tavana dayandığınızda ajan işin ortasında kesilir. Yaptığı iş çöpe
gider, çünkü sonucu toparlayıp size sunacak bütçesi kalmamıştır. Çözüm: bütçenin küçük
bir dilimini baştan ayır, sert limiti erken tetikle. Ajan tavana çarptığında elinde hâlâ
"bulduklarımı özetleyeyim" diyecek kadar para vardır.

Bir de tekrar tespitini zamana bağlıyor: "on çağrı" değil, "bir dakikada on çağrı". Yavaş
ama kronik bir döngü, sayı eşiğine hiç ulaşmadan saatlerce para yakabilir.

**Neden bu seviyede:** görünüşte Seviye 1'in sayacı gibi duruyor. Ama üstüne fiyat tablosu,
iç içe alt bütçeler, zaman pencereli tespit ve üç kademeli devre kesici biniyor. Sayaç
sayısı bir değil, bir ağaç.

---

# SEVİYE 4 · Şekil

> **Ortak fikir:** Önceki üç seviye döngüyü tespit etmeye çalışıyor. Peki neden oluşmasına
> izin veriyoruz ki?

Bu seviye tespit işini bırakıp **mimariyi** değiştiriyor. Karmaşıklığı algoritmada değil,
sistemi baştan öyle kurma zorunluluğunda — mevcut bir ajana sonradan eklenemiyor.

## 12 · Serbestliği kısıtla

*(`modexa-statemachine` — Seviye 4)*

Ajanlar belirsiz özgürlükleri sever; sistemlerin net durumlara ihtiyacı vardır.

Bu zihniyet döngüyü bir durum makinesine indirgiyor: ANLA → TOPLA → EYLEM → DOĞRULA →
YANIT → DEVRET. Ve yalnızca belirli geçişlere izin veriyor.

Ajan "doğrula" durumundan "topla"ya geri dönemiyorsa, o döngü hiç oluşmuyor. Tespit etmeye
gerek kalmıyor, çünkü mümkün değil.

Yanına bir de **geri dönüş merdiveni** koyuyor. Ajan kendi kendine tekrar deneme icat
etmiyor; sabit bir merdiveni tırmanıyor: bekleyip bir kez dene → aracı değiştir → kapsamı
daralt → kullanıcıya sor → elindekiyle cevap ver.

Merdivenin dördüncü basamağı — kullanıcıya sormak — kaynakta ayrıca tarif edilmiş, çünkü
kötü sorulan bir soru da bir kayıp: **tek soru sor**, beş değil · **neyin değişeceğini
açıkla** · kullanıcı umursamazsa **bir varsayılan sun**. Gerekçesi basit: *"40 adım boyunca
yanlış tahmin yürütmekten ucuzdur."*

Ve aynı kaynak bir şey daha söylüyor ki, bu belgedeki hiçbir mekanizmanın çözemediği bir
döngü sebebi. Beş sebep sayıyor, beşincisi başka hiçbir yerde geçmiyor: **ajanın "yanlış
yapmamak" üzerine optimize edilmesi.** Aşırı tedbir teşvik eden bir prompt, en güvenli yolu
"bir kez daha doğrula" yapıyor. Ajan bozuk değil — tam da söylendiği gibi davranıyor.

> *"Bir sorun çözücü inşa etmediniz. Bir riskten kaçınma makinesi inşa ettiniz."*

Bu döngünün kaynağı kod değil, **prompt.** Buradaki on yedi mekanizmanın hiçbiri onu
çözmüyor; hepsi ancak sonucu kesiyor. Guardrail'in kapsam sınırının en net örneği bu.

Bedeli esneklik. Bazı işler gerçekten öngörülemez ve durum makinesine sığmaz.

## 13 · Koşum başlamadan yakala

*(`autogen-static` — Seviye 4)*

En uç nokta. Ajan akışının grafiği kurulurken doğrulanıyor: bir çevrim var ve çıkış koşulu
yoksa sistem **hiç başlamıyor**, hata veriyor.

Çalışma zamanında sıfır maliyet, sıfır yanlış pozitif — çünkü çalışma zamanı hiç gelmedi.

Ama yalnızca **yapısal** döngüleri yakalar. Grafiğin şekli kusursuz olabilir ve model yine
de aynı düğümde takılabilir. Bu, kodu derlemekle çalıştırmak arasındaki fark gibi:
derleyici tip hatalarını yakalar, sonsuz döngüleri yakalamaz.

**Neden karmaşık sayılıyor:** algoritması aslında ders kitabı seviyesinde (graf üzerinde
çevrim arama). Zor olan ön koşul — ajan akışınızın **önceden bilinen bir graf** olması
gerekiyor. Serbest ReAct döngüsünde uygulanamaz.

---

# SEVİYE 5 · Kademe

> **Ortak fikir:** Tek sinyal yetmez, tek tepki de yetmez. Birden çok sinyali topla,
> müdahaleyi basamak basamak sertleştir.

Buraya kadarki her zihniyetin bir sinyali ve bir tepkisi vardı: tetiklendi → durdu. Bu
seviye ikisini de çoğaltıyor — birkaç sinyal aynı anda dinleniyor ve tepki bir merdiven
oluyor: önce uyar, sonra engelle, sonra geri sar, sonra kes.

Karmaşıklık burada patlıyor: **durum makinesi artık dedektörün içinde.**

## 14 · Kural koy, ihlal edildiğini söyle, geri sar

*(`telemetry-repair` — Seviye 5)*

Dokuz numaralı doğrulama kapısının daha somut hâli, ve arkasında ölçüm var.

Üç deterministik kontrol tanımlıyor: ajanın söylediği toplam gerçekten gördüğü verilerden
çıkıyor mu · gereken bütün araç çağrıları yapıldı mı · aracın döndürdüğü şey o aracın
üretebileceği bir biçimde mi.

Bunlar istatistik değil, kural. Eşik ayarı gerektirmiyorlar, kalibrasyon gerektirmiyorlar,
ikinci bir model gerektirmiyorlar. Ya sağlar ya sağlamaz.

Ölçülmüş sonuç çarpıcı: bu üç basit kural, aynı hataları istatistiksel bir anomali
dedektöründen **daha iyi** yakalıyor — ve sıfır yanlış pozitifle. Karmaşık olanın her
zaman daha iyi olmadığının somut örneği.

Ama asıl ilginç bulgu onarım tarafında. Bir kontrol düştüğünde ajanı son sağlam noktaya
geri sarıp yeniden çalıştırıyorlar. Soru şu: ona ne söylemeli?

Denenen seçenekler ve kurtarma oranları:
- Hiçbir şey söyleme, sadece yeniden dene → %16
- Genel bir "tekrar kontrol et" uyarısı → %36
- **Doğru cevabı doğrudan ver** → %36
- **Hangi kontrolün düştüğünü söyle** → **%45**

Yani ajana cevabı vermek, hangi kuralı çiğnediğini söylemekten daha kötü. Bu sezgiye
aykırı ama mantıklı: cevabı verdiğinizde ajan onu kopyalar, neden yanıldığını anlamaz.
Kuralı söylediğinizde kendi hatasını bulur.

Bunun bizim tasarımımıza doğrudan sonucu var: bir dedektör tetiklendiğinde ajana genel bir
öğüt vermek yerine **tetiklenen dedektörün adını** söylemek gerekiyor.

Ve bir sınır: aynı çalışma döngüde onarımın işe yaramadığını ölçmüş. Hedef sapmasında beş
vakanın dördü kurtarılıyor, ama **döngüde yeniden çalıştırma çoğu zaman aynı döngüyü
üretiyor.** Döngüde doğru hamle onarmak değil, durmak.

**Ek karmaşıklık:** kontrol noktası tutmak. Koşumu geri sarabilmek için durumu
serileştirilebilir tutmanız gerekiyor — bu, ajan mimarisine ciddi bir yük.

## 15 · Altı ucuz sinyal, kademeli müdahale

*(`pi-signature` — Seviye 5)*

Bu belgedeki en çok parçalı dedektör, ve neredeyse tamamı Seviye 2'nin çözemediği bir
boşluğu kapatmak için var: **ajan aynı şeyi farklı kelimelerle söylüyor.**

Ajan `ls -la` yazıyor, sonra `ls -al`, sonra `ls -a -l`. Üçü de aynı şey. Ama parmak izleri
farklı, dolayısıyla hiçbir imza dedektörü tetiklenmiyor.

Metin tarafında bu daha da yaygın: ajan aynı cümleyi kurmuyor ama aynı fikri yeniden ifade
ediyor. "Bir daha kontrol edeyim" → "Tekrar doğrulamam lazım" → "Bunu bir kez daha teyit
edeyim."

Getirdiği çözüm zekice ve ucuz: ardışık iki mesajın **kelime örtüşme oranına** bakıyor.
%55'ten fazla ortak kelime taşıyorlarsa, model aynı adımı yeniden ifade ediyor demektir.

Bunun güzelliği maliyetinde. Anlamsal benzerlik denince akla gömme modelleri gelir — her
adımda bir model çalıştırmak, gecikme, para. Burada yapılan iş sadece kelime saymak.
Deterministik, anında, bedava.

Ama bu tek başına bir sinyal. Yanına beş tane daha koyuyor — birebir aynı çağrı, aynı
aracın ardışık hatası, birebir aynı metin, tek mesaj içinde tekrarlanan cümle, ve yakın
benzerin *çevrim* hâli. Altısı ayrı ayrı sayılıyor, ayrı eşikleri var.

Ve tepki tarafı da kademeli: metin döngülerinde önce yönlendir, sonra kes; araç
çağrılarında önce engelle, sonra sebep söyle, tekrarlarsa turu bitir.

**Neden bu seviyede:** altı sinyal × ayrı eşikler × iki farklı müdahale merdiveni ×
engellenen çağrının pencereye girmemesi × her kullanıcı girdisinde sayaçların sıfırlanması.
Doğru uygulamak için hepsini akılda tutmanız gerekiyor. Yanlış uygularsanız sessizce
çalışmayan bir dedektör elde edersiniz — Seviye 2'deki o en kötü hata türü.

---

# SEVİYE 6 · Karar

> **Ortak fikir:** "Ne zaman dur" yanlış soru. Doğru sorular: bu adımda **ne yapmalı**, ve
> bu eşik **nereden geldi**?

Son iki zihniyet, önceki on beşinin sorduğu soruyu bırakıyor. Biri kararın kendisine
karışıyor, diğeri koşumun tamamen dışına çıkıyor. Bu yüzden en sonda.

## 16 · Bütçe bir tavan değil, bir bütçe

*(`voi-allocation` — Seviye 6)*

Bu belgedeki en farklı düşünen yaklaşım, ve adını hak ediyor.

Diğer bütün stratejiler şunu soruyor: **"ne zaman dur?"** Bu strateji şunu soruyor:
**"parayı nereye harca?"**

Fark şurada. Bir tavan koyduğunuzda ajan tavana kadar istediğini yapar, sonra kesilir.
Tavana nasıl geldiği umurunuzda değildir. Oysa aynı bütçeyle çok daha iyi bir sonuç
alınabilirdi — eğer parayı doğru yerlere harcasaydı.

Bu yüzden her adımda ajanın seçenekleri **birim bütçe başına faydaya** göre puanlanıyor.
Ham faydaya göre değil. Pahalı bir eylem biraz daha faydalı olabilir; ucuz bir eylem
neredeyse bedavadır. Oranı yüksek olan kazanır.

İki bütçe birden takip ediliyor: araç çağrısı sayısı ve token. Biri kritik seviyeye
inince keşif eylemleri pahalılaşıyor, bitirme çekici hale geliyor.

Ve bir koruma katmanı var, çünkü saf maliyet optimizasyonu tehlikeli: bütçe azalınca
"cevap ver" eylemi en ucuz seçenek olduğu için aşırı çekici hale gelir. Ajan aceleyle
kötü bir cevaba kaçar. Bu yüzden üstte deterministik kurallar var — kanıt zayıfken erken
cevap engelleniyor.

Ölçülmüş sonucu da dürüst: bu yaklaşımın kazancı **bütçe kıtken** büyük, bol kaynakta
eriyor.

**Neden en karmaşıklardan:** diğer on altısı döngünün *kenarında* durup izliyor. Bu,
döngünün *içine* girip her adımda hangi eylemin seçileceğine karışıyor. Yanlış ayarlanırsa
ajanı bozar — diğerleri en fazla erken durdurur.

## 17 · Eşiği tahmin etme, ölç

*(`improvement-loop` — Seviye 6)*

Bütün diğer zihniyetler koşumun içinde müdahale ediyor. Bu hiç etmiyor — sadece kaydediyor.
Çünkü sorduğu soru farklı: **bu eşikleri kim, neye bakarak koydu?**

"Adım limiti 12" diye yazdığınızda o 12 nereden geldi? Çoğu zaman hiçbir yerden. Birinin
makul bulduğu bir sayı.

Ve yanlış seçilmiş bir limit sinsi bir hasar verir. Arize'ın uyarısı: *"Meşru işi kesecek
kadar dar bir adım limiti, modelin kötüleştiği gibi görünen sessiz bir kalite gerilemesine
dönüşür."* Model aynıdır, limitiniz dardır, ama grafikte model kötüleşmiş görünür.

Çözüm ölçmek: başarılı koşumların adım dağılımına bak, tavanı kuyruğunun üstüne koy. Ve
sonra izle — koşumların yüzde kaçı limitte sonlanıyor? Bu oran tırmanıyorsa görev zorluğu
ya da araç güvenilirliği değişmiştir.

Üç ayrı kaynak bağımsız olarak aynı şeyi söylüyor, biri p99'dan başlamayı öneriyor. Ve
disiplin tarafı: döngü ayarları prompt'la birlikte sürümlenmeli ve bir terfi kapısından
geçmeli. **Eşik değiştirmek, kod değiştirmekle aynı süreçten geçmeli.**

**Neden en sonda:** kod olarak belki en basiti — sadece iz yazıyor, hiçbir şeye karışmıyor.
Ama *anlaması* en son gelen. Çünkü ancak diğer on altısını uyguladıktan sonra "bu sayıları
nereden buldum ben?" diye sormaya başlıyorsunuz. Merdivenin en üst basamağı, en sarp olduğu
için değil, oraya varmadan görünmediği için.

---

# Ve bir de bunların hepsini reddeden bir zihniyet

## Modeli döngüden çıkar

*(OpenAdapt — henüz uygulanmadı, seviye dışı)*

Buraya kadar okuduğunuz on yedi zihniyetin hepsi aynı varsayımı paylaşıyor: model döngünün
içinde, biz onu sınırlıyoruz.

OpenAdapt bu varsayımı reddediyor. Yaklaşımı şu: bir görevi bir kez gösterin, onu
**deterministik bir programa derlesin**. Sonraki koşumlarda model hiç çağrılmıyor.

> *"Sağlıklı koşumlar hiçbir üretken model çağrısı yapmaz."*

Model yalnızca ekran beklenenden saptığında devreye giriyor. Ve sonucu bildirmeden önce
canlı duruma karşı doğruluyor; kanıt yoksa koşum incelemeye duruyor.

Döngüye girme riskinin kökten çözümü bu: **döngüde model yoksa döngü de yok.**

Bedeli esneklik. Yalnızca tekrarlanan, önceden gösterilebilen işler için çalışıyor. Ama
o işler için tartışmasız en güvenilir yaklaşım.

Seviye merdivenine koyulmadı çünkü aynı merdivende değil — bu, merdiveni tırmanmak yerine
binadan çıkmak.

---

# Toparlarsak

## Zorluk merdiveni

| Seviye | Ne tutuyor | Ne söyleyebiliyor | Ne söyleyemiyor |
|---|---|---|---|
| **1 · Sayaç** | Bir tamsayı | "Çok oldu" | Ne olduğunu |
| **2 · Pencere** | Son N adım | "Aynı şey oluyor" | İşe yarayıp yaramadığını |
| **3 · Dünya** | Dış kanıt, oran, para | "Sonuç yanlış" / "Araç bozuk" | Ajan hiç bitirmezse sessiz |
| **4 · Şekil** | Akış tanımı | "Bu döngü mümkün değil" | Model içi takılmaları |
| **5 · Kademe** | Çok sinyal + kontrol noktası | "Şu kural düştü, geri sarıyorum" | Doğru ayarlanmazsa sessizce ölür |
| **6 · Karar** | Fayda modeli / dağılım | "Şunu yap" / "Eşiğin şu olmalı" | Koşumu tek başına korumaz |

Her seviye bir öncekinin kör noktasını kapatıyor — ama yerine geçmiyor. Seviye 3
uyguladınız diye Seviye 1'i atamazsınız: doğrulama kapısı, ajan hiç "bitirdim" demezse hiç
açılmaz. Sayaç ise her koşumda konuşur.

**Pratik öneri:** 1 → 6 → 2 sırasıyla ilerleyin. Önce sayacı koyun (bir öğleden sonra),
sonra izleri toplayıp eşikleri ölçün (17 numara), sonra pencere dedektörlerini ekleyin.
Eşiği ölçmeden pencere yazmak, hangi sayıyı yazacağınızı bilmeden yazmak demek.

## Çapraz referans: inanca göre altı aile

Zihniyetleri zorluğa göre değil **teşhislerine** göre de gruplayabilirsiniz. Bu belgenin
önceki sürümü bu sıradaydı:

| Aile | Teşhisi | Çözümü | Zihniyetler |
|---|---|---|---|
| Tekrar | Sıkışan ajan kendini tekrar eder | Tekrarı gör | 5 · 6 · 7 · 8 · 15 |
| Bütçe | Ne yaptığı değil, ne harcadığı önemli | Say ve kes | 1 · 11 · 16 |
| Doğrulama | Ajan yanlış işi doğru sanıyor | Dünyaya sor | 9 · 10 · 14 |
| Şekil | Döngü oluşmasına izin veriyoruz | İzin verme | 7 · 12 · 13 |
| İkna | Kesmek kabadır, iş çöpe gider | Modele söyle | 2 · 3 · 4 |
| Meta | Eşikler tahmin edilmiş | Ölçerek koy | 17 |

Bazı zihniyetler iki ailede birden görünüyor — 7 hem tekrar hem şekil, 15 hem tekrar hem
kademe. Bu bir hata değil; aile sınırlarının zorluk sınırlarından daha bulanık olduğunu
gösteriyor. Uygulama sırası için zorluk merdiveni, "neden böyle düşünüyorlar" sorusu için
aile tablosu.

## Ve hepsinin ortak dersi

Hiçbiri diğerinin yerine geçmiyor. Tekrar dedektörü ucuz döngüleri yakalar ama pahalı tek
seferlik işleri kaçırır. Bütçe her şeyi yakalar ama neyin yanlış gittiğini söylemez.
Doğrulama yanlış işi yakalar ama ajan hiç bitirmezse sessiz kalır.

Ve bir de şu var, PoC'de bizzat ölçtüğümüz: **iki mekanizma birbirini maskeleyebilir.**
Bütçe sınırı çok darsa döngü dedektörü hiç konuşamaz. Koşum doğru durur ama neden
durduğunu öğrenemezsiniz. Bütçe bir şeyin yanlış gittiğini söyler; döngü tespiti neyin
yanlış gittiğini.

Ve hepsinin dışında kalan bir sebep var: **döngünün kaynağı prompt olabilir.** Aşırı
tedbir teşvik eden bir talimat ajanı "bir kez daha doğrula"ya sürüklüyor ve ajan tam da
söylendiği gibi davranıyor. Buradaki on yedi mekanizmanın hiçbiri bunu çözmüyor — hepsi
ancak sonucu kesiyor. Guardrail koymak, prompt'u gözden geçirmenin yerine geçmiyor.

İşte bu yüzden hepsini ayrı ayrı uygulayıp aynı görevde yan yana koşturuyoruz.

Ve tek bir kural kalacaksa, Modexa'nınki kalsın:

> *"Her ajan döngüsünün **tasarlanmış bir çıkışı** olmalıdır — başarı, geri dönüş, sorma
> veya devretme. 'Sonsuza kadar dene' değil. 'Umarım model çözer' değil."*

---
---

# EK · On Yedi Zihniyet, Tek Tek

Yukarıdaki bölüm zihniyetleri **anlatarak** ilerledi — her biri bir öncekinin çözemediği
şeyi çözerek. Bu bölüm aynı on yediyi **aynı sırada** ama tek tek ele alıyor. Her madde
kendi başına okunabilir; bir stratejiyi seçerken doğrudan buraya bakabilirsiniz.

Numaralar iki bölümde aynı: yukarıda 9 numara neyse, burada da 9 numara o.

Her madde şu düzende: *kaynağı · zorluk seviyesi · tek cümlede ne olduğu · neye inandığı ·
nasıl çalıştığı · en güçlü yanı · en zayıf yanı*, ardından **teknik karşılığı** — tuttuğu
durum, algoritması, kancası, eşikleri ve maliyeti.

> **Listede olmayan bir kaynak:** *When Agents Do Not Stop: Uncovering Infinite Agentic
> Loops in LLM Agents* (IAL-SCAN) bilerek dışarıda. Sebebi şu: o çalışma **statik analiz**
> yapıyor — kodu çalıştırmadan, dağıtımdan önce okuyarak döngü arıyor. Bizim ihtiyacımız
> çalışma anında devreye giren bir kontrol. İkisi CI ile üretim gibi; biri diğerinin yerine
> geçmiyor ama bu listeye de girmiyor.
>
> Ondan aldığımız tek şey **soru biçimi**: "döngü var mı" değil, *"ajanik bir geri besleme
> yolu, maliyetli bir işlemi etkili bir sınır olmadan tekrar tekrar çalıştırabilir mi?"*
> Bu çerçeve belgenin girişinde duruyor.

---

## 1 · `arize-control` — Modele sorma, say

**Kaynak:** Arize — *What Is An Agent Control Loop?*

**Zorluk:** Seviye 1 (Sayaç)

**Tek cümlede:** Durma kararını modelin yargısından tamamen koparır.

**Neye inanır.** Model, işin bitip bitmediği konusunda yanılması en muhtemel bileşendir.
O yüzden durma kararı modelin söylediği hiçbir şeye bağlı olmamalı.

**Nasıl çalışır.** Beş durma koşulu tanımlar — görev tamamlandı, adım limiti, bütçe, süre,
hata politikası — ve aralarında hiyerarşi kurar. **Adım limiti birincildir**, çünkü tek
tartışmasız ölçüdür: on iki adım attıysa on iki adım atmıştır.

İkinci ve az fark edilen katkısı: her koşumun sonunda **neden durduğunu kaydeder.**
"Tamamlandı", "adım limiti", "bütçe aşıldı" ve "hata" farklı sonuçlardır.

**En güçlü yanı.** Her şeyi yakalar, çünkü hiçbir şeye bakmaz — sadece sayar. Kandırılamaz.

**En zayıf yanı.** Size *ne* olduğunu söylemez. "Çok fazla adım attı" der, "aynı çağrıyı
tekrarlıyordu" demez. Bir tavandır, teşhis değil.


### Teknik karşılığı

**Tuttuğu durum.** Yalnızca sayaçlar: `steps`, `tokens`, `cost_usd`, `elapsed`,
`consecutive_errors`. Olay geçmişi tutmuyor — bu yüzden en ucuz strateji.

**Koşul.**

```python
for eksen, kullanılan, limit in axes:
    if limit is not None and kullanılan >= limit:
        return Verdict(STOP, f"budget_{eksen}")
```

**Kanca.** Tek kanca: `before_step`. Adımdan **önce** bakılıyor — sonra bakmak, limiti aşan
çağrının parasını zaten ödedikten sonra fark etmek demek.

**Durma sebebi kaydı.** `on_stop` her koşuma bir terminal etiket yazıyor:
`completed | max_steps | budget_exceeded | timeout | error`. Bu etiket iz dosyasına ve
özet satırına giriyor; ortalama alınarak kaybolmaması için ayrı alanda tutuluyor.

**Maliyet.** O(1). Beş karşılaştırma.

---

## 2 · `agentscope-grace` — Nazikçe isteme, seçeneksiz bırak

**Kaynak:** **AgentScope** — `ReplyFinishedReason.EXCEED_MAX_ITERS`

**Zorluk:** Seviye 1 (Sayaç)

**Tek cümlede:** Limit dolduğunda ajana ek tur verir ama araç kullanmasını kilitler.

**Neye inanır.** Yarım kalmış işi çöpe atmak israftır. Ama modelden nazikçe bitirmesini
istemek de yetmez — bitirmekten başka seçeneği kalmamalı.

**Nasıl çalışır.** Limit dolduğunda koşum bitmez; ajana beş turluk bir **lütuf bütçesi**
verilir. **Ama o turlarda araç seçimi kilitlenir** — ajan yeni araç çağıramaz, yalnızca
cevap üretebilir. Nazik bir rica değil, mekanik bir kısıt.

**En güçlü yanı.** Yapılan işin çöpe gitmesini engeller ve bunu ricaya bırakmaz.

**En zayıf yanı.** Bir tespit değil, bir iniş mekanizmasıdır. Döngünün kendisini yakalamaz.


### Teknik karşılığı

**Tuttuğu durum.** `grace_left: int` — normalde 0.

**Limit dolduğunda koşum bitmiyor, mod değişiyor:**

```python
def before_step(self, ctx):
    if ctx.budget.exceeded() and self.grace_left == 0:
        self.grace_left = 5                     # lutuf butcesi
        return CONTINUE                          # DURDURMA
    if self.grace_left > 0:
        self.grace_left -= 1
        if self.grace_left == 0:
            return Verdict(DEGRADE, "grace_exhausted")
    return CONTINUE
```

**Araç kilidi — asıl mekanizma:**

```python
def decorate_request(self, req, ctx):
    if self.grace_left > 0:
        req.forced_finish = True        # "arac cagirma, cevap ver"
        # gercek API'de: tool_choice = "none"
    return req
```

Fark burada: nazik bir rica değil. `forced_finish` bayrağı modele araç çağırma seçeneğini
**bırakmıyor**. Ajanın elinde yalnızca metin üretmek kalıyor.

**Terminal durum.** Lütuf bütçesi de biterse `Status.DEGRADED` — `OK` değil. Tükenmiş bir
koşum asla temiz bitmiş sayılmıyor.

**Kanca.** `before_step` + `decorate_request`.

---

## 3 · `hermes-no-pressure` — Uyarma, sadece dur

**Kaynak:** **Hermes** — `agent/agent_init.py:986-991` (kaynak koddan)

**Zorluk:** Seviye 1 (Sayaç)

**Tek cümlede:** Ara uyarıları **bilerek kaldırır**, çünkü uyarının kendisi zarar veriyor.

**Neye inanır.** Bu, diğer bütün stratejilerin tersini söyleyen tek zihniyet — ve ölçüme
dayanıyor. Kod yorumu şöyle: *"Ara basınç uyarıları yok — modelleri karmaşık görevlerde
erken pes ettiriyordu."*

Yani "adımların azalıyor" mesajı modele "bu görev bana göre değil" gibi geliyor ve
gerçekten yapabileceği işi bırakıyor.

**Nasıl çalışır.** Adım bütçesi boyunca hiçbir uyarı vermez. Bütçe gerçekten dolduğunda
tek bir mesaj girer, ajana bir **lütuf çağrısı** verilir, ve o çağrıda da metin üretmezse
zorla özet istenir.

Ama aynı sistem **süre** bütçesinde %80'de uyarı verir. Eksene göre farklı karar
vermişlerdir.

**En güçlü yanı.** Uyarının kendi yarattığı zararı ortadan kaldırır.

**En zayıf yanı.** Uyarının kurtarabileceği koşumları kaybeder. Bu bir dengedir ve Hermes
bir tarafı seçmiştir.


### Teknik karşılığı

**İki mandal.**

```python
_budget_exhausted_injected: bool = False
_grace_call: bool = False
```

**Adım ekseni — hiç uyarı yok.** `decorate_request` adım bütçesi hakkında **hiçbir şey**
yazmıyor. Model kaç adımı kaldığını bilmiyor.

**Tükenme akışı:**

```python
if steps >= max_steps and not _budget_exhausted_injected:
    _budget_exhausted_injected = True
    req.notes.append("Iterasyon butcesi doldu. Elindekiyle nihai cevabi ver.")
    _grace_call = True
    return CONTINUE                      # BIR cagri daha hakki

if _grace_call and son_cikti_metin_degil:
    req.forced_finish = True             # zorla ozet
```

Yani: tek mesaj → bir lütuf çağrısı → metin gelmezse zorla özet. Üç kademe.

**Süre ekseni — ayrı politika.**

```python
_run_budget_wrapup_injected: bool = False   # tur basina sifirlaniyor
if elapsed >= 0.80 * max_seconds and not _wrapup:
    _wrapup = True
    req.notes.append("Sure butcenin %80'i doldu, toparlamaya basla.")
```

Aynı sistem içinde iki eksene iki farklı karar. Bu asimetri stratejinin özü; kodda da
ayrı sabitlerle duruyor, yanlışlıkla birleştirilmesin diye.

**Kanca.** `before_step` (tükenme tespiti) + `decorate_request` (enjeksiyon).

---

## 4 · `claude-advisory` — Modele saatini göster

**Kaynak:** **Claude Task Budgets** — resmî API dokümanı, beta `task-budgets-2026-03-13`

**Zorluk:** Seviye 1 (Sayaç)

**Tek cümlede:** Ajanı kesmez; ne kadar kaldığını söyleyip kendini ayarlamasına izin verir.

**Neye inanır.** Dışarıdan kesmek kaba bir çözümdür ve yapılan işi çöpe atar. Model
durumunu bilirse kendisi toparlanabilir.

**Nasıl çalışır.** Konuşmaya bir geri sayım işareti girer; model bunu görür ve bütçe
azaldıkça işi toparlar. Kritik nokta: bu **tavsiye, zorlama değil.** Model, kesilmesi
bitirilmesinden daha zararlı olacak bir işin ortasındaysa bütçeyi aşabilir. Sert tavan
ayrı bir mekanizmadır.

Uyarıyı eksene göre verir: süre azalıyorsa uyarır, adım azalıyorsa uyarmaz — bu ayrımın
gerekçesi 15. maddede.

**En güçlü yanı.** Öngörülebilir bir iniş sağlar; ajan işin ortasında kesilmez.

**En zayıf yanı.** Model dinlemezse hiçbir şey yakalamaz. Tek başına kullanılamaz.


### Teknik karşılığı

**Tuttuğu durum.** Eksen başına tek seferlik uyarı mandalı (`_warned: set[str]`). Tespit
durumu yok — bu bir dedektör değil.

**Geri sayım enjeksiyonu.** Tek iş `decorate_request`'te:

```python
kalan = limit - kullanılan
req.notes.append(f"Kalan bütçe: {kalan}/{limit} {eksen}.")
```

**Eksen bazlı uyarı politikası** — 13. maddedeki bulgunun kodu:

```python
WARN_AT = {
  "seconds":  0.80,   # süre ekseninde uyar
  "cost_usd": 0.80,
  "steps":    None,   # adım ekseninde UYARMA
  "replans":  None,
}
```

Adım ekseninde uyarı vermemek keyfi değil: ölçülmüş olarak modeli erken pes ettiriyor.

**Zorlayıcı değil.** Bu strateji **hiçbir zaman** `STOP` dönmüyor. Yalnızca `notes` yazıyor.
Sert tavanı ayrı bir strateji sağlamalı — tek başına kullanılırsa koşum sınırsız kalır.

**Prompt cache uyarısı.** Geri sayım her turda değiştiği için prompt önekini bozuyor. Cache
kullanılıyorsa geri sayımı prompt'un **sonuna** koymak gerekiyor, başına değil.

---

## 5 · `strands-entropy` — Tekrarı sayma, çeşitliliği ölç

**Kaynak:** **Strands** (AWS) — harness taramasından

**Zorluk:** Seviye 2 (Pencere)

**Tek cümlede:** "Aynı şey kaç kez oldu" yerine "kaç farklı şey oldu" diye sorar.

**Neye inanır.** Tekrarı tanımlamak zordur — ardışık mı, dönüşümlü mü, üçlü çevrim mi?
Çeşitliliği ölçmek kolaydır ve hepsini birden kapsar.

**Nasıl çalışır.** Son N adıma bakıp kaç ayrı eylem çalıştığını sayar. Sayı düşükse ajan
dar bir alanda dönüyordur — ister aynı şeyi tekrarlasın, ister iki şey arasında gidip
gelsin, ister üç adımlık bir çevrimde dönsün. Tek kural, bütün desenler.

**En güçlü yanı.** Her uzunluktaki çevrimi tek kuralla yakalar; k=2, k=3 diye ayrı tarama
yazmaya gerek kalmaz.

**En zayıf yanı.** Meşru olarak dar alanda çalışan işleri de yakalar. Bir dosyayı
defalarca düzenleyip test eden ajan ilerliyordur ama düşük çeşitlilik gösterir.


### Teknik karşılığı

**Tuttuğu durum.** Tek bir kuyruk: son N eylem kimliği.

```python
recent: deque[str] = deque(maxlen=N)     # N = 10
```

**İki ölçüm seçeneği.**

*Basit — farklı eleman sayısı:*

```python
if len(recent) == N and len(set(recent)) <= k:      # k = 2
    return Verdict(STOP, "low_diversity", {"farkli": len(set(recent))})
```

*Bilgi kuramsal — Shannon entropisi:*

```python
p = [c / N for c in Counter(recent).values()]
H = -sum(pi * log2(pi) for pi in p)
H_max = log2(N)
if H / H_max < 0.35: → tetikle
```

Normalize entropi eşikten küçükse ajan dar bir alanda dönüyordur. İkinci biçim daha
yumuşak — "üç eylem ama biri baskın" durumunu da yakalıyor.

**Neden k-taraması gerekmiyor.** A-A-A-A (k=1), A-B-A-B (k=2), A-B-C-A-B-C (k=3) — üçü de
düşük çeşitlilik üretiyor. Tek bir eşik hepsini kapsıyor; `openhands-stuck`'ın senaryo
senaryo yazdığı şeyi tek satırda yapıyor.

**Yanlış pozitif ayarı.** Meşru dar-alan işleri (düzenle→test→düzenle→test) için iki
gevşetme var: `mutates_screen` olmayan eylemleri saymamak, ve ilerleme sinyali gelen
adımlarda kuyruğu sıfırlamak.

**Maliyet.** O(N) — pratikte sabit.

---

## 6 · `openhands-stuck` — Beş desen, ve "sıkışmak" ayrı bir sonuç

**Kaynak:** **OpenHands** — `openhands-sdk/.../conversation/stuck_detector.py` (kaynak koddan)

**Zorluk:** Seviye 2 (Pencere)

**Tek cümlede:** Döngüyü tek desen olarak değil beş ayrı senaryo olarak arar, ve
yakaladığında ayrı bir terminal durum üretir.

**Neye inanır.** Döngü tek bir şey değildir. Ve ajanın sıkışması bir hata değil, ayrı bir
sonuçtur — "çöktü" ile "sıkıştı" farklı şeylerdir ve farklı raporlanmalıdır.

**Nasıl çalışır.** Beş senaryoyu birden tarar: aynı eylem aynı sonucu veriyor · aynı eylem
sürekli hata veriyor · ajan kendi kendine konuşuyor · iki eylem arasında gidip geliyor ·
bağlam penceresi hatası döngüsü. Eşikler senaryoya göre farklıdır.

Kritik ayrıntı: iki olayı karşılaştırırken **her turda zaten değişen alanları görmezden
gelir.** Bu yapılmazsa iki özdeş çağrı hiçbir zaman eşit çıkmaz ve dedektör sessizce
hiçbir şey bulmaz.

Kademelendirme yoktur: sıkıştıysan sıkışmışsındır, doğrudan durur.

**En güçlü yanı.** Dönüşümlü döngüler dahil, tekrar eden her deseni yakalar.

**En zayıf yanı.** Ajan her turda farklı bir şey deniyor ama hiçbiri işe yaramıyorsa
sessiz kalır.


### Teknik karşılığı

**Pencere.** Son kullanıcı mesajından itibaren, en fazla 20 olay. Kullanıcı araya girdiyse
önceki tekrar bu turun döngüsü sayılmıyor:

```python
evs = events[last_index_of(USER)+1:][-20:]
```

**Beş yüklem ve eşikleri** (kaynak koddan):

| Senaryo | Eşik | Koşul |
|---|---:|---|
| aynı eylem + aynı gözlem | 4 | son 4 (eylem,gözlem) çiftinin imzaları tek |
| aynı eylem + hata | 3 | son 3 çiftte eylem aynı **ve** hepsi hata |
| monolog | 3 | eylemsiz ardışık 3 asistan mesajı |
| dönüşümlü desen | 6 | son 6 olayda A-B-A-B örüntüsü |
| context-window döngüsü | — | (bu PoC'de karşılığı yok) |

İkinci senaryoda **gözlem imzası karşılaştırılmıyor** — hata metni her denemede biraz
farklı olabilir (satır numarası, süre). Eylem aynıysa ve sonuç hep hataysa döngüdür.

**İçerik tabanlı eşitlik.** Karşılaştırma öncesi oynak alanlar atılıyor:

```python
VOLATILE = {"tool_call_id","action_id","llm_response_id","request_id",
            "timestamp","trace_id","span_id","seed","nonce"}
```

Bu yapılmazsa iki özdeş çağrı hiçbir zaman eşit çıkmaz. Ama fazla atılırsa `lint(a.md)` ile
`lint(b.md)` de eşit olur ve meşru toplu işlem kesilir.

**Kademelendirme yok.** `on_observation` doğrudan `Verdict(STOP, ...)` dönüyor ve döngü
bunu `Status.STUCK` terminal durumuna çeviriyor — `ERROR` değil.

**Maliyet.** O(20). Sabit.

---

## 7 · `openclaw-pingpong` — Adlandırılmış dedektörler ve sıkıştırma tuzağı

**Kaynak:** **OpenClaw** — `tools.loopDetection` yapılandırma şeması (kurulu paketten)

**Zorluk:** Seviye 2 (Pencere)

**Tek cümlede:** Dedektörleri isimlendirip ayrı ayrı yönetir, ve bağlam sıkıştırmasının
kendisini bir döngü kaynağı sayar.

**Neye inanır.** Bir güvenilirlik mekanizması eklerken, o mekanizmanın kendisinin yeni bir
hata kaynağı olup olmadığını sormak gerekir.

**Nasıl çalışır.** Üç adlandırılmış dedektör: aynı araç aynı parametrelerle · yoklama
araçlarında ilerleme yokluğu · **ping-pong** (iki eylem arasında gidip gelme). Üç kademeli
eşik: uyarı, kritik, küresel ilerleme-yok kesicisi.

En özgün parçası **sıkıştırma sonrası koruması**: bağlam sıkıştırıldıktan sonra üç deneme
boyunca ayrı bir koruma kurulu kalır. Sebebi somut — sıkıştır, bağlam yine dolsun, tekrar
sıkıştır döngüsü belgelenmiş ve gerçek para yakmış bir desendir.

Parmak izini araç adı, argümanlar **ve sonuç** ile birlikte alır: aynı çağrı farklı sonuç
veriyorsa döngü sayılmaz.

**En güçlü yanı.** Kendi çözümünün yarattığı sorunu gören tek yaklaşım.

**En zayıf yanı.** Kutudan **kapalı** gelir. Açılmazsa hiçbir şey yakalamaz.


### Teknik karşılığı

**Parmak izi — sonuç dahil:**

```python
fp = sha256(tool + "|" + args_hash + "|" + result_hash)
```

Sonucun dahil olması kritik: aynı çağrı **farklı** sonuç veriyorsa döngü sayılmıyor.
Bizim mevcut altyapımız eylem ve gözlemi ayrı hash'liyor; bu strateji ikisini birleştiriyor.

**Üç dedektör** (pencere 30):

```python
genericRepeat:       count(fp, window) >= esik
knownPollNoProgress: tool in POLL_TOOLS and result_hash sabit
pingPong:            son 2k olayda [A,B] * k örüntüsü, k = 1..5
```

**Üç kademeli eşik — şiddet değil kapsam artıyor:**

| Kademe | Eşik | Davranış |
|---|---:|---|
| uyarı | 10 | prompt'a not |
| kritik | 20 | o araç engellenir |
| küresel kesici | 30 | ilerleme yoksa koşum durur |

**Compaction sonrası koruma:**

```python
def on_compaction(self):
    self._guard_armed = 3          # sonraki 3 denemede siki mod

def on_observation(self, ev, ctx):
    if self._guard_armed > 0:
        self._guard_armed -= 1
        esik = self.warning_threshold // 3      # cok daha dar
```

Sebebi somut: sıkıştır → bağlam yine dolsun → tekrar sıkıştır döngüsü belgelenmiş ve
gerçek para yakmış bir desen.

**Varsayılan.** Üretimde `enabled=False` geliyor. Bizim uygulamamızda **açık** olacak;
kapalı varsayılan bulgusu dokümanda ayrıca raporlanacak.

---

## 8 · `loopguard-dignity` — Onurunla dur

**Kaynak:** *Ne Zaman Durması Gerektiğini Bilen Ajanlar*

**Zorluk:** Seviye 2 (Pencere)

**Tek cümlede:** Bütçeyi bir ürün kısıtı sayar ve durmanın *nasıl* olduğunu önemser.

**Neye inanır.** Durmak başarısızlık değildir; kötü durmak başarısızlıktır. Ajan
durduğunda kullanıcının elinde bir şey kalmalı.

**Nasıl çalışır.** Sınırları **araç bazında** tutar — bozuk bir araç genel deneme hakkını
tek başına yiyemesin. İlerlemeyi dünyanın durum özetinin değişip değişmediğinden anlar.
Bilgi eksikse **kullanıcıya sorar** ve bunu meşru bir sonuç sayar, yenilgi değil.

En ayırt edici yanı durma şekli: dört şey söyler — ne denedi, ne buldu, neden durdu,
sırada ne var.

**En güçlü yanı.** Sonsuz bir döngüyü faydalı bir kısmi sonuca çevirir. Kullanıcı boş
elle kalmaz.

**En zayıf yanı.** Hızlı ve ucuz döngüler sınırlar dolmadan çok tur dönebilir.

**Kaynağın niteliği.** Kavramsal yazı + **çalışan referans kod** — ölçüm yok. Tam bir
`LoopGuard` sınıfı veriyor: `check_budget()` · `record_progress()` · `record_tool_call()` ·
`should_retry()` · `detect_repeat_pattern()`. Yani iddiaları sayıyla değil, **koşturulabilir
kodla** destekliyor. Bu listede az rastlanan bir kanıt biçimi; eşiklerinin nereden geldiğini
söylemediği için de en zayıf yanı bu.

**Beş koruma.** Brief'in kalemleriyle birebir örtüşüyor:
kesin sınırlar (adım + araç + **araç başına retry** + token/maliyet) · **üstel gecikme +
jitter** · ilerleme kontrolleri · döngü parmak izleri · **çekimser kalma yolu**.

**Örnek politika.** Kaynak somut sayı veriyor — bu listede sayı veren az kaynaktan biri:
adım ≤ 12 · araç çağrısı ≤ 8 · **araç başına deneme ≤ 2** · süre ≤ 60 sn.

**İki cümlesi.** Sunum kapanışı için:

> *"Ajanlar sadece yanıldıkları için başarısız olmazlar. **Israrcı oldukları için** başarısız
> olurlar. Ve ısrar pahalıdır."*
>
> *"Durmak başarısızlık değildir. Durmak kontroldür."*


### Teknik karşılığı

**Tuttuğu durum.** `retries: dict[araç, int]` · son 3 `state_hash` · denenen yaklaşımların
kısa listesi (rapor için).

**İki koşul.**

```python
# araç bazlı deneme
if retries[tool] >= max_retries_per_tool:   # varsayılan 2
    return Verdict(STOP, "tool_retry_exhausted", {"arac": tool})

# ilerleme yokluğu — dünyanın durumu değişmiyor
h = state_hash(); last3.append(h)
if len(last3) == 3 and len(set(last3)) == 1:
    return Verdict(STOP, "no_progress")
```

`state_hash` ham log değil, **durum özeti** üzerinden alınıyor — yoksa her turda değişen
zaman damgaları hash'i sürekli farklı yapar ve dedektör hiç konuşmaz.

**Çekimser kalma.** Bilgi eksikse `Verdict(STOP, "abstain_need_input")` dönüyor; döngü bunu
`Status.NEEDS_INPUT` terminal durumuna çeviriyor — hata değil, ayrı bir sonuç.

**Kanca.** `before_step` (araç bütçesi) + `on_observation` (ilerleme) + `on_stop` (rapor).

**Rapor.** `on_stop` dört alanlı bir `StopReport` üretiyor: `tried`, `found`, `why`,
`next_step`. Bu yapı çıktıyı "sonsuz döngü" olmaktan çıkarıp "faydalı kısmi sonuç" yapıyor.

**Parmak izine sonucu da katmak.** Kaynağın imzası `action_fingerprint(tool, args, outcome)`
— yani **sonuç da imzanın parçası.** `sde_offer_loop` bağımsız olarak aynı şeye varmış:
`sig = (name, args, result.error)`. Bizim döngümüz eylemi ve gözlemi ayrı hash'liyor
(`args_hash` / `result_hash`); birleştirmek denenmeye değer. Fark şurada: aynı çağrı farklı
sonuç veriyorsa (retry sonunda düzelen bir araç) ayrı imza üretir ve döngü sayılmaz.

**Çevrim taramasına basit bir alternatif.** `detect_repeat_pattern()` şunu diyor:

```python
# son 6 eylem <= 2 benzersiz parmak izine dusuyorsa dongu
if len(set(fingerprints[-6:])) <= 2:
    return True
```

Bizim k=1..12 çevrim taramasından **çok** daha basit, ve A-B-A-B'yi yakalıyor — çünkü A ile
B iki benzersiz imza eder, eşiği geçmez. `strands-entropy` ile aynı fikrin dar hali: tekrarın
şeklini tanımlamak yerine **benzersiz sayısını** saymak. Karşılaştırmalı test etmeye değer.

**Bu tasarımın kendi ölçtüğümüz sınırı.** Yukarıdaki iki eşik doğrudan kaynaktan alındı ve
**bu alana taşınmıyor.** `flaky` kontrol senaryosunun gerçek izinde:

```
adim  eylem        ekran_hash     hata
1     type         c6e01d5e835f   None
2     left_click   c6e01d5e835f   Gonder: gecici hata (1)
3     left_click   c6e01d5e835f   Gonder: gecici hata (2)
4     left_click   a63940752c21   None                     <-- BASARILI
```

`no_progress` eşiği 3 → adım 3'te tetiklenir. `retries` eşiği 2 → adım 4'ün başında
tetiklenir. **İkisi de başarıdan bir adım önce meşru koşumu keser.** Bizim ölçülmüş
`no_progress` değerimiz 8, ve gerekçesi kayıtlı: ilerleme yokluğu en gevşek sinyaldir, eşiği
en yüksek olan o olmalı.

İkinci ve daha derin sorun **anahtar granülaritesi**. `dict[araç, int]` bir metin ajanı
varsayıyor — orada araçlar anlamca ayrı (`search`, `read_file`, `bash`). Computer use'da
`left_click` işin %90'ını yapıyor: ad alanına odaklanmak da, göndermek de aynı "araç".
Yirmi farklı widget'a tıklamak tek bir sayacı kirletiyor. Anahtar `araç` değil
**`(eylem, hedef)`** olmalı.

> Kaynağın *zihniyeti* doğru, *sayıları* başka bir alandan. Uygularken eşikler bu alandan
> ölçülecek — 17 numaranın tam olarak söylediği şey.

---

## 9 · `verify-gate` — "Bitirdim" bir istektir

**Kaynak:** *Loop Engineering — Designing the Agent Loop*

**Zorluk:** Seviye 3 (Dünya)

**Tek cümlede:** Ajanın bitirme iddiasını dış bir doğrulamaya bağlar.

**Neye inanır.** Ajan işini bitirdiğini söylediğinde bu bir bilgi değil bir taleptir:
"durmak istiyorum". Öğrencinin kendi sınav kâğıdını okuması gibi.

**Nasıl çalışır.** Bitirme iddiası geldiğinde bir kapı açılır: testler geçti mi, dosya
yazıldı mı, ekranda beklenen şey var mı. Kapı açılmazsa iddia reddedilir — ama koşum
bitirilmez. Doğrulama sonucu ajanın gözlem akışına geri verilir ve döngü devam eder;
ajan kendi hatasını görüp düzeltir.

**En güçlü yanı.** Doğrulama modelin görüşüne değil **ortama** dayanır. Kandırılamaz.

**En zayıf yanı.** Ajan hiç "bitirdim" demezse kapı hiç açılmaz. Sonsuza kadar dönen bir
ajanı yakalamaz. Mutlaka bir bütçe stratejisiyle birlikte kullanılmalı.


### Teknik karşılığı

**Doğrulayıcı arayüzü.** Göreve özgü, dışarıdan takılıyor:

```python
Verifier = Callable[[RunContext], tuple[bool, str]]
```

**Kanca.** Tek kanca: `on_finish_claim`.

```python
def on_finish_claim(self, fin, ctx):
    ok, sebep = self.verifier(ctx)
    if ok:
        return CONTINUE                 # iddia kabul, döngü biter
    ctx.events.append(Event(OBSERVATION, "verify", {"failed": sebep}))
    ctx.note(f"dogrulama basarisiz: {sebep}")
    return Verdict(NUDGE, "verify_failed", {"mesaj": sebep})
```

Kritik nokta: başarısız doğrulama koşumu **bitirmiyor**. `NUDGE` dönüyor, olay akışına bir
gözlem giriyor, ve döngü devam ediyor. Ajan kendi hatasını görüp düzeltiyor.

**Sonsuz doğrulama döngüsü riski.** Ajan sürekli "bitirdim" der, doğrulama sürekli
reddederse yeni bir döngü doğar. Bu yüzden `max_verify_failures` sayacı var; aşılırsa
`Status.DEGRADED` ile iniliyor.

**Computer-use'da doğrulayıcı ne olur.** Ekranda beklenen metnin varlığı, bir dosyanın
oluşmuş olması, `state_hash`'in hedef duruma eşitliği.

---

## 10 · `galileo-breaker` — Suç ajanda değil, araçta

**Kaynak:** **Galileo** — ürün demosu transkripti (sessiz retry vakası)

**Zorluk:** Seviye 3 (Dünya)

**Tek cümlede:** Tekrar değil **hata oranı** sayar, ve bozuk aracı devre dışı bırakır.

**Neye inanır.** Bazı hatalar ajanın kusuru değildir. Ajan doğru davranıyordur, araç
bozuktur. Ve bu tür hatalar en tehlikelileridir, çünkü **başarıyla biterler.**

**Nasıl çalışır.** Bir aracın son çağrılarının yüzde kaçının hata verdiğine bakar. Oran
eşiği aşarsa o araç için devre kesici devreye girer.

Yakaladığı desen: ajan aracı çağırır, hata alır, tekrar dener, hata alır, üçüncüde başarır.
Kullanıcı doğru cevabı alır, koşum "başarılı" biter, kimse ticket açmaz — ama iki çağrı
boşa gitmiştir. Ve araç yarın tamamen çökerse her istek bütün deneme hakkını yakar.

**En güçlü yanı.** Sessiz israfı yakalar ve suçun yerini doğru gösterir.

**En zayıf yanı.** Araçlar düzgün çalışıyor ama ajanın kafası karışıksa hiçbir şey görmez.


### Teknik karşılığı

**Tuttuğu durum.** Araç başına sabit boyutlu boolean kuyruğu:

```python
errors: dict[str, deque[bool]]        # maxlen = window (örn. 10)
```

**Koşul — sayı değil oran:**

```python
q = errors[tool]
if len(q) >= min_calls and sum(q) / len(q) >= fail_rate_threshold:
    return Verdict(STOP, "tool_error_rate", {"arac": tool, "oran": sum(q)/len(q)})
```

`min_calls` (örn. 3) olmadan tek bir hata %100 oran üretir — bu yüzden şart.

**Neden tekrar sayımı işe yaramaz.** Yakaladığı desende çağrılar **başarıyla** bitiyor:
`hata → hata → başarı`. Tekrar dedektörü ekran değiştiği için susuyor, koşum `OK` dönüyor.
Yalnızca oran bunu görüyor.

**Devre kesici davranışı.** Eşik aşılınca o araç için üç seçenek:
`block` (araç çağrılamaz, ajana bildirilir) · `degrade` (alternatif araca yönlendir) ·
`stop` (koşumu bitir). Varsayılan `block` — ajan başka yol denesin.

**Boşa giden iş ölçüsü.** Başarıya ulaşan koşumlarda bile hata sayısı raporlanıyor:
`wasted_calls = sum(q)`. Sessiz israfın görünür olması bu stratejinin asıl çıktısı.

**Kanca.** `on_observation`.

---

## 11 · `agentbudget-dollar` — Doların kendisini say

**Kaynak:** **AgentBudget** — açık kaynak SDK (Python/Go/TS)

**Zorluk:** Seviye 3 (Dünya)

**Tek cümlede:** Token'ı değil doları sayar ve nihai cevap için pay ayırır.

**Neye inanır.** Token bir vekil ölçüdür; iki model arasında on beş kat fiyat farkı olabilir.
Ve bütçe koşum bittikten sonra raporlanan bir şey değil, koşum sırasında kesen bir tavandır.

**Nasıl çalışır.** Üç kademeli devre kesici: yumuşak limit uyarır, sert limit keser, döngü
tespiti ayrı çalışır. Ama asıl özgün yanı **nihai cevap payı** — bütçenin küçük bir dilimi
baştan ayrılır, sert limit erken tetiklenir, böylece ajan işin ortasında kesilmez.

Alt görevlere iç içe bütçe dağıtır ve harcama üst bütçeye toplanır. Tekrar tespitini de
zamana bağlar: "on çağrı" değil, "bir dakikada on çağrı".

**En güçlü yanı.** Pahalı olan her şeyi yakalar, döngü olsun olmasın.

**En zayıf yanı.** Ucuz döngüleri kaçırır. Küçük bir model saatlerce dönüp tavana hiç
değmeyebilir.


### Teknik karşılığı

**Tuttuğu durum.** Harcama defteri (`spent_usd`), rezerv oranı, ve zaman damgalı imza
kuyruğu.

**Rezervli sert limit.**

```python
hard_at = max_spend * (1 - finalization_reserve)   # örn. 1.00 * (1-0.05) = 0.95
if spent >= hard_at: → DEGRADE (nihai cevap için 0.05 duruyor)

def would_exceed(est):        # son çağrıdan önce kontrol
    return spent + est > max_spend
```

Bu, bizim mevcut `_force_finish()` yaklaşımından daha dürüst: orada nihai cevap tavanı
aştıktan **sonra** çağrılıyor, burada payı baştan ayrılmış.

**Zaman pencereli tekrar.**

```python
q.append((sig, now))
while q and now - q[0][1] > loop_window_seconds: q.popleft()
if count(q, sig) >= max_repeated_calls: → tetikle
```

Sayı değil, **birim zamandaki sayı**. Yavaş kronik döngüler sayı eşiğine hiç ulaşmadan
para yakabildiği için.

**İç içe bütçe.** `child = parent.child(max_spend=X)`; çocuğun harcaması ebeveyne toplanıyor,
ebeveyn tavanı çocuğu da bağlıyor.

**Kanca.** `before_step` (tavan + rezerv) · `on_observation` (zaman pencereli tekrar).

---

## 12 · `modexa-statemachine` — Döngünün şeklini kısıtla

**Kaynak:** Modexa — *Ajan Döngüsü Problemi: "Akıllı" Sistemler Durmadığında*

**Zorluk:** Seviye 4 (Şekil)

**Tek cümlede:** Döngüyü tespit etmeye çalışmaz; oluşamayacağı bir yapı kurar.

**Neye inanır.** Ajanlar belirsiz özgürlükleri sever, sistemlerin net durumlara ihtiyacı
vardır. Bir geçişe izin vermezseniz o döngü hiç doğmaz.

**Nasıl çalışır.** Serbest akışlı döngüyü bir durum makinesine indirger: ANLA → TOPLA →
EYLEM → DOĞRULA → YANIT → DEVRET. Yalnızca belirli geçişlere izin verir.

Yanına bir **geri dönüş merdiveni** koyar: ajan kendi kendine tekrar deneme icat etmez,
sabit bir sırayı izler — bekleyip bir kez dene, aracı değiştir, kapsamı daralt,
kullanıcıya sor, elindekiyle cevap ver.

**En güçlü yanı.** Bütün bir döngü sınıfını oluşmadan yok eder.

**En zayıf yanı.** Esnekliği azaltır. Öngörülemez işler durum makinesine sığmaz.

**Kaynağın niteliği.** Kavramsal makale, ölçüm yok. Tek bir anlatısal vaka var — destek
ajanının *"tekrar doğrula"* tuzağı. 8 numaralı kaynakla önemli ölçüde örtüşüyor; ama üç şeyi
tek başına getiriyor: aşağıdaki beşinci döngü sebebi, "kullanıcıya sor"un nasıl yapılacağı,
ve durum makinesi önerisi.

**Beşinci döngü sebebi — bu listede başka hiçbir kaynakta yok.**

> *"Ajanın **'yanlış yapmamak' üzerine optimize edilmesi.**"*

Aşırı tedbir teşvik eden bir prompt, en güvenli yolu "bir kez daha doğrula" yapıyor. Ajan
bozuk değil; **tam da söylendiği gibi davranıyor.** Kaynağın cümlesi:

> *"Bir sorun çözücü inşa etmediniz. **Bir riskten kaçınma makinesi** inşa ettiniz."*

Bunun önemi şu: bu döngünün kaynağı kod değil, **prompt.** Bu belgedeki on yedi
mekanizmanın hiçbiri onu çözmüyor — hepsi ancak sonucu keser. Guardrail'in kapsam sınırının
en net örneği.

**"Kullanıcıya sor"un nasıl yapılacağı.** Çekimser kalmak bir yetenek değil, bir tasarım işi:
**tek soru sor** (beş değil) · **neyi değiştireceğini açıkla** · kullanıcı umursamazsa
**varsayılan sun**. Gerekçesi ölçüm değil ama ikna edici:

> *"40 adım boyunca yanlış tahmin yürütmekten ucuzdur."*

**Geliştirici kuralı.** Kaynağın kapanış cümlesi, bu belgenin de kapanışı olabilir:

> *"Her ajan döngüsünün **tasarlanmış bir çıkışı** olmalıdır — başarı, geri dönüş, sorma veya
> devretme. 'Sonsuza kadar dene' değil. 'Umarım model çözer' değil."*


### Teknik karşılığı

**Tuttuğu durum.** Mevcut durum (enum) + izinli geçiş tablosu + merdiven indeksi.

```python
ALLOWED = {
  ANLA:    {TOPLA, DEVRET},
  TOPLA:   {EYLEM, TOPLA, DEVRET},     # kendine dönüş sınırlı
  EYLEM:   {DOGRULA, DEVRET},
  DOGRULA: {YANIT, EYLEM, DEVRET},     # EYLEM'e dönüş var ama TOPLA'ya YOK
  YANIT:   set(),                       # terminal
  DEVRET:  set(),
}
```

Kritik tasarım: `DOGRULA → TOPLA` geçişi **yok**. "Doğrulayamadım, baştan toplayayım"
döngüsü böyle imkânsız hale geliyor.

**Eylem → durum eşlemesi.** Her araç bir duruma ait; ajan başka durumun aracını çağırırsa
`on_action` reddediyor ve merdivene giriyor.

**Geri dönüş merdiveni.** Sıralı bir liste, indeks yalnızca ileri gidiyor:

```python
LADDER = [backoff_retry, switch_tool, narrow_scope, ask_user, best_effort_answer]
i = ladder_index; ladder_index += 1
```

Ajan kendi tekrar denemesini icat etmiyor; merdiven bittiğinde `best_effort_answer` zorunlu.

**`ask_user` basamağının sözleşmesi.** Merdivenin dördüncü basamağı serbest bırakılmıyor —
üç kuralı var: tek soru · neyin değişeceğinin açıklaması · cevapsızlıkta varsayılan. Döngüde
`Status.NEEDS_INPUT` terminal durumuna çıkıyor (8 numaranın `abstain`'iyle aynı yere).

**Kanca.** `on_action` (geçiş denetimi) + `on_stop` (merdiven sonu).

**Maliyet.** O(1) sözlük araması.

---

## 13 · `autogen-static` — Koşum başlamadan yakala

**Kaynak:** **AutoGen** — `GraphFlow` graf doğrulaması

**Zorluk:** Seviye 4 (Şekil)

**Tek cümlede:** Ajan akışının şeklini çalıştırmadan önce denetler.

**Neye inanır.** En iyi döngü tespiti, çalışma zamanında hiç yapılmayan tespittir.

**Nasıl çalışır.** Ajan grafiği kurulurken doğrulanır: bir çevrim var ve o çevrimin çıkış
koşulu yoksa sistem **hiç başlamaz**, hata verir.

**En güçlü yanı.** Çalışma zamanında sıfır maliyet, sıfır yanlış pozitif — çünkü çalışma
zamanı hiç gelmez.

**En zayıf yanı.** Yalnızca **yapısal** döngüleri yakalar. Grafiğin şekli kusursuz olabilir
ve model yine de aynı düğümde takılabilir. Derleyicinin tip hatalarını yakalayıp sonsuz
döngüleri yakalayamaması gibi.


### Teknik karşılığı

**Koşumdan önce çalışıyor.** Tek kanca `on_run_start`; oradan sonra hiçbir şey yapmıyor.

**Graf kurulumu.** Ajan yapılandırmasından bir yönlü graf çıkarılıyor: düğümler = durumlar
ya da alt-ajanlar, kenarlar = izinli geçişler.

**Çevrim tespiti — Tarjan SCC:**

```python
for scc in tarjan_scc(graph):
    if len(scc) > 1 or has_self_loop(scc):
        # bu bilesene girildiginde geri donulebiliyor
        if not any(is_exit_edge(e) for e in edges_from(scc)):
            raise ConfigError(
                f"Cycle detected without exit condition: {scc}")
```

`is_exit_edge` = bileşenin dışına çıkan ve **deterministik** bir koşula bağlı kenar.
Modelin kararına bağlı çıkışlar geçerli sayılmıyor — çünkü model her zaman "devam" diyebilir.

**Maliyet.** O(V+E), koşum başına bir kez, mikrosaniye mertebesinde. Çalışma zamanında
sıfır.

**Yanlış pozitif.** Yok — ama yalnızca ifade edilebilen yapı hakkında konuşuyor. Model
kararına bağlı döngüler bu analizin dışında kalıyor; onları çalışma anındaki pencere
dedektörleri (5–8, 15) ve sayaçlar (1, 3) yakalamak zorunda.

**Bizim mimarimizdeki karşılığı.** `modexa-statemachine`'in geçiş tablosu bu analizin
girdisi olabilir — durum makinesi tanımlandığı anda statik olarak doğrulanabilir.

---

## 14 · `telemetry-repair` — Kural koy, ihlali söyle, geri sar

**Kaynak:** 📄 **_Real-Time Detection and Repair of LLM Agent Failures_** — 2.823 bölüm, 25 veri kümesi, 3 framework

**Zorluk:** Seviye 5 (Kademe)

**Tek cümlede:** Basit kurallarla hatayı yakalar, geri sarar ve **hangi kuralın
çiğnendiğini** söyleyerek onarır.

**Neye inanır.** Yakalamak yetmez, onarmak gerekir. Ve onarırken ajana cevabı vermek
yanlıştır — kuralı söylemek daha iyidir.

**Nasıl çalışır.** Üç deterministik kontrol: ajanın söylediği toplam gerçekten gördüğü
verilerden çıkıyor mu · gereken bütün araç çağrıları yapıldı mı · aracın döndürdüğü şey o
aracın üretebileceği bir biçimde mi. Bunlar kuraldır, istatistik değil — eşik ayarı,
kalibrasyon, ikinci model gerektirmezler.

Bir kontrol düştüğünde ajan son sağlam noktaya geri sarılır ve **yalnızca hangi kontrolün
düştüğü söylenerek** yeniden çalıştırılır.

**En güçlü yanı.** Ölçülmüş: bu üç basit kural, aynı hataları istatistiksel bir anomali
dedektöründen daha iyi yakalıyor ve sıfır yanlış pozitif üretiyor. Onarım tarafında da
"hangi kontrol düştü" (%45) "doğru cevabı ver"i (%36) geçiyor.

**En zayıf yanı.** Döngüde onarım işe yaramıyor — yeniden çalıştırma çoğu zaman aynı
döngüyü üretiyor. Döngüde doğru hamle onarmak değil durmak.


### Teknik karşılığı

**Üç deterministik kontrol** — saf fonksiyonlar, eşiksiz, kalibrasyonsuz:

```python
def total_consistency(ctx) -> bool:
    # ajanin bildirdigi toplam, gordugu arac sonuclarindan yeniden hesaplaniyor
    return abs(claimed_total(ctx) - recompute_from_observations(ctx)) < eps

def required_coverage(ctx) -> bool:
    # gorev icin sart olan araclarin hepsi cagrildi mi
    return REQUIRED_TOOLS <= {e.name for e in ctx.events if e.kind is ACTION}

def tool_contract(ctx) -> bool:
    # sonuc, o aracin uretebilecegi bicimlerden biri mi
    return all(matches_schema(r, SCHEMA[r.tool]) for r in ctx.results)
```

**Checkpoint.** Son "bilgi toplama" adımının indeksi + o andaki sandbox durumu:

```python
Checkpoint = (event_index, screen_hash, budget_snapshot)
```

**Onarım.** Kontrol düşünce checkpoint'e geri sarılıyor ve **yalnızca hangi kontrolün
düştüğü** enjekte ediliyor:

```python
ctx.events = ctx.events[:cp.event_index]
req.notes.append(f"'{failed_check}' kontrolu basarisiz. Yaklasimini gozden gecir.")
```

Ölçülmüş kurtarma oranları: hiçbir şey %0 · yeniden örnekleme %16 · **hangi kontrol düştü
%45** · doğru cevabı ver %36 · genel uyarı %36.

**Onarım bütçesi.** `max_repairs=1` — ikinci kez düşerse eskalasyon. Ve **döngü sebebiyle
tetiklenmişse onarım hiç denenmiyor**, çünkü ölçüm yeniden çalıştırmanın aynı döngüyü
ürettiğini gösteriyor.

**Kanca.** `on_observation` (kontroller) + `on_stop` (onarım kararı).

---

## 15 · `pi-signature` — Altı ucuz sinyal, kademeli müdahale

**Kaynak:** `pi-anti-doom-loop` — yayımlanmış npm eklentisi

**Zorluk:** Seviye 5 (Kademe)

**Tek cümlede:** Tek bir akıllı dedektör aramak yerine altı basit sinyali birden dinler.

**Neye inanır.** Hiçbir tek sinyal kesin değildir. Ama altı tanesi aynı anda konuşuyorsa
yanılma ihtimali düşer. Ve bu sinyallerin hiçbiri pahalı olmamalı — yapay zekâ çağırmadan,
sadece sayarak çalışmalı.

**Nasıl çalışır.** Altı şeye bakar: aynı aracın aynı argümanlarla tekrarı · aynı aracın üst
üste hata vermesi · aynı metnin kelimesi kelimesine tekrarı · tek bir mesaj içinde aynı
cümlenin defalarca geçmesi · **ardışık mesajların %55'ten fazla ortak kelime taşıması** ·
ve bu benzer metinlerin pencerede birikmesi.

Tetiklendiğinde tek hamlede durdurmaz. Önce ajanı **yönlendirir** ve koşum devam eder;
ısrar ederse keser ve bir kez daha başlatır; yine ısrar ederse gerçekten durur. Ayrıca
tekrarlarda boşa giden token'ı hesaplayıp raporlar.

**En güçlü yanı.** Beşinci sinyal — "aynı şeyi farklı kelimelerle söyleme". İmza
karşılaştırmasının kör kaldığı yeri, gömme modeli kullanmadan, sadece kelime sayarak
kapatıyor.

**En zayıf yanı.** Ajan gerçekten farklı şeyler deniyor ama hiçbiri işe yaramıyorsa hiçbir
sinyal konuşmaz.


### Teknik karşılığı

**Tuttuğu durum.** Kayan pencere (varsayılan son **10** çağrı) `(araç, argümanHash)` ·
son N asistan metni · **imza başına ayrı bir engelleme sayacı** · oturum bazlı
steer/abort sayaçları.

**Altı yüklem ve varsayılanları:**

```python
# 1  ayni (arac, arguman)          — 3x / son 10 cagri
if window.count(sig) >= 3: block
# 2  ayni aracin ardisik hatasi    — 3x
if all(r.error for r in last_results(tool, 3)): block
# 3  birebir ayni metin            — 3x / pencere
if texts.count(t) >= 3: block
# 4  TEK mesaj icinde ayni cumle   — 3x+
if max(Counter(sentences(t)).values()) >= 3: block
# 5  yakin-benzer ARDISIK metin    — 3x ust uste
if all(jaccard(texts[-i], texts[-i-1]) >= 0.55 for i in (1,2,3)): block
# 6  yakin-benzer CEVRIM           — 3x / pencere, ne birebir ne ardisik
if sum(jaccard(t, x) >= 0.55 for x in window_texts) >= 3: block
```

5 ve 6 arasındaki fark kritik: beşincisi ardışık yeniden ifadeyi, altıncısı **dönüşümlü**
yeniden ifadeyi yakalıyor — model A der, B der, A'yı başka kelimelerle der.

**İki ayrı müdahale yolu.**

*Araç çağrısı:* engelleme + modele **öğretici gerekçe** ("yaklaşımını değiştir, farklı bir
araç kullan, ya da kullanıcıya sor"). Model yok sayıp aynı çağrıyı tekrar ederse tur iptal.

*Metin döngüsü — üç kademe:* **yönlendir** (koşum sürer) → **iptal + bir taze devam
direktifi** → gerçekten iptal, kontrol kullanıcıya. Otomatik devam bütçesi tavanlı.

**İnce ayrıntı — engellenen çağrı pencereye girmiyor:**

```python
if blocked:
    block_count[sig] += 1          # AYRI sayac
    if block_count[sig] >= 2: abort_turn()
    # window'a EKLENMIYOR
```

**Yanlış pozitife karşı üç savunma.** Sayaçlar **her kullanıcı istemiyle sıfırlanıyor** —
aynı oturumda meşru olarak tekrarlanan görev asla yanlış pozitif olmuyor ·
`TOOLS_EXCLUDE` ile belirli araçlar hiç izlenmiyor · `/loopcheck suspend` ile manuel
askıya alma.

**Eşikler minimum 2'ye kırpılıyor** — bozuk bir yapılandırma ajanı kilitlemesin.

**Varsayılanda KAPALI iki sinyal.** `TIME_WINDOW=0` (açılırsa uzun oturumdaki yavaş kronik
döngüleri yakalar, eski pencere girdilerini süreye göre düşürür) ve `FAIL_RATE=0`
(açılırsa aracın penceredeki hata payına bakar, en az `FAIL_RATE_MIN=3` çağrıdan sonra).
İkincisi **`galileo-breaker` zihniyetinin ta kendisi** — pi'de opsiyonel bayrak olarak duruyor.

**Token muhasebesi.** ~4 karakter/token tahminiyle tekrarlarda yakılan token hesaplanıp
engelleme gerekçesine yazılıyor: *"~N tokens burned on repeats."*

**Kanca.** Üç olay: `tool_call` · `tool_result` · `message_end`. Bizim protokolde sırasıyla
`on_action` · `on_observation` · `on_observation` (Say olayı); enjeksiyon `decorate_request`.

**Maliyet.** O(pencere) — sabit. Jaccard kelime kümeleri önceden hesaplanıyor. Model
çağrısı yok, gömme yok.
## 16 · `voi-allocation` — Bütçe bir tavan değil, bir bütçe

**Kaynak:** 📄 **_Inference-Time Budget Control for LLM Search Agents_** — 4 veri kümesi × 4 bütçe × 3 omurga

**Zorluk:** Seviye 6 (Karar)

**Tek cümlede:** "Ne zaman dur" değil, "parayı nereye harca" diye sorar.

**Neye inanır.** Bir tavan koyduğunuzda ajan tavana kadar istediğini yapar. Oysa aynı
parayla çok daha iyi bir sonuç alınabilirdi — doğru yerlere harcansaydı.

**Nasıl çalışır.** Her adımda seçenekleri **birim bütçe başına faydaya** göre puanlar. Ham
faydaya göre değil: pahalı bir eylem biraz daha faydalı olabilir, ucuz bir eylem neredeyse
bedavadır. Oranı yüksek olan kazanır.

İki bütçeyi birlikte takip eder — araç çağrısı ve token. Biri kritik seviyeye inince keşif
pahalılaşır, bitirme çekici hale gelir. Üstünde koruyucu kurallar vardır ki bütçe bitince
ajan ucuz olduğu için aceleyle kötü bir cevaba kaçmasın.

**En güçlü yanı.** İsrafı yakalar — döngü olmayan ama gereksiz olan işi.

**En zayıf yanı.** Ölçülmüş: kazancı kaynak kıtken büyük, bol kaynakta eriyor.


### Teknik karşılığı

**Bütçe baskısı.** İki eksenden **en kritik** olanı belirliyor:

```
ρ = 1 − min( b_tool / B_tool ,  b_tok / B_tok )
```

ρ→1 giderken en az bir bütçe tükeniyor demektir.

**Eylem puanı.** Her aday eylem `k` için:

```
u(k) = Δ̂(k)  +  Ψ(k)  −  Π(k; ρ)
r(k) = max(u(k), 0) / ( d(k) + ε )
```

- `Δ̂(k)` — bu eylemin nihai kaliteye tahmini katkısı
- `Ψ(k)` — yapısal sinyaller (kanıt yeterli mi, döngüye girdi mi)
- `Π(k; ρ)` — **bütçe cezası**, ρ arttıkça keşif eylemleri pahalılaşıyor
- `d(k)` — eylemin bütçe maliyeti (araç çağrısı + beklenen token)

Seçilen eylem `argmax r(k)`. Ham faydayı değil **birim bütçe başına faydayı** maksimize
ediyor — yoksa pahalı eylemler gereğinden fazla seçilir.

**Ablasyon bulgusu:** `Π(k;ρ)` terimi çıkarıldığında performans her veri kümesinde düşüyor
(bir sette F1 0,63 → 0,43). Yani asıl katkı puanlama değil, **kalan bütçenin karara açıkça
katılması**.

**Koruyucu kurallar.** Saf oran optimizasyonu tehlikeli: bütçe azalınca "cevap ver" en ucuz
seçenek olduğu için aşırı çekici hale geliyor. Üstte deterministik kapılar var —
kanıt zayıfken erken cevap bastırılıyor, bileşimsel görevde en az bir keşif zorunlu.

**Kanca.** `before_step` — ama diğerlerinden farklı olarak *durdurmuyor*, `Verdict.detail`
içinde tercih edilen eylemi taşıyor; döngü bunu modele kısıt olarak veriyor.

---

## 17 · `improvement-loop` — Bu koşumu kurtarma, eşikleri düzelt

**Kaynak:** OpenAI cookbook — *Build an Agent Improvement Loop with Traces, Evals, and Codex*

**Zorluk:** Seviye 6 (Karar)

**Tek cümlede:** Hiç müdahale etmez; kaydeder ve sonraki koşumların eşiklerini önerir.

**Neye inanır.** "Adım limiti 12" yazdığınızda o 12 çoğu zaman hiçbir yerden gelmez.
Birinin makul bulduğu bir sayıdır. Ve yanlış seçilmiş bir limit sinsi hasar verir.

**Nasıl çalışır.** Her adımı ayrıntılı kaydeder, sonunda dağılımdan eşik önerir: başarılı
koşumlar kaç adım sürmüş, kuyruğu nerede, tavan nereye konmalı. Sonra izler — koşumların
yüzde kaçı limitte sonlanıyor?

Disiplin tarafı: döngü ayarları prompt'la birlikte sürümlenir ve bir terfi kapısından
geçer. **Eşik değiştirmek, kod değiştirmekle aynı süreçten geçmeli.**

**En güçlü yanı.** Diğer on altı stratejinin eşiklerini tahminden çıkarıp ölçüme bağlar.

**En zayıf yanı.** Şu anki koşumu kurtarmaz. Bir koruma değil, bir ölçüm aracıdır.


### Teknik karşılığı

**Koşum sırasında hiçbir kanca tetiklenmiyor.** Tek işi `on_stop`'ta bir özet satırı
yazmak. `before_step` ve `on_observation` boş — bu strateji bilerek pasif.

**Çevrimdışı analiz.** `cua-lab thresholds <trace-dir>` komutu:

```python
basarili = [t for t in traces if t.status == "OK"]
adimlar  = sorted(len(t.spans) for t in basarili)
p50, p95, p99 = percentile(adimlar, [50, 95, 99])
öneri = ceil(p99 * 1.2)          # kuyrugun ustune pay
```

Aynı hesap token, süre ve maliyet için de yapılıyor.

**İzleme metriği.** Öneriden daha önemlisi: koşumların yüzde kaçı limitte sonlanıyor?

```python
limit_orani = count(status in ("BUDGET_EXHAUSTED","CEILING")) / count(all)
```

Bu oran tırmanıyorsa görev zorluğu ya da araç güvenilirliği değişmiştir — limit yanlış
değildir. Arize'ın uyarısı: dar bir limit, **modelin kötüleştiği gibi görünen** sessiz bir
kalite gerilemesine dönüşür.

**Sürümleme.** Eşik seti bir kayıt olarak tutuluyor:

```python
{"version": "v003", "status": "promoted", "promotion_gate": "manual_review",
 "max_steps": 14, "derived_from": "traces/2026-08-24/*", "n_runs": 120}
```

Eşik değiştirmek kod değiştirmekle aynı süreçten geçiyor.

---
## Hızlı seçim rehberi

| Derdiniz buysa | Buraya bakın |
|---|---|
| Hiçbir şey yok, bir yerden başlamam lazım | **1** |
| Ajan aynı şeyi tekrarlıyor | 5, 6, 15 |
| Ajan farklı şeyler deniyor ama ilerlemiyor | 5, 8 |
| Fatura kabarıyor | 1, 11, 16 |
| Ajan yarım işi "bitti" diye teslim ediyor | 9, 14 |
| Koşum başarılı bitiyor ama içi boş | 10, 14 |
| Bir araç bozuk ve ajanı sürüklüyor | 10 |
| Ajan işin ortasında kesiliyor, iş çöpe gidiyor | 2, 4, 11 |
| Limit koyunca model erken pes ediyor | 3 |
| Uzun koşumda bağlam sıkıştırma döngüye sokuyor | 7 |
| Eşikleri nereden bulacağımı bilmiyorum | 17 |
| Bu döngünün hiç oluşmamasını istiyorum | 12, 13 |
| Aynı bütçeyle daha iyi sonuç istiyorum | 16 |
