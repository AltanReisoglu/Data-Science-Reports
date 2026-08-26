# Sunum Metni — Agent Reliability: Loop Detection & Task Budget Kontrolleri

**Süre:** ~25 dakika anlatım + 5 dakika demo + soru
**Yapı:** Problem → Kanıt → Sektör ne yapıyor → Metodolojiler (6 seviye) → Demo → Değerlendirme

> **Nasıl kullanılır:** `[SLAYT]` slaytta ne yazacağını, `[SÖYLE]` senin ne diyeceğini,
> `[TAHTA]` canlı çizeceğin şeyi, `[SORARLARSA]` beklenen itirazın cevabını gösteriyor.
> `[SÖYLE]` blokları konuşma dilinde yazıldı — okuyup ezberleme, ama cümle kurulumları
> hazır dursun.

---
---

# BÖLÜM 0 · Açılış (2 dk)

## [SLAYT 1] — Başlık

> **Agent Reliability**
> Loop Detection & Task Budget Kontrolleri
>
> Araştırma + çalışan PoC
> *Altan · Ağustos 2026*

## [SÖYLE]

İki haftadır şu soruyla uğraşıyorum: **bir ajan ne zaman durur, ve bunu kim garanti eder?**

Bugün üç şey anlatacağım. Birincisi problemin ne olduğu — ve bunun bir "olabilir" değil,
ölçülmüş bir şey olduğu. İkincisi bu problemi çözmek için geliştirilmiş yöntemlerin neler
olduğu ve **nasıl çalıştıkları** — asıl kısım bu. Üçüncüsü de bunları kendi yazdığım bir
ajanda çalıştırıp ölçtüğüm sonuçlar.

Sonda avantaj-dezavantaj tablosu ve bir sisteme entegre ederken nelere ihtiyaç duyulduğu
var.

Başlarken bir uyarı: burada söyleyeceğim her sayının kaynağı var ve çoğunu kaynak kodundan
okuyarak doğruladım. Doğrulayamadıklarımı "bu iddia, ben ölçmedim" diye ayıracağım.

---
---

# BÖLÜM 1 · Problem (5 dk)

## [SLAYT 2] — Ajan nedir, aslında

> **Model:** sorarsın → cevap verir → **biter**
>
> **Ajan:** sorarsın → cevap verir → **bir şey yapar** → sonucu geri verirsin → tekrar sorarsın → ...
>
> Ajan = model + **döngü**

## [SÖYLE]

Önce bir şeyi netleştirelim, çünkü problemin tamamı burada.

Bir dil modeli tek başına **durur.** Sorarsınız, cevap verir, biter. Durma sorunu diye bir
şey yoktur.

Ajan dediğimiz şey o modeli bir döngünün içine koymaktır. Cevabı al, söylediği aracı
çalıştır, sonucu geri ver, tekrar sor. Ajanın bütün gücü buradan geliyor — kendi
adımlarına kendisi karar verebiliyor.

Ama aynı yerden bir soru doğuyor, ve bu soru modelde hiç yoktu: **bu döngüyü kim kesecek?**

## [SLAYT 3] — En doğal cevap, en kötü cevap

> **"Model kendisi karar versin."**
>
> Model her turda yalnızca **bir sonraki adıma** bakıyor.
> Kaç kez aynı şeyi denediğini görmüyor.
> Ne harcadığını görmüyor.

## [SÖYLE]

En doğal cevap "model kendisi karar versin" olur. Zaten akıllı, işi o yapıyor, bittiğini de
o söylesin.

Pratikte bu en kötü cevap. Sebebi modelin aptallığı değil — **görüş alanı.** Model her
turda önündeki konuşmaya bakıp bir sonraki adımı seçiyor. Kaç kez aynı şeyi denediğini,
işin genel gidişatını, ne kadar para harcadığını görmüyor. Görse bile o bilgiyi doğru
yorumlayacağının garantisi yok.

Arize'ın yazısındaki cümle bunu çok net koyuyor, ben de bütün çalışmanın çıkış cümlesi
yaptım:

> *"Her döngüye modelin yargısına bağlı olmayan sert bir durdurma gerekiyor, çünkü model,
> işin bitip bitmediği konusunda yanılması **en muhtemel bileşendir**."*

Bu cümleyi kabul ettiğiniz anda soru değişiyor. Model karar vermeyecekse: **kim verecek, ve
neye bakarak?**

Bugünkü sunumun geri kalanı bu sorunun farklı cevapları.

## [SLAYT 4] — Bu gerçekten oluyor mu? Üç ölçüm

> **MAST** — 1642 açıklamalı yürütme izi, 7 framework
> Adım tekrarı **%15,7** ← en sık hata modu
> Durma koşulunu tanımama **%12,4**
> **Toplam %28,1** — hataların dörtte birinden fazlası
>
> **Token Budgets** — 21 framework'ten 63 doğrulanmış üretim olayı
> *"Katalogda, bir kullanıcı parasını ödemeden önce engellenmiş tek bir bütçe aşımı vakası bulamadık."*
>
> **Anthropic'in kendi referans computer-use döngüsü**
> `while True:` — tur sayacı yok, döngü tespiti yok, bütçe yok

## [SÖYLE]

"Teoride olabilir" demek kolay. Üç tane ölçülmüş şey göstereyim.

**Birincisi MAST.** Çok ajanlı sistemlerin neden bozulduğunu inceleyen bir çalışma — 1642
tane açıklanmış yürütme izi, yedi farklı framework'ten. On dört hata modu çıkarmışlar ve
frekanslarını saymışlar.

En sık görülen hata modu **adım tekrarı**: yüzde 15,7. Yani ajanın zaten tamamladığı bir
adımı gereksiz yere tekrar etmesi. Listenin en tepesinde bu var.

Beşinci sırada da **durma koşulunu tanımama** var: yüzde 12,4. İkisini toplarsanız yüzde
28,1. Bütün hataların dörtte birinden fazlası, tek bir başlıkta: **ajan ne zaman
duracağını bilmiyor.**

**İkincisi Token Budgets.** Bu bir vaka kataloğu — 21 orkestrasyon framework'ünden 2023 ile
2026 arasındaki 63 doğrulanmış üretim olayı. Her biri alıntılanmış bir GitHub issue,
maintainer beyanı, varsa dolar zararıyla.

Kataloğun sonuç cümlesi şu — bu cümleyi slayta koydum çünkü bence tek başına problemin
tanımı:

> *"Katalogda, bir kullanıcı parasını ödemeden önce engellenmiş tek bir bütçe aşımı vakası
> bulamadık."*

Düzeltmeler geliyor — aynı gün, bir iki gün içinde. Ama **ancak fatura geldikten sonra.**
Yani sektör bu problemi çözüyor değil, ödedikten sonra yamalıyor.

**Üçüncüsü de bana en çarpıcı geleni.** Anthropic'in yayımladığı referans computer-use
döngüsünü okudum. Kodun kalbi şu:

## [TAHTA] — bunu tahtaya yaz, sadece bu

```python
while True:
    response = model(messages)
    if not tool_use:
        return messages     # tek çıkış
    result = run_tools(...)
    messages.append(result)
```

## [SÖYLE]

`while True`. Tek çıkış: modelin araç çağırmayı bırakması.

Tur sayacı yok. Döngü tespiti yok. Bütçe yok. Yani **döngüyü kesme yetkisi tamamen
modelde** — az önce "yanılması en muhtemel bileşen" dediğimiz şeyde.

Bu bir eleştiri değil, referans kod bunu göstermek zorunda değil. Ama şunu gösteriyor:
bugün bir ajan yazmaya oturduğunuzda, elinizin altındaki başlangıç noktasında hiçbir
koruma yok. Korumayı siz ekleyeceksiniz.

## [SORARLARSA] "Bu abartı değil mi, kaç kişinin başına geldi?"

Token Budgets kataloğu 63 doğrulanmış olay diyor ve hepsinin GitHub issue linki var. Ben
kataloğun tamamını değil, örnek olayları okudum — mesela **MAST-014**: tek bir gözlemci
LLM çağrısında 2 milyon token. Yani gözlemlenebilirlik katmanının kendisi maliyet
amplifikatörüne dönüşmüş. **MAST-004** de şu: `TokenLimiter` var ama döngü iterasyonu
başına tetiklenmiyor — sınır var, yanlış yerde.

---
---

# BÖLÜM 2 · Sektör ne yapıyor? (4 dk)

## [SLAYT 5] — 31 harness, kaynak kodundan

> **31 agent harness'ı + gateway katmanı** kod düzeyinde tarandı
> LangGraph · CrewAI · smolagents · pydantic-ai · Google ADK · AutoGen · OpenHands ·
> Gemini CLI · Cline · Continue · Strands · AgentScope · OpenClaw · …
>
> Etiketleme: `[K]` kaynak kodu okundu · `[D]` yalnız doküman · `[?]` doğrulanamadı

## [SÖYLE]

"Nasıl çözülmeli" sorusuna geçmeden önce **nasıl çözülüyor** diye baktım. Otuz bir tane
agent harness'ının kaynak kodunu okudum — loop detection ve bütçe zorlaması nerede, hangi
varsayılanla, hangi eşikle.

Doküman okuyup yazmadım, kaynak koduna baktım. Yazdığım her satırı `[K]` kod okundu, `[D]`
sadece doküman, `[?]` doğrulayamadım diye etiketledim. Bunu yapmamın sebebi şu — dokümanla
kod arasında sistematik bir fark var, birazdan göstereceğim.

## [SLAYT 6] — Varsayılan limitler

> | Harness | Varsayılan sınır |
> |---|---|
> | smolagents | `max_steps = 20` |
> | LangGraph | `recursion_limit = 25` |
> | CrewAI | `max_iter = 25` |
> | pydantic-ai | `request_limit = 50` |
> | OpenHands | `max_iteration_per_run = 500` |
> | **Google ADK `LoopAgent`** | **`max_iterations = None`** → sınırsız |

## [SÖYLE]

Önce iyi haber: çoğunun bir sınırı var. Ve sayılar makul aralıkta — yirmi, yirmi beş, elli.

Kötü haber en altta. Google'ın Agent Development Kit'inde `LoopAgent` diye bir sınıf var,
adı üstünde döngü ajanı. `max_iterations` parametresi `None` varsayılanla geliyor ve `None`
demek **sınırsız** demek. Yani "döngü" adını taşıyan bir bileşen, varsayılan halinde
durmuyor.

## [SLAYT 7] — Asıl bulgu: "var mı" yanlış soru

> **24 harness'ın 12'sinde mekanizma var — ama varsayılanda KAPALI**
>
> Gemini CLI: `maxSessionTurns ?? -1`, kontrol `> 0` ile korunuyor → varsayılan **kapalı**
> Continue: `depth > 50` koruması `NODE_ENV === "test"` ile birlikte → üretimde **hiç tetiklenmiyor**
> OpenClaw: `tools.loopDetection.enabled` varsayılan **`false`**
>
> Ve doküman-kod çelişkisi, aynı dosyada:
> Hermes `init_agent(max_iterations=…)` → gerçek varsayılan **`sys.maxsize`** (`:523`)
> aynı fonksiyonun docstring'i → *"default: 90"* (`:603`)
>
> **Doğru soru: "varsayılanda açık mı?"**

## [SÖYLE]

Ve şimdi taramanın asıl bulgusu.

İncelediğim yirmi dört harness'ın on ikisinde bir mekanizma **var** — ama **varsayılanda
kapalı.** Yani dokümana bakıp "bu framework loop detection destekliyor" diye yazarsanız
teknik olarak doğru söylemiş olursunuz, ama gerçekte hiçbir koruma çalışmıyor.

Üç somut örnek — üçünü de kendim kaynak kodundan doğruladım.

**Gemini CLI:** `config.ts`'te tur sınırı `params.maxSessionTurns ?? -1` diye okunuyor.
Yani parametre verilmezse eksi bir. Kontrol de `> 0` ile korunuyor. Eksi bir hiçbir zaman
sıfırdan büyük olmadığı için kontrol hiç çalışmıyor. Varsayılan: kapalı.

**Continue:** elli derinlikte bir koruma var. Ama koşulun içinde `process.env.NODE_ENV ===
"test"` de var. Hata mesajı kendini ele veriyor zaten: *"Max stream depth of 50 reached in
test"*. Üretimde hiç tetiklenmiyor.

**OpenClaw** — bunu bu bilgisayardaki kurulu sürümden okudum, `tools.loopDetection.enabled`
varsayılan değeri `false`. Üç ayrı adlandırılmış dedektörü var, eşikleri ayarlanmış,
sıkıştırma sonrası özel koruması var — hepsi yazılmış ve varsayılanda kapalı.

Ve en başta "dokümanla kod arasında sistematik bir fark var" demiştim — en net örneği şu.
Hermes'in kütüphane API'sinde `init_agent` fonksiyonu var, `max_iterations` parametresi
alıyor. Gerçek varsayılan değeri `sys.maxsize` — yani pratikte sınırsız. **Aynı fonksiyonun
seksen satır aşağıdaki docstring'i** ise "default: 90" diyor.

İkisi de aynı dosyada. Dokümana bakan "doksan iterasyon sınırı var" der. Kod sınırsız.

Bunun benim için çıkarımı şu oldu: **"bu framework'te loop detection var mı" yanlış soru.
Doğru soru "varsayılanda açık mı".**

## [SLAYT 8] — İkinci bulgu: dönüşümlü döngü

> Çoğu dedektör yalnızca **ardışık aynı çağrıyı** sayıyor
>
> `A → A → A` ✅ yakalanıyor
> `A → B → A → B → A → B` ❌ **hiçbir çağrı ardışık tekrarlamıyor**
>
> İlk 22'lik taramada bunu yakalayan: **2 tane** (Gemini CLI, OpenHands)

## [SÖYLE]

İkinci bulgu bir kör nokta.

Dedektörlerin çoğu şöyle çalışıyor: son çağrıyla bir öncekini karşılaştır, aynıysa sayacı
artır, farklıysa sıfırla. Bu `A A A` desenini yakalar.

Ama ajanlar çoğu zaman böyle sıkışmıyor. Şöyle sıkışıyor: dosyayı oku, düzenle, oku,
düzenle, oku, düzenle. Ya da üç adımlı bir çevrimde dönüyor. Bu dizide **hiçbir çağrı bir
öncekiyle aynı değil** — sayaç her adımda sıfırlanıyor, dedektör hiç konuşmuyor.

İlk taradığım 22 harness'tan bunu yakalayan sadece **iki tanesi** vardı: Gemini CLI, k'yı
beşe kadar tarayarak; ve OpenHands, k=2 için özel bir desen yazarak. Sonradan eklediğim
harness'larla birlikte listeye OpenClaw'ın `pingPong` dedektörü ve Strands'in çeşitlilik
kuralı da girdi.

Bu bulgu doğrudan koda döndü — bizim dedektörümüzde k=1'i özel durum yapmadım, genel çevrim
taramasının bir hâli olarak yazdım. Buna demoda geleceğim.

---
---

# BÖLÜM 3 · Metodolojiler — asıl bölüm (10 dk)

## [SLAYT 9] — On yedi zihniyet, altı seviye

> Literatürden ve üretimdeki harness'lardan **17 farklı yaklaşım** çıkardım.
>
> Hepsi aynı soruya cevap veriyor ama **farklı bir teşhise** dayanıyor.
> En basitten en karmaşığa altı seviye:
>
> **1 · Sayaç** → **2 · Pencere** → **3 · Dünya** → **4 · Şekil** → **5 · Kademe** → **6 · Karar**

## [SÖYLE]

Şimdi asıl bölüm: bu problemi çözmek için ne var, ve nasıl çalışıyorlar.

Okuduğum kaynaklardan ve taradığım harness'lardan on yedi farklı yaklaşım çıktı. Bunları
"iyiden kötüye" sıralamadım, çünkü öyle bir sıra yok — her biri **farklı bir teşhise**
dayanıyor ve teşhislerin çoğu doğru. Ama farklı şeyler hakkında doğru.

Onun yerine **basitten karmaşığa** sıraladım. Ölçütüm üç soruydu: ne kadar durum tutuyor,
kaç kavram gerekiyor, sisteme nereden dokunuyor.

Ortaya altı seviye çıktı. Ve şunu fark ettim — bu sıra aynı zamanda **bir hikâye.** Her
seviye bir öncekinin çözemediği bir şeyi çözüyor. O yüzden sırayla anlatacağım, çünkü
üçüncü seviyenin neden gerektiği ancak ikincinin nerede tıkandığını görünce anlaşılıyor.

---

## [SLAYT 10] — SEVİYE 1 · Sayaç

> **"Ne yaptığını anlamaya çalışma. Kaç yaptığını say."**
>
> Bir tamsayı. Geçmiş yok, karşılaştırma yok.
>
> `steps` · `tokens` · `cost_usd` · `elapsed` · `replans`
>
> ```python
> if kullanılan >= limit:
>     return STOP
> ```
>
> **Yazması:** bir öğleden sonra · **Maliyeti:** O(1)

## [SÖYLE]

En alt basamak, ve en çok işe yarayan basamak.

Fikir şu: ne yaptığını anlamaya hiç çalışma. Sadece say. Kaç adım attı, kaç token yaktı,
kaç dolar oldu, kaç saniye geçti. Sayı limiti aştıysa kes.

Bunun gücü **kandırılamaz** olmasında. Diğer bütün sinyaller yorum gerektiriyor. "Görev
tamamlandı" modelin görüşü. "Hata oldu" hatanın ne sayıldığına bağlı. Ama adım sayısı
tartışmasız: on iki adım attıysa on iki adım atmıştır.

Arize'ın bu konudaki katkısı beş durma koşulu tanımlayıp **aralarına hiyerarşi koymak**:
adım limiti birincil, çünkü modelin söylediği hiçbir şeye bağlı değil.

İkinci ve az fark edilen katkısı da şu — **durma sebebini kaydetmek.**

## [SLAYT 11] — Durma sebebini kaydetmek

> Bir sistemin başarı oranı **%70.**
> Kalan %30 ne?
>
> Hepsi adım limitine takılıyorsa → **limitiniz dar**
> Hepsi hata veriyorsa → **araçlarınız bozuk**
> Hepsi bütçe aşıyorsa → **döngüye giriyor**
>
> Ortalama üçünü de gizler.

## [SÖYLE]

Şunu düşünün: sisteminizin başarı oranı yüzde yetmiş. Kalan yüzde otuzun **ne olduğu** her
şeyi değiştiriyor.

Hepsi adım limitine takılıyorsa limitiniz dardır, sayıyı büyütürsünüz. Hepsi hata veriyorsa
araçlarınız bozuktur, modele hiç dokunmazsınız. Hepsi bütçe aşıyorsa döngüye giriyordur.

Üç tamamen farklı problem, ve tek bir "yüzde 70" sayısı üçünü de gizliyor. O yüzden her
koşum bir terminal etiketle bitmeli: `completed`, `max_steps`, `budget_exceeded`, `timeout`,
`error`.

Bu, kod olarak beş satır. Ama teşhis kabiliyeti bakımından bu sunumdaki en yüksek getirili
şey.

## [SLAYT 12] — Seviye 1'in içindeki tartışma

> Dördü de **aynı sayacı** tutuyor.
> Fark: **sayaç dolduğunda ne oluyor?**
>
> | | Sayaç dolunca |
> |---|---|
> | `arize-control` | Kes |
> | `agentscope-grace` | 5 tur lütuf, **araç seçimi kilitli** |
> | `hermes-no-pressure` | Kes — ama **önceden asla uyarma** |
> | `claude-advisory` | Modele geri sayımı göster, **tavsiye niteliğinde** |

## [SÖYLE]

Bu seviyede dört yaklaşım var ve dördü de aynı sayacı tutuyor. Aralarındaki tek fark
**sayaç dolduğunda ne olduğu** — ve şaşırtıcı biçimde asıl tartışma tam orada.

**AgentScope** şunu diyor: sert tavan ajanı işin ortasında keser, o ana kadar yaptığı iş
çöpe gider. O yüzden limit dolunca beş turluk bir lütuf bütçesi veriyor. Ama o turlarda
**araç seçimini kilitliyor** — ajan artık yeni araç çağıramıyor, sadece cevap üretebiliyor.
Nazik bir rica değil, mekanik bir kısıt. "Lütfen bitir" demek yerine bitirmekten başka
seçenek bırakmıyor.

Sonraki ikisi de bu sunumun en öğretici kısmı — birbirinin tam tersini savunuyorlar.

## [SLAYT 13] — İki olgun sistem, zıt cevaplar

> **Claude:** *"Modele kalan bütçesini göster, kendini ayarlasın."*
>
> **Hermes:** kod yorumundan, birebir:
> *"Ara basınç uyarıları yok — modelleri karmaşık görevlerde **erken pes ettiriyordu**."*
>
> Aynı sistem **süre** ekseninde %80'de uyarı veriyor.
>
> → Sonuç: **eksene göre karar ver.** Adımda uyarma, sürede uyar.

## [SÖYLE]

Anthropic'in Task Budgets özelliği şunu yapıyor: konuşmaya bir geri sayım işareti giriyor,
model kalan bütçesini görüyor ve azaldıkça işi toparlıyor. Önemli detay — bu **tavsiye,
zorlama değil.** Model, kesilmesi bitirilmesinden daha zararlı olacak bir işin ortasındaysa
bütçeyi aşabiliyor. Amaç durdurmak değil, **öngörülebilir bir iniş.**

Hermes de aynı şeyi denemiş. Sonra kaldırmış. Kod yorumu şöyle — bunu kaynak koddan
okudum, birebir yazıyorum:

> *"Ara basınç uyarıları yok — modelleri karmaşık görevlerde erken pes ettiriyordu."*

Yani "adımların azalıyor" mesajı modele "bu görev bana göre değil" gibi geliyor ve
gerçekten yapabileceği işi bırakıyor.

Ama aynı sistem **süre** bütçesinde yüzde sekseninde uyarı veriyor. Yani eksene göre farklı
karar vermişler.

Bir tahminim var, kanıtım yok: "sürenin azalıyor" mesajı "elindekiyle topla" diye
okunuyor; "adımın azalıyor" ise "yeterince iyi değilsin" diye.

Bizim aldığımız ders "hangisi haklı" değil. **Eksene göre karar vermek.** Adım ekseninde
uyarma, süre ekseninde uyar. Bunu koda böyle geçirdim.

---

## [SLAYT 14] — SEVİYE 2 · Pencere

> Seviye 1 "**çok** oldu" diyebiliyor.
> "**Aynı şey** oluyor" diyemiyor.
>
> Bunun için son N adımın geçmişini tutmak gerekiyor.
>
> Ve hemen bir soruya cevap vermek gerekiyor:
> ## "Aynı şey" ne demek?

## [SÖYLE]

Sayaç bir şeyi söyleyemiyor: "çok oldu" diyebiliyor ama "aynı şey oluyor" diyemiyor.

Onu söyleyebilmek için **geçmiş tutmak** gerekiyor — son yirmi adım, son kırk olay. Bedeli
de hemen ortaya çıkıyor: artık bir soruya cevap vermek zorundasınız. **"Aynı şey" ne
demek?**

Bu seviyenin bütün zorluğu o tanımda.

## [SLAYT 15] — Sinsi tuzak

> ```
> click(x=100, y=200)  id=call_a1f  ts=10:32:01
> click(x=100, y=200)  id=call_b7c  ts=10:32:04
> ```
> Aynı eylem. **Farklı imza.**
>
> Dedektör çalışır · hata vermez · log basar
> ## ve hiçbir zaman hiçbir şey bulmaz

## [SÖYLE]

Şu tuzağı özellikle anlatmak istiyorum, çünkü hem sinsi hem de ben de düştüm.

Her araç çağrısının bir kimliği var, bir zaman damgası var, bir istek numarası var. Ve
bunlar **her seferinde değişiyor.** Bu alanları parmak izine katarsanız, iki tamamen özdeş
çağrı hiçbir zaman eşit çıkmaz.

Sonuç: dedektörünüz çalışır. Hata vermez. Log basar. **Ve hiçbir zaman hiçbir şey bulmaz.**

Bu, hata türlerinin en kötüsü — çünkü sistem çalışıyor gibi görünür. Testleriniz geçer,
dashboard'unuz yeşildir, kimse fark etmez.

OpenHands bunu şöyle çözüyor: karşılaştırma yaparken kimlik alanlarını **bilerek** atıyor,
yalnızca içeriğe bakıyor.

Ama tersi de tuzak. Atmayı abartırsanız bu sefer gerçek argümanları da atmış olursunuz. O
zaman "aynı aracı elli farklı dosyada çağırmak" da döngü sayılır ve **meşru bir toplu
işlemi** kesersiniz.

Doğru çizgi şurada: **her turda zaten değişen şeyleri at, işin anlamını taşıyan şeyleri
tut.**

## [SLAYT 16] — Aynı problemin üç farklı çözümü

> **Strands — çeşitliliği ölç**
> Tekrarı tanımlamaktan kaç. Sor: *son N adımda kaç FARKLI şey oldu?*
> `len(set(pencere)) < eşik` → tek satır, bütün desenler
>
> **OpenHands — beş desen**
> eylem-gözlem (4) · eylem-hata (3) · monolog (3) · A-B-A-B (6) · context döngüsü
> `STUCK` = **ayrı terminal durum** (başarı da değil, hata da değil)
>
> **loopguard — sonuca bak, eyleme değil**
> *"Ajan sürekli hareket halindedir ancak sistem ilerlemiyordur."*

## [SÖYLE]

Aynı problemin üç farklı çözümü var, ve üçü de öğretici.

**Strands** soruyu tersten soruyor — bence en zarif olanı. Tekrarın tanımıyla boğuşmak
yerine şunu soruyor: *son N adımda kaç **farklı** şey oldu?*

Cevap düşükse ajan dar bir alanda dönüyordur. Aynı şeyi mi tekrarlıyor, iki şey arasında mı
gidip geliyor, üç adımlık çevrimde mi — hiç fark etmez, hepsi düşük çeşitlilik üretir.
Kodu da tek satır: pencereyi kümeye at, boyutuna bak. Az önceki A-B-A-B kör noktası bu tek
kuralla kapanıyor.

Bedeli var tabii: meşru olarak dar alanda çalışan işler de düşük çeşitlilik gösterir. Bir
dosyayı defalarca düzenleyip test eden ajan aslında ilerliyordur.

**OpenHands** tam tersini yapıyor — beş ayrı desen tanımlıyor, her birine ayrı eşik veriyor.
Daha çok kod, ama karşılığında **hangi desenin** tetiklendiğini söyleyebiliyor. Ve bir şey
daha yapıyor ki bunu ben de aldım: `STUCK`'ı **ayrı bir terminal durum** yapıyor. Ne
başarı, ne hata. Üçüncü bir sonuç.

**Üçüncüsü** tekrar saymanın kör noktasını kapatıyor. Şöyle bir ajan düşünün: her turda
gerçekten farklı bir şey deniyor. Farklı araç, farklı argüman. Hiçbir tekrar dedektörü
tetiklenmiyor, çünkü hiçbir şey tekrarlanmıyor. Ama hiçbiri de işe yaramıyor.

Modexa'nın yazısındaki cümle: *"Ajan sürekli hareket halindedir ancak sistem ilerlemiyordur.
Hareket etmek, ilerlemek demek değildir."*

Çözüm eyleme değil **sonuca** bakmak. Dünyanın durumunun bir özetini çıkar, her adımda
karşılaştır. Bunun computer use'daki karşılığı doğrudan: eylem öncesi ekranı al, sonrası
tekrar al. Değişmediyse o tıklama hiçbir şey yapmamış.

Burada da bir incelik var: bazı eylemler **meşru olarak** hiçbir şey değiştirmez. `wait`
ekranı değiştirmez ve değiştirmemesi normaldir. Ekran görüntüsü almak da öyle. Bu ayrımı
yapmazsanız modelin meşru beklemesini döngü sanarsınız.

---

## [SLAYT 17] — SEVİYE 3 · Dünya

> İki seviyedir ajanın **davranışına** bakıyorduk.
> Bu seviye **kanıta** geçiyor.
>
> **Ajanın en tehlikeli hatası döngüye girmek değil —**
> ## yanlış işi doğru sanmak.

## [SÖYLE]

Üçüncü seviye bir kavram değişikliği yapıyor.

İki seviyedir ajanın davranışına bakıyorduk: kaç adım attı, ne tekrarladı. Bu seviye
davranışı bırakıp **kanıta** geçiyor. Test geçti mi, aracın hata oranı ne, fatura kaç
dolar.

Çünkü şunu fark etmişler: ajanın en tehlikeli hatası döngüye girmek değil. Döngü en azından
gürültülü — fatura kabarır, birileri fark eder. Asıl tehlikeli olan **yanlış işi doğru
sanmak.** O sessiz.

## [SLAYT 18] — "Bitirdim" bir istektir, kanıt değil

> *"'Bitirdim dedi', ajan dünyasının 'benim makinemde derleniyor'udur."*
>
> Ajan "bitirdim" dediğinde bu bir **bilgi** değil, bir **talep**: *"durmak istiyorum."*
>
> → Bitirme iddiasını bir **kapıya** bağla
> → Kapı açılmazsa **koşumu bitirme** — doğrulama sonucunu gözleme geri ver, döngü devam etsin
>
> **Kör nokta:** ajan hiç "bitirdim" demezse bu kapı hiç açılmaz.

## [SÖYLE]

Ajan işini bitirdiğini söylediğinde bu bir bilgi değil, bir **taleptir**: "durmak
istiyorum." Bunu doğrudan kabul etmek, öğrencinin kendi sınav kâğıdını okuması gibi.

Loop Engineering yazısındaki benzetme çok yerinde: *"'Bitirdim dedi', ajan dünyasının 'benim
makinemde derleniyor'udur."*

Çözüm bitirme iddiasını bir kapıya bağlamak. Testler geçti mi? Dosya gerçekten yazıldı mı?
Ekranda beklenen şey var mı?

Ve şu detay önemli: kapı açılmazsa **koşum bitirilmiyor.** Doğrulama sonucu ajanın gözlem
akışına geri veriliyor ve döngü devam ediyor. Ajan kendi hatasını görüp düzeltiyor.

Gücü doğrulamanın **ortama** dayanması — modelin görüşü değil, dünyanın hâli.

Kör noktası da aynı yerden: ajan hiç "bitirdim" demezse bu kapı hiç açılmaz. Sonsuza kadar
dönen bir ajanı yakalamaz. **Mutlaka bir bütçe stratejisiyle birlikte kullanılmalı** — bu,
katmanların neden gerektiğinin en net örneği.

## [SLAYT 19] — Ölçülmüş bir sonuç: onarım mesajı

> Kontrol düştü, ajanı son sağlam noktaya geri sardın. **Ona ne söylersin?**
>
> | Ne söylendi | Kurtarma |
> |---|---|
> | Hiçbir şey, sadece yeniden dene | %16 |
> | Genel "tekrar kontrol et" uyarısı | %36 |
> | **Doğru cevabı doğrudan ver** | %36 |
> | **Hangi kontrolün düştüğünü söyle** | **%45** |
>
> Ve: **deterministik kontroller %96 yakalama / 0 yanlış pozitif**
> İstatistiksel davranış monitörü: %54 / %17 yanlış pozitif

## [SÖYLE]

Bu seviyeden bir ölçüm göstermek istiyorum, çünkü hem sezgiye aykırı hem de doğrudan bizim
tasarımımıza girdi.

Senaryo şu: bir doğrulama düştü, ajanı son sağlam kontrol noktasına geri sardın, tekrar
çalıştıracaksın. **Ona ne söyleyeceksin?**

Dört seçenek denemişler. Hiçbir şey söylememek yüzde 16 kurtarıyor. Genel bir "tekrar
kontrol et" uyarısı yüzde 36.

**Doğru cevabı doğrudan vermek** de yüzde 36. Yani ajana cevabı söylemek, genel bir uyarıyla
aynı.

En iyisi: **hangi kontrolün düştüğünü söylemek** — yüzde 45.

Bu ilk bakışta tuhaf: cevabı verdiğin halde neden daha kötü? Mantığı şu — cevabı
verdiğinizde ajan onu kopyalar, **neden** yanıldığını anlamaz. Bir sonraki adımda aynı
hatayı yapar. Kuralı söylediğinizde kendi hatasını bulur.

Bunun bizim koda doğrudan sonucu var: bir dedektör tetiklendiğinde ajana genel bir öğüt
vermek yerine **tetiklenen dedektörün adını** söylüyoruz.

Ve alttaki satır bence bu çalışmanın en değerli bulgusu. Üç tane basit deterministik kural
— toplam tutarlılığı, gerekli araç kapsaması, araç sözleşmesi — hataların yüzde 96'sını
yakalıyor, **sıfır yanlış pozitifle.** Aynı hatalar için eğitilmiş istatistiksel bir
davranış monitörü yüzde 54 yakalıyor ve yüzde 17 yanlış alarm veriyor.

Yani basit kural, karmaşık modeli hem yakalamada hem yanlış alarmda yeniyor. Bu, bütün
çalışmanın belki en pratik dersi.

## [SLAYT 20] — Suç ajanda olmayabilir

> Ajan aracı çağırır → hata → tekrar → hata → **üçüncüde başarır**
> Kullanıcı doğru cevabı alır. Koşum **"başarılı"** biter. Kimse ticket açmaz.
>
> *"Bu, hakkında hiç ticket açılmayan türden bir hata. Sessiz ama öldürücü."*
>
> Tekrar saymak işe yaramaz — koşum başarılı bitiyor.
> Bakılacak şey: **hata ORANI**, araç başına.
>
> *"Ajanlar doğru davrandı. Zafiyet veri çekme katmanında."*

## [SÖYLE]

Bu seviyeden son bir şey, çünkü bir varsayımı kırıyor.

Vaka şu: ajan bir aracı çağırıyor, hata alıyor, tekrar deniyor, hata alıyor, üçüncüde
başarıyor. Kullanıcı doğru cevabı alıyor. Koşum **başarılı** biter. Hiç kimse ticket açmaz.

Ama iki çağrı boşa gitti. Ve daha kötüsü: bugün geçici olan bu hata yarın kalıcı olursa,
her istek bütün deneme hakkını yakar ve alt akışı tıkar.

Bu deseni yakalamak için tekrar saymak işe yaramıyor — koşum başarıyla bitiyor. Bakılması
gereken şey **hata oranı**: bir aracın son çağrılarının yüzde kaçı hata verdi. Ve bu oran
**araç başına** tutulmalı, ajan başına değil.

Kaynaktaki teşhis cümlesi şu: *"Ajanlar doğru davrandı. Zafiyet veri çekme katmanında."*

Ders: bir guardrail her zaman ajanı suçlamamalı.

---

## [SLAYT 21] — SEVİYE 4 · Şekil

> Üç seviyedir döngüyü **tespit** etmeye çalışıyoruz.
> ## Neden oluşmasına izin veriyoruz ki?
>
> **modexa** — durum makinesi: ANLA → TOPLA → EYLEM → DOĞRULA → YANIT → DEVRET
> yalnız izinli geçişler. "Doğrula"dan "topla"ya dönülemiyorsa o döngü **hiç oluşmuyor**
>
> **AutoGen** — graf kurulurken doğrula:
> `Cycle detected without exit condition` → sistem **hiç başlamıyor**
> Çalışma zamanında **sıfır maliyet, sıfır yanlış pozitif** — çünkü çalışma zamanı hiç gelmedi

## [SÖYLE]

Dördüncü seviye tespit işini tamamen bırakıp mimariye geçiyor. Sorusu şu: üç seviyedir
döngüyü yakalamaya çalışıyoruz — neden oluşmasına izin veriyoruz ki?

**Modexa'nın** cevabı: döngüyü bir durum makinesine indirge ve yalnızca belirli geçişlere
izin ver. Ajan "doğrula" durumundan "topla"ya geri dönemiyorsa, o döngü hiç oluşmuyor.
Tespit etmeye gerek kalmıyor çünkü mümkün değil.

Yanına bir de **geri dönüş merdiveni** koyuyor: ajan kendi kendine tekrar deneme icat
etmiyor, sabit bir merdiveni tırmanıyor — bekleyip bir kez dene, aracı değiştir, kapsamı
daralt, kullanıcıya sor, elindekiyle cevap ver.

**AutoGen** en uç noktada. Ajan akışının grafiği kurulurken doğrulanıyor: bir çevrim var ve
çıkış koşulu yoksa sistem hiç başlamıyor, hata veriyor. Hata mesajı birebir şöyle: `Cycle
detected without exit condition`.

Çalışma zamanında sıfır maliyet, sıfır yanlış pozitif — çünkü çalışma zamanı hiç gelmedi.

Sınırı da net: yalnızca **yapısal** döngüleri yakalıyor. Grafiğin şekli kusursuz olabilir ve
model yine de aynı düğümde takılabilir. Bu, kodu derlemekle çalıştırmak arasındaki fark
gibi — derleyici tip hatalarını yakalar, sonsuz döngüleri yakalamaz.

Bu seviyenin bedeli de esneklik: mevcut bir ajana **sonradan eklenemiyor.** Sistemi baştan
öyle kurmuş olmanız gerekiyor.

---

## [SLAYT 22] — SEVİYE 5 · Kademe

> Buraya kadar her zihniyetin **bir sinyali** ve **bir tepkisi** vardı: tetiklendi → durdu.
>
> Bu seviye ikisini de çoğaltıyor:
> **Altı sinyal** aynı anda · tepki bir **merdiven**
>
> `pi-signature`:
> birebir aynı çağrı · aynı aracın ardışık hatası · birebir aynı metin ·
> tek mesaj içinde tekrarlanan cümle · **yakın-benzer ardışık metin** · yakın-benzer çevrim
>
> Metin döngüsü → **yönlendir** → kes
> Araç çağrısı → **engelle** → sebebini söyle → turu bitir

## [SÖYLE]

Beşinci seviye. Buraya kadarki her zihniyetin bir sinyali ve bir tepkisi vardı: tetiklendi,
durdu. Bu seviye ikisini de çoğaltıyor.

En iyi örneği `pi-signature`. Ve varlık sebebi ikinci seviyenin kapatamadığı bir boşluk:
**ajan aynı şeyi farklı kelimelerle söylüyor.**

Ajan `ls -la` yazıyor, sonra `ls -al`, sonra `ls -a -l`. Üçü de aynı şey. Ama parmak izleri
farklı, hiçbir imza dedektörü tetiklenmiyor.

Metin tarafında daha da yaygın. Ajan aynı cümleyi kurmuyor ama aynı fikri yeniden ifade
ediyor: "Bir daha kontrol edeyim" → "Tekrar doğrulamam lazım" → "Bunu bir kez daha teyit
edeyim."

Getirdiği çözüm zekice ve **ucuz**: ardışık iki mesajın kelime örtüşme oranına bakıyor.
Yüzde 55'ten fazla ortak kelime taşıyorlarsa model aynı adımı yeniden ifade ediyor demektir.

Bunun güzelliği maliyetinde. Anlamsal benzerlik denince akla gömme modelleri gelir — her
adımda bir model çalıştırmak, gecikme, para, ayrı bir servis. Burada yapılan iş sadece
**kelime saymak.** Deterministik, anında, bedava.

Ama bu tek bir sinyal. Yanına beş tane daha koyuyor, altısının ayrı eşiği var. Ve tepki de
kademeli: metin döngülerinde önce yönlendir sonra kes; araç çağrılarında önce engelle,
sonra sebebini söyle, tekrarlarsa turu bitir.

Bu seviyenin karmaşıklığı burada patlıyor: **durum makinesi artık dedektörün içinde.**
Altı sinyal, ayrı eşikler, iki farklı müdahale merdiveni, engellenen çağrının pencereye
girmemesi, her kullanıcı girdisinde sayaçların sıfırlanması. Doğru uygulamak için hepsini
akılda tutmanız gerekiyor.

Ve yanlış uygularsanız ne olur? **Seviye 2'deki o en kötü hata türü** — sessizce hiçbir şey
bulmayan bir dedektör.

---

## [SLAYT 23] — SEVİYE 6 · Karar

> Son iki zihniyet, on beşinin sorduğu soruyu bırakıyor.
>
> **`voi-allocation`:** *"Ne zaman dur"* değil → **"Parayı nereye harca?"**
> **`improvement-loop`:** *"Eşik ne olsun"* değil → **"Bu eşik nereden geldi?"**

## [SÖYLE]

Son seviye, ve iki zihniyet var. İkisi de on beşinin sorduğu soruyu bırakıyor.

**Birincisi**, bu çalışmadaki en farklı düşünen yaklaşım. Diğer bütün stratejiler "ne zaman
dur" diye soruyor. Bu strateji "parayı nereye harca" diye soruyor.

Fark şurada: bir tavan koyduğunuzda ajan tavana kadar istediğini yapar, sonra kesilir.
Tavana **nasıl** geldiği umurunuzda değildir. Oysa aynı bütçeyle çok daha iyi bir sonuç
alınabilirdi — eğer parayı doğru yerlere harcasaydı.

## [SLAYT 24] — Tavan mı, tahsis mi

> Her adımda eylemleri **birim bütçe başına faydaya** göre puanla.
>
> **Çift bütçe:** araç çağrısı **ve** token. Ajan birinde ekonomik olup diğerinde tükenebilir.
>
> Bütçe baskısı — **en kritik eksene** göre:
> ```
> ρ = 1 − min( kalan_araç/B_araç , kalan_token/B_token )
> ```
>
> **Ablasyon:** bütçe cezası çıkarıldığında F1 **0,63 → 0,43**
> **Süre:** 20,91 sn → **15,23 sn** (%27,2 düşüş) — ek hesap katmanına *rağmen*

## [SÖYLE]

Nasıl çalıştığını anlatayım, çünkü mekanizması öğretici.

Her adımda ajanın seçenekleri **birim bütçe başına faydaya** göre puanlanıyor. Ham faydaya
göre değil — pahalı bir eylem biraz daha faydalı olabilir ama ucuz bir eylem neredeyse
bedavadır. Oranı yüksek olan kazanır.

İki bütçe birden takip ediliyor: araç çağrısı sayısı **ve** token. Bu ayrım önemli, çünkü
ajan bir eksende çok ekonomik olup diğerinde tükenebilir. Tek sayaçlı bir sistem bunu
göremez.

Ve bütçe baskısı hesaplanırken **ortalama değil, minimum** alınıyor — yani en dar eksen. Üç
aramasının üçünü de harcamış bir ajan, token'ı bol diye rahat değildir.

Baskı arttıkça yeni arama pahalılaşıyor, cevap vermek çekici hale geliyor.

**Ölçülmüş sonuç şu:** bu bütçe cezası bileşenini çıkardıklarında performans her veri
setinde düşüyor. Bir veri setinde F1 0,63'ten 0,43'e iniyor. Yani makalenin asıl teknik
sonucu "puanlama kullandık" değil — **kalan bütçeyi eylem seçimine açıkça katmanın** kritik
olduğu.

İkinci sayı da "kontrol koymak yavaşlatır" itirazının cevabı. Ek bir hesaplama katmanı
olmasına **rağmen** ortalama süre 20,91 saniyeden 15,23'e iniyor, yüzde 27 düşüş. Sebebi
basit: bir aramanın ağ gecikmesi, puanlamanın maliyetinden onlarca kat büyük. Bir gereksiz
aramayı önlemek, bütün puanlama masrafını karşılıyor.

**Genel ders:** bir guardrail'in maliyeti, engellediği işin maliyetiyle karşılaştırılmalı —
mutlak olarak değil.

## [SLAYT 25] — Ve saf optimizasyonun tehlikesi

> Bütçe azalınca **"cevap ver"** en ucuz eylem → aşırı çekici hale gelir
> → ajan aceleyle **kötü bir cevaba** kaçar
> → sistem *"bütçeye uydum, hızlı bitirdim"* diye kendini ödüllendirir
>
> Üstte **deterministik guard'lar**:
> zayıf kanıtla erken cevap **engellenir** · bileşimsel soruda en az bir arama **zorunlu**

## [SÖYLE]

Ve buna bir uyarı eklemek istiyorum, çünkü guardrail tasarımının genel bir tuzağı.

"Cevap ver" eylemi her zaman en ucuz eylemdir. Yani bütçe baskısı arttıkça **otomatik
olarak** kazanır — cevabın doğru olup olmadığından bağımsız olarak.

Sonuç: ajan aceleyle kötü bir cevaba kaçar. Ve sistem bunu **başarı** olarak kaydeder,
çünkü bütçeye uymuş ve hızlı bitirmiş. Metrik yeşil, cevap uydurma.

O yüzden üstte deterministik kurallar var: kanıt zayıfken erken cevap engelleniyor,
bileşimsel bir soruda en az bir arama zorunlu tutuluyor.

**Ders:** optimizasyon hedefiyle doğruluk hedefi çakışabilir. Ucuz eylem her zaman iyi eylem
değil.

## [SLAYT 26] — Son zihniyet: eşiği tahmin etme, ölç

> **"Adım limiti 12"** — o 12 nereden geldi?
>
> *"Meşru işi kesecek kadar dar bir adım limiti, modelin kötüleştiği gibi görünen
> **sessiz bir kalite gerilemesine** dönüşür."*
>
> → Başarılı koşumların adım **dağılımına** bak, tavanı kuyruğunun üstüne koy (p99)
> → Sonra izle: koşumların yüzde kaçı **limitte** sonlanıyor?
> → Eşikler prompt'la birlikte **sürümlensin**, terfi kapısından geçsin
>
> **Eşik değiştirmek, kod değiştirmekle aynı süreçten geçmeli.**

## [SÖYLE]

Ve on yedinci zihniyet, koşumun tamamen dışında duruyor. Hiç müdahale etmiyor, sadece
kaydediyor. Çünkü sorduğu soru farklı: **bu eşikleri kim, neye bakarak koydu?**

Bugün bu sunumda çok sayı söyledim — dört, üç, on iki, yirmi. "Adım limiti 12" diye
yazdığınızda o 12 nereden geliyor? Çoğu zaman hiçbir yerden. Birinin makul bulduğu bir sayı.

Ve yanlış seçilmiş bir limit **sinsi** bir hasar veriyor. Arize'ın uyarısı şu:

> *"Meşru işi kesecek kadar dar bir adım limiti, modelin kötüleştiği gibi görünen sessiz bir
> kalite gerilemesine dönüşür."*

Model aynıdır. Limitiniz dardır. Ama grafikte **model kötüleşmiş görünür.** Haftalarca
yanlış yerde arama yaparsınız.

Çözüm ölçmek: başarılı koşumların adım dağılımına bak, tavanı kuyruğunun üstüne koy —
kaynaklardan biri p99'dan başlamayı öneriyor. Sonra izle: koşumların yüzde kaçı limitte
sonlanıyor? Bu oran tırmanıyorsa görev zorluğu ya da araç güvenilirliği değişmiştir.

Disiplin tarafı da şu — döngü ayarları prompt'la birlikte sürümlenmeli ve bir terfi
kapısından geçmeli. **Eşik değiştirmek, kod değiştirmekle aynı süreçten geçmeli.**

Bunu en sona koydum ama uygulamada **ikinci sıraya** koyardım. Buna kapanışta döneceğim.

---
---

# BÖLÜM 4 · PoC (5 dk)

## [SLAYT 27] — Ne yaptım

> `cua_lab/` — computer-use ajanı + takılabilir güvenilirlik stratejileri
>
> 17 zihniyetin tamamı aynı `Strategy` protokolüne uyuyor (8 kanca)
> Bağımlılık **yok** · API anahtarı **yok** · Python 3.10+ · 23 test
>
> Ortam bozuk, model değil: `dead_button` · `flaky` · `silent_success` · `healthy`

## [SÖYLE]

Araştırmayı okuduktan sonra bunları çalışır hale getirdim.

`cua_lab` bir computer-use ajanı — ekran görüntüsü alıp tıklayan, yazan bir döngü. On yedi
zihniyetin tamamı aynı protokole uyuyor, sekiz kancalı bir arayüz. Koşum anında hangisini
istersen seçiyorsun, ya da birkaçını üst üste koyuyorsun.

Bağımlılığı yok, API anahtarı istemiyor. Yirmi üç test var.

Bir tasarım kararını söylemek istiyorum: **hata modelde değil, ortamda.** Sahte bir masaüstü
yazdım ve içine bozuk senaryolar koydum — tıklandığında hiçbir şey yapmayan bir buton, iki
kez hata verip üçüncüde çalışan bir alan, ve sessizce başarısız olan bir kaydetme.

Sebebi şu: modeli döngüye sokmak için betikle zorlarsanız kendi kurgunuzu test etmiş
olursunuz. Ortamı bozarsanız, ajan **gerçekten** sıkışır.

## [SLAYT 28] — Ölçülen fark

> Aynı görev, aynı bozuk ortam (`dead_button`):
>
> ```
> strateji            durum       sebep          adım   token       $
> none                CEILING     hard_ceiling    300  105000   1.050
> openhands-stuck     STUCK       cycle_k2          5    1750   0.018
> ```
>
> **Sağlıklı koşumda ve meşru retry'da iki sütun BİREBİR aynı.**

## [SÖYLE]

İşte ölçüm. Aynı görev, aynı bozuk ortam.

Kontrol yokken ajan 300 adımda sert tavana çarpıyor, 105 bin token, bir dolar. Kontrol
varken **beş adımda** duruyor — 1750 token, iki sent.

Ama asıl göstermek istediğim alt satır.

Sağlıklı bir koşumda ve **meşru retry'da** iki sütun birebir aynı. Yani guardrail çalışan
bir koşuma **tek token bindirmiyor** ve meşru tekrar denemeyi kesmiyor.

Bunun neden önemli olduğunu söyleyeyim: yalnızca yakaladıklarını gösteren bir demo,
dedektörün **yanlış pozitif** oranı hakkında hiçbir şey söylemez. "Her koşumu döngü sayan"
bir dedektör de o demoyu geçerdi.

Bir dedektörün ikinci sınavı, yakalamaması gerekeni **rahat bırakmaktır.** O yüzden kontrol
senaryolarını yakalama senaryolarıyla birlikte yazdım.

## [SLAYT 29] — PoC'nin kendi öğrettikleri

> **1 · İki mekanizma birbirini maskeliyor.**
> `max_replans=3` retry döngüsünü dedektörden **önce** yakaladı → koşum doğru durdu,
> **teşhis kayboldu.** Bütçe *bir şeyin*, döngü tespiti *neyin* yanlış gittiğini söyler.
>
> **2 · Gevşek sinyal sıkıyı gizliyor.**
> "İlerleme yok" en gevşek sinyal → **eşiği en yüksek, en son tetiklenmeli.**
>
> **3 · Kendi kontrol testim kendi hatamı buldu.**
> Sağlıklı senaryoda dedektör tetikleniyordu — dedektör haklıydı, **benim betiğim döngüdeydi.**

## [SÖYLE]

Üç şeyi kod yazarken öğrendim, okuyarak değil.

**Birincisi:** ilk koşumda retry senaryosu bütçeyle durdu, döngü dedektörüyle değil.
Bütçedeki replan limiti çok darmış, dedektör konuşmaya fırsat bulamamış.

Koşum **doğru** duruyordu. Ama teşhis kayboluyordu: "çok fazla replan" bilgisini alıyorsun,
"aynı çağrı aynı hatayla dönüyor" bilgisini alamıyorsun.

Bu, iki mekanizmanın da neden gerektiğinin cevabı. **Bütçe bir şeyin yanlış gittiğini
söyler; döngü tespiti neyin.** Biri diğerinin yerine geçmiyor.

**İkincisi:** "ilerleme yok" eşiği düşükken, dönüşümlü döngü "ilerleme yok" diye
raporlanıyordu. Doğru ama işe yaramaz bir teşhis.

İlerleme yokluğu en **gevşek** sinyal — uzun bir kurulum evresi de ilerlemesiz görünür. O
yüzden eşiği en yüksek olan o olmalı ve en son tetiklenmeli. Dedektörleri sıralarken
"hangisi daha kesin konuşuyor" diye sormak gerekiyormuş.

**Üçüncüsü:** sağlıklı senaryoda dedektörüm tetiklendi. Yanlış pozitif sandım, dedektörü
gevşetecektim. Baktım ki dedektör haklı — **benim yazdığım test betiği** görev bittikten
sonra tıklamaya devam ediyordu.

Yani kendi kontrol testim kendi hatamı yakaladı. Kontrol senaryosu yazmasaydım gerçek bir
dedektörü sahte bir sebeple bozacaktım.

---
---

# BÖLÜM 5 · Değerlendirme (4 dk)

## [SLAYT 30] — Avantajlar

> **Ölçülebilir ve büyük:** 300 adım → 5 adım, 1,05 $ → 0,018 $ (aynı bozuk ortam)
> **Bedava:** sağlıklı koşuma **sıfır** ek token; deterministik kontroller O(1)–O(N)
> **Basit olan kazanıyor:** 3 deterministik kural %96 / 0 FP · istatistiksel monitör %54 / %17
> **Yavaşlatmıyor:** tahsis kontrolcüsü süreyi %27,2 **düşürüyor** (önlenen ağ çağrıları)
> **Teşhis veriyor:** durma sebebi kaydı, "%70 başarı"yı eyleme çevrilebilir hale getiriyor

## [SÖYLE]

Avantajlar tarafı — hepsi ölçülmüş, hiçbiri iddia değil.

Fark **büyük**: aynı bozuk ortamda 300 adım yerine 5, bir dolar yerine iki sent.

**Bedava**: sağlıklı koşuma sıfır ek token biniyor. Deterministik kontrollerin maliyeti
sabit ya da pencere boyutuyla doğrusal — mikrosaniye mertebesinde.

**Basit olan kazanıyor**: üç deterministik kural, eğitilmiş istatistiksel bir monitörü hem
yakalamada hem yanlış alarmda yeniyor. Bu, "önce en basitini yap" için ölçülmüş bir
gerekçe.

**Yavaşlatmıyor** — hatta hızlandırıyor: engellenen gereksiz araç çağrılarının ağ gecikmesi,
kontrol katmanının hesap maliyetinden onlarca kat büyük.

Ve en çok değer verdiğim: **teşhis veriyor.** Durma sebebi kaydı, "yüzde 70 başarı" gibi ölü
bir sayıyı eyleme çevrilebilir bir şeye dönüştürüyor.

## [SLAYT 31] — Dezavantajlar — dürüst liste

> **Yanlış pozitif gerçek bir maliyet.** Meşru toplu işlem "döngü" sayılabilir. Guardrail'in
> kendi zararı var ve ölçülmeli.
> **Sessizce ölebiliyorlar.** İmzaya oynak alan karışırsa dedektör çalışır, log basar, hiçbir
> zaman hiçbir şey bulmaz. **Testleri geçer.**
> **Eşikler tahmin.** Ölçülene kadar hepsi birinin makul bulduğu sayı. Dar limit "model
> kötüleşti" gibi görünür.
> **Birbirlerini maskeliyorlar.** Sıralama ve eşik dengesi tasarım işi.
> **Kapsam sınırlı.** Doğrulama kapısı ajan hiç bitirmezse sessiz; statik analiz model içi
> takılmayı görmez; tahsis kontrolü bol bütçede eriyor.
> **Uyarı geri tepebiliyor.** Ölçülmüş: adım ekseninde uyarmak modeli erken pes ettiriyor.

## [SÖYLE]

Dezavantajlar tarafını uzun tuttum, çünkü asıl karar verilecek yer burası.

**Yanlış pozitif gerçek bir maliyet.** Meşru bir toplu işlem — aynı aracı elli dosyada
çağırmak — kolayca döngü sayılabilir. Guardrail'in kendi zararı var ve ölçülmesi gerekiyor.

**Sessizce ölebiliyorlar.** Bu bence en tehlikelisi. İmzaya oynak bir alan karışırsa dedektör
çalışır, hata vermez, log basar ve hiçbir zaman hiçbir şey bulmaz. Ve **testleri geçer** —
çünkü testler genelde "yakaladı mı" diye bakar.

**Eşikler tahmin.** Ölçülene kadar hepsi birinin makul bulduğu bir sayı. Ve dar bir limit
grafikte "model kötüleşti" gibi görünüyor.

**Birbirlerini maskeliyorlar.** PoC'de bunu bizzat gördüm. Hangi sinyalin önce
tetikleneceği bir tasarım kararı, tesadüf değil.

**Kapsamları sınırlı.** Doğrulama kapısı ajan hiç bitirmezse hiç açılmıyor. Statik analiz
model içi takılmayı görmüyor. Tahsis kontrolü bol bütçede eriyor. Hiçbiri tek başına
yetmiyor.

**Ve uyarmak geri tepebiliyor.** Bu ölçülmüş: adım ekseninde ara uyarı vermek modeli erken
pes ettiriyor.

## [SLAYT 32] — Entegrasyon için gerekenler

> **Zorunlu — bunlar olmadan hiçbiri çalışmaz**
> 1. **İterasyon başına span.** Bütün koşum tek span'sa 4–19. adımların aynı çağrı olduğunu göremezsin.
> 2. **İmza normalizasyonu.** Hangi alanlar oynak, hangileri anlam taşıyor — envanteri çıkarılmalı.
> 3. **Terminal durum ayrımı.** `completed` / `max_steps` / `budget` / `timeout` / `error` / `stuck` ayrı ayrı.
> 4. **Çok eksenli sayaç.** Adım · replan · token · süre · dolar. Tek eksen diğerlerini gizler.
> 5. **Nihai cevap payı.** Sert limit erken tetiklensin ki ajan toparlayabilsin.
>
> **Tavsiye edilen**
> 6. Kapatma anahtarı + **varsayılan AÇIK** (12/24 harness'ın hatası)
> 7. Eşikler prompt'la **birlikte sürümlensin**
> 8. Yanlış pozitif için ayrı kontrol koşumları — CI'da

## [SÖYLE]

Son slayt: bunu bir sisteme entegre ederken neye ihtiyaç var. İlk beşi **zorunlu** — bunlar
olmadan geri kalanı çalışmıyor.

**Bir: iterasyon başına span.** Arize'ın cümlesiyle — bütün koşum tek bir span'sa dört ile
on dokuzuncu adımların aynı çağrı olduğunu göremezsiniz. Loop detection her şeyden önce bir
**gözlemlenebilirlik** gereksinimi.

**İki: imza normalizasyonu.** Hangi alanlar her turda değişiyor, hangileri işin anlamını
taşıyor — bunun envanteri çıkarılmalı. Bu iş sisteme özel, kütüphane halletmiyor. Yanlış
yaparsanız sessizce çalışmayan bir dedektör elde ediyorsunuz.

**Üç: terminal durumların ayrılması.** Beş altı ayrı sonuç, tek bir başarı oranına
karışmadan.

**Dört: çok eksenli sayaç.** Görevin kalemleri zaten bunu söylüyor — adım, replan, token,
süre. Ben doları da ekledim. Tek eksen diğerlerini gizliyor.

**Beş: nihai cevap payı.** Sert limit biraz erken tetiklensin ki ajan elindekini toparlayıp
sunacak bütçe bulsun. Yoksa yapılan iş çöpe gidiyor.

Sonraki üçü tavsiye. **Altıncısı** taramanın en net dersi: kapatma anahtarı olsun ama
**varsayılan açık** olsun. Yirmi dört harness'ın on ikisi bu hatayı yapmış — mekanizmayı
yazmış, kapalı bırakmış.

---
---

# BÖLÜM 6 · Kapanış (2 dk)

## [SLAYT 33] — Nereden başlanmalı

> Sıralamayı basitten karmaşığa yaptım. Ama **uygulama sırası farklı:**
>
> ## 1 → 6 → 2
>
> **1 · Sayaç** — bir öğleden sonra. Sert durdurma + durma sebebi kaydı.
> **6 · Ölç** — izleri topla, eşikleri dağılımdan çıkar.
> **2 · Pencere** — dedektörleri **ölçülmüş eşiklerle** yaz.
>
> Eşiği ölçmeden pencere yazmak, hangi sayıyı yazacağını bilmeden yazmak demek.

## [SÖYLE]

Kapatırken tek bir öneri.

Zihniyetleri basitten karmaşığa sıraladım ama **uygulama sırası** farklı çıktı: bir, altı,
iki.

Önce **sayacı** koyun. Bir öğleden sonra sürüyor, kandırılamıyor, ve durma sebebi kaydıyla
birlikte hemen teşhis vermeye başlıyor.

Sonra **ölçün.** İzleri toplayın, başarılı koşumların dağılımına bakın, eşikleri oradan
çıkarın.

**Sonra** pencere dedektörlerini yazın — artık ölçülmüş eşiklerle.

Çünkü eşiği ölçmeden dedektör yazmak, hangi sayıyı yazacağınızı bilmeden yazmak demek. Ve o
sayıyı yanlış koyarsanız ya meşru işi kesersiniz ya da hiçbir şey yakalamazsınız — ikisini
de aylarca fark etmezsiniz.

## [SLAYT 34] — Tek cümlede

> Model döngüyü kesemez, çünkü kesme kararı için gereken bilgi **onun görüş alanında değil.**
>
> Kesme yetkisi dışarıda olmalı — ve dışarısı **ne olduğunu da söyleyebilmeli.**

## [SÖYLE]

Tek cümleye indirirsem:

Model döngüyü kesemez. Aptal olduğu için değil — kesme kararı için gereken bilginin onun
görüş alanında olmaması nedeniyle.

O yüzden kesme yetkisi dışarıda olmalı. Ve dışarısı sadece "kes" diyebilen bir şey değil,
**ne olduğunu da söyleyebilen** bir şey olmalı.

Teşekkürler. Sorular?

---
---

# EK · Beklenen sorular

**"Bunların hepsini uygulamak gerekiyor mu?"**
Hayır. Seviye 1 tek başına vakaların çoğunu durduruyor — ama *neden* durduğunu söylemiyor.
Kaç katman ekleyeceğiniz, teşhis ihtiyacınıza bağlı. Önerim 1 → 6 → 2.

**"Gerçek bir modelle test ettin mi?"**
Hayır, bu PoC betiklenmiş ve senaryo tabanlı modellerle çalışıyor. Bilinçli bir tercih: bir
kontrol mekanizmasını göstermek için girdinin **deterministik** olması gerekiyor. Gerçek LLM
ile aynı döngüler çıkar ama her koşumda farklı çıkar ve sunumda para harcar. HF Inference
API bağlantısı planın son fazında.

**"Eşikleri nereden aldın?"**
Kaynaklardan — OpenHands'in 4/3/3/6'sı, OpenClaw'ın 10/20/30'u gibi. Ama bu **tahmin
olduğunu kabul ediyorum**, ve zaten on yedinci zihniyet tam olarak bunun eleştirisi. Doğrusu
kendi koşum dağılımınızdan çıkarmak.

**"Yanlış pozitif riski ne kadar?"**
Ölçtüğüm kadarıyla: sağlıklı koşum ve meşru retry senaryolarında sıfır — ama bu iki senaryo,
gerçek çeşitlilik değil. Literatürden alınan sayı, deterministik kontroller için 0/63.
İstatistiksel yaklaşımlar için %17. Bu, deterministik kuralları tercih etme gerekçem.

**"Bu ne kadar sürer / ne kadar iş?"**
Seviye 1 bir öğleden sonra. Gözlemlenebilirlik tarafı (iterasyon başına span) muhtemelen en
büyük kalem, çünkü mevcut loglama şemasına dokunuyor. Dedektörlerin kendisi birkaç yüz
satır.

**"IAL-SCAN / statik analiz neden yok?"**
O çalışma dağıtımdan **önce** kodu okuyarak döngü arıyor — CI'a giren bir şey. Bizim
ihtiyacımız çalışma anında devreye giren kontrol. Biri diğerinin yerine geçmiyor; ondan
aldığım tek şey soru biçimi: *"ajanik bir geri besleme yolu, maliyetli bir işlemi etkili bir
sınır olmadan tekrar tekrar çalıştırabilir mi?"*

---

# EK · Sayıların kaynağı

| Sayı | Kaynak | Doğrulama |
|---|---|---|
| %15,7 · %12,4 (MAST) | *Why Do Multi-Agent LLM Systems Fail?* — 1642 iz | `[T]` tam metin |
| 63 olay · "ödemeden önce engellenmiş tek vaka yok" | *Token Budgets: An Empirical Catalog* | `[T]` kısmi |
| `while True`, sayaç yok | `anthropics/claude-quickstarts` → `computer_use_demo/loop.py` | `[K]` kod okundu |
| `maxSessionTurns ?? -1` | Gemini CLI `packages/core/src/config/config.ts:1243` | `[K]` kendim doğruladım |
| `depth > 50` + `NODE_ENV==="test"` | Continue `gui/src/redux/thunks/streamNormalInput.ts:80` | `[K]` kendim doğruladım |
| `loopDetection.enabled = false` | openclaw 2026.7.1-2, bu makinedeki kurulum | `[K]` yerel kaynak |
| `max_iterations = None` | Google ADK `LoopAgent` | `[K]` |
| 4/3/3/6, pencere 20 | OpenHands `stuck_detector.py` | `[K]` |
| "erken pes ettiriyordu" | Hermes `agent_init.py:986–991` kod yorumu | `[K]` birebir alıntı |
| %16 / %36 / %36 / %45 | *Real-Time Detection and Repair of LLM Agent Failures* | `[T]` |
| %96 / 0-FP vs %54 / %17 | aynı kaynak | `[T]` |
| F1 0,63→0,43 · 20,91→15,23 sn | *Inference-Time Budget Control for LLM Search Agents* | `[T]` |
| 300/105000/1,050 vs 5/1750/0,018 | `cua_lab` — kendi koşumum | ölçüldü, tekrarlanabilir |

> **Not:** Görev tanımındaki "Atlas'a entegrasyon gereklilikleri" maddesi, Slayt 32'de
> sisteme özel olmayan genel gereklilikler biçiminde duruyor. Atlas'a özgü eşleme bu
> çalışmanın kapsamı dışında bırakıldı.
