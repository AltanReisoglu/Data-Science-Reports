# 23 — Sunum planı: sıra, ekran, slayt

*[19-sunum-metni.md](19-sunum-metni.md) ne söyleneceğini yazıyor. Bu belge
**hangi sırayla, hangi ekranla ve hangi slaytla** sorusunu cevaplıyor — ve
19'un omurgasını bir yerinde değiştiriyor.*

---

## §0 · Neden yeni bir plan

19'un omurgası şu cümle:

> *"AutoGen bize bir motor veriyor ama kontrol düzlemi vermiyor. OpenClaw kontrol
> düzlemini çözmüş ama güven modeli bizim kurumumuz için yanlış."*

Bu cümle hâlâ doğru. Ama artık **eksik**, ve eksikliği tam olarak sunumun en
kırılgan yerinde:

> **"Neden ölü bir çerçevenin üstüne bina kuruyorsunuz?"**

AutoGen bakım modunda (son sürüm 2025-09-30, 11 ay). Microsoft halefini
ilan etti ve MAF'ı yüksek sesle pazarlıyor. Masadaki teknik biri bunu **biliyor**.
Bu soru gelmezse şanslıyız; geldiğinde hazırlıksız yakalanmak sunumun tamamını
zayıflatır — çünkü soru mimariye değil **yargımıza** dair.

Cevap elimizde ve ölçülü. Omurgaya tek bir cümle ekleniyor:

> **Motor değişecek. Biz motoru değil, motorun etrafındaki kontrol düzlemini
> kurduk — ve bunu kanıtlayan şey ekranda bir düğme.**

Kanıt [ölçüldü]: 54 modülün **17'si** AutoGen içe aktarıyor; 16.847 satırın
**4.633'ü**, yani **%27,5**. Kalan %72,5 altında hangi motorun döndüğünü
bilmiyor. Ve MAF düğmesi bunun ekrandaki hâli.

---

## §1 · Dört perde — sıra ve gerekçesi

19'da üç perde vardı. Dördüncü perde eklemiyorum; **MAF'ı 1. perdenin dönüşü
yapıyorum**, çünkü orası zaten "AutoGen'den geriye ne kalıyor" sorusunun
sorulduğu yer.

| # | Perde | Süre | Kurduğu cümle |
|---|---|---:|---|
| 0 | Açılış | 2 dk | Ne kurduk, neden buradayız |
| 1 | **Motor** — AutoGen, postane metaforu | 12 dk | Mekanizma gerçek, ve **tuzakları ölçtük** |
| 1b | **Halef** — MAF | 4 dk | Motor değişecek; biz buna göre kurduk |
| 2 | **Kuşatma** — OpenClaw | 10 dk | Kontrol düzlemi çözülmüş, güven modeli yanlış |
| 3 | **Bizde ne var** — canlı | 8 dk | Söylediğimiz her şeyin ekranda karşılığı var |
| 4 | Kapanış + karar | 3 dk | Birinci faz için onay |

**Perde 1b'nin yeri neden orası:** AutoGen perdesini "işte bir çerçeve" diye
bitirirsen dinleyici "peki bu güncel mi" diye düşünmeye başlar ve sonraki 20
dakika boyunca o soruyu taşır. Perdenin sonunda sen sorarsan, cevabı da sen
verirsin ve konu kapanır.

---

## §2 · Perde 1 — postane

Metaforun tamamı: `postane = runtime` · `adres = AgentId(type,key)` ·
`ilan panosu = publish` · `iadeli taahhütlü = send` · `posta kutusu = topic` ·
`gişe memuru = AssistantAgent` · `dosya dolabı = model_context` ·
`pencereler = tools` · `gümrük = onay kapısı` · `mesai zili = termination`.

Yedi hamle, sırasıyla:

| # | Slayt | Söylenen tek cümle |
|---|---|---|
| ① | Üç katman | "Alt kat zarfın içine bakmaz, sadece adrese bakar." |
| ② | Kimlik: iki şey | "Şube kendiliğinden açılıyor — ilk mektup düştüğü an." |
| ③ | İki iletişim biçimi | "İlan panosunda memur düşerse **hiçbir şey olmamış gibi devam edilir**." |
| ④ | AssistantAgent + tool döngüsü | "Dosya dolabını vermezsen memurun hafızası **hiç yok**, ve hata da vermiyor." |
| ⑤ | **Fan-out / fan-in** | "Amir sıraya bakıyor, boş görüyor, günü kapatıyor. Bitmiş iki iş çöpe gidiyor." |
| ⑥ | Beş takım + faturası | "%63,7 fark. Ödediğin zekâ değil, **yönlendirme özerkliği**." |
| ⑦ | Dört sessiz varsayılan | "Hiçbiri exception fırlatmadı. Sıfır döndü, boş kaldı, asılı kaldı." |

**⑤ perdenin kalbi.** Yavaş söyle, ölçümü göster, ve şunu ekle: *"Bunu okumadık,
ölçtük. Ve düzeltmek için alt kata inmek zorunda kaldık — gişede çözülmüyordu."*

**Metaforu kendin yık, ilk soru yıkmadan.** İki cümle yeter:

> *"Postane kısmı deterministik. Gişedeki kişi değil — o bir dil modeli, hangi
> pencereyi açacağına kendi karar veriyor. Onay kapısının sebebi tam olarak bu."*
>
> *"Ve mektup taşımak bedava, memur çalıştırmak değil. Her gişe işlemi faturalı."*

---

## §3 · Perde 1b — MAF, dört slayt

| # | Slayt | Ölçülmüş çekirdek |
|---|---|---|
| ① | Halef geldi | Son AutoGen sürümü 11 ay önce · MAF 4 ayda 14 sürüm |
| ② | **1'e karşı 40** | Tool döngüsü varsayılanı — iki kurulu paketten okundu |
| ③ | **Hızın faturası** | GA'dan sonra **2 ayda 15 kırıcı değişiklik**; değişiklik rehberi bile 6 sürüm geride |
| ④ | **Harness kavram oldu** | Microsoft'un örneği: *build your own claw* — finans asistanı, `valuation` + `risk-scoring` becerileri |

Ve perdenin kapanış cümlesi — sunumun en önemli 20 saniyesi:

> *"İki risk var ve ikisi de gerçek. AutoGen'in riski **donmuş** olması: bulduğun
> hatayı düzeltecek kimse yok. MAF'ın riski tersi: GA'dan sonra iki ayda on beş
> kırıcı değişiklik, ve Microsoft'un kendi değişiklik rehberi bile güncel değil."*
>
> *"Bizim cevabımız çerçeve seçmek değil. Kodun **yüzde yetmiş ikisi** altında
> hangi motorun döndüğünü bilmiyor. Birazdan bir düğmeye basacağım ve motor
> değişecek."*

**④'ün altında söylenmesi gereken:** Microsoft'un harness örneği bir yatırım
asistanı ve becerileri değerleme ile risk skorlama. *"Yani bu desen bizim
icadımız değil — satıcının kendi yol haritasında, aynı alanda."*

---

## §4 · Demo koreografisi — sekiz durak

*Bu bölüm 19 Ağustos'ta bu makinede **koşturularak** yazıldı; aşağıdaki her sayı
o koşudan geliyor. Sunum günü sayıları ekrandan oku, buradan değil.*

### Sıfırıncı kural: sıra zorunlu

`Akış ↗` düğmesi **kayıtlı bir tur yoksa kapalı**. Sunucu yeni açıldıysa
`/api/runs` boş döner ve düğmeye basamazsın. Yani **önce soru, sonra akış** —
tersi çalışmaz. Sunumdan önce bir soru sor ve düğmenin açıldığını gör.

### Durak 1 · Ekranı tanıt — 20 saniye

Slayt kapalı değil: sağdaki panelde deste açık duruyor. Bu bilinçli.

> *"Solda ajan, sağda slayt. Ekran değiştirmiyorum — söylediğim her şeyin
> karşılığı aynı pencerede."*

Göster: sağ üstteki **`AutoGen`** rozeti. *"Şu an hangi çerçevede koştuğumuz
başlıkta yazıyor. Birazdan buna basacağım."*

### Durak 2 · Soruyu sor — **ve önce `Reset chat`**

> ⚠️ **Sunumdan önce `Reset chat`.** Ölçüldü: 15 turluk bir oturumda aynı soru
> `search_docs`'u **hiç çağırmıyor** — ajan bağlamdan cevaplayıp `memory_search`
> ile idare ediyor, tur 5,8 saniyede bitiyor ve anlatacak bir şey kalmıyor.
> Kirli oturum, demoyu sessizce boşaltıyor.

Ve bu belgede daha önce yazılı olan soru **yanlıştı**. Tarayıcıyı gerçekten
sürüp ölçtüm; üçü de temiz oturumda koştu:

| soru | adım | tool çağrısı | süre |
|---|---:|---|---:|
| ~~`search_docs ile durable execution konusunda ne dediğimizi bul`~~ | **39** | `search_docs` ×3 + `memory_search` ×2 + `memory_get` | 33,1 sn |
| **`dokümanlarda workbench nedir, docs araması yap`** | **10** | `search_docs` ×1 | 17,6 sn |
| `search_docs tool'unu kullanarak dokümanlarda GraphFlow'u ara` | **10** | `search_docs` ×1 | 16,1 sn |

Eski soru **otuz dokuz adım** üretiyor. Bu belgenin Durak 3'te anlattığı on
aşamalık şerit onunla hiç çıkmıyor: ekranda altı tool çağrısı akıyor, sen on
aşama anlatıyorsun, ve dinleyici hangisinin doğru olduğunu bilmiyor.

**Kullan:**

```
dokümanlarda workbench nedir, docs araması yap
```

Tam on adım, tek `search_docs`, ~17 saniye. Yedek:
`search_docs tool'unu kullanarak dokümanlarda GraphFlow'u ara`.

**Neden soru seçimi bu kadar önemli:** model tool çağırıp çağırmayacağına
kendisi karar veriyor. "ne dediğimizi bul" cümlesi ona *hafıza* gibi
okunuyor ve `memory_search`'e uzanıyor — tool adını cümlede yazmış olsan bile.

### Durak 3 · Şerit dolarken — 10 aşama, 7,6 saniye

Renkleri **söyle**, çünkü sunumun tamamındaki ayrım ekranda renk olarak duruyor:

> *"Turuncu bizim yazdığımız. Mor `autogen_core`. Mavi `autogen_agentchat`."*

Ölçülen sıra ve zamanlar:

| +sn | şerit | aşama |
|---:|---|---|
| 0,00 | turuncu | Bağlam kuruluyor — `CompactingChatCompletionContext` |
| 0,00 | mor | Model çağrısı |
| 3,62 | mavi | **Model bir tool istedi** — `ToolCallRequestEvent` |
| 3,62 | **turuncu** | **Kapı** — `before_tool_call → GatedWorkbench` |
| 3,62 | mor | Tool koşuyor |
| 3,88 | mavi | Sonuç döndü · Döngü devam ediyor (`max_tool_iterations=6`) |
| 3,88 | mor | Model çağrısı — **ikinci kez** |
| 6,06 | mavi | Token akışı |
| 7,64 | mavi | Tur bitti |

Duracağın tek yer **dördüncü satır**:

> *"Tool çağrısı modelden çıktı ama henüz çalışmadı. Arada turuncu bir satır var —
> orası bizim kapımız. Ve bu bir davranış kuralı değil: model 'lütfen' dese de
> geçemez."*

Ve ikinci model çağrısını göster:

> *"AutoGen'in varsayılanı **bir** tur. Bu satırın var olması için o varsayılanı
> elle değiştirmemiz gerekti — yazmasaydık ajan tool sonucunu görür ve **susardı**."*

### Durak 4 · Özet satırı

Ekranda: **`2 LLM · 1 TOOL · 8143 TOKEN · 14,6 sn`**

> ⚠️ 19'daki metinde `19775 TOKEN` yazıyor — **o sayı eski**. Ekrandan oku.
> Ezberlenmiş bir sayıyı ekrandaki başka bir sayının üstüne söylemek, sunumun
> geri kalanındaki bütün rakamları şüpheli yapar.

### Durak 5 · Akış ekranı — `Akış ↗`

*Aşağıdakiler tarayıcı gerçekten sürülerek ölçüldü —
[`pipeline/tests/drive/`](../pipeline/tests/drive/README.md).*

**Zincir adım adım doluyor** ve bu izlenecek bir şey; sayfayı açar açmaz anlat:

| an | ekranda |
|---|---|
| +0,4 sn | 4 kutu sönük, **1 yanıyor** |
| +0,8 sn | bir **ok** yanıyor — mesaj geçiyor |
| +4,8 sn | hepsi parlak, **ışık sönmüş** |

> *"Işık sönerse tur bitmiştir. Yanan kutu 'şu an burada' demek — bitmiş bir
> koşuda bir şeyin hâlâ yanıyor olması, ekranın zamanı yanlış söylemesi olurdu."*


Yeni sekmede açılıyor; ikinci ekran varsa açık kalsın.

Dört şey göster, bu sırayla:

**① İki bant.** Üst bant `AJAN · AgentChat`, alt bant `GATEWAY · bizim hat`.
> *"Üstte olan şey çerçevenin işi. Altta olan şey bizim işimiz. Bu ayrım
> sunumun tamamının konusu ve burada çizili."*

**② Kenarların üstündeki mesaj türleri.** Ölçülen dördü:
`TextMessage` → `ToolCallRequestEvent` → `ToolCallExecutionEvent` → `TaskResult`
> *"Ok değil, tür. Hangi sınıfın hangi yönde gittiği yazıyor."*

**③ Sekiz desen — ve hepsinin `kullanılmadı` demesi.** Bu şaşırtıcı görünür,
açıkla:
> *"Resmî sekiz desenden **hiçbiri** koşmadı, ve ekran bunu her biri için ayrı
> gerekçeyle söylüyor: 'bu turda tek ajan vardı, dallanma yok.' Bu tur bir tool
> döngüsü — desen değil. Sistemin kendi hakkında yanlış konuşmamasını istedik."*

**④ Sekiz bileşen.** `AssistantAgent` · `CompactingChatCompletionContext` ·
`StaticWorkbench` · `McpWorkbench` · `GatedWorkbench` ·
`OpenAIChatCompletionClient` · `SingleThreadedAgentRuntime` ·
`PythonCodeExecutionTool + Docker`

### Durak 6 · Takımlar — beş tip, gerçekten koşuyor

`Takım tipi` seçicisi + `Takımla sor`. Kadro: **Planner · Researcher · Critic**.

| id | sırayı kim belirliyor |
|---|---|
| `roundrobin` | sırayla |
| `selector` | model seçer |
| `swarm` | handoff |
| `magenticone` | planlayıcı |
| `graphflow` | DAG |

İkisini koştur — **`selector`** ve **`swarm`** — çünkü ölçülmüş uçlar onlar:

> *"Aynı görev, aynı ajanlar, tek değişen sırayı kimin belirlediği.
> Selector 204 token, Swarm 334. **%63,7 fark.** Ve ödediğin şey zekâ değil,
> **yönlendirme özerkliği** — 'kime devredeceğine sen karar ver' dediğinde fatura
> bu kadar artıyor."*

Sonra bağla: *"Agents SDK'nın tek modeli handoff. Yani AutoGen'in en pahalı
deseni, başka bir çerçevenin tek seçeneği."*

### Durak 7 · Onay kapısı — ve **reddet**

Kod isteyen bir soru sor (`VC_ALLOW_CODE_EXEC=1` açık, `python:3-slim` yerelde
hazır [ölçüldü]):

```
şu üç şirketin skorlarının standart sapmasını hesapla
```

Onay kartı çıkıyor ve **çalışacak kodun kendisi** görünüyor.

> *"Onay 'kod çalıştırılsın mı' demiyor. **Bu kodu** çalıştırayım mı diyor. İmza
> kodun üstünde — kod değişirse onay tutmaz."*

**Reddet.** Bu demonun en önemli tıklaması:

> *"Reddettim. Ve dikkat edin — tur çökmedi. Ajan gerekçeyi okudu ve size
> söyledi. Kapı bir istisna fırlatmıyor, bir **cevap** üretiyor."*

Sonra onayla → sol sütun genişliyor, terminal açılıyor, çıktı düşüyor.

> *"Terminal salt okunur. İçine komut yazılamıyor. Yeni bir yetenek açmıyor, var
> olanı görünür kılıyor."*

Ve dürüst sınırı **sen** söyle:

> *"Konteyner izole ama **ağ erişimi var**. Yukarı akış bir parametre vermiyor.
> 'Sandbox güvenli' demiyorum — 'kapı gerçek' diyorum."*

### Durak 8 · MAF düğmesi — perdenin kapanışı

Sağ üstteki rozete bas. `AutoGen` → `MAF`.

> *"Motor değişti. Kodun %72,5'i bunu fark etmedi."*

Tool'suz bir soru sor — MAF kipinde tool çağrılan turlarda `AgentResponse.text`
boş dönüyor ve cevap mesajlardan geri okunuyor. Çalışıyor ama demoda riske girme.

Akış ekranında MAF'ın **sekiz mekanizması** çiziliyor: `MAF kuruldu` ·
`Tool tanımlandı` · `Kapı çerçevede` · `Ajan kuruldu` · `Oturum açıldı` ·
`Ajan koşuyor` · `Onay istendi` · `MAF turu bitti`.

Üçüncüsünde dur:

> *"'Kapı çerçevede.' AutoGen'de kapıyı biz workbench'i sarmalayarak kurduk.
> MAF'ta bir parametre. Halefin çözdüğü şey tam olarak bu."*

Ve kapsamı **kendin daralt**, biri sormadan:

> *"MAF kipi dar: bir tool, bir onay, bir oturum. İkinci bir boru hattı değil,
> bir kıyas yüzeyi. 'MAF'ı da yaptık' demiyorum."*

## §5 · Üç liste: göster / değin / atla

### GÖSTERİLECEK — ekranda, canlı

* Akış ekranı: iki bant, mesaj türleri kenarlarda, canlı animasyon
* Takım kıyası: dört desen, aynı soru, token farkı
* Onay kapısı: kart, **reddetme**, turun ayakta kalması
* MAF düğmesi: motor değişimi
* Terminal paneli: Docker'da kod koşarken

### DEĞİNİLECEK — söylenir, gösterilmez

* **Fan-in kaybı ölçümü** — arıza enjeksiyonu demoda koşmuyor, sayıyı söyle
* **FIDES** — bizde yok; sorulursa açıklanır *(hazırda beklet, kendin açma)*
* **Purview** — uyum tarafı; M365 E5 gerektirdiğini **mutlaka** ekle
* **Dağıtık runtime** — ikisinde de bugün üretim yolu değil
* **Zamanlayıcı** — devredilmiş; yerli karşılığı yazılı ama bağlanmadı

### ATLANACAK — uzun destede kalsın

* Sekiz desenin tamamı *(dördü yeter: Concurrent, Sequential, Handoff, Code Exec)*
* Protokol tablosu (CloudEvents, gRPC, proto)
* AutoGen Studio
* Şekillerin neden elle çizilmiş göründüğü
* MAF'ın barındırma ailesi *(hepsi `alpha`)*

---

## §6 · Sorulara hazır cevaplar

**"Neden MAF değil?"**
> "Kodun %72,5'i motor bilmiyor, ve MAF düğmesi bunun kanıtı. Ama bugün MAF'a
> geçmek üç şey kaybettirir: dağıtık runtime yok, model önbelleği yok, ve ilginç
> olan her şey — harness, FIDES, beceriler — `experimental` ve gerçekten uyarı
> fırlatıyor. GA'dan sonra iki ayda on beş kırıcı değişiklik var."

**"Prompt enjeksiyonuna karşı ne yapıyorsunuz?"**
> "Kapımız tool adına ve **imzasına** bakıyor. Verinin nereden geldiğini
> izlemiyor — yani tarama sonucuna gömülü bir talimat kapımızdan geçer. Bunun
> deterministik cevabı MAF'ta var, adı FIDES, ve deneysel. Mimarimize takılabilir
> ama bugün 'bizde var' diyemem."

**"Bu üretime hazır mı?"**
> "Hayır, ve öyle sunmuyorum. Kapı gerçek, ölçümler gerçek, testler geçiyor.
> Eksikler yazılı: zamanlayıcı devredilmiş, kod yürütücünün konteyneri ağ
> görüyor, ve MAF kipi dar bir kıyas yüzeyi."

**"Kaç kişi, ne kadar sürede?"**
> "Birinci faz: otuz gün, tek kişi — onay kapısı, uyum kayıt hattı, tek dar
> kullanım. Kalan iki fazı şimdi konuşursak tahmin etmiş oluruz."

---

## §7 · Süre daralırsa

1. **Perde 1'in ortasını at** — ②③⑦ gider, ①④⑤⑥ kalır
2. **Perde 1b'yi ikiye indir** — "1'e karşı 40" ve "iki risk" cümlesi; harness
   slaytı gider
3. **Perde 2'nin ortasını at** — onay, denetim ve güven modeli asla atılmaz
4. **Perde 3'ü tek gösteriye indir** — akış ekranı + MAF düğmesi

**Kapanış hiçbir koşulda kısaltılmaz.** İstenen karar söylenmezse sunum
bilgilendirme olur, ve bilgilendirmeden karar çıkmaz.

---

*Ölçümlerin kaynakları: [22-maf-turkce.md](22-maf-turkce.md) (MAF),
[06-autogen-incelikleri.md](06-autogen-incelikleri.md) (tuzaklar),
`poc/kiyas.py` (desen faturası), `pipeline/compare_fanin.py` (kardeş kaybı).*
