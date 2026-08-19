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

## §4 · Ekran koreografisi

Sunum boyunca ekranda **üç şey** var ve geçişleri önceden bilinmeli:

| Perde | Sol ekran | Sağ panel |
|---|---|---|
| 0 · Açılış | — | `hap-autogen.pdf` kapak |
| 1 · Motor | slayt | AutoGen destesi, sen ilerletirsin |
| 1b · Halef | slayt | aynı deste, son dört slayt |
| 2 · Kuşatma | slayt | `hap-openclaw.pdf` |
| 3 · **Canlı** | **PoC sohbet + akış ekranı** | deste **kapalı** |
| 4 · Kapanış | — | — |

Perde 3'ün koreografisi — ölçülmüş sırayla:

1. Soruyu yaz: **`search_docs ile durable execution konusunda ne dediğimizi bul`**
   *(üç ölçülmüş sorudan biri; başkasını seçme — model tool çağırmayabiliyor ve
   şerit dört aşamada bitiyor)*
2. Şerit dolarken **renkleri söyle**: turuncu = bizim yazdığımız, mor = `autogen_core`,
   mavi = `autogen_agentchat`
3. Üstteki özeti oku: `2 LLM · 1 TOOL · 19775 TOKEN`
4. **Takım düğmeleri** → aynı soru, dört desen, canlı token farkı
5. **Onay kapısı** → kod yürütme sorusu, kartı göster, **reddet**, turun çökmediğini göster
6. **Sağ üst MAF düğmesi** → *"motor değişti"*
7. Akış ekranını aç → telemetri, iki bant, mekanizma isimleri

---

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
