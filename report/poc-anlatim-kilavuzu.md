# Motorları Anlatma Kılavuzu — Claude tarzı POC'u gösterirken

**Bu belge ne değil:** motor karşılaştırma raporu. O ayrı duruyor →
[`motor-secimi-workflow-desenleri-ve-karar.md`](motor-secimi-workflow-desenleri-ve-karar.md).

**Bu belge ne:** ekrana bakarken **ne söyleyeceğinin** kılavuzu. Hangi ekranda hangi
cümle, hangi sayı, hangi iyi/kötü yan. Sayıların hepsi bu POC'ta ölçüldü.

**Nerede koşuyor:** `.venv/bin/python demo-brain-agent/chat_server.py` → `http://127.0.0.1:8030`

---

## 0 · Anlatının bel kemiği

Ezberlenecek tek şey bu. Geri kalan her şey buna hizmet ediyor:

> **"Dört farklı motorda aynı grafı koşturdum. Çıktılar byte-byte aynı çıktı.
> Ayrıştıkları yer ne yaptıkları değil — ne kadar sürdüğü ve geride ne bıraktıkları."**

Bunu söyleyebilmek POC'un asıl değeri. Çünkü çoğu motor karşılaştırması "X daha
hızlı" der geçer; burada gösterilen şey **kararın nerede verildiği**.

İkinci cümle de şu olsun:

> **"Karar merci board'da. Motor sadece dağıtıcı. O yüzden motor değiştirmek
> davranışı değiştirmiyor — sadece maliyeti değiştiriyor."**

Bu cümleyi ekranda kanıtlayabiliyorsun; ezber değil, ölçüm.

---

## 1 · Ekran akışı — 12 dakikalık gösterim

| # | süre | ekran | ne yaparsın | ne söylersin |
|---|---|---|---|---|
| 1 | 1 dk | sohbet | üst barda `karşılaştır: 4 motorda koştur` seç | "Bu bir sohbet asistanı ama arkasında bir task board var." |
| 2 | 2 dk | sohbet | `"ETL pipeline kur"` yaz | "Planlayıcı bir DAG kurdu. Graf **bir kez** kuruluyor, dört motorda **aynen** koşuyor." |
| 3 | 2 dk | sohbet | dört motorun çıktısını tek tek aç | "Dördü de aynı metni üretti. Sayıların eşleşmesi beklenirdi, **metnin** eşleşmesi asıl kanıt." |
| 4 | 1 dk | **Akışlar** | az önceki akışa tıkla | "Kurduğum her graf kalıcı. Bu ekran aynı zamanda karşılaştırma ekranı." |
| 5 | 3 dk | akış detayı | bir düğüme tıkla → `kalıcı` seç → **⚡ Hepsinde koştur** | "Şimdi o düğümü bilerek patlatıyorum. Dört motorun **aynı** kararı verdiğini göreceğiz." |
| 6 | 2 dk | karşılaştırma tablosu | düğüm bazlı tabloyu göster | "Her sütun birebir aynı: 3/6 tamamlandı, 1 başarısız, 2 iptal, 3 deneme. Ayrıştıkları tek yer süre." |
| 7 | 1 dk | — | kapanış | aşağıdaki §5 |

**Zamanlama uyarısı:** adım 5'te Celery ~12-40 sn sürüyor. Konuşacak lafın hazır olsun
(bkz. §2.2'nin "kötü yanı" kutusu) — o bekleme aslında **anlatının bir parçası**.

---

## 2 · Motor motor

Her başlıkta aynı beş kutu var: **ne olduğu · iyi yanı · kötü yanı · ekranda nerede ·
söyleyeceğin cümle.**

### 2.1 · own — "hermes type" (bizim board'ımız)

**Ne olduğu.** Motor değil, **karar katmanı**. SQLite'ta bir task tablosu; her düğümün
durumu (`blocked → ready → running → done/failed/cancelled`), kaç kez denendiği,
kimin claim ettiği burada. Diğer üç motor bunun altında dağıtıcı olarak koşuyor.

**İyi yanı**
- **En hızlı: 0,01-0,05 sn.** Ağ yok, broker yok, süreç yok.
- **Tek karar noktası.** Retry, iptal zinciri, breaker, checkpoint hep burada. Motor
  değiştiğinde davranış değişmiyor — POC'un tüm iddiası buna dayanıyor.
- **`cancelled` durumu var.** Bir düğüm kalıcı patlayınca ardılları `cancelled`
  işaretleniyor. Celery ve Temporal'da böyle bir kavram yok.
- Çökme sonrası **düğüm içi checkpoint** — üçünde de yok.

**Kötü yanı**
- **Biz yazdık.** Kenar durumlarını da biz bulmak zorundayız — bu çalışmada **12 hata**
  çıktı ve hepsi bizim koddaydı.
- Tek makine, tek süreç. Dağıtık değil.
- Zamanlama, UI, operasyon aracı yok (ayrıca yazıldı ama olgun değil).
- Prod'da kanıtlanmadı; Airflow'un 10 yılı, Temporal'ın ekosistemi yok.

**Ekranda nerede.** `motor: own` seçiliyken sağ paneldeki **Board** sekmesi. Düğümler
renkli nokta ile durum gösteriyor.

> **Söyle:** *"Buna 'hermes type' diyorum çünkü Hermes-Agent'ın yaklaşımı: durumu
> kendi kalıcı tablonda tut, motoru değiştirilebilir bırak. Board karar veriyor,
> motor sadece 'şu task'ı koştur' emrini taşıyor."*

---

### 2.2 · Celery — iş kuyruğu

**Ne olduğu.** Dağıtık **görev kuyruğu**. Bir broker'a (Redis/RabbitMQ, bizde
filesystem) iş atarsın, worker'lar çeker. Workflow kavramı **canvas** ile ifade edilir:
`chain` / `group` / `chord`.

**İyi yanı**
- **Sektörde en yaygın.** Python ekibi zaten biliyor, öğrenme maliyeti düşük.
- **Gerçekten dağıtık.** Worker sayısını artırınca yatay ölçekleniyor.
- `acks_late` + `reject_on_worker_lost` ile **at-least-once teslim** kutudan geliyor.
- Bağımsız kısa işler için (e-posta, resim işleme, rapor) doğru araç.

**Kötü yanı** — burası anlatının en verimli yeri
- **Durum görünürlüğü çok zayıf.** Akış takılırsa hangi adımda olduğunu Celery
  söylemiyor. "Şu an neredeyiz?" sorusunun cevabı yok.
- **`cancelled`/`skipped` kavramı yok.** Batan bir zincirin kalanına ne olduğu belirsiz.
- **DAG'ı gönderimde sabitliyor.** Koşullu dal ya da dinamik fan-out istiyorsan
  `self.replace()` yazmak zorundasın.
- **İmza sızıntısı:** canvas'ın şekli fonksiyonun parametrelerini belirliyor. `chain`
  içindeki ikinci fonksiyon ilk parametresini kuyruktan alır — yani kompozisyon kararı
  **fonksiyon imzasına** sızıyor.
- **En yavaş: 12-40 sn.** Bunun ~6 saniyesi worker açılışı.

**Ekranda nerede.** Karşılaştırma tablosunda en alt satır, en büyük süre. Düğüm bazlı
tabloda `kosum` sütunu.

> **Söyle:** *"Şu 12 saniyeyi bekliyoruz — bu Celery'nin broker + worker açılışı. Ama
> asıl mesele hız değil: şu anda akış takılsa Celery bana 'hangi adımdayım' diyemez.
> Board olmasa bu ekranı çizemezdim."*

> **⚠ Burada bir hata bulduk, anlatmaya değer.** Düğüm bazlı tabloyu ekleyince geçici
> hatada Celery'nin düğümü **×3**, diğerlerinin **×2** koşturduğunu gördük. Sebep: iki
> ayrı yeniden-dağıtım yolu — hem `board.fail()` task'ı `ready`ye döndürüyor (dispatcher
> yeniden kuyruğa atıyor), hem `self.retry()` Celery'nin kendi retry'ını tetikliyordu.
> **Yan etkili bir düğümde bu "e-posta 3 kez gitti" demek.** `self.retry()` kaldırıldı,
> retry otoritesi board'a bırakıldı. Sonra dördü de ×2 oldu.
>
> Bu, sunumun en güçlü anı olabilir: *"İki dayanıklılık katmanını üst üste koymak
> bedava değil."*

---

### 2.3 · Airflow — veri hattı

**Ne olduğu.** Zamanlanmış, tekrarlayan **veri hatları** için orkestratör. Workflow bir
**dosya**: `PythonOperator`'ları `>>` ile bağlarsın, scheduler parse eder ve koşturur.
Veri akışı **XCom** üzerinden.

**İyi yanı**
- **Operasyonel görünürlükte açık ara birinci.** Her düğümün durumu metadata DB'de:
  `success` / `failed` / `upstream_failed` / `skipped`, `try_number`, süre.
- **`skipped` kaydı var.** Koşullu dalda seçilmeyen yol açıkça kaydediliyor — dördü
  içinde bunu yapan tek motor.
- **Zamanlama en güçlü:** cron + `catchup` (geçmişi doldurma).
- Veri mühendisliğinin fiilî standardı; işe alım ve dokümantasyon avantajı büyük.

Kalıcı hatada ölçülen tablo — bunu ekranda göster:

```
z1..z6     success            deneme=1     ← iş korundu
z7         failed             deneme=3     ← 3 kez denendi
z8,z9,z10  upstream_failed    deneme=0     ← HİÇ koşmadı, açıkça kaydedildi
```

**Kötü yanı**
- **Düğüm içi checkpoint yok.** Bir LLM düğümü yarıda kalırsa çağrılar baştan tekrar
  eder — **para yanar**.
- **Varsayılan 30 sn backoff.** Üç deneme = 62 saniye (canlı ölçüldü). Bizim akışımızın
  6.000 katı. *Kusur değil, tercih:* Airflow gerçek dış sistemlere bağlanır, orada
  30 sn beklemek doğrudur.
- **Operasyonel maliyet:** scheduler + webserver + metadata DB. Üçünü de ayakta tutmak
  gerekiyor.
- **Kurulum kırılgan** — bu POC'ta yaşandı: constraint dosyası `typing_extensions`'ı
  düşürüp `pydantic`'i kırdı, venv onarıldı.
- Graf parse zamanında sabitleniyor.

**Ekranda nerede.** Karşılaştırma tablosunda ~1,5-4,7 sn. Airflow'un çıktısı board'dan
değil **kendi XCom'undan** okunuyor — bunu vurgula.

> **Söyle:** *"Airflow'un çıktısını board'dan almıyorum, çünkü Airflow board'a hiç
> yazmıyor. Kendi XCom'undan okuyorum. Aynı metni vermesi, veri akışının o tarafta da
> doğru kurulduğunun **bağımsız** kanıtı."*

> **Panelde bir ayar değiştirdim, dürüstlük gereği söyle:** `retry_delay` varsayılan
> 30 sn'yi panelde kullanılamaz kılıyordu (tek senaryo 62 sn sürüyordu), **1 sn**'ye
> çekildi. İhracat varsayılanı değişmedi, parametreye çıkarıldı.

---

### 2.4 · Temporal — dayanıklı yürütme

**Ne olduğu.** **Workflow = kod.** Normal Python yazarsın (`await`, `if`, `for`), Temporal
her adımı event history'ye yazar. Süreç çökerse geçmişi **replay** ederek kaldığı yerden
devam eder.

**İyi yanı**
- **Çökmeden sonra devam eden tek motor.** Replay sırasında tamamlanmış activity'ler
  **atlanıyor** — dördü içinde bunu kutudan veren tek yapı.
- **Grafı ifade etmek en doğal.** Koşullu dal `if`, fan-out `for`. `BranchPythonOperator`
  ya da `self.replace()` gibi motor-özel kavram öğrenmiyorsun.
- own'dan sonra **en hızlı: 0,16-0,50 sn.**
- Ödeme, sipariş, uzun süren iş akışlarında sektörde yükselen tercih.

**Kötü yanı**
- **Determinizm disiplini.** Workflow kodunda `random`, `datetime.now()`, doğrudan IO
  yasak — replay bozulur. IO'yu activity'ye taşımak **mimariyi bölmek** demek.
- **Operasyonel maliyet en yüksek:** cluster ya da Temporal Cloud.
- **"Task/iş kalemi" kavramı yok.** Activity var, board benzeri bir model yok — düğüm
  bazlı tabloyu Temporal tek başına veremez.
- Seçilmeyen dal kaydı yok (`skipped` yok).
- **Kendi retry'ı bizimkiyle çakışabiliyor** — bu POC'ta yaşandı (aşağıda).

**Ekranda nerede.** Karşılaştırma tablosunda own'dan hemen sonra, ~0,2-0,5 sn.

> **Söyle:** *"Temporal'ın satış argümanı 'workflow = kod'. Gerçekten öyle: `if`
> yazıyorsun, dallanıyor. Bedeli determinizm — workflow'un içinde saate bile
> bakamıyorsun."*

> **⚠ Burada da bir hata bulduk.** Temporal'ın **kendi** retry'ı bizim board'ımızın
> deneme sayacını görmüyordu; `attempt` hep 0 kalınca "geçici hata" sonsuza dek ilk
> deneme sanılıyor ve düğüm kalıcı hata gibi batıyordu. Sayaç board'dan alınınca
> düzeldi. **Celery'de aynı hatanın ikizi çıktı.** İki kez aynı ders:
> *deneme sayacı board'dan gelmek zorunda.*

---

## 3 · Ekranda çıkan üç "aha" anı

Bunlar sunumun tepe noktaları. Öncesinde bir duraklama yap.

**① Dört çıktı byte-byte aynı**

```
own       ✓ 6/6 · 0,05 s
temporal  ✓ 6/6 · 0,46 s     {"dosya":"musteri_ozet.csv","satir":6,
airflow   ✓ 6/6 · 1,56 s      "csv":"musteri,adet,toplam\nm14,14,7031.0…"}
celery    ✓ 6/6 · 15,35 s
```

> *"Aynı graf, dört farklı yürütme motoru, tek bir metin. Motor bir uygulama detayı."*

**② Kalıcı hatada dört sütun da birebir**

| motor | süre | tamamlanan | başarısız | iptal | deneme |
|---|---:|---:|---:|---:|---:|
| own | **0,01 s** | 3/6 | 1 | 2 | 3 |
| temporal | 0,50 s | 3/6 | 1 | 2 | 3 |
| airflow | 4,75 s | 3/6 | 1 | 2 | 3 |
| celery | 12,39 s | 3/6 | 1 | 2 | 3 |

> *"Ayrıştıkları tek sütun süre. Çünkü kararı motor değil board veriyor."*

**③ Süre farkı 1.200 kat**

`0,01 s` → `12,39 s`. Aynı iş.

> *"Bu fark motorun yaptığı işten değil, taşıdığı altyapıdan geliyor: broker, worker
> açılışı, metadata DB. Ne kadar dayanıklılık istiyorsan o kadar ödüyorsun."*

---

## 4 · Gelecek sorular ve hazır cevaplar

**"Madem hepsi aynı sonucu veriyor, neden dördü birden?"**
> Çünkü ekip bir gün birini seçecek. Bu POC seçimi **ölçüyle** yapılabilir hale
> getiriyor. Ayrıca board'un motordan bağımsız olduğunu kanıtlıyor — yarın motor
> değiştirmek kod değişikliği değil, seçici değişikliği.

**"Celery neden bu kadar yavaş, yanlış mı kullandın?"**
> Hayır — ~6 saniyesi worker açılışı, kalanı dalga döngüsünün yoklama aralığı. Worker'ı
> açık tutmak iyileştirir. Ama asıl mesele hız değil: **görünürlük**. Bu ekranı Celery
> tek başına çizemez.

**"Board yazmak yerine hazır motor kullansaydınız?"**
> Kullanıyoruz — dördü de altında koşuyor. Board motorun **yerine** değil, **üstüne**
> geçiyor: motorların vermediği şeyi veriyor (`cancelled` zinciri, düğüm içi checkpoint,
> tek deneme sayacı). Bedeli: 12 hata bizim koddaydı.

**"Prod'da hangisini seçelim?"**
> "Hangisi" yanlış soru; doğru soru "hangi iş":
> - gecelik ETL, insan gözetimli veri hattı → **Airflow**
> - bağımsız kısa işler (e-posta, bildirim, resim) → **Celery**
> - uzun süren, çökmeye dayanması gereken, para/veri kritik → **Temporal**
> - hızlı iç akışlar, tam kontrol → **own**

**"Bu ölçümler gerçek yükü temsil ediyor mu?"**
> Hayır, ve bunu açıkça söyle. Oyuncak ölçek: filesystem broker, SQLite, tek makine,
> `SequentialExecutor`. Ölçülen şey **mekanizma**, kapasite değil. Airflow'un
> `retry_delay`'i panelde 1 sn'ye çekildi, varsayılanı 30 sn.

**"Dinamik graf kurulabiliyor mu — ajan çalışırken yeni düğüm ekleyebiliyor mu?"**
> İkiye ayır, yoksa yanlış cevap verirsin:
> - **Düğüm SAYISI** çalışma anında belirlenebiliyor. Ölçtük: Airflow `.expand()` ile
>   **5 örnek** üretti, Celery `self.replace()` ile task kendi yerine yeni akış koydu.
> - **Düğüm TÜRÜ** hiçbirinde eklenemiyor. Dördü de grafı bir noktada sabitliyor —
>   Celery gönderimde, Airflow parse'ta, Temporal kod yazımında.
>
> Yani "graf donuyor" eleştirisi sadece Airflow'a değil **hepsine** ait, ama "hiç
> dinamik değil" demek de yanlış olur.

---

## 5 · Kapanış

> *"Bu POC'un cevapladığı soru 'hangi motor daha iyi' değil. Cevapladığı soru şu:
> **kararı nereye koyarsak motor değiştirmek ucuz olur?** Cevap board oldu — ve dört
> motorda byte-byte aynı çıktıyı alarak bunu ölçtük.*
>
> *Ekibe önerim iki katmanlı: kararı board'da tut, motoru işe göre seç. Zamanlı veri
> hattıysa Airflow, dayanıklılık kritikse Temporal, kısa bağımsız işse Celery."*

---

## Ek · Ezberlenecek sayılar

| ne | değer |
|---|---|
| hatasız 6 düğüm — own / temporal / airflow / celery | 0,05 / 0,46 / 1,56 / **15,35** sn |
| kalıcı hata — dördü de | 3/6 tamamlandı · 1 başarısız · 2 iptal · 3 deneme |
| geçici hata — düğüm kaç kez koştu | dördü de **×2** (düzeltmeden önce celery ×3) |
| Airflow varsayılan backoff | 30 sn → 3 deneme = **62 sn** |
| Celery açılış maliyeti | ~6 sn / koşu |
| en hızlı ↔ en yavaş | **1.200 kat** |
| dinamik fan-out (düğüm **sayısı**) | Airflow `.expand()` → **5 örnek** · Celery `self.replace()` |
| çalışma anında yeni düğüm **türü** | **hiçbirinde yok** |
| bulunan hata sayısı | **12** |

## Ek · Kaçınılacak üç cümle

- ❌ *"Airflow yavaş."* → ✅ *"Airflow'un backoff'u batch işler için ayarlanmış; hızlı
  akışlarda bu bir uyumsuzluk."*
- ❌ *"Celery kötü."* → ✅ *"Celery iş kuyruğu; ondan akış motoru olmasını istemek
  yanlış beklenti."*
- ❌ *"Kendi motorumuzu yazdık, hepsinden hızlı."* → ✅ *"Board bir motor değil, karar
  katmanı — ve hızlı olmasının sebebi az iş yapması."*
