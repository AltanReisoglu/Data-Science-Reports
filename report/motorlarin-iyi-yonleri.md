# Motorlar — Anlatım Föyü

Nasıl çalışır · artıları · eksileri. Soru "hangisi daha iyi" değil — **"benim işim hangisine benziyor"**.

> Tek sayfa A4 baskı hâli: `report/motorlarin-iyi-yonleri.pdf`

---

## Celery — dağıtık iş kuyruğu
**En hızlı kurulan**

### Nasıl çalışır
```
.delay()  →  mesaj kuyruğa (Redis)
                  ↓
        boştaki worker kendi çeker → çalıştırır → siler
```
- Fonksiyonun **kodu gitmez**, sadece **adı** gider. Kod worker'da zaten var.
- Kimse worker seçmez — **boşta olan gelip alır.**
- Retry = aynı mesajı kuyruğa geri koymak → fonksiyon **1. satırdan** başlar.

### Artıları
- **Yarım günde kurulur** — bir Redis, o kadar.
- Öğrenmesi kolay: `@app.task` yaz, bitti.
- **Worker ekleyerek büyür** — 10 kat yük, 10 kat worker.
- Python'un standardı; Django/Flask hazır çalışır.
- Yük patlamasında sistem çökmez, **kuyruk tamponlar**.

### Eksileri
- **Çok adımlı iş kavramı yok** — her task bölünmez bir kutu.
- **Retry baştan koşar** → pahalı adımı 2 kez ödersin.
- "Nerede kaldım" defterini **sen tutarsın**.
- Aynı iş iki kez çalışabilir → idempotency sende.
- Geçmiş/denetim kaydı tutmaz.

**En parladığı yer:** E-posta gönder, resim boyutlandır, rapor üret, tek bir LLM çağrısını arka plana at.

---

## Airflow — zamanlı iş planlayıcı
**En görünür olan**

### Nasıl çalışır
```
DAG yazılır:  fetch → process → deliver
        ↓ saati gelince scheduler tetikler
adımlar sırayla koşar, durum DB'ye yazılır → UI'da görünür
```
- İşin **şekli önceden çizilir** (statik graf).
- Biten adım DB'de **"başarılı"** kalır.
- Bir adım patlarsa **sadece o adım** tekrar koşar.

### Artıları
- **Zamanlamada rakipsiz** — cron'u yaz, gerisi onun.
- **Backfill:** "bugün kurdum, son 6 ayı doldur". Başka araçta yok.
- **Arayüzü çok güçlü** — hangi adım, ne kadar sürdü, logu ne.
- Öncekiler korunur, boşa iş yapılmaz.
- Yüzlerce hazır bağlantı (S3, BigQuery, Spark, dbt…).

### Eksileri
- **Ağır kurulum** — 4 bileşen ayakta olmalı.
- **İşin şekli önceden belli olmalı** → ajanın serbest akışı sığmaz.
- Adımın **içinde** kayıt yok: 7/10'da çökerse baştan.
- Uzun beklemeler ("3 gün onay") için uygun değil.
- Tek bir işi arka plana atmak için abartı.

**En parladığı yer:** Gecelik veri işleri, günlük raporlar, veri ekibinin sabah kontrol ettiği pipeline'lar.

---

## Temporal — çökmeye dayanıklı yürütme
**En dayanıklı olan**

### Nasıl çalışır
```
Workflow normal kod olarak yazılır
        ↓ her adım kalıcı deftere yazılır
çökünce kod baştan koşar ama deftere bakıp
biten adımları ATLAR → kaldığı yerden devam
```
- Defteri tutan taraf **senin sürecinin dışında** — o yüzden çökmeyi kurtarabiliyor.
- Yan etkili her adım ayrı bir parça ("activity").

### Artıları
- **Tamamlanan adım bir daha koşmaz** — pahalı adımı iki kez ödemezsin.
- **Kurtarma kodu yazmıyorsun** — retry/timeout/devam hazır geliyor.
- **Günlerce bekler.** "3 gün insan onayı bekle" normal bir satır.
- **Dinamik + dayanıklı** aynı anda — akış koşarken belirlenebilir.
- Her adımın kaydı → geriye sarıp "tam ne oldu" görürsün.

### Eksileri
- **Cluster işletmen gerekir** (ya da Cloud'a ücret).
- **Determinizm kuralları:** workflow'da rastgele/saat/HTTP yasak.
- Öğrenmesi **haftalar** sürer — kalıcı bir disiplin.
- Kod değişince sürüm yönetimi derdi.
- Basit zamanlı iş için fazla ağır.

**En parladığı yer:** Ödeme akışları, sipariş süreçleri, çok adımlı ajan işleri — yarıda kalması pahalıya patlayan her şey.

---

## Kendi Çekirdeğimiz — SQLite üstünde hafif motor
**En hafif olan**

### Nasıl çalışır
```
İş = SQLite'ta bir satır
   ↓ worker satırı kilitleyerek kapar (tek worker alır)
kilidin süresi var → worker çökerse süre dolar
başka worker devralır → kayıttan devam eder
```
- Temporal'ın fikri, tek makinede ve birkaç yüz satırda.

### Artıları
- **Sıfır yeni servis** — sadece SQLite.
- **Kod tamamen bizim** — anlamadığımız yer yok.
- **Ölçtük:** pahalı adım **1 kez** koşuyor (Temporal'la aynı sonuç).
- Worker çökerse iş kaybolmuyor, başkası devralıyor.

### Eksileri
- **Bakım bize ait** — hata çıkarsa biz düzeltiriz.
- Tek makine ölçeği; çok sunucuda zorlanır.
- Kenar durumlar zamanla ortaya çıkar.
- Hazır arayüz / ekosistem yok.

**En parladığı yer:** Tek makine, küçük ekip, tam kontrol istenen durumlar.

---

## Tek bakışta

| Motor | En iyi yaptığı şey | En zayıf yanı | Pahalı adım kaç kez koştu? |
|---|---|---|:---:|
| **Celery** | Hızlı kurulur, kolay ölçeklenir | Hafızası yok | **2×** |
| **Airflow** | Zamanlama + görünürlük | Şekil önceden sabit | **1×** |
| **Temporal** | Çökmeye dayanıklılık | Ağır kurulum + disiplin | **1×** |
| **Kendi çekirdeğimiz** | Hafiflik + tam kontrol | Bakım bizde | **1×** |

---

## Anlatırken vurgula

**Tek soru şu:** iş yarıda çökerse "nerede kalmıştım" defterini **kim tutuyor?**
Celery'de **hiç kimse**, Airflow'da **sistem ama sadece adımlar arasında**, Temporal'da **tamamen sistem**.

Ve bunlar **rakip değil** — Airflow zaten Celery'yi kendi işçi havuzu olarak kullanabiliyor. Çoğu gerçek sistem birden fazlasını birlikte çalıştırır.

> Kanıt: `poc-task-mgmt/` — dört motor da gerçekten koşturuldu, sayılar ölçüldü.

---
---

# Çalışma Mentaliteleri — piyasa neden kullanıyor?

Her araç bir **inançtan** doğdu. Piyasada neden yaygın olduğu, o inancın kaç şirketin derdine denk düştüğüdür.

---

## Celery — Python dünyası, 2009
> ### "Kullanıcıyı bekletme. İşi birine devret, sen hemen cevap dön."

**Nereden doğdu:** Web uygulamaları büyüdükçe aynı dert çıktı: kullanıcı butona basıyor, sunucu **e-posta gönderirken 10 saniye kilitleniyor**. Celery bu tek derde çözüm olarak doğdu — ve o kadarla kaldı. Sadeliği gücü.

**Kim kullanıyor:** Python/Django ile web ürünü yazan hemen herkes. Ölçek örneği: Instagram'ın bilinen en büyük Celery+RabbitMQ kurulumlarından biri var.

**Ne için:** E-posta/bildirim, resim–video işleme, rapor üretme, ödeme sonrası işler, tek seferlik LLM çağrısı.

**Neden bu kadar yaygın:**
- **Dert evrensel** — her web uygulamasının arka plan işi var.
- Girişi bedava; mevcut Redis'in üstüne kurulur.
- Django/Flask ile **fiilen standart**; her ekipte bilen biri var.

---

## Airflow — Airbnb → Apache, 2014
> ### "Veri işleri takvime bağlıdır ve insan gözüyle izlenmelidir."

**Nereden doğdu:** Airbnb'de veri ekibi her gece onlarca rapor/tablo üretiyordu ve **cron script'leri yönetilemez** hâle gelmişti: hangisi patladı, hangisi hangisini bekliyor, dünkü eksik nasıl doldurulur? Airflow tam bu kaosu **görünür** kılmak için yazıldı.

**Kim kullanıyor:** Veri mühendisliği ekipleri — fiilen sektör standardı. Bulut sağlayıcılar yönetilen sürümünü satıyor (AWS MWAA, Google Cloud Composer, Astronomer).

**Ne için:** Gecelik ETL, veri ambarı beslemesi, dbt/Spark zincirleri, günlük iş zekâsı raporları, ML eğitim pipeline'ları.

**Neden bu kadar yaygın:**
- **Veri ekibinin dili** — "DAG" kelimesi bu yüzden yerleşti.
- **Backfill** tek başına satın alma sebebi.
- Arayüz, teknik olmayan paydaşa da gösterilebiliyor.

---

## Temporal — Uber/Cadence → 2019
> ### "'Ya yarıda kalırsa?' sorusu geliştiricinin derdi olmamalı."

**Nereden doğdu:** Uber'de bir yolculuk/ödeme akışı onlarca servise dağılıyordu. Her ekip aynı şeyi tekrar yazıyordu: durum tablosu, retry sayacı, "çökerse nereden devam" kodu. Cadence bunu **bir kez ve doğru** çözmek için yazıldı; ekibi ayrılıp Temporal'ı kurdu.

**Kim kullanıyor:** Para/işlem akışı kritik olan şirketler. Kamuya açık konuşan kullanıcılar arasında Netflix, Snap, Coinbase, Datadog, Box var.

**Ne için:** Ödeme ve sipariş akışları, kullanıcı onboarding/KYC, altyapı sağlama (provisioning), ajan iş akışları.

**Neden yükselişte:**
- Mikroservis dağıldıkça "yarıda kalma" **en pahalı hata** hâline geldi.
- Yazılmayan kod: retry/state/kurtarma katmanı komple gidiyor.
- **LLM ajanları** uzun ve pahalı adımlar ürettikçe talep arttı.

---

## Kendi Çekirdeğimiz — Hermes/OpenClaw deseni
> ### "Garantinin %80'i, maliyetin %5'iyle — ve kod tamamen bizde."

**Nereden doğdu:** Ajan araçları tek makinede koşuyor ve küçük ekipler yönetiyor. Temporal cluster'ı işletecek kişi yok; ama "worker çökerse iş kaybolmasın" yine de şart. Çözüm: **SQLite üstünde** aynı fikrin küçük hâli.

**Kim kullanıyor:** Kaynak kodunu incelediğimiz ajanlar — **Hermes** ve **OpenClaw** tam bu deseni kuruyor. **Shannon** ise tersini seçip doğrudan Temporal'a biniyor.

**Ne için:** Ajan görev yönetimi — sıraya alma, deneme sayacı, çökme sonrası devralma, kaldığı yerden devam.

**Neden bu rota:**
- Yeni servis eklemeden **bugün** çalışıyor.
- **Ölçtük:** pahalı adım 1× — Temporal'la aynı sonuç.
- İleride büyürse Temporal'a geçiş yolu açık kalıyor.

---

## Mentaliteleri tek satırda ayır

| Motor | İnancı | Çözdüğü sorun |
|---|---|---|
| **Celery** | "işi **başkasına** ver" | iş gücü |
| **Airflow** | "işi **takvime** bağla" | zaman + görünürlük |
| **Temporal** | "işi **ölümsüz** yap" | güvenilirlik |
| **Kendi çekirdeğimiz** | "işi **kendi defterimize** yaz" | maliyet + kontrol |

---

## Piyasa nereye gidiyor

Uzun yıllar **Celery + Airflow** ikilisi yetti: kısa arka plan işleri + gecelik veri işleri.

Sonra mikroservisler dağıldı, işler **uzadı ve pahalılaştı** → "yarıda kalırsa" maliyeti arttı, **durable execution** (Temporal) yükseldi.

**Şimdi LLM ajanları** aynı baskıyı katlıyor: her adım para, her akış dinamik. Bu yüzden yeni ajan altyapıları ya Temporal'a biniyor ya da kendi küçük durable çekirdeğini yazıyor — **tam bizim durduğumuz yer.**

> Kaynak: `report/task-yonetimi-altyapi-karari.md` — yedi ajanın kaynak kodu incelendi.
