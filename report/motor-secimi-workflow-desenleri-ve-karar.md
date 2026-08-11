# Motor Seçimi — Workflow Desenleri, Ölçümler ve Ekip İçin Karar

**Kapsam:** Taban çizgisi (çıplak Python) · Celery · Airflow · Temporal
**Yöntem:** İki ayrı ölçüm katmanı —
① motorlar **saf hâliyle** (board'suz, `saf-motorlar/`), ② motorlar **bizim board'ımızla** (`demo-brain-agent/`).
Airflow bu çalışmada **gerçekten kuruldu** (2.10.5) ve koşturuldu; önceki raporlarda "ölçülmedi" diye işaretliydi.
**Tarih:** 11 Ağustos 2026 · **Toplam:** 9 test paketi, 166+ kontrol, **15 hata bulundu ve düzeltildi**

> Bu belge, motor karşılaştırması üzerine yaptığımız **tüm çalışmanın tek yerde toplanmış hâlidir**:
> saf POC'ler, workflow desenleri, board'lu ölçümler, düğüm bazlı simülasyon, web paneli,
> yol boyunca bulunan hatalar ve bunlardan çıkan karar.

## Bir bakışta sonuç

| soru | cevap |
|---|---|
| Hangi motor "daha iyi"? | **Yanlış soru.** Dördü de dört deseni yapıyor, aynı çıktıyı üretiyor. |
| Peki ayrım nerede? | **Hız** (0,05 s ↔ 39 s), **ne kaydettikleri**, **işletme maliyeti** |
| Bizim işimize hangisi uyuyor? | Bugün **kendi çekirdeğimiz**, yarın (yan etki/uzun akış gelirse) **Temporal** |
| Airflow? | Yürütücümüz değil — **ihracat hedefi** ve gecelik batch için |
| En net teknik üstünlük? | **Temporal'ın replay'i** — çökme sonrası devamı kutudan veren tek motor |

---

# BÖLÜM 0 — Terimler (kısa kısa)

Rapordaki her terim. Bilinmeyen bir kelime kalmasın.

## Temel

| terim | ne demek |
|---|---|
| **workflow / pipeline / akış** | Birbirine bağlı adımlar bütünü. Üçü aynı şey; motorlar farklı isim kullanıyor. |
| **düğüm (node) / task / adım** | Akıştaki tek bir iş birimi. |
| **DAG** | *Directed Acyclic Graph* — yönlü, çevrimsiz graf. "A bitmeden B başlamaz" ilişkilerinin bütünü. Çevrim olamaz, olursa akış kilitlenir. |
| **bağımlılık** | Bir düğümün beklediği önceki düğüm. |
| **fan-out** | Bir düğümden çok dala açılma (1 → N). |
| **fan-in / join** | Çok daldan tek düğümde birleşme (N → 1). |
| **topolojik sıra** | Bağımlılıkları bozmayan çalıştırma sırası. |
| **orkestrasyon** | Kimin ne zaman koşacağına karar verme işi. |
| **yürütme (execution)** | İşin fiilen koşturulması. |
| **worker** | İşi koşturan işçi süreç. |
| **broker** | Mesaj kuyruğu (Redis/RabbitMQ/dosya sistemi). İşi worker'a taşır. |
| **backend / result backend** | Sonuçların saklandığı yer. |
| **scheduler** | Zamanı gelen akışı başlatan bileşen. |

## Dayanıklılık

| terim | ne demek |
|---|---|
| **retry** | Hata sonrası yeniden deneme. |
| **backoff** | Denemeler arası bekleme. Sabit (30 sn) ya da artan olabilir. |
| **idempotens** | Aynı işi iki kez yapmanın bir kez yapmakla aynı sonucu vermesi. Retry varsa şart. |
| **yan etki** | Dışarıya kalıcı etki (e-posta, ödeme, dosya yazma). Tekrarlanırsa zarar verir. |
| **checkpoint** | Yarım kalan işin kaydı; devralan buradan devam eder. |
| **replay** | Çökme sonrası akışı olay geçmişinden yeniden oynatma. Tamamlananlar atlanır. |
| **at-most-once** | "En fazla bir kez" — aynı iş iki worker'a aynı anda verilmez. |
| **at-least-once** | "En az bir kez" — iş mutlaka teslim edilir ama çiftlenebilir. |
| **exactly-once** | "Tam bir kez" — pratikte at-least-once + idempotens ile elde edilir. |
| **acks_late** | Celery: onayı iş bittikten sonra ver. Worker çökerse mesaj yeniden teslim edilir. |
| **lease (kira)** | Bir işi alma hakkının ömrü. Süre dolarsa başkası alabilir. |
| **determinizm** | Aynı girdiye hep aynı çıktı. Temporal'ın workflow gövdesinde şart. |

## Durum

| terim | ne demek |
|---|---|
| **success** | Başarıyla bitti. |
| **failed** | Denemeler tükendi, başarısız. |
| **up_for_retry** | Hata aldı, yeniden denenecek (Airflow). |
| **upstream_failed** | Üstündeki düğüm battığı için **hiç koşmayacak** (Airflow). |
| **skipped** | Koşullu dalda **seçilmediği için** koşmayacak (Airflow). |
| **görünürlük** | "Akış şu an nerede?" sorusuna cevap verebilme. |

## Motora özgü

| terim | motor | ne demek |
|---|---|---|
| **canvas** | Celery | `chain` / `group` / `chord` ile akış kurma yöntemi. |
| **chain** | Celery | Sırayla: `a → b → c`. Öncekinin dönüşü sonrakinin ilk argümanı olur. |
| **group** | Celery | Paralel: hepsi aynı anda. |
| **chord** | Celery | `group` bitince bir callback koş. Fan-in'in Celery'deki adı. |
| **signature (`.s()`)** | Celery | Bir task çağrısının "dondurulmuş" hâli; canvas bunlarla kurulur. |
| **`self.replace()`** | Celery | Bir task'ın kendi yerine başka bir akış koyması. Koşullu/dinamik için tek yol. |
| **workflow** | Temporal | Dayanıklı akış gövdesi. Çökse replay ile devam eder. |
| **activity** | Temporal | Workflow'un çağırdığı gerçek iş. IO burada olmak zorunda. |
| **event history** | Temporal | Kutudan gelen tam denetim izi. |
| **operator** | Airflow | Bir düğümün türü (`PythonOperator`, `BashOperator`…). |
| **XCom** | Airflow | Düğümler arası veri taşıma (*cross-communication*), metadata DB üzerinden. |
| **BranchPythonOperator** | Airflow | Koşacak düğümün **adını döndüren** operatör; seçilmeyen dal `skipped` olur. |
| **trigger_rule** | Airflow | Bir düğümün ne zaman koşacağı kuralı. Varsayılan `all_success`. |
| **Dynamic Task Mapping / `.expand()`** | Airflow | Düğüm **sayısının** çalışma anında belli olması (2.3+). |
| **parse zamanı** | Airflow | DAG dosyasının okunduğu an. Düğüm **türleri** burada sabitlenir. |
| **catchup / backfill** | Airflow | Kaçırılan zamanları sonradan koşturma. |

---

# BÖLÜM 1 — Dört motor workflow'a nasıl bakıyor

Bu, raporun en önemli bölümü. Her motorun **zihinsel modeli** farklı, ve bütün iyi/kötü yanları buradan türüyor.

## 1.1 Taban çizgisi (çıplak Python) — "workflow diye bir şey yok"

Graf, **çağrı sırasıdır**. `if` ve `for` dilin kendisidir. Ayrı bir kavram yoktur.

```python
s = None
for i in range(1, 11):
    s = zincir_adim(i, s)
```

**Zihinsel model:** Yok. Program akışı = iş akışı.
**Bedeli:** Hiçbir garanti yok. Süreç ölünce her şey gider.

## 1.2 Celery — "workflow = canvas (imza kompozisyonu)"

Graf, **task imzalarının birleştirilmesiyle** kurulur ve `apply_async()` anında **sabitlenir**.

```python
chord([chain(cek.s(), tara.s()), test.s()], chain(esle.s(), rapor.s()))
```

**Zihinsel model:** "Şu çağrıları şu şekilde zincirle, sonra gönder."

**Kritik özellik — veri akışı imzayı belirler:** chain'de önceki task'ın dönüşü sonrakinin **ilk pozisyonel argümanı** olur. Bu, fonksiyon imzalarına sızar. Ölçümde bunu canlı yaşadık:

```
TypeError: c_zincir() missing 1 required positional argument: 'onceki'
```

Zincirin **ilk** task'ı hiç argüman almadan çağrılıyor, bu yüzden `onceki=None` varsayılanı **zorunlu**. Yani fonksiyon, "ben bir zincirin parçasıyım ve belki başıyım" bilgisini taşımak zorunda.

**Koşullu ve dinamik için canvas yetmiyor.** İkisinde de `self.replace()` kullanmak gerekti — task kendi yerine çalışma anında yeni bir akış koyuyor:

```python
@app.task(bind=True)
def c_dal_sec(self, t):
    raise self.replace(c_raporla.s(t) if t["asti"] else c_atla.s(t))
```

Sonuç: **graf, task'ın içine taşındı.** Dışarıdan bakan biri akışı tek yerde göremiyor.

## 1.3 Airflow — "workflow = dosya (statik graf tanımı)"

Graf bir **dosyada** durur, scheduler onu **parse eder**.

```python
cek >> tara
[tara, test] >> esle
```

**Zihinsel model:** "Düğümleri ve aralarındaki okları bir dosyaya yaz; gerisini scheduler halleder."

**Kritik özellik — her düğümün bir DURUMU var.** Dört motor içinde yalnız Airflow, koşmayan düğümü de kaydeder:

| durum | anlamı |
|---|---|
| `success` | koştu, başardı |
| `failed` | koştu, denemeler tükendi |
| `up_for_retry` | hata aldı, yeniden denenecek |
| `upstream_failed` | **üstü battığı için hiç koşmayacak** |
| `skipped` | **koşullu dalda seçilmediği için koşmayacak** |

Bu, bizim board'a sonradan `cancelled` olarak eklediğimiz şeyin kutudan gelen hâli.

**Koşullu dal iki ayrı kavram gerektiriyor:**
1. `BranchPythonOperator` — koşacak düğümün **adını** döndürür
2. Birleşimde `trigger_rule="none_failed_min_one_success"` — **olmazsa** birleşim düğümü de `skipped` olur

**Dinamik fan-out** `.expand()` ile mümkün: düğüm **sayısı** çalışma anında belli olur, ama düğüm **türü** dosyada sabittir.

## 1.4 Temporal — "workflow = kod"

Graf ayrı bir yapı **değil**. `for`, `if`, `await`, `asyncio.gather` normal Python.

```python
if t["asti"]:
    return await workflow.execute_activity("dal_raporla", t, ...)
return await workflow.execute_activity("dal_atla", t, ...)
```

**Zihinsel model:** "Normal Python yaz; ben onu dayanıklı kılarım."

**Kritik özellik — determinizm disiplini.** Replay'de aynı sonucu vermesi gerektiği için workflow gövdesinde IO/rastgelelik/saat okuma **yasak**. Hepsi activity'ye taşınmak zorunda. Bu bir kısıt değil, **sözleşme** — dayanıklılığın bedeli.

**Sonucu:** koşullu ve dinamik desenler **ek kavram gerektirmiyor**. Ama seçilmeyen dal **hiç var olmuyor** — "atlandı" diye bir kayıt yok, çünkü ortada düğüm yok.

---

# BÖLÜM 2 — Ölçümler

## 2.1 Test edilen desenler ve neden

Tek bir elmas graf motorları ayırt etmiyor — dördü de yapıyor. Gerçek fark şu desenlerde çıkıyor:

| desen | ne | neyi ölçer |
|---|---|---|
| **D1 elmas** | paralel dal + join | temel yetenek |
| **D2 zincir** | 10 adım sıralı | uzun akışta ara durum, retry maliyeti |
| **D3 koşullu** | çalışma anında dal seçimi | "if" motorda nasıl ifade edilir, seçilmeyen dala ne olur |
| **D4 dinamik** | N çalışma anında belli | fan-out sayısı önceden bilinmiyorsa ne olur |

D3 ve D4 kritik, çünkü ikisi de **"graf ne zaman belli olur"** sorusunu soruyor.

## 2.2 Mutlu yol — hepsi doğru sonuç üretti

| desen | taban çizgisi | Celery | Airflow | Temporal |
|---|---|---|---|---|
| D1 elmas | ✓ 0,03 s | ✓ 12,23 s | ✓ (5/5 success) | ✓ 0,49 s |
| D2 zincir | ✓ 0,00 s | ✓ 0,52 s | ✓ (10/10 success) | ✓ 0,23 s |
| D3 koşullu | ✓ 0,00 s | ✓ 1,00 s | ✓ (4 success + **1 skipped**) | ✓ 0,16 s |
| D4 dinamik | ✓ 0,00 s | ✓ 2,00 s | ✓ (**5 örnek** çalışma anında) | ✓ 0,18 s |

**Dördü de dört deseni yapabiliyor.** Veri akışı hiçbirinde kopmadı. Fark *yapıp yapamamada* değil, **nasıl yaptıklarında ve ne kaydettiklerinde**.

## 2.3 Hata davranışı — asıl ayrışma

### Kalıcı hata: 10 adımlı zincirin 7. adımı her denemede patlıyor

| motor | sonuç | geride ne kaldı |
|---|---|---|
| taban çizgisi | `RuntimeError` | **hiçbir kayıt yok** — ilk 6 adımın sonucu bellekte, süreçle birlikte gitti |
| Celery | `RuntimeError` (3. denemede) | **kalan adımlara ne olduğu bilinmiyor** — "atlandı" diye bir durum yok |
| Temporal | `WorkflowFailureError` | event history'de var ama düğüm bazlı özet yok |
| **Airflow** | tam tablo ↓ | **her düğümün durumu kayıtlı** |

Airflow'un ölçülen çıktısı:

```
z1..z6   success            deneme=1     ← iş korundu
z7       failed             deneme=3     ← 3 kez denendi
z8,z9,z10 upstream_failed   deneme=0     ← HİÇ koşmadı, açıkça kaydedildi
özet: {'success': 6, 'failed': 1, 'upstream_failed': 3}
```

Bu tablo tek başına Airflow'un operasyonel üstünlüğünü anlatıyor: **sabah gelen operatör tek sorguyla ne olduğunu görüyor.**

### Geçici hata (ilk denemede patlıyor, sonra düzeliyor)

| motor | toparladı mı | deneme | backoff |
|---|---|---|---|
| taban çizgisi | ❌ **hayır** — retry yok, akış durur | 1 | — |
| Celery | ✓ evet | 3'e kadar | `countdown` (bizde 1 sn) |
| Airflow | ✓ evet (`deneme=2 success`) | 3'e kadar | **30 sn (ölçüldü)** |
| Temporal | ✓ evet | 3'e kadar | 200 ms |

Airflow'un 30 saniyelik backoff'u canlı ölçüldü — üç deneme arası tam 30'ar saniye:
```
09:47:19 → UP_FOR_RETRY
09:47:50 → UP_FOR_RETRY     (+30 sn)
09:48:21 → FAILED           (+30 sn)
```

**Bu bir kusur değil, tercih:** Airflow gerçek dış sistemlere (veritabanı, API) bağlanan batch işler için tasarlandı; oralarda 30 sn beklemek doğru. Ama toplam 62 saniye — bizim 0,01 saniyelik akışımızın 6.000 katı.

### Çökme (worker ölürse)

| motor | tamamlanmış iş | yarım kalan düğüm |
|---|---|---|
| taban çizgisi | ❌ **her şey gider** | — |
| Celery | sonuç backend'de kalır ama **chain yeniden kurulmaz** | baştan |
| Airflow | ✓ XCom kalıcı, tamamlananlar tekrar koşmaz | **baştan** (düğüm içi checkpoint yok) |
| **Temporal** | ✓ replay → **tamamlanmış activity'ler ATLANIR** | baştan (activity içi checkpoint yok) |

**Temporal burada tek başına ayrışıyor** — çökme sonrası devamı kutudan veren tek motor.

## 2.4 Yapısal karşılaştırma

| soru | taban | Celery | Airflow | Temporal |
|---|---|---|---|---|
| **"şu an neredeyiz?"** | ❌ yok | ⚠ **çok zayıf** | ✅ **en iyi** (durum + UI) | ✅ event history |
| **grafı tek yerde görebilir miyim** | ❌ örtük | ⚠ `replace` ile dağılıyor | ✅ dosyada | ✅ kodda |
| **koşullu dal** | dilin `if`'i | `self.replace()` gerekli | `BranchPythonOperator` + `trigger_rule` | dilin `if`'i |
| **seçilmeyen dal kaydı** | ❌ yok | ❌ yok | ✅ **`skipped`** | ❌ yok |
| **dinamik fan-out** | dilin `for`'u | `self.replace()` + chord | `.expand()` (sayı dinamik, tür sabit) | dilin `for`'u |
| **çalışma anında YENİ düğüm TÜRÜ** | ❌ | ❌ | ❌ | ❌ |
| **zamanlama** | ❌ yok | Celery Beat (backfill yok) | ✅ **en güçlü** (+catchup) | ✅ Schedules (+backfill) |
| **imza sızıntısı** | yok | ⚠ **var** | yok (XCom ayrı) | yok |
| **kurulum** | yok | broker | scheduler + webserver + DB | cluster/Cloud |

**Önemli ortak bulgu:** Hiçbiri çalışma anında **yeni düğüm türü** ekleyemiyor. Dördü de grafı bir noktada sabitliyor — Celery gönderimde, Airflow parse'ta, Temporal kod yazımında. Bu, daha önce Airflow'a yönelttiğim "graf donuyor" eleştirisinin aslında **hepsi için geçerli** olduğunu gösteriyor.

---

## 2.5 İkinci katman — motorlar BİZİM board'ımızla

Yukarıdaki ölçümler motorların **saf** hâlini kıyaslıyor. İkinci katman: aynı motorlar
bizim board'ımızın altında koşarken ne oluyor? Burası ekibin fiilen kullanacağı yapı.

**Kurulum:** aynı graf (`pipelines_store`'a kaydedilmiş), aynı düğüm bazlı simülasyon,
dört motor sırayla.

### Aynı graf, aynı hata → dört motor

`validate_schema` düğümü **kalıcı** patlatıldı (6 düğümlük ETL akışı):

| motor | süre | tamamlanan | başarısız | iptal | deneme |
|---|---:|---:|---:|---:|---:|
| own | **0,01 s** | 3/6 | 1 | 2 | 3 |
| temporal | 0,50 s | 3/6 | 1 | 2 | 3 |
| airflow | 4,75 s | 3/6 | 1 | 2 | 3 |
| celery | 12,39 s | 3/6 | 1 | 2 | 3 |

**Her sütun birebir aynı.** Board karar merciyse motor değiştirmek davranışı değiştirmiyor.

### Aynı graf, hatasız → çıktılar da aynı mı?

Asıl kanıt bu. Sayıların eşleşmesi beklenirdi; **üretilen metnin** eşleşmesi eşleşmiyor olabilirdi:

```
┌─ own       ✓ 6/6 · 0,05 s   {"dosya":"musteri_ozet.csv","satir":6,
┌─ temporal  ✓ 6/6 · 0,46 s    "csv":"musteri,adet,toplam\nm14,14,7031.0…"}
┌─ celery    ✓ 6/6 · 15,35 s      ← dördü de BYTE-BYTE aynı
┌─ airflow   ✓ 6/6 · 1,56 s
```

Airflow'unki özellikle anlamlı: onun çıktısı board'dan değil **kendi XCom'undan**
okunuyor (Airflow board'a hiç yazmıyor). Aynı metni vermesi, veri akışının o tarafta
da doğru kurulduğunun **bağımsız** kanıtı.

## 2.6 Düğüm bazında simülasyon — ölçümü mümkün kılan şey

Bu karşılaştırmaları yapabilmek için önce bir eksiği kapatmamız gerekti: **"şu düğüm
patlasın" demenin yolu yoktu.** `fail_at` tek bir global string'di ve **fonksiyon adı
ya da başlık alt-dizesiyle** eşleşiyordu.

Ölçülen sorun — aynı akışta iki `fetch_source` düğümü, hedef ikincisi:

| | çek A | çek B |
|---|---|---|
| eski `fail_at="fetch_source!"` | **failed** | **failed** ← ikisi birden |
| yeni `node_sim={"n2":…}` | `done` | **failed** ← yalnız hedef |

Altı mod: `normal · gecici · kalici · sonra · cokme · yavas`. Her motor ayrı bir
yürütme ortamı olduğu için ayrı taşıma kanalı gerekti:

| motor | kanal |
|---|---|
| own | doğrudan parametre |
| celery | `BRAIN_NODE_SIM` env (JSON) — worker ayrı süreç |
| temporal | `CTX["node_sim"]` — activity aynı süreçte |
| airflow | **DAG dosyasına gömülüyor** — ayrı süreç, ortak bellek yok |

**Yan fayda:** `crash_at` eskiden yalnız `own`'da çalışıyordu, celery/temporal onu
**sessizce yok sayıyordu** (raporlarda "ölçülmedi" diye işaretliydi). `cokme` modu
artık dört motorda da aynı yoldan geçiyor — bu boşluk kapandı.

## 2.7 Web paneli — ekibe gösterilebilir hâl

Ölçümler komut satırında doğruydu ama **gösterilebilir değildi**. Akış detay ekranı
karşılaştırma ekranına çevrildi (8030 → Akışlar → bir akışa tıkla):

```
[motor ▾] [▶ Koştur] [⚡ Hepsinde koştur]
graf — her düğüm TIKLANABİLİR
  └ tıkla → argümanları düzenle + "bu düğüme ne olsun?" seç
canlı log + karşılaştırma tablosu
```

Ayrıca sohbette workflow kurulduğunda (varsayılan açık) graf **bir kez** kurulup
dört motorda koşuyor ve **her birinin çıktısı ayrı ayrı** basılıyor.

---

# BÖLÜM 3 — Her motorun eksikleri

## Taban çizgisi (çıplak Python)

| eksik | sonucu |
|---|---|
| retry yok | ilk hatada akış durur — **ölçüldü** |
| kalıcı durum yok | çökme her şeyi siler — **ölçüldü** |
| görünürlük yok | "neredeyiz" sorulamaz |
| paralellik yok | sıralı koşar |
| zamanlama yok | — |
| at-most-once yok | — |

**Ne zaman yeter:** Tek seferlik script, geliştirme, prototip. Prod'da **hayır**.

## Celery

| eksik | sonucu |
|---|---|
| **durum görünürlüğü çok zayıf** | akış takılırsa hangi adımda olduğu bilinmiyor |
| **iptal/atlama kaydı yok** | batan zincirin kalanına ne olduğu belirsiz |
| checkpoint yok | retry task'ı baştan koşturur |
| DAG'ı gönderimde sabitler | koşullu/dinamik için `self.replace()` şart |
| **imza sızıntısı** | canvas şekli fonksiyon parametrelerini belirliyor |
| filesystem broker tuzağı | `data_folder_in == data_folder_out` şart, yoksa sessizce çalışmaz |
| en yavaş | 12,2 s (broker + worker açılışı) |

**Ne zaman yeter:** Bağımsız, kısa, çok sayıda iş (e-posta gönderimi, resim işleme). **Akış** değil, **iş kuyruğu** olarak.

## Airflow

| eksik | sonucu |
|---|---|
| **düğüm içi checkpoint yok** | LLM düğümü yarıda kalırsa çağrılar tekrarlanır (para) |
| graf parse zamanında sabit | yeni düğüm türü çalışırken eklenemez |
| **30 sn backoff** | hızlı akışlar için çok yavaş (ölçüldü: 62 sn) |
| koşullu dal iki kavram | `BranchPythonOperator` + `trigger_rule` unutulursa birleşim de skip olur |
| operasyonel maliyet | scheduler + webserver + metadata DB |
| kurulum kırılgan | constraint dosyası venv'i bozdu (`typing_extensions` düşürdü) — **yaşandı** |

**Ne zaman yeter:** Zamanlı, tekrarlayan, insan-gözetimli veri hatları. Prod'da **en olgun operasyon**.

## Temporal

| eksik | sonucu |
|---|---|
| "task/iş kalemi" kavramı yok | activity var, board benzeri bir model yok |
| workflow kodu sabit | determinizm şartı |
| **IO yalnız activity'de** | mimariyi bölmek zorundasın |
| operasyonel maliyet en yüksek | cluster ya da Temporal Cloud |
| seçilmeyen dal kaydı yok | "atlandı" diye bir durum yok |
| **kendi retry'ı bizimkiyle çakışabiliyor** | BUG 12 tam olarak buydu |

**Ne zaman yeter:** Uzun süren, çökmeye dayanması gereken, para/veri bütünlüğü kritik akışlar.

---

# BÖLÜM 4 — Prod'da hangisi kullanılır

Sektörde fiilî kullanım (bu POC'lerin değil, yaygın pratiğin özeti):

| motor | prod'da kullanılır mı | tipik yer |
|---|---|---|
| **taban çizgisi** | ❌ **hayır** | sadece script/prototip |
| **Celery** | ✅ **çok yaygın** | arka plan iş kuyruğu — e-posta, bildirim, resim işleme, rapor üretimi |
| **Airflow** | ✅ **veri mühendisliğinin standardı** | gecelik ETL, veri ambarı yükleme, raporlama hatları |
| **Temporal** | ✅ **artan** | ödeme akışları, sipariş yönetimi, uzun süren iş akışları, mikroservis orkestrasyonu |

**Üçü de prod'da kullanılıyor ama farklı işler için.** "Hangisi daha iyi" sorusu yanlış; doğru soru **"hangi iş"**.

## 4.1 İş türü → motor (karar tablosu)

| iş türü | örnek | doğru motor | neden (ölçümden) |
|---|---|---|---|
| **Zamanlı, tekrarlayan batch** | "her gece 02:00'de veri ambarını yükle" | **Airflow** | cron + **backfill/catchup** rakipsiz; 30 sn backoff burada doğru |
| **Bağımsız, çok sayıda kısa iş** | "kullanıcı kaydolunca hoş geldin e-postası" | **Celery** | native worker havuzu, yatay ölçekte en iyisi; DAG zaten gerekmiyor |
| **Uzun, para/veri kritik akış** | "ödemeyi al, stok düş, kargo çağır" | **Temporal** | **replay** — çökme sonrası devamı kutudan veren tek motor |
| **İstek başına üretilen graf** | sohbetten kurulan akış | **kendi çekirdeğimiz** | 0,05 sn; Celery'nin ~300 katı hızlı, kurulum sıfır |
| **Tek seferlik analiz** | script | **çıplak Python** | altyapı gereksiz |

## 4.2 Karar için üç soru

Ölçümlerden çıkan pratik eleme:

**1. Akış ne kadar sürüyor?**
- Saniyeler → **kendi çekirdeğimiz** (motor açılış maliyeti işin kendisinden büyük olmasın)
- Dakikalar/saatler → **Temporal** (çökme olasılığı gerçek hâle gelir)

**2. Zamanlı mı, istek başına mı?**
- Zamanlı + geçmişe dönük telafi lazım → **Airflow** (catchup)
- İstek başına yeni graf → Airflow **uymaz** (dosya + parse döngüsü)

**3. Yan etki var mı?** *(en belirleyici soru)*
- Yok (saf fonksiyon) → hepsi olur, hız seç
- Var (ödeme/e-posta/dosya yazma) → **Temporal** ya da idempotenslik disiplini şart.
  Ölçtük: retry düğümü **baştan** koşturuyor; iş bittikten sonra patlayan düğümün işi
  **2 kez** yapılıyor.

## 4.3 Ölçülen maliyet tablosu

Aynı 6 düğümlük akış, hatasız koşu:

| motor | süre | bunun ne kadarı işin kendisi |
|---|---:|---|
| own | **0,05 s** | ~%100 |
| airflow | 1,56 s | ~%10 (kalanı CLI açılışı ~1,5 sn) |
| temporal | 0,46 s | ~%20 (kalanı test server + worker açılışı) |
| celery | 15,35 s | ~%1 (**6 sn worker açılışı** + broker gecikmesi) |

**Okunuşu:** Celery'nin 15 saniyesinin neredeyse tamamı altyapı. Uzun süren işlerde
bu maliyet amorti olur; saniyelik işlerde ezici.

---

# BÖLÜM 5 — Ekip için karar

## 5.1 Bizim işimiz hangisine benziyor

Ölçümlerden çıkan profil:

| özellik | bizim durumumuz |
|---|---|
| akış süresi | **saniyeler** (5 düğüm ~0,01–15 sn) |
| düğüm süresi | fonksiyonlar **milisaniye**, LLM düğümleri 10–15 sn |
| akış kaynağı | **kullanıcı isteği** (sohbet), gecelik batch değil |
| akış sayısı | istek başına **yeni graf** |
| yan etki | şu an **yok** (saf fonksiyonlar) |
| çökme maliyeti | LLM çağrıları = para |
| ekip | staj/POC ölçeği, **DevOps kaynağı yok** |

**Bu profil Airflow'a uymuyor:** Airflow gecelik, sabit, uzun batch işler için. 30 sn backoff'u ve parse döngüsü bizim saniyelik, istek-başına akışlarımızda ters düşüyor. (Panelde `retry_delay=1s`'e indirerek kullanılabilir kıldık — ama bu Airflow'u kendi tasarım noktasının dışında kullanmak demek.)

**Celery'ye kısmen uyuyor** ama iki engel ölçüldü: **görünürlük eksiği** (sohbet arayüzünde "akış neresinde" göstermemiz gerekiyor, Celery bunu vermiyor) ve **6 sn worker açılışı** her koşuda ödeniyor — 15 saniyenin neredeyse tamamı altyapı.

**Temporal'a iyi uyuyor** ama operasyonel maliyeti staj/POC ölçeğinde ağır. Ayrıca ölçtük ki kendi retry katmanı bizimkiyle çakışabiliyor (BUG ①) — birleşimi düşünmeden kullanmak yeni hata sınıfı doğuruyor.

## 5.2 Önerim: iki katmanlı karar

**Bugün için: kendi çekirdeğimiz (board + SQLite).**

Gerekçe ölçümlere dayanıyor:
- **Hız:** 0,01 sn — Celery'nin 1.200 katı hızlı. İstek-başına akışta bu belirleyici.
- **Kurulum sıfır:** DevOps kaynağımız yok. Airflow kurarken venv'in bozulmasını **bugün yaşadık**.
- **Görünürlük:** Sohbet arayüzüne board durumunu akıtabiliyoruz; Celery'de bu yok.
- **İptal zinciri + checkpoint:** İkisini de yazdık ve ölçtük.

**Bedeli dürüstçe:** dayanıklılık garantisi **kendi kodumuz kadar**. 12 hatanın 8'i bu katmanda çıktı. Bu bir risk, ama ölçülmüş ve kapatılmış bir risk.

**Yarın için: Temporal'a geçiş yolu açık kalsın.**

Ne zaman geçilmeli:
- Akışlar dakikalar/saatler sürmeye başlarsa
- **Yan etkili düğüm** eklenirse (ödeme, e-posta) — çökme sonrası devam kritik olur
- Birden çok makineye yayılmak gerekirse
- Para/veri bütünlüğü kritik hâle gelirse

Geçiş bedeli düşük, çünkü board zaten motordan bağımsız — **bunu ölçtük**: yürüten üç motorda deneme sayısı, durum ve sonuç birebir aynı çıktı.

**Airflow: yürütücü değil, ihracat hedefi olarak kalsın.** "Bu akışı gecelik batch'e çevir" istendiğinde `export_airflow_dag()` var. Günlük motorumuz olmamalı.

## 5.3 Ama önce şu üçünü düzeltmek gerek

Ölçümler üç açık gösterdi:

**1. Temporal'da çift retry katmanı.** Temporal'ın `RetryPolicy`'si ile bizim breaker'ımız aynı task üstünde çalışıyor; BUG 12 bunun bedeliydi. **Öneri:** Temporal'da `maximum_attempts=1` — retry otoritesi tek olsun (board).

**2. Lease/heartbeat tutarsız.** `LEASE_SECONDS = 30` gerekçesiz bir sabit; fonksiyon düğümleri hiç nabız atmıyor. Bugün zararsız (fonksiyonlar 1 ms) ama **tek bir LLM turu 30 sn'yi aşabilir** — o zaman çift yürütme riski doğar.

**3. Backoff yok.** Dış servise giden düğüm eklenirse 0 sn arayla 3 kez vurulur. Airflow'un 30 sn'si abartı, ama 0 sn de yanlış.

## 5.4 Yol boyunca bulunan hatalar (asıl deneyim)

Motorları kıyaslarken bulduklarımız. Çoğu **görünürde çalışan** kodda çıktı —
kıyaslama yapmasak fark etmeyecektik.

## Motor entegrasyonundan çıkanlar

**① Temporal'da geçici hata KALICIYA dönüşüyordu.** Aynı hata own'da 1 retry'la
toparlanırken Temporal'da düğümü batırıyordu (3 deneme → `failed`, 2/5 tamamlandı).
Zincir: `board.fail()` claim'i temizliyor → Temporal **aynı activity'yi bayat claim'le**
yeniden çağırıyor → iş **başarıyla koşuyor** ama `complete()` fencing'e takılıp
**sonuç çöpe gidiyor** → task hâlâ `ready` → 3 turda breaker doluyor.
**İki kez başaran düğüm `failed` işaretleniyordu.**
> Ders: **iki durable katmanı üst üste bindirmek bedava değil.** Birinin "sahibi ben
> değilim" demesi gerekiyor. Öneri: Temporal'da `maximum_attempts=1`.

**② Temporal'ın deneme sayacı her çağrıda sıfırlanıyordu.** `activity.info().attempt`
her **yeni** activity çağrısında 1'e döner — bu Temporal'ın doğru semantiği, ama bizim
döngümüz her board turunda yeni bir çağrı açtığı için turlar arası denemeyi hiç
görmüyordu. Sayaç board'dan alınmak zorunda kaldı.

**③ Celery'de çöken worker kurtarılmıyordu.** Worker ayrı süreçte öldüğünde task
`running` asılı kalıyor ve Celery'nin haberi olmuyor. own bunu `WorkerCrash` yakalayıp
yapıyordu; celery'de sweep dispatcher'a düşüyordu — yoktu. Bekleme döngüsüne
`recover_stale()` eklendi.

**④ Celery'de fencing sessizce devre dışıydı.** `task` dict'i claim'den **önce**
alınıyordu → `claim_lock=None` → sahiplik kontrolü hiç çalışmıyordu.

**⑤ `crash_at` celery ve temporal'da yok sayılıyordu.** Parametre kabul ediliyor,
kullanılmıyordu. Raporlarda "çökme kurtarma o iki motorda ölçülmedi" diye
işaretlemiştik; `node_sim` ile kapandı.

**⑪ Celery düğümü FAZLADAN bir kez koşturuyordu — düğüm bazlı görünüm ortaya çıkardı.**
Toplam "deneme" sütunu bunu gizliyordu; düğüm bazlı `×N` eklenince görüldü:

```
geçici hata → validate_schema kaç kez koştu?
own ×2   ·   temporal ×2   ·   airflow ×2   ·   celery ×3   ← fazladan
```

Sebep: **iki bağımsız yeniden-dağıtım yolu.** `celery_worker` hata alınca hem
`board.fail()` çağırıyor (task `ready`ye dönüyor, dispatcher'ın dalga döngüsü onu
yeniden kuyruğa atıyor) hem de `self.retry()` ile Celery'nin kendi retry'ını
tetikliyordu. Düğüm iki yoldan da yeniden dağıtılıyordu.

Bu, **BUG ①'in (Temporal) Celery karşılığı** — aynı ders, farklı belirti: Temporal'da
sonuç çöpe gidiyordu, Celery'de iş fazladan bir kez yapılıyordu. Yan etkili bir
düğümde bu "e-posta 3 kez gitti" demek.

**Düzeltme:** `self.retry()` kaldırıldı, retry otoritesi board'a bırakıldı. Ama bu tek
başına yetmedi — kaldırınca `self.request.retries` hep 0 kaldığı için geçici hata
sonsuza dek "ilk deneme" sanıldı ve düğüm `failed ×3` oldu. **BUG ②'nin aynısı.**
Sayaç board'dan alınınca dördü de hizalandı:

```
geçici : own ×2 · temporal ×2 · celery ×2 · airflow ×2      ✓
kalıcı : dördü de failed ×3 + ardıllar cancelled/upstream_failed  ✓
```

> Üç motorda da aynı kalıp çıktı: **motorun kendi retry'ı bizim breaker'ımızla
> çakışıyor ve deneme sayacı board'dan gelmek zorunda.** Bir katman "sahibi ben
> değilim" demeli — bu, çalışmanın en tekrar eden dersi.

**⑫ Tool-trace paneli yanlış şeyi ölçüyordu — "indirgeme" EKSİ çıktı.**
Panelde `8.892 → 9.040 token · %-1,7` göründü: indirgeme adı altında **büyüme**.
İki ayrı hata üst üste binmişti.

*(a) Yanlış nüfus.* Panel akış (DAG) düğümlerini de listeliyordu. Ama bir DAG
düğümünün çıktısı **board'a ve ardıl düğüme** gider; sohbet asistanının context'ine
hiç girmez. Tool-trace'in konusu yalnız **LLM context'ine giren** trafiktir — DAG
düğümü orada işi olmayan bir satır. Olaylar artık `kaynak` ile etiketleniyor
(`"sohbet"` / `"akis"`) ve panel yalnız `sohbet` olanları alıyor.

*(b) Yanlış ölçüt.* Ölçüm `json.dumps(res_d)` üzerinden yapılıyordu — o dict
`rows`'un **tamamını** taşır. Oysa context'e giren şey `rows`/`text` atılmış `ozet`.
Yani panel, LLM'e hiç gitmeyen veriyi "LLM'e giden" diye sayıyordu:

```
extract_records — ham 4.446 token
  ESKİ ölçüm (rows dahil, context'e GİRMEYEN)   4.520 token   %-1,7   ← anlamsız
  YENİ ölçüm (context'e gerçekten giren özet)      65 token   %98,5   ← doğru
```

Not: akış tarafındaki `%-1,7` **hata değil** — `extract_records` gerçekten indirgeme
yapmayan bir *taşıyıcı* düğüm, `rows`'u aynen ardıla devrediyor. Koşu logu artık bunu
"indirgeme yok — taşıyıcı düğüm" diye yazıyor, sahte bir yüzde uydurmuyor.

> Ders: **bir metrik, ölçtüğünü iddia ettiği şeyin geçtiği yerden alınmalı.** Burada
> ölçüm noktası (`res_d`) ile iddia noktası (`msgs`'e eklenen `out_txt`) farklıydı;
> arada `rows` vardı ve sayı sessizce ters döndü.

## Simülasyon katmanından çıkanlar

**⑥ Çökme sonsuz döngüye giriyordu.** İlk uygulamada `cokme` **her denemede**
tetikleniyordu → own'da 10 çökme üst üste, celery/temporal'da düğüm asılı kaldı.
Çözüm: checkpoint'in varlığı **süreçler arası geçerli bir "bir kez çöktü" işareti**
olarak kullanıldı. Artık tek atış: çök → kurtar → tamamla.

## Arayüz/veri katmanından çıkanlar

**⑦ Sohbette kurulan akış "Akışlar" ekranında GÖRÜNMÜYORDU.** Paneli akış listesine
bağlarken çıktı: `listing()` **dosya adına** göre sıralıyordu, ama id
`int(time.time()*1000) % 1e8` ile üretiliyor ve bu sayaç **~27,7 saatte bir başa
sarıyor**. Sarma sonrası kaydedilen akış listenin dibine düşüyor, ilk 50'lik pencereye
hiç giremiyordu. Sıralama gerçek zaman damgasına çevrildi.
> Ders: **türetilmiş bir alanı sıralama anahtarı yapma.** Gerçek zaman damgası varken
> id'ye güvenmek sessiz veri kaybı üretiyor.

**⑧ `on_event` celery ve temporal'a geçirilmiyordu.** `run_saved` bunları çağırırken
olay köprüsünü vermiyordu → o iki motorda arayüze **hiç canlı log akmıyordu**.

## Kurulum/ortam

**⑨ Airflow kurulumu venv'i bozdu.** Constraint dosyası (Python 3.11 için)
`typing_extensions`'ı düşürdü, `pydantic` kırıldı. Düzeltildi ama **not edilmeli**:
Airflow'un bağımlılık pinleri agresif; ayrı venv daha güvenli olurdu.

**⑩ Airflow eşzamanlılığı yok.** `SequentialExecutor` + sqlite → iki `dags test`
paralel koşarsa **"database is locked"**. Panelde kilitle serileştirildi.

## Kalıp

15 hatanın çoğu iki sınıfa düşüyor:

| sınıf | örnek | neden gözden kaçıyor |
|---|---|---|
| **"mimari değişti, çağıran güncellenmedi"** | ③④⑤⑧ | görünürde çalışıyor; yalnız o yol koşturulunca çıkıyor |
| **"türetilmiş değere güvenmek"** | ②⑦ | doğru göründüğü sürece kimse sorgulamıyor |

---

# BÖLÜM 6 — Bu çalışmanın öğrettiği altı şey

**1. "Dinamik graf" bir ayrışma ekseni değil.** Dört motorun **hiçbiri** çalışma anında yeni düğüm türü ekleyemiyor. Üstelik ölçtük: kayıtlı 354 düğümün **sıfırı** çalışma anında üretilmiş. Ajan zaten grafı baştan kurup geri çekiliyor.

**2. Motorlar "yapabilme"de değil, "kaydetme"de ayrışıyor.** Dördü de dört deseni yaptı. Fark: Airflow koşmayan düğümü bile kaydediyor (`skipped`, `upstream_failed`), diğerlerinde o düğüm hiç var olmuyor.

**3. Retry ve DAG ifadesi çözülmüş problemler.** Üçü de kutudan veriyor. Bunları yeniden yazmak, iki katmanın çakışması riskini getiriyor.

**4. Tek net teknik üstünlük: Temporal'ın replay'i.** Çökme sonrası devamı gerçekten veren tek motor.

**5. Aynı graf dört motorda AYNI çıktıyı üretiyor — ölçüldü.** Sayıların eşleşmesi
beklenirdi, üretilen metnin byte-byte eşleşmesi eşleşmiyor olabilirdi. Bu, board'un
motordan gerçekten bağımsız olduğunun en güçlü kanıtı ve **geçiş maliyetinin düşük
olduğunu** gösteriyor.

**6. Kıyaslama kendi başına bir hata bulma yöntemi.** 15 hatanın çoğu kıyaslama
yapmasak bulunamazdı — çünkü tek motorda "çalışıyor" görünüyorlardı. İki uygulamayı
aynı işe koşturmak, tek uygulamayı test etmekten farklı bir şey buluyor.

---

# BÖLÜM 7 — Açık kalanlar

| konu | durum |
|---|---|
| Airflow **gerçek scheduler** ile ölçülmedi | `dags test` ile koştu; scheduler + webserver ile üretim davranışı farklı olabilir |
| Celery **gerçek broker** ile ölçülmedi | filesystem broker kullanıldı; Redis/RabbitMQ ile hız farklı olur |
| Temporal **gerçek cluster** ile ölçülmedi | `WorkflowEnvironment` (test ortamı) kullanıldı |
| **yük testi yok** | tek akış ölçüldü, eşzamanlı 100 akış değil |
| **backoff yok** (own, celery) | dış servis düğümü eklenirse 0 sn arayla 3 kez vurur |
| **lease 30 sn gerekçesiz** | fonksiyon düğümleri nabız atmıyor; uzun düğüm eklenirse çift yürütme riski |
| ajan düğümünde simülasyon kısmi | `cokme` çalışıyor; `gecici/kalici` ajan dalında eski semantikte |

**Kapanmış olanlar** (önceki raporlarda açıktı):
- ~~Airflow hiç ölçülmedi~~ → kuruldu, üç desen + hata senaryoları koşturuldu
- ~~çökme celery/temporal'da ölçülmedi~~ → `node_sim` `cokme` ile dört motorda da ölçüldü
- ~~scheduling yok~~ → `scheduler.py` (cron + claim/lease), 43/43

Bu ölçümler **doğru yönü** gösteriyor ama üretim sayıları değil. Karar için yeterli, kapasite planlaması için değil.

---

# BÖLÜM 8 — Test envanteri

| paket | ne ölçüyor | sonuç |
|---|---|---|
| `saf-motorlar/kiyas.py` | dört saf motora 8 soru | ✓ |
| `saf-motorlar/desen_*.py` | üç workflow deseni × motor | ✓ |
| `test_node_sim.py` | düğüm bazlı simülasyon, dört motor | **27/27** |
| `test_hata.py` | hata/çökme/bozuk girdi/altyapı | **54/54** |
| `test_tasklife.py` | task yaşam döngüsü | **42/42** |
| `test_zamanlama.py` | cron, claim yarışı, lease | **43/43** |
| `test_retry.py` | retry anatomisi, yan etki tekrarı | ✓ |
| `test_backend_hata.py` | dört motor aynı hataya tepki | ✓ |
| `test_devam.py` | çökme sonrası devam, 3 seviye | ✓ |
| `test_concurrency.py` | 6 gerçek süreç, at-most-once | ✓ 5,0× |

**Kritik ölçümler:**
- 6 süreç · 24 task · **0 çift claim** · 4,7–5,0× hızlanma
- Compaction: **30/30 çift bütünlüğü**, kritik bilgi tetiklenen 15 koşunun 8'inde korundu
- Retry: geçici hata 2 deneme, kalıcı 3 (breaker), **backoff yok**
- Yan etki: iş bittikten sonra patlayan düğümün işi **2 kez** yapılıyor

---

## Koşturma

### Web paneli — ekibe göstermek için en kolay yol

```bash
.venv/bin/python demo-brain-agent/chat_server.py     # → http://127.0.0.1:8030
```

1. Sohbete bir iş yaz (`etl akışı üret`) → graf kurulur, **dört motorda koşar**,
   her birinin çıktısı ayrı basılır (üst bardaki **karşılaştır** seçicisi ile kapatılabilir)
2. **Akışlar** → bir akışa tıkla → graf çizilir, **düğüme bas** → ne olacağını seç →
   **▶ Koştur** ya da **⚡ Hepsinde koştur**

### Saf motorlar (board'suz)

```bash
.venv/bin/python saf-motorlar/desenler.py        # referans (çıplak Python)
.venv/bin/python saf-motorlar/desen_temporal.py  # üç desen
.venv/bin/python saf-motorlar/desen_celery.py
.venv/bin/python saf-motorlar/kiyas.py           # dört motora 8 soru

export AIRFLOW_HOME=$PWD/saf-motorlar/airflow_home PYTHONPATH=$PWD/saf-motorlar
.venv/bin/airflow dags test desen_zincir         # tarih VERME → benzersiz run_id

SAF_HATA="z7"  ...   # geçici hata    SAF_HATA="z7!" ...   # kalıcı
```

### Board'lu ölçümler

```bash
.venv/bin/python demo-brain-agent/test_node_sim.py       # 27/27
.venv/bin/python demo-brain-agent/test_backend_hata.py   # dört motor
.venv/bin/python demo-brain-agent/airflow_runner.py --sim auto
```

---

**İlgili raporlar:**
`saf-motorlar-poc-raporu.md` (saf POC ayrıntısı) ·
`motorlar-saf-hali-ve-eklediklerimiz.md` (saf hâl ↔ bizim eklediğimiz) ·
`motor-paneli-poc.md` (web paneli) ·
`task-management-karsilastirma-ve-test-raporu.md` (board'un kendisi) ·
`hata-dayanikliligi-test-raporu.md` · `retry-olcum-raporu.md` · `zamanlama-cron-raporu.md`
