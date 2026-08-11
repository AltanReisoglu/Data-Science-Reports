# Saf Motorlar POC — Hiçbirine Bizim Katmanımız Eklenmeden

**Amaç:** Dört motoru **kutudan çıktığı hâliyle** aynı işe koşturmak. Board yok, düzeltme yok, yama yok. Eksiklik ve fazlalıklar kendiliğinden ortaya çıksın.
**Kod:** `saf-motorlar/` — `isakisi.py` (ortak iş) · `saf_hermes.py` · `saf_celery.py` · `saf_temporal.py` · `saf_airflow_dag.py` · `kiyas.py` (ölçüm)
**Tarih:** 10 Ağustos 2026

Bu POC iki iddiamı çürüttü. Önce onları yazıyorum.

---

# 0 · Önce iki düzeltme

## 0.1 "Dinamik graf" iddiası — ÇÜRÜDÜ

Şimdiye kadar Airflow'u şöyle eledim: *"Graf parse zamanında donuyor, oysa ajanımız çalışma anında task üretiyor."*

Ölçtüm:

```
Kayıtlı 354 düğümün kaç tanesi ÇALIŞMA ANINDA üretilmiş?
  toplam düğüm        : 354
  spawn_task ile      :   0     ← çalışma anında üretilen
  planlama turunda    : 354     ← hepsi baştan
```

**Sıfır.** `spawn_task` yeteneği var, testi de var (çalıştığı kanıtlı), ama **hiçbir gerçek koşuda tetiklenmemiş.**

Yani ajan zaten şunu yapıyor: **grafı baştan kurar, sonra geri çekilir.** Tam da Airflow'un varsaydığı model. Dinamiklik avantajı olarak yazdığım şey pratikte kullanılmamış bir yetenek.

**Doğrusu:** Bizim gerçek ihtiyacımız "çalışma anında düğüm ekleme" değil, **istek başına yeni graf üretme**. Airflow'da bunun bedeli dosya yazma + parse döngüsü; diğer üçünde bedava. Fark bu — "donmuş graf" değil, **graf üretmenin maliyeti**.

## 0.2 "Motorların hiçbiri DAG vermiyor" iddiası — ÇÜRÜDÜ

Bunu da fazla söylemişim. Saf POC'leri yazınca gördüm: **dördü de grafı ifade edebiliyor.**

| motor | grafı nasıl ifade ediyor | okunabilirlik |
|---|---|---|
| **temporal** | düz Python: `asyncio.gather` + `await` | **en okunur** |
| **airflow** | dosyada `cek >> tara`, `[tara, test] >> esle` | okunur ama donmuş |
| **celery** | canvas: `chord([chain(cek, tara), test], chain(esle, rapor))` | **imzalara sızıyor** |
| **hermes type (saf)** | fonksiyon çağrı sırası | graf örtük, kayıt yok |

Temporal'ınki gerçekten zarif — graf ayrı bir yapı değil, dilin kendi akışı:

```python
tara, test = await asyncio.gather(
    dal_kod(),                                    # cek → tara zinciri
    workflow.execute_activity("test", ...))       # paralel dal
esle = await workflow.execute_activity("esle", {"tara": tara, "test": test})
```

Celery'ninki ise bir bedel getiriyor: chain'de önceki dönüş **sonrakinin ilk argümanı** olur, chord'da grup sonuçları **liste** gelir. Yani **graf şekli fonksiyon imzalarına sızıyor** — `t_esle(self, header_sonuclari)` gibi. Board'da böyle bir sızıntı yok, `parents` ayrı bir alan.

---

# 1 · Ortak iş

Dört motor da **aynı beş adımı aynı bağımlılıkla** koşturuyor:

```
cek ─────► tara ──┐
                   ├──► esle ──► rapor
test ─────────────┘
```

`isakisi.py`'deki fonksiyonlar **saf**: yan etkisiz, deterministik, birbirini tanımıyor. Veri akışını her motor kendi yöntemiyle çözüyor — asıl kıyas noktalarından biri bu.

Hata enjeksiyonu ortak: `SAF_HATA="tara"` → geçici, `SAF_HATA="tara!"` → kalıcı.

Doğruluk kıstası ortak: rapor `4 eşleşme` ve `240 test` yazmalı. Yazmıyorsa **veri akışı kopuk** demektir.

---

# 2 · Ölçümler

## S1 · Mutlu yol

| motor | sonuç | süre | çıktı |
|---|---|---:|---|
| hermes type (saf) | ✓ doğru | **0,03 s** | 4 eşleşme · 240 test · 267B |
| temporal | ✓ doğru | 0,49 s | aynı |
| celery | ✓ doğru | **12,23 s** | aynı |
| airflow | — | — | koşturulamadı (kurulu değil) |

Üçü de **doğru sonuç** üretti — veri akışı hiçbirinde kopmadı.

## S2 · Geçici hata (en öğretici ölçüm)

`tara` ilk denemede patlıyor, sonra düzeliyor:

| motor | sonuç | süre |
|---|---|---:|
| **hermes type (saf)** | **✗ TOPARLAYAMADI** | 0,05 s |
| celery | ✓ TOPARLADI | 13,24 s |
| temporal | ✓ TOPARLADI | 0,48 s |

```
hermes type:  ✗ HATA: RuntimeError: geçici hata: tara (deneme 1)
              → Retry YOK. Akış burada durur, kalan adımlar hiç koşmaz.
```

**Bu, bizim board'ımızın en çok emek verdiğimiz özelliğinin — retry/breaker — Celery ve Temporal'da zaten var olduğunu gösteriyor.** Saf hermes type'ta yok, çünkü orada hiçbir şey yok.

## S3 · Kalıcı hata

`tara` her denemede patlıyor:

| motor | davranış |
|---|---|
| hermes type (saf) | ilk hatada durur, hiç bilgi yok |
| celery | **3 kez dener** (`max_retries=2` + ilk), sonra akış hata ile biter |
| temporal | RetryPolicy tükenir, kalan activity'ler **hiç çağrılmaz** |

Üçünde de ortak eksik: **"kalan adımlara ne oldu" sorusunun cevabı yok.**

```
celery   : → Kalan adımlara ne olduğu BİLİNMİYOR: iptal kaydı yok
temporal : → Kalan activity'ler HİÇ çağrılmadı (gövde exception ile kesildi);
             'iptal edildi' diye bir kayıt yok
```

**Airflow bu konuda üçünden iyi:** varsayılan `trigger_rule=all_success` ile ardıl düğümler `upstream_failed` durumuna geçer — yani "koşmayacak" **açık bir durum** olarak kaydedilir. Bizim `cancelled`'ımızın kutudan gelen hâli.

## S4 · Çökme — tek ayrışma noktası

| motor | çökme sonrası |
|---|---|
| hermes type (saf) | **hiçbir şey kurtarılmıyor** — bellekteki 3 sonuç gitti, her şey baştan |
| celery | mesaj yeniden teslim (`acks_late`) ama **task baştan koşar** |
| **temporal** | **replay → tamamlanmış activity'ler ATLANIR**, kaldığı yerden devam |
| airflow | XCom kalıcı → tamamlanan task'lar tekrar koşmaz, ama **task içi checkpoint yok** |

**Temporal burada tek başına ayrışıyor** — dört motor içinde çökme sonrası gerçek devamı kutudan veren tek motor.

## S5–S8 · Yapısal

| soru | hermes type (saf) | celery | temporal | airflow |
|---|---|---|---|---|
| **"şu an neredeyiz?"** | yok — bellekte | **zayıf** — chord bekliyor, hangi adımda bilinmiyor | **var** — event history | **var** — metadata DB + UI |
| **çalışma anında graf değişimi** | kodu değiştir | canvas `apply_async`'te sabitlenir | workflow kodu sabit | parse zamanında sabit |
| **zamanlama (cron)** | **yok** | Celery Beat (backfill yok) | Schedules (+backfill) | **en güçlü** (+catchup) |
| **graf ifadesi** | çağrı sırası | canvas — **imzalara sızıyor** | düz Python — **en okunur** | dosyada `>>` — donmuş |

**Hiçbiri çalışma anında graf değiştiremiyor.** Üçü de grafı bir noktada sabitliyor. Bu da 0.1'deki düzeltmeyi destekliyor: dinamiklik bir ayrışma ekseni değil, çünkü **kimsede yok**.

---

# 3 · Her motorun eksiği ve fazlası

## hermes type — SAF hâli

**Fazlası:** Hiçbiri. En hızlı (0,03 s), kurulum sıfır, bağımlılık sıfır.

**Eksikleri (ölçüldü):**
- ❌ retry yok — ilk hatada akış durur
- ❌ kalıcı durum yok — çökme her şeyi siler
- ❌ görünürlük yok — "neredeyiz" sorulamaz
- ❌ paralellik yok — sıralı koşar
- ❌ zamanlama yok
- ❌ at-most-once yok

> **Board'ı yazma sebebimiz tam olarak bu liste.** Saf hâli bir taban çizgisi, motor değil.

## Celery

**Fazlası:**
- ✅ retry kutudan (`max_retries`) — ölçüldü, toparladı
- ✅ at-least-once teslim (`acks_late`)
- ✅ canvas ile graf ifadesi (chain/group/chord)
- ✅ gerçek dağıtık worker havuzu
- ✅ Celery Beat ile zamanlama

**Eksikleri:**
- ❌ **durum görünürlüğü yok** — dört motor içinde en zayıfı. Akış takılırsa hangi adımda olduğu bilinmiyor.
- ❌ checkpoint yok — retry baştan koşturur
- ❌ iptal kaydı yok — batan dalın ardılları sessizce koşmaz
- ⚠ **canvas imzalara sızıyor** — graf şekli fonksiyon parametrelerini belirliyor
- ⚠ **en yavaş** — 12,2 s (broker + worker açılışı)

## Temporal

**Fazlası:**
- ✅ **çökme sonrası devam** — dört motor içinde tek (replay)
- ✅ retry + backoff kutudan (200 ms)
- ✅ event history — tam denetim izi
- ✅ **grafı en okunur ifade eden** (düz Python, `gather`/`await`)
- ✅ Schedules + backfill
- ✅ hızlı (0,49 s)

**Eksikleri:**
- ❌ "task" kavramı yok — iş kalemi modeli yok, activity var
- ❌ workflow kodu sabit — determinizm şartı
- ⚠ operasyonel maliyet en yüksek (cluster/Cloud)
- ⚠ IO/rastgelelik yalnız activity'de olabilir

## Airflow

**Fazlası:**
- ✅ **zamanlamada rakipsiz** — cron + backfill/catchup
- ✅ operatör UI'si + metadata DB
- ✅ **iptal semantiği kutudan** — `upstream_failed`
- ✅ XCom ile kalıcı veri akışı
- ✅ düğüm seviyesinde devam (tamamlananlar tekrar koşmaz)

**Eksikleri:**
- ❌ graf **parse zamanında donuk** — ama 0.1'de gördük ki bu bizi bağlamıyor
- ❌ **düğüm içi checkpoint yok** — ajan düğümü yarıda kalırsa LLM çağrıları tekrarlanır
- ⚠ operasyonel maliyet yüksek (scheduler + webserver + DB)
- ⚠ **ölçülemedi** — kurulu değil

---

# 4 · Peki board ne ekliyor — dürüst liste

Saf POC'ler bir şeyi net gösterdi: **board'ın eklediği şeylerin çoğu, bazı motorlarda zaten var.**

| board'ın verdiği | hermes type (saf) | celery | temporal | airflow |
|---|---|---|---|---|
| retry / breaker | ❌ ekliyoruz | ✅ **zaten var** | ✅ **zaten var** | ✅ **zaten var** |
| DAG ifadesi | ❌ ekliyoruz | ✅ zaten var (canvas) | ✅ zaten var | ✅ zaten var |
| durum görünürlüğü | ❌ ekliyoruz | ❌ **ekliyoruz** | ✅ zaten var | ✅ zaten var |
| çökme sonrası devam | ❌ ekliyoruz | ❌ **ekliyoruz** | ✅ zaten var | kısmen |
| iptal zinciri | ❌ ekliyoruz | ❌ **ekliyoruz** | ❌ **ekliyoruz** | ✅ zaten var |
| at-most-once | ❌ ekliyoruz | at-least-once | ✅ zaten var | ✅ zaten var |
| zamanlama | ❌ ekledik | zayıf | ✅ zaten var | ✅ zaten var |
| çalışma anında task | ❌ | ❌ | ❌ | ❌ |

**Board'ın gerçekten benzersiz katkısı üç madde:**

1. **Dört motora tek arayüz.** Aynı graf, aynı davranış, motor değiştirilebilir. Ölçtük: yürüten üç motorda deneme sayısı/durum/sonuç birebir aynı. Bu **meta** bir fayda ama gerçek.

2. **Celery'nin görünürlük boşluğunu kapatmak.** Saf Celery'de "akış neresinde takıldı" sorusunun cevabı **yok**. Board bunu veriyor.

3. **İptal zincirini açık durum yapmak.** Temporal ve Celery'de batan dalın ardılları "hiç çağrılmadı" — kayıt yok. Board `cancelled` yazıyor. (Airflow'da `upstream_failed` olarak zaten var.)

**Fazladan yaptıklarımız:**
- Temporal'da retry'ı **ikinci kez** uyguluyoruz — BUG 12 tam olarak bunun bedeliydi
- Temporal'da at-most-once'ı **ikinci kez** uyguluyoruz — task queue zaten veriyor
- Çalışma anında task üretme yeteneği yazdık — **hiç kullanılmadı** (0/354)

---

# 5 · Bu POC'nin gösterdiği üç şey

**1. "Dinamiklik" bir ayrışma ekseni değil.** Dört motorun **hiçbiri** çalışma anında graf değiştiremiyor, ve bizim ajanımız da bunu hiç istememiş (0/354). Airflow'u bu gerekçeyle elemek yanlıştı.

**2. Retry ve DAG ifadesi çözülmüş problemler.** Celery ve Temporal ikisini de kutudan veriyor. Bunları yeniden yazmak, iki katmanın çakışması riskini getiriyor (BUG 12).

**3. Gerçek ayrışma noktası: çökme sonrası devam.** Bu ölçümdeki tek net üstünlük Temporal'ın. Diğer üçünde ya hiç yok (saf hermes), ya task seviyesinde (Celery/Airflow baştan koşar).

---

# 6 · Bu ne demek — bir sonraki adım için

Bu POC'lerin sonucunu ciddiye alırsak iki seçenek var:

**(a) Board'ı inceltmek.** Motor zaten veriyorsa (retry, at-most-once) board o motorda devre dışı kalsın. Kazanç: BUG 12 sınıfı çakışmalar biter. Bedel: board backend'e göre koşullu hâle gelir, "tek arayüz" faydası zayıflar.

**(b) Board'ı tek otorite tutmak, motorun kendi katmanını kapatmak.** Ör. Temporal'da `maximum_attempts=1` — retry kararı yalnız board'da. Kazanç: tek otorite, çakışma yok. Bedel: motorun sınanmış retry'ını kullanmıyoruz.

Ölçüm (b)'yi destekliyor: BUG 12'nin kökü iki retry katmanının aynı task üstünde çalışmasıydı. **Bir katman "sahibi ben değilim" demeli.**

---

## Koşturma

```bash
.venv/bin/python saf-motorlar/isakisi.py        # ortak iş, tek başına
.venv/bin/python saf-motorlar/saf_hermes.py     # çıplak Python
.venv/bin/python saf-motorlar/saf_celery.py     # canvas
.venv/bin/python saf-motorlar/saf_temporal.py   # workflow + activity
.venv/bin/python saf-motorlar/kiyas.py          # dördüne aynı 8 soru

SAF_HATA=tara  .venv/bin/python saf-motorlar/saf_celery.py   # geçici hata
SAF_HATA=tara! .venv/bin/python saf-motorlar/saf_temporal.py # kalıcı hata
.venv/bin/python saf-motorlar/saf_hermes.py --cokme tara     # çökme
```

**Airflow koşturulamadı** — kurulu değil. `saf_airflow_dag.py` gerçek bir DAG dosyası; `pip install apache-airflow` + `airflow dags test saf_denetim` ile ölçülebilir. Bu rapordaki Airflow satırları **beyan edilen ayarlara ve bilinen semantiğe** dayanıyor, ölçüme değil.

**İlgili:** `motorlar-saf-hali-ve-eklediklerimiz.md` (board'lu hâl) · `task-management-karsilastirma-ve-test-raporu.md`
