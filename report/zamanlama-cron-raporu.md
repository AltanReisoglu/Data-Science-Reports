# Zamanlama (cron) — koçun 6. ekseni kapandı

**Eklenen:** `demo-brain-agent/scheduler.py` · **Test:** `test_zamanlama.py` → **43/43**
**Kaynak fikir:** Macro'nun `services/scheduled_action`'ı (bkz. `macro-analiz-bizim-caseler.md`)

---

## Neden

Testlerin her turunda scheduling'i **"yok"** diye belgeledik. `test_tasklife.py`'de bunu doğrulayan bir kontrol bile vardı:

```
✓ zamanlanmış koşu (cron/interval) UYGULANMIŞ mı  beklenen=False  gerçek=False
  → HAYIR: scheduling bu sistemde hiç yok (koçun 6 ekseninden biri açık)
```

Macro analizinde bunun kopyalanabilecek kadar sade bir çözümü olduğunu gördük. Eklendi; o kontrol artık varlığı doğruluyor.

---

## Ne eklendi

Zamanlama **kayıtlı bir akışı** (pipeline) cron ifadesiyle koşturur. Yeni bir yürütme yolu değil — mevcut `run_saved()`'ı tetikleyen bir katman.

```
cron ifadesi ──► ScheduleStore ──► Poller ──► run_saved() ──► board
                 (SQLite)          (thread)    (mevcut motor)
```

| parça | ne yapar |
|---|---|
| `cron_ayristir` / `sonraki_kosu` | bağımlılıksız 5 alanlı cron ayrıştırıcı |
| `ScheduleStore` | zamanlama + koşu geçmişi deposu, atomik claim |
| `kosturt` | kayıtlı akışı koşturur, sonucu geçmişe yazar |
| `tur` | tek yoklama turu: vakti gelenleri claim et ve koştur |
| `Poller` | arka plan iş parçacığı, sohbet sunucusuyla aynı süreçte |

### Cron ayrıştırıcı — neden kendi yazdım

`croniter` kurulu değil ve POC'nin geri kalanı bağımlılıksız. 5 alan (`dk sa gün ay hafta`), `*` · `a,b` · `a-b` · `*/n` · `a-b/n` destekli. Cron'un o meşhur kuralı da uygulanmış: **gün ve hafta alanlarının ikisi de kısıtlıysa VEYA'lanır** (ikisi de doğru olmasına gerek yok).

```
0 8 * * 1-5   → Cuma 14.08'den sonra Pazartesi 17.08 08:00   ✓ hafta sonu atlanıyor
*/15 * * * *  → 23:30, 23:45, 00:00
30 9 1 * *    → 01.09 09:30, 01.10 09:30
```

Geçersiz girdi **ValueError** ile ve nedenini söyleyerek reddediliyor (`'99' alanı 0-59 aralığında olmalı`) — HTTP katmanında 400 olarak dönüyor.

---

## Macro'dan alınan iki karar

### 1. Bayatlık kontrolü claim'in İÇİNDE

Bizim board'da iki adım var: `claim_next` (`WHERE claim_lock IS NULL`) ve **ayrı** bir `recover_stale()` süpürmesi. İkinci adım **çağrılmayı unutulabilir bir yol** — nitekim bugün Temporal'da tam bu sınıftan bir hata bulduk (bayat claim yüzünden başarılı sonuç çöpe gidiyordu).

Zamanlayıcıda o yol yok:

```python
UPDATE schedules SET claimed_at=?
WHERE id=? AND (claimed_at IS NULL OR claimed_at < ?)   # ? = now - LEASE
```

**"Sahipsiz VEYA kirası dolmuş"** tek koşulda. Ayrı kurtarma adımı olmadığı için atlanması imkânsız. `due()` de aynı predikatı kullanıyor, yani bayat bir claim otomatik olarak yeniden aday oluyor.

Ölçüldü:

```
✓ 1. worker claim aldı
✓ 2. worker AYNI ANDA alamadı
✓ claim tazeyken due() onu aday GÖSTERMİYOR
✓ kira dolmadan devralınamıyor            (lease=900s)
✓ kira DOLUNCA due() bayat claim'i yeniden aday gösteriyor
    → ayrı bir recover_stale() süpürmesi çağrılmadı
✓ kira DOLUNCA devralınıyor (tek atomik UPDATE)
```

### 2. `next_run_at` yazma anında türetilir

Cron'u okuma anında ayrıştırmak yerine **bir kez** hesaplayıp saklıyoruz. Arayüz "sonraki koşu"yu cron ayrıştırmadan gösterebiliyor, ve sıralama (`ORDER BY next_run_at`) doğrudan indeks üstünden çalışıyor.

### Küçük bir uyarlama: adil dağıtım

Macro'nun `BATCH_MIN_DURATION = 30s`'i (bir örneğin kuyruğu süpürmesini engelleme) küçültülmüş hâliyle alındı: `BATCH_MIN_SECONDS = 5`. Bir tur bundan kısa sürerse fark kadar bekleniyor.

---

## Ölçümler

### At-most-once — 8 gerçek süreç

```
8 ayrı SÜREÇ aynı zamanlamayı claim etmeye çalıştı
  claim alabilen: 1     ← tam olarak bir tane
```

Board'daki CAS testimizle aynı garanti, ayrı bir tabloda doğrulandı.

### Uçtan uca

```
[zamanlayıcı] ▶ denetim (s_d37c3c) koşuyor · akış=p_93613778
[zamanlayıcı] ✓ denetim → 5/5 düğüm · 0.0 sn · sonraki: 10.08 23:30

✓ claim serbest bırakıldı
✓ son durum 'ok'
✓ takvim İLERLEDİ (sonraki koşu geleceğe alındı)
✓ koşu geçmişine yazıldı        5/5 düğüm · 0.02s
```

### Canlı sunucuda, süreçler arası

Sunucunun yoklayıcısı açıkken **dışarıdan ayrı bir süreçle** bir zamanlamanın vadesi geçmişe alındı:

```
vade geçmişe alındı, yoklayıcı bekleniyor…
✓ 11. saniyede yoklayıcı koşturdu → durum=ok
  sonraki koşu: 10.08 23:40
  koşu geçmişi: 5/5 düğüm · 0.0 sn
```

İki ayrı süreç aynı SQLite dosyası üzerinden koordine oldu — `busy_timeout` sayesinde (BUG 4'ten öğrenilmişti).

### Hata yolu

| durum | sonuç |
|---|---|
| akış bulunamadı | `last_status='hata'`, claim serbest, **takvim yine ilerledi** |
| akışta düğüm battı | `'hata'`, sistem ayakta, geçmişe `✗1 başarısız` yazıldı |
| geçersiz cron | oluşturmada reddedildi (HTTP 400 + neden) |
| `enabled=0` | `due()`'da görünmüyor, `tur()` koşturmuyor |

**Takvimin hata durumunda da ilerlemesi bilinçli:** aksi hâlde batan bir akış her yoklamada yeniden denenir ve sonsuz döngüye girer. Bir sonraki cron vaktinde tekrar denenecek.

---

## Arayüz

Başlıkta **Zamanlama** düğmesi. Panel:

- Yeni zamanlama formu — ad, cron, akış seçimi + tıklanabilir cron örnekleri (`her gün 08:00`, `hafta içi 08:00`, `15 dakikada bir`, …)
- Mevcut zamanlamalar — cron'un insan okunur özeti, **sonraki koşu**, son durum, son 5 koşunun rozetleri (yeşil/kırmızı, üstüne gelince detay)
- Her satırda: **▶ şimdi çalıştır** · **⏸ durdur** · **✕ sil**

`şimdi çalıştır` takvimi **kaydırmaz** — elle bir koşu bir sonraki cron vaktini ötelememeli.

### HTTP uçları

```
GET /schedules              liste + son 5 koşu
GET /schedule/add           name, cron, pipeline, backend, strategy   → 400 (geçersiz cron)
GET /schedule/toggle        id                                         etkin/pasif
GET /schedule/delete        id
GET /schedule/run           id                                        → 409 (zaten koşuyor)
```

### CLI

```bash
scheduler.py ekle --ad "sabah denetimi" --cron "0 8 * * 1-5" --akis p_93613778
scheduler.py liste
scheduler.py sonraki "0 8 * * 1-5"     # sonraki 5 tetikleme
scheduler.py simdi s_2bd3a5            # elle koştur
scheduler.py gecmis s_2bd3a5           # koşu geçmişi
scheduler.py yokla                     # sürekli yoklayıcı
```

---

## Bizde olan / olmayan

| eksen | durum |
|---|---|
| cron ifadesi (5 alan, `*/n`, aralık, liste) | ✅ |
| sonraki koşu zamanı, yazma anında türetilmiş | ✅ |
| atomik claim + lease (at-most-once, 8 süreçle doğrulandı) | ✅ |
| bayat claim devralma, ayrı süpürme adımı olmadan | ✅ |
| etkin/pasif | ✅ |
| koşu geçmişi (süre, düğüm sayısı, sonuç, hata detayı) | ✅ |
| elle "şimdi çalıştır" (takvimi kaydırmadan) | ✅ |
| adil dağıtım (`BATCH_MIN_SECONDS`) | ✅ |
| **zaman dilimi (timezone)** | ❌ sunucunun yerel saati kullanılıyor |
| **kaçırılan koşuyu telafi (catchup)** | ❌ sunucu kapalıyken geçen vakit atlanır |
| **koşu başına retry/backoff** | ❌ hata → bir sonraki cron vaktini bekler |
| **zamanlanmış koşuya bildirim** | ❌ sonuç yalnız panelde |

İlk üçü Macro'da var (`timezone: Tz` alanı, `catchup=False` ile açıkça kapatılmış, bildirim `notify_completion` ile). Bilinçli olarak kapsam dışı bıraktım — istenirse `timezone` en ucuzu (`zoneinfo` stdlib'de).

---

## Regresyon

```
zamanlama          43/43
task yaşam döngüsü 42/42   (scheduling kontrolü güncellendi: "yok" → "var")
hata dayanıklılığı 54/54
```

Sunucu 8030'da zamanlayıcı açık: `ZAMANLAYICI açık · N zamanlama · yoklama 20s · lease 900s`
