# Retry Ölçümü — bir düğüm bozulunca ne oluyor?

**Ölçüm:** `demo-brain-agent/test_retry.py` · **Yöntem:** gerçek fonksiyon çağrıları sayaçlandı
(`F.call` sarmalandı) + board olay günlüğü + duvar saati damgaları.
"Retry var" iddiası değil, **kaç kez koştu / ne kadar arayla / kim tekrarladı** ölçümü.

**Bulunan bug:** 1 (Temporal'da geçici hata kalıcıya dönüşüyordu) — düzeltildi.

---

## Özet tablo

| | geçici hata | kalıcı hata |
|---|---:|---:|
| yürütme denemesi | **2** | **3** (BREAKER_LIMIT) |
| board `attempt` | 1 | 3 |
| son durum | `done` | `failed` |
| üst düğüm tekrar koştu mu | **hayır (1 kez)** | hayır (1 kez) |
| alt düğüm | koştu | **koşmadı** (`cancelled`) |
| denemeler arası bekleme | **~0 sn (backoff yok)** | ~0 sn |
| toplam süre | 0,003 sn | 0,001 sn |

---

## 1) Geçici hata — retry'ın anatomisi

Board olay zinciri, tek bir düğüm için:

```
created            fn=scan_patterns parents=['t_f7b410']
unblocked          parent'lar tamamlandı          ← DAG kapısı açıldı
claimed            worker-3                       ← 1. deneme
retry_scheduled    geçici hata … breaker=1        ← hata: ready'ye geri döndü
claimed            worker-4                       ← 2. deneme (BAŞKA worker kaptı)
completed                                         ← başarı
```

Dikkat çeken iki nokta:

- **Retry aynı worker'a bağlı değil.** Task `ready`ye döner ve *kim boşsa* kapar
  (`worker-3` → `worker-4`). Retry bir "yeniden dene" döngüsü değil, **kuyruğa geri koyma**.
  Bu yüzden çöken bir worker'ın işi de aynı mekanizmayla devralınabiliyor.
- **Breaker başarıdan sonra sıfırlandı** (`consecutive_failures = 0`) ama `attempt`
  korundu (`1`). Yani bir sonraki hata için 3 hak geri geliyor, ama toplam deneme
  geçmişi denetim izi olarak duruyor.

---

## 2) Kalıcı hata — vazgeçme noktası

```
claimed → retry_scheduled  breaker=1
claimed → retry_scheduled  breaker=2
claimed → failed           breaker=3      ← vazgeçildi
```

3. denemede `failed`. Ardından **iptal zinciri** devreye giriyor: `cross_check` ve
`render_report` hiç koşmadı (`cancelled`). Ölçüldü: **alt düğüm çağrı sayısı = 0.**
Yani hatalı/eksik veriyle aşağı doğru koşma yok.

Batan dalla ilgisiz `run_test_suite` etkilenmedi — `done`.

---

## 3) En kritik ölçüm: retry yan etkiyi tekrarlar mı?

Gerçek sistemde bu şu sorudur: *"düğüm e-postayı gönderdikten sonra patlarsa,
retry ikinci kez gönderir mi?"* İki hata modu ayrı ayrı ölçüldü:

| hata modu | bozulan düğümün İŞİ kaç kez yapıldı | üst düğümler |
|---|---:|---|
| iş **yapılmadan** patlıyor *(bağlantı kurulamadı)* | **1** | 1 kez (tekrar koşmadı) |
| iş **yapıldıktan sonra** patlıyor *(kayıt yazıldı, onay alınamadı)* | **2** | 1 kez (tekrar koşmadı) |

**Sonuç: retry düğümü BAŞTAN koşturuyor, kaldığı yerden değil.**

- İş bittikten sonra patlayan bir düğümün işi **2 kez** yapılıyor.
- Bugünkü düğümler saf/deterministik olduğu için zararsız (aynı girdi → aynı çıktı).
- Ama **yan etkili** bir düğüm eklenirse (e-posta, ödeme, dosya yazma, API POST)
  **idempotenslik zorunlu** hale gelir. Bu, sisteme yeni fonksiyon eklerken uyulması
  gereken kural — ölçümle sabitlendi, varsayım değil.

**Üst düğümler her iki modda da 1 kez koştu.** Sonuçları board'da saklı olduğu için
retry yalnız bozulan düğümü tekrarlıyor, tüm akış baştan koşmuyor
(Airflow'da tek bir task'ı yeniden çalıştırmakla aynı davranış).

---

## 4) Retry'ı kim yönetiyor — üç motor, üç farklı mekanizma

| motor | mekanizma |
|---|---|
| **own** | `board.fail()` → `status='ready'` → sonraki turda tekrar claim |
| **temporal** | activity `RetryPolicy(maximum_attempts=3)` → Temporal activity'yi tekrar çağırır |
| **celery** | `self.retry(countdown=0)`, `max_retries=3` → broker'a geri koyar |

Mekanizmalar farklı, **sonuçları aynı olmalı**. Ölçüm (board'dan, süreçler arası geçerli —
süreç-içi sayaç Celery'yi göremez):

| backend | yürütme denemesi | attempt | durum | tamamlanan | süre |
|---|---:|---:|---|---:|---:|
| own | 2 | 1 | `done` | 5/5 | 0,01 sn |
| temporal | 2 | 1 | `done` | 5/5 | 0,53 sn |
| celery | 2 | 1 | `done` | 5/5 | 40,4 sn |

**Dördü de aynı: yürütme denemesi ✓ · attempt ✓ · son durum ✓ · tamamlanan ✓**

Celery'nin 40 sn'si worker açılış + broker gecikmesi; davranış farkı değil.

---

## BUG 12 — Temporal'da geçici hata KALICIYA dönüşüyordu

Bu ölçüm olmasa görülmezdi. İlk koşumda tablo şöyleydi:

| backend | yürütme denemesi | attempt | durum | tamamlanan |
|---|---:|---:|---|---:|
| own | 2 | 1 | `done` | 5/5 |
| **temporal** | **3** | **3** | **`failed`** | **2/5** |

**Aynı geçici hata, own'da 1 retry'la toparlanırken Temporal'da düğümü kalıcı olarak
batırıyordu.** Olay günlüğü zinciri sebebi gösterdi:

```
claimed → retry_scheduled (attempt=0) → stale_write_reddedildi
claimed → retry_scheduled (attempt=0) → stale_write_reddedildi
claimed → failed          (attempt=0) → stale_write_reddedildi
```

İki ayrı kusur üst üste binmiş:

**(a) Bayat claim.** `board.fail()` claim'i temizler ve task'ı `ready` yapar. Temporal ise
**aynı activity'yi aynı payload ile** yeniden çağırır — elindeki `claim_lock` artık geçersiz.
Sonuç: iş **başarıyla koşuyor**, ama `complete()` fencing'e takılıyor ve **sonuç çöpe gidiyor**.
Task hâlâ `ready` → bir sonraki tur baştan claim → 3 turda breaker doluyor.
**İki kez başarılı olan düğüm `failed` işaretleniyordu.**

**(b) Deneme sayacı sıfırlanıyor.** `att = activity.info().attempt - 1` — Temporal her **yeni**
activity çağrısında kendi sayacını 1'e sıfırlar. Bu yüzden log'da hep `attempt=0` görünüyor:
sistem geçici hatayı sonsuza dek "ilk deneme" sanıyordu.

**Düzeltme:**
- `TaskBoard.claim(tid, claimer)` eklendi — *belirli* bir task'ı CAS ile kapar
  (`claim_next` id seçmez, bu seçer). Activity başında task `ready` ise **yeniden kapılıyor**.
- Deneme sayacı board'dan alınıyor: `att = max(temporal_attempt - 1, board_attempt)`.

**Düzeltme sonrası:** geçici → `attempt=1, done, 4/4` · kalıcı → `attempt` 0,1,2 → `failed`
+ ardıllar `cancelled` · **bayat yazma hiçbirinde yok.**

> Not: bu kusur BUG 8'in (fencing) yan ürünü olarak **görünür** hale geldi. Fencing öncesinde
> `complete()` bayat claim'le de yazabildiği için sistem "kazara" çalışıyordu — sonuç doğruydu
> ama at-most-once garantisi yoktu. Fencing gerçek tasarım çatışmasını ortaya çıkardı.

---

## 5) Breaker sıfırlama

| adım | attempt | breaker | durum |
|---|---:|---:|---|
| 1. hata | 1 | 1 | `ready` |
| 2. hata | 2 | 2 | `ready` |
| **başarı** | 2 | **0** | `done` |

Breaker **arka arkaya** hatayı sayıyor; araya bir başarı girerse sıfırlanıyor. `attempt`
sıfırlanmıyor — toplam deneme geçmişi denetim izi olarak kalıyor.

Bu doğru davranış: uzun ömürlü bir task, hayatı boyunca 2 kez toparlanıp 3. seferde
batmasın diye "3 hata" değil "**arka arkaya** 3 hata" kuralı uygulanıyor.

---

## 6) Kalıcı hatada retry atlanıyor mu?

Sözleşme hatası (bilinmeyen fonksiyon, eksik zorunlu upstream) tekrar denemekle düzelmez.

```
cross_check (zorunlu upstream yok)
   çağrı sayısı : 1        ← geçici olsaydı 3 olurdu
   attempt      : 1
   durum        : failed
   olay         : eksik upstream verisi: ['matches','failures'] — düğüm veriyi
                  üreten düğüme BAĞLI DEĞİL
```

**Retry atlandı ✓** — 2 gereksiz deneme ve 2 gereksiz upstream okuması yapılmadı.

---

## Backoff yok — bilinçli ama sınırlı bir tercih

Denemeler arası bekleme **ölçülemeyecek kadar kısa** (own'da `countdown=0`, board'da
doğrudan `ready`). Bugünkü düğümler saf, hızlı ve yerel olduğu için bu ucuz.

**Ama:** dış servise giden bir düğüm eklenirse (rate-limit'li API, geçici kesinti),
backoff'suz retry servisi 3 kez arka arkaya döver ve rate-limit'i **derinleştirir**.
Üstelik "geçici" hatanın toparlanması için gereken zaman hiç tanınmamış olur —
0 sn sonra tekrar denemek çoğu geçici hatada işe yaramaz.

Bu, sisteme dış servis düğümü eklendiğinde ilk yapılması gereken değişiklik.

---

## Tek cümlelik özet

> Retry, düğümü kuyruğa geri koyup **baştan** koşturuyor: üst düğümlerin işi korunuyor,
> yalnız bozulan düğüm tekrarlanıyor, 3 arka arkaya hatada vazgeçilip ardıllar iptal ediliyor —
> üç motorda da aynı sonuçla. Bedeli: yan etkili bir düğüm eklenirse idempotenslik zorunlu,
> ve dış servis düğümü eklenirse backoff şart.
