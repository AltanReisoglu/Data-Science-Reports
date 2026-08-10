# Hata Dayanıklılığı Testi — çalışma sırasında bir şey patlarsa ne oluyor?

**Tarih:** 10 Ağustos 2026 · **Test paketi:** `demo-brain-agent/test_hata.py` (54 kontrol, 8 bölüm)
**Sonuç:** 54/54 kontrol geçti · **5 gerçek bug bulundu ve düzeltildi** · 4 bulgu kalıntı olarak belgelendi

---

## Neden bu test

Önceki üç tur (chat matrisi, task yaşam döngüsü, concurrency, compaction) **mutlu yolu** ölçtü:
her düğüm başarılı, akış tamamlanıyor. Ama bir sistemin dayanıklılığı başarılı koşuda değil,
**bir şey patladığında** belli olur. Bu tur tam tersini kovaladı.

Sistemde birbirine karıştırılmaması gereken **iki ayrı yol** var:

| | ÇÖKME (`WorkerCrash`) | HATA (`Exception`) |
|---|---|---|
| ne oldu | iş yapıldı, `complete()` çağrılmadan worker öldü | işin kendisi patladı |
| `attempt` | **artmaz** (worker suçlu, iş değil) | artar |
| checkpoint | **durur** → devralan kaldığı yerden sürer | yazılmaz |
| sonuç | `recover_stale` → başkası devralır | breaker → `ready` / `failed` |

**Ölçüldü, ikisi gerçekten farklı davranıyor:** çökmede `attempt=0` + checkpoint korunuyor,
hatada `attempt=1` + retry. Bu ayrım daha önce iddia edilmişti; artık kanıtlı.

---

## Test edilemeyen bir yolu test edilebilir hale getirmek

İlk bulgu testin kendisinden çıktı: `fail_at` **yalnız ajan düğümlerine** hata enjekte ediyordu.
Fonksiyon-öncelikli mimaride düğümlerin çoğu fonksiyon olduğu için **baskın düğüm tipinin
hata/retry/breaker yolu hiç tetiklenemiyordu.**

`run_one_task`'a fonksiyon düğümü için hata enjeksiyonu eklendi — iki modlu:

```
fail_at="scan_patterns"    → GEÇİCİ hata (ilk denemede patlar, retry'da geçer)
fail_at="scan_patterns!"   → KALICI hata (her denemede patlar → breaker → failed)
```

Bu, BUG 1'de (`crash_after_turn` fonksiyon düğümlerinde çalışmıyordu) görülen aynı sınıf boşluk:
*mimari değişti, test kancası güncellenmedi.*

---

## Bulunan ve düzeltilen 5 bug

### BUG 7 — Batan dalın ardındaki düğümler sonsuza dek `blocked` kalıyordu · **YÜKSEK**

Bir düğüm breaker'ı doldurup `failed` olduğunda, `recompute_ready()` çocuklarını **asla**
terfi ettiremiyordu (terfi için parent'ın `done` olması gerekiyor, `failed` asla `done` olmaz).
Sonuç: düğümler süresiz `blocked`, `all_settled()` sonsuza dek `False`, koşu `break` ile sessizce
biterken **hiçbir yerde iptal damgası yok.**

> Operatörün "akış tamam mı?" sorusuna log'a bakarak cevap vermesi imkânsızdı.

**Düzeltme:** `board.cancel_downstream()` — `fail()` breaker'ı doldurduğunda batan dalın
**tüm alt soyunu** (çocuk değil, torun dahil) `cancelled` yapar. Tek noktada, `fail()` içinde:
dört backend de bedavaya aynı davranışı alır.

```
ÖNCE :  scan=failed · cross_check=blocked · report=blocked   → all_settled()=False (sonsuza dek)
SONRA:  scan=failed · cross_check=cancelled · report=cancelled → all_settled()=True
```

`cancelled` yeni bir durum: **"bekliyor" ile "asla koşmayacak" artık board'da ayrı görünüyor.**
Olay günlüğüne `cancelled` yazılıyor (denetim izi), koşu log'u `⛔ BATAN DAL KAPATILDI` diyor,
sohbet cevabına `⚠ AKIŞ YARIM KALDI` uyarısı ekleniyor, UI'da içi boş kırmızı halka ile gösteriliyor.

Batan dalla **ilgisiz** düğümler etkilenmiyor (ölçüldü: `run_test_suite=done`).

### BUG 8 — `complete()` claim sahipliğini doğrulamıyordu (fencing yok) · **YÜKSEK**

`claim_next` CAS ile korunuyordu — ama `complete()`/`fail()` korunmuyordu. Senaryo:

1. worker-A task'ı kapar, lease dolar
2. `recover_stale` task'ı geri kuyruğa atar, worker-B devralır ve bitirir
3. worker-A geç uyanır, `complete()` çağırır → **yazma geçer, B'nin sonucunu EZER**

> at-most-once garantisi **claim'de vardı, yazmada yoktu.** Bu, "at-most-once" iddiasının
> yarısının doğru olmadığı anlamına geliyordu.

**Düzeltme:** `complete(tid, result, claimer=...)` ve `fail(..., claimer=...)` artık
`WHERE ... AND claim_lock=?` ile yazıyor; `rowcount != 1` ise `stale_write_reddedildi`
olayı yazılıp `False` dönüyor. Dört çağrı yeri de (own / celery / temporal) bağlandı.

Celery'de ek bir gizli hata çıktı: `task` dict'i **claim'den ÖNCE** alınıyordu → `claim_lock=None`
→ fencing devre dışı kalırdı. Claim sonrası kayda geçirildi.

Ölçüm: geç kalan worker'ın yazması **reddedildi**, devralanın normal yazması **geçti**
(fencing fazla katı değil).

### BUG 9 — Uydurma fonksiyon adı board'a yazılıyor, 3 kez boşuna deneniyordu · **ORTA**

`create_task` `fn` alanını hiç doğrulamıyordu. LLM halüsinasyonu ya da kayıtlı bir pipeline'daki
silinmiş fonksiyon **ancak yürütmede**, hem de breaker 3 kez boşuna harcandıktan sonra fark ediliyordu.

**Düzeltme:** `create_task` kayıt anında `F.resolve()` ile doğruluyor, geçerli listeyi hata
mesajına koyuyor. `run_saved` bozuk kayıtlı akışı **yüklemede** reddediyor — yarım koşup
yürütmede patlamıyor (ölçüldü: `done=0`, kısmi yan etki yok).

### BUG 10 — Kalıcı ve geçici hata aynı kefede, breaker boşa harcanıyordu · **ORTA**

Bilinmeyen fonksiyon / geçersiz argüman gibi **sözleşme** hataları tekrar denemekle düzelmez,
ama breaker onları ağ zaman aşımı gibi 3 kez tekrarlıyordu.

**Düzeltme:** `fail(..., permanent=True)` → tek denemede `failed`.
`_dispatch_own` `ValueError/TypeError/KeyError`'ı kalıcı sayıyor, log'a
`(KALICI sözleşme hatası — retry edilmedi)` yazıyor.

### BUG 11 — Yinelenen bağımlılık düğümü sonsuza dek kilitliyordu · **YÜKSEK**

Canlı koşuda planlayıcı `depends_on`'a aynı id'yi **iki kez** yazdı:

```
render_report  parents=['t_0cae2e','t_804955','t_804955','t_a464b3','t_71f6c1']
```

`recompute_ready` `COUNT(*) ... WHERE id IN (...)` (satır sayar → 4) ile `len(parents)` (tekrarı
sayar → 5) karşılaştırıyor. **Eşitlik asla sağlanamaz** → düğüm sonsuza dek `blocked`.
Gerçek koşuda final rapor düğümü tam da böyle asılı kaldı.

**Düzeltme:** `create_task` ve `recompute_ready` parent listesini sırayı koruyarak tekilleştiriyor.
Ek olarak `_dispatch_own` koşu sonunda geride yürütülmemiş düğüm kalmışsa artık
`⚠ KOŞU BİTTİ ama N düğüm YÜRÜTÜLMEDİ` diye açıkça söylüyor.

---

## En değerli bulgu: sessizce yanlış denetim raporu

Bu, testin bulduğu en ciddi şey ve **hata enjeksiyonu olmadan** ortaya çıktı.

Canlı bir koşuda planlayıcı grafı şöyle kurdu:

```
fetch_source ──► scan_patterns          (bulguları üretir)
run_test_suite
      └──────► cross_check ← fetch_source, run_test_suite     ← scan_patterns'a BAĞLI DEĞİL
                      └──► render_report
```

`cross_check`'in **tüm işi** tarama bulgularını test hatalarıyla eşleştirmek — ama planlayıcı onu
tarayıcıya bağlamamıştı. `_merge_up(_up, "matches", [])` eksik kenarda **sessizce varsayılana**
düşüyordu. Sonuç:

> **Bir güvenlik denetimi pipeline'ı, tarayıcısı hiç veri vermemişken
> "Taranan desen eşleşmesi: 0" yazan temiz bir rapor üretti.**

Hata yok, uyarı yok, kırmızı bir şey yok. Denetim raporunda "0 eşleşme" **"açık bulunamadı"**
demektir. Yanlış cevap, çökmeden çok daha tehlikelidir.

Kök neden: **DAG'ın doğruluğu tamamen LLM planlayıcıya bırakılmıştı** ve hiçbir katman
"bu düğüm tükettiği veriyi üreten düğüme bağlı mı?" diye sormuyordu.

### Üç katmanlı düzeltme

**1) Zorunlu upstream sözleşmesi** (`functions.NEEDS`) — her düğüm olmazsa olmaz upstream
anahtarlarını beyan ediyor. Eksikse `F.call` **patlar**, varsayılanla koşmaz:

```
cross_check   → matches, failures
render_report → korelasyon, count, total
validate_schema / transform_normalize / aggregate_stats → rows
```

`render_report`'un yalnız `korelasyon` istemesi yetmiyordu: `count` ve `total` varsayılandan
gelip rapora "0 eşleşme / 0 test" diye basılıyordu. Üçü de zorunlu tutuldu.

**2) Katalogda kenar rehberi** — planlayıcının gördüğü fonksiyon kataloğu artık şunu yazıyor:

```
• cross_check((argümansız))
    ⚠ ZORUNLU: upstream'de ['matches','failures'] olmalı →
      depends_on'a ['run_test_suite','scan_patterns'] EKLE, yoksa düğüm hata verir.
```

Son canlı koşuda planlayıcı kenarları **doğru kurdu, 0 onarım gerekti** — rehber işe yarıyor.

**3) Plan doğrulama + otomatik kenar onarımı** (`orchestrator.dogrula_dag`) — planlama bittikten
sonra, yürütmeden önce her fonksiyon düğümünün zorunlu anahtarlarını üreten bir parent'ı var mı
diye bakıyor; yoksa graftaki üreticiyi parent olarak ekliyor (çevrim koruması ile) ve log'a yazıyor.

Burada ince bir tuzak vardı: doğrulama önce **ata kapanışına** bakıyordu, ama veri yalnız
**doğrudan parent'lardan** akıyor (`upstream_results`). Dedenin çıktısı torununa otomatik geçmez.
Doğrulama veri akışıyla hizalandı.

Ölçülen fark, aynı bozuk graf üzerinde:

| | ÖNCE | SONRA |
|---|---|---|
| akış | 5/5 done | 5/5 done |
| rapor | `Taranan desen eşleşmesi: **0**` · `Koşan test: **0**` | `**4**` · `**240**` |
| uyarı | yok | `⚠ EKSİK KENAR ONARILDI: rapor ← tara [gereken veri: count]` |

---

## Üç backend aynı hatada ne yapıyor

Aynı kalıcı hata, aynı graf, üç motor:

| backend | süre | scan | ardıl düğümler | ilgisiz düğüm | deneme |
|---|---:|---|---|---|---:|
| own | 0,0 sn | `failed` | `cancelled` ×2 | `done` | 3 |
| temporal | 0,5 sn | `failed` | `cancelled` ×2 | `done` | 3 |
| celery | 12,3 sn | `failed` | `cancelled` ×2 | `done` | 3 |

**Üçü de asılmadı, üçü de aynı son duruma ulaştı, retry sayısı tutarlı.** Hata dayanıklılığı
backend seçiminden bağımsız — çünkü karar board'da, tek noktada veriliyor. Celery'nin 12,3 sn'si
worker açılış maliyeti; davranış farkı değil.

---

## Diğer ölçümler

**Compaction dayanıklılığı** — 6 strateji × 6 uç girdi (boş iz, içeriksiz mesaj, yetim tool sonucu,
sıfır ve negatif bütçe): **36 kombinasyonun hiçbiri patlamadı.** Bağlam yönetimi ajanı düşürmüyor.

**Canlı sohbet, hatalı istekler** — LLM'e olmayan tool çağırtmak, eksik argüman vermek, olmayan
fonksiyonla pipeline istemek: üçünde de SSE akışı düzgün `done` ile kapandı, kullanıcıya çökme izi
sızmadı, boş cevap dönmedi, sunucu sağlıklı kaldı.

**Dev sonuç** — 500 KB'lık sonuç kırpıldı, board ayakta kaldı, kırpılan sonuç **hâlâ geçerli JSON**
(aşağı akış bozulmuyor).

**Lease + devralma** — lease dolan task geri kuyruğa döndü, devralındı, checkpoint korundu.

---

## Kalan bulgular (düzeltilmedi, bilinçli)

| ağırlık | bulgu | neden bırakıldı |
|---|---|---|
| orta | **Geçersiz argüman sessizce yutuluyor** — `olmayan_arg=42` uyarısız atılıyor, düğüm "başarılı" görünüp yanlış varsayılanla koşuyor | `add_step` girişte uyarı veriyor; `create_task` yolunda vermiyor. Düzeltmesi kolay ama argüman filtreleme demo akışlarında bilinçli bir tolerans |
| orta | **Board dosyası silinse bile yazmalar "başarılı" dönüyor** — POSIX'te açık dosya tanıtıcısı unlink sonrası yaşar; sonuçlar okunamayan bir inode'a yazılır | SQLite/POSIX semantiği, uygulama katmanında çözülmez. Gerçek dağıtımda dosya değil sunucu-tabanlı DB kullanılır |
| düşük | **Fonksiyon düğümünde hatada checkpoint yazılmıyor** — çökmede kısmi iş korunuyor, hatada retry sıfırdan | Fonksiyonlar deterministik ve kısa; uzun süren bir fonksiyon eklenirse yeniden değerlendirilmeli |
| düşük | **Fonksiyonlar boş upstream'i kendi başlarına yakalamıyor** | Board seviyesindeki iptal zinciri + `NEEDS` sözleşmesi bunu normal yolda ulaşılmaz kıldı; savunma tek katmanlı — düğümler board dışından çağrılırsa koruma yok |

---

## Regresyon

Tüm önceki paketler yeni kodla yeniden koşturuldu:

```
hata dayanıklılığı  : 54/54 kontrol geçti
task yaşam döngüsü  : 40/40 kontrol geçti
concurrency         : at-most-once ✓ · 24/24 tamamlandı · 4,9× hızlanma · çökme→devralma ✓
compaction matrisi  : 30/30 çift bütünlüğü sağlam · kritik bilgi 8/15
pipeline yaşam döngüsü (own/temporal/celery) : 5/5 · 5/5 · 5/5
```

---

## Toplam tablo

İki tur test, **11 gerçek bug**:

| # | bug | sınıf |
|---|---|---|
| 1 | çökme enjeksiyonu fonksiyon düğümlerinde hiç tetiklenmiyordu | mimari değişti, çağıran güncellenmedi |
| 2 | Celery dispatch'te `on_event` yok → SSE zaman aşımı | mimari değişti, çağıran güncellenmedi |
| 3 | dalga-bitti kontrolü kaldırılınca 25 sn takılmalar | regresyon |
| 4 | board'da `busy_timeout` yok → çok-süreçli kilit | eksik yapılandırma |
| 5 | `spawn_task` tamamen kırıktı (varsayılan `kind` değişmişti) | mimari değişti, çağıran güncellenmedi |
| 6 | Codex düşük bütçede yetim tool çifti üretiyordu → API 400 | sınır koşulu |
| 7 | batan dalın ardındakiler sonsuza dek `blocked` | eksik durum modeli |
| 8 | `complete()`/`fail()` fencing yok → zombi worker yazabiliyor | eksik eşzamanlılık koruması |
| 9 | uydurma fonksiyon adı board'a yazılıyor, 3 kez deneniyor | eksik doğrulama |
| 10 | kalıcı/geçici hata ayrımı yok, breaker boşa harcanıyor | eksik sınıflandırma |
| 11 | yinelenen bağımlılık kapıyı sonsuza dek kilitliyor | küme/liste karışıklığı |

**11 bug'ın 3'ü** *"mimari değişti, çağıran güncellenmedi"* sınıfı — fonksiyon-öncelikli geçiş ve
paket refactor'ü, görünürde çalışan ama hiç test edilmeyen yolları sessizce kırmış.
**4'ü** yalnızca hata yolu koşturulduğunda görünür hale geldi.

## Sunumda söylenebilecek tek cümle

> Sistemi başarılı koşuda değil, **patladığında** ölçtük: çöken worker'ın işi devralınıyor,
> batan dalın ardındakiler sessizce asılı kalmak yerine iptal ediliyor ve raporlanıyor,
> zombi worker başkasının sonucunu ezemiyor — ve en önemlisi, **eksik bir DAG kenarı yüzünden
> "0 bulgu" diyen temiz bir denetim raporu üretmek artık mümkün değil.**
