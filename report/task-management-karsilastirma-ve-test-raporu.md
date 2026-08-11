# Task Management — Karşılaştırma ve Test Raporu

**Kapsam:** Dört yürütme motorunun (hermes type · airflow · celery · temporal) karşılaştırması ve sistemin tamamı üzerinde yapılan ölçümler.
**Ölçüm tarihi:** 10 Ağustos 2026 · **Toplam:** 7 test paketi, **190 kontrol**, hepsi geçti
**Bulunan ve düzeltilen hata:** 12

Bu belge iki şeyi birleştirir: (1) motorların yan yana karşılaştırması, (2) hangi iddianın hangi ölçümle desteklendiği. Ölçülmemiş her şey **açıkça öyle işaretlenmiştir**.

---

# BÖLÜM 1 — Ne inşa ettik

## 1.1 Mimari: üç bağımsız katman

```
┌─ TASK MANAGEMENT ──────── board (FSM + DAG + claim + retry + iptal)
│                            motor: hermes type / airflow / celery / temporal
├─ AJAN DÖNGÜSÜ ─────────── plan → dispatch → düğüm yürüt
└─ TOOL-TRACE COMPACTION ── none / hermes / opencode / openclaw / codex / claude_code
```

Üçü **ayrı ayrı seçilebilir**. Bu raporun konusu birincisi.

## 1.2 Board — durum makinesi

```
              ┌──────────── recompute_ready ────────────┐
              │                                          ▼
 create ──► blocked ──► ready ──► running ──► done
                          ▲          │
                          │          ├──(hata: retry hakkı var)──┘
                          │          └──(breaker doldu)──► failed
                          │                                  │
                          └──(çökme: lease/PID)──────────┐    │
                                                          │    ▼
                                        recover_stale ────┘  cancelled ◄── iptal zinciri
```

| mekanizma | uygulama |
|---|---|
| DAG kapısı | `parents` + `recompute_ready()` — parent `done` olmadan çocuk açılmaz |
| CAS-claim | `UPDATE … WHERE claim_lock IS NULL` → at-most-once |
| lease + heartbeat | 30 sn; süresi dolan iş geri kuyruğa |
| çökme kurtarma | `recover_stale()` — lease dolmuş **VEYA** PID ölü |
| fencing | `complete/fail … AND claim_lock=?` — bayat worker yazamaz |
| circuit-breaker | arka arkaya 3 hata → `failed` (sonsuz retry yok) |
| kalıcı/geçici ayrımı | sözleşme hatası (`ValueError`) tek denemede kapanır |
| iptal zinciri | batan dalın **tüm alt soyu** `cancelled` |
| checkpoint | fonksiyon ve ajan düğümünde kaldığı yerden devam |
| olay günlüğü | her durum geçişi denetim izine |
| **zamanlama** | cron + timezone-suz yerel saat, atomik claim, koşu geçmişi |

## 1.3 Düğüm türleri — fonksiyon öncelikli

| tür | oran | LLM | örnek |
|---|---|---|---|
| `kind="function"` | **varsayılan** | yok | `fetch_source`, `scan_patterns`, `cross_check` |
| `kind="agent"` | istisna | var | yorumlama, serbest metin üretimi |

16 düğüm fonksiyonu, üç pakette: **audit** (6) · **data** (5) · **deploy** (5).

## 1.4 Üç tool ailesi

| aile | kim çağırır | çıktı | compaction |
|---|---|---|---|
| DAG düğüm fonksiyonları | motor | yapılandırılmış, küçük | gerekmez |
| işçi ajan tool'ları | LLM | ham, büyük | **uygulanır** |
| task-management tool'ları | LLM (planlayıcı) | kısa onay | gerekmez |

---

# BÖLÜM 2 — Dört motorun karşılaştırması

## 2.1 Yapısal fark

| | **hermes type** (kendi motorumuz) | **airflow** | **celery** | **temporal** |
|---|---|---|---|---|
| ne yapar | claim → yürüt → complete/fail | **yürütmez**, DAG dosyası üretir | broker'a atar, ayrı süreç çeker | workflow + activity |
| bağımlılık (DAG) | **board** | Airflow | **board** (Celery'de yok) | **board** (Temporal'da yok) |
| retry mekanizması | `board.fail()` → `ready` | `default_args retries=2` | `self.retry(countdown=0)` | `RetryPolicy(max_attempts=3)` |
| deneme sayısı | 3 (BREAKER_LIMIT) | 3 (retries=2) | 3 | 3 |
| **backoff** | **yok (0 sn)** | **30 sn** | yok (countdown=0) | 200 ms |
| graf | **dinamik** — koşarken büyüyebilir | **DONMUŞ** — dosyaya yazılmış | dinamik | dinamik |
| kurulum | yok (SQLite) | Airflow + scheduler + DB | broker (filesystem/Redis) | Temporal sunucu |

**Kritik nokta:** DAG, checkpoint ve retry kararı **her üç yürüten motorda da board'da** veriliyor. Celery yalnız dağıtım/retry katmanı, Temporal yalnız durable yürütme motoru. Bu yüzden davranış motordan bağımsız çıkıyor — aşağıda ölçüldü.

## 2.2 Aynı hataya tepki — ölçüm

**Kurulum:** aynı 5 düğümlü elmas graf, aynı hata enjeksiyonu, dört motor.
Karşılaştırma **board'dan** yapıldı (`claimed` olayı = bir yürütme denemesi), çünkü süreç-içi sayaç Celery'yi (ayrı süreç) göremiyor.

### Geçici hata (düğüm bir kez patlıyor)

| motor | yürütme denemesi | attempt | scan | tamamlanan | süre |
|---|---:|---:|---|---:|---:|
| hermes type | 2 | 1 | `done` | **5/5** | 0,0 s |
| airflow | **0** | 0 | `blocked` | **0/5** | 0,0 s |
| celery | 2 | 1 | `done` | **5/5** | 15,3 s |
| temporal | 2 | 1 | `done` | **5/5** | 0,5 s |

### Kalıcı hata (her denemede patlıyor)

| motor | deneme | scan | **ardıl düğümler** | tamamlanan |
|---|---:|---|---|---:|
| hermes type | 3 | `failed` | `cancelled` + `cancelled` | 2/5 |
| airflow | **0** | `blocked` | `blocked` + `blocked` | 0/5 |
| celery | 3 | `failed` | `cancelled` + `cancelled` | 2/5 |
| temporal | 3 | `failed` | `cancelled` + `cancelled` | 2/5 |

**Sonuç:** Yürüten üç motorda **yürütme denemesi, attempt sayacı, son durum ve tamamlanan düğüm sayısı birebir aynı.** Ayrıştıkları tek yer hız.

### Çökme (worker `complete()` çağrılmadan ölüyor)

| motor | çökme | kurtarma | not |
|---|---:|---:|---|
| hermes type | 1 | 1 | ✓ `recover_stale` devraldı, checkpoint korundu |
| celery | 0 | 0 | ⚠ **`crash_at` parametresi sessizce yok sayılıyor** |
| temporal | 0 | 0 | ⚠ **aynı** |
| airflow | 0 | 0 | yürütmüyor |

> **Dürüst uyarı:** Çökme kurtarma yalnız kendi motorumuzda **ölçüldü**. Celery ve Temporal `crash_at`'i kabul edip kullanmıyor — bu iki motorda çökme kurtarma *yok değil, ölçülmedi*. Sunumda "üç motorda da çökme kurtarma kanıtlı" **denilemez**.

## 2.3 Airflow neden ayrı kategoride

Airflow bizim katmanda **hiç yürütmüyor**: board `blocked/ready` kalıyor, 0 düğüm koşuyor. Yaptığı iş bir DAG dosyası yazmak (`brain_agent_plan.py` — `PythonOperator` + XCom).

Bunun üç sonucu var:

1. **Hata orada oluşur, bizde değil** → "Airflow şöyle tepki verdi" diyemeyiz
2. **Zamanlama tutarsızlığı:** biz 0 sn arayla 3 kez deniyoruz, ürettiğimiz DAG **30 sn** yazıyor. Deneme sayısı aynı, zamanlama farklı — aynı akış Airflow'a taşınınca farklı davranır
3. **DAG donuyor:** yürütme sırasında ajan yeni task üretemez, replanlama yapamaz

## 2.4 Ne zaman hangisi

| durum | seçim | gerekçe |
|---|---|---|
| geliştirme, demo, tek makine | **hermes type** | 0,01 sn, kurulum yok |
| kurumsal durable garanti, uzun akış | **temporal** | 0,5 sn, replay, RetryPolicy |
| mevcut Celery altyapısı varsa | **celery** | 15–40 sn (worker açılışı) |
| sabit, tekrarlayan, insan-gözetimli veri hattı | **airflow** | olgun operasyon; ama graf donar |

---

# BÖLÜM 3 — Test sonuçları

## 3.1 Genel tablo

| # | paket | kapsam | kontrol | sonuç |
|---|---|---|---:|---|
| 1 | `test_matrix.py` | 28 sohbet senaryosu, HTTP/SSE | 28 | ✅ |
| 2 | `test_tasklife.py` | task yaşam döngüsü | **42** | ✅ |
| 3 | `test_concurrency.py` | 6 gerçek süreç, CAS yarışı | 4 | ✅ |
| 4 | `test_compaction_matrix.py` | 6 strateji × 5 bütçe | 30 | ✅ |
| 5 | `test_hata.py` | hata / çökme / bozuk girdi / altyapı | **54** | ✅ |
| 6 | `test_retry.py` | retry anatomisi, yan etki | 6 ölçüm | ✅ |
| 7 | `test_backend_hata.py` | dört motor, aynı hata | 4 faz | ✅ |
| 8 | `test_devam.py` | çökme sonrası devam, 3 seviye | 1 | ✅ |
| 9 | `test_zamanlama.py` | cron, claim yarışı, lease | **43** | ✅ |

**Toplam 190 kontrol, tamamı geçti.**

## 3.2 Concurrency — at-most-once kanıtı

Bu iddia daha önce **tek süreçte** gösterilmişti, yani kanıt değil argümandı. Şimdi 6 ayrı **işletim sistemi süreci**:

```
6 SÜREÇ · 24 bağımsız task
  toplam claim      : 24
  benzersiz task    : 24
  ÇİFT claim edilen : 0        ← at-most-once korundu
  tamamlanan        : 24/24
  tek süreç 1,29 sn → 6 süreç 0,27 sn = 4,7× hızlanma
  os._exit(1) ile GERÇEK ölüm → recover_stale → devralındı
  checkpoint ölümden sonra korundu: {'kismi': 'w99 yarıda bıraktı'}
```

Zamanlayıcıda aynı garanti ayrı bir tabloda tekrar ölçüldü: **8 süreçten claim alabilen tam olarak 1**.

## 3.3 Retry anatomisi

Board olay zinciri, tek bir düğüm için:

```
claimed            worker-3        ← 1. deneme
retry_scheduled    breaker=1       ← hata → kuyruğa geri
claimed            worker-4        ← 2. deneme (BAŞKA worker kaptı)
completed                          ← başarı
```

**Retry aynı worker'a bağlı değil** — task `ready`ye döner, kim boşsa kapar. "Yeniden dene döngüsü" değil, **kuyruğa geri koyma**. Çöken worker'ın işinin devralınması da aynı mekanizma.

| | geçici hata | kalıcı hata |
|---|---:|---:|
| yürütme denemesi | 2 | 3 (BREAKER_LIMIT) |
| board `attempt` | 1 | 3 |
| son durum | `done` | `failed` |
| üst düğüm tekrar koştu mu | **hayır (1 kez)** | hayır (1 kez) |
| alt düğüm | koştu | **koşmadı** (`cancelled`) |
| denemeler arası bekleme | **~0 sn** | ~0 sn |

**Breaker sıfırlama:** arka arkaya 2 hata → `attempt=2, breaker=2`; araya başarı girince **breaker 0**, `attempt` korunur (denetim izi). Yani kural "3 hata" değil, "**arka arkaya** 3 hata".

**Kalıcı hatada retry atlanıyor:** eksik zorunlu upstream → 1 çağrı, `attempt=1`, `failed`. Geçici sayılsaydı 3 olurdu.

## 3.4 Yan etki tekrarı — en kritik retry ölçümü

Gerçek soru: *"düğüm e-posta gönderdikten sonra patlarsa, retry ikinci kez gönderir mi?"*

| hata modu | düğümün İŞİ kaç kez yapıldı | üst düğümler |
|---|---:|---|
| iş **yapılmadan** patlıyor (bağlantı kurulamadı) | 1 | 1 kez |
| iş **yapıldıktan sonra** patlıyor (kayıt yazıldı, onay alınamadı) | **2** | 1 kez |

**Retry düğümü baştan koşturuyor.** Bugünkü fonksiyonlar saf olduğu için zararsız — ama **yan etkili** bir düğüm eklenirse idempotenslik zorunlu. Bu, sisteme yeni fonksiyon eklerken uyulacak kural; varsayım değil, ölçüm.

## 3.5 Çökme sonrası devam — üç seviye

"Kaldığı yerden devam" karıştırılırsa cümle olduğundan güçlü duyulur. Üç ayrı seviye ölçüldü:

| seviye | durum | kaynak |
|---|---|---|
| **(A)** tamamlanmış düğümler | ✅ **tekrar koşmuyor** | **board** (checkpoint değil) |
| **(B)** yarım kalan **fonksiyon** düğümü | ✅ checkpoint geri yüklenir | checkpoint (`run_one_task`) |
| **(C)** yarım kalan **ajan** düğümü | ✅ kaldığı **turdan** devam | checkpoint (`execute_task`) |

Ölçüm (A) — 'tara' düğümünde çöktürüldü:

```
fetch_source     çökmeden ÖNCE tamamlanmıştı   1 kez  ← tekrar koşmadı
run_test_suite   çökmeden ÖNCE tamamlanmıştı   1 kez  ← tekrar koşmadı
scan_patterns    ÇÖKEN düğüm                   1 kez  ← checkpoint'ten yüklendi
cross_check      çökmeden SONRA koştu          1 kez
→ akış 4/4 tamamlandı
```

Ölçüm (C) — ajan düğümü log satırı:
```
↻ checkpoint'ten DEVAM: turn1'e kadar olan iş TEKRAR KOŞMAYACAK (4 mesaj geri yüklendi)
```

**Motor bazında:** `crash_at` celery/temporal'da yok sayıldığı için checkpoint'e **iz konup** her motorla koşuldu — iz hayatta kalırsa checkpoint geri yüklenmiş demektir:

| motor | önceki turun izi | süre |
|---|---|---:|
| own | ✅ korundu | 1,5 s |
| celery | ✅ korundu | 11,9 s |
| temporal | ✅ korundu | 2,0 s |

Üçü de aynı `run_one_task`/`execute_task` yolundan geçtiği için çalışıyor. **Devam yeteneği bize ait, motora değil.**

> Airflow'da düğüm içi checkpoint **yok** — devam semantiği Airflow'un kendi task retry'ına kalır, düğüm baştan koşar. (Ölçülmedi; ürettiğimiz DAG'ın beyan ettiği ayarlar + Airflow'un bilinen semantiği.)

## 3.6 Zamanlama (cron)

```
✓ cron ayrıştırıcı: 6 doğru an + hafta sonu atlama + 5 geçersiz girdi reddi
✓ next_run_at YAZMA anında türetiliyor
✓ 8 SÜREÇTEN claim alabilen: 1        ← at-most-once
✓ kira dolmadan devralınamıyor (lease=900s)
✓ kira DOLUNCA devralınıyor — ayrı recover_stale() çağrısı YOK
✓ uçtan uca: 5/5 düğüm koştu, takvim ilerledi, geçmişe yazıldı
✓ canlı sunucuda süreçler-arası: 11. saniyede yakaladı
```

**Tasarım farkı (Macro'dan alındı):** bayatlık kontrolü claim'in **içinde**:
```sql
WHERE id=? AND (claimed_at IS NULL OR claimed_at < now-LEASE)
```
Board'daki `claim_next` + ayrı `recover_stale()` ikilisinin tek satırlık hâli. Ayrı adım olmadığı için **atlanması imkânsız** — Temporal'da bulduğumuz bayat-claim hatası tam da "ayrı adım atlandı" sınıfındandı.

## 3.7 Compaction matrisi (bağlam katmanı)

Task management'ın konusu değil ama aynı sistemde ölçüldü ve sonucu önemli:

| strateji | ort. kazanç | mesaj | **kritik bilgi** |
|---|---:|---|---:|
| hermes | %64,8 | **KORUR** | **3/3** ✅ |
| opencode | %0,0 | **KORUR** | **3/3** ✅ |
| openclaw | **%89,2** | BİRLEŞTİRİR | **0/3** ❌ |
| codex | %74,0 | BİRLEŞTİRİR | 1/3 |
| claude_code | %69,7 | BİRLEŞTİRİR | 1/3 |

**Çift bütünlüğü: 30/30 sağlam** (hiçbir `tool_call ↔ tool_result` çifti kırılmadı — kırılsaydı gerçek API 400 döndürürdü).

**Yüksek yüzde ≠ iyi.** Mesajı **koruyan** iki strateji bilgiyi de koruyor; **birleştiren** üçü kaybediyor.

---

# BÖLÜM 4 — Bulunan 12 hata

| # | hata | sınıf | nasıl bulundu |
|---|---|---|---|
| 1 | çökme enjeksiyonu fonksiyon düğümlerinde hiç tetiklenmiyordu | mimari değişti, çağıran güncellenmedi | hata testi |
| 2 | Celery dispatch'te `on_event` yok → SSE zaman aşımı | aynı | sohbet matrisi |
| 3 | dalga-bitti kontrolü kaldırılınca 25 sn takılmalar | regresyon | sohbet matrisi |
| 4 | **board `busy_timeout` yok** → çok-süreçli kilit | eksik yapılandırma | concurrency |
| 5 | **`spawn_task` tamamen kırıktı** (varsayılan `kind` değişmiş) | mimari değişti | yaşam döngüsü |
| 6 | Codex düşük bütçede yetim tool çifti → API 400 | sınır koşulu | compaction matrisi |
| 7 | **batan dalın ardındakiler sonsuza dek `blocked`** | eksik durum modeli | hata testi |
| 8 | **`complete()`/`fail()` fencing yok** → zombi worker yazabiliyor | eksik eşzamanlılık koruması | hata testi |
| 9 | uydurma fonksiyon adı board'a yazılıyor, 3 kez deneniyor | eksik doğrulama | hata testi |
| 10 | kalıcı/geçici hata ayrımı yok, breaker boşa harcanıyor | eksik sınıflandırma | hata testi |
| 11 | **yinelenen bağımlılık kapıyı kalıcı kilitliyor** | küme/liste karışıklığı | canlı koşu |
| 12 | **Temporal'da geçici hata kalıcıya dönüşüyordu** | bayat claim | retry ölçümü |

**Dört tanesi** *"mimari değişti, çağıran güncellenmedi"* sınıfı — fonksiyon-öncelikli geçiş ve paket refactor'ü, görünürde çalışan ama hiç test edilmeyen yolları sessizce kırmıştı.

## 4.1 En ciddi bulgu — sessizce yanlış rapor

Hata enjeksiyonu **olmadan**, normal bir koşuda çıktı. Planlayıcı `cross_check`'i `scan_patterns`'a bağlamamıştı; `_merge_up` eksik kenarda sessizce varsayılana düşüyordu:

> **Bir güvenlik denetimi pipeline'ı, tarayıcısı hiç veri vermemişken "Taranan desen eşleşmesi: 0" yazan tertemiz bir rapor üretti.**

Denetim raporunda "0 eşleşme" = "açık bulunamadı". Hata yok, uyarı yok. **Yanlış cevap, çökmeden tehlikelidir** — çökme alarm üretir, sessiz yanlış sonuç güven kazanır.

Üç katmanlı düzeltme:
1. **Zorunlu upstream sözleşmesi** (`functions.NEEDS`) — eksikse düğüm patlar, varsayılanla koşmaz
2. **Katalogda kenar rehberi** — planlayıcı hangi düğüme bağlanacağını görüyor
3. **Plan doğrulama + otomatik kenar onarımı** (`dogrula_dag`) — yürütmeden önce eksik kenar bulunup eklenir

| | ÖNCE | SONRA |
|---|---|---|
| rapor | `eşleşmesi: **0**` · `test: **0**` | **`4`** · **`240`** |
| uyarı | yok | `⚠ EKSİK KENAR ONARILDI` |

Son canlı koşuda planlayıcı kenarları **kendiliğinden doğru kurdu (0 onarım)** — katalog rehberi işe yarıyor.

---

# BÖLÜM 5 — Koçun 6 ekseni

| eksen | durum | kanıt |
|---|---|---|
| task management | ✅ | board FSM + DAG + 42 kontrol |
| retry / kurtarma | ✅ | retry anatomisi + yan etki ölçümü + 3 seviye devam |
| durum takibi | ✅ | olay günlüğü, her geçiş kayıtlı |
| concurrency | ✅ | **6 gerçek süreç, 0 çift claim, 4,7×** |
| işletme karmaşıklığı | ✅ | 0,01 s / 0,5 s / 15–40 s ölçüldü |
| **scheduling** | ✅ | **cron + claim/lease + 43 kontrol** |

**6/6 ölçülmüş kanıta dayanıyor.** (Önceki turlarda scheduling "yok" diye belgelenmişti; bu turda eklendi.)

---

# BÖLÜM 6 — Bilinen sınırlar

Kapatılmamış, bilinçli olarak kapsam dışı bırakılan noktalar:

| konu | durum | etki |
|---|---|---|
| `crash_at` celery/temporal'da yok sayılıyor | **yanıltıcı API** | çökme kurtarma o iki motorda ölçülmedi |
| retry'da **backoff yok** (0 sn) | eksik | dış servis düğümü eklenirse rate-limit derinleşir |
| zamanlamada **timezone yok** | eksik | sunucunun yerel saati kullanılıyor |
| zamanlamada **catchup yok** | eksik | sunucu kapalıyken geçen vakit atlanır |
| geçersiz argüman sessizce yutuluyor | orta | `create_task` yolunda uyarı yok |
| board dosyası silinse yazmalar "başarılı" döner | orta | POSIX unlink semantiği; sunucu-tabanlı DB'de yok |
| SQLite tek yazar | mimari | çok yüksek eşzamanlılıkta darboğaz |

**Airflow için ölçüm yok** — bizim katmanda yürütmediği için hata/retry/çökme davranışı ölçülemedi. Bu belgedeki Airflow satırları ürettiğimiz DAG'ın beyan ettiği ayarlara ve Airflow'un bilinen semantiğine dayanır.

---

# Sunumda söylenebilecek üç cümle

> **1.** Task yönetimini dört motorda takılabilir hâle getirdik; ölçtük ki hata davranışı motordan **bağımsız** — çünkü karar board'da tek noktada veriliyor. Motorlar hızda ayrışıyor: 0,01 s / 0,5 s / 15–40 s.

> **2.** Sistemi başarılı koşuda değil **patladığında** ölçtük: 190 kontrol, 12 hata. Dördü "mimari değişti, çağıran güncellenmedi" sınıfıydı — görünürde çalışıyorlardı.

> **3.** En ciddi bulgu bir çökme değil, bir **sessizlikti**: eksik bir DAG kenarı yüzünden "0 bulgu" diyen temiz bir denetim raporu. Artık düğüm ihtiyacı olan veriyi alamazsa patlıyor.

---

## Testleri koşturma

```bash
.venv/bin/python demo-brain-agent/test_tasklife.py          # 42
.venv/bin/python demo-brain-agent/test_hata.py              # 54  (sunucu 8030 açıkken)
.venv/bin/python demo-brain-agent/test_zamanlama.py         # 43
.venv/bin/python demo-brain-agent/test_concurrency.py       # 6 süreç
.venv/bin/python demo-brain-agent/test_retry.py             # retry anatomisi
.venv/bin/python demo-brain-agent/test_backend_hata.py      # dört motor
.venv/bin/python demo-brain-agent/test_devam.py             # çökme sonrası devam
.venv/bin/python demo-brain-agent/test_compaction_matrix.py # 6×5
```

**İlgili raporlar:** `hata-dayanikliligi-test-raporu.md` · `retry-olcum-raporu.md` · `zamanlama-cron-raporu.md` · `kapsamli-test-raporu-tur2.md` · `macro-analiz-bizim-caseler.md`
