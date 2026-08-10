# Kapsamlı Test Raporu — 2. tur (baştan sona tüm durumlar)

> **Hedef:** 1. turda test edilmeyen ya da test edilemeyen **tüm yolları** kapatmak.
> 1. tur sohbet arayüzü üzerinden 26 senaryo koşmuştu; bu tur **board/orchestrator
> seviyesine inip** sohbetten ulaşılamayan yolları, gerçek çok-süreçli eşzamanlılığı ve
> compaction'ın tam matrisini sınadı.
>
> **Toplam:** 4 test paketi · **104 kontrol** · **5 yeni bug** bulundu ve düzeltildi.

---

## Özet tablo

| Test paketi | Kontrol | Sonuç | Bulunan bug |
|---|---:|---|---|
| Çok-süreçli CAS yarışı (`test_concurrency.py`) | 4 | ✅ tamamı | **1** (busy_timeout) |
| Task yaşam döngüsü (`test_tasklife.py`) | 40 | ✅ 40/40 | **1** (spawn_task kırık) |
| Compaction matrisi (`test_compaction_matrix.py`) | 30×2 | ✅ 30/30 çift bütünlüğü | **1** (Codex yetim çift) |
| Pipeline yaşam döngüsü (3 backend) | 3 | ✅ tamamı | (tur-1 bug'ları doğrulandı) |

Ek olarak 1. turda bulunan 3 bug'ın düzeltmesi bu turda **regresyon testinden geçti**.

---

## 1 · Çok-süreçli eşzamanlılık — artık KANITLI

Şimdiye kadar "CAS-claim at-most-once verir" **tek süreçte** gösteriliyordu — yani argümandı,
kanıt değildi. Bu turda 6 **ayrı işletim sistemi süreci** aynı board'a saldırdı.

```
6 ayrı SÜREÇ · 24 bağımsız task
  worker-0 (pid 59423) →  4 task     worker-3 (pid 59426) →  4 task
  worker-1 (pid 59424) →  4 task     worker-4 (pid 59427) →  4 task
  worker-2 (pid 59425) →  4 task     worker-5 (pid 59428) →  4 task

  toplam claim      : 24
  benzersiz task    : 24
  ÇİFT claim edilen : 0        ← ✓ at-most-once korundu
  tamamlanan        : 24/24
```

| Kontrol | Sonuç |
|---|---|
| At-most-once (çift claim yok) | ✅ 0 çakışma |
| Hepsi tamamlandı | ✅ 24/24 |
| **Gerçek paralellik** | ✅ tek süreç 1,26 sn → 6 süreç 0,28 sn = **4,6×** |
| Gerçek süreç ölümü → devralma | ✅ `os._exit(1)` → `recover_stale` → başka worker aldı, checkpoint korundu |

### 🐛 BUG 4 — board çok-süreç güvenli DEĞİLDİ

İlk koşu **kilitlendi**. Kök neden:

```python
self.conn.execute("PRAGMA journal_mode=WAL")   # vardı
# busy_timeout YOKTU → varsayılan 0
```

SQLite'ın varsayılan `busy_timeout` **0**'dır: iki süreç aynı anda yazmaya kalkınca anında
`SQLITE_BUSY` fırlatır. **WAL tek başına yetmez** — WAL okur-yazar çakışmasını çözer ama
yazarlar yine serileşir; bekleme süresi verilmezse hata alırlar.

Yani board'un "CAS-claim ile çok worker" iddiası **pratikte çalışmıyordu** ve bunu tek-süreçli
testler asla gösteremezdi.

**Düzeltme:**
```python
self.conn.execute("PRAGMA busy_timeout=10000")   # 10 sn bekle, hata verme
self.conn.execute("PRAGMA synchronous=NORMAL")
```

Düzeltmeden sonra 6 süreç × 24 task **0,28 sn**'de, sıfır çakışmayla tamamlandı.

> **Ders:** Bu, koçun 6 ekseninden "concurrency" satırının artık **ölçülmüş** olması demek.
> Önceki raporlarda bu satır dokümantasyona dayanıyordu.

---

## 2 · Task yaşam döngüsü — 40/40

Sohbet arayüzünden ulaşılamayan yollar doğrudan board seviyesinde sınandı.

| Alan | Kontrol | Sonuç |
|---|---:|---|
| **Retry** | 6 | ✅ hata → `ready`, attempt++, breaker++, yeniden claim, başarıda breaker sıfırlanır, olay günlüğüne `retry_scheduled` düşer |
| **Circuit-breaker** | 3 | ✅ `['ready','ready','failed']` — limitte kapanıyor, `failed` task artık claim edilemiyor, **sonsuz retry yok** |
| **Çökme → checkpoint → devralma** | 6 | ✅ fonksiyon düğümünde çökme tetikleniyor, checkpoint dolu, task `running` kalıyor, `recover_stale` topluyor, checkpoint **kurtarma sonrası korunuyor** |
| **DAG kapısı** | 5 | ✅ bağımlı task `blocked` doğuyor, claim edilemiyor, parent bitmeden `recompute_ready` açmıyor, bitince açıyor |
| **spawn_task + frenler** | 6 | ✅ 2 kabul → 3. **reddedildi** (task başına fren), board tavanı dolunca reddediliyor, üretilenler `worker:` ile işaretli |
| **Sınır/hata durumları** | 8 | ✅ bilinmeyen fn · bozuk JSON · olmayan bağımlılık · tanımsız argüman (board'a yazılmıyor) · boş başlık · `fn`siz function · yürütücü de reddediyor |
| **Veri akışı** | 4 | ✅ upstream görünüyor, referans (path) taşınıyor, **ham içerik taşınmıyor**, alt düğüm kullandı (`count=4`) |
| **Scheduling** | 2 | ⚠️ **UYGULANMAMIŞ** — dürüst tespit |

### 🐛 BUG 5 — `spawn_task` tamamen kırıkmış

```
✗ t_spawn PATLADI: ValueError: kind='function' için fn (fonksiyon adı) zorunlu
```

Paket refactor'ünde `create_task`'ın varsayılan `kind`'ı `"function"` yapılmıştı. `spawn_task`
ise `kind`/`fn` vermiyordu → **her çağrıda exception**. Yani "ajan yürütme sırasında yeni task
üretebilir" özelliği (daha önce çalıştığını gösterdiğimiz) sessizce ölmüştü.

**Düzeltme:** `spawn_task` artık `kind="agent"` ile açıyor — keşfedilen iş doğal dille tarif
edilir, hazır bir fonksiyona karşılık gelmez, dolayısıyla doğru tür ajan düğümüdür.

> **Ders:** İki mimari değişiklik (fonksiyon-öncelikli + paketler) iki ayrı özelliği sessizce
> kırdı (BUG 5 burada, tur-1'deki BUG 1 çökme enjeksiyonunda). İkisi de "varsayılan değer
> değişti, çağıran güncellenmedi" sınıfı.

### Scheduling — dürüst sonuç

```
✓ zamanlanmış koşu (cron/interval) UYGULANMIŞ mı   beklenen=False  gerçek=False
✓ board'da zaman alanı (next_run/schedule) var mı  beklenen=False  gerçek=False
```

Sistemde **hiç yok**. Koçun 6 ekseninden biri hâlâ açık — kodda `cron` kelimesi yalnız
yorumlarda geçiyor. Sunumda "6 eksende ölçtük" denemez; **5 eksen ölçüldü, scheduling yok**.

---

## 3 · Compaction matrisi — 6 strateji × 5 bütçe = 30 koşu

Aynı iz (11 mesaj, 1.473 token, içinde gerçek bir MFA bug'ı) üstünde. Ölçülen sadece yüzde
değil: **çift bütünlüğü** ve **kritik bilginin hayatta kalması**.

| Strateji | Tetiklendiği bütçe | Ort. kazanç | Mesaj | Kritik bilgi korundu |
|---|---:|---:|---|---:|
| `none` | 0/5 | — | KORUR | 5/5 |
| `hermes` | 3/5 | %64,8 | **KORUR** | **3/3** |
| `opencode` | 3/5 | %0,0 | **KORUR** | **3/3** |
| `openclaw` | 3/5 | **%89,3** | BİRLEŞTİRİR | **0/3** |
| `codex` | 3/5 | %74,4 | BİRLEŞTİRİR | 1/3 |
| `claude_code` | 3/5 | %69,4 | BİRLEŞTİRİR | 1/3 |

### 🐛 BUG 6 — Codex düşük bütçede `tool_call ↔ tool_result` çiftini kırıyordu

```
ÇİFT BÜTÜNLÜĞÜ: 28/30 sağlam
  ✗ KIRILANLAR: codex@200 · codex@400  → "1 tool sonucu çağrısız"
```

Windowing yaparken kuyruk `out[-keep:]` ile **körlemesine** kesiliyordu; bir `tool_result`
kendi assistant çağrısından koparılabiliyordu. Gerçek kullanımda sağlayıcı bu isteği
**400 ile reddeder** — yani ajan tamamen durur.

**Düzeltme:** kuyruk kesilirken yetim tool sonucu bırakılmıyor; ya çağrı mesajı da alınıyor
ya da yetim sonuç düşürülüyor.

**Düzeltme sonrası: 30/30 sağlam.**

### Asıl bulgu: yüksek yüzde ≠ iyi

**Tetiklenen 15 koşunun 7'sinde kritik bilgi (asıl bug satırı) KAYBOLDU** — hepsi agresif
stratejilerde:

```
kaybedenler: openclaw@200, openclaw@400, openclaw@1000,
             codex@200, codex@400, claude_code@200, claude_code@400
```

`openclaw` %89,3 ile en yüksek kazancı verdi ama **3/3 koşuda bilgiyi kaybetti**.
`hermes` %64,8 ile daha az kazandı ama **3/3 korudu** — çünkü tail'i koruyor ve mesaj silmiyor.

Bu, §I.5.1'de yazdığımız "seyreltenler vs pencere kapatanlar" ayrımının **ölçülmüş hali**:
mesajı **KORUYAN** iki strateji (hermes, opencode) bilgiyi de koruyor; **BİRLEŞTİREN** üçü
kaybediyor.

---

## 4 · Pipeline yaşam döngüsü — 3 backend

Aynı kayıtlı akış (5 düğüm, audit) üç backend'de yeniden koşturuldu:

| Backend | Sonuç | Süre |
|---|---|---:|
| own | 5/5 düğüm | **0,0 sn** |
| temporal | 5/5 düğüm | 0,6 sn |
| celery | 5/5 düğüm | 40,3 sn |

**Celery artık kopmuyor** — 1. turda 120 sn'de SSE koparak sonuçsuz kalıyordu (BUG 2/3).
Hâlâ diğerlerinden ~70× yavaş (worker süreci + broker gecikmesi), ama **doğru çalışıyor**.

---

## Bulunan tüm bug'lar (iki tur toplam)

| # | Tur | Bug | Kök neden | Etki |
|---|---|---|---|---|
| 1 | 1 | Çökme enjeksiyonu fonksiyon düğümlerinde çalışmıyor | `crash_after_turn` yalnız ajan dalında | Çökme/kurtarma yolu **test edilemez** hale gelmişti |
| 2 | 1 | Celery dispatch sessiz → SSE 120 sn'de koparıyor | `on_event` geçilmiyordu | Celery koşuları sonuçsuz görünüyordu |
| 3 | 1 | Celery dalga-bitti kontrolü yok | 25 sn boşa bekleme | 78 sn → 29 sn |
| 4 | 2 | **Board çok-süreç güvenli değil** | `busy_timeout` ayarlanmamış (varsayılan 0) | Çok worker'lı kullanım **kilitleniyordu** |
| 5 | 2 | **`spawn_task` tamamen kırık** | `create_task` varsayılanı `function` olunca `fn` zorunlu oldu | Yürütme-anında task üretimi **her çağrıda patlıyordu** |
| 6 | 2 | **Codex yetim tool çifti üretiyor** | windowing kuyruğu körlemesine kesiyor | Gerçek API'de **400 hatası** → ajan durur |

**Ortak örüntü:** 6 bug'ın 4'ü *"varsayılan/mimari değişti, çağıran güncellenmedi"* sınıfı.
Fonksiyon-öncelikli geçiş ve paket refactor'ü, görünürde çalışan ama **hiç test edilmeyen**
yolları sessizce kırmış.

---

## Koçun 6 ekseni — güncel durum

| Eksen | Durum | Kanıt |
|---|---|---|
| Task yönetimi | ✅ ölçüldü | FSM + DAG kapısı 40/40 kontrol |
| Retry / recovery | ✅ ölçüldü | retry + breaker + checkpoint'ten devam, fonksiyon düğümü dahil |
| State takibi | ✅ ölçüldü | olay günlüğü, `created→claimed→recovered→claimed→completed` |
| **Concurrency** | ✅ **artık ölçüldü** | 6 süreç, 0 çakışma, 4,6× hızlanma |
| Operasyonel karmaşıklık | ✅ pratikte görüldü | own: tek dosya · temporal: dev server · celery: broker+süreç |
| **Scheduling** | ❌ **YOK** | kodda hiç uygulanmamış — dürüst tespit |

**5/6 eksen ölçülmüş kanıta dayanıyor. Scheduling hâlâ açık** ve bu, sunumda açıkça
söylenmeli.

---

## Kalan açıklar

1. **Scheduling hiç yok** — cron/backfill uygulanmamış. En büyük eksik.
2. **Router bazen graf yerine tool döngüsü seçiyor** (deploy senaryosu) → kayıtlı pipeline
   üretilmiyor. Sonuç doğru ama tekrar kullanılabilirlik kayboluyor.
3. **Celery ~70× yavaş** yeniden koşumda — mimari (broker + süreç) gereği, düzeltilebilir değil.
4. **Ajan düğümü (kind=agent) grafta neredeyse hiç üretilmiyor** — planlayıcı hep deterministik
   fonksiyon seçiyor. Doğru davranış ama graf-içi compaction az test edilmiş kalıyor.
5. **SQLite tek yazar** — 6 süreçte iyi; çok daha yüksek eşzamanlılıkta Postgres'e taşımak gerekir.

---

## Çalıştırma

```bash
.venv/bin/python demo-brain-agent/test_concurrency.py 6 24    # çok-süreçli CAS yarışı
.venv/bin/python demo-brain-agent/test_tasklife.py            # 40 kontrol
.venv/bin/python demo-brain-agent/test_compaction_matrix.py   # 6×5 matris
.venv/bin/python demo-brain-agent/test_matrix.py              # sohbet üzerinden 26 senaryo
```

**Ham sonuçlar:** `test_concurrency_sonuc.json` · `test_tasklife_sonuc.json` ·
`test_compaction_sonuc.json` · `test_sonuclari.json`
**1. tur raporu:** `report/pipeline-test-raporu.md`
