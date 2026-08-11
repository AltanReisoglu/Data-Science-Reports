# Task Management Motorları — Terimler, Saf Hâli, Eklediklerimiz ve Tüm Test Sonuçları

**Kapsam:** hermes type (kendi çekirdeğimiz) · Airflow · Celery · Temporal
**Yöntem:** Her motorun **kutudan ne çıktığı** entegrasyon kodundan doğrulandı, üstüne **bizim eklediğimiz** ayrıldı, sonuç **ölçüldü**.
**Ölçüm:** 10 Ağustos 2026 · aynı 5 düğümlü elmas graf, aynı hata enjeksiyonu, dört motor.
**Toplam:** 9 test paketi · **190 kontrol** · hepsi geçti · **12 hata** bulundu ve düzeltildi.

---

# BÖLÜM 0 — SÖZLÜK

Belgede geçen her terim. Sıra: önce genel kavramlar, sonra bizim mekanizmalarımız, sonra motorlara özgü terimler.

## 0.1 Genel kavramlar

| terim | kısa açıklama |
|---|---|
| **task** | Yapılacak tek bir iş kalemi. Bizde board'da bir satır. |
| **düğüm (node)** | Graftaki bir task. "task" ile eş anlamlı kullanılıyor. |
| **DAG** | *Directed Acyclic Graph* — yönlü, çevrimsiz graf. "A bitmeden B başlamaz" ilişkilerinin bütünü. Çevrim olmaz, yoksa akış kilitlenir. |
| **bağımlılık (parent)** | Bir düğümün beklediği önceki düğüm. `parents=[t_a]` → `t_a` bitmeden bu düğüm başlamaz. |
| **upstream / downstream** | Üst akış = beni besleyen düğümler; alt akış = benden beslenen düğümler. |
| **worker** | Task'ı fiilen koşturan işçi (süreç ya da iş parçacığı). |
| **dispatcher** | Hangi task'ın kime verileceğine karar veren döngü. |
| **broker** | Mesaj kuyruğu (Redis/RabbitMQ). Celery task'ları buraya atar, worker'lar buradan çeker. |
| **idempotens** | Aynı işi iki kez yapmanın bir kez yapmakla aynı sonucu vermesi. Retry varsa şart. |
| **yan etki** | Fonksiyonun dışarıya kalıcı etkisi (e-posta gönderme, ödeme alma, dosya yazma). Tekrarlanırsa zarar verir. |
| **deterministik** | Aynı girdiye hep aynı çıktı. Düğüm fonksiyonlarımız böyle; LLM değil. |

## 0.2 Teslim garantileri

| terim | kısa açıklama |
|---|---|
| **at-most-once** | "En fazla bir kez" — bir task iki worker'a **aynı anda** verilmez. Kaybolabilir ama çiftlenmez. |
| **at-least-once** | "En az bir kez" — task mutlaka teslim edilir, ama çiftlenebilir. Celery'nin `acks_late` ile verdiği garanti. |
| **exactly-once** | "Tam bir kez" — pratikte at-least-once + idempotens ile elde edilir. Temporal replay'de tamamlanan activity'yi atlayarak buna yaklaşır. |

## 0.3 Bizim mekanizmalarımız

| terim | kısa açıklama |
|---|---|
| **board** | `taskboard.py` — SQLite üstündeki iş panosu. Task'lar, durumları, bağımlılıkları burada. Sistemin tek doğruluk kaynağı. |
| **FSM** | *Finite State Machine* — sonlu durum makinesi. Task'ın geçebileceği durumlar ve geçişleri sabit: `blocked → ready → running → done/failed/cancelled`. |
| **blocked** | Bağımlılığı henüz bitmemiş, sıraya bile giremeyen task. |
| **ready** | Bağımlılıkları bitmiş, alınmayı bekleyen task. |
| **running** | Bir worker tarafından alınmış, koşan task. |
| **done** | Başarıyla tamamlanmış. |
| **failed** | Breaker doldu ya da kalıcı hata; bir daha denenmeyecek. |
| **cancelled** | Üstündeki düğüm battığı için **asla koşmayacak**. "Bekliyor"dan (`blocked`) ayrı tutulur — bu ayrım BUG 7'de eklendi. |
| **CAS-claim** | *Compare-And-Swap* claim. `UPDATE … WHERE claim_lock IS NULL` — koşullu güncelleme. İki worker aynı anda denerse veritabanı tek UPDATE uygular; biri alır, diğeri boş döner. Dağıtık kilit gerekmez. |
| **claim_lock** | Task'ı hangi worker'ın aldığını gösteren alan. Boşsa task sahipsiz. |
| **claim_expires** | Claim'in ne zaman geçersizleşeceği (lease sonu). |
| **lease (kira)** | Claim'in ömrü. Worker bu süre içinde bitirmeli ya da nabız atmalı, yoksa task başkasına geçebilir. Board'da **30 sn**, zamanlayıcıda **900 sn**. |
| **heartbeat (nabız)** | "Hâlâ çalışıyorum" sinyali — lease'i uzatır. Bizde yalnız **ajan düğümünde**, tur başına atılır (bkz. §5.1). |
| **recover_stale** | Bayat claim'leri toplayan süpürme. Koşul: lease dolmuş **VEYA** worker PID'i ölü. |
| **PID canlılık kontrolü** | `os.kill(pid, 0)` ile sürecin yaşayıp yaşamadığını sorma. Ölü worker'ı lease dolmadan yakalamayı sağlar — diğer motorlarda yok. |
| **fencing** | Bayat worker'ın yazmasını engelleme. `complete/fail … AND claim_lock=?` — claim devredildiyse eski sahibin yazması reddedilir. BUG 8'de eklendi. |
| **stale write (bayat yazma)** | Lease'i dolmuş worker'ın uyanıp sonuç yazmaya çalışması. Fencing yoksa devralanın sonucunu ezer. |
| **circuit-breaker** | "Sigorta". Arka arkaya `BREAKER_LIMIT=3` hata → task `failed`. Sonsuz retry'ı önler. Araya bir başarı girerse sayaç sıfırlanır. |
| **attempt** | Toplam deneme sayısı. Başarıdan sonra **sıfırlanmaz** — denetim izi olarak kalır. |
| **consecutive_failures** | Arka arkaya hata sayacı. Breaker bunu kullanır; başarıda sıfırlanır. |
| **kalıcı / geçici hata** | Geçici = tekrar denemekle düzelebilir (ağ). Kalıcı = düzelmez (bilinmeyen fonksiyon, eksik zorunlu veri). Kalıcı olan tek denemede kapanır, breaker'ı harcamaz. BUG 10'da eklendi. |
| **checkpoint** | Yarım kalan işin kaydı. Çökme sonrası devralan worker buradan devam eder, baştan başlamaz. |
| **iptal zinciri** | Bir düğüm `failed` olunca **tüm alt soyunu** `cancelled` yapma. `cancel_downstream()`. |
| **olay günlüğü** | Her durum geçişinin kaydı (`created`, `claimed`, `retry_scheduled`, `failed`, `cancelled`, `completed`, `stale_write_reddedildi`). Denetim izi ve hata ayıklama için. |
| **recompute_ready** | Bağımlılığı biten `blocked` task'ları `ready`ye terfi ettiren tarama. "DAG kapısı" bu. |
| **upstream_results** | Parent'ların sonuçlarını çocuğa geçirme. Airflow'daki XCom'un karşılığı. **Yalnız doğrudan parent'lardan** akar — dedenin çıktısı torununa otomatik geçmez. |
| **zorunlu upstream sözleşmesi** | `functions.NEEDS` — bir düğümün olmazsa olmaz girdi anahtarları. Eksikse düğüm patlar, varsayılanla koşmaz. "Sessizce yanlış rapor" bulgusundan sonra eklendi. |
| **dogrula_dag** | Yürütmeden önce eksik veri kenarını bulup graftaki üreticiyi parent olarak ekleyen doğrulama. |
| **kind=function / kind=agent** | Düğüm türü. `function` = deterministik, LLM yok (**varsayılan**). `agent` = LLM muhakemesi (istisna). |
| **düğüm fonksiyonu** | DAG düğümü olarak koşan kayıtlı fonksiyon (`fetch_source`, `scan_patterns`…). Motor çağırır, LLM değil. |
| **fonksiyon paketi (pack)** | Alan bazlı fonksiyon kümesi: **audit** (6) · **data** (5) · **deploy** (5). |
| **spawn_task / add_step** | Ajanın çalışma anında task üretmesini sağlayan tool'lar. `add_step` planlama turunda, `spawn_task` yürütme sırasında. |
| **plan turu / dispatch turu** | Önce ajan grafı kurar (plan), sonra motor yürütür (dispatch). |

## 0.4 Zamanlama terimleri

| terim | kısa açıklama |
|---|---|
| **cron** | Tekrarlayan zamanı tanımlayan ifade. Bizde 5 alan: `dk sa gün ay hafta`. `0 8 * * 1-5` = hafta içi 08:00. |
| **next_run_at** | Bir sonraki tetikleme anı. **Yazma anında** cron'dan türetilip saklanır (okuma anında ayrıştırılmaz). |
| **poller (yoklayıcı)** | Periyodik olarak "vakti gelen var mı?" diye soran arka plan döngüsü. |
| **backfill / catchup** | Geçmişe dönük koşu — sistem kapalıyken kaçırılan zamanları sonradan çalıştırma. Airflow'da var, **bizde yok**. |
| **enabled** | Zamanlamanın etkin olup olmadığı. Kapalıysa yoklayıcı atlar. |
| **BATCH_MIN_SECONDS** | Bir yoklama turunun asgari süresi. Tek örneğin kuyruğu süpürmesini engeller (adil dağıtım). |

## 0.5 Motorlara özgü terimler

| terim | motor | kısa açıklama |
|---|---|---|
| **acks_late** | Celery | Onayı (ack) iş **bittikten sonra** verme. Worker çökerse mesaj yeniden teslim edilir → at-least-once. |
| **max_retries** | Celery | Bir task'ın en fazla kaç kez yeniden denenebileceği. |
| **countdown** | Celery | Retry'dan önce beklenecek süre. Bizde `0` → bekleme yok. |
| **prefetch_multiplier** | Celery | Worker'ın önden kaç mesaj çektiği. `1` = birer birer, adil dağıtım. |
| **result_backend** | Celery | Sonuçların saklandığı yer. Bizde `None` — sonuçları board tutuyor. |
| **filesystem broker** | Celery | Dosya sistemi üstünden mesajlaşma. Tuzağı: `data_folder_in == data_folder_out` **şart**. |
| **workflow** | Temporal | Dayanıklı (durable) iş akışı gövdesi. Çökse bile replay ile devam eder. |
| **activity** | Temporal | Workflow'un çağırdığı gerçek iş birimi. IO ve LLM çağrıları **yalnız burada** olabilir. |
| **replay** | Temporal | Çökme sonrası workflow'u olay geçmişinden yeniden oynatma. Tamamlanmış activity'ler **atlanır**. |
| **determinizm disiplini** | Temporal | Workflow gövdesinde rastgelelik/IO/saat okuma **yasak** — replay aynı sonucu vermeli. |
| **RetryPolicy** | Temporal | Activity seviyesinde retry kuralı. Bizde `maximum_attempts=3`, `initial_interval=200 ms`. |
| **event history** | Temporal | Kutudan çıkan tam denetim izi. |
| **Temporal Schedules** | Temporal | Kutudan gelen cron + backfill. |
| **PythonOperator** | Airflow | Bir Python fonksiyonunu düğüm olarak koşturan operatör. |
| **XCom** | Airflow | Düğümler arası veri taşıma (*cross-communication*). Bizim `upstream_results`'ın karşılığı. |
| **trigger_rule** | Airflow | Bir düğümün ne zaman koşacağı kuralı. Varsayılan `all_success` → üst battıysa düğüm `upstream_failed` olur (bizim `cancelled`'ımızın karşılığı). |
| **upstream_failed** | Airflow | Üstü battığı için koşmayacak düğüm. |
| **catchup** | Airflow | Geçmişe dönük koşu. Ürettiğimiz DAG'da `False`. |
| **parse zamanı** | Airflow | DAG dosyasının okunduğu an. Graf **burada** sabitlenir — çalışırken değişemez. |
| **DAG donması** | Airflow | Grafın parse zamanında sabitlenmesi; ajanın çalışma anında düğüm ekleyememesi. |

## 0.6 Ölçüm terimleri

| terim | kısa açıklama |
|---|---|
| **yürütme denemesi** | Board'daki `claimed` olayı sayısı. Süreçler arası geçerli olduğu için backend karşılaştırmasında bu kullanıldı (süreç-içi sayaç Celery'yi göremiyor). |
| **çift claim** | Aynı task'ın iki worker tarafından alınması. At-most-once ihlali. Ölçüm: **0**. |
| **hızlanma** | Tek süreç süresi ÷ çok süreç süresi. |
| **çift bütünlüğü** | Compaction sonrası her `tool_call`'un karşılık gelen `tool_result`'ının durması. Kırılırsa gerçek API 400 döndürür. |
| **kritik bilgi** | İzdeki asıl bulgu satırı (`mfa_token is None`). Compaction sonrası hayatta mı? |
| **at-emission / at-threshold** | Kırpmanın ne zaman tetiklendiği: çıktı üretilirken mi, birikim eşiği aşılınca mı. |

---

# BÖLÜM 1 — Ana tez

> **Motorların hiçbiri "task yönetimi" vermiyor. Üçü de yalnız *yürütme* veriyor.**
> Bağımlılık, durum, retry kararı, checkpoint, iptal — hepsi **board'da**, yani bizde.

Hata davranışının motordan bağımsız çıkmasının sebebi bu: karar tek noktada veriliyor, motor sadece "koştur" diyor. Motorlar **ne yaptığında** değil, **ne kadar sürdüğünde ve neyi işletmek gerektiğinde** ayrışıyor.

## Rol paylaşımı

| sorumluluk | hermes type | Airflow | Celery | Temporal |
|---|---|---|---|---|
| task'ın kendisi | **board** | Airflow | **board** | **board** |
| bağımlılık (DAG) | **board** | Airflow | **board** ← Celery'de yok | **board** ← Temporal'da yok |
| durum takibi | **board** | metadata DB | **board** ← Celery'de yok | event history + board |
| retry kararı | **board** (breaker) | Airflow | Celery + board | Temporal + board |
| checkpoint | **board** | **hiç yok** | **board** ← Celery'de yok | **board** |
| iptal zinciri | **board** | `upstream_failed` | **board** | **board** |
| dağıtım | board (CAS) | executor | **Celery** | task queue |
| durable kayıt | SQLite | metadata DB | **yok** | **Temporal** |

Koyu yazılanlar bizim yazdığımız katman.

---

# BÖLÜM 2 — HERMES TYPE (kendi çekirdeğimiz)

`taskboard.py` + `orchestrator._dispatch_own` · ~600 satır, tek SQLite dosyası.

## Saf hâli

Böyle bir motor kutudan çıkmıyor — **tamamı bizim**. Bu bölümde "saf/eklenen" ayrımı yok; diğerlerini değerlendirirken referans noktamız bu.

## İçindekiler

| mekanizma | uygulama | neden |
|---|---|---|
| FSM | `blocked → ready → running → done/failed/cancelled` | durum tek yerde |
| DAG kapısı | `parents` + `recompute_ready()` | parent `done` olmadan çocuk açılmaz |
| CAS-claim | `WHERE claim_lock IS NULL` | at-most-once, dağıtık kilit yok |
| lease + heartbeat | 30 sn (bkz. §5.1 — **sorunlu**) | çöken worker'ın işi geri kuyruğa |
| çökme kurtarma | lease dolmuş **VEYA PID ölü** | 30 sn beklemeden devralma |
| fencing | `… AND claim_lock=?` | bayat worker yazamaz |
| circuit-breaker | arka arkaya 3 hata → `failed` | sonsuz retry yok |
| kalıcı/geçici ayrım | `ValueError` → tek denemede kapat | sözleşme hatası tekrarla düzelmez |
| iptal zinciri | tüm alt soy `cancelled` | "bekliyor" ≠ "asla koşmayacak" |
| checkpoint | fonksiyon + ajan düğümü | çökmede iş tekrarlanmaz |
| olay günlüğü | her geçiş | denetim izi |
| zamanlama | `scheduler.py` cron + atomik claim | koçun 6. ekseni |

## İyi yanları

- **Kurulum sıfır** — tek dosya, broker yok, cluster yok
- **En hızlı** — 5 düğüm **0,01 sn** (Temporal 0,5 s, Celery 15,3 s)
- **Tam dinamik** — ajan çalışırken task üretebiliyor, graf büyüyebiliyor
- **Her şey görünür** — `sqlite3 board.db` ile okunabilir, kara kutu yok
- **PID canlılık kontrolü** — diğerlerinde yok; ölü worker lease beklenmeden yakalanır

## Kötü yanları

- **SQLite tek yazar** — çok yüksek eşzamanlılıkta darboğaz
- **Tek makine** — yatay ölçek yok
- **Garanti kendi kodumuz kadar** — 12 hatanın 8'i bu katmanda çıktı
- **İşletim aracı yok** — Airflow'un operatör UI'sı gibi bir şey yok
- **Board dosyası silinirse yazmalar "başarılı" döner** (POSIX unlink semantiği)
- **Lease/heartbeat tutarsız** — §5.1'de ölçüldü

---

# BÖLÜM 3 — CELERY

## Saf hâli ne verir

```python
app.conf.update(
    task_acks_late=True,              # iş bitince ack → worker çökerse YENİDEN TESLİM
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
)
@app.task(bind=True, max_retries=3, acks_late=True)
```

**Tam olarak üç şey:** dağıtım · retry (`countdown=0`, `max_retries=3`) · at-least-once teslim.

## Neyi VERMEZ

| eksik | sonucu |
|---|---|
| DAG / bağımlılık yok | "A bitmeden B başlamasın" kavramı yok |
| Checkpoint yok | `self.retry()` task'ı **baştan** koşturur |
| Durum takibi yok | `result_backend=None`; board olmasa hiçbir şey bilinmez |
| İptal zinciri yok | batan dalın ardılları kavramı yok |

Kodun kendi yorumu:
> *"Celery'de bağımlılık/DAG ve 'kaldığı yerden devam' kavramları yoktur; o katmanı board sağlar."*

## Biz ne ekledik

DAG kapısı · CAS-claim · lease · checkpoint · breaker · iptal zinciri · durum · olay günlüğü · fencing.

Celery'nin at-least-once teslimini **board'un claim'i at-most-once'a çeviriyor**.

**Burada hata bulduk (BUG 8'in parçası):** `task` dict'i claim'den **önce** alınıyordu → `claim_lock=None` → fencing sessizce devre dışıydı.

## İyi yanları
- **Yatay ölçekte en iyisi** — native worker havuzu, çok makineye yayılır
- **Olgun ve yaygın** — ekipte varsa öğrenme maliyeti sıfır
- **`acks_late` doğru varsayılan** — worker çökerse mesaj kaybolmaz
- **Celery Beat** ile zamanlama (backfill yok)

## Kötü yanları
- **En yavaş** — 15,3 sn (yeniden koşuda 40 sn'ye çıktı)
- **Broker işletmek gerekiyor** — ayrı servis, ayrı arıza noktası
- **Retry baştan koşturur** — checkpoint kavramı yok
- **Filesystem broker tuzağı** — `data_folder_in == data_folder_out` şart, yoksa sessizce çalışmaz
- **Görünürlük zayıf** — `result_backend=None`

---

# BÖLÜM 4 — TEMPORAL

## Saf hâli ne verir

```python
retry_policy=RetryPolicy(initial_interval=timedelta(milliseconds=200),
                         maximum_attempts=3)
```

**Durable execution** (workflow çökse replay ile devam) · **tamamlanan activity atlanır** · **RetryPolicy** (backoff'lu) · **event history** (kutudan denetim izi) · **Schedules** (cron + backfill).

## Neyi VERMEZ

| eksik | sonucu |
|---|---|
| DAG / bağımlılık yok | workflow gövdesini biz yazıyoruz, kapı board'da |
| Task kavramı yok | activity var ama "iş kalemi" modeli yok |
| Determinizm disiplini şart | workflow gövdesinde rastgelelik/IO yasak |

## Biz ne ekledik

Board'ın tamamı, artı **iki Temporal'a özgü düzeltme** (BUG 12):

**(a) Bayat claim onarımı.** `board.fail()` claim'i temizliyor, Temporal **aynı activity'yi aynı payload ile** yeniden çağırıyor → elimizdeki `claim_lock` geçersiz. Sonuç: iş başarıyla koşuyor ama `complete()` fencing'e takılıp **sonuç çöpe gidiyordu**; task hâlâ `ready`, 3 turda breaker doluyordu. **İki kez başaran düğüm `failed` işaretleniyordu.** Düzeltme: activity başında task `ready` ise yeniden claim.

**(b) Deneme sayacı board'dan.** `activity.info().attempt` her yeni activity çağrısında 1'e sıfırlanıyor → geçici hata sonsuza dek "ilk deneme" sanılıyordu:
```python
att = max(activity.info().attempt - 1, int(task.get("attempt") or 0))
```

## İyi yanları
- **Gerçek durable garanti** — sınanmış altyapı, kendi kodumuz değil
- **Hızlı** — 0,5 sn, Celery'den 30× hızlı
- **Kutudan denetim izi** — event history, replay, UI
- **Backoff var** (200 ms) — bizim 0 sn'lik retry'ımızın aksine
- **Scheduling + backfill** kutudan

## Kötü yanları
- **Operasyonel maliyet en yüksek** — cluster ya da Temporal Cloud
- **Determinizm disiplini** — workflow sınıfları modül seviyesinde olmak zorunda ("local classes unsupported"); bağlam `CTX` dict'iyle taşınıyor
- **Kendi retry'ı bizimkiyle çakışıyor** — BUG 12 tam olarak bu. **İki durable katmanı üst üste bindirmek bedava değil.**

---

# BÖLÜM 5 — AIRFLOW

## Saf hâli ne verir

`_dispatch_airflow` **yürütmez**, DAG dosyası **yazar**. Ürettiğimiz DAG'ın beyan ettikleri:

```python
default_args = {"retries": 2, "retry_delay": timedelta(seconds=30)}
with DAG(schedule="0 8 * * *", catchup=False, max_active_runs=1) as dag:
    t_a = PythonOperator(python_callable=_run_fn, op_kwargs={…})
    t_a >> t_b
```
Veri akışı XCom ile: `up = {p: ti.xcom_pull(task_ids=p) for p in parent_ids}`.

## Neyi VERMEZ
- **Dinamik graf yok** — DAG parse zamanında sabit
- **Düğüm içi checkpoint yok** — fail olan task Python fonksiyonunu baştan çağırır
- **Çalışma anında replanlama yok**

## Biz ne ekledik
Ekleme yok — **kaçış yolu** olarak bırakıldı. Amacı tek soruya cevap: *"Ajan önceden DAG yapabilir mi?"* → Evet.

Bedeli kodda yazıyor:
> *"DAG artık DONMUŞ. Yürütme sırasında ajan yeni task üretemez, replanlama yapamaz."*

## İyi yanları
- **Scheduling'de rakipsiz** — cron + **backfill/catchup** (bizde catchup yok)
- **Operatör UI'si** — olgun, zengin
- **Düğüm seviyesinde devam eder** — XCom kalıcı, retry'da üst düğümler tekrar koşmaz (bizimle **aynı**)
- **`upstream_failed`** — bizim `cancelled`'ımızın karşılığı

## Kötü yanları
- **Bizim mimarimize uymuyor** — dinamik graf temel varsayımını ihlal ediyor
- **Operasyonel maliyet yüksek** — scheduler + webserver + metadata DB
- **Düğüm içi checkpoint yok** — ajan düğümü 2 turu bitirip çökerse baştan koşar (LLM parası)
- **Zamanlama tutarsızlığı** — biz **0 sn**, ürettiğimiz DAG **30 sn**. Aynı akış Airflow'a taşınınca farklı davranır.

---

# BÖLÜM 5.1 — Lease ve heartbeat: ölçülmüş bir tutarsızlık

`LEASE_SECONDS = 30` **yorumsuz çıplak bir sabit** — hiçbir şeyden türetilmemiş.

## Bugün neden sorun çıkarmıyor — ölçüm

```
fetch_source     0,286 ms   → lease'in 104.897 katı marj
run_test_suite   0,271 ms   → lease'in 110.784 katı marj
scan_patterns    1,359 ms   → lease'in  22.081 katı marj
```

Düğüm fonksiyonları **milisaniye** sürüyor. 30 sn hiç test edilmiyor.

## "lease + heartbeat" ifadesi yanıltıcı

`heartbeat` **tek bir yerde** çağrılıyor (`orchestrator.py:336`, `execute_task` içinde, tur başına):

| düğüm türü | heartbeat | lease ne demek |
|---|---|---|
| **ajan** | ✅ her turda yenilenir | tur başına 30 sn |
| **fonksiyon** | ❌ **hiç yok** | **toplam süre için sert tavan** |

`F.call(...)` tek bloklayıcı çağrı — içinde nabız atma imkânı yok. Yani fonksiyon düğümleri için yazılı olmayan bir kural var: *"hiçbir fonksiyon 30 sn'den uzun sürmeyecek."*

## Aşılırsa ne oluyor — ölçüm

2 sn lease, 3 sn süren iş:
```
recover_stale() → 1 task 'kurtarıldı'   (worker HÂLÂ canlı!)
başka worker claim edebildi mi: EVET → ÇİFT YÜRÜTME
```

**PID kontrolü burada korumuyor:** koşul `force or expired or not pid_alive` — `expired` tek başına yetiyor.

```
1 saatlik TAZE lease · recover_stale(force=True) → 1 task geri alındı
→ force, lease'i TAMAMEN yok sayıyor
```

`_dispatch_own` iki yerde `force=True` kullanıyor; o yolda lease hiçbir koruma sağlamıyor.

## Asıl risk fonksiyonlarda değil, ajan düğümlerinde

Fonksiyonlar 1 ms — aşmaları imkânsız. Ama **tek bir LLM turu** 30 sn'yi rahat aşar (sohbette 12–15 sn ölçtük). Heartbeat tur *sonunda* atılıyor, tur *içinde* değil.

**Ulaşılamaz değil — sadece ölçmedik.** Prensipli değer şundan türetilmeli:
```
lease > en uzun düğüm süresi × güvenlik payı   VE   heartbeat aralığı × 3
```

---

# BÖLÜM 6 — TÜM TEST SONUÇLARI

## 6.1 Genel tablo

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

## 6.2 Dört motor, aynı hata

### Geçici hata (düğüm bir kez patlıyor)

| motor | yürütme denemesi | attempt | scan | tamamlanan | süre |
|---|---:|---:|---|---:|---:|
| hermes type | 2 | 1 | `done` | **5/5** | 0,0 s |
| airflow | **0** | 0 | `blocked` | **0/5** | 0,0 s |
| celery | 2 | 1 | `done` | **5/5** | 15,3 s |
| temporal | 2 | 1 | `done` | **5/5** | 0,5 s |

### Kalıcı hata (her denemede patlıyor)

| motor | deneme | scan | ardıl düğümler | tamamlanan | süre |
|---|---:|---|---|---:|---:|
| hermes type | 3 | `failed` | `cancelled` ×2 | 2/5 | 0,01 s |
| airflow | **0** | `blocked` | `blocked` ×2 | 0/5 | 0,0 s |
| celery | 3 | `failed` | `cancelled` ×2 | 2/5 | 12,4 s |
| temporal | 3 | `failed` | `cancelled` ×2 | 2/5 | 0,25 s |

**Yürüten üç motorda deneme sayısı, attempt, son durum, tamamlanan düğüm birebir aynı.**

### Çökme

| motor | çökme | kurtarma | not |
|---|---:|---:|---|
| hermes type | 1 | 1 | ✓ `recover_stale` devraldı, checkpoint korundu |
| celery | 0 | 0 | ⚠ **`crash_at` sessizce yok sayılıyor** |
| temporal | 0 | 0 | ⚠ aynı |
| airflow | 0 | 0 | yürütmüyor |

> **Dürüst uyarı:** Çökme kurtarma yalnız kendi motorumuzda **ölçüldü**. Celery/Temporal'da *yok değil, ölçülmedi*.

## 6.3 Concurrency — at-most-once kanıtı

Daha önce tek süreçte gösterilmişti (argüman). Şimdi 6 ayrı **işletim sistemi süreci**:

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

Zamanlayıcıda aynı garanti ayrı tabloda: **8 süreçten claim alan tam olarak 1**.

## 6.4 Retry anatomisi

Board olay zinciri:
```
claimed            worker-3        ← 1. deneme
retry_scheduled    breaker=1       ← hata → kuyruğa geri
claimed            worker-4        ← 2. deneme (BAŞKA worker kaptı)
completed                          ← başarı
```

**Retry aynı worker'a bağlı değil** — task `ready`ye döner, kim boşsa kapar. "Yeniden dene döngüsü" değil, **kuyruğa geri koyma**.

| | geçici hata | kalıcı hata |
|---|---:|---:|
| yürütme denemesi | 2 | 3 (BREAKER_LIMIT) |
| board `attempt` | 1 | 3 |
| son durum | `done` | `failed` |
| üst düğüm tekrar koştu mu | **hayır (1 kez)** | hayır (1 kez) |
| alt düğüm | koştu | **koşmadı** (`cancelled`) |
| denemeler arası bekleme | **~0 sn** | ~0 sn |

**Breaker sıfırlama:** arka arkaya 2 hata → `attempt=2, breaker=2`; araya başarı girince **breaker 0**, `attempt` korunur. Kural "3 hata" değil, "**arka arkaya** 3 hata".

**Kalıcı hatada retry atlanıyor:** eksik zorunlu upstream → 1 çağrı, `attempt=1`, `failed`. Geçici sayılsaydı 3 olurdu.

## 6.5 Yan etki tekrarı

*"Düğüm e-posta gönderdikten sonra patlarsa, retry ikinci kez gönderir mi?"*

| hata modu | düğümün İŞİ kaç kez yapıldı | üst düğümler |
|---|---:|---|
| iş **yapılmadan** patlıyor | 1 | 1 kez |
| iş **yapıldıktan sonra** patlıyor | **2** | 1 kez |

**Retry düğümü baştan koşturuyor.** Bugünkü fonksiyonlar saf olduğu için zararsız — **yan etkili** düğüm eklenirse idempotenslik zorunlu.

## 6.6 Çökme sonrası devam — üç seviye

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

Ölçüm (C) — log satırı:
```
↻ checkpoint'ten DEVAM: turn1'e kadar olan iş TEKRAR KOŞMAYACAK (4 mesaj geri yüklendi)
```

Motor bazında (checkpoint'e iz konup her motorla koşuldu):

| motor | önceki turun izi | süre |
|---|---|---:|
| own | ✅ korundu | 1,5 s |
| celery | ✅ korundu | 11,9 s |
| temporal | ✅ korundu | 2,0 s |

**Devam yeteneği bize ait, motora değil** — üçü de aynı `run_one_task`/`execute_task` yolundan geçiyor.

## 6.7 Zamanlama (cron) — 43/43

```
✓ cron ayrıştırıcı: 6 doğru an + hafta sonu atlama + 5 geçersiz girdi reddi
✓ next_run_at YAZMA anında türetiliyor
✓ 8 SÜREÇTEN claim alabilen: 1              ← at-most-once
✓ kira dolmadan devralınamıyor (lease=900s)
✓ kira DOLUNCA devralınıyor — ayrı recover_stale() çağrısı YOK
✓ uçtan uca: 5/5 düğüm koştu, takvim ilerledi, geçmişe yazıldı
✓ canlı sunucuda süreçler-arası: 11. saniyede yakaladı
✓ hata yolu: akış yok / düğüm battı → 'hata', claim serbest, takvim ilerledi
✓ enabled=0 tetiklenmiyor
```

Cron doğrulama örnekleri:
```
0 8 * * *     → her gün 08:00
*/15 * * * *  → 23:30, 23:45, 00:00
30 9 1 * *    → 01.09 09:30, 01.10 09:30
0 8 * * 1-5   → Cuma 14.08'den sonra PAZARTESİ 17.08   ✓ hafta sonu atlanıyor
99 8 * * *    → ValueError: '99' alanı 0-59 aralığında olmalı
```

## 6.8 Task yaşam döngüsü — 42/42

Retry · circuit-breaker · checkpoint · DAG kapısı · `spawn_task` frenleri (2/task, 12/board) · sınır durumlar (bilinmeyen fn, bozuk JSON, olmayan bağımlılık, boş başlık) · veri akışı · scheduling varlığı.

## 6.9 Hata dayanıklılığı — 54/54

8 bölüm: geçici hata · kalıcı hata + iptal zinciri · hata vs çökme ayrımı · üç backend · bozuk girdi · altyapı kaybı (DB silinmesi, lease, dev sonuç) · compaction dayanıklılığı (6 strateji × 6 uç girdi = **36 kombinasyon, hiçbiri patlamadı**) · canlı sohbet hatalı istekler.

## 6.10 Compaction matrisi — 30 koşu

Bağlam katmanı, task management'ın konusu değil ama aynı sistemde ölçüldü.

| strateji | bütçe | önce | sonra | kazanç | çift | kritik bilgi |
|---|---:|---:|---:|---:|:---:|:---:|
| none | tümü | 1473 | 1473 | — | ✓ | ✓ |
| hermes | 200/400/1000 | 1473 | 518 | %64,8 | ✓ | **✓** |
| opencode | 200/400/1000 | 1473 | 1473 | %0,0 | ✓ | **✓** |
| openclaw | 200 | 1473 | 155 | **%89,5** | ✓ | **✗** |
| openclaw | 400 | 1473 | 156 | %89,4 | ✓ | ✗ |
| openclaw | 1000 | 1473 | 157 | %89,3 | ✓ | ✗ |
| codex | 200 | 1473 | 244 | %83,4 | ✓ | ✗ |
| codex | 400 | 1473 | 251 | %83,0 | ✓ | ✗ |
| codex | 1000 | 1473 | 648 | %56,0 | ✓ | ✓ |
| claude_code | 200 | 1473 | 281 | %80,9 | ✓ | ✗ |
| claude_code | 400 | 1473 | 281 | %80,9 | ✓ | ✗ |
| claude_code | 1000 | 1473 | 794 | %46,1 | ✓ | ✓ |

*(3000 ve 30000 bütçelerde hiçbiri tetiklenmedi — iz zaten 1473 token.)*

```
ÇİFT BÜTÜNLÜĞÜ : 30/30 sağlam — hiçbiri kırılmadı
KRİTİK BİLGİ   : tetiklenen 15 koşudan 8'inde korundu
```

**Yüksek yüzde ≠ iyi.** Mesajı **koruyan** iki strateji (hermes, opencode) bilgiyi de koruyor; **birleştiren** üçü kaybediyor.

---

# BÖLÜM 7 — Bulunan 12 hata

| # | hata | sınıf | nasıl bulundu |
|---|---|---|---|
| 1 | çökme enjeksiyonu fonksiyon düğümlerinde tetiklenmiyordu | mimari değişti, çağıran güncellenmedi | hata testi |
| 2 | Celery dispatch'te `on_event` yok → SSE zaman aşımı | aynı | sohbet matrisi |
| 3 | dalga-bitti kontrolü kaldırılınca 25 sn takılmalar | regresyon | sohbet matrisi |
| 4 | **board `busy_timeout` yok** → çok-süreçli kilit | eksik yapılandırma | concurrency |
| 5 | **`spawn_task` tamamen kırıktı** | mimari değişti | yaşam döngüsü |
| 6 | Codex düşük bütçede yetim tool çifti → API 400 | sınır koşulu | compaction |
| 7 | **batan dalın ardındakiler sonsuza dek `blocked`** | eksik durum modeli | hata testi |
| 8 | **`complete()`/`fail()` fencing yok** → zombi worker yazabiliyor | eksik eşzamanlılık koruması | hata testi |
| 9 | uydurma fonksiyon adı board'a yazılıyor, 3 kez deneniyor | eksik doğrulama | hata testi |
| 10 | kalıcı/geçici hata ayrımı yok | eksik sınıflandırma | hata testi |
| 11 | **yinelenen bağımlılık kapıyı kalıcı kilitliyor** | küme/liste karışıklığı | canlı koşu |
| 12 | **Temporal'da geçici hata kalıcıya dönüşüyordu** | bayat claim | retry ölçümü |

**Dördü** *"mimari değişti, çağıran güncellenmedi"* sınıfı — fonksiyon-öncelikli geçiş ve paket refactor'ü, görünürde çalışan ama hiç test edilmeyen yolları sessizce kırmıştı.

## En ciddi bulgu — sessizce yanlış rapor

Hata enjeksiyonu **olmadan** çıktı. Planlayıcı `cross_check`'i `scan_patterns`'a bağlamamıştı; `_merge_up` eksik kenarda varsayılana düşüyordu:

> **Bir güvenlik denetimi pipeline'ı, tarayıcısı hiç veri vermemişken "Taranan desen eşleşmesi: 0" yazan tertemiz bir rapor üretti.**

| | ÖNCE | SONRA |
|---|---|---|
| rapor | `eşleşmesi: **0**` · `test: **0**` | **`4`** · **`240`** |
| uyarı | yok | `⚠ EKSİK KENAR ONARILDI` |

Üç katmanlı düzeltme: zorunlu upstream sözleşmesi (`NEEDS`) + katalogda kenar rehberi + otomatik kenar onarımı (`dogrula_dag`).

---

# BÖLÜM 8 — Koçun 6 ekseni

| eksen | durum | kanıt |
|---|---|---|
| task management | ✅ | board FSM + DAG + 42 kontrol |
| retry / kurtarma | ✅ | retry anatomisi + yan etki + 3 seviye devam |
| durum takibi | ✅ | olay günlüğü, her geçiş kayıtlı |
| concurrency | ✅ | **6 gerçek süreç, 0 çift claim, 4,7×** |
| işletme karmaşıklığı | ✅ | 0,01 s / 0,5 s / 15–40 s |
| **scheduling** | ✅ | **cron + claim/lease + 43 kontrol** |

**6/6 ölçülmüş kanıta dayanıyor.**

---

# BÖLÜM 9 — Ne zaman hangisi

| durum | seçim | gerekçe |
|---|---|---|
| geliştirme, demo, tek makine | **hermes type** | 0,01 s, kurulum yok, tam dinamik |
| kurumsal durable garanti, uzun akış | **temporal** | 0,5 s, replay, sınanmış altyapı |
| mevcut Celery altyapısı varsa | **celery** | 15–40 s bedeli kabul edilebilirse |
| sabit, tekrarlayan, insan-gözetimli veri hattı | **airflow** | cron + backfill + UI; ama graf donar |

---

# BÖLÜM 10 — Bu çalışmanın öğrettiği üç şey

**1. "Framework seçimi" sandığımız şey aslında sorumluluk paylaşımı seçimiydi.** Üç motorun hiçbiri task yönetimi vermiyor. Board'ı yazmaktan kaçış yok — soru sadece "yürütmeyi kim yapsın".

**2. İki durable katmanı üst üste bindirmek bedava değil.** BUG 12: Temporal'ın RetryPolicy'si ile bizim breaker'ımız aynı task üstünde çalışınca bayat claim doğdu ve **başarılı sonuçlar çöpe gitti**. Bir katman "sahibi ben değilim" demeli.

**3. Doğru garantiyi koymak yanlış varsayımları açığa çıkarıyor.** Fencing (BUG 8) eklemeseydik Temporal'ın bayat-claim kusurunu göremezdik — sistem "kazara" çalışıyordu.

---

# BÖLÜM 11 — Açık kalanlar

| konu | durum | etki |
|---|---|---|
| `crash_at` celery/temporal'da yok sayılıyor | **yanıltıcı API** | çökme kurtarma o ikisinde ölçülmedi |
| **Airflow hiç ölçülmedi** | — | bizim katmanda yürütmüyor; gerçek kurulum gerekiyor |
| retry'da **backoff yok** | eksik | dış servis düğümü eklenirse rate-limit derinleşir |
| **lease 30 sn gerekçesiz**, fonksiyon düğümü nabız atmıyor | §5.1 | uzun düğüm eklenirse çift yürütme riski |
| `recover_stale(force=True)` lease'i yok sayıyor | §5.1 | o yolda lease koruma sağlamıyor |
| zamanlamada **timezone yok** | eksik | sunucunun yerel saati |
| zamanlamada **catchup yok** | eksik | sunucu kapalıyken geçen vakit atlanır |
| geçersiz argüman sessizce yutuluyor | orta | `create_task` yolunda uyarı yok |
| SQLite tek yazar | mimari | çok yüksek eşzamanlılıkta darboğaz |

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
.venv/bin/python demo-brain-agent/test_matrix.py            # 28 sohbet senaryosu
```

**İlgili raporlar:** `task-management-karsilastirma-ve-test-raporu.md` · `hata-dayanikliligi-test-raporu.md` · `retry-olcum-raporu.md` · `zamanlama-cron-raporu.md` · `macro-analiz-bizim-caseler.md`
