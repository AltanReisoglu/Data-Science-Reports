# brain POC'ları — fonksiyon fonksiyon, kendi kodlarıyla (task management)

> Bu belge `poc-task-mgmt/` altındaki **brain** POC'larını **her fonksiyonu tek tek, gerçek
> koduyla** anlatır. Amaç: "bizim beyni (brain_chat_V2) task-management altyapılarına sarınca
> kod seviyesinde tam olarak ne oluyor?" sorusunu bitirmek.
>
> Sıra: önce hepsinin kullandığı ortak beyin (`brain_core.py`), sonra üç **wrap** (Temporal /
> Celery / Hermes), en sonda **kendi kurduğumuz** durable çekirdek (`brain_build_own_poc.py`).
>
> **Ortak senaryo:** brain işi = `retrieve → reason → act → respond`. `reason` (= LLM adımı) ilk
> denemede geçici hata verir. **Ölçtüğümüz metrik:** çökme/retry sonrası pahalı `retrieve` kaç kez
> koşar? (1× = tamamlanan iş korundu, 2× = baştan başladık.)

İçindekiler:
- [0. brain_core.py — ortak beyin](#0)
- [1. brain_on_temporal_poc.py — Temporal'a sar (buy)](#1)
- [2. brain_on_celery_poc.py — Celery'e sar (buy)](#2)
- [3. brain_on_hermes_poc.py — Hermes çekirdeğine sar (build/hazır motor)](#3)
- [4. brain_build_own_poc.py — kendi çekirdeğimiz (build)](#4)

---

<a name="0"></a>
## 0. `brain_core.py` — ortak beyin (hepsinin temeli)

Dört POC da AYNI beyni kullanır; böylece "aynı beyin, farklı altyapı" karşılaştırması adil olur.

### `STEPS`
```python
STEPS = ("retrieve", "reason", "act", "respond")
```
Beynin 4 adımının sırası. Sayaç/loglar bu sırayı kullanır. Bir "iş" (A) = bu dört adımın tümü.

### `retrieve(order_id)` — 1. adım: bağlam topla
```python
def retrieve(order_id: str) -> str:
    """1) Bağlam topla (RAG). PAHALI dış okuma."""
    return f"ctx({order_id})=[kullanıcı geçmişi + 3 döküman parçası]"
```
- Ajanın **girdi toplama** adımı: kullanıcı geçmişi + döküman parçaları (RAG) gibi **pahalı dış okuma**.
- POC'ta deterministik string döndürür; gerçekte DB/vektör-store/HTTP çağrısı olurdu.
- **Neden önemli:** en pahalı adım bu → "retry'da tekrar koşuyor mu?" sorusunun öznesi.

### `reason(ctx, attempt)` — 2. adım: düşün/planla (LLM)
```python
def reason(ctx: str, attempt: int) -> str:
    if attempt == 0:
        raise RuntimeError("geçici LLM hatası (model gateway timeout)")
    return f"plan[{ctx}] → tool=create_order(#4711)"
```
- Ajanın **karar** adımı: bağlama bakıp "hangi tool'u hangi argümanla çağırayım" planını üretir.
- Gerçek brain_chat_V2'de **bu adım = LLM çağrısının kendisi**. En kırılgan nokta (timeout/rate-limit).
- **`attempt` parametresi kritik:** `attempt==0` (ilk deneme) → geçici hata fırlatır; `attempt>=1` → başarı.
  Böylece her framework'ün retry mekanizmasını tetikleriz. `attempt`'i her framework kendi
  mekanizmasından besler (Temporal: RetryPolicy sayacı, Celery: `self.request.retries`, build/hermes: task `attempt` kolonu).

### `act(plan)` — 3. adım: uygula
```python
def act(plan: str) -> str:
    return f"aksiyon<{plan}> = sipariş_oluşturuldu"
```
- Planı **gerçekten uygular** (tool çalıştırma). **YAZMA yan etkisi** olduğu için idempotency önemli
  (aynı aksiyon iki kez uygulanmamalı → checkpoint burada da işe yarar).

### `respond(action)` — 4. adım: yanıtla
```python
def respond(action: str) -> str:
    return f"yanıt: {action} → kullanıcıya döndü"
```
- Ucuz kapanış adımı: sonucu kullanıcıya döndürür.

### `run_from(order_id, done, attempt, on_step, on_attempt)` — checkpoint motoru (KALP)
```python
def run_from(order_id, done: dict, attempt: int, on_step=None, on_attempt=None) -> str:
    def _do(step, fn):
        if step in done:                 # bu adım daha önce bitmiş → ATLA
            if on_step:
                on_step(step, done[step], True)
            return done[step]
        if on_attempt:
            on_attempt(step)             # adım GERÇEKTEN koşacak (hata verse de sayılır)
        out = fn()                       # adımı çalıştır (hata verirse buradan yükselir)
        done[step] = out                 # başarılıysa checkpoint'e yaz
        if on_step:
            on_step(step, out, False)
        return out

    ctx    = _do("retrieve", lambda: retrieve(order_id))
    plan   = _do("reason",   lambda: reason(ctx, attempt))
    action = _do("act",      lambda: act(plan))
    answer = _do("respond",  lambda: respond(action))
    return answer
```
Bu fonksiyon "kaldığı yerden devam"ın çekirdeği. Satır satır:
- **`done` sözlüğü = checkpoint**: `{step: çıktı}`. Hangi adımların bittiğini ve çıktılarını tutar.
- **`if step in done: … return`** → o adım daha önce bitmişse **tekrar çalıştırmaz**, kayıtlı çıktıyı döndürür. `retrieve`'in 1× koşmasının sebebi tam burası.
- **`on_attempt(step)`** → adım gerçekten koşmadan hemen önce çağrılır. Hata verecek olsa bile çağrıldığı için "kaç kez denendi" metriği doğru olur (build POC bununla `reason`'ı 2 sayar).
- **`out = fn()`** → adımı çalıştır. `reason` `attempt==0`'da burada exception fırlatır; `done`'a **yazılmaz** (yani hata veren adım checkpoint'e girmez), exception yukarı çıkar.
- **`done[step] = out`** → yalnız **başarılı** adım checkpoint'e yazılır.
- **`on_step(step, out, skipped)`** → başarılı adımdan sonra çağrılır (checkpoint'i diske yaz, heartbeat at, çökme simüle et gibi işler wrapper'da yapılır).

> **Özet:** `run_from`, verilen `done` checkpoint'ini alır, **eksik adımları** sırayla koşar, her
> başarılı adımı `done`'a ekler. Aynı `done`'la tekrar çağrılırsa biten adımlar atlanır → **idempotent devam**.

---

<a name="1"></a>
## 1. `brain_on_temporal_poc.py` — Temporal'a sar (buy: durable motora bin)

**Fikir:** brain'in 4 adımı birer **activity**, ajan döngüsü bir **workflow**. Worker çökünce
Temporal **replay** edip biten activity'leri atlar → `retrieve` 1×.

### Modül kurulumu
```python
logging.getLogger("temporalio").setLevel(logging.CRITICAL)  # beklenen retry uyarılarını sustur
RUNS: dict[str, int] = {}   # her adım gerçekte kaç kez koştu (retry/replay kanıtı)
```
- `RUNS`: activity'ler koştukça artan sayaç. Temporal replay'i kanıtlamanın yolu (ör. `reason` 2×).

### `a_retrieve(order_id)` — activity
```python
@activity.defn
async def a_retrieve(order_id: str) -> str:
    RUNS["retrieve"] = RUNS.get("retrieve", 0) + 1
    return brain_core.retrieve(order_id)
```
- `@activity.defn` → bunu bir Temporal **activity** yapar (yan etkili iş burada koşar).
- İçeride yalnız `brain_core.retrieve`'i çağırır; başında sayacı artırır.

### `a_reason(ctx)` — activity (retry'ın konusu)
```python
@activity.defn
async def a_reason(ctx: str) -> str:
    RUNS["reason"] = RUNS.get("reason", 0) + 1
    return brain_core.reason(ctx, attempt=RUNS["reason"] - 1)
```
- **`attempt=RUNS["reason"] - 1`** hilesi: ilk çağrıda `RUNS`=1 → `attempt`=0 → `brain_core.reason` **hata fırlatır**.
- Temporal RetryPolicy bu activity'yi otomatik yeniden çağırır → `RUNS`=2 → `attempt`=1 → başarı.
- Yani `attempt`'i Temporal'ın retry sayacından türetiyoruz.

### `a_act`, `a_respond` — activity
```python
@activity.defn
async def a_act(plan: str) -> str:
    RUNS["act"] = RUNS.get("act", 0) + 1
    return brain_core.act(plan)

@activity.defn
async def a_respond(action: str) -> str:
    RUNS["respond"] = RUNS.get("respond", 0) + 1
    return brain_core.respond(action)
```
- Aynı kalıp: sayaç + ilgili `brain_core` adımı.

### `BrainWorkflow.run(order_id)` — workflow (ajan döngüsü)
```python
@workflow.defn
class BrainWorkflow:
    @workflow.run
    async def run(self, order_id: str) -> str:
        ctx = await workflow.execute_activity(
            a_retrieve, order_id, start_to_close_timeout=timedelta(seconds=5))
        plan = await workflow.execute_activity(
            a_reason, ctx, start_to_close_timeout=timedelta(seconds=5),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(milliseconds=100), maximum_attempts=3))
        action = await workflow.execute_activity(
            a_act, plan, start_to_close_timeout=timedelta(seconds=5))
        return await workflow.execute_activity(
            a_respond, action, start_to_close_timeout=timedelta(seconds=5))
```
- **Workflow gövdesi = ajan döngüsünün kendisi**, ama saf/deterministik: sadece activity'leri sırayla `await` eder.
- **`retry_policy` yalnız `a_reason`'da** → geçici hata veren tek adım o; Temporal onu otomatik 3 kez dener.
- **Determinizm disiplini:** LLM/rastgele/IO **activity içinde**; workflow gövdesinde YOK. Böylece
  replay güvenli (Temporal workflow'u kaldığı yerden yeniden oynatırken sonuç değişmez).

### `main()` — çalıştırma akışı
```python
async with await WorkflowEnvironment.start_time_skipping() as env:   # gerçek dev server
    async with Worker(env.client, task_queue="brain",
                      workflows=[BrainWorkflow],
                      activities=[a_retrieve, a_reason, a_act, a_respond]):
        handle = await env.client.start_workflow(
            BrainWorkflow.run, "4711", id="brain-4711", task_queue="brain")
        result = await handle.result()
    # ... RUNS sayaçlarını yazdır ...
    hist = await handle.fetch_history()   # durable event history
```
- `WorkflowEnvironment.start_time_skipping()` → **gerçek** ephemeral Temporal server başlatır.
- `Worker(...)` → workflow ve activity'leri kaydeder; `task_queue="brain"` üzerinden iş çeker.
- `start_workflow(...) → handle.result()` → işi başlatır ve sonucu bekler.
- `fetch_history()` → **durable event-history**'yi çeker (SCHEDULED/STARTED/COMPLETED × adım sayısı).

**Ölçülen çıktı:** `retrieve 1×, reason 2×, act 1×, respond 1×`, ~29 durable event.
**Ders:** exactly-once + kaldığı yerden devam **yerleşik**; bedeli cluster + determinizm kuralları.

---

<a name="2"></a>
## 2. `brain_on_celery_poc.py` — Celery'e sar (buy: kuyruğa devret)

**Fikir:** bütün brain işi (A) **tek bir celery task'ı**; broker'a atılır, ayrı worker çeker.
Retry tüm task'ı **baştan** koşar → `retrieve` 2×.

### `app` — Celery + broker yapılandırması
```python
app = Celery(
    "brain_on_celery_poc",
    broker="filesystem://",
    broker_transport_options={
        "data_folder_in": str(QUEUE),
        "data_folder_out": str(QUEUE),   # filesystem transport: in==out (aynı klasör) ŞART
        "processed_folder": str(PROC),
        "store_processed": True,
    },
)
app.conf.update(
    task_acks_late=True,               # iş BİTİNCE ack → worker çökerse mesaj kaybolmaz
    task_reject_on_worker_lost=True,   # worker ölürse mesaj yeniden teslim
    worker_prefetch_multiplier=1,
    result_backend=None,
)
```
- **Broker = sunucusuz filesystem transport** (Redis/RabbitMQ kurmadan gerçek kuyruk).
- **`data_folder_in == data_folder_out`**: kombu filesystem transport'ta producer OUT'a yazar,
  consumer IN'den okur; buluşmaları için **aynı klasör** olmalı (yoksa worker mesajı hiç görmez).
- **`task_acks_late=True`**: mesaj, task **bitince** onaylanır → worker ortada çökerse mesaj
  kuyrukta kalır, yeniden teslim edilir (at-least-once dayanıklılık).

### `_log(name)` — adım kaydı
```python
def _log(name: str):
    with STEP_LOG.open("a") as f:
        f.write(name + "\n")
```
- Hangi adımın hangi denemede koştuğunu bir dosyaya yazar (ör. `attempt0:retrieve`). Worker ayrı
  süreçte olduğu için sonucu **dosya üzerinden** görürüz.

### `run_brain(self, order_id)` — brain işi = tek task
```python
@app.task(bind=True, max_retries=3, acks_late=True)
def run_brain(self, order_id: str) -> str:
    attempt = self.request.retries          # 0, sonra 1, ...
    _log(f"attempt{attempt}:retrieve")
    ctx = brain_core.retrieve(order_id)     # ← PAHALI adım HER denemede baştan
    try:
        _log(f"attempt{attempt}:reason")
        plan = brain_core.reason(ctx, attempt)
    except RuntimeError as e:
        _log(f"attempt{attempt}:reason-HATA")
        raise self.retry(exc=e, countdown=0)   # tüm task'ı yeniden kuyrukla
    _log(f"attempt{attempt}:act")
    action = brain_core.act(plan)
    _log(f"attempt{attempt}:respond")
    result = brain_core.respond(action)
    RESULT_FILE.write_text(result)
    return result
```
- **`bind=True`** → `self` verir (retry mekanizması için gerekli). **`max_retries=3`** üst sınır.
- **`attempt = self.request.retries`** → Celery'nin retry sayacı; `brain_core.reason`'a bunu besliyoruz.
- **Kritik sınır:** `retrieve` fonksiyonun EN BAŞINDA, `try`'dan önce. `reason` hata verip
  `self.retry()` çağrılınca **tüm task fonksiyonu baştan** koşar → `retrieve` **ikinci kez** çalışır.
- **`self.retry(...)`** → hatalı task'ı broker'a geri koyar; worker onu yeni bir çalıştırma olarak çeker.
- Checkpoint YOK: Celery task'ı atomik bir birim sayar; "adımın ortasından devam" kavramı yoktur.

### `main()` — çalıştırma akışı
```python
worker = subprocess.Popen([sys.executable, "-m", "celery", "-A", "brain_on_celery_poc",
                           "worker", "--pool=solo", ...])   # AYRI süreçte gerçek worker
time.sleep(6)                                               # worker broker'a bağlansın
run_brain.delay("4711")                                     # GERÇEK enqueue (broker'a mesaj)
# ... RESULT_FILE oluşana kadar bekle, STEP_LOG'u oku, retrieve sayısını say ...
```
- **`subprocess.Popen(... celery worker ...)`** → gerçek, ayrı bir worker süreci başlatır.
- **`.delay("4711")`** → task'ı broker'a atar (fire-and-forget); worker kuyruktan çekip koşar.
- Sonuç dosya (`RESULT_FILE`) + adım log'u (`STEP_LOG`) üzerinden okunur.

**Ölçülen çıktı:** `attempt0:retrieve → reason-HATA → attempt1:retrieve → reason → act → respond`,
**`retrieve 2×`**.
**Ders:** Celery güçlü dağıtım + at-least-once verir; ama "kaldığı yerden devam"ı (A-seviyesi)
**sen** kurarsın. Bir sonraki iki rota tam bunu ekliyor.

---

<a name="3"></a>
## 3. `brain_on_hermes_poc.py` — Hermes çekirdeğine sar (build / hazır motor)

**Fikir:** brain işi = **kanban kartı**; Hermes'in SQLite FSM + CAS-claim + lease + crash-recovery'sini
kullan, brain'in partial-state'ini **handoff (task_comment)** ile taşı → `retrieve` 1×. Gerçek
`hermes_cli.kanban_db` import edilir (simülasyon değil).

### `_count(step, _out, skipped)` — adım sayacı
```python
def _count(step, _out, skipped):
    if not skipped:
        RUNS[step] = RUNS.get(step, 0) + 1
```
- `run_from`'a `on_step` olarak verilir; atlanmayan (gerçekten koşan) adımı sayar.

### `_save_handoff(conn, tid, author, done)` — partial-state'i devret
```python
def _save_handoff(conn, tid, author, done):
    kb.add_comment(conn, tid, author=author,
                   body=f"{HANDOFF_TAG} {json.dumps(done, ensure_ascii=False)}")
```
- Brain'in o ana kadarki checkpoint'ini (`done`) bir **task_comment** olarak yazar. `HANDOFF_TAG`
  (`[brain-checkpoint]`) ile işaretlenir. Bu, Hermes'in "handoff" fikrinin brain'e uyarlanmış hali:
  çöken worker'ın ilerlemesi karta yazılır, devralan worker okur.

### `_load_handoff(conn, tid)` — devralınan state'i oku
```python
def _load_handoff(conn, tid) -> dict:
    done = {}
    for c in kb.list_comments(conn, tid):
        if c.body.startswith(HANDOFF_TAG):
            done = json.loads(c.body[len(HANDOFF_TAG):].strip())
    return done
```
- Kartın yorumlarını tarar, en son `[brain-checkpoint]` yorumundan `done` sözlüğünü geri kurar.
  worker-B bununla "retrieve zaten bitmiş" bilgisini alır.

### `main()` — akış, aşama aşama
Gerçek `kanban_db` fonksiyonları kullanılır:

**1) Kart oluştur:**
```python
tid = kb.create_task(conn, title="brain: kullanıcı isteğini yanıtla (#4711)",
                     body=json.dumps({"order_id": "4711"}), assignee="worker-A")
kb.recompute_ready(conn)     # parent yok → ready
```
**2) worker-A claim → retrieve → handoff:**
```python
kb.claim_task(conn, tid, claimer=f"{cid}:worker-A")
ctx = brain_core.retrieve("4711"); RUNS["retrieve"] += 1   # retrieve BURADA koşar (1. ve tek kez)
done = {"retrieve": ctx}
_save_handoff(conn, tid, "worker-A", done)                 # ilerlemeyi karta yaz
```
**3) CRASH (reason'dan önce):**
```python
conn.execute("UPDATE tasks SET claim_expires=?, last_heartbeat_at=?, worker_pid=? WHERE id=?",
             (past, past, 999999, tid))    # lease geçmiş + PID ölü
```
- worker-A `reason`'a gelmeden ölür; `retrieve` çıktısı handoff'ta güvende.

**4) Hermes otomatik recovery:**
```python
kb.release_stale_claims(conn, signal_fn=lambda *_: None)   # çökeni FARK ET → ready
```
- Bu **gerçek Hermes fonksiyonu**: lease geçmiş/PID ölü claim'i tespit eder, task'ı `ready`'ye döndürür.

**5) worker-B devralır → kaldığı yerden:**
```python
kb.claim_task(conn, tid, claimer=f"{cid}:worker-B")
resume = _load_handoff(conn, tid)                          # {'retrieve': ...}
answer = brain_core.run_from("4711", resume, attempt=1, on_step=_count)
```
- `resume` içinde `retrieve` olduğu için `run_from` onu **atlar**; `reason(attempt=1)` başarı → `act` → `respond`.

**6) Tamamla:**
```python
kb.complete_task(conn, tid, result=answer,
                 summary="worker-B devraldı; retrieve tekrar edilmedi")
```

**Ölçülen çıktı:** `run#1 reclaimed, run#2 completed`; events `created→claimed→commented→reclaimed→claimed→completed`; **`retrieve 1×`**.
**Ders:** hazır durable motor (Hermes) + handoff = düşük operasyonla otomatik recovery + kaldığı yerden devam.

---

<a name="4"></a>
## 4. `brain_build_own_poc.py` — kendi çekirdeğimiz (build, sıfırdan)

**Fikir:** hiçbir dış framework yok (yalnız `sqlite3`). Hermes'in otomatik recovery'si + Temporal'ın
checkpoint'ten devamı ~200 satırda. **brain_chat_V2 zaten state tutuyorsa en gerçekçi rota.**

### Sabitler + şema
```python
LEASE_SECONDS = 15      # claim ne kadar geçerli (heartbeat ile uzatılır)
BREAKER_LIMIT = 2       # üst üste bu kadar hata → 'failed'

SCHEMA = """
CREATE TABLE IF NOT EXISTS brain_tasks (
    id TEXT PRIMARY KEY, order_id TEXT NOT NULL, status TEXT NOT NULL,
    checkpoint TEXT NOT NULL DEFAULT '{}',   -- JSON: {step: output} biten adımlar
    attempt INTEGER NOT NULL DEFAULT 0,
    claim_lock TEXT, claim_expires INTEGER, worker_pid INTEGER,
    consecutive_failures INTEGER NOT NULL DEFAULT 0, result TEXT );
"""
```
- Tek tablo tüm durable durumu tutar: FSM (`status`), **checkpoint** (kaldığı yer), `attempt`,
  claim/lease alanları, breaker sayacı. "Durable kuyruk"un tamamı bu satırda.

### `connect(path)` — bağlan + şema
```python
def connect(path):
    conn = sqlite3.connect(path); conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL"); conn.executescript(SCHEMA); conn.commit()
    return conn
```
- `row_factory=Row` → sütunlara isimle erişim. `WAL` → eşzamanlı okuma/yazmada dayanıklılık.

### `create(conn, order_id)` — iş (A) doğar
```python
def create(conn, order_id):
    tid = "b_" + secrets.token_hex(4)
    conn.execute("INSERT INTO brain_tasks (id, order_id, status) VALUES (?,?, 'ready')",
                 (tid, order_id)); conn.commit()
    return tid
```
- Yeni bir brain işini `ready` durumunda tabloya yazar. checkpoint boş (`{}`), attempt 0.

### `claim(conn, tid, claimer, lease)` — CAS-claim (at-most-once)
```python
def claim(conn, tid, claimer, lease=LEASE_SECONDS):
    now = int(time.time())
    cur = conn.execute(
        "UPDATE brain_tasks SET status='running', claim_lock=?, claim_expires=?, worker_pid=? "
        "WHERE id=? AND status='ready' AND claim_lock IS NULL",
        (claimer, now + lease, os.getpid(), tid))
    conn.commit()
    return cur.rowcount == 1
```
- **En kritik fonksiyon.** `WHERE ... claim_lock IS NULL` → yalnız **boştaki** task'ı kapar.
- İki worker aynı anda çağırırsa SQLite UPDATE'i **tek satıra** uygular; biri `rowcount==1` (kazandı),
  diğeri `rowcount==0` (kaçırdı). **Dağıtık kilit gerekmez** — atomik UPDATE yeter (Compare-And-Swap).
- `claim_expires = now + lease` → lease (kira) başlar; `worker_pid` → çökme tespiti için.

### `heartbeat(conn, tid, claimer, lease)` — yaşadığını bildir
```python
def heartbeat(conn, tid, claimer, lease=LEASE_SECONDS):
    conn.execute("UPDATE brain_tasks SET claim_expires=? WHERE id=? AND claim_lock=?",
                 (int(time.time()) + lease, tid, claimer)); conn.commit()
```
- Worker çalışırken periyodik çağırır → lease'i ileri atar. **`AND claim_lock=?`** → yalnız
  claim'in **sahibi** uzatabilir. Worker ölünce heartbeat kesilir → lease dolar → çökme anlaşılır.

### `save_checkpoint(conn, tid, done)` — kaldığı yeri diske yaz
```python
def save_checkpoint(conn, tid, done):
    conn.execute("UPDATE brain_tasks SET checkpoint=? WHERE id=?",
                 (json.dumps(done, ensure_ascii=False), tid)); conn.commit()
```
- Biten adımları (`done`) durable yazar. **Çökme bundan SONRA olursa iş korunur** — devralan worker bu checkpoint'i okur.

### `load_state(conn, tid)` — durumu geri yükle
```python
def load_state(conn, tid):
    r = conn.execute("SELECT order_id, checkpoint, attempt FROM brain_tasks WHERE id=?",
                     (tid,)).fetchone()
    return r["order_id"], json.loads(r["checkpoint"]), r["attempt"]
```
- Devralan worker: order_id + checkpoint (`done`) + attempt'i okur → `run_from`'a verir.

### `complete(conn, tid, result)` — bitti
```python
def complete(conn, tid, result):
    conn.execute("UPDATE brain_tasks SET status='done', result=?, claim_lock=NULL, "
                 "claim_expires=NULL, worker_pid=NULL, consecutive_failures=0 WHERE id=?",
                 (result, tid)); conn.commit()
```
- `status='done'`, sonucu yaz, claim/lease temizle, breaker sayacını sıfırla.

### `fail(conn, tid)` — geçici hata + circuit-breaker
```python
def fail(conn, tid):
    r = conn.execute("SELECT consecutive_failures FROM brain_tasks WHERE id=?", (tid,)).fetchone()
    cf = r["consecutive_failures"] + 1
    status = "failed" if cf >= BREAKER_LIMIT else "ready"
    conn.execute("UPDATE brain_tasks SET status=?, attempt=attempt+1, consecutive_failures=?, "
                 "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
                 (status, cf, tid)); conn.commit()
    return status
```
- Adım geçici hata verince çağrılır. **`attempt+1`** → bir sonraki denemede `brain_core.reason`
  `attempt>=1` görüp başarılı olur.
- **`consecutive_failures`** artar; **`BREAKER_LIMIT`**'e ulaşınca `status='failed'` → **sonsuz retry engellenir** (circuit-breaker). Aksi halde `ready` → yeniden denenir.

### `_pid_alive(pid)` — süreç canlı mı
```python
def _pid_alive(pid):
    if not pid: return False
    try: os.kill(int(pid), 0); return True
    except (ProcessLookupError, PermissionError, ValueError): return pid == os.getpid()
```
- `os.kill(pid, 0)` sinyal göndermez, sadece sürecin **var olup olmadığını** sınar. Yoksa
  `ProcessLookupError` → ölü. Çökme tespitinin ikinci ayağı (lease'in yanında).

### `recover_stale(conn)` — otomatik crash-recovery
```python
def recover_stale(conn):
    now = int(time.time()); n = 0
    for r in conn.execute("SELECT id, claim_expires, worker_pid FROM brain_tasks "
                          "WHERE status='running'").fetchall():
        expired = r["claim_expires"] is not None and r["claim_expires"] < now
        if expired or not _pid_alive(r["worker_pid"]):
            conn.execute("UPDATE brain_tasks SET status='ready', claim_lock=NULL, "
                         "claim_expires=NULL, worker_pid=NULL WHERE id=?", (r["id"],)); n += 1
    conn.commit()
    return n
```
- **Hermes'in `release_stale_claims`'inin bizim versiyonumuz.** `running` task'ları tarar; lease
  **dolmuş** VEYA PID **ölü** olanı `ready`'ye döndürür → başka worker devralabilir.
- **`checkpoint`'e DOKUNMAZ** → devralan worker kaldığı yerden başlar. Otomatik recovery + resume birlikte.
- Gerçekte bunu bir dispatcher periyodik (ör. 60s) çağırır; POC'ta bir kez çağırıyoruz.

### `WorkerCrash` + `brain_attempt(conn, tid, claimer, crash_after)` — worker seansı
```python
class WorkerCrash(Exception): pass

def brain_attempt(conn, tid, claimer, crash_after=None):
    if not claim(conn, tid, claimer):        # kapamadıysa (başkası aldı) → çık
        return ("claim-fail", None)
    order_id, done, attempt = load_state(conn, tid)   # kaldığı yeri yükle

    def on_attempt(step):
        RUNS[step] = RUNS.get(step, 0) + 1   # adım gerçekten koştu (hata verse de sayılır)

    def on_step(step, out, skipped):
        save_checkpoint(conn, tid, done)     # her başarılı adımdan SONRA durable checkpoint
        heartbeat(conn, tid, claimer)        # yaşadığını bildir
        if crash_after == step and not skipped:
            raise WorkerCrash(step)          # çökme SİMÜLASYONU: claim açık kalır → recover_stale toplar

    try:
        answer = brain_core.run_from(order_id, done, attempt,
                                     on_step=on_step, on_attempt=on_attempt)
    except WorkerCrash:
        return ("crash", None)               # complete/fail çağrılmadı → recover_stale devreye girecek
    except RuntimeError as e:
        return ("fail", fail(conn, tid))     # reason geçici hatası → attempt++, breaker++
    complete(conn, tid, answer)              # hepsi bitti → done
    return ("done", answer)
```
Bir worker'ın **tek seansı**. Kod akışı:
- **`claim(...)`** başarısızsa (task başkasında) hemen çıkar → **at-most-once**.
- **`load_state`** → checkpoint + attempt yükle; `run_from`'a ver.
- **`on_attempt`** → doğru adım-sayacı (hata veren `reason` da sayılır → `reason` 2×).
- **`on_step`** → her başarılı adımdan sonra **checkpoint diske** + **heartbeat**. `crash_after`
  eşleşirse `WorkerCrash` fırlatır (worker'ın o adımdan sonra ölmesini simüle eder; claim açık kalır).
- **`except WorkerCrash`** → complete/fail çağrılmaz; claim stale kalır → `recover_stale` toparlar.
- **`except RuntimeError`** → `reason`'ın geçici hatası; `fail()` (attempt++, breaker++).
- Hepsi geçerse **`complete()`** → `done`.

### `main()` — akış, aşama aşama (canlı çıktıyla)
```
1) create                                         → ready, checkpoint=[]
2) brain_attempt("worker-A", crash_after="retrieve")
     → retrieve koşar, checkpoint=['retrieve'] yazılır, sonra CRASH
     claim("worker-X") → False                    (at-most-once: running kapılamaz)
3) (lease'i geçmişe çek) recover_stale() → 1      → status=ready (checkpoint korundu)
4) brain_attempt("worker-B")
     → retrieve ATLA (checkpoint) → reason(attempt0) HATA → fail(): attempt=1, fails=1, ready
5) brain_attempt("worker-C")
     → retrieve ATLA → reason(attempt1) OK → act → respond → complete → done

SONUÇ:  retrieve 1×  ·  reason 2×  ·  act 1×  ·  respond 1×
```
- **worker-A çöktü → `recover_stale` OTOMATİK toparladı** (Hermes-tarzı durable kuyruk).
- **checkpoint sayesinde `retrieve` çökme+retry'a rağmen 1×** (Temporal-tarzı kaldığı yerden devam).
- **`reason` geçici hatası breaker sayacıyla** yönetildi.
- Hepsi **~200 satır bizim kodumuz**, dış framework YOK.

---

## Kapanış — dört rota, tek tablo

| POC | Anahtar fonksiyon(lar) | Dayanıklılık nasıl geliyor | `retrieve` |
|---|---|---|:---:|
| Temporal wrap | `execute_activity` + `RetryPolicy` + workflow | replay biten activity'yi atlar | **1×** |
| Celery wrap | `run_brain` + `self.retry()` | at-least-once; resume yok (baştan) | **2×** ⚠️ |
| Hermes wrap | `claim_task`/`release_stale_claims` + handoff | otomatik reclaim + handoff taşır | **1×** |
| Build-ourselves | `claim`(CAS) + `recover_stale` + `save_checkpoint`/`load_state` | otomatik recovery + checkpoint | **1×** |

> **Tek cümle:** Pahalı adımın (retrieve) retry'da 1× koşması = "kaldığı yerden devam"ın altyapıdan
> gelmesi; 2× koşması = onu senin kurman. `claim`(CAS) + `recover_stale` + `checkpoint` üçlüsü, bu
> özelliği kendi ellerimizle ~200 satırda verir.

**Çalıştırma:**
```bash
.venv/bin/python poc-task-mgmt/brain_on_temporal_poc.py
.venv/bin/python poc-task-mgmt/brain_on_celery_poc.py
.venv/bin/python poc-task-mgmt/brain_on_hermes_poc.py
.venv/bin/python poc-task-mgmt/brain_build_own_poc.py
# ya da tarayıcıdan: .venv/bin/python poc-task-mgmt/web_server.py → localhost:8000 → 2. KISIM
```
İlgili belgeler: `report/brain-chat-v2-task-management-entegrasyon.md` (genel bakış + build kodu),
`report/USTA-REHBER-tool-trace-ve-task-management.md` (KISIM II teori).
