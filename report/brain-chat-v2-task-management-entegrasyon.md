# brain_chat_V2 × Task-Management — beynimizi altyapılara sarmak (+ kendi çekirdeğimizi kurmak)

> **Amaç:** Aynı beyni (brain_chat_V2) dört farklı task-management altyapısına **sarıp/simüle edip**
> ne olacağını ölçmek; sonra bu altyapıyı **kendimiz sıfırdan kurduğumuz** durumu koddan göstermek.
> Bütün rotalar **aynı beyni** (`poc-task-mgmt/brain_core.py`) kullanır → karşılaştırma adil.
> Çalışan kanıt: `poc-task-mgmt/brain_on_*_poc.py` + `brain_build_own_poc.py` (hepsi gerçekten koşuyor).

Bu belge, `poc-task-mgmt/`'daki **ilk POC'ların (hermes/temporal/celery) ALTINA** eklenen yeni
POC'ları anlatır. İlk POC'lar "framework kendi başına nasıl davranır"ı gösteriyordu; burada
**bizim beyni o framework'lerin içine koyunca** ne olduğunu ölçüyoruz.

---

## 0. Ortak beyin — `brain_core.py`

Gerçek brain_chat_V2 kapalı kaynak. Onun **çekirdek ajan-döngüsünü** temsil eden, 4 adımlı bir
"iş"i (A) modelledik. Her adım (B) idempotent ve partial-state döndürür — böylece bir
checkpoint'ten devam edilebilir.

| Adım | Ne yapar | Gerçekte | Özelliği |
|---|---|---|---|
| **retrieve** | bağlam/RAG toplar | dış kaynak okuma | **PAHALI** → retry'da tekrar-etmemesi kritik |
| **reason** | muhakeme/plan üretir | **LLM çağrısının kendisi** | en kırılgan adım (timeout/rate-limit) |
| **act** | tool çalıştırır | yazma aksiyonu | YAZMA yan etkisi → idempotency önemli |
| **respond** | nihai yanıtı üretir | kullanıcıya dönüş | ucuz |

**Senaryo:** `reason` **ilk denemede geçici hata** verir (LLM gateway timeout) → her framework'ün
retry/recovery yolunu tetikler. **Ölçtüğümüz asıl soru:** çökme/retry sonrası **pahalı `retrieve`
kaç kez koşar?** (tamamlanan işi koruyabiliyor muyuz, yoksa baştan mı başlıyoruz?)

```python
# brain_core.py (özet)
STEPS = ("retrieve", "reason", "act", "respond")

def retrieve(order_id):          return f"ctx({order_id})=[kullanıcı geçmişi + 3 döküman]"
def reason(ctx, attempt):
    if attempt == 0: raise RuntimeError("geçici LLM hatası (model gateway timeout)")
    return f"plan[{ctx}] → tool=create_order(#4711)"
def act(plan):                   return f"aksiyon<{plan}> = sipariş_oluşturuldu"
def respond(action):             return f"yanıt: {action} → kullanıcıya döndü"

def run_from(order_id, done, attempt, on_step=None):
    """`done` checkpoint'inden DEVAM eder: biten adımları atlar, kalanı koşar.
       Bir adım hata verse bile `done` korunur (checkpoint) → kaldığı yerden devam."""
    ...
```

---

## 1. Dört rota, tek tablo (CANLI ölçüm)

Dört POC'u da gerçekten koşturduk. **Aynı beyin**, dört altyapı:

| Rota | Nasıl | Taksonomi | `retrieve` | `reason` | Çökme/retry sonrası |
|---|---|---|:---:|:---:|---|
| **Temporal wrap** | brain adımları = activity, döngü = workflow | **buy** (durable motora bin) | **1×** | 2× | replay biten activity'yi **atlar** |
| **Hermes wrap** | brain işi = kanban kartı, partial-state = handoff | **build** (hazır motor) | **1×** | 1×\* | `release_stale_claims` otomatik + handoff |
| **Celery wrap** | brain işi = tek celery task | **buy** (kuyruğa devret) | **2×** ⚠️ | — | `self.retry()` **baştan** koşar |
| **Build-ourselves** | kendi SQLite durable çekirdeğimiz | **build** (sıfırdan) | **1×** | 2× | checkpoint + otomatik recovery + breaker |

\* Hermes wrap'te `reason` 1× çünkü worker-A `reason`'a **gelmeden** çöktü; worker-B onu bir kez
(attempt=1) başarıyla koştu. Diğerlerinde `reason` 2× = 1 geçici hata + 1 başarı.

> **Tek ders:** Pahalı `retrieve` üç dayanıklı rotada **1×**, naif Celery'de **2×**. Aradaki fark
> "kaldığı yerden devam"ın altyapıdan **yerleşik gelmesi** (Temporal/Hermes/build) ile onu **senin
> kurman** (Celery) arasındaki farktır.

---

## 2. Rota 1 — Temporal'a sarmak (buy: durable motora bin)  · Shannon rotası

**Fikir:** brain'in 4 adımı birer **activity**, ajan döngüsü bir **workflow** olur. Her adım
event-history'ye yazılır; worker çökünce Temporal **replay** edip biten activity'leri atlar.

```python
# brain_on_temporal_poc.py (özet)
@activity.defn
async def a_retrieve(order_id): return brain_core.retrieve(order_id)
@activity.defn
async def a_reason(ctx):        return brain_core.reason(ctx, attempt=RUNS["reason"]-1)
# ...

@workflow.defn
class BrainWorkflow:
    @workflow.run
    async def run(self, order_id):
        ctx    = await workflow.execute_activity(a_retrieve, order_id, ...)
        plan   = await workflow.execute_activity(a_reason, ctx, ...,
                    retry_policy=RetryPolicy(maximum_attempts=3))   # reason otomatik retry
        action = await workflow.execute_activity(a_act, plan, ...)
        return   await workflow.execute_activity(a_respond, action, ...)
```

**Canlı çıktı:** `retrieve 1×, reason 2×, act 1×, respond 1×` · 29 durable event.
**Determinizm disiplini:** LLM/rastgele/IO YALNIZ activity içinde; workflow gövdesi saf kalır.
**Bedel:** cluster/Cloud + determinizm kuralları. **Kazanç:** exactly-once, en güçlü dayanıklılık.

---

## 3. Rota 2 — Celery'e sarmak (buy: kuyruğa devret)

**Fikir:** bütün brain işi (A) tek bir celery task'ı; broker'a `.delay()` ile atılır, ayrı worker
çeker. `reason` hata verince `self.retry()` broker üzerinden yeniden kuyruklar.

```python
# brain_on_celery_poc.py (özet)
@app.task(bind=True, max_retries=3, acks_late=True)
def run_brain(self, order_id):
    attempt = self.request.retries
    ctx = brain_core.retrieve(order_id)          # ← HER denemede baştan (checkpoint YOK)
    try:
        plan = brain_core.reason(ctx, attempt)
    except RuntimeError as e:
        raise self.retry(exc=e, countdown=0)     # tüm task yeniden kuyruklanır
    return brain_core.respond(brain_core.act(plan))
```

**Canlı çıktı:** `attempt0:retrieve → reason-HATA → attempt1:retrieve → reason → act → respond`,
**`retrieve 2×`**.
**Ders:** Celery **at-least-once** teslim + `acks_late` verir (worker çökerse mesaj yeniden
teslim). Ama "kaldığı yerden devam" (A-seviyesi, `retrieve`'i atlamak) **otomatik değil** —
idempotency/checkpoint **senin işin**. Bir sonraki rota tam da bunu ekliyor.

---

## 4. Rota 3 — Hermes çekirdeğine sarmak (build: hazır durable motor)

**Fikir:** brain işi bir **kanban kartı** (A) olur; Hermes'in SQLite FSM + CAS-claim + lease +
crash-recovery'sini kullanırız ve brain'in partial-state'ini **handoff (task_comment)** ile
taşırız. Worker-A çökse bile worker-B, `retrieve`'i tekrar yapmadan devam eder.

```python
# brain_on_hermes_poc.py (özet) — GERÇEK hermes_cli.kanban_db
tid = kb.create_task(conn, title="brain: yanıtla (#4711)", body=json.dumps({"order_id":"4711"}))
kb.claim_task(conn, tid, claimer=f"{cid}:worker-A")
ctx = brain_core.retrieve("4711")
kb.add_comment(conn, tid, "worker-A", f"[brain-checkpoint] {json.dumps({'retrieve': ctx})}")  # handoff
# ... CRASH (lease geçmiş, PID ölü) ...
kb.release_stale_claims(conn)                    # Hermes OTOMATİK toparlar → 'ready'
kb.claim_task(conn, tid, claimer=f"{cid}:worker-B")
resume = _load_handoff(conn, tid)                # {'retrieve': ...}
answer = brain_core.run_from("4711", resume, attempt=1)   # retrieve ATLANIR
kb.complete_task(conn, tid, result=answer, summary="worker-B devraldı; retrieve tekrar edilmedi")
```

**Canlı çıktı:** `run#1 reclaimed, run#2 completed` · events `created→claimed→commented→reclaimed→claimed→completed` · **`retrieve 1×`**.
**Ders:** Kendi durable çekirdeğini **kurmak** ama hazır bir motorla (Hermes) — düşük operasyon,
handoff ile bağlam taşınır, worker çökmesi otomatik toparlanır.

---

## 5. Rota 4 — KENDİMİZ kurmak (build: sıfırdan durable çekirdek) — TAM KOD

**Fikir:** hiçbir dış framework yok, sadece `sqlite3`. Yine de **durable kuyruk** seviyesini verir
ve Hermes'in (otomatik recovery) + Temporal'ın (checkpoint'ten devam) en iyi iki özelliğini
~200 satırda birleştirir. **brain_chat_V2 zaten oturum/state tutuyorsa en gerçekçi rota budur.**

Beş yapı taşı:

| Yapı taşı | Ne verir |
|---|---|
| SQLite `brain_tasks` FSM | ready → running → done/failed, kalıcı durum |
| **CAS-claim** (`WHERE claim_lock IS NULL`) | **at-most-once**, dağıtık kilit gerekmez |
| **lease + heartbeat** | worker çökünce lease dolar |
| **recover_stale** | çökeni fark et → otomatik `ready` (kimse dokunmadan) |
| **circuit-breaker** (`consecutive_failures`) | sürekli hata → `failed` (sonsuz retry yok) |
| **checkpoint** (JSON partial-state) | biten adımlar atlanır → pahalı `retrieve` 1× |

### 5.1 Durable çekirdek (kendi kodumuz)

```python
import os, time, json, sqlite3, secrets

LEASE_SECONDS = 15      # claim ne kadar geçerli (heartbeat ile uzatılır)
BREAKER_LIMIT = 2       # üst üste bu kadar hata → 'failed'

SCHEMA = """
CREATE TABLE IF NOT EXISTS brain_tasks (
    id                   TEXT PRIMARY KEY,
    order_id             TEXT NOT NULL,
    status               TEXT NOT NULL,             -- ready|running|done|failed
    checkpoint           TEXT NOT NULL DEFAULT '{}',-- JSON: {step: output} biten adımlar
    attempt              INTEGER NOT NULL DEFAULT 0,
    claim_lock           TEXT,                      -- claimer id (NULL = boşta)
    claim_expires        INTEGER,                   -- lease bitişi (epoch)
    worker_pid           INTEGER,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    result               TEXT
);
"""

def connect(path):
    conn = sqlite3.connect(path); conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL"); conn.executescript(SCHEMA); conn.commit()
    return conn

def create(conn, order_id):
    tid = "b_" + secrets.token_hex(4)
    conn.execute("INSERT INTO brain_tasks (id, order_id, status) VALUES (?,?, 'ready')",
                 (tid, order_id)); conn.commit()
    return tid

def claim(conn, tid, claimer, lease=LEASE_SECONDS):
    """CAS-claim: yalnız boştaki (claim_lock IS NULL) ready task'ı kapar.
       İki worker aynı anda denerse SQLite tek UPDATE uygular → at-most-once."""
    now = int(time.time())
    cur = conn.execute(
        "UPDATE brain_tasks SET status='running', claim_lock=?, claim_expires=?, worker_pid=? "
        "WHERE id=? AND status='ready' AND claim_lock IS NULL",
        (claimer, now + lease, os.getpid(), tid))
    conn.commit()
    return cur.rowcount == 1                        # True = kazandı, False = kaçırdı

def heartbeat(conn, tid, claimer, lease=LEASE_SECONDS):
    conn.execute("UPDATE brain_tasks SET claim_expires=? WHERE id=? AND claim_lock=?",
                 (int(time.time()) + lease, tid, claimer)); conn.commit()

def save_checkpoint(conn, tid, done):
    """Biten adımları durable yaz — çökme buradan SONRA olursa iş korunur."""
    conn.execute("UPDATE brain_tasks SET checkpoint=? WHERE id=?",
                 (json.dumps(done, ensure_ascii=False), tid)); conn.commit()

def load_state(conn, tid):
    r = conn.execute("SELECT order_id, checkpoint, attempt FROM brain_tasks WHERE id=?",
                     (tid,)).fetchone()
    return r["order_id"], json.loads(r["checkpoint"]), r["attempt"]

def complete(conn, tid, result):
    conn.execute("UPDATE brain_tasks SET status='done', result=?, claim_lock=NULL, "
                 "claim_expires=NULL, worker_pid=NULL, consecutive_failures=0 WHERE id=?",
                 (result, tid)); conn.commit()

def fail(conn, tid):
    """Geçici hata: attempt++ , breaker++ . Limit aşılmadıysa 'ready', aşıldıysa 'failed'."""
    r = conn.execute("SELECT consecutive_failures FROM brain_tasks WHERE id=?", (tid,)).fetchone()
    cf = r["consecutive_failures"] + 1
    status = "failed" if cf >= BREAKER_LIMIT else "ready"
    conn.execute("UPDATE brain_tasks SET status=?, attempt=attempt+1, consecutive_failures=?, "
                 "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
                 (status, cf, tid)); conn.commit()
    return status

def _pid_alive(pid):
    if not pid: return False
    try: os.kill(int(pid), 0); return True
    except (ProcessLookupError, PermissionError, ValueError): return pid == os.getpid()

def recover_stale(conn):
    """Çöken worker'ları FARK ET: lease dolmuş VEYA PID ölü → 'ready'.
       checkpoint'e DOKUNMAZ → yeni worker kaldığı yerden devam eder."""
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

### 5.2 Beynimizi bu çekirdek üstünde koşturan worker

```python
class WorkerCrash(Exception): pass

def brain_attempt(conn, tid, claimer, crash_after=None):
    """Bir worker'ın tek seansı: claim → checkpoint'ten devam → done/fail/crash."""
    if not claim(conn, tid, claimer):
        return ("claim-fail", None)
    order_id, done, attempt = load_state(conn, tid)

    def on_step(step, out, skipped):
        save_checkpoint(conn, tid, done)     # her adımdan SONRA durable checkpoint
        heartbeat(conn, tid, claimer)
        if crash_after == step and not skipped:
            raise WorkerCrash(step)          # worker öldü: claim AÇIK kalır → recover_stale toplar

    try:
        answer = brain_core.run_from(order_id, done, attempt, on_step=on_step)
    except WorkerCrash:
        return ("crash", None)               # complete/fail çağrılmadı
    except RuntimeError as e:
        return ("fail", fail(conn, tid))     # reason geçici hatası → attempt++, breaker++
    complete(conn, tid, answer)
    return ("done", answer)
```

### 5.3 Ne kanıtlıyor (canlı çıktı)

```
1) create                                  → ready, checkpoint=[]
2) worker-A claim → retrieve → CHECKPOINT → CRASH
   worker-X aynı anda claim → False        (at-most-once)
   status=running, checkpoint=['retrieve']
3) recover_stale() → 1                      → status=ready (checkpoint korundu)
4) worker-B claim → retrieve ATLA → reason(attempt0) HATA → fails=1, attempt=1, ready
5) worker-C claim → retrieve ATLA → reason(attempt1) OK → act → respond → done

BRAIN ADIMLARI:  retrieve 1×  ·  reason 2×  ·  act 1×  ·  respond 1×
```

- **worker-A çöktü → `recover_stale` OTOMATİK toparladı** (Hermes-tarzı durable kuyruk).
- **checkpoint sayesinde `retrieve` çökme+retry'a rağmen 1×** (Temporal-tarzı kaldığı yerden devam).
- **`reason` geçici hatası breaker sayacıyla** yönetildi (sonsuz retry yok).
- Hepsi **~200 satır bizim kodumuz**, dış framework YOK.

> İki dünyanın en iyisi: durable kuyruğun otomatik recovery'si + event-sourcing'in "kaldığı
> yerden devam"ı, tek bir SQLite tablosu ve birkaç fonksiyonla.

---

## 6. brain_chat_V2 için öneri

- **En hızlı yol (buy):** Celery ekle (kuyruğu ona devret) — ama A-seviyesi resume için brain'in
  oturum/state'ini checkpoint olarak sen yaz (Rota 2'nin eksiğini Rota 4'ün checkpoint fikriyle kapat).
- **Tam kontrol (build), önerilen:** **Rota 4** — Hermes-tarzı hafif durable çekirdek (SQLite/Postgres,
  CAS-claim, lease, breaker, checkpoint). brain zaten state tutuyorsa en doğal uyum; ~1-2K satırda
  otomatik recovery + kaldığı yerden devam birlikte gelir.
- **Temporal'a (Rota 1) ancak** çok-makineli, uzun-bekleyen (insan/olay), exactly-once + deterministik
  replay **şart** olunca geç — operasyon ve determinizm bedeli yüksek.

---

## 7. Çalıştırma

```bash
# aynı beyin, dört altyapı — retrieve kaç kez koşuyor karşılaştır:
.venv/bin/python poc-task-mgmt/brain_on_temporal_poc.py   # buy   → retrieve 1× (replay)
.venv/bin/python poc-task-mgmt/brain_on_celery_poc.py     # buy   → retrieve 2× (baştan)
.venv/bin/python poc-task-mgmt/brain_on_hermes_poc.py     # build → retrieve 1× (handoff)
.venv/bin/python poc-task-mgmt/brain_build_own_poc.py     # build → retrieve 1× (checkpoint + recovery)
```

> Bu POC'lar `poc-task-mgmt/`'daki ilk (framework-tek-başına) POC'lara **dokunmaz**; onların
> **altına** eklenmiştir. İlgili teori: `report/USTA-REHBER-tool-trace-ve-task-management.md` (KISIM II)
> ve `report/task-yonetimi-altyapi-karari.md`.
