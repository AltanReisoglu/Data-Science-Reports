#!/usr/bin/env python3
"""
brain_chat_V2 için KENDİMİZİN KURDUĞU durable çekirdek (BUILD rotası) — tek dosya, stdlib.

Hiçbir dış framework yok: sadece `sqlite3`. Yine de "durable kuyruk" seviyesini verir ve
Hermes'in (otomatik crash-recovery) + Temporal'ın (kaldığı yerden devam / checkpoint) EN
İYİ İKİ ÖZELLİĞİNİ ~200 satırda birleştirir:

  • SQLite `brain_tasks` FSM         : ready → running → done/failed
  • CAS-claim (WHERE claim_lock IS NULL) : at-most-once, dağıtık kilit GEREKMEZ
  • lease + heartbeat                : worker çökünce lease dolar
  • recover_stale                    : çökeni FARK ET → otomatik 'ready' (kimse dokunmadan)
  • circuit-breaker (consecutive_failures) : sürekli hata → 'failed' (sonsuz retry yok)
  • checkpoint (JSON partial-state)  : biten adımlar atlanır → PAHALI retrieve 1× koşar

Senaryo (aynı beyin, brain_core):
  1) create                         → brain işi (A) doğar
  2) worker-A claim → retrieve → CHECKPOINT → CRASH (reason'dan önce)
  3) recover_stale                  → otomatik 'ready' (iş kaybolmadı)
  4) worker-B claim → retrieve ATLA (checkpoint) → reason(attempt0) HATA → breaker+1, ready
  5) worker-C claim → retrieve ATLA → reason(attempt1) OK → act → respond → done

Kanıt: retrieve **1×** (checkpoint korudu), reason **2×** (1 hata + 1 ok), act/respond 1×;
       worker çökmesi OTOMATİK toparlandı; hepsi bizim ~200 satırlık kodumuzla.

Çalıştır:  .venv/bin/python poc-task-mgmt/brain_build_own_poc.py
"""
from __future__ import annotations
import os, sys, time, json, sqlite3, tempfile, secrets
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import brain_core

LEASE_SECONDS = 15          # claim ne kadar geçerli (heartbeat ile uzatılır)
BREAKER_LIMIT = 2           # üst üste bu kadar hata → 'failed' (circuit-breaker açılır)


# ─────────────────────────── durable çekirdek (bizim kod) ───────────────────────────

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


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def create(conn, order_id: str) -> str:
    tid = "b_" + secrets.token_hex(4)
    conn.execute("INSERT INTO brain_tasks (id, order_id, status) VALUES (?, ?, 'ready')",
                 (tid, order_id))
    conn.commit()
    return tid


def claim(conn, tid: str, claimer: str, lease=LEASE_SECONDS) -> bool:
    """CAS-claim: yalnızca boştaki (claim_lock IS NULL) ready task'ı kapar.
    İki worker aynı anda denerse SQLite tek UPDATE'i uygular → at-most-once."""
    now = int(time.time())
    cur = conn.execute(
        "UPDATE brain_tasks SET status='running', claim_lock=?, claim_expires=?, worker_pid=? "
        "WHERE id=? AND status='ready' AND claim_lock IS NULL",
        (claimer, now + lease, os.getpid(), tid))
    conn.commit()
    return cur.rowcount == 1


def heartbeat(conn, tid: str, claimer: str, lease=LEASE_SECONDS) -> None:
    """Yaşadığını bildir: lease'i uzat (sadece bu claim sahibiyse)."""
    conn.execute("UPDATE brain_tasks SET claim_expires=? WHERE id=? AND claim_lock=?",
                 (int(time.time()) + lease, tid, claimer))
    conn.commit()


def save_checkpoint(conn, tid: str, done: dict) -> None:
    """Biten adımları durable yaz — çökme buradan SONRA olursa iş korunur."""
    conn.execute("UPDATE brain_tasks SET checkpoint=? WHERE id=?",
                 (json.dumps(done, ensure_ascii=False), tid))
    conn.commit()


def load_state(conn, tid: str):
    r = conn.execute("SELECT order_id, checkpoint, attempt FROM brain_tasks WHERE id=?",
                     (tid,)).fetchone()
    return r["order_id"], json.loads(r["checkpoint"]), r["attempt"]


def complete(conn, tid: str, result: str) -> None:
    conn.execute("UPDATE brain_tasks SET status='done', result=?, claim_lock=NULL, "
                 "claim_expires=NULL, worker_pid=NULL, consecutive_failures=0 WHERE id=?",
                 (result, tid))
    conn.commit()


def fail(conn, tid: str) -> str:
    """Geçici hata: attempt++ , breaker++ . Limit aşılmadıysa 'ready' (yeniden denenir),
    aşıldıysa 'failed' (circuit-breaker açık → sonsuz retry yok)."""
    r = conn.execute("SELECT consecutive_failures FROM brain_tasks WHERE id=?", (tid,)).fetchone()
    cf = r["consecutive_failures"] + 1
    new_status = "failed" if cf >= BREAKER_LIMIT else "ready"
    conn.execute("UPDATE brain_tasks SET status=?, attempt=attempt+1, consecutive_failures=?, "
                 "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
                 (new_status, cf, tid))
    conn.commit()
    return new_status


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, ValueError):
        return pid == os.getpid()  # kendi PID'imiz → canlı


def recover_stale(conn) -> int:
    """Çöken worker'ları FARK ET: lease dolmuş VEYA PID ölü → 'ready'ye döndür.
    checkpoint'e DOKUNMAZ → yeni worker kaldığı yerden devam eder."""
    now = int(time.time())
    rows = conn.execute("SELECT id, claim_expires, worker_pid FROM brain_tasks "
                        "WHERE status='running'").fetchall()
    n = 0
    for r in rows:
        expired = r["claim_expires"] is not None and r["claim_expires"] < now
        dead = not _pid_alive(r["worker_pid"])
        if expired or dead:
            conn.execute("UPDATE brain_tasks SET status='ready', claim_lock=NULL, "
                         "claim_expires=NULL, worker_pid=NULL WHERE id=?", (r["id"],))
            n += 1
    conn.commit()
    return n


# ─────────────────────── beynimizi bu çekirdek üstünde koşturan worker ───────────────────────

RUNS: dict[str, int] = {}


class WorkerCrash(Exception):
    pass


def brain_attempt(conn, tid, claimer, crash_after=None):
    """Bir worker'ın tek 'seans'ı: claim → checkpoint'ten devam → done/fail/crash."""
    if not claim(conn, tid, claimer):
        return ("claim-fail", None)
    order_id, done, attempt = load_state(conn, tid)

    def on_attempt(step):
        RUNS[step] = RUNS.get(step, 0) + 1   # adım GERÇEKTEN koştu (hata verse de sayılır)

    def on_step(step, out, skipped):
        save_checkpoint(conn, tid, done)     # her adımdan SONRA durable checkpoint
        heartbeat(conn, tid, claimer)
        if crash_after == step and not skipped:
            raise WorkerCrash(step)          # worker öldü: claim AÇIK kalır (→ stale)

    try:
        answer = brain_core.run_from(order_id, done, attempt,
                                     on_step=on_step, on_attempt=on_attempt)
    except WorkerCrash as c:
        return ("crash", str(c))             # complete/fail çağrılmadı → recover_stale toplayacak
    except RuntimeError as e:
        st = fail(conn, tid)
        return ("fail", f"{e} → status={st}")
    complete(conn, tid, answer)
    return ("done", answer)


# ───────────────────────────────────── demo ─────────────────────────────────────

def snap(conn, tid, tag):
    r = conn.execute("SELECT status, attempt, consecutive_failures, checkpoint, claim_lock "
                     "FROM brain_tasks WHERE id=?", (tid,)).fetchone()
    ck = list(json.loads(r["checkpoint"]).keys())
    print(f"  [{tag:<26}] status={r['status']:<8} attempt={r['attempt']} "
          f"fails={r['consecutive_failures']} checkpoint={ck} claim={r['claim_lock']}")


def main():
    print("=" * 82)
    print("brain_chat_V2 için KENDİ durable çekirdeğimiz (BUILD)  — SQLite, ~200 satır, stdlib")
    print(f"lease={LEASE_SECONDS}s · breaker_limit={BREAKER_LIMIT}")
    print("=" * 82)

    tmp = Path(tempfile.mkdtemp(prefix="brain_build_")) / "brain.db"
    conn = connect(tmp)
    print(f"DB: {tmp}\n")

    print("── 1) create: brain işi (A) doğar " + "─" * 40)
    tid = create(conn, "4711")
    snap(conn, tid, "oluşturuldu")

    print("\n── 2) worker-A claim → retrieve → CHECKPOINT → CRASH (reason'dan önce) " + "─" * 6)
    # at-most-once kanıtı: ikinci worker aynı anda claim deneyince alamaz
    st, _ = brain_attempt(conn, tid, "worker-A", crash_after="retrieve")
    print(f"     worker-A sonucu: {st}  (retrieve koştu, checkpoint yazıldı, sonra çöktü)")
    dup = claim(conn, tid, "worker-X")   # başka worker bu esnada kapabilir mi?
    print(f"     worker-X aynı anda claim denedi → {dup}  (at-most-once: running task kapılamaz)")
    snap(conn, tid, "crash sonrası (ham)")

    print("\n── 3) recover_stale: lease'i geçmişe çekip çökmeyi simüle et → otomatik toparla " + "─" * 1)
    # gerçekte lease {LEASE_SECONDS}s sonra kendiliğinden dolar; POC'ta hızlandırıyoruz
    conn.execute("UPDATE brain_tasks SET claim_expires=?, worker_pid=? WHERE id=?",
                 (int(time.time()) - 1, 999999, tid))
    conn.commit()
    n = recover_stale(conn)
    print(f"     recover_stale() → {n} çöken worker toparlandı (checkpoint'e DOKUNULMADI)")
    snap(conn, tid, "recovery sonrası")

    print("\n── 4) worker-B claim → retrieve ATLA → reason(attempt0) HATA → breaker+1 " + "─" * 6)
    st, msg = brain_attempt(conn, tid, "worker-B")
    print(f"     worker-B sonucu: {st}  ({msg})")
    snap(conn, tid, "reason hatası sonrası")

    print("\n── 5) worker-C claim → retrieve ATLA → reason(attempt1) OK → act → respond " + "─" * 3)
    st, answer = brain_attempt(conn, tid, "worker-C")
    print(f"     worker-C sonucu: {st}")
    snap(conn, tid, "final")

    print(f"\n── BRAIN ADIMLARI GERÇEKTE KAÇ KEZ KOŞTU " + "─" * 31)
    for s in brain_core.STEPS:
        note = {"retrieve": "  ← checkpoint korudu (çökme+retry'a rağmen 1×)",
                "reason":   "  ← 1 hata + 1 başarı"}.get(s, "")
        print(f"  {s:<10} : {RUNS.get(s, 0)} kez{note}")
    print(f"\n  brain sonucu: {answer!r}")

    print("\n" + "=" * 82)
    print("KANIT: worker-A çöktü → recover_stale OTOMATİK toparladı (Hermes-tarzı);")
    print("       checkpoint sayesinde retrieve TEKRAR koşmadı (Temporal-tarzı 'kaldığı yerden');")
    print("       reason geçici hatası breaker sayacıyla yönetildi. Hepsi ~200 satır bizim kodumuz,")
    print("       dış framework YOK — 'build' rotasının çalışan kanıtı.")
    print("=" * 82)


if __name__ == "__main__":
    main()
