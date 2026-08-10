#!/usr/bin/env python3
"""
brain_chat_V2 → HERMES kanban çekirdeğine SARILMIŞ hali (GERÇEK hermes_cli.kanban_db).

"Bizim beyni kendi durable çekirdeğimizle KUR(build)" rotasının HAZIR-motor sürümü:
Hermes'in SQLite tasks FSM + CAS-claim + lease + crash-recovery'sini kullanır ve
brain'in partial-state'ini **handoff (task_comment)** ile taşır. Böylece worker-A
çökse bile worker-B, retrieve'i TEKRAR yapmadan kaldığı yerden devam eder.

Senaryo (aynı beyin, brain_core):
  1) create_task           → brain işi (A) SQLite'a doğar
  2) worker-A claim        → retrieve'i koşar, sonucu HANDOFF olarak yazar (add_comment)
  3) CRASH                 → worker-A reason'a gelmeden çöker (lease geçmiş, PID ölü)
  4) release_stale_claims  → Hermes otomatik toparlar, task 'ready'
  5) worker-B claim        → handoff'u okur (retrieve BİTMİŞ) → reason+act+respond
  6) complete_task         → done

Kanıt: PAHALI `retrieve` toplam **1×** koşar (handoff taşıdı) — Celery'de 2× idi.

Çalıştır:  .venv/bin/python poc-task-mgmt/brain_on_hermes_poc.py
"""
from __future__ import annotations
import os, sys, time, json, tempfile
from pathlib import Path

HERMES = "/home/altan/Desktop/adapted/harnesses/hermes-agent"
sys.path.insert(0, HERMES)
sys.path.insert(0, str(Path(__file__).parent))
import hermes_cli.kanban_db as kb   # GERÇEK Hermes durable kernel
import brain_core

RUNS: dict[str, int] = {}           # brain adımı gerçekte kaç kez koştu
HANDOFF_TAG = "[brain-checkpoint]"


def _count(step, _out, skipped):
    if not skipped:
        RUNS[step] = RUNS.get(step, 0) + 1


def _save_handoff(conn, tid, author, done):
    kb.add_comment(conn, tid, author=author,
                   body=f"{HANDOFF_TAG} {json.dumps(done, ensure_ascii=False)}")


def _load_handoff(conn, tid) -> dict:
    done: dict = {}
    for c in kb.list_comments(conn, tid):
        if c.body.startswith(HANDOFF_TAG):
            try:
                done = json.loads(c.body[len(HANDOFF_TAG):].strip())
            except Exception:
                pass
    return done


def show(conn, tid, tag):
    r = conn.execute("SELECT status, claim_lock, current_run_id FROM tasks WHERE id=?",
                     (tid,)).fetchone()
    print(f"  [{tag:<24}] status={r['status']:<8} run_id={r['current_run_id']} "
          f"claim={str(r['claim_lock'])[:24]}")


def main():
    print("=" * 78)
    print("brain_chat_V2 → HERMES kanban çekirdeğine SARILI  (gerçek kanban_db + handoff)")
    print("=" * 78)

    tmp = Path(tempfile.mkdtemp(prefix="brain_hermes_")) / "kanban.db"
    conn = kb.connect(db_path=tmp)
    cid = kb._claimer_id().split(":", 1)[0]
    print(f"DB: {tmp}\n")

    # 1) brain işi (A) doğar
    print("── 1) create_task: brain işi (A) SQLite'a yazılır " + "─" * 22)
    tid = kb.create_task(conn, title="brain: kullanıcı isteğini yanıtla (#4711)",
                         body=json.dumps({"order_id": "4711"}), assignee="worker-A")
    kb.recompute_ready(conn)
    show(conn, tid, "oluşturuldu")

    # 2) worker-A claim → retrieve koşar → handoff yaz
    print("\n── 2) worker-A claim → retrieve koşar + handoff yazar " + "─" * 15)
    kb.claim_task(conn, tid, claimer=f"{cid}:worker-A")
    show(conn, tid, "worker-A claimed")
    done: dict = {}
    ctx = brain_core.retrieve("4711"); RUNS["retrieve"] = RUNS.get("retrieve", 0) + 1
    done["retrieve"] = ctx
    _save_handoff(conn, tid, "worker-A", done)
    print(f"     retrieve() koştu → handoff yazıldı: {done}")

    # 3) CRASH: reason'a gelmeden worker-A ölür
    print("\n── 3) CRASH: worker-A reason'dan ÖNCE çöktü " + "─" * 22)
    past = int(time.time()) - 3600
    conn.execute("UPDATE tasks SET claim_expires=?, last_heartbeat_at=?, worker_pid=? WHERE id=?",
                 (past, past, 999999, tid))
    conn.commit()
    show(conn, tid, "crash sonrası (ham)")

    # 4) Hermes otomatik crash-recovery
    print("\n── 4) release_stale_claims: Hermes otomatik toparlar " + "─" * 16)
    n = kb.release_stale_claims(conn, signal_fn=lambda *_: None)
    print(f"     release_stale_claims() → {n} stale claim geri alındı")
    kb.recompute_ready(conn)
    show(conn, tid, "recovery sonrası")

    # 5) worker-B claim → handoff oku → kaldığı yerden devam
    print("\n── 5) worker-B claim → handoff'u okur → kaldığı yerden " + "─" * 12)
    kb.claim_task(conn, tid, claimer=f"{cid}:worker-B")
    show(conn, tid, "worker-B claimed")
    resume = _load_handoff(conn, tid)
    print(f"     handoff okundu: retrieve zaten BİTMİŞ → {list(resume)}")
    answer = brain_core.run_from("4711", resume, attempt=1, on_step=_count)
    print(f"     run_from(attempt=1): reason+act+respond koştu, retrieve ATLANDI")

    # 6) complete
    print("\n── 6) complete_task " + "─" * 45)
    kb.complete_task(conn, tid, result=answer,
                     summary="brain tamam (worker-B, handoff'tan devraldı; retrieve tekrar edilmedi)")
    show(conn, tid, "final")

    print("\n── DENEME GEÇMİŞİ + OLAYLAR " + "─" * 40)
    runs = conn.execute("SELECT id, status, outcome FROM task_runs WHERE task_id=? ORDER BY id",
                        (tid,)).fetchall()
    for r in runs:
        print(f"     run#{r['id']}  status={r['status']:<9} outcome={r['outcome']}")
    evs = conn.execute("SELECT kind FROM task_events WHERE task_id=? ORDER BY id", (tid,)).fetchall()
    print("     events: " + " → ".join(e["kind"] for e in evs))

    print(f"\n── BRAIN ADIMLARI GERÇEKTE KAÇ KEZ KOŞTU " + "─" * 27)
    for s in brain_core.STEPS:
        note = "  ← handoff taşıdı, TEKRAR KOŞMADI" if s == "retrieve" else ""
        print(f"  {s:<10} : {RUNS.get(s, 0)} kez{note}")
    print(f"\n  brain sonucu: {answer!r}")

    print("\n" + "=" * 78)
    print("KANIT: worker-A çöktü ama Hermes otomatik toparladı; handoff (task_comment)")
    print("       brain'in partial-state'ini taşıdı → worker-B PAHALI retrieve'i TEKRAR")
    print("       yapmadan bitirdi (retrieve 1×). Gerçek kanban_db + brain_core, simülasyon değil.")
    print("=" * 78)


if __name__ == "__main__":
    main()
