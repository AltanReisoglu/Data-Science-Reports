#!/usr/bin/env python3
"""
GERÇEK Hermes agent-engine ile task-management POC.

Simülasyon DEĞİL — Hermes-Agent'ın ASIL durable çekirdeğini (hermes_cli.kanban_db,
SQLite üstünde tasks FSM + CAS-claim + lease + crash-recovery) doğrudan import edip
kullanır. Senaryo:

  1) create_task           → iş (A) SQLite 'tasks' satırı olarak doğar
  2) claim_task            → worker-A kapar (running + lease)  [CAS, at-most-once]
  3) CRASH simülasyonu     → lease süresi geçmişe çekilir, PID ölü işaretlenir
  4) release_stale_claims  → Hermes çökmeyi FARK EDER, task 'ready'ye döner (otomatik)
  5) claim_task (worker-B) → başka worker devralır (yeni deneme = yeni task_run)
  6) complete_task         → 'done'

Her adımda GERÇEK DB durumu (status, claim_lock, claim_expires, current_run_id) ve
task_runs (deneme geçmişi) yazdırılır.

Çalıştır:  .venv/bin/python poc-task-mgmt/hermes_real_poc.py
"""
from __future__ import annotations
import os, sys, time, tempfile, sqlite3
from pathlib import Path

HERMES = "/home/altan/Desktop/adapted/harnesses/hermes-agent"
sys.path.insert(0, HERMES)
import hermes_cli.kanban_db as kb   # ← GERÇEK Hermes durable kernel


def row(conn: sqlite3.Connection, tid: str) -> dict:
    r = conn.execute(
        "SELECT status, claim_lock, claim_expires, current_run_id, "
        "consecutive_failures, worker_pid FROM tasks WHERE id=?", (tid,)
    ).fetchone()
    return dict(r) if r else {}


def show(conn, tid, tag):
    r = row(conn, tid)
    exp = r.get("claim_expires")
    exp_s = "-" if not exp else ("GEÇMİŞ" if exp < time.time() else f"+{int(exp-time.time())}s")
    print(f"  [{tag:<22}] status={r.get('status'):<8} claim_lock={str(r.get('claim_lock'))[:22]:<22} "
          f"lease={exp_s:<7} run_id={r.get('current_run_id')} fails={r.get('consecutive_failures')}")


def show_runs(conn, tid):
    runs = conn.execute(
        "SELECT id, status, outcome, claim_lock, summary FROM task_runs WHERE task_id=? ORDER BY id", (tid,)
    ).fetchall()
    print(f"  task_runs (deneme geçmişi): {len(runs)} satır")
    for r in runs:
        print(f"     run#{r['id']}  status={r['status']:<9} outcome={str(r['outcome']):<10} "
              f"by={str(r['claim_lock'])[:20]:<20} summary={str(r['summary'])[:40]!r}")


def main():
    print("=" * 78)
    print("GERÇEK HERMES ile TASK-MANAGEMENT POC  (hermes_cli.kanban_db — SQLite kernel)")
    print("=" * 78)

    tmp = Path(tempfile.mkdtemp(prefix="hermes_poc_")) / "kanban.db"
    conn = kb.connect(db_path=tmp)                      # ← gerçek connect (WAL + init_db)
    print(f"DB: {tmp}\nHermes claimer-id: {kb._claimer_id()}\n")

    # 1) create_task — iş (A) doğar
    print("── 1) create_task: iş (A) SQLite'a yazılır " + "─" * 30)
    tid = kb.create_task(conn, title="Sipariş #4711 işle",
                         body="ödeme doğrula → stok düş → kargo oluştur",
                         assignee="worker-A", priority=5)
    kb.recompute_ready(conn)                            # parent yok → ready'ye terfi
    show(conn, tid, "oluşturuldu")
    print(f"  task id = {tid}")

    # 2) claim_task — worker-A CAS ile kapar
    print("\n── 2) claim_task: worker-A kapar (CAS + lease) " + "─" * 22)
    claimed = kb.claim_task(conn, tid, claimer=f"{kb._claimer_id().split(':',1)[0]}:worker-A")
    print(f"  claim_task döndü: {'Task (KAZANDI)' if claimed else 'None (kaçırdı)'}")
    show(conn, tid, "worker-A claimed")

    # ikinci worker aynı anda denerse: CAS 0 satır → None (at-most-once kanıtı)
    dup = kb.claim_task(conn, tid, claimer=f"{kb._claimer_id().split(':',1)[0]}:worker-B")
    print(f"  worker-B aynı anda claim denedi → {dup}  (at-most-once: sadece biri kapar)")

    # 3) CRASH: worker-A ölür (lease geçmişe, PID ölü)
    print("\n── 3) CRASH simülasyonu: worker-A çöktü (complete çağırmadan) " + "─" * 6)
    past = int(time.time()) - 3600
    conn.execute("UPDATE tasks SET claim_expires=?, last_heartbeat_at=?, worker_pid=? WHERE id=?",
                 (past, past, 999999, tid))            # 999999 = var olmayan PID
    conn.commit()
    show(conn, tid, "crash sonrası (ham)")
    print("  → lease GEÇMİŞ, PID ölü. Hermes bunu bir sonraki tick'te toparlamalı.")

    # 4) release_stale_claims — Hermes çökmeyi fark eder, otomatik ready'ye döndürür
    print("\n── 4) release_stale_claims: Hermes otomatik crash-recovery " + "─" * 10)
    reclaimed = kb.release_stale_claims(conn, signal_fn=lambda *_: None)
    print(f"  release_stale_claims() → {reclaimed} stale claim geri alındı")
    show(conn, tid, "recovery sonrası")
    print("  → status yeniden 'ready': iş KAYBOLMADI, başka worker devralabilir.")

    # 5) yeni worker devralır (yeni deneme)
    print("\n── 5) claim_task: worker-B devralır (yeni task_run) " + "─" * 15)
    kb.recompute_ready(conn)
    again = kb.claim_task(conn, tid, claimer=f"{kb._claimer_id().split(':',1)[0]}:worker-B")
    print(f"  claim_task döndü: {'Task (worker-B KAZANDI)' if again else 'None'}")
    show(conn, tid, "worker-B claimed")

    # 6) complete_task — done
    print("\n── 6) complete_task: iş tamamlandı " + "─" * 30)
    ok = kb.complete_task(conn, tid, result="ok",
                          summary="ödeme+stok+kargo tamam (worker-B, çökmeden sonra devraldı)")
    print(f"  complete_task() → {ok}")
    show(conn, tid, "final")

    print("\n── DENEME GEÇMİŞİ (task_runs) " + "─" * 40)
    show_runs(conn, tid)

    print("\n── OLAY GÜNLÜĞÜ (task_events) " + "─" * 40)
    evs = conn.execute("SELECT kind, created_at FROM task_events WHERE task_id=? ORDER BY id", (tid,)).fetchall()
    print("  " + " → ".join(e["kind"] for e in evs))

    print("\n" + "=" * 78)
    print("KANIT: create → claim(CAS/lease) → CRASH → release_stale_claims(otomatik) →")
    print("       worker-B devraldı → done.  İş worker çökmesine rağmen KAYBOLMADI.")
    print("       Bunların hepsi GERÇEK Hermes kodu (hermes_cli.kanban_db), simülasyon değil.")
    print("=" * 78)


if __name__ == "__main__":
    main()
