#!/usr/bin/env python3
"""
test_concurrency.py — GERÇEK çok-süreçli CAS yarışı testi (at-most-once kanıtı).

Şimdiye kadar "CAS-claim at-most-once garantisi verir" iddiası tek süreçte
gösterilmişti — yani kanıt değil, argümandı. Bu test N ayrı SÜREÇ başlatır,
hepsi AYNI board'dan aynı anda task kapmaya çalışır ve şunları ölçer:

  1) Bir task birden fazla worker tarafından claim edilebiliyor mu?  (olmamalı)
  2) Toplam claim sayısı = task sayısı mı?                            (olmalı)
  3) Paralellik gerçek mi (süre tek-süreçten kısa mı)?
  4) Çöken worker'ın task'ı başkası devralıyor mu?

    .venv/bin/python demo-brain-agent/test_concurrency.py [worker_sayisi] [task_sayisi]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from taskboard import TaskBoard  # noqa: E402


def worker(db_path: str, wid: int, calisma_sn: float, sonuc_q):
    """Bir worker süreci: board'dan task kapmaya çalış, koştur, kaydet."""
    claimed, lost, hata = [], 0, None
    try:
        board = TaskBoard(Path(db_path))
        _worker_loop(board, wid, calisma_sn, claimed)
    except Exception as e:
        hata = f"{type(e).__name__}: {e}"
    sonuc_q.put({"wid": wid, "pid": os.getpid(), "claimed": claimed,
                 "lost": lost, "hata": hata})


def _worker_loop(board, wid, calisma_sn, claimed):
    while True:
        t = board.claim_next(f"w{wid}-pid{os.getpid()}")
        if t is None:
            board.recompute_ready()
            t = board.claim_next(f"w{wid}-pid{os.getpid()}")
            if t is None:
                break
        claimed.append(t["id"])
        time.sleep(calisma_sn)                      # "iş yapılıyor"
        board.complete(t["id"], f"w{wid} tamamladı")


def crash_worker(db_path: str, wid: int, sonuc_q):
    """Task kapar, checkpoint yazar ve complete ÇAĞIRMADAN ölür (çökme)."""
    board = TaskBoard(Path(db_path))
    t = board.claim_next(f"crash{wid}-pid{os.getpid()}")
    if t:
        board.save_checkpoint(t["id"], {"kismi": f"w{wid} yarıda bıraktı"})
        sonuc_q.put({"wid": wid, "pid": os.getpid(), "claimed": [t["id"]], "crashed": True})
        # mp.Queue.put ASENKRONdur (arka planda besleyici thread yazar).
        # os._exit hemen çağrılırsa mesaj FLUSH EDİLMEDEN kaybolur ve ana süreç
        # sonsuza kadar bekler. Önce kuyruğu boşalt, sonra öl.
        sonuc_q.close()
        sonuc_q.join_thread()
        os._exit(1)                                  # ANİ ölüm — cleanup yok
    sonuc_q.put({"wid": wid, "pid": os.getpid(), "claimed": [], "crashed": False})


def main():
    N_WORKER = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    N_TASK = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    IS_SN = 0.05

    print("=" * 74)
    print("ÇOK-SÜREÇLİ CAS YARIŞI TESTİ")
    print(f"  {N_WORKER} ayrı SÜREÇ · {N_TASK} bağımsız task · task başına {IS_SN}s iş")
    print("=" * 74)

    board = TaskBoard()
    print(f"  board: {board.path}\n")
    for i in range(N_TASK):
        board.create_task(f"iş-{i:02d}", kind="function", fn="fetch_source",
                          fn_args={"path": "auth/login.py"}, priority=5)

    # ── 1) Paralel claim yarışı ──
    q = mp.Queue()
    t0 = time.time()
    procs = [mp.Process(target=worker, args=(str(board.path), i, IS_SN, q))
             for i in range(N_WORKER)]
    for p in procs:
        p.start()
    sonuclar = [q.get(timeout=180) for _ in range(N_WORKER)]
    for p in procs:
        p.join()
    sure_paralel = time.time() - t0

    tum = [tid for s in sonuclar for tid in s["claimed"]]
    say = Counter(tum)
    cift = {k: v for k, v in say.items() if v > 1}

    print("  ── 1) CLAIM DAĞILIMI (hangi worker kaç task aldı) ──")
    for s in sorted(sonuclar, key=lambda x: x["wid"]):
        print(f"     worker-{s['wid']} (pid {s['pid']}) → {len(s['claimed']):>2} task"
              + (f"   ✗ {s['hata'][:60]}" if s.get("hata") else ""))
    _hatalar = [s for s in sonuclar if s.get("hata")]
    if _hatalar:
        print(f"\n     ⚠ {len(_hatalar)} worker HATA aldı — çok-süreç güvenliği sorunlu")
    print(f"\n  toplam claim         : {len(tum)}")
    print(f"  benzersiz task       : {len(say)}")
    print(f"  ÇİFT claim edilen    : {len(cift)}  {'← ✗ AT-MOST-ONCE BOZULDU' if cift else '← ✓ at-most-once korundu'}")
    print(f"  tamamlanan (board)   : {len(board.list_tasks('done'))}/{N_TASK}")
    print(f"  süre (paralel)       : {sure_paralel:.2f} sn")

    # ── 2) Tek süreçle karşılaştırma (paralellik gerçek mi?) ──
    board2 = TaskBoard()
    for i in range(N_TASK):
        board2.create_task(f"iş-{i:02d}", kind="function", fn="fetch_source",
                           fn_args={"path": "auth/login.py"})
    q2 = mp.Queue()
    t1 = time.time()
    p = mp.Process(target=worker, args=(str(board2.path), 0, IS_SN, q2))
    p.start(); q2.get(timeout=180); p.join()
    sure_tek = time.time() - t1
    hiz = sure_tek / sure_paralel if sure_paralel else 0
    print(f"\n  ── 2) PARALELLİK ──")
    print(f"     tek süreç : {sure_tek:.2f} sn")
    print(f"     {N_WORKER} süreç  : {sure_paralel:.2f} sn   → {hiz:.1f}× hızlanma")

    # ── 3) Çökme + devralma (gerçek süreç ölümü) ──
    print(f"\n  ── 3) GERÇEK SÜREÇ ÖLÜMÜ → DEVRALMA ──")
    board3 = TaskBoard()
    tid = board3.create_task("çökecek iş", kind="function", fn="fetch_source",
                             fn_args={"path": "auth/login.py"})
    q3 = mp.Queue()
    cp = mp.Process(target=crash_worker, args=(str(board3.path), 99, q3))
    cp.start()
    r = q3.get(timeout=60)
    cp.join()
    st = board3.get(tid)
    print(f"     worker-99 (pid {r['pid']}) task'ı kaptı ve ÖLDÜ (os._exit)")
    print(f"     ölüm sonrası durum   : {st['status']}  claim={st['claim_lock']}")
    print(f"     checkpoint korundu mu: {bool(st['checkpoint'])} → {st['checkpoint']}")
    n = board3.recover_stale()          # lease dolmadı ama PID ölü → PID kontrolü
    st2 = board3.get(tid)
    print(f"     recover_stale()      : {n} task kurtarıldı → durum={st2['status']}")
    ok_devral = board3.claim_next("devralan-worker")
    print(f"     devralan worker      : {'✓ aldı' if ok_devral else '✗ alamadı'}"
          f"  (checkpoint hâlâ: {bool(board3.get(tid)['checkpoint'])})")

    # ── ÖZET ──
    print("\n" + "=" * 74)
    basarili = (not cift) and len(board.list_tasks("done")) == N_TASK and n == 1 and ok_devral
    print(f"  SONUÇ: {'✓ TÜM KONTROLLER GEÇTİ' if basarili else '✗ EN AZ BİR KONTROL BAŞARISIZ'}")
    print(f"    at-most-once      : {'✓' if not cift else '✗ ' + str(cift)}")
    print(f"    hepsi tamamlandı  : {'✓' if len(board.list_tasks('done')) == N_TASK else '✗'}")
    print(f"    paralellik        : {hiz:.1f}×")
    print(f"    çökme→devralma    : {'✓' if n == 1 and ok_devral else '✗'}")
    print("=" * 74)

    (HERE / "test_concurrency_sonuc.json").write_text(json.dumps({
        "worker": N_WORKER, "task": N_TASK,
        "toplam_claim": len(tum), "benzersiz": len(say), "cift_claim": len(cift),
        "tamamlanan": len(board.list_tasks("done")),
        "sure_paralel": round(sure_paralel, 2), "sure_tek": round(sure_tek, 2),
        "hizlanma": round(hiz, 2),
        "cokme_kurtarma": n, "devralindi": bool(ok_devral),
        "gecti": basarili,
    }, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
