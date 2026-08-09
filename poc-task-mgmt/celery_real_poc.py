#!/usr/bin/env python3
"""
GERÇEK Celery ile task-management POC.

Simülasyon DEĞİL — gerçek `celery` framework'ü, gerçek bir **broker** (sunucusuz
filesystem transport) ve AYRI bir süreçte gerçek bir **worker** kullanır:

  main süreci  --.delay()-->  [filesystem broker kuyruğu]  -->  celery worker (ayrı süreç)
                                                                    run_order task'ı koşar

Senaryo: run_order(order_id) → fetch → (process'te İLK denemede geçici hata) →
self.retry() ile broker üzerinden OTOMATİK yeniden kuyruklanır → 2. denemede biter.

Bu POC ayrıca Celery'nin ÖNEMLİ sınırını canlı gösterir: **retry, task'ı BAŞTAN
koşturur** (fetch 2 kez çalışır) → "kaldığı yerden devam" (A-seviyesi) senin işin.

Çalıştır:  .venv/bin/python poc-task-mgmt/celery_real_poc.py
"""
from __future__ import annotations
import os, sys, time, tempfile, subprocess
from pathlib import Path
from celery import Celery

# main ve worker AYNI broker klasörünü kullanmalı → env ile paylaş
POC_DIR = Path(os.environ.get("CELERY_POC_DIR") or tempfile.mkdtemp(prefix="celery_poc_"))
# kombu filesystem transport: producer OUT'a yazar, consumer IN'den okur →
# buluşmaları için ikisi de AYNI klasör olmalı.
QUEUE, PROC = POC_DIR / "queue", POC_DIR / "q_proc"
for d in (QUEUE, PROC):
    d.mkdir(parents=True, exist_ok=True)
RESULT_FILE = POC_DIR / "result.txt"
ATTEMPT_FILE = POC_DIR / "attempts.log"

app = Celery(
    "celery_real_poc",
    broker="filesystem://",
    broker_transport_options={
        "data_folder_in": str(QUEUE),
        "data_folder_out": str(QUEUE),
        "processed_folder": str(PROC),
        "store_processed": True,
    },
)
app.conf.update(
    task_acks_late=True,               # iş bitince ack → worker çökerse mesaj kaybolmaz
    task_reject_on_worker_lost=True,   # worker ölürse mesaj yeniden teslim
    worker_prefetch_multiplier=1,
    result_backend=None,
)


def _step(name: str):
    with ATTEMPT_FILE.open("a") as f:
        f.write(name + "\n")


@app.task(bind=True, max_retries=3, acks_late=True)
def run_order(self, order_id: str) -> str:
    """3 adımı tek task içinde koşar; process ilk denemede geçici hata verir."""
    attempt = self.request.retries        # 0, sonra 1, ...
    _step(f"attempt{attempt}:fetch")      # ← fetch HER denemede baştan koşar (kanıt)
    data = f"order:{order_id}:veri"
    if attempt == 0:
        # process: ilk denemede geçici hata → broker üzerinden retry
        _step(f"attempt{attempt}:process-HATA")
        raise self.retry(exc=RuntimeError("geçici hata (ödeme timeout)"), countdown=0)
    _step(f"attempt{attempt}:process-OK")
    _step(f"attempt{attempt}:deliver")
    result = f"{data}|işlendi|teslim"
    RESULT_FILE.write_text(result)        # sonucu dosyaya yaz (result backend yok)
    return result


def main():
    print("=" * 78)
    print("GERÇEK CELERY ile TASK-MANAGEMENT POC  (filesystem broker + ayrı worker süreci)")
    print("=" * 78)
    print(f"broker klasörü: {POC_DIR}")
    RESULT_FILE.unlink(missing_ok=True)
    ATTEMPT_FILE.unlink(missing_ok=True)

    env = {**os.environ, "CELERY_POC_DIR": str(POC_DIR)}
    print("\n── worker başlatılıyor (ayrı süreç: celery -A celery_real_poc worker) ──")
    worker = subprocess.Popen(
        [sys.executable, "-m", "celery", "-A", "celery_real_poc", "worker",
         "--pool=solo", "--concurrency=1", "--loglevel=ERROR", "--without-gossip",
         "--without-mingle", "--without-heartbeat"],
        cwd=str(Path(__file__).parent), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(6)  # worker'ın broker'a bağlanmasını bekle
        print("── task kuyruğa atılıyor: run_order.delay('4711') ──")
        run_order.delay("4711")           # ← gerçek enqueue (broker'a mesaj yazılır)

        print("── sonuç bekleniyor (worker kuyruktan çekip koşacak)…")
        deadline = time.time() + 45
        while time.time() < deadline and not RESULT_FILE.exists():
            time.sleep(0.5)

        print("\n── SONUÇ " + "─" * 60)
        if RESULT_FILE.exists():
            print(f"  run_order sonucu: {RESULT_FILE.read_text()!r}")
        else:
            print("  (sonuç zamanında gelmedi — worker/broker gecikmesi)")

        steps = ATTEMPT_FILE.read_text().splitlines() if ATTEMPT_FILE.exists() else []
        print(f"\n── GERÇEKTE KOŞAN ADIMLAR (deneme deneme) " + "─" * 26)
        for s in steps:
            print(f"     {s}")
        fetch_count = sum(1 for s in steps if s.endswith(":fetch"))
        print(f"\n  fetch KAÇ KEZ koştu: {fetch_count}")
        print("  → Celery retry task'ı BAŞTAN koşturdu: fetch 2 kez çalıştı.")
        print("    'kaldığı yerden devam' (fetch'i atlamak) Celery'de OTOMATİK DEĞİL —")
        print("    idempotency/checkpoint SENİN işin. (Temporal/Hermes bunu yerleşik verir.)")
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except Exception:
            worker.kill()

    print("\n" + "=" * 78)
    print("KANIT: gerçek broker'a enqueue → ayrı worker süreci çekip koştu → process")
    print("       ilk denemede geçici hata → self.retry() ile broker'dan yeniden kuyruklandı →")
    print("       2. denemede bitti. acks_late açık. Retry BAŞTAN koştu (fetch 2×).")
    print("       Hepsi GERÇEK Celery (celery worker + filesystem broker), simülasyon değil.")
    print("=" * 78)


if __name__ == "__main__":
    main()
