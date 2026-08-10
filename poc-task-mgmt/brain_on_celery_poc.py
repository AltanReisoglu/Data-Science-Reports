#!/usr/bin/env python3
"""
brain_chat_V2 → CELERY'e SARILMIŞ hali (GERÇEK celery worker + filesystem broker).

"Bizim beyni dağıtık task kuyruğuna DEVRET(buy)" rotası. Bütün brain işi (A) tek bir
celery task'ı (`run_brain`) olur; broker'a `.delay()` ile atılır, ayrı bir worker
süreci çeker. brain_core.reason ilk denemede geçici hata verir → `self.retry()` ile
broker üzerinden OTOMATİK yeniden kuyruklanır.

Bu POC Celery'nin ÖNEMLİ sınırını canlı gösterir: **retry, task'ı BAŞTAN koşturur**
→ pahalı `retrieve` 2× çalışır. "Kaldığı yerden devam" (A-seviyesi) Celery'de OTOMATİK
DEĞİL; idempotency/checkpoint SENİN işin (→ bunu build-ourselves POC'unda kuruyoruz).

Çalıştır:  .venv/bin/python poc-task-mgmt/brain_on_celery_poc.py
"""
from __future__ import annotations
import os, sys, time, tempfile, subprocess
from pathlib import Path
from celery import Celery

sys.path.insert(0, str(Path(__file__).parent))
import brain_core

POC_DIR = Path(os.environ.get("BRAIN_CELERY_DIR") or tempfile.mkdtemp(prefix="brain_celery_"))
QUEUE, PROC = POC_DIR / "queue", POC_DIR / "q_proc"
for d in (QUEUE, PROC):
    d.mkdir(parents=True, exist_ok=True)
RESULT_FILE = POC_DIR / "result.txt"
STEP_LOG = POC_DIR / "steps.log"

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
    task_acks_late=True,               # iş bitince ack → worker çökerse mesaj kaybolmaz
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    result_backend=None,
)


def _log(name: str):
    with STEP_LOG.open("a") as f:
        f.write(name + "\n")


@app.task(bind=True, max_retries=3, acks_late=True)
def run_brain(self, order_id: str) -> str:
    """Tüm brain işi (A) tek task içinde — retry BAŞTAN koşar (checkpoint YOK)."""
    attempt = self.request.retries          # 0, sonra 1, ...
    _log(f"attempt{attempt}:retrieve")
    ctx = brain_core.retrieve(order_id)     # ← PAHALI adım HER denemede baştan
    try:
        _log(f"attempt{attempt}:reason")
        plan = brain_core.reason(ctx, attempt)
    except RuntimeError as e:
        _log(f"attempt{attempt}:reason-HATA")
        raise self.retry(exc=e, countdown=0)
    _log(f"attempt{attempt}:act")
    action = brain_core.act(plan)
    _log(f"attempt{attempt}:respond")
    result = brain_core.respond(action)
    RESULT_FILE.write_text(result)
    return result


def main():
    print("=" * 78)
    print("brain_chat_V2 → CELERY'e SARILI  (gerçek worker + filesystem broker)")
    print("=" * 78)
    print(f"broker klasörü: {POC_DIR}")
    RESULT_FILE.unlink(missing_ok=True)
    STEP_LOG.unlink(missing_ok=True)

    env = {**os.environ, "BRAIN_CELERY_DIR": str(POC_DIR)}
    print("\n── worker başlatılıyor (ayrı süreç) ──")
    worker = subprocess.Popen(
        [sys.executable, "-m", "celery", "-A", "brain_on_celery_poc", "worker",
         "--pool=solo", "--concurrency=1", "--loglevel=ERROR", "--without-gossip",
         "--without-mingle", "--without-heartbeat"],
        cwd=str(Path(__file__).parent), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(6)
        print("── brain işi kuyruğa atılıyor: run_brain.delay('4711') ──")
        run_brain.delay("4711")
        print("── sonuç bekleniyor…")
        deadline = time.time() + 45
        while time.time() < deadline and not RESULT_FILE.exists():
            time.sleep(0.5)

        print("\n── SONUÇ " + "─" * 60)
        print(f"  run_brain sonucu: {RESULT_FILE.read_text()!r}" if RESULT_FILE.exists()
              else "  (sonuç zamanında gelmedi)")

        steps = STEP_LOG.read_text().splitlines() if STEP_LOG.exists() else []
        print(f"\n── GERÇEKTE KOŞAN BRAIN ADIMLARI (deneme deneme) " + "─" * 18)
        for s in steps:
            print(f"     {s}")
        retr = sum(1 for s in steps if s.endswith(":retrieve"))
        print(f"\n  retrieve KAÇ KEZ koştu: {retr}")
        print("  → Celery retry task'ı BAŞTAN koşturdu: PAHALI retrieve 2× çalıştı.")
        print("    'kaldığı yerden devam' Celery'de OTOMATİK DEĞİL — checkpoint SENİN işin.")
        print("    (Aynı beyin build-ourselves'te retrieve'i 1× koşacak → kıyasla.)")
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except Exception:
            worker.kill()

    print("\n" + "=" * 78)
    print("KANIT: broker → ayrı worker → self.retry() → 2. denemede bitti; at-least-once")
    print("       + acks_late. Retry BAŞTAN koştu (retrieve 2×). A-resume = SENİN işin.")
    print("=" * 78)


if __name__ == "__main__":
    main()
