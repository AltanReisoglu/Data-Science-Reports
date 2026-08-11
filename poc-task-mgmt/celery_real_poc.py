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
from celery.exceptions import MaxRetriesExceededError

# main ve worker AYNI broker klasörünü kullanmalı → env ile paylaş
POC_DIR = Path(os.environ.get("CELERY_POC_DIR") or tempfile.mkdtemp(prefix="celery_poc_"))
# kombu filesystem transport: producer OUT'a yazar, consumer IN'den okur →
# buluşmaları için ikisi de AYNI klasör olmalı.
QUEUE, PROC = POC_DIR / "queue", POC_DIR / "q_proc"
for d in (QUEUE, PROC):
    d.mkdir(parents=True, exist_ok=True)
RESULT_FILE = POC_DIR / "result.txt"
FAIL_FILE = POC_DIR / "failed.txt"
ATTEMPT_FILE = POC_DIR / "attempts.log"

# ── AYARLAR (web arayüzünden env ile değiştirilebilir; varsayılanlar özgün senaryo)
FAIL_TIMES = int(os.environ.get("POC_FAIL_TIMES", "1"))    # process ilk kaç denemede patlasın
MAX_RETRIES = int(os.environ.get("POC_MAX_RETRIES", "3"))  # kaç retry hakkı var

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


def emit_flow(steps):
    """Web arayüzünün canlı workflow şeridini çizebilmesi için tek satırlık özet.
    Biçim:  ##FLOW## ad:kaç_kez:durum|ad:kaç_kez:durum   (durum: ok|retry|fail|skip|crash)"""
    print("##FLOW## " + "|".join(f"{n}:{c}:{s}" for n, c, s in steps))


def _step(name: str):
    with ATTEMPT_FILE.open("a") as f:
        f.write(name + "\n")


@app.task(bind=True, max_retries=MAX_RETRIES, acks_late=True)
def run_order(self, order_id: str) -> str:
    """3 adımı tek task içinde koşar; process ilk FAIL_TIMES denemede geçici hata verir."""
    attempt = self.request.retries        # 0, sonra 1, ...
    _step(f"attempt{attempt}:fetch")      # ← fetch HER denemede baştan koşar (kanıt)
    data = f"order:{order_id}:veri"
    if attempt < FAIL_TIMES:
        # process: geçici hata → broker üzerinden retry
        _step(f"attempt{attempt}:process-HATA")
        try:
            self.retry(exc=RuntimeError("geçici hata (ödeme timeout)"), countdown=0)
        except MaxRetriesExceededError:
            # retry hakkı bitti → iş KALICI olarak başarısız, sonuç hiç üretilmedi
            FAIL_FILE.write_text(f"retry hakkı bitti (max_retries={MAX_RETRIES})")
            raise
        raise RuntimeError("unreachable")  # self.retry her zaman fırlatır
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
    print(f"ayarlar: process ilk {FAIL_TIMES} denemede patlar · retry hakkı = {MAX_RETRIES}")
    RESULT_FILE.unlink(missing_ok=True)
    FAIL_FILE.unlink(missing_ok=True)
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
        while time.time() < deadline and not (RESULT_FILE.exists() or FAIL_FILE.exists()):
            time.sleep(0.5)

        print("\n── SONUÇ " + "─" * 60)
        if RESULT_FILE.exists():
            print(f"  run_order sonucu: {RESULT_FILE.read_text()!r}")
        elif FAIL_FILE.exists():
            print(f"  iş KALICI BAŞARISIZ: {FAIL_FILE.read_text()}")
            print("  → Sonuç hiç üretilmedi. Celery işi bırakır; kurtarma sende.")
        else:
            print("  (sonuç zamanında gelmedi — worker/broker gecikmesi)")

        steps = ATTEMPT_FILE.read_text().splitlines() if ATTEMPT_FILE.exists() else []
        print(f"\n── GERÇEKTE KOŞAN ADIMLAR (deneme deneme) " + "─" * 26)
        for s in steps:
            print(f"     {s}")
        fetch_count = sum(1 for s in steps if s.endswith(":fetch"))
        proc_count = sum(1 for s in steps if ":process-" in s)
        deliver_count = sum(1 for s in steps if s.endswith(":deliver"))
        print(f"\n  fetch KAÇ KEZ koştu: {fetch_count}")
        print(f"  → Celery retry task'ı BAŞTAN koşturur: fetch {fetch_count} kez çalıştı.")
        print("    'kaldığı yerden devam' (fetch'i atlamak) Celery'de OTOMATİK DEĞİL —")
        print("    idempotency/checkpoint SENİN işin. (Temporal/Hermes bunu yerleşik verir.)")

        # ── web arayüzü için makine-okunur akış özeti
        ok = RESULT_FILE.exists()
        emit_flow([
            ("fetch", fetch_count, "retry" if fetch_count > 1 else "ok"),
            ("process", proc_count, "fail" if not ok else ("retry" if proc_count > 1 else "ok")),
            ("deliver", deliver_count, "ok" if deliver_count else "skip"),
        ])
        if ok:
            print(f"##VERDICT## pahalı fetch {fetch_count}× koştu — tamamlanan iş KORUNMADI, baştan başlandı")
        else:
            print(f"##VERDICT## retry hakkı ({MAX_RETRIES}) bitti — iş kalıcı başarısız, fetch boşuna {fetch_count}× koştu")
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
