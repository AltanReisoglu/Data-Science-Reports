#!/usr/bin/env python3
"""
saf_celery.py — Celery'nin SAF hâli: canvas ilkelleriyle, board YOK.

Celery'de "DAG" diye bir şey yoktur; onun yerine **canvas** vardır:
  chain(a, b, c)      → sırayla
  group(a, b)         → paralel
  chord(group, cb)    → paralel bitince callback

Bizim graf tam bir chord: (cek→tara) ve (test) paralel, ikisi bitince esle→rapor.

DİKKAT — canvas'ın veri akışı kuralı: chain'de bir task'ın dönüşü SONRAKİNİN
İLK ARGÜMANI olur. Bu yüzden imzalar buna göre yazılmak zorunda; graf şekli
fonksiyon imzalarına SIZAR. Board'da böyle bir sızıntı yok (`parents` ayrı bir alan).

NE YAPAR : dağıtım · retry (max_retries) · at-least-once (acks_late)
NE YAPMAZ: DAG durumu · checkpoint · "şu an neredeyiz" sorusu · iptal zinciri

    .venv/bin/python saf-motorlar/saf_celery.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from celery import Celery, chain, chord  # noqa: E402

import isakisi as J  # noqa: E402

POC = Path(os.environ.get("SAF_CELERY_DIR", tempfile.mkdtemp(prefix="saf_celery_")))
Q, P = POC / "q", POC / "p"
for d in (Q, P):
    d.mkdir(parents=True, exist_ok=True)

app = Celery("saf_celery", broker="filesystem://",
             broker_transport_options={"data_folder_in": str(Q),
                                       "data_folder_out": str(Q),
                                       "processed_folder": str(P),
                                       "store_processed": True},
             backend=f"file://{POC / 'sonuc'}")
(POC / "sonuc").mkdir(exist_ok=True)
app.conf.update(task_acks_late=True, task_reject_on_worker_lost=True,
                worker_prefetch_multiplier=1, result_expires=3600)


# ── Task'lar: imzalar CANVAS'a göre şekilleniyor (graf, imzaya sızıyor) ──

@app.task(bind=True, max_retries=2, default_retry_delay=1)
def t_cek(self):
    try:
        return J.cek()
    except Exception as e:
        raise self.retry(exc=e)


@app.task(bind=True, max_retries=2, default_retry_delay=1)
def t_test(self):
    try:
        return J.test()
    except Exception as e:
        raise self.retry(exc=e)


@app.task(bind=True, max_retries=2, default_retry_delay=1)
def t_tara(self, cek_sonuc):          # ← chain: önceki dönüş İLK argüman
    try:
        return J.tara(cek_sonuc)
    except Exception as e:
        raise self.retry(exc=e)


@app.task(bind=True, max_retries=2, default_retry_delay=1)
def t_esle(self, header_sonuclari):   # ← chord: group sonuçları LİSTE olarak
    try:
        tara_s, test_s = header_sonuclari
        return {"esle": J.esle(tara_s, test_s), "tara": tara_s, "test": test_s}
    except Exception as e:
        raise self.retry(exc=e)


@app.task(bind=True, max_retries=2, default_retry_delay=1)
def t_rapor(self, paket):             # ← esle'nin PAKETİ (tara/test'i taşımak zorunda)
    try:
        return J.rapor(paket["esle"], paket["tara"], paket["test"])
    except Exception as e:
        raise self.retry(exc=e)


def main():
    print("═" * 74)
    print("SAF CELERY — canvas (chain/chord), board YOK")
    print(f"  broker: filesystem · {POC}")
    print("═" * 74)

    worker = subprocess.Popen(
        [sys.executable, "-m", "celery", "-A", "saf_celery", "worker",
         "--pool=solo", "--concurrency=1", "--loglevel=ERROR",
         "--without-gossip", "--without-mingle", "--without-heartbeat"],
        cwd=str(HERE), env={**os.environ, "SAF_CELERY_DIR": str(POC),
                            "PYTHONPATH": str(HERE)},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(6)
        t0 = time.time()
        # chord: [cek→tara, test] paralel; bitince esle→rapor
        akis = chord(
            [chain(t_cek.s(), t_tara.s()), t_test.s()],
            chain(t_esle.s(), t_rapor.s()))
        r = akis.apply_async()
        print("  akış kuyruğa atıldı, bekleniyor…")

        son = None
        for _ in range(60):
            if r.ready():
                break
            time.sleep(1)
        if not r.ready():
            print("  ✗ 60 sn'de bitmedi")
            print("  → Celery 'akış neredeyse takıldı' sorusuna CEVAP VEREMİYOR:")
            print("    hangi adımın koştuğunu, hangisinin beklediğini bilmiyor.")
            return
        try:
            son = r.get(timeout=10)
        except Exception as e:
            print(f"  ✗ akış hatayla bitti: {type(e).__name__}: {str(e)[:120]}")
            print("  → Kalan adımlara ne olduğu BİLİNMİYOR: iptal kaydı yok,")
            print("    'hangi adımlar koştu' sorusunun cevabı yok.")
            return

        ok, not_ = J.dogrula({"rapor": son})
        print(f"\n  {'✓' if ok else '✗'} {not_} · {round(time.time()-t0, 2)} sn")
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except Exception:
            worker.kill()
        shutil.rmtree(HERE / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
