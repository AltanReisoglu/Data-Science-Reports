#!/usr/bin/env python3
"""
desen_celery.py — Üç deseni Celery'de kur.

CELERY'NİN WORKFLOW'A BAKIŞI: workflow = CANVAS (imza kompozisyonu).
`chain` / `group` / `chord` ile bir çağrı ağacı kurulur ve `apply_async()`
anında SABİTLENİR. Yani graf, gönderim anında donuyor.

Bunun üç deseni nasıl zorladığını burada göreceğiz:
  D2 zincir  : chain(...) — kolay
  D3 koşullu : canvas'ta "if" YOK. Çözüm: bir task içinde karar verip
               DİĞER TASK'I ORADAN çağırmak. Yani dal, canvas'ın dışına taşıyor.
  D4 dinamik : N gönderim anında bilinmiyor. Çözüm: bir task içinde
               chord'u ÇALIŞMA ANINDA kurup göndermek (replace / nested chord).

Her iki çözümde de ortak nokta: **kompozisyon kodun içine sızıyor**, dışarıdan
bakan biri grafı tek yerde göremiyor.

    .venv/bin/python saf-motorlar/desen_celery.py
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

from celery import Celery, chain, chord, group  # noqa: E402

import desenler as D  # noqa: E402

POC = Path(os.environ.get("DESEN_CELERY_DIR", tempfile.mkdtemp(prefix="desen_celery_")))
Q, P, R = POC / "q", POC / "p", POC / "r"
for d in (Q, P, R):
    d.mkdir(parents=True, exist_ok=True)

app = Celery("desen_celery", broker="filesystem://",
             broker_transport_options={"data_folder_in": str(Q),
                                       "data_folder_out": str(Q),
                                       "processed_folder": str(P),
                                       "store_processed": True},
             backend=f"file://{R}")
app.conf.update(task_acks_late=True, worker_prefetch_multiplier=1,
                result_expires=3600)


# ───────── D2 · ZİNCİR — canvas'ın rahat ettiği yer ─────────
@app.task(bind=True, max_retries=2, default_retry_delay=1)
def c_zincir(self, onceki=None, n: int = 1):
    """chain'de önceki dönüş İLK argüman → n'yi partial ile veriyoruz.

    NOT: `onceki=None` varsayılanı ZORUNLU — zincirin İLK task'ı hiç
    pozisyonel argüman almadan çağrılıyor. Yani canvas, fonksiyon imzasına
    "ben bir zincirin parçasıyım ve belki de başıyım" bilgisini dayatıyor.
    """
    try:
        return D.zincir_adim(n, onceki if isinstance(onceki, dict) else None)
    except Exception as e:
        raise self.retry(exc=e)


# ───────── D3 · KOŞULLU — canvas'ta "if" yok ─────────
@app.task
def c_tarama():
    return D.tarama()


@app.task
def c_raporla(t):
    return D.dal_raporla(t)


@app.task
def c_atla(t):
    return D.dal_atla(t)


@app.task(bind=True)
def c_dal_sec(self, t):
    """KARAR TASK'I — canvas bunu ifade edemediği için burada yapıyoruz.

    `self.replace(...)` ile bu task'ın yerine seçilen dalı koyuyoruz. Yani
    graf, çalışma anında task'ın İÇİNDEN değiştiriliyor — canvas dışarıdan
    bakınca "tek düğüm" görüyor.
    """
    secilen = c_raporla.s(t) if t["asti"] else c_atla.s(t)
    raise self.replace(secilen)


# ───────── D4 · DİNAMİK — N gönderim anında bilinmiyor ─────────
@app.task
def c_bul():
    return D.dosyalari_bul()


@app.task
def c_isle(path):
    return D.dosya_isle(path)


@app.task
def c_topla(sonuclar):
    return D.topla(sonuclar)


@app.task(bind=True)
def c_fanout(self, f):
    """FAN-OUT TASK'I — chord'u çalışma anında kurup kendi yerine koyuyor."""
    raise self.replace(chord([c_isle.s(p) for p in f["dosyalar"]], c_topla.s()))


def bekle(r, sn=90):
    for _ in range(sn * 2):
        if r.ready():
            return True
        time.sleep(0.5)
    return False


def main():
    print("═" * 78)
    print("CELERY — workflow = CANVAS (chain/group/chord), gönderimde SABİTLENİR")
    print(f"  broker: filesystem · {POC}")
    print("═" * 78)
    worker = subprocess.Popen(
        [sys.executable, "-m", "celery", "-A", "desen_celery", "worker",
         "--pool=solo", "--concurrency=1", "--loglevel=ERROR",
         "--without-gossip", "--without-mingle", "--without-heartbeat"],
        cwd=str(HERE), env={**os.environ, "DESEN_CELERY_DIR": str(POC),
                            "PYTHONPATH": str(HERE)},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(6)

        # D2 · zincir
        t0 = time.time()
        z = chain(*[c_zincir.s(n=i) for i in range(1, D.ZINCIR_UZUNLUK + 1)])
        r = z.apply_async()
        try:
            ok, not_ = (D.zincir_dogrula(r.get(timeout=60)) if bekle(r)
                        else (False, "60 sn'de bitmedi"))
        except Exception as e:
            # Celery hatayı ÇAĞIRANA fırlatıyor. Kalan zincir adımlarına ne
            # olduğuna dair HİÇBİR kayıt yok — 'atlandı' diye bir durum yok.
            ok, not_ = False, f"{type(e).__name__}: {str(e)[:58]}"
        print(f"  D2 zincir   {'✓' if ok else '✗'} {not_[:58]:<60} {round(time.time()-t0,2)}s")

        # D3 · koşullu
        t0 = time.time()
        r = chain(c_tarama.s(), c_dal_sec.s()).apply_async()
        try:
            ok, not_ = (D.kosullu_dogrula(r.get(timeout=60)) if bekle(r)
                        else (False, "60 sn'de bitmedi"))
        except Exception as e:
            ok, not_ = False, f"{type(e).__name__}: {str(e)[:60]}"
        print(f"  D3 koşullu  {'✓' if ok else '✗'} {not_[:58]:<60} {round(time.time()-t0,2)}s")

        # D4 · dinamik
        t0 = time.time()
        r = chain(c_bul.s(), c_fanout.s()).apply_async()
        try:
            ok, not_ = (D.dinamik_dogrula(r.get(timeout=90)) if bekle(r)
                        else (False, "90 sn'de bitmedi"))
        except Exception as e:
            ok, not_ = False, f"{type(e).__name__}: {str(e)[:60]}"
        print(f"  D4 dinamik  {'✓' if ok else '✗'} {not_[:58]:<60} {round(time.time()-t0,2)}s")
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except Exception:
            worker.kill()
        shutil.rmtree(HERE / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
