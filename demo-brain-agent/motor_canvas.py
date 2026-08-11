#!/usr/bin/env python3
"""
motor_canvas.py — Kullanıcının grafını GERÇEK Celery Canvas ile koşturur.

Board YOK. Celery'nin kendi kompozisyon ilkelleriyle:
    chain(a, b, c)          sırayla
    group(a, b)             paralel
    chord(group(...), cb)   paralel bitince callback

`saf_celery.py` sabit bir 5 adımlık senaryo koşuyor; bu modül SOHBETTE KURULAN
grafı alıp canvas'a çeviriyor. Böylece "aynı graf, board'lu ve board'suz" yan
yana ölçülebiliyor.

    python motor_canvas.py             # en son akış
    python motor_canvas.py p_5702...   # belirli akış

DÜZELTME NOTU — bu modül bir yanlış anlamayı da kapatıyor:
Celery'nin sınırı uzun süre "retry baştan koşar" diye anlatıldı. O, adımlar TEK
task'ın içindeyken doğru. Canvas'ta zincirin HER HALKASI ayrı task'tır ve ayrı
retry alır — Airflow düğümünden farkı yok. Canvas'ın gerçek sınırı burada
ÖLÇÜLÜYOR: koşarken "hangi adımdayız" diye sorulamaması.

CANVAS'IN YAPISAL KISITI (kod bunu gösteriyor, gizlemiyor):
  1. Veri akışı POZİSYONEL: chain'de önceki dönüş sonrakinin İLK argümanı olur.
     Düğüm kimliği kaybolur — hangi upstream'in hangi veriyi verdiği bilinmez.
  2. Katman senkronizasyonu: canvas i. katmanı ÖNCEKİ TÜM katmanları bekletir.
     Grafta çapraz bağımlılık varsa fazladan bekleme doğar (bkz. motor_dili).
  3. Koşarken durum SORULAMAZ: AsyncResult yalnız "bitti mi" der.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from celery import Celery, chain, group  # noqa: E402

POC = Path(os.environ.get("MOTOR_CANVAS_DIR", tempfile.mkdtemp(prefix="canvas_")))
Q, P, S = POC / "q", POC / "p", POC / "sonuc"
for d in (Q, P, S):
    d.mkdir(parents=True, exist_ok=True)

app = Celery("motor_canvas", broker="filesystem://",
             broker_transport_options={"data_folder_in": str(Q),
                                       "data_folder_out": str(Q),
                                       "processed_folder": str(P),
                                       "store_processed": True},
             backend=f"file://{S}")          # chord SONUÇ DEPOSU olmadan çalışmaz
app.conf.update(task_acks_late=True, task_reject_on_worker_lost=True,
                worker_prefetch_multiplier=1, result_expires=3600)


@app.task(bind=True, name="canvas_fn", max_retries=2, default_retry_delay=1)
def canvas_fn(self, onceki, fn: str, args: dict, sim: dict | None = None):
    """Canvas halkası. `onceki` = zincirdeki bir önceki dönüş (chain) ya da
    group sonuçlarının LİSTESİ (chord).

    Dikkat: `onceki` düğüm KİMLİĞİ taşımıyor — canvas'ta veri pozisyonel akar.
    `functions.call` ise `{düğüm_id: sonuç}` bekliyor; burada sentetik anahtar
    uyduruyoruz. İşte "graf, imzaya sızıyor" dediğimiz şeyin somut hâli.
    """
    import functions as F

    ust = {}
    if isinstance(onceki, list):
        for i, r in enumerate(onceki):
            if isinstance(r, dict):
                ust[f"_canvas_{i}"] = r
    elif isinstance(onceki, dict):
        ust["_canvas_0"] = onceki

    try:
        s = sim or {}
        if s.get("mod") in ("gecici", "kalici"):
            n = int(self.request.retries or 0)
            if s["mod"] == "kalici" or n == 0:
                raise RuntimeError(f"[simulasyon:{s['mod']}] {fn} (deneme {n})")
        return F.call(fn, args or {}, ust)
    except RuntimeError as e:
        raise self.retry(exc=e)          # canvas'ta retry: halka bazında


def _katmanlar(nodes: list) -> list[list[dict]]:
    by = {n["id"]: n for n in nodes}
    d: dict[str, int] = {}

    def der(nid, gor=()):
        if nid in d:
            return d[nid]
        if nid in gor:
            return 0
        eb = [p for p in by.get(nid, {}).get("parents", []) if p in by]
        d[nid] = 1 + max((der(p, gor + (nid,)) for p in eb), default=-1) if eb else 0
        return d[nid]

    for n in nodes:
        der(n["id"])
    en = max(d.values(), default=0)
    return [[n for n in nodes if d[n["id"]] == k] for k in range(en + 1)]


def canvas_kur(nodes: list, sim: dict | None = None):
    """Grafı chain(group(K0), group(K1), …) biçimine çevir."""
    sim = sim or {}
    kat = _katmanlar(nodes)
    parcalar = []
    for i, k in enumerate(kat):
        imzalar = []
        for n in k:
            kw = {"fn": n.get("fn"), "args": n.get("fn_args") or {},
                  "sim": sim.get(n["id"])}
            # İlk katmanda önceki yok → immutable imza (`.si`) ile başlat
            imzalar.append(canvas_fn.si(None, **kw) if i == 0
                           else canvas_fn.s(**kw))
        parcalar.append(imzalar[0] if len(imzalar) == 1 else group(imzalar))
    return chain(*parcalar) if len(parcalar) > 1 else parcalar[0], kat


def kostur(nodes: list, sim: dict | None = None, zaman_asimi: int = 90) -> dict:
    """Canvas'ı gerçekten koştur. Worker'ı kendisi açar/kapatır."""
    t0 = time.time()
    worker = subprocess.Popen(
        [sys.executable, "-m", "celery", "-A", "motor_canvas", "worker",
         "--pool=solo", "--concurrency=1", "--loglevel=ERROR",
         "--without-gossip", "--without-mingle", "--without-heartbeat"],
        cwd=str(HERE),
        env={**os.environ, "MOTOR_CANVAS_DIR": str(POC), "PYTHONPATH": str(HERE)},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = [f"worker açılıyor (broker: filesystem · {POC.name})"]
    try:
        time.sleep(6)
        log.append("worker hazır — canvas kuruluyor")
        akis, kat = canvas_kur(nodes, sim)
        log.append(f"canvas: chain({len(kat)} katman) → "
                   + " → ".join(f"group({len(k)})" if len(k) > 1 else k[0].get("fn", "?")
                                for k in kat))
        r = akis.apply_async()
        log.append("kuyruğa atıldı · AsyncResult alındı")

        # ── CANVAS'IN SINIRI TAM BURADA ──
        # Beklerken "hangi adımdayız" diye soracak bir yer YOK. Elimizde yalnız
        # `ready()` var: bitti / bitmedi. Board'da düğüm düğüm durum vardı.
        bekleme = []
        for i in range(zaman_asimi):
            if r.ready():
                break
            if i in (5, 20, 45):
                bekleme.append(f"{i} sn — Celery'ye 'neredeyiz' diye sorulamıyor, "
                               f"yalnız ready()={r.ready()}")
            time.sleep(1)
        log += bekleme

        if not r.ready():
            return {"ok": False, "sn": round(time.time() - t0, 2),
                    "hata": f"{zaman_asimi} sn'de bitmedi",
                    "log": log + ["✗ takıldı — ve nerede takıldığı BİLİNMİYOR"],
                    "katman": len(kat), "sonuc": None, "eksikler": EKSIKLER}
        try:
            sonuc = r.get(timeout=5)
            ok = True
        except Exception as e:
            sonuc, ok = {"hata": f"{type(e).__name__}: {e}"}, False
            log.append(f"✗ zincir hata ile bitti: {type(e).__name__}")
            log.append("  → Canvas'ta HANGİ halkanın battığı ayrıca kaydedilmiyor")
        return {"ok": ok, "sn": round(time.time() - t0, 2), "log": log,
                "katman": len(kat), "katman_boyu": [len(k) for k in kat],
                "sonuc": sonuc, "eksikler": EKSIKLER}
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except Exception:
            worker.kill()


# Canvas'ın VERMEDİĞİ şeyler — board/Airflow/Temporal ile farkın listesi.
EKSIKLER = [
    {"ne": "düğüm bazlı durum", "canvas": "YOK — koşarken yalnız ready() var",
     "board": "her düğüm blocked/ready/running/done/failed/cancelled"},
    {"ne": "'nerede kaldı' sorgusu", "canvas": "YOK",
     "board": "board tablosu · Airflow metadata DB · Temporal history"},
    {"ne": "iptal kaydı", "canvas": "YOK — batan zincirin kalanı sessizce ölür",
     "board": "cancelled · Airflow upstream_failed"},
    {"ne": "düğüm içi checkpoint", "canvas": "YOK",
     "board": "save_checkpoint — çökmede kısmi iş korunur"},
    {"ne": "veri akışında düğüm kimliği", "canvas": "YOK — pozisyonel (ilk argüman)",
     "board": "upstream_results {düğüm_id: sonuç}"},
    {"ne": "koşullu dallanma", "canvas": "self.replace() — mantık task'lara dağılır",
     "board": "board'da yeni düğüm; Temporal'da düpedüz if"},
]


if __name__ == "__main__":
    import pipelines as PL

    pid = next((a for a in sys.argv[1:] if a.startswith("p_")), None)
    if not pid:
        ls = PL.listing()
        if not ls:
            print("kayıtlı akış yok — sohbette bir workflow kur."); sys.exit(1)
        pid = ls[0]["id"]
    d = PL.load(pid)
    if not d:
        print(f"akış bulunamadı: {pid}"); sys.exit(1)

    import functions as F
    F.use_pack(d.get("pack", "data"))

    print("═" * 76)
    print(f"CELERY CANVAS — board YOK · akış {pid} · {len(d['nodes'])} düğüm")
    print("═" * 76)
    r = kostur(d["nodes"])
    for l in r["log"]:
        print("  " + l)
    print(f"\n  {'✓' if r['ok'] else '✗'} {r['sn']} sn · "
          f"{r['katman']} katman {r.get('katman_boyu')}")
    if r.get("sonuc"):
        print(f"  son halkanın dönüşü: {str(r['sonuc'])[:150]}")
    print(f"\n  CANVAS'IN VERMEDİĞİ ({len(r['eksikler'])})")
    for e in r["eksikler"]:
        print(f"    · {e['ne']:<32} canvas: {e['canvas'][:44]}")
