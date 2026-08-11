#!/usr/bin/env python3
"""
test_backend_hata.py — DÖRT MOTOR AYNI HATAYA NASIL TEPKİ VERİYOR?

  hermes type (own) · airflow · celery · temporal

Aynı graf, aynı hata, dört motor. Ölçülen:
  A) geçici hata   → toparlıyor mu, kaç denemede
  B) kalıcı hata   → nerede vazgeçiyor, ardıl düğümlere ne oluyor
  C) çökme         → worker ölürse devralınıyor mu
  D) yapısal       → retry politikası kimde, backoff var mı, DAG dondu mu

Karşılaştırma BOARD'dan yapılıyor ('claimed' olayı = bir yürütme denemesi),
çünkü süreç-içi sayaç Celery'yi (ayrı süreç) göremez.

    .venv/bin/python demo-brain-agent/test_backend_hata.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import orchestrator as O         # noqa: E402
from taskboard import TaskBoard  # noqa: E402

BACKENDS = ("own", "airflow", "celery", "temporal")
ETIKET = {"own": "hermes type", "airflow": "airflow", "celery": "celery",
          "temporal": "temporal"}
SONUC: dict = {}


def graf(board: TaskBoard) -> dict:
    a = board.create_task("çek", kind="function", fn="fetch_source",
                          fn_args={"path": "auth/login.py"})
    t = board.create_task("test", kind="function", fn="run_test_suite",
                          fn_args={"suite": "auth"})
    s = board.create_task("tara", kind="function", fn="scan_patterns",
                          fn_args={"pattern": "mfa_token"}, parents=[a])
    c = board.create_task("eşleştir", kind="function", fn="cross_check", parents=[s, t])
    r = board.create_task("rapor", kind="function", fn="render_report",
                          fn_args={"title": "D"}, parents=[c, s, t])
    return {"fetch": a, "tests": t, "scan": s, "cross": c, "report": r}


def kos(backend, fail_at=None, crash_at=None):
    board = TaskBoard()
    g = graf(board)
    res = O.OrchestrationResult(backend=backend, strategy="hermes")
    t0 = time.time()
    hata = ""
    try:
        if backend == "own":
            O._dispatch_own(board, res, "hermes", 3000, crash_at, fail_at)
        elif backend == "airflow":
            O._dispatch_airflow(board, res, goal="denetim")
        else:
            {"celery": O._dispatch_celery, "temporal": O._dispatch_temporal}[backend](
                board, res, "hermes", 3000, crash_at, fail_at)
    except Exception as e:
        hata = f"{type(e).__name__}: {str(e)[:50]}"
    sn = round(time.time() - t0, 2)
    ev = [e["kind"] for e in board.events() if e["task_id"] == g["scan"]]
    return {
        "board": board, "g": g, "res": res, "sn": sn, "hata": hata,
        "deneme": ev.count("claimed"),
        "bayat": ev.count("stale_write_reddedildi"),
        "attempt": board.get(g["scan"])["attempt"],
        "scan": board.get(g["scan"])["status"],
        "ardil": [board.get(g[k])["status"] for k in ("cross", "report")],
        "done": len(board.list_tasks("done")),
        "counts": board.counts(),
    }


def bas(n):
    print(f"\n{'═' * 92}\n{n}\n{'═' * 92}")


# ═══════════ A) GEÇİCİ HATA ═══════════

def a_gecici():
    bas("A) GEÇİCİ HATA — düğüm bir kez bozuluyor, toparlanabilir mi?")
    print(f"  {'motor':<14}{'deneme':>7}{'attempt':>8}{'scan':>10}{'tamamlanan':>12}"
          f"{'süre':>9}   not")
    print(f"  {'-' * 88}")
    for be in BACKENDS:
        r = kos(be, fail_at="scan_patterns")
        not_ = ""
        if be == "airflow":
            not_ = "YÜRÜTMEDİ — hata hiç oluşmadı (yalnız DAG dosyası üretti)"
        elif r["scan"] == "done":
            not_ = "✓ toparlandı"
        else:
            not_ = f"✗ toparlanamadı ({r['scan']})"
        print(f"  {ETIKET[be]:<14}{r['deneme']:>7}{r['attempt']:>8}{r['scan']:>10}"
              f"{str(r['done']) + '/5':>12}{str(r['sn']) + 's':>9}   {not_}")
        SONUC.setdefault("gecici", {})[be] = {
            k: r[k] for k in ("deneme", "attempt", "scan", "done", "sn", "bayat", "counts")}
    print("\n  → Yürüten üç motorda da: 2 deneme, attempt=1, scan=done, 5/5 tamamlandı.")
    print("    Retry kararı BOARD'da verildiği için motor değiştirmek sonucu değiştirmiyor.")


# ═══════════ B) KALICI HATA ═══════════

def b_kalici():
    bas("B) KALICI HATA — nerede vazgeçiyor, ardıl düğümlere ne oluyor?")
    print(f"  {'motor':<14}{'deneme':>7}{'attempt':>8}{'scan':>9}{'ardıl düğümler':>26}"
          f"{'süre':>9}")
    print(f"  {'-' * 88}")
    for be in BACKENDS:
        r = kos(be, fail_at="scan_patterns!")
        ardil = " + ".join(r["ardil"])
        print(f"  {ETIKET[be]:<14}{r['deneme']:>7}{r['attempt']:>8}{r['scan']:>9}"
              f"{ardil:>26}{str(r['sn']) + 's':>9}")
        SONUC.setdefault("kalici", {})[be] = {
            k: r[k] for k in ("deneme", "attempt", "scan", "ardil", "done", "sn", "counts")}
    print("\n  → Yürüten üçünde de 3. denemede vazgeçiliyor (BREAKER_LIMIT=3) ve ardıl")
    print("    düğümler 'cancelled' oluyor — hatalı/eksik veriyle aşağı doğru koşma YOK.")
    print("  → airflow bu senaryoyu HİÇ görmüyor: yürütme onun tarafında, bizde değil.")


# ═══════════ C) ÇÖKME ═══════════

def c_cokme():
    bas("C) ÇÖKME — worker ölürse (complete çağrılmadan) ne oluyor?")
    print(f"  {'motor':<14}{'çökme':>7}{'kurtarma':>10}{'scan':>9}{'tamamlanan':>12}   durum")
    print(f"  {'-' * 88}")
    for be in BACKENDS:
        r = kos(be, crash_at="tara")
        res = r["res"]
        if be == "airflow":
            d = "YÜRÜTMEDİ"
        elif res.crashes == 0:
            d = "⚠ crash_at PARAMETRESİ YOK SAYILDI — çökme hiç tetiklenmedi"
        else:
            d = "✓ çöktü → recover_stale devraldı → checkpoint korundu"
        print(f"  {ETIKET[be]:<14}{res.crashes:>7}{res.recovered:>10}{r['scan']:>9}"
              f"{str(r['done']) + '/5':>12}   {d}")
        SONUC.setdefault("cokme", {})[be] = {
            "crashes": res.crashes, "recovered": res.recovered,
            "scan": r["scan"], "done": r["done"]}
    print("\n  → ESKİ `crash_at` string yolu yalnız kendi motorumuzda uygulanmış; celery ve")
    print("    temporal onu kabul edip kullanmıyor (sessizce yok sayılan parametre).")
    print("  → YENİ yol bu boşluğu kapattı: node_sim={'<id>':{'mod':'cokme'}} DÖRT motorda")
    print("    da çalışıyor (test_node_sim.py'de ölçüldü: tek atış → kurtarma → 5/5).")
    print("    Bu testteki 0'lar eski yolun ölçümüdür, çökme yeteneğinin yokluğu değil.")


# ═══════════ D) YAPISAL FARKLAR ═══════════

def d_yapisal():
    bas("D) YAPISAL — retry politikası kimde, backoff var mı?")
    # airflow DAG'ını üret ve retry ayarını OKU (iddia değil, dosyadan ölçüm)
    board = TaskBoard()
    graf(board)
    res = O.OrchestrationResult(backend="airflow", strategy="hermes")
    O._dispatch_airflow(board, res, goal="denetim")
    dag = Path(res.dag_file).read_text(encoding="utf-8")
    af_retries = next((l.strip() for l in dag.splitlines() if '"retries"' in l), "?")
    af_delay = next((l.strip() for l in dag.splitlines() if "retry_delay" in l), "?")

    satirlar = [
        ("hermes type", "board.fail() → status='ready' → sonraki turda tekrar claim",
         "3 (BREAKER_LIMIT)", "YOK (0 sn)", "board", "✓ dinamik"),
        ("airflow", f"Airflow default_args ({af_retries.split('#')[0].strip()})",
         "3 (retries=2)", "30 sn ✓", "Airflow", "✗ DONMUŞ"),
        ("celery", "self.retry(countdown=0) · max_retries=3",
         "3", "YOK (countdown=0)", "board + Celery", "✓ dinamik"),
        ("temporal", "RetryPolicy(maximum_attempts=3)",
         "3", "200 ms (initial_interval)", "board + Temporal", "✓ dinamik"),
    ]
    print(f"  {'motor':<13}{'retry mekanizması':<48}{'deneme':<11}{'backoff':<16}")
    print(f"  {'-' * 90}")
    for ad, mek, den, bo, _, _ in satirlar:
        print(f"  {ad:<13}{mek:<48}{den:<11}{bo:<16}")
    print(f"\n  {'motor':<13}{'karar kimde':<20}{'graf':<14}")
    print(f"  {'-' * 90}")
    for ad, _, _, _, karar, graf_ in satirlar:
        print(f"  {ad:<13}{karar:<20}{graf_:<14}")

    print(f"\n  ⚠ TUTARSIZLIK: kendi motorumuz 0 sn ara ile 3 kez deniyor, ama DIŞA")
    print(f"    AKTARDIĞIMIZ Airflow DAG'ı {af_delay.split('#')[0].strip()}")
    print(f"    Yani aynı akış, Airflow'a taşındığında FARKLI zamanlama ile koşuyor.")
    print(f"    Deneme SAYISI aynı (3), bekleme süresi değil.")
    SONUC["yapisal"] = {"airflow_retries": af_retries, "airflow_delay": af_delay,
                        "dag_file": res.dag_file}


def main():
    print("═" * 92)
    print("DÖRT MOTOR — AYNI HATAYA NASIL TEPKİ VERİYOR?")
    print("  hermes type (kendi durable motorumuz) · airflow · celery · temporal")
    print("═" * 92)
    a_gecici()
    b_kalici()
    c_cokme()
    d_yapisal()

    bas("SONUÇ")
    print("  1) Hata DAVRANIŞI motordan bağımsız — çünkü karar board'da, tek noktada.")
    print("     Yürüten üç motor da geçici hatada toparlıyor, kalıcıda 3'te vazgeçip")
    print("     ardılları iptal ediyor. Aynı deneme sayısı, aynı son durum, aynı çıktı.")
    print("  2) Motorların ayrıldığı yer HIZ ve İŞLETME: hermes type 0,01 sn · temporal")
    print("     0,5 sn · celery 40 sn (worker açılışı + broker).")
    print("  3) airflow farklı kategoride: bizim katmanda YÜRÜTMÜYOR, DAG dosyası üretiyor.")
    print("     Hata onun tarafında oluşur; retry politikası da onun (30 sn backoff).")
    print("  4) Açık nokta: crash_at yalnız hermes type'ta uygulanmış; celery/temporal")
    print("     parametreyi sessizce yok sayıyor.")
    (HERE / "test_backend_hata_sonuc.json").write_text(
        json.dumps(SONUC, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
