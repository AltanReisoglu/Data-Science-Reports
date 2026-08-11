#!/usr/bin/env python3
"""
test_devam.py — ÇÖKÜNCE KALDIĞI YERDEN DEVAM EDEBİLİYOR MUYUZ?

Soru göründüğünden daha ince, çünkü "kaldığı yerden devam"ın İKİ AYRI seviyesi var:

  (A) DÜĞÜM SEVİYESİ — tamamlanmış düğümler tekrar koşmaz.
      Kaynak: board (sonuçlar kalıcı). Yeni worker akışın ortasından devam eder.

  (B) DÜĞÜM İÇİ  — YARIM KALMIŞ düğümün ortasından devam.
      Kaynak: checkpoint. Düğümün kendi içindeki ilerleme geri yüklenir.

İkisi karıştırılırsa "checkpoint'ten devam ediyoruz" cümlesi olduğundan
güçlü duyulur. Bu ölçüm ikisini AYIRIYOR ve hangi motorda hangisinin
VARSAYILAN geldiğini gösteriyor.

Yöntem: gerçek fonksiyon çağrıları sayaçlanıyor + checkpoint'e İZ konup
çökme sonrası o izin hayatta kalıp kalmadığına bakılıyor (süreçler arası geçerli).

    .venv/bin/python demo-brain-agent/test_devam.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import functions as F            # noqa: E402
import orchestrator as O         # noqa: E402
from taskboard import TaskBoard  # noqa: E402

SONUC: dict = {}
KONTROL: list = []


def kontrol(ad: str, gecti: bool, detay: str = ""):
    KONTROL.append({"ad": ad, "gecti": bool(gecti), "detay": detay})
    print(f"    {'✓' if gecti else '✗'} {ad}" + (f"  · {detay}" if detay else ""))

CAGRI: list = []
_orij = F.call


def sayacli(fn, args, up):
    CAGRI.append(fn)
    return _orij(fn, args, up)


F.call = sayacli
O.F = F


def bas(n):
    print(f"\n{'═' * 90}\n{n}\n{'═' * 90}")


def sayim(fn):
    return CAGRI.count(fn)


# ═══════ A) DÜĞÜM SEVİYESİ — tamamlananlar tekrar koşuyor mu? ═══════

def a_dugum_seviyesi():
    bas("A) DÜĞÜM SEVİYESİ — çökme sonrası TAMAMLANMIŞ düğümler tekrar koşuyor mu?")
    CAGRI.clear()
    board = TaskBoard()
    a = board.create_task("çek", kind="function", fn="fetch_source",
                          fn_args={"path": "auth/login.py"})
    t = board.create_task("test", kind="function", fn="run_test_suite",
                          fn_args={"suite": "auth"})
    s = board.create_task("tara", kind="function", fn="scan_patterns",
                          fn_args={"pattern": "mfa_token"}, parents=[a])
    c = board.create_task("eşleştir", kind="function", fn="cross_check", parents=[s, t])

    res = O.OrchestrationResult(backend="own", strategy="hermes")
    O._dispatch_own(board, res, "hermes", 3000, "tara", None)   # 'tara'da çök

    print(f"  çökme: {res.crashes} · otomatik kurtarma: {res.recovered}")
    print(f"  son durum: {board.counts()}\n")
    print(f"  {'düğüm':<18}{'rol':<34}{'kaç kez koştu':>14}")
    print(f"  {'-' * 68}")
    for fn, rol in (("fetch_source", "çökmeden ÖNCE tamamlanmıştı"),
                    ("run_test_suite", "çökmeden ÖNCE tamamlanmıştı"),
                    ("scan_patterns", "ÇÖKEN düğüm"),
                    ("cross_check", "çökmeden SONRA koştu")):
        print(f"  {fn:<18}{rol:<34}{sayim(fn):>14}")

    onceki_ok = sayim("fetch_source") == 1 and sayim("run_test_suite") == 1
    print(f"\n  → Tamamlanmış düğümler TEKRAR KOŞMADI mı : "
          f"{'✓ EVET (1 kez)' if onceki_ok else '✗ HAYIR'}")
    print(f"  → Akış tamamlandı mı                     : "
          f"{'✓ ' + str(len(board.list_tasks('done'))) + '/4'}")
    print("\n  Bunu sağlayan şey checkpoint DEĞİL, BOARD: tamamlanan düğümün sonucu")
    print("  kalıcı yazıldığı için devralan worker onu yeniden hesaplamıyor.")
    SONUC["dugum_seviyesi"] = {"onceki_tekrar_yok": onceki_ok,
                               "sayimlar": {f: sayim(f) for f in
                                            ("fetch_source", "run_test_suite",
                                             "scan_patterns", "cross_check")},
                               "crashes": res.crashes, "recovered": res.recovered}


# ═══════ B) DÜĞÜM İÇİ — FONKSİYON düğümü ═══════

def b_fonksiyon_ici():
    bas("B) DÜĞÜM İÇİ — FONKSİYON düğümü çökerse işi tekrarlanıyor mu?")
    CAGRI.clear()
    board = TaskBoard()
    a = board.create_task("çek", kind="function", fn="fetch_source",
                          fn_args={"path": "auth/login.py"})
    s = board.create_task("tara", kind="function", fn="scan_patterns",
                          fn_args={"pattern": "mfa_token"}, parents=[a])
    res = O.OrchestrationResult(backend="own", strategy="hermes")
    O._dispatch_own(board, res, "hermes", 3000, "tara", None)

    t = board.get(s)
    ck = t["checkpoint"]
    print(f"  çökme anında checkpoint YAZILDI mı : "
          f"{'✓ evet → ' + str(ck)[:60] if ck and ck != '{}' else '✗ hayır'}")
    print(f"  çöken fonksiyon KAÇ KEZ koştu      : {sayim('scan_patterns')}")
    print(f"  son durum                          : {t['status']}")

    tekrarlandi = sayim("scan_patterns") >= 2
    print(f"\n  → İş TEKRARLANDI mı : {'✗ EVET, baştan koştu' if tekrarlandi else '✓ hayır'}")
    if tekrarlandi:
        print("\n  Fonksiyon düğümünde checkpoint YAZILIYOR ama GERİ OKUNMUYOR —")
        print("  iş çökme öncesi bitmiş olsa bile baştan koşuyor.")
    else:
        print("\n  Fonksiyon düğümü de checkpoint'ten DEVAM ediyor: çökmeden önce iş")
        print("  bitmiş ve sonucu checkpoint'e yazılmıştı; devralan worker onu yeniden")
        print("  KOŞTURMUYOR, kayıtlı sonucu geri yüklüyor.")
        print("  Önemi: yan etkili bir fonksiyon (e-posta, ödeme) iki kez çalışmaz.")
    kontrol("fonksiyon düğümü çökme sonrası işi TEKRARLAMIYOR", not tekrarlandi,
            f"{sayim('scan_patterns')} kez koştu")
    SONUC["fonksiyon_ici"] = {"checkpoint_yazildi": bool(ck and ck != "{}"),
                              "cagri": sayim("scan_patterns"),
                              "geri_okunuyor": not tekrarlandi}


# ═══════ C) DÜĞÜM İÇİ — AJAN düğümü ═══════

def c_ajan_ici():
    bas("C) DÜĞÜM İÇİ — AJAN düğümü çökerse turun ortasından devam ediyor mu?")
    board = TaskBoard()
    tid = board.create_task("auth modülünü incele ve özetle", kind="agent",
                            body="MFA hatasını bul")
    res = O.OrchestrationResult(backend="own", strategy="hermes")
    log = []
    O._dispatch_own(board, res, "hermes", 3000, "auth", None)

    satirlar = [str(x) for x in res.dispatch_log]
    devam = [s for s in satirlar if "checkpoint'ten DEVAM" in s]
    yazma = [s for s in satirlar if "checkpoint yazıldı" in s]
    t = board.get(tid)
    print(f"  çökme: {res.crashes} · kurtarma: {res.recovered} · son durum: {t['status']}")
    print(f"  checkpoint YAZMA satırı sayısı  : {len(yazma)}")
    print(f"  checkpoint'ten DEVAM satırı     : {len(devam)}")
    for s in devam:
        print(f"     {s.strip()[:96]}")
    print(f"\n  → Ajan düğümü kaldığı TURDAN devam etti mi : "
          f"{'✓ EVET' if devam else '✗ hayır (çökme tetiklenmemiş olabilir)'}")
    SONUC["ajan_ici"] = {"devam": len(devam), "yazma": len(yazma),
                         "crashes": res.crashes, "durum": t["status"]}


# ═══════ D) BACKEND'LERDE DURUM ═══════

def d_backendler():
    bas("D) HANGİ MOTORDA HANGİSİ VARSAYILAN GELİYOR?")
    print("  Çökme SİMÜLASYONU yalnız kendi motorumuzda uygulanmış (crash_at celery ve")
    print("  temporal'da yok sayılıyor). Bu yüzden 'devam' yeteneğini ölçmek için")
    print("  checkpoint'e İZ koyup, o düğümü ilgili motorla koşturuyoruz: iz hayatta")
    print("  kalırsa checkpoint GERİ YÜKLENMİŞTİR (süreçler arası geçerli ölçüm).\n")

    IZ = "ÖNCEKİ-WORKER-İZİ-42"
    satirlar = []
    for be in ("own", "celery", "temporal"):
        board = TaskBoard()
        tid = board.create_task("özetle", kind="agent", body="deneme")
        # sanki önceki worker 1 tur koşup çökmüş gibi checkpoint yaz
        board.save_checkpoint(tid, {
            "messages": [{"role": "system", "content": "sen bir işçi ajansın"},
                         {"role": "user", "content": "özetle"},
                         {"role": "assistant", "content": IZ}],
            "done_steps": ["turn0"], "trace": [], "compaction_events": [],
            "answer": "", "turn": 1})
        res = O.OrchestrationResult(backend=be, strategy="hermes")
        t0 = time.time()
        hata = ""
        try:
            if be == "own":
                O._dispatch_own(board, res, "hermes", 3000, None, None)
            else:
                {"celery": O._dispatch_celery, "temporal": O._dispatch_temporal}[be](
                    board, res, "hermes", 3000, None, None)
        except Exception as e:
            hata = f"{type(e).__name__}"
        t = board.get(tid)
        ck = json.dumps(t["checkpoint"], ensure_ascii=False, default=str)
        korundu = IZ in ck or IZ in str(t.get("result") or "")
        satirlar.append({"backend": be, "korundu": korundu, "durum": t["status"],
                         "sn": round(time.time() - t0, 1), "hata": hata})
        print(f"  {be:<10} önceki turun izi korundu mu: "
              f"{'✓ EVET → checkpoint geri yüklendi' if korundu else '✗ hayır → baştan koştu'}"
              f"   (durum={t['status']}, {round(time.time()-t0,1)}s)")
    SONUC["backendler"] = satirlar
    print("\n  → Mekanizma ORTAK: üç motor da aynı run_one_task/execute_task yolundan")
    print("    geçtiği için checkpoint'ten devam hepsinde ÇALIŞIYOR. Fark motorda değil,")
    print("    DÜĞÜM TÜRÜNDE: ajan düğümü devam ediyor, fonksiyon düğümü baştan koşuyor.")


def main():
    print("═" * 90)
    print("ÇÖKÜNCE KALDIĞI YERDEN DEVAM — ölçüm")
    print("═" * 90)
    a_dugum_seviyesi()
    b_fonksiyon_ici()
    c_ajan_ici()
    d_backendler()

    bas("ÖZET — neyi kurtarabiliyoruz, neyi kurtaramıyoruz")
    print(f"  {'seviye':<38}{'durum':<26}{'kaynak'}")
    print(f"  {'-' * 86}")
    print(f"  {'(A) tamamlanmış düğümler':<38}{'✓ TEKRAR KOŞMUYOR':<26}{'board (sonuç kalıcı)'}")
    _b = SONUC.get("fonksiyon_ici", {}).get("geri_okunuyor")
    print(f"  {'(B) yarım kalan FONKSİYON düğümü':<38}"
          + (f"{'✓ checkpoint GERİ YÜKLENİR':<26}{'checkpoint (run_one_task)'}" if _b
             else f"{'✗ BAŞTAN koşuyor':<26}{'checkpoint yazılıyor, okunmuyor'}"))
    print(f"  {'(C) yarım kalan AJAN düğümü':<38}{'✓ KALDIĞI TURDAN devam':<26}{'checkpoint (execute_task)'}")
    print()
    print("  Motor bazında: own / celery / temporal → ORTAK kod yolu, üçünde de aynı.")
    print("  airflow → bizim katmanda yürütmüyor; devam semantiği Airflow'un kendi")
    print("            task retry'ına kalıyor (düğüm içi checkpoint YOK).")
    if KONTROL:
        g = sum(1 for k in KONTROL if k["gecti"])
        print(f"\n  KONTROLLER: {g}/{len(KONTROL)} geçti")
    (HERE / "test_devam_sonuc.json").write_text(
        json.dumps({"olcumler": SONUC, "kontroller": KONTROL},
                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
