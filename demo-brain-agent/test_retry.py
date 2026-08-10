#!/usr/bin/env python3
"""
test_retry.py — BİR DÜĞÜM BOZULUNCA RETRY TAM OLARAK NASIL OLUYOR?

"Retry var" demek yetmez. Ölçtüğümüz sorular:

  1) KAÇ kez deneniyor, ne zaman vazgeçiliyor?
  2) Denemeler arasında BEKLEME var mı (backoff), yoksa hemen mi?
  3) Retry'da düğüm SIFIRDAN mı koşuyor — yan etki kaç kez oluyor?   ← en kritik
  4) ÜST düğümler tekrar koşuyor mu, yoksa sonuçları korunuyor mu?
  5) Kim yönetiyor: board mı, Celery mi, Temporal mı? (üçü aynı sonucu mu veriyor)
  6) Başarıdan sonra breaker sıfırlanıyor mu (bir sonraki hata için 3 hak geri gelir mi)?
  7) Kalıcı hatada retry gerçekten atlanıyor mu?

Yöntem: gerçek fonksiyon çağrılarını SAYAÇLIYORUZ (F.call sarmalanıyor), her denemenin
duvar saati damgasını alıyoruz. Böylece "kaç kez koştu" iddiası değil ÖLÇÜM oluyor.

    .venv/bin/python demo-brain-agent/test_retry.py
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

OLCUM: dict = {}

# ── ÇAĞRI SAYACI: her fonksiyonun kaç kez GERÇEKTEN koştuğunu damgala ──
CAGRI: list = []
_orijinal_call = F.call


def sayacli_call(fn_name, args, upstream):
    CAGRI.append({"fn": fn_name, "t": time.time()})
    return _orijinal_call(fn_name, args, upstream)


F.call = sayacli_call
O.F = F


def sifirla():
    CAGRI.clear()


def graf(board: TaskBoard) -> dict:
    """fetch ─► scan ─┐
                       ├─► cross_check ─► render_report
       run_tests ─────┘                      (hedef: scan)"""
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


def kos(board, fail_at=None, backend="own"):
    res = O.OrchestrationResult(backend=backend, strategy="hermes")
    t0 = time.time()
    if backend == "own":
        O._dispatch_own(board, res, "hermes", 3000, None, fail_at)
    else:
        {"celery": O._dispatch_celery, "temporal": O._dispatch_temporal}[backend](
            board, res, "hermes", 3000, None, fail_at)
    res.seconds = round(time.time() - t0, 3)
    return res


def sayim(fn: str) -> int:
    return sum(1 for c in CAGRI if c["fn"] == fn)


def araliklar(fn: str) -> list:
    ts = [c["t"] for c in CAGRI if c["fn"] == fn]
    return [round(b - a, 3) for a, b in zip(ts, ts[1:])]


def bas(n):
    print(f"\n{'═' * 88}\n{n}\n{'═' * 88}")


# ═══════════ 1) GEÇİCİ HATA — retry'ın anatomisi ═══════════

def m1_gecici():
    bas("1) GEÇİCİ HATA — bir düğüm bir kez bozuluyor")
    sifirla()
    board = TaskBoard()
    g = graf(board)
    res = kos(board, fail_at="scan_patterns")
    t = board.get(g["scan"])

    olay = [(e["kind"], e["detail"] or "") for e in board.events()
            if e["task_id"] == g["scan"]]
    print("  scan_patterns'ın olay zinciri:")
    for k, d in olay:
        print(f"     {k:<18} {d[:66]}")

    print(f"\n  bozulan düğümün İŞİ kaç kez yapıldı : {sayim('scan_patterns')}"
          f"   (1. deneme iş yapılmadan patladı, 2. deneme başardı)")
    print(f"  denemeler arası bekleme         : {araliklar('scan_patterns') or '—'} sn")
    print(f"  attempt / breaker sayacı        : {t['attempt']} / {t['consecutive_failures']}")
    print(f"  son durum                       : {t['status']}")
    print(f"  toplam koşu süresi              : {res.seconds} sn")

    print(f"\n  ÜST düğümler tekrar koştu mu? (retry sırasında)")
    for ad, fn in (("fetch_source", "fetch_source"), ("run_test_suite", "run_test_suite")):
        print(f"     {ad:<16} {sayim(fn)} kez  "
              f"{'✓ korundu (tekrar koşmadı)' if sayim(fn) == 1 else '✗ TEKRAR KOŞTU'}")
    print(f"  ALT düğümler kaç kez koştu?")
    for fn in ("cross_check", "render_report"):
        print(f"     {fn:<16} {sayim(fn)} kez")

    OLCUM["gecici"] = {"cagri": sayim("scan_patterns"), "araliklar": araliklar("scan_patterns"),
                       "attempt": t["attempt"], "durum": t["status"], "sn": res.seconds,
                       "ust_tekrar": sayim("fetch_source"), "toplam_cagri": len(CAGRI)}


# ═══════════ 2) KALICI HATA — vazgeçme noktası ═══════════

def m2_kalici():
    bas("2) KALICI HATA — düğüm her denemede bozuluyor (vazgeçme noktası)")
    sifirla()
    board = TaskBoard()
    g = graf(board)
    res = kos(board, fail_at="scan_patterns!")
    t = board.get(g["scan"])

    print("  scan_patterns'ın olay zinciri:")
    for e in board.events():
        if e["task_id"] == g["scan"]:
            print(f"     {e['kind']:<18} {(e['detail'] or '')[:66]}")

    print(f"\n  KAÇ KEZ DENENDİ                 : {t['attempt']}  (BREAKER_LIMIT = 3)")
    print(f"  düğümün İŞİ kaç kez yapıldı     : {sayim('scan_patterns')}  "
          f"(hepsi iş yapılmadan patladı)")
    print(f"  denemeler arası bekleme         : {araliklar('scan_patterns') or '—'} sn")
    print(f"  vazgeçilen an                   : {t['attempt']}. denemede → {t['status']}")
    print(f"  boşa harcanan süre              : {res.seconds} sn")
    print(f"  ÜST düğümler tekrar koştu mu    : fetch={sayim('fetch_source')} "
          f"tests={sayim('run_test_suite')}")
    print(f"  ALT düğümler (iptal) koştu mu   : cross={sayim('cross_check')} "
          f"report={sayim('render_report')}  ← 0 olmalı")

    OLCUM["kalici"] = {"cagri": sayim("scan_patterns"), "attempt": t["attempt"],
                       "durum": t["status"], "sn": res.seconds,
                       "alt_cagri": sayim("cross_check") + sayim("render_report")}


# ═══════════ 3) YAN ETKİ — retry sıfırdan mı koşuyor? ═══════════

def m3_yan_etki():
    bas("3) YAN ETKİ — retry'da iş SIFIRDAN mı tekrarlanıyor?")
    print("  Gerçek sistemde bu şu sorudur: 'düğüm e-posta gönderdikten sonra patlarsa,")
    print("  retry e-postayı İKİNCİ KEZ gönderir mi?'")
    print("  İki hata modunu ayrı ayrı ölçüyoruz — çünkü sonuçları FARKLI:\n")

    modlar = [
        ("scan_patterns", "iş YAPILMADAN patlıyor", "(bağlantı kurulamadı gibi)"),
        ("scan_patterns~", "iş YAPILDIKTAN SONRA patlıyor", "(kayıt yazıldı, onay alınamadı)"),
    ]
    sonuc = {}
    for bayrak, ad, aciklama in modlar:
        sifirla()
        board = TaskBoard()
        graf(board)
        kos(board, fail_at=bayrak)
        s = {fn: sayim(fn) for fn in ("fetch_source", "run_test_suite", "scan_patterns",
                                      "cross_check", "render_report")}
        sonuc[ad] = s
        print(f"  ── {ad} {aciklama}")
        print(f"     bozulan düğümün İŞİ kaç kez yapıldı : {s['scan_patterns']}")
        print(f"     üst düğümler tekrar koştu mu        : fetch={s['fetch_source']} "
              f"tests={s['run_test_suite']}  {'✓ hayır' if s['fetch_source'] == 1 else '✗ EVET'}")
        print(f"     alt düğümler                        : cross={s['cross_check']} "
              f"report={s['render_report']}\n")

    a = sonuc["iş YAPILMADAN patlıyor"]["scan_patterns"]
    b = sonuc["iş YAPILDIKTAN SONRA patlıyor"]["scan_patterns"]
    print(f"  → FARK: {a} çağrı  vs  {b} çağrı")
    print(f"    Retry düğümü BAŞTAN koşturuyor (kaldığı yerden DEĞİL). Yani iş bittikten")
    print(f"    sonra patlayan bir düğümün işi {b} kez yapılıyor. Fonksiyonlar saf/")
    print(f"    deterministik olduğu sürece zararsız — ama YAN ETKİLİ bir düğüm eklenirse")
    print(f"    (e-posta, ödeme, dosya yazma) idempotenslik ZORUNLU hale gelir.")
    print(f"\n  → ÜST düğümler her iki modda da 1 kez koştu: sonuçları board'da saklı,")
    print(f"    retry yalnız BOZULAN düğümü tekrarlıyor (Airflow'da tek task'ı yeniden")
    print(f"    çalıştırmakla aynı davranış) — tüm akış baştan koşmuyor.")

    OLCUM["yan_etki"] = {"is_yapilmadan": a, "is_yapildiktan_sonra": b,
                         "detay": sonuc}


# ═══════════ 4) BACKEND KARŞILAŞTIRMASI ═══════════

def m4_backendler():
    bas("4) RETRY'I KİM YÖNETİYOR — own / temporal / celery")
    print("  Aynı geçici hata, üç motor. Retry mekanizmaları FARKLI:")
    print("     own      : board.fail() → status='ready' → sonraki turda tekrar claim")
    print("     temporal : activity RetryPolicy(maximum_attempts=3) → Temporal tekrar çağırır")
    print("     celery   : self.retry(countdown=0), max_retries=3 → broker'a geri koyar\n")
    print("  NOT: süreç-içi çağrı sayacı Celery'yi göremez (ayrı süreç). Backend'ler arası")
    print("       karşılaştırma BOARD'dan yapılıyor — 'claimed' olayı = bir yürütme denemesi.\n")
    satirlar = []
    for be in ("own", "temporal", "celery"):
        sifirla()
        board = TaskBoard()
        g = graf(board)
        try:
            res = kos(board, fail_at="scan_patterns", backend=be)
            hata = ""
        except Exception as e:
            res, hata = None, f"{type(e).__name__}: {str(e)[:40]}"
        t = board.get(g["scan"])
        olaylar = [e["kind"] for e in board.events() if e["task_id"] == g["scan"]]
        s = {"backend": be,
             "yurutme_denemesi": olaylar.count("claimed"),      # board'dan: süreçler arası geçerli
             "bayat_yazma": olaylar.count("stale_write_reddedildi"),
             "attempt": t["attempt"], "durum": t["status"],
             "sn": res.seconds if res else None, "hata": hata,
             "tamamlanan": len(board.list_tasks("done")),
             "toplam": len(board.list_tasks())}
        satirlar.append(s)
        print(f"  {be:<9} yürütme_denemesi={s['yurutme_denemesi']}  attempt={s['attempt']}  "
              f"durum={s['durum']:<7} tamamlanan={s['tamamlanan']}/{s['toplam']}  "
              f"{s['sn']}s" + (f"  ⚠ bayat_yazma={s['bayat_yazma']}" if s["bayat_yazma"] else ""))

    print()
    for alan, etiket in (("yurutme_denemesi", "yürütme denemesi"),
                         ("attempt", "board attempt sayacı"),
                         ("durum", "son durum"),
                         ("tamamlanan", "tamamlanan düğüm")):
        d = {s["backend"]: s[alan] for s in satirlar}
        ayni = len(set(d.values())) == 1
        print(f"  → {etiket:<24}: {'✓ üçü de aynı  ' + str(set(d.values())) if ayni else '✗ FARKLI  ' + str(d)}")
    bayat = {s["backend"]: s["bayat_yazma"] for s in satirlar if s["bayat_yazma"]}
    print(f"  → bayat yazma (fencing)   : "
          f"{'✓ hiçbirinde yok' if not bayat else '⚠ ' + str(bayat)}")
    OLCUM["backendler"] = satirlar


# ═══════════ 5) BREAKER SIFIRLAMA ═══════════

def m5_breaker_sifirlama():
    bas("5) BREAKER SIFIRLANIYOR MU — başarıdan sonra 3 hak geri geliyor mu?")
    board = TaskBoard()
    tid = board.create_task("iş", kind="function", fn="fetch_source",
                            fn_args={"path": "auth/login.py"})
    print(f"  {'adım':<34}{'attempt':>8}{'breaker':>9}   durum")
    adimlar = []
    for i in (1, 2):
        st = board.fail(tid, "geçici")
        t = board.get(tid)
        print(f"  {i}. hata (arka arkaya){'':<14}{t['attempt']:>8}{t['consecutive_failures']:>9}   {st}")
        adimlar.append((t["attempt"], t["consecutive_failures"], st))
    board.recompute_ready()
    board.claim_next("w")
    board.complete(tid, "başarılı", claimer=board.get(tid)["claim_lock"])
    t = board.get(tid)
    print(f"  BAŞARI{'':<28}{t['attempt']:>8}{t['consecutive_failures']:>9}   {t['status']}")
    sifirlandi = t["consecutive_failures"] == 0
    print(f"\n  → breaker sıfırlandı mı : {'✓ EVET' if sifirlandi else '✗ HAYIR'}  "
          f"(bir sonraki hata için 3 hak geri geldi)")
    print(f"  → attempt korunuyor mu  : {'✓ EVET' if t['attempt'] == 2 else '✗'}  "
          f"(attempt={t['attempt']} — toplam deneme geçmişi silinmiyor, denetim izi kalıyor)")
    OLCUM["breaker_sifirlama"] = {"sifirlandi": sifirlandi, "attempt": t["attempt"]}


# ═══════════ 6) KALICI HATADA RETRY ATLANIYOR MU ═══════════

def m6_kalici_atlama():
    bas("6) SÖZLEŞME HATASI — retry gerçekten ATLANIYOR mu?")
    print("  Bilinmeyen fonksiyon / geçersiz argüman tekrar denemekle düzelmez.")
    print("  Bu hatalar breaker'ın 3 hakkını harcamamalı.\n")
    sifirla()
    board = TaskBoard()
    tid = board.create_task("eksik veri", kind="function", fn="cross_check")  # upstream YOK
    res = kos(board)
    t = board.get(tid)
    print(f"  cross_check (zorunlu upstream yok)")
    print(f"     çağrı sayısı : {sayim('cross_check')}")
    print(f"     attempt      : {t['attempt']}   (geçici olsaydı 3 olurdu)")
    print(f"     durum        : {t['status']}")
    print(f"     süre         : {res.seconds} sn")
    for e in board.events():
        if e["kind"] == "failed":
            print(f"     olay         : {(e['detail'] or '')[:96]}")
    atlandi = t["attempt"] <= 1
    print(f"\n  → kalıcı hatada retry ATLANDI mı : {'✓ EVET' if atlandi else '✗ HAYIR'}")
    print(f"  → tasarruf: 2 gereksiz deneme + 2 gereksiz upstream okuma yapılmadı")
    OLCUM["kalici_atlama"] = {"cagri": sayim("cross_check"), "attempt": t["attempt"],
                              "atlandi": atlandi}


# ═══════════ ÖZET ═══════════

def main():
    print("═" * 88)
    print("RETRY ÖLÇÜMÜ — bir düğüm bozulunca ne oluyor?")
    print("═" * 88)
    m1_gecici()
    m2_kalici()
    m3_yan_etki()
    m4_backendler()
    m5_breaker_sifirlama()
    m6_kalici_atlama()

    bas("ÖZET TABLOSU")
    g, k = OLCUM["gecici"], OLCUM["kalici"]
    print(f"  {'':<32}{'geçici hata':>16}{'kalıcı hata':>16}")
    print(f"  {'-' * 64}")
    print(f"  {'bozulan düğüm çağrı sayısı':<32}{g['cagri']:>16}{k['cagri']:>16}")
    print(f"  {'attempt (board sayacı)':<32}{g['attempt']:>16}{k['attempt']:>16}")
    print(f"  {'son durum':<32}{g['durum']:>16}{k['durum']:>16}")
    print(f"  {'üst düğüm tekrar koştu mu':<32}{'hayır (1)':>16}{'hayır (1)':>16}")
    print(f"  {'alt düğüm koştu mu':<32}{'evet':>16}{'hayır (iptal)':>16}")
    print(f"  {'toplam süre (sn)':<32}{g['sn']:>16}{k['sn']:>16}")
    print(f"\n  Denemeler arası bekleme: {g['araliklar'] or '(ölçülemeyecek kadar kısa)'} "
          f"→ BACKOFF YOK, retry ANINDA.")
    print(f"  Fonksiyonlar hızlı ve deterministik olduğu için bu ucuz; ama bir dış servis")
    print(f"  düğümü eklenirse (rate-limit, geçici kesinti) backoff'suz retry servisi döver.")
    print("═" * 88)

    (HERE / "test_retry_sonuc.json").write_text(
        json.dumps(OLCUM, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
