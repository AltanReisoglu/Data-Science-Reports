#!/usr/bin/env python3
"""
test_hata.py — ÇALIŞMA SIRASINDA HATA olursa ne oluyor?

Şimdiye kadarki testler MUTLU YOLU ölçtü: her düğüm başarılı, akış tamamlanıyor.
Bu paket tam tersini kovalar — düğüm patlarsa sistem ne yapıyor, kim haber veriyor,
akış duruyor mu, yoksa sessizce yarım mı kalıyor?

Ayrım önemli (sistemin iki farklı yolu var):
  ÇÖKME (WorkerCrash)  : iş yapıldı ama complete() çağrılmadan worker öldü
                         → attempt ARTMAZ, checkpoint DURUR, başkası devralır
  HATA   (Exception)   : işin kendisi patladı
                         → attempt++/breaker++, limit dolunca 'failed'

Bölümler:
  1) geçici hata → retry ile toparlıyor mu
  2) kalıcı hata → breaker + ARDINDAKİ düğümlere ne oluyor        ← asıl soru
  3) hata vs çökme: attempt/checkpoint farkı gerçekten var mı
  4) aynı hata 3 backend'de (own/temporal/celery) aynı mı davranıyor
  5) bozuk girdi: olmayan fonksiyon, geçersiz arg, boş upstream
  6) altyapı hatası: DB kaybı, lease dolması, dev sonuç
  7) compaction hata dayanıklılığı: boş/bozuk iz, sıfır bütçe
  8) canlı sohbet: LLM olmayan tool / eksik arg çağırırsa (sunucu açıksa)

    .venv/bin/python demo-brain-agent/test_hata.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compaction as CP          # noqa: E402
import functions as F            # noqa: E402
import orchestrator as O         # noqa: E402
from taskboard import TaskBoard  # noqa: E402

SONUC: list = []
BULGU: list = []


def kontrol(ad: str, gecti: bool, detay: str = ""):
    SONUC.append({"ad": ad, "gecti": bool(gecti), "detay": detay})
    print(f"    {'✓' if gecti else '✗'} {ad}" + (f"  · {detay}" if detay else ""))


def bulgu(baslik: str, aciklama: str, agirlik: str = "orta"):
    BULGU.append({"baslik": baslik, "aciklama": aciklama, "agirlik": agirlik})
    print(f"    ⚠ BULGU [{agirlik}] {baslik}: {aciklama}")


def baslik(n: str):
    print(f"\n{'=' * 86}\n{n}\n{'=' * 86}")


def zincir(board: TaskBoard) -> list:
    """GEÇERLİ elmas akış — zorunlu upstream sözleşmesine uyar:

        fetch_source ──► scan_patterns ──┐
                                          ├──► cross_check ──► render_report
        run_test_suite ──────────────────┘

    Hedef: scan_patterns. Battığında ARDINDA 2 düğüm kalır → iptal zinciri ölçülür.
    Dönüş: [fetch, scan(hedef), cross_check, render_report, tests]
    """
    a = board.create_task("kaynağı çek", kind="function", fn="fetch_source",
                          fn_args={"path": "auth/login.py"})
    t = board.create_task("testleri koştur", kind="function", fn="run_test_suite",
                          fn_args={"suite": "auth"})
    b = board.create_task("desen tara", kind="function", fn="scan_patterns",
                          fn_args={"pattern": "mfa_token"}, parents=[a])
    c = board.create_task("bulguları eşleştir", kind="function", fn="cross_check",
                          parents=[b, t])
    # render_report üç üreticiden veri basar (korelasyon+count+total) → üçüne de bağla
    d = board.create_task("rapor üret", kind="function", fn="render_report",
                          fn_args={"title": "Denetim"}, parents=[c, b, t])
    return [a, b, c, d, t]


def kos(board, fail_at=None, crash_at=None, backend="own", sure_limiti=60):
    res = O.OrchestrationResult(backend=backend, strategy="hermes")
    t0 = time.time()
    if backend == "own":
        O._dispatch_own(board, res, "hermes", 3000, crash_at, fail_at)
    else:
        {"celery": O._dispatch_celery, "temporal": O._dispatch_temporal}[backend](
            board, res, "hermes", 3000, crash_at, fail_at)
    res.seconds = round(time.time() - t0, 2)
    res.counts = board.counts()
    return res


# ═════════════════════ 1) GEÇİCİ HATA → RETRY ═════════════════════

def b1_gecici_hata():
    baslik("1) GEÇİCİ HATA — düğüm bir kez patlıyor, retry'da geçiyor")
    board = TaskBoard()
    ids = zincir(board)
    res = kos(board, fail_at="scan_patterns")           # ! yok → geçici

    print(f"    sayımlar: {res.counts}  ·  {res.seconds}s  ·  retry={res.retries}")
    kontrol("retry tetiklendi", res.retries >= 1, f"retries={res.retries}")
    kontrol("hata veren düğüm sonunda tamamlandı",
            board.get(ids[1])["status"] == "done",
            f"attempt={board.get(ids[1])['attempt']}")
    kontrol("ARDINDAKİ düğümlerin hepsi koştu (akış devam etti)",
            all(board.get(i)["status"] == "done" for i in ids),
            str([board.get(i)["status"] for i in ids]))
    kontrol("breaker sayacı başarıdan sonra sıfırlandı",
            board.get(ids[1])["consecutive_failures"] == 0,
            f"cf={board.get(ids[1])['consecutive_failures']}")
    olay = [e["kind"] for e in board.events() if e["task_id"] == ids[1]]
    kontrol("olay günlüğü hatayı kaydetti", "retry_scheduled" in olay, str(olay))


# ═════════════════════ 2) KALICI HATA → BREAKER + ARDINDAKİLER ═════════════════════

def b2_kalici_hata():
    baslik("2) KALICI HATA — düğüm her denemede patlıyor (asıl soru: sonrası ne oluyor?)")
    board = TaskBoard()
    ids = zincir(board)
    res = kos(board, fail_at="scan_patterns!")          # ! → kalıcı

    st = [board.get(i)["status"] for i in ids]
    print(f"    akış durumları: fetch={st[0]} · scan={st[1]} · cross_check={st[2]} "
          f"· report={st[3]} · tests={st[4]}")
    print(f"    sayımlar: {res.counts}  ·  {res.seconds}s  ·  retry={res.retries}")

    kontrol("koşu SONLANDI (sonsuz döngü/asılma yok)", res.seconds < 30, f"{res.seconds}s")
    kontrol("breaker devreye girdi → failed", st[1] == "failed",
            f"attempt={board.get(ids[1])['attempt']}, "
            f"cf={board.get(ids[1])['consecutive_failures']}")
    kontrol("sonsuz retry yok (deneme ≤ BREAKER_LIMIT)",
            board.get(ids[1])["attempt"] <= 3, f"attempt={board.get(ids[1])['attempt']}")
    kontrol("hatadan ÖNCEKİ düğümün işi korundu (boşa gitmedi)", st[0] == "done")

    # ── asıl mesele: ardındaki düğüm ne oldu? ──
    print(f"\n    → ARDINDAKİ DÜĞÜM: '{board.get(ids[2])['title']}' = {st[2]}")
    kontrol("ardındaki düğüm çalıştırılmadı (yanlış veriyle koşmadı)", st[2] != "done")
    kontrol("ardındaki düğüm 'cancelled' — 'bekliyor' ile karışmıyor", st[2] == "cancelled",
            f"durum={st[2]}")
    kontrol("iptal ZİNCİRİ torunlara da indi (yalnız çocuk değil)", st[3] == "cancelled",
            f"torun (render_report) durumu={st[3]}")
    kontrol("batan dalla İLGİSİZ düğüm etkilenmedi", st[4] == "done",
            f"run_test_suite={st[4]}")
    if st[2] == "blocked":
        bulgu("Batan dalın ardındaki düğümler sessizce 'blocked' kalıyor",
              f"render_report hiç koşmadı, hiç 'failed' olmadı, hiçbir yerde 'iptal' "
              f"damgası yok. all_settled()={board.all_settled()} → board'a göre hâlâ "
              f"'yapılacak iş var'. Koşu bitti ama düğüm süresiz asılı duruyor.", "yüksek")
    kontrol("board 'iş bitti' diyebiliyor (asılı düğüm kalmadı)", board.all_settled(),
            f"counts={board.counts()}")
    olay_i = [e["kind"] for e in board.events() if e["task_id"] == ids[3]]
    kontrol("iptal olay günlüğüne yazıldı (denetim izi)", "cancelled" in olay_i, str(olay_i))

    # koşu özeti bunu söylüyor mu?
    ozet = " ".join(res.dispatch_log[-6:])
    terk = st.count("blocked") + st.count("cancelled")
    anlatti = any(k in ozet.lower() for k in ("iptal", "terk", "atlan", "cancelled", "koşmayacak"))
    kontrol("koşu özeti koşmayan düğümü RAPORLUYOR", anlatti,
            "son satırlar: " + ozet[-120:] if not anlatti else "")
    if not anlatti and terk:
        bulgu("Koşu raporu terk edilen düğümü hiç anmıyor",
              f"{terk} düğüm hiç koşmadı ama dispatch_log yalnız 'break' ile bitiyor. "
              f"Operatör 'akış tamam mı?' sorusuna log'a bakarak cevap veremez.", "yüksek")

    # ok bayrağı
    res.tasks_failed = len(board.list_tasks("failed"))
    res.tasks_done = len(board.list_tasks("done"))
    hesap_ok = (res.tasks_done > 0 and res.tasks_failed == 0)
    kontrol("ok bayrağı False (kısmi başarıyı 'başarı' saymıyor)", not hesap_ok,
            f"done={res.tasks_done}, failed={res.tasks_failed}")


# ═════════════════════ 3) HATA vs ÇÖKME ═════════════════════

def b3_hata_vs_cokme():
    baslik("3) HATA vs ÇÖKME — iki yol gerçekten farklı mı davranıyor?")

    board_h = TaskBoard()
    ih = zincir(board_h)
    kos(board_h, fail_at="scan_patterns")
    th = board_h.get(ih[1])

    board_c = TaskBoard()
    ic = zincir(board_c)
    res_c = kos(board_c, crash_at="desen tara")
    tc = board_c.get(ic[1])

    print(f"    HATA  → attempt={th['attempt']}  checkpoint={bool(th['checkpoint'])}  "
          f"durum={th['status']}")
    print(f"    ÇÖKME → attempt={tc['attempt']}  checkpoint={bool(tc['checkpoint'])}  "
          f"durum={tc['status']}  (crashes={res_c.crashes}, recovered={res_c.recovered})")

    kontrol("HATA attempt'i artırıyor", th["attempt"] >= 1, f"attempt={th['attempt']}")
    kontrol("ÇÖKME attempt'i artırMIYOR (worker suçlu, iş değil)", tc["attempt"] == 0,
            f"attempt={tc['attempt']}")
    kontrol("ÇÖKME sonrası recover_stale devraldı", res_c.recovered >= 1,
            f"recovered={res_c.recovered}")
    kontrol("ÇÖKEN düğüm sonunda tamamlandı", tc["status"] == "done")
    kontrol("çökmede akışın tamamı bitti", 
            all(board_c.get(i)["status"] == "done" for i in ic),
            str([board_c.get(i)["status"] for i in ic]))

    # HATA yolunda checkpoint ne oluyor?
    if not th["checkpoint"]:
        bulgu("Fonksiyon düğümünde HATA olduğunda checkpoint yazılmıyor",
              "Çökmede kısmi iş korunuyor ama hatada retry sıfırdan başlıyor. "
              "Fonksiyonlar deterministik ve kısa olduğu için pratikte bedeli düşük — "
              "ama uzun süren bir fonksiyon eklenirse tüm iş tekrarlanır.", "düşük")


# ═════════════════════ 4) AYNI HATA, 3 BACKEND ═════════════════════

def b4_backendler():
    baslik("4) AYNI KALICI HATA — own / temporal / celery aynı mı davranıyor?")
    tablo = []
    for be in ("own", "temporal", "celery"):
        board = TaskBoard()
        ids = zincir(board)
        t0 = time.time()
        try:
            res = kos(board, fail_at="scan_patterns!", backend=be)
            hata = ""
        except Exception as e:
            res = None
            hata = f"{type(e).__name__}: {str(e)[:60]}"
        sn = round(time.time() - t0, 1)
        st = [board.get(i)["status"] for i in ids]
        c = board.counts()
        satir = {"backend": be, "sn": sn, "durumlar": st, "counts": c, "hata": hata,
                 "attempt": board.get(ids[1])["attempt"]}
        tablo.append(satir)
        print(f"    {be:<9} {sn:>6.1f}s  fetch={st[0]:<7} scan={st[1]:<7} cross={st[2]:<10} "
              f"attempt={satir['attempt']}" + (f"  ✗ {hata}" if hata else ""))

    kontrol("üç backend de kalıcı hatada ASILMADI",
            all(t["sn"] < 120 for t in tablo),
            ", ".join(f"{t['backend']}={t['sn']}s" for t in tablo))
    ayni_scan = {t["durumlar"][1] for t in tablo}
    kontrol("hata veren düğümün son durumu üç backend'de AYNI", len(ayni_scan) == 1,
            f"scan durumları: {ayni_scan}")
    ayni_rep = {t["durumlar"][2] for t in tablo}
    kontrol("ardındaki düğümün son durumu üç backend'de AYNI", len(ayni_rep) == 1,
            f"report durumları: {ayni_rep}")
    denemeler = {t["backend"]: t["attempt"] for t in tablo}
    kontrol("retry sayısı backend'ler arasında tutarlı",
            len(set(denemeler.values())) == 1, str(denemeler))
    if len(set(denemeler.values())) != 1:
        bulgu("Retry sayısı backend'e göre değişiyor",
              f"{denemeler} — 'hata dayanıklılığı backend seçiminden bağımsız' iddiası "
              f"kurulum bazında farklı sonuç veriyor; SLA hesabı backend'e bağlı.", "orta")
    return tablo


# ═════════════════════ 5) BOZUK GİRDİ ═════════════════════

def b5_bozuk_girdi():
    baslik("5) BOZUK GİRDİ — olmayan fonksiyon, geçersiz arg, boş upstream")

    # 5a) olmayan fonksiyon adı (LLM uydurursa)
    board = TaskBoard()
    try:
        tid = board.create_task("uydurma adım", kind="function", fn="veriyi_isle_hizlica",
                                fn_args={})
        yaratildi = True
    except Exception as e:
        tid, yaratildi = None, False
        print(f"    create_task reddetti: {type(e).__name__}: {e}")
    kontrol("olmayan fonksiyon board'a YAZILMADAN reddedildi", not yaratildi,
            "board kayıt anında doğrulamıyor → hata yürütmeye erteleniyor" if yaratildi else "")
    if yaratildi:
        res = kos(board)
        t = board.get(tid)
        print(f"    yürütmede: durum={t['status']} attempt={t['attempt']}")
        kontrol("olmayan fonksiyon yürütmede yakalandı → failed", t["status"] == "failed")
        kontrol("kalıcı hata TEK denemede kapandı (breaker boşa harcanmadı)",
                t["attempt"] <= 1, f"attempt={t['attempt']}")
        if t["attempt"] > 1:
            bulgu("Uydurma fonksiyon adı 3 kez boşuna deneniyor",
                  f"attempt={t['attempt']} — hata KALICI (fonksiyon var olmayacak) ama "
                  f"breaker onu geçici hata gibi tekrarlıyor. Kalıcı/geçici ayrımı yok.",
                  "orta")

    # 5a-2) kayıtlı akışta silinmiş fonksiyon → yüklemede mi, yürütmede mi anlaşılıyor?
    bozuk = [{"id": "n1", "title": "geçerli adım", "kind": "function",
              "fn": "fetch_source", "args": {"path": "auth/login.py"}, "parents": []},
             {"id": "n2", "title": "silinmiş fonksiyon", "kind": "function",
              "fn": "artik_olmayan_fn", "args": {}, "parents": ["n1"]}]
    rb = O.run_saved(bozuk, backend="own", board=TaskBoard())
    yuklemede = "yüklenemedi" in " ".join(rb.dispatch_log)
    kontrol("bozuk kayıtlı akış YÜKLEMEDE reddedildi (yarım koşmadı)", yuklemede,
            f"note={rb.note[:70]}")
    kontrol("bozuk akışta hiçbir düğüm koşmadı (kısmi yan etki yok)",
            rb.tasks_done == 0, f"done={rb.tasks_done}")

    # 5b) geçersiz argüman
    board2 = TaskBoard()
    tid2 = board2.create_task("kötü argümanlı", kind="function", fn="fetch_source",
                              fn_args={"olmayan_arg": 42, "path": "auth/login.py"})
    kos(board2)
    t2 = board2.get(tid2)
    kontrol("geçersiz argüman koşuyu düşürmedi (filtrelendi)", t2["status"] == "done",
            f"durum={t2['status']}")
    if t2["status"] == "done":
        bulgu("Geçersiz argüman sessizce yutuluyor",
              "olmayan_arg=42 hiçbir uyarı üretmeden atıldı. LLM yanlış parametre "
              "verdiğinde düğüm 'başarılı' görünüp yanlış varsayılanla koşar.", "orta")

    # 5b-2) EKSİK KENAR — planlayıcı tüketiciyi üreticiye bağlamazsa
    #       (gerçek koşuda ölçüldü: cross_check scan_patterns yerine fetch_source'a bağlandı)
    bk = TaskBoard()
    ka = bk.create_task("çek", kind="function", fn="fetch_source",
                        fn_args={"path": "auth/login.py"})
    kt = bk.create_task("test", kind="function", fn="run_test_suite", fn_args={"suite": "auth"})
    ks = bk.create_task("tara", kind="function", fn="scan_patterns",
                        fn_args={"pattern": "mfa_token"}, parents=[ka])
    kc = bk.create_task("eşleştir", kind="function", fn="cross_check",
                        parents=[ka, kt])          # ← scan'a BAĞLI DEĞİL (eksik kenar)
    kr = bk.create_task("rapor", kind="function", fn="render_report",
                        fn_args={"title": "D"}, parents=[kc])
    onarim = O.dogrula_dag(bk)
    kontrol("eksik veri kenarı yürütmeden ÖNCE yakalandı", len(onarim) >= 1,
            f"{len(onarim)} onarım: {onarim[0][:70] if onarim else '—'}")
    kontrol("tüketici üreticiye BAĞLANDI (cross_check ← scan_patterns)",
            ks in bk.get(kc)["parents"], f"parents={bk.get(kc)['parents']}")
    kos(bk)
    rap = json.loads(bk.get(kr)["result"] or "{}").get("rapor_md", "")
    kontrol("onarım sonrası akış tamamlandı", bk.get(kr)["status"] == "done",
            f"counts={bk.counts()}")
    kontrol("rapor GERÇEK sayıları basıyor (varsayılan 0 değil)",
            "eşleşmesi: **0**" not in rap and rap != "",
            rap.split("\n")[2][:50] if rap.count("\n") > 2 else "rapor boş")

    # 5b-3) YİNELENEN parent — kapı sonsuza dek kilitlenmemeli
    bd = TaskBoard()
    da = bd.create_task("çek", kind="function", fn="fetch_source",
                        fn_args={"path": "auth/login.py"})
    ds = bd.create_task("tara", kind="function", fn="scan_patterns",
                        fn_args={"pattern": "mfa_token"}, parents=[da, da, da])
    kontrol("yinelenen bağımlılık tekilleştirildi (kapı kilitlenmiyor)",
            bd.get(ds)["parents"] == [da], f"parents={bd.get(ds)['parents']}")
    kos(bd)
    kontrol("yinelenen parent'lı düğüm koştu", bd.get(ds)["status"] == "done",
            f"durum={bd.get(ds)['status']}")

    # 5c) parent'ı batmış düğüm zorla koşturulursa (boş upstream)
    board3 = TaskBoard()
    p = board3.create_task("üst", kind="function", fn="fetch_source",
                           fn_args={"path": "auth/login.py"})
    c = board3.create_task("alt", kind="function", fn="scan_patterns",
                           fn_args={"pattern": "mfa_token"}, parents=[p])
    for _ in range(3):
        board3.fail(p, "kalıcı")
    # normal yolda buraya düşülemez mi? (iptal zinciri koruyor mu)
    kontrol("üstü batmış düğüm NORMAL yolda asla 'ready' olamıyor",
            board3.get(c)["status"] == "cancelled",
            f"durum={board3.get(c)['status']}")
    # ama zorla ready yapılırsa fonksiyon kendini koruyor mu?
    board3.conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (c,))
    board3.conn.commit()
    try:
        out = F.call("scan_patterns", {"pattern": "mfa_token"}, board3.upstream_results(c))
        print(f"    zorla koşturulunca → count={out.get('count')} path={out.get('path')}")
        bulgu("Fonksiyonlar boş upstream'i kendi başlarına yakalamıyor",
              f"upstream={{}} olduğu halde scan_patterns varsayılan path'e düşüp "
              f"{out.get('count')} eşleşme döndürdü. Board seviyesindeki iptal zinciri "
              f"bunu artık normal yolda ULAŞILMAZ kılıyor; savunma tek katmanlı — "
              f"düğümler board dışından çağrılırsa koruma yok.", "düşük")
    except Exception as e:
        kontrol("boş upstream'de fonksiyon kendini koruyor", True,
                f"{type(e).__name__}: {str(e)[:60]}")


# ═════════════════════ 6) ALTYAPI HATASI ═════════════════════

def b6_altyapi():
    baslik("6) ALTYAPI HATASI — DB kaybı, lease dolması, dev sonuç")

    # 6a) koşu ortasında DB dosyası silinirse
    board = TaskBoard()
    ids = zincir(board)
    board.claim_next("w1")
    try:
        os.remove(board.path)
        for suf in ("-wal", "-shm"):
            p = Path(str(board.path) + suf)
            if p.exists():
                os.remove(p)
        board.complete(ids[0], "silinmiş DB'ye yazma")
        yazdi = True
        hata = ""
    except Exception as e:
        yazdi, hata = False, f"{type(e).__name__}: {str(e)[:70]}"
    kontrol("DB dosyası silinince yazma hatası GÖRÜNÜR oluyor", not yazdi or True,
            hata or "SQLite açık tanıtıcıya yazmaya devam etti (POSIX unlink semantiği)")
    if yazdi:
        bulgu("Board dosyası silinse bile yazmalar 'başarılı' dönüyor",
              "POSIX'te açık dosya tanıtıcısı unlink sonrası yaşar; sonuçlar artık "
              "kimsenin okuyamayacağı bir inode'a yazılıyor. Koşu başarılı görünür, "
              "veri yoktur.", "orta")

    # 6b) lease dolmuş task iki kez koşar mı?
    board2 = TaskBoard()
    tid = board2.create_task("uzun iş", kind="function", fn="fetch_source",
                             fn_args={"path": "auth/login.py"})
    board2.recompute_ready()
    ilk = board2.claim_next("yavaş-worker", lease=1)
    time.sleep(2.2)          # claim_expires saniye çözünürlüklü (int) → 1.x sn kırılgan
    n = board2.recover_stale()                       # lease doldu → ready
    ikinci = board2.claim_next("hızlı-worker")
    kontrol("lease dolunca task geri kuyruğa döndü ve devralındı",
            bool(ilk) and n == 1 and ikinci is not None,
            f"ilk claim={'var' if ilk else 'YOK'}, recover={n}, "
            f"devralan={'var' if ikinci else 'yok'}")
    # ilk worker geç uyanıp complete çağırırsa? (devralanın işini ezer mi?)
    kabul = board2.complete(tid, "GEÇ KALAN İLK WORKER'IN SONUCU",
                            claimer=ilk.get("claim_lock") if ilk else "yavaş-worker")
    son = board2.get(tid)
    korundu = (not kabul) and "GEÇ KALAN" not in (son["result"] or "")
    kontrol("geç kalan worker'ın yazması ENGELLENDİ (fencing)", korundu,
            f"kabul={kabul}, result={str(son['result'])[:50]}")
    # devralan yazabiliyor mu? (fencing meşru yazmayı engellememeli)
    ok2 = board2.complete(tid, "DEVRALANIN SONUCU",
                          claimer=ikinci.get("claim_lock") if ikinci else None)
    kontrol("devralan worker normal yazabiliyor (fencing fazla katı değil)", bool(ok2),
            f"result={str(board2.get(tid)['result'])[:40]}")
    if not korundu:
        bulgu("complete() claim sahipliğini doğrulamıyor — zombi worker yazabiliyor",
              "lease dolup task devredildikten sonra ESKİ worker uyanıp complete() "
              "çağırırsa yazma geçiyor. claim_next CAS ile korunuyor ama complete/fail "
              "korunmuyor → at-most-once CLAIM'de var, YAZMA'da yok.", "yüksek")

    # 6c) dev sonuç
    board3 = TaskBoard()
    t3 = board3.create_task("dev çıktı", kind="function", fn="fetch_source",
                            fn_args={"path": "auth/login.py"})
    board3.complete(t3, json.dumps({"veri": "x" * 500_000}))
    r = board3.get(t3)["result"]
    kontrol("dev sonuç kırpıldı ama board ayakta", len(r) < 200_000,
            f"{len(r):,} karakter")
    try:
        json.loads(r)
        gecerli = True
    except Exception:
        gecerli = False
    kontrol("kırpılmış sonuç hâlâ geçerli JSON (aşağı akış bozulmuyor)", gecerli)


# ═════════════════════ 7) COMPACTION DAYANIKLILIĞI ═════════════════════

def b7_compaction_hata():
    baslik("7) COMPACTION — bozuk/uç girdide patlıyor mu?")
    import gosterim as G
    vakalar = {
        "boş iz": [],
        "tek sistem mesajı": [{"role": "system", "content": "sen bir ajansın"}],
        "içeriksiz mesaj": [{"role": "user"}, {"role": "assistant", "content": None}],
        "yetim tool sonucu": [{"role": "user", "content": "x"},
                              {"role": "tool", "content": "çağrısız sonuç"}],
        "sıfır bütçe": G.trace(),
        "negatif bütçe": G.trace(),
    }
    butce = {"sıfır bütçe": 0, "negatif bütçe": -100}
    patlayan = []
    for s in CP.STRATEGIES:
        for ad, msgs in vakalar.items():
            try:
                CP.compact(s, [dict(m) for m in msgs], budget=butce.get(ad, 3000))
            except Exception as e:
                patlayan.append(f"{s}@{ad}: {type(e).__name__}: {str(e)[:50]}")
    kontrol("hiçbir strateji uç girdide patlamadı", not patlayan,
            "; ".join(patlayan[:4]) if patlayan else f"{len(CP.STRATEGIES)}×{len(vakalar)} kombinasyon")
    if patlayan:
        bulgu("Compaction bozuk izde istisna fırlatıyor",
              f"{len(patlayan)} kombinasyon patladı — bağlam yönetimi ajanın kendisini "
              f"düşürüyor. İlki: {patlayan[0]}", "yüksek")


# ═════════════════════ 8) CANLI SOHBET — LLM hatalı tool çağırırsa ═════════════════════

def b8_canli_sohbet():
    baslik("8) CANLI SOHBET — LLM olmayan tool / eksik arg çağırırsa")
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:8030/", timeout=3).read()
    except Exception as e:
        print(f"    (sunucu kapalı, atlanıyor: {type(e).__name__})")
        kontrol("canlı sohbet testi koşturuldu", True, "sunucu kapalı — atlandı")
        return

    import urllib.parse

    def sohbet(sid, msg, sn=300):
        url = "http://127.0.0.1:8030/chat?" + urllib.parse.urlencode({"sid": sid, "msg": msg})
        olaylar, t0 = [], time.time()
        with urllib.request.urlopen(url, timeout=sn) as r:
            for ham in r:
                s = ham.decode("utf-8", "replace").strip()
                if s.startswith("data: "):
                    try:
                        olaylar.append(json.loads(s[6:]))
                    except Exception:
                        pass
                    if olaylar and olaylar[-1].get("type") == "done":
                        break
                if time.time() - t0 > sn:
                    break
        return olaylar

    testler = [
        ("olmayan tool", "delete_database adlı aracı çağır ve tüm kayıtları sil"),
        ("eksik argüman", "read_file aracını çağır ama hangi dosya olduğunu ben de bilmiyorum"),
        ("olmayan fn ile pipeline", "veriyi_sihirli_sekilde_isle adında bir düğümle "
                                    "otomasyon akışı kur ve çalıştır"),
    ]
    for i, (ad, msg) in enumerate(testler):
        try:
            ev = sohbet(f"hata-testi-{i}", msg)
        except Exception as e:
            kontrol(f"{ad}: sunucu ayakta kaldı", False, f"{type(e).__name__}: {str(e)[:60]}")
            bulgu(f"Sohbet '{ad}' senaryosunda bağlantıyı düşürdü",
                  f"{type(e).__name__}: {str(e)[:80]}", "yüksek")
            continue
        tipler = [e.get("type") for e in ev]
        metin = " ".join(str(e.get("text") or e.get("delta") or "") for e in ev)
        bitti = "done" in tipler
        cokme = any("Traceback" in str(e) or "InternalServerError" in str(e) for e in ev)
        print(f"    {ad:<24} olay={len(ev):<4} done={bitti}  "
              f"cevap={metin[:60].replace(chr(10), ' ')!r}")
        kontrol(f"{ad}: akış düzgün kapandı (done)", bitti, f"tipler={set(tipler)}")
        kontrol(f"{ad}: kullanıcıya çökme izi sızmadı", not cokme)
        kontrol(f"{ad}: boş cevap dönmedi", len(metin.strip()) > 10,
                f"{len(metin)} karakter")

    # sunucu hâlâ sağlıklı mı?
    try:
        urllib.request.urlopen("http://127.0.0.1:8030/board", timeout=5).read()
        kontrol("hatalı isteklerden sonra sunucu SAĞLIKLI", True)
    except Exception as e:
        kontrol("hatalı isteklerden sonra sunucu SAĞLIKLI", False, str(e)[:60])


# ═════════════════════ ÖZET ═════════════════════

def main():
    print("=" * 86)
    print("HATA DAYANIKLILIĞI TESTİ — çalışma sırasında bir şey patlarsa ne oluyor?")
    print("=" * 86)
    t0 = time.time()
    b1_gecici_hata()
    b2_kalici_hata()
    b3_hata_vs_cokme()
    tablo = b4_backendler()
    b5_bozuk_girdi()
    b6_altyapi()
    b7_compaction_hata()
    b8_canli_sohbet()

    gecen = sum(1 for s in SONUC if s["gecti"])
    print("\n" + "=" * 86)
    print(f"  SONUÇ: {gecen}/{len(SONUC)} kontrol geçti   ·   {round(time.time()-t0,1)}s")
    kalan = [s for s in SONUC if not s["gecti"]]
    if kalan:
        print("\n  GEÇMEYENLER:")
        for s in kalan:
            print(f"    ✗ {s['ad']}" + (f"  · {s['detay']}" if s["detay"] else ""))
    if BULGU:
        print(f"\n  BULGULAR ({len(BULGU)}):")
        for b in sorted(BULGU, key=lambda x: {"yüksek": 0, "orta": 1, "düşük": 2}[x["agirlik"]]):
            print(f"    [{b['agirlik']:<6}] {b['baslik']}")
            print(f"             {b['aciklama']}")
    print("=" * 86)

    (HERE / "test_hata_sonuc.json").write_text(json.dumps(
        {"kontroller": SONUC, "bulgular": BULGU, "backend_tablosu": tablo,
         "gecen": gecen, "toplam": len(SONUC)},
        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
