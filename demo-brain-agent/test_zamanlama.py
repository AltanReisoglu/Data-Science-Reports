#!/usr/bin/env python3
"""
test_zamanlama.py — cron zamanlama: ayrıştırıcı, claim yarışı, uçtan uca koşu.

Koçun 6. ekseni (scheduling) şimdiye kadar sistemde HİÇ yoktu; raporlarda bunu
"yok" diye belgelemiştik. Bu paket yeni eklenen zamanlayıcıyı ölçüyor:

  1) cron ayrıştırıcı — doğru anlar, sınır durumlar, geçersiz girdi
  2) next_run_at YAZMA anında türetiliyor mu
  3) claim ATOMİK mi (çok süreçli yarış → at-most-once)
  4) lease dolunca bayat claim devralınıyor mu (ayrı kurtarma adımı OLMADAN)
  5) uçtan uca: kayıtlı akış zamanlanıyor, koşuyor, takvim ilerliyor
  6) hata yolu: akış bulunamazsa / düğüm batarsa ne oluyor
  7) enabled=0 zamanlama tetiklenmiyor

    .venv/bin/python demo-brain-agent/test_zamanlama.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipelines as P               # noqa: E402
import scheduler as S               # noqa: E402

SONUC: list = []


def kontrol(ad: str, gecti: bool, detay: str = ""):
    SONUC.append({"ad": ad, "gecti": bool(gecti), "detay": detay})
    print(f"    {'✓' if gecti else '✗'} {ad}" + (f"  · {detay}" if detay else ""))


def bas(n):
    print(f"\n{'═' * 88}\n{n}\n{'═' * 88}")


def temiz_store() -> S.ScheduleStore:
    import tempfile
    return S.ScheduleStore(Path(tempfile.mkdtemp(prefix="sched_")) / "s.db")


def ornek_akis() -> str:
    """Geçerli (zorunlu upstream sözleşmesine uyan) küçük bir akış kaydet."""
    nodes = [
        {"id": "n1", "title": "kaynağı çek", "kind": "function", "fn": "fetch_source",
         "args": {"path": "auth/login.py"}, "parents": [], "priority": 5},
        {"id": "n2", "title": "testleri koştur", "kind": "function", "fn": "run_test_suite",
         "args": {"suite": "auth"}, "parents": [], "priority": 5},
        {"id": "n3", "title": "desen tara", "kind": "function", "fn": "scan_patterns",
         "args": {"pattern": "mfa_token"}, "parents": ["n1"], "priority": 5},
        {"id": "n4", "title": "eşleştir", "kind": "function", "fn": "cross_check",
         "args": {}, "parents": ["n3", "n2"], "priority": 5},
        {"id": "n5", "title": "rapor", "kind": "function", "fn": "render_report",
         "args": {"title": "Zamanlanmış Denetim"}, "parents": ["n4", "n3", "n2"],
         "priority": 5},
    ]
    return P.save(goal="zamanlanmış denetim", pack="audit", backend="own",
                  tasks=[{"id": n["id"], "title": n["title"], "kind": n["kind"],
                          "fn": n["fn"], "fn_args": n["args"], "parents": n["parents"],
                          "priority": n["priority"], "status": "done", "result": ""}
                         for n in nodes],
                  stats={"dugum": len(nodes)})


# ═══════════ 1) CRON AYRIŞTIRICI ═══════════

def b1_cron():
    bas("1) CRON AYRIŞTIRICI")
    baz = datetime(2026, 8, 10, 12, 0)          # Pazartesi 12:00

    vakalar = [
        ("0 8 * * *",     "her gün 08:00",        datetime(2026, 8, 11, 8, 0)),
        ("*/15 * * * *",  "15 dakikada bir",      datetime(2026, 8, 10, 12, 15)),
        ("30 9 1 * *",    "ayın 1'i 09:30",       datetime(2026, 9, 1, 9, 30)),
        ("0 0 * * 0",     "pazar gece yarısı",    datetime(2026, 8, 16, 0, 0)),
        ("0 8 * * 1-5",   "hafta içi 08:00",      datetime(2026, 8, 11, 8, 0)),
        ("0 12 * * *",    "bugün 12:00 GEÇTİ",    datetime(2026, 8, 11, 12, 0)),
    ]
    for cron, ad, beklenen in vakalar:
        try:
            g = S.sonraki_kosu(cron, baz)
            kontrol(f"{ad:<24} {cron:<14}", g == beklenen,
                    f"{g:%d.%m %H:%M}" + ("" if g == beklenen else f" ≠ {beklenen:%d.%m %H:%M}"))
        except Exception as e:
            kontrol(f"{ad:<24} {cron:<14}", False, f"{type(e).__name__}: {e}")

    # hafta sonu atlama
    cuma = datetime(2026, 8, 14, 9, 0)
    g = S.sonraki_kosu("0 8 * * 1-5", cuma)
    kontrol("Cuma'dan sonra hafta içi → PAZARTESİ", g == datetime(2026, 8, 17, 8, 0),
            f"{g:%A %d.%m}")

    # geçersizler
    for kotu, neden in [("99 8 * * *", "dakika 99"), ("0 8 * *", "4 alan"),
                        ("a b c d e", "harf"), ("0 8 * * 9", "hafta 9"),
                        ("*/0 * * * *", "adım 0")]:
        try:
            S.sonraki_kosu(kotu)
            kontrol(f"geçersiz reddedildi ({neden})", False, "kabul edildi!")
        except ValueError as e:
            kontrol(f"geçersiz reddedildi ({neden})", True, str(e)[:44])
        except Exception as e:
            kontrol(f"geçersiz reddedildi ({neden})", False, f"yanlış hata: {type(e).__name__}")


# ═══════════ 2) YAZMA ANINDA TÜRETME ═══════════

def b2_yazmada_turet():
    bas("2) next_run_at YAZMA ANINDA türetiliyor mu (okuma anında cron ayrıştırmadan)")
    st = temiz_store()
    pid = ornek_akis()
    sid = st.create("sabah denetimi", "0 8 * * *", pid)
    s = st.get(sid)
    beklenen = int(S.sonraki_kosu("0 8 * * *").timestamp())
    kontrol("next_run_at kayıtta dolu", s["next_run_at"] > 0,
            f"{datetime.fromtimestamp(s['next_run_at']):%d.%m %H:%M}")
    kontrol("cron'dan doğru türetilmiş", abs(s["next_run_at"] - beklenen) < 2)
    kontrol("claim boş doğuyor", s["claimed_at"] is None)
    kontrol("varsayılan etkin", s["enabled"] == 1)
    kontrol("insan okunur özet üretiliyor", S.cron_aciklama("0 8 * * *") == "her gün 08:00",
            S.cron_aciklama("0 8 * * *"))
    kontrol("geçersiz cron ile OLUŞTURULAMIYOR",
            _hata_verir(lambda: st.create("kötü", "99 8 * * *", pid)))


def _hata_verir(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


# ═══════════ 3) ÇOK SÜREÇLİ CLAIM YARIŞI ═══════════

def _yarisan(db_path: str, sid: str, q):
    st = S.ScheduleStore(Path(db_path))
    q.put(st.claim(sid))


def b3_claim_yarisi():
    bas("3) CLAIM ATOMİK Mİ — 8 ayrı SÜREÇ aynı zamanlamayı almaya çalışıyor")
    st = temiz_store()
    pid = ornek_akis()
    sid = st.create("yarış", "*/1 * * * *", pid)
    st.conn.execute("UPDATE schedules SET next_run_at=? WHERE id=?",
                    (int(time.time()) - 10, sid))          # vakti gelmiş yap

    q = mp.Queue()
    procs = [mp.Process(target=_yarisan, args=(str(st.path), sid, q)) for _ in range(8)]
    for p in procs:
        p.start()
    sonuclar = [q.get(timeout=60) for _ in procs]
    for p in procs:
        p.join()

    kazanan = sum(1 for r in sonuclar if r)
    print(f"    8 süreçten claim alabilen: {kazanan}")
    kontrol("TAM OLARAK BİR süreç claim aldı (at-most-once)", kazanan == 1,
            f"{kazanan} süreç aldı")
    kontrol("claim damgası yazıldı", st.get(sid)["claimed_at"] is not None)


# ═══════════ 4) LEASE + BAYAT DEVRALMA ═══════════

def b4_lease():
    bas("4) LEASE — kirası dolan claim AYRI KURTARMA ADIMI OLMADAN devralınıyor mu")
    st = temiz_store()
    pid = ornek_akis()
    sid = st.create("bayat", "*/1 * * * *", pid)
    now = int(time.time())
    st.conn.execute("UPDATE schedules SET next_run_at=? WHERE id=?", (now - 10, sid))

    kontrol("1. worker claim aldı", st.claim(sid, now))
    kontrol("2. worker AYNI ANDA alamadı", not st.claim(sid, now))
    # 1. worker öldü sayalım — claim kayıtta ASILI duruyor
    kontrol("claim tazeyken due() onu aday GÖSTERMİYOR",
            len(st.due(now=now + 10)) == 0)
    kontrol("kira dolmadan devralınamıyor",
            not st.claim(sid, now + S.LEASE_SECONDS - 5),
            f"lease={S.LEASE_SECONDS}s")
    # asıl kontrol: eski claim hâlâ dururken, kira dolunca tekrar aday olmalı
    kontrol("kira DOLUNCA due() bayat claim'i yeniden aday gösteriyor",
            len(st.due(now=now + S.LEASE_SECONDS + 10)) == 1,
            "ayrı bir recover_stale() süpürmesi çağrılmadı")
    kontrol("kira DOLUNCA devralınıyor (tek atomik UPDATE)",
            st.claim(sid, now + S.LEASE_SECONDS + 10))
    kontrol("devralan taze claim yazdı → tekrar aday değil",
            len(st.due(now=now + S.LEASE_SECONDS + 20)) == 0)


# ═══════════ 5) UÇTAN UCA ═══════════

def b5_uctan_uca():
    bas("5) UÇTAN UCA — zamanlanan akış gerçekten koşuyor mu")
    st = temiz_store()
    pid = ornek_akis()
    sid = st.create("denetim", "*/5 * * * *", pid)
    onceki_next = st.get(sid)["next_run_at"]
    st.conn.execute("UPDATE schedules SET next_run_at=? WHERE id=?",
                    (int(time.time()) - 5, sid))

    due = st.due()
    kontrol("vakti gelen zamanlama due() ile bulundu", len(due) == 1)

    kosan = S.tur(st, on_log=lambda m: print(f"      {m}"))
    kontrol("bir zamanlama koştu", kosan == 1, f"{kosan} koştu")

    s = st.get(sid)
    kontrol("claim serbest bırakıldı", s["claimed_at"] is None)
    kontrol("son durum 'ok'", s["last_status"] == "ok", f"durum={s['last_status']}")
    kontrol("takvim İLERLEDİ (sonraki koşu geleceğe alındı)",
            s["next_run_at"] > int(time.time()),
            f"sonraki: {datetime.fromtimestamp(s['next_run_at']):%d.%m %H:%M}")

    runs = st.runs(sid)
    kontrol("koşu geçmişine yazıldı", len(runs) == 1)
    if runs:
        r = runs[0]
        kontrol("koşu kaydı tamamlandı olarak işaretli", r["ended_at"] is not None and r["ok"] == 1)
        kontrol("düğüm sayıları kayıtta", (r["dugum"] or 0) == 5 and (r["tamamlanan"] or 0) == 5,
                f"{r['tamamlanan']}/{r['dugum']} düğüm · {r['saniye']}s · {r['detay']}")


# ═══════════ 6) HATA YOLU ═══════════

def b6_hata():
    bas("6) HATA YOLU — akış yoksa / düğüm batarsa")
    st = temiz_store()
    sid = st.create("olmayan akış", "*/5 * * * *", "p_yok_boyle_bir_sey")
    st.conn.execute("UPDATE schedules SET next_run_at=? WHERE id=?",
                    (int(time.time()) - 5, sid))
    S.tur(st)
    s = st.get(sid)
    kontrol("olmayan akış → durum 'hata'", s["last_status"] == "hata", f"{s['last_status']}")
    kontrol("claim yine de serbest bırakıldı (kilitli kalmadı)", s["claimed_at"] is None)
    kontrol("takvim yine de ilerledi (sonsuz yeniden deneme yok)",
            s["next_run_at"] > int(time.time()))
    r = st.runs(sid)[0]
    kontrol("hata koşu geçmişinde açıklamalı", r["ok"] == 0 and "bulunamadı" in (r["detay"] or ""),
            (r["detay"] or "")[:60])

    # batan düğüm: geçersiz akış (eksik zorunlu upstream) → yükleme reddedilir
    st2 = temiz_store()
    bozuk = P.save(goal="bozuk", pack="audit", backend="own",
                   tasks=[{"id": "n1", "title": "rapor", "kind": "function",
                           "fn": "render_report", "fn_args": {"title": "X"},
                           "parents": [], "priority": 5, "status": "done", "result": ""}],
                   stats={"dugum": 1})
    sid2 = st2.create("bozuk akış", "*/5 * * * *", bozuk)
    st2.conn.execute("UPDATE schedules SET next_run_at=? WHERE id=?",
                     (int(time.time()) - 5, sid2))
    S.tur(st2)
    s2 = st2.get(sid2)
    kontrol("eksik upstream'li akış → 'hata', sistem ayakta", s2["last_status"] == "hata",
            (st2.runs(sid2)[0]["detay"] or "")[:70])


# ═══════════ 7) ETKİN/PASİF ═══════════

def b7_enabled():
    bas("7) enabled=0 zamanlama tetikleniyor mu")
    st = temiz_store()
    pid = ornek_akis()
    sid = st.create("kapalı", "*/5 * * * *", pid)
    st.conn.execute("UPDATE schedules SET next_run_at=? WHERE id=?",
                    (int(time.time()) - 5, sid))
    st.set_enabled(sid, False)
    kontrol("kapalı zamanlama due()'da GÖRÜNMÜYOR", len(st.due()) == 0)
    kontrol("tur() onu koşturMUYOR", S.tur(st) == 0)
    st.set_enabled(sid, True)
    kontrol("tekrar açılınca aday oluyor", len(st.due()) == 1)


def main():
    print("═" * 88)
    print("ZAMANLAMA (CRON) TESTİ — koçun 6. ekseni")
    print("═" * 88)
    t0 = time.time()
    b1_cron()
    b2_yazmada_turet()
    b3_claim_yarisi()
    b4_lease()
    b5_uctan_uca()
    b6_hata()
    b7_enabled()

    gecen = sum(1 for s in SONUC if s["gecti"])
    print("\n" + "═" * 88)
    print(f"  SONUÇ: {gecen}/{len(SONUC)} kontrol geçti   ·   {round(time.time()-t0,1)}s")
    for s in SONUC:
        if not s["gecti"]:
            print(f"    ✗ {s['ad']}" + (f"  · {s['detay']}" if s["detay"] else ""))
    print("═" * 88)
    (HERE / "test_zamanlama_sonuc.json").write_text(
        json.dumps({"kontroller": SONUC, "gecen": gecen, "toplam": len(SONUC)},
                   ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
