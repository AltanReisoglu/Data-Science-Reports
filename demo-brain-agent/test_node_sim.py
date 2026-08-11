#!/usr/bin/env python3
"""
test_node_sim.py — DÜĞÜM BAZINDA simülasyon dört motorda da çalışıyor mu?

Bugüne kadar "şu düğüm patlasın" demenin tek yolu `fail_at` idi ve o tek bir GLOBAL
string'di, üstelik **fonksiyon adı ya da başlık alt-dizesiyle** eşleşiyordu. Aynı fn
iki düğümde kullanılırsa ikisi birden patlıyordu — yani "sadece 3. düğüm patlasın"
denemiyordu. `node_sim` bunu id bazlı çözüyor.

Ölçülenler:
  1) AYNI fn iki düğümde → yalnız HEDEFLENEN patlıyor mu (asıl senaryo)
  2) Altı mod (normal/gecici/kalici/sonra/cokme/yavas) beklendiği gibi mi
  3) Dört motor da AYNI sonucu veriyor mu
  4) Çökme TEK ATIŞ mı (sonsuz çökme döngüsü yok)
  5) Kayıtlı akış id'leri board'un yeni id'lerine doğru çevriliyor mu
  6) Eski `fail_at` yolu bozulmamış mı (geriye dönük uyum)

    .venv/bin/python demo-brain-agent/test_node_sim.py [--hizli]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import orchestrator as O         # noqa: E402
from taskboard import TaskBoard  # noqa: E402

SONUC: list = []


def kontrol(ad: str, gecti: bool, detay: str = ""):
    SONUC.append({"ad": ad, "gecti": bool(gecti), "detay": detay})
    print(f"    {'✓' if gecti else '✗'} {ad}" + (f"  · {detay}" if detay else ""))


def bas(n):
    print(f"\n{'═' * 86}\n{n}\n{'═' * 86}")


def akis() -> list:
    """AYNI fn'i İKİ düğümde kullanan akış — global fail_at'in çözemediği senaryo."""
    return [
        {"id": "n1", "title": "çek A", "kind": "function", "fn": "fetch_source",
         "args": {"path": "auth/login.py"}, "parents": [], "priority": 5},
        {"id": "n2", "title": "çek B", "kind": "function", "fn": "fetch_source",
         "args": {"path": "auth/other.py"}, "parents": [], "priority": 5},
        {"id": "n3", "title": "testler", "kind": "function", "fn": "run_test_suite",
         "args": {"suite": "auth"}, "parents": [], "priority": 5},
        {"id": "n4", "title": "tara", "kind": "function", "fn": "scan_patterns",
         "args": {"pattern": "mfa_token"}, "parents": ["n1"], "priority": 5},
        {"id": "n5", "title": "eşleştir", "kind": "function", "fn": "cross_check",
         "args": {}, "parents": ["n4", "n3"], "priority": 5},
    ]


def kos(backend="own", sim=None, args=None):
    b = TaskBoard()
    r = O.run_saved(akis(), backend=backend, board=b, node_sim=sim, arg_overrides=args)
    durum = {t["title"]: t["status"] for t in b.list_tasks()}
    deneme = {t["title"]: t["attempt"] for t in b.list_tasks()}
    return b, r, durum, deneme


# ═══════ 1) ASIL SENARYO: aynı fn, tek hedef ═══════

def b1_ayni_fn():
    bas("1) AYNI fn İKİ DÜĞÜMDE — yalnız hedeflenen patlamalı")
    print("   akışta iki `fetch_source` var: 'çek A' (n1) ve 'çek B' (n2)\n")

    b, r, d, dn = kos(sim={"n2": {"mod": "kalici"}})
    for t in ("çek A", "çek B", "tara", "testler", "eşleştir"):
        print(f"     {t:<10} {d[t]}")
    kontrol("hedeflenen düğüm (n2) battı", d["çek B"] == "failed", f"attempt={dn['çek B']}")
    kontrol("AYNI fn'li diğer düğüm (n1) ETKİLENMEDİ", d["çek A"] == "done")
    kontrol("n1'e bağlı 'tara' koştu", d["tara"] == "done")
    kontrol("ilgisiz 'testler' koştu", d["testler"] == "done")

    print("\n   ── karşılaştırma: ESKİ yol (global fail_at='fetch_source') ──")
    b2 = TaskBoard()
    O.run_saved(akis(), backend="own", board=b2, fail_at="fetch_source!")
    d2 = {t["title"]: t["status"] for t in b2.list_tasks()}
    print(f"     çek A={d2['çek A']}  çek B={d2['çek B']}")
    kontrol("eski yol İKİSİNİ birden vuruyordu (fark kanıtı)",
            d2["çek A"] == "failed" and d2["çek B"] == "failed",
            "node_sim bunu düğüm bazına indirdi")


# ═══════ 2) ALTI MOD ═══════

def b2_modlar():
    bas("2) MODLAR — her biri beklendiği gibi mi (hedef: n4 'tara')")
    print(f"   {'mod':<9}{'tara':<11}{'deneme':>7}{'akış':>10}   beklenen")
    print(f"   {'-' * 74}")

    bekl = {
        "normal": ("done", "hepsi tamamlanır"),
        "gecici": ("done", "1 retry sonrası toparlar"),
        "kalici": ("failed", "breaker → failed, ardıl cancelled"),
        "sonra":  ("done", "iş tekrarlanır ama sonunda toparlar"),
        "cokme":  ("done", "tek çökme → kurtarma → tamamlanır"),
    }
    for mod, (bekl_durum, aciklama) in bekl.items():
        b, r, d, dn = kos(sim={"n4": {"mod": mod}})
        c = b.counts()
        akis_ozet = "/".join(f"{k}:{v}" for k, v in sorted(c.items()))
        print(f"   {mod:<9}{d['tara']:<11}{dn['tara']:>7}{akis_ozet:>10}   {aciklama}")
        kontrol(f"mod={mod} → tara={bekl_durum}", d["tara"] == bekl_durum,
                f"gerçek={d['tara']}")
        if mod == "kalici":
            kontrol("  kalıcıda ardıl düğüm cancelled", d["eşleştir"] == "cancelled")
        if mod == "cokme":
            kontrol("  çökmede akış TAMAMLANDI (sonsuz döngü yok)",
                    c.get("done", 0) == 5, f"counts={c}")
            kontrol("  çökme sayacı 1 (tek atış)", r.crashes == 1, f"crashes={r.crashes}")

    t0 = time.time()
    kos(sim={"n4": {"mod": "yavas", "sn": 2}})
    sn = time.time() - t0
    kontrol("mod=yavas → akış en az 2 sn sürdü", sn >= 2.0, f"{round(sn, 2)} sn")


# ═══════ 3) DÖRT MOTOR AYNI MI ═══════

def b3_motorlar(hizli=False):
    bas("3) DÖRT MOTOR — aynı simülasyon, aynı sonuç mu?")
    motorlar = ["own", "temporal"] if hizli else ["own", "temporal", "celery"]
    print(f"   {'motor':<11}{'çek B':<10}{'eşleştir':<12}{'süre':>8}")
    print(f"   {'-' * 46}")
    goruntuler = []
    for be in motorlar:
        t0 = time.time()
        b, r, d, dn = kos(backend=be, sim={"n2": {"mod": "kalici"}})
        sn = round(time.time() - t0, 2)
        print(f"   {be:<11}{d['çek B']:<10}{d['eşleştir']:<12}{sn:>7}s")
        goruntuler.append((d["çek A"], d["çek B"], d["tara"], d["eşleştir"]))
    kontrol("motorlar AYNI düğüm durumlarını üretti", len(set(goruntuler)) == 1,
            str(set(goruntuler)) if len(set(goruntuler)) != 1 else "")

    # Airflow ayrı: kendi metadata DB'sinden okunur
    try:
        import airflow_runner as AR
        if AR.hazir()[0]:
            # n4 ('tara') hedefleniyor çünkü ARDILI var (n5 'eşleştir').
            # n2'nin ardılı yok → upstream_failed doğmaz, ölçüm anlamsız olurdu.
            b = TaskBoard()
            idmap = O.materialize(b, akis())
            sim = O.cevir_node_sim({"n4": {"mod": "kalici"}}, idmap)
            r = AR.kostur(b, node_sim=sim, goal="node_sim testi")
            s = r.get("sayim", {})
            print(f"   {'airflow':<11}{'(metadata DB)':<22}{r.get('sn')}s  → {s}")
            kontrol("airflow: 1 failed + ardıl upstream_failed",
                    s.get("failed", 0) == 1 and s.get("upstream_failed", 0) >= 1,
                    f"sayım={s}")
            kontrol("airflow: batan dalla ilgisiz düğümler koştu",
                    s.get("success", 0) >= 2, f"success={s.get('success')}")
        else:
            kontrol("airflow koşturulabildi", True, "Airflow kurulu değil — atlandı")
    except Exception as e:
        kontrol("airflow koşturulabildi", False, f"{type(e).__name__}: {str(e)[:60]}")


# ═══════ 4) ID ÇEVİRİSİ ═══════

def b4_id_cevirisi():
    bas("4) ID ÇEVİRİSİ — kayıtlı akış id'leri board'un yeni id'lerine")
    b = TaskBoard()
    idmap = O.materialize(b, akis())
    kontrol("materialize eski→yeni haritası üretti", len(idmap) == 5, f"{len(idmap)} eşleme")
    cev = O.cevir_node_sim({"n2": {"mod": "kalici"}}, idmap)
    kontrol("çeviri yeni id'ye bakıyor", list(cev) == [idmap["n2"]],
            f"{list(cev)} vs beklenen {[idmap['n2']]}")
    kontrol("zaten yeni id verilirse aynen geçer",
            list(O.cevir_node_sim({idmap["n1"]: {"mod": "kalici"}}, idmap)) == [idmap["n1"]])
    # çevrilmemiş id sessizce hiçbir şey yapmamalı ama akışı da bozmamalı
    b2, r2, d2, _ = kos(sim={"olmayan_id": {"mod": "kalici"}})
    kontrol("bilinmeyen id akışı BOZMUYOR", b2.counts().get("done") == 5,
            f"counts={b2.counts()}")


# ═══════ 5) ARGÜMAN OVERRIDE ═══════

def b5_arg_override():
    bas("5) ARGÜMAN OVERRIDE — panelden düğüm argümanı değiştirme")
    b, r, d, _ = kos(args={"n4": {"pattern": "verify"}})
    tid = [t["id"] for t in b.list_tasks() if t["title"] == "tara"][0]
    t = b.get(tid)
    kontrol("override board'a yazıldı", t["fn_args"].get("pattern") == "verify",
            f"fn_args={t['fn_args']}")
    kontrol("override'lı akış tamamlandı", d["tara"] == "done")


# ═══════ 6) GERİYE DÖNÜK UYUM ═══════

def b6_geriye_donuk():
    bas("6) GERİYE DÖNÜK UYUM — eski fail_at yolu hâlâ çalışıyor mu")
    b, r, d, dn = kos(sim=None)
    kontrol("node_sim=None → akış normal koşuyor", b.counts().get("done") == 5)

    b2 = TaskBoard()
    O.run_saved(akis(), backend="own", board=b2, fail_at="scan_patterns")
    d2 = {t["title"]: t["status"] for t in b2.list_tasks()}
    kontrol("fail_at geçici hata → toparlıyor", d2["tara"] == "done")

    b3 = TaskBoard()
    O.run_saved(akis(), backend="own", board=b3, fail_at="scan_patterns!")
    d3 = {t["title"]: t["status"] for t in b3.list_tasks()}
    kontrol("fail_at kalıcı hata → failed + cancelled",
            d3["tara"] == "failed" and d3["eşleştir"] == "cancelled")

    b4 = TaskBoard()
    r4 = O.run_saved(akis(), backend="own", board=b4, crash_at="tara")
    kontrol("crash_at eski yolu çalışıyor", r4.crashes == 1, f"crashes={r4.crashes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hizli", action="store_true", help="celery'yi atla")
    a = ap.parse_args()

    print("═" * 86)
    print("DÜĞÜM BAZINDA SİMÜLASYON TESTİ (node_sim)")
    print("═" * 86)
    t0 = time.time()
    b1_ayni_fn()
    b2_modlar()
    b3_motorlar(a.hizli)
    b4_id_cevirisi()
    b5_arg_override()
    b6_geriye_donuk()

    g = sum(1 for s in SONUC if s["gecti"])
    print("\n" + "═" * 86)
    print(f"  SONUÇ: {g}/{len(SONUC)} kontrol geçti  ·  {round(time.time() - t0, 1)}s")
    for s in SONUC:
        if not s["gecti"]:
            print(f"    ✗ {s['ad']}" + (f"  · {s['detay']}" if s["detay"] else ""))
    print("═" * 86)
    (HERE / "test_node_sim_sonuc.json").write_text(
        json.dumps({"kontroller": SONUC, "gecen": g, "toplam": len(SONUC)},
                   ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
