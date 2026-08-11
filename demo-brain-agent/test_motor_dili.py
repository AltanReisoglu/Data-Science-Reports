#!/usr/bin/env python3
"""
test_motor_dili.py — Dört çeviricinin ürettiği kodu denetler.

Üç şeyi ölçüyor:
  1. Üretilen kod SÖZDİZİMSEL GEÇERLİ mi (`ast.parse`)
  2. GRAF KORUNUYOR mu — üretilen kodda her düğüm ve her kenar var mı
  3. "birebir" etiketi DOĞRU mu — canvas'ın ifade edemediği grafı yakalıyor mu

Ayrıca `motor_dili.airflow` ile `orchestrator.export_airflow_dag` AYNI grafı
üretiyor mu diye bağlar: ikisi ayrışırsa gösterilen kod ile koşan kod farklılaşır
ve panel yalan söyler.

    python test_motor_dili.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import motor_dili as M            # noqa: E402

GECTI, KALDI = [], []


def kontrol(ad: str, kosul: bool, detay: str = ""):
    (GECTI if kosul else KALDI).append(ad)
    print(f"  {'✓' if kosul else '✗'} {ad}" + (f"   {detay}" if detay else ""))


def graf(kenarlar: dict, idler: list) -> list:
    return [{"id": i, "fn": f"fn_{i}", "fn_args": {"p": i}, "kind": "function",
             "title": f"düğüm {i}", "parents": kenarlar.get(i, [])} for i in idler]


# ── DESENLER — motor karşılaştırmasında kullanılan dört desen ───────────────
DESENLER = {
    "D1 zincir":  graf({"b": ["a"], "c": ["b"]}, ["a", "b", "c"]),
    "D2 elmas":   graf({"b": ["a"], "c": ["a"], "d": ["b", "c"]}, ["a", "b", "c", "d"]),
    "D3 atlamalı": graf({"b": ["a"], "c": ["a", "b"]}, ["a", "b", "c"]),
    "D4 fan-out": graf({"b": ["a"], "c": ["a"], "d": ["a"]}, ["a", "b", "c", "d"]),
    "D5 çapraz":  graf({"c": ["a"], "d": ["b"]}, ["a", "b", "c", "d"]),
}


def test_sozdizimi():
    print("\n1) Üretilen kod sözdizimsel geçerli mi")
    for ad, nodes in DESENLER.items():
        v = M.cevir(nodes, f"test {ad}")
        for d in v["diller"]:
            if d.get("hata"):
                kontrol(f"{ad} · {d['ad']}", False, d["hata"]); continue
            try:
                ast.parse(d["kod"])
                ok, detay = True, f"{len(d['kod'].splitlines())} satır"
            except SyntaxError as e:
                ok, detay = False, f"satır {e.lineno}: {e.msg}"
            kontrol(f"{ad} · {d['ad']}", ok, detay)


def test_graf_korunuyor():
    print("\n2) Graf korunuyor mu — her düğüm ve kenar üretilen kodda var mı")
    for ad, nodes in DESENLER.items():
        v = M.cevir(nodes, "")
        for d in v["diller"]:
            kod = d.get("kod", "")
            eksik_dugum = [n["id"] for n in nodes
                           if M._pyid(n["id"]) not in kod and n["id"] not in kod]
            kontrol(f"{ad} · {d['ad']} · düğümler", not eksik_dugum,
                    f"eksik: {eksik_dugum}" if eksik_dugum else f"{len(nodes)} düğüm")

    # Kenarlar: airflow'da `>>`, own'da parents=[…], temporal'da s['x'] okuması
    print("\n   kenarlar (bağımlılık gerçekten kurulmuş mu)")
    nodes = DESENLER["D2 elmas"]
    v = {d["ad"]: d for d in M.cevir(nodes, "")["diller"]}
    kod_af = v["airflow"]["kod"]
    kenar_sayisi = sum(1 for n in nodes for p in n["parents"]
                       if f"{M._pyid(p)} >> {M._pyid(n['id'])}" in kod_af)
    kontrol("airflow · 4 kenar `>>` ile kurulmuş", kenar_sayisi == 4,
            f"{kenar_sayisi}/4")
    kod_own = v["own"]["kod"]
    kontrol("own · parents bağı kurulmuş",
            f"parents=[{M._pyid('b')}, {M._pyid('c')}]" in kod_own)
    kod_tp = v["temporal"]["kod"]
    kontrol("temporal · upstream s['…'] ile okunuyor",
            "'b': s['b']" in kod_tp and "'c': s['c']" in kod_tp)
    kontrol("temporal · paralel katman asyncio.gather ile",
            "asyncio.gather" in kod_tp)


def test_birebir_etiketi():
    print("\n3) 'birebir' etiketi doğru mu")
    # own / airflow / temporal her DAG'ı birebir ifade eder
    for ad, nodes in DESENLER.items():
        v = {d["ad"]: d for d in M.cevir(nodes, "")["diller"]}
        for dil in ("own", "airflow", "temporal"):
            kontrol(f"{ad} · {dil} birebir", v[dil]["birebir"] is True)

    # celery: yalnız seri-paralel grafları birebir ifade eder
    beklenen = {"D1 zincir": True, "D2 elmas": True, "D3 atlamalı": True,
                "D4 fan-out": True, "D5 çapraz": False}
    print("\n   celery — canvas ifade gücü")
    for ad, bekle in beklenen.items():
        v = {d["ad"]: d for d in M.cevir(DESENLER[ad], "")["diller"]}
        c = v["celery"]
        kontrol(f"{ad} · celery birebir={bekle}", c["birebir"] is bekle,
                (c.get("kayiplar") or [""])[0][:60] if not bekle else "")

    # Birebir DEĞİLSE tuzak listesinin BAŞINDA uyarı olmalı — sessiz geçmesin
    c = {d["ad"]: d for d in M.cevir(DESENLER["D5 çapraz"], "")["diller"]}["celery"]
    kontrol("celery · ifade edilemeyen graf tuzak listesinin başında uyarıyor",
            bool(c["tuzaklar"]) and "BİREBİR İFADE EDİLEMİYOR" in c["tuzaklar"][0])


def test_airflow_ile_orchestrator_ayni():
    """Gösterilen kod ile KOŞAN kod aynı grafı mı üretiyor?

    `motor_dili.airflow` panelde gösteriliyor; `orchestrator.export_airflow_dag`
    gerçekten koşuyor. İkisi ayrışırsa panel yalan söyler. Bu test onları bağlar.
    """
    print("\n4) motor_dili.airflow ≡ orchestrator.export_airflow_dag (graf yapısı)")
    try:
        from taskboard import TaskBoard
        import orchestrator as O
    except Exception as e:
        kontrol("orchestrator yüklenebildi", False, str(e)[:60]); return

    # Board KAYITLI OLMAYAN fonksiyonu reddediyor (BUG ⑨/⑩ düzeltmesi) —
    # bu yüzden karşılaştırma grafı gerçek fonksiyon adlarıyla kurulmalı.
    import functions as F
    F.use_pack("data")
    gercek = ["extract_records", "validate_schema", "transform_normalize",
              "aggregate_stats"]
    nodes = [{"id": i, "fn": f, "fn_args": {}, "kind": "function",
              "title": f"düğüm {i}", "parents": p}
             for i, f, p in zip(["a", "b", "c", "d"], gercek,
                                [[], ["a"], ["a"], ["b", "c"]])]
    b = TaskBoard()
    idmap = {}
    for n in nodes:
        idmap[n["id"]] = b.create_task(
            n["title"], "function", fn=n["fn"], fn_args=n["fn_args"],
            parents=[idmap[p] for p in n["parents"]])

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        yol = O.export_airflow_dag(b, Path(td) / "x.py", dag_id="t", goal="")
        kosan = yol.read_text(encoding="utf-8")
    gosterilen = M.airflow(nodes, "")["kod"]

    def yapi(kod: str, ids: list) -> tuple:
        """Kodda geçen task_id'ler ve `>>` kenarları — isimden bağımsız yapı."""
        dugum = sum(1 for i in ids if repr(i) in kod)
        kenar = kod.count(" >> ")
        return dugum, kenar

    d1, k1 = yapi(gosterilen, [n["id"] for n in nodes])
    d2, k2 = yapi(kosan, list(idmap.values()))
    kontrol("düğüm sayısı aynı", d1 == d2 == len(nodes), f"gösterilen={d1} koşan={d2}")
    kontrol("kenar sayısı aynı", k1 == k2 == 4, f"gösterilen={k1} koşan={k2}")
    for kod, ad in ((gosterilen, "gösterilen"), (kosan, "koşan")):
        try:
            ast.parse(kod); ok = True
        except SyntaxError as e:
            ok = False
        kontrol(f"{ad} DAG sözdizimsel geçerli", ok)


def test_bos_ve_tek():
    print("\n5) Kenar durumlar")
    v = M.cevir([], "boş graf")
    kontrol("boş graf dört dilde de patlamıyor",
            all(not d.get("hata") for d in v["diller"]),
            f"{[d.get('hata') for d in v['diller'] if d.get('hata')]}")
    tek = graf({}, ["a"])
    v = M.cevir(tek, "tek düğüm")
    kontrol("tek düğüm — dördü de geçerli kod üretiyor",
            all(not d.get("hata") and _gecerli(d["kod"]) for d in v["diller"]))
    bagimsiz = graf({}, ["a", "b", "c"])
    v = {d["ad"]: d for d in M.cevir(bagimsiz, "")["diller"]}
    kontrol("bağımsız düğümler celery'de tek group",
            "group(" in v["celery"]["kod"] and "chain(" not in v["celery"]["kod"])
    kontrol("bağımsız düğümler airflow'da kenarsız",
            " >> " not in v["airflow"]["kod"])


def _gecerli(kod: str) -> bool:
    try:
        ast.parse(kod); return True
    except SyntaxError:
        return False


if __name__ == "__main__":
    print("=" * 78)
    print(" MOTOR DİLİ — üretilen kod denetimi")
    print("=" * 78)
    test_sozdizimi()
    test_graf_korunuyor()
    test_birebir_etiketi()
    test_airflow_ile_orchestrator_ayni()
    test_bos_ve_tek()
    n = len(GECTI) + len(KALDI)
    print("\n" + "=" * 78)
    print(f"  SONUÇ: {len(GECTI)}/{n} kontrol geçti")
    if KALDI:
        print("  KALAN:")
        for k in KALDI:
            print(f"    ✗ {k}")
    print("=" * 78)
    sys.exit(1 if KALDI else 0)
