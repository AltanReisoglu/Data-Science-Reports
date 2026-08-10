#!/usr/bin/env python3
"""
test_matrix.py — Sohbet ajanının pipeline kurma/koşturma yollarını sistemli test eder.

Chat sunucusunun (8030) HTTP uçlarını gerçek kullanıcı gibi sürer ve her senaryo için
yapılandırılmış sonuç toplar. Çıktı: test_sonuclari.json (rapor bundan yazılır).

    .venv/bin/python demo-brain-agent/test_matrix.py            # tüm matris
    .venv/bin/python demo-brain-agent/test_matrix.py hizli      # sadece own+airflow
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8030"
OUT = Path(__file__).resolve().parent / "test_sonuclari.json"
RESULTS: list = []


def _get(path: str, **params) -> str:
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    with urllib.request.urlopen(url, timeout=600) as r:
        return r.read().decode("utf-8")


def settings(sid, **kw):
    _get("/settings", sid=sid, **kw)


def chat(sid: str, msg: str) -> dict:
    """SSE akışını oku, olayları topla."""
    url = BASE + "/chat?" + urllib.parse.urlencode({"sid": sid, "msg": msg})
    evs = []
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            evs.append(ev)
            if ev.get("type") == "done":
                break
    return {"events": evs, "sn": round(time.time() - t0, 1)}


def run_saved(sid: str, pid: str) -> dict:
    url = BASE + "/runpipeline?" + urllib.parse.urlencode({"sid": sid, "id": pid})
    evs = []
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            evs.append(ev)
            if ev.get("type") == "done":
                break
    return {"events": evs, "sn": round(time.time() - t0, 1)}


def digest(res: dict) -> dict:
    """Olay akışından ölçülebilir özet çıkar."""
    evs = res["events"]
    d = {
        "sn": res["sn"],
        "yol": "sohbet",              # sohbet | tek_tool | graf | kayitli
        "pack": None,
        "dugum": 0, "tamamlanan": 0, "fn": 0, "ajan": 0,
        "cokme": 0, "kurtarma": 0, "retry": 0,
        "compaction": 0, "indirgeme": 0,
        "hata": None, "not": None, "cevap": "",
        "olay_tipleri": {},
    }
    for e in evs:
        t = e.get("type")
        d["olay_tipleri"][t] = d["olay_tipleri"].get(t, 0) + 1
        if t == "error":
            d["hata"] = e.get("text", "")[:200]
        elif t == "log" and "akış paketi seçildi" in e.get("text", ""):
            d["pack"] = e["text"].split(":")[-1].strip()
        elif t == "phase":
            tx = e.get("text", "")
            if "GRAF" in tx:
                d["yol"] = "graf"
            elif tx.startswith("tek işlem") or tx.startswith("araç") or tx.startswith("fonksiyon"):
                d["yol"] = "tek_tool"
            elif "kayıtlı akış" in tx:
                d["yol"] = "kayitli"
        elif t == "start" and "kayıtlı akış" in e.get("text", ""):
            d["yol"] = "kayitli"
        elif t == "node_added":
            tx = e.get("text", "")
            # "PLAN özeti: ..." bir düğüm DEĞİL — sayma
            if tx.startswith("PLAN +") or tx.startswith("AKIŞ düğümü") or "KURTARILDI" in tx:
                d["dugum"] += 1
        elif t == "compaction":
            d["compaction"] += 1
        elif t == "reduction":
            d["indirgeme"] += 1
        elif t == "chat":
            d["yol"] = "sohbet"
            d["cevap"] = e.get("text", "")[:300]
        elif t == "summary":
            d["cokme"] = e.get("crashes", 0)
            d["kurtarma"] = e.get("recovered", 0)
            d["retry"] = e.get("retries", 0)
            if e.get("answer"):
                d["cevap"] = e["answer"][:300]
            tx = e.get("text", "")
            import re
            m = re.search(r"(\d+) (?:yeni )?düğüm", tx)
            if m:
                d["dugum"] = int(m.group(1))       # summary otorite
            m = re.search(r"(\d+) deterministik fonksiyon", tx)
            if m:
                d["fn"] = int(m.group(1))
            m = re.search(r"(\d+) LLM ajan", tx)
            if m:
                d["ajan"] = int(m.group(1))
            m = re.search(r"(\d+)/(\d+) düğüm", tx)
            if m:
                d["tamamlanan"], d["dugum"] = int(m.group(1)), int(m.group(2))
    if d["yol"] in ("graf", "kayitli") and not d["tamamlanan"]:
        d["tamamlanan"] = d["fn"] + d["ajan"]
    return d


def T(ad: str, kategori: str, sid: str, msg: str | None = None,
      pid: str | None = None, **ayar):
    """Tek test koş, sonucu kaydet."""
    print(f"\n▶ {ad}")
    try:
        if ayar:
            settings(sid, **ayar)
        res = run_saved(sid, pid) if pid else chat(sid, msg)
        d = digest(res)
    except Exception as e:
        d = {"hata": f"{type(e).__name__}: {e}", "sn": 0, "yol": "?", "olay_tipleri": {}}
    d.update({"ad": ad, "kategori": kategori, "istek": msg or f"(kayıtlı akış {pid})",
              "ayar": ayar,
              "_ham": [e for e in (res.get("events") if isinstance(res, dict) else [])
                       if e.get("type") in ("phase", "node_added", "summary", "error",
                                            "compaction", "reduction", "chat")][:40]})
    RESULTS.append(d)
    print(f"   yol={d.get('yol')} pack={d.get('pack')} düğüm={d.get('dugum')} "
          f"tamamlanan={d.get('tamamlanan')} çökme={d.get('cokme')} retry={d.get('retry')} "
          f"{d.get('sn')}sn" + (f"  ✗ {d['hata'][:80]}" if d.get("hata") else ""))
    OUT.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


GOALS = {
    "audit": "auth/login.py'ı oku, mfa_token desenini tara, testleri koştur, bulguları eşleştir ve rapor üret",
    "data": "siparişleri çek, şemayı doğrula, TRY'ye normalize et, müşteriye göre topla ve CSV çıkar",
    "deploy": "1.4.2 sürümünü paketle, duman testi koştur, canary yayınla ve sağlığa bak",
}


def main():
    hizli = len(sys.argv) > 1 and sys.argv[1] == "hizli"
    t0 = time.time()
    print("=" * 78)
    print("PIPELINE TEST MATRİSİ")
    print("=" * 78)

    # ── A) SOHBET DAVRANIŞI (yol ayrımı doğru mu?) ──
    T("A1 · düz sohbet", "sohbet", "a1", "merhaba, sen kimsin",
      backend="own", pack="audit", budget="3000", strategy="hermes", crash_at="")
    T("A2 · yetenek sorusu", "sohbet", "a2", "neler yapabilirsin")
    T("A3 · tek tool (audit)", "tek_tool", "a3", "testleri koştur")
    T("A4 · tek tool (data, oto-paket)", "tek_tool", "a4", "siparişleri çek")
    T("A5 · yapamayacağı iş", "sohbet", "a5",
      "bana bir React web sitesi yaz ve sunucuya deploy et")

    # ── B) GRAF KURMA: backend × paket ──
    backends = ["own"] if hizli else ["own", "temporal", "celery"]
    for bk in backends:
        for pk in ("audit", "data", "deploy"):
            T(f"B · graf · {bk} × {pk}", "graf", f"b_{bk}_{pk}", GOALS[pk],
              backend=bk, pack=pk, budget="3000", strategy="hermes", crash_at="")

    # ── C) AIRFLOW (yapısal uyumsuzluk + DAG export) ──
    T("C · airflow × audit (DAG export)", "graf", "c_air", GOALS["audit"],
      backend="airflow", pack="audit")

    # ── D) HATA SENARYOLARI (own) ──
    T("D1 · çökme enjeksiyonu", "hata", "d1", GOALS["audit"],
      backend="own", pack="audit", crash_at="tara", budget="3000")
    T("D2 · çökme (farklı düğüm)", "hata", "d2", GOALS["audit"],
      backend="own", pack="audit", crash_at="test")
    T("D3 · çökme + temporal", "hata", "d3", GOALS["audit"],
      backend="temporal", pack="audit", crash_at="tara")

    # ── E) KAYITLI AKIŞI YENİDEN KOŞTURMA ──
    try:
        items = json.loads(_get("/pipelines"))["items"]
    except Exception as e:
        items = []
        print(f"[!] pipeline listesi alınamadı: {e}")
    for bk in (["own"] if hizli else ["own", "temporal", "celery"]):
        cand = next((p for p in items if p["pack"] == "data"), None) or (items[0] if items else None)
        if cand:
            T(f"E · kayıtlı akış tekrar · {bk}", "kayitli", f"e_{bk}", pid=cand["id"],
              backend=bk, crash_at="")

    # ── F) COMPACTION (ajan düğümü + strateji × bütçe) ──
    for strat in (["hermes"] if hizli else ["none", "hermes", "codex", "openclaw"]):
        T(f"F · araç + compaction · {strat}", "compaction", f"f_{strat}",
          "fetch_docs aracıyla MFA dokümantasyonunu ham olarak çek ve özetle",
          backend="own", pack="audit", strategy=strat, budget="1500", crash_at="")

    # ── G) ÇOK TURLU OTURUM (board birikiyor mu?) ──
    T("G1 · tur 1", "coklu", "g1", "auth/login.py'ı oku",
      backend="own", pack="audit", budget="3000", crash_at="")
    T("G2 · tur 2 (aynı oturum)", "coklu", "g1", "şimdi mfa_token desenini tara")
    T("G3 · tur 3 (aynı oturum)", "coklu", "g1", "testleri de koştur ve bulgularla eşleştir")

    print("\n" + "=" * 78)
    print(f"BİTTİ · {len(RESULTS)} test · {round(time.time()-t0)} sn · → {OUT.name}")
    print("=" * 78)


if __name__ == "__main__":
    main()
