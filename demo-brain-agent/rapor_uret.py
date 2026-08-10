#!/usr/bin/env python3
"""
rapor_uret.py — test_sonuclari.json'dan markdown test raporu üretir.
    .venv/bin/python demo-brain-agent/rapor_uret.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "test_sonuclari.json"
OUT = HERE.parent / "report" / "pipeline-test-raporu.md"

KAT = {
    "sohbet": "A · Sohbet yolu ayrımı",
    "tek_tool": "A · Sohbet yolu ayrımı",
    "graf": "B · Graf kurma (backend × paket)",
    "hata": "D · Hata senaryoları (çökme / retry)",
    "kayitli": "E · Kayıtlı akışı yeniden koşturma",
    "compaction": "F · Tool-trace compaction (araç + strateji)",
    "coklu": "G · Çok turlu oturum (board birikimi)",
}


def ok(r) -> str:
    if r.get("hata"):
        return "✗"
    y = r.get("yol")
    if y in ("graf", "kayitli"):
        return "✓" if r.get("tamamlanan", 0) > 0 and r.get("tamamlanan") == r.get("dugum") else "⚠"
    return "✓"


def main():
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    L = []
    A = L.append

    A("# Pipeline Test Raporu — sohbet ajanı, uçtan uca")
    A("")
    A("> Sohbet ajanının (`chat_server.py`) **gerçek HTTP uçları** üzerinden, kullanıcı gibi")
    A("> sürülerek yapılan sistemli test. Her senaryoda SSE olay akışı toplanıp ölçüldü.")
    A(f"> Toplam **{len(rows)} senaryo**. Harness: `demo-brain-agent/test_matrix.py`.")
    A("")

    # --- özet tablo ---
    tam = sum(1 for r in rows if ok(r) == "✓")
    kis = sum(1 for r in rows if ok(r) == "⚠")
    hat = sum(1 for r in rows if ok(r) == "✗")
    sure = round(sum(r.get("sn", 0) for r in rows))
    A(f"**Sonuç:** {tam} ✓ · {kis} ⚠ · {hat} ✗ · toplam {sure} sn")
    A("")

    # --- kategori kategori ---
    seen = []
    for r in rows:
        k = KAT.get(r["kategori"], r["kategori"])
        if k not in seen:
            seen.append(k)

    for k in seen:
        A(f"## {k}")
        A("")
        grp = [r for r in rows if KAT.get(r["kategori"], r["kategori"]) == k]
        A("| | Senaryo | İstek | Yol | Paket | Düğüm | Tamamlanan | Süre |")
        A("|---|---|---|---|---|---:|---:|---:|")
        for r in grp:
            A(f"| {ok(r)} | {r['ad']} | `{r['istek'][:52]}` | {r.get('yol','?')} | "
              f"{r.get('pack') or '—'} | {r.get('dugum',0)} | {r.get('tamamlanan',0)} | "
              f"{r.get('sn',0)} sn |")
        A("")
        # detay: hata/çökme/compaction olanlar
        for r in grp:
            det = []
            if r.get("hata"):
                det.append(f"**hata:** `{r['hata'][:160]}`")
            if r.get("cokme"):
                det.append(f"çökme {r['cokme']} → kurtarma {r['kurtarma']}")
            if r.get("retry"):
                det.append(f"retry {r['retry']}")
            if r.get("compaction"):
                det.append(f"compaction olayı {r['compaction']}")
            if r.get("indirgeme"):
                det.append(f"deterministik indirgeme {r['indirgeme']}")
            if r.get("fn") or r.get("ajan"):
                det.append(f"{r.get('fn',0)} fn + {r.get('ajan',0)} ajan düğümü")
            if det:
                A(f"- **{r['ad']}** — " + " · ".join(det))
        A("")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"→ {OUT}")
    print(f"   {tam} ✓ · {kis} ⚠ · {hat} ✗")


if __name__ == "__main__":
    main()
