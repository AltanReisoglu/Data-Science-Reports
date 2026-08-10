#!/usr/bin/env python3
"""
test_compaction_matrix.py — 6 strateji × 5 bütçe = 30 kombinasyon, aynı iz üstünde.

Ölçtüğü şey sadece yüzde değil, asıl önemli olan **ne kaldı**:
  • tetiklendi mi
  • mesaj sayısı korundu mu (silme var mı)
  • tool_call ↔ tool_result çifti bozuldu mu   ← kırılırsa API isteği reddedilir
  • kritik bilgi (BUG satırı) hayatta mı       ← ajan işini yapabilir mi

    .venv/bin/python demo-brain-agent/test_compaction_matrix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compaction as CP        # noqa: E402
import gosterim as G           # noqa: E402

BUTCELER = [200, 400, 1_000, 3_000, 30_000]
KRITIK = "mfa_token is None"          # asıl bug satırının imzası
SONUC: list = []


def cift_butunlugu(msgs) -> tuple[bool, str]:
    """Her tool_call'un bir tool sonucu var mı? (sıra bazlı basit kontrol)"""
    vs = CP.views(msgs)
    bekleyen = 0
    for v in vs:
        if v.role == "assistant" and v.tool_calls:
            bekleyen += len(v.tool_calls)
        elif v.role == "tool":
            bekleyen -= 1
    if bekleyen > 0:
        return False, f"{bekleyen} tool_call sonuçsuz (yetim)"
    if bekleyen < 0:
        return False, f"{-bekleyen} tool sonucu çağrısız"
    return True, ""


def main():
    print("=" * 100)
    print("COMPACTION MATRİSİ — 6 strateji × 5 bütçe, aynı iz")
    iz = G.trace()
    ham = CP.total_tokens(iz)
    print(f"  iz: {len(iz)} mesaj · {ham:,} token · kritik bilgi imzası: {KRITIK!r}")
    print("=" * 100)
    print(f"\n{'strateji':<13} {'bütçe':>7} {'sonuç':>16} {'kazanç':>8} "
          f"{'mesaj':>9} {'çift':>6} {'kritik bilgi':>13}")
    print("-" * 100)

    for s in CP.STRATEGIES:
        for b in BUTCELER:
            r = CP.compact(s, G.trace(), budget=b)
            ok_cift, cift_not = cift_butunlugu(r.messages)
            metin = "\n".join(CP._View(m).content for m in r.messages)
            kritik = KRITIK in metin
            SONUC.append({
                "strateji": s, "butce": b, "once": r.before, "sonra": r.after,
                "pct": round(r.pct, 1), "tetik": r.triggered,
                "mesaj_once": len(iz), "mesaj_sonra": len(r.messages),
                "cift_ok": ok_cift, "cift_not": cift_not, "kritik_bilgi": kritik,
            })
            print(f"{s:<13} {b:>7,} {r.before:>7,}→{r.after:<8,} "
                  f"{('%' + format(r.pct, '.1f')) if r.triggered else '  —':>8} "
                  f"{len(iz):>4}→{len(r.messages):<4} "
                  f"{'✓' if ok_cift else '✗':>6} {'✓ var' if kritik else '✗ kayıp':>13}")
        print()

    # ── analiz ──
    print("=" * 100)
    kirik = [x for x in SONUC if not x["cift_ok"]]
    print(f"  ÇİFT BÜTÜNLÜĞÜ  : {len(SONUC)-len(kirik)}/{len(SONUC)} sağlam"
          + (f"   ✗ KIRILANLAR: {[(x['strateji'], x['butce'], x['cift_not']) for x in kirik]}"
             if kirik else "   ✓ hiçbiri kırılmadı"))

    tet = [x for x in SONUC if x["tetik"]]
    kayip = [x for x in tet if not x["kritik_bilgi"]]
    print(f"  KRİTİK BİLGİ    : tetiklenen {len(tet)} koşudan {len(tet)-len(kayip)}'inde korundu")
    if kayip:
        print("     kaybedenler (strateji@bütçe):",
              ", ".join(f"{x['strateji']}@{x['butce']}" for x in kayip))

    silen = [x for x in SONUC if x["mesaj_sonra"] < x["mesaj_once"]]
    print(f"  MESAJ SİLEN     : {len(silen)}/{len(SONUC)} koşu "
          f"({sorted({x['strateji'] for x in silen})})")

    print("\n  STRATEJİ PROFİLİ (tetiklendiği koşularda ortalama kazanç):")
    for s in CP.STRATEGIES:
        v = [x for x in SONUC if x["strateji"] == s and x["tetik"]]
        if not v:
            print(f"    {s:<13} hiç tetiklenmedi")
            continue
        ort = sum(x["pct"] for x in v) / len(v)
        km = sum(1 for x in v if not x["kritik_bilgi"])
        korur = all(x["mesaj_sonra"] == x["mesaj_once"] for x in v)
        print(f"    {s:<13} {len(v)}/5 bütçede tetiklendi · ort %{ort:.1f} · "
              f"mesaj {'KORUR' if korur else 'BİRLEŞTİRİR'} · "
              f"kritik bilgi {len(v)-km}/{len(v)} korundu")
    print("=" * 100)

    (HERE / "test_compaction_sonuc.json").write_text(
        json.dumps(SONUC, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
