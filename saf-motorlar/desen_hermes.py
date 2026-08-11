#!/usr/bin/env python3
"""
desen_hermes.py — Üç deseni TABAN ÇİZGİSİNDE (çıplak Python) kur.

BAKIŞ: workflow diye bir şey YOK. Graf = çağrı sırası. `if` ve `for`
dilin kendisi. Bu yüzden üç desen de en kısa burada yazılır — ve
hiçbir garanti yoktur.

Amaç: "motor ne kadar kod ekliyor / karşılığında ne veriyor" sorusuna
sıfır noktası sağlamak.

    .venv/bin/python saf-motorlar/desen_hermes.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import desenler as D  # noqa: E402


def d2_zincir():
    s = None
    for i in range(1, D.ZINCIR_UZUNLUK + 1):
        s = D.zincir_adim(i, s)
    return s


def d3_kosullu():
    t = D.tarama()
    return D.dal_raporla(t) if t["asti"] else D.dal_atla(t)


def d4_dinamik():
    f = D.dosyalari_bul()
    return D.topla([D.dosya_isle(p) for p in f["dosyalar"]])


def main():
    print("═" * 78)
    print("TABAN ÇİZGİSİ (çıplak Python) — workflow kavramı YOK, graf = çağrı sırası")
    print("═" * 78)
    for ad, fn, dog in (("D2 zincir ", d2_zincir, D.zincir_dogrula),
                        ("D3 koşullu", d3_kosullu, D.kosullu_dogrula),
                        ("D4 dinamik", d4_dinamik, D.dinamik_dogrula)):
        t0 = time.time()
        try:
            ok, not_ = dog(fn())
        except Exception as e:
            ok, not_ = False, f"{type(e).__name__}: {str(e)[:70]}"
        print(f"  {ad}  {'✓' if ok else '✗'} {not_[:58]:<60} {round(time.time()-t0,3)}s")


if __name__ == "__main__":
    main()
