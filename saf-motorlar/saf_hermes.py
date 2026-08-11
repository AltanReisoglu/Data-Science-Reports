#!/usr/bin/env python3
"""
saf_hermes.py — TABAN ÇİZGİSİ: hiçbir altyapı olmadan aynı iş.

Board yok, kuyruk yok, kalıcılık yok. Çıplak Python. Diğer üç motorun
"neyi bedavaya verdiğini" ölçmek için sıfır noktası.

DÜZELTME — bu dosya Hermes-Agent'ı TEMSİL ETMİYOR:
Hermes-Agent'ın kendi SQLite state store'u var (`hermes_state.py`, WAL + FTS5,
`sessions`/`messages` tabloları) ve `hermes --resume` ile kaldığı oturumdan
devam ediyor. Ama kalıcılık birimi KONUŞMA TURU — `_persist_session` çağrıları
tool döngüsünden SONRA ya da terminal çıkışlarda. Yani turda 5 tool çağrılıp
3.'te çökülürse üçü de kaybolur, tur baştan koşar.
(Ayrıca `tools/checkpoint_manager.py` var ama o shadow-git ile dosya sistemi
geri alma mekanizması — iş akışı devamı değil.)

Buradaki "hermes type" adı bizim KENDİ motorumuzun takma adı; aşağıdaki kod
ise onun board'suz hâli, yani sıfır noktası.

NE YAPAR : bağımlılığa göre sıralar, çıktıyı bellekte taşır
NE YAPMAZ: retry · kalıcı durum · çökme kurtarma · paralellik · zamanlama
           · at-most-once · iptal zinciri

    .venv/bin/python saf-motorlar/saf_hermes.py [--cokme ADIM]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isakisi as J  # noqa: E402


class Cokme(Exception):
    """Worker'ın adım ortasında ölmesi."""


def kos(cokme_adim: str | None = None, log=print) -> dict:
    """Grafı topolojik sırayla koştur. Bağımlılık = fonksiyon argümanı."""
    s: dict = {}
    for ad in J.SIRA:
        t0 = time.time()
        if ad == "cek":
            s[ad] = J.cek()
        elif ad == "test":
            s[ad] = J.test()
        elif ad == "tara":
            s[ad] = J.tara(s["cek"])
        elif ad == "esle":
            s[ad] = J.esle(s["tara"], s["test"])
        elif ad == "rapor":
            s[ad] = J.rapor(s["esle"], s["tara"], s["test"])
        log(f"    ✓ {ad:<7} {round((time.time()-t0)*1000, 1)} ms")
        if cokme_adim == ad:
            # ── ÇÖKME: süreç ölüyor. Bellekteki her şey KAYBOLUYOR. ──
            raise Cokme(f"{ad} sonrası worker öldü — bellekte {len(s)} sonuç vardı")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cokme", help="bu adımdan sonra çök")
    a = ap.parse_args()

    print("═" * 74)
    print("SAF HERMES TYPE — çıplak Python, hiçbir altyapı yok")
    print("═" * 74)
    t0 = time.time()
    try:
        s = kos(a.cokme)
    except Cokme as e:
        print(f"    ✗ ÇÖKME: {e}")
        print(f"\n  → Süreç yeniden başlarsa: HİÇBİR ŞEY kurtarılamaz.")
        print(f"    Kalıcı durum yok; tamamlanan adımlar da BAŞTAN koşar.")
        return
    except Exception as e:
        print(f"    ✗ HATA: {type(e).__name__}: {e}")
        print(f"\n  → Retry YOK. Akış burada durur, kalan adımlar hiç koşmaz.")
        print(f"    Hangi adımların bittiği kaydedilmediği için kurtarma da yok.")
        return
    ok, not_ = J.dogrula(s)
    print(f"\n  {'✓' if ok else '✗'} {not_} · {round(time.time()-t0, 3)} sn")


if __name__ == "__main__":
    main()
