#!/usr/bin/env python3
"""
kiyas.py — Dört SAF motora aynı soruları sorup cevabı ÖLÇER.

Kural: hiçbir motora bizim katmanımız eklenmedi. Ne verirlerse o.
Eksiklikler burada kendiliğinden ortaya çıkacak.

Sorular:
  S1  Mutlu yol: doğru sonuç üretiyor mu, ne kadar sürede?
  S2  Bir düğüm GEÇİCİ hata verirse toparlıyor mu? (retry var mı)
  S3  Bir düğüm KALICI hata verirse ardıl düğümlere ne oluyor?
  S4  Worker ÇÖKERSE tamamlanmış iş kurtarılıyor mu?
  S5  "Şu an neredeyiz?" sorusuna cevap verebiliyor mu? (durum görünürlüğü)
  S6  Çalışma anında graf değişebiliyor mu?
  S7  Zamanlama (cron) kutudan geliyor mu?
  S8  Grafı ifade etmek için imzalar bozuluyor mu? (graf sızıntısı)

    .venv/bin/python saf-motorlar/kiyas.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = str(Path(sys.executable))
SONUC: dict = {}


def kos(betik: str, env_ek: dict | None = None, sn: int = 300) -> tuple[str, float]:
    t0 = time.time()
    p = subprocess.run([PY, str(HERE / betik)], capture_output=True, text=True,
                       env={**os.environ, **(env_ek or {})}, timeout=sn)
    return (p.stdout + p.stderr), round(time.time() - t0, 2)


def bas(n):
    print(f"\n{'═' * 92}\n{n}\n{'═' * 92}")


# ═══════════ S1 · mutlu yol ═══════════

def s1():
    bas("S1 · MUTLU YOL — doğru sonuç, ne kadar sürede?")
    print(f"  {'motor':<14}{'sonuç':<10}{'süre':>8}   çıktı")
    print(f"  {'-' * 84}")
    for ad, betik in (("hermes type", "saf_hermes.py"), ("celery", "saf_celery.py"),
                      ("temporal", "saf_temporal.py")):
        out, sn = kos(betik)
        ok = "✓ 4 eşleşme" in out
        satir = next((l.strip() for l in out.splitlines() if "eşleşme" in l), "—")
        print(f"  {ad:<14}{'✓ doğru' if ok else '✗ HATA':<10}{sn:>7}s   {satir[:56]}")
        SONUC.setdefault("s1", {})[ad] = {"ok": ok, "sn": sn}
    print(f"  {'airflow':<14}{'—':<10}{'—':>8}   KOŞTURULAMADI (Airflow kurulu değil)")
    SONUC.setdefault("s1", {})["airflow"] = {"ok": None, "sn": None}


# ═══════════ S2 · geçici hata ═══════════

def s2():
    bas("S2 · GEÇİCİ HATA — 'tara' ilk denemede patlıyor, toparlanıyor mu?")
    for ad, betik in (("hermes type", "saf_hermes.py"), ("celery", "saf_celery.py"),
                      ("temporal", "saf_temporal.py")):
        out, sn = kos(betik, {"SAF_HATA": "tara"})
        toparladi = "✓ 4 eşleşme" in out
        print(f"  {ad:<14}{'✓ TOPARLADI' if toparladi else '✗ toparlayamadı':<18}"
              f"{sn:>7}s")
        if not toparladi:
            neden = [l.strip() for l in out.splitlines()
                     if "→" in l or "✗" in l][:2]
            for n in neden:
                print(f"                   {n[:76]}")
        SONUC.setdefault("s2", {})[ad] = {"toparladi": toparladi, "sn": sn}
    print(f"  {'airflow':<14}{'(retries=2 beyan ediyor, ölçülmedi)':<18}")


# ═══════════ S3 · kalıcı hata ═══════════

def s3():
    bas("S3 · KALICI HATA — 'tara' her denemede patlıyor, ardıllara ne oluyor?")
    for ad, betik in (("hermes type", "saf_hermes.py"), ("celery", "saf_celery.py"),
                      ("temporal", "saf_temporal.py")):
        out, sn = kos(betik, {"SAF_HATA": "tara!"})
        bitti = "eşleşme" in out
        print(f"  {ad:<14}{'✗ akış tamamlanmadı' if not bitti else '?':<24}{sn:>7}s")
        for l in out.splitlines():
            s = l.strip()
            if s.startswith("→") or "HATA" in s or "hatayla bitti" in s:
                print(f"                   {s[:74]}")
        SONUC.setdefault("s3", {})[ad] = {"tamamlandi": bitti, "sn": sn}


# ═══════════ S4 · çökme ═══════════

def s4():
    bas("S4 · ÇÖKME — worker ölürse tamamlanmış iş kurtarılıyor mu?")
    p = subprocess.run([PY, str(HERE / "saf_hermes.py"), "--cokme", "tara"],
                       capture_output=True, text=True)
    print("  hermes type:")
    for l in (p.stdout + p.stderr).splitlines():
        if "ÇÖKME" in l or "→" in l or "Kalıcı durum" in l:
            print(f"     {l.strip()[:82]}")
    print("\n  celery   : acks_late=True → mesaj yeniden teslim edilir, ama task")
    print("             BAŞTAN koşar (checkpoint kavramı yok). Tamamlanan")
    print("             adımların sonucu backend'de durur, chain yeniden kurulmaz.")
    print("  temporal : replay → TAMAMLANMIŞ activity'ler ATLANIR, workflow")
    print("             kaldığı yerden devam eder. Dört motor içinde tek.")
    print("  airflow  : XCom metadata DB'de kalıcı → tamamlanan task'lar tekrar")
    print("             koşmaz; ama task İÇİNDE checkpoint yok, düğüm baştan başlar.")
    SONUC["s4"] = {"hermes": "hiçbir şey kurtarılmıyor", "celery": "task baştan",
                   "temporal": "biten activity atlanır", "airflow": "task baştan"}


# ═══════════ S5–S8 · yapısal ═══════════

def s5_s8():
    bas("S5–S8 · YAPISAL — durum görünürlüğü, dinamiklik, zamanlama, graf sızıntısı")
    t = [
        ("S5 · 'şu an neredeyiz?'",
         "yok — bellekte, süreç ölünce gider",
         "ZAYIF — chord bekliyor ama hangi adımda bilinmiyor",
         "VAR — event history, kutudan",
         "VAR — metadata DB + operatör UI"),
        ("S6 · çalışma anında graf değişimi",
         "kodu değiştir (yeniden başlat)",
         "canvas apply_async'te SABİTLENİR",
         "workflow kodu SABİT (determinizm şartı)",
         "parse zamanında SABİT"),
        ("S7 · zamanlama (cron)",
         "YOK",
         "Celery Beat (backfill yok)",
         "Temporal Schedules (+backfill)",
         "EN GÜÇLÜ (+catchup)"),
        ("S8 · graf ifadesi",
         "fonksiyon çağrı sırası",
         "canvas — İMZALARA SIZIYOR",
         "düz Python (gather/await) — en okunur",
         "dosyada `>>` — okunur ama DONMUŞ"),
    ]
    print(f"  {'soru':<34}{'hermes type':<26}{'celery':<34}")
    print(f"  {'-' * 90}")
    for s, h, c, tp, a in t:
        print(f"  {s:<34}{h:<26}{c:<34}")
    print()
    print(f"  {'soru':<34}{'temporal':<34}{'airflow':<28}")
    print(f"  {'-' * 90}")
    for s, h, c, tp, a in t:
        print(f"  {s:<34}{tp:<34}{a:<28}")
    SONUC["s5_s8"] = [dict(zip(("soru", "hermes", "celery", "temporal", "airflow"), r))
                      for r in t]


def main():
    print("═" * 92)
    print("SAF MOTOR KIYASI — hiçbirine bizim katmanımız EKLENMEDİ")
    print("  aynı iş: cek → tara ─┐")
    print("                        ├─► esle ─► rapor")
    print("           test ───────┘")
    print("═" * 92)
    s1(); s2(); s3(); s4(); s5_s8()
    (HERE / "kiyas_sonuc.json").write_text(
        json.dumps(SONUC, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + "═" * 92)
    print("  sonuç: kiyas_sonuc.json")
    print("═" * 92)


if __name__ == "__main__":
    main()
