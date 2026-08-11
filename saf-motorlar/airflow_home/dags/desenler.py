#!/usr/bin/env python3
"""
desenler.py — Dört motoru AYIRT EDEN workflow şekilleri.

Tek bir elmas graf motorları ayırt etmiyor: dördü de yapıyor. Gerçek fark
şu dört desende çıkıyor — çünkü her biri motorun workflow'a BAKIŞINI zorluyor.

  D1 · ELMAS      paralel dal + join           → temel, hepsi yapar
  D2 · ZİNCİR     10 adım sıralı               → ara durum, retry maliyeti
  D3 · KOŞULLU    çalışma anında dal seçimi    → "if" motorda nasıl ifade edilir
  D4 · DİNAMİK    N çalışma anında belli       → fan-out sayısı önceden bilinmiyor

D3 ve D4 kritik, çünkü ikisi de "graf ne zaman belli olur" sorusunu soruyor:
  • Temporal : graf = kod → `if` ve `for` normal Python
  • Airflow  : graf = dosya → BranchPythonOperator / .expand() gerekiyor
  • Celery   : graf = canvas → gönderim anında sabitlenir, dal için ek task
  • saf py   : graf = çağrı sırası → hepsi bedava ama hiçbir garanti yok

Bu dosya SADECE iş tanımlarını verir; her motor kendi yöntemiyle koşturacak.
"""
from __future__ import annotations

import hashlib
import os
import re

_DENEME: dict[str, int] = {}


def _hata_kontrol(adim: str):
    ayar = os.environ.get("SAF_HATA", "")
    if not ayar:
        return
    kalici = ayar.endswith("!")
    if ayar.rstrip("!") != adim:
        return
    _DENEME[adim] = _DENEME.get(adim, 0) + 1
    if kalici or _DENEME[adim] == 1:
        raise RuntimeError(f"{'kalıcı' if kalici else 'geçici'} hata: {adim} "
                           f"(deneme {_DENEME[adim]})")


# ═══════════════ D2 · ZİNCİR — 10 adım sıralı ═══════════════
# Neyi ölçer: uzun akışta ara durum tutuluyor mu, 7. adım patlarsa
# ilk 6 tekrar koşuyor mu.

def zincir_adim(n: int, onceki: dict | None = None) -> dict:
    """n. adım: öncekinin sayacını bir artırır. Zincir kopukluğu hemen belli olur."""
    _hata_kontrol(f"z{n}")
    gelen = (onceki or {}).get("sayac", 0)
    return {"adim": n, "sayac": gelen + 1, "iz": (onceki or {}).get("iz", "") + f"{n}·"}


ZINCIR_UZUNLUK = 10


def zincir_dogrula(son: dict) -> tuple[bool, str]:
    if not isinstance(son, dict):
        return False, "sonuç yok"
    if son.get("sayac") != ZINCIR_UZUNLUK:
        return False, f"sayac={son.get('sayac')} (beklenen {ZINCIR_UZUNLUK}) — zincir KOPUK"
    return True, f"sayac={son['sayac']} · iz={son.get('iz','')[:40]}"


# ═══════════════ D3 · KOŞULLU — çalışma anında dal ═══════════════
# Neyi ölçer: "tarama bulgu bulduysa raporla, bulmadıysa atla" gibi bir kararı
# motor NASIL ifade ediyor ve seçilmeyen dala NE oluyor.

def tarama(esik: int = 3) -> dict:
    """Bulgu sayısı üretir. esik'e göre alt dal değişecek."""
    _hata_kontrol("tarama")
    bulgu = 4                       # deterministik: her zaman 4
    return {"bulgu": bulgu, "esik": esik, "asti": bulgu >= esik}


def dal_raporla(t: dict) -> dict:
    """AŞARSA bu dal koşar."""
    _hata_kontrol("dal_raporla")
    return {"dal": "raporla", "mesaj": f"{t['bulgu']} bulgu — rapor üretildi"}


def dal_atla(t: dict) -> dict:
    """AŞMAZSA bu dal koşar."""
    _hata_kontrol("dal_atla")
    return {"dal": "atla", "mesaj": f"{t['bulgu']} bulgu — eşik altı, atlandı"}


def kosullu_dogrula(son: dict, beklenen_dal: str = "raporla") -> tuple[bool, str]:
    if not isinstance(son, dict) or "dal" not in son:
        return False, f"dal sonucu yok: {son}"
    if son["dal"] != beklenen_dal:
        return False, f"YANLIŞ dal koştu: {son['dal']} (beklenen {beklenen_dal})"
    return True, son["mesaj"]


# ═══════════════ D4 · DİNAMİK — N çalışma anında belli ═══════════════
# Neyi ölçer: fan-out sayısı önceden bilinmiyorsa motor ne yapıyor.
# Bu, "ajan çalışma anında iş üretebilir mi" sorusunun motor tarafındaki hâli.

def dosyalari_bul(kok: str = "auth") -> dict:
    """Kaç dosya olduğu ÇALIŞMA ANINDA belli oluyor (burada 5)."""
    _hata_kontrol("dosyalari_bul")
    return {"dosyalar": [f"{kok}/mod_{i}.py" for i in range(5)]}


def dosya_isle(path: str) -> dict:
    """Her dosya için bir kez koşacak — kaç kez, önceden bilinmiyor."""
    _hata_kontrol("dosya_isle")
    h = hashlib.sha1(path.encode()).hexdigest()[:8]
    return {"path": path, "sha": h, "boyut": len(path) * 100}


def topla(sonuclar: list) -> dict:
    """Fan-in: hepsini birleştir."""
    _hata_kontrol("topla")
    temiz = [s for s in sonuclar if isinstance(s, dict) and "sha" in s]
    return {"adet": len(temiz), "toplam_boyut": sum(s["boyut"] for s in temiz),
            "shalar": sorted(s["sha"] for s in temiz)}


DINAMIK_BEKLENEN = 5


def dinamik_dogrula(son: dict) -> tuple[bool, str]:
    if not isinstance(son, dict) or "adet" not in son:
        return False, f"toplama sonucu yok: {son}"
    if son["adet"] != DINAMIK_BEKLENEN:
        return False, f"adet={son['adet']} (beklenen {DINAMIK_BEKLENEN}) — fan-out EKSİK"
    return True, f"{son['adet']} dosya · toplam {son['toplam_boyut']}B"


if __name__ == "__main__":
    # Referans: saf Python'da hepsi
    s = None
    for i in range(1, ZINCIR_UZUNLUK + 1):
        s = zincir_adim(i, s)
    print("D2 zincir  :", zincir_dogrula(s))

    t = tarama()
    d = dal_raporla(t) if t["asti"] else dal_atla(t)
    print("D3 koşullu :", kosullu_dogrula(d))

    f = dosyalari_bul()
    r = topla([dosya_isle(p) for p in f["dosyalar"]])
    print("D4 dinamik :", dinamik_dogrula(r))
