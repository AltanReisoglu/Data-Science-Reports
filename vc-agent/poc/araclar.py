"""Ajanların kullandığı tool'lar.

Hepsi kasıtlı olarak **deterministik ve ağsız**: POC'un çıktısı tekrarlanabilir olsun,
ölçülen fark desenden gelsin, tool'un keyfinden değil.

Kurs (10. Tools) `FunctionTool` sarmalayıcısını gösteriyor; AssistantAgent düz
fonksiyonu da kabul ediyor. Burada ikisi de var: `FunctionTool` ile açık kayıt
(`arxiv_ara`) ve düz fonksiyon (`istatistik_hesapla`).
"""

from __future__ import annotations

import statistics
from typing import Any

from autogen_core.tools import FunctionTool

# Sahte "veri ambarı" — gerçek bir veri kaynağının yerine geçiyor.
_VERI: dict[str, list[float]] = {
    "gelir": [120.5, 133.0, 128.75, 141.2, 155.9, 149.3, 162.0],
    "hata_orani": [0.031, 0.028, 0.035, 0.022, 0.019, 0.024, 0.017],
    "gecikme_ms": [240, 231, 255, 228, 219, 235, 210],
}

_MAKALELER = [
    {
        "baslik": "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
        "yil": 2023,
        "arxiv": "2308.08155",
    },
    {
        "baslik": "Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks",
        "yil": 2024,
        "arxiv": "2411.04468",
    },
    {
        "baslik": "Why Do Multi-Agent LLM Systems Fail?",
        "yil": 2025,
        "arxiv": "2503.13657",
    },
]


def veri_getir(metrik: str) -> list[float]:
    """Adı verilen metriğin ham serisini döndürür.

    Args:
        metrik: 'gelir', 'hata_orani' ya da 'gecikme_ms'.
    """
    if metrik not in _VERI:
        raise ValueError(f"bilinmeyen metrik: {metrik}. Seçenekler: {list(_VERI)}")
    return _VERI[metrik]


def istatistik_hesapla(seri: list[float]) -> dict[str, float]:
    """Bir sayı serisinin özet istatistiklerini hesaplar.

    Args:
        seri: sayı listesi.
    """
    if not seri:
        raise ValueError("seri boş olamaz")
    ilk, son = seri[0], seri[-1]
    return {
        "n": len(seri),
        "ortalama": round(statistics.fmean(seri), 4),
        "medyan": round(statistics.median(seri), 4),
        "std": round(statistics.pstdev(seri), 4) if len(seri) > 1 else 0.0,
        "min": min(seri),
        "max": max(seri),
        "degisim_yuzde": round((son - ilk) / ilk * 100, 2) if ilk else 0.0,
    }


def arxiv_ara(sorgu: str, adet: int = 3) -> list[dict[str, Any]]:
    """Yerel makale kataloğunda arama yapar (ağa çıkmaz).

    Args:
        sorgu: aranacak ifade.
        adet: en fazla kaç sonuç döndürüleceği.
    """
    s = sorgu.lower()
    bulunan = [m for m in _MAKALELER if s in m["baslik"].lower()] or _MAKALELER
    return bulunan[:adet]


# FunctionTool ile açık kayıt — kursun 10.1'deki deseni.
arxiv_araci = FunctionTool(arxiv_ara, description="Makale kataloğunda arama yapar")

TUM_ARACLAR = [veri_getir, istatistik_hesapla, arxiv_araci]
