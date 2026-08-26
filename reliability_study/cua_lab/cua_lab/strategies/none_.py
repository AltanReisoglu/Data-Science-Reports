"""
Taban çizgisi — hiçbir kontrol yok.

Bu bir "strateji yokluğu" değil, ÖLÇÜM ARACI. Anthropic'in referans
computer-use döngüsü (`computer-use-demo/loop.py`) tam olarak böyle
davranıyor: `while True`, tek çıkış modelin araç çağırmayı bırakması.
Tur sayacı yok, döngü tespiti yok, bütçe yok.

Karşılaştırma sütunu bu. "Kontrol koymanın faydası ne" sorusunun cevabı
bu sütunla diğerleri arasındaki farktır.
"""

from __future__ import annotations

from .base import BaseStrategy, register


@register
class NoStrategy(BaseStrategy):
    id = "none"
    title = "Kontrol yok (taban cizgisi)"
    source = "anthropics/claude-quickstarts computer-use-demo/loop.py"
    mentality = "Referans dongu: model durana kadar don"
    priority = 0
    why = (
        "Kontrolun faydasini olcebilmek icin kontrolsuz bir sutun gerekiyor. Anthropic'in referans computer-use dongusu tam olarak boyle: while True, tek cikis modelin arac cagirmayi birakmasi.")
    action = "Hicbir sey — tavan, dedektor, butce yok"
    blind_spot = (
        "Her seyi kacirir. Olcum araci, strateji degil.")
    family = "-"
