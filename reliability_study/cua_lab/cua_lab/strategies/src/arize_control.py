"""
Arize — sert durdurma, modele sorulmadan.  KATEGORİ: SAYAÇ · SEVİYE 1

Kaynak: Arize AI — *What Is An Agent Control Loop?*

ZİHNİYET: Durma kararını modelin yargısından TAMAMEN koparmak.
"Her döngüye modelin yargısına bağlı olmayan sert bir durdurma gerekiyor,
çünkü model, işin bitip bitmediği konusunda yanılması en muhtemel bileşendir."

Sayaç mantığı `kinds/budget.py`'de — bu dosyada yalnızca ARİZE'İN FARKI var:

  hard=True          Sert keser. Lutuf turu yok, nihai cevap payi yok.
  primary_axis       Adim limiti BIRINCIL — tek tartismasiz olcu.
  warn_axes=()       Ara uyari YOK. Bu zihniyet uyarmaz, keser.

İkinci ve az fark edilen katkısı `axis_reason`'da: adım ekseni `max_steps`
diye adlandırılıyor, `budget_steps` diye değil. Kozmetik değil — "%70 başarı"
diyen bir sistemde kalan %30'un ne olduğu her şeyi değiştirir.
"""

from __future__ import annotations

from ..base import register
from ..kinds import BudgetStrategy


@register
class ArizeControl(BudgetStrategy):
    id = "arize-control"
    title = "Arize: sert durdurma, modele sorulmadan"
    source = "Arize — What Is An Agent Control Loop?"
    mentality = "Modele sorma, say — adim limiti birincil"
    priority = 1
    why = (
        "Her ajan sisteminin TEMEL guvenlik katmani. Sonsuz donguyu, asiri maliyeti ve uzun beklemeyi tek basina sinirlar. Baska hicbir sey uygulamayacaksan bunu uygula.")
    action = "Maksimum adim/sure/token/maliyet asilinca SERT durdurma; lutuf turu yok"
    blind_spot = (
        "Ne oldugunu SOYLEMEZ — sadece 'cok oldu' der. Teshis icin bir pencere dedektoruyle birlikte kullan.")

    hard = True
    reserve = 0.0
    primary_axis = "steps"
    warn_axes = ()

    def why_text(self, reason: str) -> str:
        return f"sert durdurma: {reason} (modelin gorusune sorulmadi)"
