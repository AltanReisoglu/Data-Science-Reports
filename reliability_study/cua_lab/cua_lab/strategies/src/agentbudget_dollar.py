"""
AgentBudget — doların kendisini say.  KATEGORİ: SAYAÇ · SEVİYE 3

Kaynak: *agentbudget_framework*

ZİHNİYET: Token bir VEKİL ölçüdür. İki model arasında on beş kat fiyat farkı
olabilir; aynı token sayısı çok farklı faturalar üretir. Asıl birimi say.

Arize ile AYNI sayacı kullanıyor. Farkı iki düğmede:

  hard=False       Sert kesmiyor: DEGRADE dondurup modelden nihai cevabi
                   istiyor. Kesilen ajanin isi copa gitmiyor.
  reserve=0.15     NIHAI CEVAP PAYI — sert limit %15 once tetikleniyor ki
                   ajanin toparlayacak butcesi kalsin.

Üçüncü farkı `extra_check`'te: ZAMAN PENCERELİ patlama tespiti. "On çağrı"
değil, "bir dakikada on çağrı" — yavaş ama kronik bir döngü sayı eşiğine hiç
ulaşmadan saatlerce para yakar.
"""

from __future__ import annotations

import time

from ...events import CONTINUE, Action, Verdict
from ..base import register
from ..kinds import BudgetStrategy


@register
class AgentBudgetDollar(BudgetStrategy):
    id = "agentbudget-dollar"
    title = "AgentBudget: dolar tavani + nihai cevap payi"
    source = "agentbudget_framework"
    mentality = "Tokeni degil dolari say — ve toparlanacak pay birak"
    priority = 4
    why = (
        "Token bir VEKIL olcudur; iki model arasinda on bes kat fiyat farki olabilir. Bu katman gercek faturayi sayar ve kesilen ajanin isini copa atmaz.")
    action = "Dolar tavani + %15 NIHAI CEVAP PAYI; zaman pencereli patlama tespiti"
    blind_spot = (
        "Fiyat tablosu bakim gerektirir; model degisince yanlis sayar.")

    hard = False
    reserve = 0.15
    primary_axis = None
    warn_axes = ()

    burst_window = 10.0   # saniye
    burst_calls = 25

    def _setup_burst(self):
        self._calls: list[float] = []

    def on_run_start(self, ctx) -> None:
        super().on_run_start(ctx)
        self._setup_burst()

    def extra_check(self, ctx) -> Verdict:
        simdi = time.monotonic()
        self._calls.append(simdi)
        self._calls = [t for t in self._calls if simdi - t <= self.burst_window]
        if len(self._calls) >= self.burst_calls:
            return Verdict(Action.DEGRADE, "budget_burst",
                           {"pencere_sn": self.burst_window,
                            "cagri": len(self._calls), "terminated_by": "burst"})
        return CONTINUE

    def why_text(self, reason: str) -> str:
        return (f"{reason} — sert limitin %{int(self.reserve * 100)} oncesinde "
                f"durduruldu ki nihai cevap uretilebilsin")
