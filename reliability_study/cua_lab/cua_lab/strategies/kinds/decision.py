"""
KATEGORİ: KARAR — hangi eylem, hangi eşik.

ORTAK OLAN: bu kategori "ne zaman dur" sorusunu HİÇ sormuyor. İkisi de
döngünün kendisine değil, döngüyü yöneten KARARA bakıyor.

AYRIŞTIKLARI TEK NOKTA: hangi karara.

    voi-allocation    bu adimda hangi eylem secilmeli (kosum ICINDE)
    improvement-loop  bu esik kac olmali (kosumun DISINDA, sonraki kosum icin)

Ortak riski de aynı: ikisi de tek başına koşumu korumuyor. Sert bir tavanla
birlikte kullanılmaları gerekiyor.
"""

from __future__ import annotations

from ...events import CONTINUE, Verdict
from ..base import BaseStrategy, StopReport


class DecisionStrategy(BaseStrategy):
    family = "src"
    kind = "decision"

    def on_run_start(self, ctx) -> None:
        self._kayit: list[dict] = []
        self._setup(ctx)

    def _setup(self, ctx) -> None:
        pass

    def observe(self, ctx) -> None:
        """Her adımda kayıt tut — iki alt sınıf da bunu yapıyor."""

    def before_step(self, ctx) -> Verdict:
        self.observe(ctx)
        return self.decide(ctx)

    def decide(self, ctx) -> Verdict:
        return CONTINUE

    def snapshot(self) -> dict:
        return {"kayit": len(self._kayit)}

    def on_stop(self, reason: str, ctx) -> StopReport | None:
        return self.summary(reason, ctx)

    def summary(self, reason: str, ctx) -> StopReport | None:
        return None
