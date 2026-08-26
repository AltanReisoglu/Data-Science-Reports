"""
KATEGORİ: ŞEKİL — döngüye izin verme.

ORTAK OLAN: tespit YOK. Bu kategori döngüyü yakalamaya çalışmıyor; oluşmasını
engelleyen bir yapı kuruyor. İkisi de koşum başlamadan ya da her geçişte
YAPISAL bir kural uyguluyor.

AYRIŞTIKLARI TEK NOKTA: kuralı ne zaman uyguluyorlar.

    modexa-statemachine  her adimda — izinli gecis tablosu
    autogen-static       kosumdan ONCE — grafta cikissiz cevrim var mi

Ortak bedeli de aynı: mevcut bir ajana SONRADAN eklenemiyor.
"""

from __future__ import annotations

from ...events import CONTINUE, Action, Verdict
from ..base import BaseStrategy, StopReport


class ShapeStrategy(BaseStrategy):
    family = "src"
    kind = "shape"
    terminal: str = "DEGRADED"

    def on_run_start(self, ctx) -> None:
        self._ihlal: list[str] = []
        self._setup(ctx)
        v = self.validate(ctx)
        if v.triggered:
            # Koşum başlamadan yakalandı. `autogen-static`in bütün iddiası bu:
            # çalışma zamanında sıfır maliyet, sıfır yanlış pozitif — çünkü
            # çalışma zamanı hiç gelmedi.
            ctx.extra["shape_precheck"] = v.reason

    def _setup(self, ctx) -> None:
        pass

    def validate(self, ctx) -> Verdict:
        """Koşum öncesi yapısal doğrulama."""
        return CONTINUE

    def before_step(self, ctx) -> Verdict:
        if (r := ctx.extra.get("shape_precheck")):
            ctx.extra.pop("shape_precheck")
            return Verdict(Action.STOP, r,
                           {"terminal": self.terminal, "asama": "kosum_oncesi"})
        return CONTINUE

    def on_stop(self, reason: str, ctx) -> StopReport | None:
        if reason not in self.reasons():
            return None
        return StopReport(
            reason=reason,
            why=self.why_text(reason, ctx),
            tried=f"{ctx.step} adim; ihlaller: {', '.join(self._ihlal[-4:]) or '-'}",
            found="yapisal kural — modelin ne dusundugune bakilmadi",
            next_step="akisin sekli mi yanlis, gorev mi bu sekle sigmiyor",
        )

    def reasons(self) -> tuple[str, ...]:
        return ()

    def why_text(self, reason: str, ctx) -> str:
        return f"yapisal kural ihlali ({reason})"
