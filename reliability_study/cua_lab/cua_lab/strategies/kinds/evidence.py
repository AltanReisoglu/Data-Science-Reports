"""
KATEGORİ: DÜNYA — modelin dışından kanıt al.

ORTAK OLAN: ajanın davranışını değil SONUCU denetleme, bir "oracle"a sorma,
kanıt yoksa reddetme, ve reddin koşumu bitirmeyip gözlem akışına dönmesi.

AYRIŞTIKLARI TEK NOKTA: kanıtı NEREDEN alıyorlar.

    verify-gate       ortamdan — is gercekten yapildi mi
    galileo-breaker   arac katmanindan — hata ORANI
    telemetry-repair  kendi izinden — uc deterministik kural + geri sarma
"""

from __future__ import annotations

from ...events import CONTINUE, Action, Verdict
from ..base import BaseStrategy, StopReport


class EvidenceStrategy(BaseStrategy):
    """Kanıta bakan zihniyetlerin ortak tabanı."""

    family = "src"
    kind = "evidence"

    max_rejections: int = 2     # kac reddedisten sonra bu bir iddia sorunu degil
    terminal: str = "DEGRADED"

    def on_run_start(self, ctx) -> None:
        self._ctx = ctx
        self._red = 0
        self._son_kanit = ""
        self._setup(ctx)

    def _setup(self, ctx) -> None:
        """Alt sınıfın kendi durumu."""

    # -- ortak reddetme merdiveni -------------------------------------------

    def _reddet(self, sebep: str, kanit: str, detay: dict | None = None) -> Verdict:
        """Kanıt yoksa: önce gözleme geri ver, ısrar ederse durdur.

        `sde_offer_loop`un kritik ayrıntısı — kapı açılmazsa KOŞUM BİTMİYOR.
        Doğrulama sonucu ajanın gözlem akışına geri veriliyor ve ajan kendi
        hatasını görüp düzeltiyor.
        """
        self._red += 1
        self._son_kanit = kanit
        d = {"kanit": kanit, "red": self._red, **(detay or {})}
        if self._red > self.max_rejections:
            return Verdict(Action.STOP, sebep, {**d, "terminal": self.terminal})
        return Verdict(Action.NUDGE, f"{sebep}_rejected",
                       {**d, "mesaj": f"DOGRULAMA DUSTU: {kanit}. Is bitmedi, devam et."})

    def on_stop(self, reason: str, ctx) -> StopReport | None:
        if reason not in self.reasons():
            return None
        return StopReport(
            reason=reason,
            why=self.why_text(reason, ctx),
            tried=f"{ctx.step} adim, {self._red} kez reddedildi",
            found=self._son_kanit or "-",
            next_step="dogrulayici mi yanlis, gorev mi imkansiz — ikisi de insan bakisi ister",
        )

    def reasons(self) -> tuple[str, ...]:
        return ()

    def why_text(self, reason: str, ctx) -> str:
        return f"kanit uretilemedi ({reason})"
