"""
KATEGORİ: PENCERE — son N olayı tut, karşılaştır.

ORTAK OLAN: olay penceresi, imza çıkarma (oynak alanlar `VOLATILE_FIELDS` ile
atılıyor), eşik sözlüğü, kademeli müdahale merdiveni, durma raporu.

AYRIŞTIKLARI TEK NOKTA: pencereye BAKMA BİÇİMİ — "aynı şey" ne demek?

    openhands-stuck    bes ayri desen, her birine ayri esik
    strands-entropy    desen tanimlamaz; kac FARKLI sey oldugunu sayar
    openclaw-pingpong  adlandirilmis dedektorler + uc kademe + sikistirma korumasi
    loopguard-dignity  olaya degil DUNYANIN DURUMUNA bakar
    pi-signature       alti ucuz sinyal, yakin-benzer metin dahil
"""

from __future__ import annotations

from collections import deque

from ...events import CONTINUE, Action, EventKind, Verdict
from ..base import BaseStrategy, StopReport


class WindowStrategy(BaseStrategy):
    """Olay penceresi tutan zihniyetlerin ortak tabanı."""

    family = "harness"
    kind = "window"

    window: int = 20              # kac olay saklanacak
    escalate: bool = False        # True = once NUDGE, tekrarda STOP
    terminal: str = "STUCK"

    def on_run_start(self, ctx) -> None:
        self._ctx = ctx
        self._win: deque = deque(maxlen=self.window)
        self._nudged: set[str] = set()
        self._setup(ctx)

    def _setup(self, ctx) -> None:
        """Alt sınıfın kendi durumu."""

    # -- kayıt -------------------------------------------------------------

    def on_action(self, ev, ctx) -> Verdict:
        self._ctx = ctx                 # snapshot() kisit durumunu okusun
        if ev.kind is EventKind.ACTION:
            self._win.append(ev)
            return self._karar(self.probe_action(ev, ctx))
        return CONTINUE

    def on_observation(self, ev, ctx) -> Verdict:
        self._ctx = ctx
        self._win.append(ev)
        return self._karar(self.probe_observation(ev, ctx))

    # -- kademeli müdahale -------------------------------------------------

    def _karar(self, v: Verdict) -> Verdict:
        """Kademelendirme ORTAK. Bir kaynağın kademelendirip
        kademelendirmediği `escalate` düğmesiyle söyleniyor.

        HAL'in 21.730 koşumluk log analizinde koşum ortasında hatasını
        düzelten ajan başarma olasılığını 1,5-4x artırıyor; tek eşikli sert
        kesme bu şansı yok ediyor. Ama OpenHands bilerek kademelendirmiyor —
        "sıkıştıysan sıkışmışsındır". İkisi de savunulabilir, o yüzden düğme.
        """
        if not v.triggered:
            return CONTINUE
        if self.escalate and v.reason not in self._nudged:
            self._nudged.add(v.reason)
            return Verdict(Action.NUDGE, v.reason,
                           {**v.detail,
                            "mesaj": self.nudge_text(v.reason, v.detail)})
        v.detail.setdefault("terminal", self.terminal)
        return v

    # -- alt sınıfın dolduracağı yerler ------------------------------------

    def probe_action(self, ev, ctx) -> Verdict:
        return CONTINUE

    def probe_observation(self, ev, ctx) -> Verdict:
        return CONTINUE

    def nudge_text(self, reason: str, detail: dict) -> str:
        """Ölçülmüş bulgu (`real_time_Detection`): ajana genel bir öğüt vermek
        yerine HANGİ KONTROLÜN DÜŞTÜĞÜNÜ söylemek en iyi kurtarma oranını
        veriyor (%45 vs %36). O yüzden mesajda dedektörün adı geçiyor."""
        return f"DEDEKTOR '{reason}' tetiklendi — yaklasimi degistir, tekrarlama."

    # -- rapor -------------------------------------------------------------

    def on_stop(self, reason: str, ctx) -> StopReport | None:
        if reason not in self.reasons():
            return None
        return StopReport(
            reason=reason,
            why=self.why_text(reason, ctx),
            tried=f"{ctx.step} adim, son eylem: "
                  f"{ctx.history[-1] if ctx.history else '-'}",
            found=f"ekran son {len(ctx.screen_hashes)} adimda "
                  f"{len(set(ctx.screen_hashes))} farkli durum gordu",
            next_step="farkli bir yaklasim ya da insan mudahalesi",
        )

    def reasons(self) -> tuple[str, ...]:
        return ()

    def why_text(self, reason: str, ctx) -> str:
        return f"tekrar deseni tespit edildi ({reason})"

    def snapshot(self) -> dict:
        return {"tur": "pencere", "olay": len(getattr(self, "_win", [])),
                "uyarilan": sorted(getattr(self, "_nudged", set()))}
