"""
KATEGORİ: SAYAÇ — bir sayı tut, aşınca ne yapacağına karar ver.

ORTAK OLAN: beş eksen (adım · replan · token · süre · dolar), eşik kontrolü,
terminal etiket, durma raporu. Hepsi burada, bir kez.

AYRIŞTIKLARI TEK NOKTA: sayaç dolduğunda ne oluyor, ve dolmadan önce modele
haber veriliyor mu. Beş alt sınıf bu iki soruya beş farklı cevap veriyor:

    arize-control       sert kes, uyarma, adim ekseni birincil
    agentbudget-dollar  rezerv ayir, kismi cevap uret
    agentscope-grace    lutuf turu ver ama arac secimini kilitle
    hermes-no-pressure  ADIM ekseninde asla uyarma (olculmus: erken pes ettiriyor)
    claude-advisory     geri sayimi prompt'a koy — tavsiye, zorlama degil
"""

from __future__ import annotations

from ...events import CONTINUE, Action, Verdict
from ..base import BaseStrategy, StopReport

# Bütün bütçe stratejilerinin ortak terminal etiketi. Arize'ın ilkesi:
# "adım limiti" ile "hata" farklı sonuçlardır.
TERMINAL = "BUDGET_EXHAUSTED"


class BudgetStrategy(BaseStrategy):
    """Beş eksenli sayaç. Alt sınıflar aşağıdaki düğmeleri çeviriyor."""

    family = "src"
    kind = "budget"

    # -- düğmeler ----------------------------------------------------------
    hard: bool = True            # True = STOP (is copa gider) · False = DEGRADE
    reserve: float = 0.0         # sert limitin yuzde kaci once tetiklensin
    primary_axis: str | None = None   # once bakilacak eksen; sebep adini alir
    warn_axes: tuple[str, ...] = ()   # geri sayim ENJEKTE edilecek eksenler
    warn_at: float = 0.80        # o eksenlerde kacinci oranda uyarilsin
    grace_turns: int = 0         # limit dolunca kac ek tur
    lock_tools_in_grace: bool = False  # lutuf turlarinda arac cagrisi yasak

    AXES = ("steps", "replans", "tokens", "seconds", "cost_usd")

    # -- yaşam döngüsü -----------------------------------------------------

    def on_run_start(self, ctx) -> None:
        self._son_ctx = ctx
        self._grace_left = self.grace_turns
        self._exhausted: str | None = None
        self._warned: set[str] = set()

    def _kullanim(self, ctx):
        s, l = ctx.budget.state, ctx.budget.limits
        return {
            "steps": (s.steps, l.max_steps),
            "replans": (s.replans, l.max_replans),
            "tokens": (s.tokens, l.max_tokens),
            "seconds": (s.seconds, l.max_seconds),
            "cost_usd": (s.cost_usd, l.max_cost_usd),
        }

    def _sirali_eksenler(self):
        if self.primary_axis:
            return (self.primary_axis,) + tuple(
                a for a in self.AXES if a != self.primary_axis)
        return self.AXES

    # -- ana karar ---------------------------------------------------------

    def before_step(self, ctx) -> Verdict:
        self._son_ctx = ctx          # snapshot() icin
        k = self._kullanim(ctx)

        # Lütuf turları: limit zaten dolmuş, ek tur harcanıyor.
        if self._exhausted is not None:
            if self._grace_left > 0:
                self._grace_left -= 1
                return CONTINUE
            return self._bitir(self._exhausted, k)

        for eksen in self._sirali_eksenler():
            kullanilan, limit = k[eksen]
            if limit is None:
                continue
            etkin = limit * (1.0 - self.reserve)
            if kullanilan >= etkin:
                self._exhausted = eksen
                if self.grace_turns:
                    # Koşum bitmiyor; ajana toparlanma turu veriliyor.
                    self._grace_left = self.grace_turns
                    return self.on_grace_start(eksen, ctx)
                return self._bitir(eksen, k)

        return self.extra_check(ctx)

    def _bitir(self, eksen: str, k) -> Verdict:
        kullanilan, limit = k[eksen]
        sebep = (self.axis_reason(eksen))
        detay = {"eksen": eksen, "kullanilan": round(float(kullanilan), 4),
                 "limit": limit, "rezerv": self.reserve,
                 "terminated_by": eksen}
        if self.hard:
            return Verdict(Action.STOP, sebep, {**detay, "terminal": TERMINAL})
        # DEGRADE: döngü modelden nihai cevabı ister (`_force_finish`).
        return Verdict(Action.DEGRADE, sebep, detay)

    # -- alt sınıfların değiştirdiği yerler --------------------------------

    def axis_reason(self, eksen: str) -> str:
        """Durma sebebinin adı. `arize-control` adım eksenini `max_steps`
        diye adlandırıyor — 'budget_tokens' demek limitin dar olduğu
        bilgisini gizler."""
        if eksen == self.primary_axis:
            return f"max_{eksen}"
        return f"budget_{eksen}"

    def on_grace_start(self, eksen: str, ctx) -> Verdict:
        """Lütuf turu başlıyor. Varsayılan: sadece devam et."""
        return CONTINUE

    def extra_check(self, ctx) -> Verdict:
        """Alt sınıfa ek kontrol yeri (ör. zaman pencereli patlama)."""
        return CONTINUE

    # -- prompt enjeksiyonu -------------------------------------------------

    def decorate_request(self, req, ctx):
        """Geri sayım — YALNIZCA `warn_axes` içindeki eksenler için.

        Eksen ayrımı kozmetik değil, ölçülmüş bir bulgu: Hermes adım ekseninde
        uyarınca modeller erken pes ediyor; süre ekseninde aynı sorun yok.
        """
        if self.lock_tools_in_grace and self._exhausted is not None:
            req.forced_finish = True
            ctx.extra["forced_finish"] = True
            ctx.extra["forced_reason"] = self.axis_reason(self._exhausted)
        if not self.warn_axes:
            return req
        k = self._kullanim(ctx)
        for eksen in self.warn_axes:
            kullanilan, limit = k.get(eksen, (0, None))
            if not limit or eksen in self._warned:
                continue
            if kullanilan >= limit * self.warn_at:
                self._warned.add(eksen)
                kalan = max(0.0, float(limit) - float(kullanilan))
                ctx.note(self.warn_text(eksen, kalan, limit))
        return req

    def warn_text(self, eksen: str, kalan: float, limit) -> str:
        return f"BUTCE: {eksen} ekseninde {kalan:.0f}/{limit} kaldi — isi toparla."

    # -- rapor -------------------------------------------------------------

    def on_stop(self, reason: str, ctx) -> StopReport | None:
        if not (reason.startswith("budget_") or reason.startswith("max_")):
            return None
        s = ctx.budget.state
        return StopReport(
            reason=reason,
            why=self.why_text(reason),
            tried=f"{s.steps} adim - {s.replans} replan - {s.tokens} token "
                  f"- ${s.cost_usd:.4f} - {s.seconds:.1f}sn",
            found=f"ekran {len(set(ctx.screen_hashes))} farkli durum gordu",
            next_step="limit dogru mu? kosum dagilimindan olc (bkz. improvement-loop)",
        )

    def why_text(self, reason: str) -> str:
        return f"butce tukendi: {reason}"

    # -- kısıt durumu ------------------------------------------------------

    def snapshot(self) -> dict:
        """Panelde gösterilecek KISIT DURUMU — karar değil, karara ne kadar
        kaldığı. Guardrail'in kararını görmek kolay; asıl anlaşılmayan şey
        tetiklenmeye ne kadar kaldığı."""
        try:
            k = self._kullanim(self._son_ctx)
        except Exception:
            return {}
        satir = []
        for eksen in self._sirali_eksenler():
            kullanilan, limit = k.get(eksen, (0, None))
            if limit is None:
                continue
            etkin = limit * (1.0 - self.reserve)
            satir.append({"ad": eksen, "kullanilan": round(float(kullanilan), 3),
                          "limit": round(float(etkin), 3), "oran": min(1.0, kullanilan / etkin)
                          if etkin else 0.0})
        return {"tur": "butce", "eksenler": satir,
                "lutuf": self._grace_left if self._exhausted else None}
