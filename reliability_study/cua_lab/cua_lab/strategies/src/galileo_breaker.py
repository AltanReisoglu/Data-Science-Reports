"""
Galileo — suç ajanda değil, araç katmanında.  KATEGORİ: DÜNYA · SEVİYE 3

Kaynak: Galileo AI — ajan hata analizi sunumu

ZİHNİYET: Suçun yerini değiştirmek. Ajan bir aracı çağırıyor, hata alıyor,
tekrar deniyor, üçüncüde başarıyor. Koşum BAŞARILI bitiyor, kimse ticket açmıyor.
    "Bu, hakkında hiç ticket açılmayan türden bir hata. Sessiz ama öldürücü."
    "Ajanlar doğru davrandı. Zafiyet veri çekme katmanında."

Kanıtı ARAÇ KATMANINDAN alıyor. Tekrar saymak işe yaramıyor — koşum başarıyla
bitiyor. Bakılan şey HATA ORANI, ve durum ARAÇ BAŞINA tutuluyor.

Sayı değil ORAN olması kritik: "üç hata" bir araç yüz kez çağrıldıysa
normaldir, dört kez çağrıldıysa değildir.

İki kademe, kasten ayrı:
  wasted_calls        kosumu KESMEZ, bosa giden cagrilari RAPORLAR.
                      `flaky` gibi mesru retry'i kesmemek icin.
  tool_circuit_open   oran esigi asilirsa o arac icin devre kesici.
"""

from __future__ import annotations

from collections import defaultdict, deque

from ...events import CONTINUE, Action, EventKind, Verdict
from ..base import StopReport, register
from ..kinds import EvidenceStrategy


@register
class GalileoBreaker(EvidenceStrategy):
    id = "galileo-breaker"
    title = "Galileo: arac bazli devre kesici (oran esigi)"
    source = "Galileo AI — ajan hata analizi"
    mentality = "Suc ajanda degil aracta — sayi degil ORAN"
    priority = 12
    why = (
        "Sorunun ajanda degil ARACTA oldugunu ortaya cikarir. Basarili gorunen ama iki cagriyi bosa harcayan kosumlari da yakalar — hakkinda hic ticket acilmayan hata.")
    action = "Arac basina hata ORANI; esigi asarsa o arac icin devre kesici"
    blind_spot = (
        "Gecici ile kalici hatayi ayirmasi tamamen esige bagli.")

    window = 5
    min_calls = 4
    error_rate = 0.75
    waste_warn = 2

    def _setup(self, ctx) -> None:
        self._calls: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self._wasted: dict[str, int] = defaultdict(int)

    def on_observation(self, ev, ctx) -> Verdict:
        if ev.kind not in (EventKind.OBSERVATION, EventKind.ERROR):
            return CONTINUE
        hata = ev.kind is EventKind.ERROR
        self._calls[ev.name].append(hata)
        if hata:
            self._wasted[ev.name] += 1

        pencere = self._calls[ev.name]
        if len(pencere) >= self.min_calls:
            oran = sum(pencere) / len(pencere)
            if oran >= self.error_rate:
                return Verdict(Action.STOP, "tool_circuit_open",
                               {"arac": ev.name, "oran": round(oran, 2),
                                "pencere": len(pencere), "esik": self.error_rate,
                                "terminal": self.terminal})

        # Kesme yok — yalnizca gorunur kil.
        if hata and self._wasted[ev.name] == self.waste_warn:
            return Verdict(Action.NUDGE, "wasted_calls",
                           {"arac": ev.name, "bosa": self._wasted[ev.name],
                            "mesaj": f"UYARI: {ev.name} icin {self._wasted[ev.name]} "
                                     f"cagri bosa gitti — arac katmani saglikli mi?"})
        return CONTINUE

    def snapshot(self) -> dict:
        satir = []
        for arac, pencere in getattr(self, "_calls", {}).items():
            n = len(pencere)
            oran = sum(pencere) / n if n else 0.0
            satir.append((f"{arac} hata orani", round(oran, 2), self.error_rate))
        return {"tur": "dunya", "sinyal": satir[:3],
                "not": f"{sum(getattr(self, '_wasted', {}).values())} cagri bosa gitti"
                       f" · min {self.min_calls} cagri gerek"}

    def reasons(self):
        return ("tool_circuit_open",)

    def on_stop(self, reason: str, ctx) -> StopReport | None:
        bosa = sum(self._wasted.values())
        if reason == "tool_circuit_open":
            arac = max(self._wasted, key=self._wasted.get, default="-")
            return StopReport(
                reason=reason,
                why=f"'{arac}' araci icin devre kesici acildi — hata orani esigi asti",
                tried=f"{ctx.step} adim, {bosa} cagri bosa gitti",
                found="ajan dogru davrandi; zafiyet arac katmaninda",
                next_step="araci/saglayiciyi degistir ya da alt akisi koru")
        if bosa:
            # Baska bir sebeple durulsa bile israfi RAPORLA.
            return StopReport(
                reason=reason,
                why=f"({reason}) — ayrica {bosa} cagri bosa gitti",
                found=", ".join(f"{k}: {v} hata" for k, v in self._wasted.items()),
                next_step="sessiz israf: kosum basarili bitse de bu maliyet gercek")
        return None
