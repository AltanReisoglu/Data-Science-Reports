"""
Strands — tekrarı sayma, çeşitliliği ölç.  KATEGORİ: PENCERE · SEVİYE 2

Kaynak: `strands-agents/sdk-python` (kod okundu)

ZİHNİYET: Tekrarın TANIMIYLA boğuşmaktan kaç. Ardışık tekrar mı sayacağız?
A-B-A-B olursa? A-B-C-A-B-C olursa? Her desen için ayrı tarama mı?

Strands soruyu tersine çeviriyor: SON N ADIMDA KAÇ FARKLI ŞEY OLDU?

    len(set(pencere)) < min_unique

Tek satır, bütün desenler. k=1..12 çevrim taramasına hiç gerek kalmıyor.

`loop_budget_source` bağımsız olarak aynı fikre varmış (`detect_repeat_pattern`:
"son 6 eylem <= 2 benzersiz parmak izine düşüyorsa döngü"). İki ayrı kaynak,
aynı içgörü — bu listede nadir.

BEDELİ: meşru olarak dar alanda çalışan işler de düşük çeşitlilik gösterir.
Savunması `window` düğmesi: pencere DOLMADAN karar verilmiyor.
"""

from __future__ import annotations

from collections import deque

from ...events import CONTINUE, Action, EventKind, Verdict
from ..base import register
from ..kinds import WindowStrategy


@register
class StrandsEntropy(WindowStrategy):
    id = "strands-entropy"
    title = "Strands: cesitlilik/entropi olcumu"
    source = "strands-agents/sdk-python"
    mentality = "Tekrari sayma, cesitliligi olc — tek kural, butun desenler"
    priority = 8
    why = (
        "Tekrarin TANIMIYLA bogusmadan butun dongu desenlerini yakalar. A-B-A-B da, uc adimlik cevrim de, ardisik tekrar da tek kuralla gorunur.")
    action = "Son N adimda kac FARKLI eylem oldugunu say; esigin altina duserse durdur"
    blind_spot = (
        "Mesru olarak dar alanda calisan is de dusuk cesitlilik gosterir.")
    family = "harness"

    escalate = False
    eylem_penceresi = 6
    min_unique = 3

    def _setup(self, ctx) -> None:
        self._acts: deque[str] = deque(maxlen=self.eylem_penceresi)

    def probe_action(self, ev, ctx) -> Verdict:
        self._acts.append(ev.signature())
        # Pencere dolmadan yargılama yok — yanlış pozitifin en ucuz savunması.
        if len(self._acts) < self.eylem_penceresi:
            return CONTINUE
        benzersiz = len(set(self._acts))
        if benzersiz < self.min_unique:
            return Verdict(Action.STOP, "low_diversity",
                           {"pencere": self.eylem_penceresi,
                            "benzersiz": benzersiz, "esik": self.min_unique})
        return CONTINUE

    def snapshot(self) -> dict:
        n = len(getattr(self, "_acts", []))
        benzersiz = len(set(getattr(self, "_acts", [])))
        return {"tur": "pencere", "sinyal": [
            ("benzersiz eylem", benzersiz if n >= self.eylem_penceresi else -1,
             self.min_unique)],
            "not": f"pencere {n}/{self.eylem_penceresi}"
                   + ("" if n >= self.eylem_penceresi else " — dolmadan yargılamaz")}

    def reasons(self):
        return ("low_diversity",)

    def why_text(self, reason: str, ctx) -> str:
        return (f"son {self.eylem_penceresi} eylem yalnizca {len(set(self._acts))} "
                f"farkli sey yapti (esik {self.min_unique})")
