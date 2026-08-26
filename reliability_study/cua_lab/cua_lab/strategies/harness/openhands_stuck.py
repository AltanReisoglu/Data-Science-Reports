"""
OpenHands — beş desen, ve "sıkışmak" ayrı bir sonuç.  KATEGORİ: PENCERE · SEVİYE 2

Kaynak: `OpenHands/software-agent-sdk` → `stuck_detector.py` (kod okundu)

ZİHNİYET: Döngü tek bir sayaçla değil, BEŞ AYRI DESENLE aranır. Ve
yakalandığında koşum "hata" vermez — `ConversationExecutionStatus.STUCK` diye
AYRI BİR TERMİNAL DURUMA geçer. Sıkışmak, patlamaktan farklı bir sonuçtur.

Pencere mantığı `kinds/window.py`'de. OpenHands'in FARKI iki yerde:

  escalate=False   Kademelendirme YOK. "Sikistiysan sikismissindir."
                   (`pi-signature` tam tersini yapiyor — dugme orada.)
  probe            Ortak dedektorun bes senaryosunu OLDUGU GIBI kullaniyor;
                   kendi kurali yok. Esikler kaynaktan: 4/3/3/6, pencere 20.

Kritik ayrıntı `_event_eq()`: karşılaştırmada `tool_call_id` gibi her turda
değişen alanlar YOK SAYILIYOR. Yapılmazsa dedektör sessizce hiçbir şey bulmaz —
hata türlerinin en kötüsü. Bizde `Event.signature()` + `VOLATILE_FIELDS`.
"""

from __future__ import annotations

from ...events import Verdict
from ..base import register
from ..kinds import WindowStrategy


@register
class OpenHandsStuck(WindowStrategy):
    id = "openhands-stuck"
    title = "OpenHands: bes senaryolu stuck detector"
    source = "OpenHands/software-agent-sdk stuck_detector.py"
    mentality = "Olay deseni eslestirme + STUCK ayri terminal durum"
    priority = 5
    why = (
        "Ajanin ayni cagriyi, ayni sonucu ya da A-B-A-B gibi desenleri tekrarladigini yakalar. Butcenin soyleyemedigi seyi soyler: NE yanlis gidiyor.")
    action = "Bes desen taramasi (4/3/3/6); yakalayinca DOGRUDAN durdurma — uyari yok"
    blind_spot = (
        "Imza normalizasyonu yanlissa SESSIZCE hicbir sey bulmaz ve testleri gecer.")
    family = "harness"

    escalate = False     # kaynakta kademelendirme yok

    def probe_observation(self, ev, ctx) -> Verdict:
        # `stage=False`: kaynakta kademelendirme YOK. Ortak dedektor kendi
        # icinde uyari-sonra-kes yapiyor; OpenHands zihniyeti onu atliyor.
        return ctx.detector.check(stage=False)

    def snapshot(self) -> dict:
        """Beş desenin HER BİRİNİN o anki sayacı — hangisi tetiklenmeye yakın."""
        d = getattr(self, "_ctx", None)
        if d is None:
            return {"tur": "pencere", "sinyal": []}
        t = d.detector.t
        pencere = d.detector._window()
        ciftler = d.detector._pairs(pencere)
        son = ciftler[-1] if ciftler else None
        ayni_gozlem = ayni_hata = 0
        for a, b in reversed(ciftler):
            if son and a.signature() == son[0].signature() and b.signature() == son[1].signature():
                if b.kind.value == "error":
                    ayni_hata += 1
                else:
                    ayni_gozlem += 1
            else:
                break
        monolog = 0
        for e in reversed(pencere):
            if e.kind.value == "message":
                monolog += 1
            else:
                break
        ekran = d.screen_hashes[-t.no_progress:]
        durgun = len(ekran) if ekran and len(set(ekran)) == 1 else 0
        return {"tur": "pencere", "sinyal": [
            ("ayni eylem+gozlem", ayni_gozlem, t.repeat_action_observation),
            ("ayni eylem+hata", ayni_hata, t.repeat_action_error),
            ("monolog", monolog, t.monologue),
            ("ilerleme yok", durgun, t.no_progress),
        ], "pencere": len(pencere)}

    def probe_observation_ctx(self, ctx):
        self._ctx = ctx

    def reasons(self):
        return ("repeat_action_observation", "repeat_action_error", "monologue",
                "cycle_k1", "cycle_k2", "cycle_k3", "cycle_k4", "cycle_k5",
                "cycle_k6", "cycle_k7", "cycle_k8", "cycle_k9", "cycle_k10",
                "cycle_k11", "cycle_k12", "no_progress")
