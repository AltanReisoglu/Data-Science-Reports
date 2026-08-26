"""
Modexa — döngünün şeklini kısıtla.  KATEGORİ: ŞEKİL · SEVİYE 4

Kaynak: Modexa — *Ajan Döngüsü Problemi: "Akıllı" Sistemler Durmadığında*

ZİHNİYET: Ajanlar belirsiz özgürlükleri sever; sistemlerin net durumlara
ihtiyacı vardır. Bir geçişe izin vermezseniz o döngü hiç doğmaz — tespit etmeye
gerek kalmıyor, çünkü mümkün değil.

Durum makinesi: ANLA → TOPLA → EYLEM → DOĞRULA → YANIT → DEVRET
Kritik tasarım: `DOGRULA → TOPLA` geçişi YOK. "Doğrulayamadım, baştan
toplayayım" döngüsü böyle imkânsız hale geliyor.

GERİ DÖNÜŞ MERDİVENİ: ajan kendi tekrar denemesini icat etmiyor, sabit bir
merdiveni tırmanıyor. Dördüncü basamak — kullanıcıya sormak — kaynakta ayrıca
tarif edilmiş: TEK soru sor · neyin değişeceğini açıkla · umursamazsa varsayılan
sun. Gerekçesi: *"40 adım boyunca yanlış tahmin yürütmekten ucuzdur."*

Kaynağın beşinci döngü sebebi bu listede başka hiçbir yerde yok ve HİÇBİR
mekanizma onu çözmüyor: *"ajanın 'yanlış yapmamak' üzerine optimize edilmesi."*
    "Bir sorun çözücü inşa etmediniz. Bir riskten kaçınma makinesi inşa ettiniz."
Kaynağı kod değil PROMPT — guardrail'in kapsam sınırının en net örneği.
"""

from __future__ import annotations

from ...events import CONTINUE, Act, Action, EventKind, Verdict
from ..base import StopReport, register
from ..kinds import ShapeStrategy

ANLA, TOPLA, EYLEM, DOGRULA, YANIT, DEVRET = (
    "ANLA", "TOPLA", "EYLEM", "DOGRULA", "YANIT", "DEVRET")

# Kritik: DOGRULA -> TOPLA YOK.
ALLOWED = {
    ANLA:    {TOPLA, DEVRET},
    TOPLA:   {EYLEM, TOPLA, DEVRET},     # kendine donus var ama sinirli
    EYLEM:   {DOGRULA, DEVRET},
    DOGRULA: {YANIT, EYLEM, DEVRET},
    YANIT:   set(),
    DEVRET:  set(),
}

# Hangi eylem hangi duruma ait.
EYLEM_DURUM = {
    Act.SCREENSHOT.value: TOPLA, Act.CURSOR_POSITION.value: TOPLA,
    Act.WAIT.value: TOPLA, Act.ZOOM.value: TOPLA,
}

LADDER = ["backoff_retry", "switch_tool", "narrow_scope", "ask_user",
          "best_effort_answer"]


@register
class ModexaStateMachine(ShapeStrategy):
    id = "modexa-statemachine"
    title = "Modexa: durum makinesi + geri donus merdiveni"
    source = "Modexa — Ajan Dongusu Problemi"
    mentality = "Serbestligi kisitla — izinsiz gecis, olusmayan dongu"
    priority = 13
    why = (
        "Donguyu tespit etmeye calismak yerine OLUSMASINA IZIN VERMEZ. Bir gecise izin yoksa o dongu hic dogmaz — yanlis pozitif problemi hic ortaya cikmaz.")
    action = "Izinli gecis tablosu + bes basamakli geri donus merdiveni (son basamak: kullaniciya sor)"
    blind_spot = (
        "Esnekligi azaltir; ongorulemez isler durum makinesine sigmaz. Sonradan eklenemez.")

    max_topla_self = 3       # TOPLA->TOPLA kac kez
    max_dogrulama_hatasi = 2  # DOGRULA kac kez dusunce merdivene girilsin
    terminal = "NEEDS_INPUT"

    def _setup(self, ctx) -> None:
        self._durum = ANLA
        self._merdiven = 0
        self._topla_sayaci = 0
        self._basarisiz = 0

    def _hedef(self, ev) -> str:
        return EYLEM_DURUM.get(ev.name, EYLEM)

    def on_action(self, ev, ctx) -> Verdict:
        if ev.kind is not EventKind.ACTION:
            return CONTINUE
        hedef = self._hedef(ev)

        # ANLA'dan dogrudan EYLEM'e gecilemez; once TOPLA.
        if self._durum is ANLA and hedef == EYLEM:
            self._durum = TOPLA

        if hedef == TOPLA and self._durum == TOPLA:
            self._topla_sayaci += 1
            if self._topla_sayaci > self.max_topla_self:
                return self._merdivene_gir("topla_self_loop", ctx)
            return CONTINUE

        if hedef not in ALLOWED[self._durum]:
            self._ihlal.append(f"{self._durum}->{hedef}")
            return self._merdivene_gir("illegal_transition", ctx)

        self._durum = hedef
        # EYLEM tamamlaninca DOGRULA'ya gecilir; DOGRULA'dan TOPLA'ya DONULEMEZ.
        if self._durum == EYLEM:
            self._durum = DOGRULA
            self._topla_sayaci = 0
        return CONTINUE

    def on_observation(self, ev, ctx) -> Verdict:
        """DOGRULA durumunun isi: eylem GERCEKTEN bir sey yapti mi.

        Durum makinesi tek basina EYLEM->DOGRULA->EYLEM cevrimini yasal sayar
        ve sonsuza kadar doner — olculdu, `none` gibi davraniyordu. Kaynagin
        DOGRULA durumu bos bir gecis degil, bir KAPI: dogrulama dusunce ajan
        kendi retry'ini icat etmiyor, MERDIVENE giriyor.
        """
        if self._durum != DOGRULA:
            return CONTINUE
        degisti = ctx.last_screen_changed()
        if degisti is False:
            self._basarisiz += 1
            if self._basarisiz >= self.max_dogrulama_hatasi:
                self._basarisiz = 0
                return self._merdivene_gir("verify_failed", ctx)
        else:
            self._basarisiz = 0
        # DURUM DOGRULA'DA KALIYOR. Sonraki eylemin DOGRULA->EYLEM gecisi
        # ALLOWED tablosunda ZATEN yasal. Burada durumu elle EYLEM'e cekmek
        # bir sonraki adimi EYLEM->EYLEM yapiyordu ve her adim "illegal
        # transition" sayiliyordu — mesru retry'i besinci adimda kesen
        # yanlis pozitifin sebebi buydu (olculdu, 4 kontrol kombinasyonunda).
        return CONTINUE

    def _merdivene_gir(self, sebep: str, ctx) -> Verdict:
        """Ajan kendi tekrar denemesini icat etmiyor; merdiveni tirmaniyor."""
        if self._merdiven >= len(LADDER) - 1:
            return Verdict(Action.STOP, "ladder_exhausted",
                           {"sebep": sebep, "basamak": LADDER[-1],
                            "terminal": "DEGRADED"})
        basamak = LADDER[self._merdiven]
        self._merdiven += 1
        if basamak == "ask_user":
            # Kaynagin sozlesmesi: TEK soru, neyin degisecegi, varsayilan.
            return Verdict(Action.STOP, "abstain_need_input",
                           {"basamak": basamak, "abstain": True,
                            "soru": "Hangi alani doldurmam gerekiyor? "
                                    "Cevap gelmezse 'Ad' alanini varsayacagim."})
        return Verdict(Action.NUDGE, f"ladder_{basamak}",
                       {"basamak": basamak, "sebep": sebep,
                        "mesaj": f"GERI DONUS MERDIVENI -> {basamak}: "
                                 f"'{sebep}' nedeniyle yaklasimi degistir."})

    def snapshot(self) -> dict:
        return {"tur": "sekil", "sinyal": [
            ("merdiven basamagi", getattr(self, "_merdiven", 0), len(LADDER)),
            ("dogrulama hatasi", getattr(self, "_basarisiz", 0), self.max_dogrulama_hatasi)],
            "not": f"durum: {getattr(self, '_durum', '?')}"
                   f" · sonraki: {LADDER[min(getattr(self, '_merdiven', 0), len(LADDER) - 1)]}"}

    def reasons(self):
        return ("illegal_transition", "topla_self_loop", "ladder_exhausted",
                "abstain_need_input", "verify_failed")

    def on_stop(self, reason: str, ctx) -> StopReport | None:
        if reason not in self.reasons():
            return None
        return StopReport(
            reason=reason,
            why=f"durum makinesi: {reason} (durum={self._durum}, "
                f"merdiven={LADDER[min(self._merdiven, len(LADDER)-1)]})",
            tried=f"{ctx.step} adim; ihlaller: {', '.join(self._ihlal[-4:]) or '-'}",
            found="dongu TESPIT edilmedi — olusmasina izin verilmedi",
            next_step="akisin sekli mi yanlis, gorev mi bu sekle sigmiyor",
        )
