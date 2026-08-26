"""
Real-Time Detection and Repair — kural koy, ihlali söyle, geri sar.
KATEGORİ: DÜNYA · SEVİYE 5

Kaynak: *Real-Time Detection and Repair of LLM Agent Failures* (tam metin)

ZİHNİYET: Doğrulama kapısının somut hâli — ve arkasında ÖLÇÜM var.

Üç DETERMİNİSTİK kontrol tanımlıyor:
  total_consistency  ajanin soyledigi toplam gercekten gordugu verilerden cikiyor mu
  required_coverage  gereken butun arac cagrilari yapildi mi
  tool_contract      aracin donduru sey o aracin uretebilecegi bicimde mi

Bunlar istatistik değil, KURAL. Eşik ayarı yok, kalibrasyon yok, ikinci model
yok. Ya sağlar ya sağlamaz. Ölçülmüş sonuç çarpıcı:

    3 deterministik kural   %96 yakalama /  0/63 yanlis pozitif
    istatistiksel monitor   %54 yakalama / 11/63 = %17 yanlis pozitif

Asıl bulgu ONARIM tarafında. Kontrol düşünce ajan son sağlam noktaya geri
sarılıyor. Ona ne söylenmeli?

    hicbir sey, sadece yeniden dene   %16
    genel "tekrar kontrol et"         %36
    DOGRU CEVABI ver                  %36
    HANGI KONTROLUN dustugunu soyle   %45   <-- en iyisi

Cevabı verince ajan onu kopyalıyor, NEDEN yanıldığını anlamıyor. Bu bulgu
`kinds/window.py`'deki `nudge_text`'i de belirledi — bütün dedektörler
tetiklenen kontrolün ADINI söylüyor.

SINIR: aynı çalışma döngüde onarımın işe yaramadığını ölçmüş. Hedef sapmasında
beş vakanın dördü kurtarılıyor, ama döngüde yeniden çalıştırma çoğu zaman aynı
döngüyü üretiyor. Döngüde doğru hamle onarmak değil, DURMAK — `max_repairs`.
"""

from __future__ import annotations

from ...events import CONTINUE, Action, EventKind, Verdict
from ..base import StopReport, register
from ..kinds import EvidenceStrategy


@register
class TelemetryRepair(EvidenceStrategy):
    id = "telemetry-repair"
    title = "Telemetri + deterministik kontrol + onarim"
    source = "Real-Time Detection and Repair of LLM Agent Failures"
    mentality = "Kural koy, ihlali soyle, checkpoint'e geri sar"
    priority = 11
    why = (
        "Uc deterministik kural (toplam tutarliligi, gerekli arac kapsamasi, arac sozlesmesi) hatalarin %96'sini SIFIR yanlis pozitifle yakaliyor — egitilmis istatistiksel monitor %54/%17'de kaliyor.")
    action = "Checkpoint'e geri sar ve HANGI KONTROLUN dustugunu soyle (olculmus en iyi: %45)"
    blind_spot = (
        "Checkpoint tutmak mimariye yuk. Ve donguede onarim ise yaramiyor — durmak gerek.")

    max_repairs = 1        # ikinci ihlalde onarim degil DURMA
    max_rejections = 1

    def _setup(self, ctx) -> None:
        self._repairs = 0
        self._checkpoint: int = 0     # son saglam adim
        self._dusen: list[str] = []

    # -- üç deterministik kontrol ------------------------------------------

    def _kontroller(self, ctx, iddia: str) -> tuple[str, str] | None:
        """Sırayla bak; ilk düşen kontrolün ADI dönüyor."""
        ekran = ctx.sandbox.describe()

        # 1) total_consistency — iddia gordugu veriden cikiyor mu
        dv = (ctx.extra or {}).get("dogrulayici")
        if callable(dv):
            try:
                ok, kanit = dv(ctx, iddia)
                if not ok:
                    return ("total_consistency", kanit)
            except Exception as e:
                return ("total_consistency", f"dogrulayici hata verdi: {e}")
        elif "gonderildi" not in ekran and "gonder" in iddia.lower():
            return ("total_consistency",
                    "iddia 'gonderildi' diyor ama ekranda o durum yok")

        # 2) required_coverage — gereken cagrilar yapildi mi
        # Gereken kume GOREVE OZEL. Sabit `{"type","left_click"}` bu paketin
        # form senaryosuna aitti; terminal gorevinde hicbir zaman saglanmiyor
        # ve her iddiayi reddediyordu. Kosum saglarsa onunki kullaniliyor.
        yapilan = {e.name for e in ctx.events if e.kind is EventKind.ACTION}
        disaridan = (ctx.extra or {}).get("gerekli_araclar")
        if disaridan is not None:
            # DISARIDAN gelen kume "en az biri" anlaminda. Genel bir ajanda
            # ayni ihtiyaci birden fazla arac karsiliyor (`terminal` ile
            # `terminal.yaz` gibi); hepsini sart kosmak yanlis pozitif uretir —
            # olculdu, dogru biten bir dosya gorevi bu yuzden reddedildi.
            gerekli = set(disaridan)
            if gerekli and not (gerekli & yapilan):
                return ("required_coverage",
                        f"su araclardan hicbiri cagrilmadi: {', '.join(sorted(gerekli))}")
        else:
            # Kendi sentetik senaryosu: HEPSI gerekli.
            gerekli = {"type", "left_click"}
            if not gerekli <= yapilan:
                return ("required_coverage",
                        f"eksik arac cagrisi: {', '.join(sorted(gerekli - yapilan))}")

        # 3) tool_contract — arac sonuclari beklenen bicimde mi
        for e in ctx.events[-6:]:
            if e.kind is EventKind.OBSERVATION and not e.payload.get("output"):
                return ("tool_contract", f"'{e.name}' bos cikti dondurdu")
        return None

    def on_finish_claim(self, fin, ctx) -> Verdict:
        dusen = self._kontroller(ctx, fin.answer)
        if dusen is None:
            self._checkpoint = ctx.step
            return CONTINUE

        ad, kanit = dusen
        self._dusen.append(ad)
        self._repairs += 1
        if self._repairs > self.max_repairs:
            # Olculmus sinir: donguede yeniden calistirma ayni donguyu uretiyor.
            return Verdict(Action.STOP, "repair_exhausted",
                           {"dusen": ad, "kanit": kanit,
                            "onarim": self._repairs, "terminal": "DEGRADED"})

        # ONARIM MESAJI: cevabi verme, HANGI KONTROLUN dustugunu soyle (%45).
        return Verdict(Action.NUDGE, "check_failed",
                       {"dusen": ad, "kanit": kanit,
                        "checkpoint": self._checkpoint,
                        "mesaj": f"KONTROL DUSTU: '{ad}'. {kanit}. "
                                 f"Adim {self._checkpoint}'e geri sariliyor — "
                                 f"bu kontrolu saglayacak sekilde devam et."})

    def snapshot(self) -> dict:
        return {"tur": "dunya", "sinyal": [
            ("onarim", getattr(self, "_repairs", 0), self.max_repairs + 1)],
            "not": (f"dusen: {', '.join(self._dusen[-2:])}" if getattr(self, "_dusen", None)
                    else "3 kural: total_consistency / required_coverage / tool_contract")}

    def reasons(self):
        return ("repair_exhausted",)

    def on_stop(self, reason: str, ctx) -> StopReport | None:
        if reason != "repair_exhausted":
            return None
        return StopReport(
            reason=reason,
            why=f"'{self._dusen[-1]}' kontrolu {self._repairs} kez dustu — "
                f"onarim butcesi bitti",
            tried=f"{ctx.step} adim, checkpoint adim {self._checkpoint}, "
                  f"dusen kontroller: {', '.join(self._dusen)}",
            found="onarim mesaji cevabi degil DUSEN KONTROLU soyluyor "
                  "(olculmus en iyi kademe: %45)",
            next_step="donguede onarim ise yaramiyor — dogru hamle durmak",
        )
