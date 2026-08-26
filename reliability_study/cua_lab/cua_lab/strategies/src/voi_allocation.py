"""
Inference-Time Budget Control — bütçe bir tavan değil, bir bütçe.
KATEGORİ: KARAR · SEVİYE 6

Kaynak: *Inference-Time Budget Control for LLM Search Agents* (tam metin)

ZİHNİYET: Diğer on altı strateji "NE ZAMAN DUR?" diye soruyor. Bu strateji
"PARAYI NEREYE HARCA?" diye soruyor.

Bir tavan koyduğunuzda ajan tavana kadar istediğini yapar, sonra kesilir; tavana
NASIL geldiği umurunuzda değildir. Oysa aynı bütçeyle daha iyi bir sonuç
alınabilirdi.

MEKANİZMA — her adımda eylemler BİRİM BÜTÇE BAŞINA FAYDAYA göre puanlanıyor.
Ham faydaya göre değil: pahalı bir eylem biraz daha faydalı olabilir, ucuz bir
eylem neredeyse bedavadır.

ÇİFT BÜTÇE ve baskı EN KRİTİK EKSENE göre — ortalama değil MINIMUM:

    ρ = 1 − min( kalan_arac/B_arac , kalan_token/B_token )

Üç aramasını harcamış bir ajan, token'ı bol diye rahat değildir.

ÖLÇÜLMÜŞ: bütçe cezası bileşeni çıkarıldığında F1 0,63 → 0,43. Ana teknik sonuç
"puanlama kullandık" değil — KALAN BÜTÇEYİ EYLEM SEÇİMİNE AÇIKÇA KATMAK.
Ve ek hesap katmanına RAĞMEN süre 20,91 → 15,23 sn (%27,2 düşüş).

KORUYUCU KURAL: "cevap ver" her zaman en ucuz eylemdir, yani ρ arttıkça
OTOMATİK kazanır — cevabın doğru olup olmadığından bağımsız. Ajan aceleyle kötü
bir cevaba kaçar ve sistem bunu BAŞARI sayar. O yüzden üstte deterministik
guard: kanıt zayıfken erken cevap ENGELLENİYOR.
"""

from __future__ import annotations

from ...events import CONTINUE, Action, EventKind, Verdict
from ..base import StopReport, register
from ..kinds import DecisionStrategy


@register
class VoiAllocation(DecisionStrategy):
    id = "voi-allocation"
    title = "Inference-Time Budget Control: tavan degil tahsis"
    source = "Inference-Time Budget Control for LLM Search Agents"
    mentality = "Butce bir tavan degil bir butce — parayi nereye harca"
    priority = 15
    why = (
        "Tek katman 'ne zaman dur' diye sormuyor: 'parayi nereye harca' diye soruyor. Ayni butceyle daha iyi sonuc alinabilir — olculdu, ceza cikarilinca F1 0,63->0,43.")
    action = "Her adimda eylemleri birim butce basina faydaya gore puanla; erken cevabi engelle"
    blind_spot = (
        "DURDURMAZ — tek basina kosumu korumaz. Bol butcede kazanci erir.")

    min_evidence = 2      # koruyucu kural: bu kadar gozlem gormeden bitirme yok

    def _setup(self, ctx) -> None:
        self._rho = 0.0
        self._engellenen = 0
        # Kac gozlem "yeterli kanit" sayilir GOREVE bagli: cok adimli bir
        # arama ile tek komutluk bir dosya isi ayni esigi paylasamaz.
        if (n := (ctx.extra or {}).get("min_kanit")) is not None:
            self.min_evidence = int(n)

    # -- bütçe baskısı ------------------------------------------------------

    def _baski(self, ctx) -> float:
        """ρ — EN KRİTİK eksene göre. Ortalama alsaydık, araması bitmiş bir
        ajan 'token'ım bol' diye rahat görünürdü. Yanlış."""
        s, l = ctx.budget.state, ctx.budget.limits
        oranlar = []
        for kullanilan, limit in ((s.steps, l.max_steps),
                                  (s.tokens, l.max_tokens),
                                  (s.cost_usd, l.max_cost_usd),
                                  (s.seconds, l.max_seconds)):
            if limit:
                oranlar.append(max(0.0, 1.0 - kullanilan / limit))
        return 1.0 - min(oranlar) if oranlar else 0.0

    def observe(self, ctx) -> None:
        self._rho = self._baski(ctx)
        self._kayit.append({"adim": ctx.step, "rho": round(self._rho, 3)})

    def decide(self, ctx) -> Verdict:
        """Baskı arttıkça keşif pahalılaşıyor, bitirme çekici hale geliyor.

        Bu bir DURDURMA değil YÖNLENDİRME — prompt'a giriyor, model karar
        veriyor. Tavan ayrı bir mekanizma; bu katman tek başına koşumu korumaz.
        """
        if self._rho >= 0.60:
            ctx.note(f"BUTCE BASKISI %{self._rho*100:.0f} — kesif eylemleri "
                     f"artik pahali. Elindeki kanitla bitirmeyi degerlendir.")
        return CONTINUE

    # -- koruyucu kural -----------------------------------------------------

    def on_finish_claim(self, fin, ctx) -> Verdict:
        gozlem = sum(1 for e in ctx.events if e.kind is EventKind.OBSERVATION)
        if gozlem < self.min_evidence:
            self._engellenen += 1
            return Verdict(Action.NUDGE, "early_answer_blocked",
                           {"gozlem": gozlem, "gereken": self.min_evidence,
                            "rho": round(self._rho, 3),
                            "mesaj": f"ERKEN CEVAP ENGELLENDI: yalnizca {gozlem} "
                                     f"gozlem var. Ucuz oldugu icin cevap vermek "
                                     f"cazip — ama kanit yetersiz."})
        return CONTINUE

    def snapshot(self) -> dict:
        return {"tur": "karar", "sinyal": [
            ("butce baskisi rho", round(getattr(self, "_rho", 0.0), 2), 1.0)],
            "not": f"{getattr(self, '_engellenen', 0)} erken cevap engellendi"
                   f" · en KRITIK eksene gore"}

    def summary(self, reason: str, ctx) -> StopReport | None:
        if not self._kayit:
            return None
        return StopReport(
            reason=reason,
            why=f"tahsis katmani: son butce baskisi rho={self._rho:.2f}",
            tried=f"{ctx.step} adim; {self._engellenen} kez erken cevap engellendi",
            found="bu katman DURDURMAZ, eylem secimine karisir — sert tavan ayri",
            next_step="bol butcede kazanc erir; kitlik altinda anlamli",
        )
