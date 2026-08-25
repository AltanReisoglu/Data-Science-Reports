"""
AutoGen — koşum başlamadan yakala.  KATEGORİ: ŞEKİL · SEVİYE 4

Kaynak: AutoGen `GraphFlow` — graf doğrulaması (kod okundu)

ZİHNİYET: En uç nokta. Ajan akışının grafiği KURULURKEN doğrulanıyor: bir
çevrim var ve çıkış koşulu yoksa sistem HİÇ BAŞLAMIYOR.

    Cycle detected without exit condition

Çalışma zamanında sıfır maliyet, sıfır yanlış pozitif — çünkü çalışma zamanı
hiç gelmedi. Bu, kodu derlemekle çalıştırmak arasındaki fark: derleyici tip
hatalarını yakalar, sonsuz döngüleri yakalamaz.

SINIRI net: yalnızca YAPISAL döngüleri görüyor. Grafiğin şekli kusursuz olabilir
ve model yine de aynı düğümde takılabilir. Model kararına bağlı döngüleri
çalışma anındaki pencere dedektörleri (5-8, 15) ve sayaçlar (1, 3) yakalamak
zorunda.

BU POC'DEKİ KARŞILIĞI: bizim akışımız serbest bir ReAct döngüsü, önceden bilinen
bir graf değil. O yüzden burada graf, BÜTÇE EKSENLERİNDEN üretiliyor: hiçbir
eksende etkili sınır yoksa döngünün çıkış koşulu yok demektir — koşum
başlatılmıyor. IAL-SCAN'in "etkili sınır" sorusunun çalıştırılabilir hâli.
"""

from __future__ import annotations

from ...events import CONTINUE, Action, Verdict
from ..base import register
from ..kinds import ShapeStrategy


@register
class AutoGenStatic(ShapeStrategy):
    id = "autogen-static"
    title = "AutoGen: kosum oncesi graf dogrulamasi"
    source = "AutoGen GraphFlow — Cycle detected without exit condition"
    mentality = "Kosum baslamadan yakala — yapisal cevrim, cikis kosulu yok"
    priority = 14
    why = (
        "En ucuz kontrol: kosum hic baslamadan yapiyi dogrular. Calisma zamaninda sifir maliyet, sifir yanlis pozitif — cunku calisma zamani hic gelmedi.")
    action = "Cikis kosulu olmayan cevrim varsa SISTEMI BASLATMA"
    blind_spot = (
        "Yalniz YAPISAL donguleri gorur; modelin ayni dugumde takilmasini goremez.")
    family = "harness"
    terminal = "DEGRADED"

    def validate(self, ctx) -> Verdict:
        """Döngünün ETKİLİ bir çıkış koşulu var mı?"""
        l = ctx.budget.limits
        eksenler = {
            "max_steps": l.max_steps, "max_replans": l.max_replans,
            "max_tokens": l.max_tokens, "max_seconds": l.max_seconds,
            "max_cost_usd": l.max_cost_usd,
        }
        etkili = {k: v for k, v in eksenler.items() if v is not None}
        if not etkili:
            return Verdict(Action.STOP, "cycle_without_exit_condition",
                           {"eksenler": list(eksenler), "etkili": []})
        ctx.extra["autogen_exit"] = sorted(etkili)
        return CONTINUE

    def reasons(self):
        return ("cycle_without_exit_condition",)

    def why_text(self, reason: str, ctx) -> str:
        return ("graf dogrulamasi: agentik geri besleme yolunun ETKILI bir cikis "
                "kosulu yok — hicbir butce ekseni acik degil. Kosum baslatilmadi.")
