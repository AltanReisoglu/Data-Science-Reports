"""
Improvement Loop — eşiği tahmin etme, ölç.  KATEGORİ: KARAR · SEVİYE 6

Kaynak: OpenAI cookbook *agent_improvement_loop* + iki bağımsız kaynak

ZİHNİYET: Bütün diğer zihniyetler koşumun içinde müdahale ediyor. Bu HİÇ
etmiyor — sadece kaydediyor. Çünkü sorduğu soru farklı: BU EŞİKLERİ KİM, NEYE
BAKARAK KOYDU?

"Adım limiti 12" diye yazdığınızda o 12 nereden geldi? Çoğu zaman hiçbir
yerden. Birinin makul bulduğu bir sayı.

Ve yanlış seçilmiş bir limit SİNSİ bir hasar veriyor:
    "Meşru işi kesecek kadar dar bir adım limiti, modelin kötüleştiği gibi
     görünen sessiz bir kalite gerilemesine dönüşür."
Model aynıdır, limitiniz dardır, ama grafikte MODEL kötüleşmiş görünür.

ÇÖZÜM ÖLÇMEK: başarılı koşumların adım dağılımına bak, tavanı kuyruğunun üstüne
koy (p99). Sonra izle: koşumların yüzde kaçı limitte sonlanıyor? Bu oran
tırmanıyorsa görev zorluğu ya da araç güvenilirliği değişmiştir.

DİSİPLİN: döngü ayarları prompt'la BİRLİKTE sürümlenmeli ve bir terfi
kapısından geçmeli. Eşik değiştirmek, kod değiştirmekle aynı süreçten geçmeli.

Bu strateji hiçbir şeyi durdurmuyor — `--strategy improvement-loop` seçmek
`none` gibi davranır ama koşum sonunda ÖLÇÜLMÜŞ EŞİK ÖNERİSİ üretir. Sert bir
tavanla birlikte kullanılmalı:

    --strategy improvement-loop,arize-control
"""

from __future__ import annotations

from ...events import CONTINUE, Verdict
from ..base import StopReport, register
from ..kinds import DecisionStrategy


def _p(dizi: list[float], q: float) -> float:
    if not dizi:
        return 0.0
    s = sorted(dizi)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[i]


@register
class ImprovementLoop(DecisionStrategy):
    id = "improvement-loop"
    title = "Improvement Loop: esigi tahmin etme, olc"
    source = "OpenAI cookbook agent_improvement_loop (+2 kaynak)"
    mentality = "Bu kosumu kurtarma — esikleri olcerek koy"
    priority = 16
    why = (
        "Buradaki butun esikleri kim, neye bakarak koydu? Yanlis secilmis bir limit 'model kotulesti' gibi gorunen SESSIZ bir kalite gerilemesi uretir.")
    action = "Mudahale YOK; basarili kosum dagilimindan p99 esik onerisi uret"
    blind_spot = (
        "Bu kosumu kurtarmaz. Veri birikmesini bekler.")

    def observe(self, ctx) -> None:
        s = ctx.budget.state
        self._kayit.append({
            "adim": s.steps, "token": s.tokens, "cost": s.cost_usd,
            "sn": round(s.seconds, 3),
            "ilerleme": bool(ctx.last_screen_changed()),
        })

    def decide(self, ctx) -> Verdict:
        return CONTINUE          # MUDAHALE YOK — bu zihniyetin tanimi

    def snapshot(self) -> dict:
        k = getattr(self, "_kayit", [])
        ilerleyen = sum(1 for x in k if x.get("ilerleme"))
        return {"tur": "karar", "sinyal": [],
                "not": f"MUDAHALE ETMEZ · {len(k)} adim kaydedildi"
                       f" · ilerleme %{int(100 * ilerleyen / len(k)) if k else 0}"}

    def summary(self, reason: str, ctx) -> StopReport | None:
        if not self._kayit:
            return None
        adimlar = [k["adim"] for k in self._kayit]
        tokenlar = [k["token"] for k in self._kayit]
        ilerleyen = sum(1 for k in self._kayit if k["ilerleme"])
        oran = ilerleyen / len(self._kayit)

        # Tavani KUYRUGUN USTUNE koy — p99, ortalama degil.
        oneri_adim = int(_p([float(a) for a in adimlar], 0.99) * 1.2) + 1
        oneri_token = int(_p([float(t) for t in tokenlar], 0.99) * 1.2) + 1

        return StopReport(
            reason=reason,
            why="bu katman mudahale etmedi — kosumu olctu",
            tried=f"{len(self._kayit)} adim kaydedildi; ilerleme orani "
                  f"%{oran*100:.0f} ({ilerleyen}/{len(self._kayit)})",
            found=f"OLCULMUS ESIK ONERISI (p99 x1.2): max_steps={oneri_adim} - "
                  f"max_tokens={oneri_token}",
            next_step="esikleri prompt'la BIRLIKTE surumle ve terfi kapisindan "
                      "gecir — esik degistirmek kod degistirmektir",
        )
