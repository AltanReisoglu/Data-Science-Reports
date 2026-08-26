"""
Lütuf bütçesi — limit dolunca kes, ama toparlanma turu ver.
KATEGORİ: SAYAÇ · SEVİYE 1 · İKİ VARYANT

Kaynaklar:
  agentscope  `agentscope` — EXCEED_MAX_ITERS + `tool_choice` kilidi (kod okundu)
  hermes      Hermes `agent/agent_init.py:986–991` (kod yorumu birebir okundu)

ZİHNİYET: Sert bir tavan ajanı işin ortasında keser ve o ana kadar yaptığı iş
ÇÖPE GİDER. Ama "lütfen bitir" demek de yetmez — model dinlemeyebilir. Çözüm:
limit dolunca koşumu bitirme, ek tur ver, AMA ARAÇ SEÇİMİNİ KİLİTLE. Nazik bir
rica değil, MEKANİK bir kısıt: bitirmekten başka seçenek yok.

NEDEN İKİ AYRI STRATEJİ DEĞİL — ÖLÇÜLDÜ:
    5 senaryo x 5 model = 25 kombinasyonun 25'inde BIREBIR ayni cikti verdiler.
    (durum, sebep, adim, token — dordu de ayni)

İkisi de aynı mekanizmayı kullanıyor: `grace_turns` + `lock_tools_in_grace`.
Ayrıştıkları tek yer UYARI EKSENİ, ve bizim senaryolarımız süre eksenini hiç
zorlamıyor (koşumlar 0,0 sn sürüyor) — o yüzden fark gözlenemiyor. Ayırt
edilemeyen iki şeyi iki ayrı "zihniyet" diye sunmak yanıltıcı olur.

VARYANTLARIN GERÇEK FARKI (kaynaklarda belgelenmiş, burada korunuyor):

    agentscope   5 lutuf turu · hicbir eksende uyari yok
    hermes       1 lutuf turu · ADIM ekseninde uyari YOK ama SURE ekseninde %80'de VAR

Hermes'in eksen ayrımı ölçülmüş bir bulguya dayanıyor:
    "Ara basınç uyarıları yok — modelleri karmaşık görevlerde erken pes
     ettiriyordu."
Yani "adımın azalıyor" mesajı modele "bu görev bana göre değil" gibi geliyor.
Ama aynı sistem SÜRE ekseninde uyarıyor.

Bu, `claude-advisory` ile birlikte okunmalı — belgedeki en öğretici karşıtlık.
Farkı GÖRMEK için süre eksenini zorlayan bir koşum gerekir:

    --strategy budget-grace:hermes --model hf --max-seconds 20
"""

from __future__ import annotations

from ...events import CONTINUE, Verdict
from ..base import register
from ..kinds import BudgetStrategy


@register
class BudgetGrace(BudgetStrategy):
    id = "budget-grace"
    title = "Lutuf butcesi: kes ama toparlanma turu ver"
    source = "agentscope EXCEED_MAX_ITERS + Hermes agent_init.py:986"
    mentality = "Nazikce isteme, seceneksiz birak — arac kilitli lutuf turu"
    priority = 2
    why = (
        "Sert tavan ajani isin ortasinda keser ve o ana kadarki is COPE GIDER. Bu katman ayni tavani korurken cikisa bir rampa ekler — kullanici bos elle kalmaz.")
    action = "Limit dolunca 1-5 ek tur + ARAC SECIMI KILITLI; ajan yalniz cevap uretebilir"
    blind_spot = (
        "Lutuf turlari da para yakar; cok uzun tutulursa tavanin anlami kalmaz.")
    family = "harness"

    hard = False
    reserve = 0.0
    lock_tools_in_grace = True

    variants = {
        # AgentScope: bes tur, hic uyari yok.
        "agentscope": {"grace_turns": 5, "warn_axes": ()},
        # Hermes: tek tur; ADIM ekseninde uyari YOK, SURE ekseninde var.
        "hermes": {"grace_turns": 1, "warn_axes": ("seconds",), "warn_at": 0.80},
    }

    def warn_text(self, eksen: str, kalan: float, limit) -> str:
        return f"SURE: {kalan:.0f}sn kaldi — elindekiyle topla."

    def on_grace_start(self, eksen: str, ctx) -> Verdict:
        if self.variant == "hermes":
            ctx.note("ITERASYON BUTCESI DOLDU. Bu son cagri — nihai cevabini ver.")
        else:
            ctx.note(f"LIMIT DOLDU ({eksen}). {self.grace_turns} tur kaldi ve arac "
                     f"cagrisi KAPALI — elindekiyle nihai cevabi ver.")
        return CONTINUE

    def why_text(self, reason: str) -> str:
        if self.variant == "hermes":
            return (f"{reason} — adim ekseninde hic ara uyari verilmedi "
                    f"(olculmus: uyari modeli erken pes ettiriyor)")
        return (f"{reason} — {self.grace_turns} turluk lutuf butcesi de doldu; "
                f"o turlarda arac secimi kilitliydi")
