"""
Claude Task Budgets — modele saatini göster.  KATEGORİ: SAYAÇ · SEVİYE 1

Kaynak: Anthropic — `output_config.task_budget`, beta `task-budgets-2026-03-13`

ZİHNİYET: Dışarıdan kesmek kaba. Modele kalan bütçesini göster, kendini
ayarlasın.

Hermes ile AYNI sayacı kullanıyor ve tam TERSİNİ savunuyor. Farkı tek düğmede:

  warn_axes = token + sure + dolar   Geri sayim prompt'a giriyor.

Kritik nokta: bu TAVSİYE, zorlama değil. Model, kesilmesi bitirilmesinden daha
zararlı olacak bir işin ortasındaysa bütçeyi aşabiliyor — o yüzden `hard=False`
ve rezerv yok. Sert tavan AYRI bir mekanizma olarak duruyor:

    --strategy claude-advisory,arize-control

Belgelenmiş yan etki: bütçe görev için açıkça yetersizse model işi hiç
denemiyor, agresif biçimde daraltıyor ya da erken duruyor. "Her ihtimale karşı
düşük tutayım" refleksi geri tepiyor.
"""

from __future__ import annotations

from ..base import register
from ..kinds import BudgetStrategy


@register
class ClaudeAdvisory(BudgetStrategy):
    id = "claude-advisory"
    title = "Claude Task Budgets: tavsiye niteliginde geri sayim"
    source = "Anthropic — output_config.task_budget"
    mentality = "Modele saatini goster — tavsiye, zorlama degil"
    priority = 3
    why = (
        "Disaridan kesmek kaba bir cozum. Modele kalan butcesini gostererek ONGORULEBILIR BIR INIS saglar — ajan isini kendi toparlar.")
    action = "Prompt'a geri sayim enjekte etme; TAVSIYE niteliginde, model asabilir"
    blind_spot = (
        "Zorlayici degil. Model uyariyi dikkate almazsa hicbir koruma saglamaz — sert tavanla birlikte kullanilmali.")

    hard = False
    reserve = 0.0
    warn_axes = ("tokens", "seconds", "cost_usd")
    warn_at = 0.75

    def warn_text(self, eksen: str, kalan: float, limit) -> str:
        birim = {"tokens": "token", "seconds": "saniye", "cost_usd": "dolar"}
        return (f"GERI SAYIM: {kalan:.0f} {birim.get(eksen, eksen)} kaldi "
                f"({limit} icinde). Isi toparlamaya basla — bu bir tavsiye, "
                f"gerekiyorsa devam edebilirsin.")

    def why_text(self, reason: str) -> str:
        return (f"{reason} — model geri sayimi gordu; bu katman durdurmak icin "
                f"degil ONGORULEBILIR INIS icin var")
