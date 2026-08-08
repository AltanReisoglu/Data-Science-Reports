"""
OpenCode (sst/opencode) — tool çıktısını zaman-damgasıyla "gizleme".  [§1.9]

Kaynak: SDK — `SessionTime{Created, Updated, Compacting}`, `SessionCompacted` event,
        `session/{id}/summarize` endpoint.

Tool-trace işi: eski tool çıktısını mesaj listesinden SİLMEZ — bir Compacting ZAMAN
SINIRI koyar; o sınırdan eski içeriği modele giden bağlamdan HARİÇ TUTAR ama transcript
deposunda olduğu gibi kalır (revert/restore ile geri alınabilir). "Soft delete by time".
  * Adım 1 (DET): pruning 20K+ token açacaksa Compacting sınırından eskiyi GİZLE.
                  skill çıktıları asla gizlenmez; son 40K token + son 2 user turn korunur.
  * Adım 2 (LLM, gerekirse): gizlenmemiş kalanı özetle; son user mesajını replay et.
"""
from __future__ import annotations

from harness import ToolResult, Conversation, est
from .base import Strategy, Fate, summarize

# OpenCode korumaları (blog 📄). POC ölçeğine indirgendi (40K/20K yerine oransal).
PROTECT_LAST_USER_TURNS = 2


class OpenCodeStrategy(Strategy):
    name = "opencode"
    repo = "sst/opencode"
    ref = "§1.9"
    blurb = "Compacting timestamp'ten eskiyi GİZLE (depoda kalır, geri alınabilir); son 2 user turn koru; sonra LLM"
    uses_llm = True  # Adım 2

    def compact(self, results: list[ToolResult], conv: Conversation, budget: int) -> str:
        if not self._over_budget(results, budget):
            return ""
        # son 40K token + son 2 user turn korunur → burada konum-tabanlı (son N tool) vekili
        protected = self._recent_ids(results)

        # Adım 1 (DET) — Compacting timestamp sınırından eskiyi gizle (skill asla gizlenmez)
        hidden = []
        for r in results:
            if not self._over_budget(results, budget):
                break
            if id(r) in protected:
                continue  # son 40K token / son user turn korunur
            r.view = "[hidden by Compacting timestamp — transcript deposunda kalır (restore edilebilir)]"
            r.fate, r.note = Fate.GIZLE, "SessionTime.Compacting"
            hidden.append(r)

        # Adım 2 (LLM) — hâlâ doluysa gizlenmemiş kalanı özetle + son user mesajını replay
        if hidden and self._over_budget(results, budget):
            blob = "\n".join(f"{r.name}({r.resource}): {r.content[:120]}" for r in hidden)
            summ = summarize(blob, "OpenCode Adım-2: gizlenen tool çıktılarını özetle.")
            return (f"[SessionCompacted — {len(hidden)} çıktı gizlendi/özetlendi; son user mesajı replay]\n{summ}")
        if hidden:
            return f"[SessionCompacted — {len(hidden)} çıktı Compacting sınırından gizlendi (geri alınabilir)]"
        return ""
