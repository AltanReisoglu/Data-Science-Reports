"""
Anthropic Claude Code — tool_result temizleme + koruma + state reconstruction.  [§1.4]

Kaynak: context editing / B.11 — eski tool sonuçları `[Old tool result content cleared]`
        placeholder'ıyla TEMİZLENİR; aktif/son tool çağrıları korunur. Compaction
        sonrası son düzenlenen dosyalar TEKRAR OKUNUR (state reconstruction).

Not: Codex "sil" (kalıcı) ≠ Claude Code "clear placeholder" (yerinde temizle,
yapı durur). Placeholder tool-result slotunda kalır → tool_call_id eşleşmesi bozulmaz.
"""
from __future__ import annotations

from harness import ToolResult, Conversation
from .base import Strategy, Fate

# Claude Code sabiti — birebir placeholder.
CLEARED_PLACEHOLDER = "[Old tool result content cleared]"

# aktif/son korunacak tool sonucu sayısı
KEEP_ACTIVE = 3


class ClaudeCodeStrategy(Strategy):
    name = "claude-code"
    repo = "anthropic/claude-code (context editing / B.11)"
    ref = "§1.4"
    blurb = "eski tool_result'ı '[Old tool result content cleared]' ile temizle; son N koru; state reconstruction"
    uses_llm = False

    def compact(self, results: list[ToolResult], conv: Conversation, budget: int) -> str:
        # aktif/son KEEP_ACTIVE sonucu koru, eskiler bütçe aşılıyken temizlenir
        protected = set(id(r) for r in results[-KEEP_ACTIVE:])
        for r in results:
            if not self._over_budget(results, budget):
                break
            if id(r) in protected:
                continue
            r.view, r.fate, r.note = CLEARED_PLACEHOLDER, Fate.SIL, "clearToolResult (B.11)"

        # state reconstruction — compaction sonrası son düzenlenen dosyaları tekrar oku
        edited = [r.resource for r in results if r.is_mutation]
        if edited and any(r.fate == Fate.SIL for r in results):
            last5 = edited[-5:]
            return (f"[state reconstruction] compaction sonrası son {len(last5)} "
                    f"düzenlenen dosya tekrar okundu: {', '.join(last5)}")
        return ""
