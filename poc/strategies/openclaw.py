"""
OpenClaw (openclaw/openclaw) — oversized tool sonucu kesme (LLM'e gitmeden önce).  [§1.5]

Kaynak: tool-result-truncation, overflow-context-recovery.ts —
        `sessionLikelyHasOversizedToolResults`, `resolveLiveToolResultMaxChars`,
        `truncateOversizedToolResultsInActiveTarget`.

Tool-trace işi: bağlam taştığında, PAHALI LLM compaction'a başvurmadan ÖNCE en büyük
tool sonuçlarını canlı mesaj kümesinde kırpar. "Önce ucuz kurtarma dene" — çoğu
overflow birkaç dev çıktıdan (build log, dosya dökümü) gelir; onları kesmek çoğu
zaman compaction'a hiç gerek bırakmaz. Sadece oversized olanlar; küçükler dokunulmaz.
"""
from __future__ import annotations

from harness import ToolResult, Conversation, est
from .base import Strategy, Fate

# Bir tool sonucunun canlı bağlamda kalabileceği maks karakter (resolveLiveToolResultMaxChars).
LIVE_TOOL_RESULT_MAX_CHARS = 1100


def sessionLikelyHasOversizedToolResults(results: list[ToolResult]) -> bool:
    """Ucuz ön-kontrol: sınırı aşan tool sonucu var mı? (tam tarama değil, hızlı tahmin)"""
    return any(len(r.content) > LIVE_TOOL_RESULT_MAX_CHARS for r in results)


def resolveLiveToolResultMaxChars() -> int:
    return LIVE_TOOL_RESULT_MAX_CHARS


class OpenClawStrategy(Strategy):
    name = "openclaw"
    repo = "openclaw/openclaw"
    ref = "§1.5"
    blurb = "overflow'da SADECE oversized tool sonuçlarını kes; yetmezse context-engine (LLM)"
    uses_llm = True  # yetmezse context-engine devreye girer

    def compact(self, results: list[ToolResult], conv: Conversation, budget: int) -> str:
        if not self._over_budget(results, budget):
            return ""
        if not sessionLikelyHasOversizedToolResults(results):
            return ""  # ucuz ön-kontrol: kesecek bir şey yok
        max_chars = resolveLiveToolResultMaxChars()
        # truncateOversizedToolResultsInActiveTarget — sadece oversized olanları kes
        for r in results:
            if len(r.content) > max_chars:
                kept = r.content[:max_chars]
                r.view = f"{kept}\n[truncated: {len(r.content) - max_chars} chars over live limit]"
                r.fate, r.note = Fate.KES, "truncateOversizedToolResultsInActiveTarget"
        # hâlâ doluysa: pahalı context-engine'e (LLM/lossless/native) düşer
        if self._over_budget(results, budget):
            return "[overflow-context-recovery: hâlâ dolu → context-engine (LLM/lossless) devreye girer]"
        return ""
