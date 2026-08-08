"""
Cline (cline/cline) — duplicate/stale DOSYA OKUMASI kaldırma (bize EN yakın).  [§1.10]

Kaynak: `formatResponse.duplicateFileReadNotice()`,
        `getNextTruncationRange(history, deletedRange, "quarter"|"half")`,
        `contextTruncationNotice()`, `CONTEXT_WINDOW_WARNING_THRESHOLD_PERCENT = 50`,
        `summarize_task (basic|agentic)`.

Tool-trace işi: aynı dosya birden çok okunduğunda ESKİ okumayı kaldırıp yerine
"en güncele bak" notu koyar → dedup + staleness (deterministik, LLM'siz). Sürüm
sayacı yok ama "son okuma tazedir" kuralı bizim staleness'imizin dosya-özel hali.
Üç katman: dosya-okuma dedup → sliding-window (quarter/half) → summarize_task (LLM).
"""
from __future__ import annotations

from harness import ToolResult, Conversation
from .base import Strategy, Fate, summarize

CONTEXT_WINDOW_WARNING_THRESHOLD_PERCENT = 50


def duplicateFileReadNotice() -> str:
    """Cline notu — birebir."""
    return ("[[NOTE] This file read has been removed to save space. "
            "Refer to the latest file read for the most up to date version.]")


class ClineStrategy(Strategy):
    name = "cline"
    repo = "cline/cline"
    ref = "§1.10"
    blurb = "aynı dosyanın eski okumasını kaldır ('en güncele bak'); sonra sliding-window; sonra summarize_task"
    uses_llm = True  # son çare summarize_task

    def compact(self, results: list[ToolResult], conv: Conversation, budget: int) -> str:
        # 1) duplicateFileReadNotice — aynı dosyanın son okuması hariç eskileri kaldır (dedup+stale)
        last_read: dict[str, ToolResult] = {}
        for r in results:
            if r.tool_type == "read_file":
                last_read[r.resource] = r  # sonuncusu en güncel
        for r in results:
            if r.tool_type == "read_file" and last_read.get(r.resource) is not r:
                r.view, r.fate, r.note = duplicateFileReadNotice(), Fate.DEDUP, "duplicateFileReadNotice"

        # 2) getNextTruncationRange — hâlâ doluysa eski mesajların çeyreğini/yarısını at
        if self._over_budget(results, budget):
            recent = self._recent_ids(results)
            old = [r for r in results if id(r) not in recent and r.fate == Fate.TAM]
            frac = 0.5 if self._shown_tokens(results) > budget * 1.5 else 0.25  # "half" vs "quarter"
            n = int(len(old) * frac)
            for r in old[:n]:
                r.view, r.fate = "[contextTruncationNotice: bu aralık kırpıldı]", Fate.SIL
                r.note = f"getNextTruncationRange({'half' if frac == 0.5 else 'quarter'})"

        # 3) summarize_task — hâlâ doluysa LLM condense (son çare)
        if self._over_budget(results, budget):
            recent = self._recent_ids(results)
            rest = [r for r in results if r.fate == Fate.TAM and id(r) not in recent]
            if rest:
                blob = "\n".join(f"{r.name}({r.resource})" for r in rest)
                summ = summarize(blob, "Cline summarize_task (agentic): görevi özetle.")
                for r in rest:
                    r.view, r.fate, r.note = "", Fate.OZET, "summarize_task"
                return f"[summarize_task — {len(rest)} girdi]\n{summ}"
        return ""
