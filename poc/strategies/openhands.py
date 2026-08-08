"""
OpenHands (OpenHands/software-agent-sdk) — Pipeline + ObservationMasking.  [§1.6]

Kaynak: condenser/ — `Pipeline`, `ObservationMaskingCondenser(attention_window=100)`,
        `BrowserOutputCondenser`, `RecentEventsCondenser`, `LLMSummarizingCondenser`.

Tool-trace işi: ajan döngüsü Event üretir — Action (karar) + Observation (tool çıktısı).
`ObservationMaskingCondenser` eski Observation gövdesini bir MASKE/placeholder ile değiştirir
(yapı durur, şişkin gövde gider, attention_window içindekiler maskelenmez). LLM çağırmaz.
`Pipeline` birden çok condenser'ı ZİNCİRLER — bizim faz-faz yaklaşımımıza benzer:
  BrowserOutput → ObservationMasking → (gerekirse) LLMSummarizing.
Lindenbauer bulgusu: masking ≈ özet kalitesi, ~yarı maliyet.
"""
from __future__ import annotations

from harness import ToolResult, Conversation
from .base import Strategy, Fate, summarize

# ObservationMaskingCondenser varsayılanı (olay sayısı). POC ölçeğinde küçük tutuldu.
ATTENTION_WINDOW = 4
_BROWSER_TYPES = {"take_snapshot", "web_extract"}


class BrowserOutputCondenser:
    """Verbose browser tool çıktısını özel temizle (tool-özel, DET)."""
    MAX_LINES = 6

    def condense(self, results: list[ToolResult]) -> None:
        for r in results:
            if r.tool_type in _BROWSER_TYPES and r.view is None:
                lines = r.content.splitlines()
                if len(lines) > self.MAX_LINES:
                    head = "\n".join(lines[:self.MAX_LINES])
                    r.view = f"{head}\n[BrowserOutputCondenser: {len(lines) - self.MAX_LINES} satır temizlendi]"
                    r.fate, r.note = Fate.KES, "BrowserOutputCondenser"


class ObservationMaskingCondenser:
    """Eski Observation (tool çıktısı) gövdesini placeholder'a çevir; attention_window korunur (DET)."""
    def __init__(self, attention_window: int = ATTENTION_WINDOW) -> None:
        self.attention_window = attention_window

    def condense(self, results: list[ToolResult], budget: int, strat: "OpenHandsStrategy") -> None:
        window = set(id(r) for r in results[-self.attention_window:])
        for r in results:
            if not strat._over_budget(results, budget):
                break
            if id(r) in window:
                continue
            n = len(r.content)
            r.view = f"<MASKED observation — {n} chars, attention_window dışı>"
            r.fate, r.note = Fate.MASKE, "ObservationMaskingCondenser"


class OpenHandsStrategy(Strategy):
    name = "openhands"
    repo = "OpenHands/software-agent-sdk"
    ref = "§1.6"
    blurb = "Pipeline: BrowserOutput → ObservationMasking(attention_window) → (gerekirse) LLMSummarizing"
    uses_llm = True  # son adım LLMSummarizingCondenser

    def compact(self, results: list[ToolResult], conv: Conversation, budget: int) -> str:
        # Pipeline — condenser'ları zincirle
        BrowserOutputCondenser().condense(results)
        ObservationMaskingCondenser(ATTENTION_WINDOW).condense(results, budget, self)

        # hâlâ doluysa: LLMSummarizingCondenser (TÜM olayları özetle — mor)
        if self._over_budget(results, budget):
            masked = [r for r in results if r.fate == Fate.MASKE]
            if masked:
                blob = "\n".join(f"{r.name}({r.resource})" for r in masked)
                summ = summarize(blob, "OpenHands LLMSummarizingCondenser: maskelenen gözlemleri özetle.")
                for r in masked:
                    r.view = ""
                return f"[LLMSummarizingCondenser — {len(masked)} maskeli gözlem özeti]\n{summ}"
        return ""
