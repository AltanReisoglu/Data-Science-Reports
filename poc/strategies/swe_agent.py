"""
SWE-agent (SWE-agent/SWE-agent) — history_processor'larla gözlem eleme.  [§1.11]

Kaynak: history_processors.py — `DefaultHistoryProcessor`, `LastNObservations(n=5)`,
        `ClosedWindowHistoryProcessor`, `TagToolCallObservations`,
        `CacheControlHistoryProcessor`.

Tool-trace işi: observation = environment/tool çıktısı. Pluggable history_processor'lar
(OpenHands condenser'larına benzer) tool çıktılarını DETERMİNİSTİK işler. `LastNObservations`
"en klasik processor, orijinal makalede kullanıldı" — son n gözlem hariç hepsini eler;
elenen yerine "Old environment output: (n lines omitted)" konur. İlişki değil, KONUM (son N).
"""
from __future__ import annotations

from harness import ToolResult, Conversation
from .base import Strategy, Fate

# LastNObservations varsayılanı n=5; POC ölçeğinde görünür olsun diye 3.
N = 3


class LastNObservations:
    """Son n gözlem hariç hepsini ele; yerine 'Old environment output: (k lines omitted)'."""
    def __init__(self, n: int = N) -> None:
        self.n = n

    def process(self, results: list[ToolResult]) -> None:
        keep = set(id(r) for r in results[-self.n:])
        for r in results:
            if id(r) in keep:
                continue
            omitted = r.content.count("\n") + 1
            r.view = f"Old environment output: ({omitted} lines omitted)"
            r.fate, r.note = Fate.SIL, f"LastNObservations(n={self.n})"


class SweAgentStrategy(Strategy):
    name = "swe-agent"
    repo = "SWE-agent/SWE-agent"
    ref = "§1.11"
    blurb = "history_processor: LastNObservations — son n gözlem hariç ele ('(n lines omitted)')"
    uses_llm = False

    def compact(self, results: list[ToolResult], conv: Conversation, budget: int) -> str:
        # LastNObservations — konum-tabanlı (bütçe aşımından bağımsız çalışır, DET)
        LastNObservations(N).process(results)
        return ""
