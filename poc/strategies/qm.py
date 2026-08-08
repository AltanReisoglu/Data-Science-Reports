"""
QM (yc-software/qm) — bütün-geçmiş LLM özeti (tool-trace-farkında DEĞİL).  [§2]

Kaynak: core/orchestrator/compaction.ts — çift-tavan (MAX 400 girdi / 120K token),
        `contextSummaryPayload` başa eklenir.

Bu bir KARŞIT örnek: QM tool çıktısına ÖZEL davranmaz — tüm SessionEntry'leri
(diyalog + tool sonucu) AYNI sayıp birlikte LLM'e özetletir. Yani *context* compaction
yapar, *tool-trace* değil. "Aynı tool'u tekrar çağırdın" / "bu okuma bayat" kavramı YOK.
POC'a manzarayı tamamlamak için kondu — deterministik farkı göstermek için.
"""
from __future__ import annotations

from harness import ToolResult, Conversation, est
from .base import Strategy, Fate, summarize

# QM çift-tavanı — birebir sabitler.
MAX_ENTRIES = 400
MAX_TOKENS = 120_000


class QmStrategy(Strategy):
    name = "qm"
    repo = "yc-software/qm"
    ref = "§2 (tool-trace-farkında DEĞİL)"
    blurb = "bütün-geçmiş LLM özeti, çift-tavan (400 girdi/120K token), contextSummaryPayload başa"
    uses_llm = True

    def compact(self, results: list[ToolResult], conv: Conversation, budget: int) -> str:
        # çift-tavan (VEYA): girdi sayısı ya da token aşarsa. (POC'ta budget = token tavanı vekili.)
        over = (len(results) > MAX_ENTRIES or self._shown_tokens(results) > budget)
        if not over:
            return ""
        # tool çıktısı = diyalog: HEPSİNİ aynı say, son N'i koru, gerisini toplu özetle
        recent = self._recent_ids(results)
        old = [r for r in results if id(r) not in recent]
        if not old:
            return ""
        blob = "\n".join(f"{r.name}({r.resource}): {r.content[:120]}" for r in old)
        payload = summarize(blob, "QM contextSummaryPayload: tüm geçmişi (tool-trace-agnostik) özetle.")
        for r in old:
            r.view, r.fate, r.note = "", Fate.OZET, "contextSummaryPayload (tool-agnostik)"
        return f"[contextSummaryPayload — {len(old)} SessionEntry toplu özet (tool-trace-farkında değil)]\n{payload}"
