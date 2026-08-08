"""
Google gemini-cli (google-gemini/gemini-cli) — bayat browser snapshot'ını supersede.  [§1.7]

Kaynak: agents/local-executor.ts (`onBeforeTurn`, `supersedeStaleSnapshots`,
        `SNAPSHOT_SUPERSEDED_PLACEHOLDER`), core/client.ts (`tryCompressChat`,
        `COMPRESSION_FAILED_INFLATED_TOKEN_COUNT`).

Tool-trace işi: `onBeforeTurn` genel kanca (model çağrısından önce geçmişi değiştir).
Shipped somut kullanım browser subagent'ına özel: `supersedeStaleSnapshots` — eski
`take_snapshot` çıktılarını (her biri tam accessibility tree; sadece EN GÜNCEL sayfa
anlamlı) placeholder'la değiştirir. Yani BELİRLİ BİR TOOL için staleness — Cline'ın
dosya-okuma dedup'ının browser-snapshot muadili (tezimizi destekleyen ikinci örnek).
`tryCompressChat` ayrı/genel; özet ham'dan büyürse INFLATED → iptal (fayda freni).
"""
from __future__ import annotations

from harness import ToolResult, Conversation, est
from .base import Strategy, Fate, summarize

# gemini-cli sabiti — birebir placeholder.
SNAPSHOT_SUPERSEDED_PLACEHOLDER = (
    "[Snapshot superseded — a newer snapshot exists later in this conversation.]")


def supersedeStaleSnapshots(results: list[ToolResult]) -> int:
    """En güncel take_snapshot hariç eskilerini bayat sayıp placeholder'la değiştir (DET, stale)."""
    snaps = [r for r in results if r.tool_type == "take_snapshot"]
    superseded = 0
    for r in snaps[:-1]:  # sonuncu (en güncel) hariç hepsi bayat
        r.view = SNAPSHOT_SUPERSEDED_PLACEHOLDER
        r.fate, r.note = Fate.SUPERSEDE, "supersedeStaleSnapshots"
        superseded += 1
    return superseded


class GeminiCliStrategy(Strategy):
    name = "gemini-cli"
    repo = "google-gemini/gemini-cli"
    ref = "§1.7"
    blurb = "onBeforeTurn: supersedeStaleSnapshots (eski take_snapshot bayat); tryCompressChat + INFLATED fren"
    uses_llm = True  # tryCompressChat genel context özeti

    def compact(self, results: list[ToolResult], conv: Conversation, budget: int) -> str:
        # onBeforeTurn kancası → supersedeStaleSnapshots (tool-trace kısmı, DET)
        supersedeStaleSnapshots(results)

        # (ayrı) tryCompressChat — hâlâ doluysa tüm konuşmayı özetle; INFLATED freni
        if self._over_budget(results, budget):
            recent = self._recent_ids(results)
            old = [r for r in results if r.fate == Fate.TAM and id(r) not in recent]
            if old:
                raw = sum(r.raw_tokens() for r in old)
                blob = "\n".join(f"{r.name}({r.resource}): {r.content[:150]}" for r in old)
                summ = summarize(blob, "gemini-cli tryCompressChat: konuşmayı özetle.")
                if est(summ) >= raw:  # COMPRESSION_FAILED_INFLATED_TOKEN_COUNT
                    return "[tryCompressChat: INFLATED → iptal (fayda freni; özet ham'dan büyük)]"
                for r in old:
                    r.view, r.fate, r.note = "", Fate.OZET, "tryCompressChat"
                return f"[tryCompressChat özeti — {len(old)} girdi]\n{summ}"
        return ""
