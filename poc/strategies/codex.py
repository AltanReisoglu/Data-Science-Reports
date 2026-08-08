"""
OpenAI Codex (openai/codex) — tool çıktısı orta-kesme + compaction'da silme.  [§1.3]

Kaynak: utils/output-truncation (`truncate_middle_chars`,
        `truncate_middle_with_token_budget`), `tool_output_token_limit`,
        `TruncationPolicy::Bytes/Tokens`, compact.rs (handoff özeti).

İki mekanizma:
  (a) INGEST — her tool çıktısı `tool_output_token_limit`'i aşıyorsa ORTASINDAN kesilir
      (baş+son kalır, "…N truncated…"). İlişkiye bakmaz, sadece boyut.
  (b) COMPACTION — bağlam taşınca eski tool çıktıları fiziksel SİLİNİR, tek bir
      LLM handoff özetine erir (kalıcı kayıp).
"""
from __future__ import annotations

from harness import ToolResult, Conversation, est
from .base import Strategy, Fate, summarize

# Codex bütçesi — birebir isim. (Token; POC ölçeğinde küçük tutuldu ki kesme görünsün.)
tool_output_token_limit = 260


def truncate_middle_chars(content: str, token_budget: int) -> str:
    """Codex `truncate_middle_chars` — ortayı at, baş+son kalır, '…N truncated…'."""
    char_budget = token_budget * 4
    if len(content) <= char_budget:
        return content
    keep = char_budget // 2
    head, tail = content[:keep], content[-keep:]
    dropped_lines = content[keep:-keep].count("\n")
    return f"{head}\n…{dropped_lines} lines truncated…\n{tail}"


class CodexStrategy(Strategy):
    name = "codex"
    repo = "openai/codex"
    ref = "§1.3"
    blurb = "tool çıktısını ortadan kes (tool_output_token_limit); taşınca sil→LLM handoff"
    uses_llm = True  # (b) compaction handoff özeti

    def compact(self, results: list[ToolResult], conv: Conversation, budget: int) -> str:
        # (a) INGEST — her çıktıya orta-kesme (boyut-bazlı, ilişkisiz)
        for r in results:
            truncated = truncate_middle_chars(r.content, tool_output_token_limit)
            if truncated != r.content:
                r.view, r.fate, r.note = truncated, Fate.KES, "truncate_middle_chars"

        # (b) COMPACTION — hâlâ taşıyorsa eski tool çıktılarını sil → tek handoff özeti
        recent = self._recent_ids(results)
        old = [r for r in results if id(r) not in recent]
        if self._over_budget(results, budget) and old:
            blob = "\n".join(f"{r.name}({r.resource}): {r.content[:200]}" for r in old)
            handoff = summarize(blob, "Codex compaction handoff: bu tool çıktılarını özetle.")
            for r in old:
                r.view, r.fate, r.note = "", Fate.SIL, "compact.rs → handoff"
            return f"[Codex handoff özeti — {len(old)} tool çıktısı silindi]\n{handoff}"
        return ""
