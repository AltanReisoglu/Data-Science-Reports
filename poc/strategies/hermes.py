"""
Hermes (NousResearch/hermes-agent) — tip-farkında tek-satır tool özeti.  [§1.1]

Kaynak: context_compressor.py — `_summarize_tool_result`, `drop_stale_api_content`,
        `_truncate_tool_call_args_json`, `_strip_historical_media`, `_PRUNED_TOOL_PLACEHOLDER`.

Tool-trace işi: büyük tool çıktısını MODEL ÇAĞIRMADAN, tool tipine göre tek
bilgilendirici satıra indirir. LLM özetinden ÖNCE çalışan "ucuz ön-pas".
Kısaltılmış çıktı tool-result slotunda KALIR (rol değişmez), sadece gövde küçülür.
Deterministik — orijinalde testler tam string eşitliğiyle doğrular (LLM değil).
"""
from __future__ import annotations

from harness import ToolResult, Conversation
from .base import Strategy, Fate

# Hermes sabiti — birebir.
_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"


def _summarize_tool_result(name: str, tool_type: str, resource: str, content: str) -> str:
    """Hermes `_summarize_tool_result` — tip-farkında tek satır. Hata → `[tool] (N chars)` fallback."""
    n_lines = content.count("\n") + 1
    n_chars = len(content)
    if tool_type == "terminal":
        # son satırdan exit code çek (Hermes cmd → exit code, N lines biçimi)
        exit_code = 0
        for ln in reversed(content.splitlines()):
            if ln.strip().startswith("exit "):
                try:
                    exit_code = int(ln.strip().split()[1])
                except (ValueError, IndexError):
                    pass
                break
        return f"[terminal] {resource} → exit {exit_code}, {n_lines} lines"
    if tool_type == "web_extract":
        more = content.count("/sub/")  # ekstra linkler
        return f"[web_extract] {resource} (+{more} more) ({n_chars} chars)"
    if tool_type == "take_snapshot":
        return f"[snapshot] {resource} → {n_lines} nodes ({n_chars} chars)"
    if tool_type == "grep":
        return f"[grep] '{resource}' → {n_lines - 1} matches"
    # fallback — bilinmeyen tip
    return f"[tool] ({n_chars} chars)"


class HermesStrategy(Strategy):
    name = "hermes"
    repo = "NousResearch/hermes-agent"
    ref = "§1.1"
    blurb = "tip-farkında tek-satır tool özeti (deterministik ön-pas, LLM'den önce)"
    uses_llm = False

    def compact(self, results: list[ToolResult], conv: Conversation, budget: int) -> str:
        recent = self._recent_ids(results)
        # ucuz ön-pas: bütçe aşılıyken en eskiden başlayarak tek-satıra indir.
        # son N tool verbatim korunur (Hermes yakın bağlamı bozmaz).
        for r in results:
            if not self._over_budget(results, budget):
                break
            if id(r) in recent:
                continue
            gist = _summarize_tool_result(r.name, r.tool_type, r.resource, r.content)
            r.view, r.fate, r.note = gist, Fate.OZET, "_summarize_tool_result"
        return ""
