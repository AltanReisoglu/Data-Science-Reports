"""
Roo-Code (RooCodeInc/Roo-Code) — dosya-okuma tool çıktılarını "katlama".  [§1.8]

Kaynak: core/condense/foldedFileContext — `generateFoldedFileContext`,
        `parseSourceCodeDefinitionsForFile` (tree-sitter), `manageContext`,
        `truncateConversation(messages, fracToRemove=0.5, taskId)` (non-destructive),
        `injectSyntheticToolResults`, `summarizeConversation`.

Tool-trace işi: eski dosya-okuma çıktılarını ham içerik yerine YAPISAL OUTLINE'a katlar
(bir IDE'nin kod-katlama özelliği gibi). "500 satır" yerine "şu fonksiyonlar/tanımlar var".
Fallback: non-destructive sliding-window — mesajları SİLMEZ, gizler (fracToRemove=0.5),
tool_call↔tool_result eşleşmesi injectSyntheticToolResults ile korunur.
Not: Cline'ın aksine duplicate file-read dedup'ı YOK — sadece fold.
"""
from __future__ import annotations

import re

from harness import ToolResult, Conversation
from .base import Strategy, Fate

fracToRemove = 0.5
_DEF_RE = re.compile(r"^\s*(def |class |async def |function |PORT = |[A-Z_]+ = )")


def parseSourceCodeDefinitionsForFile(content: str) -> list[str]:
    """tree-sitter muadili: tanım (def/class/sabit) satırlarını çıkar."""
    defs = []
    for ln in content.splitlines():
        if _DEF_RE.match(ln):
            defs.append(ln.strip().rstrip(":"))
    return defs


def generateFoldedFileContext(r: ToolResult) -> str:
    """Dosya okuma çıktısını outline'a katla (ham gövde atılır, tanım listesi kalır)."""
    defs = parseSourceCodeDefinitionsForFile(r.content)
    first = r.content.splitlines()[0] if r.content else r.resource
    outline = "\n".join(f"  · {d}" for d in defs[:20])
    return f"[folded: {first}]\n{outline or '  (tanım bulunamadı)'}"


class RooStrategy(Strategy):
    name = "roo"
    repo = "RooCodeInc/Roo-Code"
    ref = "§1.8"
    blurb = "eski dosya okumalarını tree-sitter outline'a KATLA; fallback non-destructive sliding-window"
    uses_llm = False  # fold + truncate DET (summarizeConversation opsiyonel/kapalı varsayıldı)

    def compact(self, results: list[ToolResult], conv: Conversation, budget: int) -> str:
        recent = self._recent_ids(results)
        # generateFoldedFileContext — eski dosya okumalarını katla
        for r in results:
            if not self._over_budget(results, budget):
                break
            if r.tool_type == "read_file" and id(r) not in recent:
                r.view, r.fate, r.note = generateFoldedFileContext(r), Fate.KATLA, "generateFoldedFileContext"

        # fallback: hâlâ doluysa non-destructive sliding-window truncation (fracToRemove=0.5)
        if self._over_budget(results, budget):
            candidates = [r for r in results if id(r) not in recent and r.fate != Fate.KATLA]
            n_remove = int(len(candidates) * fracToRemove)
            for r in candidates[:n_remove]:
                # SİLMEZ — gizler (truncationParent etiketi) + sentetik tool_result korur
                r.view = "[truncated — truncationParent; injectSyntheticToolResults ile eşleşme korunur]"
                r.fate, r.note = Fate.GIZLE, "truncateConversation(fracToRemove=0.5)"
        return ""
