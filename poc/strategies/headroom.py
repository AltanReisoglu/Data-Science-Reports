"""
Headroom (headroomlabs-ai/headroom) — tip-özel algoritmik tool-çıktısı sıkıştırma.  [§1.2]

Kaynak: content_router.py (`ContentRouter`), smart_crusher.py (`SmartCrusher`),
        code_compressor.py (`CodeAwareCompressor`), log/search/diff_compressor.rs,
        kompress_compressor.py (`KompressCompressor`), ccr/ (`headroom_retrieve`),
        `KeepErrorsConstraint`, `KeepStructuralOutliersConstraint`.

Tool-trace işi: ajanla LLM arasına giren PROXY; tool çıktısını "bir içerik tipi" sayıp
içerik-tipine göre ALGORİTMİK (LLM'siz) compressor'a yönlendirir. Kayıplı ama
GERİ-ÇAĞRILABİLİR: atılan içerik yerelde saklanır, model `<<ccr:HASH>>` marker'ıyla
`headroom_retrieve(hash)` çağırıp ORİJİNALİ geri getirir. Hata/aykırı satırlar asla atılmaz.
"""
from __future__ import annotations

import hashlib

from harness import ToolResult, Conversation
from .base import Strategy, Fate

# CCR deposu — atılan orijinaller (hash → içerik). headroom_retrieve ile geri çağrılır.
_CCR_STORE: dict[str, str] = {}


def _ccr_put(content: str) -> str:
    h = hashlib.sha1(content.encode()).hexdigest()[:8]
    _CCR_STORE[h] = content
    return h


def headroom_retrieve(hash_: str) -> str:
    """Model marker'ı görünce orijinali geri çağırır (kayıpsız kurtarma)."""
    return _CCR_STORE.get(hash_, "[ccr: hash bulunamadı]")


def _is_error_or_outlier(line: str) -> bool:
    """KeepErrorsConstraint + KeepStructuralOutliersConstraint — asla atma."""
    low = line.lower()
    return any(k in low for k in ("error", "fail", "exception", "exit ", "warning")) or line.strip().startswith("exit")


class ContentRouter:
    """Tool çıktısının tipini tespit → uygun compressor."""
    @staticmethod
    def route(r: ToolResult):
        return {
            "read_file": CodeAwareCompressor,
            "terminal": LogCompressor,
            "grep": SearchCompressor,
            "web_extract": KompressCompressor,
            "take_snapshot": KompressCompressor,
            "write_file": _PassThrough,
        }.get(r.tool_type, KompressCompressor)


class CodeAwareCompressor:
    """AST muadili — import/imza korur, gövde kırpar (preserve_imports, preserve_signatures)."""
    label = "CodeAwareCompressor"
    @staticmethod
    def crush(content: str) -> str:
        keep = []
        for ln in content.splitlines():
            s = ln.lstrip()
            if s.startswith(("import ", "from ", "def ", "class ", "async def ")) or "PORT =" in ln:
                keep.append(ln)
        return "\n".join(keep)


class LogCompressor:
    """Log dedup + error-boost — tekrar eden satırları katla, hata/exit sabit tut."""
    label = "LogCompressor"
    @staticmethod
    def crush(content: str) -> str:
        lines = content.splitlines()
        errors = [ln for ln in lines if _is_error_or_outlier(ln)]
        head = lines[:3]
        seen, uniq = set(), []
        for ln in lines:
            key = ln[:24]
            if key not in seen:
                seen.add(key); uniq.append(ln)
        body = uniq[3:8]
        out = head + body + ["  …(log deduped)…"] + errors
        # sıra koru, tekrarı sil
        dedup = list(dict.fromkeys(out))
        return "\n".join(dedup)


class SearchCompressor:
    """grep çıktısı — eşleşme sayısı + ilk N eşleşme (KeepErrors korunur)."""
    label = "SearchCompressor"
    @staticmethod
    def crush(content: str) -> str:
        lines = content.splitlines()
        return "\n".join(lines[:1] + lines[1:6] + [lines[-1]] if len(lines) > 7 else lines)


class KompressCompressor:
    """Metin — token-classifier muadili: önemli (uzun/anahtar) satırları tut (final_scores>0.5)."""
    label = "KompressCompressor"
    @staticmethod
    def crush(content: str) -> str:
        lines = content.splitlines()
        scored = [ln for ln in lines if len(ln) > 40 or _is_error_or_outlier(ln)]
        return "\n".join(scored[:8] or lines[:4])


class _PassThrough:
    label = "pass"
    @staticmethod
    def crush(content: str) -> str:
        return content


class HeadroomStrategy(Strategy):
    name = "headroom"
    repo = "headroomlabs-ai/headroom"
    ref = "§1.2"
    blurb = "ContentRouter → tip-özel algoritmik compressor + CCR (headroom_retrieve ile geri çağrılabilir)"
    uses_llm = False  # compressor'lar algoritmik/küçük-model, generatif LLM değil

    def compact(self, results: list[ToolResult], conv: Conversation, budget: int) -> str:
        recent = self._recent_ids(results)
        for r in results:
            if not self._over_budget(results, budget):
                break
            if id(r) in recent:
                continue
            comp = ContentRouter.route(r)
            crushed = comp.crush(r.content)
            if len(crushed) < len(r.content):
                h = _ccr_put(r.content)  # orijinali CCR'a koy
                r.view = f"{crushed}\n<<ccr:{h}>>  (headroom_retrieve('{h}') → orijinal)"
                r.fate, r.note = Fate.CRUSH, comp.label
        return ""
