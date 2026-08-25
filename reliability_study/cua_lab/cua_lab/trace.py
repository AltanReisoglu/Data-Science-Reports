"""
İz kaydı — iterasyon başına bir span, JSONL.

Arize'ın gerekliliği: *"bütün koşum tek bir span ise, 4-19. adımların aynı
çağrı olduğunu göremezsin."* O yüzden her adım ayrı satır, ve her satırda
argüman hash'i + sonuç hash'i + ekran hash'i var — tekrar deseni ham veriden
görülebilsin.

Bu dosya iki işe yarıyor: `improvement-loop` stratejisinin girdisi ve sunumun
kanıtı.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class Span:
    i: int                       # iterasyon indeksi
    t: float                     # koşum başından beri geçen saniye
    action: str = ""
    args_hash: str = ""
    result_hash: str = ""
    screen_hash: str = ""
    error: str | None = None
    executed: bool = True
    tokens: int = 0
    cost_usd: float = 0.0
    verdict: str = "continue"
    verdict_reason: str = ""
    verdict_by: str = ""         # hangi strateji tetikledi
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "_written", False)


class TraceWriter:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.spans: list[Span] = []
        self._t0 = time.monotonic()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def write(self, span: Span) -> None:
        if getattr(span, "_written", False):
            return
        object.__setattr__(span, "_written", True)
        self.spans.append(span)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                d = {f.name: getattr(span, f.name) for f in fields(span)}
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # -- improvement-loop için ------------------------------------------

    def step_count(self) -> int:
        return len(self.spans)

    def totals(self) -> dict[str, float]:
        return {
            "steps": len(self.spans),
            "tokens": sum(s.tokens for s in self.spans),
            "cost_usd": round(sum(s.cost_usd for s in self.spans), 4),
            "seconds": round(self.spans[-1].t, 2) if self.spans else 0.0,
        }


def read_trace(path: str | Path) -> list[Span]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Span(**json.loads(line)))
    return out
