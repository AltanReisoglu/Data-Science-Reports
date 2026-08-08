"""
strategies/ — her rakip sistemin tool-trace compaction mantığı, kendi repo'suna BİREBİR sadık.

REGISTRY: isim → Strategy örneği. chat.py `/strategy <isim>` ile seçer; compare.py hepsini gezer.

Sıralama landscape §'lerini izler: §1 tool-trace-farkında (yeşil) → §2 farkında-değil (karşıt) → §3 bizim.
"""
from __future__ import annotations

from .base import Strategy, Fate

from .hermes import HermesStrategy
from .headroom import HeadroomStrategy
from .codex import CodexStrategy
from .claude_code import ClaudeCodeStrategy
from .openclaw import OpenClawStrategy
from .openhands import OpenHandsStrategy
from .gemini_cli import GeminiCliStrategy
from .roo import RooStrategy
from .opencode import OpenCodeStrategy
from .cline import ClineStrategy
from .swe_agent import SweAgentStrategy
from .qm import QmStrategy
from .ours import OursStrategy

# ekleme sırası = gösterim sırası
_ORDER = [
    HermesStrategy, HeadroomStrategy, CodexStrategy, ClaudeCodeStrategy,
    OpenClawStrategy, OpenHandsStrategy, GeminiCliStrategy, RooStrategy,
    OpenCodeStrategy, ClineStrategy, SweAgentStrategy,   # §1
    QmStrategy,                                          # §2 (karşıt)
    OursStrategy,                                        # §3 (bizim)
]

REGISTRY: dict[str, Strategy] = {}
for _cls in _ORDER:
    _inst = _cls()
    REGISTRY[_inst.name] = _inst


def get(name: str) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"bilinmeyen strateji: {name!r}. Seçenekler: {', '.join(REGISTRY)}")
    return REGISTRY[name]


def all_strategies() -> list[Strategy]:
    return list(REGISTRY.values())
