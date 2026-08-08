"""
engines.py — canlı ajan motoru seçici.

VARSAYILAN: LangGraph (generic tool seti). Yedek: manuel loop (agent.py).
  - generic + LangGraph kurulu   → LangGraphAgent   (VARSAYILAN motor)
  - product (119 gerçek tool)    → ToolUseAgent (manuel) — 119 tool'u LangChain'e
                                    çevirmek yerine kanıtlı manuel loop kullanılır
  - LangGraph yoksa / hata       → ToolUseAgent (manuel) yedek

İkisi de AYNI oturum arayüzünü sunar (conv/strategy/budget/last_preamble/set_strategy/send)
ve AYNI strategies/ katmanını kullanır.
"""
from __future__ import annotations

from providers import get_provider


def make_live_agent(strategy, budget: int, toolset: str = "generic", verbose: bool = False):
    """Canlı ajan + kullanılan motorun adını döndür: (agent, engine_name)."""
    provider = get_provider(toolset)
    # VARSAYILAN: generic yolda LangGraph
    if toolset == "generic":
        try:
            from langgraph_agent import LangGraphAgent, HAVE_LANGGRAPH
            if HAVE_LANGGRAPH:
                return LangGraphAgent(strategy, budget, verbose=verbose, provider=provider), "langgraph"
        except Exception:
            pass  # kurulu değil / hata → manuel loop'a düş
    # product veya LangGraph yok → manuel loop
    from agent import ToolUseAgent
    return ToolUseAgent(strategy, budget, verbose=verbose, provider=provider), "manuel"
