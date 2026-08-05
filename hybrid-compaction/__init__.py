"""
hybrid-compaction — Hermes + deterministik-ledger birleşimi trace-compaction framework'ü.

Public API:
    from hybrid_compaction import (
        Trace, ExecutionLedger, EpisodeGraph,     # çekirdek durum
        HybridCompactor, render_event,            # birleşik sıkıştırıcı + köprü
        tool_gist,                                # Hermes tarzı tip-farkında özet
    )

Kullanım (özet):
    trace = Trace(); ledger = ExecutionLedger(tool_meta=TOOL_META)
    # ... her tool çağrısında: trace.add_tool(...) + ledger.record(...)
    comp = HybridCompactor(budget=800, protect_window=3,
                           summarize_fn=None,     # LLM katmanı istersen ver
                           filter_safe=True)
    comp.compact(trace, ledger, episode_graph=episodes)
    # modele gönderirken: render_event(ev)  → TAM/ÖZET/SİL içeriği (prefix'li)
"""
from trace import Trace, Event, TraceSummary
from ledger import ExecutionLedger
from episode_graph import EpisodeGraph, Episode
from hybrid_compactor import HybridCompactor, render_event, FILTER_SAFE_PREFIX
from tool_summary import tool_gist

__all__ = ["Trace", "Event", "TraceSummary", "ExecutionLedger", "EpisodeGraph",
           "Episode", "HybridCompactor", "render_event", "FILTER_SAFE_PREFIX",
           "tool_gist"]
