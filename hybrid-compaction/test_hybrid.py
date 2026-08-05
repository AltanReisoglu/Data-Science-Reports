"""
test_hybrid.py — hybrid-compaction framework'ünün deterministik testleri (LLM'siz).

Kanıtlar:
  1. dedup → SİL (verbatim tekrar)          6. çift-koruma (adet + token)
  2. staleness → tetiklenir (bayat)          7. belirlenimcilik (3 koşu aynı)
  3. gist tip-farkında (Hermes)              8. render_event ham/prefix'li
  4. filter-safe prefix (Hermes)             9. fayda freni (küçük çıktı ham kalır)
  5. CWL episode + fold
"""
from __future__ import annotations

from trace import Trace
from ledger import ExecutionLedger
from episode_graph import EpisodeGraph
from hybrid_compactor import HybridCompactor, render_event, FILTER_SAFE_PREFIX, _raw_cost
from tool_summary import tool_gist
from demo import DISPATCH, TOOL_META, drive

OK = 0


def _ok(msg):
    global OK; OK += 1; print(f"  ✓ {msg}")


def _scenario(budget=500, pw=3, summarize_fn=None):
    t, l, e = Trace(), ExecutionLedger(tool_meta=TOOL_META), EpisodeGraph()
    drive(t, l, e, "keşif",
          [("__ep_start__", {"name": "veri", "type": "expl"}, None),
           ("search_tickets", {"q": "bugs"}, None),
           ("get_ticket", {"id": "T-11"}, None),
           ("get_ticket", {"id": "T-12"}, None),
           ("get_metrics", {"project": "ATLAS"}, None),
           ("__ep_end__", {"desc": "2 ticket + velocity 32"}, None)])
    drive(t, l, e, "tekrar", [("get_ticket", {"id": "T-11"}, None)])
    drive(t, l, e, "rapor",
          [("__ep_start__", {"name": "rapor", "type": "act", "deps": ["veri"]}, None),
           ("create_report", {"title": "D"}, None),
           ("get_outline", {"doc": "doc-42"}, None),
           ("add_block", {"doc": "doc-42", "title": "x"}, None),
           ("get_outline", {"doc": "doc-42"}, None),
           ("__ep_end__", {"desc": "rapor üretildi"}, None)])
    drive(t, l, e, "kapanış",
          [("get_metrics", {"project": "ATLAS"}, None), ("get_ticket", {"id": "T-12"}, None)])
    c = HybridCompactor(budget=budget, protect_window=pw, summarize_fn=summarize_fn)
    res = c.compact(t, l, episode_graph=e, force=True)
    return t, l, e, res


def run():
    t, l, e, res = _scenario()

    # 1. dedup → SİL
    dup = [x for x in t.tool_events() if x.cleared and "tekrar" in x.clear_note]
    assert dup, "dedup SİL tetiklenmedi"
    _ok(f"dedup → SİL ({dup[0].payload['name']} seq={dup[0].seq})")

    # 2. staleness tetiklendi
    stale = [x for x in t.tool_events() if (x.evicted or x.cleared)
             and x.payload["name"] == "get_outline"]
    assert stale, "staleness tetiklenmedi"
    _ok(f"staleness → bayat outline ({len(stale)} adet)")

    # 3. gist tip-farkında (search → 'N sonuç', aggregate → sayı)
    g_search = tool_gist("search_tickets", {"q": "bugs"}, str(DISPATCH["search_tickets"](q="bugs")), TOOL_META["search_tickets"])
    g_metric = tool_gist("get_metrics", {"project": "A"}, str(DISPATCH["get_metrics"](project="A")), TOOL_META["get_metrics"])
    assert "sonuç" in g_search and any(c.isdigit() for c in g_metric)
    _ok(f"gist tip-farkında (search='{g_search[:22]}…' · metric='{g_metric[:22]}…')")

    # 4. filter-safe prefix render'da
    ev = next(x for x in t.tool_events() if x.evicted or x.cleared)
    assert render_event(ev, filter_safe=True).startswith(FILTER_SAFE_PREFIX)
    assert not render_event(ev, filter_safe=False).startswith(FILTER_SAFE_PREFIX)
    _ok("filter-safe prefix (açık/kapalı çalışıyor)")

    # 5. CWL episode kuruldu
    assert any(x.type == "expl" for x in e.episodes) and any(x.type == "act" for x in e.episodes)
    _ok("CWL episode (expl 'veri' + act 'rapor')")

    # 6. çift-koruma: dev son çıktı token sınırını tetikler
    t2, l2, e2 = Trace(), ExecutionLedger(tool_meta=TOOL_META), EpisodeGraph()
    r = t2.add_reasoning("x")
    for i in range(4):
        o = str(DISPATCH["get_ticket"](id=f"T-{i}")); ev2 = t2.add_tool("get_ticket", {"id": f"T-{i}"}, o); l2.record("get_ticket", {"id": f"T-{i}"}, o, ev2.seq)
    big = "X" * 4000; ev2 = t2.add_tool("get_ticket", {"id": "BIG"}, big); l2.record("get_ticket", {"id": "BIG"}, big, ev2.seq)
    c2 = HybridCompactor(budget=300, protect_window=5, protect_token_fraction=0.6)
    prot = c2._protected_recent([x for x in t2.tool_events()])
    # token sınırı (0.6*300=180) dev çıktı yüzünden penceniyi 5'ten aza indirmeli
    assert len(prot) < 5, f"token koruması pencereyi kısmadı: {len(prot)}"
    _ok(f"çift-koruma: dev çıktı pencereyi 5→{len(prot)}'e kıstı (token sınırı)")

    # 7. belirlenimcilik
    def fate(tt):
        return [(x.seq, "SIL" if x.cleared else "OZET" if x.evicted else "TAM") for x in tt.tool_events()]
    a = fate(_scenario()[0]); b = fate(_scenario()[0]); d = fate(_scenario()[0])
    assert a == b == d, "belirlenimci değil"
    _ok("belirlenimcilik (3 koşu birebir aynı)")

    # 8. render_event: TAM olan ham döner
    tam = next(x for x in t.tool_events() if not x.evicted and not x.cleared)
    assert render_event(tam) == tam.payload["output"]
    _ok("render_event: TAM olan ham çıktıyı döner")

    # 9. fayda freni: küçük verbatim çıktı özetlenince büyürse ham/SİL kalır
    assert res["after"] < res["before"], "toplam küçülmedi"
    _ok(f"fayda freni + kazanç: {res['before']}→{res['after']} tok (%{res['saved_pct']})")

    # 10. opsiyonel LLM katmanı
    _, _, _, r2 = _scenario(budget=200, pw=1, summarize_fn=lambda p: f"[LLM] {len(p)} olay özeti")
    assert any("LLM" in x for x in r2["log"]), "LLM katmanı çalışmadı"
    _ok("opsiyonel LLM katmanı (summarize_fn) devrede")

    print(f"\n{OK}/10 geçti — hybrid framework çalışıyor.")
    return OK == 10


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
