"""
run_equity.py — HF dataset case 5.8'i (XOM, T5) trace compaction sisteminden geçirir.

  python run_equity.py            (deterministik, API'siz — mock tool'lar)
  python run_equity.py --live     (gerçek Gemma ajanı; .env gerekir)

Amaç: "sistem bu case'i yapabilir mi" sorusunun çalışan cevabı. Mock tool'larla
ajan orkestrasyonu ve trace GERÇEK; genel ledger (TOOL_META) ticker kaynağına
oturuyor; compaction bu gerçek T5 araştırma trace'inde çalışıyor.
"""
from __future__ import annotations
import argparse

from trace import Trace
from ledger import ExecutionLedger
from compactor import TraceCompactor
from playbook import Playbook
from episode_graph import EpisodeGraph
from equity_tools import DISPATCH, TOOL_META

BAR = "─" * 70
# dataset case 5.8
QUERY = ("XOM için kapsamlı yatırım raporu: iş profili, gelir trendi, temel oranlar, "
         "teknik göstergeler, analist görüşü + enerji dönüşümü haberleri → görsel özet")
TICKER = "XOM"


def build_case() -> tuple[Trace, ExecutionLedger, EpisodeGraph]:
    """Ajanın case 5.8'i gerçekçi gürültüyle çözdüğü trace (mock tool'larla).

    Gürültü: fiyatı önce çeker (volatil), oranları iki kez çeker (dup), sonra
    veriyi grafiğe sentezler. tool_dag {A..F}→G episode yapısına dökülür.
    """
    t = Trace()
    L = ExecutionLedger(tool_meta=TOOL_META)   # GENEL sözleşme (ticker kaynağı + TTL)
    g = EpisodeGraph()

    def call(name, args, verbatim=False):
        r = t.add_reasoning(f"{name}({args}) çağıracağım")
        out = str(DISPATCH[name](**args))
        ev = t.add_tool(name, args, out, intent_ref=r.seq, verbatim=verbatim)
        L.record(name, args, out, ev.seq)
        g.attach(ev.seq)
        return ev

    # fiyatı erken çeker (volatil, ttl=1) — sonra çok adım geçince bayatlayacak
    g.start("veri-toplama", "expl", seq=t._seq)
    call("get_stock_price", {"ticker": TICKER})
    call("get_company_info", {"ticker": TICKER})
    call("get_income_statements", {"ticker": TICKER, "freq": "annual"}, verbatim=True)
    call("get_key_financial_ratios", {"ticker": TICKER}, verbatim=True)
    call("get_technical_indicators", {"ticker": TICKER})
    call("get_analyst_recommendations", {"ticker": TICKER}, verbatim=True)
    call("get_key_financial_ratios", {"ticker": TICKER})          # DUP — tekrar çekti
    call("web_search", {"query": "ExxonMobil XOM energy transition strategy news"})
    g.end(seq=t._seq, description="XOM verisi toplandı: gelir/oran/teknik/analist/haber")

    # sentez (act) — veri-toplamaya bağlı
    g.start("rapor", "act", seq=t._seq, dependencies=["veri-toplama"])
    call("visualize_data", {"instruction": "XOM finansal performans özet grafiği"}, verbatim=True)
    g.end(seq=t._seq)
    return t, L, g


def demo():
    t, L, g = build_case()
    print(BAR)
    print("EQUITY CASE 5.8 (XOM, T5) — trace compaction sistemi (deterministik)")
    print(BAR)
    print(f"Sorgu: {QUERY}")
    print(f"tool_dag: {{A,B,C}}∥{{D,E,F}}→G  ·  episode: {g.summary()}\n")

    tool_evs = t.tool_events()
    print(f"Ham trace: {len(tool_evs)} tool birimi, {t.total_tokens()} token")
    print("Genel ledger tespitleri (ticker kaynağı + TTL):")
    for ev in tool_evs:
        tags = []
        if L.is_stale(ev.seq):
            tags.append("BAYAT (TTL/volatilite)")
        from compactor import _detect_duplicate
        d = _detect_duplicate(t, ev, L)
        if d is not None:
            tags.append(f"DUP≡seq{d}")
        if tags:
            print(f"  seq{ev.seq:>2} {ev.payload['name']:<28} → {', '.join(tags)}")
    print()

    pb = Playbook()
    comp = TraceCompactor(TRACE_BUDGET, 2, task=QUERY, playbook=pb)
    res = comp.compact(t, L, force=True, episode_graph=g)

    print("SIKIŞTIRMA:")
    for line in res["log"]:
        print(line)
    print(f"\n{'Öncesi':<10}{res['before']:>6} token   {'Sonrası':<10}{res['after']:>6} token"
          f"   Kazanç %{res['saved_pct']}")
    print(BAR)

    korunan = [e.seq for e in tool_evs if not e.evicted and not e.cleared]
    ozet = [e.seq for e in tool_evs if e.evicted]
    silinen = [e.seq for e in tool_evs if e.cleared]
    print(f"KORUNDU   : {korunan}")
    print(f"ÖZETLENDİ : {ozet}")
    print(f"SİLİNDİ   : {silinen}")
    if pb.active_bullets():
        print("\nPlaybook (öğrenilen):")
        for b in pb.active_bullets():
            print(f"  • {b.render()}")
    print(BAR)
    print("Not: fiyat (ttl=1) YAZMA olmadan zamanla bayatladı — genel ledger'ın")
    print("volatilite eskimesi. Dosya senaryosundaki sürüm-eskimesiyle aynı çekirdek.")


def live():
    from agent import TracingAgent
    import equity_tools
    from tools import SCHEMAS as _  # noqa
    print(BAR); print("EQUITY CASE 5.8 — CANLI (Gemma)"); print(BAR)
    ag = TracingAgent(compaction=True)
    # equity tool setini enjekte et
    ag_dispatch = equity_tools.DISPATCH
    import tools as tmod
    tmod.DISPATCH = equity_tools.DISPATCH
    tmod.SCHEMAS = equity_tools.SCHEMAS
    ag.ledger = ExecutionLedger(tool_meta=equity_tools.TOOL_META)
    try:
        ans = ag.run(QUERY)
    except RuntimeError as e:
        print(f"[canlı mod yok] {e}"); return
    print("\nRAPOR:", ans)
    print("METRİKLER:", ag.metrics)


TRACE_BUDGET = 350

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    live() if args.live else demo()
