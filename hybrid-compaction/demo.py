"""
demo.py — hybrid-compaction uçtan uca (LLM'siz, kendi-yeterli).

Küçük bir mock tool seti kurar, çok adımlı bir senaryo sürer, ve birleşik
sıkıştırıcının hepsini tetiklediğini gösterir:
  dedup (SİL) · staleness (SİL/ÖZET) · keşif fold (tip-farkında gist) ·
  CWL episode · çift-koruma · filter-safe prefix · (opsiyonel) LLM katmanı.

  python demo.py
"""
from __future__ import annotations

from trace import Trace
from ledger import ExecutionLedger
from episode_graph import EpisodeGraph
from hybrid_compactor import HybridCompactor, render_event

# --- küçük mock tool dünyası -----------------------------------------------
_DB = {"ticket": {}, "doc": {"blocks": 1, "ver": 1}}


def _pad(head, n=6):
    body = [f"  · alan-{i+1}: değer={100+i*5}" for i in range(n)]
    return head + "\n" + "\n".join(body)


def get_ticket(id):
    return _pad(f"Ticket {id}:\n  status: Open\n  owner: Ayşe\n  points: 5")


def search_tickets(q):
    return _pad(f"search: '{q}' → 4 sonuç:\n  1. T-11\n  2. T-12\n  3. T-13", 4)


def get_metrics(project):
    return f"metrics: open=12 · done=30 · velocity=32 (proje={project})"


def get_outline(doc):
    return _pad(f"Outline {doc} (v{_DB['doc']['ver']}, {_DB['doc']['blocks']} blok):\n  [b1] Başlık", 6)


def add_block(doc, title):
    _DB["doc"]["ver"] += 1; _DB["doc"]["blocks"] += 1
    return f"OK: blok eklendi · {doc} → v{_DB['doc']['ver']}"


def create_report(title):
    return f"OK: rapor oluşturuldu · document_id=doc-42 (v1)"


DISPATCH = {"get_ticket": get_ticket, "search_tickets": search_tickets,
            "get_metrics": get_metrics, "get_outline": get_outline,
            "add_block": add_block, "create_report": create_report}

TOOL_META = {
    "get_ticket":    {"cat": "read",   "resource": lambda a: a["id"],     "ttl": 20, "verbatim": True},
    "search_tickets":{"cat": "search"},
    "get_metrics":   {"cat": "search", "verbatim": True},
    "get_outline":   {"cat": "read",   "resource": lambda a: a["doc"],    "ttl": 1},
    "add_block":     {"cat": "write",  "resource": lambda a: a["doc"]},
    "create_report": {"cat": "write"},
}


# --- senaryo sürücüsü (send() döngüsünün LLM'siz karşılığı) -----------------
def drive(trace, ledger, episodes, reasoning, calls):
    r = trace.add_reasoning(reasoning)
    for name, args, kind in calls:
        if name == "__ep_start__":
            episodes.start(args["name"], args["type"], trace._seq, args.get("deps"))
            continue
        if name == "__ep_end__":
            episodes.end(trace._seq, args.get("desc", "")); continue
        out = str(DISPATCH[name](**args))
        ev = trace.add_tool(name, args, out, intent_ref=r.seq,
                            verbatim=TOOL_META[name].get("verbatim", False))
        ledger.record(name, args, out, ev.seq)
        episodes.attach(ev.seq)


def main():
    trace, ledger, episodes = Trace(), ExecutionLedger(tool_meta=TOOL_META), EpisodeGraph()

    # Tur 1 — keşif episode: ticket'ları tara
    drive(trace, ledger, episodes, "Ticket'ları inceleyeceğim",
          [("__ep_start__", {"name": "veri", "type": "expl"}, None),
           ("search_tickets", {"q": "open bugs"}, None),
           ("get_ticket", {"id": "T-11"}, None),
           ("get_ticket", {"id": "T-12"}, None),
           ("get_metrics", {"project": "ATLAS"}, None),
           ("__ep_end__", {"desc": "ATLAS: 2 ticket + velocity 32"}, None)])

    # Tur 2 — TEKRAR (T-11 yeniden) → dedup
    drive(trace, ledger, episodes, "T-11'e bir daha bakayım",
          [("get_ticket", {"id": "T-11"}, None)])

    # Tur 3 — rapor episode + outline read → write → read (staleness)
    drive(trace, ledger, episodes, "Rapor hazırlayacağım",
          [("__ep_start__", {"name": "rapor", "type": "act", "deps": ["veri"]}, None),
           ("create_report", {"title": "Durum"}, None),
           ("get_outline", {"doc": "doc-42"}, None),      # v1 okundu
           ("add_block", {"doc": "doc-42", "title": "Özet"}, None),  # write → v2
           ("get_outline", {"doc": "doc-42"}, None),      # v2 → eski outline BAYAT
           ("__ep_end__", {"desc": "Durum raporu üretildi"}, None)])

    # Tur 4 — kapanış okumaları (eski outline'ı koruma penceresinden çıkarır)
    drive(trace, ledger, episodes, "Son durumu teyit edeyim",
          [("get_metrics", {"project": "ATLAS"}, None),
           ("get_ticket", {"id": "T-12"}, None)])

    # --- sıkıştır (LLM katmanı KAPALI; sadece deterministik) ---
    comp = HybridCompactor(budget=500, protect_window=3, filter_safe=True)
    res = comp.compact(trace, ledger, episode_graph=episodes, force=True)

    BAR = "─" * 74
    print(BAR)
    print(f"HYBRID COMPACTION · {res['before']} → {res['after']} tok · %{res['saved_pct']} kazanç")
    print(BAR)
    print("KADER ŞERİDİ:")
    for e in trace.tool_events():
        fate = "SİL " if e.cleared else "ÖZET" if e.evicted else "TAM "
        ep = episodes.episode_of(e.seq)
        olay = f"[{ep.type}] {ep.name}" if ep else ledger.category_of(e.seq)
        print(f"  #{e.seq:<2} {e.payload['name']:<15} {fate}  {olay}")
    print("\nLOG:")
    for l in res["log"]:
        print("  " + l.strip())

    print(f"\n{BAR}\nMODELE GİDEN İÇERİK (render_event · filter-safe prefix'li):")
    for e in trace.tool_events():
        if e.evicted or e.cleared:
            print(f"  #{e.seq}: {render_event(e, filter_safe=True)[:110]}")

    # --- Hermes LLM katmanı AÇIK varyant (sahte özetleyiciyle) ---
    print(f"\n{BAR}\nOPSIYONEL LLM KATMANI (Hermes) — summarize_fn verilirse:")
    t2, l2, e2 = Trace(), ExecutionLedger(tool_meta=TOOL_META), EpisodeGraph()
    drive(t2, l2, e2, "veri", [("get_metrics", {"project": "A"}, None),
                                ("search_tickets", {"q": "x"}, None),
                                ("get_ticket", {"id": "T-99"}, None)])
    fake_llm = lambda payloads: f"[LLM] {len(payloads)} olay özeti: ATLAS sağlıklı, 3 kayıt incelendi."
    c2 = HybridCompactor(budget=200, protect_window=1, summarize_fn=fake_llm)
    r2 = c2.compact(t2, l2, force=True)
    print(f"  {r2['before']} → {r2['after']} tok · log: {r2['log'][-1] if r2['log'] else '-'}")
    print(BAR)


if __name__ == "__main__":
    main()
