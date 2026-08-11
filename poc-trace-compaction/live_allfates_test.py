"""Canlı — her soru tipi tool trace'i doğru tetikliyor mu (dedup/staleness/graceful)."""
import chat as chatmod, product_tools as pt
from agent import TracingAgent

ag = TracingAgent(compaction=True, schemas=pt.SCHEMAS, dispatch=pt.DISPATCH,
                  tool_meta=pt.TOOL_META, system=chatmod.PRODUCT_SYSTEM, max_turns=18)
ag.compactor.budget = 800; ag.compactor.target = 400

QUESTIONS = [
    ("veri toplama", "ATLAS-101, ATLAS-102 ve ATLAS-103 issue'larının detaylarını getir."),
    ("bilinçli TEKRAR", "ATLAS-101'in detayına bir kez daha bak, değişmiş mi?"),  # → dedup/SİL
    ("tool'suz soru", "Teşekkürler, özetler misin ne bulduğunu?"),                 # → graceful, tool yok
]
BAR = "═" * 70
for i,(tip,q) in enumerate(QUESTIONS,1):
    snap=dict(ag.metrics)
    print(f"\n{BAR}\n[{i}·{tip}] {q}\n{BAR}")
    try: ans=ag.send(q)
    except Exception as e: print(f"  [HATA] {type(e).__name__}: {e}"); break
    print(f"[Asistan] {ans[:400]}")
    tools=ag.trace.tool_events()
    ev=sum(1 for e in tools if e.evicted); cl=sum(1 for e in tools if e.cleared)
    print(f"  ┄ +{ag.metrics['tool_calls']-snap['tool_calls']} tool · toplam {len(tools)} birim "
          f"({ev} özet, {cl} SİL) · render {ag.rendered_token_cost()} tok")

print(f"\n{BAR}\nTRACE SON HALİ:")
for e in ag.trace.tool_events():
    fate="SİLİNDİ" if e.cleared else "ÖZET" if e.evicted else "TAM"
    note = e.clear_note if e.cleared else (e.summary.etki if e.evicted and e.summary else "")
    print(f"  #{e.seq:<3} {e.payload['name']:<22} {fate:<8} {note[:40]}")
print(f"{BAR}\nSİL {sum(1 for e in ag.trace.tool_events() if e.cleared)} · "
      f"ÖZET {sum(1 for e in ag.trace.tool_events() if e.evicted)} · "
      f"compaction ×{ag.metrics['compaction_passes']}")
