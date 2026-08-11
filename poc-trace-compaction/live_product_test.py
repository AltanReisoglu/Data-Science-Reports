"""Canlı test — gerçek Gemma + ürün tool'ları, kullanıcı gibi çok-turlu sorular."""
import chat as chatmod
import product_tools as pt
from agent import TracingAgent

ag = TracingAgent(compaction=True, schemas=pt.SCHEMAS, dispatch=pt.DISPATCH,
                  tool_meta=pt.TOOL_META, system=chatmod.PRODUCT_SYSTEM, max_turns=18)
ag.compactor.budget = 900       # canlıda compaction görünür olsun
ag.compactor.target = 520

QUESTIONS = [
    "Atlas projesindeki açık işleri ve toplam iş sayısını çıkar.",
    "ATLAS-101'in detayına tekrar bak, bir de MPP-409'un bütçe durumu ne?",
    "Bu bulguları bir Word raporuna dök, iş dağılımı için bir grafik de ekle.",
    "Raporun outline'ını göster, sonra kısa bir özet yaz.",
]

BAR = "═" * 72
for i, q in enumerate(QUESTIONS, 1):
    snap = dict(ag.metrics)
    print(f"\n{BAR}\n[Kullanıcı {i}] {q}\n{BAR}")
    try:
        ans = ag.send(q)
    except Exception as e:
        print(f"  [HATA] {type(e).__name__}: {e}")
        break
    print(f"[Asistan] {ans[:600]}")
    m = ag.metrics
    tools = ag.trace.tool_events()
    ev = sum(1 for e in tools if e.evicted); cl = sum(1 for e in tools if e.cleared)
    raw = ag.raw_token_cost(); rend = ag.rendered_token_cost()
    saved = round(100*(raw-rend)/raw) if raw else 0
    print(f"\n  ┄ ARKADA: +{m['tool_calls']-snap['tool_calls']} tool bu tur · "
          f"toplam {len(tools)} birim ({ev} özet, {cl} sil) · "
          f"compaction ×{m['compaction_passes']-snap['compaction_passes']}")
    print(f"  ┄ MODELE GİDEN BAĞLAM: ham {raw} → sıkışık {rend} tok (%{saved}) · "
          f"episode: {ag.episodes.summary() or '-'}")

print(f"\n{BAR}\nSONUÇ: {len(ag.trace.tool_events())} tool birikti, "
      f"messages ham {ag.raw_token_cost()} → render {ag.rendered_token_cost()} tok")
print(BAR)
print("TRACE'TEKİ TOOL'LARIN EN SON HALİ (kader şeridi):")
for e in ag.trace.tool_events():
    fate = "SİLİNDİ" if e.cleared else "ÖZET" if e.evicted else "TAM  "
    ep = ag.episodes.episode_of(e.seq)
    olay = f"[{ep.type}] {ep.name}" if ep else ag.ledger.category_of(e.seq)
    detay = ""
    if e.cleared:
        detay = "→ " + e.clear_note
    elif e.evicted and e.summary:
        detay = "→ " + (e.summary.etki or e.summary.sonuc)[:46]
    print(f"  #{e.seq:<3} {e.payload['name']:<24} {fate}  {olay:<22} {detay}")
tot = len(ag.trace.tool_events())
tam = sum(1 for e in ag.trace.tool_events() if not e.evicted and not e.cleared)
ozet = sum(1 for e in ag.trace.tool_events() if e.evicted)
sil = sum(1 for e in ag.trace.tool_events() if e.cleared)
print(BAR)
print(f"KADER DAĞILIMI: {tam} TAM · {ozet} ÖZET · {sil} SİL  (toplam {tot})")
print(f"COMPACTION: {ag.metrics['compaction_passes']} pass · {ag.metrics['evicted']} birim işlendi")
print(f"MODELE GİDEN: ham {ag.raw_token_cost()} → render {ag.rendered_token_cost()} tok "
      f"(%{round(100*(ag.raw_token_cost()-ag.rendered_token_cost())/max(1,ag.raw_token_cost()))})")
print(BAR)
