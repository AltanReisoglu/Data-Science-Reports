"""
chat.py — ÇOK-TURLU chat; her mesajda trace compaction ARKADA birebir çalışır.

  python chat.py            # dosya tool'ları (sample_repo)
  python chat.py --equity   # finansal tool'lar (XOM vb.)
  python chat.py --no-compaction   # taban çizgisi (sıkıştırma kapalı)

Sen konuşurken: her mesaj Gemma'ya gider, tool'lar çalışır, ürettiği trace
ledger'a işlenir, bütçe aşılınca compaction faz faz devreye girer. Trace
turlar boyunca BİRİKİR ama sıkışık kalır. Her yanıttan sonra arkadaki durumu
tek satırda görürsün. Komutlar: /trace  /playbook  /ledger  /reset  /quit
"""
from __future__ import annotations
import sys

from agent import TracingAgent

BAR = "─" * 68
EQUITY_SYSTEM = (
    "Sen bir hisse-araştırma asistanısın. Kullanıcının sorusunu finansal "
    "tool'ları çağırarak yanıtla.\n"
    "- Çok veri toplayacaksan ÖNCE delimiter(action='start', type='expl', "
    "name='veri-toplama') çağır; toplama bitince delimiter(action='end', "
    "description='ne öğrendin') çağır.\n"
    "- Grafik/rapor üretmeden ÖNCE delimiter(action='start', type='act', "
    "name='rapor', dependencies=['veri-toplama']) çağır, sonra visualize_data, "
    "sonra delimiter(action='end').\n"
    "- Bir tool hata dönerse (ör. ticker eksik) düzeltip AYNI tool'u doğru "
    "argümanla tekrar çağır.\n"
    "- Aynı veriyi gereksiz tekrar çekme; sonunda kısa net bir sentez yaz.")


PRODUCT_SYSTEM = (
    "Sen bir kurumsal asistansın. Jira (yazılım iş takibi), NETA (proje "
    "portföyü/bütçe), LDAP (kurumsal dizin), Confluence (wiki), ve doküman "
    "üretimi (docx/pdf/pptx/xlsx) + veri analizi tool'larıyla soruları yanıtla.\n"
    "- Serbest ad geçen projeyi/kişiyi ÖNCE resolve et (jira_resolve_project, "
    "neta_resolve_project, ldap_resolve_person), sonra key/id isteyen tool'u çağır.\n"
    "- Çok veri toplayacaksan ÖNCE delimiter(action='start', type='expl', "
    "name='veri-toplama'); toplama bitince delimiter(action='end', "
    "description='ne öğrendin').\n"
    "- Doküman/grafik üretmeden ÖNCE delimiter(action='start', type='act', "
    "name='rapor', dependencies=['veri-toplama']); üretim bitince delimiter(action='end').\n"
    "- Bir tool hata dönerse (ör. eksik key) düzeltip AYNI tool'u doğru argümanla "
    "tekrar çağır. Aynı veriyi gereksiz tekrar çekme; sonunda kısa net sentez yaz.")


def make_agent(argv):
    compaction = "--no-compaction" not in argv
    if "--equity" in argv:
        import equity_tools as eq
        return TracingAgent(compaction=compaction, schemas=eq.SCHEMAS,
                            dispatch=eq.DISPATCH, tool_meta=eq.TOOL_META,
                            system=EQUITY_SYSTEM), "equity"
    if "--product" in argv:
        import product_tools as pt
        return TracingAgent(compaction=compaction, schemas=pt.SCHEMAS,
                            dispatch=pt.DISPATCH, tool_meta=pt.TOOL_META,
                            system=PRODUCT_SYSTEM, max_turns=20), "product"
    return TracingAgent(compaction=compaction), "file"


def status(ag, snap):
    """Bir turun ARKADA yaptığı trace-compaction işini tek satırda göster."""
    m = ag.metrics
    cur = ag.trace.total_tokens()
    peak = max(m["peak_trace_tokens"], cur)   # peak her zaman >= cur
    sıkışık = f"{cur}/{peak} tok" + (f" (%{round(100*(peak-cur)/peak)} sıkışık)" if peak > cur else "")
    return (f"  ┄ trace: {len(ag.trace.tool_events())} birim · {sıkışık} · "
            f"bu tur: +{m['tool_calls']-snap['tool_calls']} tool, "
            f"compaction ×{m['compaction_passes']-snap['compaction_passes']}, "
            f"evict {m['evicted']-snap['evicted']} · playbook {m['playbook_bullets']} ders")


def dump_trace(ag):
    print(BAR)
    print("TRACE (arkadaki gerçek durum) — kader:")
    for e in ag.trace.tool_events():
        fate = ("SİLİNDİ" if e.cleared else "ÖZET" if e.evicted else "TAM")
        args = ", ".join(f"{k}={v}" for k, v in e.payload.get("args", {}).items())[:40]
        print(f"  #{e.seq:<3} {e.payload['name']:<26} {fate:<8} {args}")
    print(BAR)


def dump_playbook(ag):
    print(BAR)
    print("PLAYBOOK (öğrenilen — evict'ten korunur):")
    for b in ag.playbook.active_bullets():
        print(f"  • {b.render()}")
    if not ag.playbook.active_bullets():
        print("  (henüz ders yok)")
    print(BAR)


def dump_ledger(ag):
    L = ag.ledger
    print(BAR)
    print(f"LEDGER: {len(L.observations)} gözlem · {L.global_counter} yazma · adım {L.step}")
    seen = {}
    for o in L.observations:
        cur = L.local_counters.get(o.path, 0)
        st = "BAYAT" if L.is_stale(o.event_seq) else "taze"
        seen[o.path] = f"v{cur} · {st}"
    for p, v in seen.items():
        print(f"  {p:<24} {v}")
    print(BAR)


def main():
    argv = sys.argv[1:]
    ag, domain = make_agent(argv)
    print(BAR)
    print(f"TRACE-COMPACTION CHAT · domain={domain} · "
          f"compaction={'AÇIK' if ag.compaction else 'KAPALI'}")
    print("Her mesajda tool tracing + compaction ARKADA çalışır.")
    print("Komutlar: /trace  /playbook  /ledger  /reset  /quit")
    if domain == "equity":
        print("Örnek: 'XOM için P/E ve analist hedefi ne?'  ·  'gelir trendini özetle'")
    else:
        print("Örnek: 'config.py'deki PORT'u bul'  ·  'server.py ne yapıyor?'")
    print(BAR)

    while True:
        try:
            user = input("\nsen › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\ngörüşürüz."); break
        if not user:
            continue
        low = user.lower()
        if low in ("/quit", "/exit", "/çık", "quit", "exit"):
            print("görüşürüz."); break
        if low == "/trace": dump_trace(ag); continue
        if low == "/playbook": dump_playbook(ag); continue
        if low == "/ledger": dump_ledger(ag); continue
        if low == "/reset":
            ag, domain = make_agent(argv); print("  (sohbet sıfırlandı)"); continue

        snap = dict(ag.metrics)   # turdan önceki durum (delta için)
        try:
            answer = ag.send(user)
        except RuntimeError as e:
            print(f"\n[LLM yok] {e}")
            print("İpucu: pip install -r requirements.txt · .env'de LLM_API_KEY dolu mu?")
            continue
        except Exception as e:
            print(f"\n[hata] {type(e).__name__}: {e}")
            continue

        print(f"\nasistan › {answer}")
        if ag.compaction:
            print(status(ag, snap))


if __name__ == "__main__":
    main()
