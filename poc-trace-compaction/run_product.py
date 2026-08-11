"""
run_product.py — Ürünün 119 tool'uyla uçtan uca trace-compaction demosu (LLM'siz).

Senaryo: Jira'dan veri topla → NETA bütçe → Word raporu üret. Tool trace turlar
boyunca BİRİKİR; sonraki soru anında compaction işler ve modele giden bağlamı
(messages[]) sıkışık haliyle verir. tool_call_id eşleşmesi korunur (API 400 yok).

  python run_product.py
"""
from __future__ import annotations
import json

import product_tools as pt
from agent import TracingAgent

BAR = "─" * 70
_ID = {"n": 0}


def _tc(name, args):
    _ID["n"] += 1
    return {"id": f"call_{_ID['n']:03d}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _drive(ag, reasoning, calls):
    rev = ag.trace.add_reasoning(reasoning)
    tcs = [_tc(n, a) for n, a in calls]
    ag.messages.append({"role": "assistant", "content": reasoning, "tool_calls": tcs})
    for tc, (name, args) in zip(tcs, calls):
        if name == "delimiter":
            out = ag._handle_delimiter(args)
        else:
            try:
                out = str(ag.dispatch[name](**args)); status = "ok"
            except Exception as e:
                out = f"Hata: {e}"; status = "error"
            seq = ag._record_and_maybe_compact(name, args, out, status, rev.seq)
            ag._call_seq[tc["id"]] = seq
        ag.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})


def main():
    ag = TracingAgent(compaction=True, schemas=pt.SCHEMAS, dispatch=pt.DISPATCH,
                      tool_meta=pt.TOOL_META, system="kurumsal asistan")
    ag.compactor.budget = 10 ** 9          # biriktirme fazı: artımsız birik
    ag.compactor.target = 10 ** 9
    ag.messages = [{"role": "system", "content": ag.system}]
    ag._call_seq = {}

    print(BAR)
    print(f"ÜRÜN DEMO · {len(pt.SCHEMAS)-1} tool ({len(set(m['cat'] for m in pt.TOOL_META.values()))} "
          f"kategori) · 9 toolkit · compaction AÇIK")
    print(BAR)

    _drive(ag, "Atlas projesini ve açık işleri inceleyeceğim (keşif)",
           [("delimiter", {"action": "start", "type": "expl", "name": "jira-veri"}),
            ("jira_resolve_project", {"ref": "Atlas"}),
            ("jira_get_project", {"project_key": "ATLAS"}),
            ("jira_get_issue", {"key": "ATLAS-101"}),
            ("jira_get_issue", {"key": "ATLAS-102"}),
            ("jira_aggregate", {"project_key": "ATLAS", "metric": "count"}),
            ("delimiter", {"action": "end", "description": "ATLAS: 2 issue + toplam 47 iş"})])
    _drive(ag, "ATLAS-101'i tekrar bakıp NETA bütçesine geçeceğim",
           [("jira_get_issue", {"key": "ATLAS-101"}),          # dedup
            ("neta_get_project", {"ref": "MPP-409"})])
    _drive(ag, "Bulguları Word raporuna dökeceğim (eylem)",
           [("delimiter", {"action": "start", "type": "act", "name": "rapor",
                           "dependencies": ["jira-veri"]}),
            ("docx_create", {"spec": {"title": "ATLAS Raporu"}})])
    did = [e for e in ag.trace.tool_events()
           if e.payload["name"] == "docx_create"][-1].payload["output"].split(
               "document_id=")[1].split(" ")[0]
    _drive(ag, "Rapora outline+grafik ekleyip tekrar outline alacağım",
           [("docx_get_outline", {"document_id": did}),        # v1
            ("docx_add_chart", {"document_id": did, "title": "İş Dağılımı"}),  # write → v2
            ("docx_get_outline", {"document_id": did}),        # v2 → eski outline BAYAT
            ("docx_finalize", {"document_id": did}),
            ("delimiter", {"action": "end", "description": "ATLAS raporu üretildi"})])

    # Sonraki soru anı: compaction trace'i işler
    ag.compactor.budget, ag.compactor.target = 700, 420
    res = ag.compactor.compact(ag.trace, ag.ledger, force=True, episode_graph=ag.episodes)

    raw = ag.raw_token_cost(ag.messages)
    rendered = ag.rendered_token_cost(ag.messages)
    saved = round(100 * (raw - rendered) / raw) if raw else 0

    print("\nKADER ŞERİDİ (biriken tool trace):")
    for e in ag.trace.tool_events():
        fate = "SİLİNDİ" if e.cleared else "ÖZET" if e.evicted else "TAM"
        ep = ag.episodes.episode_of(e.seq)
        olay = f"[{ep.type}] {ep.name}" if ep else ag.ledger.category_of(e.seq)
        print(f"  #{e.seq:<3} {e.payload['name']:<24} {fate:<8} {olay}")

    print("\nCOMPACTION LOG:")
    for l in res["log"]:
        print("  " + l.strip())

    print(f"\n{BAR}")
    print(f"MODELE GİDEN BAĞLAM:  ham {raw} token  →  sıkışık {rendered} token  →  KAZANÇ %{saved}")
    print(f"tool_call eşleşmesi korunuyor (API 400 yok) · episode: {ag.episodes.summary()}")
    print(BAR)
    print("Not: özetleme deterministik (LLM'siz); sıkışık çıktı messages[]'e YAZILDI,")
    print("     yani modelin bir sonraki turda gördüğü bağlam gerçekten küçüldü.")


if __name__ == "__main__":
    main()
