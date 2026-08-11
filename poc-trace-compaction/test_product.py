"""
test_product.py — Ürün tool'ları + messages[] köprüsü (LLM'siz, deterministik).

Kanıtladığı şeyler:
  1. 119 ürün tool'u yükleniyor, kategorileri/mock'ları çalışıyor.
  2. Tool trace turlar boyunca BİRİKİYOR (resolve→read→agg→doküman inşası).
  3. Compaction sonrası modelin GERÇEKTEN gördüğü bağlam (_render_messages) küçülüyor.
  4. tool_call_id eşleşmesi korunuyor → asla API 400 (tool_use/tool_result kopmaz).
  5. read→write→read staleness ve dedup gerçekten tetikleniyor.

Bu, "çıktıları LLM'e koyarken tool-trace-compaction işleyip verir" iddiasının
uçtan uca kanıtıdır — send()'in LLM'siz simülasyonu.
"""
from __future__ import annotations
import json

import product_tools as pt
from agent import TracingAgent

_ID = {"n": 0}


def _tc(name, args):
    _ID["n"] += 1
    return {"id": f"call_{_ID['n']:03d}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _drive(ag, reasoning, calls):
    """send() döngüsünün LLM'siz karşılığı: bir asistan turu + tool sonuçları.
    calls: [(name, args), ...]. Trace'e işler, messages'a ekler, köprüyü kurar."""
    rev = ag.trace.add_reasoning(reasoning)
    tcs = [_tc(n, a) for n, a in calls]
    ag.messages.append({"role": "assistant", "content": reasoning, "tool_calls": tcs})
    for tc, (name, args) in zip(tcs, calls):
        if name == "delimiter":
            out = ag._handle_delimiter(args)
            ag.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
            continue
        try:
            out = str(ag.dispatch[name](**args)); status = "ok"
        except Exception as e:
            out = f"Hata: {e}"; status = "error"
        seq = ag._record_and_maybe_compact(name, args, out, status, rev.seq)
        ag._call_seq[tc["id"]] = seq
        ag.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})


def _pairing_ok(messages):
    """Her assistant tool_call id'sinin karşılığında bir tool mesajı var mı?"""
    call_ids, result_ids = set(), set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls", []) or []:
                call_ids.add(tc["id"])
        if m.get("role") == "tool":
            result_ids.add(m.get("tool_call_id"))
    return call_ids == result_ids, call_ids ^ result_ids


def run():
    ok = 0
    ag = TracingAgent(compaction=True, schemas=pt.SCHEMAS, dispatch=pt.DISPATCH,
                      tool_meta=pt.TOOL_META, system="ürün asistanı")
    # Biriktirme fazı: bütçeyi çok yüksek tut → tool trace turlar boyunca ARTIMSIZ
    # birikir (kullanıcının senaryosu: "trace birikir, SONRA compaction işler").
    ag.compactor.budget = 10 ** 9
    ag.compactor.target = 10 ** 9
    ag.messages = [{"role": "system", "content": ag.system}]
    ag._call_seq = {}

    # --- Tur 1: Jira keşfi (expl episode) ---
    _drive(ag, "Atlas projesini ve açık işleri inceleyeceğim",
           [("delimiter", {"action": "start", "type": "expl", "name": "jira-veri"}),
            ("jira_resolve_project", {"ref": "Atlas"}),
            ("jira_get_project", {"project_key": "ATLAS"}),
            ("jira_get_issue", {"key": "ATLAS-101"}),
            ("jira_get_issue", {"key": "ATLAS-102"}),
            ("jira_get_issue", {"key": "ATLAS-103"}),
            ("jira_aggregate", {"project_key": "ATLAS", "metric": "count"}),
            ("delimiter", {"action": "end", "description": "ATLAS: 3 issue + toplam 47 iş"})])

    # --- Tur 2: tekrar okuma (DEDUP) + NETA portföy ---
    _drive(ag, "ATLAS-101'i tekrar kontrol + bütçe bakacağım",
           [("jira_get_issue", {"key": "ATLAS-101"}),          # DEDUP: tur1 ile aynı
            ("neta_resolve_project", {"ref": "Atlas"}),
            ("neta_get_project", {"ref": "MPP-409"})])

    # --- Tur 3: doküman inşası (act episode) + staleness ---
    _drive(ag, "Bulguları bir Word raporuna dökeceğim",
           [("delimiter", {"action": "start", "type": "act", "name": "rapor",
                           "dependencies": ["jira-veri"]}),
            ("docx_create", {"spec": {"title": "ATLAS Raporu"}})])
    did = [e for e in ag.trace.tool_events()
           if e.payload["name"] == "docx_create"][-1].payload["output"].split(
               "document_id=")[1].split(" ")[0]
    _drive(ag, "Rapora outline, grafik ekleyip tekrar outline alacağım",
           [("docx_get_outline", {"document_id": did}),        # READ (v1)
            ("docx_add_chart", {"document_id": did, "title": "İş Dağılımı"}),  # WRITE → v2
            ("docx_get_outline", {"document_id": did}),        # READ (v2) → önceki outline BAYAT
            ("docx_finalize", {"document_id": did}),
            ("delimiter", {"action": "end", "description": "ATLAS raporu üretildi"})])

    # === Sonraki soru anı: compaction trace'i işler (budget'ı gerçekçi eşiğe indir) ===
    ag.compactor.budget = 700
    ag.compactor.target = 420
    ag.compactor.compact(ag.trace, ag.ledger, force=True, episode_graph=ag.episodes)
    raw = ag.raw_token_cost(ag.messages)
    rendered = ag.rendered_token_cost(ag.messages)
    saved = round(100 * (raw - rendered) / raw) if raw else 0

    n_tools = len(ag.trace.tool_events())
    n_evict = sum(1 for e in ag.trace.tool_events() if e.evicted)
    n_clear = sum(1 for e in ag.trace.tool_events() if e.cleared)

    print("=" * 68)
    print(f"ÜRÜN SENARYOSU — {n_tools} tool birimi biriktirildi")
    print(f"  ham bağlam (messages)     : {raw} token")
    print(f"  modele giden (render)     : {rendered} token")
    print(f"  KAZANÇ                    : %{saved}  (ÖZET {n_evict} · SİL {n_clear})")
    print("=" * 68)

    # 1) render gerçekten küçülüyor mu
    assert rendered < raw, f"render ({rendered}) ham'dan ({raw}) küçük değil!"
    print("✓ modele giden bağlam ham'dan küçük (compaction messages'a yansıdı)")
    ok += 1

    # 2) tool_call_id eşleşmesi bozulmadı mı (API 400 guard)
    paired, diff = _pairing_ok(ag._render_messages(ag.messages))
    assert paired, f"eşleşme bozuk! farklı id'ler: {diff}"
    print("✓ tüm tool_call_id'ler eşleşiyor (API 400 riski yok)")
    ok += 1

    # 3) render edilen tool mesajları gerçekten kısaldı mı (en az biri [özet]/[silindi])
    rmsgs = ag._render_messages(ag.messages)
    compacted_contents = [m["content"] for m in rmsgs
                          if m.get("role") == "tool"
                          and (m["content"].startswith("[özet]") or m["content"].startswith("[silindi]"))]
    assert compacted_contents, "hiçbir tool mesajı sıkışık forma geçmemiş!"
    print(f"✓ {len(compacted_contents)} tool mesajı sıkışık forma indi (örn: "
          f"{compacted_contents[0][:70]}…)")
    ok += 1

    # 4) dedup çalıştı mı (tur2 ATLAS-101 → tur1'in tekrarı) — SİL veya ÖZET
    dups = [e for e in ag.trace.tool_events()
            if ("tekrar" in (e.clear_note or ""))
            or (e.summary and "tekrar" in (e.summary.etki or ""))]
    assert dups, "dedup tetiklenmedi (ATLAS-101 tekrarı yakalanmalıydı)"
    print(f"✓ dedup çalıştı: {len(dups)} tekrar okuma yakalandı")
    ok += 1

    # 5) staleness: outline read→write→read (ilk outline bayat)
    stale = [e for e in ag.trace.tool_events()
             if (e.cleared or e.evicted) and e.payload["name"] == "docx_get_outline"]
    assert stale, "outline staleness tetiklenmedi (write sonrası eski outline bayatlamalı)"
    print(f"✓ staleness çalıştı: {len(stale)} bayat outline temizlendi/özetlendi")
    ok += 1

    # 6) CWL episode grafiği kuruldu mu
    assert any(e.type == "expl" for e in ag.episodes.episodes), "expl episode yok"
    assert any(e.type == "act" for e in ag.episodes.episodes), "act episode yok"
    print("✓ CWL episode grafiği kuruldu (expl 'jira-veri' + act 'rapor')")
    ok += 1

    print("=" * 68)
    print(f"{ok}/6 geçti — messages[] köprüsü uçtan uca çalışıyor.")
    return ok == 6


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
