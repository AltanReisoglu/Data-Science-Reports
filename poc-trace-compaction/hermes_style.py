"""
hermes_style.py — Hermes'in `_summarize_tool_result()` mantığını BİZİM ürün
tool'larına uygular. Deterministik (sıfır LLM), tool-tipine özel tek-satır özet.

Hermes ilkesi: büyük çıktıyı, tool tipine göre "özün özü" bir satıra indir —
hangi tool, ne sorguldu/hangi kayıt, kaç sonuç/hangi durum, ne kadar büyüktü.
Tanınmayan/parse patlarsa: "[tool] (N chars result)" (asla çökmez).

  python hermes_style.py    # 9 toolkit'ten temsilci tool'lar için HAM → HERMES
"""
from __future__ import annotations
import json
import re

import product_tools as P


def _lines(s: str) -> int:
    return len(s.splitlines())


def summarize_tool_result(tool_name: str, tool_args: str, tool_content: str) -> str:
    """Hermes tarzı guarded tek-satır özet."""
    try:
        return _unguarded(tool_name, tool_args, tool_content)
    except Exception:
        n = len(tool_content) if isinstance(tool_content, str) else 0
        return f"[{tool_name}] ({n:,} chars result)"


def _arg(args: dict, *keys):
    for k in keys:
        if k in args and args[k] not in (None, ""):
            return args[k]
    return None


def _unguarded(name: str, tool_args: str, content: str) -> str:
    args = json.loads(tool_args) if isinstance(tool_args, str) else (tool_args or {})
    n = len(content)
    low = name.lower()

    # --- arama / resolve / listeleme: sorgu + sonuç sayısı ---
    if any(k in low for k in ("_search", "_resolve_", "_list_", "field_values")):
        q = _arg(args, "query", "ref", "name", "cql", "filter", "title") or "*"
        m = re.search(r"(\d+)\s+sonuç", content)
        cnt = m.group(1) if m else "?"
        return f"[{name}] '{q}' → {cnt} sonuç ({n} chars)"

    # --- agregasyon / sayım: sayıyı BİREBİR taşı ---
    if any(k in low for k in ("_aggregate", "_count", "_group_by", "_run_stats",
                              "_sprint_report", "_worklog", "_run_sql")):
        first = content.splitlines()[0] if content else ""
        num = re.search(r"(→|=)\s*([\d.,]+)", first)
        head = first.split(":")[0] if ":" in first else name
        if num:
            return f"[{name}] {head} → {num.group(2)} ({n} chars)"
        return f"[{name}] {first[:60]} ({n} chars)"

    # --- issue okuma: key + status ---
    if name == "jira_get_issue":
        key = _arg(args, "key") or "?"
        st = re.search(r"status:\s*(.+)", content)
        tp = re.search(r"type:\s*(.+)", content)
        return (f"[jira_get_issue] {key} → status {st.group(1).strip() if st else '?'} · "
                f"{tp.group(1).strip() if tp else ''} ({_lines(content)} lines)")

    # --- proje okuma (jira/neta): key + bir salient alan ---
    if name in ("jira_get_project", "neta_get_project"):
        key = _arg(args, "project_key", "ref") or "?"
        cost = re.search(r"planned_cost:\s*(.+)", content)
        openi = re.search(r"open_issues:\s*(\d+)", content)
        detail = (f"planned_cost {cost.group(1).strip()}" if cost
                  else f"open {openi.group(1)}" if openi else "")
        return f"[{name}] {key} → {detail} ({_lines(content)} lines)"

    # --- confluence sayfa ---
    if name == "confluence_get_page":
        pid = _arg(args, "page_id") or "?"
        title = re.search(r"Sayfa\s+\S+:\s*'([^']+)'", content)
        return f"[confluence_get_page] {pid} → '{title.group(1) if title else '?'}' ({_lines(content)} lines)"

    # --- outline: blok sayısı + sürüm ---
    if low.endswith("_get_outline"):
        did = _arg(args, "document_id") or "?"
        vb = re.search(r"\(v(\d+),\s*(\d+)\s+blok\)", content)
        if vb:
            return f"[{name}] {did} → {vb.group(2)} blok (v{vb.group(1)})"
        return f"[{name}] {did} → boş/v0 ({n} chars)"

    # --- hücre okuma ---
    if low.endswith("_read_cells"):
        did = _arg(args, "document_id") or "?"
        return f"[{name}] {did} → hücre bloğu ({_lines(content)} lines)"

    # --- yazma / doküman inşası: OK satırı + id/sürüm ---
    if content.startswith("OK:"):
        first = content.splitlines()[0]
        idm = re.search(r"(document_id|df_id|artifact_id)=(\S+)", first)
        vm = re.search(r"v(\d+)", first)
        verb = first.replace("OK:", "").strip().split("·")[0].strip()
        tail = (f"{idm.group(1)}={idm.group(2)}" if idm else "") + (f" v{vm.group(1)}" if vm else "")
        return f"[{name}] {verb} {tail}".strip()

    # --- get_schema: statik ---
    if low.endswith("_get_schema"):
        return f"[{name}] şema sözleşmesi ({n} chars)"

    # --- fallback ---
    first = content.splitlines()[0] if content else ""
    return f"[{name}] {first[:50]} ({n} chars)"


# ---------------------------------------------------------------------------
def _demo():
    BAR = "─" * 78
    cases = [
        ("jira_resolve_project", {"ref": "Atlas"}),
        ("jira_get_issue", {"key": "ATLAS-101"}),
        ("jira_get_project", {"project_key": "ATLAS"}),
        ("jira_search_issues", {"project_key": "ATLAS", "status": "Open"}),
        ("jira_aggregate", {"project_key": "ATLAS", "metric": "count"}),
        ("jira_group_by", {"project_key": "ATLAS", "group_field": "status"}),
        ("jira_sprint_report", {"project_key": "ATLAS"}),
        ("neta_get_project", {"ref": "MPP-409"}),
        ("neta_count", {"field": "status", "value": "aktif"}),
        ("ldap_org_count", {"unit": "BT"}),
        ("ldap_org_members", {"unit": "Yazılım Geliştirme"}),
        ("confluence_search", {"query": "mimari kararlar"}),
        ("confluence_get_page", {"page_id": "12345"}),
        ("analysis_run_sql", {"sql": "SELECT ay, gelir FROM t"}),
        ("docx_create", {"spec": {"title": "Rapor"}}),
        ("docx_add_chart", {"document_id": "doc_ab12", "title": "İş Dağılımı"}),
        ("docx_get_outline", {"document_id": "doc_ab12"}),
        ("xlsx_read_cells", {"document_id": "doc_ab12"}),
    ]
    # docx_get_outline gerçek versiyon göstersin diye önce bir belge kurup düzenleyelim
    c = P.DISPATCH["docx_create"](spec={}); did = c.split("document_id=")[1].split(" ")[0]
    P.DISPATCH["docx_add_chart"](document_id=did, title="x")
    print(BAR)
    print("HERMES _summarize_tool_result() BİZİM 119 TOOL'A UYGULANIRSA")
    print("(deterministik · sıfır LLM · tool-tipine özel tek satır)")
    print(BAR)
    for name, args in cases:
        a = dict(args)
        if name in ("docx_add_chart", "docx_get_outline", "xlsx_read_cells"):
            a["document_id"] = did
        out = str(P.DISPATCH[name](**a))
        one = summarize_tool_result(name, json.dumps(a), out)
        raw_lines = _lines(out)
        print(f"\n{name}")
        print(f"  HAM   : {len(out):>4} char · {raw_lines:>2} satır  |  {out.splitlines()[0][:52]}")
        print(f"  HERMES: {one}")
    print(BAR)


if __name__ == "__main__":
    _demo()
