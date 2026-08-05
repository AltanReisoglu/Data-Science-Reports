"""
tool_summary.py — Hermes'ten alınan parça: tip-farkında, deterministik tek-satır
tool özeti. Bizim 5-alan kartının `sonuç` alanını besler (naif "ilk satırı kes"
yerine "özün özü"nü çıkarır). Sıfır LLM, guarded (asla çökmez).

Hermes `_summarize_tool_result()` mantığının domain-bağımsız genellemesi:
kategori ipucu (tool_meta) + ad deseni + çıktı gövdesinden salient alanları ayıklar.
"""
from __future__ import annotations
import json
import re
from typing import Optional


def _lines(s: str) -> int:
    return len(s.splitlines())


def tool_gist(name: str, args: dict, output: str,
              meta: Optional[dict] = None) -> str:
    """Bir tool sonucunun tip-farkında tek-satır özü. Guarded."""
    try:
        return _gist(name, args or {}, output or "", meta or {})
    except Exception:
        n = len(output) if isinstance(output, str) else 0
        return f"{name} → ({n} chars)"


def _arg(args: dict, *keys):
    for k in keys:
        v = args.get(k)
        if v not in (None, ""):
            return v
    return None


def _gist(name: str, args: dict, output: str, meta: dict) -> str:
    cat = (meta.get("cat") if meta else None)
    low = name.lower()
    n = len(output)
    first = output.splitlines()[0] if output else ""

    # --- agregasyon/sayı: sayıyı BİREBİR taşı (verbatim niyeti) ---
    if any(k in low for k in ("aggregate", "count", "group_by", "stats",
                              "sprint_report", "worklog", "run_sql")):
        num = re.search(r"(→|=|:)\s*([\d.,]+)\b", first)
        head = first.split(":")[0].split("(")[0].strip() or name
        if num:
            return f"{head} → {num.group(2)}"
        return first[:70] or f"{name} → hesaplandı"

    # --- arama/resolve/listeleme: sorgu + sonuç sayısı ---
    if cat == "search" or any(k in low for k in ("search", "resolve", "list", "find")):
        q = _arg(args, "query", "ref", "name", "cql", "filter", "title", "q") or "*"
        m = re.search(r"(\d+)\s+(sonuç|results?|match)", output)
        return f"'{q}' → {m.group(1) if m else '?'} sonuç ({n} chars)"

    # --- outline / blok listesi: sürüm + blok sayısı ---
    if "outline" in low:
        did = _arg(args, "document_id", "doc_id", "path") or "?"
        vb = re.search(r"\(v(\d+),\s*(\d+)\s*(blok|blocks?)\)", output)
        if vb:
            return f"{did} → {vb.group(2)} blok (v{vb.group(1)})"
        return f"{did} → outline ({n} chars)"

    # --- yazma: OK satırı + üretilen id/sürüm ---
    if cat == "write" or output.startswith(("OK", "ok", "Done", "done")):
        idm = re.search(r"(document_id|df_id|artifact_id|id)=(\S+)", first)
        vm = re.search(r"\bv(\d+)\b", first)
        verb = re.sub(r"^(OK|ok|Done)[:\s·-]*", "", first).split("·")[0].strip()[:40]
        tail = (f"{idm.group(1)}={idm.group(2)}" if idm else "") + (f" v{vm.group(1)}" if vm else "")
        return f"{verb} {tail}".strip() or first[:70]

    # --- tekil okuma: kimlik + bir salient alan ---
    if cat == "read" or low.startswith("get") or "_get_" in low:
        ident = _arg(args, "key", "issue_key", "page_id", "project_key", "ref",
                     "unit", "id", "path") or ""
        # salient alan: status / title / cost / ilk "anahtar: değer"
        sal = (re.search(r"status:\s*(.+)", output)
               or re.search(r"planned_cost:\s*(.+)", output)
               or re.search(r"'([^']+)'", first)
               or re.search(r":\s*(.+)", output))
        salv = sal.group(1).strip()[:40] if sal else first[:40]
        head = f"{ident} → " if ident else ""
        return f"{head}{salv} ({_lines(output)} satır)"

    # --- fallback ---
    return f"{first[:60]} ({n} chars)"
