"""
tools_product.py — GERÇEK 119 ürün tool'unu POC'a bağlayan adaptör.

poc-trace-compaction/product_tools.py'daki SCHEMAS/DISPATCH/TOOL_META'yı kullanır.
TOOL_META zaten bizim ledger sözleşmesidir: {cat, resource(args), resource_param, ttl, verbatim}.
Burada onu POC'un ToolResult modeline (tool_type + resource) çeviririz — 13 stratejinin
HİÇBİRİ değişmeden gerçek Jira/NETA/LDAP/Confluence/doküman tool'ları üzerinde çalışır.

cat → tool_type eşlemesi (stratejilerin tip-özel yolları için):
  read   → read_file   (Cline dedup, Roo fold, Claude clear, bizim dedup/staleness)
  search → grep        (Headroom SearchCompressor, gist)
  write  → write_file  (mutasyon → staleness tetikler)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# product_tools.py sadece stdlib kullanır → dizini sys.path'e EKLEMEDEN, dosyadan izole
# yükle. (sys.path'e eklersek poc/agent.py ↔ poc-trace-compaction/agent.py çakışırdı.)
_PT_PATH = Path(__file__).resolve().parent.parent / "poc-trace-compaction" / "product_tools.py"
_spec = importlib.util.spec_from_file_location("product_tools_real", _PT_PATH)
_pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pt)

TOOL_META = _pt.TOOL_META
DISPATCH = _pt.DISPATCH

# LLM'e verilecek şemalar — sadece DISPATCH'te karşılığı olanlar (delimiter vb. hariç)
SCHEMAS = [s for s in _pt.SCHEMAS
           if s.get("function", {}).get("name") in DISPATCH]

_CAT_TO_TYPE = {"read": "read_file", "search": "grep", "write": "write_file"}

PRODUCT_SYSTEM = (
    "Sen bir kurumsal asistansın. Jira (iş takibi), NETA (proje portföyü/bütçe), "
    "LDAP (dizin), Confluence (wiki) ve doküman üretimi (docx/pdf/pptx/xlsx) + veri "
    "analizi tool'larıyla soruları yanıtla. Serbest ad geçen projeyi/kişiyi ÖNCE resolve "
    "et (jira_resolve_project, neta_resolve_project, ldap_resolve_person), sonra key/id "
    "isteyen tool'u çağır. Aynı veriyi gereksiz tekrar çekme. Yeterince bilgi toplayınca "
    "tool çağırmayı bırak ve kısa net bir Türkçe yanıt yaz."
)


def tool_type(name: str) -> str:
    cat = (TOOL_META.get(name) or {}).get("cat", "read")
    return _CAT_TO_TYPE.get(cat, "read_file")


def category(name: str) -> str:
    return (TOOL_META.get(name) or {}).get("cat", "other")


def resource_of(name: str, args: dict) -> str:
    """Ledger sözleşmesi: meta'daki extractor fn(args) → kaynak anahtarı (dedup/staleness için)."""
    meta = TOOL_META.get(name)
    if not meta:
        return ""
    fn = meta.get("resource")
    if fn is not None:
        try:
            return str(fn(args))
        except Exception:
            return ""
    param = meta.get("resource_param")
    return str(args.get(param, "")) if param else ""


def run(name: str, args: dict) -> str:
    """Gerçek tool'u çağır: DISPATCH[name](**args)."""
    fn = DISPATCH.get(name)
    if fn is None:
        return f"[bilinmeyen tool: {name}]"
    try:
        return str(fn(**args))
    except Exception as e:
        return f"[tool hata: {type(e).__name__}: {e}]"
