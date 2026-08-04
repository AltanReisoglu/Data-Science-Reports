"""
tool_registry.py — Ertelenmiş tool + ToolSearch (progressive disclosure; §11 K3 / B.4).

Sorun (§B.4): 50 tool'un şeması ~40K token eder ve hiçbiri kullanılmasa bile
her turda ödenir. Çözüm: tool'u ikiye ayır —
  - RESIDENT (yerleşik): sık kullanılan çekirdek, şeması hep bağlamda
  - DEFERRED (ertelenmiş): uzun kuyruk, bağlamda yalnızca ADI durur (~5 token);
    şema ancak model `tool_search("select:<ad>")` çağırınca yüklenir (+1 tur)

Bu, ACM/ledger'ın "gerekince getir" mantığının TOOL tarafındaki karşılığıdır:
bağlama neyin gireceği kararı — sonlu dikkat bütçesi (§00 tezi).

DÜRÜSTLÜK: gerçek harness'te bu `defer_loading: true` bayrağı + bir
<system-reminder> ile olur (bu oturumun kendisi böyle çalışıyor). Burada aynı
mekanizmayı ölçülebilir biçimde modelliyoruz.
"""
from __future__ import annotations
import json
import re

from config import estimate_tokens


# ToolSearch'ün kendisi her zaman yerleşiktir (yükleyici tool)
TOOL_SEARCH_SCHEMA = {
    "type": "function", "function": {
        "name": "tool_search",
        "description": (
            "Ertelenmiş bir tool'un şemasını yükler. Bağlamda yalnızca adı görünen "
            "bir tool'u kullanman gerektiğinde önce bunu çağır: query='select:<ad>' "
            "ile tam adından yükle, veya anahtar kelimeyle ara. Yüklenince tool "
            "normal şekilde çağrılabilir."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string",
                      "description": "'select:run_code' gibi tam ad, veya 'kod çalıştır' gibi anahtar kelime"}},
            "required": ["query"]}}}


def _schema_tokens(schema: dict) -> int:
    return estimate_tokens(json.dumps(schema, ensure_ascii=False))


class ToolRegistry:
    """Tool'ları resident/deferred olarak ayırır; ToolSearch ile yükleme yapar."""

    def __init__(self, all_schemas: list[dict], resident_names: list[str],
                 deferred_names: list[str]) -> None:
        self.by_name = {s["function"]["name"]: s for s in all_schemas}
        self.resident = set(resident_names)
        self.deferred = set(deferred_names)     # şeması henüz yüklenmemiş
        self.loaded: set[str] = set()           # tool_search ile yüklenmiş deferred'lar
        self.search_log: list[str] = []

    # --- bağlama giren form -------------------------------------------------

    def active_schemas(self) -> list[dict]:
        """Modele tam şemayla giden tool'lar: resident + yüklenmiş deferred + ToolSearch."""
        names = (self.resident | self.loaded) & set(self.by_name)
        out = [self.by_name[n] for n in sorted(names)]
        out.append(TOOL_SEARCH_SCHEMA)
        return out

    def deferred_stub(self) -> str:
        """Yüklenmemiş deferred tool'ların yalnızca AD listesi (<system-reminder> biçimi)."""
        pending = sorted(self.deferred - self.loaded)
        if not pending:
            return ""
        return ("Ertelenmiş tool'lar (şema YÜKLÜ DEĞİL — tool_search "
                "\"select:<ad>\" ile yükle): " + ", ".join(pending))

    # --- ToolSearch (yükleyici) --------------------------------------------

    def tool_search(self, query: str) -> list[dict]:
        """Deferred tool şemalarını yükle. 'select:a,b' tam ad, yoksa anahtar kelime."""
        picked: list[str] = []
        if query.startswith("select:"):
            for w in query[len("select:"):].split(","):
                n = w.strip()
                if n in self.by_name and n not in self.resident:
                    picked.append(n)
        else:
            terms = [t for t in re.split(r"\s+", query.lower()) if t]
            for n in sorted(self.deferred - self.loaded):
                text = (n + " " + self.by_name[n]["function"]["description"]).lower()
                if any(t in text for t in terms):
                    picked.append(n)
        self.loaded |= {p for p in picked if p in self.deferred}
        self.search_log.append(f"tool_search({query!r}) → {picked}")
        return [self.by_name[n] for n in picked if n in self.by_name]

    # --- muhasebe -----------------------------------------------------------

    def context_tokens(self) -> int:
        """Tool bölümünün ŞU AN bağlamda tuttuğu token (deferred yaklaşımı)."""
        schema = sum(_schema_tokens(s) for s in self.active_schemas())
        return schema + estimate_tokens(self.deferred_stub())

    def full_tokens(self) -> int:
        """Taban çizgi: TÜM tool şemaları yüklü olsaydı (ertelemesiz)."""
        return (sum(_schema_tokens(s) for s in self.by_name.values())
                + _schema_tokens(TOOL_SEARCH_SCHEMA))

    def stats(self) -> dict:
        return {"resident": len(self.resident), "deferred": len(self.deferred),
                "loaded": len(self.loaded),
                "context_tokens": self.context_tokens(),
                "full_tokens": self.full_tokens()}


# --- demo için gerçekçi bir "uzun kuyruk" tool seti ---------------------------

def _synthetic_specialist_schemas() -> list[dict]:
    """§B.4'ün '50 tool = 40K token' ölçeğini göstermek için sentetik uzman tool'lar.

    Gerçek repo işlevi yok — yalnızca ertelemenin token etkisini ölçmek için
    şema kütlesi sağlar (uzun kuyruk: nadiren kullanılır)."""
    specs = [
        ("web_search", "İnternette arama yapar; harici bilgi gerektiğinde."),
        ("web_fetch", "Bir URL'nin içeriğini getirir."),
        ("run_sql", "Bir SQL sorgusu çalıştırır ve satırları döndürür."),
        ("http_request", "Rastgele bir HTTP isteği gönderir."),
        ("git_log", "Depo commit geçmişini döndürür."),
        ("git_blame", "Bir dosyanın satır bazında değişim geçmişi."),
        ("send_email", "Bir e-posta gönderir."),
        ("create_ticket", "Bir sorun izleyicide kayıt açar."),
        ("read_pdf", "Bir PDF'ten metin çıkarır."),
        ("render_chart", "Verilerden bir grafik üretir."),
        ("query_metrics", "Gözlemlenebilirlik metriklerini sorgular."),
        ("deploy_service", "Bir servisi ortama dağıtır."),
    ]
    out = []
    for name, desc in specs:
        out.append({"type": "function", "function": {
            "name": name,
            "description": desc + " (bu bir uzman tool'dur; nadiren gerekir)",
            "parameters": {"type": "object", "properties": {
                "arg1": {"type": "string", "description": "birincil argüman"},
                "arg2": {"type": "string", "description": "ikincil argüman, opsiyonel"},
                "options": {"type": "object", "description": "ek seçenekler sözlüğü"}},
                "required": ["arg1"]}}})
    return out


def demo_registry() -> ToolRegistry:
    """POC'nin gerçek tool'ları (çekirdek resident) + PTC/CWL + sentetik uzun kuyruk (deferred)."""
    from tools import SCHEMAS
    all_schemas = list(SCHEMAS) + _synthetic_specialist_schemas()
    resident = ["read_file", "list_dir", "grep", "edit_file", "run_tests"]
    deferred = [s["function"]["name"] for s in all_schemas
                if s["function"]["name"] not in resident]
    return ToolRegistry(all_schemas, resident, deferred)
