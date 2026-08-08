"""
tool_schemas.py — 6 mock tool'un OpenAI function-calling şemaları + çalıştırıcı.

Ajanın (gerçek LLM) çağırabilmesi için OpenAI `tools` biçimi. Her araç, harness'taki
MockTools üreticisine bağlanır. tool adı → (tool_type, ana argüman) eşlemesi de burada.
"""
from __future__ import annotations

from harness import MockTools

# --- OpenAI tools[] şeması ---
SCHEMAS = [
    {"type": "function", "function": {
        "name": "run_terminal",
        "description": "Bir kabuk komutu çalıştır (npm test, build, vb.). Çıktı satırlarını döndürür.",
        "parameters": {"type": "object", "properties": {
            "cmd": {"type": "string", "description": "çalıştırılacak komut, ör. 'npm test'"}},
            "required": ["cmd"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Bir kaynak dosyayı oku ve tam içeriğini döndür.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "dosya yolu, ör. 'src/server.py'"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "web_extract",
        "description": "Bir web sayfasının metin içeriğini çıkar.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "sayfa URL'i"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "take_snapshot",
        "description": "Bir tarayıcı sayfasının erişilebilirlik (accessibility) snapshot'ını al.",
        "parameters": {"type": "object", "properties": {
            "page": {"type": "string", "description": "sayfa URL'i veya adı"}},
            "required": ["page"]}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Kod tabanında bir desen ara, eşleşen satırları döndür.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "aranacak desen"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Bir dosyaya yaz/düzenle (mutasyon — dosyanın sürümünü artırır).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "yazılacak dosya yolu"},
            "content": {"type": "string", "description": "(isteğe bağlı) yeni içerik"}},
            "required": ["path"]}}},
]

# tool adı → (tool_type, ana argüman adı)
_META = {
    "run_terminal": ("terminal", "cmd"),
    "read_file": ("read_file", "path"),
    "web_extract": ("web_extract", "url"),
    "take_snapshot": ("take_snapshot", "page"),
    "grep": ("grep", "query"),
    "write_file": ("write_file", "path"),
}


def tool_type(name: str) -> str:
    return _META.get(name, (name, ""))[0]


def resource_of(name: str, args: dict) -> str:
    _, key = _META.get(name, (name, ""))
    return str(args.get(key, "")) if key else ""


def run(tools: MockTools, name: str, args: dict) -> str:
    """Adı verilen tool'u MockTools ile çalıştır; ham çıktıyı döndür."""
    r = resource_of(name, args)
    fn = {
        "run_terminal": lambda: tools.terminal(r),
        "read_file": lambda: tools.read_file(r),
        "web_extract": lambda: tools.web_extract(r),
        "take_snapshot": lambda: tools.take_snapshot(r),
        "grep": lambda: tools.grep(r),
        "write_file": lambda: tools.write_file(r),
    }.get(name)
    if fn is None:
        return f"[bilinmeyen tool: {name}]"
    return fn()
