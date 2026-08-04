"""
tools.py — Gerçek tool'lar (sample_repo üzerinde çalışır) + OpenAI-uyumlu şemalar.

ek-a mentalitesi (report/ek-a-tool-referans.md):
  - name spesifik ve fiil içerir
  - description NE ZAMAN çağrılacağını söyler, sadece ne yaptığını değil
  - her property'nin açıklaması + format örneği var
  - path'ler kanonik forma çözülüp sample_repo İÇİNDE kaldığı doğrulanır
    (§13 ek-a §13 güvenlik: model çıktısı güvenilmezdir)
"""
from __future__ import annotations
from pathlib import Path

from config import SAMPLE_REPO


def _safe_path(rel: str) -> Path:
    """rel yolunu kanonikleştir; sample_repo dışına çıkıyorsa reddet."""
    root = SAMPLE_REPO.resolve()
    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"Yol repo dışında: {rel}")
    return target


# --- tool implementasyonları --------------------------------------------

def read_file(path: str) -> str:
    """Bir dosyanın tam içeriğini döndürür."""
    p = _safe_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Yok: {path}")
    lines = p.read_text().splitlines()
    return "\n".join(f"{i+1:4d}  {ln}" for i, ln in enumerate(lines))


def list_dir(path: str = ".") -> str:
    """Bir dizindeki dosyaları listeler."""
    p = _safe_path(path)
    if not p.is_dir():
        raise NotADirectoryError(f"Dizin değil: {path}")
    return "\n".join(sorted(f.name for f in p.iterdir()))


def grep(pattern: str, path: str = ".") -> str:
    """Bir desenle eşleşen satırları dosya:satır formatında arar."""
    root = _safe_path(path)
    files = [root] if root.is_file() else sorted(root.glob("*.py"))
    hits = []
    for f in files:
        for i, ln in enumerate(f.read_text().splitlines(), 1):
            if pattern in ln:
                hits.append(f"{f.name}:{i}: {ln.strip()}")
    return "\n".join(hits) if hits else f"(eşleşme yok: {pattern})"


def edit_file(path: str, old: str, new: str) -> str:
    """Bir dosyada tam bir metin parçasını yenisiyle değiştirir."""
    p = _safe_path(path)
    text = p.read_text()
    if old not in text:
        raise ValueError(f"Bulunamadı, değiştirilemedi: {old!r}")
    if text.count(old) > 1:
        raise ValueError(f"Belirsiz (>1 eşleşme): {old!r}")
    p.write_text(text.replace(old, new))
    return f"OK: {path} güncellendi ({old!r} → {new!r})"


def run_tests() -> str:
    """Depodaki basit doğrulamayı çalıştırır: server.health() portu döndürüyor mu."""
    import importlib, sys
    sys.path.insert(0, str(SAMPLE_REPO.resolve()))
    for m in ("config", "server"):
        if m in sys.modules:
            importlib.reload(sys.modules[m])
    try:
        import server  # type: ignore
        port = server.health()["port"]
        return f"TEST: health() port={port}"
    finally:
        sys.path.pop(0)


DISPATCH = {
    "read_file": read_file,
    "list_dir": list_dir,
    "grep": grep,
    "edit_file": edit_file,
    "run_tests": run_tests,
}


# --- OpenAI-uyumlu tool şemaları (function calling) ----------------------

SCHEMAS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": ("Bir dosyanın tam içeriğini satır numaralarıyla döndürür. "
                        "Bir dosyanın içeriğini görmen gerektiğinde çağır. "
                        "Aynı dosyayı zaten okuduysan tekrar okuma."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string",
                     "description": "Repo köküne göre yol, ör. 'config.py'"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": ("Bir dizindeki dosyaları listeler. Repoda hangi dosyalar "
                        "var bilmediğinde önce bunu çağır."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Dizin yolu, varsayılan '.'"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": ("Bir metin desenini repodaki .py dosyalarında arar, "
                        "dosya:satır formatında döndürür. Bir tanımın/değişkenin "
                        "nerede olduğunu bulmak için dosyaları tek tek okumak yerine bunu kullan."),
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Aranacak metin, ör. 'PORT'"},
            "path": {"type": "string", "description": "Aranacak dizin/dosya, varsayılan '.'"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": ("Bir dosyada tam bir metin parçasını yenisiyle değiştirir. "
                        "Bir değişiklik yapman gerektiğinde çağır. old parçası "
                        "dosyada tam olarak bir kez geçmeli."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Değiştirilecek dosya"},
            "old": {"type": "string", "description": "Değiştirilecek tam metin"},
            "new": {"type": "string", "description": "Yeni metin"}},
            "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": ("Depodaki doğrulamayı çalıştırır ve health() portunu döndürür. "
                        "Bir değişikliğin işe yarayıp yaramadığını görmek için çağır."),
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    # --- PTC (§11 K3 / §12.10): tool'ları TEK TEK değil, KOD olarak çağır ---
    {"type": "function", "function": {
        "name": "run_code",
        # gerçek API'de: "allowed_callers": ["code_execution_20250825"]
        "description": (
            "Python kodu çalıştırır; kodun içinde read_file/list_dir/grep'i "
            "fonksiyon olarak çağırabilirsin. ÇOK dosyayı tarayıp yalnızca "
            "SONUCU lazım olduğunda bunu kullan — her dosyayı tek tek read_file "
            "ile okuyup bağlamı doldurmak yerine döngüyle tara ve yalnızca "
            "print ettiğini geri al. Ara sonuçlar bağlama girmez, sadece print girer."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string",
                     "description": ("Çalıştırılacak Python. Sonucu print et. "
                                     "Örn: for f in ['a.py','b.py']: print(grep('PORT', f))")}},
            "required": ["code"]}}},
    # --- CWL delimiter (§5.2): ajanın kendi trace'ini tiplemesi ---
    {"type": "function", "function": {
        "name": "delimiter",
        "description": (
            "Çalışmanı bölümlere ayır. Bir bilgi toplama dizisine başlarken "
            "action='start', type='expl' çağır; onu bitirince action='end' ve "
            "ne öğrendiğini description'da yaz. Bir değişiklik yapmadan önce "
            "action='start', type='act' çağır ve dependencies'e hangi keşiflere "
            "dayandığını yaz. Bu, gereksiz geçmişin sonradan güvenle atılmasını sağlar."),
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["start", "end"]},
            "name": {"type": "string", "description": "Bölümün kısa adı (start'ta)"},
            "type": {"type": "string", "enum": ["expl", "act"],
                     "description": "expl=keşif, act=eylem (start'ta)"},
            "dependencies": {"type": "array", "items": {"type": "string"},
                             "description": "act'in dayandığı expl adları"},
            "description": {"type": "string",
                            "description": "expl end'inde: ne öğrenildi"}},
            "required": ["action"]}}},
]
