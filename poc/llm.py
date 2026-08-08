"""
llm.py — ince OpenAI-uyumlu istemci (stdlib urllib; SDK gerekmez).

Bizim internal endpoint'e (gemma, OpenAI-uyumlu) konuşur. .env'den yüklenir:
  LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME
.env aranan yerler: poc-trace-compaction/.env, ../.env, ./.env  (ilk bulunan kazanır).

Anahtar ASLA yazdırılmaz. Native tool-calling (tools/tool_calls) destekler.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ENV_CANDIDATES = [
    _HERE.parent / "poc-trace-compaction" / ".env",
    _HERE.parent / ".env",
    _HERE / ".env",
]


def _load_env() -> None:
    for p in _ENV_CANDIDATES:
        if not p.exists():
            continue
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()

BASE_URL = os.getenv("LLM_BASE_URL", "")
MODEL = os.getenv("LLM_MODEL_NAME", "")
_KEY = os.getenv("LLM_API_KEY", "")


def available() -> bool:
    return bool(BASE_URL and _KEY and MODEL)


def why_unavailable() -> str:
    miss = [k for k, v in (("LLM_BASE_URL", BASE_URL), ("LLM_API_KEY", _KEY),
                           ("LLM_MODEL_NAME", MODEL)) if not v]
    return f"eksik: {', '.join(miss)}" if miss else "hazır"


def chat(messages: list[dict], tools: list[dict] | None = None,
         tool_choice: str = "auto", max_tokens: int = 800,
         temperature: float = 0.2, timeout: int = 60) -> dict:
    """Bir chat/completions çağrısı yap; assistant mesaj dict'ini döndür.

    Dönen dict: {"content": str|None, "tool_calls": [...] | None}.
    Hata → RuntimeError (çağıran yakalar)."""
    if not available():
        raise RuntimeError(f"LLM yapılandırılmadı ({why_unavailable()})")
    payload: dict = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {_KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"LLM HTTP {e.code}: {e.read()[:200].decode(errors='replace')}") from e
    except Exception as e:  # ağ, timeout vb.
        raise RuntimeError(f"LLM bağlantı hatası: {type(e).__name__}: {e}") from e
    msg = data["choices"][0]["message"]
    return {"content": msg.get("content"), "tool_calls": msg.get("tool_calls")}
