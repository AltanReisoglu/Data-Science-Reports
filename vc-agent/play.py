#!/usr/bin/env python
"""One-shot LLM connectivity check for the VC pipeline.

    .venv/bin/python play.py

It answers a single question — *can this project reach a model right now?* — using
the exact configuration the pipeline itself reads (`config.py`, which loads
`.env`). So a green run here means `/api/chat` will work, and a red one names the
missing piece instead of failing three layers deep during a scan.

It never prints a secret: the key is shown as a length and a short prefix, which
is enough to tell "the right key is loaded" from "an empty string is loaded"
without putting the value on screen or in a transcript.

It sends exactly one tiny completion, and only when the endpoint, key and model
are all present. A tester that fired half-configured — guessing a URL, or firing
a key at an endpoint it was not issued for — would be worse than no tester.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))

import config  # noqa: E402 — this import is what loads the .env


def _mask(value: str) -> str:
    """Enough to recognise the value; not enough to leak it."""
    if not value:
        return "— boş —"
    if len(value) <= 8:
        return f"<{len(value)} karakter>"
    return f"<{len(value)} karakter · {value[:6]}…{value[-2:]}>"


def _report_config() -> list[str]:
    """What the pipeline sees, masked. Returns the list of missing pieces."""
    print("── yüklenen yapılandırma " + "─" * 44)
    print(f"  .env dosyası     : {_env_path()}")
    print(f"  VC_LLM_BASE_URL  : {config.LLM_BASE_URL or '— boş —'}")
    print(f"  VC_LLM_API_KEY   : {_mask(config.LLM_API_KEY)}")
    for tier in ("cheap", "mid", "strong"):
        print(f"  model[{tier:6}]   : {config.MODEL_TIERS.get(tier) or '— boş —'}")
    print(f"  live_llm_available(): {config.live_llm_available()}")

    missing = []
    if not config.LLM_BASE_URL:
        missing.append("VC_LLM_BASE_URL (endpoint)")
    if not config.LLM_API_KEY:
        missing.append("VC_LLM_API_KEY")
    if not config.MODEL_TIERS.get("mid"):
        missing.append("VC_MODEL_MID (en az bir model adı)")
    return missing


def _env_path() -> str:
    for candidate in getattr(config, "_ENV_CANDIDATES", []):
        if candidate.exists():
            return str(candidate)
    return "bulunamadı"


async def _one_call() -> None:
    """Send a single minimal completion through the pipeline's own client."""
    from autogen_core.models import UserMessage

    ledger = __import__("engine").Ledger()
    client = ledger.raw_client("mid")

    print("\n── tek çağrı " + "─" * 56)
    print(f"  model: {config.MODEL_TIERS['mid']}  →  {config.LLM_BASE_URL}")
    started = time.perf_counter()
    try:
        result = await client.create(
            [UserMessage(content="Reply with the single word: pong", source="user")]
        )
    except Exception as exc:  # noqa: BLE001 — a failed probe is the answer, not a crash
        elapsed = time.perf_counter() - started
        print(f"  ✗ BAŞARISIZ ({elapsed:.2f}s)")
        print(f"    {type(exc).__name__}: {exc}")
        _diagnose(exc)
        return

    elapsed = time.perf_counter() - started
    usage = getattr(result, "usage", None)
    print(f"  ✓ CEVAP ({elapsed:.2f}s)")
    print(f"    içerik      : {str(result.content)[:80]!r}")
    print(f"    finish      : {getattr(result, 'finish_reason', '?')}")
    if usage is not None:
        print(f"    token       : prompt {usage.prompt_tokens} · "
              f"completion {usage.completion_tokens}")


def _diagnose(exc: Exception) -> None:
    """Turn the usual failures into the one line that names the fix."""
    text = f"{type(exc).__name__} {exc}".lower()
    hints = [
        ("model_info is required", "Model adı bilinen bir OpenAI modeli değil → "
         "config.MODEL_INFO doldurulmalı (docs/06 §3)."),
        ("401", "Anahtar reddedildi: VC_LLM_API_KEY bu endpoint'e ait değil."),
        ("403", "Anahtar bu endpoint'te yetkisiz."),
        ("404", "Yol yanlış: VC_LLM_BASE_URL '/v1' ile bitmeli mi, bak."),
        ("connect", "Ağ: endpoint'e ulaşılamıyor — URL, VPN, DNS."),
        ("timeout", "Endpoint bağlandı ama cevap vermedi (TLS reset / yavaş model)."),
        ("name or service not known", "DNS: VC_LLM_BASE_URL host'u çözülemiyor."),
    ]
    for needle, hint in hints:
        if needle in text:
            print(f"    → {hint}")
            return


async def main() -> int:
    print("VC pipeline · LLM erişim testi\n")
    missing = _report_config()
    if missing:
        print("\n── sonuç " + "─" * 60)
        print("  ✗ Endpoint yapılandırılmadı. Eksik:")
        for item in missing:
            print(f"      · {item}")
        print("\n  .env şu dört satırı istiyor (anahtarı buraya yazma, dosyaya yaz):")
        print("      VC_LLM_BASE_URL=https://<endpoint>/v1")
        print("      VC_LLM_API_KEY=<anahtar>")
        print("      VC_MODEL_CHEAP=<model> · VC_MODEL_MID=<model> · VC_MODEL_STRONG=<model>")
        return 1
    await _one_call()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
