"""MAF modunun ana süreçteki yarısı: alt süreci kur, satırlarını topla.

`maf_runner.py` ayrı bir sanal ortamda (`.venv-maf`) koşuyor ve bu depodaki
hiçbir şeyi içe aktarmıyor. Buradaki iş onu doğru ortam değişkenleriyle
başlatmak ve stdout'undan gelen `##STAGE` / `##OUT` satırlarını ayrıştırmak —
taramanın yıllardır kullandığı protokolün aynısı.

### Neden alt süreç

`agent-framework` ile `autogen-*` aynı ortamda çözülemedi: pip'in çözücüsü on
dakikada bir karar veremedi. İkisini ayırmak riski sıfırlıyor, ve mimari olarak
da doğru: bunlar iki ayrı çerçeve, aynı süreçte yaşamaları için bir sebep yok.

### Sınır

Bu mod bir **karşılaştırma yüzeyi**, üretim yolu değil. Boru hattının tool'ları,
kapısı, belleği ve taraması AutoGen tarafında; MAF tarafında tek örnek tool var.
Amaç, aynı soruyu iki çerçevede koşturup farkı ölçüde göstermek.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator

import config
import stages as stages_module

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv-maf"
PYTHON = VENV / "bin" / "python"
RUNNER = Path(__file__).resolve().parent / "maf_runner.py"

TAG_STAGE = "##STAGE "
TAG_OUT = "##OUT "


def available() -> bool:
    """MAF modu açılabilir mi: ayrı ortam kurulu ve canlı model var mı."""
    return PYTHON.exists() and RUNNER.exists() and config.live_llm_available()


def report() -> dict[str, Any]:
    """Modun durumu — arayüz düğmeyi buna bakarak gösteriyor."""
    return {
        "available": available(),
        "venv": str(VENV) if PYTHON.exists() else "",
        "runner": RUNNER.name,
        "why": ("hazır" if available() else
                "ayrı sanal ortam kurulu değil" if not PYTHON.exists() else
                "canlı model yok"),
    }


def _env(approval: str) -> dict[str, str]:
    """Alt sürecin ortamı. Anahtar yalnız burada geçiyor, komut satırında değil."""
    return {
        **os.environ,
        "VC_MAF_MODEL": config.MODEL_TIERS.get("mid", ""),
        "VC_LLM_BASE_URL": config.LLM_BASE_URL,
        "VC_LLM_API_KEY": config.LLM_API_KEY,
        "VC_MAF_APPROVAL": approval,
    }


async def run(question: str, *, approval: str = "never_require",
              bus: Any = None) -> AsyncIterator[dict[str, Any]]:
    """MAF turunu koştur ve olaylarını akıt.

    Aşamalar `bus`'a yayınlanıyor; ekran MAF turunu AutoGen turuyla aynı
    şekilde çiziyor çünkü ikisi de aynı katalogdan geçiyor.
    """
    if not available():
        yield {"type": "error", "message": "MAF modu hazır değil: " + report()["why"]}
        return

    proc = await asyncio.create_subprocess_exec(
        str(PYTHON), str(RUNNER), question,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        env=_env(approval),
    )
    assert proc.stdout is not None
    try:
        async for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith(TAG_STAGE):
                try:
                    payload = json.loads(line[len(TAG_STAGE):])
                except (json.JSONDecodeError, ValueError):
                    continue
                if bus is not None:
                    # `emit` katalogdan geçiyor: tanımadığı kimliği sessizce
                    # düşürüyor, yani ekranda görünmeyen bir aşama katalog
                    # eksikliği demek.
                    bus.emit(str(payload.get("id", "")), **(payload.get("meta") or {}))
                continue
            if line.startswith(TAG_OUT):
                try:
                    yield json.loads(line[len(TAG_OUT):])
                except (json.JSONDecodeError, ValueError):
                    continue
                continue
            if line.strip():
                # Alt sürecin ham çıktısı. Yutmuyoruz: MAF'ın kendi uyarıları
                # bu modun en çok öğrettiği şeylerden biri.
                yield {"type": "log", "text": line[:400]}
    finally:
        if proc.returncode is None:
            proc.terminate()
        await proc.wait()
