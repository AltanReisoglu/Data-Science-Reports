"""Sebep-farkında retry sayacı (2026-09-08).

Kod hatası ile ağ denemesi AYNI bütçeden yiyemez: ağ engeli sonrası tekrar
bilerek dar (2026-09-01 kararı), kod hatası ise düzeltilebilir olduğu için
daha geniş bir bütçe alıyor.

Bu dosya iki şeyi koruyor:

1. `ag_engeli_gibi` sınıflandırması — Codex'in `is_likely_sandbox_denied()`
   deseni (PTC_Error_Recovery_Piyasa_Arastirmasi.md §7.2).
2. `sandbox_run_count(durumlar=...)`'ın DOĞRU ALANA baktığı.

(2) için test var çünkü ilk hâli `detail`'e bakıyordu; oysa
`TraceEntry(access_path, detail, status, ts)` sırasında durum `status`'te,
`detail`'de çalıştırma kimliği duruyor. Sayaç hep 0 dönüyordu, yani dar sınır
FİİLEN UYGULANMIYORDU ve hiçbir test bunu görmüyordu.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from grounded_assistant.models import SandboxRun, SandboxRunStatus, ag_engeli_gibi
from grounded_assistant.trace import Trace

AG_SAYILAN = ("denied_action", "error:ag")


def _kosum(durum: SandboxRunStatus, hata: str | None = None) -> SandboxRun:
    an = datetime.now(UTC)
    return SandboxRun(run_id="r", code="x", status=durum,
                      started_at=an, finished_at=an, error_message=hata)


@pytest.mark.parametrize("metin", [
    "socket.gaierror: [Errno -3] Temporary failure in name resolution",
    "ConnectionError: HTTPSConnectionPool(host='x'): Max retries exceeded",
    "OSError: [Errno 111] Connection refused",
    "OSError: [Errno 101] Network is unreachable",
    "socket.gaierror: [Errno -2] Name or service not known",
])
def test_ag_izleri_taniniyor(metin: str) -> None:
    assert ag_engeli_gibi(metin)


@pytest.mark.parametrize("metin", [
    "KeyError: 'yok'",
    "ModuleNotFoundError: No module named 'sklearn'",
    "ZeroDivisionError: division by zero",
    "FileNotFoundError: [Errno 2] No such file or directory: '/output/yok.csv'",
    "",
    None,
])
def test_kod_hatasi_ag_sayilmiyor(metin: str | None) -> None:
    assert not ag_engeli_gibi(metin)


def test_ag_denemesi_dar_sayaca_yaziliyor() -> None:
    t = Trace(); t.mark()
    t.record_sandbox_run(_kosum(
        SandboxRunStatus.ERROR,
        'File "/sandbox/code.py", line 2\\n'
        "socket.gaierror: [Errno -3] Temporary failure in name resolution"))
    assert t.sandbox_run_count() == 1
    assert t.sandbox_run_count(AG_SAYILAN) == 1


def test_kod_hatasi_dar_sayaca_yazilmiyor() -> None:
    t = Trace(); t.mark()
    t.record_sandbox_run(_kosum(SandboxRunStatus.ERROR, "KeyError: 'yok'"))
    assert t.sandbox_run_count() == 1
    assert t.sandbox_run_count(AG_SAYILAN) == 0


def test_basarili_kosum_hicbir_hata_sayacina_girmiyor() -> None:
    t = Trace(); t.mark()
    t.record_sandbox_run(_kosum(SandboxRunStatus.SUCCESS))
    assert t.sandbox_run_count() == 1
    assert t.sandbox_run_count(AG_SAYILAN) == 0
    assert t.sandbox_run_count(("error",)) == 0


def test_karisik_tur_ayri_sayiliyor() -> None:
    """Asıl senaryo: bir ağ denemesi + iki kod hatası."""
    t = Trace(); t.mark()
    t.record_sandbox_run(_kosum(SandboxRunStatus.ERROR, "gaierror: name resolution"))
    t.record_sandbox_run(_kosum(SandboxRunStatus.ERROR, "NameError: name 'df' is not defined"))
    t.record_sandbox_run(_kosum(SandboxRunStatus.ERROR, "TypeError: unhashable"))
    assert t.sandbox_run_count() == 3
    assert t.sandbox_run_count(AG_SAYILAN) == 1
    assert t.sandbox_run_count(("error",)) == 2


def test_mark_turu_sifirliyor() -> None:
    """Sayaç TUR başına: önceki turun ağ denemesi bu turu daraltmamalı."""
    t = Trace(); t.mark()
    t.record_sandbox_run(_kosum(SandboxRunStatus.ERROR, "gaierror: name resolution"))
    assert t.sandbox_run_count(AG_SAYILAN) == 1
    t.mark()
    assert t.sandbox_run_count() == 0
    assert t.sandbox_run_count(AG_SAYILAN) == 0
