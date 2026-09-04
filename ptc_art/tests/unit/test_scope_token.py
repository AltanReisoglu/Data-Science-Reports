"""Kapsam jetonu testleri.

Bu jeton, "artifact_id çağıranın workflow'una ait mi" kontrolünü dekoratif
olmaktan çıkaran şey. Sandbox'ta çalışan kodu LLM yazdığı için, kapsamı
çağıranın kendisi bildirseydi başka bir workflow adı yazmasını engelleyen hiçbir
şey olmazdı.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from grounded_assistant.artifacts.scope import (
    InvalidScopeToken,
    Scope,
    issue_token,
    verify_token,
)

ANAHTAR = "test-imza-anahtari"
KAPSAM = Scope(workflow_id="wf_42", run_id="run_1", owner="altan", node_id="extract")


def test_gidis_donus():
    geri = verify_token(ANAHTAR, issue_token(ANAHTAR, KAPSAM))
    assert geri == KAPSAM


def test_baska_anahtarla_uretilen_jeton_reddedilir():
    """Sandbox'ın imza anahtarı yok — kendi jetonunu üretemez."""
    sahte = issue_token("saldirganin-anahtari", Scope("wf_kurban", "run_x", "altan"))
    with pytest.raises(InvalidScopeToken):
        verify_token(ANAHTAR, sahte)


def test_govde_kurcalanirsa_reddedilir():
    """Jetonu okuyup workflow_id'yi değiştirmek işe yaramamalı."""
    jeton = issue_token(ANAHTAR, KAPSAM)
    govde, imza = jeton.split(".", 1)
    bozuk = f"{govde[:-4]}AAAA.{imza}"
    with pytest.raises(InvalidScopeToken):
        verify_token(ANAHTAR, bozuk)


def test_suresi_dolan_jeton_reddedilir():
    gecmis = datetime.now(UTC) - timedelta(hours=2)
    jeton = issue_token(ANAHTAR, KAPSAM, ttl_seconds=60, now=gecmis)
    with pytest.raises(InvalidScopeToken):
        verify_token(ANAHTAR, jeton)


def test_sure_dolmadan_gecerli():
    jeton = issue_token(ANAHTAR, KAPSAM, ttl_seconds=300)
    assert verify_token(ANAHTAR, jeton).workflow_id == "wf_42"


@pytest.mark.parametrize("bozuk", ["", "nokta-yok", "a.b.c.d", "...", "x."])
def test_bicimsiz_jetonlar(bozuk):
    with pytest.raises(InvalidScopeToken):
        verify_token(ANAHTAR, bozuk)


def test_node_id_opsiyonel():
    kapsam = Scope(workflow_id="wf_1", run_id="run_1", owner="altan")
    assert verify_token(ANAHTAR, issue_token(ANAHTAR, kapsam)).node_id is None
