"""Doğrudan-yükleme kipi (2026-09-07) — KFP'nin iki kanalı.

Sınanan: sidecar baytı depoya kendisi koyunca kayıt defterinin hâlâ
DOĞRU olması. Servis baytları görmüyor, dolayısıyla tek koruması
`register`ın nesneyi doğrulaması.
"""
from __future__ import annotations
import io, sys
sys.path.insert(0, "tests/integration")
import pytest
from fastapi.testclient import TestClient
from grounded_assistant.artifacts.metadata import open_sqlite
from grounded_assistant.artifacts.scope import Scope, issue_token
from grounded_assistant.artifacts.service import ArtifactService
from grounded_assistant.artifacts.store import BucketConfig, ObjectStore
from test_artifact_http_api import SahteMinio  # aynı sahte istemci

ANAHTAR, WF, SAHIP = "test-imza-anahtari", "wf_42", "altan"


@pytest.fixture
def kur(monkeypatch):
    from services.artifact_service import app as modul
    sahte = SahteMinio()
    cfg = BucketConfig(name="artifacts", host="localhost", port=9000,
                       access_key="a", secret_key="b", secure=False)
    servis = ArtifactService(metadata=open_sqlite(":memory:"),
                             objects=ObjectStore(cfg, client=sahte))
    monkeypatch.setattr(modul, "_SIGNING_KEY", ANAHTAR)
    modul._service.cache_clear()
    monkeypatch.setattr(modul, "_service", lambda: servis)
    return TestClient(modul.app), sahte


def jeton(wf=WF):
    return issue_token(ANAHTAR, Scope(workflow_id=wf, run_id="r1",
                                      owner=SAHIP, node_id=None))


def H():
    return {"X-Scope-Token": jeton()}


def test_allocate_register_akisi(kur):
    c, sahte = kur
    veri = b"a,b\n1,2\n"
    ozet = "sha256:" + __import__("hashlib").sha256(veri).hexdigest()

    assert c.get(f"/artifacts/by-hash/{ozet}", headers=H()).status_code == 404

    a = c.post("/artifacts/allocate", headers=H(),
               json={"name": "x.csv", "content_type": "text/csv"}).json()
    assert a["artifact_id"].startswith("art_")
    sahte.nesneler[a["key"]] = veri            # sidecar'ın yaptığı: doğrudan PUT

    r = c.post("/artifacts/register", headers=H(), json={
        "artifact_id": a["artifact_id"], "name": "x.csv", "content_type": "text/csv",
        "content_hash": ozet, "size_bytes": len(veri), "storage_uri": a["storage_uri"]})
    assert r.status_code == 201, r.text
    assert r.json()["type"] == "system.Dataset"

    # bayt gerçekten okunabiliyor
    assert c.get(f"/artifacts/{a['artifact_id']}", headers=H()).content == veri


def test_yuklenmemis_nesne_KAYIT_ACMAZ(kur):
    """ASIL KORUMA: servis baytı görmüyor. Sidecar yüklemeyi atlarsa kayıt
    defteri var olmayan bir nesneye işaret ederdi."""
    c, _ = kur
    a = c.post("/artifacts/allocate", headers=H(),
               json={"name": "yok.csv", "content_type": "text/csv"}).json()
    r = c.post("/artifacts/register", headers=H(), json={
        "artifact_id": a["artifact_id"], "name": "yok.csv", "content_type": "text/csv",
        "content_hash": "sha256:0", "size_bytes": 8, "storage_uri": a["storage_uri"]})
    assert r.status_code == 409
    assert c.get("/artifacts", headers=H()).json() == []


def test_boyut_tutmazsa_reddediliyor(kur):
    c, sahte = kur
    a = c.post("/artifacts/allocate", headers=H(),
               json={"name": "x.csv", "content_type": "text/csv"}).json()
    sahte.nesneler[a["key"]] = b"kisa"
    r = c.post("/artifacts/register", headers=H(), json={
        "artifact_id": a["artifact_id"], "name": "x.csv", "content_type": "text/csv",
        "content_hash": "sha256:0", "size_bytes": 999, "storage_uri": a["storage_uri"]})
    assert r.status_code == 409


def test_dedup_yuklemeden_once_bulunuyor(kur):
    c, sahte = kur
    veri = b"ayni icerik"
    ozet = "sha256:" + __import__("hashlib").sha256(veri).hexdigest()
    a = c.post("/artifacts/allocate", headers=H(),
               json={"name": "bir.txt", "content_type": "text/plain"}).json()
    sahte.nesneler[a["key"]] = veri
    c.post("/artifacts/register", headers=H(), json={
        "artifact_id": a["artifact_id"], "name": "bir.txt", "content_type": "text/plain",
        "content_hash": ozet, "size_bytes": len(veri), "storage_uri": a["storage_uri"]})

    v = c.get(f"/artifacts/by-hash/{ozet}", headers=H())
    assert v.status_code == 200
    assert v.json()["storage_uri"] == a["storage_uri"]

    # ikinci kayıt AYNI uri'yi gösteriyor, yeni nesne yok
    once = len(sahte.nesneler)
    b = c.post("/artifacts/allocate", headers=H(),
               json={"name": "iki.txt", "content_type": "text/plain"}).json()
    r = c.post("/artifacts/register", headers=H(), json={
        "artifact_id": b["artifact_id"], "name": "iki.txt", "content_type": "text/plain",
        "content_hash": ozet, "size_bytes": len(veri),
        "storage_uri": v.json()["storage_uri"], "dedup": True})
    assert r.status_code == 201
    assert len(sahte.nesneler) == once
    assert c.get(f"/artifacts/{b['artifact_id']}", headers=H()).content == veri


def test_bozuk_ad_allocate_de_reddediliyor(kur):
    c, _ = kur
    r = c.post("/artifacts/allocate", headers=H(),
               json={"name": "../etc/passwd", "content_type": "text/csv"})
    assert r.status_code == 400


def test_uri_ucu_sidecar_icin_storage_uri_veriyor(kur):
    c, sahte = kur
    veri = b"veri"
    a = c.post("/artifacts/allocate", headers=H(),
               json={"name": "u.txt", "content_type": "text/plain"}).json()
    sahte.nesneler[a["key"]] = veri
    c.post("/artifacts/register", headers=H(), json={
        "artifact_id": a["artifact_id"], "name": "u.txt", "content_type": "text/plain",
        "content_hash": "sha256:x", "size_bytes": len(veri),
        "storage_uri": a["storage_uri"]})
    u = c.get("/artifacts/by-name/u.txt/uri", headers=H()).json()
    assert u["storage_uri"] == a["storage_uri"]
    # künye ucu storage_uri'yi HÂLÂ gizliyor — o sandbox'a kadar gidiyor
    k = c.get(f"/artifacts/{a['artifact_id']}/metadata", headers=H()).json()
    assert "storage_uri" not in k
