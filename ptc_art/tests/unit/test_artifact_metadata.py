"""Artifact metadata deposunun davranış testleri.

Odak, CRUD'un kendisi değil — dokümanlarda karar verdiğimiz İKİ DEĞİŞMEZ KURAL:
immutability ve keşif. Bir de TTL, çünkü GC ona dayanacak.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from grounded_assistant.artifacts.metadata import (
    ArtifactExists,
    ArtifactMeta,
    open_sqlite,
)

WF = "wf_42"


@pytest.fixture
def store():
    return open_sqlite(":memory:")


def meta(
    artifact_id: str,
    name: str = "extract.tickets",
    *,
    content_hash: str = "sha256:aaa",
    node_id: str | None = "extract",
    created_at: datetime | None = None,
    ttl_seconds: int | None = None,
    parents: tuple[str, ...] = (),
) -> ArtifactMeta:
    return ArtifactMeta(
        artifact_id=artifact_id,
        name=name,
        workflow_id=WF,
        run_id="run_1",
        node_id=node_id,
        content_hash=content_hash,
        content_type="parquet",
        size_bytes=1024,
        storage_uri=f"s3://artifacts/{WF}/{artifact_id}.parquet",
        parents=parents,
        owner="altan",
        created_at=created_at or datetime.now(UTC),
        ttl_seconds=ttl_seconds,
    )


def test_gidis_donus_alanlari_korur(store):
    kaynak = meta("art_001", parents=("art_000",), ttl_seconds=3600)
    store.create(kaynak)
    geri = store.get("art_001")
    assert geri == kaynak, "dataclass eşitliği: hiçbir alan yolda kaybolmamalı"


def test_olmayan_artifact_none_doner(store):
    assert store.get("art_yok") is None


def test_ayni_id_ikinci_kez_yazilamaz(store):
    """Immutability: yeni sürüm = yeni artifact_id. Üzerine yazma YOK."""
    store.create(meta("art_001"))
    with pytest.raises(ArtifactExists):
        store.create(meta("art_001", name="baska-isim"))
    assert store.get("art_001").name == "extract.tickets", "ilk kayıt bozulmamalı"


def test_ayni_isimde_en_yenisi_doner(store):
    """Aynı isim birden çok kez üretilir; `get_artifact(name=...)` sonuncuyu ister."""
    t0 = datetime.now(UTC)
    store.create(meta("art_001", created_at=t0))
    store.create(meta("art_002", created_at=t0 + timedelta(seconds=30)))
    assert store.latest_by_name(WF, "extract.tickets").artifact_id == "art_002"


def test_baska_workflow_gorunmez(store):
    """Kapsam sınırı workflow — başka workflow'un artifact'i isimle bulunamaz."""
    store.create(meta("art_001"))
    assert store.latest_by_name("wf_baska", "extract.tickets") is None


def test_icerik_hashiyle_bulunur(store):
    """cached() bunun üzerine kurulacak: aynı içerik zaten üretilmiş mi?"""
    store.create(meta("art_001", content_hash="sha256:abc"))
    assert store.find_by_hash(WF, "sha256:abc").artifact_id == "art_001"
    assert store.find_by_hash(WF, "sha256:yok") is None


def test_node_bazli_kesif(store):
    """D okuması: transform'un agent'ı extract koşarken orada değildi."""
    store.create(meta("art_001", name="extract.tickets", node_id="extract"))
    store.create(meta("art_002", name="transform.ozet", node_id="transform"))

    hepsi = store.list(WF)
    assert {m.artifact_id for m in hepsi} == {"art_001", "art_002"}

    sadece_extract = store.list(WF, node_id="extract")
    assert [m.name for m in sadece_extract] == ["extract.tickets"]


def test_ttl_dolmasi(store):
    eski = datetime.now(UTC) - timedelta(hours=2)
    store.create(meta("art_eski", created_at=eski, ttl_seconds=60))
    store.create(meta("art_kalici", content_hash="sha256:bbb"))  # ttl yok

    dolmus = store.expired()
    assert [m.artifact_id for m in dolmus] == ["art_eski"]
    assert store.get("art_kalici").expires_at() is None


def test_silme_kaydi_kaldirir(store):
    store.create(meta("art_001"))
    assert store.delete("art_001") is True
    assert store.get("art_001") is None
    assert store.delete("art_001") is False, "ikinci silme False dönmeli"
