"""Soy ağacı SORGUSU (2026-09-05) — kayıt defterinden grafiği kurmak.

`tests/unit/test_soy_agaci.py` yazma tarafını sınıyor (sandbox `parents`'ı
doğru dolduruyor mu). Bu dosya okuma tarafını: kayıt defterinde duran
kenarlardan "bu verinin ulaştığı düğümler" ve "bu artifact'in ürünleri"
grafiği çıkıyor mu.

Bu ikisi §11.10'daki "soy ağacı: kaydediliyor, keşifte KULLANILMIYOR"
satırının iki yarısı.
"""

from __future__ import annotations

import pytest

from grounded_assistant.artifacts.metadata import open_sqlite
from grounded_assistant.artifacts.service import ArtifactService, ScopeViolation
from grounded_assistant.artifacts.store import BucketConfig, ObjectStore

from test_artifact_service import SahteMinio  # noqa: E402 — pytest rootdir sys.path

WF = "wf_soy"
SAHIP = "altan"


@pytest.fixture
def service():
    cfg = BucketConfig(name="artifacts", host="localhost", port=9000,
                       access_key="a", secret_key="b", secure=False)
    return ArtifactService(metadata=open_sqlite(":memory:"),
                           objects=ObjectStore(cfg, client=SahteMinio()))


def yaz(service, ad, parents=(), workflow_id=WF, node_id=None):
    return service.create(
        f"{ad} içeriği", name=ad, workflow_id=workflow_id, run_id="run_1",
        owner=SAHIP, parents=tuple(parents), node_id=node_id,
    )


@pytest.fixture
def zincir(service):
    """ham → temiz → {ozet, grafik};  ayrıca ham → etiketler → ozet (DAG).

        ham ─┬─> temiz ─┬─> ozet   <─┐
             │          └─> grafik   │
             └─> etiketler ──────────┘
    """
    ham = yaz(service, "ham.csv", node_id="extract")
    temiz = yaz(service, "temiz.parquet", [ham.artifact_id], node_id="transform")
    etiket = yaz(service, "etiketler.json", [ham.artifact_id], node_id="transform")
    ozet = yaz(service, "ozet.md", [temiz.artifact_id, etiket.artifact_id], node_id="report")
    grafik = yaz(service, "grafik.png", [temiz.artifact_id], node_id="report")
    return {"ham": ham, "temiz": temiz, "etiket": etiket, "ozet": ozet, "grafik": grafik}


def kimlikler(soy, yon=None):
    return {d["name"] for d in soy["nodes"] if yon is None or d["yon"] == yon}


def test_kokun_urunleri_tum_alt_agac(service, zincir):
    """"Bu verinin ulaştığı düğümler" — ham.csv'den türeyen HER ŞEY."""
    soy = service.lineage(zincir["ham"].artifact_id, owner=SAHIP)

    assert kimlikler(soy, "urun") == {"temiz.parquet", "etiketler.json",
                                      "ozet.md", "grafik.png"}
    assert kimlikler(soy, "ata") == set()   # ham öksüz
    assert kimlikler(soy, "kok") == {"ham.csv"}


def test_yapragin_atalari_koke_kadar(service, zincir):
    """Ters yön: "bu rapor neyden üretildi" — iki koldan ham.csv'ye çıkar."""
    soy = service.lineage(zincir["ozet"].artifact_id, owner=SAHIP)

    assert kimlikler(soy, "ata") == {"temiz.parquet", "etiketler.json", "ham.csv"}
    assert kimlikler(soy, "urun") == set()


def test_derinlik_isaretli(service, zincir):
    """Negatif = ata, pozitif = ürün. UI yerleşimi buna dayanıyor."""
    soy = service.lineage(zincir["temiz"].artifact_id, owner=SAHIP)
    d = {n["name"]: n["depth"] for n in soy["nodes"]}

    assert d["temiz.parquet"] == 0
    assert d["ham.csv"] == -1
    assert d["ozet.md"] == 1 and d["grafik.png"] == 1


def test_ortadaki_dugum_iki_yonu_de_gorur(service, zincir):
    soy = service.lineage(zincir["temiz"].artifact_id, owner=SAHIP)
    assert kimlikler(soy, "ata") == {"ham.csv"}
    assert kimlikler(soy, "urun") == {"ozet.md", "grafik.png"}


def test_kenarlar_yalnizca_grafikteki_dugumler_arasinda(service, zincir):
    """Düğümsüz kenar çizilmemeli — UI'da kopuk ok olurdu."""
    soy = service.lineage(zincir["grafik"].artifact_id, owner=SAHIP)
    var = {n["artifact_id"] for n in soy["nodes"]}
    assert all(k["from"] in var and k["to"] in var for k in soy["edges"])
    # grafik.png'nin atası zinciri: temiz + ham. etiketler DAHİL DEĞİL.
    assert kimlikler(soy) == {"grafik.png", "temiz.parquet", "ham.csv"}


def test_baska_workflowun_soyu_gorunmez(service, zincir):
    """Kapsam: kök doğrulanıyor, kenarlar da aynı workflow listesinden kuruluyor."""
    with pytest.raises(ScopeViolation):
        service.lineage(zincir["ham"].artifact_id, owner="baska-tenant")


def test_silinmis_ebeveyne_giden_kenar_cizilmez(service, zincir):
    """TTL reaper ebeveyni sildiyse çocuk ÖKSÜZ görünür, kopuk ok değil.

    `parents` metni kayıtta durmaya devam ediyor (immutable künye) ama
    grafiğe yalnızca iki ucu da var olan kenarlar giriyor.
    """
    service.delete(zincir["ham"].artifact_id, owner=SAHIP)

    soy = service.lineage(zincir["temiz"].artifact_id, owner=SAHIP)
    assert kimlikler(soy, "ata") == set()
    assert all(k["from"] != zincir["ham"].artifact_id for k in soy["edges"])


def test_dongu_sonsuza_sokmaz(service):
    """Kayıt immutable olduğu için döngü OLUŞMAMALI, ama BFS yine de korumalı."""
    a = yaz(service, "a.csv")
    b = yaz(service, "b.csv", [a.artifact_id])
    # a'yı b'nin çocuğu göstererek elle döngü kur (normal yoldan imkânsız)
    service.metadata.connection.execute(
        "UPDATE artifacts SET parents = ? WHERE artifact_id = ?",
        (f'["{b.artifact_id}"]', a.artifact_id),
    )
    soy = service.lineage(a.artifact_id, owner=SAHIP)
    assert kimlikler(soy) == {"a.csv", "b.csv"}
