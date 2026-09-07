"""Dizin artifact'i — tar olarak saklanır, açılmış hâlde yerleştirilir.

## Neden var

LLM `/output/model.v1/` altına birkaç dosya bırakabiliyor (çok dosyalı model,
varlıklarıyla birlikte HTML rapor). Süpürme bunu tek bir `.tar` artifact'ine
çeviriyor; yerleştirme ise geri AÇIYOR, yani sonraki çalıştırma gerçek bir
dizin görüyor.

## Kritik nokta: aç → yeniden paketle → aynı hash

Yerleştirmede tar açılıyor, süpürmede dizin YENİDEN paketleniyor. İkisi aynı
baytı üretmezse "bunu ben verdim" kontrolü tutmaz ve her çalıştırma aynı
içeriği bir kez daha yükler. Bu yüzden `_dizini_paketle` mtime/uid/gid/uname
VE kip'i sabitliyor (2026-09-07: kip eklendi — açma sırasında umask'a göre
değişiyordu).
"""

from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "sandbox_image"))
sys.path.insert(0, str(KOK / "src" / "grounded_assistant" / "artifacts"))
os.environ.setdefault("TOOL_GATEWAY_ENDPOINT", "http://yok/mcp")
os.environ.setdefault("ARTIFACT_SERVICE_ENDPOINT", "http://yok")

import entrypoint  # noqa: E402
import sidecar  # noqa: E402

WF = "wf-bu"


def tar_uret(tmp_path: Path, icerik: dict[str, str]) -> bytes:
    """Süpürmenin ürettiğiyle AYNI kuralla paketler."""
    kaynak = tmp_path / "_kaynak"
    kaynak.mkdir(exist_ok=True)
    for ad, veri in icerik.items():
        (kaynak / ad).write_text(veri)
    hedef = tmp_path / "_paket.tar"
    sidecar._dizini_paketle(str(kaynak), str(hedef))
    return hedef.read_bytes()


class SahteUst:
    def __init__(self, kayitlar: list[dict], icerik: dict[str, bytes]):
        self.kayitlar = kayitlar
        self.icerik = icerik
        self.yuklenenler: list[tuple[str, list[str]]] = []

    def list_all(self):
        return self.kayitlar

    def fetch_to_file(self, name, hedef, workflow_id=None):
        ham = self.icerik.get(f"{workflow_id}/{name}", self.icerik.get(name))
        if ham is None:
            return None
        Path(hedef).write_bytes(ham)
        return {"artifact_id": f"art_{name}", "name": name,
                "content_type": "application/x-tar", "size_bytes": len(ham)}

    def put_file(self, path, content_type, name, ttl_seconds=None, parents=None):
        self.yuklenenler.append((name, list(parents or [])))
        return {"artifact_id": f"art_{name}", "name": name, "size_bytes": 0,
                "content_type": content_type, "parents": list(parents or [])}


@pytest.fixture
def ortam(tmp_path, monkeypatch):
    cikti, scratch = tmp_path / "output", tmp_path / "scratch"
    cikti.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(sidecar, "OUTPUT_DIR", str(cikti))
    monkeypatch.setattr(sidecar, "SCRATCH_DIR", str(scratch))
    monkeypatch.setattr(sidecar, "WORKFLOW_ID", WF)
    monkeypatch.setattr(sidecar, "_sunulan_ozet", {})
    monkeypatch.setattr(sidecar, "_istenen_kimlik", set())
    monkeypatch.setattr(sidecar, "_yerlesen_kimlik", {})
    return cikti


def test_dizin_acilmis_halde_yerlesir(ortam, tmp_path, monkeypatch):
    ust = SahteUst(
        [{"name": "model.v1.tar", "workflow_id": WF, "artifact_id": "art_m"}],
        {"model.v1.tar": tar_uret(tmp_path, {"agirlik.json": "{}", "kunye.txt": "v1"})},
    )
    monkeypatch.setattr(sidecar, "istemci", ust)

    sidecar.yerlestir()

    assert (ortam / "model.v1" / "agirlik.json").read_text() == "{}"
    assert (ortam / "model.v1" / "kunye.txt").read_text() == "v1"
    # Tar'ın kendisi kalmamalı: kalsaydı `os.listdir("/output")` LLM'e hem
    # dizini hem arşivi gösterirdi.
    assert not (ortam / "model.v1.tar").exists()


def test_dokunulmayan_dizin_yeniden_yuklenmez(ortam, tmp_path, monkeypatch):
    """Aç → yeniden paketle → aynı hash. Tutmazsa depo her turda şişer."""
    ust = SahteUst(
        [{"name": "model.v1.tar", "workflow_id": WF, "artifact_id": "art_m"}],
        {"model.v1.tar": tar_uret(tmp_path, {"a.json": "1", "b.json": "2"})},
    )
    monkeypatch.setattr(sidecar, "istemci", ust)

    sidecar.yerlestir()
    sidecar.supur()

    assert ust.yuklenenler == []


def test_degistirilen_dizin_yeniden_yuklenir(ortam, tmp_path, monkeypatch):
    ust = SahteUst(
        [{"name": "model.v1.tar", "workflow_id": WF, "artifact_id": "art_m"}],
        {"model.v1.tar": tar_uret(tmp_path, {"a.json": "1"})},
    )
    monkeypatch.setattr(sidecar, "istemci", ust)

    sidecar.yerlestir()
    (ortam / "model.v1" / "c.json").write_text("3")   # LLM ekledi
    sidecar.supur()

    assert [ad for ad, _ in ust.yuklenenler] == ["model.v1.tar"]


def test_yol_gecisli_tar_disari_yazmiyor(ortam, tmp_path):
    """Depoya kötü niyetli bir tar girmiş olabilir; açan taraf kaynağına
    güvenmemeli (CWE-22 / CVE-2007-4559). `filter="data"` reddediyor."""
    kotu = tmp_path / "kotu.tar"
    with tarfile.open(kotu, "w") as t:
        veri = tmp_path / "yuk"
        veri.write_text("ele gecirildi")
        t.add(veri, arcname="../../kacis.txt")

    hedef = ortam / "acilan"
    hedef.mkdir()
    for ac in (sidecar._tari_ac, entrypoint._tari_ac):
        try:
            ac(str(kotu), str(hedef))
        except Exception:  # noqa: BLE001 — reddetmek de geçerli bir sonuç
            pass
        assert not (tmp_path / "kacis.txt").exists()
        assert not (ortam.parent / "kacis.txt").exists()


def test_baska_calistirmanin_dizini_load_artifact_ile_aciliyor(tmp_path, monkeypatch):
    """`/output`'a inmez; açıkça istenince `/artifacts/<wf>/<dizin>/` olur."""
    d = tmp_path / "artifacts"
    d.mkdir()
    monkeypatch.setattr(entrypoint, "ARTIFACTS_DIR", str(d))

    ust = SahteUst([], {"wf-baska/model.v1.tar": tar_uret(tmp_path, {"a.json": "1"})})
    yol = entrypoint._load_artifact_uret(ust)("wf-baska", "model.v1.tar")

    assert Path(yol) == d / "wf-baska" / "model.v1"
    assert (Path(yol) / "a.json").read_text() == "1"
    assert not (d / "wf-baska" / "model.v1.tar").exists()
