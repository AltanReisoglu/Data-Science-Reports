"""Dizin çıktıları (2026-09-06) — KFP launcher'ının dizin desteğinin karşılığı.

## Neden eklendi

Süpürme döngüsü `not os.path.isfile(yol): continue` diyordu; yani `/output`
altındaki bir DİZİN sessizce atlanıyordu. Canlı doğrulandı: LLM
`/output/model.v1/` altına üç dosya yazdı, hiçbiri saklanmadı, hiçbir yerde
hata çıkmadı — modelin "kaydettim" sanmasına yetecek kadar sessiz.

KFP launcher'ı bu ayrımı yapıyor: *"the launcher determines artifact type
(file vs directory), then uploads from local path to object storage URI."*

## Neden tek tar, KFP gibi çoklu nesne değil

KFP bir dizini nesne deposuna özyinelemeli yüklüyor (1 artifact = N nesne).
Bizde bu, künyenin dört değişmezini bozardı: `content_hash`, dedup,
`size_bytes`, akışlı tek-nesne put/get. Tar'layınca dördü de duruyor.

Bedeli: dizinden TEK dosya ayrı çekilemiyor. Sandbox efemer olduğu için
pratikte dizin zaten bütün hâlinde isteniyor.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tarfile
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "sandbox_image"))
sys.path.insert(0, str(KOK / "src" / "grounded_assistant" / "artifacts"))
os.environ.setdefault("TOOL_GATEWAY_ENDPOINT", "http://yok/mcp")

import entrypoint  # noqa: E402


class SahteIstemci:
    def __init__(self, depo: dict[str, bytes] | None = None):
        self.depo = depo or {}
        self.yuklenenler: list[tuple[str, str, bytes]] = []

    def list(self, workflow_id, node_id=None):
        return [{"name": ad, "artifact_id": f"art_{ad}",
                 "workflow_id": entrypoint.WORKFLOW_ID} for ad in self.depo]

    def fetch_to_file(self, name, hedef, workflow_id=None):
        if name not in self.depo:
            return None
        Path(hedef).write_bytes(self.depo[name])
        return {"artifact_id": f"art_{name}", "name": name,
                "content_type": "application/x-tar", "size_bytes": len(self.depo[name])}

    def put_file(self, path, content_type, name, ttl_seconds=None, parents=None):
        self.yuklenenler.append((name, content_type, Path(path).read_bytes()))
        return {"artifact_id": f"art_{name}", "name": name, "content_type": content_type,
                "size_bytes": 0, "parents": list(parents or [])}


@pytest.fixture
def ortam(tmp_path, monkeypatch):
    cikti = tmp_path / "output"
    cikti.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(entrypoint, "OUTPUT_DIR", str(cikti))
    monkeypatch.setattr(entrypoint, "SCRATCH_DIR", str(scratch))
    return cikti


def dizin_kur(kok: Path, icerik: dict[str, str]) -> Path:
    for gorece, veri in icerik.items():
        hedef = kok / gorece
        hedef.parent.mkdir(parents=True, exist_ok=True)
        hedef.write_text(veri)
    return kok



def tar_uret(tmp_path, icerik: dict[str, str]) -> bytes:
    """Paketleme SIDECAR'da (süpürme oraya taşındı); test onu kullanıyor."""
    import sidecar  # noqa: PLC0415

    kaynak = dizin_kur(tmp_path / "kaynak", icerik)
    sidecar._dizini_paketle(str(kaynak), str(tmp_path / "k.tar"))
    return (tmp_path / "k.tar").read_bytes()


def test_derin_yol_istenince_dizin_iniyor(ortam, tmp_path):
    """`/output/model.v1/weights.json` okunmak isteniyor, dizin daha inmemiş."""
    istemci = SahteIstemci({"model.v1.tar": tar_uret(tmp_path, {
        "weights.json": '{"w":1}', "alt/n.txt": "derin"})})
    entrypoint._tembel_oku(str(ortam / "model.v1" / "weights.json"), istemci,
                           entrypoint.Depo({"model.v1.tar"}))

    assert (ortam / "model.v1" / "weights.json").read_text() == '{"w":1}'
    assert (ortam / "model.v1" / "alt" / "n.txt").read_text() == "derin"
    # Soy ağacı ARTIK BURADA izlenmiyor — sidecar sunduğu artifact'leri kendi
    # kaydediyor (bkz. test_sidecar.py::test_soy_sidecarin_SUNDUKLARINDAN_geliyor).


def test_manifestte_olmayan_dizin_icin_aga_cikilmaz(ortam, tmp_path):
    istemci = SahteIstemci({"baska.tar": tar_uret(tmp_path, {"a": "b"})})
    entrypoint._tembel_oku(str(ortam / "yok" / "dosya.txt"), istemci,
                           entrypoint.Depo({"baska.tar"}))
    assert not (ortam / "yok").exists()


def test_yol_gecisli_tar_disari_yazmiyor(ortam, tmp_path):
    """Depoya süpürme yoluyla kötü niyetli bir tar girmiş olabilir.

    Arşivi biz üretmiş olsak da açan taraf kaynağına güvenmemeli (CWE-22 /
    CVE-2007-4559). `filter="data"` bunu reddediyor.
    """
    kotu = tmp_path / "kotu.tar"
    with tarfile.open(kotu, "w") as t:
        veri = tmp_path / "yuk"
        veri.write_text("ele gecirildi")
        t.add(veri, arcname="../../kacis.txt")

    hedef = ortam / "acilan"
    hedef.mkdir()
    try:
        entrypoint._tari_ac(str(kotu), str(hedef))
    except Exception:
        pass  # reddetmek de geçerli bir sonuç

    assert not (tmp_path / "kacis.txt").exists()
    assert not (ortam.parent / "kacis.txt").exists()


# -- iki kök: /output kendi, /artifacts/<wf>/ başkaları (2026-09-06) --------


def test_baska_calistirmanin_dizini_kimlikle_aciliyor(ortam, tmp_path, monkeypatch):
    """`/artifacts/<wf>/<dizin>/<dosya>` de açılabilmeli.

    Dizin açma yolu önce yalnızca `/output`'u biliyordu; kabul testinde
    `/artifacts/<wf>/model.v1/alt/derin.txt` FileNotFoundError veriyordu.
    """
    art = tmp_path / "artifacts"
    art.mkdir()
    monkeypatch.setattr(entrypoint, "ARTIFACTS_DIR", str(art))
    monkeypatch.setattr(entrypoint, "WORKFLOW_ID", "wf_ben")

    istemci = SahteIstemci({"model.tar": tar_uret(tmp_path, {
        "w.json": '{"w":1}', "alt/derin.txt": "derin"})})
    depo = entrypoint.Depo(kendi=set(), digerleri={"wf_baska": {"model.tar"}})

    entrypoint._tembel_oku(str(art / "wf_baska" / "model" / "alt" / "derin.txt"),
                           istemci, depo)

    assert (art / "wf_baska" / "model" / "alt" / "derin.txt").read_text() == "derin"


def test_baskasinin_dizini_output_a_SIZMIYOR(ortam, tmp_path, monkeypatch):
    """İzolasyonun dizin tarafı: başka run'ın dizini kendi /output'una inmemeli."""
    art = tmp_path / "artifacts"
    art.mkdir()
    monkeypatch.setattr(entrypoint, "ARTIFACTS_DIR", str(art))
    istemci = SahteIstemci({"model.tar": tar_uret(tmp_path, {"w.json": "1"})})
    depo = entrypoint.Depo(kendi=set(), digerleri={"wf_baska": {"model.tar"}})

    entrypoint._tembel_oku(str(ortam / "model" / "w.json"), istemci, depo)

    assert not (ortam / "model").exists(), "/output'a sızdı"

