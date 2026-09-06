"""Otomatik soy ağacı — proxy ve sidecar BİRLİKTE (2026-09-06'da yeniden yazıldı).

## Neden yeniden yazıldı, ikinci kez

İlk hâli `put_artifact`/`get_artifact` yüzeyini sınıyordu; o yüzey kaldırıldı.
İkinci hâli `entrypoint`'in süpürmesini sınıyordu; **o da kaldırıldı** —
süpürme Argo'nun `wait` container'ı gibi sidecar'a taşındı.

Bu hâli iki parçanın birleşimini sınıyor:

    sidecar proxy'den bayt sunar   ->  artifact_id'yi KENDİ kaydeder
    sandbox /output'a dosya yazar
    sidecar süpürür                ->  ebeveyn = sunduğu artifact'ler

## Neden bu daha güçlü

Defter artık sandbox'ta DEĞİL. Eskiden `okunanlar` kümesi LLM'in çalıştığı
süreçteydi — kodu yazan taraf soy ağacını etkileyebilirdi. Şimdi kayıt
sidecar'da: LLM'in etkileyebileceği tek şey hangi dosyayı okuduğu, ki bu
zaten soyun tanımı.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "sandbox_image"))
sys.path.insert(0, str(KOK / "src" / "grounded_assistant" / "artifacts"))
os.environ.setdefault("ARTIFACT_SERVICE_ENDPOINT", "http://yok")

import sidecar  # noqa: E402


class SahteUst:
    """Sidecar'ın YUKARI akıştaki Artifact Service istemcisi."""

    def __init__(self, depo: dict[str, bytes] | None = None):
        self.depo = depo or {}
        self.yuklenenler: list[tuple[str, list[str]]] = []

    def fetch_to_file(self, name, hedef, workflow_id=None):
        if name not in self.depo:
            return None
        Path(hedef).write_bytes(self.depo[name])
        return {"artifact_id": f"art_{name}", "name": name,
                "content_type": "text/csv", "size_bytes": len(self.depo[name])}

    def put_file(self, path, content_type, name, ttl_seconds=None, parents=None):
        self.yuklenenler.append((name, list(parents or [])))
        return {"artifact_id": f"art_{name}", "name": name, "size_bytes": 0,
                "content_type": content_type, "parents": list(parents or [])}


@pytest.fixture
def ortam(tmp_path, monkeypatch):
    cikti, scratch = tmp_path / "output", tmp_path / "scratch"
    cikti.mkdir(); scratch.mkdir()
    monkeypatch.setattr(sidecar, "OUTPUT_DIR", str(cikti))
    monkeypatch.setattr(sidecar, "SCRATCH_DIR", str(scratch))
    monkeypatch.setattr(sidecar, "_sunulan_ozet", {})
    monkeypatch.setattr(sidecar, "_sunulan_kimlik", set())
    return cikti


def sun(ust, ad: str) -> None:
    """Proxy'nin `/fetch` yolunun yaptığının aynısı — sidecar sunduğunu kaydeder."""
    ham = ust.depo[ad]
    sidecar._sunulan_ozet[ad] = hashlib.sha256(ham).hexdigest()
    sidecar._sunulan_kimlik.add(f"art_{ad}")


def test_sunulan_dosya_uretilenin_ebeveyni_olur(ortam, monkeypatch):
    """ASIL KURAL: sidecar'ın sunduğu girdi, ürettiğinin ebeveynidir."""
    ust = SahteUst({"ham.csv": b"a,b\n1,2\n"})
    monkeypatch.setattr(sidecar, "istemci", ust)

    sun(ust, "ham.csv")                          # sandbox okudu
    (ortam / "turev.parquet").write_bytes(b"x")  # sandbox yazdı
    sidecar.supur()

    assert dict(ust.yuklenenler)["turev.parquet"] == ["art_ham.csv"]


def test_iki_girdiden_tek_cikti_iki_ebeveyn_alir(ortam, monkeypatch):
    """Birleştirme (join): grafik ağaç değil DAG."""
    ust = SahteUst({"sol.csv": b"1", "sag.csv": b"2"})
    monkeypatch.setattr(sidecar, "istemci", ust)

    sun(ust, "sol.csv"); sun(ust, "sag.csv")
    (ortam / "birlesik.parquet").write_bytes(b"x")
    sidecar.supur()

    assert dict(ust.yuklenenler)["birlesik.parquet"] == ["art_sag.csv", "art_sol.csv"]


def test_hicbir_sey_sunulmadiysa_oksuz_kalir(ortam, monkeypatch):
    """Sıfırdan üretilen veri gerçekten öksüzdür — uydurma ebeveyn takmıyoruz."""
    ust = SahteUst()
    monkeypatch.setattr(sidecar, "istemci", ust)
    (ortam / "kok.csv").write_bytes(b"a\n1\n")
    sidecar.supur()
    assert ust.yuklenenler == [("kok.csv", [])]


def test_dizin_ciktisi_da_soy_tasiyor(ortam, monkeypatch):
    ust = SahteUst({"ham.csv": b"a\n1\n"})
    monkeypatch.setattr(sidecar, "istemci", ust)

    sun(ust, "ham.csv")
    (ortam / "model").mkdir()
    (ortam / "model" / "w.json").write_text("{}")
    sidecar.supur()

    assert dict(ust.yuklenenler)["model.tar"] == ["art_ham.csv"]


def test_sadece_okuyan_calistirma_hicbir_sey_uretmez(ortam, monkeypatch):
    """Okunan dosya "üretilmiş" sayılmamalı — sahte soy kenarı doğardı.

    Ölçü sidecar'ın SUNDUĞU baytın sha256'sı; dosya değişmediyse atlanıyor.
    """
    ust = SahteUst({"ham.csv": b"a\n1\n"})
    monkeypatch.setattr(sidecar, "istemci", ust)

    sun(ust, "ham.csv")
    (ortam / "ham.csv").write_bytes(ust.depo["ham.csv"])   # tembel okuma indirdi
    sidecar.supur()

    assert ust.yuklenenler == []
