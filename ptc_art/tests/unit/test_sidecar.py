"""Artifact sidecar — Argo'nun `wait` container'ının karşılığı (2026-09-06).

## Neden bu dosya var

Süpürme 2026-09-06'ya kadar `entrypoint.py`'deydi: yani LLM'in kodunun
çalıştığı container'ın İÇİNDE, ortam değişkenindeki kapsam jetonuyla. Bu,
`§9.6`'da KFP'yi eleştirdiğimiz konumun aynısıydı — sarmalayıcı, sınır değil.

Argo dört farklı yerleşimi (docker/kubelet/k8sapi/pns) deneyip v3.4'te hepsini
kaldırmış; kalan model init + `wait` sidecar. Bu dosya o modelin bizdeki
karşılığını sınıyor.

## Asıl kazanç

Yalnızca jetonu taşımak YETMEZDİ — LLM localhost proxy'ye de aynı çağrıyı
atabilirdi. Kazanç, **yükleme kararının sandbox'ta verilmemesi**: sidecar neyi
yükleyeceğine `/output`'a bakarak kendi karar veriyor. Ad seçmek, TTL koymak,
depo kökü belirlemek, süpürme kuralını atlamak artık mümkün değil.
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
os.environ.setdefault("ARTIFACT_SERVICE_ENDPOINT", "http://yok")

import sidecar  # noqa: E402


class SahteIstemci:
    def __init__(self):
        self.yuklenenler: list[tuple[str, str, list[str], bytes]] = []

    def put_file(self, path, content_type, name, ttl_seconds=None, parents=None):
        self.yuklenenler.append((name, content_type, list(parents or []),
                                 Path(path).read_bytes()))
        return {"artifact_id": f"art_{name}", "name": name, "size_bytes": 0,
                "content_type": content_type, "parents": list(parents or [])}


@pytest.fixture
def ortam(tmp_path, monkeypatch):
    cikti, scratch = tmp_path / "output", tmp_path / "scratch"
    cikti.mkdir(); scratch.mkdir()
    monkeypatch.setattr(sidecar, "OUTPUT_DIR", str(cikti))
    monkeypatch.setattr(sidecar, "SCRATCH_DIR", str(scratch))
    istemci = SahteIstemci()
    monkeypatch.setattr(sidecar, "istemci", istemci)
    monkeypatch.setattr(sidecar, "_sunulan_ozet", {})
    monkeypatch.setattr(sidecar, "_sunulan_kimlik", set())
    return cikti, istemci


def adlar(istemci):
    return sorted(a for a, _, _, _ in istemci.yuklenenler)


# -- süpürme ---------------------------------------------------------------


def test_dosyalar_yukleniyor(ortam):
    cikti, istemci = ortam
    (cikti / "rapor.csv").write_text("a,b\n1,2\n")
    (cikti / "ozet.json").write_text("{}")
    sidecar.supur()
    assert adlar(istemci) == ["ozet.json", "rapor.csv"]


def test_dizin_tek_tar_oluyor(ortam):
    cikti, istemci = ortam
    (cikti / "model.v1" / "alt").mkdir(parents=True)
    (cikti / "model.v1" / "w.json").write_text('{"w":1}')
    (cikti / "model.v1" / "alt" / "n.txt").write_text("derin")
    sidecar.supur()

    assert adlar(istemci) == ["model.v1.tar"]
    ham = istemci.yuklenenler[0][3]
    arsiv = cikti.parent / "k.tar"
    arsiv.write_bytes(ham)
    with tarfile.open(arsiv) as t:
        assert sorted(t.getnames()) == ["alt/n.txt", "w.json"]


def test_paketleme_TEKRARLANABILIR(ortam, tmp_path):
    """DEDUP DEĞİŞMEZİ: aynı içerik iki kez paketlenince aynı baytlar çıkmalı.

    Düz `tar` mtime/uid/gid gömer; bu olmadan aynı dizini iki kez süpürmek iki
    ayrı nesne yaratır ve içerik-hash dedup'ı sessizce ölür.
    """
    for ad in ("d1", "d2"):
        d = tmp_path / ad
        (d / "b").mkdir(parents=True)
        (d / "a.txt").write_text("bir")
        (d / "b" / "c.txt").write_text("iki")
    os.utime(tmp_path / "d2" / "a.txt", (0, 0))

    h1 = sidecar._dizini_paketle(str(tmp_path / "d1"), str(tmp_path / "1.tar"))
    h2 = sidecar._dizini_paketle(str(tmp_path / "d2"), str(tmp_path / "2.tar"))
    assert h1 == h2
    assert h1 == hashlib.sha256((tmp_path / "1.tar").read_bytes()).hexdigest()


def test_nokta_ile_baslayan_atlanir(ortam):
    cikti, istemci = ortam
    (cikti / ".gizli").write_text("x")
    (cikti / "__pycache__").mkdir()
    (cikti / "gercek.txt").write_text("y")
    sidecar.supur()
    assert adlar(istemci) == ["gercek.txt"]


def test_biri_patlarsa_digerleri_yuklenir(ortam, monkeypatch):
    """Süpürme best-effort: bir dosya reddedilse de diğerleri gitmeli."""
    cikti, istemci = ortam
    (cikti / "iyi.txt").write_text("y")
    (cikti / "kotu").mkdir()
    monkeypatch.setattr(sidecar, "_dizini_paketle",
                        lambda *a: (_ for _ in ()).throw(OSError("disk dolu")))
    sidecar.supur()
    assert adlar(istemci) == ["iyi.txt"]


def test_turkce_ad_duzeltiliyor(ortam):
    cikti, istemci = ortam
    (cikti / "Şubat raporu.csv").write_text("x")
    sidecar.supur()
    ad = adlar(istemci)[0]
    assert ad.replace("-", "").isalnum() or "." in ad
    assert ad[0].isalnum()


# -- "bunu ben verdim" kuralı ---------------------------------------------


def test_sunulan_dosya_geri_yuklenmiyor(ortam):
    """LLM sadece OKUDUYSA, sidecar'ın verdiği dosya "üretilmiş" sayılmamalı.

    Defter SANDBOX'TA DEĞİL — sidecar sunduğu baytın sha256'sını kendisi
    tutuyor, dolayısıyla LLM bunu kurcalayamıyor.
    """
    cikti, istemci = ortam
    icerik = b"depodan geldi"
    (cikti / "girdi.csv").write_bytes(icerik)
    sidecar._sunulan_ozet["girdi.csv"] = hashlib.sha256(icerik).hexdigest()

    sidecar.supur()
    assert istemci.yuklenenler == []


def test_sunulan_dosya_DEGISIRSE_yukleniyor(ortam):
    cikti, istemci = ortam
    (cikti / "girdi.csv").write_bytes(b"eski")
    sidecar._sunulan_ozet["girdi.csv"] = hashlib.sha256(b"eski").hexdigest()
    (cikti / "girdi.csv").write_bytes(b"LLM degistirdi")

    sidecar.supur()
    assert adlar(istemci) == ["girdi.csv"]


# -- soy ağacı -------------------------------------------------------------


def test_soy_sidecarin_SUNDUKLARINDAN_geliyor(ortam):
    """Ebeveynler = sidecar'ın bu koşuda sunduğu artifact'ler.

    Kayıt sandbox'ta olmadığı için kurcalanamıyor — eskiden `okunanlar` kümesi
    LLM'in çalıştığı süreçteydi.
    """
    cikti, istemci = ortam
    sidecar._sunulan_kimlik.update({"art_ham", "art_kunye"})
    (cikti / "turev.parquet").write_bytes(b"x")

    sidecar.supur()
    assert istemci.yuklenenler[0][2] == ["art_ham", "art_kunye"]


def test_hicbir_sey_sunulmadiysa_oksuz(ortam):
    cikti, istemci = ortam
    (cikti / "kok.csv").write_bytes(b"x")
    sidecar.supur()
    assert istemci.yuklenenler[0][2] == []


# -- yüzey: proxy'de YAZMA uç noktası yok ---------------------------------


def test_proxyde_yazma_yolu_YOK():
    """ASIL GÜVENLİK ÖZELLİĞİ: LLM proxy üzerinden bir şey yükleyemez.

    Jetonu sidecar'a taşımak tek başına yetmezdi — LLM localhost'a aynı çağrıyı
    atabilirdi. Kazanç, yükleme uç noktasının HİÇ OLMAMASI: neyin yükleneceğine
    sidecar `/output`'a bakarak karar veriyor.
    """
    assert not hasattr(sidecar.Proxy, "do_POST")
    assert not hasattr(sidecar.Proxy, "do_PUT")
    assert not hasattr(sidecar.Proxy, "do_DELETE")
    assert hasattr(sidecar.Proxy, "do_GET")
