"""Sandbox'ın tembel okuma mekanizması (2026-09-04).

Yerini aldığı şey: pod açılışında HER artifact'i `/output`'a indiren prefetch.
Sınanan davranışlar, o mekanizmadan devraldığımız gereksinimler:

  - `/output/x.csv` okunmak istendiğinde dosya YOKSA indiriliyor
  - dosya VARSA ağa hiç çıkılmıyor
  - manifestte olmayan isim için ağa çıkılmıyor (FileNotFoundError normal aksın)
  - `/output` DIŞINDAKİ yollar hiç dokunulmadan geçiyor
  - **indirdiğimiz dosya süpürmede geri yüklenmiyor** — prefetch döneminde
    bulunup düzeltilen kusurun (2026-09-03) tembel yoldaki karşılığı
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "sandbox_image"))
sys.path.insert(0, str(KOK / "src" / "grounded_assistant" / "artifacts"))
os.environ.setdefault("TOOL_GATEWAY_ENDPOINT", "http://yok/mcp")

import entrypoint  # noqa: E402


class SahteIstemci:
    """Yalnızca kullanılan iki metodu taklit eder, çağrıları sayar."""

    def __init__(self, icerik: dict[str, bytes]):
        self.icerik = icerik
        self.indirilenler: list[str] = []
        self.yuklenenler: list[tuple[str, str]] = []

    def list_all(self):
        return [{"name": ad, "artifact_id": f"art_{ad}",
                 "workflow_id": entrypoint.WORKFLOW_ID} for ad in self.icerik]

    def fetch_to_file(self, name, hedef, workflow_id=None):
        self.indirilenler.append(name)
        if name not in self.icerik:
            return None
        Path(hedef).write_bytes(self.icerik[name])
        return {"artifact_id": f"art_{name}", "name": name, "content_type": "text/csv",
                "size_bytes": len(self.icerik[name])}

    def put_file(self, path, content_type, name, ttl_seconds=None):
        self.yuklenenler.append((name, path))
        return {"artifact_id": f"art_{name}", "name": name, "size_bytes": 0,
                "content_type": content_type, "parents": []}


@pytest.fixture
def cikti(tmp_path, monkeypatch):
    d = tmp_path / "output"
    d.mkdir()
    monkeypatch.setattr(entrypoint, "OUTPUT_DIR", str(d))
    return d


def test_eksik_dosya_indirilir(cikti):
    istemci = SahteIstemci({"rapor.csv": b"id,tutar\n1,100\n"})
    hedef = cikti / "rapor.csv"

    entrypoint._tembel_oku(str(hedef), istemci, entrypoint.Depo({"rapor.csv"}))

    assert hedef.read_bytes() == b"id,tutar\n1,100\n"
    assert istemci.indirilenler == ["rapor.csv"]


def test_var_olan_dosya_icin_aga_cikilmaz(cikti):
    istemci = SahteIstemci({"rapor.csv": b"depodaki"})
    hedef = cikti / "rapor.csv"
    hedef.write_bytes(b"yereldeki")

    entrypoint._tembel_oku(str(hedef), istemci, entrypoint.Depo({"rapor.csv"}))

    assert hedef.read_bytes() == b"yereldeki"
    assert istemci.indirilenler == []


def test_manifestte_olmayan_isim_icin_aga_cikilmaz(cikti):
    """Var olmayan dosya için boşuna round-trip yapılmamalı."""
    istemci = SahteIstemci({"baska.csv": b"x"})
    entrypoint._tembel_oku(str(cikti / "yok.csv"), istemci, entrypoint.Depo({"baska.csv"}))
    assert istemci.indirilenler == []


def test_output_disindaki_yol_dokunulmaz(tmp_path, cikti):
    istemci = SahteIstemci({"gizli.csv": b"x"})
    entrypoint._tembel_oku(str(tmp_path / "gizli.csv"), istemci, entrypoint.Depo({"gizli.csv"}))
    assert istemci.indirilenler == []


def test_alt_dizin_dokunulmaz(cikti):
    """Süpürme yalnızca üst düzeye bakıyor; tembel okuma da öyle olmalı."""
    istemci = SahteIstemci({"x.csv": b"a"})
    entrypoint._tembel_oku(str(cikti / "alt" / "x.csv"), istemci, entrypoint.Depo({"x.csv"}))
    assert istemci.indirilenler == []


def test_dosya_nesnesi_ve_url_gecirilir(cikti):
    """`pd.read_csv(io.StringIO(...))` gibi çağrılar patlamamalı."""
    import io

    istemci = SahteIstemci({})
    entrypoint._tembel_oku(io.StringIO("a,b\n1,2\n"), istemci, entrypoint.Depo(set()))
    entrypoint._tembel_oku(None, istemci, entrypoint.Depo(set()))
    assert istemci.indirilenler == []


# -- (süpürme testleri tests/unit/test_sidecar.py'ye taşındı) ------------


# -- manifest --------------------------------------------------------------


def test_manifest_tek_istekle_isimleri_getirir():
    """Kapsam TENANT (2026-09-06): `list_all` — başka workflow'un çıktısı da gelir."""
    istemci = SahteIstemci({"a.csv": b"1", "b.parquet": b"2"})
    depo = entrypoint._manifest(istemci)
    assert depo.kendi == {"a.csv", "b.parquet"}
    assert depo.digerleri == {}, "aynı workflow'unkiler 'diğerleri'ne düşmemeli"


def test_manifest_servis_erisilemezse_bos_doner():
    """Pod açılışı asla engellenmemeli — tembel okuma sessizce devre dışı kalır."""

    class Patlayan:
        def list_all(self, *a, **kw):
            raise RuntimeError("servis kapalı")

    assert entrypoint._manifest(Patlayan()).bos()


# -- pozisyonel get_artifact ve biçim uyarısı (2026-09-04, canlı kullanımda) --


def test_kimlik_bicimi_tanınıyor():
    """`art_` + 12 hex kimliktir; başka her şey isimdir."""
    assert entrypoint._kimlik_mi("art_27dbc4345886") is True
    assert entrypoint._kimlik_mi("ticket.durumlari") is False
    assert entrypoint._kimlik_mi("art_KISA") is False
    assert entrypoint._kimlik_mi("art_ZZZZZZZZZZZZ") is False   # hex değil
    assert entrypoint._kimlik_mi(None) is False


def test_LLM_ARTIFACT_API_SI_YOK():
    """2026-09-06: `put_artifact`/`get_artifact`/`cached` KALDIRILDI.

    Onlar bizim icadımızdı — piyasada emsali yok, ve bu haftanın ciddi
    hatalarının çoğu tam o yüzeyde çıktı. Yerine KFP launcher deseni geçti:
    LLM düz Python yazıyor, taşımayı launcher yapıyor.

    Bu test kasıtlı olarak NEGATİF: yüzeyin geri sızmadığını garanti ediyor.
    """
    assert not hasattr(entrypoint, "_artifact_api")
    # 2026-09-06: aktarım da buradan gitti — süpürme ve yükleme sidecar'da.
    for gitmis in ("_launcher_api", "_ciktilari_supur", "_dizini_supur",
                   "_dizini_paketle", "SCOPE_TOKEN"):
        assert not hasattr(entrypoint, gitmis), f"{gitmis} geri sızdı"


def test_parquet_dosyasina_read_csv_acik_hata_veriyor(tmp_path):
    """Model `pd.read_csv("/output/ticket.durumlari")` deneyince eskiden
    "'utf-8' codec can't decode byte 0xe4" alıyordu — ne olduğu anlaşılmıyor.
    Artık doğru çağrıyı söyleyen bir mesaj dönüyor."""
    yol = tmp_path / "ticket.durumlari"
    yol.write_bytes(b"PAR1" + b"\x00" * 40)

    uyari = entrypoint._bicim_uyari(str(yol), "read_csv")
    assert uyari is not None
    assert "read_parquet" in uyari and "get_artifact" in uyari

    # Doğru okuyucu kullanılırsa uyarı YOK
    assert entrypoint._bicim_uyari(str(yol), "read_parquet") is None


def test_duz_metne_uyari_verilmiyor(tmp_path):
    yol = tmp_path / "rapor.csv"
    yol.write_text("a,b\n1,2\n")
    assert entrypoint._bicim_uyari(str(yol), "read_csv") is None
