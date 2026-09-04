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

    def list(self, workflow_id, node_id=None):
        return [{"name": ad, "artifact_id": f"art_{ad}"} for ad in self.icerik]

    def fetch_to_file(self, name, hedef):
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

    entrypoint._tembel_oku(str(hedef), istemci, {"rapor.csv"}, {})

    assert hedef.read_bytes() == b"id,tutar\n1,100\n"
    assert istemci.indirilenler == ["rapor.csv"]


def test_var_olan_dosya_icin_aga_cikilmaz(cikti):
    istemci = SahteIstemci({"rapor.csv": b"depodaki"})
    hedef = cikti / "rapor.csv"
    hedef.write_bytes(b"yereldeki")

    entrypoint._tembel_oku(str(hedef), istemci, {"rapor.csv"}, {})

    assert hedef.read_bytes() == b"yereldeki"
    assert istemci.indirilenler == []


def test_manifestte_olmayan_isim_icin_aga_cikilmaz(cikti):
    """Var olmayan dosya için boşuna round-trip yapılmamalı."""
    istemci = SahteIstemci({"baska.csv": b"x"})
    entrypoint._tembel_oku(str(cikti / "yok.csv"), istemci, {"baska.csv"}, {})
    assert istemci.indirilenler == []


def test_output_disindaki_yol_dokunulmaz(tmp_path, cikti):
    istemci = SahteIstemci({"gizli.csv": b"x"})
    entrypoint._tembel_oku(str(tmp_path / "gizli.csv"), istemci, {"gizli.csv"}, {})
    assert istemci.indirilenler == []


def test_alt_dizin_dokunulmaz(cikti):
    """Süpürme yalnızca üst düzeye bakıyor; tembel okuma da öyle olmalı."""
    istemci = SahteIstemci({"x.csv": b"a"})
    entrypoint._tembel_oku(str(cikti / "alt" / "x.csv"), istemci, {"x.csv"}, {})
    assert istemci.indirilenler == []


def test_dosya_nesnesi_ve_url_gecirilir(cikti):
    """`pd.read_csv(io.StringIO(...))` gibi çağrılar patlamamalı."""
    import io

    istemci = SahteIstemci({})
    entrypoint._tembel_oku(io.StringIO("a,b\n1,2\n"), istemci, set(), {})
    entrypoint._tembel_oku(None, istemci, set(), {})
    assert istemci.indirilenler == []


# -- süpürmeyle etkileşim: asıl regresyon --------------------------------


def test_indirilen_dosya_supurmede_geri_yuklenmez(cikti):
    """LLM sadece OKUDUYSA, indirdiğimiz dosya "üretilmiş" sayılmamalı.

    Bu kontrol olmadan, `pd.read_csv("/output/rapor.csv")` çağıran ve başka
    hiçbir şey yapmayan bir çalıştırma bile dosyayı geri yükler; dedup baytı
    tekilleştirse de her seferinde yeni bir artifact_id ve sahte bir "produced"
    olayı doğardı.
    """
    istemci = SahteIstemci({"rapor.csv": b"id,tutar\n1,100\n"})
    inenler: dict = {}
    entrypoint._tembel_oku(str(cikti / "rapor.csv"), istemci, {"rapor.csv"}, inenler)
    assert inenler  # indirme kaydedildi

    api = {"_put_file": lambda p, ct, ad, ttl=None: istemci.put_file(p, ct, ad, ttl)}
    entrypoint._ciktilari_supur(api, inenler)

    assert istemci.yuklenenler == []


def test_indirilen_dosya_degistirilirse_yuklenir(cikti):
    """Okuyup ÜZERİNE yazdıysa, o artık yeni bir sürüm — yüklenmeli."""
    istemci = SahteIstemci({"rapor.csv": b"eski"})
    inenler: dict = {}
    hedef = cikti / "rapor.csv"
    entrypoint._tembel_oku(str(hedef), istemci, {"rapor.csv"}, inenler)

    hedef.write_bytes(b"LLM bunu degistirdi")

    api = {"_put_file": lambda p, ct, ad, ttl=None: istemci.put_file(p, ct, ad, ttl)}
    entrypoint._ciktilari_supur(api, inenler)

    assert [ad for ad, _ in istemci.yuklenenler] == ["rapor.csv"]


def test_yeni_uretilen_dosya_yuklenir(cikti):
    """Emniyet ağı çalışıyor: put_artifact çağrılmasa da /output süpürülüyor."""
    istemci = SahteIstemci({})
    (cikti / "yeni.csv").write_bytes(b"a,b\n1,2\n")

    api = {"_put_file": lambda p, ct, ad, ttl=None: istemci.put_file(p, ct, ad, ttl)}
    entrypoint._ciktilari_supur(api, {})

    assert [ad for ad, _ in istemci.yuklenenler] == ["yeni.csv"]


# -- manifest --------------------------------------------------------------


def test_manifest_tek_istekle_isimleri_getirir(monkeypatch):
    monkeypatch.setattr(entrypoint, "WORKFLOW_ID", "wf_1")
    istemci = SahteIstemci({"a.csv": b"1", "b.parquet": b"2"})
    assert entrypoint._manifest(istemci) == {"a.csv", "b.parquet"}


def test_manifest_servis_erisilemezse_bos_doner(monkeypatch):
    """Pod açılışı asla engellenmemeli — tembel okuma sessizce devre dışı kalır."""
    monkeypatch.setattr(entrypoint, "WORKFLOW_ID", "wf_1")

    class Patlayan:
        def list(self, *a, **kw):
            raise RuntimeError("servis kapalı")

    assert entrypoint._manifest(Patlayan()) == set()
