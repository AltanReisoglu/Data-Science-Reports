"""Girdi yerleştirme — KFP/Argo deseni (2026-09-07).

## Yerini aldığı şey

Bundan önce `/output` **sahte bir görünümdü**: `os.listdir`, `os.path.exists`,
`glob`, pandas okuyucuları ve `open` yamalanıyor, bayt ancak
`pd.read_parquet(...)` çağrısının ORTASINDA iniyordu.

Piyasada bunun emsali yoktu — tek gerçek icadımızdı. Argo girdiyi `init`
container'da, KFP `driver`+`launcher` ile indiriyor; ikisinde de dosya kod
BAŞLAMADAN yerinde oluyor. Artık bizde de öyle: `sidecar.yerlestir()`.

## Sınananlar

  - açılışta bu çalıştırmanın çıktıları `/output`'a İNİYOR
  - BAŞKA çalıştırmanınkiler inmiyor (`/output` = bu çalıştırma, KFP'de
    `pipeline_root/<run-id>/`)
  - aynı ad birden çok kayıttaysa EN YENİ kazanıyor
  - yerleştirilen dosya süpürmede GERİ YÜKLENMİYOR (defter sidecar'da)
  - depo erişilemezse çalıştırma yine sürüyor
  - `load_artifact` başka çalıştırmayı açıkça getiriyor, yol geçişini reddediyor
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
os.environ.setdefault("ARTIFACT_SERVICE_ENDPOINT", "http://yok")

import entrypoint  # noqa: E402
import sidecar  # noqa: E402

WF = "wf-bu"


class SahteUst:
    """Sidecar'ın YUKARI akıştaki Artifact Service istemcisi."""

    def __init__(self, kayitlar: list[dict], icerik: dict[str, bytes]):
        self.kayitlar = kayitlar
        self.icerik = icerik
        self.indirilenler: list[tuple[str, str | None]] = []
        self.yuklenenler: list[tuple[str, list[str]]] = []

    def list_all(self):
        return self.kayitlar

    def fetch_to_file(self, name, hedef, workflow_id=None):
        self.indirilenler.append((name, workflow_id))
        anahtar = f"{workflow_id}/{name}"
        ham = self.icerik.get(anahtar, self.icerik.get(name))
        if ham is None:
            return None
        Path(hedef).write_bytes(ham)
        tip = "application/x-tar" if name.endswith(".tar") else "text/csv"
        return {"artifact_id": f"art_{name}", "name": name,
                "content_type": tip, "size_bytes": len(ham)}

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


def kayit(ad: str, wf: str = WF) -> dict:
    return {"name": ad, "workflow_id": wf, "artifact_id": f"art_{ad}"}


# ── yerleştirme ───────────────────────────────────────────────────────────


def test_kod_baslamadan_once_dosya_yerinde(ortam, monkeypatch):
    """ASIL KURAL: indirme kod başlamadan biter (Argo `init`, KFP launcher)."""
    ust = SahteUst([kayit("ham.csv")], {"ham.csv": b"a,b\n1,2\n"})
    monkeypatch.setattr(sidecar, "istemci", ust)

    assert sidecar.yerlestir() == 1
    assert (ortam / "ham.csv").read_bytes() == b"a,b\n1,2\n"


def test_baska_calistirmanin_ciktisi_output_a_INMEZ(ortam, monkeypatch):
    """`/output` = BU çalıştırma. KFP'de `pipeline_root/<run-id>/` neyse o.

    2026-09-06'da canlıda çıkan arıza buydu: düz bir isim alanında ajan başka
    bir run'ın aynı adlı dosyasını okuyup kendi sonucu sandı, cevap SESSİZCE
    yanlış oldu.
    """
    ust = SahteUst(
        [kayit("benim.csv"), kayit("baskasinin.csv", "wf-baska")],
        {"benim.csv": b"1", "baskasinin.csv": b"2"},
    )
    monkeypatch.setattr(sidecar, "istemci", ust)

    sidecar.yerlestir()

    assert (ortam / "benim.csv").exists()
    assert not (ortam / "baskasinin.csv").exists()
    assert [ad for ad, _ in ust.indirilenler] == ["benim.csv"]


def test_ayni_ad_birden_cok_kayitta_ise_en_yeni_kazanir(ortam, monkeypatch):
    """Servis yeniden-eskiye sıralı döndürüyor; ilk görülen kazanır."""
    ust = SahteUst([kayit("x.csv"), kayit("x.csv")], {"x.csv": b"yeni"})
    monkeypatch.setattr(sidecar, "istemci", ust)

    assert sidecar.yerlestir() == 1
    assert len(ust.indirilenler) == 1


def test_yerlestirilen_dosya_supurmede_geri_yuklenmez(ortam, monkeypatch):
    """Yoksa her çalıştırma aynı içeriği yeniden yazar, depo şişer.

    Defter sidecar'da: `_sunulan_ozet`'e yerleştirmede de işleniyor, yani
    LLM'in kodu bu kararı etkileyemiyor.
    """
    ust = SahteUst([kayit("ham.csv")], {"ham.csv": b"a,b\n1,2\n"})
    monkeypatch.setattr(sidecar, "istemci", ust)

    sidecar.yerlestir()
    sidecar.supur()

    assert ust.yuklenenler == []


def test_llm_degistirirse_yeniden_yuklenir(ortam, monkeypatch):
    """İçerik değişmişse artık "biz verdik" değil — yüklenir."""
    ust = SahteUst([kayit("ham.csv")], {"ham.csv": b"a,b\n1,2\n"})
    monkeypatch.setattr(sidecar, "istemci", ust)

    sidecar.yerlestir()
    (ortam / "ham.csv").write_bytes(b"a,b\n9,9\n")   # LLM üstüne yazdı
    (ortam / "turev.parquet").write_bytes(b"x")      # ve yeni bir şey üretti
    sidecar.supur()

    assert set(dict(ust.yuklenenler)) == {"ham.csv", "turev.parquet"}


# ── soy: yerleştirilen ≠ okunan (2026-09-07) ──────────────────────────────


def test_OKUNAN_ebeveyn_olur(ortam, monkeypatch):
    ust = SahteUst([kayit("ham.csv")], {"ham.csv": b"a,b\n1,2\n"})
    monkeypatch.setattr(sidecar, "istemci", ust)

    sidecar.yerlestir()
    (ortam / "ham.csv").read_bytes()                 # OKUDU
    (ortam / "turev.parquet").write_bytes(b"x")
    sidecar.supur()

    assert dict(ust.yuklenenler)["turev.parquet"] == ["art_ham.csv"]


def test_BEYAN_yerlestirmeyi_daraltir(ortam, monkeypatch):
    """ASIL DESEN (2026-09-07): Argo `inputs.artifacts`, KFP bileşen girdisi.

    Beyansız çalışınca bu çalıştırmanın HER çıktısı yerleşiyor ve hepsi soy
    ağacında ebeveyn oluyordu — üç dosya yerleşip biri okununca türev üç
    ebeveyn alıyordu. Kısa süre atime ile "hangisi okundu" ölçmeyi denedik;
    sahada emsali olmayan bir icattı. MLMD'nin cevabı beyan: olay tipinin adı
    zaten `Event.DECLARED_INPUT`.
    """
    ust = SahteUst([kayit("a.txt"), kayit("b.txt"), kayit("c.txt")],
                   {"a.txt": b"A", "b.txt": b"B", "c.txt": b"C"})
    monkeypatch.setattr(sidecar, "istemci", ust)
    monkeypatch.setattr(sidecar, "INPUTS_HAM", "a.txt")

    assert sidecar.yerlestir() == 1
    assert (ortam / "a.txt").exists()
    assert not (ortam / "b.txt").exists()
    assert not (ortam / "c.txt").exists()

    (ortam / "turev.txt").write_bytes(b"A!")
    sidecar.supur()
    assert dict(ust.yuklenenler)["turev.txt"] == ["art_a.txt"]


def test_beyan_yoksa_HEPSI_yerlesir(ortam, monkeypatch):
    """Uyumluluk yolu: ajan beyan etmezse çalıştırma kırılmamalı."""
    ust = SahteUst([kayit("a.txt"), kayit("b.txt")], {"a.txt": b"A", "b.txt": b"B"})
    monkeypatch.setattr(sidecar, "istemci", ust)
    monkeypatch.setattr(sidecar, "INPUTS_HAM", "*")

    assert sidecar.yerlestir() == 2
    (ortam / "turev.txt").write_bytes(b"x")
    sidecar.supur()
    assert dict(ust.yuklenenler)["turev.txt"] == ["art_a.txt", "art_b.txt"]


def test_bos_beyan_hicbir_sey_yerlestirmez(ortam, monkeypatch):
    """`inputs=[]` — "girdiye ihtiyacım yok". Ağa da çıkılmıyor."""
    ust = SahteUst([kayit("a.txt")], {"a.txt": b"A"})
    monkeypatch.setattr(sidecar, "istemci", ust)
    monkeypatch.setattr(sidecar, "INPUTS_HAM", "")

    assert sidecar.yerlestir() == 0
    assert not (ortam / "a.txt").exists()


def test_beyanda_dizin_adi_tar_ile_eslesir(ortam, tmp_path, monkeypatch):
    """Ajan `/output/model.v1/` görüyor, depoda ad `model.v1.tar`. İkisi de
    yazılabilmeli — yoksa beyan sessizce karşılanmazdı."""
    kaynak = tmp_path / "k"; kaynak.mkdir(); (kaynak / "w.json").write_text("1")
    paket = tmp_path / "p.tar"; sidecar._dizini_paketle(str(kaynak), str(paket))

    ust = SahteUst([{"name": "model.v1.tar", "workflow_id": WF, "artifact_id": "art_m"}],
                   {"model.v1.tar": paket.read_bytes()})
    monkeypatch.setattr(sidecar, "istemci", ust)
    monkeypatch.setattr(sidecar, "INPUTS_HAM", "model.v1")   # tar'sız hâli

    assert sidecar.yerlestir() == 1
    assert (ortam / "model.v1" / "w.json").exists()


def test_load_artifact_ile_istenen_KOSULSUZ_ebeveyn(ortam, monkeypatch):
    """`/fetch` yolunda istenmiş olmak okunmuş olmanın kanıtı — atime'a
    bakılmıyor, çünkü dosya `/artifacts` altında ve süpürülmüyor."""
    ust = SahteUst([], {})
    monkeypatch.setattr(sidecar, "istemci", ust)

    sidecar._kaydet("baskasi.csv", "ozet", "art_baskasi", istendi=True)
    (ortam / "turev.txt").write_bytes(b"x")
    sidecar.supur()

    assert dict(ust.yuklenenler)["turev.txt"] == ["art_baskasi"]


def test_depo_erisilemezse_calistirma_surer(ortam, monkeypatch):
    """Artifact deposu bir kolaylık; olmayınca soru cevapsız kalmamalı."""

    class Patlak:
        def list_all(self):
            raise RuntimeError("bağlanamadı")

    monkeypatch.setattr(sidecar, "istemci", Patlak())
    assert sidecar.yerlestir() == 0


def test_workflow_kimligi_yoksa_hicbir_sey_yerlestirilmez(ortam, monkeypatch):
    ust = SahteUst([kayit("x.csv")], {"x.csv": b"1"})
    monkeypatch.setattr(sidecar, "istemci", ust)
    monkeypatch.setattr(sidecar, "WORKFLOW_ID", "")

    assert sidecar.yerlestir() == 0
    assert ust.indirilenler == []


def test_biri_patlarsa_digerleri_yerlesir(ortam, monkeypatch):
    ust = SahteUst([kayit("iyi.csv"), kayit("yok.csv")], {"iyi.csv": b"1"})
    monkeypatch.setattr(sidecar, "istemci", ust)

    assert sidecar.yerlestir() == 1
    assert (ortam / "iyi.csv").exists()
    assert not (ortam / "yok.csv").exists()


# ── load_artifact: BAŞKA çalıştırma, açık çağrı ───────────────────────────


@pytest.fixture
def artifacts_koku(tmp_path, monkeypatch):
    d = tmp_path / "artifacts"
    d.mkdir()
    monkeypatch.setattr(entrypoint, "ARTIFACTS_DIR", str(d))
    return d


def test_load_artifact_baska_calistirmayi_getirir(artifacts_koku):
    ust = SahteUst([], {"wf-baska/rapor.csv": b"veri"})
    yol = entrypoint._load_artifact_uret(ust)("wf-baska", "rapor.csv")

    assert Path(yol).read_bytes() == b"veri"
    assert Path(yol) == artifacts_koku / "wf-baska" / "rapor.csv"
    assert ust.indirilenler == [("rapor.csv", "wf-baska")]


def test_load_artifact_olmayan_icin_acik_hata(artifacts_koku):
    ust = SahteUst([], {})
    with pytest.raises(FileNotFoundError):
        entrypoint._load_artifact_uret(ust)("wf-baska", "yok.csv")


@pytest.mark.parametrize("kotu_wf", ["../../etc", "wf/../..", "/mutlak", ""])
def test_load_artifact_yol_gecisli_kimligi_reddeder(artifacts_koku, kotu_wf):
    """Kimlik doğrudan bir dizin adına dönüşüyor — süzülmezse `/artifacts`
    dışına yazılabilirdi."""
    ust = SahteUst([], {})
    with pytest.raises(ValueError):
        entrypoint._load_artifact_uret(ust)(kotu_wf, "x.csv")
    assert ust.indirilenler == []


def test_load_artifact_yol_gecisli_adi_temizler(artifacts_koku):
    """Ad da süzülüyor: `basename` + servisin kabul ettiği biçim."""
    ust = SahteUst([], {"wf-baska/passwd": b"kok"})
    yol = entrypoint._load_artifact_uret(ust)("wf-baska", "../../etc/passwd")

    assert Path(yol) == artifacts_koku / "wf-baska" / "passwd"


def test_servis_kapaliysa_acik_hata(artifacts_koku):
    with pytest.raises(RuntimeError):
        entrypoint._load_artifact_uret(None)("wf-baska", "x.csv")


# ── yüzey: geri sızmasın ──────────────────────────────────────────────────


def test_LLM_ARTIFACT_YAZMA_API_SI_YOK():
    """2026-09-06: `put_artifact`/`get_artifact`/`cached` KALDIRILDI.

    Onlar bizim icadımızdı — piyasada emsali yok, ve o haftanın ciddi
    hatalarının çoğu tam o yüzeyde çıktı. Bu test kasıtlı olarak NEGATİF:
    yüzeyin geri sızmadığını garanti ediyor.
    """
    for gitmis in ("_artifact_api", "_launcher_api", "_ciktilari_supur",
                   "_dizini_supur", "_dizini_paketle", "SCOPE_TOKEN"):
        assert not hasattr(entrypoint, gitmis), f"{gitmis} geri sızdı"


def test_KESIF_YAMALARI_YOK():
    """2026-09-07: `os.listdir`/`glob`/pandas/`open` yamaları KALDIRILDI.

    Tek gerçek icadımızdı; yerine Argo/KFP'nin "kod başlamadan yerleştir"
    deseni geçti. Yama geri gelirse `/output` yine yalan söylemeye başlar.
    """
    for gitmis in ("_yamala_kesif", "_tembel_oku", "_tembel_okumayi_kur",
                   "_tembel_dizin_ac", "_okuyucu_sarmala", "_bicim_uyari",
                   "Depo", "_manifest", "_yolu_coz",
                   "_GERCEK_LISTDIR", "_GERCEK_EXISTS"):
        assert not hasattr(entrypoint, gitmis), f"{gitmis} geri sızdı"


def test_os_listdir_yamali_degil():
    """En doğrudan sınama: modül import edilince stdlib bozulmuş olmamalı."""
    assert os.listdir.__module__ in ("posix", "nt", "os")
