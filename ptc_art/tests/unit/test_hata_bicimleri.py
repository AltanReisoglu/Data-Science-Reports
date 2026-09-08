"""`scripts/hata_bicim_kiyas.py` — karşılaştırmanın bulgularını sabitler.

Kıyas tablosu sunuma girecek. Bir bulgu tabloda yazıyorsa burada da bir test
olarak durmalı: biçimlendiricilerden biri değişirse tablo sessizce yanlış
olmasın.

Sabitlenen bulgular:
  1. Baştan kırpma (SWE-agent varsayılan şablonu) BÜYÜK ÇIKTIDA hata tipini
     kaybediyor; `bash_only`'nin baş+son yarısı kaybetmiyor.
  2. Kırpmayan biçimler (AutoGen, OpenHands, Anthropic API) aynı arızada
     bizimkinin 10 katından fazla bayt taşıyor.
  3. Bizim biçim hata anına kadarki stdout'u koruyor, tipi ve satırı veriyor,
     talimat içeriyor ve KULLANICI DIŞI traceback karesi SIZDIRMIYOR.
  4. Tam traceback basan biçimler harness karesi sızdırıyor.
  5. Tekrar tespiti yalnızca OpenHands'te var (3. tekrarda dürtüyor).
  6. Eski hâlimiz tipi de stdout'u da kaybediyordu.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[2]


def _yukle():
    yol = KOK / "scripts" / "hata_bicim_kiyas.py"
    spec = importlib.util.spec_from_file_location("_hata_kiyas_test", yol)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass modülü sys.modules'te arıyor; exec'ten ÖNCE kaydet
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


K = _yukle()


@pytest.fixture(scope="module")
def arizalar() -> dict:
    return {a.ad: a for a in K.senaryolar()}


def _bicim(sinif) -> object:
    b = sinif()
    b.sifirla()
    return b


# ── 1. Kırpma YÖNÜ — sunumdaki en keskin bulgu ───────────────────────
def test_bastan_kirpma_hatanin_kendisini_kaybediyor(arizalar):
    """Hata en SONDA; baştan kesince modele hiç ulaşmıyor."""
    a = arizalar["buyuk"]
    assert a.tip == "ValueError"
    metin = _bicim(K.SWEAgentVarsayilan)(a)
    assert "ValueError" not in metin, "baştan kırpma tipi korumamalıydı"
    assert "clipped" in metin, "kırpma yine de BİLDİRİLMELİ"


def test_ortadan_kirpma_hatayi_koruyor(arizalar):
    a = arizalar["buyuk"]
    for sinif in (K.SWEAgentBashOnly, K.Codex, K.Smolagents, K.BizimYeni):
        metin = _bicim(sinif)(a)
        assert "ValueError" in metin, f"{sinif.ad} hatayı kaybetti"


# ── 2. Kırpmamanın bedeli ────────────────────────────────────────────
@pytest.mark.parametrize("sinif", [K.AutoGen, K.OpenHands, K.AnthropicAPI])
def test_kirpmayanlar_on_kat_pahali(arizalar, sinif):
    a = arizalar["buyuk"]
    bizim = len(_bicim(K.BizimYeni)(a).encode())
    onlar = len(_bicim(sinif)(a).encode())
    assert onlar > bizim * 10, f"{sinif.ad}: {onlar} vs {bizim}"


def test_kirpanlar_esigin_yakininda_kaliyor(arizalar):
    a = arizalar["buyuk"]
    for sinif, tavan in ((K.Codex, 12_000), (K.ClaudeCode, 14_000),
                         (K.BizimYeni, 24_000)):
        assert len(_bicim(sinif)(a)) < tavan, sinif.ad


# ── 3. Bizim biçimin taşıdıkları ─────────────────────────────────────
def test_bizim_bicim_stdout_tip_satir_talimat(arizalar):
    metin = _bicim(K.BizimYeni)(arizalar["keyerror"])
    assert "veri yuklendi" in metin          # hata anına kadarki çıktı
    assert "KeyError" in metin               # tip
    assert re.search(r"line \d+", metin)     # satır
    assert 'd["yok"]' in metin               # kaynak satırı
    assert "TEKRAR çalıştır" in metin        # talimat
    assert "UYDURMA" in metin


def test_bizim_bicim_ic_yigin_sizdirmiyor(arizalar):
    for a in arizalar.values():
        metin = _bicim(K.BizimYeni)(a)
        yabanci = [k for k in K.KARE_DESENI.findall(metin) if k != a.yol]
        assert not yabanci, f"{a.ad}: sızan kare {yabanci}"


def test_sonuc_bildirilmeyince_de_talimat_gidiyor(arizalar):
    """Exception yok ama model yine de ne yapacağını öğrenmeli."""
    metin = _bicim(K.BizimYeni)(arizalar["sonucsuz"])
    assert "set_result" in metin
    assert "hesap bitti" in metin            # stdout korunuyor
    assert "TEKRAR çalıştır" in metin


# ── 4. Tam traceback sızdırıyor ──────────────────────────────────────
@pytest.mark.parametrize("sinif", [K.ClaudeCode, K.Codex, K.AutoGen, K.OpenHands])
def test_tam_traceback_harness_karesi_sizdiriyor(arizalar, sinif):
    a = arizalar["keyerror"]
    metin = _bicim(sinif)(a)
    yabanci = [k for k in K.KARE_DESENI.findall(metin) if k != a.yol]
    assert yabanci, f"{sinif.ad} sızdırmadı — beklenen davranış değişmiş"


# ── 5. Tekrar tespiti ────────────────────────────────────────────────
def test_yalnizca_openhands_tekrari_farkediyor(arizalar):
    a = arizalar["keyerror"]
    oh = _bicim(K.OpenHands)
    ciktilar = [oh(a) for _ in range(3)]
    assert "NUDGE" not in ciktilar[0]
    assert "NUDGE" not in ciktilar[1]
    assert "NUDGE" in ciktilar[2], "3. tekrarda dürtmeliydi"

    for sinif in (K.ClaudeCode, K.Codex, K.Smolagents, K.BizimYeni,
                  K.SWEAgentBashOnly, K.AnthropicAPI, K.AutoGen):
        b = _bicim(sinif)
        assert b(a) == b(a), f"{sinif.ad} beklenmedik şekilde durum tutuyor"


# ── 6. Eski hâlimiz ──────────────────────────────────────────────────
def test_eski_bicim_tipi_ve_stdoutu_kaybediyor(arizalar):
    metin = _bicim(K.BizimEski)(arizalar["keyerror"])
    assert "KeyError" not in metin
    assert "veri yuklendi" not in metin
    assert not re.search(r"line \d+", metin)


# ── 7. smolagents'ın takası: satır no yerine KAYNAK METNİ ────────────
def test_smolagents_satir_numarasi_yerine_kaynak_veriyor(arizalar):
    metin = _bicim(K.Smolagents)(arizalar["keyerror"])
    assert 'd["yok"]' in metin
    assert "Traceback (most recent call last)" not in metin
    assert "veri yuklendi" in metin


# ── 8. Tablonun kendisi üretilebiliyor ───────────────────────────────
def test_kiyas_tablosu_uretiliyor():
    puanlar, tekrar = K.kosttur()
    assert len(puanlar) == len(K.BICIMLER) * 5
    metin = K.rapor(puanlar, tekrar)
    assert "Sinyal tablosu" in metin
    for b in K.BICIMLER:
        assert b.ad in metin
    assert tekrar["OpenHands"] is True
    assert tekrar["Bizim (şimdi)"] is False
