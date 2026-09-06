"""Artifact manifestinin modele enjeksiyonu (2026-09-04).

Kopyalanan desen: Google ADK `LoadArtifactsTool` — *"lists available artifacts
in the model instructions"*. Öncesinde keşif yumuşak garantiydi: model
`list_artifacts()`'i çağırmayı unutursa depodaki veriyi yeniden üretiyordu.

Sınanan davranışlar:
  - isimler/tipler modele görünür oluyor, BAYTLAR görünmüyor
  - servis tanımsız/erişilemezse enjeksiyon sessizce kapanıyor (tur bozulmuyor)
  - liste uzunsa kırpılıyor (manifest her model çağrısında context'e giriyor)
"""

from __future__ import annotations

from grounded_assistant.agent.artifact_context import (
    ArtifactContextMiddleware,
    kunyeleri_getir,
    manifest_metni,
)

KUNYE = [
    {"name": "satislar.csv", "type": "system.Dataset", "size_bytes": 33, "metadata": {}},
    {"name": "metrik", "type": "system.Metrics", "size_bytes": 48, "metadata": {"r2": 0.91}},
]


def test_isim_tip_ve_metadata_gorunuyor():
    metin = manifest_metni(KUNYE)
    assert "satislar.csv" in metin
    assert "Dataset" in metin and "Metrics" in metin
    assert "r2" in metin


def test_baytlar_ASLA_gorunmuyor():
    """ADK'nın kuralı: isim ucuz, içerik pahalı. İçerik context'e girmemeli."""
    metin = manifest_metni(KUNYE)
    for yasak in ("content_b64", "storage_uri", "s3://", "content_hash"):
        assert yasak not in metin


def test_model_yeniden_uretmemeye_yonlendiriliyor():
    """Manifest, modele YOLU göstermeli — artık bir fonksiyon adını değil.

    2026-09-06: `get_artifact` kaldırıldı, keşif dosya sistemi üzerinden.
    Manifest sandbox'ın gördüğü `/output` görünümünün aynısını anlatmak
    zorunda; farklı bir dil kullanırsa model olmayan bir API arar.
    """
    metin = manifest_metni(KUNYE)
    assert "YENİDEN ÜRETME" in metin
    assert "/output/" in metin
    assert "read_parquet" in metin
    # kaldırılan yüzey geri sızmamalı
    for yok in ("get_artifact", "put_artifact", "list_artifacts", "cached("):
        assert yok not in metin


def test_bos_liste_mesaj_uretmiyor():
    """Hiç artifact yoksa context'i boş yere şişirme."""
    assert manifest_metni([]) is None


def test_ayni_isim_tek_satir(): 
    """Aynı ad birden çok çalıştırmada olabilir; okuma en yeniyi çözüyor,
    manifest de tek satır göstermeli — yoksa liste tekrarla dolardı."""
    cok = [{"name": "rapor.csv", "type": "system.Dataset", "size_bytes": i}
           for i in range(5)]
    assert manifest_metni(cok).count("/output/rapor.csv") == 1


def test_uzun_liste_kirpiliyor():
    cok = [{"name": f"a{i}", "type": "system.Artifact", "size_bytes": 1} for i in range(120)]
    metin = manifest_metni(cok)
    assert "ve 80 tane daha" in metin
    assert metin.count("\n  /output/") == 40


def test_servis_tanimsizsa_ag_istegi_yok(monkeypatch):
    monkeypatch.delenv("ARTIFACT_SERVICE_URL", raising=False)
    assert kunyeleri_getir("wf", "jeton") is None


def test_erisilemezse_sessizce_none(monkeypatch):
    """Manifest çekilemedi diye kullanıcının turu bozulmamalı."""
    monkeypatch.setenv("ARTIFACT_SERVICE_URL", "http://127.0.0.1:1")  # kapalı port
    assert kunyeleri_getir("wf", "jeton") is None


def test_middleware_workflowsuz_devre_disi():
    mw = ArtifactContextMiddleware(None, lambda: "jeton")
    assert mw.before_model(None, None) is None


def test_middleware_jeton_uretilemezse_devre_disi(monkeypatch):
    monkeypatch.setenv("ARTIFACT_SERVICE_URL", "http://ornek:8080")
    mw = ArtifactContextMiddleware("wf", lambda: None)
    assert mw.before_model(None, None) is None


def test_middleware_mesaji_sistem_mesaji_olarak_ekliyor(monkeypatch):
    from langchain_core.messages import SystemMessage

    monkeypatch.setenv("ARTIFACT_SERVICE_URL", "http://ornek:8080")
    monkeypatch.setattr(
        "grounded_assistant.agent.artifact_context.kunyeleri_getir",
        lambda wf, jeton: KUNYE,
    )
    mw = ArtifactContextMiddleware("wf", lambda: "jeton")
    sonuc = mw.before_model(None, None)
    assert isinstance(sonuc["messages"][0], SystemMessage)
    assert "satislar.csv" in sonuc["messages"][0].content


# -- iki grup: bu oturum vs başka çalıştırmalar (2026-09-06) ----------------


KARISIK = [
    {"name": "benim.parquet", "type": "system.Dataset", "size_bytes": 10,
     "workflow_id": "wf_ben"},
    {"name": "baskasinin.parquet", "type": "system.Dataset", "size_bytes": 20,
     "workflow_id": "wf_baska"},
]


def test_kendi_ciktisi_ayri_bolumde():
    """ASIL REGRESYON (2026-09-06, canlı kullanımda bulundu).

    Keşif kapsamı tenant'a genişleyince manifest düz bir liste oldu. Ajan
    1. turda `ticket_analiz_raporu.pdf` üretti (HR = 60,0); 2. turda
    "az önce ürettiğin analizde HR kaçtı?" sorulunca manifestte gördüğü
    BAŞKA bir workflow'un `departman.ozet.parquet`'ini okudu ve "7,46" dedi.

    Cevap sessizce yanlıştı — hiçbir yerde hata yoktu. Model'in "benim işim"
    ile "başkasınınki"ni ayırt edebilmesi için liste ikiye bölündü.
    """
    metin = manifest_metni(KARISIK, "wf_ben")

    assert "BU OTURUMDA ÜRETİLENLER" in metin
    assert "BAŞKA ÇALIŞTIRMALARDAN" in metin
    # kendi çıktısı ilk bölümde, diğeri ikincide
    ilk, ikinci = metin.split("BAŞKA ÇALIŞTIRMALARDAN")
    assert "benim.parquet" in ilk and "baskasinin.parquet" not in ilk
    assert "baskasinin.parquet" in ikinci


def test_baskasinin_ciktisini_kendi_gibi_sunma_uyarisi():
    metin = manifest_metni(KARISIK, "wf_ben")
    assert "kendi çıktın gibi sunma" in metin


def test_workflow_verilmezse_hepsi_diger_grupta():
    """Kapsam bilinmiyorsa hiçbir şey "benim" sayılmamalı — güvenli taraf."""
    metin = manifest_metni(KARISIK, None)
    assert "BU OTURUMDA ÜRETİLENLER" not in metin
    assert "BAŞKA ÇALIŞTIRMALARDAN" in metin


def test_yalnizca_kendi_ciktisi_varsa_ikinci_bolum_yok():
    metin = manifest_metni([KARISIK[0]], "wf_ben")
    assert "BU OTURUMDA ÜRETİLENLER" in metin
    assert "BAŞKA ÇALIŞTIRMALARDAN" not in metin
