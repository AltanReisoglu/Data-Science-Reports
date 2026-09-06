"""Artifact Service davranış testleri.

Nesne deposu yerine bellek içi bir sahte kullanılıyor — MinIO gerekmiyor.
Odak, dokümanlarda karar verdiğimiz kontroller: kapsam, isim, boyut, format,
dedup ve silme sırası.
"""

from __future__ import annotations

import pickle

import pandas as pd
import pytest

from grounded_assistant.artifacts.metadata import open_sqlite
from grounded_assistant.artifacts.serialize import PARQUET, UnsafeArtifact
from grounded_assistant.artifacts.service import (
    ArtifactService,
    ArtifactTooLarge,
    InvalidArtifactName,
    ScopeViolation,
)
from grounded_assistant.artifacts.store import BucketConfig, ObjectStore

WF = "wf_42"
SAHIP = "altan"


class SahteMinio:
    """minio.Minio'nun kullandığımız yüzeyi kadarı."""

    def __init__(self):
        self.nesneler: dict[str, bytes] = {}
        self.put_sayisi = 0

    def put_object(self, bucket, key, data, length, content_type):
        self.nesneler[key] = data.read()
        self.put_sayisi += 1

    def get_object(self, bucket, key):
        veri = self.nesneler[key]

        class _Yanit:
            def read(self_inner):
                return veri

            def close(self_inner): ...
            def release_conn(self_inner): ...

        return _Yanit()

    def remove_object(self, bucket, key):
        self.nesneler.pop(key, None)


@pytest.fixture
def sahte():
    return SahteMinio()


@pytest.fixture
def service(sahte):
    cfg = BucketConfig(
        name="artifacts", host="localhost", port=9000,
        access_key="a", secret_key="b", secure=False,
    )
    return ArtifactService(
        metadata=open_sqlite(":memory:"),
        objects=ObjectStore(cfg, client=sahte),
    )


def yaz(service, value, name="extract.tickets", **kw):
    return service.create(
        value, name=name, workflow_id=kw.pop("workflow_id", WF),
        run_id=kw.pop("run_id", "run_1"), owner=SAHIP, **kw
    )


def test_dataframe_gidis_donus_tipleri_korur(service):
    df = pd.DataFrame({"id": [1, 2, 3], "departman": ["BT", "IK", "BT"]})
    meta = yaz(service, df)

    assert meta.content_type == PARQUET
    geri = service.get(meta.artifact_id, owner=SAHIP)
    pd.testing.assert_frame_equal(geri, df)


def test_isimle_en_yeni_surum(service):
    yaz(service, pd.DataFrame({"a": [1]}))
    yaz(service, pd.DataFrame({"a": [1, 2]}))
    geri = service.get_by_name("extract.tickets", owner=SAHIP)
    assert len(geri) == 2, "en son yazılan gelmeli"


def test_baska_TENANT_okuyamaz(service):
    """Yetki sınırı 2026-09-06'da workflow'dan TENANT'a taşındı.

    KFP'de de izolasyon namespace düzeyinde: bütün run'lar aynı
    `pipeline_root` altına yazar, biri diğerinin çıktısını okuyabilir.
    Çalıştırma başına mühürlemek bizim eklediğimiz bir şeydi; ürün
    "başka workflow'un artifact'ini kullanabilsin" istediği için kalktı.
    """
    meta = yaz(service, {"x": 1})
    with pytest.raises(ScopeViolation):
        service.get(meta.artifact_id, owner="baska-tenant")


def test_ayni_tenantta_BASKA_workflow_okuyabilir(service):
    """Yeni davranışın kendisi — eskiden ScopeViolation'dı."""
    meta = yaz(service, {"x": 1}, workflow_id="wf_ureten")
    assert service.get(meta.artifact_id, owner=SAHIP) == {"x": 1}
    # ve isimle de bulunabilmeli (başka bir workflow'dan çağrılıyormuş gibi)
    assert service.get_by_name("extract.tickets", owner=SAHIP,
                               workflow_id="wf_okuyan") == {"x": 1}


def test_kendi_workflowu_ayni_ismi_golgeliyor(service):
    """Aynı ad iki çalıştırmada varsa ÇAĞIRANINKİ kazanır.

    Kendi çıktının başkasının aynı adlı çıktısıyla gölgelenmesi, en az
    beklenen davranış olurdu.
    """
    yaz(service, {"kim": "baskasi"}, name="rapor.json", workflow_id="wf_a")
    yaz(service, {"kim": "benim"}, name="rapor.json", workflow_id="wf_b")
    # wf_a'dan bakınca kendi (eski) sürümü gelmeli, wf_b'nin yenisi değil
    assert service.get_by_name("rapor.json", owner=SAHIP,
                               workflow_id="wf_a") == {"kim": "baskasi"}
    # workflow tercihi verilmezse en yeni gelir
    assert service.get_by_name("rapor.json", owner=SAHIP) == {"kim": "benim"}


def test_yetkisiz_erisim_varligi_sizdirmaz(service):
    """Var-ama-yetkisiz ile hiç-yok aynı hatayı vermeli."""
    meta = yaz(service, {"x": 1})
    with pytest.raises(ScopeViolation) as var_ama_yetkisiz:
        service.get(meta.artifact_id, owner="baska-tenant")
    with pytest.raises(ScopeViolation) as hic_yok:
        service.get("art_hicyok", owner="baska-tenant")
    assert type(var_ama_yetkisiz.value) is type(hic_yok.value)


@pytest.mark.parametrize(
    "kotu_isim",
    ["/etc/shadow", "../../etc/passwd", "a/b", "", "  ", ".gizli"],
)
def test_yol_gecisi_reddedilir(service, kotu_isim):
    with pytest.raises(InvalidArtifactName):
        yaz(service, {"x": 1}, name=kotu_isim)


def test_boyut_siniri(service):
    service.size_limit = 128
    with pytest.raises(ArtifactTooLarge):
        yaz(service, {"dolgu": "x" * 500})


def test_pickle_baytlari_reddedilir(service):
    """Depoya başka yoldan girmiş pickle da okunurken yakalanmalı."""
    zehir = pickle.dumps({"kotu": "yuk"})
    with pytest.raises(UnsafeArtifact):
        yaz(service, zehir)


def test_ayni_icerik_iki_kez_yuklenmez(service, sahte):
    df = pd.DataFrame({"a": [1, 2, 3]})
    bir = yaz(service, df, name="ilk")
    iki = yaz(service, df, name="ikinci")

    assert sahte.put_sayisi == 1, "dedup: bayt bir kez yüklenmeli"
    assert bir.storage_uri == iki.storage_uri
    assert bir.artifact_id != iki.artifact_id, "içerik aynı, kimlik ayrı"
    assert bir.content_hash == iki.content_hash


def test_silme_paylasilan_bayti_korur(service, sahte):
    df = pd.DataFrame({"a": [1]})
    bir = yaz(service, df, name="ilk")
    iki = yaz(service, df, name="ikinci")

    service.delete(bir.artifact_id, owner=SAHIP)
    assert sahte.nesneler, "diğer kayıt hâlâ gösteriyor — bayt silinmemeli"
    pd.testing.assert_frame_equal(service.get(iki.artifact_id, owner=SAHIP), df)

    service.delete(iki.artifact_id, owner=SAHIP)
    assert not sahte.nesneler, "son referans da gitti — bayt silinmeli"


def test_tenant_listelemesi_workflowlari_kapsiyor(service):
    """Manifestin kaynağı: kim üretmiş olursa olsun hepsi görünüyor."""
    yaz(service, {"a": 1}, name="extract.tickets", node_id="extract",
        workflow_id="wf_a")
    yaz(service, {"b": 2}, name="transform.ozet", node_id="transform",
        workflow_id="wf_b")

    hepsi = service.list(owner=SAHIP)
    assert {m.name for m in hepsi} == {"extract.tickets", "transform.ozet"}
    # künye hangi workflow'dan geldiğini taşımaya devam ediyor
    assert {m.workflow_id for m in hepsi} == {"wf_a", "wf_b"}


def test_lineage_parents_korunur(service):
    kaynak = yaz(service, {"a": 1}, name="extract.tickets")
    turev = yaz(service, {"b": 2}, name="transform.ozet", parents=(kaynak.artifact_id,))
    assert service.metadata_of(turev.artifact_id, owner=SAHIP).parents == (
        kaynak.artifact_id,
    )


# -- depo yapılandırması: iki OpenShift sözleşmesi (2026-09-04) -------------


def test_obc_sozlesmesi_okunur():
    """ODF varsa ObjectBucketClaim bu adları üretir."""
    c = BucketConfig.from_env({
        "BUCKET_NAME": "ptc", "BUCKET_HOST": "minio.default.svc", "BUCKET_PORT": "9000",
        "AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "b",
    })
    assert (c.name, c.endpoint, c.secure) == ("ptc", "minio.default.svc:9000", False)


def test_openshift_ai_baglanti_sozlesmesi_okunur():
    """ODF YOKKEN ekibin elinde olan şey bu — OpenShift AI connection Secret'ı."""
    c = BucketConfig.from_env({
        "AWS_S3_ENDPOINT": "https://s3.kurum.local:9000", "AWS_S3_BUCKET": "ptc",
        "AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "b",
        "AWS_DEFAULT_REGION": "tr-1",
    })
    assert (c.name, c.endpoint, c.secure, c.region) == ("ptc", "s3.kurum.local:9000", True, "tr-1")


def test_semasiz_endpoint_porttan_tls_cikarir():
    c = BucketConfig.from_env({
        "AWS_S3_ENDPOINT": "s3.amazonaws.com", "AWS_S3_BUCKET": "p",
        "AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "b",
    })
    assert c.endpoint == "s3.amazonaws.com:443" and c.secure is True


def test_http_semasi_tlssiz_kabul_edilir():
    c = BucketConfig.from_env({
        "AWS_S3_ENDPOINT": "http://minio:9000", "AWS_S3_BUCKET": "p",
        "AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "b",
    })
    assert c.secure is False and c.endpoint == "minio:9000"


def test_iki_sozlesme_birden_varsa_obc_kazanir():
    c = BucketConfig.from_env({
        "BUCKET_NAME": "obc", "BUCKET_HOST": "obc.svc", "BUCKET_PORT": "9000",
        "AWS_S3_ENDPOINT": "https://baska:443", "AWS_S3_BUCKET": "baglanti",
        "AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "b",
    })
    assert c.name == "obc"


def test_hicbiri_yoksa_ikisini_de_anlatan_hata():
    with pytest.raises(KeyError, match="ObjectBucketClaim"):
        BucketConfig.from_env({"AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "b"})


def test_bucket_adi_eksikse_acik_hata():
    with pytest.raises(KeyError, match="AWS_S3_BUCKET"):
        BucketConfig.from_env({
            "AWS_S3_ENDPOINT": "https://s3:443",
            "AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "b",
        })


# -- iki yazma yolu AYNI tipi vermeli (2026-09-04'te bulunan tutarsızlık) ----


def test_csv_her_iki_yoldan_da_dataset():
    """`df.to_csv("/output/x.csv")` (süpürme) ile `put_artifact(df)` (açık API)
    AYNI veriyi yazıyor — tip de aynı olmalı.

    Bulunan kusur: `.csv` `text/plain`e eşleniyordu, süpürme yolundan gelen CSV
    `system.Artifact` oluyordu; açık API'den gelen `system.Dataset`. Aynı veri,
    iki farklı tip, sırf hangi yoldan geçtiğine göre.
    """
    from grounded_assistant.artifacts.serialize import (
        PARQUET,
        content_type_for_filename,
        tip_cikar,
    )

    supurme = tip_cikar(content_type_for_filename("satislar.csv"))
    acik_api = tip_cikar(PARQUET)          # put_artifact(df) Parquet'e çevirir
    assert supurme == acik_api == "system.Dataset"


def test_metin_aileleri_metin_olarak_cozulur():
    """`text/csv` ve `text/markdown` de metindir — yalnızca `text/plain` değil.

    Aile kontrolü olmasaydı `.csv`yi `text/csv`ye taşımak, içeriği ham bayt
    olarak döndürüp okuma yolunu bozardı.
    """
    from grounded_assistant.artifacts.serialize import deserialize

    assert deserialize(b"a,b\n1,2\n", "text/csv") == "a,b\n1,2\n"
    assert deserialize(b"# baslik", "text/markdown") == "# baslik"
    assert deserialize(b"<p>x</p>", "text/html") == "<p>x</p>"
    assert deserialize(b"duz", "text/plain") == "duz"


def test_uzanti_anahtarda_korunuyor():
    """Depo anahtarı dosya tipini yansıtmalı — `.bin`e düşmemeli."""
    from grounded_assistant.artifacts.serialize import uzanti_icin

    for ct, uzanti in [("text/csv", ".csv"), ("text/markdown", ".md"),
                       ("text/html", ".html")]:
        assert uzanti_icin(ct) == uzanti


def test_ikili_tipler_de_uzanti_aliyor():
    """REGRESYON (2026-09-06, canlı depoda görüldü).

    Uzantı eşlemesi İKİ yerde ayrı ayrı duruyordu: `serialize` uzantı->tip,
    `service` tip->uzantı. PDF/PNG birincisine eklenince ikincisi güncellenmedi
    ve depo anahtarı `art_xxx.bin` oldu — nesne deposuna bakan biri bir PDF'i
    tanıyamıyordu. (Okuma bozulmuyordu; content_type kayıt defterinden gelir.)

    Artık ters harita TÜRETİLİYOR, o yüzden bu ayrışma bir daha olamaz.
    """
    from grounded_assistant.artifacts.serialize import uzanti_icin

    assert uzanti_icin("application/pdf") == ".pdf"
    assert uzanti_icin("image/png") == ".png"
    assert uzanti_icin("application/zip") == ".zip"
    # .jpg ve .jpeg aynı tipe düşüyor; ilk tanımlanan kanoniktir
    assert uzanti_icin("image/jpeg") == ".jpg"
    # Tanınmayan tip hâlâ .bin
    assert uzanti_icin("uydurma/tip") == ".bin"


def test_uzanti_haritalari_ayrisamaz():
    """İki yönün TEK tanımdan türediğinin kanıtı: her uzantı gidip geliyor."""
    from grounded_assistant.artifacts.serialize import (
        _UZANTI_TIPLERI,
        content_type_for_filename,
        uzanti_icin,
    )

    for uzanti, tip in _UZANTI_TIPLERI.items():
        assert content_type_for_filename("x" + uzanti) == tip
        # Geri dönüş aynı uzantı olmak zorunda DEĞİL (.jpeg -> image/jpeg ->
        # .jpg), ama dönen uzantı mutlaka AYNI tipe eşlenmeli.
        assert _UZANTI_TIPLERI[uzanti_icin(tip)] == tip


def test_depo_koku_dogrulaniyor():
    """REGRESYON (2026-09-06): `X-Artifact-Root` doğrulanmıyordu.

    Sandbox'ın kendi istemcisi bu başlığı göndermiyor, ama imajda `requests`
    var — LLM'in yazdığı kod ham bir POST atıp depo kökünü kendisi seçebilirdi.
    Bu, mimarinin ana iddiasını deliyordu: anahtarın geri kalanı
    (`owner/workflow/node/run`) jetondan geliyor, kök tek boşluktu.

    KFP'de de `pipeline_root` pipeline'ı TANIMLAYANIN ayarı, adım kodunun
    değil — doğru hizalama bu.

    UYGULAMA YOLUNDAN sınanıyor. İlk hâli `_kok_dogrula`'yı doğrudan çağırıyordu
    ve `"/mutlak"`ın reddedildiğini "kanıtlıyordu" — oysa servis önce
    `.strip("/")` uyguluyor, yani gerçekte `"mutlak"` olarak KABUL ediliyor.
    Canlı denemede 201 dönünce fark edildi. Bu, projede ikinci kez görülen aynı
    kusur: testin uygulamanın hiç geçmediği bir yolu sınaması.
    """
    from grounded_assistant.artifacts.service import _kok_dogrula

    def api_gibi(kok):
        """`create_stream`/`create_bytes`'ın yaptığının aynısı."""
        return _kok_dogrula(kok.strip("/"))

    for gecerli, beklenen in [("", ""), ("kfp", "kfp"), ("takim/proje", "takim/proje"),
                              ("a/b/c/d/e", "a/b/c/d/e"),
                              # baştaki/sondaki `/` NORMALİZE ediliyor, reddedilmiyor:
                              # `pipeline_root` sahada sık sık "/kfp/root" diye yazılıyor.
                              ("/mutlak", "mutlak"), ("kfp/", "kfp")]:
        assert api_gibi(gecerli) == beklenen

    # Asıl korunan şey: yol geçişi ve bozuk segmentler.
    for kotu in ("../../etc", "a//b", "..", "a/../b", "x" * 70,
                 "-bastaTire", "a/b/c/d/e/f", "bosluk li"):
        with pytest.raises(InvalidArtifactName):
            api_gibi(kotu)


def test_kotu_kok_ile_yazma_reddediliyor(service, sahte):
    """Uçtan uca: geçersiz kök bayt YAZILMADAN reddedilmeli."""
    with pytest.raises(InvalidArtifactName):
        yaz(service, "veri", name="x.txt", root="../../baskasi")
    assert sahte.put_sayisi == 0


def test_gecerli_kok_anahtara_giriyor(service):
    """Özellik korunuyor: meşru bir kök hâlâ çalışıyor (KFP `pipeline_root`)."""
    meta = yaz(service, "veri", name="x.txt", root="takim/proje")
    assert meta.storage_uri.split("/", 3)[3].startswith("takim/proje/")
