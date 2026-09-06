"""Artifact Service'in HTTP yüzeyi — §27'nin REST API'si.

`tests/unit/test_artifact_service.py` kütüphane katmanını sınıyor; burası
2026-09-04'te eklenen HTTP katmanını sınıyor. Odak, yalnızca bu katmanda var
olan davranışlar:

  - akışlı yükleme (gövde parça parça okunuyor, tamamı belleğe alınmıyor)
  - akış SIRASINDA doğrulama: pickle ilk parçada, boyut sayarak
  - kapsam jetonunun başlıktan okunması ve workflow izolasyonu
  - künyenin yanıt başlıklarında dönmesi (ikinci istek gerekmesin diye)
"""

from __future__ import annotations

import io
import pickle

import pytest
from fastapi.testclient import TestClient

from grounded_assistant.artifacts.metadata import open_sqlite
from grounded_assistant.artifacts.scope import Scope, issue_token
from grounded_assistant.artifacts.service import ArtifactService
from grounded_assistant.artifacts.store import BucketConfig, ObjectStore

ANAHTAR = "test-imza-anahtari"
WF = "wf_42"
SAHIP = "altan"


class SahteMinio:
    """minio.Minio'nun kullandığımız yüzeyi — `read(amt)` destekli.

    Gerçek istemcinin döndürdüğü urllib3.HTTPResponse parçalı okumayı destekler;
    `iter_get` buna dayandığı için sahte de desteklemek zorunda.
    """

    def __init__(self):
        self.nesneler: dict[str, bytes] = {}

    def put_object(self, bucket, key, data, length, content_type):
        self.nesneler[key] = data.read()

    def get_object(self, bucket, key):
        akis = io.BytesIO(self.nesneler[key])

        class _Yanit:
            def read(self_inner, amt=None):
                return akis.read(amt) if amt else akis.read()

            def close(self_inner): ...
            def release_conn(self_inner): ...

        return _Yanit()

    def remove_object(self, bucket, key):
        self.nesneler.pop(key, None)

    def bucket_exists(self, bucket):
        return True


@pytest.fixture
def sahte():
    return SahteMinio()


@pytest.fixture
def client(sahte, monkeypatch):
    from services.artifact_service import app as modul

    cfg = BucketConfig(
        name="artifacts", host="localhost", port=9000,
        access_key="a", secret_key="b", secure=False,
    )
    servis = ArtifactService(
        metadata=open_sqlite(":memory:"), objects=ObjectStore(cfg, client=sahte)
    )
    monkeypatch.setattr(modul, "_SIGNING_KEY", ANAHTAR)
    modul._service.cache_clear()
    monkeypatch.setattr(modul, "_service", lambda: servis)
    return TestClient(modul.app)


def jeton(workflow_id=WF, run_id="run_1", node_id=None):
    return issue_token(
        ANAHTAR, Scope(workflow_id=workflow_id, run_id=run_id, owner=SAHIP, node_id=node_id)
    )


def bas(workflow_id=WF, ad="rapor.csv", tip="text/csv", **ek):
    return {
        "X-Scope-Token": jeton(workflow_id),
        "X-Artifact-Name": ad,
        "Content-Type": tip,
        **ek,
    }


# -- yazma + okuma ---------------------------------------------------------


def test_gidis_donus_baytlari_korur(client):
    veri = b"id,tutar\n1,100\n2,250\n"
    yaz = client.post("/artifacts", content=veri, headers=bas())
    assert yaz.status_code == 201, yaz.text
    assert yaz.json()["status"] == "stored"

    oku = client.get("/artifacts/by-name/rapor.csv", headers={"X-Scope-Token": jeton()})
    assert oku.status_code == 200
    assert oku.content == veri


def test_kunye_yanit_basliklarinda_doner(client):
    """İstemci ikinci bir metadata isteği atmak zorunda kalmasın."""
    yaz = client.post("/artifacts", content=b"merhaba", headers=bas(ad="not.txt"))
    kimlik = yaz.json()["artifact_id"]

    oku = client.get(f"/artifacts/{kimlik}", headers={"X-Scope-Token": jeton()})
    assert oku.headers["X-Artifact-Id"] == kimlik
    assert oku.headers["X-Artifact-Name"] == "not.txt"
    assert oku.headers["X-Artifact-Size"] == "7"
    assert oku.headers["X-Artifact-Content-Hash"].startswith("sha256:")


def test_buyuk_govde_akisla_yazilir(client, sahte):
    """Bellek eşiğini (8 MiB) aşan gövde diske taşarak yazılmalı, bozulmadan."""
    veri = b"x" * (9 * 1024 * 1024)
    yaz = client.post("/artifacts", content=veri, headers=bas(ad="buyuk.bin"))
    assert yaz.status_code == 201
    assert yaz.json()["size_bytes"] == len(veri)

    oku = client.get("/artifacts/by-name/buyuk.bin", headers={"X-Scope-Token": jeton()})
    assert oku.content == veri


# -- doğrulama -------------------------------------------------------------


def test_pickle_ilk_parcada_reddedilir(client, sahte):
    """CWE-502: reddedilen yükleme depoya TEK BAYT bile yazmamalı."""
    veri = pickle.dumps({"a": 1})
    yanit = client.post("/artifacts", content=veri, headers=bas(ad="kotu.bin"))
    assert yanit.status_code == 415
    assert "pickle" in yanit.json()["detail"].lower()
    assert sahte.nesneler == {}


def test_pickle_content_type_etiketinden_de_reddedilir(client):
    yanit = client.post(
        "/artifacts", content=b"zararsiz", headers=bas(ad="x.pkl", tip="application/x-pickle")
    )
    assert yanit.status_code == 415


def test_boyut_siniri_akis_sirasinda_kesilir(client, sahte):
    from services.artifact_service import app as modul

    modul._service().size_limit = 1024
    yanit = client.post("/artifacts", content=b"y" * 5000, headers=bas(ad="buyuk.bin"))
    assert yanit.status_code == 413
    assert sahte.nesneler == {}


def test_yol_gecisi_iceren_isim_reddedilir(client):
    yanit = client.post("/artifacts", content=b"x", headers=bas(ad="../../etc/shadow"))
    assert yanit.status_code == 400


def test_isimsiz_istek_reddedilir(client):
    yanit = client.post(
        "/artifacts", content=b"x", headers={"X-Scope-Token": jeton(), "Content-Type": "text/plain"}
    )
    assert yanit.status_code == 400


# -- kapsam ----------------------------------------------------------------


def test_jetonsuz_istek_401(client):
    assert client.post("/artifacts", content=b"x").status_code == 401
    assert client.get("/artifacts/by-name/rapor.csv").status_code == 401


def test_bozuk_jeton_401(client):
    yanit = client.get(
        "/artifacts/by-name/rapor.csv", headers={"X-Scope-Token": "uydurma.imza"}
    )
    assert yanit.status_code == 401


def test_baska_workflow_artifacti_GORUNUR(client):
    """2026-09-06: sınır workflow'dan TENANT'a taşındı.

    KFP'de izolasyon namespace düzeyinde — bütün run'lar aynı `pipeline_root`
    altına yazar ve biri diğerinin çıktısını okuyabilir. Çalıştırma başına
    mühürlemek bizim eklediğimiz bir şeydi; ürün "başka workflow'un
    artifact'ini gözlemleyip kullanabilsin" istediği için kalktı.
    """
    yaz = client.post("/artifacts", content=b"veri", headers=bas(workflow_id="wf_ureten"))
    kimlik = yaz.json()["artifact_id"]

    okuyan = {"X-Scope-Token": jeton(workflow_id="wf_okuyan")}
    assert client.get(f"/artifacts/{kimlik}", headers=okuyan).status_code == 200
    assert client.get("/artifacts/by-name/rapor.csv", headers=okuyan).content == b"veri"


def test_baska_TENANT_artifacti_gorunmez(client):
    """Sınır kalkmadı, TAŞINDI. Var-ama-yetkisiz ile hiç-yok yine ayrılmıyor."""
    yaz = client.post("/artifacts", content=b"gizli", headers=bas())
    kimlik = yaz.json()["artifact_id"]

    yabanci = issue_token(
        ANAHTAR, Scope(workflow_id=WF, run_id="run_1", owner="baska-tenant")
    )
    baskasi = {"X-Scope-Token": yabanci}
    assert client.get(f"/artifacts/{kimlik}", headers=baskasi).status_code == 404
    assert client.get("/artifacts/by-name/rapor.csv", headers=baskasi).status_code == 404


def test_tenant_listelemesi_workflowlari_kapsiyor(client):
    """`/artifacts` — sandbox manifestinin kaynağı, çalıştırmalar arası."""
    client.post("/artifacts", content=b"a", headers=bas(workflow_id="wf_a", ad="a.txt"))
    client.post("/artifacts", content=b"b", headers=bas(workflow_id="wf_b", ad="b.txt"))

    kayit = client.get("/artifacts", headers={"X-Scope-Token": jeton()}).json()
    assert {k["name"] for k in kayit} == {"a.txt", "b.txt"}
    assert {k["workflow_id"] for k in kayit} == {"wf_a", "wf_b"}


def test_workflow_yolu_suzuyor(client):
    """Yol parametresi artık yetki değil FİLTRE — panelin "şu çalıştırmaya bak"
    bağlantısı başka bir workflow'u da gösterebilmeli."""
    client.post("/artifacts", content=b"a", headers=bas(workflow_id="wf_a", ad="a.txt"))
    client.post("/artifacts", content=b"b", headers=bas(workflow_id="wf_b", ad="b.txt"))

    h = {"X-Scope-Token": jeton(workflow_id="wf_a")}
    kayit = client.get("/workflows/wf_b/artifacts", headers=h).json()
    assert [k["name"] for k in kayit] == ["b.txt"]


# -- listeleme (manifest yolu) ---------------------------------------------


def test_listeleme_yeniden_eskiye_ve_yalnizca_kunye(client):
    for ad in ("a.csv", "b.csv"):
        client.post("/artifacts", content=b"x,y\n1,2\n", headers=bas(ad=ad))

    yanit = client.get(f"/workflows/{WF}/artifacts", headers={"X-Scope-Token": jeton()})
    assert yanit.status_code == 200
    kayitlar = yanit.json()
    assert [k["name"] for k in kayitlar] == ["b.csv", "a.csv"]
    # Baytlar ve depo adresi ASLA sızmamalı — sandbox'ın elinde opak kimlik var.
    assert all("storage_uri" not in k and "content_b64" not in k for k in kayitlar)


def test_ayni_isim_yeni_surum_uretir(client):
    """§20: içerik değişince YENİ artifact_id; okuma en yeniyi çözer."""
    ilk = client.post("/artifacts", content=b"v1", headers=bas(ad="features")).json()
    ikinci = client.post("/artifacts", content=b"v2", headers=bas(ad="features")).json()
    assert ilk["artifact_id"] != ikinci["artifact_id"]

    oku = client.get("/artifacts/by-name/features", headers={"X-Scope-Token": jeton()})
    assert oku.content == b"v2"


def test_ayni_icerik_bayti_tekrar_yuklemez(client, sahte):
    """Dedup: yeni kayıt açılır ama depoya ikinci kez yazılmaz."""
    client.post("/artifacts", content=b"ayni", headers=bas(ad="a.txt"))
    assert len(sahte.nesneler) == 1
    client.post("/artifacts", content=b"ayni", headers=bas(ad="b.txt"))
    assert len(sahte.nesneler) == 1


# -- TTL reaper ------------------------------------------------------------


@pytest.fixture
def yonetim(monkeypatch):
    monkeypatch.setenv("PTC_ADMIN_TOKEN", "gizli-yonetim")
    return {"X-Admin-Token": "gizli-yonetim"}


def yaz_ttl(client, ad, ttl, icerik=b"veri"):
    return client.post("/artifacts", content=icerik, headers=bas(ad=ad, **{"X-Artifact-TTL": str(ttl)}))


def test_reap_jetonsuz_reddedilir(client, yonetim):
    assert client.post("/admin/reap").status_code == 401
    assert client.post("/admin/reap", headers={"X-Admin-Token": "yanlis"}).status_code == 401


def test_reap_sir_tanimsizsa_503(client, monkeypatch):
    """Açık bırakılmış bir hâli olmamalı."""
    monkeypatch.delenv("PTC_ADMIN_TOKEN", raising=False)
    assert client.post("/admin/reap", headers={"X-Admin-Token": "x"}).status_code == 503


def test_kapsam_jetonu_reape_yetmez(client, yonetim):
    """Sandbox'ın elindeki jetonla toplu silme tetiklenememeli."""
    yanit = client.post("/admin/reap", headers={"X-Admin-Token": jeton()})
    assert yanit.status_code == 401


def test_ttlsiz_artifact_hic_dokunulmaz(client, yonetim, sahte):
    """Varsayılan davranış: kimse TTL vermediyse reap hiçbir şey silmez."""
    client.post("/artifacts", content=b"kalici", headers=bas(ad="kalici.txt"))
    sonuc = client.post("/admin/reap", headers=yonetim).json()
    assert sonuc["aday"] == 0 and sonuc["silinen"] == 0
    assert len(sahte.nesneler) == 1


def test_suresi_dolan_silinir_dolmayani_kalir(client, yonetim, sahte):
    yaz_ttl(client, "eski.txt", ttl=-1, icerik=b"suresi-dolmus")   # geçmişte doldu
    yaz_ttl(client, "yeni.txt", ttl=86400, icerik=b"taze")

    sonuc = client.post("/admin/reap", headers=yonetim).json()
    assert sonuc["aday"] == 1 and sonuc["silinen"] == 1

    okunan = {"X-Scope-Token": jeton()}
    assert client.get("/artifacts/by-name/eski.txt", headers=okunan).status_code == 404
    assert client.get("/artifacts/by-name/yeni.txt", headers=okunan).content == b"taze"
    assert len(sahte.nesneler) == 1  # yalnızca taze olanın baytı kaldı


def test_dry_run_silmez(client, yonetim, sahte):
    yaz_ttl(client, "eski.txt", ttl=-1)
    sonuc = client.post("/admin/reap?dry_run=true", headers=yonetim).json()
    assert sonuc["dry_run"] is True and sonuc["aday"] == 1
    assert "silinen" not in sonuc
    assert len(sahte.nesneler) == 1  # hâlâ duruyor
    assert sonuc["artifacts"][0]["name"] == "eski.txt"


def test_dedup_paylasilan_bayt_korunur(client, yonetim, sahte):
    """İki kayıt aynı baytı gösteriyorsa, biri silinince bayt DURMALI."""
    yaz_ttl(client, "kopya-a.txt", ttl=-1, icerik=b"ayni-icerik")
    client.post("/artifacts", content=b"ayni-icerik", headers=bas(ad="kopya-b.txt"))
    assert len(sahte.nesneler) == 1  # dedup: tek bayt

    client.post("/admin/reap", headers=yonetim)

    okunan = {"X-Scope-Token": jeton()}
    assert client.get("/artifacts/by-name/kopya-a.txt", headers=okunan).status_code == 404
    # Süresi dolmayan kayıt hâlâ okunabilmeli — baytı silinmiş olsaydı 500 alırdık.
    assert client.get("/artifacts/by-name/kopya-b.txt", headers=okunan).content == b"ayni-icerik"


def test_reap_workflowlar_arasi_calisir(client, yonetim):
    """Kapsam jetonu tek workflow'a bağlı; reaper hepsini süpürmeli."""
    yaz_ttl(client, "a.txt", ttl=-1)
    client.post("/artifacts", content=b"x", headers={**bas(workflow_id="wf_ikinci", ad="b.txt"),
                                                     "X-Artifact-TTL": "-1"})
    sonuc = client.post("/admin/reap", headers=yonetim).json()
    assert sonuc["silinen"] == 2


# -- OpenShift/KFP hizalaması (2026-09-04) ---------------------------------


def test_tip_icerikten_cikarilir(client):
    """KFP'de her artifact tipli. LLM yazmasa da tip düşmeli."""
    csv = client.post("/artifacts", content=b"a,b\n1,2\n", headers=bas(ad="d.csv")).json()
    assert csv["type"] == "system.Dataset"

    duz = client.post(
        "/artifacts", content=b"selam", headers=bas(ad="n.txt", tip="text/plain")
    ).json()
    assert duz["type"] == "system.Artifact"

    html = client.post(
        "/artifacts", content=b"<p>x</p>", headers=bas(ad="r.html", tip="text/html")
    ).json()
    assert html["type"] == "system.HTML"


def test_acik_tip_cikarimi_ezer(client):
    y = client.post(
        "/artifacts", content=b"a,b\n1,2\n",
        headers={**bas(ad="m.csv"), "X-Artifact-Type": "system.Model"},
    ).json()
    assert y["type"] == "system.Model"


def test_bilinmeyen_tip_taban_tipe_duser(client):
    """Tip bir ETİKET, güvenlik kontrolü değil — reddetmek yerine düşürülüyor."""
    y = client.post(
        "/artifacts", content=b"x",
        headers={**bas(ad="x.txt", tip="text/plain"), "X-Artifact-Type": "uydurma.Tip"},
    ).json()
    assert y["type"] == "system.Artifact"


def test_kullanici_metadatasi_saklanir_ve_doner(client):
    """KFP'deki `.metadata` — serbest anahtar-değer."""
    import json as _json

    y = client.post(
        "/artifacts", content=b"a,b\n1,2\n",
        headers={**bas(ad="ds.csv"),
                 "X-Artifact-Metadata": _json.dumps({"satir": 1, "kaynak": "ticket"})},
    ).json()
    assert y["metadata"] == {"satir": 1, "kaynak": "ticket"}

    kunye = client.get(f"/artifacts/{y['artifact_id']}/metadata",
                       headers={"X-Scope-Token": jeton()}).json()
    assert kunye["metadata"]["kaynak"] == "ticket"


def test_bozuk_metadata_sessizce_yok_sayilir(client):
    y = client.post(
        "/artifacts", content=b"x",
        headers={**bas(ad="x.txt", tip="text/plain"), "X-Artifact-Metadata": "{bozuk"},
    )
    assert y.status_code == 201 and y.json()["metadata"] == {}


def test_artifact_root_anahtari_onekliyor(client, sahte):
    """KFP'nin `pipeline_root`'u: bucket'ın neresine yazılacağı YAPILANDIRMA."""
    client.post("/artifacts", content=b"x",
                headers={**bas(ad="a.txt", tip="text/plain"),
                         "X-Artifact-Root": "v2/artifacts"})
    anahtar = next(iter(sahte.nesneler))
    assert anahtar.startswith("v2/artifacts/altan/")


def test_root_verilmezse_eski_duzen_korunur(client, sahte):
    client.post("/artifacts", content=b"y", headers=bas(ad="b.txt", tip="text/plain"))
    assert next(iter(sahte.nesneler)).startswith("altan/")


def test_tip_bayt_yanit_basliginda_da_doner(client):
    y = client.post("/artifacts", content=b"a,b\n1,2\n", headers=bas(ad="t.csv")).json()
    oku = client.get(f"/artifacts/{y['artifact_id']}", headers={"X-Scope-Token": jeton()})
    assert oku.headers["X-Artifact-Type"] == "system.Dataset"
