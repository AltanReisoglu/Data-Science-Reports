"""Sandbox'ın Artifact Service istemcisi — akışlı HTTP, base64 yok.

Bu dosya `entrypoint.py`'nin yanında imaja kopyalanır. Bilerek YALNIZCA
`requests` kullanıyor: sandbox imajında zaten var, ve artifact yolu artık
MCP'ye (fastmcp) bağlı değil.

## Neden MCP değil

Önceki yolda baytlar MCP tool çağrısının içinde base64 taşınıyordu — %33 şişme
ve iki uçta tam tampon. Burada gövde ham bayt olarak akıyor: yükleme diske
yazılmış dosyadan doğrudan, indirme parça parça.

## Kapsam

Her istek `X-Scope-Token` taşır. Sandbox'ın `workflow_id`'yi kendisi söylemesi
diye bir şey yok — servis onu jetondan okur.
"""

from __future__ import annotations

import json
import os
import shutil

import requests

ENDPOINT = os.environ.get("ARTIFACT_SERVICE_ENDPOINT", "").rstrip("/")

_PARCA = 1024 * 1024
_ZAMAN_ASIMI = (5, 120)  # (bağlantı, okuma) — büyük yüklemeler için okuma uzun


#: Bayt yolu: `proxy` (servis üzerinden) ya da `direct` (depoya doğrudan).
#: Yalnızca SIDECAR'da anlamlı — sandbox'ın istemcisinde bu yol hiç yok.
def _dogrudan_mi() -> bool:
    import s3_client  # noqa: PLC0415

    return (os.environ.get("PTC_ARTIFACT_TRANSFER", "proxy") == "direct"
            and s3_client.NesneDeposu.yapilandirildi())


class ArtifactServiceError(RuntimeError):
    """Servis 2xx dışında bir şey döndürdü."""


class ArtifactClient:
    def __init__(self, endpoint: str, scope_token: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self._oturum = requests.Session()
        self._oturum.headers["X-Scope-Token"] = scope_token

    # -- yazma ------------------------------------------------------------

    def put_bytes(
        self,
        data: bytes,
        content_type: str,
        name: str,
        parents: list[str] | None = None,
        ttl_seconds: int | None = None,
        artifact_type: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        return self._post(data, content_type, name, parents, ttl_seconds,
                          artifact_type, metadata)

    def put_file(
        self,
        path: str,
        content_type: str,
        name: str,
        ttl_seconds: int | None = None,
        parents: list[str] | None = None,
    ) -> dict:
        """Dosyayı belleğe almadan yükler — `requests` dosya nesnesini akıtır.

        `parents` (2026-09-05): süpürme yolu da soy ağacı taşıyor. Önceden
        sabit `None` geçiliyordu — yani `df.to_csv("/output/x.csv")` ile
        üretilen her şey kayıt defterine ÖKSÜZ giriyordu, oysa aynı
        çalıştırmada okunan girdiler belliydi.
        """
        if _dogrudan_mi():
            return self._dogrudan_yukle(path, content_type, name, parents, ttl_seconds)
        with open(path, "rb") as f:
            return self._post(f, content_type, name, parents, ttl_seconds, None, None)

    # -- doğrudan yükleme kipi (2026-09-07) --------------------------------
    #
    # KFP'nin iki kanalı: BAYT depoya doğrudan, KÜNYE servise. Servis bayt
    # yolundan çıkıyor; buradaki üç çağrı onun yerini alıyor.
    #
    # Bu yol, servisin akış sırasında yaptığı iki kontrolü DEVRALIYOR: pickle
    # reddi ve boyut sınırı. İkisi de burada, yani hâlâ sandbox'ın DIŞINDA
    # (bu kod sidecar container'ında çalışıyor) — garanti bozulmuyor, yeri
    # değişiyor.

    def _dogrudan_yukle(self, path, content_type, name, parents, ttl_seconds) -> dict:
        import hashlib  # noqa: PLC0415

        import s3_client  # noqa: PLC0415
        import serialize  # noqa: PLC0415

        boyut = os.path.getsize(path)
        with open(path, "rb") as f:
            bas = f.read(64)
        # pickle reddi — servis baytı görmediği için burada yapılmak ZORUNDA
        serialize.guvenlik_kontrolu(bas, content_type)

        ozet = hashlib.sha256()
        with open(path, "rb") as f:
            for parca in iter(lambda: f.read(1024 * 1024), b""):
                ozet.update(parca)
        content_hash = "sha256:" + ozet.hexdigest()

        # ① Bu içerik zaten var mı — varsa hiç yükleme yapma.
        ikiz = self._oturum.get(f"{self.endpoint}/artifacts/by-hash/{content_hash}",
                                timeout=_ZAMAN_ASIMI)
        if ikiz.status_code == 200:
            # Dedup YENİ bir kayıt satırı açıyor, var olanı paylaşmıyor:
            # aynı bayt iki farklı adla/soyla var olabilir. `artifact_id=None`
            # → `_kayit_ac` yeni bir kimlik üretiyor, `storage_uri` ortak.
            return self._kayit_ac(None, name, content_type, content_hash, boyut,
                                  ikiz.json()["storage_uri"], parents,
                                  ttl_seconds, dedup=True)

        # ② Anahtar ayır, ③ doğrudan yükle, ④ kaydı aç.
        a = self._oturum.post(f"{self.endpoint}/artifacts/allocate",
                              json={"name": name, "content_type": content_type},
                              timeout=_ZAMAN_ASIMI)
        self._dogrula(a)
        ayrilan = a.json()
        s3_client.NesneDeposu().yukle(ayrilan["key"], path, content_type, boyut)
        return self._kayit_ac(ayrilan["artifact_id"], name, content_type,
                              content_hash, boyut, ayrilan["storage_uri"],
                              parents, ttl_seconds, dedup=False)

    def _kayit_ac(self, artifact_id, name, content_type, content_hash, boyut,
                  storage_uri, parents, ttl_seconds, *, dedup) -> dict:
        import uuid  # noqa: PLC0415

        govde = {
            "artifact_id": artifact_id or ("art_" + uuid.uuid4().hex[:12]),
            "name": name, "content_type": content_type,
            "content_hash": content_hash, "size_bytes": boyut,
            "storage_uri": storage_uri, "parents": list(parents or ()),
            "dedup": dedup,
        }
        if ttl_seconds is not None:
            govde["ttl_seconds"] = ttl_seconds
        yanit = self._oturum.post(f"{self.endpoint}/artifacts/register",
                                  json=govde, timeout=_ZAMAN_ASIMI)
        self._dogrula(yanit)
        return yanit.json()

    def _post(self, govde, content_type, name, parents, ttl_seconds,
              artifact_type=None, metadata=None) -> dict:
        basliklar = {"Content-Type": content_type, "X-Artifact-Name": name}
        if parents:
            basliklar["X-Artifact-Parents"] = ",".join(parents)
        if ttl_seconds is not None:
            basliklar["X-Artifact-TTL"] = str(ttl_seconds)
        if artifact_type:
            basliklar["X-Artifact-Type"] = artifact_type
        if metadata:
            basliklar["X-Artifact-Metadata"] = json.dumps(metadata)
        yanit = self._oturum.post(
            f"{self.endpoint}/artifacts",
            data=govde,
            headers=basliklar,
            timeout=_ZAMAN_ASIMI,
        )
        self._dogrula(yanit)
        return yanit.json()

    # -- okuma ------------------------------------------------------------

    def get_bytes(
        self, artifact_id: str | None = None, name: str | None = None
    ) -> tuple[bytes, dict] | None:
        """Baytlar + künye. Bulunamazsa None (istisna değil)."""
        yanit = self._ham_getir(artifact_id, name)
        if yanit is None:
            return None
        with yanit:
            return yanit.content, self._kunye(yanit)

    def fetch_to_file(self, name: str, hedef: str, workflow_id: str | None = None) -> dict | None:
        """Artifact'i doğrudan diske yazar — hiçbir noktada tamamı bellekte olmaz.

        Yerleştirme ve `load_artifact` bunu kullanır. `name` içinde `@` varsa
        alias çözümü devreye girer: `rapor.pdf@onaylanmis`.
        """
        if _dogrudan_mi():
            return self._dogrudan_indir(name, hedef, workflow_id)
        yanit = self._ham_getir(None, name, workflow_id)
        if yanit is None:
            return None
        with yanit, open(hedef, "wb") as f:
            shutil.copyfileobj(yanit.raw, f, _PARCA)
        return self._kunye(yanit)

    def _dogrudan_indir(self, name: str, hedef: str,
                        workflow_id: str | None) -> dict | None:
        """Servisten yalnızca ADRES, baytlar depodan."""
        import s3_client  # noqa: PLC0415

        url = f"{self.endpoint}/artifacts/by-name/{name}/uri"
        if workflow_id and "@" not in name:
            url += f"?workflow={workflow_id}"
        yanit = self._oturum.get(url, timeout=_ZAMAN_ASIMI)
        if yanit.status_code == 404:
            return None
        self._dogrula(yanit)
        k = yanit.json()
        n = s3_client.NesneDeposu().indir(
            s3_client.anahtar_ayikla(k["storage_uri"]), hedef)
        return {"artifact_id": k["artifact_id"], "name": k["name"],
                "content_type": k["content_type"], "size_bytes": n}

    def _ham_getir(self, artifact_id: str | None, name: str | None,
                   workflow_id: str | None = None):
        """`workflow_id` verilirse isim O ÇALIŞTIRMADA aranır, tenant'a düşmez."""
        if artifact_id is not None:
            url = f"{self.endpoint}/artifacts/{artifact_id}"
        elif name is not None:
            # `@` alias ayıracı — quote edilmemeli, servis onu ayrıştırıyor
            # (MLflow'un `models:/<ad>@<alias>`'ı). Ad ve alias biçimi zaten
            # `[A-Za-z0-9._-]` ile sınırlı, yani URL'de kaçış gerektiren
            # başka bir karakter olamaz.
            url = f"{self.endpoint}/artifacts/by-name/{name}"
            if workflow_id and "@" not in name:
                # Alias tenant genelinde çözülür (MLflow'da da registry
                # genelinde); `workflow` ile daraltmak onu anlamsız kılardı.
                url += f"?workflow={workflow_id}"
        else:
            raise ValueError("artifact_id ya da name verilmeli")
        yanit = self._oturum.get(url, stream=True, timeout=_ZAMAN_ASIMI)
        if yanit.status_code == 404:
            yanit.close()
            return None
        self._dogrula(yanit)
        # `raw` üzerinden okuyabilmek için: aksi hâlde gzip/deflate çözülmeden
        # ham gövde gelir. Servis sıkıştırma yapmıyor ama açıkça garanti edelim.
        yanit.raw.decode_content = True
        return yanit

    def list_all(self) -> list[dict]:
        """Bu tenant'ta ne var — **çalıştırmalar arası** (2026-09-06).

        Manifestin kaynağı. Eskiden yalnızca kendi workflow'unu listeliyorduk;
        KFP'de bütün run'lar aynı `pipeline_root` altına yazıyor ve birbirinin
        çıktısını görebiliyor — istenen davranış o.
        """
        yanit = self._oturum.get(f"{self.endpoint}/artifacts", timeout=_ZAMAN_ASIMI)
        self._dogrula(yanit)
        return yanit.json()

    def list(self, workflow_id: str, node_id: str | None = None) -> list[dict]:
        yanit = self._oturum.get(
            f"{self.endpoint}/workflows/{workflow_id}/artifacts",
            params={"node_id": node_id} if node_id else None,
            timeout=_ZAMAN_ASIMI,
        )
        self._dogrula(yanit)
        return yanit.json()

    def metadata(self, artifact_id: str) -> dict:
        yanit = self._oturum.get(
            f"{self.endpoint}/artifacts/{artifact_id}/metadata", timeout=_ZAMAN_ASIMI
        )
        self._dogrula(yanit)
        return yanit.json()

    # -- iç ---------------------------------------------------------------

    @staticmethod
    def _kunye(yanit) -> dict:
        """Bayt yanıtının başlıklarından künye — ikinci bir istek gerekmesin diye."""
        return {
            "artifact_id": yanit.headers.get("X-Artifact-Id"),
            "name": yanit.headers.get("X-Artifact-Name"),
            "content_hash": yanit.headers.get("X-Artifact-Content-Hash"),
            "content_type": yanit.headers.get("Content-Type"),
            "size_bytes": int(yanit.headers.get("X-Artifact-Size") or 0),
        }

    @staticmethod
    def _dogrula(yanit) -> None:
        if yanit.status_code >= 400:
            # Servisin gerekçesi LLM'e geri dönmeli: "pickle reddedildi" ya da
            # "boyut sınırı aşıldı" mesajı, modelin kendini düzeltebilmesi için
            # jenerik bir HTTP hatasından çok daha değerli.
            try:
                gerekce = yanit.json().get("detail", yanit.text)
            except ValueError:
                gerekce = yanit.text
            raise ArtifactServiceError(f"[{yanit.status_code}] {gerekce}")


class ProxyClient:
    """Sandbox'ın istemcisi — 127.0.0.1'deki sidecar'a konuşur.

    Jeton BU TARAFTA YOK. Sidecar onu ekliyor. Ve bilerek YALNIZCA OKUMA var:
    yükleme kararı sidecar'ın, `/output`'a bakarak verdiği bir karar
    (Argo'nun `wait` container'ı gibi). LLM'in kodu buradan bir şey
    yükleyemez — yükleme uç noktası hiç yok.
    """

    def __init__(self, taban: str):
        self.taban = taban.rstrip("/")
        self._oturum = requests.Session()

    def list_all(self) -> list[dict]:
        yanit = self._oturum.get(f"{self.taban}/manifest", timeout=_ZAMAN_ASIMI)
        yanit.raise_for_status()
        return yanit.json()

    def fetch_to_file(self, name: str, hedef: str,
                      workflow_id: str | None = None) -> dict | None:
        params = {"name": name}
        if workflow_id:
            params["workflow"] = workflow_id
        yanit = self._oturum.get(f"{self.taban}/fetch", params=params,
                                 timeout=_ZAMAN_ASIMI, stream=True)
        if yanit.status_code == 404:
            return None
        yanit.raise_for_status()
        with yanit, open(hedef, "wb") as f:
            shutil.copyfileobj(yanit.raw, f, _PARCA)
        return {
            "artifact_id": yanit.headers.get("X-Artifact-Id"),
            "name": name,
            "content_type": yanit.headers.get("Content-Type"),
            "size_bytes": int(yanit.headers.get("Content-Length") or 0),
        }
