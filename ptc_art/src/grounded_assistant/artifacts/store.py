"""Nesne deposu erişimi — İKİ ayrı OpenShift sözleşmesini de okur.

## Neden iki tane

**Sözleşme A — ObjectBucketClaim** (ODF/NooBaa MCG). İki kaynak üretir:

    ConfigMap : BUCKET_NAME, BUCKET_HOST, BUCKET_PORT (+ BUCKET_REGION)
    Secret    : AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

**Sözleşme B — OpenShift AI connection** (eski adıyla data connection):

    Secret    : AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
                AWS_S3_ENDPOINT, AWS_S3_BUCKET, AWS_DEFAULT_REGION

B, 2026-09-04'te eklendi. Gerekçe: **OpenShift Container Platform çekirdeğinde
nesne depolama yok** — bu ürünün depolama dokümanı baştan sona PV/PVC/CSI'dır,
S3 yoktur. Nesne deposu ya ODF ile gelir ya da harici bir üründen. ODF kapsam
dışına çıkınca `BUCKET_*` değişkenlerini üretecek kimse kalmadı.

Red Hat'in kendi dokümanı, kullanıcı kodunun S3'e nasıl erişeceğini B
sözleşmesi + boto3 ile anlatıyor. Yani "OpenShift'te nasıl yapılıyorsa öyle
yapalım" demek, pratikte B'yi okumak demek. A da bırakıldı: ileride ODF gelirse
kod değişmez.

Her iki durumda da **kod değişmiyor, yalnızca manifest değişiyor.**

## Sandbox burayı GÖRMEZ

Depo kimlik bilgileri yalnızca **Artifact Service**'tedir (2026-09-04'e kadar
Tool Gateway'deydi; bkz. o tarihli mimari kararı — artifact işi gateway'den
ayrı bir servise çıkarıldı). Sandbox'ın elinde opak bir `artifact_id` vardır,
`s3://...` değil. Gerekçe iki katlı:
  - OBC'nin ürettiği hesap **bucket'a** kilitli, oturuma/workflow'a değil —
    oturum kapsamını uygulayabileceğimiz tek yer servisin kendisi.
  - Sandbox'a imzalı URL vermek, ona ağ politikasının görmediği ikinci bir
    çıkış vermektir (araştırma §6.3).
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BucketConfig:
    """OBC'nin ürettiği bağlantı bilgisi."""

    name: str
    host: str
    port: int
    access_key: str
    secret_key: str
    secure: bool = False
    region: str | None = None

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> BucketConfig:
        """İKİ ayrı sözleşmeyi de kabul eder — hangisi varsa.

        ## Neden iki tane (2026-09-04)

        Başta yalnızca **ObjectBucketClaim** sözleşmesi okunuyordu
        (`BUCKET_NAME`/`BUCKET_HOST`/`BUCKET_PORT` + `AWS_*`), çünkü plan ODF
        üzerinden bir OBC kullanmaktı. ODF kapsam dışına çıkınca o değişkenleri
        üretecek kimse kalmadı: **OpenShift Container Platform çekirdeğinde
        nesne depolama yok** — S3, ya ODF ile ya da harici bir üründen gelir.

        Bu durumda ekibin elinde olacak şey, Red Hat OpenShift AI'ın
        **connection** (eski adıyla data connection) Secret'ıdır. Red Hat'in
        kendi dokümanı S3 erişimini bu değişkenler üzerinden, boto3 ile
        yapmayı anlatıyor:

            AWS_ACCESS_KEY_ID · AWS_SECRET_ACCESS_KEY
            AWS_S3_ENDPOINT   · AWS_S3_BUCKET · AWS_DEFAULT_REGION

        Yani "OpenShift'te nasıl yapılıyorsa öyle yapalım" demek, pratikte bu
        sözleşmeyi okumak demek. OBC yolu da bırakıldı: ileride ODF gelirse ya
        da başka bir küme OBC üretirse kod yine çalışsın.

        Öncelik OBC'de: ikisi birden varsa OBC kazanır (daha spesifik ve
        host/port'u ayrı ayrı verdiği için ayrıştırma gerektirmez).
        """
        e = env if env is not None else dict(os.environ)

        if e.get("BUCKET_HOST") and e.get("BUCKET_NAME"):
            return cls._obcden(e)
        if e.get("AWS_S3_ENDPOINT"):
            return cls._baglantidan(e)

        raise KeyError(
            "Artifact deposu yapılandırılmamış. İki sözleşmeden biri gerekli:\n"
            "  (a) ObjectBucketClaim  : BUCKET_NAME, BUCKET_HOST [, BUCKET_PORT]\n"
            "  (b) OpenShift AI conn. : AWS_S3_ENDPOINT, AWS_S3_BUCKET\n"
            "İkisinde de AWS_ACCESS_KEY_ID ve AWS_SECRET_ACCESS_KEY zorunlu."
        )

    @classmethod
    def _obcden(cls, e: dict[str, str]) -> BucketConfig:
        port = int(e.get("BUCKET_PORT", "443"))
        # OBC bir TLS bayrağı vermiyor; yerleşik konvansiyon porttan çıkarmak.
        # BUCKET_TLS ile açıkça geçersiz kılınabilir.
        tls = e.get("BUCKET_TLS")
        secure = (tls.lower() in ("1", "true", "yes")) if tls else (port == 443)
        return cls(
            name=e["BUCKET_NAME"],
            host=e["BUCKET_HOST"],
            port=port,
            access_key=cls._zorunlu(e, "AWS_ACCESS_KEY_ID"),
            secret_key=cls._zorunlu(e, "AWS_SECRET_ACCESS_KEY"),
            secure=secure,
            region=e.get("BUCKET_REGION") or None,
        )

    @classmethod
    def _baglantidan(cls, e: dict[str, str]) -> BucketConfig:
        """OpenShift AI connection Secret'ından.

        `AWS_S3_ENDPOINT` tam bir URL olabiliyor (`https://s3.example.com:9000`)
        ya da çıplak host (`s3.example.com`). minio istemcisi şema kabul
        etmiyor, host:port istiyor — ayrıştırma burada yapılıyor.
        """
        host, port, secure = _endpoint_ayristir(e["AWS_S3_ENDPOINT"])
        bucket = e.get("AWS_S3_BUCKET")
        if not bucket:
            raise KeyError(
                "AWS_S3_ENDPOINT var ama AWS_S3_BUCKET yok — hangi bucket'a "
                "yazılacağı belirsiz. OpenShift AI connection'ı normalde ikisini "
                "birlikte üretir."
            )
        return cls(
            name=bucket,
            host=host,
            port=port,
            access_key=cls._zorunlu(e, "AWS_ACCESS_KEY_ID"),
            secret_key=cls._zorunlu(e, "AWS_SECRET_ACCESS_KEY"),
            secure=secure,
            region=e.get("AWS_DEFAULT_REGION") or None,
        )

    @staticmethod
    def _zorunlu(e: dict[str, str], anahtar: str) -> str:
        deger = e.get(anahtar)
        if not deger:
            raise KeyError(f"Artifact deposu yapılandırılmamış — eksik: {anahtar}")
        return deger


def _endpoint_ayristir(endpoint: str) -> tuple[str, int, bool]:
    """`https://host:9000` / `http://host` / `host:9000` → (host, port, tls).

    Şema yoksa port'tan çıkarım yapılır (443 → TLS), OBC yolundaki
    konvansiyonun aynısı. Şema varsa o kazanır.
    """
    ham = endpoint.strip()
    secure: bool | None = None
    if "://" in ham:
        sema, ham = ham.split("://", 1)
        secure = sema.lower() == "https"
    ham = ham.split("/", 1)[0]  # yol varsa at

    if ":" in ham:
        host, _, port_metni = ham.rpartition(":")
        port = int(port_metni)
    else:
        host = ham
        port = 443 if secure is not False else 80

    if secure is None:
        secure = port == 443
    return host, port, secure


class ObjectStore:
    """S3-uyumlu depo üzerinde bayt okuma/yazma. Metadata BURADA DEĞİL."""

    def __init__(self, config: BucketConfig, client=None) -> None:
        self.config = config
        if client is not None:
            self._client = client
            return
        from minio import Minio  # noqa: PLC0415

        self._client = Minio(
            config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure,
            region=config.region,
        )

    def ensure_bucket(self) -> None:
        """Bucket yoksa yaratır.

        Yalnızca YEREL geliştirme içindir — OpenShift'te bucket'ı OBC'nin
        kendisi yaratır ve OBC hesabının zaten bucket yaratma yetkisi yoktur.
        """
        if not self._client.bucket_exists(self.config.name):
            self._client.make_bucket(self.config.name)

    def put(self, key: str, data: bytes, content_type: str) -> str:
        """Baytları yazar, `s3://bucket/key` biçiminde URI döner."""
        self._client.put_object(
            self.config.name,
            key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return self.uri(key)

    def put_stream(self, key: str, fileobj, length: int, content_type: str) -> str:
        """`put`'un akış sürümü — baytların TAMAMI bellekte olmadan yazar.

        `fileobj` okunabilir ve `length` bayt sunabilir olmalı (ör. istemciden
        gelen gövdenin biriktirildiği `SpooledTemporaryFile`). Artifact Service
        büyük yüklemelerde bunu kullanır: 100 MiB'lik bir parquet için süreç
        belleği sabit kalır, dosya diske taşar.
        """
        self._client.put_object(
            self.config.name, key, fileobj, length=length, content_type=content_type
        )
        return self.uri(key)

    def get(self, key: str) -> bytes:
        yanit = self._client.get_object(self.config.name, key)
        try:
            return yanit.read()
        finally:
            yanit.close()
            yanit.release_conn()

    def iter_get(self, key: str, chunk_size: int = 1024 * 1024):
        """`get`'in akış sürümü — parça parça verir, tamamını belleğe almaz.

        Üretici tüketilene kadar bağlantı açık kalır; `finally` her durumda
        (tüketici erken kapatsa da) bağlantıyı havuza iade eder.
        """
        yanit = self._client.get_object(self.config.name, key)
        try:
            while parca := yanit.read(chunk_size):
                yield parca
        finally:
            yanit.close()
            yanit.release_conn()

    def delete(self, key: str) -> None:
        self._client.remove_object(self.config.name, key)

    def uri(self, key: str) -> str:
        return f"s3://{self.config.name}/{key}"
