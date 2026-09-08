"""Sidecar'ın nesne deposu istemcisi — YALNIZCA sidecar imajında.

## Neden ayrı bir dosya ve ayrı bir imaj

Bu modül `minio` paketine bağımlı. Sandbox imajına koysaydık, LLM'in yazdığı
kod `import minio` yapabilirdi. Kimlik bilgisi orada olmadığı için bir şey
yapamazdı, ama doğrudan-yükleme kipinde pod'un depoya ağ rotası da AÇILIYOR —
yani geriye tek engel olarak kimlik doğrulama kalırdı.

Argo da aynısını yapıyor: `argoexec` kullanıcının imajından ayrı bir imaj.
Bu yüzden `Dockerfile.sidecar` var ve sandbox imajı `minio`suz kalıyor.
Kabul testindeki "sandbox'ta S3 SDK yok" kontrolü bu ayrımı koruyor.

## Kipler

    proxy   → baytlar Artifact Service üzerinden (ağ izolasyonu korunur)
    direct  → baytlar depoya doğrudan (KFP'nin iki kanalı)

Seçim `PTC_ARTIFACT_TRANSFER` ile; ikisi de destekleniyor çünkü aralarındaki
fark bir tercih, bir hata değil (bkz. PTC_Piyasa_Mentaliteleri §11.17).
"""

from __future__ import annotations

import io
import os


def _ayristir(endpoint: str) -> tuple[str, bool]:
    """`http://minio:9000` → `("minio:9000", False)`"""
    e = endpoint.strip()
    if e.startswith("https://"):
        return e[8:].rstrip("/"), True
    if e.startswith("http://"):
        return e[7:].rstrip("/"), False
    return e.rstrip("/"), False


class NesneDeposu:
    """S3-uyumlu depoya doğrudan okuma/yazma.

    Kimlik bilgisi ortam değişkeninden — ve o değişken YALNIZCA sidecar
    container'ında tanımlı. Container'lar ortam paylaşmıyor.
    """

    def __init__(self) -> None:
        from minio import Minio  # noqa: PLC0415

        uc = os.environ.get("PTC_S3_ENDPOINT", "")
        self.bucket = os.environ.get("PTC_S3_BUCKET", "")
        adres, guvenli = _ayristir(uc)
        self._c = Minio(
            adres,
            access_key=os.environ.get("PTC_S3_ACCESS_KEY", ""),
            secret_key=os.environ.get("PTC_S3_SECRET_KEY", ""),
            secure=guvenli,
            region=os.environ.get("PTC_S3_REGION") or None,
        )

    @staticmethod
    def yapilandirildi() -> bool:
        return bool(os.environ.get("PTC_S3_ENDPOINT")
                    and os.environ.get("PTC_S3_BUCKET")
                    and os.environ.get("PTC_S3_ACCESS_KEY"))

    def yukle(self, anahtar: str, yol: str, content_type: str, boyut: int) -> None:
        with open(yol, "rb") as f:
            self._c.put_object(self.bucket, anahtar, f, length=boyut,
                               content_type=content_type)

    def indir(self, anahtar: str, hedef: str) -> int:
        yanit = self._c.get_object(self.bucket, anahtar)
        try:
            n = 0
            with open(hedef, "wb") as f:
                for parca in yanit.stream(1024 * 1024):
                    f.write(parca)
                    n += len(parca)
            return n
        finally:
            yanit.close()
            yanit.release_conn()


def anahtar_ayikla(storage_uri: str) -> str:
    """`s3://bucket/a/b/c` → `a/b/c` — servisin `_anahtar_ayikla`'sıyla aynı."""
    return storage_uri.split("/", 3)[3]


def bos_akis(boyut: int = 0):
    return io.BytesIO(b"")
