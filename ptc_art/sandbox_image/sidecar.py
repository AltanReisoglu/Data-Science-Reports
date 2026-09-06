"""Artifact sidecar — Argo'nun `init` + `wait` container modelinin karşılığı.

## Neden ayrı bir container

2026-09-06'ya kadar aktarımı `entrypoint.py` yapıyordu: yani LLM'in kodunun
çalıştığı container'ın İÇİNDEN, ortam değişkenindeki kapsam jetonuyla. Bu,
`§9.6`'da KFP'yi eleştirdiğimiz konumun aynısıydı — sarmalayıcı, sınır değil.

Argo Workflows bu problemi yıllar önce çözmüş ve dört farklı yerleşimi
(`docker`, `kubelet`, `k8sapi`, `pns`) deneyip v3.4'te hepsini kaldırmış.
Kalan model:

    init container   → girdileri paylaşılan volume'e indirir
    main container   → kullanıcı kodu; SADECE düz dosya yolu okur/yazar
    wait  (sidecar)  → main bitince çıktıları volume'den toplayıp yükler

Bu dosya `wait` container'ın karşılığı. Kubernetes 1.29'dan beri yerleşik
sidecar semantiği tam bunu veriliyor:

    "Upon Pod termination, the kubelet postpones terminating sidecar containers
     until the main application container has fully stopped."

## Asıl kazanç: sandbox'ta YAZMA YOLU yok

Yalnızca jetonu taşımak yetmezdi — LLM'in kodu localhost proxy'ye de aynı
çağrıyı atabilirdi, yetenek değişmezdi. Kazanç, yükleme kararının artık
sandbox'ta VERİLMEMESİ: sidecar neyi yükleyeceğine `/output`'a bakarak
kendi karar veriyor. LLM'in etkileyebileceği tek şey dosya yazmak — yani
zaten kastedilen arayüz. Ad seçmek, TTL koymak, depo kökü belirlemek,
süpürme kuralını atlamak artık mümkün değil.

## Okuma neden yine proxy üzerinden

Tembel okuma çalışma SIRASINDA lazım (`pd.read_parquet("/output/x")`).
Sidecar 127.0.0.1'de küçük bir sunucu açıyor; jeton onda kalıyor. Bu,
Cloudflare/Vercel'in "kimlik-bilgisiz mount + imzalayan proxy" deseninin
pod içindeki hâli (§9.6.4).

Sidecar sunduğu her baytın sha256'sını tutuyor — süpürmede "bunu ben verdim,
LLM üretmedi" kararını buradan veriyor ve soy ağacının ebeveynlerini de
buradan çıkarıyor. İkisi de kurcalanamaz: kayıt sandbox'ta değil.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import artifact_client
import serialize

OUTPUT_DIR = os.environ.get("PTC_OUTPUT_DIR", "/output")
ARTIFACTS_DIR = os.environ.get("PTC_ARTIFACTS_DIR", "/artifacts")
SCRATCH_DIR = os.environ.get("PTC_SCRATCH_DIR", "/scratch")
WORKFLOW_ID = os.environ.get("PTC_WORKFLOW_ID", "")
SCOPE_TOKEN = os.environ.get("PTC_SCOPE_TOKEN", "")
PROXY_PORT = int(os.environ.get("PTC_PROXY_PORT", "8099"))

#: Süpürmede yok sayılacak adlar — kullanıcı çıktısı değiller.
_SUPURME_DISI = (".", "__")
_DIZIN_TIPI = "application/x-tar"
_DIZIN_SONEKI = ".tar"

istemci = artifact_client.ArtifactClient(artifact_client.ENDPOINT, SCOPE_TOKEN)

#: Sidecar'ın SUNDUĞU artifact'ler. İki işi var:
#:   ad -> sha256   : süpürmede "bunu ben verdim" kontrolü
#:   artifact_id'ler: soy ağacının ebeveynleri
_sunulan_ozet: dict[str, str] = {}
_sunulan_kimlik: set[str] = set()
_kilit = threading.Lock()


def _olay(tur: str, **alanlar) -> None:
    """Runner'ın ayrıştırdığı JSON satırı — entrypoint'inkiyle aynı sözleşme."""
    print(json.dumps({"type": tur, "timestamp": datetime.now(UTC).isoformat(),
                      **alanlar}), flush=True)


# ── Localhost proxy: sandbox'ın OKUMA yolu ────────────────────────────────


class Proxy(BaseHTTPRequestHandler):
    """Yalnızca okuma. Yazma uç noktası BİLEREK yok — yükleme kararı sidecar'ın."""

    def log_message(self, *a):  # pod log'unu HTTP gürültüsüyle doldurma
        pass

    def _json(self, kod: int, govde) -> None:
        ham = json.dumps(govde).encode()
        self.send_response(kod)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(ham)))
        self.end_headers()
        self.wfile.write(ham)

    def do_GET(self) -> None:  # noqa: N802
        yol = urlparse(self.path)
        if yol.path == "/healthz":
            self._json(200, {"status": "ok"})
            return

        if yol.path == "/manifest":
            try:
                self._json(200, istemci.list_all())
            except Exception as exc:  # noqa: BLE001
                self._json(502, {"hata": str(exc)[:200]})
            return

        if yol.path == "/fetch":
            q = parse_qs(yol.query)
            ad = (q.get("name") or [""])[0]
            wf = (q.get("workflow") or [None])[0]
            if not ad:
                self._json(400, {"hata": "name gerekli"})
                return
            gecici = os.path.join(SCRATCH_DIR, f"_proxy_{os.getpid()}_{threading.get_ident()}")
            try:
                kunye = istemci.fetch_to_file(ad, gecici, workflow_id=wf)
                if not kunye:
                    self._json(404, {"hata": "bulunamadı"})
                    return
                with open(gecici, "rb") as f:
                    ham = f.read()
            except Exception as exc:  # noqa: BLE001
                self._json(502, {"hata": str(exc)[:200]})
                return
            finally:
                if os.path.exists(gecici):
                    os.unlink(gecici)

            with _kilit:
                _sunulan_ozet[ad] = hashlib.sha256(ham).hexdigest()
                if kunye.get("artifact_id"):
                    _sunulan_kimlik.add(kunye["artifact_id"])
            _olay("artifact", op="consumed", artifact_id=kunye.get("artifact_id"),
                  name=kunye.get("name"), size_bytes=kunye.get("size_bytes"),
                  content_type=kunye.get("content_type"), parents=[])

            self.send_response(200)
            self.send_header("Content-Type", kunye.get("content_type") or "application/octet-stream")
            self.send_header("Content-Length", str(len(ham)))
            if kunye.get("artifact_id"):
                self.send_header("X-Artifact-Id", kunye["artifact_id"])
            self.end_headers()
            self.wfile.write(ham)
            return

        self._json(404, {"hata": "bilinmeyen yol"})


# ── Süpürme: sandbox bitince, SIGTERM ile ─────────────────────────────────


def _dosya_ozeti(yol: str) -> str:
    ozet = hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1024 * 1024), b""):
            ozet.update(parca)
    return ozet.hexdigest()


def _dizini_paketle(dizin: str, hedef: str) -> str:
    """Tekrarlanabilir tar — `entrypoint.py`'dekiyle aynı kural (dedup için)."""
    import tarfile  # noqa: PLC0415

    with tarfile.open(hedef, "w") as tar:
        for kok, alt, dosyalar in os.walk(dizin):
            alt.sort()
            for dosya in sorted(dosyalar):
                tam = os.path.join(kok, dosya)
                if not os.path.isfile(tam) or os.path.islink(tam):
                    continue
                bilgi = tar.gettarinfo(tam, arcname=os.path.relpath(tam, dizin))
                bilgi.mtime = 0
                bilgi.uid = bilgi.gid = 0
                bilgi.uname = bilgi.gname = ""
                with open(tam, "rb") as f:
                    tar.addfile(bilgi, f)
    return _dosya_ozeti(hedef)


def supur() -> None:
    """`/output`'un üst düzeyini artifact'e çevirir. Argo'nun `wait`'i budur.

    Ebeveynler, sidecar'ın bu koşuda SUNDUĞU artifact'ler — kayıt sandbox'ta
    olmadığı için kurcalanamıyor.
    """
    with _kilit:
        parents = sorted(_sunulan_kimlik)
        sunulan = dict(_sunulan_ozet)

    try:
        adlar = sorted(os.listdir(OUTPUT_DIR))
    except OSError:
        return

    for ad in adlar:
        yol = os.path.join(OUTPUT_DIR, ad)
        if ad.startswith(_SUPURME_DISI):
            continue
        try:
            if os.path.isdir(yol):
                paket = os.path.join(SCRATCH_DIR, f"_supurme_{ad}{_DIZIN_SONEKI}")
                try:
                    ozet = _dizini_paketle(yol, paket)
                    if sunulan.get(ad + _DIZIN_SONEKI) == ozet:
                        continue  # biz verdik, LLM dokunmadı
                    kunye = istemci.put_file(paket, _DIZIN_TIPI,
                                             _ad_duzelt(ad + _DIZIN_SONEKI),
                                             parents=parents)
                finally:
                    if os.path.exists(paket):
                        os.unlink(paket)
            elif os.path.isfile(yol):
                if sunulan.get(ad) == _dosya_ozeti(yol):
                    continue  # biz verdik, LLM dokunmadı
                kunye = istemci.put_file(yol, serialize.content_type_for_filename(ad),
                                         _ad_duzelt(ad), parents=parents)
            else:
                continue
        except Exception as exc:  # noqa: BLE001 — best-effort; biri patlarsa diğerleri sürsün
            _olay("artifact_skipped", name=ad, detail=str(exc)[:200])
            continue
        _olay("artifact", op="produced", artifact_id=kunye["artifact_id"],
              name=kunye["name"], size_bytes=kunye.get("size_bytes"),
              content_type=kunye.get("content_type"),
              parents=list(kunye.get("parents") or parents))


def _ad_duzelt(dosya_adi: str) -> str:
    """Servisin kabul ettiği biçime çevirir — `entrypoint._gecerli_artifact_adi`
    ile aynı kural, o dosyaya bağımlılık yaratmadan."""
    import re  # noqa: PLC0415

    temiz = re.sub(r"[^A-Za-z0-9._-]", "-", dosya_adi)[:128]
    return temiz if re.match(r"^[A-Za-z0-9]", temiz) else "a" + temiz[:127]


def main() -> None:
    bitti = threading.Event()

    def kapan(signum, frame):  # noqa: ARG001
        # SIGTERM = ana container bitti (kubelet sidecar'ı en son durduruyor).
        # Süpürme TAM BURADA: çıktılar hazır, LLM artık yazamıyor.
        try:
            supur()
        finally:
            # Runner bu satırı görünce dönüyor — pod'un terminal faza geçmesini
            # beklemeye gerek kalmıyor. Ölçümde bu bekleme ~3 sn tutuyordu.
            _olay("supurme_bitti")
            bitti.set()

    signal.signal(signal.SIGTERM, kapan)
    signal.signal(signal.SIGINT, kapan)

    sunucu = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), Proxy)
    threading.Thread(target=sunucu.serve_forever, daemon=True).start()
    _olay("sidecar_hazir", port=PROXY_PORT)

    bitti.wait()
    sunucu.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
