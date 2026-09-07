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

Bu dosya HER İKİSİNİN karşılığı — `init` de, `wait` de:

    yerlestir()   → açılışta, kod BAŞLAMADAN, bu çalıştırmanın çıktılarını
                    `/output`'a indirir. Argo'nun `init`'i, KFP'nin
                    driver+launcher'ı.
    supur()       → main bitince `/output`'u tarayıp yükler. Argo'nun `wait`'i.

Kubernetes 1.29'dan beri yerleşik sidecar semantiği tam bunu veriyor:

    "Upon Pod termination, the kubelet postpones terminating sidecar containers
     until the main application container has fully stopped."

## Asıl kazanç: sandbox'ta YAZMA YOLU yok

Yalnızca jetonu taşımak yetmezdi — LLM'in kodu localhost proxy'ye de aynı
çağrıyı atabilirdi, yetenek değişmezdi. Kazanç, yükleme kararının artık
sandbox'ta VERİLMEMESİ: sidecar neyi yükleyeceğine `/output`'a bakarak
kendi karar veriyor. LLM'in etkileyebileceği tek şey dosya yazmak — yani
zaten kastedilen arayüz. Ad seçmek, TTL koymak, depo kökü belirlemek,
süpürme kuralını atlamak artık mümkün değil.

## Proxy neden hâlâ var

Kendi çıktıların `yerlestir()` ile hazır geliyor, onlar için ağa çıkmak yok.
Proxy yalnızca BAŞKA bir çalıştırmanın çıktısı için: sandbox
`load_artifact(workflow_id, ad)` çağırıyor, istek 127.0.0.1'e geliyor, jeton
burada ekleniyor. Bu, Cloudflare/Vercel'in "kimlik-bilgisiz istemci +
imzalayan proxy" deseninin pod içindeki hâli (§9.6.4).

2026-09-07'ye kadar proxy'nin asıl işi TEMBEL OKUMAYDI: sandbox'ta
`os.listdir`/`glob`/pandas/`open` yamalıydı ve bayt okuma çağrısının
ortasında iniyordu. Piyasada karşılığı olmayan tek desenimizdi; kaldırıldı.

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


def _kaydet(ad: str, ozet: str, artifact_id: str | None) -> None:
    """Sidecar'ın SUNDUĞU bir baytı deftere işler.

    İki yerden çağrılıyor: açılıştaki yerleştirme ve çalışma sırasındaki
    `/fetch`. Defter sandbox'ta değil burada — süpürmedeki "bunu ben verdim"
    kararı da, soy ağacının ebeveynleri de kurcalanamıyor.
    """
    with _kilit:
        _sunulan_ozet[ad] = ozet
        if artifact_id:
            _sunulan_kimlik.add(artifact_id)


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

            _kaydet(ad, hashlib.sha256(ham).hexdigest(), kunye.get("artifact_id"))
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
                # Kip de sabitleniyor (2026-09-07): yerleştirmede açılan bir
                # dizin süpürmede YENİDEN paketleniyor. Kip diskten okunsaydı
                # umask/çıkarma farkı hash'i değiştirir, "bunu ben verdim"
                # kontrolü tutmaz ve aynı içerik ikinci kez yüklenirdi.
                bilgi.mode = 0o644
                with open(tam, "rb") as f:
                    tar.addfile(bilgi, f)
    return _dosya_ozeti(hedef)


def _tari_ac(tar_yolu: str, hedef_dizin: str) -> None:
    """Tar'ı hedef dizine açar — yol geçişine karşı süzülmüş.

    `filter="data"` (CVE-2007-4559 karşılığı) arşiv dışına yazan girdileri,
    symlink'leri ve aygıt düğümlerini reddediyor. Arşivi biz üretmiş olsak da
    depoya başka bir tar girmiş olabilir; açan taraf kaynağına güvenmemeli.
    """
    import tarfile  # noqa: PLC0415

    with tarfile.open(tar_yolu, "r") as tar:
        try:
            tar.extractall(hedef_dizin, filter="data")  # noqa: S202
        except TypeError:  # `filter` 3.11.4'ten eski sürümlerde yok
            for uye in tar.getmembers():
                if not uye.isfile() or os.path.isabs(uye.name) or ".." in uye.name.split("/"):
                    continue
                tar.extract(uye, hedef_dizin)  # noqa: S202


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


# ── Yerleştirme: kod BAŞLAMADAN girdileri /output'a koyar ──────────────────


def yerlestir() -> int:
    """KFP'nin driver + launcher'ının karşılığı.

    KFP'de bir bileşenin girdileri container doğduğunda `.path`'te HAZIR
    durur: `kfp-driver` init container'ı `.uri`'yi MLMD'den çözer, launcher
    dosyayı indirir, kullanıcı kodu yalnızca yerel bir dosya görür. Argo'da
    aynı işi `init` container yapıyor. İkisinde de indirme kod BAŞLAMADAN
    bitiyor.

    2026-09-07'ye kadar bizde öyle değildi: `/output` sahte bir görünümdü
    (`os.listdir`, `glob`, pandas okuyucuları ve `open` yamalıydı) ve bayt
    `pd.read_parquet(...)` çağrısının ORTASINDA iniyordu. Hiçbir üründe böyle
    bir desen yok — tek gerçek icadımızdı. Bu fonksiyon onun yerine geçti.

    Kapsam BU ÇALIŞTIRMA: `/output` yalnızca kendi çıktılarını gösterir
    (KFP'de `pipeline_root/<run-id>/...`). Başka bir çalıştırmanın çıktısı
    `load_artifact(workflow_id, ad)` ile AÇIKÇA isteniyor — orada da tembel
    değil, çağrıldığı anda tamamı iniyor.

    Ölçüm (2026-09-07, 163 artifact'lik depo): workflow başına medyan 3 dosya
    / 13,7 KiB, azami 7 dosya / 35,3 KiB — `/output`'un 512Mi sınırının on
    binde yedisi. Eskiden bunu O(tenant) yapan bir prefetch vardı ve sınırı
    zorluyordu; hatalı olan prefetch değil KAPSAMI'ydı.
    """
    if not WORKFLOW_ID:
        return 0
    try:
        kayitlar = istemci.list_all()
    except Exception as exc:  # noqa: BLE001 — depo yoksa çalıştırma yine sürsün
        _olay("yerlestirme_atlandi", detail=str(exc)[:200])
        return 0

    # Ad başına EN YENİ. Servis yeniden-eskiye sıralı döndürüyor, ilk görülen
    # kazanıyor — `latest_in_workflow` ile aynı kural.
    secilen: dict[str, dict] = {}
    for k in kayitlar:
        ad = k.get("name")
        if ad and k.get("workflow_id") == WORKFLOW_ID and ad not in secilen:
            secilen[ad] = k

    sayi = 0
    for ad in sorted(secilen):
        hedef = os.path.join(OUTPUT_DIR, ad)
        try:
            kunye = istemci.fetch_to_file(ad, hedef, workflow_id=WORKFLOW_ID)
            if not kunye:
                continue
            _kaydet(ad, _dosya_ozeti(hedef), kunye.get("artifact_id"))
            # Dizin artifact'i tar olarak duruyor; kod gerçek bir dizin
            # görmeli. Tar siliniyor — süpürme dizini yeniden paketleyip
            # aynı hash'i bulacak ve "bunu ben verdim" deyip atlayacak.
            if ad.endswith(_DIZIN_SONEKI) and kunye.get("content_type") == _DIZIN_TIPI:
                dizin = os.path.join(OUTPUT_DIR, ad[: -len(_DIZIN_SONEKI)])
                os.makedirs(dizin, exist_ok=True)
                _tari_ac(hedef, dizin)
                os.unlink(hedef)
        except Exception as exc:  # noqa: BLE001 — biri patlarsa diğerleri sürsün
            _olay("yerlestirme_hatasi", name=ad, detail=str(exc)[:200])
            if os.path.exists(hedef):
                os.unlink(hedef)
            continue
        sayi += 1
        _olay("artifact", op="consumed", artifact_id=kunye.get("artifact_id"),
              name=kunye.get("name"), size_bytes=kunye.get("size_bytes"),
              content_type=kunye.get("content_type"), parents=[])
    return sayi


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

    # Sinyal kancaları yerleştirmeden ÖNCE: indirme sürerken SIGTERM gelirse
    # süpürmenin yine de çalışması gerekiyor.
    signal.signal(signal.SIGTERM, kapan)
    signal.signal(signal.SIGINT, kapan)

    # SIRA ÖNEMLİ: önce yerleştirme, sonra sunucu. Sandbox container'ı
    # `/healthz` cevap verene kadar bekliyor (entrypoint._proxy_bekle), yani
    # sunucuyu en sonda açmak "girdiler hazır" el sıkışmasının kendisi.
    # Kubernetes'in yerleşik sidecar'ı ana container'ı sidecar BAŞLAYINCA
    # başlatıyor, BİTİNCE değil — bu beklemeyi kubelet garanti etmiyor.
    yerlestirilen = yerlestir()

    sunucu = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), Proxy)
    threading.Thread(target=sunucu.serve_forever, daemon=True).start()
    _olay("sidecar_hazir", port=PROXY_PORT, yerlestirilen=yerlestirilen)

    bitti.wait()
    sunucu.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
