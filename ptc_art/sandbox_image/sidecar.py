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

## Sandbox artifact için HİÇBİR ağ çağrısı yapmıyor

Girdilerin hepsi — kendi çıktıları, başka çalıştırmalarınki, alias'la
sabitlenmiş sürümler — kod BAŞLAMADAN diske konuyor. Sandbox yalnızca dosya
görüyor; ne bir API, ne bir localhost sunucusu, ne bir adres.

Bu, KFP'nin kullanıcı bileşeni için sağladığı garantinin aynısı: launcher
`.uri`'leri `.path`'e indirir, kullanıcı kodu hiçbir çağrı yapmaz.

2026-09-07'ye kadar burada 127.0.0.1'de küçük bir HTTP sunucusu vardı
(`/healthz`, `/manifest`, `/fetch`) ve `load_artifact` ona konuşuyordu.
Çapraz-workflow okuma da BEYANA taşınınca o sunucunun tek işi el sıkışma
kaldı — onun için de paylaşılan volume'de bir dosya yetiyor. Argo da 1.29
öncesinde sonlandırma sinyalini böyle veriyordu.

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

import artifact_client
import serialize

OUTPUT_DIR = os.environ.get("PTC_OUTPUT_DIR", "/output")
ARTIFACTS_DIR = os.environ.get("PTC_ARTIFACTS_DIR", "/artifacts")
SCRATCH_DIR = os.environ.get("PTC_SCRATCH_DIR", "/scratch")
WORKFLOW_ID = os.environ.get("PTC_WORKFLOW_ID", "")
SCOPE_TOKEN = os.environ.get("PTC_SCOPE_TOKEN", "")
#: "Girdiler yerinde" el sıkışması. `/scratch` iki container'da da mount
#: edilmiş ve SÜPÜRÜLMÜYOR — `/output`'a koysaydık artifact sanılırdı.
HAZIR_DOSYA = os.path.join(SCRATCH_DIR, ".ptc-girdiler-hazir")

#: BEYAN EDİLEN GİRDİLER — Argo'nun `inputs.artifacts`'i, KFP'nin bileşen
#: girdilerinin karşılığı. Virgülle ayrılmış artifact adları; artifact adı
#: `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` olduğu için virgül ayıracı güvenli.
#:
#:   "*"        → beyan YOK; bu çalıştırmanın HER çıktısı yerleşir (uyumluluk)
#:   ""         → beyan var ve BOŞ; hiçbir şey yerleşmez
#:   "a.csv,b"  → yalnızca bunlar
#:
#: Beyan neden önemli: yerleştirilen her artifact soy ağacında EBEVEYN oluyor
#: (MLMD `DECLARED_INPUT`). Beyansız çalışınca "bu çalıştırmanın her çıktısı"
#: ebeveyn sayılıyor ve grafik zamanla tam bağlı hâle geliyor.
INPUTS_HAM = os.environ.get("PTC_INPUTS", "*")

#: Süpürmede yok sayılacak adlar — kullanıcı çıktısı değiller.
_SUPURME_DISI = (".", "__")
_DIZIN_TIPI = "application/x-tar"
_DIZIN_SONEKI = ".tar"

istemci = artifact_client.ArtifactClient(artifact_client.ENDPOINT, SCOPE_TOKEN)

#: Sidecar'ın SUNDUĞU her baytın özeti — süpürmede "bunu ben verdim,
#: LLM üretmedi" kontrolü. Defter sandbox'ta değil, kurcalanamıyor.
_sunulan_ozet: dict[str, str] = {}

#: Soy ağacının ebeveynleri İKİ kaynaktan geliyor:
#:
#:   _istenen_kimlik  — `load_artifact` ile AÇIKÇA istenenler
#:   _yerlesen_kimlik — açılışta yerleştirilenler (ad -> artifact_id)
#:
#: İkisi de "beyan edilmiş girdi" sayılıyor. MLMD'nin olay tipi zaten bunu
#: söylüyor: `Event.DECLARED_INPUT` / `DECLARED_OUTPUT`. KFP, Argo ve Tekton
#: soyu beyandan çıkarıyor; hiçbiri "kod bunu gerçekten okudu mu" diye
#: BAKMIYOR. (2026-09-07'de kısa süre atime ile bakmayı denedik — sahada
#: emsali olmayan bir icattı, kaldırıldı.)
_istenen_kimlik: set[str] = set()
_yerlesen_kimlik: dict[str, str] = {}
_kilit = threading.Lock()


def _olay(tur: str, **alanlar) -> None:
    """Runner'ın ayrıştırdığı JSON satırı — entrypoint'inkiyle aynı sözleşme."""
    print(json.dumps({"type": tur, "timestamp": datetime.now(UTC).isoformat(),
                      **alanlar}), flush=True)


def _kaydet(ad: str, ozet: str, artifact_id: str | None, *, istendi: bool) -> None:
    """Sidecar'ın SUNDUĞU bir baytı deftere işler.

    `istendi=True`  : `/fetch` — kod `load_artifact` ile açıkça istedi.
    `istendi=False` : açılışta yerleştirildi; okunup okunmadığı henüz belirsiz.
    """
    with _kilit:
        _sunulan_ozet[ad] = ozet
        if not artifact_id:
            return
        if istendi:
            _istenen_kimlik.add(artifact_id)
        else:
            _yerlesen_kimlik[ad] = artifact_id


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
        istenen = set(_istenen_kimlik)
        yerlesen = dict(_yerlesen_kimlik)
        sunulan = dict(_sunulan_ozet)

    # Ebeveyn = BEYAN EDİLEN GİRDİLER. MLMD'nin `Event.DECLARED_INPUT`'u
    # neyse o: girdiyi kim beyan ettiyse soy ondan çıkar. Beyan `PTC_INPUTS`
    # ile geliyor (Argo `inputs.artifacts`, KFP bileşen girdisi); ayrıca
    # `load_artifact` çağrısının kendisi de bir beyan.
    parents = sorted(istenen | set(yerlesen.values()))

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


def _beyani_coz(ham: str) -> list[dict] | None:
    """`PTC_INPUTS` → yerleştirilecek girdilerin listesi; beyan yoksa None.

    Üç adresleme biçimi — üçü de KFP'de bir karşılığa denk geliyor:

        ad                    bu çalıştırmanın çıktısı   → /output/<ad>
        <workflow_id>/ad      BAŞKA çalıştırmanınki      → /artifacts/<wf>/<ad>
        ad@alias              sabitlenmiş sürüm          → /artifacts/_alias/<ad>

    KFP'de bunların hepsi `.uri` olarak beyan edilir ve launcher hepsini
    `.path`'e indirir; kullanıcı kodu hiçbir çağrı yapmaz. Bizde de artık öyle:
    çapraz-workflow okuma da BEYAN, çalışma anında bir çağrı değil.
    """
    if ham == "*":
        return None
    girdiler = []
    for parca in ham.split(","):
        p = parca.strip()
        if not p:
            continue
        if "@" in p:
            ad, _, takma = p.partition("@")
            girdiler.append({"ad": ad, "alias": takma, "wf": None,
                             "hedef_kok": os.path.join(ARTIFACTS_DIR, "_alias")})
        elif "/" in p:
            wf, _, ad = p.rpartition("/")
            girdiler.append({"ad": ad, "alias": None, "wf": wf,
                             "hedef_kok": os.path.join(ARTIFACTS_DIR, wf)})
        else:
            girdiler.append({"ad": p, "alias": None, "wf": WORKFLOW_ID,
                             "hedef_kok": OUTPUT_DIR})
    return girdiler


def _yerlestir_bir(g: dict) -> bool:
    """Tek bir beyan edilmiş girdiyi diske koyar. Başardıysa True."""
    ad = g["ad"]
    os.makedirs(g["hedef_kok"], exist_ok=True)
    hedef = os.path.join(g["hedef_kok"], ad)
    istek = f"{ad}@{g['alias']}" if g["alias"] else ad
    try:
        kunye = istemci.fetch_to_file(istek, hedef, workflow_id=g["wf"])
        if not kunye and not ad.endswith(_DIZIN_SONEKI) and not g["alias"]:
            # DİZİN BEYANI: ajan `/output/model.v1/` görüyor, depoda ad
            # `model.v1.tar`. İkisini de yazabilsin diye `.tar`a düşülüyor.
            ad = ad + _DIZIN_SONEKI
            hedef = os.path.join(g["hedef_kok"], ad)
            kunye = istemci.fetch_to_file(ad, hedef, workflow_id=g["wf"])
        if not kunye:
            _olay("beyan_karsilanmadi", eksik=[istek])
            return False
        _kaydet(ad, _dosya_ozeti(hedef), kunye.get("artifact_id"),
                istendi=g["wf"] != WORKFLOW_ID or bool(g["alias"]))
        # Dizin artifact'i tar olarak duruyor; kod gerçek bir dizin görmeli.
        # Tar siliniyor — süpürme dizini yeniden paketleyip aynı hash'i
        # bulacak ve "bunu ben verdim" deyip atlayacak.
        if ad.endswith(_DIZIN_SONEKI) and kunye.get("content_type") == _DIZIN_TIPI:
            dizin = hedef[: -len(_DIZIN_SONEKI)]
            os.makedirs(dizin, exist_ok=True)
            _tari_ac(hedef, dizin)
            os.unlink(hedef)
    except Exception as exc:  # noqa: BLE001 — biri patlarsa diğerleri sürsün
        _olay("yerlestirme_hatasi", name=istek, detail=str(exc)[:200])
        if os.path.exists(hedef):
            os.unlink(hedef)
        return False
    _olay("artifact", op="consumed", artifact_id=kunye.get("artifact_id"),
          name=kunye.get("name"), size_bytes=kunye.get("size_bytes"),
          content_type=kunye.get("content_type"), parents=[])
    return True


def yerlestir() -> int:
    """KFP'nin driver + launcher'ının karşılığı.

    KFP'de bir bileşenin girdileri container doğduğunda `.path`'te HAZIR
    durur: `kfp-driver` init container'ı `.uri`'yi MLMD'den çözer, launcher
    dosyayı indirir, kullanıcı kodu yalnızca yerel bir dosya görür. Argo'da
    aynı işi `init` container yapıyor.

    2026-09-07 (ikinci tur): çapraz-workflow okuma da buraya taşındı. Önce
    sandbox `load_artifact(...)` çağırıyordu — yani kodun çalışma anında bir
    ağ isteği vardı. Artık YOK: bütün girdiler, hangi çalıştırmadan gelirse
    gelsin, kod başlamadan yerleştiriliyor. Sandbox artifact için HİÇBİR
    çağrı yapmıyor.

    Beyan yoksa (`*`) bu çalıştırmanın bütün çıktıları yerleşiyor —
    uyumluluk yolu; soy ağacını genişletir, bkz. §11.14.
    """
    if not WORKFLOW_ID:
        return 0
    girdiler = _beyani_coz(INPUTS_HAM)

    if girdiler is not None:
        sayi = sum(1 for g in girdiler if _yerlestir_bir(g))
        _olay("yerlestirme_bitti", sayi=sayi, beyan=True)
        return sayi

    # ── beyansız: bu çalıştırmanın her çıktısı (uyumluluk) ────────────────
    try:
        kayitlar = istemci.list_all()
    except Exception as exc:  # noqa: BLE001 — depo yoksa çalıştırma yine sürsün
        _olay("yerlestirme_atlandi", detail=str(exc)[:200])
        return 0

    # Ad başına EN YENİ. Servis yeniden-eskiye sıralı döndürüyor, ilk görülen
    # kazanıyor — `latest_in_workflow` ile aynı kural.
    gorulen: set[str] = set()
    sayi = 0
    for k in kayitlar:
        ad = k.get("name")
        if not ad or k.get("workflow_id") != WORKFLOW_ID or ad in gorulen:
            continue
        gorulen.add(ad)
        if _yerlestir_bir({"ad": ad, "alias": None, "wf": WORKFLOW_ID,
                           "hedef_kok": OUTPUT_DIR}):
            sayi += 1
    _olay("yerlestirme_bitti", sayi=sayi, beyan=False)
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

    # SIRA ÖNEMLİ: önce yerleştirme, sonra HAZIR DOSYASI.
    #
    # Kubernetes'in yerleşik sidecar'ı ana container'ı sidecar BAŞLAYINCA
    # başlatıyor, BİTİNCE değil — yani "girdiler hazır" el sıkışmasını kubelet
    # garanti etmiyor, biz kurmak zorundayız.
    #
    # 2026-09-07 (ikinci tur): bu el sıkışma bir HTTP sunucusuydu
    # (`/healthz`). Artık paylaşılan volume'de bir DOSYA. Sebep: sandbox'ın
    # artifact için hiçbir ağ çağrısı kalmayınca, yalnızca el sıkışma uğruna
    # bir sunucu ayakta tutmanın anlamı kalmadı. Argo da 1.29 öncesinde
    # sonlandırma sinyalini paylaşılan volume'deki bir dosyayla veriyordu.
    yerlestirilen = yerlestir()
    try:
        with open(HAZIR_DOSYA, "w") as f:
            f.write(str(yerlestirilen))
    except OSError as exc:
        _olay("hazir_dosyasi_yazilamadi", detail=str(exc)[:200])
    _olay("sidecar_hazir", yerlestirilen=yerlestirilen)

    bitti.wait()
    sys.exit(0)


if __name__ == "__main__":
    main()
