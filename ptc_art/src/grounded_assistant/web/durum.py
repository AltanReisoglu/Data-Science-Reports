"""Durum paneli için veri toplayıcı — pod'lar, akışlar, artifact'ler.

## Neden yazıldı

Bu işi yapan hazır UI'lar var ama her biri **tek bir katmanı** gösteriyor:

  Hubble UI          → ağ akışları
  OpenShift Console  → pod'lar
  MinIO Console      → bucket içeriği
  Kubeflow / MLMD UI → artifact soy ağacı

Bizim yapımızda bu dördü birbirine bağlı: hangi pod hangi artifact'i üretti,
o pod'un hangi hedeflere çıkma izni var, artifact hangi workflow'a ait. Dördünü
tek ekranda gösteren hazır bir şey yok — bu modül onu besliyor.

## Veri nereden geliyor

  pod'lar    → Kubernetes API (runner ile aynı kube config)
  politika   → CiliumNetworkPolicy nesneleri (izinli kenarlar)
  artifact   → Artifact Service'in kendi REST API'si

Hepsi **salt okuma**. Panel hiçbir şeyi değiştirmiyor.

## Hata durumunda

Her toplayıcı kendi başına başarısız olabilir ve o bölüm `error` alanıyla
döner; panelin geri kalanı çalışmaya devam eder. Cluster kapalıyken UI'ın
tamamen kararmasındansa "bu bölüm okunamadı" demek daha kullanışlı.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

#: Panelde gösterilen kalıcı bileşenler ve rolleri. Sandbox'lar buraya
#: eklenmiyor — onlar efemer, ayrı bir listede akıyorlar.
BILESENLER = [
    {"ad": "tool-gateway", "etiket": "Tool Gateway", "rol": "10 tool proxy'si",
     "internet": True, "depo": False},
    {"ad": "artifact-service", "etiket": "Artifact Service", "rol": "Artifact + kayıt defteri",
     "internet": False, "depo": True},
    {"ad": "minio", "etiket": "MinIO", "rol": "S3-uyumlu nesne deposu",
     "internet": False, "depo": None},
]


def _yas(ts) -> str:
    if ts is None:
        return "—"
    saniye = (datetime.now(UTC) - ts).total_seconds()
    if saniye < 90:
        return f"{int(saniye)} sn"
    if saniye < 5400:
        return f"{int(saniye // 60)} dk"
    if saniye < 172800:
        return f"{int(saniye // 3600)} sa"
    return f"{int(saniye // 86400)} gün"


def _core():
    from kubernetes import client, config  # noqa: PLC0415

    config.load_kube_config()
    return client.CoreV1Api()


def podlar() -> dict:
    """Kalıcı bileşenlerin ve o an yaşayan sandbox'ların durumu."""
    ns = os.environ.get("PTC_NAMESPACE", "default")
    try:
        v1 = _core()
        hepsi = v1.list_namespaced_pod(namespace=ns).items
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Kubernetes API okunamadı: {exc}"}

    def _bul(ad):
        for p in hepsi:
            if (p.metadata.labels or {}).get("app") == ad:
                return p
        return None

    kalici = []
    for b in BILESENLER:
        p = _bul(b["ad"])
        kalici.append({
            **b,
            "durum": (p.status.phase if p else "Yok"),
            "hazir": bool(p and all(c.ready for c in (p.status.container_statuses or []))),
            "restart": sum(c.restart_count for c in (p.status.container_statuses or [])) if p else 0,
            "yas": _yas(p.status.start_time if p else None),
            "pod": p.metadata.name if p else None,
        })

    # Sandbox'lar efemer: birkaç saniye yaşayıp yok oluyorlar. Panelde ayrı
    # duruyorlar ki "hiç sandbox yok" hâli anormal görünmesin — normal hâl bu.
    sandboxlar = [{
        "pod": p.metadata.name,
        "durum": p.status.phase,
        "yas": _yas(p.status.start_time),
        "run_id": (p.metadata.labels or {}).get("ptc-run-id", "—"),
    } for p in hepsi if (p.metadata.labels or {}).get("app") == "ptc-sandbox"]

    return {"kalici": kalici, "sandboxlar": sandboxlar, "namespace": ns}


def akislar() -> dict:
    """İzinli ağ kenarları — CiliumNetworkPolicy'lerden okunuyor.

    Hubble akan paketleri gösteriyor; burada gösterilen **izin verilmiş
    kenarlar**. Fark önemli: Hubble "ne oldu" der, bu panel "neye izin var" der.
    İkincisi bir PoC'de daha çok işe yarıyor çünkü asıl iddia politikanın
    kendisi.
    """
    ns = os.environ.get("PTC_NAMESPACE", "default")
    try:
        from kubernetes import client, config  # noqa: PLC0415

        config.load_kube_config()
        api = client.CustomObjectsApi()
        ham = api.list_namespaced_custom_object(
            "cilium.io", "v2", ns, "ciliumnetworkpolicies"
        )["items"]
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Politikalar okunamadı: {exc}"}

    kenarlar = []
    for pol in ham:
        spec = pol.get("spec") or {}
        kaynak = ((spec.get("endpointSelector") or {}).get("matchLabels") or {}).get("app")
        for kural in spec.get("egress") or []:
            for hedef in kural.get("toEndpoints") or []:
                ad = (hedef.get("matchLabels") or {}).get("app")
                dns = (hedef.get("matchLabels") or {}).get("k8s-app")
                if ad or dns:
                    kenarlar.append({"kaynak": kaynak, "hedef": ad or dns, "tur": "ic"})
            for fqdn in kural.get("toFQDNs") or []:
                if fqdn.get("matchName"):
                    kenarlar.append({"kaynak": kaynak, "hedef": fqdn["matchName"],
                                     "tur": "dis"})
    return {"kenarlar": kenarlar, "politika_sayisi": len(ham)}


def artifactler(workflow_id: str | None, jeton_uret) -> dict:
    """Artifact Service'ten künyeler. Bayt çekilmiyor."""
    from grounded_assistant.agent import artifact_context  # noqa: PLC0415

    if not artifact_context.servis_adresi():
        return {"error": "ARTIFACT_SERVICE_URL tanımlı değil "
                         "(yerelde: kubectl port-forward svc/artifact-service 8080:8080)"}
    if not workflow_id:
        return {"kayitlar": [], "not": "Bu oturumda henüz workflow yok"}
    jeton = jeton_uret(workflow_id)
    if not jeton:
        return {"error": "Kapsam jetonu üretilemedi (ptc-scope-signing Secret'ı okunamadı)"}
    kayitlar = artifact_context.kunyeleri_getir(workflow_id, jeton)
    if kayitlar is None:
        return {"error": "Artifact Service'e ulaşılamadı"}
    return {"kayitlar": kayitlar, "workflow_id": workflow_id}


# ---------------------------------------------------------------------------
# Canlı akış — Hubble'dan
#
# Panelin diğer bölümleri "neye izin var" diyor (politikalardan okunuyor).
# Bu bölüm "ne OLDU" diyor: Cilium'un eBPF katmanının gerçekten gördüğü
# paketler. İkisi birlikte tam resmi veriyor — kural ile gerçeğin uyuşup
# uyuşmadığı ancak yan yana konunca görülüyor.
# ---------------------------------------------------------------------------

#: Hubble relay adresi. Yerelde `kubectl port-forward -n kube-system
#: svc/hubble-relay 4245:80` ile açılır; cluster içinde doğrudan servis adı.
HUBBLE_SUNUCU = os.environ.get("HUBBLE_SERVER", "localhost:4245")


def _taraf(d: dict, ip_alani: str, akis: dict) -> str:
    """Bir akışın bir ucunu okunabilir tek bir ada indirger.

    Öncelik sırası: pod adı → ayrılmış kimlik (host/world/…) → DNS adı → IP.
    Amaç, panelde `default/tool-gateway-abc123` yerine `tool-gateway` görmek.
    """
    pod = d.get("pod_name")
    if pod:
        # Deployment hash'ini at: "tool-gateway-556d5-6v2b9" -> "tool-gateway"
        ad = pod
        for is_yuku in d.get("workloads") or []:
            if is_yuku.get("name"):
                ad = is_yuku["name"]
                break
        # Sandbox'lar Job pod'u: iş yükünün ADI DA her çalıştırmada değişiyor
        # ("ptc-sandbox-975f69161007"). Bu yüzden kısaltma en sonda, iş yükü
        # adına da uygulanıyor — panelde hep tek bir "sandbox" görünsün.
        return "sandbox" if ad.startswith("ptc-sandbox") else ad
    for etiket in d.get("labels") or []:
        if etiket.startswith("reserved:"):
            return etiket.split(":", 1)[1]
    for ad in (akis.get("destination_names") or []):
        return ad
    return (akis.get("IP") or {}).get(ip_alani, "?")


def _akisi_sadelestir(akis: dict) -> dict | None:
    """Ham Hubble JSON'unu panelin göstereceği beş alana indirger.

    GÜRÜLTÜ FİLTRESİ: `reserved:host` kaynaklı akışlar atlanıyor. Bunlar
    kubelet probe'ları ve bizim kendi `port-forward`'umuz — panelin kendi
    yenilemesi akış listesini doldurup asıl trafiği görünmez yapıyordu.
    DROPPED olanlar bu filtreden MUAF: bir engelleme her zaman gösterilir.
    """
    # CEVAP PAKETLERİNİ ATLA. Hubble her akışı iki yönde de yayınlıyor;
    # "artifact-service → coredns:53" ile onun dönüşü olan
    # "coredns → artifact-service:54367" aynı olayın iki yüzü. İkisini birden
    # göstermek listeyi ikiye katlıyor ve rastgele yüksek portlar okumayı
    # zorlaştırıyor.
    if akis.get("is_reply"):
        return None

    kaynak = _taraf(akis.get("source") or {}, "source", akis)
    hedef = _taraf(akis.get("destination") or {}, "destination", akis)
    verdict = akis.get("verdict", "?")

    if kaynak in ("host", "kube-apiserver", "remote-node") and verdict != "DROPPED":
        return None
    if kaynak == hedef:
        return None

    l4 = akis.get("l4") or {}
    port = ((l4.get("TCP") or l4.get("UDP") or {}).get("destination_port"))
    return {
        "zaman": (akis.get("time") or "")[11:19],
        "kaynak": kaynak,
        "hedef": hedef,
        "port": port,
        "verdict": verdict,
        "tur": "dns" if port == 53 else ("http" if port in (80, 443, 8080, 8443, 9000) else "tcp"),
    }


#: Art arda gelen AYNI akışı bastırmak için son gönderilenin imzası.
#: Tek bir HTTP isteği bile birkaç TCP paketi üretiyor (SYN, ACK, PSH…) ve
#: hepsi ayrı akış olarak geliyor — panelde tek satır olarak görünmeleri
#: yeterli.
_SON_IMZA: dict = {}


def akis_tekrari_mi(sade: dict) -> bool:
    """Aynı saniyede aynı (kaynak, hedef, port, verdict) ikinci kez geldiyse True."""
    imza = (sade["zaman"], sade["kaynak"], sade["hedef"], sade["port"], sade["verdict"])
    if _SON_IMZA.get("v") == imza:
        return True
    _SON_IMZA["v"] = imza
    return False


# ---------------------------------------------------------------------------
# Artifact İÇERİĞİ — panelde önizleme
#
# Tablo şimdiye kadar yalnızca KÜNYE gösteriyordu (ad, tip, boyut). "Deponun
# içini görelim" isteği bunu tamamlıyor: satıra tıklayınca baytlar çekilip
# okunabilir bir önizlemeye çevriliyor.
#
# NEDEN SUNUCUDA ÇÖZÜLÜYOR: Parquet baytını tarayıcıda açmak pyarrow'un
# tarayıcı sürümünü gerektirirdi. Sunucu zaten `serialize` modülüne sahip;
# çözüp sadeleştirilmiş bir tablo/metin döndürmek hem basit hem az veri.
# ---------------------------------------------------------------------------

#: Önizleme için indirilecek azami boyut. Panel bir veri gezgini değil;
#: 100 MiB'lik bir parquet'i tarayıcıya taşımanın anlamı yok.
ONIZLEME_SINIRI = 2 * 1024 * 1024

#: DataFrame önizlemesinde gösterilecek satır sayısı.
ONIZLEME_SATIR = 20


def artifact_icerigi(workflow_id: str, artifact_id: str, jeton_uret) -> dict:
    """Bir artifact'in içeriğini okunabilir önizlemeye çevirir.

    Dönen biçim üçten biri:
      {"tablo": {"sutunlar": [...], "satirlar": [[...]], "toplam": N}}
      {"metin": "..."}
      {"bilgi": "..."}   — önizlenemeyen (çok büyük / ikili) içerik
    """
    import requests  # noqa: PLC0415

    from grounded_assistant.agent import artifact_context  # noqa: PLC0415
    from grounded_assistant.artifacts import serialize  # noqa: PLC0415

    adres = artifact_context.servis_adresi()
    if not adres:
        return {"hata": "ARTIFACT_SERVICE_URL tanımlı değil"}
    jeton = jeton_uret(workflow_id)
    if not jeton:
        return {"hata": "Kapsam jetonu üretilemedi"}

    try:
        yanit = requests.get(
            f"{adres}/artifacts/{artifact_id}",
            headers={"X-Scope-Token": jeton}, timeout=(3, 20), stream=True,
        )
        if yanit.status_code == 404:
            return {"hata": "Artifact bulunamadı (silinmiş ya da başka bir workflow'a ait)"}
        if yanit.status_code != 200:
            return {"hata": f"Servis {yanit.status_code} döndü"}

        boyut = int(yanit.headers.get("X-Artifact-Size") or 0)
        if boyut > ONIZLEME_SINIRI:
            return {"bilgi": f"{boyut} bayt — önizleme sınırı {ONIZLEME_SINIRI} bayt. "
                             "Panel bir veri gezgini değil; içeriği sandbox içinde okuyun."}
        ham = yanit.content
        content_type = yanit.headers.get("Content-Type", "")
    except Exception as exc:  # noqa: BLE001
        return {"hata": f"İçerik alınamadı: {exc}"}

    # İKİLİ İÇERİK ÖNCE (2026-09-05): PDF/PNG `deserialize`'dan ham `bytes`
    # olarak çıkıyor ve panelde "İkili içerik, N bayt — önizlenemiyor" oluyordu.
    # Oysa tarayıcı ikisini de gösterebilir; eksik olan bizim onları
    # gömülebilir bir biçime çevirmemizdi.
    ikili = _ikili_onizleme(ham, content_type)
    if ikili is not None:
        return ikili

    try:
        deger = serialize.deserialize(ham, content_type)
    except Exception as exc:  # noqa: BLE001
        # pickle reddi buraya da düşer — panelde de aynı kapı.
        return {"hata": f"Çözülemedi: {exc}"}

    return _onizleme(deger)


#: Gömülü (base64) önizleme sınırı. `ONIZLEME_SINIRI`den DAHA DAR, çünkü
#: base64 içeriği ~%33 şişiriyor ve bu JSON gövdesinde tarayıcıya gidiyor.
#: Aşan dosya için künye gösteriliyor — panel bir belge görüntüleyici değil.
GOMME_SINIRI = 1024 * 1024


def _ikili_onizleme(ham: bytes, content_type: str) -> dict | None:
    """PDF ve görselleri tarayıcının gösterebileceği hâle getirir.

    İlgilenmediğimiz bir tip için None döner — çağıran normal yola devam eder.
    """
    tur = (content_type or "").split(";")[0].strip().lower()
    if tur != "application/pdf" and not tur.startswith("image/"):
        return None
    if len(ham) > GOMME_SINIRI:
        return {"bilgi": f"{len(ham)} bayt {tur} — gömme sınırı {GOMME_SINIRI} bayt. "
                         "Dosya depoda duruyor; içeriğini sandbox içinde okuyun."}

    import base64  # noqa: PLC0415

    veri = f"data:{tur};base64," + base64.b64encode(ham).decode("ascii")
    if tur == "application/pdf":
        # Sayfa sayısı: PDF nesne sözlüklerindeki `/Type /Page` sayımı. Kesin
        # bir ayrıştırma DEĞİL (sıkıştırılmış nesne akışlarını göremez), o
        # yüzden bulunamazsa hiç göstermiyoruz — yanlış sayı, sayı yokluğundan
        # kötüdür.
        import re  # noqa: PLC0415

        sayfa = len(re.findall(rb"/Type\s*/Page[^s]", ham))
        return {"pdf": veri, "sayfa": sayfa or None, "bayt": len(ham)}
    return {"gorsel": veri, "bayt": len(ham)}


def soy_agaci(workflow_id: str, artifact_id: str, jeton_uret) -> dict:
    """Bir artifact'in soy ağacı — Artifact Service'ten olduğu gibi geçirilir.

    Panel neden aracı: tarayıcının kapsam jetonu yok ve OLMAMALI. Jeton
    imzalama anahtarı sunucu tarafında; UI, kimliği doğrulanmış oturumu
    gönderiyor, jetonu biz üretiyoruz. Artifact önizlemesinde olan düzenin
    aynısı.
    """
    import requests  # noqa: PLC0415

    from grounded_assistant.agent import artifact_context  # noqa: PLC0415

    adres = artifact_context.servis_adresi()
    if not adres:
        return {"hata": "ARTIFACT_SERVICE_URL tanımlı değil"}
    jeton = jeton_uret(workflow_id)
    if not jeton:
        return {"hata": "Kapsam jetonu üretilemedi"}
    try:
        yanit = requests.get(
            f"{adres}/artifacts/{artifact_id}/lineage",
            headers={"X-Scope-Token": jeton}, timeout=(3, 10),
        )
    except Exception as exc:  # noqa: BLE001
        return {"hata": f"Soy ağacı alınamadı: {exc}"}
    if yanit.status_code == 404:
        return {"hata": "Artifact bulunamadı"}
    if yanit.status_code != 200:
        return {"hata": f"Servis {yanit.status_code} döndü"}
    return yanit.json()


def _onizleme(deger) -> dict:
    """Çözülmüş bir değeri panelin gösterebileceği biçime indirger."""
    # DataFrame — tip adına bakarak (pandas'ı burada import etmemek için).
    if hasattr(deger, "columns") and hasattr(deger, "iloc"):
        kirpik = deger.head(ONIZLEME_SATIR)
        return {"tablo": {
            "sutunlar": [str(c) for c in kirpik.columns],
            "satirlar": [[("" if v is None else str(v)) for v in satir]
                         for satir in kirpik.astype(object).values.tolist()],
            "toplam": int(len(deger)),
            "gosterilen": int(len(kirpik)),
        }}
    if isinstance(deger, (dict, list)):
        import json  # noqa: PLC0415

        return {"metin": json.dumps(deger, ensure_ascii=False, indent=2)[:8000]}
    if isinstance(deger, str):
        return {"metin": deger[:8000] + ("\n…" if len(deger) > 8000 else "")}
    if isinstance(deger, bytes):
        return {"bilgi": f"İkili içerik, {len(deger)} bayt — önizlenemiyor."}
    return {"metin": str(deger)[:8000]}
