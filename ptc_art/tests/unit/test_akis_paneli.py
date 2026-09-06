"""Hubble akış ayrıştırması (2026-09-04).

Panelin sağ sütunu Cilium/eBPF'in GERÇEKTEN gördüğü paketleri gösteriyor.
Ham Hubble JSON'u okunabilir tek satıra indirgemek üç şey gerektiriyor:
isimleri sadeleştirmek, cevap paketlerini atmak, gürültüyü süzmek.
"""

from __future__ import annotations

from grounded_assistant.web import durum


def _akis(kaynak_pod=None, kaynak_etiket=None, hedef_pod=None, hedef_etiket=None,
          port=8080, verdict="FORWARDED", cevap=False, is_yuku=None):
    def taraf(pod, etiket, iy):
        d = {}
        if pod:
            d["pod_name"] = pod
            d["workloads"] = [{"name": iy}] if iy else []
        if etiket:
            d["labels"] = [f"reserved:{etiket}"]
        return d
    return {
        "time": "2026-09-04T14:58:13.123456Z",
        "verdict": verdict,
        "is_reply": cevap,
        "source": taraf(kaynak_pod, kaynak_etiket, is_yuku),
        "destination": taraf(hedef_pod, hedef_etiket, None),
        "l4": {"TCP": {"destination_port": port}},
    }


def test_deployment_hashi_atiliyor():
    """Panelde `tool-gateway-556d5-6v2b9` değil `tool-gateway` görünmeli."""
    s = durum._akisi_sadelestir(
        _akis(kaynak_pod="tool-gateway-556d5-6v2b9", is_yuku="tool-gateway",
              hedef_etiket="world", port=443)
    )
    assert s["kaynak"] == "tool-gateway"


def test_sandbox_adi_sabitleniyor():
    """Sandbox'lar Job pod'u: adları HER çalıştırmada değişiyor. İş yükünün adı
    da değiştiği için kısaltma en sonda uygulanmalı — yoksa panelde her koşuda
    yeni bir kaynak belirir."""
    s = durum._akisi_sadelestir(
        _akis(kaynak_pod="ptc-sandbox-975f69161007",
              is_yuku="ptc-sandbox-975f69161007", hedef_etiket="world", port=443)
    )
    assert s["kaynak"] == "sandbox"


def test_cevap_paketleri_atlaniyor():
    """Hubble her akışı iki yönde yayınlıyor; dönüş yönü listeyi ikiye
    katlıyor ve rastgele yüksek portlarla okumayı zorlaştırıyordu."""
    assert durum._akisi_sadelestir(_akis(kaynak_pod="a", hedef_pod="b", cevap=True)) is None


def test_host_gurultusu_suzuluyor():
    """`host → artifact-service` panelin KENDİ yenileme istekleri (port-forward).
    Süzülmezse gerçek trafiği bastırıyorlar."""
    assert durum._akisi_sadelestir(
        _akis(kaynak_etiket="host", hedef_pod="artifact-service", is_yuku=None)
    ) is None


def test_ENGELLENEN_akis_host_olsa_bile_gosterilir():
    """Bir engelleme her zaman gösterilir — gürültü filtresinden MUAF.
    Panelin varlık sebebi bu satırlar."""
    s = durum._akisi_sadelestir(
        _akis(kaynak_etiket="host", hedef_etiket="world", verdict="DROPPED", port=443)
    )
    assert s is not None and s["verdict"] == "DROPPED"


def test_kendine_giden_akis_atlaniyor():
    assert durum._akisi_sadelestir(_akis(kaynak_pod="a", is_yuku="x", hedef_pod="b_", hedef_etiket=None)) is not None
    ayni = _akis(kaynak_etiket="host", hedef_etiket="host", verdict="DROPPED")
    assert durum._akisi_sadelestir(ayni) is None


def test_port_turu_etiketleniyor():
    for port, tur in [(53, "dns"), (443, "http"), (9000, "http"), (12345, "tcp")]:
        s = durum._akisi_sadelestir(
            _akis(kaynak_pod="a", is_yuku="a", hedef_etiket="world", port=port)
        )
        assert s["tur"] == tur, port


def test_ardisik_ayni_akis_tekrari_bastiriliyor():
    """Tek bir HTTP isteği bile birkaç TCP paketi üretiyor (SYN/ACK/PSH);
    panelde tek satır olarak görünmeleri yeterli."""
    s = durum._akisi_sadelestir(_akis(kaynak_pod="a", is_yuku="a", hedef_etiket="world"))
    assert durum.akis_tekrari_mi(s) is False   # ilk görülüş
    assert durum.akis_tekrari_mi(s) is True    # aynısı hemen ardından


def test_zaman_saat_dakika_saniye():
    s = durum._akisi_sadelestir(_akis(kaynak_pod="a", is_yuku="a", hedef_etiket="world"))
    assert s["zaman"] == "14:58:13"


# -- artifact içerik önizlemesi ---------------------------------------------


def test_dataframe_tabloya_cevriliyor():
    import pandas as pd

    df = pd.DataFrame({"ay": ["oca", "sub"], "tutar": [100, 250]})
    o = durum._onizleme(df)
    assert o["tablo"]["sutunlar"] == ["ay", "tutar"]
    assert o["tablo"]["satirlar"][0] == ["oca", "100"]
    assert o["tablo"]["toplam"] == 2


def test_uzun_dataframe_kirpiliyor():
    """Panel bir veri gezgini değil — 10.000 satırı tarayıcıya taşımaz."""
    import pandas as pd

    o = durum._onizleme(pd.DataFrame({"x": range(500)}))
    assert o["tablo"]["gosterilen"] == durum.ONIZLEME_SATIR
    assert o["tablo"]["toplam"] == 500


def test_sozluk_okunabilir_json_oluyor():
    o = durum._onizleme({"toplam": 525.0, "ortalama": 175.0})
    assert '"toplam"' in o["metin"] and "\n" in o["metin"]


def test_metin_kirpiliyor():
    o = durum._onizleme("a" * 20000)
    assert len(o["metin"]) < 20000 and o["metin"].endswith("…")


def test_ikili_icerik_onizlenmiyor():
    o = durum._onizleme(b"\x00\x01\x02")
    assert "bilgi" in o and "önizlenemiyor" in o["bilgi"]


def test_oturumsuz_istek_hata_veriyor():
    assert "hata" in durum.artifact_icerigi("", "art_x", lambda w: None)
