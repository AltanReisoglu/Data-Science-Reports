"""Konsol — dört sekmeli tek panel, hepsi CANLI (2026-09-07).

Bu modül `durum.py`'nin yaptığı işi (pod'lar, akışlar, artifact künyeleri)
GENİŞLETİYOR, yerine geçmiyor: buraya eklenen tek yeni yetenek **gerçek bir
pipeline çalıştırmak** ve her adımını akıtmak.

## Neden pipeline var, ajan değil

Panelin sohbet tarafı (`/ws`) ajanı çalıştırıyor; ne yazacağını LLM
belirliyor, dolayısıyla adımlar önceden bilinmiyor. Bir mimari gösterimi için
adımların ÖNCEDEN BELLİ olması gerekiyor — Argo ve KFP'de de DAG yazılıdır,
çalışma anında keşfedilmez.

Burada tanımlı iki pipeline **gerçek** çalışıyor: her sandbox adımı gerçek bir
Kubernetes Job açıyor, gerçek log basıyor, çıktısı gerçekten MinIO'ya iniyor
ve kayıt defterine satır düşüyor. Simülasyon yok.

## ÜÇ node TÜRÜ var ve fark gerçek

    sandbox : gerçek PTC pod'u — kimlik bilgisi yok, ağı kapalı, süpürülüyor
    query   : host tarafında kayıt defteri sorgusu — pod açılmıyor
    alias   : host tarafında sürüm sabitleme — pod açılmıyor

Bu ayrım uydurma değil, mimarinin kendisi: **keşif de sabitleme de sandbox'ta
olmuyor.** Sandbox'ın listeleme yolu hiç yok; hangi artifact'in var olduğunu
host tarafındaki kayıt defteri sorgusu söylüyor (§11.14 süzgeç). Alias'ı da
insan/CI atıyor — MLflow'da da öyle. Panelde üç tür farklı çiziliyor ki
izleyici nerede pod açıldığını görsün.

## Çapraz workflow

B pipeline'ı A'nın çıktısını `inputs=["<A'nın workflow_id'si>/…"]` BEYANIYLA
okuyor — kod bir çağrı yapmıyor, dosya kod başlamadan yerinde. A'nın kimliğini
B'ye veren şey node 1'in kayıt defteri sorgusu; elle gömülmüş bir kimlik değil.

## Hatlar nereden geliyor (2026-09-07)

Dört hat koda gömülü (`_YERLESIK`), kullanıcının konsoldan kurdukları ise
`var/konsol-hatlari.json`'da duruyor. İkisi aynı sözleşmeyi konuşuyor:
`pipeline_calistir` hangisinden geldiğine bakmıyor. Yerleşikler silinemiyor —
gösterimin zemini onlar.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path

import requests

#: Konsolun kayıt defterine konuşurken kullandığı sözde-workflow. Listeleme
#: tenant genelinde olduğu için jetonun hangi workflow'a ait olduğu önemsiz;
#: `owner` alanı belirleyici.
KONSOL_WF = "konsol-panel"

_ZAMAN_ASIMI = (3, 20)


def _adres() -> str:
    from grounded_assistant.agent import artifact_context  # noqa: PLC0415

    return artifact_context.servis_adresi()


def _baslik(jeton_uret) -> dict[str, str] | None:
    jeton = jeton_uret(KONSOL_WF)
    return {"X-Scope-Token": jeton} if jeton else None


# ── kayıt defteri: süzgeçli listeleme (MLMD filter_query karşılığı) ────────


def depo(jeton_uret, **suzgec) -> dict:
    """`GET /artifacts` — süzgeçler doğrudan servise geçiyor.

    Panelin deposu ile ajanın gördüğü depo AYNI uç nokta; panel ayrıcalıklı
    bir yol kullanmıyor.
    """
    adres = _adres()
    if not adres:
        return {"error": "ARTIFACT_SERVICE_URL tanımlı değil "
                         "(kubectl port-forward svc/artifact-service 8080:8080)"}
    basliklar = _baslik(jeton_uret)
    if not basliklar:
        return {"error": "Kapsam jetonu üretilemedi (ptc-scope-signing okunamadı)"}
    params = {k: v for k, v in suzgec.items() if v}
    try:
        yanit = requests.get(f"{adres}/artifacts", params=params,
                             headers=basliklar, timeout=_ZAMAN_ASIMI)
        yanit.raise_for_status()
        return {"kayitlar": yanit.json(), "sorgu": params,
                "url": yanit.url.replace(adres, "")}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Artifact Service'e ulaşılamadı: {exc}"}


def soy(jeton_uret, artifact_id: str) -> dict:
    adres, basliklar = _adres(), _baslik(jeton_uret)
    if not adres or not basliklar:
        return {"error": "Artifact Service yapılandırılmamış"}
    try:
        yanit = requests.get(f"{adres}/artifacts/{artifact_id}/lineage",
                             headers=basliklar, timeout=_ZAMAN_ASIMI)
        if yanit.status_code == 404:
            return {"error": "Artifact bulunamadı"}
        yanit.raise_for_status()
        return yanit.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def alias_ata(jeton_uret, artifact_id: str, alias: str | None) -> dict:
    """MLflow'un `models:/<ad>@<alias>`'ının bizdeki karşılığı (§11.14).

    Panelden atanabiliyor çünkü alias'ı taşıyan taraf İNSAN ya da CI —
    sandbox'ın böyle bir yolu yok ve olmamalı.
    """
    adres, basliklar = _adres(), _baslik(jeton_uret)
    if not adres or not basliklar:
        return {"error": "Artifact Service yapılandırılmamış"}
    try:
        yanit = requests.put(f"{adres}/artifacts/{artifact_id}/alias",
                             params={"alias": alias} if alias else {},
                             headers=basliklar, timeout=_ZAMAN_ASIMI)
        if yanit.status_code >= 400:
            return {"error": yanit.json().get("detail", yanit.text[:160]),
                    "status": yanit.status_code}
        return yanit.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ── pipeline tanımları — kod GERÇEKTEN sandbox'ta çalışıyor ───────────────

_KOD_TOPLA = '''
import pandas as pd
satirlar = []
for i in range(60):
    d = get_ticket_status(f"T-{1000+i}")
    satirlar.append({
        "ticket": f"T-{1000+i}",
        "departman": ["BT", "IK", "Finans", "Satis"][i % 4],
        "gun": (i % 14) + 1,
        "durum": d.get("status", "open") if isinstance(d, dict) else "open",
    })
df = pd.DataFrame(satirlar)
df.to_parquet("/output/ham.tickets.parquet")
set_result({"satir": len(df), "sutun": list(df.columns)})
'''

_KOD_AYIKLA = '''
import json, pandas as pd
df = pd.read_parquet("/output/ham.tickets.parquet")
ozet = {
    "toplam": int(len(df)),
    "departmanlar": {k: int(v) for k, v in df.departman.value_counts().items()},
    "ortalama_gun": round(float(df.gun.mean()), 2),
    "acik": int((df.durum == "open").sum()),
}
json.dump(ozet, open("/output/extracted-content.json", "w"))
set_result(ozet)
'''

_KOD_ISLE = '''
import json, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ozet = json.load(open("/output/extracted-content.json"))
dep = ozet["departmanlar"]

fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(list(dep), list(dep.values()), color="#E0A24B")
ax.set_title("Ticket dagilimi"); fig.tight_layout()
fig.savefig("/output/dagilim.png", dpi=110)

turev = {
    "en_yuklu": max(dep, key=dep.get),
    "yuk_orani": round(max(dep.values()) / ozet["toplam"], 3),
    "ortalama_gun": ozet["ortalama_gun"],
    "acik_orani": round(ozet["acik"] / ozet["toplam"], 3),
}
json.dump(turev, open("/output/processed-result.json", "w"))
set_result(turev)
'''

_KOD_RAPOR = '''
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t = json.load(open("/output/processed-result.json"))
fig, ax = plt.subplots(figsize=(8.3, 5.8)); ax.axis("off")
ax.text(.05, .92, "Ticket Raporu", fontsize=20, weight="bold")
for i, (k, v) in enumerate(t.items()):
    ax.text(.05, .78 - i * .09, f"{k}", fontsize=11, color="#555")
    ax.text(.45, .78 - i * .09, f"{v}", fontsize=13, weight="bold")
with PdfPages("/output/final-report.pdf") as pdf:
    pdf.savefig(fig)
set_result({"rapor": "final-report.pdf", "alanlar": list(t)})
'''

# Çapraz-workflow girdi artık BEYAN (2026-09-07): kod bir çağrı yapmıyor,
# dosya zaten `/artifacts/<wf>/` altında hazır. Beyanı keşif adımının bulduğu
# kimlikle runtime'da kuruyoruz — KFP'de driver'ın `.uri`yi çözüp launcher'a
# vermesinin birebir karşılığı.
_KOD_YUKLE = '''
import json
yol = "/artifacts/{kaynak_wf}/processed-result.json"
veri = json.load(open(yol))
json.dump(veri, open("/output/analysis-input.json", "w"))
set_result({"kaynak_yol": yol, "alanlar": list(veri)})
'''

_KOD_ANALIZ = '''
import json
v = json.load(open("/output/analysis-input.json"))
skor = round(v["yuk_orani"] * 0.6 + v["acik_orani"] * 0.4, 3)
set_result({"denge_skoru": skor, "en_yuklu": v["en_yuklu"],
            "yorum": "dengesiz" if skor > 0.35 else "dengeli"})
'''

_KOD_ICGORU = '''
import json
v = json.load(open("/output/analysis-input.json"))
skor = round(v["yuk_orani"] * 0.6 + v["acik_orani"] * 0.4, 3)
json.dump({"denge_skoru": skor, "en_yuklu": v["en_yuklu"],
           "oneri": "Yuk dagitimi gozden gecirilmeli" if skor > 0.35
                    else "Dagilim kabul edilebilir"},
          open("/output/analysis-result.json", "w"))
set_result({"denge_skoru": skor})
'''


_KOD_SABIT_OKU = '''
import json
# Beyan `processed-result.json@konsol-sabit` idi; alias'la gelen dosya
# `/artifacts/_alias/` altına konuyor. Kod yine hiçbir çağrı yapmıyor.
yol = "/artifacts/_alias/processed-result.json"
v = json.load(open(yol))
json.dump({"kaynak": yol, "veri": v}, open("/output/sabit-surum.json", "w"))
set_result({"okunan": yol, "alanlar": list(v)})
'''

_KOD_SABIT_RAPOR = '''
import json
v = json.load(open("/output/sabit-surum.json"))["veri"]
json.dump({"en_yuklu": v["en_yuklu"], "yuk_orani": v["yuk_orani"],
           "not": "alias ile sabitlenmis surumden okundu"},
          open("/output/sabit-rapor.json", "w"))
set_result({"en_yuklu": v["en_yuklu"], "yuk_orani": v["yuk_orani"]})
'''

_KOD_DIZIN_URET = '''
import json, os
os.makedirs("/output/model.v1", exist_ok=True)
for i, ad in enumerate(["agirliklar.json", "olcumler.json", "NOTLAR.md"]):
    yol = "/output/model.v1/" + ad
    if ad.endswith(".json"):
        json.dump({"katman": i, "deger": round(0.1 * (i + 1), 3)}, open(yol, "w"))
    else:
        open(yol, "w").write("# model v1\\nkonsoldan uretildi\\n")
set_result({"dizin": "model.v1", "dosya": sorted(os.listdir("/output/model.v1"))})
'''

_KOD_DIZIN_OKU = '''
import json, os
# Beyan `model.v1` idi. Depoda `model.v1.tar` duruyor ama sidecar açıp
# gerçek bir DİZİN bırakıyor — kodun tar diye bir şeyden haberi yok.
kok = "/output/model.v1"
dosyalar = sorted(os.listdir(kok))
toplam = sum(os.path.getsize(os.path.join(kok, d)) for d in dosyalar)
json.dump({"dosya": dosyalar, "bayt": toplam}, open("/output/dizin-ozeti.json", "w"))
set_result({"gorulen_dosya": dosyalar, "toplam_bayt": toplam})
'''

_KOD_DEDUP = '''
import hashlib, json
icerik = json.dumps({"olcum": [1, 2, 3], "kaynak": "konsol"}, sort_keys=True)
for ad in ["olcum.a.json", "olcum.b.json"]:
    open("/output/" + ad, "w").write(icerik)
set_result({"iki_ad": ["olcum.a.json", "olcum.b.json"],
            "sha256": hashlib.sha256(icerik.encode()).hexdigest()[:16]})
'''


def pipeline_a() -> dict:
    return {
        "key": "a", "kod": "PL-A", "ad": "Ticket İşleme Hattı",
        "aciklama": "Canlı tool'lardan veri toplar, ayıklar, PTC sandbox'ında "
                    "türetir ve PDF rapora indirger. Dört adımın dördü de ayrı "
                    "bir sandbox pod'unda çalışır.",
        "nodes": [
            {"n": 1, "ad": "Veri Topla", "tur": "sandbox", "ikon": "i-fileplus",
             "aciklama": "Tool Gateway üzerinden canlı ticket verisi çeker ve "
                         "Parquet olarak /output'a yazar.",
             "kod": _KOD_TOPLA, "inputs": [], "bekleniyor": ["ham.tickets.parquet"]},
            {"n": 2, "ad": "İçerik Ayıkla", "tur": "sandbox", "ikon": "i-scan",
             "aciklama": "Parquet'i okur, departman dağılımı ve ortalamaları "
                         "çıkarır. Girdi BEYAN edildiği için kod başlamadan yerinde.",
             "kod": _KOD_AYIKLA, "inputs": ["ham.tickets.parquet"],
             "bekleniyor": ["extracted-content.json"]},
            {"n": 3, "ad": "PTC Türetme", "tur": "sandbox", "ikon": "i-cpu",
             "aciklama": "Türev metrikleri hesaplar ve grafik üretir. Sandbox'ın "
                         "S3 anahtarı yok, ağı kapalı; baytları sidecar taşıyor.",
             "kod": _KOD_ISLE, "inputs": ["extracted-content.json"],
             "bekleniyor": ["processed-result.json", "dagilim.png"]},
            {"n": 4, "ad": "Rapor Üret", "tur": "sandbox", "ikon": "i-pkg",
             "aciklama": "Türev veriyi tek sayfalık bir PDF rapora indirger.",
             "kod": _KOD_RAPOR, "inputs": ["processed-result.json"],
             "bekleniyor": ["final-report.pdf"]},
        ],
    }


def pipeline_b() -> dict:
    return {
        "key": "b", "kod": "PL-B", "ad": "Artifact Analiz Hattı",
        "aciklama": "BAŞKA bir çalıştırmanın çıktısını kayıt defterinden bulur, "
                    "kendi alanına yükler, analiz eder ve bulgusunu aynı depoya "
                    "geri koyar.",
        "nodes": [
            {"n": 1, "ad": "Artifact Keşfet", "tur": "query", "ikon": "i-search",
             "aciklama": "Kayıt defterine ada göre sorgu atar. Bu adım POD AÇMAZ — "
                         "keşif sandbox'ta değil, host tarafında olur. Bu hat "
                         "belirli bir alias İSTEMİYOR, dolayısıyla EN YENİ "
                         "kazanıyor; sabitlenmiş sürüm varsa log bunu söylüyor. "
                         "Sabitlemenin kendisi PL-C'de.",
             "sorgu": {"name": "processed-result.json"}, "inputs": [],
             "bekleniyor": []},
            {"n": 2, "ad": "Artifact Yükle", "tur": "sandbox", "ikon": "i-down",
             "aciklama": "Bulunan artifact'i BEYAN ederek alır. Kod hiçbir çağrı "
                         "yapmıyor — dosya /artifacts/<wf>/ altında hazır geliyor.",
             "kod": _KOD_YUKLE, "inputs": ["{kaynak_wf}/processed-result.json"],
             "bekleniyor": ["analysis-input.json"]},
            {"n": 3, "ad": "Analiz Et", "tur": "sandbox", "ikon": "i-chart",
             "aciklama": "Yerel kopyayı okur, denge skorunu hesaplar. Bu adım "
                         "artifact ÜRETMEZ — her adımın üretmesi gerekmiyor.",
             "kod": _KOD_ANALIZ, "inputs": ["analysis-input.json"], "bekleniyor": []},
            {"n": 4, "ad": "Bulgu Yayınla", "tur": "sandbox", "ikon": "i-spark",
             "aciklama": "Bulguyu aynı paylaşılan depoya yazar. Depo açısından "
                         "A'nın çıktılarından farkı yok.",
             "kod": _KOD_ICGORU, "inputs": ["analysis-input.json"],
             "bekleniyor": ["analysis-result.json"]},
        ],
    }


def pipeline_c() -> dict:
    return {
        "key": "c", "kod": "PL-C", "ad": "Sürüm Sabitleme Hattı",
        "aciklama": "Aynı ad depoda onlarca sürümle duruyor. Bu hat EN ESKİ "
                    "sürümü alias'la sabitliyor ve sonra o alias'ı BEYAN edip "
                    "okuyor — 'en yeni kazanır' kuralını bilerek deviriyor.",
        "nodes": [
            {"n": 1, "ad": "Adayları Say", "tur": "query", "ikon": "i-search",
             "aciklama": "Kayıt defterine sorar: bu adda kaç sürüm var? Pod "
                         "AÇMAZ. Sayı büyüdükçe 'hangisini alıyorum' sorusu "
                         "keskinleşiyor — hattın derdi bu.",
             "sorgu": {"name": "processed-result.json"}, "inputs": [],
             "bekleniyor": []},
            {"n": 2, "ad": "En Eskiyi Sabitle", "tur": "alias", "ikon": "i-pin",
             "aciklama": "EN ESKİ adaya `@konsol-sabit` alias'ı atar. Pod "
                         "AÇMAZ — alias'ı insan/CI koyar, MLflow'da da öyle. "
                         "En eskiyi seçmesi kasıtlı: 'en yeni' kuralı geçerli "
                         "olsaydı bu sürüm asla seçilmezdi.",
             "sorgu": {"name": "processed-result.json"},
             "alias": "konsol-sabit", "sec": "en_eski",
             "inputs": [], "bekleniyor": []},
            {"n": 3, "ad": "Sabit Sürümü Oku", "tur": "sandbox", "ikon": "i-down",
             "aciklama": "Beyanın üçüncü biçimi: `ad@alias`. Dosya "
                         "/artifacts/_alias/ altında hazır geliyor; kod yine "
                         "hiçbir çağrı yapmıyor.",
             "kod": _KOD_SABIT_OKU,
             "inputs": ["processed-result.json@konsol-sabit"],
             "bekleniyor": ["sabit-surum.json"]},
            {"n": 4, "ad": "Sabit Rapor", "tur": "sandbox", "ikon": "i-pkg",
             "aciklama": "Sabitlenmiş sürümden okunanı ayrı bir künyeye yazar.",
             "kod": _KOD_SABIT_RAPOR, "inputs": ["sabit-surum.json"],
             "bekleniyor": ["sabit-rapor.json"]},
        ],
    }


def pipeline_d() -> dict:
    return {
        "key": "d", "kod": "PL-D", "ad": "Dizin ve Dedup Hattı",
        "aciklama": "Artifact her zaman tek dosya değil: bir DİZİN üretip "
                    "beyanla geri alıyor. Son adım aynı içeriği iki ada "
                    "yazarak içerik-hash dedup'ını görünür kılıyor.",
        "nodes": [
            {"n": 1, "ad": "Dizin Üret", "tur": "sandbox", "ikon": "i-fileplus",
             "aciklama": "/output/model.v1/ altına üç dosya yazar. Süpürme "
                         "dizini yeniden-üretilebilir bir tar'a paketliyor "
                         "(mtime/uid/gid/mode sıfırlanmış) — depoda "
                         "`model.v1.tar` olarak duruyor.",
             "kod": _KOD_DIZIN_URET, "inputs": [],
             "bekleniyor": ["model.v1.tar"]},
            {"n": 2, "ad": "Dizini Geri Al", "tur": "sandbox", "ikon": "i-down",
             "aciklama": "`model.v1` beyan edilir. Sidecar tar'ı indirip AÇAR "
                         "ve tar'ı siler; kod gerçek bir dizin görür. Süpürme "
                         "aynı hash'i bulup 'bunu ben verdim' der, tekrar "
                         "yüklemez.",
             "kod": _KOD_DIZIN_OKU, "inputs": ["model.v1"],
             "bekleniyor": ["dizin-ozeti.json"]},
            {"n": 3, "ad": "Aynı İçerik İki Ad", "tur": "sandbox", "ikon": "i-copy",
             "aciklama": "Birebir aynı baytı iki ayrı adla yazar. Kayıt "
                         "defterinde İKİ künye oluşur, MinIO'da TEK nesne — "
                         "içerik-hash dedup'ı. Depo sekmesinde iki satırın "
                         "content_hash'i aynı görünür.",
             "kod": _KOD_DEDUP, "inputs": [],
             "bekleniyor": ["olcum.a.json", "olcum.b.json"]},
        ],
    }


#: Koda gömülü hatlar — silinemez, düzenlenemez. Gösterimin zemini bunlar.
_YERLESIK = (pipeline_a, pipeline_b, pipeline_c, pipeline_d)
_YERLESIK_ANAHTAR = frozenset({"a", "b", "c", "d"})


# ── kullanıcının kurduğu hatlar ───────────────────────────────────────────
#
# Konsoldan node ekleyerek kurulan hatlar burada. Yerleşiklerle AYNI
# sözleşmeyi konuşuyorlar; `pipeline_calistir` ikisini ayırt etmiyor.

_HATLAR_DOSYASI = Path(
    os.environ.get("PTC_KONSOL_HATLARI")
    or Path(__file__).resolve().parents[3] / "var" / "konsol-hatlari.json")

_ANAHTAR_BICIMI = re.compile(r"^[a-z0-9][a-z0-9-]{0,23}$")
#: Görünen adlar TÜRKÇE — `[A-Za-z0-9]` "Üret"i reddediyordu. `[^\W_]`
#: unicode-duyarlı bir harf/rakam sınıfı (alt çizgi hariç).
_AD_BICIMI = re.compile(r"^[^\W_][\w .,'\-()/]{0,59}$", re.UNICODE)
#: Beyan biçimleri: `ad`, `<wf>/ad`, `ad@alias` — sidecar'ın `_beyani_coz`'ü.
#: `{kaynak_wf}` de geçerli bir workflow segmenti: keşif adımının bulduğu
#: kimlik çalışma anında yerine konuyor (PL-B'nin yaptığı iş). Kurucudan
#: kurulan hatlar da çapraz-workflow okuyabilsin diye açık.
_BEYAN_BICIMI = re.compile(
    r"^(\{kaynak_wf\}|[\w.\-]+)?/?[\w.\-]+(@[A-Za-z0-9][\w.\-]{0,63})?$")
#: Servisin `_ALIAS_BICIMI`'yle aynı — alias adı orada da böyle denetleniyor.
_TAKMA_BICIMI = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: Çalışma anında keşif adımının bulduğu workflow kimliğiyle DEĞİŞTİRİLİYOR.
#: Bir zamanlar `str.format` ile yapılıyordu; kullanıcının yazdığı kodda dict
#: ya da f-string süslü parantezi olunca `.format` onları biçim alanı sanıp
#: patlıyordu. Düz metin ikamesi hem güvenli hem de kaçış gerektirmiyor.
_YER_TUTUCU = "{kaynak_wf}"
_AZAMI_NODE = 8
_AZAMI_KOD = 20_000


class HatGecersiz(ValueError):
    """Kullanıcının gönderdiği hat tanımı sözleşmeye uymuyor."""


def _kullanici_hatlari() -> list[dict]:
    try:
        veri = json.loads(_HATLAR_DOSYASI.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        # Bozuk dosya konsolu kapatmasın: yerleşikler yine çalışsın.
        return []
    return veri if isinstance(veri, list) else []


def _kullanici_hatlarini_yaz(hatlar: list[dict]) -> None:
    """Atomik yazma — yarıda kesilen bir yazma dosyayı bozmasın."""
    _HATLAR_DOSYASI.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=_HATLAR_DOSYASI.parent,
            prefix=".hatlar-", suffix=".tmp", delete=False) as f:
        json.dump(hatlar, f, ensure_ascii=False, indent=2)
        gecici = f.name
    os.replace(gecici, _HATLAR_DOSYASI)


def _liste(deger, alan: str) -> list[str]:
    if deger in (None, ""):
        return []
    if not isinstance(deger, list):
        raise HatGecersiz(f"{alan} bir liste olmalı.")
    return [str(x).strip() for x in deger if str(x).strip()]


def hat_dogrula(ham: dict) -> dict:
    """Kullanıcının gönderdiği tanımı sözleşmeye oturtur — ya da reddeder.

    Kod ALANI serbest: sandbox'ta zaten güvenilmeyen kod çalışıyor, kısıtlama
    oraya ait (izolasyon, ağ politikası, süpürme). Burada denetlenen şey
    tanımın BİÇİMİ — panelin ve çalıştırıcının varsaydığı alanlar.
    """
    if not isinstance(ham, dict):
        raise HatGecersiz("Hat tanımı bir nesne olmalı.")

    key = str(ham.get("key") or "").strip().lower()
    if not _ANAHTAR_BICIMI.match(key):
        raise HatGecersiz("key biçimi: küçük harf/rakam/tire, en çok 24 karakter.")
    if key in _YERLESIK_ANAHTAR:
        raise HatGecersiz(f"'{key}' yerleşik bir hat — üzerine yazılamaz.")

    ad = str(ham.get("ad") or "").strip()
    if not _AD_BICIMI.match(ad):
        raise HatGecersiz("ad: 1-60 karakter, harfle/rakamla başlamalı.")

    ham_nodes = ham.get("nodes")
    if not isinstance(ham_nodes, list) or not ham_nodes:
        raise HatGecersiz("En az bir adım gerekli.")
    if len(ham_nodes) > _AZAMI_NODE:
        raise HatGecersiz(f"En çok {_AZAMI_NODE} adım.")

    nodes = []
    for i, hn in enumerate(ham_nodes, start=1):
        if not isinstance(hn, dict):
            raise HatGecersiz(f"{i}. adım bir nesne olmalı.")
        n_ad = str(hn.get("ad") or "").strip()
        if not _AD_BICIMI.match(n_ad):
            raise HatGecersiz(f"{i}. adımın adı geçersiz.")
        tur = str(hn.get("tur") or "sandbox").strip()
        if tur not in {"sandbox", "query", "alias"}:
            raise HatGecersiz(f"{i}. adımın türü: sandbox | query | alias.")

        nd = {"n": i, "ad": n_ad, "tur": tur,
              "ikon": {"sandbox": "i-cpu", "query": "i-search",
                       "alias": "i-pin"}[tur],
              "aciklama": str(hn.get("aciklama") or "").strip()[:400],
              "inputs": _liste(hn.get("inputs"), f"{i}. adımın inputs"),
              "bekleniyor": _liste(hn.get("bekleniyor"), f"{i}. adımın bekleniyor")}

        for beyan in nd["inputs"]:
            if not _BEYAN_BICIMI.match(beyan):
                raise HatGecersiz(
                    f"{i}. adımda geçersiz beyan: '{beyan}'. "
                    "Biçimler: ad · <workflow_id>/ad · ad@alias")

        if tur == "sandbox":
            kod = str(hn.get("kod") or "").strip()
            if not kod:
                raise HatGecersiz(f"{i}. adımın kodu boş olamaz.")
            if len(kod) > _AZAMI_KOD:
                raise HatGecersiz(f"{i}. adımın kodu {_AZAMI_KOD} karakteri aşıyor.")
            nd["kod"] = kod
        else:
            # İki biçim de kabul: formun düz `sorgu_ad`'ı ve `pipelines()`in
            # döndürdüğü normalleşmiş `sorgu: {"name": …}`. UI kaydedilmiş bir
            # hattı düzenlemek için GET'ten geleni geri POST'luyor — yalnızca
            # `sorgu_ad` kabul edilseydi kendi çıktımızı reddederdik.
            mevcut = hn.get("sorgu") if isinstance(hn.get("sorgu"), dict) else {}
            sorgu_ad = str(hn.get("sorgu_ad") or mevcut.get("name") or "").strip()
            if not sorgu_ad:
                raise HatGecersiz(f"{i}. adım ({tur}) için aranacak ad gerekli.")
            nd["sorgu"] = {"name": sorgu_ad}
            if tur == "query":
                # İSTEĞE BAĞLI: belirli bir alias'ı ADIYLA çözmek
                # (MLflow `models:/<ad>@<alias>`). Boşsa en yeni kazanır.
                # "Sabitlenmiş olanı ver" diye bir seçenek YOK — iki alias
                # varsa o kural tanımsız kalıyordu, bkz. `pipeline_calistir`.
                tercih = str(hn.get("tercih_alias")
                             or mevcut.get("alias") or "").strip()
                if tercih:
                    if not _TAKMA_BICIMI.match(tercih):
                        raise HatGecersiz(
                            f"{i}. adımın istediği alias biçimi geçersiz.")
                    nd["sorgu"]["alias"] = tercih
            if tur == "alias":
                takma = str(hn.get("alias") or "").strip()
                if not _TAKMA_BICIMI.match(takma):
                    raise HatGecersiz(f"{i}. adımın alias biçimi geçersiz.")
                nd["alias"] = takma
                sec = str(hn.get("sec") or "en_yeni").strip()
                if sec not in {"en_yeni", "en_eski"}:
                    raise HatGecersiz(f"{i}. adımın seçimi: en_yeni | en_eski.")
                nd["sec"] = sec
        nodes.append(nd)

    return {"key": key, "kod": f"PL-{key.upper()[:6]}", "ad": ad,
            "aciklama": str(ham.get("aciklama") or "").strip()[:300]
                        or "Konsoldan kurulmuş hat.",
            "kullanici": True, "nodes": nodes}


def hat_kaydet(ham: dict) -> dict:
    """Doğrular ve diske yazar; aynı `key` varsa üzerine yazar."""
    hat = hat_dogrula(ham)
    hatlar = [h for h in _kullanici_hatlari() if h.get("key") != hat["key"]]
    hatlar.append(hat)
    _kullanici_hatlarini_yaz(hatlar)
    return hat


def hat_sil(key: str) -> bool:
    if key in _YERLESIK_ANAHTAR:
        raise HatGecersiz(f"'{key}' yerleşik bir hat — silinemez.")
    hatlar = _kullanici_hatlari()
    kalan = [h for h in hatlar if h.get("key") != key]
    if len(kalan) == len(hatlar):
        return False
    _kullanici_hatlarini_yaz(kalan)
    return True


def pipelines() -> list[dict]:
    """Yerleşikler + kullanıcının kurdukları. Panel bunu çiziyor."""
    return [f() for f in _YERLESIK] + _kullanici_hatlari()


def hat_bul(key: str) -> dict | None:
    return next((h for h in pipelines() if h["key"] == key), None)


# ── çalıştırma ────────────────────────────────────────────────────────────


def _damga() -> str:
    return time.strftime("%H:%M:%S")


def pipeline_calistir(key: str, kaynak_wf: str | None, jeton_uret, yay) -> dict:
    """Bir pipeline'ı GERÇEKTEN çalıştırır; her olayı `yay(dict)` ile bildirir.

    Bloklayıcı — çağıran `asyncio.to_thread` ile sarmalı.

    `kaynak_wf`: B pipeline'ı için, A'nın çalıştırma kimliği. Verilmezse node 1
    kayıt defterine sorup KENDİSİ buluyor — asıl gösterilmek istenen de bu.
    """
    from grounded_assistant.ptc.sandbox_runner import run_sandbox  # noqa: PLC0415

    hat = hat_bul(key)
    if hat is None:
        yay({"type": "log", "n": 0, "ts": _damga(),
             "msg": f"'{key}' diye bir hat yok.", "cls": "fail"})
        yay({"type": "pipeline_done", "status": "error"})
        return {"workflow_id": None, "status": "error"}
    workflow_id = str(uuid.uuid4())
    yay({"type": "pipeline_start", "key": key, "workflow_id": workflow_id,
         "ad": hat["ad"], "kod": hat["kod"], "nodes": len(hat["nodes"])})

    uretilen: list[dict] = []
    cozulen_wf = kaynak_wf

    for nd in hat["nodes"]:
        yay({"type": "node_start", "n": nd["n"], "ad": nd["ad"], "tur": nd["tur"]})
        t0 = time.monotonic()

        # ── keşif adımı: pod YOK, kayıt defteri sorgusu ──────────────────
        if nd["tur"] == "query":
            yay({"type": "log", "n": nd["n"], "ts": _damga(),
                 "msg": f"GET /artifacts?name={nd['sorgu']['name']}", "cls": ""})
            sonuc = depo(jeton_uret, **nd["sorgu"])
            if sonuc.get("error"):
                yay({"type": "log", "n": nd["n"], "ts": _damga(),
                     "msg": sonuc["error"], "cls": "fail"})
                yay({"type": "node_done", "n": nd["n"], "status": "error",
                     "dur": f"{time.monotonic()-t0:.1f}s", "sonuc": sonuc["error"]})
                yay({"type": "pipeline_done", "workflow_id": workflow_id,
                     "status": "error"})
                return {"workflow_id": workflow_id, "status": "error"}

            kayitlar = sonuc["kayitlar"]
            yay({"type": "log", "n": nd["n"], "ts": _damga(),
                 "msg": f"{len(kayitlar)} eşleşme", "cls": "art"})
            if not kayitlar:
                yay({"type": "log", "n": nd["n"], "ts": _damga(),
                     "msg": "Önce PL-A'yı çalıştırın — tüketilecek artifact yok.",
                     "cls": "fail"})
                yay({"type": "node_done", "n": nd["n"], "status": "error",
                     "dur": f"{time.monotonic()-t0:.1f}s",
                     "sonuc": "processed-result.json bulunamadı"})
                yay({"type": "pipeline_done", "workflow_id": workflow_id,
                     "status": "error"})
                return {"workflow_id": workflow_id, "status": "error"}

            # SÜRÜM SEÇİMİ — vakanın asıl gösterdiği şey burası.
            #
            # Aynı ad depoda onlarca kez var (her PL-A çalıştırması bir tane
            # daha ekliyor) ve "en yeni kazanır" kuralı SESSİZ.
            #
            # 2026-09-07: burada bir zamanlar `next(k for k in kayitlar if
            # k.get("alias"))` vardı — yani "hangisi sabitlenmişse onu ver".
            # Aynı ada İKİ farklı alias konunca (PL-C bunu yapıyor) o kural
            # tanımsız hâle geldi: liste yeniden-eskiye sıralı olduğu için
            # sessizce ilk rastlanan kazanıyordu. Çözmeye çalıştığımız sessiz
            # seçim, bir üst katta aynen tekrarlanıyordu.
            #
            # MLflow'da alias ADIYLA istenir (`models:/<ad>@<alias>`);
            # "sabitlenmiş olanı ver" diye bir çağrı yok. Biz de öyle:
            # `sorgu["alias"]` varsa TAM O alias, yoksa en yeni.
            istenen_alias = nd["sorgu"].get("alias")
            if istenen_alias:
                secilen = next((k for k in kayitlar
                                if k.get("alias") == istenen_alias), None)
                if secilen is None:
                    mesaj = f"@{istenen_alias} alias'ı bu adda atanmamış."
                    yay({"type": "log", "n": nd["n"], "ts": _damga(),
                         "msg": mesaj, "cls": "fail"})
                    yay({"type": "node_done", "n": nd["n"], "status": "error",
                         "dur": f"{time.monotonic()-t0:.1f}s", "sonuc": mesaj})
                    yay({"type": "pipeline_done", "workflow_id": workflow_id,
                         "status": "error"})
                    return {"workflow_id": workflow_id, "status": "error"}
                yay({"type": "log", "n": nd["n"], "ts": _damga(),
                     "msg": f"@{istenen_alias} ADIYLA istendi "
                            f"({len(kayitlar)} aday arasından)", "cls": "hi"})
            else:
                secilen = kayitlar[0]
                sabitler = [k["alias"] for k in kayitlar if k.get("alias")]
                yay({"type": "log", "n": nd["n"], "ts": _damga(),
                     "msg": f"alias istenmedi → en yeni seçildi "
                            f"({len(kayitlar)} aday)", "cls": ""})
                if sabitler:
                    # Sessizce geçmiyoruz: sabitlenmiş sürümler VAR ama bu
                    # adım onları istemedi. İzleyici farkı görsün.
                    yay({"type": "log", "n": nd["n"], "ts": _damga(),
                         "msg": f"not: bu adda sabitlenmiş sürüm(ler) var "
                                f"({', '.join('@' + a for a in sabitler)}) — "
                                f"bu adım hiçbirini istemedi", "cls": ""})
            cozulen_wf = secilen["workflow_id"]
            yay({"type": "log", "n": nd["n"], "ts": _damga(),
                 "msg": f"çözüldü {secilen['artifact_id']} "
                        f"({secilen['size_bytes']} bayt)", "cls": "art"})
            yay({"type": "log", "n": nd["n"], "ts": _damga(),
                 "msg": f"üreten çalıştırma {cozulen_wf}", "cls": "hi"})
            yay({"type": "node_done", "n": nd["n"], "status": "success",
                 "dur": f"{time.monotonic()-t0:.1f}s",
                 "sonuc": {"artifact_id": secilen["artifact_id"],
                           "workflow_id": cozulen_wf,
                           "secim": f"@{istenen_alias}" if istenen_alias
                                    else "en yeni",
                           "aday": len(kayitlar)}})
            continue

        # ── alias adımı: pod YOK, sürüm sabitleme (MLflow deseni) ────────
        if nd["tur"] == "alias":
            yay({"type": "log", "n": nd["n"], "ts": _damga(),
                 "msg": f"GET /artifacts?name={nd['sorgu']['name']}", "cls": ""})
            sonuc = depo(jeton_uret, **nd["sorgu"])
            kayitlar = sonuc.get("kayitlar") or []
            if sonuc.get("error") or not kayitlar:
                mesaj = sonuc.get("error") or (
                    f"{nd['sorgu']['name']} bulunamadı — önce üreten hattı "
                    "çalıştırın.")
                yay({"type": "log", "n": nd["n"], "ts": _damga(),
                     "msg": mesaj, "cls": "fail"})
                yay({"type": "node_done", "n": nd["n"], "status": "error",
                     "dur": f"{time.monotonic()-t0:.1f}s", "sonuc": mesaj})
                yay({"type": "pipeline_done", "workflow_id": workflow_id,
                     "status": "error"})
                return {"workflow_id": workflow_id, "status": "error"}

            # Servis yeniden-eskiye sıralı döndürüyor.
            hedef = kayitlar[-1] if nd.get("sec") == "en_eski" else kayitlar[0]
            yanit = alias_ata(jeton_uret, hedef["artifact_id"], nd["alias"])
            if yanit.get("error"):
                yay({"type": "log", "n": nd["n"], "ts": _damga(),
                     "msg": yanit["error"], "cls": "fail"})
                yay({"type": "node_done", "n": nd["n"], "status": "error",
                     "dur": f"{time.monotonic()-t0:.1f}s", "sonuc": yanit["error"]})
                yay({"type": "pipeline_done", "workflow_id": workflow_id,
                     "status": "error"})
                return {"workflow_id": workflow_id, "status": "error"}

            yay({"type": "log", "n": nd["n"], "ts": _damga(),
                 "msg": f"{len(kayitlar)} aday · {nd.get('sec','en_yeni')} seçildi",
                 "cls": ""})
            yay({"type": "log", "n": nd["n"], "ts": _damga(),
                 "msg": f"@{nd['alias']} → {hedef['artifact_id']} "
                        f"(üreten {hedef['workflow_id'][:8]})", "cls": "hi"})
            cozulen_wf = hedef["workflow_id"]
            yay({"type": "node_done", "n": nd["n"], "status": "success",
                 "dur": f"{time.monotonic()-t0:.1f}s",
                 "sonuc": {"alias": nd["alias"],
                           "artifact_id": hedef["artifact_id"],
                           "workflow_id": cozulen_wf,
                           "secim": nd.get("sec", "en_yeni"),
                           "aday": len(kayitlar)}})
            continue

        # ── sandbox adımı: GERÇEK pod ────────────────────────────────────
        kod = nd["kod"]
        # Beyan da çalışma anında dolduruluyor: keşif adımı kimliği buluyor,
        # yerleştirme onu kullanıyor. Kimlik hiçbir yere gömülü değil.
        girdiler = [g.replace(_YER_TUTUCU, cozulen_wf) for g in nd["inputs"]] \
            if cozulen_wf else [g for g in nd["inputs"] if _YER_TUTUCU not in g]
        if _YER_TUTUCU in kod:
            if not cozulen_wf:
                yay({"type": "log", "n": nd["n"], "ts": _damga(),
                     "msg": "Kaynak çalıştırma bilinmiyor", "cls": "fail"})
                yay({"type": "pipeline_done", "workflow_id": workflow_id,
                     "status": "error"})
                return {"workflow_id": workflow_id, "status": "error"}
            kod = kod.replace(_YER_TUTUCU, cozulen_wf)

        olaylar: list[dict] = []

        def _on_event(e: dict, _n=nd["n"], _bucket=olaylar) -> None:
            _bucket.append(e)
            sahne = e.get("stage")
            if sahne == "job_created":
                yay({"type": "log", "n": _n, "ts": _damga(),
                     "msg": f"Job ptc-sandbox-{e['run_id']} yaratıldı", "cls": ""})
            elif sahne == "pod_running":
                yay({"type": "log", "n": _n, "ts": _damga(),
                     "msg": f"pod çalışıyor · {e['job_name']}", "cls": "hi"})
            elif sahne == "artifact":
                yay({"type": "log", "n": _n, "ts": _damga(),
                     "msg": f"{e.get('op')} {e.get('name')} "
                            f"({e.get('size_bytes') or 0} bayt)", "cls": "art"})
                yay({"type": "artifact", "n": _n, **e})
            elif sahne == "artifact_skipped":
                yay({"type": "log", "n": _n, "ts": _damga(),
                     "msg": f"atlandı {e.get('name')} — {e.get('detail','')}",
                     "cls": "fail"})

        yay({"type": "log", "n": nd["n"], "ts": _damga(),
             "msg": f"beyan edilen girdiler: {girdiler or '—'}", "cls": ""})
        yay({"type": "code", "n": nd["n"], "kod": kod.strip()})

        run = run_sandbox(kod, on_event=_on_event, workflow_id=workflow_id,
                          owner="ptc", node_id=f"node-{nd['n']}",
                          inputs=girdiler)

        for o in run.artifacts:
            if o.op.value == "produced":
                uretilen.append({"artifact_id": o.artifact_id, "name": o.name,
                                 "n": nd["n"]})

        yay({"type": "log", "n": nd["n"], "ts": _damga(),
             "msg": f"{run.status.value} · {run.result_text or run.error_message or ''}",
             "cls": "hi" if run.status.value == "success" else "fail"})
        yay({"type": "node_done", "n": nd["n"], "status": run.status.value,
             "dur": f"{time.monotonic()-t0:.1f}s",
             "sonuc": run.result_text or run.error_message,
             "run_id": run.run_id})

        if run.status.value != "success":
            yay({"type": "pipeline_done", "workflow_id": workflow_id,
                 "status": "error"})
            return {"workflow_id": workflow_id, "status": "error"}

    yay({"type": "pipeline_done", "workflow_id": workflow_id, "status": "success",
         "uretilen": uretilen, "kaynak_wf": cozulen_wf})
    return {"workflow_id": workflow_id, "status": "success", "uretilen": uretilen}
