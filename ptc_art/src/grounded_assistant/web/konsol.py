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

## İki node TÜRÜ var ve fark gerçek

    sandbox : gerçek PTC pod'u — kimlik bilgisi yok, ağı kapalı, süpürülüyor
    query   : host tarafında kayıt defteri sorgusu — pod açılmıyor

Bu ayrım uydurma değil, mimarinin kendisi: **keşif sandbox'ta olmuyor.**
Sandbox'ın listeleme yolu hiç yok; hangi artifact'in var olduğunu host
tarafındaki kayıt defteri sorgusu söylüyor (§11.14 süzgeç). Panelde bu iki tür
farklı çiziliyor ki izleyici nerede pod açıldığını görsün.

## Çapraz workflow

B pipeline'ı A'nın çıktısını `load_artifact(<A'nın workflow_id'si>, ...)` ile
okuyor. A'nın kimliğini B'ye veren şey node 1'in kayıt defteri sorgusu — yani
tam olarak ürünün kendi keşif yolu, elle gömülmüş bir kimlik değil.
"""

from __future__ import annotations

import os
import time
import uuid

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

_KOD_YUKLE = '''
import json, shutil
yol = load_artifact("{kaynak_wf}", "processed-result.json")
veri = json.load(open(yol))
json.dump(veri, open("/output/analysis-input.json", "w"))
set_result({{"kaynak_yol": yol, "alanlar": list(veri)}})
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
                         "keşif sandbox'ta değil, host tarafında olur.",
             "sorgu": {"name": "processed-result.json"}, "inputs": [],
             "bekleniyor": []},
            {"n": 2, "ad": "Artifact Yükle", "tur": "sandbox", "ikon": "i-down",
             "aciklama": "Bulunan artifact'i `load_artifact` ile açıkça ister. "
                         "Çapraz-workflow okuma tam burada oluyor.",
             "kod": _KOD_YUKLE, "inputs": [], "bekleniyor": ["analysis-input.json"]},
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


def pipelines() -> list[dict]:
    return [pipeline_a(), pipeline_b()]


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

    hat = pipeline_a() if key == "a" else pipeline_b()
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

            en_yeni = kayitlar[0]
            cozulen_wf = en_yeni["workflow_id"]
            yay({"type": "log", "n": nd["n"], "ts": _damga(),
                 "msg": f"çözüldü {en_yeni['artifact_id']} "
                        f"({en_yeni['size_bytes']} bayt)", "cls": "art"})
            yay({"type": "log", "n": nd["n"], "ts": _damga(),
                 "msg": f"üreten çalıştırma {cozulen_wf}", "cls": "hi"})
            yay({"type": "node_done", "n": nd["n"], "status": "success",
                 "dur": f"{time.monotonic()-t0:.1f}s",
                 "sonuc": {"artifact_id": en_yeni["artifact_id"],
                           "workflow_id": cozulen_wf,
                           "eslesme": len(kayitlar)}})
            continue

        # ── sandbox adımı: GERÇEK pod ────────────────────────────────────
        kod = nd["kod"]
        if "{kaynak_wf}" in kod:
            if not cozulen_wf:
                yay({"type": "log", "n": nd["n"], "ts": _damga(),
                     "msg": "Kaynak çalıştırma bilinmiyor", "cls": "fail"})
                yay({"type": "pipeline_done", "workflow_id": workflow_id,
                     "status": "error"})
                return {"workflow_id": workflow_id, "status": "error"}
            kod = kod.format(kaynak_wf=cozulen_wf)

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
             "msg": f"beyan edilen girdiler: {nd['inputs'] or '—'}", "cls": ""})
        yay({"type": "code", "n": nd["n"], "kod": kod.strip()})

        run = run_sandbox(kod, on_event=_on_event, workflow_id=workflow_id,
                          owner="ptc", node_id=f"node-{nd['n']}",
                          inputs=nd["inputs"])

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
