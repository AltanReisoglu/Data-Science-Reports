"""PTC PoC — uçtan uca kabul testi (canlı cluster gerektirir).

`pytest` birim/entegrasyon testleri kodu sınıyor; bu script ÜRÜNÜ sınıyor:
gerçek pod'lar doğuyor, gerçek MinIO'ya yazılıyor, gerçek ağ politikası
deneniyor. 38 kontrol, hepsi ölçülüyor — hiçbiri varsayılmıyor.

## Ön koşullar

    kubectl port-forward svc/artifact-service 8080:8080     # kayıt defteri
    uvicorn grounded_assistant.web.app:app --port 8123      # panel (§8 için)

## Kullanım

    python scripts/kabul_testi.py            # 0 = hepsi geçti, 1 = kalan var

## Neden ada göre DEĞİL kimliğe göre arıyor

İlk hâli artifact'leri adıyla arıyordu ve tenant genelinde aynı isimde eski
kayıtlar olduğu için yanlış artifact'i seçip sahte FAIL üretiyordu — ürün
değil testin kusuruydu. Her kontrol artık çalıştırmanın KENDİ ürettiği
`artifact_id`'leri izliyor.
"""
from __future__ import annotations
import json, os, sys, time, uuid
import requests
from dotenv import load_dotenv
load_dotenv()

from grounded_assistant.ptc.sandbox_runner import run_sandbox
from grounded_assistant.agent.graph import _kapsam_jetonu

SERVIS = os.environ.get("ARTIFACT_SERVICE_URL", "http://localhost:8080")
sonuclar: list[tuple[str, bool, str]] = []

def kontrol(ad, kosul, ayrinti=""):
    sonuclar.append((ad, bool(kosul), str(ayrinti)[:110]))
    print(f"  {'PASS' if kosul else 'FAIL'}  {ad}" + (f"  — {str(ayrinti)[:90]}" if ayrinti else ""))

def kos(kod, wf, node=None, inputs="YOK"):
    kw = {} if inputs == "YOK" else {"inputs": inputs}
    return run_sandbox(kod, workflow_id=wf, node_id=node, **kw)

def basla(b): print(f"\n{'─'*70}\n{b}\n{'─'*70}")

WF_A, WF_B = str(uuid.uuid4()), str(uuid.uuid4())

# ══ 1. Üretim: düz Python, API yok ═══════════════════════════════════════
basla("1 · ÜRETİM — düz Python, hiçbir özel fonksiyon yok")
r = kos("""
import pandas as pd, json, os
df = pd.DataFrame([{"ticket": f"T-{i}", "departman": ["BT","IK","Finans"][i%3],
                    "gun": (i%14)+1, "durum": ["open","closed"][i%2]} for i in range(150)])
df.to_parquet("/output/ham.tickets.parquet")
json.dump({"kaynak":"crm","satir":len(df)}, open("/output/kunye.json","w"))
os.makedirs("/output/model.v1/alt", exist_ok=True)
open("/output/model.v1/weights.json","w").write('{"w":[1,2,3]}')
open("/output/model.v1/alt/derin.txt","w").write("derin dosya")
# API'nin GERÇEKTEN yok olduğunu kanıtla
yok = [a for a in ("put_artifact","get_artifact","list_artifacts","cached")
       if a in dir(__builtins__) or a in globals()]
set_result({"satir": len(df), "sizan_api": yok})
""", WF_A, "extract")
kontrol("üretim başarılı", r.status.value == "success", r.result_text or r.error_message)
kontrol("LLM'e artifact API'si sızmıyor", "'sizan_api': []" in (r.result_text or ""))
KIMLIK: dict[str, str] = {}          # ad -> BU çalıştırmanın ürettiği artifact_id
def kimlikleri_al(run):
    for o in run.artifacts:
        if o.op.value == "produced":
            KIMLIK[o.name] = o.artifact_id
kimlikleri_al(r)
uretilen = {o.name for o in r.artifacts if o.op.value == "produced"}
kontrol("3 artifact saklandı (dosya+json+dizin)",
        uretilen == {"ham.tickets.parquet", "kunye.json", "model.v1.tar"}, sorted(uretilen))

# ══ 2. Çalıştırmalar arası keşif + kullanım ══════════════════════════════
basla("2 · BAŞKA WORKFLOW — /output izole, load_artifact ile açık erişim")
r = kos(f"""
import os, pandas as pd, json
# KENDİ /output'u BOŞ olmalı — başka run'ın çıktısı buraya sızmamalı.
# (2026-09-06: sızıyordu ve ajan başkasının dosyasını kendi işi sanıyordu.)
sonuc = {{
  "kendi_output_bos": os.listdir("/output") == [],
  "output_sizinti":   os.path.exists("/output/ham.tickets.parquet"),
}}
# Başkasınınki BEYANLA geliyor — kod hiçbir çağrı yapmıyor, dosya hazır.
# (KFP: launcher `.uri`leri `.path`e indirir, kullanıcı kodu çağrı yapmaz.)
p = "/artifacts/{WF_A}/ham.tickets.parquet"
sonuc["satir"] = len(pd.read_parquet(p))
sonuc["kunye"] = json.load(open("/artifacts/{WF_A}/kunye.json"))["kaynak"]
sonuc["dizin"] = open("/artifacts/{WF_A}/model.v1/alt/derin.txt").read()
sonuc["yol"] = os.path.exists(p)
sonuc["cagri_yok"] = "load_artifact" not in globals()
set_result(sonuc)
""", WF_B, "kesif", inputs=[f"{WF_A}/ham.tickets.parquet", f"{WF_A}/kunye.json",
                            f"{WF_A}/model.v1"])
kontrol("başka workflow okuyabiliyor", r.status.value == "success", r.error_message or "")
if r.status.value == "success":
    d = eval(r.result_text)
    kontrol("kendi /output'u İZOLE (başkasınınki sızmıyor)",
            d["kendi_output_bos"] and not d["output_sizinti"])
    kontrol("beyan /artifacts/<wf>/ altına yerleştiriyor", d["yol"])
    kontrol("sandbox'ta artifact ÇAĞRISI yok", d["cagri_yok"])
    kontrol("parquet düz read_parquet ile okundu", d["satir"] == 150, d["satir"])
    kontrol("json düz open ile okundu", d["kunye"] == "crm")
    kontrol("dizin artifact'i açıldı", d["dizin"] == "derin dosya")

# ══ 3. Türetme + otomatik soy ════════════════════════════════════════════
basla("3 · TÜRETME — soy ağacı kendiliğinden kuruluyor")
r = kos(f"""
import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
df = pd.read_parquet("/artifacts/{WF_A}/ham.tickets.parquet")
ozet = df.groupby("departman")["gun"].mean().round(2)
ozet.to_frame("ort").to_parquet("/output/departman.ozet.parquet")
fig, ax = plt.subplots(figsize=(6,3)); ax.bar(list(ozet.index), list(ozet.values))
fig.savefig("/output/dagilim.png", dpi=100)
with PdfPages("/output/rapor.pdf") as pdf: pdf.savefig(fig)
set_result({{"ozet": ozet.to_dict()}})
""", WF_B, "report", inputs=[f"{WF_A}/ham.tickets.parquet"])
kontrol("PDF/PNG üretimi", r.status.value == "success", r.error_message or "")
kimlikleri_al(r)
soy = {o.name: list(o.parents) for o in r.artifacts if o.op.value == "produced"}
kontrol("çıktılar okunan girdiden türedi",
        all(len(v) >= 1 for v in soy.values()), soy)

# ══ 4. Kayıt defteri + tipler ════════════════════════════════════════════
basla("4 · KAYIT DEFTERİ")
H = {"X-Scope-Token": _kapsam_jetonu(WF_B)}
kayit = requests.get(f"{SERVIS}/artifacts", headers=H, timeout=15).json()
# İSİMLE DEĞİL KİMLİKLE bak: tenant'ta aynı isimde eski kayıtlar var
# (belgelenmiş "isim çakışması" açığı) ve isimle arayan bir test yanlış
# artifact'i seçiyor. Testin kendi ürettiklerini kimlikle izliyoruz.
adlar = {k["artifact_id"]: k for k in kayit}
kontrol("bu çalıştırmanın ürettikleri listede",
        all(i in adlar for i in KIMLIK.values()),
        [a for a, i in KIMLIK.items() if i not in adlar])
kontrol("tenant listelemesi iki workflow'u da kapsıyor",
        len({adlar[i]["workflow_id"] for i in
             (KIMLIK["ham.tickets.parquet"], KIMLIK["rapor.pdf"])}) == 2)
kontrol("parquet → system.Dataset",
        adlar[KIMLIK["ham.tickets.parquet"]]["type"] == "system.Dataset")
kontrol("pdf → application/pdf",
        adlar[KIMLIK["rapor.pdf"]]["content_type"] == "application/pdf")
kontrol("png → image/png",
        adlar[KIMLIK["dagilim.png"]]["content_type"] == "image/png")
kontrol("her kayıtta content_hash var", all(k.get("content_hash") for k in kayit))

# ── soy grafiği
aid = KIMLIK["ham.tickets.parquet"]
g = requests.get(f"{SERVIS}/artifacts/{aid}/lineage", headers=H, timeout=15).json()
urun = {n["name"] for n in g["nodes"] if n["yon"] == "urun"}
kontrol("soy grafiği ürünleri buluyor",
        {"departman.ozet.parquet","rapor.pdf","dagilim.png"} <= urun, sorted(urun))

# ══ 5. Güvenlik sınırları ════════════════════════════════════════════════
basla("5 · GÜVENLİK SINIRLARI")
r = kos("""
import os, socket, requests
s = {}
s["s3_kimlik"] = [k for k in os.environ if k.startswith(("AWS_ACCESS","AWS_SECRET"))]
try: import boto3; s["boto3"]="VAR"
except ImportError: s["boto3"]="yok"
try:
    socket.create_connection((os.environ["MINIO_PORT_9000_TCP_ADDR"],9000),timeout=6).close()
    s["minio_ip"]="ULASILDI"
except Exception as e: s["minio_ip"]=type(e).__name__
try:
    requests.get("https://example.com", timeout=6); s["internet"]="ACIK"
except Exception as e: s["internet"]=type(e).__name__
# 2026-09-06: jeton bu container'da YOK (sidecar taşıyor). Artık asıl soru
# "jetonla ne yapabilir" değil, "jetonsuz ne yapabilir".
s["jeton_sizinti"] = [k for k in os.environ if "SCOPE_TOKEN" in k]
uc = os.environ.get("ARTIFACT_SERVICE_ENDPOINT", "")
try:
    s["jetonsuz_yazma"] = requests.post(f"{uc}/artifacts", data=b"x",
        headers={"Content-Type":"text/plain","X-Artifact-Name":"kacis.txt"},
        timeout=10).status_code
except Exception as e:
    s["jetonsuz_yazma"] = type(e).__name__
px = os.environ.get("PTC_ARTIFACT_PROXY", "")
try:
    s["proxy_yazma"] = requests.post(f"{px}/fetch", data=b"x", timeout=5).status_code
except Exception as e:
    s["proxy_yazma"] = type(e).__name__
set_result(s)
""", str(uuid.uuid4()), "guvenlik")
kontrol("güvenlik sondası çalıştı", r.status.value == "success", r.error_message or "")
if r.status.value == "success":
    s = eval(r.result_text)
    kontrol("sandbox'ta S3 kimlik bilgisi YOK", s["s3_kimlik"] == [], s["s3_kimlik"])
    kontrol("S3 SDK kurulu değil", s["boto3"] == "yok")
    # Bu kontrol KİPE bağlı ve bilerek öyle: `direct` kipinde sidecar depoya
    # kendisi yazıyor, bunun için pod'un rotası açılıyor ve NetworkPolicy pod
    # seçtiği için sandbox da o rotayı kazanıyor. Kip anahtarı bunu
    # kendiliğinden kapatmıyor — ölçüp yazmak, sessizce geçmekten iyi.
    _kip = os.environ.get("PTC_ARTIFACT_TRANSFER", "proxy")
    if _kip == "direct":
        kontrol("[direct] MinIO'ya rota AÇIK — kipin bilinen bedeli",
                s["minio_ip"] == "ULASILDI", s["minio_ip"])
        kontrol("[direct] rotaya rağmen S3 anahtarı YOK", s["s3_kimlik"] == [],
                s["s3_kimlik"])
    else:
        kontrol("[proxy] MinIO'ya doğrudan IP ile gidilemiyor",
                s["minio_ip"] != "ULASILDI", s["minio_ip"])
    kontrol("internet kapalı", s["internet"] != "ACIK", s["internet"])
    kontrol("sandbox'ta KAPSAM JETONU da YOK (sidecar'da)",
            s["jeton_sizinti"] == [], s["jeton_sizinti"])
    # 2026-09-07 (ikinci tur): sandbox'ta artifact istemcisi de, servis
    # ADRESİ de kalmadı. Eskiden "adresi var ama jetonsuz yazamıyor" diye
    # ölçüyorduk; artık adres de yok, yani ölçülecek şey değişti.
    kontrol("sandbox'ta servis ADRESİ yok", s["jetonsuz_yazma"] == "MissingSchema",
            s["jetonsuz_yazma"])
    kontrol("sandbox'ta localhost proxy yok",
            not (isinstance(s["proxy_yazma"], int) and 200 <= s["proxy_yazma"] < 300),
            s["proxy_yazma"])
# Jetonsuz yazmanın 401 döndüğü DIŞARIDAN doğrulanıyor: sandbox'ın adresi
# olmaması, servisin açık olduğu anlamına gelmez.
kontrol("jetonsuz doğrudan yazma reddedildi (401)",
        requests.post(f"{SERVIS}/artifacts", data=b"x",
                      headers={"X-Artifact-Name": "z.txt", "Content-Type": "text/plain"},
                      timeout=15).status_code == 401)

# ── tenant sınırı (dışarıdan)
from grounded_assistant.artifacts.scope import Scope, issue_token
sir = os.popen("kubectl get secret ptc-scope-signing -o jsonpath='{.data.secret}'").read()
import base64
if sir:
    yabanci = issue_token(base64.b64decode(sir).decode(),
                          Scope(workflow_id=WF_A, run_id="r", owner="baska-tenant"))
    kod = requests.get(f"{SERVIS}/artifacts/{aid}",
                       headers={"X-Scope-Token": yabanci}, timeout=10).status_code
    kontrol("başka TENANT okuyamıyor (404)", kod == 404, kod)

# ══ 5b. pickle: süpürme yolundan da reddedilmeli ═════════════════════════
basla("5b · PICKLE — süpürme yolunda da reddediliyor")
r = kos("""
import pickle
open("/output/zehir.bin","wb").write(pickle.dumps({"k":1}))
open("/output/temiz.txt","w").write("bu gecmeli")
set_result("iki dosya yazildi")
""", WF_B, "pickle")
uretilen = {o.name for o in r.artifacts if o.op.value == "produced"}
kontrol("pickle depoya GİRMEDİ", "zehir.bin" not in uretilen, sorted(uretilen))
kontrol("yanındaki temiz dosya etkilenmedi", "temiz.txt" in uretilen, sorted(uretilen))

# ══ 6. Hata dayanıklılığı ════════════════════════════════════════════════
basla("6 · HATA DAYANIKLILIĞI")
r = kos("""
open("/output/kismi.txt","w").write("hatadan once yazildi")
raise ValueError("kasitli hata")
""", WF_B, "hata")
kontrol("hata durumu bildirildi", r.status.value == "error", r.status.value)
kontrol("hatadan ÖNCE yazılan kurtarıldı",
        any(o.name == "kismi.txt" for o in r.artifacts),
        [o.name for o in r.artifacts])

# ══ 7. Dedup ═════════════════════════════════════════════════════════════
basla("7 · DEDUP")
damga = uuid.uuid4().hex[:8]
r = kos(f"""
open("/output/kopya.a.{damga}.txt","w").write("tamamen ayni icerik {damga}")
open("/output/kopya.b.{damga}.txt","w").write("tamamen ayni icerik {damga}")
set_result("iki dosya")
""", WF_B, "dedup")
ikili = [o for o in r.artifacts if o.op.value == "produced"]
kontrol("iki kayıt oluştu", len(ikili) == 2, [o.name for o in ikili])
kayit2 = requests.get(f"{SERVIS}/artifacts", headers=H, timeout=15).json()
kimlikler = {o.artifact_id for o in ikili}
hashler = {k["content_hash"] for k in kayit2 if k["artifact_id"] in kimlikler}
kontrol("aynı içerik → aynı hash (tek bayt)", len(hashler) == 1, hashler)

# ══ 8. Panel API'leri ════════════════════════════════════════════════════
basla("8 · PANEL")
PANEL = os.environ.get("PTC_PANEL_URL", "http://127.0.0.1:8123")
try:
    d = requests.get(f"{PANEL}/api/durum", params={"session": WF_B}, timeout=25).json()
    kontrol("/api/durum yanıt veriyor", d["artifactler"].get("error") is None,
            d["artifactler"].get("error"))
    kontrol("panel artifact listeliyor", len(d["artifactler"]["kayitlar"]) > 0)
    kontrol("künyede workflow_id var",
            all("workflow_id" in k for k in d["artifactler"]["kayitlar"]))
    o = requests.get(f"{PANEL}/api/artifact/{KIMLIK['rapor.pdf']}",
                     params={"session": WF_B}, timeout=25).json()
    kontrol("PDF önizlemesi geliyor", "pdf" in o, list(o)[:2])
    sg = requests.get(f"{PANEL}/api/artifact/{aid}/soy",
                      params={"session": WF_B}, timeout=25).json()
    kontrol("panel soy grafiği", len(sg.get("nodes", [])) >= 4, len(sg.get("nodes", [])))
except Exception as e:
    kontrol("panel erişilebilir", False, f"{type(e).__name__}: {e}")

# ══ 9. Kanıtlanmış üç mekanizma (2026-09-07) ═════════════════════════════
basla("9 · BEYAN · SÜZGEÇ · ALIAS — üçü de sahadan kopya")

# 9a — BEYAN: Argo `inputs.artifacts`, KFP bileşen girdisi, MLMD DECLARED_INPUT
WF_C = str(uuid.uuid4())
kos("""
for ad in ("a.txt","b.txt","c.txt"): open("/output/"+ad,"w").write(ad[0].upper())
set_result("ok")
""", WF_C, inputs=[])
r = kos("""
import os
open("/output/turev.txt","w").write(open("/output/a.txt").read()+"!")
set_result({"gordugu": sorted(os.listdir("/output"))})
""", WF_C, inputs=["a.txt"])
kontrol("beyan çalıştı", r.status.value == "success", r.error_message or "")
if r.status.value == "success":
    kontrol("beyan yerleştirmeyi daraltıyor",
            eval(r.result_text)["gordugu"] == ["a.txt", "turev.txt"],
            eval(r.result_text)["gordugu"])
    ebeveyn = {o.name: list(o.parents) for o in r.artifacts if o.op.value == "produced"}
    kontrol("soy YALNIZCA beyan edilen girdiden", len(ebeveyn.get("turev.txt", [])) == 1,
            ebeveyn.get("turev.txt"))

r = kos("""
import os
open("/output/turev2.txt","w").write("x")
set_result({"gordugu": sorted(os.listdir("/output"))})
""", WF_C)
kontrol("beyansız uyumluluk yolu duruyor",
        r.status.value == "success" and len(eval(r.result_text)["gordugu"]) >= 4,
        r.error_message or eval(r.result_text)["gordugu"])

# 9b — SÜZGEÇ: MLMD `ListOptions(filter_query=...)` karşılığı
hepsi = requests.get(f"{SERVIS}/artifacts", headers=H, timeout=20).json()
ada = requests.get(f"{SERVIS}/artifacts", params={"name": "a.txt"}, headers=H, timeout=20).json()
tipe = requests.get(f"{SERVIS}/artifacts", params={"type": "system.Dataset"},
                    headers=H, timeout=20).json()
ara = requests.get(f"{SERVIS}/artifacts", params={"q": "turev"}, headers=H, timeout=20).json()
joker = requests.get(f"{SERVIS}/artifacts", params={"q": "%"}, headers=H, timeout=20).json()
kontrol("?name= süzüyor", 0 < len(ada) < len(hepsi), f"{len(ada)}/{len(hepsi)}")
kontrol("?type= süzüyor", 0 < len(tipe) < len(hepsi), f"{len(tipe)}/{len(hepsi)}")
kontrol("?q= ad içinde arıyor", all("turev" in k["name"] for k in ara), len(ara))
kontrol("LIKE jokeri kaçırılıyor", joker == [], len(joker))

# 9c — ALIAS: MLflow `models:/<ad>@<alias>` karşılığı
# Alias kontrolü KENDİ artifact'ini üretiyor: "a.txt" gibi ortak bir ada
# bağlanınca, depoda başka çalıştırmalardan kalan aynı adlı kayıtlar sonucu
# belirliyordu ve test depo durumuna göre geçip kalıyordu.
ALIAS_AD = f"alias.{uuid.uuid4().hex[:8]}.txt"
kos(f"""
open("/output/{ALIAS_AD}","w").write("ILK")
set_result("ok")
""", WF_C, inputs=[])
kos(f"""
open("/output/{ALIAS_AD}","w").write("IKINCI")
set_result("ok")
""", str(uuid.uuid4()), inputs=[])
surumler = [k["artifact_id"] for k in
            requests.get(f"{SERVIS}/artifacts", params={"name": ALIAS_AD},
                         headers=H, timeout=20).json()]
enyeni = requests.get(f"{SERVIS}/artifacts/by-name/{ALIAS_AD}", headers=H, timeout=20)
pa = requests.put(f"{SERVIS}/artifacts/{surumler[-1]}/alias",
                  params={"alias": "kabul"}, headers=H, timeout=20)
kontrol("alias atanıyor", pa.status_code == 200, pa.status_code)
ra = requests.get(f"{SERVIS}/artifacts/by-name/{ALIAS_AD}@kabul", headers=H, timeout=20)
kontrol("alias sürümü sabitliyor",
        ra.status_code == 200 and ra.headers.get("X-Artifact-Id") == surumler[-1],
        ra.status_code)
kontrol("alias 'en yeni' kuralından kaçırıyor",
        len(surumler) == 1 or ra.headers.get("X-Artifact-Id") != enyeni.headers.get("X-Artifact-Id"),
        f"{ra.headers.get('X-Artifact-Id')} vs {enyeni.headers.get('X-Artifact-Id')}")
kontrol("bozuk alias reddediliyor",
        requests.put(f"{SERVIS}/artifacts/{surumler[-1]}/alias",
                     params={"alias": "../etc"}, headers=H, timeout=20).status_code == 400)
r = kos(f"""
import os
yol = "/artifacts/_alias/{ALIAS_AD}"
set_result({{"yol": yol, "icerik": open(yol).read()}})
""", str(uuid.uuid4()), inputs=[f"{ALIAS_AD}@kabul"])
kontrol("sandbox alias BEYANIYLA okuyabiliyor",
        r.status.value == "success" and eval(r.result_text)["icerik"] == "ILK",
        r.error_message or r.result_text)

# ══ ÖZET ═════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
gecen = sum(1 for _, ok, _ in sonuclar if ok)
print(f"  {gecen}/{len(sonuclar)} KONTROL GEÇTİ")
kalan = [(a, d) for a, ok, d in sonuclar if not ok]
if kalan:
    print("\n  BAŞARISIZ:")
    for a, d in kalan: print(f"    ✗ {a}  — {d}")
print(f"{'═'*70}")
print(f"WF_A={WF_A}\nWF_B={WF_B}")
sys.exit(1 if kalan else 0)
