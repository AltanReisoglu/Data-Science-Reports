# PTC — Artifact İnceleme Komutları

Depoda ne var, kim üretti, kim okudu — hepsine terminalden bakmanın yolu.
Bütün komutlar 2026-09-07'de canlı cluster'da (`kind-ptc-sec`) çalıştırılıp
doğrulandı.

> Kardeş dosyalar: [PTC_Komut_Referansi.md](PTC_Komut_Referansi.md) (sıfırdan
> kurulum, Cilium, Hubble) · [PTC_Calisma_Komutlari.md](PTC_Calisma_Komutlari.md)
> (günlük servis başlatma). Bu dosya yalnızca **artifact tarafını** anlatıyor.

**Önce bilinmesi gereken:** `/output` bir `emptyDir` — pod'la birlikte ölüyor,
orada kalıcı hiçbir şey yok. Bakılacak iki yer var:

| Nerede | Ne tutuyor |
|---|---|
| **MinIO** (`ptc-artifacts` bucket) | baytlar |
| **SQLite** (`artifact-service`) | künye: ad, sahip, soy, hash, alias, TTL |

İkisinin sayısı eşit olmak zorunda değil — içerik-hash dedup'ı yüzünden aynı
bayt tek kez saklanıyor, birden çok künye onu gösterebiliyor.

---

## 0 · Tek seferlik kurulum

Her yeni terminalde bir kere yapıştır:

```bash
cd /home/altan/Desktop/Data-Science-Reports/ptc_art
export PY=/home/altan/Desktop/Data-Science-Reports/ptc_sec/.venv/bin/python

# Kayıt defteri ve MinIO'yu localhost'a aç (arka planda kalsınlar)
kubectl port-forward svc/artifact-service 8080:8080 &
kubectl port-forward svc/minio 9000:9000 &

# Kapsam jetonu üreten kısayol — servis jetonsuz 401 döndürüyor
jeton() { $PY -c "
from dotenv import load_dotenv; load_dotenv()
from grounded_assistant.agent.graph import _kapsam_jetonu
print(_kapsam_jetonu('bakis') or '')"; }
art() { curl -s -H "X-Scope-Token: $(jeton)" "http://127.0.0.1:8080$1"; }
```

Kontrol:

```bash
curl -s http://127.0.0.1:8080/healthz     # {"status":"ok"}
```

**Jeton neden gerekiyor:** servis HMAC imzalı bir kapsam jetonu istiyor; imza
anahtarı cluster'daki `ptc-scope-signing` Secret'ında. `jeton()` onu okuyup
imzalıyor. Jetonsuz her istek 401.

---

## 1 · İçini görmek — en kısa yol

Depodaki baytların çoğu ikili: parquet, pdf, png, tar. `curl | json.tool`
çalışmıyor, `cat` ekranı bozuyor. Bu betik içerik tipine bakıp doğru biçimde
açıyor:

```bash
PY=/home/altan/Desktop/Data-Science-Reports/ptc_sec/.venv/bin/python

$PY scripts/artifact_bak.py --liste                     # depoda ne var
$PY scripts/artifact_bak.py departman_ozet.parquet      # ada göre
$PY scripts/artifact_bak.py art_79f0                    # kısaltılmış id yeter
$PY scripts/artifact_bak.py rapor.pdf@onaylanmis        # alias'la sabit sürüm
$PY scripts/artifact_bak.py art_79f0 --kunye            # yalnızca künye
$PY scripts/artifact_bak.py art_79f0 --kaydet /tmp/x    # ham baytlar
```

Örnek çıktı:

```
── KÜNYE ────────────────────────────────────────────────────
artifact_id    art_79f039144ecc
name           departman_ozet.parquet
type           system.Dataset
content_type   application/vnd.apache.parquet
size_bytes     2343
content_hash   sha256:d0d3cf4251d3ed23ec4...
workflow_id    fcc91447-8828-41c9-8025-ac7a1ae9bb0a
soy            0 ebeveyn

── İÇERİK ───────────────────────────────────────────────────
satır 5 · sütun ['departman', 'ortalama_cozum_saati']

   departman  ortalama_cozum_saati
0     Finans             41.432895
1         IK             40.785500
...
```

Tipe göre ne yapıyor:

| İçerik | Ne gösteriyor |
|---|---|
| `.parquet` | pandas tablosu, ilk 30 satır + sütun listesi |
| `.json` | girintili JSON |
| `.csv` · `.md` · `.txt` | ilk 30 satır |
| `.tar` *(dizin artifact'i)* | içindeki dosyalar + boyutları |
| `.pdf` · `.png` *(ikili)* | `/tmp`'ye kaydeder, `xdg-open` komutunu yazar |

Tek ön koşul: `kubectl port-forward svc/artifact-service 8080:8080`.
Jetonu betik kendisi üretiyor.

> Elle uğraşmak istersen aşağıdaki bölümler ham uçları anlatıyor.

---

## 2 · Kayıt defteri — asıl bakılacak yer

### Son 20 artifact, tek satır tek kayıt

```bash
art "/artifacts?limit=20" | $PY -c "
import json,sys
for a in json.load(sys.stdin):
    print(f\"{a['artifact_id']}  {a['name'][:34]:<34} {a['size_bytes']:>9}B  {a['created_at'][:19]}  wf={a['workflow_id'][:8]}  parents={len(a['parents'])}\")"
```

Örnek çıktı:

```
art_22bd62302814  final-report.pdf                       14177B  2026-09-07T12:59:30  wf=4ca8de78  parents=1
art_004f4f254198  processed-result.json                     78B  2026-09-07T12:59:27  wf=4ca8de78  parents=1
art_91f5e797b287  dagilim.png                            10640B  2026-09-07T12:59:27  wf=4ca8de78  parents=1
```

### Tam JSON — bütün alanlar

```bash
art "/artifacts?limit=3" | $PY -m json.tool
```

### Süzgeçler — MLMD'nin `filter_query`'sinin karşılığı

```bash
art "/artifacts?name=processed-result.json" | $PY -m json.tool   # tam ad
art "/artifacts?type=system.Dataset" | $PY -m json.tool          # tipe göre
art "/artifacts?q=rapor" | $PY -m json.tool                      # ad içinde ara
```

### Tek bir workflow'un çıktıları

```bash
art "/workflows/<workflow_id>/artifacts" | $PY -m json.tool
```

### Bir artifact'in künyesi / içeriği / soyu

```bash
art "/artifacts/art_004f4f254198/metadata" | $PY -m json.tool
art "/artifacts/art_004f4f254198"                        # ham baytlar
art "/artifacts/art_22bd62302814/lineage" | $PY -m json.tool
```

Soy çıktısında her düğümde `depth` ve `yon` (`kok` / `ata` / `torun`) var —
konsoldaki soy ağacı sayfası da bunu çiziyor.

### Ada göre çözme + alias (MLflow deseni)

İki uç var ve karıştırılmaya müsait:

| Uç | Ne döndürür |
|---|---|
| `/artifacts/by-name/<ad>` | **ham baytlar** — `json.tool`'a verme, dosyaya yaz |
| `/artifacts/by-name/<ad>/uri` | künye (JSON): `artifact_id`, `storage_uri`, boyut |

```bash
art "/artifacts/by-name/rapor.pdf/uri" | $PY -m json.tool     # kim seçildi
art "/artifacts/by-name/rapor.pdf" > /tmp/rapor.pdf           # baytlar
```

Alias varsa `@` ile sabit sürüme gidiliyor, yoksa **en yeni** kazanıyor:

```bash
art "/artifacts/by-name/rapor.pdf@onaylanmis/uri" | $PY -m json.tool
```

**Alias `alias=` QUERY parametresiyle atanıyor — JSON gövdesiyle değil.**
Gövde sessizce yok sayılır ve `alias` `null`'a düşer, yani var olan alias'ı
*kaldırır*:

```bash
# ata
curl -s -X PUT -H "X-Scope-Token: $(jeton)" \
     "http://127.0.0.1:8080/artifacts/art_d9cd38333b93/alias?alias=onaylanmis"
# → {"artifact_id":"art_d9cd38333b93","alias":"onaylanmis"}

# kaldır (parametreyi hiç verme)
curl -s -X PUT -H "X-Scope-Token: $(jeton)" \
     "http://127.0.0.1:8080/artifacts/art_d9cd38333b93/alias"
# → {"artifact_id":"art_d9cd38333b93","alias":null}
```

Alias biçimi: `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$` — uymayan 400 alıyor.

---

## 3 · MinIO — baytlar gerçekten orada mı

MinIO container'ı minimal; içinde `find`, `mc` gibi araç **yok**. S3 API'sinden
bakılıyor:

```bash
$PY -c "
from minio import Minio
c = Minio('127.0.0.1:9000', access_key='ptcartifacts',
          secret_key='ptcartifacts-local-dev', secure=False)
o = list(c.list_objects('ptc-artifacts', recursive=True))
print(f'toplam nesne: {len(o)}')
for x in sorted(o, key=lambda x: x.last_modified)[-20:]:
    print(f'  {x.size:>9} {x.last_modified:%m-%d %H:%M}  {x.object_name}')"
```

Kimlik bilgisi `artifact-bucket` Secret'ında (ObjectBucketClaim sözleşmesi):

```bash
kubectl get secret artifact-bucket -o jsonpath='{.data.AWS_ACCESS_KEY_ID}' | base64 -d; echo
kubectl get cm artifact-bucket -o jsonpath='{.data}'; echo   # BUCKET_NAME, BUCKET_HOST…
```

**Anahtar deseni** — KFP'nin `pipeline_root`'unun aynısı, çalıştırma başına izole:

```
ptc/<workflow_id>/<node_id>/<run_id>/<artifact_id>.<uzantı>
```

`node_id` yoksa `_` yazılıyor (sohbet turu gibi düğümsüz çalıştırmalar).

---

## 4 · SQLite — defterin kendisi

Container'da `sqlite3` CLI yok, dosya dışarı kopyalanıyor:

```bash
kubectl cp default/$(kubectl get pod -l app=artifact-service \
  -o jsonpath='{.items[0].metadata.name}'):/var/lib/ptc/artifacts.db /tmp/artifacts.db

$PY -c "
import sqlite3
c = sqlite3.connect('/tmp/artifacts.db')
print('kayıt:', c.execute('SELECT COUNT(*) FROM artifacts').fetchone()[0])
for r in c.execute('''SELECT artifact_id,name,size_bytes,alias,workflow_id
                      FROM artifacts ORDER BY created_at DESC LIMIT 15'''):
    print(r)"
```

Tablo tek: `artifacts`. Sütunlar:

```
artifact_id  name  workflow_id  node_id  run_id  content_hash  content_type
size_bytes   storage_uri  parents  owner  created_at  ttl_seconds
artifact_type  user_metadata  alias
```

Sık işe yarayan sorgular:

```bash
$PY -c "
import sqlite3, collections
c = sqlite3.connect('/tmp/artifacts.db')
print('— aynı içerik kaç künyede (dedup) —')
for h,n in c.execute('''SELECT content_hash, COUNT(*) c FROM artifacts
                        GROUP BY content_hash HAVING c>1 ORDER BY c DESC LIMIT 5'''):
    print(f'  {n}x  {h[:26]}…')
print('— workflow başına artifact —')
for w,n in c.execute('''SELECT workflow_id, COUNT(*) c FROM artifacts
                        GROUP BY workflow_id ORDER BY c DESC LIMIT 5'''):
    print(f'  {n:>4}  {w}')"
```

---

## 5 · Cluster durumu

```bash
kubectl get pods                       # servisler + biten sandbox job'ları
kubectl get jobs                       # çalıştırmalar
kubectl logs -l app=artifact-service --tail=40
kubectl get cnp                        # Cilium ağ politikaları
```

Bir sandbox pod'unun **iki container'ı** ayrı ayrı okunuyor — süpürme ve
yerleştirme sidecar'da, LLM'in kodu sandbox'ta:

```bash
kubectl logs <pod-adı> -c sandbox              # LLM'in yazdığı kod
kubectl logs <pod-adı> -c artifact-sidecar     # yerleştirme + süpürme
```

---

## 6 · Konsol — görsel

```bash
$PY -m uvicorn grounded_assistant.web.app:app --port 8123
```

→ <http://127.0.0.1:8123/konsol>

Beş sekme: **Sohbet · Hatlar · Çalıştırma · Depo · Soy**. Depo ve Soy sekmeleri
yukarıdaki uçların aynısını çağırıyor; sahte veri yok.

### Hatlar

Dört yerleşik hat var, hepsi gerçek pod açıyor:

| | Ne gösteriyor |
|---|---|
| **PL-A** Ticket İşleme | dört adımlık üretim zinciri |
| **PL-B** Artifact Analiz | çapraz workflow: A'nın çıktısını defterden bulup BEYAN eder |
| **PL-C** Sürüm Sabitleme | en eskiyi `@konsol-sabit` ile sabitler, sonra `ad@alias` beyanıyla okur |
| **PL-D** Dizin ve Dedup | dizin artifact'i (tar) + aynı içerik iki ad → tek nesne |

Üç adım TÜRÜ var; ikisi pod açmıyor:

```
sandbox  gerçek PTC pod'u
query    kayıt defteri sorgusu        — pod YOK
alias    sürüm sabitleme (PUT alias)  — pod YOK
```

**Kendi hattınızı kurmak:** Hatlar sekmesinde *＋ Yeni hat kur*. Adım ekleyip
türünü seçiyorsunuz; `sandbox` adımına Python, `query`/`alias` adımına
aranacak ad giriyorsunuz. Kaydedilenler `var/konsol-hatlari.json`'da duruyor
(gitignore'da — çalışma zamanı durumu). Yerleşik dördü silinemez.

Terminalden de kurulabiliyor:

```bash
curl -s -X POST http://127.0.0.1:8123/api/pipelines \
  -H "Content-Type: application/json" -d '{
  "key":"ornek","ad":"Örnek Hat","aciklama":"Konsoldan kuruldu.",
  "nodes":[
    {"ad":"Adıyla İste","tur":"query","sorgu_ad":"processed-result.json",
     "tercih_alias":"konsol-sabit"},
    {"ad":"Oku","tur":"sandbox",
     "kod":"import json\nv=json.load(open(\"/artifacts/{kaynak_wf}/processed-result.json\"))\nset_result(v)",
     "inputs":["{kaynak_wf}/processed-result.json"]}]}'

curl -s http://127.0.0.1:8123/api/pipelines | $PY -c "
import json,sys
for h in json.load(sys.stdin)['pipelines']:
    print(h['kod'], h['ad'], len(h['nodes']), 'adım', '(konsoldan)' if h.get('kullanici') else '')"

curl -s -X DELETE http://127.0.0.1:8123/api/pipelines/ornek
```

`{kaynak_wf}` bir yer tutucu: `query`/`alias` adımının bulduğu workflow
kimliği çalışma anında yerine konur. Hem beyanda hem kod içinde geçerli.

**Alias ADIYLA isteniyor** (`tercih_alias`), "sabitlenmiş olanı ver" diye bir
seçenek yok — aynı ada iki alias konabildiği için o kural tanımsız kalıyordu.
Boş bırakılırsa en yeni kazanır ve log, istenmemiş sabit sürümleri söyler.

---

## 7 · Ürünü baştan sona sınamak

```bash
# Ön koşul: port-forward 8080 + panel 8123 ayakta olmalı
kubectl port-forward svc/artifact-service 8080:8080 &
$PY -m uvicorn grounded_assistant.web.app:app --port 8123 &

$PY scripts/kabul_testi.py                          # proxy kipi → 52/52

# direct kip: pod→MinIO rotasını açan AYRI politika gerekiyor
kubectl apply -f k8s/policies/sandbox-egress-direct.ciliumnetworkpolicy.yaml
PTC_ARTIFACT_TRANSFER=direct $PY scripts/kabul_testi.py    # → 53/53
kubectl delete -f k8s/policies/sandbox-egress-direct.ciliumnetworkpolicy.yaml
```

Birim/entegrasyon testleri (cluster gerekmez):

```bash
$PY -m pytest -q                                     # 209 test
```

**`direct` kipinde kontrol sayısı neden bir fazla:** o kipte sandbox'ın MinIO'ya
rotası açılıyor (NetworkPolicy **pod** seçiyor, container değil) ve bu bedel
ölçülüp yazılıyor — `sandbox → minio` `ULASILDI` döndürmek *zorunda*. `proxy`
kipinde aynı sonda `TimeoutError` bekliyor.

---

## 8 · Sunum ve diyagramları yeniden üretmek

```bash
$PY scripts/diyagram_uret.py       # docs/diyagram/*.png
node scripts/sunum_uret.js         # PTC_Sunum_Karsilastirma.pptx
```
