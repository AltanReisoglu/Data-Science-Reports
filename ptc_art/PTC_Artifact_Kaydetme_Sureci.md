# Bir Artifact Nasıl Kaydediliyor — CSV Örneğiyle Uçtan Uca

**Tarih:** 2026-09-03 · Ölçümler kind üzerinde canlı alınmıştır.

Bu doküman tek bir soruyu cevaplıyor: **LLM kod yazmaya başladığı andan bayt
diske düştüğü ana kadar tam olarak ne oluyor?** Örnek olarak bir CSV kullanıyor,
çünkü CSV üç farklı yoldan gidebiliyor ve aralarındaki fark tasarımın en
önemli kararlarını açığa çıkarıyor.

Mimarinin bütünü: [PTC_Mimari.md](PTC_Mimari.md)

---

## 0. Kim tetikliyor — TEK yol var (2026-09-06'da sadeleşti)

> **Bu bölüm 2026-09-06'da yeniden yazıldı.** Önceden iki tetikleyici vardı:
> LLM'in açıkça çağırdığı `put_artifact(...)` ve emniyet ağı olarak `/output`
> süpürmesi. **Açık API kaldırıldı** — piyasada emsali yoktu ve canlı
> kullanımda çıkan ciddi hataların hepsi o yüzeydeydi. Geriye tek yol kaldı:
> süpürme. Gerekçenin tamamı:
> [PTC_Piyasa_Mentaliteleri.md §11.11](PTC_Piyasa_Mentaliteleri.md).
>
> Aşağıdaki 12 adımın **9-12'si değişmedi** — servisten sonrası (denetim,
> dedup, iki depoya yazma, iz kaydı) birebir aynı. Değişen yalnızca 8. adım:
> "LLM `put_artifact` çağırır" yerine "süpürme dosyayı bulur".

Eskiden LLM'in API'yi **bilmesi** gerekiyordu. Bilmezse — model bunu unutursa
ya da hiç öğretilmemişse — ne olurdu?

Bunu SOTA'da araştırdık: **hiçbir kod-çalıştırma sistemi bu riski açık bir
tool'a bağlı bırakmıyor.** Anthropic'in `$OUTPUT_DIR`'ı, OpenAI Code
Interpreter'ın `/mnt/data`'sı, OpenHands'in workspace'i — hepsi aynı deseni
kullanıyor: **LLM sıradan kod yazar, çalışma bitince bir dizin süpürülür.**
(Tek istisna Cloudflare Code Mode — ama onun sandbox'ında dosya sistemi hiç
yok, süpürecek bir şey de yok.)

Cevabı zaten biliyoruz: **API'yi hiç sunmamak.** Artık bizde de tek tetikleyici
var ve o da onların kullandığı:

| | Tetikleyen | Ne zaman | Tip nereden |
|---|---|---|---|
| **Süpürme** | `entrypoint.py`'nin `main()`'i, `exec()` bittikten hemen sonra | Çalışma sonunda, **başarı ya da hata fark etmeksizin** | Dosya uzantısından (`.parquet` → `system.Dataset`) |

Bu, KFP launcher'ının yükleme adımının aynısı — orada da bileşen `.path`'e
yazar, launcher `.uri`'ye kopyalar.

```python
# main() — sandbox_image/entrypoint.py
try:
    exec(code, sandbox_globals)
except Exception as exc:
    _ciktilari_supur(launcher, inenler)            # ← hata olsa bile
    print(json.dumps({"status": "error", ...}))
    sys.exit(0)

_ciktilari_supur(launcher, inenler)                # ← başarı yolunda
print(json.dumps({"status": "success", ...}))
```

**Taranan yer `/output`** (emptyDir, `/scratch`ten AYRI). LLM
`df.to_parquet("/output/rapor.parquet")` yazar, gerisi olur — süpürme dosyayı
alıp servisten, aynı pickle/boyut/isim denetimlerinden geçirip artifact'e
çevirir. Üst düzey **dizinler** de alınır (tek tar olarak).

Aşağıdaki 12 adımın 8'incisi artık `_ciktilari_supur()`; 9-12 **birebir
aynı**. Fark yalnızca baytları kimin ürettiğinde: LLM'in kodu mu
(`serialize.serialize`), yoksa süpürme mi (dosyayı diskten akıtıp uzantıdan tip
tahmin ederek).

### 0.1 Okuma tarafındaki karşılığı (2026-09-04)

Yukarıdaki iki yol **yazmayı** anlatıyor. Okuma tarafında simetrik bir mekanizma
var: LLM `pd.read_csv("/output/rapor.csv")` yazdığında dosya `/output`'ta yoksa
ve o isim bu workflow'un manifestinde geçiyorsa, o an indiriliyor.

Manifest pod açılışında **tek istekle** çekiliyor (yalnızca isimler, bayt yok).
Bir önceki tasarımda pod doğarken depodaki *her* artifact indiriliyordu; maliyet
var olan her şeyle ölçekleniyordu ve 512Mi'lık `/output` altı büyük artifact
sonrası patlıyordu. Şimdi maliyet O(kullanılan).

İndirilen dosyanın (mtime, boyut) çifti kaydediliyor ve süpürme dokunulmamış
olanı **atlıyor** — yoksa sadece okuyan bir çalıştırma bile dosyayı "üretilmiş"
sayıp geri yüklerdi.

### 0.2 Taşıma: base64 değil, akış (2026-09-04)

Bu doküman ilk yazıldığında baytlar Tool Gateway'e MCP çağrısı içinde base64
gidiyordu. Artık ayrı bir servise ham bayt olarak akıyor:

- yükleme dosyadan doğrudan (`put_file`), belleğe alınmadan
- indirme parça parça, doğrudan diske (`fetch_to_file`)
- doğrulama **akış sırasında**: pickle ilk parçada (imza ilk iki bayttadır),
  boyut sayarak, sha256 birikerek

Gerekçe ve güvenlik sonuçları: `PTC_Mimari.md` §3.

---

## 1. Kuş bakışı — 12 adım

```
 LLM                    "CSV'yi oku, departmana göre topla"
  │
  ├─1─► run_ptc_code(code)                        graph.py
  │
  ├─2─► bütçe kontrolü (MAX_SANDBOX_RUNS_PER_TURN=2)
  │
  ├─3─► run_sandbox(code, workflow_id=...)        sandbox_runner.py
  │      ├─4─ run_id üret · ConfigMap'e kodu yaz
  │      ├─5─ Secret'tan imza anahtarını oku · kapsam jetonunu İMZALA
  │      └─6─ Job'u yarat (kod + jeton + gateway IP enjekte)
  │
  ├─7─► POD DOĞAR                                 entrypoint.py
  │      └─ exec(code) · SADECE set_result + tool'lar global
  │
  ├─8─► _ciktilari_supur() — /output taranır
  │      ├─ serialize()  → DataFrame ise Parquet, metin ise text/plain
  │      └─ ham bayt → POST /artifacts → Artifact Service
  │
  ├─9─► ARTIFACT SERVICE               services/artifact_service/app.py
  │      ├─ jetonun İMZASINI doğrula → kapsamı jetondan oku
  │      ├─ AKIŞ SIRASINDA: pickle (ilk parça) · boyut (sayarak) · sha256
  │      ├─ isim denetimi · aynı içerik var mı (dedup)
  │      ├─10─ MinIO'ya PUT ────────────────► PVC (baytlar)
  │      └─11─ SQLite'a INSERT ─────────────► PVC (kayıt defteri)
  │
  └─12─► "art_8dfee813f377" geri döner · Trace'e işlenir
```

---

## 2. Adım adım

### Adım 1-2 — Agent karar verir

LangGraph ajanı `run_ptc_code(code)` tool'unu çağırır. Bu, veriye erişmenin
**tek** yoludur; agent'a bağlı başka tool yoktur.

İlk kontrol bütçedir:

```python
if trace.sandbox_run_count() >= MAX_SANDBOX_RUNS_PER_TURN:   # = 2
    return "...sınıra ulaşıldı, YENİ bir sandbox çalıştırılmadı."
```

**İncelik:** bu sınır, artifact persistence'ın değerini belirleyen şey. Bir turda
iki atış hakkınız var; birincisi hata verirse ikincisi düzeltmedir. Pahalı
bloğun çıktısı `/output`'ta duruyorsa düzeltme onu **yeniden üretmez**:

```python
if os.path.exists("/output/tarama.parquet"):
    df = pd.read_parquet("/output/tarama.parquet")
else:
    df = pahali_tarama(); df.to_parquet("/output/tarama.parquet")
```

### Adım 3-6 — Çalıştırma hazırlanır

`run_sandbox` sırayla:

| # | İş | Not |
|---|---|---|
| 4 | `run_id = uuid4().hex[:12]` | Job adı, ConfigMap adı ve S3 anahtarı bundan türer |
| 4 | Kodu bir ConfigMap'e yaz (`ptc-code-{run_id}`) | **1 MiB sınırı** — kod bundan büyük olamaz |
| 5 | `ptc-scope-signing` Secret'ından anahtarı oku | Cluster API'siyle; sandbox bu Secret'ı göremez |
| 5 | `issue_token(anahtar, Scope(workflow_id, run_id, owner, node_id))` | HMAC-SHA256, 15 dk ömür |
| 6 | Gateway'in **ve Artifact Service'in** ClusterIP'lerini çöz | DNS adı değil — sandbox'ın DNS'e hiç ihtiyacı olmasın diye |
| 6 | `job-template.yaml`'ı doldur, Job'u yarat | `{run_id}`, `{tool_gateway_endpoint}`, `{artifact_service_endpoint}`, `{scope_token}`, `{workflow_id}` |

**İncelik:** kapsam jetonu **laptop'ta** imzalanıyor, yani sandbox'ın erişemediği
bir yerde. Sandbox'ın eline yalnızca imzalanmış sonuç geçiyor; imza anahtarı
hiç geçmiyor. Bu yüzden sandbox jetonu okuyabilir ama başka bir workflow için
geçerli bir jeton **üretemez**.

Jeton yoksa (`workflow_id` verilmemişse ya da Secret bulunamazsa) artifact API'si
sandbox'a **hiç sunulmaz**:

```python
istemci = ArtifactClient(ENDPOINT, SCOPE_TOKEN) if SCOPE_TOKEN and ENDPOINT else None
artifact_globals, artifact_internal = _artifact_api(istemci) if istemci else ({}, {})
```

Kapsamı doğrulanamayan bir çalıştırmanın kalıcı depoya yazması, çalıştırmalar
arası sınırı tümden kaldırmak olurdu.

### Adım 7 — Pod doğar

`entrypoint.py` `/sandbox/code.py`'yi okur ve global ortamı kurar: tool proxy'leri
+ `set_result` + (jeton varsa) beş artifact fonksiyonu. Sonra:

```python
exec(compile(code, CODE_PATH, "exec"), sandbox_globals)
```

Tek `exec`, sonra süreç ölür. Loop yok — bu bir REPL değil, script çalıştırıcı.

Pod'un yazılabilir iki yeri var, ikisi de emptyDir (512Mi) ve ikisi de **pod ile
birlikte ölür**: `/scratch` (geçici, süpürülmez) ve `/output` (süpürülür —
içindekiler artifact'e döner). Kalması gereken her şey artifact deposuna gitmek
zorunda; `/output`'a yazmak bunun kısa yolu.

### Adım 8 — Süpürme dosyayı bulur ve akıtır

`exec()` bittiği anda (hata almış olsa bile) `/output`'un üst düzeyi taranır:

```python
for ad in sorted(_GERCEK_LISTDIR(OUTPUT_DIR)):
    yol = os.path.join(OUTPUT_DIR, ad)
    if os.path.isdir(yol):
        _dizini_supur(...)                # dizin → tek tar
        continue
    api["_put_file"](yol,
                     serialize.content_type_for_filename(ad),   # uzantıdan tip
                     _gecerli_artifact_adi(ad))
# → POST /artifacts, gövde AKIŞLI dosya, kapsam X-Scope-Token başlığında
```

Yani "süpürme" fiilen **bucket'a kopyalamadır**, arada tek bir kapı vardır:

```
/output/rapor.parquet ──HTTP akış──> Artifact Service ──S3 PUT──> bucket
```

KFP'nin launcher'ı bu kopyalamayı doğrudan S3'e yapıyor (anahtar pod'da);
bizde servis yapıyor, sandbox'ta S3 anahtarı hiç yok.

**İncelik — tip nereden geliyor:** artık `serialize()` değil
`content_type_for_filename()` — yani **dosya uzantısı**. `.parquet` →
`system.Dataset`, `.png` → `system.Artifact`. Uzantısız ad verilirse tip
`application/octet-stream`'e düşer; sistem promptu bu yüzden "uzantısız ad
verme" diyor.

**İncelik — serviste `pandas` YOK.** Servis hazır baytları alır ve onları hiç
ayrıştırmaz. Bu, "LLM'in ürettiği veriyi çözmesi
gerekmeyen katman" olduğunu yapısal olarak garanti eder. (Pickle reddi bayt
imzasına bakar, ayrıştırmaya değil.)

**İncelik — log kirliliği artık yok:** 2026-09-04 öncesinde baytlar MCP
çağrısında `content_b64` olarak gidiyordu ve tool-çağrısı log satırına olduğu
gibi yazılsaydı, `_wait_and_stream` her turda tüm log'u yeniden okuduğu için
çalıştırma fiilen kilitlenirdi (log'a `<N bayt base64>` yazılıyordu). Artifact
yolu MCP'den çıktığı için bu sorun kaynağında ortadan kalktı — baytlar artık
tool çağrısı log'una hiç uğramıyor.

### Adım 9 — Servis denetler

Sırayla, ve **hepsi reddetme sebebidir**:

| Kontrol | Kod | Reddedince |
|---|---|---|
| Jeton imzası + süresi | `verify_token` | `InvalidScopeToken` |
| İsim biçimi | `_isim_dogrula` | `InvalidArtifactName` — `/`, `..`, boşluk yasak |
| Format | `guvenlik_kontrolu` | `UnsafeArtifact` — pickle |
| Boyut | `size_limit` (100 MiB) | `ArtifactTooLarge` |

Sonra:

```python
content_hash = "sha256:" + hashlib.sha256(data).hexdigest()
artifact_id  = "art_" + uuid4().hex[:12]

ikiz = self.metadata.find_by_hash(workflow_id, content_hash)
if ikiz is not None:
    storage_uri = ikiz.storage_uri          # DEDUP: bayt tekrar yüklenmez
else:
    storage_uri = self.objects.put(key, data, content_type)
```

**İncelik — dedup kimliği çoğaltmaz.** Aynı içerik ikinci kez yazılırsa bayt
tekrar yüklenmez ama **yeni bir `artifact_id` üretilir**. "Aynı içerik" ile
"aynı artifact" farklı şeylerdir; artifact'ler değişmez, bir kayıt asla
ezilmez.

### Adım 10-11 — İki ayrı depoya yazılır

```
MinIO  →  altan/wf_csv_1788439680/csv/8f3a.../art_8dfee813f377.parquet
SQLite →  artifact_id · name · workflow_id · node_id · run_id ·
          content_hash · content_type · size_bytes · storage_uri ·
          parents · owner · created_at · ttl_seconds
```

**İncelik:** ikisi ayrı olmasa elde artifact store değil sadece bir bucket olur.
Lineage, içerik hash'i, TTL, kim üretti — bunların hiçbiri bir S3 anahtarında
yaşayamaz.

### Adım 12 — Geri dönüş ve iz

Sandbox'a **yalnızca özet** döner; `storage_uri` bilerek yoktur. Sandbox
`art_8dfee813f377`'den başka bir şey görmez.

Sandbox iki satır stdout'a yazar: `tool_call` (ne çağrıldı) ve `artifact`
(hangi veri nereye gitti). Runner ikincisini `SandboxRun.artifacts`'a,
`graph.py` de `Trace`'e işler.

---

## 3. CSV örneği — üç yol, ölçülmüş

Aynı CSV içeriği (`ticket,departman,acik` + 3 satır) üç şekilde saklandı:

```python
csv_metni = "ticket,departman,acik\nT-1,BT,3\nT-2,IK,1\nT-3,BT,5\n"

df = pd.read_csv(io.StringIO(csv_metni))
df.to_parquet("/output/csv.dataframe.parquet")          # A
open("/output/csv.metin.txt", "w").write(csv_metni)     # B
df.to_csv("/output/csv.tablo.csv", index=False)         # C
```

*(Ölçüm 2026-09-04'te `put_artifact` ile yapılmıştı; API kalktı, ama içerik
ve boyutlar aynı — değişen yalnızca çağrının biçimi. Tip artık uzantıdan
geliyor: `.parquet` → Dataset, `.txt` → Artifact, `.csv` → Dataset.)*

**Yazma sonucu:**

| Yol | content_type | Boyut |
|---|---|---|
| **A — DataFrame** | `application/vnd.apache.parquet` | **2639 bayt** |
| **B — metin** | `text/plain` | 49 bayt |
| **C — bayt** | `application/octet-stream` | 49 bayt |

**Geri okuma sonucu — asıl fark burada:**

| Yol | Geri gelen tip | dtype'lar | Hemen kullanılabilir mi |
|---|---|---|---|
| **A** | `DataFrame` | `ticket: str`, `departman: str`, **`acik: int64`** | **Evet** — `df["acik"].sum()` → 9 |
| **B** | `str` | — | Hayır, yeniden parse gerekir |
| **C** | `bytes` | — | Hayır, önce decode sonra parse |

### Bundan çıkan üç ders

**1. Tip bilgisi yolda kaybolur ya da korunur — seçim burada yapılır.**
A yolunda `acik` sütunu `int64` olarak geri geliyor; sonraki adım doğrudan
toplayabiliyor. B ve C'de sonraki adım tipleri **yeniden tahmin etmek**
zorunda. Bir sonraki adımın bunu doğru tahmin edeceğini varsaymak, kalıcılığın
amacına aykırı.

**2. Parquet küçük veride DAHA BÜYÜK — ve bu normal.**
3 satır için 2639 vs 49 bayt, yani 54 kat. Parquet şema, sütun metadata'sı ve
sıkıştırma blokları taşıyor. Kazanç binlerce satırdan sonra başlar. Küçük ve
yapısız çıktılar için metin/JSON doğru tercihtir — `serialize()` zaten
DataFrame olmayan her şeyi oraya yönlendiriyor.

**3. `serialize()` sessizce string'e düşürmez.**
JSON yolunda bilerek `default=str` **yok**. Serileştirilemeyen bir nesne hata
verir; çağıran açık bir karar vermek zorunda kalır. Sessiz tip kaybı,
sonraki adımda teşhisi zor bir hataya dönüşür.

---

## 4. Ham CSV dosyası saklamak isteniyorsa

Bazen amaç dataframe değil, **dosyanın kendisidir** (rapor eki, ham kaynak,
denetim kopyası). O zaman C yolu doğru:

```python
import shutil
shutil.copy("/scratch/rapor.csv", "/output/rapor.ham.csv")
```

Kural: **uzantı tipi belirler.** `.csv` verirsen `text/csv` + `system.Dataset`
olur ve geri okunduğunda doğru açılır; uzantısız bırakırsan
`application/octet-stream`'e düşer ve tüketen taraf tipi bilmek zorunda kalır.

---

## 5. Bu yolda neler ters gidebilir

| Nerede | Ne olur | Sonuç |
|---|---|---|
| Adım 2 | Tur bütçesi dolmuş | Pod hiç yaratılmaz |
| Adım 4 | Kod > 1 MiB | ConfigMap reddeder |
| Adım 5 | Secret yok | Artifact API sandbox'a hiç sunulmaz |
| Adım 8 | Değer serileştirilemiyor | Sandbox içinde `TypeError` |
| Adım 9 | Jeton bozuk/süresi dolmuş | `InvalidScopeToken` |
| Adım 9 | İsimde `/` veya `..` | `InvalidArtifactName` |
| Adım 9 | Baytlar `\x80` ile başlıyor | `UnsafeArtifact` — pickle |
| Adım 9 | > 100 MiB | `ArtifactTooLarge` |
| Adım 10 | Gateway MinIO'ya çıkamıyor | **Ağ politikasında onaylı hedef yoksa asılı kalır** |
| B yolu (süpürme) | `/output`'taki bir dosya adım 9'da reddedilir | O dosya **atlanır** (`artifact_skipped` olayı), diğerleri ve asıl sonuç etkilenmez |

Son satır dışında yaşanmış bir olay daha var: artifact deposu eklendiğinde
gateway'in egress politikasında MinIO yoktu ve çağrı sessizce asılı kaldı
(adım 10). Projenin kendi ilkesi işledi — *her yeni dış erişim açıkça
onaylanmalı.* Artifact deposu bir istisna değil, onaylanması gereken yeni bir
kanaldır.

**B yolunun satırı da doğrulanmış bir olaydır**, farklı türden: süpürme
`/output`'ta bir pickle dosyası bulup servise gönderdiğinde adım 9'daki
`UnsafeArtifact` reddi tetiklendi. İlk denemede bu, `_wait_and_stream`'in
tanımadığı bir `"op": "skipped"` değeri yüzünden **koca çalıştırmayı
çökertti** — güvenlik kontrolünün kendisi değil, onu diğer katmana bildiren
protokoldü hatalı. Düzeltme: skip olayı `ArtifactEvent`'e hiç çevrilmiyor,
ayrı bir `"type": "artifact_skipped"` olarak yalnızca gözlemlenebilirliğe
gidiyor. Ders: emniyet ağının kendisi kadar, **başarısızlığını nasıl
raporladığı** da test edilmeli.
