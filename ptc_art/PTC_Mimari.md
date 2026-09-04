# PTC Artifact Persistence — Mimari

**Tarih:** 2026-09-03 · **Durum:** çalışır PoC, kind üzerinde doğrulanmış

Bu doküman, inşa edilmiş sistemi baştan sona anlatır. Neyi neden seçtiğimizin
gerekçeleri [PTC_Hedef_Mimari_ve_Karar_Dokumani.md](PTC_Hedef_Mimari_ve_Karar_Dokumani.md)'da,
dış kaynak taraması [PTC_Artifact_Persistence_Arastirmasi.md](PTC_Artifact_Persistence_Arastirmasi.md)
ve [PTC_Calistirma_Ortami_ve_SOTA_Arastirmasi.md](PTC_Calistirma_Ortami_ve_SOTA_Arastirmasi.md)'da.

---

## 1. Tek cümlelik tez

> **Sandbox = hesaplama ve geçici alan. Artifact store = durable çalışma ürünü
> deposu. Artifact'in kalıcılığı sandbox'ın yaşam döngüsüne bağlanmaz.**

Bir PTC çalıştırması ara ve nihai çıktılarını artifact deposuna bırakır; pod
silinir; **sonraki** bir PTC çalıştırması o çıktıyı yeniden üretmeden okur.

---

## 2. Çözdüğü problem

PTC'de LLM tek bir script yazar, script tool'ları çağırır. Sorun script
patladığında ortaya çıkar:

```
Adım 1  tickets = 40 tool çağrısı        ← pahalı
Adım 2  enriched = join(...)
Adım 3  summary  = groupby(...)
Adım 4  rapor                             ← KeyError
```

Kalıcılık yoksa düzeltilmiş sürüm 1-3'ü **baştan** koşar. Üstelik
`MAX_SANDBOX_RUNS_PER_TURN = 2` olduğu için düzeltme hakkı tektir.

"Çok adımlı workflow" üç ayrı şey demek olabiliyor ve üçü farklı çözüm istiyor:

| Okuma | Ne | Çözüm |
|---|---|---|
| **A** | Tek script içindeki aşamalar; hata → baştan | İçerik-adresli **cache** (`cached`) |
| **B** | Aynı turda ardıl çalıştırmalar (self-repair) | İsimle **handle** (`put`/`get`) |
| **C** | Turlar / oturumlar / node'lar arası | Handle + **keşif** (`list`) |

PTC her şeyi tek script'e ittiği için pratikte en çok **A** işe yarıyor.

---

## 3. Kuş bakışı

```
┌── Sandbox Pod ── her çalıştırmada YENİ, sonunda silinir ──────────┐
│  /sandbox  configMap  → LLM'in kodu (salt okunur)                 │
│  /scratch  emptyDir   → geçici (pod ile ölür, ölmesi İSTENİR)     │
│  /output   emptyDir   → süpürülür + tembel doldurulur             │
│  kalıcı disk YOK                                                  │
│                                                                   │
│  put_artifact(df, name="tickets")   ← Parquet'e çevirir, ham bayt │
│  get_artifact(name="tickets")                                     │
│  list_artifacts(node_id=...)                                      │
│  cached("tickets", pahali_fn)                                     │
│  df.to_csv("/output/x.csv")   ← API'yi hiç bilmeden de olur       │
└──────────┬─────────────────────────────────┬──────────────────────┘
           │ MCP/HTTP                        │ akışlı HTTP
           │ (yalnızca tool'lar)             │ (yalnızca artifact)
┌──────────▼──────────┐          ┌───────────▼────────────┐
│    Tool Gateway     │          │   Artifact Service     │
│  search_kb, web_... │          │  • kapsam jetonu       │
│                     │          │  • isim / mime / boyut │
│  internet ✓         │          │  • pickle reddi        │
│  depo     ✗         │          │  • sha256 + dedup      │
└─────────┬───────────┘          └───┬────────────────┬───┘
          │ 3 onaylı FQDN   S3 API   │                │ SQLite
          ▼                ┌─────────▼──────┐  ┌──────▼──────────┐
      internet             │  MinIO pod     │  │ metadata PVC    │
                           │  └ PVC 5Gi     │  │ 1Gi             │
                           │  BAYTLAR       │  │ KAYIT DEFTERİ   │
                           └────────────────┘  └─────────────────┘
                                    ▲
                        internet ✗  ┘  (Artifact Service'in FQDN'i yok)
```

**İki depo, iki ayrı iş.** Kubeflow'un `artifact store` + `metadata store`
ayrımının küçük hâli: baytlar bir yerde, "o baytın hikâyesi" başka yerde.
Bu ayrım olmadan elde artifact store değil, sadece bir bucket olur.

**İki servis, iki ayrı yetenek** (2026-09-04). Artifact işi Tool Gateway'den
ayrı bir servise çıkarıldı. İki gerekçe:

- **Taşıma.** Baytlar MCP çağrısının içinde base64 gidiyordu — %33 şişme ve iki
  uçta tam tampon. 100 MiB sınırı da aslında bunun doğal tavanıydı. Şimdi ham
  bayt akıyor.
- **Yetki ayrımı.** Tek pod üç işi birden yapıyordu: tool proxy'si, kayıt
  defteri, MinIO kimlik bilgisi taşıyıcısı. Artık gateway'in internete çıkışı
  var deposu yok, servisin deposu var internete çıkışı yok. Tek bir workload'ın
  ele geçirilmesi ikisini birden vermiyor — OpenAI'nin Ağustos 2026 olayında
  Artifactory'nin oynadığı köprü rolüne karşı yapısal önlem.

Doğrulandı (2026-09-04): gateway pod'undan `minio:9000` bağlantısı zaman
aşımına düşüyor ve pod'un ortamında `AWS_*`/`BUCKET_*` değişkenlerinden hiçbiri
yok — rota da sır da alındı.

---

## 4. Katmanlar

### 4.1 Sandbox — hesaplama

Her `run_ptc_code` çağrısı bir Kubernetes Job. Pod doğar, kodu çalıştırır, ölür.

| Volume | Cins | Ömür |
|---|---|---|
| `/sandbox` | configMap | Salt okunur, Job ile silinir |
| `/scratch` | emptyDir (512Mi) | **Pod ile ölür — bilerek** |

| `/output` | emptyDir (512Mi) | Süpürülür — içeriği artifact'e döner |

Serileştirme **burada** yapılır: bir dataframe ağdan nesne olarak geçemez,
Parquet'e çevrilip ham bayt olarak gider. Bu yüzden `pandas`/`pyarrow` sandbox
imajına gömülüdür — sandbox'ın çalışma anında ağ erişimi yok, `pip install`
yapamaz. (Anthropic'in code execution container'ı da aynı kısıtla çalışıyor.)

`serialize.py` ana kaynak ağacındaki dosyanın **ta kendisidir**, Dockerfile
tarafından kopyalanır — format kuralları ve pickle yasağı tek yerde tanımlı
kalsın, sandbox ile servis aynı kurallara uysun diye.

### 4.2 Artifact Service — yetkilendirme ve doğrulama

Artifact baytlarının tek geçtiği yer. §27'nin REST API'sini sunar
(`POST /artifacts`, `GET /artifacts/{id}`, `GET /artifacts/by-name/{ad}`,
`GET /artifacts/{id}/metadata`, `GET /workflows/{id}/artifacts`,
`DELETE /artifacts/{id}`) ve dört kontrol uygular:

| Kontrol | Neden |
|---|---|
| **Kapsam** — jetondan okunur | Paylaşılan depo, ağ politikasının GÖREMEDİĞİ bir çalıştırma-arası kanal açar |
| **İsim** — yol geçişi yok | `artifact_save("/etc/shadow")` sınıfı |
| **Boyut** — üst sınır | Veriyi sandbox içinde süzmeye zorlar |
| **Format** — pickle reddi | Deserialization kod çalıştırır (CWE-502); baytları LLM'in kodu yazdı |

Dördü de **akış sırasında** uygulanır, sonunda değil: pickle kararı ilk parçada
verilir (imza ilk iki bayttadır) ve reddedilen yükleme depoya tek bayt bile
yazmaz; boyut sayılarak kesilir; sha256 akış boyunca birikir. Gövde 8 MiB'ı
aşınca diske taşar, yani süreç belleği yükleme boyutundan bağımsız kalır.

`DELETE` sandbox'ın istemci kütüphanesinde **yoktur** — TTL reaper'ı içindir.
LLM'in ürettiği kodun artifact silebilmesi, emniyet ağı olarak kurduğumuz
kalıcılığı tek satırda geri alabilirdi.

Serviste `pandas`/`pyarrow` **bilerek yok**. Hazır baytlar gelir, servis onları
çözmez — bu hem imajı hafif tutar hem de LLM'in ürettiği veriyi hiç
ayrıştırmaması gerektiğini yapısal olarak garanti eder. (Pickle reddi bayt
imzasına bakar, ayrıştırmaya değil.)

### 4.2b Tool Gateway — artık yalnızca tool proxy'si

`search_knowledge_base`, `web_search`, `calculator` gibi 10 tool'u sunar.
Artifact tarafıyla **hiçbir ilgisi kalmadı**: ne MinIO rotası var, ne kimlik
bilgisi, ne kapsam imza sırrı. Kendi egress'i üç onaylı FQDN'le sınırlı.

### 4.3 Nesne deposu — baytlar

MinIO pod'u, 5Gi PVC'ye yazar, S3 protokolüyle servis eder.

**Neden düz bir volume değil:** PVC bir *dizin* verir, MinIO onu bir *servise*
çevirir. Üç sebep: (1) PVC `ReadWriteOnce`, gateway ile paylaşılamaz;
(2) çok yazıcılı erişimi MinIO halleder; (3) asıl sebep **API** — kod S3
konuşur, dolayısıyla OpenShift'te ODF/NooBaa ya da gerçek AWS S3 kod
değişmeden yerine geçer.

Anahtar düzeni:

```
altan/wf_demo_1788436546/extract/6b2803e76918/art_8c3cc05d0d6b.parquet
└─┬─┘ └────────┬───────┘ └──┬──┘ └─────┬────┘ └───────┬──────┘
owner     workflow        node        run          artifact
```

Argo'nun anahtarı `{{workflow.uid}}` ile parametreleme pratiğinin karşılığı.
Kullanıcı/workflow'a çapalı olması, durable deponun uçucu bir anahtara bağlı
kalmamasını sağlar.

### 4.4 Metadata — kayıt defteri

Gateway içinde SQLite, **kalıcı PVC'de** (`/var/lib/ptc/artifacts.db`).

```
artifact_id · name · workflow_id · node_id · run_id
content_hash · content_type · size_bytes · storage_uri
parents[] · owner · created_at · ttl_seconds
```

**Neden MLMD değil:** Red Hat, OpenShift AI 2.23'te Model Registry'den MLMD
sunucusunu kaldırıp kendi şemasına geçti; gerekçe "mimariyi basitleştirmek,
uzun vadeli sürdürülebilirlik". Aynı yönü izliyoruz.

**Neden SQLite:** yine Red Hat'in çizgisi — PostgreSQL üretim için, SQLite
geliştirme/test için. `open_postgres()` yazılı ve bekliyor. SQL taşınabilir
yazıldı (yalnızca TEXT/BIGINT, ISO-8601 zaman, JSON `parents`); sürücüye özgü
tek şey parametre yer tutucusu.

---

## 5. İki değişmez kural

### 5.1 Immutability

Artifact **güncellenmez**. Yeni sürüm = yeni `artifact_id`. `MetadataStore`'da
`update` diye bir metot **yoktur**; `create` var olan bir id'yi ezmek yerine
`ArtifactExists` fırlatır.

Sonuç: Modal'ın Volumes için uyardığı *"last write wins — son yazıcının sahip
olmadığı veri kaybolur"* tuzağı bizde **yapısal olarak oluşmuyor**.

### 5.2 Silme sırası: önce bayt, sonra metadata

| Sıra | En kötü sonuç |
|---|---|
| Metadata → bayt | **Yetim blob**: listelemede görünmez, ücreti işler (MLflow'un bilinen sorunu) |
| **Bayt → metadata** | **Sarkan referans**: tespit edilebilir, onarılabilir |

Görünmez maliyet yerine görünür tutarsızlığı seçiyoruz.

Dedup nedeniyle aynı baytı birden çok kayıt gösterebilir; **son referans
düşene kadar bayt silinmez**, yoksa dedup sessiz veri kaybına dönüşürdü.

---

## 6. Kapsam jetonu — kontrolü dekoratif olmaktan çıkaran şey

Servis her çağrıda "bu artifact çağıranın workflow'una ait mi" diye bakar. Ama
sandbox'ta çalışan kodu **LLM yazmıştır**; kapsamı çağıran bildirseydi
`workflow_id="baskasi"` yazmasını engelleyen hiçbir şey olmazdı.

```
sandbox_runner (laptop)          Tool Gateway (cluster)
   │ Secret'tan anahtarı okur       │ aynı Secret envFrom ile
   │ HMAC-SHA256 ile İMZALAR        │
   │                                │
   └─► PTC_SCOPE_TOKEN ──► sandbox ─┴─► her çağrıda gönderir
                                        gateway İMZAYI doğrular
                                        kapsamı JETONDAN okur
```

Sandbox jetonu okuyabilir (kendi ortam değişkeni) ama **başka bir workflow için
geçerli bir jeton üretemez** — imza anahtarı onda yok. Capability modelinin
kuralı: yetki taklit edilemez bir referansla taşınır; sahibi onu zayıflatabilir,
güçlendiremez.

Jeton yoksa artifact API'si sandbox'a **hiç sunulmaz** — kapsamı doğrulanamayan
bir çalıştırmanın kalıcı depoya yazması, çalıştırmalar arası sınırı kaldırmak
olurdu.

**Sınır açıkça:** jeton, aynı sandbox içindeki koda karşı bir sır değildir.
Koruduğu şey workflow'lar arası sınırdır.

---

## 7. Sandbox'ın gördüğü yüzey

```python
h  = put_artifact(df, name="extract.tickets")     # → "art_8c3cc05d0d6b"
df = get_artifact(name="extract.tickets")         # en yeni sürüm
df = get_artifact(artifact_id="art_8c3cc...")     # belirli sürüm
ls = list_artifacts(node_id="extract")            # keşif
df = cached("tarama", pahali_fn)                  # varsa oku, yoksa üret+sakla
```

Beş fonksiyon. Bucket adı, anahtar düzeni ve S3 kimlik bilgisi **hiç görünmez** —
elde yalnızca opak `artifact_id` vardır.

### 7.1 İkinci tetikleyici: `/output` süpürmesi

Beş fonksiyon LLM'in bu API'yi **bilmesini ve kullanmasını** gerektirir. Model
bunu bilmese ya da unutsa ürettiği dosyalar pod ile birlikte kaybolurdu — bu
riski araştırdık: **hiçbir SOTA kod-çalıştırma sistemi bu riski açık bir tool'a
bağlı bırakmıyor.**

| Sistem | Mekanizma |
|---|---|
| Anthropic | `$OUTPUT_DIR` — komut bitince dizinin **üst düzeyi** yakalanır |
| OpenAI Code Interpreter | `/mnt/data` — aynı konvansiyon |
| OpenHands | Workspace dizini — dosyalar zaten kalıcı, açık "save" tool'u yok |
| **Cloudflare Code Mode** | *(istisna)* dosya sistemi hiç yok, o yüzden açık RPC zorunlu |

Bizim sandbox'ımızın dosya sistemi var, dolayısıyla ilk kampa giriyoruz. `/output`
(emptyDir, 512Mi) — LLM sıradan kod yazar (`df.to_csv("/output/rapor.csv")`),
`_ciktilari_supur()` çalışma bitince (başarı **ya da hata**, fark etmez) dizinin
üst düzeyini artifact'e çevirir:

```
entrypoint.py main()
  try:      exec(code)
  except:   _ciktilari_supur(...)   ← hata olsa bile ÖNCE süpür
            print(error) ; exit
  else:     _ciktilari_supur(...)   ← başarıda da ÖNCE süpür
            print(result)
```

**Süpürme neden `/scratch`'i değil `/output`'u tarıyor:** `/scratch` geçici
alan — her ara dosya, her cache, her yarım çıktı artifact'e dönerdi. Anthropic'in
`$OUTPUT_DIR`'ının da boş ve ayrı bir dizin olması aynı sebepten.

**Güvenlik sınırı süpürmede de aynı:** gateway'in pickle reddi, isim/boyut
doğrulaması buradan geçen dosyalara da uygulanır. Bir dosya reddedilirse
(`artifact_skipped` olayı) yalnızca o dosya atlanır — diğerleri ve asıl sonuç
etkilenmez. Bu olay `ArtifactEvent`'e **çevrilmez**: hiç depolanmamış bir
dosyanın `artifact_id`'si olamaz, model bunu zorunlu tutuyor. Yalnızca canlı
panel için görünürlük.

**İki tetikleyici, tek güvenlik sınırı:**

| | Açık API (`put_artifact`) | Süpürme (`/output`) |
|---|---|---|
| LLM'in bilmesi gerekir mi | Evet | **Hayır** |
| Tip korunumu | Evet (Parquet/Arrow) | Yalnızca uzantıdan tahmin |
| Erken sorgu (`cached`) | Evet | Hayır — ancak çalışma bitince görünür |
| pickle/boyut/isim denetimi | Evet | Evet — aynı servis kapısından geçer |
| İsim | LLM'in verdiği | Dosya adından (Türkçe/boşluk → tire) |

İkisi rakip değil: açık API script-içi erken sorguyu ve tip korunumunu çözüyor,
süpürme LLM'in API'yi hiç kullanmadığı durumda emniyet ağı oluyor.

### 7.2 Okuma tarafı: tembel doldurma

Süpürme yazma tarafını çözüyor. Okuma tarafında simetrik soru şu: LLM
`pd.read_csv("/output/rapor.csv")` yazdığında, o dosya **başka bir pod'da**
üretilmişse ne olacak?

**Denenip terk edilen yol — prefetch (2026-09-03 → 09-04).** Pod doğarken
depodaki her artifact `/output`'a indiriliyordu. Çalıştı, ama iki sorunu vardı
ve ikisi de ölçekle büyüyordu:

- Maliyet *var olan her şeyle* ölçekleniyordu. Workflow'da N artifact varsa,
  hiçbirine dokunmayan bir script bile N indirme yapıyordu.
- `/output` 512Mi. Altı tane 100 MiB'lik artifact biriktiği anda pod, kodu
  çalıştırmadan tahliye edilirdi.

**Yerine geçen — manifest + tembel doldurma.** Pod açılışında yalnızca
**isimler** çekiliyor: tek istek, N'den bağımsız. Baytlar ancak gerçekten
okunduğunda iniyor. `pd.read_csv`/`read_parquet`/`read_json`/`read_excel`
sarmalanıyor, ayrıca sandbox'ın globals'ına tembel bir `open` konuyor.
`builtins` **değiştirilmiyor** — LLM'in doğrudan çağrısı yakalanıyor ama
kütüphanelerin iç dosya işlemleri hiç etkilenmiyor.

Maliyet O(hepsi) yerine **O(kullanılan)**.

**Simetri bilerek bozuk.** Süpürme otomatik kalıyor çünkü maliyeti *o
çalıştırmada üretilenle* ölçekleniyor — doğal olarak küçük. Prefetch ise
sınırsız büyüyordu.

**İndirilen dosya süpürmede geri yüklenmiyor.** Tembel okuma indirdiği her
dosyanın (mtime, boyut) çiftini kaydediyor; süpürme dokunulmamış olanı atlıyor.
Bu kontrol olmadan, sadece okuyan bir çalıştırma bile dosyayı "üretilmiş" sayıp
geri yükler — dedup baytı tekilleştirse de her seferinde yeni bir `artifact_id`
ve sahte bir "produced" olayı doğardı. (Prefetch döneminde gerçekten yaşandı,
2026-09-03; kontrol o yüzden tembel yolda da duruyor.)

Doğrulandı (2026-09-04, canlı cluster): ayrı bir pod'da
`pd.read_csv("/output/satislar.csv")` — kod artifact API'sini hiç bilmeden —
dosyayı getirdi ve olay `consumed` olarak kaydedildi. Hiçbir şeye dokunmayan
bir çalıştırma ise 3.62 sn sürdü, sıfır artifact olayı üretti ve kayıt
defterine sıfır satır ekledi.

---

## 8. İzlenebilirlik

Artifact temasları `tool_call`'dan **ayrı** kaydedilir. İki farklı soru:

- `tool_call` → *hangi tool çağrıldı*
- `artifact` → *hangi VERİ nereden geldi, nereye gitti*

`AccessPath.ARTIFACT_STORE` ayrı bir erişim yoludur; `source_refs` çıktısında
`produced:extract.tickets` / `consumed:extract.tickets` olarak görünür. Böylece
bir cevabı denetleyen kişi, verinin canlı sistemden **taze** mi geldiğini yoksa
saklanmış bir artifact'ten mi okunduğunu ayırt edebilir — kalıcılık eklendiği
anda bu ayrım izlenebilirliğin merkezine oturuyor.

---

## 9. Ölçülmüş sonuçlar

### 9.1 Çalıştırma maliyeti (`scripts/measure_sandbox.py`)

| | Baseline | Faz 0 sonrası |
|---|---|---|
| Toplam | ~7,21 sn | **3,14 sn** |

Kırılım: `pod_running` 1,62 sn (K8s pod kurulumu) + ilk `tool_call` 1,49 sn
(Python/fastmcp açılışı). Yani **düz bir warm pool yalnızca 1,62 sn'yi geri
alır**; kalanı için sürecin de önceden ayakta olması gerekir.

En büyük tek kazanç: entrypoint'in terminal JSON satırı görülünce dönmek,
Kubernetes'in `Job.status`'ü güncellemesini beklememek (~2,7 sn).

### 9.2 Retry maliyeti (`scripts/demo_retry_maliyeti.py`)

Senaryo: 12 tool çağrılık pahalı blok, sonra son satırda `NameError`.

| | 1. deneme | 2. deneme (düzeltme) | Toplam |
|---|---|---|---|
| Kalıcılıksız | 12 | **12** | 24 |
| `cached()` ile | 14 | **1** | 15 |

**Süre kasten raporlanmıyor**: mock tool'lar anında dönüyor, toplam süreyi 4 pod
başlatması belirliyor. Anlamlı sinyal çağrı sayısı; zaman kazancı gerçek kaynak
sistemlerin gecikmesiyle orantılı büyür.

### 9.3 Dayanıklılık

Artifact yazıldı → **gateway pod'u silindi** → yeni pod'dan okundu, veri geldi.

---

## 10. OpenShift'e ne değişiyor

| Konu | kind (şimdi) | OpenShift |
|---|---|---|
| Nesne deposu | MinIO pod + PVC | **ObjectBucketClaim** (ODF/NooBaa) — bucket + bucket'a kilitli hesap + ConfigMap/Secret'ı kendi üretir |
| Bağlantı | Elle yazılmış ConfigMap/Secret | OBC üretir — **aynı alan adları**, kod değişmez |
| Ağ | CiliumNetworkPolicy | OVN `NetworkPolicy` |
| Metadata | SQLite + PVC | PostgreSQL (`open_postgres`) |
| UID | `runAsUser` elle | SCC atar — **manifestten silinmeli** |
| İzolasyon | `runc` | `runtimeClassName: kata` (sandboxed containers) |

Kod tarafında değişen: **hiçbir şey**. Bağlantı bilgileri yalnızca
`BUCKET_NAME` / `BUCKET_HOST` / `BUCKET_PORT` / `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY`'den okunuyor ve bunlar OBC'nin ürettiği adların birebir
aynısı.

---

## 11. Dosya haritası

```
src/grounded_assistant/artifacts/     925 satır
  metadata.py    kayıt defteri (SQLite/Postgres), immutability, TTL
  store.py       S3 erişimi, OBC sözleşmesi
  serialize.py   Parquet/Arrow, pickle savunması
  service.py     beş operasyon + dört kontrol + dedup
  scope.py       HMAC kapsam jetonu

mock_services/tool_gateway/server.py  dört MCP tool'u
sandbox_image/entrypoint.py           beş sandbox fonksiyonu
src/grounded_assistant/ptc/sandbox_runner.py  jeton imzalama, olay akışı
src/grounded_assistant/trace.py       artifact kaydı

k8s/artifact-store/   minio.yaml · obc-shape.local.yaml ·
                      objectbucketclaim.yaml · scope-signing.secret.yaml
k8s/tool-gateway/     deployment.yaml · service.yaml · metadata-pvc.yaml
k8s/sandbox/          job-template.yaml

tests/unit/           36 test
scripts/              measure_sandbox.py · demo_artifact_persistence.py ·
                      demo_retry_maliyeti.py
```

---

## 12. Bilinen sınırlar

| Sınır | Not |
|---|---|
| Gateway tek replika | SQLite ReadWriteOnce PVC'de; Postgres'e geçince kalkar |
| Kata denenmedi | `runtimeClassName` tek alan, ama yerelde nested virt gerekiyor |
| Gerçek OBC denenmedi | Şekli taklit edildi; ekibin cluster'ında doğrulanacak |
| Warm pool yok | Tavanı ölçüldü: 3,14 sn'nin en fazla 1,62'si |
| TTL/GC uygulanmıyor | Şema ve `expired()` hazır, temizleyici görev yazılmadı |
| Versiyon ağacı yok | Immutable id yeterli görüldü |
| Süpürmede Türkçe/özel karakter isim çirkinleşiyor | `_gecerli_artifact_adi` ASCII olmayanı tireye çeviriyor (`Şubat.csv` → `-ubat.csv`); veri kaybı yok, yalnızca ad okunaksız. Bilerek düzeltilmedi — süpürme birincil yol değil, emniyet ağı |
