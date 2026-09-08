# PTC Sunum Rehberi — baştan sona

**Tarih:** 2026-09-08 · **Slayt:** 11 sayfa · **Süre:** 30 dk anlatım + 10 dk soru

Bu dosya sunumu yönetmek için. Slayt sırasına göre: her sayfada **ne diyeceğin**,
**hangi sayıyı söyleyeceğin**, **neyi söylemeyeceğin**.

- **Slayt (canlı):** https://claude.ai/code/artifact/ba545b49-2790-4584-b2e0-aae256a1e12a
- **Slayt (yerel, internetsiz de açılır):** `PTC_Hizli_Tur.html`
- **Ölçüm raporu:** `PTC_Hata_Bicimleri_Kiyas.md`

---

# 0 · Sunumdan 15 dakika önce

## Aç

```bash
cd ~/Desktop/Data-Science-Reports/ptc_art
S=/tmp/ptc-log; mkdir -p $S

nohup kubectl port-forward svc/artifact-service 8080:8080 > $S/pf.log 2>&1 & disown
PYTHONPATH=src nohup ../ptc_sec/.venv/bin/uvicorn \
  grounded_assistant.web.app:app --port 8123 --log-level info > $S/web.log 2>&1 & disown
nohup cilium hubble port-forward > $S/hubble.log 2>&1 & disown
nohup kubectl port-forward -n kube-system svc/hubble-ui 12000:80 > $S/hubble-ui.log 2>&1 & disown
```

## Doğrula

```bash
kubectl get pods                                    # artifact-service · minio · tool-gateway Running
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8123/konsol   # 200
hubble status | grep Healthcheck                    # Ok
```

## Ekranda hazır dursun

| Sekme | Ne |
|---|---|
| 1 | `PTC_Hizli_Tur.html` — slayt |
| 2 | `http://localhost:8123/konsol` — PoC |
| 3 | `http://localhost:12000` — Hubble UI |
| 4 | Terminal — `hubble observe --label app=ptc-sandbox -f` hazır, ENTER'a basılmamış |

## Slayt kontrolleri

```
→ ←   sayfa      o   genel bakış (11 sayfa ızgara)      Esc   kapat
```

> Depoyu **önceden doldur**: Hatlar sekmesinden PL-A'yı bir kez çalıştır.
> Demoda boş depo göstermek istemezsin.

---

# 1 · Anlatım planı — 11 sayfa

Toplam 30 dakika. Yanındaki süre **hedef**, aşarsan 6 ve 7'den kıs.

| # | Sayfa | dk | Tek cümlelik amaç |
|---|---|---|---|
| 1 | Sandbox ölür, ürettiği kalır | 2 | Problemi kur |
| 2 | Beş soru | 3 | Haritayı göster |
| 3 | Dört aile | 4 | **Tez burada** |
| 4 | Dört ürün, dört akış | 3 | Tezi örnekle |
| 5 | Red Hat / OpenShift AI | 4 | En yakın komşu, ve neden yetmiyor |
| 6 | Tekton · Argo · Databricks · Devin | 2 | Kalanları kapat |
| 7 | Keşif ve kayıt defteri | 3 | İkinci sorun: bulmak |
| 8 | Çağrı değil, beyan | 3 | Bizim cevabımız |
| 9 | İki container + OpenShift | 4 | Ölçüm ve hedef ortam |
| 10 | Kod patlıyor, ne dönüyor | 2 | İkinci eksen |
| 11 | Ölçülmüş karşılaştırma | 3 | Kanıt |

---

## Sayfa 1 — Sandbox ölür. Ürettiği kalır.

**Aç:**
> "Kod çalıştıran her sistemin aynı çelişkisi var. Sandbox yaşarsa bir çalıştırma
> diğerine bulaşır — izolasyon yok. Sandbox ölürse 40 saniyelik iş her turda
> tekrarlanır. Bizimki 4 saniye yaşıyor ve içindeki kodu **LLM yazıyor**."

**Vurgu:** Kod güvenilmez. Bu, sunumun geri kalanının tek dayanağı.

**Söyleme:** Henüz çözümden bahsetme.

---

## Sayfa 2 — Herkes aynı beş soruyu cevaplamak zorunda

Tabloyu **okuma**, iki sütunu göster:

> "Sağdaki iki sütuna bakın. **Anahtar sandbox'ta mı** ve **kayıt defteri var mı**.
> Bir örüntü var: kayıt defteri olan ürünlerde yazma yolu bir bileşenden geçiyor.
> Mount edenlerin hiçbirinde registry yok."

**Cümle:**
> **"Kayıt defteri yalnızca yazma yolu bir bileşenden geçtiğinde ayakta kalıyor."**

Bu, araştırmanın en keskin bulgusu. Sonraki sayfa bunu açıklıyor.

---

## Sayfa 3 — Baytı kim taşıyor: dört aile ★

**Sunumun tezi burada. Acele etme.**

```
A · MOUNT        kod → FUSE → bucket          araya girecek yer YOK
B · SARMALAYICI  launcher + kod AYNI container  anahtar kodun yanında
C · SINIR        kod → dizin → AYRI container   kod anahtarı göremez
D · DENETİMLİ    mount var, sürücü UC'ye soruyor
```

> "Dört yerleşim var. A'da araya girecek yer yok — o yüzden künye de yok.
> B'de var ama **atlanabilir**. C'de taşıyıcı ayrı bir güven alanında.
> D tek istisna: mount var ama sürücü her erişimde izin soruyor."

**Kapanış — Argo'nun kanıtı:**
> "Argo dört farklı yerleşim denemiş: `docker`, `kubelet`, `k8sapi`, `pns`.
> v3.4'te **hepsini kaldırmışlar**. `docker` için gerekçeleri:
> *breaks security completely*."

Bu alıntı sunumun en güçlü kartı — bir tasarım tercihini başkasının acı deneyimi
doğruluyor.

---

## Sayfa 4 — Dört ürün, dört akış

Dört çizimi tek tek gösterme; **ikiye böl**:

> "Üstteki ikisinde platform araya giriyor. Anthropic kod bitince dizine bakıyor,
> dosyayı Files API'ye koyuyor, konuşmaya bir fiş dönüyor. Alttaki ikisinde
> giren yok — Cloudflare mount ediyor, ADK ise baytı hiç taşımıyor,
> sadece isimleri her turda talimatlara yazıyor."

**Sayı:** Anthropic 30 gün · OpenAI 20 dakika. İkisi de keşif sorununu
**sandbox'ı yaşatarak** çözüyor.

---

## Sayfa 5 — Red Hat / OpenShift AI ★

En yakın komşumuz. İki bölümde anlat.

### Önce anatomi

```
init container   kfp-driver     .uri'leri MLMD'den ÇÖZER
main container   kfp-launcher   PID 1
   └─ KULLANICI KODU            launcher'ın ALT SÜRECİ
```

> "Kod S3'ü hiç görmüyor. `cikti.path`'e yazıyor, launcher `.uri`'ye taşıyor.
> İndirme kod **başlamadan** bitiyor, yükleme kod **bittikten sonra** başlıyor."

### Sonra kritik fark

> "Ama launcher kullanıcı koduyla **aynı container'da**. Alt süreç ebeveyninin
> ortamını devralır — aynı env, aynı dosya sistemi, aynı ağ. Yani kod launcher'ı
> atlayıp doğrudan S3'e konuşabilir."

**Hemen ardından — bu cümle olmazsa haksızlık edersin:**
> "**KFP için bu doğru karar.** Oradaki kodu bir insan yazdı, gözden geçirdi,
> git'te duruyor. Sarmalayıcının işi güvenlik değil kolaylık.
> Bizde kodu LLM yazıyor — aynı karar savunulamaz."

---

## Sayfa 6 — Tekton · Argo · Databricks · Devin

Hızlı geç. Vurgulanacak iki şey:

> "**Argo bizim yerleşimimizin kaynağı**: `argoexec` kullanıcının imajından
> ayrı bir imaj, anahtar orada. **Devin** ise soruyu ortadan kaldırıyor —
> sandbox'ı hiç öldürmüyor, microVM snapshot'la RAM'i bile saklıyor.
> Bedeli süresiz yaşayan bir makine. Bizim sandbox 4 saniye yaşıyor,
> ikisi de bize kapalı."

---

## Sayfa 7 — Keşif ve kayıt defteri

İkinci sorun: **taşımak değil, bulmak**.

> "Bir belirleyici var: sandbox yaşıyor mu. Yaşıyorsa `ls` yeter, keşif bedava.
> Pipeline dünyasında ise soru hiç doğmuyor — DAG'ı insan yazıyor, girdi
> bağlanmış geliyor. **Bizde iki şart birden var:** ajan karar veriyor
> ve sandbox ölüyor."

Sağdaki tablo için:
> "Aynı addan 12 sürüm birikince kim ne yapıyor? MLflow alias koyuyor,
> KFP'de çakışma imkânsız çünkü yol run-id içeriyor, DVC içerik hash'i
> kullanıyor. Biz alias'ı MLflow'dan aldık."

---

## Sayfa 8 — Çağrı değil. Beyan.

> "Eskiden kod çalışırken `load_artifact()` diye bir ağ çağrısı yapıyordu.
> Şimdi sadece **beyan** ediyor: `inputs=["ozet.json"]`. Dosya kod başlamadan
> yerine konuyor, kod düz `open()` yazıyor."

**Üç biçim:**
```
ozet.json               → /output/ozet.json            kendi
wf-abc123/ozet.json     → /artifacts/wf-abc123/…       ÇAPRAZ
rapor.pdf@onaylanmis    → /artifacts/_alias/rapor.pdf  SABİT
```

> "KFP ve Argo tam böyle yapıyor. Tek fark: onlarda beyanı **insan** yazıyor,
> YAML'da. Bizde **model**."

**Ölçüm — bu sayıyı mutlaka söyle:**
```
100 dosya ürettik, kod bir tanesini okudu

beyansız  →  100 dosya iner ·  soy 100 EBEVEYN
beyanlı   →    2 dosya iner ·  soy   2 ebeveyn
```

> "Asıl bedel indirme değil, **soy şişmesi**. Beyan tam bunun için var.
> Soyu da beyandan çıkarıyoruz — MLMD'nin `DECLARED_INPUT` olayı gibi."

---

## Sayfa 9 — İki container, bir /output ★

> "Pod'da iki container var. Sidecar'da S3 anahtarı ve kapsam jetonu.
> Sandbox'ta LLM'in kodu ve hiçbir sır. Aralarında sadece `/output` paylaşılıyor.
> Container'lar ortam paylaşmıyor."

**Ölçüm:**
```
S3 kimlik bilgisi   []
S3 SDK kurulu       hayır
MinIO'ya IP ile     TimeoutError
internet            ConnectionError

sandbox'ın ağ çağrısı: 0
```

**OpenShift kısmı — sorulmadan söyle:**
> "`restricted-v2` SCC ile tam akışı koşturduk: rastgele UID, `runAsNonRoot`,
> tüm capability'ler düşürülmüş. **SCC'nin dayattığı hiçbir kısıt bizi kırmıyor.**
> Kalıcılık için gereken tek şey bir S3-uyumlu depo: endpoint, bucket, anahtar.
> Kod hem OBC hem OpenShift AI connection sözleşmesini okuyor."

---

## Sayfa 10 — Kod patlıyor. Modele ne dönüyor?

İkinci eksen. Altı kutuyu tek tek okuma, örüntüyü söyle:

> "Üçünde aynı üçlü var: **çıkış kodu, tam traceback, hata anında korunan
> stdout**. Bizim eski hâlimiz üçünü de kaybediyordu — üstelik modele
> *dur* diyordu."

---

## Sayfa 11 — Aynı hata, on biçim: ölçülmüş

> "Bunu iddia olarak bırakmadık. On biçimlendiriciyi kurup aynı beş arızaya
> soktuk, yedi ölçütte puanladık. Kendi biçimimiz gerçek kod — `entrypoint.py`
> içe aktarılıp çağrılıyor, kopya değil."

**Üç bulgu:**

1. **Kırpma yönü hatayı yiyor.** SWE-agent'ın varsayılan şablonu baştan kesiyor;
   hata en sonda olduğu için 160 KB'lık çıktıda `ValueError` modele hiç ulaşmıyor.
   `bash_only` yapılandırması baş-yarı + son-yarı alıyor, kaybetmiyor. **23 → 26.**
2. **Kırpmamanın bedeli.** Tek bir hatada AutoGen 211 360, OpenHands 211 328,
   Anthropic 215 415 bayt. Bizim 20 615, Codex 10 113.
3. **Talimat sütunu neredeyse boş.** On biçimden dokuzu ne olduğunu anlatıyor,
   **ne yapılacağını** söylemiyor.

**Ve dürüstlük cümlesi — atlarsan sorarlar:**
> "Ölçüt listesi bizim araştırmamızdan çıktı ve bizim biçimimizi de o araştırma
> şekillendirdi. Sıralamada **dairesellik var**. Kıyaslanabilir olan sinyal değil,
> **sinyal/bayt**: Codex aynı 30/35'i 2 356 baytla taşıyor, biz 4 538'le.
> Ve bizde eksik olan bir şey var: **tekrar tespiti**, o sadece OpenHands'te."

---

# 2 · Canlı demo (5 dk, isteğe bağlı)

Slayt 9'dan sonra yap. Üç adım, hepsi bugün doğrulandı.

## ① Terminalde Hubble'ı başlat

```bash
hubble observe --label app=ptc-sandbox -f
```

## ② Konsolda PL-A'yı çalıştır

Hatlar → **Ticket İşleme Hattı** → *Hattı çalıştır*.
Dört adım, dördü de ayrı pod. Panelde adımlar sırayla yeşile döner.

> "Her adım ayrı bir Kubernetes Job'u. `query` ve `alias` adımları **pod açmaz** —
> keşif sandbox'ta değil, host tarafında."

## ③ Ürettiğinin içini göster

```bash
../ptc_sec/.venv/bin/python scripts/artifact_bak.py --liste
../ptc_sec/.venv/bin/python scripts/artifact_bak.py departman_ozet.parquet
```

Künye + soy + gerçek tablo çıkar.

## Demo çökerse

Panik yok, cümle hazır:

> "Ortam sorunu — ölçümler zaten slaytta, hepsi bu cluster'dan alındı."

Sonra sayfa 9'daki ölçüm bloğunu göster ve devam et.

---

# 3 · Soru–cevap hazırlığı

## "Files API tam olarak ne?"

> "Anthropic'in dosya dolabı. Kod `$OUTPUT_DIR`'a yazıyor, **platform o dizine
> bakıyor**, dosyayı kaydediyor ve konuşmaya bir `file_id` dönüyor. Container
> ölse bile bayt orada kalıyor."

Takip gelirse: **fiş var, soy yok.** `file_abc123` neyden türediğini söylemiyor.
Sürüm, alias, arama da yok.

İki ömrü karıştırma: **container 30 gün**, **Files API dosyası silinene kadar**.

## "Bucket, S3, MinIO — farkları ne?"

```
tablo      ≈  bucket        KAVRAM
SQL        ≈  S3 API        PROTOKOL
PostgreSQL ≈  MinIO         ÜRÜN
```

> "Bucket bir ürün değil, kavram. S3 bir ürün değil, protokol. MinIO ürün."

## "Neden MinIO?"

> "ODF yoktu. Ama bu bir icat değil: **OpenShift'in kendisinde nesne deposu yok** —
> depolama dokümanı baştan sona blok ve dosya. Red Hat'in kendi AI ürünü bile
> pipeline server için *'S3-compatible object storage bucket'* şart koşuyor,
> veritabanı için varsayılan sunarken depolama için sunmuyor.
> Seçim bizi bağlamıyor: S3 API ortak olduğu için ODF'ye ya da kurumsal S3'e
> geçiş bir endpoint değişikliği."

## "Anahtar OpenShift'te koda nasıl görünüyor?"

> "Anahtar koda verilmiyor, **pod'a** veriliyor — Secret'tan ortam değişkenine
> ya da IRSA ile. Kod launcher'ın alt süreci olduğu için ortamı devralıyor.
> Ortam temizlense bile `/proc/1/environ` aynı UID'den okunabiliyor.
> **Aynı container'da sır saklanamaz** — mesele dikkatsizlik değil, süreç modeli."

## "artifact_ara neden gerekli?"

> "Model beyanı yazarken sandbox henüz yok — keşif host tarafında olmak zorunda.
> Manifest de kesik: depoda 102 satır varken 41'i görünüyor, 176 farklı addan
> 33'ü hiç ulaşamıyor. Canlı örnek: model listede 4 eşleşme gördü, arama
> **24** buldu."

## "Hubble sandbox'ın erişemediğini kanıtlıyor mu?"

**Dikkat — hayır.**
> "Cilium politikası **pod** düzeyinde, container düzeyinde değil. Hubble'da
> `ptc-run-xxx → artifact-service` akışını gördüğünüzde bu **sidecar'ın**
> trafiği. Sandbox container'ının erişememesinin sebebi ağ politikası değil:
> anahtar yok, SDK kurulu değil, servis adresi ortamında tanımlı değil."

## "Bu tabloda kendinizi birinci yapmışsınız"

> "Haklısınız, dairesellik var — ölçütleri de biz seçtik. O yüzden sinyal
> sıralamasına değil **sinyal/bayt**'a bakın: orada Codex önde. Ve tabloda
> bizim eksiğimiz açıkça duruyor: tekrar tespiti sütunu."

## "Kaç test var?"

> "242 test geçiyor. Kabul testi bu cluster'da 52/52. Hata biçimi
> karşılaştırmasının 17 testi de bulguların kendisini sabitliyor —
> biçimlendiricilerden biri değişirse tablo sessizce yanlış olmasın diye."

---

# 4 · Söyleme listesi

```
✗ "bizimki Files API'nin aynısı"        → soy, alias, arama yok
✗ "Hubble sandbox'ı hapsediyor"          → pod düzeyi, container değil
✗ "KFP güvensiz"                         → kendi bağlamında doğru karar
✗ "ODF'de de test ettik"                 → ODF kapsam dışıydı, TEST EDİLMEDİ
✗ "MinIO OpenShift'in çözümü"            → PoC'nin vekili; üretimde ODF/harici S3
✗ "en iyi biçim bizimki"                 → en iyi = bu ölçütlerde; düzeltme
                                            başarısını ÖLÇMEDİK
```

**MinIO lisansı** (AGPLv3) sorulursa: "kurumsal kullanımda hukuk tarafına
sorulması gereken bir konu, bu projede doğrulamadım."

---

# 5 · Zaman kısalırsa

**15 dakikaya sıkışırsan** şu 5 sayfa yeter:

```
1   problem
3   dört aile          ← tez
5   Red Hat            ← en yakın komşu
9   iki container      ← bizim cevabımız + OpenShift
11  ölçüm              ← kanıt
```

2, 4, 6, 7, 10 destekleyici. 8'i atlarsan beyan mekanizması havada kalır —
9'da bir cümleyle kapat: *"model ne okuyacağını beyan ediyor, sidecar kod
başlamadan yerine koyuyor."*

---

# 6 · Kapanış cümlesi

> "Piyasada dört yerleşim var ve hepsi kodun güvenilir olduğunu varsayıyor.
> Bizim kodumuzu LLM yazıyor. O yüzden taşıyıcıyı kodun yanından çıkarıp
> ayrı bir container'a koyduk — Argo'nun yaptığı gibi. Üstüne pipeline
> dünyasından üç şey aldık: **beyan**, **soy**, **alias**.
> Hepsi OpenShift'te varsayılanlarla çalışıyor."

---

# 7 · Sunumdan sonra — kapatma

```bash
ss -ltnp | grep -E ':(8123|8080|4245|12000)'   # pid'leri gör
kill <pid>
```

> `pkill -f` kullanma — kendi komut satırıyla eşleşip kabuğu öldürüyor.
