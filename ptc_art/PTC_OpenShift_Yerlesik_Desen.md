# OpenShift Ne Öneriyor — ve Biz Neyi Değiştirmeliyiz

**Tarih:** 2026-09-04 · **Kısıt:** OpenShift Data Foundation (ODF) **yok**.

Soru şuydu: *"OpenShift'te nasıl yapılıyorsa biz de öyle yapalım, icat
çıkarmaya gerek yok."* Doğru yaklaşım. Red Hat'in birincil dokümanlarına
baktım. Aşağıdakilerin hepsi kaynaklı; doğrulayamadıklarımı ayrıca işaretledim.

---

## 1. En önemli bulgu: OpenShift'in kendisinde nesne depolama YOK

OpenShift Container Platform'un depolama dokümanı baştan sona şunlardan
oluşuyor: ephemeral storage, persistent storage, CSI, dinamik provisioning,
volume genişletme. **S3 diye bir bölüm yok.**

Desteklenen CSI sürücüleri de tamamen blok/dosya: AWS EBS, AWS EFS, Azure Disk,
Azure File, GCP PD, Google Filestore, IBM VPC Block, OpenStack Cinder,
OpenStack Manila, CIFS/SMB, vSphere.

> **Sonuç:** OpenShift'te nesne deposu ya **ODF** ile gelir ya da **harici bir
> üründen**. ODF'yi kapsam dışına aldığımıza göre, S3'ü biz getirmek
> zorundayız.

Bu, MinIO kararımızı "icat" olmaktan çıkarıyor — aşağıdaki 2. bulgu yüzünden.

---

## 2. Red Hat'in kendi AI ürünü S3'ü ZORUNLU tutuyor

OpenShift AI'da bir pipeline server kurmanın ön koşulu:

> *"You have an existing S3-compatible object storage bucket and you have
> configured write access to your S3 bucket on your storage account."*

Alternatif **yok**. Dikkat çekici olan şu: veritabanı için "Default database on
the cluster" diye bir geliştirme seçeneği sunuluyor, ama **depolama için böyle
bir seçenek sunulmuyor**. PVC ile pipeline artifact'i saklama yolu belgelenmiş
değil.

> **Sonuç:** Red Hat'in kendi ürünü bile "S3 bul, getir" diyor. Yani bizim
> MinIO'yu iş yükü olarak koşturmamız, OpenShift'te ODF'siz kalan herkesin
> yapmak zorunda olduğu şey.

---

## 3. Kullanıcı kodu S3'e nasıl erişiyor — mount DEĞİL, SDK

Red Hat'in "Connect your workbench to S3-compatible object storage" dokümanı
net:

> *"To interact with data stored in an S3-compatible object store from a
> workbench, you must create a local client to handle requests to the AWS S3
> service by using an AWS SDK such as Boto3."*

Kimlik bilgileri pod'a **ortam değişkeni** olarak geliyor:

```python
key_id   = os.environ.get('AWS_ACCESS_KEY_ID')
secret   = os.environ.get('AWS_SECRET_ACCESS_KEY')
endpoint = os.environ.get('AWS_S3_ENDPOINT')
```

**FUSE mount ya da CSI sürücüsü önerilmiyor — hiç geçmiyor.**

> **Sonuç:** "S3'ü disk gibi mount et" OpenShift'in önerdiği yol değil.
> OpenShift'in yolu SDK. Bizim yolumuz da SDK (servisin içinde).

**Ama bir farkımız var ve bunu saklamamalıyız:** Red Hat bu kimlik bilgilerini
**kullanıcının kendi kodunu** çalıştıran workbench pod'una koyuyor. Bizim
sandbox'ımızda **LLM'in yazdığı kod** çalışıyor. Anahtarı oraya koymak, tam da
kaçındığımız şey. Biz aynı deseni bir katman geriye alıyoruz: SDK evet, ama
sandbox'ta değil, servis pod'unda.

---

## 4. Pipeline artifact'leri nasıl taşınıyor — launcher deseni

OpenShift AI'ın pipeline motoru KFP v2. Orada iş şöyle bölünüyor:

- Nesne deposu ayarları **namespace başına** bir ConfigMap'te:
  > *"To configure the object store utilized by the KFP Launcher, you will need
  > to edit the `kfp-launcher` Kubernetes ConfigMap."*
  > *"this configmap needs to be deployed in the same namespace where the
  > Pipelines will be created."*
- Kimlik bilgileri Secret'lardan geliyor; AWS'de IRSA ile ServiceAccount'a da
  bağlanabiliyor.
- **Launcher** süreci adımın çıktısını depoya yüklüyor, girdisini indiriyor.
  Kullanıcının component container'ı S3 ile doğrudan konuşmuyor.

> **Sonuç:** Bizim `entrypoint.py`'deki süpürme + tembel doldurma, KFP'nin
> launcher'ıyla **aynı şekil**. Bu deseni biz icat etmedik; OpenShift'in kendi
> pipeline motoru da böyle çalışıyor.

**Farkımız — ve bu bizim lehimize:** KFP'nin kapsamı **namespace** düzeyinde.
Aynı namespace'teki her pipeline aynı bucket kimlik bilgisini kullanıyor.
Bizim kapsam jetonumuz **çalıştırma başına**. Güvenilmeyen kod çalıştırdığımız
için bu daha dar granülarite gerekli.

---

## 5. OpenShift Pipelines (Tekton) farklı bir yol izliyor

Tekton'da adımlar arası veri paylaşımı **workspace** ile, ve workspace bir
PVC'ye bağlanıyor (`volumeClaimTemplate` ile otomatik provisioning). Nesne
deposu değil, paylaşılan disk.

> **Neden bizim için uygun değil:** paylaşılan PVC'ye birden çok pod'un aynı
> anda yazması **ReadWriteMany** gerektiriyor. RWX ise yalnızca dosya-tabanlı
> StorageClass'larda var (NFS, Azure File, EFS, Filestore, Manila, CIFS) —
> blok tabanlı olanlarda (EBS, Azure Disk, GCP PD, Cinder, vSphere) yok.
> Kümede hangi StorageClass'ların olduğu **ekibe sorulacak** (§8).

---

## 6. Güvenilmeyen kod için Red Hat ne diyor — bizi doğruluyor

Red Hat'in 2026 tarihli ajan rehberi (Red Hat Developer) doğrudan bizim
durumumuza bakıyor:

**İzolasyon — katmanlı:**
> *"Running OpenShell inside an OpenShift sandboxed containers VM gives you
> both layers simultaneously."*

Yani AI-üretimi kod için **Kata** (OpenShift Sandboxed Containers) öneriliyor.

**Egress — varsayılan reddet:**
> *"The default posture is deny-all. In practice, you write a policy that
> allowlists exactly the endpoints your agent needs."*

Bu bizim `sandbox-egress` politikamızın birebir tarifi.

**Kimlik bilgisi — sandbox'ın DIŞINDA:** arama sonucunda çıkan özet, kimlik
bilgilerinin sandbox içinde saklanmadığını, ağ sınırında enjekte edildiğini
söylüyor. *(Not: bu cümleyi makalenin kendisinden birebir doğrulayamadım —
makaleyi çektiğimde dosya-sistemi izolasyonu ve egress kısımları geldi,
kimlik-enjeksiyonu kısmı gelmedi. Ekibe sunmadan önce teyit edilmeli.)*

Ayrıca Red Hat'in bu çalışmaları **henüz ürün değil**: *"early validations, not
shipping product features yet."*

---

## 7. Bulguların bizim koda karşılığı

### Değiştirdiğim şey (bugün yapıldı)

`BucketConfig.from_env()` yalnızca **OBC** sözleşmesini okuyordu
(`BUCKET_NAME`/`BUCKET_HOST`/`BUCKET_PORT`). ODF olmayınca bu değişkenleri
üretecek kimse yok. Artık **iki sözleşmeyi de** okuyor:

| Sözleşme | Değişkenler | Ne zaman |
|---|---|---|
| **A — ObjectBucketClaim** | `BUCKET_NAME`, `BUCKET_HOST`, `BUCKET_PORT`, `BUCKET_REGION` | ODF varsa |
| **B — OpenShift AI connection** | `AWS_S3_ENDPOINT`, `AWS_S3_BUCKET`, `AWS_DEFAULT_REGION` | **ODF yoksa — bizim durumumuz** |

İkisinde de `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`. İkisi birden varsa
OBC kazanıyor. `AWS_S3_ENDPOINT` hem `https://host:port` hem çıplak `host`
biçiminde kabul ediliyor, TLS şemadan (yoksa porttan) çıkarılıyor.

Böylece ekip kümede bir OpenShift AI connection Secret'ı yarattığında **kod
değişmeden** çalışacak.

### Artifact depolama mentalitesi — üç hizalama (2026-09-04)

KFP/OpenShift AI'ın artifact modeline bakınca üç gerçek fark çıktı. Üçü de
kapatıldı.

**(a) `pipeline_root` — "bucket'ın neresine" KOD değil YAPILANDIRMA.**
KFP bunu üç düzeyde ayarlatıyor: dağıtım varsayılanı (ConfigMap), pipeline
başına, çalıştırma başına. Bizde prefix sabit kodluydu. Artık
`PTC_ARTIFACT_ROOT` ortam değişkeni (dağıtım varsayılanı) ve `X-Artifact-Root`
başlığı (çağrı başına) var. Verilmezse eski düzen aynen korunuyor.

**(b) Tipli artifact.** KFP'de her artifact bir MLMD şema başlığı taşıyor.
Aynı sözlüğü birebir aldık — isim çevirisi gerekmesin diye:

    system.Artifact · system.Dataset · system.Model
    system.Metrics  · system.HTML    · system.Markdown

Tip **otomatik çıkarılıyor**: DataFrame/Parquet/CSV → `Dataset`, sayısal sözlük
→ `Metrics`, `.html` → `HTML`. Açıkça verilirse o kazanıyor. Bilinmeyen tip
reddedilmiyor, taban tipe düşürülüyor — tip bir *etiket*, güvenlik kontrolü
değil; yanlış etiket yüzünden çıktı kaybetmek orantısız olurdu.

**İnce nokta — çıkarım nerede yapılıyor:** "sayısal sözlük = metrik" bilgisi
ancak NESNEYE bakarak anlaşılır; serileştirildikten sonra o da sadece
`application/json`. Bu yüzden çıkarımın değerli hâli **sandbox tarafında**,
nesne hâlâ elde iken çalışıyor. Servis yalnızca baytı gördüğü için
content_type'a düşüyor. Ortak kural `serialize.py`'de — o dosya sandbox imajına
aynen kopyalandığı için iki taraf ayrışamıyor.

**(c) `.metadata` torbası.** KFP'de her artifact serbest anahtar-değer
taşıyabiliyor. Kayıt defterinde `user_metadata` sütunu var; `X-Artifact-Metadata`
başlığıyla doldurulur. (2026-09-06'da LLM'e sunulan `put_artifact` kalktığı için
sandbox bunu artık kendisi geçiremiyor — süpürme yolunda metadata boş kalıyor.
Açık olarak §11.10'a eklenmeli.)
Bozuk JSON sessizce yok sayılıyor (aynı gerekçe: etiket, kontrol değil).

**Doğrulandı** (canlı, 2026-09-04): tek bir çalıştırmada üç artifact —
`system.Dataset` (otomatik), `system.Metrics` (otomatik, sayısal sözlükten),
`system.Model` + `{"algoritma": "linear", "r2": 0.91}` (açık). Şema göçü 66
mevcut kayıtta çalıştı, hiçbiri kaybolmadı, eskiler hâlâ okunuyor.

### Zaten uyumlu olanlar

| Konu | OpenShift deseni | Bizde |
|---|---|---|
| S3'e erişim | SDK (boto3), mount değil | SDK (minio istemcisi) ✓ |
| Artifact taşıma | Launcher yüklüyor/indiriyor | Süpürme + tembel doldurma ✓ |
| Kimlik bilgisi yeri | Secret → pod ortamı | Secret → **servis** pod'u ✓ (daha dar) |
| Egress | Varsayılan reddet + allowlist | Aynı ✓ |
| Artifact tipi | MLMD şema başlıkları | Aynı sözlük ✓ |
| Artifact `.metadata` | Serbest anahtar-değer | Aynı ✓ |
| Depo kökü | `pipeline_root`, yapılandırılabilir | `PTC_ARTIFACT_ROOT` ✓ |
| `.uri` ↔ `.path` kopyalama | Launcher yapıyor | Süpürme + tembel doldurma ✓ |
| Metadata DB | PostgreSQL (MLMD) | SQLite — **açık** |

### Hâlâ açık olanlar

| Konu | Durum |
|---|---|
| **Kata** | Red Hat AI-üretimi kod için açıkça öneriyor; bizde yok |
| **PostgreSQL** | Hem metadata hem workflow state için; bizde SQLite |
| **S3'ü kim sağlayacak** | ODF yok → MinIO mu, harici kurumsal S3 mü (§8) |

---

## 8. Ekibe sorulacaklar (ODF olmadığı için değişti)

1. **Kurumda S3-uyumlu bir depo var mı?** (NetApp StorageGRID, Dell ECS, Ceph
   RGW, harici MinIO, AWS S3…) Varsa endpoint + bucket + anahtar yeter, kod
   hazır. Yoksa MinIO'yu iş yükü olarak koşturacağız — Red Hat'in kendi ürünü
   de S3'ü zorunlu tuttuğu için bu bir istisna değil, zorunluluk.
2. **Hangi StorageClass'lar var, RWX destekleyen var mı?** Bu, PVC tabanlı bir
   alternatifin (Tekton deseni) mümkün olup olmadığını belirler.
3. **OpenShift Sandboxed Containers (Kata) kurulu mu / kurulabilir mi?**
4. **PostgreSQL sağlanabilir mi?** İkisi için: artifact metadata + workflow state.

İlk soru artık en kritik olanı — ODF çıkınca "S3 nereden gelecek" cevapsız kaldı.

---

## 9. Doğrulanamayanlar

- Red Hat ajan makalesindeki "kimlik bilgileri sandbox'ta saklanmaz, ağ
  sınırında enjekte edilir" cümlesi: arama özetinde vardı, makalenin kendisinden
  birebir teyit edemedim.
- OCP erişim modları tablosu (hangi sürücü RWX destekliyor): docs.redhat.com
  sayfaları içeriği yerine gezinme menüsü döndürdü. §5'teki RWX listesi genel
  Kubernetes bilgisine dayanıyor, Red Hat tablosundan alıntı **değil** —
  ekibe sunmadan önce kümede `kubectl get storageclass` ile bakılmalı.
- `AWS_S3_BUCKET` değişken adı: OpenShift AI connection'ının bucket alanını bu
  adla verdiğini ikincil kaynaklardan aldım; `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_S3_ENDPOINT`, `AWS_DEFAULT_REGION` birincil
  dokümanda birebir geçiyor. Kod her iki durumda da açık hata mesajı veriyor.

---

## Kaynaklar

**OpenShift depolama**
- [OCP 4.17 — Storage](https://docs.redhat.com/en/documentation/openshift_container_platform/4.17/html-single/storage/index)
- [OCP 4.8 — Storage](https://docs.redhat.com/en/documentation/openshift_container_platform/4.8/html/storage/)

**OpenShift AI**
- [Working with data science pipelines — Managing pipelines](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.25/html/working_with_data_science_pipelines/managing-data-science-pipelines_ds-pipelines)
- [Connect your workbench to S3-compatible object storage](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.25/html-single/working_with_data_in_an_s3-compatible_object_store/index)
- [Creating an S3 client](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.3/html/working_with_data_in_an_s3-compatible_object_store/creating-an-s3-client_s3)

**Kubeflow Pipelines (OpenShift AI'ın motoru)**
- [Object Store Configuration](https://www.kubeflow.org/docs/components/pipelines/operator-guides/configure-object-store/)
- [Create, use, pass, and track ML artifacts](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/)

**OpenShift Pipelines (Tekton)**
- [Understanding OpenShift Pipelines 1.16](https://docs.redhat.com/en/documentation/red_hat_openshift_pipelines/1.16/html/about_openshift_pipelines/understanding-openshift-pipelines)

**Red Hat — AI ajanları ve izolasyon**
- [Layered sandboxing for AI agents: OpenShift and OpenShell](https://developers.redhat.com/articles/2026/07/16/layered-sandboxing-ai-agents-openshift-and-openshell)
- [Red Hat build of Agent Sandbox](https://developers.redhat.com/articles/2026/07/15/red-hat-build-agent-sandbox-isolated-workload-management-kubernetes)
