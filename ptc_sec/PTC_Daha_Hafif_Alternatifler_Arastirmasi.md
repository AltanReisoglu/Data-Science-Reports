# "Bu Kadar Bürokrasi Olmadan" — Daha Hafif Alternatifler Araştırması

Bu doküman, bir soruya cevap arıyor: **her PTC çalıştırması için sıfırdan bir Kubernetes
Job (+ CNI/Cilium setup'ı) ayağa kaldırmanın ~7 saniyelik maliyetine** gerçek,
Kubernetes ekosisteminde kabul görmüş alternatifler var mı? Kullanıcının kendi önerisi
(tek pod'da birden fazla sandbox container) da somut olarak değerlendiriliyor.

**Not — bu bir karar dokümanı DEĞİL, bir araştırma dokümanı.** Seçenekleri ve
trade-off'larını sunuyor; hangisinin uygulanacağı ayrı bir karar.

## 1. Şu anki maliyet — kısa hatırlatma

Canlı ölçülmüş (bkz. bu oturumdaki `count_open_tickets()` testi):
```
0.03s   Kubernetes API yazmaları (ConfigMap + Job)
3.08s   Job oluşturulduktan sonra İLK tool_call'a kadar geçen süre
        (scheduler + kubelet + Cilium CNI: IPAM + identity + eBPF attach + Python başlangıcı)
7.21s   Toplam (poll aralığı + cleanup dahil)
```
Yani ~3 saniyelik kısım, SADECE "yeni bir pod'un ağını hazırlama" maliyeti.

## 2. Gerçek bir emsal buldum — Kubernetes SIG-Apps'in "Agent Sandbox" projesi

Araştırırken, TAM OLARAK bu sorunu (AI agent'lar için kod çalıştırma sandbox'larının
soğuk-başlangıç maliyeti) hedefleyen, Kubernetes'in kendi SIG-Apps grubunun ürettiği
resmi bir proje bulundu: **`kubernetes-sigs/agent-sandbox`** — Google Cloud (GKE), Red
Hat ve kubernetes.io'nun kendi blogunda duyurulmuş.

**Tespit ettikleri sorun, bizimkiyle birebir örtüşüyor:**
> *"Kubernetes sandbox cold starts take ~4s+ even with the image preloaded, and for
> interactive agent workloads... that latency dominates time-to-first-action."*
(Bizim ölçtüğümüz ~3sn'lik CNI/scheduling gecikmesiyle aynı sınıfta.)

**Çözümleri — `SandboxWarmPool`:**
- Önceden ısıtılmış (ağı ZATEN kurulu, identity ZATEN atanmış) bir pod HAVUZU tutuluyor
- Yeni bir çalıştırma istendiğinde, sıfırdan pod OLUŞTURULMUYOR — havuzdan bir tane
  "claim" ediliyor (talep ediliyor)
- Havuz tükenirse ancak o zaman normal (yavaş) yol kullanılıyor
- **Sonuç (kubernetes.io blog'undan doğrudan alıntı):** *"sub-second startup latency
  for fully isolated workloads, an improvement of up to ninety percent over
  conventional cold starts."*

**CRD yapısı** (bizim mimariye çevirirsek):
| Onların CRD'si | Bizim karşılığımız |
|---|---|
| `SandboxTemplate` | `k8s/sandbox/job-template.yaml` |
| `SandboxWarmPool` | henüz yok — eklenmesi gereken yeni parça |
| `SandboxClaim` | `run_sandbox()`'un "yeni Job oluştur" yerine "havuzdan al" demesi |

**İzolasyon nasıl korunuyor (havuzdan alınan pod'un "temiz" olduğu garantisi):**
Proje, `gVisor`/`Kata Containers` gibi ÇALIŞMA-ZAMANI izolasyonu kullanıyor (bizim şu an
kullandığımız düz `runc`'tan daha güçlü) — her claim'de, önceki çalıştırmanın durumu
SİLİNİYOR ama ağ/kernel izolasyonu (havuzdan geldiği için) KALICI olarak duruyor.

**Kaynaklar:**
- [Running Agents on Kubernetes with Agent Sandbox — kubernetes.io](https://kubernetes.io/blog/2026/03/20/running-agents-on-kubernetes-with-agent-sandbox/)
- [Bringing you Agent Sandbox on GKE and Agent Substrate — Google Cloud Blog](https://cloud.google.com/blog/products/containers-kubernetes/bringing-you-agent-sandbox-on-gke-and-agent-substrate)
- [Red Hat build of Agent Sandbox — Red Hat Developer](https://developers.redhat.com/articles/2026/07/15/red-hat-build-agent-sandbox-isolated-workload-management-kubernetes)

## 3. Kullanıcının kendi fikri — "tek pod'da bir sürü sandbox container"

Bunu Cilium'un KENDİ dokümantasyonundan doğrulayarak değerlendirdim:

> *"Multiple application containers can share the same IP address in a Kubernetes Pod,
> and all application containers which share a common address are grouped together in
> what Cilium refers to as an endpoint... containers within the same pod (sharing the
> same network namespace) will typically share the same endpoint identity and
> therefore the same security policies."*

**Yani teknik olarak MÜMKÜN ama bir ayrım yapmak gerekiyor:**
- **Ağ politikası açısından SORUN YOK** — zaten TÜM sandbox çalıştırmalarımız aynı
  `sandbox-egress` kuralını paylaşıyor (hepsi `app: ptc-sandbox` etiketiyle aynı
  identity'yi alıyor). Tek pod'da 10 container olsa da, hepsi AYNI politikayı
  paylaşırdı — bu bizim durumumuzda bir KAYIP değil, zaten hep böyleydi.
- **Asıl risk, ağ değil, İZOLASYON** — aynı pod'daki farklı container'lar, AYNI ağ
  namespace'ini (ve IP'yi) paylaşsa da, kendi dosya sistemleri/süreç alanları ayrı
  kalır (container = ayrı namespace, POD = paylaşılan network namespace). Sorun şu:
  bir çalıştırmanın kodu, "kendi" container'ından TAŞIP komşu container'a bir şekilde
  erişmeye çalışırsa (ör. paylaşılan bir volume üzerinden), bu bizim şu anki
  "her run tamamen izole, hiç iz bırakmaz" garantimizi ZAYIFLATIR.
- **Pratik kazanç, warm-pool'dan DAHA AZ net:** Bir pod'da N container önceden
  başlatılmış olsa bile, HANGİ container'ın "boş" olduğunu takip etmek, ve çalışma
  bitince o container'ı GERÇEKTEN temiz bir duruma döndürmek (yeniden başlatmadan)
  ayrı bir mühendislik problemi — warm-pool yaklaşımı (tüm pod'u recycle etmek) daha
  basit bir garanti veriyor.

**Değerlendirme:** Bu fikir, Cilium açısından engellenmiş DEĞİL ama SIG-Apps'in
warm-pool yaklaşımı, aynı hedefe (hızlı + izole) daha net bir çözüm sunuyor —
"tek pod, çok container" yerine "çok pod, önceden ısıtılmış havuz" tercih edilen
desen olmuş.

## 4. Üçüncü bir seçenek — Tool Gateway'e benzer, KALICI bir "sandbox executor"

Araştırma sırasında ortaya çıkan, projenin KENDİ mimarisinden ilham alan bir fikir:
Tool Gateway zaten kalıcı, tek bir pod — sandbox'ı da AYNI ŞEKİLDE (Job yerine bir
Deployment, kod MCP/HTTP isteğiyle gönderilir, bir subprocess'te çalıştırılır) kurmak,
Kubernetes'in Job/scheduling maliyetini TAMAMEN ortadan kaldırır.

**Bunun BÜYÜK bir dezavantajı var — mevcut tasarımın asıl gücüyle ÇELİŞİYOR:**
Deck'teki "Bir Sandbox'ın Ömrü" slaydının (17/25) tam savı: *"hiçbir zaman kalıcı iz
bırakmaz."* Kalıcı bir sandbox pod'u, bu iddiayı YOK EDER — artık her çalıştırma
sıfırdan, dokunulmamış bir ortamda değil, ÖNCEKİ çalıştırmaların izini taşıyabilecek
paylaşılan bir process içinde olur. Bu, hız için ödenen gerçek bir güvenlik/temizlik
bedeli.

## 5. Hafif izolasyon runtime'ları — gVisor, Kata Containers, Firecracker

Bunlar, "Agent Sandbox" projesinin de kullandığı (bölüm 2), container'ı düz `runc`'tan
(bizim şu an sandbox image'ımızın kullandığı, hiçbir ekstra izolasyon katmanı olmayan
varsayılan runtime) DAHA GÜÇLÜ izole eden runtime'lar:

| Runtime | Mekanizma | Başlangıç süresi (kaynaklara göre değişken) |
|---|---|---|
| **gVisor** (`runsc`) | Sistem çağrılarını userspace'te yakalayıp filtreler — tam bir VM değil, ama kernel'e doğrudan erişimi keser | ~50-100ms (bazı kaynaklar) / ~680ms (başka bir akademik çalışma, tam container+app hazır olma süresi) |
| **Kata Containers** | Her pod'u hafif bir VM İÇİNDE çalıştırır — kendi kernel'i olur | ~150-300ms / ~1.9sn (aynı akademik çalışma) |
| **Firecracker** | AWS Lambda'nın kullandığı, KVM tabanlı minimal microVM | ~100-200ms / ~2.4sn (aynı çalışma) |

**Rakamlar neden bu kadar tutarsız?** Farklı kaynaklar farklı şeyi ölçüyor — biri
sadece "VM/sandbox açılışı"nı, diğeri "container + içindeki uygulamanın TAMAMEN hazır
olması"nı ölçüyor. Kesin bir sayı vermek yerine, bunu dürüstçe belirtmek daha doğru:
**gVisor en hızlısı, Kata ve Firecracker daha güçlü (VM-seviyesi) izolasyon için daha
yavaş bir bedel ödüyor.**

**Bizim asıl darboğazımızla İLİŞKİSİ — önemli bir düzeltme:** Bu üç teknoloji,
container/VM'in KENDİ açılış süresini konuşuyor — bizim ölçtüğümüz ~3sn'lik gecikme
ise ÖNCESİNDE olan Kubernetes scheduler + Cilium CNI (IPAM+identity+eBPF attach)
adımı. Yani gVisor/Kata/Firecracker'a geçmek, bizim asıl darboğazımızı ÇÖZMEZ —
tam tersine, Kata/Firecracker gibi VM-tabanlı olanlara geçmek, ÜSTÜNE ekstra
100-300ms'lik bir VM-açılış maliyeti BİNDİRİR. Bunlar **hız için değil, İZOLASYON
GÜCÜ için** tercih edilir — "Agent Sandbox" projesinin bunları warm-pool ile
BİRLİKTE kullanmasının sebebi tam bu: hız kaybını warm-pool'la telafi edip, güvenlik
kazancını (VM-seviyesi izolasyon) saf kâr olarak almak.

**Kaynaklar:**
- [Kata, gVisor, or Firecracker? Container Isolation Guide — Edera](https://edera.dev/stories/kata-vs-firecracker-vs-gvisor-isolation-compared)
- [Kata Containers vs Firecracker vs gVisor — Northflank](https://northflank.com/blog/kata-containers-vs-firecracker-vs-gvisor)

## 6. Hafif Kubernetes dağıtımları — k3s, MicroK8s, minikube (kind'a alternatif)

| Dağıtım | Karakteri | Üretime uygun mu |
|---|---|---|
| **kind** (bizim kullandığımız) | Docker container'ları içinde K8s — CI/test için tasarlanmış, ÇOK hızlı kurulum | Hayır — "gerçek kullanıcılar için değil, Kubernetes'i test etmek için" |
| **k3s / k3d** | Rancher'ın hafif dağıtımı — CNCF sertifikalı, gerçekten üretime uygun | Evet |
| **MicroK8s** | Canonical'ın tek-komutla kurulan dağıtımı — modüler, edge/küçük cihazlar için de uygun | Evet |
| **minikube** | VM içinde tek-node cluster — öğrenme/deneme amaçlı, diğerlerinden daha yavaş başlıyor | Sınırlı |

**Önemli bir düzeltme — bu karşılaştırma bizim sorumuzu ÇÖZMÜYOR:** Yukarıdaki
karşılaştırmalar hep **"cluster'ın kendisi ne kadar hızlı AYAĞA KALKIYOR"**yu
ölçüyor (`kind create cluster` gibi) — bu bizim PoC'de sadece BİR KERE ödediğimiz bir
maliyet (kurulumda). Bizim asıl sorumuz farklı: **zaten ÇALIŞAN bir cluster'da, YENİ
bir pod'un ne kadar hızlı schedule+network kurulumu** yapıldığı — bu, `kind` mi `k3s`
mi kullandığımızdan bağımsız, kubelet + CNI eklentisinin (bizde Cilium) kendi hızına
bağlı. Yani `kind`'dan `k3s`'e geçmek, bizim ~7 saniyelik per-request maliyetimizi
DEĞİŞTİRMEZ — sadece cluster'ı ilk kurarken birkaç saniye kazandırabilir ya da
kaybettirebilir, gerçek konu ise per-request. Bu yüzden bölüm 2'deki
`SandboxWarmPool` yaklaşımı, dağıtım değişikliğinden çok daha alakalı bir çözüm.

**Kaynaklar:**
- [Minikube vs MicroK8s vs Kubeadm vs Kind vs K3s — Medium](https://mohamedyassine-bensaid.medium.com/minikube-vs-microk8s-vs-kubeadm-vs-kind-vs-k3s-5a8714c6835f)
- [MicroK8s vs k3s vs Minikube — Canonical](https://canonical.com/microk8s/compare)

## 7. Daha da radikal seçenek — Firecracker'ın SNAPSHOT/RESTORE özelliği

Az önce sadece "Firecracker açılışı ~100-200ms" dedik (bölüm 5) — ama bu, Firecracker'ın
EN hızlı özelliğini atlıyor. **Snapshot/restore**: bir microVM'i bir kere aç,
hazır hâle gelince (Python yorumlayıcısı yüklenmiş, bağlantı kurulmuş) TÜM
belleğinin/CPU durumunun bir "dondurulmuş kare"sini (snapshot) diske al — sonraki
her istekte, sıfırdan AÇMAK yerine bu donmuş kareyi GERİ YÜKLE.

**Gerçek, ölçülmüş rakamlar (AWS Lambda SnapStart, bizzat Firecracker'ı kullanıyor):**
- Snapshot restore: **p50 3.2ms, p99 8.7ms**
- Somut bir örnek: toplam 28ms restore (sadece 4ms'si snapshot yükleme) — normal
  soğuk açılışın (>1sn) ~40 KATI hızlı
- Java Lambda'larda soğuk başlangıç 8+ saniyeden **saniyenin altına** düşürülmüş

**Bizim senaryomuza uyarlarsak:** Bizim ~3sn'lik "Kubernetes+Cilium setup"
maliyetimiz yerine, ÖNCEDEN bir kere açılıp "hazır" hâle getirilmiş (Tool Gateway'e
bağlanmaya hazır, Python yorumlayıcısı yüklü) bir microVM anlık görüntüsü tutulsa,
her PTC çağrısı bu görüntüyü **tek haneli milisaniyelerde** geri yükleyebilir —
teorik olarak ~7 saniyelik maliyeti ~10-30 milisaniyeye indirebilir.

**Ağ politikası (Cilium) bu senaryoda hâlâ mümkün mü?** Araştırdım — **evet, Cilium
Kubernetes'e bağımlı değil**: *"Network policies can be defined using Kubernetes
NetworkPolicy resources OR directly imported into the agent via CLI or API."* Yani
Kubernetes'i TAMAMEN devre dışı bırakıp, Firecracker'ın her microVM'ine kendi `tap`
ağ arayüzünü verip, Cilium'un eBPF motorunu (Kubernetes olmadan, doğrudan CLI/API
ile) o arayüze bağlamak TEORİK olarak mümkün.

**Ama bu, "birkaç ayar değiştirme" değil — mimariyi BAŞTAN YAZMA seviyesinde bir iş:**
Kubernetes Job/Pod modelini tamamen terk edip kendi Firecracker orkestrasyonumuzu
(VM havuzu, snapshot yönetimi, Cilium'un Kubernetes-dışı entegrasyonu) sıfırdan
kurmak gerekir — `SandboxWarmPool` gibi (bölüm 2) HAZIR, Kubernetes-native bir
çözümü benimsemekten ÇOK daha büyük bir mühendislik yükü.

**Kaynaklar:**
- [AWS Lambda SnapStart: Reducing Cold Start Times with Firecracker — ElasticScale](https://elasticscale.com/blog/aws-lambda-snapstart-reducing-cold-start-times-with-firecracker/)
- [Firecracker snapshot support — GitHub](https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/snapshot-support.md)

## 8. Tamamen farklı bir paradigma — WebAssembly (WASM/WASI)

Container/VM'den TAMAMEN farklı bir izolasyon modeli: kod, işletim sistemi
seviyesinde (namespace/cgroup/VM) değil, **DİL/BYTECODE seviyesinde** izole
ediliyor — WASM modülü, bellek güvenliğini ve hangi "yeteneklere" (capability)
sahip olduğunu (WASI standardı) ÇALIŞTIRMA ZAMANININ kendisi garanti ediyor.

**Hız:** *"Spinning up a WASM sandbox is fast and low-overhead compared to
launching a full container or VM"* — container'dan/VM'den daha da hafif, çünkü
kernel'in kendisi hiç devreye girmiyor (namespace/cgroup kurmuyor), sadece bir
bytecode yorumlayıcısı başlıyor.

**Bizim mimarimizle ilginç bir paralellik — aslında ZATEN benzer bir şey
yapıyoruz:** WASI'nin güvenlik modeli, sandboxlanmış kodun DIŞARIYA çıkmasının
TEK yolunun, host'un ona AÇIKÇA enjekte ettiği fonksiyonlar (capability) olması
— tıpkı bizim `entrypoint.py`'nin `search_knowledge_base`/`fetch_url` gibi
fonksiyonları sandbox'ın global namespace'ine enjekte etme deseni gibi! Fark şu:
WASM'da kod gerçekten HİÇBİR ham socket/syscall çağıramıyor (dil seviyesinde
engelli) — bizim mevcut sandbox'ımızda ise (`entrypoint.py`'nin başındaki yorum
bunu açıkça söylüyor) Python kodu TEORİK olarak `socket`/`requests` import edip
DENEYEBİLİR, sadece Cilium ağ seviyesinde engelliyor. WASM'a geçilseydi, bu
ikinci savunma hattına (Cilium) belki hiç gerek kalmazdı — ama bu, "LLM'in
ürettiği herhangi bir Python kodunu çalıştır" esnekliğini kaybetmek demek (WASM'a
geçmek, Python'u WASM'a derlemek/yorumlamak gibi kendi başına büyük bir teknik
sorun).

**Kaynaklar:**
- [Building a Secure Code Sandbox for LLMs with WebAssembly — Medium](https://medium.com/collaborne-engineering/building-a-secure-code-sandbox-for-llms-with-webassembly-bdd91a835f23)
- [Security — WebAssembly.org](https://webassembly.org/docs/security/)

## 9. Karşılaştırma tablosu

| Yaklaşım | Hız kazancı | Ana bedel |
|---|---|---|
| **Şu anki (her run = yeni Job)** | — (referans, ~7sn) | Yavaş ama HER çalıştırma sıfırdan, kanıtlanabilir temiz |
| **SandboxWarmPool (K8s SIG-Apps)** | ~%90 (kubernetes.io'nun kendi rakamı), sub-second | Ekstra bir CRD/controller kurulumu; boşta duran pod'lar kaynak tüketir |
| **Tek pod, çok container** | Muhtemel ama ölçülmedi, warm-pool'dan daha az net | İzolasyon garantisi zayıflar (paylaşılan network namespace) |
| **Kalıcı sandbox executor (Tool Gateway gibi)** | En hızlı (Job/CNI maliyeti sıfıra iner) | "Hiç iz bırakmaz" iddiası tamamen kaybolur |
| **Firecracker snapshot/restore** | ~10-30ms (p50 3.2ms restore + bağlantı kurma) — **en hızlı ölçülebilir seçenek** | Kubernetes'i bırakıp özel bir orkestrasyon yazmak gerekir — büyük mühendislik yükü |
| **WASM/WASI** | Muhtemelen en hafif (dil-seviyesi, kernel bile devreye girmiyor) | Python'un WASM'a taşınması kendi başına büyük bir problem; Cilium'a ihtiyaç azalır ama esneklik kaybı var |

## 10. Sonuç

En OLGUN, en az güvenlik ödünü gerektiren seçenek **`SandboxWarmPool`** deseni —
çünkü hız kazancını, HER claim'de "temiz durum" garantisini bozmadan sağlıyor (Kubernetes
SIG-Apps'in kendi projesinin tam amacı bu). Kullanıcının "tek pod'da çoklu container"
fikri Cilium açısından mümkün ama daha az net bir kazanç/risk oranı sunuyor. "Kalıcı
executor" en hızlı ama PoC'nin temel tezinin bir parçasını (ephemeral/iz bırakmama)
feda ediyor — bu, sadece bilinçli bir trade-off olarak, gerekçesiyle birlikte
seçilmeli, sessizce değil.

**"Hep böyle mi oluyor, daha optimize hâli yok mu" sorusunun cevabı — HAYIR, var,
ama bedeli büyüyor:** Firecracker'ın düz açılışı (bölüm 5) ve k3s/MicroK8s gibi
dağıtımlar (bölüm 6) bu sorunun cevabı DEĞİL — ama Firecracker'ın **snapshot/
restore** özelliği (bölüm 7, gerçekten ölçülmüş p50 3.2ms) ve **WASM/WASI**
(bölüm 8) GERÇEKTEN çok daha optimize — ~7 saniyeden tek haneli milisaniyelere
inebiliyor. Bedel de o oranda büyüyor: bunlar Kubernetes'in Job/Pod modelini
TAMAMEN terk edip özel bir orkestrasyon yazmayı gerektiriyor — `SandboxWarmPool`
gibi hazır bir Kubernetes CRD'sini benimsemekten çok daha büyük bir mühendislik
yatırımı. Bu PoC'nin ölçeğinde (demo amaçlı) muhtemelen gereğinden fazla; gerçek
bir üretim sistemine dönüşseydi, ilk bakılacak yer olurdu.

**gVisor/Kata/Firecracker'ın DÜZ açılışı VE k3s/MicroK8s/minikube — ikisi de bu
sorunun cevabı DEĞİL (yukarıdaki radikal seçeneklerden ayrı tutulmalı):** İlk
grup, hız değil İZOLASYON GÜCÜ katıyor (hatta VM-tabanlı olanlar hafif
bir yavaşlatma bile getiriyor); ikinci grup ise cluster'ın İLK kurulma hızını
konuşuyor, bizim per-request maliyetimizi değil. Bu iki araştırma yolu, sorunun kendisi
kadar, **hangi katmanda olduğunu netleştirmek** açısından değerli oldu: darboğaz
"hangi Kubernetes dağıtımını kullanıyoruz" değil, "her seferinde sıfırdan bir pod
network'ü kurmamız" — çözüm de bu yüzden `SandboxWarmPool` gibi, aynı katmanı
hedefleyen bir yaklaşımda.
