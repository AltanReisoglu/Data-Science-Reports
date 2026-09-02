# PoC'nin Tüm Kubernetes Yapısı

Bu doküman, PoC'nin cluster/node seviyesinden tek tek kaynaklara kadar tüm Kubernetes
yapısını açıklıyor. Kaynak: `k8s/` altındaki manifestler + canlı cluster'dan doğrulanmış
(`kubectl get all -A` vb.) güncel durum.

## 1. Cluster ve Node seviyesi

**Cluster**: `kind` ile oluşturulan yerel bir cluster (`k8s/kind-config.yaml`) — **1
node**, o node hem control-plane hem worker (kind'ın tek-node modu). `disableDefaultCNI:
true` kritik — kind'ın kendi varsayılan CNI'sini (kindnet) devre dışı bırakıyoruz ki
Cilium'u biz kuralım (ikisi birden çalışamaz).

**Node**: `ptc-sec-control-plane` — gerçekte bu bir Docker/containerd container'ı (kind,
"node"ları gerçek makine değil, container olarak simüle eder). Tüm pod'larımız bu TEK
node üzerinde çalışıyor.

## 2. Namespace'ler

- **`default`** — bizim tüm kaynaklarımızın yaşadığı yer (tool-gateway, sandbox
  job'ları, policy'ler)
- **`kube-system`** — Kubernetes'in kendi altyapısı + Cilium/Hubble bileşenleri
- `kube-node-lease`, `kube-public`, `local-path-storage`, `cilium-secrets` — bunlar
  bizim eklemediğimiz, kind/Cilium'un varsayılan kurulumuyla gelen namespace'ler
  (secrets için Cilium'un kendi sertifika/anahtar deposu, local-path-storage PVC'ler
  için varsayılan storage class sağlayıcısı — bizim PoC'de PVC kullanmadığımız için
  boş).

## 3. Bizim `default` namespace'teki kaynaklarımız

### Deployment → ReplicaSet → Pod (tool-gateway)

`k8s/tool-gateway/deployment.yaml` — **Deployment**, "bu pod'u sürekli ayakta tut, kaç
kopya istiyorum, hangi image'la" tarifi. Deployment kendisi pod çalıştırmaz — bir
**ReplicaSet** oluşturur, ReplicaSet de asıl **Pod**'u oluşturur ve canlı tutar (biri
ölürse yenisini açar).

`kubectl get all` çıktısında birden çok eski ReplicaSet görülebilir (hepsi
`DESIRED:0`) — bunlar hayalet değil, bu oturumda tool-gateway'i kaç kere yeniden build
edip deploy ettiğimizin (rename işlemleri, `rollout restart`'lar, image güncellemeleri)
kaydı. Deployment, her değişiklikte YENİ bir ReplicaSet oluşturur, eskisini 0'a düşürür
ama SİLMEZ — `revisionHistoryLimit` (varsayılan 10) kadarını geri-alma (rollback) için
saklar. Zararsız, sadece geçmiş.

### Service (tool-gateway)

`k8s/tool-gateway/service.yaml` — bir **ClusterIP Service**. Pod'ların IP'si her
yeniden başlamada değişir; Service, `app: tool-gateway` etiketine sahip pod'u/pod'ları
**sabit bir IP+port** (ör. `10.96.228.218:8443`) arkasında toplar. Sandbox'ın koddaki
bağlantısı bu sabit IP'ye gider (DNS'e gerek kalmadan — `sandbox_runner.py` bu IP'yi
doğrudan enjekte ediyor).

### Secret (tool-gateway-env)

YAML dosyası olarak REPO'da YOK — bilerek `kubectl create secret generic
tool-gateway-env --from-env-file=.env` ile **imperative** (komutla, dosyasız)
oluşturuluyor, çünkü `.env` gizli bilgi (LLM API anahtarı) içeriyor, asla YAML'a/git'e
gömülmemeli. Deployment, `envFrom.secretRef` ile bunu container'a ortam değişkeni
olarak enjekte ediyor.

### Job + ConfigMap (ptc-sandbox, ptc-code) — **EPHEMERAL, aktif çalıştırma yokken YOK**

Bunlar kalıcı kaynaklar DEĞİL — her PTC çalıştırmasında sıfırdan oluşturulup saniyeler
içinde silinirler:

- **ConfigMap** (`ptc-code-{run_id}`) — LLM'in ürettiği Python kodunu tutan geçici bir
  "dosya"
- **Job** (`ptc-sandbox-{run_id}`, `k8s/sandbox/job-template.yaml`'dan) — bu ConfigMap'i
  mount eden, kodu çalıştıran, **`activeDeadlineSeconds: 30`** (en fazla 30sn yaşar) ve
  **`backoffLimit: 0`** (hata olursa yeniden deneme YOK) ile sınırlı bir tek-seferlik
  çalıştırma. Job bir Pod oluşturur (`restartPolicy: Never`); iş bitince
  (`ttlSecondsAfterFinished: 300`) K8s kendisi 5dk sonra siler, ama bizim kodumuz
  (`_cleanup()`) zaten hemen, açıkça siliyor.

Aktif bir PTC çalıştırması yokken `kubectl get all` çıktısında bunlar GÖRÜNMEZ — bu
normal, kalıcı olmaları da beklenmiyor.

### CiliumNetworkPolicy × 3

Kubernetes'in kendi native kaynak türü değil, Cilium'un eklediği bir **CRD** (Custom
Resource Definition) — ama `kubectl get`/`apply` ile aynı şekilde yönetiliyor:

- **`sandbox-egress`** — sandbox pod'una uygulanır, tek izni: Tool Gateway'e 8443
- **`tool-gateway-egress`** — tool-gateway pod'una uygulanır, DNS serbest + 3 onaylı
  FQDN'e 443 (artık `serverNames`/SNI kontrolüyle)
- **`tool-gateway-ingress`** — henüz `apply` edilmedi (dosya hazır, cluster'da yok) —
  tool-gateway'e SADECE sandbox'tan gelen bağlantıyı kabul et der

## 4. `kube-system`'daki bileşenler (bizim yazmadığımız, ama PoC'nin çalışması için gerekli olanlar)

| Kaynak | Tür | Ne işe yarıyor |
|---|---|---|
| `cilium` | DaemonSet | Her node'da 1 kopya çalışan asıl CNI ajanı — eBPF programlarını yükler, policy'leri uygular |
| `cilium-envoy` | DaemonSet | L7/SNI kararları için gereken proxy (SNI/serverNames, DNS sorgu adı kontrolü buradan geçiyor) |
| `cilium-operator` | Deployment | Cluster-çapında Cilium'un koordinasyon işleri (IP havuzu yönetimi vb.) |
| `coredns` | Deployment (2 kopya) | Cluster'ın DNS sunucusu — Tool Gateway'in DNS sorguları buraya gider |
| `kube-proxy` | DaemonSet | Kubernetes'in VARSAYILAN Service yönlendirme mekanizması — hâlâ AKTİF çünkü `kubeProxyReplacement=false` (Ingress Controller'ın gerektirdiği ön koşul eksik) |
| `hubble-relay` | Deployment + Service | Tüm node'lardaki (bizde 1 tane) Hubble akışlarını TEK bir API'de toplar — `hubble` CLI/`cilium hubble port-forward` buraya bağlanır |
| `hubble-ui` | Deployment + Service | Tarayıcıdan görsel akış izleme arayüzü — relay'e bağlanır |
| `hubble-peer` | Service | Cilium ajanları ile Hubble arasındaki mTLS bağlantı noktası |

## Özet — parçalar nasıl bir araya geliyor

```
kind cluster (1 node)
├─ kube-system: Cilium (DaemonSet) — TÜM ağ trafiğinin kernel-seviyesi karar noktası
│               + cilium-envoy (L7/SNI) + Hubble (relay+UI, gözlem)
└─ default:
   ├─ tool-gateway (Deployment+Service+Secret) — kalıcı, tek dış kapı
   ├─ ptc-sandbox-{run_id} (Job+ConfigMap) — geçici, her soru için doğar/ölür
   └─ 3× CiliumNetworkPolicy — kimin kime, hangi portta, hangi isimle konuşabileceğini tarif eder
```
