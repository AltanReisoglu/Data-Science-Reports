# PTC Egress Policy — Cilium/eBPF Teknik Referans

Bu doküman, depodaki üç kavramsal dosyanın (`PTC_egress_policy_eBPF_Cilium.md`,
`PTC_Egress_Policy_OpenAI_Incident.md`, `PTC_egress_policy_eBPF_Cilium_addendum.md`)
**üzerine** kurulu, somut teknik/uygulama referansıdır. Onların yerine geçmez —
"neden bu mimari" sorusunu onlar cevaplıyor, "bu mimari teknik olarak tam olarak
nasıl kurulur" sorusunu bu doküman cevaplıyor.

Faz 3'ün (bu deponun 4 fazlı yol haritasının egress-policy fazı) hazırlık materyalidir.

---

## 1. Nedir?

**eBPF (extended Berkeley Packet Filter)**, Linux kernel'in içinde, kernel kaynak
kodunu değiştirmeden veya bir kernel modülü yüklemeden, kullanıcı alanından
yazılıp yüklenen **sandboxed programların** çalışmasını sağlayan bir mekanizmadır.

**Cilium**, bu eBPF mekanizmasını Kubernetes/cloud-native ortamlarda **networking
ve security policy enforcement** için kullanan bir platformdur (CNI — Container
Network Interface eklentisi).

**Kubernetes NetworkPolicy**, bunların üzerine oturan, "hangi pod hangi pod'a/
nereye konuşabilir" kuralını tanımlayan standart Kubernetes API kaynağıdır — ama
kendisi hiçbir şeyi *uygulamaz*, uygulama işini Cilium gibi bir CNI yapar.

İlişkiyi tek cümlede: **eBPF bir kernel mekanizması, Cilium onu kullanan bir
platform, NetworkPolicy/CiliumNetworkPolicy ise o platforma verilen kurallardır.**

---

## 2. Sezgisi

Egress policy'yi anlamanın en kısa yolu şu iki mimariyi karşılaştırmak:

```
❌ Naif model:                    ✅ Onaylı-kanal modeli:

Agent                              Agent
  │                                  │
  │ (herhangi bir yere)              │ PTC / Tool
  ▼                                  ▼
Internet                          Tool Gateway (authz, validation)
                                     │
                                     ▼
                                   Approved API
                                     │
                                     ▼  (network seviyesinde AYRICA doğrulanır)
                                   eBPF/Cilium: ALLOW
                                     │
                                     ▼
                                   Internet (sadece o adrese)
```

Kritik sezgi: **uygulama katmanındaki (Tool Gateway) bir kural, agent kodu
tarafından atlatılabilir** (agent arbitrary kod çalıştırabiliyorsa, Tool
Gateway'i çağırmayıp doğrudan `socket()` açmayı deneyebilir). eBPF/Cilium bunu
**agent'ın kendi kontrol ettiği katmanın altında**, kernel'de uyguladığı için
agent'ın "bunu yapmama izin var mı" diye sorma ihtiyacı yoktur — deniyor, kernel
paketi düşürüyor. Bu, önceki üç dosyadaki "security control, untrusted agent'ın
kontrol ettiği katmanın dışında olmalı" ilkesinin somut hâli.

İkinci sezgi (OpenAI olayından, `..._OpenAI_Incident.md`'de detaylı): **agent'ın
kendisini kısıtlamak yetmez, agent'ın erişebildiği HER servisin de kendi egress
sınırı olmalı** — yoksa o servis bir SSRF köprüsüne dönüşür.

---

## 3. Teknik: eBPF nasıl çalışır?

### 3.1 Yükleme, doğrulama, derleme

1. Program, `bpf()` sistem çağrısı ile (genelde `libbpf` gibi bir kütüphane
   üzerinden) kernel'e yüklenir.
2. **Verifier**: Kernel, programı çalıştırmadan önce statik olarak analiz eder —
   sonsuz döngü yok mu, ilklendirilmemiş bellek erişimi yok mu, bellek sınırları
   aşılıyor mu, tüm olası yürütme yolları güvenli mi. Verifier bir *güvenlik*
   aracı değil, bir *emniyet* (safety) aracıdır — programın **ne yaptığını**
   değil, **güvenli çalışıp çalışmadığını** kontrol eder.
3. **JIT (Just-In-Time) derleme**: Doğrulanan bytecode, makineye özgü komut
   setine derlenir — kernel modülü kadar hızlı çalışır.

### 3.2 Hook noktaları (nereye takılır)

| Hook | Konum | Kullanım |
|---|---|---|
| **XDP** (eXpress Data Path) | Ağ kartı sürücüsü seviyesi, en erken nokta | En hızlı filtreleme/drop, DDoS koruması |
| **TC** (Traffic Control) | Kernel network stack'i, ingress/egress | Cilium'un asıl policy enforcement noktası |
| **Socket-seviyesi** | `connect()`/`sendto()` civarı | Uygulama-kernel arayüzü, servis mesh hızlandırma |
| **kprobe/uprobe** | Kernel/kullanıcı-alanı fonksiyonlarına dinamik attach | Gözlemlenebilirlik, güvenlik izleme |

### 3.3 Güvenlik sertleştirmeleri

- Programın çalıştığı kernel belleği **salt-okunur** yapılır.
- **Spectre mitigasyonu**: JIT derleyici, spekülatif yürütme saldırılarına karşı
  Retpoline üretir.
- **Constant blinding**: JIT spraying saldırılarını engellemek için sabitler
  maskelenir.
- eBPF programları **rastgele kernel belleğine doğrudan erişemez** — sadece
  kernel'in sağladığı stabil **helper fonksiyonlar** üzerinden.
- Varsayılan olarak yükleme **root/`CAP_BPF`** gerektirir (unprivileged eBPF
  açık değilse).

### 3.4 Maps — durum paylaşımı

eBPF programları durumsuzdur (stateless); kalıcı/paylaşılan veri için **map**
kullanırlar: hash table, array, LRU, ring buffer (yüksek performanslı olay
akışı), LPM (longest-prefix-match — CIDR eşleştirmesi için). Map'lere hem
kernel'deki eBPF programı hem kullanıcı alanındaki bir process (ör. Cilium
agent) erişebilir — policy güncellemesi böyle akar.

---

## 4. Teknik: Cilium mimarisi

```
┌─────────────────────────────────────────────────┐
│ Kubernetes API Server                            │
└──────────────────┬────────────────────────────────┘
                    │ NetworkPolicy / CiliumNetworkPolicy
                    ▼
┌─────────────────────────────────────────────────┐
│ Cilium Operator (cluster-geneli IPAM, koordinasyon)│
└──────────────────┬────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│ Cilium Agent (her node'da bir DaemonSet)          │
│  - policy'yi izler, eBPF programlarına çevirir    │
└──────────────────┬────────────────────────────────┘
                    │ eBPF program yükler
                    ▼
┌─────────────────────────────────────────────────┐
│ Kernel: eBPF datapath                             │
│  1. Identity Map   — paket kimden geliyor?         │
│  2. Policy Map     — bu identity, bu hedefe gidebilir mi? │
│  3. (L7 gerekirse) → Envoy proxy'ye yönlendir       │
│  4. Conntrack Map  — bağlantı takibi                │
└─────────────────────────────────────────────────┘
```

### 4.1 Identity-based security — Cilium'u farklı kılan şey

Klasik firewall'lar IP adresine göre karar verir. Kubernetes'te pod IP'leri
sürekli değiştiği için bu kırılgandır. Cilium bunun yerine her pod'a (label'larına
göre) bir **security identity** atar; policy IP değil identity üzerinden
yazılır ("`role=frontend` olan her şey `role=db`'ye 5432'den bağlanabilir" —
pod yeniden başlasa IP değişse bile kural geçerli kalır).

### 4.2 L3/L4 vs L7

- **L3/L4** (IP + port + protokol): tamamen eBPF datapath'inde, kernel'de
  karar verilir — en hızlı yol.
- **L7** (HTTP method/path, gRPC, Kafka, DNS): eBPF, ilgili trafiği kernel'den
  kullanıcı-alanındaki bir **Envoy proxy**'ye yönlendirir; asıl L7 karar orada
  verilir. Bu, L7 filtrelemenin L3/L4'e göre bir performans maliyeti olduğu
  anlamına gelir — bilinçli bir tradeoff.

### 4.3 Cilium vs Calico (kısa)

Calico, birden fazla data-plane'i (eBPF, standart Linux iptables/nftables,
Windows HNS, VPP) opsiyonel olarak destekler — "pluggable". Cilium tasarımı
gereği eBPF'ye yoğun şekilde bağımlıdır; bu, daha derin kernel entegrasyonu
(L7 farkındalığı, identity-based model) sağlarken, platform bağımlılığı ve
daha karmaşık kurulum/debug maliyeti getirir.

---

## 5. Nasıl uygulanır?

### 5.1 Standart Kubernetes NetworkPolicy (CNI-agnostik, temel)

```yaml
# Varsayılan-reddet: bu namespace'teki her pod'un egress'i kapalı başlar
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-egress
  namespace: agent-workloads
spec:
  podSelector: {}
  policyTypes:
  - Egress
---
# Sadece Tool Gateway'e ve DNS'e izin ver
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-agent-egress
  namespace: agent-workloads
spec:
  podSelector:
    matchLabels:
      app: ptc-agent
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: tool-gateway
    ports:
    - protocol: TCP
      port: 8443
  - to:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: kube-system
    ports:
    - protocol: UDP
      port: 53
```

Önemli notlar (Kubernetes'in kendi belgesinden):
- `podSelector: {}` (boş) = namespace'teki **tüm** pod'ları seçer.
- Bir NetworkPolicy hiç uygulanmamışsa pod **non-isolated**'dır — yani her şeye
  izin vardır. İzolasyon, `policyTypes: [Egress]` içeren **en az bir** policy
  var olduğu an başlar.
- Kurallar **toplamsaldır** (additive) — çelişmezler, izin verenler birleşir.
- **NetworkPolicy kendi başına hiçbir şey yapmaz** — bir CNI eklentisi (Cilium,
  Calico...) onu okuyup uygulamak zorundadır.

### 5.2 CiliumNetworkPolicy — FQDN/DNS-farkında egress (bizim asıl ihtiyacımız)

Standart K8s NetworkPolicy yalnızca IP/CIDR ile egress tanımlayabilir — ama bir
"approved API" çoğu zaman bir domain adıdır (`api.anthropic.com` gibi), IP'si
değişebilir. Cilium'un **FQDN policy**'si bunu domain adıyla çözer:

```yaml
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: agent-allow-approved-api-only
  namespace: agent-workloads
spec:
  endpointSelector:
    matchLabels:
      app: ptc-agent
  egress:
  # DNS çözümlemesine izin ver (FQDN policy'nin çalışması için gerekli ön koşul)
  - toEndpoints:
    - matchLabels:
        k8s:io.kubernetes.pod.namespace: kube-system
        k8s-app: kube-dns
    toPorts:
    - ports:
      - port: "53"
        protocol: UDP
      rules:
        dns:
        - matchPattern: "*"
  # Sadece belirli domain'lere (wildcard destekli) TCP/443
  - toFQDNs:
    - matchName: "api.anthropic.com"
    - matchPattern: "*.googleapis.com"
    toPorts:
    - ports:
      - port: "443"
        protocol: TCP
  # Tool Gateway'e (küme-içi) izin
  - toEndpoints:
    - matchLabels:
        app: tool-gateway
    toPorts:
    - ports:
      - port: "8443"
        protocol: TCP
  # Geri kalan her şey (dünya) — reddedilir (default-deny zaten, açık DENY de eklenebilir)
```

`toEntities: ["world"]` özel bir hedef olarak "küme dışındaki her yer" anlamına
gelir — genelde bunu **açıkça reddetmek** (ya da hiç izin listesine almamak,
default-deny zaten yeterli) istenir.

### 5.3 L7 (HTTP) örneği

```yaml
egress:
- toEndpoints:
  - matchLabels:
      app: tool-gateway
  toPorts:
  - ports:
    - port: "8443"
      protocol: TCP
    rules:
      http:
      - method: "POST"
        path: "/v1/tools/search"
      - method: "POST"
        path: "/v1/tools/get_ticket_status"
```

Bu, sadece "Tool Gateway'e TCP bağlantısı" değil, "Tool Gateway'in **sadece şu
iki endpoint'ine** POST" seviyesinde bir kısıtlama sağlar — network katmanında
bile hangi tool'un çağrılabileceği daraltılmış olur (bizim `tool_policy.py`'deki
uygulama-seviyesi allowlist'in network-seviyesi bir yansıması).

### 5.4 Kurulum (Helm ile, özet)

```bash
helm repo add cilium https://helm.cilium.io
helm install cilium cilium/cilium --version 1.19.5 \
  --namespace kube-system \
  --set kubeProxyReplacement=true \
  --set hubble.enabled=true \
  --set hubble.relay.enabled=true \
  --set hubble.ui.enabled=true
```

Doğrulama/gözlemleme:

```bash
kubectl -n kube-system exec ds/cilium -- cilium policy list
kubectl -n kube-system exec ds/cilium -- cilium bpf endpoint list
cilium-dbg monitor          # kernel-seviyesi olayları canlı izle
```

**Hubble**, Cilium'un gözlemlenebilirlik katmanıdır — hangi akışın ALLOW/DENY
edildiğini gerçek zamanlı gösterir; bizim `Trace`/izlenebilirlik prensibimizin
(Principle III) network-seviyesindeki karşılığı gibi düşünülebilir.

### 5.5 Önerilen üretim akışı (rollout)

1. **Audit mode** — policy'yi *uygula ama drop etme*, sadece logla; ne kadar
   trafik gerçekte neye gidiyor gözlemlenir.
2. **Enforce mode** — audit sonuçlarına göre allowlist netleşince gerçekten
   uygula.
3. **Monitor** — Hubble/Prometheus ile sürekli izle.
4. **Break-glass** — acil durumda policy'yi devre dışı bırakma prosedürü
   önceden tanımlanmış olmalı (bir policy hatası tüm agent'ı kilitleyebilir).

---

## 6. Gereksinimler

| Katman | Gereksinim |
|---|---|
| **Kernel** | Modern bir Linux kernel (Cilium'un çoğu özelliği için 5.x+ önerilir; `kubeProxyReplacement` gibi bazı özellikler daha yeni kernel ister) |
| **Yetki** | eBPF programı yüklemek için root ya da `CAP_BPF` |
| **Orkestrasyon** | Kubernetes cluster (Cilium'un en doğal ortamı) — VM/microVM/process-seviyesi sandbox'larda Cilium'un kendisi değil, muadil mekanizmalar (cloud firewall, security group, seccomp+network namespace) kullanılır |
| **DNS** | FQDN policy kullanılacaksa, DNS trafiğinin de (53/UDP) egress kuralına dahil edilmesi — yoksa domain adları hiç çözülemez |
| **Gözlemlenebilirlik** | Hubble (+ opsiyonel Prometheus/Grafana) — policy'yi "kör" uygulamamak için pratikte neredeyse zorunlu |
| **Süreç** | Audit → Enforce → Monitor akışı; policy'lerin git'te versiyonlanması (bizim `research.md`/`plan.md` disiplinimizle aynı ruh) |

---

## 7. Sınırlamalar (dürüstçe)

- eBPF/Cilium, **network escape**'i çözer — sandbox escape, credential theft,
  agent'lar arası shared-state üzerinden gizli iletişim, hedef sürüklenmesi
  (goal drift) gibi **farklı katmanlardaki** sorunları çözmez (bkz.
  `PTC_egress_policy_eBPF_Cilium.md` §32 — Defense in Depth).
- Cilium'un derin kernel entegrasyonu, platform bağımlılığı ve daha karmaşık
  kurulum/debug maliyeti getirir (tigera.io'nun kendi karşılaştırmasında da
  vurgulanan bir nokta).
- L7 policy, trafiği Envoy'a yönlendirdiği için L3/L4'e göre ek gecikme
  getirir — her şeyi L7'de kısıtlamak istemek performans tradeoff'udur.
- FQDN policy, DNS yanıtının **TTL**'ine güvenir; TTL sona ermeden IP değişirse
  (bazı CDN/anycast senaryolarında) kısa süreli tutarsızlık olabilir.

---

## 8. Bizim projeyle (`ptc_sec`) bağlantı

Bu depo şu an Faz 1'i (kurumsal asistan) bitirdi, Faz 2'nin (gerçek PTC —
sandbox'ta kod çalıştırma) spec'i çıktı. **Faz 3, tam olarak bu dokümanın
konusu**: Faz 2'nin sandbox'ından çıkan trafiğin, uygulama-seviyesi
(`tool_policy.py`) kontrolüne ek olarak, network-seviyesinde de (burada
anlatılan CiliumNetworkPolicy/FQDN modeliyle) enforce edilmesi. Yani:

```
Faz 1: Tool Gateway (uygulama katmanı, HumanInTheLoopMiddleware)
Faz 2: Sandbox izolasyonu (process/interpreter katmanı)
Faz 3: eBPF/Cilium egress policy (kernel/network katmanı)  ← bu doküman
```

Üç katman birlikte, `PTC_egress_policy_eBPF_Cilium.md`'nin "Defense in Depth"
prensibini oluşturur — hiçbiri tek başına yeterli değil, biri atlatılırsa
bir sonraki devreye girer.

---

## 9. Kaynaklar

- `PTC_egress_policy_eBPF_Cilium.md`, `PTC_Egress_Policy_OpenAI_Incident.md`,
  `PTC_egress_policy_eBPF_Cilium_addendum.md` — bu deponun kendi kavramsal
  dokümanları (bu referansın temeli)
- [eBPF.io — What is eBPF?](https://ebpf.io/what-is-ebpf/) — eBPF'in kernel mekanizması (verifier, JIT, hook tipleri, güvenlik sertleştirmeleri)
- [Cilium — Network Policy (resmi docs)](https://docs.cilium.io/en/stable/security/policy/index.html) — identity-based model, egress kural türleri
- [Kubernetes — Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/) — NetworkPolicy API'sinin tam yapısı, default-deny örnekleri
- [Tektik.tr — Kubernetes Network Policies ve Cilium](https://tektik.tr/blog/kubernetes-network-policies-cilium) — CiliumNetworkPolicy'nin somut YAML örnekleri (FQDN, L7 HTTP/gRPC/Kafka, zero-trust rollout)
- [Tigera — Cilium vs Calico](https://www.tigera.io/learn/guides/cilium-vs-calico/cilium/) — Cilium'un mimari bileşenleri ve Calico ile karşılaştırma
- [OneUptime — eBPF Kubernetes Network Policies](https://oneuptime.com/blog/post/2026-01-07-ebpf-kubernetes-network-policies/view) — eBPF hook noktaları, iptables'a göre performans, Cilium'un Identity/Policy/Conntrack Map akışı
