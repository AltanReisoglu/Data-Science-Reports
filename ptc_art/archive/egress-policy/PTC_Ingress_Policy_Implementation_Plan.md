# CiliumNetworkPolicy Ingress Kuralı — Uygulama Planı

Bu doküman, `PTC_Egress_Policy_Implementation_Walkthrough.md`'un tamamlayıcısı. O doküman
baştan sona **egress**'i (pod'ların DIŞARI çıkışını) anlatıyor; bu doküman Cilium'un
**ingress** tarafını ele alıyor — hem kavramsal olarak hem de bu PoC'ye somut bir ekleme
olarak.

## 1. İki farklı "ingress" kavramı — karıştırılmamalı

| | **CiliumNetworkPolicy `ingress` kuralı** | **Cilium Ingress Controller / Gateway API** |
|---|---|---|
| Ne kısıtlar | Bir pod'a **GELEN** trafik (kim bağlanabilir) | Cluster **DIŞINDAN** gelen HTTP(S) trafiğinin yönlendirilmesi |
| Trafik yönü | Pod-to-pod (east-west) | Dünya-to-cluster (north-south) |
| Katman | L3/L4 (+ opsiyonel L7: HTTP path/method, DNS, SNI) | Her zaman L7 — Envoy zorunlu |
| Bizim egress kurallarımızla ilişkisi | Birebir simetrik (aynı `toPorts`/`rules` zenginliği, ters yön) | İlgisiz — tamamen ayrı bir bileşen |
| Bu PoC'de uygulanabilir mi | **Evet** — bu doküman bunu planlıyor | **Hayır** — cluster'a dışarıdan giren HTTP trafiği yok |

Kaynaklar:
- [Layer 3 Policies — Cilium 1.20 documentation](https://docs.cilium.io/en/stable/security/policy/layer3/)
- [Kubernetes Ingress Support — Cilium 1.20 documentation](https://docs.cilium.io/en/stable/network/servicemesh/ingress/)

## 2. Neden bunu ekliyoruz — mevcut boşluk

`tool-gateway-egress.ciliumnetworkpolicy.yaml`, Tool Gateway'in **egress**'ini (nereye
gidebileceğini) kısıtlıyor. Ama Tool Gateway'in **ingress**'i hiç kısıtlı değil — yani
"Tool Gateway'e kim bağlanabilir" sorusuna şu an cluster'ın hiçbir politikası cevap
vermiyor. Pratikte tek meşru istemci `ptc-sandbox` pod'u olduğu hâlde, teorik olarak
cluster içindeki BAŞKA bir pod da (yanlışlıkla ya da art niyetle) Tool Gateway'e
bağlanabilir.

Bu, `PTC_Egress_Policy_OpenAI_Incident.md`'nin dersiyle aynı ilke: *"agent'ın kısıtlı
olması yetmez, agent'ın eriştiği servisin de kendi sınırları olmalı"* — o doküman bunu
Tool Gateway'in egress'i için uyguladı, biz şimdi aynı ilkeyi ingress yönünde
tamamlıyoruz. Cilium'un kendi dokümantasyonu bunu şöyle özetliyor: *"policy must be
configured on both sides (sender and receiver)"* — sandbox'ın egress'i zaten var, eksik
olan Tool Gateway'in ingress'i.

## 3. Somut değişiklik — `tool-gateway-ingress.ciliumnetworkpolicy.yaml`

`sandbox-egress.ciliumnetworkpolicy.yaml` ile simetrik, yeni bir dosya:

```yaml
# Tool Gateway'in kendi ingress'i — "supporting service" ilkesinin ingress yönü
# (tool-gateway-egress.ciliumnetworkpolicy.yaml'ın egress'i kısıtlaması gibi, bu da
# "Tool Gateway'e kim bağlanabilir" sorusunu kısıtlıyor). Cilium: "policy must be
# configured on both sides" — sandbox-egress.ciliumnetworkpolicy.yaml zaten "sandbox
# sadece Tool Gateway'e gidebilir" diyordu; bu dosya "Tool Gateway sadece sandbox'tan
# gelen bağlantıyı kabul eder" diyerek aynı kuralı diğer taraftan tamamlıyor.
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: tool-gateway-ingress
spec:
  endpointSelector:
    matchLabels:
      app: tool-gateway
  ingress:
  - fromEndpoints:
    - matchLabels:
        app: ptc-sandbox
    toPorts:
    - ports:
      - port: "8443"
        protocol: TCP
```

`fromEndpoints` identity-based olduğu için (`toEndpoints` gibi), sandbox pod'u her
çalıştırmada yeniden doğsa, IP'si değişse bile kural geçerli kalıyor — IP tabanlı bir
kuralın aksine.

**Not — bu, mevcut hiçbir davranışı bozmaz.** Sandbox zaten yalnızca 8443/TCP ile Tool
Gateway'e bağlanıyor (`sandbox-egress`'in tek izni bu); bu yeni kural sadece "başkası
bağlanamaz" ekliyor, "sandbox bağlanabilir"i zaten olduğu gibi bırakıyor.

## 4. Uygulama adımları

```bash
# 1. Önce dry-run ile şemayı doğrula (yazım hatası yakalamak için)
kubectl apply --dry-run=server -f k8s/policies/tool-gateway-ingress.ciliumnetworkpolicy.yaml

# 2. Uygula
kubectl apply -f k8s/policies/tool-gateway-ingress.ciliumnetworkpolicy.yaml

# 3. Canlıya yansıdığını doğrula
kubectl get ciliumnetworkpolicy tool-gateway-ingress -o jsonpath='{.status.conditions}'
```

## 5. Test planı — pozitif ve negatif

**Pozitif (mevcut akış bozulmamalı):** Normal bir PTC çalıştırması (`fetch_url` tool
çağrısı içeren herhangi bir soru) hâlâ uçtan uca çalışmalı — sandbox, Tool Gateway'e
8443'ten bağlanabiliyor olmalı.

**Negatif (yeni kısıtlama gerçekten çalışıyor mu):** Cluster içinde, `app: ptc-sandbox`
etiketi TAŞIMAYAN geçici bir pod'dan Tool Gateway'e bağlanmayı dene — reddedilmeli:

```bash
kubectl run ingress-test --rm -it --restart=Never --image=curlimages/curl -- \
  curl -sv --max-time 5 http://tool-gateway.default.svc.cluster.local:8443/healthz
```

Beklenen: bağlantı zaman aşımına uğramalı (SYN drop — `sandbox-egress`'teki DENY
paternine benzer), ve Hubble'da bu deneme için `Policy denied DROPPED` görülmeli:

```bash
hubble observe --pod tool-gateway --since 1m
```

## 6. Cilium Ingress Controller / Gateway API — neden bu PoC'de yok

Bu bileşen, cluster dışından gelen north-south HTTP(S) trafiğini yönlendirmek için var
(bkz. tablo, bölüm 1). Bizim mimarimizde dışarıdan cluster'a giren hiçbir trafik yok — tek
giriş noktası, geliştiricinin kendi `kubectl port-forward`'u (Hubble UI, Hubble relay)
ve bunlar zaten Cilium'un policy motorunun tamamen dışında, kubectl'in kendi RBAC/API
server erişimiyle korunuyor. Dolayısıyla bu özelliği etkinleştirmek (`cilium install
--set ingressController.enabled=true` + bir `Ingress`/`Gateway` objesi) bu PoC'nin tehdit
modeline hiçbir şey eklemiyor — belgelenmesi, ileride kurumsal asistanın kendisi bir HTTP
API olarak dışarıya açılırsa (ör. laptop yerine bir Deployment olarak) gündeme
gelebileceği için.
