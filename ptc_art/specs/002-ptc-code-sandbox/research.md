# Research: PTC Kod Sandbox'ı (Faz 2)

Altan'ın önceliği: *"asıl amaç Cilium/eBPF — policy'lerin nasıl yazıldığı önemli."*
Bu doküman, iskelet kararlarını (K8s client, kod/sonuç taşıma, Tool Gateway) hızlıca
geçip **CiliumNetworkPolicy tasarımına** orantısız detay veriyor.

## 1. Pod orkestrasyonu: Python `kubernetes` client

- **Decision**: Resmi `kubernetes` PyPI paketi (`from kubernetes import client, config`).
- **Rationale**: Altan "en kolay olanı seç" dedi. `kubectl` subprocess'e göre hata
  yönetimi daha temiz (Python exception'ları, JSON parse etmeye gerek yok).
- **Kaynak**: Altan'ın kararı (2026-08-27, "kolay olan").

## 2. Kod/sonuç taşıma: ConfigMap + pod log'ları

- **Decision**: LLM'in ürettiği kod bir `ConfigMap`'e yazılır, sandbox Job'una
  volume olarak mount edilir. Job bitince ana asistan pod'un log'larını
  (`kubernetes.client.CoreV1Api().read_namespaced_pod_log(...)`) okuyup nihai
  sonucu (`entrypoint.py`'nin stdout'a yazdığı) alır.
- **Rationale**: Sandbox'ın "sonucu bildirmek" için ekstra bir ağ çağrısı yapmasına
  gerek kalmıyor — bu da sandbox'ın egress ihtiyacını **tek bir hedefe** (Tool
  Gateway) indiriyor, Cilium policy'sini daha temiz/gösterilebilir kılıyor.
- **Alternatives considered**: Sandbox'ın Tool Gateway'e "sonucu bildir" çağrısı
  yapması (reddedildi — gereksiz karmaşıklık, ekstra bir endpoint gerektirir).
- **Kaynak**: Altan'ın kararı (2026-08-27, "kolay olan").

## 3. Tool Gateway: FastMCP HTTP transport, mock live-system in-process

- **Decision**: Tool Gateway, `fastmcp`'in HTTP transport'uyla çalışan tek bir pod.
  Faz 1'in `knowledge_base.py` (4 kaynak + Hybrid Search + RRF) ve mock
  live-system mantığını (T019'daki sahte ticket verisi) **kendi içinde,
  in-process** çalıştırır — ayrı bir "mock MCP sunucusu" pod'una ihtiyaç yok.
- **Rationale**: Zaten kurulu (`fastmcp`), yeni framework yok. Mock live-system'i
  ayrı bir pod yapmak, network topolojisini gereksiz karmaşıklaştırırdı
  (sandbox → Tool Gateway → mock-MCP-pod gibi 2 hop yerine, sandbox → Tool
  Gateway tek hop) — Altan'ın "kolay tut" talimatına daha uygun.
- **Alternatives considered**: Düz REST (FastAPI/Flask) — reddedildi, MCP'nin
  zaten sunduğu tool-şeması/keşif mekanizmasını yeniden yazmak gerekirdi.
- **Kaynak**: Altan'ın kararı (2026-08-27, "kolay olan").

## 4. CiliumNetworkPolicy tasarımı — bu fazın çekirdeği

### 4.1 Sandbox'ın egress'i: tek kural, DNS'siz

- **Decision**: Sandbox pod'u, Tool Gateway'in Service ClusterIP'sini bir ortam
  değişkeni olarak (Job oluşturulurken ana asistan tarafından enjekte edilir)
  alır — **DNS çözümlemesi hiç gerekmez**. Policy tek bir `toEndpoints` kuralı:

  ```yaml
  apiVersion: cilium.io/v2
  kind: CiliumNetworkPolicy
  metadata:
    name: sandbox-egress
  spec:
    endpointSelector:
      matchLabels:
        app: ptc-sandbox
    egress:
    - toEndpoints:
      - matchLabels:
          app: tool-gateway
      toPorts:
      - ports:
        - port: "8443"
          protocol: TCP
  ```

  Buradaki `endpointSelector`/`toEndpoints`, Cilium'un **identity-based**
  modelidir — Tool Gateway'in pod'u yeniden başlasa, IP'si değişse bile kural
  geçerli kalır (`PTC_Egress_Policy_Cilium_eBPF_Technical_Reference.md` §4.1).
  Başka HİÇBİR egress kuralı yok — default-deny zaten geri kalan her şeyi
  (internet dahil) kapatıyor. Bu, Altan'ın istediği "en temiz, en gösterilebilir"
  policy — tek kural, tek amaç.
- **Rationale**: DNS'i (dolayısıyla bir egress kuralı daha) devre dışı bırakmak,
  policy'yi "sandbox SADECE Tool Gateway'e gidebilir" diye tek cümlede
  özetlenebilir kılıyor — öğretici/demo değeri yüksek.
- **Alternatives considered**: FQDN ile (`toFQDNs: matchName: tool-gateway...`)
  — reddedildi, Tool Gateway zaten cluster-içi, DNS'e ihtiyaç yaratmadan
  `toEndpoints` yeterli ve daha basit.

### 4.2 Tool Gateway'in kendi egress'i — "supporting service" ilkesi

- **Decision**: Tool Gateway'in TEK dış hedefi, `.env`'deki embedding/LLM
  gateway'i (`mia.csp.kloudeks.com`) — burada **gerçekten** `toFQDNs` gerekir
  (dış, gerçek internet hedefi):

  ```yaml
  apiVersion: cilium.io/v2
  kind: CiliumNetworkPolicy
  metadata:
    name: tool-gateway-egress
  spec:
    endpointSelector:
      matchLabels:
        app: tool-gateway
    egress:
    - toEndpoints:  # DNS çözümlemesi için
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
    - toFQDNs:
      - matchName: "mia.csp.kloudeks.com"
      toPorts:
      - ports:
        - port: "443"
          protocol: TCP
  ```

  Bu, `PTC_Egress_Policy_OpenAI_Incident.md`'nin ana dersinin doğrudan
  uygulanması: **agent'ın (sandbox'ın) egress'i kısıtlı olması yetmez, agent'ın
  eriştiği supporting service'in (Tool Gateway) egress'i de ayrıca kısıtlanmalı**
  — yoksa Tool Gateway, tıpkı Artifactory gibi, bir SSRF köprüsüne dönüşebilir.
- **Rationale**: İki farklı Cilium kural türünü (identity-based `toEndpoints`,
  FQDN-based `toFQDNs`) aynı PoC içinde, gerçek bir ihtiyaçtan doğan şekilde
  göstermek — Altan'ın "policy'ler nasıl yazılıyor" öğrenme hedefine tam uyuyor.
- **Kaynak**: `PTC_Egress_Policy_Cilium_eBPF_Technical_Reference.md` §5.2,
  `PTC_Egress_Policy_OpenAI_Incident.md` §5-7.

### 4.3 Sandbox image: minimal, ağ kütüphanesi bilerek kısıtlanmamış

- **Decision**: Sandbox'ın Python image'ı (`python:3.11-slim`), MCP client
  kütüphanesini (Tool Gateway'e bağlanmak için) İÇEREN normal bir image. Kod
  seviyesinde `import socket`'i engellemeye ÇALIŞMIYORUZ — çünkü zaten
  denemesi anlamsız: Cilium, hangi kütüphaneyi kullandığından bağımsız olarak
  paketi kernel'de düşürüyor (bkz. daha önceki tartışma — "hangi kütüphaneyi
  kullanırsan kullan, dışarı çıkamıyorsun").
- **Rationale**: Bu, "uygulama-seviyesi kısıtlama" (Faz 1 tarzı, `import`
  allowlist) ile "network-seviyesi kısıtlama" (Faz 2, Cilium) arasındaki farkı
  netleştiriyor — ikinci katman, birinciyi atlatan bir kodu bile durdurabiliyor.
  Bunu SC-002'nin "kaçış denemesi" test senaryosunda bilerek sergileyeceğiz:
  sandbox'a `requests.get("https://google.com")` yazan bir kod verip, bunun
  network seviyesinde (Cilium tarafından) engellendiğini göstereceğiz — Python
  tarafında hiçbir "yasak" yok, hepsi kernel'de oluyor.

## 5. Kubernetes kaynak türü: Job (Pod değil)

- **Decision**: Sandbox, çıplak bir `Pod` değil, bir `Job` olarak oluşturulur
  (`restartPolicy: Never`, `backoffLimit: 0`, `ttlSecondsAfterFinished: 300`).
- **Rationale**: Job, "bir kere çalış, bitince tamamlanmış say" semantiğini ve
  otomatik temizliği (`ttlSecondsAfterFinished`) native olarak sağlıyor — elle
  pod silme kodu yazmaya gerek kalmıyor (Principle V).
- **Kaynak**: Kubernetes'in kendi Job kaynağı, standart pratik.

## 6. Timeout ve kaynak sınırları

- **Decision**: Job seviyesinde `activeDeadlineSeconds: 30` (FR-006'daki üst
  sınır); pod'a `resources.limits: {cpu: "500m", memory: "256Mi"}`.
- **Rationale**: Spec'in Assumptions'ı bu sayıları planlama aşamasına
  bırakmıştı — burada somutlaştırıldı. 30 saniye, bir PoC orkestrasyon görevi
  için makul bir üst sınır; CPU/bellek sınırları "sonsuz döngü CPU'yu
  tüketsin" senaryosuna karşı ek bir savunma katmanı.
- **Kaynak**: Bu dokümanda önerilen makul varsayım (düşük riskli, kolayca
  değiştirilebilir).
