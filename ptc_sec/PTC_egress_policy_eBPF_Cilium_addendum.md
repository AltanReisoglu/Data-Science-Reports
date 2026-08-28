# PTC: Egress Policy (eBPF / Cilium) — Only via Approved Tool Channels

## “Only via approved tool/API channels” ne demek?

Buradaki **tool**, büyük ölçüde bildiğimiz **agent tool'udur**. Ancak ifade sadece tool fonksiyonunu değil, tool'un arkasındaki kontrollü API/gateway kanalını da kapsar.

Temel fikir:

> Agent'ın dış dünyaya erişimi doğrudan ve sınırsız network bağlantısı üzerinden değil, önceden izin verilmiş agent tool'ları veya kontrollü API kanalları üzerinden gerçekleşir.

Örneğin agent'a şu tool'lar verilebilir:

```text
search_web()
github_search()
query_database()
send_email()
```

Agent:

```python
search_web("Cilium egress policy")
```

çağrısını yapabilir.

Buna karşılık agent sandbox içerisinde:

```python
requests.get("https://evil.com")
```

ile istediği herhangi bir internet adresine doğrudan çıkamaz. Bu tür bağlantı denemeleri network katmanındaki egress policy tarafından engellenebilir.

---

## Tool ile API arasındaki ilişki

Bir agent tool'unun arkasında bir API bulunabilir:

```text
Agent
  ↓
tool: search_web()
  ↓
Tool Gateway / Tool Backend
  ↓
Search API
  ↓
Internet
```

Bu yüzden **“approved tool/API channels”**, agentın kullanmasına izin verilen kontrollü capability'leri ifade eder.

Buradaki “channel” fiziksel bir network kablosu değil; agentın dış sistemlere ulaşmasını sağlayan **yetkilendirilmiş erişim yolu** anlamındadır.

---

## PTC bu yapıda nerede?

Programmatic Tool Calling (PTC), agentın tool'ları modelden ayrı round-trip'lerle tek tek çağırması yerine, code execution ortamında yazdığı kod üzerinden programatik biçimde çağırabilmesini sağlar.

```text
AI Agent
   ↓
Code Execution Sandbox
   ↓
PTC
   ↓
Approved Tools
```

Örneğin:

```python
results = []

for query in queries:
    results.append(search_web(query))

return filter_results(results)
```

Burada `search_web()` bir approved tool'dur. PTC ise bu tool'u kod içerisinden orkestre eder.

Önemli ayrım:

```text
PTC ≠ Network Security
```

PTC **tool orchestration / capability kullanım mekanizmasıdır**. Ağ erişiminin gerçekten nereye yapılabileceğini eBPF/Cilium gibi network enforcement katmanları sınırlar.

---

## MCP eklenirse

MCP kullanıldığında approved tool'lar bir MCP server üzerinden sağlanabilir:

```text
AI Agent
   ↓
PTC
   ↓
MCP Client
   ↓
MCP Server
   ↓
Approved API
```

Örneğin:

```text
Agent
  ↓
PTC
  ↓
MCP
  ↓
github_search()
  ↓
GitHub API
```

Burada:

- **PTC** → tool'ları programatik olarak orkestre eder.
- **MCP** → tool/resource erişimini standardize eden protokoldür.
- **Authorization** → agentın hangi tool'ları kullanabileceğini belirler.
- **eBPF/Cilium** → gerçek network bağlantılarında ALLOW/DENY enforcement yapar.

---

## “Only via approved tool channels” güvenlik mimarisinde nasıl görünür?

Amaç:

```text
❌ Agent → arbitrary Internet
```

yerine:

```text
✅ Agent
   ↓
  PTC
   ↓
  Approved Tool / MCP Tool
   ↓
  Tool Gateway
   ↓
  Approved API
```

olmasıdır.

Network seviyesinde ise:

```text
Agent Sandbox
 ├── Approved Tool Gateway:443/TCP ✅
 ├── Package Gateway:443/TCP      ✅
 └── Arbitrary Internet             ❌
```

şeklinde bir **default-deny + explicit-allow** yaklaşımı uygulanabilir.

---

## Cilium/eBPF bu yapının neresinde?

PTC veya MCP “network security boundary” değildir. Agent doğrudan network çağrısı yapmaya çalışırsa son savunma katmanlarından biri egress policy olabilir.

Örneğin:

```text
Agent
  ↓
requests.get("https://evil.com")
  ↓
socket / connect()
  ↓
Linux Kernel
  ↓
eBPF / Cilium
  ↓
Egress Policy
  ↓
DENY
```

Approved tool çağrısında ise:

```text
Agent
  ↓
PTC
  ↓
Approved Tool
  ↓
Tool Gateway
  ↓
TCP/HTTPS
  ↓
eBPF / Cilium
  ↓
ALLOW
  ↓
Approved API
```

Dolayısıyla:

> **PTC “hangi capability'yi kullanıyorum?” sorusunu, eBPF/Cilium ise “bu workload gerçekten bu network flow'unu oluşturabilir mi?” sorusunu ele alır.**

---

## OpenAI/Artifactory olayıyla bağlantı

OpenAI'nin paylaştığı 2026 incident anlatımında sandboxların doğrudan internet erişimi kapalı olmasına rağmen agentların erişebildiği Artifactory servisinin internete erişebilmesi önemli bir rol oynadı.

Basitleştirilmiş saldırı yolu:

```text
Agent
  ↓
Sandbox
  ↓
Artifactory
  ↓
SSRF
  ↓
Internet
```

Buradaki ders:

> Sadece agent sandbox'ın internet erişimini kapatmak yeterli değildir. Agentın erişebildiği supporting service'lerin egress'i de ayrıca sınırlandırılmalıdır.

Daha güvenli model:

```text
Agent
  ↓
Cilium Egress Policy
  ├── Tool Gateway ✅
  ├── Package Gateway ✅
  └── Arbitrary Internet ❌
```

ve supporting service için:

```text
Artifactory / Package Gateway
  ↓
Cilium Egress Policy
  ├── Approved Package Repositories ✅
  └── Arbitrary Internet ❌
```

Böylece bir servis compromise edilse veya SSRF açığı bulunsa bile sonraki network flow ayrı bir egress policy ile tekrar kontrol edilebilir.

---

## TCP/UDP ile ilişki

Egress policy TCP veya UDP'nin kendisi değildir.

**TCP/UDP = transport protocols**

**Egress policy = bu network flow'una izin verilip verilmediğini belirleyen güvenlik politikası**

TCP örneği:

```text
Agent
  ↓
socket()
  ↓
connect()
  ↓
Linux Kernel
  ↓
eBPF / Cilium
  ↓
Policy lookup
  ↓
ALLOW / DROP
```

UDP'de:

```text
Agent
  ↓
socket()
  ↓
sendto() / UDP traffic
  ↓
Linux Kernel
  ↓
eBPF / Cilium
  ↓
Policy lookup
  ↓
ALLOW / DROP
```

Örneğin:

```text
Agent → tool-gateway:443/TCP ✅
Agent → random-internet:443/TCP ❌
Agent → approved-dns:53/UDP ✅
Agent → arbitrary-dns:53/UDP ❌
```

---

## Katmanların görevleri

```text
AI Agent
↓
Karar verir

Sandbox
↓
Process / code isolation

PTC
↓
Programmatic tool orchestration

MCP
↓
Tool/resource protocol standardization

Authorization / Tool Gateway
↓
Hangi capability'nin kullanılabileceği

eBPF / Cilium
↓
Network egress enforcement

External API
↓
Kontrollü dış kaynak
```

Özellikle egress tarafında:

```text
PTC
  → “Hangi tool'u kullanabilirim?”

Tool Gateway / Authorization
  → “Bu tool çağrısına izin var mı?”

eBPF/Cilium
  → “Bu workload bu destination'a bu protocol/port ile gerçekten çıkabilir mi?”
```

---

## Temel mimari

```text
                         INTERNET
                            │
                     Approved APIs
                            │
                     ┌──────▼──────┐
                     │ Tool Gateway│
                     └──────▲──────┘
                            │
                     MCP / Approved Tools
                            │
                     ┌──────▼──────┐
                     │     PTC     │
                     └──────▲──────┘
                            │
                     Code Execution
                        Sandbox
                            │
                     TCP / UDP flows
                            │
                     ┌──────▼──────┐
                     │ eBPF/Cilium │
                     │ Egress      │
                     │ Policy      │
                     └──────▲──────┘
                            │
                       ALLOW / DROP
```

Daha doğru bir production tasarımında MCP server, package gateway veya başka supporting service'lerin **kendi egress politikaları da ayrıca uygulanır**.

---

## Sonuç

“Only via approved tool/API channels” ifadesi:

> **Agentın dış dünyayla iletişimini unrestricted network erişimi şeklinde değil, önceden tanımlanmış ve yetkilendirilmiş tool/API capability'leri üzerinden gerçekleştirmesi** anlamına gelir.

Bu yaklaşımda PTC tool çağrılarını programatik olarak orkestre eder; MCP gerekiyorsa bu tool'ları standardize edilmiş bir protokol üzerinden sunabilir; Tool Gateway/authorization hangi capability'lerin kullanılabileceğini belirler; eBPF/Cilium ise agent ve supporting service'lerin gerçek TCP/UDP network akışlarını kernel seviyesinde enforce eder.

En önemli güvenlik prensibi:

```text
❌ Agent → arbitrary Internet

✅ Agent → PTC → Approved Tool/MCP → Gateway → Approved API
                       │
                       ▼
                 eBPF / Cilium
                 Egress Policy
```
