# Contract: CiliumNetworkPolicy'ler

Bu fazın **asıl** kontratı — sandbox'ın ve Tool Gateway'in ağ üzerinde tam olarak
neye erişebileceğini tanımlar. Tam YAML'lar `research.md` §4'te; burası bunları
tek bir referans olarak özetler, implementasyon bunlara birebir uymalı.

## `sandbox-egress` — sandbox pod'unun izin listesi

| Hedef | Protokol/Port | Karar |
|---|---|---|
| Tool Gateway pod'u (`app: tool-gateway`) | TCP/8443 | ALLOW |
| *(başka her şey — internet dahil)* | * | **DENY** (default-deny, açık kural yok) |

DNS gerekmiyor — Tool Gateway'in adresi Job'a ortam değişkeni olarak
enjekte ediliyor (bkz. `sandbox_job_contract.md`).

## `tool-gateway-egress` — Tool Gateway pod'unun izin listesi

| Hedef | Protokol/Port | Karar |
|---|---|---|
| `kube-dns` (cluster içi) | UDP/53 | ALLOW (sadece FQDN çözümlemesi için) |
| `mia.csp.kloudeks.com` (FQDN) | TCP/443 | ALLOW |
| *(başka her şey)* | * | **DENY** |

## Doğrulama komutları (quickstart.md'de tekrar kullanılır)

```bash
cilium policy list                                  # yüklü policy'ler
hubble observe --pod ptc-sandbox --verdict DENIED    # sandbox'ın engellenen denemeleri
hubble observe --pod ptc-sandbox --verdict FORWARDED # sandbox'ın izinli akışları
```
