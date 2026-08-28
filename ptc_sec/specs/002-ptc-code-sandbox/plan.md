# Implementation Plan: PTC Kod Sandbox'ı (Faz 2)

**Branch**: `002-ptc-code-sandbox` | **Date**: 2026-08-27 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-ptc-code-sandbox/spec.md`

**Kapsam notu**: Altan'ın açık önceliği: *"asıl amacımız Cilium/eBPF kullanımı — policy'lerin nasıl yazıldığı önemli, sandbox'ın kurulumu tamamen önemli; PoC'de nasıl kolay olacaksa öyle yapalım."* Yani bu plan, pod orkestrasyonu/kod-taşıma gibi "iskelet" kararlarını en basit yoldan geçiyor, ama CiliumNetworkPolicy tasarımına ve doğrulanmasına orantısız önem veriyor — bu fazın gerçek değeri orada.

## Summary

Faz 1'deki asistan laptop'ta çalışmaya devam eder. PTC gerektiren bir görev geldiğinde, ana asistan **Python `kubernetes` client** ile laptop'taki bir `kind` (Kubernetes in Docker) cluster'ında **bir Job/Pod** ayağa kaldırır. Kod, bir **ConfigMap** ile pod'a taşınır; pod bitince ana asistan **pod log'larından** sonucu okur. Sandbox pod'unun tek erişebildiği ağ hedefi, **CiliumNetworkPolicy** ile cluster-içi bir **Tool Gateway** servisidir (FastMCP'nin HTTP transport'u ile, Faz 1'in `knowledge_base`/`live_systems` mantığını saran) — başka hiçbir yere (internet dahil) çıkamaz. Tool Gateway'in kendi egress'i de ayrıca kısıtlanır (yalnızca embedding gateway + mock MCP sunucusu — "supporting service"in kendi egress'i, OpenAI olayının dersi).

## Technical Context

**Language/Version**: Python 3.11+ (ana asistan, sandbox kodu, Tool Gateway servisi — hepsi aynı sürüm ailesi)

**Primary Dependencies**:
- `kubernetes` (resmi K8s Python client — pod/ConfigMap oluşturma, log okuma) — **yeni**
- `fastmcp` (zaten kurulu, Faz 1'den) — Tool Gateway'in HTTP transport'u için
- Faz 1'in mevcut bağımlılıkları (`langgraph`, `langchain`, vb.) değişmiyor

**Altyapı gereksinimleri (pip ile kurulmaz, laptop'a ayrı kurulur)**: Docker, `kind`, `helm`, `kubectl`, Cilium CLI

**Storage**: Değişmedi (Faz 1'in embedding index'i); ayrıca ConfigMap'ler (kod taşıma) ve pod log'ları (sonuç taşıma) geçici depolama olarak kullanılır, kalıcı değil.

**Testing**: pytest (mevcut) + `cilium connectivity test` (Cilium kurulumunun kendisini doğrulamak için) + elle yazılmış "kaçış denemesi" senaryoları (SC-002)

**Target Platform**: Laptop (Linux), Docker üzerinde `kind` ile tek-makinelik Kubernetes cluster'ı

**Project Type**: Hibrit — mevcut tek-proje Python uygulaması (Faz 1) + yeni bir Kubernetes manifest/policy katmanı (`k8s/`)

**Performance Goals**: PoC seviyesi; pod oluşturma/silme gecikmesi saniyeler-onlarca saniye mertebesinde kabul edilebilir

**Constraints**:
- Sandbox pod'unun CiliumNetworkPolicy'si **default-deny + yalnızca Tool Gateway'e explicit-allow** olmalı (Principle II, bu fazın çekirdeği)
- Tool Gateway'in kendi CiliumNetworkPolicy'si de ayrıca kısıtlı olmalı (yalnızca embedding gateway FQDN'i + mock MCP sunucusu) — "supporting service" ilkesi
- Sandbox'a kod ConfigMap ile, sonuç pod log'larından — Altan'ın "kolay olanı seç" kararı
- Tool Gateway = FastMCP HTTP transport — Altan'ın "kolay olanı seç" kararı

**Scale/Scope**: Tek demo kullanıcı, aynı anda tek sandbox pod (Faz 1'in tek-oturum varsayımıyla tutarlı)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| İlke | Durum | Not |
|---|---|---|
| I. Zemine Dayalılık | **PASS** | Sandbox'tan dönen sonuç, Faz 1'deki aynı `Answer`/`Trace` modeline akar; hiçbir tool-proxy çağrısı yapılmadan üretilen iddia kabul edilmez (FR-011, değişmedi). |
| II. Yalnızca Onaylı Kanal | **PASS** | Bu fazın tam odağı: CiliumNetworkPolicy default-deny + yalnızca Tool Gateway'e explicit-allow (sandbox), Tool Gateway'in kendi egress'i ayrıca kısıtlı (supporting-service ilkesi, `PTC_Egress_Policy_OpenAI_Incident.md`'nin dersi). |
| III. İzlenebilirlik | **PASS** | Üç seviyeli: (1) sandbox pod log'u — ne çalıştı, (2) Tool Gateway'in kendi request log'u — hangi tool çağrıldı, (3) Hubble/Cilium flow log'u — network seviyesinde ALLOW/DENY. |
| IV. Teknoloji Kararları Kullanıcıya Aittir | **PASS** | K8s client kütüphanesi, kod/sonuç taşıma deseni, Tool Gateway implementasyonu — üçü de Altan'a soruldu; Altan "en kolay olanı seç" dedi (bu da bir onay biçimi, körü körüne varsayım değil). |
| V. Basitlik | **PASS** | Altan'ın kendi talimatı: "pod orkestrasyonunu basit tut, asıl efor Cilium policy'lerine gitsin." Bu ilkeyi doğrudan teyit ediyor. |

**Post-Phase-1 re-check**: `research.md`/`data-model.md`/`contracts/` yukarıdaki
gate'leri ihlal etmiyor. Özellikle Principle II somutlaştı: sandbox'ın tek izinli
hedefi Tool Gateway (`toEndpoints`, DNS'siz), Tool Gateway'in tek izinli dış
hedefi embedding gateway FQDN'i (`toFQDNs` + DNS) — iki katmanlı, "supporting
service" ilkesine uygun. Complexity Tracking'e girecek bir sapma yok.

## Project Structure

### Documentation (this feature)

```text
specs/002-ptc-code-sandbox/
├── plan.md              # Bu dosya
├── research.md          # Faz 0 çıktısı
├── data-model.md         # Faz 1 çıktısı
├── quickstart.md         # Faz 1 çıktısı
├── contracts/            # Faz 1 çıktısı
└── tasks.md              # Faz 2 çıktısı (/speckit-tasks ile)
```

### Kaynak kod ve altyapı (repository root)

```text
k8s/
├── kind-config.yaml              # kind cluster tanımı (disableDefaultCNI: true, 1 control-plane)
├── tool-gateway/
│   ├── deployment.yaml           # Tool Gateway pod'u
│   └── service.yaml              # Cluster-içi DNS adı (sandbox'ın toEndpoints ile hedefleyeceği)
├── sandbox/
│   └── job-template.yaml         # Sandbox'ın taban Job tanımı (ConfigMap volume mount ile)
└── policies/
    ├── sandbox-egress.ciliumnetworkpolicy.yaml     # Sandbox -> yalnızca Tool Gateway (default-deny)
    └── tool-gateway-egress.ciliumnetworkpolicy.yaml # Tool Gateway -> yalnızca embedding gateway FQDN + mock MCP

sandbox_image/
├── Dockerfile                    # python:3.11-slim tabanlı, minimal (ağ kütüphaneleri: sadece MCP client)
└── entrypoint.py                 # ConfigMap'ten kodu okur, çalıştırır, sonucu stdout'a yazar (pod log = sonuç)

mock_services/tool_gateway/
├── server.py                     # FastMCP HTTP transport; Faz 1'in knowledge_base/live_systems mantığını sarar
└── Dockerfile                    # kind cluster'a yüklenecek image

src/grounded_assistant/ptc/
├── __init__.py
└── sandbox_runner.py              # kubernetes client: ConfigMap yaz, Job oluştur, bitmesini bekle, log oku, temizle
```

**Structure Decision**: Faz 1'in `src/grounded_assistant/` paketi bozulmuyor, sadece yeni bir `ptc/` alt-modülü ekleniyor (sandbox_runner). Kubernetes manifest'leri (`k8s/`) ve container image'ları (`sandbox_image/`, `mock_services/tool_gateway/`) yeni, ayrı üst-düzey dizinler — Python paketinin kendisiyle karışmıyor, çünkü bunlar farklı bir çalışma zamanına (K8s cluster) ait.

## Complexity Tracking

> Gate ihlali yok, bu bölüm boş.

## Post-İmplementasyon Constitution Check (T030)

Faz 2'nin tüm görevleri (T001-T029) tamamlandıktan, gerçek `kind` cluster'ına
karşı uçtan uca test edildikten sonra, Phase-1 tasarım aşamasındaki (yukarıdaki
tablo) 5 ilke iddiasının fiilen doğru çıkıp çıkmadığının kontrolü:

| İlke | Durum | Kanıt (tasarım iddiası değil, gerçek test) |
|---|---|---|
| I. Zemine Dayalılık | **PASS** | T018: gerçek LLM ile SC-001 senaryosu `grounded=True`, `source_refs` dolu döndü. T026: sentetik bir TIMEOUT `SandboxRun` ile `_build_answer` `grounded=False` + jenerik "veri bulunamadı" ürettiği doğrudan test edildi (fabrikasyon yok). |
| II. Yalnızca Onaylı Kanal | **PASS** | T022: `escape_attempt.py` gerçek clusterda `status=DENIED_ACTION` üretti, Hubble ham çıktısında `Policy denied DROPPED` (önce DNS seviyesinde, kube-dns'e giden paket bile düşüyor — sandbox-egress'in DNS'siz tasarımı beklenenden de sıkı çıktı). T023: aynı mekanizma izinli çağrıda `FORWARDED`/`status=SUCCESS` üretti — yanlış pozitif yok. |
| III. İzlenebilirlik | **PASS** (küçük bir netleştirmeyle) | Üç seviye gerçekten var ve ayrı ayrı doğrulandı: (1) sandbox pod log'u — `entrypoint.py`'nin `tool_call`/nihai JSON satırları; (2) Tool Gateway'in HTTP erişim log'u — `kubectl logs -l app=tool-gateway` her isteği (`POST /mcp 200 OK` vb.) gösteriyor, ama tool ADINI değil (bu detay sadece 1. ve Hubble'da değil, `Trace`'te var); (3) Hubble flow log'u — L3/L4 ALLOW/DENY. Üçü birlikte tam resmi veriyor, tek başına hiçbiri yetmiyor — bu da Principle III'ün "bağımsız denetlenebilir" şartını güçlendiriyor. |
| IV. Teknoloji Kararları Kullanıcıya Aittir | **PASS** (bir ek notla) | Planlanan tüm seçimler (kubernetes client, ConfigMap+Job, FastMCP HTTP) aynen kullanıldı. Implementasyon sırasında TEK yeni araç eklendi: `hubble` CLI (~21MB) — bu, zaten onaylı olan Cilium/Hubble'ın resmi gözlem istemcisi, yeni bir mimari/framework kararı değil (tıpkı daha önce kurulan `cilium` CLI'nin bir devamı); yine de şeffaflık için burada not düşülüyor. |
| V. Basitlik | **PASS** | Eklenen HER ŞEY (denied_actions alanı, record_sandbox_run/record_denied_action, `_ARG_NAMES` eşlemesi) spec'in kendi FR/SC'lerini (FR-007, FR-008, FR-011, SC-002, SC-003) karşılamak için zorunluydu — hiçbiri spekülatif değildi. Bilerek YAPILMAYAN: Tool Gateway'de auth/mTLS (contracts/tool_gateway_mcp.md'de açıkça kapsam dışı bırakıldı), sandbox'ta Python-seviyesi import kısıtlaması (research.md §4.3 — enforcement kasıtlı olarak sadece Cilium'da). |

**İmplementasyon sırasında bulunup düzeltilen, tasarım aşamasında öngörülemeyen 5 gerçek hata** (hepsi tasks.md'de ilgili görevin notunda da kayıtlı):
1. `kubernetes` client'ın `read_namespaced_pod_log`'u JSON log'u sessizce `str(dict)`'e çeviriyor, kontratı bozuyordu → `_preload_content=False` ile düzeltildi (T013).
2. `sandbox_runner.py`, Tool Gateway'i DNS adıyla enjekte ediyordu — bu, `sandbox-egress`'in DNS'siz tasarımını (research.md §4.1) fiilen kıracaktı → ClusterIP doğrudan enjekte edilecek şekilde düzeltildi (T016).
3. Tool proxy'ler yalnızca keyword-argüman kabul ediyordu, LLM'in doğal pozisyonel çağrısı patlıyordu → `_ARG_NAMES` eşlemesi eklendi (T015).
4. `SandboxRun`'da `denied_actions` alanı hiç yoktu (data-model.md'nin kendi durum-akışı diyagramıyla tutarsız bir eksiklik) → eklendi (T020).
5. Job'un zaman aşımı koşulunun `type`'ı beklenen `"Failed"` değil `"FailureTarget"` çıktı (bu Kubernetes sürümünde) — `reason`'a bakacak şekilde düzeltildi, aksi halde her timeout sessizce `error`'a düşüyordu (T025).

Bu 5 bulgu, "T030'un amacı planın kağıt üzerinde tutarlı olduğunu değil, planın GERÇEKTEN çalıştığını doğrulamak" ilkesini destekliyor — hepsi ancak gerçek cluster'a karşı çalıştırılarak ortaya çıktı, statik inceleme ile yakalanamazdı.

**6. bulgu — bilinçli olarak düzeltilmeyen, canlı gözlemlenen bir sınırlama (2026-08-28)**:
`sample_docs/`'a gerçek içerik (KKB 2024 Faaliyet Raporu, Altan'ın eklediği PDF'ten çıkarılmış) eklenip Tool Gateway image'ı yeniden build edildikten sonra, `search_knowledge_base` gerçek bir embedding çağrısı denedi. Hubble'da görülen: Tool Gateway, `mia.csp.kloudeks.com`'a DEĞİL, `openaipublic.blob.core.windows.net`'e DNS sorgusu attı (muhtemelen `langchain_openai.OpenAIEmbeddings`'in arkasındaki `tiktoken`/`openai` kütüphanesinin, tanımadığı bir model adı — "Qwen3-Embedding-8B" — için varsayılan bir BPE encoding dosyası indirmeye çalışması; bu, kütüphanenin kaynak koduna bakılarak DOĞRULANMADI, DNS örüntüsünden çıkarılan bir varsayım). `tool-gateway-egress` policy'si bunu **tam olarak tasarlandığı gibi** ele aldı: DNS sorgusuna izin verdi (kube-dns kuralı her sorguya izin veriyor), ama çözümlenen IP'ye (`57.150.97.129:443`) bağlantıyı reddetti (`Policy denied DROPPED`, `toFQDNs` listesinde yok) — sandbox çalıştırması bu yüzden 30sn'de `TIMEOUT`'a düştü, gerçek embedding çağrısı hiç sırasına gelemedi.

Bu, `PTC_Egress_Policy_OpenAI_Incident.md`'nin dersinin (agent'ın kullandığı bir kütüphane, hesaba katılmamış bir "destekleyici" hedefe çıkmaya çalışabilir) PoC içinde kendiliğinden, gerçek bir örneği. Altan'ın kararı: politika GENİŞLETİLMEYECEK — mevcut kanıt (DNS'e izin, onaysız FQDN'e ret) Principle II'nin tezini zaten kanıtlıyor; `mia.csp.kloudeks.com`'a giden gerçek bir `FORWARDED` trafiği bu haliyle hâlâ canlı gözlemlenmedi (bilinen, kabul edilmiş bir doğrulama boşluğu — bkz. `README.md`/quickstart.md'ye ayrıca not düşülebilir).
