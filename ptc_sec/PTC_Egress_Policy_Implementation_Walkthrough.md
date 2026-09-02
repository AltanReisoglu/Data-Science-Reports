# PTC Egress Policy — Uygulamanın Gerçek Çalışma Mantığı (Baştan Sona)

Bu belge, depodaki diğer `PTC_Egress_Policy_*.md` dosyalarından (`PTC_egress_policy_eBPF_Cilium.md`,
`PTC_egress_policy_eBPF_Cilium_addendum.md`, `PTC_Egress_Policy_Cilium_eBPF_Technical_Reference.md`,
`PTC_Egress_Policy_OpenAI_Incident.md`) **farklı bir amaca hizmet eder**: onlar implementasyondan
ÖNCE yazılmış kavramsal/araştırma belgeleridir ("neden bu mimari", "eBPF nedir"). Bu belge ise
implementasyon TAMAMLANDIKTAN ve gerçek cluster üzerinde canlı test edildikten sonra yazıldı —
her iddia, bu depoda çalıştırılmış gerçek komutlar ve gözlemlenmiş gerçek Hubble/Cilium çıktılarıyla
doğrulanmıştır. Varsayım yok, teori yok — sadece "şu an gerçekten böyle çalışıyor."

---

## 1. Genel resim: iki katmanlı savunma

Sistemde İKİ ayrı `CiliumNetworkPolicy` var, iki farklı pod'a uygulanıyor:

```
┌─────────────────┐        ┌──────────────────┐        ┌───────────────────────┐
│   Sandbox Pod    │──①───▶│   Tool Gateway    │──②───▶│  3 onaylı dış hedef    │
│ (app: ptc-sandbox)│       │ (app: tool-gateway)│       │  (mia.csp.kloudeks.com,│
│                  │        │                   │        │  openaipublic...,     │
│  LLM'in ürettiği │        │  Gerçek tool       │        │  html.duckduckgo.com) │
│  kod burada      │        │  implementasyonları│        │                       │
│  çalışır         │        │  (web_search,      │        │                       │
│                  │        │  search_knowledge_ │        │                       │
│                  │        │  base, vb.) burada │        │                       │
└─────────────────┘        └──────────────────┘        └───────────────────────┘
   sandbox-egress                tool-gateway-egress
   (Katman 1)                    (Katman 2)
```

**Katman 1** (`sandbox-egress`), sandbox pod'unun TEK gidebileceği yerin Tool Gateway olduğunu
zorluyor. **Katman 2** (`tool-gateway-egress`), Tool Gateway'in KENDİSİNİN de yalnızca 3 belirli
dış hedefe çıkabildiğini zorluyor. İkisi de bağımsız, ikisi de aynı anda uygulanıyor — biri
atlatılsa bile diğeri hâlâ orada (defense in depth, `PTC_Egress_Policy_OpenAI_Incident.md`'nin
"supporting service" ilkesi).

Önemli: LLM'in sandbox içinde çalıştırdığı kod, **istediği kütüphaneyi import edebilir**
(`socket`, `requests`, vb.) — Python seviyesinde HİÇBİR kısıtlama yok, bilinçli bir tasarım
kararı (`sandbox_image/entrypoint.py`'nin kendi docstring'i: "enforcement Cilium'da, burada
değil"). Güvenlik tamamen ağ/kernel seviyesinde.

---

## 2. Katman 1 — Sandbox Pod'un Egress'i

Dosya: `k8s/policies/sandbox-egress.ciliumnetworkpolicy.yaml`

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

### Kritik nokta: DNS kuralı YOK

Bu politikada `port: 53` / DNS için hiçbir kural yok. Cilium'un varsayılan davranışı
"default-deny" olduğundan (bir `CiliumNetworkPolicy` bir pod'u seçtiği anda, o politikada
AÇIKÇA izin verilmeyen HER ŞEY kapanır), sandbox pod'unun **DNS sorgusu göndermesi bile
mümkün değil**.

Bu bilinçli bir tasarım: `sandbox_runner.py`, Tool Gateway'in Service ClusterIP'sini
(`_resolve_tool_gateway_endpoint`) DNS ADI yerine DOĞRUDAN IP olarak sandbox koduna enjekte
ediyor — yani sandbox'ın hiçbir zaman bir domain adı çözmeye ihtiyacı yok. Tek hedefi zaten
sabit ve IP olarak elinde.

### `toEndpoints` — identity-based, IP-based değil

`toEndpoints.matchLabels: {app: tool-gateway}` bir IP adresi DEĞİL, bir Kubernetes pod
ETİKETİ eşleştirmesi. Cilium bunu kendi kimlik modeliyle (CiliumIdentity) çözüyor — Tool
Gateway pod'u yeniden başlasa, IP'si değişse bile bu kural hâlâ doğru pod'u buluyor.

### Sonuç: bir kaçış denemesinde ne olur?

Sandbox koduna `socket.create_connection(("github.com", 443))` yazılırsa:

1. Python'un DNS çözücüsü `github.com` için bir UDP:53 sorgusu gönderir (CoreDNS'e).
2. Bu paket, `sandbox-egress` politikasında hiç izinli olmadığı için **kernel seviyesinde
   (eBPF datapath'te) DROP edilir** — CoreDNS'e ULAŞMADAN.
3. Cevap gelmediği için işletim sisteminin DNS çözücüsü **tekrar tekrar dener** — clusterda
   2 CoreDNS replikası olduğu için (doğrulandı: `coredns-...-b252z` ve `coredns-...-6qslh`),
   her ikisine de ayrı ayrı, birden fazla kez denenir. Bu yüzden **tek bir mantıksal
   bağlantı denemesi, Hubble'da 16 ayrı `DROPPED` kaydı olarak görünür** (8 deneme × 2
   nameserver) — bu bizim tasarladığımız bir şey değil, DNS resolver'ın "sessiz drop"a
   verdiği doğal retry tepkisi.
4. Sonunda Python `socket.gaierror` (isim çözülemedi) fırlatır — IP'ye bile ulaşılamadan.

**Yani sandbox pod'u, onaylı olmayan bir domain'in İSMİNİ BİLE çözemez** — bu, Tool
Gateway'in DNS'i serbest bırakan (ama TCP'yi kısıtlayan) modelinden daha sıkı bir kısıtlama.

---

## 3. Katman 2 — Tool Gateway'in Kendi Egress'i

Dosya: `k8s/policies/tool-gateway-egress.ciliumnetworkpolicy.yaml`

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
        - matchName: "openaipublic.blob.core.windows.net"
        - matchName: "html.duckduckgo.com"
      toPorts:
        - ports:
            - port: "443"
              protocol: TCP
```

### Neden bu katman DNS'i serbest bırakıyor (`matchPattern: "*"`) ama Katman 1 hiç bırakmıyor?

Sandbox'ın tek hedefi zaten sabit (Tool Gateway) — DNS'e hiç ihtiyacı yok. Ama Tool
Gateway'in **gerçekten 3 farklı, isimle bilinen dış servise** çıkması gerekiyor:
- `mia.csp.kloudeks.com` — LLM/embedding gateway (tiktoken encoding tablosu için,
  defense-in-depth; Dockerfile artık dosyayı build zamanında gömdüğü için normalde
  kullanılmıyor).
- `openaipublic.blob.core.windows.net` — aynı sebep.
- `html.duckduckgo.com` — `web_search` tool'unun TEK dış hedefi.

Bu üçü domain ADIYLA biliniyor, IP'leri CDN/yük dengeleme yüzünden değişebiliyor (canlı
doğrulandı: `html.duckduckgo.com` bir seferinde `40.114.177.156`'ya, `mia.csp.kloudeks.com`
ailesi `185.199.89.67`'ye çözüldü). Bu yüzden Cilium'un **FQDN-farkında (`toFQDNs`)**
mekanizması gerekiyor: DNS sorgusunu serbest bırak (`matchPattern: "*"`), ama Cilium DNS
CEVABINI izliyor ve "bu IP, şu anda `html.duckduckgo.com`'un bir örneği" diye workload'a
bağlıyor (dinamik, TTL'li bir eşleme) — sadece o (o an geçerli) IP'lere TCP:443 izni
veriyor.

### Sonuç: Tool Gateway onaylı olmayan bir yere gitmeye çalışırsa?

Bunu bu konuşmada CANLI test ettik (kod DEĞİL, uygulama testi):

```bash
kubectl exec deploy/tool-gateway -- python3 -c "
import socket
socket.create_connection(('example.com', 443), timeout=5)
"
```

- DNS çözümü **BAŞARILI** oldu (`172.66.147.243` döndü — DNS serbest çünkü `matchPattern: "*"`).
- TCP bağlantısı **`OSError: Network is unreachable`** ile başarısız oldu.
- Hubble'da kanıt: `DROPPED | 172.66.147.243 | {'TCP': {...SYN...}} | POLICY_DENIED` —
  aynı IP'ye giden SYN paketi tekrar tekrar (retransmit) denenip her seferinde DROP edildi.

Yani: **DNS'te serbestlik, TCP'de bağlantı özgürlüğü anlamına gelmiyor.** Bir domain'in
adını öğrenebilmek başka, ona bağlanabilmek başka — Cilium bu ikisini ayrı ayrı kontrol
ediyor.

---

## 4. Uçtan uca akış: bir PTC çalıştırması sırasında ne olur

1. LLM, `run_ptc_code(code)` tool'unu çağırır → `sandbox_runner.run_sandbox()` bir
   ConfigMap (kod) + bir Kubernetes Job (sandbox pod) oluşturur.
2. Sandbox pod'u `entrypoint.py`'yi çalıştırır: kod, `search_knowledge_base`,
   `get_ticket_status`, `web_search`, `calculator` gibi fonksiyonları normal Python
   fonksiyonuymuş gibi çağırabilir — bunlar aslında `fastmcp.Client` ile Tool Gateway'e
   (Katman 1'in izin verdiği TEK hedef, IP olarak enjekte edilmiş) HTTP çağrısı yapan
   proxy'ler.
3. Tool Gateway, gelen çağrıyı gerçek implementasyona yönlendirir. Eğer bu implementasyon
   dışarı çıkması gerekiyorsa (ör. `web_search` → DuckDuckGo), bunu KENDİ egress'i
   (Katman 2) izin verdiği ölçüde yapar.
4. Sonuç, aynı yoldan (Tool Gateway → sandbox → LLM) geri döner.
5. `sandbox_runner.py`, çalıştırma bitince Hubble'dan HEM sandbox pod'unun HEM Tool
   Gateway'in (bu run'ın zaman penceresinde) `DROPPED` akışlarını sorgular
   (`get_denied_actions`) — herhangi biri varsa, bunlar `DeniedAction` olarak kaydedilir
   ve PTC panelinde (`⛔ ENGELLENDİ [sandbox|tool-gateway]: ... — hubble: <ham satır>`)
   canlı gösterilir.

Eğer LLM'in kodu bu akışın DIŞINA çıkmaya çalışırsa (doğrudan bir domain'e bağlanmaya
çalışırsa), Katman 1 bunu DNS seviyesinde anında durdurur — kod hiçbir zaman gerçek bir
IP'ye bile ulaşamaz.

---

## 5. "Ama github.com'u tanımlamadık, yine de içeriği geldi" — bu bir ihlal mi?

Hayır. Bu konuşmada test edildi: `web_search("site:github.com/...")` çağrıldığında,
sonuçlarda github.com URL'leri ve metinleri görünüyor, ama Hubble kanıtı şunu gösteriyor:
Tool Gateway'in bu sırada gittiği TEK dış IP, DuckDuckGo'nunki. **github.com'a hiçbir
paket gitmedi.**

Sebep: DuckDuckGo (onaylı TEK arama motoru) github.com'u zaten kendi başına, açık
internette crawl'lamış ve kendi index'inde tutuyor. Biz DuckDuckGo'ya soru sorduğumuzda,
o bize KENDİ sunucusundan (onaylı bağlantı üzerinden) bir cevap gönderiyor — o cevabın
İÇİNDE "github.com" geçen metin olması, bizim sistemimizin github.com'a bağlandığı
anlamına gelmiyor. Cilium "kim NEREYE bağlanıyor" sorusunu kontrol ediyor, "cevap
metninde hangi kelimeler geçiyor" sorusunu değil (bu, teknik olarak da kontrol edilebilir
bir şey değil — TLS şifrelemesi altında içerik incelemesi ayrı, çok daha karmaşık bir
konu, `toFQDNs`'in kapsamında değil).

---

## 6. Gözlemlenebilirlik: Hubble

Her iki katmanın DROP kararı da Hubble'da görünür, `drop_reason_desc: POLICY_DENIED` ile
etiketlenmiş:

```bash
cilium hubble port-forward &          # Hubble Relay'e bağlan (bir kere)
hubble observe --pod default/<pod-adı> --verdict DROPPED -o json --last 200
```

Sistem bunu otomatikleştiriyor (`sandbox_runner.get_denied_actions`):
- Sandbox pod'u için: pod adına göre (her run'ın kendine özel, kısa ömürlü bir pod'u var).
- Tool Gateway için: paylaşılan/uzun-ömürlü olduğu için, sorgusu run'ın başladığı zamandan
  itibaren (`--since <N>s`) SINIRLANIYOR — yoksa başka bir run'ın (veya elle yapılan bir
  testin) drop'u yanlışlıkla bu run'a mal edilir.

**Bilinen sınırlama:** Hubble gözlemlenebilirliği best-effort — `hubble` CLI'ye
ulaşılamazsa veya port-forward açık değilse, `get_denied_actions` sessizce `[]` döner.
Bu, ENGELLEMENİN kendisini etkilemez (o zaten Cilium'da, kernel seviyesinde gerçekleşti)
— sadece bunu UYGULAMANIN GÖRÜP RAPORLAYAMAMASI riski var. Demo öncesi
`nc -zv localhost 4245` ile port-forward'ın açık olduğunu doğrulamak gerekir.

### 6.1 Canlı yakalanan ikinci bir best-effort örneği — akış tamponu doluluğu

Bu oturumda (2026-09-01), aynı sınırlamanın SOMUT bir örneği daha yakalandı. Birkaç gün
kesintisiz çalışan cluster'da, `altan.com`/`alta.com` gibi hedeflere yapılan denemeler
uygulama seviyesinde doğru şekilde engellendi (`"Bağlantı zaman aşımına uğradı"`), ama
Hubble'ın flow-log'unda (ne CLI ne UI) BUNLARDAN HİÇBİRİ görünmedi.

**Kök neden bulundu:**
```bash
$ hubble status
Current/Max Flows: 4,095/4,095 (100.00%)   # <- tampon TAMAMEN dolu
```
Hubble'ın akış tamponu sabit boyutlu (4095) ve `cilium-agent`'ın kendi RAM'inde
tutuluyor — gün(ler)ce kesintisiz çalışan bir cluster'da, sistem gürültüsü (coredns/
apiserver heartbeat'leri, DNS alt-sorgu yığınları) bu tamponu doldurup GERÇEK bir ring
buffer'a çevirebiliyor: yeni her olay en eskinin üzerine yazılıyor. Sayıca AZ olan
(1-2 satır) TCP-drop kayıtları, aynı test sırasında üretilen SAYICA ÇOK DAHA FAZLA
(15-20+ satır) DNS alt-sorgu gürültüsü tarafından kolayca "ezilip" tamponun dışına
itilebiliyor.

**Engellemenin kendisi etkilenmedi mi — bağımsız kanıtla doğrulandı:** Hubble'ın
flow-log'undan TAMAMEN bağımsız, Cilium'un kendi kümülatif eBPF sayacı (`hubble`/
flow-log'a hiç ihtiyaç duymayan bir metrik) kontrol edildi:
```bash
$ cilium-dbg metrics list | grep 'cilium_drop_count_total.*Policy denied'
# ÖNCE: 199.000000
# (altan.com denemesi)
# SONRA: 204.000000   -> +5, GERÇEKTEN düşürülmüş
```
Yani engelleme %100 çalıştı — sadece flow-log GÖRSEL olarak yakalayamadı. Bu, "best-effort
gözlem" ile "engellemenin kendisi" arasındaki farkın somut, canlı kanıtı.

**Düzeltme (geçici, tampon dolduğunda):**
```bash
kubectl rollout restart daemonset/cilium -n kube-system   # tampon RAM'de, restart sıfırlar
# ardından port-forward'ı da tazelemek gerekir (eski pod'a bağlı kalmış olabilir):
pkill -f "cilium hubble port-forward"
cilium hubble port-forward &
```
Doğrulama: `hubble status` → `Current/Max Flows` `4095/4095`'ten (`100%`) `477/4095`'e
(`%11.65`) düştü.

**Demo için pratik çıkarım:** Flow-log (CLI/UI) yerine, özellikle YOĞUN DNS trafiği
üreten senaryolarda, `cilium_drop_count_total` gibi kümülatif sayaçlar DAHA GÜVENİLİR
bir kanıt kaynağı — tampon dolup taşmasından hiç etkilenmiyorlar.

---

## 7. Bilinen sınırlamalar (dürüst liste)

- **Tool Gateway sorgusunun zaman-penceresi bazlı olması**: PoC'nin "tek kullanıcılı,
  kimliksiz" varsayımı altında (`spec.md`) düşük risk, ama teorik olarak eşzamanlı bir
  istek/manuel test, yanlış run'a `denied_action` olarak mal edilebilir.
- **DNS'in Tool Gateway için serbest olması**: hangi domain'in SORULDUĞU gizlenmiyor
  (yalnızca hangi domain'e BAĞLANILABİLECEĞİ kısıtlı) — bu bir bilgi sızıntısı değil
  (aktif bağlantı yok), ama "hangi isimler sorgulandı" bilgisi teorik olarak DNS trafiğini
  izleyen biri için görünür olurdu (bu PoC'de tehdit modeli kapsamında değil).
- **`web_search`'ün içerik sınırı yok**: DuckDuckGo üzerinden ONAYLANMAMIŞ domain'ler
  HAKKINDA bilgi edinilebiliyor (network bağlantısı olmadan) — bu ağ-seviyesi bir sorun
  değil, ama tool'un YETENEK tasarımıyla ilgili ayrı bir konu (bkz. bölüm 5).

---

## 8. "Pod olmasa, düz bir container olsa yine çalışır mıydı?"

Bu, mimarinin ne kadar Kubernetes'e bağımlı olduğunu anlamak için önemli bir soru —
kısa cevap: **hayır, şu anki haliyle çalışmaz**, ama nüans önemli.

### Neden Kubernetes'e bağımlı

`CiliumNetworkPolicy`, bir Kubernetes CRD'si (`apiVersion: cilium.io/v2`) — var olabilmesi
için bir K8s API server'a ihtiyaç var. Politika, pod ETİKETLERİNİ eşliyor
(`endpointSelector.matchLabels: {app: tool-gateway}` gibi); Cilium'un kimlik modeli
(CiliumIdentity) de bu K8s pod/namespace etiketlerinden türetiliyor. Düz bir
`docker run` container'ının böyle bir etiketi/kimliği yok — Cilium'un K8s-CNI modu onu
hiç "endpoint" olarak görmez, dolayısıyla bu politika ona hiç uygulanamaz. Bir K8s
API server olmadan `CiliumNetworkPolicy` CRD'sinin kendisi de zaten var olamaz.

### Nüans 1 — aynı Pod içinde ayrı bir container (sidecar) olsaydı

Bu durumda **ÇALIŞIR**, çünkü Kubernetes'te bir Pod'daki TÜM container'lar aynı network
namespace'i (dolayısıyla aynı IP'yi, aynı Cilium endpoint'ini) paylaşır. Cilium container
bazında değil, Pod/network-namespace bazında politika uyguluyor — sandbox'a ikinci bir
container eklesek (aynı Pod'da, ör. bir sidecar), egress kısıtlaması otomatik olarak o
container'ı da kapsardı; ayrıca bir tanımlamaya gerek yok.

### Nüans 2 — eBPF'in kendisi K8s'e özgü değil

Kernel seviyesindeki asıl teknik (paket filtreleme için `tc`/XDP hook'larına BPF programı
bağlamak) K8s'e bağımlı bir şey değil — herhangi bir Linux network namespace'ine (K8s'siz
bir Docker container'ı dahil) uygulanabilir. Ama bunun için Cilium'un bugün sunduğu hazır
CRD/`toFQDNs`/Hubble entegrasyonu YOK — bunları kendin (raw `tc`+BPF, ya da Cilium'un
tarihsel Docker network-plugin modu gibi) yeniden inşa etmen gerekirdi; bu konuda Cilium'un
güncel desteğinin ne durumda olduğu doğrulanmadı, iddialı bir şey söylemiyoruz.

### Sonuç

Demo'nun "Kubernetes Pod + Cilium CRD" mimarisi bilinçli bir seçimdi çünkü bu, PoC'nin en
olgun, üretime-yakın yolu — plain container'a geçmek istenirse bu, egress-kontrol
katmanının TAMAMEN yeniden yazılması anlamına gelir, küçük bir konfig değişikliği değil.

---

## 9. `toFQDNs`'in gizli zayıflığı: paylaşılan IP sorunu

Bu, bu konuşmada bulunan ve CANLI KANITLANMIŞ gerçek bir bulgu — teorik değil.

### Önce açık bir yanlış anlaşılmayı düzeltelim

Bu, pod'larla İLGİSİ OLMAYAN bir konu. `altan.com` da, `virus.altan.com` da (ya da
`console-mia.csp.kloudeks.com` da) bizim cluster'ımızın TAMAMEN DIŞINDaki, internetteki
sıradan domain'ler — hiçbiri bir pod'da çalışmıyor, hiçbiri bizim bir kaynağımız değil.
Sorunun kökeni pod'larda değil, **DNS çözümlemesi + Cilium'un onaylama mekanizmasının
KENDİSİNDE**.

### Mekanizma — Cilium isme değil, IP'ye bakıyor

`toFQDNs`/`matchName` şöyle çalışır: Cilium, onaylı bir ismi (ör. `mia.csp.kloudeks.com`)
DNS ile çözerken dönen CEVABI izliyor ve **"bu IP artık onaylı"** diye bir kayıt açıyor.
Ama bu kayıt "hangi İSİM için" değil, sadece "hangi IP" bilgisini tutuyor. Sonraki bir TCP
bağlantısı geldiğinde, Cilium'un eBPF programı SADECE paketin hedef IP'sine bakabiliyor
(TLS şifreli bir bağlantının içindeki gerçek hedef ismini — SNI'yi — göremiyor, bizim
kurulumumuzda buna bakan bir Envoy/L7 katmanı yok). Yani:

- `altan.com` onaylı listeye eklenir → Cilium onun IP'sini (`X.X.X.X`) öğrenir.
- `virus.altan.com` **AYNI IP'ye (`X.X.X.X`)** çözülüyorsa → bağlantı **GEÇER** — Cilium
  hangi ismin sorulduğunu değil, sadece "bu IP daha önce onaylı bir isim için görüldü mü"
  diye bakıyor, cevap evet.
- `virus.altan.com` FARKLI bir IP'ye çözülüyorsa → engellenir (o IP hiç onaylı değil).

### Canlı kanıt — bu gerçekten oldu

Bu konuşmada test edildi:

```bash
$ python3 -c "import socket; print(socket.gethostbyname('mia.csp.kloudeks.com'))"
185.199.89.67
$ python3 -c "import socket; print(socket.gethostbyname('console-mia.csp.kloudeks.com'))"
185.199.89.67
```

**Aynı IP.** Daha önceki bir testte (bkz. bölüm 6'daki Hubble akışları), Tool Gateway'in
`console-mia.csp.kloudeks.com`'a giden trafiği Hubble'da `world` (tanınmayan kimlik) olarak
etiketlenmişti — AMA yine de `FORWARDED` (izinli) çıkmıştı. Sebep tam olarak bu: isim
tanınmıyordu ama IP zaten `mia.csp.kloudeks.com`'un çözümlemesinden onaylıydı.

### Neden gerçek bir risk

Paylaşımlı barındırma/CDN (Cloudflare, AWS CloudFront, Azure Front Door, vb.) kullanan
ortamlarda bu YAYGIN bir durum — birçok farklı domain aynı kenar (edge) IP'lerini
paylaşıyor. Bir saldırgan, onaylı domain'inizle AYNI paylaşımlı IP altyapısını kullanan
kendi domain'ini kaydettirebilirse (ya da zaten var olan, alakasız bir domain o IP'yi
paylaşıyorsa), trafiği o IP'ye "isim" hiç kontrol edilmeden geçebilir.

### Kapatma yolu — uygulandı ve canlı doğrulandı

Bu artık teorik bir öneri değil — `tool-gateway-egress.ciliumnetworkpolicy.yaml`'a
Cilium'un TLS SNI-farkında filtreleme özelliği (`serverNames`, Envoy tabanlı L7 proxy
üzerinden) eklendi ve cluster'a uygulandı.

**Değişiklik** — `toPorts` bloğuna, `ports`'un kardeşi olarak yeni bir alan:

```yaml
    toPorts:
    - ports:
        - port: "443"
          protocol: TCP
      serverNames:                              # yeni
      - "mia.csp.kloudeks.com"
      - "openaipublic.blob.core.windows.net"
      - "html.duckduckgo.com"
```

`serverNames` boş değilse, Cilium bu portu artık saf eBPF (L3/L4) yerine `cilium-envoy`'a
yönlendiriyor — TLS handshake'inin ilk mesajının (`ClientHello`) DÜZ METİN taşıdığı SNI
alanını okuyor (şifre çözme/MITM YOK — SNI, şifreleme daha başlamadan gönderiliyor) ve bu
listedeki isimlerden biriyle eşleşmiyorsa bağlantıyı kesiyor.

**Canlı doğrulama — iki farklı katmanda iki farklı sonuç görüldü:**

1. `mia.csp.kloudeks.com` hiç sorgulanmamışken `console-mia.csp.kloudeks.com` denendi →
   Cilium'un onaylı-IP kaydı henüz yokken saf **L3/L4** seviyesinde reddedildi (Hubble:
   `Policy denied DROPPED (TCP Flags: SYN)` — SYN hiç geçmedi; Python tarafında
   `TimeoutError`).
2. `mia.csp.kloudeks.com` bir kere sorgulanıp IP'si zaten "onaylı" olarak cache'lendikten
   SONRA `console-mia.csp.kloudeks.com` tekrar denendi — saldırının gerçekte işe yaraması
   için gereken tam senaryo budur, çünkü onaylı domain zaten çözülmüş olmalı → bu sefer
   **L3/L4 hâlâ kandı**: SYN/SYN-ACK/ACK Hubble'da `ALLOWED`/`FORWARDED` göründü, çünkü IP
   zaten onaylı listedeydi. Ama TLS `ClientHello` (SNI=`console-mia...`) gönderilir
   gönderilmez ayrı bir olay belirdi: `l7-request DROPPED ()` — plain L3/L4'ün
   `Policy denied` etiketinden FARKLI bir sınıf, Envoy'un kendi L7 kararı — ve bağlantı
   orada sonlandı.

Bu ikinci sonuç, savunmanın tam olarak tasarlandığı gibi çalıştığının kanıtı: **L3/L4
paylaşılan-IP'ye hâlâ aldanıyor (beklenen — bu katman ismi hiç göremiyor), ama L7/SNI
katmanı impersonation'ı orada yakalıyor.** Onaylı 3 hedef (`mia.csp.kloudeks.com`,
`openaipublic.blob.core.windows.net`, `html.duckduckgo.com`) etkilenmeden çalışmaya devam
ediyor.
