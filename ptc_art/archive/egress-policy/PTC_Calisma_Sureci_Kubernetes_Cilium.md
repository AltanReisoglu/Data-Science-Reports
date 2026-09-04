# Bir PTC Çalıştırması — Kubernetes ve Cilium Süreçleri Baştan Sona

Bu doküman, web arayüzünden bir soru sorulduğunda (ve bu sorunun `run_ptc_code`'u
tetiklediğinde), **Kubernetes'in** ve **Cilium'un** (özellikle `cilium-agent`'ın) TAM
OLARAK ne zaman, hangi sırayla devreye girdiğini anlatıyor. İkisi iç içe geçmiş iki ayrı
süreç — bu doküman onları hem ayrı ayrı hem birlikte, zaman sırasıyla açıklıyor.

## 1. Genel bakış — kim ne zaman devreye giriyor

```
LLM "run_ptc_code" çağırır
        │
        ▼
[KUBERNETES] ConfigMap oluşturulur (kod yazılır)
        │
        ▼
[KUBERNETES] Job oluşturulur  ──┐
        │                       │  BU ANDA (pod'un container'ı
        ▼                       │  BAŞLAMADAN ÖNCE) Cilium devreye girer:
[KUBERNETES] Scheduler,         │  kubelet → cilium-cni → cilium-agent
   Job'un pod'unu bir node'a    │  (IPAM, identity, eBPF programlarını
   atar                         │  yükleme) — 2. bölüme bakın
        │                       │
        ▼                       │
[KUBERNETES] kubelet, pod'u  ◄──┘
   ayağa kaldırır (container
   çalışmaya başlar)
        │
        ▼
[CILIUM] her paket, eBPF'in ALLOW/DENY kararından geçer (3. bölüm)
        │
        ▼
[KUBERNETES] Job biter (Succeeded/Failed/DeadlineExceeded)
        │
        ▼
[KUBERNETES] sandbox_runner.py, pod log'unun son satırını okur
        │
        ▼
[KUBERNETES] ConfigMap + Job açıkça silinir (cleanup)
```

## 2. Kubernetes süreci — adım adım, kod referanslarıyla

### 2.1 ConfigMap oluşturma

`sandbox_runner.py`'nin `run_sandbox()` fonksiyonu çağrıldığında, önce LLM'in ürettiği
kod bir ConfigMap'e yazılır:

```python
# sandbox_runner.py:84-89
def _create_configmap(core_v1, run_id, code):
    configmap = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=f"ptc-code-{run_id}"),
        data={"code.py": code},
    )
    core_v1.create_namespaced_config_map(namespace=NAMESPACE, body=configmap)
```

Bu an, PTC panelindeki `configmap_created` event'i olarak görünür. Henüz HİÇBİR pod
yok — sadece kod, cluster'ın etcd'sinde bir obje olarak duruyor.

### 2.2 Job oluşturma

`k8s/sandbox/job-template.yaml`, `{run_id}` ve `{tool_gateway_endpoint}` yer
tutucuları doldurularak Kubernetes API'sine gönderilir. Bu, `job_created` event'ini
tetikler. Job'un tanımı:
- `activeDeadlineSeconds: 30` — en fazla 30sn yaşar
- `backoffLimit: 0` — hata olursa yeniden deneme YOK
- ConfigMap'i `/sandbox` altına volume olarak mount eder

### 2.3 Scheduler ve kubelet — pod GERÇEKTEN doğuyor

Job objesi oluşunca, Kubernetes'in **scheduler**'ı bu Job'un pod'unu bir node'a atar
(bizde tek node olduğu için seçim basit). Ardından o node'un **kubelet**'i pod'u ayağa
kaldırmaya başlar — AMA container'lar çalışmaya başlamadan ÖNCE, kubelet bir CNI
(Container Network Interface) eklentisini çağırmak ZORUNDADIR: pod'un ağı henüz yok.
**İşte Cilium tam burada, bu noktada devreye giriyor** — bölüm 3'e bakın.

### 2.4 Job'un bitmesini bekleme

`sandbox_runner.py`, Job'un durumunu 1 saniyede bir sorgulayarak (`_POLL_INTERVAL_SECONDS`)
bekler; pod'un log'undaki JSON satırlarını okuyup PTC paneline akıtır (`tool_call`
event'leri buradan gelir — `entrypoint.py`'nin her tool çağrısında yazdığı satırlar).

### 2.5 Temizlik

Job Succeeded/Failed/DeadlineExceeded olunca, `sandbox_runner.py` ConfigMap'i ve Job'u
**açıkça, beklemeden** siler (`ttlSecondsAfterFinished: 300` zaten var ama Principle V
gereği hemen de siliniyor). Bu an itibariyle pod da (Job'un çocuğu olduğu için)
kaybolur — `kubectl get pods` artık onu göstermez.

## 3. Cilium süreci — pod doğarken ne oluyor (CNI akışı)

Bu, projedeki EN KRİTİK ama en az görünür an — pod'un container'ı çalışmaya
başlamadan önce, sessizce gerçekleşiyor:

```
kubelet ──(CNI çağrısı)──► cilium-cni (eklenti)
                                  │
                                  ▼ (UNIX socket ile)
                            cilium-agent
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
         IPAM: pod'a IP      Security identity    eBPF programlarını
         atanır              atanır (etiketlere    veth'in node tarafına
         (10.244.0.x)        göre — ör. sandbox    TAKAR (tc hook)
                             pod'ları hep AYNI
                             identity'yi paylaşır)
```

**Bu adım BİTMEDEN pod "Running" durumuna geçmez** — yani bir sandbox pod'u
`Running` olduğunda, o ana kadar ZATEN kendi eBPF programları takılmış, kendi
identity'si atanmış durumdadır. `sandbox-egress` politikası (eğer o identity'yi
seçiyorsa) bu noktada ZATEN eBPF map'ine yazılmış olur — pod'un ilk paketi bile
gitmeden önce kısıtlama hazırdır (yani bir "yarış durumu" / kısa süreli açık pencere
yoktur).

**Kanıt — canlı `cilium-dbg endpoint list` çıktısından** (bu oturumda çalıştırdık):
her pod bir `ENDPOINT` numarası (Cilium'un kendi iç ID'si) ve bir `IDENTITY`
numarası (etiketten türetilen) alıyor; `POLICY (egress) ENFORCEMENT` sütunu
`Enabled`/`Disabled` — bir politika o identity'yi seçtiyse `Enabled` görünüyor.

## 4. Cilium süreci — çalışırken (runtime enforcement)

Pod çalışırken, HER paket için:

1. **eBPF (`tc` hook, kernel içinde)** — paketin hedef IP'sini bir `ipcache`
   map'inde identity'ye çevirir, o identity+port+protokol kombinasyonu için
   ÖNCEDEN derlenmiş policy map'ine bakar → ALLOW ya da DROP, mikrosaniyeler
   içinde, userspace'e hiç çıkmadan.
2. **L7 istisnası — DNS sorgu adı filtreleme** — `cilium-agent`'ın kendi
   GÖMÜLÜ DNS proxy'sine (Envoy DEĞİL, ayrı bir bileşen) yönlendirilir.
3. **L7 istisnası — TLS SNI (`serverNames`)** — TPROXY ile `cilium-envoy`
   DaemonSet'ine (node başına 1 tane, ayrı bir proxy) yönlendirilir; SNI
   düz-metin olarak okunur, şifre çözme YOK.
4. **Hubble** — bu kararların bir kısmını (best-effort — bkz.
   `PTC_Egress_Policy_Implementation_Walkthrough.md` §6/§6.1) flow-log olarak
   kaydeder; engellemenin KENDİSİ Hubble'dan bağımsız gerçekleşir.

## 5. İkisini birleştiren tam zaman çizelgesi

| Zaman | Kubernetes | Cilium |
|---|---|---|
| t=0 | ConfigMap oluşturulur | — |
| t=0.1 | Job oluşturulur | — |
| t=0.2 | Scheduler pod'u node'a atar | — |
| t=0.3 | kubelet, CNI'yi çağırır | `cilium-cni` → `cilium-agent`: IPAM + identity + eBPF attach |
| t=0.5 | Pod `Running` | eBPF ZATEN hazır — ilk paket bile bu kısıtlamayla karşılaşır |
| t=0.5-Ns | Container kodu çalışır, `fetch_url` vb. çağırır | Her paket eBPF'ten (gerekirse DNS-proxy/Envoy'dan) geçer |
| t=N | Job biter | — |
| t=N+0.1 | `sandbox_runner.py` sonucu okur | — |
| t=N+0.2 | ConfigMap + Job silinir | Pod'un identity/eBPF durumu da temizlenir |

## 6. Neden bu sıralama önemli — güvenlik açısından

Kritik nokta: Cilium'un kısıtlaması, pod'un uygulama kodu çalışmaya başlamadan
**ÖNCE** hazır. Yani LLM'in ürettiği kod ne kadar "kötü niyetli" olursa olsun, ilk
paketini göndermeden önce zaten bir eBPF duvarının arkasındadır — kod çalışırken
"araya girip" bir kısıtlama eklemek gibi bir senaryo yok, kısıtlama zaten en baştan
var.
