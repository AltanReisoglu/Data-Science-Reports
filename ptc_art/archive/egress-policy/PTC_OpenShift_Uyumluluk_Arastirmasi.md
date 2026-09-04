# OpenShift Uyumluluk Araştırması

Bu doküman, "bizim PoC OpenShift'e uygun mu" sorusunu araştırıyor — özellikle **CNI
değişimi**nin ne demek olduğunu, bizim `kind` sürecimizle karşılaştırarak açıklıyor.
Bir karar/uygulama dokümanı DEĞİL, sadece araştırma.

## 1. CNI nedir — kısa hatırlatma

`PTC_Calisma_Sureci_Kubernetes_Cilium.md`'de (§3) zaten anlattık: bir pod doğarken,
container'ları çalışmaya başlamadan ÖNCE, kubelet bir **CNI (Container Network
Interface)** eklentisini çağırmak ZORUNDADIR — pod'a IP atamak, ağını kurmak için.
Bizim durumumuzda bu eklenti **Cilium** (`cilium-cni`).

**Kritik nokta:** Bir cluster'da AYNI ANDA sadece **BİR TANE** CNI olabilir — iki CNI
birbiriyle çakışır, hangi paketi kimin yöneteceği belirsizleşir.

## 2. Bizim kendi sürecimiz — "boş bir yuvaya yerleştirme"

Bizim `k8s/kind-config.yaml`'a bakalım:
```yaml
kind: Cluster
networking:
  disableDefaultCNI: true   # <- kind'ın KENDİ CNI'sini (kindnet) hiç kurmuyoruz
```

Kurulum sırasıyla:
```bash
kind create cluster --config=k8s/kind-config.yaml
# Bu an itibariyle cluster VAR ama HİÇ CNI YOK — pod'lar network alamıyor,
# hepsi "Pending" durumunda bekliyor (CNI'sız bir pod ağa çıkamaz).

helm install cilium cilium/cilium --version 1.20.0 --namespace kube-system ...
# Cilium, BOŞ bir yuvaya, İLK ve TEK CNI olarak yerleşiyor.
```

Yani bizim sürecimiz **temiz bir sayfaya yazmak** gibi — hiçbir şeyi SÖKMEDİK, sadece
boş bir yere Cilium'u koyduk. Çakışma riski yok, geri dönüş (rollback) basit
(`kind delete cluster`, baştan başla).

## 3. OpenShift'te durum TAMAMEN FARKLI — "dolu bir yuvayı değiştirme"

OpenShift, kurulumdan İTİBAREN kendi CNI'sini (**OVN-Kubernetes**) ZATEN kurulu ve
ÇALIŞIR durumda getiriyor. Boş bir yuva yok — **dolu bir yuva var**. Cilium'u
kurmak için önce ESKİSİNİ SÖKMEK, sonra YENİSİNİ TAKMAK gerekiyor — bu bambaşka bir
risk sınıfı.

### Gerçek göç süreci (Cilium'un OVN-Kubernetes'ten göç rehberinden) — 7 adım

1. **Cluster Network Operator'ı durdur** — OpenShift'in kendi otomatik ağ
   yönetimini geçici olarak devre dışı bırak (yoksa OpenShift, senin yaptığın
   değişikliği "hata" sanıp geri alır)
2. **Machine Config Pool'ları duraklat** — node'ların otomatik yeniden başlamasını
   engelle (değişiklikler henüz tamamlanmadan node'lar restart olmasın diye)
3. **Eski ağ durumunu temizle** — `applied-cluster` configmap'ini sil (yeni CNI
   tanımının kabul edilmesi için)
4. **Ağ objelerini yeniden yapılandır** — pod CIDR'ını OVN'in varsayılanından
   (`10.128.0.0/14`) Cilium'un aralığına değiştir, `networkType: "Cilium"` yap,
   kube-proxy'yi devre dışı bırak
5. **Cilium Operator'ı kur** — OLM (Operator Lifecycle Manager) manifestleri İKİ
   KEZ uygulanır (önce CRD'ler, sonra asıl `CiliumConfig` kaynağı)
6. **Cluster yönetimini geri aç** — durdurulan operator'ları tekrar etkinleştir,
   artık Cilium'a işaret ediyor olacaklar
7. **Node'ları yeniden başlat** — duraklatılan Machine Config Pool'ları serbest
   bırak, TÜM node'lar sırayla reboot olur

**Elle müdahale gerekebilir:** Bazı node'lar otomatik ilerlemeyip
`schedulingDisabled` durumunda takılabiliyor — bu durumda elle `cordon` → `drain`
→ `oc debug node/` ile node'a girip `systemctl reboot` gerekiyor.

### Bizim süreçle yan yana karşılaştırma

| | Bizim `kind` sürecimiz | OpenShift göçü |
|---|---|---|
| Başlangıç durumu | CNI hiç yok (boş yuva) | OVN-Kubernetes zaten ÇALIŞIYOR (dolu yuva) |
| Adım sayısı | 2 (cluster oluştur + helm install) | 7+ (durdur, duraklat, temizle, yeniden yapılandır, kur, geri aç, reboot) |
| Node reboot gerekir mi | Hayır | **Evet, TÜM node'lar** |
| Kesinti (downtime) | Yok (henüz hiçbir iş yükü yok) | **Var — kaynak, bunu "significant cluster downtime" olarak tanımlıyor** |
| Geri dönüş | `kind delete cluster`, baştan başla | **Belgelenmemiş** — kaynağın kendi ifadesiyle *"reverting... would require reversing these steps and likely involves cluster reinstallation"* |
| Resmi destek durumu | — | Çelişkili: bir kaynak "desteklenen ama dikkatli sıralanmış", göç rehberinin kendisi *"this operation is officially unsupported"* diyor |

## 4. Dürüst bir uyarı — kaynaklar birbiriyle çelişiyor

Bölüm 1'deki araştırmam "Cilium OpenShift'i resmi olarak destekliyor" diyordu
(YENİ bir cluster'a Cilium'u BAŞTAN kurmak bağlamında — tıpkı bizim `kind`
sürecimiz gibi, boş bir cluster'a). Ama BU bölümdeki göç rehberi, MEVCUT bir
OpenShift cluster'ını (zaten OVN-Kubernetes çalışırken) Cilium'a GEÇİRMENİN
*"officially unsupported"* olduğunu söylüyor. Bu ikisi ÇELİŞMİYOR aslında — iki
FARKLI senaryo:

- **Senaryo A — sıfırdan bir OpenShift cluster'ı kurarken Cilium'u seçmek:**
  Resmi olarak destekleniyor, bizim `kind` sürecimize benzer (temiz kurulum).
- **Senaryo B — ÇALIŞAN bir OpenShift cluster'ını sonradan Cilium'a geçirmek:**
  Riskli, kesintili, resmi desteği belirsiz/tartışmalı.

**Bizim PoC'miz için pratik sonuç:** Eğer OpenShift'e taşınacaksa, cluster'ın
SIFIRDAN, Cilium ile kurulması (Senaryo A) — mevcut bir OpenShift cluster'ını
sonradan Cilium'a çevirmeye (Senaryo B) çalışmaktan çok daha güvenli bir yol.

## 5. Sonuç

CNI değişimi, bizim `kind` deneyimimizdeki "boş bir yuvaya bir şey koymak"tan
TAMAMEN farklı bir operasyon sınıfı — "dolu bir yuvadaki şeyi söküp yenisini
takmak," ve bunun (OpenShift özelinde) node reboot'ları, kesinti ve belirsiz bir
geri-dönüş yolu içerdiği görülüyor. Bizim PoC'nin GÜVENLİK MANTIĞI (CiliumNetworkPolicy
dosyaları) taşınabilir olsa da, ALTINDAKİ ağ altyapısının nasıl kurulacağı — sıfırdan
mı, yoksa mevcut bir cluster'ın göçürülmesiyle mi — sonucu büyük ölçüde belirliyor.
