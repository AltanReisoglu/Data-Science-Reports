# PTC — Günlük Çalışma Komutları

Bu oturum boyunca fiilen kullanılan, işe yarayan komutların temiz hâli — her
biri "ne işe yaradığı" başlığıyla. (`PTC_Komut_Referansi.md`'nin daha kapsamlı/
sıfırdan-kurulum odaklı hâlinden farklı olarak, bu dosya GÜNLÜK kullanılan,
tekrar eden komutlara odaklanıyor.)

> Depoda ne var, kim üretti, kim okudu →
> [PTC_Artifact_Inceleme_Komutlari.md](PTC_Artifact_Inceleme_Komutlari.md)

---

## 0. Şu anki kurulum — aç, doğrula, kapat

*2026-09-08'de canlı cluster'da doğrulandı.*

### Ne nerede

| | Adres | Ne için |
|---|---|---|
| Web konsolu | `http://localhost:8123/konsol` | sohbet · hatlar · depo · soy ağacı |
| artifact-service | `localhost:8080` | manifest + `artifact_ara` bu adresi kullanıyor |
| Hubble UI | `http://localhost:12000` | pod'lar arası akış haritası |
| Hubble relay | `localhost:4245` | `hubble observe` bunu kullanıyor |

### Aç

```bash
cd ~/Desktop/Data-Science-Reports/ptc_art
S=/tmp/ptc-log; mkdir -p $S

# 1) Kayıt defteri — .env'deki ARTIFACT_SERVICE_URL buna bakıyor.
#    Açık değilse manifest SESSİZCE devre dışı kalır (2026-09-06 dersi).
nohup kubectl port-forward svc/artifact-service 8080:8080 > $S/pf.log 2>&1 & disown

# 2) Web konsolu
PYTHONPATH=src nohup ../ptc_sec/.venv/bin/uvicorn \
  grounded_assistant.web.app:app --port 8123 --log-level info > $S/web.log 2>&1 & disown

# 3) Hubble (isteğe bağlı — yalnızca izlemek için)
nohup cilium hubble port-forward > $S/hubble.log 2>&1 & disown
nohup kubectl port-forward -n kube-system svc/hubble-ui 12000:80 > $S/hubble-ui.log 2>&1 & disown
```

### Doğrula

```bash
kubectl get pods            # artifact-service · minio · tool-gateway → Running
curl -s -o /dev/null -w "konsol %{http_code}\n" http://localhost:8123/konsol   # 200
curl -s http://localhost:8123/api/depo | head -c 200                            # kayıtlar
hubble status | grep -E "Healthcheck|Flows"
```

> `curl http://localhost:8080/artifacts` **401 döner** — bu doğru davranış,
> servis kapsam jetonu istiyor. Jetonlu çağrılar için
> [PTC_Artifact_Inceleme_Komutlari.md](PTC_Artifact_Inceleme_Komutlari.md) §0.

### Kapat

```bash
ss -ltnp | grep -E ':(8123|8080|4245|12000)'   # pid'leri gör
kill <pid>                                      # tek tek
```

> **`pkill -f` KULLANMA.** Kendi komut satırıyla eşleşip kabuğu öldürüyor —
> bu oturumda üç kez oldu (çıkış 144). `ss` çıktısındaki pid'i kullan.

---

## 1. Bir politikayı değiştirdikten sonra yeniden uygulama

```bash
kubectl apply -f k8s/policies/tool-gateway-egress.ciliumnetworkpolicy.yaml
```
YAML dosyasında değişiklik yaptıktan sonra (ör. `serverNames` eklemek gibi) bunu
çalıştırıp cluster'a yansıtmak için. `configured` çıktısı gerçek bir değişiklik
oldu demek, `unchanged` çıktısı dosya zaten cluster'daki hâliyle aynı demek.

## 2. Cilium pod'unu ELLE yeniden başlatma (tek pod, alternatif yöntem)

```bash
# 1. Pod adını bul
kubectl get pods -n kube-system -l k8s-app=cilium

# 2. Sil (DaemonSet otomatik yeni bir tane yaratır)
kubectl delete pod cilium-xpwtq -n kube-system

# 3. Yeni pod'un sağlıklı ayağa kalkmasını bekle
kubectl -n kube-system rollout status daemonset/cilium --timeout=90s
```
`kubectl rollout restart daemonset/cilium`'un yaptığı işin AYNISI, ama tek
pod'u doğrudan hedefleyerek — DaemonSet'in "eksik pod'u otomatik tamamlama"
davranışından faydalanıyor. Hubble'ın akış tamponu doluysa bunu boşaltmanın
bir yolu (bkz. bölüm 8).

## 3. Servisleri başlatma (temel — sadece 3 servis)

```bash
cd /home/altan/Desktop/Data-Science-Reports/ptc_sec
source .venv/bin/activate
nohup uvicorn grounded_assistant.web.app:app --port 8123 --log-level info > /tmp/uvicorn.log 2>&1 &
disown

nohup cilium hubble port-forward > /tmp/hubble-pf.log 2>&1 &
disown

nohup kubectl port-forward -n kube-system svc/hubble-ui 12000:80 > /tmp/hubble-ui-pf.log 2>&1 &
disown
```
Web arayüzü + Hubble Relay (CLI'nin `hubble observe` çalıştırabilmesi için) +
Hubble UI (görsel arayüz). `nohup ... & disown` kombinasyonu, terminal
kapansa/oturum bitse bile servislerin arka planda çalışmaya devam etmesini
sağlıyor.

## 4. Başlatmadan önce/sonra sağlık kontrolü (genişletilmiş)

```bash
# Her şeyin sağlıklı olduğunu doğrula
kubectl get pods -A
kubectl get ciliumnetworkpolicy

# Servisleri başlat
cilium hubble port-forward &
kubectl -n kube-system port-forward svc/hubble-ui 12000:80 &
uvicorn grounded_assistant.web.app:app --port 8123 --log-level info &
```
Bölüm 3'ün daha kısa/hızlı hâli — arka plan loglarını dosyaya yazmadan, hızlıca
başlatmak için (tek seferlik/geçici oturumlarda yeterli).

## 5. Pod'ları izleme

```bash
# Tüm namespace'lerdeki tüm pod'lar (en kapsamlısı)
kubectl get pods -A

# Sadece bizim namespace'imiz (tool-gateway + varsa aktif sandbox job pod'u)
kubectl get pods

# Sadece PTC sandbox pod'ları (etikete göre filtrele)
kubectl get pods -l app=ptc-sandbox

# Canlı izlemek için (bir soru sorup sandbox pod'unun doğup ölmesini anlık görmek istersen)
kubectl get pods -w
```
Özellikle son komut (`-w`, watch) demo sırasında "sandbox pod'u gerçekten
doğup ölüyor" iddiasını CANLI göstermek için kullanışlı.

## 6. Tool Gateway'i kod değişikliğinden sonra güncelleme

```bash
cd /home/altan/Desktop/Data-Science-Reports/ptc_sec
docker build -t tool-gateway:local -f mock_services/tool_gateway/Dockerfile . 2>&1 | tail -5
kind load docker-image tool-gateway:local --name ptc-sec 2>&1
kubectl rollout restart deploy/tool-gateway
kubectl rollout status deploy/tool-gateway --timeout=60s
```
`mock_services/tool_gateway/` altında (ya da onun bağımlı olduğu `src/` kodunda)
bir değişiklik yaptıktan sonra bunu ÇALIŞAN pod'a yansıtmak için — build →
kind cluster'ına yükle → deployment'ı yeniden başlat (yeni image'ı çeksin) →
sağlıklı ayağa kalkmasını bekle. `--name ptc-sec` ZORUNLU — cluster'ın adı
`kind` değil, `ptc-sec`, yoksa `kind load` "no nodes found" hatası verir.

## 7. Bilgisayar kapanıp açıldıktan sonra tam kontrol + başlatma

```bash
# 1. Cluster'ın zaten çalıştığını kontrol
docker ps --filter "name=ptc-sec" --format "{{.Names}}: {{.Status}}"
kubectl get nodes
kubectl get pods -A

# 2. Zaten çalışan port-forward/uvicorn var mı kontrol
ps aux | grep -E "port-forward|uvicorn" | grep -v grep

# 3. Politikaların geçerliliğini kontrol
kubectl get ciliumnetworkpolicy

# 4-6. Servisleri arka planda başlat
cilium hubble port-forward &
kubectl -n kube-system port-forward svc/hubble-ui 12000:80 &
source .venv/bin/activate && uvicorn grounded_assistant.web.app:app --port 8123 --log-level info &

# 7. Üçünün de gerçekten açık olduğunu doğrula
nc -zv localhost 4245
nc -zv localhost 12000
nc -zv localhost 8123

# 8. Uçtan uca son kontrol
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8123/
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:12000/
hubble observe --last 3 -o compact
```
Docker/kind cluster reboot'u genelde atlatıyor (container olarak kalıcı) ama
port-forward'lar/web arayüzü işlemleri KAYBOLUYOR — bunları yeniden başlatmak
gerekiyor. Bu, "her şey kapandı, baştan aç" senaryosunun TAM kontrol listesi.

## 8. Hubble ile pod'lar arası ilişkiyi izleme

Etiketler politikalardan geliyor: `app=ptc-sandbox`, `app=tool-gateway`,
`app=artifact-service`, `app=minio`.

```bash
# Sandbox'ın TÜM trafiği — canlı
hubble observe --label app=ptc-sandbox -f

# İzin verilen tek yol: sandbox → gateway
hubble observe --from-label app=ptc-sandbox --to-label app=tool-gateway

# ASIL GÖSTERİLECEK: reddedilenler
hubble observe --label app=ptc-sandbox --verdict DROPPED

# Artifact yolu: sidecar → servis → minio
hubble observe --to-label app=artifact-service --last 50
hubble observe --from-label app=artifact-service --to-label app=minio

# Bir hattı koştururken yakala (ayrı terminal)
hubble observe --label app=ptc-sandbox -f --output compact
```

`sandbox-egress` politikası ne diyor:

```
endpointSelector: app=ptc-sandbox
  ├─ tool-gateway:8443       İZİN
  ├─ artifact-service:8080   İZİN   ← sidecar için
  └─ diğer her şey           örtük RED
```

> **Sunumda dikkat:** Cilium politikası **pod** düzeyinde çalışıyor, container
> düzeyinde değil. Hubble'da `ptc-run-xxx → artifact-service` akışını
> gördüğünde bu **sidecar'ın** trafiği — ama Hubble bunu ayırt etmiyor.
> Sandbox container'ının erişememesinin sebebi ağ politikası DEĞİL: anahtar
> yok, SDK kurulu değil, servis adresi ortamında tanımlı değil.
>
> Hubble ayrıca **kodun bağımlılığı değil** — `DENIED_ACTION` sinyali
> 2026-09-03'te kaldırıldı, yerine `ag_engeli_gibi()` metin sınıflandırması
> geldi. Hubble artık yalnızca gözlem aracı.

---

## 9. Hubble UI/CLI yenilenmiyor / akış göstermiyor (tampon dolu)

```bash
# Tamponu sıfırla (cilium-agent'ı yeniden başlat)
kubectl rollout restart daemonset/cilium -n kube-system

# Eski port-forward artık ölü pod'a bağlı kalmış olur — tazele.
# pkill DEĞİL: kendi komut satırıyla eşleşip kabuğu öldürüyor.
kill $(ss -ltnp | grep :4245 | grep -oP 'pid=\K[0-9]+' | head -1)
cilium hubble port-forward &

# Doğrula
kubectl get pods -A          # hepsi Running mi
kubectl get ciliumnetworkpolicy   # politikalar hâlâ VALID mi
hubble status                # tampon gerçekten boşaldı mı: 4095/4095 -> 477/4095
```
Cluster günlerce kesintisiz çalışınca Hubble'ın sabit boyutlu (4095) akış
tamponu dolup gerçek bir ring buffer'a dönüşüyor — yeni engellemeler eskilerin
üzerine yazılıp GÖRÜNMEZ oluyor (engellemenin kendisi etkilenmiyor, sadece
GÖRÜNÜRLÜĞÜ). `cilium-agent` RAM'de tuttuğu için, agent'ı yeniden başlatmak
tamponu sıfırlıyor — ama bunun yan etkisi olarak eski port-forward bağlantısı
(artık silinmiş pod'a işaret ettiği için) da tazelenmesi gerekiyor.
