# PTC PoC — Tam Komut Referansı

Bu doküman, PoC'yi sıfırdan kurmak, çalıştırmak, gözlemlemek ve test etmek için
gereken TÜM komutları tek yerde topluyor. Kaynak: `README.md`, `specs/*/quickstart.md`
ve bu oturumda canlı doğrulanmış komutlar.

**Önemli düzeltme:** `specs/002-ptc-code-sandbox/quickstart.md`'deki `docker build`
komutları ARTIK YANLIŞ — Dockerfile'lar daha sonra repo-kökü build context'i
gerektirecek şekilde değişti (`COPY src/`, `COPY sample_docs/`, `COPY sandbox_image/entrypoint.py`
gibi yollar sadece repo kökünden çalışır). Bu dokümandaki komutlar, doğrulanmış/güncel
olan `README.md`'deki hâliyle veriliyor.

---

## 1. Sıfırdan kurulum — cluster + Cilium + Hubble

```bash
# Yerel Kubernetes cluster (kind)
kind create cluster --config=k8s/kind-config.yaml

# Cilium + Hubble (relay + UI dahil, tek adımda) — cluster'ın şu anki canlı hâliyle birebir
helm repo add cilium https://helm.cilium.io/
helm install cilium cilium/cilium --version 1.20.0 --namespace kube-system \
  --set ipam.mode=kubernetes \
  --set operator.replicas=1 \
  --set hubble.enabled=true \
  --set hubble.relay.enabled=true \
  --set hubble.ui.enabled=true

# Cilium'un ayağa kalkmasını bekle
cilium status --wait
```

## 2. PoC image'larını build edip cluster'a yükleme

```bash
# Repo kökünden çalıştırılmalı — ikisi de repo-kökü context'i gerektiriyor
docker build -t tool-gateway:local -f mock_services/tool_gateway/Dockerfile .
kind load docker-image tool-gateway:local

docker build -t ptc-sandbox:local -f sandbox_image/Dockerfile .
kind load docker-image ptc-sandbox:local
```

## 3. Tool Gateway'i deploy etme + politikaları uygulama

```bash
# .env'deki LLM/embedding gateway bilgilerini Secret olarak aktar
kubectl create secret generic tool-gateway-env --from-env-file=.env

# Tool Gateway (Deployment + Service) + Cilium politikaları
kubectl apply -f k8s/tool-gateway/ -f k8s/policies/
```

## 4. Uygulamayı çalıştırma

```bash
# CLI (Faz 1/2)
python -m grounded_assistant.cli ask "Uzaktan çalışma politikamız nedir?" --trace

# Web arayüzü (Faz 4) — bu oturumda kullanılan port
uvicorn grounded_assistant.web.app:app --port 8123 --log-level info
# tarayıcı: http://localhost:8123
```

## 5. Hubble gözlem komutları

```bash
# Hubble Relay'e bağlan (arka planda) — hubble CLI'nin localhost:4245'e bağlanmasını sağlar
cilium hubble port-forward &

# Hubble UI'yi tarayıcıda açmak için (bu oturumda kullanılan port)
kubectl -n kube-system port-forward svc/hubble-ui 12000:80 &
# tarayıcı: http://localhost:12000

# Genel akış izleme
hubble observe --pod default/<pod-adı> --last 50

# Sadece reddedilenleri göster
hubble observe --pod default/<pod-adı> --verdict DROPPED --last 20
hubble observe --pod default/<pod-adı> --verdict DENIED --last 20

# Sadece geçenleri göster (izinli akış gerçekten çalışıyor mu kontrolü)
hubble observe --pod default/<pod-adı> --verdict FORWARDED --last 10

# Belirli bir süre öncesinden itibaren (run'ın başladığı zamana göre)
hubble observe --pod default/tool-gateway-xxxxx --since 2m -o compact

# Hubble'ın akış tamponu doluluk oranı (2026-09-01: 100% dolunca eski akışlar
# ezilip kayboluyor, yeni testler flow-log'da görünmeyebiliyor — bkz.
# PTC_Egress_Policy_Implementation_Walkthrough.md §6.1)
hubble status

# Tampon doluysa (Current/Max Flows ~100%) — cilium-agent'ı yeniden başlatıp sıfırla
kubectl rollout restart daemonset/cilium -n kube-system
kubectl rollout status daemonset/cilium -n kube-system --timeout=120s
# ardından port-forward'ı da tazele (eski pod'a bağlı kalmış olur):
pkill -f "cilium hubble port-forward"
cilium hubble port-forward &

# Flow-log'dan BAĞIMSIZ, tampon doluluğundan hiç etkilenmeyen bir doğrulama yolu —
# engellemenin GERÇEKTEN olduğunu kümülatif sayaçla kanıtla (önce/sonra karşılaştır)
kubectl exec -n kube-system ds/cilium -- cilium-dbg metrics list | grep 'cilium_drop_count_total.*Policy denied'
```

## 6. Cilium/Kubernetes durum ve şema kontrolü

```bash
# Cilium agent durumu (kısa)
kubectl exec -n kube-system ds/cilium -- cilium-dbg status --brief

# kube-proxy replacement durumu (Ingress Controller ön koşulu)
kubectl exec -n kube-system ds/cilium -- cilium-dbg status | grep -i kubeproxy

# Uygulanan tüm CiliumNetworkPolicy'ler ve geçerlilik durumları
kubectl get ciliumnetworkpolicy

# Bir policy'nin tam içeriği (canlı, cluster'daki hâli)
kubectl get ciliumnetworkpolicy tool-gateway-egress -o yaml

# Policy şemasını doğrulamak için (yeni bir alan eklerken)
kubectl explain ciliumnetworkpolicy.spec.egress.toPorts --recursive

# Bir policy'yi GERÇEKTEN uygulamadan önce şemasını doğrula (dry-run)
kubectl apply --dry-run=server -f k8s/policies/<dosya>.yaml

# Node/pod genel sağlık kontrolü
kubectl get nodes -o wide
kubectl get pods -A
kubectl describe node <node-adı>
```

## 7. Test/doğrulama senaryoları

```bash
# Kaçış denemesi (sandbox'tan doğrudan dışarı) — beklenen: DNS seviyesinde engellenir
# (LLM'e "google.com'a bağlanmayı dene" tarzı bir soru sorularak tetiklenir)

# Tool Gateway'in KENDİ egress'ini test etme (pod'un içinden doğrudan)
kubectl exec deploy/tool-gateway -- python3 -c "
import requests
print(requests.get('https://example.com', timeout=5).status_code)
"

# SNI/serverNames doğrulaması (paylaşılan-IP senaryosu)
kubectl exec deploy/tool-gateway -- python3 -c "
import socket, ssl
ctx = ssl.create_default_context()
with socket.create_connection(('console-mia.csp.kloudeks.com', 443), timeout=5) as sock:
    with ctx.wrap_socket(sock, server_hostname='console-mia.csp.kloudeks.com') as ssock:
        print('BAGLANDI (beklenmeyen!)')
"

# Gerçek çalışan sunucu üzerinden bir tool'u seed etmek (demo verisi hazırlama)
# — DİKKAT: düz `python3 -c "from mock_live_system.data import ..."` KULLANMA,
# sunucunun GERÇEK hafızasına yazmaz. Doğru yöntem (bkz. PTC_Live_Demo_Script.md):
kubectl exec deploy/tool-gateway -- python3 -c "
import asyncio
from fastmcp import Client
async def main():
    async with Client('http://localhost:8443/mcp') as client:
        result = await client.call_tool('create_support_ticket', {'title': '...', 'description': '...'})
        print(result.data)
asyncio.run(main())
"
```

## 8. Temizlik / sıfırlama

```bash
# Tool Gateway'i yeniden başlat (in-memory mock veriyi sıfırlar — ör. test ticket'larını temizlemek için)
kubectl rollout restart deploy/tool-gateway
kubectl rollout status deploy/tool-gateway

# Bir politikayı geri almak
kubectl delete -f k8s/policies/<dosya>.yaml

# Cluster'ı tamamen kaldırmak (DİKKAT — geri dönüşü yok)
kind delete cluster
```

## 9. Geliştirme

```bash
ruff check .
pytest
```
