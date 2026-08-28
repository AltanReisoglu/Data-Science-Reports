# Quickstart: PTC Kod Sandbox'ı (Faz 2)

Uygulama kodu değil, uçtan uca doğrulama rehberidir.

## Ön koşullar

- Docker, `kind` (>=v0.7.0), `helm` (>=v3.13.0), `kubectl`, Cilium CLI
- `pip install kubernetes` (Faz 2'nin yeni bağımlılığı, T0xx)
- Faz 1'in `.env`'i (embedding/LLM gateway bilgileri) — Tool Gateway pod'una
  bir Kubernetes Secret olarak aktarılmış olmalı

## Kurulum

```bash
# 1. Yerel cluster
kind create cluster --config=k8s/kind-config.yaml

# 2. Cilium
helm repo add cilium https://helm.cilium.io/
helm install cilium cilium/cilium --version 1.20.1 --namespace kube-system --set ipam.mode=kubernetes
cilium status --wait

# 3. Tool Gateway image'ını kind'a yükle ve deploy et
docker build -t tool-gateway:local mock_services/tool_gateway/
kind load docker-image tool-gateway:local
kubectl apply -f k8s/tool-gateway/

# 4. Sandbox image'ını kind'a yükle (Job'lar bunu kullanacak)
docker build -t ptc-sandbox:local sandbox_image/
kind load docker-image ptc-sandbox:local

# 5. Cilium policy'lerini uygula
kubectl apply -f k8s/policies/
```

## Doğrulama senaryoları

### Senaryo 1 — Çoklu-adımlı orkestrasyon (US1, P1)

```bash
python -m grounded_assistant.cli ask "4 kaynaktaki tüm dokümanları tara, X konusunu geçen kaç tanesi var?"
```

**Beklenen**: Tek bir sandbox Job'u çalışır (birden fazla `search_knowledge_base`
çağrısı Job'un içinde döngüyle yapılır); ana asistan sadece 1 model turu daha
harcar (SC-001 — Faz 1'in adım-adım tool-calling'ine göre en az %50 daha az).

### Senaryo 2 — Onaylı-kanal dışına çıkış engellenir (US2, P2) — **bu fazın asıl testi**

Bilerek kaçış deneyen bir kod ver (`sandbox_test_fixtures/escape_attempt.py`):

```python
import urllib.request
urllib.request.urlopen("https://google.com", timeout=8)
```

(`requests` değil — sandbox image bilerek minimal tutulduğu için (research.md
§4.3) stdlib kullanılıyor; test edilen şey aynı: ağ-seviyesi engelleme.)

```bash
hubble observe --pod ptc-sandbox --verdict DENIED --last 10
```

**Beklenen**: `google.com`'a giden paket `DENIED` olarak görünür; sandbox'ın
kendi çıktısı bir bağlantı hatası/timeout gösterir (gerçek bir yanıt almaz);
`DeniedAction` kaydı `--trace` çıktısında görünür.

### Senaryo 3 — Zaman aşımı/hata zarifçe ele alınır (US3, P3)

```python
while True:
    pass
```

**Beklenen**: `activeDeadlineSeconds` (30s) dolunca Job sonlandırılır, asistan
çökmez, kullanıcıya açık bir "tamamlanamadı" yanıtı döner (exit code 0, CLI
seviyesinde).

### Kontrol: izinli akış gerçekten çalışıyor mu?

```bash
hubble observe --pod ptc-sandbox --verdict FORWARDED --last 10
```

**Beklenen**: Yalnızca Tool Gateway'e giden akışlar `FORWARDED` görünür —
Senaryo 1 çalıştıktan hemen sonra bunu çalıştırıp doğrula.
