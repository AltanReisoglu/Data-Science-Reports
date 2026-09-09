# PTC Sunum Kartı

Cebe girecek hâli. Ayrıntı için `PTC_Sunum_Rehberi.md`; konuşma metni orada §8.

## Sunumdan önce

```
kubectl port-forward svc/artifact-service 8080:8080 &
PYTHONPATH=src ../ptc_sec/.venv/bin/uvicorn grounded_assistant.web.app:app --port 8123 &
cilium hubble port-forward &
kubectl port-forward -n kube-system svc/hubble-ui 12000:80 &

kubectl get pods                 → 3 servis Running
curl -s -o /dev/null -w "%{http_code}\n" localhost:8123/konsol   → 200
hubble status | grep Healthcheck → Ok
```

**Hatlar → "Ticket İşleme Hattı" (PL-A) bir kez çalıştır** — depo dolsun, PL-B'nin kaynağı oluşsun.

Ekranda: slayt · `localhost:8123/konsol` · `localhost:12000` · terminal (`hubble observe --label app=ptc-sandbox -f`)

Slayt: `→ ←` sayfa · `o` genel bakış · `Esc` kapat

## 11 sayfa — ne söyleyeceksin, nereye bağlayacaksın

| # | dk | Sayfada söylenecek tek şey | Geçiş cümlesi |
|---|---|---|---|
| 1 | 2 | Sandbox ölecek, ürettiği kalacak. Ama **kodu LLM yazıyor** — güvenilmez. | "Bu problemi ilk biz görmedik." |
| **2** | **4** | On ürün, **beş soru**. Defter sütunu boş olanların hepsi **mount ediyor**. | "Neden? Cevap baytı kimin taşıdığında." |
| **3** | **4** | **A** mount: araya girecek yer yok · **B** sarmalayıcı: atlanabilir · **C** sınır: kod erişemez · **D** sürücü izin soruyor. Argo v3.4'te dördünü kaldırdı: *breaks security completely.* | "Bu aileleri gerçek ürünlerde görelim." |
| 4 | 3 | Üstteki ikisinde platform araya giriyor, alttaki ikisinde girmiyor. 30 gün / 20 dk — ikisi de sandbox'ı **yaşatarak** çözüyor. | "Pipeline tarafına, bize en yakın olana." |
| **5** | **4** | Driver çözer, launcher taşır. Ama launcher **aynı container'da**, kod **alt süreç** → ortamı devralır. **KFP için doğru karar** — orada kodu insan yazdı. | "Kalan dört desen." |
| 6 | 2 | Argo = bizim yerleşim (`argoexec` ayrı imaj). Devin soruyu ortadan kaldırıyor. | "Taşımayı konuştuk. Bir de **bulmak** var." |
| 7 | 2 | Sandbox yaşıyorsa `ls` yeter. **Bizde ajan karar veriyor VE sandbox ölüyor** — ikisi bir arada. | "Piyasa bitti, şimdi biz." |
| 8 | 3 | Çağrı değil **beyan**. 100 dosya: beyansız **100 ebeveyn**, beyanlı **2**. Asıl bedel soy şişmesi. | "Peki bu nerede çalışıyor?" |
| **9** | **4** | İki container, tek `/output`. **Ağ çağrısı: 0.** `restricted-v2` SCC hiçbir şeyi kırmıyor. | "Bir eksen daha: kod patlayınca." |
| 10 | 2 | İyi sinyalin üçlüsü: çıkış kodu · tam traceback · **korunan stdout**. | "İddia bırakmadık, ölçtük." |
| **11** | **3** | Baştan kırpma hatayı yiyor (**23→26**) · kırpmayanlar **211 KB** · talimat sütunu 9/10 boş. **Dairesellik var** — sinyal/bayt'a bakın. | — |

★ 2 · 3 · 5 · 9 · 11 dokunulmaz. 30 dk'ya sığdırman gerekirse 6 ve 7'den kıs.

## Sayfa 2 — dört beat

```
① 30sn  "On ürün. Hepsi farklı problem çözdüğünü sanıyor, hepsi aynı beş soruyu cevaplamış."
② 45sn  NEREDE · ÖMÜR · VERİ · ANAHTAR · DEFTER      (S1+S3 "tezi" sütununda)
③ 90sn  DÖRT satır:  Anthropic (defter var, SOY YOK)
                     Cloudflare (hücrede "olamaz" yazıyor — bir saniye dur)
                     Red Hat (en zengin defter + anahtar EVET → 5. sayfaya borç)
                     Databricks (mount ama defter var: aracı sürücünün içinde)
④ 45sn  Defter sütununu gez → "boş olanların hepsi mount ediyor"
        "Kayıt defteri yalnızca yazma yolu bir bileşenden geçtiğinde ayakta kalıyor."
```

**"Argo'da bileşen var ama defter yok"** → *"Tez tek yöne. Mount ederseniz tutamazsınız. Bileşen koyarsanız tutabilirsiniz — Argo tutmamayı seçmiş, çünkü yol run-id içeriyor. Ön koşul, garanti değil."*

## Demo — 9. sayfadan sonra, 6 dk

```
① PL-B "Artifact Analiz Hattı"   1. adım pod AÇMAZ · 2. adım BEYAN, kodda çağrı yok
                                  Soy ağacı: ok başka workflow'a gidiyor
② hubble observe -f + PL-A        sandbox → tool-gateway, başka hiçbir şey
③ sohbet: "https://api.example.com/tickets adresinden veri çek"
                                  ConnectionError → zengin hata metni → sınır 5 değil 2
```

Vakit kalırsa: `bak rapor.json` (sürüm 3) vs `bak rapor.json@onaylanmis` (sürüm 1)

**Gösterme:** hat kurucusu · silme butonları · konsol turu

**Çökerse:** *"Ortam sorunu — ölçümler zaten slaytta, hepsi bu cluster'dan."*

## Soru–cevap

| Soru | Cevap |
|---|---|
| **Files API ne?** | Kod `$OUTPUT_DIR`'a yazar, **platform dizine bakar**, fiş döner. Fiş var, **soy yok**. Container 30 gün, dosya silinene kadar. |
| **Bucket / S3 / MinIO?** | tablo ≈ bucket (kavram) · SQL ≈ S3 API (protokol) · PostgreSQL ≈ MinIO (ürün) |
| **Neden MinIO?** | **OpenShift'te nesne deposu yok.** Red Hat'in kendi AI ürünü bile "S3-compatible bucket" şart koşuyor. Geçiş bir endpoint değişikliği. |
| **Anahtar koda nasıl görünüyor?** | Koda değil **pod'a** veriliyor; kod alt süreç, ortamı devralıyor. Temizlense bile `/proc/1/environ`. Aynı container'da sır saklanamaz. |
| **`artifact_ara` neden?** | Beyan yazılırken sandbox yok. Manifest kesik: 102 kayıttan 41'i görünüyor. Canlı: model 4 gördü, arama **24** buldu. |
| **Hubble kanıtlıyor mu?** | **Hayır.** Politika **pod** düzeyinde. Sandbox'ın erişememesi ağdan değil ortamdan: anahtar yok, SDK yok, adres yok. |
| **Kendinizi birinci yapmışsınız** | Dairesellik var, ölçütleri biz seçtik. **Sinyal/bayt**'a bakın: orada Codex önde. Eksiğimiz tabloda: tekrar tespiti. |
| **Kaç test?** | 242 test · kabul testi 52/52 · kıyasın 17 testi bulguları sabitliyor. |

## Söyleme

```
✗ "bizimki Files API'nin aynısı"     soy, alias, arama yok
✗ "Hubble sandbox'ı hapsediyor"       pod düzeyi
✗ "KFP güvensiz"                      kendi bağlamında doğru karar
✗ "ODF'de de test ettik"              TEST EDİLMEDİ, kapsam dışıydı
✗ "MinIO OpenShift'in çözümü"         PoC'nin vekili
✗ "en iyi biçim bizimki"              bu ölçütlerde; düzeltme başarısı ÖLÇÜLMEDİ
```

MinIO lisansı (AGPLv3) sorulursa: *"kurumsal kullanımda hukuk tarafına sorulmalı, bu projede doğrulamadım."*

## Kapanış

> "Piyasada dört yerleşim var ve hepsi kodun güvenilir olduğunu varsayıyor. Bizim kodumuzu LLM yazıyor. O yüzden taşıyıcıyı kodun yanından çıkarıp ayrı bir container'a koyduk — Argo'nun yaptığı gibi. Üstüne pipeline dünyasından üç şey aldık: **beyan**, **soy**, **alias**. Hepsi OpenShift'te varsayılanlarla çalışıyor."
