# PTC Artifact Persistence — Sunum

**13 sayfa. Anlatılacak tek hikâye var: sandbox ölür, ürettiği kalır — ve
bunu yaparken hiçbir şey icat etmedik.**

Tarih: 2026-09-07 · Slaytlar: `node scripts/sunum_uret.js` · Diyagramlar: `python scripts/diyagram_uret.py`

> Bu dosya **sunum**; arşiv değil. Bütün karşılaştırmalar, alıntılar ve
> ölçümler [PTC_Piyasa_Mentaliteleri.md](PTC_Piyasa_Mentaliteleri.md) ve
> [PTC_Karsilastirma_Tablolari.md](PTC_Karsilastirma_Tablolari.md) içinde.
> Soru gelirse oradan açılır.

---

## Sayfa 1 — Problem

> Sandbox'ın **ölmesi** güvenlik için gerekli.
> Ürettiğinin **kalması** iş için gerekli.

| | Ne olur | Sonuç |
|---|---|---|
| Sandbox yaşarsa | Bir çalıştırma diğerine bulaşır | İzolasyon yok |
| Artifact ölürse | 40 sn'lik iş her turda tekrarlanır | Kullanılamaz |

**Kural:** Sandbox'ın yaşam süresi, ürettiğinin yaşam süresini belirlememeli.

**Karıştırılmaması gereken üçlü:** geçici dosya (ölmesi *istenir*) · artifact
(kalıcı depo) · state (ayrı, o da kalıcı).

---

## Sayfa 2 — Bir adım nasıl çalışıyor

<!-- diyagram: d1-yasam-dongusu -->

**Kodun gördüğü tek şey bir dosya yolu.** Artifact fonksiyonu yok, depo adresi
yok, anahtar yok — ve **hiçbir ağ çağrısı yok.** Ölçüldü:

| sandbox'ın içinden | |
|---|---|
| artifact fonksiyonları | `[]` |
| servis adresi | yok |
| localhost proxy | yok |
| `minio` | `gaierror` — DNS'te bile yok |

KFP'nin kullanıcı bileşenine verdiği garantinin aynısı.

---

## Sayfa 3 — Baytı kim taşıyor: üç yerleşim

<!-- diyagram: d2-aracinin-yeri -->

**Seçimi zorlayan tek şey:** kodu LLM yazıyor. KFP'nin launcher'ı kullanıcı
container'ının içinde — anahtar orada dururdu. Argo'nun yerleşimi doğru ama
kayıt defteri yok. Biz **yerleşimi Argo'dan, kanalı MLflow'dan** aldık.

---

## Sayfa 4 — Baytlar hangi yoldan: iki kip

| | Baytlar | S3 anahtarı | Kayıt defteri |
|---|---|---|---|
| **KFP / OpenShift AI** | launcher → S3 doğrudan | **kullanıcı container'ında** | MLMD |
| **Argo Workflows** | wait sidecar → S3 doğrudan | ayrı container | **yok** |
| **MLflow** (proxied) | client → **HTTP** → server | **server'da** | tracking DB |
| **BİZ · `proxy`** | sidecar → HTTP → servis | serviste | SQLite |
| **BİZ · `direct`** | sidecar → S3 doğrudan | **sidecar'da** | SQLite |

`PTC_ARTIFACT_TRANSFER` ile seçiliyor. `direct` **KFP'nin iki kanalı**:
bayt depoya, künye kayıt defterine.

**`direct`'in ölçülmüş bedeli:** NetworkPolicy **pod** seçer, container değil.
Sidecar'a depo rotası açmak sandbox'a da açmaktır.

| | `proxy` | `direct` |
|---|---|---|
| `sandbox → minio` (IP ile) | **TimeoutError** | **ULASILDI** |
| sandbox'ta S3 anahtarı | yok | yok |

**Hangisi ne zaman:** < ~50 MB `proxy` (kontroller tek yerde) · > ~50 MB
`direct` (GB'ları tek bir servisten akıtmak israf). MLflow'un sunduğu seçimin
aynısı — `--serve-artifacts` var/yok.

---

## Sayfa 5 — Çapraz workflow: vakanın kendisi

<!-- diyagram: d3-capraz-workflow -->

**B, A'nın çalıştırma kimliğini kayıt defterinden öğreniyor.** Aralarında
doğrudan bağ yok; A çoktan bitmiş, pod'u silinmiş olabilir.

---

## Sayfa 6 — Ne yazılı, ne çalışma anında bulunuyor

| | Nereden geliyor |
|---|---|
| Ne aranacak (`processed-result.json`) | **yazılı** — hat tanımında, Argo/KFP gibi |
| Hangi çalıştırma üretmiş | çalışma anında, kayıt defteri sorgusundan |
| Hangi sürüm (8 aday arasından) | çalışma anında, alias/en-yeni kuralıyla |

```
inputs=["{kaynak_wf}/processed-result.json"]
         ^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^
         bulunuyor     yazılı
```

Bu bir **çağrı değil, beyan**: kod `open("/artifacts/<wf>/…")` yazıyor, dosya
zaten orada. Sandbox hiçbir yere bağlanmıyor.

**Adı da modelin seçtiği yol var:** Sohbet sekmesi. Orada hiçbir şey yazılı
değil — manifest isimleri veriyor, neyi okuyacağına model karar veriyor.

---

## Sayfa 7 — Keşif: ajan çekeceğini nasıl anlıyor

| Desen | Nasıl | Kim |
|---|---|---|
| Sadece tool tarifi | Model çağırmayı *seçmek* zorunda | *(eskiden biz)* |
| Dosya sistemi + `ls` | Sandbox yaşıyorsa model bakar | Anthropic, Google |
| **İsimler prompt'a enjekte** | İsimler talimatlarda, içerik talep üzerine | **Google ADK**, **BİZ** |
| Kayıt defterine sorgu | `filter_query` ile süzülmüş liste | **MLMD**, **BİZ** |
| **Keşif YOK** | DAG statik, girdi bağlanmış | KFP, Argo, Airflow, Tekton |

**Klasik pipeline'da 5. adım bir şey anlamaz — kendisine söylenir. Ajan
dünyasında sormak zorundadır, çünkü kendisi de o an icat edilmiştir.**

---

## Sayfa 8 — Beyan · Süzgeç · Alias

| Açık | Kaynak | Bizdeki hâli |
|---|---|---|
| Hangi girdi okundu (soy şişiyordu) | **MLMD** `DECLARED_INPUT` | `run_ptc_code(kod, inputs=[…])` |
| 62 addan 39'u görünüyor, arama yok | **MLMD** `filter_query` | `?name= ?type= ?q=` |
| Aynı ad 17 kez, hep en yeni geliyor | **MLflow** `models:/<ad>@<alias>` | `by-name/rapor.pdf@onaylanmis` |

**Beyanın üç biçimi** — üçü de KFP'de `.uri` beyanına denk:

```
inputs=["ozet.json"]             → /output/ozet.json
inputs=["wf-abc/ozet.json"]      → /artifacts/wf-abc/ozet.json
inputs=["rapor.pdf@onaylanmis"]  → /artifacts/_alias/rapor.pdf
```

**Sürüm seçimi canlı doğrulandı:**

| | Seçilen | Üreten çalıştırma |
|---|---|---|
| alias yokken | `art_dd422fd3e66b` — **en yeni** | 119d1f91 |
| `@onaylanmis` sabitliyken | `art_cfa44f0298f5` — **en eski** | bfc62bbe |

---

## Sayfa 9 — Dört icat ettik, dördünü de attık

| İcat | Ne oldu | Yerine geçen |
|---|---|---|
| LLM'e artifact API'si (5 fonksiyon) | sessiz `None`'lar, anlaşılmaz hatalar | düz Python + `/output` — **KFP** |
| Şeffaf tembel okuma (~120 satır yama) | `/output` yalan söylüyordu | kod başlamadan yerleştir — **Argo/KFP** |
| `atime` ile soy ölçümü | çalışıyordu, ama emsali yok | beyan — **MLMD** |
| sandbox'ta `load_artifact` | çalışma anında ağ çağrısı | çapraz girdi de **beyan** — KFP |

**Hataların hepsi bizim icat ettiğimiz yerlerde çıktı; kopyaladığımız hiçbir
parçadan çıkmadı.**

| | Öncesi | Sonrası |
|---|---|---|
| `entrypoint.py` | 651 satır | **298 satır** |
| Yama satırı | ~120 | **0** |
| Sandbox'ın ağ çağrısı | 1 (localhost proxy) | **0** |
| Okuyan çalıştırma | 4,11 sn | **3,13 sn** |

---

## Sayfa 10 — Hata kurtarma: sinyali zenginleştirdik

**Sandbox patlayınca modele giden metin. Önce ve sonra:**

```
ÖNCE   Hata: 'yok'          ← tip yok, satır yok, stdout yok
       Tahmini bir değer üretme.        ← modele DUR diyor

SONRA  File "/sandbox/code.py", line 6, in ic
           def ic(): return d["yok"]
                            ~^^^^^^^
       KeyError: 'yok'                  ← tip + satır + ifade
       Hata anına kadar yazılan çıktı:
       adim 1: veri yuklendi            ← print'ler korunuyor
       Hatayı düzeltip TEKRAR çalıştır. Aynı kodu aynen gönderme…
```

| Parça | Kimden kopya |
|---|---|
| Yalnızca **kullanıcı kodunun** kareleri | **SWE-agent** — "tip olmadan model yanlış teşhis koyuyor" |
| Hata anına kadarki **stdout** | **smolagents** — bunun için ayrı testi var |
| **20 000 karakterde ortadan** kırpma | **smolagents · Codex** |
| Kırpıldığını **söyle** + ne yapacağını **öğret** | **SWE-agent** |
| "tekrar dene" **ve** "aynısını tekrarlama" | **smolagents + OpenHands** |
| **Sebep-farkında** sayaç | **Anthropic `error_code` · Codex `is_likely_sandbox_denied`** |

**Sayaç neden bölündü:** tek sayaç kod hatasını, ağ engelini ve timeout'u aynı
kutuya koyuyordu. Sınır ağ engeli için konmuştu; kod hatası da aynı bütçeden
yediği için self-repair'e **1 deneme** kalıyordu. Artık ağ engeli **2**, kod
hatası **5**.

> **Bütçeyi büyütmek tek başına çözüm değil.** Olausson (ICLR 2024):
> self-repair kazancı *"mütevazı, bazen hiç yok"*; darboğaz deneme sayısı
> değil **geri bildirim kalitesi** — aynı koda insan geri bildirimi verilince
> başarı **%33 → %52**. Sıra: önce sinyal, en son bütçe.

## Sayfa 11 — Biz neredeyiz

| Parça | Kimden |
|---|---|
| init + wait sidecar | **Argo Workflows** |
| girdiyi kod başlamadan yerleştir | **Argo `init` / KFP `driver`+`launcher`** |
| girdi beyanı · beyandan soy | **Argo `inputs.artifacts` · MLMD `DECLARED_INPUT`** |
| run-scoped anahtar yolu · kayıt defteri | **KFP `pipeline_root` · MLMD** |
| künye süzgeci · sürüm alias'ı | **MLMD `filter_query` · MLflow Model Registry** |
| isimler prompt'ta | **Google ADK `LoadArtifactsTool`** |
| bayt yolu · `proxy` | **MLflow** proxied artifact access |
| bayt yolu · `direct` | **KFP / Argo** — bayt depoya, künye kayıt defterine |
| **sandbox'ta sıfır ağ çağrısı** | **KFP** — kullanıcı bileşeni hiçbir şey çağırmaz |
| hata sinyali · kırpma · "ne yapmalı" | **SWE-agent · smolagents · Codex** |
| sebep-farkında retry sayacı | **Anthropic `error_code` · Codex `is_likely_sandbox_denied`** |
| **İzolasyon** | **Kimse — bizimki daha zayıf (düz container)** |

**Emsalsiz desen kalmadı.**

---

## Sayfa 12 — Canlı konsol + açıklar

**`/konsol` — beş sekme, sahte veri yok:** Sohbet · Hatlar · Çalıştırma
(gerçek Kubernetes Job'ları) · Depo · Soy ağacı.

| Açık | Durum |
|---|---|
| **İzolasyon** | Düz container, Kata yok — Red Hat'in önerisine uymuyoruz |
| **Ağ politikaları** | Cilium'a bağımlı; OVN karşılığı yazılmadı |
| **Auth** | Yok — jetonu üretebilen tenant'ın tamamını okur |
| **Metadata DB** | SQLite, tek replika |
| `SIGKILL` | OOM/deadline'da süpürme çalışmaz (Argo'da da aynı) |
| Soy imzasız | Tekton Chains bunu çözüyor, bizde yok |

**209 test · 52/52 (`proxy`) + 53/53 (`direct`) canlı kabul kontrolü · bütün ölçümler cluster'dan.**

---

## Sayfa 13 — Ekibe dört soru

| # | Soru | Neden önemli |
|---|---|---|
| 1 | **Kurumda S3-uyumlu depo var mı?** | ODF yok → S3'ü biz getireceğiz |
| 2 | **RWX destekleyen StorageClass var mı?** | Tekton'un PVC deseni mümkün mü |
| 3 | **Sandboxed Containers (Kata) kurulu mu?** | En büyük açığımız; kod değişikliği gerektirmez |
| 4 | **PostgreSQL sağlanabilir mi?** | Hem artifact metadata hem workflow state |

**1. soru en kritik** — varsa endpoint + bucket + anahtar yeter, **kod hazır**
(hem OBC hem OpenShift AI connection sözleşmesini okuyor).
