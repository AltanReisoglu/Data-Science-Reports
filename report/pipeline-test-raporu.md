# Pipeline Test Raporu — sohbet ajanı, uçtan uca

> **Ne yapıldı:** Sohbet ajanının (`demo-brain-agent/chat_server.py`) **gerçek HTTP uçları**
> üzerinden, kullanıcı gibi sürülerek yapılan sistemli test. 26 senaryo, SSE olay akışı
> toplanıp ölçüldü. Kapsam: sohbet yolu ayrımı, graf kurma (4 backend × 3 paket), Airflow
> DAG export, hata senaryoları, kayıtlı akışı yeniden koşturma, tool-trace compaction,
> çok-turlu oturum.
>
> **Harness:** `demo-brain-agent/test_matrix.py` · **Ham veri:** `demo-brain-agent/test_sonuclari.json`
> **Toplam:** 26 senaryo · 478 sn

---

## Özet — testin asıl çıktısı: 3 gerçek bug

Test "her şey çalışıyor" demek için değil, **kırılan yeri bulmak** için yapıldı. Üç gerçek
kusur çıktı, üçü de düzeltildi ve düzeltme doğrulandı:

| # | Bug | Belirti | Kök neden | Durum |
|---|---|---|---|---|
| **1** | **Çökme enjeksiyonu fonksiyon düğümlerinde hiç çalışmıyordu** | D1/D2/D3 testlerinin üçünde de `çökme=0`, `kurtarma=0` | `run_one_task`'ta `crash_after_turn` yalnız **ajan** dalında kullanılıyordu; fonksiyon dalı parametreyi görmezden geliyordu | ✅ düzeltildi |
| **2** | **Celery dispatch sessiz → SSE 120 sn'de koparıyordu** | `celery × audit` 133 sn sürdü, `summary`/`done` olayı hiç gelmedi | `_dispatch_celery` `on_event` almıyordu; olay yayınlamayan uzun döngü SSE kuyruk zaman aşımını tetikliyordu | ✅ düzeltildi |
| **3** | **Celery dalga-bitti kontrolü yoktu → her dalgada 25 sn boşa bekleme** | `celery × data` 29 sn, ilk düzeltme sonrası 78 sn'ye çıktı | Kuyruğa atılan dalga tükendiğinde dış döngüye dönülmüyor, takılma freni bekleniyordu | ✅ düzeltildi |

**Bir de ölçüm artefaktı:** harness `node_added` olaylarını sayarken planlayıcının "PLAN özeti"
satırını da düğüm sanıyordu → grafiklerde düğüm sayısı 1 fazla, `tamamlanan` eksik görünüyordu
(`7 düğüm / 6 tamamlanan` aslında `6/6`). Harness düzeltildi; aşağıdaki tablolarda **`fn+ajan`
sayısı otorite** alındı.

---

## A · Sohbet yolu ayrımı — 5/5 ✓

Router'ın üç yolu (düz sohbet / tek tool / graf) doğru ayırıp ayırmadığı.

| | Senaryo | İstek | Seçilen yol | Süre |
|---|---|---|---|---:|
| ✓ | A1 | `merhaba, sen kimsin` | **sohbet** (tool yok) | 1,3 sn |
| ✓ | A2 | `neler yapabilirsin` | **sohbet** | 8,2 sn |
| ✓ | A3 | `testleri koştur` | **tek tool** → `run_test_suite` | 1,7 sn |
| ✓ | A4 | `siparişleri çek` | **tek tool** → `extract_records`, **paket `data`'ya otomatik geçti** | 1,1 sn |
| ✓ | A5 | `bana bir React web sitesi yaz ve deploy et` | **sohbet** — yapamayacağını söyledi | 5,0 sn |

**Bulgu:** Yol ayrımı sağlam. A4'te ajan `audit` paketindeyken `data` paketine kendiliğinden
geçti — paket otomatik seçimi çalışıyor. A5'te uydurmadı, sınırını bildirdi.

---

## B · Graf kurma — backend × paket

| | Backend | Paket | Yol | Düğüm | Tamamlanan | Süre |
|---|---|---|---|---:|---:|---:|
| ✓ | own | audit | graf | 4 | **4** | 13,2 sn |
| ✓ | own | data | graf | 6 | **6** | 15,8 sn |
| ⚠ | own | deploy | **tek_tool** | — | — | 5,4 sn |
| ✓ | temporal | audit | graf | 5 | **5** | 13,9 sn |
| ✓ | temporal | data | graf | 6 | **6** | 15,8 sn |
| ✓ | temporal | deploy | graf | 5 | **5** | 13,1 sn |
| ✗→✓ | celery | audit | graf | 4 | 0 → *(bug 2, düzeltildi)* | 133,1 sn |
| ✓ | celery | data | graf | 6 | **6** | 29,2 sn |
| ⚠ | celery | deploy | **tek_tool** | — | — | 6,1 sn |

### ⚠ Bulgu: deploy hedefi graf yerine tool döngüsüne gidiyor (hata değil)

`own × deploy` ve `celery × deploy` senaryolarında router `plan_workflow` yerine **tek tool
yolunu** seçti — ama sonuç **doğru**: 4 turluk tool döngüsünde tüm deploy zincirini koşturdu ve
şunu raporladı:

> *"1.4.2 sürümü için işlemler başlatıldı ancak duman testlerinde 20 testten 1 tanesi başarısız
> olduğu için **canary yayınlama işlemi gerçekleştirilemedi.** Paketleme: Başarılı
> (`app-1.4.2-da5050803b5f.tar.gz`) · Duman Testi: Başarısız (19/20)"*

Yani **koşullu kapı doğru işledi** (duman testi kalınca canary yayınlamadı). Ama:

**Bedeli:** tool döngüsü yolunda **kayıtlı pipeline üretilmiyor** — graf saklanmıyor, "Akışlar"da
görünmüyor, yeniden koşturulamıyor. Tekrar kullanılabilirlik isteniyorsa graf yolu şart.

**Neden böyle oluyor:** deploy zinciri 4 adım ve sohbet döngüsü tam 4 tura kadar tool
çağırabiliyor — model "graf kurmaya gerek yok, elimle yaparım" diyor. Router prompt'unda
"çok adımlı → plan_workflow" kuralı var ama model bunu bağlayıcı görmüyor.

---

## C · Airflow — yapısal uyumsuzluk doğrulandı

| | Senaryo | Sonuç | Süre |
|---|---|---|---:|
| ✓ | airflow × audit | Graf kuruldu (7 düğüm), **koşturulmadı**, gerçek DAG dosyası üretildi | 16,4 sn |

Beklendiği gibi: Airflow düğümleri çalıştırmadı, `brain_agent_plan.py` DAG dosyası üretti
(`PythonOperator` + `>>` bağımlılıkları + XCom veri akışı). Ajanın planladığı graf **teknik
olarak** Airflow'a çevrilebiliyor ama **donmuş** hale geliyor.

---

## D · Hata senaryoları — bug 1'in ortaya çıktığı yer

| | Senaryo | Beklenen | Ölçülen (düzeltme ÖNCESİ) |
|---|---|---|---|
| ✗ | D1 · çökme @ `tara` | çökme 1 → kurtarma 1 | **çökme 0** |
| ✗ | D2 · çökme @ `test` | çökme 1 → kurtarma 1 | **çökme 0** |
| ✗ | D3 · çökme + temporal | çökme sinyali | **çökme 0** |

**Kök neden (bug 1):** `run_one_task()` fonksiyon dalında `crash_after_turn` parametresi
**hiç kullanılmıyordu**:

```python
if task.get("kind") == "function":
    out = F.call(task["fn"], task.get("fn_args") or {}, up)
    return "function", ...      # crash_after_turn burada YOK
```

Fonksiyon-öncelikli mimariye geçtiğimizde (düğümlerin çoğu artık `kind="function"`),
**çökme/kurtarma yolu fiilen ulaşılamaz** hale gelmişti — yani task-management'ın en değerli
özelliği sessizce test edilemez olmuştu.

### Düzeltme sonrası — doğrulandı ✓

```
ÇÖKME: scan_patterns() sonrası worker öldü (claim açık kaldı → stale)
recover_stale → 1 task 'ready'; checkpoint DURUYOR → devralan worker kaldığı yerden sürer
"crashes": 1, "recovered": 1
```

Artık fonksiyon düğümü de: iş yapılır → sonuç checkpoint'e yazılır → `complete()` çağrılmadan
worker ölür → `recover_stale` toparlar → başka worker devralır.

---

## E · Kayıtlı akışı yeniden koşturma — asıl "pipeline" davranışı

| | Backend | Düğüm | Süre | Not |
|---|---|---:|---:|---|
| ✓ | own | 6 | **0,1 sn** | planlama yok |
| ✓ | temporal | 6 | **0,3 sn** | planlama yok |
| ✗→✓ | celery | 6 | 120,0 sn → *(bug 2/3, düzeltildi)* | SSE koptu |

**En çarpıcı ölçüm:** aynı akış
- **ilk kurulum:** 15,8 sn (LLM planlaması dahil)
- **yeniden koşu:** **0,1 sn** (planlama yok)

**≈150× hızlanma.** Airflow'da bir DAG'ın yeni `DagRun`'ı gibi: graf sabit, koşu yeni. Pipeline
kavramının değeri burada somutlaşıyor.

---

## F · Tool-trace compaction — 4/4 ✓

`fetch_docs` aracıyla ham doküman çekip (15.070 token) farklı stratejilerle sıkıştırma:

| | Strateji | Compaction olayı | Süre |
|---|---|---:|---:|
| ✓ | none | 1 (tetiklenmedi kaydı) | 10,1 sn |
| ✓ | hermes | 1 | 6,1 sn |
| ✓ | codex | 1 | 2,8 sn |
| ✓ | openclaw | 1 | 6,3 sn |

Dördünde de araç tetiklendi, ham çıktı context'e girdi, seçilen strateji uygulandı ve panele
düştü. `none` dahil hepsi kayıt üretti (tetiklenmeyen de görünüyor — bu testin bir önceki
turda eklenen davranışı).

---

## G · Çok turlu oturum — board birikimi ✓

Aynı oturumda üç ardışık mesaj; her turda board büyüdü ve önceki düğümlerin çıktısı yenilere
aktı. Oturum durumu kalıcılığı çalışıyor.

---

## Backend karşılaştırması — ölçülen

| Eksen | own | temporal | celery | airflow |
|---|---|---|---|---|
| **Graf kurma** | ✓ 2/3 (deploy tool'a gitti) | ✓ **3/3** | ✓ 2/3 | ✓ (koşturmaz) |
| **En hızlı graf** | 13,2 sn | 13,1 sn | 29,2 sn | 16,4 sn (export) |
| **Yeniden koşu** | **0,1 sn** | 0,3 sn | ~30 sn | — |
| **Çökme/kurtarma** | ✓ (düzeltme sonrası) | replay ile | mesaj yeniden teslim | düğüm retry |
| **Kurulum yükü** | yok | dev server (~5 sn) | worker süreci (~6 sn) | çalışmıyor |
| **Gözlenebilirlik** | tam (olay günlüğü) | 95 durable event | zayıf (board olmasa yok) | DAG dosyası |

**Temporal bu testin en tutarlı backend'i çıktı** — 3/3 paket, en hızlı graf (13,1 sn), sorunsuz.
**own** en hızlı yeniden koşuma ve en düşük operasyon yüküne sahip.
**celery** en yavaş ve en kırılgan (iki bug da orada çıktı) — worker süreci + broker gecikmesi.

---

## Kalan açıklar (dürüstlük notu)

1. **Router bazen graf yerine tool döngüsü seçiyor** (deploy senaryosu). Sonuç doğru ama
   pipeline saklanmıyor. Prompt sıkılaştırılabilir ya da "4+ adım → zorunlu graf" kuralı konabilir.
2. **Concurrency hâlâ kanıtlanmadı** — `own` dispatch tek süreçte sıralı çalışıyor; gerçek
   paralel worker yarışı test edilmedi.
3. **Scheduling hiç test edilmedi** — cron/backfill çalışan kanıtı yok (koçun 6 ekseninden biri).
4. **Ajan düğümü (kind=agent) grafta neredeyse hiç üretilmiyor** — planlayıcı hep deterministik
   fonksiyon seçiyor. Doğru davranış ama compaction'ın graf içindeki yolu az test edilmiş oluyor.
5. **Celery'nin dalga-dalga ilerlemesi** düzeltildi ama hâlâ diğerlerinden ~2× yavaş.

---

## Sonuç

26 senaryoluk test **3 gerçek bug** ortaya çıkardı ve üçü de düzeltilip doğrulandı. En kritiği,
fonksiyon-öncelikli mimariye geçişte **çökme/kurtarma yolunun sessizce ulaşılamaz hale gelmesiydi**
— mimari değişikliğin görünmez yan etkisi. Bu, "çalışıyor gibi görünen ama test edilmeyen yol"
sınıfının tipik örneği.

Sistem şu an: **sohbet yolu ayrımı sağlam · graf kurma 4 backend'de çalışıyor · kayıtlı akış
150× hızlı yeniden koşuyor · çökme/kurtarma fonksiyon düğümlerinde de doğrulandı.**

**Çalıştırma:**
```bash
.venv/bin/python demo-brain-agent/chat_server.py          # sunucu (8030)
.venv/bin/python demo-brain-agent/test_matrix.py          # tüm matris
.venv/bin/python demo-brain-agent/test_matrix.py hizli    # sadece own
.venv/bin/python demo-brain-agent/rapor_uret.py           # ham tablolar
```
