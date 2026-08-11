# Motor Karşılaştırma Paneli — Web POC

**Nerede:** 8030'daki sohbet arayüzü → üst barda **Motorlar**
**Kod:** `demo-brain-agent/chat_server.py` (panel + uçlar) · `airflow_runner.py` (yeni) · `orchestrator.py` (`node_sim`)
**Test:** `test_node_sim.py` → **27/27**

Şimdiye kadar motor karşılaştırmasını komut satırından yapıyorduk: her ölçüm ayrı bir
terminal komutu, sonuçlar ayrı dosyalarda. Bu panel onu tek ekrana indiriyor —
**bir kez workflow üret, düğüm düğüm ne olacağını seç, dört motorda koştur, yan yana gör.**

---

## Akış

**Akış detay ekranının kendisi = motor karşılaştırma ekranı.** Ayrı bir panel yok.

```
Sohbette workflow oluştur  →  Akışlar listesinde en üstte belirir
        ↓  akışa tıkla
┌─ akış detayı = karşılaştırma ekranı ─────────────────────────┐
│  [motor ▾] [▶ Koştur] [⚡ Hepsinde koştur]  [Sıfırla (2)]    │
│                                                               │
│  graf — her düğüm TIKLANABİLİR                                │
│  └ tıkla → argümanları düzenle + "bu düğüme ne olsun?" seç    │
│                                                               │
│  canlı log + karşılaştırma tablosu                            │
└───────────────────────────────────────────────────────────────┘
```

Aynı graf dört motorda da **aynen** koşar — bu yüzden akışı bir kez üretmek yetiyor.

**İki giriş noktası, aynı ekrana çıkıyor:**
- **Akışlar** → bir akışa tıkla
- **Motorlar** → akış kartlarından seç (aynı liste, kısayol)

## Sohbetten dört motorda koşturma

Üst bardaki **karşılaştır** seçicisi `4 motorda koştur`a alınırsa, sohbette bir
workflow istediğinde şu olur:

```
graf BİR KEZ kurulur → Akışlar'a kaydedilir
   ↓
own       ✓  4/4 düğüm · 0,02 sn     ┐
temporal  ✓  4/4 düğüm · 0,45 sn     │ her birinin ÇIKTISI ayrı
celery    ✓  4/4 düğüm · 39,3 sn     │ katlanır blok olarak basılır
airflow   ✓  4/4 düğüm · 1,50 sn     ┘
   ↓
karşılaştırma tablosu + "TÜM MOTORLAR AYNI ÇIKTIYI ÜRETTİ ✓"
```

**Ölçüldü:** dört motorun ürettiği metin **birebir aynı**. Sayılar aynı olabilir ama
asıl kanıt bu — aynı graf, aynı sonuç, farklı yürütme motoru.

Airflow'un çıktısı board'dan değil **kendi XCom'undan** çıkarılıyor (`_airflow_cikti`),
çünkü Airflow board'a yazmıyor. Aynı metni üretmesi, veri akışının o tarafta da
doğru kurulduğunun kanıtı.

**Bedeli açık:** Celery tek başına ~12-40 sn ekliyor. Bu yüzden kip **varsayılan kapalı**;
kapalıyken eski davranış (tek motor) aynen korunuyor — doğrulandı.

### "Motorlar"daki üretici neden var

İkincil bir yol olarak duruyor: sohbet yolu LLM router'ından geçiyor ve `plan_workflow`
yerine tool-loop seçebiliyor (bu oturumda gözlendi). Oradaki üretici `plan_phase()`'i
**doğrudan** çağırıyor → graf garantisi.

---

## Çekirdek: düğüm bazında simülasyon

Bu panelin asıl değeri burada. Önceden "şu düğüm patlasın" demenin yolu yoktu:
`fail_at` tek bir **global string**di ve **fonksiyon adı ya da başlık alt-dizesiyle**
eşleşiyordu. Aynı `fn` iki düğümde kullanılırsa **ikisi birden** patlıyordu.

Yeni taşıma sözlüğü id bazlı:

```python
node_sim = {"t_a1b2c3": {"mod": "kalici"}, "t_d4e5f6": {"mod": "yavas", "sn": 5}}
```

| mod | ne yapar |
|---|---|
| `normal` | değişiklik yok |
| `gecici` | ilk denemede patlar, retry'da geçer |
| `kalici` | her denemede patlar → breaker → `failed` + ardıllar `cancelled` |
| `sonra` | iş **yapıldıktan sonra** patlar (yan etki oluşur, sonuç yazılmaz) |
| `cokme` | iş biter, checkpoint yazılır, `complete` çağrılmadan worker ölür |
| `yavas` | N saniye bekler (lease/timeout gözlemi) |

**Ölçülen fark** — aynı akışta iki `fetch_source` düğümü, hedef ikincisi:

| | çek A (n1) | çek B (n2) |
|---|---|---|
| eski `fail_at="fetch_source!"` | **failed** | **failed** ← ikisi birden |
| yeni `node_sim={"n2":…}` | `done` | **failed** ← yalnız hedef |

### Taşıma kanalları

Her motor ayrı bir yürütme ortamı olduğu için ayrı kanal gerekti:

| motor | kanal |
|---|---|
| own | doğrudan parametre |
| celery | `BRAIN_NODE_SIM` env (JSON) — worker ayrı süreç |
| temporal | `CTX["node_sim"]` — activity aynı süreçte |
| airflow | **DAG dosyasına gömülüyor** — ayrı süreç, ortak bellek yok |

`materialize()` kayıtlı akış id'lerini board'un yeni id'lerine çeviriyor
(`cevir_node_sim`). Çeviri olmasa ayar hiçbir düğüme denk gelmez ve **sessizce
hiçbir şey olmazdı** — en kolay gözden kaçacak hata buydu.

---

## Airflow artık gerçekten koşuyor

Önceki raporlarda "bizim katmanda yürütmüyor, ölçülmedi" diye işaretliydi. `airflow_runner.py`:

```
board → DAG dosyası üret → `airflow dags test <dag_id>` → sonucu airflow.db'den oku
```

Uygulanan üç ölçülmüş kısıt:

- **Tarih argümanı verilmiyor.** Verilmezse Airflow `utcnow()` kullanıp her tetiklemeye
  benzersiz `run_id` veriyor; sabit tarih verilirse aynı run çakışıyor.
- **Koşular kilitle serileştiriliyor.** `SequentialExecutor` + sqlite → eşzamanlı iki
  `dags test` "database is locked" veriyor.
- **`retry_delay=1s`.** Varsayılan 30 sn paneli kullanılamaz kılıyordu (kalıcı hata
  senaryosu 62 sn sürüyordu). İhracat varsayılanı değişmedi, parametreye çıkarıldı.

Sonuçlar `airflow.db`'den **read-only** okunuyor: `dag_run.state`, `task_instance`
(`state`/`try_number`), `xcom`. Ölçülen süre: hatasız ~1,5 sn, kalıcı hatalı ~3,6 sn.

---

## Ölçüm: aynı akış, aynı simülasyon, dört motor

6 düğümlü ETL akışı, `validate_schema` düğümü **kalıcı** patlatıldı:

| motor | süre | tamamlanan | başarısız | iptal | deneme |
|---|---:|---:|---:|---:|---:|
| own | **0,01 s** | 3/6 | 1 | 2 | 3 |
| temporal | 0,50 s | 3/6 | 1 | 2 | 3 |
| airflow | 4,75 s | 3/6 | 1 | 2 | 3 |
| celery | 12,39 s | 3/6 | 1 | 2 | 3 |

**Dört motor da her sütunda birebir aynı.** Ayrıştıkları tek yer süre — çünkü karar
board'da tek noktada veriliyor. Panelin ekibe göstereceği asıl şey bu tablo.

---

## Yol boyunca kapanan üç eksik

**1. Çökme artık dört motorda da çalışıyor.** Eski `crash_at` yalnız `own`'da uygulanmıştı;
celery ve temporal parametreyi kabul edip **sessizce yok sayıyordu** (raporlarda
"ölçülmedi" diye işaretliydi). `node_sim` `cokme` modu üçünde de aynı yoldan geçiyor.

**2. Çökme sonsuz döngüsü.** İlk uygulamada `cokme` her denemede tetikleniyordu →
own'da 10 çökme üst üste, celery/temporal'da düğüm asılı kaldı. Checkpoint'in varlığı
**süreçler arası geçerli bir "bir kez çöktü" işareti** olarak kullanıldı; artık tek atış.

**3. Celery'de çöken worker kurtarılmıyordu.** Worker ayrı süreçte öldüğünde task
`running` asılı kalıyor ve Celery'nin haberi olmuyordu. Dispatcher'ın bekleme
döngüsüne `recover_stale()` süpürmesi eklendi (own bunu zaten yapıyordu).

Ayrıca: `run_saved` artık `on_event`'i celery ve temporal'a da geçiriyor — önceden
geçmediği için o iki motorda arayüze **hiç canlı log akmıyordu**.

**4. Yeni akışlar "Akışlar" ekranında görünmüyordu.** Paneli akış listesine bağlarken
çıktı: `pipelines.listing()` **dosya adına** göre sıralıyordu, ama id
`int(time.time()*1000) % 1e8` ile üretiliyor ve bu sayaç **~27,7 saatte bir başa
sarıyor**. Sarma sonrası kaydedilen akış listenin dibine düşüyor ve ilk 50'lik
pencereye hiç giremiyordu — yani sohbette kurduğun graf ekranda yoktu. Sıralama
gerçek zaman damgasına (`at`) çevrildi.

```
ÖNCE : p_35007900 (10:56, az önce üretildi) → listede YOK
SONRA: p_35007900 → listenin EN ÜSTÜNDE
```

---

## Testler

`test_node_sim.py` — **27/27**:
- aynı fn iki düğümde → yalnız hedeflenen patlıyor (+ eski yolun ikisini birden vurduğu kanıtı)
- altı mod tek tek
- dört motor aynı sonucu veriyor
- çökme tek atış
- id çevirisi (bilinmeyen id akışı bozmuyor)
- argüman override
- **geriye dönük uyum**: eski `fail_at` ve `crash_at` yolları bozulmadı

Regresyon: `test_hata` 54/54 · `test_tasklife` 42/42 · `test_zamanlama` 43/43 ·
`test_concurrency` at-most-once ✓ 5,0× · `test_devam` 3 seviye ✓

---

## Bilinen sınırlar

| konu | durum |
|---|---|
| Airflow eşzamanlılığı | kilitle serileştirildi — iki panel koşusu paralel gitmez |
| Celery'nin 6 sn açılışı | her koşuda ödeniyor; worker'ı açık tutmak iyileştirme olurdu |
| Temporal test-server binary'si `/tmp`'de | yeniden başlatmada silinirse ilk koşu indirir |
| `airflow_home` şişmesi | `wf_*.py` budanıyor (son 12), `airflow.db` `.gitignore`'a eklendi |
| ajan düğümü simülasyonu | `cokme` çalışıyor; `gecici/kalici` ajan dalında eski `fail_at` semantiğinde |

**Koşturma:** `.venv/bin/python demo-brain-agent/chat_server.py` → `http://127.0.0.1:8030` → **Motorlar**
