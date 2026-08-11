# Motor İnceleme Katmanı — Tasarım

**Tarih:** 2026-08-11
**Nerede:** `demo-brain-agent/` → `http://127.0.0.1:8030`
**Durum:** onaylandı, uygulamaya geçiliyor

---

## Amaç

Ekibe dört yürütme motorunu **kanıtlı olarak inceletmek**: her birinin modülleri,
özellikleri, neyi öne çıkardığı, neyi kötü yaptığı — hepsi ekranda, hepsi bir
kanıta bağlı.

Bu bir karşılaştırma tablosu değil. Her motor kendi **künyesini** alıyor.

## Neden şimdi

Mevcut POC (`chat_server.py`, :8030) şunları zaten yapıyor: ajan hedefi task'lara
bölüyor (`plan_phase`), board FSM'i durumu tutuyor, dört motor gerçekten koşuyor,
retry/iptal/çökme ölçülüyor. Ölçülmüş sonuç: **dört motor byte-byte aynı çıktı**.

Eksik olan: motorların **birbirinden nasıl farklı olduğu**. Mevcut karşılaştırma
tablosunun sütunları motordan bağımsız (`süre · tamamlanan · başarısız · iptal ·
deneme`) — o tablo "dördü aynı davranıyor"u kanıtlamak için tasarlandı. Celery'nin
canvas'ı, Airflow'un XCom'u, Temporal'ın event history'si arayüzde **hiç görünmüyor**.

---

## Çerçeve: üçü aynı kategoride değil

Bu, ekrandaki ilk mesaj olmalı — en yaygın hata bunları "3 rakip" sanmak.

| | temel soyutlama | bir cümlede |
|---|---|---|
| **Celery** | fonksiyon çağrısı | "Bu fonksiyonu başka bir süreçte çalıştır" |
| **Airflow** | DAG (statik graf) | "Her gece 08:00'de şu adım dizisini koştur" |
| **Temporal** | workflow = kod | "Bu programı, makine çökse bile kaldığı yerden bitir" |
| **own (hermes)** | task tablosu | "Durumu kendi kalıcı tablonda tut, motoru değiştirilebilir bırak" |

İç içe geçebilirler: Airflow'un `CeleryExecutor`'ı işleri Celery'ye dağıtır. Yani
"Airflow vs Celery" çoğu zaman yanlış sorudur.

**Terminoloji tuzağı** — "task" her araçta farklı katmanı gösterir:

- **A = iş/job** (yaşam döngüsü olan iş birimi) → Airflow'da `DagRun`, Temporal'da
  `Workflow`, Celery'de **karşılığı yok**
- **B = adım** (tek fonksiyon çağrısı) → Airflow'da `task`, Temporal'da `activity`,
  Celery'de `task`

"Airflow task retry eder" = bir adımı baştan koşar. "Temporal kaldığı yerden sürdürür"
= bütün işi kurtarır. Aynı kelime, farklı katman.

---

## Mimari — Yaklaşım A: ayrı modüller, ince uç

`chat_server.py` şu an 1788 satır, 13 HTTP ucu, artı tüm sayfa gömülü string.
Yeni yetenek oraya eklenirse 3000+ satır olur. Bunun yerine:

```
motor_kunye.py    künye · analoji · iç mimari · modül haritası · güçlü/zayıf + kanıt
motor_dili.py     board grafı → motorun doğal dilinde KAYNAK KOD (4 çevirici)
motor_ayar.py     ayar şeması: ad · varsayılan · bizde · etkili mi · etkisizse neden
chat_ui.html      sayfa gövdesi dosyaya çıkar (1788 → ~700 satır)
chat_server.py    3 yeni uç: /motor/kunye · /motor/dil · /motor/kos — boru, gövde değil
```

**Gerekçe:** her modül HTTP olmadan tek başına koşturulabilir ve test edilebilir.
Bu desen bu repoda zaten kanıtlandı — `poc/kiyas.py` aynı şekilde kuruldu ve
çalıştı (POC'ları import edip yapılandırılmış veri üretiyor, sunucudan bağımsız).

---

## Ekran düzeni

Akış detay ekranına motor başına sekme. Her sekme yedi bölüm:

```
┌ own · hermes │ celery │ airflow │ temporal ┐
│                                            │
│ ① KÜNYE     tek cümle · analoji ·          │
│             iç mimari şeması               │
│                                            │
│ ② MODÜLLER  bileşen haritası — her biri ne │
│             yapar, BİZİM koşumuzda devrede │
│             mi, değilse neden              │
│                                            │
│ ③ BU GRAF   aynı graf, bu motorun dilinde  │
│             (üretilen kaynak kod)          │
│                                            │
│ ④ KOŞUM     canlı, düğüm düğüm + motorun   │
│             kendi kaydı                    │
│                                            │
│ ⑤ AYARLAR   etkileşimli · etkisizler ⚠     │
│                                            │
│ ⑥ GÜÇLÜ/ZAYIF  her madde → kanıt rozeti    │
│                                            │
│ ⑦ İNCELİKLER   YALNIZ bu motorda olan      │
│                davranışlar — çalışma        │
│                mantığının ince ayrıntıları  │
└────────────────────────────────────────────┘
```

### ⑦ İncelikler — motora münhasır davranışlar

Güçlü/zayıf listesi "seçim" sorusunu cevaplar. Bu bölüm ayrı bir soruyu
cevaplar: **bu motoru kullanırsan neyi bilmen gerekir?** Yalnız o motorda
karşına çıkan, başka hiçbirinde karşılığı olmayan davranışlar.

Örnekler (tam liste `motor_kunye.py` içinde):

| motor | incelik |
|---|---|
| Celery | **visibility timeout tuzağı** — task bu süreden uzun sürerse broker mesajı 'kaybolmuş' sayıp yeniden teslim eder → aynı iş İKİ worker'da paralel koşar |
| Celery | **prefork tuzağı** — varsayılan pool fork tabanlı; her worker bellek kopyası, LLM/ML kütüphaneleriyle şişer |
| Airflow | **zombie detection** — worker heartbeat'i kaçırırsa scheduler task'ı zombie ilan edip fail/retry eder |
| Airflow | **koşullu dal İKİ kavram** — `BranchPythonOperator` + `trigger_rule`; ikincisi unutulursa birleşim düğümü de skip olur |
| Temporal | **başarısız deneme history'ye YAZILMAZ** — history kompakt kalsın diye; attempt numarası `ActivityTaskStarted` event'inde taşınır |
| Temporal | **continue-as-new** — history ~50K event / 50MB'ı aşınca temiz history ile devam etmek gerekir |
| own | **fencing** — `complete`/`fail` çağrısı `claimer` ile eşleşmezse yazma reddedilir; bayat claim sonucu çöpe gitmesin diye |

### ① Künye — tek cümle, analoji, iç mimari

Analojiler ekranda kalıcı olarak duracak; sunumda ağızdan çıkacak cümle bu.

| motor | analoji |
|---|---|
| **Celery** | Restoran mutfağındaki sipariş fişi. Garson fişi çiviye takar ve gider; boştaki aşçı alır, yapar. Fiş yanarsa yenisi takılır — ama aşçı yemeğe **sıfırdan** başlar. |
| **Airflow** | Fabrika üretim hattı + duvardaki dev pano. Hat sabit, panoda hangi istasyonda ne olduğu canlı görünür. Bir istasyon bozulursa **sadece o istasyon** tekrarlanır. |
| **Temporal** | Kaydedilmiş oyun. Elektrik kesildi diye baştan başlamıyorsun; motor son duruma yükleyip devam ettiriyor. Üstelik kaydı **sen yapmıyorsun**. |
| **own** | Mutfaktaki sipariş defteri. Fişleri aşçıya sen dağıtıyorsun ama defter sende: hangi sipariş hangi aşamada, kaç kez denendi, kim aldı. |

İç mimari şemaları ASCII olarak gömülü (kaynak: kullanıcının hazırladığı referans):

```
CELERY
  Producer ──msg──▶ Broker (Redis/RabbitMQ) ──pull──▶ Worker havuzu
                          │                              │
                          ▼                              ▼
                    Result backend ◀──────sonuç──────────┘

AIRFLOW
  DAG dosyaları → Scheduler → Executor → Worker'lar
                      │           │
                      ▼           ▼
                 Metadata DB (tek doğruluk kaynağı)
                      ▲
              Webserver/UI · Triggerer

TEMPORAL
  Client ──▶ Cluster (Frontend · History · Matching · Persistence)
                      │ long-poll
                      ▼
              SDK Worker (senin kodun) — cluster'da ÇALIŞMAZ

OWN
  plan_phase → board (SQLite) → dispatcher → motor
                  │
                  └─ FSM · CAS claim · lease · breaker · checkpoint · olay günlüğü
```

### ② Modüller — "tüm modülleri" isteğinin karşılığı

| motor | modüller |
|---|---|
| **Airflow** | DAG parser · Scheduler · Executor (Sequential/Local/Celery/K8s) · Metadata DB · Webserver · Triggerer · Operator · XCom · Sensor · `trigger_rule` · catchup/backfill · Pool · `max_active_runs` · SLA · Hook/Connection |
| **Celery** | Broker · Worker (prefork/solo/gevent) · Result backend · Canvas (`chain`/`group`/`chord`) · Beat · Flower · `acks_late` · prefetch · queue routing · `autoretry_for` · visibility timeout |
| **Temporal** | Workflow · Activity · Worker · Task queue · Event history · Replay · Signal/Query/Update · Timer · Child workflow · Continue-as-new · RetryPolicy · Schedule · Versioning (`patched`) · Activity heartbeat · Sticky execution |
| **own** | Task tablosu · FSM · CAS claim · lease+heartbeat · `recover_stale` · circuit breaker · `cancel_downstream` · checkpoint · olay günlüğü |

Her modülün yanında **"bizim koşumuzda devrede mi"** işareti — asıl öğretici kısım:

```
Celery   Canvas          ⊘  board DAG kapısını devraldı → delay() teker teker
Celery   max_retries     ⊘  retry otoritesi board'da (bkz. hata ⑪)
Airflow  Scheduler       ⊘  `dags test` ile tetikliyoruz
Airflow  XCom            ✓  veri akışı gerçekten oradan geçiyor
Temporal RetryPolicy     ⊘  board sayıyor (bkz. hata ②)
Temporal Event history   ✓  replay gerçekten çalışıyor
```

Ne kazandığımız ve **ne feda ettiğimiz** aynı tabloda.

### ③ Bu graf, bu motorun dilinde — `motor_dili.py`

Board grafı → dört ayrı kaynak dosya. Airflow'unki hazır
(`orchestrator.export_airflow_dag`), oraya taşınıyor; üçü yazılacak.

```
board grafı              →  celery:    chain/group/chord ağacı
(düğüm + parents + args)    airflow:   PythonOperator + >> + XCom   [var]
                            temporal:  workflow fn + asyncio.gather
                            own:       board.create_task çağrıları
```

Tez: **"aynı graf, dört dil."** Her dilin kendi tuzağı üretilen kodun içinde
görünür hale geliyor — Celery'de imza sızıntısı, Airflow'da `trigger_rule`,
Temporal'da determinizm kısıtı.

### ④ Koşum — canlı, düğüm düğüm

Mevcut `wf_kostur` + düğüm bazlı tablo yeniden kullanılıyor. Yeni olan: her motorun
**kendi kaydı** da gösteriliyor.

| motor | kendi kaydı |
|---|---|
| Airflow | `task_instance` (state, `try_number`), `xcom` |
| Temporal | event history olayları |
| own | board olay günlüğü (created/claimed/completed/failed/recovered) |
| Celery | **yok** — ve bu, gösterilecek bir bulgu |

### ⑤ Ayarlar — etkileşimli, etkisizler işaretli

Her ayar: `ad · varsayılan · bizde · etkili mi · etkisizse neden · ölçülen fark`.

```
Airflow  retries       [3 ▾]   ✓ etkili
         retry_delay   [1s ▾]  ✓ etkili   → 30s seçilirse koşu 62 sn (ölçüldü)
         max_active_ti [1 ▾]   ⚠ ETKİSİZ  → SequentialExecutor zaten seri
Celery   max_retries   [3 ▾]   ⚠ ETKİSİZ  → retry otoritesi board'da (hata ⑪)
         acks_late     [✓ ]    ✓ etkili   → worker çökerse mesaj yeniden teslim
```

Etkisizlik gizlenmiyor; mimarinin sonucu ve gerçek hatalarımıza bağlı.

### ⑥ Güçlü / zayıf — her madde bir kanıta bağlı

Kanıt türleri:

| tür | ne | canlı mı |
|---|---|---|
| `ölçüm` | koşudan çıkan sayı (süre, durum, deneme) | canlı |
| `dosya` | o graftan üretilen kaynak kod | canlı |
| `kayıt` | motorun kendi kaydı (XCom, event history, olay günlüğü) | canlı |
| `deney` | ayarı değiştir → koştur → farkı ölç | canlı |
| `repo` | `harnesses/` altındaki gerçek kaynak satırı | sabit, doğrulanabilir |

Son satırın disiplini bu repoda kanıtlandı: bugün `poc/` altındaki tool-trace
POC'ları gerçek klonlara karşı denetlendi ve iki sapma bulundu.

**Çapa ölçüm** (her sekmede tekrarlanır): aynı iş `fetch → process → deliver`,
`process` ilk denemede patlıyor. Tek soru: **pahalı `fetch` kaç kez koştu?**

```
Celery    2×  ← baştan koşar, defteri kimse tutmaz
Airflow   1×  ← sadece hatalı düğüm tekrarlanır
Temporal  1×  ← history replay, biten activity atlanır
own       1×  ← board düğüm bazında sayar
```

---

## Veri akışı

```
sohbet: "ETL kur"
   → plan_phase()          graf kurulur, pipelines_store'a kaydedilir   [mevcut]
   → /motor/kunye?id=…     künye + modül durumu + kanıtlar              [yeni]
   → /motor/dil?id=…       motor_dili → 4 kaynak dosya                  [yeni]
   → /motor/kos?…&ayar=    ayarlı koşum, SSE, fark ölçümü               [yeni]
```

---

## Hata durumları

- **Kod üretici desteklemeyen desen görürse** — sessiz geçmez; o motorun panelinde
  *"bu graf X'te doğrudan ifade edilemez, çünkü …"* yazar. Bu bir bulgu, gizlenecek
  şey değil.
- **Motor koşumu patlarsa** — diğer üçü sürer, o sekme hata paketi gösterir
  (`poc/kiyas.py`'deki desen).
- **Ayar geçersizse** — koşum başlamadan reddedilir, neden yazılır.
- **Airflow eşzamanlılığı** — `SequentialExecutor` + sqlite; koşumlar kilitle
  serileştirilir (mevcut `airflow_runner._KILIT`).

---

## Test

- `test_motor_dili.py` — dört üretici × dört desen (zincir/elmas/koşullu/dinamik):
  üretilen kod sözdizimsel geçerli (`ast.parse`) ve graf yapısını koruyor
  (düğüm sayısı, kenarlar).
- `test_motor_ayar.py` — "etkili" işaretli her ayar gerçekten farkı değiştiriyor;
  "etkisiz" işaretli hiçbiri değiştirmiyor. Bu test etiketlerimizin doğruluğunu
  ölçüyor — geçmesi zor, tam da bu yüzden değerli.
- Regresyon: `test_node_sim` 27 · `test_hata` 54 · `test_tasklife` 42 ·
  `test_zamanlama` 43.

---

## Kapsam dışı

Gerçek Airflow scheduler/webserver ayağa kaldırma · gerçek broker (filesystem
yeterli) · yük testi · Temporal cluster · üretilen Celery/Temporal kodunu ayrıca
koşturmak (kod gösterilir, koşum board üzerinden).

---

## Bilinen sınırlar (ekranda da yazılacak)

- Ölçek oyuncak: filesystem broker, SQLite, tek makine, `SequentialExecutor`.
  Ölçülen şey **mekanizma**, kapasite değil.
- Airflow `retry_delay` panelde 1 sn'ye çekildi; varsayılanı 30 sn.
- Celery'nin ~6 sn'si worker açılışı.
