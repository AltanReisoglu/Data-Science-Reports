# Brain Agent Demo — LangGraph üstünde seçilebilir compaction + seçilebilir task management

> **Ne yaptık:** Kendi brain agent'ımızı **LangGraph** üstünde, **gerçek LLM** ile kurduk. İki ekseni
> de çalışma anında **seçilebilir** yaptık: (1) tool-trace compaction stratejisi, (2) task-management
> altyapısı. Böylece koçun sorduğu karşılaştırma artık teori değil, **tek tıkla koşan canlı ölçüm**.
>
> Kod: `demo-brain-agent/` · Web: `.venv/bin/python demo-brain-agent/web_server.py` → localhost:8020

---

## 1. Neden bu demo

Şimdiye kadar iki konuyu ayrı ayrı POC'ladık: tool-trace compaction (5 sistem) ve task management
(4 altyapı). Bu demo ikisini **tek bir çalışan ajanda** birleştirir ve şu soruyu somutlaştırır:

> *"Kendi ajanımızı yazsak, hangi compaction stratejisini ve hangi task-management altyapısını
> seçmeliyiz — ve seçim ne fark yaratır?"*

Cevabı okumak yerine **deneyerek** görüyoruz: dropdown'dan seç, çalıştır, metrikleri karşılaştır.

---

## 1.5 ÖNEMLİ DÜZELTME — "task management" ne demek

İlk kurduğum mimaride **ajan koşusunun tamamı = 1 task**tı; yani task'ları ajan üretmiyordu, ben
dışarıdan sarıyordum. **Doğrusu bu değil.** Koçun brief'inde ilk madde "task'ların
**oluşturulması**" — yani:

> **Ajan çalışma anında kendisi task üretir; task-management araçları o süreci yönetir.**

Bu, Hermes'in kanban modelidir: ajan `create_task(...)` diye **tool çağırır**, motor da o task'ları
yönetir. Mimari buna göre yeniden kuruldu (`taskboard.py` + `orchestrator.py`).

### Akış

```
1) PLANLAMA TURU — ajana YALNIZCA task tool'ları verilir
   LLM hedefi okur → create_task(...) × N  → task'lar DOĞAR (dinamik, bağımlılıklı)
        ↓
2) DISPATCH DÖNGÜSÜ — task-management motoru
   recompute_ready()  → parent'ları biten task'lar 'ready' (DAG kapısı)
   claim_next()       → CAS ile bir task kapılır (at-most-once)
   execute()          → o task için İŞÇİ AJAN koşar (iş tool'ları + compaction)
   complete / fail    → retry hakkı / circuit-breaker
   recover_stale()    → çöken worker'ın task'ı geri kuyruğa (checkpoint korunur)
```

**İki seviye net ayrıldı:** A = task (ajanın ürettiği iş birimi, motor bunu yönetir),
B = adım (task'ı yürütürken tool çağrısı).

### Ölçülen (own backend, `--crash-at MFA`, gerçek koşu)

```
PLANLAMA: ajan 4 task üretti (zincirleme bağımlılıkla)
[tur 1] claim: t_356996 'auth modülü kod analizini yap' → ✓ done
[tur 2] recompute_ready → 1 task blocked→ready (bağımlılık kapısı açıldı)
[tur 2] claim: t_5b4ff9 'MFA akışını test et'
        ✗ ÇÖKME: worker öldü (claim açık kaldı → stale)
        recover_stale → 1 task 'ready' (checkpoint KORUNDU)
[tur 3] claim: t_5b4ff9 (başka worker devraldı) → ✓ done
[tur 4] recompute_ready → 1 task ready → ✓ done

SONUÇ: üretilen 4 · tamamlanan 4 · çökme 1 · otomatik kurtarma 1
       compaction: 72.351 → 1.092 token (%98.5)
olaylar: created×4 → claimed → completed → unblocked → claimed → recovered → claimed → completed
```

### Backend'ler bu akışta ne yapıyor

| Backend | Board (task'lar, DAG, checkpoint) | Motorun kattığı | Ölçülen |
|---|---|---|---|
| **own** | bizde (SQLite) | claim/lease/recover_stale/breaker — hepsi bizde | 4/4 done, çökme kurtarıldı |
| **temporal** | bizde | durable yürütme + activity retry + replay | 3/3 done, **47 durable event** |
| **celery** | bizde | dağıtım + at-least-once + retry | task'lar broker'a atılır |
| **airflow** | — | **yapısal olarak uyumsuz** | DAG üretilemiyor (aşağıda) |

**Airflow bulgusu (demonun en net sonucu):** Airflow DAG'ı **parse zamanında** bilmek zorundadır;
bizim task'lar ise **çalışma anında LLM kararıyla** doğar. Yani "ajan task üretir" senaryosunda
Airflow'a verecek bir DAG **yoktur**. Üç kaçış yolu da bedelli: (1) `.expand` ile dinamik mapping —
sayı runtime'da belirlenir ama **graf şekli sabit**, ajanın kurduğu keyfi bağımlılıkları temsil
edemez; (2) ajan koşusunu tek düğüme sıkıştır — o zaman Airflow yalnız *scheduling* katmanıdır,
task yönetimini yine sen yaparsın; (3) ajan DAG dosyası üretsin — parse gecikmesi + kırılganlık.

> Airflow'un rakipsiz olduğu yer değişmedi: **cron + backfill/catchup + operatör UI** (statik batch).
> Ajanın dinamik task üretimi için doğru araç değil.

### "Ajan önceden DAG yapabilir mi?" — EVET, ama uyarlanabilirliği kaybedersin

Planlama turu **zaten bir DAG üretiyor** (task'lar + `parents`). `export_airflow_dag()` bu grafı
gerçek, sözdizimsel olarak geçerli bir Airflow DAG dosyasına döküyor — `PythonOperator`'lar +
ajanın kurduğu `>>` bağımlılıkları. Yani Airflow'un "statik DAG" şartı **teknik olarak**
sağlanabiliyor.

**Ama bedeli, ajanı ajan yapan şeyi kaybetmek:**

| | Board (own/temporal) | Dışa aktarılan DAG (Airflow) |
|---|---|---|
| Task'lar ne zaman doğar | çalışma anında, sürekli | **yalnız planlama anında (donmuş)** |
| Bir task sonucu yeni task doğurabilir mi | evet (replanlama) | **hayır** |
| Yeni hedef | aynı motor | **yeni DAG dosyası + parse gecikmesi** |
| Bağımlılık şekli | keyfi, runtime'da | dosyaya yazıldığı gibi sabit |

> **Tek cümle:** Ajan önceden DAG yapabilir; ama o DAG bir *anlık görüntüdür*. Ajanın asıl gücü
> "sonucu görünce planı değiştirmek"tir — DAG'a dökünce bundan vazgeçmiş olursun. Bu yüzden
> Airflow'u *scheduler* olarak kullanıp task yönetimini board'da bırakmak en dengeli yol.

### Eski (yanlış) model sistemden KALDIRILDI

İlk sürümdeki "ajan koşusunun tamamı = 1 task" sarmalayıcısı (`taskmgmt.py`, `celery_app.py`)
**silindi**; `agent.py`'nin `--backend` seçeneği kaldırıldı. Artık `agent.py` yalnızca *işçi
ajandır*, task yönetimi tek yerdedir: `taskboard.py` + `orchestrator.py`. Kod tabanında iki
farklı "task management" anlayışının yan yana durmaması için bilinçli temizlik.

### Gerçek dünyadan bir kusur: bozuk tool çağrısı
Model (dahili LLM) bir task'ı yapılandırılmış tool çağrısı yerine **düz metin olarak**
(`<|tool_call>call:create_task{...}`) yazdı ve token sınırında kesildi. Ajan buna dayanıklı olmalı:
`_salvage_tool_calls()` bu bozuk formatı ayrıştırıp task'ı yine de yaratıyor, olmazsa özetten
temizliyor. Üretimde LLM'e körü körüne güvenilemeyeceğinin somut örneği.

## 2. Mimari — üç katman, üçü de değiştirilebilir

```
┌─ TASK MANAGEMENT (taskmgmt.py) ──── own │ temporal │ celery │ airflow
│     oluştur → kuyruk → claim → adım adım koştur → checkpoint →
│     retry → durum takibi → çökme sonrası devam
│
├─ AJAN DÖNGÜSÜ (agent.py — LangGraph StateGraph)
│     reason (gerçek LLM) ──tool_calls?──► act (tool'ları koştur) ──► compact ──┐
│        ▲                                                                      │
│        └──────────────────────────────────────────────────────────────────────┘
│     tool_calls yoksa ──► END (nihai yanıt)
│
└─ TOOL-TRACE COMPACTION (compaction.py) ── none │ hermes │ opencode │
      openclaw │ codex │ claude_code
```

**Katmanlar birbirini bilmez** — asıl tasarım kararı bu:
- Task-management ajanın içini bilmez; sadece `step(state) → (done, state, label)` der ve
  `state`'i checkpoint'ler. Bu yüzden **aynı ajan** dört altyapıda değişmeden koşar.
- Compaction task-management'i bilmez; sadece mesaj listesini küçültür.

### LangGraph ne yapıyor (dekoratif değil)
`StateGraph` üç düğüm ve bir döngüden oluşur:
- **`reason`** — gerçek LLM çağrısı (tool-calling ile). Tool istemezse `END`'e gider.
- **`act`** — LLM'in istediği tool'ları gerçekten koşturur, sonuçları mesajlara ekler.
- **`compact`** — seçilen tool-trace stratejisini uygular.
- Koşullu kenar: `reason → act` (tool varsa) / `reason → END` (yoksa); `act → compact → reason`.
- `interrupt_after=["compact"]` → **her tool turu bir checkpoint sınırı** olur; task-management
  katmanı tam buradan yakalar.

---

## 3. Seçilebilir eksen #1 — tool-trace compaction

`compaction.py`, beş gerçek sistemin stratejisini **tek imza** arkasında toplar:

```python
res = compact("hermes", messages, budget=3000)
res.before, res.after, res.saved, res.pct, res.log
```

| Strateji | Ekol | LLM? | Ne yapar |
|---|---|:---:|---|
| `none` | — | — | temel çizgi (sıkıştırma yok) |
| `hermes` | deterministik | ✗ | 4 geçiş: dedup → tek-satır özet → arg-kırpma → basınç demotion |
| `opencode` | deterministik | ✗ | backward-prune: son-2-turn + en-yeni-N + korunan tool; fayda-freni |
| `openclaw` | LLM-özet | ✓ | grupla → parçala → chunk başına LLM özeti → uygula |
| `codex` | hibrit | ✓ | ortadan-kesme (`truncate_middle`) + model-turn windowing (handoff) |
| `claude_code` | hibrit | ✓ | microcompaction (diske dök + referans) + auto-compaction (konuşma özeti) |

**Ölçülen (gerçek koşu, hermes, bütçe 3.000):**
```
ACT → read_file(auth/login.py) → 23.530 token HAM çıktı
COMPACT [hermes] 23.661 → 148 token (−23.513 · %99.4 kazanç)
ACT → run_tests(auth)          →  3.696 token HAM çıktı
COMPACT [hermes]  3.850 → 165 token (−3.685 · %95.7 kazanç)
```
Bir koşuda toplam: **51.212 → 501 token (%99.0 kazanç)**.

### Altı stratejinin canlı taraması (aynı görev, backend=own, bütçe 3.000, 2 tur)

| Strateji | Ham → Sıkıştırılmış | Kazanç | Süre |
|---|---|---:|---:|
| `none` | 51.024 → 51.024 | %0.0 | **43,5 s** |
| `hermes` | 27.511 → 313 | %98.9 | 12,2 s |
| `opencode` | 28.017 → 5.010 | %82.1 | 13,3 s |
| `openclaw` | 23.661 → 60 | %99.7 | 4,5 s |
| `codex` | 29.010 → 3.315 | %88.6 | 12,1 s |
| `claude_code` | 27.664 → 778 | %97.2 | 11,7 s |

**Beklenmedik ama mantıklı bulgu — compaction ajanı HIZLANDIRIYOR da.** `none` ile koşu 43,5 saniye
sürdü; sıkıştıran stratejilerle ~12 saniye. Sebep: sıkıştırma yoksa her turda LLM'e 51K token
gönderiliyor → hem pahalı hem yavaş. Yani compaction sadece "pencereye sığdırma" değil, **gecikme ve
maliyet** meselesi: bu koşuda ~**3,5× hızlanma**.

> `opencode` diğerlerinden düşük görünüyor (%82) ama bu bilinçli: Katman A ile büyük çıktıyı diske
> döküp referans bırakıyor, Katman B ise son-2-turn'ü koruyor — muhafazakâr ve geri-çağrılabilir.

### Dürüst bulgu: agresif compaction'ın bedeli
`hermes` + düşük bütçede tool çıktısı tek satıra indiği için ajan bazen **aynı dosyayı tekrar
okudu**. Yani çok sıkıştırırsan bağlamı kaybedip yeniden tool çağırırsın — kazandığın token'ın
bir kısmını geri harcarsın. Bu, demoda bütçeyi/stratejiyi değiştirerek görülebilen gerçek bir
denge; "en yüksek yüzde en iyi" olmadığının canlı kanıtı.

---

## 4. Seçilebilir eksen #2 — task management

`taskmgmt.py`, dört altyapıyı **tek imza** arkasında toplar:

```python
r = run_job("own", job, crash_at="read_file", fail_at="run_tests")
r.steps_run, r.steps_skipped, r.retries, r.crashes, r.recovered
```

### Koçun 6 ekseni

| Eksen | own (SQLite) | Temporal | Celery | Airflow |
|---|---|---|---|---|
| **Task yönetimi** | tasks FSM: ready→running→done/failed | iş=Workflow, adım=Activity | iş=tek task; çok-adımlı iş kavramı yok | statik DAG düğümleri |
| **Retry / recovery** | checkpoint'ten devam + otomatik `recover_stale` | event-history replay → biten adım **atlanır** (exactly-once) | `self.retry()` task'ı **BAŞTAN** koşar | sadece hatalı düğüm retry |
| **State takibi** | SQLite satırı + checkpoint JSON (tam görünür) | durable event-history (tam denetim izi) | zayıf (result backend gerekir) | metadata DB + operatör UI |
| **Scheduling** | yok (dış cron eklenebilir) | Schedules + backfill | Beat (telafi yok) | **en güçlü** (cron + catchup) |
| **Concurrency** | CAS-claim ile çok worker (at-most-once) | task queue + worker havuzu | native worker havuzu (en iyi) | executor pool/slot |
| **Operasyonel karmaşıklık** | **çok düşük** (tek dosya, ek servis yok) | **yüksek** (cluster + determinizm disiplini) | orta (broker işletmek) | yüksek (scheduler+web+DB) |

### `own` — kendi durable çekirdeğimiz (Hermes tarzı)
`OwnCore` sınıfı, ~200 satırda: `create` → `claim` (CAS, at-most-once) → `heartbeat` (lease) →
`checkpoint` → `complete` / `fail` (circuit-breaker) → `recover_stale` (lease dolmuş / PID ölü →
otomatik `ready`, **checkpoint'e dokunmadan**).

**Ölçülen (gerçek koşu, `--crash-at read_file`):**
```
create → ready
worker-1 claim etti (CAS+lease) · attempt=0
   adım: turn0: read_file → koştu + compact
   ✗ ÇÖKME: worker öldü (claim açık kaldı → stale)
   recover_stale() → 1 iş otomatik 'ready' (checkpoint KORUNDU)
worker-2 claim etti (CAS+lease)
   ↻ checkpoint'ten DEVAM: 1 adım zaten bitmiş → TEKRAR KOŞMAYACAK (turn0:read_file)
   adım: turn1: run_tests → koştu + compact  …  ✓ complete → done

SONUÇ: adım=4 · checkpoint'ten atlanan=1 · çökme=1 · otomatik kurtarma=1
events: created → claimed → recovered → claimed → completed
```

### `temporal` — gerçek durable execution
İş = workflow (saf/deterministik gövde), her ajan turu = activity. Ölçülen koşu:
`✓ workflow tamamlandı · 29 durable event` (her adım SCHEDULED→STARTED→COMPLETED).
Not: workflow sınıfı **modül seviyesinde** tanımlanmalı (Temporal yerel sınıfı reddediyor) ve
LLM gibi non-deterministik iş **yalnız activity içinde** olmalı — determinizm disiplini.

### `celery` — gerçek kuyruk + worker
Ajan işi tek celery task'ı; gerçek worker ayrı süreçte, filesystem broker ile. Geçici hata →
`self.retry()` → **tüm task baştan** koşar; tamamlanmış turlar tekrarlanır. `acks_late=True`
açık (worker çökerse mesaj yeniden teslim). A-seviyesi resume istiyorsan checkpoint'i sen kurarsın.

### `airflow` — referans DAG
Çalıştırılabilir `brain_agent_dag.py` üretir ama demoda koşturulmaz (scheduler+webserver+DB gerekir).
**Ajan için neden doğal değil:** Airflow statik DAG bekler, ajan sıradaki adıma çalışma anında karar
verir → ajan tek düğüme sıkışır, Airflow yalnız *scheduling* katmanı olur. Rakipsiz olduğu yer:
cron + backfill/catchup + operatör UI.

---

## 5. Nasıl çalıştırılır

```bash
# Web (önerilen): iki dropdown + çökme/hata enjeksiyonu + canlı metrikler
.venv/bin/python demo-brain-agent/web_server.py        # → http://127.0.0.1:8020

# Terminal
cd demo-brain-agent
../.venv/bin/python agent.py --backend none     --strategy hermes          # saf LangGraph
../.venv/bin/python agent.py --backend own      --strategy hermes --crash-at read_file
../.venv/bin/python agent.py --backend temporal --strategy codex
../.venv/bin/python agent.py --backend celery   --strategy opencode --fail-at run_tests
../.venv/bin/python agent.py --backend airflow                              # DAG üret

# Modüller tek başına
../.venv/bin/python compaction.py    # tüm stratejiler aynı geçmiş üstünde
../.venv/bin/python taskmgmt.py      # sahte iş: çökme + retry + kurtarma
```

---

## 6. Sonuç ve öneri

Demo, önceki raporların iddialarını **tek bir çalışan ajanda** doğruluyor:

1. **Tool-trace compaction şart** — tek bir `read_file` 23K token; sıkıştırma olmadan birkaç turda
   pencere dolar. Hermes bunu %99'a varan oranda düşürüyor.
2. **Strateji seçimi bir denge** — çok agresif sıkıştırma ajanı tekrar tool çağırmaya itiyor.
   Bütçe ve strateji birlikte ayarlanmalı.
3. **Task management'ta asıl ayrım "kaldığı yerden devam"** — `own` ve `temporal` tamamlanan işi
   koruyor, `celery` baştan koşuyor, `airflow` ajana yapısal olarak uymuyor.
4. **brain_chat_V2 için öneri değişmedi:** Hermes-tarzı hafif durable çekirdek (**`own`**) — bu
   demoda çalışır halde; otomatik crash-recovery + checkpoint'ten devam, ek servis olmadan.
   Temporal'a ancak çok-makineli/uzun-bekleyen ihtiyaç netleşince geçilmeli.

**İlgili belgeler:** `report/USTA-REHBER-tool-trace-ve-task-management.md` (teori),
`report/brain-chat-v2-task-management-entegrasyon.md` (4 rota + build kodu),
`report/brain-poc-fonksiyon-fonksiyon-anlatim.md` (fonksiyon fonksiyon),
`demo-brain-agent/README.md` (bu demonun kullanım kılavuzu).
