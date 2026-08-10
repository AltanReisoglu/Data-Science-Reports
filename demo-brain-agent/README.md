# Brain Agent Demo — LangGraph + Tool-Trace Compaction + Task Management

> **İki mod var. Asıl olan birincisi:**
>
> **1) Ajan TASK ÜRETİR → motor yönetir** (`orchestrator.py`) — ajan çalışma anında
> `create_task(...)` tool'unu çağırarak hedefi task'lara böler (bağımlılıklarıyla birlikte);
> task-management motoru bu task'ları yönetir: kuyruk, DAG kapısı, claim, retry, durum takibi,
> çökme sonrası devam. **Task'ları ajan üretir, motor yönetir.**
>
> **2) Tek ajan koşusu** (`agent.py`) — task üretimi yok; yalnız compaction ölçümü için sade mod.

```bash
# 1. mod (asıl): ajan task üretir, motor yönetir
../.venv/bin/python orchestrator.py --backend own --strategy hermes --crash-at MFA

# web'de: Mod = "Ajan TASK ÜRETİR → motor yönetir"
.venv/bin/python demo-brain-agent/web_server.py    # → localhost:8020
```

**Ölçülen (own backend, gerçek koşu):** ajan **4 task üretti** (zincirleme bağımlılıkla) →
motor sırayla yürüttü → ortadaki task'ta **worker çöktü** → `recover_stale` otomatik topladı →
başka worker devraldı → **4/4 tamamlandı**; tool-trace compaction **72.351 → 1.092 token (%98.5)**.

### Task'ların ÇOĞU deterministik fonksiyondur (Airflow DAG gibi)

Bir task **düz bir fonksiyon** olabilir — ve olması gereken de budur. LLM sadece gerçekten
muhakeme gereken düğümde devreye girer.

| | `kind="function"` (**varsayılan**) | `kind="agent"` (istisna) |
|---|---|---|
| Ne çalışır | kayıtlı Python fonksiyonu | LLM ajan döngüsü |
| Maliyet | ~0 | LLM çağrısı |
| Tekrarlanabilir | ✅ aynı girdi → aynı çıktı | ❌ |
| Retry güvenli | ✅ idempotent | dikkatli olmalı |
| Ne zaman | okuma, tarama, test, dönüştürme, rapor | yorum, karar, serbest metin |

**Fonksiyon kaydı** (`functions.py`) — ajan yalnız bunlardan DAG kurabilir, yeni fonksiyon icat edemez:
```
fetch_source(path)          → path, lines, bytes, sha1
scan_patterns(pattern,path) → count, matches
run_test_suite(suite)       → total, passed, failed, failures[]
cross_check()               → korelasyon[] (İKİ upstream'i birleştirir = join)
render_report(title)        → rapor_md
checksum_guard(expected)    → degisti?
```

**Düğümler arası veri akışı** (Airflow XCom karşılığı): bir düğümün sonucu, çocuğuna
`_up = {parent_id: result}` olarak geçer. **Büyük veri sonuçtan akmaz — referans (path) akar.**

**Ölçülen (gerçek koşu, 7 düğüm):**
```
[tur 1] fn fetch_source({'path':'auth/login.py'}) ← upstream —
        → {'path':..., 'lines':2291, 'bytes':98689, 'sha1':'de3d7bb5d251'}   ✓ LLM kullanılmadı
[tur 3] fn scan_patterns({'pattern':'mfa_token'}) ← upstream ['t_d243bf']
        → {'count':4, 'matches':[{'line':2286,'text':'def login(user, password, mfa_token=N…'}]}
[tur 4] fn cross_check({}) ← upstream ['t_d243bf','t_c21010']        ← JOIN düğümü
[tur 7] fn render_report(...) → rapor_md
```
**Yürütmede LLM hiç çağrılmadı** — LLM sadece planı kurdu, gerisi saf fonksiyon akışı.

Planlayıcının iki tool'u var:
- **`add_step(fn, args_json, depends_on)`** → deterministik düğüm ← **önce bu**
- `add_agent_step(title, body, depends_on)` → LLM düğümü ← sadece gerekince

### Ajan task'ı NE ZAMAN üretebilir? → İki yerde

Task üretmek de bir **tool çağrısıdır** — ama "iş tool'u" (read_file) değil, **task-yönetim tool'u**:

| Nerede | Tool | Kim çağırır | Sonuç |
|---|---|---|---|
| **Planlama turu** | `create_task(title, body, depends_on)` | planlayıcı ajan | hedefi task'lara böler, DAG kurar |
| **Yürütme sırasında** | `spawn_task(title, body)` | işçi ajan | iş yaparken keşfettiği YENİ işi task olarak açar (**replanlama**) |

**Ölçülen (gerçek koşu):**
```
PLANLAMA: 1 task üretildi
[tur 1] claim: 'Testleri koştur…'
   SPAWN (yürütme anında) → t_003986 'Payment Gateway Timeout Hatası'
   SPAWN (yürütme anında) → t_546bf6 'User Session Expired Hatası'
   ✓ done
[tur 2] claim: t_003986 → ✓ done     ← ajanın ÇALIŞIRKEN doğurduğu task
[tur 3] claim: t_546bf6 → ✓ done
ÖZET: 1 (planlamada) + 2 (YÜRÜTME ANINDA) · tamamlanan 3/3
```

**Frenler** (kaçak üretime karşı): task başına `MAX_SPAWN_PER_TASK=2`, board toplamı
`MAX_TASKS_TOTAL=12`. Aşılınca tool reddediyor. Her task'ın `created_by` alanı kimin
ürettiğini tutuyor (`agent` = planlama, `worker:<id>` = yürütme anında).

> **Not (dürüst gözlem):** Yetenek doğrulandı ve çalışıyor, ama model kendiliğinden nadiren
> `spawn_task` çağırıyor — genelde planlayıcının verdiği kapsamda kalıyor. Görev metninde
> açıkça belirtilince güvenilir şekilde kullanıyor. Yani sınır kod değil, **modelin yargısı**.

### "Ajan önceden DAG yapabilir mi?" → EVET, ama bedeli var

Planlama turu zaten bir DAG üretir (task'lar + `parents`). `export_airflow_dag()` bunu **gerçek,
sözdizimsel olarak geçerli bir Airflow DAG dosyasına** döker:

```bash
../.venv/bin/python orchestrator.py --backend airflow    # → brain_agent_plan.py üretir
```
```python
    t_ad926a = PythonOperator(task_id='t_ad926a', python_callable=_run_task,
        op_kwargs={'task_title': 'Auth modülünü ve MFA akışını analiz et', ...})
    ...
    t_ad926a >> t_0bd219        # ajanın kurduğu bağımlılıklar
    t_0bd219 >> t_13a1d8
```

**Bedeli:** DAG artık **donmuştur**. Yürütme sırasında ajan yeni task üretemez, replanlama
yapamaz; her yeni hedef = yeni DAG dosyası + scheduler'ın parse gecikmesi. Yani Airflow'a
uydurmak için ajanın en güçlü yanından (uyarlanabilirlik) vazgeçmiş olursun.

### Eski (yanlış) model KALDIRILDI

İlk sürümde "ajan koşusunun tamamı = 1 task" varsayan bir sarmalayıcı vardı (`taskmgmt.py`,
`celery_app.py`). O modelde task'ları **ajan üretmiyordu**. Bu dosyalar **silindi**;
`agent.py`'nin `--backend` seçeneği kaldırıldı (artık yalnız işçi ajan). Task yönetimi
tek yerde: `taskboard.py` + `orchestrator.py`.

---


Kendi **brain agent**'ımız: gerçek bir LLM ile tool çağıran, **LangGraph** üstünde koşan bir ajan.
İki eksen de **çalışma anında seçilebilir**:

| Eksen | Seçenekler | Nerede |
|---|---|---|
| **Tool-trace compaction stratejisi** | `none` · `hermes` · `opencode` · `openclaw` · `codex` · `claude_code` | `compaction.py` |
| **Task-management altyapısı** | `own` · `temporal` · `celery` · `airflow` · `none` | `taskmgmt.py` |

Ajan Hermes tarzıdır (deterministik çekirdek, tool'lar büyük çıktı üretir) ama altyapı bizim
seçimimize bırakılmıştır — koçun istediği karşılaştırmayı **canlı, ölçülebilir** hale getirir.

---

## Mimari — üç katman, üçü de değiştirilebilir

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

**Katmanlar birbirini bilmez:** task-management ajanın içini bilmez (sadece "bir adım ilerlet ve
checkpoint'le" der); compaction da task-management'i bilmez. Bu yüzden ikisi bağımsız seçilebilir.

---

## Çalıştırma

### Web arayüzü (önerilen)
```bash
.venv/bin/python demo-brain-agent/web_server.py     # → http://127.0.0.1:8020
```
İki dropdown'dan strateji + backend seç, istersen çökme/geçici hata enjekte et, **▶ Ajanı çalıştır**.
Ölçümler (ham→sıkıştırılmış token, kazanç %, koşan/atlanan adım, retry, çökme, kurtarma), ajan izi,
task-management izi ve ajanın nihai yanıtı canlı görünür.

### Terminal
```bash
cd demo-brain-agent

# saf LangGraph (task-mgmt yok)
../.venv/bin/python agent.py --backend none --strategy hermes

# kendi durable çekirdeğimiz + çökme enjeksiyonu (checkpoint'ten devam kanıtı)
../.venv/bin/python agent.py --backend own --strategy hermes --crash-at read_file

# Temporal (gerçek dev server) — event-history
../.venv/bin/python agent.py --backend temporal --strategy codex

# Celery (gerçek worker + broker) + geçici hata → retry BAŞTAN koşar
../.venv/bin/python agent.py --backend celery --strategy opencode --fail-at run_tests

# Airflow referans DAG üret
../.venv/bin/python agent.py --backend airflow
```

Modülleri tek başına da test edebilirsin:
```bash
../.venv/bin/python compaction.py    # tüm stratejiler aynı geçmiş üstünde
../.venv/bin/python taskmgmt.py      # sahte iş: çökme + retry + kurtarma
```

---

## Altı stratejinin canlı taraması (backend=own, bütçe 3.000, 2 tur)

| Strateji | Ham → Sıkıştırılmış | Kazanç | Süre |
|---|---|---:|---:|
| `none` | 51.024 → 51.024 | %0.0 | **43,5 s** |
| `hermes` | 27.511 → 313 | %98.9 | 12,2 s |
| `opencode` | 28.017 → 5.010 | %82.1 | 13,3 s |
| `openclaw` | 23.661 → 60 | %99.7 | 4,5 s |
| `codex` | 29.010 → 3.315 | %88.6 | 12,1 s |
| `claude_code` | 27.664 → 778 | %97.2 | 11,7 s |

**Compaction ajanı hızlandırıyor da:** `none` 43,5 s, sıkıştıranlar ~12 s (~3,5× hızlanma) —
çünkü sıkıştırma yoksa her turda LLM'e 51K token gidiyor.

## Ölçülen şeyler

**Tool-trace compaction** — her turda: ham tool çıktısı → compaction sonrası, kazanç %.
Örnek koşu (hermes, bütçe 3.000):
```
ACT → read_file(auth/login.py) → 23.530 token HAM çıktı
COMPACT [hermes] 23.661 → 148 token (−23.513 · %99.4 kazanç)
```

**Task management** — koşan adım, checkpoint'ten atlanan adım, retry, çökme, otomatik kurtarma.
Örnek koşu (own, `--crash-at read_file`):
```
worker-1 claim etti (CAS+lease) · attempt=0
   adım: turn0: read_file → koştu + compact
   ✗ ÇÖKME: worker öldü (claim açık kaldı → stale)
   recover_stale() → 1 iş otomatik 'ready' (checkpoint KORUNDU)
worker-2 claim etti (CAS+lease)
   ↻ checkpoint'ten DEVAM: 1 adım zaten bitmiş → TEKRAR KOŞMAYACAK (turn0:read_file)
   adım: turn1 … → ✓ complete → done
```

---

## Backend karşılaştırması (koçun 6 ekseni)

| Eksen | own (SQLite) | Temporal | Celery | Airflow |
|---|---|---|---|---|
| **Task yönetimi** | tasks FSM: ready→running→done/failed | iş=Workflow, adım=Activity | iş=tek task; çok-adımlı iş kavramı yok | statik DAG düğümleri |
| **Retry / recovery** | checkpoint'ten devam + otomatik `recover_stale` | event-history replay → biten adım **atlanır** (exactly-once) | `self.retry()` task'ı **BAŞTAN** koşar | sadece hatalı düğüm retry |
| **State takibi** | SQLite satırı + checkpoint JSON | durable event-history | zayıf (result backend gerekir) | metadata DB + operatör UI |
| **Scheduling** | yok (dış cron) | Schedules + backfill | Beat (telafi yok) | **en güçlü** (cron + catchup) |
| **Concurrency** | CAS-claim ile çok worker (at-most-once) | task queue + worker havuzu | native worker havuzu (en iyi) | executor pool/slot |
| **Operasyonel karmaşıklık** | **çok düşük** (tek dosya) | **yüksek** (cluster + determinizm) | orta (broker) | yüksek (scheduler+web+DB) |

**Airflow neden ajan için doğal değil:** statik DAG bekler, ajan ise sıradaki adıma çalışma anında
karar verir. Ajan tek düğüme sıkışır → Airflow yalnız *scheduling* katmanı olur. (Referans DAG
`brain_agent_dag.py` olarak üretilir.)

---

## Dosyalar

| Dosya | Ne yapar |
|---|---|
| **`taskboard.py`** | **Ajanın ürettiği task'ların durable board'u:** FSM + DAG bağımlılık kapısı (`recompute_ready`) + CAS-claim + lease + breaker + olay günlüğü. Ayrıca ajana verilen `create_task` / `list_tasks` tool'ları |
| **`orchestrator.py`** | **Asıl akış:** PLANLAMA (ajan task üretir) → DISPATCH (motor yönetir). Backend'e göre dispatch: own / celery / temporal / airflow |
| `temporal_defs.py` | Board dispatch'i için Temporal workflow + activity'ler (modül seviyesinde) |
| `celery_worker.py` | Board'daki tek task'ı yürüten gerçek Celery worker |
| `agent.py` | **İşçi ajan**: LangGraph döngüsü (reason→act→compact), iş tool'ları, `BrainAgentJob`, CLI |
| `compaction.py` | 5 tool-trace stratejisi + `none`, tek `compact(strategy, messages, budget)` arayüzü |
| `taskmgmt.py` | 4 backend + `run_job(backend, job)` arayüzü; `OwnCore` durable çekirdeği |
| `celery_app.py` | Gerçek Celery worker uygulaması (ayrı süreçte koşar) |
| `web_server.py` | Tarayıcı arayüzü (stdlib, port 8020) |
| `brain_agent_dag.py` | Airflow backend'i seçilince üretilen referans DAG |

---

## Notlar

- **LLM:** `poc/llm.py` üzerinden OpenAI-uyumlu iç endpoint. Anahtar `.env`'de tutulur, arayüze
  gönderilmez, log'a yazılmaz. LLM erişimi yoksa LLM-tabanlı stratejiler deterministik
  fallback'e düşer (ajan çalışmaz — reason adımı gerçek LLM ister).
- **Tool'lar bilerek büyük çıktı üretir** (20K+ token) ki compaction'ın etkisi ölçülebilsin.
- **Gözlem — agresif compaction'ın bedeli:** `hermes` + düşük bütçede tool çıktısı tek satıra
  indiğinden ajan bazen aynı dosyayı **tekrar okur**. Bu gerçek bir denge: çok sıkıştırırsan
  bağlamı kaybedip yeniden tool çağırırsın. Bütçeyi/stratejiyi değiştirip karşılaştır.
- **Çökme enjeksiyonu** yalnız `own` backend'de anlamlıdır (Temporal'da worker çökse workflow
  zaten replay ile devam eder; Celery'de mesaj yeniden teslim edilir).
