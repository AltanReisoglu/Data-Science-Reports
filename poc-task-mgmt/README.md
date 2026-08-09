# Task-Management POC'ları — GERÇEK framework'lerle

Bu klasör, task orkestrasyonunu **gerçek framework'lerle** (simülasyon değil) gösteren
çalıştırılabilir POC'lar içerir. Hepsi **aynı senaryoyu** kullanır:

> 3-adımlı iş: **fetch → process → deliver**; `process` ilk denemede **geçici hata**
> verir ve **retry** edilir. Ek olarak worker **çökmesi** sonrası ne olduğu incelenir.

Amaç: "retry" ve "hata sonrası devam"ın her framework'te **gerçekte** nasıl davrandığını
kanıtlamak — özellikle **retry'da işin baştan mı yoksa kaldığı yerden mi koştuğunu**.

---

## Web'den görsel test (önerilen)

```bash
.venv/bin/python poc-task-mgmt/web_server.py       # → http://127.0.0.1:8000
```
Tarayıcıda aç, her POC'un **"▶ Çalıştır"** butonuna bas: gerçek framework yerelde koşar,
çıktı adım-adım + **sonuç rozetleri** (ör. Celery "fetch ×2 (BAŞTAN)", Temporal "process ×2",
Hermes "worker-B devraldı ✓") olarak görünür. Sunucu yalnız 127.0.0.1'e bağlanır ve sadece
beyaz-listedeki üç scripti çalıştırır. Temporal ilk çalıştırmada dev server indirir (~15-30s).

## Terminalden çalıştırma

```bash
# proje kökünden, venv ile:
.venv/bin/python poc-task-mgmt/hermes_real_poc.py     # gerçek Hermes kanban_db (SQLite kernel)
.venv/bin/python poc-task-mgmt/temporal_real_poc.py   # gerçek temporalio SDK + dev server
.venv/bin/python poc-task-mgmt/celery_real_poc.py     # gerçek celery worker + filesystem broker
# Airflow (ağır kurulum; DAG gerçek/çalıştırılabilir ama burada koşturulmadı):
#   pip install "apache-airflow==2.10.*" && airflow db migrate
#   cp poc-task-mgmt/airflow_dag.py $AIRFLOW_HOME/dags/ && airflow dags test daily_sales 2026-08-09
```

Bağımlılıklar: `temporalio`, `celery` (kuruldu). Hermes için ek kurulum yok — klonlu
`harnesses/hermes-agent` kaynağından `hermes_cli.kanban_db` doğrudan import edilir.
Temporal ilk çalıştırmada ephemeral dev server'ı indirir (ağ gerekir).

---

## Her POC ne kanıtlıyor

### 1) `hermes_real_poc.py` — GERÇEK agent-engine (Hermes kanban_db)
Gerçek `create_task → claim_task(CAS+lease) → CRASH → release_stale_claims → reclaim → complete_task`.
Gözlenen çıktı:
```
create → ready
worker-A claim → running (lease +899s);  worker-B aynı anda → None   (at-most-once)
CRASH (lease geçmiş, PID ölü) → release_stale_claims() = 1 → status=ready (otomatik recovery)
worker-B devraldı → run_id=2 → complete → done
task_runs: run#1 reclaimed, run#2 completed     events: created→claimed→reclaimed→claimed→completed
```
→ **Kendi SQLite durable çekirdeği**; worker çökse de iş kaybolmaz, başka worker devralır.

### 2) `temporal_real_poc.py` — GERÇEK Temporal (temporalio + dev server)
Workflow `fetch → process(RetryPolicy) → deliver`. Gözlenen çıktı:
```
workflow sonucu: 'order:4711:veri|işlendi|teslim'
fetch_data:1  process:2 (1 hata→otomatik retry→başarı)  deliver:1
event history: 23 durable event; ACTIVITY_TASK_SCHEDULED/STARTED/COMPLETED × 3
```
→ **Kaldığı yerden devam / exactly-once**: sadece `process` yeniden koştu; `fetch` **1 kez**.
Otomatik retry'lanan geçici hata history'ye yazılmaz (kompakt kalır).

### 3) `celery_real_poc.py` — GERÇEK Celery (worker + filesystem broker)
`.delay()` ile enqueue → ayrı worker süreci çeker → `self.retry()`. Gözlenen çıktı:
```
run_order sonucu: 'order:4711:veri|işlendi|teslim'
adımlar: attempt0:fetch, attempt0:process-HATA, attempt1:fetch, attempt1:process-OK, attempt1:deliver
fetch KAÇ KEZ koştu: 2
```
→ **Retry task'ı BAŞTAN koşturur** (`fetch` **2 kez**). "Kaldığı yerden devam" Celery'de
**otomatik değil** — idempotency/checkpoint senin işin. `acks_late` açık (çökerse redelivery).

### 4) `airflow_dag.py` — GERÇEK Airflow DAG (referans, burada koşturulmadı)
Statik DAG `fetch → process → deliver`, `schedule="0 8 * * *"`, `retries=2`, `catchup=False`.
→ **Task-seviyesi retry**: sadece hatalı düğüm (`process`) retry olur; `fetch` düğümü
**done kalır, yeniden koşmaz** (statik-DAG kısmi resume). En güçlü yanı **cron + backfill**.

---

## Çapraz karşılaştırma — asıl ders

Aynı `process` hatası, ama **retry'da `fetch` kaç kez koşuyor?** — dayanıklılık farkının özü:

| POC | Framework (gerçek) | Retry seviyesi | Çökme/retry sonrası | `fetch` (retry'da) |
|---|---|---|---|---|
| **Hermes** | `kanban_db` (SQLite) | task + breaker | `release_stale_claims` → otomatik reclaim → **handoff özeti** | başka worker devralır (run#2) |
| **Temporal** | `temporalio` + server | activity RetryPolicy | **replay**, biten activity atlanır | **1×** (sadece process 2×) |
| **Celery** | worker + broker | task `self.retry` | broker redelivery (`acks_late`) | **2×** (baştan) |
| **Airflow** | DAG (ref) | task `retries` | sadece hatalı düğüm retry | **1×** (düğüm ayrı) |

**Sonuç:** Temporal ve Airflow işin tamamlanan kısmını korur (fetch 1×) — Temporal
event-history replay'iyle, Airflow ayrı DAG düğümleriyle. Celery ise tüm task'ı baştan
koşturur (fetch 2×) → A-seviyesi "kaldığı yerden devam"ı **sen** kurarsın. Hermes kendi
SQLite çekirdeğiyle çökmeyi otomatik toparlar ve handoff özetiyle bağlamı taşır.

> Bu, `report/task-yonetimi-altyapi-karari.md` ve `report/task-management-sunum-ve-flowchart.md`
> dökümanlarındaki iddiaların **çalışan kanıtıdır**.
