# Ajanlarda Task Management — Derin Analiz ve Altyapı Kararı (Airflow / Celery / Temporal / brain_chat_V2)

> **Amaç (koçun brief'i):** Agent/workflow task'larının *oluşturulması, kuyruğa alınması, planlanması, çalıştırılması, retry edilmesi, durumunun takip edilmesi ve hata sonrası devam ettirilmesi* için hangi altyapının kullanılacağını belirlemek. Karşılaştırma ekseni: **task yönetimi · retry/recovery · state takibi · scheduling · concurrency · operasyonel karmaşıklık.** Adaylar: **Airflow, Celery, Temporal, mevcut brain_chat_V2 engine.**
>
> **Bu dökümanın açısı:** Kararı havada vermek yerine, aynı problemi *production'da çözmüş* beş ajan çatısının (Hermes, OpenCode, Codex, Claude Code, OpenClaw) kaynak kodunu söküp "bu işi gerçekte nasıl yapıyorlar" sorusunu yanıtlıyoruz — sonra bu bulguları Airflow/Celery/Temporal/brain_chat_V2 eksenine bindiriyoruz.
>
> Not: `brain_chat_V2` **internal** bir motor; bu repoda kaynağı yok. Onu, diğer dördünün ve beş ajanın ortaya koyduğu desenlerle *konumlandırıyoruz*; kesin skoru için kodunu görmem gerekir (§9).

---

## 0. Yönetici özeti (önce sonuç)

1. **Beş ajandan yalnızca biri (Hermes) tam bir dayanıklı (durable) task orchestrator'ı gerçekten uyguluyor.** Diğer dördü (OpenCode, Codex, Claude Code, OpenClaw) *interaktif* ajanlar: dayanıklılığı ya buluta (Codex cloud), ya oturum kaydına (rollout), ya git-snapshot'a devrediyorlar. "Fleet of autonomous agents" senaryosunda referans alınacak tek mimari **Hermes**.

2. **Hermes'in mimarisi, aslında SQLite üstüne kurulmuş bir "mini-Temporal + mini-Airflow" karışımı.** Durable state machine + retry/circuit-breaker + crash-recovery + DAG bağımlılık + cron scheduling + idempotency + oturum-resume, hepsi tek dosyada. Bu, "kendi motorunu yaz" (build) yolunun ne kadar ileri gidebileceğinin kanıtı.

3. **İki farklı retry katmanı olduğunu ayırt etmek şart.** (a) *API-çağrısı retry* (429/5xx/timeout → exponential backoff+jitter) — bunu **hepsi** yapıyor, hafif. (b) *Task/workflow retry* (bir işin baştan/kaldığı yerden yeniden çalıştırılması, crash sonrası) — bunu **sadece Hermes** yapıyor. Koçun asıl sorduğu (b).

4. **Karar için pratik çerçeve (detay §8):**
   - Adımlar **saatler-günler** sürüyor, süreç **insan onayı/harici olay** için duraklıyor, "tam olarak-bir-kez" (exactly-once) ve deterministik replay şart → **Temporal**.
   - İş **DAG/batch/veri-pipeline**, zamanlanmış (cron), operatör görünürlüğü (UI) önemli → **Airflow**.
   - Sadece **dağıtık task kuyruğu + worker havuzu + basit retry** lazım, workflow state'i sende → **Celery**.
   - `brain_chat_V2` zaten mesaj/oturum akışını yönetiyorsa ve tek ihtiyaç *dayanıklı kuyruk + retry + crash-recovery* ise → **Hermes-tarzı "SQLite/Postgres üstü hafif orchestrator"** (build) veya **Celery** (buy) en düşük operasyonel maliyet.

---

## 1. Problemi doğru parçalara ayırmak

Koçun tek cümlesi aslında **yedi ayrı yetenek** istiyor. Kıyaslamayı bunlar üzerinden yapacağız çünkü her aday bunların *farklı bir alt kümesinde* güçlü:

| # | Yetenek | Tam olarak ne demek | Kritik soru |
|---|---------|---------------------|-------------|
| 1 | **Oluşturma (create)** | Bir task'ı tanımlayıp sisteme sokmak | Task'ın *tanımı* nerede yaşıyor? Bellekte mi, diskte mi, DB'de mi? |
| 2 | **Kuyruğa alma (enqueue)** | Çalıştırılmayı bekleyen işleri sıraya koymak | Kuyruk **dayanıklı** mı (restart'ı atlatır mı)? Öncelik var mı? |
| 3 | **Planlama (schedule)** | *Ne zaman* çalışacağına karar | Cron/zamanlı tetik var mı? Bağımlılıkla mı tetikleniyor? |
| 4 | **Çalıştırma (execute)** | İşi bir worker'a atayıp koşturmak | Aynı işi iki worker kapabilir mi? (exactly/at-most-once) |
| 5 | **Retry** | Başarısızlıkta yeniden deneme | *Neyi* retry ediyoruz — API çağrısını mı, tüm task'ı mı? Kaç kez? |
| 6 | **State takibi** | Her işin o anki durumu + geçmişi | Durum bir **FSM** mi? Audit log var mı? |
| 7 | **Hata sonrası devam (recovery/resume)** | Worker çökerse iş kaybolmasın, kaldığı yerden/baştan sürsün | Crash *nasıl* tespit ediliyor? Lease/heartbeat var mı? |

> **En sık yapılan hata:** "retry var mı?" diye sorup evet cevabıyla yetinmek. OpenCode, Codex, Claude Code, hepsi "retry" yapıyor — ama sadece **#5a (API çağrısı)** seviyesinde. Koçun sorduğu **#7 (worker çöktü, task kaybolmasın)** apayrı bir şey ve onu yalnızca Hermes yapıyor.

---

## 2. İki okul: "Durable Orchestrator" vs "In-Process Session Engine"

Beş ajanı incelerken net bir ayrım çıktı:

### Okul A — Durable Orchestrator (dış-süreç, DB-destekli kuyruk)
Task **dış bir kalıcı depoda** (DB) yaşar. Bir **dispatcher** periyodik olarak "hazır" işleri **worker süreçlerine** dağıtır. Worker çökse bile task DB'de durur ve **başka bir worker devralır**. Bu tam olarak Airflow/Celery/Temporal'ın dünyası.
- **Tek örnek: Hermes** (Kanban kernel — SQLite).

### Okul B — In-Process Session Engine (tek-süreç, oturum-ağacı)
Task ≈ bir **konuşma oturumu**. Ajan bir döngüde çalışır; "alt-task" dediğimiz şey aslında bir **subagent oturumu** (aynı süreç içinde ya da arka-plan job). Süreç ölürse iş de ölür; kurtarma diskteki **oturum kaydını** (rollout) tekrar yükleyerek veya **git-snapshot**'tan dosyaları geri alarak yapılır. Dayanıklı *kuyruk yok*, cron yok, circuit-breaker yok — çünkü bunlar interaktif geliştirici araçları, otonom filo değil.
- **OpenCode, Codex, Claude Code, OpenClaw.**

> **Neden önemli:** Koçun "agent/workflow task'ları" ifadesi Okul A'yı tarif ediyor. Dolayısıyla mimari referansımız **Hermes**; diğerleri "interaktif ajan bu işi nasıl *hafifletiyor*" bağlamında öğretici (özellikle retry-backoff ve snapshot-revert desenleri).

---

## 3. Hermes — SQLite üstünde tam durable orchestrator (referans mimari)

Kaynak: `harnesses/hermes-agent/hermes_cli/kanban_db.py` (~10.3K satır), `kanban_swarm.py`, `cron/scheduler_provider.py`, `tools/delegate_tool.py`.

Hermes'in "Kanban" dediği şey bir board metaforu değil — **SQLite tabanlı bir iş kuyruğu + durum makinesi + dispatcher**'ın adı. Yedi yeteneğin **hepsini** karşılıyor.

### 3.1 Veri modeli (task'ın tanımı nerede yaşıyor → diskteki SQLite)

Beş tablo (`kanban_db.py:1185+`):

| Tablo | Rolü | Airflow/Temporal karşılığı |
|-------|------|----------------------------|
| `tasks` | Her task'ın **canlı durumu** (FSM state, öncelik, atanan, claim kilidi, sayaçlar) | DagRun / Workflow Execution |
| `task_runs` | Her **deneme (attempt)** ayrı satır — retry olunca çok satır | TaskInstance try'ları / Activity attempts |
| `task_links` | parent→child kenarları = **DAG bağımlılığı** | Airflow DAG edges |
| `task_events` | Değişmez **audit log** (claim, crash, gave_up…) | Temporal event history |
| `task_comments` | Worker'lar arası **blackboard** / koordinasyon | — (Hermes'e özgü) |
| `kanban_notify_subs` | Bitince isteği yapan insana **bildirim** (human-in-the-loop) | Airflow callbacks |

`tasks` tablosundaki kritik kolonlar (hepsi gerçek, `kanban_db.py:1185-1278`):
- **`status`** — 9-durumlu FSM: `triage → todo → scheduled → ready → running → blocked → review → done → archived` (`VALID_STATUSES`, satır 102).
- **`claim_lock` + `claim_expires`** — *lease* (kira). Kim, ne zamana kadar tuttuğu. Bu, exactly-once çalıştırmanın anahtarı (§3.3).
- **`consecutive_failures`** + **`max_retries`** — **circuit breaker** sayacı ve per-task override (satır 1208-1248).
- **`current_run_id`** — o an açık olan denemeye pointer.
- **`idempotency_key`** — aynı task'ı iki kez oluşturmayı engeller (§3.6).
- **`session_id`** — task'ı yaratan chat/agent oturumu → **resume** için (satır 1259).
- **`block_kind`** — *tipli* engel: `dependency | needs_input | capability | transient` (satır 125). Retry davranışını bu belirliyor.
- **`goal_mode` + `goal_max_turns`** — "Ralph loop": bir yargıç her turdan sonra "iş bitti mi?" diye bakar, bitene kadar aynı oturuma devam ettirir.
- **`workflow_template_id` + `current_step_key`** — v2 çok-adımlı workflow routing için (v1'de tutuluyor, henüz yönlendirmede kullanılmıyor).

`task_runs.outcome` alanının değer kümesi, hata taksonomisini tek yerde topluyor (satır 1324):
`completed | blocked | crashed | timed_out | spawn_failed | gave_up | reclaimed`.

### 3.2 Yaşam döngüsü (7 yeteneğin akışı)

```
create_task ──> todo ──(parents done? recompute_ready)──> ready
                                                            │
                              dispatcher tick (cron 60s)    │ claim_task (CAS)
                                                            ▼
                                                         running ──> done
                                                            │  ├─ complete_task
                                        crash/timeout/block │  ├─ block_task(kind)
                                                            ▼
                                    detect_crashed_workers / release_stale_claims
                                                            │
                                          _record_task_failure (++consecutive_failures)
                                          ┌──────────────────┴───────────────────┐
                                 sayaç < limit                         sayaç ≥ limit
                                    → ready (retry)                    → blocked (breaker tripti)
```

### 3.3 Çalıştırma + exactly-once: CAS ile claim (dağıtık kilit YOK)

Bu, tüm sistemin en zarif parçası. `claim_task` (`kanban_db.py:4226`) tek bir koşullu UPDATE:

```sql
UPDATE tasks
   SET status='running', claim_lock=?, claim_expires=?, started_at=COALESCE(started_at,?)
 WHERE id=? AND status='ready' AND claim_lock IS NULL
```
- `rowcount == 1` → **sen kaptın**. `rowcount == 0` → başkası kapmış, sessizce geç.
- SQLite'ın WAL yazma kilidi writer'ları serileştirdiği için **aynı anda en fazla bir claimer kazanır**. Zookeeper/Redis-lock **yok**; "no retry loops, no distributed-lock machinery" (dosya başlığı, satır 62-65).
- Ek güvenlik: claim anında **parent'lar `done` değilse** task `ready→todo`'ya geri düşürülür (DAG davetsiz-hazır durumuna karşı tek zorlama noktası, satır 4250-4266).

> Bu, **at-most-once dispatch**'i tek makinede kilitsiz sağlayan bir desen. Temporal aynı garantiyi dağıtık ortamda çok daha ağır makinayla verir; Celery ise varsayılanda **at-least-once** (dolayısıyla task'ların idempotent olması *senin* sorumluluğun).

### 3.4 Planlama (schedule): iki eksen

- **Bağımlılıkla tetik (data/DAG):** `recompute_ready` (`kanban_db.py:4135`) — tüm parent'ları `done/archived` olan `todo` task'ları `ready`'ye terfi ettirir. Airflow'un "upstream başarılı → downstream schedulable" mantığının aynısı.
- **Zaman ile tetik (cron):** `cron/scheduler_provider.py` — pluggable bir `CronScheduler` arayüzü. Yerleşik `InProcessCronScheduler` 60 sn'lik daemon-thread ticker; harici sağlayıcı (Chronos) webhook ile de kurulabilir. *Ne zaman* tetikleneceğine scheduler karar verir; *ne yapılacağı* (`run_job`/`_deliver_result`) ortaktır.

### 3.5 Retry / recovery — gerçek workflow-seviyesi (koçun asıl sorusu)

Üç ayrı kurtarma yolu var, üçü de **task'ı kaybetmeden** çalışıyor:

1. **Lease süresi dolması (`release_stale_claims`):** Bir `running` task'ın `claim_expires`'ı geçmişse (varsayılan **15 dk**, `DEFAULT_CLAIM_TTL_SECONDS`), sonraki dispatcher tick onu geri alır. Uzun işler `heartbeat_claim()` ile kirayı tazeler.
2. **Crash tespiti (`detect_crashed_workers`, `kanban_db.py:7518`):** Worker PID'i artık canlı değilse task `running→ready`'ye düşer, `crashed` event'i yazılır. TTL'i beklemez, anında. İki incelik:
   - **Protocol violation:** Worker rc=0 ile çıkmış ama `kanban_complete/block` çağırmamışsa → "işi yaptı ama kâğıt işini atladı" kabul edilip breaker ilk seferde tripler (sonsuz döngüyü keser).
   - **Rate-limit exit code:** Worker sağlayıcı kotasına toslayıp özel exit-code ile çıktıysa → **failure sayılmaz**, `ready`'ye döner, respawn kota penceresi açılana kadar ertelenir. (Uzun kota duvarı breaker'ı boşuna tripletmesin diye.)
3. **Circuit breaker (`_record_task_failure`, `kanban_db.py:7788`):** Her başarısızlık `consecutive_failures`'ı artırır; eşik (`max_retries` → dispatcher `kanban.failure_limit` → `DEFAULT_FAILURE_LIMIT=2`) aşılınca task otomatik `blocked`'a çekilir ve `gave_up` event'i yazılır. Başarı sayacı sıfırlar. Bu, "sonsuz retry fırtınası"nı engelleyen kısım.

**Tipli engel routing'i** retry'ı akıllı yapıyor (satır 105-134):
- `dependency` → `blocked` değil **`todo`**'ya (parent-gating otomatik açsın, insan/cron gerekmesin).
- `transient` → retry edilebilir geçici hata.
- `needs_input` / `capability` → gerçekten `blocked` (insan lazım). Cron bunları döngüye sokarsa `BLOCK_RECURRENCE_LIMIT=2` sonra `triage`'a atıp insanı zorlar.

### 3.6 State takibi + idempotency + resume

- **State:** `tasks.status` (canlı) + `task_events` (değişmez geçmiş) + `task_runs` (deneme-deneme sonuç/summary/error). Bu üçlü, Temporal'ın "event history"sinin hafif karşılığı.
- **Idempotency:** `idempotency_key` + unique index → aynı işi iki kez oluşturma girişimi mevcut task'a döner (swarm tekrar-çalıştırmada topolojiyi yeniden kurmaz, `kanban_swarm.py:129`).
- **Resume/handoff:** Her `task_runs` satırında **structured summary** var. Bir sonraki deneme veya downstream worker, `build_worker_context` ile önceki denemelerin özetini + parent handoff'larını + yorumları okuyarak **kaldığı bağlamdan** devam eder. "Hata sonrası devam" tam burada.

### 3.7 Concurrency + MAS topolojisi (swarm)

`kanban_swarm.py` ikinci bir scheduler *kurmuyor*; mevcut kernel'e küçük bir graf yazıyor:
```
planning root (hemen done, ayrıca shared blackboard)
   ├─ paralel specialist worker'lar (ready)   ← concurrency burada
   └─ verifier (worker'lar bitene kadar todo)
        └─ synthesizer (verifier bitene kadar todo)
```
Paralellik "birden çok `ready` worker + birden çok dispatcher claim"den doğar; koordinasyon **root task üzerindeki JSON yorumlar** (blackboard). Yeni servis yok — dashboard, notifier, dispatcher hepsi çalışmaya devam eder.

### 3.8 Hermes ⇄ altyapı adayları eşlemesi

| Hermes parçası | Temporal | Airflow | Celery |
|----------------|----------|---------|--------|
| `tasks.status` FSM + `task_events` | Workflow execution + event history | DagRun/TaskInstance state | task state (result backend) |
| `claim_task` CAS + lease | Task queue + sticky execution | Executor slot | worker prefetch + ack |
| `consecutive_failures` breaker | `RetryPolicy` + `maximumAttempts` | `retries` + `retry_delay` | `max_retries` + `autoretry_for` |
| `detect_crashed_workers` / heartbeat | Worker heartbeat + timeout | Zombie detection | `task_reject_on_worker_lost` / visibility timeout |
| `task_links` + `recompute_ready` | Child workflows / signals | DAG edges | Canvas (chain/group/chord) |
| cron `scheduler_provider` | Schedules | Scheduler (cron) | Celery Beat |
| `task_runs.summary` handoff | Workflow replay/continue-as-new | XCom | — |
| `idempotency_key` | Workflow ID reuse policy | `dagrun` uniqueness | idempotency (senin işin) |

**Sonuç:** Hermes = *"SQLite üstünde, tek makine için, insan-döngülü Temporal + Airflow hafif melezi."* Tek makinede bunu bu kadar açık koda dökmüş olması, "build" seçeneğinin fizibıl olduğunun somut kanıtı.

---

## 4. OpenCode — in-process session engine (durable kuyruk YOK)

Kaynak: `harnesses/opencode/packages/opencode/src/{tool/task.ts, session/retry.ts, snapshot/index.ts, session/revert.ts, util/queue.ts}`.

- **Task = subagent oturumu (`tool/task.ts`).** `Task` tool'u bir alt-ajanı çağırır: `subagent_type`, `prompt`, opsiyonel `task_id`.
  - **Foreground (varsayılan):** sonucu bekler, bloklar.
  - **Background (`background=true`, deneysel):** `BackgroundJob` servisi ile async koşar, bitince ana oturuma **synthetic mesaj** enjekte ederek haber verir. "Poll etme, uyuma, işi tekrarlama" talimatı tool açıklamasına gömülü.
  - **Resume:** `task_id` verilirse **aynı subagent oturumu** kaldığı yerden sürer (yeni oturum açmaz). Bu bir "resume" ama *oturum-içi*, DB-kuyruk değil.
  - **Depth limit:** `subagent_depth` (varsayılan 1) — özyinelemeli alt-ajan patlamasını sınırlar.
- **Retry = sadece API çağrısı (`session/retry.ts`).** Exponential backoff (`RETRY_INITIAL_DELAY=2000`, `factor=2`, `retry-after` header'a saygı), retryable pattern seti (429/500/502/503/504/524, rate-limit, network/timeout). **Task-seviyesi retry değil** — LLM isteği patlarsa tekrar dener, o kadar.
- **Recovery = git-snapshot (`snapshot/index.ts`).** Çalışma ağacını gölge bir git deposunda `track/patch/restore/revert/diff` ile snapshot'lar. Ajan yanlış yaptıysa **dosya durumunu** geri alırsın (`session/revert.ts` de mesaj/turn geri alır). Bu, "workflow checkpoint"in **dosya-sistemi** karşılığı; state DB'si değil.
- **Yok:** dayanıklı kuyruk, cron, circuit-breaker, lease/claim, crash-recovery (süreç ölürse arka-plan job da ölür).

**Konum:** OpenCode interaktif bir kodlama ajanı. "Task management"i = oturum ağacı + arka-plan job + git-checkpoint. Airflow/Temporal karşılığı **yok**; en fazla "Celery'nin `.delay()` + result bekleme"sinin in-process, kalıcı-olmayan bir taklidi.

---

## 5. Codex — rollout persistence + cloud delegation

Kaynak: `harnesses/codex/codex-rs/{rollout/*, codex-client/src/retry.rs, cloud-tasks/*}`.

- **Persistence/resume = rollout (`rollout/` crate'i).** Her oturum JSONL olarak diske yazılır (`recorder`, `compression`), `state_db` (SQLite) bunların **index**'ini tutar (listeleme/arama/telemetri). Bir oturum **fork/resume** edilebilir; "hata sonrası devam" = **rollout'u yeniden yükleyip** kaldığı yerden sürdürmek. Bu bir **event-sourcing/replay** deseni (Temporal'ın felsefesine en yakın parça) — ama görev *kuyruğu* değil, oturum *kaydı*.
- **Retry = HTTP seviyesi (`codex-client/src/retry.rs`).** `RetryPolicy { max_attempts, base_delay, retry_on: {429, 5xx, transport} }`, `backoff = base·2^(n-1)·jitter(0.9–1.1)`. Yine **API-çağrısı retry'ı**, task retry'ı değil.
- **cloud-tasks = uzağa delege (`cloud-tasks/` crate'i).** Bu bir **CloudBackend istemcisi** (`codex_cloud_tasks_client::{CloudBackend, TaskId, TaskStatus}`). Task'ı OpenAI'nin **bulut** ortamında koşturur; yerel taraf task listeler, diff gösterir (`scrollable_diff`), sonucu `ApplyJob` ile uygular. Yani Codex'in "dayanıklı, uzun-süren task yönetimi" cevabı: **kendisi kuyruk kurmaz, buluttaki servise devreder.**
- **Yok (yerelde):** kendi durable kuyruğu, cron, circuit-breaker.

**Konum:** İki net desen veriyor — (1) **rollout = replay ile resume** (Temporal-vari düşünce), (2) **ağır işi managed bir backend'e devret** ("buy/hosted" felsefesi). İkisi de brain_chat_V2 için düşünülebilir seçenekler.

---

## 6. Claude Code — in-process loop + subagent izolasyonu (kapalı kaynak)

Kaynak: docs + Agent SDK + `report/claude-code-harness.md` (kapalı kaynak; gözleme dayalı).

- **Task = ajan döngüsü + `Task`/subagent tool'u.** Alt-iş **ayrı context penceresinde** koşar, ana pencereye **sadece özet** döner (context izolasyonu; bir tür "map-reduce" ama dayanıklılık için değil, bağlam-basıncı için).
- **Retry:** API-çağrısı seviyesinde (SDK). **Task-seviyesi durable retry yok.**
- **State/recovery:** oturum içi; `PreCompact/PostCompact` hook'ları, mikro-kompaksiyon (büyük çıktı diske). Dayanıklı iş kuyruğu **yok** — Claude Code bir CLI/geliştirici aracı, otonom filo motoru değil.

**Konum:** Referans-dışı (interaktif). Sadece "büyük task'ı subagent'a bölüp ana bağlamı koru" deseni ilgili.

---

## 7. OpenClaw — SQLite state + cron, ama hafif

Kaynak: `harnesses/openclaw/` (`scripts/bench-sqlite-state.ts`, `scripts/control-ui-mock-cron.ts`, extensions/*).

- Oturum durumunu **SQLite**'ta tutuyor (bench-sqlite-state) ve bir **cron** kavramı var (mock-cron), ama Hermes'teki gibi bir claim/lease + circuit-breaker + DAG kernel'i **yok**. Session-key ağacı + focus/routing ile çoklu-ajan yönlendirir.
- **Konum:** Okul B'ye yakın; "SQLite'ı state store olarak kullan" fikrini destekliyor ama tam orchestrator değil.

---

## 8. Karşılaştırma: 5 ajan × 7 yetenek

| Yetenek | **Hermes** | OpenCode | Codex | Claude Code | OpenClaw |
|---------|:----------:|:--------:|:-----:|:-----------:|:--------:|
| Oluşturma | DB satırı (durable) | oturum | oturum (rollout) | oturum | oturum (SQLite) |
| **Dayanıklı kuyruk** | ✅ SQLite `tasks` | ❌ | ❌ (buluta delege) | ❌ | ~ (state var, kuyruk yok) |
| Scheduling (cron) | ✅ pluggable | ❌ | ❌ | ❌ | ~ mock-cron |
| Bağımlılık/DAG | ✅ `task_links` | ❌ | ❌ | ❌ | ❌ |
| Exactly/at-most-once | ✅ CAS+lease | ❌ | (backend) | ❌ | ❌ |
| **Task-seviyesi retry** | ✅ breaker+taksonomi | ❌ | ❌ | ❌ | ❌ |
| API-çağrısı retry | ✅ | ✅ backoff | ✅ backoff+jitter | ✅ (SDK) | ✅ |
| Crash-recovery | ✅ PID+TTL+heartbeat | ❌ (job ölür) | ~ rollout replay | ❌ | ❌ |
| State/audit | ✅ events+runs | ~ mesajlar | ✅ rollout JSONL | ~ | ~ |
| Resume/handoff | ✅ run summary | ~ task_id | ✅ rollout | ❌ | ~ |
| Concurrency | ✅ swarm+dispatcher | ~ bg-job | (bulut) | ~ subagent | ~ |
| Op. karmaşıklık | Orta (tek süreç+SQLite) | Düşük | Düşük (yerel) | Düşük | Düşük |

**Okunuş:** Otonom, hata-toleranslı, uzun-süren task filosu istiyorsak referans **yalnızca Hermes**. Diğerleri "interaktif tek kullanıcı" senaryosuna optimize; onlardan alınacak dersler: **API-retry backoff+jitter** (hepsi), **git-snapshot checkpoint** (OpenCode), **rollout/replay ile resume** (Codex), **managed backend'e delege** (Codex cloud).

---

## 9. Asıl karar: Airflow vs Celery vs Temporal vs brain_chat_V2

Ajanlardan öğrendiğimiz desenleri koçun altı eksenine bindirelim. (brain_chat_V2 skorları **varsayımsal** — kodunu görünce netleşir, §10.)

| Eksen | **Airflow** | **Celery** | **Temporal** | **brain_chat_V2 (mevcut)** |
|-------|-------------|-----------|--------------|----------------------------|
| **Task yönetimi** | DAG-merkezli; task = operatör düğümü. Batch/pipeline için ideal | Kuyruk + worker; task = fonksiyon çağrısı. Kuyruk mantığı hazır | Workflow = kod; deterministik, uzun-ömürlü. En güçlü model | *Mesaj/oturum akışı yönetiyor; iş "task" soyutlaması muhtemelen zayıf* |
| **Retry/recovery** | task-seviyesi `retries`+delay; zombie reaping | `max_retries`+`autoretry_for`; worker-lost ele alınır (ayar) | **En güçlü:** activity retry + workflow **replay** (crash'ten deterministik devam) | *API-retry olabilir; workflow-replay/crash-devam muhtemelen yok* |
| **State takibi** | Metadata DB + zengin UI (görsel) | Result backend (Redis/DB); UI zayıf (Flower) | Event history + Web UI; **tam denetlenebilir** | *Oturum/mesaj state'i var; iş-seviyesi FSM/audit belirsiz* |
| **Scheduling** | **En güçlü:** cron + veri-farkındalıklı, backfill | Celery Beat (basit cron) | Schedules (iyi) | *Muhtemelen yok/harici* |
| **Concurrency** | Executor havuzları (Celery/K8s executor) | **Native** worker havuzu + prefetch | Task queue + worker fleet; sticky | *Süreç-içi; yatay ölçek belirsiz* |
| **Op. karmaşıklık** | **Yüksek** (scheduler+web+DB+executor+broker) | **Orta** (broker+worker+result backend) | **Yüksek** (Temporal cluster/Cloud + SDK worker) | **Düşük** (zaten var; sıfır yeni bağımlılık) |

### Karar ağacı (net tavsiye)

1. **İş uzun-süren, insan/harici-olay için duruyor, "tam-bir-kez" ve deterministik replay şart mı?**
   → **Temporal.** (Ajan dünyasında bunun izdüşümü: Hermes'in state-machine+resume'u + Codex'in rollout-replay'i. Temporal ikisini birden, dağıtık ve battle-tested yapar.) Bedeli: en yüksek öğrenme+operasyon eğrisi.

2. **İş bir veri/batch DAG'ı, zamanlı tetik + operatör görünürlüğü (UI) merkezi mi?**
   → **Airflow.** (İzdüşüm: Hermes `task_links` + `recompute_ready` + cron.) Bedeli: ağır kurulum; uzun-süren "insan bekleyen" adımlar için ideal değil.

3. **Sadece dağıtık kuyruk + worker havuzu + basit retry lazım, workflow state'ini kendin tutacaksın?**
   → **Celery.** (İzdüşüm: Hermes'in claim/lease/worker kısmı, ama Celery hazır veriyor.) En hızlı benimseme; ama exactly-once ve crash-devam **senin** işin.

4. **brain_chat_V2 zaten oturum akışını yönetiyor ve tek eksik "dayanıklı kuyruk + task retry + crash-recovery" mı?**
   → İki yol:
   - **Buy:** brain_chat_V2 + **Celery** (kuyruk/retry/concurrency'i devret, oturum akışı sende kalsın). En düşük yeni-karmaşıklık.
   - **Build:** brain_chat_V2 içine **Hermes-tarzı hafif orchestrator** (Postgres/SQLite `tasks`+`task_runs` tabloları, CAS-claim, lease+heartbeat, consecutive_failures breaker, cron tick). Hermes bunun tam çalışan referans implementasyonu — 10K satırın çekirdeği ~1-2K satıra sığar. Sıfır yeni servis bağımlılığı, tam kontrol; ama bakım sende.

### Neden "otomatik Temporal değil"

Temporal en güçlü modeldir ama **ajan iş yükleri için iki gerçek**: (a) LLM adımları zaten *non-deterministik* — Temporal'ın replay-determinizmini korumak için her LLM/tool çağrısını activity'e sarman ve yan-etkileri dışarı itmen gerekir (disiplin maliyeti). (b) Cluster/Cloud operasyonu ekip için yeni bir yük. Hermes'in kanıtladığı şey: **tek makinede SQLite + CAS-claim + lease + breaker**, ajan-task'larının %90'ı için Temporal'ın verdiği garantilerin *pratik* karşılığını çok daha ucuza veriyor. Temporal'a, ancak *çok makineli, yüksek-hacimli, katı exactly-once/uzun-durma* gereksinimi netleştiğinde geç.

---

## 10. brain_chat_V2 için sonraki adım (kodunu görmem lazım)

Kesin skor için brain_chat_V2'de şunları arayacağım (bu dökümanı ona göre güncellerim):
- Task **kalıcı mı** saklanıyor (DB tablosu var mı) yoksa süreç-belleğinde mi?
- Bir worker çökerse iş **ne oluyor** (kayboluyor mu, devralınıyor mu)?
- Retry **hangi seviyede** (API çağrısı mı, tüm iş mi)? Sayaç/limit var mı?
- **Idempotency** ve **exactly/at-least-once** garantisi ne?
- Zamanlı tetik (cron) ihtiyacı var mı, ne sıklıkta?
- Beklenen **eşzamanlılık** (kaç paralel iş) ve **süre** (saniye mi, saat mi)?

Bu altı cevap, §9'daki karar ağacında brain_chat_V2'yi tam yerine oturtur.

---

## Ek: kaynak referansları (doğrulanabilirlik)

- Hermes durable kernel: `harnesses/hermes-agent/hermes_cli/kanban_db.py` — şema `:1185`, FSM `:102`, block taksonomi `:105-134`, lease sabitleri `:213-239`, `recompute_ready :4135`, `claim_task :4226` (CAS), `detect_crashed_workers :7518`, `_record_task_failure :7788`, `DEFAULT_FAILURE_LIMIT=2 :6742`.
- Hermes swarm/cron: `hermes_cli/kanban_swarm.py`, `cron/scheduler_provider.py`.
- OpenCode: `packages/opencode/src/tool/task.ts` (fg/bg subagent, resume), `session/retry.ts` (backoff), `snapshot/index.ts` (git checkpoint), `session/revert.ts`.
- Codex: `codex-rs/rollout/src/lib.rs` (+`state_db.rs`, resume), `codex-client/src/retry.rs` (HTTP backoff+jitter), `cloud-tasks/src/lib.rs` (CloudBackend delege).
- Claude Code: `report/claude-code-harness.md` (kapalı kaynak, gözlem).
- OpenClaw: `harnesses/openclaw/scripts/{bench-sqlite-state.ts, control-ui-mock-cron.ts}`.
