# Macro (macro-inc/macro) — Bizim Caseler Açısından Baştan Sona Analiz

**Kaynak:** `github.com/macro-inc/macro` · lokal klon `harnesses/macro` · 250 MB, 13.253 dosya
**Yöntem:** Kaynak kod okuması. Her iddia bir dosya/satıra dayanıyor. Çalıştırmadım (Rust + Postgres + AWS gerekiyor) — yani **kod analizi**, ölçüm değil.

---

## 0. Bu nedir, neden bizi ilgilendiriyor

Macro bir **ürün**: e-posta + mesaj + doküman + task + ajan + CRM'i tek arayüzde birleştiren "şirket işletim sistemi". SolidJS + Rust, ~15 kişilik ekip 2 yıl kendi üstünde kullanmış, MIT değil ama kaynağı açık.

Bizi ilgilendiren yanı: **173 Rust crate'inin içinde tam teşekküllü bir ajan altyapısı var** — ve bizim üç casemizin (tool-trace compaction · task management · ajan harness'ı) hepsine ürün ölçeğinde bir cevap veriyor. Üstelik cevapları çoğu yerde **bizimkinden farklı bir felsefeye** dayanıyor; asıl değeri bu.

**Bizim incelediğimiz diğer sistemlerden farkı:** Hermes/Codex/OpenCode/Claude Code hepsi *kodlama ajanı*. Macro bir *iş yazılımı*. Ajanın dokunduğu şey dosya değil; e-posta, kanal mesajı, doküman, task, CRM kaydı. Tool'ları da buna göre.

### Bir bakışta harita

```
apps/           web (SolidJS) · docs
crates/  173    agent · ai_toolset · ai_tools · anthropic · prompt · skills
                system_skills · memory · task_dedup · projects · reminders …
services/  44   scheduled_action · mcp_service · coding-agent-worker
                ai-editing-worker · sync-service · websocket-service …
```

Ajanla doğrudan ilgili çekirdek:

| crate | satır | ne |
|---|---:|---|
| `agent` | 5.268 | ajan döngüsü, model router, hook köprüsü, stream |
| `ai_toolset` | — | tool sözleşmesi, şema üretimi, tool search primitifleri |
| `ai_tools` | — | tool kompozisyonu, Subagent, SearchTools, LoadTools |
| `prompt` | — | 15 parça statik sistem-promptu fragmanı |
| `skills` / `system_skills` | — | kullanıcı-yazımı (markdown) + kod-tanımlı skill'ler |
| `task_dedup` | 5.369 | task tekrar tespiti (embedding→rerank→LLM judge) |
| `services/scheduled_action` | 2.464 | **cron + claim/lease ile zamanlanmış ajan görevleri** |

---

# CASE 1 — Tool-trace compaction

## En önemli bulgu: post-hoc compaction katmanı YOK

Bütün `crates/agent` ve `crates/ai_toolset` içinde **geçmişi sıkıştıran, tool sonuçlarını buda(yan/dedup'layan/özetleyen) tek bir katman yok.** `agent_loop.rs`'in `Session`'ı `history: Vec<Message>` tutuyor ve `send_message` her çağrıda `self.history = messages;` diyerek onu **çağıranın verdiğiyle değiştiriyor**. Yani:

> Macro'da bağlam yönetimi ajan katmanında değil; **tool katmanında ve tool tasarımında.**

Bu bizim 13 stratejiyle incelediğimiz manzaraya doğrudan bir karşı-tez. Onlar "sonucu üretip sonra kısalt" yerine **"zaten büyük üretme"** diyor.

Bunu üç ayrı mekanizmayla yapıyorlar:

## 1.A — Tool çıktısında yayım-anı tavan + omission ledger

`crates/channels/src/inbound/toolset/types.rs`:

```rust
pub(crate) const DEFAULT_LIMIT: u16 = 25;
pub(crate) const MAX_LIMIT: u16 = 100;
pub(crate) const DEFAULT_MAX_CHARS_PER_MESSAGE: usize = 4_000;
pub(crate) const MAX_CHARS_PER_MESSAGE: usize = 16_000;
```

Kırpma tool'un **kendi içinde**, sonucu üretirken oluyor (bizim taksonomimizde **at-emission tetikleyici**). Ama asıl zarif kısım kırpma değil, **neyin atıldığının yapılandırılmış olarak bildirilmesi**:

```rust
pub enum ToolOmissionKind {
    OlderMessages,      // pencerenin dışında daha eski mesaj var
    NewerMessages,      // daha yeni mesaj var
    ThreadReplies,      // thread yanıtları atlandı
    TruncatedContent,   // içerik kesildi
}

pub struct ToolOmission {
    pub kind: ToolOmissionKind,
    pub message_id: Option<Uuid>,
    pub thread_id: Option<Uuid>,
    pub count: Option<i64>,      // kaç tane atlandı (biliniyorsa)
    pub cursor: Option<String>,  // ← devam etmek için imleç
}
```

**`cursor` alanı kritik.** Bizim compaction stratejilerimiz "[7] oversized, kesildi" gibi bir not bırakıyor — ajan kesilen kısma **erişemiyor**. Macro'da not bir *kurtarma işaretçisi*: ajan isterse aynı tool'u cursor ile çağırıp devamını alabilir.

Bu, kayıplı sıkıştırmayı **kayıpsız erteleme**ye çeviriyor. Bizim USTA-REHBER'de "compacted damgası" diye anlattığımız şeyin çalışır hâli.

## 1.B — Tool TANIMLARINI bağlamdan çıkarma (tool search)

Bizim hiç ele almadığımız eksen. Bağlamı şişiren yalnız tool *sonuçları* değil, tool *şemaları* da. Macro bunu şöyle çözüyor (`ai_toolset/src/tool_search.rs`, `agent_loop.rs`):

```
toolset.searchable_catalog()   → her istekte GÖNDERİLMEYEN tool'ların kataloğu
SearchTools (tool)             → model sorgu yazar, katalogda eşleşme arar
ToolLoader                     → eşleşenleri loaded_buffer'a iter
RegisterFn                     → canlı tool sunucusuna kaydeder
                               → SONRAKİ turda çağrılabilir hâle gelir
```

Yani MCP tool'ları (kullanıcı başına değişen, sayısı büyüyen) **her istekte gönderilmiyor**; model önce arıyor, sonra yükleniyor. Sistem promptuna yalnız hangi entegrasyonların *bağlı* olduğu enjekte ediliyor (`prompt::connected_toolsets::render`).

İki ince ayrıntı, ikisi de gerçek hatadan öğrenilmiş:

```rust
// Aynı tool iki kez kaydedilirse çift tool tanımı gider → 400
let loaded_names: Arc<Mutex<HashSet<String>>> = ...;
if !loaded_names.lock().insert(tool.name.clone()) { continue; }
```

```rust
pub(crate) const MAX_INVALID_TOOL_CALL_RETRIES: usize = 2;
```

İkincisi bizim de yaşadığımız bir hatanın çözümü. Yorumu aynen şöyle:

> *"rig'in varsayılan fail-fast'i tüm stream'i öldürüyordu ve kullanıcıya bir tool çağrısı duyurup sonra sessizleşen bir tur olarak görünüyordu."*

Çözüm: model yüklenmemiş bir tool çağırırsa **hata verme, tool'u yükle ve modele tekrar dene de**:

```rust
InvalidToolCallHookAction::retry(format!(
    "The tool `{}` exists but was not loaded when you called it. \
     It is loaded now — call it again with the same arguments.", ...))
```

Hiç var olmayan bir ad içinse:

```rust
"Unknown tool `{}`: no tool with that name exists ... Use `SearchTools` to find the
 right tool, or continue without it."
```

**Bu bir retry bütçesi (2) ile sınırlı ve retry'lar çok-tur derinliğinden düşüyor** — sonsuz halüsinasyon döngüsü yok.

## 1.C — Tool tasarım doktrini (asıl kaldıraç)

`crates/ai_toolset/TOOL_DESIGN.md` — 133 satır, ve crate'in `CLAUDE.md`'si tek cümle: *"You are not allowed to EVER change this crate."*

Bizim casemizi doğrudan ilgilendiren dört ilke:

**"Tools Are Not Endpoints"** — endpoint↔tool 1:1 eşlemesi *en yaygın hata*. Tool'lar iş akışı etrafında tasarlanmalı.

**"Filters Over Pagination"** — *"List endpoint'leri AI'a ASLA pagination parametresi açmamalı."* Gerekçe bizim ölçtüğümüzle aynı: ajan ne zaman duracağını bilmez, kısmi sonuçlara bağlam harcar, ilgili kaydı kaçırır. Yerine semantik filtreler + makul varsayılan + gerekirse "234 sonucun 50'si gösteriliyor" özeti.

**"Return What's Needed, Not Everything"** — `response_format: concise | detailed` önerisi: muhakeme için özet, aşağı akış işlemi için tam kimlikler.

**"Curate, Don't Expose"** — *"Daha çok tool daha iyi sonuç garantilemez; her tool ajanın dikkati için yarışır."*

Toplam 58 `add_tool` çağrısı var (12 domain crate'ine dağılmış). 119 tool'lu bizim ürün setimizle karşılaştırıldığında bilinçli bir kısıtlama.

## Case 1 — karşılaştırma tablosu

| eksen | biz (POC, 13 strateji) | Macro |
|---|---|---|
| ana strateji | **sonrası** — üret, sonra sıkıştır | **öncesi** — zaten küçük üret |
| tetikleyici | at-emission + at-threshold | yalnız at-emission (tool içinde) |
| kayıp bilgi | not bırakılır, **erişilemez** | omission + **cursor** → erişilebilir |
| tool tanımları | hepsi her istekte | **tool search** ile talep üzerine |
| dedup / staleness | var (bizim ayırt edici özelliğimiz) | **yok** |
| LLM özeti | 3 stratejide var | **yok** |
| geçmiş sıkıştırma | var | **yok** |

**Dürüst sonuç:** Macro'nun yaklaşımı bizimkinin rakibi değil, **tamamlayıcısı**. Onların yolu uzun sohbetlerde yetmez (geçmiş sınırsız büyür, `max_turns=16` bunu sınırlıyor ama çözmüyor). Bizim yolumuz da tool'lar kötü tasarlandığında ancak hasarı azaltır. **İkisi birlikte doğru cevap:** tool'u küçük çıktı verecek şekilde tasarla (Macro), kalan birikimi strateji ile yönet (biz).

---

# CASE 2 — Task management

## Önce bir kavram ayrımı

Macro'da "task" **iki ayrı şey** ve karıştırılırsa yanlış karşılaştırma yapılır:

| | Macro | biz |
|---|---|---|
| **A · Ürün task'ı** | insanın Linear'daki gibi açtığı iş kaydı | yok |
| **B · Yürütme task'ı** | `scheduled_action` — zamanlanmış ajan koşusu | board'daki düğüm |

Bizim board'ımızın karşılığı **B**. **A** ise bizde hiç olmayan, tamamen farklı bir problem — ama çözümü öğretici.

## 2.A — Ürün task'ı: tekrar tespiti (`task_dedup`, 5.369 satır)

Bir ekip aynı işi iki kez açtığında yakalayan tam bir ML hattı:

```
NewTask
  → OpenAI text-embedding-3-small ile gömme
  → Postgres/pgvector benzerlik araması   (vector_candidate_limit)
  → Cohere reranker
  → LLM judge (yapılandırılmış çıktı)      → JudgeResult
  → aktif duplicate kaydı + canlı bildirim
```

Bizi ilgilendiren yanı **judge promptu** — bir sınıflandırma görevinin nasıl yazıldığının çok iyi bir örneği:

> *"Duplicate demek, birini tamamlamak diğerini büyük ölçüde tamamlar ve büyük bir ek karar/iş kalmaz."*

Karar çerçevesi olarak beş boyut tanımlıyor: kullanıcıya görünen sonuç · birincil eylem · değişen nesne · tetikleyici · gereken uygulama alanı. Sonra **false olması gereken beş durumu açıkça sayıyor** (aynı ürün alanı, aynı varlık farklı davranış, ortak anahtar kelime, aynı büyük projenin parçası, aynı ekranda uygulanabilir olma). Ve **somut bir false örneği** koyuyor:

> A: "E-postaları bugün/son 7 gün diye ayır (Superhuman gibi)"
> B: "E-postalar için superhuman j/k tuşlarını geri getir"
> → false: biri zaman-bazlı gruplama, diğeri klavye gezinme. Aynı özellik alanı ve aynı ilham kaynağı, farklı sonuç.

Ayrıca `crates/task_dedup/src/eval/` — **kendi eval korpusu var** (`pull_task_corpus.rs`, `expand_eval_corpus.rs`). Yani bu prompt tahminle değil ölçümle ayarlanmış.

**Bizim için ders:** biz LLM'e karar verdirdiğimiz her yerde (planlayıcı, router) böyle bir negatif-örnek + eval korpusu disiplinimiz yok. Bugünkü testlerimizde planlayıcının DAG kenarını yanlış kurduğunu bulduk; çözümümüz prompt'a uyarı eklemek oldu — **eval'le doğrulamadık**.

## 2.B — Yürütme task'ı: `scheduled_action` (bizim eksik eksenimiz)

Bu, koçun 6 ekseninden **bizde hiç uygulanmamış olanı**: scheduling. Macro'nunki tam ve bizim board'la neredeyse birebir eşleşiyor.

### Veri modeli

```rust
pub const MAX_ACTION_TIME: Duration = Duration::minutes(20);   // ← lease

pub struct ScheduledAction {
    pub id: Option<Uuid>,
    pub owner: MacroUserIdStr<'static>,
    pub schedule: Schedule,          // cron (6-7 alan: sec min hour dom mon dow [year])
    pub kind: ActionKind,            // şu an yalnız Agent
    pub timezone: Tz,                // ← zaman dilimi ayrı tutuluyor
    pub task: Value,                 // opak JSON payload
    pub claimed: Option<DateTime<Utc>>,   // ← claim damgası
    pub next_run_at: DateTime<Utc>,       // cron'dan yazma anında türetilir
    pub enabled: bool,
}

pub struct AgentTask { pub model: String, pub prompt: String, pub user_prompt: String }
```

`next_run_at`'in **yazma anında** hesaplanmasının gerekçesi kodda yazıyor: *"UI cron'u kendisi ayrıştırmadan 'sonraki koşu'yu gösterebilsin."* Küçük ama iyi bir karar — türetilmiş değeri okuma anında değil yazma anında sabitlemek.

### Claim: tek atomik UPDATE

```rust
async fn claim_action(&self, id: &Uuid) -> Result<()> {
    let now = Utc::now();
    let stale_threshold = now - MAX_ACTION_TIME;
    let result = sqlx::query!(r#"
        UPDATE scheduled_action
        SET claimed = $1, updated_at = now()
        WHERE id = $2
          AND (claimed IS NULL OR claimed < $3)
    "#, now, *id, stale_threshold).execute(&self.pool).await?;

    if result.rows_affected() == 0 {
        return Err(anyhow::Error::new(AlreadyRunningError { action_id: *id }));
    }
    Ok(())
}
```

**Bizim `claim_next` + `recover_stale` ikilisinin tek satırlık hâli.** Kritik fark:

| | biz | Macro |
|---|---|---|
| claim koşulu | `WHERE claim_lock IS NULL` | `WHERE claimed IS NULL OR claimed < stale_threshold` |
| bayat işi kurtarma | **ayrı** `recover_stale()` süpürmesi | **claim predikatının içinde** |
| ölü PID tespiti | `os.kill(pid, 0)` ile ek kontrol | yok — yalnız zaman |

**Ders:** Bizde `recover_stale()` ayrı bir adım olduğu için *çağrılmayı unutulabilir* bir yol. Nitekim bugün Temporal'da tam da bu sınıftan bir hata bulduk (bayat claim). Macro'da böyle bir yol yok — **bayatlık kontrolü claim'in kendisinde**, atlanması imkânsız. Bu bizim board'a uygulanabilir, sade bir sadeleştirme.

Karşılığında bizde olan, onlarda olmayan: **PID canlılık kontrolü.** Bizim `recover_stale`, lease dolmasını beklemeden ölü süreci tespit edip 20 dakika beklemeden devralabiliyor. Macro'da worker ölürse task **20 dakika** boyunca kimse alamaz.

### Dispatcher: bilinçli yavaşlatma

```rust
const BATCH_SIZE: i64 = 10;
const BATCH_MIN_DURATION: Duration = Duration::from_secs(30);
```

Yorumdaki gerekçe çok iyi:

> *"Bu, polling döngüsünü tempolar ve backlog varken eş örneklerin iş kapmasına şans verir (tek bir örneğin kuyruğu boşaltmasını engeller)."*

Yani **kasten yavaş**. Bizim dispatcher'ımız açgözlü: `_dispatch_own` her turda claim edebildiğini alır. Tek süreçte sorun değil ama çok worker'lı gerçek kurulumda bir worker'ın kuyruğu süpürmesi mümkün. Bugün concurrency testinde 6 süreçle 4,9× hızlanma ölçtük — ama **adil dağılım** ölçmedik.

Bir tasarım kararı daha:

> *"`DispatchEvent`'ler alınır ve atılır — polling her tick'te durumu doğrudan DB'den okur, dolayısıyla create/update/delete olayları gereksizdir."*

Yani olay akışını **bilerek yok sayıyorlar**; tek doğruluk kaynağı DB. Bellekte cron tutmuyorlar. Bu, çok örnekli çalışmayı bedavaya getiriyor.

### Koşu geçmişi ve canlı durum

```rust
pub struct ActionExecutionRecord {
    pub action_id: Uuid,
    pub resource_id: Option<String>,   // üretilen kaynak (ör. sohbet thread'i)
    pub start_time, pub end_time,
    pub is_success: bool,
    pub result: Value,
}

pub enum ScheduledActionUpdate {
    Started { owner, action_id, chat_id },
    Stopped { owner, action_id, chat_id, is_success },
}
```

Her koşu kalıcı kayda giriyor, canlı durum WebSocket ile sahibine yayınlanıyor. **Bizim `board.events()` olay günlüğümüzün ürünleşmiş hâli** — üstelik "hangi kaynağı üretti" (`resource_id`) alanıyla. Bizde koşunun çıktısına giden bir işaretçi yok; sohbet özetinden okumak gerekiyor.

### Ajan görevi ne yapıyor

`inprocess_executor/agent_task.rs`: zamanlanan iş bir **sohbet açıyor** (`create_run_chat`), kullanıcı mesajını kaydediyor, `all_tools()` ile tool döngüsünü koşturuyor, konuşmayı saklıyor, sonucu bildirim olarak gönderiyor. Ayrıca `fetch_user_memory` ile **kullanıcı hafızasını** prompt'a katıyor.

Yani "her sabah 8'de şunu yap" dediğinde, çıktı bir log değil **gerçek bir sohbet thread'i** — kullanıcı içine girip devam edebiliyor. Bu, otomasyon çıktısını insanla aynı yüzeyde tutan güzel bir karar.

## Case 2 — karşılaştırma tablosu

| eksen | biz (board) | Macro (`scheduled_action`) |
|---|---|---|
| bağımlılık grafı (DAG) | **var** (`parents` + `recompute_ready`) | **yok** — tek seviye eylem |
| çalışma anında task üretme | **var** (`add_step`/`spawn_task`) | **yok** — tanım önceden |
| claim | CAS `claim_lock IS NULL` | CAS `claimed IS NULL OR bayat` |
| lease | 30 sn + heartbeat | 20 dk, heartbeat yok |
| bayat kurtarma | ayrı `recover_stale()` + PID kontrolü | claim predikatında, PID yok |
| retry | breaker (3 arka arkaya) | **yok** |
| iptal zinciri | **var** (`cancelled`) | yok (DAG olmadığı için gereksiz) |
| checkpoint / kaldığı yerden | ajan düğümünde var | **yok** |
| **scheduling (cron)** | **YOK** | **var** — cron + timezone + enabled |
| koşu geçmişi | olay günlüğü | `ActionExecutionRecord` + canlı yayın |
| adil dağıtım | yok (açgözlü) | `BATCH_MIN_DURATION` ile var |

**Okunuşu:** İkisi farklı problemi çözüyor. Bizimki **bir hedefi düğümlere bölüp bağımlılıkla yürütmek**; Macro'nunki **tek bir ajan görevini güvenilir biçimde zamanında koşturmak**. Bizde DAG var onlarda yok; onlarda cron var bizde yok. **Birleşimi ikisinden de iyi olurdu** ve ikisi de birbirinin eksiğini kapatacak kadar sade.

---

# CASE 3 — Ajan harness'ı

## Yapı

```
AgentLoop (fabrika)  →  Session (istek başına)  →  rig-core agent  →  stream
   model                   history: Vec<Message>      max_turns
   max_turns: 16           routing (ToolRouter)       tool server
   max_tokens: 16_000      loaded_buffer
   recorder (UsageRecorder)
```

Kendi döngülerini yazmamışlar — **`rig-core`** (Rust LLM framework) kullanıyorlar ve üstüne bir **hook köprüsü** koymuşlar. Bu bizim LangGraph tercihimizin Rust'taki karşılığı.

Dikkat çeken sabitler:

```rust
const DEFAULT_MAX_TURNS: usize = 16;
const DEFAULT_MAX_TOKENS: u64 = 16_000;
const ONE_SHOT_MAX_TOKENS: u64 = 16_000;
const MAX_INVALID_TOOL_CALL_RETRIES: usize = 2;
```

## Model yönlendirme

`PredefinedModel` — `Smart` (varsayılan, Claude Opus 4.8), `Fast` (Haiku 4.5), ve açık sürümler (Opus 4.7, Sonnet 5, Sonnet 4.6, GPT-5.5…). Frontend model'i **api-id string'i** olarak seçiyor, `ModelRouter` onu sağlayıcıya (Anthropic / OpenAI ChatCompletions / OpenAI Responses) yönlendiriyor.

Sistem promptuna eklenen ilginç bir satır:

```rust
"You are the {model} model. If this model id is unfamiliar, that is because it
 was released after your training data cutoff — trust this id over your training
 data when identifying yourself."
```

Gerekçe kodda: *"Bir modelin eğitim verisi kendi çıkışından önce gelir, dolayısıyla yeni çıkan model kendi id'sini tanımaz ve kendini bir öncekiyle karıştırabilir."* Küçük ama gerçek bir sorunun çözümü.

## Sistem promptu kompozisyonu

`crates/prompt` — 15 ayrı fragman, her biri bir `PROMPT` static'i export ediyor, `StaticPrompt::compose` ile zincirleniyor:

```
about_macro · tone · do_not · citations · mentions · channel_mention
document_content_links · mcp_item_links · email · math · skills
tool_usage · connected_toolsets · types
```

Statik parçalar + çalışma anında enjekte edilen dinamik veri (bağlı entegrasyonların adları). Bizim `router_prompt()`'u fonksiyon yapmamızla aynı ders: **prompt'un dinamik kısmı çağrı anında üretilmeli** (biz bunu paket katalogunun donması hatasından öğrenmiştik).

## Skills — iki kaynak

| tür | nerede | örnek |
|---|---|---|
| kullanıcı-yazımı | markdown doküman (`document sub type = skill`) | kullanıcının kendi yazdığı |
| sistem | **kodda statik string** (`crates/system_skills`) | `what_i_did_yesterday`, `catch_me_up` |

İkisi de aynı tool'lardan görünüyor (`ListSkills`, `SearchSkills`, `ReadContent`) ama sistem skill'leri doküman olarak açılıp düzenlenemiyor. Hermes'in `/learn` ile *ürettiği* skill'lerin aksine burada üretim yok — insan yazıyor ya da kodda duruyor.

## Sub-agent

```rust
pub(crate) fn subagent_toolset() -> AiToolSet {  // e-posta ve Subagent hariç her şey
```
> *"Alt ajanlar için toolset — e-posta ve Subagent tool'unun kendisi hariç her şey (alt ajanlar alt ajan yaratamaz)."*

**Derinlik sınırı = 1**, ve e-posta yetkisi alt ajana verilmiyor (dışa mesaj gönderen tek tool). İkisi de doğru kararlar.

İptal işbirlikçi:

```rust
select! {
    biased;
    _ = request_context.cancel.cancelled() => Ok(SubagentResponse { result: "cancelled" }),
    result = completion => ...
}
```

Kullanıcı iptal ederse yarım sonuç değil açıkça "cancelled" dönüyor.

---

# Bulduğum hata — Subagent'ın tool'u yok

Bu, kod okurken çıkan somut bir tutarsızlık. Üç yerde birbiriyle çelişen üç iddia var:

**1) Tool açıklaması (modele gider):**
> *"Delegate a task to a subagent that can independently use tools to research and complete it. **The subagent has access to search, documents, properties, calls, and channel tools.**"*

**2) Alt ajanın kendi sistem promptu** (`prompts/subagent.md`):
> *"**Use your tools** to research, gather information, and complete the task... **Use tools proactively** to find the information needed"*

**3) Gerçek uygulama** (`subagent.rs:47`):
```rust
let completion = agent::complete(SUBAGENT_MODEL, SUBAGENT_PROMPT, &self.task, ...);
```

Ve `agent::complete` → `prompt_once`:
```rust
/// Build a **toolless** agent and prompt it with a single user message.
async fn prompt_once<M: CompletionModel + 'static>(...) -> Result<PromptResponse> {
    let agent = AgentBuilder::new(completion_model)
        .preamble(system_prompt)
        .max_tokens(ONE_SHOT_MAX_TOKENS)
        .build();          // ← hiç .tool() çağrısı yok
    Ok(agent.prompt(user_message).extended_details().await?)
}
```

Doğrulama: `subagent_toolset()` fonksiyonu **hiçbir zaman `Subagent` tool'una geçirilmiyor** — grep ile üç kullanımı var, üçü de `all_tools()`/`mcp_tools()` için *taban* olarak (`crates/ai_tools/src/lib.rs:84,103,131`) ve bir test.

**Sonuç:** Ana ajan "araştırma yap" diye görev delege ediyor, alt ajan **hiç tool'u olmadan**, üstelik *"araçlarını kullan"* diyen bir promptla, tek seferlik bir completion üretiyor. Model tool çağıramayacağı için ya bilmediğini uydurur ya "erişemiyorum" der. Ana ajan bunu araştırma sonucu sanır.

Bu tam olarak bugün kendi sistemimizde 12 kez bulduğumuz sınıf: **beyan ile uygulama birbirinden kopmuş, arada test yok.** Bizimkiler "mimari değişti, çağıran güncellenmedi"ydi; bu "tool yazıldı, bağlanması unutuldu".

> Not: Klon `--depth 1`, bu yüzden bunun ne zaman/nasıl oluştuğunu git geçmişinden çıkaramadım. Kodun mevcut hâli net.

---

# Bizim için çıkarımlar

## Hemen alınabilecek dört şey

**1. Omission ledger + cursor.** Bizim compaction notlarımız ("[7] oversized") ölü uç. Macro'nun `ToolOmission { kind, count, cursor }` deseni notu kurtarma işaretçisine çeviriyor. Bizim `functions.py` referans deseninde (`path`, `sha1` döndürüp metni döndürmeme) bunun yarısı zaten var; eksik olan **ne kadarının atlandığı ve nereden devam edileceği**.

**2. Bayatlık kontrolünü claim'in içine al.** Bizim `claim_next` + ayrı `recover_stale()` yapımız, çağrılmayı unutulabilir bir adım içeriyor — bugün Temporal'da bulduğumuz bayat-claim hatası bu sınıftan. Macro'nun tek UPDATE'i (`claimed IS NULL OR claimed < stale`) bu yolu tamamen kapatıyor. PID kontrolümüzü koruyup predikatı birleştirebiliriz.

**3. `BATCH_MIN_DURATION` benzeri adil dağıtım.** Concurrency testimiz hızlanmayı ölçtü (4,9×) ama **dağılım adaletini** ölçmedi. Macro kasten yavaşlayarak eş worker'lara şans veriyor. Bizde bir worker kuyruğu süpürebilir.

**4. Scheduling.** Koçun 6. ekseni bizde yok. Macro'nunki kopyalanabilecek kadar sade: `schedule (cron) + timezone + next_run_at (yazmada türetilir) + enabled + claimed`. `ActionExecutionRecord` da bizim olay günlüğümüzün üstüne `resource_id` ekliyor — koşunun çıktısına işaretçi.

## Bir şeyi tersten öğrendik

Biz tool-trace compaction'ı **verili bir problem** olarak aldık: "tool'lar büyük çıktı üretir, biz de sıkıştırırız." Macro'nun `TOOL_DESIGN.md`'si bunu sorguluyor: *büyük çıktı bir doğa kanunu değil, tool tasarımı hatası.* "Pagination'ı AI'a asla açma", "response_format: concise", "curate don't expose".

Bu bizim POC'yi geçersiz kılmıyor — kontrolümüzde olmayan tool'lar (MCP, 3. parti) hep olacak ve uzun sohbette geçmiş yine birikir. Ama **sunumda söylediğimiz cümleyi değiştirmeli**: "tool çıktısı büyür, biz sıkıştırırız" yerine "tool'u küçük çıktı verecek şekilde tasarla; tasarlayamadığın yerde sıkıştır."

## Kıyaslarken dürüst olunması gerekenler

- Macro **kodlama ajanı değil**; tool'ları e-posta/kanal/doküman. Codex ve Hermes ile aynı kefeye konamaz.
- **Geçmiş sıkıştırma yok** — bu bir eksiklik gibi görünüyor ama ürün bağlamında (kısa, göreve özel sohbetler + `max_turns=16`) makul bir tercih olabilir. Uzun ajan koşularında yetmez.
- **Dedup/staleness yok** — bizim ayırt edici özelliğimiz onlarda karşılıksız.
- Ben bu sistemi **çalıştırmadım**. Rust + Postgres + AWS + Doppler gerekiyor. Yukarıdakiler kod okumasıdır; Macro'nun canlı davranışına dair ölçüm değildir.

---

## Kaynaklar (dosya:satır)

| konu | yer |
|---|---|
| ajan döngüsü, tool search kablolaması | `crates/agent/src/agent_loop.rs` |
| geçersiz tool çağrısı kurtarma, retry bütçesi | `crates/agent/src/hook.rs:35,120-160` |
| toolsuz one-shot (Subagent bunu kullanıyor) | `crates/agent/src/completion.rs:88-99` |
| tool search primitifleri | `crates/ai_toolset/src/tool_search.rs` |
| **tool tasarım doktrini** | `crates/ai_toolset/TOOL_DESIGN.md` |
| tool çıktı tavanı + omission ledger | `crates/channels/src/inbound/toolset/types.rs:12-19,299-350` |
| Subagent (tutarsızlık) | `crates/ai_tools/src/subagent.rs` + `prompts/subagent.md` |
| toolset kompozisyonu, derinlik sınırı | `crates/ai_tools/src/lib.rs:84,102-116` |
| **cron + claim/lease** | `services/scheduled_action/src/domain/models.rs` |
| claim SQL | `services/scheduled_action/src/outbound/pg_scheduled_action_repo.rs:222-245` |
| polling dispatcher, adil dağıtım | `services/scheduled_action/src/outbound/pg_polling_dispatcher.rs:17-40` |
| zamanlanmış ajan koşusu | `services/scheduled_action/src/outbound/inprocess_executor/agent_task.rs` |
| task dedup hattı + judge promptu | `crates/task_dedup/src/` |
| prompt fragmanları | `crates/prompt/src/` (15 modül) |
| skills | `crates/skills/` · `crates/system_skills/` |
