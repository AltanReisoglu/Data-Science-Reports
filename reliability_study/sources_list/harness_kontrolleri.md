# Harness Kontrolleri — Loop Detection & Budget Enforcement Kod Envanteri

Tarama tarihi: 2026-08-21 · Yöntem: **yalnızca birincil kaynak** — GitHub'daki gerçek kaynak
kodu (`raw.githubusercontent.com`), GitHub Code Search (`gh api search/code`) ve resmî
dokümantasyon. Blog yazısı, "awesome-list" ve ikincil özet kullanılmadı.

Kardeş dosya: `loop_budget.md` (akademik kaynaklar ve olay katalogları).

**Doğrulama etiketleri:**
`[K]` kaynak kodu bizzat okundu, sayılar dosyadan alındı ·
`[D]` resmî dokümantasyondan alındı, kod görülmedi ·
`[?]` doğrulanamadı / emin değilim.

**Her harness'a sorulan üç soru:**
1. Döngü/stuck tespiti var mı — hangi heuristik, hangi eşik, hangi dosya+satır?
2. Bütçe limitleri neler — parametre adı, **varsayılan değer**, istisna sınıfı?
3. Limit dolunca ne oluyor — sert durdurma / nudge / zarif bozulma / kullanıcıya sorma?

**Kapsam:** 28 satır — 22 agent harness/framework, 5 gateway/gözlemlenebilirlik katmanı,
1 standart. Bunların 9'u bu turda ilk kez kod düzeyinde incelendi
(Gemini CLI, Codex CLI, SWE-agent, Aider, Roo Code, Goose, opencode, Continue, Agno/DSPy/Letta),
1'i ikinci elden gelen değerler için doğrulandı (Cline), 5'i gateway/gözlemlenebilirlik
katmanı (Helicone, Langfuse, Weave, AgentOps, OTel GenAI semconv), kalanı önceki turlardan
gelip tabloya işlendi. **Ana çıktı §5'teki desen sentezi.**

⚠️ Tarama sırasında **dört depo taşınması** ortaya çıktı (Goose, opencode, Letta, OTel
GenAI); ayrıntı §6'nın sonunda.

---

## 1. Ana karşılaştırma tablosu

Kısaltmalar: **Sert** = istisna fırlatıp koşumu bitirir · **Nudge** = prompt'a/geçmişe uyarı
enjekte eder, koşum devam eder · **Zarif** = modelden nihai cevap/patch isteyip öyle biter ·
**Yok** = mekanizma bulunamadı (kod okundu, gerçekten yok).

| Harness | Loop tespiti | Heuristik | Adım/tur limiti (varsayılan) | Token | Maliyet | Süre | Limit dolunca | Et. |
|---|---|---|---|---|---|---|---|---|
| **OpenHands** (software-agent-sdk) | ✅ ayrı sınıf | 5 senaryo: aynı eylem+gözlem (4), aynı eylem+hata (3), monolog (3), A-B-A-B (6), context-window hata döngüsü; pencere 20 olay | `max_iteration_per_run=500` | — | `max_budget_per_run=None` | — | Sert — `ConversationExecutionStatus.STUCK` ayrı terminal durum | `[K]` |
| **Gemini CLI** | ✅ en kapsamlısı | 3 katman: tool-call cycle hash'i (k=1..5, R=5), içerik "chanting" chunk hash'i (10/50 char), **LLM yargıcı** (30. turdan sonra, 5–15 tur arayla, güven ≥ 0,9) | `maxSessionTurns` **varsayılan −1 = sınırsız**; iç `MAX_TURNS=100` özyineleme tavanı | — | — | — | **Kademeli**: 1. tespit → `_recoverFromLoop()` (kendini toparlama turu), 2. tespit → `LoopDetected` olayı, koşum durur | `[K]` |
| **Cline** | ✅ soft/hard iki kademe | Ardışık aynı `(toolName, JSON.stringify(sortKeys(input)))` imzası | `maxConsecutiveMistakes` **6** (core), CLI'da `--retries ?? 3` | — | — | `timeoutSeconds` (CLI) | soft(3) → **nudge** (geçmişe user mesajı), hard(5) → `forceAtLimit` mistake → sert durdurma | `[K]` |
| **SWE-agent** | ⚠️ dolaylı | Döngü heuristiği **yok**; yerine hata sayaçları: `max_requeries=3`, `max_consecutive_execution_timeouts=3` | `per_instance_call_limit=0` (=kapalı) | `max_input_tokens=None` → `ContextWindowExceededError` | **`per_instance_cost_limit=3.0` $**, `total_cost_limit=0.0` (kapalı) | `execution_timeout=30` s/komut, `total_execution_timeout=1800` s | **Zarif** — `attempt_autosubmission_after_error()` mevcut patch'i yine de gönderir; `exit_cost` / `exit_context` / `exit_total_execution_time` statüsü | `[K]` |
| **Aider** | ⚠️ zayıf | Döngü heuristiği yok; `max_reflections=3` yalnızca kendi kendine tetiklenen tekrar denemeyi sayar | `max_reflections=3` (sınıf sabiti, CLI'dan ayarlanamaz) | `--max-chat-history-tokens=None` (özetlemeyi tetikler, **durdurmaz**) | **Yok** — `total_cost` yalnızca raporlanır | `--timeout=None` (tek API çağrısı için) | Uyarı basıp `return` — "Only 3 reflections allowed, stopping." | `[K]` |
| **LangGraph** | ❌ | — | `DEFAULT_RECURSION_LIMIT=25` | — | — | — | Sert — `GraphRecursionError` | `[K]` |
| **smolagents** | ❌ | — | `max_steps=20` | — | — | — | **Zarif** — `_handle_max_steps_reached()` modelden nihai cevap ister | `[K]` |
| **CrewAI** | ❌ | — | `Agent.max_iter=25`, `PlanningConfig.max_steps=20`, `max_step_iterations=15` | `respect_context_window=True` | — | `max_execution_time`, `step_timeout=None`, `max_rpm` | Sert / context budama | `[K]` |
| **pydantic-ai** | ❌ | — | `UsageLimits.request_limit=50` | var, varsayılan `None` | — | — | Sert — `UsageLimitExceeded` | `[K]` |
| **OpenAI Agents SDK** | ❌ | — | `max_turns` (zorunlu argüman) | — | — | — | Sert — `MaxTurnsExceeded` | `[K]` |
| **AutoGen** | ❌ | — | `MaxMessageTermination` | `TokenUsageTermination` | — | `TimeoutTermination` | Sonlanma koşulu — `&` / `\|` ile birleştirilebilir | `[K]` |
| **Google ADK** | ❌ | — | `LoopAgent.max_iterations` **varsayılan `None` = sınırsız** | — | — | — | `RunConfig.max_llm_calls` → `LlmCallsLimitExceededError` | `[K]` |
| **Claude Agent SDK** | ❌ | — | `max_turns=None` | — | `max_budget_usd=None` | — | — | `[K]` |
| **LiteLLM** (gateway) | ❌ | — | `max_parallel_requests` | `tpm_limit` | `max_budget` / `soft_budget` / `budget_duration` | `rpm_limit` | Sert — `BudgetExceededError` → HTTP 429 | `[K]` |
| **deer-flow** | ✅ | İlerleme yokluğu sayacı | `max_continuations=8`, `max_no_progress_continuations=2` | — | — | — | Sert | `[K]` |
| **OpenAI Codex CLI** | ❌ **yok** | — | **Yok** — ana araç döngüsünde tur sayacı yok | `rollout_budget.limit_tokens` (**feature default kapalı**) | — | — | Bütçe açıksa: eşiklerde **nudge**, tükenince sert (`CodexErr::SessionBudgetExceeded`). Varsayılan kurulumda **hiçbir sınır yok** | `[K]` |
| **Roo Code** | ✅ `ToolRepetitionDetector` | Ardışık aynı `stringify({name, params})`; limit = `consecutiveMistakeLimit` | Eklenti **3**, CLI **10**; `allowedMaxRequests` **varsayılan `Infinity`** | — | `allowedMaxCost` **varsayılan `Infinity`** | `commandExecutionTimeout` (CLI'da 300 s) | **Kullanıcıya sorma** — `ask("mistake_limit_reached")`, kullanıcı yönlendirirse sayaç sıfırlanıp devam | `[K]` |
| **Goose** | ⚠️ dolaylı | Döngü heuristiği yok; tur bütçesi var | `GOOSE_MAX_TURNS` **varsayılan 1000**; gateway'de `GOOSE_GATEWAY_MAX_TURNS=5` | — | — | — | **İki kademe**: yarıda `<turn-budget>N/M used</turn-budget>` nudge'ı; limitte **kullanıcıya sorma** ("Would you like me to continue?") | `[K]` |
| **opencode** | ✅ "doom loop" | Son 3 mesaj parçası aynı araç + aynı JSON girdi | `agent.steps` **varsayılan `Infinity`** | — | — | — | Loop → **izin sistemi** (`permission.ask({permission:"doom_loop"})`); adım limiti → **zarif** (`MAX_STEPS_PROMPT` + araçlar kapatılır) | `[K]` |
| **Continue** | ❌ **yok** | — | **Yok** — GUI'de `depth>50` guard'ı yalnızca `NODE_ENV==="test"`'te; CLI'de `while(true)` sayaçsız | — | — | — | **Hiçbir şey** — döngü yalnızca model araç istemeyi bırakınca biter | `[K]` |
| **Agno** | ❌ | — | `tool_call_limit` **varsayılan `None`** | — | — | — | **Nudge + koşum devam** — araç sonucu yerine "Tool call limit reached… Don't try to execute it again" hata mesajı | `[K]` |
| **DSPy (ReAct)** | ❌ | — | `max_iters=20` (docstring "10" diyor — **tutarsız**) | — | — | — | **Zarif** — döngü biter, `self.extract` biriken trajectory'den cevap üretir | `[K]` |
| **Letta** | ❌ | — | `DEFAULT_MAX_STEPS=50` (**arşiv dalı**; yeni istemci limiti sunucudan `stop_reason:'max_steps'` olarak alıyor) | — | — | — | Sert — `stop_reason: max_steps` | `[K]`/`[?]` |
| **Helicone** (gateway) | ❌ | — | `[kota];w=[pencere];u=request` | — | `u=cents` — **dolar bütçesi** (pencere ≥ 60 s, ≤ 1 yıl) | pencere tabanlı | Sert — HTTP **429** + `X-Helicone-Error: rate_limited` | `[K]` |
| **Langfuse** | ❌ | — | — | — | — | — | **Yalnızca raporlar** — rate limit'i kendi API'sine ait (org planı bazlı), agent harcamasına değil | `[K]` |
| **W&B Weave** | ❌ | — | — | — | — | — | **Yalnızca raporlar** — `token_costs.py` maliyeti hesaplar, hiçbir yerde eşikle karşılaştırmaz | `[D]` |
| **AgentOps** | ❌ (yol haritasında 🚧) | — | — | — | — | — | **Yalnızca raporlar** — "Infinite loops and recursive thought detection" README'de 🚧 | `[K]`/`[D]` |
| **OTel GenAI semconv** | — (standart) | — | `gen_ai.invoke_agent.inference_calls`, `gen_ai.invoke_agent.tool_calls` (metrik) | `gen_ai.usage.*_tokens` (14 alan) | **Maliyet alanı YOK** — "cost" kelimesi registry'de 0 kez geçiyor | `gen_ai.invoke_agent.duration` | Standart yalnızca **gözlem** tanımlar, zorlama tanımlamaz | `[K]` |

---

## 2. Harness alt bölümleri — Öncelik 1: kodlama harness'ları

### 2.1 Gemini CLI (`google-gemini/gemini-cli`) `[K]`

**Bu envanterdeki en olgun loop detection implementasyonu.** Tek dosyada üç bağımsız
dedektör var ve üçü farklı hata moduna bakıyor.

Dosya: `packages/core/src/services/loopDetectionService.ts` (781 satır)

**Sabitler (satır 28–64):**

```ts
const TOOL_CALL_LOOP_THRESHOLD = 5;
const CONTENT_LOOP_THRESHOLD = 10;
const CONTENT_CHUNK_SIZE = 50;
const MAX_HISTORY_LENGTH = 5000;
const LLM_LOOP_CHECK_HISTORY_COUNT = 20;   // LLM'e verilen son tur sayısı
const LLM_CHECK_AFTER_TURNS = 30;          // LLM kontrolü bu turdan önce hiç çalışmaz
const DEFAULT_LLM_CHECK_INTERVAL = 10;
const MIN_LLM_CHECK_INTERVAL = 5;
const MAX_LLM_CHECK_INTERVAL = 15;
const LLM_CONFIDENCE_THRESHOLD = 0.9;
```

**Dedektör 1 — tool-call döngüsü (`checkToolCallLoop`, satır 313–349).** Her araç çağrısı
`sha256(name + JSON args)` ile bir anahtara indirgeniyor (`getToolCallKey`, satır 174–178).
Sadece "aynı çağrı 5 kez üst üste" değil, **k uzunluğunda tekrarlayan çevrim** aranıyor:

```ts
// Check for repeating patterns of cycle length k from 1 to 5
for (let k = 1; k <= 5; k++) {
  const requiredLength = k * R;   // R = TOOL_CALL_LOOP_THRESHOLD = 5
  ...
}
```

Yani A-B-A-B-A-B… (k=2) ya da A-B-C-A-B-C… (k=3) desenleri de 5 tekrardan sonra yakalanıyor.
Geçmiş `5 * 5 = 25` anahtarla sınırlı tutuluyor. OpenHands'in `alternating_pattern` eşiğine
(6) karşılık gelen ama **k'yı 5'e kadar genelleştiren** tek implementasyon bu.

**Dedektör 2 — içerik "chanting" döngüsü (`checkContentLoop`, satır 362+).** Model akan
metinde aynı şeyi tekrarlıyorsa: metin 50 karakterlik chunk'lara bölünüp hash'leniyor,
aynı hash kısa mesafede 10 kez görünürse döngü. Özgün ayrıntı: **kod bloğu içinde dedektör
kapatılıyor** (`inCodeBlock`), çünkü tekrar eden kod yapıları doğal olarak yanlış pozitif
üretiyor.

**Dedektör 3 — LLM yargıcı (`checkForLoopWithLLM`, satır 563+).** Bir promptta 30 tur
geçtikten sonra devreye giriyor, son 20 turu ayrı bir modele (`loop-detection-double-check`
alias'ı) gönderiyor, `unproductive_state_confidence ≥ 0.9` ise döngü sayıyor. **Kontrol
aralığı güvene göre dinamik**: `updateCheckInterval()` (satır 733) güven yüksekse 5'e iner,
düşükse 15'e çıkar — pahalı kontrolü sadece şüphe arttığında sıklaştıran bir tasarım.

Sistem prompt'u (satır 68+) yanlış pozitifi açıkça hedefliyor; iki koşulu birden şart koşuyor:
"en az 5 ardışık model eylemi boyunca tekrar deseni" **ve** "hedefe doğru net ilerleme yok".
Ve normal iş akışlarını elle muaf tutuyor:

> "re-running a build to verify a fix is normal workflow" · "If the assistant is modifying
> different code or getting different errors, that is debugging progress, not a loop."

**Limit dolunca — kademeli tepki (`packages/core/src/core/client.ts`, satır 747–760 ve 810–855):**

```ts
const loopResult = await this.loopDetector.turnStarted(signal);
if (loopResult.count > 1) {
  yield { type: GeminiEventType.LoopDetected };
  return turn;
} else if (loopResult.count === 1) {
  if (boundedTurns <= 1) { yield { type: GeminiEventType.MaxSessionTurns }; return turn; }
  return yield* this._recoverFromLoop(loopResult, signal, prompt_id, boundedTurns, displayContent);
}
```

**İlk tespitte durdurmuyor, `_recoverFromLoop()` ile kurtarma turu deniyor; ancak ikinci
tespitte (`count > 1`) koşumu bitiriyor.** Bu envanterde "önce nudge, sonra sert" kademesini
loop detection'a uygulayan iki harness'tan biri (diğeri Cline).

**Tur bütçesi (`client.ts`):** `MAX_TURNS = 100` (satır 79) sabit bir özyineleme tavanı;
`sendMessageStream` her çağrıda `Math.min(turns, MAX_TURNS)` uyguluyor (satır 960).
Kullanıcıya açık ayar `maxSessionTurns` ise `packages/core/src/config/config.ts:1243`'te
**`params.maxSessionTurns ?? -1`** — yani **varsayılanı sınırsız**; `client.ts:626`'da
`getMaxSessionTurns() > 0` koşuluyla korunuyor, negatif değer kontrolü tamamen atlıyor.
Loop detection ayrıca `getDisableLoopDetection()` ile ve oturum içi `disabledForSession`
bayrağıyla kapatılabiliyor.

### 2.2 Cline (`cline/cline`) — Öncelik 4 doğrulaması `[K]`

İkinci elden gelen `soft=3 / hard=5` ve `maxConsecutiveMistakes=6` değerleri **doğrulandı**,
ama iki önemli düzeltme var.

**Loop detection.** Dosya: `sdk/packages/core/src/runtime/safety/loop-detection.ts`

```ts
const DEFAULT_CONFIG: LoopDetectionConfig = {
	softThreshold: 3,
	hardThreshold: 5,
};
```

İmza fonksiyonu (`toolCallSignature`, satır 51–60) argümanları **anahtarları özyinelemeli
sıralayarak** JSON'a çeviriyor (`sortKeys`) — yani `{a:1,b:2}` ile `{b:2,a:1}` aynı imza.
Heuristik OpenHands'e göre çok daha dar: sadece **ardışık** aynı `(araç adı, imza)` çifti
sayılıyor (`checkRepeatedToolCall`, satır 66–86); araya farklı bir çağrı girerse sayaç
1'e sıfırlanıyor. A-B-A-B deseni Cline'da yakalanmaz.

**Tepki iki kademeli** (`sdk/packages/core/src/runtime/orchestration/session-runtime-orchestrator.ts`,
satır 1298–1324):

```ts
if (verdict.kind === "soft") {
    this.conversation.appendMessage({ role: "user", content: [{ type: "text", text: verdict.message }] });
    return;
}
// Hard escalation.
this.enqueueMistakeRecord({ iteration, reason: "tool_execution_failed", forceAtLimit: true, ... });
```

3. tekrarda geçmişe **user rolünde bir nudge** enjekte ediliyor ("consider trying a different
approach"), koşum devam ediyor. 5. tekrarda `forceAtLimit: true` ile mistake sayacı doğrudan
tavana çekiliyor ve koşum duruyor — yani hard escalation ayrı bir durdurma yolu değil,
**mistake mekanizmasına kısayol**.

**Düzeltme 1 — varsayılan açık değil.** `apps/cli/src/runtime/defaults.ts` bunu açıkça yazıyor:

> "The agent core leaves loop detection off by default; the CLI enables it with these settings."

`CLI_DEFAULT_LOOP_DETECTION = { softThreshold: 3, hardThreshold: 5 }`. `execution.loopDetection`
`false` verilirse `loopDetectionDisabled` bayrağı ile tracker hiç beslenmiyor
(orchestrator satır 318–322, 463–469).

**Düzeltme 2 — `maxConsecutiveMistakes` değeri bağlama göre değişiyor.**
- Core varsayılanı **6**: `session-runtime-orchestrator.ts:430` →
  `const maxMistakes = config.execution?.maxConsecutiveMistakes ?? 6;`
  (`sdk/packages/shared/src/agents/types.ts:254` de `@default 6` diyor.)
- **CLI bunu eziyor**: `apps/cli/src/main.ts:1070` → `maxConsecutiveMistakes: args.retries ?? 3`.
  Yani `cline` komut satırından koşarsanız gerçek eşik **3**, 6 değil.

Mistake sebepleri sınıflandırılmış: `"api_error" | "invalid_tool_call" | "tool_execution_failed"`
(`types.ts:224`). Durdurma mesajı sayacı da raporluyor
(`mistake-tracker.ts:178`): `Stopped after {n}/{max} consecutive mistakes ({reason}) at iteration {i}.`

**Üçüncü bir nudge mekanizması daha var, o da varsayılan kapalı.**
`sdk/packages/shared/src/agents/types.ts:249–256` → `reminderAfterIterations`,
**`@default 0`** (0 = devre dışı). Açıklaması: *"After this many consecutive iterations
with tool calls, inject a reminder text block asking the agent to answer if it has enough
info."* Varsayılan metin de tanımlı:

> "REMINDER: If you have gathered enough information to answer the user's question, please
> provide your final answer now without using any more tools."

Yani Cline'da üç ayrı guardrail var — loop detection (core'da kapalı), mistake tracker
(açık, 6/3) ve iterasyon hatırlatıcısı (kapalı) — ve **ikisi kutudan çıktığı hâliyle
çalışmıyor.**

### 2.3 SWE-agent (`SWE-agent/SWE-agent`) `[K]`

**Döngü heuristiği yok** — bunun yerine bütçe tarafı bu envanterdeki en ayrıntılı
implementasyon. Tek harness içinde **altı ayrı limit** ve her biri için ayrı istisna sınıfı var.

Dosya: `sweagent/exceptions.py` — istisna hiyerarşisi:

```python
class CostLimitExceededError(Exception): ...
class InstanceCostLimitExceededError(CostLimitExceededError): ...
class TotalCostLimitExceededError(CostLimitExceededError): ...
class InstanceCallLimitExceededError(CostLimitExceededError): ...
class ContextWindowExceededError(Exception): ...
```

**Varsayılanlar** (`sweagent/agent/models.py`, satır 73–78):

| Parametre | Varsayılan | Not |
|---|---|---|
| `per_instance_cost_limit` | **3.0** (dolar) | Bu envanterde **varsayılanı sıfırdan farklı tek dolar limiti** |
| `total_cost_limit` | `0.0` | 0 = kapalı (`if 0 < limit < value` deseni) |
| `per_instance_call_limit` | `0` | 0 = kapalı |
| `max_input_tokens` | `None` | `None` ise modelin `litellm.model_cost` kaydından okunuyor (satır 599–602) |

Zorlama noktası `models.py:_update_stats()` (satır 632–668) — her API çağrısından sonra:

```python
if 0 < self.config.total_cost_limit < GLOBAL_STATS.total_cost: ... raise TotalCostLimitExceededError(msg)
if 0 < self.config.per_instance_cost_limit < self.stats.instance_cost: ... raise InstanceCostLimitExceededError(msg)
if 0 < self.config.per_instance_call_limit < self.stats.api_calls: ...
```

Token tarafı `models.py:695–703`: girdi token'ı `model_max_input_tokens`'ı aşarsa
`ContextWindowExceededError`. Yani SWE-agent context taşmasını modele göndermeden **önce**
kendisi yakalıyor.

**Süre limitleri** (`sweagent/tools/tools.py`, satır 139–151):

```python
execution_timeout: int = 30                  # tek komut
install_timeout: int = 300
total_execution_timeout: int = 1800          # tüm oturumdaki komut süresi toplamı (30 dk)
max_consecutive_execution_timeouts: int = 3
```

`agents.py:1018` toplamı kontrol edip `_TotalExecutionTimeExceeded` fırlatıyor; `agents.py:970`
ardışık timeout sayacını `max_consecutive_execution_timeouts` ile karşılaştırıyor.
`max_requeries: int = 3` (`agents.py:158`) format/blocklist/bash sözdizimi hatalarında
modele kaç kez yeniden sorulacağını sınırlıyor — döngü dedektörü değil ama işlevsel olarak
"aynı hataya saplanma"yı kesen mekanizma.

**Limit dolunca — zarif bozulma.** `agents.py:1160–1210` her limit istisnasını yakalayıp
`attempt_autosubmission_after_error()` çağırıyor; docstring'i (satır 823–824) şöyle:

> "For most exceptions, we attempt to still extract the patch and submit that."

Yani bütçe bitince iş çöpe atılmıyor, o ana kadarki patch yine gönderiliyor ve koşuma bir
çıkış statüsü yazılıyor: `exit_cost`, `exit_context`, `exit_total_execution_time`,
`exit_command_timeout`, `exit_api`, `exit_environment_error`, `exit_format`. **Tek istisna
`TotalCostLimitExceededError`** — o `raise` ile yukarı geçiyor (satır 1180–1181), çünkü
toplam bütçe bittiğinde tüm batch durmalı, sadece bu instance değil.

**Özgün tasarım kararı — bütçenin denemeler arasında paylaştırılması.** `RetryAgentConfig`
aynı görevi birden çok kez deniyor; `_setup_attempt()` (satır 304–310) her denemeden önce
kalan bütçeyi hesaplayıp alt-agent'ın limitini kısıyor:

```python
remaining_budget = self.config.retry_loop.cost_limit - self._total_instance_stats.instance_cost
if remaining_budget < agent_config.model.per_instance_cost_limit:
    agent_config.model.per_instance_cost_limit = remaining_budget
```

Ayrıca satır 336'da `> 1.1 * cost_limit` şeklinde **%10 toleranslı bir ikinci ağ** var —
alt-agent kendi limitini geç yakalarsa üst katman yine de kesiyor. Hiyerarşik bütçe
devri (budget delegation) için doğrudan kopyalanabilir bir örnek.

### 2.4 Aider (`Aider-AI/aider`) `[K]`

**Bütçe tarafı bu envanterdeki en zayıf harness'lardan biri.** Tek gerçek sınır
`max_reflections`.

Dosya: `aider/coders/base_coder.py`

```python
# satır 97-101 (sınıf düzeyi öznitelikler)
num_exhausted_context_windows = 0
num_malformed_responses = 0
num_reflections = 0
max_reflections = 3
```

Zorlama `run_one()` içinde (satır 932–944):

```python
while message:
    self.reflected_message = None
    list(self.send_message(message))
    if not self.reflected_message:
        break
    if self.num_reflections >= self.max_reflections:
        self.io.tool_warning(f"Only {self.max_reflections} reflections allowed, stopping.")
        return
    self.num_reflections += 1
    message = self.reflected_message
```

"Reflection" = aider'ın kendi kendine tetiklediği yeniden deneme (lint hatası, test hatası,
bozuk edit bloğu, eksik dosya). Yani bu bir **self-feedback döngüsü sayacı** — `loop_budget.md`
§2'deki IAL-SCAN taksonomisinin "sınırsız retry geri beslemesi" desenine (%25,0) karşılık
gelen yolun burada 3 ile kapatılmış hali. Sınıf sabiti olarak duruyor; `aider/args.py` içinde
buna karşılık gelen bir CLI bayrağı **yok** — kullanıcı değiştiremiyor.

`num_exhausted_context_windows` (satır 1546) ve `num_malformed_responses` (satır 2306)
**sadece sayaç** — hiçbir yerde bir eşikle karşılaştırılmıyor, yalnızca oturum sonu raporuna
giriyor.

**Maliyet/token/süre:**
- `total_cost` (satır 384, 2046) yalnızca **biriktirilip raporlanıyor**:
  `f"Cost: ${...} message, ${...} session."` (satır 2058–2061). Bir tavanla karşılaştırılmıyor.
  `gh api search/code` ile tüm repoda `total_cost` taraması yapıldı; `aider/` altında
  yalnızca `base_coder.py`, `architect_coder.py`, `commands.py` ve leaderboard verilerinde
  geçiyor — **hiçbirinde zorlama yok**.
- `--max-chat-history-tokens` (`aider/args.py:221–227`) varsayılan `None`; docstring'inde
  açıkça "**Soft limit** on tokens for chat history, after which **summarization begins**"
  yazıyor — durdurmuyor, özetlemeyi tetikliyor.
- `--timeout` (`aider/args.py:158–162`) varsayılan `None` ve **tek bir API çağrısı** için;
  görev geneli wall-clock bütçesi değil.

Yani Aider'da **dolar veya süre üst sınırı diye bir şey yok**; kontrolden çıkan bir koşumu
kesecek tek şey `max_reflections=3` ve o da sadece aider'ın kendi tetiklediği tekrarları
sayıyor, modelin araç çağrısı döngüsünü değil.

Not: `aider/coders/context_coder.py:37` aynı sayacı `>= self.max_reflections - 1` ile,
yani **bir tur erken** kesiyor — alt-coder'ın üst coder'a dönüş bütçesi bırakması için.

### 2.5 OpenAI Codex CLI (`openai/codex`) `[K]`

**Bu envanterin en çarpıcı negatif bulgusu.** Codex CLI'ın ana araç döngüsünde
(`codex-rs/core/src/session/turn.rs`, `run_turn()`, satır 153+) **hiçbir tur sayacı yok**.
Fonksiyonun kendi docstring'i döngüyü şöyle tarif ediyor:

> "runs a loop where, at each sampling request, the model replies with either: requested
> function calls / an assistant message … If the model requests a function call, we execute
> it and send the output back to the model in the next sampling request."

Çıkış koşulu tek: model artık araç çağırmayıp düz mesaj döndürsün. Bulunan tek `max_retries`
(satır 1363) **HTTP akış hatası için** yeniden deneme sayısı — ajan döngüsü için değil.

**Loop detection yok.** `codex-rs` genelinde `stuck`, `consecutive_identical` ve benzeri
desenler arandı (grep.app + GitHub Code Search); ajan döngüsünde tekrar tespiti yapan
hiçbir kod bulunamadı. `stuck` geçen yerler TUI paste-burst, websocket worker, SQLite
job kurtarma gibi tamamen ilgisiz bağlamlar.

**İki bütçe mekanizması var ama ikisi de varsayılan olarak kapalı.**
`codex-rs/features/src/lib.rs`, satır 1412–1423:

```rust
FeatureSpec { id: Feature::TokenBudget,   key: "token_budget",   stage: Stage::UnderDevelopment, default_enabled: false },
FeatureSpec { id: Feature::RolloutBudget, key: "rollout_budget", stage: Stage::UnderDevelopment, default_enabled: false },
```

**(a) `token_budget` — saf nudge, durdurma yok.** `codex-rs/core/src/session/token_budget.rs`,
`maybe_record()` (satır 67–121): kalan context penceresi `reminder_threshold_tokens`'ın
altına düşünce konuşmaya bir hatırlatma mesajı ekliyor
(`{n_remaining}` şablonuyla); tam sıfırda `auto_compact_fallback_prompt` ile modele
"not al, pencere dönecek" diyor. Hiçbir noktada istisna fırlatmıyor — bu bir
**context yönetimi** aracı, bütçe zorlaması değil.

**(b) `rollout_budget` — asıl zorlama, ve envanterdeki tek "ağırlıklı token" muhasebesi.**
`codex-rs/core/src/rollout_budget.rs`, `record_usage()` (satır 46–65):

```rust
usage.output_tokens.max(0) as f64 * state.config.sampling_token_weight
    + usage.non_cached_input() as f64 * state.config.prefill_token_weight
...
Ok(state.weighted_tokens_used >= state.config.limit_tokens as f64)
```

Çıktı token'ı ile cache'lenmemiş girdi token'ı **ayrı katsayılarla** ağırlıklandırılıyor
(`sampling_token_weight`, `prefill_token_weight`) — yani "token" değil, fiyat yapısını
yansıtan bir birim sayılıyor. Sunucu isterse kendi birimini
(`usage.codex_rollout_budget_units`) dayatabiliyor.

Zorlama `codex-rs/core/src/session/rollout_budget.rs`:

```rust
pub(crate) fn record_rollout_budget_usage(&self, usage: &TokenUsage) -> CodexResult<()> {
    if self.services.agent_control.rollout_budget().record_usage(usage)? {
        return Err(CodexErr::SessionBudgetExceeded);
    }
    Ok(())
}
```

`CodexErr::SessionBudgetExceeded` mesajı: `"shared rollout token budget exhausted"`
(`codex-rs/protocol/src/error.rs:85–86`) ve `is_retryable()` içinde **açıkça yeniden
denenemez** olarak işaretli (satır 364–368).

**Özgün tasarım kararı — bütçe alt-agent'larla paylaşılıyor.** Konfigürasyon yorumu
(`codex-rs/core/src/config/mod.rs:995`) şöyle: *"Shared token budget for the root thread
and its sub-agents."* Bütçe `AgentControl` üzerinde `Arc<RolloutBudget>` olarak duruyor
(`codex-rs/core/src/agent/control.rs:120`), yani ana thread ve tüm alt-agent'lar **tek bir
sayaca** yazıyor. SWE-agent'ın denemeler arası bütçe devri ile birlikte, hiyerarşik bütçe
için iki farklı referans model.

**Kademeli tepki:** `reminder_at_remaining_tokens` bir **liste** (test dosyası
`codex-rs/core/tests/suite/rollout_budget.rs:30` içinde `vec![75, 50, 25]`); her eşik
geçildiğinde modele `<rollout_budget>You have N weighted tokens left in the shared session
token budget.</rollout_budget>` enjekte ediliyor. Yani **birden çok nudge, sonra tek sert
kesme** — bu envanterdeki en ince taneli kademelendirme.

**Sonuç:** kutudan çıktığı hâliyle Codex CLI'da ne loop detection ne de bütçe zorlaması var.
İkisi de `Stage::UnderDevelopment` ve elle açılması gerekiyor.

### 2.6 Roo Code (`RooCodeInc/Roo-Code`) `[K]`

**Bu envanterde limit dolduğunda kullanıcıyı devreye sokan iki harness'tan biri** (diğeri
Goose). Roo'da hiçbir limit koşumu tek başına sonlandırmıyor.

**Loop detection.** Dosya: `src/core/tools/ToolRepetitionDetector.ts` (89 satır).
Sınıf docstring'i amacı açıkça yazıyor: *"to prevent the AI from getting stuck in a loop."*
Serileştirme `safe-stable-stringify` ile `{name, params, nativeArgs?}` üzerinden
(satır 74–85) — Cline'ın `sortKeys`'ine denk bir kanonikleştirme.

Eşik ayrı bir sabit değil, `consecutiveMistakeLimit` ile **aynı değer**:
`src/core/task/Task.ts:513` → `this.toolRepetitionDetector = new ToolRepetitionDetector(this.consecutiveMistakeLimit)`.

**Varsayılan iki farklı yerde iki farklı sayı** — Cline'daki aynı tuzak:
- VS Code eklentisi: `packages/types/src/provider-settings.ts:28` → `DEFAULT_CONSECUTIVE_MISTAKE_LIMIT = 3`
- CLI: `apps/cli/src/types/constants.ts` → `DEFAULT_FLAGS = { ..., consecutiveMistakeLimit: 10 }`

Yani `roo` komut satırından koşarsanız hem mistake hem de tekrar eşiği **10**, 3 değil.

**Sayaç mantığında dikkat edilmesi gereken bir kayma var**: yeni bir araç görülünce sayaç
`0`'a sıfırlanıyor (`1`'e değil), sonra her tekrarda artıyor. Yani limit 3 iken tetikleme
**4. özdeş çağrıda** oluyor:

```ts
if (this.previousToolCallJson === currentToolCallJson) { this.consecutiveIdenticalToolCallCount++ }
else { this.consecutiveIdenticalToolCallCount = 0; this.previousToolCallJson = currentToolCallJson }
```

Tetiklendiğinde çalıştırma engelleniyor (`allowExecution: false`) **ve sayaçlar hemen
sıfırlanıyor** — kod yorumu gerekçeyi veriyor: *"Reset counters to allow recovery if user
guides the AI past this point."* Sonra `askUser: { messageKey: "mistake_limit_reached" }`.

**Mistake limiti de kullanıcıya soruyor** (`src/core/task/Task.ts:2483–2500`):

```ts
if (this.consecutiveMistakeLimit > 0 && this.consecutiveMistakeCount >= this.consecutiveMistakeLimit) {
    const { response, text, images } = await this.ask("mistake_limit_reached", t("common:errors.mistake_limit_guidance"))
    if (response === "messageResponse") { currentUserContent.push(... formatResponse.tooManyMistakes(text) ...) }
    this.consecutiveMistakeCount = 0
}
```

Kullanıcı yön verirse sayaç sıfırlanıp koşum devam ediyor. `consecutiveMistakeLimit = 0`
verilirse mekanizma tamamen kapalı. Ayrıca ayrı bir "araç kullanmama" sayacı var
(`Task.ts:3487–3493`): model iki tur üst üste hiç araç çağırmazsa `MODEL_NO_TOOLS_USED`
hatası basılıp mistake sayacı artıyor — yani "monolog" moduna karşı OpenHands'in
`monologue=3` eşiğinin daha gevşek bir karşılığı (eşik **2**).

**Bütçe limitleri — ikisi de varsayılan sınırsız.** `src/core/auto-approval/AutoApprovalHandler.ts`:

```ts
const maxRequests = state?.allowedMaxRequests || Infinity
const maxCost     = state?.allowedMaxCost     || Infinity
```

İkisi de `packages/types/src/global-settings.ts:116–117`'de `z.number().nullish()` —
şemada varsayılan yok, UI'da placeholder "Unlimited". Aşıldığında yine sert durdurma değil,
`askForApproval("auto_approval_max_req_reached", ...)`; kullanıcı onaylarsa
`lastResetMessageIndex` güncellenip sayaç sıfırlanıyor. Yani bunlar **bütçe değil,
oto-onay penceresi** — "N istek / X dolar sonra bana tekrar sor" anlamına geliyor.
Maliyet karşılaştırmasında kayan nokta hassasiyeti için `EPSILON = 0.0001` payı var.

### 2.7 Goose (`block/goose` → **`aaif-goose/goose`**) `[K]`

⚠️ **Depo taşınmış**: `github.com/block/goose` artık `github.com/aaif-goose/goose`'a
HTTP 301 ile yönleniyor (`raw.githubusercontent.com/block/goose/main/...` hâlâ çalışıyor,
ama kanonik depo yeni adres).

**Loop detection yok.** Tek kontrol tur bütçesi — ama bu envanterdeki **en yumuşak** tepkiyi
veren tasarım o.

`crates/goose/src/agents/agent.rs:86` → `const DEFAULT_MAX_TURNS: u32 = 1000;`
Çözüm sırası (satır 1670–1673): açık argüman → `GOOSE_MAX_TURNS` config parametresi → 1000.
Resmî dokümantasyon (`documentation/docs/guides/environment-variables.md`) da 1000 diyor.
Gateway (Telegram vb.) oturumları için ayrı ve çok daha sıkı bir tavan var
(`crates/goose/src/gateway/handler.rs`): `DEFAULT_GATEWAY_MAX_TURNS: u32 = 5`, öncelik
`GOOSE_GATEWAY_MAX_TURNS` → `GOOSE_MAX_TURNS` → 5. Kod yorumu gerekçeyi veriyor: *"Chat
platforms like Telegram favor short, snappy replies, so the gateway keeps a stricter default
than the global GOOSE_MAX_TURNS ceiling."* — **aynı harness, bağlama göre 200 kat farklı
varsayılan.**

**İki kademeli tepki.** `crates/goose/src/agents/state_machine/ops_maxturns.rs` (68 satır):

*Kademe 1 — bütçe farkındalığı nudge'ı.* Bütçenin **yarısı** dolunca modele her turda
kalan bütçe enjekte ediliyor:

```rust
fn turn_budget_part(turns_taken: u32, max_turns: u32) -> Option<String> {
    if max_turns == 0 || turns_taken.saturating_mul(2) < max_turns { return None; }
    Some(format!("<turn-budget>{turns_taken}/{max_turns} used</turn-budget>"))
}
```

Bu envanterde **modele kendi kalan bütçesini söyleyen** yalnızca iki mekanizmadan biri
(diğeri Codex `rollout_budget`). Amaç modelin kalan turları kendi planlamasına katması.

*Kademe 2 — kullanıcıya sorma.* Limitte istisna yok, mesaj var:

```rust
pub const MAX_TURNS_MESSAGE: &str = "I've reached the maximum number of actions I can do without user input. Would you like me to continue?";
```

`agent.rs:2526–2530` bu metni asistan mesajı olarak yayınlayıp döngüden `break` ediyor.
CLI konfigürasyon diyaloğu (`crates/goose-cli/src/commands/configure.rs:1918+`) de aynı
sözleşmeyi anlatıyor: *"goose will ask for input after N consecutive actions"*. Yani
`GOOSE_MAX_TURNS` bir **otonomi bütçesi**, bir görev bütçesi değil — kullanıcı "devam" derse
sayaç sıfırdan başlıyor. Ayrıca `compaction_attempts >= 2` (satır 3084) ile sıkıştırma
denemeleri sınırlı.

### 2.8 opencode (`sst/opencode` → **`anomalyco/opencode`**) `[K]`

⚠️ **Depo taşınmış**: `github.com/sst/opencode` → `github.com/anomalyco/opencode` (HTTP 301).
Varsayılan dal `dev`. `Kilo-Org/kilocode` aynı çekirdeği fork'lamış durumda.

**Loop detection — "doom loop" ve envanterdeki en özgün tepki biçimi.**
Dosya: `packages/opencode/src/session/processor.ts`

```ts
const DOOM_LOOP_THRESHOLD = 3
...
const recentParts = parts.slice(-DOOM_LOOP_THRESHOLD)
if (recentParts.length !== DOOM_LOOP_THRESHOLD ||
    !recentParts.every((part) => part.type === "tool" && part.tool === value.name &&
        part.state.status !== "pending" &&
        JSON.stringify(part.state.input) === JSON.stringify(input))) { return }
...
yield* permission.ask({ permission: "doom_loop", patterns: [value.name], ..., ruleset: agent.permission })
```

Heuristik Cline/Roo ile aynı ailede (son 3 parça aynı araç + aynı JSON girdi; `JSON.stringify`
anahtar sırasını kanonikleştirmiyor, yani Cline/Roo'dan biraz daha kırılgan).

**Ama tepki tamamen farklı: döngü tespiti bir izin (permission) olayına dönüştürülüyor.**
Ayrı bir "durdur" yolu yok; `"doom_loop"` diğer araç izinleri gibi agent'ın
`permission` ruleset'inden geçiyor. Pratik sonucu: aynı mekanizma konfigürasyona göre
kullanıcıya sorabiliyor, otomatik reddedebiliyor ya da (`always: [value.name]` ile)
kullanıcı bir kez "hep izin ver" derse **tamamen devre dışı kalabiliyor**. Loop detection'ı
mevcut yetkilendirme katmanına yeniden kullandıran tek örnek bu.

**Adım limiti — varsayılan sınırsız, tepkisi zarif bozulma.**
`packages/opencode/src/session/prompt.ts:1178–1179`:

```ts
const maxSteps = agent.steps ?? Infinity
const isLastStep = step >= maxSteps
```

Ana döngü `while (true)` (satır 1088) ve sayaç yalnızca `isLastStep` hesabı için tutuluyor.
`steps` şemada her yerde **isteğe bağlı** (`packages/core/src/config/agent.ts:22`,
`packages/schema/src/agent.ts:29`; açıklaması *"Maximum number of agentic iterations before
forcing text-only response"*) ve yerleşik agent tanımlarının (`build`, `plan`, `general` —
`packages/opencode/src/agent/agent.ts:148+`) hiçbiri `steps` set etmiyor. Yani
**kutudan çıktığı hâliyle opencode'da adım sınırı yok**; kullanıcı `opencode.json` içinde
`agent: { build: { steps: 50 } }` yazmak zorunda.
Limite gelindiğinde istisna atılmıyor; konuşmaya bir asistan mesajı ekleniyor
(`packages/core/src/session/runner/max-steps.ts`):

> `CRITICAL - MAXIMUM STEPS REACHED` … "Tools are disabled until next user input. Respond
> with text only." … "Statement that maximum steps for this agent have been reached /
> Summary of what has been accomplished so far / List of any remaining tasks that were not
> completed / Recommendations for what should be done next"

Yeni çekirdek koşucusunda (`packages/core/src/session/runner/llm.ts`) buna **`toolChoice:
isLastStep ? "none" : undefined`** eşlik ediyor — yani araçlar prompt'la rica edilmiyor,
protokol düzeyinde kapatılıyor. smolagents'in `_handle_max_steps_reached()` yaklaşımının
en gelişmiş hâli: model son turda **yapılanı özetlemek ve kalanı listelemek zorunda**.
PoC'de "zarif bozulma" yazacaksak kopyalanacak metin bu.

`packages/opencode/src/session/retry.ts` ayrıca `RETRY_MAX_RETRIES = 5` ve
`RETRY_MAX_DELAY_NO_HEADERS = 30_000` ms ile API yeniden denemelerini sınırlıyor
(sağlayıcı rate-limit'i; görev bütçesi değil).

### 2.9 Continue (`continuedev/continue`) `[K]`

**Bu envanterde hiçbir sınırı olmayan tek harness.** Ne loop detection, ne adım, ne token,
ne maliyet, ne süre limiti var. İki ayrı çalıştırma yolu da ayrı ayrı incelendi.

**GUI (VS Code / JetBrains) yolu** — `gui/src/redux/thunks/streamNormalInput.ts`.
Bir `depth` sayacı var ama **üretimde hiçbir şey yapmıyor** (satır 85–89):

```ts
if (process.env.NODE_ENV === "test" && depth > 50) {
  const message = `Max stream depth of ${50} reached in test`;
  console.error(message, JSON.stringify(getState(), null, 2));
  throw new Error(message);
}
```

Guard `NODE_ENV === "test"` ile korunuyor; üretim derlemesinde koşul asla sağlanmıyor.
Kodun kendi yorumu da niyeti açıkça söylüyor (satır 364):
`// auto stream cases increase thunk depth by 1 for debugging`. Yani `depth` bir
**hata ayıklama sayacı**, bir bütçe değil. Araç çağrısı → `callToolById({depth: depth + 1})`
→ `streamResponseAfterToolCall({depth: depth + 1})` zinciri sınırsız derinleşebiliyor.

**CLI yolu** — `extensions/cli/src/stream/streamChatResponse.ts`, satır 443:
`while (true)` — iterasyon sayacı hiç yok. Döngüden çıkış tek koşula bağlı (satır 577–580):

```ts
// Check if we should continue (skip break if auto-continuing after compaction)
if (!shouldContinue && !shouldAutoContinue) { break; }
```

`shouldContinue` model araç çağırdığı sürece `true`. Yani koşum **yalnızca model kendi
kendine durmaya karar verdiğinde** bitiyor — brief'in "agent kendi adımlarına kendi karar
veriyor, ne zaman duracağı garanti edilmeli" cümlesinin birebir karşı örneği.

Bulunan tek koruma sıkıştırmayla ilgili: `extensions/cli/src/compaction.ts:104`
budama geçmişi değiştirmediyse döngüyü kırıyor (*"prevents infinite loop"*) ve
`compactionOccurredThisTurn` bayrağı sıfırlanıyor (*"Reset flag to avoid infinite
continuation"*, `streamChatResponse.ts:572–575`). İkisi de **iç mekanizmanın kendi
döngüsüne** karşı; modelin araç döngüsüne karşı değil. Depoda
`extensions/cli/src/compaction.infiniteLoop.test.ts` diye bir test dosyasının bulunması,
bu sınıf hatanın projede fiilen yaşandığını gösteriyor.

Not: bu bulgu `main` dalındaki mevcut kod içindir. Bir konfigürasyon anahtarıyla açılan
gizli bir limit aranmadı; `maxTurns`, `MAX_TOOL_CALLS`, `toolCallLimit`, `MAX_AUTO`
desenleri hem grep.app hem GitHub Code Search ile tarandı, **0 sonuç**.

---

## 3. Gateway ve gözlemlenebilirlik katmanı — zorluyor mu, sadece raporluyor mu?

Bu katmana sorulan tek soru: **bir eşiğe gelince isteği reddediyor mu, yoksa yalnızca
sayıyı kaydedip gösteriyor mu?** Cevap keskin biçimde ikiye ayrılıyor.

### 3.1 Helicone — zorluyor `[K]`

Envanterde LiteLLM dışında **gerçekten zorlama yapan tek gözlemlenebilirlik ürünü**.
Mekanizma bir HTTP başlığı: `Helicone-RateLimit-Policy`.

Dosya: `worker/src/lib/rate-limit/policyParser.ts`. Dilbilgisi (satır 66):

```ts
/^(\d+(?:\.\d+)?);w=(\d+)(?:;u=(request|cents))?(?:;s=([\w-]+))?$/i
```

`[kota];w=[pencere_saniye];u=[birim];s=[segment]`. Dosyanın kendi örnekleri:

- `"1000;w=3600"` → saatte 1000 istek, global
- `"5000;w=86400;u=cents"` → **günde 5000 sent (50 $)**, global
- `"0.5;w=60;u=cents"` → dakikada yarım sent (küçük bütçeleri test etmek için)
- `"100;w=60;s=user"` → kullanıcı başına dakikada 100 istek

**`RateLimitUnit = "request" | "cents"`** (satır 19) — yani dolar bütçesi birinci sınıf bir
birim, sonradan eklenmiş bir rapor alanı değil. Segment üç tip:
`{type:"global"} | {type:"user"} | {type:"property"; name}` (satır 21–24) — özel bir
Helicone property'sine göre bölmek, "görev başına bütçe"yi bu katmanda kurmanın yolu.

Doğrulama kısıtları: pencere **en az 60 saniye**, en fazla 31.536.000 saniye (1 yıl);
kota pozitif ve ondalıklı olabilir (satır 81–110). Aşıldığında
`worker/src/lib/ResponseBuilder.ts:buildRateLimitedResponse()` → **HTTP 429** ve
`X-Helicone-Error: rate_limited` başlığı.

Not: pencerenin ≥ 60 s olması, bunu "tek bir görevin adım bütçesi" değil **zaman pencereli
bir hız/harcama tavanı** yapıyor. PoC'de per-task bütçe için LiteLLM'in kapsam modeli
(`loop_budget.md` §4c) daha yakın; Helicone'un katkısı `cents` biriminin ve segment
kavramının sadeliği.

### 3.2 Langfuse — yalnızca raporluyor `[K]`

`web/src/features/public-api/server/RateLimitService.ts` içinde bir rate limit servisi var
ama kapsamı yanlış anlaşılmaya çok müsait: dosyanın kendi yorumu (satır 47) diyor ki
*"rate limit strategy is based on org-id, org plan, and resources"*. Yani sınırlanan şey
**Langfuse'un kendi API kaynaklarına yapılan istekler** (org planına göre), agent'ın LLM
harcaması değil. `points` / `durationInSec` konfigürasyonu plan bazlı
(`getPlanBasedRateLimitConfig`, satır 239) ve aşıldığında Langfuse'un kendi endpoint'i
429 dönüyor.

`budget` / `spendLimit` / `usage_limit` desenleri repoda tarandı: agent bütçesine dair
tek eşleşme `web/src/features/widgets/chart-library/useChartTickBudget.ts` — grafik
ekseninde kaç tick çizileceğiyle ilgili, konuyla alakasız. **Langfuse bir kontrol
düzlemi değil, bir gözlem düzlemi.**

### 3.3 W&B Weave — yalnızca raporluyor `[D]`

Maliyet hesabı `weave/trace_server/token_costs.py` içinde: model fiyatlarını
(`llm_token_prices` tablosu, `costs/insert_costs.py` ile dolduruluyor) çağrı özetindeki
token kullanımıyla çarpıp sonucu `summary_dump` alanına yazıyor. `WeaveClient.add_cost()`
ile elle fiyat eklenebiliyor. Hiçbir noktada bir eşikle karşılaştırma ya da çağrıyı
engelleme yok. `cost_limit` deseni repoda **0 sonuç**.

⚠️ Etiket `[D]`: bu bölüm DeepWiki cevabına dayanıyor; adı geçen dosyaların varlığı GitHub
Code Search ile ayrıca doğrulandı (`token_costs.py`, `costs/insert_costs.py`,
`agents/span_costs.py` mevcut) ama dosyaların içi satır satır okunmadı.

### 3.4 AgentOps — yalnızca raporluyor, loop detection **yol haritasında** `[K]`

Bu, "yok" cevabının bir bulgu olduğu en net vaka. AgentOps'un kendi README'sindeki
"Debugging Roadmap" tablosu durumu açıkça yazıyor:

| Özellik | Durum (README'den) |
|---|---|
| Infinite loops and recursive thought detection | 🚧 (yapım aşamasında) |
| Token limit overflow flags | 🚧 |
| Context limit overflow flags | 🔜 (planlanan) |
| API bill tracking | ✅ |
| Agent workflow execution pricing | ✅ |

Yani maliyet **takibi** hazır, döngü **tespiti** değil. `agentops/config.py` içindeki
tek "max" parametreleri `max_wait_time` ve `max_queue_size` — bunlar telemetri
kuyruğunun batch'leme ayarları, bütçe değil.

DeepWiki de bağımsız olarak aynı sonucu verdi ("does not currently enforce per-task budget
limits… roadmap indicates future plans"); README ve `config.py` doğrudan okunarak
doğrulandı.

### 3.5 OpenTelemetry GenAI semantic conventions `[K]`

⚠️ **Depo taşınmış.** Brief'te verilen `open-telemetry/semantic-conventions` altındaki
`docs/gen-ai/` sayfaları artık şu uyarıyı taşıyor:

> "GenAI semantic conventions have moved to the
> [OpenTelemetry GenAI semantic conventions repository](https://github.com/open-telemetry/semantic-conventions-genai).
> This page has moved and is no longer maintained in this repository."

Eski depoda `model/gen-ai/` altında yalnızca `deprecated/` klasörü kalmış. Güncel kaynak:
**`open-telemetry/semantic-conventions-genai`**, `model/gen-ai/{registry,metrics,spans}.yaml`.

**Token için zengin, maliyet için hiçbir şey.** `model/gen-ai/registry.yaml` içinde
14 ayrı `gen_ai.usage.*` token alanı var — toplamlar ve alt kırılımlar:

```
gen_ai.usage.input_tokens                    gen_ai.usage.output_tokens
gen_ai.usage.cache_read.input_tokens         gen_ai.usage.reasoning.output_tokens
gen_ai.usage.cache_write.input_tokens        gen_ai.usage.text.output_tokens
gen_ai.usage.text.input_tokens               gen_ai.usage.image.output_tokens
gen_ai.usage.image.input_tokens              gen_ai.usage.audio.output_tokens
gen_ai.usage.audio.input_tokens              gen_ai.usage.text.cache_read.input_tokens
gen_ai.usage.image.cache_read.input_tokens   gen_ai.usage.audio.cache_read.input_tokens
```

Hiyerarşi açık: alt kırılımlar toplamın alt kümesi (`registry.yaml:279–285` örneği:
100 metin token'ı (40'ı cache) + 200 görsel token'ı → `input_tokens: 300`).
Faturalama konusunda net bir kural var: sağlayıcı hem "faturalanan" hem "tüketilen"
sayıyı veriyorsa **enstrümantasyon faturalananı raporlamalı**
(`registry.yaml:274–277`, `metrics.yaml:37`: *"instrumentation MUST report billable tokens"*).

**Maliyet alanı yok.** `registry.yaml`, `metrics.yaml` ve `spans.yaml` dosyalarında
`cost` / `currency` / `usd` / `price` / `billing` / `budget` kelimelerinin toplam geçiş
sayısı **0**. Standart parayı hiç konuşmuyor; token'ı sayıp fiyatlandırmayı geride
bırakıyor. Bu, gateway katmanının (LiteLLM, Helicone) neden kendi maliyet modelini taşımak
zorunda kaldığının standart tarafındaki açıklaması.

**Bizim konumuz için doğrudan yararlı iki metrik var** (`model/gen-ai/metrics.yaml`):

| Metrik | Birim | Ne sayıyor |
|---|---|---|
| `gen_ai.invoke_agent.inference_calls` | `{inference_call}` | Bir agent çağrımında agent'ın **kendi** yaptığı model çağrısı sayısı (histogram) |
| `gen_ai.invoke_agent.tool_calls` | `{tool_call}` | Bir agent çağrımında agent'ın **kendi** tetiklediği araç çağrısı sayısı (histogram) |
| `gen_ai.invoke_agent.duration` | `s` | Agent çağrımı süresi |
| `gen_ai.client.token.usage` | `{token}` | Girdi + çıktı token'ı (`gen_ai.token.type` ile ayrılır) |

`inference_calls` ve `tool_calls`, per-task adım bütçesinin **standartlaşmış ölçüm
karşılığı** — yani PoC'de "max steps" sayacını bu isimlerle dışa vurursak standarda
uyumlu oluruz. Muhasebe kuralı da tanımlı: alt-agent'ların ve devredilen agent'ların
çağrıları **kendi çağrımlarına** yazılıyor, böylece çağrı ağacında her çağrı tam bir kez
sayılıyor (`metrics.yaml:191–196`). Ayrıca sunucu tarafında sağlayıcının çalıştırdığı
built-in araçlar (web arama, kod çalıştırma) `tool_calls`'a **dahil edilmiyor**.

Tüm bu metrikler ve `gen_ai.usage.*` alanları `stability: development` — yani henüz
kararlı (stable) değil, değişebilir.

---

## 4. Framework katmanı — Agno, Letta, DSPy

### 4.1 Agno (`agno-agi/agno`) `[K]`

**Loop detection yok.** Tek sınır `tool_call_limit: Optional[int] = None` — yani
**varsayılan sınırsız** (`libs/agno/agno/team/team.py:265`, aynı alan `Agent` tarafında da
var; yorumu: *"Maximum number of tool calls allowed."*).

Limite gelindiğinde **istisna atılmıyor, araç sonucu sahteleniyor**
(`libs/agno/agno/models/base.py:2123–2131`):

```python
def create_tool_call_limit_error_result(self, function_call: FunctionCall) -> Message:
    return Message(
        role=self.tool_message_role,
        content=f"Tool call limit reached. Tool call {function_call.function.name} not executed. Don't try to execute it again.",
        ...
        tool_call_error=True,
    )
```

Model bunu normal bir araç hatası gibi görüyor ve koşum devam ediyor. Agno'nun kendi
değerlendirme motoru bu tasarımın sonucunu kayda geçmiş — envanterdeki **en dürüst kod
yorumu** (`libs/agno/agno/environments/_engine.py:51–53`):

> "A fact, not a policy: `tool_call_limit` refuses further calls but the run still
> completes, so exhaustion is invisible to a status check. The attempt is still scored;
> downstream consumers filter on the flag."

Bu yüzden değerlendirme sonucuna ayrı bir `tool_call_limit_hit: bool` bayrağı eklemişler
(`_engine.py:57`, `runner.py:243`). **Bütçe tükenmesinin "başarı" gibi görünmesi** —
`loop_budget.md` §3'teki olay kataloğunun sessiz başarısızlık kategorisinin kod düzeyinde
kanıtı ve PoC'de kaçınmamız gereken tuzağın adı.

Agno'da ayrıca `reasoning_min_steps` / `reasoning_max_steps` var ama bunlar akıl yürütme
zincirinin uzunluğunu **prompt metniyle** rica ediyor (`libs/agno/agno/reasoning/default.py:80`:
*"Adhere strictly to a minimum of {min_steps} and maximum of {max_steps} steps"*) —
zorlanan bir sayaç değil.

### 4.2 Letta (`letta-ai/letta`) — ⚠️ depo boşaltılmış `[K]`/`[?]`

**Bu bir depo taşınması değil, bir kaynak kodu çekilmesi.** `letta-ai/letta` deposunun
`main` dalında artık **hiç kaynak kodu yok** — yalnızca 9 dosya (README, LICENSE, AGENTS.md,
PRIVACY/TERMS/SECURITY vb.) ve bir `.github` klasörü. README durumu kendisi anlatıyor:

> "This repository now serves as a landing page for the Letta project. The retired Letta V1
> server source is preserved on the [`archive`](https://github.com/letta-ai/letta/tree/archive)
> branch for historical reference."

Güncel kaynak: **`letta-ai/letta-code`** (TypeScript). ⚠️ Bu, grep.app indeksinin eski
kaldığı somut bir vaka: grep.app hâlâ `letta/agents/letta_agent_v3.py` gibi dosyaları
döndürüyor, ama `main`'de o yol yok (`gh api .../contents/letta/agents` → 404).

**Arşiv dalındaki Python V1 sunucusu** `[K]`: `letta/constants.py:70` →

```python
# Max steps for agent loop
DEFAULT_MAX_STEPS = 50
```

Tüm ajan sınıfları (`letta_agent.py`, `letta_agent_v2.py`, `letta_agent_v3.py`,
`sleeptime_multi_agent_v2/v3/v4.py`, `letta_agent_batch.py`, `voice_sleeptime_agent.py`)
`step(..., max_steps: int = DEFAULT_MAX_STEPS, ...)` imzasını paylaşıyor. Loop detection
mekanizması bulunamadı.

**Yeni TypeScript istemcisinde** `[?]`: `src/types/protocol.ts:335` şunu diyor —
*"Uses `StopReasonType` from `letta-client` (e.g., 'error', **'max_steps'**, 'llm_api_error')"*.
Yani `max_steps` hâlâ bir sonlanma sebebi, **ama limit istemcide tanımlı değil**:
zorlama Letta App Server / Letta Cloud tarafında yapılıyor ve istemci yalnızca
`stop_reason` alıyor. Sunucu tarafı bu depoda olmadığı için **güncel varsayılan değer
doğrulanamadı** — 50 rakamı yalnızca arşivlenmiş V1 için geçerlidir.

### 4.3 DSPy (`stanfordnlp/dspy`, `ReAct`) `[K]`

**Loop detection yok.** Tek sınır `max_iters`.
Dosya: `dspy/predict/react.py`

```python
def __init__(self, signature: type["Signature"], tools: list[Callable], max_iters: int = 20):
```

⚠️ **Kod ile docstring çelişiyor**: imza `max_iters: int = 20` diyor, hemen altındaki
docstring (satır 28) *"The maximum number of iterations to run. Defaults to 10."* diyor.
Gerçek varsayılan **20**. (Bu tür tutarsızlıklar tam da dokümantasyona değil koda bakmayı
gerektiren şey.)

Limit çağrı başına ezilebiliyor (satır 97): `max_iters = input_args.pop("max_iters", self.max_iters)`.

**Tepki — zarif bozulma.** `for idx in range(max_iters)` döngüsü sessizce bitiyor ve
biriken `trajectory` ayrı bir `extract` modülüne veriliyor (satır 120):

```python
extract = self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
```

Yani smolagents'in `_handle_max_steps_reached()` deseniyle aynı felsefe: bütçe bitince
istisna değil, **o ana kadar toplanan gözlemlerden bir cevap üret**. Döngü ayrıca araç
seçim hatası ve model hatası durumunda da `break` ediyor (satır 102–106) — bunlar
loop detection değil, hata kaçışı.

---

## 5. Desen sentezi

Bu bölüm sunumun ana malzemesi. Sayılar yukarıdaki tabloda kod düzeyinde incelenen
**28 satır** üzerinden: 22 agent harness/framework + 5 gateway/gözlemlenebilirlik katmanı
+ 1 standart (OTel). Loop detection ve adım limiti sayımları yalnızca **22 harness**
üzerinden; gateway katmanı ayrı sayılıyor.

### 5.1 Loop detection: azınlık, ve yenilerde yoğunlaşıyor

| Durum | Sayı | Kimler |
|---|---|---|
| **Gerçek tekrar/stuck dedektörü var** | **6** | OpenHands, Gemini CLI, Cline, Roo Code, opencode, deer-flow |
| Dolaylı — hata/timeout sayacı var, döngü heuristiği yok | 3 | SWE-agent, Aider, Goose |
| **Hiç yok** | **13** | Codex CLI, Continue, LangGraph, smolagents, CrewAI, pydantic-ai, OpenAI Agents SDK, AutoGen, Google ADK, Claude Agent SDK, Agno, Letta, DSPy |

*(22 harness = 6 + 3 + 13. Gateway/gözlemlenebilirlik katmanının hiçbirinde de loop
detection yok; AgentOps'ta yol haritasında 🚧 olarak duruyor.)*

**Çıkarım 1 — loop detection bir kodlama-agent'ı özelliği, bir framework özelliği değil.**
Dedektörü olan 6'nın 5'i uzun soluklu, otonom kodlama harness'ı. Genel amaçlı
framework'lerin (LangGraph, CrewAI, pydantic-ai, DSPy, Agno, AutoGen, ADK) **hiçbirinde**
tekrar tespiti yok; hepsi tek bir sayaçla yetiniyor. Bu, `loop_budget.md` §2'deki
IAL-SCAN bulgusuyla birebir örtüşüyor: 68 gerçek döngünün %66,2'si LangGraph + AutoGen'de,
çünkü bu ikisi geri beslemeyi API semantiğiyle kuruyor ve yolu kapsayan bir sınır koymuyor.

**Çıkarım 2 — heuristikler tek bir çekirdek fikrin varyasyonları.** Hepsi "araç çağrısı
imzası" üzerine kurulu; fark kanonikleştirme ve pencere genişliğinde:

| Harness | İmza | Pencere / desen | Eşik |
|---|---|---|---|
| Cline | `JSON.stringify(sortKeys(input))` | yalnızca **ardışık** | soft 3 / hard 5 |
| Roo Code | `safe-stable-stringify({name, params})` | yalnızca **ardışık** | 3 (fiilen 4. çağrıda) |
| opencode | `JSON.stringify(input)` (sıralama yok) | son **3** parça | 3 |
| Gemini CLI | `sha256(name + args)` | **k=1..5 çevrim**, 25 anahtarlık geçmiş | 5 tekrar |
| OpenHands | içerik karşılaştırması (`_event_eq`, ID'siz) | son **20** olay | 4 / 3 / 3 / 6 |

Yani **A-B-A-B tipi dönüşümlü döngüyü yalnızca ikisi yakalıyor**: Gemini CLI (k'yı 5'e
kadar genelleştirerek) ve OpenHands (`alternating_pattern=6`). Cline, Roo ve opencode'un
"ardışık aynı çağrı" heuristiği bu deseni tamamen kaçırıyor — PoC'de kopyalanacak
minimum seviye bu değil, k-çevrim taraması olmalı.

**Çıkarım 3 — LLM yargıcı tek bir yerde ve ihtiyatla kullanılıyor.** Yalnızca Gemini CLI
bir LLM'e "döngüde misin" diye soruyor; onu da 30. turdan önce hiç çalıştırmıyor, güvene
göre 5–15 tur arayla seyreltiyor ve 0,9 güven eşiği koyuyor. Bu, `loop_budget.md` §2'deki
IAL-SCAN sonucuyla uyumlu: LLM yargıcı ucuz deterministik kontrolün **yerine geçmiyor,
üstüne biniyor**.

### 5.2 Bütçe türleri: adım yaygın, para nadir, süre neredeyse yok

| Limit türü | 22 harness'ın kaçında | Not |
|---|---|---|
| **Adım / tur / iterasyon / çağrı** | **19** | Fiilen evrensel. Olmayan üçü: **Cline** (yalnızca mistake sayacı), **Codex CLI**, **Continue** |
| **Token** | 5 | SWE-agent, pydantic-ai, AutoGen, Codex (kapalı) + LiteLLM (gateway). CrewAI'ın `respect_context_window`'u limit değil, budama |
| **Maliyet (dolar)** | **5** | SWE-agent (3.0 $), Claude Agent SDK (`None`), Roo Code (`Infinity`) + LiteLLM, Helicone (gateway) |
| **Wall-clock süre** | **4** | SWE-agent (1800 s), CrewAI, AutoGen, Cline CLI |

**Çıkarım 4 — "adım" ölçüsü fiili standart, ama en kötü vekil (proxy).** Bir adımın
maliyeti 1.000 token da olabilir 500.000 token da; `loop_budget.md` §1'deki 50× maliyet
salınımı tam olarak bunun sonucu. Yine de harness'ların büyük çoğunluğu yalnızca adım
sayıyor, çünkü ölçmesi bedava. **Token ve dolar limitini birlikte koyan tek harness
SWE-agent.**

**Çıkarım 5 — varsayılanı dolu tek dolar limiti SWE-agent'ta.**
`per_instance_cost_limit: float = 3.0`. Diğer bütün dolar/token limitleri kutudan
`None` / `0` / `Infinity` çıkıyor: Claude Agent SDK `max_budget_usd=None`,
Roo Code `allowedMaxCost || Infinity`, SWE-agent `total_cost_limit=0.0`,
Codex `rollout_budget` feature'ı kapalı, LiteLLM'de bütçe elle tanımlanıyor.
**Yani "para bitince dur" davranışı sektörde varsayılan değil, opt-in.**

### 5.3 Varsayılanı kapalı ya da sınırsız olanlar — sunumun en çarpıcı listesi

Bir mekanizmanın var olması, açık olması demek değil:

| Harness | Mekanizma | Kutudan çıktığı hâl |
|---|---|---|
| **Google ADK** | `LoopAgent.max_iterations` | `None` = **sınırsız** |
| **Gemini CLI** | `maxSessionTurns` | `-1` = **sınırsız** (loop detection ise açık) |
| **opencode** | `agent.steps` | `Infinity` |
| **Agno** | `tool_call_limit` | `None` |
| **Roo Code** | `allowedMaxRequests`, `allowedMaxCost` | `Infinity` (ikisi de) |
| **Claude Agent SDK** | `max_turns`, `max_budget_usd` | `None` (ikisi de) |
| **Codex CLI** | `token_budget`, `rollout_budget` | `Stage::UnderDevelopment`, `default_enabled: false` |
| **Cline** | loop detection | **core'da kapalı**, yalnızca CLI açıyor |
| **Cline** | `reminderAfterIterations` | `0` = kapalı |
| **SWE-agent** | `total_cost_limit`, `per_instance_call_limit` | `0` = kapalı |
| **Continue** | (hiçbiri) | mekanizma **hiç yok** |

Kaba sayım: 22 harness'ın **10'unda** en az bir guardrail
mekanizması var ama varsayılan konfigürasyonda **çalışmıyor**.

**Çıkarım 6 — "harness'ta loop detection var mı" yanlış soru.** Doğru soru:
**"varsayılan konfigürasyonda etkin mi"**. Cline'ın kendi kodu bunu itiraf ediyor
(*"The agent core leaves loop detection off by default"*), Codex'inki de
(`default_enabled: false`). Atlas'a öneri yazarken ölçüt bu olmalı.

**Çıkarım 7 — aynı harness bağlama göre çok farklı varsayılan koyabiliyor.** Üç örnek,
üçü de aynı depo içinde:

| Harness | Bağlam A | Bağlam B | Oran |
|---|---|---|---|
| Goose | CLI/desktop `GOOSE_MAX_TURNS=1000` | Telegram gateway `DEFAULT_GATEWAY_MAX_TURNS=5` | **200×** |
| Roo Code | VS Code eklentisi `3` | CLI `DEFAULT_FLAGS = 10` | 3,3× |
| Cline | core `?? 6` | CLI `args.retries ?? 3` | 2× |

**Bütçe teknik bir sabit değil, ürün kararı** — ve iyi harness'lar bunu koda yazılı
gerekçeyle yapıyor (Goose'un yorumu: *"Chat platforms like Telegram favor short, snappy
replies"*). Pratik sonuç: bir harness'ın "varsayılanı" diye tek bir sayı aktarmak yanlış;
hangi arayüzden koşulduğu sorulmalı.

### 5.4 Limit dolunca ne oluyor — dört strateji

| Strateji | Kaç | Kimler |
|---|---|---|
| **Sert durdurma** (istisna / terminal durum) | **10** | LangGraph `GraphRecursionError`, pydantic-ai `UsageLimitExceeded`, OpenAI Agents SDK `MaxTurnsExceeded`, Google ADK `LlmCallsLimitExceededError`, LiteLLM `BudgetExceededError`(429), Codex `SessionBudgetExceeded`, Helicone HTTP 429, deer-flow, OpenHands `STUCK`, Cline (hard) |
| **Nudge** (prompt'a/geçmişe uyarı, koşum devam) | **5** | Cline (soft, 3), Codex (`reminder_at_remaining_tokens` listesi), Goose (`<turn-budget>` yarıda), Agno (sahte araç hatası), Gemini CLI (`_recoverFromLoop`, 1. tespit) |
| **Zarif bozulma** (modelden nihai cevap) | **4** | smolagents `_handle_max_steps_reached()`, SWE-agent `attempt_autosubmission_after_error()`, opencode `MAX_STEPS_PROMPT`, DSPy `self.extract` |
| **Kullanıcıya sorma** | **3** | Goose ("Would you like me to continue?"), Roo Code (`ask("mistake_limit_reached")`), opencode (`permission.ask({permission:"doom_loop"})`) |
| **Geri sarma (rollback)** | **0** | Hiçbiri. Cline/Roo checkpoint tutuyor ama limit dolunca **otomatik geri sarmıyor** |

**Çıkarım 8 — kademelendirme en iyi harness'ların ortak imzası.** Tek başına "sert"
ya da tek başına "nudge" seçen harness'lar zayıf uçta; en olgun dördü **iki ya da üç
kademe** kuruyor:

- **Gemini CLI**: 1. tespit → kurtarma turu; 2. tespit → dur.
- **Cline**: 3 tekrar → geçmişe nudge; 5 tekrar → dur.
- **Codex**: eşik listesinde her geçişte nudge (`75, 50, 25`); sıfırda sert kesme.
- **Goose**: bütçenin yarısında `<turn-budget>` farkındalık mesajı; sonunda kullanıcıya sor.

**PoC için tavsiye edilecek şablon bu**: tek eşik yerine `soft → nudge`, `hard → stop`,
ve mümkünse arada bir "toparlan" turu.

**Çıkarım 9 — modele kendi bütçesini söylemek yeni ve nadir bir fikir.** Yalnızca ikisi
yapıyor: Goose (`<turn-budget>{n}/{max} used`) ve Codex
(`<rollout_budget>You have N weighted tokens left…`). İkisi de sınırı gizli bir tavan
olmaktan çıkarıp **modelin planlama girdisine** dönüştürüyor. Ucuz, uygulaması kolay ve
sunumda ayırt edici bir öneri.

**Çıkarım 10 — bütçe tükenmesinin sessizce başarı gibi görünmesi gerçek bir tuzak.**
Agno'nun kod yorumu (*"exhaustion is invisible to a status check"*) ve SWE-agent'ın
ayrı çıkış statüleri (`exit_cost`, `exit_context`, `exit_total_execution_time`) aynı
sorunun iki ucu. **Her limitin ayrı ve makine-okunur bir sonlanma sebebi olmalı** —
OpenHands'in `ConversationExecutionStatus.STUCK`'ı ayrı bir terminal durum yapması,
Letta'nın `stop_reason: 'max_steps'`'i, Agno'nun `tool_call_limit_hit` bayrağı hep bunun
farklı çözümleri. PoC'de ölçülebilirlik buna bağlı.

### 5.5 Hiyerarşik bütçe — iki referans implementasyon

Alt-agent'lar konusunda iki farklı model var, ikisi de kopyalanabilir:

- **Codex — paylaşımlı tek sayaç.** `Arc<RolloutBudget>` `AgentControl` üzerinde; kök
  thread ve tüm alt-agent'lar aynı sayaca yazıyor (*"Shared token budget for the root
  thread and its sub-agents"*).
- **SWE-agent — devredilen bütçe.** Her denemeden önce kalan bütçe hesaplanıp alt-agent'ın
  limiti ona kısılıyor, üstüne `> 1.1 * cost_limit` toleranslı bir ikinci ağ konuyor.

OTel tarafında da muhasebe kuralı tanımlı: alt-agent çağrıları **kendi** çağrımlarına
yazılıyor, böylece ağaçta her çağrı tam bir kez sayılıyor.

---

## 6. Boşluklar — dürüst kayıt

**İncelenemeyenler / eksik kalanlar:**

1. **Letta'nın güncel adım limiti doğrulanamadı** `[?]`. `letta-ai/letta` `main` dalında
   kaynak kodu yok; `letta-ai/letta-code` istemcisi limiti sunucudan `stop_reason` olarak
   alıyor ve sunucu kodu açık değil. `DEFAULT_MAX_STEPS = 50` yalnızca **arşiv dalındaki
   Python V1** için geçerli. Güncel Letta Cloud/App Server varsayılanı bilinmiyor.
2. **W&B Weave `[D]` etiketli.** Dosya adları GitHub Code Search ile doğrulandı ama
   `token_costs.py` satır satır okunmadı; "hiçbir yerde eşikle karşılaştırmıyor" iddiası
   DeepWiki cevabına dayanıyor. Zayıf halka.
3. **Gemini CLI'ın `checkContentLoop` iç ayrıntısı yüzeysel incelendi.** `CONTENT_LOOP_THRESHOLD=10`,
   `CONTENT_CHUNK_SIZE=50`, `MAX_HISTORY_LENGTH=5000` sabitleri okundu; `isLoopDetectedForChunk`
   içindeki mesafe hesabı (`Math.floor(CONTENT_LOOP_THRESHOLD / 2)`) ve
   `isActualContentMatch` hash çakışma kontrolü satır satır doğrulanmadı.
4. **Roo Code'un `commandExecutionTimeout: 300` değeri zayıf kaynaklı.** Yalnızca
   `apps/cli/src/agent/extension-host.ts:219+` içindeki başlangıç ayarından alındı;
   zorlamanın nerede yapıldığı ve eklenti tarafındaki varsayılanı izlenmedi.
   (`DEFAULT_FLAGS.consecutiveMistakeLimit = 10` ise `apps/cli/src/types/constants.ts`'ten
   doğrudan okundu — bu boşluk kapatıldı.)
5. **Codex'in `token_budget` model varsayılanları okunamadı.** `apply_model_defaults()`
   değerleri `model_info.model_messages.token_budget` üzerinden **sunucudan** alıyor;
   `reminder_threshold_tokens`'ın gerçekte hangi sayı olduğu depoda yok. Aynı şey
   `rollout_budget.limit_tokens` için de geçerli — depodaki tek somut değer
   (`limit_tokens: 100`, `reminder_at_remaining_tokens: vec![75, 50, 25]`) bir **test
   fixture'ı**, üretim varsayılanı değil.
6. **Gözlemlenebilirlik katmanının ticari/bulut tarafı kapsam dışı.** Langfuse Cloud,
   Helicone Cloud, AgentOps Cloud'un panel üzerinden tanımlanan (kodda olmayan) bütçe
   uyarıları olabilir; yalnızca açık kaynak depolar incelendi.

**Zayıf iddialar (etiketlerine dikkat):**

- Weave'in "zorlamıyor" iddiası `[D]`.
- Letta'nın güncel varsayılanı `[?]`.
- AgentOps'un yol haritası satırları README'den okundu `[K]` ama "hiç kod yok" iddiası
  yalnızca `config.py` + iki arama üzerinden; SDK'nın tamamı taranmadı.

**Depo taşınmaları — bu envanterde ortaya çıkan dört tanesi:**

| Brief'te verilen | Gerçek güncel adres | Not |
|---|---|---|
| `block/goose` | **`aaif-goose/goose`** | HTTP 301; eski raw URL'ler hâlâ çalışıyor |
| `sst/opencode` | **`anomalyco/opencode`** | HTTP 301; varsayılan dal `dev` |
| `open-telemetry/semantic-conventions` (gen-ai) | **`open-telemetry/semantic-conventions-genai`** | Eskisinde yalnızca `deprecated/` kaldı |
| `letta-ai/letta` | **`letta-ai/letta-code`** | Eski depo landing page'e dönüşmüş; V1 `archive` dalında |

⚠️ **Araç güvenilirliği notu:** grep.app indeksi bu taşınmalarda **eski hâli** döndürüyor
(Letta'nın silinmiş Python dosyalarını hâlâ listeliyor). GitHub Code Search ise doğru
ama saatlik kotası düşük — tarama sırasında bir kez HTTP 403 rate limit'e takıldı.
Her "bulamadım" sonucu ikinci bir araçla doğrulandı.
