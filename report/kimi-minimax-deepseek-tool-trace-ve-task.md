# Yeni Ajanlar — Kimi Code · MiniMax Mini-Agent · DeepSeek-code

> Bu belge, sonradan klonlanan üç ajanı **iki ayrı bölümde** inceler (istek üzerine tasarım): **önce BÖLÜM 1 = Tool-Trace Compaction**, **sonra BÖLÜM 2 = Task Management**. Her bölümde üç ajan alt alt işlenir; sonunda yerleşim tablosu vardır.
>
> Kaynak: `harnesses/kimi-code`, `harnesses/minimax-mini-agent`, `harnesses/deepseek-code` (kaynak koddan sökülmüştür).

## Ajanlar bir bakışta

| Ajan | Repo | Statü | Dil | Not |
|---|---|---|---|---|
| **Kimi Code** | `MoonshotAI/kimi-code` | **resmî** | TypeScript (pnpm monorepo) | olgun; ACP-native; v1→v2 geçişte |
| **MiniMax Mini-Agent** | `MiniMax-AI/Mini-Agent` | **resmî** | Python | temiz tek-ajan demo; M2.5 modeli |
| **DeepSeek-code** | `yksanjo/deepseek-code` | ⚠️ **topluluk** | Python | DeepSeek'in RESMÎ kod-ajanı yok (sadece DeepSeek-Coder *modeli*) |

---

# BÖLÜM 1 — TOOL-TRACE COMPACTION

*(Bir işin İÇİNDEKİ tool çıktılarını, context penceresine sığdırmak için küçültme. İki ekol: deterministik kural vs LLM-özet.)*

## 1.1 Kimi Code — **iki katmanlı hibrit** (en olgunlarından)

Kimi bu işi iki seviyede yapar:

**Katman A — per-tool-result deterministik kırpma** (`packages/agent-core-v2/src/tool/result-builder.ts`): her tool çıktısı üretilirken `maxChars` (DEFAULT_MAX_CHARS) + `maxLineLength` ile kırpılır; işaret `[...truncated]` + *"Output is truncated to fit in the message."* → B-adımı çıktısı üretim anında capping'e girer (OpenCode Katman A / Codex `truncate_middle` muadili). Ayrıca `media/image-compress.ts` ile görsel sıkıştırma (multimodal).

**Katman B — LLM handoff-özet full-compaction** (`agent/contextMemory/` + `agent/fullCompaction/`):
- **Event-sourced context:** compaction bir **op**'tur (`context.apply_compaction`); "splice-shaped mutations (clear/applyCompaction/undo)" ve **wire-replay / snapshot reducer** ile **resume edilebilir** (Codex windowing felsefesi).
- **Eşik tetiği** (`fullCompaction/strategy.ts`): `reservedContextSize = 50_000`; `shouldCompact(usedSize)` → kullanılan bağlam limiti aşınca otomatik.
- **Şekil** (`compactionHandoff.ts`): bir LLM özeti üretir (gerçek boyut `summaryOutputTokens` ile ölçülür); user-mesajlarının **head'ini (`COMPACT_USER_MESSAGE_HEAD_TOKENS = 2_000`) + tail'ini** korur, **ortayı elide eder** (`COMPACTION_ELISION_VARIANT = 'compaction_elision'`), sıkışan user-mesajını `COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000`'e sınırlar; tokensBefore/After + droppedCount muhasebesi tutar; token tahmini **enjekte edilebilir** (`TokenEstimate`).

**Ekol:** ikisi birden. En yakın akrabaları **Codex** (handoff-özet + replay/windowing) ve **OpenCode** (per-tool deterministik kırpma).
> **Tek cümle:** Kimi, tool çıktısını üretimde deterministik kırpar (A), context dolunca event-sourced/replay-edilebilir bir LLM handoff-özetiyle sıkıştırır (B) — user head/tail korunur, orta elide edilir.

## 1.2 MiniMax Mini-Agent — **token-limit LLM özeti** (tek katman)

`mini_agent/agent.py` içinde:
- **Tetik (`_summarize_messages`)**: `estimated_tokens > token_limit` **veya** `api_total_tokens > token_limit` — **çift tetik** (yerel `tiktoken cl100k_base` tahmini **ya da** API'nin raporladığı gerçek token). Varsayılan `token_limit = 80_000`.
- **Şekil:** özet sonrası yapı `system → user1 → summary1 → user2 → summary2 → user3 …` — yani **user turları korunur**, aralarındaki asistan/tool işi **LLM özetine** iner (interleaved).
- **Anti-thrash:** `_skip_next_token_check` — özetten hemen sonra bir kontrol atlanır (api token'ı güncellenene dek arka arkaya tetiklenmesin).

**Ekol:** saf **LLM-özet** (deterministik per-tool katmanı yok). OpenClaw / Claude Code auto-compaction / Kimi'nin Katman B'siyle aynı okul; en sade referans implementasyonu.
> **Tek cümle:** MiniMax, tahmini veya gerçek token 80K'yı aşınca konuşmayı user-turları koruyarak LLM'e özetletir; deterministik kırpma katmanı yoktur.

## 1.3 DeepSeek-code — **basit compaction tetiği** (topluluk)

`deepseek_code/conversation.py` içinde `ConversationHistory`:
- `max_messages = 100` (yumuşak sınır); `needs_compaction()` → `len(messages) > 100` olunca **öneri** döndürür.
- Gerçek sıkıştırma minimal/öneri-seviyesi; per-tool kırpma veya LLM handoff-özet mimarisi yok.

**Ekol:** ilkel — sadece "mesaj sayısı > 100 → compaction gerekebilir" sinyali. Referans değeri düşük.
> **Tek cümle:** DeepSeek-code yalnızca 100 mesajı aşınca "compaction lazım" der; olgun bir tool-trace mekanizması yoktur.

## 1.4 Tool-trace — yerleşim

| Ajan | Ekol | Ana mekanizma | Korunan | Olgunluk |
|---|---|---|---|---|
| **Kimi Code** | **Hibrit** | per-tool kırpma (A) + eşikte LLM handoff-özet (B), event-sourced/replay | user head(2K)+tail, elision | Yüksek |
| **MiniMax** | LLM-özet | token>80K → user-koruyan özet | user turları | Orta |
| **DeepSeek-code** | İlkel | >100 mesaj → öneri | — | Düşük |

Mevcut beşliyle birlikte: **Kimi ≈ Codex+OpenCode birleşimi** (hibrit, en olgun); **MiniMax ≈ OpenClaw/Claude Code** (saf LLM-özet); **DeepSeek-code** en alt uç.

---

# BÖLÜM 2 — TASK MANAGEMENT

*(İşin KENDİSİNİ yönetmek: kuyruk, subagent, retry, çökme sonrası devam. Taksonomi: build / buy / delegate / in-process.)*

## 2.1 Kimi Code — olgun **in-process / persisted-session** (durable kuyruk değil)

- **Task modeli** (`packages/protocol/src/task.ts`): session-scoped `Task` — `kind: subagent | bash | tool`, `status: running/completed/failed/cancelled` (FSM), `output_preview` + `output_bytes` (çıktı-görünüm kısaltma), subagent için `model` + `thinking_effort`. Bu **oturum içi çalışan/arka-plan task takibi**; restart'ı atlatan durable kuyruk **değil**.
- **Subagent** (`session/subagent/runAgentTurn.ts`): bir prompt (**veya retry**) turu ayrı bir ajan scope'unda koşar ve bitince **context'ten özet damıtır** (Claude Code subagent-izolasyonu muadili).
- **Concurrency (gerçek fark):** `tool/toolContract.ts`'teki **`ToolAccesses`** — her execution hangi kaynağa eriştiğini bildirir; "host scheduler **çakışmayan çağrıları eşzamanlı** koşar (çatışma semantiğiyle)". Tool seviyesinde **kaynak-farkındalıklı paralel yürütme**.
- **Persistence:** `persistence` + event-sourced context → oturumlar kalıcı/replay-edilebilir (**resume**). Ama **cron / lease / circuit-breaker / dağıtık kuyruk yok**.

> **Tek cümle:** Kimi, task'ı durable-kuyruk yerine olgun bir in-process/persisted-session modeliyle yönetir — subagent+background-task FSM, event-sourced resume ve kaynak-farkındalıklı bir concurrency scheduler ile.

## 2.2 MiniMax Mini-Agent — **retry decorator + in-process**

- **Retry** (`mini_agent/retry.py`): zarif bir dekoratör — `max_retries = 3`, **exponential backoff** (`exponential_base`), `retryable_exceptions` ile hedeflenen istisnalar. Bu **API-çağrısı seviyesi** retry (task-seviyesi değil).
- **Model** (`agent.py`): tek-ajan döngüsü + interleaved thinking; MCP tool yükleme (`tools/mcp_loader.py`), ACP sunucusu (`acp/server.py`). Subagent/kuyruk/cron **yok**.
- **Durable?** Hayır — süreç-içi. Çökme sonrası "kaldığı yerden devam" mekanizması yok.

> **Tek cümle:** MiniMax, API çağrılarını exponential-backoff ile retry eder; ama task-seviyesi dayanıklılık (kuyruk/crash-recovery) taşımayan saf in-process bir ajandır.

## 2.3 DeepSeek-code — task-yönetimi **yok** (topluluk)

- Subagent / retry / kuyruk / scheduler yok. Basit bir tek-ajan REPL/CLI (`agent.py` + `conversation.py` + `tools/`).
- **Durable?** Hayır. Sadece etkileşimli çalışır.

> **Tek cümle:** DeepSeek-code'da task-yönetimi katmanı yoktur; interaktif tek-ajan.

## 2.4 Task-management — taksonomiye yerleştirme

Dört rota (hatırlatma): **kur** (Hermes/OpenClaw SQLite kernel) · **bin** (Shannon→Temporal) · **delege** (Codex cloud) · **in-process** (dayanıksız).

| Ajan | Rota | Subagent | Retry | Çökme sonrası devam | Durable kuyruk |
|---|---|---|---|---|---|
| **Kimi Code** | in-process (**olgun**) | ✅ (özet damıtır) | ✅ (turn retry) | ~ (persisted-session/replay) | ❌ |
| **MiniMax** | in-process | ❌ | ✅ (API-backoff) | ❌ | ❌ |
| **DeepSeek-code** | in-process (ilkel) | ❌ | ❌ | ❌ | ❌ |

Üçü de **in-process** kategorisinde. **Kimi**, bu kategorinin en olgun sürümü (event-sourced resume + concurrency scheduler + task FSM) — OpenCode/Claude Code'un ötesinde ama Hermes/Temporal'ın durable-kuyruk/crash-recovery seviyesinde değil. **MiniMax** temiz ama sade; **DeepSeek-code** en alt uç.

---

## Özet

- **Tool-trace:** Kimi **hibrit ve en olgun** (deterministik per-tool + LLM handoff-özet, event-sourced/replay); MiniMax **saf LLM-özet** (80K eşiği, user-koruyan); DeepSeek-code **ilkel** (100-mesaj sinyali).
- **Task-mgmt:** Üçü de **in-process**; Kimi kategorinin zirvesi (subagent + concurrency scheduler + persisted-session resume), MiniMax API-retry'lı sade in-process, DeepSeek-code task-yönetimsiz.
- **Statü:** Kimi ve MiniMax **resmî**; DeepSeek-code **topluluk** (DeepSeek'in resmî kod-ajanı yok).

## Kaynak dosyalar
- Kimi: `packages/agent-core-v2/src/tool/result-builder.ts`, `agent/contextMemory/compactionHandoff.ts`, `agent/fullCompaction/strategy.ts`, `protocol/src/task.ts`, `session/subagent/runAgentTurn.ts`, `tool/toolContract.ts`.
- MiniMax: `mini_agent/agent.py` (`_summarize_messages`, `token_limit`), `mini_agent/retry.py`.
- DeepSeek-code: `deepseek_code/conversation.py` (`ConversationHistory`, `needs_compaction`).
