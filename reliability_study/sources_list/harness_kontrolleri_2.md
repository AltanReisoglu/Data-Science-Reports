# Harness Kontrolleri — 2. Tur: Yeni Harness'lar ve Derin Kazı

Tarama tarihi: 2026-08-24 · Yöntem: **yalnızca birincil kaynak** — `raw.githubusercontent.com`
üzerinden gerçek kaynak kodu, `gh api` ile GitHub Code Search, grep.app kod araması ve resmî
dokümantasyon. Blog/ikincil özet kullanılmadı; kullanıldıysa etiketi `[?]`.

Kardeş dosyalar: `harness_kontrolleri.md` (1. tur, 28 satırlık ana envanter),
`loop_budge.md` (akademik kaynaklar), `tespit_sinirlari.md`.

**Doğrulama etiketleri:**
`[K]` kaynak kodu bizzat okundu, sayılar dosyadan alındı ·
`[D]` resmî dokümantasyondan alındı, kod görülmedi ·
`[?]` doğrulanamadı / emin değilim.

**Bu turun 1. tura göre eklediği:**
- **Bölüm A** — 1. turda hiç bakılmamış harness'lar (OpenClaw, Hermes, Devin/Cognition,
  Qwen-Agent, AgentScope, CAMEL, MetaGPT, ChatDev, Kimi, OpenHands varyantları).
- **Bölüm B** — 1. turda yalnızca "temel parametre" düzeyinde geçen framework'lerde
  (LangGraph, CrewAI, Google ADK, AutoGen, LlamaIndex, Haystack, Semantic Kernel,
  Mastra, VoltAgent, Strands) bu konuya özel **ikincil mekanizma** kazısı.

---

## Bölüm A — Yeni harness'lar

### A.0 Karşılaştırma tablosu

Kısaltmalar 1. turdakiyle aynı: **Sert** = istisna/terminal durum · **Nudge** = prompt'a
uyarı, koşum devam · **Zarif** = modelden nihai cevap istenip bitirilir · **Sor** =
kullanıcıya sorulur · **Sessiz** = hiçbir sinyal üretmeden biter.

| Harness | Loop tespiti | Heuristik | Adım/tur limiti (varsayılan) | Token | Maliyet | Süre | Limit dolunca | Et. |
|---|---|---|---|---|---|---|---|---|
| **OpenClaw** | ✅ **6 dedektör** (envanterin en zengini) — *varsayılan kapalı* | `unknown_tool_repeat`(10), `global_circuit_breaker`(30), `known_poll_no_progress`(20/10), `ping_pong`(20/10), `generic_repeat`(20/10), `argument_churn`(10); pencere 30; **sonuç hash'i** de karşılaştırılıyor | **Ana döngüde yok**; code-mode 5, cron 5, A2A ping-pong 5 (tavan 20) | — | — | — | warning→**nudge**, critical→araç **bloke**, 2. critical→**sert** | `[K]` |
| **Hermes** (`NousResearch/hermes-agent`) | ✅ 3 araç dedektörü + içerik tekrarı — **uyarılar varsayılan AÇIK**, sert kesme opt-in | `repeated_exact_failure`(2/5), `same_tool_failure`(3/8), `idempotent_no_progress`(2/5); + `repetition_guard` (60 char pencere, %50 baskınlık) | CLI `agent.max_turns` **500**; kütüphane API'si `sys.maxsize`=**sınırsız**; subagent `delegation.max_iterations` **45** | — | **Yok** | `run_budget_seconds` **None**=kapalı; aktifse %80'de nudge | Bütçe: **zarif** (1 mesaj + 1 grace call + özet zorlaması). Guardrail: `allow\|warn\|block\|halt` | `[K]` |
| **Kimi CLI** (`MoonshotAI/kimi-cli`) | ❌ | — | `max_steps_per_turn=1000`, `max_retries_per_step=3`, `max_ralph_iterations=0` | — | — | — | doğrulanamadı `[?]` | `[K]` |
| **Devin / Cognition** | ❌ (koşum zamanı kapalı kaynak) | — | **Yok** | — | **`max_acu_limit`** (ACU = satıcı tanımlı bileşik birim), varsayılan `null`=**sınırsız** | — | doğrulanamadı | `[D]` |
| **Qwen-Agent** | ❌ | — | `MAX_LLM_CALL_PER_RUN=20` (env ile ezilebilir) | `DEFAULT_MAX_INPUT_TOKENS=58000` (limit değil, **budama**) | — | — | **Sessiz** — `while` biter, istisna/mesaj/`stop_reason` **yok** | `[K]` |
| **AgentScope** | ❌ | — | `ReActConfig.max_iters=50` + `structured_output_grace_iters=5` | — | — | — | **Zarif (zorlamalı)** — `tool_choice` kilitleniyor, `ReplyFinishedReason.EXCEED_MAX_ITERS` | `[K]` |
| **CAMEL** | ⚠️ kullanıcı takar | `ResponseWordsTerminator(words_dict)` — kelime tekrarı sayacı; kutudan **boş** | `ChatAgent.max_iteration=None` = **sınırsız** | — | — | — | `_step_terminate` + sebep dizesi | `[K]` |
| **MetaGPT** | ❌ | — | `Team.run(n_round=3)`; `Role.max_react_loop=1` | — | **`max_budget=10.0` $ / `investment=10.0`** — varsayılanı dolu | — | **Sert** — `NoMoneyException`; kontrol **tur başında**, aşım mümkün | `[K]` |
| **ChatDev** (yeniden yazılmış) | ❌ | — | `LoopCounterConfig.max_iterations=10` (**graf düğümü**) | — | — | `LoopTimerConfig.max_duration=60.0 s` (**graf düğümü**) | **Yönlendirme** — düğüm `message` çıktısı salıp graf devam eder | `[K]` |
| **OpenHands** varyantları | ✅ (1. turdaki `StuckDetector` geçerli) | microagents = **prompt katmanı**, döngü kontrolü değil | 1. turdaki gibi | — | — | — | 1. turdaki gibi | `[K]`/`[?]` |

### A.1 OpenClaw (`openclaw/openclaw`) `[K]`

**Bu iki turun tamamındaki en zengin loop detection implementasyonu.** Gemini CLI 3 dedektör
kullanıyordu; OpenClaw **6 dedektör + iki kademe (warning/critical) + global devre kesici**
kuruyor ve — bütün envanterde tek — **araç sonucunu da hash'liyor**, yalnızca çağrıyı değil.

Kaynak: yerel checkout `/home/altan/Desktop/adapted/harnesses/openclaw`, sürüm `2026.7.2`
(`package.json`); npm'deki `latest` `2026.7.1-2`. Kanonik depo **`openclaw/openclaw`**
(`steipete/openclaw` ve `vercel-labs/openclaw` fork/ayna).

#### Dedektörler — `src/agents/tool-loop-detection.ts` (786 satır)

Sabitler (satır 49–63) ve `src/agents/tool-loop-thresholds.ts`:

```ts
const TOOL_CALL_HISTORY_SIZE = 30;              // kayan pencere
export const UNKNOWN_TOOL_THRESHOLD = 10;
const CRITICAL_THRESHOLD = 20;
const GLOBAL_CIRCUIT_BREAKER_THRESHOLD = 30;
export const TOOL_LOOP_WARNING_THRESHOLD = 10;  // tool-loop-thresholds.ts:1
const DEFAULT_LOOP_DETECTION_CONFIG = { enabled: false, ... };
```

`LoopDetectorKind` (satır 28–34) altı değer alıyor; `detectToolCallLoop()` (satır 510–678)
bunları **sabit bir öncelik sırasıyla** deniyor:

| # | Dedektör | Ne arıyor | Eşik | Seviye |
|---|---|---|---|---|
| 1 | `unknown_tool_repeat` | var olmayan bir aracı tekrar tekrar çağırma | 10 | critical |
| 2 | `global_circuit_breaker` | aynı araç + **aynı sonuç hash'i** serisi | 30 | critical |
| 3 | `known_poll_no_progress` | "polling" tipi araçta ilerlemesiz tekrar | 20 / 10 | critical / warning |
| 4 | `ping_pong` | A-B-A-B dönüşümlü çağrı kuyruğu | 20 (+ilerleme yok kanıtı) / 10 | critical / warning |
| 5 | `generic_repeat` | aynı araç + aynı argüman hash'i | 20 (sonuç bazlı) / 10 (çağrı bazlı) | critical / warning |
| 6 | `argument_churn` | argümanları **değiştirerek** aynı sonucu almak | 10 | warning |

**Ayırt edici tasarım kararı — "no progress" sonuç hash'iyle tanımlanıyor.**
1. turdaki 6 dedektörün hepsi (Cline, Roo, opencode, Gemini CLI, OpenHands, deer-flow)
çağrı imzasına bakıyordu. OpenClaw `src/agents/tool-loop-no-progress.ts`'te
`countNoProgressStreak()` ile **hem `argsHash` hem `resultHash`** zincirini takip ediyor:
sonuç değiştiği anda seri kırılıyor (satır 58–60: `if (record.resultHash !== latestResultHash) break;`).
Yani "aynı komutu 20 kez çalıştırdım ama çıktı her seferinde farklıydı" **döngü sayılmıyor**.
Kritik `generic_repeat` yalnızca sonuç bazlı; warning ise çağrı bazlı — bu, yanlış pozitifi
ucuz kademede tutup pahalı kademeyi kanıta bağlayan bir tasarım.

**`argument_churn` envanterde eşi olmayan bir fikir** (`src/agents/tool-loop-argument-churn.ts`).
Model "aynı çağrıyı tekrarlama" uyarısını alınca argümanı azıcık değiştirerek uyarıdan kaçar;
bu dedektör tam olarak o kaçışı yakalıyor. Ayrıca `livenessSignal: "argument_churn"` olarak
diğer dedektörlerin sonucuna **iliştiriliyor** (satır 529–533) — yani "model hâlâ canlı ama
verimsiz" ile "model tamamen kilitlenmiş" ayırt ediliyor.

`exec` aracına özel bir istisna var (`tool-loop-no-progress.ts:11–14`): gerçek terminal
hataları farklı argümanlarla da tekrarlayabildiği için `terminal-exec-failure` tipli
bitişik kuyruk ayrıca sayılıyor ve iki sayımdan büyüğü alınıyor.

#### Limit dolunca ne oluyor — üç kademe

`src/agents/tool-loop-admission.ts:44–83`:

- **warning** → araç **çalıştırılır**, modele uyarı metni verilir
  (`shouldEmitLoopWarning` ile tekrarlı uyarı bastırılıyor). Klasik nudge.
- **critical** → araç çağrısı **bloke edilir** (`action: "block"`), geriye
  `ToolLoopIntervention { kind: "critical-tool-loop" }` döner. Aynı batch'teki diğer
  çağrılar da çalıştırılmaz: *"This tool was not executed because another call in the batch
  triggered critical tool-loop recovery"* (`agent-loop.ts:1432`).
- **ikinci critical** → koşum **biter**. `packages/agent-core/src/agent-loop.ts:289–290`
  `toolLoopRecoveryState.criticalToolLoopSeen` bayrağını tutuyor; ikinci kritik döngüde
  `terminal: criticalToolLoopSeen` (satır 662) devreye girip
  `TOOL_LOOP_RECOVERY_TERMINATED_MESSAGE` (satır 62–63) ile duruyor:
  *"OpenClaw stopped this run because tool-loop recovery encountered another critical loop."*

Bu, Gemini CLI'ın `_recoverFromLoop()` → 2. tespitte dur şemasının **araç bazlı** karşılığı:
kurtarma turu bir kez veriliyor, tekrarlarsa sert kesme.

#### Varsayılan: **kapalı**

`src/config/types.tools.ts:161–164`:

```ts
export type ToolLoopDetectionConfig = {
  /** Enable tool-loop protection (default: false). */
  enabled?: boolean;
};
```

`tool-loop-detection.ts:53` `enabled: false`, `detectToolCallLoop` satır 518–520 ilk iş
olarak `if (!resolvedConfig.enabled) return { stuck: false }` diyor. Yani **envanterin en
gelişmiş dedektörü kutudan kapalı geliyor** — 1. turdaki Çıkarım 6'nın (Cline core, Codex
`default_enabled:false`) en güçlü örneği.

**Ayrıca tüm sayısal ayar imkânı kaldırılmış.** `tool-loop-thresholds.ts:3–6` yorumu:
*"Numeric loop tuning was retired in #111382. Keep every admission path on the same built-in
threshold so policy rewrites cannot drift from detection."* `windowSize`, `historySize`,
`warningThreshold`, `criticalThreshold`, `detectors`, `pingPong`, `genericRepeat`,
`globalCircuitBreakerThreshold` config anahtarları
`src/commands/doctor/shared/legacy-config-migrations.runtime.retired-media.ts:150–160`'ta
**emekli anahtar listesine** taşınmış. Config yüzeyinde geriye tek bir `enabled` boolean'ı
kalmış. Bu, PoC için doğrudan alınabilecek bir ders: *ayarlanabilir eşik = tespit ile
uygulamanın birbirinden kayması*.

#### Bütçe: ana döngüde **yok**, kenarlarda var

Ana `agentLoop` (`packages/agent-core/src/agent-loop.ts`, 1614 satır) içinde **tur sayacı,
adım limiti, token bütçesi veya dolar bütçesi yok**. `maxTurns|maxSteps|maxIterations|
costLimit|budgetUsd|maxCost` desenleri `packages/` ve `src/` genelinde ana koşum yolunda
sıfır sonuç veriyor. Döngüyü durduran tek şey: model araç istemeyi bırakması, abort sinyali,
ya da yukarıdaki kritik loop müdahalesi. **Yani Continue'nun sayaçsız `while(true)`'suyla
aynı sınıfta — ama üstünde ciddi bir dedektör var.**

Bütçe yalnızca **üç dar bağlamda**, hepsi de küçük sabitler:

| Bağlam | Sabit | Değer | Dosya |
|---|---|---|---|
| Code-mode headless | `DEFAULT_HEADLESS_TOOL_CALLS` | **5** araç çağrısı | `src/agents/code-mode-runtime.ts:36` |
| Cron tetikleyici | `HEADLESS_TRIGGER_TOOL_BUDGET` | **5** araç çağrısı | `src/cron/trigger-script.ts:66` |
| Agent-to-agent ping-pong | `DEFAULT_AGENTNG_PONG_TURNS` / `MAX_PING_PONG_TURNS` | **5**, sert tavan **20** | `src/agents/tools/sessions-send-helpers.ts:20–21` |

Aşıldığında code-mode sert hata veriyor:
`code mode headless tool budget exceeded (${maxToolCalls})` (`code-mode-headless.ts:286`).

Bu, 1. turdaki **Çıkarım 7**'nin (aynı harness bağlama göre farklı varsayılan) yeni ve en
uç örneği: ana etkileşimli döngüde **sınırsız**, cron/headless bağlamında **5**. Oran
sonsuz. Gerekçe Goose'unkiyle aynı sınıftan: gözetimsiz koşumda tavan sert, gözetimli
koşumda kullanıcı zaten müdahale edebiliyor.

#### Modele bütçesini söyleme

1. turda yalnızca Goose ve Codex'te bulunan desen OpenClaw'da da var — A2A ping-pong'da
her turda modele prompt olarak veriliyor
(`src/agents/tools/sessions-send-helpers.ts:125`):

```ts
`Turn ${params.turn} of ${params.maxTurns}.`,
"If you want to stop the ping-pong, reply exactly \"${REPLY_SKIP_TOKEN}\".",
```

Ek olarak modele **erken çıkış jetonu** veriliyor: sınır dolmadan da kendi isteğiyle
konuşmayı bitirebiliyor. Goose ve Codex yalnızca kalan bütçeyi bildiriyordu; **erken çıkış
protokolü sunan tek örnek bu.**



---

### A.2 Hermes (`NousResearch/hermes-agent`) `[K]`

#### Önce isim belirsizliği — brief'te sorulan soru

Brief "Hermes birden fazla şeye ait olabilir, harness olanı bul" diyordu. Cevap: **ikisi de
gerçek ve ikisi de Nous Research'e ait.**

- **Hermes model ailesi** (Hermes 2/3/4, `NousResearch/Hermes-*` HF depoları) — bir LLM
  ailesi, harness değil. Bu envantere girmez.
- **`NousResearch/hermes-agent`** — *"The agent that grows with you"*, Python, `main` dalı,
  235.000+ yıldız. **Tam teşekküllü bir agent harness'ı.** Bu bölüm bunu inceliyor.

Ekosistemde ayrıca çok sayıda üçüncü taraf eklentisi var (`hermes-webui`, `hermes-lcm`,
`oh-my-hermes`, `hermes-agent-self-evolution`); bunlar harness değil, eklenti. `hermes-go`,
`schnetzlerjoe/hermes` gibi adaş ama alâkasız projeler de mevcut — karıştırmamak gerekiyor.

#### İki bağımsız guardrail ailesi

Hermes bu konuda envanterdeki en **modüler** tasarıma sahip: döngü tespiti ve bütçe
birbirinden tamamen ayrı dosyalarda, farklı varsayılanlarla.

**(a) `agent/tool_guardrails.py` (855 satır) — araç çağrısı döngü guardrail'i**

`ToolCallGuardrailConfig` (satır 110–128), config.yaml'daki `tool_loop_guardrails`
bölümünden besleniyor:

```python
warnings_enabled: bool = True          # ← UYARILAR VARSAYILAN AÇIK
hard_stop_enabled: bool = False        # ← sert kesme opt-in
exact_failure_warn_after: int = 2
exact_failure_block_after: int = 5
same_tool_failure_warn_after: int = 3
same_tool_failure_halt_after: int = 8
no_progress_warn_after: int = 2
no_progress_block_after: int = 5
```

**Bu, iki turda loop detection'ı varsayılan olarak AÇIK gelen ilk harness.** 1. turdaki
Çıkarım 6 ("mekanizma var ama kapalı") burada kısmen kırılıyor: uyarı kademesi açık, yalnızca
sert kesme kapalı. Kod yorumu gerekçeyi veriyor (satır 112–115): *"Warnings are enabled by
default and never prevent tool execution. Hard stops are explicit opt-in so interactive
CLI/TUI sessions get a gentle nudge unless the user enables circuit-breaker behavior."*

Üç dedektör:

| Kod | Ne arıyor | Uyarı | Sert |
|---|---|---|---|
| `repeated_exact_failure` | aynı araç + aynı kanonik argüman, **başarısız** | 2 | 5 |
| `same_tool_failure` | aynı araç, farklı argüman, hep başarısız | 3 | 8 |
| `idempotent_no_progress` | **salt-okunur** araç, aynı sonuç hash'i | 2 | 5 |

Özgün nokta: **araçlar idempotent / mutating diye önceden sınıflandırılmış**
(`IDEMPOTENT_TOOL_NAMES` satır 20–37, `MUTATING_TOOL_NAMES` satır 40–57). `read_file`,
`search_files`, `web_search`… salt-okunur kabul ediliyor; bunlarda "aynı sonuç" kesinlikle
ilerleme yokluğu demek. `terminal`, `write_file`, `patch`… mutating; onlarda aynı çağrının
tekrarı meşru olabiliyor. 1. turdaki hiçbir harness bu ayrımı yapmıyordu — hepsi tüm
araçlara aynı heuristiği uyguluyordu.

Buna ek olarak bir **poller muafiyet listesi** var (satır 61–79): `process`,
`bfl_flux3_get_result` ve `_get_result` / `_poll` **son ekiyle** biten her araç
"stall guard"dan muaf. OpenClaw'ın `isKnownPollToolCall`'una denk düşen fikir, ama son ek
deseniyle MCP/üretilmiş araç yüzeylerine genelleştirilmiş.

**`STALL_GUARD_IDENTICAL_CALL_THRESHOLD = 3`** (satır 88) — aynı araç, aynı argüman, aynı
sonuç üçüncü kez geldiğinde modele "döngü kırıcı" notu ekleniyor. Ayrıca
`IDENTICAL_RESULT_STUB_MIN_CHARS = 512` (satır 95): **ikinci kez** aynı sonuç geldiğinde,
512 karakterden büyükse yükün kendisi bağlamda kısa bir referans stub'ıyla değiştiriliyor.
Yani döngü hem tespit ediliyor hem de tespit edilmeden önce **maliyeti düşürülüyor**.
Hata sonuçları asla stub'lanmıyor (*"the model must see every fresh error verbatim"*).

**(b) `LoopCapConfig` (satır 186–219) — tur başına kaba tavan**

```python
_DEFAULT_MAX_WEB_SEARCHES_PER_TURN = 50
_DEFAULT_MAX_SUBAGENTS_PER_TURN = 50
```

Docstring, kaynağını açıkça yazıyor: *"Inspired by Claude Code v2.1.212 (Week 29, July 2026),
which added caps on WebSearch calls and subagent spawns to stop runaway search / delegation
loops."* Bu tavanlar `hard_stop_enabled`'dan **bağımsız** çalışıyor (`before_call`,
satır 379–387) — yani kapalı konfigürasyonda bile 51. web aramasını / 51. subagent
spawn'ını bloke ediyor. Sayaçlar her turda sıfırlanıyor (`reset_for_turn`).

**Bu, "belirli araca özel tavan" fikrinin envanterdeki tek örneği.** Diğer herkes genel adım
sayacı koyuyor; Hermes runaway'e en yatkın iki aracı ayrıca sınırlıyor.

**(c) `agent/repetition_guard.py` — içerik tekrarı (chanting) dedektörü**

Gemini CLI'ın `checkContentLoop`'unun karşılığı ama farklı bir yaralanmadan doğmuş.
Modül docstring'i olayı isim isim veriyor (issue #86581): *"a single turn produced a
60,698-char response delivered as 31 Discord messages."* `finish_reason=length` ile kesilen
yanıtın devam ettirilmesi yolunda çalışıyor:

```python
MIN_FRAGMENT_LENGTH = 400      # bunun altında hiç bakma
_REPEAT_WINDOW = 60            # 60 karakterlik birebir tekrar penceresi
_MIN_REPEAT_COUNT = 5
_DOMINANCE_RATIO = 0.5         # tekrarlar parçanın yarısını kaplıyorsa
```

Gemini CLI 50 karakterlik chunk + 10 tekrar kullanıyordu; Hermes 60 karakterlik pencere +
**baskınlık oranı** kullanıyor — yani mutlak sayı değil, parçanın ne kadarını kapladığı.
İki hızlı yol var: önce satır bazlı (`_line_repetition_dominated`), sonra karakter kayan
penceresi. Tasarım açıkça **fail-open**: *"Returns False for non-string / empty / short
inputs (never blocks a continuation the guard cannot confidently judge)."*

#### Bütçe — `agent/iteration_budget.py`

```python
class IterationBudget:          # thread-safe sayaç
    def consume(self) -> bool   # kota dolduysa False
    def refund(self) -> None    # execute_code turları geri veriliyor
```

**`refund()` envanterde eşi olmayan bir mekanizma.** `execute_code` (programatik araç
çağırma) turları bütçeden düşülmüyor — çünkü bir `execute_code` turu içinde 10 araç
çalıştırmak, 10 ayrı LLM turundan ucuz. Bütçe böylece "LLM turu" cinsinden ölçülüyor,
"iş" cinsinden değil. 1. turdaki Çıkarım 4'e (adım en kötü vekil) verilen ilk somut cevap
bu: **bazı adımları saymamak.**

**Varsayılan değerler tutarsız — üç ayrı sayı dolaşıyor** `[K]`:

| Yer | Değer | Dosya |
|---|---|---|
| Kütüphane API'si `init_agent(max_iterations=…)` | **`sys.maxsize` = sınırsız** | `agent/agent_init.py:523` |
| Aynı fonksiyonun docstring'i | *"default: 90"* | `agent/agent_init.py:603` |
| CLI varsayılan config sözlüğü `agent.max_turns` | **500** | `cli.py:477` |
| `IterationBudget` docstring'i | *"default 500"* / subagent *"default 50"* | `agent/iteration_budget.py:5–7` |
| CLI subagent varsayılanı `delegation.max_iterations` | **45** | `cli.py:540` |

Yani docstring'lerdeki 90 ve 50 sayıları kodda karşılıksız. (1. turdaki DSPy `max_iters=20`
vs docstring "10" tutarsızlığının aynısı — **doküman/kod ayrışması bu alanda sistematik.**)

`hermes_cli/config.py`'de "sınırsız" ayrı bir tip olarak değil, **sentinel sayı** olarak
kurulmuş — envanterdeki en dürüst mühendislik yorumlarından biri (satır 3085–3092):

```python
TURN_LIMIT_UNLIMITED = sys.maxsize
```

Gerekçe: her `<`, `>=`, `remaining = max - used` karşılaştırması özel bir "unlimited"
değerini öğrenmek zorunda kalmasın; *"is large enough that no real conversation will ever
reach it (a turn takes seconds; 9.2e18 turns would take ~10^11 years)."* `resolve_turn_limit()`
`none` / `null` / `unlimited` / `infinite` / `inf` / `∞` / `-1` / `0` yazımlarının hepsini
sınırsıza çeviriyor.

**Hiyerarşik bütçe — üçüncü model.** 1. turda iki model vardı: Codex'in *paylaşımlı tek
sayacı* ve SWE-agent'ın *devredilen bütçesi*. Hermes üçüncüsünü kuruyor:
**bağımsız alt bütçeler**, ve bunu bir uyarıyla birlikte belgeliyor
(`tools/delegate_tool.py:1764–1767`):

> *"Each subagent gets its own iteration budget capped at max_iterations (configurable via
> delegation.max_iterations, default 50). This means total iterations across parent +
> subagents can **exceed** the parent's max_iterations."*

Yani **toplam maliyetin üst sınırı yok** — parent 500 + her subagent 45, subagent sayısı
`max_subagents=50` tavanına kadar. Kaba üst sınır: 500 + 50×45 = 2.750 iterasyon. Kodun
kendisi bu sızıntıyı itiraf ediyor; PoC'de "hiyerarşik bütçe" tartışılırken bu
**karşı-örnek** olarak kullanılabilir.

#### Limit dolunca ne oluyor — ve nudge'a karşı kanıt

`agent/agent_init.py:986–991`, envanterdeki en değerli tek yorum bloğu olabilir:

```python
# Iteration budget: the LLM is only notified when it actually exhausts
# the iteration budget (api_call_count >= max_iterations).  At that
# point we inject ONE message, allow one final API call, and if the
# model doesn't produce a text response, force a user-message asking
# it to summarise.  No intermediate pressure warnings — they caused
# models to "give up" prematurely on complex tasks (#7915).
```

**Bu, Goose ve Codex'in "modele kalan bütçeyi söyle" desenine doğrudan ampirik itiraz.**
1. turdaki Çıkarım 9 bu deseni "ucuz ve ayırt edici bir öneri" diye sunuyordu; Hermes aynı
şeyi denemiş ve geri almış — ara uyarılar modellerin karmaşık görevlerde **erken pes
etmesine** yol açmış (#7915). PoC'de bütçe farkındalığı önerilecekse bu risk mutlaka
yanına yazılmalı.

Limit dolunca davranış **zarif bozulma**: bir mesaj enjekte, bir "grace call"
(`agent._budget_grace_call`), model metin üretmezse özet isteyen bir user mesajı zorla.
smolagents'ın `_handle_max_steps_reached()`'iyle aynı sınıfta ama iki kademeli.

Araç guardrail'inde ise dört ayrı eylem var (`ToolGuardrailDecision.action`, satır 258):
`allow | warn | block | halt`. `block` yalnızca o çağrıyı sentetik bir tool result'la
(`toolguard_synthetic_result`, satır 728) reddediyor, koşum devam ediyor; `halt` turu
bitiriyor.

#### Süre bütçesi

`agent.run_budget_seconds` — `run_conversation` turu başına duvar saati bütçesi.
**Varsayılan `None` = özellik tamamen kapalı** (`_normalize_run_budget_seconds`,
`agent_init.py:493–509`; malformed değer asla makineyi çalıştıramıyor, yalnızca uykuda
bırakıyor). Aktifken **%80'de bir kez "wrap-up" notu** enjekte ediliyor
(`agent._run_budget_wrapup_injected`, satır 1003–1004).

İlginç çelişki: Hermes iterasyon bütçesinde ara uyarıyı #7915 yüzünden kaldırmış, ama
**süre bütçesinde %80 uyarısını korumuş.** İkisi arasındaki farkın gerekçesi kodda yazılı
değil `[?]`.

Ayrıca `agent/deadline.py` ayrı bir "birleşik deadline katmanı" — depoda **en az altı ayrı
yerel timeout mekanizması** olduğunu ve her stall raporunun listeyi bir uzattığını itiraf
edip (issue #85125) bunları tek primitife taşıyor. `run_bounded_async` özellikle
`asyncio.wait_for`'un event loop bloke olduğunda **sessizce devre dışı kalması** sorununu
daemon `threading.Timer` ile çözüyor — timeout tasarımı hakkında doğrudan alıntılanabilir
bir gözlem.

#### Maliyet bütçesi: **yok**

`cost_limit`, `max_cost`, `spend_limit`, `budget_usd` desenleri depoda ajan koşum yolunda
sonuç vermiyor (`hermes_cli/session_filters.py`'deki `max_cost` geçmiş oturumları
**listelerken** kullanılan bir filtre, zorlama değil). `agent/credits_tracker.py`,
`agent/usage_pricing.py`, `agent/account_usage.py` maliyeti **izliyor**, eşikle
karşılaştırmıyor `[K]`.

---

### A.3 Kimi CLI (`MoonshotAI/kimi-cli`) `[K]`

Brief "Kimi / Moonshot Kimi-Dev" diyordu. **Kimi-Dev bir model** (SWE-bench için eğitilmiş,
kendi harness'ı yok — SWE-agent/OpenHands üzerinde koşuluyor). Moonshot'ın gerçek harness'ı
**`MoonshotAI/kimi-cli`** (11.257 yıldız, Python). Ekosistemde ayrıca `kimi-code`,
`kimi-agent-sdk`, `kimi-agent-rs` (Rust sunucu) ve `kosong` (LLM soyutlama katmanı) var.

Tek ve net bir yerde toplanmış: `src/kimi_cli/config.py:75–87`.

```python
class LoopControl(BaseModel):
    """Agent loop control configuration."""
    max_steps_per_turn: int = Field(default=1000, ge=1, ...)
    max_retries_per_step: int = Field(default=3, ge=1)
    max_ralph_iterations: int = Field(default=0, ge=-1)
    """Extra iterations after the first turn in Ralph mode. Use -1 for unlimited."""
```

Üç gözlem:

1. **`LoopControl` diye ayrı, adı konmuş bir config sınıfı olması nadir.** Diğer
   harness'larda bu parametreler config'in içine dağılmış durumda; burada tek yerde.
2. `max_steps_per_turn` **1000** — Goose'un `GOOSE_MAX_TURNS=1000` değeriyle aynı; pratikte
   "sınırsıza yakın" bir tavan.
3. `max_ralph_iterations` ("Ralph mode" — görevi tekrar tekrar yeniden koşma modu)
   varsayılan **0**, `-1` sınırsız. Yani en riskli mod kutudan kapalı.

**Loop detection: yok** `[K]`. `src/` altında `loop`, `repeat`, `stuck` aramaları yalnızca
sinyal işleme, telemetri ve prompt metinleri döndürüyor; tekrar/stuck heuristiği bulunmadı.
**Maliyet/token bütçesi: yok.**

---

### A.4 Devin / Cognition (`CognitionAI`) `[D]`

**Ajan koşum zamanı kamuya açık değil.** `CognitionAI` organizasyonunda yalnızca entegrasyon
ve dağıtım araçları var: `devin-cli`, `terraform-provider-devin`, `devin-outpost-k8s`
(Kubernetes operator referans implementasyonu), `actions`, `plugin-template`,
`team-marketplace-template`. Hiçbirinde agent döngüsü yok. Kamuya açık tek metodoloji
belgesi `CognitionAI/devin-swebench-results` (SWE-bench sonuçları) — orada da harness
detayı yok. `cognition-ai` diye bir org mevcut değil, `Cognition-Labs` ise **başka bir
şirket** (biyoinformatik) — isim çakışması.

Kod düzeyinde inceleme yapılamadı; geriye **resmî API dokümanı** kalıyor.

Session oluşturma uç noktası (`POST /v1/sessions`, docs.devin.ai) `[D]` şu isteğe bağlı
parametreleri alıyor: `idempotent`, `knowledge_ids`, **`max_acu_limit`**, `playbook_id`,
`secret_ids`, `session_secrets`, `snapshot_id`, `structured_output_schema`, `tags`,
`title`, `unlisted`.

**Kaynak tüketimini sınırlayan tek parametre `max_acu_limit`** ("Maximum ACU limit, must be
positive", pozitif tamsayı ya da `null`).

Bu, iki turun tamamındaki **beşinci bütçe türü**: adım / token / dolar / süre değil,
**satıcı tanımlı bileşik hesaplama birimi (ACU — Agent Compute Unit)**. ACU'nun neyi
kapsadığı (LLM tokenı + VM süresi + araç çağrıları) dokümanda kesin formülle verilmiyor,
yani müşteri tarafından **kodla doğrulanamayan** bir bütçe. Karşılaştırma tablosuna
"kapalı kutu" olarak girer.

Varsayılan `null` = **sınırsız**. Tur ya da süre sınırı yok. Yani 1. turdaki Çıkarım 6
("mekanizma var ama varsayılan kapalı") ticari üründe de aynen geçerli.

---

### A.5 Qwen-Agent (`QwenLM/Qwen-Agent`) `[K]`

Tek bir global sabit: `qwen_agent/settings.py:24`

```python
MAX_LLM_CALL_PER_RUN: int = int(os.getenv('QWEN_AGENT_MAX_LLM_CALL_PER_RUN', 20))
```

`FnCallAgent`, `ReActChat`, `TIRAgent`, `VirtualMemoryAgent` — hepsi bunu kullanıyor.

**En dikkat çekici bulgu limitin nasıl uygulandığı** (`qwen_agent/agents/fncall_agent.py:75–78`):

```python
num_llm_calls_available = MAX_LLM_CALL_PER_RUN
while True and num_llm_calls_available > 0:
    num_llm_calls_available -= 1
```

Kota bitince `while` **sessizce sonlanıyor.** İstisna yok, modele mesaj yok, `stop_reason`
yok, log yok, çağırana hiçbir sinyal yok. Üretici (`Iterator`) sonlanıyor ve dışarıdan
bakan taraf bunu **normal tamamlanmayla ayırt edemiyor.**

Bu, 1. turdaki **Çıkarım 10**'un (*"bütçe tükenmesinin sessizce başarı gibi görünmesi"* —
Agno'nun *"exhaustion is invisible to a status check"* yorumu) en saf örneği. Agno hiç
değilse bir bayrak koyuyordu; Qwen-Agent hiçbir şey koymuyor.

`DEFAULT_MAX_INPUT_TOKENS = 58000` (satır 20) bir bütçe değil, **budama** eşiği — limit
aşılınca girdi mesajları kısaltılıyor (CrewAI'ın `respect_context_window`'uyla aynı sınıf).

**Loop detection: yok. Maliyet/süre bütçesi: yok.**

---

### A.6 AgentScope (`agentscope-ai/agentscope`) `[K]`

⚠️ **Depo taşınmış:** brief'teki `modelscope/agentscope` → **`agentscope-ai/agentscope`**
(29.440 yıldız). Bu, iki turda tespit edilen **beşinci** depo taşınması.

`src/agentscope/agent/_config.py:297–318`:

```python
class ReActConfig(BaseModel):
    max_iters: int = Field(default=50, ...)
    structured_output_grace_iters: int = Field(default=5, gt=0, ...)
    """The extra iterations allowed beyond ``max_iters`` to generate the
    required structured output."""
    stop_on_reject: bool = Field(default=False, ...)
```

**İki özgün mekanizma:**

**(1) `structured_output_grace_iters = 5` — "lütuf bütçesi".** Limit dolduğunda ajan hemen
durmuyor; yapılandırılmış çıktı üretebilmesi için **5 ek iterasyon** veriliyor
(`_agent.py:3337–3341`). Üstelik kademeli: `cur_iter >= max_iters` olduğunda
`tool_choice` **zorlanıyor** — model artık başka araç seçemiyor, yalnızca
`_GenerateStructuredOutput` aracını çağırabiliyor (satır 3368–3380), ve prompt'a
*"You have reached the maximum reasoning-acting iterations, so call this tool at once"*
ekleniyor. Ancak `max_iters + grace_iters` da dolarsa sert çıkış.

Bu, 1. turdaki "zarif bozulma" stratejisinin **en gelişmiş hâli**: smolagents ve DSPy
modelden nihai cevabı *rica ediyordu*; AgentScope araç seçimini kısıtlayarak **zorluyor**,
ve bunun için ayrı bir bütçe ayırıyor.

**(2) Makine-okunur sonlanma sebebi.** `ReplyFinishedReason.EXCEED_MAX_ITERS`
(`_agent.py:3357`, `3418+`) ile `ReplyFinishedReason.COMPLETED` ayrı enum değerleri, ve
ayrıca bir `ExceedMaxItersEvent` olayı yayınlanıyor. Çıkarım 10'un istediği şey tam olarak
bu — Qwen-Agent'ın sessiz çıkışının tam zıddı, aynı kültürel havzadan iki depo.

**Loop detection: yok** `[K]` — `max_iters` / `max_iterations` dışında tekrar heuristiği
aranıp bulunamadı. **Maliyet/süre bütçesi: yok.**

---

### A.7 CAMEL (`camel-ai/camel`) `[K]`

Varsayılan dal `master`. `camel/agents/chat_agent.py`:

```python
max_iteration: Optional[int] = None      # satır 512, 626
```

**Varsayılan `None` = sınırsız** (`chat_agent.py:3112–3113`,
`3138–3142`: `if self.max_iteration is not None and iteration_count >= self.max_iteration`).
1. turdaki "kutudan sınırsız" listesine (ADK, Gemini CLI, opencode, Agno, Roo, Claude
Agent SDK) yeni bir isim. Limit dolunca `_step_terminate` çağrılıyor —
`f"Max iteration {self.max_iteration} reached without ..."` sebebiyle, yani sonlanma
sebebi **kayıtlı**.

**Özgün mekanizma — `ResponseTerminator` eklenti API'si.** `camel/terminators/`:

- `ResponseTerminator` (soyut taban, `base.py`)
- `ResponseWordsTerminator(words_dict, case_sensitive=False)`
  (`response_terminator.py:23–63`) — *"Terminate agent when some words reached to
  occurrence limit by any message of the response."*

`ChatAgent(response_terminators=[...])` ile takılıyor ve her adımda
`terminator.is_terminated(messages)` çağrılıyor (`chat_agent.py:663`, `3122–3126`).

**Envanterdeki tek "kullanıcı tanımlı içerik döngüsü dedektörü".** Gemini CLI ve Hermes
kendi chanting dedektörlerini gömüyor; CAMEL bunu **genişletme noktası** olarak sunuyor.
Ama kutudan boş geliyor (`response_terminators or []`, satır 625) ve `words_dict`'i
kullanıcı doldurmak zorunda. Yani: *altyapı var, politika yok.*

**Maliyet/süre bütçesi: yok.** `camel/societies/role_playing.py`'de tur sınırı yok;
`chat_turn_limit` örneklerde (`examples/ai_society/role_playing.py`) çağıran tarafın
`for` döngüsünde duruyor — yani **sınır framework'te değil, örnek kodda.**

---

### A.8 MetaGPT (`FoundationAgents/MetaGPT`) `[K]`

**İki turun tamamında, varsayılanı dolu bir dolar bütçesi olan ikinci harness** (birincisi
SWE-agent'ın `per_instance_cost_limit=3.0`).

`metagpt/utils/cost_manager.py:26–33`:

```python
class CostManager(BaseModel):
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_budget: float = 0
    max_budget: float = 10.0        # ← kutudan 10 dolar
    total_cost: float = 0
    token_costs: dict[str, dict[str, float]] = TOKEN_COSTS
```

`metagpt/team.py`:

```python
investment: float = Field(default=10.0)                 # satır 41

def invest(self, investment: float):
    """Invest company. raise NoMoneyException when exceed max_budget."""
    self.cost_manager.max_budget = investment           # satır 92–96

def _check_balance(self):                               # satır 98–100
    if self.cost_manager.total_cost >= self.cost_manager.max_budget:
        raise NoMoneyException(self.cost_manager.total_cost,
                               f"Insufficient funds: {self.cost_manager.max_budget}")
```

**Uygulama noktası tur başında** (`Team.run`, satır 123–137):

```python
async def run(self, n_round=3, idea="", send_to="", auto_archive=True):
    """Run company until target round or no money"""
    while n_round > 0:
        if self.env.is_idle: break
        n_round -= 1
        self._check_balance()
        await self.env.run()
```

Üç gözlem:

1. **Bütçe metaforu ürün diline taşınmış** — `invest()`, `NoMoneyException`,
   *"Run company until target round or no money"*. Envanterdeki tek örnek; bütçeyi teknik
   bir tavan değil, kullanıcının verdiği bir **kaynak tahsisi** yapıyor. CLI'da
   `--investment` bayrağı olarak görünüyor.
2. **Kontrol tur başında, çağrı başında değil.** `_check_balance()` yalnızca `while`
   döngüsünün tepesinde. Bir `env.run()` turu içinde çok sayıda rol çok sayıda LLM çağrısı
   yapabildiği için **aşım mümkün** — sert bir tavan değil, tur granülaritesinde bir kontrol.
   `total_cost >= max_budget` karşılaştırması da aşımdan **sonra** yakalıyor.
3. `n_round=3` varsayılanı envanterin **en düşük tur limiti** (karşılaştırma: Kimi 1000,
   Goose 1000, OpenHands 500, Hermes 500, AgentScope 50, LangGraph 25). MetaGPT'nin
   "yazılım şirketi simülasyonu" mimarisiyle uyumlu: bir tur = tüm rollerin bir geçişi.

`metagpt/roles/role.py:113` ayrıca rol düzeyinde `max_react_loop: int = 1` tutuyor —
**varsayılan 1**, yani standart bir rol tek `_think → _act` yapıp bitiyor
(`_set_react_mode` docstring, satır 273–275). Yalnızca `react_mode="react"` seçilirse
gerçek bir ReAct döngüsü açılıyor (`_react`, satır 454–461:
`while actions_taken < self.rc.max_react_loop`).

**Loop detection: yok** `[K]`. Tekrar/stuck heuristiği bulunamadı; koruma tamamen
bütçe + tur sayacı.

---

### A.9 ChatDev (`OpenBMB/ChatDev`) `[K]` — MAST'ın incelediği sürüm artık mevcut değil

⚠️ **Depo yeniden yazılmış.** MAST çalışmasının incelediği ChatDev (`chatdev/chat_chain.py`,
gömülü `camel/` kopyası, `chat_turn_limit` parametresi) `main` dalında **yok**. Bugünkü
`main` bir **graf tabanlı iş akışı motoru**: `entity/configs/node/` altında `agent`,
`human`, `memory`, `thinking`, `tooling`, `subgraph`, `python_runner`, `skills`,
`passthrough`, `literal` ve — bu çalışma açısından kritik — **`loop_counter`** ve
**`loop_timer`** düğüm tipleri var.

Bu, envanterde eşi olmayan bir mimari karar: **döngü kontrolü bir çalışma zamanı
parametresi değil, grafın birinci sınıf düğümü.**

`entity/configs/node/loop_counter.py:18–45`:

```python
@dataclass
class LoopCounterConfig(BaseConfig):
    max_iterations: int = 10
    reset_on_emit: bool = True
    message: Optional[str] = None
```

Alan açıklaması: *"How many times the loop can run before this node emits an output."*
`max_iterations < 1` config hatası. `reset_on_emit=True` — düğüm çıktı verdikten sonra
sayaç sıfırlanıyor, yani çok geçişli akışlarda aç kalmıyor (Hermes'in `reset_for_turn`
fikriyle aynı).

`entity/configs/node/loop_timer.py:18–48`:

```python
@dataclass
class LoopTimerConfig(BaseConfig):
    max_duration: float = 60.0
    duration_unit: str = "seconds"     # seconds | minutes | hours
    reset_on_emit: bool = True
    message: Optional[str] = None
    passthrough: bool = False
```

**Duvar saati bütçesini birinci sınıf yapan tek framework.** 1. turda süre limiti yalnızca
4 harness'ta vardı ve hepsinde yan bir parametreydi; burada bir **düğüm tipi**.

İki düğüm de "guard node" olarak adlandırılıyor ve `message` alanıyla limit dolduğunda
akışa ne salınacağı belirleniyor — yani sert istisna değil, **grafın devam edeceği ayrı
bir kenar**. Bu, "limit dolunca ne oluyor" sorusuna beşinci bir cevap:
**yönlendirme (routing)**.

**Loop detection (tekrar heuristiği): yok.** Sayaç ve saat var, imza karşılaştırması yok.

---

### A.10 OpenHands varyantları `[K]`/`[?]`

⚠️ **Depo taşınmış:** `All-Hands-AI/OpenHands` → **`OpenHands/OpenHands`** (84.938 yıldız).
Altıncı taşınma.

1. tur `OpenHands` (software-agent-sdk) `StuckDetector`'ını zaten kod düzeyinde işlemişti
(5 senaryo, 20 olaylık pencere, `ConversationExecutionStatus.STUCK`). Bu turda **varyantlara**
bakıldı:

- **Micro-agent / microagents** — bunlar **prompt katmanı**, döngü kontrolü değil.
  Depodaki `microagents/` dizini tetikleyici anahtar kelimelere bağlı markdown talimat
  dosyalarından ibaret; koşum döngüsüne, tur sayacına ya da bütçeye dokunmuyorlar `[K]`.
  Yani "OpenHands Micro-agent'ın loop detection'ı" diye bir şey yok — aynı çekirdek
  `StuckDetector` geçerli.
- **OpenHands Cloud** — barındırılan ürün; koşum zamanı ayarları (varsa hesap düzeyi
  bütçe/kredi tavanları) açık depoda değil `[?]`. Kod düzeyinde doğrulanamadı.

**Sonuç: OpenHands için 1. turdaki satır geçerliliğini koruyor**, varyantlar ayrı bir
kontrol mekanizması eklemiyor.

---

## Bölüm B — Mevcut framework'lerde derin kazı

Her başlıkta iki satır: **temel parametre** (1. turda zaten kayıtlı) ve **bu turda bulunan
ek mekanizma**.

### B.1 LangGraph `[K]`

**Temel (1. tur):** `DEFAULT_RECURSION_LIMIT = 25` → `GraphRecursionError` (sert durdurma).
Loop detection ❌.

**Bu turda bulunan — 1. turdaki satır eksikti, düzeltilmesi gerekiyor:**

#### (1) `RemainingSteps` var — ve `create_react_agent` bunu zarif bozulma için kullanıyor

Brief'in "RemainingSteps diye bir şey var mı" sorusunun cevabı: **evet.**
`libs/langgraph/langgraph/managed/is_last_step.py`:

```python
class RemainingStepsManager(ManagedValue[int]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.stop - scratchpad.step

RemainingSteps = Annotated[int, RemainingStepsManager]

class IsLastStepManager(ManagedValue[bool]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> bool:
        return scratchpad.step == scratchpad.stop - 1
```

Yani kalan adım sayısı **graf durumunun okunabilir bir alanı**. `AgentState`
(`libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py:62`) bunu içeriyor:
`remaining_steps: NotRequired[RemainingSteps]`.

Ve `create_react_agent` docstring'i (satır 432–440) davranışı net yazıyor:

> *"`remaining_steps` is used to limit the number of steps the react agent can take.
> Calculated roughly as `recursion_limit` - `total_steps_taken`. **If `remaining_steps` is
> less than 2 and tool calls are present in the response, the react agent will return a
> final AI Message with the content "Sorry, need more steps to process this request.".
> No `GraphRecursionError` will be raised in this case.**"*

**Bu 1. turun LangGraph satırını değiştiriyor.** Ham `StateGraph` için "sert durdurma"
doğru; ama **prebuilt ReAct ajanı için davranış zarif bozulma.** Yani LangGraph aynı depoda
iki farklı strateji taşıyor — 1. turdaki Çıkarım 7'nin (bağlama göre farklı davranış) yeni
bir örneği. Karşılaştırma tablosunda LangGraph'ı tek satırda "Sert" diye göstermek yanıltıcı.

Ayrıca `IsLastStep`, "bu son adım" bilgisini **düğüme** verdiği için özel bir kapanış
düğümü yazmayı mümkün kılıyor — smolagents'ın `_handle_max_steps_reached()`'ini kullanıcının
kendisinin yazabilmesi demek. Altyapı var, politika kullanıcıda.

#### (2) Checkpointer / durable execution ile "geri sarma" — **manuel, otomatik değil** `[D]`

1. turun Çıkarım 8'indeki *"geri sarma (rollback): 0 — hiçbiri"* satırı LangGraph için
**nüanslı**: mekanizma var ama tetikleyici yok.

- `get_state_history(config)` → checkpoint'ları ters kronolojik listeliyor.
- `invoke(None, checkpoint_config)` → o checkpoint'tan **yeniden oynatıyor** (replay).
- `update_state(checkpoint_config, values={...})` → o noktadan **çatallayan** yeni bir
  checkpoint yaratıyor; `invoke(None, fork_config)` yeni dalı sürdürüyor.

Ama resmî doküman açıkça uyarıyor: *"Replay re-executes nodes — it doesn't just read from
cache. LLM calls, API requests, and interrupts fire again and may return different results."*
Ve **hiçbir hata ya da limit otomatik geri sarma tetiklemiyor** — time travel kasıtlı bir
kullanıcı eylemi.

Yani PoC açısından: *"döngü tespit edildi → N adım geri sar → farklı bir dal dene"*
LangGraph'ta **yazılabilir** (`get_state_history` + `update_state` + `invoke`), ama
framework bunu kendisi yapmıyor. **Envanterdeki en yakın "geri sarma" altyapısı bu.**

#### (3) `interrupt()` bir döngü kırıcı olarak — evet, ama insan gerektiriyor

`interrupt()` grafı durdurup insana soruyor, `Command(resume=...)` ile devam ediyor.
Roo Code'un `ask("mistake_limit_reached")` ve opencode'un `permission.ask({doom_loop})`
desenleriyle **aynı sınıf**, ama LangGraph'ta bunu döngüye bağlayan gömülü bir tetikleyici
**yok** — `RemainingSteps` okuyup eşik altında `interrupt()` çağıran düğümü kullanıcı yazar.
Yine: altyapı var, politika yok.

#### (4) LangSmith tarafında maliyet/tur alarmı `[?]`

Açık kaynak depoda bulunamadı; LangSmith bulut ürününün panel üzerinden tanımlanan
uyarıları kapsam dışı (1. turdaki §6.6 boşluğunun aynısı). **Doğrulanamadı.**

---

### B.2 CrewAI `[K]`

**Temel (1. tur):** `Agent.max_iter=25`, `PlanningConfig.max_steps=20`,
`max_step_iterations=15`, `respect_context_window=True`, `max_execution_time`,
`step_timeout=None`, `max_rpm`. Loop detection ❌.

**Bu turda bulunan:**

#### (1) Guardrail API'si var — ve **sınırlı sayıda yeniden denemeye** bağlı

`lib/crewai/src/crewai/utilities/guardrail_types.py`:

```python
GuardrailCallable: TypeAlias = Callable[[TaskOutput | LiteAgentOutput], tuple[bool, Any]]
GuardrailType: TypeAlias = GuardrailCallable | str
GuardrailsType: TypeAlias = Sequence[GuardrailType] | GuardrailType
```

Bir guardrail ya bir **fonksiyon** ya bir **doğal dil dizesi** (dize verilirse
`crewai/tasks/llm_guardrail.py` bunu bir LLM yargıcına çeviriyor). Ayrıca hazır bir
`HallucinationGuardrail` (`crewai/tasks/hallucination_guardrail.py`) var.

Kritik nokta — **guardrail başarısızlığı bir yeniden deneme döngüsü açıyor ve bu döngü
sınırlı** (`lib/crewai/src/crewai/task.py:279–282`):

```python
guardrail_max_retries: int = Field(default=3, description="Maximum number of retries when guardrail fails")
retry_count: int = Field(default=0, description="Current number of retries")
```

(`max_retries` aynı işi yapan eski ad; v1.0.0'da kalkıyor, satır 275–278 + 574–583.)
Çoklu guardrail'de her biri için **ayrı sayaç**: `_guardrail_retry_counts: dict[int, int]`
(satır 304–306). Yani 5 guardrail × 3 deneme = 15 ek görev koşumu mümkün — bütçe
guardrail sayısıyla çarpılıyor. **Envanterde "doğrulama döngüsü" için ayrı bütçesi olan
tek framework**, ama bu bütçenin çarpımsal olduğu kodda uyarılmıyor.

Guardrail bu konuda **dolaylı bir döngü kırıcı**: "aynı yanlış çıktıyı tekrar üretme"
davranışını yakalayabiliyor, ama bunu tekrar heuristiğiyle değil, **çıktı doğrulamasıyla**
yapıyor.

#### (2) Checkpoint + fork — 1. turdaki "rollback: 0" satırını kıran ikinci örnek

`lib/crewai/src/crewai/crew.py`:

```python
checkpoint: Annotated[... , BeforeValidator(_coerce_checkpoint), ...]   # satır 397–402
                                # "Automatic checkpointing configuration."
@classmethod
def from_checkpoint(cls, config: CheckpointConfig) -> Crew:             # satır 429–451
    """Restore a Crew from a checkpoint, ready to resume via kickoff()."""
@classmethod
def fork(cls, config: CheckpointConfig) -> Crew:                        # satır 459+
    """Fork a Crew from a checkpoint, creating a new execution branch."""
```

Granülarite **görev (task)** düzeyinde: *"Call kickoff() to resume from the last completed
task."* LangGraph'ın düğüm düzeyi checkpoint'inden kaba, ama aynı fikir. Yine **manuel** —
limit ya da döngü otomatik geri sarma tetiklemiyor.

⚠️ Kayda değer bir tuzak, kodun kendi uyarısı
(`crewai/utilities/guardrail.py:14–31`): **callable guardrail'ler checkpoint'e
serileştirilemiyor ve sessizce düşürülüyor.**

> *"Callable {field_name!r} cannot be JSON-serialized and will be dropped during
> checkpointing; restored checkpoints will not run this guardrail."*

Yani checkpoint'ten geri yüklenen bir crew **guardrail'siz** koşabiliyor. Yalnızca dize
(doğal dil) guardrail'ler hayatta kalıyor. Bu, "güvenlik mekanizması kurtarma yolunda
kayboluyor" tipinde birinci sınıf bir bulgu — PoC'de checkpoint önerilecekse mutlaka
yanına yazılmalı.

#### (3) `step_callback` ile döngü tespiti yazılabilir mi — evet

`Crew.step_callback` (`crew.py:296`) ve `Agent.step_callback` her ajan adımından sonra
çalışıyor (`crew.py:194`: *"Callback to be executed after each step for every agents
execution"*). Bir kullanıcı burada araç imzası hash'i biriktirip tekrar sayabilir.
Ama **callback'in koşumu durdurma yetkisi yok** — dönüş değeri kullanılmıyor, yalnızca
gözlem. Durdurmak için `Task.guardrail` ya da bir istisna fırlatmak gerekiyor.
Yani: *gözlem noktası var, kesme noktası yok.*

#### (4) `crew.kickoff()` düzeyinde bütçe — **yok**

Crew düzeyinde token/dolar/adım tavanı bulunamadı. `max_rpm` (satır 322) bir hız limiti,
bütçe değil. Bütçe yalnızca ajan düzeyinde (`max_iter`) ve görev düzeyinde
(`max_execution_time`, `guardrail_max_retries`). **Toplam koşum maliyetine bir tavan
konamıyor** — MetaGPT'nin `Team.investment`'ının tam zıddı.

---

### B.3 Google ADK `[K]`

**Temel (1. tur):** `LoopAgent.max_iterations` varsayılan `None` = sınırsız;
`RunConfig.max_llm_calls` → `LlmCallsLimitExceededError`. Loop detection ❌.

**Bu turda bulunan:**

#### (1) `max_llm_calls`'ın sayısı: **500**, ve env ile ezilebiliyor

`src/google/adk/agents/run_config.py:37–50, 227–238`:

```python
_DEFAULT_MAX_LLM_CALLS = 500
def _default_max_llm_calls() -> int:
    if env_val := os.getenv('ADK_MAX_LLM_CALLS'):
        return int(env_val)      # geçersizse uyarı + varsayılan
    ...
max_llm_calls: int = Field(default_factory=_default_max_llm_calls, ...)
```

1. tur bu sayıyı vermiyordu. **500**, OpenHands ve Hermes'le aynı büyüklük sınıfında.
Yani ADK'nın "sınırsız" görünen tarafı `LoopAgent.max_iterations`; koşum düzeyinde
gerçek bir tavan var ve **varsayılan olarak açık.** Bu, 1. turdaki §5.3 listesinde ADK'nın
konumunu yumuşatıyor.

#### (2) `LoopAgent` **kullanımdan kaldırıldı** ⚠️

`src/google/adk/agents/loop_agent.py:53–74`:

> *"LoopAgent is deprecated in favor of Workflow and will be removed in a [future release]."*

Döngü hâlâ aynı: `while (not self.max_iterations or times_looped < self.max_iterations)
and not (should_exit or pause_invocation)` (satır 95–97). `max_iterations=None` ise
*"the loop agent will run indefinitely until a sub-agent escalates"* (satır 76–80).
Yeni `Workflow` API'sinde döngü sınırının nasıl taşındığı bu turda **doğrulanamadı** `[?]`.

#### (3) `escalate` — döngüden çıkışın asıl mekanizması

`src/google/adk/events/event_actions.py:125–126`:

```python
escalate: Optional[bool] = None
"""The agent is escalating to a higher level agent."""
```

`LoopAgent` bir alt-ajan `escalate=True` içeren bir olay yayınladığında duruyor
(`loop_agent.py:60–61`). **Bu bir sayaç değil, bir sinyal** — ve `max_iterations=None`
varsayılanında **tek çıkış yolu** bu. Yani ADK'nın döngü kontrolü, modelin (ya da alt-ajanın)
"bitti" demesine bağlı. Kaçırılan bir `escalate` = sonsuz döngü.

Karşılaştırma: AutoGen'in `TerminationCondition`'ı da sinyal tabanlı ama **birleştirilebilir**
ve `MaxMessageTermination` ile sayaç eklenebiliyor; ADK'da `escalate` ile
`max_iterations` arasında böyle bir bileşim yok, ikisi ayrı `while` koşulunda `and`'leniyor.

#### (4) Plugin sistemi **var** ve callback yüzeyi geniş

`src/google/adk/plugins/` — 15 dosya, `BasePlugin` **13 callback** tanımlıyor
(`base_plugin.py`): `on_user_message_callback`, `before_run_callback`, `on_event_callback`,
`after_run_callback`, `before_agent_callback`, `after_agent_callback`,
`before_model_callback`, `after_model_callback`, `on_model_error_callback`,
`before_tool_callback`, `after_tool_callback`, `on_tool_error_callback`,
`on_agent_error_callback`, `on_run_error_callback`.

**Döngü kırmak için kullanılabilir mi — evet, ve bunun hazır bir örneği depoda var.**
`ReflectAndRetryToolPlugin` (`plugins/reflect_retry_tool_plugin.py`):

```python
def __init__(self, name="reflect_retry_tool_plugin",
             max_retries: int = 3,
             throw_exception_if_retry_exceeded: bool = True,
             tracking_scope: TrackingScope = TrackingScope.INVOCATION):
```

Docstring'inden: *"intercepts tool failures, provides structured guidance to the LLM for
reflection and correction, and retries the operation up to a configurable limit"*,
*"Granular Tracking: Failure counts are tracked per-tool within the defined scope.
A success with one tool resets its counter without affecting others."*

Bu, **Cline'ın `maxConsecutiveMistakes`'inin plugin olarak paketlenmiş hâli** — ama ADK'da
ardışık *hata* sayılıyor, ardışık aynı *çağrı* değil. Yani hâlâ tam bir tekrar dedektörü
değil. `throw_exception_if_retry_exceeded=False` seçilirse limit dolunca istisna yerine
yönlendirme dönüyor (nudge/zarif ikilisi arasında seçim).

`TrackingScope.INVOCATION` (varsayılan) / `GLOBAL` ayrımı da kayda değer: sayacın ömrünü
tur mu yoksa oturum mu belirleyecek sorusuna açık bir cevap. Hermes'in `reset_for_turn`'ü
ve ChatDev'in `reset_on_emit`'i aynı sorunun başka cevapları.

**Kutudan kayıtlı mı?** Hayır — plugin listede ama kullanıcı `Runner(plugins=[...])`
ile eklemedikçe çalışmıyor `[K]`. Bir kez daha: *altyapı var, politika kullanıcıda.*

---

### B.4 AutoGen `[K]`

**Temel (1. tur):** `MaxMessageTermination`, `TokenUsageTermination`, `TimeoutTermination`;
`&` / `|` ile birleştirilebilen `TerminationCondition`. Loop detection ❌.

**Bu turda bulunan:**

#### (1) `GraphFlow` — envanterdeki tek **yapısal (statik) döngü koruması**

Bu, iki turun en özgün tek bulgusu olabilir. Diğer herkes döngüyü **çalışma zamanında**
tespit ediyor; AutoGen'in `GraphFlow`'u grafı **kurulurken doğruluyor.**

`autogen-agentchat/.../teams/_group_chat/_graph/_digraph_group_chat.py:149–197`:

```python
def has_cycles_with_exit(self) -> bool:
    """Check if the graph has any cycles and validate that each cycle has at least
    one conditional edge.
    Raises:
        ValueError: If there is a cycle without any conditional edge.
    """
    ...
            if all(edge.condition is None and edge.condition_function is None
                   for edge in cycle_edges):
                raise ValueError(
                    f"Cycle detected without exit condition: "
                    f"{' -> '.join(cycle_nodes + cycle_nodes[:1])}"
                )
```

DFS ile çevrim aranıyor; bulunan her çevrimin kenarlarından **en az biri koşullu olmak
zorunda**, yoksa `graph_validate()` (satır 207+) grafı reddediyor.

**Anlamı:** "çıkışı olmayan döngü" bir çalışma zamanı hatası değil, bir **konfigürasyon
hatası**. Bu, IAL-SCAN'in AutoGen'i döngülerin yoğunlaştığı iki framework'ten biri olarak
göstermesine karşı doğrudan bir cevap — ama yalnızca `GraphFlow` topolojisi için.
`RoundRobinGroupChat` / `Swarm` / `SelectorGroupChat` bu korumadan yararlanmıyor.

Ek olarak `DiGraphEdge.activation_group` ve `activation_condition` ("all" | "any")
alanları var (satır 48–62): docstring örneği aynen bir döngü — *"In a graph containing a
cycle like A->B->C->B, the two edges pointing to B (A->B and C->B) can be in different
activation groups to control how B is activated."* Yani AutoGen döngüyü yasaklamıyor,
**modelliyor**.

#### (2) `Swarm`'da tur sınırı — taşınıyor ama varsayılan yok

`_swarm_group_chat.py`: `Swarm.__init__(..., max_turns: int | None = None, ...)` ve
docstring (satır 147): *"The maximum number of turns in the group chat before stopping.
**Defaults to None, meaning no limit.**"* Değer `SwarmConfig.max_turns` (satır 122)
üzerinden `BaseGroupChatManager`'a aktarılıyor (satır 28, 41).

Yani **topolojiler arası taşıma mekanik olarak var** (aynı `max_turns` alanı
`BaseGroupChat`'ten türeyen tüm takımlarda) ama **varsayılan kapalı.** 1. turdaki
"kutudan sınırsız" listesine bir isim daha.

#### (3) Alt-takımlara bütçe devri — **yok** `[K]`

İç içe takımlarda (bir takımın katılımcısı olarak başka bir takım) üst takımın kalan
`max_turns`'ünün alt takıma devredildiğine dair bir mekanizma bulunamadı. Her takım kendi
`max_turns`'ünü taşıyor. Yani hiyerarşik bütçe modeli **Hermes'inkiyle aynı sınıfta**
(bağımsız alt bütçeler, toplamda tavan yok) — Codex'in paylaşımlı sayacı ya da
SWE-agent'ın devredilen bütçesi gibi bir şey yok.

`TerminationCondition`'ın bileşilebilirliği burada da yardım etmiyor: bir koşul nesnesi
iki takım arasında paylaşılırsa durum da paylaşılır, ama bu **kullanıcının kurması gereken
bir şey**, framework'ün yaptığı bir şey değil.

---

### B.5 LlamaIndex Workflows `[K]`

**Temel (1. tur):** envantere girmemişti.

**Bu turda bulunan:** `llama-index-core/llama_index/core/agent/workflow/base_agent.py`

```python
DEFAULT_MAX_ITERATIONS = 20                                    # satır 67

early_stopping_method: Literal["force", "generate"] = Field(   # satır 139–142
    default="force",
    description="Method to handle max iterations. 'force' raises an error (default). "
                "'generate' makes one final LLM call to generate a response.",
)
```

**Limit dolunca ne olacağını kullanıcıya seçtiren tek framework.** Diğer herkes tek bir
strateji koda gömüyor (LangGraph sert, smolagents zarif, Goose kullanıcıya sorar);
LlamaIndex ikisini de sunup varsayılanı **sert** yapıyor.

`generate` seçilirse `_generate_early_stopping_response()` (satır 482–491) çalışıyor ve
şu prompt'u kullanıyor (`agent/workflow/prompts.py:17–19`):

```
You have reached the maximum number of iterations ({max_iterations}).
Based on the information gathered so far, please provide a helpful final response
to the user's original query.
Do not attempt to use any more tools. Simply summarize what you have learned and
provide the best possible answer.
```

Envanterdeki en açık "zarif bozulma prompt'u" bu — smolagents ve DSPy'nin yaptığı şeyin
metni. PoC'de doğrudan uyarlanabilir. (Not: prompt yalnızca *rica ediyor*
— "Do not attempt to use any more tools" bir kural değil; AgentScope'un `tool_choice`
zorlaması bundan daha güçlü.)

`max_iterations` çalışma zamanında da ezilebiliyor: `ev.get("max_iterations")` başlangıç
olayından okunuyor (satır 295–299), ve `Context.store`'da tutuluyor — yani **koşum
sırasında değiştirilebilir bir bütçe**. Sayaç `num_iterations` (satır 309) her koşumda
sıfırlanıyor.

Ayrıca `Workflow` düzeyinde `timeout` parametresi ve `WorkflowTimeoutError`
(`core/workflow/errors.py`) var — süre limiti birinci sınıf.

**Loop detection: yok.** Tekrar heuristiği aranıp bulunamadı. **Maliyet bütçesi: yok.**

---

### B.6 Haystack `[K]`

**Temel (1. tur):** envantere girmemişti.

`haystack/components/agents/agent.py`:

```python
_EXIT_REASON_MAX_STEPS = "max_agent_steps"        # satır 71
max_agent_steps: int = 100                        # satır 374, 404
exit_conditions: list[str] | None = None          # satır 372 → varsayılan ["text"] (satır 453–454)
```

Döngü: `while exe_context.counter < self.max_agent_steps:` (satır 872).

**İki iyi tasarım kararı:**

1. **Makine-okunur çıkış sebebi, ve modül başında yorumla gerekçelendirilmiş** (satır 69):
   *"`max_agent_steps` budget running out. A tool exit condition instead reports the tool's
   name."* Yani `exit_reason` ya `"max_agent_steps"` ya da çıkışı tetikleyen aracın adı —
   iki durum karışmıyor. Çıkarım 10'un istediği şey.
2. **`exit_conditions` bir liste** ve varsayılanı `["text"]` (model metin ürettiğinde çık).
   Kullanıcı buraya araç adı yazarak "şu araç çağrıldıysa bitir" diyebiliyor. AutoGen'in
   `TerminationCondition`'ının basitleştirilmiş hâli.

Ayrıca `streaming_callback` her adımdan sonra çalışıyor (satır 843: *"after each step
completes, including the final step that hits an exit condition or `max_agent_steps`"*)
ve bir hook **kendi çıkış sebebini** sağlayabiliyor (satır 850: *"or a custom reason a
hook supplied"*). Yani CrewAI'ın `step_callback`'inin aksine **buradaki hook koşumu
durdurabiliyor** — kullanıcının döngü dedektörü yazması için gerçek bir kesme noktası.

`max_agent_steps` telemetriye de yazılıyor: `"haystack.agent.max_steps"` ve
`"haystack.agent.exit_conditions"` span attribute'ları (satır 679–681). OTel GenAI
semconv'un tanımlamadığı bir alanı Haystack kendi ad alanında tanımlamış.

**Loop detection: yok. Maliyet/süre bütçesi: yok** (adım sayacı tek koruma).

---

### B.7 Semantic Kernel `[K]`

**Temel (1. tur):** envantere girmemişti.

İki ayrı katmanda iki ayrı limit:

**(1) Araç çağırma katmanı** —
`python/semantic_kernel/connectors/ai/function_choice_behavior.py:18, 58`:

```python
DEFAULT_MAX_AUTO_INVOKE_ATTEMPTS = 5
maximum_auto_invoke_attempts: int = DEFAULT_MAX_AUTO_INVOKE_ATTEMPTS
```

Zarif bir ayrıntı (satır 66–73): `auto_invoke_kernel_functions` bir **türetilmiş özellik**
— `maximum_auto_invoke_attempts > 0` demek. Setter'ı `True` için 5, `False` için 0 koyuyor.
Yani **"otomatik araç çağırmayı kapat" ile "bütçeyi sıfırla" aynı şey.** Boolean bir bayrak
yerine sayısal bir bütçe kullanan tek tasarım; "kapalı" durumu bütçenin özel bir hâli.

**5, envanterdeki en düşük araç-çağrısı bütçesi** (karşılaştırma: LangGraph 25,
LlamaIndex 20, Haystack 100, ADK 500). SK bir "agent framework"ten çok bir "kernel"
olduğu için varsayılanı muhafazakâr.

**(2) Çoklu ajan katmanı** —
`python/semantic_kernel/agents/strategies/termination/termination_strategy.py:19–24`:

```python
class TerminationStrategy(KernelBaseModel):
    maximum_iterations: int = Field(default=99)
    automatic_reset: bool = False
    agents: list[Agent] = Field(default_factory=list)
```

`maximum_iterations = 99` ve **`automatic_reset = False`** — ikincisi kayda değer: sayaç
sohbet turları arasında **sıfırlanmıyor**, yani 99 iterasyon tüm `AgentGroupChat`'in ömrü
boyunca geçerli. Hermes'in `reset_for_turn`'ü ve ChatDev'in `reset_on_emit`'i tersini
yapıyor. SK burada **daha güvenli** varsayılanı seçmiş: sayacın sıfırlanması opt-in.

**Loop detection: yok. Maliyet/süre bütçesi: yok.**

---

### B.8 Mastra `[K]`

**Temel (1. tur):** envantere girmemişti.

`packages/core/src/loop/types.ts:225–226`:

```ts
stopWhen?: StopCondition | Array<StopCondition>;
maxSteps?: number;
```

**İkisi de opsiyonel, ikisinin de varsayılanı yok.** Uygulama noktası
`packages/core/src/loop/workflows/agentic-loop/index.ts:190–201`:

```ts
// Only call stopWhen if we're continuing (not on the final step)
if (rest.stopWhen && typedInputData.stepResult?.isContinued && accumulatedSteps.length > 0) {
  const conditions = await Promise.all(
    (Array.isArray(rest.stopWhen) ? rest.stopWhen : [rest.stopWhen]).map(condition =>
      condition({ steps })),
  );
```

`rest.stopWhen &&` guard'ı kritik: **kullanıcı `stopWhen` vermezse hiçbir koşul
değerlendirilmiyor** ve döngünün tek sonu modelin araç istemeyi bırakması oluyor.
Yani Mastra kutudan **Continue ve OpenClaw ile aynı sınıfta** — sayaçsız.

`StopCondition` bir dizi olabildiği için `[stepCountIs(20), tokenBudgetExceeded(...)]`
gibi bileşimler yazılabiliyor (AutoGen'in `TerminationCondition` bileşimiyle aynı fikir,
Vercel AI SDK kökenli). `accumulatedSteps` biriktirildiği için koşul **tüm adım geçmişini**
görüyor — bir kullanıcı burada tekrar dedektörü yazabilir. Envanterdeki en esnek
kullanıcı-tanımlı durdurma API'si, ama boş geliyor.

`packages/core/src/tool-loop-agent/tool-loop-processor.ts` adına rağmen **döngü tespiti
değil** — AI SDK v6'nın `ToolLoopAgent` tipini Mastra ajanına uyarlayan bir adaptör `[K]`.

**Loop detection: yok. Maliyet bütçesi: yok.**

---

### B.9 VoltAgent `[K]`

**Temel (1. tur):** envantere girmemişti.

`packages/core/src/agent/subagent/index.ts:220–228`:

```ts
public calculateMaxSteps(agentMaxSteps?: number): number {
    if (agentMaxSteps !== undefined) return agentMaxSteps;
    // Fall back to original logic
    return this.subAgentConfigs.length > 0 ? 10 * this.subAgentConfigs.length : 10;
}
```

**Envanterdeki tek topolojiye göre uyarlanan varsayılan bütçe.** Yorum gerekçeyi veriyor
(satır 217–219): *"Calculate maximum number of steps based on sub-agents. More sub-agents
means more potential steps."*

Yani `maxSteps` verilmezse: alt-ajan yoksa **10**, n alt-ajan varsa **10n**. Bir supervisor
5 alt-ajanla kurulduysa bütçesi otomatik 50 oluyor.

Bu, 1. turdaki Çıkarım 4'e (adım kötü bir vekil) ikinci cevap: Hermes bazı adımları
saymıyordu, VoltAgent **tavanı işin şekline göre ölçekliyor**. İkisi de "tek sabit sayı"
sorununa kısmi çözüm.

Zayıflığı da açık: 10n **doğrusal** ama alt-ajan etkileşimi doğrusal değil; ayrıca
alt-ajanların kendi adımları bu sayaca yazılmıyor (üst düzey adımlar sayılıyor), yani
gerçek maliyet üstünde bir tavan değil.

**Loop detection: yok. Maliyet/süre bütçesi: yok** `[K]`.

---

### B.10 Strands Agents (AWS) `[K]`

⚠️ **Depo taşınmış:** `strands-agents/sdk-python` → **`strands-agents/harness-sdk`**
(6.992 yıldız). İki turda tespit edilen **yedinci** taşınma. Kaynaklar artık
`strands-py/src/strands/` ve `strands-ts/src/` altında (tek depoda iki dil).

**Bu turda bulunan — Strands'ın gerçek bir döngü dedektörü var, ve heuristiği benzersiz.**

`strands-py/src/strands/multiagent/swarm.py:269–309`:

```python
def __init__(self, nodes: list[Agent], *,
    max_handoffs: int = 20,
    max_iterations: int = 20,
    execution_timeout: float = 900.0,
    node_timeout: float = 300.0,
    repetitive_handoff_detection_window: int = 0,      # ← varsayılan KAPALI
    repetitive_handoff_min_unique_agents: int = 0,
```

Dedektör (satır 222–233):

```python
if repetitive_handoff_detection_window > 0 and len(self.node_history) >= repetitive_handoff_detection_window:
    recent = self.node_history[-repetitive_handoff_detection_window:]
    unique_nodes = len(set(recent))
    if unique_nodes < repetitive_handoff_min_unique_agents:
        return False, (f"Repetitive handoff: {unique_nodes} unique nodes "
                       f"out of {repetitive_handoff_detection_window} recent iterations")
```

**Bu, iki turda görülen bütün heuristiklerden farklı bir fikir.** Herkes *imza eşitliği*
arıyor (aynı araç + aynı argüman + belki aynı sonuç). Strands **çeşitlilik/entropi**
ölçüyor: son N adımda kaç **farklı** düğüm çalıştı? Eşiğin altındaysa döngü.

Avantajı: A-B-A-B, A-B-C-A-B-C ve "üç ajan arasında sonsuz top çevirme" desenlerinin
hepsini **tek bir kuralla** yakalıyor — Gemini CLI'ın k=1..5 çevrim taramasına gerek
kalmadan. Dezavantajı: **meşru** olarak aynı iki ajan arasında gidip gelen bir iş akışını
(kodlayıcı ↔ gözden geçirici) ayırt edemiyor. Muhtemelen bu yüzden **varsayılan kapalı**
(`0`) ve kod hiçbir öneri değeri vermiyor.

Kalan sınırlar tek bir yerde ve hepsinin varsayılanı **dolu** — bu turda görülen en
"kapalı kutu güvenli" ayar seti:

| Sınır | Varsayılan | Sebep metni (satır 210–220) |
|---|---|---|
| `max_handoffs` | 20 | `"Max handoffs reached: {max_handoffs}"` |
| `max_iterations` | 20 | `"Max iterations reached: {max_iterations}"` |
| `execution_timeout` | **900.0 s** | `"Execution timed out: {execution_timeout}s"` |
| `node_timeout` | **300.0 s** | düğüm başına |

Süre limitini **hem toplamda hem düğüm başına** koyan tek framework (SWE-agent'ın
`total_execution_timeout=1800` + `execution_timeout=30` çiftinin ajan düzeyi karşılığı).
Her sınırın **ayrı, insan-okunur bir sebep dizesi** var.

**Tek ajan tarafında ise sayaç yok** `[K]`: `strands-py/src/strands/event_loop/event_loop.py`
`recurse_event_loop()` (satır 407, 959) ve `while True:` (satır 485) — özyineleme derinliği
sayacı bulunamadı. `MAX_ATTEMPTS = 6` / `MAX_DELAY = 240` (satır 62–64) **throttling
yeniden denemesi** için, döngü için değil. Yani: **çok-ajanlı katman iyi korunuyor,
tek-ajan döngüsü korunmuyor** — 1. turdaki Çıkarım 7'nin (aynı depo, farklı bağlam,
farklı varsayılan) bir örneği daha.

Ayrıca `_InflightTurn` (satır 245–265) bir **geri sarma** primitifi taşıyor:
`outcome: Literal["open", "committed", "rolled_back"]`, ve *"A rolled-back turn stays
recorded, so the next checkpoint still knows the node owes work."* Checkpoint/rollback
altyapısı var; döngü tespitine bağlı **otomatik** geri sarma yok.

---

## C. Desen sentezi

Sayılar bu turda kod düzeyinde incelenen **9 yeni harness** (Bölüm A; OpenHands varyantları
1. turda sayıldığı için tekrar sayılmıyor) ve **10 framework derin kazısı** (Bölüm B)
üzerinden. Toplam iki tur: **31 harness/framework**.

### C.1 Loop detection oranı yeni bakılanlarda: 9'da 3 — ve dağılım tesadüfi değil

| Durum | Sayı | Kimler |
|---|---|---|
| **Gerçek tekrar/stuck dedektörü var** | **3** | OpenClaw, Hermes, Strands (swarm katmanı) |
| Kullanıcı takarsa var (altyapı sunulmuş, politika boş) | **2** | CAMEL (`ResponseWordsTerminator`), Mastra (`stopWhen` dizisi) |
| **Hiç yok** | **8** | Kimi CLI, Devin, Qwen-Agent, AgentScope, MetaGPT, ChatDev, LlamaIndex, Haystack, Semantic Kernel, VoltAgent |

**Çıkarım 11 — 1. turun Çıkarım 1'i doğrulanıyor ve keskinleşiyor.** "Loop detection bir
kodlama-agent'ı özelliği, framework özelliği değil" iddiası bu turda da tuttu:
dedektörü olan üçünden ikisi (OpenClaw, Hermes) uzun soluklu kişisel asistan/kodlama
harness'ı. Üçüncüsü (Strands) ise **yalnızca çok-ajanlı katmanında** dedektör taşıyor,
tek-ajan döngüsünde taşımıyor — yani "kim döngüye giriyorsa oraya dedektör konuyor"
kuralı depo içinde bile geçerli.

**Çıkarım 12 — dedektör olgunluğu yeni harness'larda 1. turdakini geçti.** İki tur birlikte
sıralandığında en zengin üç implementasyon: **OpenClaw (6 dedektör)**, **Gemini CLI
(3 dedektör + LLM yargıcı)**, **Hermes (3 araç dedektörü + içerik dedektörü + araç-özel
tavanlar)**. OpenHands'in 5 senaryosu dördüncü sırada. Bu üçü de son bir yıl içinde
yazılmış — alan hızla olgunlaşıyor.

### C.2 Heuristikler: iki turda dört farklı aile ortaya çıktı

1. tur "hepsi araç imzası üzerine kurulu, fark kanonikleştirmede" diyordu. Bu tur bunu
**yanlışladı** — dört ayrı fikir var:

| Aile | Fikir | Kim |
|---|---|---|
| **A. İmza eşitliği** | aynı araç + aynı argüman hash'i | Cline, Roo, opencode, Gemini CLI, OpenHands, deer-flow |
| **B. Sonuç eşitliği** | aynı araç + aynı argüman + **aynı sonuç hash'i** | **OpenClaw**, **Hermes** (`idempotent_no_progress`) |
| **C. Entropi / çeşitlilik** | son N adımda **kaç farklı** aktör/düğüm çalıştı | **Strands** (`repetitive_handoff`) |
| **D. İçerik tekrarı (chanting)** | üretilen metinde birebir tekrar baskınlığı | Gemini CLI, **Hermes** (`repetition_guard`), CAMEL (kullanıcı tanımlı) |

**Çıkarım 13 — B ailesi (sonuç hash'i) PoC için en önemli yeni bilgi.** A ailesi
"aynı komutu 20 kez çalıştırdım ama çıktı her seferinde farklıydı" durumunu **yanlış
pozitif** olarak işaretler; B ailesi işaretlemez. OpenClaw bunu `resultHash` zinciriyle
(`tool-loop-no-progress.ts:58–60`), Hermes idempotent/mutating araç sınıflandırmasıyla
(`tool_guardrails.py:20–57`) çözüyor. **İlerlemeyi tanımlayan şey çağrı değil, sonuç.**

**Çıkarım 14 — C ailesi (entropi) ucuz ve genel, ama ayarlanamaz.** Strands'in
`len(set(recent)) < min_unique` kuralı tek satırda A-B-A-B, A-B-C-A-B-C ve n-taraflı
top çevirmeyi yakalıyor. Ama meşru döngüsel iş akışlarını ayırt edemediği için **varsayılan
kapalı** ve kod hiçbir önerilen değer vermiyor. Çok-ajanlı PoC için Gemini CLI'ın k-çevrim
taramasından çok daha ucuz bir alternatif.

**Çıkarım 15 — `argument_churn` kaçış davranışını hedefleyen tek dedektör.** OpenClaw
(`tool-loop-argument-churn.ts`) modelin "uyarı aldım, argümanı azıcık değiştireyim"
kaçamağını ayrıca sayıyor ve `livenessSignal` olarak işaretliyor. **Guardrail'in kendisinin
yeni bir hata modu ürettiğinin farkında olan tek implementasyon.**

### C.3 Framework'ler döngü tespitini kullanıcıya bırakıyor — üç farklı olgunlukta

Brief'in ana sorusu buydu. Cevap net: **framework'ler döngü tespitini kendileri yapmıyor,
kullanıcıya bırakıyor.** Ama "bırakma"nın kalitesi çok değişiyor:

| Olgunluk | Ne sunuluyor | Kim |
|---|---|---|
| **1. Sadece gözlem** — callback var, durduramıyor | `step_callback` (dönüş değeri kullanılmıyor) | **CrewAI** |
| **2. Gözlem + kesme** — hook koşumu durdurabiliyor | `streaming_callback` + özel `exit_reason`; 13 plugin callback'i; `interrupt()`; `stopWhen` dizisi | **Haystack**, **Google ADK**, **LangGraph**, **Mastra** |
| **3. Hazır politika paketi** — dedektör yazılmış, takılması yeterli | `ReflectAndRetryToolPlugin(max_retries=3)`, `ResponseWordsTerminator` | **Google ADK**, **CAMEL** |

**Çıkarım 16 — "altyapı var, politika yok" bu turun en tekrarlayan cümlesi.** LangGraph'ta
`RemainingSteps` + `interrupt()` var ama bunları birleştiren gömülü kural yok; ADK'da 13
callback ve hazır bir retry plugin'i var ama `Runner(plugins=[...])`'e eklemeyen hiçbir şey
almıyor; CAMEL'de `ResponseWordsTerminator` var ama `words_dict`'i kullanıcı doldurur;
Mastra'da `stopWhen` bileşilebilir ama `undefined` geldiğinde **hiç değerlendirilmiyor.**

Bu, Atlas'a öneri yazarken doğrudan kullanılacak çerçeve: *soru "framework destekliyor mu"
değil, "framework varsayılan olarak yapıyor mu".*

### C.4 Varsayılan olarak açık olan dedektör: iki turda **bir tane**

1. turun Çıkarım 6'sı ("mekanizma var ama kapalı") bu turda tek bir yerde kırıldı:

| Harness | Dedektör varsayılanı |
|---|---|
| **Hermes** | `warnings_enabled = True` ← **uyarı kademesi AÇIK**; `hard_stop_enabled = False` |
| OpenClaw | `enabled: false` — 6 dedektörün hepsi kapalı |
| Strands | `repetitive_handoff_detection_window = 0` — kapalı |
| CAMEL | `response_terminators = []` — boş |
| Mastra | `stopWhen = undefined` — değerlendirilmiyor |
| Gemini CLI (1. tur) | açık — ama loop detection, tur limiti değil |

**Çıkarım 17 — Hermes'in ikiye bölme çözümü kopyalanmaya değer.** "Uyarı ucuz ve
geri döndürülemez zarar vermez → varsayılan açık; sert kesme kullanıcının işini
yarıda kesebilir → opt-in." Kod yorumu gerekçeyi taşıyor
(`tool_guardrails.py:112–115`). Bu, "varsayılan açık mı kapalı mı" ikilemine üçüncü bir
cevap: **kademeyi böl, ucuz kademeyi aç.**

### C.5 "Geri sarma / checkpoint" sunan kaç tane var — 1. turdaki "0" düzeltiliyor

1. turun tablosu *"Geri sarma (rollback): 0 — hiçbiri"* diyordu. Bu tur bunu düzeltiyor:
**checkpoint altyapısı yaygın, otomatik geri sarma hâlâ sıfır.**

| Framework | Checkpoint granülaritesi | API | Otomatik tetikleyici |
|---|---|---|---|
| **LangGraph** | düğüm (adım) | `get_state_history()` → `invoke(None, ckpt_cfg)`; `update_state()` = **fork** | ❌ tamamen manuel |
| **CrewAI** | görev (task) | `Crew.from_checkpoint()`, `Crew.fork()` | ❌ manuel |
| **Strands** | düğüm turu | `_InflightTurn(outcome="rolled_back")` | ❌ manuel |
| **Google ADK** | ajan durumu | `LoopAgentState` + `_load_agent_state` (resume) | ❌ manuel |
| Diğer 27 | — | — | — |

**Çıkarım 18 — "geri sarma" bir kurtarma özelliği değil, bir hata ayıklama özelliği.**
Dördünde de checkpoint **insan** içindir: yanlış giden bir koşumu geri alıp elle
dallandırmak için. **Hiçbir framework "döngü tespit ettim → 3 adım geri sar → farklı bir
yol dene" yapmıyor.** LangGraph dokümanı bunu ayrıca zorlaştırıyor: *"Replay re-executes
nodes… LLM calls, API requests, and interrupts fire again and may return different
results"* — yani geri sarma **ücretsiz değil**, tekrar para harcatıyor.

PoC'de "otomatik geri sarma" önerilecekse: **envanterde hiç örneği yok**, ama LangGraph'ın
`get_state_history` + `update_state` üçlüsü bunu yazmak için hazır bir zemin.

⚠️ Ve kurtarma yolunun kendi tuzağı var: **CrewAI'da callable guardrail'ler checkpoint'e
serileştirilemiyor ve sessizce düşüyor** (`utilities/guardrail.py:22–29`). Yani
checkpoint'ten dönen koşum **korumasız** devam edebiliyor.

### C.6 Bütçe türleri — beşinci tür ortaya çıktı, ve dolar limiti ikiye yükseldi

| Limit türü | 1. tur (22 harness) | Bu turda eklenen | Not |
|---|---|---|---|
| Adım / tur / çağrı | 19 | +8 | Hâlâ fiili standart |
| Token | 5 | +0 | Qwen'in 58k'sı budama, limit değil |
| **Maliyet (dolar)** | 5 (varsayılanı dolu: **1**) | **+1 → varsayılanı dolu: 2** | **MetaGPT `max_budget=10.0`** |
| Wall-clock süre | 4 | **+4** | ChatDev (`LoopTimerConfig`), Strands (ikili: 900 s + 300 s), Hermes (`run_budget_seconds`, kapalı), LlamaIndex (`timeout`) |
| **Bileşik satıcı birimi (ACU)** | 0 | **+1** | **Devin `max_acu_limit`** — yeni tür |

**Çıkarım 19 — dolar bütçesi hâlâ nadir ama artık tek örnek değil.** SWE-agent
(`per_instance_cost_limit=3.0`) yanına MetaGPT (`max_budget=10.0`) eklendi. İkisi de
**akademik/araştırma kökenli** projeler; ticari harness'ların hiçbirinde varsayılanı dolu
bir dolar limiti yok. Bu bir tesadüf değil: araştırma projesi n×1000 koşum yapar ve
maliyeti önceden bilmek zorundadır.

**Çıkarım 20 — MetaGPT'nin bütçesi sert bir tavan değil.** `_check_balance()` yalnızca
`while` döngüsünün tepesinde (`team.py:133`) ve `total_cost >= max_budget` karşılaştırması
**aşımdan sonra** yakalıyor. Bir turda tüm roller çalıştığı için gerçek harcama bütçeyi
belirgin biçimde geçebilir. *Bütçe kontrolünün granülaritesi bütçenin kendisi kadar önemli.*

**Çıkarım 21 — Devin'in ACU'su ölçülemez bir bütçe.** `max_acu_limit` müşteriye tek bir
kaldıraç veriyor ama ACU'nun formülü (token + VM süresi + araç çağrısı?) yayımlanmamış.
Yani kullanıcı **bütçesini kendi araçlarıyla doğrulayamıyor.** Açık kaynak harness'larla
ticari harness'lar arasındaki en keskin fark bu.

### C.7 Adımın kötü bir vekil olmasına iki yeni cevap

1. turun Çıkarım 4'ü *"adım en yaygın ama en kötü ölçü"* diyordu ve çözüm önermiyordu.
Bu turda iki farklı çözüm bulundu:

- **Hermes — bazı adımları sayma.** `IterationBudget.refund()` (`iteration_budget.py:41–46`)
  `execute_code` turlarını bütçeye geri veriyor; gerekçe: bir `execute_code` turunda 10 araç
  çalıştırmak 10 ayrı LLM turundan ucuz. **Bütçe "iş" değil, "LLM turu" cinsinden ölçülüyor.**
- **VoltAgent — tavanı işin şekline göre ölçekle.** `calculateMaxSteps()`
  (`subagent/index.ts:220–228`): alt-ajan yoksa 10, n alt-ajan varsa 10n.

İkisi de kısmi. Ama ikisi de "tek sabit sayı" varsayımını kıran, PoC'de tartışılabilir
somut mekanizmalar.

### C.8 Sonlanma sebebi: iki uç, aynı kültürel havzadan

**Çıkarım 22 — 1. turun Çıkarım 10'u ("bütçe tükenmesi sessizce başarı gibi görünüyor")
bu turda hem en kötü hem en iyi örneğini buldu.**

| | Harness | Ne yapıyor |
|---|---|---|
| **En kötü** | **Qwen-Agent** | `while True and num_llm_calls_available > 0` — kota bitince istisna yok, mesaj yok, `stop_reason` yok, log yok. Çağıran taraf normal tamamlanmadan **ayırt edemiyor** |
| **En iyi** | **AgentScope** | `ReplyFinishedReason.EXCEED_MAX_ITERS` (≠ `COMPLETED`) + ayrı `ExceedMaxItersEvent` olayı |
| | **Haystack** | `_EXIT_REASON_MAX_STEPS = "max_agent_steps"`, araç çıkışından ayrı tutulmuş, span attribute'una yazılıyor |
| | **Strands** | her limit için ayrı insan-okunur sebep dizesi (`"Max handoffs reached: 20"` vs `"Repetitive handoff: 2 unique nodes out of 5"`) |
| | **CAMEL** | `_step_terminate(reason=...)` |

Ölçülebilirlik tamamen buna bağlı: PoC'de "kaç koşum bütçe yüzünden bitti" sorusu
Qwen-Agent'ta **cevaplanamaz**, AgentScope'ta bir enum sayımı.

### C.9 Limit dolunca ne oluyor — beşinci strateji eklendi

1. tur dört strateji sayıyordu (sert / nudge / zarif / kullanıcıya sor). Bu tur ikisini
ekliyor:

| Strateji | Bu turda kimler |
|---|---|
| **Sert** | MetaGPT (`NoMoneyException`), LlamaIndex (`early_stopping_method="force"`, varsayılan), OpenClaw (2. critical), Hermes (`halt`) |
| **Nudge** | Hermes (guardrail `warn`, süre bütçesinde %80), OpenClaw (`warning`), ADK (`ReflectAndRetryToolPlugin`) |
| **Zarif** | AgentScope (`tool_choice` **zorlamalı** + 5 lütuf turu), LlamaIndex (`"generate"`), Hermes (1 grace call + özet zorlaması) |
| **Kullanıcıya sor** | — (bu turda yeni örnek yok) |
| **🆕 Yönlendirme (routing)** | **ChatDev** — `LoopCounter`/`LoopTimer` düğümü `message` salıyor, graf **başka bir kenardan devam ediyor** |
| **🆕 Sessiz** | **Qwen-Agent** — hiçbir sinyal yok |

**Çıkarım 23 — "zarif bozulma"nın da kademeleri var.** En zayıfı LlamaIndex'in prompt'u
(*"Do not attempt to use any more tools"* — sadece rica). En güçlüsü AgentScope: araç
seçimi API düzeyinde `tool_choice` ile kilitleniyor, model başka bir şey **yapamıyor**,
üstelik bunun için ayrılmış 5 turluk ayrı bir bütçe var. **PoC'de zarif bozulma
uygulanacaksa modelin uyacağına güvenilmemeli; API kısıtı konmalı.**

### C.10 Yapısal (statik) koruma — envanterin tek örneği

**Çıkarım 24 — AutoGen `GraphFlow`, döngüyü çalışma zamanında değil, graf doğrulamada
yakalayan tek mekanizma.** `has_cycles_with_exit()` her çevrimin en az bir **koşullu
kenarı** olmasını şart koşuyor, yoksa `ValueError: Cycle detected without exit condition`.

Bu, sorunu bütünüyle farklı bir yere taşıyor: *çıkışı olmayan döngü bir çalışma zamanı
hatası değil, bir konfigürasyon hatası.* Çalışma zamanı dedektörlerinin yanlış pozitif /
yanlış negatif problemi burada hiç doğmuyor.

Sınırı: yalnızca **graf topolojisi bilinen** sistemlerde işe yarıyor. Serbest bir ReAct
döngüsünde (model hangi aracı çağıracağına kendi karar veriyor) uygulanabilir değil. Ve
AutoGen'in kendi diğer topolojileri (`RoundRobinGroupChat`, `Swarm`, `SelectorGroupChat`)
bu korumadan yararlanmıyor.

### C.11 Bütçe farkındalığı — 1. turun Çıkarım 9'una ampirik itiraz

1. tur *"modele kendi bütçesini söylemek ucuz, uygulaması kolay ve sunumda ayırt edici bir
öneri"* diyordu (Goose `<turn-budget>`, Codex `<rollout_budget>`). Bu turda:

**Destekleyen:** OpenClaw A2A ping-pong'da her turda `Turn ${turn} of ${maxTurns}` veriyor
**ve** modele erken çıkış jetonu sunuyor (`REPLY_SKIP_TOKEN`) — envanterdeki tek
"erken çıkış protokolü". Hermes süre bütçesinde %80'de wrap-up notu enjekte ediyor.

**Çelişen — ve bu daha önemli:** Hermes iterasyon bütçesinde ara uyarıları **denemiş ve
geri almış** (`agent_init.py:986–991`):

> *"No intermediate pressure warnings — they caused models to 'give up' prematurely on
> complex tasks (#7915)."*

**Çıkarım 25 — bütçe farkındalığı bedava değil; modelin davranışını değiştiriyor, ve
değişim her zaman istenen yönde olmuyor.** PoC'de bu desen önerilecekse:
(a) yalnızca gerçekten tükendiğinde mi yoksa ara eşiklerde mi bildirileceği ayrı bir
karar; (b) "erken pes etme" bir ölçüm hipotezi olarak eklenmeli. Hermes'in aynı depoda
iterasyonda uyarıyı kaldırıp sürede koruması, sorunun **bütçe türüne bağlı** olabileceğini
düşündürüyor `[?]` — gerekçe kodda yazılı değil.

### C.12 Ayarlanabilirlik bir risk olabiliyor

**Çıkarım 26 — OpenClaw eşik ayarlarını kasten kaldırmış.** `tool-loop-thresholds.ts:3–5`:

> *"Numeric loop tuning was retired in #111382. Keep every admission path on the same
> built-in threshold so policy rewrites cannot drift from detection."*

`windowSize`, `historySize`, `warningThreshold`, `criticalThreshold`, `detectors`,
`pingPong`, `genericRepeat`, `globalCircuitBreakerThreshold` — hepsi config yüzeyinden
emekli anahtar listesine taşınmış; geriye tek bir `enabled` boolean'ı kalmış.

Gerekçe: **tespit eşiği ile uygulama eşiği ayrı ayrı ayarlanabilirse birbirinden kayıyor.**
Diğer uçta Hermes 6 eşiği de config'e açıyor, Strands ise iki parametreyi açıp
**hiçbir önerilen değer vermiyor** (ikisi de 0 = kapalı), ki bu pratikte hiç kimsenin
açmaması demek. PoC'de kaç ayar sunulacağı bir tasarım kararı, kolaylık meselesi değil.

### C.13 Doküman/kod ayrışması sistematik

1. turda DSPy'de bir örnek vardı (`max_iters=20` vs docstring "10"). Bu turda:

| Harness | Kodda | Dokümanda/docstring'de |
|---|---|---|
| **Hermes** | `init_agent(max_iterations=sys.maxsize)` | *"default: 90"* |
| **Hermes** | CLI `agent.max_turns = 500`, `delegation.max_iterations = 45` | `IterationBudget` docstring: *"default 500"* / subagent *"default 50"* |

**Çıkarım 27 — bu alandaki hiçbir sayı dokümandan alınmamalı.** İki turda dört bağımsız
tutarsızlık bulundu; hepsi de tam olarak sunumda alıntılanacak türden sayılar.
`[K]` / `[D]` etiketleri bu yüzden var.

---

## D. Boşluklar — dürüst kayıt

**İncelenemeyenler / eksik kalanlar:**

1. **Google ADK'nın yeni `Workflow` API'si `[?]`.** `LoopAgent` deprecated ve
   *"will be removed in favor of Workflow"* diyor, ama `Workflow`'un döngü sınırını nasıl
   taşıdığı bu turda bulunamadı (`src/google/adk/workflows` dizini API üzerinden 404
   döndü). ADK'nın döngü kontrolü hakkındaki en güncel bilgi eksik.
2. **Devin/Cognition tamamen `[D]`.** Koşum zamanı kapalı kaynak; yalnızca resmî API
   dokümanı okundu. ACU'nun formülü, oturum içi tur/süre sınırlarının varlığı, döngü
   tespiti olup olmadığı **bilinmiyor**. Devin'in `max_acu_limit` dolunca ne yaptığı
   (sert kesme mi, uyarı mı) doğrulanamadı.
3. **Kimi CLI'ın limit-doldu davranışı `[?]`.** `max_steps_per_turn=1000` bulundu ama
   döngü gövdesindeki uygulama noktası ve dolduğunda ne olduğu okunmadı.
4. **OpenHands Cloud `[?]`.** Barındırılan üründeki hesap düzeyi kredi/bütçe tavanları
   açık depoda yok. 1. turun §6.6 boşluğunun aynısı, kapatılamadı.
5. **LangSmith maliyet/tur alarmları `[?]`.** Brief'te açıkça soruluyordu; açık kaynak
   depoda bulunamadı, bulut panel özellikleri kapsam dışı bırakıldı. **Cevaplanmadı.**
6. **AutoGen'in `Swarm`/`SelectorGroupChat`'te iç içe takım bütçesi `[K]` ama zayıf.**
   "Devir yok" iddiası `max_turns`'ün `BaseGroupChatManager`'a nasıl aktarıldığını okuyarak
   çıkarıldı; iç içe takım senaryosu için ayrı bir test/örnek izlenmedi.
7. **Mastra'nın `maxSteps`'inin nerede uygulandığı `[?]`.** `stopWhen`'in uygulama noktası
   okundu (`agentic-loop/index.ts:190–201`) ama `maxSteps` alanının ayrıca bir yerde
   `stepCountIs()`'e çevrilip çevrilmediği izlenmedi. "Kutudan sayaçsız" iddiası
   `rest.stopWhen &&` guard'ına dayanıyor — **zayıf halka**.
8. **VoltAgent'ta alt-ajan adımlarının üst sayaca yazılıp yazılmadığı `[?]`.**
   `calculateMaxSteps()` okundu; muhasebenin nerede yapıldığı izlenmedi.
9. **Hermes'in `agent/conversation_loop.py`'si okunmadı.** `iteration_budget`'ın tüketim
   noktası ve `turn_finalizer.py` yalnızca dolaylı olarak (arama sonuçları + `agent_init.py`
   yorumları) çıkarıldı. Bütçe muhasebesinin tam yolu doğrulanmadı.
10. **OpenClaw'ın `stall_guards`/`postCompactionGuard` benzeri kalıntıları.**
    Emekli config anahtarları arasında `postCompactionGuard` görünüyor — bağlam sıkıştırma
    sonrası ayrı bir döngü koruması olduğu anlamına gelebilir; bu turda incelenmedi.
11. **ChatDev'in çalışma zamanı.** `loop_counter` / `loop_timer` **config şemaları** okundu;
    bu düğümleri gerçekten yürüten motor kodu (`entity/` dışındaki çalışma zamanı)
    incelenmedi. "Yönlendirme ile devam ediyor" iddiası şema alanlarına (`message`,
    `passthrough`, `reset_on_emit`) dayanıyor, yürütücü koda değil `[K]`/`[?]`.

**Zayıf iddialar (etiketlerine dikkat):**

- Devin'in **tamamı** `[D]`.
- Hermes'in süre bütçesinde uyarıyı koruyup iterasyonda kaldırmasının gerekçesi `[?]`
  — kodda yazılı değil, çıkarım.
- Mastra'nın "kutudan sayaçsız" olması `[?]` — bkz. boşluk 7.
- ChatDev'in limit-doldu davranışı `[K]`/`[?]` — bkz. boşluk 11.
- OpenHands varyantları için "microagents döngü kontrolüne dokunmuyor" `[K]` ama
  yalnızca dizin yapısı ve tetikleyici mekanizması üzerinden; her microagent dosyası
  okunmadı.

**Bu turda ortaya çıkan depo taşınmaları (1. turdaki dördün üstüne üç tane daha):**

| Brief'te / bilinen adres | Gerçek güncel adres | Not |
|---|---|---|
| `modelscope/agentscope` | **`agentscope-ai/agentscope`** | 29.440 yıldız |
| `All-Hands-AI/OpenHands` | **`OpenHands/OpenHands`** | 84.938 yıldız |
| `strands-agents/sdk-python` | **`strands-agents/harness-sdk`** | Tek depoda `strands-py/` + `strands-ts/` |

Ayrıca **`OpenBMB/ChatDev` taşınmadı ama tamamen yeniden yazıldı** — MAST çalışmasının
incelediği kod tabanı `main` dalında artık yok. Bu, "MAST'ta adı geçen framework'ün gerçek
limitleri" sorusuna verilebilecek en dürüst cevap: **o sürüm artık mevcut değil.**

⚠️ **Araç güvenilirliği notu (1. turdakiyle tutarlı):** `raw.githubusercontent.com` tarama
sırasında bir kez HTTP **429** döndürdü (Hermes deposu); `gh api .../contents` + base64
ile aşıldı. Kimlik doğrulamasız `api.github.com` çağrıları da kota sınırına takıldı —
`gh api` ile tekrarlandı. GitHub Code Search'ün `path:` filtresi monorepo'larda
(CrewAI'ın `lib/crewai/src/...` düzeni) beklenmedik şekilde boş döndü; filtresiz arama
gerekti. **Hiçbir "bulamadım" sonucu tek araçla "yok" diye yazılmadı.**
