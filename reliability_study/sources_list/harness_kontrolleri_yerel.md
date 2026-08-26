# Yerel Kaynak Kodundan Harness İncelemesi

Tarih: 2026-08-21 · Yöntem: bu makinedeki kurulu paketlerin ve klonlanmış repoların
**doğrudan okunması**. Dışarıdan elde edilemeyecek veri — bu iki harness'ın davranışı
dokümanlarında bu ayrıntıda yazmıyor.

Etiketler: `[K]` kaynak kodu okundu · `[Ş]` yapılandırma şemasından okundu · `[?]` doğrulanamadı.

---

## Özet tablo

| Harness | Döngü tespiti | Varsayılan | Adım/tur sınırı | Token/maliyet | Süre | Durum büyümesi koruması |
|---|---|---|---|---|---|---|
| **openclaw** 2026.7.1-2 | **3 dedektör + 3 kademe** | **KAPALI** | yok | yok | var | **kapsamlı** |
| **openclaude** 0.1.2 | **hiç yok** | — | `maxTurns` opsiyonel, **varsayılan yok** | task_budget geçişi | — | compaction |

İki harness birbirinin tam tersi profilde: openclaw döngü tespitinde ve durum büyümesi
korumasında güçlü, tur/maliyet bütçesinde hiç yok; openclaude'da tur sınırı mekanizması var
ama döngü tespiti hiç yok.

---

## 1 · openclaw `2026.7.1-2`

Kurulum: `~/.npm-global/lib/node_modules/openclaw` · yapılandırma: `~/.openclaw/openclaw.json`
Kendi tanımı: *"Multi-channel AI gateway with extensible messaging integrations"*.

Dağıtılan paket derlenmiş (`dist/*.js`); bulgular yapılandırma şemasının gömülü açıklama
metinlerinden ve `before-tool-call` modülünden okundu.

### 1.1 Döngü tespiti — kataloğumuzdaki en olgun sistem `[Ş]`

`tools.loopDetection` altında yedi ayar:

| Ayar | Varsayılan | Ne yapıyor |
|---|---|---|
| `enabled` | **`false`** | Tekrarlayan araç-çağrısı döngü tespiti ve backoff güvenlik kontrolleri |
| `historySize` | 30 | Araç geçmişi pencere boyutu |
| `warningThreshold` | 10 | Tekrarlayan desende uyarı eşiği |
| `criticalThreshold` | 20 | Kritik eşik |
| `globalCircuitBreakerThreshold` | 30 | **Küresel ilerleme-yok devre kesici** |
| `unknownToolThreshold` | 10 | Aynı **mevcut olmayan** araca tekrarlanan çağrıyı engelleme |
| `postCompactionGuard.windowSize` | 3 | Compaction sonrası korumanın kaç deneme boyunca kurulu kalacağı |

Ve **üç adlandırılmış dedektör**, özellik açıkken hepsi varsayılan `true`:

- `detectors.genericRepeat` — aynı araç / aynı parametre
- `detectors.knownPollNoProgress` — bilinen yoklama araçlarında ilerleme yokluğu
- **`detectors.pingPong` — ping-pong (dönüşümlü) döngü tespiti**

### 1.2 Bunun neden önemli olduğu

**Üç bulgu, üçü de daha önceki envanteri düzeltiyor.**

**(a) Dönüşümlü döngüyü yakalayan üçüncü harness.** `harness_kontrolleri.md`'de "incelenen 22
harness'ın yalnızca ikisi (Gemini CLI ve OpenHands) A-B-A-B dönüşümlü döngüyü yakalıyor"
yazıyordu. openclaw üçüncüsü — ve bunu ayrı adlandırılmış bir dedektör olarak sunuyor
(`pingPong`), yani tesadüfen değil kasten.

**(b) Compaction sonrası koruma — kataloğumuzda başka hiç kimsede yok.** `postCompactionGuard`,
bağlam sıkıştırmasından sonra üç deneme boyunca kurulu kalıyor. Bu, Token Budgets
kataloğundaki **Claude Code compaction-loop imzasının** (CCDE-001: tek kullanıcı, 4 günde
235 $) doğrudan karşılığı olan bir savunma. Compaction'ın kendisinin döngü ürettiği bilinen
bir vaka ve openclaw buna özel bir kapı koymuş.

**(c) Üç kademeli eşik.** 10 (uyarı) → 20 (kritik) → 30 (küresel devre kesici). Bizim PoC'de
iki kademe var (nudge → dur); `pi-anti-doom-loop` üç kademeli (steer → abort+devam → dur).
openclaw da üç kademeli ama farklı bir eksende: şiddet değil **kapsam** artıyor — tek desenden
küresel ilerleme-yok kesicisine.

### 1.3 Parmak izi yöntemi `[K]`

`dist/agent-tools.before-tool-call-C95DXQXZ.js` içindeki `recordToolCallOutcome`, her araç
çağrısı sonucunu şu alanlarla kaydediyor:

```
{ toolName, argsHash, resultHash, toolCallOrdinal, ... }
```

Yani parmak izi **araç adı + argüman hash'i + SONUÇ hash'i**. Bu, `loop_budget_source` ve
`sde_offer_loop`'un bağımsız olarak önerdiği `fingerprint(tool, args, outcome)` desenini
üretimde uygulayan tek örnek. Bizim PoC eylem ve gözlemi ayrı hash'liyor.

### 1.4 Durum büyümesi korumaları — IAL-SCAN'in önerisini uygulayan nadir örnek `[Ş]`

IAL-SCAN'in framework yazarlarına verdiği üçüncü öneri "durum büyümesine açık koruma koyun
(mesaj geçmişi boyut sınırı)" idi ve taranan framework'lerin neredeyse hiçbirinde yoktu.
openclaw'da bir `contextLimits` ailesi var:

| Ayar | Varsayılan |
|---|---|
| `bootstrapMaxChars` | 20.000 |
| `bootstrapTotalMaxChars` | 60.000 |
| `startupContext.maxFileBytes` | 16.384 |
| `startupContext.maxFileChars` | 1.200 |
| `startupContext.maxTotalChars` | 2.800 |
| `contextLimits.toolResultMaxChars` | model-bağlamına göre otomatik |
| `contextLimits.postCompactionMaxChars` | — |
| `compaction.maxHistoryShare` | 0,1–0,9 aralığı |
| `compaction.maxActiveTranscriptBytes` | eşik aşılınca compaction tetikleniyor |

Şemadaki kendi ifadesi dikkat çekici: *"sınırlı okuma/enjeksiyon boyutlarını, **herhangi bir
sınırsız çağrı yolunu yeniden açmadan** ayarlamak için kullanın."* Yani "sınırsız yol"
kavramı tasarımda açıkça adlandırılmış.

### 1.5 Bütçe tarafındaki boşluk `[Ş]`

Şemada **tur/adım sınırı yok, dolar bütçesi yok.** Bulunanlar:

- `agents.defaults.runRetries.max` — **160**, açıklaması: *"kaçak yürütmeyi önlemek için
  koşum yeniden deneme iterasyonlarının mutlak üst sınırı"*
- `agents.defaults.timeoutSeconds` ve `agents.defaults.heartbeat.timeoutSeconds`
  (ikincisi ayarlanmazsa heartbeat kadansı, 600 sn'de tavanlanıyor)
- `agents.defaults.compaction.timeoutSeconds` — 180
- `tools.exec.reviewer.timeoutMs` — 30.000

Yani süre ekseni sağlam, **tur ve maliyet ekseni yok**. Bir agent 160 retry iterasyonuna
kadar gidebiliyor ve token maliyeti üzerinde hiçbir tavan yok.

### 1.6 Ana kusur

**`tools.loopDetection.enabled` varsayılanı `false`.** Kataloğumuzun en gelişmiş döngü
tespit sistemi kutudan çıktığında kapalı geliyor. Bu, `harness_kontrolleri.md`'nin ana
desenini bir kez daha doğruluyor: *"var mı" yanlış soru, "varsayılanda açık mı" doğru soru.*

Üstelik burada bedel daha yüksek, çünkü kapalı olan şey basit bir sayaç değil — üç dedektör,
üç kademeli eşik ve compaction koruması içeren bütün bir alt sistem.

---

## 2 · openclaude `0.1.2` (`@gitlawb/openclaude`)

Konum: `~/Desktop/openclaude` · 55 MB TypeScript kaynak · son commit `2d7aa9c`
Kendi tanımı: *"Claude Code opened to any LLM — OpenAI, Gemini, DeepSeek, Ollama, and 200+ models"*.

Yani Claude Code'un yeniden uygulaması. Kaynak açık olduğu için ana koşum döngüsü satır satır
okunabiliyor — bu, kataloğumuzdaki en şeffaf inceleme.

### 2.1 Döngü tespiti: **hiç yok** `[K]`

`src/` altında `loop.?detect`, `stuck`, `repetit`, `doom`, `identical` desenleri arandı.
Dönen sonuçların tamamı alakasız: prompt cache yorumları, dosya kilidi yorumları, UI ilerleme
göstergesi eşiği. **Ajanın tekrara girdiğini fark eden hiçbir mekanizma yok.**

Bu, `harness_kontrolleri.md`'deki Codex CLI ve Continue bulgularıyla aynı kategori: ana araç
döngüsünde tekrar tespiti bulunmayan harness'lar.

### 2.2 Tur sınırı: mekanizma var, varsayılan yok `[K]`

`maxTurns` zorlanıyor — `src/query.ts:1705`:

```ts
// Check if we've reached the max turns limit
if (maxTurns && nextTurnCount > maxTurns) {
  yield createAttachmentMessage({ type: 'max_turns_reached', maxTurns, turnCount: nextTurnCount })
  return { reason: 'max_turns', turnCount: nextTurnCount }
}
```

İki iyi ayrıntı: durma sebebi (`reason: 'max_turns'`) döndürülüyor — Arize'ın "her koşumda
bir durma sebebi kaydedin" gerekliliğini karşılıyor. Ve iptal (abort) yolunda da ayrıca
kontrol ediliyor (`src/query.ts:1508`), yani kaçış yolu bırakılmamış.

Ama SDK şemasında (`src/entrypoints/sdk/coreSchemas.ts:1148`):

```ts
maxTurns: z.number().int().positive().optional()
```

**`.optional()`, varsayılan yok.** Ana koşum döngüsü `while (true)` (`src/query.ts:307`) ve
çağıran `maxTurns` vermezse tur sınırı hiç devreye girmiyor.

### 2.3 En ilginç bulgu: kendi alt-agent'larını sıkı, kullanıcıyı serbest bırakıyor `[K]`

Harness kendi iç işleri için çağırdığı alt-agent'lara açık ve dar sınırlar veriyor:

| İç alt-agent | Sınır | Yer |
|---|---:|---|
| Compaction | `maxTurns: 1` | `services/compact/compact.ts:1194` |
| Hafıza çıkarma | `maxTurns: 5` | `services/extractMemories/extractMemories.ts:426` |
| Prompt spekülasyonu | `MAX_SPECULATION_TURNS = 20` | `services/PromptSuggestion/speculation.ts:58` |

Yani geliştiriciler tur sınırının gerekli olduğunu **biliyor** ve kendi kontrol ettikleri her
yerde uyguluyorlar — 1, 5, 20. Ama kullanıcının ana döngüsünde varsayılan koymuyorlar.

Bu, sunuma girecek türden bir gözlem: **sınırın gerekliliği tartışmalı değil; tartışmalı olan
kimin üstlendiği.** Harness kendi maliyetini sınırlıyor, kullanıcının maliyetini kullanıcıya
bırakıyor.

### 2.4 Claude task_budget geçişi `[K]`

`src/services/api/claude.ts:479` içindeki `configureTaskBudgetParams`, Claude'un beta
`task_budget` özelliğini (`output_config.task_budget`, beta başlığı `task-budgets-2026-03-13`)
API isteğine ekliyor. Kodun kendi yorumu:

```
// API task_budget (output_config.task_budget, beta task-budgets-2026-03-13).
// Distinct from the tokenBudget +500k auto-continue feature. `total` is the
// budget for the whole agentic turn; `remaining` is computed per iteration
// from cumulative API usage.
```

Üç not:
- Özellik `shouldIncludeFirstPartyOnlyBetas()` kapısının arkasında — yani yalnızca birinci
  taraf sağlayıcıda etkin, "200+ model" iddiasının diğer sağlayıcılarında yok.
- Yorumda `tokenBudget` diye **ayrı** bir özellikten söz ediliyor (+500k otomatik devam).
  İkisi karıştırılmasın diye kodda not düşülmüş.
- `remaining` istemci tarafında kümülatif API kullanımından hesaplanıyor — Claude
  dokümanının "geri sayım yalnızca modele görünür, kendi sayacını tut" uyarısına uygun.

Bu, `claude_platform_budget_control.md`'de okuduğumuz model-tarafı öz-düzenleme katmanının
üçüncü taraf bir harness'ta çalışan uygulaması. Kataloğumuzdaki tek örnek.

### 2.5 Diğer sınırlar `[K]`

| Sabit | Değer | Dosya |
|---|---:|---|
| `DEFAULT_MAX_RETRIES` | 10 | `services/api/withRetry.ts:52` |
| `PERSISTENT_MAX_BACKOFF_MS` | 5 dk | `services/api/withRetry.ts:96` |
| `MAX_OUTPUT_TOKENS_RECOVERY_LIMIT` | 3 | `query.ts:164` |
| `MAX_TOTAL_SESSION_MEMORY_TOKENS` | 12.000 | `services/SessionMemory/prompts.ts:9` |
| `POST_COMPACT_MAX_TOKENS_PER_FILE` | 5.000 | `services/compact/compact.ts:124` |
| `MAX_OUTPUT_TOKENS_FOR_SUMMARY` | 20.000 | `services/compact/autoCompact.ts:30` |
| `DEFAULT_MAX_INPUT_TOKENS` | 180.000 | `services/compact/apiMicrocompact.ts:16` |
| `MAX_RECONNECT_ATTEMPTS` (MCP) | 5 | `services/mcp/useManageMCPConnections.ts:88` |
| `maxDepth` (hata serileştirme) | 5 | `services/api/errorUtils.ts:51` — yorumu: *"Prevent infinite loops"* |
| `maxDepth` (dosya sistemi) | 40 | `utils/fsOperations.ts:316` — *"Prevent runaway loops"* |

Son iki satır ilginç: **sonsuz döngü kaygısı kodda açıkça var**, ama yardımcı fonksiyonlarda.
Ana ajan döngüsünde aynı refleks uygulanmamış.

---

## 3 · İki harness'ın birlikte söylediği

**1 · Profiller birbirini tamamlıyor, hiçbiri tam değil.** openclaw'ın döngü tespiti ve durum
büyümesi koruması var, tur/maliyet bütçesi yok. openclaude'un tur mekanizması ve model-tarafı
bütçe geçişi var, döngü tespiti yok. İkisini birleştirmek gerekiyor — ki bu bizim PoC'nin
tasarım gerekçesi.

**2 · "Varsayılanda kapalı" deseni bir kez daha.** openclaw'ın bütün loop detection alt
sistemi `enabled: false`; openclaude'un `maxTurns`'ü `.optional()`. İki bağımsız harness,
aynı tercih. `harness_kontrolleri.md`'deki 10/22 oranı buna göre 12/24 oluyor.

**3 · Geliştiriciler sınırın gerekliliğini biliyor.** openclaude kendi alt-agent'larına 1, 5,
20 tur veriyor; openclaude'un yardımcı fonksiyonlarında *"Prevent infinite loops"* yorumları
var; openclaw'ın şeması "sınırsız çağrı yolu" kavramını adlandırıyor. Bilgi eksik değil —
**varsayılan seçimi bir ürün kararı** ve iki üründe de kullanıcı aleyhine verilmiş.

**4 · Compaction bir döngü kaynağı olarak tanınmış.** openclaw'ın `postCompactionGuard`'ı ve
openclaude'un `autoCompact.ts:259`'daki yorumu (*"hammer the API with doomed compaction
attempts on every turn"*) aynı riski işaret ediyor. Token Budgets kataloğundaki Claude Code
compaction-loop vakası laboratuvar bulgusu değil, iki harness'ın ayrı ayrı savunma geliştirdiği
gerçek bir desen.

---

## 4 · Boşluklar

- **openclaw derlenmiş dağıtılıyor.** Eşik değerleri ve dedektör adları şema açıklamalarından
  okundu; algoritmaların kendisi (`pingPong` tam olarak nasıl çalışıyor, pencere nasıl kayıyor)
  minifiye kodda doğrulanamadı. Kaynak repo `github.com/openclaw/openclaw` — açıksa oradan
  bakılmalı.
- **openclaude'un `tokenBudget` +500k otomatik devam özelliği** incelenmedi; kod yorumunda
  adı geçiyor ama uygulaması ayrıca okunmalı.
- Bu makinedeki diğer büyük repolar (`openshell-deepagent` 178 MB, `UCP-AGENT` 632 MB,
  `llm-agents` 7,1 GB) taranmadı — ilk grep'te bu konuya dair belirgin sinyal vermediler,
  ama `llm-agents/services/token_tracker.py` ve `utils/context_pruner.py` bakılmaya değer.
