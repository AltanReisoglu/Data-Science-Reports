# Tool-Trace Compaction — Diğer Adaylar (OpenClaw · OpenCode · Codex · Claude Code)

> **Kapsam:** Hermes için yazdığımız [hermes-tool-trace-compaction.md](hermes-tool-trace-compaction.md)'nin aynısını **diğer dört aday** için yapar — her sistemin tool izini (tool çağrıları + sonuçları) nasıl küçülttüğünü **en baştan en sona, adım adım**. İstenildiği gibi **OpenClaw'dan başlar**. Her iddia klonlanmış kaynak koda dayanır ([../harnesses/](../harnesses/)); Claude Code hariç (kaynağı kapalı → resmî docs + gözlem).

> **İki felsefe (önce bunu bilelim):** Tool-trace compaction iki ana yolla yapılır:
> - **Deterministik per-tool budama** (LLM'siz): her tool sonucunu tek tek kısalt/dedup/özetle. → Hermes, OpenCode, Codex-truncation.
> - **LLM-chunk özetleme** (model'e sor): tool-çiftlerini bozmadan chunk'la, chunk'ları bir modele özetlet. → OpenClaw, Codex-history, Claude Code.
>
> Çoğu sistem **ikisini birden** kullanır: önce ucuz deterministik budama, dolmaya devam ederse LLM özeti.

---

## 1. OpenClaw — worker-thread'de planlanan, tool-çifti-korumalı chunk özetleme

**Kimlik/felsefe:** OpenClaw, tool sonuçlarını tek tek *budamaz*; bunun yerine **bir plan hesaplar** (kaç chunk, hangi oran, neyi atla), bu planı **ayrı bir worker-thread'de** üretir (ana döngü bloklanmaz), sonra chunk'ları LLM'e özetletir. Ana kaygıları: **güvenlik** (hassas tool detayı modele sızmasın) ve **tool-çifti bütünlüğü** (call/result asla ayrılmasın).

Kaynak: [compaction-planning.ts](../harnesses/openclaw/src/agents/compaction-planning.ts), [compaction-planning.worker.ts](../harnesses/openclaw/src/agents/compaction-planning.worker.ts), [session-transcript-repair.ts](../harnesses/openclaw/src/agents/session-transcript-repair.ts), [agent-compaction-constants.ts](../harnesses/openclaw/src/agents/agent-compaction-constants.ts).

### Sabitler
```ts
BASE_CHUNK_RATIO = 0.4          // pencerenin %40'ı bir chunk hedefi
MIN_CHUNK_RATIO  = 0.15         // adaptif küçültmenin alt sınırı
SAFETY_MARGIN    = 1.2          // token tahmini yanılma payı
SUMMARIZATION_OVERHEAD_TOKENS = 4096   // özet prompt+system+önceki özet+sarmalayıcı payı
MIN_PROMPT_BUDGET_TOKENS = 8000        // mutlak prompt tabanı
MIN_PROMPT_BUDGET_RATIO  = 0.5         // pencerenin en az yarısı prompt'a kalmalı
```

### Adım adım
**Adım 0 — Tetik (bütçe-tabanlı):** Pencerenin en az yarısı (`MIN_PROMPT_BUDGET_RATIO=0.5`) prompt'a kalamıyorsa compaction gerekir; mutlak taban `MIN_PROMPT_BUDGET_TOKENS=8000`.

**Adım 1 — Güvenlik şeritlemesi (SECURITY, kritik):**
```ts
sanitizeCompactionMessages = stripToolResultDetails ∘ stripRuntimeContextCustomMessages
```
Koddaki yorum: *"SECURITY: toolResult.details ve runtime-context transcript girdileri ASLA LLM-facing compaction'a girmez."*
**Neden?** Tool sonuçlarının `details` alanı hassas iç bilgi (ham komut çıktısı, kimlik) taşıyabilir; bunu **özetleyici modele** göndermek sızıntıdır. Runtime-context girdileri de model-görünür değil → token tahmininde **0** sayılır.

**Adım 2 — Token tahmini (per-mesaj, hizalı):** `estimatePerMessageTokens` diziyle 1:1 hizalı tahmin üretir; model-görünmez girdiler 0. `estimateCompactionPlanningTokens = estimateTokens + omittedChars/4` (projeksiyonda atlanan karakterleri de sayar).

**Adım 3 — Tool-çiftine göre gruplama (bütünlük):** `groupCompactionMessages`, mesajları gezerken `pendingToolCallIds` set'i tutar. Bir grup **ancak `pendingToolCallIds.size === 0`** olunca kapatılır.
Koddaki yorum: *"Yerinden oynamış bir user turn'ü hâlâ bitmemiş bir call/result batch'ine aittir; bölmek, ortaya çıkan provider transcript'lerinden birini geçersiz kılar."*
→ Yani bir tool çağrısı ile sonucu **asla farklı chunk'lara düşmez** (yoksa sağlayıcı 400).

**Adım 4 — Adaptif chunk oranı:** `computeAdaptiveChunkRatio` — ortalama mesaj context'in %10'undan büyükse oranı küçültür:
```ts
if avgRatio > 0.10:
    reduction = min(avgRatio*2, BASE-MIN)
    return max(MIN_CHUNK_RATIO, BASE_CHUNK_RATIO - reduction)   // 0.4 → 0.15'e kadar
```
**Neden?** Mesajlar büyükse, büyük chunk'lar model limitini aşar; küçük chunk = güvenli özet.

**Adım 5 — Chunk'lama (çift-korumalı):** `chunkCompactionMessageGroups`, atomik grupları `maxTokens`'a kadar chunk'lara paketler — **bir call/result çiftini asla bölmeden**.

**Adım 6 — Oversized fallback (tek mesaj devse):** `buildOversizedFallbackPlan` — eşik `contextWindow * 0.5`. Bir mesajın `tokens * SAFETY_MARGIN(1.2)` bu eşiği aşıyorsa, o mesaj bir özet isteğine **sığmaz** → içerik yerine bir **`oversizedNote`** (metin not) konur.
**Örnek:** 300K token'lık pencerede, tek bir `read_file` 200K token → `200K*1.2 > 150K` → içeriği değil, "[oversized: read_file result, 200K tokens, omitted]" gibi bir not özetlenir.

**Adım 7 — Aşama bölme (`buildStageSplitPlan`):** özetleme tek chunk mı (`single`) çok chunk mı (`split: chunks[]`) — karar verilir.

**Adım 8 — Transcript onarımı:** `repairToolUseResultPairing` — eksik tool sonucu varsa **sentetik bir hata sonucu** ekler: *"[openclaw] missing tool result in session history; inserted synthetic error result for transcript repair."* `sanitizeToolCallInputs` bozuk tool çağrı girdilerini onarır. → Compaction sonrası transcript **replay için geçerli** kalır.

**Adım 9 — Worker'da çalıştır:** Tüm planlama `compaction-planning.worker.ts` içinde ayrı thread'de; ana döngü bloklanmaz. Sonra chunk'lar LLM'e özetletilir.

### Öne çıkan
OpenClaw'ın tool-trace mantığı **plan-ağırlıklı ve LLM-özetleme-tabanlı**: tek tek tool budamaz, ama (a) tool detayını modelden **saklar** (güvenlik), (b) call/result çiftini **asla bölmez**, (c) devasa tek mesajı **nota indirir**, (d) hepsini **worker-thread'de** planlar, (e) sonrasında transcript'i **onarır**.

---

## 2. OpenCode — iki katman: canlı tool-output truncation + turn-tabanlı prune (POC'a en yakın)

**Kimlik/felsefe:** OpenCode iki **bağımsız** katman kullanır: (a) tool çıktısı üretilirken **anında** kısaltma (spill-to-disk), (b) eşikte **turn-tabanlı** deterministik prune + gerekirse LLM özeti. Sabitleri bizim POC'a şaşırtıcı yakın.

Kaynak: [tool/truncate.ts](../harnesses/opencode/packages/opencode/src/tool/truncate.ts), [session/compaction.ts](../harnesses/opencode/packages/opencode/src/session/compaction.ts), [session/overflow.ts](../harnesses/opencode/packages/opencode/src/session/overflow.ts).

### Katman A — canlı tool-output truncation (üretim anında)
```ts
MAX_LINES = 2000     MAX_BYTES = 50*1024     RETENTION = 7 gün
direction: "head" | "tail"
```
Bir tool çıktısı `MAX_LINES` veya `MAX_BYTES`'ı aşarsa:
1. **Tam metin** truncation-dir'e yazılır (`write`).
2. Modele **önizleme + "kayıtlı dosyaya bak" ipucu** döner (`{content, truncated:true, outputPath}`).
3. 7 gün saklanır, sonra temizlenir (`cleanup`).
`hasTaskTool` kontrolü: ajan Task tool'una sahipse davranış ayarlanır (subagent'a delege edilebilir çünkü).
→ Bu = **spill-to-disk**: dev çıktı context'e girmez, diske gider, referansla erişilir.

### Katman B — turn-tabanlı prune (eşikte)
```ts
PRUNE_MINIMUM = 20_000      PRUNE_PROTECT = 40_000
TOOL_OUTPUT_MAX_CHARS = 2_000
PRUNE_PROTECTED_TOOLS = ["skill"]
DEFAULT_TAIL_TURNS = 2
MIN/MAX_PRESERVE_RECENT_TOKENS = 2_000 / 8_000
```

**Adım 1 — Turn'lere böl:** `turns()` — her user mesajı bir turn başlatır; `compaction` part'ı taşıyan user'lar atlanır. Her turn'ün `[start, end)` aralığı hesaplanır.

**Adım 2 — Yakın bütçeyi koru:** `preserveRecentBudget = cfg.preserve_recent_tokens ?? min(8K, max(2K, usable×0.25))` — son pencere token koruması. Ayrıca `DEFAULT_TAIL_TURNS = 2` → **son 2 turn dokunulmaz** (bizim `RECENT=2` ile birebir).

**Adım 3 — Prune tara:** parça parça gez:
```ts
if (part.state.status !== "completed") continue    // yarım tool'a dokunma
if (part.state.time.compacted) break               // zaten sıkışmış → dur
const estimate = Token.estimate(part.state.output)
pruned += estimate
```
**Adım 4 — Eşik geçilirse işaretle:** `if (pruned > PRUNE_MINIMUM)` → tamamlanmış tool çıktıları **`part.state.time.compacted = Date.now()`** ile işaretlenir. Bu **timestamp bir "fate" bayrağıdır** (bizim `fate` alanının muadili).
Serialize sırasında `compacted` işaretli çıktılar `TOOL_OUTPUT_MAX_CHARS = 2000`'e iner.

**Adım 5 — Korunan tool:** `PRUNE_PROTECTED_TOOLS = ["skill"]` → skill tool çıktıları **asla budanmaz** (bizim korunan-tool mantığı).

**Adım 6 — Gerekirse LLM özeti:** turn'ler hâlâ büyükse `splitTurn` ile turn içinde bölme noktası bulunur, `summary.ts` + `compaction.txt` prompt'uyla özetlenir. `completedCompactions()` hangi turn'lerin zaten özetlendiğini izler.

**Overflow yolu:** [overflow.ts](../harnesses/opencode/packages/opencode/src/session/overflow.ts) `usable()` / `isOverflow()` ile pencere taşması tespit edilir. Medya çok büyükse: sıkıştırılır + **kaldırılır** ve modele açıklama enjekte edilir (*"attachments were too large... try again with smaller files"*).

### Öne çıkan
OpenCode = **bizim POC'un en yakın akrabası**: `DEFAULT_TAIL_TURNS=2` (=RECENT=2), `compacted` timestamp (=fate), `PRUNE_PROTECTED_TOOLS` (=korunan tool), `PRUNE_MINIMUM` (=fayda-freni). Üstüne spill-to-disk ve turn-bazlı LLM özeti.

---

## 3. Codex — ortadan-kesme truncation + compaction'ı bir "turn" olarak çalıştırma

**Kimlik/felsefe:** Codex iki mekanizma kullanır: (a) tool çıktısını **ortasından keserek** (head+tail koru) truncate etme, (b) tarih compaction'ını **tam bir model turn'ü** olarak çalıştırıp özetle değiştirme + **pencereli** (windowed) saklama.

Kaynak: [utils/output-truncation/src/lib.rs](../harnesses/codex/codex-rs/utils/output-truncation/src/lib.rs), [core/src/compact.rs](../harnesses/codex/codex-rs/core/src/compact.rs), [protocol/src/compacted_item.rs](../harnesses/codex/codex-rs/protocol/src/compacted_item.rs).

### Katman A — tool/exec çıktısı truncation (ortadan-kesme)
```rust
pub fn truncate_text(content: &str, policy: TruncationPolicy) -> String {
    match policy {
        TruncationPolicy::Bytes(n)  => truncate_middle_chars(content, n),
        TruncationPolicy::Tokens(n) => truncate_middle_with_token_budget(content, n).0,
    }
}
```
**Kilit fikir — ORTADAN kes:** Codex çıktının **başını ve sonunu korur, ortasını atar** (`truncate_middle`). Neden? Bir komut çıktısında en değerli kısımlar genelde **baş** (ne çalıştı) ve **son** (sonuç/exit) — orta genelde tekrarlı gövde.

**Bilgilendirici sarmalama:** `formatted_truncate_text` bir uyarı başlığı ekler:
```
Warning: truncated output (original token count: 12043)
Total output lines: 2000

<head>...[orta atlandı]...<tail>
```
→ Model kesildiğini **ve ne kadar** kesildiğini bilir.

**Multimodal koruma:** Sadece `InputText` segmentleri birleştirilip truncate edilir; `InputImage` / `InputAudio` / `EncryptedContent` **dokunulmaz**.

`TruncationPolicy` iki modda: byte-bütçesi veya token-bütçesi (`approx_token_count` ile).

### Katman B — tarih compaction'ı bir turn olarak (`compact.rs`)
**Adım 1 — Talimatı kaydet:** compaction talimatı `initial_input_for_turn` olarak history'ye **truncation_policy ile** kaydedilir (kayıt anında per-item truncation uygulanır).

**Adım 2 — Tek client session:** retry'lar boyunca korunur ("sticky routing, websocket incremental request tracking turn-scoped state survive etsin").

**Adım 3 — Model çağrısı döngüsü:** `drain_to_completed` ile model özeti üretir; `SessionBudgetExceeded` / `Interrupted` özel ele alınır, `stream_max_retries` kadar retry.

**Adım 4 — Cache-dostu trim:** *"trim from the beginning to preserve cache (prefix-based) and keep recent messages intact"* — prefix cache bozulmasın diye **baştan** kırpar, son mesajları korur.

**Adım 5 — Pencereli saklama (windowing):** `CompactedItem` → `window_number` + `first_window_id` + `previous_window_id`. Her compaction bir **pencere** yaratır; pencereler zincirlenir → eski rollout'lar **resume edilebilir** (legacy `window_id` şeklini bile kabul eder).

**Ekstra:**
- `DoNotInject` varyantı — pre-turn/manuel compaction özeti history'ye enjekte etmez (mid-turn ise eder).
- `compact_remote*.rs` — **sunucu-tarafı compaction** (v1/v2) + `compact_model_fallback`.
- `run_pre_compact_hooks` / `run_post_compact_hooks` — Pre/Post compact kancaları.
- Tetik: `run_inline_auto_compact_task` (otomatik) + manuel `run_compact_task`.

### Öne çıkan
Codex = **ortadan-kesme** (head+tail koru) + compaction'ı **turn olarak** çalıştırma + **pencereli/geri-sarılabilir** tarih + **sunucu-tarafı** compaction seçeneği. Windowing onu diğerlerinden ayırır: compaction "yok etme" değil, "geri sarılabilir katman" olarak saklanır.

---

## 4. Claude Code — auto-compaction + microcompaction (docs; kaynak kapalı)

**Kimlik/felsefe:** Claude Code'un CLI kaynağı yayınlanmaz (minify bundle); mekanizma resmî docs + çalışma-zamanı gözleminden. İki katmanlı.

**Adım 1 — Microcompaction (tool-trace düzeyi):** Büyük tool çıktıları **diske yazılıp referansa** indirilir. Bu oturumda birebir gördük: 93 KB'lık bir WebFetch sonucu diske yazıldı, context'te sadece "Full output saved to: …/tool-results/…txt" referansı kaldı + 2KB önizleme. → OpenCode spill-to-disk'in muadili.

**Adım 2 — Auto-compaction (konuşma düzeyi):** Token eşiğine gelince eski turn'ler bir **konuşma özetine** indirgenir; döngü devam eder. (Bu belgeyi üreten oturumun başındaki "continued from previous conversation" özeti tam bu.)

**Adım 3 — Kancalar:** `PreCompact` (bloklanabilir) / `PostCompact` (bildirim) özet üretimini sarar. `/compact` manuel tetik.

**Adım 4 — İkinci katman (izolasyon):** Aynı pencerede *sıkıştırmak* yerine yan-işi **subagent'a taşımak** — Task tool ile ayrı context penceresi, sadece özet döner. Bu, compaction'a alternatif bir tool-trace yönetimidir.

### Öne çıkan
Claude Code = **microcompaction (tool çıktısı → disk referansı)** + **auto-compaction (konuşma özeti)** + **subagent izolasyonu**. Tool-trace'i ya diske taşır ya ayrı pencereye delege eder.

---

## 5. Beş sistem yan yana — tool-trace merceği

| Eksen | Hermes | OpenClaw | OpenCode | Codex | Claude Code |
|---|---|---|---|---|---|
| **Ana yöntem** | deterministik 3-pass prune | **worker-plan + chunk özet** | prune + spill + LLM özet | ortadan-kes + turn-özet | micro + auto compaction |
| **Tool çıktısı** | informatif 1-satır + dedup | özetleyiciden **detay şeritle** | 2000satır/50KB **disk spill** | **ortadan kes** (head+tail) | **disk referans** |
| **LLM özeti** | orta-turn (ayrı faz) | **evet (chunk'lar)** | evet (turn'ler) | evet (turn olarak) | evet (konuşma) |
| **Çift bütünlüğü** | id koru, silme | **grup-bölme yasağı** | completed-part | truncation kaydı | (docs) |
| **Yakın koruma** | tail-token + floor 8 | budget %50 | **TAIL_TURNS=2** | recent+cache | (belirsiz) |
| **Ayırt edici** | anti-thrash + micro-compact | **worker + güvenlik-şerit + oversized-note** | POC'a en yakın sabitler | **windowing + uzak/sunucu** | **subagent'a taşıma** |
| **Nerede çalışır** | ana + kilitli | **ayrı worker-thread** | ana | ana + sunucu | ana + hook |

---

## 6. POC eşlemesi — hangi fikrimiz kimde var

| Bizim POC (`poc/`) | Kimde birebir var |
|---|---|
| `fate` işaretleme (tool gövdesi yeniden yazma) | OpenCode `part.state.time.compacted` · Hermes Pass 2 |
| `fate=DEDUP` | Hermes Pass 1 (byte-identik) |
| `fate=ÖZET` (informatif not) | Hermes `_summarize_tool_result` · Codex `formatted_truncate` uyarısı |
| Fayda-freni | Hermes #60451 · OpenCode `PRUNE_MINIMUM` |
| `RECENT=2` koruma | **OpenCode `DEFAULT_TAIL_TURNS=2`** · Hermes floor-8 |
| `tool_call_id` bütünlüğü | **OpenClaw grup-bölme yasağı** · hepsi |
| Referansa indirme (spill) | OpenCode disk-dir · Claude Code disk-referans · Hermes context_references |
| Ortadan-kesme (head+tail) | **sadece Codex** (bizde yok) |
| Worker-thread + güvenlik-şerit | **sadece OpenClaw** (bizde yok) |
| Windowing / geri-sarılabilir | **sadece Codex** (bizde yok) |

**Sonuç:** POC'umuzun çekirdek fikirleri (fate + dedup + informatif özet + RECENT koruma + id bütünlüğü + fayda-freni) bu adayların **hepsinde dağınık halde** var — en yoğun OpenCode ve Hermes'te. Bizde **olmayan** üç güçlü fikir: (1) Codex'in **ortadan-kesmesi** (head+tail koru) ve **windowing**'i, (2) OpenClaw'ın **worker-thread planı + güvenlik-şeritlemesi**, (3) hepsinin **spill-to-disk** referansı.

---

## Kaynaklar
- OpenClaw: `src/agents/compaction-planning.ts` · `compaction-planning.worker.ts` · `session-transcript-repair.ts` · `agent-compaction-constants.ts`
- OpenCode: `packages/opencode/src/tool/truncate.ts` · `session/compaction.ts` · `session/overflow.ts`
- Codex: `codex-rs/utils/output-truncation/src/lib.rs` · `core/src/compact.rs` · `protocol/src/compacted_item.rs`
- Claude Code: `code.claude.com/docs` + çalışma-zamanı gözlemi (microcompaction)
- Eş belge: [hermes-tool-trace-compaction.md](hermes-tool-trace-compaction.md) · [poc/](../poc/)
