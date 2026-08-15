# OpenClaw Sistemi — Teknik Mimari Analizi

> Birincil kaynak: OpenClaw v2026.7.1-2 resmi dokümantasyonu ve paket yapısı
> (`/home/altan/.npm-global/lib/node_modules/openclaw`). MIT lisanslı, açık kaynak.
> İkincil kaynak: Paolo Perazzo, "OpenClaw Architecture, Explained" (Products for Humans, Şub 2026) —
> anlatısal genel bakış; çakışan noktalarda birincil (paket-temelli) kaynak esas alınmıştır.
> Hazırlanma tarihi: 2026-08-14

---

## 0. Özet (TL;DR)

OpenClaw, **kendi donanımında (self-hosted) çalışan, çok kanallı bir AI-agent geçididir (gateway)**.
Tek bir uzun ömürlü **Gateway** süreci; Discord, Telegram, WhatsApp, Signal, Slack, iMessage,
Matrix, WebChat ve daha fazlasını tek noktadan yönetir; gelen her mesajı bir **agent** oturumuna
yönlendirir; agent, araç kullanımı (tool use), oturum belleği ve çok-agent yönlendirmesiyle yanıt üretir.

Mimarinin üç ana ekseni vardır:

1. **Gateway (kontrol düzlemi + taşıma):** Tek WebSocket sunucusu, tüm istemciler ve node'lar için.
2. **Agent Runtime (yürütme çekirdeği):** Mesajı → bağlam → model çıkarımı → araç yürütme → yanıt döngüsü.
3. **Plugin/Capability sistemi (genişletme):** Kanallar, model sağlayıcıları, araçlar, bellek/bağlam motorları.

Kod düzeni: TypeScript, pnpm monorepo. Protokol TypeBox şemalarıyla tanımlanır, JSON Schema ve Swift
modelleri bu şemalardan üretilir.

---

## 1. Yüksek Seviye Mimari

```
   Chat uygulamaları + kanal plugin'leri
                  │
                  ▼
           ┌─────────────┐
           │   GATEWAY    │  ◄── CLI
           │  (tek süreç) │  ◄── Web Control UI
           │  WS :18789   │  ◄── macOS uygulaması
           └─────────────┘  ◄── iOS / Android node'ları
                  │
                  ▼
          OpenClaw Agent Runtime
       (oturum, bellek, araçlar, model)
```

**Temel ilke:** Gateway; oturumlar, yönlendirme ve kanal bağlantıları için **tek doğruluk kaynağıdır
(single source of truth)**. Host başına yalnızca **bir** Gateway çalışır ve WhatsApp (Baileys) gibi tekil
oturumları yalnızca o açar.

### Bileşenler

| Bileşen | Rol |
|---|---|
| **Gateway (daemon)** | Sağlayıcı bağlantılarını tutar, tipli WS API sunar, gelen çerçeveleri JSON Schema'ya karşı doğrular, `agent`/`chat`/`presence`/`health`/`heartbeat`/`cron` olayları yayınlar. |
| **İstemciler (operator)** | CLI, Web UI, macOS uygulaması, otomasyonlar. Her biri tek WS bağlantısı açar; istek gönderir, olaylara abone olur. |
| **Node'lar** | macOS/iOS/Android/headless cihazlar. Aynı WS sunucusuna `role: node` ile bağlanır; kamera, ekran, konum, canvas gibi yetenekleri (caps/commands) sunar. |
| **WebChat** | Statik UI; sohbet geçmişi ve gönderim için Gateway WS API'sini kullanır. |
| **Canvas host** | Gateway HTTP sunucusunda `/__openclaw__/canvas/` ve `/__openclaw__/a2ui/` altında; agent'ın düzenleyebildiği HTML/CSS/JS yüzeyi. |

---

## 2. Çekirdek (Core) — Agent Runtime

OpenClaw yerleşik agent runtime'ının sahibidir; harici bir agent framework'üne bağımlı değildir.

### 2.1 Kod düzeni

| Yol | Sorumluluk |
|---|---|
| `src/agents/embedded-agent-runner/` | Yerleşik "attempt loop" (`run.ts`), model seçimi/sağlayıcı normalizasyonu, provider parametreleri, compaction, transcript & oturum bağlama. |
| `src/agents/sessions/` | Oturum kalıcılığı (`session-manager.ts`), kaynak keşfi, in-session extension yükleme, prompt şablonları, skills, temalar, TUI araç görselleştiricileri. |
| `packages/agent-core/` (`@openclaw/agent-core`) | Yeniden kullanılabilir agent çekirdeği: agent döngüsü, harness tipleri, mesajlar, compaction yardımcıları, prompt şablonları, oturum saklama sözleşmeleri. |
| `src/agents/runtime/` | `@openclaw/agent-core`'u plugin SDK LLM runtime'ına bağlayan OpenClaw cephesi (facade). |
| `src/agents/agent-tools*.ts` | OpenClaw'a ait araç tanımları, parametre şemaları, araç politikası, before/after tool-call adaptörleri, host/sandbox edit araçları. |
| `src/agents/agent-hooks/` | Yerleşik runtime hook'ları: compaction güvenlik önlemi, compaction talimatları, bağlam budama (context pruning). |
| `src/agents/harness/` | Harness kayıt defteri, seçim politikası ve yaşam döngüsü. |
| `src/llm/` | Model/sağlayıcı kayıt defteri, taşıma yardımcıları, sağlayıcıya özel akış (stream) implementasyonları (`src/llm/providers/`). |

**Sınırlar:** Çekirdek, yerleşik runtime'ı OpenClaw modülleri ve SDK "barrel" dosyaları üzerinden çağırır.
Plugin'ler yalnızca belgelenmiş `openclaw/plugin-sdk/*` giriş noktalarını kullanır, `src/**` iç bileşenlerini import etmez.

### 2.2 Runtime seçimi (Harness)

- Yerleşik runtime id'si: **`openclaw`**. Eski `pi` takma adı buna normalize olur; `codex-app-server` → `codex`.
- Plugin harness'ları ek runtime id'leri kaydeder (ör. `codex`).
- Politika model/sağlayıcı bazlı `agentRuntime.id` ile belirlenir (model girişi sağlayıcı girişini yener).
- `auto`: sağlayıcı/modeli destekleyen bir plugin harness varsa onu, yoksa yerleşik OpenClaw runtime'ını seçer.
- `openai` sağlayıcısı resmi API uç noktasında varsayılan olarak **`codex`** harness'ına gider; özel `baseUrl` kendi davranışını korur.

### 2.3 Agent döngüsü (Agent Loop)

Agent döngüsü, **oturum başına seri (serialized)** çalışan; bir mesajı eyleme ve yanıta çeviren süreçtir:
alım → bağlam kurulumu → model çıkarımı → araç yürütme → akış (streaming) → kalıcılık.

**Çalışma sırası:**

1. `agent` RPC parametreleri doğrular, oturumu çözer, meta veriyi kalıcı hale getirir ve **anında** `{ runId, acceptedAt }` döner.
2. `agentCommand` turu çalıştırır: model + thinking/verbose/trace varsayılanlarını çözer, skills anlık görüntüsünü yükler, `runEmbeddedAgent`'ı çağırır.
3. `runEmbeddedAgent`: çalıştırmaları oturum ve global kuyruklarla serileştirir, model + auth profilini çözer, oturumu kurar, runtime olaylarına abone olur, assistant/tool delta'larını akıtır, zaman aşımını uygular.
4. `subscribeEmbeddedAgentSession`: runtime olaylarını `agent` akışına köprüler — araç olayları → `tool`, assistant delta'ları → `assistant`, yaşam döngüsü → `lifecycle` (`start`/`end`/`error`).
5. `agent.wait`: bir `runId` üzerinde **lifecycle end/error** bekler; `{ status: ok|error|timeout, ... }` döner.

**Kuyruklama ve eşzamanlılık:**
Çalıştırmalar oturum anahtarı başına (session lane) ve isteğe bağlı global lane üzerinden serileştirilir; bu araç/oturum yarışlarını önler. Transcript yazımları ayrıca **süreç-farkında, dosya-tabanlı bir yazma kilidiyle** (session write lock) korunur (varsayılan bekleme `60000 ms`).

**Prompt kurulumu:** Sistem promptu; OpenClaw temel promptu + skills promptu + bootstrap bağlamı + tur bazlı override'lardan üretilir. Modele özgü limitler ve compaction rezerv token'ları uygulanır.

### 2.4 Hook sistemi (iki katman)

**Dahili (Gateway) hook'ları** — olay güdümlü scriptler:
- `agent:bootstrap`: sistem promptu sonlanmadan bootstrap dosyaları kurulurken çalışır.
- Komut hook'ları: `/new`, `/reset`, `/stop` vb.

**Plugin hook'ları** — agent/gateway boru hattı içinde çalışan uzatma noktaları:

| Hook | Ne zaman çalışır |
|---|---|
| `before_model_resolve` | Oturum öncesi; provider/model'i deterministik olarak override etmek için. |
| `before_prompt_build` | Oturum yüklendikten sonra; `prependContext`/`systemPrompt` vb. enjekte etmek için. |
| `before_agent_reply` | Inline eylemlerden sonra, LLM çağrısından önce; turu üstlenip sentetik yanıt döndürebilir veya susturabilir. |
| `agent_end` | Tamamlanma sonrası, nihai mesaj listesi + meta veriyle. |
| `before_compaction` / `after_compaction` | Compaction döngülerini gözlemler/işaretler. |
| `before_tool_call` / `after_tool_call` | Araç parametrelerini/sonuçlarını yakalar. |
| `tool_result_persist` | Araç sonuçlarını transcript'e yazılmadan önce senkron dönüştürür. |
| `message_received` / `message_sending` / `message_sent` | Gelen/giden mesaj hook'ları. |
| `session_start` / `session_end`, `gateway_start` / `gateway_stop` | Yaşam döngüsü sınırları. |

Karar kuralları: `before_tool_call` → `{ block: true }` terminaldir; `message_sending` → `{ cancel: true }` terminaldir.

### 2.5 Akış (Streaming) ve yanıt şekillendirme

- Assistant delta'ları `assistant` olarak akar; blok akışı `text_end`/`message_end`'de kısmi yanıt yayabilir.
- Araç start/update/end olayları `tool` akışında yayılır; sonuçlar boyut/görsel için sanitize edilir.
- Nihai yanıt; assistant metni (+ isteğe bağlı reasoning) + inline araç özetlerinden derlenir.
- **`NO_REPLY`** sessiz token'ı giden yükten filtrelenir; mesajlaşma aracı dublikeleri kaldırılır.

---

## 3. Gateway Protokolü (Kontrol Düzlemi)

Gateway WS protokolü, OpenClaw'un **tek kontrol düzlemi ve node taşıma katmanıdır**. Her istemci
(CLI, web UI, macOS, iOS/Android, headless) WebSocket üzerinden bağlanır ve el sıkışma anında bir
**rol (role)** ve **kapsam (scope)** bildirir. Güncel protokol sürümü **v4**'tür.

### 3.1 Taşıma ve çerçeveleme

- WebSocket, metin çerçeveleri, JSON yük.
- İlk çerçeve **mutlaka** `connect` isteği olmalıdır. Aksi (JSON olmayan / connect olmayan) → sert kapatma.
- Ön-kimlik (pre-auth) çerçeveleri **64 KiB** ile sınırlı; el sıkışma sonrası `hello-ok.policy.maxPayload` geçerli (varsayılan 25 MB).

**Çerçeve şekilleri:**
```
İstek : {type:"req",   id, method, params}
Yanıt : {type:"res",   id, ok, payload|error}
Olay  : {type:"event", event, payload, seq?, stateVersion?}
```

Yan etkili metotlar (`send`, `agent`) güvenli tekrar için **idempotency key** ister.

### 3.2 El sıkışma (Handshake)

1. Gateway ön-kimlik **challenge** yollar: `{event:"connect.challenge", payload:{nonce, ts}}`.
2. İstemci `connect` gönderir: `minProtocol`/`maxProtocol`, `client`, `role`, `scopes`, `caps`, `commands`, `permissions`, `auth`, ve imzalı `device` kimliği.
3. Gateway `hello-ok` döner: `protocol`, `server`, `features` (methods/events), `snapshot` (presence+health), `auth` (rol/scope, gerekiyorsa `deviceToken`), `policy` (payload/buffer/tick limitleri).

Tüm bağlantılar sunucunun verdiği `connect.challenge` nonce'unu **imzalamalıdır**. İmza yükü `v3`;
`platform` ve `deviceFamily`'yi de bağlar. Yeni cihaz kimlikleri **eşleştirme (pairing) onayı** gerektirir.

### 3.3 Roller ve kapsamlar (Scopes)

**Roller:** `operator` (kontrol düzlemi istemcisi), `node` (yetenek sunucusu — kamera/ekran/canvas/system.run).

**Operator scope kapalı kümesi:**
`operator.read`, `operator.write`, `operator.admin`, `operator.approvals`, `operator.pairing`, `operator.talk.secrets`.

Rezerve çekirdek metot önekleri her zaman `operator.admin`'e çözülür: `config.*`, `exec.approvals.*`, `wizard.*`, `update.*`.
Node eşleştirme onayı, bildirilen komutlara göre ek kapsam ister (ör. `system.run` içeriyorsa `operator.pairing + operator.admin`).

### 3.4 Kimlik doğrulama modları

`gateway.auth.mode`: `none` | `token` | `password` | `trusted-proxy`.
- Paylaşımlı sır: `connect.params.auth.token` veya `.password`.
- Kimlik taşıyan modlar (Tailscale Serve, non-loopback trusted-proxy) auth'u istek başlıklarından karşılar.
- `none`: paylaşımlı sır kontrolünü tümden kapatır — **yalnızca özel/güvenilir ingress'te**.
- Eşleştirme sonrası Gateway, role+scope'a bağlı **device token** verir (`hello-ok.auth.deviceToken`).

### 3.5 RPC metot aileleri (seçme)

- **Sistem/kimlik:** `health`, `status`, `system-presence`, `gateway.identity.get`, `diagnostics.stability`.
- **Model/kullanım:** `models.list`, `usage.status`, `usage.cost`, `sessions.usage`.
- **Kanallar:** `channels.status`, `channels.logout`, `web.login.start/wait`.
- **Oturum kontrolü:** `sessions.list/create/send/steer/abort/patch/reset/delete/compact`, `chat.history/send/abort/inject`.
- **Agent:** `agents.list/create/update/delete`, `agent.wait`, `agents.workspace.list/get`, `audit.list`, `tasks.list/get/cancel`.
- **Node:** `node.pair.*`, `node.list/describe/invoke`, `node.event`, `node.pending.pull/ack/enqueue/drain`.
- **Onaylar:** `exec.approval.*`, `plugin.approval.*`.
- **Otomasyon/araçlar:** `wake`, `cron.*`, `commands.list`, `skills.*`, `tools.catalog/effective/invoke`.
- **Talk/TTS:** `talk.*` (realtime, transcription, managed-room, telefoni), `tts.*`.
- **Sır/config/update:** `secrets.*`, `config.get/set/patch/apply/schema`, `update.run/status`, `wizard.*`.

**Broadcast kapsam kapılama:** Sohbet/agent/tool-result çerçeveleri en az `operator.read` ister;
kapsamsız oturumlar bu çerçeveleri atlar. Bilinmeyen olay aileleri varsayılan olarak fail-closed'dur.

### 3.6 Sürümleme ve kod üretimi

- Sürüm sabitleri `packages/gateway-protocol/src/version.ts` (`PROTOCOL_VERSION=4`, `MIN_NODE=3`).
- Node'lar N-1 (v3) uyumluluk penceresi kullanabilir.
- Şemalar TypeBox'tan üretilir: `pnpm protocol:gen`, `protocol:gen:swift`, `protocol:check`.
- Referans istemci: `packages/gateway-client/src/` (RPC timeout 30 s, reconnect backoff 1→30 s, tick 15–30 s).

---

## 4. Oturum Yönetimi (Sessions)

Gelen her mesaj kaynağına göre bir **oturuma** yönlendirilir; tüm oturum durumu Gateway'e aittir.

### 4.1 Yönlendirme

| Kaynak | Davranış |
|---|---|
| Direkt mesajlar (DM) | Varsayılan olarak paylaşımlı tek oturum |
| Grup sohbetleri | Grup başına izole |
| Odalar/kanallar | Oda başına izole |
| Cron işleri | Her çalıştırmada taze oturum |
| Webhook'lar | Hook başına izole |

**DM izolasyonu** (çok kullanıcılı kurulumlar için kritik) — `session.dmScope`:
`main` (hepsi tek) | `per-peer` | `per-channel-peer` (önerilen) | `per-account-channel-peer`.

### 4.2 Yaşam döngüsü ve durum

- **Günlük reset** (varsayılan): yapılandırılan yerel saatte (`atHour`, varsayılan 4) yeni oturum.
- **Idle reset:** `idleMinutes` hareketsizlik sonrası. Heartbeat/cron/exec sistem olayları oturumu canlı tutmaz.
- **Manuel:** `/new`, `/reset` (`/new <model>` modeli de değiştirir).

**Durum yeri:**
- Store: `~/.openclaw/agents/<agentId>/sessions/sessions.json`
- Transcript: `~/.openclaw/agents/<agentId>/sessions/<sessionId>.jsonl`

Ayrı yaşam döngüsü zaman damgaları: `sessionStartedAt` (günlük reset), `lastInteractionAt` (idle), `updatedAt`.
`session.maintenance` ile depolama sınırlanır (`pruneAfter: 30d`, `maxEntries: 500`).

---

## 5. Çok-Agent Yönlendirme (Multi-Agent)

Tek Gateway sürecinde birden çok **izole agent** çalışır; her biri kendi workspace, state dizini
(`agentDir`) ve oturum deposuna sahiptir. Gelen mesajlar **bindings** ile doğru agent'a yönlendirilir.

- **Agent** = tam persona kapsamı: workspace dosyaları, auth profilleri, model kayıt defteri, oturum deposu.
- **Binding** = bir kanal hesabını (Slack workspace, WhatsApp numarası) bir agent'a eşler.
- **accountId** = bir kanal hesabı örneği (ör. WhatsApp `personal` vs `biz`).

**Yollar:**

| Ne | Varsayılan |
|---|---|
| Config | `~/.openclaw/openclaw.json` (`OPENCLAW_CONFIG_PATH`) |
| State dizini | `~/.openclaw` (`OPENCLAW_STATE_DIR`) |
| Varsayılan agent workspace | `~/.openclaw/workspace` |
| Diğer agent workspace | `<stateDir>/workspace-<agentId>` |
| Agent dizini | `~/.openclaw/agents/<agentId>/agent` |
| Oturumlar | `~/.openclaw/agents/<agentId>/sessions` |

**Yönlendirme kuralları:** Bindings deterministiktir, **en spesifik kazanır** (exact peer → parent peer →
peer wildcard → guild+roles → guild → team → account → channel → default agent). Aynı kademede birden
çok eşleşme varsa config sırasında ilk olan kazanır. `AND` semantiği (birden çok match alanı hepsi eşleşmeli).

Uyarı: `agentDir` agent'lar arasında **asla** paylaşılmamalı (auth/oturum çakışması). Agent-to-agent
mesajlaşma varsayılan olarak **kapalıdır**, açıkça etkinleştirilip allowlist'lenmelidir.

**Session araçları (agent-to-agent koordinasyon):** Etkinleştirildiğinde bir agent, oturumlar arası
koordinasyon için şu araçları kullanır:

| Araç | İşlev |
|---|---|
| `sessions_list` | Aktif oturumları/subagent'ları keşfeder. |
| `sessions_send` | Başka bir oturuma mesaj yollar (ör. `announceStep: "ANNOUNCE_SKIP"` ile sessizce iş devri). |
| `sessions_history` | Başka bir oturumun transcript'ini çeker (bağlam paylaşımı). |
| `sessions_spawn` | İş devretmek için programatik olarak yeni (izole) oturum/subagent yaratır. |
| `sessions_yield` | Turu bitirip spawn edilen subagent'ların tamamlanmasını bekler. |

---

## 6. Bağlam Motoru (Context Engine) ve Compaction

### 6.1 Context Engine

Her model çağrısında bağlamı **hangi mesajlar, nasıl özetlenir, subagent sınırları arası nasıl yönetilir**
sorularını yanıtlayan katmandır. Varsayılan yerleşik motor: **`legacy`**. Eklenti motorlar (`plugins.slots.contextEngine`)
farklı toplama/compaction/hatırlama davranışı sağlar.

Dört yaşam döngüsü noktası: **Ingest** (mesaj eklendiğinde) → **Assemble** (her model çalıştırması öncesi,
token bütçesine sığan sıralı mesaj kümesi + isteğe bağlı `systemPromptAddition`) → **Compact** (pencere dolduğunda / `/compact`) → **After turn**.

`ownsCompaction: true` → motor kendi compaction'ını yönetir; OpenClaw'un yerleşik auto-compaction'ı devre dışı kalır.
**Hata izolasyonu:** Seçili plugin motoru yüklenemez/çökerse, OpenClaw onu o Gateway süreci için karantinaya alır
ve `legacy` motora düşer; agent susmaz.

### 6.2 Compaction

Her modelin bir bağlam penceresi vardır; limite yaklaşıldığında OpenClaw eski mesajları bir özete **compact** eder.

- Eski turlar özetlenir, özet transcript'e kaydedilir, güncel mesajlar korunur.
- Araç çağrıları eşleşen `toolResult` girişleriyle birlikte tutulur (bölme noktası bir araç bloğunun içine düşerse sınır kaydırılır).
- **Auto-compaction** varsayılan açık; limite yaklaşınca veya sağlayıcı overflow hatası dönünce (compact + retry) çalışır.
  OpenClaw düzinelerce sağlayıcıya özel overflow hata dizesini tanır (Anthropic, OpenAI, Bedrock, Gemini, Ollama...).
- **Memory flush:** Compaction öncesi OpenClaw, önemli notları diske yazmak için sessiz bir tur çalıştırabilir (bağlam kaybını önler).
- Farklı model (`compaction.model`), identifier koruma (`identifierPolicy: strict`), byte guard, successor transcript gibi ince ayarlar mevcuttur.

**Compaction vs Pruning:** Compaction sohbeti özetler (kalıcı); pruning yalnızca eski araç sonuçlarını kırpar (bellekte, istek başına).

---

## 7. Bellek (Memory)

OpenClaw hatırlamayı **workspace'te düz Markdown dosyaları yazarak** yapar; gizli durum yoktur.

- **`MEMORY.md`** — uzun vadeli, küratörlü bellek. Oturum başında yüklenir.
- **`memory/YYYY-MM-DD.md`** — günlük notlar. `memory_search`/`memory_get` için indekslenir; her turda prompt'a enjekte edilmez.
- **`DREAMS.md`** (opsiyonel) — Dream Diary ve "dreaming" özeti.

**Araçlar:** `memory_search` (hibrit: vektör benzerliği + anahtar kelime), `memory_get` (belirli dosya/satır aralığı).
Aktif bellek plugin'i tarafından sağlanır (varsayılan: `memory-core`).

**Backend'ler:** Builtin (SQLite, varsayılan) · QMD (local-first sidecar, reranking) · Honcho (AI-native cross-session) · LanceDB (plugin).
Varsayılan embedding sağlayıcısı OpenAI'dir; Gemini/Voyage/Mistral/Bedrock/Ollama/LM Studio vb. seçilebilir.

**Dreaming:** Opsiyonel arka plan konsolidasyonu — kısa vadeli hatırlama sinyallerini toplar, puanlar, yalnızca
eşiği geçen öğeleri `MEMORY.md`'ye terfi ettirir. `memory-wiki` plugin'i ise kanıt-zengin bir bilgi kasası (knowledge vault) katmanı ekler.

---

## 8. Plugin / Capability Sistemi (Genişletme)

Genişletmenin kalbi **capability (yetenek) modelidir**. Her native plugin bir veya daha çok yetenek türü kaydeder.

### 8.1 Yetenek türleri (seçme)

| Yetenek | Kayıt metodu | Örnek |
|---|---|---|
| Text inference | `api.registerProvider(...)` | anthropic, openai |
| CLI inference backend | `api.registerCliBackend(...)` | anthropic, openai |
| Embeddings | `api.registerEmbeddingProvider(...)` | vektör plugin'leri |
| Speech / Realtime voice / Transcription | `registerSpeechProvider` / `registerRealtimeVoiceProvider` / `registerRealtimeTranscriptionProvider` | elevenlabs, openai, google |
| Media understanding | `api.registerMediaUnderstandingProvider(...)` | google, openai |
| Image/Music/Video generation | `registerImage/Music/VideoGenerationProvider` | fal, google, minimax |
| Web fetch / Web search | `registerWebFetchProvider` / `registerWebSearchProvider` | firecrawl, brave, google |
| Channel / messaging | `api.registerChannel(...)` | matrix, msteams |
| Gateway discovery | `api.registerGatewayDiscoveryService(...)` | bonjour |

**Plugin şekilleri:** `plain-capability` (tek yetenek), `hybrid-capability` (çok yetenek — ör. openai text+speech+media+image),
`hook-only` (yalnızca hook, hâlâ desteklenir), `non-capability` (araç/komut/servis/route ama yetenek yok).

### 8.2 Anahtar tasarım ayrımı

- **plugin = sahiplik sınırı** (bir şirketin/özelliğin tüm yüzeyi tek plugin'de).
- **capability = çekirdek sözleşme** (birden çok plugin uygular veya tüketir).

Yeni bir alan eklenirken ilk soru "hangi sağlayıcı bunu hardcode etsin?" değil, **"çekirdek yetenek sözleşmesi nedir?"** olmalıdır.
Katmanlama: Core capability (orkestrasyon/politika/fallback/tipli sözleşme) → Vendor plugin (vendor API/auth/katalog) → Channel/feature plugin (yüzeyde sunum).

### 8.3 Yükleme boru hattı ve yürütme modeli

Dört katman: **Manifest+keşif** (`openclaw.plugin.json` + bundle manifestleri) → **Etkinleştirme+doğrulama** →
**Runtime yükleme** (native plugin'ler in-process yüklenir, merkezi registry'ye kaydolur) → **Yüzey tüketimi** (araç/kanal/hook/route/CLI/servis).

> **Güvenlik uyarısı:** Native plugin'ler Gateway ile **aynı süreçte, sandbox'sız** çalışır. Yüklü bir native plugin
> çekirdek kod ile aynı süreç-seviyesi güven sınırına sahiptir; kötü niyetli bir native plugin, OpenClaw süreci içinde
> keyfi kod yürütmeye eşdeğerdir. Bundle'lar (çoğunlukla skills) metadata/içerik paketi olarak daha güvenlidir.
> Non-bundle plugin'ler için allowlist ve açık yükleme yolları kullanın.

Startup'ta tek `PluginMetadataSnapshot` kurulur (yalnızca metadata; yüklü modül/SDK içermez); tekrarlı kararlar bu snapshot ile hızlı yolda kalır.

---

## 9. Node'lar (Mobil / Cihaz Yetenekleri)

Node'lar aynı WS sunucusuna `role: node` ile bağlanır ve bağlantı anında yetenek iddialarını bildirir:

- **caps:** yüksek seviye kategoriler — `camera`, `canvas`, `screen`, `location`, `voice`, `talk`.
- **commands:** invoke için komut allowlist'i (`camera.snap`, `canvas.navigate`, `screen.record`, `location.get`).
- **permissions:** granüler anahtarlar (`camera.capture`, `screen.record`).

Gateway bunları **iddia** olarak alır ve sunucu tarafı allowlist'lerle zorlar. Eşleştirme cihaz-tabanlıdır;
onay device pairing store'da yaşar. Node'lar arka planda `node.presence.alive` olayıyla "canlıyım" bildirebilir
(tetikleyiciler: `background`, `silent_push`, `bg_app_refresh`, `significant_location`, `manual`, `connect`).

Offline node'lar için dayanıklı iş kuyruğu: `node.pending.enqueue/drain`; bağlı node'lar için `node.pending.pull/ack`.

---

## 10. Güvenlik ve Sandbox

### 10.1 Sandboxing

Araç yürütmesini bir sandbox backend'inde çalıştırarak "blast radius"ı azaltır. **Varsayılan kapalı.**
Gateway süreci **her zaman** host'ta kalır; yalnızca araç yürütmesi (enable edilince) sandbox'a taşınır.

Üç bağımsız ayar:

| Ayar | Anahtar | Değerler | Varsayılan |
|---|---|---|---|
| Mode | `sandbox.mode` | `off`, `non-main`, `all` | `off` |
| Scope | `sandbox.scope` | `agent`, `session`, `shared` | `agent` |
| Backend | `sandbox.backend` | `docker`, `ssh`, `openshell` | `docker` |

**Docker backend** varsayılan: `network: "none"` (egress yok), `readOnlyRoot: true`, `capDrop: ["ALL"]`, imaj `openclaw-sandbox:bookworm-slim`.
**Workspace erişimi:** `none` (izole) / `ro` (`/agent` salt-okunur) / `rw` (`/workspace` okuma-yazma).
Bind mount'larda tehlikeli kaynaklar (`/etc`, `/proc`, `/root`, docker soketleri, `~/.ssh`, `~/.aws`...) varsayılan bloklanır.

### 10.2 Katmanlı savunma

- **Tool policy** (allow/deny) sandbox kurallarından **önce** uygulanır.
- **`tools.elevated`** açık bir kaçış: `exec`'i sandbox dışında çalıştırır (global + per-agent kapı, ikisi de izin vermeli).
- **Exec approvals:** Bir exec onay gerektirdiğinde Gateway `exec.approval.requested` yayınlar; operatör `exec.approval.resolve` ile çözer.
  `host=node` için kanonik `systemRunPlan` zorunludur; prepare ile approve arasında komut/cwd değiştirilirse çalıştırma reddedilir.
- **Audit ledger:** `audit.list` — yalnızca metadata (prompt/mesaj/araç argümanı/çıktı **saklanmaz**); 30 gün, 100.000 kayıt sınırı.

---

## 11. Kalıcı Durum (State) Yerleşimi

State, OpenClaw state dizininde yaşar (`~/.openclaw` varsayılan, `OPENCLAW_STATE_DIR` ile override):

| Yol | İçerik |
|---|---|
| `openclaw.json` | Config |
| `state/openclaw.sqlite` | Paylaşımlı runtime state veritabanı |
| `agents/<agentId>/agent/openclaw-agent.sqlite` | Per-agent model auth profilleri (API key + OAuth) ve runtime state |
| `credentials/` | Auth profil deposu dışındaki sağlayıcı/kanal kimlik bilgileri |
| `agents/<agentId>/sessions/` | Oturum transcript'leri + `sessions.json` indeksi |
| `workspace/` | Varsayılan agent workspace |

Legacy `auth-profiles.json` artık runtime'da okunmaz; `openclaw doctor --fix` bunları SQLite deposuna aktarır.

---

## 12. Kontrol Arayüzleri (Operatör)

Gateway ile etkileşimin dört yolu vardır; hepsi aynı tipli WS API'sine bağlanır:

| Arayüz | Konum / giriş | Özet |
|---|---|---|
| **Web Control UI** | Gateway'in kendisi sunar; varsayılan `http://127.0.0.1:18789/` | Lit tabanlı web bileşenleri. Sohbet, config yönetimi, oturum denetimi, node yönetimi, health. Ayrı web sunucusu gerekmez. |
| **CLI** | `openclaw.mjs` → `src/cli/` (Commander.js) | `openclaw gateway` (başlat), `openclaw onboard` (rehberli kurulum), `openclaw channels login` (WhatsApp/Signal eşleştirme), `openclaw message send`, `openclaw pairing approve`, `openclaw doctor`. |
| **macOS uygulaması** | `apps/macos/` (Swift) | Menü çubuğu: Gateway başlat/durdur/yeniden başlat, WebChat gömülü WebKit görünümü, Voice Wake push-to-talk overlay, uzak Gateway'i SSH üzerinden kontrol. |
| **Mobil node'lar** | iOS / Android | `role: "node"` ile bağlanır; kamera, ekran kaydı, konum, Canvas gibi cihaz yeteneklerini `node.invoke` ile sunar (bkz. §9). |

---

## 13. Etkileşim Yüzeyleri: Canvas / A2UI ve Ses

### 13.1 Canvas ve A2UI (Agent-to-UI)

Canvas, agent'ın düzenleyebildiği görsel bir çalışma yüzeyidir. **Ayrı bir süreç/port değildir**;
Gateway HTTP sunucusunda `/__openclaw__/canvas/` ve `/__openclaw__/a2ui/` altında sunulur ve loopback
dışına çıktığında Gateway auth'u ile korunur.

**A2UI döngüsü** — agent, JavaScript yazmadan etkileşimli arayüz üretir; özel `a2ui-*` öznitelikleri kullanır:

```html
<div a2ui-component="task-list">
  <button a2ui-action="complete" a2ui-param-id="123">Tamamla</button>
</div>
```

Akış: agent canvas günceller → Canvas host HTML + `a2ui-*` özniteliklerini ayrıştırır → içeriği WS
üzerinden bağlı istemcilere iter → istemci render eder. Kullanıcı butona bastığında istemci bir action
olayı gönderir; host bunu **agent'a bir tool-call** olarak iletir; agent durumu günceller ve yeni canvas'ı
iter, ekran otomatik tazelenir. Render: macOS/iOS native WebKit, Android WebView, web UI'da sekme.

### 13.2 Voice Wake ve Talk Mode

- **Voice Wake** (macOS/iOS/Android): her zaman açık uyandırma sözcüğü ("Hey OpenClaw") veya push-to-talk.
  Ses transkripsiyon sağlayıcısına akar, agent işler, yanıt TTS ile geri oynatılır (ör. ElevenLabs).
- **Talk Mode:** kesintisiz, eller serbest diyalog; agent konuşurken araya girme (interruption) tespiti.
  Özelleştirilebilir uyandırma sözcükleri. Protokolde `talk.*` metot ailesiyle yönetilir (bkz. §3.5).

---

## 14. Uçtan Uca Mesaj Akışı (WhatsApp örneği)

§2.3'teki agent döngüsünün kanal perspektifinden altı fazlı görünümü:

| Faz | Ne olur |
|---|---|
| **1. Alım** | Baileys, WhatsApp'tan WS olayı alır; `src/whatsapp/` adaptörü metin/medya/gönderen meta verisini çıkarır. |
| **2. Erişim kontrolü + yönlendirme** | Allowlist / DM pairing kontrolü. Geçerse oturum çözülür: operatör → `agent:main`, DM → `agent:main:whatsapp:dm:+…`, grup → `agent:main:whatsapp:group:…@g.us`. |
| **3. Bağlam kurulumu** | Runtime oturumu diskten yükler; `AGENTS.md`/`SOUL.md`/`TOOLS.md`'yi okur, ilgili skill'leri enjekte eder, `memory_search` ile anlamsal olarak benzer geçmişi çeker. |
| **4. Model çağrısı** | Derlenen bağlam yapılandırılmış sağlayıcıya (Anthropic/OpenAI/Gemini/yerel) token-by-token akıtılır. |
| **5. Araç yürütme** | Model tool-call isterse runtime araya girer; non-main oturumsa Docker sandbox'ta çalıştırır; tarayıcı istenirse CDP ile Chromium sürer. Sonuç akışa geri beslenir. |
| **6. Yanıt teslimi** | Yanıt parçaları Gateway'den geçer; WhatsApp adaptörü markdown'ı dönüştürüp boyut limitine böler, Baileys ile gönderir. Runtime tüm turu JSONL transcript'e kalıcılaştırır. |

> Yaklaşık gecikme bütçesi (ikincil kaynak, ortam bağımlı): erişim kontrolü <10 ms · oturum yükleme
> <50 ms · sistem promptu kurulumu <100 ms · ilk token 200–500 ms · bash aracı <100 ms · tarayıcı 1–3 s.

---

## 15. Dağıtım Mimarileri

Mimari tüm desenlerde aynıdır; değişen tek şey **Gateway'in nerede koştuğu** ve istemcilerin nasıl bağlandığıdır.
Güvenli varsayılan her yerde aynı: Gateway loopback'te kalır, uzak erişim SSH tüneli veya Tailscale ile sağlanır.

| Desen | Kurulum | Erişim |
|---|---|---|
| **Yerel geliştirme** (macOS/Linux) | `pnpm dev` (hot reload) veya `openclaw gateway`; `127.0.0.1:18789`. | Yalnızca localhost; auth gerekmez (loopback güvenilir kabul edilir). |
| **Üretim macOS** (menü çubuğu) | LaunchAgent arka plan servisi, login'de otomatik başlar; menü çubuğu uygulaması yaşam döngüsünü yönetir. iMessage + Voice Wake burada mümkün. | Yerel loopback; uzak için SSH/Tailscale. |
| **Linux/VPS** (uzak Gateway) | `systemd` servisi, loopback'te bağlı. | **A) SSH tüneli (önerilen):** `ssh -N -L 18789:127.0.0.1:18789 user@vps`. **B) Tailscale Serve:** tailnet-only HTTPS, Gateway loopback'te kalır. |
| **Fly.io** (konteyner) | Docker imajı Fly Machine'de; kalıcı volume `OPENCLAW_STATE_DIR=/data`; süreç `gateway --bind lan --port 3000`. Fly yönetilen HTTPS + TLS sonlandırma sağlar; sırlar `fly secrets`. | Public internet — **güçlü auth zorunlu** (token/password); sertleştirilmiş `deploy/fly.private.toml` varyantı public IP'siz. |

**Tailscale modları** (`gateway.tailscale.mode`): `serve` (tailnet-only HTTPS, loopback korunur) · `funnel`
(public HTTPS, **paylaşımlı parola şart**) · `off` (varsayılan). `lan`/`tailnet`'e bind edildiğinde Gateway
paylaşımlı sır (`gateway.auth.token`/`.password`) ister — auth bir trusted-proxy'ye devredilmediyse.

---

## 16. Teknoloji Yığını Özeti

| Katman | Teknoloji |
|---|---|
| Dil / paket | TypeScript, pnpm monorepo (`pnpm-workspace.yaml`) |
| Runtime | Node.js 24.15+ önerilir (22 LTS / 25.9+ uyumlu) |
| Protokol tanımı | TypeBox → JSON Schema → Swift model üretimi |
| Taşıma | WebSocket (kontrol düzlemi + node), HTTP (canvas/A2UI host) |
| Depolama | SQLite (runtime state, auth, audit), JSONL (transcript), Markdown (memory) |
| Kanal kütüphaneleri | WhatsApp→Baileys, Telegram→grammY, ayrıca Slack/Discord/Signal/iMessage vb. |
| TUI | `@earendil-works/pi-tui` (üçüncü taraf terminal bileşen kiti) |
| Lisans | MIT |

---

## 17. Kaynaklar

- Yerel dokümanlar: `/home/altan/.npm-global/lib/node_modules/openclaw/docs`
- Ayna: https://docs.openclaw.ai
- Kaynak kod: https://github.com/openclaw/openclaw
- Özellikle: `agent-runtime-architecture.md`, `gateway/protocol.md`, `concepts/agent-loop.md`,
  `concepts/architecture.md`, `plugins/architecture.md`, `concepts/multi-agent.md`, `concepts/session.md`,
  `concepts/context-engine.md`, `concepts/compaction.md`, `concepts/memory.md`, `gateway/sandboxing.md`,
  `network.md`, `vps.md`, `gateway/tailscale.md`, `install/fly.md`
- İkincil: Paolo Perazzo, "OpenClaw Architecture, Explained",
  https://ppaolo.substack.com/p/openclaw-system-architecture-overview (Products for Humans, Şub 2026).
