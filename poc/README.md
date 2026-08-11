# POC — Tool-Trace Compaction: tek chat, 13 mimari

Her rakip sistemin **tool-trace compaction** mantığını, kendi orijinal repo'suna
**birebir sadık** (gerçek fonksiyon adları, sabitleri, placeholder string'leri) bir
strateji olarak koştururuz. Tek bir chat mekanizması; her mesajda **seçili** sistemin
mantığı **aynı** trace üzerinde çalışır. Böylece "aynı hammadde → farklı sıkıştırma"
yan yana görülür.

> Kapsam: **yalnız tool-trace** (tool çıktıları ve çağrılar-arası ilişki). Genel
> konuşma-özeti (reason/context compaction) bu POC'un işi değil.
> Kaynak manzara: [`../report/13-ek-landscape-tam.md`](../report/13-ek-landscape-tam.md).

## Çalıştırma

### 🖥️ Görsel demo (önerilen — tıpkı önceki chat demosu gibi)

```bash
cd poc
../.venv/bin/python demo_server.py   # → http://127.0.0.1:8077  (motor: LangGraph)
# veya bağımlısız/manuel motor:  python demo_server.py
```

Tarayıcıda: **sistem seçicisinden** bir mimari seç → sohbet et → sağdaki panel o sistemin
tool-trace compaction'ını **renk-kodlu kader**lerle çizer (DEDUP/SİL/ÖZET/KES/MASKE/...).
**Sistemi değiştirdiğinde AYNI trace farklı compaction ile anında yeniden çizilir** — demonun
özü bu. "13 sistemi karşılaştır" ile hepsini yan yana gör. Mod: **mock** (offline, anında)
veya **canlı** (internal gemma, native tool-use). Örnek çipleri her mekanizmayı tetikler.

### ⌨️ CLI

```bash
../.venv/bin/python chat.py                 # canlı LLM tool-use · motor: LangGraph (varsayılan)
../.venv/bin/python chat.py --strategy cline
../.venv/bin/python chat.py --product       # GERÇEK 119 ürün tool'u · motor: manuel loop
python chat.py --mock                       # deterministik scripted brain (ağ gerekmez, stdlib)
python compare.py                           # kanonik senaryo → 13 sistem tasarruf tablosu
python test_poc.py                          # 191 güvence kontrolü
../.venv/bin/python langgraph_agent.py      # LangGraph standalone demo
```

### Ajan motoru (canlı)

`engines.make_live_agent` seçer:
- **generic tool seti → LangGraph** (`create_react_agent`, VARSAYILAN). `../.venv/bin/python` ile
  çalıştır (langgraph+langchain-openai orada).
- **product tool seti (119) → manuel loop** (`agent.py`) — 119 tool'u LangChain'e çevirmek yerine
  kanıtlı manuel döngü.
- **LangGraph yoksa** (ör. sistem `python3`'ü) → otomatik **manuel loop** yedeği.

Her iki motor da AYNI `strategies/` katmanını kullanır; ajan modu/motoru banner'da ve demo KPI'da yazar.

### İki tool seti

- **generic** (6 mock): terminal/read/web/snapshot/grep/write — offline (mock) veya canlı.
  Tip-çeşitli olduğu için her sistemin tip-özel yolunu tetikler.
- **product** (gerçek 119): `poc-trace-compaction/product_tools.py` — Jira/NETA/LDAP/Confluence/
  docx/pdf/pptx/xlsx/analiz. `TOOL_META` (`{cat, resource, ttl, verbatim}`) doğrudan bizim ledger
  sözleşmemiz; `tools_product.py` onu ToolResult'a çevirir → **13 strateji değişmeden** gerçek
  tool'lar üzerinde çalışır. Sadece **canlı** LLM (gemma tool'ları kendi seçer). Demo'da "tool seti"
  menüsünden, CLI'da `--product` ile.

**Kanıt (canlı gemma + gerçek tool'lar):**
- `confluence_get_page(12345)` iki kez → **ours DEDUP** (122→11 tok, %45).
- `docx_get_outline(D1)` → `docx_add_diagram(D1)` → `docx_get_outline(D1)`: **ours ilk okumayı
  SİL `is_stale (mutasyon)`** (write bayatlattı), **cline aynı okumayı DEDUP** (write'ı göremez).
  Bizim tek ayırt eden: stale-by-mutation ≠ salt tekrar. + **fayda freni** küçük çıktıyı sıkıştırıp
  zarar vermeyi engeller (Cline'da yok).

Chat komutları: `/strategy <isim>` · `/list` · `/trace` (kader) · `/view` (modelin
gördüğü) · `/compare` (canlı trace'i hepsinden geçir) · `/budget <n>` · `/reset` · `/quit`.

**İki mod:**
- **Canlı** (`chat.py`, anahtar varsa varsayılan): gerçek LLM (internal gemma, OpenAI-uyumlu
  `.env`) **native tool-calling** ile tool'ları kendi seçer. `agent.py` manuel tool-use
  döngüsü — framework yok. Tool sonuçları **seçili compaction stratejisinden geçmiş** hâliyle
  modele geri döner (messages[] köprüsü).
- **Mock** (`--mock`, `compare.py`, `test_poc.py`): LLM yok, `ScriptedBrain` deterministik plan.
  Tekrarlanabilir; karşılaştırma tabloları için ideal. Sadece stdlib.

LLM-tabanlı **strateji adımları** (QM, Codex handoff, OpenHands LLMSummarizing, OpenCode adım-2,
Cline/gemini son çare) offline modda deterministik extractive özetle taklit; canlı modda
`LLM_LIVE=1` ile gerçek endpoint'e gider (`strategies/base.py`).

## Mimari

```
harness.py            ortak model: ToolResult/Turn/Conversation (messages[] köprüsüyle birebir),
                      token tahmini, tip-çeşitli mock tool'lar, ScriptedBrain, ChatSession (mock mod)
llm.py                ince OpenAI-uyumlu istemci (urllib; SDK yok) — internal gemma, .env
tool_schemas.py       6 mock tool'un OpenAI function-calling şeması + çalıştırıcı
agent.py              GERÇEK tool-use ajanı (canlı mod): manuel döngü + compaction köprüsü
strategies/base.py    Strategy sözleşmesi + Fate etiketleri + (offline/canlı) özetleyici
strategies/<sistem>.py her sistem, kendi repo'suna sadık
demo_server.py        görsel demo backend (stdlib http.server) — gerçek stratejileri koşturur
demo.html             görsel demo arayüzü (self-contained; strateji seçici + trace paneli)
langgraph_agent.py    (opsiyonel) LangGraph adaptörü: strateji katmanı değişmeden pre_model_hook
chat.py               tek chat, canlı/mock otomatik, /strategy ile mimari seç
compare.py            kanonik senaryoyu hepsinden geçir (mock)
test_poc.py           güvence
```

### Canlı ajan — compaction köprüsü (kilit)

`agent.py` iki liste tutar: **ham `messages[]`** (yapı kaynağı — `tool_call_id` eşleşmesi) ve
**`conv`** (içerik + kader). Her LLM çağrısından önce `_render_messages` her tool mesajının
**gövdesini** kaderine göre (özet/maske/gizle/sil) yeniden yazar, preamble'ı system olarak başa
koyar — ama **`tool_call_id` yapısını asla bozmaz** (yoksa OpenAI-uyumlu API 400 verir). Böylece
"trace compaction" gerçekten **modelin gördüğü bağlamı** küçültür. Kanıt (canlı gemma, tek tur):
model `src/x`'i iki kez okur → `ours` ilk okumayı **DEDUP** (587→13 tok) → model sıkışmış bağlamı
görür ama yine doğru yanıtı üretir (%34–48 tasarruf).

**Sözleşme (tek yönlü, yerinde):** strateji her `ToolResult`'ın `.fate/.view/.note`
alanını doldurur — `call_id` / mesaj yapısı **asla** bozulmaz (tool_call↔tool_result
eşleşmesi korunur, yoksa gerçek API 400 verir), sadece **gövde** küçülür. İsteğe bağlı
bir `preamble` döner (ör. QM `contextSummaryPayload`). Modelin gördüğü bağlam =
`preamble + Σ result.shown()`.

**Neden mock tool + scripted brain?** Amaç LLM'in tool *seçimini* değil,
*compaction'ı* karşılaştırmak. Tool'lar tip-çeşitli ve gerçekçe-büyük çıktı verir; brain
kullanıcı cümlesini deterministik bir tool planına çevirir — böylece her sistemin
**tip-özel** yolu gerçek bir sohbette tetiklenir (aynı dosyayı iki kez oku → dedup;
snapshot'ı iki kez al → supersede; düzenle sonra oku → staleness).

## POC ↔ repo eşlemesi (birebir semboller)

| # | strateji | repo | § | tetiklediği gerçek semboller | kader |
|---|----------|------|---|------------------------------|-------|
| 1 | `hermes` | NousResearch/hermes-agent | §1.1 | `_summarize_tool_result`, `_PRUNED_TOOL_PLACEHOLDER` | ÖZET (tip-farkında tek satır) |
| 2 | `headroom` | headroomlabs-ai/headroom | §1.2 | `ContentRouter`, `CodeAwareCompressor`, `LogCompressor`, `SearchCompressor`, `KompressCompressor`, CCR `<<ccr:HASH>>` + `headroom_retrieve` | CRUSH (algoritmik + geri-çağrılabilir) |
| 3 | `codex` | openai/codex | §1.3 | `tool_output_token_limit`, `truncate_middle_chars`; compaction→handoff | KES + SİL |
| 4 | `claude-code` | anthropic/claude-code | §1.4 | `[Old tool result content cleared]`, keep-active, state reconstruction | SİL (placeholder) |
| 5 | `openclaw` | openclaw/openclaw | §1.5 | `sessionLikelyHasOversizedToolResults`, `resolveLiveToolResultMaxChars`, `truncateOversizedToolResultsInActiveTarget` | KES (sadece oversized) |
| 6 | `openhands` | OpenHands/software-agent-sdk | §1.6 | `Pipeline`, `BrowserOutputCondenser`, `ObservationMaskingCondenser(attention_window)`, `LLMSummarizingCondenser` | MASKE |
| 7 | `gemini-cli` | google-gemini/gemini-cli | §1.7 | `onBeforeTurn`, `supersedeStaleSnapshots`, `SNAPSHOT_SUPERSEDED_PLACEHOLDER`, `tryCompressChat`, `COMPRESSION_FAILED_INFLATED_TOKEN_COUNT` | SUPERSEDE (bayat snapshot) |
| 8 | `roo` | RooCodeInc/Roo-Code | §1.8 | `generateFoldedFileContext`, `parseSourceCodeDefinitionsForFile`, `truncateConversation(fracToRemove=0.5)`, `injectSyntheticToolResults` | KATLA (+GİZLE fallback) |
| 9 | `opencode` | sst/opencode | §1.9 | `SessionTime.Compacting`, `SessionCompacted`; son 2 user turn koru | GİZLE (depoda tut, geri alınabilir) |
| 10 | `cline` | cline/cline | §1.10 | `duplicateFileReadNotice`, `getNextTruncationRange(quarter\|half)`, `summarize_task` | DEDUP + SİL |
| 11 | `swe-agent` | SWE-agent/SWE-agent | §1.11 | `LastNObservations(n)`, `"Old environment output: (n lines omitted)"`, `history_processors` | SİL (konum-tabanlı) |
| 12 | `qm` | yc-software/qm | §2 | `contextSummaryPayload`, çift-tavan (400 girdi / 120K token) — **tool-trace-farkında DEĞİL** | ÖZET (bütün-geçmiş, karşıt örnek) |
| 13 | `ours` | adapted/hybrid-compaction | §3 | `ledger.record`, `is_stale` (mutasyon/TTL), `_detect_duplicate`, `tool_gist`, fayda freni | DEDUP + SİL + ÖZET (ilişki-farkında) |

Yeşil (saf DET): hermes, headroom, claude-code, roo, swe-agent, ours.
Mor (son çare/adım LLM): codex, openclaw, openhands, gemini-cli, opencode, cline, qm.

## Ayırt edici sonuç

`compare.py --detail`, aynı trace'te `src/server.py`'nin iki okuması + arada bir
`write_file`'ı işler. Yalnız **`ours`** iki okumayı **farklı** etiketler:

- ilk okuma → **SİL** `is_stale (mutasyon)` — sonraki write kaynağı bayatlattı,
- ikinci (aynı sürüm) tekrar → **DEDUP** `_detect_duplicate`.

Cline ikisini de `duplicateFileReadNotice` ile **DEDUP** sayar (sürüm/mutasyon ayrımı
yok); diğerleri ilişkiyi hiç görmez (boyut/konum/tip). Manzaranın tezi budur:
tool-trace-farkında sistemlerin çoğu **tekil çıktıyı** küçültür (output compaction);
**çağrılar-arası ilişkiyi** (dedup/bayat/sürüm) genel ve sürüm-farkında biçimde
gören tek sistem bizimki (Cline dosya-özel, sürümsüz bir istisna).

## Tarayıcı paneli — her mantık ayrı sekmede (`web_server.py`, :8010)

```bash
.venv/bin/python poc/web_server.py     # → http://127.0.0.1:8010
```

Önceden bu sayfa beş POC'u subprocess ile koşturup **stdout'u** ekrana basıyordu.
Stdout insan için yazılmış bir anlatı: hangi tool'un ham çıktısının ne olduğu ve
sıkıştırmadan sonra context'te ne kaldığı oradan çıkarılamıyordu. Sayfa şimdi üç
şey gösteriyor:

**1 · Her ajan mantığı kendi sekmesinde.** Hermes · OpenCode · OpenClaw · Codex ·
Claude Code. Sekme başlığında o mantığın kazancı yazılı.

**2 · EK-2'nin adım şeridi — ve bu koşuda hangisinin vurduğu.** Yalnız **tool izine
dokunan** adımlar (ölçüt: adım ya tool çıktısına dokunuyor ya da
`tool_call ↔ tool_result` çiftini ilgilendiriyor). Vuran adım renkli, vurmayan soluk;
her adımda "ne kaybediliyor" ve kaç birime dokunduğu yazılı. Elenmiş genel
context-yönetimi adımları da listeleniyor — neyin **bilerek** dışarıda bırakıldığı
görünür olsun diye.

**3 · Her tool birimi için ÖNCE / SONRA, kırpılmadan.** Satır açılınca solda ham tool
çıktısı, sağda o mantığın context'te bıraktığı metin; üstünde birime dokunan
adım zinciri (`sicak40k → damga → serialize` gibi) ve o birime tam olarak ne
olduğunun cümlesi.

Ham stdout kaybolmadı: her sekmenin altındaki düğme gerçek
`*_tool_trace_poc.py` subprocess'ini eskisi gibi koşturuyor.

Veri `poc/kiyas.py`'den geliyor. O modül POC'ları **import ederek** kendi gerçek
fonksiyonlarını (`prune_old_tool_results`, `prune`, `step1..step11`, `CodexSession`,
`ClaudeCodeSession`) çağırıyor; POC'ların `main()` yolu değişmedi.

Ölçülen (her mantık kendi POC senaryosunda):

```
                ham        final     kazanç   tool izine dokunan adımlardan vuran
Hermes        33.063 →     2.101     %93,6    dedup(1) · tip-farkında özet(3)
OpenCode     148.413 →    76.167     %48,7    spill(1) · skill koruma(1) · 40K(8) · damga+serialize(3)
OpenClaw     138.850 →       104     %99,9    sanitize(3) · projection(3) · gruplama(2) · oversized(1)
Codex        123.254 →       153     %99,9    truncate_middle(13) · [+windowing, kapsam dışı]
Claude Code  126.487 →    13.190     %89,6    microcompaction(1) · subagent(1) · [+auto, kapsam dışı]
```

Sayıları yan yana koymak yanıltıcı olur — senaryolar aynı değil, her POC kendi
sistemini gösteren bir iz kuruyor. Karşılaştırılabilir olan **mekanizma**: aynı tool
birimine kim ne yapıyor ve geriye ne bırakıyor.

Panelin ortaya çıkardığı iki şey:

- **Bir birime birden çok adım sırayla vurabiliyor.** Codex'te `shell` çıktısı önce
  Katman A'da ortadan kesiliyor (18.536 → 5.026), sonra B2 windowing onu tamamen
  düşürüyor. Tek bir "18.536 → 0" satırı bu ara adımı yutardı; zincir gösterimi
  bunu görünür kılıyor. Aynısı Claude Code'da `WebFetch` için: önce diske
  (microcompaction), sonra konuşma özetine.
- **Bir koşuda vurmayan adımlar da bilgi.** Codex'in `placeholder trim`'i hiç
  tetiklenmedi (log: "budanacak eski çıktı YOK → compaction gerekecek"), Hermes'in
  basınç demotion'ı gerekmedi, OpenClaw'ın onarımı gerekmedi (batch düşerken çift
  birlikte düştüğü için yetim kalmadı). Mekanizmanın **ne zaman** devreye girmediğini
  görmek, girdiğini görmek kadar öğretici.

`kapsam dışı` etiketi EK-2'nin ölçütünü uyguluyor: Codex'in windowing'i ve Claude
Code'un auto-compaction'ı konuşma seviyesidir, tool izine ait değildir — ama bu
koşuda iz üzerinde ölçülebilir etkileri olduğu için gizlenmiyor, kesikli çerçeveyle
işaretleniyor.

## Gerçek repoya sadakat — doğrulama (11.08.2026)

POC'lar "sadık simülasyon" iddiasında. `harnesses/` altındaki gerçek klonlara karşı
tek tek denetlendi. Sonuç: **dördü doğrulandı, biri doğrulanamaz, ikisinde sapma var.**

| Sistem | Kaynak | Sabitler | Mekanizma |
|---|---|---|---|
| **Hermes** | `hermes-agent/agent/context_compressor.py` (6.883 satır) | **7/7 birebir** | 4 geçiş ✓ |
| **OpenCode** | `opencode/…/tool/truncate.ts` + `session/compaction.ts` | **6/6 birebir** | 2 katman ✓ |
| **OpenClaw** | `openclaw/src/agents/compaction-planning*.ts` | **8/8 birebir** | 5 tool adımı ✓ |
| **Codex** | `codex/codex-rs/` | POC ölçeği (belirtilmiş) | A ✓ · B2 ✓ · **B1 sapıyor** |
| **Claude Code** | ✗ **kapalı kaynak** | gözlem | gözlem |

Doğrulanan sabitler:

```
Hermes    _MAX_TAIL_MESSAGE_FLOOR=8 · _PRESSURE_KEEP_RECENT_MESSAGES=3
          _SKILL_VIEW_PRUNE_MIN_CHARS=5000 · dedup floor 200 · arg eşiği 500 · arg head 200
          _PRUNED_TOOL_PLACEHOLDER ve SKILL_PRUNED_MARKER_PREFIX metinleri birebir
OpenCode  MAX_LINES=2000 · MAX_BYTES=50*1024 · PRUNE_MINIMUM=20_000 · PRUNE_PROTECT=40_000
          TOOL_OUTPUT_MAX_CHARS=2_000 · PRUNE_PROTECTED_TOOLS=["skill"] · DEFAULT_TAIL_TURNS=2
OpenClaw  BASE_CHUNK_RATIO=0.4 · MIN_CHUNK_RATIO=0.15 · SAFETY_MARGIN=1.2
          SUMMARIZATION_OVERHEAD_TOKENS=4096 · TEXT_TRUNCATE_THRESHOLD_CHARS=32_768
          TEXT_SAMPLE_CHARS=8_192 · PLANNING_MAX_CHARS=256*1024 · oversized eşiği ×0.5
Codex     "Warning: truncated output (original token count: N)\nTotal output lines: M"
          → output-truncation/src/lib.rs:21 ile birebir
```

Hermes'in dört geçişi de gerçek fonksiyon adlarıyla eşleşiyor
(`_prune_old_tool_results`, `_summarize_tool_result`, `_truncate_tool_call_args_json`);
OpenClaw'ın beş tool adımı da (`stripToolResultDetails`, `groupCompactionMessages`,
`repairToolUseResultPairing`) gerçekte var. OpenCode'un `prune`'u aynı yönde
(sondan başa) yürüyor ve `turns < 2` / `msg.info.summary` kırılımları birebir.

### Bulunan iki sapma

**1 · Hermes dedup markeri — düzeltildi.** POC
`[Duplicate of newer result at #N — content omitted]` yazıyordu; gerçekte
(`context_compressor.py:2889`) `[Duplicate tool output — same content as a more
recent call]` — `#N` indeksi **yok**. Panel bu metni SONRA kutusunda aynen
gösterdiği için birebir hale getirildi.

**2 · Codex "B1 placeholder trim" — gerçekte böyle çalışmıyor.** POC, history
sığmayınca eski `function_call_output`'ları `[context window truncated output]`
yer tutucusuna çeviriyor. Gerçek Codex'te bu string **hiç yok**; bağlam taşınca
`ContextManager::remove_first_item()` çağrılıyor — en eski öğe **tamamen siliniyor**,
`normalize::remove_corresponding_for` ile çift eşi de birlikte
(`codex-rs/core/src/compact.rs:310`, `context_manager/history.rs:191`).

> Korunan **değişmez aynı** (çift bütünlüğü hiç bozulmuyor), ama **teknik farklı**:
> gerçek Codex *içeriği yer tutucuyla değiştirmiyor, öğeyi kaldırıyor*. POC'un
> gösterdiği "iskelet kalır" davranışı Codex'in kendisinde değil. Panelde bu adım
> zaten "vurmadı" çıkıyor (log: "budanacak eski çıktı YOK"), yani ekranda yanlış
> bir şey gösterilmiyor — ama mekanizma anlatımı düzeltilmeli.
>
> Bu POC'un davranışını **değiştirmedim**: Katman A ve B2 doğru, B1'i kaldırmak
> POC'un anlattığı hikâyeyi değiştirir. Karar senin.

### Claude Code neden doğrulanamıyor

`harnesses/claude-code` **kaynak kod içermiyor** — public issue/changelog reposu
(`CHANGELOG.md`, `feed.xml`, `plugins/`, `scripts/`). POC'un kendi başlığı bunu
zaten söylüyor: *"kapalı kaynak → gözlem"*. Yani microcompaction eşiği (~4K token),
önizleme boyutu (~500 token) ve auto-compact eşiği (~%80) **ölçülmüş değil,
gözlemlenmiş** değerler. Panelde bu sekmenin sayıları diğer dördüyle aynı
güvenilirlikte okunmamalı.

### Ortak sınır

Beş POC da gerçek sistemlerin **tool-trace ile ilgili alt kümesini** uyguluyor —
tam kopya değil. Ölçek farkı somut: Hermes'in gerçek `context_compressor.py`'si
6.883 satır, POC'u 406. Sabitler ve invariant'lar birebir; çevresindeki üretim
kodu (retry, telemetri, config, çoklu sağlayıcı yolları) POC'ta yok. Sayfadaki
uyarı bunu söylüyor ve doğru.

## Sonraki adım (isteğe bağlı)

- Mock tool'ları gerçek ürün tool'larıyla değiştir (`../toolsmockproduct/` → 119 tool);
  harness `ToolResult` sözleşmesi aynı kalır.
- `LLM_LIVE=1` ile gerçek endpoint'e bağla (yanıt + LLM-tabanlı stratejiler).
