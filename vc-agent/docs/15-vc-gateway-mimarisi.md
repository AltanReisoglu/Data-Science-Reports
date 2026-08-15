# 15 — VC Gateway: OpenClaw mimarisi, AutoGen motoru

*Mimari ve mentalite [13](13-openclaw-teknik-analiz.md)'ten, motor AutoGen'den.
Bu belge neyin alındığını, neyin **bilerek alınmadığını** ve her ikisinin de
gerekçesini yazıyor. Kurulum: [README](../README.md).*

---

## 0 — Neden

[13](13-openclaw-teknik-analiz.md) OpenClaw'un mimarisini çıkardı. Bizim
`pipeline/` o mimarinin *işlevini* değil **şeklini** eksik yapıyordu:

| Eksik olan | Nasıl görünüyordu |
|---|---|
| Oturum | `conversation.STATE_PATH` — **tek dosya, tüm kullanıcılar için** |
| Bağlam sınırı | `BufferedChatCompletionContext(24)` — mesaj sayıyor, token değil ([06 §2]) |
| Uzatma noktası | Tek bir `InterventionHandler` fonksiyonu |
| Durum yeri | `pipeline/data/` — repo içinde, her koşuda kirleniyor |
| Bellek | Yok |
| Zamanlanmış iş | Yok |

Bunların hiçbiri AutoGen'in problemi değil ve AutoGen hiçbirini cevaplamıyor.
OpenClaw hepsini **açık kaynak olarak** cevaplamış. Kopyalanan şey kod değil,
**karar**.

---

## 1 — Eşleme

| OpenClaw (docs/13) | Bizde | Motor tarafı |
|---|---|---|
| Gateway — tek doğruluk kaynağı | `pipeline/gateway/` + `server.py` | **tek `SingleThreadedAgentRuntime`** (`gateway/runtime.py`) |
| Oturum yönlendirme, `dmScope` | `gateway/sessions.py` | **`SessionAgent(RoutedAgent)`** — oturum id'si ajan anahtarı (`05:670`) |
| Kanal adaptörü | `channels.py` | **`ChannelAgent(RoutedAgent)`** + `TypeSubscription` |
| Session lane serileştirme | Oturum başına `asyncio.Lock` | — |
| `runId` + `agent.wait` | `gateway/runs.py` | `CancellationToken` |
| Plugin hook'ları (13 nokta) | `gateway/hooks.py` | `InterventionHandler` + `EVENT_LOGGER_NAME` |
| `before_tool_call → {block:true}` | `gateway/approval.py` | `GatedWorkbench` |
| Context engine: Ingest/Assemble/Compact | `context_engine.py` | `ChatCompletionContext` alt sınıfı |
| Memory: `MEMORY.md` + `memory/*.md` | `memory.py` | `docs_index` TF-IDF |
| Capability = çekirdek sözleşme | `caps.py` | `typing.Protocol` |
| Audit ledger — yalnız metadata | `policy.py` | `ToolCallEvent` |
| Cron: her koşuda taze oturum | `gateway/cron.py` | — |
| State dizini | `~/.vcagent` (`VC_STATE_DIR`) | — |

### 1.1 En değerli tek eşleme — ve artık gerçek

OpenClaw'ın **"oturum anahtarı → izole ajan"** modeli, `autogen_core`'un
**topic kaynağı → ajan anahtarı** mekanizmasının aynısı ([14 §3.6], `05:670`).
İki farklı sistem, aynı sonuca ayrı ayrı varmış.

**Bu artık bir benzetme değil.** `TypeSubscription(topic_type="turn",
agent_type="session")` kayıtlı; `TopicId("turn", "agent:main:web:dm:alice")`'a
yayın yapmak `AgentId("session", "agent:main:web:dm:alice")` ajanını yaratıyor.
Oturum kimliklerimiz zaten CloudEvents desenine (`^[\w\-\.\:\=]+\Z`) uyduğu
için ara katman gerekmedi.

> **`[ölçüldü]`** `test_session_agent.py::test_the_topic_source_becomes_the_agent_key`
> bunu iddia değil **test** olarak tutuyor: iki oturum → iki ajan örneği, elle
> tutulan sözlük yok.

### 1.2 Kontrol düzleminin neresi core'a taşındı, neresi taşınmadı

| Taşındı | Taşınmadı | Gerekçe |
|---|---|---|
| Oturumlar (`SessionAgent`) | Hook kayıt defteri | Hook **karar** veriyor; `block` bir mesaj değil, bir cevap |
| Kanallar (`ChannelAgent`) | Onay kapısı | Aynı sebep — asenkron bir kapı kapı değildir |
| — | Relay, cron | Yönlendirme tablosu ve zamanlayıcı; mesajlaşmaya çevirmek tören |

**Ve akış yolu ikiye ayrıldı, bunu saklamıyoruz:** SSE token token akış
istediği için dashboard `Conversation`'ı doğrudan çağırıyor; relay webhook'u,
cron ve MCP `SessionManager.dispatch()` ile **yayın** yapıyor. İkisi de aynı
transcript'i yazıyor ve aynı lane'i tutuyor — ayrım *teslimatta*, oturum
kavramında değil. "Bütün gateway mesaj güdümlü" demek, en çok kullanılan yol
öyle değilken, yanlış olurdu.

### 1.3 Core runtime'ın ölçülmüş riski ve üç karşılık

`06 §8`: çöken handler → `_process_publish` içindeki `gather` erken dönüyor →
`stop_when_idle()` bariyeri erken açılıyor → **tamamlanmış kardeş sonuçlar
sessizce kayboluyor**. Kontrol düzleminde kaybolan şey birinin mesajı olur.

1. **`stop_when_idle()` hiç çağrılmıyor.** Kaybın kaynağı o bariyer. Gateway
   runtime'ı sunucuyla açılıyor, sunucuyla kapanıyor.
2. **`SessionAgent` fırlatmıyor.** `fanin.BranchWorker` disiplini birebir:
   *"a failure is published, not raised."* Testi var: bir oturum çökerken
   diğeri cevap vermeye devam ediyor.
3. **Transcript modelden önce yazılıyor.** Çöküş cevabı kaybettiriyor, sorunun
   kaydını değil. Bunun da testi var.

---

## 2 — Bileşenler

### `gateway/sessions.py` — kim konuşuyor

Bir oturum anahtarı `(channel, kind, peer, account)`. Yönlendirme tablosu
OpenClaw'ınkiyle aynı:

| Kaynak | Davranış |
|---|---|
| DM | `dm_scope`'a göre (varsayılan `per-channel-peer`) |
| Grup / oda | **Oda başına izole** — kapsam ne olursa olsun |
| **Cron** | **Her koşuda taze** (`SessionKey.ephemeral`) |
| Webhook | Hook başına izole |
| MCP | Çağıran peer başına |

**Üç zaman damgası, bir değil.** `session_started_at` günlük reseti,
`last_interaction_at` idle reseti sürüyor. Birleştirmek ikisini birden bozuyor.

> **Ölçülmüş ayrıntı:** peer kimliği *dışarıdan* geliyor (OpenClaw peer'i,
> webhook yükü) ve oturum kimliği bir dosya adına dönüşüyor. Ayraçlar zaten
> temizleniyordu ama `..` hayatta kalıyordu; test yazarken çıktı, `_slug` nokta
> dizilerini de sıkıştırıyor. → `test_sessions.py`

### `gateway/hooks.py` — nereden uzatılır

OpenClaw'ın 13 hook noktası birebir. Karar kuralları da:

- `before_tool_call → {"block": True}` **terminaldir**
- `message_sending → {"cancel": True}` **terminaldir**
- `before_agent_reply → {"reply": ...}` turu üstlenir
- Gerisi *katkı* verir; sonraki hook öncekinin güncellemesini görür

**Karantina, ve neden asimetrik.** Çöken bir hook kaydedilir, devre dışı bırakılır
ve boru hattı **devam eder** — OpenClaw'ın context engine kuralı: *agent susmaz*.
Ama:

> **Çöken hook atlanır, "blokla" diyen hook dinlenir.**
> Çökmede açığa düşmek ile kararda kapalıya düşmek farklı sorular; ikisine aynı
> cevabı vermek birini yanlış yapar.

O yüzden onay kapısı kendi işini kendi `try`'ında yapar ve hata hâlinde
`{"block": True}` döner. **Bozuk kapı açık kapı olmaz.**

### `gateway/approval.py` — dışa dönük çağrılar

`observability.py` baştan beri şunu yazıyordu: *"ilk mutasyon yapan tool
geldiğinde kapı hazır."* OpenClaw köprüsü o tool.

**Ada göre değil, alt-dizeye göre.** Tool adları **uzak** bir sunucudan geliyor;
tam adlar listesi upstream bir şeyi yeniden adlandırdığı anda **açığa düşer**.
Alt-dize kuralı kapalıya düşer.

> **Bunun bedelini ölçtük.** İlk taslakta `sessions_send` ve `sessions_spawn`
> yazmıştım — tahminle. `openclaw mcp serve`'ün gerçek yüzeyi ikisini de
> içermiyor. İçerdiği şey **`permissions_respond`**: OpenClaw'ın *kendi* bekleyen
> izin isteklerini cevaplayan tool. Ajanımız onu çağırabilseydi OpenClaw'ın
> onaylarını operatör adına verebilir, **iki bağımsız kapıyı bire indirirdi**.
> İçinde "send" gibi bir fiil yok; hayalden yazılmış bir liste onu geçirirdi.
> Varsayılan işaretçilere `respond` ve `approve` bu yüzden eklendi.

Onay **tek çağrılık**: `(tool, argümanlar)` digest'i onaylanır ve tüketilir.
Bir mesaja "evet" demek her mesaja "evet" demek değil.

### `gateway/workbench.py` — kapı nereye kondu

`InterventionHandler` runtime'ın mesaj yolunda oturuyor; çıplak bir
`AssistantAgent`'ta öyle bir yol yok. `Workbench` ise **her** tool çağrısının
geçtiği yer — yerel fonksiyon da, uzak MCP tool'u da.

Sarmalayınca kural **ajan yazılırken var olmayan** tool'lar için de geçerli
oluyor; bir workbench'in tool *kaynağı* olmasının tüm sebebi bu.

**Kapılamak ile filtrelemek iki ayrı karar.** Kapılı tool görünür kalır ve
çağrısı reddedilir — `messages_send` için doğrusu bu, çünkü ajan *"mesaj
atardım ama onayın lazım"* diyebiliyor. Filtreli tool ise prompt'a hiç
girmiyor; iki sebeple:

- **Prompt maliyeti.** Şema her turda ödeniyor. `06` yedi şemayla canlı bir
  zaman aşımı kaydediyor; 10 yerel + 9 OpenClaw + 2 DeepWiki = 21 yuvarlama
  hatası değil.
- **Meşru kullanımı olmayan tool.** `permissions_respond` için hiçbir istek
  "doğru çağrı" değil; göstermek ve reddetmek bir şey kazandırmıyor.

`VC_OPENCLAW_TOOLS` varsayılanı dördü tutuyor: `conversations_list`,
`conversation_get`, `messages_read`, `messages_send`. Filtrelenen tool ada göre
çağrılsa da reddediliyor — liste modele bir ipucu, kontrol ise kod.

İki davranış bilerek:
- **Bloklanan çağrı exception değil, hata sonucu döner** — ajan reddedildiğini
  öğrenip söyleyebilsin.
- **`list_tools` kapılanmaz** — tool'u gizlemek modele körlemesine denetiyor,
  reddetmek sormayı öğretiyor.

### `context_engine.py` — bağlam

Dört yaşam döngüsü noktası, `ChatCompletionContext` üstünde. **Token sayıyor**,
mesaj değil — [06 §2]'deki kusurun düzeltmesi.

Zorunlu doğruluk ayrıntısı, OpenClaw'ın açıkça yazdığı: **araç çağrısı ile
sonucu bir bütün.** Bölme noktası bir araç bloğunun içine düşerse sınır kaydırılır.
Çağrısı özetlenip gitmiş bir `toolResult`, modelin göremediği bir sorunun cevabı
— sağlayıcılar diziyi reddediyor. `_safe_boundary` bu kural.

Özetleyici çalışmazsa **kırpar ve kırptığını söyler**:

> *"…were dropped to stay inside the context window. They were not summarised, so
> anything only stated there is no longer available — ask again rather than
> assuming."*

Özetleme **ucuz kademeyle** yapılıyor: compaction analiz değil defter tutma.

### `memory.py` — hatırlama

OpenClaw'ın cümlesi: *"gizli durum yoktur."* İki dosya, iki iş:

| Dosya | Ne zaman okunur | Maliyet |
|---|---|---|
| `MEMORY.md` | **Oturum başında, prompt'a** | Her turda ödenir → kısa kalır |
| `memory/YYYY-MM-DD.md` | Yalnız arandığında | Büyümesi bedava |

**Yeni arama motoru yazılmadı.** `docs_index`'in bölüm bölme + TF-IDF + atıf
makinesi ikinci bir kök alacak şekilde parametreleştirildi. Bellek notları ve
dokümanlar **ayrı** indeksleniyor: biri projenin muhakemesi, diğeri operatörün
hatırlanmasını istediği şey.

`preamble()` yalnız **madde**leri alıyor — dosyanın kendi kullanım talimatları
operatöre yazılmış, modele değil, ve her turda ödenirdi.

Fark dürüstçe: OpenClaw'ınki **hibrit** (vektör + anahtar kelime), bizimki
**leksikal**. Gerekçe `docs_index.py`'nin kendi docstring'inde.

### `gateway/cron.py` — zamanlanmış iş

`docs/04 §8`'in **Faz 5 + Faz 7**'si.

**Her koşu taze oturum.** Bağlam biriktiren zamanlanmış iş her gün biraz daha
pahalı olur ve sonunda bu sabahın verisi yerine geçen haftanın hatırasından
cevap verir.

**Bulgu ≠ bildirim.** `Threshold` okunabilir olsun diye aptal: yıldız deltası,
yeni push, adı geçen mention. "Neden beni uyandırdın" sorusunun cevabı insanın
okuyabileceği bir cümle olmalı — öğrenilmiş bir skorlayıcı bunu yapamaz.

Ve projenin baştan beri ısrar ettiği ayrım burada da: **"bakamadım" ile "değişmedi"
farklı.** Ulaşılamayan kaynak *raporlanır*, asla *bildirilmez* — GitHub bizi rate
limit'lediği için birini uyandırmak gürültü, saklamak ise eksiltme.

### `caps.py` — sözleşmeler

OpenClaw'ın ayrımı alındı: **plugin = sahiplik sınırı · capability = çekirdek
sözleşme**. Alınanlar: `Protocol` sözleşmeleri, registry, **beyan edilmiş
fallback**, karantina.

**Alınmayan: manifest / keşif / aktivasyon boru hattı.** OpenClaw'un ihtiyacı
üçüncü taraf plugin'lerden geliyor. Bize kimse plugin göndermiyor. Hepsi bu
repoda duran uygulamalar için yükleyici yazmak, bakımı gereken ve çalıştırılamayan
tören olurdu. Değişirse dikiş tek fonksiyon: `Registry.discover`.

**Fallback karantinaya alınmaz.** Son çare de devre dışı bırakılabilseydi, arıza
`None` olarak görünürdü.

---

## 3 — Köprü, iki yön

```
  Telegram ──▶ OpenClaw Gateway ──stdio/MCP──▶ pipeline/mcp_server.py
                     ▲                              │
                     │                         state dizini
              stdio/MCP (gated)                     │
                     │                              ▼
              pipeline/openclaw.py ◀────────── VC Gateway (server.py)
                                                AutoGen motoru
```

### 3.1 İçeri: `mcp_server.py`

`FastMCP` + stdio. Sekiz tool, hepsi `gateway/tools.py`'deki **aynı
callable'lar** — ajanın kullandıklarının ta kendisi. İki uygulama olsaydı
kaçınılmaz olarak ayrışır, ve ayrışma "OpenClaw aynı soruya web sohbetinden
farklı cevap veriyor" diye görünürdü.

**stdio'nun getirdiği kısıt, açıkça:** OpenClaw bu süreci kendi başlatıyor; süreç
çalışan gateway'in **belleğini göremiyor**. Bu yüzden okumalar state dizininden,
`vc_start_scan` ise loopback üzerinden gateway'e. Gateway ayakta değilse **net
hata** döner:

```
Not started: no gateway reachable at http://127.0.0.1:8777/api/scan (...).
Start it with `python -m pipeline.server` and try again.
```

Sessizce görünmeyen ikinci bir tarama başlatmak daha kötü cevaptı.

**Her MCP çağrısı bizde bir oturumdur** (`agent:main:mcp:dm:openclaw`) —
transcript ve audit ile. Makine trafiğini anonim saymak, sistemin kendi
trafiğinin yarısı hakkında "bu nereden geldi" sorusuna cevap verememesinin yolu.

### 3.2 Dışarı: `openclaw.py`

`McpWorkbench(StdioServerParams(command="openclaw", args=["mcp","serve"]))` —
mekanizma DeepWiki eklentisinde zaten vardı.

**İki kapı, bilerek.** Bağlantı `VC_MCP_OPENCLAW` ile kapalı gelir; her dışa
dönük çağrı ayrıca onay ister. Tek anahtar olsaydı "kanal farkındalığı" açmak
"insanlara mesaj atabilme" açmakla aynı şey olurdu.

**Ölçülmüş yüzey** (openclaw 2026.7.1-2, 2026-08-14):

| Serbest | Kapılı |
|---|---|
| `conversations_list` · `conversation_get` · `messages_read` · `events_poll` · `events_wait` · `attachments_fetch` · `permissions_list_open` | **`messages_send`** · **`permissions_respond`** |

**Sırlar:** `~/.openclaw/openclaw.json` içinde Telegram bot token'ı ve sağlayıcı
anahtarları var. **Kodumuz o dosyayı okumuyor.** `openclaw` ikilisiyle konuşuyoruz,
o kendi sırlarıyla konuşuyor.

---

## 4 — Doğrulandı

**Birim:** 278 test geçiyor (73 → 214 → 278).

**Canlı** — çalışan OpenClaw'a karşı, 2026-08-14:

```
$ openclaw mcp probe vc-agent
- vc-agent: 8 tools, resources, prompts
$ openclaw mcp doctor
- vc-agent: ok
```

```
⇢ BİZDEN OPENCLAW'A            connected (9 tools)
    conversations_list  → error=False · conversations: 1
    messages_send       → error=True
      Refused: messages_send reaches outside and needs approval.
      Approve request 5e44f2f78662 to let it through.
    bekleyen onay: 1 · keys=['conversationId', 'text']
```

Yani: okuma geçti, gönderme **gerçekten** bloklandı, onay kaydı **argüman
değerleri olmadan** tutuldu.

---

## 5 — Bilerek yapılmayanlar

| Ne | Neden |
|---|---|
| Plugin manifest / loader | Üçüncü taraf plugin yok (§2 `caps.py`) |
| Node'lar, Canvas/A2UI, Voice/Talk | Cihaz yeteneği ve kod yürütme yok |
| Sandbox | Model üretimi kod çalıştırmıyoruz |
| gRPC dağıtık runtime | `grpcio` kurulu değil; tek süreç yetiyor |
| Kendi kanal adaptörlerimiz | Telegram OpenClaw'da; ikinci kez yazmak tekrar |
| `dreaming` (otomatik bellek terfisi) | Kendi recall'ını puanlayıp neye inanacağına sessizce karar veren sistem |
| Hibrit (vektör) bellek araması | Embedding endpoint'i düşünce *dokümantasyon* aramasının düşmesi |

---

## 6 — Bu belgede ölçülmemiş olanlar

- **Telegram'dan uçtan uca soru sorulmadı.** MCP köprüsünün iki ucu da ayrı ayrı
  canlı doğrulandı (probe + doctor + gerçek tool çağrıları), ama telefondan
  yazılıp cevap alınması denenmedi.
- **Cron gerçek bir zamanlayıcıyla koşmadı.** `tick()` çağıran bir döngü henüz
  gateway'e bağlı değil; `Scheduler` testlerle doğrulandı, takvimle değil.
- **Compaction canlı modelle denenmedi.** Kuru modda kırpma yolu ve tool-çifti
  kuralı testli; `model_summariser` gerçek bir uzun sohbette koşmadı.
- **Aynı anda iki gateway** durumu (tek-örnek kilidi) yazılmadı.
- **Dashboard akışı hâlâ doğrudan çağrı.** Oturumlar core'da ama SSE yolu
  `Conversation`'ı doğrudan çağırıyor (§1.2). Yayın yolu relay/cron/MCP'de
  kullanılıyor ve testli; en çok kullanılan rota değil.
- **Sorgu katmanının eksen-boşluk eşlemesi sezgisel.** `missing_data` skorlayıcının
  kendi kelimeleriyle yazılıyor (`founder_identities`), eksen adlarıyla değil.
  Anahtar kelime eşlemesi skoru **ince** (`5?`) işaretliyor, **yok** demiyor, ve
  ham boşluk listesi tablonun altında basılıyor — okuyan bağlantıyı kendi kursun.

---

## 7 — Not: OpenClaw gateway'i `0.0.0.0`'a bağlı

`~/.openclaw/openclaw.json` içinde `gateway.bind` loopback değil; süreç
`0.0.0.0:18789` dinliyor. `gateway.auth.mode` token'lı olduğu için açık değil,
ama OpenClaw'ın **kendi belgesi** güvenli varsayılan olarak loopback + SSH tüneli
ya da Tailscale öneriyor (13 §15).

Bizim işimiz değil, ve bir şey değiştirmedik — ama görülen bir şeyi yazmamak da
bu belgenin kuralına aykırı olurdu.

---

**İlgili:** [13](13-openclaw-teknik-analiz.md) OpenClaw analizi ·
[04](04-vc-agentic-akis.md) faz planı · [06](06-autogen-incelikleri.md) ölçülmüş
tuzaklar · [14](14-autogen-protokoller-ve-farklar.md) protokoller
