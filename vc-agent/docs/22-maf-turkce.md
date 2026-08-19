# 22 — Microsoft Agent Framework: Türkçe rehber

*Kaynaklar: [20-maf-user-guide.md](20-maf-user-guide.md) (Learn kılavuzu, 106 sayfa)
ve [21-maf-tasarim-kararlari.md](21-maf-tasarim-kararlari.md) (ADR'ler + depo
belgeleri). Başlıklardaki `20:NNNN` / `21:NNNN` o dosyalardaki satır numarasıdır —
iddiayı doğrulamak için oradan gir.*

## Bu belge ne, ne değil

[11-core-guide-turkce.md](11-core-guide-turkce.md) AutoGen Core için ne yapıyorsa
bu belge MAF için onu yapıyor: kılavuzu Türkçeye çevirmiyor, **bizim projemizin
karşılığıyla yan yana koyuyor.** Her bölümün sonunda "bizde ne var" satırı var, ve
o satır bazen "bizde yok" diyor.

Evin kuralı burada da geçerli — her iddia etiketli:

| Etiket | Anlamı |
|---|---|
| **[ölçüldü]** | Bu makinede koşturuldu; `.venv-maf` ya da `.venv`'den okundu |
| **[kaynak]** | Birincil metinden doğrulandı; satır numarası verildi |
| **[teyitsiz]** | Okundu, koşturulmadı |

**Sürüm:** `agent-framework` **1.14.0** (PyPI'daki son sürüm, 2026-08-14) [ölçüldü].
Kılavuz metinleri `MicrosoftDocs/semantic-kernel-docs@6e4b13ec5395`, tasarım
kayıtları `microsoft/agent-framework@26b9200c214f`, ikisi de 2026-08-19'da çekildi.

---

# BÖLÜM 0 — Başlamadan bilinmesi gerekenler

## Üç isim, iki depo · `20:21978`

MAF, **AutoGen ve Semantic Kernel'in birleşimi**. Kılavuzun kendi cümlesiyle:
*"developed by the core AutoGen and Semantic Kernel teams at Microsoft, and is
designed to be a new foundation for building AI applications going forward"*
(`20:21990`) [kaynak].

Birleşmenin henüz tamamlanmadığının en somut kanıtı belgenin **kendi adresinde**:
MAF'ın kullanıcı kılavuzu `microsoft/agent-framework` deposunda **yok**. Learn'de
yayımlanıyor ve kaynağı hâlâ `MicrosoftDocs/semantic-kernel-docs` içindeki
`agent-framework/` klasöründe duruyor [ölçüldü — `docs/tools/fetch_maf_docs.py`
tam olarak oradan çekiyor].

Kod deposunda ise AutoGen'in hiç yayımlamadığı bir şey var: **35 tasarım kaydı
(ADR)**, kararın gerekçesiyle ve *reddedilen alternatifleriyle* birlikte. Bir
çerçeveyi değerlendirirken en çok işe yarayan malzeme budur; API'nin ne olduğunu
okumak kolay, hangi tercihin hangi bedelle alındığını okumak zordur.

## Hız farkı, tek satırda

| | AutoGen | MAF |
|---|---|---|
| Son sürüm | `0.7.5`, **2025-09-30** | `1.14.0`, **2026-08-14** |
| Durum | bakım modu, yeni özellik yok | 1.0 GA'dan beri 14 ara sürüm |
| Aradan geçen | ~11 ay donmuş | ~4 ayda 14 sürüm ≈ **3,5/ay** |

[ölçüldü — PyPI sürüm listesi ve kurulu paketler.] Bu tablo bir tavsiye değil, bir
**risk ölçüsü**: AutoGen'de bulacağın hatayı düzeltecek kimse yok; MAF'ta bulduğun
API iki ay sonra değişmiş olabilir. İkisi de maliyet, ve hangisinin daha ucuz
olduğu projeye göre değişir.

## Olgunluk: 36 paket, 8'i `released` · `21:22749`

`PACKAGE_STATUS.md` paketleri `alpha` / `beta` / `rc` / `released` diye ayırıyor
[kaynak]. Sayım [ölçüldü]:

* **`released` (8):** `agent-framework`, `-core`, `-openai`, `-foundry`,
  `-orchestrations`, `-declarative`, `-ag-ui`, `-github-copilot`
* **`beta` (22):** `-anthropic`, `-gemini`, `-ollama`, `-mistral`,
  `-bedrock`, `-claude`, `-redis`, `-mem0`, `-devui`, `-tools`, …
* **`alpha` (6):** bütün `hosting-*` ailesi ve `azure-cosmos-memory`
* **`deprecated` (1):** `agent-framework-azure-ai` — `foundry`'ye taşındı

Yani **36 aktif paketin 8'i** kararlı; kalan 28'i `beta` ya da `alpha` [ölçüldü].

Buna bir de **özellik seviyesinde** aşama etiketleri ekleniyor: `HARNESS`,
`FIDES`, `MCP_SKILLS`, `AGENT_HOOKS`, `EVALS`, `SESSION_STORE` — hepsi
`experimental`, ve içe aktarıldıklarında gerçekten `ExperimentalWarning`
fırlatıyorlar [ölçüldü].

> **Sunumda söylenecek cümle:** "MAF'ın çekirdeği kararlı; bu sunumda anlatacağım
> en ilginç şeylerin çoğu — harness, FIDES, beceriler — henüz deneysel." Bunu
> söylememek, ilk soruda söyletilmekten çok daha pahalıya gelir.

## Kurulum tuzağı: aynı ortama girmiyorlar

`agent-framework` ile `autogen-*` **aynı sanal ortamda çözülemedi** — pip 10
dakikalık bir bağımlılık aramasından sonra vazgeçti [ölçüldü]. Bu yüzden PoC'ta
MAF ayrı bir `.venv-maf` içinde yaşıyor ve alt süreç olarak konuşuluyor
([pipeline/maf_runner.py](../pipeline/maf_runner.py)).

İronisi şu: Microsoft'un kendi göç örneği ikisini yan yana kurmayı söylüyor —
`pip install "autogen-agentchat autogen-ext[openai] agent-framework"` (`21:23497`)
[kaynak]. Bizim ortamımızda bu komut çalışmadı. Göç belgesi bir şeyi söyler,
çözücü başka bir şey yapar; ölçmeden inanılmaz.

---

# BÖLÜM 1 — Ajan

## `Agent`: tek çalıştırma yüzeyi · `20:6060`

AutoGen'de `AssistantAgent` (agentchat) ile `RoutedAgent` (core) iki ayrı
dünyaydı. MAF'ta tek bir `Agent` var; sınıf yazmak istersen `BaseAgent`'ten
türetiyorsun ama arayüz aynı kalıyor (`20:1841`).

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

agent = Agent(client=OpenAIChatClient(), instructions="…", tools=[my_tool])
result = await agent.run("Görev")
```

Göç kılavuzunun kendi karşılaştırması (`20:22003`) [kaynak]:

```python
# AutoGen
agent = AssistantAgent(name="assistant", model_client=client, tools=[my_tool])
# MAF
agent = Agent(name="assistant", client=client, tools=[my_tool])
```

Yüzey neredeyse aynı. **Davranış değil.**

## En büyük tek davranış farkı: 1'e karşı 40

| | varsayılan tool döngüsü |
|---|---:|
| AutoGen `AssistantAgent.max_tool_iterations` | **1** |
| MAF `DEFAULT_MAX_ITERATIONS` (`_tools.py:95`) | **40** |

[ölçüldü — iki kurulu paketten okundu.] Microsoft bunu göç kılavuzunda da yazıyor:
*"`AssistantAgent` is single-turn unless you increase `max_tool_iterations`.
`Agent` is multi-turn by default and keeps invoking tools until it can return a
final answer"* (`20:22019`) [kaynak].

Bu, [06-autogen-incelikleri.md](06-autogen-incelikleri.md)'deki 4. tuzağın
Microsoft tarafından teyididir: AutoGen'de ajan bir tool çağırır, sonucu görür ve
**durur**; zincirleme davranış sessizce imkânsızdır. MAF'ta tam tersi risk var —
kimse durdurmazsa 40 tur döner, ve 40 tur gerçek bir faturadır.

> **Bizde:** [pipeline/conversation.py](../pipeline/conversation.py) bu değeri
> açıkça set ediyor. Her iki çerçevede de bu satırı yazmayan proje, varsayılanın
> hangi yöne kaydığını fark etmeden yanlış davranış alıyor.

## `AgentSession`: durum artık ajanın içinde değil · `20:1564`

MAF ajanları **varsayılan olarak durumsuz**. Çok turlu konuşma istiyorsan oturumu
sen taşıyorsun:

```python
session = agent.create_session()
await agent.run("…", session=session)
```

AutoGen'de `model_context` ajana aitti; MAF'ta oturum çağrıya ait. Fark pratik:
aynı ajan nesnesi eşzamanlı iki kullanıcıya hizmet verebiliyor, ve oturumu
serileştirmek (`SessionStore`, `FileSessionStore`) çerçevenin işi.

> **Bizde:** [pipeline/gateway/sessions.py](../pipeline/gateway/sessions.py) aynı
> izolasyonu AutoGen'in *topic kaynağı → ajan anahtarı* eşlemesiyle kuruyor
> (`05:670`). MAF'ta bu bir dil özelliği; bizde bir desen. İkisi de çalışıyor,
> ama MAF'ınkini yanlış yazmak daha zor.

## Boru hattı: bir turun içinde ne oluyor · `20:672`

MAF bir çalıştırmanın aşamalarını isimlendiriyor — ve bu isimler bizim akış
ekranımızdaki kutularla birebir örtüşüyor:

```
istek → agent middleware → context provider → chat middleware → model
      → function middleware → tool → (döngü) → yanıt
```

Dört ayrı ara katman noktası var, ve her biri isteği **durdurabiliyor**. AutoGen'de
bunun karşılığı yalnız runtime seviyesindeki `InterventionHandler`'dır: tek nokta,
ve mesaj bazlı. MAF'ta katman ajan / sohbet / fonksiyon seviyelerine ayrılmış.

---

# BÖLÜM 2 — Middleware: AutoGen'de olmayan katman · `20:4081`

Bu, MAF'ın AutoGen'e eklediği en yapısal şey. Beş alt başlık ve her birinin ayrı
kılavuz sayfası var:

| Sayfa | Ne çözüyor |
|---|---|
| Ajan / çalıştırma kapsamı · `20:2172` | Ara katman ajanın tamamına mı, tek çağrıya mı |
| Sohbet ara katmanı · `20:2799` | Model isteğine dokunmak |
| Sonuç değiştirme · `20:4818` | Tool sonucunu **yerine geçerek** döndürmek |
| Paylaşılan durum · `20:5489` | Katmanlar arası veri taşımak |
| Sonlandırma ve korkuluk · `20:5652` | Turu ortada kesmek |

`MiddlewareTermination` — turu ara katmandan kesme yeteneği — AutoGen'de
`DropMessage`'ın yaptığı işin genelleştirilmiş hâli, ama **tool seviyesinde**.

> **Bizde:** [pipeline/observability.py](../pipeline/observability.py)
> `InterventionHandler` + `DropMessage` ile bir kapı kuruyor ve çalışıyor. MAF
> karşılığı `ToolApprovalMiddleware`; aynı işi yapıyor ve yazmamız gerekmezdi.
> Bunu kaybetmiş olmuyoruz — kapının *nerede* durduğunu biz seçtik ve
> test ettik ([pipeline/tests/test_codeexec.py](../pipeline/tests/test_codeexec.py)).

---

# BÖLÜM 3 — Bağlam yönetimi

## Context provider · `20:1296`

Ajana her çalıştırmada **ek talimat, ek tool ve ek bellek** enjekte eden nesne.
AutoGen'in `Memory` protokolüne benziyor ama daha geniş: bellek de, todo listesi
de, çalışma kipi de aynı arayüzden geliyor.

## Compaction · `20:869` · karar kaydı `21:11358`

Bağlam penceresi dolduğunda ne yapılacağı. MAF dört strateji sunuyor
(`ContextWindowCompactionStrategy`, `ToolResultCompactionStrategy`,
`SelectiveToolCallCompactionStrategy`, `SummarizationStrategy`) [ölçüldü — kurulu
pakette dördü de var].

ADR `0019` bunun **neden** böyle olduğunu anlatıyor (`21:11358`) — AutoGen'de
karşılığı olmayan bir belge türü.

> **Bizde:** [pipeline/context_engine.py](../pipeline/context_engine.py), 364
> satır. AutoGen'in `BufferedChatCompletionContext`'i **mesaj sayıyor**, token
> değil; biz token sayan ve tool-çağrısı/sonucu çiftini bölmeyen bir sıkıştırma
> yazdık. MAF'ta bu **hazır geliyor**, ve aynı kuralı koyuyor.
>
> Yani: bizim en çok emek verdiğimiz modüllerden biri, MAF'a geçilse silinirdi.
> Bu kötü haber değil — o modülü yazarken öğrendiğimiz kural (bölme noktası bir
> tool bloğunun içine düşemez), MAF'ın da aynı kuralı koyduğunu görünce
> **doğrulanmış** oldu.

---

# BÖLÜM 4 — Tool'lar ve onay

## `@tool` ve şema çıkarımı

AutoGen `FunctionTool(func, description=…)` sarmalayıcısı istiyor; MAF `@tool`
dekoratörüyle şemayı imzadan ve `Annotated` açıklamalarından **kendisi çıkarıyor**
(`20:14185`) [kaynak]. Docstring hâlâ arayüz — ama sarmalama yükü yok.

## Onay: `approval_mode` · `20:14990` · karar kaydı `21:2782`

```python
@tool(approval_mode="always_require")
async def place_trade(...): ...
```

MAF'ta onay bir **tool özelliği**. Üç kip var, ve `ToolApprovalMiddleware`
"bir daha sorma" tipi **kalıcı kurallar** ve sezgisel otomatik onay taşıyor.

ADR `0006` (`21:2782`) kararın gerekçesini veriyor [kaynak].

> **Bizim itirazımız aynen duruyor.** Harness'te *tool approval* varsayılan olarak
> **açık** ve "standing approvals" ile geliyor (`20:6479`) [kaynak]. Bizim
> kapımızda onay **bir kez tüketilir** ve imza kodun/argümanların kendisi
> üstündedir; aynı çağrı ikinci kez geldiğinde yeniden sorar. Bu daha yorucu, ve
> bilinçli olarak öyle: "bir daha sorma", ajanın bir sonraki sefer *aynı adı
> taşıyan başka bir işi* yapmasına kapıyı açar.

## Ve Microsoft bunu kendisi yazıyor: tool adı çakışması

Harness belgesinin güvenlik notu, kelimesi kelimesine (`21:23703`) [kaynak]:

> *"Auto-approval rules may match by name, so any other local tool registered
> under one of these names — for example the shell tool given a caller-configurable
> name — may also be auto-approved, bypassing the human approval boundary."*

Yani **ada göre otomatik onay, insan onay sınırını atlatabiliyor** ve bunu
çerçevenin kendi belgesi uyarı olarak yazıyor. Bizim kapımızın ada değil
**imzaya** bakması tam da bu sınıf hatayı kapatıyor.

> Bu, sunumda "biz neden kendi kapımızı yazdık" sorusunun en iyi cevabı: yazmadık,
> *farklı* yazdık, ve farkın gerekçesi satıcının kendi uyarısında duruyor.

---

# BÖLÜM 5 — Workflow: en büyük mimari fark

## Kontrol akışından veri akışına · `20:22698`

Göç kılavuzunun kendi tanımı (`20:22778`) [kaynak]:

* **GraphFlow (AutoGen):** *control-flow based* — kenarlar geçiş, mesajlar
  **herkese yayınlanıyor**, geçişler yayınlanan içeriğe göre koşullanıyor.
* **Workflow (MAF):** *data-flow based* — mesajlar **belirli kenarlardan**
  yönleniyor, yürütücüler girdileri hazır olduğunda tetikleniyor, eşzamanlı
  yürütme destekleniyor.

Bu fark, bizim [pipeline/fanin.py](../pipeline/fanin.py)'de ölçtüğümüz sessiz
kardeş kaybının kökeni. GraphFlow'da paralel dalın sonucu yayınla geliyor ve
bariyer erken açılabiliyor; MAF'ta kenar tipli ve yürütücü girdisi hazır olmadan
çalışmıyor.

> **Bizim ölçümümüz:** GraphFlow ham hata altında **0–1** sonuç döndürüp süre
> sınırını dolduruyor; core pub/sub + `ClosureAgent` kuyruğu **2** sonucu ~3 ms'de
> topluyor. MAF'ın veri akışı modeli üçüncü bir cevap ve bunu **ölçmedik**
> [teyitsiz]. Ölçülmeye değer.

## Checkpoint · `20:15957`

Workflow durumu diske yazılıp geri yüklenebiliyor (`FileCheckpointStorage`,
`InMemoryCheckpointStorage`). AutoGen'de takım seviyesinde `save_state` /
`load_state` var ama **iş ortasında** duraklama-devam etme yok.

## Request/response: insan döngüde, çerçevenin içinde · `20:18077`

Kılavuzun cümlesi (`20:23264`) [kaynak]: *"AutoGen's `Team` abstraction runs
continuously once started and doesn't provide built-in mechanisms to pause
execution for human input. Any human-in-the-loop functionality requires custom
implementations outside the framework."*

Bizim onay kapımız tam olarak o "custom implementation outside the framework".
MAF'ta `ctx.request_info()` + `@response_handler` ile çerçevenin içinde.

## Beş orkestrasyon · `20:19624`

| AutoGen takımı | MAF karşılığı | Kılavuz |
|---|---|---|
| `RoundRobinGroupChat` | `SequentialBuilder` | `20:19993` |
| — (bizde `fanin.py`) | `ConcurrentBuilder` | `20:18434` |
| `SelectorGroupChat` | `GroupChatBuilder(selection_func=…)` | `20:18769` |
| `Swarm` | `HandoffBuilder` | `20:19092` |
| `MagenticOneGroupChat` | `MagenticBuilder` | `20:19652` |
| `GraphFlow` | `WorkflowBuilder` | `20:7313` |

## Ve burada kılavuz bayat

Göç kılavuzu "Future Patterns" başlığı altında şunu diyor (`20:23251`) [kaynak]:

> *"The Agent Framework roadmap includes several AutoGen patterns currently in
> development: **Swarm pattern**, **SelectorGroupChat**."*

Ama `agent-framework-orchestrations` 1.14.0'da `HandoffBuilder` ve
`GroupChatBuilder` **ikisi de var ve `released`** [ölçüldü — `.venv-maf`'tan
içe aktarıldı; `21:22749`'da paket `released`].

Yani Microsoft'un göç kılavuzu, kendi yayımladığı paketin gerisinde. Bu, tek
kaynağa güvenmenin bedeli: kılavuz "yok" diyor, paket "var" diyor, ve doğru cevap
**paketi açıp bakmak**.

---

# BÖLÜM 6 — Harness: Microsoft bunu kavram yaptı · `20:6438`

Kılavuzun tanımı (`20:6453`) [kaynak]:

> *"An agent harness is the runtime scaffolding that turns a language model into
> an agent that can perform work. It drives model and tool calls, manages
> conversation state and context, applies approval policies, and can keep the
> agent progressing through a multi-step task."*

Bu, [13-openclaw-teknik-analiz.md](13-openclaw-teknik-analiz.md)'in baştan beri
anlattığı şeyin Microsoft tarafından yazılmış tanımı. **Harness artık resmî bir
kavram.**

## Yetenek matrisi — ve bizim karşılıklarımız · `20:6476`

| MAF harness yeteneği | Varsayılan | Bizde |
|---|---|---|
| Fonksiyon çağırma döngüsü | açık | ✔ `conversation.py` |
| Her model çağrısından sonra kalıcılık | açık | ✔ `save_state` |
| Compaction | token sınırı verilirse | ✔ `context_engine.py` |
| Todo takibi | **açık** | ✘ |
| Plan / execute kipleri | **açık** | ✘ |
| Dosya belleği | açık | ✔ `memory.py` (Markdown) |
| Tool onayı + kalıcı kurallar | **açık** | ✔ ama *tek kullanımlık* imza |
| OpenTelemetry | açık | ✔ `telemetry.py` |
| Web arama | istemci destekliyorsa | ✔ (kapı arkasında) |
| Beceriler (Skills) | Python'da opt-in | ✘ |
| Arka plan ajanları | opt-in, deneysel | ✘ |
| Kabuk yürütme | opt-in | ✘ (Docker kod yürütme var) |
| Döngü (looping) | opt-in | ✘ |

## *Build your own claw* · `21:23717`

Ve işin en çarpıcı yeri. Microsoft'un kendi harness örneğinin adı **"build your
own claw"**, bir blog serisiyle geliyor, ve örnek uygulama bir **kişisel finans /
yatırım asistanı** [kaynak]:

* `valuation` ve `risk-scoring` adlı iki **beceri**, `SKILL.md` dosyaları olarak
* `place_trade` tool'u `approval_mode="always_require"` ile
* `portfolio.csv` üzerinde dosya erişimi, salt-okunur araçlar otomatik onaylı
* CodeAct ile portföy hesabı, arka plan ajanlarıyla ticker başına araştırma

Yani: **bizim yaptığımız işin şeklini, Microsoft aynı alanda örnek olarak
yayımlamış.** Bu sunum için bir zayıflık değil, en güçlü doğrulama —
"OpenClaw'ın harness deseni + AutoGen'in motoru" tezi, satıcının kendi yol
haritasında duruyor.

---

# BÖLÜM 7 — FIDES: bankanın soracağı soru · `20:12073` · `21:14943`

MAF'ın AutoGen'de karşılığı **hiç olmayan** en ciddi mekanizması.

**Sorun** (`20:12077`) [kaynak]: prompt enjeksiyonu OWASP LLM Top 10'un birincisi,
ve üretimdeki ajanların çoğu ona iki sezgiyle karşılık veriyor — savunmacı sistem
prompt'u ya da elle yazılmış izin listesi. İkisi de **deterministik değil**.

**Çözüm:** her içerik parçası bir **bütünlük** (güvenilir / güvenilmez) ve bir
**gizlilik** (public / private / user-identity) etiketi taşıyor; etiketler tool
çağrıları boyunca **kendiliğinden yayılıyor**; ve politika hassas bir tool
çalışmadan **önce** zorlanıyor.

Dört parça (`20:12128`) [kaynak]:

| Parça | Ne yapıyor |
|---|---|
| `ContentLabel` | Her `Content` ile birlikte gezen köken etiketi |
| `LabelTrackingFunctionMiddleware` | Girdilerin en kısıtlayıcı etiketini çıktıya taşır |
| `PolicyEnforcementFunctionMiddleware` | Her tool çağrısını etikete karşı kontrol eder |
| `quarantined_llm` + `ContentVariableStore` | Güvenilmez içeriği, ham baytları ana modele hiç göstermeden ayrı ve tool'suz bir modelle işler |

Kılavuzun kendi cümlesi meselenin özü (`20:12099`): *"The model is still in charge
of deciding what to do, but the framework is in charge of deciding what is allowed
to happen."*

> **Bizde:** yok, ve olması gerektiğini iddia etmiyoruz. Bizim kapımız **tool
> adına ve imzaya** bakıyor; verinin nereden geldiğini izlemiyor. Yani bir tarama
> sonucunun içine gömülmüş talimat, bizim kapımızdan **geçer** — kapı çağrıyı
> görür, çağrıyı doğuran metni değil.
>
> Bu, PoC'un bilinen ve yazılı sınırı. Banka bağlamında sorulacak ilk teknik soru
> da muhtemelen budur, ve cevabı "MAF'ta bunun adı FIDES, deneysel, ve bizim
> mimarimize takılabilir" olmalı — "bizde de var" değil.

**Uyarı:** `agent_framework.security` içe aktarıldığında gerçekten
`ExperimentalWarning` fırlatıyor [ölçüldü]. Üretime bugün konulacak bir şey değil.

---

# BÖLÜM 8 — Beceriler (Agent Skills) · `20:12500` · `21:12733`

AutoGen'de **"skill" diye bir soyutlama yok**; en yakın karşılığı
`dump_component`/`load_component` idi (`11:249`). MAF'ta beceri birinci sınıf:
`SKILL.md` dosyaları, frontmatter, referans belgeleri, ve çalıştırılabilir
scriptler. Kaynak çeşitleri kurulu pakette [ölçüldü]:

`FileSkillsSource` · `MCPSkillsSource` · `InMemorySkillsSource` ·
`AggregatingSkillsSource` · `CachingSkillsSource` · `FilteringSkillsSource` ·
`DeduplicatingSkillsSource` · `DelegatingSkillsSource`

**Kademeli yükleme** kilit fikir: ajan önce yalnız becerinin *adını ve
açıklamasını* görüyor, gövdesi ancak gerektiğinde bağlama giriyor. Bu, bizim
[memory.py](../pipeline/memory.py)'deki `MEMORY.md` (hep yüklü) ile
`memory/YYYY-MM-DD.md` (aranınca yüklenir) ayrımının aynısı — ve OpenClaw'ın
becerileri de böyle çalışıyor.

> **Bizde:** yok. Eklenirse en doğal yer `docs_index` + `memory` ikilisinin yanı;
> altyapının yarısı zaten duruyor.

---

# BÖLÜM 9 — Gözlemlenebilirlik ve barındırma

## OTel: aynı sözleşme, daha az kurulum · `20:11151` · `20:23548`

İkisi de OTel GenAI sözleşmesini konuşuyor. Fark kurulum yükünde: AutoGen'de
`SingleThreadedAgentRuntime(tracer_provider=…)` diye elle veriyorsun; MAF'ta
`OTEL_EXPORTER_OTLP_ENDPOINT` ortam değişkeni yetiyor, ve workflow seviyesi de
kapsanıyor (`20:18288`).

> **Bizde:** [pipeline/telemetry.py](../pipeline/telemetry.py) span'leri bellekte
> toplayıp akış ekranına basıyor. Dışarı bir toplayıcıya göndermiyoruz; sunumda
> ikinci bir servis istemedik.

## Barındırma: MAF'ın açık ara önde olduğu yer · `20:26240`

Azure Functions + Durable, Foundry hosted agents, kendi kendine barındırma,
OpenAI Responses uçları, MCP sunucusu olarak yayınlama, A2A, hatta Telegram.
AutoGen'de bunun karşılığı **hiç yok** — AutoGen bir kütüphane, dağıtım senin
problemin.

Not: `hosting-*` paketlerinin **hepsi `alpha`** (`21:22749`).

---

# BÖLÜM 10 — MAF'ta *olmayan* şeyler

Göç kılavuzunun kendi tablosundan ve kurulu paketten:

| Yetenek | AutoGen | MAF |
|---|---|---|
| Dağıtık runtime (gRPC, çok makine) | var (`autogen_ext.runtimes.grpc`, `[grpc]` ekstrası) | **yok** — "planned" (`20:22021`) [ölçüldü: `agent_framework`'te tek `Grpc*` adı yok] |
| Model yanıtı önbelleği | `ChatCompletionCache` | **yok** — "🚧 Planned" (`20:22036`) [ölçüldü] |
| Aktör modeli / topic aboneliği | `autogen-core` | **yok** — workflow modeli yerine geçti |
| Anthropic / Ollama istemcisi | var | ayrı paketlerde, **`beta`** |

İlk satır bir bankada önemli: AutoGen'in *tek* gerçek ölçekleme yolu (ajanları
makinelere dağıtmak, kod değişmeden) MAF'ta henüz yok. MAF'ın cevabı yatay
dağıtım değil, **barındırma** — ki farklı bir cevap, ve bazı iş yükleri için
yetmez.

---

# BÖLÜM 11 — Bizim PoC'ta MAF

[pipeline/maf.py](../pipeline/maf.py) + [pipeline/maf_runner.py](../pipeline/maf_runner.py),
toplam 284 satır. Ekranın sağ üstündeki düğme AutoGen ↔ MAF kipini değiştiriyor.

**Var:** `Agent`, `FunctionTool(approval_mode=…, max_invocations=…)`,
`ToolApprovalMiddleware`, `AgentSession`, `WorkflowBuilder`,
`FileCheckpointStorage` — ve hepsi akış ekranında sekiz mekanizma olarak çiziliyor.

**Yok:** harness (`create_harness_agent`), FIDES, beceriler, orkestrasyon
builder'ları, barındırma.

**Açık uç:** tool kullanan turlarda `response.text` boş dönüyor; cevap
`messages` içinde ama metin içeriği olmayan bir biçimde. Sohbet balonunda bu
dürüstçe gösteriliyor ama çözülmedi.

---

## Ek — MAF okurken aklında tutulacaklar

**1. Kılavuz, pakete göre bayat olabiliyor.** Swarm/Selector örneği ölçüldü:
kılavuz "geliştiriliyor" derken paket `released`. Sürüm ayda üçse, belge geride
kalır. **Paketi aç, `dir()` çek.**

**2. "Released" paket ≠ released özellik.** `agent-framework-core` kararlı ama
içindeki harness, FIDES, beceriler `experimental` ve gerçekten uyarı fırlatıyor.
Aşama etiketi paket seviyesinde değil, **özellik seviyesinde** okunmalı
(`21:22749`).

**3. ADR'ler kılavuzdan daha çok şey anlatıyor.** Bir davranışın *neden* öyle
olduğunu arıyorsan `21`'e bak; `20` yalnız *ne olduğunu* söyler. AutoGen'de bu
seçenek hiç yoktu.

**4. Varsayılanlar ters yönde kayıyor.** AutoGen sessizce **az** yapıyordu
(1 tool turu, bellek yok); MAF sessizce **çok** yapıyor (40 tur, kalıcı onaylar,
varsayılan açık yetenekler). İkisinde de tehlike aynı: varsayılanı yazmadan
koşturmak.

**5. Aynı ortama girmiyorlar.** Karşılaştırma yapacaksan iki venv kur. Bunu
öğrenmek 10 dakika sürdü.

---

*Kaynaklar: [20-maf-user-guide.md](20-maf-user-guide.md) ·
[21-maf-tasarim-kararlari.md](21-maf-tasarim-kararlari.md) · kurulu
`agent-framework` 1.14.0 (`.venv-maf`) ve `autogen-*` 0.7.5 (`.venv`).
Telif MIT, Microsoft Corporation; bu belgedeki yorum ve ölçümler bize ait.*
