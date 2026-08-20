"""İki wiki'nin bölüm metinleri: AutoGen+MAF ve OpenClaw.

`make_wiki.py` motoru taşıyor (şema gömme, `.excalidraw` üretimi, doğrulama);
burası yalnız metin. Ayrı dosya, çünkü üç belgenin gövdesi tek dosyada bin
satırı geçiyordu ve motoru okumak isteyen kişi metinlerin içinde kayboluyordu.

`bind()` ile motorun `svg()` yardımcısı ve `figures` modülü enjekte ediliyor —
bu dosya `make_wiki`'yi içe aktarmıyor, yoksa döngüsel bağımlılık olurdu.
"""

from __future__ import annotations

from typing import Any, Callable

svg: Callable[[str, str], str]
figures: Any


def bind(svg_fn: Callable[[str, str], str], figures_mod: Any) -> None:
    global svg, figures
    svg, figures = svg_fn, figures_mod


# ══════════════════════════════════════════════ 26 — AutoGen + MAF
def autogen_maf() -> str:
    return f"""# AutoGen ve MAF — çerçeve wiki'si

> **Bu ne:** AutoGen'i kullanacak ya da MAF'a geçmeyi düşünen bir mühendis için.
> Tek dosya, arayarak okunmak için.
>
> **Sürümler:** `autogen-core` / `agentchat` / `ext` **0.7.5** · `agent-framework`
> **1.14.0** — ikisi de kurulu ve ölçüldü.
>
> **Etiketler:** **[ölçüldü]** koşturuldu · **[kaynak]** birincil metinden ·
> **[teyitsiz]** okundu, koşturulmadı.

---

## İçindekiler

1. [Dört isim, tek karmaşa](#s1)
2. [Üç katman](#s2)
3. [Aktör modeli](#s3)
4. [Kimlik: bir şey değil, iki şey](#s4)
5. [İki iletişim biçimi](#s5)
6. [Tool döngüsü ve tarif](#s6)
7. [Beş takım tipi ve faturaları](#s7)
8. [Durmayı öğretmek](#s8)
9. [Sekiz resmî desen](#s9)
10. [Ölçülmüş tuzaklar](#s10)
11. [MAF: halef ne getirdi](#s11)
12. [MAF: ne kaybettirdi](#s12)
13. [Geçiş haritası](#s13)

---

## 1 · Dört isim

Karıştırılan dört ayrı şey var:

| İsim | Ne | Durum |
|---|---|---|
| **microsoft/autogen v0.4+** | `autogen-core` + `agentchat` + `ext` | Bakım modu, **0.7.5** |
| AutoGen v0.2 | `ConversableAgent`, `initiate_chat` | Terk edilmiş |
| **ag2ai/ag2** | v0.2 kolundan fork, ayrı ekip | Aktif — ama `pip install ag2` artık `import autogen` **sunmuyor** |
| **microsoft/agent-framework** | AutoGen + Semantic Kernel birleşimi | Resmî halef, **1.14.0** |

> **Filtre:** Bir kaynakta `ConversableAgent` ya da `initiate_chat` görüyorsan
> o kaynak v0.2 ya da AG2 anlatıyor — bu sürümle **uyumsuz**.

---

## 2 · Üç katman

{svg("f_layers", "AutoGen'in üç katmanı")}

Ayıran şey **`autogen_core`**: ajanlar gerçekten aktör — kendi mailbox'ı olan,
mesajı **tipe göre** yönlendiren, makinelere dağıtılabilen birimler.

LangGraph'ın graf yürütücü + checkpointer'ı **dayanıklılık** sağlıyor,
eşzamanlılık modeli değil. *"AutoGen mı LangGraph mı"* çoğu zaman yanlış
sorulmuş soru — farklı katmanlar.

---

## 3 · Aktör modeli

{svg("f_actor", "Ajan ajanı çağırmıyor")}

Bir ajan başka bir ajanın nesnesini tutmuyor; runtime'a mesaj veriyor.

**Bedeli:** *"kim kimi çağırdı"* yığın izinde görünmüyor.
**Karşılığı:** yeni ajan eklemek çağıran kodu değiştirmiyor · bütün mesajlar tek
noktadan geçtiği için müdahale ve ölçüm oraya takılıyor · aynı sınıftan istediğin
kadar örnek bedava.

---

## 4 · Kimlik

{svg("f_identity", "AgentId = tip + anahtar")}

`AgentId(type, key)` — **iki parçalı**. Ve en az konuşulan, en çok işe yarayan
mekanizma şu:

> **Topic kaynağı, ajan anahtarına dönüşüyor.**
> `TopicId("tur", "oturum-42")`'ye yayın yapmak `AgentId("session", "oturum-42")`
> ajanını **yaratıyor** — oturum başına izole örnek, elle sözlük tutmadan.

Bu projede gateway oturumları tam olarak böyle çalışıyor. Ölçek gerektiğinde
ilk bakılacak yer burası.

---

## 5 · İki iletişim biçimi

{svg("f_send_vs_publish", "Doğrudan ve yayın — asimetri hatada")}

| | `send_message` | `publish_message` |
|---|---|---|
| Alıcı | tek `AgentId` | topic'e abone olan herkes |
| Dönüş | **var** | **yok** |
| Handler çökerse | çağırana **fırlatır** | **loglanır, fırlatmaz** |

Son satır bir tasarım kararı: sonucu bekleyeceksen doğrudan, olay duyuracaksan
yayın.

### Ve buradan doğan en pahalı arıza

{svg("f_fanout", "Fan-out / fan-in — ve sessiz kayıp")}

Çöken bir handler `_process_publish` içindeki `gather`'ı erken döndürüyor,
`stop_when_idle()` bariyeri erken açılıyor, ve **tamamlanmış kardeş sonuçlar
sessizce kayboluyor.**

Aynı arıza enjeksiyonuyla ölçüldü **[ölçüldü]**:

| Motor | Temiz | Sarmalayıcı arkasında | Ham hata |
|---|---:|---:|---:|
| GraphFlow | 3 | 2 | **0–1**, süre sınırı dolar |
| core pub/sub + `ClosureAgent` kuyruğu | 3 | 2 | **2**, ~3 ms |

> Resmî desenler bu konuda **birbiriyle çelişiyor**: *Concurrent Agents* kuyrukla
> topluyor, *Mixture of Agents* `asyncio.gather(...)` ile — sessiz kaybın kaynağı
> olan yapı.

---

## 6 · Tool döngüsü

{svg("f_tool_loop", "Model ister · kapı · çalışır · sonucu görür · döngü")}

### Tarif = arayüz

Model fonksiyonu görmüyor; **adını, tarifini ve parametre şemasını** görüyor.
`description` prompt'a giren tek metin, ve modelin o tool'a *ne zaman*
uzanacağına karar verdiği şey o.

### Varsayılan tavan — altı çerçeve, altı cevap

Hepsi kurulu paketten okundu **[ölçüldü]**:

| Çerçeve | Alan | Varsayılan |
|---|---|---:|
| **AutoGen** | `max_tool_iterations` | **1** |
| OpenAI Agents SDK | `Runner.run(max_turns=)` | 10 |
| CrewAI | `Agent.max_iter` | 25 |
| **MAF** | `DEFAULT_MAX_ITERATIONS` | **40** |
| LangGraph | `recursion_limit` | 10007 |
| Google ADK | `LoopAgent.max_iterations` | sınırsız |

AutoGen'de **1**: ajan tool'u çağırır, sonucu görür, **durur** — ve hata vermez.
Microsoft bunu göç kılavuzunda kendisi yazıyor **[kaynak]**.

---

## 7 · Beş takım

{svg("f_teams", "Değişen tek şey: sırayı kim belirliyor")}

Aynı görev, aynı ajanlar **[ölçüldü]**:

| Desen | Sırayı kim belirliyor | Mesaj | LLM | Tool | Token |
|---|---|---:|---:|---:|---:|
| **SelectorGroupChat** | model her turda seçiyor | 8 | 5 | 2 | **204** |
| GraphFlow | önceden çizilmiş DAG | 11 | 7 | 3 | 270 |
| RoundRobinGroupChat | sırayla | 9 | 6 | 2 | 274 |
| **Swarm** | ajan devrediyor | 14 | 7 | 4 | **334** |

**%63,7 fark.** Ödenen zekâ değil **yönlendirme özerkliği**.

### GraphFlow — boruyu çizmek

{svg("f_graphflow", "DiGraphBuilder ile akış")}

Kenarlar **veri taşımıyor**, yalnız sırayı belirliyor. Join'de
`activation_condition="all"` demezsen ilk gelen dal akışı ilerletiyor.

---

## 8 · Durmayı öğretmek

{svg("f_termination", "On bir sonlandırma koşulu")}

Sonlandırma koşulu olmayan takım **sonsuza kadar** konuşuyor, ve fatura gerçek.
On bir koşul var; en çok kullanılan dördü:

* `MaxMessageTermination` — mesaj sayar
* `TokenUsageTermination` — **token sayar**, faturaya en yakın olan
* `TimeoutTermination` — süre
* `TextMentionTermination` — bir kelime geçince

> Koşullar `&` ve `|` ile birleşiyor. Yalnız mesaj sayan bir koşul, uzun
> cevaplarla dolu bir turu ucuz sanıyor.

---

## 9 · Sekiz desen

{svg("f_patterns", "Resmî sekiz orkestrasyon deseni")}

Eşzamanlı · Sıralı · Group Chat · Handoff · Mixture of Agents · Münazara ·
Yansıma · Kod yürütme.

Bunlar **kütüphane değil, tarif**. Hiçbiri `import` edilmiyor; kılavuzda kodla
anlatılan yapılar.

---

## 10 · Ölçülmüş tuzaklar

{svg("f_gotchas", "Hiçbiri istisna fırlatmıyor")}

| Tuzak | Sonuç |
|---|---|
| `tools=` ve `workbench=` birlikte | `ValueError` — tek net hata |
| `model_context` verilmemiş | Ajanın **belleği yok**, hata da vermiyor |
| OpenAI-*uyumlu* endpoint | `model_info` **zorunlu** |
| `max_tool_iterations` = 1 | Zincirleme davranış sessizce imkânsız |
| Dış runtime, ajan çöküyor | **Fırlatmaz, asar** |
| `description` boş ajan | `SelectorGroupChat` **kör** seçiyor |
| `Handoff` adı küçük harfe düşüyor | Elle yazınca eşleşmiyor |
| `stop_when_idle()` | Handler çökerse bariyer erken açılıyor |

> **Ortak nokta:** bulunan hataların **hiçbiri istisna fırlatmadı.** Sıfır
> döndü, boş kaldı, asılı kaldı, ya da hata metnini cevap diye sundu.
> Core'u öğrenmenin yolu API'sini okumak değil, **arıza davranışını ölçmek**.

---

## 11 · MAF ne getirdi

{svg("f_components", "MAF'ın eklediği katmanlar")}

AutoGen'de **karşılığı olmayan** beş şey:

| Yetenek | Ne yapıyor |
|---|---|
| **Middleware** | Ajan / sohbet / fonksiyon seviyelerinde ara katman; her biri turu **durdurabiliyor** |
| **Checkpoint** | Workflow durumu diske yazılıp geri yükleniyor |
| **İnsan döngüde** | `ctx.request_info()` + `@response_handler` — çerçevenin **içinde** |
| **Harness** | `create_harness_agent` — todo, plan/execute kipleri, dosya belleği, onay, OTel |
| **FIDES** | Bütünlük + gizlilik etiketleri; politika hassas tool çalışmadan **önce** zorlanıyor |

Kılavuzun kendi cümlesi **[kaynak]**:

> *"AutoGen's `Team` abstraction runs continuously once started and doesn't
> provide built-in mechanisms to pause execution for human input."*

### Mimari fark: kontrol akışından veri akışına

* **GraphFlow** — *control-flow*: kenarlar geçiş, mesajlar **herkese** yayınlanır
* **Workflow** — *data-flow*: mesajlar **belirli kenarlardan**, yürütücü girdisi
  hazır olunca tetikleniyor

Bu, §5'teki sessiz kardeş kaybının kökeni.

---

## 12 · MAF ne kaybettirdi

| Yetenek | AutoGen | MAF |
|---|---|---|
| Dağıtık runtime (gRPC) | var (deneysel) | **yok** — "planned" |
| Model yanıtı önbelleği | `ChatCompletionCache` | **yok** — "🚧 Planned" |
| Aktör modeli / topic | `autogen-core` | **yok** |

### Ve hızın faturası

* **1.0 GA'dan sonra iki ayda 15 kırıcı değişiklik** — Microsoft'un kendi
  🔴 işaretlemesiyle **[kaynak]**
* 36 paketin **8'i** kararlı; 22 `beta`, 6 `alpha`
* Harness, FIDES, beceriler → hepsi `experimental` ve gerçekten
  `ExperimentalWarning` fırlatıyor **[ölçüldü]**

### Sürüm hızı — ölçüldü

| Paket | Son sürüm | Kaç gün önce |
|---|---|---:|
| **autogen-agentchat** | 0.7.5 | **323** |
| agent-framework | 1.14.0 | 5 |
| langgraph | 1.2.11 | 8 |
| openai-agents | 0.22.0 | **0** |

---

## 13 · Geçiş haritası

Microsoft'un kendi göç kılavuzundan **[kaynak]**:

| AutoGen | MAF |
|---|---|
| `AssistantAgent(model_client=…)` | `Agent(client=…)` |
| `FunctionTool(fn)` | `@tool` — şemayı imzadan çıkarıyor |
| `RoundRobinGroupChat` | `SequentialBuilder` |
| `SelectorGroupChat` | `GroupChatBuilder(selection_func=…)` |
| `Swarm` | `HandoffBuilder` |
| `MagenticOneGroupChat` | `MagenticBuilder` |
| `GraphFlow` | `WorkflowBuilder` |
| `model_context` (ajana ait) | `AgentSession` (çağrıya ait) |

> **Dikkat:** kılavuz Swarm ve Selector'ı *"currently in development"* diyor,
> ama ikisi de 1.14.0'da **var ve `released`** **[ölçüldü]**. Kılavuz kendi
> paketinin gerisinde — paketi aç, `dir()` çek.

---

<sub>Üretim: `python docs/tools/make_wiki.py` · şemalar `docs/diagrams/figures.py`
· kaynak metinler `docs/05` · `docs/08` · `docs/20` · `docs/21` · Türkçe rehberler
`docs/11` · `docs/22`</sub>
"""


# ══════════════════════════════════════════════════════ 27 — OpenClaw
def openclaw() -> str:
    return f"""# OpenClaw — harness wiki'si

> **Bu ne:** OpenClaw'ın nasıl çalıştığı, ve bir kurumda **neyinin alınıp
> neyinin alınmayacağı**. Tek dosya, arayarak okunmak için.
>
> **Sürüm:** OpenClaw `@01cc7106` · 22 paket · 51 tool (44'ü canlı kurulumda)
>
> **Ayrım:** OpenClaw bir kütüphane değil, **çalışan bir sistem**. AutoGen'i
> kodunuza gömüyorsunuz; OpenClaw'ı çalıştırıyorsunuz.

---

## İçindekiler

1. [Harness ne demek](#s1)
2. [Mimari kuşbakışı](#s2)
3. [Üç kontrol ekseni](#s3)
4. [Yetki kapsamları](#s4)
5. [Onay: komuta değil, plana](#s5)
6. [Dış içerik veri, talimat değil](#s6)
7. [Bellek katmanları](#s7)
8. [Kademeli açığa çıkarma](#s8)
9. [Bağlam motoru](#s9)
10. [Zamanlama](#s10)
11. [Dayanıklılık](#s11)
12. [İki kayıt hattı](#s12)
13. [Niş yüzeyler](#s13)
14. [Ne alınır, ne alınmaz](#s14)

---

## 1 · Harness ne demek

Microsoft'un kendi tanımı — MAF kılavuzundan, ve kelime artık resmî
**[kaynak]**:

> *"An agent harness is the runtime scaffolding that turns a language model into
> an agent that can perform work. It drives model and tool calls, manages
> conversation state and context, applies approval policies, and can keep the
> agent progressing through a multi-step task."*

OpenClaw bunun bugün çalışan, olgun bir örneği.

---

## 2 · Mimari

{svg("f_oc_arch", "Gateway, ajan, kanallar, node'lar")}

**Tek Gateway süreci.** Oturumlar, onaylar, zamanlama, kanallar hepsi orada.
Kurulu sistemde ölçüldü:

| Ne | Sayı |
|---|---:|
| Sohbet komutu | **89** |
| Denetim olayı (canlı) | **100** + devam imleci |
| Tool | 51 kaynakta · **44** canlı kurulumda |
| Paket | 22 |

---

## 3 · Üç kontrol ekseni

{svg("f_three_axes", "Sandbox · tool policy · elevated")}

"İzin" tek kavram değil, **üç ayrı soru**:

| Eksen | Soru | Anahtar |
|---|---|---|
| **Sandbox** | Tool **nerede** koşuyor? | `agents.*.sandbox.mode` |
| **Tool policy** | **Hangi** tool çağrılabilir? | `tools.allow` / `tools.deny` |
| **Elevated** | Kutunun **dışına** çıkış var mı? | `tools.elevated.*` — yalnız `exec` |

**Kurallar:** `deny` her zaman kazanır · `allow` doluysa listede olmayan her şey
bloklu · tool policy sert duraktır, `/exec` bile reddedilmiş `exec`'i geri
getiremez.

### Ve OpenClaw'ın kendi belgesindeki uyarı

> *"Tool policy tool'u **adına göre** filtreler; `exec` içindeki yan etkileri
> incelemez. `exec` serbestse, `write`/`edit`/`apply_patch`'i reddetmek shell
> komutlarını salt-okunur yapmaz."*

**"Yazma tool'unu kapattık, artık read-only" cümlesi yanlıştır.** Üç ekseni
karıştırmak en yaygın yapılandırma hatası, ve sonucu güvenlik tiyatrosu.

---

## 4 · Yetki kapsamları

{svg("f_scopes", "Kapsam çağrının parametresinden türetiliyor")}

Sekiz yetki kapsamı var, ama asıl fikir tabloda değil:

> **Aynı metot, farklı parametreyle farklı yetki istiyor.**

`agent` metodu sıradan bir tur için yazma yetkisiyle geçiyor, ama `/reset` için
yönetici istiyor. Yani yetki **metot adına** değil, **çağrının içeriğine**
bakıyor.

### Taşınacak fikir: rol bir tool listesi değil, **grup adı**

13 tool grubu var (`group:fs`, `group:runtime`, `group:web`…). Kurumda bu
`group:musteri-verisi`, `group:kredi-sorgu`, `group:rapor`, `group:dis-erisim`
olur. **Yeni bir tool eklendiğinde 40 rol dosyası güncellenmiyor.**

---

## 5 · Onay

{svg("f_frozen_plan", "Onay plana bağlanıyor, komuta değil")}

**Donmuş plan:** onay bir komuta değil, **plana** bağlanıyor. Onaydan sonra
argümanlar yeniden doğrulanıyor; dosya değiştiyse koşu reddediliyor.

Bizim karşılığımız: imza `(tool, argümanlar)` üstünde ve **bir kez tüketiliyor**.

> Neden önemli: modelden aynı işi ikinci kez istediğinde farklı bir program
> yazıyor **[ölçüldü]** — imzalar `029f4d1f…` ve `107fdfd1…`. Onaylananla
> çalışanın aynı olmasının tek yolu, çalıştırılacak olanın **onaylanan metin**
> olması.

---

## 6 · Dış içerik

{svg("f_external_content", "Veri, talimat değil")}

Web'den, dosyadan, tool sonucundan gelen içerik **veri** olarak işaretleniyor —
talimat olarak değil. Prompt enjeksiyonuna karşı ilk savunma bu ayrım.

**Dürüst sınır:** bu heuristik bir savunma, deterministik değil. Deterministik
karşılığı MAF'ta var — **FIDES**, bütünlük ve gizlilik etiketleriyle — ve
deneysel.

---

## 7 · Bellek

{svg("f_memory_tiers", "Beş bellek katmanı")}

{svg("f_memory_write", "Güvenlik sınırı YAZMA yolunda")}

Belleğin güvenlik sınırı **okuma** tarafında değil, **yazma** tarafında. Bir
kere yanlış yazılan olgu orada yaşıyor ve her turda geri okunuyor.

Bizim karşılığımız: bellek düz **Markdown** dosyaları, gizli depo yok.
Okunabilen, düzeltilebilen, silinebilen, sürüm kontrolüne konabilen bellek.

---

## 8 · Kademeli açığa çıkarma

{svg("f_skill_disclosure", "Bir satır tarif · gövde seçilince")}

Prompt'a giren şey yalnız **frontmatter**: ad + bir satır açıklama. Skill'in
gövdesi girmiyor; ancak seçilince yükleniyor.

Elli skill'in tamamını prompt'a koymak elli skill'lik token demek. Bir satırla
elli tanesi taşınabiliyor.

### Skill arama üç katman **[kaynak]**

* **Yerel keşif** — diskte `SKILL.md` dosyaları, `agent-filter` ile bu ajana
  görünenler süzülüyor
* **Kademeli yükleme** — gövde ancak `/skill <ad>` ile geliyor
* **Uzak arama** — `/clawhub`, bir kayıt defteri; *npm'in skill'ler için hâli*

Ve döngüyü kapatan: **`/learn`** — az önce yaptığın düzeltmeyi yeniden
kullanılabilir bir skill'e çeviriyor.

### Aynı fikrin tool tarafı

{svg("f_tool_search", "Büyük katalog, küçük prompt")}

---

## 9 · Bağlam motoru

{svg("f_ctx_engine", "Dört yaşam döngüsü noktası")}

**Ingest** (mesaj eklendi) · **Assemble** (bütçeye sığan sıralı küme) ·
**Compact** (pencere doldu) · **After turn**.

### Sıkıştırmayı doğru yapan kural

> Bir tool çağrısı ve sonucu **tek birimdir.** Bölme noktası bir tool bloğunun
> içine düşerse sınır kaydırılıyor.

Aksi hâlde: cevabı görünen ama sorusu özetlenip silinmiş bir tool sonucu kalıyor,
ve sağlayıcılar bu diziyi doğrudan **reddediyor**.

Bizim `context_engine.py` bu kuralı AutoGen'in `ChatCompletionContext`'i üstünde
yeniden kuruyor — çünkü AutoGen'in `BufferedChatCompletionContext`'i **mesaj
sayıyor, token değil**.

---

## 10 · Zamanlama

{svg("f_task_stack", "Zamanlama yığını")}

{svg("f_task_axes", "Üç eksen, tip düzeyinde ayrılmış")}

Beş zamanlama türü var ve **ikisi zamana bakmıyor** — koşul gözcüsü zamana değil
**duruma** bağlanıyor.

### Cron'un OR tuzağı

Gün alanları **OR**'lanıyor: `0 9 * * 1` ile `0 9 1 * *` birlikte yazılırsa hem
pazartesileri hem ayın birinde koşuyor. OpenClaw bunu **ayrı işler açarak**
çözüyor.

### Task defteri — zamanlayıcı değil, **kayıt**

{svg("f_task_lifecycle", "Bir işin yaşam döngüsü")}

Defter ne zaman koşacağına karar vermiyor; ne koştuğunu **yazıyor**. İkisini
karıştırmak, yeniden başlatmada geçmiş işleri yeniden oynatmaya götürüyor.

---

## 11 · Dayanıklılık

{svg("f_durable", "Dayanıklı durum — ama durable execution değil")}

**Önemli ayrım:** OpenClaw dayanıklı **durum** tutuyor; dayanıklı **yürütme**
değil. Süreç ortada ölürse, yarım kalan tur kaldığı yerden devam etmiyor.

`Temporal`/`durable execution` bekleyen biri bunu bilmeli.

{svg("f_failover", "Model failover")}

{svg("f_loopguard", "Döngü kırıcı ve sıkıştırma sonrası nöbetçi")}

---

## 12 · İki kayıt hattı

{svg("f_two_ledgers", "Uyum kaydı ile hata ayıklama kaydı ayrı")}

| | Uyum kaydı | Hata ayıklama kaydı |
|---|---|---|
| Değişmez mi | **evet** | hayır |
| Saklama süresi | var | kısa |
| Sır taşır mı | **asla** | taşıyabilir |
| Kim okur | denetçi | mühendis |

Tek hatla ikisini birden yapmak **ikisini de bozar**: ya denetim kaydına sır
sızar, ya hata ayıklama kaydı ömür boyu saklanır.

{svg("f_secrets", "Sırlar ve telemetri")}

---

## 13 · Niş yüzeyler

Sorulursa açılacak, kendiliğinden anlatılmayacak konular.

{svg("f_repair", "Tool call repair — bozuk çağrıyı kurtarmak")}

Model bozuk JSON ürettiğinde turu çöpe atmak yerine onarmaya çalışıyor.

{svg("f_result_middleware", "tokenjuice — komuta değil SONUCA dokunuyor")}

Tool **sonucunu** küçültüyor. Tool'un ne yaptığı değişmiyor; **modelin ne kadarını
gördüğü** değişiyor.

{svg("f_trajectory", "Trajectory — oturumun uçuş kayıt cihazı")}

`/export-trajectory` redakte edilmiş bir destek paketi çıkarıyor. Kullanıcının
gerçekten rapor göndermesini sağlayan şey bu.

{svg("f_self_learning", "Düzeltmeyi skill'e çevirmek")}

### Doğrulanmamış olanlar

| Ne | Durum |
|---|---|
| **Lobster** — tipli iş akışı runtime'ı | **[teyitsiz]** — resmî **eklenti**, çekirdekte değil, kurmadık. Katalog: *"typed pipelines + resumable approvals"* |
| **Code Mode / Swarm** | **[teyitsiz]** — okundu, koşturulmadı |

---

## 14 · Ne alınır, ne alınmaz

{svg("f_atlas", "Üç ayrı ilişki")}

### Alınacaklar — **karar kuralları**, kod değil

* Üç kontrol ekseninin **ayrı** tutulması
* Rol = grup adı, tool listesi değil
* Onay plana bağlanır, komuta değil
* Dış içerik veri, talimat değil
* Bellek güvenlik sınırı **yazma** yolunda
* İki kayıt hattı
* Kademeli açığa çıkarma
* Zamanlanmış koşu **taze oturum** alır

### Alınmayacak — **güven modeli**

OpenClaw **tek bir güvenilen operatör** varsayıyor. Kurumda o varsayım geçerli
değil.

Ölçülen kanıt, sistemin kendi cümlesi — bir `/openclaw` satırı gönderdiğimizde
kapımız tuttu **[ölçüldü]**:

> *"O ajanın kabuk erişimi var ve şu an onay sormadan çalıştırıyor
> (**exec: mode=full, ask=off**); bizim kapımız içeride ne yapacağını görmez."*

### Sonuç: üç ayrı ilişki

**AutoGen'i gömüyoruz** (motor, ince arayüz arkasında) · **OpenClaw'ı
öğreniyoruz** (karar kuralları) · **OpenClaw'ı mühendislikte kullanmaya devam
ediyoruz**.

> Atlas olarak OpenClaw **kurmuyoruz**.

---

<sub>Üretim: `python docs/tools/make_wiki.py` · kaynak `docs/13` (mimari analiz) ·
`docs/16` (kurumsal okuma) · `docs/18` (zamanlama ve dayanıklılık) · canlı
ölçümler kurulu OpenClaw `@01cc7106`'dan</sub>
"""
