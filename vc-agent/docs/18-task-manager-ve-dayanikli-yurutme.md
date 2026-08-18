# 18 — Task manager, akış motoru, dayanıklılık ve eşzamanlılık

> **Ne bu belge:** OpenClaw'ın arka plan iş sistemini — zamanlayıcı, task defteri,
> akış motoru, kalıcılık ve iş parçacığı modeli — kaynak kodundan okunmuş hâliyle
> anlatır. Amaç OpenClaw'ı öğretmek değil; Atlas'ta aynı problemi çözerken hangi
> ayrımların **tip düzeyinde** yapılması gerektiğini göstermek.
>
> Kaynak: `github.com/openclaw/openclaw` @ `01cc7106`. Her iddianın altında dosya
> ve satır var. Etiketler: `[ölçüldü]` koddan okundu · `[kaynak]` belge böyle diyor.

---

## 0. Neden bu konu en sondan en başa önemli

Bir asistanın **reaktif** olması kolaydır: soru gelir, cevap gider, iş biter. Zor
olan **proaktif** olmasıdır — kimse sormadan çalışması, uzun süren işi kullanıcıyı
bekletmeden yürütmesi, ve yürüttüğünün kaydını tutması.

Bu üçü aynı anda istendiğinde ortaya çıkan sorular şunlar, ve hiçbiri "bir cron
kur" ile cevaplanmıyor:

- İş **ne zaman** koşacak? Ve "her gün 9'da" ile "gelen kutusuna ara sıra bak"
  aynı mekanizma mı?
- İş koşarken sunucu yeniden başlarsa **ne oluyor**?
- İş bitti ama sonucu teslim edilemedi — bu **başarı mı başarısızlık mı**?
- İki iş aynı anda aynı oturuma yazarsa **hangisi kazanıyor**?
- Bir işin "kaybolduğunu" **neye dayanarak** söylüyorsun?

OpenClaw bu soruların her birine ayrı bir mekanizmayla cevap veriyor, ve
ayrımların çoğu **şemada** — yani yanlış yapmak derleme hatası veriyor, çalışma
zamanı sürprizi değil.

---

## 1. Altı mekanizma, altı ayrı soru

`docs/automation/index.md` bir karar rehberiyle başlıyor. Özü şu: "arka planda iş
çalıştır" tek bir ihtiyaç değil.

| mekanizma | cevapladığı soru | task kaydı üretir mi |
|---|---|---|
| **Automations** | Tam zamanlama gerekiyor | **evet, her koşuda** |
| **Heartbeat** | Yaklaşık, bağlamlı yoklama yeter | **hayır, asla** |
| **Tasks** | Ne oldu, ne zaman, nasıl bitti | (kendisi kayıt) |
| **Task Flow** | Çok adımlı iş, adımlar arası durum | akış + adım başına task |
| **Hooks** | Yaşam döngüsü olayına tepki | hayır |
| **Standing orders** | Kalıcı talimat / yetki | hayır |

Ayrımın kalbi ilk iki satırda: **Automations** tam zamanlı ve izole; **Heartbeat**
yaklaşık ve tam bağlamlı. `[kaynak]` `docs/automation/index.md`

Neden ikisi birden var: "her sabah 9'da raporu gönder" **izolasyon** istiyor —
sohbetin gürültüsü rapora karışmasın. "Her yarım saatte gelen kutusuna bak"
**bağlam** istiyor — asistanın son konuşulanları bilmesi lazım. Tek mekanizmayla
ikisini iyi yapmak mümkün değil.

Ve heartbeat kendini geri çekiyor: ana kuyruk ya da automation işi meşgulse, aynı
ajan için başka bir cevap koşuyorsa, ya da hedef oturumda kuyrukta iş varsa
**erteleniyor**. Periyodik yoklama gerçek işin önüne geçmiyor. `[kaynak]`

---

## 2. Zamanlama: beş tür, ve ikisi zamana bakmıyor

Şema tek kaynak: `packages/gateway-protocol/src/schema/cron.ts:217`. Kapalı bir
birleşim (`Type.Union`), yani altıncı bir tür uydurulamıyor.

```ts
const CronScheduleSchema = Type.Union([
  closedObject({ kind: Type.Literal("at"),    at: NonEmptyString }),
  closedObject({ kind: Type.Literal("every"), everyMs: Type.Integer({ minimum: 1, … }),
                                              anchorMs: Type.Optional(…) }),
  closedObject({ kind: Type.Literal("cron"),  expr: NonEmptyString,
                                              tz: Type.Optional(Type.String()),
                                              staggerMs: Type.Optional(…) }),
  closedObject({
    // Event-driven trigger: fires once when the gateway-owned watcher running
    // `command` exits. Survives per-turn CLI teardown (runs under the gateway
    // ProcessSupervisor, not the turn process tree).
    kind: Type.Literal("on-exit"), command: NonEmptyString, cwd: Type.Optional(…) }),
  closedObject({ kind: Type.Literal("stream"), command: Type.Array(NonEmptyString, { minItems: 1 }),
                                               mode: "line" | "match", match, batchMs, … }),
]);
```

`[ölçüldü]` Beş tür: `at` · `every` · `cron` · `on-exit` · `stream`.

**Son ikisi zamana hiç bakmıyor.** `on-exit`, izlenen bir komut çıktığında bir kez
tetikliyor; `stream`, uzun ömürlü bir komutun stdout/stderr satırlarından
tetikliyor. Bunlara "zamanlayıcı" demek yanlış — olay kaynağı bunlar.

Ve `on-exit`'in kod yorumu tek başına bir tasarım dersi: **watcher, turun süreç
ağacında değil, gateway'in `ProcessSupervisor`'ında koşuyor.** Yani turu bitirmek
watcher'ı öldürmüyor. Sahiplik doğru yere konmuş.

### İki incelik, ikisi de ölçekte fark ediyor

**Yük dağıtma.** Saat başına denk gelen tekrarlı işler (dakika `0`, saat joker)
kendiliğinden **5 dakikaya kadar** kaydırılıyor. Yüz iş aynı anda uyanıp yük
tepesi yapmasın diye. `--exact` ile kapatılıyor. `[kaynak]` `cron-jobs.md:68`

**Cron'un OR tuzağı.** `[kaynak]` `cron-jobs.md:123` — ayın-günü ve haftanın-günü
alanlarının ikisi de joker değilse, `croner` **ya biri ya öteki** eşleştiğinde
tetikliyor:

```bash
# Niyet: "ayın 15'i, ama yalnız Pazartesiyse"
0 9 15 * 1
# Gerçek: her ayın 15'inde 9'da, VE her Pazartesi 9'da
# → ayda 0-1 yerine 5-6 kez
```

Standart Vixie cron davranışı, yani hata değil. Ama bir kurumsal asistanda "ayda
beş kez rapor gönderdi" bir arıza kaydıdır. Çözüm: croner'ın `+` değiştiricisi
(`0 9 15 * +1`) ya da bir alanı işin içinde kontrol etmek.

---

## 3. Koşul gözcüsü: zamana değil **duruma** bağlanmak

Bir *event trigger*, `every` / `cron` / `stream` zamanlamasının üstüne başsız bir
koşul betiği ekliyor. Zamanlama geldiğinde önce betik koşuyor; yük ancak
`fire: true` dönerse çalışıyor. `[kaynak]` `cron-jobs.md:135`

```js
// her 30 saniyede bak — ama yalnız DEĞİŞİNCE tetikle
const res = await tools.call('exec', {
  command: "gh pr checks 123 --json state -q '.[].state' | sort -u"
});
const status = String(res?.result?.details?.aggregated ?? '').trim();
json({
  fire:    status !== trigger.state?.status,          // ← kapı
  message: `PR 123 CI: ${trigger.state?.status ?? 'unknown'} -> ${status}`,
  state:   { status },                                 // ← bir sonraki değerlendirmeye taşınır
});
```

Kritik olan `trigger.state`: kalıcı. Yani betik bir önceki değerlendirmeyi
hatırlıyor ve **durumu değil değişimi** tespit edebiliyor. Bu olmadan her 30
saniyede "CI yeşil" mesajı atardı.

**Kurumsal karşılığı doğrudan:** *limit aşıldığında*, *itiraz üç günü geçtiğinde*,
*skor eşiği düştüğünde* — takvim değil, **iş kuralı**.

**Ve bir güvenlik sınırı:** koşul betikleri `cron.triggers.enabled: true` istiyor,
varsayılan kapalı. Gerekçe: gözetimsiz koşan bir betik ayrı bir güven sınıfı.

---

## 4. Task defteri: kayıt, zamanlayıcı değil

> **Tasks are records, not schedulers** — automations ve heartbeat işin *ne zaman*
> koşacağına karar verir; task'lar *ne olduğunu* izler. `[kaynak]` `tasks.md:25`

Bu ayrım küçük görünüyor ve değil. Zamanlayıcı bir **operasyon** bileşeni; defter
bir **kanıt** bileşeni. Karıştırılırsa denetime götürülecek şey, aynı zamanda
performans için optimize edilen şey olur — ve ikisi çatışır.

**Task üretenler** `[kaynak]` `tasks.md:97`:

| kaynak | runtime tipi | varsayılan bildirim |
|---|---|---|
| ACP arka plan koşuları | `acp` | `done_only` |
| Subagent spawn | `subagent` | `done_only` |
| Automation işleri (hepsi) | `cron` | `silent` |
| CLI işlemleri | `cli` | `silent` |
| Medya üretimi | `cli` | `silent` |

**Üretmeyenler:** heartbeat turları, normal sohbet turları, doğrudan `/komut`
cevapları.

Automation'ların `silent` olması ilginç: kayıt üretiyorlar ama kendi bildirimlerini
üretmiyorlar — **teslimat yolunun sahibi zamanlayıcı**. İki bildirim yolu olsaydı
kullanıcı her rapor için iki mesaj alırdı.

---

## 5. Üç eksen — ve tip düzeyinde ayrılmış olmaları

Belgedeki en değerli tasarım kararı bu, ve şemada duruyor:
`packages/gateway-protocol/src/schema/tasks.ts:15`

```ts
// ① yürütme ne oldu
const TaskStatusSchema = Type.Union([
  Type.Literal("queued"), Type.Literal("running"),
  Type.Literal("completed"), Type.Literal("failed"),
  Type.Literal("cancelled"), Type.Literal("timed_out"),
]);

// ② sonuç geri verilebildi mi
const TaskDeliveryStatusSchema = Type.Union([
  Type.Literal("pending"), Type.Literal("delivered"),
  Type.Literal("session_queued"), Type.Literal("failed"),
  Type.Literal("dismissed"), Type.Literal("parent_missing"),
  Type.Literal("not_applicable"),
]);

// ③ ikisinin bileşimi
const TaskTerminalOutcomeSchema = Type.Union([
  Type.Literal("succeeded"), Type.Literal("blocked"),
]);
```

`[ölçüldü]` Üç ayrı enum. Yani **yürütme ile teslimat aynı şey değil** ve bunu
unutmak mümkün değil — tip sistemi hatırlatıyor.

Somut senaryo: bir subagent işini bitirdi (`status: completed`), ama sonucu geri
verilecek oturum kapanmıştı (`deliveryStatus: parent_missing`). Sonuç
`terminalOutcome: blocked` — **`failed` değil**.

Belgenin gerekçesi:

> *"Bu, tamamlanmış sonucu korur; çocuk yürütmesini yanlışlıkla başarısız diye
> raporlamak yerine."* `[kaynak]` `tasks.md:125`

Fark pratik: `failed` deseydin, biri gider işi yeniden koşturur. `blocked` diyorsan
iş zaten yapılmış, eksik olan teslimat — çözüm yeniden koşturmak değil, teslimatı
düzeltmek.

### İki kural daha

**Terminal yapışkan.** `[kaynak]` Bir task terminal olduktan sonra gelen yaşam
döngüsü sinyalleri onu **düşüremiyor**. Operatör iptal ettiyse, sonradan gelen bir
başarı sinyali kararı değiştirmiyor. Yarış koşullarının sessizce kayıt bozmasını
engelleyen basit bir kural.

**`lost` çalışma-zamanı farkında.** Her kaynak için kanıt standardı ayrı:

- **ACP:** yalnız Gateway içinde *canlı* bir in-process tur kanıt sayılıyor.
  Kalıcı oturum metadata'sı yetmiyor.
- **Automation:** önce automations runtime, sonra dayanıklı koşu geçmişi.
- **Subagent:** hedef ajan deposundan çocuk oturumun kaybolması.

Ve şu cümle:

> *"Çevrimdışı CLI denetimi, kendi boş in-process durumunu otorite saymaz."*

Yani **kanıtın yokluğu, yokluğun kanıtı değil**. Bir CLI aracı Gateway'e bağlı
değilse "bende görünmüyor" diyebilir, "yok" diyemez. Bu tek cümle, bir sistemin
denetlenebilirlik olgunluğunun göstergesi.

**Saklama:** terminal kayıtlar 7 gün, `lost` olanlar 24 saat, sonra otomatik
budanıyor. `[kaynak]` `tasks.md:346`

---

## 6. Yoklama yanlış şekil

> **Tamamlanma itmelidir (push-driven):** ayrılmış iş bittiğinde doğrudan
> bildirebilir ya da isteyen oturumu / heartbeat'i uyandırabilir — **bu yüzden
> durum yoklama döngüleri genelde yanlış şekildir.** `[kaynak]` `tasks.md:25`

Bir ajana "işi başlat, sonra bitti mi diye sor" dedirtmek doğal geliyor. Ama her
yoklama bir **model turu**. İş beş dakika sürüyorsa ve otuz saniyede bir
yokluyorsan, on tur boşa gidiyor — ve her turda bütün bağlam yeniden gönderiliyor.

Doğru şekil tersine: iş bitince **o** seni uyandırıyor.

Medya üretiminde bunun somut hâli var: `image_generate` bitince tamamlanma,
orijinal ajan oturumuna bir **iç uyandırma** olarak dönüyor, ajan da takip mesajını
yazıp medyayı ekliyor. Oturum artık aktif değilse ve uyandırma başarısız olursa,
OpenClaw eksik medyayı **idempotent** bir doğrudan mesajla orijinal hedefe
gönderiyor. `[kaynak]` `tasks.md:97`

İki katmanlı geri düşüş: önce ajanı uyandır, olmazsa doğrudan gönder. Ve
idempotent — iki kez tetiklenirse iki mesaj gitmiyor.

---

## 7. Akış motoru: Task Flow

Tek bir arka plan işi **task**. Çok adımlı bir boru hattı **flow**.
`[kaynak]` `docs/automation/taskflow.md`

Bir flow'un taşıdıkları: kendi durumu, keyfi **JSON adım durumu**, bir **revizyon
sayacı**, ve bağlı task kayıtları. Flow'lar gateway restart'ını sağ atlatıyor;
task'lar ayrılmış işin birimi olarak kalıyor.

**İki kip:**

| kip | sürücü | ne zaman |
|---|---|---|
| **managed** | Plugin kodu — flow'u yaratıp adımları açıkça ilerletiyor | Çok adımlı boru hattı |
| **mirrored** | Otomatik | Ayrılmış ACP / subagent spawn'ları |

### Revizyon sayacı — bir dayanıklılık deseni

Her değişiklik flow'un **beklenen revizyonunu** taşıyor. Bayat bir yazma, daha
yeni durumu ezmek yerine **revizyon çakışması** olarak reddediliyor.

`[ölçüldü]` Bu desen OpenClaw'da tek yerde değil, bir ev kuralı — aynı
karşılaştır-ve-değiştir üç ayrı altsistemde:

```
src/config/runtime-snapshot.ts:185        expectedRevision: number;
src/config/runtime-snapshot.ts:191        …metadata.revision !== params.expectedRevision
src/config/sessions/session-accessor.reset.ts:167   if (revision !== params.expectedRevision)
src/secrets/runtime-state.ts:938          expectedRevision: number;
```

Neden önemli: dağıtık olmayan bir sistemde bile **eşzamanlı yazarlar** var — bir
plugin adımı ilerletirken bir operatör iptal edebilir. Kilitle çözersen kilitlenme
riski alırsın; revizyonla çözersen kaybeden yazar temiz bir hatayla döner.

**İptal semantiği:** iptal istendikten sonra yeni çocuk task kabul edilmiyor, ve
flow **aktif çocuk kalmayınca** `cancelled` olarak sonlanıyor. Yani iptal anlık
değil, **drenajlı**.

---

## 8. Dayanıklılık — ama **durable execution değil**

> **Düzeltme.** Bu bölümün ilk hâli "dayanıklı yürütme" başlığını taşıyordu ve bu
> yanlıştı. `durable execution` bir terimdir — Temporal, Restate, DBOS'un yaptığı
> şey: iş akışı fonksiyonu bir olay geçmişinden **yeniden oynatılır**, tamamlanmış
> adımlar **yeniden koşmaz**, ve yürütme kesildiği komuttan devam eder.
>
> OpenClaw bunu yapmıyor. Ölçtüm: `durable execution` terimi belgelerinde
> **hiç geçmiyor**, kaynakta deterministik replay, olay geçmişi oynatması ya da
> adım memoizasyonu **yok**. Olan şey daha zayıf ve farklı bir şey — ama kendi
> içinde iyi tasarlanmış.

> *"Gateway'i yeniden başlatmak ajan durumunu kaybetmez."* `[kaynak]`
> `docs/gateway/restart-recovery.md`

| durum | depolama | yeniden başlatmada |
|---|---|---|
| Konuşma geçmişi | ajan başına SQLite | dokunulmuyor, kaldığı yerden |
| Yarıda kalan main-session turu | SQLite oturum satırı + transkript | **otomatik devam ettiriliyor** ya da uzlaştırılıyor |
| Subagent koşuları | paylaşılan SQLite | kayıt boot'ta geri yükleniyor, koşular sürdürülüyor |
| Zamanlanmış işler | paylaşılan SQLite | tanımlar, çalışma durumu ve koşu geçmişi korunuyor |
| Kuyruğa alınmış giden mesajlar | SQLite | drenaj ediliyor |

### "Automatic resume" tam olarak ne yapıyor

Belgenin kendi cümlesi, ve terimi çözen şey bu:

> *"Açılıştan birkaç saniye sonra gateway her işaretli oturumu, ajana önceki
> turunun yeniden başlatmayla kesildiğini ve **mevcut transkriptten devam
> etmesini** söyleyen bir **sentetik sistem mesajıyla** yeniden gönderir."*

Yani devam eden şey fonksiyon değil, **modele yazılan bir cümle**. Turu yeniden
koşturuyor ve ajana "yarıda kalmıştın" diyor. Zaten üretilmiş ama teslim
edilmemiş bir cevap varsa metni de ekleniyor, ki ajan işi yeniden yapmak yerine
teslim edebilsin.

**Pratik sonucu, ve önemli olan bu:** bir tur yan etkili bir tool'u çağırdıktan
*sonra* ama cevabı kaydetmeden çöktüyse, kurtarma ajanı bir notla yeniden
gönderiyor. Modelin o tool'u **ikinci kez çağırmasını mekanik olarak engelleyen
hiçbir şey yok** — koruma, transkriptte önceki tool sonucunun görünüyor olması ve
modelin bunu *fark etmesi gerektiği*. Bu bir **model yargısı** güvencesi, bir
çalışma zamanı güvencesi değil.

Destedeki ayrım burada da geçerli: **yumuşak yönlendirme, sert kontrol değildir.**

### Peki ne var

| var | yok |
|---|---|
| **Dayanıklı durum** — SQLite: transkript, oturum, iş, task, teslimat kuyruğu | Deterministik replay |
| **Dayanıklı idempotensi** — her kurtarma tek bir kalıcı gönderim kimliği kullanıyor; belirsiz bağlantı hatası aynı kurtarmayı iki kez başlatamıyor | Tamamlanmış adımların memoizasyonu |
| **Dayanıklı yeniden deneme bütçesi** — üç ücretlendirilmiş deneme, yeniden başlatmalar arasında korunuyor, sonra oturum mezar taşlanıyor | Fonksiyon ortasından devam |
| **Tombstone'lar** — tamamlanmış turlar yeniden bağlanan bir outbox tarafından yeniden yürütülmeden emekliye ayrılabiliyor | Yürütme geçmişi |

Bütçe mekanizması ince: deneme **gönderimden önce** ücretlendiriliyor, gateway
isteği kabul öncesi açıkça reddederse iade ediliyor, ve gönderim sonrası sonuç
**belirsizse ücret korunuyor** — *"işi yeniden oynatmaktan kaçınmak için."* Yani
belirsizlik hep güvenli tarafa yuvarlanıyor.

Üç ayrıntı, üçü de bu dayanıklılığı ciddi yapan şeyler:

**① Gecikmeli yeniden zamanlama.** Gateway açılışında **gecikmiş izole
agent-turn işleri yeniden zamanlanıyor, anında tekrar oynatılmıyor.** Gerekçe:
model/tool önyükleme işini kanal bağlanma penceresinin dışında tutmak. `[kaynak]`
`cron-jobs.md:42`

Naif bir sistem açılışta tüm gecikmiş işleri birden koştururdu — ve en kırılgan
anda (soğuk başlangıç) en ağır yükü yaratırdı.

**② Kurtarma sınırlı.** *"Tekrar tekrar başarısız olan kurtarma sınırlıdır ve siz
inceleyene kadar bir oturumu karantinaya alabilir."* Yani sonsuz kurtarma döngüsü
yok — üç kez düşen bir oturum karantinaya giriyor.

**③ Uzlaştırma runtime-öncelikli, geçmiş-destekli.** Bir automation task'ı,
automations runtime o işi hâlâ koşuyor sayarken **canlı kalıyor** — eski bir çocuk
oturum satırı dursa bile. Runtime sahipliği bıraktıktan ve **5 dakikalık** grace
penceresi dolduktan sonra bakım, kalıcı koşu loglarına ve iş durumuna bakıyor.
`cron:<jobId>:<startedAt>` eşleşen bir terminal sonuç varsa defteri kapatıyor;
yoksa task `lost` işaretlenebiliyor. `[kaynak]` `cron-jobs.md:42`

İki kanıt kaynağı, belirli bir öncelik sırası, ve bir grace penceresi. "Kayboldu"
demek için acele etmiyor.

---

## 9. Eşzamanlılık: üç katman

Node tek iş parçacıklı bir event loop. Ama OpenClaw'ın eşzamanlılık modeli tek
katmanlı değil — `[ölçüldü]` kaynak taramasıyla üç ayrı katman çıkıyor.

### Katman 1 — Event loop: ajan işinin kendisi

Model çağrıları, tool çağrıları, oturum yönetimi. Hepsi tek event loop'ta, async
olarak. Paralellik burada **G/Ç beklemesinden** geliyor, iş parçacığından değil.

### Katman 2 — Worker thread'ler: cevabı bloklamaması gereken defter işleri

```
src/audit/audit-event-writer.worker.ts                  ← denetim kaydı yazarı
src/config/sessions/session-accessor.sqlite-archive.worker.ts  ← SQLite arşivleme
src/config/sessions/session-transcript-reconcile.worker.ts     ← transkript uzlaştırma
```

Üçünün ortak yanı: **cevap yolunda olmamaları gereken, ama yapılması gereken**
işler. Denetim belgesindeki *"yazmalar sınırlı bir arka plan işçisinden geçer"*
cümlesinin kod karşılığı bu — kayıt yazımı gerçekten ayrı bir iş parçacığında.

Sonucu: kuyruk dolarsa **kayıt düşüyor, koşu düşmüyor**. Bu bilinçli bir öncelik
kararı ve `docs/16 §2.3`'te tartıştığımız "iki hat" ihtiyacının kaynağı — çünkü
düzenlenmiş bir kurumda uyum hattının bu önceliği **ters** olmalı.

### Katman 3 — Çocuk süreçler: gateway'e ait, tura ait değil

```
src/gateway/cron-exit-watchers.ts     ← ProcessSupervisor
src/gateway/cron-stream-watchers.ts   ← ProcessSupervisor
src/gateway/cron-stream-job-owner.ts
```

`on-exit` ve `stream` zamanlamalarının izlediği komutlar burada koşuyor. Sahiplik
kritik: **gateway'in `ProcessSupervisor`'ı, turun süreç ağacı değil.** Bir tur
bittiğinde watcher ölmüyor; gateway kapandığında ise süreç-ağacı yıkımı
bekleniyor.

Ve akış kaynaklarının kendi dayanıklılığı var: hızlı başarısızlıklar geri
çekilmeli yeniden başlatmayla toparlanıyor, ama **60 saniyeden kısa beş ardışık
koşu** işi hata durumuna alıyor ve manuel yeniden etkinleştirme gerektiriyor.
`[kaynak]` `cron-jobs.md:90` — sonsuz çökme-yeniden başlatma döngüsüne karşı.

### Lane'li kuyruk: paralellik nasıl sınırlanıyor

`src/auto-reply/reply/queue/` — lane farkındalıklı bir FIFO. İki katmanlı:

```
runEmbeddedAgent
   ├─ lane "session:<key>"  → oturum başına TEK aktif koşu
   └─ global lane "main"    → toplam paralellik agents.defaults.maxConcurrent
```

`[ölçüldü]` Varsayılanlar, `src/config/types.agent-defaults.ts:327,335`:

```ts
/** Max concurrent agent runs across all conversations.
    Default: min(16, max(8, available CPU parallelism)). */
maxConcurrent?: number;

/** Max concurrent sub-agent runs (global lane: "subagent"). Default: 8. */
maxConcurrent?: number;
```

Yapılandırılmamış lane'ler için varsayılan **1**.

Neden iki katman: oturum lane'i **doğruluk** için — aynı oturuma iki koşu yazarsa
bağlam bozulur. Global lane **kaynak** için — otuz oturum aynı anda model
çağırırsa hız sınırına çarparsın.

Ve bir kullanıcı deneyimi ayrıntısı: **yazıyor göstergesi kuyruğa girer girmez**
tetikleniyor, koşu sırasını beklerken bile. Kullanıcı sistemin uyandığını görüyor.

---

## 10. Atlas'a ne taşınır

| desen | neden değerli | Atlas'taki karşılığı |
|---|---|---|
| Kayıt ile zamanlayıcının ayrılması | "ne zaman" operasyon, "ne oldu" kanıt | denetime giden şey defter |
| Üç eksen (durum / teslimat / sonuç) | biten iş, teslim edilemedi diye başarısız sayılmıyor | "rapor üretildi ama e-posta gitmedi" ayırt edilebilir |
| Terminal yapışkanlığı | geç sinyal kararı bozmuyor | iptal edilen sorgu dirilmiyor |
| `lost` kanıt standardı | kanıtın yokluğu, yokluğun kanıtı değil | denetimde "bilmiyoruz" diyebilmek |
| İtmeli tamamlanma | yoklama döngüsü token yakıyor | uzun sorgu kullanıcıyı bekletmiyor |
| Revizyonlu CAS | eşzamanlı yazar, kilitsiz | akış durumu sessizce ezilmiyor |
| Lane'li kuyruk | oturum izolasyonu + kaynak tavanı | departmanlar birbirini beklemiyor |
| Worker thread'de defter | kayıt cevabı bloklamıyor | **ama uyum hattında tersine çevrilmeli** |
| Gecikmeli yeniden zamanlama | soğuk başlangıçta yük tepesi yok | restart sonrası sistem ayakta kalıyor |
| Dayanıklı idempotensi | belirsizlik güvenli tarafa yuvarlanıyor | aynı iş iki kez teslim edilmiyor |
| Koşul gözcüsü | takvim değil iş kuralı | "limit aşıldığında" tetiklemesi |

### Ve tek bir ters çevirme

Yukarıdaki listede dokuz satır doğrudan alınıyor. **Biri ters çevrilmeli:**

OpenClaw'da denetim kaydı worker thread'de, best-effort, ve kuyruk dolarsa
**kayıt düşüyor, koşu sürüyor**. Bir geliştirici aracı için doğru öncelik.

Düzenlenmiş bir kurumda uyum hattının önceliği **tersi** olmak zorunda: kayıt
yazılamıyorsa **koşu düşmeli**. Operasyonel hat OpenClaw'ın modelinde kalabilir;
uyum hattı ayrı, senkron ve fail-closed olmalı. `docs/16 §2.3`

---

## 11. Bizde bugün ne var, ne yok

`[ölçüldü]` `pipeline/` taraması:

| mekanizma | durum |
|---|---|
| Zamanlanmış iş | **yok** — her şey istek içinde bitiyor |
| Task defteri | **yok** |
| Ayrılmış (detached) iş | **yok** |
| Akış motoru | kısmen — `graph.py` bir DAG koşturuyor ama kalıcı değil, restart'ı sağ çıkarmıyor |
| Kuyruk / lane | **yok** — eşzamanlı istek koruması yok |
| Dayanıklı durum | kısmen — tarama sonuçları diskte, koşu durumu değil |
| Worker thread | **yok** |

Yani bu belgedeki hemen her şey Atlas için **yeni iş**. İyi haber: sıra belli ve
`docs/17`'deki fazlarla örtüşüyor — önce task defteri (kayıt), sonra lane'li kuyruk
(doğruluk), sonra zamanlayıcı (proaktiflik). Akış motoru en sona kalabilir; çok
adımlı dayanıklı iş, ilk pilotta gerekmiyor.

---

## Kaynak künyesi

| bölüm | kaynak |
|---|---|
| §1 | `docs/automation/index.md` |
| §2 | `packages/gateway-protocol/src/schema/cron.ts:217` · `docs/automation/cron-jobs.md:68,90,123` |
| §3 | `docs/automation/cron-jobs.md:135` |
| §4–5 | `packages/gateway-protocol/src/schema/tasks.ts:15` · `docs/automation/tasks.md:25,97,125,346` |
| §6 | `docs/automation/tasks.md:25,97` |
| §7 | `docs/automation/taskflow.md` · `src/config/runtime-snapshot.ts:185` · `src/config/sessions/session-accessor.reset.ts:167` · `src/secrets/runtime-state.ts:938` |
| §8 | `docs/gateway/restart-recovery.md` · `docs/automation/cron-jobs.md:42` |
| §9 | `src/audit/audit-event-writer.worker.ts` · `src/config/sessions/*.worker.ts` · `src/gateway/cron-*-watchers.ts` · `src/auto-reply/reply/queue/` · `src/config/types.agent-defaults.ts:327,335` · `docs/concepts/queue.md` |

**Ölçmediklerim:** hiçbirini koşturarak doğrulamadım — bu belge kaynak kodu ve
belge okumasıdır, canlı bir zamanlayıcı koşusu değil. Performans iddiası yok.
