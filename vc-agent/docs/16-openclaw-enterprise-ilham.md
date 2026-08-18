# 16 — OpenClaw'dan Atlas'a: enterprise asistan için ne alınır, ne alınmaz

**Varsayım (açıkça yazıyorum, çünkü belge buna göre şekillendi):** Atlas, KKB
gibi düzenlenmiş bir finansal veri kurumunun iç kullanıcılarına hizmet edecek
bir enterprise AI asistanı. Yani: çok kullanıcılı, kimlik doğrulamalı, PII'ye
dokunan, denetlenebilir olması *zorunlu*, ve "ajan yanlış şeyi çağırdı" cümlesinin
maliyeti bir geliştirici makinesindekinden birkaç mertebe büyük. Bu varsayım
yanlışsa §3 ve §5 değişir, §2 büyük ölçüde ayakta kalır.

---

## 0. Bu belge neye dayanıyor

Üç kaynak, üçü de birincil:

| Kaynak | Ne | Nasıl ölçüldü |
|---|---|---|
| Resmî repo | `github.com/openclaw/openclaw`, commit `01cc7106`, 558 MB | Diskte: `~/Desktop/adapted/harnesses/openclaw`, doğrudan okundu |
| Canlı gateway | 351 RPC metodu, 44 tool / 14 grup / 4 profil, 74 skill kurulu → 40 model-görünür | Daha önce RPC ile ölçüldü → `docs/pdf/openclaw-ici.pdf` |
| Kendi analizimiz | `docs/13-openclaw-teknik-analiz.md` | Mimari haritası, bu belge onun üstüne biner |

Repo ölçüleri: **161 extension**, **22 paket**, **764 belge dosyası**. Bu belge
o 764'ün enterprise açısından anlamlı olan ~20'sini okuyup çıkardıklarımı taşıyor;
her iddianın altında dosya adı var.

Not — hoş bir tesadüf: OpenClaw'ın kendi tehdit modeli dosyasının adı
`docs/security/THREAT-MODEL-ATLAS.md`. Oradaki "ATLAS" MITRE ATLAS (yapay zekâ
sistemleri için düşman taktik çerçevesi), bizim Atlas'la ilgisi yok — ama §2.12'de
göreceğin gibi, o dosyanın *biçimi* Atlas için doğrudan kopyalanacak bir şey.

---

## 1. Tez

> OpenClaw'dan alınacak şey ajan döngüsü değil. Ajan döngüsü herkeste var ve
> AutoGen'de zaten var. Alınacak şey **ajanı kuşatan kontrol düzlemi** — ve daha
> da önemlisi, o düzlemin **kendi sınırlarını dürüstçe beyan etme alışkanlığı**.

İkinci yarı, birincisinden daha kıymetli. OpenClaw'ın belgeleri sürekli şunu
yapıyor: bir mekanizmayı anlatıyor, sonra "bu şunu **kanıtlamaz**" diye kendi
iddiasını daraltıyor. Denetim kaydı için: *"Bir satırın yokluğu hiçbir şey
kanıtlamaz."* Yetki kapsamları için: *"`operator.read` düşmanca çok-kiracılı bir
izolasyon sınırı değildir."* Bellek için: *"Bu korelasyondur, anonimleştirme
değildir."*

Bir bankada/kredi bürosunda bir asistanı öldüren şey, mekanizmanın olmaması değil
— **olduğu sanılan bir mekanizmanın denetim toplantısında çökmesi**. OpenClaw bu
cümleleri kodun yanına yazmış. Atlas'ın alması gereken ilk şey bu.

---

## 2. Alınacaklar

Öncelik sırasına göre. Sıralama "enterprise'da ne kadar erken lazım olur"a göre.

### 2.1 Üç kontrol ekseni birbirine karıştırılmıyor

**Kaynak:** `docs/gateway/sandbox-vs-tool-policy-vs-elevated.md`

OpenClaw "izin" diye tek bir kavram tutmuyor. Üç ayrı soru, üç ayrı mekanizma:

| Eksen | Cevapladığı soru | Anahtar |
|---|---|---|
| Sandbox | Tool **nerede** koşuyor? | `agents.*.sandbox.mode` = `off` / `non-main` / `all` |
| Tool policy | **Hangi** tool çağrılabilir? | `tools.allow` / `tools.deny` / `profile` |
| Elevated | Sandbox dışına **kaçış** var mı? | `tools.elevated.*` — yalnız `exec` için |

Kurallar (doğrudan alıntı değeri taşıyanlar):

- **`deny` her zaman kazanır.** `allow` doluysa geri kalan her şey bloklu sayılır.
- **Tool policy sert duraktır**: `/exec` bile reddedilmiş `exec` tool'unu geri getiremez.
- Ve şu dürüstlük cümlesi: *"Tool policy tool'u **adına göre** filtreler; `exec`
  içindeki yan etkileri incelemez. `exec` serbestse, `write`/`edit`/`apply_patch`'i
  reddetmek shell komutlarını salt-okunur yapmaz."*

Son madde Atlas için kritik. "Yazma tool'unu kapattık, artık read-only" cümlesi
**yanlıştır** ve OpenClaw bunu belgeye yazmış. Atlas'ta salt-okunur bir rol
istiyorsan `group:runtime`'ı da kapatman gerekir, yoksa güvenlik tiyatrosu yaparsın.

**Atlas'a taşınacak:** 13 tool grubu (`group:fs`, `group:runtime`, `group:web`…)
şeklindeki kısayol sistemi. KKB'de bu `group:musteri-verisi`, `group:kredi-sorgu`,
`group:rapor`, `group:dis-erisim` olur. Rol tanımı bir tool listesi değil, birkaç
grup adı olur — ve bir tool eklendiğinde 40 rol dosyası güncellenmez.

**Ve bir teşhis komutu:** `openclaw sandbox explain --session ... --json` efektif
politikayı, nereden geldiğini ve düzeltilecek config anahtarını basıyor. Atlas'ta
bunun karşılığı olmazsa, "neden bloklandı" sorusu her seferinde bir mühendisin
yarım gününü yer.

### 2.2 Onay komuta değil, **plana** bağlanıyor

**Kaynak:** `docs/tools/exec-approvals.md`

Bu, belgedeki en teknik ve en kolay atlanan fikir. Naif bir onay akışı şudur:
kullanıcıya komut gösterilir → onaylar → komut çalışır. Arada TOCTOU boşluğu vardır.

OpenClaw bunu kapatıyor:

- Onay isteği **kanonik bir plan** taşıyor: `cwd`, tam `argv`, env binding, sabitlenmiş
  executable yolu.
- Onaylandıktan sonra çağrı **saklanan planı** yeniden kullanıyor, çağıranın sonradan
  gönderdiği alanları değil. `command`, `rawCommand`, `cwd`, `agentId` veya `sessionKey`
  değiştiyse → **approval mismatch**, reddedilir.
- Shell script / yorumlayıcı çağrılarında bir somut yerel dosya operandına da bağlanıyor:
  *"O dosya onaydan sonra ama çalıştırmadan önce değişirse, kaymış içeriği çalıştırmak
  yerine koşu reddedilir."*
- Ve yine bir sınır beyanı: *"Dosya bağlama en iyi çabadır, her yorumlayıcı yükleyici
  yolunun tam modeli değildir. Tam olarak bir somut dosya belirlenemiyorsa OpenClaw
  tam kapsama varmış gibi davranmak yerine onay-destekli koşu üretmeyi reddeder."*

Beş mod var: `deny` · `allowlist` · `ask` · `auto` · `full`. `auto` ilginç: allowlist
ıskalarını önce **otomatik bir gözden geçiriciye** yolluyor, o da çözemezse insana.

`strictInlineEval` ayrı bir incelik: `python -c`, `node -e`, `awk`, `find -exec`,
`xargs` gibi satır-içi eval biçimleri, yorumlayıcı ikilisi allowlist'te olsa bile
onaya zorlanıyor. Çünkü onlar tek bir kararlı dosya operandına eşlenmiyor.

**Atlas'a taşınacak:** Onay = **dondurulmuş plan**. KKB'de bu şu demek: "12345678901
TCKN'si için kredi notu sorgula" onayı, o TCKN'ye bağlanır. Onay alındıktan sonra
parametre değiştirilemez; değiştiyse istek düşer, sessizce yeni parametreyle koşmaz.
Bu, bizim `pipeline/gateway/approval.py`'de zaten yarısı var (id + tool + args),
ama argüman kayması kontrolü yok — eklenmeli.

### 2.3 Denetim kaydı: yalnız metadata, keyed pseudonym, ve açık sınır beyanı

**Kaynak:** `docs/gateway/audit.md` (284 satır — tek başına okunmaya değer)

Tasarım kararı: **denetim kaydı içerik tutmuyor.** Kimlik, sıra, köken, eylem,
durum, normalize sonuç kodu tutuyor. Şunları *asla* tutmuyor: prompt, mesaj gövdesi,
tool argümanı, tool sonucu, ek dosya, dosya adı, URL, komut çıktısı, ham hata metni.

Üç kayıt ailesi:

| Aile | Eylemler | Varsayılan |
|---|---|---|
| Ajan koşuları | `agent.run.started` / `finished` | açık |
| Tool eylemleri | `tool.action.started` / `finished` | açık |
| Mesajlar | `message.inbound.processed` / `message.outbound.finished` | **kapalı** |

Gizlilik modeli: platform kimlikleri ham saklanmıyor, kurulum-yerel keyed pseudonym
olarak çıkıyor — `hmac-sha256:v1:<keyId>:<digest>`. Anahtar tanımlayıcı türüne göre
domain-separated. Ve dürüstlük: *"Bu korelasyondur, anonimleştirme değildir: state
veritabanını okuyabilen anahtarı da okur ve aday ham kimlikleri pseudonym'lere karşı
test edebilir."*

Anahtar kaybolursa/bozulursa **fail closed**: yeni mesaj kayıtları düşürülür, sessizce
yeni anahtara dönülmez (dönülse korelasyon ikiye bölünürdü).

Ve bölümün adı: **"Kapsama ve kanıt sınırları"**:

> *"Bir satırın yokluğu hiçbir şey kanıtlamaz."* … *"Bu kayıt hata ayıklama ve
> operasyonel incelemeyi destekler. Kayıpsız bir uyum arşivi değildir; öyle bir şey
> gerekiyorsa OpenTelemetry ile beslenen harici bir sistem kullanın."*

**Atlas'a taşınacak — ama tersine çevrilerek.** OpenClaw'ın "kayıpsız uyum arşivi
değil" dediği yer, KKB'nin tam olarak **ihtiyacı olan** yer. Alınacak olan mekanizma
değil, **ayrım**: iki ayrı hat olmalı.

| Hat | Amaç | Özellik |
|---|---|---|
| Operasyonel kayıt | "ne oldu, neden yavaş" | best-effort, sınırlı, 30 gün, hot path dışında |
| Uyum arşivi | denetçiye gösterilen | kayıpsız, senkron, yazılamazsa **koşu düşer** |

OpenClaw'ın best-effort kuyruğu (kuyruk dolarsa kaydı düşürür, koşuyu asla iptal
etmez) Atlas'ın operasyonel hattı için doğru, uyum hattı için yanlıştır. Bu ayrımı
baştan yapmamak, sonradan "audit tablosunda neden boşluk var" toplantısıdır.

### 2.4 Yetki: "metot kapsamı yalnızca ilk kapı"

**Kaynak:** `docs/gateway/operator-scopes.md`

8 kapsam var: `operator.read`, `.write`, `.admin`, `.pairing`, `.approvals`,
`.questions`, `.talk`, `.talk.secrets`. Ama asıl fikir tabloda değil, şu başlıkta:
**"Method scope is only the first gate."**

- Kapsam **parametre-duyarlı** türetiliyor: `agent` metodu normal turlar için
  `operator.write` ister, ama `/new` veya `/reset` için `operator.admin`.
  `node.invoke` normalde `write`, ama `browser.proxy` / `fs.listDir` için `admin`.
- Sonra handler daha sıkı kontroller uyguluyor: `chat.send` write-scoped'dır, ama
  içindeki `/config set` komutu **çağıranın chat kapsamı ne olursa olsun**
  `operator.admin` ister.
- Yetki yükseltme yasağı: `device.pair.approve` `operator.pairing` ile erişilebilir,
  ama *"bir operatör cihazını onaylamak yalnızca çağıranın zaten sahip olduğu
  kapsamları basabilir veya koruyabilir."* Yani kimse kendinden fazlasını veremiyor.
- Eşleşmiş cihaz sessizce genişlemiyor: daha geniş rol isteyen yeniden bağlanma
  **yeni bir bekleyen yükseltme talebi** doğuruyor.

Ve bilinmeyen gelecekteki `operator.*` kapsamları, çağıranda `operator.admin` yoksa
**tam eşleşme** ister — yani yeni bir kapsam eklendiğinde eski token'lar onu
kendiliğinden kazanmaz.

**Atlas'a taşınacak:** Kapsamın **çağrının kendisinden** türetilmesi. "Bu kullanıcı
`sorgu.calistir` metodunu çağırabilir mi" yetmez; "bu kullanıcı *bu parametrelerle*
çağırabilir mi" gerekir. Ve yetki yükseltme yasağı (kimse kendinden fazlasını
onaylayamaz) — bunun testi ilk gün yazılmalı, sonradan eklenen bir kontrol değil.

### 2.5 Dış içerik güvenilmez içeriktir — ve bunun bir kodu var

**Kaynak:** `src/security/external-content.ts` (468 satır)

Prompt injection'a karşı çoğu yerde "sistem prompt'una uyar" yazılıp geçilir.
OpenClaw'da bu bir modül:

1. **Rastgele sınır işaretçisi.** Dış içerik `<<<EXTERNAL_UNTRUSTED_CONTENT id="<8 bayt hex>">>>`
   ile sarılıyor. Rastgele id'nin sebebi kodda yazılı: *"kötücül içeriğin sahte sınır
   işaretçisi enjekte ettiği spoofing saldırılarını önlemek."* Sabit bir etiket olsaydı,
   içerik kendi kapanış etiketini yazıp sarmalayıcıdan çıkardı.
2. **Güvenlik uyarısı** başa ekleniyor: veri sil / komut çalıştır / davranış değiştir /
   sır sızdır / üçüncü tarafa mesaj at talimatlarını yok say.
3. **Özel token temizliği.** 22 literal model kontrol token'ı siliniyor —
   `<|im_start|>`, `<|eot_id|>`, `[INST]`, `<<SYS>>`, `<start_of_turn>`, `<|channel|>`…
   artı `<|reserved_special_token_\d+|>` deseni. Yani dış içerik model konuşma
   şablonunu kıramıyor.
4. **Homoglif katlaması.** 28 Unicode açılı-ayraç eşleniği ASCII'ye katlanıyor
   (`＜`, `〈`, `«`, `❮`…) artı tam-genişlik harfler. Sınır işaretçisini Unicode
   benzeriyle taklit etme yolu kapalı.
5. **14 şüpheli desen** tespit ediliyor ("ignore previous instructions", "you are now a",
   `elevated\s*=\s*true`, `rm -rf`…) — ama **içerik yine de işleniyor**, sadece
   loglanıyor. Yani tespit bir engelleme değil, bir sinyal. Doğru karar: desen
   eşleştirmeyle injection engellenemez, ama izlenebilir.

Kaynak sınıfları kapalı bir küme: `email`, `webhook`, `api`, `browser`,
`channel_metadata`, `web_search`, `web_fetch`, `unknown`.

**Atlas'a taşınacak:** Aynısı, neredeyse birebir. KKB'de dış içerik = müşteri
e-postası, yüklenen PDF, dış API cevabı, web'den çekilen sayfa. Bunların hiçbiri
talimat değil, hepsi veri. Sınırın rastgele id'li olması ve özel token temizliği
30 satırlık iş, ama olmadığında sistemi ele geçiren şey tam olarak bu.

Bizim tarafta karşılığı yok — `pipeline/`'da dış içerik doğrudan bağlama giriyor.
Bu, kısa vadeli somut bir açık.

### 2.6 Kademeli açığa çıkarma — iki katmanda

**Kaynaklar:** `docs/tools/tool-search.md`, `docs/tools/skills.md`

İki ayrı yerde aynı fikir:

**Skill katmanında** (daha önce ölçtük): 74 skill kurulu, prompt'ta yalnızca indeks
var; gövde `read` ile çekiliyor. Ölçülen tasarruf **%93**.

**Tool katmanında** (Tool Search): büyük tool kataloğu prompt'a girmiyor. Model
sınırlı bir yetenek dizini görüyor, `search` → `describe` → `call` yapıyor.

Enterprise için kritik detaylar:

- Dizin **18.000 karakterle sınırlı**, tool adına göre sıralı, ve **cache sınırının
  üstüne** konuyor. Yani prompt KV-cache'i turlar arasında yeniden kullanılabiliyor.
  Kullanıcı mesajı, tur-başı tool tahminleri ve güvenilmeyen MCP metadata'sı dizine
  **girmiyor** — girse cache her turda bozulurdu.
- Güvenilmeyen MCP/istemci tool'larının **şemaları indekslenmiyor**; yalnız ad ve
  açıklama üzerinden eşleşiyorlar ve `input: "unknown"` olarak erteleniyorlar.
- Kod köprüsü izole bir Node alt sürecinde: **boş environment, dosya sistemi yok,
  ağ yok, alt süreç yok**, duvar-saati timeout'u (1.000–60.000 ms'e kıskaçlanıyor).
  Alt süreç plugin implementasyonlarını, MCP istemci nesnelerini veya **sırları
  tutmuyor** — her gerçek çağrı köprüden Gateway'e geri dönüyor, orada normal politika,
  onay, hook, log akışı işliyor.
- Ve fail-closed listesi açık: politika dışı tool arama sonucunda **çıkmamalı**.

**Atlas'a taşınacak:** KKB'nin iç API yüzeyi büyükse (ki büyüktür — sorgu, rapor,
limit, itiraz, KKB Anadolu, Findeks…), hepsini prompt'a koymak hem pahalı hem hatalı.
Ama asıl alınacak olan **cache sınırı disiplini**: değişmeyen şey (yetenek dizini,
sistem talimatı) sınırın üstünde, değişen şey (kullanıcı mesajı) altında. Bu, token
maliyetinde tek kalemde en büyük kazanç ve mimari bir karar — sonradan eklenmiyor.

### 2.7 Bellek: güvenlik sınırı **yazma yolunda**

**Kaynak:** `docs/concepts/memory-architecture.md`

Beş tasarım ilkesi var, üçü doğrudan Atlas'a yazılabilir:

> **"Yazma yolu güvenlik sınırıdır."** Belleğin içerik düzeyinde taranması zehirlenmiş
> olguları güvenilir biçimde yakalayamaz, bu yüzden OpenClaw yazma anında **köken**
> zorunlu kılar ve terfiyi yapısal olarak kapıya bağlar — sonradan kötü belleği tespit
> etmeye çalışmak yerine.

> **"Deterministik kapılar, içlerinde model yargısı."** Skorlama, eşikler, uygunluk,
> eşleşme ve yaşam döngüsü deterministik koddur. Dil modeli yalnızca gerçekten dil
> yargısı gerektiren yerde, deterministik kodun dayattığı sınırların **içinde** kullanılır.

> **"Hatalar cevabı asla bloklamaz."** Cevap yolundaki her bellek adımının bir timeout'u,
> bir geri düşüşü veya ikisi vardır. Çöken bir bellek altsistemi geri çağırma kalitesini
> düşürür; bir turu asla yemez.

Somut mekanizmalar:

- **Köken sınıfı kapalı bir küme**: `owner` / `agent` / `untrusted` / `system`. SQLite
  sütununda tutuluyor — *modelin düzyazıyla yazamayacağı* bir yerde. Sınıflandırma
  muhafazakâr: belirlenemeyen köken dışsalsa `untrusted` sayılır, **asla `owner`
  varsayılmaz**.
- **Oturum-türü kapısı**: cron, heartbeat ve alt-ajan oturumları kalıcı bellek adayı
  **üretmez**. İş çıktısı yazabilirler, ama hiçbiri terfiye uygun değildir.
- **Geri-çağırma döngüsü önleme**: bellekten bağlama enjekte edilmiş içerik yapısal
  olarak işaretlenir ve **yeni bellek olarak asla yeniden çıkarılmaz**. *"Yüz kez
  hatırlanan bir olgu tek bir olgu olarak kalır."*
- **Supersession key**: yeni gözlem eskisinin yanına birikmiyor, onu geçersiz kılıyor.

Beş katman: Instructions (yalnız insan yazar) · Curated core (kapılı konsolidasyon) ·
Episodic (aranabilir, hiç enjekte edilmez) · Prospective (yalnız tetiklenince) ·
Review (insan okuması için).

**Atlas'a taşınacak:** Bunun tamamı. Bir kurumsal asistanda "asistan bir müşteri
kaydından okuduğu şeyi kalıcı olgu sanıp başka bir müşteriye söyledi" senaryosu
şirketi mahkemeye götürür. Buna karşı savunma içerik taraması değil, **köken sınıfının
şemada zorunlu olması** ve `untrusted` içeriğin yapısal olarak terfi edememesidir.
Kapalı küme + "asla `owner` varsayma" kuralı doğrudan kopyalanmalı.

### 2.8 Sırlar: SecretRef + egress anında enjeksiyon

**Kaynak:** `docs/gateway/secrets.md` (767 satır)

- Sırlar bellekte bir **anlık görüntüye** çözülüyor — istek yolunda tembel çözülme yok.
  Böylece sır sağlayıcısının kesintisi sıcak yolun dışında kalıyor.
- Model sağlayıcı kimlik bilgileri için süreç-yerel opak bir **sentinel** basılıyor
  (`oc-sent-v1-...`). Auth deposu, stream seçenekleri, SDK yapılandırması, loglar ve
  hata nesneleri gerçek anahtarı değil sentinel'i görüyor. Gerçek değer isteğin
  süreçten çıkmasından hemen önce yerine konuyor.
- **Bilinmeyen sentinel-şekilli değerler ağ etkinliğinden önce fail-closed**: OpenClaw
  çözülmemiş bir sentinel'i sağlayıcıya iletmektense isteği göndermeyi reddediyor.
- Yine sınır beyanı: *"Sentinel'ler süreç izolasyonu değildir. Gerçek değer aynı
  süreçte bellekte vardır ve son adaptör sınırında görünür."*
- Ve göç bir **kapı** olarak tanımlanmış: `openclaw secrets audit --check` temiz
  değilse göç bitmemiştir. Config, SQLite auth deposu, `.env` ve üretilmiş
  `models.json` dosyalarındaki düz metin kalıntısı ayrıca temizlenmeli.

**Atlas'a taşınacak:** İki şey. (1) Sırların ajanın okuyabildiği dosyalarda düz metin
durmaması — çünkü ajanın `read`/`exec` yetkisi varsa API düzeyi redaksiyon anlamsızdır,
bu belgede açıkça yazıyor. (2) **Göçün bir denetim komutuyla bitmiş sayılması**. "Sırları
vault'a taşıdık" cümlesi, `audit --check` temiz çıkmadan doğru değildir.

Bizim tarafta doğrudan yakıcı: bugün `play.py` çalıştırırken gördüğümüz gibi anahtarlar
`.env`'de düz metin. `.gitignore`'da olması sızıntıyı değil, yalnız *commit* sızıntısını
engelliyor.

### 2.9 Dayanıklılık: cooldown, döngü kırıcı, compaction sonrası nöbetçi

**Kaynaklar:** `docs/concepts/model-failover.md`, `docs/tools/loop-detection.md`

- **Failover**: profil rotasyonu, auth-hatası atlama önbelleği, cooldown'lar, faturalama
  kaynaklı devre dışı bırakmalar — ve **oturum yapışkanlığı** (session stickiness),
  açıkça "cache-friendly" gerekçesiyle. Yani model değiştirmek ucuz değil, prompt
  cache'ini yakıyor; bu yüzden gereksiz yere değiştirilmiyor.
- **Döngü tespiti** varsayılan **kapalı** (`enabled: false`) — küçük modeller için
  açılması öneriliyor, flagship modeller için gerekmiyor.
- **Compaction sonrası nöbetçi** ise `enabled` açıkça `false` yapılmadıkça **açık**.
  Bağlam taşması → compaction → aynı döngü zincirini kırmak için, compaction-retry'den
  sonra kısa pencereli bir nöbetçi kuruluyor: aynı `(tool, args, result)` üçlüsü
  tekrarlanırsa koşu iptal ediliyor.
- İnce ayar: `exec` için ilerleme-yok hash'i kararlı sonuçları (status, exit code,
  timeout bayrağı, çıktı) karşılaştırıyor, **oynak metadata'yı** (süre, PID, session
  id, çalışma dizini) yok sayıyor. Giden mesajlarda ise tersi: oynak id'ler
  (message id, timestamp) **çıkarılıyor** ki iki farklı "gönderildi" sonucu birbirinin
  aynısı görünmesin.

**Atlas'a taşınacak:** Özellikle compaction sonrası nöbetçi. Uzun kurumsal oturumlarda
en pahalı hata modu sonsuz döngüdür ve fatura oradan patlar. İki ayarın varsayılanının
**farklı** olması da bilinçli bir karar: agresif olan kapalı, ucuz ve yüksek getirili
olan açık.

### 2.10 Gözlemlenebilirlik: içerik varsayılan **kapalı**

**Kaynak:** `docs/gateway/opentelemetry.md` (753 satır)

> *"Ham model/tool içeriği varsayılan olarak dışa aktarılmaz."*

Span'ler sınırlı tanımlayıcılar taşıyor: kanal, sağlayıcı, model, hata kategorisi,
**yalnız-hash istek id'leri**, tool kaynağı, skill adı. Prompt metni, cevap metni,
tool girdisi/çıktısı, skill dosya yolları, session key'ler **yok**. `agent:` ile
başlayan session-key benzeri değerler düşük-kardinaliteli özniteliklerde `unknown`
ile değiştiriliyor.

Ama ölçüm yine de zengin — çünkü metin yerine **boyut** aktarılıyor:
`prompt.input_messages_count`, `input_messages_chars`, `system_prompt_chars`,
`tool_definitions_count`, `tool_definitions_chars`, `total_chars`,
`request_bytes`, `response_bytes`, `time_to_first_byte_ms`.

**Atlas'a taşınacak:** Bu tam olarak KKB'nin isteyeceği şey. "Prompt'ta ne kadar
sistem talimatı, ne kadar tool tanımı vardı" sorusunu **prompt'un kendisini
saklamadan** cevaplıyor. Maliyet analizi, cache verimliliği ve regresyon takibi
bu sayılarla yapılabiliyor; PII hiç ayrılmıyor. Bir bankada telemetri sistemine
prompt metni akıtmak, veri sınıflandırma politikasını sessizce delen en yaygın yoldur.

Ek: `usage-tracking` tarafında maliyet iki ayrı yüzeyden okunuyor — sağlayıcının
kendi kota/faturalama API'si ve OpenClaw'ın kendi oturum-türevli tahmini — ve
belge *"bu ikisi bilerek farklı soruları cevaplar, birleştirilmez"* diyor. Atlas'ta
departman bazlı geri-faturalama (chargeback) yapılacaksa bu ayrım baştan kurulmalı.

### 2.11 Tedarik zinciri: iç yetenek kayıt defteri

**Kaynak:** `docs/security/THREAT-MODEL-ATLAS.md` §2.1 (Trust Boundary 5)

ClawHub (skill pazaryeri) için savunma katmanları: semver + zorunlu `SKILL.md`,
statik desen + AST-komşusu moderasyon taraması, LLM tabanlı ajanik risk incelemesi,
VirusTotal taraması, **GitHub hesap yaşı doğrulaması (14 gün)**.

**Atlas'a taşınacak — ama içeri çevrilerek.** KKB'de public bir pazaryeri olmaz;
olacak olan **iç yetenek kayıt defteri**: bir ekibin yazdığı skill/tool'un başka
ekiplerin asistanına ulaşması. O zaman sorular aynı: kim yayınladı, hangi sürüm,
gözden geçirildi mi, hangi tool gruplarına dokunuyor, geri alınabilir mi. Bunu
kurmadan "herkes kendi skill'ini yazsın" demek, iç kaynaklı bir tedarik zinciri
riski üretir. Yetenek dağıtımı bir **onay iş akışı** olmalı, bir dosya kopyalama değil.

### 2.12 Tehdit modeli yaşayan bir belge

**Kaynak:** `docs/security/THREAT-MODEL-ATLAS.md` (561 satır)

MITRE ATLAS taktiklerine göre düzenlenmiş, her tehdit için sabit bir tablo şeması:
ATLAS ID · açıklama · saldırı vektörü · etkilenen bileşenler · **mevcut azaltmalar** ·
**artık risk** · öneriler. Beş güven sınırı ve altı veri akışı diyagramda tanımlı.
Ve katkı için ayrı bir belge var (`CONTRIBUTING-THREAT-MODEL.md`) — yani belge
canlı tutuluyor.

Dikkat çeken dürüstlük: bazı tehditlerde "Mevcut azaltmalar: **Yok**" yazıyor
(T-RECON-002). Bilinen ama kapatılmamış riskler gizlenmiyor, "artık risk: düşük"
gerekçesiyle yazılıyor.

**Atlas'a taşınacak:** Bu tablo şeması, ilk gün. KKB'de zaten bir güvenlik komitesi
vardır ve o komiteye gidilecek belge budur. "Artık risk" sütununun boş bırakılmaması,
denetimde en çok işe yarayan alışkanlık.

---

## 3. Alınmayacaklar (ve KKB'de neden yetmez)

Bu bölüm §2'den daha önemli olabilir. OpenClaw'ın kendi belgeleri bu sınırları
söylüyor; onları görmezden gelip mekanizmayı kopyalamak, olmayan bir güvenceyi
varsaymak olur.

| Ne | OpenClaw ne diyor | Atlas'ta neden yetmez |
|---|---|---|
| `operator.read` ile veri ayrımı | *"düşmanca çok-kiracılı izolasyon sınırı değildir"* — aynı operatör alanındaki her istemci denetim verisini alabilir | KKB'de departmanlar birbirinin sorgusunu görmemeli. Gerçek per-user authz gerekir; OpenClaw'ın cevabı "**ayrı gateway çalıştırın**" |
| Multi-user sahiplik/presence | *"kullanılabilirlik özellikleridir, güvenlik sınırı değil"* · *"Bir ajanı işletebilen herkes o ajanın yapabildiği her şeyi yaptırabilir"* | Sahiplik avatarı yetki değildir. Atlas'ta kullanıcı kimliği **tool politikasına** girmeli, yalnız arayüze değil |
| Denetim kaydı | *"kayıpsız bir uyum arşivi değildir"* · *"bir satırın yokluğu hiçbir şey kanıtlamaz"* | Denetçi "kayıp olabilir" cevabını kabul etmez. §2.3'teki ikinci hat şart |
| Exec approvals | *"per-user auth sınırı veya salt-okunur dosya politikası değildir"* | Onay, kazara çalıştırmayı azaltır; kötü niyetli operatörü durdurmaz |
| SecretRef sentinel | *"süreç izolasyonu değildir"* | Gerçek değer aynı süreçte. KKB'de HSM/vault + süreç ayrımı ayrıca gerekir |
| Sandbox `docker.binds` | *"`/var/run/docker.sock` bağlamak sandbox'a host kontrolünü verir"* | Bind'ler sandbox'ı deler. Varsayılan `:ro`, ve bind listesi gözden geçirme konusu olmalı |
| 161 extension / sohbet kanalları | — | WhatsApp/Telegram/Discord bir kurumsal asistanın yüzeyi değil. Kanal = kurum içi kimlikli arayüz. Bu 161'in ~155'i Atlas'ta yok |
| Public ClawHub | — | Dış kaynaklı skill kurulumu KKB'de doğrudan tedarik zinciri riski. §2.11'deki iç kayıt defteri ile değiştirilir |

Bir cümlelik özet: **OpenClaw tek bir güvenilen operatörün etrafında tasarlanmış.**
Atlas çok kullanıcılı ve karşılıklı güvenmeyen departmanlar içerecek. Mekanizmalar
taşınır, güven modeli taşınmaz — yeniden kurulmalı.

---

## 4. Atlas'ın şekli

OpenClaw'dan alınanlarla, KKB'ye göre yeniden kurulmuş güven modeliyle:

```
  Kullanıcı (kurumsal SSO kimliği)
        │
   ═════╪═══════ SINIR 1 — Kimlik.  Gerçek per-user authz.
        │         (OpenClaw'ın "ayrı gateway" cevabı yerine)
        ▼
   ┌─────────────────────────────────────────────┐
   │ KONTROL DÜZLEMİ                              │
   │  · kapsam parametreden türetilir  (§2.4)     │
   │  · tool grupları = rol tanımı     (§2.1)     │
   │  · onay = dondurulmuş plan        (§2.2)     │
   └─────────────────────────────────────────────┘
        │
   ═════╪═══════ SINIR 2 — Yetki.  deny kazanır; allow doluysa gerisi kapalı.
        ▼
   ┌─────────────────────────────────────────────┐
   │ AJAN DÖNGÜSÜ  (AutoGen — bizde zaten var)    │
   │  · yetenek dizini cache sınırının üstünde    │
   │  · compaction sonrası nöbetçi     (§2.9)     │
   └─────────────────────────────────────────────┘
        │                              │
   ═════╪═══ SINIR 3 — Yürütme    ═════╪═══ SINIR 4 — Dış içerik
        ▼                              ▼
   ┌──────────────┐            ┌────────────────────────┐
   │ Tool / API   │            │ rastgele-id sarmalama  │
   │ sandbox      │            │ özel token temizliği   │
   └──────────────┘            │ homoglif katlaması     │
        │                      └────────────────────────┘
        ▼
   ┌─────────────────────────────────────────────┐
   │ İKİ AYRI KAYIT HATTI          (§2.3)         │
   │  operasyonel: best-effort, metadata, 30 gün  │
   │  uyum       : kayıpsız, senkron, fail-closed │
   └─────────────────────────────────────────────┘
        │
        ▼  telemetri: içerik yok, boyut var  (§2.10)
```

Bellek (§2.7) bu diyagramda ayrı bir kutu değil, **her sınırdan geçen bir sütun**:
her yazılan olgu köken sınıfını taşır, ve `untrusted` olan hiçbir şey terfi edemez.

---

## 5. İlk üç iş

Sıra, "en çok korur / en az maliyetli"e göre. Üçü de bizim mevcut `pipeline/`'a
prototiplenebilir.

**1. Dış içerik sarmalayıcı.** (~1 gün) `src/security/external-content.ts`'in
Python karşılığı: rastgele id'li sınır, 22 özel token temizliği, homoglif katlaması,
14 desenin loglanması. En küçük iş, en büyük tekil koruma. Şu an bizde **hiç yok** —
`docs_index` sonuçları ve dış kaynak metinleri bağlama düz giriyor.

**2. Onayı plana bağlama.** (~1–2 gün) `pipeline/gateway/approval.py` bugün onay
id'sini ve tool adını tutuyor; argümanların **kanonik hash'ini** tutmuyor. Onay
verildikten sonra argüman değişirse şu an fark edilmez. `argv` hash'i + mismatch
reddi eklenecek. Testi kolay: onayla, argümanı değiştir, reddedilmeli.

**3. İki hatlı kayıt ayrımı.** (~2–3 gün) Bugünkü kayıtlarımız tek hat. Ayrım
şimdi yapılırsa ucuz, sonradan yapılırsa şema göçü. Uyum hattının tek sert kuralı:
**yazılamazsa koşu düşer** — best-effort değil.

Bunların ardından sırada §2.6 (cache sınırı disiplini — maliyet kazancı) ve
§2.7 (bellek köken sınıfı — şema kararı, geç kalınırsa pahalı) var.

---

## 6. Ölçüm künyesi

Bu belgedeki her iddianın kaynağı. Repo yolu `~/Desktop/adapted/harnesses/openclaw`,
commit `01cc7106`.

| § | Kaynak dosya | Satır |
|---|---|---|
| 2.1 | `docs/gateway/sandbox-vs-tool-policy-vs-elevated.md` | 154 |
| 2.2 | `docs/tools/exec-approvals.md` | 545 |
| 2.3 | `docs/gateway/audit.md` | 284 |
| 2.4 | `docs/gateway/operator-scopes.md` | 162 |
| 2.5 | `src/security/external-content.ts` | 468 |
| 2.6 | `docs/tools/tool-search.md` | 432 |
| 2.7 | `docs/concepts/memory-architecture.md` | 425 |
| 2.8 | `docs/gateway/secrets.md` | 767 |
| 2.9 | `docs/concepts/model-failover.md`, `docs/tools/loop-detection.md` | 362 + 158 |
| 2.10 | `docs/gateway/opentelemetry.md`, `docs/concepts/usage-tracking.md` | 753 + 352 |
| 2.11–2.12 | `docs/security/THREAT-MODEL-ATLAS.md` | 561 |
| 3 | `docs/concepts/multi-user.md` + yukarıdakilerin sınır beyanları | 45 |

Sayımlar doğrudan kaynaktan: 14 şüpheli desen, 22 özel token literali, 28 homoglif
eşlemesi, 13 tool grubu, 8 operatör kapsamı, 161 extension, 22 paket, 764 belge.

**Ölçmediklerim (dürüstlük payı):** Çalışma zamanı davranışını bu turda yeniden
ölçmedim — §2.6'daki %93 skill tasarrufu ve 351/44/74 sayıları önceki turun canlı
RPC ölçümünden geliyor (`docs/pdf/openclaw-ici.pdf`), bu commit'te yeniden
doğrulanmadı. Performans iddiası hiç yok; hiçbir benchmark koşmadım.
