# 17 — AutoGen ve OpenClaw'dan şirket içinde nasıl yararlanılır

**Varsayım:** Atlas, düzenlenmiş bir finansal veri kurumunun (KKB) iç kullanıcılarına
hizmet edecek kurumsal AI asistanı. Çok kullanıcılı, kimlik doğrulamalı, PII'ye
dokunan, denetlenebilirliği zorunlu. `docs/16` bu varsayımla yazıldı; bu belge onun
devamı ve **karar belgesi** — orada "neyi ilham alalım", burada "ne yapalım".

---

## 0. Soru üç ayrı soru

"AutoGen ve OpenClaw'dan nasıl yararlanırız" tek bir soru gibi duruyor. Değil. Bir
açık kaynak sistemden üç farklı şekilde yararlanılır ve üçünün riski, maliyeti,
geri dönüşü farklıdır:

| biçim | ne demek | ne zaman doğru |
|---|---|---|
| **KULLAN** | Sistemi olduğu gibi konuşlandır | Güven modeli sana uyuyorsa |
| **GÖM** | Kütüphane olarak kendi uygulamana al | Sen kontrol düzlemini yazacaksan |
| **ÖĞREN** | Tasarım kararlarını al, kodu alma | Güven modeli uymuyor ama kararlar iyi |

Cevap bileşene göre değişiyor, ve üç ayrı cevap çıkıyor:

> **AutoGen'i GÖM. OpenClaw'dan ÖĞREN. OpenClaw'ı ayrıca — Atlas olarak değil,
> mühendislik ekibinin kendi aracı olarak — KULLAN.**

Geri kalan bölümler bu üç cümlenin her birinin gerekçesi.

---

## 1. Ölçülen zemin

Karar vermeden önce doğrulanması gereken dört şey vardı. Dördü de ölçüldü:

| soru | cevap | nereden |
|---|---|---|
| Lisans engeli var mı? | **Yok.** İkisi de MIT | `LICENSE`, `package.json:16`, `pip show autogen-core` |
| Veriyi içeride tutabilir miyiz? | **Evet** — vLLM / LM Studio / Ollama / herhangi bir OpenAI uyumlu endpoint | `docs/gateway/local-models.md` |
| Telemetri varsayılan açık mı? | **Hayır** — `otel.enabled` varsayılan `false` | `configuration-reference.md:1331` |
| Kendi altyapımızda koşar mı? | **Evet** — `docker-compose.yml`, gateway + CLI servisleri | repo kökü |

**MIT olması hukuki engeli kaldırıyor** ama kararı vermiyor. MIT, "kullanabilirsin"
der; "kullanmalısın" demez. Aşağıdaki üç bölüm asıl kararı veriyor.

### Ve bir sert maliyet gerçeği

On-prem model çalıştırmak istiyorsan — ki bir kredi bürosunda bu muhtemelen
pazarlık konusu değil — OpenClaw'ın kendi belgesi donanım tabanını yazıyor:

> *"Rahat bir ajan döngüsü için **2+ tam donanımlı Mac Studio ya da eşdeğer bir GPU
> rig (~30.000 $+)** hedefleyin. Tek bir **24 GB** GPU yalnız hafif istemleri, daha
> yüksek gecikmeyle kaldırır."*

Ve bir uyarı daha, güvenlik açısından daha önemli:

> *"Küçük veya agresif kuantize edilmiş kontrol noktaları **prompt injection riskini
> yükseltir**"* — çünkü sağlayıcı tarafındaki güvenlik filtrelerini atlarlar.

**Karar için anlamı:** "veri dışarı çıkmasın" ile "küçük model kullanalım" aynı
cümlede duramaz. Veriyi içeride tutmanın bedeli ciddi donanım; bedeli ödemezsen
elde ettiğin şey daha güvenli değil, **daha savunmasız** bir sistem olur.

---

## 2. AutoGen'i GÖM

### Neden göm

AutoGen bir uygulama değil, bir kütüphane. Kendi Python sürecinde koşuyor, kendi
kodunla çağırıyorsun, veri senin sürecinden çıkmıyor. Bir kurumsal uygulamaya
gömülmesi için ek bir mimari karara gerek yok — normal bir bağımlılık.

Verdiği şey somut: aktör modeli, beş takım tipi, GraphFlow ile eşzamanlı boru hattı,
on bir sonlandırma koşulu, yapılandırılmış olay akışı (maliyet muhasebesi bunun
üstüne kuruluyor), MCP entegrasyonu.

Bu depoda **çalışıyor**: 214+ test, GraphFlow tabanlı tarama, akış, kapı entegrasyonu.
Yani "çalışır mı" sorusu bizde zaten cevaplanmış durumda.

### Bakım modu riski ve karşılığı

`docs/16` ve slaytlarda `teyitsiz` etiketiyle duran bir okuma var: AutoGen aktif
özellik geliştirmeden bakım moduna geçiyor, enerji Microsoft'un MAF'ına kayıyor.
Bu bir duyuru değil, bir okuma — ama karar verirken hesaba katılmalı.

**Karşılığı mimari:** motor katmanını **ince ve arayüz arkasında** tut.

```
Atlas uygulama katmanı
        │
   ── ENGINE arayüzü ──      ← burası bizim, dar ve stabil
        │
   AutoGen (bugün)  ·  MAF / LangGraph / düz kod (yarın)
```

Bu depodaki dizilim zaten buna yakın: kapı, politika, kayıt ve huni **bizim**;
AutoGen yalnız ajan döngüsünü ve graf koşumunu yapıyor. Motor değişirse kontrol
düzlemi ayakta kalır.

**Pratik kural:** `autogen_*` import'ları uygulamanın her yerine değil, iki üç
modüle sıkışsın. Bugün ölçülebilir bir disiplin, yarın bir sigorta.

### Ne zaman gömmemeli

Adım sayısı önceden belliyse ajan gerekmiyor — `if`/`for` yeter. Kurumsal iş
akışlarının çoğu bu kategoride. AutoGen'i "her yere" koymak, çözmediği bir problemi
pahalıya çözmek olur.

---

## 3. OpenClaw'dan ÖĞREN — ama Atlas olarak konuşlandırma

### Neden konuşlandırmamalı

Tek bir cümleye iniyor ve bu cümle OpenClaw'ın **kendi belgelerinden**:

> `operator.read` *"düşmanca çok-kiracılı bir izolasyon sınırı değildir."*
> Çok kullanıcı sahipliği *"bir kullanılabilirlik özelliğidir, güvenlik sınırı değil."*
> Denetim kaydı *"kayıpsız bir uyum arşivi değildir."*

OpenClaw **tek bir güvenilen operatörün** etrafında tasarlanmış. O modelde herkes
güvenilirdir, dolayısıyla ayrımlar birer kolaylıktır. KKB'de departmanlar birbirine
karşı da sınırlıdır — aynı ayrımlar orada birer açık olur.

OpenClaw'ın kendi cevabı da bu: gerçek ayrım gerekiyorsa **ayrı gateway'ler**
çalıştırın. Bunu departman sayısınca yapmak bir mimari değil, bir kaçınmadır.

Buna ek olarak: 161 extension'ın büyük çoğunluğu (WhatsApp, Telegram, Discord,
Signal, iMessage) bir kurumsal asistanın yüzeyi değil. Konuşlandırılan her extension
bakılması gereken bir yüzey demek.

### Neyi öğren

`docs/16` bunu ayrıntısıyla veriyor; karar açısından en yüksek getirili altısı:

| ne | tek cümlede | Atlas'taki karşılığı |
|---|---|---|
| Üç kontrol ekseni | Sandbox / tool policy / elevated ayrı sorulardır | "neden bloklandı" tek soruya tek cevap |
| Onay = donmuş plan | Onaylanan argüman değişemez | onaylanan TCKN'yle koşulur, başkasıyla değil |
| İçeriksiz denetim | Kayıt metadata tutar, prompt tutmaz | PII log altyapısına girmez |
| Dış içerik sarmalayıcı | Rastgele id'li sınır + token temizliği | müşteri PDF'i talimat değildir |
| Bellek kökeni | Köken şemada zorunlu, kapalı küme | `untrusted` olan terfi edemez |
| Yumuşak ≠ sert | Prompt niyet, politika kontroldür | denetime götürülen ikinci liste |

### Ve iki dürüstlük alışkanlığı

Mekanizmalardan daha kıymetli olan iki şey var ve ikisi de bedava:

1. **Her mekanizmanın yanına neyi kanıtlamadığını yaz.** OpenClaw'ın belgeleri
   bunu sürekli yapıyor. Bir denetim toplantısında "anonim" dediğin şeyin anonim
   olmadığı ortaya çıkarsa bütün kaydın güvenilirliği gider.
2. **Tehdit modelini yaşayan bir tablo olarak tut.** `THREAT-MODEL-ATLAS.md`'nin
   şeması: tehdit · vektör · etkilenen bileşen · mevcut azaltma · **artık risk** ·
   öneri. Bazı satırlarda "mevcut azaltma: yok" yazıyor ve gizlenmiyor.

---

## 4. OpenClaw'ı KULLAN — ama başka bir işte

Bu, en çok atlanan seçenek. OpenClaw Atlas olamaz, ama **mühendislik ekibinin kendi
aracı** olarak bugün kurulabilir — Claude Code'un kurulduğu gibi.

| neden düşük riskli | |
|---|---|
| Kullanıcılar | Birbirine güvenen küçük bir mühendislik ekibi — tek-operatör modeli **burada geçerli** |
| Veri | Üretim müşteri verisi değil; kod, doküman, iç araçlar |
| Getiri | Ekip, Atlas'ta kuracağı mekanizmaları **çalışırken** görür |

Ve asıl kazanç şu: OpenClaw bir **referans implementasyon ve ölçüm hedefi** olur.
"Onay akışımız doğru mu" sorusunun cevabı, çalışan bir onay akışını yanına koyunca
çok daha hızlı bulunuyor. Bu depoda ölçtüğümüz her sayı (351 metot, 44 tool,
%93 skill tasarrufu) böyle çıktı — belgeden okuyarak değil, **çalışan sisteme
sorarak**.

**Sınır:** kurulduğu makine, üretim verisine erişimi olmayan bir geliştirme ortamı
olmalı; ve `~/.openclaw/` sır tutuyor, o makine buna göre sınıflandırılmalı.

---

## 5. Ortaya çıkan mimari

```
   Kullanıcı · kurumsal SSO
        │
   ═════╪═══ SINIR 1 — kimlik ────── BİZİM. Gerçek per-user authz.
        ▼
   ┌────────────────────────────────────────────┐
   │ KONTROL DÜZLEMİ            ← OpenClaw'dan  │
   │  kapsam parametreden       ← ÖĞRENİLDİ,    │
   │  rol = yetenek grubu          kod bizim    │
   │  onay = donmuş plan                        │
   └────────────────────────────────────────────┘
        │
   ═════╪═══ SINIR 2 — yetki ─────── deny kazanır
        ▼
   ┌────────────────────────────────────────────┐
   │ MOTOR arayüzü              ← BİZİM (ince)  │
   │   └─ AutoGen               ← GÖMÜLDÜ       │
   └────────────────────────────────────────────┘
        │                    │
   ═════╪═ SINIR 3 ═════════╪═ SINIR 4 — dış içerik
        ▼                    ▼
   iç API / tool         sarmalayıcı (öğrenildi, kod bizim)
        │
        ▼
   İKİ KAYIT HATTI: operasyonel (best-effort) + uyum (kayıpsız, fail-closed)
```

Üç renk var ve karışmamalı:
**BİZİM** (kimlik, kontrol düzlemi, motor arayüzü, kayıt) ·
**GÖMÜLDÜ** (AutoGen) ·
**ÖĞRENİLDİ** (OpenClaw'ın kararları, kendi kodumuzla).

OpenClaw kodu bu diyagramda **hiç yok**. Bu bilinçli.

---

## 6. Doksan günlük plan

Her faz kendi başına değer üretir; bir sonraki iptal olsa bile öncekiler ayakta kalır.

### Faz 1 — Zemin (0–30 gün)

| iş | çıktı | ölçütü |
|---|---|---|
| OpenClaw'ı mühendislik ortamına kur | Ekip aracı + referans implementasyon | Ekip haftada kullanıyor |
| Dış içerik sarmalayıcı | Python modülü | Sahte kapanış etiketi ve özel token etkisiz |
| Motor arayüzünü çiz | `autogen_*` import'ları 2–3 modülde | Kalan modüllerde sıfır import |
| Tehdit modeli tablosu v0 | Güvenlik komitesine gidecek belge | "Artık risk" sütunu dolu |

### Faz 2 — Kontrol düzlemi (30–60 gün)

| iş | çıktı | ölçütü |
|---|---|---|
| Yetenek grupları | `group:musteri-verisi`, `group:kredi-sorgu`… | Rol = ≤3 grup adı |
| Onayı plana bağla | Argüman hash'i + mismatch reddi | Onayla → argümanı değiştir → **reddedilmeli** |
| İki hatlı kayıt | Operasyonel + uyum ayrımı | Uyum hattı yazılamazsa **koşu düşer** |
| Kapsamı parametreden türet | Metot değil, çağrı yetkilendiriliyor | Aynı metot, farklı parametre, farklı karar |

### Faz 3 — Pilot (60–90 gün)

| iş | çıktı | ölçütü |
|---|---|---|
| Tek departmanla dar pilot | Sınırlı tool seti, gerçek kullanıcılar | Denetim kaydı denetçiye gösterilebiliyor |
| Bellek köken şeması | Kapalı küme, SQLite sütunu | `untrusted` terfi edemiyor (testli) |
| Cache sınırı disiplini | Sabit önek ayrıldı | Token/tur ölçülüyor ve düşüyor |
| On-prem model kararı | Donanım ya da onaylı dış sağlayıcı | §1'deki maliyet açıkça kabul edilmiş |

**Pilotun kapsamı dışı (bilerek):** çok departmanlı yayılım, yazma yetkisi olan
tool'lar, otomatik zamanlanmış koşular. Üçü de kontrol düzlemi kanıtlanmadan
açılmamalı.

---

## 7. Riskler

| risk | neden gerçek | karşılık |
|---|---|---|
| AutoGen bakım modu | `teyitsiz` ama makul bir okuma | Motor arayüz arkasında, import'lar sıkışık |
| On-prem maliyeti kabul edilmez | ~30.000 $+ donanım tabanı ölçülü | Alternatif: sözleşmeli, veri-işlemesiz dış sağlayıcı. Küçük model **çözüm değil** |
| Kontrol düzlemi "sonra" ertelenir | En yaygın hata | Faz 2 pilotun **önkoşulu**, paralel iş değil |
| OpenClaw ekip aracı üretim verisine bulaşır | `~/.openclaw/` sır tutuyor | Ayrı ortam, ayrı sınıflandırma, üretim erişimi yok |
| "Prompt'a yazdık, kontrol var" | Belgede açıkça yumuşak yönlendirme deniyor | Her kural iki kutuya ayrılır: niyet / kontrol |
| Yalnız mekanizma kopyalanır, sınır atlanır | OpenClaw'ın sınırları kendi belgelerinde | `docs/16` §3 tablosu gözden geçirme listesi olur |

---

## 8. Karar için hâlâ bilmediklerim

Bu belge açıkça belirtilmiş varsayımlar üstünde duruyor. Dördü doğrulanırsa plan
sağlamlaşır; yanlış çıkarsa değişecek bölümler yanlarında yazılı:

1. **Atlas'ın gerçek kullanıcı ve departman sayısı** — §3'teki "tek-operatör modeli
   uymaz" gerekçesinin ağırlığını belirler.
2. **Veri yerleşimi zorunluluğu ne kadar sert** — "yurt içi" mi, "kurum içi" mi,
   "süreç içi" mi? §1'in maliyet tablosu buna bağlı.
3. **Mevcut kimlik altyapısı** (SSO, dizin, yetki modeli) — §5'teki SINIR 1'in ne
   kadarının hazır olduğunu belirler.
4. **İlk kullanım senaryosu** — okuma ağırlıklı bir senaryo (rapor, sorgu özeti)
   pilotu ciddi biçimde kolaylaştırır; yazma yetkisi gerektiren bir senaryo Faz 2'yi
   önkoşul olmaktan çıkarıp **zorunluluk** yapar.

---

## Tek cümlelik özet

> **AutoGen'i ince bir arayüz arkasına göm; OpenClaw'ı Atlas olarak kurma ama
> kararlarını al ve ekip aracı olarak kullanarak çalışırken izle; kontrol düzlemini
> kendin yaz ve onu pilotun önkoşulu say.**

İlgili belgeler: `docs/16` (ne alınır/alınmaz, ayrıntısıyla) · `docs/15` (bu deponun
gateway mimarisi) · `docs/pdf/ogretici.pdf` (mekanizmaların nasıl kurulduğu) ·
`docs/pdf/slaytlar.pdf` (anlatım destesi).
