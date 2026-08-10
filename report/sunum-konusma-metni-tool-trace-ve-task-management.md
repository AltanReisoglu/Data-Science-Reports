# Sunum Konuşma Metni — Tool-Trace Compaction + Task Management

> **Ne bu belge:** Sunumda **ağzından çıkacak cümleler**. Her slayt için "de ki:" bloğu = doğrudan
> okunabilecek konuşma; "geçiş / vurgu" = sahne notu. İki konuyu tek akışta anlatır.
> **Süre:** ~22–25 dk (Bölüm 1 ~10 dk, Bölüm 2 ~12 dk, +demo/soru). İstersen 12 dk'ya da sıkışır (★ ile işaretli slaytlar çekirdek).
> **Tek cümlelik tez:** *"İki katman var — işin İÇİNİ yönetmek (tool-trace) ve işin KENDİSİNİ yönetmek (task management). İkisini de gerçek kodla ölçtük."*

---

## AÇILIŞ (★ ~1.5 dk)

**De ki:**
> "Bugün iki şey anlatacağım, ikisi de bir ajanın uzun süre ayakta kalması için şart. Bir ajanın iki
> derdi var: **birincisi, konuşma uzadıkça context şişer** — buna *tool-trace compaction* diyoruz, işin
> *içini* yönetmek. **İkincisi, worker çökerse iş kaybolmasın** — buna *task management* diyoruz, işin
> *kendisini* yönetmek. İkisini de sadece okuyup geçmedim; **gerçek framework'lerle POC yazıp ölçtüm**,
> rakamları göstereceğim."

**Geçiş:** "Önce içeriyi, sonra dışarıyı. Başlayalım."

---

# BÖLÜM 1 — TOOL-TRACE COMPACTION (işin içini yönetmek)

## 1.1 Problem (★ ~1.5 dk)

**De ki:**
> "Bir ajan tool çağırır — dosya okur, web'den çeker, komut koşar. Her tool çıktısı context'e eklenir.
> Bir `WebFetch` 90 KB dönebilir; birkaç büyük çıktıdan sonra pencere dolar. Dolunca model ya durur ya
> da eski bilgiyi unutur. **Demek ki tool çıktılarını akıllıca küçültmemiz lazım — ama işe yarayan
> bilgiyi kaybetmeden.** İşte tool-trace compaction bu."

**Vurgu:** "Kritik nokta: rastgele kırpmıyoruz; **hangi bilginin taşınacağına** karar veriyoruz."

## 1.2 İki ekol (★ ~2 dk)

**De ki:**
> "Sistemler bunu iki şekilde yapıyor. **Birincisi deterministik:** LLM kullanmadan, kurallarla
> kırpıyorlar — büyük çıktıyı diske döküp yerine kısa bir referans koymak gibi. Hızlı, ucuz,
> öngörülebilir. **İkincisi LLM-özet:** eski tool trace'ini bir modele verip 'bunu 3 cümlede özetle'
> diyorlar. Daha akıllı ama bir LLM çağrısı maliyeti var. Bazıları ikisini birden kullanıyor — hibrit."

**Geçiş:** "Ben sekiz sistemi inceledim. Hepsini tek tek değil, kalıpları göstereyim."

## 1.3 Sekiz sistem, kalıplar (~1.5 dk)

**De ki:**
> "Deterministik kanatta **Hermes** var — 4 geçişli, LLM'siz, çok temiz. **OpenCode** canlı spill +
> prune yapıyor. LLM-özet kanadında **OpenClaw** 12 adımlı bir boru hattı kuruyor, **Codex** ve
> **Claude Code** context dolunca eskiyi özete indiriyor. **Kimi** hibrit — hem kırpıyor hem
> LLM-handoff özeti tutuyor, en olgunlarından. Detaya girmeyeceğim; önemli olan **hepsinin aynı
> soruyu farklı yanıtlaması:** 'neyi at, neyi tut, nasıl özetle?'"

## 1.4 POC — kaçtan kaça (★ ~2.5 dk, DEMO)

**Vurgu:** Bu slayt sunumun kanıtı. Rakamları yavaş söyle.

**De ki:**
> "Beş sistemi çalışır POC olarak yazdım, gerçek Python backend'iyle, LLM-özet adımlarında **gerçek
> LLM API'siyle**. Aynı ağır senaryoyu koşturunca context şöyle düştü:"

| Sistem | Kaçtan kaça | Kazanç |
|---|---|---|
| Hermes | 33.063 → 2.101 | %93.6 |
| OpenClaw | 138.850 → ~100 | %99.9 |
| OpenCode | 111.714 → 76.167 | %31.8 |
| Codex | 123.254 → 136 | %99.9 |
| Claude Code | 126.487 → 13.189 | %89.6 |

> "Dikkat: **OpenCode %31.8**'de kalıyor çünkü canlı çalışıyor, agresif değil; **Codex %99.9** çünkü
> pencereyi tamamen özete indiriyor. Yani 'en yüksek yüzde en iyi' değil — **her sistem farklı bir
> denge** kuruyor: ne kadar kaybı göze alırsın, ne kadar bağlam tutarsın."

**DEMO notu:** İnteraktif POC'u aç, bir sistemi çalıştır, rozetlerdeki kazancı göster. "İsteyen tarayıcıdan kendi de deneyebilir."

**Geçiş:** "İçeriyi hallettik. Peki ya iş worker'ın elinde çökerse? Bölüm iki."

---

# BÖLÜM 2 — TASK MANAGEMENT (işin kendisini yönetmek)

## 2.1 "task" ne demek + tuzak (★ ~2 dk)

**De ki:**
> "Önce bir kelime tuzağını temizleyeyim, yoksa her şey karışır. **'task' üç farklı şeyi anlatabilir:**
> A = bütün İŞ ('sipariş #4711'i işle'), B = tek ADIM (bir tool çağrısı), C = alt-ajan. **Tuzak şu:**
> Airflow ve Celery dokümanında 'task' bir *adımı* (B) anlatır; Temporal ve ajan-motorlarında ise
> bütün *işi* (A) anlatır. Yani 'Airflow task retry yapar' bir adımı; 'Temporal task'ı sürdürür'
> koca işi kurtarır. **Ben hep A seviyesinden — bütün işten — konuşuyorum.**"

## 2.2 İki retry'ı ayır (~1.5 dk)

**De ki:**
> "İki tür retry var, karıştırılmaması lazım. **Birincisi API-retry:** 429/timeout için tekrar dene —
> hafif, hepsinde var, önemsiz. **İkincisi task-retry:** worker çöktü, iş baştan mı yoksa kaldığı
> yerden mi devam etsin — **asıl zor olan bu, koçumun sorduğu da bu.**"

## 2.3 Dayanıklılık merdiveni (★ ~2 dk)

**De ki:**
> "Bütün sistemleri üç basamağa koyabiliriz. **Birinci basamak in-process:** her şey bellekte; süreç
> ölürse **her şey uçar**, sıfırdan başlarsın. **İkinci basamak persisted-session:** diske yazılır,
> veri kaybolmaz — **ama devamını SEN açarsın**, bekleyen-iş kuyruğu yoktur. **Üçüncü basamak durable
> kuyruk:** iş kuyrukta bekler, worker çökerse **sistem otomatik başka worker'a devreder.**"

**Vurgu:** "En çok karışan yer ikinci ve üçüncü basamak: ikisi de diske yazar. Fark **pasif vs aktif** —
persisted'da işi *sen* açarsın, durable kuyrukta *sistem* devralır. **Otonom, hata-toleranslı bir filo
ancak durable kuyrukla olur.**"

## 2.4 ASIL SORU + POC (★ ~2.5 dk, DEMO)

**De ki:**
> "Bütün karar tek soruya iniyor: **worker işin ortasında çökerse, retry'da tamamlanmış adımlar tekrar
> koşar mı?** Bunu ölçtüm. Senaryo: `fetch → process → deliver`, `process` ilk denemede hata veriyor.
> Soru: `fetch` kaç kez koşuyor?"

| Framework (gerçek) | fetch (retry'da) | Ne demek |
|---|:---:|---|
| **Temporal** | **1×** | biten adımı atlar (replay) — kaldığı yerden |
| **Celery** | **2×** | tüm task'ı baştan koşar |
| **Hermes** | başka worker devralır | otomatik reclaim + handoff |

> "Yani **Temporal tamamlanan işi koruyor**, `fetch`'i tekrar etmiyor. **Celery ise baştan koşuyor** —
> `fetch` iki kez. Bu kötü değil, sadece 'kaldığı yerden devam'ı **sana bırakıyor.** **Hermes** kendi
> SQLite çekirdeğiyle çökmeyi otomatik toparlıyor, işi başka worker'a devrediyor."

**DEMO notu:** localhost:8000 → 1. KISIM → üç butonu çalıştır, rozetleri göster (Temporal process ×2, Celery fetch ×2, Hermes worker-B devraldı).

## 2.5 Bizim beyni bağlamak — sar vs kur (★ ~2.5 dk, DEMO)

**De ki:**
> "Peki bu bizim brain_chat_V2'ye nasıl uyar? **Aynı beyni dört altyapıya sardım** ve pahalı bir adımı
> — bağlam toplamayı — çökme sonrası kaç kez koştuğunu ölçtüm:"

| Rota | retrieve | Yorum |
|---|:---:|---|
| brain → Temporal (buy) | **1×** | replay korudu |
| brain → Hermes (build) | **1×** | handoff taşıdı |
| brain → **kendi çekirdeğimiz** (build) | **1×** | checkpoint + otomatik recovery |
| brain → Celery (buy) | **2×** | baştan koştu |

> "Ve en önemlisi: **kendi durable çekirdeğimizi ~200 satırda kurdum** — sadece SQLite. Hermes'in
> otomatik crash-recovery'siyle Temporal'ın 'kaldığı yerden devam'ını **tek dosyada** birleştiriyor.
> Worker çöktü, sistem otomatik topladı, pahalı adım **bir daha koşmadı.** Dış framework yok."

**DEMO notu:** localhost:8000 → 2. KISIM → brain build + brain celery'yi çalıştır, `retrieve ×1` vs `retrieve ×2 (BAŞTAN)` rozetlerini yan yana göster.

## 2.6 Karar / öneri (★ ~1.5 dk)

**De ki:**
> "Önerim net. brain zaten oturum/state tuttuğu için: **hızlı istiyorsak Celery'ye devrederiz**,
> resume'u checkpoint'le kendimiz ekleriz. **Tam kontrol istiyorsak — ki önerim bu — Hermes-tarzı
> hafif durable çekirdeği kendimiz kurarız**, ~1-2K satır, otomatik recovery + kaldığı yerden devam
> birlikte gelir. **Temporal'a ancak** çok-makineli, uzun-bekleyen, exactly-once şart olunca geçeriz —
> operasyon ve determinizm bedeli, henüz ihtiyacımız olmayan garantiler için yüksek."

---

## KAPANIŞ — iki katman tek resim (★ ~1.5 dk)

**De ki:**
> "Toparlayayım. **Tool-trace compaction** işin içini yönetir — context'i %90+ küçültüp bilgiyi
> korur. **Task management** işin kendisini yönetir — worker çökse de iş kaybolmasın. İkisi birlikte
> bir ajanı hem *uzun* hem *dayanıklı* yapar. Ve hepsini iddia olarak değil, **çalışan, ölçülebilir
> kodla** gösterdim — tarayıcıdan herkes deneyebilir. Sorularınızı alayım."

---

## Muhtemel sorular — hazır cevaplar

**"Neden Temporal'ı seçmedik, en dayanıklı o değil mi?"**
> "En dayanıklı o, evet — ama bedeli cluster işletmek + determinizm disiplini (LLM'i activity'e sarmak
> zorundasın). Bizim ölçekte durable kuyruk seviyesi yetiyor; onu ~200 satırda kendimiz veriyoruz.
> İhtiyaç büyürse Temporal'a geçiş yolu açık — Shannon projesi bunun canlı örneği."

**"POC'lar gerçek mi, simülasyon mu?"**
> "Task tarafında gerçek: Hermes'in gerçek `kanban_db`'si, gerçek `temporalio` + dev server, gerçek
> Celery worker + broker. Tool-trace tarafında beş sistemin adımları gerçek Python'da, LLM-özet
> adımlarında gerçek LLM API'siyle. Kapalı-kaynak olanlar (Claude Code) gözleme dayalı, onu belirttim."

**"En yüksek sıkıştırma yüzdesi en iyisi mi?"**
> "Hayır. Yüksek yüzde çok bağlam attığın anlamına da gelebilir. Doğru metrik: *kaybı göze alabileceğin
> kadar sıkıştır, işe yarayan bağlamı tut.* OpenCode bilinçli olarak %31'de kalıyor çünkü canlı ve muhafazakâr."

**"Compaction bilgi kaybettirmez mi?"**
> "Kaybettirebilir — o yüzden iki ekol var. Deterministik olan büyük çıktıyı diske döküp **referans**
> bırakır (geri çağrılabilir); LLM-özet olan **damıtır** (kaybı özetle telafi eder). Kritik veriyi
> 'protect window' ile hiç dokundurmadan koruyabilirsin."

**"brain'e hangisini koyalım, bugün?"**
> "Bugün başlıyor olsak: Hermes-tarzı SQLite çekirdek. Kanıtı elimde — `brain_build_own_poc.py` çöküp
> kendini toparlıyor ve pahalı adımı tekrarlamıyor. Riski düşük, tam bizim kontrolümüzde."

---

## Sunum öncesi kontrol listesi
- [ ] `web_server.py` açık (localhost:8000) — 1. ve 2. KISIM görünüyor.
- [ ] Tool-trace interaktif POC hazır (kaçtan-kaça rozetleri).
- [ ] Temporal dev server bir kez ısıtıldı (ilk çalıştırma ~20s sürmesin).
- [ ] Elinde tek tablo: tool-trace kaçtan-kaça + task-mgmt fetch/retrieve sayıları.
- [ ] Belgeler açık: `USTA-REHBER…`, `brain-chat-v2-task-management-entegrasyon…`, `brain-poc-fonksiyon-fonksiyon…`.

**İlgili belgeler:** `report/USTA-REHBER-tool-trace-ve-task-management.md` (iki konu tek rehber),
`report/task-management-sunum-ve-flowchart.md` (flowchart'lar), `report/brain-chat-v2-task-management-entegrasyon.md`,
`report/brain-poc-fonksiyon-fonksiyon-anlatim.md` (kod detayı).
