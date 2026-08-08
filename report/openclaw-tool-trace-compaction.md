# OpenClaw Tool-Trace Compaction — Dadının Anlattığı Gibi 🎒

> **Hayal et:** Bir çocuğun **sırt çantası** var. Okuldan her gelişte içine kâğıtlar, oyuncaklar, kocaman kitaplar tıkıştırıyor. Çanta dolunca artık yeni bir şey girmiyor, sırtı da ağrıyor. İşte bir **dadı** geliyor ve çantayı **akıllıca** düzenliyor: gereksizleri özetliyor, devasa şeyleri fotoğraflayıp bırakıyor, ama çocuğun en sevdiği ve en yeni şeylerine **dokunmuyor**.
>
> Bu belgede o **çanta = context penceresi**, o **eşyalar = tool çağrıları/sonuçları (tool-trace)**, o **dadı = compaction**. Her şeyi bu benzetmeyle anlatıyoruz. Kod gerçek: [../harnesses/openclaw/src/agents/](../harnesses/openclaw/src/agents/).

---

## 🎒 En büyük resim (tek cümle)

> **Dadı, çanta taşamadan önce içini düzenler: soruyla cevabı asla ayırmaz, sırları saklar, devasa şeyleri "fotoğraf notuna" indirir, gerisini kısa bir masala özetler — hepsini arka odada yapar ki çocuk oyununa devam etsin.**

OpenClaw'ın özelliği: dadı eşyayı **kendi başına küçültmez** (Hermes öyle yapar). Onun yerine eşyaları **kutulara paketleyip bir "masalcıya" (LLM) verir**, masalcı da kısa özet yazar.

---

## 📖 Gerçek adı ne? (benzetme ↔ teknik)

| Dadı dili | Gerçek adı |
|---|---|
| Sırt çantası | context window (token sınırı) |
| Çantadaki eşyalar | tool-trace (tool çağrıları + sonuçları) |
| Dadı | compaction |
| Sır saklama | sanitize (`stripToolResultDetails`) |
| Eşyayı tartmak | token estimation |
| Taşımak yerine etiket | projection (planlama projeksiyonu) |
| Soru + cevabı ayırmama | tool-pair bütünlüğü (`pendingToolCallIds`) |
| Kutulara koyma | chunking |
| Devasa şeyin fotoğrafı | oversized note |
| Masalcı | özet modeli (LLM) |
| Arka oda | worker-thread |
| Yarım kalanı tamamlama | transcript repair |

---

## 🧸 Hikâyedeki çanta (örneğimiz)

Çanta en fazla **200.000 birim** ağırlık taşıyor. İçinde şunlar var:

```
#0 🪪  "Ben OpenClaw'ım, auth işine bak"  (kimlik kartı)        ~2.000
#1 🗣️  "auth modülünü düzelt"  (çocuğun isteği)                    ~30
#2 ❓  "şu iki şeye bak: dosyayı oku + login ara"  (iki soru)
#3 📄  auth.py'nin TAMAMI + gizli notlar(details)               ~40.000
#4 📄  "47 sonuç buldum" + gizli notlar                          ~8.000
#5 ❓  "şu dev dosyayı oku"  (bir soru daha)
#6 📚  200KB'lık KOCAMAN dosya + gizli notlar                  ~110.000
#7 💬  "Planım: login'i böl, token'ı güvenli yap"  (en yeni)        ~40
#8 👻  (görünmez iç not — çocuk bunu görmüyor bile)                 0
```

Toplam ≈ **160.070**. Çanta 200.000'lik. Neredeyse doldu. Dadı iş başına.

---

## Adım adım — dadı çantayı düzenliyor

### Adım 0 · 🎒 "Çanta yarıdan fazla doldu mu?"
**Dadı dili:** Dadı sürekli bakar: *"Yeni oyuncaklara yer kalıyor mu?"* Çantanın en az **yarısı** boş kalmalı. Kalmıyorsa temizlik başlar.
**Gerçekte:** `MIN_PROMPT_BUDGET_RATIO = 0.5` (yarısı boş kalmalı), taban `MIN_PROMPT_BUDGET_TOKENS = 8.000`.
**Örnekte:** Boş yer = 200.000 − 160.070 = **39.930**. Gereken = 100.000. `39.930 < 100.000` → **temizlik zamanı!** ✅

### Adım 1 · 🤫 "Önce sırları sakla"
**Dadı dili:** Dadı çantayı **başkasına (masalcıya)** göstermeden önce, içindeki **kişisel/gizli notları** çıkarır — şifreler, ev adresi gibi. Kimse görmesin.
**Gerçekte:** `stripToolResultDetails` her tool sonucundan `details` alanını **siler**; runtime iç notları da çıkar.
**Örnekte:** `#3`'ün içindeki `details:{env:{API_KEY:...}, cwd:"/gizli"}` **silinir**; `#8` (görünmez iç not) tamamen çıkar.
**Neden akılda kalsın:** *Masalcıya çantayı vermeden önce, cüzdanı çıkar.* 🤫

### Adım 2 · ⚖️ "Her şeyi tek tek tart"
**Dadı dili:** Dadı her eşyayı teraziye koyar: *"Bu kaç kilo?"* Görünmez notlar **sıfır kilo** (zaten kimse taşımıyor).
**Gerçekte:** `estimatePerMessageTokens` → her mesaja bir ağırlık; model-görünmez = 0.
**Örnekte:** `[2000, 30, 15, 40000, 8000, 8, 110000, 40, 0]`.
**Neden akılda kalsın:** *Düzenlemeden önce tart, neyin ağır olduğunu bil.* ⚖️

### Adım 3 · 🏷️ "Devasa şeyi taşıma — üstüne kilosunu yaz"
**Dadı dili:** 200KB'lık kocaman kitabı arka odaya **taşımak** çok zahmetli. Dadı akıllıdır: kitabın **ilk sayfalarından küçük bir örnek** alır, üstüne bir **etiket** yapıştırır: *"bu kitap ~110 kilo, gerisi rafta."* Böylece planı yaparken kitabın **ne kadar ağır olduğunu bilir** ama sırtında taşımaz.
**Gerçekte:** `projection` — büyük gövde 8KB örneğe iner + `omittedChars` damgası; ağırlık = `örnek + omittedChars/4`. Toplam 256KB'la sınırlı.
**Örnekte:** `#6` (110.000) → 8KB örnek + "atılan: 101.808" damgası → ağırlık yine ~27.500 doğru hesaplanır.
**Neden akılda kalsın:** *Fili taşıma, üstüne "fil, 3 ton" yaz.* 🏷️🐘

### Adım 4 · 📦 "Eşyalar büyükse KÜÇÜK kutu al"
**Dadı dili:** Dadı kutu seçer. Eşyalar minicikse **kocaman kutu** alır (hepsi bir kutuya sığar). Ama eşyalar iri iri ise **küçük kutu** alır — yoksa kutu masalcının kucağına sığmaz!
**Gerçekte:** `computeAdaptiveChunkRatio` — chunk boyu 0.40'tan 0.15'e kadar iner.
**Örnekte:** Ortalama eşya = 160.070/8 ≈ 20.000. `avgRatio = 20.000×1.2/200.000 = 0.12 > 0.10` → kutu küçülür → **32.000'lik kutu** (80.000'den indi).
**Neden akılda kalsın:** *Büyük eşya → küçük kutu (yoksa kimse taşıyamaz).* 📦

### Adım 5 · 🔗 "Soruyla cevabı ASLA ayırma"
**Dadı dili:** En kutsal kural. Bir **soru kâğıdı** ile onun **cevap kâğıdı** birbirine aittir. Dadı bunları **asla farklı kutulara koymaz** — yoksa okulda "cevabı olmayan soru" ya da "sorusu olmayan cevap" çıkar, öğretmen kızar (sağlayıcı 400 hatası verir).
**Gerçekte:** `pendingToolCallIds` — bir grup, **tüm cevaplar gelene kadar** kapanmaz. (`aborted/error` ise cevap hiç gelmeyecek, o çağrı yok sayılır.)
**Örnekte:**
```
#2 iki soru sordu (c1, c2)  → bekleyen = {c1, c2}
#3 c1'in cevabı geldi        → bekleyen = {c2}   (grup HÂLÂ kapanmaz!)
#4 c2'nin cevabı geldi        → bekleyen = {}    (şimdi kapanır)
→ [#2, #3, #4] hep BİRLİKTE kalır
```
**Neden akılda kalsın:** *Soru ve cevap el ele. Asla ayırma.* 🔗

### Adım 6 · 📦 "Eşyaları kutulara paketle"
**Dadı dili:** Dadı, birbirine ait grupları alır, **küçük kutuya sığdığı kadar** koyar; kutu dolunca yenisini açar. Ama bir grubu (soru+cevap) **asla ikiye bölmez**.
**Gerçekte:** `chunkCompactionMessageGroups` — grupları `maxChunk`'a (32K) kadar paketler, grup bölünmez.
**Örnekte:** Kutular → `[kimlik+istek]`, `[soru+2 cevap]`, `[dev dosya sorusu+cevabı]`, `[plan]`.
**Neden akılda kalsın:** *Kutuya koy ama çifti ayırma.* 📦

### Adım 7 · 📸 "Kocaman şey kutuya sığmıyorsa — fotoğrafını çek, kendisini bırak"
**Dadı dili:** 200KB'lık dev dosya hiçbir kutuya sığmaz. Dadı onu masalcıya **veremez**. O yüzden dosyanın kendisini bırakır ve yerine küçük bir **not** koyar: *"Burada kocaman bir dosya vardı (~110 kilo), özete koyamadım."* Çocuğun **kendi sözü (istek)** varsa o **asla atılmaz**.
**Gerçekte:** `oversizedThreshold = pencere × 0.5`. `#6`: `110.000×1.2 = 132.000 > 100.000` → içerik yerine `[Large toolResult (~110K tokens) omitted from summary]`; onun batch'i (#5) da düşer, ama gerçek user mesajı hayatta kalır.
**Neden akılda kalsın:** *Fili özete koyamazsın; "burada fil vardı" notu bırak.* 📸🐘

### Adım 8 · 🎞️ "Tek masal mı, birkaç masal mı?"
**Dadı dili:** Dadı karar verir: az eşya varsa **tek bir masal** anlatır (bir özet). Çok ve iri eşya varsa **birkaç parçaya bölüp** ayrı ayrı özetler.
**Gerçekte:** `buildStageSplitPlan` — `single` vs `split`; en az 4 mesaj ve bütçeyi aşıyorsa böl.
**Örnekte:** Özetlenecek 3 kâğıt kaldı (`< 4`) → **tek masal (single)**.
**Neden akılda kalsın:** *Az şey = tek masal; çok şey = bölerek anlat.* 🎞️

### Adım 9 · 🚪 "Temizliği ARKA ODADA yap"
**Dadı dili:** Dadı bütün bu tartma/kutulama/planlama işini **çocuğun yanında** yapmaz (çocuk sıkılır, oyun durur). **Arka odaya** geçer, orada planlar; çocuk bu sırada oynamaya devam eder.
**Gerçekte:** Tüm planlama `compaction-planning.worker.ts` içinde **ayrı iş-parçacığında** (worker-thread) koşar → ana döngü bloklanmaz.
**Neden akılda kalsın:** *Temizliği arka odada yap, çocuk oyununa devam etsin.* 🚪

### Adım 10 · 📖 "Uzun hikâyeyi kısa masala çevir"
**Dadı dili:** Masalcı (LLM) kutuyu alır, içindeki uzun uzun kâğıtları okur ve **kısa bir masala** çevirir: *"auth.py okundu, 47 login yeri bulundu, token güvenli yapıldı."*
**Gerçekte:** Hazır chunk özet modeline verilir; `SUMMARIZATION_OVERHEAD_TOKENS = 4096` pay ayrılır.
**Örnekte:** `[soru+2 cevap]` kutusu → *"auth.py okundu; 'login' için 47 eşleşme; login() 45. satırda."*
**Neden akılda kalsın:** *Kalın kitap → tek paragraf masal.* 📖

### Adım 11 · 🩹 "Yarım kalanı tamamla, boşluk bırakma"
**Dadı dili:** Bazen bir **cevap kâğıdı** düşmüş ama **soru** çantada kalmıştır (yetim soru). Dadı boşluk bırakmaz: yerine küçük bir kâğıt koyar: *"bu sorunun cevabı kayboldu."* Böylece öğretmen (sağlayıcı) kızmaz.
**Gerçekte:** `repairToolUseResultPairing` — eksik tool sonucuna **sentetik hata sonucu** ekler: *"missing tool result... inserted synthetic error result."*
**Neden akılda kalsın:** *Sorusu var cevabı yok mu? Boş kâğıt koy, boşluk bırakma.* 🩹

### Adım 12 · 🎒 "Yeni düzeni çantaya koy, eski tartı fişlerini at"
**Dadı dili:** Dadı yeni, düzenli çantayı çocuğa verir: kimlik + istek + kısa masal + "fil vardı" notu + en yeni plan. Eski, artık yanlış olan **tartı fişlerini** de çöpe atar.
**Gerçekte:** Yeni transcript yazılır; `stripStaleAssistantUsageBeforeLatestCompaction` eski usage snapshot'larını sıfırlar.
**Örnekte SONRA:**
```
🪪 kimlik                                                   ~2.000
🗣️ "auth modülünü düzelt"                                     ~30
📖 masal: "auth.py okundu, 47 eşleşme, login() 45. satır"    ~250
📸 not: "burada ~110K'lık dosya vardı, özete koyamadım"       ~15
💬 "Planım: login'i böl, token güvenli"  (en yeni, korundu)   ~40
```
**160.070 → ~2.335** (%98 hafifledi!) · sırlar sızmadı · soru-cevaplar tam · çocuğun sözü korundu.

---

## 🧠 Aklında kalsın (dadının 12 altın kuralı)

1. 🎒 **Çanta yarıdan fazla dolduysa** → temizle.
2. 🤫 **Masalcıya vermeden sırları çıkar.**
3. ⚖️ **Önce tart**, neyin ağır olduğunu bil.
4. 🏷️ **Fili taşıma, üstüne "fil" yaz** (etiket).
5. 📦 **Büyük eşya → küçük kutu.**
6. 🔗 **Soru ile cevabı ASLA ayırma.**
7. 📦 **Kutula ama çifti bölme.**
8. 📸 **Kutuya sığmayan devi → "burada dev vardı" notu.**
9. 🎞️ **Az şey tek masal, çok şey bölerek.**
10. 🚪 **Temizliği arka odada yap.**
11. 📖 **Uzun hikâye → kısa masal.**
12. 🩹 **Yarım kalanı tamamla, çöpü at.**

---

## 🍼 Peki Hermes dadısı ne farklı yapardı?

- **OpenClaw dadısı:** eşyaları **kutulayıp masalcıya verir** (LLM özeti). "Ben özetleyemem, masalcı özetlesin."
- **Hermes dadısı:** eşyayı **kendi eliyle** tek satıra indirir (masalcı çağırmadan): *"[kitap] 300 sayfa okundu."* Aynıysa **tekini tutar** (dedup).

İkisi de aynı kutsal kurala uyar: **soru-cevabı ayırma, çocuğun sözünü koru.** Fark: biri masalcı çağırır (OpenClaw), biri kendi halleder (Hermes).

---

## 🔧 Gerçek dosyalar (dadı masalı değil, kod)
- `compaction-planning.ts` — `groupCompactionMessages` (Adım 5) · `chunkCompactionMessageGroups` (6) · `computeAdaptiveChunkRatio` (4) · `buildOversizedFallbackPlan` (7) · `buildStageSplitPlan` (8) · `sanitizeCompactionMessages` (1)
- `compaction-planning-projection.ts` — `projection` (Adım 3): 8KB örnek + `omittedChars`, 256KB bütçe
- `compaction-planning.worker.ts` — arka oda (Adım 9)
- `session-transcript-repair.ts` — `repairToolUseResultPairing` (Adım 11) · `stripToolResultDetails` (1)
- `agent-compaction-constants.ts` — `MIN_PROMPT_BUDGET_TOKENS/RATIO` (Adım 0)
- Kardeş masallar: [hermes-tool-trace-compaction.md](hermes-tool-trace-compaction.md) · [opencode-tool-trace-compaction.md](opencode-tool-trace-compaction.md) · [codex-tool-trace-compaction.md](codex-tool-trace-compaction.md)
