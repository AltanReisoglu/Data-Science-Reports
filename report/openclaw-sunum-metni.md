# OpenClaw Tool-Trace Compaction — Sunum Metni (Teknik)

> **Nasıl kullan:** Her bölümde 🖼️ **Ekranda ne olsun** (slayt içeriği) + 🗣️ **Ne diyeceksin** (konuşmacı notu) + 💡 **Vurgu** var. Hedef süre ~15 dk. Kaynak: `openclaw/src/agents/compaction-planning.ts` (+ projection/worker/repair). Tüm sayılar çalışan POC'tan (`openclaw_tool_trace_poc.py`).

---

## 0 · AÇILIŞ — problem (45 sn)

🖼️ *Başlık: "Tool-trace neden şişer?" · Bir grafik: turn arttıkça token doğrusal büyüyor, sınıra dayanıyor.*

🗣️ "Bir tool-use ajanı çalışırken her tool çağrısı ve sonucu context'e eklenir ve model her turn'de tüm geçmişi yeniden görür. Sorun: tool **sonuçları** devasa olabilir — bir dosya okuması 100 bin token. Bunlar birikince context penceresi dolar, maliyet ve gecikme patlar. Bugün OpenClaw'ın bu tool izini **taşmadan** nasıl küçülttüğünü, 12 adımlık planlama hattı üzerinden anlatacağım. Açık kaynak, ve sonunda çalışan bir simülasyon var."

💡 **Context penceresi sonlu; tool sonuçları en büyük şişme kaynağı.**

---

## 1 · MİMARİ TEZ (1 dk)

🖼️ *İki kutu: "Deterministik budama (Hermes/OpenCode)" vs "LLM-chunk özetleme (OpenClaw)". Altına: "OpenClaw = güvenli chunk hazırlama + LLM özeti"*

🗣️ "Compaction'ın iki yaklaşımı var. Birincisi deterministik: tool sonucunu LLM'siz, kural tabanlı küçültmek — dedup, tek-satır özet. OpenClaw ikinci yolu seçer: tool-çiftlerini bozmadan **chunk'lar ve chunk'ları bir özet modeline gönderir.** OpenClaw'ın mühendislik değeri, o chunk'ları **doğru hazırlamakta**: güvenlik şeritlemesi, tool-pair bütünlüğü, oversized fallback, transcript onarımı. 12 adım bu hazırlık hattı."

💡 **OpenClaw tool sonucunu yeniden yazmaz; güvenli chunk'lar üretip LLM'e özetletir.**

---

## 2 · ÖRNEK TRACE (45 sn)

🖼️ *Örnek trace (POC girdisi):*
```
#2 assistant → read_file(auth.py)=c1, search_files(login)=c2   (paralel 2 çağrı)
#3 toolResult c1  31.222 tok  + details:{API_KEY}
#4 toolResult c2   3.097 tok  + details
#5 assistant → read_file(big.py)=c3
#6 toolResult c3 104.472 tok   ← window×0.5'i aşıyor
#7 assistant "Refactor planı..."   (tail)
#8 runtime (model-görünmez)
```

🗣️ "Örneğimiz: ajan auth modülünü refactor ediyor. Dikkat edeceğimiz iki nokta var: #2'de **iki tool aynı anda** çağrılmış (c1, c2) — bu bütünlük testi olacak. Ve #6, tek başına 104 bin token'lık bir sonuç — pencerenin yarısından büyük, oversized testi olacak. Context window = 200 bin."

💡 İki kritik vaka: **paralel çift (c1,c2)** ve **oversized (#6)**.

---

## 3 · 12 ADIM (her biri ~40 sn — ritim tut)

> Slayt formatı: adım no + fonksiyon adı + tek satır kural + örnek sonuç.

### Adım 0 — Tetik · `MIN_PROMPT_BUDGET_RATIO`
🖼️ `free < window×0.5 ?  →  61.150 < 100.000 → TETİK`
🗣️ "Tetik bütçe-tabanlı: pencerenin en az yarısı prompt'a kalmalı, taban 8 bin token. Örnekte boş yer 61 bin, gereken 100 bin → compaction tetiklenir."
💡 `free < window × 0.5` (veya < 8K).

### Adım 1 — Sanitize · `stripToolResultDetails`
🖼️ `sanitize = stripToolResultDetails ∘ stripRuntimeContextCustomMessages` · `3 details silindi, 1 runtime çıktı`
🗣️ "Compaction metni bir özet modeline gidecek. O yüzden önce her toolResult'tan `details` alanı silinir — ham stdout, env, API_KEY. Ve model-görünmez runtime mesajları çıkarılır. Örnekte 3 details silindi, API_KEY sızmadı."
💡 Hassas `details` özet modeline asla girmez.

### Adım 2 — Estimate · `estimatePerMessageTokens`
🖼️ `[11, 12, 14, 31222, 3097, 9, 104472, 13]`
🗣️ "Her mesaj için sanitized token tahmini, diziyle 1:1 hizalı; model-görünmez mesajlar sıfır. Bu dizi, sonraki tüm adımların girdisi."
💡 Per-mesaj ağırlık = tüm kararların temeli.

### Adım 3 — Projection · `omittedChars`
🖼️ `content > 32.768 kar → 8.192 örnek + omittedChars` · `#6: örnek + atılan 409.697; ağırlık 104.473`
🗣️ "Kilit adım. 400 kilobaytlık gövdeyi worker'a taşımak verimsiz. Bunun yerine ilk 8 kilobaytlık örnek alınır, atılan karakter sayısı `omittedChars` olarak damgalanır. Ağırlık = `est(örnek) + omittedChars/4` → 8KB gövdeyle bile mesajın gerçek boyutu (104 bin) korunur. Toplam projeksiyon 256KB ile sınırlı."
💡 İçerik hafif (8KB), boyut doğru (`+omittedChars/4`).

### Adım 4 — Adaptif oran · `computeAdaptiveChunkRatio`
🖼️ `avgRatio = ort×1.2/window` · `0.104 → ratio 0.19 → maxChunk 38.344` · `BASE 0.4, MIN 0.15`
🗣️ "Chunk hedefi, ortalama mesaj boyutunun pencereye oranına göre 0.4'ten 0.15'e iner. Mantık: mesajlar büyükse chunk küçük olmalı, yoksa chunk özet-modelinin limitini aşar. Örnekte ratio 0.19, maxChunk 38 bin."
💡 Büyük mesaj → küçük chunk (özet-modeli boğulmasın).

### Adım 5 — Gruplama · `pendingToolCallIds`
🖼️ `grup pending boşalınca kapanır` · `G2 = [assistant(c1,c2) + result c1 + result c2]`
🗣️ "En kritik bütünlük kuralı. Bir assistant N tool çağırırsa `pendingToolCallIds` N id ile dolar; her sonuç geldikçe silinir; grup **ancak küme boşalınca** kapanır. Örnekte c1,c2 çağrısı ve iki sonucu tek grupta kaldı. Bir çağrıyla sonucu farklı chunk'lara düşerse sağlayıcı 400 döner. `aborted/error` çağrıları yok sayılır — sonuç hiç gelmeyecek."
💡 call/result çifti asla bölünmez → 400 önlenir.

### Adım 6 — Chunk · `chunkCompactionMessageGroups`
🖼️ `grupları effMax'a paketle, grup bölme` · `effMax = 38.344/1.2 ≈ 31.953 → 4 chunk`
🗣️ "Atomik gruplar maxChunk'a kadar paketlenir; `effectiveMax = maxChunk/SAFETY_MARGIN` (tahmin şişmesine karşı). Grup asla bölünmez — dev grup tek başına eşiği aşsa bile bütün kalır."
💡 Kutula, ama grubu bölme.

### Adım 7 — Oversized · `buildOversizedFallbackPlan`
🖼️ `tokens×1.2 > window×0.5 → not` · `#6: 104K×1.2=125K > 100K → "[Large toolResult (~104K) omitted]"`
🗣️ "Tek bir mesaj özet isteğine sığmayacak kadar büyükse — pencerenin yarısını aşarsa — içeriği yerine bir not konur ve batch'i düşer. Örnekte #6 oversized, c3 batch'i düştü. Ama araya sıkışmış gerçek user mesajı olsaydı korunurdu."
💡 Sığmayan mesaj → içerik yerine "omitted" notu.

### Adım 8 — Stage-split · `buildStageSplitPlan`
🖼️ `count<4 veya total≤maxChunk → single, değilse split` · `3 mesaj < 4 → single`
🗣️ "Özetlenecek içerik tek istekte mi, birkaç parçada mı? En az 4 mesaj ve bütçeyi aşıyorsa `DEFAULT_PARTS=2` ile bölünür, yine çift bozmadan. Örnekte geriye 3 mesaj kaldı → single."
💡 Az/küçük → tek özet; çok/büyük → böl.

### Adım 9 — Worker · `compaction-planning.worker.ts`
🖼️ `planlama worker-thread'de (node:worker_threads)`
🗣️ "Adım 4-8'in hesabı CPU-yoğun. Ana döngüde yapılsa ajan donar. Bu yüzden tüm planlama ayrı bir worker-thread'de koşar; ana döngü bloklanmaz."
💡 Planlama ayrı thread → ajan donmaz.

### Adım 10 — LLM özet · `buildSummaryChunks`
🖼️ `chunk → özet modeli` · `SUMMARIZATION_OVERHEAD_TOKENS=4096` · `[ÖZET: 3 mesaj · read_file, search_files]`
🗣️ "Hazır chunk özet modeline verilir; 4096 token prompt/system/önceki-özet payı ayrılır. OpenClaw'ın deterministik sistemlerden farkı tam burada: tool sonucunu kendi küçültmez, özeti modele yazdırır."
💡 Chunk → LLM → kısa özet.

### Adım 11 — Onarım · `repairToolUseResultPairing`
🖼️ `yetim call → sentetik error result` · `0 yetim (çift birlikte düştü)`
🗣️ "Compaction sonrası eşi kayıp bir call/result kaldıysa, eksik sonuç için sentetik bir hata sonucu eklenir — transcript geçerli kalsın, replay 400 vermesin. Örnekte batch bütün düştüğü için yetim kalmadı."
💡 Yetim çift → sentetik sonuçla onar.

### Adım 12 — Uygula · `stripStaleAssistantUsageBeforeLatestCompaction`
🖼️ `yeni transcript yaz + usage snapshot sıfırla` · `5 mesaj`
🗣️ "Son adım: yeni transcript oturuma yazılır; compaction öncesi assistant usage snapshot'ları artık geçersiz olduğu için sıfırlanır."
💡 Yeni transcript + eski usage sayaçları sıfır.

---

## 4 · CANLI DEMO (2 dk)

🖼️ *Terminal: `python3 openclaw_tool_trace_poc.py` çıktısı*

🗣️ "12 adımı gerçek mantıkla uygulayan simülasyonu çalıştıralım." *(çıktıyı göster)*
- "Girdi: **138.850 token, 9 mesaj.**"
- "Adım 1: `details` silindi (API_KEY). Adım 3: #3 ve #6 → 8KB örnek + omittedChars, ağırlık 31.223 / 104.473 korundu. Adım 5: c1-c2 çifti G2'de birlikte. Adım 7: #6 → oversized not."
- "Sonuç:" *(en alt)* "**138.850 → 66 token. Çift bütünlüğü ✓. details sızmadı ✓.**"

💡 "Sayı düşük çünkü demo özeti mock (gerçek LLM özeti ~250 token). Kritik olan: her karar **doğru ağırlıkla, doğru sırada** verildi — özellikle #6 oversized kararı 8KB örneği değil, `omittedChars`'la kurulmuş **104K ağırlığı** kullandı."

---

## 5 · KAPANIŞ — 3 tasarım ilkesi (1 dk)

🖼️ *"OpenClaw'ın üç mühendislik kararı"*

🗣️ "Üç ilke aklınızda kalsın:"
1. **Güvenlik önce** — `stripToolResultDetails`: hassas veri özet modeline asla gitmez (Adım 1).
2. **Bütünlük kutsal** — `pendingToolCallIds`: call/result çifti asla bölünmez (Adım 5 & 11). 400 hatalarının kökü.
3. **Boyut ≠ içerik** — projection + oversized: gövdeyi taşımadan doğru boyut kararı; sığmayanı nota indir (Adım 3 & 7).

🗣️ "Tek cümle: **OpenClaw tool sonucunu yeniden yazmaz; güvenli, çift-bütünlüklü chunk'lar hazırlayıp bir LLM'e özetletir — hepsini ayrı bir worker-thread'de planlayarak.** Sorular?"

---

## 6 · OLASI SORULAR (hazırlık)

**S: Neden deterministik budama yerine LLM özeti?**
C: Esneklik — LLM bağlamı yakalar ("47 eşleşmenin önemlisi 45. satır"). Bedeli bir model çağrısı. Hermes/OpenCode deterministik: ucuz ama daha mekanik.

**S: 8KB örnek boyutu yanlış tahmin etmez mi?**
C: Örnek yalnız planlama için; boyut `omittedChars/4` ile tam korunur. Demo bunu gösteriyor: #6'nın örneği ~2K token ama ağırlığı 104K — oversized kararı ağırlığı kullanır, örneği değil.

**S: Oversized mesaj tümden kayboluyor mu?**
C: Özetten çıkar ama bir not kalır ("~104K omitted"). Ham gövde geçmişte/diskte durabilir; ajan gerekirse yeniden okur.

**S: Neden worker-thread?**
C: Büyük transcript'te token tahmini + gruplama + chunk'lama CPU-yoğun. Ana döngüde yapılsa ajan yanıt veremez; worker ana döngüyü bloklamaz.

**S: Diğer ajanlarla fark?**
C: Hermes deterministik 3-pass; OpenCode prune + spill-to-disk; Codex ortadan-kes + pencereli/resume; Claude Code microcompaction + subagent. OpenClaw = güvenli-chunk + LLM özeti + worker planlama.

---

## 7 · KÜRSÜ AKIŞ KARTI (tek bakış)

```
0 Problem: tool sonucu şişmesi              (45s)
1 Tez: budama vs LLM-chunk; OpenClaw=chunk  (1dk)
2 Örnek: paralel çift (c1,c2) · oversized #6 (45s)
3 12 ADIM (~40s):
   0 Tetik(free<½)  1 stripDetails  2 estimate  3 projection(omittedChars)
   4 adaptif ratio  5 pendingToolCallIds  6 chunk(effMax)  7 oversized(½)
   8 stage-split  9 worker-thread  10 LLM özet  11 repair  12 apply+usage
4 DEMO: 138.850→66, ✓ bütünlük ✓ sır       (2dk)
5 Kapanış: güvenlik / bütünlük / boyut≠içerik (1dk)
6 Q&A
```

---

### Kaynaklar (sende dursun)
- Teknik belge: [openclaw-tool-trace-compaction-teknik.md](openclaw-tool-trace-compaction-teknik.md)
- Çalışan POC: [../poc/openclaw_tool_trace_poc.py](../poc/openclaw_tool_trace_poc.py)
- Kod: `openclaw/src/agents/compaction-planning.ts` · `compaction-planning-projection.ts` · `session-transcript-repair.ts`
