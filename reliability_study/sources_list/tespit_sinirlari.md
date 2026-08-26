# Tespit Sınırları — Loop Detection ve Bütçe Zorlamasının Nerede Yanıldığı

Tarama tarihi: 2026-08-21 · Yöntem: **birincil kaynak** — GitHub'daki gerçek kaynak kodu
(`raw.githubusercontent.com`), makalelerin tam metni (alphaXiv PDF sorgulama), klasik
algoritma referansları.

**Kardeş dosyalar (önce okundu, tekrar edilmedi):**
`loop_budget.md` (akademik kaynaklar + olay katalogları) · `harness_kontrolleri.md`
(22 harness + 5 gateway kod envanteri). Bu dosya onların üstüne **sınırlar katmanını**
koyuyor: aynı mekanizmalar burada "nerede yanılıyor" sorusuyla ele alınıyor.

**Doğrulama etiketleri:**
`[K]` kaynak kodu bizzat okundu · `[T]` makale tam metni okundu ·
`[D]` resmî doküman · `[Ö]` sadece abstract · `[?]` doğrulanamadı.

**Her başlığın şablonu:** *yöntem nasıl çalışır → nerede yanılır → ölçülmüş veri →
PoC'de nasıl gösterilir.* Son madde brief'in "uygulanabilir olsun" şartını karşılıyor.

---

## 0. Tek cümlelik özet

Bu taramanın ana bulgusu üç kalemde toplanıyor:

1. **İmza tabanlı tekrar tespiti, tespit etmesi en kolay döngüyü tespit eder.**
   Gerçek üretim döngülerinin büyük kısmı ya sözdizimsel olarak farklı görünüyor
   (değişken alanlar, yeniden ifade) ya da hiç tekrar etmiyor (agent sürekli *yeni*
   iş yapıyor ama hedefe yaklaşmıyor). İkinci sınıf, imza karşılaştırmasına yapısal
   olarak görünmez.
2. **Adım/tur limiti ile dolar limiti aynı şey değil ve ölçülmüş fark büyük.**
   Token Budgets makalesinin baş-başa deneyinde LangGraph `recursion_limit=20`,
   CrewAI `max_iter=5`, AutoGen `max_turns=4` ve LiteLLM post-call proxy'sinin
   **dördü de eşleştirilmiş dolar tavanını 30/30 koşumda aştı** `[T]`.
3. **Kontrolün kendisi maliyet ve saldırı yüzeyi.** LLM yargıcı ölçüldüğünde
   varsayılandan çok daha kötü (p_detect 0,548 — varsayım 0,90) `[T]`; LLM tabanlı
   guardrail zehirlenmiş tek bir belgeyle 13–63× token, 148× gecikme amplifikasyonuna
   sokulabiliyor `[T]`.

---

## 1. Tekrar tespiti algoritmaları — yöntem, kırılma noktaları

### 1.1 Tam eşleşme / imza karşılaştırması: hangi alanlar sorun çıkarır

**Nasıl çalışır.** Her eylem `(araç adı, argümanlar)` çiftinden kanonik bir imzaya
indirgeniyor; ardışık (veya pencere içinde) aynı imza sayılıyor. `harness_kontrolleri.md`
§5.1'deki beş implementasyonun tamamı bu ailede.

**Nerede yanılır — 1: değişken alanlar.** İmzaya giren tek bir değişken alan tespiti
tamamen kapatır. OpenHands bunu bilerek çözüyor; `stuck_detector.py` `_event_eq()`
metodu ID'leri **açıkça** atlıyor `[K]`:

```python
if isinstance(event1, ActionEvent) and isinstance(event2, ActionEvent):
    return (
        event1.source == event2.source
        and event1.thought == event2.thought
        and event1.action == event2.action
        and event1.tool_name == event2.tool_name
        # Ignore tool_call_id, llm_response_id, action_id as they vary
    )
```

Atlanan üç alan: `tool_call_id`, `llm_response_id`, `action_id`. **Ama bu liste eksik.**
Kodda kalan alanlardan hangileri her turda değişebilir:

| Alan | Nerede | Neden değişir | Sonuç |
|---|---|---|---|
| `event1.thought` | ActionEvent karşılaştırması | Modelin serbest metin gerekçesi. T>0'da neredeyse hiçbir zaman byte-özdeş değil | **En büyük yanlış negatif kaynağı.** Aynı komutu 4 kez çağıran agent, her seferinde "Let me try again" / "Trying once more" yazarsa OpenHands'in 1. ve 4. senaryosu tetiklenmez |
| `observation` içeriği | ObservationEvent karşılaştırması | Zaman damgası, süre ("took 1.4s"), PID, port, geçici dosya adı, satır sayısı, ANSI renk kodu, "attempt N" sayacı | Senaryo 1 **hem** eylem hem gözlem eşitliği istiyor; gözlemde tek bir değişken alan yeterli |
| argüman anahtar sırası | Gemini CLI `getToolCallKey` | `JSON.stringify(toolCall.args)` — **anahtarları sıralamıyor** `[K]` | `{"path":"a","mode":"r"}` ile `{"mode":"r","path":"a"}` farklı sha256 üretir. Cline (`sortKeys`) ve Roo (`safe-stable-stringify`) bu tuzağı kapatmış, Gemini CLI ve opencode kapatmamış |
| kayan nokta / rastgele tohum | argümanlarda | `temperature`, `seed`, `timestamp`, `request_id` alanları argümana giriyorsa | imza her turda yeni |

**Bu, PoC'nin en kolay gösterilebilir sınırı.** Aynı `bash("pytest")` çağrısını 5 kez
tekrarlayan bir izde, gözlem çıktısına yalnızca `"finished in 1.42s"` gibi değişen bir
süre eklemek OpenHands senaryo 1'i kapatır.

**Nerede yanılır — 2: anlamsal olarak aynı, sözdizimsel farklı.**
`ls -la` / `ls -al` / `ls -a -l`, `grep -n foo` / `grep --line-number foo`,
`cat a.py` / `sed -n '1,200p' a.py`, aynı sorunun yeniden ifade edilmesi. Hiçbir
harness'ın imza fonksiyonu komut satırını normalize etmiyor (`harness_kontrolleri.md`
§5.1'deki beş imza fonksiyonu da düz serileştirme). Gemini CLI'ın **LLM yargıcı** bunu
prompt düzeyinde hedefliyor ("semantically equivalent arguments") ama yargıç 30. turdan
önce hiç çalışmıyor `[K]` — yani ilk 30 turdaki anlamsal tekrar hiçbir katmanda
yakalanmıyor.

**Nerede yanılır — 3: gözlem hiç bakılmıyor.** Gemini CLI, Cline, Roo Code ve
opencode yalnızca **isteği** hash'liyor; aracın döndürdüğü sonuca bakmıyor `[K]`.
İki yönlü hata:
- *Yanlış pozitif:* aynı çağrı farklı sonuç veriyor olabilir (`git log` sayfalama,
  `poll_status()`, kuyruk tüketimi). Tekrar var, döngü yok.
- *Yanlış negatif:* farklı çağrılar aynı hatayı veriyor olabilir. Agent argümanı her
  turda biraz değiştiriyor ama duvar aynı. Yalnızca OpenHands (senaryo 2:
  eylem+`AgentErrorEvent` serisi) ve deer-flow bu tarafa bakıyor.

### 1.2 Dönüşümlü (A-B-A-B) ve daha uzun çevrimler

`harness_kontrolleri.md` §5.1 şunu tespit etmişti: 22 harness'tan yalnızca **2'si**
(Gemini CLI, OpenHands) dönüşümlü döngüyü yakalıyor. Bu turda ikisinin de kodu satır
satır okundu; **ikisi de sanıldığından dar.**

#### Gemini CLI — k=1..5 çevrim taraması `[K]`

`packages/core/src/services/loopDetectionService.ts`, `checkToolCallLoop` (satır 313–349):

```ts
const key = this.getToolCallKey(toolCall);
this.toolCallHistory.push(key);
const maxRequiredLength = 5 * TOOL_CALL_LOOP_THRESHOLD;   // 25
if (this.toolCallHistory.length > maxRequiredLength) {
  this.toolCallHistory = this.toolCallHistory.slice(-maxRequiredLength);
}
const n = this.toolCallHistory.length;
const R = TOOL_CALL_LOOP_THRESHOLD;                       // 5

for (let k = 1; k <= 5; k++) {
  const requiredLength = k * R;                           // 5,10,15,20,25
  if (n >= requiredLength) {
    const cycle = this.toolCallHistory.slice(-k);         // son k anahtar = varsayılan çevrim
    let isPatternMatch = true;
    for (let i = 0; i < requiredLength; i++) {
      const indexFromEnd = requiredLength - i;
      const actualKey = this.toolCallHistory[n - indexFromEnd];
      const expectedKey = cycle[i % k];
      if (actualKey !== expectedKey) { isPatternMatch = false; break; }
    }
    if (isPatternMatch) return true;
  }
}
```

**Nasıl çalışır:** son `k` anahtar bir "çevrim şablonu" varsayılıyor ve bu şablonun
son `k×5` pozisyonda **kesintisiz** tekrar edip etmediği kontrol ediliyor. k=1'den
başlayıp ilk eşleşmede duruyor.

**Kaçırdığı durumlar — dört tane, hepsi PoC'de tetiklenebilir:**

1. **k > 5 çevrimler yapısal olarak imkânsız.** Geçmiş tamponu 25 anahtarla sınırlı
   (`slice(-25)`), k=6 için 30 anahtar gerekirdi. Yani **6 adımlık bir döngü sonsuza
   kadar dönebilir ve bu dedektör onu hiç göremez.** Ajanlar tipik olarak
   `read → edit → build → test → read → edit → build → test` gibi 4–8 adımlık
   çevrimlere girer; 6'nın üstü bu dedektörün kör noktası.
2. **Tek bir yabancı çağrı deseni sıfırlar.** Kontrol tam eşleşme; A-B-A-B-A-**X**-A-B-A-B
   dizisinde döngü kırılmış sayılır ve sayaç yeniden 5 tekrar bekler. Gerçek agentlar
   döngü içinde ara sıra bir `ls` ya da `read_file` atar; bu, gürültülü ama gerçek
   bir döngüyü görünmez yapar.
3. **Faz kayması.** Şablon **son k anahtardan** alınıyor. Dizi A-B-A-B-A-B ise k=2
   şablonu `[A,B]` ve geriye doğru kontrol tutar. Ama dizi henüz yarım bir çevrimle
   bitiyorsa (…A-B-A-B-A) k=2 için son 10 pozisyon `[B,A]` şablonuyla test edilir ve
   yine tutar — bu tamam. Asıl sorun **k'nın gerçek periyodun katı olması**: A-B-A-B
   için k=4 şablonu `[A,B,A,B]` de eşleşir, ama k=1'den başlandığı için k=2'de zaten
   yakalanır. Yani faz değil, **eşiğin çarpımsal olması** sorun: k=5 çevrim için
   **25 araç çağrısı** gerekiyor. Sonnet sınıfı bir modelde bu ≈25 tur × çağrı başına
   maliyet demek — döngü, tespit edilene kadar zaten pahalı.
4. **Yalnızca araç çağrısı sayılıyor.** `addAndCheck` sadece `ToolCallRequest` ve
   `Content` olaylarına bakıyor `[K]`. Araç çağırmadan düşünüp duran agent
   (OpenHands'in "monolog" senaryosu) tool-call dedektöründe görünmez; onu ancak
   `checkContentLoop` (aynı 50 karakterlik chunk'ın 10 kez tekrarı) veya 30. turdan
   sonra LLM yargıcı yakalayabilir.

Ek bir sınır: `resetContentTracking()` her araç çağrısında içerik takibini sıfırlıyor
ama `toolCallHistory` yalnızca `reset(promptId)`'de (satır 744–765) — yani **yeni bir
kullanıcı prompt'unda** sıfırlanıyor. Bu doğru tasarım, ama tersi anlamına da geliyor:
kullanıcı araya girerse döngü kanıtı silinir.

#### OpenHands — `alternating_pattern=6` gerçekte ne yapıyor `[K]`

`stuck_detector.py`, `_is_stuck_alternating_action_observation`:

```python
threshold = self.alternating_pattern_threshold   # 6
# ... son 6 action ve son 6 observation ayrı listelere toplanıyor ...
actions_equal = all(
    self._event_eq(last_actions[i], last_actions[i + 2])
    for i in range(threshold - 2)          # i = 0,1,2,3
)
observations_equal = all(
    self._event_eq(last_observations[i], last_observations[i + 2])
    for i in range(threshold - 2)
)
```

**Bu, k=2 dışında hiçbir periyodu yakalamaz.** `i+2` sabit; kontrol "her eleman kendinden
iki önceki ile aynı mı" sorusu. A-B-C-A-B-C (k=3) deseni **tamamen kaçar**. Yani
`harness_kontrolleri.md`'nin "A-B-A-B'yi yakalayan iki harness" ifadesi doğru, ama
OpenHands için tam olarak *sadece* A-B-A-B'dir; genel çevrim taraması yalnızca
Gemini CLI'da var ve orada da k≤5.

**Eşik 6 nereden geliyor?** Kodda ve `types.py`'deki `StuckDetectionThresholds` alan
açıklamasında **hiçbir gerekçe yok** `[K]`:

```python
alternating_pattern: int = Field(
    default=6, ge=1, description="Threshold for alternating pattern detection"
)
```

Tek dolaylı ipucu tarama penceresi sabitinin yorumu:

```python
# Maximum recent events to scan for stuck detection.
# This window should be large enough to capture repetitive patterns
# (4 repeats × 2 events per cycle = 8 events minimum, plus buffer for user messages)
MAX_EVENTS_TO_SCAN_FOR_STUCK_DETECTION: int = 20
```

Yani pencere 20, `action_observation=4` eşiğinden türetilmiş (4×2=8 + tampon). 6 sayısı
için karşılık gelen bir hesap yok. **Ölçülmüş bir kalibrasyon bulunamadı** — ne kodda,
ne test dosyalarında, ne de OpenHands makalesinde. Aynı şey `4 / 3 / 3` için de geçerli.
(§4'te eşik gerekçelerinin tam dökümü.)

**İki ek OpenHands sınırı — envanterde yazılı olmayanlar:**

- **5 senaryodan biri fiilen çalışmıyor.** `_is_stuck_context_window_error` gövdesi
  koşulsuz `return False` `[K]`:
  ```python
  def _is_stuck_context_window_error(self, _events: list[Event]) -> bool:
      # TODO: blocked by https://github.com/OpenHands/agent-sdk/issues/282
      return False
  ```
  Yani sınıf docstring'i 5 senaryo sayıyor, gerçekte **4 tanesi aktif**. Context-window
  hata döngüsü — `loop_budget.md` §3'teki Claude Code CCDE-001 "compaction-loop"
  imzasının tam karşılığı — OpenHands'te de yakalanmıyor.
- **Pencere son kullanıcı mesajından sonrasıyla sınırlı** (`_events_since_last_user_message`).
  Etkileşimli oturumda kullanıcının "devam et" demesi tespit durumunu sıfırlıyor.
  Roo Code'un `mistake_limit_reached` sonrası sayaç sıfırlaması ile aynı sınıf tuzak:
  **insan müdahalesi, döngü kanıtını siliyor.**

#### Klasik algoritmalar bu probleme nasıl uyarlanır — ve neden çoğu uymaz

Brief Floyd/Brent'i soruyor. Kısa cevap: **uymuyorlar, ve nedeni öğretici.**

| Algoritma | Ne varsayar | Karmaşıklık | Agent izine uyar mı |
|---|---|---|---|
| **Floyd tortoise-hare** (Knuth, TAOCP Vol. 2, §3.1 Alıştırma 6) | `x_{n+1} = f(x_n)` — **deterministik** ardıl fonksiyon, sonlu durum uzayı | O(μ+λ) zaman, **O(1) bellek** | ❌ Agent durumu deterministik ardıl değil: aynı durumdan farklı eylem üretilebilir (T>0), üstelik durum monoton büyüyor (bağlam birikiyor) → aynı durum ikinci kez hiç ziyaret edilmez. Fonksiyonel grafik varsayımı çöker |
| **Brent** (R. P. Brent, *BIT* 20(2):176–184, 1980) | aynı | O(μ+λ), O(1) bellek, Floyd'dan daha az `f` çağrısı | ❌ aynı gerekçe |
| **KMP hata fonksiyonu ile en küçük periyot** (Knuth–Morris–Pratt, *SIAM J. Comput.* 6(2), 1977) | Bir **dizi** (izlenen imza dizisi) verili | **O(n)** zaman/bellek; `periyot = n − failure[n]`, dizi `n % periyot == 0` ise tam tekrar | ✅ **Doğru araç.** Çevrim uzunluğu bilinmezken tek geçişte en küçük periyodu verir. Gemini CLI'ın k=1..5 döngüsünün (O(k·k·R) = O(125) sabit iş) genelleştirilmiş, k sınırı olmayan hali |
| **Z-algoritması / suffix automaton ile tekrar bulma** (Main–Lorentz 1984; Crochemore 1981) | dizi | O(n log n) / O(n) | ✅ tüm maksimal tekrarları bulur; agent izinde aşırı güçlü ama "iç içe tekrar" analizinde kullanılabilir |
| **Rabin–Karp yuvarlanan hash** (Karp & Rabin, *IBM J. Res. Dev.* 31(2), 1987) | dizi + sabit uzunluk | O(n) beklenen, çakışma riski | ✅ Gemini CLI'ın `checkContentLoop`'u fiilen bunun basitleştirilmiş hali: 50 karakterlik chunk hash'i + `isActualContentMatch()` ile çakışma doğrulaması |
| **n-gram tekrar sayımı** | dizi | O(n·k) | ✅ en ucuz yaklaşım; opencode/Cline/Roo'nun "son 3 aynı" heuristiği n=1 özel hali |

**PoC için somut öneri:** imza dizisinin **son bir penceresi** üzerinde KMP hata
fonksiyonu ile en küçük periyodu hesapla. Tek geçiş, O(n), k üst sınırı yok, faz
kaymasından etkilenmiyor. Gemini CLI'ın k≤5 sınırını ve OpenHands'in k=2 sınırını
aynı anda kaldırır. Eşik olarak "periyot p, pencerede en az R kez tekrarlanmış" kuralı
korunur.

⚠️ **Ama bu da tam eşleşmeye dayanır** — §1.1'deki bütün değişken-alan sorunları aynen
geçerli. Periyot bulma, imza kalitesini iyileştirmez; yalnızca *hangi periyotların*
görülebildiğini genişletir.

### 1.3 Meşru tekrar ile döngüyü ayırmak — yayınlanmış ölçüt var mı

**Evet, iki tane var, ikisi de tam metin okundu.**

**(a) IAL-SCAN'in "benign loop filtering" listesi** `[T]` (https://www.alphaxiv.org/abs/2607.01641).
Bound verification'dan **önce** elenen döngü sınıfları makalede açıkça sayılıyor:

> "Cycles that correspond to bounded iteration patterns, stream consumers, parsers,
> **pagination loops**, lifecycle loops, or test scaffolding are filtered before bound
> verification."

Ve bu filtrenin ölçülmüş katkısı var (ablasyon, Tablo III): benign-loop filtreleme
kapatılınca alarm 74 → **103**, yanlış pozitif 6 → **41**, token 4,2K → 16,3K.
Yani meşru tekrarı ayıklamak, yanlış pozitiflerin **yaklaşık %85'ini** üretiyor.
Bu, "aynı aracın 50 dosya üzerinde çağrılması döngü değil" sezgisinin ölçülmüş hali.

**(b) Gemini CLI'ın LLM yargıcı sistem prompt'u** `[K]` — bu envanterdeki **tek**
operasyonel "meşru tekrar" tanımı, ve doğrudan kopyalanabilir. İki koşulu **birlikte**
şart koşuyor:

> "An unproductive state requires BOTH of the following to be true:
> 1. The assistant has exhibited a repetitive pattern over at least 5 consecutive model actions…
> 2. The repetition produces NO net change or forward progress toward the user's goal."

ve dört sınıfı açıkça muaf tutuyor:

| Muaf sınıf | Prompt'taki tanım |
|---|---|
| **Cross-file batch operations** | "same tool name but targeting different files… adding license headers to 20 files" |
| **Incremental same-file edits** | "different line ranges, different functions, or different text content" |
| **Sequential processing** | "read or search operations on different files/paths" |
| **Retry with variation** | "Re-attempting a failed operation with modified arguments" |

Ayrıca argüman analizini zorunlu kılıyor ("You MUST compare the **arguments** of each
call, not just the tool name") ve **kullanıcı isteğini bağlama katıyor**: istek
"update all files" gibi toplu bir işi ima ediyorsa tekrar beklenen davranış sayılıyor.

**Kritik boşluk:** bu ölçüt yalnızca **LLM yargıcı** katmanında uygulanıyor. Gemini
CLI'ın ucuz deterministik dedektörü (`checkToolCallLoop`) bu muafiyetlerin hiçbirini
bilmiyor — çünkü hash'ler zaten farklı olacağı için pratikte tetiklenmiyor. Yani
"meşru tekrar" ayrımı deterministik katmanda **imzanın farklı çıkmasına** güveniyor,
açık bir kurala değil. 50 dosyaya aynı header'ı ekleme senaryosunda argümanlar farklı
olduğu için hash farklı → alarm yok. Ama **sayfalama** (`fetch(page=1)`, `fetch(page=2)`…)
aynı şekilde argüman farkıyla kurtuluyor; **yeniden deneme politikası** (aynı çağrı,
üstel geri çekilme) kurtulmuyor. Yani deterministik katmanda meşru tekrarın bir kısmı
korunuyor, bir kısmı korunmuyor ve bu ayrım **kasıtlı değil, tesadüfi.**

### 1.4 PoC'de nasıl test edilir — §1 için senaryolar

| # | Senaryo | Ne göstermeli | Beklenen sonuç |
|---|---|---|---|
| **S1** | Aynı `bash("pytest -q")` çağrısı 6 kez; gözlem çıktısına her turda değişen bir süre damgası (`finished in 1.4s` / `1.6s`…) ekle | Değişken alan yanlış negatifi | Ardışık-imza dedektörleri tetiklenir (istek aynı), **OpenHands senaryo 1 tetiklenmez** (gözlem farklı) |
| **S2** | Aynı çağrı 6 kez, ama modelin `thought` metni her turda farklı ifade edilmiş | `thought` alanının imzaya dahil olmasının bedeli | OpenHands'in 4 aktif senaryosunun hiçbiri tetiklenmez |
| **S3** | `ls -la` → `ls -al` → `ls -a -l` → … 6 tur | Anlamsal denklik | Hiçbir deterministik dedektör tetiklenmez; yalnızca LLM yargıcı (30. turdan sonra) |
| **S4** | 6 adımlık çevrim: `read → edit → build → test → log → revert` × 5 | Gemini k≤5 sınırı ve OpenHands k=2 sınırı | Her ikisi de kaçırır; **KMP periyot bulucu yakalar** |
| **S5** | A-B-A-B-A-B-**X**-A-B-A-B-A-B (araya tek yabancı çağrı) | Kesintisiz eşleşme şartı | Gemini CLI sayacı sıfırlanır; kayan pencereli periyot analizi yakalar |
| **S6** | Aynı `fetch_page(n)` aracı, n=1..50 (sayfalama) | **Yanlış pozitif kontrolü** | Hiçbir dedektör tetiklenmemeli. Tetikleniyorsa imza fonksiyonu argümanı yok sayıyor demektir |
| **S7** | Aynı `write_file` aracı, 50 farklı dosyaya aynı lisans başlığı | **Yanlış pozitif kontrolü** (Gemini prompt'undaki birinci muafiyet) | Tetiklenmemeli |
| **S8** | Üstel geri çekilmeli retry: aynı çağrı, aralar 1s/2s/4s/8s | Meşru retry vs. döngü ayrımı | Deterministik dedektörler döngü der. **Bu kasıtlı bir yanlış pozitif ve bilinen sınır olarak yazılmalı** |

