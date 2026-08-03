# 9. Ölçüm ve Değerlendirme

> **Bölümün tezi:** Bağlam mühendisliği hataları **standart değerlendirmelerde görünmez.** Kısa test oturumlarında geçen bir ajan uzun oturumda bozulur ve suç yanlışlıkla modele atılır. Bu bölüm, bağlam kalitesini ayrıştırıp ölçülebilir hâle getiren yöntemleri sunar.

---

## 9.1 Neden standart eval yetmez

> *"Context engineering failures are often **invisible in standard evaluations.** An agent may perform well in short test sessions but degrade in longer ones, with failures **incorrectly attributed to reasoning** instead of context management."*
> — ML Mastery, Nis 2026

Sorunun yapısı:

```
Kısa test oturumu (3–5 tur)     → bağlam küçük, rot yok, her şey çalışır  ✅
Üretim oturumu (40+ tur)        → bağlam büyük, compaction devrede,
                                   tool sonuçları budanmış, hafıza bayat  ❌
```

Test ile üretim arasındaki fark modelde değil, **bağlamın durumunda**. Dolayısıyla ölçüm de bağlam durumunu hedeflemelidir.

---

## 9.2 Probe tabanlı değerlendirme

Yöntem: **sıkıştırma veya getirme adımından sonra**, saklanmış belirli bilgiyi gerektiren hedefli sorular sor. Doğru cevap → ilgili bağlam korunmuş. Yanlış cevap → sıkıştırma veya retrieval kalitesinde sorun.

Factory.ai çerçevesinde üç probe tipi:

| Probe | Sorusu | Neyi test eder | Hangi hatayı yakalar |
|---|---|---|---|
| **Recall** | "X hakkında ne karar vermiştik?" | Belirli olguları hatırlıyor mu | Aşırı agresif compaction |
| **Artifact** | "Şu ana kadar hangi dosyaları değiştirdin?" | Kendi eylemlerinin farkında mı | Tool sonucu budama hataları |
| **Continuation** | "Şu an hangi adımdasın, sıradaki ne?" | Çok adımlı görevi sürdürebiliyor mu | Durum kaybı, drift |

### Uygulama

```python
PROBES = {
    "recall":       "Bu oturumda hangi kütüphaneyi seçtik ve gerekçesi neydi?",
    "artifact":     "Şu ana kadar hangi dosyaları oluşturdun veya değiştirdin? Listele.",
    "continuation": "Şu an görevin hangi adımındasın ve sıradaki adım nedir?",
}

def probe(messages, ground_truth: dict[str, str]) -> dict[str, bool]:
    """Compaction/budama sonrası bağlam kalitesini ölçer.
    messages: o anki bağlam. ground_truth: beklenen cevabın anahtar bilgisi."""
    results = {}
    for kind, question in PROBES.items():
        r = client.messages.create(
            model="claude-opus-5", max_tokens=512,
            messages=messages + [{"role": "user", "content": question}],
        )
        answer = "".join(b.text for b in r.content if b.type == "text")
        results[kind] = grade(answer, ground_truth[kind])
    return results


def grade(answer: str, expected_key: str) -> bool:
    """Basit: anahtar bilgi geçiyor mu. Sağlamı: LLM-as-judge."""
    return expected_key.lower() in answer.lower()
```

> **Not:** Probe sorusu bağlama eklenir ve orayı kirletir. Üretimde probe'ları **ayrı bir kopya** üzerinde çalıştır, ana konuşmaya karıştırma.

### Ne zaman çalıştırılır

```
compaction tetiklendikten HEMEN SONRA          → sıkıştırma kalitesi
context editing uygulandıktan sonra            → budama kalitesi
her N turda bir (ör. 10)                       → kademeli drift takibi
oturum sonunda                                 → toplam sağlık
```

---

## 9.3 Metrikler

| Metrik | Tanım | Nasıl ölçülür | Sağlıklı aralık |
|---|---|---|---|
| **Context utilization rate** | Bütçenin yüzde kaçı kullanılıyor | `total_prompt / max_input_tokens` | %60–80 ⚠️ sezgisel |
| **Compression ratio** | Özetlemeyle token azalması | `post_compaction / pre_compaction` | Göreve bağlı |
| **Retrieval precision** | Getirilen chunk'lar **gerçekten kullanılıyor mu** | Aşağıda | Yüksek |
| **Cache hit rate** | `cache_read / total_prompt` | `usage` alanlarından | 2. turdan sonra yüksek |
| **Tur başına token** | Ortalama prompt büyüklüğü | Turlar boyunca izle | Doğrusal büyümeli, üstel değil |

### Retrieval precision nasıl ölçülür

Metrik kolay tanımlanır, ölçmesi zordur. Üç yaklaşım:

```python
# a) Atıf zorunluluğu — en güvenilir
#    Her chunk'a id ver, modelden kullandıklarını belirtmesini iste
prompt = "Aşağıdaki kaynakları kullan. Cevabında kullandığın her kaynağın [id]'sini belirt."
used = extract_ids(response)          # → precision = len(used) / len(injected)

# b) Ablasyon — en kesin, en pahalı
#    Chunk'ı çıkar, cevap değişiyor mu bak
for i, chunk in enumerate(chunks):
    baseline = ask(chunks)
    without  = ask(chunks[:i] + chunks[i+1:])
    if similar(baseline, without):
        print(f"chunk {i} etkisiz")

# c) İşaretleyici — hızlı, kaba
#    Her chunk'a nadir bir token koy, cevapta ara
```

---

## 9.4 Context drift göstergeleri

Uzun koşan oturumlarda **davranışsal** uyarı işaretleri:

```
⚠  ajan zaten işlediği dosyayı yeniden okuyor
⚠  zaten verdiği kararı yeniden ifade ediyor
⚠  görevi orijinal kullanıcı niyetinden uzağa yeniden çerçeveliyor
```

> *"These patterns appear in **step-level traces before they surface in output quality**."*

Bunlar **öncü göstergelerdir** — çıktı bozulmadan önce izlerde görünürler. Otomatik tespit mümkündür:

```python
class DriftDetector:
    def __init__(self):
        self.files_read: dict[str, int] = {}
        self.decisions: list[str] = []
        self.original_intent: str | None = None

    def observe_tool_use(self, name: str, inp: dict):
        if name in ("Read", "read_file"):
            path = inp.get("file_path") or inp.get("path")
            self.files_read[path] = self.files_read.get(path, 0) + 1
            if self.files_read[path] >= 3:
                yield f"DRIFT: {path} {self.files_read[path]} kez okundu"

    def observe_text(self, text: str):
        # basit yaklaşım: karar cümlelerini topla, tekrar oranına bak
        for d in extract_decision_sentences(text):
            if any(similar(d, prev) > 0.85 for prev in self.decisions):
                yield f"DRIFT: karar tekrarı — {d[:60]}"
            self.decisions.append(d)

    def check_intent(self, current_summary: str):
        drift = 1 - similarity(self.original_intent, current_summary)
        if drift > 0.4:
            yield f"DRIFT: niyet sapması {drift:.0%}"
```

Bu detektör, §08.5'teki **context poisoning**'e karşı da erken uyarı sağlar: model yanlış bir varsayımı tekrar tekrar ifade etmeye başlarsa "karar tekrarı" sinyali verir.

---

## 9.5 Deneyler

Raporun iddiaları ölçülebilir. Dört deney önerilir.

### Deney 1 — "Bilme bağlamdadır" (Bulgu 3)

Tek değişken: `description` string'i. Model, ağırlıklar, soru sabit.

```python
BASE = {"name": "search",
        "input_schema": {"type": "object",
                         "properties": {"query": {"type": "string"}},
                         "required": ["query"]}}

VARYANTLAR = {
    "boş":         "",
    "tanım":       "Web'de arama yapar.",
    "tetikleyici": ("Web'de arama yapar. Cevap konuşmada bulunmayan güncel bilgiye "
                    "bağlıysa (son olaylar, güncel fiyatlar) hafızadan cevaplama, "
                    "önce bunu çağır."),
}

N = 20
for etiket, desc in VARYANTLAR.items():
    hit = sum(
        any(b.type == "tool_use" for b in client.messages.create(
            model="claude-opus-5", max_tokens=1024,
            tools=[{**BASE, "description": desc}],
            messages=[{"role": "user", "content": "Bugün Bitcoin kaç dolar?"}],
        ).content)
        for _ in range(N)
    )
    print(f"{etiket:<12} → tetiklenme: {hit}/{N}")
```

**Beklenen:** aşağı doğru oran artar. İkinci varyant: `name`'i `search` → `xq7` yap, description sabit tut — ad da sinyaldir, düşüş beklenir.

### Deney 2 — Cache disiplini

```python
VARYANTLAR = {
    "A_kontrol":      lambda: (SABIT_SYSTEM, SABIT_TOOLS),
    "B_zaman_damgasi": lambda: (SABIT_SYSTEM + f"\nŞu an: {datetime.now()}", SABIT_TOOLS),
    "C_karisik_tool":  lambda: (SABIT_SYSTEM, random.sample(SABIT_TOOLS, len(SABIT_TOOLS))),
}
```

| Varyant | Beklenen |
|---|---|
| A (kontrol) | 2. turdan itibaren `cache_read > 0` |
| B (zaman damgası) | **Her turda `cache_read = 0`** |
| C (tool sırası değişken) | **Her turda `cache_read = 0`** |

### Deney 3 — `defer_loading` etkisi

30 sahte tool tanımla, 25'ine `defer_loading: True` ver. Tur başına toplam prompt token'ını iki durumda karşılaştır.

### Deney 4 — Arama hunisi vs tam okuma

Aynı soruyu iki yolla cevaplat:

| Yol | Yöntem |
|---|---|
| A | İlgili tüm dosyaları `Read` ile oku, sonra cevapla |
| B | `glob → rg -l → rg -C → Read(offset)` hunisi |

Ölçülen: toplam prompt token'ı, tur sayısı, cevabın doğruluğu. **Bu deneyin grafiği raporun en ikna edici tek görseli olabilir.**

### Ortak ölçüm kodu

```python
def log_usage(turn: int, response) -> dict:
    u = response.usage
    total = (u.input_tokens
             + (u.cache_creation_input_tokens or 0)
             + (u.cache_read_input_tokens or 0))
    row = {
        "turn": turn,
        "total_prompt": total,
        "fresh": u.input_tokens,
        "cache_read": u.cache_read_input_tokens or 0,
        "cache_create": u.cache_creation_input_tokens or 0,
        "output": u.output_tokens,
        "stop_reason": response.stop_reason,
    }
    print(f"tur {turn:>2} | stop={row['stop_reason']:<9} "
          f"prompt={total:>7} fresh={row['fresh']:>6} "
          f"cache_read={row['cache_read']:>7} out={row['output']}")
    return row
```

> **Metodolojik uyarı:** `usage.input_tokens` yalnızca cache'lenmemiş kısmı gösterir. Toplam prompt = `input_tokens + cache_creation + cache_read`. Bu üçü toplanmadan yapılan her maliyet analizi yanlıştır.

---

## 9.6 Wire log — bağlamı gözle görmek

Teoriyi doğrulamanın en hızlı yolu, giden gövdeyi diske yazmaktır.

```python
import json, pathlib
OUT = pathlib.Path("wire-log"); OUT.mkdir(exist_ok=True)

# her turda:
(OUT / f"turn-{turn:02d}-request.json").write_text(
    json.dumps(request, ensure_ascii=False, indent=2))
(OUT / f"turn-{turn:02d}-response.json").write_text(
    response.to_json(indent=2))
```

`turn-01-request.json` ile `turn-02-request.json` arasındaki diff, "bağlam her turda yeniden gönderiliyor" iddiasını **gözlemsel kanıta** dönüştürür.

---

## 9.7 Optimizasyon döngüsü

```
gerçek oturumlarda baseline metrikleri belirle
        ↓
yüksek maliyetli / düşük kaliteli segmentleri bul
        ↓
hedefli düzeltme uygula
        ↓
etkiyi ölç
        ↓ (tekrarla)
```

Ve son uyarı:

> *"**Over-compression can save tokens but hurt accuracy, shifting the problem instead of solving it.** The goal is the **minimum viable context** that still lets the agent complete its task correctly."*

Bu, raporun tezinin ölçüm tarafındaki karşılığıdır: hedef en küçük bağlam değil, **görevi doğru tamamlamaya yeten en küçük bağlamdır.** İki metrik birlikte izlenmelidir — yalnızca token'a bakan bir optimizasyon, kaliteyi sessizce feda eder.

---

## 9.8 Bulgu

> **Bulgu 13.** Bağlam kalitesi, çıktı kalitesinden **bağımsız olarak ölçülebilir ve ölçülmelidir.** Probe tabanlı değerlendirme (recall / artifact / continuation), sıkıştırma ve budama adımlarının kayıplarını izole eder; drift göstergeleri ise bozulmayı çıktıya yansımadan önce yakalar. Ölçüm yapılmayan bir bağlam mühendisliği, tanım gereği bir tahmindir.
