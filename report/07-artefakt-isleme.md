# 7. Artefakt İşleme: Bağlama Girmeyen Veriyle Çalışmak

> **Bölümün tezi:** Bağlam mühendisliğinin en saf hâli, veriyi bağlama **hiç sokmamaktır.** docx/pptx/xlsx işleme bunun kanonik örneğidir: modele bilgi değil, **bilgiyi çıkaran kod** ürettirilir. Sonuç: 5 MB'lık bir artefakt ~4.000 token'lık bir bütçeyle düzenlenebilir.

---

## 7.1 Neden bağlama koyulamaz

`.docx` / `.pptx` / `.xlsx` (OOXML) bir **ZIP arşividir**:

```
rapor.docx  (ZIP)
├── [Content_Types].xml
├── _rels/.rels
└── word/
    ├── document.xml          ← asıl içerik
    ├── styles.xml
    ├── numbering.xml
    ├── media/image1.png      ← binary
    └── _rels/document.xml.rels
```

Ve sinyal/gürültü oranı feci. 800 kelimelik iki sayfalık bir belgenin `document.xml`'i yaklaşık 80 KB:

```xml
<w:p><w:pPr><w:pStyle w:val="Normal"/><w:spacing w:before="0" w:after="160"
w:line="259" w:lineRule="auto"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Calibri"
w:hAnsi="Calibri" w:cs="Calibri"/><w:sz w:val="22"/></w:rPr><w:t>Gelir</w:t></w:r>
```

Bu blok tek bir kelime taşır: **"Gelir"**. Geri kalanı biçim markup'ıdır. Bağlama koymak, token'ın %95'ini gürültüye harcamak demektir.

---

## 7.2 Mekanizma: model dosyayı okumaz, okuyan kodu yazar

```
❌ dosya  →  bağlam  →  model
✅ dosya  →  sandbox  →  model KOD yazar  →  kod çalışır  →  SADECE ÇIKTI bağlama girer
```

Bu, Agent Skills'in (docx/pptx/xlsx/pdf) çalışma prensibidir. Model belge üretmez — **belge üreten kodu üretir.**

İki ortam:

| | Nerede çalışır | Nasıl kurulur |
|---|---|---|
| **API — Agent Skills** | Anthropic sandbox konteyneri | `container.skills` + `code_execution_20260521` |
| **Claude Code** | Yerel makine | `bash` + yerel Python |

Hazır kütüphaneler: `python-docx`, `python-pptx`, `openpyxl`, `pypdf`, `pdfplumber`, `pillow`, `matplotlib`.

### API akışı

```python
# 1. Yükle
up = client.beta.files.upload(file=open("rapor.docx", "rb"),
                              betas=["files-api-2025-04-14"])

# 2. Sandbox'a mount et + skill'i aç
r = client.beta.messages.create(
    model="claude-opus-5", max_tokens=16000,
    betas=["code-execution-2025-08-25", "skills-2025-10-02"],
    container={"skills": [{"type": "anthropic", "skill_id": "docx", "version": "latest"}]},
    tools=[{"type": "code_execution_20260521", "name": "code_execution"}],
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "3. bölümdeki 2025 rakamlarını 2026 ile güncelle."},
        {"type": "container_upload", "file_id": up.id},
    ]}],
)

# 3. Üretilen dosyayı indir
for b in r.content:
    if b.type == "bash_code_execution_tool_result":
        for f in (b.content.content or []):
            if f.type == "bash_code_execution_output":
                client.beta.files.download(f.file_id).write_to_file("rapor_v2.docx")
```

Dosya bağlama **hiç girmedi.** Bağlama giren: modelin yazdığı kod ve o kodun `stdout`'u.

---

## 7.3 Okuma yolu: belge yapısı üzerinde progressive disclosure

§06'daki arama hunisinin belgeye uygulanmış hâli. Tanımlayıcılar burada **paragraf indeksleri**.

**Adım 1 — haritayı çıkar (ucuz):**

```python
from docx import Document
doc = Document("/mnt/user-data/rapor.docx")

for i, p in enumerate(doc.paragraphs):
    if p.style.name.startswith("Heading"):
        print(f"{i:>4}  {p.style.name:<10}  {p.text[:60]}")
```

```
   0  Heading 1   Yönetici Özeti
  14  Heading 2   Metodoloji
  47  Heading 1   Finansal Sonuçlar
 112  Heading 2   2025 Gelir Analizi
 145  Heading 2   Maliyet Kırılımı
```

**~40 satır bağlama girdi. 5 MB'lık belge girmedi.**

**Adım 2 — model karar verir, sadece o bölümü çeker:**

```python
print("\n".join(p.text for p in doc.paragraphs[112:145]))
```

**Adım 3 — tablo gerekirse:**

```python
for row in doc.tables[2].rows:
    print(" | ".join(c.text for c in row.cells))
```

### Aynı desen, diğer formatlar

```python
# pptx — slayt haritası
from pptx import Presentation
prs = Presentation(path)
for i, slide in enumerate(prs.slides):
    title = next((sh.text_frame.text for sh in slide.shapes
                  if sh.has_text_frame and sh.text_frame.text), "—")
    print(f"slide {i:>2}: {title[:50]}  ({len(slide.shapes)} shape)")

# xlsx — önce şekil, sonra aralık
import openpyxl
wb = openpyxl.load_workbook(path, data_only=True)
for ws in wb.worksheets:
    print(f"{ws.title}: {ws.dimensions}  ({ws.max_row}×{ws.max_column})")
for row in wb["Gelir"]["A1:D20"]:
    print([c.value for c in row])
```

---

## 7.4 Düzenleme yolu

### İki yaklaşım, biri veri kaybı

```python
# ✅ CERRAHİ — sadece dokunduğunu değiştirir
doc = Document(path)          # mevcut belgeyi AÇ
...değişiklik...
doc.save(out)
# stiller, görseller, üstbilgi/altbilgi, yorumlar, değişiklik takibi → KORUNUR

# ❌ YENİDEN ÜRETİM — dokunmadığın her şeyi yok eder
doc = Document()              # BOŞ belge
doc.add_heading("Rapor")
...
# → orijinal biçimlendirme, görseller, meta veri: kayıp
```

"Yapay zekâ belgemi düzenledi ve formatı bozdu" şikâyetinin kaynağı ikincisidir. Skill'in işlevinin büyük kısmı, modele **birinciyi** yaptırmaktır.

### `runs` problemi — docx'in klasik tuzağı

Word metni **biçime göre parçalara (run) böler.** `"Gelir 2025'te $5M oldu"` bellekte şu olabilir:

```
run[0] = "Gelir "
run[1] = "2025"        ← kalın
run[2] = "'te $5M oldu"
```

Sonuçları:

```python
p.text                                    # okuma için doğru
p.text = p.text.replace("2025", "2026")   # ✗ p.text salt okunur

# ✗ Tüm run'ları silip tek run yazmak → kalın biçim gider

# ✓ Run bazında, biçimi koruyarak
for p in doc.paragraphs:
    for run in p.runs:
        if "2025" in run.text:
            run.text = run.text.replace("2025", "2026")
```

**Daha sinsi durum:** aranan metin run sınırını aşıyorsa hiçbir run içinde eşleşmez.

```
run[0] = "2"
run[1] = "025"      ← Word yazım denetimi böyle bölmüş olabilir
```

`"2025" in run.text` her ikisi için de `False`. **Sessiz başarısızlık.**

Sağlam çözüm — run haritası kurup sınır aşan eşleşmeleri yakala:

```python
def replace_in_paragraph(p, old, new):
    """Run sınırlarını aşan eşleşmeleri de yakalar, biçimi korur."""
    runs = p.runs
    full = "".join(r.text for r in runs)
    if old not in full:
        return False

    spans, pos = [], 0
    for i, r in enumerate(runs):
        spans.append((pos, pos + len(r.text), i))
        pos += len(r.text)

    start = full.index(old)
    end = start + len(old)
    touched = [i for (a, b, i) in spans if a < end and b > start]

    first = touched[0]
    for i in touched:
        a, b, _ = spans[i]
        seg_start, seg_end = max(a, start), min(b, end)
        local = runs[i].text
        runs[i].text = (local[: seg_start - a]
                        + (new if i == first else "")
                        + local[seg_end - a :])
    return True
```

> **Bu tür bir yardımcı fonksiyon skill'in `references/` klasöründe hazır bulunur.** Model her seferinde yeniden keşfetmez — okur ve kullanır. Skill'in üçüncü katmanı (§04.2) tam olarak bu işe yarar.

### xlsx'in kendi tuzağı

```python
wb = openpyxl.load_workbook(path, data_only=True)
wb.save(path)          # ☠️ TÜM FORMÜLLER YOK OLDU
```

`data_only=True` formüllerin **önbelleğe alınmış değerlerini** okur, formülün kendisini değil. Bu hâlde kaydedilirse `=SUM(B2:B10)` yerine `4200` yazılır — kalıcı veri kaybı.

| Amaç | Yükleme |
|---|---|
| Değer okumak | `data_only=True`, **kaydetme** |
| Düzenlemek | `data_only=False` (varsayılan) |

### pptx düzenleme

```python
prs = Presentation(path)
slide = prs.slides[4]

for sh in slide.shapes:
    if sh.has_text_frame:
        for p in sh.text_frame.paragraphs:
            for run in p.runs:                    # ← aynı run mantığı
                run.text = run.text.replace("Q3", "Q4")

if slide.has_notes_slide:                          # konuşmacı notları ayrı nesne
    print(slide.notes_slide.notes_text_frame.text)

prs.save(out)
```

Grafik düzenlemek daha derindir: pptx içindeki grafiğin verisi **gömülü bir xlsx parçasıdır**; basit değişiklikler için `chart.replace_data()` kullanılır.

### XML seviyesi

`python-docx` her OOXML özelliğini sarmalamaz. Gerekirse alt seviyeye inilir:

```python
from docx.oxml.ns import qn
sectPr = doc.sections[0]._sectPr
for el in sectPr.iter(qn("w:pgMar")):
    el.set(qn("w:left"), "1440")   # 1 inç = 1440 twip
```

---

## 7.5 PDF farklı bir yol izler

PDF, OOXML gibi yapılandırılmış-düzenlenebilir değildir. Üç seçenek vardır:

| Yol | Nasıl | Ne zaman | Maliyet |
|---|---|---|---|
| **Metin çıkarma** | `pdfplumber` / `pypdf` sandbox'ta | Metin ağırlıklı, sayfa çok | Sadece çıkarılan metin |
| **Native PDF** | `document` content bloğu | Düzen/tablo/grafik görsel olarak önemli | **~1.500–3.000 token/sayfa** |
| **Sayfayı görsele çevir** | `pdf2image` → `image` bloğu | Belirli sayfayı görsel incelemek | Görsel başına |

Native PDF'in ayrıcalığı: **model belgeye gerçekten bakar** — sayfalar hem metin hem görsel olarak işlenir. Tablo hizası, grafik, imza, damga yalnızca bu yolla görülür.

```python
{"type": "document",
 "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
 "citations": {"enabled": True}}     # atıf üretmesini sağlar
```

Sınırlar: 32 MB istek, 600 sayfa (200K bağlamlı modellerde 100). 300 sayfalık bir PDF'i native göndermek ~750K token demektir — o ölçekte metin çıkarma yoluna geçilir.

**Düzenleme:** PDF genelde yeniden üretilir (içeriği çıkar → `reportlab` ile yaz, ya da docx üretip dönüştür). Yerinde düzenleme, `pypdf` ile sayfa birleştirme/bölme/döndürme dışında pratik değildir.

---

## 7.6 Görsel doğrulama döngüsü

Kod yazıldı, dosya üretildi. **Gerçekten düzgün görünüyor mu?** Kod bunu söyleyemez.

```bash
soffice --headless --convert-to pdf sunum.pptx
pdftoppm -png -r 80 sunum.pdf slide
```

`slide-05.png` bir `image` bloğu olarak bağlama girer ve model **kendi çıktısına bakar**: metin kutudan taşmış mı, grafik ezilmiş mi, hizalama bozuk mu.

Bu, Opus 4.7 ve sonrasında belirgin şekilde iyileşen "kendi çıktısını görsel doğrulama" davranışının somut hâlidir ve raporun genel çerçevesinde **ajanın kendi ürettiği artefaktı denetleme döngüsü** olarak konumlandırılabilir.

---

## 7.7 Token muhasebesi

5 MB'lık, 40 slaytlık bir sunumda tek bir slaydı düzenlemek:

| Ne | Bağlama girer mi | ~Token |
|---|---|---|
| `.pptx` dosyası (5 MB) | ❌ | 0 |
| Açılmış XML (~2 MB) | ❌ | 0 |
| Skill talimatları | ✅ (bir kez) | 2.000 |
| Modelin yazdığı kod | ✅ | 300 |
| Slayt haritası (`stdout`) | ✅ | 400 |
| Hedef slaydın metni | ✅ | 200 |
| Düzenleme kodu | ✅ | 250 |
| Doğrulama `stdout` | ✅ | 100 |
| Render edilmiş görsel (opsiyonel) | ✅ | 1.500 |
| **Toplam** | | **~3.000–4.750** |

Ham XML bağlama konsaydı ~500.000 token olurdu ve büyük kısmı `<w:rPr>` etiketi olurdu.

---

## 7.8 Bulgu

> **Bulgu 10.** Artefakt işleme, üç mekanizmanın aynı anda çalıştığı bir vakadır: **(1)** dosya bağlam dışı durumda kalır (RAM/disk ayrımının disk tarafı, §01.4); **(2)** yapı haritası üzerinden kademeli açılma yapılır (progressive disclosure, §04.5); **(3)** modele bilgi değil, bilgiyi çıkaran kod ürettirilir. Üçüncüsü genel bir ilkeye işaret eder: **kod, bağlamın sıkıştırma katmanıdır.** Bir veriyi bağlama almak yerine onu işleyen kodu bağlama almak, sıkıştırma oranını yüzlerce kata çıkarabilir.

---

## 7.9 Genelleme: programmatic tool calling

Bulgu 10'daki ilke belge işlemeye özgü değildir. Genel hâline **programmatic tool calling (PTC)** denir ve üretimde bir harness tasarım deseni olarak kullanılmaktadır.

**Fikir:** tool'lar modele yalnızca konuşma turlarında çağrılan uç noktalar olarak değil, **sandbox içinden çağrılabilen Python fonksiyonları** olarak açılır. Model küçük bir program yazar; program getirir, filtreler, dallanır, döngü kurar, diske yazar ve eylem alır — tek çalıştırmada.

```python
# Model bunu yazar; sandbox çalıştırır. Bağlama sadece son print girer.
customers = crm.search(segment="enterprise", renewal_within_days=90)   # tool çağrısı
at_risk = []
for c in customers:                                     # 200 iterasyon
    tickets = support.list_tickets(c.id, since="-90d")  # tool çağrısı
    if sum(t.severity for t in tickets) > THRESHOLD:
        at_risk.append({"name": c.name, "score": ..., "top_issue": ...})

Path("/mnt/out/at_risk.json").write_text(json.dumps(at_risk))
print(f"{len(at_risk)}/{len(customers)} müşteri riskli. Detay: /mnt/out/at_risk.json")
```

Konuşmalı döngüde bu iş 400+ tur ve yüz binlerce token demektir. PTC'de **tek tur ve ~50 token'lık bir çıktı**.

### Dört kazanç

| Kazanç | Mekanizma |
|---|---|
| **Bağlam verimliliği** | Ara veriler Python değişkenlerinde ve dosyalarda kalır; orchestrator yalnızca özeti ve yapılandırılmış metadata'yı görür |
| **Gecikme** | Bağımsız işlemler tek çalıştırmada paralelleşir, LLM gidiş-dönüşlerine serileşmez |
| **Güvenilirlik** | Döngü, filtre, join ve koşullu dallanma kodda; konuşma turlarında modelin planı her adımda yeniden türetmesine gerek kalmaz (§03.5.3) |
| **Hata toparlanması** | **Stack trace** — ajan yapılandırılmış bir hata sinyali alır. Konuşmalı zincirde `is_error: true` + serbest metin varken, burada satır numarası ve çağrı yığını vardır |
| **Tutarlılık** | Script'ler skill olarak kaydedilebilir → yaygın işler için aynı çağrı dizisi tekrarlanabilir (§04) |

Sonuncusu bir döngüyü kapatır: **kod → skill → yeniden kullanılabilir bağlam-verimli yordam.** Ajanın bir kez keşfettiği verimli işlem dizisi kalıcı hâle gelir.

### API tarafında

```python
tools=[
    {"type": "code_execution_20260120", "name": "code_execution"},
    {"name": "crm_search", "description": "...", "input_schema": {...},
     "allowed_callers": ["code_execution_20260120"]},   # ← kod içinden çağrılabilir
]
```

`allowed_callers` alanı, tool'un konuşma turundan değil **çalışan kodun içinden** çağrılabileceğini bildirir. Çağrı yapıldığında konteyner durur, tool çalıştırılır ve sonuç **modelin bağlamına değil, çalışan koda** döner.

> **Kısıtlar:** `strict: true`, `disable_parallel_tool_use`, zorlanmış `tool_choice` ve MCP tool'larıyla birlikte kullanılamaz. Bekleyen bir programatik çağrıya yanıt verirken user mesajı **yalnızca** `tool_result` blokları içermelidir.

### Ne zaman PTC, ne zaman konuşmalı döngü

| PTC uygun | Konuşmalı döngü uygun |
|---|---|
| Çok sayıda ardışık/paralel çağrı | Az sayıda çağrı |
| Büyük ara sonuçlar (filtrelenmeli) | Küçük sonuçlar |
| Döngü, join, koşullu dallanma | Doğrusal akış |
| Toplu işlem, sayfalama | Tek seferlik sorgu |
| Adım adım kullanıcı onayı **gerekmiyor** | Her adımda insan kapısı **gerekiyor** |

Son satır önemli bir takastır: PTC hızı ve bağlam verimliliğini artırır ama **harness'in adım adım araya girme yeteneğini azaltır** — 20 tool çağrısı tek bir sandbox çalıştırmasının içinde gerçekleşir, harness her birini ayrı ayrı gate'leyemez. Yıkıcı eylemler için §03.9'daki onay kapıları hâlâ konuşma seviyesinde tutulmalıdır.

### Skill katmanlarının bu vakadaki karşılığı

| Skill katmanı | `docx` skill'inde |
|---|---|
| 1 — ad + açıklama | *"docx: Word belgesi oluşturma ve düzenleme"* — hep bağlamda |
| 2 — gövde | Cerrahi düzenleme prensibi, kütüphane seçimi, temel akış |
| 3 — referanslar | `runs` sınırı aşan replace fonksiyonu, stil koruma yardımcıları, XML kaçış yolları — **model ihtiyaç duyduğunda okur** |
