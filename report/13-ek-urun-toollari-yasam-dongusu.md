# 13-Ek — Ürün Tool'larının Trace-Compaction Yaşam Döngüsü (Kanıtlı)

**Ağustos 2026 · §13 uygulama eki · `poc-trace-compaction/product_tools.py` (119 tool, 9 toolkit)**

Bu ek, ürünün gerçek tool envanterinin (`toolsmockproduct/*.yaml`) trace-compaction sisteminden geçişini **çalışan koddan alınan kanıtlarla** anlatır. Beş soruya sırayla, her birine gerçek çıktıyla cevap verir:

1. Bu tool'larda **neleri barındırıyoruz?**
2. Bunlar **LLM'e nasıl verilecek?**
3. **Çıktıları** nasıl olacak?
4. **Trace'te nasıl değerlendirilecek?**
5. **Compact edildikten sonra** hali nasıl olacak?

> Her `KANIT` bloğu, ilgili scriptin gerçek çıktısıdır (üretilmiş, uydurulmamış). Yeniden üretmek için: `python run_product.py` ve `python test_product.py`.

**Kapsam kanıtı:** `YAML tool: 119 | DISPATCH: 119 | META: 119 | eksik: YOK | jenerik-fallback: YOK`. Kategori dağılımı: 55 write · 37 search · 27 read.

**İçindekiler**
1. [Bu tool'larda ne barındırıyoruz](#s1)
2. [LLM'e nasıl veriliyor](#s2)
3. [Çıktılar nasıl](#s3)
4. [Trace'te nasıl değerlendiriliyor](#s4)
5. [Compact edilmiş hal](#s5)
6. [Uçtan uca: bir soru, tam döngü](#s6)
7. [Kanıt özeti ve testler](#s7)

---

<a name="s1"></a>
## 1. Bu tool'larda ne barındırıyoruz

Her tool **iki katman** taşır: (a) LLM'e giden **tanım** (şema), (b) trace-compaction'a giden **sözleşme** (`{cat, resource, ttl, verbatim}`). Kod bunları `SCHEMAS` ve `TOOL_META` olarak ayrı tutar.

**Tanım (şema)** = `name + description + parameters`. Modelin tool'u ne zaman/nasıl çağıracağını belirler. Gerçek envanterin açıklamaları birebir korunur.

**Sözleşme (meta)** = trace-compaction'ın "bu tool nasıl davranmalı" bilgisi:

| Alan | Ne söyler | Örnek |
|---|---|---|
| `cat` | okuma mı / arama mı / yazma mı | `read`, `search`, `write` |
| `resource` | hangi varlığa dokunuyor (kimlik) | `key`, `document_id`, `page_id` |
| `ttl` | kaç adımda kendiliğinden bayatlar | fiyat/outline=1, issue=20, dizin=None |
| `verbatim` | çıktı birebir mi korunmalı | bütçe/sayı=True, liste=False |

**KANIT-3 — temsilci tool'ların sözleşmesi** (gerçek `TOOL_META` dökümü):
```
jira_get_issue        cat=read   res=key         ttl=20   verbatim=True
jira_aggregate        cat=search res=-           ttl=None verbatim=True
neta_get_project      cat=read   res=ref         ttl=None verbatim=True
docx_get_outline      cat=read   res=document_id ttl=1    verbatim=False
docx_add_chart        cat=write  res=document_id ttl=None verbatim=False
confluence_get_page   cat=read   res=page_id     ttl=25   verbatim=True
```

Bu tablo sistemin domain mantığını nasıl kavradığını gösterir: **Jira issue** gün içinde değişir (ttl=20); **NETA bütçesi** birebir korunur ama zamanla bayatlamaz (verbatim, ttl=None); **docx outline** her düzenlemede bayatlar (ttl=1); **Confluence sayfası** düzenlenebilir (ttl=25).

---

<a name="s2"></a>
## 2. LLM'e nasıl veriliyor

Tool, LLM'e **yalnızca tanım katmanıyla** verilir — `tools` alanında OpenAI şeması olarak. Sözleşme (cat/resource/ttl) modele **gitmez**; o sadece trace-compaction'ın iç bilgisidir.

**KANIT-1 — `jira_get_issue`'nun LLM'e giden gerçek tanımı:**
```json
{
  "type": "function",
  "function": {
    "name": "jira_get_issue",
    "description": "Key'i bilinen TEK bir issue'nun tam detayını okur: summary,
      description, status, priority, assignee, reporter, tarihler, attachment ve
      bağlantılar. include ile ek bloklar açılır: changelog, comments, worklogs,
      linked_issues, children, parent_chain, transitions",
    "parameters": {
      "type": "object",
      "properties": { "key": { "type": "string", ... } },
      "required": ["key"]
    }
  }
}
```

Model bu tanıma bakıp çağırma kararı verir. **Niyet/sonuç burada YOKTUR** — çünkü tool henüz çağrılmadı. Onlar (§5) çıktı sıkışırken doğar.

**Girdi ekseni notu (kapsam dışı ama kritik):** 119 tool'un 119 şemasını birden bağlama koymak ölçeklemez. Ürünün **tool-manager**'ı (pgvector `discover_tools`) tam bunu çözer: hiyerarşik retrieval ile soruya göre ~6 tool tanımı seçer. Bu ayrı bir eksendir (tanım sıkıştırma); bu ek **çıktı eksenini** (trace sıkıştırma) anlatır. İkisi art arda çalışır.

---

<a name="s3"></a>
## 3. Çıktılar nasıl

Tool çalışınca ham çıktı üretir. Mock'lar **deterministik** (aynı okuma → aynı çıktı, bu dedup için şart) ve **gerçekçi boyutlu** (compaction'a anlamlı yem).

**KANIT-2 — `jira_get_issue(key="ATLAS-101")` ham çıktısı:**
```
Issue ATLAS-101:
  summary: ATLAS-101 — modül davranışı düzeltmesi
  type: Task
  status: In Progress
  assignee: Zeynep Ak
  story_points: 3
  priority: Low
  · detay-1: (mock alan) değer=100
  ... (8 dolgu satırı — gerçekçi boyut)
```
Bu çıktı ~**130 token**. Trace'te ham haliyle birikir; bağlam dolunca sıkışacak olan da budur.

**Çıktı tipleri toolkit doğasına göre farklı:**
- **read** (jira_get_issue, neta_get_project, confluence_get_page) → çok-satırlı detay
- **search** (jira_resolve_project, confluence_search) → skorlu sonuç listesi
- **agg** (jira_aggregate, neta_count) → sayısal özet ("metric=count → 47")
- **write** (docx_add_chart, docx_create) → kısa onay + kimlik ("document_id=doc_x (v2)")
- **outline** (docx_get_outline) → **güncel duruma göre değişen** blok listesi (staleness kaynağı)

---

<a name="s4"></a>
## 4. Trace'te nasıl değerlendiriliyor

Çıktı üç deftere yazılır: **trace** (ham olay), **ledger** (kaynak/sürüm/kategori/ttl), **CWL episode** (ajan-bildirimli grup). Ledger her çağrıda kategoriye göre işler ve iki soruyu cevaplar: *bu okuma bayat mı? bu çağrı tekrar mı?*

**KANIT-4 — ledger değerlendirmesi (gerçek durum dökümü):**
```
ledger: 4 gözlem · 2 yazma · adım 6
  jira_get_issue(ATLAS-101) tekrar mı → is_stale(ilk)=False   dup: aynı key+sürüm
  docx_get_outline(v1) seq4 → add_chart YAZDI → is_stale=True  (mutasyon)
```

İki mekanizma somut:
- **Dedup:** İkinci `jira_get_issue(ATLAS-101)`, aynı `key` + aynı sürüm → önceki gözlemin tekrarı. İlk okuma bayat değil (`is_stale=False`), yani gerçek bir duplicate.
- **Staleness (mutasyon):** `docx_get_outline` v1 okundu; sonra `docx_add_chart` **aynı `document_id`'ye yazdı** → sürüm ilerledi → `is_stale(seq4)=True`. Outline artık bayat. Bu, `read→write→invalidate` çekirdeğinin ürün üzerinde canlı kanıtı.

Ayrıca ajan `delimiter` çağırdıysa CWL episode kurulur: bir `[expl] jira-veri` grubu + ona bağlı `[act] rapor`. Faz 6 bu bağımlılığı kullanır (act atılmadan expl atılamaz).

---

<a name="s5"></a>
## 5. Compact edilmiş hal

Bütçe aşılınca compaction çalışır ve her tool'un kaderini belirler: **TAM / ÖZET / SİL**. Kritik olan — bu kader `messages[]`'e **gerçekten yazılır**, `tool_call_id` korunur (API 400 yok). Karar da içerik de **sıfır LLM**.

**KANIT-5 — iki tool'un compact hali (gerçek üretim):**

*Verbatim tekrar* (`jira_get_issue` duplicate) → ÖZET engellenir (verbatim özet=ham), **SİL**'e düşer:
```
ham: 130 token → SİL stub: "[silindi] tekrar ≡ seq=0 (aynı içerik canlı)"
```

*Bayat outline* (`docx_get_outline`, verbatim=False) → **5-alan ÖZET**:
```
[özet] niyet: docx_get_outline çağrıldı · girdi: document_id=doc_fe96de ·
       sonuç: Outline doc_fe96de (v1, 1 blok):… · durum: ok · etki: bayat (eskidi)
ham: 141 token → özet: 40 token
```

**KANIT-6 — köprü: `messages[]` render öncesi vs sonrası** (modelin gerçekten gördüğü):
```
tool_call_id=call_002  (AYNI korunur → API 400 yok)
  ÖNCE (ham, 437 karakter): Issue ATLAS-101: ...
  SONRA (compact):          [özet] niyet: [jira-veri] keşif episode'u · girdi:
                            jira_get_issue · sonuç: x · durum: ok · ...
ham messages: 683 tok → render: 602 tok
```

Dikkat: `tool_call_id` **değişmedi**, sadece `content` küçüldü. Ham çıktı `messages[]`'te kaynak olarak durur; modele giden **render edilmiş** kopyada içerik sıkışık. Ölçtüğümüz kazanç modelin gerçekten gördüğü bağlama yansır.

**Compact hallerinin özeti (tool tipine göre):**

| Tool tipi | Bayat/tekrar olunca | Compact hali |
|---|---|---|
| verbatim read tekrarı (jira_get_issue, neta_get_project) | aynı içerik canlı | **SİL** stub (~10 tok) |
| non-verbatim read bayat (docx_get_outline) | mutasyon/ttl | **ÖZET** 5-alan (~40 tok) |
| keşif dizisi (resolve→get→get) | ardışık okuma | tek **bulgu**ya katlanır (playbook'a yazılır) |
| CWL expl grubu | bağlı act atıldıysa | episode **description**'ına iner (tek cümle) |
| verbatim veri (aktif, gerekli) | — | **TAM** korunur (model'e lazım) |
| son N olay | — | **TAM** (koruma penceresi) |

---

<a name="s6"></a>
## 6. Uçtan uca: bir soru, tam döngü

`run_product.py` — Jira keşfi → NETA bütçe → Word raporu senaryosu (LLM'siz, gerçek akış):

**KANIT-7 — gerçek kader şeridi + sonuç:**
```
#1-5  jira keşfi   → ÖZET  [expl] jira-veri   (keşif dizisi [1..5] bulguya katlandı)
#7    jira tekrar  → SİLİNDİ (≡ seq=3, aynı içerik canlı)
#12   eski outline → ÖZET  (write sonrası bayat)
#8,10,13-15  verbatim veri + rapor → TAM (model'e lazım, doğru korunuyor)

COMPACTION LOG:
  seq=7  jira_get_issue → SİLİNDİ · tekrar ≡ seq=3 (aynı içerik canlı)
  seq=12 docx_get_outline → ÖZET · bayat (eskidi)
  keşif dizisi [1..5] → bulguya katlandı (5 adım)

MODELE GİDEN BAĞLAM: ham 1883 token → sıkışık 1630 token → KAZANÇ %13
episode: [expl] jira-veri (5 olay) · [act] rapor ←['jira-veri'] (5 olay)
```

**%13 neden "az" ve neden doğru:** Bu kısa senaryoda içeriğin çoğu **verbatim kritik veri** (issue detayı, bütçe rakamları) ve sistem onları bilinçle **tam tutuyor** — model bunlara ihtiyaç duyar. Sistem "her şeyi ezen" değil, "gereksizi (tekrar/bayat) atıp gerekeni koruyan". Tekrarlı uzun oturumlarda (equity senaryosu: 1851→611) **%67**'ye çıkar. Dürüst sıkıştırma, agresif sıkıştırmadan iyidir: yanlış veri atmak ajanı bozar.

---

<a name="s7"></a>
## 7. Kanıt özeti ve testler

| Soru | Kanıt | Sonuç |
|---|---|---|
| Hepsi kapsandı mı | kapsam denetimi | **119/119**, eksik yok, jenerik-fallback yok |
| LLM'e tanım gidiyor mu | KANIT-1 | gerçek OpenAI şeması (name+desc+params) |
| Çıktılar gerçekçi mi | KANIT-2 | deterministik, ~130 tok, dolgu ile |
| Trace değerlendiriyor mu | KANIT-4 | dedup + staleness (mutasyon) canlı |
| Compact messages'a yansıyor mu | KANIT-6 | ham 683 → render 602, tool_call_id korunur |
| Regresyon | `test_deterministic.py` | **26/26** |
| Köprü uçtan uca | `test_product.py` | **6/6** (render<ham · eşleşme tam · dedup · staleness · CWL) |
| Equity korundu mu | `run_equity.py` | **%67** |

**Deterministik test (`test_product.py`) neyi kanıtlıyor:**
1. Modele giden bağlam ham'dan küçük (compaction messages'a yansıdı)
2. Tüm `tool_call_id`'ler eşleşiyor (API 400 riski yok)
3. Tool mesajları sıkışık forma indi ([özet]/[silindi])
4. Dedup çalıştı (verbatim tekrar → SİL)
5. Staleness çalıştı (write sonrası eski outline bayat)
6. CWL episode kuruldu (expl jira-veri + act rapor)

---

## Özet — beş sorunun kanıtlı cevabı

1. **Ne barındırıyoruz:** her tool'da iki katman — LLM'e giden **tanım** (şema) + trace-compaction'a giden **sözleşme** (`cat/resource/ttl/verbatim`). 119 tool tam kapsandı.
2. **LLM'e nasıl:** yalnızca tanım (`tools` şeması); sözleşme modele gitmez. 119 şema retrieval ile daraltılır (ayrı eksen).
3. **Çıktılar:** deterministik, gerçekçi boyutlu; tipe göre detay/liste/sayı/onay.
4. **Trace'te değerlendirme:** ledger kategoriye göre işler — dedup (aynı kaynak+sürüm), staleness (mutasyon + ttl); CWL episode ajan-bildirimli grup.
5. **Compact hali:** TAM (gerekli veri) / ÖZET (5-alan, ~40 tok) / SİL (stub, ~10 tok) — hepsi `messages[]`'e yazılır, `tool_call_id` korunur, sıfır LLM.

*Bu ek `poc-trace-compaction/` (product_tools.py, agent.py, compactor.py, ledger.py) Ağustos 2026 durumundan üretilen gerçek çıktılara dayanır.*
