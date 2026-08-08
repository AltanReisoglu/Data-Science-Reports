# 13-Ek — İç Mekanizma: Ledger, CWL ve Fazlar (Baştan Sona)

**Ağustos 2026 · §13'ün derinleştirme eki · POC'ye dayalı**

Bu ek, §13'te tarif edilen trace-compaction sisteminin **çalışan POC'sini** (`poc-trace-compaction/`) satır satır açar. §13 "ne yapılmalı"yı anlatır; bu ek "kod tam olarak nasıl yapıyor"u anlatır. Üç defter (trace, ledger, CWL episode grafiği), bunların ürettiği sinyaller, 7 fazlı eviction hattı ve tüm güvenlik frenleri — hepsi gerçek fonksiyon/satır referanslarıyla.

> **Altın kural:** Sıkıştırmanın **kararı da içeriği de sıfır LLM**. Her karar tamsayı karşılaştırması; her özet string şablonu. LLM yalnızca asıl işi yapan ajan döngüsünde kullanılır (§5 complexity trap: deterministik kırpma, LLM özetinin ~yarı maliyetiyle benzer kalite).

**İçindekiler**
1. [Büyük resim: üç defter, bir hat](#s1)
2. [Trace: ham olay dizisi](#s2)
3. [Ledger: deterministik yürütme defteri](#s3)
4. [ObservationRecord: bir okumanın makbuzu](#s4)
5. [Bayatlık ve dedup: iki kapı](#s5)
6. [CWL episode grafiği: ajanın iş defteri](#s6)
7. [Bağımlılık-farkında eviction](#s7)
8. [5-alan özet: LLM'siz sıkıştırma](#s8)
9. [7 faz: kademeli eviction hattı](#s9)
10. [Güvenlik frenleri](#s10)
11. [Üç kader: silme yok](#s11)
12. [Uçtan uca örnek: sayılarla](#s12)
13. [Ürün tool'larına uyarlama: sözleşme](#s13)
14. [Sınırlar ve açık uçlar](#s14)

---

<a name="s1"></a>
## 1. Büyük resim: üç defter, bir hat

Bir tool çağrıldığında sonucu **üç deftere** birden yazılır:

```
tool çalıştı (dispatch)
   │
   ├─ trace.add_tool(...)   → ham olay #N (metin gövdesi)
   ├─ ledger.record(...)    → kaynak/sürüm/kategori/TTL (deterministik durum)
   └─ episodes.attach(#N)   → aktif CWL episode'a bağla (ajan-bildirimli yapı)
        │
        └─ compactor.compact()  → bütçe aşıldıysa 7 faz devreye girer
```

Her defterin işi farklı:

| Defter | Kim doldurur | Neyi bilir | Cevapladığı soru |
|---|---|---|---|
| **trace** | sistem | ham olay dizisi | "ne oldu, hangi sırada" |
| **ledger** | sistem (otomatik) | kaynak + sürüm + zaman | "bu okuma bayat mı / tekrar mı?" |
| **CWL** | ajan (delimiter ile) | grup + bağımlılık | "bu keşif grubu güvenle atılabilir mi?" |

Ledger **atomik** düşünür (tek tool). CWL **grup + bağımlılık** düşünür. Fazlar bu iki defterin sinyallerini kademeli olarak kullanır: önce ucuz/güvenli (ledger dedup/stale), sonra pahalı/yapısal (CWL episode).

**Kritik zamanlama:** Sıkıştırma turun sonunda değil, **her tool çağrısından hemen sonra** yoklanır — döngü `for tc in msg.tool_calls` içinde her tool bittiğinde `_record_and_maybe_compact` çağrılır (`agent.py:207-241`). Bir tur 5 tool çağırırsa sıkıştırma o tur içinde 5 kez yoklanır (ama sadece bütçe aşılınca gerçekten çalışır).

---

<a name="s2"></a>
## 2. Trace: ham olay dizisi

Trace, olayların değiştirilemez günlüğü (`trace.py`). Üç olay tipi:

- `reasoning` — modelin düşünce metni (niyet buradan geri kazanılır)
- `tool` — bir tool çağrısı: `{name, args, output}` + `status` + `verbatim` bayrağı
- `answer` — modelin nihai yanıtı

Her olay bir `seq` (artan tamsayı) taşır. Tool olayının kritik alanı `intent_ref`: onu tetikleyen reasoning olayının seq'i. Niyet ham trace'te **ayrı alan olarak yok** — bu bağ üzerinden geri kazanılır.

Bir olayın üç kader alanı var (başta hepsi kapalı):
- `evicted` (bool) + `summary` (TraceSummary) → **ÖZET** oldu
- `cleared` (bool) + `clear_note` (str) → **SİL** oldu
- ikisi de yoksa → **TAM**

Önemli: bu bayraklar olayı **yerinde işaretler**, listeden çıkarmaz. Nedeni §11'de.

---

<a name="s3"></a>
## 3. Ledger: deterministik yürütme defteri

`ledger.py` — Ledger (arXiv 2608.00808) uyarlaması. LLM çağırmadan, üç kayıt tipi + iki sayaç + bir saat ile trace'in yürütme durumunu izler.

### Sayaçlar ve saat (`ExecutionLedger.__init__`, `ledger.py:68-74`)

```python
self.observations   = []    # okuma makbuzları
self.commands       = []    # her tool'un imza + kategori kaydı
self.local_counters = {}    # kaynak → sürüm sayacı
self.global_counter = 0     # toplam yazma
self.step           = 0     # mantıksal saat (her record'da +1) — TTL için
```

### Sözleşme: domain-bağımsızlık (`ledger.py:76-94`)

Ledger tool'un **ne olduğunu** bilmez; her tool bir sözleşme bildirir. `tool_meta` verilmezse dosya varsayılanı, verilirse domain-bağımsız:

```python
def _category(self, name):   # "read|search|write|test|other"
    return self.tool_meta[name].get("cat", "other") if meta else _CATEGORY.get(name, "other")

def _resource(self, name, args):   # kaynak anahtarı: path / ticker / document_id / org_unit
    fn = self.tool_meta[name].get("resource")
    return str(fn(args)) if fn else str(args.get("path", ""))

def _ttl(self, name):        # volatil kaynak: kaç adım sonra bayat (fiyat=1, haber=6, None=hiç)
    return self.tool_meta[name].get("ttl") if meta else None
```

Bu sayede **aynı ledger** hem `config.py` dosyasına, hem `XOM` ticker'ına, hem `PROJ-123` Jira issue'suna çalışır — her tool `{cat, resource, ttl}` üçlüsünü kendisi bildirir.

### Kayıt akışı — kategoriye göre yönlendirme (`ledger.py:98-142`)

`record()` bir yönlendirici. Her tool çağrısında `step += 1`, sonra kategoriye göre:

| Kategori | Ne yapar | ObservationRecord? | Sürüm sayacı |
|---|---|---|---|
| `write` | kaynağın sürümünü **artırır**, eski gözlemleri geçersizler | ❌ | ++ (başkalarını bayatlatır) |
| `read` | gözlem makbuzu keser, önce dedup kontrolü | ✅ | okur (etkilemez) |
| `search` | imza tekrarıyla dedup | ❌ | etkilemez |
| `test`/`other` | sadece iz | ❌ | etkilemez |

**Herkes en az bir `CommandRecord` bırakır** (`ledger.py:140-141`) — hiçbir tool izsiz geçmez. Fark, ek olarak ne bıraktığında.

---

<a name="s4"></a>
## 4. ObservationRecord: bir okumanın makbuzu

`read` kategorili her tool bir **makbuz** keser (`ledger.py:21-30`). "Şu kaynağı, şu anda, şu sürümde okudum." 7 alan:

| Alan | Ne söyler | Neden var |
|---|---|---|
| `path` | hangi kaynak (`XOM`, `config.py`, `PROJ-123`) | aynı varlığın okumalarını bağlamak |
| `content_hash` | dönen içeriğin hash'i | içerik gerçekten değişti mi |
| `local_counter` | okuma anında kaynağın **kaçıncı sürümü** | mutasyon tespiti |
| `global_counter` | o an global yazma sayacı | genel sıralama |
| `event_seq` | bu gözlemi doğuran **trace olayı** | makbuz ↔ ham olay köprüsü |
| `step` | okuma anındaki **mantıksal saat** | TTL/zaman-eskimesi |
| `ttl` | kaç adımda **kendiliğinden bayatlar** | uçucu kaynaklar (fiyat/haber) |

Makbuz **değiştirilemez** — okuma anındaki sürümü ve saati dondurur, bir fotoğraf gibi. Dünya sonradan değişse de makbuz "ben sürüm 0'ı, adım 2'de gördüm" der. `event_seq` kritik: compactor "#2 bayat mı?" derken bu makbuzu bulup ona bakar.

**Neden sadece read'de?** Çünkü bayatlayabilen tek şey bir okumanın döndürdüğü **durum**dur. Write durumu *değiştirir* (eski sürümü kavramı yok), search *sorguya bağlı* (kaynağı yok), test *nötr*. Bir grafiği "bayat" saymak anlamsız; bir fiyat okumasını saymak zorunlu.

---

<a name="s5"></a>
## 5. Bayatlık ve dedup: iki kapı

### Dedup — okuma anında (`ledger.py:124-127`)

Makbuz kesilmeden önce ledger sorar: *"Aynı `path` için **aynı `local_counter`** bir gözlem zaten var mı?"* Varsa → `duplicate_of = eski_seq`. "İki çıktı stringi eşit mi?" değil — **"aynı kaynağın aynı sürümü mü?"** Sürüm eşitse zorunlu olarak aynı içerik.

Search için dedup imza üzerinden (`ledger.py:133-138`): aynı normalize `tool(arg=val)` imzası tekrar gelirse duplicate.

### Bayatlık — `is_stale`'in iki kapısı (`ledger.py:146-159`)

```python
def is_stale(self, event_seq):
    obs = event_seq'e karşılık gelen makbuz
    current = self.local_counters.get(obs.path, 0)
    if obs.local_counter < current:                    # KAPI 1: MUTASYON
        return True
    if obs.ttl is not None and (self.step - obs.step) > obs.ttl:   # KAPI 2: VOLATİLİTE
        return True
    return False
```

**Kapı 1 — Mutasyon:** Makbuz sürüm 0'ı gördü, ama sonradan biri kaynağa yazdı, sayaç 1 oldu. `0 < 1` → bayat. Write geldiğinde (`ledger.py:111-117`) aynı `path`'in sayacını artırır — makbuza dokunmadan. Bu boşluk = eskime (MVCC/cache-invalidation mantığı).

**Kapı 2 — Volatilite:** Kimse yazmasa bile mantıksal saat her çağrıda ilerler. Fiyat (`ttl=1`) adım 2'de okundu, şimdi adım 10 → `10-2=8 > 1` → bayat. Zaman geçtiği için. Gelir tablosu (`ttl=None`) böyle bayatlamaz — zamanla değişmez.

### read → write → read: sıra önemli

```
#1 read(config.py)   → makbuz: sürüm 0
#2 write(config.py)  → sürüm 0→1              (#1 artık bayat: 0<1)
#3 read(config.py)   → makbuz: sürüm 1        (write'tan SONRA → taze)
```

Sadece write'tan **önceki** okumalar bayatlar; sonrakiler yeni sürümü aldığı için taze. Makbuz okuma anındaki sürümü dondurduğu için bu ayrım otomatik. Ayrıca **farklı kaynağa** yazma (`server.py`) `config.py` okumasını bozmaz — `obs.path == path` şartı.

---

<a name="s6"></a>
## 6. CWL episode grafiği: ajanın iş defteri

`episode_graph.py` — Beyond Compaction (arXiv 2606.11213). Ledger otomatik ve atomik; CWL ise **ajanın kendi eliyle** doldurduğu, grup + bağımlılık bilen defter. Ajan diyor ki: *"Şu 8 tool tek bir 'veri-toplama' keşfiydi; sonraki 'rapor' eylemi buna dayanıyor."* Ledger bunu bilemez — niyet ajanın kafasında.

### Veri yapısı: Episode (`episode_graph.py:23-35`)

| Alan | Ne tutar |
|---|---|
| `name` | episode adı (`veri-toplama`, `rapor`) |
| `type` | `expl` (keşif) veya `act` (eylem) |
| `dependencies` | act → hangi expl'lere dayanıyor (`rapor ← [veri-toplama]`) |
| `event_seqs` | bu episode'a düşen tool olayları |
| `description` | kapanışta ajanın yazdığı özet — **eviction sonrası kalan tek şey** |
| `end_seq` | kapandı mı (None = aktif) |

### delimiter protokolü — üç primitif

Ajan `delimiter` tool'uyla yazar:

**1) `start(name, type, seq, deps)`** (`episode_graph.py:47-58`) — yeni episode açar, `_active` yapar. `act` ise bağımlılık bildirmeli.

**2) `attach(seq)`** (`episode_graph.py:70-73`) — her tool çağrısından sonra agent otomatik çağırır (`agent.py:149`). Aktif episode varsa olayın seq'i `event_seqs`'e eklenir.

**3) `end(seq, description)`** (`episode_graph.py:60-68`) — aktif episode'u kapatır. **expl'de description zorunlu** — tüm olaylar atıldığında geriye sadece bu cümle kalacak.

```
ajan: delimiter(start, expl, "veri-toplama")
  #0 get_stock_price   → attach → event_seqs=[0]
  #3 get_company_info  → attach → [0,3]
  ... 8 olay ...
ajan: delimiter(end, "XOM finansalları toplandı")   → description sabitlendi
```

### Gemma tuhaflığı: text-format delimiter (`agent.py:119-138`)

Bazı modeller (Gemma) tool çağrısını düzgün `tool_calls` yerine düz metin (`<|tool_call|>...`) olarak yayar. `_strip_tool_markup` bunu yanıttan temizler; `_maybe_apply_text_delimiter` yine de parse edip episode grafiğine uygular — böylece CWL bu formatta da oluşur, yanıta da sızmaz.

---

<a name="s7"></a>
## 7. Bağımlılık-farkında eviction

CWL'nin asıl gücü (`episode_graph.py:83-98`). Kritik kural (§5.2): **bir expl episode ANCAK ona bağlı TÜM act'ler zaten atıldıysa atılabilir.**

```python
def evictable_expl(self, evicted_seqs):
    for ep in episodes where type=="expl" and kapalı:
        dependents = [act'ler where ep.name in act.dependencies]
        all_evicted = tüm dependent act'lerin tüm olayları zaten evicted mı?
        if all_evicted: result.append(ep)
```

**Neden?** Keşif, eylemin **gerekçesi**. "Rapor" hâlâ bağlamda canlıyken onu besleyen "veri-toplama"yı atarsan model raporun neye dayandığını kaybeder — bağlam çöker. O yüzden önce eylem (act), sonra keşif (expl), ve ancak dayanağı kalmayınca.

---

<a name="s8"></a>
## 8. 5-alan özet: LLM'siz sıkıştırma

Bir olay ÖZET'e indiğinde `_summarize_deterministic` (`compactor.py:68-97`) çalışır — LLM'siz, 5 sabit alan:

| Alan | Nereden | Olmasa ne kaybolur |
|---|---|---|
| `niyet` | reasoning'den (`_intent_of`, intent_ref) | model *neden* çağırdığını unutur |
| `girdi` | tool argümanları | *hangi* kaynağa baktığı |
| `sonuc` | çıktının ilk 1-2 satırı (kırpık) veya verbatim | asıl bilgi |
| `durum` | `ev.status` (ok/HATA) | başarılı mıydı |
| `etki` | sıkıştırma sebebi ("tekrar ≡ seq=7") | izin nereye gittiği |

### Göreve-koşullu kırpma (K5, `compactor.py:80-89`)

- `verbatim=True` → `sonuc = output` **birebir** (gelir, oran, bütçe, sayımlar)
- `verbatim=False` → çıktı kullanıcının sorusuyla örtüşüyorsa 2 satır/120 karakter, örtüşmüyorsa 1 satır/60 karakter

"Özetleme" dediğimiz şey aslında **çıkarım + kırpma**: argümanları al, çıktının ilk satırlarını kes, ledger'ın verdiği sebebi ekle. Hiçbir yerde model yok. (Opsiyonel `summarize_fn` ile LLM özeti enjekte edilebilir ama POC'de **kapalı** — complexity trap.)

---

<a name="s9"></a>
## 9. 7 faz: kademeli eviction hattı

`compact()` (`compactor.py:145+`) bütçe aşılınca çalışır. Kademeli: en ucuz/güvenli önce, pahalı sona. Her faz `target`'a inince durabilir (histerezis).

| Faz | Ne yapar | Hangi defter | Kod |
|---|---|---|---|
| **Ön-koruma** | son N olay + çözülmemiş hatalar korunur | trace | `:155-167` |
| **Kapı** | `before <= budget` ise hiç çalışma | — | `:169-173` |
| **Faz 1** | DUPLICATE'ler → ÖZET (en güvenli) | ledger | `:175-182` |
| **Faz 2** | STALE gözlemler: canlı kopya varsa SİL, yoksa ÖZET | ledger | `:184-199` |
| **Faz 3** | HATA-ZİNCİRİ katlama (hata mesajı verbatim korunur) | trace | `:201-223` |
| **Faz 4** | ardışık KEŞİF dizisi → tek bulgu (playbook'a yazılır) | ledger kategori | `:225-260` |
| **Faz 5** | kademeli: önce ACT, sonra EXPL evict | ledger kategori | `:262-290` |
| **Faz 6** | CWL EPISODE eviction (bağımlılık-farkında) | CWL | `:292-320` |
| **Faz 7** | ACİL: en büyük token önce (kategori/konum değil, boyut) | trace | `:322-336` |

Fazlar 4-7 yalnızca hâlâ `target`'ın üstündeysek devreye girer (`> self.target` kontrolü). Faz 1-2 her zaman çalışır (en güvenli, bedava kazanç).

### Faz 2'nin B.11 vs B.12 ayrımı (`compactor.py:189-199`)

```python
if ledger.is_stale(ev.seq):
    fresh = _has_fresher_live_read(trace, ledger, ev)   # aynı tool + aynı kaynak, daha taze, canlı?
    if fresh: _clear_event(ev, ...)     # B.11 context editing → SİL (sıfıra yakın)
    else:     _evict_event(ev, ...)     # B.12 compaction → ÖZET
```

Bayat birimin güncel kopyası bağlamda canlıysa → **SİL** (bilgi başka yerde duruyor, güvenli). Değilse → **ÖZET** (yine de iz bırak).

### Faz 6'nın description'a inişi (`compactor.py:300-320`)

Atılabilir her expl episode'un canlı olayları, ajanın kapanışta yazdığı **tek cümleye** iner. 8 tool → `"XOM finansalları toplandı"`. Son olay o cümleyi taşır, diğerleri `(episode'a katlandı)`.

---

<a name="s10"></a>
## 10. Güvenlik frenleri

### İki-eşik histerezis (Wegent'ten, `compactor.py:104-108`)

- `budget` = **TETİKLE** (bu aşılınca başla)
- `target` = `budget × 0.6` = **BURAYA KADAR İN** (belirgin altı)

Bir kez sıkıştırıp target'a inince, bir sonraki tool çağrısı hemen tekrar tetiklemez — "sawtooth"/testere önlenir.

### Koruma penceresi (`compactor.py:155-159`)

Son N tool birimi (`protect_window`) asla sıkıştırılmaz. Bu turun sonucu her zaman TAM kalır → model en güncel veriyi tam görür. (N=0'da `liste[-0:]` tüm listeyi döndürür — açıkça boş küme kullanılır.)

### Çözülmemiş hata koruması (NexAU'dan, `compactor.py:160-167`, `_unresolved_errors`)

Düzeltilmemiş bir hata (`status=error` olup sonrasında aynı tool'un başarılısı YOK) konumu ne olursa olsun korunur — ajan hâlâ çözmeye çalışıyor. Konumsal pencere bu "in-flight" durumu kaçırırdı.

### Fayda güvencesi (complexity trap, `compactor.py:116-129`, `131-143`)

```python
if summary.token_cost() >= _raw_cost(ev):
    return False   # özet ham'dan küçük değilse sıkıştırma ZARARLI → ham bırak
```

Küçük tool çıktılarında 5-alan özet ham'dan büyük olabilir. Bu durumda evict geri alınır. (Bu güvence eklenmeden önce POC negatif %-6 sıkıştırma üretiyordu; eklendikten sonra %+67.)

---

<a name="s11"></a>
## 11. Üç kader: silme yok

Bir olay üç kaderden birine düşer, **ama üçü de olayı yerinde tutar**:

| Kader | Bayrak | messages[]'te içerik |
|---|---|---|
| **TAM** | — | ham çıktı |
| **ÖZET** | `evicted=True` + `summary` | 5-alan kart (~30 token) |
| **SİL** | `cleared=True` + `clear_note` | tek satır stub |

**Neden listeden çıkarmıyoruz?** LLM'e giden `messages[]`'te her `tool_call` id'sinin karşısında bir `tool` mesajı olmalı. Ham sonucu **çıkarırsan** eşleşme kırılır → API **400**. Bu yüzden olay hep yerinde kalır; sadece `content` küçülür. Trace katmanında kader belirlenir, `_render_messages` bu kaderi LLM'in gerçekten gördüğü diziye yansıtır.

> **Köprü (tamamlandı):** `_render_messages` artık her tool mesajının `content`'ini kader'e göre yeniden yazar — `self._call_seq` ile `tool_call_id → ev.seq` eşlemesi kurulur, `cleared` olay tek satır stub'a, `evicted` olay `TraceSummary.render()` 5-alan kartına iner, `tool_call_id` aynı kalır (eşleşme bozulmaz). Böylece ölçülen sıkışma modelin GERÇEKTEN gördüğü `messages[]`'e yansır. `agent.rendered_token_cost()` bunu ölçer; `test_product.py` render < ham VE tüm id'lerin eşleştiğini deterministik doğrular.

---

<a name="s12"></a>
## 12. Uçtan uca örnek: sayılarla

`run_equity.py` çıktısı (XOM kapsamlı rapor senaryosu):

```
Ham trace: 9 tool birimi, 1851 token
Ledger tespitleri:
  seq 1  get_stock_price          → BAYAT (TTL/volatilite, ttl=1)
  seq13  get_key_financial_ratios → DUP ≡ seq7

SIKIŞTIRMA:
  seq=13 → ÖZET · tekrar (≡ seq=7)
  keşif dizisi [1..11] → bulguya katlandı (6 adım)

Öncesi 1851 token → Sonrası 611 token → Kazanç %67.0
KORUNDU  : [15, 17]        (koruma penceresi)
ÖZETLENDİ: [1,3,5,7,9,11,13]
SİLİNDİ  : []
```

Akış:
1. **Faz 1** seq13'ü seq7'nin tekrarı bulur (aynı ticker, aynı sürüm) → ÖZET
2. **Faz 4** ardışık keşif dizisini tek "bulgu"ya katlar → playbook'a yazılır (evict'ten korunur)
3. **Fayda güvencesi** küçük olayları ham bırakır
4. **Koruma penceresi** son 2 olayı (15,17) TAM tutar
5. Sonuç: 1851 → 611, bağlamın üçte ikisi arındı, öğrenilen bulgu playbook'ta yaşıyor

---

<a name="s13"></a>
## 13. Ürün tool'larına uyarlama: sözleşme

Sistem motoruna **hiç dokunmadan** yeni bir domaine bağlanır — tek iş her tool'a `TOOL_META = {cat, resource, ttl, verbatim}` yazmak.

```python
# okuma + kimlik + zaman-eskimesi
"jira_get_issue":    {"cat":"read", "resource": lambda a: a["issue_key"],   "ttl": 20, "verbatim": True},
"neta_get_project":  {"cat":"read", "resource": lambda a: a["project_key"], "ttl": None, "verbatim": True},
"ldap_org_members":  {"cat":"read", "resource": lambda a: a["org_unit"],    "ttl": None, "verbatim": True},
# sorguya bağlı
"jira_search_issues":{"cat":"search"},
"confluence_search": {"cat":"search"},
# stateful yazma (doküman inşası)
"docx_edit_block":   {"cat":"write", "resource": lambda a: a["document_id"]},
"docx_get_outline":  {"cat":"read",  "resource": lambda a: a["document_id"], "ttl": 1},   # her düzenlemede bayat!
```

TTL mantığı domaine göre: LDAP/NETA dizin verisi neredeyse hiç değişmez (`None`); Jira issue gün içinde değişir (orta ttl); `docx_get_outline` her düzenlemede bayatlar (ttl=1 + write mutasyonu). Verbatim: sayı/bütçe/detay taşıyan okumalar birebir korunur.

Bir doküman inşası tam CWL örneği: `[expl] jira-veri → [act] docx-rapor (deps=[jira-veri])`. Rapor bitince (act evict) jira keşfi tek description'a iner. Resolver→read→aggregate→build zinciri birebir expl→act haritasına oturur.

---

<a name="s14"></a>
## 14. Sınırlar ve açık uçlar

1. **messages[] geri-yazımı (TAMAMLANDI):** `_render_messages` artık kader'i messages'a uyguluyor (id↔seq köprüsü + content overwrite). Sıkışık çıktı modele gerçekten gidiyor; `test_product.py` doğruluyor. Kalan iş: gerçek endpoint'te canlı token/para ölçümü.
2. **Girdi tarafı ayrı eksen:** Bu sistem tool **çıktılarını** sıkıştırır. 119 tool'un **tanımlarını** bağlama sığdırmak ayrı iş — progressive disclosure / `discover_tools` (ürünün tool-manager'ı zaten çözüyor). İki eksen art arda çalışır: retrieval tanımları, trace-compaction sonuçları küçültür. `product_tools.py` (119 tool) çıktı eksenini tam kapsar; girdi ekseni gelecek iş.
3. **CWL ajana bağımlı:** Episode grafiği ajanın `delimiter` çağrılarına dayanır. Ajan bildirmezse Faz 6 devreye girmez — sistem yine de Faz 1-5 + 7 ile çalışır (graceful degradation), ama yapısal kazanç kaçar.
4. **content_hash kullanımı zayıf:** Makbuzda hash tutuluyor ama şu an bayatlık kararı sürüm/TTL üzerinden; içerik-eşitliğiyle "değişti ama aynı" tespiti ileri iş.

---

*Bu ek `poc-trace-compaction/` kodunun Ağustos 2026 durumuna dayanır. Fonksiyon/satır referansları o commit'e göredir; kod değişirse referanslar kayabilir.*
