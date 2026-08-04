# Tool Trace Compaction — Uçtan Uca Akış

**POC + teori (§13) tek belgede.** Her birimin ne olduğu, nasıl çalıştığı ve
5N1K'sı (Ne · Neden · Nasıl · Nerede · Ne zaman · Kim). Diyagramlar ASCII.

> Teorinin tamamı: `../report/13-tool-trace-compaction.md`. Bu belge onun
> çalışan koddaki karşılığını baştan sona izler.

---

## 0. Büyük resim — teori ↔ kod haritası

```
TEORİ (§13)                          KOD (poc-trace-compaction/)
─────────────────                    ──────────────────────────
§4  5-alan özet şeması        ────►  trace.py     (TraceSummary)
§7  Format 3 tiplenmiş olay   ────►  trace.py     (Event)
§6.5.1 Ledger change counter  ────►  ledger.py    (ExecutionLedger)
§5.2 CWL delimiter + episode  ────►  episode_graph.py
§5  6 sıkıştırma yöntemi       ────►  compactor.py (6 faz)
§6  tetikleme + eskime         ────►  compactor.py + agent.py
§6.7 uçtan uca döngü          ────►  agent.py     (TracingAgent.run)
ek-a §3/§13 tool mentalitesi  ────►  tools.py
```

### Sistemin 5N1K'sı (genel)

| | |
|---|---|
| **Ne** | Bir ajanın tool etkileşim geçmişini (trace) küçülten bir katman |
| **Neden** | Uzun görevde trace bağlam penceresini doldurur; gürültü sinyali gömer |
| **Nasıl** | Deterministik tespit (tekrar/bayat) + kademeli evict + 5-alan özet |
| **Nerede** | Ajan döngüsünde model çağrısı ile tool yürütmesi arasındaki boşlukta |
| **Ne zaman** | Her tool sonrası ledger; faz sınırı/eşikte compaction |
| **Kim** | Ledger/compactor deterministik yapar; LLM yalnızca ajan + opsiyonel özet |

---

## 1. Uçtan uca akış — bir turun tamamı

```
KULLANICI: "config.py'deki portu 9090 yap, doğrula"
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  agent.py — AJAN DÖNGÜSÜ (her tur tekrar)                     │
└─────────────────────────────────────────────────────────────┘
     │
     ▼ 1. Gemma'ya sor (tool şemaları + geçmiş)
     │    ← reasoning "portu bulmam lazım" + tool_use(grep, PORT)
     │
     ▼ 2. ═══ TRACE KATMANI (boşluk) ═══
     │    ├─ trace.py    olayı kaydet (reasoning + tool, Format 3)
     │    ├─ ledger.py   observation/command kaydet + change counter
     │    ├─ episode.py  aktif episode'a iliştir (CWL)
     │    └─ compactor.py bütçe aşıldıysa sıkıştır (6 faz)
     │
     ▼ 3. tools.py       grep'i sample_repo'da ÇALIŞTIR → "config.py:6: PORT=8080"
     │
     ▼ 4. tool_result'ı messages'e ekle → 1'e dön
     │
     ▼ (model tool istemeyince) → NİHAİ YANIT
```

Kritik: **2. adım** (trace katmanı) ile **3. adım** (tool yürütme) arasındaki
sıra — ek-a'nın "harness = boşluk" mentalitesi. Compaction tam buraya girer.

---

## 2. Her birim — ne, nasıl, 5N1K

### 2.1 `trace.py` — olayların defteri

**Ne:** Yörüngedeki her adımı tiplenmiş olay (Event) olarak tutar; evict
edilen birimlerin 5-alan özetini (TraceSummary) taşır.

```
Event(seq, type, payload, intent_ref, verbatim, evicted, summary)
  type: "reasoning" | "tool" | "answer"
  intent_ref: bu tool'u tetikleyen reasoning'in seq'i   ← niyet bağı
```

| 5N1K | Cevap |
|---|---|
| Ne | Trace'in veri yapısı — tiplenmiş olay dizisi |
| Neden | Format 3 (tiplenmiş) niyeti yakalar + tip-duyarlı sıkışmayı açar (§7) |
| Nasıl | `add_reasoning/add_tool/add_answer` ile seq artırarak ekler |
| Nerede | Her modül buna yazar/okur; merkezî durum |
| Ne zaman | Her model çıktısı + her tool sonucunda |
| Kim | agent.py doldurur, compactor.py evict işaretler |

**Nasıl çalışır:** her olayın `token_cost()`'u var; evict edilirse ham payload
yerine `summary` gösterilir (`render()`). Silme yok — sadece gösterim değişir.

### 2.2 `ledger.py` — deterministik yürütme durumu

**Ne:** Trace'in yanında tutulan defter (Ledger, 2608.00808). Trace "ne oldu"yu,
ledger "şu an ne durumda"yı bilir.

```
observations: [ObservationRecord(path, local_counter, event_seq)]
commands:     [CommandRecord(signature, category, event_seq)]
local_counters: {dosya: sürüm},   global_counter: toplam yazma
```

| 5N1K | Cevap |
|---|---|
| Ne | Sürüm sayaçları + gözlem/komut kayıtları |
| Neden | Trace düz liste; ilişki (tekrar/bayat) tutmaz — ledger bunu ekler |
| Nasıl | Her tool'da `record()`: read→gözlem, write→sayaç++, search→dedup |
| Nerede | Compactor'ın Faz 1-2 kararının kanıt kaynağı |
| Ne zaman | Her tool sonrası güncellenir (ucuz, LLM'siz) |
| Kim | Tümüyle deterministik; hiç LLM yok |

**Nasıl çalışır — change counter:**
```
read config  → damga = local_counter[config] = 0
WRITE config → local_counter[config]: 0 → 1
is_stale(read)? → damga(0) < güncel(1) → BAYAT
```

### 2.3 `episode_graph.py` — CWL ajan-tipli yapı

**Ne:** Ajanın `delimiter` tool'uyla bildirdiği tiplenmiş bağımlılık grafiği
(CWL, 2606.11213). Ledger tipi *otomatik* çıkarır; CWL ajanın *açıkça*
bildirmesini sağlar.

```
Episode(name, type="expl"|"act", dependencies, description, event_seqs)
  expl → keşif (bilgi toplama)
  act  → eylem (yazma/test); hangi expl'e dayandığını bildirir
```

| 5N1K | Cevap |
|---|---|
| Ne | Ajanın delimiter ile kurduğu episode DAG'ı |
| Neden | "Bu 5 okuma tek bir keşif" bilgisi otomatik çıkarımdan zengin |
| Nasıl | `start/end/attach`; act bağımlılık deklare eder |
| Nerede | Compactor Faz 6 (episode eviction) |
| Ne zaman | Ajan delimiter çağırdığında |
| Kim | Yapıyı LLM (delimiter) kurar, eviction kararı deterministik |

**Nasıl çalışır — bağımlılık kısıtı:** bir expl episode ANCAK ona bağlı TÜM
act'ler evict edildiyse atılabilir (`evictable_expl`). Atılınca ajanın yazdığı
`description`'a iner.

### 2.4 `compactor.py` — 6 fazlı sıkıştırma eleği

**Ne:** Bütçe aşılınca trace'i, ledger+graph sinyalleriyle küçülten motor.

```
Faz 1  dedup         (aynı çağrı)            sıfır kayıp
Faz 2  staleness     (yazma sonrası bayat)   sıfır kayıp
Faz 3  hata-zinciri  (hata+düzeltme → ders)  düşük kayıp
Faz 4  keşif katlama (ls/grep dizisi→bulgu)  bulgu korunur
Faz 5  kategori      (act önce, expl sonra)  son çare
Faz 6  CWL episode   (bağımlılık-farkında)   ajan-tipli
```

| 5N1K | Cevap |
|---|---|
| Ne | Kademeli eviction politikası (CWL) |
| Neden | Bağlam sonlu; ama körlemesine silmek kritik bilgiyi kaybeder |
| Nasıl | 6 faz, kayıp artan sırada; her fazdan sonra bütçeyi ölç |
| Nerede | agent.py her tool sonrası çağırır |
| Ne zaman | Bütçe aşılınca (veya demo'da force) |
| Kim | Deterministik; Faz 5'in opsiyonel kolu LLM özet kullanabilir |

**Nasıl çalışır:** önce koruma penceresi (son N birim) ayrılır → dokunulmaz.
Sonra fazlar sırayla; en güvenli (dedup/stale) her zaman, kayıplı olanlar
(keşif/kategori) yalnızca bütçe hâlâ aşılıyorsa.

### 2.5 `tools.py` — gerçek dünya

**Ne:** sample_repo üzerinde gerçekten çalışan 5 tool + CWL delimiter, ek-a
mentalitesiyle.

| 5N1K | Cevap |
|---|---|
| Ne | read_file/list_dir/grep/edit_file/run_tests + delimiter |
| Neden | Ajanın dış dünyaya tek erişimi; gerçek dosya işi |
| Nasıl | `_safe_path` ile repo dışına çıkış reddedilir (model güvenilmez) |
| Nerede | agent.py DISPATCH ile çağırır |
| Ne zaman | Model tool_use ürettiğinde |
| Kim | Sen (host) çalıştırırsın; model sadece çağrı üretir |

### 2.6 `agent.py` — döngü + orkestrasyon

**Ne:** OpenAI-uyumlu ajan döngüsü (ek-a §7); trace katmanını boşluğa enjekte eder.

| 5N1K | Cevap |
|---|---|
| Ne | Gemma döngüsü + trace/ledger/episode/compactor orkestrasyonu |
| Neden | Üç deterministik yapıyı LLM döngüsüne bağlayan kabuk |
| Nasıl | while: model→tool_use→[kaydet+sıkıştır]→tool_result→tekrar |
| Nerede | POC'nin giriş noktası (TracingAgent.run) |
| Ne zaman | `run_demo.py --live` çağrılınca |
| Kim | Döngüyü kod çevirir; kararı Gemma verir |

---

## 3. Sıkıştırma mekanizması — tam diyagram

```
╔══ A. TETİKLEME ══════════════════════════════════════════════╗
   faz sınırı (write/test/konu değişimi)  VEYA  token > bütçe
                          │ tetiklendi
╔══ B. ESKİME (change counter) ════════════════════════════════╗
   read damgalanır (sürüm) → WRITE sayacı artırır →
   is_stale = damga < güncel ?   → yazmadan önceki okumalar BAYAT
   "eski" = SAAT değil, BAĞIMLILIK (yazmayla geçersizleşen)
                          │
╔══ C. EVICT ELEĞİ (6 faz, kayıp artan) ═══════════════════════╗
   koruma penceresi ayrılır (son N → dokunulmaz)
      │
      ▼ 1 dedup        aynı çağrı        [sıfır kayıp]
      ▼ 2 staleness    B'deki bayat      [sıfır kayıp]
      ▼ 3 hata-zinciri hata+düzeltme     [ders verbatim korunur]
      ▼ 4 keşif katla  ls/grep dizisi    [bulgu verbatim korunur]
      ══ hâlâ bütçe aşılıyorsa ══
      ▼ 5 kategori     act→expl          [açgözlü, yetince dur]
      ▼ 6 CWL episode  bağımlılık-farkında
                          │
╔══ D. ÇIKTI (5-alan özet §4) ═════════════════════════════════╗
   payload ~200 token         →    compacted ~15 token
   {name, args, output}            {niyet, girdi, sonuç, durum, etki}
   İçerik gitti, niyet+sonuç+neden KALDI.   Silme değil, gösterim değişir.
```

---

## 4. 5-alan özet nasıl üretiliyor — niyet geri kazanımı

Evict edilen bir birim (`_summarize_deterministic`) 5 alana iner. Her alanın
kaynağı **farklı**:

```
Event(payload={name:read_file, args:{path:config}, output:"40 satır"},
      intent_ref=4, verbatim=False)
                    │
       ┌────────────┼─────────────┬──────────────┬─────────────┐
       ▼            ▼             ▼              ▼             ▼
     niyet        girdi         sonuc          durum         etki
   intent_ref→   args'tan    verbatim?        status'tan   compactor'ın
   reasoning     düz         birebir : ilk    hata özel    reason'ı
   metnini AL                satır            işaretli     (evict sebebi)
       │
       └─ EN KRİTİK: niyet payload'da YOK.
          seq4 reasoning "portu bulmak için" → niyet bu.
          Ham trace'te olmayan, geri kazanılan tek alan (§4).
```

Somut:
```
seq5 read config, intent_ref=4, verbatim=False, reason="bayat"
  niyet ← seq4 reasoning → "portu bulmak için config oku"
  girdi ← args           → "path=config.py"
  sonuc ← verbatim=False → "1  HOST=..." (ilk satır)
  durum ← "ok"           → "ok"
  etki  ← reason         → "bayat (dosya değişti)"
```

**Neden niyet en değerli:** sonucu bir *soruya* bağlar → aranabilir yapar.
"config okundu" değil, "**port bulmak için** config okundu → 8080". Ajan sonra
port'a ihtiyaç duyunca cevap hazır, tekrar okumaz.

---

## 5. LLM nerede kullanılıyor, nerede değil

```
LLM = AJAN (agent.py)        → trace'i ÜRETİR      kaçınılmaz (işin kendisi)
LLM = özetleyici (opsiyonel) → Faz 5 use_llm_summary=True  varsayılan KAPALI
LLM = CWL etiketi (delimiter)→ yapıyı bildirir     karar yine deterministik
────────────────────────────────────────────────────────────────
compaction ÇEKİRDEĞİ         → LLM YOK
  ledger, episode_graph, Faz 1-4/6, is_stale, 5-alan özet
```

Kanıt: `test_deterministic.py` **API key olmadan 8/8 geçer** — çünkü sıkıştırma
kararının tamamı deterministik. Bu, §13'ün "complexity trap" ilkesinin (basit
deterministik ≈ LLM özeti, yarı maliyet) doğrudan uygulaması.

---

## 6. Baştan sona — 5N1K ile özet

**Ne:** Ajanın tool geçmişini (trace) küçülten deterministik bir katman —
tekrarı, bayatı, hata zincirini, keşif gürültüsünü 5-alan özete indirir.

**Neden:** Uzun görevde trace bağlamı doldurur; asıl israf tek çıktıda değil,
çıktılar *arasındaki* ilişkide (aynı dosya 3 kez, yazma sonrası bayat okuma).

**Nasıl:** Ledger sürüm sayaçlarıyla ilişkiyi (tekrar/bayat) deterministik
tespit eder; compactor 6 fazlı elekle en güvenliden en agresife sıkıştırır;
her birim niyet+sonucu koruyan 5-alan özete iner; koruma penceresi dokunulmaz.

**Nerede:** Ajan döngüsünde model çağrısı ile tool yürütmesi arasındaki boşlukta
(harness). Prefix (sistem promptu/tool tanımları) bozulmaz.

**Ne zaman:** Ledger her turda güncellenir (ucuz); compaction faz sınırında veya
token eşiğinde tetiklenir (seyrek). Eskime, is_stale her çağrıldığında anlık
hesaplanır — kalıcı bayrak yok.

**Kim:** Kararın tamamı deterministik (ledger + compactor + graph). LLM yalnızca
trace'i üreten ajan olarak, opsiyonel özetleyici olarak, ve CWL'de yapı etiketi
koyan taraf olarak belirir — sıkıştırmanın *kendisi* için gerekmez.

---

## Çalıştır

```bash
python run_demo.py --synthetic     # çekirdek, API'siz — %66 azaltma gösterir
python test_deterministic.py       # 8/8 (API'siz)
python run_demo.py --live          # gerçek Gemma (.env gerekir)
```

**İlgili:** `README.md` (kurulum/mimari) · `../report/13-tool-trace-compaction.md` (teori)
