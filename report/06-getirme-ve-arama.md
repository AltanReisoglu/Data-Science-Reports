# 6. Getirme ve Arama: Bilgiyi Ne Zaman Bağlama Almalı

> **Bölümün tezi:** Retrieval bir ön-işleme adımı değil, **bir bütçe kararıdır.** Asıl soru "hangi belgeler alakalı" değil, "bağlam bütçesinin ne kadarı buna harcanmalı ve ne zaman harcanmalı"dır.

---

## 6.1 İki paradigma

| | **Ön-hesaplanmış getirme** | **Just-in-time (JIT)** |
|---|---|---|
| Ne zaman | Çıkarım **öncesi** | Çalışma **anında** |
| Nasıl | Embedding tabanlı arama; ilgili chunk'lar önden bağlama konur | Hafif **tanımlayıcılar** tutulur (dosya yolu, sorgu, link); tool ile gerektiğinde yüklenir |
| Kim karar verir | Boru hattı | **Model** |
| Analoji | Her şeyi ezberlemek | **İndeksleme sistemi kullanmak** |
| Bayatlama | İndeks eskir | **İmkânsız** — canlı okur |
| Gecikme | Düşük (önceden hazır) | Yüksek (tur gerektirir) |

Anthropic'in analojisi:

> *"we generally don't memorize entire corpuses of information, but rather introduce external organization and indexing systems like **file systems, inboxes, and bookmarks** to retrieve relevant information on demand."*

Bu, §01.4'teki RAM/disk metaforunun getirme tarafındaki karşılığıdır: veri diskte durur, RAM'e sadece gerekeni alırsın.

---

## 6.2 Kod tabanında grep vs embedding

Kod üzerinde çalışan ajanlar için Claude Code'un tercihi nettir:

> *"primitives like `glob` and `grep` allow it to navigate its environment and retrieve files just-in-time, effectively bypassing the issues of **stale indexing and complex syntax trees**."*

| | Embedding indeksi | grep / ripgrep |
|---|---|---|
| Kurulum | Chunk'la, embed'le, sakla, güncelle | **Yok** |
| Bayatlama | Kod değişir, indeks eskir | **İmkânsız** |
| Maliyet | Embedding API + vektör DB | ≈0 |
| Semantik eşleşme | ✅ "auth mantığı" → `verify_credentials()` | ❌ Birebir / regex |
| Kesinlik | Yaklaşık, skorlu | **Kesin** — ne bulduğunu tam bilirsin |
| Kapsam | İndekslenen ne varsa | Diskte ne varsa |

**Semantik boşluk kısmen kapanır:** model tek turda birden çok aday terim üretebilir. `auth` bulunamazsa `login|credential|session|token|verify` denenir. Bir aramanın maliyeti ~0 olduğu için beş arama yapmak ucuzdur.

---

## 6.3 Arama hunisi

Ucuzdan pahalıya, genişten dara. Her kademe bir sonrakini daraltmak için harcanır:

```
1. glob / rg --files   →  hangi dosyalar var       ~50 token
2. rg -l               →  hangilerinde geçiyor     ~30 token
3. rg -n -C3           →  nerede ve çevresi ne     ~300 token
4. Read(offset, limit) →  o bölgeyi tam oku      ~1.000 token
```

**Kademe atlamak felakettir.** 1'den doğrudan 4'e gidilirse tüm dosyalar okunur.

### Örnek: "retry mantığı nerede?"

```bash
# 1. Kapsam
rg --files -g '*.py' src/ | head -50

# 2. Hangi dosyalarda (SADECE dosya adları)
rg -l 'retry|backoff|max_attempts' src/
#   src/http/client.py
#   src/queue/worker.py
#   src/db/pool.py

# 3. Nerede, bağlamıyla
rg -n -C3 'def.*retry|retry\s*=' src/http/client.py
#   47:  def _retry_with_backoff(self, fn, max_attempts=3):

# 4. Sadece o bölge
Read(file_path="src/http/client.py", offset=40, limit=60)
```

**Toplam: ~1.500 token.** Üç dosyanın tamamı okunsaydı ~40.000.

---

## 6.4 Doğrudan gözlem: bu oturumdaki arama

Rapor oturumunun ilk turunda çalıştırılan komut:

```bash
grep -rEil 'openai|langchain_openai|google.generativeai|genai|mistralai|cohere|ollama' . \
  --exclude-dir=.git | head -20
```

Çıktı üç satırdı:

```
lists/agents.md
lists/ai.md
lists/reinforcement.md
```

Amaç: `claude-api` skill'inin SKIP kuralını değerlendirmek — "bu repoda başka bir LLM sağlayıcısı üzerinde mi çalışılıyor?"

Flag seçimleri bilinçliydi:

| Flag | Gerekçe |
|---|---|
| `-r` | Özyinelemeli |
| `-E` | `\|` alternasyonu için genişletilmiş regex |
| `-i` | `OpenAI` / `openai` ayrımı önemsiz |
| **`-l`** | **Sadece dosya adları** — karar için yeterli |
| `--exclude-dir=.git` | Git nesnelerinde binary gürültü |
| `\| head -20` | Patlamaya karşı üst sınır |

`-l` olmasaydı yüzlerce eşleşen satır dönerdi. **Karar için dosya adı yeterliydi; içerik gerekmiyordu.**

Karşıt örnek aynı oturumdan: `lists/agents.md` dosyası okunduğunda 31.289 token'lık bir çıktı üretti ve harness ~25K tavanında kesti (§08.2). Aynı soruya grep ile ~20 token'la cevap verilmişti.

---

## 6.5 Ajan için önemli flag'ler

```bash
# ── KAPSAM DARALTMA (önce) ──────────────────────────
rg 'pattern' --type py
rg 'pattern' -g '*.ts' -g '!*.test.ts'
rg 'pattern' src/ --max-depth 3

# ── ÇIKTI ŞEKLİ (bağlam maliyetini bu belirler) ─────
rg -l 'pattern'          # sadece dosya adları       ← EN UCUZ
rg -c 'pattern'          # dosya başına sayı         ← keşif için
rg -n 'pattern'          # satır no + satır
rg -n -C3 'pattern'      # + çevresinden 3 satır
rg -o 'v\d+\.\d+\.\d+'   # sadece eşleşen parça

# ── EŞLEŞME ─────────────────────────────────────────
rg -i 'pattern'          # harf duyarsız
rg -w 'get'              # tam kelime — 'target' eşleşmez
rg -F 'a.b.c'            # sabit string, regex değil
rg 'foo|bar|baz'         # TEK aramada üç terim

# ── SİGORTALAR ──────────────────────────────────────
rg 'pattern' | head -50
rg -m5 'pattern'         # dosya başına en fazla 5
```

**`rg` (ripgrep) tercih edilir:** `.gitignore`'a saygı duyar, `node_modules`/`.git`'i otomatik atlar, çok daha hızlıdır. Claude Code'un `Grep` tool'u ripgrep tabanlıdır.

### Keşif deseni: `-c` ile ısı haritası

```bash
rg -c 'TODO|FIXME|HACK' --type py | sort -t: -k2 -rn | head -10
# src/legacy/parser.py:47
# src/api/handlers.py:12
# src/utils/format.py:3
```

Dosya başına tek satır. `47` ile `3` arasındaki fark nereye bakılacağını söylüyor.

---

## 6.6 Anti-desenler

| ❌ | Sorun | ✅ |
|---|---|---|
| `rg 'function' .` | Binlerce eşleşme, bağlam taşar | `rg -c 'function' --type js \| head` |
| `Read(bigfile.py)` | 30K token, %2'si gerekli | `rg -n 'hedef'` → `Read(offset=…)` |
| `rg 'a'; rg 'b'; rg 'c'` | Üç ayrı tur | `rg 'a\|b\|c'` — tek tur |
| Sınırsız `rg` | Patlama riski | `\| head -50` veya `-m` |
| `rg` ile JSON ayrıştırma | Tırnak/kaçış kırar | `jq` |

Son satır önemli: **grep satır tabanlıdır**, yapılandırılmış veri için kırılgandır.

---

## 6.7 Yapılandırılmış veri ve büyük dosyalar

```bash
# JSON → jq
curl -s api/data | jq -r '.items[] | select(.status=="failed") | .id'

# Tablolu → awk
awk -F, '$3 > 1000 {print $1, $3}' sales.csv | head -20

# Devasa dosyanın ŞEKLİNİ gör, içeriğini değil
head -3 data.csv          # başlık + örnek satır
wc -l data.csv            # kaç satır
tail -3 data.csv          # son kayıtlar
```

Anthropic'in cümlesinin birebir karşılığı:

> *"leverage Bash commands like `head` and `tail` to analyze large volumes of data **without ever loading the full data objects into context**"*

---

## 6.8 Metadata bir sinyaldir

Getirme yapılmadan bilgi taşıyan bir katman var: **ortamın kendi düzeni.**

> *"the presence of a file named `test_utils.py` in a `tests` folder implies a different purpose than a file with the same name located in `src/core_logic/`"*

```bash
ls -la src/                                # boyut → karmaşıklık göstergesi
find . -name '*.py' -newermt '-7 days'     # son değişenler → alaka
git log --oneline -20 -- src/auth/         # bu modülde ne olmuş
rg --files | sed 's|/[^/]*$||' | sort -u   # dizin yapısı = mimari
```

Klasör hiyerarşisi, adlandırma kuralları ve zaman damgaları hem insana hem ajana bilginin **nasıl ve ne zaman** kullanılacağını söyler.

> **Bulgu 9.** İyi organize edilmiş bir depo, bağlama hiç girmeden ajana bilgi verir. Bu, bağlam mühendisliğinin dosya sistemine taşan kısmıdır: *dizin yapısı bir prompt'tur.*

---

## 6.9 Retrieval bir bütçe kararı olarak

Kod dışı korpuslarda (destek kayıtları, sözleşmeler, doküman tabanları) embedding tabanlı retrieval hâlâ doğru araçtır. Ancak ML Mastery'nin uyarısı geçerlidir:

> Yaygın hata: retrieval'ı basit bir yukarı-akış adımı sanmak — chunk getir, enjekte et, devam et. **Bağlam bütçesinin ne kadarını tüketmesi gerektiği** hiç sorulmaz.

### Dört teknik

| Teknik | Ne yapar |
|---|---|
| **Post-retrieval filtering** | Getirilen sonuçları enjekte etmeden **önce** puanla ve seç — *"one of the highest-leverage optimizations"* |
| **Semantic chunking** | Sabit boyut yerine doğal konu sınırlarından böl → anlam korunur |
| **Hybrid retrieval** | Semantik + anahtar kelime/metadata filtresi |
| **Agent-controlled triggering** | Aşağıda |

Hybrid için kanonik örnek: *"son 30 gündeki faturalama sorunları"* — semantik ilgi **ve** zaman kısıtı gerekir. Ne salt embedding ne salt keyword yeterlidir.

### Otomatik mi, ajan kontrollü mü

| | Artı | Eksi |
|---|---|---|
| **Otomatik** (her turdan önce) | Basit | **Yararlı olsun olmasın token enjekte eder** |
| **Ajan kontrollü** (tool olarak) | Hedefli sorgu, muhakeme zincirinde doğru anda | Modelin ihtiyacı fark etmesi gerekir |

> *"For most production systems, **agent-controlled retrieval is the better default** once the system is stable."*
> — ML Mastery

**Bu, Anthropic'in JIT tezine bağımsız olarak yakınsıyor.** İki ayrı kaynağın aynı sonuca varması, bulgunun gücünü artırır.

---

## 6.10 Ne zaman grep yetmez

| Durum | Alternatif |
|---|---|
| Semantik sorgu ("hata yönetimi nasıl yapılıyor") | Model çoklu terim üretir: `try\|except\|raise\|error\|Result<` |
| Tanım vs kullanım ayrımı | LSP / `ctags`, ya da `rg 'def foo'` vs `rg 'foo\('` |
| Çağrı grafiği, tip ilişkileri | AST aracı (`ast-grep`, `tree-sitter`) |
| Kod dışı doğal dil korpusu | **Gerçekten** embedding retrieval |

---

## 6.11 Hibrit — pratikteki cevap

Claude Code'un fiilî stratejisi ikisinin karışımıdır:

```
CLAUDE.md      → önden, naif şekilde bağlama düşer      (statik / ön-hesaplanmış)
glob, grep     → çalışma anında dosya getirir           (dinamik / JIT)
```

Hibrit, içeriği daha az dinamik olan alanlarda (hukuk, finans) daha da uygundur.

Ve bir trend öngörüsü:

> *"As model capabilities improve, agentic design will trend towards **letting intelligent models act intelligently**, with progressively less human curation."*

Pratik tavsiye ise değişmiyor: *"do the simplest thing that works."*

---

## 6.12 Takas — dürüstçe

JIT bedavaya gelmez:

> *"runtime exploration is **slower** than retrieving pre-computed data."*
> *"Without proper guidance, an agent can waste context by misusing tools, chasing dead-ends, or failing to identify key information."*

İki koşulu vardır: **iyi tool'lar** (§03) ve **iyi sezgiler** (system prompt yönlendirmesi). İkisi de yoksa ajan bağlamı boşa harcar — yanlış aramalar yapar, çıkmazları kovalar, kilit bilgiyi tanıyamaz.
