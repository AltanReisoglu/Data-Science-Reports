# POC — Tool Trace Compaction

`report/13-tool-trace-compaction.md`'de tasarlanan trace compaction'ın çalışan
kanıtı. Deterministik bir **execution ledger** (Ledger, arXiv 2608.00808) ile
**kademeli eviction** (CWL, arXiv 2606.11213) uygular; opsiyonel olarak gerçek
bir LLM ajan döngüsüne (OpenAI-uyumlu Gemma endpoint) takılır.

> **Not — LLM sağlayıcısı:** endpoint OpenAI-uyumlu (`/v1`, `base_url`, `api_key`)
> ve model **Gemma**. Bu yüzden kod Anthropic SDK değil, `openai` kütüphanesini
> `base_url` ile bu endpoint'e yönlendirir. Tool tanımlama mentalitesi yine de
> `report/ek-a-tool-referans.md`'den (spesifik isim, "ne zaman çağrılacağını"
> söyleyen description, path güvenliği).

## Çekirdek fikir

Trace compaction'ın değeri **LLM gerektirmez** — bu yüzden POC iki modda çalışır:

| Mod | Komut | API key | Ne gösterir |
|---|---|---|---|
| **Sentetik** | `python run_demo.py --synthetic` | ❌ gerekmez | Deterministik çekirdek: dedup + staleness + eviction + 5-alan özet + playbook + üç katman |
| **PTC** | `python run_demo.py --ptc` | ❌ gerekmez | Klasik vs sandbox bağlam kıyası (uzaysal, §12.10) |
| **Canlı** | `python run_demo.py --live` | ✅ gerekir | Gerçek Gemma ajan döngüsü, trace katmanı devrede |
| **Test** | `python test_deterministic.py` | ❌ gerekmez | Çekirdeğin birim testleri (12/12) |

## Hızlı başlangıç

```bash
# 1. Çekirdeği hemen gör (kurulum gerekmez):
python run_demo.py --synthetic
python test_deterministic.py

# 2. Canlı mod için:
pip install -r requirements.txt
cp .env.example .env            # LLM_API_KEY'i doldurun
python run_demo.py --live                 # compaction açık
python run_demo.py --live --no-compaction # taban çizgisi (karşılaştırma)
```

## Sentetik demo ne yapıyor

`config.py`'deki portu 9090 yapıp doğrulayan bir ajanın **gürültülü** trace'ini
taklit eder — gerçek ajanların yaptığı gibi: aynı dosyayı 3 kez okur, grep'i
tekrarlar, sonra yazar. Ölçülen sonuç:

```
Ham trace: 12 tool birimi, 3453 token
  → 1 tekrar, 3 bayat gözlem tespit edildi (LLM'siz)
Sıkıştırma: 9 birim işlendi (6 özetlendi B.12, 3 silindi B.11)
Öncesi 3453 → Sonrası 1050 token   (%69,6 azalma)
Koruma penceresi (son 3 birim) dokunulmadı; 2 ders playbook'a yazıldı
```

## Mimari — §13 §6.7 diyagramının kodu

```
agent.py       ajan döngüsü (ek-a §7 manuel döngü, OpenAI-uyumlu)
   │  model çağrısı ile tool yürütmesi ARASINDAKİ boşluğa girer:
   ▼
trace.py       Format 3 tiplenmiş olaylar (§7) + 5-alan özet (§4)
ledger.py      execution ledger: observation/command records + change counter
   │           → dedup (aynı çağrı) + staleness (yazma sonrası bayat)  [§6.5.1]
compactor.py   kademeli eviction (§5.2 CWL), 6 faz güvenlik sırasında:
   │             1. dedup          (aynı çağrı → tekrar)         sıfır kayıp
   │             2. staleness      (yazma sonrası bayat gözlem)  sıfır kayıp
   │             3. hata-zinciri   (başarısız+düzeltme → ders)   düşük kayıp
   │             4. keşif katlama  (ls/grep dizisi → bulgu)      bulgu korunur
   │             5. kategori       (act önce, expl sonra)        son çare
   │             6. CWL episode    (ajan-tipli, bağımlılık-farkında)
   │           koruma penceresi (§6.5) + verbatim koruma (§7)
episode_graph.py CWL delimiter protokolü: ajan trace'ini expl/act olarak
   │           tipler + bağımlılık bildirir; expl ancak bağlı act evict
   │           edildiyse atılabilir (§5.2 bağımlılık kısıtı)
playbook.py    ACE öğrenen bağlam (§11 K4): compactor bir hata-dersi/bulgu
   │           çıkarınca buraya ARTIMLI DELTA yazar; trace evict edilse de
   │           ders kalıcı (context collapse yok). Bağlamın üstüne enjekte edilir.
ptc.py         PTC sandbox (§11 K3 / §12.10): run_code kodunu kısıtlı namespace'te
   │           çalıştırır; iç tool çağrıları trace'e girmez, yalnızca print girer.
tools.py       gerçek tool'lar + CWL delimiter + run_code (PTC) + şemalar
```

## §11 modern metotları — POC'ye eklenenler

Deterministik çekirdeğin üstüne §11/§12'nin üç güncel yöntemi bağlandı:

| Metod | §11 | Ne yapar | Nerede |
|---|---|---|---|
| **ACE playbook** | K4 / §12.7 | Evict edilen hata-dersi ve bulguları AYRI kalıcı playbook'a yazar (append-only delta → context collapse yok). Trace sıkışsa da ders durur. | `playbook.py`, compactor Faz 3/4 |
| **Göreve-koşullu sıkıştırma** | K5 / §12.8 | Özet detayını göreve alaka ile ayarlar: görev anahtar kelimesi geçen çıktı daha çok, geçmeyen daha az korunur. | `_task_relevant`, `_summarize_deterministic(task)` |
| **Context editing ≠ compaction** | B.11 vs B.12 | Bayat birimin TAZE kopyası bağlamda canlıysa **siler** (yer tutucu, ~0 token); değilse **özetler**. İki ayrı bilgi-kaybı katmanı. | `Event.cleared`, compactor Faz 2 |
| **PTC (programatik tool çağrısı)** | K3 / §12.10 | Model tool'ları TEK TEK değil KOD olarak çağırır; N çağrı sandbox'ta olur, bağlama yalnızca `print` girer. **Uzaysal** (ara sonuç hiç girmez), zamansal değil. | `ptc.py`, `run_code` tool'u |

Böylece §13'ün bilgi-kaybı merdiveni tamamlandı:
**KORU (pencere) → ÖZETLE (B.12) → SİL (B.11, çünkü ders playbook'ta).**
PTC ise merdivenin *öncesinde* durur: ara sonuç **hiç girmediği** için sonradan
sıkıştırmaya gerek kalmaz (zamansal vs uzaysal ayrımı, §08.8).

PTC demosu (`python run_demo.py --ptc`) — 3 .py dosyasında PORT arama:
```
KLASİK: 3 read_file olayı → 375 token (ham içerik bağlamda)
PTC   : 1 run_code olayı  → 107 token (sadece print) · 4 çağrı sandbox'ta
→ %71 az; hata olursa stack trace sandbox'ta kalır (tur harcamaz)
```

Sentetik demoda ölçülen sonuç (12 birim, 3453 token):
```
KORUNDU (tam)   : [19, 21, 23]        ← koruma penceresi
ÖZETLENDİ (B.12): [1, 3, 7, 11,13,17] ← 5-alan özet
SİLİNDİ  (B.11) : [5, 9, 15]          ← taze kopya canlı → sıfıra yakın
→ 3453 → 1050 token (%69,6); playbook 2 kalıcı ders (47 token)
```

| Dosya | §13 karşılığı |
|---|---|
| `trace.py` | §4 (5-alan şema), §7 (Format 3 tiplenmiş olay) |
| `ledger.py` | §6.5.1 (Ledger change counter, dedup/staleness) |
| `compactor.py` | §5.2 (CWL kademeli eviction), §6 (eskime, koruma penceresi) |
| `agent.py` | §6.7 (uçtan uca akış), ek-a §7 (manuel döngü) |
| `tools.py` | ek-a §3 (description), ek-a §13 (path güvenliği) |

## Çok-turlu chat — tracing arkada birebir çalışır

`chat.py` — sen konuşurken her mesajda Gemma + tool'lar çalışır, trace
compaction ARKADA işler. Trace turlar boyunca **birikir** ama sıkışık kalır.

```bash
python chat.py            # dosya tool'ları (sample_repo)
python chat.py --equity   # finansal tool'lar (XOM vb.)
python chat.py --no-compaction   # taban çizgisi
```

Her yanıttan sonra arkadaki iş tek satırda görünür:
```
sen › XOM P/E ve hedef?
asistan › XOM P/E 13.2, hedef $128.5.
  ┄ trace: 3 birim · 1107 tok · bu tur: +3 tool, compaction ×2, evict 0 · playbook 0 ders
sen › analist ve teknik?
asistan › Analist Moderate Buy; RSI 54 nötr.
  ┄ trace: 5 birim · 916 tok (%20 sıkışık) · bu tur: +2 tool, compaction ×2, evict 2 · playbook 2 ders
```
Komutlar: `/trace` (kaderler) · `/playbook` (öğrenilen) · `/ledger` · `/reset` · `/quit`.
`agent.py` artık çok-turlu (`send()`); trace/ledger/playbook/mesajlar turlar arası korunur.

## Tarayıcı paneli — aynı trace, altı mantık

```bash
python chat_server.py --port 8010     # → http://localhost:8010
```

Sağ sütundaki tool listesi artık makine değil **insan** okuyacak biçimde:
ne çağrıldı, ne için, ne döndü, ona ne oldu ve **neden** olduğu tek bakışta.

Üstteki **⇄ mantık karşılaştır** düğmesi asıl ekranı açıyor: o ana kadar birikmiş
**tek bir gerçek trace**, altı ayrı compaction mantığından geçiriliyor ve her biri
kendi sekmesinde duruyor.

| sekme | ekol | ne yapar |
|---|---|---|
| **CWL · bu POC** | deterministik | kademeli eviction: dedup → bayat → hata-zinciri → keşif katlama → kategori → episode |
| **hermes** | deterministik | dedup → tip-farkında tek satır özet → arg-kırp → basınç demotion |
| **opencode** | deterministik | canlı spill (diske dök) + backward-prune |
| **openclaw** | LLM-özet | grupla → parçala → LLM chunk-özeti |
| **codex** | hibrit | ortadan-kesme + model-turn windowing (handoff özeti) |
| **claude_code** | hibrit | microcompaction (diske dök + referans) + auto-compaction |

Beş harness mantığı `../demo-brain-agent/compaction.py`'den geliyor — orada zaten
gerçek sistemlerin davranışı taklit edilmiş, burada yeniden yazılmadı.

Her tool satırı açılıyor ve **ÖNCE (ham çıktı) / SONRA (context'te kalan)** yan yana,
**kırpılmadan** görünüyor. Yanında o mantığın o birime neden dokunduğu yazıyor.
Birleştiren stratejilerde birimin gittiği yer de gösteriliyor: "bu birim ayrı bir
mesaj olarak kalmadı" + o mantığın **ürettiği** yeni metin (handoff/konuşma özeti).

Ölçülen (8 tool birimi, 1.957 token ham bağlam, bütçe 1.200):

```
CWL · bu POC   1.957 → 1.465   %25,1      3 birim özete indi, 5'i korundu
hermes         1.957 →   968   %50,5      3 birim tek satıra indi
opencode       1.957 → 1.957   %0,0       çalıştı, hiçbir birime DOKUNMADI
openclaw       1.957 →   214   %89,1      8 birim tek LLM özetinde birleşti
codex          1.957 →   709   %63,8      7 birim handoff özetine girdi
claude_code    1.957 →   690   %64,7      7 birim konuşma özetine girdi
```

`opencode`'un %0'ı hata değil, **bulgu**: eşikleri (2000 satır / 50KB spill,
en yeni 40K token korunur) 2 bin token'lık bir bağlamda hiç aşılmıyor. OpenCode'un
mantığı büyük çıktılar için tasarlanmış; küçük ama çok sayıda tool birimi olan bir
trace onun radarına hiç girmiyor. Sekmedeki log bunu satır satır gösteriyor.

Kazanç sıralaması yanıltıcı okunmasın: **ne kadar küçülttüğü kadar geriye ne
bıraktığı önemli.** openclaw %89 kazanıyor ama 8 birimin hepsi tek bir LLM özetine
giriyor — ÖNCE/SONRA panelinde tam olarak neyin kaybolduğu görülebiliyor.

> **Yol boyunca çıkan hata.** Düğüm bazlı ÖNCE/SONRA görünümü eklenince
> `get_company_info` biriminin **31 → 32 token**'a çıktığı görüldü: sıkıştırma
> bağlamı BÜYÜTÜYORDU. Sebep, `_evict_event`'teki fayda güvencesinin trace
> muhasebesine (payload JSON vs özet dict) bakması, ama özetin bağlama
> `summary.render()` metni olarak düşüp yalnızca `output`'un yerini alması —
> **iki farklı cetvel.** Ayrıca keşif-katlama fazı bu kontrolü hiç yapmıyordu.
> `_fayda_var()` artık iki ölçeği birden kontrol ediyor ve faz onu kullanıyor.
> Ders: *ölçüm, etkinin düştüğü yerden alınmalı.*

## Genel ledger + equity case (HF dataset case 5.8)

Sistem dosya senaryosuna değil, **herhangi bir domain'e** oturacak şekilde
genelleştirildi. `ledger.py` artık her tool'un bildirdiği sözleşmeyi kullanır:

```python
TOOL_META = {
  "get_key_financial_ratios": {"cat":"read",  "resource": lambda a: a["ticker"]},
  "get_stock_price":          {"cat":"read",  "resource": lambda a: a["ticker"], "ttl":1},
  "visualize_data":           {"cat":"write", "resource": lambda a: "chart"},
}
ExecutionLedger(tool_meta=TOOL_META)   # kaynak=ticker, dosya değil
```

İki genelleştirme:
- **Kaynak anahtarı** artık `args.path` sabit değil — tool bildirir (path / ticker / tablo)
- **İki eskime türü**: mutasyon (yazma → sürüm eskir) **VE** volatilite (`ttl` →
  fiyat/web YAZMA olmadan zamanla bayatlar). §13'ün eksik kalan zaman-volatilite ucu.

`equity_tools.py` — dataset'in 10 finansal tool'unu simüle eder (mock, gerçekçi
boyutta). `run_equity.py` — **case 5.8'i (XOM, T5) uçtan uca çalıştırır**:
```
Ham: 9 tool birimi, 1851 token
  seq1 get_stock_price → BAYAT (TTL, yazma yok)   seq13 ratios → DUP≡seq7
Sıkıştırma: 1851 → 611 token (%67) · keşif dizisi bulguya katlandı
Playbook: revenue 2023=344.6 · P/E=13.2 · analist hedef $128.5 (filler değil, metrik)
```

Ek olarak **fayda güvencesi** (complexity trap): özet ham'dan küçük değilse
sıkıştırma yapılmaz. Küçük tool çıktılarında 5-alan özet ham'dan büyük olabilir;
guard bunu yakalar (`_evict_event`/`_clear_event` bool döner). Kanıt: mock veri
küçükken sistem %−6 "kazanç" verdi → guard ve gerçekçi veriyle %+67.

## Zarf iyileştirmeleri (üretim repolarından ilham)

Deterministik çekirdek bizim farkımız; ama *tetikleme/koruma/kademe* zarfında
üç üretim harness'inden ilham alındı (hepsi test edildi):

| İyileştirme | İlham | Ne yapar |
|---|---|---|
| **İki eşik** (`trigger` vs `target`) | Wegent `context_guard.py` | budget'ta tetikle, `target=0.6·budget`'a kadar in → histerezis (ACM "sawtooth"). `3234→2327` yerine `3234→1375` |
| **Çözülmemiş hata koruması** | NexAU `user_model_full_trace_adaptive.py` | düzeltilmemiş bir hata konumdan bağımsız korunur (ajan hâlâ çözecek) — koruma penceresinin kaçırdığı in-flight durum |
| **Greedy-by-size acil faz** (Faz 7) | Wegent emergency re-render | fazlardan sonra hâlâ hedef üstündeyse kalan ham birimleri **boyutça en büyük önce** evict eder; Faz 5'in atladığı `other` kategorisini de yakalar |

Bu repolarda olmayıp bizde olan: deterministik ledger/staleness, B.11/B.12 ayrımı,
ACE playbook. Onlarda olup bizde de artık olan: yukarıdaki üç zarf deseni.

## Tasarım kararları

- **Deterministik önce.** dedup/staleness/eviction hiç LLM çağırmaz — "complexity
  trap" (Lindenbauer 2508.21433) ve Ledger'ın sıfır-LLM sonucu. LLM yalnızca
  `--live` döngüsünde ve opsiyonel `use_llm_summary` özetinde.
- **Koruma penceresi.** Son N tool birimi asla evict edilmez (Glean `keep`).
- **Verbatim.** `grep`/`run_tests` çıktıları (dosya:satır, port) birebir korunur —
  parafraz edilmez (§7, CoACT NAP mantığı).
- **Niyet geri kazanımı.** 5-alan özetteki `niyet`, tool'u tetikleyen reasoning
  olayından (`intent_ref`) çıkarılır — ham trace'te olmayan alan (§4).

## Sınırlar (dürüstlük)

- Token ölçümü kaba tahmin (`char/4`), gerçek tokenizer değil.
- `--live` yolunda sıkıştırılmış geçmişin `messages[]`'e geri yazılması
  basitleştirilmiş; POC sıkışmayı `trace.total_tokens()` ile ölçer, mesaj
  dizisini birebir yeniden kurmaz. Üretimde bu bölüm CWL'nin in-place eviction'ı
  gibi tam uygulanmalı (ve KV-cache etkisi ölçülmeli — §5.2 uyarısı).
- Gemma'nın native tool-calling desteği endpoint gateway'ine bağlı; desteklenmiyorsa
  `--live` tool çağrısı üretmeyebilir. Çekirdek (sentetik) bundan bağımsızdır.
- NAP doğrulaması (§7) POC'de uygulanmadı — sonraki adım.
