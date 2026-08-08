# 13-Ek — Tool Çıktısını Özetleme: Hermes vs Bizim Sistem (örneklerle)

**Ağustos 2026 · §13 karşılaştırma eki · gerçek koddan örnekler**

Bu ek tek bir işi mercek altına alır: **büyük bir tool çıktısı, model çağırmadan (deterministik) nasıl tek/az satıra indirilir?** İki sistemi gerçek örneklerle yan yana koyar:
- **Hermes** (`NousResearch/hermes-agent`) → `_summarize_tool_result()`
- **Bizim** (`poc-trace-compaction`) → `_summarize_deterministic()` (5-alan ÖZET) + SİL

> Her iki örnek kümesi de **çalışan koddan** alındı. Hermes örnekleri repo'nun kendi testlerinden (`tests/agent/test_context_compressor.py`); bizim örnekler `python3` ile üretilen gerçek çıktı.

**Ortak nokta (en önemli):** İkisi de **sıfır LLM**. Tool çıktısını kısaltan bu adım, asıl LLM özetinden **önce** çalışan ucuz, deterministik bir koddur. Fark, şablonun zenginliğinde ve verbatim veriye davranışta.

---

## 1. Hermes — `_summarize_tool_result()`: tool-tipine göre tek satır

Hermes, büyük bir tool çıktısını **tool tipine özel bir şablonla** tek satıra indirir. Kod deterministik (test'ler tam string eşitliği iddia ediyor → LLM olamaz).

### Örnek A — terminal
```
GİRİŞ (ham, ~47 satır):
  $ npm test
  > jest --coverage
  PASS src/a.test.js
  PASS src/b.test.js
  ... 43 satır daha ...
  Tests: 128 passed, 128 total

ÇIKIŞ (tek satır):
  [terminal] npm test → exit 0, 47 lines output
```
Ayıklanan: **tool adı** (terminal) · **komut** (args'tan) · **çıkış kodu** (content'ten) · **satır sayısı**.

### Örnek B — web_extract (repo testinden birebir)
```
GİRİŞ (ham, ~500 karakter, 2 URL + gövdeler)

ÇIKIŞ:
  [web_extract] https://example.com/a (+1 more) (500 chars)
```
Ayıklanan: **ilk URL** · **kaç tane daha** · **toplam karakter**.

### Örnek C — bilinmeyen tool / hata (fallback)
```python
except Exception:
    return f"[{tool_name}] ({_len:,} chars result)"
# → "[custom_tool] (12,431 chars result)"
```
Şablon tanınmazsa ya da parse patlarsa: en azından "hangi tool, kaç karakter." **Özet asla compaction'ı çökertmez.**

**Mekanizma:** JSON parse (`tool_args`) + alan çıkarımı + f-string. Tool-tipi başına ayrı bir dal. Model yok.

**Kritik davranış — her zaman tek satır:** Hermes bu adımda **verbatim ayrımı yapmaz.** Çıktı ister log, ister kritik sayı olsun, hepsi tek satıra iner (kayıplı). Kritik detay gerekiyorsa onu **kuyruk koruması** (son mesajlar) ya da sonraki LLM özeti taşır.

---

## 2. Bizim sistem — `_summarize_deterministic()`: 5-alan kart + SİL

Biz de deterministik (sıfır LLM), ama **jenerik 5-alan** şablon + **ledger sinyali** kullanırız, ve **verbatim veriyi ayrı ele alırız**.

5 alan: `niyet · girdi · sonuç · durum · etki`. `etki`, ledger'ın *neden* sıkıştırdığını taşır (tekrar/bayat).

### Örnek A — confluence_search (non-verbatim) → ÖZET küçülür
```
HAM (152 tok, 10 satır):
  confluence_search: 'mimari kararlar' için 5 sonuç:
    1. CONFLUENCE-f13898 · 'mimari kararlar' ile eşleşme (skor 0.9)
    2. CONFLUENCE-5bf67a · ...
    ...(10 satır)

→ ÖZET (65 tok):
  [özet] niyet: confluence_search sonucuna bakmam gerekiyor · girdi: query=mimari kararlar ·
         sonuç: confluence_search: 'mimari kararlar' için 5 sonuç: / 1. CONFLUENCE-f13898…
         · durum: ok · etki: keşif katlandı
```
152 → 65 tok. `sonuç` kırpıldı, `etki` "neden"i taşıyor.

### Örnek B — jira_search_issues (non-verbatim) → ÖZET küçülür
```
HAM (128 tok) → ÖZET (48 tok):
  [özet] niyet: jira_search_issues sonucuna bakmam gerekiyor · girdi: project_key=ATLAS, status=Open ·
         sonuç: jira_search_issues: '*' için 5 sonuç:… · durum: ok · etki: tekrar (≡ seq=2)
```
Not: `etki: tekrar (≡ seq=2)` — bu, Hermes'te **olmayan** bir bilgi: "bu çağrı seq=2'nin tekrarı." Ledger'ımız bunu biliyor.

### Örnek C — jira_get_issue (VERBATİM) → ÖZET büyürdü → **SİL**
```
HAM (130 tok):
  Issue ATLAS-101:
    summary: ... type: Task  status: In Progress ... (15 satır kritik veri)

→ özet 146 ≥ ham 130 → FAYDA FRENİ → SİL:
  [silindi] tekrar ≡ seq=4 (aynı içerik canlı)
```
Kritik fark: jira_get_issue **verbatim** (issue detayı birebir korunmalı). 5-alan özet onu kırpmadığı için özet ham'dan büyük olur → **fayda freni** devreye girer → özet yerine **SİL** (aynı içerik seq=4'te canlı olduğu için güvenli).

### Örnek D — jira_aggregate (verbatim, küçük) → **SİL**
```
HAM (35 tok): jira_aggregate: metric=count → 47 (proje=ATLAS)
→ özet 49 ≥ ham 35 → fayda freni → SİL: [silindi] tekrar ≡ seq=4
```
Küçük verbatim çıktı — özetlemek büyütürdü, SİL doğru kader.

---

## 3. Yan yana — aynı iş, iki felsefe

| | **Hermes** `_summarize_tool_result` | **Bizim** `_summarize_deterministic` + SİL |
|---|---|---|
| LLM? | ❌ sıfır | ❌ sıfır |
| Şablon | tool-tipine özel tek satır | jenerik 5-alan (niyet/girdi/sonuç/durum/etki) |
| "Neden" izi | ❌ yok | ✅ `etki` (tekrar ≡ seq / bayat) |
| Verbatim veri | tek satıra iner (**kayıplı**) | korunur → TAM ya da SİL (fayda freni) |
| Tekrar bilgisi | ❌ | ✅ ledger dedup (≡ seq) |
| Bayat bilgisi | kaba (`drop_stale`) | ✅ sürüm/TTL |
| Fallback | `[tool] (N chars)` | fayda freni → SİL |
| Örnek | `[terminal] npm test → exit 0, 47 lines` | `[özet] niyet:… girdi:… sonuç:… durum:ok · etki:tekrar ≡ seq=2` |

## 4. Üç somut fark

**① Hermes tool-tipi bilir, biz kaynak-ilişkisi biliriz.**
Hermes'in şablonu `terminal` mi `web_extract` mı diye bilir ve ona göre alan ayıklar — **çıktının tipi** hakkında zengin. Bizimki tool-tipi bilmez ama **ledger sinyali** taşır: "bu tekrar mı, bayat mı, hangi seq'in kopyası." Yani Hermes **çıktının içeriğine**, biz **çıktının geçmişteki ilişkisine** odaklanırız.

**② Hermes her şeyi kırpar, biz verbatim'i koruruz.**
Hermes bu adımda kritik sayıyı da log'u da tek satıra indirir (kritik veri korumasını kuyruğa/LLM'e bırakır). Biz `verbatim` işaretli tool'ları (issue detayı, bütçe, sayım) **kırpmayız** — ya TAM tutar ya (kopyası canlıysa) SİL'e indiririz. Bu yüzden bizde "özet ham'dan büyük olursa SİL" mantığı var; Hermes'te buna gerek yok çünkü zaten her şeyi eziyor.

**③ İkisi de asıl LLM özetinden ÖNCE gelen ucuz ön-pas.**
En önemli ortak nokta: bu adım her iki sistemde de **model çağırmayan, bedava** bir ön-temizlik. Hermes sonra bir LLM özeti yapar; biz **hiç yapmayız** (deterministik yeterli). Yani Hermes'in ucuz ön-pası + LLM özeti = bizim tek katmanımızın (deterministik ÖZET/SİL) yaptığı işin iki adıma bölünmüş hali.

---

## 5. Özet

- **Hangi iş:** büyük tool çıktısını model çağırmadan kısaltmak.
- **Hermes:** tool-tipine özel **tek satır** (`[terminal] npm test → exit 0, 47 lines`). Zengin tip-farkındalığı, ama her şeyi kırpar (kayıplı), tekrar/bayat bilmez.
- **Biz:** jenerik **5-alan kart** + **SİL** (fayda freniyle). Tip bilmez ama **ledger ilişkisini** taşır (tekrar/bayat/seq) ve **verbatim veriyi korur**.
- **Ortak:** ikisi de **sıfır LLM**, deterministik, asıl özetten önce gelen ucuz ön-pas. Test edilebilir olmaları (tam string eşitliği) bunun kanıtı.

**Alınabilecek fikir:** Hermes'in **tool-tipine özel şablonları** bizim 5-alan `sonuç` alanını zenginleştirebilir — ör. terminal çıktısı için "exit code + satır sayısı", search için "N sonuç + ilk eşleşme" gibi tip-farkında `sonuç` üretmek. Böylece bizim ledger-ilişki avantajımızı Hermes'in içerik-farkındalığıyla birleştirebiliriz.

---

*Hermes örnekleri `agent/context_compressor.py` + `tests/agent/test_context_compressor.py`'den; bizim örnekler `poc-trace-compaction/` gerçek çıktısından (Ağustos 2026).*
