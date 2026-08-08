# 13-Ek — Hermes'in Tool Özeti Bizim Tool'larla Çalışsaydı

**Ağustos 2026 · §13 eki · gerçek çıktı**

Bu ek tek soruyu cevaplar: **Hermes'in `_summarize_tool_result()` fonksiyonu bizim 119 ürün tool'umuzla çalışsaydı, her tool çıktısı için ne dönerdi?**

Hermes'in mantığını (`agent/context_compressor.py`) bizim tool'lara uyguladık: [`poc-trace-compaction/hermes_style.py`](../poc-trace-compaction/hermes_style.py). Aşağıdaki tüm satırlar `python3 hermes_style.py` ile üretilen **gerçek çıktı** — deterministik, sıfır LLM.

> **Hermes ilkesi:** Büyük bir tool çıktısını, tool tipine özel bir şablonla "özün özü" tek satıra indir — hangi tool, hangi kayıt/sorgu, ne sonuç, ne kadar büyüktü. Tanınmayan tool'a fallback: `[tool] (N chars result)`. Asla model çağırmaz, asla çökmez.

---

## 1. Zengin tek-satır özetler (tip-farkında)

Hermes bu tool tiplerini tanıyıp içeriğin özünü ayıklardı:

| Tool (girdi) | HAM | **Hermes dönerdi** |
|---|---|---|
| `jira_get_issue(ATLAS-101)` | 15 satır | `[jira_get_issue] ATLAS-101 → status In Progress · Task (15 lines)` |
| `jira_get_project(ATLAS)` | 12 satır | `[jira_get_project] ATLAS → open 17 (12 lines)` |
| `neta_get_project(MPP-409)` | 14 satır | `[neta_get_project] MPP-409 → planned_cost 4.2M TL (14 lines)` |
| `confluence_get_page(12345)` | 13 satır | `[confluence_get_page] 12345 → 'Mimari Kararlar' (13 lines)` |
| `confluence_search('mimari kararlar')` | 520 char | `[confluence_search] 'mimari kararlar' → 5 sonuç (520 chars)` |
| `jira_search_issues(ATLAS, Open)` | 10 satır | `[jira_search_issues] '*' → 5 sonuç (407 chars)` |
| `jira_resolve_project('Atlas')` | 10 satır | `[jira_resolve_project] 'Atlas' → 5 sonuç (433 chars)` |
| `jira_aggregate(count)` | 1 satır | `[jira_aggregate] → 47 (47 chars)` |
| `neta_count` / `ldap_org_count` | 1 satır | `[neta_count] → 17` · `[ldap_org_count] → 17` |
| `docx_create` | 1 satır | `[docx_create] belge oluşturuldu document_id=doc_a197ab v1` |
| `docx_add_chart(doc, İş Dağılımı)` | 1 satır | `[docx_add_chart] add_chart uygulandı v3` |
| `docx_get_outline(doc)` | 14 satır | `[docx_get_outline] doc_fe96de → 3 blok (v3)` |
| `xlsx_read_cells(doc)` | 10 satır | `[xlsx_read_cells] doc_fe96de → hücre bloğu (10 lines)` |

**Ayıklama mantığı (tool ailesine göre, deterministik):**
- **read (issue/proje/sayfa):** kimlik + bir salient alan (status / planned_cost / title)
- **search / resolve:** sorgu + "N sonuç"
- **aggregate / count:** **sayıyı birebir** taşı (47, 17)
- **write (create/add):** OK satırı + üretilen id/sürüm
- **outline:** blok sayısı + sürüm

---

## 2. Şablonu olmayan tool'lar → fallback (Hermes'in gerçek davranışı)

Hermes tanımadığı tool'a **"ilk satır + N char"** fallback'i verir. Bizim tarafta şablon yazmadığımız tool'lar aynen böyle döner:

```
jira_group_by      → [jira_group_by] jira_group_by(status): (181 chars)
jira_sprint_report → [jira_sprint_report] Sprint raporu: (243 chars)
analysis_run_sql   → [analysis_run_sql] SQL sonucu (SELECT ay, gelir FROM t): (261 chars)
ldap_org_members   → [ldap_org_members] ...kayıt bulund (376 chars)
```

Bu bir kusur değil — Hermes'in tasarımı bu: bilinen tiplere zengin özet, bilinmeyene güvenli fallback. Yeni tool tipleri için elle şablon eklenir.

---

## 3. Ham → Hermes: tam liste (gerçek ölçüm)

```
jira_resolve_project   433 char · 10 satır → [jira_resolve_project] 'Atlas' → 5 sonuç (433 chars)
jira_get_issue         437 char · 15 satır → [jira_get_issue] ATLAS-101 → status In Progress · Task (15 lines)
jira_get_project       364 char · 12 satır → [jira_get_project] ATLAS → open 17 (12 lines)
jira_search_issues     407 char · 10 satır → [jira_search_issues] '*' → 5 sonuç (407 chars)
jira_aggregate          47 char ·  1 satır → [jira_aggregate] → 47 (47 chars)
jira_group_by          181 char ·  8 satır → [jira_group_by] jira_group_by(status): (181 chars)
jira_sprint_report     243 char ·  9 satır → [jira_sprint_report] Sprint raporu: (243 chars)
neta_get_project       414 char · 14 satır → [neta_get_project] MPP-409 → planned_cost 4.2M TL (14 lines)
neta_count              23 char ·  1 satır → [neta_count] → 17 (23 chars)
ldap_org_count          27 char ·  1 satır → [ldap_org_count] → 17 (27 chars)
ldap_org_members       376 char · 11 satır → [ldap_org_members] ...kayıt bulund (376 chars)
confluence_search      520 char · 10 satır → [confluence_search] 'mimari kararlar' → 5 sonuç (520 chars)
confluence_get_page    489 char · 13 satır → [confluence_get_page] 12345 → 'Mimari Kararlar' (13 lines)
analysis_run_sql       261 char ·  8 satır → [analysis_run_sql] SQL sonucu (...): (261 chars)
docx_create             51 char ·  1 satır → [docx_create] belge oluşturuldu document_id=doc_a197ab v1
docx_add_chart          41 char ·  1 satır → [docx_add_chart] add_chart uygulandı v3
docx_get_outline       515 char · 14 satır → [docx_get_outline] doc_fe96de → 3 blok (v3)
xlsx_read_cells        305 char · 10 satır → [xlsx_read_cells] doc_fe96de → hücre bloğu (10 lines)
```

---

## 4. Hermes bizim tool'larda neyi KAZANIR, neyi KAYBEDER

**Kazanır:** Çok satırlı okumaları (issue 15 satır, outline 14 satır, sayfa 13 satır) tek bilgilendirici satıra indirir — bağlamda büyük yer açar, "ne olduğu" bilgisini korur.

**Kaybeder (bizim sistemin taşıdığı ama Hermes'in bilmediği iki şey):**
1. **"Neden" izi yok.** Hermes "bu ATLAS-101'in 2. kez okunması (tekrar)" ya da "bu okuma bayat" diyemez — ledger tutmuyor. Bizim `etki: tekrar (≡ seq=2)` bilgisi Hermes'te yok.
2. **Verbatim koruması yok.** Hermes `jira_get_issue`'yu da tek satıra ezer (`status In Progress`) → **story_points, assignee, priority gibi detay gider**. Bizim sistem verbatim tool'u kırpmaz: TAM tutar ya da (kopyası canlıysa) SİL'e indirir. Yani Hermes kritik veriyi bu adımda kaybedebilir; onu geri getirmek için **kuyruk koruması** veya sonraki **LLM özetine** güvenir.

---

## 5. Özet

Hermes'in `_summarize_tool_result` mantığı bizim tool'larla **sorunsuz çalışır** — tanıdığı tiplere zengin tek-satır özet, tanımadığına güvenli fallback verir; hepsi deterministik ve sıfır LLM. Somut örnek: `jira_get_issue`'nun 15 satırı → `[jira_get_issue] ATLAS-101 → status In Progress · Task`.

Fark bizim sistemle: Hermes **çıktının içeriğine** odaklanır (tip-farkında özet), bizim ledger **çıktının ilişkisine** (tekrar/bayat/seq) ve **verbatim korumasına**. İkisi birleştirilse en güçlüsü olurdu: Hermes'in tip-farkında `sonuç` üretimi + bizim ledger sinyali ve verbatim koruması.

---

*Örnekler `poc-trace-compaction/hermes_style.py` gerçek çıktısından (Ağustos 2026). Hermes mantığı `NousResearch/hermes-agent` `agent/context_compressor.py`'den uyarlandı.*
