# Bağlam Mühendisliği: LLM Ajanlarında Bağlamın Yaşam Döngüsü

**Staj raporu** · Ağustos 2026

---

## Raporun tezi

> **Bir ajanın yeteneği modelinin değil, bağlamının fonksiyonudur.**
>
> Dil modeli durumsuz (stateless) bir fonksiyondur; hafızası, araçları, becerileri ve sürekliliği yoktur. Bunların hepsi her turda **harness tarafından yeniden inşa edilen bağlamın** içindedir. Dolayısıyla bir ajan sistemi tasarlamak, model seçmekten çok **sonlu bir dikkat bütçesine neyin gireceğine karar vermektir.**

Rapor bu tezi altı mekanizma üzerinden inceliyor: tool, skill, memory, getirme (retrieval), artefakt işleme ve bağlam basıncı yönetimi. Her biri aynı problemin farklı bir çözümü: *bilgi bağlama girmesin, gerektiğinde girsin, ya da ucuza girsin.*

---

## Bölümler

| # | Bölüm | İçerik |
|---|---|---|
| [01](01-giris.md) | **Giriş** | Model/harness ayrımı, prompt→context engineering geçişi, context rot ve lost-in-the-middle, dikkat bütçesi |
| [02](02-bir-akisin-hayati.md) | **Bir akışın hayatı** | İstek gövdesinin tam anatomisi, tur tur bağlam akışı, `<system-reminder>`, streaming, `stop_reason` |
| [03](03-tool-katmani.md) | **Tool katmanı** | ReAct→native geçişi, "model tool'ları nasıl biliyor", döngü sahipliği, framework karşılaştırması |
| [04](04-skill-katmani.md) | **Skill katmanı** | Üç katmanlı progressive disclosure, koşullu prompt enjeksiyonu, tetikleme kuralları |
| [05](05-memory-katmani.md) | **Memory katmanı** | Dosya tabanlı hafıza, index deseni, yetki ve bayatlama sorunları, anchored summarization |
| [06](06-getirme-ve-arama.md) | **Getirme ve arama** | JIT vs ön-hesaplanmış retrieval, grep hunisi, metadata sinyali |
| [07](07-artefakt-isleme.md) | **Artefakt işleme** | docx/pptx/xlsx/pdf — bağlama girmeyen veriyle çalışmak |
| [08](08-baglam-basinci.md) | **Bağlam basıncı** | Kırpma, context editing, compaction, subagent, cache ekonomisi |
| [09](09-olcum.md) | **Ölçüm ve değerlendirme** | Probe tabanlı değerlendirme, metrikler, drift göstergeleri, deneyler |
| [10](10-sonuc.md) | **Sonuç** | Bulguların sentezi, mekanizma tablosu, sınırlar |
| [11](11-guncel-durum-ve-harness-atlasi.md) | **Güncel durum ve harness atlası** | Alanın 2026 hâlinden çekirdeğe 8 katman; her harness yapısının tam I/O formu; uçtan uca tur izi; kontrol listesi |
| [12](12-mcp-ve-modern-yontemler.md) | **MCP ve modern yöntemler** | MCP'nin wire mekaniği (JSON-RPC, transport, yaşam döngüsü, güvenlik); §11 yöntemlerinin çalıştırılabilir kod hâli (ACE, sıkıştırma, ajanik arama, PTC) |
| [Ek A](ek-a-tool-referans.md) | **Tool referansı** | Uygulama düzeyinde referans, dil eşlemeleri, kontrol listesi |

**Önerilen okuma sırası:** 01 → 02 → 03 → (04–07 herhangi bir sırada) → 08 → 09 → 10 → 11 → 12.

Aceleniz varsa: **01 + 02 + 10** raporun iskeletini verir.
Tek dosya okuyacaksanız: **[11](11-guncel-durum-ve-harness-atlasi.md)** — kendi başına okunabilecek şekilde yazıldı; hem alanın güncel haritasını hem her mekanizmanın wire düzeyinde formunu içerir.

---

## Yöntem

Rapor üç tür malzemeye dayanıyor:

**1. Doğrudan gözlem.** Raporun yazıldığı Claude Code oturumunun kendisi vaka çalışması olarak kullanıldı. Bağlamda ne olduğu, tool şemalarının nasıl bölündüğü, skill'in nasıl enjekte edildiği, tool çıktısının nerede kırpıldığı — bunlar teorik anlatım değil, oturum içinde gözlemlenip kaydedilmiş olgular. İlgili yerlerde açıkça işaretlendi.

> ⚠️ **Sınır:** Gözlem, modelin kendi bağlamına erişimiyle sınırlıdır. Harness'in iç implementasyonu (tam serialization şablonu, kırpma algoritması, cache anahtarlama) gözlemlenemez; bu noktalarda çıkarım yapıldığı belirtildi.

**2. Birincil kaynaklar.** Sağlayıcı dokümantasyonu ve mühendislik blogları.

**3. İkincil kaynaklar.** Sentez niteliğindeki teknik yazılar, eleştirel okumayla.

Sayısal iddialar (token maliyetleri, cache oranları) §09'daki deneylerle doğrulanabilir; script'ler rapora dahildir.

---

## Kaynakça

| Kaynak | Tür | Tarih | Not |
|---|---|---|---|
| Anthropic — *Effective context engineering for AI agents* | Birincil (satıcı) | Eyl 2025 | Context rot, attention budget, JIT retrieval, compaction/note-taking/subagent üçlüsü. **Satıcı yayını** — kavramsal iddiaları bağımsız doğrulanmalı |
| Bala Priya C — *Effective Context Engineering: A Developer's Guide* (ML Mastery) | İkincil (sentez) | Nis 2026 | RAM/disk metaforu, 4 katman taksonomisi, bloat vs poisoning, **probe tabanlı değerlendirme** |
| Glean Engineering — *The harness as the context manager* | Birincil (satıcı) | 2026 | *"Harness = dağıtık bağlam yönetim sistemi"*; PTC, skill indeksi, uzaysal/zamansal ayrımı. **Üçüncü taraf ölçümleri içerir** (LangChain, Vercel) ⚠️ *bu ölçümler birincil kaynaktan doğrulanmalı* |
| Anthropic Claude API dokümantasyonu | Birincil | Sürekli | Tool use, prompt caching, context editing, compaction, tool search, Agent Skills |
| Claude Code oturum gözlemi | Birincil (özgün) | Ağu 2026 | §02, §03, §04, §08'deki canlı kanıtlar |
| Liu et al. — *Lost in the Middle* | Akademik | 2023 | Konumsal dikkat yanlılığı ⚠️ *doğrulanmalı* |
| Drew Breunig — *How Long Contexts Fail* | İkincil | — | Poisoning/distraction/confusion/clash ayrımı ⚠️ *okunmalı* |
| Yao et al. — *ReAct* | Akademik | 2022 | §03'teki tarihsel karşılaştırmanın referansı ⚠️ *doğrulanmalı* |
| Zhang et al. — *Agentic Context Engineering (ACE)*, arXiv 2510.04618 | Akademik | Eki 2025 | Bağlam = evrilen playbook; **brevity bias** ve **context collapse** teşhisi; artımlı delta güncelleme; +%10,6 ajan / +%8,6 finans. §11 K4 |
| *ACON*, arXiv 2510.00615 · *Squeez* 2604.04979 · *Self-GC* 2607.00692 | Akademik | 2025–2026 | Öğrenilmiş / göreve koşullu bağlam sıkıştırma; ACON %26–54 tepe token azalması. §11 K5 |
| Claude Code'un vektör aramayı bırakması (May 2025) + AAAI 2026 Amazon Science ölçümü | İkincil | 2025–2026 | Ajanik arama = RAG sadakatinin %94,5'i, sıfır vektör deposu. §11 K2 ⚠️ *ikincil aktarım* |
| Harness engineering derlemeleri (Osmani, Greyling, MindStudio, LLM-Harness survey) | İkincil | 2026 | *Ajan = Model + Harness*; en iyi/en kötü harness arası **23,8 puan** fark. §11 K7 ⚠️ *benchmark birincilden doğrulanmalı* |
| Bağlam grafiği / ontoloji / semantik katman yazıları (Gartner D&A 2026 aktarımı) | İkincil | 2026 | Context layer vs Intelligence layer ayrımı; 522 sorguda %38 doğruluk farkı. §11 K6 ⚠️ *doğrulanmalı* |
| Kullanıcı kaynak listeleri — `lists/agents.md`, `lists/context_eng_*.md` | Birincil (özgün derleme) | Ağu 2026 | Literatür haritası ve token azaltma depoları envanteri. §11 A.8 ⚠️ *depo iddiaları bağımsız ölçülmedi* |

⚠️ işaretli kaynaklar rapora dahil edilmeden önce birincil metinden doğrulanmalıdır — bu raporda ikincil aktarımla kullanılmışlardır.

---

## Tarihsel not

Alan hızlı hareket ediyor. Rapordaki API özellikleri Ağustos 2026 itibarıyla geçerlidir. Kaynakların yayın tarihiyle bugün arasındaki farklar §03 ve §08'de açıkça işaretlendi — özellikle Anthropic yazısının (Eyl 2025) "sorun" olarak tanımladığı bazı konuların (şişkin tool setleri, tool sonucu birikimi) API seviyesindeki çözümleri **yazıdan sonra** ürünleşti.
