# 15 — AutoGen Araştırma Projesi: Kaynak Haritası

*Hazırlanma: 2026-08-12 · Kapsam: starlanan repolar + `agentic_tuffs.md` taraması + birincil kaynak doğrulaması*

---

## §0 — Önce şunu bil: AutoGen bakım modunda

Bu, projeye başlamadan önce bilmen gereken tek kritik gerçek. **GitHub'daki kendi README'sinden** (birincil kaynak, 2026-08-12'de doğrulandı):

> ⚠️ **Maintenance Mode** — "AutoGen is now in maintenance mode. It will not receive new features or enhancements and is community managed going forward. New users should start with Microsoft Agent Framework."

Repo sinyalleri bunu doğruluyor:

| Repo | ⭐ | Son commit | Son sürüm | Durum |
|---|---:|---|---|---|
| [microsoft/autogen](https://github.com/microsoft/autogen) | 60.4k | 2026-04-15 | `python-v0.7.5` (2025-09-30) | **Bakım modu** (arşivli değil) |
| [microsoft/agent-framework](https://github.com/microsoft/agent-framework) | 12.8k | 2026-08-11 | `python-1.13.0` (2026-07-30) | **Aktif — resmi halef** |
| [ag2ai/ag2](https://github.com/ag2ai/ag2) | 4.9k | 2026-08-12 | `v1.0.1` (2026-07-29) | **Aktif — topluluk fork'u** |
| [microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel) | 28.4k | 2026-08-11 | — | MAF'a devredildi |

**Olan biten:** Microsoft, AutoGen (araştırma kökenli, çok-ajanlı orkestrasyon) ile Semantic Kernel'i (kurumsal, üretim altyapısı) birleştirip **Microsoft Agent Framework (MAF)** yaptı. Ekim 2025'te public preview, **Nisan 2026'da 1.0 GA**. Her iki eski framework de bakım moduna alındı: hata ve güvenlik yaması evet, yeni özellik hayır.

Ayrıca ikinci bir çatallanma daha var — bunu karıştırmamak önemli:
- **microsoft/autogen v0.4+** → aktör modelli, event-driven yeniden yazım (`autogen-core`). Şimdi bakım modunda, halefi MAF.
- **AG2** → AutoGen'in **v0.2 kolundan** ayrılan topluluk fork'u, "AgentOS" olarak devam ediyor, v1.0'a ulaşmış durumda. Ayrı bir proje, ayrı API.

### Bu senin projen için ne demek

Bu bir sorun değil, **projenin çerçevesi**. Bir intern'in "AutoGen öğrendim" demesi 2026'da eksik; asıl değerli olan şu üçlüyü kurabilmek:

1. **AutoGen ne getirdi** — konuşma-merkezli çok-ajanlı programlama modeli, `GroupChat`, `UserProxyAgent`, aktör modelli çekirdek
2. **Neden yetmedi** — üretimde tıkanan yerler (durum yönetimi, tip güvenliği, telemetri, maliyet, güvenilirlik)
3. **Halefi ne yaptı** — MAF bu derslerin hangisini nasıl içselleştirdi

Yani projeyi "AutoGen tutorial" olarak değil, **"bir çok-ajan framework'ünün doğuşu, üretimde çarptığı duvar ve konsolidasyonu"** olarak kurgula. Aynı emekle çok daha güçlü bir çıktı olur ve zaten mevcut [14-agentic-mega-atlas.md](14-agentic-mega-atlas.md) çalışmanın doğal devamı.

---

## §1 — Starlarındaki AutoGen ile alakalı repolar

893 starın **isim + açıklama + topic** alanlarında `autogen|ag2|agentchat|magentic` taraması yapıldı. Sonuç dürüstçe şu: **doğrudan alakalı tek repo var.**

| Repo | ⭐ | Starladığın | İlgi |
|---|---:|---|---|
| [microsoft/autogen](https://github.com/microsoft/autogen) | 60.4k | 2026-03-02 | **Doğrudan — projenin konusu** |
| [langfuse/langfuse](https://github.com/langfuse/langfuse) | 33.0k | 2026-04-18 | Dolaylı — AutoGen entegrasyonu var, trace/eval için |
| [AgentOps-AI/agentops](https://github.com/AgentOps-AI/agentops) | 5.8k | 2026-04-18 | Dolaylı — AutoGen'i resmi destekleyen ajan gözlemlenebilirliği |
| [microsoft/markitdown](https://github.com/microsoft/markitdown) | 173.3k | 2026-05-28 | Zayıf — sadece AutoGen ekibinden çıkmış olması |
| [raia-live/amfs](https://github.com/raia-live/amfs) | 59 | 2026-05-03 | Zayıf — açıklamasında geçiyor |

**Starlarında olmayan ama projede kesinlikle lazım olacaklar** (henüz starlamamışsın):
`microsoft/agent-framework` · `ag2ai/ag2` · `microsoft/autogen-vscode` benzeri araçlar.

### Karşılaştırma ekseni için elindeki repolar

AutoGen'i boşlukta değil, rakiplerine karşı konumlandırmak gerekiyor. Bunların hepsi **zaten senin starlarında**:

| Repo | ⭐ | Karşılaştırmada rolü |
|---|---:|---|
| [openai/openai-agents-python](https://github.com/openai/openai-agents-python) | 28.6k | Minimal handoff modeli — AutoGen'in tam zıttı felsefe |
| [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | 57.0k | Rol-oyunu/ekip metaforu — AutoGen GroupChat'e en yakın rakip |
| [microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel) | 28.4k | Birleşmenin diğer yarısı — **birleşme hikâyesi için şart** |
| [agentscope-ai/agentscope](https://github.com/agentscope-ai/agentscope) | 28.9k | Alibaba'nın çok-ajan framework'ü, mesaj-merkezli |
| [OpenBMB/ChatDev](https://github.com/OpenBMB/ChatDev) | 34.0k | Yazılım-şirketi metaforu, akademik köken |
| [google/adk-python](https://github.com/google/adk-python) | 21.1k | Google'ın muadili — üç büyük satıcı karşılaştırması tamamlanır |
| [agno-agi/agno](https://github.com/agno-agi/agno) | 41.7k | Performans-odaklı alternatif |
| [a2aproject/A2A](https://github.com/a2aproject/A2A) | 25.3k | Ajanlar-arası protokol — MAF'ın desteklediği interop katmanı |
| [confident-ai/deepeval](https://github.com/confident-ai/deepeval) | 17.6k | Değerlendirme tarafı |

LangGraph tarafı için `langflow` ve `deer-flow` da starlarında; graf-tabanlı orkestrasyonu temsil ediyorlar.

---

## §2 — `agentic_tuffs.md` taraması

2420 satırın tamamı tarandı. **Net sonuç: AutoGen listende hiç geçmiyor.** Tek eşleşme "autogenesis" (satır 153, 169) ve o tamamen farklı bir şey — kendini geliştiren ajan protokolü.

Ama listende AutoGen projesinin **tez cümlesini besleyecek** satırlar var. Bunlar altın değerinde, çünkü hepsi "çok-ajanlı sistemler neden zor" sorusuna bakıyor:

| Satır | Konu | Projede nerede kullanılır |
|---|---|---|
| 80 | *On the Reliability Limits of LLM-Based Multi-Agent Planning* | **§Neden yetmedi** bölümünün omurgası |
| 92 | *Detecting Multi-Agent Collusion Through Multi-Agent Interpretability* | Hata modu analizi |
| 317 | "Multi-agent debate makes models reason better. It also burns tokens..." | **Maliyet/fayda dengesi** — kritik eleştiri |
| 187 | Walden Yan'ın multi-agent paper'ı (muhtemelen *Don't Build Multi-Agents*) | Karşı-tez; dengeli analiz için şart |
| 429 | "9 New approaches to Multi-Agent Systems" | Landscape taraması |
| 474 | *Four Agent Orchestration Patterns You Should Know About* | Desen sınıflandırması |
| 83 | *Experience as a Compass: Multi-agent RAG with Evolving Orchestration* | İleri seviye orkestrasyon |
| 462 | Conductor: RL ile eğitilmiş 7B orkestratör | Konuşmacı-seçimi probleminin modern hâli |
| 270, 282, 330 | Üretimde çok-stratejili orkestrasyon, ICLR2026 Conductor, recursive MAS | Güncel akademik bağlam |

Özellikle **317 ve 187 numaralı satırlar** projenin en değerli kısmını verir: AutoGen'in konuşma-merkezli modeli token açısından pahalı ve güvenilirlik açısından kırılgan — bakım moduna alınmasının teknik gerekçesi tam olarak burada yatıyor.

---

## §3 — Mevcut atlas'ındaki AutoGen bölümü için düzeltme

[14-agentic-mega-atlas.md](14-agentic-mega-atlas.md) dosyanda `## A.3 AutoGen / AG2` bölümü var (satır 157-190). İçeriği doğru ama **iki farklı framework'ü tek başlıkta birleştiriyor** ve anlattığı API aslında AG2'nin:

- `AutoPattern`, `AgentTarget`, `RevertToUserTarget` → bunlar **AG2** (v0.2 kolu) API'leri
- `microsoft/autogen` v0.4+ ise `RoundRobinGroupChat`, `SelectorGroupChat`, `Swarm`, `Handoff` ve `autogen_core` runtime'ı kullanıyor — tamamen farklı bir API yüzeyi

Projenin ilk somut çıktılarından biri bu bölümü **A.3a AutoGen (v0.2 → v0.4 → bakım modu)** ve **A.3b AG2** olarak ikiye ayırmak, üçüncü olarak da **MAF**'ı eklemek olabilir. Elindeki 8-eksen şablonu bunun için hazır.

---

## §4 — Birincil kaynaklar

Sıralama önem derecesine göre. Blog yazısı yerine bunlardan oku.

### Kod & doküman
| Kaynak | Link | Not |
|---|---|---|
| AutoGen repo | https://github.com/microsoft/autogen | Bakım modu uyarısı README'nin başında |
| AutoGen stable docs | https://microsoft.github.io/autogen/stable/ | v0.4+ API'nin resmi referansı |
| v0.2 → v0.4 migration guide | https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html | **Mimari kırılmayı anlamanın en iyi yolu** |
| AutoGen → MAF migration guide | https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/ | Halefe geçişin resmi haritası |
| Microsoft Agent Framework | https://github.com/microsoft/agent-framework · https://learn.microsoft.com/en-us/agent-framework/overview/ | Projenin "şimdi" ayağı |
| AG2 | https://github.com/ag2ai/ag2 | Fork'un gittiği yön |

### Makaleler
| Makale | arXiv | Neden |
|---|---|---|
| **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** (Wu, Bansal, Zhang, Wang et al., 2023) | [2308.08155](https://arxiv.org/abs/2308.08155) | **Kurucu makale.** Konuşma-merkezli programlama modelinin tezi |
| **Magentic-One: A Generalist Multi-Agent System** (2024) | [2411.04468](https://arxiv.org/abs/2411.04468) | AutoGen üstüne kurulu amiral gemisi sistem; Orchestrator + ledger deseni |
| **AutoGen Studio: No-Code Developer Tool for Multi-Agent Systems** (EMNLP 2024) | [2408.15247](https://arxiv.org/abs/2408.15247) | Araç/DX tarafı |
| **Why Do Multi-Agent LLM Systems Fail?** (Cemri, Pan et al., NeurIPS 2025) | [2503.13657](https://arxiv.org/abs/2503.13657) | **MAST taksonomisi** — 7 framework'ten 1600+ trace, 14 hata modu, 3 küme. Projenin eleştiri bölümü için en güçlü tek kaynak |

MAST'ın bulgusu senin tezini doğrudan destekliyor: hataların **~%42'si sistem tasarımı/spesifikasyon**, **~%37'si ajanlar-arası koordinasyon**, **~%21'i doğrulama eksikliği** kaynaklı — yani model kalitesinden değil, **harness tasarımından**. Bu tam olarak [14-agentic-mega-atlas.md](14-agentic-mega-atlas.md) çalışmanın çıkış noktasıyla aynı.

### Mimari: bilmen gereken üç katman

AutoGen v0.4'ün yeniden yazımı üç paketten oluşuyor — bunu ezberle, her yerde karşına çıkacak:

| Katman | Paket | Ne yapar |
|---|---|---|
| **Core** | `autogen-core` | Aktör modeli, event-driven runtime, asenkron mesajlaşma. Ajanlar birbirinden izole; biri çökerse diğerleri ayakta kalır, süreçlere/makinelere dağıtılabilir |
| **AgentChat** | `autogen-agentchat` | Görev-odaklı yüksek seviye API: `AssistantAgent`, group chat, termination koşulları, streaming, state |
| **Extensions** | `autogen-ext` | Model istemcileri (OpenAI/Azure), kod yürütücüler, üçüncü parti entegrasyonlar |

v0.2'de tek paket ve senkron bir konuşma döngüsü vardı; v0.4'te aktör modeline geçiş bu projenin **en ilginç mühendislik hikâyesi**.

---

## §5 — Önerilen proje çerçevesi

Intern seviyesi için "okudum-özetledim"in üstüne çıkacak, elindeki malzemeyle 2-3 haftada bitecek bir kurgu:

**Başlık:** *AutoGen'den Microsoft Agent Framework'e: Bir çok-ajan framework'ünün mimari evrimi ve üretim dersleri*

1. **Tarihçe** — v0.2 (2023, konuşma-merkezli) → v0.4 (2025, aktör modeli) → bakım modu (2026) → MAF. AG2 çatallanması yan kol olarak
2. **Mimari analiz** — üç katman, GroupChat/speaker-selection mekaniği, `UserProxyAgent`'ın çift rolü (insan-döngü + kod yürütücü). Kendi 8-eksen şablonunla
3. **Uygulamalı** — aynı görevi 3 framework'te çöz: AutoGen v0.4, MAF, ve bir rakip (OpenAI Agents SDK ya da CrewAI). Token/latency/kod satırı/hata oranı ölç. `saf-motorlar/` ve motor karşılaştırma panelin bu iş için zaten hazır altyapı
4. **Eleştiri** — MAST taksonomisiyle hata modlarını sınıflandır, `agentic_tuffs.md` 317/187'deki maliyet-fayda tartışmasını işle
5. **Sonuç** — konsolidasyon neden kaçınılmazdı, çok-ajan hangi durumda gerçekten kazandırıyor

3. adım kritik: ölçüm yapan bir intern raporu, özet çıkaran bir intern raporundan bambaşka bir lige geçer.

---

## §5b — Çalışan POC

§5'in 2. ve 3. adımı için iskelet hazır: **[../autogen/](../autogen/)**

AutoGen v0.7.5 ile yazıldı ve koşuldu. Aynı görevi beş farklı orkestrasyon
deseniyle çözüp hepsini aynı metriklerle ölçüyor (mesaj, LLM çağrısı, tool
çağrısı, token, süre). API anahtarı gerektirmiyor — anahtar yoksa
`ReplayChatCompletionClient` ile deterministik koşuyor.

| desen | mesaj | LLM | token |
|---|---:|---:|---:|
| RoundRobinGroupChat | 9 | 6 | 274 |
| SelectorGroupChat | 8 | 5 | **204** |
| Swarm (handoff) | 14 | 7 | **334** |
| GraphFlow (paralel + join) | 11 | 7 | 270 |
| `autogen_core` aktör modeli | 9 | 0 | 0 |

Aynı görev, aynı ajanlar, **%63.7 token farkı** — ödenen şey yönlendirme özerkliği.

POC'un asıl bulgusu 5. desende: "aktör modeli hata izolasyonu verir" iddiası test
edildi ve **kısmen yanlış** çıktı. Çöken bir handler `asyncio.gather`'ı erken
döndürüyor, hemen ardından `task_done()` çağrıldığı için `stop_when_idle()` kuyruğu
boş sanıyor; kardeş handler'lar bitmeden bariyer açılıyor. Yayından sonra hemen
`close()` çağrılırsa sağlam ajanların sonuçları **sessizce kayboluyor** — ne
exception yükseliyor ne uyarı çıkıyor. Runtime korunuyor, veri korunmuyor.

Bu, MAST'ın *system design* + *task verification* kümesinin laboratuvar
koşullarında yeniden üretilmiş hâli: hatanın kaynağı model değil, harness.

## §6 — İlk adımlar

- [ ] `microsoft/agent-framework` ve `ag2ai/ag2` repolarını starla — projenin diğer iki ayağı
- [ ] AutoGen v0.2→v0.4 migration guide'ı oku (mimari kırılmayı en hızlı burada kavrarsın)
- [ ] MAST makalesini ([2503.13657](https://arxiv.org/abs/2503.13657)) oku, 14 hata modunu çıkar
- [ ] `pip install "autogen-agentchat" "autogen-ext[openai]"` ile hello-world + bir `SelectorGroupChat` örneği çalıştır
- [ ] Aynı görevi MAF ile kur, iki kod tabanını yan yana koy
- [ ] [14-agentic-mega-atlas.md](14-agentic-mega-atlas.md) A.3'ü AutoGen / AG2 / MAF olarak üçe ayır

---

---

## §7 — Ek: mayank953 AutoGen Crash Course taraması

Kaynak: [mayank953/Youtube — Agentic AI/Autogen Crash Course](https://github.com/mayank953/Youtube/tree/main/Agentic%20AI/Autogen%20Crash%20Course)
Sparse-clone ile çekilip tamamı tarandı: **12 modül, 21 kod dosyası** (13 notebook + 8 `.py`), 8.9 MB (büyük kısmı Excalidraw çizimleri). İçerik 2025-07-25'te eklenmiş, son dokunuş 2026-02-07. Repo'da **lisans dosyası yok** — kodu kendi projene kopyalayacaksan bunu not et.

### En önemli tespit: API'si güncel

`requirements.txt` ve tüm import'lar taranınca çıkan sonuç — bu kurs **saf v0.4+ API'si** kullanıyor:

```
autogen-agentchat · autogen-core · autogen-ext · autogenstudio
```

Kullanılan API yüzeyi: `AssistantAgent` (44), `TextMessage` (21), `RoundRobinGroupChat` (16), `Console` (13), `TaskResult` (10), `TextMentionTermination` (9), `SelectorGroupChat` (4), `MaxMessageTermination` (4), `MultiModalMessage`, `UserProxyAgent`, `FunctionTool`, `CancellationToken`.

Yani **v0.2 mirası yok** (`ConversableAgent`, `initiate_chat` hiç geçmiyor) ve **AG2 API'si yok**. Bu iyi haber: kurs, [stable dokümanla](https://microsoft.github.io/autogen/stable/) ve §0'da bakım moduna alındığını tespit ettiğimiz kolla **aynı** API'yi öğretiyor. Yani "eski şeyi öğrenmek" riski yok — tam olarak analiz edeceğin sürümü öğretiyor.

### Modül haritası

| # | Modül | İçerik |
|---|---|---|
| 1 | Introduction & Installation | Python `asyncio` temelleri (kahve-bagel analojisi), kurulum |
| 2 | First Autogen Agent | İlk `AssistantAgent` |
| 3 | Autogen Architecture | **Sadece teori** (0 kod hücresi) — Core/AgentChat/Extensions üç katmanı, Magentic-One, Bench, Studio |
| 4 | Agents in Autogen | `on_message()`, `on_messages_stream()`, streaming |
| 5 | Configuring Models | OpenAI, Gemini, **OpenRouter (ücretsiz key)**, Ollama |
| 6 | Multimodal & Structured Output | `MultiModalMessage`, `AGImage`, Pydantic ile yapılandırılmış çıktı |
| 7 | Teams | Tek-ajan vs çok-ajan karşılaştırması, `RoundRobinGroupChat`, **`SelectorGroupChat` + custom selector function** |
| 8 | Termination Condition | `TextMentionTermination`, `MaxMessageTermination` |
| 9 | Human in Loop | `UserProxyAgent`, input fonksiyonu, run-sonrası insan müdahalesi |
| 10 | Tools | Custom `FunctionTool`, yerleşik tool'lar, üçüncü parti (`autogen_ext.tools.http`) |
| 12 | Projects | **arXiv literatür-tarama ajanı** — `search_agent` + `summarizer`, `RoundRobinGroupChat`, Streamlit arayüzü |

*(11 numaralı modül repo'da yok — atlanmış.)*

Ayrıca **4 büyük Excalidraw dosyası** var (toplam ~9 MB): "ExcaliDraw Class Notes", "Data Analyst Automation", "Industry Ready project". Mimari diyagramlar için işine yarayabilir.

### Kapsamadığı şeyler — projen açısından asıl önemli kısım

Bu bir **AgentChat katmanı kursu**. Taramada şunların **hiç geçmediğini** doğruladım:

| Eksik | Neden senin için önemli |
|---|---|
| `Swarm` / `Handoff` | v0.4'ün handoff-tabanlı takım modeli — OpenAI Agents SDK ile karşılaştırmanın tam kalbi |
| `DockerCommandLineCodeExecutor` / kod yürütme | AutoGen'in **kurucu özelliklerinden biri**; kurs bunu hiç göstermiyor |
| `autogen_core` ile runtime seviyesi programlama | `CancellationToken` ve `Image` dışında kullanılmıyor. **Aktör modeli, event-driven runtime, dağıtık gRPC** — yani v0.4'ün asıl mühendislik hikâyesi tamamen dışarıda |
| `Memory` / `ListMemory` / vektör bellek | Bellek katmanı yok |
| `McpWorkbench` / MCP entegrasyonu | Modern interop katmanı |
| State kaydetme/yükleme, serialization | Üretim konusu |
| Magentic-One | Sadece 3. modülde teorik olarak anılıyor, uygulama yok |
| AutoGen Studio | `requirements.txt`'te var ama dersi yok |
| Değerlendirme / AutoGenBench | Yok |

### Değerlendirme

**Ne işe yarar:** §5'teki proje çerçevesinin **2. adımı (mimari analiz)** ve **3. adımı (uygulamalı karşılaştırma)** için hızlı bir rampa. AgentChat seviyesinde çalışan kod elde etmen ~1-2 gün sürer; 12. modüldeki arXiv literatür-tarama ajanı da senin araştırma projene tematik olarak zaten yakın — doğrudan başlangıç iskeleti olarak kullanılabilir.

**Ne işe yaramaz:** Bir intern **araştırma** projesinin derinliğine ulaşmıyor. `autogen_core`'un aktör modeli, kod yürütme sandbox'ı ve dağıtık runtime — yani "AutoGen neden v0.4'te baştan yazıldı ve neden yine de yetmedi" sorusunun cevabı — bu kursta yok. O kısım için birincil kaynak §4'teki [migration guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html) ve `autogen-core` dokümanı olmalı.

**Pratik sıra:** kursu 2, 4, 7, 9, 10, 12. modüller üzerinden hızlı geç (3. modülü atla, teorisi §4'teki kaynaklarda daha iyi) → sonra doğrudan `autogen_core` dokümanına ve MAST makalesine geç.

Klasör şu an scratchpad'de duruyor. Repo kökündeki `autogen/` dizinin boş — istersen oraya kopyalayabilirim (Excalidraw dosyaları hariç ~200 KB, dahil 8.9 MB).

---

*Bu dosyadaki repo istatistikleri ve README alıntısı 2026-08-12'de GitHub API'sinden doğrudan çekildi. Star taraması [../docs/github-starred-repos.md](../docs/github-starred-repos.md) ile aynı veri setine dayanıyor.*
