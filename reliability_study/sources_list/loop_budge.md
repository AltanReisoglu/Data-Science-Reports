# Agent Reliability — Loop Detection & Task Budget Kontrolleri: Kaynak Envanteri

Tarama tarihi: 2026-08-20 · Kullanılan MCP'ler: alphaxiv, deepwiki, grep(grep.app), WebSearch

**Kapsam:** yalnızca loop detection ve per-task budget enforcement (max steps / replans /
tokens / süre). Konu dışı kalan 33 kaynak §8'de gerekçesiyle listelendi.
Harness'ların kod düzeyinde karşılaştırması ayrı dosyada: `harness_kontrolleri.md`.

**Doğrulama etiketleri** — bu listedeki her satır aynı güvende değil:
`[K]` kaynak kodu bizzat okundu, sayılar dosyadan alındı ·
`[T]` makalenin tam metni/raporu okundu ·
`[Ö]` sadece abstract/özet düzeyinde, içerik doğrulanmadı.

---

## 0. İki eksen, tek problem


Brief iki ayrı kontrol istiyor ama literatür bunları tek bir yapısal hataya bağlıyor:
**bir geri besleme yolu (feedback path) maliyetli bir işlemi tekrar tetikliyor ve bu yolu
fiilen kapsayan bir sınır yok.** Loop detection = belirtiyi çalışırken yakalamak;
budget enforcement = belirti yakalanamasa bile zararı tavanla kesmek. İkisi
birbirinin yedeği, alternatifi değil.

---

## 1. Çerçeve makalesi — Towards a Science of AI Agent Reliability `[T]`

*(Bu çalışmanın referans çerçevesi. Kullanıcı tarafından verildi.)*

https://www.alphaxiv.org/abs/2602.16666 · ICML 2026 · Rabanser, Kapoor, Kirgis, Liu, Utpala, Narayanan · **75 oy / 1130 görüntüleme**

Bu çalışmanın çerçeve kaynağı. Güvenilirliği tek metrikten çıkarıp **4 boyut / 12 ölçülebilir metriğe**
ayırıyor; boyutlar havacılık, nükleer ve otomotiv güvenlik mühendisliğinden türetilmiş:

| Boyut | Metrikler | Bizim konuya bağı |
|---|---|---|
| **Tutarlılık** (R_Con) | `C_out` sonuç, `C_d_traj` eylem dağılımı, `C_s_traj` eylem sırası (Levenshtein), **`C_res` kaynak** | `C_res` doğrudan bütçe kalemimiz |
| **Dayanıklılık** (R_Rob) | `R_fault` altyapı hatası, `R_env` ortam değişimi, `R_prompt` yeniden ifade | fault injection PoC senaryosu olarak kullanılabilir |
| **Öngörülebilirlik** (R_Pred) | `P_cal` kalibrasyon, `P_AUROC` ayrım gücü, `P_brier` | agent kendi başarısızlığını haber verebiliyor mu |
| **Güvenlik** (R_Saf) | `S_comp` uyum, `S_harm` zarar şiddeti | **kasten genel ortalamanın dışında** — kuyruk riski ortalamada kaybolmasın diye |

**Kaynak tutarlılığı (C_res) tanımı ve bulgusu — brief'in "öngörülemez tüketim" cümlesinin ölçülmüş hali:**
`C_res = exp(−ortalama varyasyon katsayısı)`, sadece **başarılı** koşumlar üzerinden hesaplanıyor
(maliyet varyansı başarısızlıkla karışmasın diye). Makalenin kendi örneği:

> "Bir veri analizi agent'ı aynı istek üzerinde bir koşumda 1.000 token ve 3 araç çağrısı,
> bir başkasında 50.000 token ve 47 araç çağrısı kullanabilir."

Yani aynı girdide **50× maliyet salınımı**. Makalenin yorumu doğrudan bizim gerekçemiz:
düşük `C_res`'li bir agent "teknik olarak yetenekli ama operasyonel olarak kullanılamaz" —
finansal planlama imkânsızlaşıyor, üretimde rate limit ve bütçe alarmlarını tetikliyor.

Ölçülen değerler: GAIA'da `C_res` 0,54–0,77 · τ-bench'te 0,76–0,86. Açık uçlu ortamda
kaynak öngörülebilirliği belirgin şekilde daha kötü. Ayrıca **zor görevlerde `C_res` çoğu
modelde daha da bozuluyor**; Gemini ve Claude modelleri zor görevlerde eylem sayısını
ciddi biçimde artıran bir **"daha çok uğraş" (try harder) stratejisi** benimsiyor — yani
maliyet tam da en riskli anda öngörülemez hale geliyor.

Diğer ana bulgular:
- **24 aylık model gelişimine rağmen güvenilirlik neredeyse yerinde sayıyor.** GAIA'da eğim
  0,03/yıl, τ-bench'te 0,09/yıl. Doğruluk ile korelasyon GAIA'da r=0,46 (eğim 0,03), τ-bench'te r=0,86.
- **Doğruluk ve güvenilirlik ayrışıyor**; üç sağlayıcının frontier modelleri benzer kümeleniyor —
  yani bu bir satıcı sorunu değil, sektör genelinde bir plato.
- Sonuç tutarlılığı düşük: görevi çözebilen agent onu **tutarlı biçimde** çözemiyor (GAIA 0,58–0,84).
- **"Ne, ama ne zaman değil"**: dağılımsal tutarlılık yüksek, sıra tutarlılığı düşük.
  Agent hangi araçları kullanacağını biliyor, hangi sırayla kullanacağını bilmiyor.
- Tutarlılık model büyüklüğüyle **ters** gidebiliyor — küçük modeller sıklıkla daha tutarlı;
  büyük modellerin birden çok çözüm yolu koşumlar arası varyansı artırıyor.
- Gerçek olaylar: Replit asistanının üretim veritabanını silmesi, OpenAI Operator'ün onaysız
  satın alma yapması, NYC belediye chatbot'unun yasa dışı iş tavsiyesi vermesi. Makale Ek G.1'de
  her birinin **hangi metrikle önceden yakalanabileceğini** tablolaştırıyor — Atlas'ın dağıtım
  öncesi eşik listesi için hazır şablon.
- Anthropic'in 80.000 kişilik çalışmasında **güvenilmezlik, yapay zekâya dair en sık dile
  getirilen kaygı** olarak çıkmış.

## 2. Çekirdek akademik kaynaklar — Loop Detection


| # | Kaynak | Neden önemli | Dur. |
|---|---|---|---|
| 1 | **When Agents Do Not Stop: Uncovering Infinite Agentic Loops in LLM Agents** — https://www.alphaxiv.org/abs/2607.01641 (Huazhong Univ. of Sci. & Tech., 2026-07) | **Bu işin omurgası.** IAL-SCAN adlı statik analiz aracı; 6.549 Python agent reposu (246.748 dosya) tarandı, 74 bulgu, elle doğrulanan 68 gerçek IAL / 47 projede, uçtan uca **%91,9 kesinlik**. Loop'u "agentic feedback path + maliyetli/state-büyüten işlem + etkin sınır yokluğu" üçlüsüyle tanımlıyor. | `[T]` |
| 2 | **Real-Time Detection and Repair of LLM Agent Failures** — https://www.alphaxiv.org/abs/2608.02464 (2026-08) | Tam bizim sorumuz: her adımı ikinci bir LLM'e yargılatmak pahalı — bunun yerine ucuz gerçek-zamanlı dedektörler. Döngü, araç hatası zinciri, hedeften sapma birlikte. | `[Ö]` |
| 3 | **LoopTrap: Termination Poisoning Attacks on LLM Agents** — https://www.alphaxiv.org/abs/2605.05846 (2026-05, 9 oy) | Sonlanma kararının kendisi saldırı yüzeyi: agent'ın "iş bitti" yargısını zehirleyip sonsuz döngüye sokma. Tehdit modeli tarafı. | `[Ö]` |
| 4 | **From Shield to Target: Denial-of-Service Attacks on LLM-Based Agent Guardrails** — https://www.alphaxiv.org/abs/2606.14517 | **Karşı-argüman kaynağı:** LLM tabanlı guardrail'in kendisi DoS hedefi olabiliyor. Sunumun "dezavantajlar" bölümü için. | `[Ö]` |
| 5 | **Model or Harness? An Interaction-Centric Taxonomy for Localizing Agent Failures** — https://www.alphaxiv.org/abs/2607.28802 (**52 oy / 565 görüntüleme**) | Hata modele mi harness'a mı ait? Döngü modelin mi harness'ın mı kusuru sorusunun çerçevesi — harness karşılaştırmasının teorik zemini. | `[Ö]` |
| 6 | **Cognitive Fatigue in Autoregressive Transformers: Formalization and Measurement** — https://www.alphaxiv.org/abs/2605.30981 | **Döngünün model düzeyindeki kökeni:** uzun üretimde tekrarlayan metin, talimat uyumunun kaybı, kararsız entropi. Neden döngüye giriyor sorusunun cevabı. | `[Ö]` |
| 7 | **Agentic Abstention: Do Agents Know When to Stop Instead of Act?** — https://www.alphaxiv.org/abs/2606.28733 (**29 oy / 306 görüntüleme**) | Agent kendi kendine durabiliyor mu — harici dedektöre ne kadar mecbur olduğumuzun ölçüsü. | `[Ö]` |

### IAL-SCAN'den çıkan sayılar (doğrudan kullanılabilir)

Bulunan 68 gerçek döngünün desen dağılımı:
- Sınırsız **retry** geri beslemesi — 17 (%25,0)
- Sınırsız **tool-call iterasyonu** — 16 (%23,5)
- Tur sınırı olmayan **çok-agent sohbeti** — 14 (%20,6)

Etkileri (çakışabiliyor):
- API maliyet tükenmesi — 65 (%95,6) · Model DoS — 65 (%95,6)
- State büyümesinden context penceresi tükenmesi — 19 (%27,9)
- Harici araç rate-limit tükenmesi — 5 (%7,4)

Kök neden: **68'inin 68'i de** tekrar eden yolu kapsayan güçlü bir sınırın yokluğuna
iniyor. Framework dağılımında LangGraph + AutoGen tek başına 45 bulgu (%66,2) —
çünkü bu ikisi geri beslemeyi açık `while` yerine API semantiğiyle kuruyor,
yani döngü kodda göze görünmüyor.

Ayrıca ölçülmüş bir uyarı: aynı işi genel amaçlı LLM'e yaptırmak çalışmıyor.
Saf LLM API taraması 68 hatanın sadece 23'ünü, kodlama agent'ı 50'sini yakalamış;
buna karşılık alarm sayıları 183 ve 140. IAL-SCAN 68/68'i 6 yanlış pozitifle,
proje başına 4,2K token ve 31,2 saniyede. **Tespit için LLM yargıcı, ucuz
deterministik kontrolün yerine geçmiyor.**

---

## 3. Çekirdek akademik kaynaklar — Budget Enforcement


| # | Kaynak | Neden önemli | Dur. |
|---|---|---|---|
| 1 | **Token Budgets: An Empirical Catalog of 63 LLM-Agent Budget-Overrun Incidents** — https://www.alphaxiv.org/abs/2606.04056 (2026-06) | **"Neden" bölümünün kanıt tabanı.** 21 orkestrasyon framework'ünden 2023–2026 arası **63 doğrulanmış üretim olayı**, her biri alıntılanmış GitHub issue + maintainer/kullanıcı beyanı + (varsa) dolar zararı ile. 8 kümeli hata taksonomisi, Cohen's κ = 0,837. | `[T]` (kısmi) |
| 2 | **BAGEN: Are LLM Agents Budget-Aware?** — https://www.alphaxiv.org/abs/2606.00198 (10 oy) | Bütçeyi *ölçülen* değil *kontrol sinyali* olarak ele alma tezi. "Maliyeti koşumdan sonra raporlamak yetmez" argümanının kaynağı. | `[Ö]` |
| 3 | **When Replanning Becomes the Bottleneck: Budgeted Replanning** — https://www.alphaxiv.org/abs/2608.01428 | Brief'teki **"max replans"** kalemini birebir karşılayan tek makale. Her replan çağrısının biriken bağlamı taşıması. | `[Ö]` |
| 4 | **Inference-Time Budget Control for LLM Search Agents** — https://www.alphaxiv.org/abs/2605.05701 (10 oy) | **Çift bütçe** (araç çağrısı + üretilen token) altında davranış. Tek sayaç yerine iki eksenli tavan. | `[Ö]` |
| 5 | **R³-Bench: LLMs Struggle with Resource-Rational Reasoning under Shared Budgets** — https://www.alphaxiv.org/abs/2608.16033 | **Paylaşılan** bütçe altında davranış. Çoğu çalışma görev başına bağımsız bütçe varsayıyor; üretimde bütçe paylaşılıyor. | `[Ö]` |
| 6 | **Grounded Scaling: Why Agentic AI Needs Deterministic Environments** — https://www.alphaxiv.org/abs/2606.22495 | **Adım bütçesinin matematiksel gerekçesi:** adım başına determinizm δ < 1 iken k adımlık zincirin başarısı δ^k ile bozuluyor. Uzayan koşum sadece pahalı değil, üstel olarak daha başarısız. | `[Ö]` |
| 7 | **OpenCodeReview: Determinism over Non-Determinism for Cost-Effective Agent-Based Code Review** — https://www.alphaxiv.org/abs/2608.09290 | **Sınırsız araç kullanımının** sonuçları kararsız yaptığı tezi + maliyet-etkin alternatif. Limitin sadece maliyet değil *tekrarlanabilirlik* aracı olduğu argümanı. | `[Ö]` |
| 8 | **RCWT: Measuring Task-Budget Displacement from Coordination Content** — https://www.alphaxiv.org/abs/2607.12216 | Çok-agent'ta koordinasyon metninin görev bütçesini yemesi. Ölçüm metriği. | `[Ö]` |
| 9 | **A Stackelberg Framework for Resource-Aware LLM Agents** — https://www.alphaxiv.org/abs/2606.23026 | Statik eşiklerin yetersizliği; uyarlanabilir bütçe tahsisi. Teorik, uygulama uzak. | `[Ö]` |
| 10 | **Green SARC: Predictive Cost and Carbon Governance for Agentic AI** — https://www.alphaxiv.org/abs/2606.15954 | **Kontroller dashboard'da değil yürütme yolunda olmalı** tezi — GPU kapasitesi kaygısının doğrudan karşılığı. | `[Ö]` |

### Token Budgets kataloğundan çıkan somut olaylar (sunumda doğrudan kullanılabilir)

- **"Katalogda, bir kullanıcı parasını ödemeden önce engellenmiş tek bir bütçe aşımı vakası bulamadık."** Düzeltmeler hızlı geliyor (aynı gün / 1–2 gün) ama ancak hata patladıktan sonra.
- **Claude Code (CCDE-001):** tek kullanıcı, 4 günde **235 $** — compaction-loop imzası. CCDE-002'nin aktivite kaydı aynı imzayla açılmış en az 10 kardeş issue'ya atıf veriyor.
- **Pydantic AI (PYAI-001):** çok-agent dokümantasyon örneği, **kendi `total_tokens_limit` varsayılanında** patlıyor.
- **OpenAI Agents SDK (OAAS-002):** `max_turns` aşıldığında zarif bozulma için maintainer itirafı — *"We don't have anything amazing here right now."*
- **smolagents (SMAG-002/006):** 13+ aydır açık; maintainer'ın gerekçesi *"adımları kırpmamaya karar verdik çünkü sessiz hatalar doğurur."* → Kırpma ile sessiz hata arasındaki gerçek gerilim.
- **MAST-014:** kataloğun en büyük tek çağrı şişmesi — **tek bir gözlemci LLM çağrısında 2 milyon token**; yani gözlemlenebilirlik katmanının kendisi maliyet amplifikatörüne dönüşüyor.
- **MAST-004:** `TokenLimiter` döngü iterasyonu başına tetiklenmiyor — *sınır var ama yanlış yerde* örneği.

### Üç katmanlı zorlama taksonomisi (bu makaleden, Atlas tasarımı için doğrudan iskelet)

| Katman | Ne yapar | Ne zaman yakalar | Örnek |
|---|---|---|---|
| **Derleme zamanı** | Tip sistemi bütçe değerinin kopyalanmasını / iki kez harcanmasını / devredildikten sonra kullanılmasını derleme hatası yapar | Dağıtımdan önce | makalenin `token-budgets` Rust crate'i (1.180 satır, affine ownership) |
| **Yazılım (runtime middleware)** | Harcamayı izler, eşik aşılınca duraklatır | Harcama olduktan sonra | AgentGuard tarzı bütçe callback'i, LiteLLM proxy |
| **Transport** | Cüzdan bittiğinde ağ sınırında **HTTP 402** döner | İstek çıktıktan sonra | ATXP |

Bunlar rakip değil tamamlayıcı; yüksek riskli üretimde üçü birden konuşlandırılabilir.
Not: makale kendi affine mekanizmasının varsayılan tahmincisinin **6,20× ortalama /
2,51× medyan aşırı rezervasyon** yaptığını açıkça raporluyor (N=5.190 çağrı) — yani
ön-rezervasyon yaklaşımının işletme sermayesi maliyeti var. Dürüst bir dezavantaj kaydı.

---

## 4. Standart / uyum tarafı


- **OWASP LLM10:2025 — Unbounded Consumption** · https://genai.owasp.org/llmrisk/llm102025-unbounded-consumption/
  *(Not: sayfa otomatik erişime 403 veriyor; içerik ikincil kaynaklardan derlendi, sunum öncesi elle açıp doğrula.)*
  Kontrolsüz çıkarımın DoS, ekonomik kayıp ve servis bozulmasına yol açması. **"Denial of Wallet"** terimi buradan.
  Önerilen azaltmalar arasında kullanıcı/uygulama başına rate limit, bütçe tavanı, max-token zorlaması,
  **açıkça "agent framework'lerinde loop detection"**, maliyet panosu + alarm, pahalı sorgular için
  insan onaylı yükseltme, kuyruğa alınan ve toplam eylem sayısına sınır, ağır yük altında zarif bozulma.
  → İki kalemimizi tek bir tanınmış standart maddesine bağlayan çerçeve. Atlas gereklilikleri bölümünün dayanağı.
  Kritik nüans: **rate limit tek başına yetmez** — istek sıklığını sınırlar ama tek bir derin iç içe araç
  döngüsünün kısa bir patlamada devasa kaynak tüketmesini engellemez.

---

## 5. Referans implementasyonlar — kim neyi gerçekten kod olarak yapıyor


Aşağıdaki tüm sayılar `main` dalındaki kaynaktan okundu.

### 4a. Loop / stuck detection

| Proje | Mekanizma | Eşikler | Tetiklenince | Dur. |
|---|---|---|---|---|
| **OpenHands** — `openhands-sdk/openhands/sdk/conversation/stuck_detector.py` (repo: `OpenHands/software-agent-sdk`) | `StuckDetector`, **5 senaryo**: ① aynı eylem+aynı gözlem ② aynı eylem+hata ③ monolog (kullanıcı girdisi olmadan tekrarlayan agent mesajları) ④ dönüşümlü A-B-A-B deseni ⑤ context-window hata döngüsü | `action_observation=4`, `action_error=3`, `monologue=3`, `alternating_pattern=6`; tarama penceresi `MAX_EVENTS_TO_SCAN=20`; pencere **son kullanıcı mesajından itibaren** | Konuşma `ConversationExecutionStatus.STUCK` durumuna geçiyor — hata değil, **birinci sınıf terminal durum**. `action_error` için önce bir "nudge" mesajı enjekte ediliyor. | `[K]` |
| **Cline** — `sdk/packages/core/src/runtime/safety/loop-detection.ts` + `mistake-tracker.ts` | `LoopDetectionTracker`: araç adı + girdinin kanonik serileştirmesinden **imza** üretip ardışık aynı çağrıyı sayıyor. Ayrı `MistakeTracker`: API turu hatası, geçersiz/eksik tool argümanı, tüm araç çağrılarının başarısız olduğu iterasyon | `softThreshold=3`, `hardThreshold=5`, `maxConsecutiveMistakes=6` | **İki kademeli:** soft → konuşmaya "farklı bir yol dene" notu enjekte; hard → mistake-limit karar yoluna `forceAtLimit` ile düşüyor, CLI kullanıcıya "farklı yaklaşım dene / bu koşumu durdur" soruyor, auto-approve açıksa doğrudan durduruyor | `[K]`* |
| **Google Cloud PS** — `tools/gemini-computer-use-eval/.../middleware/stalemate_detection.py` | Computer-use için **state hash'i**: eylem öncesi ekran/durum hash'i alınıp sonrasıyla karşılaştırılıyor; UI değişmediyse `stagnation_counter++`, değiştiyse sıfırlanıyor. Ayrı `failure_counters`, strict/loose çift eşik (`loose_threshold=5`), `_loop_injected` bayrağıyla tek seferlik refleksiyon enjeksiyonu | dosyada | Refleksiyon motoru devreye giriyor | `[K]` |
| **ByteDance deer-flow** — `backend/packages/harness/deerflow/runtime/goal.py` | **Anlamsal ilerleme bütçesi:** adım saymak yerine hedef değerlendiriciyle "ilerleme oldu mu" sorusu. `should_continue_goal()` + `_stand_down_reason()` | `max_continuations=8`, `max_no_progress_continuations=2` | Devam etmeyi kesip gerekçe döndürüyor | `[K]` |

\* Cline satırı DeepWiki'nin repo indeksinden geldi; dosya yolları ve sabitler oradan alıntı, ben dosyayı bizzat açmadım — PoC'den önce doğrula.

### 4b. Bütçe / adım limitleri

| Framework | Parametre | Varsayılan | Aşılınca | Nerede |
|---|---|---|---|---|
| **LangGraph / LangChain** | `recursion_limit` | **25** (`DEFAULT_RECURSION_LIMIT`, `langchain_core/runnables/config.py:171`) | `GraphRecursionError` (`ErrorCode.GRAPH_RECURSION_LIMIT`) | `libs/langgraph/langgraph/pregel/main.py`, `loop.status == "out_of_steps"` |
| **OpenHands** | `max_iteration_per_run` / `max_iterations` | **500** | koşum sonlanır | `conversation/state.py`, `impl/local_conversation.py` |
| | `max_budget_per_run` (USD) | `None` (kapalı) | sert maliyet tavanı | aynı dosya, satır ~466 |
| | `stuck_detection` | **`True`** (varsayılan açık) | — | — |
| **smolagents** | `max_steps` | **20** | `state="max_steps_error"`; ama önce `_handle_max_steps_reached()` modelden bir **nihai cevap** istiyor → sert çökme değil zarif bozulma | `src/smolagents/agents.py` |
| **CrewAI** | `Agent.max_iter` | **25** | — | — |
| | `PlanningConfig.max_steps` / `max_step_iterations` | **20** / **15** | — | `lib/crewai/src/crewai/agent/planning_config.py` |
| | `step_timeout`, `max_execution_time`, `max_rpm` | hepsi `None` | adım "failed" işaretlenip devam/replan kararı | — |
| | `respect_context_window` | `True` → otomatik özetleme; `False` → hata ile dur | — | — |
| **pydantic-ai** | `UsageLimits.request_limit` | **50** | `UsageLimitExceeded` | `pydantic_ai/usage.py` → `_agent_graph.py:ModelRequestNode._prepare_request` |
| | `tool_calls_limit`, `input/output/total_tokens_limit`, `per_request_input_tokens_limit` | `None` | `UsageLimitExceeded` | `check_before_request` / `check_tokens` |
| **OpenAI Agents SDK** | `max_turns` | belirtilmemiş | `MaxTurnsExceeded` | `src/agents/run_internal/run_loop.py` |
| **AutoGen** | `MaxMessageTermination(max_messages, include_agent_event=False)` | — | koşulu birleştirmek için `&` / `\|` operatörleri | `autogen_agentchat/conditions/_terminations.py` |
| | `TokenUsageTermination(max_total_token, max_prompt_token, max_completion_token)` | hepsi `None`, en az biri zorunlu | | aynı dosya |
| | `TimeoutTermination(timeout_seconds)` | — | | aynı dosya |
| | ayrıca `StopMessageTermination`, `TextMentionTermination`, `HandoffTermination`, `SourceMatchTermination`, `FunctionalTermination` (özel mantık) + `BaseGroupChatManager._max_turns` | | | |
| **Google ADK** | `LoopAgent.max_iterations` | **`None` = sınırsız** ⚠️ | alt-agent escalate edene kadar döner | `src/google/adk/agents/loop_agent.py` |
| | `RunConfig.max_llm_calls` | — | `LlmCallsLimitExceededError` | `agents/invocation_context.py` → `_InvocationCostManager.increment_and_enforce_llm_calls_limit` |
| **Claude Agent SDK** | `ClaudeAgentOptions.max_turns` | `None` | sorgu durur | `src/claude_agent_sdk/types.py` |
| | `max_budget_usd` | `None` | bütçe aşılınca sorgu durur | aynı dosya |

Bu tablonun kendisi bir bulgu: **varsayılanların çoğu ya kapalı ya da tek eksenli.**
ADK'nın `LoopAgent`'ı varsayılanda sınırsız; OpenHands hariç hiçbirinde bütçe
varsayılan olarak açık değil; hiçbirinde adım/token/süre/maliyet aynı anda
zorlanmıyor. IAL-SCAN'in "framework'ler varsayılan sınırı geri beslemenin
yaratıldığı runtime kapsamında zorlamalı" önerisi tam da buraya bakıyor.

### 4c. Gateway katmanı — Atlas entegrasyonu için en yakın hazır çözüm

**LiteLLM proxy** (`BerriAI/litellm`, `litellm/proxy/auth/auth_checks.py` → `common_checks()`):

- Kapsamlar: **key, user, team, team-member, organization, tag, end-user** — yedi ayrı seviyede bütçe
- `max_budget` (sert, engeller) · `soft_budget` (engellemez, Slack alarmı) · `budget_duration` (`"30s"/"30m"/"30h"/"30d"` ile otomatik sıfırlama)
- `tpm_limit`, `rpm_limit`, `max_parallel_requests`
- Aşılınca `litellm.BudgetExceededError`, **HTTP 429**, `rate_limit_type = RateLimitType.BUDGET` — yani sağlayıcı kaynaklı rate limit ile kendi bütçe limiti ayırt edilebiliyor
- Referans dosya: repodaki `quota_management.yaml` tüm kota özelliklerini tek yerde listeliyor

Bu, Token Budgets makalesindeki **"yazılım katmanı"**nın olgun bir örneği. Atlas
zaten bir LLM gateway'i kullanıyorsa loop detection'ı agent runtime'ında,
bütçeyi gateway'de zorlamak iki katmanı doğal olarak ayırıyor.

---

## 6. PoC için doğrudan kopyalanabilir referanslar


Brief "döngüye giren / limit aşan senaryolar" istiyor. Sıfırdan tasarlamaya gerek yok:

1. **Tespit mantığı iskeleti** → OpenHands `stuck_detector.py` (370 satır, MIT).
   Beş senaryonun tamamı saf Python, LLM çağrısı yok, olay listesi üzerinde çalışıyor.
   `_event_eq()` metodu **ID'leri (tool_call_id, action_id, llm_response_id) yok sayarak
   içerik karşılaştırması** yapıyor — naif eşitlik kontrolünün neden çalışmadığının cevabı burada.
2. **Aşama-tabanlı müdahale** → Cline'ın soft/hard eşiği: önce prompt'a nudge, sonra durdur.
   Tek eşikli tasarımdan belirgin şekilde iyi, çünkü agent bazen uyarıyla kendi kendine çıkıyor.
3. **İlerleme temelli bütçe** → deer-flow `should_continue_goal`. Adım sayısı yerine
   "son N turda ilerleme oldu mu" — farklı ama tekrarlayan eylemleri yakalar, imza
   karşılaştırması yakalayamaz.
4. **Zarif bozulma** → smolagents `_handle_max_steps_reached`. Limit dolunca istisna
   fırlatmak yerine modelden elindeki bilgiyle nihai cevabı isteme. OAAS-002'deki
   "bu konuda elimizde iyi bir şey yok" itirafının cevabı.
5. **Durum hash'i ile durgunluk** → Google Cloud `stalemate_detection.py` (Apache-2.0),
   araç çıktısı/ekran değişmiyorsa sayaç artırma.
6. **Durdurmanın alternatifi: geri sarma** → AgentRewind
   (https://www.alphaxiv.org/abs/2608.14380, 19 oy) `[Ö]`. Limit dolunca koşumu bitirmek
   yerine hatanın girdiği noktaya dönmek. PoC'de "limit dolunca ne yapmalı" sorusunun
   dördüncü seçeneği — sert durdurma / nudge / zarif bozulma yanına.

Demo senaryoları olarak IAL-SCAN'in üç baskın desenini kullanmak listeyi kanıta bağlar:
sınırsız retry (%25) · sınırsız tool-call iterasyonu (%23,5) · tur sınırsız çok-agent sohbeti (%20,6).
IAL-SCAN makalesinde ikisinin gerçek kod vakası var: `2456868764/LiteRAG` (iç içe
`while not success` → `llm.invoke`) ve `NVIDIA-AI-Blueprints/ai-virtual-assistant`
(`while True` + boş/bozuk çıktıda mesaj geçmişine düzeltici prompt ekleme).

---

## 7. Çerçeveyi kuran diğer üç makale (tam metin okundu)

Çerçeve makalesinin öncülleri ve çok-agent tarafındaki karşılığı.

### 7.1 AI Agents That Matter `[T]`
https://www.alphaxiv.org/abs/2407.01502 · Kapoor, Stroebl, Siegel, Nadgir, Narayanan · 2024-07

2602.16666'nın öncülü ve **bütçe argümanının kökeni**. Tezi: agent değerlendirmeleri
maliyet kontrollü olmak zorunda, çünkü doğruluk bilimsel olarak anlamsız yöntemlerle
(sadece tekrar deneyerek) yükseltilebiliyor.

HumanEval'de üç basit taban çizgisi (Retry / Warming / Escalation) SOTA agent mimarilerini
geçiyor — ölçülmüş tablo:

| Ajan | Doğruluk | Toplam maliyet (164 görev) |
|---|---|---|
| LATS (GPT-4) | 88,0 | **134,50 $** |
| LDB (GPT-4) | 93,3 | 6,36 $ |
| Reflexion (GPT-4) | 87,8 | 3,90 $ |
| GPT-4 (taban, agent yok) | 89,6 | 1,93 $ |
| **Warming (GPT-4)** | **93,2** | **2,45 $** |
| **Retry (GPT-4)** | 92,0 | 2,51 $ |
| **Escalation** (ucuzdan pahalıya tırmanma) | 85,0 | **0,27 $** |

Warming, LATS'ten **~55× ucuza** eşit ya da daha iyi doğruluk veriyor. Yazarların sonucu:
"System 2" yaklaşımlarının (planlama, refleksiyon, hata ayıklama) doğruluk kazancından
sorumlu olduğu iddiası, basit taban çizgileriyle karşılaştırılmadığı için **kanıtlanmamış**.

Bizim konumuz için üç somut nokta:
1. **Yayınlanmış bir SOTA agent'ta gerçek bir sonsuz döngü var.** Robustness kontrollerinde
   LATS (GPT-3.5) HumanEval/83 görevinde **5 saatten uzun süre durmadı**; görev analizden
   çıkarılmak zorunda kalındı. Bu, "IAL laboratuvar dışında da oluyor" iddiasının en temiz kanıtı.
2. **Koşum başına dolar tavanı zaten yerleşik bir pratik:** SWE-Agent yazarları her koşumu
   **4 $** ile sınırlamış. Bütçe zorlamasının egzotik değil standart olduğunu gösteriyor.
   (Aynı hesapla SWE-bench'in tamamı tek koşumda 8.000 $ üzeri.)
3. **Değerlendirme maliyeti o kadar yüksek ki hata payı hesaplanamıyor** — bu yüzden agent
   sonuçları neredeyse hiç error bar'la yayımlanmıyor; yazarlar raporlanan skorların kendi
   5 koşumluk maksimumlarının üstünde olduğu vakalar bulmuş.

Ayrıca WebArena vakası: benchmark'taki Reddit klonunun **rate limit'i** görev sırasına bağlı
başarısızlık yaratıyor — harici araç limitlerinin sonuçları sessizce bozması, IAL-SCAN'in
"harici araç rate-limit tükenmesi" etkisiyle aynı madalyonun diğer yüzü.


### 7.2 Holistic Agent Leaderboard (HAL) `[T]`
https://www.alphaxiv.org/abs/2510.11977 · Kapoor, Stroebl ve 30+ yazar (Princeton, Stanford, Berkeley, UIUC, OSU) · 2025-10

Yukarıdaki iki makalenin **altyapıya dönüşmüş hali** — Atlas'ın ölçüm katmanı için en yakın şablon.
21.730 agent koşumu, 9 model × 9 benchmark, toplam ≈ **40.000 $**, 2,5 milyar token log,
yüzlerce VM üzerinde paralel çalıştırma.

Doğrudan bizim konumuza değen bulgular:
- Makalenin kendi gerekçesinde agentlar için ayrı altyapı gerekçesi olarak **"felaketle
  sonuçlanabilir ya da döngülere takılabilirler"** deniyor — loop, evaluation altyapısının
  birinci sınıf tasarım gerekçesi.
- **Doğruluk–maliyet Pareto sınırı hem dik hem seyrek.** 9 benchmark'ın yalnızca 1'inde en
  pahalı model sınırda; ortalamada modellerin üçte birinden azı sınırda. Bir vakada **9× maliyet
  farkı yalnızca 2 puan doğruluk** getiriyor.
- **6/9 benchmark'ta token kullanımı ile doğruluk pozitif korelasyonlu** — yani inference-time
  scaling henüz verimlilik kazancına dönüşmemiş; daha çok token = daha çok doğruluk, daha çok maliyet.
- **36 koşumun 21'inde daha yüksek "reasoning effort" doğruluğu artırmıyor.** Bütçeyi artırmanın
  otomatik kazanç getirmediğinin doğrudan kanıtı.
- Benchmark maliyetleri arasında büyüklük mertebesi farkı: ScienceAgentBench ortalama 13 $,
  Online Mind2Web 450 $ üzeri. Claude Opus 4.1'i Online Mind2Web'de çalıştırmaktan
  **tahminî 20.000 $ maliyet yüzünden vazgeçmişler**.
- Otomatik log analizi (Docent ile, 1.634 transkript) şunları çıkardı: agentlar görevi çözmek
  yerine **cevabı HuggingFace'te arıyor**; uçuş rezervasyonunda **yanlış kredi kartı kullanıyor**;
  en güçlü modeller bile tek bir araç çağrısı hatası olmadan koşum tamamlayamıyor;
  başarısız görevlerin **%60'ından fazlasında** açık bir talimat ihlali var; başarısız görevlerin
  yaklaşık **%40'ında** ortam/scaffold kaynaklı engel var.
- **Kurtarma işe yarıyor:** koşum ortasında bir araç çağrısı hatasını veya talimat ihlalini
  düzelten agent, başarma olasılığını **1,5–4×** artırıyor; sonucu araçla doğrulayanlar **%13–87**
  daha başarılı. → PoC'de "sert durdurma" yerine **önce nudge** tasarımının ampirik gerekçesi.
- HAL harness'ı **LiteLLM** (model uyumluluğu + maliyet takibi) ve **Weave** (loglama) üzerine
  kurulu. Atlas için aynı ikiliyi öneren bağımsız bir referans.


### 7.3 Why Do Multi-Agent LLM Systems Fail? (MAST) `[T]`
https://www.alphaxiv.org/abs/2503.13657 · UC Berkeley + Intesa Sanpaolo · NeurIPS 2025 D&B ·
**1031 oy / 33.464 görüntüleme — bu taramanın en yüksek sinyalli makalesi**

1642 açıklamalı yürütme izi, 7 MAS framework'ü, Grounded Theory ile 150 iz üzerinden kurulmuş
**14 hata modu / 3 kategori**; insan değerlendiriciler arası κ = 0,88, LLM annotator κ = 0,77.

**Bizim iki kalemimiz taksonomide birinci ve üçüncü sırada:**

| Hata modu | Pay | Bizim konu |
|---|---|---|
| **FM-1.3 Step repetition** (tamamlanmış adımların gereksiz tekrarı) | **%15,7** | ← **en sık hata modu**; loop detection |
| FM-2.6 Reasoning-action mismatch | %13,2 | |
| **FM-1.5 Unaware of termination conditions** (durma kriterini tanımama) | **%12,4** | ← budget/termination |
| FM-1.1 Disobey task specification | %11,8 | |
| FM-3.3 Incorrect verification | %9,10 | |
| FM-3.2 No or incomplete verification | %8,20 | |
| FM-2.3 Task derailment | %7,40 | |
| FM-2.2 Fail to ask for clarification | %6,80 | |
| **FM-3.1 Premature termination** | %6,20 | ← ters yönlü risk: erken kesme |
| FM-1.4 Loss of conversation history | %2,80 | |
| FM-2.1 Conversation reset | %2,20 | |
| FM-2.5 Ignored other agent's input | %1,90 | |
| FM-1.2 Disobey role specification | %1,50 | |
| FM-2.4 Information withholding | %0,85 | |

**Adım tekrarı + durma koşulunu tanımama = tüm MAS hatalarının %28,1'i.** Tek bir sunum
cümlesine sığan en güçlü gerekçe bu. Kategori düzeyinde: Sistem Tasarımı %44,2,
Agentlar Arası Uyumsuzluk %32,3, Görev Doğrulama %23,5.

Diğer kullanılabilir bulgular:
- 7 SOTA açık kaynak MAS'ta gözlenen hata oranı **%41 – %86,7**.
- **FM-1.5 neredeyse yalnızca başarısız koşumlarda görülüyor** — yazarlar bunu "ölümcül"
  hata modu olarak ayırıyor. Yani durma koşulunu tanımamak, düzeltilebilir bir kusur değil,
  görevi bitiren bir kusur.
- Framework'e göre profil değişiyor: OpenManus adım tekrarına, HyperAgent adım tekrarı +
  hatalı doğrulamaya, AppWorld erken sonlandırmaya eğilimli. **Tek beden çözüm yok.**
- Müdahale çalışmaları: yalnızca rol tanımını iyileştirmek ChatDev'de **+%9,4**; üst düzey
  görev hedefi doğrulaması eklemek **+%15,6** başarı getiriyor — aynı model, aynı prompt.
- **`pip install agentdash`** ile MAST bir Python kütüphanesi olarak kullanılabiliyor.
  PoC'de üretilen izleri sınıflandırmak için hazır araç.
- Ek N.1'de FM-1.3'ün gerçek bir izi var: HyperAgent'ta Planner'ın aynı "Thought" bloğunu
  kelimesi kelimesine tekrar etmesi. OpenHands'in `_is_stuck_monologue` senaryosunun canlı örneği.

---

## 8. Kapsam dışı bırakılanlar — ne çıkarıldı, neden

Liste 2026-08-20'de brief'e göre elden geçirildi. Başlangıçtaki 55 makalenin **33'ü
çıkarıldı**, 22'si kaldı. Çıkarma ölçütü tek: *loop detection ya da kaynak bütçesi
zorlamasıyla doğrudan ilgili mi?* İlgili ama farklı bir problemi çözen çalışmalar
aşağıda gruplandı — konu açılırsa geri getirilebilir diye giriş noktalarıyla birlikte.

**A. Güvenlik guardrail'i, kaynak guardrail'i değil (8 makale).** Bunlar *tehlikeli
eylemi* (geri alınamaz silme, yetkisiz ödeme) engelliyor; bizim sorunumuz *sınırsız
tüketim*. Farklı problem, farklı mekanizma. Giriş noktası gerekirse:
AgentSpec (https://www.alphaxiv.org/abs/2503.18666, 101 oy) alanın en çok atıf alan
runtime enforcement DSL'i; kural yazıp runtime'a uygulatma modeli teorik olarak adım
limitini de ifade edebilir. Diğerleri: VIGIL, SHE, DreamGuard, NEXUS, AgentBound,
"Toward Safe LLM Agents" derlemesi, "Beyond Component Testing".

**B. Hata teşhisi ve atfetme (7 makale).** Koşum bittikten sonra "nerede bozuldu"
sorusu. Bizimki çalışma anında müdahale. Tek istisna, PoC'de işe yarayabilecek açık
kaynak araç: **AgentDebugX** (https://www.alphaxiv.org/abs/2607.18754, 22 oy).
Diğerleri: Who&When Pro, TRAJDEBUG, LongRCA Bench, "Seeing the Whole Elephant",
"Tracing Agentic Failure", "Coordination as an Architectural Layer".

**C. Değerlendirme metodolojisi (9 makale).** Benchmark kalitesi, kaç koşum yeterli,
değerlendirme maliyeti. Kontrol mekanizması değil ölçüm bilimi. Bu hattın en iyi üç
örneği zaten §7'de tam metin olarak duruyor (AI Agents That Matter, HAL, MAST);
gerisi tekrar. Çıkarılanlar: "Beyond pass@1", "Consistency as a Testable Property",
"Deployment Decision Reliability", PACE, BenchGuard, "Benchmarking the Benchmarks",
"No Task Fails Every Time", "Stop Shipping AI Agents on Faith", EcoAgent-Bench.

**D. Genel hata taksonomisi ve uzun ufuk (7 makale).** Agentların neden bozulduğuna
dair geniş çerçeveler. Döngü bunların içinde bir alt başlık ama makaleler onu ayrıca
ele almıyor. MAST (§7.3) bu işi zaten sayısal olarak yapıyor. Çıkarılanlar:
"When Errors Become Narratives", "Silent Failure: The Entropy Principle",
"Long-Horizon Task Mirage", Long-Horizon-Terminal-Bench, "Agent Lifespan Engineering",
"How Do Agents Fail on AutoResearch", "Entropy-Based Observability".

**E. Kelime çakışması (3 makale).** Başlıkta "budget" ya da "stop" geçiyor ama başka
şeyden bahsediyor: BRA-Audit (denetim noktası yerleşimi), Organizational Control Layer
(eylem onayı), Agentic Confidence Calibration + AgentAbstain + Critic Experience Bank
(güven tahmini — "Agentic Abstention" §2'de tutuldu, o gerçekten durma kararı hakkında).

Bu ayıklamanın kendisi bir bulgu: **alanın "agent reliability" başlığı altındaki
üretiminin büyük kısmı ölçüm ve teşhis üzerine; çalışma anında tüketimi sınırlayan
mekanizma üzerine yazan az.** IAL-SCAN ile Token Budgets bu boşlukta duruyor.

---

## 9. Bu listenin boşlukları — dürüst kayıt

- **Doğrulama dağılımı:** tam metin okunan 6 makale — IAL-SCAN, Towards a Science of
  AI Agent Reliability, AI Agents That Matter, HAL, MAST, Token Budgets (kısmen: §1.2,
  §2.6, §2.7). Kalan 16 makale `[Ö]`, yani sadece abstract. Bir iddiayı sunuma koymadan
  önce ilgili makale açılmalı.
- **Kaynakların çoğu 2026 tarihli ve oy sayıları düşük.** Yeni, hakem sürecinden
  geçmemiş olabilir. Kalan listede yüksek sinyalli olanlar: MAST (1031 oy), Towards a
  Science (75), Model-or-Harness (52), Agentic Abstention (29).
- **OWASP LLM10 sayfası otomatik erişime 403 verdi**, içerik ikincil kaynaklardan
  derlendi. Sunumdan önce elle açılıp doğrulanmalı.
- **Cline sayıları DeepWiki indeksinden geldi**, dosyadan okunmadı. `harness_kontrolleri.md`
  bunu doğruluyor olmalı.
- **Atıf grafiği taraması yapılmadı.** Hangi makalenin gerçekten dayanak alındığı,
  hangisinin izole kaldığı görülmedi.
- **Harness karşılaştırması ayrı dosyada:** `harness_kontrolleri.md` — Aider, Codex CLI,
  Gemini CLI, SWE-agent, Roo Code, Continue, Goose, opencode, Agno, Letta, DSPy ve
  gateway/gözlemlenebilirlik katmanı orada inceleniyor. Bu dosyadaki §5, o çalışmanın
  önceden doğrulanmış çekirdeği.
