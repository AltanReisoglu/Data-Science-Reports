# Research: Kurumsal Zemine-Dayalı Asistan (Faz 1)

Bu doküman, plan.md'deki Technical Context'te yer alan seçimlerin gerekçesini ve
değerlendirilen alternatifleri kaydeder. Büyük kararların çoğu doğrudan Altan ile
konuşularak teyit edildi (bkz. her maddenin "Kaynak" satırı); geriye kalan küçük/düşük
riskli seçimler (CLI kütüphanesi, embedding modeli, RRF sabiti) makul varsayımlar olarak
işaretlendi ve kolayca değiştirilebilir.

## 1. Agent framework: LangGraph (deepagents YOK)

- **Decision**: Yalnızca LangGraph (durable graph orchestration). `deepagents`
  KULLANILMIYOR — Altan'ın 2026-08-27 tarihli açık kararıyla kaldırıldı.
- **Rationale**: `deepagents`in asıl cazibesi "programmatic subagents (RLM-like)"
  desteğiydi (Faz 2 PTC hedefi için); ama Faz 1 kapsamında PTC henüz yok, bu destek
  bugün kullanılmıyor — gereksiz bir bağımlılık olurdu. Faz 2'de PTC işine
  başlanınca bu karar yeniden değerlendirilebilir.
- **Alternatives considered**: deepagents (kaldırıldı — bugün gereksiz), Pydantic AI,
  OpenAI Agents SDK / Google ADK (Anthropic dışı ekosistem).
- **Kaynak**: Altan'ın doğrudan kararı (2026-08-27, "DeepAgent kullanmana gerek yok").

## 2. Kurumsal bilgi bankası: Hybrid Search (BM25 + dense) + RRF

- **Decision**: `rank_bm25` (lexical) + **`Qwen3-Embedding-8B`** (dense embedding,
  mevcut `.env`'deki `LLM_BASE_URL` gateway'i üzerinden, OpenAI-uyumlu embeddings
  endpoint'i ile) — ikisinin sonuçları **Reciprocal Rank Fusion (RRF)** ile birleştirilir.
  Yerel `sentence-transformers` modeli KULLANILMIYOR — Altan'ın 2026-08-27 kararıyla,
  zaten yapılandırılmış/onaylı gateway'deki embedding modeline geçildi.
- **Rationale**: Gateway zaten `.env`'de tanımlı ve onaylı (ek credential gerekmiyor,
  "Model ve API Kullanımı" kuralıyla uyumlu); ayrı bir yerel model indirip çalıştırmaya
  gerek kalmıyor.
- **LLM (chat) modeli**: `gemma-4-31B-it` (Google), aynı gateway üzerinden (`.env` →
  `LLM_MODEL_NAME`) — değişmedi. Hem chat hem embedding, aynı OpenAI-uyumlu gateway'den.
- **RRF sabiti**: `k=60` — literatürde (Cormack et al.) standart varsayılan değer.
- **Alternatives considered**: LightRAG/GraphRAG (reddedildi, Principle V), hosted
  vector DB — Qdrant/Weaviate/Pinecone (reddedildi), yerel `sentence-transformers`
  (ilk kararımızdı, gateway'de zaten embedding modeli olduğu keşfedilince değiştirildi).
- **Kaynak**: Altan'ın doğrudan kararı (2026-08-27) + `.env`/gateway keşfi (bu konuşma).

## 3. Canlı sistemler erişim yolu: mock ama gerçek protokol (MCP)

- **Decision**: Canlı sistem, `fastmcp` ile yazılmış, verisi tamamen sahte olan ama
  gerçek bir MCP sunucusu olarak ayrı bir process'te çalışan bir mock servis. Ajan buna
  `langchain-mcp-adapters` (`MultiServerMCPClient`) üzerinden bağlanır.
- **Rationale**: Altan açıkça "mock olacak ama sanki gerçek ürün gibi bağlantı
  yapabileceğim" dedi — trivial bir in-process Python fonksiyonu bunu karşılamıyor.
  MCP seçimi ayrıca Faz 2/3'teki PTC çalışmasına doğal bir geçiş sağlıyor: PTC
  araştırmamızda (Anthropic/OpenAI/Glean/Flyte karşılaştırması) MCP'nin tool erişimi
  için fiilen kullanılan gerçek dünya protokolü olduğunu doğrulamıştık.
- **⚠️ Not (Altan'ın onayına açık)**: Bu protokol seçimi (MCP vs düz REST/FastAPI mock),
  ayrı bir soru olarak Altan'a sorulmadı — önceki tur'daki genel MCP tartışmasından
  türetilen bir sonuç. Faz 1'de MCP kullanmak istemiyorsa (ör. düz bir REST mock
  yeterli görülüyorsa), bu kolayca değiştirilebilir; mimari başka hiçbir yeri etkilemez
  çünkü `access_paths/live_systems.py` bu bağlantıyı izole ediyor.
- **Alternatives considered**: Düz FastAPI REST mock (daha basit ama Faz 2 MCP işine
  hazırlık sağlamıyor), trivial Python fonksiyon stub'ı (Altan tarafından açıkça
  reddedildi).
- **Kaynak**: Altan'ın "mock ama gerçek ürün gibi" kararı + bu dokümanda türetilen
  protokol seçimi (onaya açık).

## 4. Onaylı-kanal (Principle II) denetiminin yeri: LangChain HumanInTheLoopMiddleware

- **Decision**: Backend'de özel bir "gateway.py" modülü YAZILMIYOR. Bunun yerine
  LangChain'in resmi `langchain.agents.middleware.HumanInTheLoopMiddleware`'i,
  `interrupt_on` + saf-kod `when` predicate'iyle yapılandırılıyor — mock canlı sistem
  tool'larına (get_ticket_status, list_open_tickets) yapılan her çağrı, çalıştırılmadan
  önce bu predicate'ten geçiyor. Predicate `False` dönerse çağrı hiç durmadan otomatik
  yürütülür (insan gerekmez); `True` dönerse framework bir "interrupt" üretir ve bizim
  kendi otomatik karar-verici kodumuz (insan değil) bunu anında "reject" ile resume eder.
- **Rationale**: PTC vendor araştırmasında hiçbir sağlayıcının `allowed_callers` benzeri
  metadata'sının gerçek bir güvenlik sınırı olmadığı doğrulandı ("do not rely on it as
  a security boundary" — Anthropic PTC docs). Gerçek denetim noktasının framework-dışı,
  bizim politikamızda olması gerekiyor — LangChain'in middleware'i bunu resmi/üretim
  standardı bir mekanizmayla veriyor; elle yazılmış, test edilmemiş bir allowlist
  fonksiyonundan daha güvenilir ve daha az kod (Principle V).
- **Önemli sınırlama**: Middleware'in kendisi audit log tutmuyor (dokümantasyonda
  açıkça yok) — bu boşluğu zaten planlanan `trace.py` (T007) dolduruyor, ekstra iş değil.
  Ayrıca reddetme her zaman bir "interrupt" (durma noktası) üretiyor; tam sessiz/anlık
  red için bizim kendi otomatik karar-verici kodumuzun bu interrupt'ı hemen "reject"
  kararıyla resume etmesi gerekiyor (insan gerekmez, ama teknik olarak bir ekstra adım).
- **Altan'ın kararı (2026-08-27)**: Bu iki mock tool için bugün gerçek bir risk olmasa
  da, desen Faz 2/3'teki (PTC + egress, gerçek risk) ihtiyaca hazırlık olarak **Faz
  1'den itibaren** kuruluyor.
- **Alternatives considered**: Elle yazılmış özel `gateway.py` (reddedildi — framework'ün
  resmi mekanizması varken gereksiz); ayrı dedicated gateway ürünü/servisi — Kong,
  LiteLLM proxy (reddedildi, Faz 1 için aşırı altyapı, Principle V); `wrap_tool_call`
  middleware hook'u — daha basit, interrupt/resume gerektirmeyen bir alternatifti
  (Altan'ın işaret ettiği "Managed Deep Agents Middleware" sayfası üzerinden keşfedildi),
  ama Altan açıkça `HumanInTheLoopMiddleware`'i tercih etti.
- **Kaynak**: PTC vendor araştırması + [LangChain Human-in-the-Loop middleware](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
  dokümantasyonu (bu konuşmada doğrulandı).
- **DeepSeek Harness'tan (deepseek-ai/deepseek-harness, MIT) alınan ilham**: İki ders
  taşındı: (1) `LiveToolCall.status`'a DSH'nin kapalı outcome sözlüğüne benzer şekilde
  `unavailable` durumu eklendi; (2) kalıcı "her zaman izin ver" durumu kasıtlı olarak
  yok — her karar tek seferlik.
- **Kaynak**: [deepseek-ai/deepseek-harness — docs/subsystems/approval.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/approval.md)
- **Düzeltme (2026-08-27, kaynak kod doğrulaması)**: "Tanımsız/politikasız tool için
  varsayılan durdur/reddet (fail-closed)" iddiası yanlıştı. `HumanInTheLoopMiddleware`'in
  gerçek kaynak kodu (`langchain-ai/langchain`,
  `libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py`) açıkça şunu
  söylüyor: *"If a tool doesn't have an entry [in `interrupt_on`], it's auto-approved
  by default."* — yani fail-**open**, fail-closed değil. Bizim fail-closed güvencemiz
  middleware'in varsayılanından değil, `tools=[...]` listesine yalnızca 2 bilinen
  tool'u (get_ticket_status, list_open_tickets) koymamızdan geliyor (closed-world tool
  kaydı) — ikisi de `interrupt_on`'da açık bir `InterruptOnConfig` girdisine sahip
  olacak, `when` predicate'i her ikisi için de her zaman `True` dönüp bizim otomatik
  karar-verici kodumuzun her çağrıda fiilen devreye girmesini sağlayacak.
- **Ek gereksinim**: `interrupt()`/resume akışı bir **checkpointer** gerektiriyor
  (ör. `langgraph.checkpoint.memory.InMemorySaver`) — bu, T009'a (graph.py) eklenecek.

## 5. Hafıza erişim yolu: bu fazda yok

- **Decision**: Memory access path Faz 1'de implemente edilmiyor.
- **Rationale**: Altan'ın açık kararı ("memory'i şuan eklemeyeceğim").
- **Not**: Spec'teki US3 (hafıza senaryosu) Faz 1 kapsamı dışında kalıyor; ileride ayrı
  bir faz/spec güncellemesiyle eklenecek.

## 6. Arayüz: CLI

- **Decision**: `typer` tabanlı bir CLI (`assistant ask "<soru>"` şeklinde).
- **Rationale**: Altan'ın seçimi; UI Faz 4'te geliyor, bu fazda hızlı test/iterasyon
  önceliği var (Principle V).
- **Alternatives considered**: FastAPI backend (Faz 4 UI için hazırlık sağlardı ama
  şimdi gereksiz altyapı), Jupyter notebook (daha az reprodükte edilebilir).
- **Kaynak**: Altan'ın doğrudan kararı.

## 7. Örnek dokümanların kaynağı

- **Decision**: 4 paralel bilgi bankası kaynağı için örnek/sentetik dokümanları
  **Altan sağlayacak** — Claude tarafından üretilmeyecek, depodaki mevcut dosyalar da
  kullanılmayacak.
- **Not**: Bu, implementasyon için bir **dış bağımlılık**tır — `sample_docs/` dizini
  boş bir iskelet olarak oluşturulacak, dokümanlar geldiğinde ingestion adımı
  tamamlanabilecek. tasks.md'de bu açıkça bir bağımlılık olarak işaretlenecek.
- **Kaynak**: Altan'ın doğrudan kararı.
