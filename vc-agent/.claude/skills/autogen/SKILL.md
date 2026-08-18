---
name: autogen
description: AutoGen (microsoft/autogen v0.4+ — autogen-core, autogen-agentchat, autogen-ext) hakkında herhangi bir soru geldiğinde kullan. Mimari ve aktör modeli, ajan kimliği, topic/abonelik, doğrudan mesaj ve yayın, RoutedAgent, runtime, takımlar, sonlandırma, tool/workbench, MCP, gözlemlenebilirlik, dokuz çok-ajan deseni, konuştuğu protokoller, rakiplerle karşılaştırma (LangGraph, CrewAI, Google ADK, OpenAI Agents SDK, Swarm, DeepAgents, MAF, AG2) ve pratikte ısıran tuzaklar. Bu repodaki ölçülmüş belgelere ve çalışan koda dayanır — ezberden cevap vermek yerine bunu kullan.
---

# AutoGen

Bu repo AutoGen üzerine **koşturularak** yazılmış ~2.300 satır Türkçe belge ve çalışan
bir kod tabanı taşıyor. Ezberden cevap verme; buradaki dosyalara bak, satır numarası ver.

Yollar `vc-agent/` köküne göredir. Kurulu sürüm: **`autogen-core` / `autogen-agentchat` /
`autogen-ext` 0.7.5** (`.venv`'de doğrulandı).

---

## 1 — Cevaba başlamadan önce doğru olması gerekenler

Modelin en sık yanlış bildiği üç şey. Konu açıldığında **söylenmeli**, çünkü kullanıcının
kararını değiştirir:

**① AutoGen bakım modunda.** Kendi README'sinden: *"AutoGen is now in maintenance mode.
It will not receive new features or enhancements... New users should start with Microsoft
Agent Framework."* Son sürüm `python-v0.7.5` (2025-09-30), son commit 2026-04-15. Resmî
halef **microsoft/agent-framework (MAF)**, Nisan 2026'da 1.0 GA. → [docs/01 §0](../../../docs/01-autogen-kaynak-haritasi.md)

**② Dört isim var, karıştırılıyor.** → [docs/09 §10](../../../docs/09-framework-karsilastirma.md)

| İsim | Ne | Durum |
|---|---|---|
| **microsoft/autogen v0.4+** | `autogen-core` + `agentchat` + `ext`. Bu projenin kullandığı | Bakım modu, v0.7.5 |
| **AutoGen v0.2** | `ConversableAgent`, `initiate_chat` | Terk edilmiş |
| **ag2ai/ag2** | v0.2 kolundan fork, ayrı ekip | Aktif v1.0.1. v1.0'da `autogen` isim alanını "AG2 Classic"e taşıdı — `pip install ag2` artık `import autogen` **sunmuyor** |
| **microsoft/agent-framework** | AutoGen + Semantic Kernel birleşimi | Aktif, resmî halef |

> **Filtre:** Bir kaynakta `ConversableAgent` ya da `initiate_chat` görüyorsan o kaynak
> v0.2 veya AG2 Classic anlatıyor, bu projeyle **uyumsuz**.

**③ Aktör modeli runtime'ı korur, veriyi korumaz.** Çöken bir handler `_process_publish`
içindeki `gather`'ı erken döndürüyor, `stop_when_idle()` bariyeri erken açılıyor,
tamamlanmış kardeş sonuçlar **sessizce** kayboluyor. Ölçüldü, deterministik bile değil.
→ [docs/06 §8](../../../docs/06-autogen-incelikleri.md)

---

## 2 — Evin kuralı: her iddia etiketli

Bu repodaki belgeler iddiaları etiketliyor; cevaplar da öyle olmalı:

| Etiket | Anlamı |
|---|---|
| **[ölçüldü]** | Bu projede kod koşturularak elde edildi, ölçüm dosyası gösterilebilir |
| **[kaynak]** | Birincil kaynaktan doğrulandı (repo, README, resmî doküman, kurulu paket) |
| **[teyitsiz]** | Okunandan; koşturulmadı |

Rakip framework'ler hakkındaki **mimari** iddialar çoğunlukla [teyitsiz] — AutoGen ve
Google ADK gerçekten koşturuldu, LangGraph/CrewAI/Agents SDK/MetaGPT koşturulmadı. Bunu
gizleme.

---

## 3 — Soru hangi belgeye gidiyor

| Soru | Belge |
|---|---|
| "AutoGen nedir, baştan sona anlat" | [docs/12](../../../docs/12-autogen-bastan-sona.md) — 301 satır, üç katman |
| Core kavramları: kimlik, runtime, topic, abonelik | [docs/11](../../../docs/11-core-guide-turkce.md) — `05:NNNN` satır atıflı |
| Günlük iş: ajanlar, takımlar, sonlandırma | [docs/10](../../../docs/10-agentchat-turkce.md) |
| **Ne ısırıyor / neden çalışmıyor** | [docs/06](../../../docs/06-autogen-incelikleri.md) — 13 tuzak, en yüksek getirili belge |
| Protokoller, rakip karşılaştırması, mekanizma mekanizma fark | [docs/14](../../../docs/14-autogen-protokoller-ve-farklar.md) — 1453 satır |
| Framework seçimi, ne zaman kullanılmaz, bakım modunun bedeli | [docs/09](../../../docs/09-framework-karsilastirma.md) |
| Kavram kodda nerede yaşıyor | [docs/07](../../../docs/07-kod-rehberi.md) |
| Tam İngilizce metin, satır numarasıyla | [docs/05](../../../docs/05-autogen-core-user-guide.md) (core), [docs/08](../../../docs/08-autogen-agentchat-user-guide.md) (agentchat) |

`docs/11` ve `docs/14` başlıkları `05:670` gibi **satır atıfları** taşıyor — kaynak metne
o satırdan girilir. Cevapta bu atfı ver.

---

## 4 — Üç katman

```
autogen_ext        dış dünya: model istemcileri, MCP, kod yürütücüler, üçüncü parti
autogen_agentchat  günlük iş: AssistantAgent, beş takım tipi, 11 sonlandırma koşulu
autogen_core       aktör modeli: AgentId(type,key), runtime, topic, abonelik
```

Ayıran şey **`autogen_core`**: ajanlar gerçekten aktör — kendi mailbox'ı olan, mesajı
**tipe göre** yönlendiren, makinelere dağıtılabilen birimler. LangGraph'ın graf yürütücü +
checkpointer'ı **durability** sağlıyor, eşzamanlılık modeli değil. "AutoGen mı LangGraph
mı" çoğu zaman yanlış sorulmuş soru — farklı katmanlar.

### Bilinmesi en çok işe yarayan mekanizma

**Topic kaynağı, ajan anahtarına dönüşür** (`05:670`). `TopicId("turn", "oturum-42")`'a
yayın yapmak `AgentId("session", "oturum-42")` ajanını yaratır — oturum başına izole örnek,
elle sözlük tutmadan. Bu repoda gateway oturumları tam olarak böyle çalışıyor
([pipeline/gateway/sessions.py](../../../pipeline/gateway/sessions.py)).

### İki iletişim biçimi — fark adresleme değil, hata

| | Doğrudan (`send_message`) | Yayın (`publish_message`) |
|---|---|---|
| Alıcı | tek `AgentId` | topic'e abone olan herkes |
| Dönüş | var | **yok** |
| Handler çökerse | çağırana **fırlatır** | **loglanır, fırlatmaz** |

Bu tek satır tasarım kararı: bir sonucu bekleyeceksen doğrudan, bir olayı duyuracaksan
yayın. → [docs/14 §3.3](../../../docs/14-autogen-protokoller-ve-farklar.md)

---

## 5 — Ölçülmüş sayılar

**Desen seçiminin faturası** [ölçüldü] — `poc/kiyas.py`, aynı görev, yalnız orkestrasyon değişiyor:

| desen | mesaj | LLM | tool | token |
|---|---:|---:|---:|---:|
| **SelectorGroupChat** | 8 | 5 | 2 | **204** |
| GraphFlow | 11 | 7 | 3 | 270 |
| RoundRobinGroupChat | 9 | 6 | 2 | 274 |
| **Swarm** (handoff) | 14 | 7 | 4 | **334** |

**%63,7 fark.** Ödenen şey zekâ değil **yönlendirme özerkliği**. Bunun karşılaştırmaya
çevirisi: *Agents SDK'nın tek modeli olan handoff, AutoGen'in en pahalı desenidir.*

**Fan-in'de kardeş kaybı** [ölçüldü] — `pipeline/compare_fanin.py`, aynı arıza enjeksiyonu:

| motor | temiz | sarmalayıcı arkasında | ham hata |
|---|---:|---:|---:|
| GraphFlow | 3 | 2 | **0–1, süre sınırı dolar** |
| core pub/sub + `ClosureAgent` kuyruğu | 3 | 2 | **2, ~3 ms** |

Resmî desenler bu konuda **birbiriyle çelişiyor**: *Concurrent Agents* kuyrukla topluyor,
*Mixture of Agents* `asyncio.gather(...)` ile — sessiz kaybın kaynağı olan yapı.

---

## 6 — Pratikte ısıranlar

Tamamı [docs/06](../../../docs/06-autogen-incelikleri.md)'da, 13 madde. Sık sorulanlar:

| Tuzak | Sonuç |
|---|---|
| `tools=` ve `workbench=` aynı ajana | `ValueError: Tools cannot be used with a workbench.` |
| `model_context` verilmemiş | Ajanın **belleği yoktur**, hata da vermez |
| OpenAI-*uyumlu* endpoint | `model_info` **zorunlu** |
| `max_tool_iterations` varsayılanı **1** | Ajan bir tool çağırır, sonucu görür, **durur**. Zincirleme davranış sessizce imkânsız |
| Dış runtime verilmiş, ajan çöküyor | **Fırlatmaz, asar** |
| `output_content_type` + takım | `Message type StructuredMessage[X] is not registered` → `custom_message_types=[...]` |
| `description` boş ajan | `SelectorGroupChat` kör seçim yapar |
| `Handoff` tool adı küçük harfe düşer | Elle yazarsan eşleşmez; `Handoff(target=X).name` ile üret |
| Sonlandırma koşulu yok | Sonsuz ajan döngüsü = gerçek fatura |
| `stop_when_idle()` | Handler çökerse bariyer erken açılır — güvenme, **beklenen sonucu say** |
| Bağımlılık üst sınırı yok | Bakım modundaki proje kurulamaz hâle gelir; düzeltecek kimse yok |

---

## 7 — Konuştuğu protokoller

Derlenmiş proto descriptor'larından ve kurulu paketten okundu [kaynak] — dokümandan değil.
→ [docs/14 §2](../../../docs/14-autogen-protokoller-ve-farklar.md)

| Katman | Protokol | Kanıt |
|---|---|---|
| Olay formatı | **CloudEvents v1** | `_topic.py` docstring · `io.cloudevents.v1` proto |
| Dağıtık taşıma | **gRPC + protobuf**, çift yönlü akış | `AgentRpc` servisi |
| Tool federasyonu | **MCP** (stdio · SSE · streamable HTTP) | `ext.tools.mcp` |
| Gözlemlenebilirlik | **OTel GenAI conventions** | `_genai.py`, `gen_ai.system="autogen"` |
| Model erişimi | **OpenAI Chat Completions** (fiilî) | `OpenAIChatCompletionClient` |
| Serileştirme | Kendi `ComponentModel` şeması | JSON schema |
| Ajanlar arası federasyon | **yok** | `a2a` modülü mevcut değil |

Dışa açılan her yerde standart; yalnız bileşen serileştirmesinde kendi şeması. A2A yok.

---

## 8 — Bu repoda çalışan karşılığı

Soyut anlatma — buradaki dosyayı göster, kullanıcı koşturabilir:

| AutoGen yüzeyi | Bizde |
|---|---|
| `AssistantAgent`, tool'lar, yapısal çıktı | [pipeline/agents/](../../../pipeline/agents/) |
| `GraphFlow` + `DiGraphBuilder`, join `activation_condition="all"` | [pipeline/graph.py](../../../pipeline/graph.py) |
| core pub/sub, `RoutedAgent`, `ClosureAgent` | [pipeline/fanin.py](../../../pipeline/fanin.py) |
| `InterventionHandler`, `DropMessage`, olay akışı | [pipeline/observability.py](../../../pipeline/observability.py) |
| `model_context`, `StaticWorkbench`, `McpWorkbench`, `save_state` | [pipeline/conversation.py](../../../pipeline/conversation.py) |
| `ReplayChatCompletionClient` (deterministik kuru mod) | [pipeline/engine.py](../../../pipeline/engine.py) |
| topic source → agent key ile oturum izolasyonu | [pipeline/gateway/sessions.py](../../../pipeline/gateway/sessions.py) |
| Uzun ömürlü runtime, `stop_when_idle()` **çağrılmıyor** | [pipeline/gateway/runtime.py](../../../pipeline/gateway/runtime.py) |

**Henüz kullanılmayanlar:** dağıtık runtime (gRPC), `dump_component`/`load_component`, kod
yürütücüler, `Memory` protokolü, Magentic-One, OpenTelemetry, `Handoff`/`Swarm`.

---

## 9 — Nasıl cevaplanır

1. **Belgeye bak, ezberden yazma.** Yukarıdaki yönlendirme tablosundan doğru dosyayı aç.
2. **Atıf ver** — `docs/06 §8` ya da `05:670`. Kullanıcı doğrulayabilmeli.
3. **İddiayı etiketle** — ölçülmüş bir sayı ile okunmuş bir cümle aynı tonda durmamalı.
4. **Bakım modunu ve isim karışıklığını saklama**; bir öneri verirken MAF alternatifini söyle.
5. **Bir tuzak varsa önce onu söyle.** "Nasıl yapılır"ın cevabı çoğu zaman
   [docs/06](../../../docs/06-autogen-incelikleri.md)'daki bir maddedir ve kullanıcı onu
   bilmeden koşturursa sessizce yanlış sonuç alır.
6. **Bulamadıysan bunu söyle.** Belgelerde olmayan bir konuda uydurma; `docs/05` ve
   `docs/08` tam metin, orada ara — yoksa "ölçmedik" de.
