# 09 — AutoGen ve diğerleri: fark nerede, neden kullanılır, ne zaman kullanılmaz

*Bu belge bir savunma değil, bir karar belgesi. AutoGen'i bu projede kullandık ve
kullanmaya devam ediyoruz — ama kullanmama gerekçeleri de burada, çünkü bir aracı
seçmenin tek dürüst yolu ne zaman yanlış olduğunu da yazmak.*

---

## 0 — Bu belgenin kuralı: her iddia etiketli

Framework karşılaştırmalarının çoğu okunamaz, çünkü ölçülmüş bir sayı ile bir blog
yazısından alınmış bir cümle yan yana, aynı tonda duruyor. Burada her iddianın
nereden geldiği yazılı:

| Etiket | Anlamı |
|---|---|
| **[ölçüldü]** | Bu projede kod koşturularak elde edildi. Ölçüm dosyası gösterilebilir |
| **[kaynak]** | 2026-08-14'te birincil kaynaktan doğrulandı (repo, README, resmî doküman) |
| **[teyitsiz]** | Okuduklarımdan/bildiklerimden; **koşturulmadı, bugün doğrulanmadı** |

**En büyük dürüstlük sınırı şu:** AutoGen'i ve Google ADK'yı gerçekten koşturdum.
LangGraph, CrewAI, OpenAI Agents SDK ve MetaGPT hakkındaki **mimari** iddialar
[teyitsiz] — repo sinyallerini bugün doğruladım ama kodlarını çalıştırmadım.
Adil kıyas, ancak taraflara aynı önkoşul verilirse mümkün; burada verilmedi ve
bunu gizlemiyorum.

---

## 1 — Bugünkü tablo

**[kaynak]** 2026-08-14, `gh api` ile:

| Framework | ⭐ | Son commit | Son sürüm |
|---|---:|---|---|
| **microsoft/autogen** | 60.414 | **2026-04-15** | **python-v0.7.5 (2025-09-30)** |
| microsoft/agent-framework | 12.791 | 2026-08-14 | python-1.14.0 (**bugün**) |
| ag2ai/ag2 | 4.856 | 2026-08-14 | v1.0.1 (2026-07-29) |
| langchain-ai/langgraph | 39.648 | 2026-08-14 | 1.2.11 (2026-08-11) |
| crewAIInc/crewAI | 57.059 | 2026-08-14 | 1.15.16 (bugün) |
| openai/openai-agents-python | 28.631 | 2026-08-14 | v0.20.0 (2026-08-11) |
| google/adk-python | 21.104 | 2026-08-14 | v2.7.0 (2026-08-13) |
| geekan/MetaGPT | **69.812** | **2026-01-21** | **v0.8.1 (2024-04-22)** |

### Bu tablodan çıkan ilk ders: yıldız gecikmeli bir göstergedir

Listedeki **en yıldızlı iki proje**, en az hareket edenler. MetaGPT 69.812 yıldızla
zirvede ama son sürümü **2024 Nisan**, son commit'i 2026 Ocak. AutoGen 60.414
yıldızla ikinci ama Nisan'dan beri commit almamış.

Yıldız, **birikmiş ilgiyi** ölçüyor — güncel sağlığı değil. Bir framework seçerken
bakılacak üç alan şu: son commit tarihi, son sürüm tarihi, ve sürüm temposu.
Yıldız dördüncü sırada bile değil.

*(Kendi star envanterimde de aynı sapma vardı: 893 starın 167'si agent framework'ü,
ama çoğu bir kez bakılıp bırakılmış repolar. Bkz. `docs/github-starred-repos.md`.)*

---

## 2 — AutoGen nedir: üç katman ve bir yeniden yazım

**[kaynak]** AutoGen v0.4, v0.2'nin devamı değil — **sıfırdan yeniden yazımı.**
Gerekçeyi resmî migration guide'ın kendisi söylüyor:

> Since the release of AutoGen in 2023, we have intensively listened to our
> community and users… Based on that feedback, we built AutoGen `v0.4`, a
> **from-the-ground-up rewrite adopting an asynchronous, event-driven
> architecture** to address issues such as **observability, flexibility,
> interactive control, and scale**.

Bu cümleyi aklında tut — belgenin sonunda buna döneceğiz. Sayılan dört ihtiyaç
(gözlemlenebilirlik, esneklik, etkileşimli kontrol, ölçek) tam olarak **üretimin**
istediği şeyler.

Katmanlar:

```
  autogen-ext        model istemcileri · tool'lar · MCP · kod yürütücüler
       ↑
  autogen-agentchat  AssistantAgent · takımlar · sonlandırma · yapısal çıktı
       ↑             ← tutorial'ların bittiği yer
  autogen-core       aktör modeli · event-driven runtime · pub/sub · gRPC
                     ← asıl mühendislik hikâyesi burada
```

Çoğu karşılaştırmanın atladığı nokta: **AutoGen'i ayıran şey üst katman değil, alt
katman.** AgentChat'te gördüğün `AssistantAgent` her framework'te var. `autogen_core`
karşılığı çoğunda **yok**.

Bu projede üç katmanı da kullandık: AgentChat günlük iş için (`graph.py`), core
gözlemlenebilirlik ve alternatif toplama için (`observability.py`, `fanin.py`),
ext model istemcisi ve MCP için (`engine.py`, `conversation.py`).

---

## 3 — Metafor ekseni: her framework'ün bir dünya görüşü var

Her framework'ün bir **metaforu** var ve o metafor API'nin tamamını belirliyor.
Framework seçmek, aslında metafor seçmek.

| Framework | Metafor | Temel primitif | Kaynak |
|---|---|---|---|
| **AutoGen** | **Konuşma** | Ortak mesaj thread'i; ajanlar "conversable" | [kaynak] |
| LangGraph | **Graf** | Düğüm + kenar + state; checkpointer | [teyitsiz] |
| CrewAI | **İnsan ekibi** | Agent · Task · Crew (rol/backstory) | [teyitsiz] |
| OpenAI Agents SDK | **Devir** | Agent · Tool · Handoff · Guardrail | [teyitsiz] |
| Google ADK 2.x | **Graf** (derlenen) | `Agent` + `Workflow`, `validate_graph()` | [kaynak] |
| MetaGPT | **SOP / montaj hattı** | Sabit roller (PM→Mimar→Mühendis→QA) | [teyitsiz] |

Pratik sonucu şu: **AutoGen'de akış konuşmanın içinden doğuyor.** Ajanlar
birbirine mesaj *göndermiyor*, ortak bir thread'e yayın yapıyor ve herkes görüyor.
"Kim konuşacak" kararı bir strateji nesnesi — değiştirilebilir.

Bu yüzden AutoGen'de **tek kütüphane içinde beş desen** var (RoundRobin, Selector,
Swarm, GraphFlow, MagenticOne). Diğerlerinde çoğu zaman **desen = framework
seçimi**: graf istiyorsan LangGraph, devir istiyorsan Agents SDK.

Bunun bedeli de var ve ölçtük: ortak thread modeli bağlamı şişiriyor, ve
konuşmacı-seçimi kırılganlık üretiyor.

---

## 4 — Asıl fark: `autogen_core` aktör modeli

**[kaynak]** AutoGen, ajanları **aktör** yapan neredeyse tek yaygın framework:
kendi mailbox'ı olan, mesajı tipe göre yönlendiren, süreçlere ve makinelere
dağıtılabilen birimler.

Diğerlerinde bu katman yok **[teyitsiz]**:
- LangGraph'ın altında graf yürütücü + checkpointer var — **durability** sağlıyor,
  eşzamanlılık modeli değil
- CrewAI düpedüz bir Python döngüsü
- Agents SDK'da `Runner` tek bir tur döngüsü

Yani "AutoGen mi LangGraph mı" sorusu çoğu zaman yanlış sorulmuş oluyor: ikisi
**farklı katmanlarda** duruyor. LangGraph'ın grafı bir *yürütme planı*;
`autogen_core` bir *eşzamanlılık modeli*.

### Ama vaat tam tutmuyor — ve bunu ölçtük

**[ölçüldü]** `poc/desen_5_core_aktor.py`: "aktör modeli hata izolasyonu verir"
iddiasını üç deneyle test ettim.

Çöken bir handler, `_process_publish` içindeki `asyncio.gather`'ı erken
döndürüyor; hemen ardından `task_done()` çağrılıyor; kuyruk kardeşler hâlâ
çalışırken "boşaldı" sayılıyor. `stop_when_idle()` bariyeri erken açılıyor ve
**tamamlanmış kardeş sonuçlar sessizce kayboluyor** — ne exception, ne uyarı.
Üç koşuda birebir tekrarlandı.

> **Aktör modeli runtime'ı koruyor, veriyi korumuyor.**

**[ölçüldü]** Aynı şey bir kat yukarıda, AgentChat'te de var. `compare_fanin.py`,
iki toplama mekanizmasını **aynı arıza enjeksiyonuyla** ölçüyor:

| motor | temiz | sarmalayıcı arkasında hata | ham hata |
|---|---:|---:|---:|
| `GraphFlow` (AgentChat) | 3 dal | 2 dal | **0–1 dal, süre sınırı dolar** |
| pub/sub + `ClosureAgent` kuyruğu (core) | 3 dal | 2 dal | **2 dal, ~3 ms** |

Son sütun: kendisiyle ilgisi olmayan bir dalın çökmesi, AgentChat motorunda
**tamamlanmış** kardeş dalların işini yok ediyor. Ve **kaç tanesini yok ettiği
deterministik değil** — tekrarlı koşularda 0 ve 1.

Bu, bir kütüphane hatasından fazlası: **resmî desenler bu konuda birbiriyle
çelişiyor.** *Concurrent Agents* sonuçları kuyrukla topluyor; *Mixture of Agents*
`asyncio.gather` ile — yani kaybın kaynağı olan yapıyla.

---

## 5 — Eksen eksen matris

**[kaynak]** AutoGen ve ADK sütunları; **[teyitsiz]** diğerleri.

| eksen | AutoGen | LangGraph | CrewAI | Agents SDK | Google ADK 2.x |
|---|---|---|---|---|---|
| **iletişim** | pub/sub ortak thread | paylaşılan state + `Command` | task-context zinciri | yalnızca handoff | graf kenarları |
| **akış kararı** | konuşmacı seçimi (**değiştirilebilir**) | kenarlar (önceden çizili) | manager delegasyonu | ajanın kendisi | kenarlar (**derlenen**) |
| **hata ne zaman görünür** | çalışma zamanında | çalışma zamanında | çalışma zamanında | çalışma zamanında | **graf kurulurken** (`validate_graph`) |
| **runtime** | **aktör modeli, dağıtılabilir** | graf yürütücü | Python döngüsü | tur döngüsü | graf motoru |
| **durability** | zayıf | **checkpointer + time-travel** | katmanlı bellek | session | `Session` + takılabilir servis |
| **kod yürütme** | **birinci sınıf** | manuel | tool | tool | tool |
| **desen çeşitliliği** | **5+ takım tipi** | supervisor/swarm | sequential/hierarchical | tek | graf + Task API |
| **eval** | ayrı araç (AutoGenBench) | ayrı | ayrı | ayrı | **framework içinde** |
| **deploy** | senin işin | LangGraph Platform | — | — | Vertex/Cloud Run hazır |
| **öğrenme eğrisi** | orta-yüksek | yüksek | düşük | **en düşük** | orta |
| **yaşam döngüsü** | **bakım modu** | aktif | aktif | aktif | aktif, 2 haftada bir |

### İki ince nokta

**1. Graf konusunda taraflar yer değiştirdi.** AutoGen grafı `GraphFlow` olarak
**beş takım tipinden biri** diye ekledi. ADK ise grafı **merkeze** aldı ve konuşma
modelini hiç benimsemedi.

**2. ADK "oturmuş" değil.** ADK 2.0 kırıcı bir sürümdü ve 1.x hattı hâlâ paralel
sürüyor. "AutoGen kararsız, ADK stabil" demek yanlış olur — ADK bir yılda temel
API'sini değiştirdi.

---

## 6 — Ölçüm: desen seçiminin faturası

**[ölçüldü]** `poc/kiyas.py` — aynı görev, aynı ajanlar, yalnız orkestrasyon
deseni değişiyor:

| desen | mesaj | LLM | tool | token |
|---|---:|---:|---:|---:|
| **SelectorGroupChat** | 8 | 5 | 2 | **204** |
| GraphFlow | 11 | 7 | 3 | 270 |
| RoundRobinGroupChat | 9 | 6 | 2 | 274 |
| **Swarm** (handoff) | 14 | 7 | 4 | **334** |

**%63,7 fark.** Ödenen şey zekâ değil **yönlendirme özerkliği**:

- **Selector** yönlendirmeyi bir Python fonksiyonuyla yapıp gereksiz ajanı atlıyor
- **Swarm**'da kararı ajanın kendisi veriyor; her devir bir tool çağrısı + hiç iş
  üretmeyen bir LLM turu harcıyor
- **RoundRobin** kimseyi atlayamıyor

Bu tablonun framework karşılaştırmasına çevirisi şu: **Agents SDK'nın *tek* modeli
olan handoff, AutoGen'in en pahalı desenidir.** LangGraph'ın kenar modeli ise
GraphFlow'a denk.

> "Hangi framework daha iyi" sorusu aslında **"hangi yönlendirme maliyetini
> ödemeye razısın"** sorusu.

---

## 7 — Bu projede AutoGen neden kullanıldı

Dürüst sıralama — en güçlü gerekçeden en zayıfına:

**1. Araştırma projesinin konusu buydu.** Staj projesi "AutoGen'i öğren ve
kullan" diye başladı. Bu bir mühendislik gerekçesi değil ama en dürüst olanı, ve
saklanması anlamsız.

**2. Tek kütüphanede beş desen olması, ölçüm yapmayı mümkün kıldı.** %63,7'lik
tablo, beş deseni **aynı koşulda** koşturabildiğim için çıktı. Beş ayrı framework
kurmak gerekseydi, ölçülen şey desenin değil kurulumun farkı olurdu — "adil kıyas"
ilkesinin ihlali.

**3. `autogen_core` gerçekten farklı bir şey sunuyor.** Fan-in karşılaştırmasında
işe yarayan çözüm (pub/sub + `ClosureAgent` kuyruğu) AgentChat'te yok, core'da
var. Aynı kütüphanede iki katmanın olması, bir sorunu **bir kat aşağı inerek**
çözmemi sağladı.

**4. `ReplayChatCompletionClient` disiplini.** Anahtarsız, deterministik kuru mod
sayesinde sistem LLM gelmeden günlerce geliştirildi ve testler bugün hâlâ ağsız.
**[ölçüldü]** 52 test, 5,4 saniye.

**5. Gözlemlenebilirlik yüzeyi geniş.** Olay akışı, `InterventionHandler`,
`save_state` — bunların hepsini kullandık.

### Karşı argüman (ve neden yine de devam ediyoruz)

En güçlü karşı argüman: **AutoGen bakım modunda.** Yeni özellik gelmeyecek, ve
aşağıdaki §9'da somut bedelini ölçtük.

Devam etme gerekçesi: bu bir **araştırma projesi**, üretim taahhüdü değil. Ve
projenin tezi zaten "AutoGen'in yükselişi ve kapanışı" — kapanmış olması konuyu
bozmuyor, **çerçeveliyor**. Üretim sistemi kuruyor olsaydık cevap farklı olurdu:
Microsoft Agent Framework ya da ADK.

---

## 8 — Ne zaman AutoGen kullanılmaz

Bir aracı savunmanın en zayıf yolu, hiç kullanılmayacağı durumu yazmamak.

| Durum | Kullanılacak | Neden |
|---|---|---|
| **Bugün üretime çıkacak yeni sistem** | Microsoft Agent Framework | AutoGen bakım modunda; MAF resmî halefi, bugün sürüm çıkarmış [kaynak] |
| **Akışın önceden bilindiği, öngörülebilirlik istenen iş** | LangGraph veya ADK | Graf derleme zamanında doğrulanıyor; AutoGen'de hata çalışma zamanında |
| **Durum kalıcılığı ve "geri sar" gereken iş** | LangGraph | Checkpointer + time-travel; AutoGen'in `save_state`'i bunun yanında ilkel |
| **Tek ajan + birkaç tool, hızlı teslim** | OpenAI Agents SDK | En düşük öğrenme eğrisi; AutoGen'in katmanları gereksiz yük |
| **Rol tabanlı basit iş bölümü, hızlı prototip** | CrewAI | Kavramsal olarak en yakın metafor, en az kod |
| **Google bulut yığını, hazır deploy ve eval** | ADK | Vertex/Cloud Run hedefleri ve framework içi eval hazır |
| **Kod yürütme merkezdeyse** | AutoGen (hâlâ) | Kod yürütme birinci sınıf vatandaş; kurucu makalenin konusu |
| **Ajanları makinelere dağıtmak gerekiyorsa** | AutoGen (dikkatle) | Aktör modeli + gRPC; ama §4'teki ölçüm hatırlanmalı |

**Basit karar kuralı:** akışını **önceden çizebiliyorsan** graf tabanlı bir
framework al (LangGraph/ADK) — daha erken hata verir. Akışın **konuşmadan doğması**
gerekiyorsa AutoGen ailesine bak. Tek bir ajan yetiyorsa hiçbirini alma.

---

## 9 — "Bakım modu" ne demek: ölçülmüş bedeli

**[kaynak]** AutoGen README'sinin kendi rozeti ve cümlesi:

> ⚠️ **Maintenance Mode** — AutoGen is now in maintenance mode. It will not
> receive new features or enhancements and is community managed going forward.

Bu soyut bir etiket değil. **[ölçüldü]** somut bedeli:

`autogen-ext 0.7.5` bağımlılığını `mcp>=1.11.0` diye yazmış — **üst sınır yok.**
MCP SDK 2.0 çıkınca kurulum:

```
ImportError: cannot import name 'RequestContext' from 'mcp.shared.context'
```

Aynı gün `google-adk` aynı bağımlılığı `mcp>=1.24,<2` diye sınırlamış.

> **Aktif proje üst sınır koyar, bakım modundaki koymaz — ve düzeltecek kimse
> yoktur.** Yeni özellik gelmemesi bir şey; **var olanın çürümesi** başka bir şey.

`requirements.txt`'teki `mcp>=1.24,<2` pini bu yüzden zorunlu, ve bu tür pinlerin
sayısı zamanla **artacak**.

---

## 10 — İsim karışıklığı: AutoGen / AG2 / MAF

Bu dört isim sürekli karıştırılıyor. **[kaynak]** bugünkü durum:

| İsim | Ne | Durum |
|---|---|---|
| **microsoft/autogen** (v0.4+) | `autogen-agentchat` + `autogen-core` + `autogen-ext`. Bu projenin kullandığı | **Bakım modu**, v0.7.5 |
| **AutoGen v0.2** | Eski API: `ConversableAgent`, `initiate_chat` | Terk edilmiş; `0.2` dalında |
| **ag2ai/ag2** | v0.2 kolundan ayrılan fork, ayrı ekip | Aktif, v1.0.1. **v1.0'da `autogen` isim alanını "AG2 Classic"e taşıdı** — `pip install ag2` artık `import autogen` sunmuyor |
| **microsoft/agent-framework** | AutoGen + Semantic Kernel birleşimi, resmî halef | Aktif, bugün sürüm çıktı. AutoGen'den **göç kılavuzu** var |

**Pratik uyarı:** Bir tutorial'da `ConversableAgent` ya da `initiate_chat`
görüyorsan, o kaynak **v0.2 ya da AG2 Classic** anlatıyor ve bu projeyle uyumsuz.
Bu, hazır kurs değerlendirirken uyguladığım ilk filtre oldu.

---

## 11 — Sektörün yönü: tek bir şirketin kararı değil

Karşılaştırmanın en güçlü çıkarımı bu.

**[kaynak]** İki büyük satıcı, **bağımsız olarak**, aynı yöne gitti:

- **Microsoft**, AutoGen'i (konuşma-merkezli, özerk) bakım moduna aldı ve
  Semantic Kernel ile birleştirip **Agent Framework**'ü çıkardı
- **Google**, ADK'yı 1.x'in "workflow-agent first" hiyerarşisinden 2.x'in
  **graf-native** modeline taşıdı — ve bunu kırıcı bir sürümle yaptı

Yani ikisi de **konuşma-merkezli özerklikten yapılandırılmış akışa** geçti.

Ve buraya §2'deki cümle geri geliyor. AutoGen v0.4, kendi ifadesiyle
"observability, flexibility, interactive control, and scale" için sıfırdan
yazılmıştı. Dördünü de kazandı — **ve yine de yetmedi.** Üretimin istediği
diğer şeyler vardı: durum kalıcılığı, tip güvenliği, öngörülebilir maliyet,
derleme zamanında doğrulama.

**[ölçüldü]** Bunu kendi ölçümlerim de destekliyor: aynı görevde desen seçimi
%63,7 token farkı yaratıyor, ve bir dalın çökmesi kaç kardeşi götüreceği
deterministik değil. Esnekliğin faturası **öngörülemezlik**.

Destekleyen bağımsız kanıt: **[kaynak]**
[Why Do Multi-Agent LLM Systems Fail? (2503.13657)](https://arxiv.org/abs/2503.13657),
NeurIPS 2025 — 7 framework, 1600+ trace. Hataların ~%42'si sistem tasarımı,
~%37'si koordinasyon, ~%21'i doğrulama kaynaklı. Yani **model kalitesinden değil,
harness tasarımından.**

### Projenin tez cümlesi

> **AutoGen'in diğerlerinden farkı, sonunda onun kapanış sebebi oldu: araştırma
> framework'ü olarak kazandığı esneklik, üretim framework'ü olarak kaybettiği
> öngörülebilirlikti.**

---

## 12 — Özet: üç cümlede

1. **AutoGen'i ayıran şey `AssistantAgent` değil, `autogen_core`.** Aktör modeli,
   çoğu rakipte karşılığı olmayan bir katman — ve bu projede işe yaradı. Ama
   vaadi tam tutmuyor, ölçtük.

2. **Framework seçmek metafor seçmektir.** Akışını önceden çizebiliyorsan graf
   tabanlı olan daha erken hata verir; akış konuşmadan doğuyorsa AutoGen ailesi.
   Tek ajan yetiyorsa hiçbirini alma.

3. **Yıldıza değil, son commit tarihine bak.** Listedeki en yıldızlı iki proje
   en az hareket edenler; AutoGen ikinci sırada ama Nisan'dan beri commit almamış.

---

## Kaynaklar

**Birincil (bugün doğrulandı)**
- [microsoft/autogen](https://github.com/microsoft/autogen) — README bakım modu rozeti
- [Migration Guide v0.2 → v0.4](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html) — yeniden yazımın resmî gerekçesi
- [microsoft/agent-framework](https://github.com/microsoft/agent-framework) — AutoGen'den göç kılavuzu
- [ag2ai/ag2](https://github.com/ag2ai/ag2) — v1.0'da `autogen` isim alanı ayrımı
- [google/adk-python](https://github.com/google/adk-python) — v2 graf-native
- Repo sinyalleri: `gh api repos/<owner>/<repo>` ve `/releases/latest`

**Tam metinler (bu repoda)**
- [05 — Core User Guide](05-autogen-core-user-guide.md) — 42 sayfa
- [08 — AgentChat User Guide](08-autogen-agentchat-user-guide.md) — 25 sayfa

**Makaleler**
- [AutoGen (2308.08155)](https://arxiv.org/abs/2308.08155) — kurucu makale
- [Magentic-One (2411.04468)](https://arxiv.org/abs/2411.04468)
- [Why Do Multi-Agent LLM Systems Fail? (2503.13657)](https://arxiv.org/abs/2503.13657) — MAST

**Bu projenin ölçümleri**
- `poc/kiyas.py` — beş desen, %63,7 token farkı
- `poc/desen_5_core_aktor.py` — aktör modelinin sessiz veri kaybı
- `pipeline/compare_fanin.py` — iki fan-in motoru, aynı arıza
- [06 — Pratikte ısıran incelikler](06-autogen-incelikleri.md) — 13 madde

---

## Ek — Bu belgede ölçülmemiş olanlar

Dürüstlük borcu:

- **LangGraph, CrewAI, Agents SDK, MetaGPT koşturulmadı.** Mimari iddialar
  [teyitsiz]. Aynı görevi bu framework'lerle kurup `poc/kiyas.py` tablosuna
  satır eklemek, karşılaştırmayı ölçüme çevirir — ve senin kendi dersin gereği
  ("bir şeyi kıyasa sokmak, kıyasın kendisindeki hataları açığa çıkarır") muhtemelen
  mevcut tablodaki bir hatayı da bulur.
- **Microsoft Agent Framework denenmedi.** Halef olduğu için en değerli
  karşılaştırma bu olurdu: aynı görev, AutoGen ve MAF ile yan yana.
- **Maliyet karşılaştırması tek göreve dayanıyor.** %63,7 farkı bir görevden
  çıktı; farklı görev profillerinde oran değişir.
- **Uç noktadaki diğer modeller adil ölçülmedi.** İlk ölçümde muhakeme yapan
  modellere kısa token bütçesi verdim; ölçtüğüm şey yetenek değil bütçe olmuş
  olabilir.
