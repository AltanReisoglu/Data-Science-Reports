# Feature Specification: PTC Kod Sandbox'ı (Faz 2)

**Feature Branch**: `002-ptc-code-sandbox`

**Created**: 2026-08-27

**Status**: Draft

**Input**: User description: "Faz 2: Asistana gerçek PTC (Programmatic Tool Calling) yeteneği ekle. Model, tool'ları tek tek çağırmak yerine bir Python kod bloğu yazıp bunu bir sandbox içinde çalıştırarak tool'ları programatik olarak orkestre edebilecek. Kendi sandbox'ımızı inşa edeceğiz (Anthropic'in 'self-managed sandboxed execution' deseni + Flyte/Monty'nin 'deny-by-default capability grant' felsefesi): kod ayrı bir subprocess'te çalışır, networking/dosya-sistemi import edilemez, dışarı çıkmanın tek yolu enjekte edilen tool-proxy fonksiyonlarıdır (Faz 1'deki search_knowledge_base, get_ticket_status, list_open_tickets), timeout ile korunur."

## Background & Motivation

Faz 1'de (`specs/001-ptc-grounded-assistant/`) kurulan kurumsal asistan, tool'larını klasik/doğrudan tool-calling ile çağırıyor: her tool çağrısı ayrı bir model turu gerektiriyor. Bu, deponun asıl araştırma sorusunu (`docs/topic_is_this.md` — PTC egress-policy: "ajana sınırsız erişim yerine yalnızca onaylı tool kanalları") henüz sınamıyor.

Bu özellik, Faz 1'in üzerine **gerçek PTC (Programmatic Tool Calling)** yeteneğini ekliyor: model, birden fazla tool çağrısını, koşulu, filtrelemeyi tek bir Python kod bloğu içinde orkestre edebiliyor — kod bir sandbox'ta çalışıyor, sadece nihai sonuç modelin bağlamına giriyor. Anthropic'in kendi native PTC'si (Claude API'sine özgü) burada kullanılamıyor çünkü asistanın modeli (Faz 1'de seçilen, OpenAI-uyumlu bir gateway üzerinden erişilen model) Anthropic'e bağlı değil — bu yüzden kendi sandbox'ımız inşa ediliyor.

**Mimari netleştirmesi (2026-08-27)**: Sandbox, ana kurumsal asistandan ayrı, her PTC çalıştırması için ayağa kaldırılan **bir Kubernetes pod'udur** (laptop'ta `kind` ile yerel bir cluster). Bu pod'un gerçek ağ yeteneği vardır — ama dış erişimi **eBPF/Cilium** ile kernel seviyesinde, yalnızca onaylı Tool Gateway/API hedefleriyle sınırlanır (bkz. `PTC_Egress_Policy_Cilium_eBPF_Technical_Reference.md`). Yani "ağ yeteneği yok" değil, "ağ yeteneği var ama daraltılmış" — tam olarak `docs/topic_is_this.md`'nin tanımı: *"Sandbox veya agent ortamlarının dış ağ erişiminin eBPF/Cilium ile merkezi olarak kontrol edilmesi... Sadece onaylı tool/API kanallarına erişim verilerek..."* Bu, sandbox'ın **onaylı-kanal dışına asla çıkamaması**nı sağlamayı hedefleyen, deponun kök amacına en yakın fazdır — ve Faz 3 ile Faz 2 bu noktada iç içe geçer (aşağıdaki Assumptions'a bkz.).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Kod yazarak çoklu-adımlı görev orkestre etme (Priority: P1) 🎯 MVP

Kullanıcı, birden fazla tool çağrısı ve veri işleme gerektiren bir soru sorar (ör. "4 kaynaktaki tüm dokümanları tara ve X konusunu geçen kaçını bul"). Asistan, bu görevi tek tek tool çağırıp her seferinde modele dönmek yerine, bir Python kod bloğu yazıp sandbox'ta çalıştırarak (döngü, filtreleme, birden fazla tool çağrısı kodun içinde) tek bir özet sonuçla tamamlar.

**Why this priority**: Bu, PTC'nin çekirdek değer önermesi — kodun kendisi çalışmazsa hiçbir şey yok. Diğer her şey bunun üzerine kurulu.

**Independent Test**: Faz 1'deki tool'ları (Foundational'da zaten var) kullanan, en az 2 tool çağrısı gerektiren bir görev verilip, tek bir sandbox çalıştırmasıyla (modelin ayrı ayrı her tool için tekrar çağrılmasına gerek kalmadan) doğru sonucun üretildiği doğrulanabilir.

**Acceptance Scenarios**:

1. **Given** birden fazla tool çağrısı gerektiren bir görev, **When** asistan bir orkestrasyon kodu yazıp çalıştırır, **Then** kod, gerekli tool-proxy fonksiyonlarını (kaç tane gerekiyorsa) çağırıp sonuçları işleyip tek bir özet döndürür.
2. **Given** sandbox çalışması tamamlanmış, **When** sonuç modele döner, **Then** yalnızca özet/nihai veri modelin bağlamına girer — ara sonuçlar (ör. ham doküman listeleri) modele taşınmaz.

---

### User Story 2 - Onaylı-kanal dışına çıkışın engellenmesi (Priority: P2)

Bir güvenlik gözden geçireni, sandbox'ın gerçekten yalnızca onaylı tool-proxy fonksiyonları üzerinden dışarı çıkabildiğini, doğrudan ağ veya dosya sistemi erişimi olmadığını doğrulamak ister.

**Why this priority**: Bu fazın var oluş nedeni budur (Principle II) — orkestrasyon çalışıyor olsa bile bu sağlanmazsa PoC'nin tezi geçersiz kalır. P1'den ayrı test edilebilir olduğu için P2.

**Independent Test**: Sandbox'a, onaylı tool-proxy'ler dışında bir şeye (ör. doğrudan bir ağ bağlantısı açmaya, keyfi bir dosya okumaya, izinsiz bir modül import etmeye) çalışan bir kod verilip bunun her seferinde engellendiği doğrulanabilir.

**Acceptance Scenarios**:

1. **Given** sandbox'ta çalışan kod, onaylı tool-proxy fonksiyonları dışında bir şeye erişmeye çalışır, **When** bu girişim gerçekleşir, **Then** erişim engellenir, gerçek bir ağ/dosya işlemi asla yapılmaz.
2. **Given** engellenen bir erişim girişimi, **When** bu olay gerçekleşir, **Then** izlenebilirlik kaydına (Faz 1'deki Trace) "reddedilen eylem" olarak açıkça yazılır.

---

### User Story 3 - Zaman aşımı ve hatanın zarifçe ele alınması (Priority: P3)

Sandbox'ta çalışan kod sonsuz döngüye girer, çöker veya beklenmedik bir hata verirse, asistan çökmeden bunu açık bir hata/zaman-aşımı sonucu olarak ele alır ve kullanıcıya tahmini bir değer sunmaz.

**Why this priority**: Değerli bir sağlamlık katmanı ama P1/P2 çalışıyorsa bağımsız olarak eklenip test edilebilir, bu yüzden en düşük öncelik.

**Independent Test**: Bilerek sonsuz döngüye giren veya hata fırlatan bir kod sandbox'a verilip, asistanın (a) çökmediği, (b) makul bir sürede sonlandırıldığı, (c) kullanıcıya "erişilemedi/tamamlanamadı" dediği, tahmini bir sonuç üretmediği doğrulanabilir.

**Acceptance Scenarios**:

1. **Given** sonsuz döngüye giren bir sandbox kodu, **When** yapılandırılmış zaman aşımı süresi dolar, **Then** çalıştırma sonlandırılır, asistan çökmez, kullanıcıya açık bir "tamamlanamadı" yanıtı döner.
2. **Given** sandbox kodu beklenmedik bir hata fırlatır, **When** bu olur, **Then** hata izlenebilirlik kaydına yazılır ve asistan tahmini bir değer üretmez.

---

### Edge Cases

- Sandbox kodu, granted olmayan bir tool-proxy adını (yanlış yazım veya kasıtlı) çağırmaya çalışırsa ne olur? → Açık bir hata (ör. tanımsız isim), reddedilen eylem olarak kaydedilir, gerçek bir erişim asla verilmez.
- Sandbox process'i beklenmedik şekilde çökerse (OOM, sinyal) ne olur? → Hata sonucu olarak ele alınır, kaydedilir; model kısmi/tamamlanmamış sonuca güvenmemesi gerektiğini bilir.
- Aynı anda birden fazla sandbox çalışması (çoklu kullanıcı/oturum) olursa ne olur? → Her çalışma birbirinden izole olmalı, aralarında durum sızıntısı olmamalı.
- Sandbox'ın çağırdığı gerçek tool (ör. canlı sistem) zaman aşımına uğrarsa/hata verirse ne olur? → Faz 1'deki "erişilemedi, tahmini değer yok" davranışı (FR-011, US2) sandbox içinden çağrılan koda da aynen yansır — sandbox kodu bunu net bir hata/işaret olarak alır, uydurmaz.
- Sandbox'a bugün onaylı olmayan yeni bir tool eklemek istenirse ne olur? → **Çözüldü (sektör taramasıyla, 2026-08-27)**: Faz 2, Faz 1'in doğrudan tool-calling'ini KALDIRMAZ — ikisi bir arada var olur. Hangi tool'un hangi modda (doğrudan mı, sandbox-orkestrasyonu mu) çağrılabileceği geliştirme zamanında (harness tasarımı) sabit olarak belirlenir; model bunu görev bazında kendisi seçmez (Anthropic'in `allowed_callers`'ı her tool için tek mod öneriyor; LangChain'in `CodeInterpreterMiddleware(ptc=[...])`'i de allowlist'i geliştirici belirliyor).
- Sandbox içinden yapılan bir tool-proxy çağrısı, Faz 1'deki Tool Gateway'in (HumanInTheLoopMiddleware) her-çağrı onay/red mekanizmasından tekrar geçer mi? → **Çözüldü, mimari netleştirmesiyle güncellendi**: Sandbox artık ayrı bir pod olduğu için LangGraph'ın HITL middleware'i (ana asistan process'inde yaşıyor) sandbox'ın içine doğal olarak uzanmıyor. Bunun yerine onay iki farklı seviyede uygulanır: (1) **ağ seviyesi, bir kere** — Cilium, pod'un Tool Gateway'e ulaşıp ulaşamayacağına kapsam/pod-yaşam-döngüsü başına karar verir; (2) **istek seviyesi, her çağrıda** — Tool Gateway servisinin kendisi, gelen her isteği kendi yetkilendirme mantığıyla (mevcut `tool_policy.ALLOWED_TOOLS`'un servis-tarafı karşılığı) kontrol eder ve izlenebilirlik kaydına yazar. Yani "bir kere mi her seferinde mi" sorusu artık "hangi seviyede" sorusuna dönüşüyor — ikisi de var, farklı katmanlarda.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Sistem, asistanın birden fazla tool çağrısını, koşulu ve veri işlemeyi tek bir Python kod bloğu içinde orkestre etmesine izin vermeli — her tool çağrısı için ayrı bir model turu gerektirmeden.
- **FR-002**: Sistem, bu kodu ana uygulamadan ayrı, her çalıştırma için ayağa kaldırılan bir Kubernetes pod'unda (laptop'ta yerel bir cluster üzerinde) çalıştırmalı.
- **FR-003**: Sandbox pod'unun dış ağ erişimi, eBPF/Cilium ile kernel/network seviyesinde, yalnızca onaylı Tool Gateway/API hedefleriyle (belirli IP/FQDN + port) sınırlı olmalı; bunun dışındaki her hedef (internet dahil) reddedilmeli — pod'un ağ yeteneğinin kendisi kaldırılmaz, sadece hedefi daraltılır.
- **FR-004**: Sandbox pod'u, kendi çalışması için gerekenin ötesinde keyfi dosya sistemi erişimine sahip olmamalı.
- **FR-005**: Sandbox kodunun dışarıyla (herhangi bir gerçek sisteme) etkileşiminin TEK yolu, o çalıştırma için enjekte edilmiş, önceden onaylı tool-proxy fonksiyonları olmalı (Faz 1'deki `search_knowledge_base`, `get_ticket_status`, `list_open_tickets`).
- **FR-006**: Sistem, sandbox kodunun çalışma süresine bir üst sınır (timeout) uygulamalı; bu süre aşıldığında çalıştırma sonlandırılmalı.
- **FR-007**: Sandbox kodu beklenmeyen bir hata verdiğinde veya zaman aşımına uğradığında, sistem bunu asistanı çökertmeden açık, sınırlı bir hata/zaman-aşımı sonucu olarak ele almalı.
- **FR-008**: Her sandbox çalıştırması (başlangıç, çalıştırma sırasında yapılan her tool-proxy çağrısı, nihai sonuç: başarı/hata/zaman-aşımı/reddedilen-eylem) Faz 1'deki izlenebilirlik mekanizmasına (Trace) kaydedilmeli (Principle III).
- **FR-009**: Bir sandbox çalıştırmasında mevcut olan tool-proxy fonksiyonlarının kümesi, o oturum için Faz 1'in Tool Gateway politikasınca (`tool_policy.ALLOWED_TOOLS`/`LOCAL_TOOLS`) zaten onaylanmış olanlarla birebir sınırlı olmalı — sessizce ek/farklı bir yetenek tanıtılamaz.
- **FR-010**: Sandbox kodu onaylanmamış bir yetenek kullanmaya çalıştığında (izinsiz modül import etme, soket açma vb.), sistem bunu gerçek bir erişim vermeden engellemeli ve bu girişimi izlenebilirlik kaydına reddedilen eylem olarak yazmalı.
- **FR-011**: Sandbox çalıştırması sonrasında üretilen nihai yanıt, Faz 1'in temel zemine-dayalılık kuralına (Principle I) uymalı — olgusal iddialar, fiilen gerçekleşen tool-proxy çağrı sonuçlarına izlenebilir olmalı, uydurulmamalı.

### Key Entities *(include if feature involves data)*

- **Sandbox Çalıştırması (SandboxRun)**: LLM'in ürettiği bir kod bloğunun tek bir yürütülmesi; çalıştırma kimliği, kod, başlangıç/bitiş zamanı, nihai sonuç (başarı/hata/zaman-aşımı/reddedilen-eylem) ve bu çalıştırma sırasında yapılan tool-proxy çağrılarının listesini taşır.
- **Yetenek Tanımı (Capability Grant)**: Bir sandbox çalıştırmasına enjekte edilen tool-proxy fonksiyonlarının kümesi; Faz 1'in onaylı tool listesine referans verir.
- **Reddedilen Eylem (Denied Action)**: Sandbox kodunun onaylanmamış bir yetenek kullanmaya çalıştığı, engellenen bir girişimin kaydı — hangi eylem, ne zaman, hangi SandboxRun'da.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Birden fazla tool çağrısı ve veri işleme gerektiren temsili bir görevde, sandbox tabanlı orkestrasyon, eşdeğer adım-adım tool-calling yaklaşımına (Faz 1) göre en az %50 daha az model turu gerektirir.
- **SC-002**: Kontrollü bir test setinde, sandbox kodunun onaylı tool-proxy'ler dışına çıkma girişimlerinin (ağ, dosya sistemi, izinsiz import) %100'ü engellenir — sıfır başarılı kaçış.
- **SC-003**: Her sandbox çalıştırması (başarı, hata, zaman aşımı veya reddedilen-eylem sonucu) izlenebilirlik kaydında görünür ve mevcut `--trace` mekanizmasıyla erişilebilir.
- **SC-004**: Yapılandırılmış zaman aşımını aşan bir sandbox çalıştırması, öngörülebilir bir sürede (zaman aşımı eşiğinden birkaç saniye içinde) sonlandırılır, süresiz asılı kalmaz.
- **SC-005**: Sandbox tabanlı bir çalıştırma sonucu üretilen yanıttaki her olgusal iddia, bir tool-proxy çağrı sonucuna izlenebilir — denetim örnekleminde sıfır uydurma iddia (Faz 1'in SC-001'iyle aynı çıta, Faz 2'ye genişletilmiş).

## Assumptions

- Faz 2, Faz 1'in onaylı tool setini (`search_knowledge_base`, `get_ticket_status`, `list_open_tickets`) doğrudan kullanır; bu fazda yeni bir dış tool tanıtılmaz.
- Hedeflenen izolasyon seviyesi, "PTC + onaylı-kanal desenini araştırma/sergileme amacıyla yeterli" düzeydedir — kararlı, kaynaklı bir saldırgana karşı sertifikalı bir izolasyon (tam container/VM seviyesi) bu PoC'nin kapsamı dışındadır (Principle V). Tehdit modeli değişirse bu varsayım yeniden değerlendirilmeli.
- Sandbox dili Python'dur (asistanın kendi implementasyon diliyle ve LLM'lerin kod üretme gücüyle tutarlı).
- Hafıza erişim yolu bu fazda da kapsam dışıdır (Faz 1 kararı değişmedi).
- UI bu fazın kapsamında değildir (Faz 4).
- Faz 2'nin izolasyon mekanizmasının kendisi eBPF/Cilium'dur (Altan'ın 2026-08-27 kararı) — Faz 3 ile ayrılmıyor, iç içe. Yerel PoC ortamı: laptop'ta `kind` (Kubernetes in Docker) ile tek makinelik bir cluster + Cilium (Helm ile kurulu) — bkz. `PTC_Egress_Policy_Cilium_eBPF_Technical_Reference.md` §5.4-5.5.
- Sandbox pod'unun çağırdığı Tool Gateway, artık gerçek bir ağ hedefi (IP/FQDN + port) olmak zorunda — yani Faz 1'deki gibi salt in-process Python fonksiyonları değil, ayrı bir servis/pod olarak çalışan bir Tool Gateway gerekiyor. Bu servisin kendi yetkilendirme mantığı (mevcut `tool_policy.ALLOWED_TOOLS`'un servis-tarafı karşılığı) planlama aşamasında netleştirilecek.
- Zaman aşımı süresi ve olası CPU/bellek sınırları gibi somut sayısal değerler, planlama aşamasında (teknoloji seçimiyle birlikte) belirlenecek; bu spesifikasyon yalnızca "bir üst sınır olmalı" iş kuralını sabitler.
