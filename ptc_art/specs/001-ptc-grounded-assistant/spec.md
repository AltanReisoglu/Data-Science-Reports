# Feature Specification: Kurumsal Zemine-Dayalı Asistan (PTC Grounded Assistant PoC)

**Feature Branch**: `001-ptc-grounded-assistant`

**Created**: 2026-08-27

**Status**: Draft

**Input**: User description: "bu pocde bir kurumsal asistan yapacağız. temel görevi, kullanıcı isteklerini uydurmadan, araçlardan gelen gerçek veriye dayandırarak yanıtlamaktır. üç erişim yolu üzerinde çalışır: kurumsal bilgi bankası (4 paralel kaynak), canlı sistemler (skill+tool mekanizması) ve hafıza (kullanıcıya özel kalıcı bilgi). Bu PoC'nin asıl amacı bir kurumsal ajanın PTC (programmatic tool calling) özelliğini, docs/topic_is_this.md konusundaki onaylı-kanal / egress prensibi bağlamında sınamaktır."

## Background & Motivation

Bu özellik, `docs/topic_is_this.md` ve ilgili PTC (Programmatic Tool Calling) egress-policy dokümanlarında tarif edilen prensibi iş değeri üzerinden test eden bir kanıt (PoC) niteliğindedir:

> Ajana sınırsız/doğrudan erişim vermek yerine, dış dünyayla her etkileşim yalnızca onaylı tool/skill kanalları üzerinden gerçekleşir.

Bu PoC, o prensibin kısıtlayıcı olmadığını — aksine kurumsal bir asistanın güvenilirliğinin (uydurmama, kaynağa dayanma) temelini oluşturduğunu göstermeyi amaçlar. Asistanın her yanıtı, üç tanımlı erişim yolundan biri veya birkaçı üzerinden **fiilen çağrılmış bir araçtan** gelen veriye dayanmalıdır; araç çağrılmadan/veri dönmeden üretilen hiçbir iddia kabul edilemez.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Kurumsal bilgi bankasından zemine dayalı yanıt (Priority: P1)

Bir çalışan, şirket içi politika/prosedür/doküman bilgisi gerektiren bir soru sorar (örn. "Uzaktan çalışma politikamız nedir?"). Asistan, kurumsal bilgi bankasını oluşturan 4 paralel kaynağın tümünü sorgular, dönen sonuçları birleştirir ve yanıtı hangi kaynak(lar)dan geldiğini belirterek sunar.

**Why this priority**: Bu, "uydurmadan yanıtlama" temel misyonunun en sık karşılaşılacak ve en kolay doğrulanabilir senaryosudur; PoC'nin çekirdek değer önermesini tek başına kanıtlar.

**Independent Test**: Bilgi bankasında karşılığı olan bir soru sorularak, yanıtın en az bir kaynağa atıfla geldiği ve kaynaklardan biri/birkaçı boş sonuç döndürse bile genel yanıtın bozulmadığı doğrulanabilir.

**Acceptance Scenarios**:

1. **Given** sorunun cevabı 4 kaynaktan en az birinde mevcut, **When** çalışan soruyu sorar, **Then** asistan tüm 4 kaynağı paralel sorgular ve yanıtta hangi kaynak(lar)ın kullanıldığını belirtir.
2. **Given** 4 kaynaktan biri geçici olarak yanıt vermiyor, **When** çalışan soruyu sorar, **Then** asistan kalan kaynaklardan elde ettiği veriyle yanıt verir ve ulaşılamayan kaynağı açıkça belirtir.
3. **Given** 4 kaynağın hiçbiri soruyla ilgili veri döndürmüyor, **When** çalışan soruyu sorar, **Then** asistan bilgi bulunamadığını açıkça söyler ve konuyla ilgili bir iddia üretmez.

---

### User Story 2 - Canlı sistemlerden güncel veriyle yanıt (Priority: P2)

Bir çalışan, anlık/güncel durum bilgisi gerektiren bir soru sorar (örn. "Şu an açık kritik ticket sayısı kaç?"). Asistan, tanımlı skill/tool mekanizması üzerinden ilgili canlı sisteme erişir ve güncel veriyi zaman damgasıyla birlikte sunar.

**Why this priority**: Kurumsal bilgi bankasından farklı olarak, bu senaryo "veri her an değişebilir, statik değildir" durumunu kapsar ve PTC prensibinin (yalnızca onaylı tool kanalıyla dış sisteme erişim) en somut şekilde sınandığı yerdir.

**Independent Test**: Canlı bir sistemden anlık değer gerektiren bir soru sorularak, yanıtın ilgili tool/skill çağrısı üzerinden alınan güncel veriye dayandığı ve yanıtta bir zaman/güncellik göstergesi bulunduğu doğrulanabilir.

**Acceptance Scenarios**:

1. **Given** canlı sistem erişilebilir durumda, **When** çalışan güncel duruma ilişkin bir soru sorar, **Then** asistan ilgili skill/tool'u çağırır ve dönen veriyi zaman damgasıyla yanıtlar.
2. **Given** canlı sistem geçici olarak erişilemez/zaman aşımına uğruyor, **When** çalışan soru sorar, **Then** asistan erişim sağlanamadığını açıkça belirtir ve tahmini/uydurma bir değer vermez.

---

### User Story 3 - Kişiye özel hafızadan bağlam kullanma (Priority: P3)

Daha önce sistemle etkileşmiş bir çalışan, önceki oturumlarda belirttiği tercih/bağlamı tekrar belirtmek zorunda kalmadan asistanın bunu hatırlamasını ve yanıtı buna göre kişiselleştirmesini bekler. Hafızadan gelen bilgi, doğrulanmış bir gerçek olarak değil, kişiselleştirme bağlamı olarak kullanılır.

**Why this priority**: Değer katan ama çekirdek "uydurmama" misyonu için zorunlu olmayan bir katmandır; P1 ve P2 olmadan da bağımsız şekilde değerlendirilebilir, bu yüzden en düşük öncelikte yer alır.

**Independent Test**: Bir kullanıcı tercihini önceki bir oturumda kaydettirip yeni bir oturumda aynı tercihi tekrar belirtmeden ilgili soruyu sorarak, asistanın hafızadaki bilgiyi doğru hatırladığı ve bunu bir "doğrulanmış gerçek" gibi değil bağlam olarak sunduğu doğrulanabilir.

**Acceptance Scenarios**:

1. **Given** kullanıcı için önceden kaydedilmiş bir tercih/bağlam mevcut, **When** kullanıcı yeni bir oturumda ilgili bir soru sorar, **Then** asistan hafızadaki bilgiyi tercihi tekrar sormadan kullanır.
2. **Given** hafızadaki bir bilgi ile bilgi bankası/canlı sistemden gelen güncel veri çelişiyor, **When** asistan yanıt üretir, **Then** araçtan gelen güncel veriyi esas alır ve hafızadaki bilgiyi yalnızca bağlam olarak belirtir.

---

### Edge Cases

- Üç erişim yolunun (bilgi bankası, canlı sistem, hafıza) hiçbiri soruyla ilgili veri döndürmezse ne olur? → Asistan açıkça "zemine dayalı yanıt bulunamadı" der, spekülasyon yapmaz.
- 4 paralel kaynaktan biri ya da birkaçı hata/zaman aşımı verirse yanıt nasıl etkilenir? → Kalan kaynaklardan yanıt üretilir, eksik kaynak açıkça belirtilir.
- Kullanıcı, onaylı hiçbir tool/skill kanalının erişemeyeceği bir bilgi isterse (kapsam dışı/onaysız kanal gerektiren) ne olur? → Asistan bu isteği karşılayamayacağını belirtir, onaysız bir kanaldan veri çekmeye veya tahmin üretmeye çalışmaz.
- Hafızadaki bir bilgi ile araçtan gelen gerçek veri çelişirse hangisi önceliklidir? → Araçtan gelen veri (US3, senaryo 2).
- Aynı soruya birden fazla erişim yolu (örn. hem bilgi bankası hem canlı sistem) yanıt verebiliyorsa nasıl önceliklendirilir? → Bu önceliklendirme kuralı henüz kararlaştırılmamıştır; planlama aşamasında netleştirilecek açık bir karardır (bkz. Assumptions).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Sistem, kullanıcı sorularını YALNIZCA üç tanımlı erişim yolu (kurumsal bilgi bankası, canlı sistemler, hafıza) üzerinden fiilen çağrılan araçlardan elde edilen veriye dayanarak yanıtlamalıdır; araç çağrısına dayanmayan hiçbir olgusal iddia üretilemez.
- **FR-002**: Sistem, ilgili bir soru geldiğinde kurumsal bilgi bankasını oluşturan 4 paralel kaynağın tümünü eşzamanlı (paralel) olarak sorgulamalıdır.
- **FR-003**: Sistem, 4 paralel kaynaktan dönen sonuçları tek, tutarlı bir yanıtta birleştirmeli ve yanıtta hangi kaynak(lar)ın katkı sağladığını belirtmelidir.
- **FR-004**: Sistem, canlı/güncel sistem verisine yalnızca tanımlı skill/tool mekanizması üzerinden erişmeli; doğrudan veya onaysız bir kanaldan veri çekmemelidir.
- **FR-005**: Sistem, kullanıcıya özel bilgiyi (tercih, bağlam) oturumlar arası kalıcı olacak şekilde saklamalı ve sonraki oturumlarda geri çağırabilmelidir.
- **FR-006**: Sistem, hafızadan gelen bilgi ile araç çağrısıyla doğrulanmış veriyi ayrıştırmalı; hafıza içeriğini, bir araçla doğrulanmadığı sürece kesin olgu gibi sunmamalıdır.
- **FR-007**: Üç erişim yolundan hiçbiri sorguyla ilgili veri döndürmediğinde, sistem bunu açıkça belirtmeli ve zemin bulunmayan bir yanıt üretmemelidir.
- **FR-008**: Sistem, tüm dış veri erişimini yalnızca her erişim yolu için tanımlanmış onaylı tool/skill kanalları üzerinden gerçekleştirmeli; onaysız/doğrudan bir kanaldan veri çekmeye çalışmamalıdır.
- **FR-009**: Sistem, üretilen her yanıt için hangi erişim yolu/yolları ve hangi kaynak(lar)ın kullanıldığını izlenebilir şekilde kaydetmelidir (denetlenebilirlik).
- **FR-010**: Sistem, 4 paralel kaynaktan bir veya daha fazlasının hata verdiği/zaman aşımına uğradığı durumlarda, kalan kaynaklardan elde edilen veriyle yanıt üretmeye devam etmeli ve eksik/başarısız kaynağı yanıtta açıkça belirtmelidir.
- **FR-011**: Sistem, kapsam dışı kalan veya yalnızca onaysız bir kanaldan erişilebilecek bir istek aldığında, bu isteği karşılayamayacağını belirtmeli ve alternatif/tahmini bir yanıt üretmemelidir.

### Key Entities *(include if feature involves data)*

- **Sorgu (Query)**: Kullanıcının sorduğu soru; kullanıcı kimliği/oturum bilgisi ve zaman damgası ile ilişkilidir.
- **Bilgi Bankası Kaynağı (Knowledge Base Source)**: Kurumsal bilgi bankasını oluşturan, birbirinden farklı 4 içerik deposundan her biri — politika/prosedür dokümanları, kurumsal wiki, destek talebi arşivi ve teknik dokümantasyon; her sorgu için ayrı bir erişim/sonuç durumu (başarılı, boş, hata) taşır.
- **Canlı Sistem Erişimi (Live Tool Call)**: Skill/tool mekanizması üzerinden yapılan bir canlı sistem çağrısı kaydı; zaman damgası, sonuç ve başarı/hata durumu içerir.
- **Hafıza Kaydı (Memory Record)**: Belirli bir kullanıcıya özel, oturumlar arası kalıcı tercih/bağlam bilgisi.
- **Yanıt (Answer)**: Kullanıcıya sunulan nihai cevap; katkı sağlayan erişim yolu/yolları ve kaynak(lar)a atıf ile, zemine-dayalı mı yoksa "veri bulunamadı" durumunda mı olduğu bilgisini taşır.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Yanıtlardaki olgusal iddiaların %100'ü, üç erişim yolundan en az birinden gelen bir kaynağa geri izlenebilir; hiçbir yanıt bir araç sonucuna dayanmayan bir iddia içermez.
- **SC-002**: Bir sorgu hiçbir erişim yolundan veri döndürmediğinde, sistem bunu vakaların %100'ünde açıkça belirtir (sessiz uydurma sıfır vakadır).
- **SC-003**: Bilgi bankası sorgularında, 4 kaynağın tamamı paralel olarak sorgulanır ve tipik bir sorgu için sonuçlar birleştirilmiş halde makul bir sürede (ör. birkaç saniye içinde) kullanıcıya sunulur.
- **SC-004**: Geri dönen kullanıcıların önceden kaydedilmiş tercihleri, takip eden oturumların %100'ünde, kullanıcı tekrar belirtmek zorunda kalmadan doğru şekilde hatırlanır.
- **SC-005**: Örnek bir denetimde (N yanıt üzerinden), bağımsız bir gözden geçiren kişi her olgusal iddiayı belirli bir araç/kaynak kaydına geri izleyebilir — bu, "zemine dayalılık" iddiasının gerçekten uygulandığını, yalnızca beyan edilmediğini gösterir.
- **SC-006**: Kapsam dışı/onaysız kanal gerektiren isteklerin %100'ünde sistem, isteği karşılayamayacağını belirtir ve tahmini bir yanıt üretmez.

## Assumptions

- Bu PoC, tek bir kurumsal ortam ve sınırlı sayıda temsili demo kullanıcı ile sınırlıdır; çok-kiracılı (multi-tenant) rol/izin yönetimi kapsam dışıdır.
- Kaynakların sayısının (4) ve bunların eşzamanlı/paralel sorgulanması gerekliliğinin sabit bir iş kuralı olduğu, kaynakların da politika/prosedür dokümanları, kurumsal wiki, destek talebi arşivi ve teknik dokümantasyon olduğu kararlaştırılmıştır.
- Canlı sistem ile bilgi bankasının aynı konuda çelişen/örtüşen veri döndürmesi durumunda hangi kaynağın esas alınacağı **açık bir karardır** ve `/speckit-plan` aşamasında, seçilecek protokol/mimariyle birlikte netleştirilecektir; bu spesifikasyon şimdilik yalnızca her iki durumun da (çelişki ve örtüşme) izlenebilir şekilde ele alınması gerektiğini sabitler.
- Hafızada saklanan kullanıcıya özel bilgi, tercih ve etkileşim bağlamıyla sınırlıdır; hassas kişisel veya kurumsal gizli bilginin hafızada saklanıp saklanamayacağı planlama aşamasında güvenlik/uyumluluk gereksinimleriyle birlikte netleştirilecektir.
- "Onaylı tool/skill kanalı" kavramı bu spesifikasyonda iş kuralı olarak sabittir (yalnızca onaylı kanaldan erişim); kanalların somut protokol/mimari uygulaması `/speckit-plan` aşamasında, kullanıcının belirteceği protokol/framework/kütüphane tercihlerine göre tasarlanacaktır — bu spesifikasyon herhangi bir teknoloji seçimi varsaymaz.
- Canlı sistemlerin "güncellik" beklentisi, standart bir kurumsal iç araç için makul kabul edilen birkaç saniyelik yanıt süresi olarak varsayılmıştır; kesin bir SLA belirtilmemiştir.
