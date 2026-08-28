# Feature Specification: Web Arayüzü + Canlı PTC İzleme Paneli (Faz 4)

**Feature Branch**: `003-web-ui-live-trace`

**Created**: 2026-08-28

**Status**: Draft

**Input**: User description: "Faz 4: Kurumsal asistan için bir web UI'ı. Kullanıcı, tarayıcıda soru sorup yanıt alabilecek (Faz 1/2'nin CLI'sinin ("assistant" komutu) web karşılığı — grounded=True/False, source_refs, partial_failure_notes hep gösterilmeli, Faz 1'in fabrikasyon-yok ilkesi UI'da da geçerli). Ekranın sol-alt köşesinde, gerçek zamanlı bir "PTC yaşam döngüsü" paneli olacak — terminal gibi, kayan bir log görünümünde: bir run_ptc_code çağrısı tetiklendiğinde, o sandbox çalıştırmasının TÜM adımları (ConfigMap yazılması, Job oluşturulması, LLM'in ürettiği kodun kendisi, sandbox içindeki her tool-proxy çağrısı, Hubble'dan gelen bir DeniedAction varsa o, nihai sonuç/hata/timeout) ANINDA bu panelde akmalı — kullanıcı bir PTC çalıştırmasının tüm hayatını canlı izleyebilmeli."

## Background & Motivation

Faz 1 ve Faz 2, kurumsal asistanı ve onun PTC (Programmatic Tool Calling) yeteneğini yalnızca CLI (`assistant` komutu) üzerinden erişilebilir kıldı. Bu, deponun 4 fazlı yol haritasının son adımı (Faz 4 — UI, önceki fazlarda kapsam dışı bırakılmıştı): aynı yetenekleri bir tarayıcı arayüzünden erişilebilir kılmak.

Bu fazın kendine özgü değeri sadece "CLI'nin web karşılığı" değil — PTC'nin (Faz 2'nin asıl tezi: kodun ayrı, ağ-kısıtlı bir sandbox'ta çalışması) şu ana kadar tamamen "kara kutu" olan iç işleyişini (`--trace` bittikten SONRA görülen statik bir JSON) **çalışırken, adım adım, canlı** görünür kılmak. Kullanıcı, bir sorunun arkasında bir Kubernetes pod'unun doğduğunu, kod çalıştırdığını, Tool Gateway'e çağrılar yaptığını (ya da bir şeyin engellendiğini) ve öldüğünü — gerçek zamanlı olarak izleyebilecek.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Tarayıcıda soru sorup zemine-dayalı yanıt alma (Priority: P1) 🎯 MVP

Kullanıcı, tarayıcıda bir soru yazar, gönderir; asistanın yanıtını, bu yanıtın zemine dayalı olup olmadığını (grounded), hangi kaynakların katkı sağladığını ve varsa kısmi hata notlarını görür — CLI'nin `assistant "soru" --trace` çıktısının web karşılığı.

**Why this priority**: Bu olmadan Faz 4'ün hiçbir değeri yok — temel etkileşim budur, diğer her şey bunun üzerine kurulu.

**Independent Test**: Bir soru sorulup, CLI'de aynı soru sorulduğunda alınan yanıtla (metin, grounded durumu, kaynaklar) tutarlı bir sonucun tarayıcıda göründüğü doğrulanabilir.

**Acceptance Scenarios**:

1. **Given** kullanıcı tarayıcıda bir soru yazıp gönderir, **When** asistan yanıtı üretir, **Then** yanıt metni, grounded durumu (evet/hayır) ve katkı sağlayan kaynaklar ekranda görünür.
2. **Given** hiçbir erişim yolu ilgili veri döndürmez, **When** yanıt üretilir, **Then** kullanıcıya bunun açıkça belirtildiği, tahmini bir değer İÇERMEYEN bir mesaj gösterilir (Principle I, Faz 1'deki gibi).
3. **Given** bazı erişim yolları başarısız/boş kalır ama en az biri veri döndürür, **When** yanıt üretilir, **Then** kısmi hata notları da (hangi kaynak/erişim yolunun eksik kaldığı) ekranda görünür.

---

### User Story 2 - Bir PTC çalıştırmasının tüm yaşam döngüsünü canlı izleme (Priority: P2)

Bir soru, asistanın kod yazıp sandbox'ta çalıştırmasını (PTC) gerektirdiğinde, kullanıcı bu çalıştırmanın adımlarını — kodun kendisi, sandbox içinde yapılan her tool çağrısı, varsa engellenen bir erişim girişimi, nihai sonuç/hata/zaman aşımı — GERÇEKLEŞTİKÇE, ekranın sol-alt köşesindeki bir panelde, terminal benzeri kayan bir log olarak izler.

**Why this priority**: Bu, Faz 4'ün kendine özgü katkısı — Faz 2'nin (PTC + Cilium/eBPF) şu ana kadar görünmeyen iç işleyişini şeffaflaştırıyor. P1 (temel soru-cevap) olmadan bağımsız bir değeri yok, bu yüzden ikinci öncelik.

**Independent Test**: Bilinçli olarak bir PTC çalıştırması tetikleyen bir soru sorulup, panelde en az şu adımların, çalıştırma bitmeden ÖNCE (biriktirilip sona bırakılmadan) sırayla göründüğü doğrulanabilir: çalıştırmanın başladığı, kodun kendisi, en az bir tool çağrısı, nihai sonuç.

**Acceptance Scenarios**:

1. **Given** bir soru bir PTC çalıştırmasını tetikler, **When** çalıştırma sürüyor, **Then** panel, çalıştırma bitmeden önce en azından "çalıştırma başladı" ve "kod bu" adımlarını gösterir (her şey sona toplanıp bir kerede basılmaz).
2. **Given** sandbox içinde bir tool-proxy çağrısı yapılır, **When** bu çağrı gerçekleşir, **Then** panelde bu çağrı (hangi tool, ne zaman, başarılı/başarısız) görünür.
3. **Given** sandbox onaylanmamış bir hedefe erişmeye çalışır ve bu engellenir, **When** bu olay tespit edilir, **Then** panelde bu, açıkça "engellendi" olarak işaretlenmiş bir satır olarak görünür.
4. **Given** çalıştırma sona erer (başarı/hata/zaman aşımı), **When** bu olur, **Then** panelde nihai durum açıkça görünür ve panel önceki adımları (scrollback) kaybetmez.
5. **Given** bir soru PTC kullanmadan (doğrudan tool-calling ile) yanıtlanır, **When** bu olur, **Then** panelde bu sorgu için sandbox kullanılmadığı açıkça belirtilir (panel sessizce boş kalmaz).

---

### User Story 3 - Aynı anda birden fazla sorgu birbirine karışmaz (Priority: P3)

Kullanıcı (ya da birden fazla tarayıcı sekmesi) aynı anda farklı sorular sorarsa, her sorgunun yanıtı ve PTC paneli yalnızca KENDİ sorgusuna ait bilgiyi gösterir.

**Why this priority**: Doğruluk/güven açısından önemli ama P1/P2 çalışıyorsa bağımsız olarak eklenip test edilebilir bir sağlamlık katmanı, bu yüzden en düşük öncelik.

**Independent Test**: İki farklı tarayıcı sekmesinden, biri PTC tetikleyen biri tetiklemeyen iki farklı soru aynı anda sorulup, her sekmenin yalnızca kendi sorgusunun yanıtını/panelini gösterdiği, birbirine karışmadığı doğrulanabilir.

**Acceptance Scenarios**:

1. **Given** iki farklı tarayıcı sekmesinden aynı anda iki farklı soru sorulur, **When** yanıtlar/PTC panelleri üretilir, **Then** her sekme yalnızca kendi sorgusuna ait yanıtı ve panel içeriğini gösterir.

---

### Edge Cases

- Bir PTC çalıştırması hâlâ sürerken kullanıcı yeni bir soru sorarsa ne olur? → Yeni sorgu, önceki çalıştırmayı kesmeden kendi sırasında işlenir; panel, hangi çalıştırmanın hangi soruya ait olduğunu karıştırmaz.
- Tarayıcı ile sunucu arasındaki canlı bağlantı, bir PTC çalıştırması SÜRERKEN kopar da kullanıcı sayfayı yeniler/geri gelirse ne olur? → Kullanıcı, en azından çalıştırmanın son bilinen durumunu (tamamlandı mı, hâlâ sürüyor mu) görür; canlı akışın koptuğu dönemdeki ADIM ADIM ayrıntı kalıcı olarak kaybolabilir (bu PoC'nin kapsamı, kalıcı bir olay günlüğü değil — bkz. Assumptions).
- Bir soru ne PTC ne doğrudan bir tool-calling gerektirmezse (ör. asistan sadece genel bir sohbet yanıtı verirse) panelde ne olur? → Panel o soru için hiçbir yeni satır eklemez; önceki bir çalıştırmanın kaydı varsa o durduğu yerde kalır.
- Çok uzun bir PTC kodu/çıktısı panelde nasıl gösterilir? → Panel kayan bir günlük görünümündedir (terminal benzeri); uzunluk kendisi bir engel değildir, kullanıcı geriye kaydırarak önceki satırları görebilir.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Sistem, kullanıcının bir tarayıcı sayfasında soru yazıp göndermesine ve asistanın yanıtını aynı sayfada görmesine izin vermelidir.
- **FR-002**: Sistem, her yanıt için zemine-dayalılık durumunu (grounded/değil) açıkça göstermelidir — Principle I (Faz 1) UI'da da geçerlidir; zemin bulunamayan bir durumda tahmini bir değer "yanıt" olarak sunulamaz.
- **FR-003**: Sistem, bir yanıta katkı sağlayan kaynakları (source_refs) ve varsa kısmi hata notlarını (partial_failure_notes) kullanıcıya göstermelidir.
- **FR-004**: Bir soru bir PTC (sandbox kod çalıştırma) çalıştırmasını tetiklediğinde, sistem bu çalıştırmanın adımlarını (çalıştırmanın başlaması, çalıştırılan kodun kendisi, sandbox içindeki her tool-proxy çağrısı, varsa engellenen bir erişim girişimi, nihai sonuç/hata/zaman aşımı) ekranın ayrı, belirgin bir bölgesinde (sol-alt köşe) göstermelidir.
- **FR-005**: PTC yaşam döngüsü paneli, adımları GERÇEKLEŞTİKÇE göstermelidir — çalıştırma bitene kadar biriktirilip sonradan tek seferde basılmamalıdır (FR-004'ün "canlı" niteliği).
- **FR-006**: Bir soru PTC kullanmadan yanıtlandığında, sistem bu paneli sessizce boş bırakmak yerine bunun açıkça belirtildiği bir durum göstermelidir.
- **FR-007**: Sistem, aynı anda birden fazla sorgu işlendiğinde, her sorgunun yanıtını ve PTC panelini yalnızca o sorguyu başlatan kullanıcı/oturuma göstermeli, sorgular arasında karışma olmamalıdır.
- **FR-008**: PTC panelinde gösterilen bilgiler, mevcut `SandboxRun`/`LiveToolCall`/`DeniedAction`/`Trace` veri modelleriyle (Faz 1/2) tutarlı olmalıdır — UI, kendi ayrı bir izlenebilirlik veri modeli tanımlamamalı, mevcut modelin bir görünümü olmalıdır (Principle III, Principle V).
- **FR-009**: Panel, bir çalıştırma bittikten sonra da önceki adımları (scrollback) korumalı, yeni bir çalıştırma başladığında öncekini silmemelidir (terminal benzeri davranış).

### Key Entities *(include if feature involves data)*

- **Kullanıcı Oturumu**: Bir tarayıcı sekmesinin, hangi sorgu(lar)ın/PTC çalıştırmalarının kendisine ait olduğunu ayırt etmesini sağlayan bağlam — Faz 1'in `session_id`/`thread_id` kavramının web arayüzündeki karşılığı. Yeni bir varlık değil, mevcut kavramın bu fazdaki kullanımı.
- Yeni bir veri modeli TANIMLANMIYOR — bu özellik, Faz 1/2'nin `Answer`, `Trace`, `SandboxRun`, `LiveToolCall`, `DeniedAction` modellerini olduğu gibi kullanır (FR-008).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Kullanıcı, bir soru gönderdikten sonra, asistanın kendi yanıt üretme süresine ek olarak arayüzün kendisinden kaynaklanan gözle görülür bir gecikme yaşamaz (arayüz, backend hazır olur olmaz yanıtı gösterir).
- **SC-002**: Bir PTC çalıştırması tetiklendiğinde, kullanıcı çalıştırmanın İLK adımını, çalıştırma tamamlanmadan ÖNCE panelde görür — denetim örnekleminde %100 (hiçbir çalıştırma "sessizce" bitip sonra topluca görünmez).
- **SC-003**: Zemin bulunamayan (grounded=False) her yanıt, kullanıcıya açıkça işaretli gösterilir — denetim örnekleminde %100 (Faz 1'in SC-001'iyle aynı çıta, UI'a genişletilmiş).
- **SC-004**: İki farklı tarayıcı sekmesinden eş zamanlı sorgularda, çapraz karışma (bir sekmenin diğerinin yanıtını/panelini göstermesi) oranı %0'dır.

## Assumptions

- Bu web arayüzü, Faz 1/2'nin geri kalanıyla aynı yerel/araştırma kapsamındadır — kimlik doğrulama, çok-kullanıcılı yetkilendirme veya uzaktan/internet erişimi bu fazın kapsamı DIŞINDADIR (Principle V; Faz 1'in kimliksiz Tool Gateway kararıyla tutarlı). Arayüz yalnızca yerel makinede (localhost) çalışır.
- Bir sorgu/oturum, CLI'nin zaten sahip olduğu çok-turlu konuşma yeteneğini (thread_id ile) kullanabilir — yani web arayüzü bir "sohbet" gibi davranabilir, her soru bağımsız bir CLI çağrısı gibi ele alınmak zorunda değildir.
- PTC panelinin canlı akışı, bağlantı kesildiği dönem için kalıcı bir olay günlüğü/geçmişi TUTMAK zorunda değildir — kullanıcı o an bakıyorsa görür; bu PoC bir üretim gözlemlenebilirlik sistemi değildir (Principle V).
- Bu özellik, Faz 1/2'nin `assistant` CLI'sini KALDIRMAZ — web arayüzü, mevcut agent/graph.py + Trace + sandbox_runner mekanizmasının üzerine ikinci bir giriş noktası olarak eklenir, CLI paralel olarak çalışmaya devam eder.
- Somut teknoloji seçimleri (backend framework, gerçek-zamanlı iletişim mekanizması, frontend yaklaşımı) bu belgenin kapsamı dışıdır — Principle IV gereği planlama aşamasında proje sahibiyle netleştirilecektir.
