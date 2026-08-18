"""AutoGen destesi, genişletilmiş: her bileşene ve her desene birer sayfa.

Genel bakış slaytları (`Components Guide`, `Çok-ajan desenleri`) harita olarak
kalıyor; buradaki on dokuz sayfa araziyi geziyor. `make_hap.py` yüklüyor ve `A`
listesine ekliyor.

Her başlık kendi `05:NNNN` satırını taşıyor, yani iddia kaynak kılavuzda
açılabilir. Bunlar kılavuzun kendi bölümleri, bizim tasnifimiz değil.

Yazım kuralı (2026-08-18'de yeniden yazıldı): önce **ne olduğu**, sonra **nasıl
çalıştığı**, en sonda **neden umursaman gerektiği**. Bir terim tanımlanmadan
kullanılmıyor. Uzun tire bağlacın yerini tutmuyor.
"""

# ═══════════════════════════════════════════ BİLEŞENLER, teker teker

A.append(card(
    "AutoGen · bileşen 1/6", "Model Clients",
    f_model_clients(),
    "05:1984 — dördü de aynı ChatCompletionClient protokolünü uyguluyor",
    "Model istemcisi, sağlayıcıyla konuşan katman. Ajan \"şu mesajları gönder, "
    "cevabı getir\" diyor; nasıl gönderileceğini istemci biliyor. "
    "Dört yerleşik istemci var: <b>OpenAI ve uyumlular</b> (Gemini dahil), "
    "<b>Azure OpenAI</b>, <b>Azure ve GitHub'da barındırılan</b> modeller, bir de "
    "<b>ReplayChatCompletionClient</b>. "
    "Sonuncusu gerçek bir modele hiç bağlanmıyor; önceden yazdığın cevapları "
    "sırayla döndürüyor. Ama diğerleriyle aynı protokolü uyguladığı için ajan "
    "aradaki farkı göremiyor. "
    "Bunun pratik değeri büyük: <b>test ve kuru mod için ayrı bir kod yolu "
    "yazmıyorsun.</b> Bu depodaki <code>engine.Ledger</code> tam olarak bunu "
    "kullanıyor, ve API anahtarı yokken tarama uçtan uca deterministik koşuyor.",
    cap_mm=44,
    foot="Uyumlu bir endpoint kullanıyorsan model_info tuzağı burada bekliyor — destenin tuzak slaytına bak."))

A.append(card(
    "AutoGen · bileşen 2/6", "Model Context",
    f_context(),
    "05:2341 — modele NE gideceğine karar veren şey",
    "Model istemcisi sağlayıcıyla <i>nasıl</i> konuşulacağını bilir. Model "
    "context ise <i>ne</i> söyleneceğine karar verir. Mesajları saklıyor ve her "
    "model çağrısından önce sıralı listeyi veriyor. "
    "Yerleşik dört uygulamanın dördü de aynı şeyi yapıyor: fazlasını "
    "<b>kırpıyor</b>. <code>Unbounded</code> hiç kırpmıyor, "
    "<code>Buffered</code> son n mesajı tutuyor, <code>HeadAndTail</code> baş ve "
    "son kısmı tutup ortayı atıyor, <code>TokenLimited</code> token limitine "
    "kadar tutuyor. "
    "Yani AutoGen'de <b>özetleyerek sıkıştıran hazır bir uygulama yok</b>. "
    "Sıkıştırma istiyorsan yazacaksın; biz de öyle yaptık. "
    "Ve buradaki tuzak sessiz: <code>model_context</code> vermezsen ajanın "
    "belleği hiç olmuyor, hiçbir hata da çıkmıyor.",
    cap_mm=40,
    foot="Bizde: pipeline/context_engine.py — dört yaşam noktası, ve tool çağrısını sonucundan ayırmayan bir sıkıştırma."))

A.append(card(
    "AutoGen · bileşen 3/6", "Tools",
    f_tools_component(),
    "05:2473 — şema, fonksiyon imzasından ve docstring'den türetiliyor",
    "Tool, ajanın bir eylem yapmak için koşturduğu koddur. Bir hesap makinesi "
    "kadar basit ya da bir üçüncü taraf API çağrısı kadar karmaşık olabilir. "
    "Modelin kendisi kodu çalıştırmıyor. Model yalnızca \"şu tool'u şu "
    "argümanlarla çağır\" diyen bir çıktı üretiyor; çalıştırmayı çerçeve yapıyor. "
    "Şaşırtıcı olan şu: tool'un işe yarayıp yaramaması büyük ölçüde <b>koda "
    "değil, açıklamaya</b> bağlı. Model, bu tool'u ne zaman çağıracağına "
    "docstring'i okuyarak karar veriyor; ne göndereceğine de tip ipuçlarına "
    "bakarak. Tip ipucu yazmazsan model tahmin eder. "
    "Yani tool yazmanın yarısı Python, yarısı <b>arayüz tasarımı</b>.",
    cap_mm=44))

A.append(card(
    "AutoGen · bileşen 4/6", "Workbench (ve MCP)",
    f_workbench_component(),
    "05:2841 — tek tool değil, durum ve kaynak paylaşan bir koleksiyon",
    "<code>Tool</code> tek bir tool'a arayüz verir. <b>Workbench</b> ise birden "
    "çok tool'a birden verir, ve o tool'lar <b>durum ve kaynak paylaşır</b>. "
    "Fark neden önemli? Çünkü bir MCP sunucusuna açılan <i>tek</i> bağlantı, o "
    "sunucudaki bütün tool'ları taşıyabiliyor. Her tool için ayrı bağlantı "
    "gerekmiyor. "
    "Üç uygulama var: <code>StaticWorkbench</code> elindeki Python "
    "fonksiyonlarını sarıyor, <code>McpWorkbench</code> stdio ya da HTTP "
    "üstünden uzak bir sunucuya bağlanıyor, ve <code>GatedWorkbench</code> "
    "<b>bizim eklediğimiz</b> onay kapısı. "
    "Ajan açısından üçü de aynı: <code>list_tools()</code> ve "
    "<code>call_tool()</code>. <b>Kapıyı araya koyabilmemizin tek sebebi bu.</b>",
    cap_mm=42,
    foot="Tuzak: aynı ajana hem tools= hem workbench= verirsen ValueError alırsın. İkisi birbirinin alternatifi."))

A.append(card(
    "AutoGen · bileşen 5/6", "Code Executors",
    f_code_executors(),
    "05:3054 — her blok bir dosyaya yazılıp AYRI SÜREÇTE koşuyor",
    "Modelin yazdığı kodu çalıştırmanın en basit biçimi şu: her kod bloğu bir "
    "dosyaya kaydediliyor, sonra o dosya çalıştırılıyor. "
    "Bunun doğrudan bir sonucu var ve şaşırtıyor — <b>bloklar arasında değişken "
    "paylaşımı yok.</b> Her blok yeni bir süreç olduğu için ilk blokta "
    "tanımladığın değişken ikincisinde yok. "
    "İki uygulama var ve aralarındaki fark teknik değil, <b>güvenle ilgili</b>. "
    "<code>DockerCommandLineCodeExecutor</code> kodu bir konteynerde koşturuyor. "
    "<code>LocalCommandLineCodeExecutor</code> ise <b>senin makinende</b>. "
    "İkincisi, bir dil modelinin yazdığı kodu kendi diskine erişimle "
    "çalıştırmak demek. Bu bir performans tercihi değil, bir <b>güven kararı</b>, "
    "ve varsayılan olarak seçilmemesi gereken taraf.",
    cap_mm=42))

A.append(card(
    "AutoGen · bileşen 6/6", "Component config — plan ile fotoğrafın farkı",
    f_component_config(),
    "05:1888 — config bir PLAN, state bir FOTOĞRAF",
    "Kılavuzun kendi ayrımı, ve karıştırılması çok kolay. "
    "<b>Component config</b> bir nesnenin <i>planı</i>: hangi sınıf, hangi "
    "ayarlarla. <code>dump_component</code> ile JSON'a yazılıyor, "
    "<code>load_component</code> ile geri kuruluyor. Aynı plandan istediğin kadar "
    "yeni örnek damgalayabiliyorsun. "
    "<b>State</b> ise nesnenin o anki hâli: mesaj geçmişi dahil her şey. Bunu "
    "geri yüklerken <b>tam olarak aynı</b> nesneye yüklemek zorundasın, çünkü "
    "state kendi başına ne olduğunu bilmiyor. "
    "Pratik sonucu şu: yapılandırma artık kod değil <b>veri</b>. AutoGen "
    "Studio'nun tıklayarak ajan kurma deneyimi bunun üstünde duruyor, ve AutoGen "
    "dışında tanımlanmış bileşenler de aynı sisteme katılabiliyor.",
    cap_mm=40))

# ═══════════════════════════════════════════ DESENLER, teker teker

A.append(card(
    "AutoGen · desen 1/8", "Concurrent Agents",
    f_fanout(),
    "05:3236 — eşzamanlılık abonelikten geliyor, bir çağrıdan değil",
    "Kılavuz bu deseni üç alt başlıkta anlatıyor. <b>Tek mesaj, çok işleyici</b>: "
    "aynı topic'e abone birden çok ajan aynı mesajı aynı anda işliyor. <b>Çok "
    "mesaj, çok işleyici</b>: mesaj tipleri topic'lere göre ayrı ajanlara "
    "gidiyor. Bir de <b>doğrudan mesajlaşma</b>. "
    "Bizim tarama boru hattımız birincisini kullanıyor. "
    "Ve ölçtüğümüz kardeş kaybı problemi tam burada yaşıyor: dallardan biri "
    "sessizce ölürse toplayıcı sonsuza kadar bekliyor. Sebebi mekanik — "
    "<code>publish_message</code> hata fırlatmıyor, yalnız logluyor. Toplayıcı "
    "de eksik dalın öldüğünü öğrenemiyor. "
    "Çözüm küçük ama zorunlu: <b>sayacı <code>finally</code> bloğunda artır.</b> "
    "Bu desenin fiyatı o satır.",
    cap_mm=42))

A.append(card(
    "AutoGen · desen 2/8", "Sequential Workflow",
    f_sequential(),
    "05:3504 — deterministik sıra, her ajan bir alt görev",
    "Ajanlar önceden belli bir sırayla çalışıyor. Her biri bir mesajı alıyor, "
    "işliyor, çıktısını bir sonrakine veriyor. Kimin konuşacağına <b>hiç kimse "
    "karar vermiyor</b>, çünkü sıra zaten kodda yazılı. "
    "Kılavuzun örneği dört ajanlı bir pazarlama hattı: <b>Concept Extractor</b> "
    "ürün açıklamasından özellikleri ve hedef kitleyi çıkarıyor, <b>Writer</b> "
    "pazarlama metnini yazıyor, <b>Format &amp; Proof</b> dilbilgisini ve tonu "
    "düzeltiyor, <b>User</b> sonucu sunuyor. "
    "core katmanında bu, her ajanın bir sonrakinin topic'ine yayın yapmasıyla "
    "kuruluyor. Yani <b>zincir aboneliklerde yazılı</b>, ortada zinciri yöneten "
    "bir orkestratör yok. "
    "Sıra belliyse bu en ucuz desen: fazladan hiç model çağrısı yapmıyor.",
    cap_mm=40))

A.append(card(
    "AutoGen · desen 3/8", "Group Chat",
    f_groupchat(),
    "05:3772 — herkes aynı topic'e hem abone hem yayıncı",
    "Bir grup ajan <b>ortak bir mesaj dizisini</b> paylaşıyor. Hepsi aynı "
    "topic'e abone, hepsi aynı topic'e yayın yapıyor. Yani her ajan, "
    "diğerlerinin yazdığı her şeyi görüyor. "
    "Katılımcılar farklı işlerde uzman oluyor: yazar, çizer, editör gibi. "
    "İstersen ajanları yönlendirmesi için araya bir <b>insan katılımcı</b> da "
    "koyabiliyorsun. "
    "Gücü de bedeli de aynı yerden geliyor. Herkes aynı konuşmayı gördüğü için "
    "hiçbir bilgi kaybolmuyor. Ama <b>bağlam da herkes için aynı ve büyük</b>: "
    "her ajanın her turu, bütün konuşmanın maliyetini ödüyor. "
    "Ayrılmaya değer bir bağlam varsa bu yanlış desen. O durumda fan-out doğru "
    "olan.",
    cap_mm=48))

A.append(card(
    "AutoGen · desen 4/8", "Handoffs",
    f_handoffs(),
    "05:4349 — devretme, özel bir tool çağrısından ibaret",
    "Fikir OpenAI'ın <b>Swarm</b> adlı deneysel projesinden geliyor. Bir ajan, "
    "görevi başka bir ajana devrediyor; ve devretme özel bir tool çağrısı olarak "
    "yapılıyor. "
    "Dışarıda kimi kimin izleyeceğine karar veren bir yönlendirici yok. Devretme "
    "kararını <b>ajanın kendisi</b> veriyor. "
    "AutoGen bunun üstüne üç şey eklemiş: dağıtık runtime'a ölçeklenebilmesi, "
    "kendi ajan uygulamanı getirebilmen, ve doğal async API. "
    "<b>Ölçtüğümüz sonuç:</b> bu, karşılaştırdığımız dört desen içinde en "
    "pahalısı. 334 token, SelectorGroupChat'in <b>%63,7 üstünde</b>. Sebebi de "
    "mekanik: her devirde bağlam yeniden kuruluyor. "
    "Ödediğin şey zekâ değil, <b>yönlendirme özerkliği</b>.",
    cap_mm=40,
    foot="Tuzak: Handoff tool adı küçük harfe düşürülüyor. Elle yazarsan eşleşmez — Handoff(target=X).name ile üret."))

A.append(card(
    "AutoGen · desen 5/8", "Mixture of Agents",
    f_mixture(),
    "05:4989 — ileri-beslemeli bir sinir ağından modellenmiş",
    "İki tip ajan var: bir sürü <b>işçi</b> ve tek bir <b>orkestratör</b>. "
    "İşçiler katmanlara ayrılmış ve her katmanda sabit sayıda işçi var. Bir "
    "katmandaki işçilerin çıktıları <b>birleştirilip</b> sonraki katmandaki "
    "bütün işçilere gönderiliyor. "
    "Sinir ağı benzetmesi bizim değil, kılavuzun kendi ifadesi. "
    "Pratikte olan şu: aynı soruya farklı uzmanlıklar bakıyor, sonra bir "
    "birleştirici hepsini tek cevaba indiriyor. "
    "İşe yaradığı yer, uzmanlık alanlarının <b>gerçekten ayrı</b> olduğu "
    "durumlar. Ayrı değilse aynı cevabın pahalı kopyalarını üretmiş olursun. "
    "Kaynak makale: arXiv:2406.04692.",
    cap_mm=44))

A.append(card(
    "AutoGen · desen 6/8", "Multi-Agent Debate",
    f_debate(),
    "05:5358 — her turda cevaplar karşılıklı değiş tokuş ediliyor",
    "Çok turlu bir etkileşim. Her turda ajanlar cevaplarını birbirine "
    "gösteriyor, sonra <i>diğerlerinin cevaplarına bakarak</i> kendi cevabını "
    "düzeltiyor. "
    "İki tip ajan var: <b>çözücüler</b> ve tek bir <b>toplayıcı</b>. "
    "Ve bir tasarım ayrıntısı önemli: çözücüler <b>seyrek</b> bağlanmış. Herkes "
    "herkesle konuşmuyor. Bu kasıtlı, çünkü tam bağlı bir grup hem pahalı hem "
    "de hızla tek bir görüşe yakınsıyor. "
    "Kılavuz bunu GSM8K matematik problemleri üstünde gösteriyor, yani "
    "<b>cevabı doğrulanabilen</b> bir alanda. Bu ipucu önemli: münazara, "
    "yanlışın tespit edilebildiği yerlerde işe yarıyor. Görüş meselelerinde "
    "yalnızca daha uzun konuşuyorsun.",
    cap_mm=42))

A.append(card(
    "AutoGen · desen 7/8", "Reflection",
    f_reflection(),
    "05:5822 — ikinci üretim, birincinin çıktısına koşullanmış",
    "Bir model çıktı üretiyor, ardından ikinci bir model o çıktının kritiğini "
    "yazıyor. İkinci üretim birincinin sonucuna koşullanmış olduğu için ona "
    "\"yansıtma\" deniyor. "
    "Kod yazma örneğinde: birinci model kodu yazıyor, ikincisi kodu inceleyip "
    "neyin yanlış olduğunu söylüyor. "
    "Ajan dünyasında bu bir <b>çift</b> olarak kuruluyor. Biri mesaj üretiyor, "
    "diğeri o mesaja cevap veriyor, ve kritik tatmin olana kadar konuşma devam "
    "ediyor. "
    "Bu depodaki karşılığı <code>RiskAuditor</code>. Klasik yansıtmadan bir "
    "farkı var: kendi çıktısını değil, <b>üç analistin çıktısını</b> denetliyor. "
    "Aradığı şey çelişki ve kaynak gösterilmemiş iddia.",
    cap_mm=46))

A.append(card(
    "AutoGen · desen 8/8", "Code Execution",
    f_codeexec_pattern(),
    "05:6188 — kodu yazan ajan ile koşturan ajan ayrı",
    "Sekizincisi aslında bir orkestrasyon deseni değil, bir <b>yetenek</b>. "
    "İki ajan var: <code>Assistant</code> kodu yazıyor, <code>Executor</code> "
    "koşturuyor. Aralarında tek bir <code>Message</code> veri sınıfı gidip "
    "geliyor. "
    "AgentChat'te bunun hazır karşılıkları da var "
    "(<code>AssistantAgent</code>, <code>CodeExecutorAgent</code>). Ama kılavuz "
    "bilerek elle yazmayı gösteriyor, çünkü amaç hazır sınıfı tanıtmak değil; "
    "<b>hafif bir özel ajanın nasıl yazıldığını</b> göstermek. "
    "Örneği somut: Tesla ile Nvidia hisse getirilerinin grafiğini çizdirmek. "
    "Ve hangi yürütücüyü seçtiğin bir güven kararı — bir önceki bileşen "
    "slaytındaki Docker/local ayrımı.",
    cap_mm=48,
    foot="Bu desen bizde YOK: taramada modelin yazdığı kod koşmuyor, bütün tool'lar önceden yazılmış."))

# ═══════════════════════════════════════════ AGENTCHAT "ADVANCED"
#
# Takım slaytlarını yazarken Selector, Swarm, GraphFlow ve Logging zaten
# geçmişti. Boşluk bu beşiydi: Magentic-One tek bir tablo satırıydı; Custom
# Agents, Memory/RAG, Serializing ve Tracing hiç yoktu.

A.append(card(
    "AutoGen · advanced 1/5", "Custom Agents",
    f_custom_agent(),
    "08:7248 — BaseChatAgent + üç üye, hepsi bu",
    "Hazır ajanlar istediğin davranışı vermiyorsa kendi ajanını yazıyorsun. "
    "Sözleşme şaşırtıcı derecede küçük: <code>BaseChatAgent</code>'tan türeyip "
    "üç şey uyguluyorsun. "
    "<b>on_messages</b> gelen mesaja nasıl cevap vereceğini söylüyor ve bir "
    "<code>Response</code> döndürüyor. <b>on_reset</b> ajanı başlangıç durumuna "
    "döndürüyor. <b>produced_message_types</b> hangi mesaj tiplerini "
    "üretebildiğini bildiriyor. "
    "Bunu bilmenin asıl değeri şu: <code>AssistantAgent</code> da bu sınıfın bir "
    "alt sınıfı. Yani \"hazır ajan\" ile \"özel ajan\" arasında <b>ayrıcalık "
    "farkı yok</b>. Senin yazdığın ajan da takımlara giriyor, akışa katılıyor, "
    "sonlandırma koşullarına takılıyor — hepsi aynı şekilde.",
    cap_mm=42,
    foot="Bizde: pipeline/agents/ hazır AssistantAgent kullanıyor; özel ajana henüz ihtiyaç olmadı."))

A.append(card(
    "AutoGen · advanced 2/5", "Memory ve RAG",
    f_memory_rag(),
    "08:6242 — add · query · update_context · clear · close",
    "Bir olgu deposu tutup, onu <b>belirli bir adımdan hemen önce</b> ajanın "
    "bağlamına eklemek istediğin durumlar için. Klasik RAG deseni bunun bir "
    "örneği: sorgu bir veritabanından ilgili bilgiyi getiriyor, o bilgi bağlama "
    "giriyor. "
    "Protokolün beş metodu var ama <b>kritik olan üçüncüsü</b>. "
    "<code>update_context</code>, ajanın <i>kendi</i> "
    "<code>model_context</code>'ini değiştiriyor. "
    "Bunun anlamı önemli: getirme bir tool çağrısı <b>değil</b>. Model \"bilgi "
    "getir\" diye bir şey çağırmıyor; bağlam, model çağrısından hemen önce "
    "sessizce zenginleşiyor. Model neyin nereden geldiğini görmüyor. "
    "Hazır uygulamalar: <code>ListMemory</code>, "
    "<code>ChromaDBVectorMemory</code>, Redis.",
    cap_mm=42,
    foot="Bizde YOK ve bu bilinçli: docs_index bir tool olarak duruyor, bağlama kendiliğinden enjekte etmiyor."))

A.append(card(
    "AutoGen · advanced 3/5", "Serializing Components — ve tek sert uyarı",
    f_serialize_agentchat(),
    "08:6834 — dump_component() / load_component()",
    "Ajan, takım, model istemcisi — hepsi bildirimsel bir şartnameye yazılıp "
    "geri kurulabiliyor. Hata ayıklamak, görselleştirmek ve <b>çalışmanı "
    "başkasıyla paylaşmak</b> için. "
    "Ama AgentChat kılavuzunun bütün bölümü içindeki <b>tek büyük harfli "
    "uyarı</b> burada duruyor, ve sebebi mekanik: her bileşen kendi "
    "serileştirme mantığını kendisi yazıyor. Bir nesneyi geri kurmak <b>kod "
    "çalıştırmayı içerebiliyor</b>; serileştirilmiş bir fonksiyon buna örnek. "
    "Sonuç açık: paylaşılan bir takım JSON'u, paylaşılan bir betikle <b>aynı "
    "güven sınıfında</b>. "
    "Bir kurumda bunu \"yapılandırma dosyası\" diye elden ele dolaştırmak, kod "
    "dağıtmakla aynı şey.",
    cap_mm=42))

A.append(card(
    "AutoGen · advanced 4/5", "Tracing ve Observability",
    f_tracing(),
    "08:6975 — OpenTelemetry, GenAI Semantic Conventions",
    "Logging sana tek tek olayları verir. <b>Tracing</b> ise bir koşunun "
    "<i>yapısını</i> verir: hangi ajan hangi tool'u ne kadar sürede çağırdı, "
    "hangi adım hangisinin içinde geçti. "
    "Yerleşik ve <b>OpenTelemetry</b>'ye dayanıyor. Yani arka ucu sen "
    "seçiyorsun: Jaeger, Zipkin, ya da herhangi bir OTLP toplayıcı. Çerçeve "
    "sana bir gözlem ürünü dayatmıyor. "
    "Ayrıca <b>GenAI Semantic Conventions</b>'ı takip ediyor. Bu hâlâ "
    "geliştirilmekte olan bir standart, ve anlamı şu: AutoGen'in ürettiği "
    "kayıtlar başka çerçevelerinkiyle aynı sözlüğü kullanıyor, dolayısıyla "
    "karşılaştırılabiliyor. "
    "<b>Bizde kurulu değil.</b> Olay akışıyla yetiniyoruz, ve bu bir eksik.",
    cap_mm=42,
    foot="OpenClaw tarafındaki karşılığını ölçtük: içerik varsayılan olarak kapalı, yalnız boyut bilgisi aktarılıyor."))

A.append(card(
    "AutoGen · advanced 5/5", "Magentic-One",
    f_magentic(),
    "08:6053 — orkestratör + dört uzman ajan, arXiv:2411.04468",
    "Açık uçlu web ve dosya görevleri için kurulmuş <b>genelci</b> bir çok-ajan "
    "sistemi. Belirli bir işe göre değil, \"ne gelirse\" diye tasarlanmış. "
    "Orkestratör bir plan kuruyor, görevleri dağıtıyor, ilerlemeyi izliyor, ve "
    "tıkandığında planı <b>koşarken revize ediyor</b>. "
    "Dört uzman var: <code>MultimodalWebSurfer</code> tarayıcı kullanıyor, "
    "<code>FileSurfer</code> dosya okuyor, <code>MagenticOneCoderAgent</code> "
    "kod yazıyor, terminal ajanı da koşturuyor. "
    "Önce <code>autogen-core</code> üstüne yazılmış, sonra AgentChat'e "
    "taşınmış. Yani bugün <b>sıradan bir AgentChat takımı</b>, ve dört ajanı "
    "tek tek başka akışlarda da kullanabiliyorsun. "
    "Kılavuzun kendi uyarısı altı maddeli ve ciddi: konteyner kullan, sanal "
    "ortam kullan, logları izle, <b>insan gözetimi koy</b>, erişimi kısıtla, "
    "veriyi koru.",
    cap_mm=40,
    foot="\"İnsanlar için tasarlanmış dijital bir dünyayla etkileşmek doğası gereği risk taşır.\" — kılavuzun kendi cümlesi."))
