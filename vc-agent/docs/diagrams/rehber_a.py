"""Rehber, birinci yarı: harita · AutoGen · MAF.

`make_rehber.py` tarafından exec ediliyor; motorun yardımcıları oradan geliyor.
"""

from make_ogretici import (  # noqa: F401 — motorun yardımcıları
    chapter, code, dene, fig, h3, neden, olcum, out, p, part, shell, table,
    tuzak, two,
)
from figures import (  # noqa: F401
    f_actor, f_components, f_debate, f_fanout, f_gotchas, f_graphflow,
    f_groupchat, f_handoffs, f_identity, f_layers, f_message_types, f_mixture,
    f_model_clients, f_reflection, f_send_vs_publish, f_sequential, f_teams,
    f_termination, f_tool_loop, f_topic, f_workbench_component,
)

# ═════════════════════════════════════════════════════════ KISIM 1 — harita

part("1", "Harita",
     "Üç sistem, üç ayrı soru. Karıştırıldıklarında seçim tartışması "
     "anlamsızlaşıyor, çünkü aynı yarışta değiller.")

chapter(
    "1", "Üç ayrı şey, üç ayrı soru",
    ["AutoGen, MAF ve OpenClaw hangi soruya cevap veriyor",
     "Neden bunlar birbirinin alternatifi değil",
     "Bu belgedeki etiketler ne anlama geliyor"],
    p("Bir kurum “hangi ajan çerçevesini kullanalım” diye sorduğunda genellikle "
      "üç ayrı soruyu tek soruya sıkıştırmış oluyor. Ayırdığında cevap "
      "kendiliğinden çıkıyor.")
    + table(
        ["Sistem", "Ne", "Cevapladığı soru", "Durum"],
        [["<b>AutoGen</b>", "kütüphane (Python)",
          "Birden çok ajanı nasıl koşturup birbirine bağlarım?",
          "bakım modu · v0.7.5"],
         ["<b>MAF</b>", "kütüphane (.NET · Python · Go)",
          "Aynı iş, kurumsal yüzeylerle: oturum, tip, ara katman, telemetri",
          "aktif · 1.14.0"],
         ["<b>OpenClaw</b>", "harness (uçtan uca araç)",
          "Bir ajanı gerçek bir makinede güvenle nasıl koştururum?",
          "aktif"]],
        "Kütüphane ile harness aynı yarışta değil: biri parça, öteki ürün.")
    + p("AutoGen ve MAF birbirinin alternatifi — biri diğerinin <b>halefi</b>. "
        "OpenClaw ise ikisinin de alternatifi değil: onlarla kurulmuş bir "
        "sistemin etrafına geçen kontrol düzlemi. Bir kurumda üçü de olabilir, "
        "ve çoğu zaman olmalı.")
    + neden(
        p("Bu ayrım pratikte şuraya bağlanıyor: <b>kütüphane seçimi geri "
          "alınabilir, harness kararı alınamaz.</b> Kütüphaneyi değiştirmek "
          "ajanları yeniden yazmak demek — birkaç haftalık iş. Harness'ı "
          "değiştirmek ise onay akışını, denetim kaydını, sır yönetimini ve "
          "operasyon alışkanlıklarını değiştirmek demek."))
    + fig(f_layers(), "Bir ajan sisteminin katmanları. Çerçeve ortada duruyor; "
                      "harness onu kuşatıyor."),
)

chapter(
    "2", "İddialar nasıl etiketleniyor",
    ["Üç etiket ve ne garanti ettikleri",
     "Neden rakip çerçeveler çoğunlukla [teyitsiz]"],
    p("Bu belgedeki her iddia üç etiketten birini taşıyor. Etiket, iddianın "
      "doğruluğunu değil <b>dayanağını</b> söylüyor.")
    + table(
        ["Etiket", "Anlamı", "Ne gösterilebilir"],
        [["<b>[ölçüldü]</b>", "Bu depoda kod koşturularak elde edildi",
          "ölçüm dosyası ve çıktısı"],
         ["<b>[kaynak]</b>", "Birincil kaynaktan doğrulandı",
          "resmî doküman satırı, kurulu paketin API'si, depo commit'i"],
         ["<b>[teyitsiz]</b>", "Okunandan; koşturulmadı",
          "yalnız okunan belgenin adresi"]])
    + p("AutoGen, MAF ve OpenClaw bu makinede gerçekten koşturuldu. "
        "LangGraph, CrewAI, OpenAI Agents SDK ve Google ADK <b>koşturulmadı</b> "
        "— onlar hakkındaki mimari cümleler <b>[teyitsiz]</b>. Bunu gizlemek "
        "belgenin geri kalanına olan güveni de düşürürdü.")
    + tuzak(
        p("Bir sayıyı slayta ya da rapora koyarken yanına <b>ölçüldüğü dosyayı</b> "
          "da yaz. Bu projede bir kez yapılmadı: “%63,7” sayısı doğru ölçülmüştü "
          "ama açıklaması hatırlanarak yazıldı ve yanlış şeye atfedildi. "
          "Kaynağı yazılmayan sayı, birkaç gün sonra yanlış anlatılıyor.")),
)

# ═══════════════════════════════════════════════════════ KISIM 2 — AutoGen

part("2", "AutoGen",
     "Aktör modelinden tool döngüsüne. Her mekanizmanın ne olduğu, nasıl "
     "çalıştığı, ve nerede ısırdığı.")

chapter(
    "3", "Üç katman",
    ["autogen_core · autogen_agentchat · autogen_ext ne yapıyor",
     "Hangi katmandan başlamalı",
     "Katmanları karıştırmanın maliyeti"],
    p("AutoGen tek bir kütüphane değil, üst üste duran üç katman. Hangi "
      "katmanda çalıştığını bilmemek, bu çerçevede en pahalı karışıklık.")
    + fig(f_layers(), "Üç katman: aşağı indikçe güç artıyor, kolaylık azalıyor.")
    + table(
        ["Katman", "İçeriği", "Ne zaman"],
        [["<code>autogen_core</code>", "aktör modeli: AgentId, runtime, topic, abonelik",
          "gerçek eşzamanlılık, dağıtık çalışma, kendi kapını kurmak"],
         ["<code>autogen_agentchat</code>", "AssistantAgent, beş takım tipi, "
          "on bir sonlandırma koşulu", "günlük iş — buradan başla"],
         ["<code>autogen_ext</code>", "model istemcileri, MCP, kod yürütücüler, "
          "üçüncü parti", "dış dünyaya bağlanmak"]])
    + neden(
        p("Kural basit ve pratik: <b>yukarıdan başla</b>. AgentChat'in zaten "
          "çözdüğü bir problemi core'da yeniden çözmek, aynı işi daha az testle "
          "yapmak demek. Aşağı inmek her zaman mümkün — ve gerektiğinde "
          "iniyoruz: bu projedeki fan-in ölçümü core katmanında yazıldı."))
    + olcum(
        p("Bu projede core'a inilen tek yer <code>pipeline/fanin.py</code>. "
          "Sebebi ölçüm: GraphFlow'un birleşme bariyeri bir dal çökünce "
          "tamamlanmış kardeşleri de kaybettiriyor. Core'da kuyrukla toplayan "
          "sürüm aynı arıza altında iki sonucu ~3 ms'de topladı; GraphFlow "
          "sıfır ya da bir sonuçla süre sınırına girdi. <b>[ölçüldü]</b> — "
          "<code>pipeline/compare_fanin.py</code>")),
)

chapter(
    "4", "Aktör modeli ve kimlik",
    ["Ajanın gerçekten ne olduğu",
     "AgentId'nin iki parçası ve neden ikisi de gerekli",
     "Topic kaynağının ajan anahtarına dönüşmesi"],
    p("AutoGen'i LangGraph'tan ya da CrewAI'dan ayıran şey burada: ajanlar "
      "gerçekten <b>aktör</b> — kendi mailbox'ı olan, mesajı <b>tipe göre</b> "
      "yönlendiren, makinelere dağıtılabilen birimler.")
    + fig(f_actor(), "Aktör modeli: bir ajan başka bir ajanın nesnesini elinde "
                     "tutmuyor, runtime'a mesaj veriyor.")
    + p("Bunun bedeli bir katman: “kim kimi çağırdı” yığın izinde görünmüyor. "
        "Karşılığında üç şey kazanılıyor — sisteme yeni ajan eklemek çağıran "
        "kodu <b>hiç</b> değiştirmiyor, bütün mesajlar tek noktadan geçtiği "
        "için müdahale ve ölçüm oraya takılıyor, ve aynı sınıftan istediğin "
        "kadar örnek doğurmak bedava.")
    + fig(f_identity(), "AgentId iki parçalı: tip <i>ne yaptığını</i>, anahtar "
                        "<i>hangi örnek olduğunu</i> söylüyor.")
    + neden(
        p("En çok işe yarayan mekanizma bu: <b>topic kaynağı, ajan anahtarına "
          "dönüşüyor.</b> <code>TopicId(\"turn\", \"oturum-42\")</code>'ye yayın "
          "yapmak <code>AgentId(\"session\", \"oturum-42\")</code> ajanını "
          "yaratıyor — oturum başına izole örnek, elle sözlük tutmadan. "
          "<b>[kaynak]</b> <code>05:670</code>. Bu projedeki gateway oturumları "
          "tam olarak böyle çalışıyor."))
    + fig(f_topic(), "Topic bir kanal adı değil, iki parçalı adres."),
)

chapter(
    "5", "İki iletişim biçimi, iki asimetri",
    ["send_message ile publish_message farkı",
     "Farkın adreslemede değil HATADA olduğu",
     "Hangisini ne zaman seçmeli"],
    p("İki yol var ve aralarındaki asıl fark kaç alıcı olduğu değil — "
      "<b>hata olduğunda ne olduğu</b>.")
    + fig(f_send_vs_publish(), "Doğrudan mesaj dönüş taşıyor ve fırlatıyor; "
                               "yayın ikisini de yapmıyor.")
    + table(
        ["", "Doğrudan (<code>send_message</code>)", "Yayın (<code>publish_message</code>)"],
        [["Alıcı", "tek <code>AgentId</code>", "topic'e abone olan herkes"],
         ["Dönüş değeri", "var", "<b>yok</b>"],
         ["Handler çökerse", "çağırana <b>fırlatır</b>", "<b>loglanır</b>, fırlatmaz"]])
    + p("Bu tek satır bir tasarım kararı: bir sonucu bekleyeceksen doğrudan, "
        "bir olayı duyuracaksan yayın. <b>[kaynak]</b> docs/14 §3.3")
    + tuzak(
        p("Yayının hata yutması, fan-out'ta sessiz veri kaybının kaynağı. "
          "Toplayıcı üç sonuç bekliyor, dallardan biri sessizce ölüyor, ve "
          "kimse haberdar olmuyor. Çözüm mekanik: <b>beklenen sonucu say</b>, "
          "runtime'ın “boşta” demesine güvenme."))
    + fig(f_fanout(), "Tek yayın, üç ajan. Eşzamanlılık abonelikten geliyor, "
                      "bir çağrıdan değil."),
)

chapter(
    "6", "AgentChat: ajan ve tool döngüsü",
    ["AssistantAgent'ın içinde ne var",
     "Tool döngüsünün varsayılan olarak neden bir adımda durduğu",
     "reflect_on_tool_use ile max_tool_iterations'ın iki ayrı anahtar olması"],
    p("Günlük iş bu katmanda. <code>AssistantAgent</code> model istemcisini, "
      "tool kaynağını ve belleği tek nesnede topluyor.")
    + fig(f_tool_loop(), "Tool döngüsü: model karar veriyor, workbench çağırıyor, "
                         "sonuç bağlama dönüyor.")
    + code(
        "agent = AssistantAgent(\n"
        "    name=\"analyst\",\n"
        "    model_client=client,\n"
        "    model_context=CompactingChatCompletionContext(...),  # bellek BURADA\n"
        "    workbench=[static_wb, mcp_wb],   # tools= ile BİRLİKTE kullanılamaz\n"
        "    model_client_stream=True,\n"
        "    max_tool_iterations=6,           # varsayılan 1\n"
        ")",
        "Bu projedeki kurulum — pipeline/conversation.py")
    + tuzak(
        p("<b>İki ayrı anahtar, iki ayrı sonuç.</b> "
          "<code>max_tool_iterations</code> varsayılanı <b>1</b>: ajan bir tool "
          "çağırır, sonucu görür ve <b>durur</b>. Zincirleme davranış sessizce "
          "imkânsız. <code>reflect_on_tool_use</code> ise ayrı bir anahtar — "
          "kapalıyken ajan sonucu okuyup cevabı yazmıyor, ham tool çıktısı "
          "doğrudan kullanıcıya gidiyor (<code>ToolCallSummaryMessage</code>). "
          "İkisi karıştırıldığında “neden cevap yazmıyor” sorusu yanlış yerde "
          "aranıyor. <b>[ölçüldü]</b> · <b>[kaynak]</b> <code>08:2298</code>"))
    + p("<code>model_context</code> verilmezse ajanın <b>belleği hiç olmuyor</b> "
        "ve bu hata da vermiyor — her tur sıfırdan başlıyor. Aynı şekilde "
        "<code>tools=</code> ile <code>workbench=</code> birlikte verilirse "
        "<code>ValueError: Tools cannot be used with a workbench.</code>")
    + fig(f_workbench_component(), "Workbench bir tool <i>kaynağı</i>: listeler "
                                   "ve çağırır. Kaynak olduğu için sarmalanabiliyor.")
    + neden(
        p("Ajana düz bir tool listesi yerine bir <b>kaynak</b> vermek, kaynağı "
          "sarmalanabilir yapıyor. Bu projedeki onay kapısı tam olarak bunu "
          "kullanıyor: <code>GatedWorkbench</code> herhangi bir workbench'i "
          "sarıp <code>before_tool_call</code> kancasını çalıştırıyor. Liste "
          "verilseydi araya girecek yer olmazdı.")),
)

chapter(
    "7", "Beş takım: sırayı kim belirliyor",
    ["Beş takım tipinin tek ayırt edici sorusu",
     "Ölçülen token farkı ve neyin ödendiği",
     "Sonlandırma koşulu olmayan takımın maliyeti"],
    p("Bir takımı diğerinden ayıran şey ajanları değil, <b>konuşma sırasının "
      "nereden geldiği</b>.")
    + fig(f_teams(), "Beş takım tipi ve sırayı belirleyen mekanizmaları.")
    + table(
        ["Takım", "Sırayı belirleyen", "Ölçülen token"],
        [["<code>SelectorGroupChat</code>", "her turdan önce bir model çağrısı", "<b>204</b>"],
         ["<code>GraphFlow</code>", "önceden çizilmiş yönlü graf", "270"],
         ["<code>RoundRobinGroupChat</code>", "sabit döngü, model karışmıyor", "274"],
         ["<code>Swarm</code>", "ajanın kendi devri (tool çağrısı)", "<b>334</b>"],
         ["<code>MagenticOneGroupChat</code>", "görev defteri tutan bir yönetici", "en pahalı"]],
        "Aynı görev, yalnız orkestrasyon değişiyor.")
    + olcum(
        p("<b>%63,7 fark</b> — Selector 204, Swarm 334. Ödenen şey zekâ değil, "
          "<b>yönlendirme özerkliği</b>. Bunun karşılaştırmaya çevirisi şu: "
          "OpenAI Agents SDK'nın tek modeli olan handoff, AutoGen'in en pahalı "
          "desenidir. <b>[ölçüldü]</b> · <code>poc/kiyas.py</code>")
        + p("Bu projede beşi de gerçekten koşturuldu ve süreleri ölçüldü: "
            "GraphFlow 13 sn · Swarm 14 sn · RoundRobin 29 sn · Selector 56 sn · "
            "MagenticOne <b>110 sn</b>. <b>[ölçüldü]</b> · "
            "<code>pipeline/teams.py</code>"))
    + fig(f_termination(), "On bir sonlandırma koşulu; birleştirilebiliyorlar.")
    + tuzak(
        p("Sonlandırma koşulu olmayan takım <b>sonsuza kadar</b> konuşuyor ve "
          "fatura gerçek. <code>MaxMessageTermination</code> bir üslup tercihi "
          "değil, maliyet tavanı."))
    + tuzak(
        p("<code>SelectorGroupChat</code> sıradaki konuşmacıyı ajanların "
          "<code>description</code> alanına bakarak seçiyor. Boşsa seçim "
          "<b>kör</b> yapılıyor — ve hata vermiyor.")
        + p("<code>Swarm</code>'ın devri özel bir tool çağrısı. Tool adı "
            "<b>küçük harfe düşüyor</b>; elle yazınca eşleşmiyor, "
            "<code>Handoff(target=X).name</code> ile üretmek gerekiyor. "
            "<b>[ölçüldü]</b>")),
)

chapter(
    "8", "Sekiz resmî desen",
    ["Kılavuzun saydığı sekiz başlık ve satır numaraları",
     "Hangisi orkestrasyon, hangisi yetenek",
     "Desenler arasındaki çelişki"],
    p("Kılavuz sekiz başlık sayıyor. Yedisi orkestrasyon deseni, sonuncusu "
      "(<i>Code Execution</i>) bir yetenek. <b>[kaynak]</b> "
      "<code>docs/05:3206</code>")
    + table(
        ["#", "Desen", "Satır", "Bir cümlede"],
        [["1", "Concurrent Agents", "05:3236", "tek yayın → çok dal → toplayıcı"],
         ["2", "Sequential Workflow", "05:3504", "her ajan bir sonrakine devrediyor"],
         ["3", "Group Chat", "05:3772", "bir yönetici konuşma sırasını dağıtıyor"],
         ["4", "Handoffs", "05:4349", "ajan işi kendisi devrediyor"],
         ["5", "Mixture of Agents", "05:4989", "aynı soru, farklı uzmanlar, birleştirici"],
         ["6", "Multi-Agent Debate", "05:5358", "birden çok tur karşılıklı eleştiri"],
         ["7", "Reflection", "05:5822", "üretici + eleştirmen, kalite döngüsü"],
         ["8", "Code Execution", "05:6188", "modelin yazdığı kod bir yürütücüde koşuyor"]])
    + two(fig(f_sequential(), "Sequential Workflow — sıra deterministik."),
          fig(f_groupchat(), "Group Chat — tek mesaj dizisi, ortak bağlam."))
    + two(fig(f_handoffs(), "Handoffs — devir bir tool çağrısı."),
          fig(f_mixture(), "Mixture of Agents — katman katman işçiler."))
    + two(fig(f_debate(), "Multi-Agent Debate — seyrek bağlı çözücüler."),
          fig(f_reflection(), "Reflection — üretici ve eleştirmen."))
    + tuzak(
        p("Resmî desenler bu konuda <b>birbiriyle çelişiyor</b>: "
          "<i>Concurrent Agents</i> sonuçları kuyrukla topluyor, "
          "<i>Mixture of Agents</i> <code>asyncio.gather(...)</code> ile — ve "
          "sessiz kaybın kaynağı ikincisi. Kılavuzdaki bir örneği kopyalarken "
          "hangisini kopyaladığını bilmek gerekiyor."))
    + neden(
        p("<i>Multi-Agent Debate</i> kılavuzda GSM8K matematik problemleri "
          "üstünde gösteriliyor — yani <b>cevabı doğrulanabilen</b> bir alanda. "
          "Bu ipucu önemli: münazara, yanlışın tespit edilebildiği yerlerde işe "
          "yarıyor. Görüş meselelerinde yalnızca daha uzun konuşuyorsun.")),
)

chapter(
    "9", "Dört sessiz varsayılan",
    ["Hata vermeden yanlış sonuç üreten dört ayar",
     "Neden en pahalı şeyin makul görünen varsayılan olduğu"],
    p("Bir çerçevenin en pahalı şeyi, makul görünen sessiz varsayılanıdır. "
      "AutoGen'de dört tane sayıldı ve dördü de sistemi <b>çalıştırıp</b> "
      "sonucu bozuyor. Hiçbiri hata vermiyor.")
    + fig(f_gotchas(), "Dört sessiz varsayılan ve ısırdıkları yer.")
    + table(
        ["Varsayılan", "Görünen davranış", "Gerçek sonuç"],
        [["<code>max_tool_iterations=1</code>", "ajan çalışıyor",
          "zincir ilk tool'dan sonra duruyor"],
         ["<code>reflect_on_tool_use=False</code>", "cevap geliyor",
          "gelen şey ham tool çıktısı"],
         ["<code>model_context</code> verilmemiş", "sohbet akıyor",
          "ajanın belleği <b>yok</b>"],
         ["<code>stop_when_idle()</code>", "toplama bitti sanılıyor",
          "bir dal çökünce bariyer erken açılıyor"]])
    + tuzak(
        p("Dış runtime verilmiş bir ajan çöktüğünde <b>fırlatmıyor, asılıyor</b>. "
          "Süre sınırı koymayan bir çağrı orada sonsuza kadar bekliyor.")),
)

# ══════════════════════════════════════════════════════════ KISIM 3 — MAF

part("3", "Microsoft Agent Framework",
     "Halef geldi. Neyi değiştirdi, neyi henüz vermiyor, ve geçiş nasıl "
     "yapılıyor — satıcının kendi kılavuzundan.")

chapter(
    "10", "Halef: dört alan",
    ["MAF'ın kendi tanımladığı dört ana alan",
     "AutoGen ve Semantic Kernel ile ilişkisi",
     "Hangi dillerde ne kadar hazır"],
    p("Microsoft Agent Framework, AutoGen ile Semantic Kernel'in <b>ikisinin "
      "birden</b> halefi. Kılavuzun kendi cümlesi: <i>“The Agent Framework is "
      "the direct successor, created by the same teams… In short, Agent "
      "Framework is the next generation of both Semantic Kernel and AutoGen.”</i> "
      "<b>[kaynak]</b>")
    + table(
        ["Alan", "İçeriği"],
        [["<b>Agents</b>", "LLM'i çağıran, tool ve MCP sunucusu kullanan tekil ajanlar"],
         ["<b>Harness Agent</b>", "“batteries-included” ajan: planlama ve todo takibi, "
          "bağlam sıkıştırma, dosya erişimi ve bellek, <b>bir daha sorma</b> tool "
          "onayı, gözlemlenebilirlik"],
         ["<b>Workflows</b>", "ajanları ve fonksiyonları açık yürütme yollarıyla "
          "bağlayan graf tabanlı akışlar"],
         ["<b>Integrations</b>", "model sağlayıcıları, ajan servisleri, tool'lar, "
          "bağlam sağlayıcıları, ara katman, değerlendirme, arayüz"]],
        "MAF'ın kendi ana sayfasındaki dörtlü. [kaynak]")
    + p("Bunların altında temel yapı taşları duruyor: model istemcileri, durum "
        "için <b>agent session</b>, bellek için <b>context providers</b>, "
        "araya girmek için <b>middleware</b>, ve tool entegrasyonu için MCP "
        "istemcileri.")
    + neden(
        p("<b>Harness Agent</b> bu listede en dikkat çekici madde. İçindekiler "
          "tanıdık gelmeli: planlama, todo takibi, bağlam sıkıştırma, dosya "
          "erişimi, tool onayı, gözlemlenebilirlik. Bunlar bir <i>harness</i>'ın "
          "işleri — yani OpenClaw'ın yaptığı şey. Halef çerçeve, harness "
          "katmanının bir kısmını <b>kütüphanenin içine</b> alıyor."))
    + tuzak(
        p("Diller eşit değil. Go sürümü <b>public preview</b> ve kılavuzun "
          "kendi uyarısına göre <i>declarative agents, RAG, CodeAct ve "
          "functional workflows henüz yok</i>. <b>[kaynak]</b>")),
)

chapter(
    "11", "Kılavuzun kendi saydığı dört fark",
    ["Microsoft'un AutoGen → MAF geçiş kılavuzundaki dört başlık",
     "Hangisinin ilerleme, hangisinin geri adım olduğu"],
    p("Geçiş kılavuzu farkları dört maddede topluyor. Alıntılar birebir. "
      "<b>[kaynak]</b>")
    + table(
        ["#", "Konu", "AutoGen", "MAF"],
        [["1", "Orkestrasyon", "olay güdümlü core + üst düzey <code>Team</code>",
          "tipli, graf tabanlı <code>Workflow</code>; kenarlarda <b>veri</b> akıyor, "
          "girdiler hazır olunca executor tetikleniyor"],
         ["2", "Tool'lar", "<code>FunctionTool</code> ile sarmalama",
          "<code>@tool</code>, şemayı otomatik çıkarıyor; ayrıca <b>hosted</b> "
          "tool'lar (kod yorumlayıcı, web arama)"],
         ["3", "Ajan davranışı",
          "<code>AssistantAgent</code> <b>tek turlu</b> — "
          "<code>max_tool_iterations</code> artırılmadıkça",
          "<code>Agent</code> <b>varsayılan olarak çok turlu</b>; nihai cevabı "
          "verene kadar tool çağırmayı sürdürüyor"],
         ["4", "Runtime", "gömülü + <b>deneysel dağıtık</b> runtime",
          "bugün <b>tek süreç</b>; dağıtık çalışma planlanan"]])
    + olcum(
        p("Üçüncü madde bu projede bağımsız olarak ölçülmüştü: "
          "<code>max_tool_iterations</code> varsayılanı 1 ve zincir bir adım "
          "sonra duruyor. Satıcının kendi kılavuzu aynı şeyi söylüyor ve halefte "
          "varsayılanı <b>tersine çevirmiş</b>. <b>[ölçüldü]</b> + <b>[kaynak]</b>"))
    + tuzak(
        p("Dördüncü madde bir <b>geri adım</b>: AutoGen'de dağıtık runtime "
          "(deneysel de olsa) var; MAF bugün tek süreç. Birden çok makineye "
          "yayılmayı planlayan bir mimari için bu, halefe geçişin bedeli."))
    + fig(f_model_clients(), "Model istemcisi katmanı — iki çerçevede de aynı rolü "
                             "oynuyor, adı ve seçenek sistemi değişiyor."),
)

chapter(
    "12", "Workflow ile GraphFlow: kontrol akışı mı, veri akışı mı",
    ["İki grafın kenarlarının farklı şeyler taşıması",
     "Bu farkın pratikte neyi değiştirdiği",
     "Duraklama ve checkpoint"],
    p("Bu, iki çerçeve arasındaki <b>en derin</b> fark, ve kılavuz onu açıkça "
      "yazıyor. <b>[kaynak]</b>")
    + table(
        ["", "AutoGen <code>GraphFlow</code>", "MAF <code>Workflow</code>"],
        [["Temel", "kontrol akışı", "veri akışı"],
         ["Kenar ne demek", "geçiş (transition)", "tipli veri yolu"],
         ["Mesaj nasıl gidiyor", "<b>herkese yayın</b>; koşullar yayınlanan "
          "içeriğe bakıyor", "belirli kenarlardan <b>yönlendiriliyor</b>"],
         ["Düğüm ne", "ajan", "executor: ajan, fonksiyon ya da alt-akış"],
         ["Duraklama", "yok", "request/response ile <b>duraklayabiliyor</b>"],
         ["Checkpoint", "yok", "<b>var</b>"]])
    + fig(f_graphflow(), "AutoGen'de graf sırayı belirliyor; mesaj zaten "
                         "herkese gitmiş oluyor.")
    + neden(
        p("Pratikteki sonuç şu: AutoGen'de <b>her ajan her mesajı görüyor</b>. "
          "Bütün katılımcılar tek bir paylaşılan <code>group_topic_type</code>'a "
          "abone; graf yalnız sıranın kimde olduğunu söylüyor. Bağlam herkes "
          "için aynı ve büyük — ayırmaya değer bir bağlam varsa bu yanlış "
          "yapı. MAF'ta kenar bir veri yolu, dolayısıyla bir executor yalnız "
          "kendisine gönderileni görüyor."))
    + tuzak(
        p("Bu fark maliyete doğrudan yansıyor. Yayın modelinde bir ajan eklemek, "
          "<b>bütün</b> konuşmayı bir kez daha modele göndermek demek. "
          "Veri akışı modelinde yalnız o kenardan geçen taşınıyor.")),
)

chapter(
    "13", "MAF'ta ölçtüklerimiz",
    ["Kurulu paketten doğrulanan API farkları",
     "Onayın çerçeveye gömülü hâli",
     "Ortak tuzak: tool sonrası boş cevap"],
    p("Aşağıdakiler bu makinede <code>agent-framework 1.14.0</code> kurulup "
      "koşturularak elde edildi. <b>[ölçüldü]</b>")
    + table(
        ["Konu", "AutoGen", "MAF 1.14.0"],
        [["Onay", "yok — <code>GatedWorkbench</code>'i biz yazdık",
          "<code>FunctionTool(approval_mode=…)</code> — <b>tool'un kendi alanı</b>"],
         ["Çağrı tavanı", "<code>max_tool_iterations</code>, <b>ajanın tamamına</b>",
          "<code>max_invocations</code>, <b>tool başına</b>"],
         ["Reddin dönüşü", "hata işaretli <code>ToolResult</code> (elle kurduk)",
          "<code>AgentResponse.user_input_requests</code> — <b>cevabın birinci "
          "sınıf alanı</b>"],
         ["Akış", "<code>run()</code> ve <code>run_stream()</code> iki ayrı metot",
          "<code>run(stream=True)</code> — tek metot, parametre"],
         ["Ara katman", "yok", "<code>ToolApprovalMiddleware</code> hazır"]])
    + code(
        "tool = af.FunctionTool(\n"
        "    func=sirket_sayisi,\n"
        "    approval_mode=\"always_require\",   # AutoGen'de karşılığı YOK\n"
        "    max_invocations=2,                # tool başına tavan\n"
        ")\n"
        "agent = af.Agent(client=client, tools=[tool],\n"
        "                 middleware=[af.ToolApprovalMiddleware(...)])\n"
        "result = await agent.run(soru, session=af.AgentSession(session_id=\"x\"))",
        "MAF'ta kapı — sarmalayıcı yazmadan")
    + olcum(
        p("<code>approval_mode=\"always_require\"</code> ile koşulduğunda tur "
          "<b>duruyor</b>: <code>finish_reason=\"tool_calls\"</code> ve cevap "
          "yerine bir <code>function_approval_request</code> dönüyor. Onay, "
          "AutoGen'de bizim elle kurduğumuz bir sonuç tipi; MAF'ta cevabın "
          "kendi alanı."))
    + tuzak(
        p("İki ölçülmüş sürpriz. Birincisi: "
          "<code>ToolApprovalMiddleware</code> bir <code>AgentSession</code> "
          "olmadan <code>RuntimeError</code> atıyor — onay, oturumu olan bir "
          "koşuya bağlı.")
        + p("İkincisi ve daha ilginci: <b>tool çağrıldığında "
            "<code>response.text</code> boş kalıyor</b>, cevap "
            "<code>messages</code> içinde. Bu, AutoGen'in "
            "<code>reflect_on_tool_use=False</code> varsayılanının tıpatıp aynı "
            "sonucu. <b>İki çerçeve de tool sonrası nihai cevabı varsayılan "
            "olarak yazdırmıyor</b> — yalnız <code>text</code>'e bakan bir "
            "arayüz, tool kullanan her turda boş ekran gösteriyor."))
    + neden(
        p("İki çerçeveyi aynı ortamda kurmayı denedik ve <b>olmadı</b>: pip'in "
          "bağımlılık çözücüsü on dakikada karar veremedi. Bu projede MAF ayrı "
          "bir sanal ortamda alt süreç olarak koşuyor "
          "(<code>pipeline/maf_runner.py</code>). Yan yana çalıştırmayı "
          "planlıyorsan bunu baştan hesaba kat. <b>[ölçüldü]</b>")),
)
