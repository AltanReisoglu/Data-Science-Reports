"""Slide content, first half: opening, core, AgentChat, gotchas.

Split out of `make_slides.py` when the bodies got long enough that the engine
was hard to find among them. The engine imports this; nothing here runs alone.
"""

# ══════════════════════════════════════════════════════ AÇILIŞ

SLIDES.append(  # noqa: F821 — injected by make_slides
    '<section class="slide cover">'
    '<div class="ceyebrow">ölçülmüş çalışma notları · Ağustos 2026</div>'
    "<h1>Bütün yüzey</h1>"
    '<p class="csub">AutoGen core · AgentChat · OpenClaw · ve bir enterprise '
    "asistanın buradan ne alabileceği</p>"
    '<div class="cmeta">vc-agent · AutoGen v0.7.5 · OpenClaw @01cc7106<br>'
    "her iddia etiketli: ölçüldü · kaynak · teyitsiz</div>"
    "</section>"
)

slide("açılış", "Bu deste ne, ve neden böyle sıralı",
      cols(
          "<p>Bir çerçeve tanıtımı değil. AutoGen'in ve OpenClaw'ın <b>ne yaptığını</b> "
          "anlatan sayfalar zaten var: 34 sayfalık bir PDF, 1.18&nbsp;MB resmî belge, "
          "iki Türkçe kılavuz. Onları tekrar etmenin bir değeri yok.</p>"
          "<p>Bu deste, o belgeleri okuyup <b>kod yazarken çarptığımız</b> şeylerin "
          "destesi. Bu yüzden sıralama pedagojik değil, <b>maliyet sırasına</b> göre: "
          "önce bilmezsen seni yakan şeyler, sonra bilmek güzel olanlar.</p>"
          "<p>Somut örnek: <code>max_tool_iterations</code>'ın varsayılanı, "
          "AgentChat kılavuzunun 2.000. satırında geçen bir parametre. Bizde ikinci "
          "sıraya yakın, çünkü onu bilmeden kurulan her ajan <b>yanlış cevap "
          "veriyor</b> ve hiçbir yerde hata çıkmıyor.</p>",
          "<p>Üç etiket var ve ciddiye alınmaları gerekiyor:</p>"
          f"<p>{tag('m')} bir koşudan veya kaynak koddan çıkan sayı. Yeniden "
          "üretilebilir — nasıl ölçüldüğü slaytta ya da dipnotta yazıyor.</p>"
          f"<p>{tag('k')} bir belge böyle diyor; atıf slaytta. Doğruluğunu biz değil "
          "o belge üstleniyor.</p>"
          f"<p>{tag('t')} inanıyoruz, ölçmedik. Destedeki en tehlikeli etiket bu — "
          "çünkü bir slaytı kimse durdurup kontrol etmez. Bu yüzden ayrı bir renkle "
          "duruyor.</p>"),
      foot="Etiketsiz bir cümle varsa o bir tanımdır, bir iddia değil.")

slide("açılış", "Tez — tek slaytta",
      "<p class='lead'>Üç sistemi yan yana koyunca ortaya çıkan tek cümle:</p>"
      + quote("<b>Ajan döngüsü artık ilginç değil.</b> Model çağır, tool çağır, "
              "sonucu geri ver — bunu herkes yapıyor ve AutoGen'de zaten hazır. "
              "Ayrımı yaratan şey <b>döngüyü kuşatan kontrol düzlemi</b>: neyin "
              "çağrılabildiği, kimin onayladığı, neyin kaydedildiği — ve bir "
              "mekanizmanın <b>neyi kanıtlamadığının</b> yazılı olması.")
      + cols(
          "<p><b>AutoGen</b> motoru veriyor: aktör modeli, takımlar, akış, "
          "sonlandırma. Kontrol düzlemi <b>yok</b> — \"kapı\" diye bir kavramı yok. "
          "Bir ajanın dış dünyaya yaptığı çağrıyı durdurup insana sormak istiyorsan, "
          "onu kendin kurman gerekiyor.</p>",
          "<p><b>OpenClaw</b> tam tersi: kontrol düzlemi zengin — onay, kapsam, "
          "denetim, sandbox, dış içerik sınırı. Ama ajan döngüsünün kendisi sıradan; "
          "AutoGen'in takım soyutlamalarına karşılık gelen bir şey yok.</p>")
      + "<p>Bizim yaptığımız ikisini birleştirmek oldu. Atlas için önerilen de bu: "
        "motoru bir yerden, kuşatmayı başka yerden almak.</p>",
      foot="Bu tezin gerekçesi Kısım III (ölçülen tuzaklar) ve Kısım VI (enterprise).")

# ══════════════════════════════════════════════════════ KISIM I — core

part("I", "AutoGen core", "Aktör modeli, kimlik, topic, yayın — ve dört tuzak",
     ["Aktör modeli ve runtime", "Ajan kimliği: type + key",
      "İki iletişim biçimi ve asimetrileri", "Topic, abonelik, source→key kuralı",
      "Fan-out / fan-in ve ClosureAgent", "Müdahale ve gözlemlenebilirlik",
      "Mesaj sözleşmesi, dağıtık runtime, protokoller",
      "Dokuz desen ve ölçülen maliyetleri"])

slide("I · core", "Aktör modeli — neden böyle kurulmuş",
      fig(f_actor(), "Runtime mesajı taşır; ajanlar birbirinin referansını tutmaz.",
          cap_mm=41)
      + cols(
          "<p>Ajanlar birbirini <b>çağırmıyor</b>. Bir ajan başka bir ajanın nesnesini "
          "elinde tutmuyor, metodunu çağırmıyor. Runtime'a bir mesaj veriyor; teslimatı "
          "runtime yapıyor.</p>"
          "<p>Bunun bir bedeli var: araya bir dolaylılık katmanı giriyor ve "
          "\"kim kimi çağırdı\" sorusu artık yığın izinden okunmuyor.</p>",
          "<p>Karşılığında üç şey geliyor:</p>"
          "<p>· <b>Ajan eklemek çağıran kodu değiştirmiyor.</b> Dördüncü bir analist "
          "eklediğinde koordinatör dosyasına dokunmuyorsun — yeni ajan aynı topic'e "
          "abone oluyor, o kadar.<br>"
          "· <b>Teslimat noktası tek.</b> Bütün mesajlar tek bir yerden geçtiği için "
          "müdahale, log ve ölçüm oraya takılıyor.<br>"
          "· <b>Aynı sınıftan çok örnek bedava</b> — bir sonraki slaytın konusu.</p>"),
      foot="autogen_core · SingleThreadedAgentRuntime")

slide("I · core", "Ajan kimliği — bir şey değil, iki şey",
      fig(f_identity(), cap_mm=49)
      + cols(
          f"<p>{tag('k')} <code>AgentId = (type, key)</code>. Çoğu core hatası bu "
          "ikisini tek şey sanmaktan çıkıyor.</p>"
          "<p><code>type</code> <b>davranış</b>: hangi sınıf, hangi handler'lar. "
          "Kayıt (<code>register</code>) bu düzeyde yapılır.</p>"
          "<p><code>key</code> <b>örnek</b>: aynı davranışın hangi kopyası, hangi "
          "durumu taşıyor. Kaydedilmez.</p>",
          "<p>Örnekler <b>talep üzerine</b> doğuyor. <code>analyst/hn</code>'e ilk "
          "mesaj gittiğinde runtime o örneği yaratıyor, fabrika fonksiyonunu "
          "çağırıyor, sonra teslim ediyor.</p>"
          "<p>Yani \"ajanı kaydetmedim ama mesaj gitti\" diye bir durum yok; ama "
          "\"üç ajan kaydettim\" de yanlış bir cümle — <b>bir tip</b> kaydettin, "
          "üç örnek doğdu.</p>")
      + quote("Pratik sonuç: durumu <code>key</code> taşır. Aynı <code>key</code>'e "
              "iki kez mesaj gönderirsen aynı örneğe, aynı belleğe gider. Farklı "
              "<code>key</code> demek sıfırdan bir ajan demektir.", "b"))

slide("I · core", "İki iletişim biçimi — ve simetrik olmadıkları",
      fig(f_send_vs_publish(), cap_mm=51)
      + cols(
          "<p><code>send_message</code> bir <b>fonksiyon çağrısı gibidir</b>: tek "
          "alıcı, dönüş değeri var, hata çağırana fırlar. Bir ajandan somut bir "
          "cevap bekliyorsan bu.</p>"
          "<p><code>publish_message</code> bir <b>duyurudur</b>: kaç dinleyici "
          "olduğunu bilmezsin, dönüş değeri yoktur, ve handler içinde patlayan "
          "istisna sana ulaşmaz — yalnız loglanır.</p>",
          f"<p>{tag('m')} Bu asimetriyi ölçtük. Yayın yolunda handler'ın attığı "
          "istisna çağırana <b>ulaşmıyor</b>; aynı istisna doğrudan mesajda "
          "çağırana fırlıyor.</p>"
          "<p>Sıfır abone de geçerli bir sonuç: kimse dinlemiyorsa "
          "<code>publish_message</code> sessizce başarılı olur. Yani "
          "\"aboneliği yazmayı unuttum\" hatası da sessizdir.</p>")
      + quote("Sonuç: fan-out mimarisinde bir dal sessizce ölebilir ve toplayıcı "
              "sonsuza kadar bekler. Bu yüzden sayaç <b>hata durumunda da</b> "
              "ilerlemek zorunda — Kısım III'te kodu var.", "r"))

slide("I · core", "Topic nedir — gerçekten",
      fig(f_topic(), cap_mm=52)
      + cols(
          "<p>Topic bir <b>kanal adı değil</b>, iki parçalı bir adres. "
          "<code>type</code> ne olduğunu söyler (\"bu bir görev\"), "
          "<code>source</code> hangi iş için olduğunu (\"7 numaralı koşu\").</p>"
          "<p>Abonelik <code>type</code>'a yapılır: <code>TypeSubscription("
          "\"task\", \"analyst\")</code> demek \"task tipindeki her yayını analyst "
          "tipine ver\" demektir.</p>",
          f"<p>{tag('k')} <b>Kural — core kılavuzu 05:670:</b> teslim edilen "
          "örneğin <code>key</code>'i, topic'in <code>source</code>'undan gelir. "
          "Doğrudan, dönüşümsüz.</p>"
          "<p>Yani <code>source</code>'u her istekte değiştirirsen her istekte "
          "<b>yeni bir ajan örneği</b> doğar. Önceki örneğin biriktirdiği durum "
          "kaybolmaz — erişilemez hale gelir, ki bu daha kötüdür.</p>")
      + quote("En sık görülen core hatası bu, ve hata vermez. Sistem çalışır, "
              "ajanlar cevap verir, sadece hiçbiri bir öncekini hatırlamaz.", "r"),
      foot="Doğru kullanım: source = iş kimliği. Aynı işin bütün adımları aynı source'u paylaşır.")

slide("I · core", "publish anını kim belirler — ve parametreleri kim verir",
      cols(
          "<p>Kimse otomatik yapmıyor. <b>Yazan kişi</b> belirliyor: kodunda "
          "<code>await runtime.publish_message(msg, TopicId(...))</code> satırını "
          "nereye koyduysan, publish anı orasıdır.</p>"
          "<p>Parametreleri de yazan kişi veriyor: mesaj nesnesinin içeriği, topic "
          "type'ı, source'u. Framework hiçbirini türetmiyor, tahmin etmiyor, "
          "varsayılan atamıyor.</p>"
          "<p>Bir handler'ın içinden yayın yapılabilir — o zaman zincir kurulur: "
          "A biter, B'yi tetikler, B biter, C'yi tetikler. Ama bu zinciri de yazan "
          "kişi kuruyor.</p>",
          quote("Bu soru bize üç farklı biçimde soruldu ve cevabı hep aynı çıktı: "
                "<b>AutoGen'de örtük hiçbir şey yok.</b> \"Kim tetikledi\" "
                "sorusunun cevabı her zaman senin yazdığın satırdır.", "b")
          + "<p>Bu, LangGraph'tan temel farkı. LangGraph'ta kenarlar tetikler — "
            "grafiği çizersin, çalıştırma sırasını çerçeve türetir. AutoGen'de "
            "grafiği <b>sen koşturursun</b>.</p>"
          + "<p>Bedeli: daha çok kod. Karşılığı: sürpriz yok.</p>"))

slide("I · core", "Fan-out / fan-in — eşzamanlılık nereden geliyor",
      fig(f_fanout(), "Tek publish üç ajanı birden uyandırır; toplama ayrı bir ajandır.",
          cap_mm=44)
      + cols(
          f"<p>{tag('k')} Eşzamanlılık ajanlardan gelmiyor, <b>abonelikten</b> "
          "geliyor. Üç analist de aynı topic tipine abone olduğu için tek bir yayın "
          "üçünü birden uyandırıyor; runtime üçünü paralel koşturuyor.</p>"
          "<p>Yani \"paralel çalıştır\" diye bir çağrı yok. Paralellik, aynı "
          "duyuruyu birden fazla kişinin dinlemesinin doğal sonucu.</p>",
          "<p>Toplama için <b>ayrı bir ajan</b> gerekiyor ve bunun sebebi mekanik: "
          "<code>publish_message</code> hiçbir şey döndürmüyor. Dönüş değeri olmayan "
          "bir çağrıdan sonuç toplayamazsın.</p>"
          "<p>O yüzden analistler sonuçlarını <b>ikinci bir topic'e</b> yayınlıyor, "
          "toplayıcı ona abone oluyor ve kaç tane beklediğini kendisi sayıyor.</p>")
      + quote("Sayacı <b>sen</b> tutuyorsun. Framework \"üç dal vardı, üçü de "
              "bitti\" demiyor — kaç dal olduğunu bilmiyor bile.", "r"))

slide("I · core", "ClosureAgent — sınıf yazmadan ajan",
      cols(
          code("async def collect(ctx, message, cx):\n"
               "    await queue.put(message)\n\n"
               "await ClosureAgent.register_closure(\n"
               "    runtime, \"collector\", collect,\n"
               "    subscriptions=lambda: [\n"
               "        TypeSubscription(\"result\", \"collector\")\n"
               "    ],\n"
               ")"),
          "<p>Toplayıcının bütün davranışı tek bir fonksiyon kadarsa, ona bir sınıf "
          "yazmak gürültüdür. <code>ClosureAgent</code> bunu kapatıyor: bir "
          "fonksiyon veriyorsun, o ajan oluyor.</p>"
          "<p>Tipik kullanım: sonuçları bir <code>asyncio.Queue</code>'ya yazmak ve "
          "runtime'ın dışındaki kodun oradan okuması. Bu, aktör dünyasından normal "
          "async dünyaya çıkışın en temiz kapısı.</p>"
          f"<p>{tag('t')} Dikkat edilecek yer: runtime kapanışı ile kuyruk tüketimi "
          "arasındaki yarışı sen yönetiyorsun. <code>stop_when_idle()</code> "
          "döndüğünde kuyrukta okunmamış eleman kalabilir.</p>"))

slide("I · core", "Müdahale — kapının AutoGen'deki tek atası",
      fig(f_intervention(), cap_mm=54)
      + cols(
          "<p><code>InterventionHandler</code> runtime'a takılıyor. Her "
          "<code>on_send</code> ve <code>on_publish</code> ondan geçiyor — yani "
          "sistemdeki <b>bütün</b> mesaj trafiği tek bir noktadan görünüyor.</p>"
          "<p><code>DropMessage</code> döndürürse mesaj yok oluyor. Alıcı hiç "
          "haberdar olmuyor; gönderen de bir hata almıyor.</p>",
          quote("Bu, AutoGen'in onaya en yakın parçası. Ama <b>onay değil</b>: "
                "insana sormuyor, gerekçe döndürmüyor, kaydı yok, ve kararı geri "
                "bildirmiyor. Sadece bir süzgeç.", "b")
          + "<p>Bizim kapımız bu yüzden burada değil, bir katman yukarıda — "
            "<code>workbench</code> düzeyinde kuruldu. Orada tool'un adı, "
            "argümanları ve bir gerekçe döndürme imkânı var (Kısım IV).</p>"))

slide("I · core", "Gözlemlenebilirlik — hazır gelen kısım",
      cols(
          code("import logging\n"
               "from autogen_core import EVENT_LOGGER_NAME\n\n"
               "logging.getLogger(EVENT_LOGGER_NAME)\\\n"
               "       .addHandler(MyHandler())"),
          "<p>AutoGen yapılandırılmış olayları <b>zaten</b> yayıyor; tek yapman "
          "gereken standart <code>logging</code> üstünden dinlemek.</p>"
          "<p><code>LLMCallEvent</code> — istem ve tamamlama token sayıları<br>"
          "<code>LLMStreamStartEvent</code> / <code>LLMStreamEndEvent</code><br>"
          "<code>ToolCallEvent</code> — hangi tool, hangi argümanlarla</p>"
          f"<p>{tag('m')} Bizim maliyet muhasebemiz tamamen bunun üstünde. Ayrı bir "
          "sayaç yazmadık; token sayıları modelden geldiği gibi kaydediliyor, "
          "tahmin edilmiyor.</p>")
      + quote("Bir çerçevenin ölçülebilir olması, ölçüm kodu yazmak zorunda "
              "kalmamak demektir. Burada olay akışı hazır — kullanmamak bir tercih, "
              "eksiklik değil.", "g"))

slide("I · core", "Mesaj sözleşmesi ve serileştirme",
      cols(
          code("@dataclass\n"
               "class Signal:\n"
               "    source: str\n"
               "    url: str\n"
               "    score: float"),
          "<p>Mesajlar düz veri sınıfları — <code>dataclass</code> ya da pydantic "
          "modeli. Runtime onları serileştirebilmek zorunda, çünkü dağıtık runtime'da "
          "süreç sınırını geçiyorlar.</p>"
          f"<p>{tag('k')} Tek süreçte serileştirme yapılmıyor (gereksiz kopya olurdu), "
          "ama <b>sözleşme aynı</b>. Yani tek süreçte doğru yazılmış bir tasarım "
          "dağıtığa taşındığında mesaj tarafında hiçbir şey değişmiyor.</p>")
      + quote("Pratik kural: mesajın içine canlı nesne koyma — açık dosya tanıtıcısı, "
              "veritabanı bağlantısı, model istemcisi. Serileşemeyen bir mesaj tek "
              "süreçte çalışır, dağıtıkta patlar, ve patladığı yer taşındığın gündür.",
              "r"))

slide("I · core", "Tek süreç ve dağıtık runtime",
      table(["", "SingleThreadedAgentRuntime", "GrpcWorkerAgentRuntime"],
            [["nerede koşar", "tek süreç, tek iş parçacığı", "süreç/makine başına bir worker"],
             ["taşıma", "bellek içi kuyruk", "gRPC + bir host süreci"],
             ["ne zaman", "geliştirme ve çoğu üretim", "ajanlar ayrı ölçeklenmeli"],
             ["ajan kodu", "aynı", "aynı — yalnız kayıt ve başlatma değişir"]])
      + cols(
          f"<p>{tag('k')} Aynı ajan sınıfı ikisinde de değişmeden koşuyor. Ayrım "
          "kayıt ve başlatmada — davranışta değil. Bu, aktör modelinin asıl "
          "kazandırdığı şey.</p>",
          quote("Pratikte tek süreçle başlamamak için sebep yok. Dağıtık runtime bir "
                "<b>ölçek cevabı</b>; ölçek problemi ölçülmeden ödenmemesi gereken "
                "bir karmaşıklık.", "b")))

slide("I · core", "Konuştuğu protokoller",
      table(["protokol", "ne çözer", "AutoGen'deki yeri"],
            [["<b>MCP</b>", "tool sunucusu ↔ istemci", "<code>McpWorkbench</code> — birinci sınıf"],
             ["<b>A2A</b>", "ajan ↔ ajan, kurumlar arası", "dışarıdan; çekirdekte yok"],
             ["<b>ACP</b>", "ajan ↔ istemci oturumu", "dışarıdan"],
             ["OpenAI uyumlu HTTP", "model çağrısı", "<code>OpenAIChatCompletionClient</code>"]])
      + cols(
          f"<p>{tag('k')} Bizim kullandığımız MCP. OpenClaw köprüsü de onun üstünde "
          "ve <b>iki yönlü</b>: onların tool'ları bize geliyor, bizim tool'larımız "
          "onlara gidiyor. Aynı protokol iki yöne de çalışıyor.</p>",
          quote("A2A ve ACP'nin bu projede karşılığı yok. Olmadığını söylemek, "
                "“destekliyor” demekten daha yararlı — çünkü ikincisi denemeye "
                "çağırır ve deneme boşa gider.", "b")))

slide("I · core", "Rakiplerle — ne zaman hangisi",
      table(["çerçeve", "güçlü yanı", "bizim için sorunu"],
            [["<b>AutoGen</b>", "aktör modeli, takımlar, olay akışı", "bakım modu okuması"],
             ["LangGraph", "durum makinesi, kalıcılık, tekrar oynatma", "grafik dili ağır, öğrenme eğrisi dik"],
             ["CrewAI", "hızlı başlangıç, rol metaforu", "kontrol az; kapı/politika kavramı yok"],
             ["OpenAI Agents SDK", "sadelik, az kavram", "tek sağlayıcıya yakın duruyor"],
             ["Google ADK", "kurumsal entegrasyon", "ekosistem bağı"]])
      + f"<p>{tag('t')} Bu tablo bir <b>okuma</b>, bir kıyaslama koşusu değil. "
        "Ölçtüğümüz tek çerçeve AutoGen; diğerleri kendi belgelerinden okundu. "
        "Aynı görevi beşine de koşturup token ve süre karşılaştırması yapmadık — "
        "yapılsaydı bu tablo değişebilirdi.</p>",
      foot="docs/09-framework-karsilastirma.md · docs/14-autogen-protokoller-ve-farklar.md")

slide("I · core", "Resmî çok-ajan desenleri — sekizi",
      table(["desen", "05 satırı", "ne zaman"],
            [["Concurrent Agents", "3236", "bağımsız kaynaklar paralel taranacak"],
             ["Sequential Workflow", "3504", "adımlar birbirine bağımlı"],
             ["Group Chat", "3772", "tartışma gerçekten gerekiyor"],
             ["Handoffs", "4349", "devretme mantığı ajanın kendisinde"],
             ["Mixture of Agents", "4989", "aynı soruya farklı uzmanlıklar"],
             ["Multi-Agent Debate", "5358", "birden çok tur karşılıklı eleştiri"],
             ["Reflection", "5822", "kalite kritik, ikinci göz lazım"],
             ["Code Execution", "6188", "modelin yazdığı kod koşacak"]])
      + quote(f"{tag('k')} Bu liste <b>core kılavuzunun kendi bölümlemesi</b> "
              "(05:3206'dan itibaren), bizim tasnifimiz değil. Satır numarası "
              "verilmesinin sebebi bu: her satır kaynak metinde açılabilir.", "b"),
      foot="Planlayıcı-yürütücü, kara tahta, yönlendirici gibi desenler literatürde var ama core kılavuzunda YOK.")

slide("I · core", "Runtime yaşam döngüsü",
      cols(
          code("runtime = SingleThreadedAgentRuntime()\n"
               "await Agent.register(runtime, \"worker\", factory)\n"
               "runtime.start()\n"
               "await runtime.publish_message(task, topic)\n"
               "await runtime.stop_when_idle()"),
          "<p><code>start()</code> mesaj işleyen döngüyü açar — ondan önce yapılan "
          "yayınlar kuyruğa girer ama işlenmez.</p>"
          "<p><code>stop_when_idle()</code> kuyruk boşalınca döner; yani bütün "
          "zincir tamamlanmış olur.</p>"
          "<p><code>stop()</code> ise <b>hemen</b> durur — işlenmemiş mesajlar "
          "kuyrukta kalır ve kaybolur.</p>")
      + quote("Testlerde <code>stop()</code> kullanmak, yarısı işlenmiş bir sistemi "
              "doğru sanmanın en hızlı yoludur. Sonuç yeşil görünür çünkü assert'ler "
              "henüz gelmemiş mesajları beklemez.", "r"))

# ══════════════════════════════════════════════════════ KISIM II — AgentChat

part("II", "AutoGen AgentChat", "Ajanlar, takımlar, mesajlar, akış — kullanıma hazır katman",
     ["core ile ilişkisi", "AssistantAgent ve tool döngüsü", "Workbench ve MCP",
      "Beş takım tipi", "GraphFlow ve DAG kurma", "On bir sonlandırma koşulu",
      "Mesaj ve olay türleri", "Bağlam, cache sınırı, durum"])

slide("II · agentchat", "İki katman — hangisi ne zaman",
      table(["", "autogen_core", "autogen_agentchat"],
            [["soyutlama", "aktör, mesaj, topic, abonelik", "ajan, takım, görev"],
             ["kim yazar", "altyapı/protokol kuran", "uygulama yazan"],
             ["eşzamanlılık", "abonelikten doğar", "takım tipinden gelir"],
             ["hata", "yayında loglanır, kaybolur", "TaskResult'ta görünür"],
             ["ne zaman", "kendi desenini kuruyorsan", "iş yaptırıyorsan"]])
      + cols(
          "<p>AgentChat core'un <b>üstünde</b> duruyor, yerine değil. Bir "
          "<code>AssistantAgent</code> aslında bir <code>RoutedAgent</code>; bir "
          "takım da arka planda bir runtime kuruyor.</p>",
          quote("Aşağı inmek her zaman mümkün ve gerektiğinde iniyoruz. Ama önce "
                "yukarıdan denemek doğru sıra: AgentChat'in çözdüğü bir problemi "
                "core'da yeniden çözmek, çoğu zaman aynı kodu daha az testle "
                "yazmak demek.", "b")))

slide("II · agentchat", "AssistantAgent ve tool döngüsü",
      fig(f_tool_loop(), cap_mm=52)
      + cols(
          f"<p>{tag('m')} <b>En pahalı varsayılan burada:</b> "
          "<code>max_tool_iterations=1</code>.</p>"
          "<p>Model tool'u çağırıyor, tool koşuyor, sonuç dönüyor — ve tur "
          "<b>bitiyor</b>. Model o sonucu hiç görmüyor. Kullanıcıya giden cevap, "
          "tool'un bulduğu şeyi içermiyor.</p>",
          "<p>Hiçbir uyarı çıkmıyor. Sistem çalışıyor, ajan cevap veriyor, cevap "
          "makul görünüyor — sadece tool'un getirdiği bilgi orada değil.</p>"
          "<p>Bizde 6'ya çekildi. Doğru sayı göreve bağlı; doğru olmayan tek şey "
          "varsayılanı <b>bilmeden</b> bırakmak.</p>")
      + quote("Bu tuzağın imzası şu: \"tool çağrılıyor mu?\" diye loga bakarsın, "
              "çağrılıyor. \"Sonuç dönüyor mu?\" diye bakarsın, dönüyor. Yine de "
              "cevap yanlış. Çünkü eksik olan çağrı değil, <b>ikinci model turu</b>.",
              "r"))

slide("II · agentchat", "Workbench — tool'ların toplandığı yer",
      cols(
          "<p><code>StaticWorkbench</code> — elindeki Python fonksiyonları. Şemaları "
          "imzadan türetiliyor.</p>"
          "<p><code>McpWorkbench</code> — bir MCP sunucusunun tool'ları; stdio ile "
          "alt süreç ya da HTTP ile uzak sunucu. Ajan açısından fark yok.</p>"
          f"<p>{tag('k')} Ajan hangi workbench'le konuştuğunu <b>bilmiyor</b>. "
          "Arayüz tek: <code>list_tools()</code> ve <code>call_tool()</code>.</p>"
          "<p>Kapıyı kurmayı mümkün kılan tam olarak bu: "
          "<code>GatedWorkbench</code> de sadece bir workbench.</p>",
          code("workbench = McpWorkbench(\n"
               "    StdioServerParams(\n"
               "        command=\"openclaw\",\n"
               "        args=[\"mcp\", \"serve\"],\n"
               "    )\n"
               ")\n"
               "agent = AssistantAgent(\n"
               "    \"vc\", model_client=client,\n"
               "    workbench=workbench,\n"
               "    max_tool_iterations=6,\n"
               ")")))

slide("II · agentchat", "Beş takım tipi",
      fig(f_teams(), cap_mm=40)
      + table(["takım", "sırayı kim belirler", "tipik kullanım", "maliyet notu"],
              [["RoundRobinGroupChat", "sabit döngü", "yazar–eleştirmen çiftleri", "öngörülebilir"],
               ["SelectorGroupChat", "model, her turda", "rolleri belirsiz tartışma", "her turda ek çağrı"],
               ["Swarm", "ajanın kendisi (handoff)", "devretme akışları", "devir sayısına bağlı"],
               ["MagenticOne", "planlayıcı ajan", "açık uçlu görevler", "en pahalısı"],
               ["GraphFlow", "önceden çizilmiş DAG", "bilinen boru hattı", "ek model çağrısı yok"]])
      + f"<p>{tag('k')} Beşi de aynı arayüzü sunuyor: <code>run()</code> / "
        "<code>run_stream()</code> → <code>TaskResult</code>. Takım değiştirmek "
        "çağıran kodu değiştirmiyor.</p>")

slide("II · agentchat", "GraphFlow — bizim taramanın kullandığı",
      fig(f_graphflow(), cap_mm=50)
      + cols(
          code("b = DiGraphBuilder()\n"
               "b.add_node(intake)\n"
               "for a in analysts:\n"
               "    b.add_node(a)\n"
               "    b.add_edge(intake, a)\n"
               "    b.add_edge(a, join)\n"
               "flow = GraphFlow(b.build(),\n"
               "                 participants=[...])"),
          "<p>Grafiği <b>önceden</b> çiziyorsun; sırayı model belirlemiyor. Bu, "
          "boru hattı belliyse hem daha ucuz hem daha öngörülebilir.</p>"
          f"<p>{tag('m')} <b>Bir düzeltme:</b> uzun süre taramanın core pub/sub "
          "kullandığını sandık ve panele o şemayı çizdik. Kod okununca çıktı — "
          "tarama <code>graph.py</code>'deki GraphFlow'u kullanıyor, "
          "<code>fanin.py</code>'yi <b>hiç çağırmıyor</b>.</p>"
          "<p>Şimdi bunu bir test tutuyor.</p>"),
      foot="test_the_scan_is_graphflow_not_core_pubsub — varsayımı değil, kodu doğrulayan test")

slide("II · agentchat", "On bir sonlandırma koşulu",
      cols(
          "<p><code>MaxMessageTermination</code> — mesaj sayısı<br>"
          "<code>TokenUsageTermination</code> — token bütçesi<br>"
          "<code>TimeoutTermination</code> — duvar saati<br>"
          "<code>TextMentionTermination</code> — bir kelime geçti<br>"
          "<code>TextMessageTermination</code> — belirli ajandan metin<br>"
          "<code>SourceMatchTermination</code> — belirli ajan konuştu</p>",
          "<p><code>HandoffTermination</code> — devir istendi<br>"
          "<code>StopMessageTermination</code> — açık dur mesajı<br>"
          "<code>FunctionCallTermination</code> — belirli tool çağrıldı<br>"
          "<code>FunctionalTermination</code> — kendi yazdığın koşul<br>"
          "<code>ExternalTermination</code> — dışarıdan tetiklenen</p>")
      + code("termination = MaxMessageTermination(20) | TokenUsageTermination(50_000)")
      + f"<p>{tag('k')} <code>|</code> ile birleşenlerden <b>biri</b> yeterli, "
        "<code>&amp;</code> ile <b>hepsi</b> gerekli.</p>"
      + quote("Üretimde en az bir <b>sert tavan</b> (mesaj ya da token) olmadan takım "
              "koşturmak, faturayı modelin kararına bırakmaktır. Semantik koşullar "
              "— \"TERMINATE yaz\" gibi — model uymazsa çalışmaz.", "r"))

slide("II · agentchat", "Mesaj ve olay türleri",
      fig(f_message_types(), cap_mm=66)
      + cols(
          "<p><b>Ayrım:</b> mesaj konuşmanın parçası, olay ise ne olduğunun "
          "anlatımı. Mesajlar bağlama girer; olaylar gözlem içindir.</p>",
          f"<p>{tag('k')} <code>StructuredMessage</code> pydantic şeması taşıyor — "
          "serbest metin ayrıştırmak yerine tip alıyorsun. Bizim tarama sonuçları "
          "bunu kullanıyor: aday nesnesi metin değil, alanları belli bir yapı.</p>"))

slide("II · agentchat", "run ve run_stream",
      cols(
          code("result = await team.run(task=\"...\")\n"
               "print(result.stop_reason)"),
          code("async for ev in team.run_stream(task=\"...\"):\n"
               "    if isinstance(ev, TaskResult):\n"
               "        final = ev\n"
               "    else:\n"
               "        handle(ev)  # ara olaylar"))
      + quote("<code>run_stream</code> ara olayları verir <b>ve</b> son "
              "<code>TaskResult</code>'ı akışın son elemanı olarak verir. Döngüde "
              "tip kontrolü yapmazsan sonucu bir olay sanıp işler, sonra da "
              "\"sonuç gelmedi\" diye ararsın.", "r")
      + f"<p>{tag('m')} Panelimiz tam olarak bu akıştan besleniyor: her olay tipi "
        "bir mekanizma şemasına eşleniyor, ve ekrana o an koşan şey çiziliyor "
        "(Kısım IV).</p>")

slide("II · agentchat", "Bağlam ve cache sınırı",
      fig(f_context(), cap_mm=48)
      + cols(
          "<p><code>ChatCompletionContext</code> modele ne gideceğine karar veriyor. "
          "Bizimki üstüne sıkıştırma ekliyor: eşik aşılınca eski turlar bir özete "
          "iniyor, özet bağlamda kalıyor.</p>",
          f"<p>{tag('k')} <b>Sıralama mimari bir karar.</b> Değişmeyen şey (sistem "
          "talimatı, tool tanımları, yetenek dizini) önde durursa prompt cache'i "
          "turlar arasında yeniden kullanılır. Değişken bir şey öne kaçarsa cache "
          "her turda bozulur ve ödeme her turda tam yapılır.</p>")
      + quote("Bu yüzden \"prompt'u kısalt\" tavsiyesi çoğu zaman yanlış hedef. "
              "Asıl soru <b>neyin sabit kalabildiği</b>. Sabit kısım büyük olabilir; "
              "yeter ki gerçekten sabit olsun.", "g"),
      foot="Aynı ilkenin OpenClaw'daki karşılığı: Tool Search dizini — Kısım VI.")

slide("II · agentchat", "Durum: kaydet ve geri yükle",
      cols(
          code("state = await team.save_state()\n"
               "# … süreç yeniden başlar …\n"
               "await team.load_state(state)"),
          f"<p>{tag('t')} Kaydedilen şey <b>takımın</b> durumu: katılımcı ajanların "
          "bağlamları, sıra bilgisi, sonlandırma sayaçları.</p>"
          "<p>Workbench'in durumu <b>değil</b>. MCP oturumu geri gelmiyor; süreç "
          "yeniden başladığında yeniden kuruluyor. Tool tarafında kalıcı bir şey "
          "istiyorsan onu kendin saklaman gerekiyor.</p>"
          "<p>Bunu ölçmedik — belge böyle söylüyor ve mimariyle tutarlı.</p>"))

slide("II · agentchat", "AutoGen Studio — ve neden ayrı kurduk",
      cols(
          f"<p>{tag('m')} <code>autogenstudio 0.4.2.2</code>, "
          "<code>autogen-core&lt;0.6</code> pinliyor ve 0.5.7 kuruyor. Bizim proje "
          "0.7.5 üstünde çalışıyor.</p>"
          "<p>Aynı ortama kurmak projeyi <b>geriye</b> düşürürdü — ve bunu sessizce "
          "yapardı, çünkü pip uyumlu bir sürüm bulup indirir.</p>",
          "<p>Çözüm: ayrı sanal ortam, <code>~/autogenstudio/.venv</code>. Studio "
          "8080'de koşuyor, proje kendi 0.7.5'iyle kalıyor.</p>"
          f"<p>{tag('m')} Kurulumdan <b>önce</b> <code>.gitignore</code>'a eklendi: "
          "<code>myapp/</code>, <code>autogenstudio/</code>, "
          "<code>.autogenstudio/</code> — appdir veritabanı API anahtarı tutuyor.</p>")
      + quote("Bir aracı denemek için ana projenin bağımlılıklarını düşürmek, "
              "denemenin maliyetini projeye yazmaktır. Ayrı ortam beş dakika, "
              "geri alma yarım gün.", "r"))

# ══════════════════════════════════════════════════════ KISIM III — tuzaklar

part("III", "Ölçülen tuzaklar", "Belgeden değil, çarparak öğrenilenler",
     ["model_info zorunluluğu", "Sessiz varsayılanlar", "Sessiz veri kaybı",
      "Dört isim karmaşası", "Bakım modu tezi", "Desen seçmenin faturası"])

slide("III · tuzaklar", "model_info — OpenAI uyumlu her endpoint'te",
      cols(
          code("OpenAIChatCompletionClient(\n"
               "    model=\"deepseek-v3\",\n"
               "    base_url=\"https://api.deepseek.com/v1\",\n"
               ")\n"
               "# ValueError: model_info is required"),
          f"<p>{tag('m')} AutoGen model adını bilinen bir OpenAI modeli olarak "
          "tanıyamazsa <b>başlamayı reddediyor</b>. Sebebi makul: vision var mı, "
          "function calling destekliyor mu, JSON çıktı verebiliyor mu — bunları "
          "bilmeden ajan kurarsa yanlış varsayımla koşar.</p>"
          "<p>Çözüm <code>model_info</code>'yu elle vermek. Bizde "
          "<code>config.LIVE_MODEL_INFO</code> bunu yapıyor, bu yüzden endpoint "
          "değiştirdiğimizde bu hata artık çıkmıyor.</p>")
      + quote("Bugün DeepSeek'i denerken ölçtük: tuzak <b>tetiklenmedi</b>, çünkü "
              "config zaten dolduruyordu. Hata 401'de çıktı — yani sorun anahtardı, "
              "model adı değil. Kapatılmış bir tuzağın kanıtı, başka bir hatanın "
              "önce gelmesidir.", "g"))

slide("III · tuzaklar", "Sessiz varsayılanlar",
      table(["ayar", "varsayılan", "ne olur", "nasıl fark edilir"],
            [["<code>max_tool_iterations</code>", "1",
              "model tool sonucunu görmeden cevaplar", "cevapta tool bulgusu yok"],
             ["<code>reflect_on_tool_use</code>", "False",
              "tool çıktısı ham döner, yorumlanmaz", "cevap JSON gibi görünür"],
             ["sonlandırma koşulu", "yok",
              "takım tavan olmadan koşar", "fatura"],
             ["<code>model_client_stream</code>", "False",
              "akış olayı hiç yayılmaz", "arayüzde token akmaz"]])
      + quote("Dördünün ortak yanı: <b>hiçbiri hata vermiyor</b>. Sistem çalışıyor, "
              "sonuç yanlış oluyor. Bir çerçevede en pahalı şey, sessizce makul "
              "görünen varsayılandır — çünkü onu aramak için önce yanlış olduğunu "
              "bilmen gerekir.", "r")
      + f"<p>{tag('m')} Dördü de bizde açıkça ayarlanıyor. Varsayılana güvenmek "
        "yerine değeri yazmak, bir sonraki sürümde varsayılan değişirse de korur.</p>")

slide("III · tuzaklar", "Sessiz veri kaybı",
      f"<p>{tag('m')} Yayın yolunda handler bir istisna atarsa: mesaj kaybolur, "
      "çağıran haber almaz, log'a bir satır düşer. Toplayıcı beklemeye devam eder ve "
      "sistem asılı kalır.</p>"
      + cols(
          code("# YANLIŞ — sayaç yalnız başarıda artar\n"
               "async def on_result(self, msg, ctx):\n"
               "    data = parse(msg)      # patlarsa?\n"
               "    self.seen += 1\n"
               "    if self.seen == self.expected:\n"
               "        await self.finish()"),
          code("# DOĞRU — her dal sayılır\n"
               "async def on_result(self, msg, ctx):\n"
               "    try:\n"
               "        data = parse(msg)\n"
               "    except Exception as exc:\n"
               "        data = Failed(str(exc))\n"
               "    finally:\n"
               "        self.seen += 1\n"
               "        if self.seen == self.expected:\n"
               "            await self.finish()"))
      + "<p>Bizim tarama bu yüzden <b>başarısız kaynakları da</b> sonuç nesnesinde "
        "taşıyor: <code>failed_sources</code> boş değilse rapor bunu söylüyor. "
        "\"Üç kaynaktan iki sonuç\" cümlesi, \"iki sonuç\" cümlesinden farklıdır.</p>")

slide("III · tuzaklar", "Dört isim, tek karmaşa",
      table(["isim", "ne", "durum", "import satırı"],
            [["AutoGen v0.2", "eski API", "terk edildi", "<code>from autogen import ...</code>"],
             ["AutoGen v0.4+", "bugünkü mimari", "bizim kullandığımız",
              "<code>from autogen_agentchat... import ...</code>"],
             ["AG2", "v0.2'den çatallanan topluluk sürümü", "ayrı proje", "<code>from autogen import ...</code>"],
             ["MAF / Agent Framework", "Microsoft'un yeni birleşik çerçevesi",
              "AutoGen'in devamı olarak konumlanıyor", "ayrı paket"]])
      + f"<p>{tag('k')} <b>Ayırt etme kuralı import satırında:</b> "
        "<code>from autogen import AssistantAgent</code> gören her örnek eski API "
        "ya da AG2'dir. Bizimki her zaman alt paket adıyla gelir: "
        "<code>autogen_agentchat</code>, <code>autogen_core</code>, "
        "<code>autogen_ext</code>.</p>"
      + quote("İnternetteki örneklerin çoğu hâlâ v0.2. Bu ayrımı bilmeden Stack "
              "Overflow okumak, çalışmayan kodu kendi hatan sanmanın en hızlı "
              "yoludur.", "r"))

slide("III · tuzaklar", "Bakım modu — tez ve kanıtın sınırı",
      cols(
          f"<p>{tag('t')} <b>Tez:</b> AutoGen aktif özellik geliştirmeden bakım "
          "moduna geçiyor; enerji MAF'a kayıyor.</p>"
          "<p><b>Neye dayanıyor:</b> depo etkinliğinin şekli, Microsoft'un MAF "
          "konumlandırması, ve resmî belgelerdeki yönlendirme.</p>"
          "<p><b>Neye dayanmıyor:</b> bir duyuruya. Ortada \"AutoGen dondu\" diyen "
          "resmî bir cümle yok.</p>",
          quote("<b>Neyi kanıtlamaz:</b> AutoGen'in çalışmadığını. v0.7.5 üstünde "
                "kurduğumuz her şey koşuyor ve testleri geçiyor. Bakım modu "
                "<i>yeni özellik beklememek</i> demektir — <i>bugün kırık</i> "
                "demek değil.", "b")
          + "<p><b>Karar açısından anlamı:</b> yeni bir proje bugün başlıyorsa bu "
            "bilgi seçime girer. Çalışan bir proje için taşınma gerekçesi değildir; "
            "taşınmanın maliyeti, beklenmeyen özelliğin değerinden büyüktür.</p>"),
      foot="Etiket bilerek 'teyitsiz' — bu bir okuma, bir ölçüm değil.")

slide("III · tuzaklar", "Desen seçmenin faturası",
      big("%63.7", "aynı görev, aynı ajanlar — yalnız orkestrasyon değişince "
                   "ortaya çıkan token farkı")
      + cols(
          table(["desen", "mesaj", "LLM", "tool", "token"],
                [["<b>SelectorGroupChat</b>", "8", "5", "2", "<b>204</b>"],
                 ["GraphFlow", "11", "7", "3", "270"],
                 ["RoundRobinGroupChat", "9", "6", "2", "274"],
                 ["<b>Swarm</b> (handoff)", "14", "7", "4", "<b>334</b>"]]),
          f"<p>{tag('m')} <code>poc/kiyas.py</code> — aynı görevi beş desenle "
          "koşturuyor, yalnız orkestrasyon değişiyor. Anahtar yoksa replay "
          "modunda çalışıyor, yani sayılar <b>tekrarlanabilir</b>.</p>"
          + quote("Ödenen şey zekâ değil <b>yönlendirme özerkliği</b>. Swarm "
                  "her devirde bağlamı yeniden kuruyor; Selector tek bir seçim "
                  "çağrısıyla idare ediyor.", "g")),
      foot="Karşılaştırmaya çevirisi: Agents SDK'nın tek modeli olan handoff, AutoGen'in en pahalı desenidir.")
