"""Tutorial content, parts A and B: building one agent, then several.

Runs against `make_ogretici.py`'s namespace; nothing here executes alone.
"""

# ═══════════════════════════════════════════════════════════ giriş

chapter("0", "Bu belge nasıl okunur", [
    "Belgenin hangi soruyu cevapladığı ve hangisini cevaplamadığı",
    "Neye ihtiyacın olduğu — kurulum ve ön bilgi",
    "Dört kutu türünün ne anlama geldiği",
],
    p("Bu belge bir <b>referans değil</b>. Referans zaten var: <code>docs/05</code> "
      "ve <code>docs/08</code> AutoGen'in resmî kılavuzlarının tam metnini taşıyor, "
      "toplam 1.18&nbsp;MB. Orada her sınıf, her parametre yazıyor.")
    + p("Referansın cevaplamadığı soru şu: <i>hiçbir şeyden başlayıp çalışan bir "
        "sisteme nasıl gidilir, ve yolda ne beni ısırır?</i> Bu belge onu cevaplıyor. "
        "Bu yüzden sıralama API yüzeyine göre değil, <b>yapabileceğin bir sonraki "
        "şeye</b> göre.")
    + p("Üç kuralı var:")
    + "<ol><li><b>Her bölüm çalışan bir şeyle biter.</b> Kopyalayıp koşturamadığın "
      "bir örnek, tanıma öğretir, yapma öğretmez.</li>"
      "<li><b>Tuzak, ısırdığı yerde durur.</b> <code>max_tool_iterations</code>'ı "
      "ekte değil, tool döngüsünü anlattıktan iki paragraf sonra göreceksin — "
      "mekanizma hâlâ aklındayken.</li>"
      "<li><b>Deponun gösteremediği hiçbir şey iddia edilmiyor.</b> Bir sayı "
      "geçiyorsa onu üreten koşu adıyla yazılı.</li></ol>"

    + h3("Ne gerekiyor")
    + p("Python 3.11 veya üstü, ve bir modele erişim. Modelin OpenAI uyumlu bir "
        "endpoint sunması yeterli — OpenAI'ın kendisi olması gerekmiyor.")
    + shell("python -m venv .venv\n"
            ".venv/bin/pip install -U \\\n"
            "    autogen-agentchat autogen-ext[openai] autogen-core")
    + p("Ön bilgi olarak Python'un <code>async</code>/<code>await</code> tarafını "
        "bilmen gerekiyor. Bilmiyorsan da devam edebilirsin: buradaki her "
        "<code>await</code>, \"bu satır bitene kadar bekle ama bu arada başka işler "
        "koşabilsin\" demektir; ilk on bölüm için bu kadarı yeter.")

    + h3("Dört kutu")
    + neden("Bir tasarım kararının <b>gerekçesi</b>. Çerçeve neden böyle yapmış, "
            "başka nasıl yapılabilirdi. Atlarsan kod yine çalışır, ama bir sonraki "
            "kararı kendin veremezsin.")
    + tuzak("Sessizce yanlış davranan bir şey. Hepsi bizi ısırdı; hepsinin "
            "belirtisi ve düzeltmesi yazılı. <b>Bu kutuları atlama.</b>")
    + dene("Elini klavyeye koyduğun yer. Kısa, ve çoğu bir şeyi <i>kırmanı</i> "
           "isteyecek — çünkü bir mekanizmayı en iyi, bozulduğunda ne olduğunu "
           "görerek öğrenirsin.")
    + olcum("Bu depoda ölçülmüş bir sayı ve nasıl ölçüldüğü. Bunlar tahmin değil; "
            "aynı koşuyu tekrarlayıp doğrulayabilirsin.")

    + h3("Yol haritası")
    + table(["kısım", "ne öğrenirsin", "sonunda elinde ne olur"],
            [["A", "tek bir ajanı kurmak, tool vermek, durdurmak",
              "sorulara tool kullanarak cevap veren bir ajan"],
             ["B", "birden çok ajan, takımlar, aktör modeli",
              "eşzamanlı çalışan çok kaynaklı bir boru hattı"],
             ["C", "gözlem, maliyet, dış dünyaya bağlama",
              "ne yaptığını gösteren ve kapıdan geçen bir sistem"],
             ["D", "kontrol düzlemi — OpenClaw'ın dersleri",
              "onay, denetim, dış içerik sınırı, bellek kökeni"],
             ["E", "kurumsal asistan",
              "neyi alacağının ve neyi almayacağının listesi"]]))

# ═══════════════════════════════════════════════════════ KISIM A

part("A", "Bir ajan kurmak",
     "Tek bir ajanla başlıyoruz. Altı bölümün sonunda tool kullanan, akan, "
     "duran ve ne kadara mal olduğunu söyleyen bir ajanın olacak.")

chapter("1", "Ajan nedir — ve ne değildir", [
    "Ajan döngüsünün üç adımı",
    "\"Ajan\" ile \"model çağrısı\" arasındaki tek fark",
    "Ne zaman ajana ihtiyacın <b>olmadığı</b>",
],
    p("Bir ajan, sihirli bir şey değil. Üç adımlık bir döngü:")
    + "<ol><li>Modele bir bağlam gönder (talimat + konuşma + kullanılabilir "
      "tool'ların tanımı).</li>"
      "<li>Model ya <b>metin</b> döndürür ya da <b>bir tool çağırmak istediğini</b> "
      "söyler.</li>"
      "<li>Tool istediyse: tool'u koştur, sonucu bağlama ekle, birinci adıma dön. "
      "Metin döndürdüyse: bitti.</li></ol>"
    + p("Hepsi bu. Bir ajan çerçevesinin yaptığı şey bu döngüyü senin yerine yazmak, "
        "ve etrafına biriktirdiğin şeyleri — geçmiş, bütçe, durdurma koşulu, "
        "gözlem — tutarlı bir yere koymak.")
    + neden("Bunu bilmek önemli, çünkü çerçevenin ne zaman fazla geldiğini de "
            "gösteriyor. Tool'un yoksa ve tek bir cevap istiyorsan, ajan değil "
            "<b>model çağrısı</b> yapıyorsun. Döngüyü kurmak sana hiçbir şey "
            "kazandırmaz, üstelik bir katman borç bırakır.")

    + h3("Ne zaman gerçekten ajan gerekir")
    + table(["durum", "gerekir mi", "neden"],
            [["Metni özetle", "hayır", "tek çağrı, tool yok, döngü yok"],
             ["Metni özetle, sonra kaydet", "sınırda", "iki adım biliniyor — düz kod yeter"],
             ["Soruyu cevapla, gerekirse ara", "<b>evet</b>", "kaç adım gerektiği önceden bilinmiyor"],
             ["Beş kaynağı tara, birleştir", "<b>evet</b>", "dallanma + toplama + hata yönetimi"]])
    + p("Ayrım şurada: <b>adım sayısı önceden biliniyorsa</b> ajan gereksiz. "
        "Bilinmiyorsa — çünkü modelin bulduğu şey bir sonraki adımı belirliyor — "
        "ajan tam da bu belirsizliği yönetmek için var.")

    + tuzak("En yaygın israf, bilinen bir iş akışını ajanla kurmaktır. Model her "
            "adımda \"şimdi ne yapsam\" diye düşünür, sen zaten biliyorsundur, ve "
            "her tur için para ödersin. Bilinen sıra <code>if</code>/<code>for</code> "
            "ile yazılır.")

    + dene("Bir kâğıda son yaptığın üç otomasyonu yaz ve her biri için tek soruyu "
           "cevapla: <i>adım sayısı önceden belli miydi?</i> Cevabı \"evet\" olan "
           "hiçbiri ajan istemiyordu."))

chapter("2", "İlk ajan", [
    "<code>AssistantAgent</code> ile çalışan bir ajan",
    "Model istemcisi ve <code>model_info</code> tuzağı",
    "<code>run()</code> ne döndürür",
],
    p("En küçük çalışan hâli üç satır: bir model istemcisi, bir ajan, bir koşu.")
    + code("import asyncio\n"
           "from autogen_agentchat.agents import AssistantAgent\n"
           "from autogen_ext.models.openai import OpenAIChatCompletionClient\n\n"
           "async def main():\n"
           "    client = OpenAIChatCompletionClient(model=\"gpt-4o\")\n"
           "    agent = AssistantAgent(\"asistan\", model_client=client)\n\n"
           "    result = await agent.run(task=\"Bir cümlede kendini tanıt.\")\n"
           "    print(result.messages[-1].content)\n\n"
           "asyncio.run(main())", "ilk_ajan.py")
    + p("<code>run()</code> bir <code>TaskResult</code> döndürüyor. İçinde "
        "<code>messages</code> var — turun tamamı, senin görevinden başlayarak. "
        "Son mesaj ajanın cevabı.")

    + h3("Kendi endpoint'in varsa")
    + p("OpenAI dışında bir sağlayıcı kullanıyorsan (yerel bir model, DeepSeek, "
        "bir şirket içi endpoint) iki alan daha veriyorsun:")
    + code("client = OpenAIChatCompletionClient(\n"
           "    model=\"deepseek-chat\",\n"
           "    base_url=\"https://api.deepseek.com/v1\",\n"
           "    api_key=os.environ[\"LLM_API_KEY\"],\n"
           ")")
    + tuzak("<p>Bunu koşturunca büyük ihtimalle şunu göreceksin:</p>"
            "<pre>ValueError: model_info is required when model name is not a "
            "valid OpenAI model</pre>"
            "<p>AutoGen model adını tanıyamazsa <b>başlamayı reddediyor</b>. "
            "Sebebi makul: bu model görüntü alabiliyor mu, function calling "
            "destekliyor mu, JSON çıktı verebiliyor mu — bunları bilmeden ajan "
            "kurarsa yanlış varsayımla koşar.</p>"
            "<p>Çözüm, yetenekleri elle bildirmek:</p>"
            "<pre>client = OpenAIChatCompletionClient(\n"
            "    model=\"deepseek-chat\",\n"
            "    base_url=\"https://api.deepseek.com/v1\",\n"
            "    model_info={\n"
            "        \"vision\": False,\n"
            "        \"function_calling\": True,\n"
            "        \"json_output\": True,\n"
            "        \"family\": \"unknown\",\n"
            "        \"structured_output\": False,\n"
            "    },\n"
            ")</pre>")
    + olcum("Bu depoda <code>pipeline/config.py</code> içindeki "
            "<code>LIVE_MODEL_INFO</code> tam olarak bunu yapıyor. Bu yüzden "
            "endpoint değiştirdiğimizde bu hata artık çıkmıyor: bugün DeepSeek'i "
            "denerken tuzak <b>tetiklenmedi</b>, hata bir adım ileride — "
            "kimlik doğrulamada — çıktı. Kapatılmış bir tuzağın kanıtı, başka bir "
            "hatanın önce gelmesidir.")

    + h3("Sistem talimatı")
    + p("Ajanın kim olduğunu <code>system_message</code> söylüyor. Bu metin her "
        "turda bağlamın en başında duruyor — yani en pahalı ve en etkili metin.")
    + code("agent = AssistantAgent(\n"
           "    \"analist\",\n"
           "    model_client=client,\n"
           "    system_message=(\n"
           "        \"Yatırım sinyallerini değerlendiriyorsun. \"\n"
           "        \"Kaynağı olmayan hiçbir iddiada bulunma.\"\n"
           "    ),\n"
           ")")
    + dene("Aynı görevi iki kez koştur: bir kez <code>system_message</code> "
           "olmadan, bir kez yukarıdaki metinle. Cevabın <i>uzunluğunun</i> ve "
           "<i>tonunun</i> nasıl değiştiğine bak. Sistem talimatı bir tercih değil, "
           "ajanın davranışının çoğu."))

chapter("3", "Tool vermek — ve döngüyü anlamak", [
    "Bir Python fonksiyonunu tool'a çevirmek",
    "Tool döngüsünün gerçekte nasıl aktığı",
    "<code>max_tool_iterations</code> — bu belgedeki en pahalı varsayılan",
],
    p("Tool, modelin çağırabildiği bir fonksiyon. AutoGen'de bir Python "
      "fonksiyonunu vermen yeterli — şemasını imzasından ve docstring'inden "
      "çıkarıyor.")
    + code("async def kur_getir(sehir: str) -> str:\n"
           "    \"\"\"Bir şehrin güncel hava durumunu döndürür.\"\"\"\n"
           "    return f\"{sehir}: 18°C, parçalı bulutlu\"\n\n"
           "agent = AssistantAgent(\n"
           "    \"asistan\",\n"
           "    model_client=client,\n"
           "    tools=[kur_getir],\n"
           ")", "tool_veren_ajan.py")
    + p("Docstring önemli: modelin bu tool'u <i>ne zaman</i> çağıracağına karar "
        "verdiği metin o. Tip ipuçları da öyle — <code>sehir: str</code> yazmazsan "
        "model ne göndereceğini bilemez.")

    + fig(f_tool_loop(), "Tool döngüsü: model ister, workbench koşturur, sonuç "
                         "modele geri döner — ve döngü buna izin verirse.")

    + h3("Döngü gerçekte nasıl akıyor")
    + "<ol><li>Model bağlamı görür ve <code>kur_getir(\"İstanbul\")</code> çağırmak "
      "istediğini söyler.</li>"
      "<li>Çerçeve fonksiyonu koşturur, sonucu alır.</li>"
      "<li><b>Sonucu bağlama ekler ve modele ikinci kez sorar.</b></li>"
      "<li>Model bu sefer metin döndürür: \"İstanbul'da hava 18 derece.\"</li></ol>"
    + p("Üçüncü adım kritik. Model tool'un ne döndürdüğünü ancak <b>ikinci turda</b> "
        "görüyor. Birinci tur, yalnız \"şunu çağırmak istiyorum\" demekten ibaret.")

    + tuzak("<p><b>AgentChat'te <code>max_tool_iterations</code> varsayılan olarak "
            "1'dir.</b> Yani yukarıdaki üçüncü adım <i>gerçekleşmez</i>.</p>"
            "<p>Model tool'u çağırır, tool koşar, sonuç döner — ve tur biter. "
            "Kullanıcıya giden cevap, tool'un bulduğu bilgiyi <b>içermez</b>.</p>"
            "<p>Hiçbir hata çıkmaz. Loga bakarsın: tool çağrılıyor. Sonuç dönüyor. "
            "Yine de cevap yanlış — çünkü eksik olan çağrı değil, <b>ikinci model "
            "turu</b>.</p>"
            "<pre>agent = AssistantAgent(\n"
            "    \"asistan\",\n"
            "    model_client=client,\n"
            "    tools=[kur_getir],\n"
            "    max_tool_iterations=6,   # &lt;-- bunu yaz\n"
            ")</pre>")
    + neden("Varsayılanın 1 olması saçma değil: bir tool çağrısının sonsuz döngüye "
            "girmesi gerçek bir risk ve varsayılan güvenli tarafta duruyor. Sorun "
            "varsayılanın <i>değeri</i> değil, <b>sessizliği</b> — yanlış davranış "
            "hiçbir yerde görünmüyor.")
    + olcum("Bu depoda 6 kullanılıyor. Doğru sayı göreve bağlı: tek tool çağıran "
            "bir ajanda 2 yeter, arayıp okuyup karşılaştıran bir ajanda 6 azdır. "
            "Doğru olmayan tek şey, varsayılanı bilmeden bırakmak.")

    + dene("<p><code>max_tool_iterations</code>'ı yazmadan koştur ve cevaba bak. "
           "Sonra 6 yaz ve tekrar koştur. İki cevabı yan yana koy.</p>"
           "<p>Bu farkı bir kez gördükten sonra bir daha unutmazsın — ve bir "
           "sonraki projede ilk yazacağın parametre bu olur.</p>"))

chapter("4", "Akış — ne olduğunu görmek", [
    "<code>run</code> ile <code>run_stream</code> farkı",
    "Ara olayları okumak",
    "Akışın son elemanının neden özel olduğu",
],
    p("<code>run()</code> her şey bitince tek bir sonuç veriyor. Uzun süren bir "
      "ajanda bu, ekranda hiçbir şey olmadan otuz saniye beklemek demek. "
      "<code>run_stream()</code> ara olayları da veriyor.")
    + code("async for ev in agent.run_stream(task=\"İstanbul'da hava nasıl?\"):\n"
           "    print(type(ev).__name__, \"·\", getattr(ev, \"content\", \"\"))")
    + out("ToolCallRequestEvent · [FunctionCall(name='kur_getir', ...)]\n"
          "ToolCallExecutionEvent · [FunctionExecutionResult(content='İstanbul: 18°C...')]\n"
          "TextMessage · İstanbul'da hava 18 derece, parçalı bulutlu.\n"
          "TaskResult · ")

    + fig(f_message_types(), "Konuşmanın parçası olan mesajlar ve ne olduğunu "
                             "anlatan olaylar — ikisi aynı akıştan geliyor.")

    + tuzak("<p>Akışın <b>son elemanı</b> bir olay değil, <code>TaskResult</code>'ın "
            "kendisi. Döngüde tip kontrolü yapmazsan onu bir olay sanıp işlersin, "
            "sonra da \"sonuç gelmedi\" diye ararsın.</p>"
            "<pre>final = None\n"
            "async for ev in agent.run_stream(task=\"...\"):\n"
            "    if isinstance(ev, TaskResult):\n"
            "        final = ev\n"
            "    else:\n"
            "        handle(ev)</pre>")

    + h3("Token token akış")
    + p("Yukarıdaki akış <i>olay</i> düzeyinde. Cevabın kelime kelime akmasını "
        "istiyorsan bir alan daha gerekiyor:")
    + code("agent = AssistantAgent(\n"
           "    \"asistan\", model_client=client,\n"
           "    model_client_stream=True,   # ModelClientStreamingChunkEvent yayar\n"
           ")")
    + tuzak("Bu da varsayılan olarak <b>kapalı</b>. Açmadan token akışı beklersen "
            "hiç gelmez — ve yine hata çıkmaz, sadece arayüzde bir şey akmaz.")

    + olcum("Bu deponun canlı mekanizma paneli tam olarak bu akıştan besleniyor: "
            "her olay tipi bir mekanizma şemasına eşleniyor ve ekrana o an koşan "
            "şey çiziliyor. Kod: <code>pipeline/conversation.py</code> ve "
            "<code>pipeline/stages.py</code>.")

    + dene("Yukarıdaki döngüyü koştur ve çıkan olay tiplerini <b>sırayla</b> yaz. "
           "Sonra <code>max_tool_iterations=1</code> yapıp tekrar koştur. Hangi "
           "olayın kaybolduğunu göreceksin — ve üçüncü bölümdeki tuzağın ne "
           "olduğunu artık gözünle görmüş olacaksın."))

chapter("5", "Bağlam — model neyi hatırlıyor", [
    "Bağlamın neden büyüdüğü ve neyi içerdiği",
    "Sıkıştırma (compaction) ne yapar",
    "Cache sınırı: maliyetin asıl belirleyicisi",
],
    p("Model hiçbir şey hatırlamıyor. Her turda ona <b>bağlamın tamamı</b> yeniden "
      "gönderiliyor: sistem talimatı, tool tanımları, geçmiş konuşma, ve yeni soru. "
      "\"Hatırlıyor\" dediğimiz şey, bu tekrar gönderme.")
    + p("Bu yüzden bağlam her turda büyüyor ve her tur bir öncekinden pahalı oluyor.")

    + fig(f_context(), "Bağlamın parçaları ve cache sınırı: sabit olanlar önde, "
                       "değişenler arkada.")

    + h3("Ne kadarı gerçekten senin sorun")
    + p("Bir turda modele giden şeyin çoğu <b>kullanıcının yazdığı şey değildir</b>: "
        "sistem talimatı, tool tanımları ve bütün geçmiş her turda yeniden gider. "
        "Bu yüzden \"prompt'u kısalt\" tavsiyesi çoğu zaman yanlış hedeftir — "
        "kısaltılacak şey kullanıcının cümlesi değil.")
    + olcum("Bu dağılımı biz <b>ölçmedik</b>. Ölçtüğümüz şey farklı ve 15. bölümde: "
            "aynı görevi farklı orkestrasyon desenleriyle koşturunca çıkan "
            "<b>%63.7</b>'lik token farkı (<code>poc/kiyas.py</code>). Bağlam "
            "bileşimini kendi işinde ölçmek istersen tarif 14. bölümde.")

    + neden("<p><b>Cache sınırı.</b> Model sağlayıcıları, isteğin başındaki "
            "<i>değişmeyen</i> kısmı önbellekleyebiliyor ve onu çok daha ucuza "
            "faturalandırıyor. Ama önbellek <b>önekten</b> çalışıyor: ilk farklı "
            "bayta kadar geçerli.</p>"
            "<p>Yani değişken bir şeyi (kullanıcının mesajı, bir zaman damgası, "
            "her turda karılan bir tool listesi) başa koyarsan, arkasındaki her şey "
            "önbellekten düşer. Sabit kısım ne kadar büyük olursa olsun, her turda "
            "tam ödersin.</p>"
            "<p>Doğru soru \"prompt nasıl kısalır\" değil, <b>\"ne gerçekten sabit "
            "kalabilir\"</b>.</p>")

    + h3("Sıkıştırma")
    + p("Konuşma uzayınca bir noktada bağlam pencereye sığmaz. Sıkıştırma "
        "(compaction), eski turları bir özete indirip özeti bağlamda tutuyor.")
    + tuzak("Sıkıştırma yazarken bozulmaması gereken bir kural var: <b>bir tool "
            "çağrısı sonucundan ayrılmaz</b>. Ayırırsan model, cevabını hiç "
            "görmediği bir çağrı yapmış gibi görünür — ve çoğu sağlayıcı bu şekli "
            "geçersiz sayıp isteği reddeder. OpenClaw'ın sıkıştırıcısı bu kuralı "
            "açıkça koruyor; kendi sıkıştırıcını yazarsan ilk yazacağın test bu.")

    + dene("Bir ajanla on tur konuş ve her turda token sayısını yazdır "
           "(<code>result.messages</code> içindeki <code>models_usage</code> "
           "alanları). Grafiği çizmene gerek yok — sayının nasıl büyüdüğünü görmek "
           "yeterli."))

chapter("6", "Durmayı öğretmek", [
    "On bir sonlandırma koşulu ve hangisinin ne zaman",
    "Koşulları birleştirmek",
    "Neden en az bir sert tavan gerektiği",
],
    p("Tek bir ajan görevini bitirince durur. Ama bir takım — ve tool döngüsü olan "
      "bir ajan — kendi kendine durmayabilir. Durma koşulunu sen veriyorsun.")
    + code("from autogen_agentchat.conditions import (\n"
           "    MaxMessageTermination, TokenUsageTermination, TextMentionTermination,\n"
           ")\n\n"
           "durdur = MaxMessageTermination(20) | TokenUsageTermination(50_000)")
    + p("<code>|</code> ile birleşenlerden <b>biri</b> yeterli; <code>&amp;</code> "
        "ile <b>hepsi</b> gerekli.")

    + table(["koşul", "ne zaman durur", "not"],
            [["<code>MaxMessageTermination</code>", "mesaj sayısı doldu", "en basit sert tavan"],
             ["<code>TokenUsageTermination</code>", "token bütçesi doldu", "maliyet tavanı"],
             ["<code>TimeoutTermination</code>", "süre doldu", "duvar saati"],
             ["<code>TextMentionTermination</code>", "metinde bir kelime geçti", "modelin işbirliğine bağlı"],
             ["<code>TextMessageTermination</code>", "belirli ajandan metin geldi", "boru hattı sonu"],
             ["<code>SourceMatchTermination</code>", "belirli ajan konuştu", "rol tabanlı"],
             ["<code>HandoffTermination</code>", "devir istendi", "Swarm ile"],
             ["<code>StopMessageTermination</code>", "açık dur mesajı", "protokol tabanlı"],
             ["<code>FunctionCallTermination</code>", "belirli tool çağrıldı", "\"bitir\" tool'u"],
             ["<code>FunctionalTermination</code>", "senin yazdığın koşul", "her şey"],
             ["<code>ExternalTermination</code>", "dışarıdan tetiklendi", "kullanıcı iptali"]])

    + tuzak("<p>Semantik koşullar — \"model TERMINATE yazınca dur\" gibi — "
            "<b>modelin işbirliğine bağlı</b>. Model yazmazsa koşul hiç tetiklenmez.</p>"
            "<p>Bu yüzden üretimde her zaman en az bir <b>sert tavan</b> olmalı: "
            "mesaj sayısı ya da token bütçesi. Tavansız koşan bir takım, faturayı "
            "modelin kararına bırakır.</p>")

    + dene("<code>TextMentionTermination(\"BİTTİ\")</code> ile tek başına bir takım "
           "koştur ve sistem talimatında \"bitince BİTTİ yaz\" deme. Ne kadar "
           "koştuğunu izle — sonra <code>| MaxMessageTermination(10)</code> ekle.")

    + h3("Kısım A'nın sonu — elinde ne var")
    + code("import asyncio, os\n"
           "from autogen_agentchat.agents import AssistantAgent\n"
           "from autogen_agentchat.base import TaskResult\n"
           "from autogen_ext.models.openai import OpenAIChatCompletionClient\n\n"
           "async def kur_getir(sehir: str) -> str:\n"
           "    \"\"\"Bir şehrin güncel hava durumunu döndürür.\"\"\"\n"
           "    return f\"{sehir}: 18°C, parçalı bulutlu\"\n\n"
           "async def main():\n"
           "    client = OpenAIChatCompletionClient(model=\"gpt-4o\")\n"
           "    agent = AssistantAgent(\n"
           "        \"asistan\",\n"
           "        model_client=client,\n"
           "        system_message=\"Kısa ve kaynaklı cevap ver.\",\n"
           "        tools=[kur_getir],\n"
           "        max_tool_iterations=6,\n"
           "        model_client_stream=True,\n"
           "    )\n"
           "    async for ev in agent.run_stream(task=\"İstanbul'da hava nasıl?\"):\n"
           "        if isinstance(ev, TaskResult):\n"
           "            print(\"\\n---\\n\", ev.messages[-1].content)\n\n"
           "asyncio.run(main())", "kisim_a_sonu.py")
    + p("Beş satırlık ajandan buraya geldik. Dört şey ekledik ve <b>dördü de "
        "varsayılan olmayan</b> değerler: sistem talimatı, tool döngüsü tavanı, "
        "akış, ve sonucu doğru yakalayan tip kontrolü."))

# ═══════════════════════════════════════════════════════ KISIM B

part("B", "Birden çok ajan",
     "Bir ajan yetmediğinde ne yapılır — ve çoğu zaman yettiğini fark etmek. "
     "Takımlar, sonra bir katman aşağısı: aktör modeli.")

chapter("7", "İkinci ajan ne zaman gerekir", [
    "Çok-ajanın gerçekten kazandırdığı iki şey",
    "Kazandırmadığı ve pahalıya mal olduğu durumlar",
    "Karar kuralı",
],
    p("Çok-ajan mimarisi moda, ve modanın bedeli var: her ek ajan bir bağlam, bir "
      "model çağrısı ve bir eşgüdüm problemi demek. Ne zaman değer?")
    + h3("Gerçekten kazandırdığı iki şey")
    + "<ol><li><b>Eşzamanlılık.</b> Beş kaynak birbirini beklemeden taranabiliyorsa "
      "beş ajan gerçek zaman kazandırır. Tek ajan sırayla gider.</li>"
      "<li><b>Bağlam ayrımı.</b> Her ajan yalnız kendi işine dair bağlamı taşır. "
      "Beş kaynağın tamamını tek bağlamda taşımak hem pahalı hem dikkat "
      "dağıtıcıdır.</li></ol>"
    + h3("Kazandırmadığı yerler")
    + p("\"Bir yazar, bir eleştirmen\" kurulumu çoğu zaman bir ajanın kendine iki "
        "kez sormasıyla aynı sonucu verir — ve iki ajan iki bağlam demektir. "
        "Rol vermek, yeteneği artırmıyor; <b>bağlamı ayırıyor</b>. Ayırmaya değer "
        "bir bağlam yoksa kazanç da yok.")
    + olcum("Bu depoda ölçülen ek maliyetler, tek turluk temel çizgiye göre: "
            "fan-out/fan-in <b>204</b> token, grup sohbeti <b>270</b>, yansıtma "
            "<b>274</b>, karışık uzmanlar <b>334</b>. Desen seçmek ücretsiz değil.")
    + neden("Karar kuralı basit: <b>eşzamanlılık ya da bağlam ayrımı</b> "
            "kazanıyorsan çok-ajan doğru. Yalnız \"daha akıllı olur\" diye "
            "kuruyorsan, muhtemelen daha pahalı ve daha yavaş olur.")
    + dene("Kısım A'daki ajana ikinci bir tool ekle ve aynı işi <b>tek</b> ajanla "
           "yaptır. Sonra iki ajana böl. Token toplamlarını karşılaştır — "
           "çoğu görevde tek ajan kazanır."))

chapter("8", "Takımlar", [
    "Beş takım tipi ve sırayı kimin belirlediği",
    "Hangisinin ne zaman doğru olduğu",
    "Ortak arayüz: takım değiştirmek kodu değiştirmez",
],
    p("AgentChat beş takım tipi sunuyor. Aralarındaki tek anlamlı fark şu: "
      "<b>sırayı kim belirliyor?</b>")
    + fig(f_teams(), "Beş takım, tek arayüz: run() / run_stream() → TaskResult.")
    + table(["takım", "sırayı kim belirler", "ne zaman", "ek model çağrısı"],
            [["<code>RoundRobinGroupChat</code>", "sabit döngü", "yazar–eleştirmen çiftleri", "yok"],
             ["<code>SelectorGroupChat</code>", "model, her turda", "rolleri belirsiz tartışma", "her turda bir"],
             ["<code>Swarm</code>", "ajanın kendisi (handoff)", "devretme akışları", "yok"],
             ["<code>MagenticOne</code>", "planlayıcı ajan", "açık uçlu görevler", "en çok"],
             ["<code>GraphFlow</code>", "önceden çizilmiş DAG", "bilinen boru hattı", "yok"]])
    + code("from autogen_agentchat.teams import RoundRobinGroupChat\n"
           "from autogen_agentchat.conditions import MaxMessageTermination\n\n"
           "takim = RoundRobinGroupChat(\n"
           "    [yazar, elestirmen],\n"
           "    termination_condition=MaxMessageTermination(8),\n"
           ")\n"
           "sonuc = await takim.run(task=\"Bu paragrafı düzelt: ...\")")
    + neden("Beşi de aynı arayüzü sunuyor: <code>run()</code>, "
            "<code>run_stream()</code>, <code>TaskResult</code>. Yani takım "
            "değiştirmek çağıran kodu değiştirmiyor. Bu, deneme yapmayı ucuzlatan "
            "bir tasarım — üç takımı aynı görevde koşturup ölçebilirsin.")
    + tuzak("<code>SelectorGroupChat</code> her turda \"sırada kim var\" diye "
            "modele soruyor. Yani her turda <b>iki</b> model çağrısı oluyor: biri "
            "seçim, biri iş. Sıra belliyse bu ödenmemesi gereken bir maliyet.")
    + dene("Aynı görevi <code>RoundRobinGroupChat</code> ve "
           "<code>SelectorGroupChat</code> ile koştur, token toplamlarını "
           "karşılaştır. Farkı gördükten sonra hangisini varsayılan yapacağını "
           "bileceksin."))

chapter("9", "GraphFlow — boru hattını çizmek", [
    "<code>DiGraphBuilder</code> ile grafik kurmak",
    "Eşzamanlı dallar ve <code>join</code>",
    "Neden bu depodaki tarama bunu kullanıyor",
],
    p("Boru hattın belliyse — \"önce topla, sonra üç kaynağı paralel tara, sonra "
      "birleştir\" — sırayı modele sordurmanın anlamı yok. Grafiği çiz, koştur.")
    + fig(f_graphflow(), "Bir giriş, üç eşzamanlı dal, hepsini bekleyen bir join, "
                         "sonra tek çıkış.")
    + code("from autogen_agentchat.teams import DiGraphBuilder, GraphFlow\n\n"
           "b = DiGraphBuilder()\n"
           "b.add_node(giris)\n"
           "for analist in analistler:\n"
           "    b.add_node(analist)\n"
           "    b.add_edge(giris, analist)      # dallan\n"
           "    b.add_edge(analist, birlestir)  # topla\n"
           "b.add_node(birlestir)\n\n"
           "akis = GraphFlow(b.build(), participants=[giris, *analistler, birlestir])")
    + p("Aynı düğüme birden çok kenar giriyorsa o düğüm bir <b>join</b>. "
        "Varsayılan politika <code>all</code>: bütün dallar bitene kadar bekler. "
        "<code>any</code> ilk bitene devam eder.")
    + neden("GraphFlow'un kazandırdığı üç şey: eşzamanlılık bedava geliyor, ek "
            "model çağrısı yok (sıra zaten belli), ve grafiğin kendisi bir "
            "<b>belge</b> — sistemin ne yaptığı koda bakınca görünüyor.")
    + olcum("<p>Bu depoda tarama tam olarak böyle kurulu: <code>pipeline/graph.py</code>, "
            "beş katılımcı, eşzamanlı dal + <code>join(all)</code> + sıralı sayım.</p>"
            "<p>Uzun süre bunun core pub/sub olduğunu sandık ve arayüze yanlış şema "
            "çizdik. Kod okununca çıktı. Şimdi bir test tutuyor: "
            "<code>test_the_scan_is_graphflow_not_core_pubsub</code>.</p>")
    + tuzak("Grafikte döngü kurarsan <code>build()</code> sırasında değil, "
            "koşarken bir sorunla karşılaşırsın. DAG'ın \"A\"sı <i>acyclic</i> — "
            "döngü istiyorsan takım tipi değiştirmen gerekir.")
    + dene("Üç dallı bir GraphFlow kur ve dallardan birini kasten patlat "
           "(<code>raise</code>). <code>join(all)</code> ne yapıyor? Sonra "
           "<code>any</code> yap ve tekrar dene."))

chapter("10", "Bir katman aşağı: aktör modeli", [
    "core'un AgentChat'ten farkı",
    "Runtime'ın ne yaptığı",
    "Ne zaman aşağı inmen gerektiği",
],
    p("Şimdiye kadar her şey <code>autogen_agentchat</code> katmanındaydı. Altında "
      "<code>autogen_core</code> var ve orada soyutlamalar farklı: ajan, takım, "
      "görev değil — <b>aktör, mesaj, topic, abonelik</b>.")
    + fig(f_actor(), "Runtime mesajı taşır; ajanlar birbirinin referansını tutmaz.")
    + p("Aktör modelinde ajanlar birbirini çağırmıyor. Bir ajan başka bir ajanın "
        "nesnesini elinde tutmuyor, metodunu çağırmıyor. Runtime'a bir mesaj "
        "veriyor; teslimatı runtime yapıyor.")
    + neden("<p>Bedeli: araya bir dolaylılık katmanı giriyor, ve \"kim kimi çağırdı\" "
            "sorusu artık yığın izinden okunmuyor.</p>"
            "<p>Karşılığı üç şey: ajan eklemek çağıran kodu değiştirmiyor; bütün "
            "mesajlar tek noktadan geçtiği için müdahale ve ölçüm oraya takılıyor; "
            "ve aynı sınıftan çok örnek bedava geliyor.</p>")
    + h3("Ne zaman aşağı inersin")
    + table(["durum", "katman"],
            [["Bir görevi ajanlara yaptırıyorsun", "AgentChat"],
             ["Beş takım tipinden biri işini görüyor", "AgentChat"],
             ["Kendi eşgüdüm desenini kuruyorsun", "core"],
             ["Mesaj akışına müdahale etmen gerekiyor", "core"],
             ["Ajanları ayrı süreçlere dağıtacaksın", "core"]])
    + tuzak("AgentChat'in çözdüğü bir problemi core'da yeniden çözmek, aynı kodu "
            "daha az testle yazmaktır. Önce yukarıdan dene; aşağı inmek her zaman "
            "mümkün."))

chapter("11", "Kimlik: type + key", [
    "<code>AgentId</code>'nin iki parçası",
    "Örneklerin ne zaman doğduğu",
    "Durumun nerede yaşadığı",
],
    p("core'da bir ajanın kimliği iki parçalı: <code>AgentId = (type, key)</code>.")
    + fig(f_identity(), "Bir tip, üç örnek: aynı davranış, üç ayrı durum.")
    + two(
        p("<code>type</code> <b>davranıştır</b>: hangi sınıf, hangi handler'lar. "
          "Kayıt bu düzeyde yapılır — <code>register</code> bir tip kaydeder.")
        + p("<code>key</code> <b>örnektir</b>: aynı davranışın hangi kopyası, hangi "
            "durumu taşıyor. Kaydedilmez."),
        p("Örnekler <b>talep üzerine</b> doğar. <code>analyst/hn</code>'e ilk mesaj "
          "gittiğinde runtime o örneği yaratır, fabrika fonksiyonunu çağırır, sonra "
          "teslim eder.")
        + p("Yani \"üç ajan kaydettim\" yanlış bir cümle: <b>bir tip</b> kaydettin, "
            "üç örnek doğdu."))
    + code("class Analist(RoutedAgent):\n"
           "    def __init__(self, kaynak: str) -> None:\n"
           "        super().__init__(f\"{kaynak} analisti\")\n"
           "        self.kaynak = kaynak\n"
           "        self.gorulen = 0        # bu örneğe ait durum\n\n"
           "await Analist.register(\n"
           "    runtime, \"analyst\",\n"
           "    lambda: Analist(kaynak=\"arxiv\"),\n"
           ")")
    + tuzak("Durumu <code>key</code> taşır. Aynı <code>key</code>'e iki kez mesaj "
            "gönderirsen aynı örneğe, aynı belleğe gider. Farklı <code>key</code> "
            "demek <b>sıfırdan bir ajan</b> demektir — ve bir sonraki bölümde "
            "göreceğin gibi, <code>key</code>'i sandığından daha kolay "
            "değiştiriyorsun."))

chapter("12", "Topic ve abonelik", [
    "<code>TopicId</code>'nin iki parçası",
    "<code>TypeSubscription</code> nasıl eşleşir",
    "En sık görülen core hatası: source → key kuralı",
],
    p("Yayın yapmak için bir adres gerekiyor: <code>TopicId</code>. O da iki "
      "parçalı — ve bu tesadüf değil.")
    + fig(f_topic(), "Topic'in iki parçası ve teslim edilen örneği belirleyen kural.")
    + p("<code>type</code> ne olduğunu söyler (\"bu bir görev\"), "
        "<code>source</code> hangi iş için olduğunu (\"7 numaralı koşu\").")
    + code("await runtime.publish_message(\n"
           "    Gorev(sorgu=\"ai altyapı\"),\n"
           "    TopicId(type=\"task\", source=\"job-7\"),\n"
           ")")
    + p("Abonelik ise <code>type</code>'a yapılır:")
    + code("await runtime.add_subscription(\n"
           "    TypeSubscription(topic_type=\"task\", agent_type=\"analyst\")\n"
           ")")
    + tuzak("<p><b>Teslim edilen örneğin <code>key</code>'i, topic'in "
            "<code>source</code>'undan gelir.</b> Doğrudan, dönüşümsüz. "
            "(core kılavuzu, 05:670)</p>"
            "<p>Yani <code>source</code>'u her istekte değiştirirsen her istekte "
            "<b>yeni bir ajan örneği</b> doğar. Önceki örneğin biriktirdiği durum "
            "kaybolmaz — <i>erişilemez</i> hale gelir, ki bu daha kötüdür çünkü "
            "hafızada durur ve hiçbir hata çıkmaz.</p>"
            "<p>Sistem çalışır, ajanlar cevap verir, sadece hiçbiri bir öncekini "
            "hatırlamaz.</p>")
    + neden("Doğru kullanım: <code>source</code> = <b>iş kimliği</b>. Aynı işin "
            "bütün adımları aynı <code>source</code>'u paylaşır. Farklı iş, farklı "
            "<code>source</code>, ve o zaman ayrı örnek <i>istediğin</i> şeydir.")
    + dene("İki mesajı aynı <code>source</code> ile, sonra iki mesajı farklı "
           "<code>source</code> ile yayınla. Ajanın içine bir sayaç koy ve "
           "yazdır. Sayacın ne zaman sıfırlandığını göreceksin."))

chapter("13", "Fan-out / fan-in — kendi ellerinle", [
    "Tek yayının birden çok ajanı nasıl tetiklediği",
    "<code>ClosureAgent</code> ile toplama",
    "İki asimetri: dönüş değeri ve hata",
],
    p("Şimdi hepsini birleştiriyoruz. Amaç: tek bir yayınla üç analisti "
      "tetiklemek, sonuçlarını toplamak.")
    + fig(f_fanout(), "Tek publish üç ajanı birden uyandırır; toplama ayrı bir "
                      "ajandır çünkü publish hiçbir şey döndürmez.")

    + h3("Önce iki asimetriyi bil")
    + fig(f_send_vs_publish(), "İki iletişim biçimi ve aralarındaki iki fark.")
    + table(["", "<code>send_message</code>", "<code>publish_message</code>"],
            [["alıcı", "tek, bilinen", "abone olan herkes"],
             ["dönüş", "cevabı döndürür", "<b>hiçbir şey</b>"],
             ["hata", "çağırana fırlar", "<b>yalnız loglanır</b>"],
             ["0 alıcı", "hata", "geçerli sonuç"]])
    + tuzak("<p>İkinci ve üçüncü satır birlikte bir tuzak kuruyor: fan-out'ta bir "
            "dal patlarsa <b>kimse haberdar olmaz</b> ve toplayıcı sonsuza kadar "
            "bekler.</p>"
            "<p>Bu yüzden sayaç <b>hata durumunda da</b> ilerlemek zorunda:</p>"
            "<pre>async def on_result(self, msg, ctx):\n"
            "    try:\n"
            "        veri = ayristir(msg)\n"
            "    except Exception as exc:\n"
            "        veri = Basarisiz(str(exc))\n"
            "    finally:\n"
            "        self.gorulen += 1\n"
            "        if self.gorulen == self.beklenen:\n"
            "            await self.bitir()</pre>")

    + h3("Çalışan örnek")
    + code("import asyncio\n"
           "from dataclasses import dataclass\n"
           "from autogen_core import (\n"
           "    ClosureAgent, ClosureContext, MessageContext, RoutedAgent,\n"
           "    SingleThreadedAgentRuntime, TopicId, TypeSubscription, message_handler,\n"
           ")\n\n"
           "@dataclass\n"
           "class Gorev:\n"
           "    sorgu: str\n\n"
           "@dataclass\n"
           "class Sonuc:\n"
           "    kaynak: str\n"
           "    bulgu: str\n\n"
           "class Analist(RoutedAgent):\n"
           "    def __init__(self, kaynak: str) -> None:\n"
           "        super().__init__(f\"{kaynak} analisti\")\n"
           "        self.kaynak = kaynak\n\n"
           "    @message_handler\n"
           "    async def calis(self, msg: Gorev, ctx: MessageContext) -> None:\n"
           "        bulgu = f\"{self.kaynak}: '{msg.sorgu}' için 3 kayıt\"\n"
           "        await self.publish_message(\n"
           "            Sonuc(kaynak=self.kaynak, bulgu=bulgu),\n"
           "            TopicId(type=\"result\", source=ctx.topic_id.source),\n"
           "        )\n\n"
           "async def main() -> None:\n"
           "    runtime = SingleThreadedAgentRuntime()\n"
           "    kuyruk: asyncio.Queue[Sonuc] = asyncio.Queue()\n\n"
           "    for kaynak in (\"arxiv\", \"hn\", \"github\"):\n"
           "        await Analist.register(\n"
           "            runtime, f\"analyst-{kaynak}\",\n"
           "            lambda k=kaynak: Analist(k),\n"
           "        )\n"
           "        await runtime.add_subscription(TypeSubscription(\n"
           "            topic_type=\"task\", agent_type=f\"analyst-{kaynak}\",\n"
           "        ))\n\n"
           "    async def topla(ctx: ClosureContext, msg: Sonuc, mc: MessageContext) -> None:\n"
           "        await kuyruk.put(msg)\n\n"
           "    await ClosureAgent.register_closure(\n"
           "        runtime, \"toplayici\", topla,\n"
           "        subscriptions=lambda: [TypeSubscription(\n"
           "            topic_type=\"result\", agent_type=\"toplayici\",\n"
           "        )],\n"
           "    )\n\n"
           "    runtime.start()\n"
           "    await runtime.publish_message(\n"
           "        Gorev(sorgu=\"ai altyapı\"),\n"
           "        TopicId(type=\"task\", source=\"job-7\"),\n"
           "    )\n"
           "    await runtime.stop_when_idle()\n\n"
           "    while not kuyruk.empty():\n"
           "        print((await kuyruk.get()).bulgu)\n\n"
           "asyncio.run(main())", "fanin.py — tek yayın, üç dal, bir toplayıcı")
    + p("Dikkat edilecek üç yer: üç analist <b>ayrı tip</b> olarak kaydedildi "
        "(aynı tip olsalardı <code>source</code> aynı olduğu için tek örnek "
        "paylaşırlardı); sonuçlar ikinci bir topic'e yayınlanıyor; ve toplayıcı bir "
        "<code>ClosureAgent</code> — davranışı tek fonksiyon kadar olduğu için sınıf "
        "yazmaya değmedi.")
    + tuzak("<code>stop_when_idle()</code> kuyruk boşalınca döner. "
            "<code>stop()</code> ise <b>hemen</b> durur ve işlenmemiş mesajlar "
            "kaybolur. Testlerde <code>stop()</code> kullanmak, yarısı işlenmiş bir "
            "sistemi doğru sanmanın en hızlı yoludur.")
    + olcum("Bu depoda çalışan karşılığı <code>pipeline/fanin.py</code>. Ama dikkat: "
            "<b>tarama bunu kullanmıyor</b>; tarama GraphFlow'la kurulu "
            "(<code>pipeline/graph.py</code>). İkisi de duruyor çünkü ikisi farklı "
            "şeyler öğretiyor.")
    + dene("Yukarıdaki dosyayı koştur. Sonra bir analistin içine "
           "<code>raise RuntimeError(\"patla\")</code> koy ve tekrar koştur. Kaç "
           "sonuç geldi? Hata nerede göründü? Bu, bölümün başındaki asimetriyi "
           "gözünle görmen."))
