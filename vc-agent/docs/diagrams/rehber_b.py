"""Rehber, ikinci yarı: OpenClaw · karşılaştırma · seçim.

`make_rehber.py` tarafından exec ediliyor.
"""

from make_ogretici import (  # noqa: F401
    chapter, code, dene, fig, h3, neden, olcum, out, p, part, shell, table,
    tuzak, two,
)
from figures import (  # noqa: F401
    f_failover, f_lobster, f_loopguard, f_memory_write, f_packages,
    f_profiles, f_repair, f_result_middleware, f_self_learning,
    f_session_tools, f_tool_catalog, f_tool_search, f_trajectory,
    f_atlas, f_ctx_engine, f_durable, f_external_content, f_frozen_plan,
    f_gate, f_gateway, f_memory_tiers, f_oc_arch, f_scopes, f_secrets,
    f_task_stack, f_three_axes, f_two_ledgers,
)

# ═══════════════════════════════════════════════════════ KISIM 4 — OpenClaw

part("4", "OpenClaw: harness katmanı",
     "Kütüphane değil, uçtan uca bir araç. Bir ajanı gerçek bir makinede "
     "koştururken ortaya çıkan soruların cevapları burada.")

chapter(
    "18", "Harness nedir, kütüphaneden farkı ne",
    ["Harness'ın kapsadığı işler",
     "Neden bu işler kütüphanede olmuyor",
     "OpenClaw'ın mimarisi kuşbakışı"],
    p("Bir kütüphane “ajanları nasıl bağlarım” sorusunu cevaplıyor. Bir "
      "<b>harness</b> ise “bu ajanı gerçek bir makinede, gerçek dosyalarla, "
      "gerçek sırlarla, gerçek bir insanın yanında nasıl koştururum” sorusunu.")
    + fig(f_oc_arch(), "OpenClaw kuşbakışı: gateway, oturumlar, tool'lar, "
                       "zamanlayıcı ve denetim.")
    + table(
        ["Harness'ın işi", "AutoGen'de", "MAF'ta"],
        [["Tool onayı", "yok — elle kurulur", "<code>approval_mode</code> · "
          "<code>ToolApprovalMiddleware</code>"],
         ["Bağlam sıkıştırma", "yok", "Harness Agent içinde"],
         ["Dosya erişimi ve bellek", "<code>Memory</code> protokolü (kurulum sende)",
          "context providers · Harness Agent"],
         ["Zamanlama", "yok", "yok"],
         ["Sır yönetimi", "yok", "yok"],
         ["Denetim kaydı", "olay akışı var, kayıt yolu sende",
          "telemetri var, uyum kaydı sende"]],
        "Harness'ın kapsadığı işlerin çoğu iki kütüphanede de yok.")
    + neden(
        p("Bu tablo, MAF'ın <b>Harness Agent</b>'ı neden eklediğini de "
          "açıklıyor: kütüphane katmanı, harness işlerinin bir kısmını içeri "
          "almaya başladı. Ama zamanlama ve sır yönetimi hâlâ dışarıda — ve "
          "bir kurumda bu ikisi çoğu zaman en sert gereksinim.")),
)

chapter(
    "19", "Üç kontrol ekseni",
    ["Yetkinin üç ayrı eksende ayrılması",
     "Metot kapsamının neden yalnızca ilk kapı olduğu",
     "Donmuş plan: onay komuta değil plana bağlanır"],
    p("OpenClaw'ın en taşınabilir fikri bu: yetki tek bir “izin var/yok” "
      "anahtarı değil, <b>üç ayrı eksen</b>.")
    + fig(f_three_axes(), "Üç eksen: kim, ne, nerede — ayrı ayrı daraltılıyor.")
    + fig(f_scopes(), "Metot kapsamı ilk kapı; arkasında ikinci bir katman var.")
    + tuzak(
        p("Metot kapsamı <b>yalnızca ilk kapı</b>. Bir metoda izin vermek, o "
          "metodun ulaşabileceği her şeye izin vermek demek değil — ama "
          "kapsamı tek savunma sanmak tam olarak bu hataya yol açıyor."))
    + fig(f_frozen_plan(), "Donmuş plan: onaylanan şey komut değil, çalışacak "
                           "planın kendisi.")
    + olcum(
        p("Bu fikir bu projede ölçülerek doğrulandı. Onayı <b>çözülmüş</b> "
          "zamanlama üstünde imzaladığımızda onay <b>hiç tüketilemedi</b>: "
          "<code>\"5dk sonra\"</code> her ayrıştırmada farklı bir zaman "
          "damgasına çözülüyor, yani her seferinde yeni bir istek doğuyor. "
          "İmzayı kullanıcının yazdığı satıra taşıyınca düzeldi. <b>[ölçüldü]</b>")
        + p("Aynısı kod yürütmede de yaşandı: model aynı soruya her seferinde "
            "<b>farklı bir program</b> yazıyor. Onayı yeniden üretilen koda "
            "bağlarsan hiç tüketilemiyor; onaylanan <b>metnin</b> saklanması "
            "gerekiyor. <b>[ölçüldü]</b>"))
    + fig(f_gate(), "Kapı: her tool çağrısının geçtiği tek nokta."),
)

chapter(
    "20", "Bellek, bağlam ve dış içerik",
    ["Beş bellek katmanı ve hangisinin ne zaman okunduğu",
     "Bağlam motoru ve sıkıştırmanın bedeli",
     "Dış içeriğin veri olarak işaretlenmesi"],
    fig(f_memory_tiers(), "Beş bellek katmanı — hepsi aynı anda okunmuyor.")
    + p("Belleğin katmanlara ayrılmasının sebebi maliyet: her turda her şeyi "
        "göndermek, en kısa soruyu bile tam tarifeden ödetiyor.")
    + fig(f_ctx_engine(), "Bağlam motoru: bütçeye göre seçim, aşınca özet.")
    + tuzak(
        p("Sıkıştırma bir <b>kayıptır</b>. Sıkıştırma sonrası ajan, konuşmanın "
          "erken kısmındaki bir ayrıntıyı artık bilmiyor olabilir — ve "
          "bilmediğini de bilmiyor. Bir denetim sorusunun cevabı o ayrıntıdaysa, "
          "cevap sessizce yanlış oluyor."))
    + fig(f_external_content(), "Dış içerik veri olarak işaretleniyor, talimat "
                                "olarak değil.")
    + neden(
        p("Bir web sayfasından ya da e-postadan gelen metin, modelin gözünde "
          "kullanıcının yazdığı metinle aynı görünüyor. Sarmalayıcı bu ikisini "
          "ayırıyor. Bu, MagenticOne kılavuzunun da açıkça uyardığı yer: "
          "<i>“prompt injection attacks from webpages”</i>. <b>[kaynak]</b>")),
)

chapter(
    "21", "Zamanlama, dayanıklılık ve denetim",
    ["Zamanlayıcının gerçekte nerede koştuğu",
     "Dayanıklı durum ile dayanıklı yürütmenin farkı",
     "Neden iki ayrı kayıt hattı gerekiyor"],
    fig(f_task_stack(), "Zamanlama yığını: kayıt, tetikleyici, ve koşan süreç.")
    + tuzak(
        p("Zamanlama yalnız gateway <b>ayaktayken</b> çalışıyor. Bu makinede "
          "servis systemd kullanıcı servisi ve <code>Linger=no</code> — oturum "
          "kapanınca zamanlama da duruyor. “Zamanlama var” demekle “her zaman "
          "çalışır” demek arasındaki fark burada. <b>[ölçüldü]</b>"))
    + fig(f_durable(), "Dayanıklı durum var; dayanıklı yürütme yok.")
    + neden(
        p("“Dayanıklı” kelimesi iki farklı şeyi anlatıyor. OpenClaw'da dayanıklı "
          "<b>durum</b> var — her şey SQLite'ta. Ama dayanıklı <b>yürütme</b> "
          "yok: kurtarma bir replay değil, modele yazılan sentetik bir cümle. "
          "Yan etkili bir tool çağrıldıktan sonra çökülürse, ikinci kez "
          "çağrılmasını mekanik olarak engelleyen hiçbir şey yok. "
          "<b>[kaynak]</b> docs/18"))
    + fig(f_two_ledgers(), "İki kayıt hattı: operasyonel ve uyum.")
    + p("Operasyonel kayıt <b>kayıp toleranslı</b> olmalı — kayıt uğruna işi "
        "durdurmak saçma. Uyum kaydı ise <b>kayıpsız, senkron ve fail-closed</b> "
        "olmalı: yazılamıyorsa koşu düşmeli. Aynı hattan ikisini birden "
        "beklemek, ikisinden de olmak demek.")
    + fig(f_secrets(), "Sır yalnız son sınırda gerçek değerine dönüşüyor."),
)


# ─────────────────────────────────────────────── OpenClaw: niş yüzeyler

chapter(
    "22", "İskelet ve tool kataloğu",
    ["Kodun nasıl bölündüğünün ne anlattığı",
     "51 tool'un dağılımı ve tasarım niyeti",
     "“Kaç tool var” sorusunun neden üç cevabı olduğu"],
    p("Bir sistemin nasıl düşünüldüğünü öğrenmenin en hızlı yolu, kodun nasıl "
      "bölündüğüne bakmaktır. OpenClaw <b>22 pakete</b> bölünmüş.")
    + fig(f_packages(), "22 paket — çekirdeğin her ilginç parçası ayrı.")
    + p("En büyük üçü: <b>ai</b> (118 dosya) model sağlayıcılarıyla konuşuyor, "
        "<b>gateway-protocol</b> (108 dosya) kontrol düzleminin tipli şemasını "
        "tutuyor, <b>memory-host-sdk</b> (83 dosya) bellek sağlayıcılarının "
        "uyması gereken sözleşmeyi tanımlıyor. Üçünün ayrı olması bir tercih: "
        "model erişimi, kontrol düzlemi ve bellek <b>birbirinden bağımsız "
        "değişebiliyor</b>.")
    + neden(
        p("İlginç olan en küçükler. <code>tool-call-repair</code> altı dosya, "
          "<code>retry</code> tek dosya. Ama ikisi de olmadığında sistem "
          "çökmüyor — <b>sessizce yanlış çalışıyor</b>, ki bu daha kötü."))
    + fig(f_tool_catalog(), "51 tool, on bir grup.")
    + p("Asıl bilgi sayıda değil <b>dağılımda</b>. En kalabalık grup 15 tool'la "
        "<b>sessions</b>: alt-ajan başlatmak, iş devretmek, cevap beklemek, "
        "aralarında mesajlaşmak. Dosya işlemleri için 4, komut çalıştırmak için "
        "3 tool var. Yatırımın büyük kısmı dosyaya ya da kabuğa değil, "
        "<b>ajanların birbirini yönetmesine</b> yapılmış — tasarımın niyeti bu: "
        "tek asistan değil, çok ajanlı bir işletim ortamı.")
    + fig(f_profiles(), "Üç sayı, üç anlam: 51 kaynakta, 44 canlıda, doküman eskimiş.")
    + olcum(
        p("“Kaç tool var?” sorusuna üç farklı cevap geliyor ve üçü de doğru, "
          "çünkü üçü farklı şeyi sayıyor. <b>51</b> kaynak kodda tanımlı; "
          "<b>44</b> canlı gateway'in gerçekten sunduğu <b>[ölçüldü]</b>; "
          "dokümandaki tablo ise <b>eskimiş</b>. Daralma üç aşamada: profil bir "
          "taban liste veriyor, <code>allow</code>/<code>deny</code> kesiyor, "
          "sandbox politikası bir kez daha kesiyor. Üçü de yalnız "
          "<b>daraltıyor</b>; hiçbiri listeye tool <i>ekleyemiyor</i>.")
        + p("Sonuç: bir kurulumun gerçek tool yüzeyini öğrenmenin tek yolu "
            "<b>çalışan sisteme sormaktır</b>.")),
)

chapter(
    "23", "Tool katmanının iki inceliği",
    ["Bozuk tool çağrısını kurtarmak",
     "Sonuca dokunan ara katman ve neden komuta dokunmadığı"],
    fig(f_repair(), "Tool Call Repair: düz yazıyı gerçek çağrıya çevirmek.")
    + p("Modelin bir tool çağırmasının doğru yolu, sağlayıcının function calling "
        "biçimini kullanmaktır. Ama bazı modeller bunu beceremiyor: çağrıyı "
        "cevabın içine <b>düz yazı olarak</b> yazıyorlar. OpenClaw dört farklı "
        "biçimi tanıyıp gerçek çağrıya çeviriyor — "
        "<code>[END_TOOL_REQUEST]</code> bloğu, Harmony işaretçileri, ve "
        "XML'e benzeyen <code>&lt;function=ad&gt;</code> etiketleri.")
    + tuzak(
        p("Kritik adım ikincisi: modelin yazdığı ad, sağlayıcının <i>o istekte</i> "
          "izin verdiği tam tool adına eşleniyor. Yakın ama birebir olmayan bir "
          "ad işe yaramıyor. Onarmazsan tur boşa gidiyor ve <b>hiçbir yerde "
          "sebebi görünmüyor</b>."))
    + fig(f_result_middleware(), "Tokenjuice: komuta değil, sonuca dokunuyor.")
    + p("Bir <code>exec</code> çağrısı 40.000 satır log döndürebilir; o log "
        "olduğu gibi bağlama girerse tek komut bütün pencereyi yiyor. Tokenjuice "
        "bunu komut <b>zaten koştuktan sonra</b> çözüyor: çıktıyı kısaltıyor.")
    + neden(
        p("Hangi katmanda durduğu tasarımın kendisi. Komutu yeniden yazmıyor, "
          "tekrar koşturmuyor, çıkış kodunu değiştirmiyor — yalnızca modele "
          "dönen <code>tool_result</code>'a dokunuyor. Komutu değiştirseydin "
          "sonucun <i>doğruluğuna</i> karışmış olurdun; sonucu kısaltmak ise "
          "yalnızca <b>bağlam muhasebesi</b>.")),
)

chapter(
    "24", "Lobster ve Code Mode: orkestrasyonu modelden almak",
    ["Çok adımlı işi tipli bir runtime'a vermek",
     "Kapıyı runtime'ın tutmasının önemi",
     "Büyük katalog, küçük prompt"],
    fig(f_lobster(), "Lobster: tek tool çağrısı, gömülü onay kapıları, devam token'ı.")
    + p("Çok adımlı bir işi modele orkestra ettirirsen her adım ayrı bir tur "
        "olur; dört adımlı iş dört model turu demektir ve her turda bütün bağlam "
        "yeniden gönderilir. Lobster bu orkestrasyonu modelden alıp <b>tipli bir "
        "runtime'a</b> veriyor: model tek çağrı yapıyor, runtime bütün boru "
        "hattını koşturuyor.")
    + neden(
        p("İçinde onay kapıları da var. Bir adım yan etkiliyse akış <b>orada "
          "duruyor</b> ve bir devam token'ı döndürüyor; onayladıktan sonra "
          "baştan başlamıyor, kaldığı yerden devam ediyor. Önemli olan şu: "
          "<b>kapıyı runtime tutuyor, model değil.</b> Model “onay sormayı "
          "atlayayım” diye karar veremiyor."))
    + fig(f_tool_search(), "Code Mode: model şemaları görmüyor, köprüye kod yazıyor.")
    + p("Katalog büyüdükçe bütün tool şemalarını prompt'a koymak imkânsızlaşıyor. "
        "<b>Code Mode</b> bunu şöyle çözüyor: model şemaları görmüyor, küçük bir "
        "JavaScript köprüsüne <code>search</code>, <code>describe</code>, "
        "<code>call</code> yazıyor. <b>Swarm</b> aynı köprüden eşzamanlı "
        "alt-ajanlar başlatıyor.")
    + neden(
        p("İkisinin güvenlik hikâyesi aynı ve dikkat çekici. Köprü izole bir Node "
          "alt sürecinde koşuyor: <b>environment boş, dosya sistemi yok, ağ yok, "
          "eklenti kodu yok, sır yok</b>. Köprüde koşan kod tek başına hiçbir şey "
          "yapamıyor; gerçek her çağrı Gateway'e geri dönüyor ve normal politika, "
          "onay, hook ve log yolundan geçiyor.")),
)

chapter(
    "25", "Koşan bir tura müdahale, ve öğrenmenin kalıcı birimi",
    ["Tur başladıktan sonra yapılabilecek dört şey",
     "Düzeltmenin belleğe değil skill'e yazılması",
     "Öğrenilen şeyin doğrudan davranışa yazılmaması"],
    fig(f_session_tools(), "Dört müdahale yolu: /steer · /btw · /goal · /loop")
    + p("Sıradan bir sohbet arayüzünde mesajı gönderdikten sonra yapabileceğin "
        "tek şey beklemektir. Bu dört komut aynı soruya farklı cevaplar veriyor: "
        "<b>tur çoktan başlamışken ne yapabilirsin?</b>")
    + table(
        ["Komut", "Ne yapıyor", "İnceliği"],
        [["<code>/steer</code>", "koşan turu yönlendiriyor",
          "runtime kabul etmezse mesajı çöpe atmıyor, sıradan prompt olarak gönderiyor"],
         ["<code>/btw</code>", "araya yan soru sokuyor",
          "cevabı konuşma geçmişine <b>eklemiyor</b> — asıl işin bağlamını kirletmiyor"],
         ["<code>/goal</code>", "oturuma kalıcı hedef bağlıyor",
          "hem operatör hem model aynı hedefi görüyor"],
         ["<code>/loop</code>", "kendini tekrarlayan iş kuruyor",
          "konuşmaya bağlı"]])
    + fig(f_self_learning(), "Skill Workshop: öner → tara → uygula.")
    + p("Bir asistana “öyle değil, böyle yapacaksın” dediğinde o bilgi nereye "
        "gidiyor? Akla ilk gelen cevap belleğe yazmak. Ama bellekteki bir satır "
        "bir <i>olgu</i>dur; sonraki oturumun izleyeceği bir <i>prosedür</i> "
        "değildir. “Kullanıcı tabloları sever” ile “rapor yazarken önce şunu, "
        "sonra şunu yap” aynı şey değil.")
    + neden(
        p("OpenClaw'ın cevabı: kalıcı birim <b>skill</b>. Bir düzeltme ya da "
          "başarıyla biten bir iş, yönetilen bir yoldan geçip yeniden "
          "kullanılabilir bir prosedüre dönüşüyor. Kritik olan aradaki kapı: "
          "öğrenilen şey <b>doğrudan davranışa yazılmıyor</b> — önce öneriliyor, "
          "sonra taranıyor, sonra uygulanıyor.")),
)

chapter(
    "26", "Trajectory ve dayanıklılık frenleri",
    ["“Ajan neden bunu yaptı” sorusunun cevabını baştan kaydetmek",
     "Model failover'daki dört fren",
     "Oturum yapışkanlığının maliyet kararı olması",
     "Döngü kırıcının iki yönlü ince ayarı"],
    fig(f_trajectory(), "Trajectory: oturumun uçuş kayıt cihazı.")
    + p("“Ajan neden bunu yaptı?” sorusunun cevabı normalde elde yoktur; modele "
        "tam olarak ne gittiğini görmek için prompt'u elle yeniden kurman "
        "gerekir. Trajectory bunu baştan kaydediyor: modele giden prompt ve "
        "sistem prompt'u, gönderilen tool tanımları, çağrı ve sonuç zinciri, "
        "süreler ve hatalar.")
    + neden(
        p("Ve <b>redaksiyon varsayılan</b>. Bu küçük bir ayrıntı gibi duruyor "
          "ama değil: hata raporu paylaşmanın sır paylaşmak anlamına gelmemesi, "
          "insanların gerçekten rapor göndermesini sağlayan şey."))
    + fig(f_failover(), "Failover: profil rotasyonu ve üç fren.")
    + p("Bir model sağlayıcısı düştüğünde sırayla başka profiller deneniyor — "
        "ama körlemesine değil. <b>Cooldown</b> az önce düşen profili bir süre "
        "atlıyor, <b>auth-hata önbelleği</b> aynı 401'i tekrar yemeyi "
        "engelliyor, <b>faturalama kilidi</b> kotası bitmiş profili devre dışı "
        "bırakıyor.")
    + neden(
        p("En ilginç ayrıntı dördüncüsü: <b>oturum yapışkanlığı</b>. Aynı oturum "
          "mümkün olduğunca aynı profilde kalmaya çalışıyor ve gerekçesi belgede "
          "açıkça yazılı — <i>cache-friendly</i>. Model değiştirmek prompt "
          "cache'ini yakıyor, çünkü cache isteğin başındaki değişmeyen kısımdan "
          "çalışıyor. Yani gereksiz yere model değiştirmemek bir <b>maliyet</b> "
          "kararı. <b>[kaynak]</b>"))
    + fig(f_loopguard(), "Döngü kırıcı: aynı üçlü tekrarlanırsa koşu iptal.")
    + tuzak(
        p("En pahalı hata sonsuz döngü, ve en sinsi biçimi şu zincir: bağlam "
          "doluyor, sıkıştırma çalışıyor, model sıkıştırma yüzünden ne yaptığını "
          "unutup aynı tool'u aynı argümanla tekrar çağırıyor, bağlam yine "
          "doluyor.")
        + p("İnce ayar <b>iki yönlü</b> ve ikisi de gerekli. "
            "<code>exec</code> için hash hesaplanırken <b>oynak</b> alanlar "
            "(süre, PID, çalışma dizini) dışarıda bırakılıyor — yoksa aynı komut "
            "her seferinde farklı görünür ve döngü hiç yakalanmaz. Giden "
            "mesajlarda ise tam tersi: oynak id'ler çıkarılıyor ki iki ayrı "
            "gönderim yanlışlıkla aynı sanılmasın.")),
)

chapter(
    "27", "Değiştirilebilirlik ve sessiz altyapı",
    ["Bağlam motorunun dört eklenti noktası",
     "Belleğin bir sözleşme olması",
     "Genişletme noktasının güvenlik sınırının üstünde durması",
     "Hiçbir özellik listesinde yer almayan beş paket"],
    fig(f_ctx_engine(), "Bağlam motoru dört noktada değiştirilebilir.")
    + p("Çoğu harness'ta “modele ne gönderilecek” kararı çekirdeğe gömülüdür. "
        "OpenClaw'da bu kararın verildiği <b>dört an da eklenti yüzeyi</b>: "
        "mesaj bağlama eklenirken, model koşusundan hemen önce, pencere "
        "dolduğunda, ve tur bittiğinde. Sıkıştırma bile devralınabiliyor — bir "
        "eklenti <code>ownsCompaction</code> bayrağını kaldırırsa sıkıştırmanın "
        "tamamı ona geçiyor.")
    + tuzak(
        p("Bütün bu esnekliğin altında pazarlığa açık olmayan tek bir kural "
          "duruyor: <b>tool çağrısı sonucundan ayrılmaz</b>. Ayrılırsa model, "
          "cevabını hiç görmediği bir çağrı yapmış gibi görünür, ve çoğu "
          "sağlayıcı bu şekli geçersiz sayıp isteği reddeder."))
    + fig(f_memory_write(), "Bellek bir sağlayıcı sözleşmesi: builtin · honcho · qmd.")
    + p("Bellekte tek bir uygulama yok; bir <b>sözleşme</b> var ve onu uygulayan "
        "birden çok sağlayıcı. 83 dosyalık ayrı bir SDK paketi ayrılmış olması "
        "işin ne kadar ciddiye alındığını gösteriyor.")
    + neden(
        p("Sözleşmenin değişmeyen kısmı: her bellek kaydının bir <b>köken</b> "
          "sınıfı var, kapalı bir kümeden seçiliyor ve ayrı bir sütunda duruyor "
          "— yani <b>model onu düzyazıyla yazamıyor</b>. Sağlayıcı değişse de bu "
          "kural değişmiyor. Sonuç önemli: “belleği değiştirebilirsin” ile "
          "“güvenlik sınırını gevşetebilirsin” aynı şey değil. Genişletme "
          "noktası, sınırın <i>üstünde</i> duruyor."))
    + p("<b>Sessiz altyapı.</b> Bir harness'ın görünmeyen yarısı beş pakette: "
        "<code>normalization-core</code> farklı kanallardan geleni tek biçime "
        "indiriyor, <code>markdown-core</code> modelin yazdığını her kanalın "
        "kaldırabileceği biçime çeviriyor (WhatsApp'ın markdown'ı Slack'inki "
        "değil), <code>terminal-core</code> TUI çıktısını üretiyor, "
        "<code>retry</code> yeniden deneme politikasını topluyor, "
        "<code>net-policy</code> IP ayrıştırması, SSRF engellemesi ve URL "
        "redaksiyonu yapıyor.")
    + tuzak(
        p("Sonuncusu iyi bir örnek: <code>net-policy</code> olmadan "
          "<code>web_fetch</code>, modelin eline verilmiş bir <b>iç ağ "
          "tarayıcısına</b> dönüşüyor. Hiçbir özellik listesinde yer almayan bu "
          "paketler, biri eksik olduğunda hemen fark ediliyor.")),
)

# ═══════════════════════════════════════════════ KISIM 5 — karşılaştırma

part("5", "Karşılaştırma ve seçim",
     "Beş çerçeve yan yana, ve “hangisi” sorusunun işe yarar hâli: "
     "hangi kısıt altında hangisi.")

chapter(
    "28", "Beş çerçeve, tek tabloda",
    ["Her çerçevenin merkezî metaforu",
     "Eksen eksen karşılaştırma",
     "Hangi iddiaların koşturularak doğrulandığı"],
    p("Aşağıdaki tabloda <b>AutoGen, MAF ve Google ADK sütunları "
      "[kaynak]</b> — koşturuldu ya da birincil kaynaktan doğrulandı. "
      "Diğerleri <b>[teyitsiz]</b>.")
    + table(
        ["Çerçeve", "Merkezî metafor", "Temel yapı taşları"],
        [["<b>AutoGen</b>", "Aktör", "<code>AgentId(type,key)</code> · runtime · topic · abonelik"],
         ["<b>MAF</b>", "Tipli akış", "<code>Agent</code> · <code>Workflow</code> · executor · session"],
         ["LangGraph", "Graf", "düğüm + kenar + state; checkpointer"],
         ["CrewAI", "İnsan ekibi", "Agent · Task · Crew (rol/backstory)"],
         ["OpenAI Agents SDK", "Devir", "Agent · Tool · Handoff · Guardrail"],
         ["Google ADK", "Derlenen graf", "<code>Agent</code> + <code>Workflow</code>, <code>validate_graph()</code>"]])
    + table(
        ["Eksen", "AutoGen", "MAF", "LangGraph", "Agents SDK", "ADK"],
        [["iletişim", "pub/sub ortak thread", "tipli kenarlar",
          "paylaşılan state", "yalnız handoff", "graf kenarları"],
         ["akış kararı", "konuşmacı seçimi", "kenarlar + hazır girdi",
          "kenarlar", "ajanın kendisi", "derlenen kenarlar"],
         ["hata ne zaman", "çalışma zamanı", "çalışma zamanı", "çalışma zamanı",
          "çalışma zamanı", "<b>graf kurulurken</b>"],
         ["runtime", "<b>aktör, dağıtılabilir</b>", "tek süreç (bugün)",
          "graf yürütücü", "tur döngüsü", "graf motoru"],
         ["durability", "zayıf", "<b>checkpoint</b>", "<b>checkpointer</b>",
          "session", "<code>Session</code>"],
         ["desen çeşitliliği", "<b>5 takım tipi</b>", "workflow + hazır kalıplar",
          "supervisor/swarm", "tek", "graf + Task"],
         ["yaşam döngüsü", "<b>bakım modu</b>", "aktif", "aktif", "aktif", "aktif"]])
    + tuzak(
        p("<b>İsim karışıklığı.</b> “AutoGen” dört ayrı şeye işaret ediyor: "
          "<code>microsoft/autogen</code> v0.4+ (bu belgedeki), terk edilmiş "
          "v0.2 (<code>ConversableAgent</code>, <code>initiate_chat</code>), "
          "<code>ag2ai/ag2</code> forku, ve halef MAF. Bir kaynakta "
          "<code>ConversableAgent</code> görüyorsan o kaynak v0.2 ya da AG2 "
          "anlatıyor ve buradaki hiçbir şeyle uyumlu değil. <b>[kaynak]</b>"))
    + neden(
        p("<b>Graf konusunda taraflar yer değiştirdi.</b> AutoGen grafı "
          "<code>GraphFlow</code> olarak beş takım tipinden <i>biri</i> diye "
          "ekledi; ADK grafı <b>merkeze</b> aldı; MAF ise grafı veri akışına "
          "çevirdi. Yani “graf mı konuşma mı” sorusu üç çerçevede üç farklı "
          "cevap alıyor.")),
)

chapter(
    "29", "Ne zaman hangisi",
    ["Kısıta göre seçim tablosu",
     "“Ajan gerekmiyor” cevabının ne zaman doğru olduğu"],
    p("“Hangisi daha iyi” sorusunun cevabı yok; “<b>bu kısıt altında hangisi</b>” "
      "sorusunun var.")
    + table(
        ["Kısıt / ihtiyaç", "Öneri", "Neden"],
        [["Yeni bir proje, Microsoft dünyası", "<b>MAF</b>",
          "AutoGen bakım modunda; halef aktif ve aynı ekipten"],
         ["Mevcut AutoGen kodu, çalışıyor", "<b>AutoGen'de kal, gömerek</b>",
          "Geçişin bedeli var; ince bir arayüzün arkasına al, sonra taşı"],
         ["Gerçek eşzamanlılık, çok makine", "<b>AutoGen</b> (bugün)",
          "MAF tek süreç; dağıtık planlanan ama yok <b>[kaynak]</b>"],
         ["Akış önceden belli, öngörülebilirlik şart", "<b>ADK</b> ya da LangGraph",
          "Graf derleme zamanında doğrulanıyor; AutoGen'de hata çalışma zamanında"],
         ["Uzun süren iş, kaldığı yerden devam", "<b>MAF</b> ya da LangGraph",
          "Checkpoint ve duraklama birinci sınıf"],
         ["En düşük öğrenme eğrisi, tek devir modeli", "Agents SDK",
          "Tek desen; ama ölçülen en pahalı desen de o <b>[ölçüldü]</b>"],
         ["Gerçek makinede, insan gözetiminde koşacak", "<b>harness</b> (OpenClaw gibi)",
          "Onay, sır, zamanlama ve denetim kütüphanede yok"],
         ["İş bir fonksiyonla yapılabiliyor", "<b>ajan kullanma</b>",
          "MAF kılavuzunun kendi cümlesi <b>[kaynak]</b>"]])
    + neden(
        p("Son satır ciddi bir öneri ve kaynağı satıcının kendisi: <i>“If you "
          "can write a function to handle the task, do that instead of using an "
          "AI agent.”</i> Bir ajan, belirsizlik olmayan bir işte yalnızca "
          "pahalı ve öngörülemez bir <code>if</code> zinciridir.")),
)

chapter(
    "30", "Kurumsal soru: ne alınır, ne alınmaz",
    ["Mekanizmanın taşınabilir, güven modelinin taşınamaz olması",
     "Bir kurumun kütüphaneye ekleme yapması gereken yerler",
     "Karar özeti"],
    fig(f_gateway(), "Kurumsal bir asistanda çerçeve ile kontrol düzleminin "
                     "ayrılması.")
    + p("Bu belgede anlatılan her <b>mekanizma</b> kurumsal bir asistana "
        "taşınabilir: kapı, donmuş plan, köken sınıfı, iki kayıt hattı, üç "
        "eksen. Taşınamayan tek şey <b>güven modeli</b>.")
    + tuzak(
        p("OpenClaw <b>tek bir güvenilen operatörün</b> etrafında tasarlanmış. "
          "Belgelerindeki bütün “bu bir güvenlik sınırı değildir” cümleleri "
          "buradan geliyor: o modelde zaten herkes güvenilir, dolayısıyla "
          "ayrımlar bir kolaylıktan ibaret. Birbirine güvenmeyen departmanların "
          "olduğu bir kurumda ise <b>aynı cümlelerin her biri bir açık</b>."))
    + table(
        ["Kurumun eklemesi gereken", "Neden kütüphanede yok"],
        [["Fail-closed uyum kaydı", "Kütüphane operasyonel kayda göre tasarlanmış"],
         ["Kimlik ve yetki (kim, ne, nerede)", "Kütüphane tek operatör varsayıyor"],
         ["Sır sınırı", "Model istemcisine anahtar veriliyor, ötesi sende"],
         ["Yan etki tekrarı koruması", "Dayanıklı yürütme iki çerçevede de yok"],
         ["Dış içerik sarmalayıcısı", "Prompt injection savunması uygulama katmanında"]])
    + fig(f_atlas(), "Kurumsal asistan: çerçeve içeride, kontrol düzlemi dışarıda.")
    + neden(
        p("Karar üç cümlede özetlenebiliyor. <b>Çerçeveyi göm</b> — ince bir "
          "arayüzün arkasına al, çünkü AutoGen bakım modunda ve halefe geçiş "
          "bir gün gelecek. <b>Harness'tan öğren</b> — mekanizmaları al, "
          "TypeScript kodunu değil. <b>Güven modelini yeniden kur</b> — "
          "taşınamayan tek parça o, ve kurumsal riskin tamamı orada.")),
)
