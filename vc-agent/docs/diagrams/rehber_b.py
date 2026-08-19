"""Rehber, ikinci yarı: OpenClaw · karşılaştırma · seçim.

`make_rehber.py` tarafından exec ediliyor.
"""

from make_ogretici import (  # noqa: F401
    chapter, code, dene, fig, h3, neden, olcum, out, p, part, shell, table,
    tuzak, two,
)
from figures import (  # noqa: F401
    f_atlas, f_ctx_engine, f_durable, f_external_content, f_frozen_plan,
    f_gate, f_gateway, f_memory_tiers, f_oc_arch, f_scopes, f_secrets,
    f_task_stack, f_three_axes, f_two_ledgers,
)

# ═══════════════════════════════════════════════════════ KISIM 4 — OpenClaw

part("4", "OpenClaw: harness katmanı",
     "Kütüphane değil, uçtan uca bir araç. Bir ajanı gerçek bir makinede "
     "koştururken ortaya çıkan soruların cevapları burada.")

chapter(
    "14", "Harness nedir, kütüphaneden farkı ne",
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
    "15", "Üç kontrol ekseni",
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
    "16", "Bellek, bağlam ve dış içerik",
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
    "17", "Zamanlama, dayanıklılık ve denetim",
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

# ═══════════════════════════════════════════════ KISIM 5 — karşılaştırma

part("5", "Karşılaştırma ve seçim",
     "Beş çerçeve yan yana, ve “hangisi” sorusunun işe yarar hâli: "
     "hangi kısıt altında hangisi.")

chapter(
    "18", "Beş çerçeve, tek tabloda",
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
    "19", "Ne zaman hangisi",
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
    "20", "Kurumsal soru: ne alınır, ne alınmaz",
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
