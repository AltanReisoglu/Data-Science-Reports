"""Slide content, second half: our gateway, OpenClaw's internals, Atlas.

Companion to `slides_a.py`; see the note there. The engine injects the helpers.
"""

# ══════════════════════════════════════════════════════ KISIM IV — VC Gateway

part("IV", "VC Gateway — bizim kurduğumuz", "OpenClaw'ın biçimi, AutoGen'in motoru",
     ["Neden iki sistemi birleştirdik", "Maliyet gradyanlı huni", "Dört ilke",
      "Onay kapısı ve bulduğumuz hata", "/openclaw kaçış kapağı",
      "İki yönlü MCP köprüsü", "Canlı mekanizma paneli",
      "Belge araması, testler, hukuki sınır"])

slide("IV · gateway", "Ne kurduk ve neden",
      fig(f_gateway(), cap_mm=49)
      + cols(
          "<p>AutoGen bir kapı kavramı sunmuyor: bir ajanın dış dünyaya yaptığı "
          "çağrıyı durdurup insana sormak istiyorsan, o mekanizmayı kendin "
          "kuruyorsun.</p>"
          "<p>OpenClaw o mekanizmayı sunuyor ama ajan döngüsü sıradan — takım, "
          "graf, sonlandırma gibi soyutlamaları yok.</p>"
          "<p>Birleştirdik: <b>biçim</b> OpenClaw'dan (kapı, politika, kayıt), "
          "<b>motor</b> AutoGen'den (GraphFlow, akış, sonlandırma).</p>",
          f"<p>{tag('m')} 214 testle başladı, bugün daha fazla. Testlerin bir kısmı "
          "çerçeveyi değil <b>bizim varsayımlarımızı</b> tutuyor — taramanın "
          "GraphFlow olduğu, onay metninin id taşıdığı, engelli alan adlarının "
          "koşulsuz olduğu gibi.</p>"
          + quote("Bir üçüncü taraf kütüphaneyi test etmek genelde israftır. Kendi "
                  "varsayımını test etmek değildir — çünkü kayan şey kütüphane "
                  "değil, senin ona dair inancındır.", "b")))

slide("IV · gateway", "Huni — maliyet gradyanlı",
      table(["kademe", "ne yapar", "aday sayısına etkisi", "maliyet"],
            [["1 · toplama", "anahtarsız kaynaklar, hız sınırlı", "yüzlerce girer", "≈0"],
             ["2 · kural filtresi", "tarih, dil, tekilleştirme", "kabaca yarıya iner", "≈0"],
             ["3 · ucuz model", "sinyal mi, değil mi", "onlara iner", "düşük"],
             ["4 · orta model", "zenginleştirme, yapılandırma", "aynı kalır", "orta"],
             ["5 · güçlü model", "yalnız finalistler", "birkaç tane", "yüksek"]])
      + cols(
          quote("İlke: <b>pahalı olan en sona.</b> Bir aday beşinci kademeye "
                "geldiğinde onu oraya taşıyan dört ucuz karar zaten verilmiştir. "
                "Ters sıralama aynı sonucu 10–20 kat pahalıya üretir.", "g"),
          f"<p>{tag('m')} Sayılar <code>funnel</code> nesnesinde taşınıyor ve rapor "
          "kaç adayın hangi kademede elendiğini söylüyor. Bu, huninin çalışıp "
          "çalışmadığını <b>görünür</b> kılıyor: ikinci kademe hiçbir şey elemiyorsa "
          "filtre yanlış yazılmıştır.</p>"))

slide("IV · gateway", "Dört ilke",
      cols(
          "<p><b>1 · Her cevap kaynağını söyler.</b> Söyleyemiyorsa cevap değildir. "
          "Belge aramasında her isabet dosya adı ve satır numarasıyla geliyor.</p>"
          "<p><b>2 · Sıfır sonuç bir açıklama borcudur.</b> Boş liste "
          "\"bulunamadı\" demek değil; \"şu kaynaklar şu yüzden düştü, şu filtre "
          "şu kadarını eledi\" demektir.</p>",
          "<p><b>3 · Pahalı olan en sona.</b> Huni bunun uygulaması.</p>"
          "<p><b>4 · Dışarı giden her çağrı kapıdan geçer.</b> Tek istisnası, "
          "yazarın kendi elleriyle yazdığı <code>/openclaw</code> — ve o istisna "
          "bilinçli, belgeli ve sınırlı.</p>")
      + quote("Dördü de koda gömülü, slogan değil: her birinin bir testi var. Bir "
              "ilke test edilemiyorsa o bir niyet beyanıdır, bir kural değil.", "b"),
      foot="Bu dördü docs/04-vc-agentic-akis.md'de gerekçeleriyle duruyor.")

slide("IV · gateway", "Onay kapısı",
      fig(f_gate(), cap_mm=50)
      + cols(
          "<p>Kapı workbench katmanında: ajan <code>call_tool</code> çağırıyor, "
          "kapı araya giriyor. İzin verirse tool koşuyor; vermezse ajana bir "
          "<b>gerekçe</b> dönüyor — sessiz bir düşürme değil.</p>"
          "<p>Gerekçenin ajana dönmesi önemli: model neden reddedildiğini görüp "
          "başka bir yol deneyebiliyor.</p>",
          f"<p>{tag('m')} <b>Bulduğumuz hata:</b> çağıran kendi gerekçesini verince, "
          "varsayılan metnin <b>yerine</b> geçiyordu — ve varsayılan metin onay "
          "id'sini taşıyordu.</p>"
          "<p>Sonuç: arayüz, onaylanabilir ve bekleyen bir istek için \"onay yolu "
          "yok\" çiziyordu. İstek duruyordu, düğme yoktu.</p>"
          "<p>Düzeltme: id her zaman <b>eklenir</b>, asla değiştirilmez.</p>"),
      foot="Test istemcinin gerçek regex'ine göre yazıldı ve önce eski koda karşı düşürüldü.")

slide("IV · gateway", "/openclaw — kaçış kapağı",
      cols(
          code("/openclaw adın ne\n"
               "  → OpenClaw'ın kendi ajanına gider\n\n"
               "/openclaw skills.status\n"
               "  → RPC metodu olarak gider"),
          "<p>Ayrımı <b>nokta</b> yapıyor: token'da nokta varsa metot, yoksa cümle. "
          "Basit bir kural, ama tahmin etmiyor — belirsizse metot saymıyor.</p>"
          "<p>Cümleler <code>chat.send</code> ile gidiyor "
          "(<code>idempotencyKey</code> zorunlu — ilk denemede öğrendik), cevap ise "
          "<code>chat.history</code> yoklanarak alınıyor.</p>"
          f"<p>{tag('m')} Çünkü <code>chat.send</code> cevabı değil "
          "<code>{runId, status:\"started\"}</code> döndürüyor. Koşu sürerken "
          "geçmişte boş asistan satırları görünebiliyor; onları boş sayıp "
          "beklemek gerekiyor.</p>")
      + quote("<b>Bilerek kapıyı atlıyor</b>: bunu yazan zaten yazarın kendisi. Ama "
              "YASAK metotlar yine reddediliyor. Uygulama çok kullanıcılı olursa bu "
              "<code>peer==\"local\"</code> varsayımı kırılır — ve o gün bu slayt "
              "bir hata raporudur.", "r"))

slide("IV · gateway", "İki yönlü MCP köprüsü",
      cols(
          "<p><b>Onlardan bize:</b> <code>McpWorkbench</code>, OpenClaw'ın "
          "tool'larını AutoGen ajanına veriyor. Ajan bunları kendi tool'larından "
          "ayırt etmiyor.</p>"
          "<p><b>Bizden onlara:</b> kendi tool'larımızı bir MCP sunucusu olarak "
          "sunuyoruz; OpenClaw ajanı onları çağırabiliyor.</p>"
          "<p>Aynı protokol iki yöne de çalışıyor — MCP'nin asıl kazandırdığı bu.</p>",
          f"<p>{tag('m')} <b>Çizdiğimiz sınır:</b> "
          "<code>~/.openclaw/openclaw.json</code> sır tutuyor ve kodumuz onu "
          "<b>okumuyor</b>. Yalnız <code>openclaw</code> ikilisiyle konuşuyoruz.</p>"
          "<p>Bu bilinçli: yapılandırma dosyasını okumak, o sırları bizim süreç "
          "sınırımıza taşımak olurdu. Okumadığın şey, sızdıramadığın şeydir.</p>")
      + quote("Aynı ilke OpenClaw'ın kendi SecretRef kararında da var (Kısım VI): "
              "sırrı okuyabilen her bileşen, sırrın saldırı yüzeyinin parçasıdır.",
              "b"))

slide("IV · gateway", "Canlı mekanizma paneli",
      cols(
          f"<p>{tag('m')} Sorulan her soruda ekranda <b>o an hangi AutoGen "
          "mekanizmasının koştuğu</b> çiziliyor: bağlam kurulumu → model çağrısı → "
          "token akışı → tool isteği → kapı → tool koşumu → döngü → bitiş.</p>"
          "<p>Her mekanizma gerçek sınıf adını ve modül dosyasını söylüyor. Bir test "
          "o dosyaların diskte gerçekten var olduğunu doğruluyor — yani panel "
          "kayarsa test kırılıyor.</p>"
          "<p>Sunucu tarafı <code>stages.py</code>: mekanizma kataloğu tek kaynakta, "
          "ve alt süreçten gelen aşamalar <code>##STAGE</code> satırlarıyla "
          "taşınıyor.</p>",
          "<p><b>Panelin dürüstlük kuralı:</b> core şeridi <code>routed: 0</code> "
          "yazıyor — çünkü sohbet turu core pub/sub'ı gerçekten kullanmıyor.</p>"
          + quote("Uydurulmuş bir kutuyu yakmak, panelin öğrettiği her şeyi "
                  "değersizleştirirdi. Bir öğretme aracının ilk görevi doğru "
                  "olmaktır; ikinci görevi etkileyici olmak.", "b")
          + "<p>Grafiği kaldırıp yerine <b>yalnız o an koşan mekanizmanın kendi "
            "şemasını</b> çizmeye geçtik — akış çubuğu her şeyi aynı anda "
            "gösterdiği için hiçbir şeyi göstermiyordu.</p>"))

slide("IV · gateway", "Belge araması — ve bir kirlenme",
      cols(
          "<p><code>docs/</code> 1.18&nbsp;MB. Prompt'a sığmaz, aranması gerekir. "
          "TF-IDF seçildi, embedding değil — üç gerekçeyle:</p>"
          "<p>· indeks ikinci bir bayatlayabilir şeydir, ve bayat indeks eski "
          "belgeden kendinden emin cevap verir<br>"
          "· 675 bölümü gömmek endpoint'in ayakta olmasını gerektirir — "
          "<i>belge</i> aramasının model sağlayıcısıyla düşmesi saçmadır<br>"
          "· bu belgeler tam terimlerle dolu (<code>model_info</code>, "
          "<code>ClosureAgent</code>), ki bu leksik skorlamanın en güçlü olduğu yer</p>",
          f"<p>{tag('m')} <b>Ölçülen kirlenme:</b> 63&nbsp;KB'lık kaydedilmiş bir "
          "blog sayfası <code>docs/</code>'a düşünce idf ağırlığı kaydı ve kendi "
          "tuzaklar sayfamız, sahibi olduğu bir sorguda ilk dörtten çıktı.</p>"
          "<p>Düzeltme: yalnız numaralı seri indeksleniyor. <code>docs/</code>'a "
          "okumak için bir şey koymak artık aramayı bozmuyor.</p>"
          + quote("Arama kalitesinin, birisi okumak için bir şey kaydettiği için "
                  "bozulması, sahip olmaya değer bir hata modu değildir.", "g")))

slide("IV · gateway", "Anahtar olayı ve play.py",
      cols(
          f"<p>{tag('m')} Bugün ölçüldü: <code>.env</code>'deki anahtar DeepSeek'e "
          "karşı <b>401</b> aldı. Endpoint'e ulaşıldı (1.52 sn), istek biçimi "
          "doğruydu, <code>model_info</code> tuzağı tetiklenmedi — reddedilen şey "
          "yalnızca anahtardı.</p>"
          "<p>Ek gözlem: 78 karakterlik bir anahtar, DeepSeek biçimine "
          "(<code>sk-</code> + ~32) uymuyor.</p>",
          "<p><code>play.py</code> bunun için yazıldı: boru hattının <b>kendi</b> "
          "config'ini kullanan tek atışlık bir erişim testi. Yeşil çıkarsa "
          "<code>/api/chat</code> de çalışır.</p>"
          "<p>İki kuralı var:</p>"
          "<p>· anahtarı asla basmaz — uzunluk + kısa önek maskesi<br>"
          "· eksik yapılandırmayla <b>ateş etmez</b></p>")
      + quote("Yarım yapılandırmayla — URL tahmin ederek, ya da bir anahtarı ait "
              "olmadığı endpoint'e göndererek — koşan bir test, hiç testten "
              "kötüdür. Sonucu yorumlanamaz, ve yan etkisi gerçektir.", "r"))

slide("IV · gateway", "Test disiplini",
      cols(
          "<p>Testlerin bir kısmı çerçeveyi değil <b>bizim varsayımlarımızı</b> "
          "tutuyor:</p>"
          "<p>· taramanın GraphFlow olduğu<br>"
          "· onay metninin id taşıdığı<br>"
          "· <code>docs/</code>'a düşen yabancı dosyanın indekslenmediği<br>"
          "· engelli alan adlarının koşulsuz olduğu<br>"
          "· panelin adını verdiği modüllerin diskte var olduğu</p>",
          f"<p>{tag('m')} <b>Kural:</b> bir hatanın testi, düzeltmeden <b>önce</b> "
          "eski koda karşı düşürülür. Kırmızı olduğunu görmeden düzeltme "
          "yapılmıyor.</p>"
          + quote("Onay id'si hatasında bunu uyguladık: test önce kırmızıydı, sonra "
                  "düzeltme geldi. Ters sırada yazılan bir test, düzeltmeyi değil "
                  "<b>kendini</b> doğrular — çünkü zaten geçen bir dünyada "
                  "yazılmıştır.", "g")))

slide("IV · gateway", "Hukuki sınır — koşulsuz",
      cols(
          "<p>Engelli alan adları: LinkedIn · Facebook · Instagram · X/Twitter · "
          "Crunchbase · PitchBook</p>"
          "<p>Bu liste <b>koşulsuz</b>. Yapılandırmayla açılamıyor, bir bayrakla "
          "atlanamıyor, \"sadece test için\" istisnası yok.</p>",
          f"<p>{tag('k')} Gerekçe hukuki: bu sitelerin kullanım şartları otomatik "
          "erişimi yasaklıyor. Bir VC aracı için toplanan veri, <b>toplanma biçimi "
          "yüzünden</b> kullanılamaz hale gelirse hiç toplanmamış sayılır.</p>"
          "<p><code>test_policy.py</code> bu sınırı tutuyor — yani birisi ileride "
          "gevşetmeye çalışırsa test kırılıyor.</p>")
      + quote("Bu yüzden agent-reach'in Twitter/Reddit kazıma yüzeyi bizde doğrudan "
              "kullanılamıyor. Ekleme kararı verilmeden önce bu çatışmanın "
              "çözülmesi gerekiyor — \"sonra bakarız\" bir çözüm değil.", "r"))

slide("IV · gateway", "Şekiller neden elle çizilmiş görünüyor",
      cols(
          f"<p>{tag('m')} <b>Ölçülen tuzak:</b> <code>feTurbulence</code> + "
          "<code>feDisplacementMap</code> tarayıcıda titrek görünüyor, ama Chrome "
          "PDF'e basarken filtreyi <b>sessizce düşürüyor</b>. Filtreli ve filtresiz "
          "çıktı piksel piksel aynıydı — ve hiçbir uyarı yoktu.</p>"
          "<p>Çözüm: titreşimi yol verisine <b>pişirmek</b>. Buradaki her dikdörtgen, "
          "köşeleri kaydırılmış gerçek bir <code>path</code>; render anında hiçbir "
          "şeyin işbirliği yapması gerekmiyor.</p>",
          "<p>Üç kural var, üçüncüsü en çok atlanan:</p>"
          "<p><b>1.</b> Köşeler ıskalar — her tepe bir iki piksel kayar<br>"
          "<b>2.</b> Kenarlar yay çizer — her kenar hafif şişkin bir eğri<br>"
          "<b>3.</b> <b>Her kontur iki kez çizilir</b>, farklı sapmayla — kalem "
          "üstünden geçmiş gibi. Excalidraw görünümünün asıl kaynağı bu.</p>"
          f"<p>{tag('k')} Sapma tohumlu bir PRNG'den geliyor; aynı belge her "
          "derlemede bayt-eş çıkıyor.</p>")
      + quote("Kendini her derlemede yeniden karan bir diyagram, her derlemeyi "
              "gözden geçirmede bir değişiklik gibi gösterir.", "b"))

# ══════════════════════════════════════════════════════ KISIM V — OpenClaw içi

part("V", "OpenClaw'ın içi", "Canlı RPC ile ölçüldü — iddia değil, sayı",
     ["Yüzeyin büyüklüğü", "Skill kademeli açığa çıkarma", "Sürüm önbelleği",
      "Beş bellek katmanı", "İki şeritli geri çağırma",
      "Bağlam motoru ve sıkıştırma", "Zamanlama", "Protokol, eklenti, node"])

slide("V · openclaw", "Yüzey — ölçülen sayılar",
      cols(
          table(["ne", "sayı", "nasıl ölçüldü"],
                [["Gateway RPC metodu", "351", "canlı RPC"],
                 ["Tool", "44", "<code>tools.catalog</code>"],
                 ["Tool grubu", "14", "aynı çağrı"],
                 ["Tool profili", "4", "aynı çağrı"],
                 ["Kurulu skill", "74", "<code>skills.status</code>"],
                 ["Model-görünür skill", "40", "aynı çağrı"]]),
          f"<p>{tag('m')} Hepsi <b>çalışan bir gateway'e sorularak</b> alındı, "
          "belgeden okunarak değil. Bu ayrım önemli: belge sürümler arasında "
          "kayabilir, çalışan sistem kayamaz.</p>"
          "<p>74 kurulu skill'in yalnız 40'ının model-görünür olması bir "
          "yapılandırma kararı — geri kalanı yüklü ama kapalı. Yani \"kurulu\" ile "
          "\"etkin\" farklı iki sayı, ve ikisini karıştırmak yüzeyi iki kat büyük "
          "göstermek olurdu.</p>"),
      foot="Ölçümün tamamı: docs/pdf/openclaw-ici.pdf")

slide("V · openclaw", "Kademeli açığa çıkarma",
      fig(f_skill_disclosure(), cap_mm=52)
      + cols(
          "<p>Mekanizma basit: prompt'a yalnız skill'lerin <b>indeksi</b> giriyor — "
          "ad ve tek satır açıklama. Gövdeler diskte duruyor.</p>"
          "<p>Model bir skill'e ihtiyaç duyduğunda <code>read</code> ile gövdesini "
          "çekiyor. İhtiyaç duymadıklarının gövdesi hiç yüklenmiyor.</p>",
          f"<p>{tag('m')} <b>%93</b> tasarruf: diskteki gövde boyutlarının toplamıyla "
          "prompt'a giren indeks boyutu karşılaştırılarak hesaplandı.</p>"
          "<p>Bunun ikinci bir faydası var: model 74 talimatı aynı anda görmediği "
          "için dikkati dağılmıyor. Yani kazanç yalnız token değil, <b>isabet</b>.</p>")
      + quote("Aynı fikir tool katmanında da var (Tool Search — Kısım VI). İki yerde "
              "bağımsız olarak ortaya çıkması, bunun bir numara değil bir "
              "<b>desen</b> olduğunu gösteriyor.", "g"))

slide("V · openclaw", "Sürüm önbelleği — sha256 sinyali",
      cols(
          f"<p>{tag('m')} Skill dosyalarının sha256'sı tutuluyor. Hash değişmediyse "
          "dosya yeniden okunmuyor.</p>"
          "<p>Bu, önbelleğin <b>geçerlilik sinyali</b>: zaman aşımı değil, içerik "
          "hash'i. Bir dosya bir yıl değişmezse bir yıl yeniden okunmuyor; bir "
          "saniye önce değiştiyse hemen yeniden okunuyor.</p>",
          quote("Zaman tabanlı önbellek \"ne zaman bayatlar\" sorusunu <b>tahmin "
                "etmeye</b> çalışır ve her zaman ya çok erken ya çok geç tahmin "
                "eder. İçerik tabanlı önbellek o soruyu ortadan kaldırır.", "g")
          + "<p>Bizde karşılığı yok ama olması gereken bir yer var: "
            "<code>docs_index</code> her içe aktarımda bütün korpusu yeniden "
            "işliyor. Dosya hash'i tutulsa aynı kazanç orada da olurdu.</p>"))

slide("V · openclaw", "Sistem prompt'u koddan değil, dosyalardan derleniyor",
      cols(
          "<p>Ajanın davranışı kaynak kodda değil, çalışma alanındaki markdown "
          "dosyalarında tanımlı. Runtime her oturumda bunları derleyip prompt'a "
          "koyuyor:</p>"
          "<p><code>AGENTS.md</code> — işletim talimatı, pazarlıksız kurallar<br>"
          "<code>SOUL.md</code> — ses tonu ve üslup<br>"
          "<code>IDENTITY.md</code> — ad, kimlik<br>"
          "<code>USER.md</code> — kullanıcı modeli<br>"
          "<code>MEMORY.md</code> — küratörlü uzun dönem bellek<br>"
          "<code>BOOTSTRAP.md</code> — yalnız yepyeni çalışma alanında</p>",
          f"<p>{tag('m')} Sınırlar ölçülü: dosya başına <b>20.000</b> karakter, "
          "toplam <b>60.000</b>. Aşan dosya kesiliyor ve kesildiği <b>uyarılıyor</b> "
          "(<code>bootstrapPromptTruncationWarning</code>, varsayılan "
          "<code>always</code>).</p>"
          "<p><code>memory/GG-AA-YY.md</code> günlükleri bu derlemeye <b>girmiyor</b> "
          "— yalnız <code>memory_search</code> ile isteğe bağlı geliyor.</p>")
      + quote("Enterprise açısından değerli ayrım: <b>davranış dosyada, zorlama "
              "runtime'da</b>. Bir ekip asistanın üslubunu ve kurallarını kod "
              "değiştirmeden düzenleyebiliyor; ama izin, sandbox ve onay hâlâ "
              "runtime'ın elinde — ve markdown'la gevşetilemiyor.", "g"),
      foot="docs/concepts/system-prompt.md · docs/concepts/agent-workspace.md")

slide("V · openclaw", "Beş bellek katmanı",
      fig(f_memory_tiers(), cap_mm=72)
      + f"<p>{tag('k')} Sınır ikinci ile üçüncü arasında. <b>Curated</b> küçük, "
        "her oturumda bağlamda, ve yalnız kapılı konsolidasyonla yazılıyor. "
        "<b>Episodic</b> büyük, ekleme dostu, ve yalnız arama yoluyla erişilebiliyor. "
        "Episodic'ten curated'a hiçbir şey kapıdan geçmeden çıkmıyor.</p>")

slide("V · openclaw", "Geri çağırma — iki şerit",
      cols(
          "<p><b>Şerit 1 — her zaman açık, sıfır model çağrısı.</b><br>"
          "Deterministik skor: <code>önem × tazelik × ilgi</code>. Eşiği "
          "(≥&nbsp;0.72) geçen tetikleyici bağlama enjekte ediliyor.</p>"
          "<p><b>Şerit 2 — tırmanma.</b><br>"
          "Birinci şerit yetmezse model çağrısı içeren arama devreye giriyor.</p>",
          f"<p>{tag('m')} Ayrımın maliyeti şu: normal turların çoğu bellek için "
          "<b>hiç model çağırmıyor</b>. Model yalnız gerçekten dil yargısı "
          "gerektiğinde, ve deterministik kodun çizdiği sınırların içinde devreye "
          "giriyor.</p>")
      + quote("\"Deterministik kapılar, içlerinde model yargısı.\" Bu cümle "
              "OpenClaw'ın bellek belgesinin tasarım ilkelerinden biri. Skorlama, "
              "eşikler, uygunluk ve yaşam döngüsü <b>kod</b>; model yalnız kodun "
              "izin verdiği yerde konuşuyor.", "g"),
      foot="Bu, notların üstündeki getirme katmanından ayrı: memory-search embedding + BM25 hibrit çalışıyor. İkisi çelişmiyor — farklı katman.")

slide("V · openclaw", "Bağlam motoru ve sıkıştırma",
      cols(
          "<p>Bağlam motorunun <b>dört yaşam döngüsü noktası</b> var ve eklenti "
          "olarak değiştirilebiliyor — yani bağlam kurma stratejisi çekirdeğe "
          "gömülü değil.</p>"
          f"<p>{tag('k')} Sıkıştırma bir kuralı asla bozmuyor: <b>tool çağrısını "
          "sonucundan ayırmıyor</b>. Ayırsaydı model, cevabını hiç görmediği bir "
          "çağrı yapmış gibi görünürdü — ve çoğu sağlayıcı bu şekli geçersiz "
          "sayar.</p>",
          f"<p>{tag('k')} Hatalar cevabı bloklamıyor: cevap yolundaki her "
          "bellek/bağlam adımının bir timeout'u, bir geri düşüşü veya ikisi var.</p>"
          + quote("Çöken bir bellek altsistemi geri çağırma kalitesini düşürür; bir "
                  "turu asla yemez. Bu, üretimde koşan bir asistan için pazarlık "
                  "konusu olmayan bir özellik.", "g")))

slide("V · openclaw", "Zamanlama — önce haritanın kendisi",
      fig(f_task_stack(), "Soldan sağa: ne tetikler → kim karar verir → ne "
                          "kaydedilir → nasıl serileşir → nerede koşar.", cap_mm=48)
      + cols(
          "<p>\"Zamanlanmış iş\" tek bir mekanizma değil. OpenClaw bunu <b>altı ayrı "
          "parçaya</b> bölmüş ve her biri farklı bir soruyu cevaplıyor:</p>"
          "<p><b>Automations</b> — tam zamanlama<br>"
          "<b>Heartbeat</b> — yaklaşık, bağlamlı yoklama<br>"
          "<b>Tasks</b> — ne olduğunun kaydı<br>"
          "<b>Task Flow</b> — çok adımlı dayanıklı akış<br>"
          "<b>Hooks</b> — yaşam döngüsü olaylarına tepki<br>"
          "<b>Standing orders</b> — kalıcı talimat</p>",
          f"<p>{tag('k')} Belgenin kendi karar rehberi tek soruyla ayırıyor: "
          "<i>tam zamanlama mı gerekiyor, esnek mi?</i> Tam ise Automations, esnek "
          "ise Heartbeat.</p>"
          "<p>Bu ayrım önemsiz görünüyor ama arkasında bambaşka iki yürütme modeli "
          "var — sonraki slayt.</p>"
          ),
      foot="Bu bölüm uzun çünkü proaktif bir asistan demek, zamanlanmış işin doğru çalışması demek. · docs/automation/index.md")

slide("V · openclaw", "Beş zamanlama türü",
      table(["tür", "bayrak", "ne yapar"],
            [["<code>at</code>", "<code>--at</code>",
              "tek seferlik zaman damgası (ISO 8601 ya da <code>20m</code> gibi göreli)"],
             ["<code>every</code>", "<code>--every</code>",
              "sabit aralık — <code>10m</code>, <code>1h</code>, <code>1d</code>"],
             ["<code>cron</code>", "<code>--cron</code>",
              "5 ya da 6 alanlı cron ifadesi, isteğe bağlı <code>--tz</code>"],
             ["<code>on-exit</code>", "<code>--on-exit</code>",
              "izlenen bir komut <b>çıkınca</b> bir kez — tur yıkımından sağ çıkar"],
             ["<code>stream</code>", "<code>--stream-command</code>",
              "uzun ömürlü bir komutun stdout/stderr <b>satırlarından</b>"]])
      + cols(
          f"<p>{tag('m')} Son ikisi zamana <b>hiç</b> bakmıyor. "
          "<code>on-exit</code> ve <code>stream</code> olay güdümlü — "
          "\"zamanlayıcı\" kelimesi ikisini de yanlış anlatıyor.</p>"
          "<p>Zaman dilimi kuralı: dilimsiz zaman damgaları <b>UTC</b> sayılıyor. "
          "<code>--tz</code> yalnız <code>at</code> ve <code>cron</code> ile "
          "geçerli.</p>",
          f"<p>{tag('k')} <b>Yük dağıtma:</b> saat başına denk gelen tekrarlı "
          "işler (dakika <code>0</code>, saat joker) kendiliğinden <b>5 dakikaya "
          "kadar</b> kaydırılıyor — aynı anda uyanan yüz işin yük tepesi "
          "oluşturmaması için. <code>--exact</code> ile kapatılıyor.</p>"
          "<p>Bu, ölçekte fark eden ama kimsenin akla getirmediği türden bir "
          "varsayılan.</p>"))

slide("V · openclaw", "Cron'un OR tuzağı",
      cols(
          f"<p>{tag('k')} Cron ifadeleri <code>croner</code> ile ayrıştırılıyor. "
          "Ayın-günü ve haftanın-günü alanlarının <b>ikisi de</b> joker değilse, "
          "croner <b>ya biri ya öteki</b> eşleştiğinde tetikliyor — ikisi birden "
          "değil.</p>"
          "<p>Bu standart Vixie cron davranışı, yani hata değil. Ama neredeyse "
          "herkesin yanlış okuduğu bir davranış.</p>",
          code("# Niyet: \"ayın 15'i, ama yalnız Pazartesiyse\"\n"
               "0 9 15 * 1\n\n"
               "# Gerçek: her ayın 15'inde 9'da\n"
               "#         VE her Pazartesi 9'da")
          + f"<p>{tag('m')} Ayda 0–1 kez yerine <b>ayda 5–6 kez</b> çalışıyor.</p>")
      + quote("Çözüm: croner'ın <code>+</code> değiştiricisi (<code>0 9 15 * +1</code>), "
              "ya da bir alanı zamanlamaya bırakıp diğerini işin kendi içinde "
              "kontrol etmek. Bir kurumsal asistanda \"ayda beş kez rapor gönderdi\" "
              "bir arıza kaydıdır.", "r"))

slide("V · openclaw", "Koşul gözcüsü — zamana değil duruma bağlanmak",
      cols(
          "<p>Bir <b>event trigger</b>, <code>every</code> / <code>cron</code> / "
          "<code>stream</code> zamanlamasının üstüne başsız bir <b>koşul betiği</b> "
          "ekliyor. Zamanlama geldiğinde önce betik koşuyor; yük ancak "
          "<code>fire: true</code> dönerse çalışıyor.</p>"
          "<p>Betik kalıcı bir <code>trigger.state</code> taşıyor — yani bir "
          "önceki değerlendirmeyi hatırlıyor. Böylece <b>değişim</b> tespit "
          "edilebiliyor, yalnız durum değil.</p>",
          code("// her 30 sn'de bak, ama yalnız DEĞİŞİNCE tetikle\n"
               "const s = await tools.call('exec', {...});\n"
               "json({\n"
               "  fire: s !== trigger.state?.status,\n"
               "  message: `CI: ${trigger.state?.status} -> ${s}`,\n"
               "  state: { status: s },\n"
               "});"))
      + cols(
          f"<p>{tag('k')} Bu, \"kurumsal karşılık\"ın tam olarak oturduğu yer: "
          "<i>limit aşıldığında</i>, <i>itiraz üç günü geçtiğinde</i>, <i>skor "
          "eşiği düştüğünde</i> — takvim değil, <b>iş kuralı</b>.</p>",
          f"<p>{tag('k')} Ve bir güvenlik sınırı: koşul betikleri "
          "<code>cron.triggers.enabled: true</code> istiyor, çünkü gözetimsiz "
          "koşan bir betik ayrı bir güven sınıfı. Varsayılan kapalı.</p>"))

slide("V · openclaw", "İki zamanlayıcı — ve neden ikisi birden var",
      table(["boyut", "Automations", "Heartbeat"],
            [["zamanlama", "tam (cron, tek seferlik)", "yaklaşık — varsayılan 30 dk"],
             ["oturum bağlamı", "taze (izole) <b>veya</b> paylaşılan", "tam main-session bağlamı"],
             ["task kaydı", "<b>her zaman</b> yaratılır", "<b>asla</b> yaratılmaz"],
             ["teslimat", "kanal, webhook, ya da sessiz", "main session içinde satır içi"],
             ["en uygun", "raporlar, hatırlatmalar, arka plan işleri", "gelen kutusu, takvim, bildirim"]])
      + cols(
          "<p><b>Neden ikisi?</b> Çünkü \"her 30 dakikada gelen kutusuna bak\" ile "
          "\"her sabah 9'da raporu gönder\" farklı şeyler istiyor.</p>"
          "<p>Birincisi <b>bağlam</b> istiyor — asistanın konuştuklarını bilmesi "
          "lazım. İkincisi <b>izolasyon</b> istiyor — sohbetin gürültüsü rapora "
          "karışmasın.</p>",
          f"<p>{tag('k')} Heartbeat meşgulken <b>kendiliğinden erteleniyor</b>: "
          "ana kuyruk ya da automation işi doluysa, aynı ajan için başka bir cevap "
          "koşuyorsa, ya da hedef oturumda aktif/kuyrukta iş varsa.</p>"
          "<p>Yani periyodik yoklama, gerçek işin önüne geçmiyor.</p>"))

slide("V · openclaw", "Task ledger — kayıt, zamanlayıcı değil",
      fig(f_task_lifecycle(), "Beş terminal durum, biri ölçüm.", cap_mm=58)
      + cols(
          quote("<b>Tasks are records, not schedulers</b> — automations ve heartbeat "
                "işin <i>ne zaman</i> koşacağına karar verir; task'lar <i>ne olduğunu</i> "
                "izler.", "g")
          + f"<p>{tag('k')} Task yaratanlar: ACP koşuları, subagent spawn'ları, "
          "<b>her</b> automation koşusu, CLI işlemleri, medya üretimi.</p>",
          f"<p>{tag('k')} <b>Yaratmayanlar:</b> heartbeat turları, normal sohbet "
          "turları, doğrudan <code>/komut</code> cevapları.</p>"
          "<p>Saklama: terminal kayıtlar <b>7 gün</b>, <code>lost</code> olanlar "
          "<b>24 saat</b>, sonra otomatik budanıyor.</p>"))

slide("V · openclaw", "Üç kural ki tasarımı bu yapıyor",
      cols(
          "<p><b>① Yürütme ile teslimat ayrı.</b></p>"
          f"<p>{tag('k')} Bir subagent task'ı <code>succeeded</code> kalabilir "
          "ama <code>deliveryStatus</code> <code>failed</code> olabilir. O zaman "
          "sonuç <code>blocked</code> oluyor — <b>failed değil</b>.</p>"
          + quote("Gerekçesi belgede: <i>\"Bu, tamamlanmış sonucu korur; çocuk "
                  "yürütmesini yanlışlıkla başarısız diye raporlamak yerine.\"</i>",
                  "g")
          + "<p><b>② Terminal yapışkan.</b></p>"
          f"<p>{tag('k')} Bir task terminal olduktan sonra gelen yaşam döngüsü "
          "sinyalleri onu <b>düşüremiyor</b>. Operatör iptal ettiyse, sonradan gelen "
          "bir başarı sinyali kararı değiştirmiyor.</p>",
          "<p><b>③ <code>lost</code> çalışma-zamanı farkında.</b></p>"
          f"<p>{tag('k')} Her kaynak için kanıt standardı farklı: ACP için "
          "<i>yalnız</i> canlı bir in-process tur kanıt sayılıyor — kalıcı oturum "
          "metadata'sı yetmiyor. Automation için önce runtime, sonra dayanıklı "
          "koşu geçmişi kontrol ediliyor.</p>"
          + quote("Ve en güzel cümlesi: <i>\"Çevrimdışı CLI denetimi, kendi boş "
                  "in-process durumunu otorite saymaz.\"</i> Yani <b>kanıtın "
                  "yokluğu, yokluğun kanıtı değil</b> — bu tek cümle bir "
                  "mühendislik olgunluğu göstergesi.", "b")))

slide("V · openclaw", "Yoklama yanlış şekil",
      "<p class='lead'>Task belgesinin en yüksek getirili cümlesi:</p>"
      + quote("<b>Tamamlanma itmeli (push-driven):</b> ayrılmış iş bittiğinde "
              "doğrudan bildirebilir ya da isteyen oturumu / heartbeat'i uyandırabilir "
              "— <b>bu yüzden durum yoklama döngüleri genelde yanlış şekildir.</b>", "g")
      + cols(
          "<p>Bir ajana \"işi başlat, sonra bitti mi diye sor\" dedirtmek doğal "
          "geliyor. Ama her yoklama bir model turu, ve iş beş dakika sürüyorsa on "
          "tur boşa gidiyor.</p>"
          "<p>Doğru şekil: iş bitince <b>o</b> seni uyandırıyor.</p>",
          f"<p>{tag('m')} Bizim tarafta karşılığı yok — henüz. "
          "<code>pipeline/</code>'da ayrılmış iş kavramı yok, her şey istek "
          "içinde bitiyor.</p>"
          "<p>Atlas'ta olacak: uzun süren bir sorgu ya da rapor, kullanıcıyı "
          "bekletmeden koşup bitince haber vermeli.</p>"),
      foot="Aynı ilke Task Flow'da da var: adımlar arası durum kalıcı, akış gateway restart'ını sağ atlatıyor.")

slide("V · openclaw", "Kuyruk — koşular nasıl çakışmıyor",
      cols(
          f"<p>{tag('k')} Gelen bütün otomatik cevap koşuları küçük bir "
          "<b>lane-aware FIFO</b> kuyruktan geçiyor. Amaç çakışmayı önlemek: "
          "oturum durumu, loglar, CLI stdin paylaşılan kaynaklar.</p>"
          "<p>İki katmanlı:</p>"
          "<p>· <code>session:&lt;key&gt;</code> lane → <b>oturum başına tek</b> "
          "aktif koşu<br>"
          "· global <code>main</code> lane → toplam paralellik "
          "<code>maxConcurrent</code> ile sınırlı</p>",
          f"<p>{tag('m')} Ölçülen varsayılanlar: <code>main</code> lane "
          "<code>min(16, max(8, CPU çekirdeği))</code>, <code>subagent</code> "
          "lane <b>8</b>, yapılandırılmamış lane'ler <b>1</b>.</p>"
          "<p>Ve bir kullanıcı deneyimi ayrıntısı: yazıyor göstergesi kuyruğa "
          "girer girmez tetikleniyor — koşu sırasını beklerken bile kullanıcı "
          "sistemin uyandığını görüyor.</p>")
      + quote("Bu, Atlas'a doğrudan taşınacak: aynı kullanıcının iki isteği "
              "birbirinin bağlamını bozmamalı, ama farklı kullanıcılar "
              "birbirini beklememeli. Ayrımı <b>lane</b> yapıyor.", "g"))

slide("V · openclaw", "Task Flow — çok adımlı ve dayanıklı",
      cols(
          "<p>Tek bir arka plan işi <b>task</b>. Çok adımlı bir boru hattı "
          "<b>flow</b>. Flow'un kendi durumu, JSON durumu, <b>revizyon sayacı</b> ve "
          "bağlı task kayıtları var.</p>"
          "<p>İki kip:</p>"
          "<p><b>managed</b> — plugin kodu sürücü; adımları açıkça ilerletiyor<br>"
          "<b>mirrored</b> — ayrılmış ACP/subagent spawn'ları için otomatik</p>",
          f"<p>{tag('k')} <b>Revizyon sayacı neden var:</b> her değişiklik flow'un "
          "beklenen revizyonunu taşıyor. Bayat bir yazma, daha yeni durumu "
          "ezmek yerine <b>revizyon çakışması</b> olarak reddediliyor.</p>"
          "<p>Ve iptal: iptal istendikten sonra yeni çocuk task kabul edilmiyor; "
          "flow, aktif çocuk kalmayınca <code>cancelled</code> olarak "
          "sonlanıyor.</p>")
      + quote("Flow'lar gateway restart'ını <b>sağ atlatıyor</b>; task'lar ayrılmış "
              "işin birimi olarak kalıyor. Kurumsal karşılığı: \"üç günlük bir "
              "itiraz sürecini takip et\" işinin sunucu yeniden başlayınca "
              "kaybolmaması.", "g"))

slide("V · openclaw", "Zamanlamadan Atlas'a ne taşınır",
      table(["mekanizma", "neden değerli", "Atlas'ta karşılığı"],
            [["Kayıt ile zamanlayıcının ayrılması",
              "\"ne zaman\" ile \"ne oldu\" farklı sorular",
              "denetime giden şey kayıt; zamanlama operasyon"],
             ["Yürütme ≠ teslimat",
              "biten iş, teslim edilemedi diye başarısız sayılmıyor",
              "\"rapor üretildi ama e-posta gitmedi\" ayırt edilebilir"],
             ["Terminal yapışkanlığı", "geç gelen sinyal kararı bozmuyor",
              "iptal edilen bir sorgu geri dirilmiyor"],
             ["<code>lost</code>'un kanıt standardı",
              "kanıtın yokluğu, yokluğun kanıtı değil",
              "denetimde \"bilmiyoruz\" diyebilmek"],
             ["İtmeli tamamlanma", "yoklama döngüsü token yakıyor",
              "uzun sorgu kullanıcıyı bekletmiyor"],
             ["Lane'li kuyruk", "oturum izolasyonu + kontrollü paralellik",
              "departmanlar birbirini beklemiyor"],
             ["Koşul gözcüsü", "takvim değil iş kuralı",
              "\"limit aşıldığında\" tetiklemesi"]]),
      foot="Bu yedi satır, docs/17'deki Faz 2–3 işlerinin zamanlama tarafı.")

slide("V · openclaw", "Gateway protokolü",
      cols(
          "<p>WebSocket üstünde JSON-RPC benzeri çerçeveleme. Bağlantı bir <b>el "
          "sıkışma</b> ile başlıyor: rol (<code>operator</code> / <code>node</code>), "
          "istenen kapsamlar, istemci kimliği.</p>"
          f"<p>{tag('m')} 351 metot ölçüldü. Aileler: <code>chat.*</code>, "
          "<code>session.*</code>, <code>tools.*</code>, <code>skills.*</code>, "
          "<code>audit.*</code>, <code>device.*</code>, <code>node.*</code>, "
          "<code>config.*</code>.</p>",
          f"<p>{tag('m')} <b>Ölçerken çarptığımız:</b> "
          "<code>workspace.read_file</code> diye bir metot <b>yok</b>. Uydurmuştuk; "
          "canlı gateway “unknown method” dedi.</p>"
          + quote("Bir RPC yüzeyini belgeden değil <b>kendisinden</b> sormanın "
                  "değeri bu: var sandığın metot yoksa ilk çağrıda öğrenirsin, "
                  "üç saat sonra değil.", "g")))

slide("V · openclaw", "Eklenti ve yetenek sistemi",
      cols(
          f"<p>{tag('m')} 161 extension, 22 paket. Yetenek türleri: tool sağlayıcı, "
          "kanal, model sağlayıcı, bağlam motoru, bellek sağlayıcı, sıkıştırma "
          "sağlayıcı.</p>"
          "<p>Yani <b>çekirdeğin her ilginç parçası değiştirilebilir</b> — "
          "sıkıştırma bile. Bu, mimari bir duruş: çekirdek bir çerçeve, bir ürün "
          "değil.</p>",
          "<p><b>Anahtar ayrım:</b> <i>skill</i> ile <i>plugin</i> aynı şey değil.</p>"
          "<p><b>Skill</b> = markdown talimat + isteğe bağlı betik. Modelin "
          "<i>okuduğu</i> şey.</p>"
          "<p><b>Plugin</b> = kod. Runtime'a <i>yetenek ekleyen</i> şey.</p>"
          f"<p>{tag('k')} Bir plugin tool ekler; bir skill o tool'un <b>ne zaman</b> "
          "kullanılacağını anlatır.</p>")
      + quote("İkisini karıştırmak, yetenek eklemekle talimat eklemeyi "
              "karıştırmaktır. Yeni bir tool yazman gereken yere skill yazarsan "
              "model yapamayacağı bir şeyi anlatan bir metin okur.", "r"))

slide("V · openclaw", "Node'lar — cihaz yetenekleri",
      cols(
          "<p>Gateway merkezde; <b>node</b>'lar cihaz yeteneklerini sunuyor — macOS, "
          "iOS, Android, headless. <code>node.invoke</code> ile çağrılıyorlar.</p>"
          "<p>Güven <b>pairing</b> (eşleşme) ile kuruluyor: node bağlanır, "
          "bekleyen bir talep doğar, bir operatör onaylar.</p>",
          f"<p>{tag('k')} <b>İlginç kısım:</b> eşleşme onayı, düğümün <b>bildirdiği "
          "komut listesinden</b> ek kapsam türetiyor. <code>system.run</code>, "
          "<code>fs.listDir</code>, <code>browser.proxy</code> gibi komutlar "
          "<code>operator.admin</code> istiyor; sıradan komutlar istemiyor.</p>")
      + quote("Yetki, bağlanan şeyin <b>ne yapabildiğine</b> göre türetiliyor — "
              "sabit bir role göre değil. Bu, Kısım VI'daki \"kapsam parametreden "
              "türetilir\" fikrinin donanım tarafındaki karşılığı.", "g"))

# ══════════════════════════════════════════════════════ KISIM VI — enterprise

part("VI", "Enterprise: Atlas", "Resmî repodan ne alınır, ne alınmaz",
     ["Resmî repo ölçüleri", "Tez: mekanizma taşınır, güven modeli taşınmaz",
      "Üç kontrol ekseni", "Onay = donmuş plan",
      "Denetim ve dürüst sınır beyanı", "Yetki ve dış içerik sınırı",
      "Tool Search, bellek, sırlar, telemetri",
      "ALINMAYACAKLAR", "Atlas'ın şekli ve ilk üç iş"])

slide("VI · atlas", "Kaynak — resmî repo",
      cols(
          table(["ne", "değer"],
                [["depo", "<code>github.com/openclaw/openclaw</code>"],
                 ["commit", "<code>01cc7106</code>"],
                 ["boyut", "558 MB"],
                 ["extension", "161"],
                 ["paket", "22"],
                 ["belge dosyası", "764"]]),
          f"<p>{tag('m')} Diskte: <code>~/Desktop/adapted/harnesses/openclaw</code>. "
          "Bu kısımdaki her iddianın altında bir dosya adı var; 764 belgenin "
          "enterprise açısından anlamlı ~20'si okundu.</p>"
          + quote("Hoş tesadüf: OpenClaw'ın kendi tehdit modeli dosyasının adı "
                  "<code>THREAT-MODEL-ATLAS.md</code>. Oradaki ATLAS, MITRE'nin "
                  "yapay zekâ tehdit çerçevesi — bizim Atlas'la ilgisi yok. Ama "
                  "<b>biçimi</b> doğrudan kopyalanacak cinsten.", "b")))

slide("VI · atlas", "Tez — ne taşınır, ne taşınmaz",
      fig(f_thesis(), cap_mm=52)
      + cols(
          quote("OpenClaw'dan alınacak şey ajan döngüsü değil, <b>ajanı kuşatan "
                "kontrol düzlemi</b> — ve daha da değerlisi, o düzlemin kendi "
                "sınırlarını dürüstçe beyan etme alışkanlığı.", "g"),
          "<p>İkinci yarı birincisinden kıymetli. OpenClaw'ın belgeleri sürekli şunu "
          "yapıyor: bir mekanizmayı anlatıyor, sonra \"bu şunu <b>kanıtlamaz</b>\" "
          "diye kendi iddiasını daraltıyor.</p>"
          "<p>Bir kredi bürosunda asistanı öldüren şey mekanizmanın olmaması değil — "
          "<b>olduğu sanılan bir mekanizmanın denetim toplantısında çökmesi</b>.</p>"))

slide("VI · atlas", "Üç kontrol ekseni",
      fig(f_three_axes(), cap_mm=56)
      + cols(
          f"<p>{tag('k')} <code>docs/gateway/sandbox-vs-tool-policy-vs-elevated.md</code>. "
          "Üç ayrı soru, üç ayrı mekanizma — ve karıştırılmaları en yaygın "
          "yapılandırma hatası.</p>",
          "<p>Son satır özellikle önemli: <b>\"write tool'unu kapattık, artık "
          "read-only\"</b> cümlesi yanlıştır. <code>exec</code> serbestse shell "
          "üstünden yazma zaten mümkündür. Salt-okunur bir rol istiyorsan "
          "<code>group:runtime</code>'ı da kapatman gerekir.</p>"),
      foot="Ve bir teşhis komutu var: `openclaw sandbox explain --json` efektif politikayı ve düzeltilecek anahtarı basıyor.")

slide("VI · atlas", "Roller tool listesi değil, grup adı",
      cols(
          f"<p>OpenClaw'da <b>13 grup</b> {tag('m')}: <code>group:runtime</code>, "
          "<code>group:fs</code>, <code>group:web</code>, <code>group:memory</code>, "
          "<code>group:sessions</code>, <code>group:agents</code>, "
          "<code>group:media</code>, <code>group:nodes</code>…</p>"
          "<p>Politika bunlarla yazılıyor: <code>allow: [\"group:fs\", "
          "\"group:memory\"]</code>.</p>",
          "<p><b>KKB karşılığı</b> (öneri):</p>"
          "<p><code>group:musteri-verisi</code><br><code>group:kredi-sorgu</code><br>"
          "<code>group:rapor</code><br><code>group:dis-erisim</code></p>"
          "<p>Rol tanımı birkaç grup adı olur, kırk satırlık bir tool listesi "
          "değil.</p>")
      + quote("Kazanç bakımda: yeni bir tool eklendiğinde kırk rol dosyası "
              "güncellenmez. Tool doğru gruba girer, roller kendiliğinden doğru "
              "kalır.", "g"))

slide("VI · atlas", "Onay = dondurulmuş plan",
      fig(f_frozen_plan(), cap_mm=58)
      + cols(
          f"<p>{tag('k')} <code>docs/tools/exec-approvals.md</code>. Naif bir onay "
          "akışında onay ile çalıştırma arasında bir TOCTOU boşluğu vardır: kullanıcı "
          "gördüğü şeyi onaylar, çalışan başka bir şey olabilir.</p>",
          "<p><b>KKB'de somut anlamı:</b> \"şu TCKN için kredi notu sorgula\" onayı "
          "<i>o TCKN'ye</i> bağlanır. Onay alındıktan sonra parametre "
          "değiştirilemez; değiştiyse istek düşer, sessizce yeni parametreyle "
          "koşmaz.</p>"
          f"<p>{tag('m')} Bizim <code>approval.py</code> bugün argüman hash'i "
          "<b>tutmuyor</b>. Açık iş — Kısım VI'nın sonundaki listede ikinci sırada.</p>"))

slide("VI · atlas", "Denetim: iki hat gerekiyor",
      fig(f_two_ledgers(), cap_mm=58)
      + cols(
          f"<p>{tag('k')} OpenClaw'ın denetim kaydı içerik tutmuyor: prompt, mesaj "
          "gövdesi, tool argümanı, URL, komut çıktısı — hiçbiri. Kimlikler "
          "<code>hmac-sha256:v1:&lt;keyId&gt;:&lt;digest&gt;</code> olarak "
          "çıkıyor.</p>",
          quote("Ve kendi sınırını söylüyor: <i>\"Bu korelasyondur, anonimleştirme "
                "değildir — veritabanını okuyabilen anahtarı da okur ve aday ham "
                "kimlikleri pseudonym'lere karşı test edebilir.\"</i>", "b")))

slide("VI · atlas", "Yetki: metot kapsamı yalnızca ilk kapı",
      cols(
          f"<p>8 kapsam var {tag('m')}: <code>read</code>, <code>write</code>, "
          "<code>admin</code>, <code>pairing</code>, <code>approvals</code>, "
          "<code>questions</code>, <code>talk</code>, <code>talk.secrets</code>. "
          "Ama asıl fikir tabloda değil, başlıkta.</p>"
          "<p><b>Kapsam parametreden türetiliyor.</b> <code>agent</code> metodu "
          "normal turlar için <code>write</code>, <code>/reset</code> için "
          "<code>admin</code> istiyor. <code>chat.send</code> write-scoped, ama "
          "içindeki <code>/config set</code> komutu — çağıranın chat kapsamı ne "
          "olursa olsun — <code>admin</code> istiyor.</p>",
          f"<p>{tag('k')} <b>Yetki yükseltme yasak.</b> Bir cihazı onaylamak yalnız "
          "çağıranın <i>zaten sahip olduğu</i> kapsamları basabiliyor.</p>"
          "<p>Ve sessiz genişleme yok: daha geniş rol isteyen bir yeniden bağlanma, "
          "yeni bir <b>bekleyen yükseltme talebi</b> doğuruyor.</p>"
          + quote("Atlas'a taşınacak: \"bu kullanıcı bu metodu çağırabilir mi\" "
                  "yetmez — \"<b>bu parametrelerle</b> çağırabilir mi\" gerekir.",
                  "g")))

slide("VI · atlas", "Dış içerik = veri, talimat değil",
      fig(f_external_content(), cap_mm=56)
      + cols(
          f"<p>{tag('m')} <code>src/security/external-content.ts</code>, 468 satır. "
          "Sayımlar kaynaktan: <b>14</b> şüpheli desen, <b>22</b> özel token "
          "literali, <b>28</b> homoglif eşlemesi.</p>",
          quote("Desen eşleştirme injection'ı <b>engellemiyor</b> — içerik yine "
                "işleniyor, desenler yalnız loglanıyor. Doğru karar: tespit bir "
                "sinyal, bir savunma değil. Savunma sarmalayıcının kendisi.", "b")))

slide("VI · atlas", "Tool Search — büyük katalog, küçük prompt",
      cols(
          "<p>Model bütün tool şemalarını görmüyor. Sınırlı bir <b>yetenek dizini</b> "
          "görüyor, sonra <code>search → describe → call</code> yapıyor.</p>"
          f"<p>{tag('k')} Dizin 18.000 karakterle sınırlı, ada göre sıralı, ve "
          "<b>cache sınırının üstünde</b>. Kullanıcı mesajı, tur-başı tahminler ve "
          "güvenilmeyen MCP metadata'sı dizine <b>girmiyor</b> — girse cache her "
          "turda bozulurdu.</p>",
          f"<p>{tag('k')} Kod köprüsü izole bir Node alt sürecinde: <b>boş "
          "environment, dosya sistemi yok, ağ yok, alt süreç yok</b>. Alt süreç "
          "plugin implementasyonlarını, MCP istemci nesnelerini veya <b>sırları</b> "
          "tutmuyor.</p>"
          "<p>Her gerçek çağrı köprüden Gateway'e geri dönüyor ve normal "
          "politika / onay / hook / log akışından geçiyor.</p>")
      + quote("Ve fail-closed: politika dışı bir tool <b>aramada çıkmıyor</b>. "
              "Gizlemek yetmez; bulunamaz olmalı.", "r"),
      foot="Aynı cache sınırı disiplini Kısım II'deki bağlam slaytında da vardı.")

slide("VI · atlas", "Bellek: güvenlik sınırı yazma yolunda",
      cols(
          quote("Belleğin içerik düzeyinde taranması zehirlenmiş olguları güvenilir "
                "biçimde yakalayamaz — bu yüzden <b>yazma anında köken</b> zorunlu "
                "kılınır ve terfi yapısal olarak kapıya bağlanır.", "g")
          + "<p>Köken sınıfı <b>kapalı bir küme</b> ve SQLite sütununda — modelin "
            "düzyazıyla yazamayacağı bir yerde:</p>"
            "<p><code>owner</code> · <code>agent</code> · <code>untrusted</code> · "
            "<code>system</code></p>",
          "<p>Sınıflandırma muhafazakâr: belirlenemeyen köken dışsalsa "
          "<code>untrusted</code> sayılıyor, <b>asla <code>owner</code> "
          "varsayılmıyor</b>.</p>"
          f"<p>{tag('k')} Ve döngü önleme: bellekten bağlama enjekte edilen içerik "
          "yapısal olarak işaretleniyor ve yeni bellek olarak <b>yeniden "
          "çıkarılmıyor</b>.</p>"
          "<p><i>\"Yüz kez hatırlanan bir olgu tek bir olgu olarak kalır.\"</i></p>")
      + quote("KKB'de karşılığı doğrudan: asistanın bir müşteri kaydından okuduğu "
              "şeyi kalıcı olgu sanıp başka bir bağlamda kullanması. Buna karşı "
              "savunma içerik taraması değil, <b>şemada zorunlu köken sınıfı</b>.",
              "r"))

slide("VI · atlas", "Sırlar ve telemetri",
      cols(
          "<p><b>SecretRef + egress sentinel.</b> Loglar, SDK yapılandırması ve hata "
          "nesneleri gerçek anahtarı değil <code>oc-sent-v1-…</code> görüyor; gerçek "
          "değer istek süreçten çıkmadan hemen önce yerine konuyor.</p>"
          f"<p>{tag('k')} Bilinmeyen sentinel-şekilli bir değer <b>ağ etkinliğinden "
          "önce fail-closed</b>: çözülmemiş sentinel sağlayıcıya iletilmektense "
          "istek hiç gönderilmiyor.</p>"
          f"<p>{tag('k')} Ve göç bir <b>kapı</b>: <code>secrets audit --check</code> "
          "temiz değilse göç bitmemiştir.</p>",
          "<p><b>Telemetri: içerik yok, boyut var.</b> Prompt metni dışa "
          "aktarılmıyor. Ama aktarılanlar zengin:</p>"
          "<p><code>system_prompt_chars</code>, <code>tool_definitions_count</code>, "
          "<code>tool_definitions_chars</code>, <code>request_bytes</code>, "
          "<code>time_to_first_byte_ms</code></p>"
          f"<p>{tag('k')} Yani \"prompt'ta ne kadar tool tanımı vardı\" sorusu "
          "<b>prompt saklanmadan</b> cevaplanıyor.</p>")
      + quote("Bir bankada telemetri sistemine prompt metni akıtmak, veri "
              "sınıflandırma politikasını sessizce delen en yaygın yoldur.", "r"))

slide("VI · atlas", "Dayanıklılık: compaction sonrası nöbetçi",
      cols(
          f"<p>{tag('k')} Rolling döngü tespiti varsayılan <b>kapalı</b> — flagship "
          "modellerde nadiren gerekiyor. Ama compaction sonrası nöbetçi <b>açık</b>.</p>"
          "<p>Kırdığı zincir şu: bağlam taşması → sıkıştırma → aynı döngü → taşma. "
          "Sıkıştırma-yeniden denemesinden sonra kısa bir pencerede aynı "
          "<code>(tool, args, result)</code> üçlüsü tekrarlanırsa koşu iptal "
          "ediliyor.</p>",
          f"<p>{tag('k')} İnce ayar iki yönlü: <code>exec</code> için hash "
          "<b>oynak</b> metadata'yı (süre, PID, cwd) yok sayıyor — yoksa aynı komut "
          "hep farklı görünürdü. Giden mesajlarda ise tersi: oynak id'ler "
          "<b>çıkarılıyor</b> ki iki farklı \"gönderildi\" aynı görünmesin.</p>"
          + quote("İki varsayılanın <b>farklı</b> olması bilinçli: agresif olan "
                  "kapalı, ucuz ve yüksek getirili olan açık.", "g")))

slide("VI · atlas", "Tehdit modeli yaşayan bir belge",
      cols(
          "<p><code>THREAT-MODEL-ATLAS.md</code>, 561 satır, MITRE ATLAS taktiklerine "
          "göre düzenli. Her tehdit için <b>sabit bir tablo şeması</b>:</p>"
          "<p>ATLAS ID · açıklama · saldırı vektörü · etkilenen bileşenler · "
          "<b>mevcut azaltmalar</b> · <b>artık risk</b> · öneriler</p>"
          "<p>Beş güven sınırı ve altı veri akışı diyagramda tanımlı. Katkı için "
          "ayrı bir belge var — yani canlı tutuluyor.</p>",
          f"<p>{tag('k')} Dikkat çeken dürüstlük: bazı tehditlerde "
          "<b>\"Mevcut azaltmalar: Yok\"</b> yazıyor. Bilinen ama kapatılmamış "
          "riskler gizlenmiyor; \"artık risk: düşük\" gerekçesiyle yazılıyor.</p>"
          + quote("Atlas'a taşınacak: bu tablo şeması, ilk gün. KKB'de zaten bir "
                  "güvenlik komitesi vardır ve o komiteye gidilecek belge budur. "
                  "\"Artık risk\" sütununu boş bırakmamak, denetimde en çok işe "
                  "yarayan alışkanlık.", "g")))

slide("VI · atlas", "Yumuşak yönlendirme, sert kontrol değildir",
      "<p class='lead'>Destenin en kısa ve en çok işe yarayan cümlesi, OpenClaw'ın "
      "kendi güvenlik belgesinden:</p>"
      + quote("<b>Sistem prompt'undaki güvenlik kuralları yumuşak yönlendirmedir. "
              "Zorlama; kanal erişim denetimi, tool politikası, sandbox kapsaması "
              "ve — geçerliyse — açık çalıştırma onaylarından gelir.</b>", "r")
      + cols(
          "<p>Yani <code>AGENTS.md</code>'ye “müşteri verisini dışarı gönderme” "
          "yazmak bir <b>kontrol değil</b>. Model çoğu zaman uyar; uymadığı gün "
          "hiçbir şey onu durdurmaz.</p>"
          "<p>Tersi de doğru ve daha az söyleniyor: tool politikası doğru "
          "kurulmuşsa prompt'taki kural zaten gereksizdir.</p>",
          "<p><b>Atlas için pratik ayrım:</b></p>"
          "<p>· prompt'a yazılan → <i>niyet</i><br>"
          "· tool politikası, sandbox, onay, kapı → <i>kontrol</i></p>"
          "<p>Denetim toplantısında gösterilecek olan ikinci liste. Birincisi bir "
          "belge, bir güvence değil.</p>")
      + f"<p>{tag('k')} Ve OpenClaw bunu ekliyor: sandbox <b>opt-in</b>. Varsayılan "
        "olarak kapalı olan bir kontrolün var olması, uygulandığı anlamına "
        "gelmiyor.</p>")

slide("VI · atlas", "ALINMAYACAKLAR",
      table(["ne", "OpenClaw ne diyor", "Atlas'ta neden yetmez"],
            [["<code>operator.read</code> ile veri ayrımı",
              "\"düşmanca çok-kiracılı izolasyon sınırı değildir\"",
              "departmanlar birbirinin sorgusunu görmemeli — gerçek per-user authz gerekir"],
             ["multi-user sahiplik / presence",
              "\"kullanılabilirlik özelliğidir, güvenlik sınırı değil\"",
              "kimlik tool politikasına girmeli, yalnız arayüze değil"],
             ["denetim kaydı", "\"kayıpsız bir uyum arşivi değildir\"",
              "denetçi \"kayıp olabilir\" cevabını kabul etmez"],
             ["exec approvals", "\"per-user auth sınırı değildir\"",
              "kazayı azaltır; kötü niyetli operatörü durdurmaz"],
             ["SecretRef sentinel", "\"süreç izolasyonu değildir\"",
              "gerçek değer aynı süreçte — HSM/vault + süreç ayrımı ayrıca gerekir"],
             ["sandbox <code>docker.binds</code>", "\"docker.sock bağlamak host kontrolünü verir\"",
              "bind'ler sandbox'ı deler; gözden geçirme konusu olmalı"],
             ["161 extension / sohbet kanalları", "—",
              "WhatsApp/Telegram bir kurumsal asistanın yüzeyi değil"]]),
      foot="Kısım VI'nın en önemli slaytı: mekanizmayı kopyalayıp sınırını atlamak, olmayan bir güvenceyi varsaymaktır.")

slide("VI · atlas", "Tek cümlelik ayrım",
      big("Mekanizma taşınır.", "Güven modeli taşınmaz — yeniden kurulur.")
      + cols(
          "<p><b>OpenClaw tek bir güvenilen operatörün etrafında tasarlanmış.</b> "
          "Bütün \"bu bir güvenlik sınırı değildir\" cümleleri buradan geliyor: o "
          "modelde zaten herkes güvenilir, dolayısıyla ayrım bir kolaylık.</p>",
          "<p><b>Atlas çok kullanıcılı ve karşılıklı güvenmeyen departmanlar "
          "içerecek.</b> Aynı cümleler orada birer açık olur.</p>"
          "<p>OpenClaw'ın kendi cevabı da bu: gerçek ayrım gerekiyorsa "
          "<b>ayrı gateway'ler</b> çalıştırın. Atlas'ta bu, gateway çoğaltmak değil, "
          "kimliği kontrol düzlemine gerçekten sokmak demek.</p>"))

slide("VI · atlas", "Atlas'ın şekli",
      fig(f_atlas(), "Alınan mekanizmalar, KKB'ye göre yeniden kurulmuş güven modeliyle.",
          cap_mm=76)
      + "<p>Bellek bu diyagramda ayrı bir kutu değil, <b>her sınırdan geçen bir "
        "sütun</b>: her yazılan olgu köken sınıfını taşır, ve <code>untrusted</code> "
        "olan hiçbir şey terfi edemez.</p>")

slide("VI · atlas", "İlk üç iş",
      table(["#", "iş", "süre", "neden bu sırada"],
            [["1", "Dış içerik sarmalayıcı", "~1 gün",
              "en küçük iş, en büyük tekil koruma — ve bizde <b>hiç yok</b>: "
              "docs_index sonuçları ve dış metinler bağlama düz giriyor"],
             ["2", "Onayı plana bağlama", "~1–2 gün",
              "<code>approval.py</code> argüman hash'i tutmuyor; onay sonrası "
              "değişiklik fark edilmiyor. Testi kolay: onayla, argümanı değiştir, "
              "reddedilmeli"],
             ["3", "İki hatlı kayıt ayrımı", "~2–3 gün",
              "şimdi ucuz, sonra şema göçü. Uyum hattının tek sert kuralı: "
              "yazılamazsa koşu düşer"]])
      + f"<p>{tag('m')} Üçü de bugünkü <code>pipeline/</code>'a prototiplenebilir. "
        "Ardından sırada: cache sınırı disiplini (maliyet kazancı) ve bellek köken "
        "sınıfı (şema kararı — geç kalınırsa pahalı).</p>")

slide("kapanış", "Geriye kalan üç cümle",
      "<p class='lead'>Seksen slayttan sonra taşınmaya değer olanlar:</p>"
      + quote("<b>1.</b> AutoGen'de örtük hiçbir şey yok. \"Kim tetikledi\" "
              "sorusunun cevabı her zaman senin yazdığın satırdır — bu hem gücü "
              "hem yükü.", "b")
      + quote("<b>2.</b> Bir çerçevede en pahalı şey, sessizce makul görünen "
              "varsayılandır. <code>max_tool_iterations=1</code> hata vermez, "
              "sadece yanlış cevap verir.", "r")
      + quote("<b>3.</b> Bir mekanizmanın değeri, ne yaptığı kadar <b>neyi "
              "yapmadığını söylemesindedir</b>. OpenClaw'ın belgeleri bunu yapıyor; "
              "Atlas'ın alması gereken ilk alışkanlık bu.", "g"),
      foot="vc-agent · docs/ · her iddianın kaynağı orada")

slide("kapanış", "Kaynak künyesi",
      cols(
          "<p><b>Belgeler</b></p>"
          "<p><code>docs/05</code> AutoGen core kılavuzu (resmî, birebir)<br>"
          "<code>docs/08</code> AgentChat kılavuzu (resmî, birebir)<br>"
          "<code>docs/06</code> ölçülen tuzaklar<br>"
          "<code>docs/09</code> çerçeve karşılaştırması<br>"
          "<code>docs/13</code> OpenClaw teknik analizi<br>"
          "<code>docs/14</code> protokoller ve farklar<br>"
          "<code>docs/15</code> gateway mimarisi<br>"
          "<code>docs/16</code> enterprise ilham — bu destenin Kısım VI'sı</p>",
          "<p><b>PDF'ler</b></p>"
          "<p><code>agentchat-kilavuzu.pdf</code> — 34 sayfa, bütün yüzey<br>"
          "<code>openclaw-ici.pdf</code> — 12 sayfa, ölçülmüş içi<br>"
          "<code>autogen-openclaw-kilavuzu.pdf</code></p>"
          "<p><b>Kod</b></p>"
          "<p><code>pipeline/</code> — gateway, graph, stages, kapı<br>"
          "<code>docs/diagrams/rough.py</code> — bu şekilleri çizen kalem<br>"
          "<code>docs/diagrams/make_slides.py</code> — bu desteyi üreten script</p>"))
