"""Deste 3: OpenClaw harness'ının iç işleyişi — kimsenin listelemediği özellikler.

`make_hap.py` tarafından yükleniyor. Diğer iki deste *kavram* anlatıyor; bu deste
*uygulama* anlatıyor: kod, bozuk bir tool çağrısına ne yapıyor, 40.000 satırlık
bir loga ne yapıyor, onayda duran bir iş akışına ne yapıyor.

Yazım kuralı (2026-08-18'de yeniden yazıldı): her slayt önce **konunun ne
olduğunu** gündelik Türkçeyle söyler, sonra **nasıl çalıştığını**, en sonda
**neden önemli olduğunu**. Terim, tanımlanmadan kullanılmaz. Uzun tire bir
bağlacın yerini tutmaz; "çünkü", "ama", "yani" yazılır.
"""

C: list[str] = [cover(
    "hap deste · 3/3 · niş",
    "OpenClaw harness'ı, içeriden",
    "Bir harness'ın gerçekten ne yaptığı: bozuk tool çağrısını onarmak, "
    "gürültülü çıktıyı kesmek, koşan turu yönlendirmek, düzeltmeyi skill'e "
    "çevirmek. Her slaytta mekanizma ve nasıl uygulandığı.",
    "github.com/openclaw/openclaw @ 01cc7106 · 22 paket · 161 extension<br>"
    "kavramsal anlatım: hap-openclaw.pdf · ölçümler: docs/pdf/openclaw-ici.pdf",
)]

C.append(card(
    "niş · iskelet", "Paket haritası",
    f_packages(),
    "22 paket · çekirdeğin her ilginç parçası ayrı",
    "Bir sistemin nasıl düşünüldüğünü öğrenmenin en hızlı yolu, kodun nasıl "
    "bölündüğüne bakmaktır. OpenClaw 22 pakete bölünmüş. "
    "En büyük üçü şunlar: <b>ai</b> (118 dosya) model sağlayıcılarıyla konuşuyor, "
    "<b>gateway-protocol</b> (108 dosya) kontrol düzleminin tipli şemasını "
    "tutuyor, <b>memory-host-sdk</b> (83 dosya) bellek sağlayıcılarının uyması "
    "gereken sözleşmeyi tanımlıyor. "
    "Bu üçünün ayrı olması bir tercih: model erişimi, kontrol düzlemi ve bellek "
    "<b>birbirinden bağımsız değişebiliyor</b>. Sağlayıcı eklemek kontrol "
    "düzlemine dokunmuyor. "
    "İlginç olan ise en küçükler. <code>tool-call-repair</code> 6 dosya, "
    "<code>retry</code> tek dosya. Ama ikisi de olmadığında sistem çökmüyor, "
    "<b>sessizce yanlış çalışıyor</b> — ki bu daha kötü.",
    cap_mm=44))

C.append(card(
    "niş · tool katmanı", "Built-in tool kataloğu",
    f_tool_catalog(),
    "51 tool · 11 somut grup · src/agents/tool-catalog.ts",
    "Bir ajanın ne yapabildiğini, elindeki tool listesi belirler. OpenClaw'ın "
    "kutudan çıkan listesi <b>51 tool</b>, on bir grupta toplanmış. "
    "Asıl bilgi sayıda değil <b>dağılımda</b>. En kalabalık grup 15 tool'la "
    "<b>sessions</b>: alt-ajan başlatmak, ona iş devretmek, cevabını beklemek, "
    "aralarında mesajlaşmak. Buna karşılık dosya işlemleri için 4, komut "
    "çalıştırmak için 3 tool var. "
    "Yani yatırımın büyük kısmı dosyaya ya da kabuğa değil, <b>ajanların "
    "birbirini yönetmesine</b> yapılmış. Bu da tasarımın niyetini söylüyor: "
    "tek bir asistan değil, <b>çok ajanlı bir işletim ortamı</b> kurmuşlar.",
    cap_mm=54,
    foot="Ayrıca iki meta grup var: group:openclaw (built-in'lerin çoğu) ve group:plugins (eklentiler + MCP sunucuları)."))

C.append(card(
    "niş · tool katmanı", "Üç sayı, üç anlam",
    f_profiles(),
    "51 kaynakta · 44 canlı kurulumda · doküman tablosu eskimiş",
    "\"Kaç tool var?\" sorusuna üç farklı cevap geliyor ve üçü de doğru, çünkü "
    "üçü farklı şeyi sayıyor. "
    "<b>51</b>, kaynak kodda tanımlı tool sayısı. <b>44</b>, bizim ölçtüğümüz "
    "canlı gateway'in gerçekten sunduğu sayı; aradaki fark filtrelerde eriyor. "
    "Dokümandaki tablo ise üçüncü bir sayı veriyor ve o <b>eskimiş</b>: "
    "listede olmayan <code>agents_wait</code>, <code>dashboard</code>, "
    "<code>mobile_ui</code> eksik, buna karşılık artık var olmayan bir "
    "<code>cron</code> tool'u duruyor. "
    "Daralma üç aşamada oluyor. Profil bir taban liste veriyor, "
    "<code>allow</code>/<code>deny</code> onu kesiyor, sandbox'tayken sandbox "
    "politikası bir kez daha kesiyor. Üçü de yalnız <b>daraltıyor</b>; hiçbiri "
    "listeye tool ekleyemiyor.",
    cap_mm=44,
    foot="Sonuç: bir kurulumun gerçek tool yüzeyini öğrenmenin tek yolu, çalışan sisteme sormaktır."))

C.append(card(
    "niş · tool katmanı", "Tool Call Repair — bozuk çağrıyı kurtarmak",
    f_repair(),
    "packages/tool-call-repair — ayrıştır → ada eşle → gerçek çağrıya çevir",
    "Modelin bir tool çağırmasının doğru yolu, sağlayıcının function calling "
    "biçimini kullanmaktır. Ama bazı modeller bunu beceremiyor: tool çağrısını "
    "cevabın içine <b>düz yazı olarak</b> yazıyorlar. "
    "OpenClaw bu düz yazıyı tanıyıp gerçek çağrıya çeviriyor. Dört farklı biçim "
    "tanınıyor: <code>[END_TOOL_REQUEST]</code> bloğu, Harmony'nin "
    "<code>&lt;|channel|&gt;</code> / <code>&lt;|message|&gt;</code> / "
    "<code>&lt;|call|&gt;</code> işaretçileri, ve XML'e benzeyen "
    "<code>&lt;function=ad&gt;</code> etiketleri. "
    "Asıl kritik adım ikincisi: modelin yazdığı ad, sağlayıcının <i>o istekte</i> "
    "izin verdiği tam tool adına eşleniyor. Yakın ama birebir olmayan bir ad "
    "işe yaramaz. "
    "Onarmazsan tur boşa gider ve <b>hiçbir yerde sebebi görünmez</b>.",
    cap_mm=44,
    foot="Altı dosyalık küçük bir paket. Küçük ya da yerel modellerle çalışıyorsan turların bir kısmını bu kurtarıyor."))

C.append(card(
    "niş · tool katmanı", "Tool sonucu ara katmanı",
    f_result_middleware(),
    "tokenjuice — komuta değil, SONUCA dokunuyor",
    "Bir <code>exec</code> çağrısı 40.000 satır log döndürebilir. O log olduğu "
    "gibi bağlama girerse tek bir komut bütün pencereyi yer. "
    "Tokenjuice bu sorunu, komut <b>zaten koştuktan sonra</b> çözüyor: çıktıyı "
    "kısaltıyor. "
    "Hangi katmanda durduğu tasarımın kendisi. Kabuğa giden komutu yeniden "
    "yazmıyor, komutu tekrar koşturmuyor, çıkış kodunu değiştirmiyor. Yalnızca "
    "modele dönen <code>tool_result</code>'a dokunuyor. "
    "Sebebi şu: komutu değiştirseydin sonucun <i>doğruluğuna</i> karışmış "
    "olurdun. Sonucu kısaltmak ise yalnızca <b>bağlam muhasebesi</b> — ne "
    "koştuğu değişmiyor, modelin ne kadarını gördüğü değişiyor.",
    cap_mm=44))

C.append(card(
    "niş · iş akışı", "Lobster — tipli iş akışı runtime'ı",
    f_lobster(),
    "tek tool çağrısı · gömülü onay kapıları · devam token'ı",
    "Çok adımlı bir işi modele orkestra ettirirsen her adım ayrı bir tur olur. "
    "Dört adımlı bir iş, dört model turu demektir; her turda bütün bağlam "
    "yeniden gönderilir. "
    "Lobster bu orkestrasyonu modelden alıp <b>tipli bir runtime'a</b> veriyor. "
    "Model tek bir çağrı yapıyor, runtime bütün boru hattını koşturuyor, geriye "
    "tek bir yapılandırılmış sonuç dönüyor. "
    "İçinde onay kapıları da var. Bir adım yan etkiliyse (mesaj gönder, yayınla, "
    "sil) akış <b>orada duruyor</b> ve bir devam token'ı döndürüyor. Onayladıktan "
    "sonra baştan başlamıyorsun, kaldığı yerden devam ediyor. "
    "Önemli olan şu: <b>kapıyı runtime tutuyor, model değil.</b> Model \"onay "
    "sormayı atlayayım\" diye karar veremiyor.",
    cap_mm=42,
    foot="Task Flow'un bir katman altında duruyor: Lobster tek çağrının içini, Task Flow ayrı işler arasını yönetiyor."))

C.append(card(
    "niş · eşzamanlılık", "Code Mode ve Swarm",
    f_tool_search(),
    "openclaw.tools.* köprüsü — izole alt süreç, sır yok",
    "Katalog büyüdükçe bütün tool şemalarını prompt'a koymak imkânsızlaşıyor. "
    "<b>Code Mode</b> bunu şöyle çözüyor: model tool şemalarını görmüyor, bunun "
    "yerine küçük bir JavaScript köprüsüne <code>search</code>, "
    "<code>describe</code>, <code>call</code> yazıyor. "
    "<b>Swarm</b> aynı köprüden eşzamanlı alt-ajanlar başlatıyor ve sonuçlarını "
    "yapılandırılmış biçimde topluyor. "
    "İkisinin güvenlik hikâyesi aynı ve dikkat çekici. Köprü, izole bir Node alt "
    "sürecinde koşuyor: <b>environment boş, dosya sistemi yok, ağ yok, eklenti "
    "kodu yok, sır yok</b>. Yani köprüde koşan kod tek başına hiçbir şey "
    "yapamıyor. "
    "Gerçek her çağrı Gateway'e geri dönüyor ve normal politika, onay, hook ve "
    "log yolundan geçiyor.",
    cap_mm=42))

C.append(card(
    "niş · oturum", "Koşan bir tura müdahale etmenin dört yolu",
    f_session_tools(),
    "/steer · /btw · /goal · /loop",
    "Sıradan bir sohbet arayüzünde mesajı gönderdikten sonra yapabileceğin tek "
    "şey beklemektir. Bu dört komut aynı soruya farklı cevaplar veriyor: "
    "<b>tur çoktan başlamışken ne yapabilirsin?</b> "
    "<code>/steer</code> koşan turu yönlendiriyor. Runtime müdahaleyi kabul "
    "etmezse mesajı çöpe atmıyor, sıradan bir prompt olarak gönderiyor. "
    "<code>/btw</code> araya bir yan soru sokuyor ve cevabı <b>konuşma geçmişine "
    "eklemiyor</b>, yani asıl işin bağlamını kirletmiyor. "
    "<code>/goal</code> oturuma kalıcı bir hedef bağlıyor; hem operatör hem model "
    "aynı hedefi görüyor. <code>/loop</code> ise konuşmaya bağlı, kendini "
    "tekrarlayan bir iş kuruyor.",
    cap_mm=42))

C.append(card(
    "niş · öğrenme", "Self-learning — düzeltmeyi skill'e çevirmek",
    f_self_learning(),
    "Skill Workshop: öner → tara → uygula → yaşam döngüsü",
    "Bir asistana \"öyle değil, böyle yapacaksın\" dediğinde o bilgi nereye "
    "gidiyor? "
    "Akla ilk gelen cevap belleğe yazmak. Ama bellekteki bir satır bir "
    "<i>olgu</i>dur; sonraki oturumun izleyeceği bir <i>prosedür</i> değildir. "
    "\"Kullanıcı tabloları sever\" ile \"rapor yazarken önce şunu, sonra şunu "
    "yap\" aynı şey değil. "
    "OpenClaw'ın cevabı şu: kalıcı birim <b>skill</b>. Bir düzeltme ya da "
    "başarıyla biten bir iş, Skill Workshop'un yönetilen yolundan geçip yeniden "
    "kullanılabilir bir prosedüre dönüşüyor. Elle yazılan skill'ler de "
    "<b>aynı</b> yoldan geçiyor. "
    "Kritik olan aradaki kapı: öğrenilen şey doğrudan davranışa yazılmıyor. Önce "
    "öneriliyor, sonra taranıyor, sonra uygulanıyor.",
    cap_mm=42))

C.append(card(
    "niş · hata ayıklama", "Trajectory — oturumun uçuş kayıt cihazı",
    f_trajectory(),
    "/export-trajectory → redakte edilmiş destek paketi",
    "\"Ajan neden bunu yaptı?\" sorusunun cevabı normalde elde yoktur. Modele "
    "tam olarak ne gittiğini görmek için prompt'u elle yeniden kurman gerekir. "
    "Trajectory bunu baştan kaydediyor. Her ajan koşusu için yapılandırılmış bir "
    "zaman çizelgesi tutuluyor: modele giden prompt ve sistem prompt'u, "
    "gönderilen tool tanımları, tool çağrı ve sonuç zinciri, süreler ve hatalar. "
    "<code>/export-trajectory</code> bu kaydı tek komutla bir destek paketine "
    "çeviriyor. "
    "Ve <b>redaksiyon varsayılan</b>. Bu küçük bir ayrıntı gibi duruyor ama "
    "değil: hata raporu paylaşmanın sır paylaşmak anlamına gelmemesi, insanların "
    "gerçekten rapor göndermesini sağlayan şey.",
    cap_mm=44))

C.append(card(
    "niş · dayanıklılık", "Model failover",
    f_failover(),
    "profil rotasyonu · cooldown · auth-hata önbelleği · oturum yapışkanlığı",
    "Bir model sağlayıcısı düştüğünde ne oluyor? Sırayla başka profiller "
    "deneniyor. Ama körlemesine değil, üç frenle. "
    "<b>Cooldown</b>, az önce düşen profili bir süre atlıyor. <b>Auth-hata "
    "önbelleği</b>, aynı 401'i tekrar tekrar yemeyi engelliyor. <b>Faturalama "
    "kilidi</b>, kotası bitmiş profili devre dışı bırakıyor. "
    "En ilginç ayrıntı sonuncusu: <b>oturum yapışkanlığı</b>. Aynı oturum, "
    "mümkün olduğunca aynı profilde kalmaya çalışıyor ve gerekçesi belgede "
    "açıkça yazılı — <i>cache-friendly</i>. "
    "Model değiştirmek prompt cache'ini yakıyor, çünkü cache isteğin başındaki "
    "değişmeyen kısımdan çalışıyor ve model değişince o kısım da değişiyor. "
    "Yani gereksiz yere model değiştirmemek bir <b>maliyet</b> kararı.",
    cap_mm=42))

C.append(card(
    "niş · dayanıklılık", "Döngü kırıcı ve sıkıştırma sonrası nöbetçi",
    f_loopguard(),
    "aynı (tool, argüman, sonuç) üçlüsü tekrarlanırsa → koşu iptal",
    "En pahalı hata sonsuz döngüdür, ve en sinsi biçimi şu zincirdir: bağlam "
    "doluyor, sıkıştırma çalışıyor, model sıkıştırma yüzünden ne yaptığını "
    "unutup aynı tool'u aynı argümanla tekrar çağırıyor, bağlam yine doluyor. "
    "Nöbetçi, sıkıştırmadan hemen sonra kısa bir pencere açıyor. O pencerede "
    "aynı <code>(tool, argüman, sonuç)</code> üçlüsü tekrarlanırsa koşuyu iptal "
    "ediyor. "
    "İnce ayar iki yönlü ve ikisi de gerekli. <code>exec</code> için hash "
    "hesaplanırken <b>oynak</b> alanlar (süre, PID, çalışma dizini) dışarıda "
    "bırakılıyor, yoksa aynı komut her seferinde farklı görünür ve döngü hiç "
    "yakalanmaz. Giden mesajlarda ise tam tersi yapılıyor: oynak id'ler "
    "çıkarılıyor ki iki ayrı gönderim yanlışlıkla aynı sanılmasın.",
    cap_mm=40,
    foot="İki varsayılan bilerek farklı: sürekli döngü tespiti KAPALI, sıkıştırma sonrası nöbetçi AÇIK."))

C.append(card(
    "niş · bağlam", "Bağlam motoru dört noktada değiştirilebilir",
    f_ctx_engine(),
    "eklenti sözleşmesi — sıkıştırmanın kendisi bile takılabilir",
    "Çoğu harness'ta \"modele ne gönderilecek\" kararı çekirdeğe gömülüdür ve "
    "dışarıdan değiştirilemez. "
    "OpenClaw'da bu kararın verildiği dört an da eklenti yüzeyi: mesaj bağlama "
    "eklenirken, model koşusundan hemen önce, pencere dolduğunda, ve tur "
    "bittiğinde. "
    "Sıkıştırma bile devralınabiliyor. Bir eklenti <code>ownsCompaction</code> "
    "bayrağını kaldırırsa sıkıştırmanın tamamı ona geçiyor. "
    "Bütün bu esnekliğin altında pazarlığa açık olmayan tek bir kural duruyor: "
    "<b>tool çağrısı sonucundan ayrılmaz.</b> Ayrılırsa model, cevabını hiç "
    "görmediği bir çağrı yapmış gibi görünür, ve çoğu sağlayıcı bu şekli "
    "geçersiz sayıp isteği reddeder.",
    cap_mm=42))

C.append(card(
    "niş · bellek", "Bellek bir sağlayıcı sözleşmesi",
    f_memory_write(),
    "memory-host-sdk · 83 dosya · builtin / honcho / qmd",
    "OpenClaw'da bellek tek bir uygulama değil. Bir <b>sözleşme</b> var, ve o "
    "sözleşmeyi uygulayan birden çok sağlayıcı: yerleşik olan, Honcho, QMD. "
    "Bunun için 83 dosyalık ayrı bir SDK paketi ayrılmış olması, işi ne kadar "
    "ciddiye aldıklarını gösteriyor. "
    "Sözleşmenin değişmeyen kısmı şu: her bellek kaydının bir <b>köken</b> "
    "sınıfı var, bu sınıf kapalı bir kümeden seçiliyor ve SQLite'ta ayrı bir "
    "sütunda duruyor. Yani <b>model onu düzyazıyla yazamıyor</b>. Sağlayıcı "
    "değişse de bu kural değişmiyor. "
    "Sonuç önemli: \"belleği değiştirebilirsin\" ile \"güvenlik sınırını "
    "gevşetebilirsin\" aynı şey değil. Genişletme noktası, sınırın <i>üstünde</i> "
    "duruyor.",
    cap_mm=42))

C.append(card(
    "niş · çıktı", "Sessiz altyapı",
    f_packages(),
    "normalization-core · markdown-core · terminal-core · retry · net-policy",
    "Bir harness'ın görünmeyen yarısı bu beş pakette. Hiçbiri bir özellik "
    "listesinde yer almaz, ama biri eksik olduğunda hemen fark edilir. "
    "<b>normalization-core</b> farklı kanallardan gelen içeriği tek biçime "
    "indiriyor. <b>markdown-core</b> modelin yazdığını her kanalın "
    "kaldırabileceği biçime çeviriyor, çünkü WhatsApp'ın markdown'ı Slack'inki "
    "değil. <b>terminal-core</b> TUI çıktısını üretiyor. <b>retry</b> yeniden "
    "deneme politikasını tek dosyada topluyor. <b>net-policy</b> IP "
    "ayrıştırması, SSRF engellemesi ve URL redaksiyonu yapıyor. "
    "Sonuncusu iyi bir örnek: <code>net-policy</code> olmadan "
    "<code>web_fetch</code>, modelin eline verilmiş bir <b>iç ağ tarayıcısına</b> "
    "dönüşüyor.",
    cap_mm=42))

C.append(card(
    "niş", "Bizim harness'ımıza ne taşınır",
    f_repair(),
    "sıra: onarım → sonuç kısaltma → müdahale → trajectory",
    "Bu destedeki mekanizmalardan dördü bugün <code>pipeline/</code>'a girebilir, "
    "ve dördü de küçük iş. "
    "<b>Tool call repair</b> — küçük ya da yerel bir model kullanacaksak boşa "
    "giden turların bir kısmını kurtarır. <b>Sonuç ara katmanı</b> — bir "
    "<code>exec</code> çıktısının bağlamı yemesini engeller, ve zaten bir "
    "workbench sarmalayıcımız olduğu için tam oraya oturur. <b>Steer ve btw</b> — "
    "uzun koşularda kullanıcıyı beklemekten kurtarır. <b>Trajectory</b> — panel "
    "zaten aşama yayınlıyor, onları tek bir pakete çevirmek küçük bir ek. "
    "Lobster ve self-learning ise daha büyük kararlar. İkisi de kontrol "
    "düzleminin olgunlaşmasını bekliyor.",
    cap_mm=42,
    foot="Ayrıntı: docs/16 (ne alınır, ne alınmaz) · docs/18 (task manager ve dayanıklılık)"))
