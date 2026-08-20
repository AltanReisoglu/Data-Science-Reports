"""AutoGen destesinin kuyruğu: halef framework — Microsoft Agent Framework.

`make_hap.py` yüklüyor ve `A` listesine ekliyor. Deste "AutoGen ne veriyor,
ne vermiyor" diye bitiyordu; bu altı slayt o cümlenin devamı: **vermediklerini
halefi ne yaptı.**

### Kaynak disiplini

Buradaki her iddia `[kaynak]` — Microsoft'un kendi belgelerinden okundu,
**koşturulmadı**. `pip install agent-framework` bu makinede zaman aşımına
uğradı, yani AutoGen tarafında yaptığımız gibi ölçemedik. Ayrım slaytlarda
yazılı; ölçülmüş bir sayıyla okunmuş bir cümle aynı tonda sunulmuyor.

Okunanlar (2026-08-19):
* `learn.microsoft.com/agent-framework/migration-guide/from-autogen/`
  (2026-04-01, güncelleme 2026-06-26, 6.070 kelime)
* `.../overview/agent-framework-overview` (güncelleme 2026-08-10)
* `.../concepts/harness` (güncelleme 2026-08-12)
* `.../agents/tools/tool-approval` (güncelleme 2026-08-10)

### Neden destenin sonunda

Çünkü sırası bu. Önce mekanizmayı öğreniyorsun, sonra sınırını görüyorsun,
en sonda o sınırın kapanıp kapanmadığını. Başa alınsaydı geri kalan otuz yedi
slayt "zaten eskimiş bir şeyi anlatıyor" gibi dururdu — ki değil: MAF'ın
kendi geçiş kılavuzu GraphFlow bilenin işinin kolay olacağını yazıyor.
"""

A.append(card(
    "AutoGen · halef", "Microsoft Agent Framework — halef geldi",
    f_layers(),
    "learn.microsoft.com/agent-framework · 1.0 GA Nisan 2026",
    "AutoGen'in README'si nereye gideceğini söylüyor, ve o yer artık gerçek bir "
    "ürün. <b>MAF</b>, AutoGen ile Semantic Kernel'in birleşimi ve aynı ekipler "
    "tarafından yazılıyor. "
    "Belgesinin kendi cümlesi: <i>“Agent Framework, AutoGen'in basit ajan "
    "soyutlamalarını Semantic Kernel'in kurumsal özellikleriyle — oturum tabanlı "
    "durum yönetimi, tip güvenliği, middleware, telemetri — birleştiriyor ve "
    "üstüne graf tabanlı iş akışları ekliyor.”</i> "
    "Dört alanı var: <b>Agents</b> · <b>Harness Agent</b> · <b>Workflows</b> · "
    "<b>Integrations</b>. Üç dil: .NET, Python, Go (Go önizlemede).",
    cap_mm=42,
    foot="Artık [ölçüldü]: MAF 1.14.0 ayrı bir .venv-maf içinde kurulu. Aynı ortama girmiyorlar — pip ikisini on dakikada çözemedi, ve PoC bu yüzden alt süreç kullanıyor."))

A.append(card(
    "AutoGen · halef", "Kılavuzun kendi saydığı dört fark",
    f_components(),
    "migration-guide/from-autogen · “Key Differences”",
    "<b>① Orkestrasyon.</b> AutoGen: olay-güdümlü core + üstünde <code>Team</code>. "
    "MAF: tek bir <b>tipli, graf tabanlı <code>Workflow</code></b> — kenarlar veri "
    "taşıyor, executor <i>girdileri hazır olunca</i> tetikleniyor. "
    "<b>② Tool'lar.</b> <code>FunctionTool</code> yerine <code>@tool</code>; şemayı "
    "kendisi çıkarıyor, üstüne <i>hosted tool</i> geliyor (kod yorumlayıcı, web "
    "arama). "
    "<b>③ Ajan davranışı</b> — bu bizi doğrudan ilgilendiriyor, bir sonraki slayt. "
    "<b>④ Runtime.</b> AutoGen'de gömülü + <i>deneysel dağıtık</i>. MAF bugün "
    "<b>tek süreç</b>; dağıtık yürütme planlanıyor.",
    cap_mm=42,
    foot="Yani dağıtık runtime, AutoGen'in MAF'ta henüz karşılığı olmayan yeteneği."))

A.append(card(
    "AutoGen · halef", "Tool döngüsü: halef varsayılanı değiştirmiş",
    f_tool_loop(),
    "“single-turn unless you increase max_tool_iterations”",
    "Destenin dokuzuncu slaytında anlattığımız tuzağı Microsoft kendi geçiş "
    "kılavuzunda <b>birebir</b> yazıyor: "
    "<i>“<code>AssistantAgent</code> is <b>single-turn</b> unless you increase "
    "<code>max_tool_iterations</code>. <code>Agent</code> is <b>multi-turn by "
    "default</b> and keeps invoking tools until it can return a final answer.”</i> "
    "Yani bu bizim keşfettiğimiz bir gariplik değil, <b>üreticinin de düzeltmeye "
    "değer bulduğu bir tasarım kararı</b>. Halefte varsayılan tersine çevrilmiş: "
    "ajan cevabı verebilene kadar tool çağırmaya devam ediyor. "
    "<b>Ve sayı:</b> AutoGen <code>max_tool_iterations=1</code>, MAF "
    "<code>DEFAULT_MAX_ITERATIONS=40</code> — ikisi de kurulu paketten okundu. "
    "<b>Kırk kat.</b> Yön de ters: AutoGen sessizce <i>az</i> yapıyordu, MAF "
    "sessizce <i>çok</i> yapıyor. Tehlike aynı — varsayılanı yazmadan koşturmak.",
    cap_mm=44,
    foot="[ölçüldü] .venv → autogen_agentchat 0.7.5 · .venv-maf → agent_framework 1.14.0 (_tools.py:95)."))

A.append(card(
    "AutoGen · halef", "“AutoGen Limitations” — üreticinin kendi başlığı",
    f_gotchas(),
    "geçiş kılavuzunda iki kez birebir bu başlık var",
    "<b>İnsan müdahalesi yok.</b> <i>“AutoGen'in <code>Team</code> soyutlaması "
    "başladıktan sonra <b>kesintisiz koşar</b> ve insan girdisi için duraklatmanın "
    "yerleşik bir yolunu sunmaz. Her human-in-the-loop işlevi <b>çerçevenin dışında</b> "
    "özel olarak yazılmalıdır.”</i> "
    "<b>Checkpoint yok.</b> <i>“<code>Team</code> soyutlaması yerleşik checkpoint "
    "yeteneği <b>sunmaz</b>.”</i> "
    "<b>Middleware yok.</b> <i>“Agent Framework, AutoGen'in <b>eksik olduğu</b> "
    "middleware yeteneklerini getiriyor.”</i> "
    "<b>İki modelli olmanın bedeli.</b> Kılavuzun kendi “Challenges” listesi: alt "
    "seviye çoğu kullanıcı için fazla karmaşık, üst seviye karmaşık davranışta "
    "sınırlayıcı, ve ikisi arasında köprü kurmak ek karmaşıklık.",
    cap_mm=40,
    foot="Bu dördü, destenin 18. slaytında “vermediği şey” diye saydığımız listenin üretici tarafından yazılmış hâli."))

A.append(card(
    "AutoGen · halef", "Harness Agent — tanıdık bir liste",
    f_tool_catalog(),
    "concepts/harness · “opinionated, batteries-included”",
    "MAF'ın dört alanından biri doğrudan <b>harness</b>. Belgenin tanımı: "
    "<i>“bir dil modelini iş yapabilen bir ajana çeviren <b>runtime iskelesi</b> — "
    "model ve tool çağrılarını sürer, konuşma durumunu ve bağlamı yönetir, "
    "<b>onay politikalarını uygular</b>.”</i> "
    "Varsayılan açık gelenler: <b>todo takibi</b> · <b>plan ve yürüt modları</b> · "
    "<b>oturum dosya belleği</b> · <b>tool onayı</b> · <b>OpenTelemetry</b> · web "
    "arama. Token sınırı verilirse <b>sıkıştırma</b>. Ayrıca <b>skill'ler</b>, "
    "<b>kabuk çalıştırma</b>, <b>arka plan ajanları</b>. "
    "Bu liste, ikinci destede OpenClaw için saydığımız listenin neredeyse aynısı — "
    "ve artık Microsoft'un paketinde geliyor.",
    cap_mm=44,
    foot="Sınır: create_harness_agent yayında, ama arka plan ajanları / dosya erişimi / looping deneysel, kabuk aracı ön-sürüm."))

A.append(card(
    "AutoGen · halef", "Onay: aynı kelime, farklı disiplin",
    f_frozen_plan(),
    "standing approvals VARSAYILAN AÇIK — bizim tersimiz",
    "MAF'ın onayı çağrının <b>adını ve argümanlarını</b> gösteriyor, yani "
    "isim-bazlı bir onaydan güçlü. Ama harness'ın varsayılan middleware'i "
    "<b>“standing approvals”</b> uyguluyor: <i>“daha önceki kullanıcı "
    "cevaplarından gelen kalıcı onayları uygular.”</i> "
    "Yani varsayılan davranış <b>“bir kez onayla, bir daha sorma”</b>. Bizim "
    "kapımızda onay <b>tüketiliyor</b>: imza <code>(tool, argümanlar)</code> "
    "üstünde ve bir kez geçiyor. "
    "Belgelerde ayrıca OpenClaw'ın <b>donmuş plan</b> disiplininin karşılığı "
    "<b>görünmüyor</b> — onaydan sonra argümanların yeniden doğrulanması, dosya "
    "değiştiyse koşunun reddi. Yok demiyoruz; <b>belgede yazmıyor ve biz "
    "ölçmedik</b>.",
    cap_mm=42,
    foot="Kurumsal fark burada: “bir daha sorma” bir kolaylık kararıdır, ve düzenlenmiş bir kurumda varsayılanı açık olmamalıdır."))

A.append(card(
    "AutoGen · halef", "Hızın faturası — GA'dan sonra 15 kırıcı değişiklik",
    f_gotchas(),
    "support/upgrade/python-2026-significant-changes · 🔴/🟡 işaretleri Microsoft'un",
    "Halef hızlı — ve hızın faturasını <b>üreticinin kendisi yayımlıyor</b>. "
    "Her değişiklik 🔴 <i>kırıcı</i> ya da 🟡 <i>özellik</i> diye işaretli. Sayım: "
    "2026'nın tamamında <b>63 kırıcı</b> / 48 özellik, 25 sürümde. "
    "<b>1.0 GA'dan sonra</b> (2 Nisan → 4 Haziran, 10 sürüm): <b>15 kırıcı</b>. "
    "Örnekler: <code>Message(..., text=...)</code> <b>tamamen kaldırıldı</b> · "
    "telemetri <b>varsayılan açıldı</b> · checkpoint depolaması pickle çözümlemesini "
    "kısıtladı, yani <b>eski checkpoint'ler okunmuyor</b> · beceri API'si "
    "<b>dört kez</b> değişti. "
    "Ve sayfanın kendisi <b>1.8.0'da bitiyor</b>; kurulu sürüm 1.14.0 — değişiklik "
    "rehberi <b>altı sürüm geride</b>.",
    cap_mm=42,
    foot="İki risk, ikisi de gerçek: AutoGen donmuş — düzeltecek kimse yok. MAF hareketli — sabitlemezsen kırılıyorsun, sabitlersen yamayı kaçırıyorsun."))

A.append(card(
    "AutoGen · halef", "“Build your own claw” — Microsoft'un kendi örneği",
    f_tool_catalog(),
    "samples/02-agents/harness/build_your_own_claw · devblogs blog serisi",
    "Microsoft'un harness örneğinin adı <b>“build your own claw”</b>, bir blog "
    "serisiyle geliyor, ve örnek uygulama bir <b>kişisel finans / yatırım "
    "asistanı</b>. İçindekiler: "
    "<b><code>valuation</code></b> ve <b><code>risk-scoring</code></b> adlı iki "
    "<b>beceri</b>, <code>SKILL.md</code> dosyaları olarak · "
    "<code>place_trade</code> tool'u <b><code>approval_mode=\"always_require\"</code></b> "
    "ile · <code>portfolio.csv</code> üzerinde dosya erişimi, salt-okunur araçlar "
    "otomatik onaylı · CodeAct ile portföy hesabı · ticker başına <b>arka plan "
    "ajanlarıyla</b> araştırma. "
    "Yani bu destenin tezini — <i>OpenClaw'ın harness deseni + AutoGen'in motoru</i> — "
    "<b>satıcı aynı alanda örnek olarak yayımlamış</b>.",
    cap_mm=42,
    foot="Bu bir zayıflık değil, en güçlü doğrulama: kurduğumuz şeklin Microsoft'un yol haritasında karşılığı var."))
