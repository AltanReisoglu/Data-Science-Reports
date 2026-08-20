"""Destenin kıyas bölümü: ölçülmüş tablolar ve soru-cevap.

Bu bölümün tamamı `[ölçüldü]`. Rakip çerçeveler hakkındaki iddialar `docs/09`'da
çoğunlukla `[teyitsiz]`di — okunmuş, koşturulmamış. Burada dördü de **kuruldu**
ve sayılar kurulu paketten okundu; kaynak `docs/tools/olc_cerceveler.py`, ham
sonuç `docs/data/cerceve-olcum.json`.

Bir kez yanlış çıktı ve o yüzden yöntem böyle: CrewAI'da kod yürütmeyi
`crewai_tools` altında aradım, bulamadım, "yok" yazacaktım — oysa
`Agent.allow_code_execution` diye duruyordu. Bulunamayan her şey artık "yok"
değil **"bu adlarla yok"**.

Slaytlar `hap_maf.py`'den sonra yükleniyor: kıyas, halef anlatıldıktan sonra
anlam kazanıyor.
"""

# `A`, `card`, `table` ve `f_*` şekilleri `make_hap.py` tarafından global
# alana konuyor — `exec` ile yükleniyoruz, `import` ile değil.

A.append(card(
    "AutoGen · kıyas", "Beş desenin faturası — aynı görev, tek değişen sıra",
    f_teams(),
    "poc/kiyas.py · aynı görev, aynı ajanlar, yalnız orkestrasyon değişiyor",
    "<b>%63,7 fark.</b> Ödediğin şey zekâ değil, <b>yönlendirme özerkliği</b>. "
    "Ajanlara “kime devredeceğine sen karar ver” dediğin anda fatura bu kadar "
    "artıyor — çünkü her devir bir tur, ve her tur bir model çağrısı."
    + deck.table(
        ["desen", "sırayı kim belirliyor", "mesaj", "LLM", "tool", "token"],
        [["<b>SelectorGroupChat</b>", "model her turda seçiyor", "8", "5", "2",
          "<b>204</b>"],
         ["GraphFlow", "önceden çizilmiş DAG", "11", "7", "3", "270"],
         ["RoundRobinGroupChat", "sırayla, kararsız", "9", "6", "2", "274"],
         ["<b>Swarm</b> (handoff)", "ajanın kendisi devrediyor", "14", "7", "4",
          "<b>334</b>"]])
    + "<p class='hapsay'>Kıyasa çevirisi: <b>Agents SDK'nın tek modeli olan "
      "handoff, AutoGen'in en pahalı desenidir.</b></p>",
    cap_mm=28,
    foot="[ölçüldü] poc/kiyas.py · kiyas_sonuc.json — dört koşu, aynı model, aynı görev."))


A.append(card(
    "AutoGen · kıyas", "Altı çerçeve, altı varsayılan — 1'den sınırsıza",
    "",   # tablo slaytı: altı satır zaten şemanın işini yapıyor
    "hepsi kurulu paketten okundu · docs/data/cerceve-olcum.json",
    "Ajan bir tool çağırdıktan sonra <b>kaç kez daha</b> dönebilir? Hiçbir "
    "çerçeve aynı cevabı vermiyor, ve hiçbiri bunu öne çıkarmıyor."
    + deck.table(
        ["çerçeve", "alan", "varsayılan", "ne demek"],
        [["<b>AutoGen</b>", "<code>max_tool_iterations</code>", "<b>1</b>",
          "tool'u çağırır, sonucu görür, <b>durur</b>"],
         ["OpenAI Agents SDK", "<code>Runner.run(max_turns=)</code>", "10", "orta yol"],
         ["CrewAI", "<code>Agent.max_iter</code>", "25", "cömert"],
         ["<b>MAF</b>", "<code>DEFAULT_MAX_ITERATIONS</code>", "<b>40</b>",
          "AutoGen'in <b>kırk katı</b>"],
         ["LangGraph", "<code>recursion_limit</code>", "10007", "pratikte sınırsız"],
         ["Google ADK", "<code>LoopAgent.max_iterations</code>", "<b>None</b>",
          "gerçekten sınırsız"]])
    + "<p class='hapsay'>Tehlike iki uçta da aynı: <b>varsayılanı yazmadan "
      "koşturmak.</b> Bir uçta ajan sessizce hiçbir şey yapmıyor, öbür uçta "
      "sessizce durmuyor — ikisi de hata vermiyor.</p>",
    cap_mm=24,
    foot="[ölçüldü] 2026-08-19 · langgraph 1.2.11 · crewai 1.15.16 · openai-agents 0.22.0 · google-adk 2.7.1"))


A.append(card(
    "AutoGen · kıyas", "Bakım modu bir söylenti değil — 323 güne karşı 13",
    "",   # tablo slaytı: şekil yer kaplıyor, bir şey anlatmıyor
    "PyPI son sürüm tarihleri · 2026-08-19'da çekildi",
    "“AutoGen bakım modunda” cümlesi bu destede iki kez geçiyor. İşte sayısı: "
    "<b>rakiplerin hepsi son iki hafta içinde sürüm çıkardı, AutoGen on bir ay "
    "önce.</b>"
    + deck.table(
        ["paket", "son sürüm", "tarih", "kaç gün önce"],
        [["<b>autogen-agentchat</b>", "0.7.5", "2025-09-30", "<b>323</b>"],
         ["semantic-kernel", "1.44.1", "2026-08-06", "13"],
         ["langgraph", "1.2.11", "2026-08-11", "8"],
         ["agent-framework (MAF)", "1.14.0", "2026-08-14", "5"],
         ["crewai", "1.15.16", "2026-08-14", "5"],
         ["ag2 (AutoGen forku)", "1.0.2", "2026-08-15", "4"],
         ["google-adk", "2.7.1", "2026-08-17", "2"],
         ["openai-agents", "0.22.0", "2026-08-19", "<b>0</b>"]])
    + "<p class='hapsay'>Tavsiye değil, <b>risk ölçüsü</b>: AutoGen'de hatayı "
      "düzeltecek kimse yok — ama MAF'ta da GA'dan sonra iki ayda <b>15 kırıcı "
      "değişiklik</b> var. İkisi de maliyet.</p>",
    cap_mm=22,
    foot="[ölçüldü] docs/tools/olc_cerceveler.py — sayılar ağdan, tek komutla yeniden üretilebilir."))


A.append(card(
    "AutoGen · kıyas", "Yetenek matrisi — ve AutoGen'in tek başına tuttuğu şey",
    "",   # tablo slaytı: şekil yer kaplıyor, bir şey anlatmıyor
    "dördü de kuruldu ve içe aktarıldı · sembol adıyla doğrulandı",
    "Dört rakip kurulup <b>sembol sembol</b> tarandı. Çoğu yetenek herkeste var; "
    "ilginç olan tek satır en altta."
    + deck.table(
        ["yetenek", "AutoGen", "LangGraph", "CrewAI", "Agents SDK", "ADK"],
        [["graf / akış kurucu", "✔ GraphFlow", "✔ StateGraph", "✔ Flow", "—",
          "✔ SequentialAgent"],
         ["checkpoint", "—", "✔ InMemorySaver", "—", "—", "—"],
         ["insan döngüde", "elle", "✔ interrupt", "elle", "—", "elle"],
         ["oturum", "elle", "✔ InMemoryStore", "—", "✔ SQLiteSession",
          "✔ SessionService"],
         ["kod yürütücü", "✔ Docker", "—", "✔ opt-in", "—", "✔ BuiltIn"],
         ["korkuluk / kapı", "InterventionHandler", "ToolNode", "—",
          "✔ input_guardrail", "callback"],
         ["<b>dağıtık runtime</b>", "<b>✔ gRPC</b>", "<b>✘</b>", "<b>✘</b>",
          "<b>✘</b>", "<b>✘</b>"]])
    + "<p class='hapsay'>Son satır beklenmedik: <b>ajanları makinelere bölmek</b> "
      "yalnız AutoGen'de var — o da deneysel, ve MAF'ta bile yok.</p>",
    cap_mm=22,
    foot="“—” = bu adlarla bulunamadı. Ad tahmini yanlış olabilir; CrewAI'ın kod yürütücüsünü ilk denemede kaçırdık."))


A.append(card(
    "AutoGen · kıyas", "Sorulacak altı soru — ve hazır cevapları",
    "",   # tablo slaytı: şekil yer kaplıyor, bir şey anlatmıyor
    "sunumda gerçekten sorulanlar · cevaplar ölçüme bağlı",
    deck.table(
        ["soru", "cevap"],
        [["<b>“Neden ölü bir çerçeve?”</b>",
          "Kodun <b>%72,5'i</b> motor bilmiyor — 54 modülün 17'si AutoGen içe "
          "aktarıyor. Ekrandaki MAF düğmesi bunun kanıtı."],
         ["<b>“Neden MAF değil?”</b>",
          "Bugün geçmek üç şey kaybettirir: dağıtık runtime yok, önbellek yok, "
          "ve ilginç olan her şey (harness, FIDES, beceriler) <b>experimental</b> "
          "ve gerçekten uyarı fırlatıyor."],
         ["<b>“LangGraph daha iyi değil mi?”</b>",
          "Farklı katman. LangGraph'ın verdiği <b>dayanıklılık</b> (checkpoint, "
          "interrupt); AutoGen'in verdiği <b>eşzamanlılık modeli</b>. “AutoGen mı "
          "LangGraph mı” çoğu zaman yanlış sorulmuş soru."],
         ["<b>“Prompt enjeksiyonu?”</b>",
          "Kapımız tool <b>adına ve imzasına</b> bakıyor, verinin nereden "
          "geldiğini izlemiyor. Deterministik cevabı MAF'ta var, adı <b>FIDES</b>, "
          "ve deneysel. “Bizde de var” diyemem."],
         ["<b>“Üretime hazır mı?”</b>",
          "Hayır, ve öyle sunmuyorum. Kapı gerçek, ölçümler gerçek, 484 test "
          "geçiyor. Eksikler yazılı: zamanlayıcı devredilmiş, konteynerin ağı var."],
         ["<b>“Kaç kişi, ne kadar?”</b>",
          "Birinci faz: <b>otuz gün, tek kişi</b>. Kalan iki fazı şimdi "
          "konuşursak tahmin etmiş oluruz."]]),
    cap_mm=20,
    foot="Bir soruya “bilmiyorum” demek, yanlış cevaptan ucuz. Ölçmediğimiz üç şey slayt 6'da."))


A.append(card(
    "AutoGen · kıyas", "Ölçmediklerimiz — ve neden burada yazıyor",
    "",   # tablo slaytı: şekil yer kaplıyor, bir şey anlatmıyor
    "bu destedeki tek [teyitsiz] listesi",
    "Bu destenin her sayısı ölçüldü. Ölçülmeyenler de <b>sayılabilir</b> "
    "olmalı, yoksa “her şeyi ölçtük” cümlesi kendini çürütür."
    + deck.table(
        ["ne", "durum", "neden ölçmedik"],
        [["MAF'ın veri-akışı fan-in davranışı", "<b>[teyitsiz]</b>",
          "GraphFlow'daki kardeş kaybını ölçtük; MAF'ın modeli üçüncü bir cevap "
          "ve arıza enjeksiyonu koşturulmadı."],
         ["LangGraph/CrewAI'ın arıza davranışı", "<b>[teyitsiz]</b>",
          "Kuruldular ve sembolleri tarandı, ama <b>koşturulmadılar</b>. "
          "“Var” demek, “çalışıyor” demek değil."],
         ["Kod yürütücünün ağ izolasyonu", "<b>bilinen açık</b>",
          "Yukarı akışta parametre yok. Konteyner izole, ama <b>ağı var</b> — "
          "“sandbox güvenli” cümlesini kurmuyoruz."],
         ["Zamanlayıcının yerli hâli", "<b>yazıldı, bağlanmadı</b>",
          "<code>gateway/cron.py</code> 322 satır ve testli, ama hiçbir yerden "
          "çağrılmıyor. Bugün OpenClaw'a devredilmiş."]])
    + "<p class='hapsay'><b>[ölçüldü]</b> koşturuldu · <b>[kaynak]</b> birincil "
      "metinden doğrulandı · <b>[teyitsiz]</b> okundu, koşturulmadı. Üçünü aynı "
      "tonda söylemek, üçünü de değersizleştirir.</p>",
    cap_mm=22,
    foot="Bu slayt destede kalıyor. Bilinen bir sınırı kendin söylemezsen, ilk soruda söyletilirsin."))
