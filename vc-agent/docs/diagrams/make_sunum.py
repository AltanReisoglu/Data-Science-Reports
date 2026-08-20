#!/usr/bin/env python
"""Sunum kartı: dört A4 sayfa — AutoGen · MAF · OpenClaw · Atlas.

    python docs/diagrams/make_sunum.py
    → docs/pdf/sunum-dort-sayfa.html  (+ .pdf, 4 sayfa)

`kart-autogen.pdf` ile `kart-openclaw.pdf` **konuşana** bakıyor: sunum sırasında
açılan, göz gezdirilen kartlar. Bu farklı bir okuyucu için: **toplantıdan sonra
masada kalan kâğıt.** Onu okuyan kişi sunumda yoktu, ya da vardı ve üç hafta
geçti, ve kararı o verecek.

O yüzden tasarım kısıtı da farklı:

* **Her sayfa tek bir çerçeve.** Dördüncüsü bizim ne kurduğumuz ve ne istediğimiz.
  Bir sayfayı yırtıp birine verebilmek, dört sayfayı da birlikte vermekten daha
  sık lazım oluyor.
* **Her sayı etiketli.** `[ölçüldü]` bu depoda koşturuldu · `[kaynak]` birincil
  metinden · `[teyitsiz]` okundu, koşturulmadı. Etiketsiz bir sayı, üç hafta
  sonra kimin uydurduğu belli olmayan bir sayıdır.
* **Sayfa sayısı sabit: dört.** Beşinci sayfa okunmaz, o yüzden var olmamalı.

Motor `make_kart.py`'den: aynı geometri, aynı blok tipleri. İki ayrı tasarım,
aynı projede iki ayrı belge gibi okunurdu.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_kart as kart  # noqa: E402 — ortak motor

blok, s, w, sayfa = kart.blok, kart.s, kart.w, kart.sayfa
TARIH = "20 Ağustos 2026"


def tablo(rows: list[tuple[str, str]]) -> str:
    body = "".join(f'<tr><td>{a}</td><td class="n">{b}</td></tr>' for a, b in rows)
    return f"<table>{body}</table>"


# ═════════════════════════════════════════════════════ 1 — AutoGen
P1 = "".join([
    blok(
        "ne olduğu",
        s("Ajanların birbirine <b>mesaj yolladığı</b> bir aktör sistemi — "
          "ve üstünde günlük işi kolaylaştıran bir katman."),
        w("Üç katman: <code>autogen_core</code> aktör modeli · "
          "<code>autogen_agentchat</code> hazır ajan ve beş takım tipi · "
          "<code>autogen_ext</code> dış dünya (model, MCP, Docker). "
          "Ajan bir ajanı çağırmıyor; runtime'a mesaj veriyor."),
        kind="hi"),
    blok(
        "ilk söylenecek sayı",
        s("Aynı görev, dört farklı orkestrasyon: <b>%63,7 fiyat farkı</b>."),
        w("Selector 204 · GraphFlow 270 · RoundRobin 274 · Swarm 334 token. "
          "Ödenen şey zekâ değil <b>yönlendirme özerkliği</b>: ajanlara "
          "“kime devredeceğine sen karar ver” dediğin an fatura artıyor. "
          "<b>[ölçüldü]</b> <code>poc/kiyas.py</code>"),
        tablo([("SelectorGroupChat · model seçer", "204"),
               ("GraphFlow · önceden çizilmiş DAG", "270"),
               ("RoundRobinGroupChat · sırayla", "274"),
               ("Swarm · ajan devrediyor", "334")])),
    blok(
        "en pahalı tuzak",
        s("<code>max_tool_iterations</code> varsayılanı <b>1</b>: ajan tool'u "
          "çağırır, sonucu görür ve <b>durur</b>."),
        w("Zincirleme davranış sessizce imkânsız, ve hata verilmiyor. "
          "Bulduğumuz on üç tuzağın hiçbiri istisna fırlatmadı: sıfır döndü, "
          "boş kaldı, asılı kaldı, ya da hata metnini cevap diye sundu. "
          "<b>[ölçüldü]</b> <code>docs/06</code>"),
        kind="w"),
    blok(
        "sessiz veri kaybı",
        s("Paralel dallardan biri çökerse, <b>bitmiş kardeş sonuçlar da "
          "kayboluyor</b> — alarm yok."),
        w("Çöken bir handler <code>stop_when_idle()</code> bariyerini erken "
          "açıyor. GraphFlow ham hata altında 3 sonuç yerine <b>0–1</b> "
          "döndürüp süre sınırını dolduruyor; core kuyruğu <b>2</b> sonucu "
          "~3 ms'de topluyor. <b>[ölçüldü]</b> <code>compare_fanin.py</code>"),
        kind="w"),
    blok(
        "durumu — ve bunun bedeli",
        s("<b>Bakım modunda.</b> Son sürüm 30 Eylül 2025; <b>323 gün</b>."),
        w("Aynı gün ölçüldü: langgraph 8 gün, crewai 5, google-adk 2, "
          "openai-agents <b>0</b>. Yani rakiplerin hepsi son iki hafta içinde "
          "sürüm çıkardı. Bulacağın hatayı düzeltecek kimse yok. "
          "<b>[ölçüldü]</b> <code>docs/tools/olc_cerceveler.py</code>"),
        kind="w"),
    blok(
        "buna rağmen tek başına tuttuğu şey",
        s("<b>Dağıtık runtime.</b> Ajanları makinelere böl, ajan kodunu "
          "değiştirme."),
        w("LangGraph, CrewAI, Agents SDK, Google ADK — dördünde de yok. "
          "MAF'ta da yok. Ve AutoGen'de de <b>deneysel</b>. Yani bu kutu "
          "herkeste boş; yalnız boşluğun şekli farklı. <b>[ölçüldü]</b>")),
])

# ═════════════════════════════════════════════════════ 2 — MAF
P2 = "".join([
    blok(
        "ne olduğu",
        s("AutoGen'in <b>resmî halefi</b>: AutoGen ile Semantic Kernel'in "
          "birleşimi, aynı ekipler tarafından."),
        w("<code>agent-framework</code> 1.14.0, 1.0 GA Nisan 2026. "
          "Kullanıcı kılavuzu kod deposunda <b>değil</b> — hâlâ Semantic "
          "Kernel'in doküman deposunda duruyor, yani birleşme belgede "
          "tamamlanmamış. <b>[kaynak]</b> <code>docs/20</code>, 177 sayfa"),
        kind="hi"),
    blok(
        "ilk söylenecek sayı",
        s("Tool döngüsü varsayılanı <b>1'e karşı 40</b> — kırk kat."),
        w("Altı çerçevede altı ayrı cevap, hepsi kurulu paketten okundu. "
          "Tehlike iki uçta da aynı: <b>varsayılanı yazmadan koşturmak</b>. "
          "Bir uçta ajan sessizce hiçbir şey yapmıyor, öbür uçta sessizce "
          "durmuyor. <b>[ölçüldü]</b>"),
        tablo([("AutoGen · max_tool_iterations", "1"),
               ("OpenAI Agents SDK · max_turns", "10"),
               ("CrewAI · Agent.max_iter", "25"),
               ("MAF · DEFAULT_MAX_ITERATIONS", "40"),
               ("LangGraph · recursion_limit", "10007"),
               ("Google ADK · LoopAgent", "sınırsız")])),
    blok(
        "getirdikleri",
        s("<b>Middleware</b> · <b>checkpoint</b> · <b>insan döngüde</b> · "
          "<b>harness</b> · <b>FIDES</b> — hiçbirinin AutoGen'de karşılığı yok."),
        w("Kılavuzun kendi cümlesi: <i>“AutoGen'in <code>Team</code> soyutlaması "
          "başladıktan sonra kesintisiz koşar ve insan girdisi için duraklatmanın "
          "yerleşik bir yolunu sunmaz.”</i> Bizim onay kapımız tam olarak o "
          "“çerçeve dışında yazılmış” parça. <b>[kaynak]</b> <code>20:23335</code>")),
    blok(
        "hızın faturası",
        s("1.0 GA'dan sonra <b>iki ayda 15 kırıcı değişiklik</b>."),
        w("Microsoft'un kendi işaretlemesiyle: 2026'nın tamamında 63 kırıcı / "
          "48 özellik, 25 sürümde. <code>Message(text=)</code> tamamen "
          "kaldırıldı, telemetri varsayılan açıldı, checkpoint depolaması "
          "değişti. Ve o rehber <b>1.8.0'da bitiyor</b> — kurulu sürüm 1.14.0. "
          "<b>[kaynak]</b> <code>20:36236</code>"),
        kind="w"),
    blok(
        "olgunluk — kararı bu satır belirliyor",
        s("36 paketin <b>8'i</b> kararlı. Anlatmaya değer her şey "
          "<b>deneysel</b>."),
        w("Harness, FIDES, beceriler, evals — hepsi <code>experimental</code> "
          "ve içe aktarıldığında gerçekten <code>ExperimentalWarning</code> "
          "fırlatıyor. 22 paket <code>beta</code>, 6'sı <code>alpha</code> "
          "(barındırma ailesinin tamamı). <b>[ölçüldü]</b>"),
        kind="w"),
    blok(
        "bizi doğrulayan yer",
        s("Microsoft'un kendi harness örneğinin adı <b>“build your own claw”</b> "
          "— ve örnek uygulama bir <b>yatırım asistanı</b>."),
        w("Becerileri <code>valuation</code> ve <code>risk-scoring</code>, "
          "<code>place_trade</code> tool'u <code>approval_mode=\"always_require\"</code> "
          "ile. Yani kurduğumuz şeklin satıcının yol haritasında karşılığı var. "
          "<b>[kaynak]</b> <code>21:23717</code>"),
        kind="hi"),
])

# ═════════════════════════════════════════════════════ 3 — OpenClaw
P3 = "".join([
    blok(
        "ne olduğu",
        s("Bir <b>harness</b>: dil modelini iş yapabilen bir ajana çeviren "
          "runtime iskelesi. Kütüphane değil, çalışan bir sistem."),
        w("22 paket · 51 tool (44'ü canlı kurulumda) · tek Gateway süreci. "
          "Kontrol düzlemi burada <b>çözülmüş</b>: oturum, onay, denetim, "
          "bellek, zamanlama, kanallar. <b>[ölçüldü]</b> <code>docs/13</code>"),
        kind="hi"),
    blok(
        "kurumsal izleyicinin slaytı — iki kayıt hattı",
        s("<b>Uyum kaydı ile hata ayıklama kaydı aynı şey değildir.</b>"),
        w("Denetim kaydı: değişmez, sıralı, redakte edilmiş, saklama süresi olan. "
          "Hata ayıklama kaydı: ayrıntılı, kısa ömürlü, sırlar taşıyabilir. "
          "Tek hatla ikisini birden yapmaya çalışmak, ikisini de bozar. "
          "Canlı ölçüm: <code>audit.list</code> → <b>100 olay</b> ve devam "
          "imleci. <b>[ölçüldü]</b>")),
    blok(
        "aldığımız karar kuralları",
        s("Onay <b>komuta değil, plana</b> bağlanır. Dış içerik <b>veri</b>, "
          "talimat değil. Yetki <b>metot kapsamı</b> ile başlar ama orada "
          "bitmez."),
        w("Roller bir tool listesi değil, bir <b>grup adı</b>. Bellek yazma "
          "yolunda bir güvenlik sınırı var. Yumuşak yönlendirme sert kontrol "
          "değildir. Bunların hepsi <b>karar kuralı</b> — kod değil, ve "
          "taşınabilir olan kısım bu. <b>[kaynak]</b> <code>docs/16</code>")),
    blok(
        "niş yüzeyler — sorulursa",
        s("Tool call repair · tokenjuice · Lobster · Code Mode · self-learning "
          "· trajectory · failover · döngü kırıcı."),
        w("Canlı ölçüm: <code>commands.list</code> → <b>89 komut</b>, içinde "
          "<code>/steer</code> (koşan tura yön ver), <code>/btw</code> (bağlamı "
          "kirletmeden sor), <code>/goal</code>, "
          "<code>/export-trajectory</code>. Koşan bir tura müdahale etmenin "
          "dört ayrı yolu var. <b>[ölçüldü]</b> <code>hap-openclaw-nis.pdf</code>")),
    blok(
        "almadığımız şey — ve nedeni",
        s("Güven modeli. OpenClaw <b>tek bir güvenilen operatör</b> varsayıyor; "
          "bizde o varsayım geçerli değil."),
        w("Ölçülen kanıt, sistemimizin kendi cümlesi: bir <code>/openclaw</code> "
          "satırı gönderdiğimizde kapımız tuttu ve gerekçeyi kendisi yazdı — "
          "<i>“O ajanın kabuk erişimi var ve şu an onay sormadan çalıştırıyor "
          "(exec: mode=full, ask=off); bizim kapımız içeride ne yapacağını "
          "görmez.”</i> <b>[ölçüldü]</b>"),
        kind="w"),
    blok(
        "üç ayrı ilişki",
        s("AutoGen'i <b>gömüyoruz</b> · OpenClaw'ı <b>öğreniyoruz</b> · "
          "OpenClaw'ı mühendislikte <b>kullanmaya devam ediyoruz</b>."),
        w("Atlas olarak OpenClaw <b>kurmuyoruz</b>. Karar kurallarını "
          "alıyoruz, kodunu değil."),
        kind="hi"),
])

# ═════════════════════════════════════════════════════ 4 — Atlas
P4 = "".join([
    blok(
        "kurduğumuz şey",
        s("Bir VC huni ajanı ve etrafındaki <b>kontrol düzlemi</b> — "
          "16.847 satır, <b>484 test</b> geçiyor."),
        w("Tarama · triyaj · zenginleştirme · risk denetimi · skorlama · memo. "
          "17 tool, üç workbench (yerel + iki MCP), Docker kod yürütücü, "
          "onay kapısı, bağlam motoru, telemetri, akış ekranı."),
        kind="hi"),
    blok(
        "motor değiştirilebilir — ve bu ölçüldü",
        s("54 modülün <b>17'si</b> AutoGen içe aktarıyor. Kodun "
          "<b>%72,5'i</b> altında hangi motorun döndüğünü bilmiyor."),
        w("Ekranın sağ üstündeki düğme AutoGen ile MAF arasında geçiyor: "
          "aynı soru, ikinci bir çerçeve, ve akış ekranında MAF'ın sekiz "
          "mekanizması çiziliyor. “Neden ölü bir çerçeve?” sorusunun cevabı "
          "bu düğme. <b>[ölçüldü]</b>")),
    blok(
        "kapı gerçek — reddedince de çalışıyor",
        s("Kapı bir istisna fırlatmıyor, bir <b>cevap</b> üretiyor."),
        w("Ölçüldü: kod yürütme reddedildiğinde ajan çökmedi, hesabı elle "
          "yapıp cevabı yine verdi ve onayın beklediğini söyledi. Onaylandığında "
          "konteyner <b>2 saniyede</b> koştu ve terminal çıktıyı bastı. "
          "Onay imzası kodun kendisi üstünde: kod değişirse onay tutmuyor."),
        tablo([("kapıdan geçen tool çağrısı", "17"),
               ("onay bir kez tüketiliyor", "evet"),
               ("reddedilince tur çöküyor mu", "hayır"),
               ("konteyner açılma süresi", "2 sn")])),
    blok(
        "söylenecek üç dürüst sınır",
        s("Zamanlayıcı <b>devredilmiş</b> · konteynerin <b>ağı var</b> · "
          "prompt enjeksiyonunu <b>izlemiyoruz</b>."),
        w("Zamanlayıcı OpenClaw'ın cron'una devredildi; yerli karşılığı "
          "<code>gateway/cron.py</code> yazıldı ve testli ama bağlanmadı. "
          "Kod yürütücünün konteyneri izole ama ağ erişimi var — yukarı akış "
          "parametre sunmuyor, ve “sandbox güvenli” cümlesini kurmuyoruz. "
          "Kapımız tool adına ve imzasına bakıyor, verinin nereden geldiğine "
          "değil; deterministik cevabı MAF'ta var, adı <b>FIDES</b>, deneysel."),
        kind="w"),
    blok(
        "istenen karar",
        s("Doksan günlük planın <b>birinci fazı</b> için onay: onay kapısı, "
          "uyum kayıt hattı, ve tek bir dar kullanım. <b>Otuz gün, tek kişi.</b>"),
        w("Bugün bir ürün kararı istemiyoruz. Birinci faz bittiğinde elimizde "
          "ölçülmüş bir şey olacak, ve kalan iki fazın süresini o zaman "
          "konuşabiliriz. Şimdi konuşursak tahmin etmiş oluruz."),
        kind="hi"),
])


if __name__ == "__main__":
    kart.PDF_DIR.mkdir(parents=True, exist_ok=True)
    print("sunum kartı:")
    kart.kart("Atlas — dört sayfa", [
        sayfa("1 · AutoGen", "motor", P1,
              "deste: hap-autogen.pdf · 38 slayt", TARIH),
        sayfa("2 · Microsoft Agent Framework", "halef", P2,
              "docs/20 · 21 · 22 — 177 sayfa kılavuz + 35 tasarım kaydı", TARIH),
        sayfa("3 · OpenClaw", "kontrol düzlemi", P3,
              "deste: hap-openclaw.pdf · 19 + niş 17 slayt", TARIH),
        sayfa("4 · Atlas", "bizde ne var", P4,
              "484 test · docs/23 sunum planı", TARIH),
    ], "sunum-dort-sayfa.html")
