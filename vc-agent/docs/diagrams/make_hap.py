#!/usr/bin/env python
"""Two condensed decks: one diagram per slide, under twenty slides each.

    python docs/diagrams/make_hap.py
    → docs/pdf/hap-autogen.html   (+ .pdf)
    → docs/pdf/hap-openclaw.html  (+ .pdf)

`slaytlar.pdf` is 94 slides and answers "tell me everything". These two answer a
different question: **show me the mechanism and tell me what it means**. So every
slide is a figure plus the two or three sentences that make the figure worth
looking at — no tables of options, no citation lists, no code.

The constraint that shapes them is the one-diagram rule. If an idea cannot be
drawn, it does not get a slide here; it stays in the long deck. That is why these
skip several genuinely important things (the funnel, the four principles, the
enterprise decision tables) — they are prose, and prose belongs elsewhere.

The deck engine — page geometry, the viewer script, print CSS — is imported from
`make_slides.py`, so all three decks stay one design rather than three that drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_slides as deck  # noqa: E402 — the shared engine
from figures import (  # noqa: E402
    f_actor, f_context, f_ctx_engine, f_durable, f_external_content, f_fanout,
    f_frozen_plan, f_gotchas, f_graphflow, f_identity, f_intervention, f_layers,
    f_memory_tiers, f_memory_write, f_message_types, f_oc_arch, f_scopes,
    f_secrets, f_send_vs_publish, f_skill_disclosure, f_task_axes,
    f_task_lifecycle, f_task_stack, f_teams, f_termination, f_threads,
    f_three_axes, f_tool_loop, f_tool_search, f_topic, f_two_ledgers,
    f_failover, f_loopguard, f_packages, f_repair, f_result_middleware,
    f_lobster, f_self_learning, f_session_tools, f_trajectory,
    f_profiles, f_tool_catalog, f_components, f_patterns,
    f_code_executors, f_codeexec_pattern, f_component_config, f_debate,
    f_groupchat, f_handoffs, f_mixture, f_model_clients, f_reflection,
    f_sequential, f_tools_component, f_workbench_component,
    f_custom_agent, f_magentic, f_memory_rag, f_serialize_agentchat,
    f_tracing,
)

PDF_DIR = Path(__file__).resolve().parent.parent / "pdf"


# ────────────────────────────────────────────────────────────────── the shell
#
# The long deck's CSS with one block added: these slides are a figure with text
# under it, so the figure gets the room and the text gets a wider measure.

EXTRA_CSS = """
.hapfig{margin:0 0 5mm}
.hapfig svg{width:100%;height:auto}
.hapsay{font-size:13.6pt;line-height:1.5;max-width:250mm}
.hapsay b{color:var(--ink)}
.hapkey{font-family:var(--mono);font-size:10.4pt;color:var(--ochre);
        letter-spacing:-.01em;margin:0 0 2.5mm}
.slide h2{font-size:21pt}
"""


def build_deck(slides: list[str], title: str, out_name: str) -> Path:
    """Render one deck. `slides` is already-rendered HTML sections."""
    body = "".join(slides)
    html = (
        '<!doctype html>\n<html lang="tr"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f"<style>{deck.CSS}{EXTRA_CSS}</style></head><body>\n"
        f'<div id="deck">{body}</div>\n'
        '<div id="bar"></div><div id="cnt"></div>'
        '<div id="help">← → gez · yazdır: Ctrl+P</div>\n'
        f"<script>{deck.JS}</script>\n</body></html>\n"
    )
    path = PDF_DIR / out_name
    path.write_text(html, encoding="utf-8")
    print(f"  {path.name}  ·  {len(slides)} slayt  ·  "
          f"{html.count('<svg')} şekil  ·  {len(html)/1024:.0f} KB")
    return path


def cover(eyebrow: str, title: str, sub: str, meta: str) -> str:
    return (
        '<section class="slide cover">'
        f'<div class="ceyebrow">{eyebrow}</div><h1>{title}</h1>'
        f'<p class="csub">{sub}</p><div class="cmeta">{meta}</div></section>'
    )


def card(part: str, title: str, svg: str, key: str, say: str, foot: str = "",
         *, cap_mm: float = 52.0) -> str:
    """One slide: a heading, the figure, the line that matters, then the point.

    `cap_mm` is the height the drawing may claim. A figure left at full width
    renders about 80 mm tall on a 167 mm slide and pushes the explanation off
    the bottom — which defeats the whole point of a deck whose text is the
    payload. Width is therefore derived from the drawing's own aspect ratio so
    its *height* lands on the cap.
    """
    import re

    box_mm = 250.0
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    if m:
        w, h = float(m.group(1)), float(m.group(2))
        box_mm = min(250.0, cap_mm * w / h)
    head = f'<div class="hd"><span>{part}</span><span class="n"></span></div>'
    ft = f'<div class="ft">{foot}</div>' if foot else ""
    return (
        f'<section class="slide">{head}<h2>{title}</h2><div class="bd">'
        f'<div class="hapfig" style="max-width:{box_mm:.0f}mm;margin:0 auto 5mm">{svg}</div>'
        f'<p class="hapkey">{key}</p>'
        f'<p class="hapsay">{say}</p>'
        f"</div>{ft}</section>"
    )


# ══════════════════════════════════════════════════════ DESTE 1 — AutoGen
#
# Yazım kuralı (2026-08-18): önce **ne olduğu** gündelik Türkçeyle, sonra
# **nasıl çalıştığı**, en sonda **neden umursaman gerektiği**. Terim
# tanımlanmadan kullanılmıyor; uzun tire bir bağlacın yerini tutmuyor.

A: list[str] = [cover(
    "hap deste · 1/2",
    "AutoGen — ve halefi",
    "Aktör modelinden tool döngüsüne: her slaytta bir mekanizma, nasıl çalıştığı "
    "ve neden önemli olduğu. Son altı slayt, AutoGen'in vermediklerini halefi "
    "Microsoft Agent Framework'ün ne yaptığı.",
    "autogen-core · agentchat · ext v0.7.5  ·  MAF 1.0 GA Nisan 2026"
    "<br>uzun anlatım: docs/pdf/slaytlar.pdf",
)]

A.append(card(
    "AutoGen", "Üç katman",
    f_layers(),
    "autogen_core → autogen_agentchat → autogen_ext",
    "AutoGen tek bir kütüphane değil, üst üste duran üç katman. "
    "En alttaki <code>autogen_core</code> <b>aktör modelini</b> kuruyor: ajanlar "
    "birbirini doğrudan çağırmıyor, runtime'a mesaj veriyor. Ortadaki "
    "<code>autogen_agentchat</code> günlük işi kolaylaştırıyor: hazır ajan, beş "
    "takım tipi, on bir sonlandırma koşulu. Üstteki <code>autogen_ext</code> dış "
    "dünyaya bakıyor: model istemcileri, MCP, kod yürütücüler. "
    "Kural basit ve pratik: <b>yukarıdan başla.</b> AgentChat'in zaten çözdüğü "
    "bir problemi core'da yeniden çözmek, aynı işi daha az testle yapmak "
    "demektir.",
    foot="Aşağı inmek her zaman mümkün, ve gerektiğinde iniyoruz — fan-in ölçümümüz core katmanında yazıldı."))

A.append(card(
    "AutoGen · core", "Aktör modeli",
    f_actor(),
    "runtime.publish_message(msg, topic)  —  ajan ajanı çağırmaz",
    "Bir ajan, başka bir ajanın nesnesini elinde tutmuyor ve metodunu "
    "çağırmıyor. Runtime'a bir mesaj veriyor, teslimatı runtime yapıyor. "
    "Bunun iki bedeli var. Araya fazladan bir katman giriyor. Ve bir şey ters "
    "gittiğinde \"kim kimi çağırdı\" sorusunun cevabı yığın izinde görünmüyor, "
    "çünkü ortada bir çağrı zinciri yok. "
    "Karşılığında üç şey kazanıyorsun. Sisteme yeni bir ajan eklemek, çağıran "
    "kodu <b>hiç değiştirmiyor</b>. Bütün mesajlar tek bir noktadan geçtiği için "
    "müdahale ve ölçüm oraya bir kez takılıyor. Ve aynı sınıftan istediğin kadar "
    "örnek doğurmak bedava geliyor.",
    cap_mm=48))

A.append(card(
    "AutoGen · core", "Kimlik: bir şey değil, iki şey",
    f_identity(),
    "AgentId = (type, key)  —  davranış ve örnek",
    "Her ajanın kimliği iki parçadan oluşuyor. <b>type</b> hangi sınıf olduğunu "
    "söylüyor; <b>key</b> ise o sınıfın hangi kopyası olduğunu. "
    "Fark önemli, çünkü runtime'a kaydettiğin şey yalnızca <b>type</b>. Key "
    "kaydedilmiyor; ilk kez o key'e mesaj gittiğinde örnek kendiliğinden "
    "doğuyor. "
    "Yani \"üç ajan kaydettim\" yanlış bir cümle. <b>Bir tip</b> kaydettin, üç "
    "örnek doğdu. "
    "Günlük hayatta bunun tek bir sonucu var ve onu bilmek yeter: <b>durumu key "
    "taşıyor.</b> Aynı key'e ikinci kez mesaj gönderirsen aynı belleğe gidiyor. "
    "Farklı bir key yazarsan sıfırdan, hiçbir şey hatırlamayan yeni bir ajan "
    "alıyorsun.",
    cap_mm=48))

A.append(card(
    "AutoGen · core", "İki iletişim biçimi, iki asimetri",
    f_send_vs_publish(),
    "send → cevap döner, hata fırlar  ·  publish → None döner, hata loglanır",
    "<code>send_message</code> sıradan bir fonksiyon çağrısı gibi davranıyor: "
    "tek bir alıcı var, dönüş değeri var, ve içeride bir hata olursa hata sana "
    "kadar geliyor. "
    "<code>publish_message</code> ise bir duyuru. Kaç dinleyici olduğunu "
    "bilmiyorsun, dönüş değeri yok, ve <b>handler içinde patlayan istisna sana "
    "ulaşmıyor</b>; yalnızca loglanıyor. "
    "Üstelik hiç abone olmaması da hata sayılmıyor. Yayın başarılı kabul "
    "ediliyor. Yani \"aboneliği yazmayı unuttum\" hatası da sessiz kalıyor. "
    "Bu iki asimetri birleştiğinde ortaya şu tablo çıkıyor: bir dal sessizce "
    "ölüyor, ve onu bekleyen toplayıcı sonsuza kadar bekliyor.",
    cap_mm=48))

A.append(card(
    "AutoGen · core", "Topic: kanal adı değil, iki parçalı adres",
    f_topic(),
    "topic.source  →  agent.key      (core kılavuzu 05:670)",
    "Topic da iki parçalı. <b>type</b> mesajın ne olduğunu söylüyor, "
    "<b>source</b> hangi iş için olduğunu. "
    "Abonelik <i>type</i>'a yapılıyor. Ama mesajın hangi <i>örneğe</i> "
    "teslim edileceğini <b>source</b> belirliyor, hiçbir dönüşüm olmadan: "
    "<code>topic.source</code> doğrudan <code>agent.key</code> oluyor. "
    "Bunun sonucu, farkında olmazsan sinsi. Source'u her istekte "
    "değiştirirsen her istekte <b>yepyeni bir ajan örneği</b> doğuyor. Önceki "
    "örneğin belleği silinmiyor, sadece <i>ulaşılamaz</i> hale geliyor. "
    "Ve bu daha kötü, çünkü hiçbir hata çıkmıyor: sistem çalışıyor, ajanlar "
    "cevap veriyor, ama hiçbiri bir öncekini hatırlamıyor.",
    cap_mm=48,
    foot="Doğru kullanım: source = iş kimliği. Aynı işin bütün adımları aynı source'u paylaşmalı."))

A.append(card(
    "AutoGen · core", "Fan-out / fan-in",
    f_fanout(),
    "eşzamanlılık ajandan değil, ABONELİKTEN geliyor",
    "Üç analist aynı topic tipine abone olduğu için tek bir yayın üçünü birden "
    "uyandırıyor, ve runtime üçünü paralel koşturuyor. "
    "Dikkat: ortada \"paralel çalıştır\" diyen bir çağrı yok. Paralellik, aynı "
    "duyuruyu birden fazla kişinin dinlemesinin doğal sonucu. "
    "Sonuçları toplamak içinse <b>ayrı bir ajan</b> gerekiyor, ve sebebi "
    "mekanik: <code>publish_message</code> hiçbir şey döndürmüyor. Dönüşü "
    "olmayan bir çağrıdan sonuç toplayamazsın. "
    "Kaç dal beklediğini de çerçeve saymıyor, <b>sen</b> sayıyorsun. Bu yüzden "
    "sayacı <code>finally</code> içinde artırmak zorunlu: dal çökse bile sayaç "
    "ilerlemezse toplayıcı hiç uyanmaz.",
    cap_mm=46))

A.append(card(
    "AutoGen · core", "Müdahale — kapının tek atası",
    f_intervention(),
    "InterventionHandler → DropMessage",
    "Runtime'a takılan bir kanca. Her <code>on_send</code> ve "
    "<code>on_publish</code> buradan geçiyor, ve <code>DropMessage</code> "
    "döndürürsen mesaj yok oluyor. Alıcı hiç haberdar olmuyor. "
    "Ama bu <b>bir onay mekanizması değil</b>, ve farkı bilmek önemli. İnsana "
    "soru sormuyor. Gerekçe döndürmüyor. Kararın kaydını tutmuyor. Ve reddi "
    "ajana bildirmiyor. "
    "Son madde en can alıcısı: ajan neden reddedildiğini bilmediği için başka "
    "bir yol da deneyemiyor. Sadece cevabı gelmemiş bir çağrı görüyor. "
    "Gerçek bir onay kapısı bir katman yukarıda, <b>workbench düzeyinde</b> "
    "kurulmak zorunda. Bizim yaptığımız da bu.",
    cap_mm=46))

A.append(card(
    "AutoGen · agentchat", "AssistantAgent ve tool döngüsü",
    f_tool_loop(),
    "iki ayrı anahtar: max_tool_iterations=1  ·  reflect_on_tool_use=False",
    "Model tool'u çağırıyor, tool koşuyor, sonuç dönüyor — ve varsayılanda tur "
    "orada bitiyor. Modele giden ikinci bir tur yok, dolayısıyla cevabı da model "
    "yazmıyor: kullanıcıya <b>ham tool çıktısı</b> gidiyor "
    "(<code>ToolCallSummaryMessage</code>). "
    "<b>Ölçtük.</b> İki adımlı bir işte — önce id bul, sonra o id ile detay çek — "
    "\"çalışan sayısı kaç\" sorusuna dönen cevap <code>{\"id\": \"KA-9931\"}</code> "
    "oldu. İkinci tool <b>hiç çağrılmadı</b>, ve bunu söyleyen hiçbir şey yok: "
    "log'da tek başarılı çağrı var, hata yok. Sessiz olan ham çıktı değil, "
    "<b>duran zincir</b>. "
    "İki anahtar karıştırılıyor: <code>max_tool_iterations</code> zincirlemeyi "
    "açıyor, <code>reflect_on_tool_use</code> modelin sonucu okuyup cevabı "
    "yazmasını.",
    cap_mm=40,
    foot="reflect_on_tool_use, output_content_type verildiğinde True'ya döner — yani yapılandırılmış çıktı istemek davranışı değiştirir."))

A.append(card(
    "AutoGen · agentchat", "Beş takım — sırayı kim belirliyor",
    f_teams(),
    "aralarındaki tek fark: sıra kararını kim veriyor",
    "Beş takım tipi var ve hepsi aynı işi yapıyor: birden çok ajanı sırayla "
    "konuşturmak. Aralarındaki tek fark, sıraya kimin karar verdiği. "
    "<b>RoundRobin</b> sabit bir döngü izliyor. <b>Selector</b> her turda "
    "modele soruyor. <b>Swarm</b> kararı ajanın kendisine bırakıyor. "
    "<b>MagenticOne</b> bir planlayıcıya veriyor. <b>GraphFlow</b> önceden "
    "çizilmiş bir grafiği takip ediyor. "
    "Maliyet farkı da buradan doğuyor. Selector her turda <b>iki</b> model "
    "çağrısı yapıyor: biri kimin konuşacağını seçmek için, biri asıl iş için. "
    "Sıra zaten belliyse bu ödenmemesi gereken bir maliyet. "
    "İyi haber: beşi de aynı arayüzü sunuyor, yani takım değiştirmek çağıran "
    "kodu değiştirmiyor. Denemek ucuz.",
    cap_mm=46))

A.append(card(
    "AutoGen · agentchat", "GraphFlow — boru hattını çizmek",
    f_graphflow(),
    "DiGraphBuilder → add_edge(...) → join(all)",
    "Adımların sırası zaten belliyse, sırayı modele sordurmanın anlamı yok. "
    "GraphFlow'da grafiği çiziyorsun ve koşturuyorsun. "
    "Aynı düğüme birden fazla ok giriyorsa orası bir <b>birleşme noktası</b>. "
    "Varsayılan politika <code>all</code>, yani bütün dallar bitene kadar "
    "bekliyor. "
    "Kazandırdığı üç şey var. Eşzamanlılık bedava geliyor, çünkü aynı düğümden "
    "çıkan dallar paralel koşuyor. Fazladan model çağrısı yok, çünkü sırayı "
    "kimseye sormuyorsun. Ve <b>grafiğin kendisi bir belge</b>: sistemin ne "
    "yaptığı, kodun kendisine bakınca görünüyor.",
    cap_mm=48))

A.append(card(
    "AutoGen · agentchat", "Mesaj mı, olay mı",
    f_message_types(),
    "mesaj = konuşmanın parçası  ·  olay = ne olduğunun anlatımı",
    "Ayrım kulağa akademik geliyor ama tamamen pratik. <b>Mesajlar</b> bağlama "
    "giriyor, yani bir sonraki model çağrısında tekrar gönderiliyor ve tekrar "
    "para ediyor. <b>Olaylar</b> yalnız gözlem için; bağlama hiç girmiyor. "
    "<code>StructuredMessage</code> bunun üstüne bir pydantic şeması taşıyor. "
    "Yani modelin cevabını serbest metin olarak ayrıştırmak yerine, doğrudan "
    "tipli bir nesne alıyorsun. "
    "Ve bir tuzak: akışın <b>son elemanı</b> bir olay değil, "
    "<code>TaskResult</code>'ın kendisi. Döngüde tip kontrolü yapmazsan onu da "
    "bir olay sanıp işlersin, sonra da \"sonuç hiç gelmedi\" diye ararsın.",
    cap_mm=48))

A.append(card(
    "AutoGen · core", "Components Guide — beş değiştirilebilir yüzey",
    f_components(),
    "core kılavuzu 05:1977 — Components Guide",
    "core'un \"bileşen\" dediği şey, ajan döngüsünün sökülüp değiştirilebilen "
    "parçaları. Beş tane var. "
    "<b>Model client</b> sağlayıcıyla konuşuyor. <b>Model context</b> modele ne "
    "gideceğine karar veriyor. <b>Tools</b> modelin çağırabildiği fonksiyonlar. "
    "<b>Workbench</b> o tool'ların toplandığı arayüz. <b>Code executors</b> "
    "kodun nerede koşacağını belirliyor. "
    "Beşinin ortak özelliği şu ve mimarinin kilidi burada: <b>ajan, hangi "
    "uygulamayla konuştuğunu bilmiyor.</b> Onay kapısını bir workbench "
    "sarmalayıcısı olarak kurabilmemizin sebebi tam olarak bu. "
    "Ayrıca hepsi <code>dump_component</code> ile JSON'a yazılıp geri "
    "yüklenebiliyor: yapılandırma kod değil, <b>veri</b>.",
    cap_mm=42,
    foot="Bizde: model client engine.py · context context_engine.py · workbench gateway/workbench.py"))

A.append(card(
    "AutoGen · core", "Çok-ajan desenleri — resmî sekizi",
    f_patterns(),
    "core kılavuzu 05:3206 — Multi-Agent Design Patterns",
    "Kılavuzun kendi bölümlemesinde sekiz başlık var. Yedisi orkestrasyon "
    "deseni; sonuncusu (<b>Code Execution</b>) bir orkestrasyon değil, bir "
    "yetenek. "
    "Asıl dikkat edilecek şey listede <b>olmayanlar</b>. Planlayıcı-yürütücü, "
    "kara tahta, yönlendirici gibi desenler literatürde var ama core "
    "kılavuzunda yok. "
    "Yani bir kaynakta \"AutoGen'in dokuz deseni\" diye bir tablo görürsen, o "
    "tasnif kılavuzun değil <b>yazarın</b>. Bunu kontrol etmenin yolu da açık: "
    "her başlığın 05 satır numarası var. "
    "Bizim fiilen kullandığımız iki tane: <b>Concurrent Agents</b> (tarama boru "
    "hattı) ve kısmen <b>Reflection</b> (risk denetçisi üç analizi çapraz "
    "kontrol ediyor).",
    cap_mm=48))

A.append(card(
    "AutoGen", "Bağlam ve cache sınırı",
    f_context(),
    "sabit olan önde  →  önek önbelleğe uygun kalır",
    "Model hiçbir şey hatırlamıyor. Her turda bağlamın tamamı baştan "
    "gönderiliyor. \"Hatırlıyor\" dediğimiz şey işte bu tekrar gönderme, ve her "
    "tur bir öncekinden pahalı. "
    "Sağlayıcılar bunu ucuzlatmak için isteğin başındaki <i>değişmeyen</i> "
    "kısmı çok daha düşük ücretlendiriyor. Ama önbellek <b>önekten</b> "
    "çalışıyor: baştan başlayıp ilk farklı bayta kadar. "
    "Dolayısıyla değişken bir şeyi başa koyarsan, arkasındaki her şey önbellekten "
    "düşüyor — tek bir tarih damgası bütün sistem prompt'unu yakabilir. "
    "Sorulacak doğru soru \"prompt nasıl kısalır\" değil, <b>\"ne gerçekten "
    "sabit kalabilir\"</b>.",
    cap_mm=44,
    foot="Ayrı bir ölçüm: aynı görevde yalnız orkestrasyon desenini değiştirmek %63,7 token farkı yaratıyor (poc/kiyas.py)."))

A.append(card(
    "AutoGen · agentchat", "Durmayı öğretmek",
    f_termination(),
    "sert tavan modelden bağımsız  ·  anlamsal koşul modele bağımlı",
    "On bir sonlandırma koşulu var, ama pratikte ikiye ayrılıyorlar. "
    "<b>Sert tavanlar</b> — mesaj sayısı, token bütçesi, geçen süre — modelden "
    "tamamen bağımsız çalışıyor ve her zaman tutuyor. "
    "<b>Anlamsal koşullar</b> — \"model BİTTİ yazınca dur\" gibi — modelin "
    "işbirliğine bağlı. Model o kelimeyi yazmazsa koşul hiç tetiklenmiyor. "
    "Buradan çıkan kural net: üretimde her zaman <b>en az bir sert tavan</b> "
    "olmalı. Anlamsal koşul iyi bir kolaylık ama tek başına bir güvence değil. "
    "Tavansız koşan bir takım, faturayı modelin kararına bırakır.",
    cap_mm=48))

A.append(card(
    "AutoGen", "Dört sessiz varsayılan",
    f_gotchas(),
    "hiçbiri hata vermiyor",
    "Bir çerçevede en pahalı şey, makul görünen ama sessizce yanlış olan "
    "varsayılandır. AutoGen'de dört tane var, ve dördü de sistemi çalıştırıp "
    "sonucu bozuyor. "
    "<code>max_tool_iterations=1</code> tool sonucunu modelden saklıyor. "
    "<code>model_context</code> vermezsen ajanın belleği hiç olmuyor. "
    "<code>model_client_stream=False</code> ile token akışı hiç yayılmıyor. "
    "Sonlandırma koşulu yazmazsan takım tavansız koşuyor. "
    "Dördünü de <b>açıkça yazmak</b> iki şey kazandırıyor: bugün ne olduğunu "
    "bilirsin, ve bir sonraki sürümde varsayılan değişirse kodun etkilenmez.",
    cap_mm=46))

A.append(card(
    "AutoGen", "Geriye kalan",
    f_layers(),
    "motor hazır, kuşatma değil",
    "AutoGen sana bir <b>motor</b> veriyor: aktör modeli, takımlar, akış, "
    "sonlandırma, olay yayını. Bunlar sağlam, ve bu destede gördüğün her şey "
    "gerçekten çalışıyor. "
    "Vermediği şey <b>kontrol düzlemi</b>, ve üçünün de yakını var ama üçü de "
    "eksik. <b>Kapı</b> mesaj katmanında duruyor: bir mesajı düşürebiliyor ama "
    "\"bu ajan şu komutu çalıştırmak istiyor\"u görmüyor, ve reddi ajana "
    "gerekçesiz dönüyor. <b>Onay</b> var — yalnız <code>CodeExecutorAgent</code> "
    "içinde, deneysel, kod çalıştırmaya özel, verilmezse sadece uyarı; ve onaylı "
    "bir ajan <b>yapılandırmaya yazılamıyor</b>, kod bunu reddediyor. "
    "<b>Denetim kaydı</b> ise Python logging'i: teslim garantisi yok, ve modelin "
    "bütün mesajlarını içine koyuyor. "
    "İkinci deste aynı problemi <i>ters uçtan</i> çözmüş bir sistem — ve onun "
    "denetim kaydının sorunu tam tersi olacak: hiç içerik tutmuyor.",
    cap_mm=40,
    foot="Uzun anlatım, ölçümler ve atıflar: docs/pdf/slaytlar.pdf · adım adım öğretici: docs/pdf/ogretici.pdf"))


# ══════════════════════════════════════════════════════ DESTE 2 — OpenClaw

B: list[str] = [cover(
    "hap deste · 2/2",
    "OpenClaw, on sekiz şemada",
    "Ajanı kuşatan kontrol düzlemi: yetki, onay, denetim, bellek, "
    "zamanlama ve yeniden başlatma dayanıklılığı.",
    "github.com/openclaw/openclaw @ 01cc7106<br>uzun anlatım: docs/pdf/openclaw-ici.pdf · docs/16 · docs/18",
)]

B.append(card(
    "OpenClaw", "Kuşbakışı",
    f_oc_arch(),
    "her şey Gateway'den geçer",
    "Solda kanallar var: web arayüzü, komut satırı, sohbet uygulamaları, "
    "webhook'lar, cihaz düğümleri. Sağda yetenekler var: ajan runtime'ı, "
    "tool'lar ve skill'ler, bellek. "
    "Ortada <b>Gateway</b> duruyor, ve içinde kim olduğun, neye yetkili olduğun, "
    "hangi oturuma düştüğün, ne zamanlandığı ve ne kaydedildiği var. "
    "Mimarinin özü tek cümlede: <b>ajan runtime'ı bunların hiçbirini "
    "bilmiyor.</b> Kimlik, yetki ve denetim ajan döngüsünün <i>dışında</i> "
    "kalıyor. "
    "Ve tam bu yüzden değiştirilebilir. Ajan motorunu söküp yerine başkasını "
    "koyabilirsin; kontrol düzlemi yerinde kalır.",
    cap_mm=46))

B.append(card(
    "OpenClaw · yetki", "Üç kontrol ekseni",
    f_three_axes(),
    "sandbox: nerede  ·  tool policy: hangisi  ·  elevated: kaçış",
    "\"İzin\" tek bir kavram değil, üç ayrı soru. Kodun <i>nerede</i> koştuğu, "
    "<i>hangi</i> tool'un çağrılabildiği, ve kutunun <i>dışına</i> çıkmanın bir "
    "yolu olup olmadığı. Bu üçünü karıştırmak en yaygın yapılandırma hatası. "
    "Filtre kuralı basit: <code>deny</code> her zaman kazanıyor, ve "
    "<code>allow</code> listesi doluysa listede olmayan her şey bloklu sayılıyor. "
    "Ama asıl bilinmesi gereken şu: <b>tool politikası tool'u yalnız adına göre "
    "filtreliyor.</b> Bir <code>exec</code> çağrısının içinde ne yapıldığına "
    "bakmıyor. "
    "Yani \"write tool'unu kapattık, artık salt-okunur\" cümlesi yanlıştır. "
    "<code>exec</code> serbestse kabuk üstünden yazmak zaten mümkün.",
    cap_mm=44))

B.append(card(
    "OpenClaw · yetki", "Metot kapsamı yalnızca ilk kapı",
    f_scopes(),
    "kapsam çağrının PARAMETRESİNDEN türetiliyor",
    "Sekiz yetki kapsamı var, ama asıl fikir tabloda değil. "
    "Fikir şu: <b>aynı metot, farklı parametreyle farklı yetki istiyor.</b> "
    "<code>agent</code> metodu sıradan bir tur için yazma yetkisiyle geçiyor, "
    "ama <code>/reset</code> için yönetici istiyor. <code>node.invoke</code> "
    "normal komutlar için yazma, <code>browser.proxy</code> için yönetici. "
    "İki kural bunu tamamlıyor. <b>Yetki yükseltme yasak</b>: bir cihazı "
    "onaylarken yalnız zaten sahip olduğun kapsamları verebiliyorsun. Ve "
    "<b>sessiz genişleme yok</b>: daha geniş bir rol isteyen yeniden bağlanma "
    "otomatik geçmiyor, yeni bir onay talebi doğuruyor.",
    cap_mm=44))

B.append(card(
    "OpenClaw · onay", "Onay komuta değil, plana bağlanır",
    f_frozen_plan(),
    "argüman değiştiyse → approval mismatch",
    "Naif bir onay akışında onay ile çalıştırma arasında bir boşluk kalıyor: "
    "kullanıcı gördüğü şeyi onaylıyor, ama çalışan başka bir şey olabiliyor. "
    "Aradan geçen sürede argümanlar değişmiş olabilir. "
    "OpenClaw onay isteğinin içine <b>kanonik bir plan</b> koyuyor: çalışma "
    "dizini, tam argüman listesi, sabitlenmiş dosya yolu. "
    "Onaylandıktan sonra <i>saklanan planı</i> çalıştırıyor, çağıranın "
    "sonradan gönderdiğini değil. "
    "Bir dosyaya bağlıysa ve dosya onaydan sonra değiştiyse, kaymış içeriği "
    "çalıştırmak yerine koşuyu <b>reddediyor</b>. "
    "Yani onayladığın şey bir cümle değil, <b>donmuş bir plan</b>.",
    cap_mm=44))

B.append(card(
    "OpenClaw · güvenlik", "Dış içerik veri, talimat değil",
    f_external_content(),
    "rastgele id'li sınır + 22 token temizliği + 28 homoglif katlaması",
    "Modelin bağlamına giren her şey aynı görünüyor: metin. Senin sistem "
    "talimatın da metin, müşterinin gönderdiği PDF de metin. "
    "O PDF'in içinde \"önceki talimatları yok say\" yazıyorsa, model bunu neden "
    "talimat saymasın? Ayırt edecek bir işaret yok. "
    "OpenClaw dış içeriği bir sarmalayıcının içine koyuyor, ve sarmalayıcının "
    "id'si <b>rastgele</b>. Sabit olsaydı içerik kendi kapanış etiketini yazıp "
    "kutudan çıkardı. "
    "Ve şüpheli desenler yalnızca <b>loglanıyor</b>, engellenmiyor. Sebebi "
    "dürüst: desen eşleştirmeyle injection engellenemez. Tespit bir sinyal; "
    "asıl savunma sarmalayıcının kendisi.",
    cap_mm=44))

B.append(card(
    "OpenClaw · prompt", "Kademeli açığa çıkarma",
    f_skill_disclosure(),
    "prompt'ta indeks, gövde talep üzerine  →  %93 tasarruf",
    "Kurulu 74 skill var, ama prompt'a hepsinin gövdesi girmiyor. Yalnız "
    "<b>indeksleri</b> giriyor: ad ve tek satırlık açıklama. "
    "Model bir skill'e ihtiyaç duyduğunda gövdesini <code>read</code> ile "
    "çekiyor. İhtiyaç duymadıklarının gövdesi hiç yüklenmiyor. "
    "Kazanç yalnız token değil, aynı zamanda <b>isabet</b>. Model 74 talimatı "
    "aynı anda görmediği için dikkati dağılmıyor; yalnız işine yarayanı "
    "okuyor. "
    "Ve aynı fikir tool katmanında da <i>bağımsız olarak</i> ortaya çıkmış. Yani "
    "bu tek seferlik bir numara değil, tekrar eden bir <b>desen</b>.",
    cap_mm=46))

B.append(card(
    "OpenClaw · prompt", "Tool Search — büyük katalog, küçük prompt",
    f_tool_search(),
    "dizin cache sınırının üstünde  ·  köprü izole  ·  fail-closed",
    "Model bütün tool şemalarını görmüyor. Sınırlı bir yetenek dizini görüyor ve "
    "<code>search</code> → <code>describe</code> → <code>call</code> sırasını "
    "izliyor. "
    "Dizin ada göre sıralı, ve <b>cache sınırının üstüne</b> konmuş. Yani "
    "kullanıcı mesajının içine girmiyor. Girseydi her turda değişir ve prompt "
    "cache'i her turda bozulurdu. "
    "Aradaki kod köprüsü izole bir alt süreçte koşuyor: environment boş, dosya "
    "sistemi yok, ağ yok, <b>sır yok</b>. Köprüdeki kod tek başına hiçbir şey "
    "yapamıyor; her gerçek çağrı Gateway'e geri dönüyor ve normal politika, onay "
    "ve log yolundan geçiyor. "
    "Ve fail-closed: politikaya takılan bir tool <b>aramada hiç görünmüyor</b>.",
    cap_mm=42))

B.append(card(
    "OpenClaw · bellek", "Beş katman",
    f_memory_tiers(),
    "asıl sınır ikinci ile üçüncü katmanın arasında",
    "Beş katman var ama hepsini ezberlemene gerek yok; bilmen gereken tek şey "
    "sınırın nerede olduğu. "
    "<b>Curated</b> katmanı küçük, her oturumda bağlama giriyor, ve oraya "
    "yalnızca kapılı bir konsolidasyondan geçerek yazılabiliyor. "
    "<b>Episodic</b> katmanı büyük, ekleme dostu, ve <b>hiç enjekte "
    "edilmiyor</b>; yalnız arama yoluyla erişilebiliyor. "
    "Kural şu: episodic'ten curated'a hiçbir şey <b>kapıdan geçmeden</b> "
    "çıkmıyor. Yani büyük ve denetimsiz olan taraf, küçük ve her zaman okunan "
    "tarafa kendiliğinden sızamıyor. "
    "En üstteki katmanı yalnız insan yazıyor, en alttakini yalnız insan okuyor.",
    cap_mm=44))

B.append(card(
    "OpenClaw · bellek", "Güvenlik sınırı yazma yolunda",
    f_memory_write(),
    "köken şemada zorunlu, kapalı küme, model yazamıyor",
    "Belleğe zehirli bir olgu girdikten sonra onu içerik taramasıyla yakalamak "
    "güvenilir değil. \"Şirketin CEO'su X\" cümlesi doğru mu yanlış mı, metne "
    "bakarak anlaşılmıyor. "
    "Bu yüzden savunma \"kötü belleği sonradan bul\" değil, <b>kötü belleğin "
    "terfi edememesi</b> üstüne kurulmuş. "
    "Her kaydın bir <b>köken</b> sınıfı var. Bu sınıf kapalı bir kümeden "
    "seçiliyor ve SQLite'ta ayrı bir sütunda duruyor, yani modelin düzyazıyla "
    "yazamayacağı bir yerde. "
    "Sınıflandırma da muhafazakâr: kökeni belirlenemeyen dışsal bir içerik "
    "<code>untrusted</code> sayılıyor, <b>asla owner varsayılmıyor</b>. "
    "Ve döngü kırılıyor: bellekten enjekte edilen içerik, yeni bellek olarak "
    "geri yazılmıyor.",
    cap_mm=42))

B.append(card(
    "OpenClaw · bağlam", "Bağlam motoru ve sıkıştırma",
    f_ctx_engine(),
    "dört yaşam noktası, bir bozulmaz kural",
    "\"Modele ne gönderilecek\" kararı çekirdeğe gömülü değil. Dört ayrı anda "
    "eklentiyle değiştirilebiliyor: mesaj eklenirken, model koşusundan hemen "
    "önce, pencere dolduğunda, ve tur bittiğinde. "
    "Altta pazarlığa açık olmayan tek bir kural duruyor: <b>tool çağrısı "
    "sonucundan ayrılmıyor.</b> "
    "Sebebi teknik ve kesin. Ayrılsaydı model, cevabını hiç görmediği bir çağrı "
    "yapmış gibi görünürdü — ve çoğu sağlayıcı bu şekli geçersiz sayıp isteği "
    "<b>reddediyor</b>. Yani sıkıştırma hatası bir kalite sorunu olarak değil, "
    "bir API hatası olarak karşına çıkıyor. "
    "Kendi sıkıştırıcını yazacaksan ilk yazacağın test bu.",
    cap_mm=42))

B.append(card(
    "OpenClaw · zamanlama", "Zamanlama yığını",
    f_task_stack(),
    "ne tetikler → kim karar verir → ne kaydedilir → nasıl serileşir",
    "\"Arka planda iş çalıştır\" tek bir ihtiyaç değil, ve OpenClaw bunu dört "
    "ayrı soruya bölmüş. "
    "Beş tetikleyici türü var: <code>at</code>, <code>every</code>, "
    "<code>cron</code>, <code>on-exit</code>, <code>stream</code>. Son ikisi "
    "zamana <b>hiç</b> bakmıyor; onlar olay kaynağı. "
    "Karar iki ayrı zamanlayıcıya bölünmüş. <b>Automations</b> tam zamanında "
    "koşuyor, izole bir oturum kullanıyor, ve her koşuda bir task kaydı "
    "üretiyor. <b>Heartbeat</b> yaklaşık zamanlı, tam bağlamlı, ve <b>hiç</b> "
    "kayıt üretmiyor. "
    "Ayrımın sebebi pratik: \"her sabah 9'da rapor\" izolasyon istiyor, \"ara "
    "sıra gelen kutusuna göz at\" ise bağlam istiyor.",
    cap_mm=42))

B.append(card(
    "OpenClaw · zamanlama", "Task defteri — kayıt, zamanlayıcı değil",
    f_task_lifecycle(),
    "terminal durum yapışkan  ·  lost bir kanıt standardı",
    "Automations ve heartbeat işin <i>ne zaman</i> koşacağına karar veriyor. "
    "Task defteri ise <i>ne olduğunu</i> izliyor. İkisi ayrı bileşen, ve ayrı "
    "kalmaları önemli. "
    "Zamanlayıcı bir operasyon bileşeni; hızlı olmalı. Defter bir kanıt "
    "bileşeni; doğru olmalı. Karıştırırsan denetime götüreceğin kayıt, aynı "
    "zamanda performans için budanmış olan kayıt olur. "
    "<b>Terminal durum yapışkan</b>: bir iş iptal edildikten sonra gelen başarı "
    "sinyali operatörün kararını geri alamıyor. "
    "Ve <code>lost</code> için her kaynağın kendi kanıt standardı var. "
    "Kılavuzun cümlesi şu: \"çevrimdışı bir denetim, kendi boş durumunu otorite "
    "saymaz.\"",
    cap_mm=42,
    foot="Yani: bir şeyi görememek, o şeyin olmadığının kanıtı değil."))

B.append(card(
    "OpenClaw · zamanlama", "Üç eksen, tip düzeyinde ayrı",
    f_task_axes(),
    "status ≠ deliveryStatus ≠ terminalOutcome",
    "Şemada üç ayrı enum var: işin <i>yürütülmesi</i>, sonucun <i>teslimatı</i>, "
    "ve işin <i>nihai sonucu</i>. Üçü ayrı tip olduğu için karıştırmak "
    "<b>mümkün değil</b> — tip sistemi hatırlatıyor. "
    "Somut senaryo: bir alt-ajan işini bitirdi, ama sonucu vereceği oturum "
    "kapanmıştı. Bu iş <code>blocked</code>, <b><code>failed</code> değil</b>. "
    "Fark tamamen pratik. <code>failed</code> deseydin biri gelip işi yeniden "
    "koşturur, yani yapılmış işi ikinci kez yaptırırsın. <code>blocked</code> "
    "diyorsan iş zaten yapılmış, eksik olan yalnız teslimat. "
    "Çözüm de ona göre değişiyor: yeniden koşturmak değil, teslimatı düzeltmek.",
    cap_mm=42))

B.append(card(
    "OpenClaw · çalışma", "Eşzamanlılık: üç katman",
    f_threads(),
    "event loop · worker thread · çocuk süreç",
    "Node tek bir event loop'la çalışıyor, ama sistem tek katmanlı değil. "
    "<b>Ajan işi event loop'ta</b> koşuyor. Buradaki paralellik iş "
    "parçacığından gelmiyor, G/Ç beklemesinden geliyor: bir istek cevap "
    "beklerken başka iş yapılıyor. "
    "<b>Worker thread'ler</b>, cevap yolunda olmaması gereken defter işleri "
    "için: denetim kaydı yazmak, SQLite arşivlemek, transkript uzlaştırmak. "
    "Bunun doğrudan bir sonucu var — kuyruk dolarsa <b>kayıt düşüyor, koşu "
    "düşmüyor</b>. "
    "<b>Çocuk süreçler</b> uzun ömürlü izleyiciler için. Ve sahiplik kritik: "
    "izleyici gateway'e ait, tura değil. Tur bittiğinde izleyici ölmüyor.",
    cap_mm=42))

B.append(card(
    "OpenClaw · dayanıklılık", "Dayanıklı durum — ama durable execution değil",
    f_durable(),
    "kurtarma bir REPLAY değil, modele yazılan bir cümle",
    "Her şey SQLite'ta duruyor: konuşma geçmişi, yarıda kalan tur, alt-ajan "
    "koşuları, zamanlanmış işler, kuyruktaki giden mesajlar. Yani süreç ölse de "
    "<b>durum kaybolmuyor</b>. "
    "Ama kurtarmanın <i>ne olduğu</i> önemli. Gateway, oturumu ajana geri "
    "veriyor ve yanına <b>sentetik bir sistem mesajı</b> ekliyor: \"önceki turun "
    "kesildi, mevcut transkriptten devam et.\" "
    "Yani devam eden şey bir fonksiyon değil, bir <b>istem</b>. Deterministik "
    "replay yok. Tamamlanmış adımların memoizasyonu yok. "
    "Sonucu şu: bir tur yan etkili bir tool'u çağırdıktan sonra çöktüyse, o "
    "tool'un ikinci kez çağrılmasını <b>mekanik olarak engelleyen hiçbir şey "
    "yok</b>. Tek koruma, modelin transkripti okuyup fark etmesi.",
    cap_mm=40,
    foot="Var olan: dayanıklı durum · dayanıklı idempotensi (tek gönderim kimliği) · üç denemelik dayanıklı bütçe, sonra tombstone."))

B.append(card(
    "OpenClaw · sırlar", "Sentinel: gerçek değer son sınırda",
    f_secrets(),
    "loglar sentinel görür, sağlayıcı gerçek değeri",
    "Sırlar yapılandırmada düz metin olarak durmuyor. Yerlerine süreç-yerel, "
    "anlamsız bir <b>sentinel</b> yazılıyor. "
    "Auth deposu, SDK yapılandırması, loglar ve hata nesneleri hep <i>o "
    "sentinel'i</i> görüyor. Gerçek değer ancak istek süreçten çıkmadan hemen "
    "önce yerine konuyor. "
    "Ve tanınmayan, sentinel'e benzeyen bir değer görülürse istek <b>ağ "
    "etkinliğinden önce</b> durduruluyor. Çözülmemiş bir sentinel'i sağlayıcıya "
    "göndermektense hiç göndermemek tercih ediliyor. "
    "Kendi sınırını da yazmışlar, ve doğru yazmışlar: <b>bu bir süreç "
    "izolasyonu değil.</b> Gerçek değer aynı süreçte, bellekte duruyor.",
    cap_mm=42))

B.append(card(
    "OpenClaw · denetim", "İki kayıt hattı gerekiyor",
    f_two_ledgers(),
    "\"Bir satırın yokluğu hiçbir şey kanıtlamaz.\"",
    "OpenClaw'ın denetim kaydı <b>içerik tutmuyor</b>: prompt yok, tool "
    "argümanı yok, URL yok, komut çıktısı yok. Kimlikler de kurulum-yerel bir "
    "anahtarla takma ada çevriliyor. "
    "Kendi sınırını da açıkça yazıyorlar: <i>\"Bu korelasyondur, "
    "anonimleştirme değildir.\"</i> "
    "Ve kayıt <b>best-effort</b>: kuyruk dolarsa satır düşüyor, koşu devam "
    "ediyor. Bir geliştirici aracı için bu doğru öncelik — kayıt uğruna işi "
    "durdurmak saçma olurdu. "
    "Düzenlenmiş bir kurumda ise önceliğin <b>tam tersi</b> gerekiyor: uyum "
    "hattı kayıpsız, senkron ve fail-closed olmalı. Yazılamıyorsa koşu düşmeli.",
    cap_mm=42,
    foot="Alınacak olan mekanizma değil, AYRIM: operasyonel hat ile uyum hattı farklı garantiler ister."))

B.append(card(
    "OpenClaw", "Geriye kalan tek cümle",
    f_oc_arch(),
    "mekanizma taşınır, güven modeli taşınmaz",
    "Bu destedeki her mekanizma bir kurumsal asistana taşınabilir. Kapı, donmuş "
    "plan, köken sınıfı, iki kayıt hattı — hepsi. "
    "Taşınamayan tek şey şu: OpenClaw <b>tek bir güvenilen operatörün</b> "
    "etrafında tasarlanmış. "
    "Belgelerindeki bütün \"bu bir güvenlik sınırı değildir\" cümleleri buradan "
    "geliyor. O modelde zaten herkes güvenilir, dolayısıyla ayrımlar bir "
    "kolaylıktan ibaret. "
    "Birbirine güvenmeyen departmanların olduğu bir kurumda ise aynı cümleler "
    "birer <b>açık</b> haline geliyor. "
    "Yapılacak iş net: mekanizmaları al, <b>güven modelini yeniden kur</b>.",
    cap_mm=46,
    foot="Ayrıntı: docs/16 (ne alınır, ne alınmaz) · docs/17 (şirket planı) · docs/18 (task manager)"))

if __name__ == "__main__":
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    # The third deck lives in its own file: it is implementation detail rather
    # than concept, and keeping it apart lets the first two stay teachable.
    # Sıra önemli: `hap_maf.py` en sona yükleniyor çünkü halef bölümü destenin
    # kuyruğu — önce mekanizma, sonra sınırı, en sonda o sınırın kapanıp
    # kapanmadığı.
    for _f in ("hap_autogen_derin.py", "hap_maf.py", "hap_nis.py"):
        _src = (Path(__file__).resolve().parent / _f).read_text(encoding="utf-8")
        exec(compile(_src, _f, "exec"), globals())  # noqa: S102 — our own files

    print("hap desteler:")
    build_deck(A, "AutoGen — ve halefi MAF", "hap-autogen.html")
    build_deck(B, "OpenClaw, on sekiz şemada", "hap-openclaw.html")
    build_deck(C, "OpenClaw harness'ı, içeriden", "hap-openclaw-nis.html")
