"""Hand-drawn figures shared by the deck and the tutorial.

`rough.py` gives shapes; this gives *labelled* shapes and the specific diagrams
both documents draw. It lives apart from either generator so a figure fixed in
one place is fixed in both — a diagram that drifts between two documents about
the same system is worse than no diagram.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rough import Pen, figure, mono, text  # noqa: E402

# ─────────────────────────────────────────────────────────────── drawing kit
#
# rough.py gives shapes; these give *labelled* shapes, which is all a slide
# figure ever needs. Keeping them here rather than in rough.py leaves the pen
# unaware of what it is drawing.


def box(pen, x, y, w, h, title, sub="", colour="ink", *, dash="", size=9.4):
    """A rectangle that says what it is. `sub` is the class/module underneath."""
    out = [pen.rect(x, y, w, h, colour, dash=dash)]
    cy = y + h / 2 + (0 if not sub else -4)
    out.append(text(x + w / 2, cy + 3.2, title, size=size, colour="#1e1e1e",
                    weight="bold", anchor="middle"))
    if sub:
        out.append(mono(x + w / 2, cy + 14, sub, size=6.8, colour="#767d84",
                        anchor="middle"))
    return "".join(out)


def panel(pen, x, y, w, h, title, colour="ink", *, size=9.4):
    """A box that expects content drawn inside it, so the title goes on top."""
    return (pen.rect(x, y, w, h, colour)
            + text(x + w / 2, y + 15, title, size=size, colour="#1e1e1e",
                   weight="bold", anchor="middle"))


def arrow(pen, x1, y1, x2, y2, label="", colour="ink", *, dash="", above=True):
    out = [pen.line(x1, y1, x2, y2, colour, dash=dash)]
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        out.append(text(mx, my - 6 if above else my + 13, label, size=7.4,
                        colour="#767d84", anchor="middle"))
    return "".join(out)


def note(x, y, s, colour="#767d84", size=7.6, anchor="start"):
    return text(x, y, s, size=size, colour=colour, anchor=anchor)


def band(x, y, w, h, label, colour="grey"):
    """A lane: dashed container with a label in its corner.

    The seed is `crc32`, not `hash`. Python randomises string hashing per
    process, so seeding from `hash(label)` made every rebuild redraw these
    borders differently — breaking exactly the determinism `rough.py` promises,
    and turning every regeneration into a diff.
    """
    p = Pen(zlib.crc32(label.encode("utf-8")) % 9999)
    return (p.rect(x, y, w, h, colour, fill=False, dash="7 5", width=1.2)
            + text(x + 7, y + 13, label, size=7.2, colour="#868e96"))


# ──────────────────────────────────────────────────────────────── the figures


def f_actor():
    p = Pen(11)
    o = [band(6, 6, 588, 168, "SingleThreadedAgentRuntime")]
    o += [box(p, 30, 46, 118, 54, "Agent A", "type + key")]
    o += [box(p, 240, 34, 118, 40, "Agent B", "", "blue")]
    o += [box(p, 240, 96, 118, 40, "Agent C", "", "blue")]
    o += [box(p, 452, 46, 118, 54, "ClosureAgent", "toplayıcı", "green")]
    o += [arrow(p, 150, 66, 238, 54, "publish"), arrow(p, 150, 82, 238, 112)]
    o += [arrow(p, 360, 54, 450, 66), arrow(p, 360, 116, 450, 82)]
    o += [note(30, 122, "mesaj kuyruğu · tek iş parçacığı · sıra korunur")]
    o += [note(30, 136, "ajanlar birbirini çağırmaz — runtime taşır", "#c92a2a")]
    return figure(600, 180, "".join(o))


def f_identity():
    p = Pen(12)
    o = [panel(p, 24, 24, 250, 82, "AgentId", "violet")]
    o += [mono(46, 62, 'type = "analyst"', size=9.6, colour="#1e1e1e")]
    o += [mono(46, 86, 'key  = "arxiv"', size=9.6, colour="#1e1e1e")]
    o += [note(24, 122, "type = DAVRANIŞ (hangi sınıf)")]
    o += [note(24, 136, "key  = ÖRNEK (hangi kopya)", "#c92a2a")]
    o += [box(p, 330, 22, 110, 34, "analyst/arxiv", "", "blue", size=7.6)]
    o += [box(p, 330, 64, 110, 34, "analyst/hn", "", "blue", size=7.6)]
    o += [box(p, 330, 106, 110, 34, "analyst/gh", "", "blue", size=7.6)]
    o += [note(456, 46, "aynı sınıf,")]
    o += [note(456, 60, "üç ayrı örnek,")]
    o += [note(456, 74, "üç ayrı durum")]
    o += [note(330, 158, "örnek talep üzerine doğar — önceden kaydedilmez", "#868e96")]
    return figure(600, 172, "".join(o))


def f_send_vs_publish():
    p = Pen(13)
    o = [note(20, 18, "send_message — DOĞRUDAN", "#1971c2", 8.6)]
    o += [box(p, 20, 30, 96, 40, "gönderen", "", "blue", size=8)]
    o += [box(p, 178, 30, 96, 40, "alıcı", "", "blue", size=8)]
    o += [arrow(p, 118, 50, 176, 50, "1 → 1")]
    o += [note(20, 88, "· cevabı DÖNDÜRÜR")]
    o += [note(20, 102, "· hata ÇAĞIRANA fırlar", "#c92a2a")]
    o += [note(20, 116, "· tek alıcı, bilinen adres")]

    o += [note(330, 18, "publish_message — YAYIN", "#2f9e44", 8.6)]
    o += [box(p, 330, 30, 96, 40, "yayıncı", "", "green", size=8)]
    o += [box(p, 488, 12, 92, 30, "abone 1", "", "green", size=7.4)]
    o += [box(p, 488, 48, 92, 30, "abone 2", "", "green", size=7.4)]
    o += [arrow(p, 428, 44, 486, 28), arrow(p, 428, 56, 486, 60)]
    o += [note(330, 88, "· None DÖNER — cevap yok", "#c92a2a")]
    o += [note(330, 102, "· hata yalnız LOGLANIR", "#c92a2a")]
    o += [note(330, 116, "· 0 abone de geçerli bir sonuç")]
    o += [note(20, 142, "Bu asimetri ölçüldü: yayında düşen bir handler sessizce düşer.",
                "#8a5208", 8)]
    return figure(600, 154, "".join(o))


def f_topic():
    p = Pen(14)
    o = [panel(p, 20, 30, 150, 74, "TopicId", "orange")]
    o += [mono(34, 68, 'type   = "task"', size=8.4, colour="#1e1e1e")]
    o += [mono(34, 90, 'source = "job-7"', size=8.4, colour="#1e1e1e")]
    o += [box(p, 236, 20, 150, 44, "TypeSubscription", 'topic_type="task"', "violet",
              size=8)]
    o += [arrow(p, 172, 66, 234, 46, "eşleşir")]
    o += [box(p, 450, 20, 130, 44, "analyst/job-7", "", "green", size=8)]
    o += [arrow(p, 388, 42, 448, 42)]
    o += [note(236, 88, "KURAL — core kılavuzu 05:670:", "#c92a2a", 8.2)]
    o += [note(236, 104, "topic.source  →  agent.key")]
    o += [note(236, 118, "source değişirse AYRI bir örnek doğar", "#c92a2a")]
    o += [note(20, 122, "type  = ne olduğu")]
    o += [note(20, 136, "source = hangi iş için")]
    return figure(600, 150, "".join(o))


def f_fanout():
    p = Pen(15)
    o = [box(p, 20, 56, 104, 44, "koordinatör", "", "ink", size=8.4)]
    o += [box(p, 196, 8, 116, 36, "arXiv", "RoutedAgent", "blue", size=8)]
    o += [box(p, 196, 58, 116, 36, "HN", "RoutedAgent", "blue", size=8)]
    o += [box(p, 196, 108, 116, 36, "GitHub", "RoutedAgent", "blue", size=8)]
    for ty in (26, 76, 126):
        o += [arrow(p, 126, 78, 194, ty)]
    o += [box(p, 392, 56, 128, 44, "ClosureAgent", "toplayıcı", "green", size=8.4)]
    for fy in (26, 76, 126):
        o += [arrow(p, 314, fy, 390, 78)]
    o += [note(196, 162, "hepsi AYNI topic'e abone — tek publish üçünü birden uyandırır")]
    o += [note(392, 116, "sayaç 3'e ulaşınca")]
    o += [note(392, 130, "kuyruğa yazar")]
    o += [note(20, 116, "1 publish", "#2f9e44")]
    o += [note(20, 130, "0 dönüş değeri", "#c92a2a")]
    return figure(600, 172, "".join(o))


def f_intervention():
    p = Pen(16)
    o = [box(p, 18, 44, 96, 42, "gönderen", "", "blue", size=8)]
    o += [box(p, 172, 34, 140, 62, "InterventionHandler", "on_send / on_publish",
              "orange", size=8.6)]
    o += [box(p, 386, 20, 106, 38, "alıcı", "", "green", size=8)]
    o += [arrow(p, 116, 66, 170, 66)]
    o += [arrow(p, 314, 54, 384, 40, "geçti")]
    o += [p.cross(360, 96, 12)]
    o += [note(378, 100, "DropMessage", "#c92a2a", 8.2)]
    o += [arrow(p, 314, 80, 348, 92, colour="red")]
    o += [note(172, 122, "runtime'a takılır — ajanların hiçbiri bunu görmez")]
    o += [note(172, 136, "bizim onay kapımız bu noktanın karşılığı", "#8a5208")]
    return figure(600, 150, "".join(o))


def f_tool_loop():
    p = Pen(21)
    o = [box(p, 16, 52, 104, 44, "AssistantAgent", "", "ink", size=8)]
    o += [box(p, 176, 52, 104, 44, "model", "create_stream", "blue", size=8)]
    o += [box(p, 336, 52, 104, 44, "workbench", "call_tool", "orange", size=8)]
    o += [box(p, 486, 52, 96, 44, "sonuç", "", "green", size=8)]
    o += [arrow(p, 122, 74, 174, 74)]
    o += [arrow(p, 282, 74, 334, 74, "tool isteği")]
    o += [arrow(p, 442, 74, 484, 74)]
    o += [p.curve([(536, 52), (500, 16), (300, 8), (120, 22), (68, 50)], arrow=True)]
    o += [note(280, 6, "döngü — max_tool_iterations")]
    o += [note(16, 122, "VARSAYILAN 1: model tool sonucunu GÖRMEDEN cevap verir",
                "#c92a2a", 8.4)]
    o += [note(16, 138, "ölçüldü — bizde 6'ya çekildi", "#2f9e44")]
    return figure(600, 152, "".join(o))


def f_teams():
    p = Pen(22)
    names = [("RoundRobin", "sırayla"), ("Selector", "model seçer"),
             ("Swarm", "handoff"), ("MagenticOne", "planlayıcı"),
             ("GraphFlow", "DAG")]
    o = []
    for i, (n, s) in enumerate(names):
        x = 12 + i * 118
        o += [box(p, x, 26, 104, 52, n, s, "violet", size=8)]
    o += [note(12, 100, "Beşi de aynı arayüz: run() / run_stream() → TaskResult")]
    o += [note(12, 116, "Taramamız GraphFlow kullanıyor — eşzamanlı dal + join(all)",
                "#2f9e44")]
    return figure(600, 128, "".join(o))


def f_graphflow():
    p = Pen(23)
    o = [box(p, 16, 60, 92, 40, "giriş", "", "ink", size=8)]
    o += [box(p, 168, 10, 104, 36, "analist A", "", "blue", size=7.8)]
    o += [box(p, 168, 62, 104, 36, "analist B", "", "blue", size=7.8)]
    o += [box(p, 168, 114, 104, 36, "analist C", "", "blue", size=7.8)]
    for ty in (28, 80, 132):
        o += [arrow(p, 110, 80, 166, ty)]
    o += [box(p, 336, 58, 104, 44, "join(all)", "hepsini bekler", "orange", size=8)]
    for fy in (28, 80, 132):
        o += [arrow(p, 274, fy, 334, 80)]
    o += [box(p, 490, 58, 96, 44, "sayım", "", "green", size=8)]
    o += [arrow(p, 442, 80, 488, 80)]
    o += [note(168, 170, "DiGraphBuilder ile kurulur · dallar EŞZAMANLI koşar")]
    o += [note(336, 118, "join politikası: all | any")]
    return figure(600, 182, "".join(o))


def f_message_types():
    p = Pen(24)
    rows = [
        ("TextMessage", "düz metin", "blue"),
        ("StructuredMessage", "pydantic şema", "blue"),
        ("ToolCallRequestEvent", "model tool istedi", "orange"),
        ("ToolCallExecutionEvent", "tool koştu", "orange"),
        ("ModelClientStreamingChunkEvent", "token akışı", "green"),
        ("TaskResult", "bitiş + stop_reason", "violet"),
    ]
    o = []
    for i, (n, s, c) in enumerate(rows):
        y = 10 + i * 30
        o += [box(p, 14, y, 268, 24, "", "", c)]
        o += [mono(24, y + 16, n, size=8, colour="#1e1e1e")]
        o += [note(300, y + 16, s)]
    o += [note(14, 200, "Ayrım: mesaj konuşmanın parçası, olay ise ne olduğunun anlatımı.",
                "#8a5208", 8)]
    return figure(600, 212, "".join(o))


def f_context():
    p = Pen(25)
    o = [band(8, 8, 584, 128, "ChatCompletionContext")]
    o += [box(p, 26, 40, 92, 40, "sistem", "", "grey", size=8)]
    o += [box(p, 132, 40, 92, 40, "özet", "compaction", "orange", size=8)]
    o += [box(p, 238, 40, 74, 40, "tur n-2", "", "blue", size=7.6)]
    o += [box(p, 322, 40, 74, 40, "tur n-1", "", "blue", size=7.6)]
    o += [box(p, 406, 40, 74, 40, "tur n", "", "blue", size=7.6)]
    o += [box(p, 490, 40, 84, 40, "yeni soru", "", "green", size=7.6)]
    o += [note(26, 100, "sabit — cache sınırının ÜSTÜ", "#2f9e44")]
    o += [note(300, 100, "değişken — cache sınırının ALTI", "#c92a2a")]
    o += [p.line(288, 30, 288, 92, "red", arrow=False, dash="4 4")]
    o += [note(288, 24, "CACHE SINIRI", "#c92a2a", 7.4, anchor="middle")]
    return figure(600, 144, "".join(o))


def f_gateway():
    p = Pen(31)
    o = [box(p, 18, 54, 96, 44, "kullanıcı", "", "grey", size=8)]
    o += [box(p, 156, 20, 150, 112, "GATEWAY", "onay · politika · kayıt", "orange",
              size=9.4)]
    o += [arrow(p, 116, 76, 154, 76)]
    o += [box(p, 356, 14, 116, 40, "AutoGen motoru", "GraphFlow", "blue", size=7.8)]
    o += [box(p, 356, 62, 116, 40, "workbench", "GatedWorkbench", "green", size=7.8)]
    o += [box(p, 356, 110, 116, 34, "OpenClaw", "MCP köprüsü", "violet", size=7.8)]
    for ty in (34, 82, 127):
        o += [arrow(p, 308, 76, 354, ty)]
    o += [box(p, 508, 62, 78, 40, "dış dünya", "", "red", size=7.6)]
    o += [arrow(p, 474, 82, 506, 82)]
    o += [note(156, 152, "Kapı TEK yer: dışarı giden her çağrı buradan geçer.", "#8a5208")]
    return figure(600, 164, "".join(o))


def f_gate():
    p = Pen(32)
    o = [box(p, 16, 46, 96, 42, "ajan", "", "blue", size=8)]
    o += [box(p, 158, 34, 132, 64, "GatedWorkbench", "call_tool", "orange", size=8.6)]
    o += [arrow(p, 114, 66, 156, 66)]
    o += [box(p, 344, 12, 112, 38, "İZİN", "tool koşar", "green", size=8)]
    o += [box(p, 344, 82, 112, 38, "RET", "gerekçe döner", "red", size=8)]
    o += [arrow(p, 292, 56, 342, 32, colour="green")]
    o += [arrow(p, 292, 78, 342, 98, colour="red")]
    o += [box(p, 490, 82, 96, 38, "onay isteği", "id + argüman", "violet", size=7.6)]
    o += [arrow(p, 458, 100, 488, 100)]
    o += [note(16, 130, "Onay metni id'yi TAŞIMALI — düşerse arayüz düğmeyi çizemez.",
                "#c92a2a", 8)]
    o += [note(16, 146, "Bu hatayı ölçtük: testi önce eski koda karşı düşürdük.", "#2f9e44")]
    return figure(600, 158, "".join(o))


def f_skill_disclosure():
    p = Pen(41)
    o = [note(20, 16, "PROMPT'TA NE VAR", "#1e1e1e", 8.6)]
    o += [box(p, 20, 26, 240, 44, "74 skill'in İNDEKSİ", "ad + tek satır", "green",
              size=8.4)]
    o += [note(20, 88, "gövdeler diskte bekler:")]
    for i in range(6):
        o += [p.rect(20 + i * 40, 96, 32, 26, "grey")]
    o += [note(268, 96, "…74 tane")]
    o += [box(p, 340, 26, 240, 44, "TAMAMI prompt'ta olsaydı", "", "red", size=8.4)]
    o += [note(340, 88, "ölçülen fark:", "#1e1e1e", 8.4)]
    o += [note(340, 106, "%93 token tasarrufu", "#2f9e44", 11)]
    o += [note(20, 140, "Gövde yalnız model `read` ile isteyince yüklenir — kademeli açığa çıkarma.")]
    return figure(600, 152, "".join(o))


def f_memory_tiers():
    p = Pen(42)
    rows = [("Instructions", "AGENTS.md", "yalnız insan yazar", "her oturum", "grey"),
            ("Curated core", "MEMORY.md", "kapılı konsolidasyon", "her oturum", "green"),
            ("Episodic", "memory/GG.md", "ajan çalışırken", "hiç — aranır", "blue"),
            ("Prospective", "SQLite", "intent tool", "tetiklenince", "orange"),
            ("Review", "DREAMS.md", "dreaming", "hiç — insan okur", "violet")]
    o = [note(16, 12, "KATMAN", "#868e96", 7), note(150, 12, "YÜZEY", "#868e96", 7),
         note(268, 12, "KİM YAZAR", "#868e96", 7), note(440, 12, "NE ZAMAN ENJEKTE", "#868e96", 7)]
    for i, (a, b, c, d, col) in enumerate(rows):
        y = 20 + i * 32
        o += [p.rect(12, y, 576, 26, col, width=1.2)]
        o += [text(20, y + 17, a, size=8.4, colour="#1e1e1e", weight="bold")]
        o += [mono(150, y + 17, b, size=7.4, colour="#454c53")]
        o += [note(268, y + 17, c)]
        o += [note(440, y + 17, d)]
    o += [note(12, 196, "Sınır ikinci ile üçüncü arasında: episodic'ten curated'a "
                        "hiçbir şey kapıdan geçmeden çıkmaz.", "#8a5208", 8)]
    return figure(600, 206, "".join(o))


def f_three_axes():
    p = Pen(51)
    o = [box(p, 14, 30, 178, 62, "SANDBOX", "tool NEREDE koşar", "blue", size=9)]
    o += [box(p, 210, 30, 178, 62, "TOOL POLICY", "HANGİ tool çağrılır", "orange", size=9)]
    o += [box(p, 406, 30, 178, 62, "ELEVATED", "sandbox'tan KAÇIŞ", "red", size=9)]
    o += [mono(24, 108, "sandbox.mode", size=7.4, colour="#767d84")]
    o += [mono(220, 108, "tools.allow / deny", size=7.4, colour="#767d84")]
    o += [mono(416, 108, "tools.elevated.*", size=7.4, colour="#767d84")]
    o += [note(14, 134, "deny HER ZAMAN kazanır · allow doluysa gerisi kapalı", "#c92a2a", 8.4)]
    o += [note(14, 150, "Ama: tool policy ADA göre filtreler — exec'in İÇİNİ görmez.",
                "#8a5208", 8.4)]
    o += [note(14, 164, "\"write'ı kapattık, artık read-only\" YANLIŞTIR.", "#c92a2a", 8.4)]
    return figure(600, 174, "".join(o))


def f_frozen_plan():
    p = Pen(52)
    o = [box(p, 14, 20, 128, 56, "istek", "argv + cwd + dosya", "blue", size=8)]
    o += [arrow(p, 144, 48, 194, 48)]
    o += [box(p, 196, 20, 132, 56, "PLAN DONDU", "hash alındı", "orange", size=8)]
    o += [arrow(p, 330, 48, 380, 48, "insan onayı")]
    o += [box(p, 382, 20, 96, 56, "onaylandı", "", "green", size=8)]
    o += [arrow(p, 480, 48, 522, 48)]
    o += [box(p, 524, 20, 64, 56, "koşar", "", "green", size=8)]
    o += [p.line(300, 96, 300, 118, "red", arrow=False)]
    o += [box(p, 196, 120, 200, 38, "argüman DEĞİŞTİ mi?", "", "red", size=8)]
    o += [note(410, 144, "→ approval mismatch, REDDEDİLİR", "#c92a2a", 8.4)]
    o += [note(14, 176, "Dosya onaydan sonra değişirse de reddedilir — kaymış içerik koşmaz.")]
    o += [note(14, 190, "Bizim approval.py bugün argüman hash'i TUTMUYOR. Açık iş.", "#8a5208")]
    return figure(600, 200, "".join(o))


def f_external_content():
    p = Pen(53)
    o = [box(p, 14, 26, 118, 50, "dış içerik", "e-posta · web · PDF", "red", size=8)]
    o += [arrow(p, 134, 50, 178, 50)]
    o += [panel(p, 180, 12, 176, 100, "SARMALAYICI", "orange", size=9)]
    o += [note(192, 44, "· rastgele id'li sınır")]
    o += [note(192, 60, "· 22 özel token silinir")]
    o += [note(192, 76, "· 28 homoglif katlanır")]
    o += [note(192, 92, "· 14 desen loglanır")]
    o += [arrow(p, 358, 62, 402, 62)]
    o += [box(p, 404, 26, 184, 50, "bağlam", "VERİ — talimat değil", "green", size=8.4)]
    o += [mono(14, 138, '<<<EXTERNAL_UNTRUSTED_CONTENT id="a3f9…">>>', size=7.6,
               colour="#454c53")]
    o += [note(14, 156, "id rastgele ÇÜNKÜ sabit olsaydı içerik kendi kapanışını yazıp "
                        "sarmalayıcıdan çıkardı.", "#c92a2a", 8)]
    return figure(600, 166, "".join(o))


def f_two_ledgers():
    p = Pen(54)
    o = [panel(p, 14, 24, 274, 92, "OPERASYONEL", "blue", size=9.2)]
    o += [note(28, 60, "best-effort · yalnız metadata")]
    o += [note(28, 76, "30 gün · 100.000 satır")]
    o += [note(28, 92, "kuyruk dolarsa DÜŞER, koşu sürer")]
    o += [panel(p, 312, 24, 274, 92, "UYUM ARŞİVİ", "green", size=9.2)]
    o += [note(326, 60, "kayıpsız · senkron")]
    o += [note(326, 76, "denetçiye gösterilir")]
    o += [note(326, 92, "yazılamazsa KOŞU DÜŞER", "#c92a2a")]
    o += [note(14, 140, "OpenClaw yalnız solu yapıyor ve bunu açıkça söylüyor:", "#1e1e1e", 8.4)]
    o += [note(14, 156, "\"Bir satırın yokluğu hiçbir şey kanıtlamaz.\"", "#c92a2a", 9)]
    o += [note(14, 172, "KKB'nin ihtiyacı sağ taraf. Ayrımı baştan yapmak ucuz, sonradan şema göçü.")]
    return figure(600, 182, "".join(o))


def f_atlas():
    p = Pen(55)
    o = [box(p, 200, 6, 200, 30, "kullanıcı · kurumsal SSO", "", "grey", size=8)]
    o += [p.line(300, 38, 300, 52, arrow=False, dash="3 3")]
    o += [note(410, 50, "SINIR 1 — kimlik", "#c92a2a", 7.4)]
    o += [box(p, 120, 54, 360, 42, "KONTROL DÜZLEMİ",
              "kapsam parametreden · rol = tool grubu · onay = donmuş plan", "orange",
              size=8.6)]
    o += [p.line(300, 98, 300, 112, arrow=False, dash="3 3")]
    o += [note(410, 110, "SINIR 2 — yetki", "#c92a2a", 7.4)]
    o += [box(p, 120, 114, 360, 40, "AJAN DÖNGÜSÜ (AutoGen)",
              "yetenek dizini cache sınırının üstünde", "blue", size=8.6)]
    o += [p.line(220, 156, 220, 170, arrow=False, dash="3 3")]
    o += [p.line(390, 156, 390, 170, arrow=False, dash="3 3")]
    o += [box(p, 120, 172, 176, 38, "tool / API", "sandbox", "green", size=8)]
    o += [box(p, 312, 172, 176, 38, "dış içerik", "sarmalayıcı", "red", size=8)]
    o += [note(14, 192, "SINIR 3", "#c92a2a", 7.4)]
    o += [note(500, 192, "SINIR 4", "#c92a2a", 7.4)]
    o += [box(p, 120, 224, 360, 34, "İKİ KAYIT HATTI + telemetri (içerik yok, boyut var)",
              "", "violet", size=8)]
    o += [p.line(300, 212, 300, 222, arrow=False, dash="3 3")]
    return figure(600, 266, "".join(o))


def f_thesis():
    p = Pen(61)
    o = [box(p, 20, 20, 250, 70, "MEKANİZMA", "kopyalanabilir", "blue", size=10)]
    o += [box(p, 330, 20, 250, 70, "GÜVEN MODELİ", "kopyalanamaz", "red", size=10)]
    o += [note(20, 114, "OpenClaw TEK bir güvenilen operatör etrafında tasarlanmış.")]
    o += [note(20, 130, "Atlas çok kullanıcılı ve karşılıklı güvenmeyen departmanlar içerecek.")]
    o += [note(20, 152, "Mekanizmalar taşınır. Güven modeli yeniden kurulur.", "#c92a2a", 9.4)]
    return figure(600, 164, "".join(o))


def f_task_stack():
    """The scheduling stack: what fires, what decides, what records, what runs."""
    p = Pen(71)
    o = [box(p, 8, 8, 128, 116, "", "", "grey")]
    o += [text(72, 26, "TETİKLEYİCİ", size=8.4, colour="#1e1e1e", weight="bold",
               anchor="middle")]
    for i, s in enumerate(("at · tek sefer", "every · aralık", "cron · ifade",
                           "on-exit · komut", "stream · satır", "webhook · dışarıdan")):
        o += [mono(18, 44 + i * 13, s, size=6.6, colour="#454c53")]
    o += [box(p, 160, 8, 132, 52, "AUTOMATIONS", "tam zamanlama", "blue", size=8.4)]
    o += [box(p, 160, 72, 132, 52, "HEARTBEAT", "~30 dk · main oturum", "violet",
              size=8.4)]
    o += [arrow(p, 138, 40, 158, 34), arrow(p, 138, 84, 158, 96)]
    o += [box(p, 316, 8, 124, 52, "TASK KAYDI", "queued→running→…", "green", size=8.4)]
    o += [arrow(p, 294, 34, 314, 34, "her koşu")]
    o += [p.line(294, 98, 380, 98, "red", arrow=False, dash="4 3")]
    o += [note(300, 112, "heartbeat task ÜRETMEZ", "#c92a2a", 6.8)]
    o += [box(p, 464, 8, 124, 52, "KUYRUK", "lane · FIFO", "orange", size=8.4)]
    o += [arrow(p, 442, 34, 462, 34)]
    o += [box(p, 464, 76, 124, 42, "yürütme", "izole / paylaşılan", "ink", size=8)]
    o += [arrow(p, 526, 62, 526, 74)]
    o += [note(8, 140, "Sıra soldan sağa: ne tetikler → kim karar verir → "
                       "ne kaydedilir → nasıl serileşir → nerede koşar.")]
    return figure(600, 152, "".join(o))


def f_task_lifecycle():
    """The five terminal states, and the one that is a measurement."""
    p = Pen(72)
    o = [box(p, 14, 62, 92, 40, "queued", "", "grey", size=8.6)]
    o += [box(p, 156, 62, 92, 40, "running", "", "blue", size=8.6)]
    o += [arrow(p, 108, 82, 154, 82, "ajan başladı")]
    states = [("succeeded", 6, "green"), ("failed", 46, "red"),
              ("timed_out", 86, "red"), ("cancelled", 126, "orange")]
    for name, y, col in states:
        o += [box(p, 320, y, 108, 32, name, "", col, size=7.8)]
        o += [arrow(p, 250, 82, 318, y + 16)]
    o += [box(p, 468, 86, 112, 32, "lost", "", "violet", size=7.8)]
    o += [arrow(p, 430, 102, 466, 102)]
    o += [note(468, 132, "5 dk grace sonrası", "#5f3dc4", 6.8)]
    o += [note(468, 144, "dayanak durumu yok", "#5f3dc4", 6.8)]
    o += [note(14, 124, "Terminal YAPIŞKAN: sonradan gelen", "#c92a2a", 7.2)]
    o += [note(14, 136, "başarı sinyali kararı düşürmez.", "#c92a2a", 7.2)]
    o += [note(14, 152, "Yürütme ile teslimat ayrı: succeeded olan bir iş "
                        "deliveryStatus=failed ise sonuç 'blocked'.")]
    return figure(600, 164, "".join(o))


# ─────────────────────────────────────────── the condensed decks ("hap") need
# a few diagrams the long deck told in prose. Each one carries a whole slide, so
# they are drawn wider and label more of themselves than the figures above.


def f_layers():
    """The three packages, and which question each answers."""
    p = Pen(81)
    rows = [("autogen_ext", "dış dünya", "model istemcileri · MCP · kod yürütücüler", "grey"),
            ("autogen_agentchat", "günlük iş", "AssistantAgent · 5 takım · 11 sonlandırma", "blue"),
            ("autogen_core", "aktör modeli", "AgentId(type,key) · runtime · topic · abonelik", "violet")]
    o = []
    for i, (name, what, detail, col) in enumerate(rows):
        y = 12 + i * 44
        o += [p.rect(20, y, 560, 36, col)]
        o += [mono(34, y + 16, name, size=9.4, colour="#1e1e1e")]
        o += [text(34, y + 30, what, size=7.6, colour="#454c53")]
        o += [text(210, y + 24, detail, size=8, colour="#454c53")]
    o += [note(20, 156, "Ayıran şey en alt katman: ajanlar gerçekten aktör — kendi "
                        "mailbox'ı olan, mesajı tipe göre yönlendiren birimler.")]
    o += [note(20, 170, "Yukarıdan başla; aşağı inmek her zaman mümkün.", "#2f9e44")]
    return figure(600, 182, "".join(o))


def f_termination():
    """Hard ceilings versus semantic conditions."""
    p = Pen(82)
    o = [box(p, 16, 22, 258, 96, "", "", "green")]
    o += [text(145, 42, "SERT TAVAN", size=9, colour="#1e1e1e", weight="bold", anchor="middle")]
    for i, s in enumerate(("MaxMessageTermination(20)", "TokenUsageTermination(50_000)",
                           "TimeoutTermination(300)")):
        o += [mono(30, 62 + i * 17, s, size=7.4, colour="#454c53")]
    o += [note(30, 112, "modelden bağımsız — her zaman tutar", "#2f9e44", 7)]

    o += [box(p, 326, 22, 258, 96, "", "", "red")]
    o += [text(455, 42, "ANLAMSAL KOŞUL", size=9, colour="#1e1e1e", weight="bold", anchor="middle")]
    for i, s in enumerate(('TextMentionTermination("BİTTİ")', "HandoffTermination(...)",
                           "FunctionCallTermination(...)")):
        o += [mono(340, 62 + i * 17, s, size=7.4, colour="#454c53")]
    o += [note(340, 112, "modelin işbirliğine bağlı — yazmazsa hiç tetiklenmez", "#c92a2a", 7)]

    o += [mono(150, 140, "termination = MaxMessageTermination(20) | TokenUsageTermination(50_000)",
               size=7.6, colour="#1e1e1e")]
    o += [note(16, 160, "| → biri yeterli   ·   & → hepsi gerekli", "#454c53")]
    o += [note(16, 174, "Üretimde en az bir SERT tavan olmadan koşmak, faturayı "
                        "modelin kararına bırakmaktır.", "#c92a2a")]
    return figure(600, 186, "".join(o))


def f_gotchas():
    """The four silent defaults, on one card each."""
    p = Pen(83)
    items = [("max_tool_iterations", "1", "model tool sonucunu GÖRMEDEN cevaplar"),
             ("model_context", "yok", "ajanın belleği yoktur"),
             ("model_client_stream", "False", "token akışı hiç yayılmaz"),
             ("sonlandırma", "yok", "takım tavansız koşar")]
    o = []
    for i, (name, val, effect) in enumerate(items):
        y = 10 + i * 38
        o += [p.rect(16, y, 568, 30, "red", width=1.3)]
        o += [mono(28, y + 13, name, size=8.2, colour="#1e1e1e")]
        o += [mono(28, y + 25, f"varsayılan: {val}", size=6.8, colour="#c92a2a")]
        o += [text(210, y + 19, effect, size=8.4, colour="#454c53")]
    o += [note(16, 176, "Dördünün ortak yanı: HİÇBİRİ HATA VERMEZ.", "#c92a2a", 9)]
    o += [note(16, 190, "Sistem çalışır, sonuç yanlış olur — ve aramak için önce "
                        "yanlış olduğunu bilmen gerekir.")]
    return figure(600, 200, "".join(o))


def f_oc_arch():
    """OpenClaw from above: channels in, gateway in the middle, capabilities out."""
    p = Pen(84)
    o = [box(p, 12, 20, 96, 108, "KANALLAR", "", "grey", size=8.4)]
    for i, s in enumerate(("web · CLI", "sohbet", "webhook", "node")):
        o += [note(24, 52 + i * 17, s, size=7)]
    o += [box(p, 158, 14, 150, 120, "GATEWAY", "", "orange", size=10)]
    for i, s in enumerate(("kimlik · eşleşme", "kapsam · politika", "oturum yönlendirme",
                           "zamanlayıcı", "denetim defteri")):
        o += [note(170, 48 + i * 16, s, size=7)]
    o += [arrow(p, 110, 74, 156, 74)]
    o += [box(p, 358, 12, 128, 40, "AJAN RUNTIME", "bağlam · döngü", "blue", size=8)]
    o += [box(p, 358, 60, 128, 34, "TOOL / SKILL", "44 · 74", "green", size=8)]
    o += [box(p, 358, 102, 128, 34, "BELLEK", "5 katman", "violet", size=8)]
    for ty in (32, 77, 119):
        o += [arrow(p, 310, 74, 356, ty)]
    o += [box(p, 522, 60, 64, 34, "model", "", "grey", size=7.6)]
    o += [arrow(p, 488, 32, 520, 70)]
    o += [note(12, 152, "Her şey Gateway'den geçiyor: kimlik, yetki, kayıt ve zamanlama "
                        "tek yerde. Ajan runtime bunların hiçbirini bilmiyor.")]
    return figure(600, 164, "".join(o))


def f_scopes():
    """Method scope is only the first gate."""
    p = Pen(85)
    o = [box(p, 16, 18, 120, 44, "istek", "metot + parametre", "grey", size=8)]
    o += [arrow(p, 138, 40, 178, 40)]
    o += [box(p, 180, 10, 140, 60, "KAPI 1", "metot kapsamı", "blue", size=8.6)]
    o += [note(180, 84, "read / write / admin …", "#2c4a6b", 7)]
    o += [arrow(p, 322, 40, 362, 40)]
    o += [box(p, 364, 10, 140, 60, "KAPI 2", "parametreden türetilen", "orange", size=8.6)]
    o += [note(364, 84, "/reset → admin", "#8a5208", 7)]
    o += [note(364, 96, "browser.proxy → admin", "#8a5208", 7)]
    o += [arrow(p, 506, 40, 546, 40)]
    o += [box(p, 548, 18, 44, 44, "koş", "", "green", size=8)]
    o += [note(16, 122, "Yetki yükseltme yasak: bir cihazı onaylamak yalnız çağıranın "
                        "ZATEN sahip olduğu kapsamları basabilir.", "#c92a2a")]
    o += [note(16, 138, "Ve sessiz genişleme yok — daha geniş rol isteyen yeniden "
                        "bağlanma, yeni bir bekleyen talep doğurur.")]
    return figure(600, 150, "".join(o))


def f_tool_search():
    """Big catalogue, small prompt."""
    p = Pen(86)
    o = [box(p, 16, 26, 200, 52, "YETENEK DİZİNİ", "≤ 18.000 karakter", "green", size=8.6)]
    o += [note(16, 92, "cache sınırının ÜSTÜNDE", "#2f9e44", 7.4)]
    o += [note(16, 106, "ada göre sıralı, deterministik", "#767d84", 7)]
    o += [note(16, 120, "kullanıcı mesajı GİRMEZ", "#c92a2a", 7)]
    o += [box(p, 262, 8, 96, 30, "search", "", "blue", size=7.8)]
    o += [box(p, 262, 46, 96, 30, "describe", "", "blue", size=7.8)]
    o += [box(p, 262, 84, 96, 30, "call", "", "blue", size=7.8)]
    o += [arrow(p, 218, 44, 260, 24), arrow(p, 218, 52, 260, 60), arrow(p, 218, 62, 260, 98)]
    o += [panel(p, 404, 8, 180, 66, "İZOLE NODE ALT SÜRECİ", "red", size=8)]
    o += [note(416, 44, "boş env · fs yok · ağ yok · sır yok", "#8c2f1d", 7)]
    o += [note(416, 58, "timeout 1.000–60.000 ms", "#8c2f1d", 7)]
    o += [box(p, 404, 88, 180, 30, "→ Gateway'e geri döner", "", "orange", size=7.8)]
    o += [arrow(p, 360, 99, 402, 103)]
    o += [note(404, 134, "politika · onay · hook · log burada işler", "#8a5208", 7.2)]
    o += [note(16, 148, "Fail-closed: politika dışı bir tool ARAMADA ÇIKMAZ. "
                        "Gizlemek yetmez, bulunamaz olmalı.", "#c92a2a")]
    return figure(600, 160, "".join(o))


def f_memory_write():
    """The write path is the security boundary."""
    p = Pen(87)
    o = [box(p, 14, 16, 108, 40, "içerik geldi", "", "grey", size=8)]
    o += [arrow(p, 124, 36, 164, 36)]
    o += [box(p, 166, 8, 150, 56, "KÖKEN ATANIR", "SQLite sütunu", "orange", size=8.4)]
    o += [note(166, 80, "owner · agent · untrusted · system", "#8a5208", 7.2)]
    o += [note(166, 94, "kapalı küme — model düzyazıyla yazamaz", "#767d84", 7)]
    o += [note(166, 108, "belirlenemezse → untrusted, ASLA owner", "#c92a2a", 7)]
    o += [arrow(p, 318, 36, 358, 36)]
    o += [box(p, 360, 8, 110, 56, "TERFİ KAPISI", "", "red", size=8.4)]
    o += [arrow(p, 472, 24, 512, 24)]
    o += [box(p, 514, 8, 76, 32, "curated", "", "green", size=7.8)]
    o += [p.cross(492, 56, 9)]
    o += [note(508, 60, "untrusted geçemez", "#c92a2a", 7)]
    o += [note(14, 132, "İki hijyen kuralı: cron/heartbeat/alt-ajan oturumları aday "
                        "ÜRETMEZ · bellekten enjekte edilen içerik yeniden ÇIKARILMAZ.")]
    o += [note(14, 148, "\"Yüz kez hatırlanan bir olgu tek bir olgu olarak kalır.\"", "#2f9e44")]
    return figure(600, 158, "".join(o))


def f_ctx_engine():
    """Four lifecycle points, and the rule that makes compaction correct."""
    p = Pen(88)
    pts = [("Ingest", "add_message", "mesaj eklendi"),
           ("Assemble", "get_messages", "bütçeye sığanı ver"),
           ("Compact", "compact()", "pencere doldu"),
           ("After turn", "after_turn", "tur bitti")]
    o = []
    for i, (name, fn, what) in enumerate(pts):
        x = 12 + i * 148
        o += [box(p, x, 16, 132, 50, name, fn, "blue", size=8.6)]
        o += [note(x, 82, what, size=7.2)]
        if i < 3:
            o += [arrow(p, x + 134, 41, x + 146, 41)]
    o += [p.rect(12, 104, 576, 40, "red", width=1.4)]
    o += [text(300, 122, "TOOL ÇAĞRISI SONUCUNDAN AYRILMAZ", size=9, colour="#c92a2a",
               weight="bold", anchor="middle")]
    o += [note(300, 136, "ayrılırsa: modelin görmediği bir çağrıya cevap → sağlayıcı diziyi reddeder",
               "#767d84", 7.2, anchor="middle")]
    return figure(600, 154, "".join(o))


def f_task_axes():
    """Three enums, because execution and delivery are not the same question."""
    p = Pen(89)
    cols_ = [("① YÜRÜTME", "status", ["queued", "running", "completed",
                                       "failed", "cancelled", "timed_out"], "blue"),
             ("② TESLİMAT", "deliveryStatus", ["pending", "delivered", "session_queued",
                                                "failed", "parent_missing", "…"], "orange"),
             ("③ SONUÇ", "terminalOutcome", ["succeeded", "blocked"], "green")]
    o = []
    for i, (title, field, vals, col) in enumerate(cols_):
        x = 14 + i * 194
        o += [p.rect(x, 12, 178, 118, col)]
        o += [text(x + 89, 30, title, size=8.6, colour="#1e1e1e", weight="bold", anchor="middle")]
        o += [mono(x + 89, 43, field, size=6.8, colour="#767d84", anchor="middle")]
        for j, v in enumerate(vals):
            o += [mono(x + 14, 60 + j * 12, v, size=6.8, colour="#454c53")]
    o += [note(14, 148, "completed + parent_missing  →  blocked, FAILED DEĞİL", "#c92a2a", 8.6)]
    o += [note(14, 164, "\"Bu, tamamlanmış sonucu korur; çocuk yürütmesini yanlışlıkla "
                        "başarısız diye raporlamak yerine.\"")]
    return figure(600, 174, "".join(o))


def f_threads():
    """Three concurrency layers, each owning a different failure mode."""
    p = Pen(90)
    o = [panel(p, 12, 14, 184, 74, "① EVENT LOOP", "blue", size=9)]
    o += [note(24, 42, "tek iş parçacığı", "#767d84", 7)]
    o += [note(24, 56, "ajan işi: model · tool · oturum", size=7)]
    o += [note(24, 70, "paralellik G/Ç beklemesinden", size=7)]
    o += [panel(p, 208, 14, 184, 74, "② WORKER THREAD", "green", size=9)]
    o += [mono(218, 48, "audit-event-writer", size=6.6, colour="#454c53")]
    o += [mono(218, 60, "sqlite-archive", size=6.6, colour="#454c53")]
    o += [mono(218, 72, "transcript-reconcile", size=6.6, colour="#454c53")]
    o += [panel(p, 404, 14, 184, 74, "③ ÇOCUK SÜREÇ", "orange", size=9)]
    o += [note(416, 42, "ProcessSupervisor", "#767d84", 7)]
    o += [note(416, 56, "on-exit / stream watcher", size=7)]
    o += [note(416, 70, "gateway'e ait, TURA DEĞİL", size=7, colour="#8a5208")]
    o += [note(12, 108, "② neden var: cevap yolunda olmaması gereken defter işleri. "
                        "Kuyruk dolarsa KAYIT düşer, KOŞU düşmez.")]
    o += [note(12, 124, "Bu bir geliştirici aracı için doğru öncelik — düzenlenmiş bir "
                        "kurumda uyum hattında TERSİ olmalı.", "#c92a2a")]
    return figure(600, 134, "".join(o))


def f_durable():
    """What survives a restart, and the three details that make it real."""
    p = Pen(91)
    rows = [("konuşma geçmişi", "ajan SQLite", "dokunulmaz"),
            ("yarıda kalan tur", "oturum satırı", "otomatik devam"),
            ("subagent koşuları", "paylaşılan SQLite", "boot'ta geri yüklenir"),
            ("zamanlanmış işler", "paylaşılan SQLite", "tanım + geçmiş korunur"),
            ("giden mesajlar", "SQLite", "drenaj edilir")]
    o = [note(20, 14, "DURUM", "#868e96", 7), note(230, 14, "DEPOLAMA", "#868e96", 7),
         note(400, 14, "YENİDEN BAŞLATMADA", "#868e96", 7)]
    for i, (a, b, c) in enumerate(rows):
        y = 22 + i * 24
        o += [p.rect(16, y, 568, 19, "green", width=1.1)]
        o += [text(24, y + 13, a, size=7.8, colour="#1e1e1e")]
        o += [mono(230, y + 13, b, size=7, colour="#454c53")]
        o += [text(400, y + 13, c, size=7.8, colour="#454c53")]
    o += [note(16, 158, "① Gecikmiş işler YENİDEN ZAMANLANIR, anında oynatılmaz — "
                        "soğuk başlangıçta yük tepesi olmasın.")]
    o += [note(16, 172, "② Kurtarma sınırlı: tekrar tekrar düşen oturum karantinaya alınır.")]
    o += [note(16, 186, "③ Uzlaştırma iki kanıtlı: önce runtime, 5 dk grace sonrası "
                        "dayanıklı koşu geçmişi.")]
    return figure(600, 196, "".join(o))


def f_secrets():
    """The sentinel: the real value appears only at the last boundary."""
    p = Pen(92)
    o = [box(p, 14, 30, 116, 44, "SecretRef", "config'te sır yok", "green", size=8)]
    o += [arrow(p, 132, 52, 168, 52)]
    o += [box(p, 170, 24, 150, 56, "SENTINEL", "oc-sent-v1-…", "orange", size=8.6)]
    o += [note(170, 96, "auth deposu · SDK · loglar", "#8a5208", 7)]
    o += [note(170, 110, "hata nesneleri bunu görür", "#8a5208", 7)]
    o += [arrow(p, 322, 52, 358, 52)]
    o += [box(p, 360, 24, 116, 56, "EGRESS", "son adaptör", "red", size=8.4)]
    o += [note(360, 96, "gerçek değer burada", "#c92a2a", 7)]
    o += [arrow(p, 478, 52, 512, 52)]
    o += [box(p, 514, 30, 72, 44, "sağlayıcı", "", "grey", size=7.6)]
    o += [note(14, 138, "Bilinmeyen sentinel-şekilli değer AĞ ETKİNLİĞİNDEN ÖNCE fail-closed: "
                        "çözülmemişi iletmektense istek gönderilmez.", "#c92a2a")]
    o += [note(14, 154, "Ve sınır beyanı: \"Sentinel'ler süreç izolasyonu DEĞİLDİR\" — "
                        "gerçek değer aynı süreçte bellekte.")]
    return figure(600, 164, "".join(o))


# ──────────────────────────────────────────────── harness internals ("niş")
# These draw implementation, not concept: what the code actually does to a
# malformed tool call, a noisy result, a halted workflow.


def f_packages():
    p = Pen(101)
    groups = [("ai · llm-core · model-catalog-core", "sağlayıcı adaptörleri, akış runtime'ı", "blue", 118),
              ("gateway-protocol · gateway-client · sdk", "tipli şema + doğrulayıcı + istemci", "violet", 108),
              ("memory-host-sdk", "bellek sağlayıcı sözleşmesi", "green", 83),
              ("plugin-sdk · plugin-package-contract", "eklenti yüzeyi", "orange", 25),
              ("terminal-core · markdown-core · media-*", "çıktı biçimlendirme", "grey", 21),
              ("tool-call-repair · retry · net-policy", "sessiz altyapı", "red", 6)]
    o = []
    for i, (name, what, col, n) in enumerate(groups):
        y = 10 + i * 26
        o += [p.rect(14, y, 570, 21, col, width=1.2)]
        o += [mono(24, y + 14, name, size=7, colour="#1e1e1e")]
        o += [text(300, y + 14, what, size=7.4, colour="#454c53")]
        o += [mono(560, y + 14, f"{n}", size=7, colour="#767d84", anchor="end")]
    o += [note(14, 178, "22 paket · dosya sayıları en büyük modülü gösteriyor. "
                        "Çekirdeğin her ilginç parçası ayrı paket — sıkıştırma bile.")]
    return figure(600, 188, "".join(o))


def f_repair():
    p = Pen(102)
    o = [panel(p, 14, 10, 168, 86, "MODEL DÜZ METİN YAZDI", "red", size=8)]
    for i, s in enumerate(("[END_TOOL_REQUEST]", "<|channel|> <|message|>",
                           "<|call|>", "<function=ad>…</function>")):
        o += [mono(24, 40 + i * 15, s, size=6.4, colour="#8c2f1d")]
    o += [arrow(p, 184, 53, 220, 53)]
    o += [box(p, 222, 8, 140, 40, "parse", "grammar.ts", "blue", size=8)]
    o += [box(p, 222, 56, 140, 40, "promote", "promote.ts", "orange", size=8)]
    o += [arrow(p, 364, 76, 400, 76)]
    o += [box(p, 402, 56, 182, 40, "gerçek tool çağrısı", "ada göre çözümlenmiş", "green", size=8)]
    o += [note(402, 20, "ad çözümleyici: modelin yazdığı adı", "#2f9e44", 7)]
    o += [note(402, 32, "SAĞLAYICININ İZİN VERDİĞİ ada eşler", "#2f9e44", 7)]
    o += [note(14, 116, "Neden var: bazı modeller function calling'i düzgün üretmiyor, "
                        "tool çağrısını metin olarak yazıyor.")]
    o += [note(14, 130, "Onarmadan atarsan tur boşa gider ve sebebi görünmez. "
                        "Bu 6 dosyalık paket o turu kurtarıyor.", "#8a5208")]
    return figure(600, 142, "".join(o))


def f_result_middleware():
    p = Pen(103)
    o = [box(p, 14, 34, 104, 44, "exec / bash", "", "grey", size=8)]
    o += [arrow(p, 120, 56, 156, 56)]
    o += [box(p, 158, 26, 150, 60, "TOOL SONUCU", "40.000 satır log", "red", size=8.4)]
    o += [arrow(p, 310, 56, 346, 56)]
    o += [box(p, 348, 26, 150, 60, "middleware", "tokenjuice", "orange", size=8.4)]
    o += [arrow(p, 500, 56, 536, 56)]
    o += [box(p, 538, 34, 52, 44, "bağlam", "", "green", size=7.6)]
    o += [note(158, 104, "komut ZATEN koştu", "#c92a2a", 7.2)]
    o += [note(348, 104, "yalnız tool_result değişir", "#8a5208", 7.2)]
    o += [note(348, 116, "komut yeniden koşmaz, exit kodu değişmez", "#767d84", 7)]
    o += [note(14, 140, "Ayrım önemli: bu bir çıktı KISALTMA katmanı, bir komut "
                        "değiştirme katmanı değil. Gürültülü log bağlamı yemeden önce kesiliyor.")]
    return figure(600, 152, "".join(o))


def f_lobster():
    p = Pen(104)
    o = [note(16, 14, "LOBSTER OLMADAN — her adım bir model turu", "#c92a2a", 8)]
    for i in range(4):
        x = 16 + i * 96
        o += [box(p, x, 26, 78, 26, f"adım {i+1}", "", "red", size=7.4)]
        if i < 3:
            o += [arrow(p, x + 80, 39, x + 94, 39)]
    o += [note(410, 42, "4 tur · 4 bağlam · 4 fatura", "#c92a2a", 7.4)]

    o += [note(16, 76, "LOBSTER İLE — tek tool çağrısı", "#2f9e44", 8)]
    o += [box(p, 16, 88, 368, 44, "tipli boru hattı", "tek yapılandırılmış sonuç", "green",
              size=8.4)]
    o += [box(p, 404, 88, 88, 44, "ONAY", "yan etki", "orange", size=8)]
    o += [box(p, 508, 88, 78, 44, "resume", "token", "violet", size=8)]
    o += [arrow(p, 386, 110, 402, 110), arrow(p, 494, 110, 506, 110)]
    o += [note(16, 150, "Yan etki (gönder, sil, yayınla) akışı DURDURUYOR ve bir "
                        "devam token'ı döndürüyor.")]
    o += [note(16, 164, "Onayla ve devam et — baştan başlamadan. Onay kapıları "
                        "runtime'ın parçası, model kararı değil.", "#2f9e44")]
    return figure(600, 176, "".join(o))


def f_session_tools():
    p = Pen(105)
    items = [("/steer", "koşan turu yönlendir", "kabul edilmezse normal prompt olarak gider", "blue"),
             ("/btw", "yan soru", "geçmişe EKLENMEZ · aynı model · tek atış", "violet"),
             ("/goal", "oturuma bağlı hedef", "kalıcı · operatör ve model aynı hedefi görür", "green"),
             ("/loop", "kendi kendini tekrarlayan iş", "sahibe özel · konuşmaya bağlı", "orange")]
    o = []
    for i, (cmd, what, detail, col) in enumerate(items):
        y = 10 + i * 34
        o += [p.rect(14, y, 570, 27, col, width=1.2)]
        o += [mono(26, y + 18, cmd, size=8.4, colour="#1e1e1e")]
        o += [text(112, y + 17, what, size=8, colour="#1e1e1e")]
        o += [text(268, y + 17, detail, size=7.4, colour="#454c53")]
    o += [note(14, 158, "Dördü de aynı fikrin farklı yüzü: BİR TURA MÜDAHALE ETMEK. "
                        "Klasik sohbet arayüzünde bunların hiçbiri yok.")]
    return figure(600, 168, "".join(o))


def f_self_learning():
    p = Pen(106)
    o = [box(p, 14, 30, 116, 44, "düzeltme", "\"öyle değil, böyle\"", "red", size=8)]
    o += [arrow(p, 132, 52, 168, 52)]
    o += [box(p, 170, 22, 160, 60, "SKILL WORKSHOP", "öneri → tarama → uygula", "orange",
              size=8.4)]
    o += [arrow(p, 332, 52, 368, 52)]
    o += [box(p, 370, 30, 116, 44, "skill", "SKILL.md", "green", size=8)]
    o += [arrow(p, 488, 52, 524, 52)]
    o += [box(p, 526, 30, 60, 44, "indeks", "", "grey", size=7.6)]
    o += [note(170, 100, "aynı yönetilen yol: açık skill yazımı da buradan geçer", "#8a5208", 7.2)]
    o += [note(14, 130, "Kalıcı birim SKILL — bir bellek satırı değil. Fark: skill bir "
                        "PROSEDÜR, gelecekteki oturumlar bulup izleyebiliyor.")]
    o += [note(14, 146, "Ve kapıdan geçiyor: öğrenilen şey doğrudan davranışa yazılmıyor, "
                        "önerilip taranıp uygulanıyor.", "#2f9e44")]
    return figure(600, 158, "".join(o))


def f_trajectory():
    p = Pen(107)
    o = [box(p, 14, 20, 130, 96, "OTURUM", "uçuş kayıt cihazı", "blue", size=8.6)]
    o += [box(p, 190, 8, 168, 26, "prompt + sistem prompt", "", "grey", size=7.4)]
    o += [box(p, 190, 40, 168, 26, "modele giden tool'lar", "", "grey", size=7.4)]
    o += [box(p, 190, 72, 168, 26, "tool çağrı/sonuç zinciri", "", "grey", size=7.4)]
    o += [box(p, 190, 104, 168, 26, "zamanlama ve hatalar", "", "grey", size=7.4)]
    for ty in (21, 53, 85, 117):
        o += [arrow(p, 146, 68, 188, ty)]
    o += [arrow(p, 360, 68, 396, 68)]
    o += [box(p, 398, 44, 186, 48, "/export-trajectory", "REDAKTE destek paketi", "green",
              size=8.4)]
    o += [note(14, 144, "Bir hata raporunun \"ne oldu\" sorusunu prompt'u elle "
                        "kopyalamadan cevaplıyor — ve redakte edilmiş çıkıyor.")]
    return figure(600, 154, "".join(o))


def f_failover():
    p = Pen(108)
    o = [box(p, 14, 20, 112, 44, "profil 1", "birincil", "green", size=8)]
    o += [arrow(p, 128, 42, 160, 42, "hata")]
    o += [box(p, 162, 20, 112, 44, "profil 2", "yedek", "blue", size=8)]
    o += [arrow(p, 276, 42, 308, 42)]
    o += [box(p, 310, 20, 112, 44, "profil 3", "", "grey", size=8)]
    o += [box(p, 452, 12, 134, 30, "cooldown", "", "orange", size=7.8)]
    o += [box(p, 452, 48, 134, 30, "auth-hata önbelleği", "", "orange", size=7.4)]
    o += [box(p, 452, 84, 134, 30, "faturalama kilidi", "", "red", size=7.4)]
    o += [note(14, 92, "OTURUM YAPIŞKANLIĞI", "#2f9e44", 8)]
    o += [note(14, 106, "aynı oturum aynı profilde kalır", "#767d84", 7.2)]
    o += [note(14, 118, "gerekçe açıkça yazılı: cache-friendly", "#2f9e44", 7.2)]
    o += [note(14, 144, "Model değiştirmek ucuz değil — prompt cache'ini yakıyor. "
                        "Bu yüzden gereksiz yere değiştirilmiyor.")]
    return figure(600, 156, "".join(o))


def f_loopguard():
    p = Pen(109)
    o = [note(16, 14, "ZİNCİR: bağlam taşması → sıkıştırma → aynı döngü → taşma", "#c92a2a", 8.4)]
    for i, s in enumerate(("taşma", "sıkıştırma", "aynı tool", "taşma")):
        x = 16 + i * 116
        o += [box(p, x, 28, 96, 30, s, "", "red", size=7.6)]
        if i < 3:
            o += [arrow(p, x + 98, 43, x + 114, 43)]
    o += [p.curve([(492, 43), (520, 20), (300, 8), (60, 20), (60, 26)], "red", arrow=True)]
    o += [box(p, 16, 78, 240, 44, "NÖBETÇİ", "sıkıştırma sonrası kısa pencere", "green",
              size=8.4)]
    o += [note(16, 134, "aynı (tool, args, result) üçlüsü tekrarlanırsa → koşu iptal", "#2f9e44", 7.6)]
    o += [box(p, 300, 78, 284, 44, "hash oynak metadata'yı yok sayar", "süre · PID · cwd",
              "blue", size=8)]
    o += [note(300, 134, "giden mesajda TERSİ: oynak id'ler çıkarılır", "#2c4a6b", 7.6)]
    o += [note(16, 158, "İki varsayılan bilerek farklı: agresif olan (rolling tespit) "
                        "KAPALI, ucuz ve yüksek getirili olan (nöbetçi) AÇIK.")]
    return figure(600, 170, "".join(o))


def f_tool_catalog():
    """All 51 built-in tools, by the group that owns them.

    Drawn from `src/agents/tool-catalog.ts`'s `CORE_TOOL_DEFINITIONS`, which is
    where the group map is built — not from the docs table, which has drifted
    (it omits `agents_wait`, `dashboard`, `mobile_ui` and still lists a `cron`
    tool under automation).

    The aspect ratio is deliberate. Eleven stacked rows at a comfortable row
    height make a tall figure, and `card()` sizes a figure by its *height*, so
    a tall drawing renders narrow and its labels become unreadable. Short rows
    keep the box wide, which is what makes 6 pt monospace legible on the page.
    """
    p = Pen(110)
    groups = [
        ("fs", 4, "read  write  edit  apply_patch", "orange"),
        ("runtime", 3, "exec  process  code_execution", "red"),
        ("web", 3, "web_search  web_fetch  x_search", "blue"),
        ("memory", 2, "memory_search  memory_get", "green"),
        ("sessions", 15, "sessions{,_list,_history,_search,_send,_spawn,_yield}  "
                         "conversations_{list,send,turn}  agents_wait  subagents  "
                         "session_status  {spawn,dismiss}_task"[:138], "violet"),
        ("ui", 6, "browser  screen  dashboard  terminal  canvas  show_widget", "blue"),
        ("messaging", 1, "message", "green"),
        ("automation", 2, "heartbeat_respond  gateway", "orange"),
        ("nodes", 3, "nodes  computer  mobile_ui", "grey"),
        ("agents", 7, "agents_list  get_goal  create_goal  update_goal  update_plan  "
                      "ask_user  skill_workshop", "violet"),
        ("media", 5, "image  image_generate  music_generate  video_generate  tts", "red"),
    ]
    o = []
    for i, (name, n, ids, col) in enumerate(groups):
        y = 5 + i * 12
        o += [p.rect(10, y, 580, 10, col, width=1.0)]
        o += [mono(17, y + 7.6, f"group:{name}", size=6.4, colour="#1e1e1e")]
        o += [mono(112, y + 7.6, str(n), size=6.4, colour="#767d84", anchor="end")]
        size = 6.0 if len(ids) <= 138 else 5.4
        o += [mono(122, y + 7.6, ids, size=size, colour="#454c53")]
    return figure(600, 142, "".join(o))


def f_profiles():
    """Four profiles, and the two layers that narrow them further."""
    p = Pen(111)
    profs = [("minimal", "en dar", "grey"), ("coding", "fs + runtime", "orange"),
             ("messaging", "mesaj + oturum", "green"), ("full", "hepsi", "blue")]
    o = []
    for i, (name, what, col) in enumerate(profs):
        x = 14 + i * 145
        o += [box(p, x, 14, 130, 44, name, what, col, size=8.4)]
    o += [note(14, 76, "① PROFİL — taban allowlist", "#1e1e1e", 8)]
    o += [p.line(300, 84, 300, 100, arrow=False, dash="3 3")]
    o += [box(p, 14, 104, 280, 34, "② tools.allow / tools.deny", "genel + ajan başına",
              "violet", size=8)]
    o += [box(p, 306, 104, 280, 34, "③ tools.sandbox.tools.*", "yalnız sandbox'tayken",
              "red", size=8)]
    o += [note(14, 158, "Üçü de daraltır, hiçbiri genişletmez. deny her zaman kazanır; "
                        "allow doluysa gerisi kapalı.")]
    o += [note(14, 174, "byProvider ile model başına ayrı politika da var: "
                        "tools.byProvider[anthropic].deny gibi.", "#767d84")]
    return figure(600, 186, "".join(o))


def f_patterns():
    """The eight sections under `Multi-Agent Design Patterns` in the core guide."""
    p = Pen(112)
    rows = [("Concurrent Agents", "3236", "tek yayın → çok dal → toplayıcı", "blue"),
            ("Sequential Workflow", "3504", "her ajan bir sonrakine devrediyor", "green"),
            ("Group Chat", "3772", "bir yönetici konuşma sırasını dağıtıyor", "violet"),
            ("Handoffs", "4349", "ajan işi kendisi devrediyor", "orange"),
            ("Mixture of Agents", "4989", "aynı soru, farklı uzmanlar, birleştirici", "blue"),
            ("Multi-Agent Debate", "5358", "birden çok tur karşılıklı eleştiri", "violet"),
            ("Reflection", "5822", "üretici + eleştirmen, kalite döngüsü", "green"),
            ("Code Execution", "6188", "modelin yazdığı kod yürütücüde koşuyor", "red")]
    o = []
    for i, (name, line, what, col) in enumerate(rows):
        y = 6 + i * 17
        o += [p.rect(12, y, 576, 14, col, width=1.1)]
        o += [text(22, y + 10, name, size=7.6, colour="#1e1e1e", weight="bold")]
        o += [mono(180, y + 10, f"05:{line}", size=6.6, colour="#767d84")]
        o += [text(248, y + 10, what, size=7.4, colour="#454c53")]
    o += [note(12, 158, "Bu, core kılavuzunun KENDİ bölümlemesi — bizim tasnifimiz "
                        "değil. Satır numarası verilmesinin sebebi bu.")]
    return figure(600, 168, "".join(o))


def f_components():
    """The Components Guide: five surfaces plus the serialisation contract."""
    p = Pen(113)
    comps = [("Model Clients", "1984", "sağlayıcıya konuşan şey", "blue"),
             ("Model Context", "2341", "modele NE gideceğine karar veren şey", "violet"),
             ("Tools", "2473", "modelin çağırabildiği fonksiyonlar", "green"),
             ("Workbench (+ MCP)", "2841", "tool'ların toplandığı arayüz", "orange"),
             ("Code Executors", "3054", "kodu nerede koşturacağın", "red")]
    o = []
    for i, (name, line, what, col) in enumerate(comps):
        y = 8 + i * 24
        o += [p.rect(12, y, 380, 20, col, width=1.2)]
        o += [text(22, y + 14, name, size=8, colour="#1e1e1e", weight="bold")]
        o += [mono(160, y + 14, f"05:{line}", size=6.6, colour="#767d84")]
        o += [text(216, y + 14, what, size=7.4, colour="#454c53")]
    o += [panel(p, 412, 8, 176, 116, "Component config", "grey", size=8.4)]
    o += [mono(500, 30, "05:1888", size=6.6, colour="#767d84", anchor="middle")]
    o += [note(424, 56, "dump_component / load_component", "#454c53", 6.6)]
    o += [note(424, 70, "her bileşen JSON'a yazılıp", "#767d84", 7)]
    o += [note(424, 82, "geri yüklenebiliyor", "#767d84", 7)]
    o += [note(424, 100, "→ yapılandırma kod değil VERİ", "#2f9e44", 7)]
    o += [note(12, 146, "Beşi de değiştirilebilir yüzey. Bir ajan bunların hangi "
                        "uygulamasıyla konuştuğunu bilmiyor — kapıyı kurmayı mümkün "
                        "kılan da bu.")]
    return figure(600, 156, "".join(o))


# ──────────────────────────────── one figure per component / per pattern, so
# the condensed AutoGen deck can give each its own page.


def f_model_clients():
    p = Pen(120)
    o = [box(p, 190, 8, 220, 34, "ChatCompletionClient", "protokol sınıfı", "violet",
             size=8.6)]
    impls = [("OpenAIChatCompletionClient", "OpenAI + uyumlu (Gemini…)", "blue"),
             ("AzureOpenAIChatCompletionClient", "Azure OpenAI", "blue"),
             ("AzureAIChatCompletionClient", "GitHub + Azure barındırılan", "blue"),
             ("ReplayChatCompletionClient", "deterministik kuru mod", "green")]
    for i, (name, what, col) in enumerate(impls):
        y = 62 + i * 24
        o += [p.rect(60, y, 480, 19, col, width=1.1)]
        o += [mono(70, y + 13, name, size=6.8, colour="#1e1e1e")]
        o += [text(300, y + 13, what, size=7.2, colour="#454c53")]
        if i == 0:
            o += [arrow(p, 300, 44, 300, 58)]
    o += [note(60, 176, "Hepsi aynı protokolü uyguluyor — ajan hangisiyle konuştuğunu "
                        "bilmiyor. Kuru mod istemcisi de bir istemci.", "#454c53")]
    return figure(600, 186, "".join(o))


def f_tools_component():
    p = Pen(121)
    o = [box(p, 14, 24, 132, 46, "Python fonksiyonu", "tip ipuçları + docstring",
             "grey", size=8)]
    o += [arrow(p, 148, 47, 184, 47)]
    o += [box(p, 186, 16, 140, 62, "FunctionTool", "şema TÜRETİLİYOR", "green", size=8.4)]
    o += [arrow(p, 328, 47, 364, 47)]
    o += [box(p, 366, 24, 108, 46, "modele giden", "JSON şema", "blue", size=8)]
    o += [arrow(p, 476, 47, 512, 47)]
    o += [box(p, 514, 24, 74, 46, "çağrı", "", "orange", size=8)]
    o += [note(186, 96, "docstring = modelin NE ZAMAN çağıracağına karar verdiği metin",
               "#2f9e44", 7.4)]
    o += [note(186, 110, "tip ipucu yoksa model ne göndereceğini bilemiyor", "#c92a2a", 7.4)]
    o += [note(14, 138, "Tool bir kod parçası; ajan onu modelin ürettiği fonksiyon "
                        "çağrısına karşılık koşturuyor. Yazdığın açıklama, arayüzün "
                        "kendisi.")]
    return figure(600, 150, "".join(o))


def f_workbench_component():
    p = Pen(122)
    o = [box(p, 210, 10, 180, 40, "Workbench", "list_tools / call_tool", "violet",
             size=8.6)]
    o += [box(p, 20, 84, 160, 44, "StaticWorkbench", "elindeki fonksiyonlar", "green",
              size=8)]
    o += [box(p, 220, 84, 160, 44, "McpWorkbench", "stdio ya da HTTP", "blue", size=8)]
    o += [box(p, 420, 84, 160, 44, "GatedWorkbench", "bizim — kapı", "orange", size=8)]
    for x in (100, 300, 500):
        o += [arrow(p, x, 82, x, 54)]
    o += [note(20, 148, "Tool tek bir arayüz; workbench BİR KOLEKSİYON — durum ve "
                        "kaynak paylaşan tool'lar, tek tip sonuç.")]
    o += [note(20, 164, "Ajan hangisiyle konuştuğunu bilmiyor. Kapıyı araya koymayı "
                        "mümkün kılan tek şey bu.", "#8a5208")]
    return figure(600, 176, "".join(o))


def f_code_executors():
    p = Pen(123)
    o = [box(p, 16, 30, 118, 44, "kod bloğu", "modelden", "grey", size=8)]
    o += [arrow(p, 136, 52, 172, 52)]
    o += [box(p, 174, 8, 190, 40, "LocalCommandLine…", "host makinede", "red", size=8)]
    o += [box(p, 174, 60, 190, 40, "DockerCommandLine…", "konteynerde", "green", size=8)]
    o += [arrow(p, 366, 28, 402, 28), arrow(p, 366, 80, 402, 80)]
    o += [box(p, 404, 8, 184, 40, "→ host'un her şeyi", "", "red", size=7.8)]
    o += [box(p, 404, 60, 184, 40, "→ yalıtılmış", "", "green", size=7.8)]
    o += [note(16, 122, "Her kod bloğu bir dosyaya yazılıp AYRI BİR SÜREÇTE koşuyor — "
                        "yani bloklar arası değişken paylaşımı yok.")]
    o += [note(16, 138, "Local, modelin yazdığı kodu makinende koşturur. Bu bir "
                        "tercih değil, bir güven kararıdır.", "#c92a2a")]
    return figure(600, 150, "".join(o))


def f_component_config():
    p = Pen(124)
    o = [panel(p, 14, 20, 250, 100, "COMPONENT CONFIG", "green", size=9.2)]
    o += [note(28, 54, "· nesnenin PLANI", "#14594a", 7.6)]
    o += [note(28, 70, "· defalarca damgalanabilir", "#14594a", 7.6)]
    o += [note(28, 86, "· dump_component / load_component", "#454c53", 7)]
    o += [note(28, 102, "· AutoGen dışı bileşenler de", "#767d84", 7)]
    o += [panel(p, 336, 20, 250, 100, "STATE", "blue", size=9.2)]
    o += [note(350, 54, "· nesnenin KENDİSİ", "#2c4a6b", 7.6)]
    o += [note(350, 70, "· mesaj geçmişi dahil HER ŞEY", "#2c4a6b", 7.6)]
    o += [note(350, 86, "· save_state / load_state", "#454c53", 7)]
    o += [note(350, 102, "· geri yüklenen TAM AYNI nesne", "#767d84", 7)]
    o += [note(14, 144, "Kılavuzun kendi vurgusu: config bir <plan>, state bir "
                        "<fotoğraf>. Aynı config'ten yüz örnek çıkar; aynı state'ten "
                        "tek bir nesne.")]
    o += [note(14, 160, "AutoGen Studio'nun yapılandırma tabanlı deneyimi bunun üstünde.",
               "#767d84")]
    return figure(600, 172, "".join(o))


def f_sequential():
    p = Pen(125)
    stages = [("Concept\nExtractor", "özellik · hedef kitle"), ("Writer", "pazarlama metni"),
              ("Format &\nProof", "dilbilgisi · ton"), ("User", "sunum")]
    o = []
    for i, (name, what) in enumerate(stages):
        x = 14 + i * 148
        o += [box(p, x, 26, 124, 44, name.replace("\n", " "), "", "blue", size=8)]
        o += [note(x, 84, what, size=7)]
        if i < 3:
            o += [arrow(p, x + 126, 48, x + 144, 48)]
    o += [note(14, 116, "Sıra DETERMİNİSTİK: her ajan bir alt görevi yapıp bir "
                        "sonrakine devrediyor. Kim konuşacak diye kimse karar vermiyor.")]
    o += [note(14, 132, "core'da bu, her ajanın bir sonrakinin topic'ine yayın "
                        "yapmasıyla kuruluyor — zincir aboneliklerde yazılı.", "#767d84")]
    return figure(600, 144, "".join(o))


def f_groupchat():
    p = Pen(126)
    o = [box(p, 232, 12, 136, 40, "ORTAK TOPIC", "hepsi abone + yayıncı", "orange",
             size=8.4)]
    roles = [("yazar", 26, 96), ("çizer", 180, 110), ("editör", 334, 110), ("insan", 470, 96)]
    for name, x, y in roles:
        o += [box(p, x, y, 104, 34, name, "", "blue", size=8)]
        o += [arrow(p, x + 52, y, 300, 56)]
    o += [box(p, 232, 156, 136, 30, "yönetici", "sırayı dağıtır", "violet", size=7.8)]
    o += [note(14, 176, "Tek bir mesaj dizisi: herkes aynı konuşmayı görüyor.", "#454c53")]
    o += [note(14, 192, "Bedeli: bağlam herkes için aynı ve büyük. Ayırmaya değer bir "
                        "bağlam varsa yanlış desen.", "#c92a2a")]
    return figure(600, 204, "".join(o))


def f_handoffs():
    p = Pen(127)
    o = [box(p, 16, 40, 128, 46, "triyaj ajanı", "", "blue", size=8.4)]
    o += [arrow(p, 146, 52, 196, 32, "transfer_to_x")]
    o += [arrow(p, 146, 74, 196, 96)]
    o += [box(p, 198, 14, 128, 38, "iade ajanı", "", "green", size=8)]
    o += [box(p, 198, 78, 128, 38, "satış ajanı", "", "green", size=8)]
    o += [note(350, 30, "devretme ÖZEL BİR TOOL ÇAĞRISI", "#8a5208", 7.6)]
    o += [note(350, 46, "modelin kendi kararı — dışarıdan", "#767d84", 7.2)]
    o += [note(350, 58, "bir yönlendirici yok", "#767d84", 7.2)]
    o += [note(350, 82, "ölçülen: en pahalı desen, 334 token", "#c92a2a", 7.6)]
    o += [note(350, 96, "her devirde bağlam yeniden kuruluyor", "#767d84", 7.2)]
    o += [note(16, 140, "OpenAI'ın Swarm projesinden geliyor. AutoGen'in eklediği: "
                        "dağıtık runtime'a ölçeklenebilmesi ve kendi ajan "
                        "uygulamanı getirebilmen.")]
    return figure(600, 152, "".join(o))


def f_mixture():
    p = Pen(128)
    o = [box(p, 14, 62, 92, 40, "görev", "", "grey", size=8)]
    for layer in (0, 1):
        x = 138 + layer * 150
        for j in range(3):
            y = 14 + j * 52
            o += [box(p, x, y, 116, 38, f"işçi {layer+1}.{j+1}", "", "blue", size=7.6)]
            if layer == 0:
                o += [arrow(p, 108, 82, 136, y + 19)]
            else:
                o += [arrow(p, 256, 33 + j * 52, 286, y + 19, dash="2 3")]
    o += [box(p, 444, 62, 142, 40, "orkestratör", "birleştirir", "green", size=8.4)]
    for j in range(3):
        o += [arrow(p, 404, 33 + j * 52, 442, 82)]
    o += [note(14, 128, "İleri-beslemeli sinir ağı mimarisinden modellenmiş: katman "
                        "katman işçiler, bir önceki katmanın çıktıları BİRLEŞTİRİLİP "
                        "sonrakine gidiyor.")]
    o += [note(14, 144, "arXiv:2406.04692 · aynı soru, farklı uzmanlıklar, tek "
                        "birleştirici.", "#767d84")]
    return figure(600, 156, "".join(o))


def f_debate():
    p = Pen(129)
    o = []
    for t in range(3):
        x = 20 + t * 152
        o += [note(x + 50, 18, f"tur {t+1}", "#767d84", 7.4, anchor="middle")]
        for j in range(2):
            y = 26 + j * 48
            o += [box(p, x, y, 104, 36, f"çözücü {j+1}", "", "blue", size=7.4)]
        if t < 2:
            o += [arrow(p, x + 106, 44, x + 150, 44, dash="3 3")]
            o += [arrow(p, x + 106, 92, x + 150, 92, dash="3 3")]
            o += [arrow(p, x + 106, 52, x + 150, 84, "çapraz", dash="2 3")]
    o += [box(p, 480, 50, 106, 40, "toplayıcı", "", "green", size=8)]
    o += [arrow(p, 428, 44, 478, 62), arrow(p, 428, 92, 478, 78)]
    o += [note(20, 122, "Her turda ajanlar cevaplarını DEĞİŞ TOKUŞ edip birbirinin "
                        "cevabına göre kendilerininkini düzeltiyor.")]
    o += [note(20, 138, "Çözücüler seyrek bağlı — herkes herkesle değil. GSM8K matematik "
                        "problemleri üstünde gösteriliyor.", "#767d84")]
    return figure(600, 150, "".join(o))


def f_reflection():
    p = Pen(130)
    o = [box(p, 60, 40, 160, 54, "ÜRETİCİ", "kod yazar", "blue", size=9)]
    o += [box(p, 372, 40, 160, 54, "ELEŞTİRMEN", "kritik üretir", "orange", size=9)]
    o += [arrow(p, 222, 56, 370, 56, "taslak")]
    o += [p.curve([(370, 82), (300, 108), (224, 82)], "orange", arrow=True)]
    o += [note(296, 124, "düzeltme isteği", "#8a5208", 7.4, anchor="middle")]
    o += [note(60, 150, "İkinci LLM üretimi, birincinin ÇIKTISINA koşullanmış. Döngü "
                        "eleştirmen tatmin olana kadar sürüyor.")]
    o += [note(60, 166, "Bizde karşılığı: RiskAuditor — üç analizi çelişki ve kaynaksız "
                        "iddia için çapraz kontrol ediyor.", "#2f9e44")]
    return figure(600, 178, "".join(o))


def f_codeexec_pattern():
    p = Pen(131)
    o = [box(p, 30, 46, 150, 52, "Assistant", "kodu YAZAR", "blue", size=9)]
    o += [box(p, 390, 46, 150, 52, "Executor", "kodu KOŞTURUR", "green", size=9)]
    o += [arrow(p, 182, 60, 388, 60, "kod bloğu")]
    o += [p.curve([(388, 88), (285, 116), (182, 88)], "green", arrow=True)]
    o += [note(285, 132, "çıktı ya da hata", "#14594a", 7.4, anchor="middle")]
    o += [note(30, 158, "İki ayrı ajan, tek bir Message veri sınıfı. AgentChat'te hazır "
                        "karşılığı var (CodeExecutorAgent) ama kılavuz burada "
                        "elle yazmayı gösteriyor.")]
    o += [note(30, 174, "Kılavuzun örneği: Tesla ve Nvidia hisse getirilerinin grafiğini "
                        "çizdirmek.", "#767d84")]
    return figure(600, 186, "".join(o))


# ──────────────────────────── AgentChat "Advanced" — the five the deck skipped


def f_custom_agent():
    p = Pen(140)
    o = [box(p, 176, 8, 248, 34, "BaseChatAgent", "bütün AgentChat ajanlarının atası",
             "violet", size=8.6)]
    members = [("on_messages", "mesaja nasıl cevap verir → Response", "blue"),
               ("on_reset", "başlangıç durumuna nasıl döner", "green"),
               ("produced_message_types", "hangi mesaj tiplerini üretebilir", "orange")]
    for i, (name, what, col) in enumerate(members):
        y = 62 + i * 30
        o += [p.rect(60, y, 480, 24, col, width=1.2)]
        o += [mono(72, y + 16, name, size=7.4, colour="#1e1e1e")]
        o += [text(250, y + 16, what, size=7.6, colour="#454c53")]
        if i == 0:
            o += [arrow(p, 300, 44, 300, 58)]
    o += [note(60, 168, "Üçünü uygulayan her sınıf bir ajandır — ve takımlara "
                        "hazır ajanlarla aynı şekilde girer.", "#454c53")]
    o += [note(60, 184, "AssistantAgent da bunun bir alt sınıfı. \"Hazır ajan\" ile "
                        "\"özel ajan\" arasında ayrıcalık farkı yok.", "#2f9e44")]
    return figure(600, 196, "".join(o))


def f_memory_rag():
    p = Pen(141)
    o = [box(p, 14, 46, 104, 44, "kullanıcı", "sorusu", "grey", size=8)]
    o += [arrow(p, 120, 68, 156, 68)]
    o += [box(p, 158, 26, 150, 84, "Memory protokolü", "", "violet", size=8.6)]
    for i, s in enumerate(("add", "query", "update_context", "clear · close")):
        o += [mono(170, 56 + i * 14, s, size=6.6, colour="#454c53")]
    o += [arrow(p, 310, 68, 346, 68, "ilgili kayıtlar")]
    o += [box(p, 348, 46, 128, 44, "model_context", "MUTASYONA UĞRAR", "orange", size=7.8)]
    o += [arrow(p, 478, 68, 512, 68)]
    o += [box(p, 514, 46, 74, 44, "model", "", "blue", size=8)]
    o += [note(14, 128, "Kritik olan üçüncü metot: update_context, ajanın KENDİ "
                        "model_context'ini değiştiriyor — yani getirilen bilgi bağlama "
                        "model çağrısından hemen önce giriyor.", "#8a5208")]
    o += [note(14, 152, "Uygulamalar: ListMemory (basit liste) · ChromaDBVectorMemory · "
                        "Redis. Klasik RAG deseni bu protokolün bir örneği.")]
    return figure(600, 164, "".join(o))


def f_tracing():
    p = Pen(142)
    o = [box(p, 14, 34, 130, 46, "ajan koşusu", "", "blue", size=8.4)]
    o += [arrow(p, 146, 57, 182, 57)]
    o += [box(p, 184, 20, 176, 74, "OpenTelemetry", "GenAI semantic conventions",
              "violet", size=8.6)]
    o += [arrow(p, 362, 57, 398, 57)]
    o += [box(p, 400, 12, 188, 30, "Jaeger · Zipkin", "", "green", size=7.8)]
    o += [box(p, 400, 48, 188, 30, "OTLP toplayıcı", "", "green", size=7.8)]
    o += [box(p, 400, 84, 188, 30, "herhangi OTel arka ucu", "", "green", size=7.6)]
    o += [note(184, 110, "ajanlar VE tool'lar için span", "#5f3dc4", 7.2)]
    o += [note(14, 138, "AutoGen'in tracing'i yerleşik ve OpenTelemetry'ye dayanıyor — "
                        "yani arka ucu sen seçiyorsun, çerçeve dayatmıyor.")]
    o += [note(14, 154, "GenAI Semantic Conventions'ı takip ediyor (hâlâ geliştirilme "
                        "aşamasında). Bizde KURULU DEĞİL — olay akışıyla yetiniyoruz.",
                        "#c92a2a")]
    return figure(600, 166, "".join(o))


def f_serialize_agentchat():
    p = Pen(143)
    o = [box(p, 20, 30, 150, 50, "bileşen", "ajan · takım · istemci", "blue", size=8.4)]
    o += [arrow(p, 172, 55, 214, 55, "dump_component()")]
    o += [box(p, 216, 30, 150, 50, "JSON", "bildirimsel şartname", "green", size=8.4)]
    o += [p.curve([(216, 88), (290, 112), (366, 88)], "green", arrow=False)]
    o += [arrow(p, 368, 55, 410, 55, "load_component()")]
    o += [box(p, 412, 30, 150, 50, "yeni nesne", "", "blue", size=8.4)]
    o += [p.rect(20, 116, 566, 34, "red", width=1.6)]
    o += [text(303, 130, "ONLY LOAD COMPONENTS FROM TRUSTED SOURCES",
               size=9.2, colour="#c92a2a", weight="bold", anchor="middle")]
    o += [note(303, 144, "nesne kurmak KOD ÇALIŞTIRMAYI içerebilir "
                         "(örn. serileştirilmiş bir fonksiyon)",
               "#8c2f1d", 7.2, anchor="middle")]
    o += [note(20, 172, "Her bileşen kendi serileştirme mantığını yazıyor — yani "
                        "yükleme davranışını bileşen belirliyor, çerçeve değil.")]
    return figure(600, 184, "".join(o))


def f_magentic():
    p = Pen(144)
    o = [box(p, 214, 8, 172, 42, "Orchestrator", "plan kurar · ilerlemeyi izler",
             "violet", size=8.6)]
    workers = [("MultimodalWebSurfer", "tarayıcı", 12), ("FileSurfer", "dosya", 162),
               ("MagenticOneCoder", "kod yazar", 312), ("ComputerTerminal", "kod koşturur", 462)]
    for name, what, x in workers:
        o += [box(p, x, 92, 126, 40, name[:19], what, "blue", size=7.2)]
        o += [arrow(p, 300, 52, x + 63, 90)]
    o += [note(214, 66, "planı DİNAMİK olarak revize eder", "#5f3dc4", 7.2)]
    o += [note(12, 152, "MagenticOneGroupChat artık sıradan bir AgentChat takımı — "
                        "dört ajanı da başka akışlarda tek tek kullanabiliyorsun.")]
    o += [note(12, 168, "GAIA benchmark'ında ölçülmüş (arXiv:2411.04468). Kılavuzun "
                        "kendi uyarısı: konteyner, sanal ortam, log izleme, İNSAN "
                        "GÖZETİMİ, kısıtlı erişim.", "#c92a2a")]
    return figure(600, 182, "".join(o))
