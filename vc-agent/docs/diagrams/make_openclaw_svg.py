"""Figures for `docs/pdf/openclaw-ici.html` — OpenClaw's internals, drawn.

Same hand as the AutoGen guide: `rough.py` primitives, one seeded `Pen` per
figure so a rebuild is byte-identical. Run after editing a diagram; it replaces
the `<svg>` blocks in the HTML in order and leaves the prose alone.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rough import Pen, figure, mono, text  # noqa: E402

HTML = Path(__file__).resolve().parents[1] / "pdf" / "openclaw-ici.html"

INK, GREY, MUTE = "#1e1e1e", "#767d84", "#454c53"
BLUE, GREEN, RED, ORANGE, VIOLET = "#1971c2", "#2f9e44", "#c92a2a", "#e8590c", "#5f3dc4"


def fig_surface() -> str:
    """What is actually there, counted rather than described."""
    p, o = Pen(3101), []
    cells = [
        ("351", "gateway metodu", "sessions 51 · skills 28 · talk 20 · node 19", "blue"),
        ("44", "tool", "14 grup · 4 profil · core + plugin", "green"),
        ("74", "skill kurulu", "51 paket · 20 kişisel · 3 ek", "orange"),
        ("40", "skill modele görünür", "34'ü bağımlılık eksiğinden elendi", "red"),
    ]
    for i, (big, label, sub, colour) in enumerate(cells):
        x = 8 + i * 158
        o.append(p.rect(x, 20, 148, 88, colour, width=1.8))
        o.append(text(x + 74, 56, big, size=22, anchor="middle", weight="bold", colour=INK))
        o.append(text(x + 74, 74, label, size=8.6, anchor="middle", weight="bold", colour=MUTE))
        o.append(text(x + 74, 92, sub, size=6.8, anchor="middle", colour=GREY))
    return figure(640, 118, "".join(o))


def fig_prompt() -> str:
    """The cache boundary: what is stable and what is rewritten every turn."""
    p, o = Pen(3202), []
    o.append(p.rect(14, 22, 612, 132, "green", width=1.9))
    stable = [
        "Sabit bölümler — Tooling · Execution Bias · Safety · Workspace",
        "TOOL ŞEMALARI — policy ile süzülmüş her tool'un tam JSON schema'sı",
        "<available_skills> — indeks: name · description · location · sha256",
        "MEMORY.md + USER.md — bootstrap sınırıyla (20k/dosya, 60k toplam)",
        "AGENTS.md · SOUL.md · TOOLS.md · IDENTITY.md · HEARTBEAT.md",
    ]
    o.append(text(28, 40, "SINIRIN ÜSTÜ — kararlı, prefix-cache'lenir", size=9,
                  weight="bold", colour=GREEN))
    for i, s in enumerate(stable):
        o.append(text(28, 60 + i * 17, "· " + s, size=7.8, colour=MUTE))

    o.append(p.line(14, 164, 626, 164, "red", width=2.2, arrow=False, dash="7 4"))
    o.append(text(320, 178, "CACHE SINIRI", size=8.6, anchor="middle", weight="bold",
                  colour=RED))

    o.append(p.rect(14, 190, 612, 62, "orange", width=1.9))
    o.append(text(28, 208, "SINIRIN ALTI — her turda değişir", size=9, weight="bold",
                  colour=ORANGE))
    o.append(text(28, 226, "· Messaging · Runtime satırı · Date/Time · output direktifleri",
                  size=7.8, colour=MUTE))
    o.append(text(28, 242, "· context engine'in assemble()'dan dönen systemPromptAddition'ı "
                           "en öne eklenir", size=7.8, colour=MUTE))
    return figure(640, 262, "".join(o))


def fig_skill_index() -> str:
    """Progressive disclosure, with the measurement that justifies it."""
    p, o = Pen(3303), []
    o.append(p.rect(14, 24, 296, 128, "green", width=1.9))
    o.append(p.rect(330, 24, 296, 128, "grey", width=1.5, dash="5 3"))
    o += [
        text(28, 44, "PROMPT'A GİREN — indeks", size=9, weight="bold", colour=GREEN),
        mono(28, 66, "<available_skills>", size=7.4),
        mono(38, 80, '<name>diagram-maker</name>', size=7),
        mono(38, 92, '<description>SVG/Excalidraw…</description>', size=7),
        mono(38, 104, '<location>…/SKILL.md</location>', size=7),
        mono(38, 116, '<version>sha256:…</version>', size=7),
        mono(28, 130, "</available_skills>", size=7.4),
        text(28, 146, "40 skill · ~2.900 token", size=8.4, weight="bold", colour=GREEN),

        text(344, 44, "PROMPT'A GİRMEYEN — gövde", size=9, weight="bold", colour=GREY),
        text(344, 66, "SKILL.md'nin tam metni: yordam, örnekler,", size=7.8, colour=GREY),
        text(344, 80, "komutlar, uyarılar. Model gerektiğinde", size=7.8, colour=GREY),
        text(344, 94, "`read` tool'uyla <location>'ı açıyor.", size=7.8, colour=GREY),
        text(344, 116, "40 skill · ~40.076 token", size=8.4, weight="bold", colour=GREY),
        text(344, 132, "160.304 bayt, diskte duruyor", size=7.4, colour=GREY),
    ]
    o.append(p.line(310, 88, 326, 88, "grey", width=1.3, dash="3 2"))
    o.append(p.rect(14, 166, 612, 46, "red", width=2))
    o += [
        text(28, 184, "%93 TASARRUF", size=11, weight="bold", colour=RED),
        text(140, 184, "— ve sha256 sürümü önbellek geçersizleştirme sinyali:", size=8.4),
        mono(28, 202, 'system-prompt.ts:312  "Changed <version>: re-read."', size=7.6),
        text(330, 202, "sürüm değişmediyse model gövdeyi tekrar okumuyor.", size=7.8, colour=MUTE),
    ]
    return figure(640, 222, "".join(o))


def fig_gating() -> str:
    """74 installed, 40 reach the model. The gate is dependency checking."""
    p, o = Pen(3404), []
    stages = [
        (8, 200, "74 kurulu", "6 kaynaktan yüklendi", "grey"),
        (232, 178, "68 platform uygun", "6'sı darwin-only", "orange"),
        (438, 190, "40 modele görünür", "34'ü eksik ikili/env", "green"),
    ]
    for x, w, label, sub, colour in stages:
        o.append(p.rect(x, 24, w, 52, colour, width=1.8))
        o.append(text(x + w / 2, 46, label, size=10, anchor="middle", weight="bold", colour=INK))
        o.append(text(x + w / 2, 63, sub, size=7.4, anchor="middle", colour=GREY))
    o.append(p.line(208, 50, 228, 50, "red", width=1.6))
    o.append(p.line(410, 50, 434, 50, "red", width=1.6))
    o += [
        text(218, 90, "os", size=7.4, anchor="middle", colour=RED, weight="bold"),
        text(422, 90, "requires", size=7.4, anchor="middle", colour=RED, weight="bold"),
    ]
    o.append(p.rect(8, 104, 620, 56, "grey", width=1.4))
    o += [
        text(22, 122, "KAPI — yükleme anında, metadata.openclaw ile", size=8.6,
             weight="bold", colour=INK),
        mono(22, 140, 'requires: { bins: ["uv"], anyBins: [], env: ["GEMINI_API_KEY"], '
                      'config: ["browser.enabled"] }', size=7.2),
        text(22, 154, "always: true → bütün kapıları atlar. Eksik bağımlılıklı skill "
                      "prompt'a hiç girmiyor: kullanamayacağın şeye token ödemiyorsun.",
             size=7.6, colour=MUTE),
    ]
    return figure(640, 170, "".join(o))


def fig_tools() -> str:
    """The catalogue: groups, sources, profiles."""
    p, o = Pen(3505), []
    groups = [
        ("fs", 4, "read write edit apply_patch"), ("runtime", 3, "exec process code_execution"),
        ("web", 3, "web_search web_fetch x_search"), ("memory", 2, "memory_search memory_get"),
        ("sessions", 7, "list history send spawn yield subagents"),
        ("ui", 2, "browser canvas"), ("messaging", 1, "message"),
        ("automation", 3, "heartbeat_respond cron gateway"), ("nodes", 1, "nodes"),
        ("agents", 6, "goal plan skill_workshop"), ("media", 5, "image music video tts"),
    ]
    for i, (name, n, tools) in enumerate(groups):
        col, row = i % 2, i // 2
        x, y = 8 + col * 312, 40 + row * 24
        o.append(p.rect(x, y, 300, 20, "blue", width=1.2))
        o.append(text(x + 8, y + 14, name, size=7.8, weight="bold", colour=BLUE))
        o.append(text(x + 74, y + 14, str(n), size=7.8, weight="bold", colour=INK))
        o.append(text(x + 92, y + 14, tools[:44], size=6.8, colour=GREY))
    o.append(text(8, 30, "CORE GRUPLARI — 37 tool", size=8.6, weight="bold", colour=BLUE))
    y = 40 + ((len(groups) + 1) // 2) * 24 + 8
    o.append(text(8, y + 6, "PLUGIN GRUPLARI — 7 tool", size=8.6, weight="bold", colour=VIOLET))
    for i, (name, n, tools) in enumerate([
        ("plugin:file-transfer", 4, "dir_fetch dir_list file_fetch file_write"),
        ("plugin:tavily", 2, "tavily_search tavily_extract"),
        ("plugin:ollama", 1, "node_inference"),
    ]):
        o.append(p.rect(8 + (i % 2) * 312, y + 14 + (i // 2) * 24, 300, 20, "violet", width=1.2))
        o.append(text(16 + (i % 2) * 312, y + 28 + (i // 2) * 24, name, size=7.4,
                      weight="bold", colour=VIOLET))
        o.append(text(120 + (i % 2) * 312, y + 28 + (i // 2) * 24, f"{n} · {tools[:36]}",
                      size=6.8, colour=GREY))
    o.append(p.rect(8, y + 66, 620, 30, "orange", width=1.6))
    o.append(text(20, y + 85, "PROFİLLER —  minimal · coding · messaging · full   "
                              "(hangi grupların açık olduğunu profil belirliyor)",
                  size=8, weight="bold", colour=ORANGE))
    return figure(640, y + 106, "".join(o))


def fig_context_engine() -> str:
    """Four lifecycle points, and what the default engine does at each."""
    p, o = Pen(3606), []
    points = [
        ("1 · Ingest", "yeni mesaj eklendiğinde", "kendi store'una indexler", "legacy: no-op"),
        ("2 · Assemble", "HER model run'ından ÖNCE", "bütçeye sığan sıralı set", "legacy: pass-through"),
        ("3 · Compact", "pencere dolunca / /compact", "eski geçmişi özetler", "legacy: built-in özet"),
        ("4 · After turn", "run bitince", "state persist, arka plan", "legacy: no-op"),
    ]
    for i, (title, when, what, legacy) in enumerate(points):
        x = 8 + i * 158
        o.append(p.rect(x, 24, 148, 92, "violet" if i == 1 else "blue",
                        width=2.4 if i == 1 else 1.5))
        o.append(text(x + 74, 44, title, size=9.4, anchor="middle", weight="bold",
                      colour=VIOLET if i == 1 else BLUE))
        o.append(text(x + 74, 62, when, size=7, anchor="middle", colour=MUTE))
        o.append(text(x + 74, 78, what, size=7, anchor="middle", colour=MUTE))
        o.append(text(x + 74, 100, legacy, size=6.8, anchor="middle", colour=GREY))
        if i < 3:
            o.append(p.line(x + 148, 70, x + 156, 70, width=1.2))
    o.append(p.rect(8, 128, 620, 44, "green", width=1.7))
    o += [
        text(22, 146, "assemble() ayrıca systemPromptAddition döndürebilir", size=8.6,
             weight="bold", colour=GREEN),
        text(22, 163, "— sistem prompt'un en önüne eklenir. Motorlar böylece dinamik "
                      "hatırlama yönergesi enjekte ediyor; statik dosya gerekmiyor.",
             size=7.8, colour=MUTE),
    ]
    return figure(640, 182, "".join(o))


def fig_compaction() -> str:
    """The split point that never separates a tool call from its result."""
    p, o = Pen(3707), []
    blocks = [
        (14, 70, "eski turlar", "grey"), (92, 46, "tool call", "orange"),
        (146, 52, "toolResult", "orange"), (206, 60, "mesaj", "grey"),
        (274, 60, "mesaj", "green"), (342, 60, "mesaj", "green"),
    ]
    for x, w, label, colour in blocks:
        o.append(p.rect(x, 40, w, 30, colour, width=1.5))
        o.append(text(x + w / 2, 59, label, size=7, anchor="middle", colour=INK))
    o.append(p.line(120, 30, 120, 82, "red", width=2, arrow=False, dash="4 3"))
    o.append(text(120, 24, "yanlış sınır", size=7.4, anchor="middle", colour=RED, weight="bold"))
    o.append(p.line(202, 30, 202, 82, "green", width=2.2, arrow=False))
    o.append(text(214, 24, "kaydırılmış sınır", size=7.4, colour=GREEN, weight="bold"))
    o.append(p.rect(420, 36, 206, 40, "green", width=1.6))
    o.append(text(432, 52, "özet", size=8.4, weight="bold", colour=GREEN))
    o.append(text(432, 68, "transkripte yazılır, disk tam kalır", size=7, colour=GREY))
    o.append(p.line(406, 56, 416, 56, "green", width=1.3))

    o.append(p.rect(14, 96, 612, 62, "blue", width=1.6))
    o += [
        text(28, 114, "ÜÇ TETİK", size=8.6, weight="bold", colour=BLUE),
        text(28, 131, "· eşik: oturum bağlam sınırına yaklaşınca (varsayılan açık)", size=7.8),
        text(28, 145, "· taşma: model context-overflow hatası dönünce → sıkıştır ve YENİDEN DENE", size=7.8),
        text(360, 131, "· elle: /compact", size=7.8),
        text(360, 145, "Öncesinde ajana \"notlarını belleğe yaz\" hatırlatması gidiyor.",
             size=7.4, colour=MUTE),
    ]
    return figure(640, 168, "".join(o))


def fig_memory() -> str:
    """Five tiers, and the boundary that matters."""
    p, o = Pen(3808), []
    tiers = [
        ("Instructions", "AGENTS.md, workspace", "yalnız insan", "HER ZAMAN", "green"),
        ("Curated core", "MEMORY.md, USER.md", "dreaming; kullanıcı", "HER ZAMAN, bütçeli", "green"),
        ("Episodic", "memory/YYYY-MM-DD.md", "ajan; transkript", "ASLA — aranabilir", "orange"),
        ("Prospective", "standing intents, cron", "intent tool", "tetik ateşlerse", "blue"),
        ("Review", "DREAMS.md", "dreaming", "ASLA — insan için", "grey"),
    ]
    o.append(text(20, 18, "KATMAN", size=7, weight="bold", colour=GREY))
    o.append(text(150, 18, "YÜZEY", size=7, weight="bold", colour=GREY))
    o.append(text(320, 18, "KİM YAZAR", size=7, weight="bold", colour=GREY))
    o.append(text(470, 18, "PROMPT'A GİRER Mİ", size=7, weight="bold", colour=GREY))
    for i, (name, surface, writer, injected, colour) in enumerate(tiers):
        y = 26 + i * 28
        o.append(p.rect(14, y, 612, 24, colour, width=1.4))
        o.append(text(24, y + 16, name, size=8.2, weight="bold", colour=INK))
        o.append(mono(150, y + 16, surface, size=7))
        o.append(text(320, y + 16, writer, size=7.4, colour=MUTE))
        o.append(text(470, y + 16, injected, size=7.6, weight="bold",
                      colour=GREEN if "HER" in injected else GREY))
    # The boundary that matters, drawn where it falls: between tier 2 and tier 3.
    o.append(p.line(12, 79, 628, 79, "red", width=2.4, arrow=False))
    o.append(p.rect(14, 176, 612, 58, "red", width=1.8))
    o += [
        text(28, 194, "ASIL SINIR — curated ile episodic arasında", size=8.6,
             weight="bold", colour=RED),
        text(28, 210, "Otomatik enjeksiyon YALNIZ curated katmandan. Günlük notlar ve "
                      "transkriptler, eşleşme ne kadar", size=7.6, colour=MUTE),
        text(28, 223, "güçlü olursa olsun asla enjekte edilmiyor — bu bir ayar tercihi "
                      "değil, güvenlik özelliği.", size=7.6, colour=MUTE),
    ]
    return figure(640, 244, "".join(o))


def fig_recall() -> str:
    """Two lanes, split by cost."""
    p, o = Pen(3909), []
    o.append(p.rect(14, 22, 300, 158, "green", width=1.9))
    o.append(p.rect(326, 22, 300, 158, "orange", width=1.9))
    o += [
        text(28, 42, "ŞERİT 1 — hep açık, SIFIR model çağrısı", size=8.8,
             weight="bold", colour=GREEN),
        text(28, 64, "· Bootstrap: MEMORY.md + USER.md oturum", size=7.6),
        text(38, 76, "başında, her turda tazelenir", size=7.6),
        text(28, 94, "· Sıralı arama: hibrit alaka × üstel tazelik", size=7.6),
        text(38, 106, "(30 gün yarı ömür) × önem (1-10)", size=7.6),
        text(28, 124, "· Tetik enjeksiyonu: yazarların iliştirdiği", size=7.6),
        text(38, 136, "tetik ifadeleri ön-elemeden geçer;", size=7.6),
        text(38, 148, "skor ≥ 0.72 → turda en fazla 3 giriş", size=7.6),
        text(28, 170, "Önem yazma anında verildiği için sorgu", size=7, colour=GREY),
        text(28, 178, "anında model çağrısı gerekmiyor.", size=7, colour=GREY),

        text(340, 42, "ŞERİT 2 — tırmanma", size=8.8, weight="bold", colour=ORANGE),
        text(340, 64, "Gerçek bir alt-ajan turu: konuşma", size=7.6),
        text(340, 78, "geçmişinde arama ve okuma yapabilir,", size=7.6),
        text(340, 92, "konuşmalar arası transkript hatırlama", size=7.6),
        text(340, 106, "dahil.", size=7.6),
        text(340, 130, "Pahalı, o yüzden yalnız gereken", size=7.6, colour=ORANGE, weight="bold"),
        text(340, 144, "turlarda çalışıyor.", size=7.6, colour=ORANGE, weight="bold"),
        text(340, 170, "Bu ayrım maliyete göre yapılmış:", size=7, colour=GREY),
        text(340, 178, "varsayılan şerit gecikme eklemiyor.", size=7, colour=GREY),
    ]
    return figure(640, 190, "".join(o))


def fig_schedule() -> str:
    """Five schedule kinds plus the condition watcher."""
    p, o = Pen(4010), []
    kinds = [
        ("at", "tek seferlik zaman damgası", "ISO 8601 ya da `20m`"),
        ("every", "sabit aralık", "10m · 1h · 1d"),
        ("cron", "5 ya da 6 alanlı ifade", "--tz ile IANA saat dilimi"),
        ("on-exit", "izlenen komut çıkınca", "tur yıkımından sağ çıkar"),
        ("stream", "uzun ömürlü komuttan", "toplu satırlardan tetiklenir"),
    ]
    for i, (kind, what, detail) in enumerate(kinds):
        y = 24 + i * 26
        o.append(p.rect(14, y, 612, 22, "blue", width=1.3))
        o.append(mono(26, y + 15, kind, size=8, colour=BLUE))
        o.append(text(96, y + 15, what, size=7.8, colour=INK))
        o.append(text(300, y + 15, detail, size=7.2, colour=GREY))
    o.append(p.rect(14, 162, 612, 62, "orange", width=1.8))
    o += [
        text(28, 180, "OLAY TETİKLEYİCİ — koşul gözcüsü", size=8.8, weight="bold", colour=ORANGE),
        text(28, 197, "every/cron/stream'e başsız bir koşul betiği eklenir. Zamanlayıcı "
                      "asıl yükü YALNIZ betik", size=7.8),
        text(28, 210, "fire:true dönerse koşturuyor. Yani zamanlama ile karar ayrı: "
                      "\"her 30 sn bak, ama sadece değiştiyse uyandır.\"", size=7.8),
    ]
    o.append(text(14, 240, "Tepe saat ifadeleri yük yığılmasını azaltmak için 5 dakikaya "
                           "kadar otomatik kaydırılıyor (--exact ile kapatılır).",
                  size=7.4, colour=GREY))
    return figure(640, 250, "".join(o))


FIGURES = [
    fig_surface, fig_prompt, fig_tools, fig_gating, fig_skill_index,
    fig_context_engine, fig_compaction, fig_memory, fig_recall, fig_schedule,
]


def main() -> int:
    html = HTML.read_text(encoding="utf-8")
    drawn = [f() for f in FIGURES]
    blocks = list(re.finditer(r"<svg\b.*?</svg>", html, re.S))
    if len(blocks) != len(drawn):
        print(f"figure count mismatch: html has {len(blocks)}, script draws {len(drawn)}")
        return 1
    for block, svg in zip(reversed(blocks), reversed(drawn)):
        html = html[: block.start()] + svg + html[block.end():]
    HTML.write_text(html, encoding="utf-8")
    print(f"{len(drawn)} figures redrawn into {HTML.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
