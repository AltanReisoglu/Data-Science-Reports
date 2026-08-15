"""Build an Excalidraw scene of AutoGen, from what docs/12 and docs/14 established.

Written as a generator rather than hand-typed JSON because the format needs ~20
fields per element and the layout is worth being able to change.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(14)

# Palette matches the artifacts: cool ground, ochre accent, green for measured.
INK = "#1e1e1e"
MUTED = "#5c6470"
OCHRE = "#a1620a"
GREEN = "#0f6b57"
BLUE = "#2c4a6b"
GREY = "#868e96"

BG_PAPER = "#f4f5f2"
BG_OCHRE = "#fbf0dd"
BG_GREEN = "#e2efe9"
BG_BLUE = "#e3eaf2"
BG_NONE = "transparent"

ELEMENTS: list[dict] = []


def _seed() -> int:
    return random.randint(1, 2**31 - 1)


def _base(kind: str, x: float, y: float, w: float, h: float, **over) -> dict:
    element = {
        "id": f"{kind}-{len(ELEMENTS)}-{_seed()}",
        "type": kind,
        "x": x, "y": y, "width": w, "height": h,
        "angle": 0,
        "strokeColor": INK,
        "backgroundColor": BG_NONE,
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3},
        "seed": _seed(),
        "version": 1,
        "versionNonce": _seed(),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
    }
    element.update(over)
    ELEMENTS.append(element)
    return element


def box(x, y, w, h, *, fill=BG_NONE, stroke=INK, dashed=False, sharp=False, width=1):
    return _base(
        "rectangle", x, y, w, h,
        backgroundColor=fill, strokeColor=stroke, strokeWidth=width,
        strokeStyle="dashed" if dashed else "solid",
        roundness=None if sharp else {"type": 3},
    )


# Cascadia is noticeably wider than Helvetica; both are approximated so text
# lands inside its box rather than being measured by a browser we do not have.
WIDTH_FACTOR = {1: 0.58, 2: 0.52, 3: 0.62}


def text(x, y, body, *, size=16, color=INK, font=2, align="left", w=None):
    lines = body.split("\n")
    longest = max((len(line) for line in lines), default=1)
    width = w if w is not None else longest * size * WIDTH_FACTOR[font]
    height = len(lines) * size * 1.25
    return _base(
        "text", x, y, width, height,
        strokeColor=color, backgroundColor=BG_NONE, roundness=None,
        text=body, originalText=body,
        fontSize=size, fontFamily=font,
        textAlign=align, verticalAlign="top",
        containerId=None, lineHeight=1.25, baseline=size,
    )


def label(x, y, body, **kw):
    """Centred text, given the centre point."""
    size = kw.get("size", 16)
    font = kw.get("font", 2)
    longest = max((len(line) for line in body.split("\n")), default=1)
    width = longest * size * WIDTH_FACTOR[font]
    height = len(body.split("\n")) * size * 1.25
    return text(x - width / 2, y - height / 2, body, align="center", w=width, **kw)


def arrow(points, *, color=INK, dashed=False, end="arrow", start=None, width=1):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ox, oy = xs[0], ys[0]
    return _base(
        "arrow", ox, oy, max(xs) - min(xs), max(ys) - min(ys),
        strokeColor=color, strokeWidth=width,
        strokeStyle="dashed" if dashed else "solid",
        roundness={"type": 2},
        points=[[p[0] - ox, p[1] - oy] for p in points],
        lastCommittedPoint=None,
        startBinding=None, endBinding=None,
        startArrowhead=start, endArrowhead=end,
        elbowed=False,
    )


def line(points, *, color=GREY, dashed=False, width=1):
    return arrow(points, color=color, dashed=dashed, end=None, width=width)


# ═══════════════════════════════════════════════════════ 1 · three layers

text(60, 40, "AutoGen", size=36, font=1)
text(60, 90, "üç katman · protokoller · iletişim · desenler", size=16, color=MUTED)
text(60, 116, "vc-agent/docs/12 + 14  ·  autogen 0.7.5", size=12, color=GREY, font=3)

text(60, 180, "1 · ÜÇ KATMAN", size=14, color=OCHRE, font=3)

layers = [
    ("autogen-ext", "model istemcileri · tool'lar · MCP · kod yürütücüler", BG_PAPER),
    ("autogen-agentchat", "AssistantAgent · takımlar · sonlandırma · yapısal çıktı", BG_PAPER),
    ("autogen-core", "aktör modeli · event-driven runtime · pub/sub · gRPC", BG_OCHRE),
]
for i, (name, sub, fill) in enumerate(layers):
    y = 210 + i * 78
    box(60, y, 560, 66, fill=fill, width=2 if i == 2 else 1)
    text(80, y + 12, name, size=20, font=3)
    text(80, y + 40, sub, size=13, color=MUTED)
    if i < 2:
        arrow([(340, y + 78), (340, y + 66)], color=GREY)

text(636, 372, "← asıl mühendislik", size=13, color=OCHRE)
text(636, 392, "  hikâyesi burada", size=13, color=OCHRE)
text(636, 294, "← tutorial'lar", size=13, color=GREY)
text(636, 312, "  burada biter", size=13, color=GREY)

text(60, 456, "Ayıran şey üst katman değil, alt katman.", size=15)
text(60, 478, "AssistantAgent her framework'te var. autogen_core karşılığı çoğunda yok.",
     size=13, color=MUTED)

# ═══════════════════════════════════════════════════════ 2 · protocols

text(780, 180, "2 · KONUŞTUĞU PROTOKOLLER", size=14, color=OCHRE, font=3)

protocols = [
    ("olay formatı", "CloudEvents v1", "TopicId docstring → CNCF spec", True),
    ("dağıtık taşıma", "gRPC + protobuf", "AgentRpc · çift yönlü akış", True),
    ("tool federasyonu", "MCP", "stdio · SSE · streamable HTTP", True),
    ("gözlemlenebilirlik", "OTel GenAI conventions", 'gen_ai.system = "autogen"', True),
    ("model erişimi", "OpenAI Chat Completions", "fiilî standart", True),
    ("serileştirme", "ComponentModel", "kendi şeması", True),
    ("ajan federasyonu", "A2A", "YOK — ADK'da var", False),
]
for i, (kind, proto, note, present) in enumerate(protocols):
    y = 214 + i * 52
    line([(786, y + 20), (786, y + 44)], color=GREY if present else "#c9ccc4")
    box(778, y + 14, 16, 16, fill=BG_OCHRE if present else BG_NONE,
        stroke=OCHRE if present else GREY, sharp=False)
    text(812, y + 4, kind, size=11, color=GREY, font=3)
    text(812, y + 20, proto, size=16, font=3, color=INK if present else GREY)
    text(1090, y + 22, note, size=12, color=MUTED)
    if not present:
        line([(808, y + 30), (808 + len(proto) * 10, y + 30)], color=GREY)

text(778, 600, "Dışa açılan her yerde standart. Yalnız serileştirme kendi şeması,", size=13)
text(778, 622, "ve ajanlar arası federasyon hiç kapsamda değil.", size=13, color=MUTED)

# ═══════════════════════════════════════════════════════ 3 · communication

text(60, 700, "3 · İKİ İLETİŞİM BİÇİMİ — fark adresleme değil, arıza davranışı", size=14,
     color=OCHRE, font=3)

# direct
box(60, 736, 520, 250, fill=BG_GREEN, stroke=GREEN)
text(84, 756, "DOĞRUDAN MESAJ", size=13, color=GREEN, font=3)
text(84, 780, "send_message(msg, AgentId(tip, anahtar))", size=14, font=3)

box(100, 826, 150, 56, fill="#ffffff", stroke=INK)
label(175, 854, "OuterAgent", size=14, font=3)
box(390, 826, 150, 56, fill="#ffffff", stroke=INK)
label(465, 854, "InnerAgent", size=14, font=3)
arrow([(250, 844), (390, 844)], color=INK)
arrow([(390, 866), (250, 866)], color=GREEN, dashed=True)
text(268, 820, "send", size=12, color=MUTED, font=3)
text(262, 872, "return", size=12, color=GREEN, font=3)

text(84, 902, "AgentId(tip, self.id.key) — aynı anahtar, farklı tip.", size=12, color=MUTED)
text(84, 924, "Kayıt defteri tutmadan eşlenik ajan.", size=12, color=MUTED)
text(84, 952, "hata → çağırana fırlatılır", size=14, color=GREEN, font=3)

# broadcast
box(620, 736, 660, 250, fill=BG_OCHRE, stroke=OCHRE)
text(644, 756, "YAYIN", size=13, color=OCHRE, font=3)
text(644, 780, "publish_message(msg, TopicId(tip, kaynak))", size=14, font=3)

box(650, 830, 130, 50, fill="#ffffff", stroke=INK)
label(715, 855, "publisher", size=13, font=3)
for i in range(3):
    bx = 900 + i * 128
    box(bx, 830, 112, 50, fill="#ffffff", stroke=INK)
    label(bx + 56, 855, f"abone {i + 1}", size=13, font=3)
    arrow([(780, 855), (bx, 855)], color=OCHRE)

text(644, 902, "cevap yok — her zaman None. return yazarsan atılır.", size=12, color=MUTED)
text(644, 924, "kendi yayınını duymazsın — sonsuz döngüye karşı.", size=12, color=MUTED)
text(644, 952, "hata → loglanır, yayınlayana GİTMEZ", size=14, color=OCHRE, font=3)

box(60, 1006, 1220, 46, fill=BG_PAPER, stroke=OCHRE, dashed=True)
text(84, 1020, "Sessiz kardeş kaybı bu yüzden yayın tarafında.  "
                "Ölçüldü: GraphFlow 0–1 dal / süre sınırı  ·  pub/sub + kuyruk 2 dal / ~3 ms",
     size=13)

# ═══════════════════════════════════════════════════════ 4 · topic → key

text(60, 1104, "4 · KILAVUZUN EN ATLANAN SAYFASI", size=14, color=OCHRE, font=3)

box(60, 1140, 400, 60, fill="#ffffff", stroke=INK)
label(260, 1170, 'TopicId("gorev", "sirket-42")', size=15, font=3)
arrow([(260, 1200), (260, 1240)], color=OCHRE, width=2)
text(276, 1206, "runtime", size=12, color=MUTED, font=3)
box(60, 1240, 400, 60, fill=BG_OCHRE, stroke=OCHRE, width=2)
label(260, 1270, 'AgentId("analist", "sirket-42")', size=15, font=3)

text(500, 1146, "Topic kaynağı, ajan anahtarına dönüşür.", size=18, font=1)
text(500, 1180, "Ajan kimliği zaten (type, key) ve örnekler lazy doğuyor.", size=13, color=MUTED)
text(500, 1204, "İkisi birleşince:", size=13, color=MUTED)
text(500, 1236, "şirket başına izole ajan örneği bedava", size=16, color=OCHRE, font=3)
text(500, 1268, "Çok kiracılığın hazır mekanizması. Hacim sorusu", size=13, color=MUTED)
text(500, 1290, "cevaplanınca: kod değişikliği değil, kaynak seçimi.", size=13, color=MUTED)

# ═══════════════════════════════════════════════════════ 5 · teams

text(60, 1372, "5 · BEŞ TAKIM — kim konuşacağına kim karar veriyor", size=14, color=OCHRE, font=3)

teams = [
    ("RoundRobinGroupChat", "sıra", "274"),
    ("SelectorGroupChat", "bir model\n(ya da selector_func)", "204"),
    ("Swarm", "ajanın kendisi\ndevir tool'uyla", "334"),
    ("GraphFlow", "önceden çizilmiş graf", "270"),
    ("MagenticOne", "genel orkestratör", "—"),
]
for i, (name, who, tokens) in enumerate(teams):
    x = 60 + i * 250
    cheap = tokens == "204"
    dear = tokens == "334"
    box(x, 1408, 230, 128,
        fill=BG_GREEN if cheap else (BG_OCHRE if dear else BG_PAPER),
        stroke=GREEN if cheap else (OCHRE if dear else INK))
    text(x + 16, 1424, name, size=13, font=3)
    text(x + 16, 1452, who, size=12, color=MUTED)
    text(x + 16, 1500, f"{tokens} token", size=15, font=3,
         color=GREEN if cheap else (OCHRE if dear else MUTED))

text(60, 1556, "Aynı görev, aynı ajanlar — %63,7 fark. Ödenen şey zekâ değil, yönlendirme özerkliği.",
     size=14)
text(60, 1580, "Agents SDK'nın tek modeli olan handoff, AutoGen'in en pahalı deseni.",
     size=13, color=MUTED)

# ═══════════════════════════════════════════════════════ 6 · patterns

text(60, 1652, "6 · DOKUZ DESEN — ve bizde ne var", size=14, color=OCHRE, font=3)

patterns = [
    ("Concurrent Agents", "fanin.py", "yüksek", GREEN),
    ("Sequential Workflow", "kodla", "düşük", OCHRE),
    ("Group Chat", "—", "en ayırt edici", GREEN),
    ("Handoffs", "—", "en pahalı", MUTED),
    ("Mixture of Agents", "—", "gather tuzağı", OCHRE),
    ("Multi-Agent Debate", "—", "ilkeli, pahalı", MUTED),
    ("Reflection", "RiskAuditor", "durma ölçütü yok", OCHRE),
    ("Code Execution", "—", "en yüksek", GREEN),
]
for i, (name, ours, verdict, color) in enumerate(patterns):
    col, row = i % 4, i // 4
    x, y = 60 + col * 310, 1688 + row * 96
    box(x, y, 290, 78, fill=BG_PAPER if ours == "—" else BG_GREEN,
        stroke=color if color != MUTED else GREY)
    text(x + 14, y + 12, name, size=14, font=3)
    text(x + 14, y + 36, f"bizde: {ours}", size=12, color=MUTED)
    text(x + 14, y + 56, verdict, size=12, color=color, font=3)

box(60, 1888, 1220, 60, fill=BG_OCHRE, stroke=OCHRE, dashed=True)
text(84, 1900, "Üçünde açık ara güçlü, üçünde ortalama, üçünde geride.", size=15, font=1)
text(84, 1924, "Geride kaldığı üçü, Agno'nun Workflow ilkelleriyle birinci sınıf yaptığı üçü.",
     size=13, color=MUTED)

# ═══════════════════════════════════════════════════════ footer

line([(60, 1996), (1280, 1996)], color=GREY)
text(60, 2010, "Her sayı ölçüldü: poc/kiyas.py · pipeline/compare_fanin.py     "
               "Protokoller kurulu pakete introspeksiyonla.", size=12, color=GREY, font=3)

scene = {
    "type": "excalidraw",
    "version": 2,
    "source": "vc-agent/docs",
    "elements": ELEMENTS,
    "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
    "files": {},
}

out = Path(__file__).parent / "autogen-mimari.excalidraw"
out.write_text(json.dumps(scene, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{out}  ·  {len(ELEMENTS)} eleman  ·  {out.stat().st_size // 1024} KB")
