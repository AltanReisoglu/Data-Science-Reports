"""Draw the twelve figures of `docs/pdf/agentchat-kilavuzu.html` and splice them in.

Run it after editing a diagram; it rewrites the `<svg>` blocks in the HTML **in
order** and leaves the prose alone. Each figure gets its own `Pen` seed, so
changing one diagram cannot reshuffle the wobble of the others and turn a
one-line edit into a whole-document diff.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rough import Pen, figure, mono, text  # noqa: E402

HTML = Path(__file__).resolve().parents[1] / "pdf" / "agentchat-kilavuzu.html"

GREY, INK, MUTE = "#767d84", "#1e1e1e", "#454c53"
BLUE, GREEN, RED, ORANGE, VIOLET = "#1971c2", "#2f9e44", "#c92a2a", "#e8590c", "#5f3dc4"


def fig_layers() -> str:
    p, o = Pen(101), []
    o.append(p.rect(24, 18, 592, 58, "violet", width=1.7))
    o.append(p.rect(24, 92, 592, 78, "orange", width=2.6))
    o.append(p.rect(24, 186, 592, 58, "blue", width=1.7))
    o.append(p.line(330, 170, 330, 186))
    o += [
        text(42, 40, "autogen_ext", size=11, colour=VIOLET, weight="bold"),
        text(42, 57, "dış dünya · OpenAIChatCompletionClient · McpWorkbench · kod yürütücüler", size=8.4),
        text(42, 70, "extra'larla gelir: [openai] [azure] [docker] [mcp]", size=8, colour=GREY),
        text(42, 114, "autogen_agentchat — bu belgenin konusu", size=11, colour=ORANGE, weight="bold"),
        text(42, 132, "AssistantAgent · UserProxyAgent · CodeExecutorAgent", size=8.4),
        text(42, 147, "RoundRobin · Selector · Swarm · GraphFlow · MagenticOne", size=8.4),
        text(42, 162, "11 sonlandırma koşulu · Memory · save_state · dump_component", size=8.4),
        text(42, 208, "autogen_core", size=11, colour=BLUE, weight="bold"),
        text(42, 225, "aktör modeli · AgentId(type,key) · runtime · topic · abonelik", size=8.4),
        text(42, 238, "AgentChat tavan değil — kılavuzun bittiği yerde burası başlıyor", size=8, colour=GREY),
        text(342, 182, "üstüne kurulu", size=7.6, colour=GREY),
    ]
    return figure(640, 256, "".join(o))


def fig_rewrite() -> str:
    p, o = Pen(202), []
    o.append(p.rect(18, 26, 252, 168, "red", width=1.8))
    o.append(p.rect(370, 26, 252, 168, "green", width=1.8))
    o.append(p.line(276, 110, 364, 110, "orange", width=2))
    old = ["ConversableAgent", "initiate_chat(...)", "GroupChat + GroupChatManager",
           "register_function", 'llm_config={"config_list": [...]}']
    new = ["AssistantAgent", "team.run(task=...) / run_stream(...)",
           "RoundRobinGroupChat, SelectorGroupChat", "tools=[...]",
           "model_client=OpenAIChatCompletionClient(...)"]
    o.append(text(34, 46, "v0.2 — terk edilmiş", size=10, colour=RED, weight="bold"))
    o.append(text(386, 46, "v0.4+ — bu kılavuz", size=10, colour=GREEN, weight="bold"))
    for i, (a, b) in enumerate(zip(old, new)):
        y = 70 + i * 16
        o.append(mono(34, y, a, size=7.4))
        o.append(mono(386, y, b, size=7.4))
    o += [
        text(34, 166, "senkron", size=9, colour=RED, weight="bold"),
        text(34, 184, "AG2 bu koldan çatallandı", size=7.6, colour=GREY),
        text(386, 166, "async — await, async for", size=9, colour=GREEN, weight="bold"),
        text(386, 184, "halefi: Microsoft Agent Framework", size=7.6, colour=GREY),
        text(320, 100, "sıfırdan", size=7.6, colour=ORANGE, anchor="middle"),
        text(320, 130, "yeniden yazım", size=7.6, colour=ORANGE, anchor="middle"),
    ]
    return figure(640, 210, "".join(o))


def fig_ladder() -> str:
    p, o = Pen(303), []
    steps = [
        (176, "1 · model istemcisi", "modelle konuşan tek nokta", "green"),
        (260, "2 · mesajlar", "iki aile: iletişim / iç olay", "green"),
        (344, "3 · ajanlar", "AssistantAgent + tool + bağlam", "orange"),
        (428, "4 · takımlar", "birden çok ajanı yönetmek", "orange"),
        (512, "5 · insan döngüde", "döngüye insanı sokmak", "blue"),
        (596, "6 · sonlandırma → 7 · durum", "koşulsuz takım = gerçek fatura", "blue"),
    ]
    for i, (w, label, note, colour) in enumerate(steps):
        y = 26 + i * 34
        o.append(p.rect(20, y, w, 30, colour, width=1.6))
        o.append(text(34, y + 20, label, size=9, colour=INK, weight="bold"))
        o.append(text(w + 34, y + 20, note, size=7.8, colour=GREY))
    return figure(640, 244, "".join(o))


def fig_client() -> str:
    p, o = Pen(404), []
    o.append(p.rect(22, 38, 122, 46, "orange", width=1.8))
    o.append(p.rect(200, 38, 172, 46, "violet", width=1.8))
    o.append(p.rect(428, 38, 150, 46, "blue", width=1.8))
    o.append(p.line(144, 61, 196, 61))
    o.append(p.line(372, 61, 424, 61))
    o.append(p.rect(150, 116, 440, 82, "red", width=2))
    o.append(p.line(286, 84, 286, 112, "red", dash="4 3"))
    o += [
        text(83, 58, "AssistantAgent", size=9, colour=INK, weight="bold", anchor="middle"),
        text(83, 74, "modeli hiç görmez", size=7.6, colour=GREY, anchor="middle"),
        text(286, 58, "ChatCompletionClient", size=9, colour=INK, weight="bold", anchor="middle"),
        text(286, 74, "önbellek · usage sayacı · yapısal çıktı", size=7.4, colour=GREY, anchor="middle"),
        text(503, 58, "endpoint", size=9, colour=INK, weight="bold", anchor="middle"),
        text(503, 74, "OpenAI · Azure · Anthropic · Ollama", size=7.4, colour=GREY, anchor="middle"),
        text(166, 136, "EN PAHALI TUZAK — ve kılavuzda yazmıyor", size=9, colour=RED, weight="bold"),
        text(166, 152, "Model adı bilinen bir OpenAI modeli değilse model_info ZORUNLU:", size=8),
        mono(166, 168, "ValueError: model_info is required when model name is", size=7.4, colour="#8c2f1d"),
        mono(166, 179, "not a valid OpenAI model", size=7.4, colour="#8c2f1d"),
        text(166, 194, "Ve model_info bir BEYANDIR, ölçüm değil — yanlış beyan huninin sonunda patlar.", size=8),
    ]
    return figure(640, 210, "".join(o))


def fig_messages() -> str:
    p, o = Pen(505), []
    o.append(p.rect(18, 22, 296, 158, "green", width=1.8))
    o.append(p.rect(326, 22, 296, 158, "blue", width=1.8))
    o.append(p.rect(18, 192, 604, 34, "orange", width=1.6))
    chat = ["TextMessage", "MultiModalMessage", "HandoffMessage", "StopMessage",
            "ToolCallSummaryMessage", "StructuredMessage[T]"]
    events = ["ToolCallRequestEvent", "ToolCallExecutionEvent",
              "ModelClientStreamingChunkEvent", "MemoryQueryEvent", "UserInputRequestedEvent"]
    o += [
        text(34, 42, "BaseChatMessage", size=9.6, colour=GREEN, weight="bold"),
        text(34, 57, "ajanlar arası İLETİŞİM", size=7.8, colour=GREY),
        text(342, 42, "BaseAgentEvent", size=9.6, colour=BLUE, weight="bold"),
        text(342, 57, "ajanın İÇ OLAYLARI", size=7.8, colour=GREY),
    ]
    for i, s in enumerate(chat):
        o.append(mono(34, 80 + i * 16, s, size=8))
    for i, s in enumerate(events):
        o.append(mono(342, 80 + i * 16, s, size=8))
    o.append(text(342, 170, "arayüzde tool çağrılarının ayrı satır olması bu sayede", size=7.4, colour=GREY))
    o.append(text(34, 206, "TUZAK · Takım, tanımadığı mesaj tipini YÖNLENDİRMEZ", size=8.4,
                  colour="#8c2f1d", weight="bold"))
    o.append(mono(34, 219, 'GraphFlow(..., custom_message_types=[StructuredMessage[Score]])', size=7.6))
    return figure(640, 234, "".join(o))


def fig_toolloop() -> str:
    p, o = Pen(606), []
    boxes = [(26, "görev", "orange"), (186, "model çağrısı", "violet"),
             (346, "tool çalışır", "green"), (506, "cevap", "orange")]
    for x, label, colour in boxes:
        o.append(p.rect(x, 34, 108, 40, colour, width=1.8))
        o.append(text(x + 54, 59, label, size=8.6, colour=INK, weight="bold", anchor="middle"))
    for x in (134, 294, 454):
        o.append(p.line(x, 54, x + 48, 54))
    o.append(p.curve([(400, 74), (400, 106), (80, 106), (80, 78)], "red", width=1.7, dash="5 3"))
    o.append(text(240, 122, "bu dönüş varsayılan olarak YOKTUR", size=8, colour=RED,
                  weight="bold", anchor="middle"))
    o.append(p.rect(150, 140, 424, 64, "red", width=2))
    o += [
        text(166, 160, "max_tool_iterations varsayılanı = 1", size=9, colour=RED, weight="bold"),
        text(166, 177, "Ajan bir tool çağırır, sonucu görür, SUSAR. \"Önce ara, sonra incele\" gibi", size=8),
        text(166, 192, "zincirleme davranış varsayılanla imkânsız — ve hata vermez.", size=8),
    ]
    return figure(640, 214, "".join(o))


def fig_teams() -> str:
    p, o = Pen(707), []

    # RoundRobin
    o.append(p.rect(14, 22, 196, 138, "grey", width=1.6))
    for cx, cy in ((66, 90), (112, 62), (158, 90)):
        o.append(p.circle(cx, cy, 16))
    o.append(p.line(81, 80, 97, 70))
    o.append(p.line(127, 70, 143, 80))
    o.append(p.curve([(151, 104), (112, 124), (74, 106)]))
    o += [text(26, 40, "RoundRobinGroupChat", size=8.8, colour=INK, weight="bold"),
          text(26, 140, "sırayla · kimseyi atlayamaz", size=7.4, colour=GREY),
          text(26, 153, "274 token", size=8.6, colour="#8c2f1d", weight="bold")]

    # Selector
    o.append(p.rect(222, 22, 196, 138, "green", width=1.8))
    o.append(p.rect(292, 46, 58, 22, "green", width=1.5))
    for cx in (262, 320, 378):
        o.append(p.circle(cx, 112, 15))
    o.append(p.line(306, 68, 270, 96, "green", width=1.3))
    o.append(p.line(320, 68, 320, 96, "green", width=1.3))
    o.append(p.line(336, 68, 370, 96, "green", width=1.3, dash="3 2", arrow=False))
    o += [text(234, 40, "SelectorGroupChat", size=8.8, colour=GREEN, weight="bold"),
          text(321, 61, "seçici", size=7.4, colour=INK, weight="bold", anchor="middle"),
          text(234, 140, "description okur · selector_func = LLM'siz", size=7.4, colour=GREY),
          text(234, 153, "204 token — en ucuz", size=8.6, colour=GREEN, weight="bold")]

    # Swarm
    o.append(p.rect(14, 172, 196, 138, "red", width=1.8))
    o.append(p.circle(60, 228, 16))
    o.append(p.circle(122, 228, 16))
    o.append(p.circle(180, 228, 14, "red", dash="3 2"))
    o.append(p.line(77, 228, 104, 228, "red", width=1.5))
    o.append(p.line(139, 228, 164, 228, "red", width=1.5))
    o += [text(26, 190, "Swarm (handoff)", size=8.8, colour=RED, weight="bold"),
          text(90, 214, "transfer_to_", size=6.8, colour=RED, anchor="middle"),
          text(180, 252, "user", size=6.8, colour=RED, anchor="middle"),
          text(26, 278, "ajan kendi devreder", size=7.4, colour=GREY),
          text(26, 290, "her devir = 1 tool + 1 boş LLM turu", size=7.4, colour=GREY),
          text(26, 303, "334 token — en pahalı", size=8.6, colour=RED, weight="bold")]

    # GraphFlow
    o.append(p.rect(222, 172, 196, 138, "blue", width=1.8))
    o.append(p.circle(252, 228, 14))
    o.append(p.circle(318, 202, 14))
    o.append(p.circle(318, 254, 14))
    o.append(p.circle(386, 228, 15, "blue", width=1.8))
    o.append(p.line(266, 220, 302, 207, "blue", width=1.3))
    o.append(p.line(266, 236, 302, 249, "blue", width=1.3))
    o.append(p.line(333, 207, 370, 220, "blue", width=1.3))
    o.append(p.line(333, 249, 370, 236, "blue", width=1.3))
    o += [text(234, 190, "GraphFlow", size=8.8, colour=BLUE, weight="bold"),
          text(386, 199, 'join "all"', size=6.8, colour=BLUE, anchor="middle"),
          text(234, 278, "önceden çizili graf", size=7.4, colour=GREY),
          text(234, 290, "paralel dal + birleşme", size=7.4, colour=GREY),
          text(234, 303, "270 token", size=8.6, colour=BLUE, weight="bold")]

    # MagenticOne
    o.append(p.rect(430, 172, 196, 138, "violet", width=1.8))
    o.append(p.rect(492, 198, 72, 21, "violet", width=1.4))
    for cx in (458, 500, 542, 584):
        o.append(p.circle(cx, 252, 12))
        o.append(p.line(528, 219, cx + 4, 239, "violet", width=1.1, arrow=False))
    o += [text(442, 190, "MagenticOneGroupChat", size=8.8, colour=VIOLET, weight="bold"),
          text(528, 212, "orkestratör", size=7, colour=INK, weight="bold", anchor="middle"),
          text(442, 282, "surfer · file · coder · terminal", size=7.2, colour=GREY),
          text(442, 296, "kılavuz GÜVENLİK UYARISIYLA veriyor", size=7.4, colour="#8c2f1d",
               weight="bold")]

    # Legend
    o.append(p.rect(430, 22, 196, 138, "grey", width=1.5))
    o += [
        text(444, 42, "Sayılar = token", size=8.6, colour="#8c2f1d", weight="bold"),
        text(444, 60, "Aynı görev, aynı ajanlar,", size=7.6),
        text(444, 73, "yalnız orkestrasyon değişiyor.", size=7.6),
        text(444, 96, "%63,7 fark", size=13, colour=RED, weight="bold"),
        text(444, 116, "Ödenen şey zekâ değil,", size=7.4, colour=GREY),
        text(444, 128, "YÖNLENDİRME ÖZERKLİĞİ.", size=7.4, colour=GREY),
        text(444, 148, "ölçüm: poc/kiyas.py", size=7, colour=GREY),
    ]
    return figure(640, 318, "".join(o))


def fig_termination() -> str:
    p, o = Pen(808), []
    o.append(p.rect(30, 24, 200, 34, "blue", width=1.6))
    o.append(p.rect(292, 24, 218, 34, "blue", width=1.6))
    o.append(p.rect(30, 74, 480, 32, "green", width=1.6))
    o += [
        mono(42, 46, "MaxMessageTermination(10)", size=8),
        text(252, 48, "|", size=15, colour=ORANGE, weight="bold"),
        text(266, 46, "veya", size=8, colour=GREY),
        mono(304, 46, 'TextMentionTermination("TERMINATE")', size=8),
        text(42, 95, "& ile de birleşir — ikisi birden gerçekleşince durur", size=8.4,
             colour=GREEN, weight="bold"),
        text(524, 44, "bizim her", size=8, colour=GREY),
        text(524, 58, "takımımızda", size=8, colour=GREY),
        text(524, 74, "MaxMessage", size=8, colour="#8c2f1d", weight="bold"),
        text(524, 88, "sigortası var", size=8, colour="#8c2f1d", weight="bold"),
    ]
    return figure(640, 118, "".join(o))


def fig_human() -> str:
    p, o = Pen(909), []
    o.append(p.rect(16, 22, 296, 146, "red", width=1.8))
    o.append(p.rect(328, 22, 296, 146, "green", width=1.8))
    o.append(p.rect(44, 74, 88, 30))
    o.append(p.rect(190, 74, 88, 30, "red", width=1.6))
    o.append(p.line(132, 89, 186, 89))
    o.append(p.rect(352, 60, 84, 28))
    o.append(p.rect(488, 60, 108, 28, "green", width=1.6))
    o.append(p.line(436, 74, 484, 74, "green", width=1.3))
    o.append(p.curve([(542, 88), (542, 122), (396, 122), (394, 90)], "green", dash="4 3"))
    o += [
        text(32, 42, "1 · UserProxyAgent", size=9.2, colour=RED, weight="bold"),
        text(88, 93, "takım", size=8, colour=INK, anchor="middle"),
        text(234, 93, "insan", size=8, colour=RED, anchor="middle"),
        text(32, 136, "Takımın içinde bir ajan gibi durur.", size=7.8),
        text(32, 152, "Basit — ama takımı BLOKLAR.", size=8.4, colour=RED, weight="bold"),
        text(344, 42, "2 · bitir, geri dön  (önerilen)", size=9.2, colour=GREEN, weight="bold"),
        text(394, 79, "ajan", size=8, colour=INK, anchor="middle"),
        text(542, 78, "HandoffTermination", size=7.2, colour=GREEN, anchor="middle"),
        text(344, 136, "Takım durur, cevabı alırsın, run() tekrar.", size=7.8),
        text(344, 152, "Uzun insan beklemesinde takımı ayakta", size=8.4, colour=GREEN, weight="bold"),
        text(344, 164, "tutmak kırılgandır.", size=8.4, colour=GREEN, weight="bold"),
    ]
    return figure(640, 178, "".join(o))


def fig_graphflow() -> str:
    p, o = Pen(1010), []
    o.append(p.rect(16, 20, 300, 150, "blue", width=1.8))
    o.append(p.rect(326, 20, 298, 150, "red", width=1.8))
    # clean
    o.append(p.circle(52, 96, 14))
    for cy in (62, 96, 130):
        o.append(p.circle(140, cy, 14))
        o.append(p.line(66, 96 + (cy - 96) * 0.22, 125, cy, "blue", width=1.3))
        o.append(p.line(155, cy, 232, 96 + (cy - 96) * 0.2, "blue", width=1.3))
    o.append(p.circle(250, 96, 16, "blue", width=1.9))
    # broken
    o.append(p.circle(362, 96, 14))
    for cy, colour in ((62, "grey"), (96, "grey"), (130, "red")):
        o.append(p.circle(450, cy, 14, colour))
        o.append(p.line(376, 96 + (cy - 96) * 0.22, 435, cy, colour, width=1.3))
    o.append(p.cross(450, 130))
    o.append(p.circle(560, 96, 16, "red", width=1.9, dash="4 3"))
    o += [
        text(30, 38, 'temiz koşu — join "all"', size=9, colour=BLUE, weight="bold"),
        text(250, 100, "join", size=7.4, colour=INK, weight="bold", anchor="middle"),
        text(30, 158, "üç dal koşar, birleşme üçünü de bekler", size=8),
        text(340, 38, "bir dal fırlatınca", size=9, colour=RED, weight="bold"),
        text(560, 101, "?", size=10, colour=RED, weight="bold", anchor="middle"),
        text(340, 158, "TAMAMLANMIŞ kardeşlerin işi de gider", size=8, colour=RED, weight="bold"),
    ]
    o.append(p.rect(16, 182, 608, 58, "grey", width=1.5))
    o += [
        text(30, 200, "ÖLÇÜLDÜ · pipeline/compare_fanin.py · aynı arıza enjeksiyonu",
             size=8, colour="#8c2f1d", weight="bold"),
        text(30, 217, "GraphFlow (AgentChat)", size=8),
        text(190, 217, "temiz 3 · sarmalayıcı arkasında 2 ·", size=8),
        text(390, 217, "ham hata 0–1 dal, süre sınırı dolar", size=8.4, colour=RED, weight="bold"),
        text(30, 233, "core pub/sub + ClosureAgent", size=8),
        text(190, 233, "temiz 3 · sarmalayıcı arkasında 2 ·", size=8),
        text(390, 233, "ham hata 2 dal, ~3 ms", size=8.4, colour=GREEN, weight="bold"),
    ]
    return figure(640, 248, "".join(o))


def fig_memory() -> str:
    p, o = Pen(1111), []
    o.append(p.rect(16, 22, 296, 136, "orange", width=1.8))
    o.append(p.rect(328, 22, 296, 136, "violet", width=1.8))
    o += [
        text(32, 42, "model_context", size=9.4, colour=ORANGE, weight="bold"),
        text(32, 57, "O KONUŞMANIN son N mesajı", size=7.8, colour=GREY),
        mono(32, 82, "BufferedChatCompletionContext", size=8),
        text(32, 100, "verilmezse → ajan DURUMSUZ, hata da vermez", size=8),
        text(32, 124, "save_state bunu kaydeder", size=8.4, colour=ORANGE, weight="bold"),
        text(32, 142, "bağlam yoksa kaydedilecek sohbet de yok", size=7.6, colour=GREY),
        text(344, 42, "Memory protokolü", size=9.4, colour=VIOLET, weight="bold"),
        text(344, 57, "KONUŞMALAR ARASI kalıcı bilgi", size=7.8, colour=GREY),
        mono(344, 82, "ListMemory · ChromaDBVectorMemory · Mem0Memory", size=7.2),
        text(344, 100, "her turda sorgulanır, dönen içerik bağlama eklenir", size=8),
        text(344, 124, "MemoryQueryEvent akışta görünür", size=8.4, colour=VIOLET, weight="bold"),
        text(344, 142, "\"ajan neyi hatırladı\" izlenebilir", size=7.6, colour=GREY),
    ]
    return figure(640, 168, "".join(o))


def fig_logging() -> str:
    p, o = Pen(1212), []
    o.append(p.rect(24, 24, 128, 30, "green", width=1.6))
    o.append(p.rect(24, 80, 128, 30, "red", width=1.6))
    for i, y in enumerate((24, 60, 96)):
        o.append(p.rect(250, y, 152, 30))
    o.append(p.line(152, 39, 246, 39, "green"))
    o.append(p.line(152, 88, 246, 75, "red"))
    o.append(p.line(152, 95, 246, 111, "red"))
    o += [
        mono(38, 43, "create()", size=8.2, colour=GREEN),
        mono(34, 99, "create_stream()", size=8.2, colour=RED),
        mono(262, 43, "LLMCallEvent", size=8.2, colour=INK),
        mono(262, 79, "LLMStreamEndEvent", size=8.2, colour=INK),
        mono(262, 115, "ToolCallEvent", size=8.2, colour=INK),
        text(422, 50, "TUZAK · ÖLÇÜLDÜ", size=8.6, colour="#8c2f1d", weight="bold"),
        text(422, 68, "Akış kullanınca yalnız", size=7.8),
        text(422, 81, "LLMStreamEndEvent yayılır.", size=7.8),
        text(422, 99, "Sadece LLMCallEvent", size=7.8, colour=RED, weight="bold"),
        text(422, 112, "dinlersen maliyet 0 görünür.", size=7.8, colour=RED, weight="bold"),
    ]
    return figure(640, 148, "".join(o))


# ------------------------------------------------------------------ core (§17-)


def fig_identity() -> str:
    p, o = Pen(1313), []
    o.append(p.rect(16, 20, 296, 150, "blue", width=1.8))
    o.append(p.rect(326, 20, 298, 150, "orange", width=1.8))
    o.append(p.rect(48, 56, 230, 34, "blue", width=1.6))
    o += [
        text(32, 40, "kimlik = (type, key)", size=9.4, colour=BLUE, weight="bold"),
        mono(62, 78, 'AgentId("analist", "sirket-42")', size=8.6, colour=INK),
        text(32, 112, "type — hangi sınıf / rol", size=8.2),
        text(32, 128, "key  — hangi örnek", size=8.2),
        text(32, 152, "type regex'ten geçer, key HİÇ doğrulanmaz", size=8, colour=RED,
             weight="bold"),
    ]
    o.append(p.rect(348, 54, 120, 30, "orange", width=1.5))
    o.append(p.rect(496, 54, 108, 30, "grey", width=1.5, dash="4 3"))
    o.append(p.line(468, 69, 492, 69, "orange", width=1.4))
    o.append(p.line(556, 88, 556, 116, "orange", width=1.4))
    o.append(p.rect(496, 118, 108, 30, "orange", width=1.6))
    o += [
        text(342, 40, "tembel yaratım", size=9.4, colour=ORANGE, weight="bold"),
        mono(358, 73, "register(type,", size=7.6, colour=INK),
        mono(358, 82, "         fabrika)", size=7.6, colour=INK),
        text(550, 66, "örnek yok", size=7.6, colour=GREY, anchor="middle"),
        text(550, 78, "(henüz)", size=7.6, colour=GREY, anchor="middle"),
        text(550, 137, "ilk mesajda doğar", size=7.6, colour=ORANGE, anchor="middle",
             weight="bold"),
        text(342, 162, "ToolCallEvent'in agent_id'si: handler içinde dolu, çıplak run()'da None",
             size=7.4, colour=GREY),
    ]
    return figure(640, 178, "".join(o))


def fig_topic() -> str:
    p, o = Pen(1414), []
    o.append(p.rect(16, 18, 606, 116, "green", width=1.8))
    for i, (src, cy) in enumerate((("sirket-42", 52), ("sirket-99", 92))):
        o.append(p.rect(34, cy - 16, 176, 30, "grey", width=1.5))
        o.append(p.line(212, cy - 1, 268, cy - 1, "green", width=1.5))
        o.append(p.rect(272, cy - 16, 176, 30, "green", width=1.6))
        o.append(mono(44, cy + 4, f'TopicId("gorev","{src}")', size=7.4, colour=INK))
        o.append(mono(282, cy + 4, f'AgentId("analist","{src}")', size=7.4, colour=INK))
    o += [
        text(34, 36, "yayın", size=8, colour=GREY, weight="bold"),
        text(272, 36, "runtime'ın yarattığı ajan örneği", size=8, colour=GREEN, weight="bold"),
        text(466, 60, "topic KAYNAĞI", size=8.4, colour=GREEN, weight="bold"),
        text(466, 74, "ajan ANAHTARINA", size=8.4, colour=GREEN, weight="bold"),
        text(466, 88, "dönüşür", size=8.4, colour=GREEN, weight="bold"),
        text(466, 108, "sözlük tutmadan", size=7.4, colour=GREY),
        text(466, 120, "kaynak başına izolasyon", size=7.4, colour=GREY),
    ]
    o.append(p.rect(16, 146, 606, 74, "grey", width=1.5))
    o += [
        text(30, 164, "CANLI KOŞULDU · autogen-core 0.7.5 · üç yayın, iki örnek",
             size=8, colour="#8c2f1d", weight="bold"),
        mono(30, 181, "publish TopicId('gorev','sirket-42')  ->  key='sirket-42'", size=7.4),
        mono(30, 195, "publish TopicId('gorev','sirket-99')  ->  key='sirket-99'", size=7.4),
        mono(30, 209, "publish TopicId('gorev','sirket-42')  ->  key='sirket-42'  (aynı örnek)", size=7.4),
        text(430, 196, "kayıtlı örnek: 2", size=8.4, colour=GREEN, weight="bold"),
    ]
    return figure(640, 228, "".join(o))


def fig_messaging() -> str:
    p, o = Pen(1515), []
    o.append(p.rect(14, 20, 300, 226, "green", width=1.8))
    o.append(p.rect(326, 20, 298, 226, "orange", width=1.8))
    # direct
    o.append(p.rect(40, 74, 104, 34))
    o.append(p.rect(196, 74, 104, 34))
    o.append(p.line(144, 84, 192, 84))
    o.append(p.curve([(192, 100), (168, 108), (146, 100)], "green", dash="3 2"))
    o += [
        text(28, 40, "DOĞRUDAN MESAJ", size=9.2, colour=GREEN, weight="bold"),
        mono(28, 58, "send_message(msg, AgentId(...))", size=7.6),
        text(92, 96, "OuterAgent", size=8, colour=INK, anchor="middle"),
        text(248, 96, "InnerAgent", size=8, colour=INK, anchor="middle"),
        text(28, 132, "alıcı: tam bir tane", size=8),
        text(28, 148, "dönüş: handler'ın return'ü", size=8),
        text(28, 164, "bağ: sıkı — örneği bilmen gerekir", size=8),
    ]
    o.append(p.rect(28, 180, 272, 52, "green", width=1.7))
    o += [
        text(40, 198, "hata → çağırana FIRLATILIR", size=8.6, colour=GREEN, weight="bold"),
        text(40, 213, '"the exception will be propagated', size=7.6),
        text(40, 226, ' back to the sender"', size=7.6),
    ]
    # broadcast
    o.append(p.rect(352, 74, 96, 34, "orange", width=1.6))
    for i, cy in enumerate((66, 100, 134)):
        o.append(p.circle(556, cy, 13))
        o.append(p.line(450, 88, 540, cy, "orange", width=1.2))
    o += [
        text(340, 40, "YAYIN", size=9.2, colour=ORANGE, weight="bold"),
        mono(340, 58, "publish_message(msg, TopicId(...))", size=7.6),
        text(400, 96, "topic", size=8, colour=INK, anchor="middle"),
        text(340, 156, "alıcı: kim abone ise", size=8),
        text(340, 172, "dönüş: YOK — return yazan uyarı almaz", size=8),
    ]
    o.append(p.rect(340, 180, 272, 52, "orange", width=1.7))
    o += [
        text(352, 198, "hata → LOGLANIR, fırlatılmaz", size=8.6, colour=ORANGE, weight="bold"),
        text(352, 213, "bu bir kısıt değil, tasarım: yayınla", size=7.6),
        text(352, 226, "haberleşen ajan diğerini tanımaz", size=7.6),
    ]
    return figure(640, 254, "".join(o))


def fig_routing() -> str:
    p, o = Pen(1616), []
    o.append(p.rect(24, 26, 130, 34, "grey", width=1.5))
    o.append(p.rect(240, 20, 160, 30))
    o.append(p.rect(240, 60, 160, 30))
    o.append(p.rect(240, 100, 160, 30, "grey", width=1.4, dash="4 3"))
    o.append(p.line(154, 43, 236, 35))
    o.append(p.line(154, 45, 236, 75))
    o.append(p.line(154, 50, 236, 115, "grey", width=1.2, dash="3 2"))
    o += [
        text(38, 47, "gelen mesaj", size=8.4, colour=INK, weight="bold"),
        mono(250, 39, "handle_text(TextMessage)", size=7.4),
        mono(250, 79, "handle_image(ImageMessage)", size=7.4),
        mono(250, 119, "eşleşme yok → düşer", size=7.4, colour=GREY),
        text(414, 34, "1. TİP anotasyonu seçer", size=8.2, colour=INK, weight="bold"),
        text(414, 50, "if bloğu değil, tip sistemi", size=7.6, colour=GREY),
        text(414, 74, "2. sonra match= koşulu", size=8.2, colour=INK, weight="bold"),
        text(414, 90, "handler ADLARININ alfabetik", size=7.6, colour=GREY),
        text(414, 102, "sırasıyla denenir", size=7.6, colour=GREY),
    ]
    o.append(p.rect(24, 148, 592, 62, "red", width=1.9))
    o += [
        text(38, 167, "TUZAK — bizi yakaladı · kayıt sırasında çıplak NameError", size=8.8,
             colour=RED, weight="bold"),
        text(38, 183, "Tip çıkarımı get_type_hints() ile. `from __future__ import annotations` varken", size=8),
        text(38, 197, "MessageContext'i fonksiyon içinde import edersen KAYITTA patlar. Ve parametre", size=8),
        text(38, 208, "adları bağlayıcı: `message` ve `ctx` olmak zorunda.", size=8),
    ]
    return figure(640, 220, "".join(o))


def fig_runtime() -> str:
    p, o = Pen(1717), []
    o.append(p.rect(16, 20, 296, 116, "green", width=1.8))
    o.append(p.rect(326, 20, 298, 116, "grey", width=1.6, dash="5 3"))
    o.append(p.rect(52, 62, 224, 40, "green", width=1.6))
    o += [
        text(32, 40, "Standalone — bizde bu", size=9.2, colour=GREEN, weight="bold"),
        mono(66, 86, "SingleThreadedAgentRuntime", size=8.2, colour=INK),
        text(32, 124, "tek süreç, tek dil", size=7.8, colour=GREY),
    ]
    o.append(p.rect(346, 58, 88, 26, "grey", width=1.4))
    for cx in (470, 540, 604):
        o.append(p.circle(cx, 96, 13, "grey"))
        o.append(p.line(400, 84, cx - 4, 82, "grey", width=1.1, arrow=False))
    o += [
        text(342, 40, "Distributed — bizde YOK", size=9.2, colour=GREY, weight="bold"),
        text(390, 75, "host", size=7.6, colour=INK, anchor="middle"),
        text(342, 124, "gRPC · çok makine · Python ↔ .NET · AJAN KODU DEĞİŞMEZ", size=7.6, colour=GREY),
    ]
    o.append(p.rect(16, 150, 606, 96, "red", width=1.9))
    o += [
        text(30, 169, "ÖLÇÜLDÜ · Runtime'ı KENDİN verirsen hata semantiği değişiyor",
             size=8.8, colour=RED, weight="bold"),
        text(30, 186, "InterventionHandler takmanın tek yolu runtime'ı kendin kurmak. Ama o zaman:", size=8),
        text(44, 202, "çöken ajan  run_stream'i FIRLATMIYOR — ASIYOR   (gömülü runtime'da fırlatıyordu)", size=8, colour=RED),
        text(44, 216, "MaxMessageTermination kurtaramıyor, çünkü yeni mesaj da gelmiyor", size=8),
        text(44, 230, "tek çare: duvar saati sınırı", size=8, colour=RED, weight="bold"),
        text(30, 243, "→ 06 §7", size=7.4, colour=GREY),
    ]
    return figure(640, 254, "".join(o))


def fig_workbench() -> str:
    p, o = Pen(1818), []
    o.append(p.rect(24, 30, 118, 42, "orange", width=1.8))
    o.append(p.rect(200, 30, 150, 42, "violet", width=1.8))
    o.append(p.line(142, 51, 196, 51))
    for i, (label, cy) in enumerate((("stdio", 30), ("SSE", 76), ("streamable HTTP", 122))):
        o.append(p.rect(430, cy, 176, 32, "blue", width=1.5))
        o.append(p.line(350, 51, 424, cy + 16, "violet", width=1.2))
        o.append(text(442, cy + 21, label, size=8.2, colour=INK))
    o += [
        text(83, 48, "AssistantAgent", size=8.6, colour=INK, weight="bold", anchor="middle"),
        text(83, 63, "tek workbench", size=7.4, colour=GREY, anchor="middle"),
        text(275, 48, "Workbench", size=8.6, colour=VIOLET, weight="bold", anchor="middle"),
        text(275, 62, "list_tools() / call_tool()", size=7.2, colour=GREY, anchor="middle"),
        text(430, 172, "MCP taşımaları — ajan yazılırken var olmayan tool'lar listelenebilir",
             size=7.6, colour=GREY),
    ]
    o.append(p.rect(24, 96, 326, 76, "red", width=1.8))
    o += [
        text(38, 115, "tools= ile workbench= AYNI AJANA VERİLEMEZ", size=8.4, colour=RED,
             weight="bold"),
        mono(38, 131, "ValueError: Tools cannot be used", size=7.4, colour="#8c2f1d"),
        mono(38, 142, "            with a workbench.", size=7.4, colour="#8c2f1d"),
        text(38, 159, "Çare: yerel fonksiyonları StaticWorkbench'e sar,", size=7.8),
        text(38, 169, "workbench'e LİSTE ver.", size=7.8),
    ]
    return figure(640, 186, "".join(o))


def fig_intervention() -> str:
    p, o = Pen(1919), []
    o.append(p.rect(24, 40, 104, 36, "orange", width=1.6))
    o.append(p.rect(206, 30, 150, 56, "red", width=2))
    o.append(p.rect(430, 40, 104, 36, "green", width=1.6))
    o.append(p.line(128, 58, 202, 58))
    o.append(p.line(356, 58, 426, 58, "green"))
    o.append(p.curve([(281, 86), (281, 116), (200, 116)], "red", dash="4 3"))
    o += [
        text(76, 62, "mesaj", size=8.4, colour=INK, weight="bold", anchor="middle"),
        text(281, 52, "InterventionHandler", size=8.6, colour=RED, weight="bold", anchor="middle"),
        mono(281, 68, "on_send / on_publish", size=7.2, colour=INK, anchor="middle"),
        mono(281, 79, "on_response", size=7.2, colour=INK, anchor="middle"),
        text(482, 62, "ajan", size=8.4, colour=INK, weight="bold", anchor="middle"),
        mono(118, 122, "return DropMessage", size=8, colour=RED),
        text(556, 54, "geçen mesaj", size=7.6, colour=GREY),
        text(556, 66, "ajana ulaşır", size=7.6, colour=GREY),
        text(556, 84, "denetim kaydı", size=7.6, colour=GREY),
        text(556, 96, "burada yazılır", size=7.6, colour=GREY),
    ]
    return figure(640, 136, "".join(o))


def fig_fanin() -> str:
    p, o = Pen(2020), []
    o.append(p.rect(16, 20, 300, 168, "red", width=1.8))
    o.append(p.rect(326, 20, 298, 168, "green", width=1.8))
    # barrier
    for i, cy in enumerate((66, 100, 134)):
        o.append(p.circle(66, cy, 13, "grey" if i < 2 else "red"))
    o.append(p.rect(130, 56, 26, 92, "red", width=2))
    o.append(p.circle(230, 100, 15, "red", dash="4 3"))
    for cy in (66, 100):
        o.append(p.line(80, cy, 126, cy + (100 - cy) * 0.5, "grey", width=1.2))
    o.append(p.line(80, 134, 126, 120, "red", width=1.3))
    o.append(p.line(158, 100, 213, 100, "red", width=1.3, dash="4 3"))
    o += [
        text(30, 40, "bariyer  ·  stop_when_idle()", size=9, colour=RED, weight="bold"),
        text(143, 52, "|", size=7, colour=RED, anchor="middle"),
        text(30, 164, "Bir handler çökünce gather erken döner,", size=7.8),
        text(30, 177, "bariyer erken açılır, KARDEŞ SONUÇ KAYBOLUR.", size=7.8, colour=RED,
             weight="bold"),
    ]
    # queue
    for i, cy in enumerate((66, 100, 134)):
        o.append(p.circle(376, cy, 13, "grey" if i < 2 else "red"))
        o.append(p.line(390, cy, 440, cy, "green", width=1.2))
    o.append(p.rect(444, 52, 34, 96, "green", width=1.8))
    o.append(p.line(480, 100, 528, 100, "green", width=1.4))
    o.append(p.circle(546, 100, 15, "green", width=1.8))
    o += [
        text(340, 40, "kuyruk  ·  ClosureAgent", size=9, colour=GREEN, weight="bold"),
        text(461, 82, "2", size=9, colour=GREEN, weight="bold", anchor="middle"),
        text(461, 100, "1", size=9, colour=GREEN, weight="bold", anchor="middle"),
        text(546, 104, "sen", size=7.4, colour=INK, anchor="middle"),
        text(340, 164, "Sonuç üretildiği anda yayınlanıyor;", size=7.8),
        text(340, 177, "kuyruk onu ÇOKTAN TUTUYOR.", size=7.8, colour=GREEN, weight="bold"),
    ]
    o.append(p.rect(16, 200, 606, 40, "grey", width=1.5))
    o += [
        text(30, 218, "Güvenilmeyecek bariyer yok, çünkü bariyer yok. Beklenen sonucu SAY,",
             size=8.6, colour=INK, weight="bold"),
        text(30, 233, "runtime'ın \"boşta\" demesini bekleme.  ·  ölçüm: 0–1 dal (bariyer) vs 2 dal, ~3 ms (kuyruk)",
             size=8),
    ]
    return figure(640, 248, "".join(o))


def fig_cookbook() -> str:
    """Problem → tarif. Cookbook baştan sona okunmaz; derdine göre girilir."""
    p, o = Pen(2121), []
    rows = [
        ("Ajanım gerçek dünyada bir şey yapacak", "Tool yürütme için kullanıcı onayı", "6638", "red"),
        ("Faturayı görmem lazım", "LLM kullanımını logger ile takip", "8840", "red"),
        ("Veri dışarı çıkamaz", "Yerel LLM: LiteLLM ve Ollama", "8030", "orange"),
        ("Anahtar config'de duramaz", "Azure OpenAI + AAD kimlik doğrulama", "6490", "orange"),
        ("Zaten LangGraph/LlamaIndex var", "… destekli ajan — ikisi birlikte", "7482", "green"),
        ("Çok kiracılı olacak", "Topic abonelik senaryoları", "8233", "green"),
        ("Cevap şemaya uymalı", "GPT-4o ile yapısal çıktı", "8745", "blue"),
        ("Nerede yavaşlıyor bilmiyorum", "Kodunu yerelde izleme (OTel + Jaeger)", "8192", "blue"),
    ]
    o.append(text(28, 20, "DERDİN", size=7.4, colour=GREY, weight="bold"))
    o.append(text(300, 20, "TARİF", size=7.4, colour=GREY, weight="bold"))
    o.append(text(590, 20, "satır", size=7.4, colour=GREY, weight="bold"))
    for i, (problem, recipe, line, colour) in enumerate(rows):
        y = 32 + i * 30
        o.append(p.rect(24, y, 250, 24, colour, width=1.4))
        o.append(p.line(276, y + 12, 296, y + 12, colour, width=1.2))
        o.append(text(34, y + 16, problem, size=7.8, colour=INK))
        o.append(text(300, y + 16, recipe, size=7.8, colour=INK, weight="bold"))
        o.append(mono(584, y + 16, f"05:{line}", size=7.2, colour=GREY))
    return figure(640, 288, "".join(o))


def fig_wrapping() -> str:
    """İki framework birlikte — cookbook'un en az bilinen, en kurtarıcı sayfası."""
    p, o = Pen(2222), []
    o.append(p.rect(16, 22, 606, 128, "green", width=1.9))
    o.append(p.rect(52, 54, 128, 64, "orange", width=1.7))
    o.append(p.rect(250, 54, 150, 64, "violet", width=1.7))
    o.append(p.rect(452, 54, 136, 64, "violet", width=1.7, dash="4 3"))
    o.append(p.line(180, 86, 246, 86, "green", width=1.4))
    o.append(p.line(400, 86, 448, 86, "green", width=1.3, dash="3 2"))
    o += [
        text(32, 42, "RoutedAgent — AutoGen mesajlaşmayı yönetir", size=8.8, colour=GREEN,
             weight="bold"),
        text(116, 80, "@message_handler", size=7.6, colour=INK, anchor="middle"),
        text(116, 96, "topic · abonelik", size=7.4, colour=GREY, anchor="middle"),
        text(325, 80, "LangGraph grafı", size=8.2, colour=INK, weight="bold", anchor="middle"),
        text(325, 96, "iç akışı O yönetir", size=7.4, colour=GREY, anchor="middle"),
        text(520, 80, "LlamaIndex RAG", size=8.2, colour=INK, weight="bold", anchor="middle"),
        text(520, 96, "aynı fikir", size=7.4, colour=GREY, anchor="middle"),
        text(32, 138, "Seçmek zorunda değilsin: dışarısı aktör modeli, içerisi ne istersen.",
             size=8.2, colour=INK),
    ]
    return figure(640, 158, "".join(o))


FIGURES = [
    fig_layers, fig_rewrite, fig_ladder, fig_client, fig_messages, fig_toolloop,
    fig_teams, fig_termination, fig_human, fig_graphflow, fig_memory, fig_logging,
    # core
    fig_identity, fig_topic, fig_messaging, fig_routing, fig_runtime, fig_workbench,
    fig_intervention, fig_fanin,
    # cookbook
    fig_cookbook, fig_wrapping,
]


def main() -> int:
    html = HTML.read_text(encoding="utf-8")
    drawn = [f() for f in FIGURES]
    blocks = list(re.finditer(r"<svg\b.*?</svg>", html, re.S))
    if len(blocks) != len(drawn):
        print(f"figure count mismatch: html has {len(blocks)}, script draws {len(drawn)}")
        return 1
    # Right to left, so earlier spans stay valid as we substitute.
    for block, svg in zip(reversed(blocks), reversed(drawn)):
        html = html[: block.start()] + svg + html[block.end():]
    HTML.write_text(html, encoding="utf-8")
    print(f"{len(drawn)} figures redrawn into {HTML.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
