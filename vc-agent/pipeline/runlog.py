"""Bir turun tamamı, bittikten sonra çizilebilsin diye kaydedilir.

`stages.py` bir turun *içinden* geçen mekanizmaları adlandırıyor ve akarken
yayınlıyor. Bu modül onun eksik yarısı: olaylar akıp gittikten sonra geriye
hiçbir şey kalmıyordu, dolayısıyla "az önce sorduğum soruda ne oldu" sorusu
ancak tur sürerken cevaplanabiliyordu. Burada tur bir **kayıt** hâline geliyor,
ve kayıt bir **grafa** çevriliyor: düğümler ajanlar ve tool'lar, kenarlar
aralarında gerçekten geçen mesaj tipleri.

### Neden ayrı bir ekran

Aynı bilgi sohbetin içinde bir şeritteydi ve orada iki işi birden yapmaya
çalışıyordu: hem cevabı okutmak hem makineyi anlatmak. Anlatım kendi ekranını
hak ediyor — bu dosya o ekranın veri tarafı.

### Evin kuralı burada da geçerli

Ekrandaki her açıklama metni **sunucuda** yaşıyor, tarayıcıda değil. Sebebi
`stages.py`'nin docstring'inde yazılı: JavaScript'e gömülen öğretici metin,
anlattığı koddan bir iki sürümde sessizce uzaklaşıyor ve uzaklaştığında hiçbir
şey bozulmuyor. Bir de ikincisi var — bu ekranın iddiaları destede yazılı
iddialarla aynı olmak zorunda, ve deste bu satır atıflarını taşıyor.

### En önemli tasarım kararı: kayıt uydurmaz

Bir sohbet turunda beş takım tipinden **hiçbiri** kullanılmıyor;
`agent.run_stream()` doğrudan çağrılıyor. Ekranın "Swarm koştu" demesi kolay
olurdu ve yanlış olurdu. Onun yerine beşi de listeleniyor, koşan işaretleniyor,
koşmayanların yanına **neden koşmadığı** yazılıyor. Aynısı sekiz resmî desen
için de geçerli. Sunumda gösterilecek şey bu: sistemin ne yaptığı değil, ne
yapmadığını da bilerek söyleyebilmesi.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import config
import stages as stages_module

# --------------------------------------------------------------- beş takım tipi
#
# AgentChat'in sunduğu takımların tamamı. "seçici" sütunu tek ayırt edici soruyu
# cevaplıyor: sırayı kim belirliyor. Bir takımı diğerinden ayıran şey ajanları
# değil, konuşma sırasının nereden geldiği.
#
# Ölçülmüş bedel (`poc/kiyas.py`, aynı görev, yalnız orkestrasyon değişiyor):
# Selector 204 · GraphFlow 270 · RoundRobin 274 · Swarm 334 token — %63,7 fark.
# Ödenen şey zekâ değil, yönlendirme özerkliği.
TEAMS: list[dict[str, Any]] = [
    {
        "id": "roundrobin", "name": "RoundRobinGroupChat", "picker": "sırayla",
        "ref": "08:1789", "cost": 274,
        "what": "Sıra sabit bir döngüde dolaşıyor. Model sıraya karışmıyor, "
                "dolayısıyla en ucuz ve en öngörülebilir takım.",
    },
    {
        "id": "selector", "name": "SelectorGroupChat", "picker": "model seçer",
        "ref": "08:1908", "cost": 204,
        "what": "Her turdan önce bir model çağrısı 'şimdi kim konuşsun' diye "
                "soruyor. Ajanların description'ı boşsa seçim kör yapılıyor.",
    },
    {
        "id": "swarm", "name": "Swarm", "picker": "handoff",
        "ref": "08:2093", "cost": 334,
        "what": "Sırayı ajanın kendisi devrediyor: devir bir tool çağrısı olarak "
                "gerçekleşiyor. En pahalı desen, çünkü her devir bir tur.",
    },
    {
        "id": "magenticone", "name": "MagenticOneGroupChat", "picker": "planlayıcı",
        "ref": "08:2210", "cost": None,
        "what": "Yönetici İKİ defter tutuyor: olguları ve planı taşıyan görev "
                "defteri, ve her adımda kendine bakan ilerleme defteri. "
                "İlerleme durursa görev defterini güncelleyip yeni bir plan "
                "yapıyor. Ölçüldü: aynı soruda 110 sn ve 88 span — açık ara en "
                "pahalısı, çünkü her turda bir de yönetici konuşuyor.",
    },
    {
        "id": "graphflow", "name": "GraphFlow", "picker": "DAG",
        "ref": "08:5398", "cost": 270,
        "what": "Sırayı önceden çizilmiş bir yönlü graf belirliyor. Gerçek "
                "eşzamanlılığı olan tek takım: birden çok dal aynı anda koşuyor.",
    },
]

# --------------------------------------------------------- sekiz resmî desen
#
# `docs/05`'in desen listesi, satır numaraları dosyadan okundu (uydurulmadı —
# bu tablo bir kez kılavuza bakmadan yazıldı ve üç satırı hayaliydi).
#
# Sekizincisi diğerlerinden farklı: Code Execution bir orkestrasyon deseni değil,
# bir yetenek. Listede duruyor çünkü kılavuz onu oraya koymuş.
PATTERNS: list[dict[str, Any]] = [
    {"id": "concurrent", "name": "Concurrent Agents", "ref": "05:3236",
     "what": "tek yayın → çok dal → toplayıcı"},
    {"id": "sequential", "name": "Sequential Workflow", "ref": "05:3504",
     "what": "her ajan bir sonrakine devrediyor"},
    {"id": "groupchat", "name": "Group Chat", "ref": "05:3772",
     "what": "bir yönetici konuşma sırasını dağıtıyor"},
    {"id": "handoffs", "name": "Handoffs", "ref": "05:4349",
     "what": "ajan işi kendisi devrediyor"},
    {"id": "mixture", "name": "Mixture of Agents", "ref": "05:4989",
     "what": "aynı soru, farklı uzmanlar, birleştirici"},
    {"id": "debate", "name": "Multi-Agent Debate", "ref": "05:5358",
     "what": "birden çok tur karşılıklı eleştiri"},
    {"id": "reflection", "name": "Reflection", "ref": "05:5822",
     "what": "üretici + eleştirmen, kalite döngüsü"},
    {"id": "codeexec", "name": "Code Execution", "ref": "05:6188",
     "what": "modelin yazdığı kod bir yürütücüde koşuyor"},
]

# ------------------------------------------------------------- mesaj tipleri
#
# Kenarların üstünde yazan şey. Bir tur boyunca hangi tipin geçtiği, o turun ne
# tür bir makine olduğunu tek başına söylüyor: `ToolCallRequestEvent` gördüysen
# modelin karar verdiği bir tool döngüsü vardır, `StructuredMessage[...]`
# gördüysen şema zorlanmıştır.
MESSAGES: dict[str, dict[str, str]] = {
    "TextMessage": {
        "lane": "agentchat", "ref": "08:0731",
        "what": "Düz metin. Kullanıcının sorusu ve ajanın nihai cevabı bu tiple taşınıyor.",
    },
    "ModelClientStreamingChunkEvent": {
        "lane": "agentchat", "ref": "08:1043",
        "what": "Token akışı. Yalnız `model_client_stream=True` iken doğuyor; "
                "kapalıyken cevap tek parça gelir ve arayüz donmuş görünür.",
    },
    "ToolCallRequestEvent": {
        "lane": "agentchat", "ref": "08:0894",
        "what": "Model bir tool çağırmaya karar verdi. Henüz hiçbir şey koşmadı — "
                "istek ile yürütme ayrı olaylar, ve kapı tam aralarına giriyor.",
    },
    "ToolCallExecutionEvent": {
        "lane": "agentchat", "ref": "08:0921",
        "what": "Tool koştu ve sonucunu döndürdü. Reddedilen çağrı da bu tiple "
                "dönüyor: red bir istisna değil, hata işaretli bir sonuç.",
    },
    "ToolCallSummaryMessage": {
        "lane": "agentchat", "ref": "08:2298",
        "what": "Ham tool çıktısının doğrudan kullanıcıya gitmesi. "
                "`reflect_on_tool_use=False` varsayılanının görünen yüzü.",
    },
    "StructuredMessage[Score]": {
        "lane": "agentchat", "ref": "08:5398",
        "what": "Şemaya bağlanmış mesaj. Takıma `custom_message_types` ile beyan "
                "edilmezse runtime 'is not registered' diye düşüyor.",
    },
    "TaskResult": {
        "lane": "agentchat", "ref": "08:2813",
        "what": "Turun sonu. İki şey taşıyor: bütün konuşma, ve neden durduğu.",
    },
}


# ------------------------------------------------- aşama → grafın neresi
#
# Canlı vurgunun tablosu: bir aşama koşarken grafın hangi parçası yanmalı.
# Sunucuda duruyor, tarayıcıda değil — aşama adı ile graf düğümü arasındaki
# eşleşme bir *anlam* kararı, ve `stages.py` ile birlikte değişiyor.
#
# `"tool"` ve `"request"` sabit bir düğüm değil, "en son olan" demek: bir turda
# birden çok tool çağrısı var ve yanması gereken, o an koşan.
STAGE_TARGET: dict[str, tuple[str, str]] = {
    # --- üst bant: ajan mekanizması
    "model": ("node", "agent"),
    "stream": ("node", "agent"),
    "loop": ("node", "agent"),
    "tool_request": ("edge", "request"),
    "done": ("node", "answer"),
    # --- alt bant: gateway hattı. Bunlar AutoGen'in özellikleri değil, ve
    #     ışığın onları ajan kutusunda yakması tam da bu ayrımı silerdi.
    "context": ("node", "ctx"),
    "compaction": ("node", "ctx"),
    "gate": ("node", "gate"),
    "tool_exec": ("node", "tool"),
    "tool_result": ("node", "tool"),
    "code_request": ("node", "exec"),
    "code_result": ("node", "exec"),
    # --- tarama
    "graph_build": ("node", "start"),
    "graph_run": ("node", "branches"),
    "analysts": ("node", "branches"),
    "join": ("node", "risk"),
    "count": ("node", "scorer"),
    "intervention": ("node", "audit"),
    "runtime_start": ("node", "runtime"),
    "subscribe": ("node", "topic"),
    "publish": ("node", "topic"),
    "runtime_stop": ("node", "runtime"),
    # Zamanlama.
    "cron_parse": ("node", "parse"),
    "cron_gate": ("node", "gate"),
    "cron_done": ("node", "cron"),
    # Takım tarafı. `team_tool`'un hedefi koşuya göre değişiyor (hangi ajan,
    # hangi tool) — bu yüzden sabit değil, `_live_target` çözüyor.
    "speaker": ("node", "*speaker"),
    "team_tool": ("node", "*team_tool"),
    "handoff": ("node", "*speaker"),
}


# ------------------------------------------------------------- iç mimari
#
# Graf, kutuların birbirine nasıl bağlandığını gösteriyor. Bu tablo kutunun
# **içini** gösteriyor: bir `AssistantAgent`'ın içinde ne var, hangi parça
# hangisine bağlı. Ajanlarda özellikle işe yarıyor — dışarıdan tek kutu, içeride
# dört ayrı karar noktası.
#
# Veri olarak duruyor, çizim olarak değil: tarayıcı tarafında tek bir küçük
# çizici var ve bütün iç mimariler ondan geçiyor. On tane elle çizilmiş şema
# yazmak, on tanesini birbirinden ayrı bakımda tutmak demekti.
INNER: dict[str, dict[str, Any]] = {
    "agent": {
        "title": "AssistantAgent · içi",
        "nodes": [
            {"id": "sys", "name": "system_message", "sub": "sabit metin", "kind": "in"},
            {"id": "ctx", "name": "model_context", "sub": "geçmiş + bütçe", "kind": "ours"},
            {"id": "client", "name": "model_client", "sub": "create_stream()", "kind": "ext"},
            {"id": "wb", "name": "workbench", "sub": "tool kaynağı", "kind": "core"},
            {"id": "loop", "name": "tool döngüsü", "sub": "max_tool_iterations=6",
             "kind": "agent"},
        ],
        "edges": [
            {"src": "sys", "dst": "client", "label": "her turda"},
            {"src": "ctx", "dst": "client", "label": "seçilmiş mesajlar"},
            {"src": "client", "dst": "loop", "label": "karar"},
            {"src": "loop", "dst": "wb", "label": "tool çağrısı"},
            {"src": "wb", "dst": "ctx", "label": "sonuç geri", "back": True},
        ],
        "note": "Dışarıdan tek kutu, içeride dört ayrı karar noktası. Bellek "
                "`model_context`'te yaşıyor; verilmezse ajanın belleği hiç olmuyor "
                "ve bu hata da vermiyor.",
    },
    "tool": {
        "title": "FunctionTool · içi",
        "nodes": [
            {"id": "sig", "name": "imza + docstring", "sub": "Python", "kind": "in"},
            {"id": "schema", "name": "JSON şema", "sub": "modele giden", "kind": "core"},
            {"id": "fn", "name": "fonksiyon", "sub": "gerçek iş", "kind": "agent"},
        ],
        "edges": [
            {"src": "sig", "dst": "schema", "label": "üretiliyor"},
            {"src": "schema", "dst": "fn", "label": "argümanlarla"},
        ],
        "note": "Docstring dokümantasyon değil, **arayüz**: modelin bu tool'u ne "
                "zaman ve nasıl çağıracağına karar verdiği metin o.",
    },
    "ctx": {
        "title": "CompactingChatCompletionContext · içi",
        "nodes": [
            {"id": "hist", "name": "geçmiş", "sub": "bütün tur", "kind": "in"},
            {"id": "count", "name": "token sayacı", "sub": "bizim", "kind": "ours"},
            {"id": "out", "name": "seçilmiş küme", "sub": "modele giden", "kind": "core"},
            {"id": "sum", "name": "özet", "sub": "aşarsa", "kind": "ours"},
        ],
        "edges": [
            {"src": "hist", "dst": "count", "label": "ölçülüyor"},
            {"src": "count", "dst": "out", "label": "bütçeye sığan"},
            {"src": "count", "dst": "sum", "label": "aşan kısım"},
        ],
        "note": "AutoGen'in `BufferedChatCompletionContext`'i **mesaj** sayar. Bu "
                "**token** sayıyor — bütçe token cinsinden konuşulduğu için.",
    },
    "gate": {
        "title": "GatedWorkbench · içi",
        "nodes": [
            {"id": "call", "name": "call_tool()", "sub": "istek", "kind": "in"},
            {"id": "hooks", "name": "before_tool_call", "sub": "kanca zinciri",
             "kind": "ours"},
            {"id": "inner", "name": "iç workbench", "sub": "sarmalanan", "kind": "core"},
            {"id": "err", "name": "hata işaretli sonuç", "sub": "red", "kind": "block"},
        ],
        "edges": [
            {"src": "call", "dst": "hooks", "label": "her çağrı"},
            {"src": "hooks", "dst": "inner", "label": "izin"},
            {"src": "hooks", "dst": "err", "label": "red"},
        ],
        "note": "Red bir **istisna değil**, bir sonuç. İstisna turu düşürürdü; "
                "sonuç, ajanın gerekçeyi okuyup kullanıcıya söylemesine izin veriyor.",
    },
    "wb": {
        "title": "Workbench · içi",
        "nodes": [
            {"id": "static", "name": "StaticWorkbench", "sub": "yerel fonksiyonlar",
             "kind": "core"},
            {"id": "mcp", "name": "McpWorkbench", "sub": "OpenClaw · DeepWiki",
             "kind": "ext"},
            {"id": "list", "name": "tek liste", "sub": "ajan için aynı arayüz",
             "kind": "agent"},
        ],
        "edges": [
            {"src": "static", "dst": "list", "label": "yerel"},
            {"src": "mcp", "dst": "list", "label": "uzak"},
        ],
        "note": "Uzak tool'lar yerel tool'larla aynı listede görünüyor ve aynı "
                "kapıdan geçiyor — federasyonun bedava gelen kısmı bu.",
    },
    "exec": {
        "title": "PythonCodeExecutionTool · içi",
        "nodes": [
            {"id": "code", "name": "modelin kodu", "sub": "onaylanan metin", "kind": "in"},
            {"id": "ctr", "name": "konteyner", "sub": "python:3-slim", "kind": "ext"},
            {"id": "net", "name": "ağ", "sub": "AÇIK", "kind": "block"},
            {"id": "out", "name": "stdout + exit", "sub": "sonuç", "kind": "core"},
        ],
        "edges": [
            {"src": "code", "dst": "ctr", "label": "yazılıyor"},
            {"src": "ctr", "dst": "out", "label": "koşuyor"},
            {"src": "ctr", "dst": "net", "label": "erişebiliyor"},
        ],
        "note": "`DockerCommandLineCodeExecutor`'da `network_mode` diye bir "
                "parametre yok — ölçüldü. Konteyner izole ama ağı var.",
    },
    "analyst": {
        "title": "Analist ajanı · içi",
        "nodes": [
            {"id": "sys", "name": "kendi prompt'u", "sub": "alan tarifi", "kind": "in"},
            {"id": "client", "name": "model_client", "sub": "paylaşılan", "kind": "ext"},
            {"id": "out", "name": "metin", "sub": "TextMessage", "kind": "agent"},
        ],
        "edges": [
            {"src": "sys", "dst": "client", "label": "alan + görev"},
            {"src": "client", "dst": "out", "label": "tek geçiş"},
        ],
        "note": "Üç analistin farkı yalnız prompt'u. Aynı model, aynı istemci — "
                "eşzamanlılık ajandan değil, graftan geliyor.",
    },
    "risk": {
        "title": "Risk denetçisi · içi",
        "nodes": [
            {"id": "wait", "name": "bariyer", "sub": 'activation_condition="all"',
             "kind": "ours"},
            {"id": "count", "name": "dal sayacı", "sub": "beklenen 3", "kind": "ours"},
            {"id": "miss", "name": "missing_data", "sub": "gelmeyenler", "kind": "block"},
            {"id": "out", "name": "denetim metni", "sub": "TextMessage", "kind": "agent"},
        ],
        "edges": [
            {"src": "wait", "dst": "count", "label": "sayıyor"},
            {"src": "count", "dst": "out", "label": "gelenlerle"},
            {"src": "count", "dst": "miss", "label": "gelmeyen"},
        ],
        "note": "Beklenen dal **sayısı** sayılıyor; runtime'ın \"boşta\" demesi "
                "beklenmiyor — o bariyer bir dal çökünce erken açılıyor.",
    },
    "scorer": {
        "title": "Skorlayıcı · içi",
        "nodes": [
            {"id": "sys", "name": "prompt", "sub": "skorlama kuralı", "kind": "in"},
            {"id": "schema", "name": "Score şeması", "sub": "output_content_type",
             "kind": "ours"},
            {"id": "out", "name": "StructuredMessage", "sub": "doğrulanmış",
             "kind": "ext"},
        ],
        "edges": [
            {"src": "sys", "dst": "schema", "label": "zorlanıyor"},
            {"src": "schema", "dst": "out", "label": "alan alan"},
        ],
        "note": "Takıma `custom_message_types=[...]` ile beyan edilmezse runtime "
                "`is not registered` diyerek düşüyor.",
    },
    "runtime": {
        "title": "SingleThreadedAgentRuntime · içi",
        "nodes": [
            {"id": "box", "name": "mailbox", "sub": "ajan başına", "kind": "core"},
            {"id": "route", "name": "tip yönlendirme", "sub": "@message_handler",
             "kind": "core"},
            {"id": "subs", "name": "abonelik tablosu", "sub": "topic → ajan tipi",
             "kind": "agent"},
        ],
        "edges": [
            {"src": "subs", "dst": "box", "label": "kime gidecek"},
            {"src": "box", "dst": "route", "label": "tipe göre"},
        ],
        "note": "Topic kaynağı ajan anahtarına dönüşüyor: oturum başına izole "
                "örnek, elle sözlük tutmadan.",
    },
    "topic": {
        "title": "group_topic_type · içi",
        "nodes": [
            {"id": "pub", "name": "yayınlayan", "sub": "kaç abone bilmiyor",
             "kind": "in"},
            {"id": "topic", "name": "tek topic", "sub": "paylaşılan", "kind": "agent"},
            {"id": "all", "name": "beş katılımcı", "sub": "hepsi abone", "kind": "core"},
        ],
        "edges": [
            {"src": "pub", "dst": "topic", "label": "tek yayın"},
            {"src": "topic", "dst": "all", "label": "herkese"},
        ],
        "note": "Grafın kenarları veri taşımıyor: mesaj zaten herkese gitti, graf "
                "yalnız sıranın kimde olduğunu söylüyor.",
    },
    "orchestrator": {
        "title": "MagenticOneOrchestrator · içi",
        "nodes": [
            {"id": "task", "name": "görev defteri", "sub": "olgular + plan", "kind": "ours"},
            {"id": "prog", "name": "ilerleme defteri", "sub": "her adımda", "kind": "agent"},
            {"id": "pick", "name": "sıradaki ajan", "sub": "yönetici seçer", "kind": "core"},
            {"id": "replan", "name": "yeniden plan", "sub": "takılırsa", "kind": "block"},
        ],
        "edges": [
            {"src": "task", "dst": "pick", "label": "plana göre"},
            {"src": "pick", "dst": "prog", "label": "sonuç"},
            {"src": "prog", "dst": "replan", "label": "ilerleme yok"},
            {"src": "replan", "dst": "task", "label": "defteri güncelle", "back": True},
        ],
        "note": "Kadroda olmayan bir ajan: takım onu kendisi yaratıyor. "
                "`MagenticOneGroupChat` yöneticiyi SENİN ajanlarının üstüne "
                "koyuyor; `MagenticOne` preset'i ise kendi kadrosunu getiriyor "
                "(WebSurfer · FileSurfer · Coder · ComputerTerminal) ve asıl "
                "güvenlik uyarıları orada.",
    },
    "audit": {
        "title": "AuditingInterventionHandler · içi",
        "nodes": [
            {"id": "msg", "name": "mesaj", "sub": "teslimattan önce", "kind": "in"},
            {"id": "log", "name": "denetim kaydı", "sub": "kim kime", "kind": "ours"},
            {"id": "drop", "name": "DropMessage", "sub": "teslimat yok",
             "kind": "block"},
        ],
        "edges": [
            {"src": "msg", "dst": "log", "label": "yazılıyor"},
            {"src": "msg", "dst": "drop", "label": "reddedilirse"},
        ],
        "note": "Runtime'a takılan tek kapı. Bedeli: runtime'ı kendin kurmak "
                "zorundasın — ve kendi runtime'ında çöken ajan asılıyor.",
    },
}


# Takım kadrosu — `teams.ROSTER` ile aynı, ama oradan içe aktarmıyoruz:
# `teams` canlı model istemcisi kuruyor ve kayıt katmanının ona bağımlı olması,
# modelsiz bir ortamda raporu okunamaz hâle getirirdi.
TEAM_ROSTER = [{"name": "Planner"}, {"name": "Researcher"}, {"name": "Critic"}]


# Hangi ajan hangi deseni koşuyor.
#
# Bir ajanı "ajan" diye kutuya almak, sorulan soruyu cevaplamıyor: o ajan hangi
# desene göre çalışıyor? Sohbet turumuzda cevap sekiz desenden **biri değil** ve
# bunu yazmak, olmayan bir takımı iddia etmekten daha değerli.
#
# `role`, o düğümün desen içindeki yeri: aynı deseni koşan iki kutu farklı
# roldeyse (dal ile toplayıcı gibi) ikisi aynı şeyi yapmıyor demektir.
NODE_PATTERN: dict[str, dict[str, str]] = {
    "agent": {
        "id": "toolloop", "name": "tool döngüsü", "ref": "08:2298",
        "role": "Sekiz desenden biri değil. Takım kurulmuyor; ajan modeli ve "
                "tool'ları kendi döngüsünde işletiyor.",
    },
    "technical": {
        "id": "concurrent", "name": "Concurrent Agents", "ref": "05:3236",
        "role": "Üç paralel daldan biri. Tek yayın üçünü birden uyandırdı; bu dal "
                "kardeşlerini beklemiyor.",
    },
    "risk": {
        "id": "concurrent", "name": "Concurrent Agents · toplayıcı", "ref": "05:3236",
        "role": "Desenin toplayıcı ucu. Dalları o bekliyor, ve beklemeyi "
                "runtime'ın bariyerine değil kendi sayacına dayandırıyor.",
    },
    "scorer": {
        "id": "sequential", "name": "Sequential Workflow", "ref": "05:3504",
        "role": "Zincirin son halkası. Sırayı kimse seçmiyor; graf önceden yazılı.",
    },
}
NODE_PATTERN["market"] = {**NODE_PATTERN["technical"],
                          "role": "Üç paralel daldan biri — pazar tarafı."}
NODE_PATTERN["team"] = {**NODE_PATTERN["technical"],
                        "role": "Üç paralel daldan biri — ekip tarafı."}


# Hangi kutunun içi hangi şemadan. Düğüm kimliğine göre, çünkü aynı iç mimari
# birden çok kutuda geçebiliyor: üç analist ajanının içi birbirinin aynı.
INNER_OF = {
    "agent": "agent", "ctx": "ctx", "gate": "gate", "wb": "wb", "exec": "exec",
    "trace": None, "user": None, "answer": None, "start": None, "done": None,
    "technical": "analyst", "market": "analyst", "team": "analyst",
    "risk": "risk", "scorer": "scorer",
    "runtime": "runtime", "topic": "topic", "audit": "audit",
    "MagenticOneOrchestrator": "orchestrator",
}


def _attach_inner(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Her düğüme, varsa, iç mimarisini iliştir.

    Tool düğümlerinin kimliği çağrı başına değişiyor (`tool:ad:0`), o yüzden
    kimliğe değil türe bakılıyor.
    """
    for node in nodes:
        key = INNER_OF.get(node["id"], "tool" if node["kind"] == "tool" else None)
        inner = INNER.get(key) if key else None
        if inner:
            node["inner"] = inner
        pattern = NODE_PATTERN.get(node["id"])
        if pattern:
            node["pattern"] = pattern
    return nodes


# ------------------------------------------------------- sıra diyagramı
#
# Kim kime konuşuyor. Şeritler AutoGen'in kendi katmanları değil, bir turda
# gerçekten ayrı duran taraflar: kapı ayrı bir şerit çünkü ayrı bir karar
# noktası, ve onu ajanın içine saklamak diyagramın anlattığı tek şeyi silerdi.
CHAT_LANES = [
    {"id": "user", "name": "Kullanıcı", "lane": ""},
    {"id": "agent", "name": "Ajan", "lane": "agentchat", "sub": "AssistantAgent"},
    {"id": "model", "name": "Model", "lane": "ext", "sub": "ChatCompletionClient"},
    {"id": "gate", "name": "Kapı", "lane": "ours", "sub": "GatedWorkbench"},
    {"id": "tool", "name": "Tool / MCP", "lane": "core", "sub": "Workbench"},
]

CHAT_SEQUENCE: dict[str, tuple[str, str, str]] = {
    "context": ("agent", "agent", "Bağlam bütçeye göre seçildi"),
    "compaction": ("agent", "agent", "Eski mesajlar özete indi"),
    "model": ("agent", "model", "istek + bağlam + tool şemaları"),
    "tool_request": ("model", "agent", "ToolCallRequestEvent"),
    "gate": ("agent", "gate", "izin ister"),
    "tool_exec": ("gate", "tool", "call_tool()"),
    "tool_result": ("tool", "agent", "ToolCallExecutionEvent"),
    "code_request": ("agent", "gate", "kod · onay bekliyor"),
    "code_result": ("tool", "agent", "konteyner çıktısı"),
    "loop": ("agent", "agent", "döngü sürüyor · max_tool_iterations"),
    "stream": ("model", "agent", "token akışı"),
    "done": ("agent", "user", "TaskResult · nihai cevap"),
}

SCAN_LANES = [
    {"id": "user", "name": "Tarama", "lane": ""},
    {"id": "flow", "name": "GraphFlow", "lane": "agentchat", "sub": "DiGraph"},
    {"id": "branches", "name": "Analistler", "lane": "agentchat", "sub": "3 dal"},
    {"id": "risk", "name": "Risk", "lane": "agentchat", "sub": "join(all)"},
    {"id": "scorer", "name": "Skorlayıcı", "lane": "ext", "sub": "Structured"},
]

SCAN_SEQUENCE: dict[str, tuple[str, str, str]] = {
    "graph_build": ("user", "flow", "graf kuruldu · DiGraphBuilder"),
    "graph_run": ("flow", "branches", "tek yayın → üç dal"),
    "analysts": ("branches", "branches", "üçü paralel koşuyor"),
    "join": ("branches", "risk", 'activation_condition="all"'),
    "count": ("risk", "scorer", "sıralı devir"),
    "intervention": ("flow", "flow", "her mesaj müdahale kapısından"),
    "runtime_stop": ("scorer", "user", "StructuredMessage[Score]"),
}


def _seq_label(stage_id: str, base: str, meta: dict[str, Any]) -> str:
    """Adımın etiketi, o adımın kendi verisiyle.

    Sabit metin, tekrar eden adımlarda hiçbir şey anlatmıyor: üç analist dalı
    üst üste "üçü paralel koşuyor" yazıyordu ve diyagram, üç ayrı olayın aynı
    olay olduğunu ima ediyordu. Hangi dalın kaçıncı sırada geldiği ise tam da
    eşzamanlılığın görünür olduğu yer.
    """
    meta = meta or {}
    try:
        if stage_id == "analysts" and meta.get("branch"):
            return (f"{meta['branch']} · {meta.get('arrived', '?')}"
                    f"/{meta.get('expected', '?')}")
        if stage_id == "graph_build" and meta.get("branches"):
            return f"{len(meta['branches'])} dal · {meta.get('termination', '')}".strip()
        if stage_id in ("gate", "tool_exec") and meta.get("tool"):
            return f"{base} · {meta['tool']}"
        if stage_id == "tool_request" and meta.get("tools"):
            return f"{base} · {', '.join(map(str, meta['tools']))}"
        if stage_id == "context" and meta.get("tokens") is not None:
            return f"{base} · {meta['tokens']}/{meta.get('budget', '?')} token"
    except Exception:  # noqa: BLE001 — etiket bir turu düşüremez
        pass
    return base


# ------------------------------------------------------------------ the record
@dataclass
class Run:
    """Tek bir tur: sorusu, olayları, ve olaylardan türeyen her şey.

    Olaylar ham tutuluyor. Türetme (graf, desen, bileşen) okuma anında yapılıyor
    çünkü türetme kuralı kayıttan daha sık değişiyor — kural düzeltildiğinde eski
    turlar da doğru çiziliyor, yeniden koşturmaya gerek kalmıyor.
    """

    id: str
    kind: str                      # "chat" | "scan"
    question: str
    session: str = "local"
    started: float = field(default_factory=time.time)
    finished: float | None = None
    status: str = "running"        # running | done | cancelled | error
    # Takım koşusunda hangi tip koştu. Sohbet ve taramada boş.
    variant: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    spans: list[dict[str, Any]] = field(default_factory=list)

    # -------------------------------------------------------------- recording
    def event(self, raw: dict[str, Any]) -> None:
        """Bir SSE olayını kaydet. Asla fırlatmaz — kayıt turu düşüremez."""
        try:
            kind = str(raw.get("type", ""))
            if kind not in ("stage", "tool", "tool_result", "done",
                            "cancelled", "error"):
                return                      # chunk'lar: binlerce, hiçbiri çizilmiyor
            item = dict(raw)
            # `t` is when the stage happened; without it, when we heard about it.
            # The difference is not cosmetic: the gate runs underneath an awaited
            # tool call and its event waits for the next drain.
            item["at"] = round(float(raw.get("t") or time.time()) - self.started, 3)
            self.events.append(item)
            if len(self.events) > 600:       # bir tur bunu aşıyorsa zaten bozuk
                del self.events[:200]
        except Exception:  # noqa: BLE001
            pass

    def end(self, status: str = "done") -> None:
        self.finished = time.time()
        self.status = status
        _trace(self)

    # ------------------------------------------------------------------ trace
    def trace_record(self) -> dict[str, Any]:
        """Diske yazılan satır: turun şekli, içeriği değil.

        Cevap metni burada yok. Soru var, çünkü oturum transkripti onu zaten
        yazıyor — yeni bir maruziyet açmıyor, ve izi soruyla eşleştirememek onu
        işe yaramaz hâle getirirdi.
        """
        totals = self.totals()
        return {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self.started)),
            "id": self.id,
            "kind": self.kind,
            "session": self.session,
            "question": self.question,
            "status": self.status,
            "seconds": self.seconds,
            "team": next((t["name"] for t in self.teams() if t["used"]), None),
            "patterns": [p["id"] for p in self.patterns() if p["used"]],
            "messages": {m["name"]: m["count"] for m in self.messages()},
            "tools": sorted({str(e.get("name", "")) for e in self._tool_calls()}),
            "stages": [s["id"] for s in self.timeline()],
            "code_runs": len(self.code_runs()),
            **{k: totals[k] for k in ("llm_calls", "tokens", "tools_requested",
                                      "tools_ran", "tools_blocked", "stop_reason")},
        }

    @property
    def seconds(self) -> float:
        return round((self.finished or time.time()) - self.started, 2)

    # ------------------------------------------------------------ derivations
    def _stages(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == "stage"]

    def _stage_ids(self) -> set[str]:
        return {str(e.get("id", "")) for e in self._stages()}

    def _tool_calls(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == "tool"]

    def totals(self) -> dict[str, Any]:
        """Turun faturası — tool sayıları **kaydın kendisinden**.

        İki ölçülmüş tuzak, ikisi de aynı sayaçta:

        1. `done` aşamasının `tool_calls` sayacı yalnız **koşan** çağrıları
           sayıyor. Kapı bir çağrıyı tuttuğunda sıfır kalıyor, ve sıfıra bakıp
           "model tool çağırmıyor" demek yanlış bir teşhis oluyor. Bir kez
           koydum; sekiz koşu boyunca yanlış yeri aradım.
        2. Daha kötüsü: o sayaç `BaseTool`'dan doğan `ToolCallEvent`'leri
           sayıyor, ve **MCP çağrıları o olayı yaymıyor**. Ölçüldü (chat-0010):
           `ask_question` kapıdan geçti, `tool_exec` yayınlandı, 11,6 saniye
           sonra sonuç döndü — ve sayaç 0 dedi. Ekran kendi sıra diyagramıyla
           çelişiyordu.

        Onun için koşan sayısı `tool_exec` aşamalarından sayılıyor: o aşama,
        kapının geçirdiği ve workbench'e girilen her çağrıda yayılıyor —
        çağrının hangi workbench'ten gittiğinden bağımsız.

        3. Ve aynı hata üçüncü kez, takım tarafında. Takım koşusu `tool_exec`
           yaymıyor — kendi `team_tool` aşamasını yayıyor. Ölçüldü (team-0001):
           graf `search_docs · Critic çağırdı, 4 kez` yazarken üstteki sayaç
           `TOOL · KOŞTU 0` diyordu. Ekranın kendi diyagramıyla çelişmesi, aynı
           sayacın üçüncü kez aynı şekilde kırılması demek — sayaç artık
           **çağrının hangi yoldan geldiğini değil, çağrı olup olmadığını**
           soruyor.
        """
        done = next((e for e in self.events if e.get("type") == "done"), None)
        # `done` sohbetin, `team_done` takımın bitiş aşaması. Yalnız birincisine
        # bakmak, takım koşusunda maliyeti sıfır gösteriyordu — ölçüldü
        # (team-0009): 16 saniye, üç konuşmacı, ve ekranda "0 LLM çağrısı".
        stage_done = next((e for e in self._stages()
                           if e.get("id") in ("done", "team_done")), None)
        meta = (stage_done or {}).get("meta", {}) if stage_done else {}
        stages_seen = self._stages()
        team_tools = sum(1 for e in stages_seen if e.get("id") == "team_tool")
        requested = len(self._tool_calls()) + team_tools
        blocked = sum(1 for e in stages_seen
                      if e.get("id") == "gate" and (e.get("meta") or {}).get("blocked"))
        ran = sum(1 for e in stages_seen
                  if e.get("id") in ("tool_exec", "team_tool"))
        return {
            "llm_calls": meta.get("llm_calls", 0),
            "tokens": meta.get("tokens", 0),
            "tools_requested": requested,
            "tools_ran": ran,
            "tools_blocked": blocked,
            "stop_reason": (done or {}).get("stop_reason", ""),
            "seconds": self.seconds,
        }

    def graph(self) -> dict[str, Any]:
        """Düğümler ve kenarlar — kenarların üstünde gerçek mesaj tipleri.

        Bu grafın sekiz köşeli kutu diyagramlarından farkı şu: kenar bir okun
        çizilmesi değil, **taşınan mesajın tipi**. AutoGen'de bunu bilmek
        gerekiyor çünkü GraphFlow'un kenarları veri taşımıyor — yalnız sırayı
        belirliyor — ve iki şeyi aynı okla çizmek tam da bu farkı siliyor.
        """
        if self.kind == "team":
            return _team_graph(self)
        if self.kind == "maf":
            return _maf_graph(self)
        if self.kind == "cron":
            return _cron_graph(self)
        return _scan_graph(self) if self.kind == "scan" else _chat_graph(self)

    def teams(self) -> list[dict[str, Any]]:
        if self.kind == "maf":
            # Beş takım AutoGen'in tipleri. MAF'ta hiçbiri koşmadı ve
            # koşmayacak — bu ekranın uydurmaması gereken yerlerden biri.
            return [{**t, "used": False,
                     "why": "AutoGen'in takım tipi; bu tur MAF'ta koştu."}
                    for t in TEAMS]
        used = self.variant if self.kind == "team" else (
            "graphflow" if self.kind == "scan" else None)
        out = []
        for t in TEAMS:
            row = dict(t)
            row["used"] = t["id"] == used
            row["why"] = (
                "Bu turda koşan takım bu." if row["used"]
                else "Bu turda kullanılmadı." if used
                else "Bu turda hiçbir takım kurulmadı: `agent.run_stream()` "
                     "doğrudan çağrıldı."
            )
            out.append(row)
        return out

    def patterns(self) -> list[dict[str, Any]]:
        """Sekiz desenden hangisi gerçekten koştu — kanıtıyla.

        Kanıt kaydın kendisinden çıkıyor: bir deseni 'koştu' diye işaretlemek
        için o turda görülmüş bir aşama gerekiyor. Böylece ekran, sistemin
        yapılandırmasını değil o soruda olan biteni anlatıyor.
        """
        ids = self._stage_ids()
        ran_code = "code_result" in ids or "code_request" in ids
        scan = self.kind == "scan"
        if self.kind == "team":
            return self._team_patterns()
        if self.kind == "maf":
            return [{**p, "used": False,
                     "why": "Kılavuzun deseni AutoGen için; MAF turunda "
                            "orkestrasyon tek ajanda."} for p in PATTERNS]
        evidence = {
            "concurrent": (scan, "Üç analist dalı tek graf koşusunda paralel çalıştı."
                                 if scan else "Bu turda tek ajan vardı, dallanma yok."),
            "sequential": (scan, "Dallar → risk denetçisi → skorlayıcı zinciri koştu."
                                 if scan else "Devredilecek ikinci bir ajan yok."),
            "groupchat": (False, "Sırayı dağıtan bir yönetici ajan kurulmadı."),
            "handoffs": (False, "`Handoff` tool'u tanımlı değil; devir hiç doğmuyor."),
            "mixture": (False, "Aynı soruyu birden çok uzmana sorup birleştiren "
                               "katman yok."),
            "debate": (False, "Karşılıklı eleştiri turu yok; tek geçiş."),
            "reflection": (False, "Üretici–eleştirmen döngüsü bu hatta kurulu değil; "
                                  "kalite denetimi şemayla yapılıyor."),
            "codeexec": (ran_code, "Model kod yazdı, onaydan geçti ve konteynerde koştu."
                                   if ran_code else
                                   "Bu turda kod yürütülmedi (varsayılan kapalı)."),
        }
        out = []
        for p in PATTERNS:
            used, why = evidence[p["id"]]
            row = dict(p)
            row["used"] = bool(used)
            row["why"] = why
            out.append(row)
        return out

    def _team_patterns(self) -> list[dict[str, Any]]:
        """Koşan takımın karşılık geldiği resmî desen.

        Eşleşme mekanik değil, iddia: RoundRobin/Selector/MagenticOne bir
        yöneticinin sırayı dağıttığı Group Chat; Swarm'ın devri Handoffs;
        bizim GraphFlow kurulumumuz bir dala açılıyor, yani Concurrent. Hangi
        takımın hangi desene karşılık geldiğini yazmak, ikisini aynı şey sanmaya
        yol açabilir — o yüzden gerekçe her satırda duruyor.
        """
        mapping = {
            "roundrobin": ("groupchat", "Sırayı sabit bir döngü dağıtıyor; "
                                        "yönetici mekanizması Group Chat'inki."),
            "selector": ("groupchat", "Sırayı bir model çağrısı dağıtıyor — "
                                      "Group Chat'in yöneticisi burada model."),
            "magenticone": ("groupchat", "Yönetici bir görev defteri tutuyor; "
                                         "yine sırayı dağıtan bir yönetici var."),
            "swarm": ("handoffs", "Devir bir tool çağrısı olarak gerçekleşti."),
            "graphflow": ("concurrent", "Planner'dan iki dal birden açıldı."),
        }
        target, why = mapping.get(self.variant, ("", ""))
        out = []
        for p in PATTERNS:
            row = dict(p)
            row["used"] = p["id"] == target
            row["why"] = why if row["used"] else "Bu koşuda kullanılmadı."
            out.append(row)
        return out

    def messages(self) -> list[dict[str, Any]]:
        """Bu turda gerçekten uçan mesaj tipleri, sayılarıyla."""
        counts = _message_counts(self)
        out = []
        for name, n in counts.items():
            meta = MESSAGES.get(name, {"lane": "agentchat", "ref": "", "what": ""})
            out.append({"name": name, "count": n, **meta})
        return out

    def components(self) -> list[dict[str, Any]]:
        """Hangi bileşen bu turda ne yaptı.

        "Kurulu" ile "kullanıldı" ayrı sütunlar. Bir bileşenin ayakta olması
        onun o soruda iş yaptığı anlamına gelmiyor, ve ikisini aynı listede
        göstermek bu ekranın en çok işe yarayan yeri: core runtime *ayakta*,
        ama sohbet turu ona hiç uğramıyor.
        """
        ids = self._stage_ids()
        totals = self.totals()
        ctx = next((e.get("meta", {}) for e in self._stages()
                    if e.get("id") == "context"), {})
        tools = sorted({str(e.get("name", "")) for e in self._tool_calls()})
        mcp_used = any(t in ("read_wiki_structure", "read_wiki_contents",
                             "ask_question") for t in tools)

        rows = [
            {"name": "AssistantAgent", "lane": "agentchat", "ref": "08:0512",
             "used": True,
             "did": f"Turu koşturan tek ajan. max_tool_iterations=6 — varsayılan 1 "
                    f"olsaydı zincir ilk tool'dan sonra dururdu.",
             "what": "Model + tool + bellek üçlüsünü tek nesnede toplayan sarmalayıcı."},
            {"name": "CompactingChatCompletionContext", "lane": "ours",
             "ref": "05:2341", "used": "context" in ids,
             "did": (f"Modele giden mesaj kümesi bütçeye göre seçildi: "
                     f"{ctx.get('tokens', '?')} / {ctx.get('budget', '?')} token."
                     if ctx else "Bu turda bağlam kurulmadı."),
             "what": "AutoGen'in Buffered hâli mesaj sayar; bu bizimki token sayıyor."},
            {"name": "StaticWorkbench", "lane": "core", "ref": "05:3020",
             "used": True,
             "did": f"Yerel tool'ları tek kaynak olarak sundu. Bu turda çağrılanlar: "
                    f"{', '.join(tools) if tools else '—'}.",
             "what": "Workbench bir tool *kaynağı*: listeler ve çağırır. Ajana "
                     "`tools=` ile liste vermek yerine kaynak vermek, kaynağı "
                     "sarmalanabilir yapıyor — kapı tam da bunu kullanıyor."},
            {"name": "McpWorkbench", "lane": "ext", "ref": "05:3120",
             "used": mcp_used,
             "did": ("DeepWiki'nin tool'ları bu turda çağrıldı."
                     if mcp_used else
                     "Bağlı ama bu turda çağrılmadı."),
             "what": "Aynı Workbench arayüzünün MCP sunucusuna bakan hâli. Uzak "
                     "tool'lar yerel tool'larla aynı listede görünüyor, ve aynı "
                     "kapıdan geçiyor — federasyonun bedava gelen kısmı bu."},
            {"name": "GatedWorkbench", "lane": "ours", "ref": "—",
             "used": "gate" in ids,
             "did": (f"{totals['tools_requested']} çağrı istendi, "
                     f"{totals['tools_blocked']} tanesi tutuldu."),
             "what": "Her tool çağrısının geçtiği tek nokta. Reddi bir istisna "
                     "değil, hata işaretli bir sonuç olarak dönüyor — böylece "
                     "ajan gerekçeyi okuyup kullanıcıya söyleyebiliyor."},
            {"name": "OpenAIChatCompletionClient", "lane": "ext", "ref": "05:1837",
             "used": totals["llm_calls"] > 0,
             "did": f"{totals['llm_calls']} model çağrısı, {totals['tokens']} token.",
             "what": "OpenAI-uyumlu her endpoint buradan konuşuyor. Uyumlu ama "
                     "OpenAI olmayan bir endpoint'te `model_info` zorunlu."},
            {"name": "SingleThreadedAgentRuntime", "lane": "core", "ref": "05:0490",
             "used": self.kind == "scan" and "runtime_stop" in ids,
             "did": ("Gateway'de ayakta ve abonelikler kurulu, ama bu turda "
                     "yönlendirilen mesaj yok — sohbet core'a uğramıyor."
                     if self.kind != "scan" else
                     "Graf koşusu bittikten sonra süre sınırıyla kapatıldı."),
             "what": "Aktör runtime'ı: mesajı tipe göre yönlendiren, ajanları "
                     "gerektiğinde yaratan katman."},
            {"name": "PythonCodeExecutionTool + Docker", "lane": "ext",
             "ref": "05:6188", "used": "code_result" in ids,
             "did": ("Model kod yazdı, onaylandı, konteynerde koştu."
                     if "code_result" in ids else
                     "Bu turda kullanılmadı."),
             "what": "Tool'u olmayan bir iş için kaçış kapağı. Konteyner izole, "
                     "ama ağ erişimi var — yukarı akışta kapatacak parametre yok."},
        ]
        return rows

    def topics(self) -> dict[str, Any]:
        """Topic iletişimi: bu turda oldu mu, olduysa nasıl.

        Bu ekranın en kolay yanlış anlaşılacak yeri, o yüzden ayrı duruyor.
        GraphFlow'un kenarları **veri taşımıyor**: bütün katılımcılar tek bir
        paylaşılan `group_topic_type`'a abone, herkes her mesajı görüyor, ve
        kenarlar yalnız sıranın kimde olduğunu belirliyor. "Publisher/subscriber
        gibi mi" sorusunun cevabı: taşıma katmanı evet, ama kenarların anlamı
        hayır.
        """
        if self.kind == "scan":
            return {
                "active": True,
                "topic": "group_topic_type (GraphFlow'un tek paylaşılan konusu)",
                "ref": "08:5398",
                "note": "Beş katılımcının hepsi AYNI topic'e abone. Kenarlar veri "
                        "taşımıyor, sırayı belirliyor: mesaj zaten herkese gitti, "
                        "graf yalnız kimin konuşacağını söylüyor.",
            }
        return {
            "active": False,
            "topic": "—",
            "ref": "05:0670",
            "note": "Bu tur core'a hiç uğramadı: /api/chat doğrudan "
                    "conversation.stream'i çağırıyor. Runtime ayakta ve "
                    "abonelikler kurulu, ama yönlendirilen mesaj yok.",
        }

    def code_runs(self) -> list[dict[str, Any]]:
        """Konteynerde koşan programlar — terminalin kalıcı hâli."""
        out: list[dict[str, Any]] = []
        pending: dict[str, Any] | None = None
        for e in self._stages():
            meta = e.get("meta") or {}
            if e.get("id") == "code_request":
                pending = {"code": meta.get("code", ""), "at": e.get("at", 0)}
            elif e.get("id") == "code_result":
                out.append({**(pending or {"code": "", "at": e.get("at", 0)}),
                            "output": meta.get("output", ""),
                            "is_error": bool(meta.get("is_error")),
                            "seconds": meta.get("seconds")})
                pending = None
        if pending:
            out.append({**pending, "output": "", "is_error": False,
                        "seconds": None, "running": True})
        return out

    def active(self) -> dict[str, Any] | None:
        """Şu an grafın neresinde koşuluyor — canlı vurgu için.

        Tur bittiyse `None`. Bitmiş bir turda bir kutuyu yakıp bırakmak, ekranın
        anlattığı en temel şeyi bozardı: yanan yer *şu an* olan yer.
        """
        if self.status != "running":
            return None
        stages_seen = self._stages()
        if not stages_seen:
            return None
        last = stages_seen[-1]
        node, edge = self._target(self.graph(), str(last.get("id", "")),
                                  last.get("meta") or {})
        return {
            "stage": last.get("id", ""),
            # `Mechanism` alanının adı `title`; `name` diye okumak sessizce boş
            # string veriyordu ve aşama adları ekranda hiç görünmüyordu.
            "name": last.get("title", ""),
            "lane": last.get("lane", ""),
            "klass": last.get("klass", ""),
            "since": last.get("at", 0),
            "node": node,
            "edge": edge,
        }

    @classmethod
    def _target(cls, graph: dict[str, Any], stage_id: str,
                meta: dict[str, Any] | None = None) -> tuple[Any, int | None]:
        """Bir aşamanın grafta yandığı yer: (düğüm, kenar). İkisi de olmayabilir."""
        where = STAGE_TARGET.get(stage_id)
        if where is None:
            return None, None
        what, target = where
        if what == "node":
            return cls._resolve_node(graph, target, meta or {}), None
        return None, cls._resolve_edge(graph)

    @staticmethod
    def _resolve_node(graph: dict[str, Any], target: str,
                      meta: dict[str, Any] | None = None) -> str | list[str] | None:
        ids = [n["id"] for n in graph["nodes"]]
        # Takım aşamalarının hedefi sabit değil: hangi ajanın sırası geldiği ve
        # hangi tool'un çağrıldığı aşamanın kendi meta'sında yazıyor. Sabit bir
        # hedef koymak, üç ajanlı bir koşuda ışığı hep aynı kutuda yakardı.
        if target.startswith("*"):
            meta = meta or {}
            who = str(meta.get("who", ""))
            if target == "*speaker":
                return who if who in ids else None
            tool = str(meta.get("tool", ""))
            tid = f"tool:{who}:{tool}"
            if tid in ids:
                return [tid, who] if who in ids else [tid]
            return who if who in ids else None
        if target == "tool":
            # İki kutu birden: üstte koşan tool, altta onu çağıran workbench.
            # Tek başına biri yanınca "tool nerede koşuyor" sorusunun yarısı
            # eksik kalıyordu — çağrı ajanın değil, hattın üstünde yürüyor.
            tools = [n["id"] for n in graph["nodes"] if n["kind"] == "tool"]
            pair = ([tools[-1]] if tools else []) + (["wb"] if "wb" in ids else [])
            return pair or None
        if target == "branches":
            branches = [i for i in ("technical", "market", "team") if i in ids]
            return branches or None
        return target if target in ids else None

    @staticmethod
    def _resolve_edge(graph: dict[str, Any]) -> int | None:
        """En son istek kenarı. Kapı da burada oturuyor, ve tam da bu yüzden
        `tool_request` ile `gate` aynı kenarı yakıyor: ikisi de o geçişte."""
        for index in range(len(graph["edges"]) - 1, -1, -1):
            if graph["edges"][index].get("message") == "ToolCallRequestEvent":
                return index
        return None

    def details(self) -> dict[str, Any]:
        """Bu turda geçen aşamaların uzun anlatımları.

        Yalnız geçenler: katalogun tamamını göndermek, bu turda hiç olmamış
        yirmi mekanizmanın metnini her istekte taşımak olurdu.
        """
        return {sid: d for sid in self._stage_ids()
                if (d := stages_module.detail(sid)) is not None}

    def sequence(self) -> dict[str, Any]:
        """Aynı tur, sıra diyagramı olarak: kim kime, hangi sırayla.

        Graf **yapıyı** gösteriyor — hangi kutu neye bağlı. Bu ise **zamanı**:
        aynı ajan üç kez konuşuyorsa grafta tek kutu, burada üç ok. İkisi aynı
        şeyin iki farklı sorusu, ve bir sohbet turunda asıl merak edilen ikincisi.
        """
        if self.kind == "team":
            return _team_sequence(self)
        if self.kind == "maf":
            return _maf_sequence(self)
        lanes = (SCAN_LANES if self.kind == "scan" else CHAT_LANES)
        table = (SCAN_SEQUENCE if self.kind == "scan" else CHAT_SEQUENCE)
        steps: list[dict[str, Any]] = []

        if self.kind != "scan":
            steps.append({"src": "user", "dst": "agent", "at": 0.0,
                          "label": "Soru · TextMessage", "kind": "call"})

        blocked_at: list[int] = []
        for row in self.timeline():
            move = table.get(row["id"])
            if move is None:
                continue
            src, dst, label = move
            steps.append({"src": src, "dst": dst, "at": row["at"],
                          "label": _seq_label(row["id"], label, row["meta"]),
                          "stage": row["id"],
                          "kind": "self" if src == dst else "call"})
            if row["id"] == "gate" and (row.get("meta") or {}).get("blocked"):
                blocked_at.append(len(steps))
                steps.append({"src": "gate", "dst": "agent", "at": row["at"],
                              "label": "RED · gerekçeyle döndü", "stage": "gate",
                              "kind": "return", "blocked": True})

        # Kutular: hangi adımlar bir döngünün, hangileri bir koşulun içinde.
        groups: list[dict[str, Any]] = []
        first_model = next((i for i, s in enumerate(steps)
                            if s.get("stage") == "model"), None)
        if first_model is not None:
            groups.append({"label": "Agentic Loop — görev bitene kadar tekrar",
                           "kind": "loop", "from": first_model,
                           "to": len(steps) - 1})
        first_tool = next((i for i, s in enumerate(steps)
                           if s.get("stage") == "tool_request"), None)
        last_tool = next((i for i in range(len(steps) - 1, -1, -1)
                          if steps[i].get("stage") in ("tool_result", "code_result")),
                         None)
        if first_tool is not None and last_tool is not None and last_tool >= first_tool:
            groups.append({"label": "alt · [Tool/MCP çağrısı gerekiyor]",
                           "kind": "alt", "from": first_tool, "to": last_tool})

        return {"lanes": lanes, "steps": steps, "groups": groups,
                "blocked": bool(blocked_at)}

    def design(self) -> dict[str, Any]:
        """O an hangi ajan tasarımı koşuyor — tek satırda, ekranın en üstünde.

        Metin `stages.RUNS`'tan geliyor, buradan değil: aynı cümle sohbet
        tarafında da kullanılıyordu ve iki yerde iki kopya tutmak, birini
        düzeltip ötekini unutmanın en kısa yolu.
        """
        if self.kind == "maf":
            return {
                "team": "MAF · agent_framework",
                "team_note": "AutoGen'in resmî halefi · ayrı sanal ortamda koşuyor",
                "pattern": "tek ajan · tool döngüsü",
                "pattern_note": "Onay ve çağrı tavanı tool'un kendi alanları; "
                                "kapıyı yazmak gerekmiyor.",
                "declared": "ToolApprovalMiddleware",
            }
        if self.kind == "team":
            names = {"roundrobin": "RoundRobinGroupChat", "selector": "SelectorGroupChat",
                     "swarm": "Swarm", "magenticone": "MagenticOneGroupChat",
                     "graphflow": "GraphFlow"}
            pickers = {"roundrobin": "sırayla", "selector": "model seçer",
                       "swarm": "handoff", "magenticone": "planlayıcı",
                       "graphflow": "DAG"}
            used = next((p["name"] for p in self._team_patterns() if p["used"]), "—")
            return {
                "team": names.get(self.variant, self.variant),
                "team_note": "Sırayı belirleyen: " + pickers.get(self.variant, "?") +
                             " · kadro Planner · Researcher · Critic",
                "pattern": used,
                "pattern_note": next((p["why"] for p in self._team_patterns()
                                      if p["used"]), ""),
                "declared": "MaxMessageTermination",
            }
        run = stages_module.RUNS.get(self.kind, {})
        return {
            "team": run.get("team", "—"),
            "team_note": run.get("team_note", ""),
            "pattern": run.get("pattern", "—"),
            "pattern_note": run.get("pattern_note", ""),
            "declared": run.get("declared", ""),
        }

    def timeline(self) -> list[dict[str, Any]]:
        """Aşamalar, sırayla — hangi katman, ne zaman, ne oldu, grafta neresi.

        Her satır kendi hedefini taşıyor, yalnız sonuncusu değil. Sebebi ölçüldü:
        SSE olayları birlikte boşalıyor ve hızlı bir turda sekiz aşama tek anda
        geliyor. Yalnız "şu anki"ni yakan bir ekran, kapıyı ve tool'u hiç
        göstermiyor — ilginç olan kısmı tam olarak o. Ekran sırayı bu hedeflerle
        yürütüyor; sıra da zamanlar da kaydın kendisinden, uydurma değil.
        """
        graph = self.graph()
        rows = []
        for e in self._stages():
            node, edge = self._target(graph, str(e.get("id", "")),
                                      e.get("meta") or {})
            rows.append(
                {"at": e.get("at", 0), "id": e.get("id", ""), "lane": e.get("lane", ""),
                 "name": e.get("title", ""), "klass": e.get("klass", ""),
                 "ref": e.get("ref", ""), "module": e.get("module", ""),
                 "note": e.get("note", ""), "meta": e.get("meta") or {},
                 "node": node, "edge": edge})
        return rows

    # ------------------------------------------------------------------ output
    def summary(self) -> dict[str, Any]:
        """Liste görünümü için — bütün olayları taşımadan."""
        return {"id": self.id, "kind": self.kind, "question": self.question,
                "status": self.status, "seconds": self.seconds,
                "started": self.started, "steps": len(self._stages()),
                "session": self.session}

    def report(self) -> dict[str, Any]:
        """Ekranın çizdiği her şey. Tek istek, tek cevap."""
        return {
            **self.summary(),
            "totals": self.totals(),
            "design": self.design(),
            "sequence": self.sequence(),
            "details": self.details(),
            "active": self.active(),
            "graph": self.graph(),
            "teams": self.teams(),
            "patterns": self.patterns(),
            "messages": self.messages(),
            "components": self.components(),
            "topics": self.topics(),
            "code": self.code_runs(),
            "spans": self.spans,
            "overhead": _overhead(self),
            "timeline": self.timeline(),
        }


# ------------------------------------------------------------------ tepegöz
#
# Runtime'ın **kendi** işleri, kılavuzun kendi cümlesinden (`05:920`):
#
#   *"an agent runtime provides the necessary infrastructure to facilitate
#   communication between agents, manage agent lifecycles, enforce security
#   boundaries, and support monitoring and debugging."*
#
# Dördü de her turda çalışıyor ama grafta hiçbiri görünmüyordu: graf mesajın
# yolunu çiziyor, runtime'ın o yolu taşırken ne yaptığını değil. Tepegöz o
# katman — turun üstünde duran, hangi işin şu an döndüğünü gösteren şerit.
#
# Hücreler kılavuzun sırasıyla; hangi aşamanın hangi hücreyi yaktığı aşağıda.
# Bir aşama birden çok hücre yakabiliyor: bir tool çağrısı hem iletişim hem
# güvenlik sınırı, ve ikisini ayırmak yanlış olurdu.
OVERHEAD_CELLS = [
    {"id": "comm", "name": "İletişim", "lane": "core",
     "sub": "send · publish · topic",
     "note": "Mesajı doğru ajana taşımak. Yayında dönüş yok, doğrudanda var."},
    {"id": "life", "name": "Yaşam döngüsü", "lane": "core",
     "sub": "AgentId(type, key)",
     "note": "Ajanı gerektiğinde yaratmak. Topic kaynağı ajan anahtarına dönüşüyor."},
    {"id": "guard", "name": "Güvenlik sınırı", "lane": "ours",
     "sub": "InterventionHandler · GatedWorkbench",
     "note": "Çağrıyı geçirmeden önce durdurabilen tek yer. Bizim kapımız burada."},
    {"id": "watch", "name": "İzleme", "lane": "ext",
     "sub": "OTel · gen_ai.*",
     "note": "Her iş bir span. Şelale ve bu şerit aynı kaynaktan besleniyor."},
]

# Hangi aşama tepegözde neyi yakıyor. Boş bırakılan aşama hiçbir hücreyi
# yakmıyor — "her aşama bir şey yakmalı" diye zorlamak, şeridi anlamsız
# yanıp sönen bir süse çevirirdi.
OVERHEAD_OF: dict[str, tuple[str, ...]] = {
    "runtime_start": ("life", "comm"),
    "subscribe": ("life", "comm"),
    "publish": ("comm",),
    "speaker": ("comm",),
    "handoff": ("comm",),
    "model": ("watch",),
    "stream": ("watch",),
    "tool_request": ("comm", "guard"),
    "gate": ("guard",),
    "tool_exec": ("comm", "guard", "watch"),
    "team_tool": ("comm", "guard", "watch"),
    "tool_result": ("comm",),
    "code_request": ("guard",),
    "code_result": ("guard", "watch"),
    "intervention": ("guard",),
    "context": ("life",),
    "compaction": ("life",),
    "graph_build": ("life",),
    "graph_run": ("comm", "life"),
    "analysts": ("comm",),
    "join": ("comm",),
    "runtime_stop": ("watch", "life"),
    "done": ("watch",),
    "team_done": ("watch",),
    "maf_gate": ("guard",),
    "maf_run": ("comm", "watch"),
    "maf_approval": ("guard",),
    # Zamanlama. `comm` ve `life` bilerek boş: bu turda AutoGen runtime'ı hiç
    # çalışmıyor, ve sıfır göstermek doğru olanı söylüyor. Kapı ile kayıt ise
    # bizim hattımızda gerçekten dönüyor.
    "cron_gate": ("guard",),
    "cron_done": ("watch",),
}


def _overhead(run: Run) -> dict[str, Any]:
    """Tepegöz: runtime'ın dört işi, hangisi kaç kez ve şu an hangisi.

    Sayı önemli çünkü "güvenlik sınırı" hücresinin sıfır olduğu bir tur,
    kapının o turda hiç devreye girmediğini söylüyor — ve bunu grafta aramak
    kutu kutu dolaşmak demek.
    """
    counts: dict[str, int] = {c["id"]: 0 for c in OVERHEAD_CELLS}
    for e in run._stages():
        for cell in OVERHEAD_OF.get(str(e.get("id", "")), ()):
            counts[cell] = counts.get(cell, 0) + 1

    live: list[str] = []
    if run.status == "running":
        seen = run._stages()
        if seen:
            live = list(OVERHEAD_OF.get(str(seen[-1].get("id", "")), ()))

    return {
        "cells": [{**c, "hits": counts.get(c["id"], 0),
                   "live": c["id"] in live} for c in OVERHEAD_CELLS],
        "spans": len(run.spans),
        "running": run.status == "running",
        "ref": "05:920",
        "quote": ("an agent runtime provides the necessary infrastructure to "
                  "facilitate communication between agents, manage agent "
                  "lifecycles, enforce security boundaries, and support "
                  "monitoring and debugging"),
    }


# ------------------------------------------------------------- graf türetme
def _message_counts(run: Run) -> dict[str, int]:
    """Hangi mesaj tipinden kaç tane uçtu.

    Sayımlar olaylardan çıkıyor, `RUNS` tablosundan değil: tablo o tür bir turda
    hangi tiplerin *olabileceğini* söylüyor, burada olanı sayıyoruz.
    """
    ids = run._stage_ids()
    tools = run._tool_calls()
    counts: dict[str, int] = {"TextMessage": 1}                     # soru
    if "stream" in ids:
        counts["ModelClientStreamingChunkEvent"] = 1
    if tools:
        counts["ToolCallRequestEvent"] = len(tools)
        counts["ToolCallExecutionEvent"] = len(
            [e for e in run.events if e.get("type") == "tool_result"])
    if run.kind == "scan":
        counts["StructuredMessage[Score]"] = 1
    if any(e.get("type") == "done" for e in run.events):
        counts["TextMessage"] = counts.get("TextMessage", 0) + 1   # cevap
        counts["TaskResult"] = 1
    return counts


def _chat_graph(run: Run) -> dict[str, Any]:
    """İki bant: üstte ajan mekanizması, altta o soru boyunca gateway hattı.

    Tek sıra hâlinde çizildiğinde iki ayrı şey aynı zincire diziliyordu ve
    ekrandaki en önemli ayrım kayboluyordu: **üst bant AutoGen, alt bant biz.**
    Kapı, bağlam bütçesi ve iz AutoGen'in özellikleri değil; onları
    `AssistantAgent` ile aynı sırada göstermek, destede yaptığımız ayrımı
    ekranda çürütürdü.

    Bantlar arası tek kesikli ok var ve taşıdığı cümle şu: her tool çağrısı
    yukarıdan aşağı iner, kapıdan geçer, sonra çalışır.
    """
    nodes: list[dict[str, Any]] = [
        {"id": "user", "band": 0, "kind": "user", "name": "Soru",
         "sub": "TextMessage",
         "note": "Tur burada başlıyor. Metin, ajanın bağlamına bir mesaj olarak giriyor."},
        {"id": "agent", "band": 0, "kind": "agent", "name": "Analyst",
         "sub": "AssistantAgent", "lane": "agentchat",
         "note": "Model, tool listesi ve bellek tek nesnede. Tool çağırıp çağırmayacağına burada karar veriliyor."},
    ]
    edges: list[dict[str, Any]] = [
        {"src": "user", "dst": "agent", "message": "TextMessage",
         "at": 0.0, "note": "Kullanıcının sorusu."},
    ]

    gates = [e for e in run._stages() if e.get("id") == "gate"]
    results = [e for e in run.events if e.get("type") == "tool_result"]
    calls = run._tool_calls()
    for i, call in enumerate(calls):
        name = str(call.get("name", "tool"))
        nid = f"tool:{name}:{i}"
        nodes.append({"id": nid, "band": 0, "kind": "tool", "name": name,
                      "sub": "workbench.call_tool", "lane": "core",
                      "note": "Önceden yazılmış bir fonksiyon. Modelin seçtiği "
                              "argümanlarla, workbench üzerinden çağrıldı."})
        gate = gates[i] if i < len(gates) else None
        blocked = bool((gate or {}).get("meta", {}).get("blocked"))
        edges.append({
            "src": "agent", "dst": nid, "message": "ToolCallRequestEvent",
            "at": call.get("at", 0), "gate": "red" if blocked else "izin",
            "note": "Model çağırmaya karar verdi; kapı " +
                    ("reddetti." if blocked else "geçirdi."),
        })
        result = results[i] if i < len(results) else None
        if result is not None:
            edges.append({
                # Dönüş kenarı. Katman hesabında ileri kenarlarla aynı sayılırsa
                # ajan kendi tool'unun sağına düşer ve graf tersine döner.
                "src": nid, "dst": "agent", "message": "ToolCallExecutionEvent",
                "back": True, "at": result.get("at", 0),
                "note": "Sonuç ajana döndü" +
                        (" — red de bir sonuçtur." if blocked else "."),
            })

    if any(e.get("type") == "done" for e in run.events):
        nodes.append({"id": "answer", "band": 0, "kind": "user", "name": "Cevap",
                      "sub": "TaskResult", "terminal": True,
                      "note": "İki şey birlikte döndü: bütün konuşma, ve turun "
                              "neden durduğu."})
        edges.append({"src": "agent", "dst": "answer", "message": "TaskResult",
                      "at": run.seconds,
                      "note": "Bütün konuşma ve durma sebebi birlikte döndü."})

    # ---- alt bant: gateway ------------------------------------------------
    #
    # Sırası zamana göre: bağlam kuruluyor, çağrı kapıdan geçiyor, workbench
    # tool'u çağırıyor, ve tur ize yazılıyor. Hepsi bizim kodumuz — `ours`
    # şeridi bilerek ayrı renkte.
    ids = run._stage_ids()
    tools_named = sorted({str(c.get("name", "")) for c in calls})
    mcp = [t for t in tools_named
           if t in ("read_wiki_structure", "read_wiki_contents", "ask_question")]
    blocked_count = sum(1 for g in gates if (g.get("meta") or {}).get("blocked"))

    nodes += [
        {"id": "ctx", "band": 1, "kind": "component", "name": "Bağlam bütçesi",
         "sub": "CompactingChatCompletionContext", "lane": "ours",
         "note": "Modele ne gideceğini token bütçesine göre seçiyor. AutoGen'in "
                 "kendi hâli mesaj sayar; bu token sayıyor."},
        {"id": "gate", "band": 1, "kind": "gate", "name": "Kapı",
         "sub": "GatedWorkbench", "lane": "ours",
         "note": "Her tool çağrısının geçtiği tek nokta. Red bir istisna değil, "
                 "hata işaretli bir sonuç — ajan gerekçeyi okuyabiliyor."},
        {"id": "wb", "band": 1, "kind": "component",
         "name": "Workbench" + (" + MCP" if mcp else ""),
         "sub": "StaticWorkbench" + (" · McpWorkbench" if mcp else ""),
         "lane": "ext" if mcp else "core",
         "note": "Tool *kaynağı*: listeler ve çağırır. Kaynak olduğu için "
                 "sarmalanabiliyor, kapı da tam bunu kullanıyor."},
    ]
    edges += [
        {"src": "ctx", "dst": "gate", "message": "her tur", "at": 0.0,
         "note": "Modele ne gideceği seçildikten sonra, çağrılar kapıya geliyor."},
        {"src": "gate", "dst": "wb", "message":
            ("izin" if not blocked_count else f"{blocked_count} red"),
         "at": 0.0,
         "note": "Kapıdan geçen çağrı workbench'e iniyor; geçmeyen hiç inmiyor."},
    ]

    if "code_result" in ids or "code_request" in ids:
        nodes.append({"id": "exec", "band": 1, "kind": "exec",
                      "name": "Docker yürütücü",
                      "sub": "PythonCodeExecutionTool", "lane": "ext",
                      "note": "Tool'u olmayan iş için kaçış kapağı. Konteyner "
                              "izole, ama ağ erişimi var."})
        edges.append({"src": "wb", "dst": "exec", "message": "kod", "at": 0.0,
                      "note": "Onaylanan program izole konteynerde koştu."})

    nodes.append({"id": "trace", "band": 1, "kind": "component", "name": "İz",
                  "sub": "runs.jsonl", "lane": "ours", "terminal": True,
                  "note": "Turun şekli diske yazıldı: takım, desen, tool, token. "
                          "Cevap metni yazılmıyor."})
    edges.append({"src": "exec" if "code_result" in ids else "wb",
                  "dst": "trace", "message": "kayıt", "at": run.seconds,
                  "note": "Turun şekli diske yazıldı: takım, desen, tool, token."})

    # Tek bantlar arası ok. Kapı bir ajan değil, ajanın altından geçen bir hat.
    edges.append({"src": "agent", "dst": "gate", "message": "her tool çağrısı",
                  "cross": True, "at": 0.0,
                  "note": "Ajanın uyum göstermeyi seçmesine değil, hattın "
                          "kendisine dayanıyor."})

    return {"nodes": _attach_inner(nodes), "edges": edges,
            "bands": [{"id": "agent", "label": "AJAN · AgentChat"},
                      {"id": "gateway", "label": "GATEWAY · bizim hat"}],
            "shape": "tek ajan · tool döngüsü", "team": None}


# GraphFlow'un katılımcıları, `pipeline/graph.py`'deki kurulumun aynısı. Elle
# yazılı olması bilinçli: graf orada da elle kuruluyor ve tek doğru bu.
# Her kutunun altında ne olduğu. Kutu adı ne olduğunu, bu satır ne yaptığını
# söylüyor — ikisi ayrı sorular ve ikincisi ekranda hiç cevaplanmıyordu.
SCAN_NOTES = {
    "technical": "Kendi alanında tek geçiş. Üçü aynı anda koşuyor — "
                 "eşzamanlılığın kaynağı bu, model kararı değil.",
    "market": "Pazar tarafı. Kardeşlerinden habersiz: kimse kimseyi beklemiyor.",
    "team": "Ekip tarafı. Üç dal da aynı görevi farklı gözle okuyor.",
    "risk": "Üç dal da gelmeden başlamıyor. Gelmeyen dal missing_data'ya "
            "yazılıyor: sessiz eksik, beyan edilmiş bilgi yokluğuna çevriliyor.",
    "scorer": "Şemaya bağlı çıktı üretiyor. Takıma custom_message_types ile "
              "beyan edilmezse runtime 'is not registered' diye düşüyor.",
}


SCAN_NODES = [
    ("technical", "Teknik analist", "AssistantAgent"),
    ("market", "Pazar analisti", "AssistantAgent"),
    ("team", "Ekip analisti", "AssistantAgent"),
    ("risk", "Risk denetçisi", "AssistantAgent"),
    ("scorer", "Skorlayıcı", "AssistantAgent · StructuredMessage[Score]"),
]


def _scan_graph(run: Run) -> dict[str, Any]:
    """Çok ajanlı akış: üç paralel dal → join(all) → sıralı kuyruk.

    Bu grafın anlattığı şey kılavuzun iki deseninin üst üste konması: soldaki
    üç kutu Concurrent Agents, sağdaki zincir Sequential Workflow. Üç dalın
    aynı sütunda yan yana durması tesadüf değil — eşzamanlılığın ekranda
    görünen tek kanıtı o.
    """
    nodes: list[dict[str, Any]] = [
        {"id": "start", "band": 0, "kind": "user", "name": "Şirket",
         "sub": "TextMessage",
         "note": "Tek yayın. Kaç dalın dinlediğini yayınlayan taraf bilmiyor."},
    ]
    nodes += [{"id": nid, "band": 0, "kind": "agent", "name": name, "sub": sub,
               "lane": "agentchat", "note": SCAN_NOTES.get(nid, "")}
              for nid, name, sub in SCAN_NODES]
    edges: list[dict[str, Any]] = []
    for nid in ("technical", "market", "team"):
        edges.append({"src": "start", "dst": nid, "message": "TextMessage",
                      "at": 0.0, "parallel": True,
                      "note": "Aynı görev üç dala birden — tek yayın, çok dal."})
        edges.append({"src": nid, "dst": "risk", "message": "TextMessage",
                      "at": 0.0, "join": True,
                      "note": 'activation_condition="all" — üçü de gelmeden '
                              "risk denetçisi başlamıyor."})
    edges.append({"src": "risk", "dst": "scorer", "message": "TextMessage",
                  "at": 0.0, "note": "Sıralı devir."})
    nodes.append({"id": "done", "band": 0, "kind": "user", "name": "Skor",
                  "sub": "TaskResult", "terminal": True,
                  "note": "Beklenen dal SAYISI sayıldı — runtime'ın 'boşta' "
                          "demesi beklenmedi."})
    edges.append({"src": "scorer", "dst": "done",
                  "message": "StructuredMessage[Score]", "at": run.seconds,
                  "note": "Şemaya bağlanmış çıktı."})

    # ---- alt bant: bu koşuda gateway ne yaptı -----------------------------
    nodes += [
        {"id": "runtime", "band": 1, "kind": "component", "name": "Runtime",
         "sub": "SingleThreadedAgentRuntime", "lane": "core",
         "note": "Mesajı tipe göre yönlendiren katman. Bu runtime kısa ömürlü; "
                 "gateway'inki sürekli koşuyor."},
        {"id": "topic", "band": 1, "kind": "component", "name": "Tek topic",
         "sub": "group_topic_type", "lane": "agentchat",
         "note": "Beş katılımcının hepsi AYNI konuya abone. Kenarlar veri "
                 "taşımıyor, yalnız sırayı belirliyor."},
        {"id": "audit", "band": 1, "kind": "component", "name": "Müdahale kapısı",
         "sub": "AuditingInterventionHandler", "lane": "ours",
         "note": "Runtime'a takılan tek kapı. Takmanın bedeli: runtime'ı kendin "
                 "kurmak zorundasın."},
        {"id": "trace", "band": 1, "kind": "component", "name": "İz",
         "sub": "runs.jsonl", "lane": "ours", "terminal": True,
         "note": "Koşunun şekli diske yazıldı."},
    ]
    edges += [
        {"src": "runtime", "dst": "topic", "message": "abonelik", "at": 0.0,
         "note": "Beş katılımcının hepsi aynı konuya abone."},
        {"src": "topic", "dst": "audit", "message": "her mesaj", "at": 0.0,
         "note": "Runtime'a takılı tek kapı: her mesaj buradan geçiyor."},
        {"src": "audit", "dst": "trace", "message": "kayıt", "at": run.seconds,
         "note": "Koşunun şekli diske yazıldı."},
        {"src": "start", "dst": "topic", "message": "yayın", "cross": True,
         "at": 0.0,
         "note": "Kenarlar veri taşımıyor: mesaj zaten topic üstünden herkese "
                 "gitti, graf yalnız sıranın kimde olduğunu söylüyor."},
    ]

    return {"nodes": _attach_inner(nodes), "edges": edges,
            "bands": [{"id": "agent", "label": "AJANLAR · GraphFlow"},
                      {"id": "gateway", "label": "GATEWAY · bizim hat"}],
            "shape": "eşzamanlı dal + join(all) → sıralı", "team": "GraphFlow"}


# Takım koşusunun şeritleri. Kadro sabit, o yüzden burada da sabit.
TEAM_LANES = [
    {"id": "user", "name": "Görev", "lane": ""},
    {"id": "Planner", "name": "Planner", "lane": "agentchat", "sub": "AssistantAgent"},
    {"id": "Researcher", "name": "Researcher", "lane": "agentchat", "sub": "AssistantAgent"},
    {"id": "Critic", "name": "Critic", "lane": "agentchat", "sub": "AssistantAgent"},
]


def _cast(run: Run) -> list[str]:
    """Sahnedeki herkes: kadro + kadroda olmayıp gerçekten konuşanlar.

    Ölçüldü (team-0003, magenticone): takım kendi yöneticisini yaratıyor
    (`MagenticOneOrchestrator`) ve konuşmaların yarısı ona ait. Kadroyu sabit
    saymak, o konuşmaları hiç olmayan bir kutuya yönlendiriyordu — grafta
    kayboluyor, sıra diyagramında var olmayan bir şeride ok çiziliyordu.
    """
    names = [a["name"] for a in TEAM_ROSTER]
    for s in _speakers(run):
        if s["who"] not in names:
            names.append(s["who"])
    return names


def _speakers(run: Run) -> list[dict[str, Any]]:
    """Gerçekten konuşan ajanlar, sırayla — kayıttan."""
    out = []
    for e in run._stages():
        if e.get("id") == "speaker":
            meta = e.get("meta") or {}
            out.append({"who": str(meta.get("who", "?")), "at": e.get("at", 0),
                        "turn": meta.get("turn")})
    return out


def _team_tools(run: Run) -> dict[str, list[dict[str, Any]]]:
    """Hangi ajan hangi tool'u kaç kez çağırdı — kayıttan, sırasıyla.

    Konuşma sırası "kim konuştu"yu söylüyor; bu "ne yaptı"yı. İkisi olmadan
    beş takım tipi ekranda birbirinin aynı üç kutu olarak duruyor, ve grafın
    anlatması gereken tek fark tam olarak iş bölümü.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for e in run._stages():
        if e.get("id") != "team_tool":
            continue
        meta = e.get("meta") or {}
        who, tool = str(meta.get("who", "?")), str(meta.get("tool", "?"))
        rows = out.setdefault(who, [])
        hit = next((r for r in rows if r["tool"] == tool), None)
        if hit is None:
            rows.append({"tool": tool, "n": 1, "at": e.get("at", 0)})
        else:
            hit["n"] += 1
    return out


def _team_sequence(run: Run) -> dict[str, Any]:
    """Takımın sıra diyagramı: her konuşma bir ok, devirler işaretli."""
    order = _speakers(run)
    lanes = [TEAM_LANES[0]] + [
        {"id": name, "name": name, "lane": "agentchat",
         "sub": "AssistantAgent" if name in [a["name"] for a in TEAM_ROSTER]
                else "takımın kendi yöneticisi"}
        for name in _cast(run)
    ]
    handoffs = {}
    for e in run._stages():
        if e.get("id") == "handoff":
            meta = e.get("meta") or {}
            handoffs[round(float(e.get("at", 0)), 3)] = str(meta.get("to", "?"))

    steps: list[dict[str, Any]] = []
    prev = "user"
    for s in order:
        steps.append({"src": prev, "dst": s["who"], "at": s["at"],
                      "label": f"tur {s['turn']}", "stage": "speaker",
                      "kind": "self" if prev == s["who"] else "call"})
        prev = s["who"]
    for at, target in handoffs.items():
        steps.append({"src": prev, "dst": target, "at": at,
                      "label": "Handoff · transfer_to_" + target.lower(),
                      "stage": "handoff", "kind": "call"})
    steps.sort(key=lambda x: x["at"])

    groups = []
    if steps:
        groups.append({"label": "Takım döngüsü — sonlandırma koşuluna kadar",
                       "kind": "loop", "from": 0, "to": len(steps) - 1})
    return {"lanes": lanes, "steps": steps, "groups": groups, "blocked": False}


def _team_graph(run: Run) -> dict[str, Any]:
    """Takım koşusu: kadro sabit, ama **kimin ne zaman konuştuğu** koşuya ait.

    Beş takım tipinin aynı kadroyla farklı çıkması gereken tek yer bu graf.
    Kutuları elle dizmek yerine gözlenen konuşma sırasından çiziliyor: sabit bir
    şema, RoundRobin ile Swarm'ı aynı gösterirdi ve ekranın anlatması gereken
    tek fark tam olarak o.
    """
    order = _speakers(run)
    roster = _cast(run)

    nodes: list[dict[str, Any]] = [
        {"id": "user", "band": 0, "kind": "user", "name": "Görev",
         "sub": "TextMessage", "note": "Takıma verilen tek görev."},
    ]
    for name in roster:
        turns = [s["turn"] for s in order if s["who"] == name]
        nodes.append({
            "id": name, "band": 0, "kind": "agent", "name": name,
            "sub": "AssistantAgent", "lane": "agentchat",
            "note": (f"{len(turns)} kez konuştu (tur {', '.join(map(str, turns))})."
                     if turns else "Bu koşuda sırası hiç gelmedi."),
        })

    edges: list[dict[str, Any]] = []
    if order:
        edges.append({"src": "user", "dst": order[0]["who"], "message": "TextMessage",
                      "at": order[0]["at"], "note": "Görev ilk konuşmacıya gitti."})

    # Sıra DÖNGÜSEL olabiliyor: RoundRobin üçüncü ajandan sonra birinciye
    # dönüyor. Katman hesabı döngüsüz graf varsayıyor, ve dönüş kenarı ileri
    # sayıldığında her tur bir sütun daha ekliyordu — ölçüldü: altı konuşmada
    # graf yirmi sütuna çıkıp karta sığdırılınca kılcal çizgilere döndü.
    #
    # Çözüm iki parçalı: ilk görünme sırasına göre geriye giden her geçiş
    # `back` işaretleniyor (ve alttan dolanan bir ok olarak çiziliyor), aynı iki
    # ajan arasındaki tekrarlar tek kenarda toplanıyor.
    rank = {name: i for i, name in enumerate(roster)}
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        if a["who"] == b["who"]:
            continue
        key = (a["who"], b["who"])
        row = merged.get(key)
        if row is None:
            merged[key] = {
                "src": a["who"], "dst": b["who"], "turns": [b["turn"]],
                "at": b["at"],
                "back": rank.get(b["who"], 0) <= rank.get(a["who"], 0),
            }
        else:
            row["turns"].append(b["turn"])
    for row in merged.values():
        turns = [str(t) for t in row["turns"] if t is not None]
        edges.append({
            "src": row["src"], "dst": row["dst"], "at": row["at"],
            "back": row["back"],
            "message": ("tur " + ", ".join(turns)) if turns else "devir",
            "note": ("Sıra başa döndü — döngüsel." if row["back"]
                     else "Sıra devretti.") +
                    (f" {len(turns)} kez." if len(turns) > 1 else ""),
        })

    # Ajanın çağırdığı tool'lar, ajanın hemen sağında. Sütun hesabı ileri
    # kenarlardan çıktığı için tool düğümü ajanı bir sağa itmiyor — kendi
    # sütununu alıyor ve dönüş oku `back` ile alttan dolanıyor.
    for who, rows in _team_tools(run).items():
        if who not in roster:
            continue
        for i, row in enumerate(rows):
            tid = f"tool:{who}:{row['tool']}"
            nodes.append({
                "id": tid, "band": 0, "kind": "tool", "name": row["tool"],
                "sub": "workbench.call_tool", "lane": "core",
                "note": (f"{who} çağırdı"
                         + (f", {row['n']} kez." if row["n"] > 1 else ".")),
            })
            edges.append({"src": who, "dst": tid, "at": row["at"],
                          "message": "ToolCallRequestEvent",
                          "note": f"{who} bu tool'a uzandı."})
            edges.append({"src": tid, "dst": who, "at": row["at"], "back": True,
                          "message": "ToolCallExecutionEvent",
                          "note": "Sonuç ajanın bağlamına döndü."})
            del i

    done = next((e for e in run._stages() if e.get("id") == "team_done"), None)
    if done:
        meta = done.get("meta") or {}
        nodes.append({"id": "result", "band": 0, "kind": "user", "name": "TaskResult",
                      "sub": "stop_reason", "terminal": True,
                      "note": str(meta.get("stop_reason", ""))[:110] or
                              "Sonlandırma koşulu tetiklendi."})
        if order:
            edges.append({"src": order[-1]["who"], "dst": "result",
                          "message": "TaskResult", "at": done.get("at", 0),
                          "note": "Bütün konuşma ve durma sebebi."})

    nodes += [
        {"id": "runtime", "band": 1, "kind": "component", "name": "Runtime",
         "sub": "SingleThreadedAgentRuntime", "lane": "core",
         "note": "Takım kendi runtime'ında koştu; span'ler buradan çıkıyor."},
        {"id": "otel", "band": 1, "kind": "component", "name": "Telemetri",
         "sub": "OTel · gen_ai.*", "lane": "ext",
         "note": "Her ajan ve tool çağrısı bir span. Şelale bundan çiziliyor."},
        {"id": "trace", "band": 1, "kind": "component", "name": "İz",
         "sub": "runs.jsonl", "lane": "ours", "terminal": True,
         "note": "Koşunun şekli diske yazıldı."},
    ]
    edges += [
        {"src": "runtime", "dst": "otel", "message": "span", "at": 0.0,
         "note": "tracer_provider runtime'a verildi."},
        {"src": "otel", "dst": "trace", "message": "kayıt", "at": run.seconds,
         "note": "Takım, desen, konuşmacılar ve token."},
        {"src": "user", "dst": "runtime", "message": "takım koşusu", "cross": True,
         "at": 0.0, "note": "Takımın mesajları runtime üstünden gidiyor."},
    ]

    return {"nodes": _attach_inner(nodes), "edges": edges,
            "bands": [{"id": "agent", "label": "TAKIM · AgentChat"},
                      {"id": "gateway", "label": "GATEWAY · bizim hat"}],
            "shape": "gözlenen konuşma sırası", "team": run.variant}


def _cron_graph(run: Run) -> dict[str, Any]:
    """Zamanlama turu: kısa, ve tamamı alt bantta.

    Üst bant boş kalıyor ve bu **doğru**: zamanlamada AutoGen'in hiçbir parçası
    çalışmıyor. Bir kütüphane saat tutmaz, ve bunu boş bir bant olarak göstermek
    "her şey AutoGen'in içinde oluyor" izlenimini düzelten tek yer.
    """
    ids = run._stage_ids()
    meta: dict[str, dict[str, Any]] = {
        str(e.get("id")): (e.get("meta") or {}) for e in run._stages()
    }
    parse, gate, done = meta.get("cron_parse", {}), meta.get("cron_gate", {}), \
        meta.get("cron_done", {})

    nodes: list[dict[str, Any]] = [
        {"id": "user", "band": 1, "kind": "user", "name": "Cümle",
         "sub": "/openclaw schedule",
         "note": str(run.question)[:110] or "Zamanlama isteği."},
        {"id": "parse", "band": 1, "kind": "component", "name": "Ayrıştırıcı",
         "sub": "scheduler.parse_command", "lane": "ours",
         "note": (f"Hata: {parse['error']}"[:110] if parse.get("error")
                  else "Türkçe cümle cron ifadesine çevrildi.")},
        {"id": "gate", "band": 1, "kind": "gate", "name": "Kapı",
         "sub": "GATE.require · cron.add", "lane": "ours",
         "note": ("Onay bekliyor — imza yazılan cümlenin üstünde."
                  if gate.get("held") else
                  "İmza yazılan cümlenin üstünde, çözülmüş zamanın değil.")},
        {"id": "cron", "band": 1, "kind": "component", "name": "Zamanlayıcı",
         "sub": "openclaw cron.add", "lane": "ext", "terminal": True,
         "note": (f"{done.get('when', '')}"[:110] if done.get("when")
                  else "OpenClaw'ın Gateway sürecinde, SQLite'ta.")},
    ]
    edges: list[dict[str, Any]] = [
        {"src": "user", "dst": "parse", "message": "metin", "at": 0.0,
         "note": "Ne yazıldıysa o — ayrıştırma sonrası değil."},
        {"src": "parse", "dst": "gate", "message": "cron ifadesi", "at": 0.0,
         "note": "Dışarı yazan bir çağrı; kapıdan geçmesi gerekiyor."},
        {"src": "gate", "dst": "cron",
         "message": "red" if gate.get("held") else "cron.add",
         "at": 0.0,
         "note": ("Onay tüketilmedi: iş yaratılmadı." if gate.get("held")
                  else "İş OpenClaw'ın zamanlayıcısına yazıldı.")},
    ]
    if "cron_done" not in ids and "cron_gate" in ids:
        nodes[-1]["note"] = "Henüz yazılmadı."
    return {"nodes": _attach_inner(nodes), "edges": edges,
            "bands": [{"id": "agent", "label": "AJAN · bu turda yok"},
                      {"id": "gateway", "label": "GATEWAY · bizim hat"}],
            "shape": "zamanlama devri — AutoGen'in payı yok"}


MAF_LANES = [
    {"id": "user", "name": "Soru", "lane": ""},
    {"id": "agent", "name": "Agent", "lane": "maf", "sub": "agent_framework"},
    {"id": "gate", "name": "Onay", "lane": "maf", "sub": "ToolApprovalMiddleware"},
    {"id": "tool", "name": "Tool", "lane": "maf", "sub": "FunctionTool"},
]

MAF_SEQUENCE: dict[str, tuple[str, str, str]] = {
    "maf_build": ("user", "agent", "OpenAIChatClient kuruldu"),
    "maf_tool": ("agent", "tool", "FunctionTool · approval_mode"),
    "maf_gate": ("agent", "gate", "ToolApprovalMiddleware takıldı"),
    "maf_session": ("agent", "agent", "AgentSession açıldı"),
    "maf_run": ("user", "agent", "Agent.run(session=...)"),
    "maf_approval": ("gate", "user", "user_input_requests · onay istendi"),
    "maf_done": ("agent", "user", "AgentResponse"),
}


def _maf_sequence(run: Run) -> dict[str, Any]:
    steps = []
    for row in run.timeline():
        move = MAF_SEQUENCE.get(row["id"])
        if move is None:
            continue
        src, dst, label = move
        steps.append({"src": src, "dst": dst, "at": row["at"], "label": label,
                      "stage": row["id"],
                      "kind": "self" if src == dst else "call"})
    groups = []
    if steps:
        groups.append({"label": "MAF turu — tek Agent.run çağrısı",
                       "kind": "loop", "from": 0, "to": len(steps) - 1})
    return {"lanes": MAF_LANES, "steps": steps, "groups": groups, "blocked": False}


def _maf_graph(run: Run) -> dict[str, Any]:
    """MAF turu. Üst bant çerçevenin kendi parçaları, alt bant bizim hattımız.

    Dikkat çeken şey alt bandın **boşluğu**: AutoGen turunda orada kapı,
    workbench ve bağlam bütçesi var — hepsi bizim yazdığımız. MAF'ta onların
    karşılığı üst banda, çerçevenin içine taşınıyor. Ekranın anlattığı fark bu.
    """
    ids = run._stage_ids()
    meta = {e["id"]: (e.get("meta") or {}) for e in run._stages()}
    tool_meta = meta.get("maf_tool", {})

    nodes: list[dict[str, Any]] = [
        {"id": "user", "band": 0, "kind": "user", "name": "Soru",
         "sub": "str", "note": "Tek çağrı: Agent.run(messages, session=...)."},
        {"id": "agent", "band": 0, "kind": "agent", "name": "Agent",
         "sub": "agent_framework.Agent", "lane": "maf",
         "note": "Ayrı bir run_stream() yok; akış run(stream=True) parametresi."},
        {"id": "gate", "band": 0, "kind": "component", "name": "Onay",
         "sub": "ToolApprovalMiddleware", "lane": "maf",
         "note": "Kapı çerçevenin içinde. AutoGen'de bunu GatedWorkbench olarak "
                 "biz yazmıştık."},
        {"id": "tool", "band": 0, "kind": "tool", "name": "FunctionTool",
         "sub": str(tool_meta.get("name", "sirket_sayisi")), "lane": "maf",
         "note": f"approval_mode={tool_meta.get('approval_mode', '?')} · "
                 f"max_invocations={tool_meta.get('max_invocations', '?')} — "
                 "ikisi de tool'un kendi alanı."},
    ]
    edges: list[dict[str, Any]] = [
        {"src": "user", "dst": "agent", "message": "str", "at": 0.0,
         "note": "Soru doğrudan ajana."},
        {"src": "agent", "dst": "gate", "message": "tool çağrısı", "at": 0.0,
         "note": "Her çağrı ara katmandan geçiyor."},
        {"src": "gate", "dst": "tool", "message": "izin", "at": 0.0,
         "note": "approval_mode karar veriyor."},
    ]

    if "maf_approval" in ids:
        nodes.append({"id": "req", "band": 0, "kind": "user", "name": "Onay isteği",
                      "sub": "user_input_requests", "terminal": True,
                      "note": "Tur DURDU: finish_reason='tool_calls' ve cevap "
                              "yerine bir function_approval_request döndü."})
        edges.append({"src": "gate", "dst": "req", "message": "function_approval_request",
                      "at": run.seconds, "note": "Onay cevabın birinci sınıf alanı."})
    else:
        done = meta.get("maf_done", {})
        nodes.append({"id": "resp", "band": 0, "kind": "user", "name": "AgentResponse",
                      "sub": str(done.get("finish", "")) or "finish_reason",
                      "terminal": True,
                      "note": ("Tool çağrıldığında text BOŞ kalıyor; cevap "
                               "messages içinde." if done.get("from_messages")
                               else "Cevap doğrudan response.text içinde.")})
        edges.append({"src": "tool", "dst": "resp", "message": "AgentResponse",
                      "at": run.seconds, "note": "Tur bitti."})

    nodes += [
        {"id": "venv", "band": 1, "kind": "component", "name": "Ayrı ortam",
         "sub": ".venv-maf", "lane": "ours",
         "note": "İki çerçeve aynı bağımlılık ağacını paylaşamıyor; MAF ayrı "
                 "bir sanal ortamda alt süreç olarak koşuyor."},
        {"id": "bridge", "band": 1, "kind": "component", "name": "Köprü",
         "sub": "##STAGE satırları", "lane": "ours",
         "note": "Taramanın kullandığı protokolün aynısı: stdout'tan tek "
                 "satırlık JSON."},
        {"id": "trace", "band": 1, "kind": "component", "name": "İz",
         "sub": "runs.jsonl", "lane": "ours", "terminal": True,
         "note": "MAF turu da aynı ize yazılıyor."},
    ]
    edges += [
        {"src": "venv", "dst": "bridge", "message": "stdout", "at": 0.0,
         "note": "Alt sürecin çıktısı."},
        {"src": "bridge", "dst": "trace", "message": "kayıt", "at": run.seconds,
         "note": "Aynı katalogdan geçiyor, aynı ekran çiziyor."},
        {"src": "agent", "dst": "venv", "message": "alt süreç", "cross": True,
         "at": 0.0, "note": "Ajan bu süreçte değil, ötekinde."},
    ]

    return {"nodes": _attach_inner(nodes), "edges": edges,
            "bands": [{"id": "agent", "label": "MAF · agent_framework"},
                      {"id": "gateway", "label": "GATEWAY · bizim hat"}],
            "shape": "tek ajan · çerçeve içi kapı", "team": None}


def _trace(run: Run) -> None:
    """Turu ize yaz. Asla fırlatmaz — bir turu, kendini kaydederken düşüremeyiz.

    JSONL ve append-only: iki süreç aynı anda yazabilir (sunucu ve tarama alt
    süreci ayrı ayrı bitirebilir) ve satır bütünlüğü tek `write` çağrısıyla
    korunuyor. Döndürme yok; dosya büyürse elle alınır, ve bunu gizlemiyoruz.
    """
    try:
        line = json.dumps(run.trace_record(), ensure_ascii=False)
        with config.RUN_TRACE.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ the store
class RunLog:
    """Son N tur, bellekte. Kalıcı değil ve olmaması bilinçli.

    Bu bir denetim kaydı değil — o `audit` tarafında ve orada kalmalı. Bu ekran
    "az önce ne oldu"yu cevaplıyor, ve o soru birkaç tur sonra sorulmuyor.
    Kalıcı yapmak, sohbet metnini diske yazmak demek olurdu; ayrı bir karar.
    """

    def __init__(self, cap: int = 40) -> None:
        self._runs: dict[str, Run] = {}
        self._order: list[str] = []
        self._cap = cap
        self._seq = itertools.count(1)

    def begin(self, kind: str, question: str, session: str = "local") -> Run:
        rid = f"{kind}-{next(self._seq):04d}"
        run = Run(id=rid, kind=kind, question=question, session=session)
        self._runs[rid] = run
        self._order.append(rid)
        while len(self._order) > self._cap:
            self._runs.pop(self._order.pop(0), None)
        return run

    def get(self, rid: str) -> Run | None:
        return self._runs.get(rid)

    def latest(self, session: str | None = None) -> Run | None:
        for rid in reversed(self._order):
            run = self._runs[rid]
            if session is None or run.session == session:
                return run
        return None

    def listing(self, limit: int = 20) -> list[dict[str, Any]]:
        out = [self._runs[r].summary() for r in reversed(self._order[-limit:])]
        return out

    def record(self, run: Run, events: Iterable[dict[str, Any]]) -> None:
        for e in events:
            run.event(e)


LOG = RunLog()
