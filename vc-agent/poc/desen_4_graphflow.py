"""Desen 4 — GraphFlow: yürütme grafiği (DiGraph) + paralel dal.

Diğer üç desende akış **doğrusal**: bir anda tek ajan konuşur. GraphFlow bunu
kırar — kenarları önceden çizersin, birbirinden bağımsız düğümler **eşzamanlı**
koşar, birleşme noktası `activation_condition="all"` ile bekler.

```
                 ┌── AnalistGelir ──┐
Arastirmaci ─────┤                  ├──→ Yazar   (join: ikisini de bekler)
                 └── AnalistHata ───┘
```

Bu, AutoGen'in LangGraph'a verdiği cevap. Atlas'taki (14-agentic-mega-atlas.md)
"orkestrasyon = graf/koordinatör" ailesine AutoGen'in de girdiği yer.

Uygulama notu: burada her ajanın **kendi model istemcisi** var. Paralel dallar
tek bir replay istemcisini paylaşsaydı yanıtları hangi sırayla tüketecekleri
belirsiz olurdu — ayrı istemci hem determinizmi korur hem de gerçek hayatta
ajan başına farklı model kullanmanın (ucuz model → analiz, güçlü model → yazım)
doğal yolu.
"""

from __future__ import annotations

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

import araclar
import motor

GOREV = "'gelir' ve 'hata_orani' serilerini paralel analiz et, tek raporda birleştir."

SENARYO_ARASTIRMACI = [
    motor.arac_cagrisi("veri_getir", '{"metrik": "gelir"}'),
    "Gelir ve hata_orani serileri hazır; iki analist paralel başlayabilir.",
]
SENARYO_GELIR = [
    motor.arac_cagrisi(
        "istatistik_hesapla",
        '{"seri": [120.5, 133.0, 128.75, 141.2, 155.9, 149.3, 162.0]}',
    ),
    "GELİR: ortalama 141.52, std 13.96, dönem artışı %34.44.",
]
SENARYO_HATA = [
    motor.arac_cagrisi(
        "istatistik_hesapla",
        '{"seri": [0.031, 0.028, 0.035, 0.022, 0.019, 0.024, 0.017]}',
    ),
    "HATA ORANI: ortalama 0.0251, dönem değişimi -%45.16 (iyileşme).",
]
SENARYO_YAZAR = [
    (
        "# Birleşik Rapor\n"
        "Gelir %34.44 artarken hata oranı %45.16 geriledi. "
        "İki eğilim aynı yönde olumlu.\nRAPOR_TAMAM"
    )
]


def takim_kur():
    """Takımı ve kullanılan istemcileri birlikte döndürür (ölçüm hepsini toplayacak)."""
    c_ara = motor.istemci(SENARYO_ARASTIRMACI)
    c_gelir = motor.istemci(SENARYO_GELIR)
    c_hata = motor.istemci(SENARYO_HATA)
    c_yazar = motor.istemci(SENARYO_YAZAR)

    arastirmaci = AssistantAgent(
        "Arastirmaci",
        model_client=c_ara,
        tools=[araclar.veri_getir],
        reflect_on_tool_use=True,
        system_message="İki seriyi de çek ve hazır olduğunu bildir.",
    )
    analist_gelir = AssistantAgent(
        "AnalistGelir",
        model_client=c_gelir,
        tools=[araclar.istatistik_hesapla],
        reflect_on_tool_use=True,
        system_message="Yalnızca gelir serisini analiz et. Cevabına GELİR: ile başla.",
    )
    analist_hata = AssistantAgent(
        "AnalistHata",
        model_client=c_hata,
        tools=[araclar.istatistik_hesapla],
        reflect_on_tool_use=True,
        system_message="Yalnızca hata oranını analiz et. Cevabına HATA ORANI: ile başla.",
    )
    yazar = AssistantAgent(
        "Yazar",
        model_client=c_yazar,
        system_message="İki analizi tek raporda birleştir. Bitirince RAPOR_TAMAM yaz.",
    )

    # Grafı kur: fan-out → join
    kurucu = DiGraphBuilder()
    for a in (arastirmaci, analist_gelir, analist_hata, yazar):
        kurucu.add_node(a)
    kurucu.add_edge(arastirmaci, analist_gelir)
    kurucu.add_edge(arastirmaci, analist_hata)
    # activation_condition="all": Yazar İKİ analisti de bekler (join bariyeri).
    kurucu.add_edge(analist_gelir, yazar, activation_group="analiz", activation_condition="all")
    kurucu.add_edge(analist_hata, yazar, activation_group="analiz", activation_condition="all")
    kurucu.set_entry_point(arastirmaci)

    durma = TextMentionTermination("RAPOR_TAMAM") | MaxMessageTermination(30)
    takim = GraphFlow(
        participants=[arastirmaci, analist_gelir, analist_hata, yazar],
        graph=kurucu.build(),
        termination_condition=durma,
    )
    return takim, [c_ara, c_gelir, c_hata, c_yazar]


async def calistir() -> motor.Olcum:
    takim, istemciler = takim_kur()
    try:
        return await motor.olc(
            "Desen 4 · GraphFlow (paralel dal + join)", takim, GOREV, istemciler
        )
    finally:
        for c in istemciler:
            await c.close()


if __name__ == "__main__":
    asyncio.run(calistir())
