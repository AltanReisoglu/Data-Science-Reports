"""Desen 1 — RoundRobinGroupChat: sabit sıra.

Harness'ın beyni yok; sıra **deterministik**. Ajanlar dairesel olarak konuşur.
Ortak mesaj thread'ini herkes görür (AutoGen'in kurucu makalesindeki
"conversable agents" fikri).

Ne zaman doğru seçim: adımlar **önceden biliniyor** ve hep aynı sırada işliyorsa.
Ne zaman yanlış: bir ajanın katkısı gereksizken bile sıra ona uğrar → boşa token.
Bu maliyet `kiyas.py` çıktısında Desen 2 ile yan yana görülüyor.
"""

from __future__ import annotations

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat

import araclar
import motor

GOREV = "Son 7 günün 'gelir' serisini çek, istatistiklerini hesapla ve kısa bir rapor yaz."

# Replay senaryosu: LLM'in sırayla ne döndüreceği. Gerçek modda yok sayılır.
SENARYO = [
    motor.arac_cagrisi("veri_getir", '{"metrik": "gelir"}'),
    "Gelir serisini çektim: 120.5 → 162.0 arası 7 gözlem. Analiste devrediyorum.",
    motor.arac_cagrisi(
        "istatistik_hesapla",
        '{"seri": [120.5, 133.0, 128.75, 141.2, 155.9, 149.3, 162.0]}',
    ),
    "Ortalama 141.52, medyan 141.2, std 13.96; seri başından sonuna %34.44 artmış.",
    # ↓ Sıra Eleştirmen'e uğramak ZORUNDA — söyleyecek bir şeyi olmasa bile.
    #   Desen 2'de aynı ajan atlanıyor; fark ölçümde görünüyor.
    "Veri temiz, aykırı değer yok, itirazım yok.",
    (
        "# Gelir Raporu\n"
        "7 günlük gelir ortalaması 141.52 (medyan 141.2, std 13.96). "
        "Seri baştan sona %34.44 artış göstermiş; tek gerileme 3. günde. "
        "Trend pozitif.\nRAPOR_TAMAM"
    ),
]


def takim_kur(istemci):
    arastirmaci = AssistantAgent(
        "Arastirmaci",
        model_client=istemci,
        tools=[araclar.veri_getir],
        reflect_on_tool_use=True,
        system_message="Veri çekersin. Tool'u çağır, sonucu tek cümleyle özetle.",
    )
    analist = AssistantAgent(
        "Analist",
        model_client=istemci,
        tools=[araclar.istatistik_hesapla],
        reflect_on_tool_use=True,
        system_message="İstatistik hesaplarsın. Tool'u çağır, sayıları yorumla.",
    )
    elestirmen = AssistantAgent(
        "Elestirmen",
        model_client=istemci,
        system_message="Veri kalitesini denetlersin. Sorun yoksa kısa yaz.",
    )
    yazar = AssistantAgent(
        "Yazar",
        model_client=istemci,
        system_message=(
            "Kısa bir markdown rapor yazarsın. Bitirince son satıra RAPOR_TAMAM yaz."
        ),
    )

    # İki koşuldan biri yeterli: OR ile birleştirilir. Sonsuz döngüye karşı sigorta.
    durma = TextMentionTermination("RAPOR_TAMAM") | MaxMessageTermination(20)
    return RoundRobinGroupChat(
        [arastirmaci, analist, elestirmen, yazar], termination_condition=durma
    )


async def calistir() -> motor.Olcum:
    istemci = motor.istemci(SENARYO)
    takim = takim_kur(istemci)
    try:
        return await motor.olc("Desen 1 · RoundRobinGroupChat (sabit sıra)", takim, GOREV, istemci)
    finally:
        await istemci.close()


if __name__ == "__main__":
    asyncio.run(calistir())
