"""Desen 3 — Swarm: handoff ile devir.

Kursta **olmayan** desen; OpenAI Agents SDK ile karşılaştırmanın tam kalbi.

Fark şurada: Desen 1'de sırayı harness dayatır, Desen 2'de harness seçer,
burada **ajanın kendisi** "bu iş bende değil" deyip devreder. Devir bir mesaj
değil, bir **tool çağrısıdır**: `Handoff(target="X")` AssistantAgent'a
`transfer_to_x` adında bir tool takar. Model o tool'u çağırınca Swarm konuşma
sırasını hedefe verir.

Bunun bedeli ölçümde görünür: her devir **bir LLM çağrısı harcar** ve devir
kararı görev üretmez. Desen 2'de aynı yönlendirmeyi sıfır token'a yapan bir
Python fonksiyonu vardı. Özerklik ile maliyet arasındaki takas tam burada.

Not: `HandoffTermination(target="user")` ile insana da devredilebilir —
kursun 9. modülündeki human-in-the-loop'un Swarm karşılığı budur.
"""

from __future__ import annotations

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Handoff
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import Swarm

import araclar
import motor

GOREV = "Son 7 günün 'gelir' serisini çek, istatistiklerini hesapla ve kısa bir rapor yaz."

# Handoff tool adı hedefin küçük harfli hâlidir (transfer_to_veriuzmani).
# Elle yazmak yerine kütüphaneden okuyoruz ki isim değişirse senaryo kırılmasın.
DEVIR_VERI = Handoff(target="VeriUzmani").name
DEVIR_YAZAR = Handoff(target="Yazar").name

SENARYO = [
    # Triyaj: kendisi iş yapmaz, doğru uzmana yönlendirir.
    motor.arac_cagrisi(DEVIR_VERI, "{}"),
    # VeriUzmani: veriyi çeker → yorumlar → istatistik alır → yorumlar → devreder.
    motor.arac_cagrisi("veri_getir", '{"metrik": "gelir"}'),
    "Seri elimde: 7 gözlem, 120.5 → 162.0.",
    motor.arac_cagrisi(
        "istatistik_hesapla",
        '{"seri": [120.5, 133.0, 128.75, 141.2, 155.9, 149.3, 162.0]}',
    ),
    "Ortalama 141.52, std 13.96, %34.44 artış. Yazıma hazır.",
    motor.arac_cagrisi(DEVIR_YAZAR, "{}"),
    (
        "# Gelir Raporu\n"
        "7 günlük ortalama 141.52, std 13.96, dönem artışı %34.44.\nRAPOR_TAMAM"
    ),
]


def takim_kur(istemci):
    triyaj = AssistantAgent(
        "Triyaj",
        model_client=istemci,
        handoffs=["VeriUzmani"],
        description="Görevi doğru uzmana yönlendirir.",
        system_message=(
            "Sen bir triyaj ajanısın. İşi kendin yapma; veri gerektiren her görevi "
            "VeriUzmani'na devret."
        ),
    )
    veri_uzmani = AssistantAgent(
        "VeriUzmani",
        model_client=istemci,
        tools=[araclar.veri_getir, araclar.istatistik_hesapla],
        handoffs=["Yazar"],
        reflect_on_tool_use=True,
        description="Veriyi çeker ve istatistiğini çıkarır.",
        system_message=(
            "Veriyi çek, istatistiğini hesapla. İşin bitince Yazar'a devret."
        ),
    )
    yazar = AssistantAgent(
        "Yazar",
        model_client=istemci,
        description="Nihai raporu yazar.",
        system_message="Kısa markdown rapor yaz. Bitirince RAPOR_TAMAM yaz.",
    )

    durma = TextMentionTermination("RAPOR_TAMAM") | MaxMessageTermination(25)
    # Swarm'da ilk ajan başlar; sıra yalnızca HandoffMessage ile değişir.
    return Swarm([triyaj, veri_uzmani, yazar], termination_condition=durma)


async def calistir() -> motor.Olcum:
    istemci = motor.istemci(SENARYO)
    takim = takim_kur(istemci)
    try:
        return await motor.olc("Desen 3 · Swarm (handoff ile devir)", takim, GOREV, istemci)
    finally:
        await istemci.close()


if __name__ == "__main__":
    asyncio.run(calistir())
