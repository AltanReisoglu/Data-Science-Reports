"""Desen 2 — SelectorGroupChat: konuşmacıyı seçen harness.

Desen 1 ile **aynı 4 ajan, aynı görev**. Tek fark: sırayı sabit tutmak yerine
her turda "şimdi kim konuşmalı?" sorusu sorulur. Atlas'ta (14-agentic-mega-atlas.md)
"konuşmacı-seçimi harness'ın beyni" dediğimiz yer tam olarak burası.

İki seçim yolu var:

1. **LLM seçici** (varsayılan) — her turda fazladan bir LLM çağrısı. Esnek ama pahalı.
2. **`selector_func`** — deterministik Python fonksiyonu. Sıfır ek token.
   Kursun 7.2'de gösterdiği yol; burada da bu kullanılıyor.

Ölçümdeki kritik satır: Eleştirmen **atlanıyor** (veri temiz olduğu için), yani
Desen 1'e göre bir LLM çağrısı ve bir tur mesaj daha az. Yönlendirme zekâsının
token cinsinden karşılığı budur.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat

import araclar
import motor

GOREV = "Son 7 günün 'gelir' serisini çek, istatistiklerini hesapla ve kısa bir rapor yaz."

SENARYO = [
    motor.arac_cagrisi("veri_getir", '{"metrik": "gelir"}'),
    "Gelir serisini çektim: 120.5 → 162.0 arası 7 gözlem.",
    motor.arac_cagrisi(
        "istatistik_hesapla",
        '{"seri": [120.5, 133.0, 128.75, 141.2, 155.9, 149.3, 162.0]}',
    ),
    "Ortalama 141.52, medyan 141.2, std 13.96; %34.44 artış. Aykırı değer yok.",
    # Eleştirmen'in turu YOK — selector onu atlıyor.
    (
        "# Gelir Raporu\n"
        "7 günlük gelir ortalaması 141.52 (medyan 141.2, std 13.96). "
        "Seri baştan sona %34.44 artış göstermiş.\nRAPOR_TAMAM"
    ),
]


def secici(mesajlar: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    """Sıradaki konuşmacıyı belirler.

    `None` dönerse AutoGen varsayılan (LLM tabanlı) seçiciye düşer — yani bu
    fonksiyon bir *kısayol*, tam ikame olmak zorunda değil.

    Kural: veri → analiz → (gerekiyorsa eleştiri) → yazım.
    Eleştirmen yalnızca analizde şüpheli bir sinyal varsa devreye girer.
    """
    if not mesajlar:
        return "Arastirmaci"

    son = mesajlar[-1]
    kaynak = getattr(son, "source", None)
    icerik = str(getattr(son, "content", ""))

    if kaynak == "user":
        return "Arastirmaci"
    if kaynak == "Arastirmaci":
        return "Analist"
    if kaynak == "Analist":
        # Şüphe sinyali varsa denetime yolla, yoksa doğrudan yazıma geç.
        supheli = any(k in icerik.lower() for k in ("aykırı değer var", "eksik", "tutarsız", "şüpheli"))
        return "Elestirmen" if supheli else "Yazar"
    if kaynak == "Elestirmen":
        return "Yazar"
    return None


def takim_kur(istemci):
    arastirmaci = AssistantAgent(
        "Arastirmaci",
        model_client=istemci,
        tools=[araclar.veri_getir],
        reflect_on_tool_use=True,
        description="Veri kaynağından ham seriyi çeker.",
        system_message="Veri çekersin. Tool'u çağır, sonucu tek cümleyle özetle.",
    )
    analist = AssistantAgent(
        "Analist",
        model_client=istemci,
        tools=[araclar.istatistik_hesapla],
        reflect_on_tool_use=True,
        description="Seriden istatistik çıkarır ve veri kalitesini işaretler.",
        system_message="İstatistik hesaplarsın. Aykırı değer görürsen açıkça belirt.",
    )
    elestirmen = AssistantAgent(
        "Elestirmen",
        model_client=istemci,
        description="Yalnızca veri şüpheliyse çağrılan denetçi.",
        system_message="Veri kalitesini denetlersin.",
    )
    yazar = AssistantAgent(
        "Yazar",
        model_client=istemci,
        description="Nihai markdown raporu yazar.",
        system_message="Kısa bir markdown rapor yazarsın. Bitirince RAPOR_TAMAM yaz.",
    )

    durma = TextMentionTermination("RAPOR_TAMAM") | MaxMessageTermination(20)
    return SelectorGroupChat(
        [arastirmaci, analist, elestirmen, yazar],
        model_client=istemci,   # selector_func None dönerse bu istemci seçim yapar
        selector_func=secici,
        termination_condition=durma,
        allow_repeated_speaker=False,
    )


async def calistir() -> motor.Olcum:
    istemci = motor.istemci(SENARYO)
    takim = takim_kur(istemci)
    try:
        return await motor.olc(
            "Desen 2 · SelectorGroupChat (custom selector_func)", takim, GOREV, istemci
        )
    finally:
        await istemci.close()


if __name__ == "__main__":
    asyncio.run(calistir())
