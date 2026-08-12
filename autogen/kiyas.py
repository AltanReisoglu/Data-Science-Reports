"""Beş deseni sırayla koşar, ölçümleri toplar, karşılaştırma tablosu basar.

Çıktı: `kiyas_sonuc.json` (saf-motorlar/kiyas.py ile aynı gelenek).

    python kiyas.py

Anahtar yoksa replay modunda çalışır ve sonuçlar **tekrarlanabilir** olur.
`OPENAI_API_KEY` ya da `OPENROUTER_API_KEY` verirsen aynı desenler gerçek
modelle koşar; o zaman sayılar koşudan koşuya değişir — asıl kıyas da odur.
"""

from __future__ import annotations

import asyncio
import json
import pathlib

import desen_1_roundrobin
import desen_2_selector
import desen_3_swarm
import desen_4_graphflow
import desen_5_core_aktor
import motor

CIKTI = pathlib.Path(__file__).parent / "kiyas_sonuc.json"

DESENLER = [
    ("roundrobin", desen_1_roundrobin),
    ("selector", desen_2_selector),
    ("swarm", desen_3_swarm),
    ("graphflow", desen_4_graphflow),
    ("core_aktor", desen_5_core_aktor),
]


def tablo(olcumler: list[motor.Olcum]) -> str:
    basliklar = ["desen", "mesaj", "LLM", "tool", "token", "ms", "durma nedeni"]
    satirlar = []
    for o in olcumler:
        # "Desen 1 · RoundRobinGroupChat (sabit sıra)" → "RoundRobinGroupChat"
        ad = o.desen.split("·", 1)[-1].strip().split("(")[0].strip()
        satirlar.append(
            [ad, str(o.mesaj_sayisi), str(o.llm_cagrisi), str(o.arac_cagrisi),
             str(o.toplam_token), str(o.sure_ms), (o.durma_nedeni or "—")[:34]]
        )

    genislik = [max(len(basliklar[i]), *(len(s[i]) for s in satirlar)) for i in range(len(basliklar))]
    ciz = lambda k, o, s: k + o.join("─" * (g + 2) for g in genislik) + s  # noqa: E731
    hizala = lambda h: "│" + "│".join(  # noqa: E731
        f" {h[i]:<{genislik[i]}} " if i == 0 or i == len(h) - 1 else f" {h[i]:>{genislik[i]}} "
        for i in range(len(h))
    ) + "│"

    return "\n".join([ciz("┌", "┬", "┐"), hizala(basliklar), ciz("├", "┼", "┤"),
                      *(hizala(s) for s in satirlar), ciz("└", "┴", "┘")])


async def main() -> None:
    olcumler: list[motor.Olcum] = []
    for _, modul in DESENLER:
        olcumler.append(await modul.calistir())

    print(f"\n\n{'═' * 78}")
    print(f"  KARŞILAŞTIRMA   [{'gerçek model' if motor.gercek_mod() else 'replay — deterministik'}]")
    print(f"{'═' * 78}\n")
    print(tablo(olcumler))

    # LLM kullanan dört deseni token açısından kıyasla (Desen 5 LLM'siz).
    llmli = [o for o in olcumler if o.toplam_token]
    if llmli:
        en_ucuz = min(llmli, key=lambda o: o.toplam_token)
        en_pahali = max(llmli, key=lambda o: o.toplam_token)
        fark = en_pahali.toplam_token - en_ucuz.toplam_token
        yuzde = round(fark / en_ucuz.toplam_token * 100, 1)
        print(
            f"\n  En ucuz : {en_ucuz.desen.split('·')[-1].strip()} → {en_ucuz.toplam_token} token"
            f"\n  En pahalı: {en_pahali.desen.split('·')[-1].strip()} → {en_pahali.toplam_token} token"
            f"\n  Aynı görev, aynı ajanlar: %{yuzde} fark. Ödenen şey yönlendirme özerkliği."
        )

    CIKTI.write_text(
        json.dumps(
            {
                "mod": "gercek" if motor.gercek_mod() else "replay",
                "desenler": [o.sozluk() for o in olcumler],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n  → {CIKTI.name} yazıldı.")


if __name__ == "__main__":
    asyncio.run(main())
