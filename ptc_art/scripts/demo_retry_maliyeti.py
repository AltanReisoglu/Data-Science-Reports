"""Kalıcılığın retry maliyetine etkisi — ölçülmüş karşılaştırma.

Senaryo, PTC'nin gerçek başarısızlık biçimi: LLM tek bir script yazar, script
pahalı işi yapar, SONRA bir hata verir. Düzeltilmiş sürüm baştan koşar.

    Kalıcılıksız : pahalı iş HER denemede yeniden yapılır
    Kalıcılıkla  : ilk denemede saklanır, düzeltmede artifact'ten okunur

Ölçülen şey tool çağrısı sayısı ve süre — ikisi de `on_event` akışından
sayılıyor, `Trace`'e ihtiyaç yok.

Kullanım:
    python scripts/demo_retry_maliyeti.py
"""

from __future__ import annotations

import time

from grounded_assistant.ptc.sandbox_runner import run_sandbox

# Pahalı blok: 12 tool çağrısı. Gerçekte bu bir kaynak sistem taraması olurdu.
PAHALI_BLOK = """
def pahali_tarama():
    import pandas as pd
    satirlar = []
    for i in range(4):
        durum = get_ticket_status(f"T-{i}")
        kisiler = search_employee_directory("BT")
        sayim = count_open_tickets()
        satirlar.append({
            "ticket": f"T-{i}",
            "acik": sayim["open_count"],
            "kisi_sayisi": len(kisiler),
        })
    return pd.DataFrame(satirlar)
"""

# 1. deneme: pahalı iş biter, SONRA hata. (`toplam` diye bir isim yok.)
HATALI_SON = """
set_result(int(df["acik"].sum()) + toplam)
"""

# 2. deneme: düzeltilmiş son satır.
DUZELTILMIS_SON = """
set_result(int(df["acik"].sum()))
"""

KALICILIKSIZ = "df = pahali_tarama()\n"
KALICILIKLA = 'df = cached("tarama.sonucu", pahali_tarama)\n'


def kosu(kod: str, workflow: str) -> tuple[int, float, str]:
    """Bir çalıştırma yapar; (tool çağrısı sayısı, süre, durum) döner."""
    sayac = {"n": 0}
    baslangic = time.monotonic()
    run = run_sandbox(
        kod,
        on_event=lambda e: sayac.__setitem__("n", sayac["n"] + 1)
        if e.get("stage") == "tool_call"
        else None,
        workflow_id=workflow,
        owner="altan",
    )
    return sayac["n"], time.monotonic() - baslangic, run.status.value


def senaryo(baslik: str, kurulum: str, workflow: str) -> tuple[int, float]:
    print(f"\n{baslik}")
    ilk = kosu(PAHALI_BLOK + kurulum + HATALI_SON, workflow)
    print(f"  1. deneme (hata)     : {ilk[0]:>2} tool çağrısı · {ilk[1]:5.2f} sn · {ilk[2]}")
    ikinci = kosu(PAHALI_BLOK + kurulum + DUZELTILMIS_SON, workflow)
    print(
        f"  2. deneme (düzeltme) : {ikinci[0]:>2} tool çağrısı · "
        f"{ikinci[1]:5.2f} sn · {ikinci[2]}"
    )
    return ilk[0] + ikinci[0], ilk[1] + ikinci[1]


def main() -> None:
    damga = int(time.time())
    a_cagri, a_sure = senaryo(
        "KALICILIKSIZ — pahalı blok her denemede yeniden koşuyor",
        KALICILIKSIZ,
        f"wf_kalicisiz_{damga}",
    )
    b_cagri, b_sure = senaryo(
        "KALICILIKLA — cached() ilk denemede saklıyor, düzeltmede okuyor",
        KALICILIKLA,
        f"wf_kalicili_{damga}",
    )

    print(f"\n{'':<14}{'tool çağrısı':>14}{'toplam süre':>14}")
    print("─" * 42)
    print(f"{'kalıcılıksız':<14}{a_cagri:>14}{a_sure:>13.2f}s")
    print(f"{'kalıcılıkla':<14}{b_cagri:>14}{b_sure:>13.2f}s")
    if a_cagri:
        print(f"{'fark':<14}{a_cagri - b_cagri:>13}↓{a_sure - b_sure:>13.2f}s")

    print(
        "\nSüre sütununu OLDUĞUNDAN FAZLA okumayın: buradaki tool'lar sahte ve\n"
        "anında dönüyor, toplam süreyi 4 pod başlatması (~3,9 sn/adet) belirliyor.\n"
        "Anlamlı sinyal TOOL ÇAĞRISI SAYISI — gerçek bir kaynak sisteme giden 12\n"
        "sorgunun 1'e inmesi. Zaman kazancı, o çağrıların gerçek gecikmesiyle\n"
        "orantılı olarak büyür; pod maliyetiyle değil."
    )


if __name__ == "__main__":
    main()
