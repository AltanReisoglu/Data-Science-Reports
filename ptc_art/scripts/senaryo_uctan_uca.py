"""Uçtan uca vaka: "Açık ticket'ları departmana göre raporla".

Beş sahne, tek anlatı. Her sahne mimarinin bir iddiasını canlı gösterir:

  1. Pahalı tarama + hata            → iş boşa gitmez, artifact'e yazıldı
  2. Self-repair                     → /output'taki çıktı pahalı bloğu ATLATIR (A)
  3. Sonraki tur, YENİ pod           → artifact'ten okur, tool çağırmaz   (B/C)
  4. Başka workflow okumaya çalışır  → REDDEDİLİR
  5. Kod pickle saklamaya çalışır    → REDDEDİLİR

Kullanım:
    python scripts/senaryo_uctan_uca.py
"""

from __future__ import annotations

import time

from grounded_assistant.ptc.sandbox_runner import run_sandbox

WF = f"wf_rapor_{int(time.time())}"
BASKA_WF = f"wf_baska_{int(time.time())}"

# Pahalı tarama: 12 tool çağrısı. Gerçekte bir kaynak sistem taraması olurdu.
TARAMA = """
def ticket_taramasi():
    import pandas as pd
    satirlar = []
    for i in range(4):
        get_ticket_status(f"T-{i}")
        kisiler = search_employee_directory("BT")
        sayim = count_open_tickets()
        satirlar.append({
            "ticket": f"T-{i}",
            "departman": ["BT", "IK", "Finans", "BT"][i],
            "acik": sayim["open_count"],
            "ilgili": len(kisiler),
        })
    return pd.DataFrame(satirlar)
"""


def kosu(baslik: str, kod: str, workflow: str = WF, node: str | None = None) -> None:
    araclar: list[str] = []
    olaylar: list = []
    t0 = time.monotonic()

    def on_event(e: dict) -> None:
        if e.get("stage") == "tool_call":
            araclar.append(e["tool_name"])

    run = run_sandbox(kod, on_event=on_event, workflow_id=workflow, owner="altan", node_id=node)
    olaylar = run.artifacts
    sure = time.monotonic() - t0

    print(f"\n{'━' * 66}\n{baslik}\n{'━' * 66}")
    print(f"  pod        {run.run_id}   workflow: {workflow}")
    print(f"  tool       {len(araclar)} çağrı  {araclar if len(araclar) <= 4 else ''}")
    for o in olaylar:
        boyut = f" · {o.size_bytes} bayt" if o.size_bytes else ""
        print(f"  artifact   {o.op.value:<9} {o.name} ({o.artifact_id}){boyut}")
    print(f"  süre       {sure:.2f} sn")
    print(f"  durum      {run.status.value}")
    print(f"  sonuç      {run.result_text or run.error_message}")


def main() -> None:
    print(f"VAKA: 'Açık ticket'ları departmana göre raporla'   workflow={WF}")

    # 1 — LLM ilk denemesini yazar: tarama biter, son satırda hata.
    kosu(
        "1 · İLK DENEME — tarama biter, sonra NameError",
        TARAMA
        + 'import os, pandas as pd\n'
        'if os.path.exists("/output/rapor.tarama.parquet"):\n'
        '    df = pd.read_parquet("/output/rapor.tarama.parquet")\n'
        'else:\n'
        '    df = ticket_taramasi()\n'
        '    df.to_parquet("/output/rapor.tarama.parquet")\n'
        + 'set_result(df.groupby("departman")["acik"].sum().to_dict() + toplam)\n',
        node="extract",
    )

    # 2 — Self-repair: aynı turda düzeltilmiş kod. cached() taramayı atlar.
    kosu(
        "2 · SELF-REPAIR — aynı tur, düzeltilmiş kod",
        TARAMA
        + 'import os, pandas as pd\n'
        'if os.path.exists("/output/rapor.tarama.parquet"):\n'
        '    df = pd.read_parquet("/output/rapor.tarama.parquet")\n'
        'else:\n'
        '    df = ticket_taramasi()\n'
        '    df.to_parquet("/output/rapor.tarama.parquet")\n'
        + 'ozet = df.groupby("departman")["acik"].sum().reset_index()\n'
        + 'ozet.to_parquet("/output/rapor.ozet.parquet")\n'
        + 'set_result(ozet.to_dict("records"))\n',
        node="extract",
    )

    # 3 — Sonraki tur: yeni pod, hiç tool çağırmadan artifact'ten okur.
    kosu(
        "3 · SONRAKİ TUR — yeni pod, kaynak sisteme HİÇ gitmez",
        'import pandas as pd\n'
        'ozet = pd.read_parquet("/output/rapor.ozet.parquet")\n'
        'bt = ozet[ozet["departman"] == "BT"]["acik"].iloc[0]\n'
        'set_result(f"BT departmanı: {bt} açık ticket")\n',
        node="report",
    )

    # 4 — 2026-09-06: sınır workflow'dan TENANT'a taşındı. Başka bir workflow
    #     ARTIK OKUYABİLİR — KFP'de de öyle (`pipeline_root` paylaşımlı).
    kosu(
        "4 · ÇALIŞTIRMALAR ARASI — başka workflow aynı artifact'i okuyor",
        'import os, pandas as pd\n'
        'var = os.path.exists("/output/rapor.ozet.parquet")\n'
        'v = pd.read_parquet("/output/rapor.ozet.parquet") if var else None\n'
        'set_result("okundu: " + str(v.to_dict("records")) if var else "bulunamadi")\n',
        workflow=BASKA_WF,
    )

    # 5 — Sınır: kod pickle saklamaya çalışır.
    kosu(
        "5 · SINIR — kod pickle saklamaya çalışıyor",
        "import pickle\n"
        'open("/output/zararsiz.gorunen.txt","wb").write(pickle.dumps({"kotu":"yuk"}))\n'
        'set_result("dosya yazildi — supurmede reddedilmeli")\n',
    )


if __name__ == "__main__":
    main()
