"""Artifact persistence'ın uçtan uca kanıtı — LLM olmadan.

İki AYRI PTC çalıştırması yapar. Birincinin pod'u silinir; ikincisi tamamen
yeni bir pod'da doğar ve birincinin ürettiği veriyi YENİDEN ÜRETMEDEN okur.

Gösterilen tez:
    Sandbox = hesaplama ve geçici alan
    Artifact store = durable çalışma ürünü deposu
    Artifact'in kalıcılığı sandbox'ın yaşam döngüsüne BAĞLI DEĞİL.

Kullanım:
    python scripts/demo_artifact_persistence.py
"""

from __future__ import annotations

import time

from grounded_assistant.ptc.sandbox_runner import run_sandbox

WORKFLOW = f"wf_demo_{int(time.time())}"

# --- PTC #1 — "extract" node'u: pahalı işi yapar, çıktısını bırakır ----------
PTC_1 = """
sayim = count_open_tickets()

import pandas as pd
df = pd.DataFrame([
    {"ticket": f"T-{i}", "departman": ["BT", "IK", "Finans"][i % 3]}
    for i in range(sayim["open_count"])
])

df.to_parquet("/output/extract.tickets.parquet")
set_result(f"{len(df)} satir uretildi")
"""

# --- PTC #2 — "transform" node'u: AYRI bir pod, veriyi geri okur -------------
# Dikkat: count_open_tickets() BURADA HİÇ ÇAĞRILMIYOR.
PTC_2 = """
import pandas as pd
df = pd.read_parquet("/output/extract.tickets.parquet")
ozet = df.groupby("departman").size().reset_index(name="adet")
ozet.to_parquet("/output/transform.ozet.parquet")
set_result(ozet.to_dict("records"))
"""

# --- PTC #3 — keşif: kendisinden önce ne üretildiğini bilmiyor ---------------
PTC_3 = """
import os
set_result(sorted(os.listdir("/output")))
"""


def kosu(baslik: str, kod: str, node: str | None) -> None:
    print(f"\n{'─' * 62}\n{baslik}\n{'─' * 62}")
    araclar: list[str] = []
    run = run_sandbox(
        kod,
        on_event=lambda e: araclar.append(e["tool_name"])
        if e.get("stage") == "tool_call"
        else None,
        workflow_id=WORKFLOW,
        owner="altan",
        node_id=node,
    )
    print(f"  pod       : {run.run_id}  (bu çalıştırmadan sonra silinir)")
    print(f"  tool'lar  : {araclar or '—'}")
    for olay in run.artifacts:
        boyut = f" · {olay.size_bytes} bayt" if olay.size_bytes else ""
        print(f"  artifact  : {olay.op.value:<9} {olay.name} ({olay.artifact_id}){boyut}")
    print(f"  durum     : {run.status.value}")
    print(f"  sonuç     : {run.result_text or run.error_message}")


def main() -> None:
    print(f"workflow: {WORKFLOW}")
    kosu("PTC #1 — extract: ticket'ları çek, artifact'e yaz", PTC_1, "extract")
    kosu("PTC #2 — transform: YENİ pod, veriyi yeniden üretmeden oku", PTC_2, "transform")
    kosu("PTC #3 — keşif: bu workflow'da ne üretilmiş?", PTC_3, None)

    print(f"\n{'─' * 62}")
    print("  PTC #2'de count_open_tickets ÇAĞRILMADI — veri artifact'ten geldi.")
    print("  Üç çalıştırmanın pod'u da silindi; artifact'ler duruyor.")


if __name__ == "__main__":
    main()
