"""PTC sandbox çalıştırma maliyetinin aşama aşama ölçümü (PoC planı Faz 0).

LLM'i hiç devreye sokmadan `run_sandbox`'ı doğrudan çağırır — böylece ölçülen
şey model gecikmesi değil, YALNIZCA orkestrasyon yolu olur.

Kullanım:
    python scripts/measure_sandbox.py [tekrar_sayısı]
"""

from __future__ import annotations

import statistics
import sys
import time

from grounded_assistant.ptc.sandbox_runner import run_sandbox

# Tek bir tool çağrısı yapan en küçük anlamlı iş: "ilk tool_call" kilometre
# taşını verir, ama ölçümü kendi hesabıyla şişirmez.
SAMPLE_CODE = """
sonuc = count_open_tickets()
set_result(sonuc["open_count"])
"""

MILESTONES = ["configmap_created", "job_created", "pod_running", "tool_call", "final"]


def bir_kosu() -> dict[str, float]:
    """Bir çalıştırma yapar, aşama zamanlarını (saniye, başlangıca göre) döner."""
    t0 = time.monotonic()
    zamanlar: dict[str, float] = {}

    def on_event(event: dict) -> None:
        stage = event.get("stage")
        # tool_call birden fazla gelebilir; İLKİNİ tut
        if stage in MILESTONES and stage not in zamanlar:
            zamanlar[stage] = time.monotonic() - t0

    run = run_sandbox(SAMPLE_CODE, on_event=on_event)
    zamanlar["return"] = time.monotonic() - t0
    zamanlar["_status"] = run.status.value
    return zamanlar


def main() -> None:
    tekrar = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"{tekrar} çalıştırma ölçülüyor...\n")

    kosular: list[dict] = []
    for i in range(1, tekrar + 1):
        z = bir_kosu()
        kosular.append(z)
        print(f"  {i}. koşu: {z['return']:.2f} sn  (durum: {z['_status']})")

    basarili = [k for k in kosular if k["_status"] == "success"]
    if not basarili:
        print("\n⚠ Hiçbir koşu başarılı olmadı — aşağıdaki tablo yanıltıcı olabilir.")
        basarili = kosular

    print(f"\n{'aşama':<22}{'medyan':>9}{'min':>9}{'max':>9}{'Δ önceki':>11}")
    print("─" * 60)
    onceki = 0.0
    for stage in [*MILESTONES, "return"]:
        degerler = [k[stage] for k in basarili if stage in k]
        if not degerler:
            print(f"{stage:<22}{'— (hiç gelmedi)':>38}")
            continue
        med = statistics.median(degerler)
        print(
            f"{stage:<22}{med:>8.2f}s{min(degerler):>8.2f}s{max(degerler):>8.2f}s"
            f"{med - onceki:>10.2f}s"
        )
        onceki = med


if __name__ == "__main__":
    main()
