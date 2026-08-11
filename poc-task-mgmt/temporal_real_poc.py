#!/usr/bin/env python3
"""
GERÇEK Temporal ile task-management POC.

Simülasyon DEĞİL — gerçek `temporalio` SDK + gerçek (ephemeral) Temporal dev
server (WorkflowEnvironment). Aynı senaryo:

  Workflow OrderWorkflow: fetch_data → process → deliver
  'process' activity'si İLK denemede hata verir (geçici); Temporal RetryPolicy
  ile OTOMATİK yeniden dener, 2. denemede başarılı olur. Workflow durable koşar;
  biten activity'ler event-history'ye yazılır (exactly-once).

Kanıt olarak: her activity'nin GERÇEKTE kaç kez çalıştığı + workflow event-history
özeti (retry kaydı) yazdırılır.

Çalıştır:  .venv/bin/python poc-task-mgmt/temporal_real_poc.py
"""
from __future__ import annotations
import asyncio
import logging
import os
from datetime import timedelta

# ── AYARLAR (web arayüzünden env ile değiştirilebilir; varsayılanlar özgün senaryo)
FAIL_TIMES = int(os.environ.get("POC_FAIL_TIMES", "1"))     # process ilk kaç denemede patlasın
MAX_ATTEMPTS = int(os.environ.get("POC_MAX_RETRIES", "3"))  # activity'nin toplam deneme hakkı

# Temporal, ilk-deneme activity hatasını WARNING/traceback olarak loglar; bu BEKLENEN
# (retry tetikleyicisi). POC çıktısı temiz kalsın diye SDK loglarını kısıyoruz.
logging.getLogger("temporalio").setLevel(logging.CRITICAL)
logging.getLogger("temporalio.activity").setLevel(logging.CRITICAL)

from temporalio import workflow, activity
from temporalio.worker import Worker
from temporalio.testing import WorkflowEnvironment
from temporalio.common import RetryPolicy

# activity'lerin GERÇEKTE kaç kez koştuğunu sayan sayaç (retry kanıtı)
RUNS: dict[str, int] = {}


@activity.defn
async def fetch_data(order_id: str) -> str:
    RUNS["fetch_data"] = RUNS.get("fetch_data", 0) + 1
    activity.logger.info(f"fetch_data çalıştı (#{RUNS['fetch_data']})")
    return f"order:{order_id}:veri"


@activity.defn
async def process(data: str) -> str:
    RUNS["process"] = RUNS.get("process", 0) + 1
    n = RUNS["process"]
    activity.logger.info(f"process çalıştı (#{n})")
    if n <= FAIL_TIMES:
        # İlk FAIL_TIMES deneme: geçici hata → Temporal RetryPolicy devreye girer
        raise RuntimeError("geçici hata (ödeme ağ geçidi timeout)")
    return f"{data}|işlendi"


@activity.defn
async def deliver(processed: str) -> str:
    RUNS["deliver"] = RUNS.get("deliver", 0) + 1
    activity.logger.info(f"deliver çalıştı (#{RUNS['deliver']})")
    return f"{processed}|teslim"


@workflow.defn
class OrderWorkflow:
    @workflow.run
    async def run(self, order_id: str) -> str:
        # 1) fetch — tek deneme yeter
        a = await workflow.execute_activity(
            fetch_data, order_id, start_to_close_timeout=timedelta(seconds=5)
        )
        # 2) process — RetryPolicy ile (ilk hata otomatik retry edilir)
        b = await workflow.execute_activity(
            process, a,
            start_to_close_timeout=timedelta(seconds=5),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(milliseconds=100),
                maximum_attempts=MAX_ATTEMPTS,
            ),
        )
        # 3) deliver
        c = await workflow.execute_activity(
            deliver, b, start_to_close_timeout=timedelta(seconds=5)
        )
        return c


async def main():
    print("=" * 78)
    print("GERÇEK TEMPORAL ile TASK-MANAGEMENT POC  (temporalio SDK + gerçek dev server)")
    print("=" * 78)
    print(f"ayarlar: process ilk {FAIL_TIMES} denemede patlar · activity deneme hakkı = {MAX_ATTEMPTS}")
    print("Ephemeral Temporal dev server başlatılıyor (ilk çalıştırmada indirilebilir)…")

    async with await WorkflowEnvironment.start_time_skipping() as env:
        print("Server hazır. Worker + workflow çalıştırılıyor.\n")
        async with Worker(
            env.client,
            task_queue="orders",
            workflows=[OrderWorkflow],
            activities=[fetch_data, process, deliver],
        ):
            handle = await env.client.start_workflow(
                OrderWorkflow.run, "4711",
                id="order-4711", task_queue="orders",
            )
            try:
                result = await handle.result()
                failed = None
            except Exception as e:
                # deneme hakkı bitti → workflow başarısız. History YİNE DE durur (kanıt).
                result, failed = None, str(e).splitlines()[0][:120]

        print("── SONUÇ " + "─" * 60)
        if failed:
            print(f"  workflow BAŞARISIZ: {failed}")
            print(f"  → activity {MAX_ATTEMPTS} denemede de patladı; deneme hakkı bitti.")
        else:
            print(f"  workflow sonucu: {result!r}")
        print(f"\n── ACTIVITY'LER GERÇEKTE KAÇ KEZ KOŞTU " + "─" * 30)
        for name in ("fetch_data", "process", "deliver"):
            n = RUNS.get(name, 0)
            note = f"   ← {min(FAIL_TIMES, n)} hata → RetryPolicy devreye girdi" if name == "process" and n > 1 else ""
            print(f"  {name:<12} : {n} kez{note}")

        # event-history: retry gerçekten kalıcı loglanmış mı?
        print(f"\n── EVENT HISTORY (durable log — retry kaydı) " + "─" * 22)
        from temporalio.api.enums.v1 import EventType
        counts: dict[str, int] = {}
        hist = await handle.fetch_history()
        for ev in hist.events:
            try:
                name = EventType.Name(ev.event_type).replace("EVENT_TYPE_", "")
            except Exception:
                name = str(ev.event_type)
            counts[name] = counts.get(name, 0) + 1
        print(f"  toplam {len(hist.events)} durable event kaydedildi:")
        for k in ("WORKFLOW_EXECUTION_STARTED", "ACTIVITY_TASK_SCHEDULED",
                  "ACTIVITY_TASK_STARTED", "ACTIVITY_TASK_COMPLETED",
                  "ACTIVITY_TASK_FAILED", "WORKFLOW_EXECUTION_COMPLETED"):
            if k in counts:
                print(f"  {k:<30} × {counts[k]}")
        print(f"  → Her activity: SCHEDULED→STARTED→COMPLETED. process'in başarısız")
        print(f"    denemeleri otomatik retry'landığı için history'ye YAZILMAZ (kompakt kalır);")
        print(f"    retry kanıtı yukarıdaki RUNS sayacı. Attempt no, STARTED event'inde.")

        # ── web arayüzü için makine-okunur akış özeti
        f, p, d = (RUNS.get("fetch_data", 0), RUNS.get("process", 0), RUNS.get("deliver", 0))
        print("##FLOW## " + "|".join([
            f"fetch:{f}:ok",
            f"process:{p}:" + ("fail" if failed else ("retry" if p > 1 else "ok")),
            f"deliver:{d}:" + ("skip" if not d else "ok"),
        ]))
        if failed:
            print(f"##VERDICT## deneme hakkı ({MAX_ATTEMPTS}) bitti — ama fetch yine {f}× koştu; "
                  f"history duruyor, iş kaldığı yerden sürdürülebilir")
        else:
            print(f"##VERDICT## process {p}× denendi ama pahalı fetch sadece {f}× koştu — "
                  f"tamamlanan iş KORUNDU (replay atladı)")

    print("\n" + "=" * 78)
    print(f"KANIT: process {RUNS.get('process',0)} kez koştu (hata→otomatik retry) ama fetch "
          f"{RUNS.get('fetch_data',0)} kez;")
    print("       tamamlanan activity replay'de ATLANIR — exactly-once activity completion.")
    print("       Hepsi GERÇEK Temporal (temporalio SDK + gerçek server), simülasyon değil.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
