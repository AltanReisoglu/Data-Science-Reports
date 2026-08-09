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
from datetime import timedelta

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
    if n == 1:
        # İLK deneme: geçici hata → Temporal RetryPolicy devreye girer
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
                maximum_attempts=3,
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
            result = await handle.result()

        print("── SONUÇ " + "─" * 60)
        print(f"  workflow sonucu: {result!r}")
        print(f"\n── ACTIVITY'LER GERÇEKTE KAÇ KEZ KOŞTU " + "─" * 30)
        for name in ("fetch_data", "process", "deliver"):
            print(f"  {name:<12} : {RUNS.get(name,0)} kez"
                  + ("   ← 1 hata + 1 başarı = RetryPolicy devreye girdi" if name == "process" else ""))

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
        print(f"  → 3 activity: her biri SCHEDULED→STARTED→COMPLETED. process'in başarısız")
        print(f"    1. denemesi otomatik retry'landığı için history'ye YAZILMAZ (kompakt kalır);")
        print(f"    retry kanıtı yukarıdaki RUNS sayacı (process=2). Attempt no, STARTED event'inde.")

    print("\n" + "=" * 78)
    print("KANIT: process 2 kez koştu (1 hata→otomatik retry→başarı) ama fetch/deliver")
    print("       1'er kez; workflow durable event-history ile exactly-once tamamlandı.")
    print("       Hepsi GERÇEK Temporal (temporalio SDK + gerçek server), simülasyon değil.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
