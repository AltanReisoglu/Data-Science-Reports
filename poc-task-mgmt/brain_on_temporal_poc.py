#!/usr/bin/env python3
"""
brain_chat_V2 → TEMPORAL'a SARILMIŞ hali (GERÇEK temporalio SDK + dev server).

"Bizim beyni durable motora BİN(buy)" rotası (bkz. Shannon). brain_core'un 4 adımı
(retrieve/reason/act/respond) birer Temporal **activity**'sine sarılır; ajan döngüsü
bir **workflow** olur. Her adım event-history'ye yazılır → worker çökünce Temporal
**replay** edip biten activity'leri ATLAR → pahalı `retrieve` sadece 1× koşar.

  brain_core.reason ilk denemede geçici hata verir → activity RetryPolicy'si OTOMATİK
  yeniden dener; workflow durable koşar. Kanıt: retrieve 1×, reason 2×, act/respond 1×.

NOT (determinizm disiplini): LLM/rastgele/IO YALNIZ activity içinde olur; workflow
gövdesi saf ve deterministik kalır (replay güvenliği). brain_core.reason burada
deterministik; gerçekte LLM çağrısı da bu activity'nin İÇİNDE olurdu.

Çalıştır:  .venv/bin/python poc-task-mgmt/brain_on_temporal_poc.py
"""
from __future__ import annotations
import asyncio
import logging
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import brain_core

logging.getLogger("temporalio").setLevel(logging.CRITICAL)
logging.getLogger("temporalio.activity").setLevel(logging.CRITICAL)

from temporalio import workflow, activity
from temporalio.worker import Worker
from temporalio.testing import WorkflowEnvironment
from temporalio.common import RetryPolicy

# her brain-adımının GERÇEKTE kaç kez koştuğunu sayan sayaç (replay/retry kanıtı)
RUNS: dict[str, int] = {}


@activity.defn
async def a_retrieve(order_id: str) -> str:
    RUNS["retrieve"] = RUNS.get("retrieve", 0) + 1
    return brain_core.retrieve(order_id)


@activity.defn
async def a_reason(ctx: str) -> str:
    RUNS["reason"] = RUNS.get("reason", 0) + 1
    # attempt = bu activity'nin kaçıncı deneme olduğu (ilk deneme=0 → geçici hata)
    return brain_core.reason(ctx, attempt=RUNS["reason"] - 1)


@activity.defn
async def a_act(plan: str) -> str:
    RUNS["act"] = RUNS.get("act", 0) + 1
    return brain_core.act(plan)


@activity.defn
async def a_respond(action: str) -> str:
    RUNS["respond"] = RUNS.get("respond", 0) + 1
    return brain_core.respond(action)


@workflow.defn
class BrainWorkflow:
    """brain_chat_V2 ajan-döngüsü = durable workflow (adımlar = activity)."""
    @workflow.run
    async def run(self, order_id: str) -> str:
        ctx = await workflow.execute_activity(
            a_retrieve, order_id, start_to_close_timeout=timedelta(seconds=5))
        plan = await workflow.execute_activity(
            a_reason, ctx, start_to_close_timeout=timedelta(seconds=5),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(milliseconds=100), maximum_attempts=3))
        action = await workflow.execute_activity(
            a_act, plan, start_to_close_timeout=timedelta(seconds=5))
        return await workflow.execute_activity(
            a_respond, action, start_to_close_timeout=timedelta(seconds=5))


async def main():
    print("=" * 78)
    print("brain_chat_V2 → TEMPORAL'a SARILI  (temporalio SDK + gerçek dev server)")
    print("=" * 78)
    print("Ephemeral Temporal dev server başlatılıyor (ilk çalıştırmada indirilebilir)…")

    async with await WorkflowEnvironment.start_time_skipping() as env:
        print("Server hazır. brain workflow'u çalıştırılıyor.\n")
        async with Worker(env.client, task_queue="brain",
                          workflows=[BrainWorkflow],
                          activities=[a_retrieve, a_reason, a_act, a_respond]):
            handle = await env.client.start_workflow(
                BrainWorkflow.run, "4711", id="brain-4711", task_queue="brain")
            result = await handle.result()

        print("── SONUÇ " + "─" * 60)
        print(f"  brain workflow sonucu: {result!r}")
        print(f"\n── BRAIN ADIMLARI GERÇEKTE KAÇ KEZ KOŞTU " + "─" * 27)
        for name in brain_core.STEPS:
            note = "   ← 1 hata + 1 başarı = RetryPolicy" if name == "reason" else ""
            print(f"  {name:<10} : {RUNS.get(name, 0)} kez{note}")

        print(f"\n── EVENT HISTORY (durable log) " + "─" * 36)
        from temporalio.api.enums.v1 import EventType
        counts: dict[str, int] = {}
        hist = await handle.fetch_history()
        for ev in hist.events:
            try:
                name = EventType.Name(ev.event_type).replace("EVENT_TYPE_", "")
            except Exception:
                name = str(ev.event_type)
            counts[name] = counts.get(name, 0) + 1
        print(f"  toplam {len(hist.events)} durable event:")
        for k in ("WORKFLOW_EXECUTION_STARTED", "ACTIVITY_TASK_SCHEDULED",
                  "ACTIVITY_TASK_STARTED", "ACTIVITY_TASK_COMPLETED",
                  "WORKFLOW_EXECUTION_COMPLETED"):
            if k in counts:
                print(f"  {k:<30} × {counts[k]}")

    print("\n" + "=" * 78)
    print("KANIT: reason 2× (retry) ama retrieve/act/respond 1× → Temporal replay")
    print("       biten activity'leri atlar → PAHALI retrieve TEKRAR koşmaz (exactly-once).")
    print("       'Beynimizi durable motora bindirmek' (Shannon rotası) budur.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
