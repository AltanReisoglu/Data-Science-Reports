#!/usr/bin/env python3
"""
saf_temporal.py — Temporal'ın SAF hâli: workflow + activity, board YOK.

Temporal'da graf, workflow gövdesinde NORMAL PYTHON AKIŞI olarak yazılır.
Ayrı bir DAG yapısı yoktur — `await` sırası bağımlılıktır, `asyncio.gather`
paralelliktir. Bu, dört motor içinde grafın en okunur ifade edildiği yer.

Veri akışı: activity dönüşleri workflow gövdesinde YEREL DEĞİŞKEN.
Ne XCom ne board gerekiyor — dilin kendi değişkenleri yetiyor.

NE YAPAR : durable yürütme · replay (biten activity atlanır) · RetryPolicy
           (backoff'lu) · event history · timeout
NE YAPMAZ: task kavramı · "board" görünürlüğü · çalışma anında graf değişimi
           (workflow kodu sabittir)

    .venv/bin/python saf-motorlar/saf_temporal.py [--cokme]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import isakisi as J  # noqa: E402

logging.getLogger("temporalio").setLevel(logging.CRITICAL)

from temporalio import activity, workflow  # noqa: E402
from temporalio.client import Client  # noqa: E402
from temporalio.common import RetryPolicy  # noqa: E402
from temporalio.worker import Worker  # noqa: E402


# ───────────────── activity'ler: gerçek iş burada ─────────────────
# IO ve rastgelelik YALNIZ burada olabilir (determinizm disiplini).

@activity.defn(name="cek")
async def a_cek() -> dict:
    return J.cek()


@activity.defn(name="test")
async def a_test() -> dict:
    return J.test()


@activity.defn(name="tara")
async def a_tara(cek: dict) -> dict:
    return J.tara(cek)


@activity.defn(name="esle")
async def a_esle(arg: dict) -> dict:
    return J.esle(arg["tara"], arg["test"])


@activity.defn(name="rapor")
async def a_rapor(arg: dict) -> dict:
    return J.rapor(arg["esle"], arg["tara"], arg["test"])


# ───────────────── workflow: GRAF, düz Python akışı ─────────────────

RP = RetryPolicy(initial_interval=timedelta(milliseconds=200), maximum_attempts=3)
TO = timedelta(seconds=60)


@workflow.defn(name="SafDenetim", sandboxed=False)
class SafDenetim:
    """Graf burada. `gather` = paralel dal, `await` sırası = bağımlılık."""

    @workflow.run
    async def run(self, _: str) -> dict:
        # cek→tara zinciri ile test dalı PARALEL
        async def dal_kod():
            cek = await workflow.execute_activity(
                "cek", start_to_close_timeout=TO, retry_policy=RP)
            return await workflow.execute_activity(
                "tara", cek, start_to_close_timeout=TO, retry_policy=RP)

        tara, test = await asyncio.gather(
            dal_kod(),
            workflow.execute_activity("test", start_to_close_timeout=TO,
                                      retry_policy=RP))

        esle = await workflow.execute_activity(
            "esle", {"tara": tara, "test": test},
            start_to_close_timeout=TO, retry_policy=RP)

        return await workflow.execute_activity(
            "rapor", {"esle": esle, "tara": tara, "test": test},
            start_to_close_timeout=TO, retry_policy=RP)


async def _kos() -> dict:
    from temporalio.testing import WorkflowEnvironment
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(env.client, task_queue="saf",
                          workflows=[SafDenetim],
                          activities=[a_cek, a_test, a_tara, a_esle, a_rapor]):
            return await env.client.execute_workflow(
                SafDenetim.run, "go", id="saf-denetim",
                task_queue="saf",
                retry_policy=RetryPolicy(maximum_attempts=1))


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    print("═" * 74)
    print("SAF TEMPORAL — workflow + activity, board YOK")
    print("  graf = workflow gövdesindeki düz Python akışı (gather + await)")
    print("═" * 74)
    import time
    t0 = time.time()
    try:
        son = asyncio.run(_kos())
    except Exception as e:
        msg = str(e)
        print(f"  ✗ workflow hatayla bitti: {type(e).__name__}")
        print(f"    {msg[:200]}")
        print("\n  → Temporal RetryPolicy tükendi. Kalan activity'ler HİÇ çağrılmadı")
        print("    (gövde exception ile kesildi). 'İptal edildi' diye bir kayıt yok;")
        print("    event history'de sadece 'çağrılmamış' olarak görünür.")
        return
    ok, not_ = J.dogrula({"rapor": son})
    print(f"\n  {'✓' if ok else '✗'} {not_} · {round(time.time()-t0, 2)} sn")


if __name__ == "__main__":
    main()
