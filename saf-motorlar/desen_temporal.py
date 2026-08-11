#!/usr/bin/env python3
"""
desen_temporal.py — Üç deseni Temporal'da kur.

TEMPORAL'IN WORKFLOW'A BAKIŞI: workflow = KOD.
Graf ayrı bir yapı değil; `for`, `if`, `await`, `asyncio.gather` normal Python.
Bu yüzden koşullu dal ve dinamik fan-out EK BİR KAVRAM GEREKTİRMİYOR —
dilin kendi kontrol akışı yetiyor.

Bedeli: determinizm. Workflow gövdesi replay'de aynı sonucu vermeli, bu yüzden
IO/rastgelelik/saat okuma yasak — hepsi activity'ye taşınmak zorunda.

    .venv/bin/python saf-motorlar/desen_temporal.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import desenler as D  # noqa: E402

logging.getLogger("temporalio").setLevel(logging.CRITICAL)
from temporalio import activity, workflow  # noqa: E402
from temporalio.common import RetryPolicy  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

RP = RetryPolicy(initial_interval=timedelta(milliseconds=50), maximum_attempts=3)
TO = timedelta(seconds=30)


# ───────── activity'ler ─────────
@activity.defn(name="zincir_adim")
async def a_zincir(arg: dict) -> dict:
    return D.zincir_adim(arg["n"], arg.get("onceki"))


@activity.defn(name="tarama")
async def a_tarama() -> dict:
    return D.tarama()


@activity.defn(name="dal_raporla")
async def a_raporla(t: dict) -> dict:
    return D.dal_raporla(t)


@activity.defn(name="dal_atla")
async def a_atla(t: dict) -> dict:
    return D.dal_atla(t)


@activity.defn(name="dosyalari_bul")
async def a_bul() -> dict:
    return D.dosyalari_bul()


@activity.defn(name="dosya_isle")
async def a_isle(path: str) -> dict:
    return D.dosya_isle(path)


@activity.defn(name="topla")
async def a_topla(sonuclar: list) -> dict:
    return D.topla(sonuclar)


# ───────── workflow'lar: graf = düz Python ─────────

@workflow.defn(name="D2Zincir", sandboxed=False)
class D2Zincir:
    @workflow.run
    async def run(self, _: str) -> dict:
        s = None
        for i in range(1, D.ZINCIR_UZUNLUK + 1):        # ← düz for döngüsü
            s = await workflow.execute_activity(
                "zincir_adim", {"n": i, "onceki": s},
                start_to_close_timeout=TO, retry_policy=RP)
        return s


@workflow.defn(name="D3Kosullu", sandboxed=False)
class D3Kosullu:
    @workflow.run
    async def run(self, _: str) -> dict:
        t = await workflow.execute_activity("tarama", start_to_close_timeout=TO,
                                            retry_policy=RP)
        # ← düz if. Seçilmeyen dal HİÇ ÇAĞRILMIYOR — "atlandı" kaydı da yok,
        #   çünkü ortada bir düğüm yok. Graf zaten çalışırken oluşuyor.
        if t["asti"]:
            return await workflow.execute_activity("dal_raporla", t,
                                                   start_to_close_timeout=TO,
                                                   retry_policy=RP)
        return await workflow.execute_activity("dal_atla", t,
                                               start_to_close_timeout=TO,
                                               retry_policy=RP)


@workflow.defn(name="D4Dinamik", sandboxed=False)
class D4Dinamik:
    @workflow.run
    async def run(self, _: str) -> dict:
        f = await workflow.execute_activity("dosyalari_bul",
                                            start_to_close_timeout=TO, retry_policy=RP)
        # ← N çalışma anında belli; gather ile paralel fan-out. Ek kavram yok.
        sonuclar = await asyncio.gather(*[
            workflow.execute_activity("dosya_isle", p, start_to_close_timeout=TO,
                                      retry_policy=RP)
            for p in f["dosyalar"]])
        return await workflow.execute_activity("topla", list(sonuclar),
                                               start_to_close_timeout=TO,
                                               retry_policy=RP)


async def _kos(wf, wid):
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(env.client, task_queue="desen",
                          workflows=[D2Zincir, D3Kosullu, D4Dinamik],
                          activities=[a_zincir, a_tarama, a_raporla, a_atla,
                                      a_bul, a_isle, a_topla]):
            return await env.client.execute_workflow(
                wf.run, "go", id=wid, task_queue="desen",
                retry_policy=RetryPolicy(maximum_attempts=1))


def main():
    print("═" * 78)
    print("TEMPORAL — workflow = KOD (for / if / gather normal Python)")
    print("═" * 78)
    for ad, wf, wid, dog in (
            ("D2 zincir ", D2Zincir, "d2", D.zincir_dogrula),
            ("D3 koşullu", D3Kosullu, "d3", D.kosullu_dogrula),
            ("D4 dinamik", D4Dinamik, "d4", D.dinamik_dogrula)):
        t0 = time.time()
        try:
            r = asyncio.run(_kos(wf, wid))
            ok, not_ = dog(r)
        except Exception as e:
            ok, not_ = False, f"{type(e).__name__}: {str(e)[:90]}"
        print(f"  {ad}  {'✓' if ok else '✗'} {not_[:60]:<62} {round(time.time()-t0,2)}s")


if __name__ == "__main__":
    main()
