#!/usr/bin/env python3
"""
temporal_defs.py — Board dispatch'i için Temporal workflow/activity tanımları.

Temporal, @workflow.defn sınıflarının MODÜL SEVİYESİNDE tanımlanmasını şart koşar,
o yüzden ayrı dosyada duruyorlar.

Rol paylaşımı:
  • BOARD    : task'lar, DAG kapısı, durum, checkpoint          → BİZDE
  • TEMPORAL : durable yürütme, retry, replay ile çökme kurtarma → TEMPORAL'DA

Workflow (BoardWorkflow) saf/deterministik kalır: sadece activity çağırır.
Yan etkili her şey (LLM, board yazma) activity'nin İÇİNDEDİR — determinizm disiplini.
"""
from __future__ import annotations

import json
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

# süreç-içi bağlam (worker aynı süreçte koştuğu için activity buradan okur)
CTX: dict = {"board": None, "strategy": "hermes", "budget": 3000,
             "fail_at": None, "compaction_events": []}


@activity.defn(name="next_ready_task")
async def next_ready_task(_: str) -> str:
    """Bağımlılık kapısını aç, sıradaki hazır task'ı CAS ile kap."""
    board = CTX["board"]
    promoted = board.recompute_ready()
    if board.all_settled():
        return json.dumps({"task": None, "settled": True, "promoted": promoted})
    t = board.claim_next(f"temporal-{activity.info().workflow_run_id[:8]}")
    return json.dumps({"task": t, "settled": False, "promoted": promoted},
                      ensure_ascii=False, default=str)


@activity.defn(name="execute_one_task")
async def execute_one_task(payload: str) -> str:
    """Tek bir task'ı işçi ajanla yürüt ve board'a yaz.

    Hata fırlatırsa Temporal RetryPolicy devreye girer; workflow çökse bile
    tamamlanmış activity'ler replay'de ATLANIR (exactly-once).
    """
    import orchestrator
    task = json.loads(payload)["task"]
    board = CTX["board"]

    # ── BAYAT CLAIM ONARIMI ──
    # Bir önceki denemede board.fail() claim'i temizledi ve task'ı 'ready'ye çevirdi.
    # Temporal ise AYNI activity'yi AYNI payload ile tekrar çağırır → elimizdeki
    # claim_lock artık geçersiz. Onarmazsak iş başarıyla koşar ama complete()
    # fencing'e takılır, sonuç ÇÖPE gider ve task sonsuza dek 'ready' kalır
    # (ölçüldü: iki kez başaran düğüm 3. turda 'failed' işaretleniyordu).
    guncel = board.get(task["id"])
    if guncel and guncel["status"] == "ready":
        yeniden = board.claim(task["id"], f"temporal-{activity.info().workflow_run_id[:8]}")
        if yeniden:
            task = yeniden
    elif guncel and guncel["claim_lock"]:
        task = guncel                       # claim hâlâ bizde: güncel kaydı kullan

    # Deneme sayacı BOARD'dan gelmeli: Temporal her yeni activity çağrısında kendi
    # sayacını 1'e sıfırlar → geçici hata sonsuza dek 'ilk deneme' sanılır ve
    # geçici hata KALICIYA dönüşürdü.
    att = max(activity.info().attempt - 1, int(task.get("attempt") or 0))
    try:
        # ORTAK yönlendirici: kind='function' → LLM'siz fonksiyon, 'agent' → LLM ajan
        kind, result, extra = orchestrator.run_one_task(
            task, board, CTX["strategy"], CTX["budget"],
            fail_at=CTX["fail_at"], attempt=att)
    except Exception as e:
        board.fail(task["id"], str(e), claimer=task.get("claim_lock"))
        raise
    if kind == "agent":
        CTX["compaction_events"] += (extra or {}).get("compaction_events", [])
    CTX.setdefault("kinds", []).append(kind)
    board.complete(task["id"], result, claimer=task.get("claim_lock"))
    return json.dumps({"id": task["id"], "title": task["title"], "kind": kind},
                      ensure_ascii=False)


@workflow.defn(name="BoardWorkflow", sandboxed=False)
class BoardWorkflow:
    """Dispatch döngüsü = durable workflow. Gövde saf; iş activity'lerde."""

    @workflow.run
    async def run(self, _: str) -> str:
        log = []
        for rnd in range(1, 16):
            got = await workflow.execute_activity(
                "next_ready_task", "go",
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=3))
            d = json.loads(got)
            if d.get("promoted"):
                log.append(f"[tur {rnd}] recompute_ready → {d['promoted']} task "
                           f"blocked→ready (bağımlılık kapısı)")
            if d.get("settled"):
                break
            task = d.get("task")
            if not task:
                break
            log.append(f"[tur {rnd}] claim: {task['id']} '{task['title'][:44]}' (CAS)")
            try:
                await workflow.execute_activity(
                    "execute_one_task",
                    json.dumps({"task": task}, ensure_ascii=False, default=str),
                    start_to_close_timeout=timedelta(seconds=240),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(milliseconds=200),
                        maximum_attempts=3))          # activity-seviyesi retry
                log.append(f"          ✓ tamamlandı → done")
            except Exception as e:
                log.append(f"          ✗ activity kalıcı hata: {str(e)[:70]}")
        return json.dumps({"log": log}, ensure_ascii=False)
