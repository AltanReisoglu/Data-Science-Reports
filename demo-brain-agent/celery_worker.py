#!/usr/bin/env python3
"""
celery_worker.py — Board'daki TEK bir task'ı yürüten gerçek Celery worker.

Rol paylaşımı (önemli):
  • BOARD  : task'ların kendisi, bağımlılık (DAG) kapısı, durum, checkpoint  → BİZDE
  • CELERY : dağıtım + retry + at-least-once teslim                          → CELERY'DE

Çünkü Celery'de bağımlılık/DAG ve "kaldığı yerden devam" kavramları yoktur; o katmanı
board sağlar. Celery yalnız "şu task'ı bir worker'a dağıt ve gerekirse yeniden dene" der.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from celery import Celery

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

POC_DIR = Path(os.environ.get("BRAIN_CELERY_DIR", "/tmp/brain_orc_celery"))
QUEUE, PROC = POC_DIR / "queue", POC_DIR / "q_proc"
for d in (QUEUE, PROC):
    d.mkdir(parents=True, exist_ok=True)

app = Celery(
    "brain_board_worker",
    broker="filesystem://",
    broker_transport_options={
        "data_folder_in": str(QUEUE),
        "data_folder_out": str(QUEUE),   # filesystem transport: in == out ŞART
        "processed_folder": str(PROC),
        "store_processed": True,
    },
)
app.conf.update(
    task_acks_late=True,                 # iş bitince ack → worker çökerse yeniden teslim
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    result_backend=None,
)


@app.task(bind=True, max_retries=3, acks_late=True)
def run_task(self, task_id: str) -> str:
    """Board'dan task'ı al, işçi ajanla yürüt, sonucu board'a yaz."""
    from taskboard import TaskBoard
    import orchestrator

    board = TaskBoard(Path(os.environ["BRAIN_BOARD_DB"]))
    strategy = os.environ.get("BRAIN_STRATEGY", "hermes")
    budget = int(os.environ.get("BRAIN_BUDGET", "3000"))
    fail_at = os.environ.get("BRAIN_FAIL_AT") or None

    task = board.get(task_id)
    if not task or task["status"] == "done":
        return "skip"

    # kuyruktan çekildi → claim (CAS): başka worker aynı task'ı almasın
    claimed = board.claim_next(f"celery-{self.request.id[:8]}")
    if claimed is None:
        return "no-ready-task"
    # claim SONRASI kaydı kullan: claim_lock dolu olmalı, yoksa fencing devre dışı kalır
    # (önceden claim öncesi 'task' dict'i tutuluyordu → claim_lock=None → guard yok).
    task = claimed

    try:
        # ORTAK yönlendirici: fonksiyon düğümü → LLM'siz; ajan düğümü → LLM
        kind, result, _extra = orchestrator.run_one_task(
            task, board, strategy, budget,
            fail_at=fail_at, attempt=self.request.retries)
    except RuntimeError as e:
        board.fail(task["id"], str(e), claimer=task.get("claim_lock"))
        raise self.retry(exc=e, countdown=0)      # Celery retry: task BAŞTAN koşar
    if not board.complete(task["id"], result, claimer=task.get("claim_lock")):
        return "stale-write-reddedildi"
    return f"ok:{kind}"
