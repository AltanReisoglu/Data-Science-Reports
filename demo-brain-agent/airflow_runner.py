#!/usr/bin/env python3
"""
airflow_runner.py — Bir board grafını GERÇEKTEN Airflow'da koşturur.

Diğer üç motor bizim süreçte (ya da bizim başlattığımız worker'da) koşuyor.
Airflow koşamaz: ağır global durum, kendi metadata DB'si, kendi CLI'si. Bu yüzden
tek doğru yol **subprocess**:

    board → DAG dosyası üret → `airflow dags test <dag_id>` → sonucu airflow.db'den oku

Ölçülmüş kısıtlar (bu oturumda doğrulandı):
  • `SequentialExecutor` + sqlite → İKİ koşu aynı anda "database is locked" verir.
    Bu yüzden modül düzeyinde bir KİLİT var; Airflow koşuları serileştirilir.
  • Tarih argümanı VERİLMEZ. Verilmezse Airflow `utcnow()` kullanır ve her tetikleme
    benzersiz `run_id` alır. Sabit tarih verilirse aynı run_id çakışır.
  • CLI açılışı ~1,5-2 sn; hatasız DAG toplam ~2-3 sn.
  • DAG başarısızsa CLI sıfırdan farklı exit code döner.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AIRFLOW_HOME = Path(os.environ.get(
    "BRAIN_AIRFLOW_HOME", HERE.parent / "saf-motorlar" / "airflow_home"))
DAGS = AIRFLOW_HOME / "dags"
DB = AIRFLOW_HOME / "airflow.db"
AIRFLOW_BIN = Path(sys.executable).parent / "airflow"

# Airflow sqlite + SequentialExecutor → eşzamanlı koşu YOK. Serileştir.
_KILIT = threading.Lock()

# Panelden üretilen DAG'lar birikmesin
MAX_URETILEN_DAG = 12


def hazir() -> tuple[bool, str]:
    """Airflow koşturulabilir mi? (kurulu + db var mı)"""
    if not AIRFLOW_BIN.exists():
        return False, f"airflow CLI bulunamadı: {AIRFLOW_BIN}"
    if not DB.exists():
        return False, (f"airflow.db yok: {DB}\n"
                       f"kurulum: AIRFLOW_HOME={AIRFLOW_HOME} "
                       f"{AIRFLOW_BIN} db migrate")
    return True, "hazır"


def _env() -> dict:
    return {**os.environ,
            "AIRFLOW_HOME": str(AIRFLOW_HOME),
            "AIRFLOW__CORE__LOAD_EXAMPLES": "False",
            # düğüm fonksiyonları demo-brain-agent'tan geliyor
            "PYTHONPATH": f"{HERE}:{os.environ.get('PYTHONPATH','')}"}


def _budama():
    """Panelden üretilen eski DAG dosyalarını temizle (dags/ şişmesin)."""
    uretilen = sorted(DAGS.glob("wf_*.py"), key=lambda p: p.stat().st_mtime,
                      reverse=True)
    for p in uretilen[MAX_URETILEN_DAG:]:
        p.unlink(missing_ok=True)


def _db_oku(dag_id: str, run_id: str | None = None) -> dict:
    """Sonuçları airflow.db'den READ-ONLY oku (yazan tek süreç CLI'dir)."""
    if not DB.exists():
        return {}
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    try:
        if run_id is None:
            r = con.execute("SELECT run_id, state FROM dag_run WHERE dag_id=? "
                            "ORDER BY id DESC LIMIT 1", (dag_id,)).fetchone()
            if not r:
                return {}
            run_id, durum = r["run_id"], r["state"]
        else:
            r = con.execute("SELECT state FROM dag_run WHERE dag_id=? AND run_id=?",
                            (dag_id, run_id)).fetchone()
            durum = r["state"] if r else None
        dugumler = [dict(x) for x in con.execute(
            "SELECT task_id, state, try_number, map_index, duration "
            "FROM task_instance WHERE dag_id=? AND run_id=? "
            "ORDER BY task_id, map_index", (dag_id, run_id)).fetchall()]
        xcom = {}
        for x in con.execute("SELECT task_id, value FROM xcom WHERE dag_id=? AND run_id=?",
                             (dag_id, run_id)).fetchall():
            try:
                xcom[x["task_id"]] = json.loads(x["value"])
            except Exception:
                xcom[x["task_id"]] = None
        return {"run_id": run_id, "durum": durum, "dugumler": dugumler, "xcom": xcom}
    finally:
        con.close()


def kostur(board, node_sim: dict | None = None, goal: str = "",
           on_event=None, timeout: int = 300) -> dict:
    """Board grafını Airflow'da koştur ve düğüm bazlı sonucu döndür.

    Dönüş: {ok, durum, dag_id, dag_file, dugumler:[...], sayim:{...}, sn, hata}
    """
    import orchestrator as O

    def yay(tip, **kw):
        if on_event:
            on_event({"type": tip, **kw})

    ok, mesaj = hazir()
    if not ok:
        yay("log", text=f"✗ Airflow koşturulamıyor: {mesaj}")
        return {"ok": False, "hata": mesaj, "dugumler": [], "sayim": {}, "sn": 0.0}

    dag_id = f"wf_{int(time.time() * 1000) % 100000000:08d}"
    dag_file = DAGS / f"{dag_id}.py"
    DAGS.mkdir(parents=True, exist_ok=True)

    # DAG'ı üret — kısa retry_delay (30 sn varsayılanı paneli kullanılamaz kılıyor),
    # zamanlama yok (elle tetikleyeceğiz).
    O.export_airflow_dag(board, out_path=dag_file, dag_id=dag_id, goal=goal,
                         node_sim=node_sim or {}, retry_delay_sn=1, schedule=None)
    yay("log", text=f"[airflow] DAG üretildi: {dag_file.name} "
                    f"({len(board.list_tasks())} düğüm, retry_delay=1s)")

    t0 = time.time()
    with _KILIT:                       # sqlite tek yazar — koşuları serileştir
        yay("log", text=f"[airflow] `airflow dags test {dag_id}` başlıyor "
                        f"(CLI açılışı ~2 sn)…")
        try:
            p = subprocess.run(
                [str(AIRFLOW_BIN), "dags", "test", dag_id],   # ← TARİH VERME
                cwd=str(HERE), env=_env(), capture_output=True, text=True,
                timeout=timeout)
            cikis, ciktilar = p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            yay("log", text=f"[airflow] ⏱ {timeout} sn'de bitmedi")
            return {"ok": False, "hata": "timeout", "dag_id": dag_id,
                    "dag_file": str(dag_file), "dugumler": [], "sayim": {},
                    "sn": round(time.time() - t0, 2)}
    sn = round(time.time() - t0, 2)

    d = _db_oku(dag_id)
    dugumler = d.get("dugumler", [])
    sayim: dict[str, int] = {}
    for x in dugumler:
        sayim[x["state"] or "?"] = sayim.get(x["state"] or "?", 0) + 1

    for x in dugumler:
        yay("log", text=f"[airflow] {x['task_id']:<12} {str(x['state']):<16} "
                        f"deneme={x['try_number']}")
    yay("log", text=f"[airflow] DagRun={d.get('durum')} · {sayim} · {sn} sn "
                    f"· exit={cikis}")

    return {"ok": d.get("durum") == "success" and cikis == 0,
            "durum": d.get("durum"), "dag_id": dag_id, "dag_file": str(dag_file),
            "run_id": d.get("run_id"), "dugumler": dugumler, "xcom": d.get("xcom", {}),
            "sayim": sayim, "sn": sn, "exit": cikis,
            "hata": "" if cikis == 0 else ciktilar[-400:]}


if __name__ == "__main__":
    import argparse
    from taskboard import TaskBoard
    ap = argparse.ArgumentParser(description="Board grafını Airflow'da koştur")
    ap.add_argument("--sim", default="", help='JSON: {"t_x":{"mod":"kalici"}}')
    a = ap.parse_args()

    print(f"AIRFLOW_HOME = {AIRFLOW_HOME}")
    print(f"hazır mı     : {hazir()}")
    b = TaskBoard()
    x = b.create_task("çek", kind="function", fn="fetch_source",
                      fn_args={"path": "auth/login.py"})
    t = b.create_task("test", kind="function", fn="run_test_suite", fn_args={"suite": "auth"})
    s = b.create_task("tara", kind="function", fn="scan_patterns",
                      fn_args={"pattern": "mfa_token"}, parents=[x])
    c = b.create_task("eşleştir", kind="function", fn="cross_check", parents=[s, t])
    b.create_task("rapor", kind="function", fn="render_report",
                  fn_args={"title": "Deneme"}, parents=[c, s, t])
    if a.sim == "auto":
        sim = {s: {"mod": "kalici"}}       # 'tara' düğümünü kalıcı patlat
    else:
        sim = json.loads(a.sim) if a.sim else {}
    r = kostur(b, node_sim=sim, goal="deneme", on_event=lambda e: print("  ", e.get("text", e)))
    print(f"\nsonuç: ok={r['ok']} durum={r.get('durum')} sayım={r['sayim']} {r['sn']} sn")
    _budama()
