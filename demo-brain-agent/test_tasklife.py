#!/usr/bin/env python3
"""
test_tasklife.py — Task yaşam döngüsünün SOHBETTEN ULAŞILAMAYAN yolları.

1. turda test edilemeyen (ya da hiç test edilmemiş) yollar burada doğrudan
board/orchestrator seviyesinde sınanır:

  1) RETRY          — geçici hata → attempt++ → yeniden claim → başarı
  2) CIRCUIT-BREAKER— üst üste hata → 'failed', sonsuz retry YOK
  3) CHECKPOINT     — çökme sonrası kaldığı yerden (fonksiyon düğümü)
  4) DAG KAPISI     — parent bitmeden çocuk 'ready' olmuyor
  5) spawn_task     — yürütme anında yeni task + frenler (2/task, 12/board)
  6) SINIR DURUMLAR — bilinmeyen fn, bozuk arg, olmayan bağımlılık, boş başlık
  7) VERİ AKIŞI     — upstream sonucu alt düğüme geçiyor mu
  8) SCHEDULING     — var mı? (dürüst tespit)

    .venv/bin/python demo-brain-agent/test_tasklife.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import functions as F          # noqa: E402
import orchestrator as O       # noqa: E402
import taskboard as TB         # noqa: E402
from taskboard import TaskBoard, make_task_tools, make_worker_task_tool  # noqa: E402

SONUC: list = []


def kontrol(ad: str, beklenen, gercek, detay: str = ""):
    ok = beklenen == gercek
    SONUC.append({"ad": ad, "beklenen": str(beklenen), "gercek": str(gercek),
                  "gecti": ok, "detay": detay})
    print(f"  {'✓' if ok else '✗'} {ad:<52} beklenen={beklenen}  gerçek={gercek}"
          + (f"   {detay}" if detay else ""))
    return ok


# ═══════════════ 1) RETRY ═══════════════
def t_retry():
    print("\n── 1) RETRY — geçici hata → yeniden denenir ──")
    b = TaskBoard()
    tid = b.create_task("retry denemesi", kind="function", fn="fetch_source")
    b.claim_next("w1")
    st = b.fail(tid, "geçici ağ hatası")
    t = b.get(tid)
    kontrol("hata sonrası durum 'ready'", "ready", st)
    kontrol("attempt artıyor", 1, t["attempt"])
    kontrol("breaker sayacı artıyor", 1, t["consecutive_failures"])
    again = b.claim_next("w2")
    kontrol("başka worker yeniden claim edebiliyor", True, again is not None)
    b.complete(tid, "ikinci denemede oldu")
    kontrol("tamamlanınca breaker sıfırlanıyor", 0, b.get(tid)["consecutive_failures"])
    kontrol("olay günlüğü retry'ı kaydetti", True,
            "retry_scheduled" in [e["kind"] for e in b.events(tid)])


# ═══════════════ 2) CIRCUIT-BREAKER ═══════════════
def t_breaker():
    print(f"\n── 2) CIRCUIT-BREAKER — limit={TB.BREAKER_LIMIT} ──")
    b = TaskBoard()
    tid = b.create_task("hep patlayan iş", kind="function", fn="fetch_source")
    durumlar = []
    for i in range(TB.BREAKER_LIMIT + 1):
        if b.get(tid)["status"] == "ready":
            b.claim_next(f"w{i}")
            durumlar.append(b.fail(tid, f"hata {i}"))
    kontrol("son durum 'failed' (sonsuz retry yok)", "failed", b.get(tid)["status"],
            f"durum dizisi: {durumlar}")
    kontrol("failed task artık claim edilemiyor", None, b.claim_next("wX"))
    kontrol("breaker limitte kapandı", TB.BREAKER_LIMIT,
            b.get(tid)["consecutive_failures"])


# ═══════════════ 3) CHECKPOINT (fonksiyon düğümü, düzeltme sonrası) ═══════════════
def t_checkpoint():
    print("\n── 3) ÇÖKME → CHECKPOINT → DEVRALMA (fonksiyon düğümü) ──")
    b = TaskBoard()
    tid = b.create_task("çökecek fn", kind="function", fn="fetch_source",
                        fn_args={"path": "auth/login.py"})
    t = b.claim_next("w1")
    caught = None
    try:
        O.run_one_task(t, b, crash_after_turn=1)      # ← düzeltme buradaydı
    except O.WorkerCrash as e:
        caught = str(e)
    kontrol("fonksiyon düğümünde çökme TETİKLENİYOR", True, caught is not None, caught or "")
    kontrol("çökme sonrası checkpoint DOLU", True, bool(b.get(tid)["checkpoint"]))
    kontrol("task hâlâ 'running' (complete çağrılmadı)", "running", b.get(tid)["status"])
    n = b.recover_stale(force=True)
    kontrol("recover_stale topluyor", 1, n)
    kontrol("durum 'ready'ye döndü", "ready", b.get(tid)["status"])
    kontrol("checkpoint KORUNDU (kurtarma silmedi)", True, bool(b.get(tid)["checkpoint"]))


# ═══════════════ 4) DAG KAPISI ═══════════════
def t_dag():
    print("\n── 4) DAG KAPISI — parent bitmeden çocuk açılmıyor ──")
    b = TaskBoard()
    a = b.create_task("A", kind="function", fn="fetch_source")
    c = b.create_task("C (A ve B'ye bağlı)", kind="function", fn="cross_check", parents=[a])
    kontrol("bağımlı task 'blocked' doğuyor", "blocked", b.get(c)["status"])
    kontrol("blocked task claim EDİLEMİYOR", a, (b.claim_next("w1") or {}).get("id"))
    kontrol("recompute_ready parent bitmeden açmıyor", 0, b.recompute_ready())
    b.complete(a, "ok")
    kontrol("parent bitince kapı açılıyor", 1, b.recompute_ready())
    kontrol("çocuk artık 'ready'", "ready", b.get(c)["status"])


# ═══════════════ 5) spawn_task + frenler ═══════════════
def t_spawn():
    print(f"\n── 5) spawn_task — yürütme anında task (fren: {TB.MAX_SPAWN_PER_TASK}/task, "
          f"{TB.MAX_TASKS_TOTAL}/board) ──")
    b = TaskBoard()
    parent = b.create_task("ana iş", kind="function", fn="fetch_source")
    log: list = []
    tool = make_worker_task_tool(b, parent, log=log)["spawn_task"][0]
    r1 = tool(title="keşfedilen iş 1")
    r2 = tool(title="keşfedilen iş 2")
    r3 = tool(title="keşfedilen iş 3")
    kontrol("1. spawn kabul", True, "oluşturuldu" in r1 or "açıldı" in r1)
    kontrol("2. spawn kabul", True, "oluşturuldu" in r2 or "açıldı" in r2)
    kontrol("3. spawn REDDEDİLDİ (task başına fren)", True, "reddedildi" in r3)
    yeni = [t for t in b.list_tasks() if str(t["created_by"]).startswith("worker:")]
    kontrol("yürütme-anında üretilenler işaretli", 2, len(yeni))
    kontrol("spawn'lanan task bağımsız (parents boş)", True,
            all(not t["parents"] for t in yeni))
    # board tavanı
    b2 = TaskBoard()
    p2 = b2.create_task("ana", kind="function", fn="fetch_source")
    for i in range(TB.MAX_TASKS_TOTAL + 2):
        b2.create_task(f"dolgu{i}", kind="function", fn="fetch_source")
    t2 = make_worker_task_tool(b2, p2)["spawn_task"][0]
    kontrol("board tavanı dolunca REDDEDİLİYOR", True, "reddedildi" in t2(title="fazladan"))


# ═══════════════ 6) SINIR DURUMLAR ═══════════════
def t_sinir():
    print("\n── 6) SINIR / HATA DURUMLARI ──")
    b = TaskBoard()
    tools = make_task_tools(b)
    add = tools["add_step"][0]
    kontrol("bilinmeyen fonksiyon reddediliyor", True,
            "kayıtlı değil" in add(fn="deploy_to_prod"))
    kontrol("bozuk JSON argüman reddediliyor", True,
            "geçerli JSON" in add(fn="fetch_source", args_json="{bozuk"))
    kontrol("olmayan bağımlılık reddediliyor", True,
            "bilinmeyen bağımlılık" in add(fn="fetch_source", depends_on="t_yok"))
    r = add(fn="scan_patterns", args_json='{"pattern":"x","gizli":"rm -rf /"}')
    kontrol("tanımsız argüman ELENİYOR", True, "YOK SAYILAN" in r)
    tid = r.split()[2]
    kontrol("çöp argüman board'a YAZILMIYOR", ["pattern"], list(b.get(tid)["fn_args"]))
    try:
        b.create_task("", kind="function", fn="fetch_source")
        bos_ok = False
    except ValueError:
        bos_ok = True
    kontrol("boş başlık reddediliyor", True, bos_ok)
    try:
        b.create_task("fn'siz fonksiyon", kind="function")
        fnsiz = False
    except ValueError:
        fnsiz = True
    kontrol("kind=function ama fn yok → reddediliyor", True, fnsiz)
    try:
        F.call("yok_boyle", {}, {})
        cagri = False
    except ValueError:
        cagri = True
    kontrol("yürütücü de bilinmeyen fn'yi reddediyor", True, cagri)


# ═══════════════ 7) VERİ AKIŞI ═══════════════
def t_veri():
    print("\n── 7) VERİ AKIŞI — upstream sonucu alt düğüme geçiyor mu ──")
    b = TaskBoard()
    a = b.create_task("oku", kind="function", fn="fetch_source",
                      fn_args={"path": "auth/login.py"})
    c = b.create_task("tara", kind="function", fn="scan_patterns",
                      fn_args={"pattern": "mfa_token"}, parents=[a])
    t = b.claim_next("w1")
    _k, res, _ = O.run_one_task(t, b)
    b.complete(a, res)
    b.recompute_ready()
    up = b.upstream_results(c)
    kontrol("upstream sonucu görünüyor", True, a in up)
    kontrol("upstream'de path (referans) var", True, "path" in up.get(a, {}))
    kontrol("ham içerik sonuçta TAŞINMIYOR", False, "text" in up.get(a, {}))
    t2 = b.claim_next("w2")
    _k2, res2, _ = O.run_one_task(t2, b)
    d = json.loads(res2)
    kontrol("alt düğüm upstream'i KULLANDI (eşleşme buldu)", True, d["count"] > 0,
            f"count={d['count']}")


# ═══════════════ 8) SCHEDULING — dürüst tespit ═══════════════
def t_scheduling():
    print("\n── 8) SCHEDULING — sistemde var mı? ──")
    import inspect
    kaynak = ""
    for m in (TB, O):
        kaynak += inspect.getsource(m)
    # yorum/metin içindeki 'cron' kelimesini sayma — GERÇEK uygulama arıyoruz
    kod = "\n".join(ln for ln in kaynak.splitlines()
                    if not ln.strip().startswith("#") and '"' not in ln and "'" not in ln)
    cron = any(k in kod for k in ("schedule_at", "next_run", "interval_sec", "crontab"))
    kontrol("zamanlanmış koşu (cron/interval) UYGULANMIŞ mı", False, cron,
            "→ HAYIR: scheduling bu sistemde hiç yok (koçun 6 ekseninden biri açık)")
    kontrol("board'da zaman alanı (next_run/schedule) var mı", False,
            "next_run" in TB.SCHEMA or "schedule" in TB.SCHEMA)


def main():
    print("=" * 88)
    print("TASK YAŞAM DÖNGÜSÜ — kapsamlı test (sohbetten ulaşılamayan yollar)")
    print("=" * 88)
    for f in (t_retry, t_breaker, t_checkpoint, t_dag, t_spawn, t_sinir, t_veri, t_scheduling):
        try:
            f()
        except Exception as e:
            SONUC.append({"ad": f.__name__, "gecti": False,
                          "detay": f"{type(e).__name__}: {e}"})
            print(f"  ✗ {f.__name__} PATLADI: {type(e).__name__}: {e}")
    g = sum(1 for s in SONUC if s["gecti"])
    print("\n" + "=" * 88)
    print(f"  SONUÇ: {g}/{len(SONUC)} kontrol geçti")
    kalan = [s for s in SONUC if not s["gecti"]]
    for s in kalan:
        print(f"    ✗ {s['ad']}  (beklenen={s.get('beklenen')} gerçek={s.get('gercek')})"
              f"  {s.get('detay','')}")
    print("=" * 88)
    (HERE / "test_tasklife_sonuc.json").write_text(
        json.dumps(SONUC, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
