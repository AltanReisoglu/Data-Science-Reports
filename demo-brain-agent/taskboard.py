#!/usr/bin/env python3
"""
taskboard.py — Ajanın ÇALIŞMA ANINDA ürettiği task'ları yöneten durable board.

Buradaki "task management" tanımı (doğrusu bu):
    Ajan bir hedefi alır, ÇALIŞMA ANINDA `create_task(...)` tool'unu çağırarak işi
    task'lara böler (dinamik). Board bu task'ları yönetir: kuyruk, bağımlılık
    (DAG) kapısı, claim, retry, durum takibi, çökme sonrası devam.

Yani task'ları AJAN üretir; board/motor onları YÖNETİR.

    A = TASK  (ajanın ürettiği iş birimi)   ← board bunu yönetir
    B = ADIM  (task'ı yürütürken tool çağrısı)
    C = ALT-AJAN (bir task'ı koşturan worker)

Task FSM:
    blocked ──(parent'lar done)──► ready ──(CAS claim)──► running ──► done
       ▲                             ▲                       │
       └───── yeni bağımlılık ───────┴──(hata: retry hakkı)───┤
                                     └──(breaker doldu)──► failed

Özellikler (Hermes kanban çekirdeğinin sadeleştirilmiş hali):
  • DAG bağımlılığı  : `parents` → `recompute_ready()` kapıyı açar
  • CAS-claim        : `WHERE claim_lock IS NULL` → at-most-once, dağıtık kilit yok
  • lease+heartbeat  : worker ölürse lease dolar
  • recover_stale    : çökeni fark et → ready (checkpoint korunur)
  • circuit-breaker  : üst üste hata → failed (sonsuz retry yok)
  • olay günlüğü     : created/claimed/completed/failed/recovered (denetim izi)
"""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import tempfile
import time
from pathlib import Path

LEASE_SECONDS = 30
BREAKER_LIMIT = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    body                 TEXT NOT NULL DEFAULT '',
    status               TEXT NOT NULL,               -- blocked|ready|running|done|failed|cancelled
    kind                 TEXT NOT NULL DEFAULT 'function', -- function (DETERMİNİSTİK) | agent (LLM)
    fn                   TEXT,                        -- kind='function' ise kayıtlı fonksiyon adı
    fn_args              TEXT NOT NULL DEFAULT '{}',  -- JSON argümanlar
    parents              TEXT NOT NULL DEFAULT '[]',  -- JSON: bağımlı olunan task id'leri
    priority             INTEGER NOT NULL DEFAULT 5,
    created_by           TEXT NOT NULL DEFAULT 'agent',
    claim_lock           TEXT,
    claim_expires        INTEGER,
    worker_pid           INTEGER,
    attempt              INTEGER NOT NULL DEFAULT 0,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    checkpoint           TEXT NOT NULL DEFAULT '{}',  -- yürütme sırasındaki kısmi durum
    result               TEXT,
    created_at           INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS task_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT, at INTEGER NOT NULL
);
"""


MAX_RESULT_CHARS = 64_000     # sonuç alanı tavanı (veri akışı bunun üstünde taşınmamalı)


def _cap_result(result, tid, board=None) -> str:
    """Sonucu tavana kadar sakla. Kesilmesi gerekiyorsa SESSİZCE bozma —
    geçerli JSON bırak ve kesildiğini işaretle (aksi halde upstream json.loads patlar)."""
    s = str(result)
    if len(s) <= MAX_RESULT_CHARS:
        return s
    import json as _j
    return _j.dumps({"_truncated": True, "_orig_chars": len(s),
                     "_hint": "büyük veriyi sonuçtan akıtma; referans (path/id) geçir",
                     "_preview": s[:2000]}, ensure_ascii=False)


class TaskBoard:
    """Ajanın ürettiği task'ların durable kaydı ve yaşam döngüsü motoru."""

    def __init__(self, db_path: Path | None = None):
        self.path = db_path or (Path(tempfile.mkdtemp(prefix="brain_board_")) / "board.db")
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        # ÇOK SÜREÇ İÇİN ŞART: varsayılan busy_timeout=0'dır; iki süreç aynı anda
        # yazmaya kalkınca anında SQLITE_BUSY fırlatır. WAL tek başına yetmez —
        # yazarlar yine seri çalışır, bekleme süresi verilmezse hata alırlar.
        self.conn.execute("PRAGMA busy_timeout=10000")   # 10 sn bekle, hata verme
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------------- olay günlüğü ----------------
    def _ev(self, task_id: str, kind: str, detail: str = ""):
        self.conn.execute(
            "INSERT INTO task_events (task_id, kind, detail, at) VALUES (?,?,?,?)",
            (task_id, kind, detail, int(time.time())))
        self.conn.commit()

    def events(self, task_id: str | None = None) -> list:
        if task_id:
            rows = self.conn.execute(
                "SELECT task_id, kind, detail FROM task_events WHERE task_id=? ORDER BY id",
                (task_id,)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT task_id, kind, detail FROM task_events ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    # ---------------- 1) OLUŞTURMA (ajan bunu tool ile çağırır) ----------------
    def create_task(self, title: str, body: str = "", parents: list | None = None,
                    priority: int = 5, created_by: str = "agent",
                    kind: str = "function", fn: str | None = None,
                    fn_args: dict | None = None) -> str:
        """Yeni task yarat. parents doluysa 'blocked', boşsa 'ready' doğar.

        kind='function' (VARSAYILAN): deterministik fonksiyon düğümü — Airflow
            operatörü gibi. `fn` kayıtlı fonksiyon adı, `fn_args` argümanları.
        kind='agent': LLM muhakemesi gereken istisnai düğüm.
        """
        if not title or not title.strip():
            raise ValueError("task başlığı zorunlu")
        if kind not in ("function", "agent"):
            raise ValueError("kind 'function' ya da 'agent' olmalı")
        if kind == "function" and not fn:
            raise ValueError("kind='function' için fn (fonksiyon adı) zorunlu")
        if kind == "function":
            # KAYIT ANINDA DOĞRULA. Önceden board her adı kabul ediyordu; uydurma bir
            # fonksiyon adı (LLM halüsinasyonu ya da kayıtlı pipeline'da silinmiş bir fn)
            # ancak YÜRÜTMEDE, hem de 3 boşuna denemeden sonra fark ediliyordu.
            import functions as _F
            if _F.resolve(fn) is None:
                _gecerli = sorted({k for v in _F.PACKS.values() for k in v["fns"]})
                raise ValueError(f"bilinmeyen fonksiyon: {fn} "
                                 f"(geçerli: {', '.join(_gecerli)})")
        # sırayı koruyarak tekilleştir (yinelenen bağımlılık kapıyı kilitler)
        parents = list(dict.fromkeys(p for p in (parents or []) if p))
        tid = "t_" + secrets.token_hex(3)
        status = "blocked" if parents else "ready"
        self.conn.execute(
            "INSERT INTO tasks (id,title,body,status,kind,fn,fn_args,parents,priority,"
            "created_by,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (tid, title.strip(), body or "", status, kind, fn,
             json.dumps(fn_args or {}, ensure_ascii=False),
             json.dumps(parents, ensure_ascii=False), int(priority), created_by,
             int(time.time())))
        self.conn.commit()
        self._ev(tid, "created",
                 f"by={created_by} kind={kind}" + (f" fn={fn}" if fn else "") +
                 f" parents={parents}")
        return tid

    # ---------------- 2) PLANLAMA (bağımlılık kapısı) ----------------
    def recompute_ready(self) -> int:
        """Bütün parent'ları 'done' olan blocked task'ları 'ready'ye terfi ettir."""
        promoted = 0
        for r in self.conn.execute(
                "SELECT id, parents FROM tasks WHERE status='blocked'").fetchall():
            # TEKİLLEŞTİR: aynı parent iki kez yazılırsa (planlayıcı depends_on'da
            # tekrar edebiliyor) COUNT(*) satır sayar, len(parents) tekrarı sayar →
            # eşitlik ASLA sağlanmaz ve düğüm sonsuza dek 'blocked' kalır.
            parents = list(dict.fromkeys(json.loads(r["parents"] or "[]")))
            if not parents:
                ok = True
            else:
                qs = ",".join("?" * len(parents))
                done = self.conn.execute(
                    f"SELECT COUNT(*) c FROM tasks WHERE id IN ({qs}) AND status='done'",
                    parents).fetchone()["c"]
                ok = done == len(parents)
            if ok:
                self.conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (r["id"],))
                self._ev(r["id"], "unblocked", "parent'lar tamamlandı")
                promoted += 1
        self.conn.commit()
        return promoted

    # ---------------- 3) KUYRUKTAN ALMA (CAS-claim) ----------------
    def claim_next(self, claimer: str, lease: int = LEASE_SECONDS) -> dict | None:
        """En yüksek öncelikli 'ready' task'ı ATOMİK olarak kap (at-most-once).

        İki worker aynı anda çağırırsa SQLite tek UPDATE uygular → biri alır,
        diğeri None döner. Dağıtık kilit gerekmez.
        """
        row = self.conn.execute(
            "SELECT id FROM tasks WHERE status='ready' AND claim_lock IS NULL "
            "ORDER BY priority DESC, created_at ASC LIMIT 1").fetchone()
        if not row:
            return None
        tid = row["id"]
        cur = self.conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, worker_pid=? "
            "WHERE id=? AND status='ready' AND claim_lock IS NULL",
            (claimer, int(time.time()) + lease, os.getpid(), tid))
        self.conn.commit()
        if cur.rowcount != 1:
            return None                      # yarışı kaybetti
        self._ev(tid, "claimed", claimer)
        return self.get(tid)

    def claim(self, tid: str, claimer: str, lease: int = LEASE_SECONDS) -> dict | None:
        """BELİRLİ bir task'ı CAS ile kap (claim_next id seçmez, bu seçer).

        Gerekçe: motor (Temporal/Celery) aynı task'ı yeniden denerken elindeki kayıt
        BAYAT olabilir — fail() claim'i temizlemiştir. Yeniden denemeden önce task'ı
        yeniden kapmak gerekir, yoksa yapılan iş fencing'e takılıp ÇÖPE gider.
        """
        cur = self.conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, worker_pid=? "
            "WHERE id=? AND status='ready' AND claim_lock IS NULL",
            (claimer, int(time.time()) + lease, os.getpid(), tid))
        self.conn.commit()
        if cur.rowcount != 1:
            return None
        self._ev(tid, "claimed", claimer)
        return self.get(tid)

    def heartbeat(self, tid: str, claimer: str, lease: int = LEASE_SECONDS):
        self.conn.execute(
            "UPDATE tasks SET claim_expires=? WHERE id=? AND claim_lock=?",
            (int(time.time()) + lease, tid, claimer))
        self.conn.commit()

    def save_checkpoint(self, tid: str, checkpoint: dict):
        self.conn.execute("UPDATE tasks SET checkpoint=? WHERE id=?",
                          (json.dumps(checkpoint, ensure_ascii=False, default=str), tid))
        self.conn.commit()

    # ---------------- 4) TAMAMLAMA / HATA ----------------
    def complete(self, tid: str, result: str = "", claimer: str | None = None) -> bool:
        """Task'ı tamamla. claimer verilirse SAHİPLİK doğrulanır (fencing).

        Neden gerekli: lease dolup task başkasına devredildikten sonra ESKİ worker
        uyanıp complete() çağırabilir. Sahiplik kontrolü olmadan bu yazma geçer ve
        devralanın sonucunu EZER — at-most-once yalnız claim'de kalır, yazmada kaybolur.
        """
        sql = ("UPDATE tasks SET status='done', result=?, claim_lock=NULL, "
               "claim_expires=NULL, worker_pid=NULL, consecutive_failures=0 WHERE id=?")
        par = [_cap_result(result, tid, self), tid]
        if claimer is not None:
            sql += " AND claim_lock=?"
            par.append(claimer)
        cur = self.conn.execute(sql, par)
        self.conn.commit()
        if cur.rowcount != 1:
            self._ev(tid, "stale_write_reddedildi", f"claim artık {claimer!r} değil")
            return False
        self._ev(tid, "completed")
        return True

    def fail(self, tid: str, error: str = "", claimer: str | None = None,
             permanent: bool = False) -> str:
        """Geçici hata: attempt++/breaker++. Limit aşılmadıysa 'ready', aşıldıysa 'failed'.

        claimer verilirse complete() ile aynı fencing uygulanır: devredilmiş bir task'ı
        eski sahibi 'failed' işaretleyip devralanın işini boşa çıkaramaz.

        permanent=True: SÖZLEŞME hatası (bilinmeyen fonksiyon, geçersiz argüman) —
        tekrar denemek aynı sonucu verir. Tek denemede 'failed'. Ağ/zaman aşımı gibi
        GEÇİCİ hatalarla aynı kefeye konursa breaker 3 kez boşuna koşturur.
        """
        cf = self.conn.execute("SELECT consecutive_failures FROM tasks WHERE id=?",
                               (tid,)).fetchone()["consecutive_failures"] + 1
        status = "failed" if (permanent or cf >= BREAKER_LIMIT) else "ready"
        sql = ("UPDATE tasks SET status=?, attempt=attempt+1, consecutive_failures=?, "
               "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?")
        par = [status, cf, tid]
        if claimer is not None:
            sql += " AND claim_lock=?"
            par.append(claimer)
        cur = self.conn.execute(sql, par)
        self.conn.commit()
        if cur.rowcount != 1:
            self._ev(tid, "stale_write_reddedildi", f"claim artık {claimer!r} değil")
            return "stale"
        self._ev(tid, "failed" if status == "failed" else "retry_scheduled",
                 f"{error[:120]} breaker={cf}" + (" KALICI(retry yok)" if permanent else ""))
        if status == "failed":
            # BATAN DALI KAPAT: parent'ı 'failed' olan bir düğüm asla 'done' olamaz,
            # dolayısıyla recompute_ready() onu HİÇ terfi ettiremez → sonsuza dek
            # 'blocked' kalırdı. Görünürde iş var, gerçekte yürütülemez: all_settled()
            # False döner, operatör "akış bitti mi?" sorusuna cevap veremez.
            # Bunu tek noktada (fail) kapatıyoruz ki dört backend de aynı davransın.
            self.last_cancelled = self.cancel_downstream(tid, f"üst düğüm battı: {tid}")
        return status

    def cancel_downstream(self, tid: str, reason: str = "") -> list:
        """tid'in tüm ALT soyunu 'cancelled' yap. Dönüş: iptal edilen id listesi.

        Yürütülmemiş ama artık yürütülemeyecek işi 'blocked' değil 'cancelled' olarak
        işaretler — 'bekliyor' ile 'asla koşmayacak' ayrımı board'da GÖRÜNÜR olur.
        """
        iptal, sinir = [], [tid]
        while sinir:
            ebeveyn = sinir.pop()
            for r in self.conn.execute(
                    "SELECT id, parents FROM tasks WHERE status IN ('blocked','ready')"
            ).fetchall():
                if ebeveyn in json.loads(r["parents"] or "[]") and r["id"] not in iptal:
                    self.conn.execute("UPDATE tasks SET status='cancelled' WHERE id=?",
                                      (r["id"],))
                    self._ev(r["id"], "cancelled", reason or f"üst düğüm battı: {ebeveyn}")
                    iptal.append(r["id"])
                    sinir.append(r["id"])
        self.conn.commit()
        return iptal

    # ---------------- 5) ÇÖKME SONRASI DEVAM ----------------
    @staticmethod
    def _pid_alive(pid) -> bool:
        if not pid:
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except (ProcessLookupError, PermissionError, ValueError):
            return pid == os.getpid()

    def recover_stale(self, force: bool = False) -> int:
        """Lease'i dolmuş / PID'i ölmüş 'running' task'ları 'ready'ye döndür.

        checkpoint'e DOKUNMAZ → devralan worker kaldığı yerden sürdürür.
        force=True: lease'e bakmadan tüm running'leri topla (demo hızlandırma).
        """
        now, n = int(time.time()), 0
        for r in self.conn.execute(
                "SELECT id, claim_expires, worker_pid FROM tasks WHERE status='running'").fetchall():
            expired = r["claim_expires"] is not None and r["claim_expires"] < now
            if force or expired or not self._pid_alive(r["worker_pid"]):
                self.conn.execute(
                    "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
                    "worker_pid=NULL WHERE id=?", (r["id"],))
                self._ev(r["id"], "recovered", "lease/pid")
                n += 1
        self.conn.commit()
        return n

    # ---------------- sorgular ----------------
    def get(self, tid: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["parents"] = json.loads(d["parents"] or "[]")
        d["checkpoint"] = json.loads(d["checkpoint"] or "{}")
        d["fn_args"] = json.loads(d.get("fn_args") or "{}")
        return d

    def list_tasks(self, status: str | None = None) -> list:
        q = "SELECT * FROM tasks"
        args: tuple = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        q += " ORDER BY created_at ASC"
        out = []
        for r in self.conn.execute(q, args).fetchall():
            d = dict(r)
            d["parents"] = json.loads(d["parents"] or "[]")
            d["fn_args"] = json.loads(d.get("fn_args") or "{}")
            d.pop("checkpoint", None)
            out.append(d)
        return out

    def upstream_results(self, tid: str) -> dict:
        """Bu task'ın parent'larının SONUÇLARINI döndür → düğümler arası veri akışı.

        Airflow'daki XCom'un karşılığı: bir düğümün çıktısı, çocuğuna girdi olur.
        """
        t = self.get(tid)
        if not t or not t["parents"]:
            return {}
        out = {}
        for pid in t["parents"]:
            r = self.conn.execute("SELECT result FROM tasks WHERE id=?", (pid,)).fetchone()
            if r and r["result"]:
                try:
                    out[pid] = json.loads(r["result"])
                except Exception:
                    out[pid] = {"text": r["result"]}
        return out

    def counts(self) -> dict:
        c = {}
        for r in self.conn.execute(
                "SELECT status, COUNT(*) n FROM tasks GROUP BY status").fetchall():
            c[r["status"]] = r["n"]
        return c

    def all_settled(self) -> bool:
        """Yürütülecek iş kalmadı mı? (ready/running/blocked yok)"""
        r = self.conn.execute(
            "SELECT COUNT(*) n FROM tasks WHERE status IN ('ready','running','blocked')"
        ).fetchone()
        return r["n"] == 0

    def board_text(self) -> str:
        lines = []
        for t in self.list_tasks():
            dep = f" ← {','.join(t['parents'])}" if t["parents"] else ""
            tag = f"fn:{t['fn']}" if t["kind"] == "function" else "AJAN"
            lines.append(f"  {t['id']}  [{t['status']:<7}] {tag:<18} "
                         f"{t['title'][:44]}{dep}")
        return "\n".join(lines) or "  (board boş)"


# ═══════════════ AJANIN ÇAĞIRABİLECEĞİ TASK TOOL'LARI ═══════════════
# Bunlar LLM'e tool olarak verilir; ajan çalışma anında task ÜRETİR.

def make_task_tools(board: TaskBoard, log: list | None = None) -> dict:
    """Planlayıcı ajana verilen tool'lar — FONKSİYON ÖNCELİKLİ.

    add_step        : deterministik fonksiyon düğümü (Airflow operatörü gibi) ← ASIL
    add_agent_step  : LLM muhakemesi gereken istisnai düğüm                   ← İSTİSNA
    list_tasks      : board'u gör
    """
    import functions as F

    def add_step(fn: str, title: str = "", args_json: str = "",
                 depends_on: str = "", priority: int = 5) -> str:
        """DETERMİNİSTİK fonksiyon düğümü ekle (varsayılan task türü)."""
        if fn not in F.REGISTRY:
            return (f"[hata] '{fn}' kayıtlı değil. Kullanılabilir fonksiyonlar: "
                    f"{', '.join(F.REGISTRY)}")
        try:
            args = json.loads(args_json) if args_json else {}
            if not isinstance(args, dict):
                args = {}
        except Exception:
            return "[hata] args_json geçerli JSON olmalı, ör: {\"path\": \"auth/login.py\"}"
        # tanımsız argümanları GİRİŞTE ele (board'a çöp yazılmasın)
        allowed = set(F.REGISTRY[fn][2])
        unknown = [k for k in args if k not in allowed]
        args = {k: v for k, v in args.items() if k in allowed}
        parents = [p.strip() for p in (depends_on or "").split(",") if p.strip()]
        missing = [p for p in parents if board.get(p) is None]
        if missing:
            return f"[hata] bilinmeyen bağımlılık: {missing} — önce o düğümü ekle."
        tid = board.create_task(title=title or f"{fn}()", body="", parents=parents,
                                priority=priority, created_by="agent",
                                kind="function", fn=fn, fn_args=args)
        t = board.get(tid)
        if log is not None:
            log.append(f"PLAN + fonksiyon düğümü: {tid} = {fn}({args}) [{t['status']}]"
                       + (f" ← {parents}" if parents else ""))
        return (f"düğüm eklendi: {tid} · fn={fn} · durum={t['status']}"
                + (f" · bağımlı={parents}" if parents else " · hemen hazır")
                + (f" · YOK SAYILAN argümanlar: {unknown} (bu fonksiyon "
                   f"yalnız {sorted(allowed)} kabul eder)" if unknown else ""))

    def add_agent_step(title: str, body: str = "", depends_on: str = "",
                       priority: int = 5) -> str:
        """LLM MUHAKEMESİ gereken düğüm ekle — sadece deterministik fonksiyonla
        yapılamayacak iş için (yorumlama, karar, serbest metin üretimi)."""
        parents = [p.strip() for p in (depends_on or "").split(",") if p.strip()]
        missing = [p for p in parents if board.get(p) is None]
        if missing:
            return f"[hata] bilinmeyen bağımlılık: {missing}"
        tid = board.create_task(title=title, body=body, parents=parents,
                                priority=priority, created_by="agent", kind="agent")
        t = board.get(tid)
        if log is not None:
            log.append(f"PLAN + AJAN düğümü (LLM): {tid} '{title[:40]}' [{t['status']}]"
                       + (f" ← {parents}" if parents else ""))
        return (f"ajan düğümü eklendi: {tid} · durum={t['status']}"
                + (f" · bağımlı={parents}" if parents else ""))

    def list_tasks(status: str = "") -> str:
        """Board'daki düğümleri listele."""
        ts = board.list_tasks(status or None)
        if not ts:
            return "board boş"
        return "\n".join(
            f"{t['id']} [{t['status']}] "
            + (f"fn={t['fn']}" if t["kind"] == "function" else "AJAN")
            + f" {t['title']}"
            + (f" (bağımlı: {','.join(t['parents'])})" if t["parents"] else "")
            for t in ts)

    return {
        "add_step": (add_step,
                     "DETERMİNİSTİK fonksiyon düğümü ekle (VARSAYILAN — önce bunu kullan). "
                     "fn: kayıtlı fonksiyon adı, args_json: JSON argümanlar, "
                     "depends_on: önce bitmesi gereken düğüm id'leri (virgüllü).",
                     {"fn": {"type": "string", "description": "kayıtlı fonksiyon adı"},
                      "title": {"type": "string", "description": "düğüm başlığı"},
                      "args_json": {"type": "string",
                                    "description": 'JSON argüman, ör: {"path": "auth/login.py"}'},
                      "depends_on": {"type": "string", "description": "id'ler (virgüllü)"},
                      "priority": {"type": "integer", "description": "1-9"}},
                     ["fn"]),
        "add_agent_step": (add_agent_step,
                           "LLM muhakemesi gereken düğüm ekle — SADECE deterministik "
                           "fonksiyonla yapılamayacak iş için (yorum, karar, serbest metin).",
                           {"title": {"type": "string", "description": "ne yapılacak"},
                            "body": {"type": "string", "description": "detay"},
                            "depends_on": {"type": "string", "description": "id'ler (virgüllü)"},
                            "priority": {"type": "integer", "description": "1-9"}},
                           ["title"]),
        "list_tasks": (list_tasks, "Board'daki düğümleri listele.",
                       {"status": {"type": "string",
                                   "description": "blocked|ready|running|done|failed"}},
                       []),
    }


# Kaçak task üretimine karşı tavan: ajan sonsuz task açıp kendini kilitlemesin.
MAX_TASKS_TOTAL = 12
MAX_SPAWN_PER_TASK = 2


def make_worker_task_tool(board: TaskBoard, parent_task_id: str,
                          log: list | None = None) -> dict:
    """İŞÇİ ajana verilen task-açma tool'u — YÜRÜTME SIRASINDA replanlama.

    Ajan bir task'ı yaparken YENİ İŞ keşfederse (ör. "burada ayrıca şu modül de
    denetlenmeli") bunu ayrı bir task olarak açabilir. Board onu kuyruğa alır,
    dispatch döngüsü sıradaki turda yürütür.

    Kaçak üretime karşı iki fren: board toplamı MAX_TASKS_TOTAL, bu task'ın
    açabileceği MAX_SPAWN_PER_TASK.
    """
    spawned = {"n": 0}

    def spawn_task(title: str, body: str = "", priority: int = 5) -> str:
        """Yürütme sırasında keşfedilen YENİ işi ayrı bir task olarak aç."""
        if spawned["n"] >= MAX_SPAWN_PER_TASK:
            return (f"[reddedildi] bu task en fazla {MAX_SPAWN_PER_TASK} alt-task açabilir; "
                    f"kalan işi kendi sonucunda anlat.")
        if len(board.list_tasks()) >= MAX_TASKS_TOTAL:
            return (f"[reddedildi] board tavanı ({MAX_TASKS_TOTAL} task) doldu; "
                    f"yeni task açma, mevcut işi bitir.")
        # Keşfedilen iş doğal dille tarif edilir; hazır bir fonksiyona karşılık gelmez
        # → AJAN düğümü olarak açılır. (create_task varsayılanı 'function' olduğu için
        #  kind belirtilmezse "fn zorunlu" hatası alınır — bu satır o yüzden kritik.)
        tid = board.create_task(title=title, body=body, parents=[], priority=int(priority),
                                created_by=f"worker:{parent_task_id}", kind="agent")
        spawned["n"] += 1
        if log is not None:
            log.append(f"SPAWN (yürütme anında) {parent_task_id} → {tid} '{title[:44]}'")
        return (f"yeni task açıldı: {tid} (durum=ready, kuyruğa alındı). "
                f"Sen kendi task'ına devam et.")

    return {
        "spawn_task": (spawn_task,
                       "Bu task'ı yaparken KEŞFETTİĞİN ve ayrı ele alınması gereken yeni bir "
                       "işi task olarak aç. Sadece gerçekten ayrı bir iş varsa kullan.",
                       {"title": {"type": "string", "description": "yeni task başlığı"},
                        "body": {"type": "string", "description": "ne yapılacağı"},
                        "priority": {"type": "integer", "description": "1-9"}},
                       ["title"]),
    }


if __name__ == "__main__":
    # bağımsız test: bağımlılık kapısı + CAS claim + çökme + kurtarma
    b = TaskBoard()
    print("=" * 76)
    print("taskboard.py — bağımsız test (ajan yerine elle task üretiliyor)")
    print(f"DB: {b.path}")
    print("=" * 76)

    t1 = b.create_task("ödeme doğrula", priority=7)
    t2 = b.create_task("stok düş", parents=[t1])
    t3 = b.create_task("kargo oluştur", parents=[t2])
    t4 = b.create_task("audit log yaz", priority=3)     # bağımsız → paralel
    print("\n1) Ajan 4 task üretti (t2←t1, t3←t2):")
    print(b.board_text())
    print(f"   durumlar: {b.counts()}")

    print("\n2) claim_next: yalnız 'ready' olanlar alınabilir (blocked'lar kapıda bekler)")
    got = b.claim_next("worker-A")
    print(f"   worker-A aldı → {got['id']} '{got['title']}' (p{got['priority']})")
    dup = b.claim_next("worker-B")
    print(f"   worker-B aldı → {dup['id'] if dup else None} "
          f"(aynı task'ı ALAMADI = at-most-once; sıradaki bağımsız task'ı aldı)")

    print("\n3) ÇÖKME: worker-A t1'i bitiremeden öldü")
    b.save_checkpoint(got["id"], {"kismi": "ödeme sorgusu yapıldı"})
    n = b.recover_stale(force=True)
    print(f"   recover_stale() → {n} task 'ready'ye döndü (checkpoint korundu)")
    print(f"   t1 checkpoint: {b.get(t1)['checkpoint']}")

    print("\n4) t1 tamamlanınca bağımlılık kapısı açılır")
    b.claim_next("worker-C")
    b.complete(t1, "ödeme ok")
    print(f"   recompute_ready() → {b.recompute_ready()} task blocked→ready")
    print(b.board_text())

    print(f"\n5) Olay günlüğü ({t1}):")
    print("   " + " → ".join(e["kind"] for e in b.events(t1)))
    print("=" * 76)
