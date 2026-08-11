#!/usr/bin/env python3
"""
scheduler.py — ZAMANLANMIŞ KOŞU (koçun 6. ekseni).

Şimdiye kadar sistemde scheduling HİÇ yoktu; testlerde bunu "yok" diye
belgelemiştik. Bu modül onu kapatıyor: kayıtlı bir akış (pipeline) cron
ifadesiyle zamanlanır, bir yoklayıcı (poller) vakti gelenleri alıp koşturur.

TASARIM — Macro'nun `services/scheduled_action`'ından alınan iki karar:

  1) BAYATLIK KONTROLÜ CLAIM'İN İÇİNDE.
     Bizim board'da `claim_next` (claim_lock IS NULL) ve ayrı bir
     `recover_stale()` süpürmesi var — ikinci adım çağrılmayı UNUTULABİLİR
     bir yol, nitekim Temporal'da tam bu sınıftan bir hata bulmuştuk.
     Burada tek atomik UPDATE:

         WHERE id=? AND (claimed_at IS NULL OR claimed_at < now-LEASE)

     Yani "sahipsiz VEYA kirası dolmuş" tek koşulda. Ayrı kurtarma adımı yok
     → atlanması imkânsız.

  2) NEXT_RUN_AT YAZMA ANINDA TÜRETİLİR.
     Cron'u okuma anında ayrıştırmak yerine bir kez hesaplayıp saklıyoruz;
     arayüz "sonraki koşu"yu cron ayrıştırmadan gösterebiliyor.

Cron ayrıştırıcı bağımlılıksız (croniter yok): 5 alan `dk sa gün ay hafta`,
`*`, `a,b`, `a-b`, `*/n` ve `a-b/n` destekli.

    .venv/bin/python demo-brain-agent/scheduler.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DB = HERE / "scheduler.db"
LEASE_SECONDS = 900          # 15 dk — koşu bu süreyi aşarsa bayat sayılır
POLL_SECONDS = 20            # yoklama aralığı
BATCH = 10                   # bir turda en fazla kaç zamanlama alınır
BATCH_MIN_SECONDS = 5        # tur en az bu kadar sürsün → eş yoklayıcılara şans

SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    cron         TEXT NOT NULL,          -- "dk sa gün ay hafta"
    pipeline_id  TEXT NOT NULL,          -- pipelines_store'daki kayıtlı akış
    backend      TEXT NOT NULL DEFAULT 'own',
    strategy     TEXT NOT NULL DEFAULT 'hermes',
    enabled      INTEGER NOT NULL DEFAULT 1,
    next_run_at  INTEGER NOT NULL,       -- YAZMA anında cron'dan türetilir
    claimed_at   INTEGER,                -- claim damgası (lease ile birlikte)
    last_run_at  INTEGER,
    last_status  TEXT,
    created_at   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS schedule_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    schedule_id  TEXT NOT NULL,
    started_at   INTEGER NOT NULL,
    ended_at     INTEGER,
    ok           INTEGER,
    dugum        INTEGER,
    tamamlanan   INTEGER,
    saniye       REAL,
    detay        TEXT
);
"""


# ═════════════════════ CRON AYRIŞTIRICI (bağımlılıksız) ═════════════════════

def _alan(ifade: str, alt: int, ust: int) -> set[int]:
    """Tek bir cron alanını izin verilen değer kümesine çevir."""
    izin: set[int] = set()
    for parca in ifade.split(","):
        parca = parca.strip()
        adim = 1
        if "/" in parca:
            parca, _, a = parca.partition("/")
            adim = int(a)
            if adim < 1:
                raise ValueError(f"adım 1'den küçük olamaz: /{a}")
        if parca == "*":
            bas, son = alt, ust
        elif "-" in parca:
            b, _, s = parca.partition("-")
            bas, son = int(b), int(s)
        else:
            bas = son = int(parca)
        if bas < alt or son > ust or bas > son:
            raise ValueError(f"'{ifade}' alanı {alt}-{ust} aralığında olmalı")
        izin.update(range(bas, son + 1, adim))
    if not izin:
        raise ValueError(f"boş cron alanı: '{ifade}'")
    return izin


def cron_ayristir(cron: str) -> tuple[set[int], ...]:
    """'dk sa gün ay hafta' → beş küme. Geçersizse ValueError."""
    p = cron.split()
    if len(p) != 5:
        raise ValueError(f"cron 5 alan olmalı (dk sa gün ay hafta), {len(p)} verildi: {cron!r}")
    return (_alan(p[0], 0, 59), _alan(p[1], 0, 23), _alan(p[2], 1, 31),
            _alan(p[3], 1, 12), _alan(p[4], 0, 6))          # 0=Pazar


def sonraki_kosu(cron: str, andan: datetime | None = None, limit_gun: int = 366) -> datetime:
    """cron'un `andan` SONRAKİ ilk tetikleme anı (yerel saat, saniye=0).

    Dakika dakika ilerler — 5 alanlı cron için en fazla ~527k adım/yıl, pratikte
    ilk birkaç yüz adımda bulunur. Bulunamazsa (ör. 30 Şubat) ValueError.
    """
    dk, sa, gun, ay, hafta = cron_ayristir(cron)
    t = (andan or datetime.now()).replace(second=0, microsecond=0) + timedelta(minutes=1)
    sinir = t + timedelta(days=limit_gun)
    while t < sinir:
        # cron kuralı: gün ve hafta alanlarının İKİSİ de kısıtlıysa VEYA'lanır
        gun_kisitli = gun != set(range(1, 32))
        hafta_kisitli = hafta != set(range(0, 7))
        gun_ok = (t.day in gun)
        hafta_ok = ((t.weekday() + 1) % 7) in hafta          # Python: Pzt=0 → cron: Paz=0
        if gun_kisitli and hafta_kisitli:
            tarih_ok = gun_ok or hafta_ok
        elif hafta_kisitli:
            tarih_ok = hafta_ok
        else:
            tarih_ok = gun_ok
        if t.minute in dk and t.hour in sa and t.month in ay and tarih_ok:
            return t
        t += timedelta(minutes=1)
    raise ValueError(f"cron {limit_gun} gün içinde hiç tetiklenmiyor: {cron!r}")


def cron_aciklama(cron: str) -> str:
    """İnsan okunur kısa özet (arayüzde gösterilir)."""
    try:
        p = cron.split()
        gunler = ["Paz", "Pzt", "Sal", "Çar", "Per", "Cum", "Cmt"]
        if p[0].isdigit() and p[1].isdigit():
            saat = f"{int(p[1]):02d}:{int(p[0]):02d}"
            if p[2] == "*" and p[3] == "*" and p[4] == "*":
                return f"her gün {saat}"
            if p[4] != "*" and p[2] == "*":
                try:
                    ad = ", ".join(gunler[i] for i in sorted(_alan(p[4], 0, 6)))
                    return f"{ad} günleri {saat}"
                except Exception:
                    pass
            if p[2] != "*" and p[4] == "*":
                return f"ayın {p[2]}. günü {saat}"
        if p[0].startswith("*/"):
            return f"her {p[0][2:]} dakikada"
        if p[1].startswith("*/") and p[0].isdigit():
            return f"her {p[1][2:]} saatte ({int(p[0]):02d}. dakika)"
    except Exception:
        pass
    return cron


# ═════════════════════ DEPO ═════════════════════

class ScheduleStore:
    """Zamanlama deposu. Claim mantığı Macro'nunkiyle aynı: tek atomik UPDATE."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or DB)
        self.conn = sqlite3.connect(self.path, check_same_thread=False,
                                    isolation_level=None, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=10000")   # çok-süreç: WAL tek başına yetmez
        self.conn.executescript(SCHEMA)

    # ---------- CRUD ----------
    def create(self, name: str, cron: str, pipeline_id: str,
               backend: str = "own", strategy: str = "hermes",
               enabled: bool = True) -> str:
        import secrets
        nxt = sonraki_kosu(cron)                     # YAZMA anında türet (Macro deseni)
        sid = "s_" + secrets.token_hex(3)
        self.conn.execute(
            "INSERT INTO schedules (id,name,cron,pipeline_id,backend,strategy,enabled,"
            "next_run_at,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, name.strip(), cron, pipeline_id, backend, strategy,
             1 if enabled else 0, int(nxt.timestamp()), int(time.time())))
        return sid

    def list(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM schedules ORDER BY next_run_at").fetchall()]

    def get(self, sid: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM schedules WHERE id=?", (sid,)).fetchone()
        return dict(r) if r else None

    def set_enabled(self, sid: str, enabled: bool) -> bool:
        cur = self.conn.execute("UPDATE schedules SET enabled=? WHERE id=?",
                                (1 if enabled else 0, sid))
        return cur.rowcount == 1

    def delete(self, sid: str) -> bool:
        return self.conn.execute("DELETE FROM schedules WHERE id=?", (sid,)).rowcount == 1

    # ---------- claim / release ----------
    def due(self, limit: int = BATCH, now: int | None = None) -> list[dict]:
        """Vakti gelmiş, etkin ve (sahipsiz veya kirası dolmuş) zamanlamalar."""
        now = now if now is not None else int(time.time())
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM schedules WHERE enabled=1 AND next_run_at<=? "
            "AND (claimed_at IS NULL OR claimed_at < ?) "
            "ORDER BY next_run_at LIMIT ?",
            (now, now - LEASE_SECONDS, limit)).fetchall()]

    def claim(self, sid: str, now: int | None = None) -> bool:
        """ATOMİK claim. Bayatlık kontrolü predikatın İÇİNDE — ayrı kurtarma adımı yok.

        İki yoklayıcı aynı anda çağırırsa SQLite tek UPDATE uygular → biri True,
        diğeri False alır (at-most-once).
        """
        now = now if now is not None else int(time.time())
        cur = self.conn.execute(
            "UPDATE schedules SET claimed_at=? WHERE id=? "
            "AND (claimed_at IS NULL OR claimed_at < ?)",
            (now, sid, now - LEASE_SECONDS))
        return cur.rowcount == 1

    def release(self, sid: str, ok: bool, ileri_al: bool = True):
        """Claim'i bırak, sonucu yaz ve bir sonraki tetiklemeyi hesapla."""
        s = self.get(sid)
        nxt = None
        if s and ileri_al:
            try:
                nxt = int(sonraki_kosu(s["cron"]).timestamp())
            except Exception:
                nxt = None
        self.conn.execute(
            "UPDATE schedules SET claimed_at=NULL, last_run_at=?, last_status=?"
            + (", next_run_at=?" if nxt else "") + " WHERE id=?",
            ((int(time.time()), "ok" if ok else "hata", nxt, sid) if nxt
             else (int(time.time()), "ok" if ok else "hata", sid)))

    # ---------- koşu geçmişi ----------
    def run_basla(self, sid: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO schedule_runs (schedule_id, started_at) VALUES (?,?)",
            (sid, int(time.time())))
        return cur.lastrowid

    def run_bitir(self, run_id: int, ok: bool, dugum: int = 0, tamamlanan: int = 0,
                  saniye: float = 0.0, detay: str = ""):
        self.conn.execute(
            "UPDATE schedule_runs SET ended_at=?, ok=?, dugum=?, tamamlanan=?, "
            "saniye=?, detay=? WHERE id=?",
            (int(time.time()), 1 if ok else 0, dugum, tamamlanan, round(saniye, 2),
             detay[:600], run_id))

    def runs(self, sid: str, limit: int = 20) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM schedule_runs WHERE schedule_id=? ORDER BY id DESC LIMIT ?",
            (sid, limit)).fetchall()]


# ═════════════════════ YÜRÜTÜCÜ ═════════════════════

def kosturt(store: ScheduleStore, s: dict, on_event=None) -> tuple[bool, str]:
    """Zamanlamanın işaret ettiği kayıtlı akışı koştur. (ok, özet) döndürür."""
    import orchestrator as O
    import pipelines as P
    from taskboard import TaskBoard

    run_id = store.run_basla(s["id"])
    t0 = time.time()
    doc = P.load(s["pipeline_id"])
    if not doc:
        store.run_bitir(run_id, False, detay=f"akış bulunamadı: {s['pipeline_id']}")
        return False, f"akış bulunamadı: {s['pipeline_id']}"
    try:
        board = TaskBoard()
        res = O.run_saved(doc["nodes"], backend=s["backend"], strategy=s["strategy"],
                          board=board, on_event=on_event)
        batan = len(board.list_tasks("failed"))
        iptal = len(board.list_tasks("cancelled"))
        ok = res.tasks_done > 0 and batan == 0
        ozet = (f"{res.tasks_done}/{res.tasks_created} düğüm · {round(time.time()-t0,1)} sn"
                + (f" · ✗{batan} başarısız" if batan else "")
                + (f" · ⛔{iptal} iptal" if iptal else ""))
        store.run_bitir(run_id, ok, res.tasks_created, res.tasks_done,
                        time.time() - t0, ozet)
        return ok, ozet
    except Exception as e:
        ozet = f"{type(e).__name__}: {str(e)[:160]}"
        store.run_bitir(run_id, False, saniye=time.time() - t0, detay=ozet)
        return False, ozet


def tur(store: ScheduleStore, on_log=None) -> int:
    """Tek yoklama turu: vakti gelenleri claim et ve koştur. Koşan sayısını döner."""
    n = 0
    for s in store.due():
        if not store.claim(s["id"]):
            if on_log:
                on_log(f"[zamanlayıcı] {s['id']} başka yoklayıcı tarafından alındı, atlandı")
            continue                      # at-most-once: yarışı kaybettik
        if on_log:
            on_log(f"[zamanlayıcı] ▶ {s['name']} ({s['id']}) koşuyor · akış={s['pipeline_id']}")
        ok, ozet = kosturt(store, s)
        store.release(s["id"], ok)
        n += 1
        if on_log:
            yeni = store.get(s["id"])
            on_log(f"[zamanlayıcı] {'✓' if ok else '✗'} {s['name']} → {ozet} · "
                   f"sonraki: {datetime.fromtimestamp(yeni['next_run_at']):%d.%m %H:%M}")
    return n


class Poller(threading.Thread):
    """Arka plan yoklayıcı. Sohbet sunucusuyla aynı süreçte koşar.

    BATCH_MIN_SECONDS: tur bu süreden kısa sürerse fark kadar beklenir — birden
    çok yoklayıcı varken tek örneğin kuyruğu süpürmesini engeller (Macro'nun
    BATCH_MIN_DURATION'ının küçük hâli).
    """

    daemon = True

    def __init__(self, store: ScheduleStore | None = None, on_log=None,
                 aralik: int = POLL_SECONDS):
        super().__init__(name="zamanlayici")
        self.store = store or ScheduleStore()
        self.on_log = on_log
        self.aralik = aralik
        self._dur = threading.Event()

    def run(self):
        while not self._dur.is_set():
            t0 = time.time()
            try:
                tur(self.store, self.on_log)
            except Exception as e:
                if self.on_log:
                    self.on_log(f"[zamanlayıcı] hata: {type(e).__name__}: {e}")
            gecen = time.time() - t0
            if gecen < BATCH_MIN_SECONDS:
                self._dur.wait(BATCH_MIN_SECONDS - gecen)
            self._dur.wait(max(1, self.aralik - max(gecen, BATCH_MIN_SECONDS)))

    def durdur(self):
        self._dur.set()


# ═════════════════════ CLI ═════════════════════

def main():
    ap = argparse.ArgumentParser(description="Zamanlanmış akış koşusu (cron)")
    sub = ap.add_subparsers(dest="komut", required=True)

    e = sub.add_parser("ekle", help="yeni zamanlama")
    e.add_argument("--ad", required=True)
    e.add_argument("--cron", required=True, help='"dk sa gün ay hafta" ör: "0 8 * * 1-5"')
    e.add_argument("--akis", required=True, help="pipelines_store'daki akış id'si")
    e.add_argument("--backend", default="own")
    e.add_argument("--strategy", default="hermes")

    sub.add_parser("liste", help="zamanlamaları göster")
    sub.add_parser("tur", help="tek yoklama turu koştur")

    d = sub.add_parser("sil", help="zamanlama sil");        d.add_argument("id")
    k = sub.add_parser("kapat", help="devre dışı bırak");   k.add_argument("id")
    a = sub.add_parser("ac", help="etkinleştir");           a.add_argument("id")
    s = sub.add_parser("simdi", help="hemen koştur");       s.add_argument("id")
    g = sub.add_parser("gecmis", help="koşu geçmişi");      g.add_argument("id")

    n = sub.add_parser("sonraki", help="cron'un sonraki 5 tetiklemesi")
    n.add_argument("cron")

    p = sub.add_parser("yokla", help="sürekli yoklayıcı çalıştır")
    p.add_argument("--aralik", type=int, default=POLL_SECONDS)

    a_ = ap.parse_args()
    store = ScheduleStore()

    if a_.komut == "ekle":
        sid = store.create(a_.ad, a_.cron, a_.akis, a_.backend, a_.strategy)
        s = store.get(sid)
        print(f"✓ {sid} · {a_.ad} · {cron_aciklama(a_.cron)} · "
              f"sonraki: {datetime.fromtimestamp(s['next_run_at']):%d.%m.%Y %H:%M}")

    elif a_.komut == "liste":
        rows = store.list()
        if not rows:
            print("(zamanlama yok)")
            return
        print(f"{'id':<10}{'ad':<26}{'cron':<16}{'ne zaman':<22}"
              f"{'sonraki':<18}{'durum':<8}akış")
        print("-" * 116)
        for s in rows:
            print(f"{s['id']:<10}{s['name'][:24]:<26}{s['cron']:<16}"
                  f"{cron_aciklama(s['cron'])[:20]:<22}"
                  f"{datetime.fromtimestamp(s['next_run_at']):%d.%m %H:%M}     "
                  f"{('açık' if s['enabled'] else 'KAPALI'):<8}{s['pipeline_id']}"
                  + (f"  (son: {s['last_status']})" if s["last_status"] else ""))

    elif a_.komut == "tur":
        print(f"{tur(store, on_log=print)} zamanlama koştu")

    elif a_.komut == "sil":
        print("✓ silindi" if store.delete(a_.id) else "✗ bulunamadı")

    elif a_.komut in ("kapat", "ac"):
        ok = store.set_enabled(a_.id, a_.komut == "ac")
        print(("✓ etkin" if a_.komut == "ac" else "✓ devre dışı") if ok else "✗ bulunamadı")

    elif a_.komut == "simdi":
        s = store.get(a_.id)
        if not s:
            print("✗ bulunamadı"); return
        if not store.claim(a_.id):
            print("✗ zaten koşuyor (claim alınamadı)"); return
        ok, ozet = kosturt(store, s)
        store.release(a_.id, ok, ileri_al=False)     # elle koşu takvimi kaydırmaz
        print(f"{'✓' if ok else '✗'} {ozet}")

    elif a_.komut == "gecmis":
        for r in store.runs(a_.id):
            bit = datetime.fromtimestamp(r["ended_at"]) if r["ended_at"] else None
            print(f"  {datetime.fromtimestamp(r['started_at']):%d.%m %H:%M:%S}  "
                  f"{'✓' if r['ok'] else '✗' if r['ok'] is not None else '…'}  "
                  f"{r['tamamlanan'] or 0}/{r['dugum'] or 0} düğüm  "
                  f"{r['saniye'] or 0:.1f}s  {r['detay'] or ''}")

    elif a_.komut == "sonraki":
        t = None
        for _ in range(5):
            t = sonraki_kosu(a_.cron, t)
            print(f"  {t:%d.%m.%Y %H:%M}  ({t:%A})")

    elif a_.komut == "yokla":
        print(f"yoklayıcı başladı (aralık {a_.aralik}s, lease {LEASE_SECONDS}s) · Ctrl+C ile çık")
        p = Poller(store, on_log=print, aralik=a_.aralik)
        p.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            p.durdur()
            print("\nyoklayıcı durdu")


if __name__ == "__main__":
    main()
