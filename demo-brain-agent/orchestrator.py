#!/usr/bin/env python3
"""
orchestrator.py — "Ajan task üretir, motor yönetir" akışının tamamı.

    ┌── 1) PLANLAMA TURU ────────────────────────────────────────────┐
    │  Ajana YALNIZCA task tool'ları verilir (create_task/list_tasks) │
    │  LLM hedefi okur → ÇALIŞMA ANINDA task'lar üretir (dinamik),    │
    │  aralarına bağımlılık koyar → board'a yazılır                  │
    └────────────────────────────────────────────────────────────────┘
                                  ↓
    ┌── 2) DISPATCH DÖNGÜSÜ (task-management motoru) ────────────────┐
    │  recompute_ready() → bağımlılığı biten task'lar 'ready'        │
    │  claim_next()      → CAS ile bir task kapılır (at-most-once)   │
    │  execute()         → o task için AJAN koşar (iş tool'ları +    │
    │                      tool-trace compaction), checkpoint yazılır│
    │  complete/fail     → retry hakkı / circuit-breaker             │
    │  recover_stale()   → çöken worker'ın task'ı geri kuyruğa       │
    │  … board'da iş kalmayana kadar                                 │
    └────────────────────────────────────────────────────────────────┘

Backend seçilebilir: own | celery | temporal | airflow.
Backend'in değiştirdiği şey DISPATCH'in nasıl koştuğudur; board hep aynıdır
(Airflow hariç — orada yapısal uyumsuzluk gösterilir).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agent as agent_mod          # noqa: E402
import compaction                  # noqa: E402
from taskboard import (TaskBoard, make_task_tools,  # noqa: E402
                       make_worker_task_tool)

BACKENDS = ("own", "temporal", "celery", "airflow")

# Koçun 6 ekseni — AJANIN ÜRETTİĞİ task'ları yönetme açısından
BACKEND_INFO = {
    "own": {
        "etiket": "Kendi çekirdeğimiz (SQLite board)",
        "task_yonetimi": "ajanın ürettiği task'lar board'da; FSM + DAG kapısı",
        "retry_recovery": "checkpoint'ten devam + otomatik recover_stale + breaker",
        "state_takibi": "SQLite satırı + olay günlüğü (tam görünür)",
        "scheduling": "cron (scheduler.py) — atomik claim + lease + koşu geçmişi",
        "concurrency": "CAS-claim ile çok worker (at-most-once)",
        "operasyonel": "ÇOK DÜŞÜK — tek dosya, ek servis yok",
        "dinamik_task": "TAM — ajan çalışma anında task üretir/bağlar",
    },
    "temporal": {
        "etiket": "Temporal (durable execution)",
        "task_yonetimi": "board bizde; dispatch=workflow, her task=activity",
        "retry_recovery": "activity RetryPolicy + replay (biten activity atlanır)",
        "state_takibi": "durable event-history (tam denetim izi)",
        "scheduling": "Temporal Schedules + backfill",
        "concurrency": "task queue + worker havuzu (yatay)",
        "operasyonel": "YÜKSEK — cluster/Cloud + determinizm disiplini",
        "dinamik_task": "TAM — workflow döngüsü board'u sorar",
    },
    "celery": {
        "etiket": "Celery (dağıtık kuyruk)",
        "task_yonetimi": "board bizde; Celery yalnız dağıtım katmanı",
        "retry_recovery": "self.retry() task'ı BAŞTAN koşar (DAG/checkpoint yok)",
        "state_takibi": "zayıf — board olmasa hiç yok",
        "scheduling": "Celery Beat (telafi/backfill yok)",
        "concurrency": "native worker havuzu (en iyi yatay ölçek)",
        "operasyonel": "ORTA — broker (Redis/RabbitMQ) işletmek",
        "dinamik_task": "KISMİ — dağıtır ama bağımlılığı board yönetir",
    },
    "airflow": {
        "etiket": "Airflow (zamanlı statik DAG)",
        "task_yonetimi": "DAG PARSE ZAMANINDA sabit olmalı",
        "retry_recovery": "sadece hatalı düğüm retry; upstream done kalır",
        "state_takibi": "metadata DB + zengin operatör UI",
        "scheduling": "EN GÜÇLÜ — cron + backfill/catchup",
        "concurrency": "executor havuzu (pool/slot)",
        "operasyonel": "YÜKSEK — scheduler + webserver + metadata DB",
        "dinamik_task": "YOK — graf çalışma anında değişemez (bkz. DAG export)",
    },
}

import functions as F  # noqa: E402

def planner_prompt() -> str:
    """Planlayıcı prompt'u — katalog AKTİF PAKETTEN, ÇAĞRI ANINDA üretilir.
    (Modül import'unda sabitlenirse paket değişince eski katalog kalır.)"""
    return (
        "Sen bir PLANLAYICISIN. Kullanıcının hedefini bir İŞ AKIŞI GRAFI (DAG) olarak kur.\n\n"
        "TEMEL KURAL: Düğümlerin ÇOĞU DETERMİNİSTİK FONKSİYON olmalı (`add_step`). "
        "LLM düğümü (`add_agent_step`) SADECE deterministik fonksiyonla yapılamayacak iş için "
        "kullanılır (yorumlama, karar, serbest metin). Önce hep fonksiyon dene.\n\n"
        f"BU AKIŞTA KULLANABİLECEĞİN FONKSİYONLAR (paket: {F.ACTIVE_PACK} — dışına çıkma):\n"
        + F.catalog_text() +
        "\n\nNASIL:\n"
        "• Her düğüm için `add_step(fn=..., title=..., args_json='{...}', depends_on='id1,id2')`.\n"
        "• Bir düğüm başka düğümün ÇIKTISINA ihtiyaç duyuyorsa `depends_on`'a onun id'sini yaz "
        "(id'yi tool cevabından alırsın). Çıktı otomatik olarak alt düğüme akar.\n"
        "• Bağımsız düğümleri paralel bırak (depends_on boş).\n"
        "• 3-6 düğüm kur. Bitince tool çağırmayı bırak ve tek cümlelik plan özeti yaz."
    )


WORKER_PROMPT = (
    "Sen bir İŞÇİ ajansın. Sana verilen TEK task'ı tamamla. Elindeki tool'ları kullan. "
    "Tool çıktıları çok büyük olabilir; özetlenmiş içerik görebilirsin, elindekiyle ilerle. "
    "Task bitince kısa ve net bir SONUÇ yaz (bulgu + kanıt). En fazla 2 iş-tool çağrısı yap.\n"
    "ÖNEMLİ: Bu task'ı yaparken kapsamı DIŞINDA, ayrı ele alınması gereken yeni bir iş "
    "keşfedersen `spawn_task` tool'uyla onu YENİ BİR TASK olarak aç (kendin yapma). "
    "Sadece gerçekten ayrı bir iş varsa kullan."
)


@dataclass
class OrchestrationResult:
    backend: str
    strategy: str
    ok: bool = False
    plan_trace: list = field(default_factory=list)
    dispatch_log: list = field(default_factory=list)
    tasks: list = field(default_factory=list)
    events: list = field(default_factory=list)
    compaction_events: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    tasks_created: int = 0
    tasks_done: int = 0
    tasks_failed: int = 0
    retries: int = 0
    crashes: int = 0
    recovered: int = 0
    claims_lost: int = 0
    seconds: float = 0.0
    note: str = ""
    dag_file: str = ""
    tasks_spawned_runtime: int = 0
    fn_tasks_run: int = 0
    agent_tasks_run: int = 0
    # kayıtlı akış id'si → board'un yeni id'si. Motorları kıyaslarken her koşu
    # YENİ id üretiyor; arayüzün düğümleri hizalayabilmesi için bu harita şart.
    idmap: dict = field(default_factory=dict)


# ═══════════════════ 1) PLANLAMA TURU — ajan task ÜRETİR ═══════════════════

_MALFORMED_MARKERS = ("<|tool_call", "<|tool_call>", "call:create_task")


def _salvage_tool_calls(content: str, board: TaskBoard, trace: list) -> int:
    """Bazı modeller tool çağrısını yapılandırılmış alan yerine DÜZ METİN olarak yazar
    (ör. `<|tool_call>call:create_task{title:...}`). Bunları kurtarmaya çalış.

    Gerçek hayatta karşılaşılan bir LLM kusuru; ajan buna dayanıklı olmalı.
    """
    import re
    n = 0
    for m in re.finditer(r"call:create_task\{(.*?)(?:\}|$)", content, re.S):
        blob = m.group(1)
        def _f(key):
            mm = re.search(rf"{key}\s*:\s*<\|\"\|>(.*?)<\|\"\|>", blob, re.S)
            if mm:
                return mm.group(1).strip()
            mm = re.search(rf"{key}\s*:\s*([^,}}]+)", blob)
            return mm.group(1).strip().strip('"') if mm else ""
        title = _f("title") or _f("body")[:60]
        if not title:
            continue
        parents = [p.strip() for p in _f("depends_on").split(",") if p.strip()]
        parents = [p for p in parents if board.get(p)]
        try:
            tid = board.create_task(title=title, body=_f("body"), parents=parents,
                                    priority=int(_f("priority") or 5), created_by="agent")
        except Exception:
            continue
        n += 1
        trace.append(f"TASK-TOOL (KURTARILDI: model bozuk format yazmıştı) "
                     f"create_task('{title[:40]}') → {tid} [{board.get(tid)['status']}]")
    return n


def dogrula_dag(board: TaskBoard, log: list | None = None) -> list:
    """Kurulan grafta EKSİK KENAR var mı? Varsa onar, yoksa bildir.

    Neden gerekli: DAG'ın doğruluğu tamamen LLM planlayıcıya bırakılmıştı. Ölçüm
    gösterdi ki planlayıcı, veriyi TÜKETEN düğümü ÜRETEN düğüme bağlamayı düzenli
    olarak atlıyor (ör. cross_check'i scan_patterns yerine fetch_source'a bağlamak).
    Eksik kenar hata vermez — düğüm varsayılanla koşar ve "0 eşleşme" gibi TEMİZ
    ama YANLIŞ bir sonuç üretir. Denetim raporunda bu, "açık bulunamadı" demektir.

    Burada her fonksiyon düğümü için zorunlu upstream anahtarlarının ATALARINDAN
    biri tarafından üretilip üretilmediğine bakıyoruz; üretilmiyorsa graftaki
    üretici düğümü parent olarak EKLİYORUZ (çevrim yaratmıyorsa).
    """
    tasks = {t["id"]: t for t in board.list_tasks()}

    def atalar(tid, gorulen=None):
        gorulen = gorulen if gorulen is not None else set()
        for p in tasks.get(tid, {}).get("parents", []) or []:
            if p not in gorulen:
                gorulen.add(p)
                atalar(p, gorulen)
        return gorulen

    onarim = []
    for tid, t in tasks.items():
        if t.get("kind") != "function":
            continue
        gerekli = F.NEEDS.get(t.get("fn") or "", ())
        if not gerekli:
            continue
        # DİKKAT: veri yalnız DOĞRUDAN parent'lardan akar (upstream_results), ata
        # kapanışından değil. Doğrulamayı da buna göre yapmak zorundayız; ataya
        # bakmak "veri var" sanıp eksik kenarı gözden kaçırır (dede'nin çıktısı
        # torununa OTOMATİK geçmez).
        uretilen = {k for p in (t["parents"] or [])
                    for k in F._URETILEN.get(tasks.get(p, {}).get("fn") or "", ())}
        for anahtar in gerekli:
            if anahtar in uretilen:
                continue
            aday = next((oid for oid, o in tasks.items()
                         if oid != tid and anahtar in F._URETILEN.get(o.get("fn") or "", ())
                         and tid not in atalar(oid)), None)   # çevrim koruması
            if aday is None:
                onarim.append(f"✗ {t['title'][:34]} ({t['fn']}) '{anahtar}' verisini "
                              f"bekliyor ama grafta ÜRETEN düğüm YOK")
                continue
            yeni = list(dict.fromkeys(list(t["parents"]) + [aday]))
            board.conn.execute("UPDATE tasks SET parents=? WHERE id=?",
                               (json.dumps(yeni, ensure_ascii=False), tid))
            board.conn.commit()
            t["parents"] = yeni
            uretilen |= set(F._URETILEN.get(tasks[aday].get("fn") or "", ()))
            onarim.append(f"⚠ EKSİK KENAR ONARILDI: {t['title'][:30]} ({t['fn']}) "
                          f"← {tasks[aday]['title'][:30]} ({tasks[aday]['fn']}) "
                          f"[gereken veri: {anahtar}]")
    if onarim:
        # kenar eklendi → 'ready' doğmuş düğümler yeniden 'blocked' olmalı
        for tid, t in tasks.items():
            if t["parents"] and board.get(tid)["status"] == "ready":
                board.conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (tid,))
        board.conn.commit()
        if log is not None:
            log.extend(onarim)
    return onarim


def plan_phase(board: TaskBoard, goal: str, max_calls: int = 6,
               on_event=None) -> list:
    """Ajana yalnız task tool'larını ver; hedefi task'lara böldür (dinamik üretim)."""
    trace: list = []

    class _L(list):
        def append(self, x):
            super().append(x)
            if on_event:
                on_event({"type": "node_added", "text": str(x)})
    trace = _L()
    tools = make_task_tools(board, log=trace)
    specs = []
    for name, (_fn, desc, props, req) in tools.items():
        specs.append({"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}})

    msgs = [{"role": "system", "content": planner_prompt()},
            {"role": "user", "content": f"Hedef: {goal}"}]

    for _ in range(max_calls):
        r = agent_mod.llm.chat(msgs, tools=specs, max_tokens=900, temperature=0.2)
        tcs = r.get("tool_calls") or []
        content = (r.get("content") or "").strip()
        msgs.append({"role": "assistant", "content": content, "tool_calls": tcs})
        if not tcs:
            # model tool çağrısını düz metne sızdırmış olabilir → kurtarmayı dene
            if any(mk in content for mk in _MALFORMED_MARKERS):
                _salvage_tool_calls(content, board, trace)
                content = content.split("<|tool_call")[0].strip()
            if content:
                trace.append(f"PLAN özeti: {content[:200]}")
            break
        for tc in tcs:
            fname = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except Exception:
                args = {}
            fn = tools.get(fname, (None,))[0]
            out = fn(**args) if fn else f"[hata] bilinmeyen tool {fname}"
            msgs.append({"role": "tool", "name": fname,
                         "tool_call_id": tc.get("id", ""), "content": str(out)})
    return trace


# ═══════════════════ 2) TEK TASK'I YÜRÜTME — işçi ajan ═══════════════════

class WorkerCrash(Exception):
    """Worker'ın task ortasında ölmesi (simülasyon)."""


def execute_task(task: dict, strategy: str, budget: int, max_turns: int = 3,
                 fail_at: str | None = None, attempt: int = 0,
                 board: TaskBoard | None = None, spawn_log: list | None = None,
                 crash_after_turn: int | None = None, log: list | None = None):
    """Bir task'ı işçi ajanla koştur — HER TURDAN SONRA checkpoint yazarak.

    Checkpoint sayesinde worker task'ın ORTASINDA çökse bile, task'ı devralan
    worker kaldığı turdan devam eder (baştan başlamaz).

    board verilirse işçi ajana `spawn_task` tool'u da verilir → ajan iş yaparken
    YENİ İŞ keşfedip task açabilir (yürütme anında replanlama).
    """
    job = agent_mod.BrainAgentJob(
        goal=f"{task['title']}\n\nDetay: {task.get('body') or '(yok)'}",
        strategy=strategy, budget=budget, job_id=task["id"], max_turns=max_turns)

    state = task.get("checkpoint") or {}
    resumed_from = 0
    if state.get("messages"):
        resumed_from = state.get("turn", 0)
        if log is not None and resumed_from:
            log.append(f"          ↻ checkpoint'ten DEVAM: turn{resumed_from}'e kadar olan iş "
                       f"TEKRAR KOŞMAYACAK ({len(state['messages'])} mesaj geri yüklendi)")
    else:
        state = job.initial_state()
        state["messages"][0]["content"] = WORKER_PROMPT      # işçi persona'sı

    if board is not None:
        agent_mod.EXTRA_TOOLS = make_worker_task_tool(board, task["id"], log=spawn_log)
    try:
        done = False
        while not done:
            done, state, _label = job.step(state, attempt=attempt, fail_at=fail_at)
            # --- TUR-İÇİ DURABLE CHECKPOINT (asıl "kaldığı yerden devam"ı sağlayan şey) ---
            if board is not None:
                board.save_checkpoint(task["id"], state)
                board.heartbeat(task["id"], task.get("claim_lock") or "worker")
            turn = state.get("turn", 0)
            if log is not None:
                log.append(f"          {_label} → checkpoint yazıldı")
            # --- çökme simülasyonu: tur bitti, checkpoint YAZILDI, sonra worker ölür ---
            if crash_after_turn is not None and turn >= crash_after_turn and not done:
                raise WorkerCrash(f"turn{turn - 1} sonrası worker öldü")
    finally:
        agent_mod.EXTRA_TOOLS = {}                            # yetkiyi geri al
    return state


# ═══════════════════ DÜĞÜM BAZINDA SİMÜLASYON ═══════════════════
# fail_at/crash_at tek bir GLOBAL string ve fonksiyon adı ya da başlık ALT-DİZESİYLE
# eşleşiyor. Aynı fn iki düğümde kullanılırsa ikisi birden patlıyor; "sadece 3. düğüm
# patlasın" demenin yolu yok. node_sim bunu id bazlı çözer:
#
#     node_sim = {"t_a1b2c3": {"mod": "kalici"}, "t_d4e5": {"mod": "yavas", "sn": 5}}
#
# Anahtar = board'daki GÜNCEL task id'si. Kayıtlı akıştan gelen eski id'ler
# materialize() içinde yeni id'lere çevrilir (idmap zaten orada üretiliyor).

SIM_MODLARI = ("normal", "gecici", "kalici", "sonra", "cokme", "yavas")

SIM_ACIKLAMA = {
    "normal": "değişiklik yok",
    "gecici": "ilk denemede patlar, retry'da geçer",
    "kalici": "her denemede patlar → breaker → failed + ardıllar cancelled",
    "sonra":  "iş YAPILDIKTAN SONRA patlar (yan etki oluşur, sonuç yazılmaz)",
    "cokme":  "iş biter, checkpoint yazılır, complete çağrılmadan worker ölür",
    "yavas":  "N saniye bekler (lease/timeout gözlemi)",
}


def _sim_of(task: dict, node_sim: dict | None) -> dict:
    """Bu düğüm için simülasyon ayarı. Yoksa boş sözlük."""
    if not node_sim:
        return {}
    s = node_sim.get(task["id"]) or {}
    return s if isinstance(s, dict) else {}


def cevir_node_sim(node_sim: dict | None, idmap: dict) -> dict:
    """Kayıtlı akış id'lerini board'un yeni id'lerine çevir.

    Kullanıcı arayüzde kayıtlı düğüm id'sine tıklıyor; materialize() ise her koşuda
    YENİ id üretiyor. Çeviri olmazsa ayar hiçbir düğüme denk gelmez (sessizce hiçbir
    şey olmaz) — bu, en kolay gözden kaçacak hata olurdu.
    """
    if not node_sim:
        return {}
    out = {}
    for eski, ayar in node_sim.items():
        yeni = idmap.get(eski, eski)          # zaten yeni id verilmişse aynen geçir
        out[yeni] = ayar
    return out


# ═══════════════════ ORTAK YÜRÜTÜCÜ — kind'a göre yönlendirir ═══════════════════
# Dört backend de (own/temporal/celery/airflow) bunu çağırır: tek doğruluk kaynağı.

def run_one_task(task: dict, board: TaskBoard, strategy: str = "hermes",
                 budget: int = 3000, fail_at: str | None = None, attempt: int = 0,
                 crash_after_turn: int | None = None, log: list | None = None,
                 spawn_log: list | None = None, on_event=None,
                 node_sim: dict | None = None) -> tuple[str, str, dict]:
    """Tek bir düğümü yürüt. Dönüş: (kind, result_json, extra)

    kind='function' → LLM YOK, kayıtlı fonksiyon çağrılır (upstream verisiyle)
    kind='agent'    → işçi ajan koşar (LLM), upstream verisi prompt'a eklenir
    """
    up = board.upstream_results(task["id"])

    if task.get("kind") == "function":
        # HATA ENJEKSİYONU — fonksiyon düğümü için.
        # fail_at yalnız AJAN düğümlerinde çalışıyordu; fonksiyon-öncelikli mimaride
        # düğümlerin çoğu fonksiyon olduğundan hata/retry/breaker yolu hiç tetiklenemiyordu.
        #   fail_at="scan_patterns"   → GEÇİCİ hata (ilk denemede patlar, retry'da geçer)
        #   fail_at="scan_patterns!"  → KALICI hata (her denemede patlar → breaker → failed)
        #   fail_at="scan_patterns"   → GEÇİCİ, iş YAPILMADAN patlar (temiz hata)
        #   fail_at="scan_patterns!"  → KALICI  (her denemede) → breaker → failed
        #   fail_at="scan_patterns~"  → iş YAPILDIKTAN SONRA patlar → YAN ETKİ oluşur,
        #                               retry işi baştan koşturur (yan etki tekrarlanır)
        _vur = None
        _sim = _sim_of(task, node_sim)
        _mod = _sim.get("mod", "normal")

        if _mod != "normal":
            # ── DÜĞÜM BAZLI (id ile eşleşir, fn adı/başlık değil) ──
            if _mod == "yavas":
                _sn = float(_sim.get("sn", 3))
                if log is not None:
                    log.append(f"          ⏳ simülasyon: {task['fn']}() {_sn} sn bekliyor")
                time.sleep(_sn)
            elif _mod in ("gecici", "kalici", "sonra"):
                _kalici = _mod == "kalici"
                _sonra = _mod == "sonra"
                if _kalici or attempt == 0:
                    _vur = RuntimeError(
                        f"[simülasyon:{_mod}] {task['fn']}() başarısız "
                        f"(attempt={attempt}, düğüm={task['id']}, "
                        f"iş {'YAPILDIKTAN SONRA' if _sonra else 'YAPILMADAN'})")
                    if not _sonra:
                        raise _vur
            # "cokme" aşağıda, iş bittikten sonra ele alınıyor
        elif fail_at:
            # ── ESKİ YOL (geriye dönük uyum) — fn adı ya da başlık alt-dizesi ──
            kalici = "!" in fail_at
            sonra = "~" in fail_at
            hedef = fail_at.replace("!", "").replace("~", "")
            if hedef and (hedef == task["fn"] or hedef.lower() in task["title"].lower()) \
                    and (kalici or attempt == 0):
                _vur = RuntimeError(
                    f"{'kalıcı' if kalici else 'geçici'} hata: {task['fn']}() başarısız "
                    f"(attempt={attempt}, iş {'YAPILDIKTAN SONRA' if sonra else 'YAPILMADAN'})")
                if not sonra:
                    raise _vur
        # ── CHECKPOINT'TEN DEVAM (fonksiyon düğümü) ──
        # Önceki deneme fonksiyonu KOŞTURMUŞ ve sonucu checkpoint'e yazmış, ama
        # complete() çağrılmadan worker ölmüştü. Sonuç elimizde duruyorken işi
        # yeniden koşturmak boşa iş — ve YAN ETKİLİ bir fonksiyonda işi İKİNCİ KEZ
        # yapmak demek (e-posta iki kez, ödeme iki kez). Varsa yeniden kullan.
        _ck = task.get("checkpoint") or {}
        _ck_devam = isinstance(_ck, dict) and "fn_sonucu" in _ck and _vur is None
        if _ck_devam:
            out = _ck["fn_sonucu"]
            if log is not None:
                log.append(f"          ↻ checkpoint'ten DEVAM: {task['fn']}() ZATEN "
                           f"koşmuştu, sonucu geri yüklendi (tekrar KOŞTURULMADI)")
        else:
            out = F.call(task["fn"], task.get("fn_args") or {}, up)
        if _vur is not None:              # iş bitti, sonucu yazmadan patla
            raise _vur
        # ÇÖKME SİMÜLASYONU — fonksiyon düğümünde de geçerli:
        # iş yapıldı, sonuç checkpoint'e yazıldı, ama complete() ÇAĞRILMADAN worker öldü.
        # (Önceden yalnız ajan düğümlerinde çalışıyordu → fonksiyon-öncelikli mimaride
        #  çökme/kurtarma yolu hiç test edilemiyordu.)
        # node_sim "cokme" modu da buraya düşer — böylece çökme DÖRT motorda da
        # aynı yoldan geçer (crash_at yalnız own'da çalışıyordu, celery/temporal
        # onu sessizce yok sayıyordu).
        #
        # TEK ATIŞ: "worker bir kez öldü" demek. `_ck_devam` doğruysa bu zaten
        # çökme SONRASI devralan koşu — tekrar çökersek düğüm sonsuza dek ölür
        # (ölçüldü: own'da 10 çökme üst üste, celery/temporal'da düğüm asılı kaldı).
        # Checkpoint'in varlığı süreçler arası geçerli bir "bir kez çöktü" işareti.
        if (crash_after_turn is not None or _mod == "cokme") and not _ck_devam:
            board.save_checkpoint(task["id"], {"fn_sonucu": out})
            if log is not None:
                log.append(f"          fn {task['fn']}() koştu, sonuç checkpoint'e yazıldı")
            raise WorkerCrash(f"{task['fn']}() sonrası worker öldü")
        # DETERMİNİSTİK İNDİRGEME: fonksiyon ham veriyi yapılandırılmış sonuca çevirir
        _raw = F.raw_chars(out)
        _res_chars = len(json.dumps(out, ensure_ascii=False, default=str))
        if _raw and on_event:
            # kaynak="akis": bu düğümün çıktısı BOARD'a ve ardıl düğümlere gider,
            # sohbet asistanının context'ine GİRMEZ. Tool-trace paneli yalnız
            # kaynak="sohbet" olanları listeler (bkz. chat_server.py).
            on_event({"type": "reduction", "kaynak": "akis",
                      "task": task["id"], "fn": task["fn"],
                      "title": task["title"][:40], "args": task.get("fn_args") or {},
                      "raw_tokens": _raw // 4, "out_tokens": _res_chars // 4,
                      "pct": round((1 - _res_chars / _raw) * 100, 1) if _raw else 0,
                      "result": json.dumps(
                          {k: v for k, v in out.items()
                           if k not in ("_raw_chars", "rows", "text")},
                          ensure_ascii=False, indent=1)[:2500]})
        if log is not None:
            ozet = {k: v for k, v in out.items() if k not in ("text", "rapor_md")}
            log.append(f"          fn {task['fn']}({task.get('fn_args') or {}}) "
                       f"← upstream {list(up) or '—'}")
            log.append(f"          → {str(ozet)[:110]}")
        return "function", json.dumps(out, ensure_ascii=False, default=str), {}

    # --- ajan düğümü (istisna) ---
    t = dict(task)
    if up:
        # Upstream sonuçları LLM CONTEXT'ine giriyor → bu da tool-trace'tir.
        # Sabit kesme yerine SEÇİLEN compaction stratejisinden geçir.
        _up_msgs = [{"role": "tool", "name": (v.get("_fn") or k),
                     "content": json.dumps(v, ensure_ascii=False, default=str)}
                    for k, v in up.items()]
        _r = compaction.compact(strategy, _up_msgs, budget=max(budget // 2, 400))
        if _r.triggered and on_event:
            on_event({"type": "compaction", "kaynak": "akis", "task": task["id"],
                      "title": f"upstream→{task['title'][:28]}",
                      "events": [{"strategy": _r.strategy, "before": _r.before,
                                  "after": _r.after, "saved": _r.saved,
                                  "pct": round(_r.pct, 1), "triggered": True,
                                  "log": _r.log[:3]}]})
        t["body"] = (t.get("body") or "") + "\n\nÖnceki adımların çıktısı:\n" + \
            "\n".join(compaction._View(m).content for m in _r.messages)
    state = execute_task(t, strategy, budget, fail_at=fail_at, attempt=attempt,
                         board=board, spawn_log=spawn_log,
                         crash_after_turn=crash_after_turn, log=log)
    return "agent", state.get("answer", ""), state


# ═══════════════════ 3) DISPATCH — backend'e göre ═══════════════════

def _dispatch_own(board: TaskBoard, res: OrchestrationResult, strategy: str, budget: int,
                  crash_at: str | None, fail_at: str | None, max_rounds: int = 12,
                  on_event=None, node_sim: dict | None = None):
    """Kendi durable motorumuz: recompute_ready → claim → execute → complete/fail."""
    crashed_once = False

    class _L(list):
        def append(self, x):
            super().append(x)
            if on_event:
                on_event({"type": "log", "text": str(x)})
    if not isinstance(res.dispatch_log, _L):
        _n = _L(res.dispatch_log)
        res.dispatch_log = _n

    for rnd in range(1, max_rounds + 1):
        promoted = board.recompute_ready()
        if promoted:
            res.dispatch_log.append(f"[tur {rnd}] recompute_ready → {promoted} task "
                                    f"blocked→ready (bağımlılık kapısı açıldı)")
        if board.all_settled():
            break
        task = board.claim_next(f"worker-{rnd}")
        if task is None:
            n = board.recover_stale(force=True)
            if n:
                res.recovered += n
                res.dispatch_log.append(f"[tur {rnd}] recover_stale → {n} takılı task geri kuyruğa")
                continue
            break
        res.dispatch_log.append(
            f"[tur {rnd}] claim: {task['id']} '{task['title'][:44]}' (CAS+lease)")

        # çökmeyi task'ın ORTASINDA tetikle (1 tur bitip checkpoint yazıldıktan sonra)
        # Düğüm bazlı "cokme" ayarı run_one_task içinde ele alınıyor; buradaki
        # crash_at eski (başlık alt-dizesi) yolu, geriye dönük uyum için duruyor.
        crash_turn = None
        if crash_at and not crashed_once and crash_at.lower() in task["title"].lower():
            crash_turn = 1
        elif _sim_of(task, node_sim).get("mod") == "cokme" and not (task.get("checkpoint") or {}):
            crash_turn = 1                     # ajan düğümünde tur ortasında çökme
            # boş checkpoint şartı = TEK ATIŞ; devralan koşuda tekrar çökmesin

        try:
            att = task.get("attempt", 0)
            kind, result, extra = run_one_task(
                task, board, strategy, budget, fail_at=fail_at, attempt=att,
                crash_after_turn=crash_turn, log=res.dispatch_log,
                spawn_log=res.dispatch_log, on_event=on_event, node_sim=node_sim)
            if not board.complete(task["id"], result, claimer=task.get("claim_lock")):
                # lease dolmuş, task devredilmiş → bu worker'ın yazması REDDEDİLDİ
                res.dispatch_log.append("          ⚠ yazma reddedildi: claim devredilmiş "
                                        "(bayat worker sonucu ezemez)")
                continue
            res.tasks_done += 1
            if kind == "function":
                res.fn_tasks_run += 1
                res.dispatch_log.append("          ✓ tamamlandı → done (LLM kullanılmadı)")
            else:
                res.agent_tasks_run += 1
                _ce = extra.get("compaction_events", [])
                res.compaction_events += _ce
                if on_event and _ce:
                    on_event({"type": "compaction", "kaynak": "akis", "task": task["id"],
                              "title": task["title"][:40], "events": _ce})
                res.dispatch_log.append("          ✓ tamamlandı → done (LLM ajan)")
        except WorkerCrash as c:
            crashed_once = True
            res.crashes += 1
            res.dispatch_log.append(f"          ✗ ÇÖKME: {c} (claim açık kaldı → stale; "
                                    f"complete çağrılmadı)")
            n = board.recover_stale(force=True)
            res.recovered += n
            res.dispatch_log.append(f"          recover_stale → {n} task 'ready'; "
                                    f"checkpoint DURUYOR → devralan worker kaldığı turdan sürer")
            continue
        except Exception as e:
            # SÖZLEŞME hatası (bilinmeyen fn / geçersiz argüman) tekrar denemekle
            # düzelmez → breaker'ı 3 kez boşuna harcamak yerine tek denemede kapat.
            kalici = isinstance(e, (ValueError, TypeError, KeyError))
            st = board.fail(task["id"], str(e), claimer=task.get("claim_lock"),
                            permanent=kalici)
            res.retries += 1
            res.dispatch_log.append(
                f"          ✗ hata: {str(e)[:70]} → status={st} "
                + ("(KALICI sözleşme hatası — retry edilmedi)" if kalici
                   else "(attempt++, breaker++)"))
            if st == "failed":
                iptal = [board.get(i) for i in getattr(board, "last_cancelled", [])]
                if iptal:
                    res.dispatch_log.append(
                        f"          ⛔ BATAN DAL KAPATILDI: {len(iptal)} ardıl düğüm "
                        f"'cancelled' — bunlar HİÇ koşmayacak: "
                        + ", ".join(f"{t['id']}({t['title'][:22]})" for t in iptal[:4]))

    # ── KOŞU SONU DENETİMİ ──
    # Döngü 'break' ile bittiğinde geride yürütülmemiş düğüm kalmış olabilir
    # (tur limiti doldu, ya da bağımlılık kapısı hiç açılmadı). Sessizce bitirmek
    # operatöre "akış tamamlandı" izlenimi verir; ASILI KALANI açıkça söyle.
    asili = board.list_tasks("blocked") + board.list_tasks("ready")
    if asili:
        res.dispatch_log.append(
            f"⚠ KOŞU BİTTİ ama {len(asili)} düğüm YÜRÜTÜLMEDİ (bağımlılığı hiç "
            f"açılmadı ya da tur limiti doldu): "
            + ", ".join(f"{t['id']}({t['title'][:24]}·{t['status']})" for t in asili[:5]))
        if on_event:
            on_event({"type": "log", "text": res.dispatch_log[-1]})


def _dispatch_celery(board: TaskBoard, res: OrchestrationResult, strategy: str, budget: int,
                     crash_at: str | None, fail_at: str | None, on_event=None,
                     node_sim: dict | None = None):
    """Celery: her ready task broker'a atılır, ayrı worker süreci çeker.

    NOT: Celery'nin kendi 'task' kavramı bizim A-seviyesi task'ımızla birebir örtüşür,
    ama BAĞIMLILIK (DAG) ve checkpoint yönetimi Celery'de YOKTUR → onları yine biz
    (board) yönetiriz. Celery sadece dağıtım/retry katmanıdır.
    """
    import os
    import subprocess
    import tempfile
    poc_dir = Path(tempfile.mkdtemp(prefix="brain_orc_celery_"))
    env = {**os.environ, "BRAIN_CELERY_DIR": str(poc_dir),
           "BRAIN_BOARD_DB": str(board.path),
           "BRAIN_STRATEGY": strategy, "BRAIN_BUDGET": str(budget),
           "BRAIN_FAIL_AT": fail_at or "",
           # düğüm bazlı simülasyon worker sürecine JSON olarak taşınır
           "BRAIN_NODE_SIM": json.dumps(node_sim or {}, ensure_ascii=False),
           "PYTHONPATH": f"{HERE}:{os.environ.get('PYTHONPATH','')}"}
    res.dispatch_log.append(f"gerçek celery worker (ayrı süreç) + filesystem broker · {poc_dir}")
    worker = subprocess.Popen(
        [sys.executable, "-m", "celery", "-A", "celery_worker", "worker",
         "--pool=solo", "--concurrency=1", "--loglevel=ERROR",
         "--without-gossip", "--without-mingle", "--without-heartbeat"],
        cwd=str(HERE), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        os.environ.update({k: env[k] for k in
                           ("BRAIN_CELERY_DIR", "BRAIN_BOARD_DB", "BRAIN_STRATEGY",
                            "BRAIN_BUDGET", "BRAIN_FAIL_AT", "BRAIN_NODE_SIM")})
        import importlib
        import celery_worker
        importlib.reload(celery_worker)
        time.sleep(6)

        for rnd in range(1, 13):
            promoted = board.recompute_ready()
            if promoted:
                res.dispatch_log.append(f"[tur {rnd}] recompute_ready → {promoted} task ready")
            if board.all_settled():
                break
            ready = board.list_tasks("ready")
            if not ready:
                if board.list_tasks("running"):
                    time.sleep(2)
                    continue
                break
            for t in ready:
                celery_worker.run_task.delay(t["id"])
                res.dispatch_log.append(f"[tur {rnd}] kuyruğa atıldı: {t['id']} "
                                        f"'{t['title'][:40]}' → run_task.delay()")
            # worker'ların bitirmesini bekle — SSE sessiz kalmasın diye NABIZ yayınla
            deadline = time.time() + 90
            son_ilerleme = time.time()
            onceki = len(board.list_tasks("done"))
            while time.time() < deadline and not board.all_settled():
                time.sleep(1.5)
                # bu DALGA bitti mi? (kuyruğa atılanların hepsi tüketildi)
                # → dış döngü recompute_ready yapıp bir sonraki dalgayı kuyruklasın
                if not board.list_tasks("running") and not board.list_tasks("ready"):
                    break
                simdi = len(board.list_tasks("done"))
                if simdi != onceki:
                    onceki, son_ilerleme = simdi, time.time()
                    if on_event:
                        on_event({"type": "log",
                                  "text": f"[tur {rnd}] worker ilerledi: {simdi} düğüm tamamlandı"})
                elif time.time() - son_ilerleme > 25:
                    # 25 sn'dir ilerleme yok → ÖNCE bayat claim'leri süpür.
                    # Worker AYRI SÜREÇTE çöktüyse (WorkerCrash) task 'running'
                    # asılı kalır ve Celery'nin bundan haberi olmaz — kimse
                    # kurtarmazsa akış orada ölür (ölçüldü). own bunu WorkerCrash'i
                    # yakalayıp yapıyor; celery'de sweep dispatcher'a düşüyor.
                    n = board.recover_stale(force=True)
                    if n:
                        res.recovered += n
                        res.crashes += n
                        if on_event:
                            on_event({"type": "log",
                                      "text": f"[tur {rnd}] ⚠ {n} takılı düğüm bulundu → "
                                              f"recover_stale ile geri kuyruğa (çöken worker)"})
                        son_ilerleme = time.time()
                        for t in board.list_tasks("ready"):
                            celery_worker.run_task.delay(t["id"])
                        continue
                    if on_event:
                        on_event({"type": "log",
                                  "text": f"[tur {rnd}] ⚠ 25 sn ilerleme yok — worker takıldı, "
                                          f"bekleme sonlandırılıyor"})
                    break
                elif on_event and int(time.time()) % 10 == 0:
                    on_event({"type": "log", "text": f"[tur {rnd}] worker bekleniyor…"})
        res.tasks_done = len(board.list_tasks("done"))
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except Exception:
            worker.kill()
        import shutil
        for junk in (HERE / "control", HERE / "__pycache__"):
            shutil.rmtree(junk, ignore_errors=True)


def _dispatch_temporal(board: TaskBoard, res: OrchestrationResult, strategy: str, budget: int,
                       crash_at: str | None, fail_at: str | None, on_event=None,
                       node_sim: dict | None = None):
    """Temporal: dispatch bir workflow, her task yürütmesi bir activity.

    Board yine bizim (bağımlılık + durum), Temporal ise durable yürütme motoru:
    activity başarısız olursa RetryPolicy devreye girer, workflow çökse replay eder.
    """
    import asyncio
    import logging
    logging.getLogger("temporalio").setLevel(logging.CRITICAL)
    from temporalio.worker import Worker
    from temporalio.testing import WorkflowEnvironment
    import temporal_defs

    temporal_defs.CTX.update({"board": board, "strategy": strategy, "budget": budget,
                              "fail_at": fail_at, "node_sim": node_sim or {},
                              "compaction_events": []})

    async def _go():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(env.client, task_queue="brain-board",
                              workflows=[temporal_defs.BoardWorkflow],
                              activities=[temporal_defs.execute_one_task,
                                          temporal_defs.next_ready_task]):
                h = await env.client.start_workflow(
                    temporal_defs.BoardWorkflow.run, "go",
                    id=f"board-{int(time.time())}", task_queue="brain-board")
                out = await h.result()
                hist = await h.fetch_history()
                return json.loads(out), len(hist.events)

    res.dispatch_log.append("gerçek Temporal dev server başlatılıyor (ephemeral)…")
    d, n_events = asyncio.run(_go())
    res.dispatch_log += d.get("log", [])
    res.dispatch_log.append(f"✓ dispatch workflow bitti · {n_events} durable event")
    res.compaction_events += temporal_defs.CTX.get("compaction_events", [])
    res.tasks_done = len(board.list_tasks("done"))


def export_airflow_dag(board: TaskBoard, out_path: Path | None = None,
                       dag_id: str = "brain_agent_plan", goal: str = "",
                       node_sim: dict | None = None, retry_delay_sn: int = 30,
                       schedule: str | None = "0 8 * * *") -> Path:
    """Ajanın kurduğu grafı GERÇEK bir Airflow DAG'ına çevir.

    Artık düğümler DETERMİNİSTİK FONKSİYON olduğu için çeviri BİREBİR:
      board düğümü  → PythonOperator
      parents       → `>>` bağımlılığı
      upstream veri → XCom (`ti.xcom_pull`)

    Bedeli değişmedi: DAG DONMUŞTUR — yürütme sırasında ajan yeni düğüm ekleyemez.
    """
    tasks = board.list_tasks()
    out_path = out_path or (HERE / f"{dag_id}.py")

    def _pyid(tid: str) -> str:
        return tid.replace("-", "_")

    L = [
        '"""OTOMATİK ÜRETİLDİ — ajanın planlama turunda kurduğu iş akışı grafı.',
        "",
        f"Hedef: {goal or '(belirtilmedi)'}",
        f"Üretim anı: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Düğümler `functions.py` kaydındaki DETERMİNİSTİK fonksiyonlardır;",
        "düğümler arası veri Airflow XCom ile akar (board'daki upstream_results karşılığı).",
        "",
        "DİKKAT: Bu DAG bir ANLIK GÖRÜNTÜdür — yürütme sırasında graf büyüyemez.",
        "",
        "Kurulum:",
        '    pip install "apache-airflow==2.10.*" && airflow db migrate',
        f"    cp {out_path.name} $AIRFLOW_HOME/dags/",
        f"    airflow dags test {dag_id} {time.strftime('%Y-%m-%d')}",
        '"""',
        "from datetime import datetime, timedelta",
        "from airflow import DAG",
        "from airflow.operators.python import PythonOperator",
        "",
        "default_args = {",
        '    "retries": 2,                       # DÜĞÜM bazında retry',
        f'    "retry_delay": timedelta(seconds={int(retry_delay_sn)}),',
        "}",
        "",
        "# ── DÜĞÜM BAZLI SİMÜLASYON ──",
        "# Airflow AYRI SÜREÇTE koşuyor; ortak bellek yok. Panelden gelen ayarlar",
        "# DAG dosyasına GÖMÜLEREK taşınıyor (celery'de env, temporal'da CTX ile aynı iş).",
        f"NODE_SIM = {dict(node_sim or {})!r}",
        "_DENEME = {}",
        "",
        "",
        "def _sim_uygula(task_id, asama):",
        '    """asama: \'once\' (iş yapılmadan) | \'sonra\' (iş bittikten sonra)."""',
        "    ayar = NODE_SIM.get(task_id) or {}",
        "    mod = ayar.get('mod', 'normal')",
        "    if mod == 'normal':",
        "        return",
        "    if mod == 'yavas':",
        "        if asama == 'once':",
        "            import time as _t; _t.sleep(float(ayar.get('sn', 3)))",
        "        return",
        "    if asama == 'once':",
        "        _DENEME[task_id] = _DENEME.get(task_id, 0) + 1",
        "    n = _DENEME.get(task_id, 1)",
        "    if mod == 'gecici' and asama == 'once' and n == 1:",
        "        raise RuntimeError(f'[simulasyon:gecici] {task_id} (deneme {n})')",
        "    if mod in ('kalici', 'cokme') and asama == 'once':",
        "        # Airflow'da 'çökme' ayrı bir kavram değil: düğüm ölür, retry devralır",
        "        raise RuntimeError(f'[simulasyon:{mod}] {task_id} (deneme {n})')",
        "    if mod == 'sonra' and asama == 'sonra' and n == 1:",
        "        raise RuntimeError(f'[simulasyon:sonra] {task_id} - is yapildi, sonuc yazilmadi')",
        "",
        "",
        "def _run_fn(fn_name, args, parent_ids, **ctx):",
        '    """Kayıtlı deterministik fonksiyonu koştur; upstream veriyi XCom\'dan al."""',
        "    from functions import call",
        "    _sim_uygula(ctx['ti'].task_id, 'once')",
        "    ti = ctx[\"ti\"]",
        "    up = {p: ti.xcom_pull(task_ids=p) for p in parent_ids}",
        "    up = {k: v for k, v in up.items() if v is not None}",
        "    _sonuc = call(fn_name, args, up)",
        "    _sim_uygula(ctx['ti'].task_id, 'sonra')",
        "    return _sonuc",
        "",
        "",
        "def _run_agent(title, body, parent_ids, **ctx):",
        '    """LLM düğümü (istisna): işçi ajanı koştur."""',
        "    from agent import BrainAgentJob",
        "    ti = ctx[\"ti\"]",
        "    up = {p: ti.xcom_pull(task_ids=p) for p in parent_ids}",
        "    job = BrainAgentJob(goal=f\"{title}\\n\\nDetay: {body}\\n\\nUpstream: {up}\",",
        '                        strategy="hermes", budget=3000, max_turns=2)',
        '    return job.run_sync()["answer"]',
        "",
        "",
        "with DAG(",
        f'    dag_id="{dag_id}",',
        "    default_args=default_args,",
        (f'    schedule={schedule!r},               # Airflow\'un asıl gücü: cron'
         if schedule else "    schedule=None,"),
        "    start_date=datetime(2026, 1, 1),",
        "    catchup=False,",
        "    max_active_runs=1,",
        '    tags=["brain", "agent", "otomatik-uretilmis"],',
        ") as dag:",
        "",
    ]
    for t in tasks:
        pid = _pyid(t["id"])
        parents = [_pyid(x) for x in t["parents"]]
        if t["kind"] == "function":
            L.append(f"    # {t['id']} · DETERMİNİSTİK · fn={t['fn']}")
            L.append(f"    {pid} = PythonOperator(")
            L.append(f"        task_id={t['id']!r},")
            L.append("        python_callable=_run_fn,")
            L.append(f"        op_kwargs={{'fn_name': {t['fn']!r}, "
                     f"'args': {t.get('fn_args') or {}!r}, "
                     f"'parent_ids': {t['parents']!r}}},")
            L.append("    )")
        else:
            L.append(f"    # {t['id']} · LLM AJAN düğümü")
            L.append(f"    {pid} = PythonOperator(")
            L.append(f"        task_id={t['id']!r},")
            L.append("        python_callable=_run_agent,")
            L.append(f"        op_kwargs={{'title': {t['title']!r}, "
                     f"'body': {(t.get('body') or '')[:200]!r}, "
                     f"'parent_ids': {t['parents']!r}}},")
            L.append("    )")
    L.append("")
    L.append("    # ajanın kurduğu bağımlılıklar (parents → child)")
    any_dep = False
    for t in tasks:
        for par in t["parents"]:
            L.append(f"    {_pyid(par)} >> {_pyid(t['id'])}")
            any_dep = True
    if not any_dep:
        L.append("    # (bağımsız düğümler — hepsi paralel koşar)")
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path


_AIRFLOW_NOTE = """Airflow bu akışı YAPISAL OLARAK kaldıramaz — demonun en net bulgusu:

  • Airflow DAG'ı PARSE ZAMANINDA bilinmek zorundadır (statik graf).
  • Bizim task'lar ise ÇALIŞMA ANINDA, LLM'in kararıyla doğuyor (kaç tane, hangi
    bağımlılıkla — önceden bilinmiyor).
  • Yani "ajan task üretir" senaryosunda Airflow'a verecek bir DAG YOKTUR.

Pratikte üç kaçış yolu var, üçü de bedelli:
  1) Dynamic Task Mapping (.expand): sayı runtime'da belirlenebilir ama graf ŞEKLİ
     yine sabit; ajanın kurduğu keyfi bağımlılıkları temsil edemez.
  2) Ajan koşusunu TEK düğüme sıkıştır: o zaman Airflow yalnız SCHEDULING katmanıdır,
     task yönetimini yine sen yaparsın (bizim board gibi).
  3) Ajan DAG dosyası üretsin: parse gecikmesi + operasyonel kırılganlık.

Airflow'un rakipsiz olduğu yer: cron + backfill/catchup + operatör UI (statik batch).
Ajanın dinamik task üretimi için doğru araç DEĞİL."""


def _dispatch_airflow(board: TaskBoard, res: OrchestrationResult, *a, **kw):
    res.note = _AIRFLOW_NOTE
    res.dispatch_log += _AIRFLOW_NOTE.splitlines()
    res.dispatch_log.append("")
    res.dispatch_log.append("Ajanın planlama turunda ÜRETTİĞİ graf:")
    res.dispatch_log.append(board.board_text())

    # "Ajan önceden DAG yapabilir mi?" → EVET: planlanan grafı DAG'a döküyoruz
    path = export_airflow_dag(board, goal=kw.get("goal", ""))
    res.dag_file = str(path)
    res.dispatch_log.append("")
    res.dispatch_log.append(f"→ KAÇIŞ YOLU (3) UYGULANDI: bu graf gerçek bir Airflow DAG'ına "
                            f"döküldü: {path.name}")
    res.dispatch_log.append("  Yani ajan ÖNCEDEN DAG yapabilir — planlama turu zaten bir DAG üretir.")
    res.dispatch_log.append("  BEDELİ: DAG artık DONMUŞ. Yürütme sırasında ajan yeni task üretemez,")
    res.dispatch_log.append("  replanlama yapamaz; her yeni hedef için yeni DAG dosyası + parse gecikmesi.")


# ═══════════════════ genel giriş noktası ═══════════════════

def orchestrate(goal: str, backend: str = "own", strategy: str = "hermes",
                budget: int = 3_000, crash_at: str | None = None,
                fail_at: str | None = None, board: TaskBoard | None = None,
                on_event=None) -> OrchestrationResult:
    t0 = time.time()
    board = board or TaskBoard()
    res = OrchestrationResult(backend=backend, strategy=strategy)

    # 1) PLANLAMA: ajan task ÜRETİR
    if on_event:
        on_event({"type": "phase", "text": "PLANLAMA — graf kuruluyor"})
    res.plan_trace = plan_phase(board, goal, on_event=on_event)
    # PLAN DOĞRULAMA: eksik veri kenarlarını yürütmeden ÖNCE yakala/onar
    for _o in dogrula_dag(board, log=res.plan_trace):
        if on_event:
            on_event({"type": "log", "text": _o})
    res.tasks_created = len(board.list_tasks())
    res.dispatch_log.append(f"── PLANLAMA bitti: ajan {res.tasks_created} task üretti "
                            f"(çalışma anında, dinamik)")

    # 2) DISPATCH: motor task'ları yönetir
    try:
        if on_event:
            on_event({"type": "phase", "text": f"DISPATCH — {backend} motoru yürütüyor"})
        if backend == "airflow":
            _dispatch_airflow(board, res, goal=goal)
        elif backend == "own":
            _dispatch_own(board, res, strategy, budget, crash_at, fail_at,
                          on_event=on_event)
        else:
            {"celery": _dispatch_celery, "temporal": _dispatch_temporal}[backend](
                board, res, strategy, budget, crash_at, fail_at, on_event=on_event)
    except Exception as e:
        res.dispatch_log.append(f"✗ dispatch hatası: {type(e).__name__}: {str(e)[:220]}")

    res.tasks = board.list_tasks()
    res.tasks_spawned_runtime = sum(
        1 for t in res.tasks if str(t.get('created_by', '')).startswith('worker:'))
    # düğüm türü sayımını BOARD'dan yap (backend'den bağımsız, tek doğruluk kaynağı)
    _done = [t for t in res.tasks if t["status"] == "done"]
    res.fn_tasks_run = sum(1 for t in _done if t.get("kind") == "function")
    res.agent_tasks_run = sum(1 for t in _done if t.get("kind") == "agent")
    res.events = board.events()
    res.counts = board.counts()
    res.tasks_done = len(board.list_tasks("done"))
    res.tasks_failed = len(board.list_tasks("failed"))
    res.ok = (backend == "airflow") or (res.tasks_done > 0 and res.tasks_failed == 0)
    res.seconds = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ajan task üretir, motor yönetir")
    ap.add_argument("--backend", default="own", choices=["own", "celery", "temporal", "airflow"])
    ap.add_argument("--strategy", default="hermes")
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--goal", default="auth modülünü güvenlik açısından denetle ve MFA hatasını raporla")
    ap.add_argument("--crash-at", default=None, help="başlığında bu geçen task'ta worker çökmesi")
    ap.add_argument("--fail-at", default=None, help="bu tool'da geçici hata")
    ap.add_argument("--json", action="store_true", help="yapılandırılmış JSON çıktı (web UI)")
    a = ap.parse_args()

    if a.json:
        r = orchestrate(a.goal, backend=a.backend, strategy=a.strategy, budget=a.budget,
                        crash_at=a.crash_at, fail_at=a.fail_at)
        ce = r.compaction_events
        ham = sum(e["before"] for e in ce)
        kal = sum(e["after"] for e in ce)
        print("<<<JSON>>>" + json.dumps({
            "ok": r.ok, "backend": r.backend, "strategy": r.strategy, "goal": a.goal,
            "plan_trace": r.plan_trace, "dispatch_log": r.dispatch_log,
            "tasks": r.tasks, "events": r.events, "counts": r.counts,
            "tasks_created": r.tasks_created,
            "tasks_spawned_runtime": r.tasks_spawned_runtime,
            "fn_tasks_run": r.fn_tasks_run, "agent_tasks_run": r.agent_tasks_run, "tasks_done": r.tasks_done,
            "tasks_failed": r.tasks_failed, "retries": r.retries,
            "crashes": r.crashes, "recovered": r.recovered,
            "compaction": {"ham": ham, "kalan": kal, "olay": len(ce),
                           "pct": round((ham - kal) / ham * 100, 1) if ham else 0},
            "note": r.note, "dag_file": r.dag_file, "seconds": r.seconds,
        }, ensure_ascii=False, default=str))
        sys.exit(0)

    print("=" * 84)
    print("AJAN TASK ÜRETİR → MOTOR YÖNETİR")
    print(f"  backend={a.backend} · strateji={a.strategy} · bütçe={a.budget:,}")
    print(f"  hedef: {a.goal}")
    print("=" * 84)

    r = orchestrate(a.goal, backend=a.backend, strategy=a.strategy, budget=a.budget,
                    crash_at=a.crash_at, fail_at=a.fail_at)

    print("\n── 1) PLANLAMA (ajan task üretti) " + "─" * 46)
    for t in r.plan_trace:
        print("  " + t)

    print("\n── 2) DISPATCH (motor yönetti) " + "─" * 49)
    for l in r.dispatch_log:
        print("  " + l)

    print("\n── 3) BOARD (son durum) " + "─" * 56)
    for t in r.tasks:
        dep = f" ← {','.join(t['parents'])}" if t["parents"] else ""
        print(f"  {t['id']} [{t['status']:<7}] {t['title'][:50]}{dep}")
        if t.get("result"):
            print(f"           → {str(t['result'])[:100]}")

    ce = r.compaction_events
    ham = sum(e["before"] for e in ce)
    kal = sum(e["after"] for e in ce)
    print("\n── ÖZET " + "─" * 72)
    print(f"  düğüm türü: {r.fn_tasks_run} DETERMİNİSTİK fonksiyon + "
          f"{r.agent_tasks_run} LLM ajan")
    print(f"  üretilen task: {r.tasks_created} (planlamada) + {r.tasks_spawned_runtime} "
          f"(YÜRÜTME ANINDA) · tamamlanan: {r.tasks_done} · "
          f"başarısız: {r.tasks_failed} · durumlar: {r.counts}")
    print(f"  retry: {r.retries} · çökme: {r.crashes} · otomatik kurtarma: {r.recovered}")
    if ham:
        print(f"  tool-trace compaction: {ham:,} → {kal:,} token "
              f"(%{(ham-kal)/ham*100:.1f} kazanç, {len(ce)} olay)")
    print(f"  süre: {r.seconds}s")
    print("=" * 84)


# ═══════════════════ KAYITLI AKIŞI YENİDEN KOŞTURMA ═══════════════════
# Planlama YOK: graf zaten kurulu. Bu, gerçek "pipeline" davranışıdır —
# aynı akış, yeni koşu (Airflow'da bir DAG'ın yeni DagRun'ı gibi).

def materialize(board: TaskBoard, nodes: list, arg_overrides: dict | None = None) -> dict:
    """Kayıtlı düğümleri board'a YENİ id'lerle yaz; parent bağlarını koru.

    Dönüş: {eski_id: yeni_id}. Katman sırasıyla yazılır ki parent'lar önce var olsun.
    """
    import pipelines as _P
    idmap: dict[str, str] = {}
    for layer in _P.layers(nodes):
        for n in layer:
            args = dict(n.get("args") or {})
            args.update((arg_overrides or {}).get(n["id"], {}))
            parents = [idmap[p] for p in (n.get("parents") or []) if p in idmap]
            new = board.create_task(
                title=n["title"], body="", parents=parents,
                priority=int(n.get("priority", 5)), created_by="pipeline",
                kind=n["kind"], fn=n.get("fn"), fn_args=args)
            idmap[n["id"]] = new
    return idmap


def run_saved(nodes: list, backend: str = "own", strategy: str = "hermes",
              budget: int = 3_000, board: TaskBoard | None = None,
              crash_at: str | None = None, fail_at: str | None = None,
              arg_overrides: dict | None = None, node_sim: dict | None = None,
              on_event=None) -> OrchestrationResult:
    """Kayıtlı bir akışı PLANLAMADAN koştur (yeni koşu)."""
    t0 = time.time()
    board = board or TaskBoard()
    res = OrchestrationResult(backend=backend, strategy=strategy)

    if on_event:
        on_event({"type": "phase", "text": "kayıtlı akış yükleniyor (planlama YOK)"})
    try:
        idmap = materialize(board, nodes, arg_overrides)
    except ValueError as e:
        # kayıtlı akış artık geçersiz (ör. fonksiyon kaldırılmış) → YÜKLEMEDE söyle,
        # yarım koşup yürütmede patlama.
        res.dispatch_log.append(f"✗ akış yüklenemedi: {e}")
        res.note = f"kayıtlı akış geçersiz: {e}"
        res.counts = board.counts()
        res.seconds = round(time.time() - t0, 1)
        if on_event:
            on_event({"type": "log", "text": f"✗ akış yüklenemedi: {e}"})
        return res
    for _o in dogrula_dag(board, log=res.dispatch_log):
        if on_event:
            on_event({"type": "log", "text": _o})
    res.tasks_created = len(idmap)
    res.idmap = dict(idmap)
    # Kayıtlı akış id'lerini board'un YENİ id'lerine çevir — yoksa ayar hiçbir
    # düğüme denk gelmez ve sessizce hiçbir şey olmaz.
    sim = cevir_node_sim(node_sim, idmap)
    if on_event:
        for old, new in idmap.items():
            t = board.get(new)
            lbl = t["fn"] if t["kind"] == "function" else "AJAN"
            _s = (sim.get(new) or {}).get("mod")
            on_event({"type": "node_added",
                      "text": f"AKIŞ düğümü: {new} = {lbl}({t.get('fn_args') or {}}) "
                              f"[{t['status']}]" + (f"  ⚙ simülasyon: {_s}" if _s else "")})
        on_event({"type": "phase", "text": f"DISPATCH — {backend} motoru yürütüyor"})

    try:
        if backend == "own":
            _dispatch_own(board, res, strategy, budget, crash_at, fail_at,
                          on_event=on_event, node_sim=sim)
        elif backend == "airflow":
            _dispatch_airflow(board, res, goal="(kayıtlı akış)", node_sim=sim,
                              on_event=on_event)
        else:
            # on_event ARTIK geçiliyor — önceden geçilmediği için celery ve temporal
            # koşularında arayüze hiç canlı log akmıyordu.
            {"celery": _dispatch_celery, "temporal": _dispatch_temporal}[backend](
                board, res, strategy, budget, crash_at, fail_at,
                on_event=on_event, node_sim=sim)
    except Exception as e:
        res.dispatch_log.append(f"✗ dispatch hatası: {type(e).__name__}: {str(e)[:200]}")

    res.tasks = board.list_tasks()
    res.events = board.events()
    res.counts = board.counts()
    _done = [t for t in res.tasks if t["status"] == "done"]
    res.fn_tasks_run = sum(1 for t in _done if t.get("kind") == "function")
    res.agent_tasks_run = sum(1 for t in _done if t.get("kind") == "agent")
    res.tasks_done = len(_done)
    res.tasks_failed = len(board.list_tasks("failed"))
    res.ok = res.tasks_done > 0 and res.tasks_failed == 0
    res.seconds = round(time.time() - t0, 1)
    return res
