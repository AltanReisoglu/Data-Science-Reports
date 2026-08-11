#!/usr/bin/env python3
"""
motor_dili.py — AYNI grafı dört motorun KENDİ dilinde yazar.

Sohbette kurulan graf (`pipelines_store`'daki düğüm listesi) girdi; çıktı dört
ayrı kaynak dosya. Tez: **tek graf, dört dil** — ve her dilin kendi tuzağı
üretilen kodun içinde görünür hale geliyor.

    python motor_dili.py                 # en son akış, dört dil
    python motor_dili.py p_50647636      # belirli akış
    python motor_dili.py p_… celery      # tek dil

ÖNEMLİ — bu modül GÖSTERİM içindir, yürütme değil. Yürütme yolu board üzerinden
gidiyor. Airflow'da ikisi de kod üretiyor: buradaki gösterim,
`orchestrator.export_airflow_dag` ise yürütme (o ayrıca node_sim'i dosyaya gömer).
İkisinin AYNI grafı ürettiği `test_motor_dili.py` ile bağlanmıştır — biri
değişirse test kırılır.

İFADE GÜCÜ FARKI (asıl bulgu):
  own / temporal / airflow  → her DAG'ı BİREBİR ifade eder
  celery                    → canvas yalnız SERİ-PARALEL grafları birebir ifade
                              eder. Katman sınırında "bu düğüm aslında sadece X'i
                              bekliyor" bilgisi kaybolur; canvas onu TÜM katmanı
                              beklemeye zorlar. Bu gerçek bir kısıt, gizlenmiyor.
"""
from __future__ import annotations

import time

DILLER = ("own", "celery", "celery_canvas", "airflow", "temporal")


# ───────────────────────────── graf yardımcıları ────────────────────────────

def _katmanlar(nodes: list) -> list[list[dict]]:
    """Düğümleri bağımlılık derinliğine göre katmanlara ayır."""
    by_id = {n["id"]: n for n in nodes}
    derinlik: dict[str, int] = {}

    def d(nid, gorulen=()):
        if nid in derinlik:
            return derinlik[nid]
        if nid in gorulen:                      # döngü koruması
            return 0
        n = by_id.get(nid)
        ebeveyn = [p for p in (n or {}).get("parents", []) if p in by_id]
        if not n or not ebeveyn:
            derinlik[nid] = 0
            return 0
        derinlik[nid] = 1 + max(d(p, gorulen + (nid,)) for p in ebeveyn)
        return derinlik[nid]

    for n in nodes:
        d(n["id"])
    en = max(derinlik.values(), default=0)
    return [[n for n in nodes if derinlik[n["id"]] == k] for k in range(en + 1)]


def _pyid(tid: str) -> str:
    return "n_" + str(tid).replace("-", "_").replace(".", "_")


def _atalar(nodes: list) -> dict[str, set]:
    """Her düğümün GEÇİŞLİ ata kümesi (ebeveynin ebeveyni de atadır)."""
    by_id = {n["id"]: n for n in nodes}
    bellek: dict[str, set] = {}

    def a(nid, gorulen=frozenset()):
        if nid in bellek:
            return bellek[nid]
        if nid in gorulen:                      # döngü koruması
            return set()
        s = set()
        for p in by_id.get(nid, {}).get("parents", []):
            if p in by_id:
                s.add(p)
                s |= a(p, gorulen | {nid})
        bellek[nid] = s
        return s

    for n in nodes:
        a(n["id"])
    return bellek


def _seri_paralel_mi(nodes: list, kat: list[list[dict]]) -> tuple[bool, list[str]]:
    """Canvas bu grafı KAYIPSIZ ifade edebilir mi?

    Celery canvas `chain(group(K0), group(K1), …)` biçiminde yazıldığında i.
    katmandaki düğüm, ÖNCEKİ TÜM KATMANLARIN bitmesini bekler. Bu, düğümün
    GEÇİŞLİ ATALARINI beklemesi kadarıyla zararsız — ata zaten beklenmeli.
    Sorun, düğümü ATASI OLMAYAN bir işi beklemeye zorlaması: sonuç doğru çıkar
    ama paralellik kaybolur ve graf yanlış anlatılır.

    (İlk sürüm "ebeveyn kümesi == önceki katman" diye bakıyordu ve elmas grafı
     yanlışlıkla 'birebir değil' işaretliyordu — ölçüt geçişli ata olmalı.)
    """
    atalar = _atalar(nodes)
    kayiplar = []
    onceki_hepsi: set = set()
    for i, k in enumerate(kat):
        if i:
            for n in k:
                fazla = onceki_hepsi - atalar[n["id"]]
                if fazla:
                    kayiplar.append(
                        f"{n['id']} ({n.get('fn') or n.get('title', '')[:20]}) atası "
                        f"OLMAYAN {sorted(fazla)} düğümünü beklemek zorunda kalır")
        onceki_hepsi |= {n["id"] for n in k}
    return (not kayiplar), kayiplar


def _fn_bilgi(n: dict) -> tuple[str, dict]:
    return n.get("fn") or "?", (n.get("fn_args") or {})


def _baslik(goal: str, dil: str, ek: str = "") -> list[str]:
    return [
        '"""OTOMATİK ÜRETİLDİ — sohbette kurulan grafın ' + dil + ' karşılığı.',
        "",
        f"Hedef : {goal or '(belirtilmedi)'}",
        f"Üretim: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        *( ["", ek] if ek else [] ),
        '"""',
    ]


# ─────────────────────────────────── OWN ────────────────────────────────────

def own(nodes: list, goal: str = "") -> dict:
    L = _baslik(goal, "board (own / hermes tarzı)",
                "Board bir motor değil, KARAR KATMANI: graf burada tablo olarak\n"
                "duruyor. Bağımlılık kapısını `parents` tutuyor; hangi motorun\n"
                "dağıtacağı ayrı bir seçim.")
    L += [
        "from taskboard import TaskBoard",
        "",
        "board = TaskBoard()",
        "",
    ]
    for n in nodes:
        fn, args = _fn_bilgi(n)
        pid = _pyid(n["id"])
        if n.get("kind") == "function":
            L.append(f"{pid} = board.create_task(")
            L.append(f"    {n.get('title', '')!r},")
            L.append('    "function",')
            L.append(f"    fn={fn!r},")
            L.append(f"    fn_args={args!r},")
        else:
            L.append(f"{pid} = board.create_task(")
            L.append(f"    {n.get('title', '')!r},")
            L.append('    "agent",')
        if n["parents"]:
            L.append(f"    parents=[{', '.join(_pyid(p) for p in n['parents'])}],")
        L.append(")")
    L += [
        "",
        "# Bağımlılık kapısı: parents done olunca blocked→ready",
        "board.recompute_ready()",
    ]
    return {
        "ad": "own", "dil": "board API (Python)", "dosya": "board_grafi.py",
        "kod": "\n".join(L) + "\n",
        "birebir": True,
        "notlar": [
            "Graf bir TABLO. Düğüm eklemek `create_task` çağırmak demek — çalışma "
            "anında da yapılabilir.",
            "`parents` bağımlılık kapısı; `recompute_ready()` blocked→ready geçişini yapar.",
        ],
        "tuzaklar": [
            "Retry, iptal zinciri, breaker, checkpoint — hepsini SEN yazarsın. "
            "Bu çalışmada 12 hata çıktı ve hepsi bu katmandaydı.",
        ],
    }


# ────────────────────────────────── CELERY ──────────────────────────────────

def celery(nodes: list, goal: str = "") -> dict:
    kat = _katmanlar(nodes)
    birebir, kayiplar = _seri_paralel_mi(nodes, kat)

    ek = ("Celery'nin workflow'a en yakın yapısı CANVAS: chain / group / chord.\n"
          "chain SIRAYLA koşar, group PARALEL, chord ise group bitince callback.")
    if not birebir:
        ek += ("\n\nDİKKAT — bu graf canvas ile BİREBİR ifade EDİLEMİYOR.\n"
               "Canvas katman sınırında senkronize eder; aşağıdaki düğümler\n"
               "gerçekte beklemedikleri işleri beklemek zorunda kalıyor.")

    L = _baslik(goal, "Celery canvas", ek)
    L += [
        "from celery import chain, group, chord",
        "from celery_worker import app",
        "",
        "# Düğümler `functions.py` kaydındaki deterministik fonksiyonlar.",
        "# Canvas'ta her adım bir task imzası (`.s(...)`).",
        "",
    ]
    for i, k in enumerate(kat):
        L.append(f"# ── katman {i} ──")
        for n in k:
            fn, args = _fn_bilgi(n)
            arg_s = ", ".join(f"{a}={v!r}" for a, v in args.items())
            L.append(f"{_pyid(n['id'])} = app.signature('fn_cagir', "
                     f"kwargs={{'fn': {fn!r}, 'args': {args!r}}})"
                     + (f"   # {fn}({arg_s})" if arg_s else f"   # {fn}()"))
    L.append("")
    L.append("# ── canvas ──")

    parcalar = []
    for i, k in enumerate(kat):
        idler = [_pyid(n["id"]) for n in k]
        parcalar.append(idler[0] if len(idler) == 1 else "group(" + ", ".join(idler) + ")")

    if len(parcalar) == 1:
        L.append(f"akis = {parcalar[0]}")
    else:
        # Son katman tek düğümse ve öncesi group ise: chord (paralel bitince callback)
        L.append("akis = chain(")
        for p in parcalar:
            L.append(f"    {p},")
        L.append(")")
        if len(kat) >= 2 and len(kat[-2]) > 1 and len(kat[-1]) == 1:
            L.append("")
            L.append("# Aynı şeyin chord biçimi (son katman tek düğüm, öncesi paralel):")
            L.append(f"# akis = chord({parcalar[-2]}, {parcalar[-1]})")
    L += ["", "sonuc = akis.apply_async()"]

    tuzaklar = [
        "İMZA SIZINTISI: chain'de bir task'ın DÖNÜŞÜ sonrakinin İLK ARGÜMANI olur. "
        "Yani kompozisyon kararı fonksiyon imzasına sızar — fonksiyonlar birbirinden "
        "bağımsız yazılamaz.",
        "Canvas DURABLE DEĞİL: zincir ortasında worker çökerse Celery '3. adımdaydım' "
        "bilgisini TUTMAZ. O defteri kendi DB'nde tutman gerekir.",
        "'Atlandı / iptal' diye bir durum yok — batan zincirin kalanına ne olduğu belirsiz.",
    ]
    if not birebir:
        tuzaklar.insert(0,
            "BU GRAF BİREBİR İFADE EDİLEMİYOR — canvas katman katman senkronize eder. "
            + " · ".join(kayiplar[:3])
            + (f" (+{len(kayiplar)-3} tane daha)" if len(kayiplar) > 3 else ""))

    return {
        "ad": "celery", "dil": "Celery canvas (chain/group/chord)",
        "dosya": "celery_canvas.py", "kod": "\n".join(L) + "\n",
        "birebir": birebir, "kayiplar": kayiplar,
        "notlar": [
            f"{len(kat)} katman → chain içinde {sum(1 for p in parcalar if p.startswith('group'))} "
            f"paralel grup.",
            "Celery'de 'A seviyesi iş' kavramı yok: canvas bir kolaylık, kalıcı bir "
            "iş kaydı değil.",
        ],
        "tuzaklar": tuzaklar,
    }


# ───────────────────────────────── AIRFLOW ──────────────────────────────────

def airflow(nodes: list, goal: str = "", schedule: str | None = "0 8 * * *",
            retries: int = 2, retry_delay_sn: int = 30) -> dict:
    L = _baslik(goal, "Airflow DAG",
                "Airflow'da workflow bir DOSYA: PythonOperator'lar `>>` ile bağlanır,\n"
                "veri XCom üzerinden akar. Graf PARSE ZAMANINDA sabitlenir.")
    L += [
        "from datetime import datetime, timedelta",
        "from airflow import DAG",
        "from airflow.operators.python import PythonOperator",
        "",
        "default_args = {",
        f'    "retries": {retries},                       # DÜĞÜM bazında retry',
        f'    "retry_delay": timedelta(seconds={retry_delay_sn}),',
        "}",
        "",
        "",
        "def _run_fn(fn_name, args, parent_ids, **ctx):",
        '    """Kayıtlı fonksiyonu koştur; upstream veriyi XCom\'dan al."""',
        "    from functions import call",
        '    ti = ctx["ti"]',
        "    up = {p: ti.xcom_pull(task_ids=p) for p in parent_ids}",
        "    return call(fn_name, args, {k: v for k, v in up.items() if v is not None})",
        "",
        "",
        "with DAG(",
        '    dag_id="uretilen_akis",',
        "    default_args=default_args,",
        (f"    schedule={schedule!r},          # Airflow'un asıl gücü: cron + catchup"
         if schedule else "    schedule=None,"),
        "    start_date=datetime(2026, 1, 1),",
        "    catchup=False,",
        "    max_active_runs=1,",
        ") as dag:",
        "",
    ]
    for n in nodes:
        fn, args = _fn_bilgi(n)
        L.append(f"    {_pyid(n['id'])} = PythonOperator(")
        L.append(f"        task_id={n['id']!r},")
        L.append("        python_callable=_run_fn,")
        L.append(f"        op_kwargs={{'fn_name': {fn!r}, 'args': {args!r}, "
                 f"'parent_ids': {n['parents']!r}}},")
        L.append("    )")
    L.append("")
    L.append("    # ajanın kurduğu bağımlılıklar")
    kenar = 0
    for n in nodes:
        for p in n["parents"]:
            L.append(f"    {_pyid(p)} >> {_pyid(n['id'])}")
            kenar += 1
    if not kenar:
        L.append("    # (bağımsız düğümler — hepsi paralel)")

    return {
        "ad": "airflow", "dil": "Airflow DAG (PythonOperator + >>)",
        "dosya": "uretilen_akis.py", "kod": "\n".join(L) + "\n",
        "birebir": True,
        "notlar": [
            f"{len(nodes)} PythonOperator, {kenar} kenar. Çeviri BİREBİR: "
            "düğüm→operator, parents→`>>`, veri→XCom.",
            "Bu dosya yürütmede de kullanılıyor (orchestrator.export_airflow_dag "
            "ayrıca node_sim'i gömer).",
        ],
        "tuzaklar": [
            "DAG DONMUŞTUR: yürütme sırasında ajan yeni düğüm ekleyemez. Graf parse "
            "zamanında sabitlenir.",
            "XCom küçük veri içindir — metadata DB'de saklanır, büyük veri koyma.",
            "Koşullu dal İKİ kavram gerektirir: BranchPythonOperator + trigger_rule. "
            "trigger_rule unutulursa birleşim düğümü de skip olur.",
        ],
    }


# ──────────────────────────────── TEMPORAL ──────────────────────────────────

def temporal(nodes: list, goal: str = "") -> dict:
    kat = _katmanlar(nodes)
    L = _baslik(goal, "Temporal workflow",
                "Temporal'da workflow DÜPEDÜZ KOD: `if`, `for`, `await`. Motor her\n"
                "adımı event history'ye yazar; çökünce replay edip biteni ATLAR.\n"
                "Bedeli: workflow gövdesi DETERMİNİST olmak zorunda — yan etki yasak.")
    L += [
        "import asyncio",
        "from datetime import timedelta",
        "from temporalio import workflow",
        "from temporalio.common import RetryPolicy",
        "",
        "",
        "@workflow.defn",
        "class UretilenWorkflow:",
        "    @workflow.run",
        "    async def run(self) -> dict:",
        "        s = {}          # düğüm sonuçları (upstream veri buradan akar)",
        "",
    ]
    for i, k in enumerate(kat):
        L.append(f"        # ── katman {i} — {'paralel' if len(k) > 1 else 'tek düğüm'} ──")
        cagrilar = []
        for n in k:
            fn, args = _fn_bilgi(n)
            up = "{" + ", ".join(f"{p!r}: s[{p!r}]" for p in n["parents"]) + "}"
            cagrilar.append(
                "            workflow.execute_activity(\n"
                f'                "fn_cagir",\n'
                f"                {{'fn': {fn!r}, 'args': {args!r}, 'up': {up}}},\n"
                "                start_to_close_timeout=timedelta(seconds=60),\n"
                "                retry_policy=RetryPolicy(maximum_attempts=3)),")
        hedef = ", ".join(f"s[{n['id']!r}]" for n in k)
        if len(k) == 1:
            L.append(f"        {hedef} = await (")
            L.append(cagrilar[0].rstrip(","))
            L.append("        )")
        else:
            L.append(f"        {hedef} = await asyncio.gather(")
            L += cagrilar
            L.append("        )")
        L.append("")
    L.append("        return s")

    return {
        "ad": "temporal", "dil": "Temporal workflow (kod)",
        "dosya": "uretilen_workflow.py", "kod": "\n".join(L) + "\n",
        "birebir": True,
        "notlar": [
            f"{len(kat)} katman → {sum(1 for k in kat if len(k) > 1)} `asyncio.gather`.",
            "Her DAG birebir ifade edilebilir: bağımlılık `s[...]` okumasıyla doğal "
            "olarak kuruluyor — motor-özel bir kavram öğrenmiyorsun.",
            "Koşullu dal `if`, dinamik fan-out `for` — dilin kendi yapıları yeterli.",
        ],
        "tuzaklar": [
            "DETERMİNİZM: workflow gövdesinde `random()`, `datetime.now()`, doğrudan "
            "HTTP/DB YASAK — replay bozulur. Yan etkili her şey activity'ye sarılmalı.",
            "Bu mimariyi BÖLMEK demek: orkestrasyon ile IO ayrı dosyalarda yaşar.",
            "VERSIONING: canlıda koşan workflow varken kodu değiştirirsen replay "
            "bozulabilir; `workflow.patched()` ile yönetilir.",
        ],
    }


# ───────────────────────────── dışa açık giriş ──────────────────────────────

# celery_canvas AYRI bir sekme ama ürettiği kod celery canvas'ının ta kendisi —
# tek fark, o sekmede bu kod GERÇEKTEN koşuyor (board'suz).
URETICILER = {"own": own, "celery": celery, "celery_canvas": celery,
              "airflow": airflow, "temporal": temporal}


def cevir(nodes: list, goal: str = "", dil: str | None = None) -> dict:
    """Grafı bir ya da dört dile çevir. Bir üretici patlarsa diğerleri sürer."""
    hedefler = [dil] if dil else list(DILLER)
    out = []
    for d in hedefler:
        try:
            x = URETICILER[d](nodes, goal)
            x["ad"] = d               # celery_canvas kendi adıyla dönsün
            out.append(x)
        except Exception as e:
            import traceback
            out.append({"ad": d, "dil": d, "dosya": "", "kod": "",
                        "birebir": False, "notlar": [], "tuzaklar": [],
                        "hata": f"{type(e).__name__}: {e}",
                        "iz": traceback.format_exc().splitlines()[-4:]})
    kat = _katmanlar(nodes)
    return {
        "dugum": len(nodes),
        "kenar": sum(len(n["parents"]) for n in nodes),
        "katman": len(kat),
        "katman_boyu": [len(k) for k in kat],
        "diller": out,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pipelines as P

    arg = [a for a in sys.argv[1:]]
    pid = next((a for a in arg if a.startswith("p_")), None)
    dil = next((a for a in arg if a in DILLER), None)
    if not pid:
        lst = P.listing()
        if not lst:
            print("kayıtlı akış yok — önce sohbette bir workflow kur."); sys.exit(1)
        pid = lst[0]["id"]
    d = P.load(pid)
    if not d:
        print(f"akış bulunamadı: {pid}"); sys.exit(1)

    v = cevir(d["nodes"], d.get("goal", ""), dil)
    print(f"akış {pid} · {v['dugum']} düğüm · {v['kenar']} kenar · "
          f"{v['katman']} katman {v['katman_boyu']}\n")
    for x in v["diller"]:
        print("═" * 78)
        im = "birebir ✓" if x.get("birebir") else "BİREBİR DEĞİL ⚠"
        print(f" {x['dil']}   [{im}]   → {x['dosya']}")
        print("═" * 78)
        if x.get("hata"):
            print("  HATA:", x["hata"]); continue
        print(x["kod"])
        for n in x["notlar"]:
            print(f"  · {n}")
        for t in x["tuzaklar"]:
            print(f"  ⚠ {t}")
        print()
