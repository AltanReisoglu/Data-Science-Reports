#!/usr/bin/env python3
"""
chat_server.py — SOHBET arayüzü: sen yazdıkça ajan çalışır (canlı akış).

Form değil sohbet: her mesajın ajanın grafına yeni düğümler ekler, motor onları
yürütür, olaylar SSE ile ANINDA ekrana düşer. Board konuşma boyunca KALICIDIR —
"şimdi bir de session.py'a bak" dediğinde aynı board büyür.

    .venv/bin/python demo-brain-agent/chat_server.py    # → http://127.0.0.1:8030

Yalnız 127.0.0.1'e bağlanır.
"""
from __future__ import annotations

import json
import queue
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

sys.path.insert(0, str(HERE.parent / 'poc'))
import functions as F            # noqa: E402
import orchestrator as O         # noqa: E402
import pipelines as P            # noqa: E402
import scheduler as SCH          # noqa: E402
import airflow_runner as AR      # noqa: E402
from compaction import STRATEGIES, STRATEGY_INFO   # noqa: E402
from taskboard import TaskBoard  # noqa: E402

PORT = 8030
SESSIONS: dict[str, dict] = {}
_LOCK = threading.Lock()
# Zamanlama deposu — oturumdan bağımsız, sunucu ömrü boyunca tek örnek.
# (Oturumlar RAM'de, zamanlamalar diskte: sunucu yeniden başlasa da yaşarlar.)
SCH_STORE = SCH.ScheduleStore()


def get_session(sid: str) -> dict:
    with _LOCK:
        if sid not in SESSIONS:
            SESSIONS[sid] = {
                "board": TaskBoard(),          # konuşma boyunca KALICI
                "history": [],
                "settings": {"backend": "own", "strategy": "hermes", "karsilastir": "1", "budget": 3000,
                             "pack": "audit"},
                "busy": False,
            }
        return SESSIONS[sid]


# ═══════════ MOTOR KARŞILAŞTIRMA — ortak koşum yardımcısı ═══════════

def _airflow_cikti(xcom: dict) -> str:
    """Airflow board'a yazmıyor — nihai çıktıyı kendi XCom'undan çıkar.

    Diğer üç motorun `wf_cikti`'sının Airflow karşılığı; aynı metni üretmeli ki
    "dört motor aynı çıktıyı verdi mi" karşılaştırması anlamlı olsun.
    """
    for v in reversed(list(xcom.values())):
        if isinstance(v, dict):
            for k in ("rapor_md", "md", "ozet"):
                if k in v:
                    return str(v[k])
    for v in reversed(list(xcom.values())):
        if isinstance(v, dict):
            kisa = {k: x for k, x in v.items()
                    if k not in ("_raw_chars", "rows", "text", "matches", "eslesme")}
            return json.dumps(kisa, ensure_ascii=False, indent=1)[:1200]
    return "(çıktı yok)"


def wf_cikti(tasks: list) -> str:
    """Akışın NİHAİ ÇIKTISI — son tamamlanan düğümün insan-okur sonucu.

    Motorları kıyaslarken asıl soru "aynı sayılar mı" değil, **aynı çıktıyı mı
    üretiyorlar**. Bu yüzden her motorun koşusundan bu metni çıkarıp ekrana
    ayrı ayrı basıyoruz.
    """
    for t in reversed([x for x in tasks if x["status"] == "done"]):
        rr = t.get("result") or ""
        try:
            d = json.loads(rr)
            if isinstance(d, dict):
                for k in ("rapor_md", "md", "ozet"):
                    if k in d:
                        return str(d[k])
                # yapılandırılmış sonuç — okunur bir özet çıkar
                kisa = {k: v for k, v in d.items()
                        if k not in ("_raw_chars", "rows", "text", "matches", "eslesme")}
                return json.dumps(kisa, ensure_ascii=False, indent=1)[:1200]
        except Exception:
            if rr and not rr.startswith("{"):
                return rr[:1200]
    return "(çıktı yok)"



def wf_kostur(nodes: list, backend: str, sim: dict, args: dict,
              strategy: str = "hermes", budget: int = 3000, on_event=None) -> dict:
    """Kayıtlı akışı TEK bir motorda koştur, karşılaştırılabilir özet döndür.

    Dört motorun ölçümü aynı sözlük şemasıyla dönsün ki tablo tek yerde kurulsun.
    Airflow ayrı süreçte koştuğu için board'u değil kendi metadata DB'sini raporlar.
    """
    t0 = time.time()
    if backend == "airflow":
        # Airflow'un koşabilmesi için önce board'a düğümleri yazmalıyız
        # (DAG üreteci board'dan okuyor). Yürütmeyi Airflow yapacak.
        b = TaskBoard()
        idmap = O.materialize(b, nodes, args or None)
        sim_yeni = O.cevir_node_sim(sim, idmap)
        _ters = {v: k for k, v in idmap.items()}
        r = AR.kostur(b, node_sim=sim_yeni, goal="panel koşusu", on_event=on_event)
        sayim = r.get("sayim", {})
        return {"backend": "airflow", "ok": r.get("ok", False),
                "sn": r.get("sn", round(time.time() - t0, 2)),
                "dugum": len(nodes),
                "done": sayim.get("success", 0),
                "failed": sayim.get("failed", 0),
                # Airflow'un 'upstream_failed' + 'skipped'i bizim 'cancelled'ımızın karşılığı
                "cancelled": sayim.get("upstream_failed", 0) + sayim.get("skipped", 0),
                # board'daki `attempt` TOPLAM denemeyi sayıyor (kalıcı hatada 3).
                # Airflow'un try_number'ı da toplam denemedir → doğrudan kullan,
                # `-1` yaparsak sütun diğer motorlarla kıyaslanamaz hale gelir.
                "retries": sum(max(0, (x.get("try_number") or 0)) - (1 if x.get("state") == "success" else 0)
                               for x in r.get("dugumler", [])),
                "crashes": 0, "recovered": 0,
                "dag_file": r.get("dag_file", ""), "not": r.get("hata", "")[:200],
                # Airflow board'a yazmıyor; çıktıyı kendi XCom'undan al
                "cikti": _airflow_cikti(r.get("xcom", {})),
                # düğümleri KAYITLI id'ye geri bağla — her koşu yeni id ürettiği için
                # arayüz aksi hâlde motorları aynı satırda hizalayamaz
                "dugumler": [{"nid": _ters.get(x["task_id"], x["task_id"]),
                              "durum": x["state"],
                              "deneme": max(0, (x["try_number"] or 1) - 1),
                              "kosum": x["try_number"] or 0,     # zaten koşum sayısı
                              "sn": round(x.get("duration") or 0, 2)}
                             for x in r.get("dugumler", [])]}

    b = TaskBoard()
    res = O.run_saved(nodes, backend=backend, strategy=strategy, budget=budget,
                      board=b, arg_overrides=args or None, node_sim=sim or None,
                      on_event=on_event)
    _ters = {v: k for k, v in (res.idmap or {}).items()}
    c = b.counts()
    # `res.retries` yalnız _dispatch_own tarafından artırılıyor; celery/temporal
    # koşularında hep 0 kalıyordu. Board'daki `attempt` toplamı üç motorda da
    # doğru ve karşılaştırılabilir (Airflow tarafı try_number'dan aynı sayıyı verir).
    _deneme = sum(max(0, int(t.get("attempt") or 0)) for t in b.list_tasks())
    return {"backend": backend, "ok": res.tasks_done > 0 and res.tasks_failed == 0,
            "sn": round(time.time() - t0, 2), "dugum": len(nodes),
            "done": c.get("done", 0), "failed": c.get("failed", 0),
            "cancelled": c.get("cancelled", 0), "retries": max(res.retries, _deneme),
            "crashes": res.crashes, "recovered": res.recovered,
            "dag_file": res.dag_file, "not": res.note[:200],
            "cikti": wf_cikti(b.list_tasks()),
            # `attempt` board'da HATA sayısıdır; Airflow'un `try_number`'ı ise
            # KOŞUM sayısı. Aynı sütunda kıyaslanabilmesi için ikisini de
            # "kaç kez koştu"ya normalleştiriyoruz.
            "dugumler": [{"nid": _ters.get(t["id"], t["id"]), "durum": t["status"],
                          "deneme": t["attempt"],
                          "kosum": int(t["attempt"] or 0)
                                   + (1 if t["status"] == "done" else 0),
                          "fn": t.get("fn"), "baslik": t["title"]}
                         for t in b.list_tasks()]}


def board_snapshot(board: TaskBoard) -> list:
    out = []
    for t in board.list_tasks():
        out.append({"id": t["id"], "title": t["title"], "status": t["status"],
                    "kind": t["kind"], "fn": t.get("fn"),
                    "parents": t["parents"], "attempt": t["attempt"],
                    "created_by": t["created_by"],
                    "result": (str(t.get("result") or "")[:400])})
    return out


def router_prompt() -> str:
    """Router prompt'u — TÜM paketlerin yetenekleriyle, çağrı anında üretilir."""
    lines = []
    for pk, v in F.PACKS.items():
        lines.append(f"  • [{pk}] {v['aciklama']}\n      fonksiyonlar: {', '.join(v['fns'])}")
    return (
        "Sen bir OTOMASYON asistanısın. Elindeki hazır fonksiyonlarla iş akışları kurar "
        "ve çalıştırırsın. Üç şeyden birini yaparsın:\n\n"
        "1) SIRADAN SOHBET / SORU → hiç tool çağırma, kısa cevap yaz.\n"
        "2) TEK BİR İŞLEM yeterliyse → ilgili fonksiyonu DOĞRUDAN çağır.\n"
        "3) ÇOK ADIMLI iş / 'sistem kur', 'pipeline kur', 'otomasyon' istekleri → "
        "`plan_workflow(goal, pack)` çağır; bir DAG kurulup çalıştırılır.\n\n"
        "ELİNDEKİ AKIŞ PAKETLERİ:\n" + "\n".join(lines) + "\n\n"
        "AYRICA ELİNDEKİ ARAÇLAR (doğrudan kullanabilirsin, arka arkaya birkaç kez de "
        "çağırabilirsin): read_file (dosyanın tamamını okur), search_code (kodda arar), "
        "run_tests (test çıktısı), fetch_docs (dokümantasyon). Bunların çıktısı ÇOK BÜYÜK "
        "olabilir; özetlenmiş/kırpılmış görebilirsin, elindekiyle ilerle.\n\n"
        "ÖNEMLİ:\n"
        "• Kullanıcı ETL/veri işleme isterse pack='data', yayın/deploy isterse pack='deploy', "
        "kod denetimi isterse pack='audit' kullan.\n"
        "• 'ETL sistemi kur' gibi bir istekte GENEL TAVSİYE VERME — elindeki data paketiyle "
        "gerçek akışı KUR ve ÇALIŞTIR.\n"
        "• Bir fonksiyonu doğrudan çağırırsan, o fonksiyonun paketi otomatik seçilir.\n"
        "• Yapamayacağın bir şey istenirse ne yapabildiğini paketlerle birlikte söyle."
    )


def _fn_pack(fn_name: str) -> str | None:
    for pk, v in F.PACKS.items():
        if fn_name in v["fns"]:
            return pk
    return None


def _router_tools() -> list:
    """TÜM paketlerin fonksiyonları + plan_workflow (paket seçimiyle)."""
    specs, seen = [], set()
    for pk, v in F.PACKS.items():
        for name, (_fn, desc, args) in v["fns"].items():
            if name in seen:
                continue
            seen.add(name)
            props = {k: {"type": "string", "description": val} for k, val in args.items()}
            specs.append({"type": "function", "function": {
                "name": name, "description": f"[{pk}] {desc}",
                "parameters": {"type": "object", "properties": props, "required": []}}})
    # İŞÇİ TOOL'LARI — sohbetin kendi çalışması sırasında doğrudan kullanabildikleri.
    # Bunların çıktısı HAM ve BÜYÜK olur → LLM context'ine girer → tool-trace compaction burada.
    import agent as _A
    for name, (_fn, desc, props, req) in _A.TOOLS.items():
        specs.append({"type": "function", "function": {
            "name": name, "description": f"[araç] {desc}",
            "parameters": {"type": "object", "properties": props, "required": req}}})

    specs.append({"type": "function", "function": {
        "name": "plan_workflow",
        "description": ("Çok adımlı bir iş akışı (DAG) kur ve çalıştır. "
                        "'sistem kur', 'pipeline kur', 'otomasyon' istekleri için kullan."),
        "parameters": {"type": "object", "properties": {
            "goal": {"type": "string", "description": "kurulacak akışın hedefi"},
            "pack": {"type": "string",
                     "description": "akış paketi: " + " | ".join(F.PACKS)}},
            "required": ["goal"]}}})
    return specs


def run_turn(sess: dict, msg: str, q: queue.Queue):
    """Bir sohbet turu: ajan KENDİSİ karar verir —
    düz konuş / tek tool çağır / task grafı kur."""
    import llm
    st = sess["settings"]
    board = sess["board"]
    t0 = time.time()
    F.use_pack(st.get("pack", "audit"))      # başlangıç paketi (router değiştirebilir)

    def emit(ev):
        q.put(ev)

    try:
        # ---- ROUTER: ne yapılacağına LLM karar verir ----
        hist = [{"role": h["role"], "content": h["text"]} for h in sess["history"][-6:]]
        r = llm.chat([{"role": "system", "content": router_prompt()}] + hist,
                     tools=_router_tools(), max_tokens=600, temperature=0.3)
        tcs = r.get("tool_calls") or []
        text = (r.get("content") or "").strip()

        # ---- 1) DÜZ SOHBET ----
        if not tcs:
            if not text:
                text = ("Merhaba! Bir kod tabanını denetleyebilirim: dosya okuma, desen tarama, "
                        "test koşturma, bulguları eşleştirme ve rapor üretme. Ne yapmamı istersin?")
            sess["history"].append({"role": "assistant", "text": text})
            emit({"type": "chat", "text": text})
            emit({"type": "summary", "text": f"sohbet · {round(time.time()-t0,1)} sn",
                  "crashes": 0, "recovered": 0, "retries": 0, "answer": ""})
            return

        call = tcs[0]["function"]
        fname = call["name"]
        try:
            fargs = json.loads(call.get("arguments") or "{}")
        except Exception:
            fargs = {}

        # ---- 3) TASK GRAFI KUR ----
        if fname == "plan_workflow":
            goal = fargs.get("goal") or msg
            want = (fargs.get("pack") or "").strip()
            if want in F.PACKS and want != st.get("pack"):
                F.use_pack(want)
                st["pack"] = want
                emit({"type": "log", "text": f"akış paketi seçildi: {want}"})
            # ── KARŞILAŞTIRMA KİPİ ──
            # Grafı BİR KEZ kur, sonra DÖRT motorda da koştur ve her birinin
            # çıktısını ayrı ayrı bas. Asıl soru: aynı graf hepsinde aynı sonucu
            # veriyor mu? (Bedeli: celery tek başına ~12-15 sn ekliyor.)
            if str(st.get("karsilastir", "0")) == "1":
                emit({"type": "phase", "text": "çok adımlı iş → GRAF kuruluyor "
                                              "(sonra DÖRT motorda koşacak)"})
                pb = TaskBoard()
                O.plan_phase(pb, goal, on_event=emit)
                nodes_raw = pb.list_tasks()
                if not nodes_raw:
                    emit({"type": "chat", "text": "Planlayıcı bu hedef için düğüm üretemedi."})
                    emit({"type": "summary", "text": "graf kurulamadı", "answer": ""})
                    return
                pid = P.save(goal=goal, pack=st.get("pack", "audit"), backend="own",
                             tasks=nodes_raw, stats={"dugum": len(nodes_raw),
                                                     "kaynak": "sohbet-karsilastirma"})
                emit({"type": "saved", "id": pid, "n": len(nodes_raw)})
                nodes = P.load(pid)["nodes"]

                satirlar, ciktilar = [], []
                for be in O.BACKENDS:
                    emit({"type": "motor_basladi", "backend": be,
                          "text": f"══ {be} koşuyor ══"})
                    try:
                        r = wf_kostur(nodes, be, {}, {},
                                      strategy=st["strategy"], budget=st["budget"],
                                      on_event=emit)
                    except Exception as e:
                        r = {"backend": be, "ok": False, "sn": 0.0, "dugum": len(nodes),
                             "done": 0, "failed": 0, "cancelled": 0, "retries": 0,
                             "crashes": 0, "recovered": 0, "dugumler": [],
                             "not": f"{type(e).__name__}: {str(e)[:150]}"}
                    satirlar.append(r)
                    # her motorun ÇIKTISI ayrı ayrı ekrana
                    ck = (r.get("cikti") or "").strip() or "(çıktı yok)"
                    ciktilar.append((be, ck))
                    emit({"type": "motor_cikti", "backend": be, "cikti": ck,
                          "sn": r["sn"], "ok": r["ok"],
                          "ozet": f"{r['done']}/{r['dugum']} düğüm · {r['sn']} sn"})
                    emit({"type": "motor_bitti", **r})

                emit({"type": "karsilastirma", "satirlar": satirlar})
                ayni = len({c for _, c in ciktilar}) == 1
                emit({"type": "board", "tasks": board_snapshot(pb)})
                ozet = (f"{len(nodes)} düğümlü graf · {len(O.BACKENDS)} motorda koştu · "
                        + ("TÜM MOTORLAR AYNI ÇIKTIYI ÜRETTİ ✓" if ayni
                           else "⚠ ÇIKTILAR FARKLI — aşağıda karşılaştır"))
                sess["history"].append({"role": "assistant", "text": ozet})
                emit({"type": "summary", "text": ozet, "crashes": 0, "recovered": 0,
                      "retries": sum(x["retries"] for x in satirlar),
                      "answer": ciktilar[0][1] if ciktilar else ""})
                return

            emit({"type": "phase", "text": "çok adımlı iş → GRAF kuruluyor"})
            before_ids = {t["id"] for t in board.list_tasks()}
            res = O.orchestrate(goal, backend=st["backend"], strategy=st["strategy"],
                                budget=st["budget"], board=board,
                                crash_at=st.get("crash_at") or None,
                                fail_at=st.get("fail_at") or None, on_event=emit)
            new = [t for t in board.list_tasks() if t["id"] not in before_ids]
            done_new = [t for t in new if t["status"] == "done"]
            fn_n = sum(1 for t in done_new if t["kind"] == "function")
            ag_n = sum(1 for t in done_new if t["kind"] == "agent")
            answer = ""
            for t in reversed(done_new):
                rr = t.get("result") or ""
                try:
                    d = json.loads(rr)
                    if isinstance(d, dict) and "rapor_md" in d:
                        answer = d["rapor_md"]; break
                except Exception:
                    if rr and not rr.startswith("{"):
                        answer = rr; break
            # BAŞARISIZ / İPTAL düğümleri cevaba yansıt: yarım kalan bir akış
            # "N düğüm tamamlandı" diye özetlenirse kullanıcı eksiği fark etmez.
            batan = [t for t in new if t["status"] == "failed"]
            iptal = [t for t in new if t["status"] == "cancelled"]
            if batan or iptal:
                uyari = (f"\n\n⚠ AKIŞ YARIM KALDI — {len(batan)} düğüm başarısız"
                         + (f", {len(iptal)} ardıl düğüm iptal edildi" if iptal else "")
                         + ".\n"
                         + "".join(f"  ✗ {t['title'][:50]} (fn={t.get('fn')}, "
                                   f"{t['attempt']} deneme)\n" for t in batan)
                         + "".join(f"  ⛔ {t['title'][:50]} — hiç koşmadı\n" for t in iptal))
                answer = (answer or f"{len(done_new)} düğüm tamamlandı.") + uyari
            if not answer and done_new:
                answer = f"{len(done_new)} düğüm tamamlandı."
            sess["history"].append({"role": "assistant", "text": answer[:600]})
            # kurulan grafı KALICI olarak sakla (sayfadan görüntülenecek)
            try:
                pid = P.save(goal=goal, pack=st.get("pack", "audit"),
                             backend=st["backend"], tasks=new,
                             stats={"dugum": len(new), "fn": fn_n, "ajan": ag_n,
                                    "cokme": res.crashes, "kurtarma": res.recovered,
                                    "retry": res.retries,
                                    "sn": round(time.time() - t0, 1)})
                emit({"type": "saved", "id": pid, "n": len(new)})
            except Exception:
                pass
            emit({"type": "board", "tasks": board_snapshot(board)})
            emit({"type": "summary",
                  "text": (f"{len(new)} düğüm · {fn_n} deterministik fonksiyon"
                           + (f" + {ag_n} LLM ajan" if ag_n else "")
                           + (f" · ✗{len(batan)} başarısız" if batan else "")
                           + (f" · ⛔{len(iptal)} iptal" if iptal else "")
                           + f" · {round(time.time()-t0,1)} sn"),
                  "crashes": res.crashes, "recovered": res.recovered,
                  "retries": res.retries, "failed": len(batan),
                  "cancelled": len(iptal), "answer": answer})
            return

        # ---- 2) ARAÇ DÖNGÜSÜ (graf kurmadan, sohbetin kendi çalışması) ----
        import agent as A
        import compaction as CP
        msgs = [{"role": "system", "content": router_prompt()}] + hist
        used, answer, shots = [], "", []

        for _round in range(4):
            if not tcs:
                answer = text
                break
            msgs.append({"role": "assistant", "content": text, "tool_calls": tcs})

            for tc in tcs:
                fn = tc["function"]["name"]
                try:
                    fa = json.loads(tc["function"].get("arguments") or "{}")
                except Exception:
                    fa = {}
                used.append(fn)

                if fn in A.TOOLS:
                    # ARAÇ: ham/büyük çıktı → LLM context'ine girer
                    out_txt = A.TOOLS[fn][0](**{k: v for k, v in fa.items()
                                                if k in A.TOOLS[fn][2]})
                    emit({"type": "phase", "text": f"araç → {fn}()"})
                    emit({"type": "log",
                          "text": f"araç {fn}({fa}) → {CP.est(out_txt):,} token HAM çıktı"})
                else:
                    # DÜĞÜM FONKSİYONU: yapılandırılmış, küçük çıktı
                    _pk = _fn_pack(fn)
                    if _pk and _pk != F.ACTIVE_PACK:
                        F.use_pack(_pk); st["pack"] = _pk
                        emit({"type": "log", "text": f"akış paketi seçildi: {_pk}"})
                    res_d = F.call(fn, fa, {})
                    # ÖNCE context'e ne gireceğini üret, SONRA ölç. Eskiden ölçüm
                    # `json.dumps(res_d)` üzerinden yapılıyordu — o dict `rows`'un
                    # TAMAMINI taşır ama context'e giren `ozet` (rows/text atılmış).
                    # Sonuç: extract_records gibi taşıyıcı fonksiyonlarda "indirgeme"
                    # EKSİ çıkıyordu (8.892 → 9.040, %-1,7). Artık ölçülen şey
                    # gerçekten LLM'e giden metin.
                    ozet = {k: v for k, v in res_d.items()
                            if k not in ("_raw_chars", "rows", "text")}
                    out_txt = json.dumps(ozet, ensure_ascii=False, default=str)
                    raw = F.raw_chars(res_d)
                    if raw:
                        emit({"type": "reduction", "kaynak": "sohbet",
                              "task": "-", "fn": fn, "title": "sohbet",
                              "raw_tokens": raw // 4, "out_tokens": CP.est(out_txt),
                              "pct": round((1 - len(out_txt) / raw) * 100, 1),
                              "args": fa,
                              "result": json.dumps(ozet, ensure_ascii=False,
                                                   indent=1)[:2500]})
                    emit({"type": "phase", "text": f"fonksiyon → {fn}()"})
                    emit({"type": "log", "text": f"fn {fn}({fa}) → {str(ozet)[:160]}"})
                    if "rapor_md" in res_d:
                        answer = res_d["rapor_md"]

                shots.append({"fn": fn, "args": fa,
                              "raw": out_txt[:3500],
                              "raw_full_chars": len(out_txt),
                              "raw_tokens": CP.est(out_txt)})
                msgs.append({"role": "tool", "name": fn,
                             "tool_call_id": tc.get("id", ""), "content": out_txt})

            # ---- TOOL-TRACE COMPACTION: araç çıktıları context'te birikti ----
            cres = CP.compact(st["strategy"], msgs, budget=st["budget"])
            # sıkıştırma SONRASI her tool mesajının hali (tıklayınca gösterilecek)
            after_by_fn = {}
            for m in cres.messages:
                v = CP._View(m)
                if v.role == "tool" and v.tool_name:
                    after_by_fn[v.tool_name] = v.content[:3500]
            det = [{**sh, "after": after_by_fn.get(sh["fn"], "(mesaj kaldırıldı/özete indi)")}
                   for sh in shots]
            if cres.triggered:
                msgs = cres.messages
            # TETİKLENMESE DE yayınla: panel bütün tool trafiğini göstermeli
            if shots:
                emit({"type": "compaction", "kaynak": "sohbet", "task": "-",
                      "title": f"sohbet ({', '.join(used[-2:])})",
                      "detail": det,
                      "events": [{"strategy": cres.strategy, "before": cres.before,
                                  "after": cres.after, "saved": cres.saved,
                                  "pct": round(cres.pct, 1), "triggered": cres.triggered,
                                  "log": cres.log[:3],
                                  "budget": st["budget"]}]})
            shots = []

            r2 = llm.chat(msgs, tools=_router_tools(), max_tokens=700, temperature=0.3)
            tcs = r2.get("tool_calls") or []
            text = (r2.get("content") or "").strip()
            if not tcs:
                answer = text or answer
                break

        if not answer:
            answer = "(sonuç üretilemedi)"
        sess["history"].append({"role": "assistant", "text": answer[:600]})
        emit({"type": "summary",
              "text": f"{len(used)} tool çağrısı · {', '.join(used)} · {round(time.time()-t0,1)} sn",
              "crashes": 0, "recovered": 0, "retries": 0, "answer": answer})

    except Exception as e:
        emit({"type": "error", "text": f"{type(e).__name__}: {e}"})
    finally:
        emit({"type": "done"})
        sess["busy"] = False


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _sse_pompa(self, evq, timeout: int = 240):
        """Olay kuyruğunu SSE olarak akıt. 'done' gelince ya da sessizlikte biter.

        (Mevcut /chat ve /runpipeline dallarındaki döngünün aynısı — yeni /wf/*
        uçları için tekrar yazmak yerine tek yerde.)
        """
        try:
            while True:
                ev = evq.get(timeout=timeout)
                self.wfile.write(("data: " + json.dumps(ev, ensure_ascii=False,
                                                        default=str) + "\n\n").encode())
                self.wfile.flush()
                if ev.get("type") == "done":
                    break
        except queue.Empty:
            pass
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}

        if u.path == "/":
            self._send(200, PAGE)

        elif u.path == "/meta":
            if q.get("sid"):
                F.use_pack(get_session(q["sid"])["settings"].get("pack", "audit"))
            self._send(200, json.dumps({
                "strategies": STRATEGY_INFO,
                "backends": O.BACKEND_INFO,
                "packs": F.pack_list(),
                "functions": {k: {"desc": v[1], "args": v[2]}
                              for k, v in F.REGISTRY.items()},
            }, ensure_ascii=False), "application/json; charset=utf-8")

        elif u.path == "/settings":
            sess = get_session(q.get("sid", "default"))
            for k in ("backend", "strategy", "crash_at", "fail_at", "pack",
                      "karsilastir"):
                if k in q:
                    sess["settings"][k] = q[k]
            if "budget" in q:
                sess["settings"]["budget"] = int(q["budget"])
            self._send(200, json.dumps({"ok": True, "settings": sess["settings"]},
                                       ensure_ascii=False),
                       "application/json; charset=utf-8")

        elif u.path == "/reset":
            sid = q.get("sid", "default")
            with _LOCK:
                SESSIONS.pop(sid, None)
            self._send(200, json.dumps({"ok": True}), "application/json; charset=utf-8")

        elif u.path == "/runpipeline":
            sid = q.get("sid", "default")
            doc = P.load(q.get("id", ""))
            sess = get_session(sid)
            if not doc:
                self._send(404, json.dumps({"error": "akış bulunamadı"}), "application/json")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            evq: queue.Queue = queue.Queue()

            def _go():
                try:
                    evq.put({"type": "start",
                             "text": f"kayıtlı akış {doc['id']} · {doc['pack']} · "
                                     f"{len(doc['nodes'])} düğüm"})
                    st = sess["settings"]
                    res = O.run_saved(doc["nodes"], backend=st["backend"],
                                      strategy=st["strategy"], budget=st["budget"],
                                      board=sess["board"],
                                      crash_at=st.get("crash_at") or None,
                                      fail_at=st.get("fail_at") or None,
                                      on_event=evq.put)
                    evq.put({"type": "board", "tasks": board_snapshot(sess["board"])})
                    evq.put({"type": "summary",
                             "text": (f"yeniden koşu · {res.tasks_done}/{res.tasks_created} "
                                      f"düğüm · {res.fn_tasks_run} fn · {res.seconds} sn"),
                             "crashes": res.crashes, "recovered": res.recovered,
                             "retries": res.retries,
                             "answer": f"'{doc['goal'][:80]}' akışı yeniden çalıştırıldı: "
                                       f"{res.tasks_done}/{res.tasks_created} düğüm tamamlandı "
                                       f"(planlama yapılmadı)."})
                except Exception as e:
                    evq.put({"type": "error", "text": f"{type(e).__name__}: {e}"})
                finally:
                    evq.put({"type": "done"})

            threading.Thread(target=_go, daemon=True).start()
            try:
                while True:
                    try:
                        ev = evq.get(timeout=240)
                    except queue.Empty:
                        break
                    self.wfile.write(("data: " + json.dumps(ev, ensure_ascii=False,
                                                            default=str) + "\n\n").encode())
                    self.wfile.flush()
                    if ev.get("type") == "done":
                        break
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif u.path == "/pipelines":
            self._send(200, json.dumps({"items": P.listing()}, ensure_ascii=False),
                       "application/json; charset=utf-8")

        # ───────── ZAMANLAMA (cron) ─────────
        elif u.path == "/schedules":
            items = SCH_STORE.list()
            for s in items:
                s["ne_zaman"] = SCH.cron_aciklama(s["cron"])
                s["son_kosular"] = SCH_STORE.runs(s["id"], limit=5)
            self._send(200, json.dumps({"items": items}, ensure_ascii=False),
                       "application/json; charset=utf-8")

        elif u.path == "/schedule/add":
            try:
                sid = SCH_STORE.create(
                    name=q.get("name", "").strip() or "adsız",
                    cron=q.get("cron", "").strip(),
                    pipeline_id=q.get("pipeline", "").strip(),
                    backend=q.get("backend", "own"),
                    strategy=q.get("strategy", "hermes"))
            except Exception as e:
                # geçersiz cron / eksik alan → 400 ve NEDENİ söyle
                self._send(400, json.dumps({"error": str(e)}, ensure_ascii=False),
                           "application/json; charset=utf-8")
                return
            s = SCH_STORE.get(sid)
            self._send(200, json.dumps({"ok": True, "id": sid, "next_run_at": s["next_run_at"],
                                        "ne_zaman": SCH.cron_aciklama(s["cron"])},
                                       ensure_ascii=False),
                       "application/json; charset=utf-8")

        elif u.path in ("/schedule/toggle", "/schedule/delete"):
            sid = q.get("id", "")
            if u.path.endswith("delete"):
                ok = SCH_STORE.delete(sid)
            else:
                cur = SCH_STORE.get(sid)
                ok = bool(cur) and SCH_STORE.set_enabled(sid, not cur["enabled"])
            self._send(200 if ok else 404,
                       json.dumps({"ok": ok}), "application/json; charset=utf-8")

        elif u.path == "/schedule/run":
            # "şimdi çalıştır" — takvimi KAYDIRMAZ (elle koşu bir sonraki cron'u ötelemez)
            sid = q.get("id", "")
            s = SCH_STORE.get(sid)
            if not s:
                self._send(404, json.dumps({"error": "zamanlama bulunamadı"}),
                           "application/json")
                return
            if not SCH_STORE.claim(sid):
                self._send(409, json.dumps({"error": "zaten koşuyor"}),
                           "application/json; charset=utf-8")
                return
            ok, ozet = SCH.kosturt(SCH_STORE, s)
            SCH_STORE.release(sid, ok, ileri_al=False)
            self._send(200, json.dumps({"ok": ok, "ozet": ozet}, ensure_ascii=False),
                       "application/json; charset=utf-8")

        # ───────── MOTOR KARŞILAŞTIRMA (workflow paneli) ─────────
        elif u.path == "/wf/plan":
            # Sohbet yolu LLM router'ından geçiyor ve plan_workflow yerine tool-loop
            # seçebiliyor. Panel plan_phase'i DOĞRUDAN çağırır → her seferinde graf.
            goal = (q.get("goal") or "").strip()
            if not goal:
                self._send(400, json.dumps({"error": "hedef boş"}), "application/json")
                return
            pack = q.get("pack") or "audit"
            if pack in F.PACKS:
                F.use_pack(pack)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            evq: queue.Queue = queue.Queue()

            def _plan():
                try:
                    evq.put({"type": "phase", "text": f"PLANLAMA — '{goal[:50]}' "
                                                      f"({pack} paketi)"})
                    b = TaskBoard()
                    O.plan_phase(b, goal, on_event=evq.put)
                    tasks = b.list_tasks()
                    if not tasks:
                        evq.put({"type": "error", "text": "planlayıcı hiç düğüm üretmedi"})
                        return
                    pid = P.save(goal=goal, pack=pack, backend="own", tasks=tasks,
                                 stats={"dugum": len(tasks), "kaynak": "panel"})
                    evq.put({"type": "saved", "id": pid, "n": len(tasks)})
                    evq.put({"type": "summary",
                             "text": f"{len(tasks)} düğümlü akış kuruldu · {pid}"})
                except Exception as e:
                    evq.put({"type": "error", "text": f"{type(e).__name__}: {str(e)[:200]}"})
                finally:
                    evq.put({"type": "done"})

            threading.Thread(target=_plan, daemon=True).start()
            self._sse_pompa(evq)

        elif u.path == "/wf/get":
            d = P.load(q.get("id", ""))
            if not d:
                self._send(404, json.dumps({"error": "akış bulunamadı"}), "application/json")
                return
            d["layers"] = [[n["id"] for n in lay] for lay in P.layers(d["nodes"])]
            # düğüm ayar çekmecesi için fonksiyon şeması (arg adları BİREBİR olmalı,
            # yoksa functions.call() bilinmeyen argümanı sessizce atıyor)
            d["katalog"] = {ad: {"aciklama": v[1], "args": v[2]}
                            for pk in F.PACKS.values() for ad, v in pk["fns"].items()}
            d["sim_modlari"] = {m: O.SIM_ACIKLAMA[m] for m in O.SIM_MODLARI}
            d["airflow_hazir"] = AR.hazir()[0]
            self._send(200, json.dumps(d, ensure_ascii=False),
                       "application/json; charset=utf-8")

        # ───────── MOTOR İNCELEME ─────────
        # Künye/dil/ayar verisi ayrı modüllerden geliyor; burası yalnız BORU.
        # Modüller HTTP bilmiyor, tek başlarına koşturulabiliyor (bkz. dosya başlıkları).
        elif u.path == "/motor/kunye":
            import motor_kunye as MK
            import motor_ayar as MA
            try:
                mot = q.get("motor") or ""
                veri = MK.hepsi() if not mot else {"motorlar": [MK.kunye(mot)]}
                veri["ayar"] = MA.hepsi()
            except Exception as e:
                veri = {"error": f"{type(e).__name__}: {e}"}
            self._send(200, json.dumps(veri, ensure_ascii=False),
                       "application/json; charset=utf-8")

        elif u.path == "/motor/deney":
            import motor_deney as MDY
            d = P.load(q.get("id", ""))
            try:
                veri = MDY.hepsi(d["nodes"] if d else [])
            except Exception as e:
                veri = {"error": f"{type(e).__name__}: {e}"}
            self._send(200, json.dumps(veri, ensure_ascii=False),
                       "application/json; charset=utf-8")

        elif u.path == "/motor/canvas":
            # Celery CANVAS — board YOK. Kullanıcının grafını chain/group/chord
            # ile gerçekten koşturur. Board'lu koşuyla farkı burada ölçülüyor.
            import motor_canvas as MC
            import functions as _F
            d = P.load(q.get("id", ""))
            if not d:
                self._send(404, json.dumps({"error": "akış bulunamadı"}),
                           "application/json; charset=utf-8")
                return
            try:
                _F.use_pack(d.get("pack", "data"))
                sim = json.loads(q.get("sim") or "{}")
                veri = MC.kostur(d["nodes"], sim=sim or None,
                                 zaman_asimi=int(q.get("bekle") or 90))
            except Exception as e:
                veri = {"ok": False, "hata": f"{type(e).__name__}: {e}",
                        "log": [], "eksikler": MC.EKSIKLER}
            self._send(200, json.dumps(veri, ensure_ascii=False),
                       "application/json; charset=utf-8")

        elif u.path == "/motor/dil":
            import motor_dili as MD
            d = P.load(q.get("id", ""))
            if not d:
                self._send(404, json.dumps({"error": "akış bulunamadı"}),
                           "application/json; charset=utf-8")
                return
            try:
                veri = MD.cevir(d["nodes"], d.get("goal", ""), q.get("motor") or None)
            except Exception as e:
                veri = {"error": f"{type(e).__name__}: {e}"}
            self._send(200, json.dumps(veri, ensure_ascii=False),
                       "application/json; charset=utf-8")

        elif u.path in ("/wf/run", "/wf/runall"):
            d = P.load(q.get("id", ""))
            if not d:
                self._send(404, json.dumps({"error": "akış bulunamadı"}), "application/json")
                return
            try:
                sim = json.loads(q.get("sim") or "{}")
                args = json.loads(q.get("args") or "{}")
            except Exception as e:
                self._send(400, json.dumps({"error": f"sim/args JSON değil: {e}"}),
                           "application/json")
                return
            hepsi = u.path.endswith("runall")
            motorlar = (list(O.BACKENDS) if hepsi
                        else [q.get("backend") or "own"])
            st = get_session(q.get("sid", "default"))["settings"]

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            evq2: queue.Queue = queue.Queue()

            def _go():
                satirlar = []
                try:
                    for be in motorlar:
                        evq2.put({"type": "motor_basladi", "backend": be,
                                  "text": f"══ {be} başlıyor ══"})
                        try:
                            r = wf_kostur(d["nodes"], be, sim, args,
                                          strategy=st.get("strategy", "hermes"),
                                          budget=st.get("budget", 3000),
                                          on_event=evq2.put)
                        except Exception as e:
                            r = {"backend": be, "ok": False, "sn": 0.0,
                                 "dugum": len(d["nodes"]), "done": 0, "failed": 0,
                                 "cancelled": 0, "retries": 0, "crashes": 0,
                                 "recovered": 0, "dugumler": [],
                                 "not": f"{type(e).__name__}: {str(e)[:160]}"}
                            evq2.put({"type": "log", "text": f"✗ {be}: {r['not']}"})
                        satirlar.append(r)
                        evq2.put({"type": "motor_bitti", **r})
                    if hepsi:
                        evq2.put({"type": "karsilastirma", "satirlar": satirlar})
                    evq2.put({"type": "summary",
                              "text": " · ".join(f"{x['backend']} {x['sn']}s "
                                                 f"({x['done']}/{x['dugum']})"
                                                 for x in satirlar)})
                except Exception as e:
                    evq2.put({"type": "error", "text": f"{type(e).__name__}: {str(e)[:200]}"})
                finally:
                    evq2.put({"type": "done"})

            threading.Thread(target=_go, daemon=True).start()
            self._sse_pompa(evq2)

        elif u.path == "/pipeline":
            d = P.load(q.get("id", ""))
            if not d:
                self._send(404, json.dumps({"error": "bulunamadı"}), "application/json")
                return
            d["layers"] = [[n["id"] for n in lay] for lay in P.layers(d["nodes"])]
            self._send(200, json.dumps(d, ensure_ascii=False),
                       "application/json; charset=utf-8")

        elif u.path == "/board":
            sess = get_session(q.get("sid", "default"))
            self._send(200, json.dumps({"tasks": board_snapshot(sess["board"]),
                                        "events": sess["board"].events()[-40:]},
                                       ensure_ascii=False),
                       "application/json; charset=utf-8")

        elif u.path == "/chat":
            sid = q.get("sid", "default")
            msg = (q.get("msg") or "").strip()
            sess = get_session(sid)
            if not msg:
                self._send(400, json.dumps({"error": "boş mesaj"}), "application/json")
                return
            if sess["busy"]:
                self._send(409, json.dumps({"error": "önceki tur sürüyor"}), "application/json")
                return
            sess["busy"] = True
            sess["history"].append({"role": "user", "text": msg})

            # --- SSE akışı ---
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            evq: queue.Queue = queue.Queue()
            th = threading.Thread(target=run_turn, args=(sess, msg, evq), daemon=True)
            th.start()
            try:
                while True:
                    try:
                        ev = evq.get(timeout=240)
                    except queue.Empty:
                        break
                    payload = json.dumps(ev, ensure_ascii=False, default=str)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    if ev.get("type") == "done":
                        break
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                sess["busy"] = False
        else:
            self._send(404, "not found")


PAGE = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Brain Agent</title>
<style>
:root{
  --bg:#faf9f7; --surface:#fff; --raised:#f2f0ed; --line:#e5e2dd; --line2:#d8d4ce;
  --ink:#1f1e1d; --ink2:#3d3c39; --muted:#6b6a67; --faint:#8f8d89;
  --accent:#c96442; --accent-ink:#a34f33; --accent-soft:#f5e9e3;
  --ok:#2f7a54; --warn:#9a6b1f; --bad:#b4423a;
  --code:#f5f3f0; --code-ink:#33322f;
  --sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  --serif:ui-serif,Georgia,"Times New Roman",serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --r:12px;
}
@media(prefers-color-scheme:dark){:root{
  --bg:#262624; --surface:#30302e; --raised:#3a3a37; --line:#43423f; --line2:#4e4d49;
  --ink:#f5f4f2; --ink2:#e3e1de; --muted:#a8a6a1; --faint:#87857f;
  --accent:#d97757; --accent-ink:#e89b80; --accent-soft:#3a2a23;
  --ok:#6cc08b; --warn:#d4a24c; --bad:#e07a6f;
  --code:#262624; --code-ink:#d8d6d2;
}}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.65;-webkit-font-smoothing:antialiased;display:flex;flex-direction:column}

/* ── üst bar ── */
header{display:flex;align-items:center;gap:8px;padding:10px 18px;border-bottom:1px solid var(--line);
  background:var(--bg);flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:9px;font-weight:600;font-size:15px;margin-right:6px}
.spark{width:17px;height:17px;flex:none}
.sp{flex:1}
.ctl{display:flex;align-items:center;gap:5px;background:var(--surface);border:1px solid var(--line);
  border-radius:999px;padding:3px 4px 3px 11px}
.ctl label{font-size:11px;color:var(--faint);font-weight:500}
select{font-family:inherit;font-size:12.5px;padding:3px 6px;border-radius:999px;border:none;
  background:transparent;color:var(--ink);cursor:pointer;outline:none}
.iconbtn{background:transparent;border:1px solid var(--line);color:var(--muted);
  border-radius:999px;padding:5px 12px;font-size:12.5px;font-family:inherit;cursor:pointer}
.iconbtn:hover{background:var(--raised);color:var(--ink)}

/* ── gövde ── */
.body{flex:1;display:flex;min-height:0}
.col{flex:1;display:flex;flex-direction:column;min-width:0}
.stream{flex:1;overflow-y:auto;padding:34px 22px 10px}
.thread{max-width:44rem;margin:0 auto}

/* ── karşılama ── */
.hero{text-align:center;padding:54px 10px 30px}
.hero h1{font-family:var(--serif);font-size:31px;font-weight:400;margin:0 0 10px;letter-spacing:-.01em}
.hero p{color:var(--muted);margin:0 auto;max-width:34rem;font-size:14.5px}
.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:22px}
.sug{background:var(--surface);border:1px solid var(--line);border-radius:999px;
  padding:7px 15px;font-size:13px;color:var(--ink2);cursor:pointer;font-family:inherit}
.sug:hover{border-color:var(--accent);color:var(--accent-ink)}

/* ── mesajlar ── */
.turn{margin-bottom:26px}
.turn.me{display:flex;justify-content:flex-end}
.turn.me .txt{background:var(--raised);border-radius:var(--r);padding:11px 16px;max-width:82%;
  white-space:pre-wrap}
.who{display:flex;align-items:center;gap:8px;margin-bottom:9px;font-size:12.5px;
  color:var(--muted);font-weight:600}
.av{width:20px;height:20px;border-radius:5px;background:var(--accent);flex:none;
  display:grid;place-items:center;color:#fff;font-size:11px;font-weight:700}
.md{white-space:pre-wrap}
.md strong{font-weight:650}

/* ── tool bloğu (katlanır) ── */
.tool{border:1px solid var(--line);border-radius:10px;background:var(--surface);
  margin:10px 0;overflow:hidden}
.tool>summary{list-style:none;cursor:pointer;padding:9px 13px;display:flex;align-items:center;
  gap:9px;font-size:13px;color:var(--ink2);user-select:none}
.tool>summary::-webkit-details-marker{display:none}
.tool>summary:hover{background:var(--raised)}
.chev{transition:transform .15s;color:var(--faint);flex:none}
.tool[open] .chev{transform:rotate(90deg)}
.tool .lab{font-family:var(--mono);font-size:12px}
.tool .meta{margin-left:auto;font-size:11.5px;color:var(--faint);font-variant-numeric:tabular-nums}
.tool .out{padding:10px 13px;border-top:1px solid var(--line);background:var(--code);
  color:var(--code-ink);font-family:var(--mono);font-size:11.5px;line-height:1.6;
  white-space:pre-wrap;word-break:break-word;max-height:320px;overflow:auto}
.out .k{color:var(--ok)}.out .r{color:var(--bad)}.out .b{color:var(--accent-ink)}.out .d{color:var(--faint)}

/* ── durum satırı ── */
.status{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted);margin:2px 0 8px}
.pulse{width:6px;height:6px;border-radius:50%;background:var(--accent);animation:pl 1.2s ease-in-out infinite}
@keyframes pl{0%,100%{opacity:.25;transform:scale(.85)}50%{opacity:1;transform:scale(1)}}

/* ── rozetler ── */
.foot{display:flex;flex-wrap:wrap;gap:6px;margin-top:11px}
.tag{font-size:11.5px;padding:3px 10px;border-radius:999px;background:var(--raised);
  color:var(--muted);border:1px solid var(--line)}
.tag.good{background:var(--accent-soft);color:var(--accent-ink);border-color:transparent}
.tag.warn{color:var(--warn)}.tag.bad{color:var(--bad)}

/* ── yazma alanı ── */
.dock{padding:14px 22px 20px;background:linear-gradient(to bottom,transparent,var(--bg) 28%)}
.box{max-width:44rem;margin:0 auto;background:var(--surface);border:1px solid var(--line2);
  border-radius:16px;padding:11px 12px 9px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.box:focus-within{border-color:var(--accent)}
textarea{width:100%;border:none;outline:none;resize:none;background:transparent;color:var(--ink);
  font-family:inherit;font-size:15px;line-height:1.6;max-height:180px;min-height:24px}
.boxrow{display:flex;align-items:center;gap:8px;margin-top:7px}
.hintline{font-size:11.5px;color:var(--faint);flex:1}
.send{width:32px;height:32px;border-radius:9px;border:none;background:var(--accent);color:#fff;
  cursor:pointer;display:grid;place-items:center;flex:none}
.send:disabled{opacity:.35;cursor:default}

/* ── yan panel ── */
aside{width:322px;border-left:1px solid var(--line);background:var(--surface);
  overflow-y:auto;padding:16px 15px}
aside.hide{display:none}
.tabs{display:flex;gap:4px;margin-bottom:13px}
.tab{flex:1;font-size:12px;font-weight:600;padding:6px 8px;border-radius:8px;background:transparent;
  color:var(--muted);border:1px solid transparent;cursor:pointer;font-family:inherit}
.tab.on{background:var(--raised);color:var(--ink);border-color:var(--line)}
aside h3{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--faint);
  margin:0 0 8px;font-weight:650}
.card{border:1px solid var(--line);border-radius:9px;padding:8px 11px;margin-bottom:7px;background:var(--bg)}
.card .t{font-size:12.5px;font-weight:600;line-height:1.4}
.card .m{font-family:var(--mono);font-size:10.5px;color:var(--faint);margin-top:3px}
.dot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:5px;vertical-align:1px}
.d-done{background:var(--ok)}.d-ready{background:var(--accent)}.d-blocked{background:var(--line2)}
.d-running{background:var(--warn)}.d-failed{background:var(--bad)}
/* iptal: üstü battığı için ASLA koşmayacak düğüm — 'bekliyor'dan (blocked) ayrı görünsün */
.d-cancelled{background:transparent;border:2px solid var(--bad);opacity:.75}
.num{font-family:var(--mono);font-size:13px;font-variant-numeric:tabular-nums}
.gain{color:var(--ok);font-weight:650}
.meter{height:4px;border-radius:2px;background:var(--line);margin-top:6px;overflow:hidden}
.meter i{display:block;height:100%;background:var(--accent)}
.total{border:1px solid var(--accent);background:var(--accent-soft);border-radius:10px;
  padding:10px 12px;margin-bottom:12px}
.total .num{font-size:16px;font-weight:700;color:var(--accent-ink)}
.note{font-size:12px;color:var(--faint);line-height:1.6}
@media(max-width:920px){aside{display:none}}

/* ── akış galerisi (overlay) ── */
.ov{position:fixed;inset:0;background:rgba(0,0,0,.34);display:none;z-index:40;
  padding:26px;overflow:auto}
.ov.on{display:block}
.sheet{max-width:62rem;margin:0 auto;background:var(--bg);border:1px solid var(--line);
  border-radius:14px;padding:20px 22px;min-height:60vh}
.sheet .top{display:flex;align-items:center;gap:10px;margin-bottom:16px}
.sheet h2{font-family:var(--serif);font-weight:400;font-size:22px;margin:0}
.plist{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
.pcard{border:1px solid var(--line);border-radius:11px;padding:12px 14px;background:var(--surface);
  cursor:pointer}
.pcard:hover{border-color:var(--accent)}
.pcard .g{font-size:13.5px;font-weight:600;line-height:1.45;margin-bottom:6px}
.pcard .s{font-size:11.5px;color:var(--faint);font-family:var(--mono)}
.pcard .b{display:flex;gap:5px;margin-top:8px;flex-wrap:wrap}

/* ── DAG çizimi ── */
.dag{display:flex;gap:0;align-items:stretch;overflow-x:auto;padding:8px 2px 14px}
.lay{display:flex;flex-direction:column;gap:12px;justify-content:center;min-width:190px}
.arrow{display:flex;align-items:center;color:var(--line2);padding:0 4px;flex:none}
.nd{border:1px solid var(--line2);border-radius:10px;padding:9px 12px;background:var(--surface);
  min-width:180px;max-width:230px}
.nd.fn{border-left:3px solid var(--accent)}
.nd.agent{border-left:3px solid var(--warn)}
.nd .n1{font-family:var(--mono);font-size:11.5px;font-weight:650;color:var(--accent-ink)}
.nd.agent .n1{color:var(--warn)}
.nd .n2{font-size:12px;line-height:1.4;margin:3px 0 4px}
.nd .n3{font-family:var(--mono);font-size:10.5px;color:var(--faint);word-break:break-all}
/* ── MOTOR PANELİ: tıklanabilir düğüm + simülasyon işareti ── */
.nd.tik{cursor:pointer}
.nd.tik:hover{border-color:var(--accent)}
.nd.sec{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent-soft)}
.nd.simli{border-left-width:5px}
.nd.sim-gecici,.nd.sim-yavas{border-left-color:var(--warn)}
.nd.sim-kalici,.nd.sim-sonra,.nd.sim-cokme{border-left-color:var(--bad)}
.simrozet{display:inline-block;font-size:9.5px;padding:1px 5px;border-radius:6px;
  margin-top:4px;background:var(--bad);color:#fff;letter-spacing:.02em}
.simrozet.uyari{background:var(--warn)}
/* ── MOTOR İNCELEME ── */
.mtabs{display:flex;gap:4px;flex-wrap:wrap;margin:14px 0 0}
.mtb{font-family:var(--sans);font-size:13px;text-align:left;color:var(--muted);
 background:var(--raised);border:1px solid var(--line);border-bottom-color:transparent;
 border-radius:10px 10px 0 0;padding:9px 14px;cursor:pointer;display:flex;
 flex-direction:column;gap:2px;line-height:1.25}
.mtb:hover{color:var(--ink)}
.mtb.on{color:var(--ink);background:var(--surface);border-color:var(--accent);
 border-bottom-color:var(--surface);font-weight:650}
.mtb .g{font-family:var(--mono);font-size:10px;color:var(--accent)}
.mpane{background:var(--surface);border:1px solid var(--line);border-radius:0 12px 12px 12px;
 padding:18px;margin-top:-1px}
.mh{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
 color:var(--faint);margin:20px 0 9px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.mh:first-child{margin-top:0}
.mcumle{font-family:var(--serif);font-size:17px;line-height:1.5;color:var(--ink);margin:0 0 12px}
.manaloji{background:var(--accent-soft);border-left:3px solid var(--accent);
 border-radius:0 8px 8px 0;padding:10px 13px;margin-bottom:12px}
.manaloji b{display:block;font-size:12.5px;color:var(--accent-ink);margin-bottom:3px}
.manaloji span{font-size:13px;color:var(--ink2);line-height:1.55}
.mmim{font-family:var(--mono);font-size:11px;line-height:1.45;background:var(--code);
 color:var(--code-ink);border:1px solid var(--line);border-radius:9px;padding:12px;
 white-space:pre;overflow-x:auto;margin-bottom:6px}
.mrow{display:flex;gap:9px;padding:8px 11px;border-bottom:1px solid var(--line);align-items:flex-start}
.mrow:last-child{border-bottom:none}
.mim{font-family:var(--mono);font-size:13px;width:15px;flex:none;padding-top:1px}
.mim.a{color:var(--ok)} .mim.k{color:var(--faint)} .mim.e{color:var(--warn)}
.mrow .b{flex:1;min-width:0}
.mrow .t{font-size:13px;font-weight:600}
.mrow .d{font-size:12px;color:var(--muted);line-height:1.5;margin-top:2px}
.mrow .n{font-size:12px;color:var(--accent-ink);line-height:1.5;margin-top:3px}
.mrow .o{font-family:var(--mono);font-size:11px;color:var(--faint);margin-top:3px}
.mkanit{font-family:var(--mono);font-size:9.5px;color:var(--faint);border:1px solid var(--line);
 border-radius:5px;padding:1px 6px;white-space:nowrap;margin-left:6px}
.minc{border:1px solid var(--line);border-left:3px solid var(--warn);border-radius:0 9px 9px 0;
 padding:10px 13px;margin-bottom:8px;background:var(--raised)}
.minc .t{font-size:13px;font-weight:650;color:var(--ink)}
.minc .d{font-size:12.5px;color:var(--ink2);line-height:1.55;margin-top:4px}
.minc .s{font-size:12.5px;color:var(--warn);line-height:1.5;margin-top:4px}
.mgz{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:820px){.mgz{grid-template-columns:1fr}}
.mgz .kt2{border:1px solid var(--line);border-radius:9px;padding:11px 13px;background:var(--raised)}
.mgz .kt2 h4{margin:0 0 7px;font-family:var(--mono);font-size:10px;letter-spacing:.09em;
 text-transform:uppercase}
.mgz .g h4{color:var(--ok)} .mgz .z h4{color:var(--bad)}
.mgz li{font-size:12.5px;line-height:1.55;margin-bottom:7px;color:var(--ink2)}
.mgz ul{margin:0;padding-left:16px}
.mil{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px;
 background:var(--raised)}
.mil .t{font-size:13.5px;font-weight:650}
.mil .k{font-size:12.5px;color:var(--ink2);line-height:1.55;margin-top:5px}
.mil .y{font-size:12.5px;color:var(--accent-ink);line-height:1.5;margin-top:5px;font-style:italic}
.mtab{width:100%;border-collapse:collapse;font-size:12px}
.mtab th{text-align:left;font-family:var(--mono);font-size:10px;letter-spacing:.07em;
 text-transform:uppercase;color:var(--faint);padding:6px 8px;border-bottom:1px solid var(--line)}
.mtab td{padding:6px 8px;border-bottom:1px solid var(--line);color:var(--ink2);vertical-align:top}
.mtab td.m{font-family:var(--mono);font-size:11px}
.mkod{font-family:var(--mono);font-size:11px;line-height:1.5;background:var(--code);
 color:var(--code-ink);border:1px solid var(--line);border-radius:9px;padding:12px;
 white-space:pre-wrap;word-break:break-word;max-height:340px;overflow:auto;margin:0}
.muyari{background:var(--accent-soft);border:1px solid var(--accent);border-radius:9px;
 padding:10px 13px;font-size:12.5px;line-height:1.55;color:var(--ink2);margin-bottom:12px}

.uyari-kutu{border:1px solid var(--warn);border-radius:10px;padding:11px 14px;
 margin:14px 0 6px;background:var(--raised);font-size:12.5px;line-height:1.6;color:var(--ink2)}
.uyari-kutu b{color:var(--ink)}
.uyari-kutu code{font-family:var(--mono);font-size:11px;background:var(--code);
 color:var(--code-ink);padding:1px 5px;border-radius:4px}
.uyari-kutu li{margin-bottom:5px}
.mbay{border:1px solid var(--accent);border-radius:10px;padding:12px 14px;margin-bottom:12px;
 background:var(--accent-soft)}
.mbay .bh{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;
 color:var(--accent-ink);margin-bottom:5px}
.mbay .bt{font-size:15px;font-weight:700;color:var(--ink);margin-bottom:5px}
.mbay .bo{font-size:12.5px;line-height:1.6;color:var(--ink2)}
.mbay .bn{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px;
 font-size:12px;color:var(--ink2)}
.btag{font-family:var(--mono);font-size:10px;padding:2px 8px;border-radius:5px;
 border:1px solid currentColor;white-space:nowrap}
.btag.deney{color:var(--ok)} .btag.goster{color:var(--warn)} .btag.anlat{color:var(--faint)}
.mbay .bs{font-size:11.5px;color:var(--muted);margin-top:7px;padding-top:7px;
 border-top:1px solid var(--line);line-height:1.5}
.mbir{border:1px solid var(--line);border-radius:9px;padding:11px 13px;margin-bottom:8px;
 background:var(--raised)}
.mbir.bizde{border-color:var(--ok)}
.mbir .yt{font-size:13.5px;font-weight:650;display:flex;gap:8px;align-items:center}
.ybiz{font-family:var(--mono);font-size:9.5px;color:var(--ok);border:1px solid var(--ok);
 border-radius:5px;padding:1px 6px}
.mbir .yn{font-size:12.5px;color:var(--ink2);line-height:1.55}
.mbir .yo{font-size:12px;color:var(--muted);line-height:1.5;margin-top:4px;font-style:italic}
.mger{border:1px solid var(--line);border-radius:9px;padding:11px 13px;background:var(--raised)}
.mger .gd{font-size:12.5px;color:var(--ink2);line-height:1.6;margin-bottom:4px}
.mger ul{margin:7px 0 0;padding-left:17px}
.mger li{font-size:12.5px;line-height:1.55;color:var(--ink2)}
.mger .gu{font-size:11px;color:var(--faint);margin-top:8px;padding-top:8px;
 border-top:1px solid var(--line);font-style:italic}
.miyi .in{font-family:var(--mono);font-size:9.5px;letter-spacing:.09em;
 text-transform:uppercase;color:var(--muted);margin:8px 0 3px}
.miyi .iki{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:6px}
@media(max-width:820px){.miyi .iki{grid-template-columns:1fr}}
.mmen{border:1px solid var(--line);border-left:3px solid var(--accent);
 border-radius:0 10px 10px 0;padding:12px 14px;margin-bottom:14px;background:var(--raised)}
.mmen .mh2{font-family:var(--mono);font-size:10px;letter-spacing:.09em;
 text-transform:uppercase;color:var(--faint);margin-bottom:7px;display:flex;
 justify-content:space-between;gap:8px}
.mmen .mh2 span{color:var(--accent-ink)}
.mmen .minanc{font-family:var(--serif);font-size:15.5px;line-height:1.45;
 color:var(--ink);margin-bottom:9px}
.mmen .md2{font-size:12.5px;line-height:1.6;color:var(--ink2);margin-bottom:5px}
.mmen ul{margin:3px 0 0;padding-left:17px}
.mmen li{font-size:12.5px;line-height:1.55;color:var(--ink2);margin-bottom:3px}
.miyi{border:1px solid var(--ok);border-radius:10px;padding:12px 14px;margin-bottom:14px;
 background:color-mix(in srgb,var(--ok) 7%,var(--surface))}
.miyi .ih{font-size:13px;color:var(--ok);margin-bottom:5px}
.miyi .ih b{font-size:14.5px}
.miyi .ic{font-family:var(--serif);font-size:15px;color:var(--ink);margin-bottom:8px;line-height:1.45}
.miyi ul{margin:0;padding-left:17px}
.miyi li{font-size:12.5px;line-height:1.6;color:var(--ink2);margin-bottom:4px}
.miyi .ip{font-size:12.5px;color:var(--ink2);margin-top:8px;padding-top:8px;
 border-top:1px solid var(--line);line-height:1.5}
.mnz{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
@media(max-width:820px){.mnz{grid-template-columns:1fr}}
.mnz>div{font-size:12.5px;line-height:1.55;color:var(--ink2);background:var(--raised);
 border:1px solid var(--line);border-radius:8px;padding:9px 12px}
.mnz b{color:var(--ink)}
.mden{border:1px solid var(--line);border-radius:10px;padding:11px 13px;margin-bottom:9px;
 background:var(--raised)}
.mden .dh{display:flex;gap:10px;align-items:flex-start}
.mden .dh .b{flex:1;min-width:0}
.mden .dh .t{font-size:13.5px;font-weight:650}
.mden .dh .d{font-size:12.5px;color:var(--muted);line-height:1.5;margin-top:3px}
.mden .dk{font-family:var(--mono);font-size:10.5px;color:var(--faint);margin-top:7px}
.mden .db{font-size:12.5px;color:var(--ink2);line-height:1.5;margin-top:4px}
.mden .dc{font-size:12px;color:var(--accent-ink);line-height:1.5;margin-top:4px;font-style:italic}
.mden .dfark{font-size:12.5px;line-height:1.6;color:var(--ink2);margin-top:9px;
 padding:9px 12px;background:var(--surface);border:1px solid var(--accent);border-radius:8px}
.mden .dgos{margin-top:8px;padding:8px 11px;background:var(--surface);
 border:1px solid var(--ok);border-radius:8px;font-size:12px;line-height:1.5}
.mden .dgos b{color:var(--ok)}
.mden .dgos ul{margin:4px 0 0;padding-left:17px}
.mden .dgos li{color:var(--ink2);margin-bottom:2px}
.dyon{font-family:var(--mono);font-size:10.5px;color:var(--accent-ink);
 border:1px dashed var(--accent);border-radius:6px;padding:5px 9px;white-space:nowrap}
.mden .dsonuc{margin-top:9px}
.mden .dsn{font-family:var(--mono);font-size:12px;padding:7px 10px;border-radius:7px;
 background:var(--surface);border:1px solid var(--line)}
.mden .dlog{font-family:var(--mono);font-size:10.5px;color:var(--muted);white-space:pre-wrap;
 background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:8px 10px}
details.mdet{margin-top:12px;border:1px solid var(--line);border-radius:9px;padding:0}
details.mdet>summary{cursor:pointer;padding:10px 13px;font-size:12.5px;color:var(--muted);
 font-family:var(--mono)}
details.mdet>summary:hover{color:var(--ink)}
details.mdet[open]>summary{border-bottom:1px solid var(--line)}
details.mdet>*:not(summary){margin:12px 13px}

/* karşılaştırma tablosu */
.kt{width:100%;border-collapse:collapse;font-size:12px;margin-top:10px}
.kt th,.kt td{padding:6px 8px;text-align:left;border-bottom:1px solid var(--line)}
.kt th{font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--faint)}
.kt td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}
.kt tr.iyi td:first-child{border-left:3px solid var(--ok)}
.kt tr.kotu td:first-child{border-left:3px solid var(--bad)}
.wflog{font-family:var(--mono);font-size:11px;line-height:1.55;max-height:230px;
  overflow-y:auto;background:var(--code);border:1px solid var(--line);border-radius:9px;
  padding:9px 11px;white-space:pre-wrap;margin-top:10px}
.exp{border:1px solid var(--line);border-radius:9px;margin-bottom:7px;background:var(--bg);overflow:hidden}
.exp>summary{list-style:none;cursor:pointer;padding:8px 11px}
.exp>summary::-webkit-details-marker{display:none}
.exp>summary:hover{background:var(--raised)}
.exp .body2{border-top:1px solid var(--line);padding:8px 11px;background:var(--code)}
.exp .body2 h4{margin:0 0 4px;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--faint);font-weight:650}
.exp pre{margin:0 0 9px;font-family:var(--mono);font-size:10.5px;line-height:1.55;
  color:var(--code-ink);white-space:pre-wrap;word-break:break-word;max-height:190px;overflow:auto}
.nd .n4{font-family:var(--mono);font-size:10.5px;color:var(--muted);margin-top:6px;
  padding-top:6px;border-top:1px dashed var(--line);white-space:pre-wrap;max-height:66px;overflow:auto}
</style></head><body>

<header>
  <div class="brand">
    <svg class="spark" viewBox="0 0 24 24" fill="var(--accent)"><path d="M12 2l2.4 6.6L21 11l-6.6 2.4L12 20l-2.4-6.6L3 11l6.6-2.4z"/></svg>
    Brain Agent
  </div>
  <div class="ctl"><label>akış</label><select id="pack"></select></div>
  <div class="ctl"><label>motor</label><select id="backend"></select></div>
  <div class="ctl"><label>compaction</label><select id="strategy"></select></div>
  <div class="ctl"><label>bütçe</label><select id="budget">
    <option value="1500">1.5K</option><option value="3000" selected>3K</option>
    <option value="8000">8K</option><option value="30000">30K</option></select></div>
  <div class="ctl"><label>karşılaştır</label><select id="cmpall"
    title="Açıkken sohbette kurulan graf DÖRT motorda da koşar ve her birinin çıktısı ayrı basılır. Celery ~15-40 sn ekler.">
    <option value="1" selected>4 motorda koştur</option>
    <option value="0">kapalı (tek motor)</option></select></div>
  <div class="ctl"><label>çökme</label><select id="crash">
    <option value="">—</option><option value="oku">oku</option><option value="tara">tara</option>
    <option value="test">test</option><option value="rapor">rapor</option></select></div>
  <span class="sp"></span>
  <button class="iconbtn" id="flows">Akışlar</button>
  <button class="iconbtn" id="wf">Motorlar</button>
  <button class="iconbtn" id="mi">Akış üret</button>
  <button class="iconbtn" id="scheds">Zamanlama</button>
  <button class="iconbtn" id="panel">Panel</button>
  <button class="iconbtn" id="reset">Yeni</button>
</header>

<div class="ov" id="ov"><div class="sheet">
  <div class="top">
    <h2 id="ovtitle">Kurulan akışlar</h2><span class="sp"></span>
    <button class="iconbtn" id="ovback" style="display:none">← liste</button>
    <button class="iconbtn" id="ovclose">Kapat</button>
  </div>
  <div id="ovbody"></div>
</div></div>

<div class="body">
  <div class="col">
    <div class="stream" id="stream"><div class="thread" id="thread">
      <div class="hero" id="hero">
        <h1>Ne yapalım?</h1>
        <p>Sohbet edebilir, tek bir işlem çalıştırabilir ya da çok adımlı bir iş akışı (DAG) kurabilirim.
           Düğümlerin çoğu deterministik fonksiyondur; LLM yalnız gerekince devreye girer.</p>
        <div class="chips" id="sugs"></div>
      </div>
    </div></div>
    <div class="dock"><div class="box">
      <textarea id="inp" rows="1" placeholder="Bir şey iste…"></textarea>
      <div class="boxrow">
        <span class="hintline" id="hintline">Enter gönder · Shift+Enter satır</span>
        <button class="send" id="send" title="Gönder">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 19V5M5 12l7-7 7 7"/></svg></button>
      </div>
    </div></div>
  </div>

  <aside id="aside">
    <div class="tabs">
      <button class="tab on" data-t="board">Board</button>
      <button class="tab" data-t="trace">Tool-trace</button>
      <button class="tab" data-t="fns">Fonksiyonlar</button>
    </div>
    <div id="p-board"><p class="note">Henüz düğüm yok.</p></div>
    <div id="p-trace" style="display:none">
      <div id="ttot"></div>
      <p class="note" style="margin:0 0 10px">Yalnız <b>sohbet asistanının kendi
        context'ine giren</b> tool trafiği. Akış (DAG) düğümleri burada listelenmez —
        onların çıktısı board'a ve ardıl düğüme gider, sohbetin context'ine girmez.</p>
      <h3>1 · Deterministik indirgeme</h3>
      <div id="tred"><p class="note">Henüz yok — sohbette bir fonksiyon çağrılınca dolar.</p></div>
      <h3 style="margin-top:15px">2 · LLM compaction</h3>
      <div id="ttrc"><p class="note">Henüz yok — compaction yalnız LLM context'ine giren veride çalışır.</p></div>
    </div>
    <div id="p-fns" style="display:none"></div>
  </aside>
</div>

<script>
const SID="s_"+Math.random().toString(36).slice(2,10);
let META=null, busy=false, TRACE=[], RED=[];
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const paint=s=>esc(s)
  .replace(/(✓|done|tamamlandı|KORUNDU|DEVAM|hazır)/g,'<span class="k">$1</span>')
  .replace(/(✗|ÇÖKME|hata|HATA|failed|reddedildi)/g,'<span class="r">$1</span>')
  .replace(/\b(fn|claim|recover_stale|recompute_ready|PLAN|upstream)\b/g,'<span class="b">$1</span>')
  .replace(/(←[^\n]*)/g,'<span class="d">$1</span>');
const SUGS={
  audit:["auth/login.py'ı oku ve mfa_token tara","testleri koştur",
         "oku, tara, testleri koştur, eşleştir ve rapor üret"],
  data:["siparişleri çek","çek, doğrula, normalize et, topla ve CSV çıkar"],
  deploy:["1.4.2 paketle ve duman testi koştur","paketle, test et, canary yayınla, sağlığa bak"],
};

async function loadMeta(){
  META=await (await fetch("/meta?sid="+SID)).json();
  const ps=$("pack"); for(const [k,v] of Object.entries(META.packs)){
    const o=document.createElement("option");o.value=k;o.textContent=k;o.title=v.aciklama;ps.appendChild(o);}
  const bs=$("backend"); for(const k of Object.keys(META.backends)){
    const o=document.createElement("option");o.value=k;o.textContent=k;bs.appendChild(o);}
  const ss=$("strategy"); for(const k of Object.keys(META.strategies)){
    const o=document.createElement("option");o.value=k;o.textContent=k;
    if(k==="hermes")o.selected=true;ss.appendChild(o);}
  ps.onchange=async()=>{await push();await fns();sugs();};
  for(const el of ["backend","strategy","budget","crash","cmpall"]) $(el).onchange=push;
  await push(); await fns(); sugs();
}
async function push(){
  await fetch("/settings?"+new URLSearchParams({sid:SID,pack:$("pack").value,
    backend:$("backend").value,strategy:$("strategy").value,
    budget:$("budget").value,crash_at:$("crash").value}));
}
function sugs(){
  const pk=$("pack").value||"audit";
  $("sugs").innerHTML=(SUGS[pk]||[]).map(s=>`<button class="sug">${esc(s)}</button>`).join("");
  document.querySelectorAll(".sug").forEach(b=>b.onclick=()=>{$("inp").value=b.textContent;send();});
}
async function fns(){
  const m=await (await fetch("/meta?sid="+SID)).json();
  const pk=$("pack").value||"audit", p=m.packs[pk]||{fns:[]};
  $("p-fns").innerHTML=`<p class="note" style="margin:0 0 10px">${esc(p.aciklama||"")}</p>`+
    (p.fns||[]).map(k=>{const v=m.functions[k]||{args:{}};
      return `<div class="card"><div class="t">${esc(k)}</div>
      <div class="m">${esc(Object.keys(v.args||{}).join(" · ")||"argümansız")}</div></div>`;}).join("");
}
function board(tasks){
  if(!tasks||!tasks.length){$("p-board").innerHTML='<p class="note">Henüz düğüm yok.</p>';return;}
  $("p-board").innerHTML=tasks.map(t=>`<div class="card">
    <div class="t"><span class="dot d-${t.status}"></span>${esc(t.title)}</div>
    <div class="m">${t.kind==="function"?"fn:"+esc(t.fn):"LLM ajan"}${
      t.parents.length?"  ←  "+esc(t.parents.join(", ")):""}</div></div>`).join("");
}
// ───────── TOOL-TRACE FİLTRESİ ─────────
// Panel YALNIZ sohbet asistanının kendi context'ine giren tool trafiğini gösterir.
// Akış (DAG) düğümlerinin çıktısı board'a + ardıl düğüme gider, sohbetin context'ine
// hiç girmez → tool-trace'e ait değildir. Olaylar `kaynak` ile etiketleniyor:
//   "sohbet" → araç döngüsü, LLM context'ine girer  → panelde
//   "akis"   → workflow düğümü, board'a gider        → yalnız koşu logunda
const izlenir=ev=>ev.kaynak==="sohbet";
function izleRed(ev){ if(!izlenir(ev))return; RED.push(ev); red(); }
function izleTrc(ev){ if(!izlenir(ev))return;
  for(const c of ev.events) TRACE.push({...c,node:ev.title,detail:ev.detail}); trc(); }
function totals(){
  const T2=TRACE.filter(e=>e.triggered!==false);
  const h=T2.reduce((a,e)=>a+e.before,0)+RED.reduce((a,e)=>a+e.raw_tokens,0);
  const k=T2.reduce((a,e)=>a+e.after,0)+RED.reduce((a,e)=>a+e.out_tokens,0);
  if(!h)return;
  $("ttot").innerHTML=`<div class="total"><h3 style="margin:0 0 3px">toplam indirgeme</h3>
    <div class="num">${h.toLocaleString("tr")} → ${k.toLocaleString("tr")}</div>
    <div class="note">%${((h-k)/h*100).toFixed(1)} · ${RED.length} fonksiyon + ${T2.length} compaction${
      TRACE.length>T2.length?` · ${TRACE.length-T2.length} tetiklenmedi`:""}</div></div>`;
}
function red(){ if(!RED.length)return;
  $("tred").innerHTML=RED.map(e=>`<details class="exp"><summary>
    <div class="t">fn:${esc(e.fn)}</div>
    <div class="num">${e.raw_tokens.toLocaleString("tr")} → ${e.out_tokens.toLocaleString("tr")}
      <span class="gain">−${e.pct}%</span></div>
    <div class="meter"><i style="width:${Math.min(100,e.pct)}%"></i></div>
    </summary><div class="body2">
      ${e.args&&Object.keys(e.args).length?`<h4>argümanlar</h4><pre>${esc(JSON.stringify(e.args))}</pre>`:""}
      <h4>ham veri</h4><pre>${e.raw_tokens.toLocaleString("tr")} token işlendi (LLM'e hiç gitmedi)</pre>
      <h4>yapılandırılmış sonuç — düğümün ÇIKTISI</h4>
      <pre>${esc(e.result||"(yok)")}</pre>
    </div></details>`).reverse().join("");
  totals();}
function trc(){ if(!TRACE.length)return;
  $("ttrc").innerHTML=TRACE.map(e=>{const p=e.before?((e.before-e.after)/e.before*100):0;
    const det=(e.detail||[]).map(d=>`
      <h4>${esc(d.fn)}${d.args&&Object.keys(d.args).length?" "+esc(JSON.stringify(d.args)):""}
        — ÖNCE (${d.raw_tokens.toLocaleString("tr")} token, ${d.raw_full_chars.toLocaleString("tr")} kar)</h4>
      <pre>${esc(d.raw)}${d.raw_full_chars>d.raw.length?"\n…[önizleme kesildi]":""}</pre>
      <h4>${esc(d.fn)} — SONRA (compaction sonrası context'te kalan)</h4>
      <pre>${esc(d.after)}</pre>`).join("");
    const off = e.triggered===false;
    return `<details class="exp"${off?' style="opacity:.72"':''}><summary>
      <div class="t">${esc(e.strategy)} · <span style="color:var(--faint);font-weight:400">${esc(e.node||"")}</span></div>
      <div class="num">${e.before.toLocaleString("tr")} → ${e.after.toLocaleString("tr")}
        ${off?'<span style="color:var(--faint);font-size:11px">tetiklenmedi</span>'
             :`<span class="gain">−${p.toFixed(1)}%</span>`}</div>
      ${off?`<div class="note" style="font-size:10.5px;margin-top:3px">bütçe ${(e.budget||0).toLocaleString("tr")} · context altında kaldı</div>`
           :`<div class="meter"><i style="width:${Math.min(100,p)}%"></i></div>`}
      </summary><div class="body2">
        ${det||"<h4>içerik yakalanmadı (graf içi ajan düğümü)</h4>"}
        <h4>strateji ne yaptı</h4><pre>${esc((e.log||[]).join("\n"))}</pre>
      </div></details>`;}).reverse().join("");
  totals();}
const down=()=>{const s=$("stream");s.scrollTop=s.scrollHeight;};

function send(){
  const inp=$("inp"), msg=inp.value.trim();
  if(!msg||busy)return;
  busy=true;$("send").disabled=true;inp.value="";inp.style.height="auto";
  const hero=$("hero"); if(hero)hero.remove();

  $("thread").insertAdjacentHTML("beforeend",
    `<div class="turn me"><div class="txt">${esc(msg)}</div></div>`);
  const id="t"+Date.now();
  $("thread").insertAdjacentHTML("beforeend",`<div class="turn" id="${id}">
    <div class="who"><span class="av">B</span>Brain Agent</div>
    <div class="status" id="${id}-s"><span class="pulse"></span><span id="${id}-st">düşünüyor…</span></div>
    <details class="tool" id="${id}-t" style="display:none"><summary>
      <svg class="chev" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="3" stroke-linecap="round"><path d="M9 6l6 6-6 6"/></svg>
      <span class="lab" id="${id}-lab">çalışıyor</span>
      <span class="meta" id="${id}-meta"></span></summary>
      <div class="out" id="${id}-o"></div></details>
    <div id="${id}-a"></div></div>`);
  down();

  const S=$(id+"-st"), T=$(id+"-t"), L=$(id+"-lab"), M=$(id+"-meta"),
        O=$(id+"-o"), A=$(id+"-a"), ST=$(id+"-s");
  let lines=[], nodes=0;
  const show=()=>{T.style.display="";O.innerHTML=paint(lines.join("\n"));O.scrollTop=O.scrollHeight;};

  const es=new EventSource(`/chat?sid=${SID}&msg=${encodeURIComponent(msg)}`);
  es.onmessage=e=>{
    const ev=JSON.parse(e.data);
    if(ev.type==="done"){es.close();busy=false;$("send").disabled=false;ST.style.display="none";return;}
    if(ev.type==="chat"){ A.innerHTML=`<div class="md">${esc(ev.text)}</div>`; }
    else if(ev.type==="phase"){ S.textContent=ev.text; L.textContent=ev.text; }
    else if(ev.type==="start"){ S.textContent="karar veriyor…"; }
    else if(ev.type==="board"){ board(ev.tasks); }
    else if(ev.type==="saved"){
      lines.push(`  ⤷ akış kaydedildi: ${ev.id} (${ev.n} düğüm)`); show();
      // Graf kurulur kurulmaz motor incelemesine geçiş — kullanıcı menüde aramasın.
      A.insertAdjacentHTML("beforeend",
        `<div class="foot"><button class="iconbtn migit" data-p="${ev.id}"
           style="border-color:var(--accent);color:var(--accent-ink)"
           >🔬 Bu grafı motorların gözünden incele (${ev.n} düğüm)</button></div>`);
      A.querySelectorAll(".migit").forEach(b=>b.onclick=()=>openMi(b.dataset.p));
      down();
    }
    else if(ev.type==="node_added"){ nodes++; M.textContent=nodes+" düğüm";
      lines.push(ev.text); show(); }
    else if(ev.type==="reduction"){ izleRed(ev);
      lines.push(`  ⤷ fn:${ev.fn}  ${ev.raw_tokens}→${ev.out_tokens} token `
        +(ev.pct>0?`(−%${ev.pct}, LLM yok)`:`(indirgeme yok — taşıyıcı düğüm, LLM yok)`)); show(); }
    else if(ev.type==="compaction"){ izleTrc(ev);
      lines.push(`  ⤷ compaction ${ev.title}: `+ev.events.map(c=>`${c.before}→${c.after} (−%${c.pct})`).join(", ")); show(); }
    // ── KARŞILAŞTIRMA KİPİ: her motorun ÇIKTISI ayrı bir blok olarak basılır ──
    else if(ev.type==="motor_basladi"){ S.textContent=ev.backend+" koşuyor…";
      lines.push(ev.text||("══ "+ev.backend)); show(); }
    else if(ev.type==="motor_cikti"){
      A.insertAdjacentHTML("beforeend",
        `<details class="exp" open><summary><span class="lab">${ev.ok?"✓":"✗"} ${esc(ev.backend)}</span>
           <span class="meta">${esc(ev.ozet||"")}</span></summary>
         <div class="body2"><pre>${esc(ev.cikti||"")}</pre></div></details>`);
      down(); }
    else if(ev.type==="karsilastirma"){
      const t=`<table class="kt"><thead><tr><th>motor</th><th>süre</th><th>tamamlanan</th>
        <th>başarısız</th><th>iptal</th>
        <th>retry<div style="font-weight:400;font-size:9px;color:var(--faint)"
          >hata / düğüm ×koşum</div></th></tr></thead><tbody>`
        +ev.satirlar.map(x=>`<tr class="${x.ok?"iyi":"kotu"}"><td><b>${esc(x.backend)}</b></td>
          <td class="n">${x.sn}s</td><td class="n">${x.done}/${x.dugum}</td>
          <td class="n">${x.failed||0}</td><td class="n">${x.cancelled||0}</td>
          <td class="n">${_retryOzet(x)}</td></tr>`).join("")+`</tbody></table>`;
      // düğüm bazlı döküm — hangi düğüm hangi motorda kaç kez koştu
      A.insertAdjacentHTML("beforeend", t + wfDugumTabloHam(ev.satirlar)); down(); }
    else if(ev.type==="summary"){
      if(ev.answer) A.innerHTML=`<div class="md">${esc(ev.answer)}</div>`;
      let tags=[];
      if(!ev.text.startsWith("sohbet")) tags.push(`<span class="tag good">${esc(ev.text)}</span>`);
      if(ev.crashes) tags.push(`<span class="tag bad">çökme ${ev.crashes} → kurtarma ${ev.recovered}</span>`);
      if(ev.retries) tags.push(`<span class="tag warn">retry ${ev.retries}</span>`);
      if(tags.length) A.innerHTML+=`<div class="foot">${tags.join("")}</div>`;
      L.textContent="ayrıntılar";
    }
    else if(ev.type==="error"){ lines.push("✗ "+ev.text); show(); }
    else { lines.push(ev.text); show(); }
    down();
  };
  es.onerror=()=>{es.close();busy=false;$("send").disabled=false;ST.style.display="none";};
}

document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.remove("on"));
  b.classList.add("on");
  for(const t of ["board","trace","fns"]) $("p-"+t).style.display=(t===b.dataset.t?"":"none");});
$("panel").onclick=()=>$("aside").classList.toggle("hide");

async function openFlows(){
  $("ov").classList.add("on"); $("ovback").style.display="none";
  $("ovtitle").textContent="Kurulan akışlar";
  const d=await (await fetch("/pipelines")).json();
  if(!d.items.length){ $("ovbody").innerHTML=
    '<p class="note">Henüz kayıtlı akış yok. Çok adımlı bir otomasyon iste — kurulan graf buraya düşer.</p>';
    return; }
  $("ovbody").innerHTML='<div class="plist">'+d.items.map(p=>{
    const t=new Date(p.at*1000).toLocaleString("tr");
    const st=p.stats||{};
    return `<div class="pcard" data-id="${p.id}">
      <div class="g">${esc(p.goal)}</div>
      <div class="s">${esc(p.pack)} · ${esc(p.backend)} · ${p.n} düğüm · ${esc(t)}</div>
      <div class="b">
        ${st.fn?`<span class="tag good">${st.fn} fn</span>`:""}
        ${st.ajan?`<span class="tag warn">${st.ajan} ajan</span>`:""}
        ${st.cokme?`<span class="tag bad">çökme ${st.cokme}</span>`:""}
        ${st.sn?`<span class="tag">${st.sn} sn</span>`:""}
      </div></div>`;}).join("")+'</div>';
  document.querySelectorAll(".pcard").forEach(c=>c.onclick=()=>openFlow(c.dataset.id));
}

// Akış detayı = MOTOR KARŞILAŞTIRMA EKRANI.
// Ayrı bir panel yerine burada: graf çizilir, düğüme tıklanır, ne olacağı seçilir,
// istenen motorda (ya da hepsinde) koşturulur. Aynı akış dört motorda da AYNEN koşar.
async function openFlow(id){
  $("ovback").style.display="";
  $("ovback").onclick=openFlows;
  await wfYukle(id);
}

function runPipeline(id, goal){
  if(busy) return;
  busy=true; $("send").disabled=true;
  const hero=$("hero"); if(hero)hero.remove();
  $("thread").insertAdjacentHTML("beforeend",
    `<div class="turn me"><div class="txt">▶ kayıtlı akışı çalıştır: ${esc(goal)}</div></div>`);
  const tid="t"+Date.now();
  $("thread").insertAdjacentHTML("beforeend",`<div class="turn" id="${tid}">
    <div class="who"><span class="av">B</span>Brain Agent</div>
    <div class="status" id="${tid}-s"><span class="pulse"></span><span id="${tid}-st">akış yükleniyor…</span></div>
    <details class="tool" id="${tid}-t" style="display:none"><summary>
      <svg class="chev" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="3" stroke-linecap="round"><path d="M9 6l6 6-6 6"/></svg>
      <span class="lab" id="${tid}-lab">yeniden koşu</span>
      <span class="meta" id="${tid}-meta"></span></summary>
      <div class="out" id="${tid}-o"></div></details>
    <div id="${tid}-a"></div></div>`);
  down();
  const S=$(tid+"-st"),T=$(tid+"-t"),M=$(tid+"-meta"),O=$(tid+"-o"),A=$(tid+"-a"),ST=$(tid+"-s");
  let lines=[],n=0;
  const show=()=>{T.style.display="";O.innerHTML=paint(lines.join("\n"));O.scrollTop=O.scrollHeight;};
  const es=new EventSource(`/runpipeline?sid=${SID}&id=${id}`);
  es.onmessage=e=>{
    const ev=JSON.parse(e.data);
    if(ev.type==="done"){es.close();busy=false;$("send").disabled=false;ST.style.display="none";return;}
    if(ev.type==="board"){ board(ev.tasks); }
    else if(ev.type==="phase"){ S.textContent=ev.text; }
    else if(ev.type==="node_added"){ n++; M.textContent=n+" düğüm"; lines.push(ev.text); show(); }
    else if(ev.type==="reduction"){ izleRed(ev);
      lines.push(`  ⤷ fn:${ev.fn}  ${ev.raw_tokens}→${ev.out_tokens} token `
        +(ev.pct>0?`(−%${ev.pct})`:`(indirgeme yok — taşıyıcı düğüm)`)); show(); }
    else if(ev.type==="summary"){
      if(ev.answer) A.innerHTML=`<div class="md">${esc(ev.answer)}</div>`;
      let tags=[`<span class="tag good">${esc(ev.text)}</span>`];
      if(ev.crashes) tags.push(`<span class="tag bad">çökme ${ev.crashes} → kurtarma ${ev.recovered}</span>`);
      A.innerHTML+=`<div class="foot">${tags.join("")}</div>`;
    }
    else { lines.push(ev.text); show(); }
    down();
  };
  es.onerror=()=>{es.close();busy=false;$("send").disabled=false;ST.style.display="none";};
}
// ───────── MOTOR KARŞILAŞTIRMA PANELİ ─────────
// Bir kez workflow üret → düğümlere tıklayıp ne olacağını seç → dört motorda koştur.
let WF={id:null, d:null, sim:{}, args:{}, sec:null};
const WF_ORNEK=[["ETL akışı kur: kayıtları çek, doğrula, normalize et, topla, dışa aktar","data"],
  ["auth modülünü güvenlik açısından denetle ve raporla","audit"],
  ["yeni sürümü kademeli yayınla ve sağlığını doğrula","deploy"]];

async function openWf(pid){
  $("ov").classList.add("on"); $("ovback").style.display="";
  $("ovtitle").textContent="Motor karşılaştırma";
  const pl=await (await fetch("/pipelines")).json();
  const paketler=Object.keys((META&&META.packs)||{"audit":1,"data":1,"deploy":1});

  // BİRİNCİL yol: sohbette üretilmiş, Akışlar ekranındaki akışlardan seç.
  // Panelin kendi üreticisi ikincil — yeni bir akış gerektiğinde diye duruyor.
  const kartlar=pl.items.length? pl.items.map(p=>{
    const t=new Date(p.at*1000).toLocaleString("tr",{day:"2-digit",month:"2-digit",
      hour:"2-digit",minute:"2-digit"});
    const st=p.stats||{};
    return `<div class="pcard" data-wfp="${p.id}">
      <div class="g">${esc(p.goal)}</div>
      <div class="s">${esc(p.pack)} · ${p.n} düğüm · ${esc(t)}</div>
      <div class="b">${st.fn?`<span class="tag good">${st.fn} fn</span>`:""}
        ${st.ajan?`<span class="tag warn">${st.ajan} ajan</span>`:""}
        <span class="tag">${esc(p.id)}</span></div></div>`;}).join("")
    : '<p class="note">Henüz akış yok. Sohbette çok adımlı bir iş iste — kurulan graf buraya düşer.</p>';

  $("ovbody").innerHTML=`
    <div class="pcard" style="cursor:default">
      <div class="g">1 · Hangi akış?</div>
      <div class="s">Sohbette ürettiğin akışlar (Akışlar ekranındakiler). Seçtiğin akış
        dört motorda da AYNEN koşacak — motorları kıyaslamak için tek graf gerekiyor.</div>
      <div class="plist" style="margin-top:10px">${kartlar}</div>
      <details class="exp" style="margin-top:10px">
        <summary><span class="lab">＋ ya da buradan yeni bir akış üret</span></summary>
        <div class="body2">
          <div style="display:grid;grid-template-columns:1fr 130px;gap:8px">
            <input id="wfgoal" placeholder="ne yapılsın?" value="${esc(WF_ORNEK[0][0])}"
              style="padding:7px 9px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font-size:13px">
            <select id="wfpack" style="padding:7px 9px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font-size:13px">
              ${paketler.map(p=>`<option value="${p}">${p}</option>`).join("")}</select>
          </div>
          <div class="b" style="margin-top:8px">
            ${WF_ORNEK.map((o,i)=>`<span class="tag" style="cursor:pointer" data-orn="${i}">${esc(o[0].slice(0,34))}…</span>`).join("")}
          </div>
          <div style="margin-top:9px;display:flex;align-items:center;gap:10px">
            <button class="iconbtn" id="wfplan" style="border-color:var(--accent);color:var(--accent-ink)">＋ Üret</button>
            <span class="note" id="wfmsg" style="margin:0"></span>
          </div>
        </div>
      </details>
    </div>
    <div id="wfgraf"></div>`;
  // akış kartına tıkla → detay = motor karşılaştırma ekranı (openFlow)
  document.querySelectorAll("[data-wfp]").forEach(c=>c.onclick=()=>openFlow(c.dataset.wfp));
  document.querySelectorAll("[data-orn]").forEach(b=>b.onclick=()=>{
    const o=WF_ORNEK[+b.dataset.orn]; $("wfgoal").value=o[0]; $("wfpack").value=o[1];});
  $("wfplan").onclick=wfUret;
  if(pid) openFlow(pid);                 // doğrudan bir akışla açıldıysa
}

async function wfUret(){
  const goal=$("wfgoal").value.trim(); if(!goal) return;
  $("wfmsg").textContent="planlanıyor…"; $("wfmsg").style.color="";
  $("wfgraf").innerHTML='<div class="wflog" id="wfplog">planlayıcı çalışıyor…</div>';
  const q=new URLSearchParams({sid:SID,goal,pack:$("wfpack").value});
  const es=new EventSource("/wf/plan?"+q); let pid=null, lines=[];
  es.onmessage=e=>{const ev=JSON.parse(e.data);
    if(ev.type==="saved") pid=ev.id;
    if(ev.text) lines.push(ev.text);
    const el=$("wfplog"); if(el){el.textContent=lines.slice(-14).join("\n"); el.scrollTop=el.scrollHeight;}
    if(ev.type==="error"){$("wfmsg").textContent="✗ "+ev.text; $("wfmsg").style.color="var(--bad)";}
    if(ev.type==="done"){es.close(); $("wfmsg").textContent=pid?("✓ "+pid):"✗ üretilemedi";
      if(pid) wfYukle(pid);}};
  es.onerror=()=>{es.close();};
}

async function wfYukle(pid){
  WF.id=pid; WF.sim={}; WF.args={}; WF.sec=null;
  WF.d=await (await fetch("/wf/get?id="+pid)).json();
  wfCiz();
}

function wfCiz(){
  const d=WF.d, byId={}; (d.nodes||[]).forEach(n=>byId[n.id]=n);
  const dag=(d.layers||[]).map(lay=>`<div class="lay">`+lay.map(id=>{
    const n=byId[id]; if(!n) return "";
    const s=WF.sim[id], mod=s&&s.mod&&s.mod!=="normal"?s.mod:null;
    const cls=["nd", n.kind==="function"?"fn":"agent", "tik",
               WF.sec===id?"sec":"", mod?("simli sim-"+mod):""].filter(Boolean).join(" ");
    const uyari=(mod==="gecici"||mod==="yavas")?" uyari":"";
    return `<div class="${cls}" data-nid="${id}">
      <div class="n1">${esc(n.fn||"LLM AJAN")}</div>
      <div class="n2">${esc(n.title||"")}</div>
      <div class="n3">${esc(JSON.stringify(WF.args[id]||n.args||{}))}</div>
      ${mod?`<span class="simrozet${uyari}">${esc(mod)}${s.sn?" "+s.sn+"s":""}</span>`:""}
    </div>`;}).join("")+`</div>`).join(
      `<div class="arrow"><svg width="20" height="14" viewBox="0 0 20 14" fill="none"
        stroke="currentColor" stroke-width="1.6"><path d="M1 7h16M13 3l4 4-4 4"/></svg></div>`);

  const motorlar=Object.keys((META&&META.backends)||{own:1,temporal:1,celery:1,airflow:1});
  const simSay=Object.keys(WF.sim).length, argSay=Object.keys(WF.args).length;
  $("ovtitle").textContent=d.goal||"akış";
  $("ovbody").innerHTML=`
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:4px">
      <select id="wfbackend" style="padding:7px 9px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font-size:13px">
        ${motorlar.map(m=>`<option value="${m}">${m}</option>`).join("")}</select>
      <button class="iconbtn" id="wfrun" style="border-color:var(--accent);color:var(--accent-ink)">▶ Koştur</button>
      <button class="iconbtn" id="wfrunall">⚡ Hepsinde koştur</button>
      ${(simSay||argSay)?`<button class="iconbtn" id="wfsifirla">Sıfırla (${simSay+argSay})</button>`:""}
      <button class="iconbtn" id="wfmi" title="Bu akışı dört motorun gözünden incele">🔬 Motor gözüyle</button>
      ${d.airflow_hazir?"":'<span class="tag bad">airflow kurulu değil</span>'}
    </div>
    <div class="note" style="margin:0 0 10px">
      paket <b>${esc(d.pack||"")}</b> · ${(d.nodes||[]).length} düğüm ·
      aynı graf dört motorda da AYNEN koşar —
      <b>düğüme tıkla</b>, ona ne olacağını seç, sonra koştur.
    </div>
    <div class="dag">${dag}</div>
    <div id="wfayar"></div>
    <div id="wfsonuc"></div>`;
  document.querySelectorAll("[data-nid]").forEach(el=>el.onclick=()=>wfSec(el.dataset.nid));
  $("wfrun").onclick=()=>wfKos(false);
  $("wfrunall").onclick=()=>wfKos(true);
  if($("wfsifirla")) $("wfsifirla").onclick=()=>{WF.sim={}; WF.args={}; WF.sec=null; wfCiz();};
  if($("wfmi")) $("wfmi").onclick=()=>openMi(WF.id);   // BU akışı motor gözünden incele
  if(WF.sec) wfAyarCiz(WF.sec);
}

function wfSec(nid){ WF.sec=(WF.sec===nid?null:nid); wfCiz(); }

function wfAyarCiz(nid){
  const n=(WF.d.nodes||[]).find(x=>x.id===nid); if(!n) return;
  const kat=(WF.d.katalog||{})[n.fn]||{args:{},aciklama:""};
  const cur=WF.sim[nid]||{mod:"normal"};
  const modlar=WF.d.sim_modlari||{};
  const argIn=Object.entries(kat.args||{}).map(([k,v])=>{
    const val=(WF.args[nid]||{})[k]!==undefined?(WF.args[nid])[k]:((n.args||{})[k]!==undefined?(n.args)[k]:"");
    return `<label style="display:block;font-size:11px;color:var(--faint);margin-top:6px">${esc(k)} — ${esc(v)}
      <input data-arg="${esc(k)}" value="${esc(String(val))}"
        style="width:100%;padding:5px 8px;margin-top:2px;border-radius:7px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font-family:var(--mono);font-size:11.5px"></label>`;}).join("");
  $("wfayar").innerHTML=`
    <div style="border:1px solid var(--accent);border-radius:10px;padding:11px 13px;margin-top:6px;background:var(--bg)">
      <div style="font-family:var(--mono);font-size:12px;font-weight:650;color:var(--accent-ink)">${esc(n.fn||"AJAN")} <span style="color:var(--faint);font-weight:400">· ${esc(nid)}</span></div>
      <div class="note" style="margin:3px 0 0">${esc(kat.aciklama||"")}</div>
      ${argIn?`<div style="margin-top:6px"><b style="font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--faint)">argümanlar</b>${argIn}</div>`:""}
      <div style="margin-top:9px"><b style="font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--faint)">bu düğüme ne olsun?</b>
        <select id="wfmod" style="width:100%;margin-top:4px;padding:6px 9px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font-size:12.5px">
          ${Object.entries(modlar).map(([m,a])=>`<option value="${m}"${m===cur.mod?" selected":""}>${m} — ${esc(a)}</option>`).join("")}
        </select></div>
      <div id="wfsn" style="margin-top:6px;${cur.mod==="yavas"?"":"display:none"}">
        <label style="font-size:11px;color:var(--faint)">kaç saniye beklesin
          <input id="wfsnv" type="number" min="1" max="60" value="${cur.sn||3}"
            style="width:80px;margin-left:6px;padding:4px 7px;border-radius:7px;border:1px solid var(--line);background:var(--bg);color:var(--ink)"></label></div>
      <div style="margin-top:10px;display:flex;gap:8px">
        <button class="iconbtn" id="wfuygula" style="border-color:var(--accent);color:var(--accent-ink)">Uygula</button>
        <button class="iconbtn" id="wfkapat">Kapat</button></div>
    </div>`;
  $("wfmod").onchange=()=>{$("wfsn").style.display=$("wfmod").value==="yavas"?"":"none";};
  $("wfkapat").onclick=()=>{WF.sec=null; wfCiz();};
  $("wfuygula").onclick=()=>{
    const mod=$("wfmod").value;
    if(mod==="normal") delete WF.sim[nid];
    else WF.sim[nid]=(mod==="yavas")?{mod,sn:+$("wfsnv").value||3}:{mod};
    const a={}; document.querySelectorAll("[data-arg]").forEach(i=>{
      if(i.value!=="") a[i.dataset.arg]=i.value;});
    if(Object.keys(a).length) WF.args[nid]=a; else delete WF.args[nid];
    WF.sec=null; wfCiz();};
}

function wfKos(hepsi){
  const q=new URLSearchParams({sid:SID,id:WF.id,sim:JSON.stringify(WF.sim),
    args:JSON.stringify(WF.args)});
  if(!hepsi) q.set("backend",$("wfbackend").value);
  $("wfsonuc").innerHTML=`<div class="wflog" id="wfrlog">başlıyor…</div>`;
  const lines=[], satirlar=[];
  const es=new EventSource("/wf/"+(hepsi?"runall":"run")+"?"+q);
  es.onmessage=e=>{const ev=JSON.parse(e.data);
    if(ev.text) lines.push(ev.text);
    if(ev.type==="motor_bitti"){satirlar.push(ev);
      lines.push(`══ ${ev.backend}: ${ev.ok?"✓":"✗"} ${ev.sn}s · done=${ev.done} failed=${ev.failed} iptal=${ev.cancelled}`);}
    const el=$("wfrlog"); if(el){el.textContent=lines.slice(-200).join("\n"); el.scrollTop=el.scrollHeight;}
    if(ev.type==="karsilastirma") wfTablo(ev.satirlar);
    if(ev.type==="done"){es.close(); if(!hepsi&&satirlar.length) wfTablo(satirlar);}};
  es.onerror=()=>{es.close();};
}

// RETRY sütunu: toplam sayı tek başına "hangi düğüm" sorusunu cevaplamıyordu.
// Artık hücrede hem sayı hem HANGİ düğümün kaç kez koştuğu yazıyor.
function _retryOzet(x){
  const dl=(x.dugumler||[]).filter(d=>(d.deneme||0)>0);
  const n=x.retries||0;
  if(!n && !dl.length) return '<span style="color:var(--faint)">yok</span>';
  const ad=d=>{const nd=((WF.d&&WF.d.nodes)||[]).find(v=>v.id===d.nid);
               return (nd&&(nd.fn||nd.title))||d.fn||d.nid;};
  return `<b style="color:var(--warn)">${n}</b>` + (dl.length?
    `<div style="font-size:10px;color:var(--muted);font-family:var(--mono)">`
    + dl.map(d=>`${esc(String(ad(d)).slice(0,16))} ×${d.kosum??"?"}`).join("<br>")
    + `</div>` : "");
}

function wfTablo(satirlar){
  const t=`<table class="kt"><thead><tr><th>motor</th><th>süre</th><th>tamamlanan</th>
    <th>başarısız</th><th>iptal</th>
    <th title="kaç kez hata alındı — altında hangi düğümün kaç kez koştuğu">retry
      <div style="font-weight:400;font-size:9px;color:var(--faint)">hata / düğüm ×koşum</div></th>
    <th>çökme/kurtarma</th><th>not</th></tr></thead><tbody>`
    +satirlar.map(x=>`<tr class="${x.ok?"iyi":"kotu"}">
      <td><b>${esc(x.backend)}</b></td><td class="n">${x.sn}s</td>
      <td class="n">${x.done}/${x.dugum}</td><td class="n">${x.failed||0}</td>
      <td class="n">${x.cancelled||0}</td><td class="n">${_retryOzet(x)}</td>
      <td class="n">${x.crashes||0}/${x.recovered||0}</td>
      <td style="font-size:11px;color:var(--faint)">${esc((x.not||"").slice(0,60))}</td></tr>`).join("")
    +`</tbody></table>`;
  $("wfsonuc").insertAdjacentHTML("afterbegin", t + wfDugumTablo(satirlar));
}

// Sohbet turunda kayıtlı akış yüklü DEĞİL (WF.d yok) — düğüm listesini sonuçların
// kendisinden türet. Airflow'un düğümlerinde fn/başlık yok, o yüzden etiketleri
// board'lu motorlardan alıyoruz; hizalama `nid` (kayıtlı akış id'si) ile.
function wfDugumTabloHam(satirlar){
  const etiket={};
  satirlar.forEach(x=>(x.dugumler||[]).forEach(d=>{
    if(d.fn && !etiket[d.nid]) etiket[d.nid]={id:d.nid, fn:d.fn, title:d.baslik||""};}));
  const nodes=Object.values(etiket);
  if(!nodes.length) return "";
  const eski=WF.d; WF.d={nodes};                 // wfDugumTablo aynı veriyi bekliyor
  const html=wfDugumTablo(satirlar);
  WF.d=eski;
  return html;
}

// DÜĞÜM BAZLI sonuç: satır = düğüm, sütun = motor, hücre = durum + kaç kez koştu.
// Toplam "deneme" sayısı hangi düğümün kaç kez koştuğunu göstermiyor; Celery'nin
// retry'ı task'ı BAŞTAN koşturduğu için asıl fark burada görünüyor.
function wfDugumTablo(satirlar){
  const gecerli=satirlar.filter(x=>(x.dugumler||[]).length);
  if(!gecerli.length) return "";
  const nodes=(WF.d&&WF.d.nodes)||[];
  const harita={};                        // {motor: {nid: {durum,deneme,sn}}}
  gecerli.forEach(x=>{harita[x.backend]={};
    (x.dugumler||[]).forEach(d=>{harita[x.backend][d.nid]=d;});});
  const motorlar=gecerli.map(x=>x.backend);
  const rozet=d=>{
    if(!d) return '<span class="tag">—</span>';
    const k=d.durum==="done"||d.durum==="success"?"good"
           :(d.durum==="failed"?"bad":(d.durum==="cancelled"||d.durum==="upstream_failed"
             ||d.durum==="skipped"?"warn":""));
    const k2=+d.kosum||0, h=+d.deneme||0;
    const ipucu=`${d.durum} · ${k2} kez koştu`+(h?` · ${h} hata`:"");
    return `<span class="tag ${k}" title="${esc(ipucu)}">${esc(d.durum)}</span>`
         + (k2>1?` <b class="num" title="${esc(ipucu)}">×${k2}</b>`:"");
  };
  // "Celery'nin ÖNCEKİ düğümleri neden ×1?" — ekranda en çok sorulan şey.
  // Cevap mimaride: board her düğümü AYRI bir Celery task'ı olarak gönderiyor
  // ve defteri kendisi tutuyor. Saf Celery'de bu satır başka çıkardı.
  const retryVar = gecerli.some(x=>(x.dugumler||[]).some(d=>(d.kosum||0)>1));
  const aciklama = retryVar ? `
    <div class="uyari-kutu">
      <b>Neden ÖNCEKİ düğümler ×1?</b> Çünkü board her düğümü <b>ayrı bir task</b>
      olarak gönderiyor ve "hangi düğüm bitti" defterini kendisi tutuyor. Yalnız
      patlayan düğüm yeniden kuyruğa giriyor.
      <div style="margin-top:6px">SAF Celery'de iki yol var, ikisinin de bedeli var:</div>
      <ul style="margin:4px 0 0;padding-left:18px">
        <li><b>Adımlar tek task'ın içindeyse</b> → <code>self.retry()</code> fonksiyonu
          1. satırdan başlatır, <b>önceki adımlar da tekrar koşar</b>
          (çapa ölçümü: <code>fetch ×2</code>). Yan etkiliyse iş iki kez yapılır.</li>
        <li><b>Adımları <code>chain</code>'e bölersen</b> → her halka ayrı retry alır,
          ama <b>canvas durable değildir</b>: worker zincir ortasında çökerse Celery
          "3. adımdaydım" bilgisini tutmaz, zincirin kalanı sessizce ölür.</li>
      </ul>
      <div style="margin-top:6px">Board'un eklediği şey tam olarak bu: <b>hem</b> düğüm
        bazlı retry <b>hem</b> kalıcı defter. Airflow da aynı yeri doldurur —
        Temporal ise event history ile.</div>
    </div>` : "";

  return aciklama + `<div class="note" style="margin:14px 0 4px">
      <b>düğüm bazlı sonuç</b> — <code>×N</code> = o düğüm <b>N kez koştu</b>.
      Retry düğümü BAŞTAN koşturur; yan etkili bir düğümde bu, işin N kez
      yapılması demektir. Üstüne gelince hata sayısı da görünür.
    </div>
    <table class="kt"><thead><tr><th>düğüm</th>
      ${motorlar.map(m=>`<th>${esc(m)}</th>`).join("")}</tr></thead><tbody>`
    + nodes.map(n=>`<tr>
        <td><span style="font-family:var(--mono);font-size:11px;color:var(--accent-ink)">${esc(n.fn||"AJAN")}</span>
          <div style="font-size:11px;color:var(--faint)">${esc((n.title||"").slice(0,30))}</div></td>
        ${motorlar.map(m=>`<td>${rozet(harita[m][n.id])}</td>`).join("")}</tr>`).join("")
    + `</tbody></table>`;
}

// ───────── ZAMANLAMA (cron) ─────────
const CRON_ORNEK=[["0 8 * * *","her gün 08:00"],["0 8 * * 1-5","hafta içi 08:00"],
  ["*/15 * * * *","15 dakikada bir"],["0 */6 * * *","6 saatte bir"],
  ["30 9 1 * *","ayın 1'i 09:30"],["0 18 * * 5","Cuma 18:00"]];

async function openScheds(){
  $("ov").classList.add("on"); $("ovback").style.display="none";
  $("ovtitle").textContent="Zamanlanmış koşular";
  const [d,pl]=await Promise.all([
    (await fetch("/schedules")).json(), (await fetch("/pipelines")).json()]);

  const opts=pl.items.map(p=>`<option value="${p.id}">${esc(p.goal.slice(0,52))} · ${p.n} düğüm</option>`).join("");
  const form=pl.items.length? `
    <div class="pcard" style="cursor:default">
      <div class="g">Yeni zamanlama</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px">
        <input id="sname" placeholder="ad (ör. sabah denetimi)"
          style="padding:7px 9px;border-radius:8px;border:1px solid var(--line);
                 background:var(--bg);color:var(--ink);font-size:13px">
        <input id="scron" placeholder="0 8 * * *" value="0 8 * * *"
          style="padding:7px 9px;border-radius:8px;border:1px solid var(--line);
                 background:var(--bg);color:var(--ink);font-family:var(--mono);font-size:13px">
      </div>
      <select id="spipe" style="width:100%;margin-top:8px;padding:7px 9px;border-radius:8px;
        border:1px solid var(--line);background:var(--bg);color:var(--ink);font-size:13px">${opts}</select>
      <div class="b" style="margin-top:8px">
        ${CRON_ORNEK.map(([c,t])=>`<span class="tag" style="cursor:pointer"
          onclick="document.getElementById('scron').value='${c}'">${t}</span>`).join("")}
      </div>
      <div style="margin-top:9px;display:flex;align-items:center;gap:10px">
        <button class="iconbtn" id="saddbtn"
          style="border-color:var(--accent);color:var(--accent-ink)">+ Zamanla</button>
        <span class="note" id="saddmsg" style="margin:0"></span>
      </div>
    </div>`
    : '<p class="note">Önce bir akış kur — zamanlama kayıtlı akışları çalıştırır.</p>';

  const liste = d.items.length ? d.items.map(s=>{
    const nx=new Date(s.next_run_at*1000).toLocaleString("tr",
      {day:"2-digit",month:"2-digit",hour:"2-digit",minute:"2-digit"});
    const gec=(s.son_kosular||[]).map(r=>{
      const ok=r.ok===1?"good":r.ok===0?"bad":"";
      const t=new Date(r.started_at*1000).toLocaleString("tr",
        {day:"2-digit",month:"2-digit",hour:"2-digit",minute:"2-digit"});
      return `<span class="tag ${ok}" title="${esc(r.detay||"")}">${t}</span>`;}).join("");
    return `<div class="pcard" style="cursor:default${s.enabled?"":";opacity:.55"}">
      <div class="g">${esc(s.name)} ${s.enabled?"":'<span class="tag">KAPALI</span>'}</div>
      <div class="s"><code>${esc(s.cron)}</code> · ${esc(s.ne_zaman)}
        · sonraki: <b>${nx}</b> · akış ${esc(s.pipeline_id)}
        ${s.last_status?` · son: <span class="tag ${s.last_status==="ok"?"good":"bad"}">${esc(s.last_status)}</span>`:""}</div>
      ${gec?`<div class="b">${gec}</div>`:""}
      <div class="b" style="margin-top:6px">
        <span class="tag" style="cursor:pointer" data-run="${s.id}">▶ şimdi çalıştır</span>
        <span class="tag" style="cursor:pointer" data-tog="${s.id}">${s.enabled?"⏸ durdur":"▶ etkinleştir"}</span>
        <span class="tag bad" style="cursor:pointer" data-del="${s.id}">✕ sil</span>
      </div></div>`;}).join("")
    : '<p class="note">Henüz zamanlama yok.</p>';

  $("ovbody").innerHTML=form+'<div class="plist" style="margin-top:12px">'+liste+'</div>';

  const add=$("saddbtn");
  if(add) add.onclick=async()=>{
    const q=new URLSearchParams({name:$("sname").value,cron:$("scron").value,
      pipeline:$("spipe").value});
    const r=await fetch("/schedule/add?"+q);
    const j=await r.json();
    if(!r.ok){ $("saddmsg").textContent="✗ "+(j.error||"eklenemedi");
      $("saddmsg").style.color="var(--bad)"; return; }
    openScheds();
  };
  document.querySelectorAll("[data-run]").forEach(b=>b.onclick=async()=>{
    b.textContent="… koşuyor";
    const j=await (await fetch("/schedule/run?id="+b.dataset.run)).json();
    alert((j.ok?"✓ ":"✗ ")+(j.ozet||j.error||""));
    openScheds();});
  document.querySelectorAll("[data-tog]").forEach(b=>b.onclick=async()=>{
    await fetch("/schedule/toggle?id="+b.dataset.tog); openScheds();});
  document.querySelectorAll("[data-del]").forEach(b=>b.onclick=async()=>{
    if(!confirm("Zamanlama silinsin mi?")) return;
    await fetch("/schedule/delete?id="+b.dataset.del); openScheds();});
}

$("flows").onclick=openFlows;


// Deneyi GERÇEKTEN koştur — mevcut /wf/run yolunu kullanır, ayrı yürütme yolu YOK.
let MIvurgu={};
async function miDeneyKos(i, hepsi){
  const m=MI.motorlar[MIsec];
  const d=(MIdeney.motorlar[m.ad]||[])[i]; if(!d) return;
  const kutu=$("dson"+i);
  const btns=[document.querySelector(`.dkos[data-i="${i}"]`),
              document.querySelector(`.dhep[data-i="${i}"]`)];
  btns.forEach(b=>b&&(b.disabled=true));
  const aktif=btns[hepsi?1:0]; const eskiAd=aktif.textContent; aktif.textContent="koşuyor…";
  MIvurgu = d.hedef_id ? {[d.hedef_id]: (d.sim||{}).mod||"normal"} : {};
  miVurguYenile();
  kutu.innerHTML='<div class="dlog">başlıyor…</div>';

  const sim = d.hedef_id && d.sim && d.sim.mod ? {[d.hedef_id]: d.sim} : {};
  const q=new URLSearchParams({sid:SID, id:MIpid, backend:d.backend,
                               sim:JSON.stringify(sim), args:"{}"});
  let satir=[], sonuclar=[];
  await new Promise(res=>{
    const es=new EventSource((hepsi?"/wf/runall?":"/wf/run?")+q);
    es.onmessage=e=>{
      const ev=JSON.parse(e.data);
      if(ev.text) satir.push(ev.text);
      if(ev.type==="motor_bitti") sonuclar.push(ev);
      const lg=kutu.querySelector(".dlog");
      if(lg&&satir.length) lg.textContent=satir.slice(-5).join("\n");
      if(ev.type==="done"){ es.close(); res(); }
    };
    es.onerror=()=>{ es.close(); res(); };
  });
  btns.forEach(b=>b&&(b.disabled=false)); aktif.textContent=eskiAd;
  if(!sonuclar.length){ kutu.innerHTML='<div class="dlog">⚠ sonuç alınamadı</div>'; return; }

  const iyi=["done","success"], kotu=["failed"];
  const ad=nid=>{const n=(MIgraf&&MIgraf.nodes||[]).find(x=>x.id===nid);
                 return n?(n.fn||n.title||nid):nid;};
  const renk=st=>iyi.includes(st)?"var(--ok)":kotu.includes(st)?"var(--bad)":"var(--warn)";

  if(!hepsi){
    const o=sonuclar[0], dg=o.dugumler||[];
    kutu.innerHTML=`<div class="dsn"><b>${o.ok?"✓":"✗"} ${esc(d.backend)}</b> ·
      ${o.sn} sn · ${o.done}/${o.dugum} tamam · ${o.failed||0} fail ·
      ${o.cancelled||0} iptal · <b>retry ${o.retries||0}</b> ·
      çökme ${o.crashes||0}/kurtarma ${o.recovered||0}</div>
      ${dg.length?`<table class="mtab" style="margin-top:8px"><thead><tr><th>düğüm</th>
        <th>durum</th><th>koşum</th></tr></thead><tbody>${dg.map(v=>{
        const h=v.nid===d.hedef_id;
        return `<tr${h?' style="background:var(--accent-soft)"':''}>
          <td class="m">${esc(ad(v.nid))}${h?" ← hedef":""}</td>
          <td class="m" style="color:${renk(v.durum)}"><b>${esc(v.durum||"")}</b></td>
          <td class="m">${v.kosum??""}</td></tr>`;}).join("")}</tbody></table>`:""}
      <details class="mdet"><summary>koşu logu (${satir.length} satır)</summary>
        <pre class="mkod">${esc(satir.join("\n"))}</pre></details>`;
    return;
  }

  // ── DÖRT MOTOR YAN YANA: aynı deney, aynı düğüm ──
  // Nişin kanıtı burada: ötekiler aynı kurulumda NE yapıyor?
  const nidler=[]; sonuclar.forEach(o=>(o.dugumler||[]).forEach(v=>{
    if(!nidler.includes(v.nid)) nidler.push(v.nid);}));
  const bul=(o,nid)=>(o.dugumler||[]).find(v=>v.nid===nid);
  const bu=sonuclar.find(o=>o.backend===d.backend);

  kutu.innerHTML=`
    <div class="dsn">⚖ aynı deney, dört motor —
      <b>${esc(ad(d.hedef_id)||"—")}</b> düğümüne
      <b>${esc((d.sim||{}).mod||"hatasız")}</b></div>
    <table class="mtab" style="margin-top:8px"><thead><tr><th>motor</th><th>süre</th>
      <th>retry</th><th>çökme</th>
      ${nidler.map(n=>`<th>${esc(ad(n))}${n===d.hedef_id?" ←":""}</th>`).join("")}
    </tr></thead><tbody>
    ${sonuclar.map(o=>{const bumu=o.backend===d.backend;
      return `<tr${bumu?' style="background:var(--accent-soft)"':''}>
      <td class="m"><b>${esc(o.backend)}</b>${bumu?" ★":""}</td>
      <td class="m">${o.sn}s</td>
      <td class="m">${o.retries||0}</td>
      <td class="m">${o.crashes||0}${o.recovered?"/"+o.recovered:""}</td>
      ${nidler.map(n=>{const v=bul(o,n);
        return `<td class="m" style="color:${v?renk(v.durum):"var(--faint)"}">
          ${v?esc(v.durum):"—"}${v&&v.kosum>1?` ×${v.kosum}`:""}</td>`;}).join("")}
    </tr>`;}).join("")}</tbody></table>
    <div class="dfark"><b>bak:</b> ${esc(d.bak)}<br>
      <b>beklenen:</b> ${esc(d.beklenen)}
      ${d.karsit?`<br><b>karşıt:</b> ${esc(d.karsit)}`:""}</div>
    <details class="mdet"><summary>koşu logu (${satir.length} satır)</summary>
      <pre class="mkod">${esc(satir.join("\n"))}</pre></details>`;
}

// Grafta hedef düğümü işaretle (tam yeniden çizim yapmadan)
function miVurguYenile(){
  document.querySelectorAll("[data-mnid]").forEach(el=>{
    const v=MIvurgu[el.dataset.mnid];
    el.className = "nd fn" + (v?" simli sim-"+v:"");
    const eski=el.querySelector(".simrozet"); if(eski) eski.remove();
    if(v) el.insertAdjacentHTML("beforeend",
      `<span class="simrozet${v==="gecici"?" uyari":""}">${v}</span>`);
  });
}


// Celery Canvas — board'suz koşu. Farkın kanıtı: koşarken durum SORULAMIYOR.
async function miCanvasKos(hatali){
  const btn=$(hatali?"cvhata":"cvkos"), kutu=$("cvson");
  const eskiAd=btn.textContent;
  [$("cvkos"),$("cvhata")].forEach(b=>b&&(b.disabled=true));
  btn.textContent="koşuyor…";
  // Hata deneyi: deney kataloğundan hedef düğümü al (ardılı olan ilk düğüm).
  // Bu olmadan 'canvas takılıyor' iddiası panelden DENENEMİYORDU — ölçüm yalnız
  // terminalden yapılabiliyordu.
  let sim={};
  if(hatali){
    const dl=(MIdeney&&MIdeney.motorlar&&MIdeney.motorlar.celery_canvas)||[];
    const d=dl.find(x=>x.hedef_id&&x.sim&&x.sim.mod);
    if(d) sim={[d.hedef_id]: d.sim};
  }
  kutu.innerHTML=`<div class="dlog">worker açılıyor (~6 sn)…${
    hatali?"\nkalıcı hata uygulanacak — canvas 60 sn bekleyip takılacak":""}</div>`;
  let r;
  try{ r=await (await fetch(`/motor/canvas?id=${MIpid}&bekle=70&sim=`
        +encodeURIComponent(JSON.stringify(sim)))).json(); }
  catch(e){ kutu.innerHTML='<div class="dlog">⚠ '+esc(e.message)+'</div>';
            [$("cvkos"),$("cvhata")].forEach(b=>b&&(b.disabled=false));
            btn.textContent=eskiAd; return; }
  [$("cvkos"),$("cvhata")].forEach(b=>b&&(b.disabled=false));
  btn.textContent=eskiAd;
  const ek=(r.eksikler||[]).map(x=>`<tr><td><b>${esc(x.ne)}</b></td>
      <td style="color:var(--bad)">${esc(x.canvas)}</td>
      <td style="color:var(--ok)">${esc(x.board)}</td></tr>`).join("");
  kutu.innerHTML=`
    <div class="dsn"><b>${r.ok?"✓":"✗"} canvas</b> · ${r.sn||"?"} sn
      ${r.katman?` · ${r.katman} katman [${(r.katman_boyu||[]).join(", ")}]`:""}
      ${r.hata?` · <span style="color:var(--bad)">${esc(r.hata)}</span>`:""}</div>
    ${r.sonuc?`<div class="db" style="margin-top:6px"><b>son halkanın dönüşü:</b>
      <code>${esc(JSON.stringify(r.sonuc).slice(0,180))}</code></div>`:""}
    <div class="dlog" style="margin-top:7px">${esc((r.log||[]).join("\n"))}</div>
    ${hatali?`<div class="uyari-kutu" style="margin-top:8px">
      <b>Aynı senaryo board'lu motorlarda:</b>
      <table class="mtab" style="margin-top:5px"><tbody>
        <tr><td class="m"><b>canvas</b></td><td style="color:var(--bad)">
          ${r.ok?"bitti":"TAKILDI"} · ${r.sn} sn · nerede takıldığı BİLİNMİYOR</td></tr>
        <tr><td class="m"><b>own</b></td><td style="color:var(--ok)">
          3/6 done · 1 failed · 2 CANCELLED</td></tr>
        <tr><td class="m"><b>airflow</b></td><td style="color:var(--ok)">
          validate FAILED · ardıllar UPSTREAM_FAILED (kendi tablosunda)</td></tr>
      </tbody></table>
      <div style="margin-top:5px">Üstteki deney düğmeleriyle bu satırları
        kendin koşturabilirsin.</div></div>`:""}
    <div class="dfark" style="margin-top:8px"><b>Canvas'ın VERMEDİĞİ</b>
      <table class="mtab" style="margin-top:5px"><thead><tr><th></th>
        <th>Canvas</th><th>board / Airflow / Temporal</th></tr></thead>
        <tbody>${ek}</tbody></table></div>`;
}

// ───────── MOTOR İNCELEME ─────────
// Dört motorun kanıtlı künyesi: künye · modüller · incelikler · güçlü/zayıf ·
// ayarlar · ilişkiler · bu grafın o motorun dilindeki hâli.
// Veri motor_kunye.py / motor_ayar.py / motor_dili.py'den geliyor; burası çizim.
let MI=null, MIsec=0, MIdil=null, MIpid=null, MIakislar=[], MIdeney=null, MIgraf=null;

// pid verilirse O AKIŞ incelenir; verilmezse en son akış seçilir.
async function openMi(pid){
  $("ov").classList.add("on"); $("ovback").style.display="none";
  $("ovtitle").textContent="Motor inceleme";
  $("ovbody").innerHTML='<p class="note">yükleniyor…</p>';
  try{ MI=await (await fetch("/motor/kunye")).json(); }
  catch(e){ $("ovbody").innerHTML='<p class="note">⚠ '+esc(e.message)+'</p>'; return; }
  if(MI.error){ $("ovbody").innerHTML='<p class="note">⚠ '+esc(MI.error)+'</p>'; return; }
  MIdil=null;
  try{
    const pl=await (await fetch("/pipelines")).json();
    MIakislar=pl.items||[];
    MIpid = pid || (MIakislar.length? MIakislar[0].id : null);
    if(MIpid){
      MIdil=await (await fetch("/motor/dil?id="+MIpid)).json();
      MIdeney=await (await fetch("/motor/deney?id="+MIpid)).json();
      MIgraf=await (await fetch("/wf/get?id="+MIpid)).json();
    }
  }catch(e){ MIdil=null; }
  miCiz();
}

function miCiz(){
  const kf=x=>({acik:"a",kapali:"k",esgudum:"e"}[x]||"k");
  const rozet=k=>k?`<span class="mkanit">${esc(k.tur)} · ${esc(k.ref).slice(0,30)}</span>`:"";
  const m=MI.motorlar[MIsec];
  const ay=(MI.ayar?MI.ayar.motorlar:[]).find(a=>a.motor===m.ad)||{ayarlar:[],sayim:{}};
  const dl=MIdil&&MIdil.diller?MIdil.diller.find(d=>d.ad===m.ad):null;
  const dny=(MIdeney&&MIdeney.motorlar)?(MIdeney.motorlar[m.ad]||[]):[];

  // ── akış seçici ──
  const sec = MIakislar.length ? `
    <div style="display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:10px">
      <select id="miakis" style="flex:1;min-width:200px;padding:7px 9px;border-radius:8px;
        border:1px solid var(--line);background:var(--bg);color:var(--ink);font-size:13px">
        ${MIakislar.map(a=>`<option value="${a.id}" ${a.id===MIpid?"selected":""}
          >${esc(a.goal.slice(0,60))} · ${a.n} düğüm</option>`).join("")}</select>
      <button class="iconbtn" id="migeri">← akış ekranı</button>
    </div>` : `<p class="note">Kayıtlı akış yok — sohbette bir workflow kur.</p>`;

  // ── graf: üzerinde deney yapılacak şey ──
  let graf="";
  if(MIgraf&&MIgraf.nodes){
    const byId={}; MIgraf.nodes.forEach(n=>byId[n.id]=n);
    graf=`<div class="dag">`+(MIgraf.layers||[]).map(lay=>`<div class="lay">`+
      lay.map(id=>{const n=byId[id]; if(!n)return"";
        const v=MIvurgu[id];
        return `<div class="nd fn${v?" simli sim-"+v:""}" data-mnid="${id}">
          <div class="n1">${esc(n.fn||"LLM")}</div>
          <div class="n2">${esc((n.title||"").slice(0,26))}</div>
          ${v?`<span class="simrozet${v==="gecici"?" uyari":""}">${esc(v)}</span>`:""}
        </div>`;}).join("")+`</div>`).join(
      `<div class="arrow"><svg width="18" height="13" viewBox="0 0 20 14" fill="none"
        stroke="currentColor" stroke-width="1.6"><path d="M1 7h16M13 3l4 4-4 4"/></svg></div>`)
      +`</div>`;
  }

  $("ovbody").innerHTML = sec + graf + `
  <div class="mtabs">${MI.motorlar.map((x,i)=>`
    <button class="mtb ${i===MIsec?"on":""}" data-i="${i}">
      <span>${esc(x.baslik)}</span><span class="g">${esc((x.iyi_yonu&&x.iyi_yonu.lakap)
        ||x.katman)}</span></button>`).join("")}</div>
  <div class="mpane">

    <div class="mh">▶ bu grafı koştur — ${esc(m.baslik)} neyi ayırt ediyor</div>
    <p class="note" style="margin:-4px 0 9px">Yukarıdaki graf üstünde koşar.
      <b>tek motorda</b> = izole gör · <b>4 motorda karşılaştır</b> = aynı kurulum
      dördünde birden, fark tablosu çıkar.<br>
      <b>Zaman kazanmak için:</b> her deney birden çok iddiayı aynı anda gösteriyor
      (kartlardaki listeye bak). En verimli tek hareket — herhangi bir deneyde
      <b>⚖ 4 motorda karşılaştır</b>.</p>
    ${dny.map((d,i)=>`
      <div class="mden" id="mden${i}">
        <div class="dh">
          <div class="b"><div class="t">▶ ${esc(d.ad)}</div>
            <div class="d">${esc(d.nis)}</div></div>
          <div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end">
            ${d.dis_dugme?`<span class="dyon">↓ aşağıdaki canvas düğmesiyle</span>`:`
            <button class="iconbtn dkos" data-i="${i}"
              ${d.uygulanabilir?"":"disabled"}>${d.uygulanabilir?"tek motorda":"uygulanamaz"}</button>
            <button class="iconbtn dhep" data-i="${i}"
              style="border-color:var(--accent);color:var(--accent-ink)"
              ${d.uygulanabilir?"":"disabled"}>⚖ 4 motorda karşılaştır</button>`}
          </div>
        </div>
        <div class="dk">
          <span>kurulum: ${d.hedef_id?`<b>${esc(d.hedef_id)}</b> düğümüne
            <b>${esc((d.sim||{}).mod||"—")}</b>`:"hatasız koşu"} · motor
            <b>${esc(d.backend)}</b> · ≈${esc(d.sure)}</span>
        </div>
        <div class="db"><b>bak:</b> ${esc(d.bak)}</div>
        <div class="db"><b>beklenen:</b> ${esc(d.beklenen)}</div>
        ${(d.gosterir||[]).length?`<div class="dgos">
          <b>bu TEK koşu ${d.gosterir.length} şeyi birden gösteriyor:</b>
          <ul>${d.gosterir.map(x=>`<li>${esc(x)}</li>`).join("")}</ul></div>`:""}
        ${d.karsit?`<div class="dc">↔ ${esc(d.karsit)}</div>`:""}
        ${d.neden_olmaz?`<div class="dc">⚠ ${esc(d.neden_olmaz)}</div>`:""}
        <div class="dsonuc" id="dson${i}"></div>
      </div>`).join("")|| (m.ad==="celery_canvas"
        ? `<div class="muyari">Canvas board'suz koşuyor, bu yüzden ayrı bir uçtan
             tetikleniyor — <b>aşağıdaki iki düğmeyi</b> kullan:
             <b>hatasız koştur</b> ve <b>kalıcı hata ile koştur</b>.
             Board'lu motorlarla karşılaştırma sonucun altında çıkıyor.</div>`
        : '<p class="note">bu motor için deney tanımlı değil</p>')}


    <p class="mcumle">« ${esc(m.tek_cumle)} »</p>
    <div class="manaloji"><b>${esc(m.analoji.baslik)}</b>
      <span>${esc(m.analoji.metin)}</span></div>

    ${m.ad==="celery_canvas"?`
    <div class="mden" style="border-color:var(--accent)">
      <div class="dh">
        <div class="b"><div class="t">▶ Celery CANVAS — board YOK</div>
          <div class="d">Aynı grafı Celery'nin KENDİ kompozisyonuyla koştur:
            <code>chain(group(K0), group(K1), …)</code>. Board devre dışı —
            defteri kimse tutmuyor. Farkı burada gör.</div></div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end">
          <button class="iconbtn" id="cvkos"
            style="border-color:var(--accent);color:var(--accent-ink)">hatasız koştur</button>
          <button class="iconbtn" id="cvhata"
            style="border-color:var(--bad);color:var(--bad)">kalıcı hata ile koştur</button>
        </div>
      </div>
      <div class="dk">worker açılışı ~6 sn · hatasız ~9 sn · KALICI hatada
        60 sn bekleyip takılır (nerede takıldığı bilinmez)</div>
      <div class="db"><b>bak:</b> koşarken "hangi adımdayız" diye sorulamıyor —
        elde yalnız <code>ready()</code> var</div>
      <div class="dsonuc" id="cvson"></div>
    </div>

    ${MI.uc_desen?`<div class="mh">${esc(MI.uc_desen.soru)} — üç desen, aynı kütüphane</div>
      <div class="mgz" style="grid-template-columns:1fr 1fr 1fr">
        ${MI.uc_desen.desenler.map(x=>`<div class="kt2">
          <h4 style="color:var(--accent-ink)">${esc(x.ad)} · zinciri ${esc(x.kuran)} kurar</h4>
          <pre class="mkod" style="max-height:none;font-size:10px">${esc(x.kod)}</pre>
          <div style="font-size:12px;line-height:1.5;color:var(--ink2);margin-top:6px"
            >${esc(x.sorun)}</div>
          <div style="font-family:var(--mono);font-size:10.5px;color:var(--faint);margin-top:5px"
            >gönderim: ${esc(x.gonderim)}</div>
        </div>`).join("")}
      </div>
      <table class="mtab" style="margin-top:9px"><thead><tr><th></th>
        <th>düz Celery</th><th>Canvas</th><th>board</th></tr></thead><tbody>
      ${MI.uc_desen.matris.map(r=>`<tr><td><b>${esc(r[0])}</b></td>
        ${[1,2,3].map(i=>`<td style="color:${r[i]==="✓"?"var(--ok)":"var(--bad)"};
          font-weight:700">${esc(r[i])}</td>`).join("")}</tr>`).join("")}
      </tbody></table>
      <p class="note" style="margin-top:6px">ölçüldü: ${esc(MI.uc_desen.olcum)}</p>
      <div class="muyari" style="margin-top:8px">${esc(MI.uc_desen.ders)}</div>`:""}

    ${MI.gecis_anlari?(()=>{const g=MI.gecis_anlari;return `
      <div class="mh">Canvas ne zaman yeter, ne zaman yetmez</div>
      <div class="miyi">
        <div class="ih">✓ <b>${esc(g.yeter.baslik)}</b></div>
        <div class="ic">${esc(g.yeter.kosul)}</div>
        <div class="ip">${esc(g.yeter.neden)}</div>
      </div>
      ${g.gecisler.map(x=>`<div class="minc" style="border-left-color:var(--bad)">
        <div class="t">→ ${esc(x.nereye)}'a geçme anı: ${esc(x.tetik)}</div>
        <div class="d" style="margin-top:5px"><b>Canvas'ın yapamadıkları:</b></div>
        <ul style="margin:4px 0 0;padding-left:17px">
          ${x.canvas_ne_yapamaz.map(y=>
            `<li style="font-size:12.5px;line-height:1.55;color:var(--ink2)">${esc(y)}</li>`).join("")}
        </ul>
        <pre class="mkod" style="margin-top:7px;max-height:none;font-size:10.5px"
          >${esc(x.olculdu)}</pre>
        <div class="s" style="margin-top:6px">${esc(x.kilit)}</div>
      </div>`).join("")}
      <div class="uyari-kutu">
        <b>${esc(g.bizim_is.baslik)}</b>
        <div style="margin-top:4px">${esc(g.bizim_is.metin)}</div>
        <div style="margin-top:6px">${esc(g.bizim_is.sonuc)}</div>
        <div style="margin-top:7px;padding-top:7px;border-top:1px dashed var(--line)">
          <b>${esc(g.bizim_is.vurgu)}</b></div>
      </div>`;})():""}
    `:""}

    ${m.one_cikan&&m.one_cikan.ad?`
    <div class="mbay">
      <div class="bh">🚩 öne çıkan özellik</div>
      <div class="bt">${esc(m.one_cikan.ad)}</div>
      <div class="bo">${esc(m.one_cikan.ozet)}</div>
      <div class="bn">
        <span class="btag ${esc(m.one_cikan.test)}">${
          m.one_cikan.test==="deney"?"▶ bu grafta KOŞULABİLİR":
          m.one_cikan.test==="goster"?"👁 üretilen kodda GÖRÜNÜR":"anlatılır"}</span>
        <span>${esc(m.one_cikan.nerede)}</span>
      </div>
      <div class="bs">⚠ sınır: ${esc(m.one_cikan.sinir)}</div>
    </div>`:""}

    ${m.iyi_yonu&&m.iyi_yonu.lakap?`
    <div class="miyi">
      <div class="ih">${esc(m.baslik)}${m.iyi_yonu.tanim?` — ${esc(m.iyi_yonu.tanim)}`:""}
        · <b>${esc(m.iyi_yonu.lakap)}</b></div>
      <div class="ic">${esc(m.iyi_yonu.tek_cumle)}</div>
      ${m.iyi_yonu.sema?`<div class="mmim" style="margin:8px 0">${esc(m.iyi_yonu.sema)}</div>`:""}
      ${(m.iyi_yonu.nasil||[]).length?`<div class="in">nasıl çalışır</div>
        <ul>${m.iyi_yonu.nasil.map(x=>`<li>${esc(x)}</li>`).join("")}</ul>`:""}
      <div class="iki">
        <div><div class="in" style="color:var(--ok)">artıları</div>
          <ul>${m.iyi_yonu.maddeler.map(x=>`<li>${esc(x)}</li>`).join("")}</ul></div>
        <div><div class="in" style="color:var(--bad)">eksileri</div>
          <ul>${(m.iyi_yonu.eksiler||[]).map(x=>`<li>${esc(x)}</li>`).join("")}</ul></div>
      </div>
      <div class="ip"><b>en parladığı yer:</b> ${esc(m.iyi_yonu.parladigi)}</div>
    </div>`:""}

    ${m.mentalite&&m.mentalite.inanc?`
    <div class="mmen">
      <div class="mh2">çalışma mentalitesi — piyasa neden kullanıyor
        <span>${esc(m.mentalite.donem)}</span></div>
      <div class="minanc">« ${esc(m.mentalite.inanc)} »</div>
      <div class="md2"><b>nereden doğdu:</b> ${esc(m.mentalite.dogus)}</div>
      <div class="md2"><b>kim kullanıyor:</b> ${esc(m.mentalite.kim)}</div>
      <div class="md2"><b>ne için:</b> ${esc(m.mentalite.ne_icin)}</div>
      <div class="md2"><b>neden yaygın:</b></div>
      <ul>${m.mentalite.neden_yaygin.map(x=>`<li>${esc(x)}</li>`).join("")}</ul>
    </div>`:""}

    <div class="mh">avantaj / dezavantaj — her madde bir kanıta bağlı</div>
    <div class="mgz">
      <div class="kt2 g"><h4>avantaj (${m.guclu.length})</h4><ul>${m.guclu.map(x=>
        `<li>${esc(x.iddia)}${rozet(x.kanit)}</li>`).join("")}</ul></div>
      <div class="kt2 z"><h4>dezavantaj (${m.zayif.length})</h4><ul>${m.zayif.map(x=>
        `<li>${esc(x.iddia)}${rozet(x.kanit)}</li>`).join("")}</ul></div>
    </div>
    <div class="mnz">
      <div><b>ne zaman:</b> ${esc(m.ne_zaman)}</div>
      <div><b>ne zaman DEĞİL:</b> ${esc(m.ne_zaman_degil)}</div>
    </div>

    ${(m.birlikte||[]).length?`<div class="mh">birlikte kullanım — hangi yığın, kim ne yapar</div>
      ${m.birlikte.map(b=>`<div class="mbir${b.bizde?" bizde":""}">
        <div class="yt">${esc(b.yigin)}${b.bizde?'<span class="ybiz">bu POC\'ta</span>':""}</div>
        <table class="mtab" style="margin:7px 0"><tbody>${b.kim_ne.map(k=>
          `<tr><td class="m" style="width:110px"><b>${esc(k[0])}</b></td>
           <td>${esc(k[1])}</td></tr>`).join("")}</tbody></table>
        <div class="yn"><b>niçin:</b> ${esc(b.nicin)}</div>
        <div class="yo">${esc(b.not)}</div>
      </div>`).join("")}`:""}

    ${m.gercek_dunya&&m.gercek_dunya.desen?`
      <div class="mh">gerçek dünyada nerede kullanılıyor</div>
      <div class="mger">
        <div class="gd"><b>doğduğu yer:</b> ${esc(m.gercek_dunya.dogdugu_yer)}</div>
        <div class="gd"><b>desen:</b> ${esc(m.gercek_dunya.desen)}</div>
        <ul>${m.gercek_dunya.tipik_isler.map(x=>`<li>${esc(x)}</li>`).join("")}</ul>
        <div class="gu">${esc(MI.gercek_dunya_uyari||"")}</div>
      </div>`:""}

    ${dl?`<div class="mh">bu graf, ${esc(m.baslik)} dilinde
      ${dl.birebir?"":'<span style="color:var(--bad)">· BİREBİR İFADE EDİLEMİYOR</span>'}</div>
      ${(dl.tuzaklar||[]).slice(0,2).map(t=>`<div class="muyari">⚠ ${esc(t)}</div>`).join("")}
      <details class="mdet"><summary>üretilen kaynak kodu göster
        (${(dl.kod||"").split("\n").length} satır)</summary>
        <pre class="mkod">${esc(dl.kod||dl.hata||"")}</pre></details>`:""}

    <details class="mdet"><summary>derinlemesine — modüller · incelikler · güçlü/zayıf ·
      ayarlar · ilişkiler</summary>
      <div class="mmim" style="margin-top:10px">${esc(m.mimari)}</div>

      <div class="mh">modüller (${m.modul_sayim.acik} devrede ·
        ${m.modul_sayim.kapali} devre dışı · ${m.modul_sayim.esgudum} eşgüdüm)</div>
      ${m.moduller.map(x=>`<div class="mrow">
        <span class="mim ${kf(x.durum)}">${esc(MI.durum_etiket[x.durum].im)}</span>
        <div class="b"><div class="t">${esc(x.ad)}${rozet(x.kanit)}</div>
          <div class="d">${esc(x.ne)}</div>
          ${x.neden?`<div class="n">↳ ${esc(x.neden)}</div>`:""}</div></div>`).join("")}

      <div class="mh">incelikler — yalnız bu motorda (${m.incelikler.length})</div>
      ${m.incelikler.map(x=>`<div class="minc">
        <div class="t">◆ ${esc(x.ad)}${rozet(x.kanit)}</div>
        <div class="d">${esc(x.ne)}</div>
        <div class="s">→ ${esc(x.sonuc)}</div></div>`).join("")}

      <div class="mh">ayarlar — ${ay.sayim.acik||0} etkili ·
        ${ay.sayim.kapali||0} ETKİSİZ · ${ay.sayim.sabit||0} sabit</div>
      <table class="mtab"><thead><tr><th></th><th>ayar</th><th>varsayılan</th>
        <th>bizde</th><th>not</th></tr></thead><tbody>
      ${ay.ayarlar.map(a=>`<tr>
        <td class="m">${esc((MI.ayar.durum_etiket[a.durum]||{}).im||"")}</td>
        <td class="m"><b>${esc(a.ad)}</b></td>
        <td class="m">${esc(a.varsayilan)}</td><td class="m">${esc(a.bizde)}</td>
        <td>${esc(a.neden||"")}${a.olculen?`<div style="font-family:var(--mono);
          font-size:10.5px;color:var(--faint)">ölçüldü: ${esc(a.olculen)}</div>`:""}</td>
      </tr>`).join("")}</tbody></table>

      ${m.iliskiler.length?`<div class="mh">ilişkiler — bunlar rakip değil</div>
        ${m.iliskiler.map(i=>`<div class="mil">
          <div class="t">${esc(i.cift[0])} ↔ ${esc(i.cift[1])} · ${esc(i.baslik)}</div>
          <div class="k"><b>kilit:</b> ${esc(i.kilit)}</div>
          <div class="y">⚠ ${esc(i.yanlis_soru)}</div></div>`).join("")}`:""}

      ${MI.kavramlar&&m.ad==="temporal"?(()=>{const k=MI.kavramlar.cluster;return `
        <div class="mh">kavram — ${esc(k.soru)}</div>
        <div class="minc"><div class="t">${esc(k.tanim)}</div>
          <div class="d"><b>neden:</b> ${esc(k.neden)}</div>
          <div class="s">⚠ ${esc(k.yanlis_anlama)}</div></div>
        <p class="note">→ ${esc(k.ozet)}</p>`;})():""}

      <div class="mh">hafıza merdiveni — ${esc(MI.hafiza_merdiveni.ders)}</div>
      <table class="mtab"><tbody>${MI.hafiza_merdiveni.basamaklar.map(b=>
        `<tr${b.motor===m.ad?' style="background:var(--accent-soft)"':''}>
        <td class="m"><b>${esc(b.motor)}</b></td><td>${esc(b.disarida)}</td>
        <td class="m">${esc(b.agirlik)}</td><td>${esc(b.bedeli)}</td></tr>`).join("")}
      </tbody></table>
    </details>

    ${MI.ifade_gucu?`<div class="mh">ifade gücü — Canvas / Airflow / Temporal
        <span style="font-weight:400;text-transform:none;letter-spacing:0"
          >· ${esc(MI.ifade_gucu.senaryo)}</span></div>
      <div class="mgz" style="grid-template-columns:1fr 1fr 1fr">
        ${["canvas","airflow","temporal"].map(k=>`<div class="kt2">
          <h4>${k==="canvas"?"Celery Canvas":k}</h4>
          <pre class="mkod" style="max-height:none;font-size:10px">${
            esc(MI.ifade_gucu.ornekler[k])}</pre></div>`).join("")}
      </div>
      <table class="mtab" style="margin-top:9px"><thead><tr>
        <th>boyut<div style="font-weight:400;font-size:9px;color:var(--faint)"
          >▶ koşturulabilir · ○ yalnız metin</div></th>
        <th>Celery Canvas</th><th>Airflow</th><th>Temporal</th></tr></thead><tbody>
      ${MI.ifade_gucu.satirlar.map(r=>`<tr>
        <td><b>${esc(r[0])}</b>${r[4]?`<div style="font-size:10px;color:var(--ok);
          font-family:var(--mono)">▶ ${esc(r[4])}</div>`:`<div style="font-size:10px;
          color:var(--faint);font-family:var(--mono)">○ yalnız metin</div>`}</td>
        ${[1,2,3].map(i=>{const v=r[i];
          const c=v.startsWith("✓")?"var(--ok)":v.startsWith("✗")?"var(--bad)"
                 :v.startsWith("⚠")?"var(--warn)":"var(--ink2)";
          return `<td style="color:${c}">${esc(v)}</td>`;}).join("")}
      </tr>`).join("")}</tbody></table>
      ${MI.ifade_gucu.kapsam?`<div class="muyari" style="margin-top:8px">
        <b>Kapsam:</b> ${esc(MI.ifade_gucu.kapsam)}</div>`:""}

      <div class="mh">kim nerede kazanır</div>
      ${MI.ifade_gucu.kim_nerede_kazanir.map(r=>`<div class="mrow">
        <div class="b"><div class="t">${esc(r[0])}</div>
          <div class="d">${esc(r[1])}</div></div></div>`).join("")}

      <div class="minc" style="border-left-color:var(--bad)">
        <div class="t">◆ ${esc(MI.ifade_gucu.canvas_sinir.baslik)}</div>
        <pre class="mkod" style="margin-top:6px;font-size:10.5px">${
          esc(MI.ifade_gucu.canvas_sinir.kod)}</pre>
        <div class="d" style="margin-top:6px"><b>Çalışır ama:</b></div>
        <ul style="margin:4px 0 0;padding-left:17px">
          ${MI.ifade_gucu.canvas_sinir.calisir_ama.map(x=>
            `<li style="font-size:12.5px;line-height:1.55;color:var(--ink2)">${esc(x)}</li>`).join("")}
        </ul>
        <div class="s" style="margin-top:6px">${esc(MI.ifade_gucu.canvas_sinir.ders)}</div>
      </div>`:""}

    ${MI.karar_gerekcesi?`<div class="mh">karar gerekçesi — düzeltilmiş hâli</div>
      <div class="uyari-kutu">
        <div><b>Yanlış gerekçe:</b> <s>${esc(MI.karar_gerekcesi.yanlis)}</s></div>
        <div style="margin-top:4px">${esc(MI.karar_gerekcesi.neden_yanlis)}</div>
        <div style="margin-top:7px;padding-top:7px;border-top:1px dashed var(--line)">
          <b>Doğru gerekçe:</b> ${esc(MI.karar_gerekcesi.dogru)}</div>
        <ul style="margin:5px 0 0;padding-left:18px">
          ${MI.karar_gerekcesi.neden.map(x=>`<li>${esc(x)}</li>`).join("")}</ul>
      </div>`:""}

    ${MI.tek_bakista?`<div class="mh">tek bakışta — ${esc(MI.tek_bakista.cerceve)}</div>
      <table class="mtab"><thead><tr><th>motor</th><th>en iyi yaptığı</th>
        <th>en zayıf yanı</th><th>pahalı adım kaç kez</th></tr></thead>
      <tbody>${MI.tek_bakista.satirlar.map(r=>
        `<tr><td class="m"><b>${esc(r[0])}</b></td><td>${esc(r[1])}</td>
         <td style="color:var(--bad)">${esc(r[2])}</td>
         <td class="m" style="color:${r[3]==="1×"?"var(--ok)":"var(--bad)"}">
           <b>${esc(r[3])}</b></td></tr>`).join("")}
      </tbody></table>
      <div class="muyari" style="margin-top:9px"><b>Anlatırken vurgula:</b>
        ${esc(MI.tek_bakista.tek_soru)}</div>
      <div class="muyari" style="margin-top:6px">${esc(MI.tek_bakista.not)}</div>`:""}

    ${MI.mentalite?`<div class="mh">mentaliteleri tek satırda ayır</div>
      <table class="mtab"><thead><tr><th>motor</th><th>inancı</th>
        <th>çözdüğü sorun</th></tr></thead><tbody>
      ${MI.mentalite.tek_satir.map(r=>`<tr><td class="m"><b>${esc(r[0])}</b></td>
        <td style="font-style:italic">« ${esc(r[1])} »</td><td>${esc(r[2])}</td></tr>`).join("")}
      </tbody></table>
      <div class="mh">${esc(MI.mentalite.piyasa.baslik)}</div>
      ${MI.mentalite.piyasa.adimlar.map((x,i)=>`<div class="mrow">
        <span class="mim a">${i+1}</span><div class="b"><div class="d">${esc(x)}</div></div>
      </div>`).join("")}
      <p class="note" style="margin-top:6px">kaynak: ${esc(MI.mentalite.piyasa.kaynak)}</p>`:""}

    <div class="mh">çapa ölçüm — ${esc(MI.capa.soru)}</div>
    <table class="mtab"><thead><tr><th>motor</th><th>fetch</th><th>çökme sonrası</th>
      <th>defteri kim tutar</th></tr></thead><tbody>
    ${Object.entries(MI.capa.satirlar).map(([k,v])=>`<tr${k===m.ad?
      ' style="background:var(--accent-soft)"':''}>
      <td class="m"><b>${esc(k)}</b></td>
      <td class="m" style="color:${v.fetch>1?'var(--bad)':'var(--ok)'}"><b>×${v.fetch}</b></td>
      <td>${esc(v.cokme_sonrasi)}</td><td>${esc(v.defter)}</td></tr>`).join("")}</tbody></table>
  </div>`;

  if($("cvkos")) $("cvkos").onclick=()=>miCanvasKos(false);
  if($("cvhata")) $("cvhata").onclick=()=>miCanvasKos(true);
  document.querySelectorAll(".dkos").forEach(b=>b.onclick=()=>miDeneyKos(+b.dataset.i,false));
  document.querySelectorAll(".dhep").forEach(b=>b.onclick=()=>miDeneyKos(+b.dataset.i,true));
  document.querySelectorAll(".mtb").forEach(b=>b.onclick=()=>{MIsec=+b.dataset.i;miCiz();
    $("ovbody").scrollTop=0;});
  const sel=$("miakis"); if(sel) sel.onchange=()=>openMi(sel.value);
  const ger=$("migeri"); if(ger) ger.onclick=()=>openFlow(MIpid);
}

$("wf").onclick=()=>openMi();      // Motorlar = EN SON graf + motor nişleri
$("mi").onclick=openWf;            // ikincil: akış seçimi / yeni akış üretici
$("scheds").onclick=openScheds;
$("ovclose").onclick=()=>$("ov").classList.remove("on");
$("ovback").onclick=openFlows;
$("send").onclick=send;
$("inp").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}});
$("inp").addEventListener("input",e=>{e.target.style.height="auto";
  e.target.style.height=Math.min(e.target.scrollHeight,180)+"px";});
$("reset").onclick=async()=>{await fetch("/reset?sid="+SID);location.reload();};
loadMeta();
</script></body></html>"""


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    # Arka plan yoklayıcı: vakti gelen zamanlamaları claim edip koşturur.
    # Claim atomik olduğu için birden çok sunucu örneği açılsa bile aynı
    # zamanlama iki kez koşmaz (at-most-once).
    poller = SCH.Poller(SCH_STORE, on_log=lambda m: print(m))
    poller.start()
    print("=" * 66)
    print("BRAIN AGENT — SOHBET arayüzü")
    print(f"  Tarayıcıda aç:  http://127.0.0.1:{PORT}")
    print("  Yaz → ajan grafı kurar → motor yürütür → olaylar canlı akar")
    print("  Board konuşma boyunca KALICI (aynı oturumda büyür)")
    n_sch = len(SCH_STORE.list())
    print(f"  ZAMANLAYICI açık · {n_sch} zamanlama · yoklama {SCH.POLL_SECONDS}s "
          f"· lease {SCH.LEASE_SECONDS}s")
    print("  Ctrl+C ile durdur")
    print("=" * 66)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nkapatılıyor…")
        poller.durdur()
        srv.shutdown()


if __name__ == "__main__":
    main()
