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
from compaction import STRATEGIES, STRATEGY_INFO   # noqa: E402
from taskboard import TaskBoard  # noqa: E402

PORT = 8030
SESSIONS: dict[str, dict] = {}
_LOCK = threading.Lock()


def get_session(sid: str) -> dict:
    with _LOCK:
        if sid not in SESSIONS:
            SESSIONS[sid] = {
                "board": TaskBoard(),          # konuşma boyunca KALICI
                "history": [],
                "settings": {"backend": "own", "strategy": "hermes", "budget": 3000,
                             "pack": "audit"},
                "busy": False,
            }
        return SESSIONS[sid]


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
                    raw = F.raw_chars(res_d)
                    rc = len(json.dumps(res_d, ensure_ascii=False, default=str))
                    if raw:
                        emit({"type": "reduction", "task": "-", "fn": fn, "title": "sohbet",
                              "raw_tokens": raw // 4, "out_tokens": rc // 4,
                              "pct": round((1 - rc / raw) * 100, 1),
                              "args": fa,
                              "result": json.dumps(
                                  {k: v for k, v in res_d.items()
                                   if k not in ("_raw_chars", "rows", "text")},
                                  ensure_ascii=False, indent=1)[:2500]})
                    ozet = {k: v for k, v in res_d.items()
                            if k not in ("_raw_chars", "rows", "text")}
                    out_txt = json.dumps(ozet, ensure_ascii=False, default=str)
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
                emit({"type": "compaction", "task": "-",
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
            for k in ("backend", "strategy", "crash_at", "fail_at", "pack"):
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
  <div class="ctl"><label>çökme</label><select id="crash">
    <option value="">—</option><option value="oku">oku</option><option value="tara">tara</option>
    <option value="test">test</option><option value="rapor">rapor</option></select></div>
  <span class="sp"></span>
  <button class="iconbtn" id="flows">Akışlar</button>
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
      <h3>1 · Deterministik indirgeme</h3>
      <div id="tred"><p class="note">Henüz yok.</p></div>
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
  for(const el of ["backend","strategy","budget","crash"]) $(el).onchange=push;
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
    else if(ev.type==="saved"){ lines.push(`  ⤷ akış kaydedildi: ${ev.id} (${ev.n} düğüm) — "Akışlar"dan görüntüle`); show(); }
    else if(ev.type==="node_added"){ nodes++; M.textContent=nodes+" düğüm";
      lines.push(ev.text); show(); }
    else if(ev.type==="reduction"){ RED.push(ev); red();
      lines.push(`  ⤷ fn:${ev.fn}  ${ev.raw_tokens}→${ev.out_tokens} token (−%${ev.pct}, LLM yok)`); show(); }
    else if(ev.type==="compaction"){ for(const c of ev.events) TRACE.push({...c,node:ev.title,detail:ev.detail}); trc();
      lines.push(`  ⤷ compaction ${ev.title}: `+ev.events.map(c=>`${c.before}→${c.after} (−%${c.pct})`).join(", ")); show(); }
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

async function openFlow(id){
  const d=await (await fetch("/pipeline?id="+id)).json();
  $("ovtitle").textContent=d.goal;
  $("ovback").style.display="";
  const byId={}; d.nodes.forEach(n=>byId[n.id]=n);
  const html=d.layers.map((lay,i)=>{
    const col='<div class="lay">'+lay.map(nid=>{const n=byId[nid];
      const kind=n.kind==="function"?"fn":"agent";
      return `<div class="nd ${kind}">
        <div class="n1">${n.kind==="function"?esc(n.fn):"LLM AJAN"}</div>
        <div class="n2">${esc(n.title)}</div>
        ${Object.keys(n.args||{}).length?`<div class="n3">${esc(JSON.stringify(n.args))}</div>`:""}
        ${n.result?`<div class="n4">${esc(n.result)}</div>`:""}
      </div>`;}).join("")+'</div>';
    const arr=i<d.layers.length-1?`<div class="arrow">
      <svg width="22" height="14" viewBox="0 0 24 14" fill="none" stroke="currentColor" stroke-width="1.6">
      <path d="M1 7h20M16 2l5 5-5 5" stroke-linecap="round" stroke-linejoin="round"/></svg></div>`:"";
    return col+arr;}).join("");
  const st=d.stats||{};
  $("ovbody").innerHTML=`<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
      <button class="iconbtn" id="runp" style="border-color:var(--accent);color:var(--accent-ink)">▶ Bu akışı çalıştır</button>
      <span class="note" style="margin:0">planlama yapılmaz — kayıtlı graf aynen koşar</span></div>
    <div class="note" style="margin-bottom:10px">
      paket <b>${esc(d.pack)}</b> · motor <b>${esc(d.backend)}</b> · ${d.nodes.length} düğüm
      ${st.sn?" · "+st.sn+" sn":""}${st.cokme?" · çökme "+st.cokme+" → kurtarma "+st.kurtarma:""}
    </div><div class="dag">${html}</div>`;
  $("runp").onclick=()=>{ $("ov").classList.remove("on"); runPipeline(id, d.goal); };
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
    else if(ev.type==="reduction"){ RED.push(ev); red();
      lines.push(`  ⤷ fn:${ev.fn}  ${ev.raw_tokens}→${ev.out_tokens} token (−%${ev.pct})`); show(); }
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
$("flows").onclick=openFlows;
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
    print("=" * 66)
    print("BRAIN AGENT — SOHBET arayüzü")
    print(f"  Tarayıcıda aç:  http://127.0.0.1:{PORT}")
    print("  Yaz → ajan grafı kurar → motor yürütür → olaylar canlı akar")
    print("  Board konuşma boyunca KALICI (aynı oturumda büyür)")
    print("  Ctrl+C ile durdur")
    print("=" * 66)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nkapatılıyor…")
        srv.shutdown()


if __name__ == "__main__":
    main()
