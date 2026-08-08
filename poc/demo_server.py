"""
demo_server.py — POC'un görsel demosu (stdlib http.server; bağımlılık yok).

Tarayıcıda çalışan, tıpkı önceki chat_server+chat_ui demosu gibi: strateji seçersin,
sohbet edersin, arkada GERÇEK Python stratejileri trace'i sıkıştırır. Stratejiyi
değiştirince AYNI trace farklı tool-trace compaction ile yeniden çizilir.

  python demo_server.py            # http://127.0.0.1:8077  (mock varsayılan)
  python demo_server.py --port 9000

Mod:
  - mock: LLM yok, ScriptedBrain deterministik (her ortamda çalışır, anında).
  - live: internal gemma (native tool-use) — .env'de LLM_* dolu olmalı; UI'dan seç.

Endpoint'ler (JSON):
  GET  /                 → demo.html
  GET  /api/strategies   → 13 sistemin meta'sı
  GET  /api/state        → mevcut trace
  POST /api/send         → {message,strategy?,budget?,mode?} bir tur işle
  POST /api/strategy     → {strategy} AYNI trace'i yeniden sıkıştır
  POST /api/budget       → {budget} yeniden sıkıştır
  POST /api/compare      → mevcut trace'i 13 sistemden geçir
  POST /api/reset        → sohbeti sıfırla
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import llm
import strategies
from harness import ChatSession
from engines import make_live_agent

_HERE = Path(__file__).resolve().parent
_LOCK = threading.Lock()

# --- tek oturumlu global durum (yerel demo) ---
STATE: dict = {"session": None, "mode": "mock", "budget": 1500,
               "strategy": "ours", "toolset": "generic", "engine": "-"}


def _make_session(strategy_name: str, budget: int, mode: str, toolset: str):
    strat = strategies.get(strategy_name)
    # product tool'ları (119 gerçek) sadece canlı LLM ile anlamlı çalışır
    if mode == "live" and llm.available():
        agent, engine = make_live_agent(strat, budget, toolset, verbose=False)
        STATE["engine"] = engine  # langgraph | manuel
        return agent, "live"
    STATE["engine"] = "-"
    return ChatSession(strat, budget), "mock"  # mock → generic scripted


def _ensure_session():
    if STATE["session"] is None:
        STATE["session"], STATE["mode"] = _make_session(
            STATE["strategy"], STATE["budget"],
            STATE["mode"] if llm.available() else "mock", STATE["toolset"])
    return STATE["session"]


def _recompact(session):
    session.conv.reset_fates()
    if session.conv.all_results():
        session.last_preamble = session.strategy.compact(
            session.conv.all_results(), session.conv, session.budget)


def _serialize(session, answer: str = "") -> dict:
    results = session.conv.all_results()
    tools = [{
        "turn": r.turn, "name": r.name, "tool_type": r.tool_type,
        "resource": r.resource, "fate": r.fate,
        "raw": r.raw_tokens(), "shown": r.shown_tokens(),
        "note": r.note, "view": (r.shown() or "")[:600],
    } for r in results]
    raw = session.conv.raw_tokens()
    shown = session.conv.shown_tokens(session.last_preamble)
    fates: dict[str, int] = {}
    for r in results:
        fates[r.fate] = fates.get(r.fate, 0) + 1
    is_live = getattr(session, "is_live_agent", False)
    return {
        "tools": tools, "raw": raw, "shown": shown,
        "saved_pct": round(100 * (raw - shown) / raw) if raw else 0,
        "preamble": session.last_preamble, "fates": fates,
        "strategy": session.strategy.name, "budget": session.budget,
        "mode": STATE["mode"], "toolset": STATE["toolset"], "is_live": is_live,
        "engine": STATE.get("engine", "-"), "answer": answer,
        "llm_available": llm.available(),
    }


def _strategies_meta() -> list[dict]:
    return [{"name": s.name, "repo": s.repo, "ref": s.ref, "blurb": s.blurb,
             "uses_llm": s.uses_llm} for s in strategies.all_strategies()]


def _compare(session) -> dict:
    conv = session.conv
    raw = conv.raw_tokens()
    rows = []
    for s in strategies.all_strategies():
        conv.reset_fates()
        pre = s.compact(conv.all_results(), conv, session.budget)
        shown = conv.shown_tokens(pre)
        fates: dict[str, int] = {}
        for r in conv.all_results():
            if r.fate != "TAM":
                fates[r.fate] = fates.get(r.fate, 0) + 1
        rows.append({"name": s.name, "ref": s.ref, "blurb": s.blurb,
                     "shown": shown, "saved": round(100 * (raw - shown) / raw) if raw else 0,
                     "fates": fates, "uses_llm": s.uses_llm})
    _recompact(session)  # aktif stratejiyi geri uygula
    return {"raw": raw, "units": len(conv.all_results()), "rows": rows}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # sessiz
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self):
        p = _HERE / "demo.html"
        body = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._html()
        if self.path == "/api/strategies":
            return self._json({"strategies": _strategies_meta(),
                               "llm_available": llm.available()})
        if self.path == "/api/state":
            with _LOCK:
                return self._json(_serialize(_ensure_session()))
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        data = self._read_json()
        with _LOCK:
            if self.path == "/api/send":
                # opsiyonel: bu turdan önce strateji/bütçe/mod/toolset değiştir
                if data.get("mode") and data["mode"] != STATE["mode"]:
                    STATE["mode"] = data["mode"]; STATE["session"] = None
                if data.get("toolset") and data["toolset"] != STATE["toolset"]:
                    STATE["toolset"] = data["toolset"]; STATE["session"] = None
                if data.get("budget"):
                    STATE["budget"] = int(data["budget"])
                if data.get("strategy"):
                    STATE["strategy"] = data["strategy"]
                sess = _ensure_session()
                sess.budget = STATE["budget"]
                if data.get("strategy"):
                    sess.set_strategy(strategies.get(data["strategy"]))
                msg = (data.get("message") or "").strip()
                if not msg:
                    return self._json(_serialize(sess))
                try:
                    out = sess.send(msg)
                except Exception as e:  # canlı LLM hatası vb.
                    return self._json({"error": f"{type(e).__name__}: {e}",
                                       **_serialize(sess)}, 200)
                return self._json(_serialize(sess, out.get("answer", "")))

            if self.path == "/api/strategy":
                sess = _ensure_session()
                STATE["strategy"] = data.get("strategy", STATE["strategy"])
                sess.set_strategy(strategies.get(STATE["strategy"]))
                _recompact(sess)
                return self._json(_serialize(sess))

            if self.path == "/api/budget":
                sess = _ensure_session()
                STATE["budget"] = int(data.get("budget", STATE["budget"]))
                sess.budget = STATE["budget"]
                _recompact(sess)
                return self._json(_serialize(sess))

            if self.path == "/api/compare":
                return self._json(_compare(_ensure_session()))

            if self.path == "/api/reset":
                STATE["session"] = None
                if data.get("strategy"):
                    STATE["strategy"] = data["strategy"]
                if data.get("mode"):
                    STATE["mode"] = data["mode"]
                if data.get("toolset"):
                    STATE["toolset"] = data["toolset"]
                return self._json(_serialize(_ensure_session()))

        self._json({"error": "not found"}, 404)


def main():
    argv = sys.argv[1:]
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8077
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    live = "VAR (canlı gemma seçilebilir)" if llm.available() else "yok (mock)"
    print(f"POC demo → http://127.0.0.1:{port}")
    print(f"  canlı LLM: {live} · 13 strateji · strateji değiştir → farklı compaction")
    print("  Ctrl+C ile durdur.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\ndurduruldu.")


if __name__ == "__main__":
    main()
