"""
chat_server.py — localhost'ta tarayıcı chatbot'u. Gerçek Gemma + trace compaction.

  python chat_server.py            # http://localhost:8000  (equity domain)
  python chat_server.py --file     # dosya tool'ları
  python chat_server.py --port 8010

stdlib http.server kullanır (ekstra bağımlılık yok). Kalıcı bir TracingAgent
tutar → sohbet çok-turlu, trace turlar boyunca birikir, compaction arkada işler.
Tarayıcı sayfası aynı sunucudan servis edilir; /chat POST'una mesaj gönderir,
yanıt + arkadaki trace durumunu (kader/token/playbook/ledger) JSON alır.
"""
from __future__ import annotations
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent import TracingAgent
from config import estimate_tokens
import chat as chatmod

HERE = Path(__file__).parent
LOCK = threading.Lock()
STATE = {"ag": None, "domain": "equity"}


def new_agent(domain: str) -> TracingAgent:
    if domain == "equity":
        import equity_tools as eq
        ag = TracingAgent(compaction=True, schemas=eq.SCHEMAS, dispatch=eq.DISPATCH,
                          tool_meta=eq.TOOL_META, system=chatmod.EQUITY_SYSTEM, max_turns=16)
    elif domain == "product":
        import product_tools as pt
        ag = TracingAgent(compaction=True, schemas=pt.SCHEMAS, dispatch=pt.DISPATCH,
                          tool_meta=pt.TOOL_META, system=chatmod.PRODUCT_SYSTEM, max_turns=20)
    else:
        ag = TracingAgent(compaction=True, max_turns=16)
    ag.compactor.budget = 1200          # tarayıcıda compaction'ı görebilmek için
    ag.compactor.target = 550           # daha agresif tool-çıktı sıkıştırma (0.46×budget)
    return ag


def snapshot(ag) -> dict:
    """Arkadaki trace-compaction durumunu tarayıcıya JSON olarak ver."""
    evs = ag.trace.tool_events()

    def _olay(seq):
        """Bu tool olayı HANGİ olaya ait? Önce ajan-bildirimli CWL episode,
        yoksa ledger'ın otomatik çıkardığı kategori."""
        ep = ag.episodes.episode_of(seq)
        if ep is not None:
            return {"olay": ep.name, "tip": ep.type, "kaynak": "cwl"}
        return {"olay": ag.ledger.category_of(seq), "tip": ag.ledger.category_of(seq),
                "kaynak": "kategori"}

    import harness_kiyas as HK
    sozluk = HK.tool_sozlugu(getattr(ag, "schemas", None))
    trace = [{"seq": e.seq, "name": e.payload["name"],
              "aciklama": sozluk.get(e.payload["name"], ""),
              "args": HK.arg_metni(e.payload.get("args") or {}),
              "dondu": HK.ilk_satir(str(e.payload.get("output", ""))),
              "tok": estimate_tokens(str(e.payload.get("output", ""))),
              "neden": e.neden,
              "fate": "SİLİNDİ" if e.cleared else "ÖZET" if e.evicted else "TAM",
              **_olay(e.seq)} for e in evs]
    L = ag.ledger
    seen = {}
    for o in L.observations:
        if o.path:
            seen[o.path] = {"resource": o.path, "version": L.local_counters.get(o.path, 0),
                            "stale": L.is_stale(o.event_seq)}
    m = ag.metrics
    return {"trace": trace, "tokens": ag.trace.total_tokens(),
            "peak": max(m["peak_trace_tokens"], ag.trace.total_tokens()),
            "budget": ag.compactor.budget,
            "metrics": {"tool_events": len(evs), "compaction_passes": m["compaction_passes"],
                        "evicted": m["evicted"], "playbook": m["playbook_bullets"]},
            "playbook": [{"tag": b.tag, "text": b.text[:130], "helpful": b.helpful}
                         for b in ag.playbook.active_bullets()],
            "ledger": list(seen.values())}


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._reply(200, (HERE / "chat_ui.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._reply(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            data = {}

        if self.path == "/reset":
            with LOCK:
                STATE["ag"] = new_agent(STATE["domain"])
            self._reply(200, b'{"ok":true}')
            return

        # AYNI trace, ALTI mantık. Sohbet turunda değil TALEP ÜZERİNE koşuyor:
        # üç harness (openclaw/codex/claude_code) gerçek LLM çağırıyor, her mesajda
        # koşturmak sohbeti yavaşlatırdı.
        if self.path == "/harness":
            import harness_kiyas as HK
            with LOCK:
                ag = STATE["ag"]
                if not ag._call_seq:
                    self._reply(200, json.dumps(
                        {"bos": True, "mantiklar": []}, ensure_ascii=False).encode())
                    return
                try:
                    out = HK.kiyasla(ag, budget=data.get("budget") or None)
                except Exception as e:
                    out = {"error": f"{type(e).__name__}: {e}"}
            self._reply(200, json.dumps(out, ensure_ascii=False).encode("utf-8"))
            return

        if self.path == "/chat":
            msg = (data.get("message") or "").strip()
            if not msg:
                self._reply(400, json.dumps({"error": "boş mesaj"}).encode()); return
            with LOCK:
                ag = STATE["ag"]
                snap = dict(ag.metrics)
                try:
                    answer = ag.send(msg)
                except Exception as e:
                    self._reply(200, json.dumps(
                        {"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False).encode())
                    return
                s = snapshot(ag)
                s["answer"] = answer
                s["delta"] = {"tools": ag.metrics["tool_calls"] - snap["tool_calls"],
                              "compactions": ag.metrics["compaction_passes"] - snap["compaction_passes"],
                              "evicted": ag.metrics["evicted"] - snap["evicted"]}
            self._reply(200, json.dumps(s, ensure_ascii=False).encode("utf-8"))
            return

        self._reply(404, b"{}")

    def log_message(self, *a):
        pass   # sessiz


def main():
    argv = sys.argv[1:]
    STATE["domain"] = ("product" if "--product" in argv
                       else "file" if "--file" in argv else "equity")
    port = 8000
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])
    STATE["ag"] = new_agent(STATE["domain"])
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("═" * 56)
    print(f"  Trace Compaction Chat  →  http://localhost:{port}")
    print(f"  domain={STATE['domain']} · model=Gemma · compaction AÇIK")
    print("  Tarayıcıda aç, yaz. Durdurmak için Ctrl+C.")
    print("═" * 56)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nkapatıldı.")


if __name__ == "__main__":
    main()
