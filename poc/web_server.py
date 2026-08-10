#!/usr/bin/env python3
"""
Tool-Trace Compaction POC'larını TARAYICIDAN çalıştıran GERÇEK PYTHON BACKEND'i (stdlib).

JS taklidi DEĞİL — bu sunucu, gerçek `poc/*_tool_trace_poc.py` dosyalarını subprocess
olarak çalıştırır ve ASIL Python çıktısını (adım adım + sonuç) tarayıcıya döndürür.
(POC'lar, ajanların compaction mantığının sadık Python simülasyonlarıdır; bu backend
onları gerçekten koşturur — artık tarayıcıda yeniden-yazılmış JS değil.)

Çalıştır:
    .venv/bin/python poc/web_server.py
Sonra:  http://127.0.0.1:8010

Sadece 127.0.0.1'e bağlanır; yalnızca beyaz-listedeki 5 POC'u çalıştırır.
"""
from __future__ import annotations
import json, sys, subprocess, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable

POCS = {
    "hermes":      {"script": HERE / "hermes_tool_trace_poc.py",      "label": "Hermes — deterministik 4 geçiş"},
    "openclaw":    {"script": HERE / "openclaw_tool_trace_poc.py",    "label": "OpenClaw — LLM chunk-özet (12 adım)"},
    "opencode":    {"script": HERE / "opencode_tool_trace_poc.py",    "label": "OpenCode — 2 katman (spill + prune)"},
    "codex":       {"script": HERE / "codex_tool_trace_poc.py",       "label": "Codex — ortadan-kes + windowing"},
    "claude_code": {"script": HERE / "claude_code_tool_trace_poc.py", "label": "Claude Code — micro + auto + subagent"},
}


def run_poc(name: str) -> dict:
    spec = POCS[name]
    try:
        p = subprocess.run([PY, str(spec["script"])], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=60)
        out = (p.stdout or "")
        if p.returncode != 0 and p.stderr:
            out += "\n[stderr]\n" + p.stderr[-2000:]
        return {"ok": p.returncode == 0, "returncode": p.returncode, "output": out}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": -1, "output": "[zaman aşımı 60s]"}
    except Exception as e:
        return {"ok": False, "returncode": -1, "output": f"[hata] {e}"}


class H(BaseHTTPRequestHandler):
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
        if u.path == "/":
            self._send(200, PAGE)
        elif u.path == "/run":
            name = (parse_qs(u.query).get("poc") or [""])[0]
            if name not in POCS:
                self._send(400, json.dumps({"ok": False, "output": "bilinmeyen poc"}), "application/json")
                return
            self._send(200, json.dumps(run_poc(name)), "application/json; charset=utf-8")
        else:
            self._send(404, "not found")


PAGE = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tool-Trace Compaction — Gerçek Python Backend</title>
<style>
:root{--bg:#f4f6f9;--surface:#fff;--surface2:#eef1f5;--border:#d7dde5;--ink:#1a2230;--soft:#4a5568;--faint:#8a95a5;
--accent:#0d7d78;--accentink:#0a5a56;--accentsoft:#d3ece9;--ok:#2f855a;--bad:#c0392b;--warn:#b7791f;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
--term-bg:#0f141b;--term-ink:#cdd6e2;}
@media(prefers-color-scheme:dark){:root{--bg:#0f141b;--surface:#171e28;--surface2:#1e2732;--border:#2b3543;
--ink:#e6ebf2;--soft:#a7b2c1;--faint:#6b7787;--accent:#2dd4bf;--accentink:#7ff0e3;--accentsoft:#123a37;
--ok:#4ade80;--bad:#f87171;--warn:#e0a94a;--term-bg:#0b0f15;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:24px 18px 60px}
h1{font-size:22px;margin:0 0 2px}h1 span{color:var(--accent)}
.sub{color:var(--soft);margin:0 0 8px;max-width:78ch}
.scenario{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:10px 14px;font-size:13px;margin:14px 0 18px}
.scenario code{font-family:var(--mono);background:var(--surface);padding:1px 5px;border-radius:4px}
.bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.bar button{font-family:inherit;font-size:13px;font-weight:650;padding:8px 14px;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--ink);cursor:pointer}
.bar button.all{border-color:var(--accent);background:var(--accent);color:#052}
@media(prefers-color-scheme:dark){.bar button.all{color:#05201e}}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;overflow:hidden;display:flex;flex-direction:column}
.chead{padding:12px 15px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;gap:8px}
.chead h2{font-size:14px;margin:0}
.run{font-family:inherit;font-size:12.5px;font-weight:650;padding:6px 12px;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#052;cursor:pointer;white-space:nowrap}
@media(prefers-color-scheme:dark){.run{color:#05201e}}
.run:disabled{opacity:.5;cursor:default}
.cbody{padding:12px 15px;display:flex;flex-direction:column;gap:9px}
.pill{font-size:11px;font-weight:700;padding:2px 9px;border-radius:20px;display:none;align-items:center;gap:5px;align-self:flex-start}
.pill.show{display:inline-flex}.pill.run-{background:var(--accentsoft);color:var(--accentink)}
.pill.ok{background:#d7f0e0;color:#1c6b3f}.pill.bad{background:#fbdcd7;color:#b0311f}
@media(prefers-color-scheme:dark){.pill.ok{background:#123a24;color:#4ade80}.pill.bad{background:#3a1a15;color:#f87171}}
.badges{display:flex;flex-wrap:wrap;gap:6px}
.badge{font-size:11px;font-weight:650;padding:3px 9px;border-radius:20px;background:var(--surface2);border:1px solid var(--border)}
.badge.good{background:var(--accentsoft);color:var(--accentink);border-color:var(--accentsoft)}
.term{background:var(--term-bg);color:var(--term-ink);font-family:var(--mono);font-size:10.5px;line-height:1.45;
padding:11px 13px;border-radius:9px;white-space:pre-wrap;word-break:break-word;max-height:420px;overflow:auto;display:none}
.term.show{display:block}
.term .h{color:#79b8ff}.term .k{color:#4ade80}.term .n{color:#f0a35e}
.spin{width:12px;height:12px;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;display:inline-block;animation:sp .7s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
footer{margin-top:20px;color:var(--faint);font-size:12px}
</style></head><body><div class="wrap">
<h1>Tool-Trace Compaction <span>· Gerçek Python Backend</span></h1>
<p class="sub">Butona bas → sunucu <b>gerçek <code>poc/*_tool_trace_poc.py</code></b> dosyasını çalıştırır → ASIL Python çıktısı (adım adım + sonuç) aşağıda görünür. Bu, tarayıcıda JS taklidi değil; arka planda gerçek Python koşuyor.</p>
<div class="scenario"><b>Not:</b> POC'lar her ajanın compaction mantığının <b>sadık Python simülasyonu</b>dur (gerçek sabitler/adımlar/invariant'lar). Backend onları gerçekten koşturur; ama bunlar ajanların canlı üretim kodu değildir.</div>
<div class="bar"><button class="all" id="runAll">▶ Tümünü çalıştır</button></div>
<div class="grid" id="cards"></div>
<footer>Sunucu: stdlib http.server (yalnız 127.0.0.1). POC'lar hızlıdır (~1s). Çıktı gerçek subprocess stdout'udur.</footer>
</div>
<script>
const CARDS=[
 {id:"hermes",title:"Hermes — deterministik 4 geçiş"},
 {id:"openclaw",title:"OpenClaw — LLM chunk-özet (12 adım)"},
 {id:"opencode",title:"OpenCode — 2 katman (spill + prune)"},
 {id:"codex",title:"Codex — ortadan-kes + windowing"},
 {id:"claude_code",title:"Claude Code — micro + auto + subagent"},
];
const BADGE=(o)=>{const b=[];let m;
 if(m=o.match(/([\d.,]+)\s*→\s*([\d.,]+)\s*token/)) b.push([`${m[1]}→${m[2]} token`,"good"]);
 if(m=o.match(/\(([\d.]+)%\s*kazan/)) b.push([`%${m[1]} kazanç`,"good"]);
 if(/bütünlüğü\s*:?\s*✓|çift bütünlüğü\s*:\s*✓|tool_call_id bütünlüğü\s*:\s*✓/.test(o)) b.push(["çift bütünlüğü ✓","good"]);
 if(/sır sızıntısı\s*:\s*✓|sır sızıntısı yok/.test(o)) b.push(["sır sızmadı ✓","good"]);
 if(/skill korundu\s*:\s*✓/.test(o)) b.push(["skill korundu ✓","good"]);
 if(m=o.match(/window sayısı[^:]*:\s*(\d+)/)) b.push([`window zinciri: ${m[1]}`,"good"]);
 if(m=o.match(/auto-compaction sayısı\s*:\s*(\d+)/)) b.push([`auto-compaction: ${m[1]}`,"good"]);
 if(m=o.match(/subagent kaçışı\s*:\s*(\d+)/)) b.push([`subagent: ${m[1]}`,"good"]);
 return b;};
function esc(s){return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function colorize(s){return esc(s)
 .replace(/^(=+.*|─.*|.*─{3,}.*|##.*|\[\d+\].*|── .*)$/gm,'<span class="h">$1</span>')
 .replace(/(✓|KAZANÇ|kazanç|Pass \d|COMPACTION|SONUÇ|done|completed)/g,'<span class="k">$1</span>')
 .replace(/(SPILL|MICRO|OVERSIZED|placeholder|düştü|not\b|NOT)/g,'<span class="n">$1</span>');}
const wrap=document.getElementById("cards");
for(const c of CARDS){const el=document.createElement("div");el.className="card";
 el.innerHTML=`<div class="chead"><h2>${c.title}</h2><button class="run" data-id="${c.id}">▶ Çalıştır</button></div>
 <div class="cbody"><span class="pill" id="pill-${c.id}"></span><div class="badges" id="badges-${c.id}"></div><div class="term" id="term-${c.id}"></div></div>`;
 wrap.appendChild(el);}
async function runOne(id){
 const pill=document.getElementById("pill-"+id),term=document.getElementById("term-"+id),badges=document.getElementById("badges-"+id),
       btn=document.querySelector(`.run[data-id="${id}"]`);
 btn.disabled=true;badges.innerHTML="";pill.className="pill run- show";pill.innerHTML='<span class="spin"></span> çalışıyor…';
 term.className="term show";term.textContent="çalışıyor…";const t0=performance.now();
 try{const r=await fetch("/run?poc="+id);const j=await r.json();const secs=((performance.now()-t0)/1000).toFixed(2);
  term.innerHTML=colorize(j.output||"(çıktı yok)");
  pill.className="pill show "+(j.ok?"ok":"bad");pill.textContent=(j.ok?"✓ başarılı":"✗ hata")+" · "+secs+"s";
  for(const [t,k] of BADGE(j.output||"")){const s=document.createElement("span");s.className="badge "+k;s.textContent=t;badges.appendChild(s);}
 }catch(e){pill.className="pill show bad";pill.textContent="✗ "+e;term.textContent=String(e);}
 finally{btn.disabled=false;}
}
document.querySelectorAll(".run").forEach(b=>b.onclick=()=>runOne(b.dataset.id));
document.getElementById("runAll").onclick=async()=>{for(const c of CARDS){await runOne(c.id);}};
</script></body></html>"""


def main():
    port = 8010
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print("=" * 70)
    print("Tool-Trace Compaction — Gerçek Python Backend")
    print(f"  Tarayıcıda aç:  http://127.0.0.1:{port}")
    print("  Butona bas → gerçek poc/*_tool_trace_poc.py koşar → ASIL çıktı görünür.")
    print("  Durdur: Ctrl+C")
    print("=" * 70)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
