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
        elif u.path == "/kiyas":
            # Aynı POC mantığı, ama stdout yerine YAPILANDIRILMIŞ veri: her tool
            # birimi için ÖNCE (ham) / SONRA (context'te kalan). Subprocess değil,
            # import — POC'un kendi fonksiyonları aynı süreçte koşuyor.
            name = (parse_qs(u.query).get("poc") or [""])[0]
            try:
                import kiyas
                veri = kiyas.kosur(name) if name in POCS else kiyas.hepsi()
            except Exception as e:
                veri = {"error": f"{type(e).__name__}: {e}"}
            self._send(200, json.dumps(veri, ensure_ascii=False),
                       "application/json; charset=utf-8")
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
.wrap{max-width:1120px;margin:0 auto;padding:24px 18px 60px}
h1{font-size:22px;margin:0 0 2px}h1 span{color:var(--accent)}
.sub{color:var(--soft);margin:0 0 8px;max-width:78ch}
.scenario{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:10px 14px;font-size:13px;margin:14px 0 18px}
.scenario code{font-family:var(--mono);background:var(--surface);padding:1px 5px;border-radius:4px}

/* ── sekmeler: her ajan mantığı ayrı ── */
.tabs{display:flex;gap:4px;flex-wrap:wrap;margin-top:18px}
.tb{font-family:var(--sans);font-size:13px;text-align:left;color:var(--soft);background:var(--surface2);
 border:1px solid var(--border);border-bottom-color:transparent;border-radius:10px 10px 0 0;padding:9px 14px;
 cursor:pointer;display:flex;flex-direction:column;gap:2px;line-height:1.3}
.tb:hover{color:var(--ink)}
.tb.on{color:var(--ink);background:var(--surface);border-color:var(--accent);border-bottom-color:var(--surface);font-weight:650}
.tb .g{font-family:var(--mono);font-size:10.5px;color:var(--accent);font-variant-numeric:tabular-nums;font-weight:400}
.pane{background:var(--surface);border:1px solid var(--border);border-radius:0 12px 12px 12px;padding:18px;margin-top:-1px}
.meta{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin-bottom:9px}
.k{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--soft);
 border:1px solid var(--border);border-radius:6px;padding:2px 8px}
.k.llm{color:var(--warn);border-color:var(--warn)}
.k.src{text-transform:none;letter-spacing:0;color:var(--faint)}
.pdesc{color:var(--soft);margin:0 0 15px;max-width:82ch;line-height:1.6}
.big{font-family:var(--mono);font-size:15px;margin-bottom:16px;font-variant-numeric:tabular-nums}
.big b{font-size:20px;color:var(--accent)}
.big .m{font-size:12.5px;color:var(--faint);font-family:var(--sans)}

/* ── EK-2 adım şeridi: bu mantık tool izine hangi adımlarla dokundu ── */
.tetik{background:var(--surface2);border:1px solid var(--border);border-radius:10px;
 padding:11px 14px;margin-bottom:16px;font-size:12.5px;line-height:1.6}
.tetik b{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--accent)}
.tetik .r{margin-top:6px}.tetik .r:first-child{margin-top:0}
.hd{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;
 color:var(--faint);margin:20px 0 9px;padding-bottom:6px;border-bottom:1px solid var(--border)}
.zin{display:flex;gap:0;flex-wrap:wrap;align-items:stretch;margin-bottom:14px}
.ad{flex:1 1 150px;min-width:150px;border:1px solid var(--border);background:var(--surface2);
 border-radius:9px;padding:10px 12px;margin:0 5px 8px 0;position:relative;opacity:.45}
.ad.hit{opacity:1;border-color:var(--accent);background:var(--accentsoft)}
.ad.dis{border-style:dashed}
.ad .n{font-size:12.5px;font-weight:650;line-height:1.3}
.ad .e{font-family:var(--mono);font-size:10.5px;color:var(--soft);margin-top:3px;line-height:1.35}
.ad .c{font-family:var(--mono);font-size:10px;margin-top:6px;color:var(--accent);font-weight:650}
.ad.miss .c{color:var(--faint);font-weight:400}
.ad .kb{font-family:var(--mono);font-size:9.5px;color:var(--faint);margin-top:2px}
.adx{border:1px solid var(--border);border-radius:9px;padding:11px 13px;margin-bottom:8px;background:var(--surface2)}
.adx.miss{opacity:.6}
.adx .t{font-size:13px;font-weight:650;display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.adx .t .e{font-family:var(--mono);font-size:11px;font-weight:400;color:var(--soft)}
.adx .t .c{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--accent)}
.adx.miss .t .c{color:var(--faint)}
.adx .d{font-size:12.5px;color:var(--soft);line-height:1.6;margin-top:5px}
.adx .k2{font-family:var(--mono);font-size:10.5px;color:var(--faint);margin-top:5px}
.adx .nt{font-size:12px;color:var(--faint);font-style:italic;margin-top:5px}
.dis-tag{font-family:var(--mono);font-size:9px;text-transform:uppercase;letter-spacing:.07em;
 color:var(--warn);border:1px solid var(--warn);border-radius:5px;padding:1px 6px}
.disar{font-size:12px;color:var(--faint);margin-top:12px;line-height:1.6}
.adref{font-family:var(--mono);font-size:10px;color:var(--accent);border:1px solid var(--accent);
 border-radius:5px;padding:1px 6px;white-space:nowrap}

/* ── tool birimi: okunur satır + açılır ham ÖNCE/SONRA ── */
details.u{border:1px solid var(--border);border-radius:10px;margin-bottom:8px;background:var(--surface2)}
details.u>summary{list-style:none;cursor:pointer;padding:11px 14px;display:flex;gap:11px;align-items:flex-start}
details.u>summary::-webkit-details-marker{display:none}
details.u>summary::before{content:"▸";font-family:var(--mono);color:var(--faint);padding-top:1px}
details.u[open]>summary::before{content:"▾"}
details.u>summary:hover{background:var(--surface)}
.u .h{flex:1;min-width:0}
.u .h1{font-size:13.5px;font-weight:650}
.u .h1 .arg{font-family:var(--mono);font-size:11.5px;font-weight:400;color:var(--soft)}
.u .h2{font-size:12px;color:var(--faint);margin-top:2px}
.fate{font-family:var(--mono);font-size:10px;padding:3px 8px;border-radius:6px;border:1px solid currentColor;white-space:nowrap}
.fate.tam{color:var(--ok)}.fate.orta{color:var(--warn)}.fate.git{color:var(--bad)}
.num{font-family:var(--mono);font-size:11.5px;color:var(--soft);text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
.num .g{display:block;font-size:10px;color:var(--accent)}
.ub{padding:0 14px 14px;display:flex;flex-direction:column;gap:12px}
.why{font-size:12.5px;color:var(--soft);border-left:3px solid var(--accent);padding:7px 11px;
 background:var(--surface);border-radius:0 7px 7px 0;line-height:1.55}
.ba{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:900px){.ba{grid-template-columns:1fr}}
.bx{border:1px solid var(--border);border-radius:9px;overflow:hidden;background:var(--surface);min-width:0}
.bh{font-family:var(--mono);font-size:10px;letter-spacing:.09em;text-transform:uppercase;padding:8px 11px;
 border-bottom:1px solid var(--border);display:flex;justify-content:space-between;gap:8px;color:var(--faint)}
.bx.s .bh{color:var(--accent)}
.bx pre{margin:0;padding:11px;font-family:var(--mono);font-size:11px;line-height:1.5;color:var(--soft);
 white-space:pre-wrap;word-break:break-word;max-height:320px;overflow:auto}
.bx pre.bos{font-style:italic;color:var(--faint)}
.extra{border:1px solid var(--warn);border-radius:10px;padding:12px 14px;margin-top:14px;background:var(--surface2)}
.extra .eh{font-family:var(--mono);font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--warn);margin-bottom:8px}
.extra pre{margin:0;font-family:var(--mono);font-size:11.5px;line-height:1.55;color:var(--soft);
 white-space:pre-wrap;word-break:break-word;max-height:260px;overflow:auto}
details.lg{margin-top:14px}
details.lg summary{cursor:pointer;font-family:var(--mono);font-size:12px;color:var(--faint);padding:6px 0}
.term{background:var(--term-bg);color:var(--term-ink);font-family:var(--mono);font-size:10.5px;line-height:1.45;
padding:11px 13px;border-radius:9px;white-space:pre-wrap;word-break:break-word;max-height:420px;overflow:auto;margin-top:8px}
.term .h{color:#79b8ff}.term .k2{color:#4ade80}.term .n{color:#f0a35e}
.runbtn{font-family:inherit;font-size:12.5px;font-weight:650;padding:7px 13px;border-radius:8px;
 border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer}
.runbtn:disabled{opacity:.5;cursor:default}
.spin{color:var(--faint);font-family:var(--mono);font-size:12.5px;padding:30px 2px}
footer{margin-top:22px;color:var(--faint);font-size:12px}
</style></head><body><div class="wrap">
<h1>Tool-Trace Compaction <span>· Gerçek Python Backend</span></h1>
<p class="sub">Beş ajanın compaction mantığı, <b>her biri kendi sekmesinde</b>. Her tool birimi için
<b>ÖNCE (ham çıktı)</b> ve <b>SONRA (context'te kalan)</b> kırpılmadan açılabiliyor — o mantığın
o birime <b>ne yaptığı ve neden yaptığı</b> yazılı.</p>
<div class="scenario"><b>Not:</b> POC'lar her ajanın compaction mantığının <b>sadık Python simülasyonu</b>dur
(gerçek sabitler/adımlar/invariant'lar). Bu sayfa onların <b>gerçek fonksiyonlarını</b> koşturur;
ham stdout çıktısı da her sekmenin altındaki düğmeyle alınabilir. Bunlar ajanların canlı üretim kodu değildir.</div>
<div class="tabs" id="tabs"></div>
<div id="pane"><div class="spin">beş mantık koşuyor… (üçü gerçek LLM çağırıyor, birkaç saniye sürebilir)</div></div>
<footer>Sunucu: stdlib http.server (yalnız 127.0.0.1). Sekme verisi <code>poc/kiyas.py</code>'den,
ham çıktı <code>poc/*_tool_trace_poc.py</code> subprocess'inden gelir.</footer>
</div>
<script>
const $=i=>document.getElementById(i);
function esc(s){return String(s==null?"":s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function tok(n){return Number(n||0).toLocaleString("tr");}
function insanAd(n){const s=String(n||"").replace(/_/g," ").trim();return s.charAt(0).toUpperCase()+s.slice(1);}
// kader → renk sınıfı: korundu / kısmen kaldı / bağlamdan gitti
function fateK(k){if(k==="tam")return "tam";
 if(/düştü|silindi|girdi|birleşti|indi/.test(k))return "git"; return "orta";}
function colorize(s){return esc(s)
 .replace(/^(=+.*|─.*|.*─{3,}.*|##.*|\[\d+\].*|── .*)$/gm,'<span class="h">$1</span>')
 .replace(/(✓|KAZANÇ|kazanç|Pass \d|COMPACTION|SONUÇ)/g,'<span class="k2">$1</span>')
 .replace(/(SPILL|MICRO|OVERSIZED|placeholder|düştü)/g,'<span class="n">$1</span>');}

let VERI=null, SEC=0;

async function yukle(){
  try{const r=await fetch("/kiyas");VERI=await r.json();}
  catch(e){$("pane").innerHTML='<div class="spin">⚠ '+esc(e.message)+'</div>';return;}
  if(VERI.error){$("pane").innerHTML='<div class="spin">⚠ '+esc(VERI.error)+'</div>';return;}
  ciz();
}

function ciz(){
  $("tabs").innerHTML=VERI.mantiklar.map((m,i)=>
    `<button class="tb ${i===SEC?"on":""}" data-i="${i}"><span>${esc(m.baslik)}</span>
     <span class="g">${tok(m.once)} → ${tok(m.sonra)} · %${m.pct}</span></button>`).join("");
  document.querySelectorAll(".tb").forEach(b=>b.onclick=()=>{SEC=+b.dataset.i;ciz();});
  const m=VERI.mantiklar[SEC];
  $("pane").innerHTML=`<div class="pane">
    <div class="meta"><span class="k">${esc(m.ekol)}</span>
      <span class="k ${m.llm?"llm":""}">${m.llm?"LLM çağırır":"LLM'siz"}</span>
      <span class="k src">${esc(m.kaynak)}</span></div>
    <p class="pdesc">${esc(m.ozet)}</p>
    <div class="big"><b>${tok(m.once)} → ${tok(m.sonra)}</b> token · %${m.pct} kazanç
      <span class="m">· ${m.toollar.length} tool birimi</span></div>
    <div class="tetik">
      ${m.tetik.uretim?`<div class="r"><b>üretim anında</b> — tek çıktının boyutuna bakar<br>${esc(m.tetik.uretim)}</div>`:""}
      <div class="r"><b>eşikte</b> — toplam context'e bakar<br>${esc(m.tetik.esik)}</div>
    </div>
    <div class="hd">tool izine dokunan adımlar — bu koşuda ne oldu</div>
    <div class="zin">${m.adimlar.map(adimKutu).join("")}</div>
    ${m.adimlar.map(adimDetay).join("")}
    <p class="disar"><b>Tool-trace dışı (bu şemada yok):</b> ${m.disarida.map(esc).join(" · ")}.
      Bunlar genel context yönetimidir — tool çıktısına ya da tool_call ↔ tool_result
      çiftine dokunmadıkları için elendi.</p>
    <div class="hd">tool birimleri — her birinin ÖNCE / SONRA'sı</div>
    ${m.toollar.map(birim).join("")||'<p class="pdesc">tool birimi yok</p>'}
    ${(m.ek_mesajlar||[]).map(e=>`<div class="extra">
      <div class="eh">bu mantığın ÜRETTİĞİ / bıraktığı metin · ${esc(e.rol)} · ${tok(e.tok)} token</div>
      <pre>${esc(e.metin)}</pre></div>`).join("")}
    ${(m.log||[]).length?`<details class="lg"><summary>mantığın kendi logu (${m.log.length} satır)</summary>
      <div class="term">${esc(m.log.join("\n"))}</div></details>`:""}
    <details class="lg"><summary>ham POC stdout'u — gerçek subprocess</summary>
      <div style="padding:8px 0"><button class="runbtn" id="rb">▶ ${esc(m.ad)}_tool_trace_poc.py çalıştır</button></div>
      <div class="term" id="rt" style="display:none"></div></details>
  </div>`;
  $("rb").onclick=()=>hamKos(m.ad);
}

function adimKutu(a){
  const hit=a.sayi>0;
  return `<div class="ad ${hit?"hit":"miss"} ${a.tool_izi?"":"dis"}">
    <div class="n">${esc(a.ad)}</div>
    <div class="e">${esc(a.etiket)}</div>
    <div class="c">${hit?`● ${a.sayi} birime vurdu`:"○ bu koşuda vurmadı"}</div>
    <div class="kb">kayıp: ${esc(a.kayip)}</div>
  </div>`;
}
function adimDetay(a){
  const hit=a.sayi>0;
  return `<div class="adx ${hit?"":"miss"}">
    <div class="t">${esc(a.ad)} <span class="e">${esc(a.etiket)}</span>
      ${a.tool_izi?"":'<span class="dis-tag">kapsam dışı</span>'}
      <span class="c">${hit?`${a.sayi} birim`:"vurmadı"}</span></div>
    <div class="d">${esc(a.ozet)}</div>
    ${a.ek_not?`<div class="nt">${esc(a.ek_not)}</div>`:""}
    <div class="k2">kayıp: ${esc(a.kayip)}${a.vurdu.length?` · vurduğu birimler: ${a.vurdu.map(esc).join(", ")}`:""}</div>
  </div>`;
}
function birim(t){
  const gitti=!t.sonra;
  return `<details class="u"><summary>
    <div class="h">
      <div class="h1">${esc(insanAd(t.ad))} <span class="arg">${esc(t.arg||"")}</span></div>
      <div class="h2">${(t.zincir||[]).length?(t.zincir.map(esc).join(" → ")+" · "):""}${esc(t.neden).slice(0,90)}…</div>
    </div>
    <span class="fate ${fateK(t.kader)}">${esc(t.kader)}</span>
    <span class="num">${tok(t.once_tok)} → ${tok(t.sonra_tok)}
      ${t.pct>0?`<span class="g">−%${t.pct}</span>`:""}</span>
  </summary>
  <div class="ub">
    <div class="why">${(t.zincir||[]).length?`<div style="margin-bottom:5px">
        ${t.zincir.map(z=>`<span class="adref">${esc(z)}</span>`).join(' <b>→</b> ')}</div>`:""}
      <b>bu tool'a ne oldu:</b> ${esc(t.neden)}</div>
    <div class="ba">
      <div class="bx"><div class="bh"><span>önce · ham tool çıktısı</span><span>${tok(t.once_tok)} tok</span></div>
        <pre>${esc(t.once)}</pre></div>
      <div class="bx s"><div class="bh"><span>sonra · context'te kalan</span><span>${tok(t.sonra_tok)} tok</span></div>
        ${gitti?'<pre class="bos">Bu birim context\'te AYRI bir mesaj olarak kalmadı — yukarıdaki kutuda o mantığın bıraktığı metne bak.</pre>'
               :`<pre>${esc(t.sonra)}</pre>`}</div>
    </div>
  </div></details>`;
}

async function hamKos(ad){
  const b=$("rb"), t=$("rt");
  b.disabled=true; t.style.display="block"; t.textContent="çalışıyor…";
  try{const r=await fetch("/run?poc="+ad);const j=await r.json();
    t.innerHTML=colorize(j.output||"(çıktı yok)");}
  catch(e){t.textContent="⚠ "+e.message;}
  finally{b.disabled=false;}
}
yukle();
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
