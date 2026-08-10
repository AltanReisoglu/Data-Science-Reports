#!/usr/bin/env python3
"""
web_server.py — Brain Agent demosu için tarayıcı arayüzü (stdlib, sıfır ek bağımlılık).

İki ekseni de ÇALIŞMA ANINDA seçtirir:
  • tool-trace compaction stratejisi : none | hermes | opencode | openclaw | codex | claude_code
  • task-management backend'i        : own | temporal | celery | airflow
Ek olarak çökme/geçici-hata enjeksiyonu ile retry & crash-recovery yollarını canlı gösterir.

Çalıştır:
    .venv/bin/python demo-brain-agent/web_server.py      # → http://127.0.0.1:8020

Yalnız 127.0.0.1'e bağlanır ve sadece kendi agent.py'sini çalıştırır.
"""
from __future__ import annotations

import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
PY = sys.executable
PORT = 8020

sys.path.insert(0, str(HERE))
from compaction import STRATEGIES, STRATEGY_INFO   # noqa: E402
from orchestrator import BACKEND_INFO             # noqa: E402

TIMEOUTS = {"own": 420, "none": 420, "celery": 480, "temporal": 480, "airflow": 30}


def run_board(params: dict) -> dict:
    """Ajan task ÜRETİR → motor yönetir (orchestrator.py)."""
    strategy = params.get("strategy", "hermes")
    backend = params.get("backend", "own")
    if strategy not in STRATEGIES:
        return {"error": f"geçersiz strateji: {strategy}"}
    if backend not in ("own", "temporal", "celery", "airflow"):
        return {"error": f"geçersiz backend: {backend}"}

    cmd = [PY, str(HERE / "orchestrator.py"), "--json",
           "--strategy", strategy, "--backend", backend,
           "--budget", str(int(params.get("budget", 3000)))]
    if params.get("goal"):
        cmd += ["--goal", params["goal"]]
    if params.get("crash_at"):
        cmd += ["--crash-at", params["crash_at"]]
    if params.get("fail_at"):
        cmd += ["--fail-at", params["fail_at"]]
    try:
        p = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True,
                           timeout=TIMEOUTS.get(backend, 480))
        out = p.stdout or ""
        if "<<<JSON>>>" in out:
            return json.loads(out.split("<<<JSON>>>", 1)[1].strip())
        return {"error": "JSON çıktı alınamadı",
                "stderr": (p.stderr or "")[-1500:], "stdout": out[-1500:]}
    except subprocess.TimeoutExpired:
        return {"error": f"zaman aşımı ({TIMEOUTS.get(backend, 480)}s)"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def run_agent(params: dict) -> dict:
    strategy = params.get("strategy", "hermes")
    backend = params.get("backend", "own")
    if strategy not in STRATEGIES:
        return {"error": f"geçersiz strateji: {strategy}"}
    cmd = [PY, str(HERE / "agent.py"), "--json",
           "--strategy", strategy,
           "--budget", str(int(params.get("budget", 3000))),
           "--max-turns", str(int(params.get("max_turns", 4)))]
    if params.get("goal"):
        cmd += ["--goal", params["goal"]]
    try:
        p = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True,
                           timeout=TIMEOUTS.get(backend, 420))
        out = p.stdout or ""
        marker = "<<<JSON>>>"
        if marker in out:
            return json.loads(out.split(marker, 1)[1].strip())
        return {"error": "JSON çıktı alınamadı",
                "stderr": (p.stderr or "")[-1500:], "stdout": out[-1500:]}
    except subprocess.TimeoutExpired:
        return {"error": f"zaman aşımı ({TIMEOUTS.get(backend, 420)}s)"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


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
        elif u.path == "/meta":
            self._send(200, json.dumps({"strategies": STRATEGY_INFO,
                                        "backends": BACKEND_INFO},
                                       ensure_ascii=False),
                       "application/json; charset=utf-8")
        elif u.path == "/run":
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            self._send(200, json.dumps(run_agent(q), ensure_ascii=False),
                       "application/json; charset=utf-8")
        elif u.path == "/board":
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            self._send(200, json.dumps(run_board(q), ensure_ascii=False),
                       "application/json; charset=utf-8")
        else:
            self._send(404, "not found")


PAGE = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Brain Agent — LangGraph + Compaction + Task Management</title>
<style>
:root{--bg:#f5f7fa;--surface:#fff;--surface2:#eef2f7;--border:#d5dde6;--ink:#182231;--soft:#4a5768;
--faint:#8794a5;--accent:#0e7c7b;--accentink:#0a5b5a;--accentsoft:#d5eceb;--ok:#2f855a;--bad:#c0392b;
--warn:#b7791f;--term-bg:#0e131a;--term-ink:#ccd6e2;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
@media(prefers-color-scheme:dark){:root{--bg:#0e131a;--surface:#161d27;--surface2:#1d2530;--border:#2a3441;
--ink:#e5ebf3;--soft:#a5b1c0;--faint:#6a7686;--accent:#2dd4bf;--accentink:#7ff0e3;--accentsoft:#113a37;
--ok:#4ade80;--bad:#f87171;--warn:#e0a94a;--term-bg:#0a0e14;--term-ink:#ccd6e2;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.55}
.wrap{max-width:1120px;margin:0 auto;padding:24px 18px 64px}
h1{font-size:23px;margin:0 0 3px}h1 span{color:var(--accent)}
.sub{color:var(--soft);margin:0 0 16px;max-width:78ch}
.arch{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:12px 15px;
font-family:var(--mono);font-size:11.5px;white-space:pre;overflow-x:auto;margin-bottom:18px;color:var(--soft)}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px 18px;margin-bottom:16px}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}
label{display:block;font-size:11.5px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;color:var(--faint);margin-bottom:5px}
select,input{width:100%;font-family:inherit;font-size:13px;padding:8px 10px;border-radius:8px;
border:1px solid var(--border);background:var(--surface2);color:var(--ink)}
.hint{font-size:11.5px;color:var(--faint);margin-top:5px;min-height:2.4em}
.actions{display:flex;align-items:center;gap:12px;margin-top:16px;flex-wrap:wrap}
button{font-family:inherit;font-size:13.5px;font-weight:700;padding:10px 20px;border-radius:9px;
border:1px solid var(--accent);background:var(--accent);color:#04211f;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.pill{font-size:11.5px;font-weight:700;padding:3px 11px;border-radius:20px;display:none;align-items:center;gap:6px}
.pill.show{display:inline-flex}.pill.run-{background:var(--accentsoft);color:var(--accentink)}
.pill.ok{background:#d7f0e0;color:#1c6b3f}.pill.bad{background:#fbdcd7;color:#b0311f}
@media(prefers-color-scheme:dark){.pill.ok{background:#12381f;color:#4ade80}.pill.bad{background:#3a1a15;color:#f87171}}
.spin{width:13px;height:13px;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;
display:inline-block;animation:sp .7s linear infinite}@keyframes sp{to{transform:rotate(360deg)}}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:4px 0 6px}
.metric{background:var(--surface2);border:1px solid var(--border);border-radius:11px;padding:11px 13px}
.metric .k{font-size:11px;text-transform:uppercase;letter-spacing:.4px;color:var(--faint);font-weight:700}
.metric .v{font-size:20px;font-weight:750;margin-top:3px;font-variant-numeric:tabular-nums}
.metric.good .v{color:var(--ok)}.metric.warn .v{color:var(--warn)}.metric.bad .v{color:var(--bad)}
h2{font-size:15.5px;margin:22px 0 8px}
.term{background:var(--term-bg);color:var(--term-ink);font-family:var(--mono);font-size:11.5px;line-height:1.5;
padding:12px 14px;border-radius:10px;white-space:pre-wrap;word-break:break-word;max-height:400px;overflow:auto}
.term .h{color:#79b8ff}.term .k{color:#4ade80}.term .b{color:#f0a35e}.term .r{color:#f87171}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:8px}
th,td{border:1px solid var(--border);padding:6px 9px;text-align:left;vertical-align:top}
th{background:var(--surface2);font-size:11.5px}
.answer{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:14px 16px;
white-space:pre-wrap;max-height:340px;overflow:auto;font-size:13px}
.badge{display:inline-block;font-size:11px;font-weight:700;padding:2px 9px;border-radius:20px;
background:var(--accentsoft);color:var(--accentink);margin-left:6px}
footer{margin-top:26px;color:var(--faint);font-size:12px}
.hidden{display:none}
</style></head><body><div class="wrap">

<h1>Brain Agent <span>· LangGraph + Tool-Trace Compaction + Task Management</span></h1>
<p class="sub"><b>Ajan bir iş akışı grafı (DAG) kurar</b> — düğümlerin çoğu <b>deterministik fonksiyon</b>
(Airflow operatörü gibi), LLM düğümü yalnız muhakeme gerekince. Sonra <b>task-management altyapısını</b>
(own/temporal/celery/airflow) ve <b>tool-trace compaction stratejisini</b> seçip koştururuz.
Çökme/geçici hata enjekte ederek retry ve crash-recovery yollarını canlı görebilirsin.</p>

<div class="arch">1) PLANLAMA — ajan grafı kurar (LLM, çalışma anında)
     add_step(fn=..., args=..., depends_on=...)   → DETERMİNİSTİK düğüm  ← çoğunluk
     add_agent_step(title=..., depends_on=...)    → LLM düğümü          ← istisna
                              ↓
2) DISPATCH — motor yönetir   [ own │ temporal │ celery │ airflow ]   (seçilebilir)
     recompute_ready (DAG kapısı) → claim (CAS) → run_one_task → complete/fail
     çökme → recover_stale → checkpoint'ten devam
                              ↓
3) DÜĞÜM YÜRÜTME
     kind=function → functions.py'deki kayıtlı fonksiyon (LLM YOK), upstream verisi akar
     kind=agent    → LangGraph ajan döngüsü: reason → act → compact  [ compaction seçilebilir ]</div>

<div class="panel">
  <div class="row">
    <div>
      <label>Mod</label>
      <select id="mode">
        <option value="board" selected>Ajan TASK ÜRETİR → motor yönetir</option>
        <option value="single">Tek ajan koşusu (task üretmeden)</option>
      </select>
      <div class="hint" id="hint-mode"></div>
    </div>
    <div>
      <label>Tool-trace compaction stratejisi</label>
      <select id="strategy"></select>
      <div class="hint" id="hint-strategy"></div>
    </div>
    <div>
      <label>Task-management backend</label>
      <select id="backend"></select>
      <div class="hint" id="hint-backend"></div>
    </div>
    <div>
      <label>Context bütçesi (token)</label>
      <select id="budget">
        <option value="1500">1.500 — çok sıkı</option>
        <option value="3000" selected>3.000 — sıkı (önerilen)</option>
        <option value="8000">8.000 — rahat</option>
        <option value="30000">30.000 — çok rahat</option>
      </select>
      <div class="hint">Bu eşiği aşınca compaction tetiklenir.</div>
    </div>
  </div>
  <div class="row" style="margin-top:14px">
    <div>
      <label>Çökme enjekte et (crash)</label>
      <select id="crash">
        <option value="">— yok —</option>
        <option value="read_file">read_file sonrası worker öldür</option>
        <option value="search_code">search_code sonrası worker öldür</option>
        <option value="run_tests">run_tests sonrası worker öldür</option>
      </select>
      <div class="hint">Adım checkpoint'lendikten SONRA worker ölür → kurtarma + devam.</div>
    </div>
    <div>
      <label>Geçici hata enjekte et (retry)</label>
      <select id="fail">
        <option value="">— yok —</option>
        <option value="read_file">read_file zaman aşımı</option>
        <option value="search_code">search_code zaman aşımı</option>
        <option value="run_tests">run_tests zaman aşımı</option>
      </select>
      <div class="hint">İlk denemede hata → backend'in retry yolu tetiklenir.</div>
    </div>
    <div>
      <label>Maksimum ajan turu</label>
      <select id="turns">
        <option value="3">3</option><option value="4" selected>4</option><option value="6">6</option>
      </select>
      <div class="hint">Her tur = 1 LLM çağrısı (+ tool + compaction).</div>
    </div>
  </div>
  <div class="actions">
    <button id="go">▶ Ajanı çalıştır</button>
    <span class="pill" id="pill"></span>
    <span style="font-size:12px;color:var(--faint)">Gerçek LLM + gerçek framework'ler yerelde koşar (30–120 sn sürebilir).</span>
  </div>
</div>

<div id="results" class="hidden">
  <h2>Ölçümler</h2>
  <div class="cards" id="metrics"></div>

  <div id="boardwrap" class="hidden">
    <h2>1) PLANLAMA — ajanın ÜRETTİĞİ task'lar</h2>
    <div class="term" id="plantrace"></div>
    <h2>Board (task'lar + bağımlılıklar)</h2>
    <div id="boardtable"></div>
    <h2>2) DISPATCH — motorun yönetimi</h2>
    <div class="term" id="dispatchlog"></div>
    <h2>Olay günlüğü (denetim izi)</h2>
    <div class="term" id="boardevents"></div>
  </div>

  <div id="singlewrap">
  <h2>Ajan izi <span class="badge" id="llmbadge"></span></h2>
  <div class="term" id="trace"></div>

  <h2 id="tmtitle">Task-management izi</h2>
  <div class="term" id="tmlog"></div>

  <h2>Compaction olayları (tur tur)</h2>
  <div id="cevents"></div>

  <h2>Ajanın nihai yanıtı</h2>
  <div class="answer" id="answer"></div>
  </div>
</div>

<h2>Task-management karşılaştırma (koçun 6 ekseni)</h2>
<div id="cmp"></div>

<footer>Yerel stdlib http.server (yalnız 127.0.0.1) · agent.py alt süreç olarak koşar ·
LLM anahtarı .env'de tutulur, arayüze hiç gönderilmez.</footer>
</div>

<script>
let META = null;

function esc(s){return String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function colorize(s){return esc(s)
  .replace(/^(──.*|=+.*)$/gm,'<span class="h">$1</span>')
  .replace(/(✓|KORUNDU|complete|done|completed|DEVAM|kazanç)/g,'<span class="k">$1</span>')
  .replace(/(✗|ÇÖKME|hata|HATA|BAŞTAN|TEKRAR)/g,'<span class="r">$1</span>')
  .replace(/(REASON|ACT|COMPACT|recover_stale|checkpoint|claim)/g,'<span class="b">$1</span>');}

async function loadMeta(){
  META = await (await fetch("/meta")).json();
  const ss = document.getElementById("strategy");
  for(const [k,v] of Object.entries(META.strategies)){
    const o=document.createElement("option");o.value=k;
    o.textContent = k + (v.llm ? "  (LLM özet)" : (k==="none"?"":"  (deterministik)"));
    if(k==="hermes") o.selected=true;
    ss.appendChild(o);
  }
  const bs = document.getElementById("backend");
  const order = ["own","temporal","celery","airflow"];
  for(const k of order){
    const o=document.createElement("option");o.value=k;
    o.textContent = META.backends[k]?.etiket || k;
    if(k==="own") o.selected=true;
    bs.appendChild(o);
  }
  ss.onchange=()=>{const v=META.strategies[ss.value];
    document.getElementById("hint-strategy").textContent = v ? `${v.ekol} · ${v.ozet}` : "";};
  bs.onchange=()=>{const v=META.backends[bs.value];
    document.getElementById("hint-backend").textContent = v
      ? `${v.retry_recovery} · operasyonel: ${v.operasyonel}`
      : "LangGraph tek başına koşar; kuyruk/retry/kurtarma yok.";
    document.getElementById("crash").disabled = (bs.value!=="own");
    document.getElementById("crash").title = (bs.value!=="own")
      ? "Çökme enjeksiyonu yalnız 'own' backend'de destekleniyor" : "";};
  const ms = document.getElementById("mode");
  ms.onchange=()=>{
    const board = ms.value==="board";
    document.getElementById("hint-mode").textContent = board
      ? "Ajan create_task ile ÇALIŞMA ANINDA task üretir; motor kuyruk/bağımlılık/retry/kurtarmayı yönetir."
      : "Tek bir ajan koşusu; task üretimi yok (yalnız compaction ölçümü).";
    bs.disabled = !board;   // tek-koşu modunda backend anlamsız
    document.getElementById("crash").title = board
      ? "Başlığında bu geçen task'ta worker çökmesi (own backend)" : "";
  };
  ss.onchange(); bs.onchange(); ms.onchange();
  renderCompare();
}

function renderBoard(j){
  document.getElementById("boardwrap").classList.remove("hidden");
  document.getElementById("singlewrap").classList.add("hidden");

  document.getElementById("plantrace").innerHTML =
    colorize((j.plan_trace||[]).join("\n") || "(planlama izi yok)");

  const tasks = j.tasks||[];
  const cls = {done:"good",failed:"bad",running:"warn",ready:"",blocked:""};
  let h = "<table><thead><tr><th>id</th><th>durum</th><th>düğüm türü</th><th>task (ajanın ürettiği)</th>"+
          "<th>kim üretti</th><th>bağımlı</th><th>deneme</th></tr></thead><tbody>";
  for(const t of tasks){
    const badge = t.status==='cancelled'
        ? `<b style="color:var(--bad)" title="üst düğüm battı — bu düğüm hiç koşmadı">⛔ iptal</b>`
        : `<b style="color:var(--${t.status==='done'?'ok':t.status==='failed'?'bad':t.status==='running'?'warn':'soft'})">${esc(t.status)}</b>`;
    h += `<tr><td style="font-family:var(--mono)">${esc(t.id)}</td><td>${badge}</td>`+
         `<td>${t.kind==="function"
             ? '<b style="color:var(--accent)">fn:'+esc(t.fn||"")+'</b>'
             : '<b style="color:var(--warn)">LLM ajan</b>'}</td><td>${esc(t.title)}</td>`+
         `<td style="font-size:11px">${String(t.created_by||"").startsWith("worker:")
             ? '<b style="color:var(--accent)">yürütme anında</b>' : "planlama"}</td>`+
         `<td style="font-family:var(--mono);font-size:11px">${esc((t.parents||[]).join(", ")||"—")}</td>`+
         `<td>${t.attempt||0}</td></tr>`;
  }
  h += "</tbody></table>";
  document.getElementById("boardtable").innerHTML = h;

  document.getElementById("dispatchlog").innerHTML =
    colorize((j.dispatch_log||[]).join("\n") || "(dispatch izi yok)");

  const evs = (j.events||[]).map(e=>`${e.task_id}  ${e.kind}${e.detail?"  ("+e.detail+")":""}`);
  document.getElementById("boardevents").innerHTML = colorize(evs.join("\n")||"(olay yok)");
}

function renderCompare(){
  const axes = [["task_yonetimi","Task yönetimi"],["retry_recovery","Retry / recovery"],
    ["state_takibi","State takibi"],["scheduling","Scheduling"],
    ["concurrency","Concurrency"],["operasyonel","Operasyonel karmaşıklık"],
    ["dinamik_task","Ajanın DİNAMİK task üretimi"]];
  const keys = ["own","temporal","celery","airflow"];
  let h = "<table><thead><tr><th>Eksen</th>" +
    keys.map(k=>`<th>${esc(META.backends[k].etiket)}</th>`).join("") + "</tr></thead><tbody>";
  for(const [ak,al] of axes){
    h += `<tr><td><b>${al}</b></td>` + keys.map(k=>`<td>${esc(META.backends[k][ak])}</td>`).join("") + "</tr>";
  }
  h += "</tbody></table>";
  document.getElementById("cmp").innerHTML = h;
}

function metric(k,v,kind){return `<div class="metric ${kind||''}"><div class="k">${k}</div><div class="v">${v}</div></div>`;}

document.getElementById("go").onclick = async ()=>{
  const btn=document.getElementById("go"), pill=document.getElementById("pill");
  const q = new URLSearchParams({
    strategy: document.getElementById("strategy").value,
    backend: document.getElementById("backend").value,
    budget: document.getElementById("budget").value,
    max_turns: document.getElementById("turns").value,
  });
  const cr=document.getElementById("crash").value, fa=document.getElementById("fail").value;
  if(cr && !document.getElementById("crash").disabled) q.set("crash_at", cr);
  if(fa) q.set("fail_at", fa);

  btn.disabled=true;
  pill.className="pill run- show"; pill.innerHTML='<span class="spin"></span> ajan çalışıyor…';
  document.getElementById("results").classList.remove("hidden");
  document.getElementById("trace").textContent="LLM düşünüyor, tool'lar koşuyor…";
  document.getElementById("tmlog").textContent=""; document.getElementById("cevents").innerHTML="";
  document.getElementById("answer").textContent=""; document.getElementById("metrics").innerHTML="";

  const mode = document.getElementById("mode").value;
  const t0=performance.now();
  try{
    const j = await (await fetch((mode==="board"?"/board?":"/run?")+q.toString())).json();
    const secs=((performance.now()-t0)/1000).toFixed(1);
    if(j.error){
      pill.className="pill show bad"; pill.textContent="✗ hata · "+secs+"s";
      document.getElementById("singlewrap").classList.remove("hidden");
      document.getElementById("trace").innerHTML='<span class="r">'+esc(j.error)+"</span>\n"+esc(j.stderr||"");
      btn.disabled=false; return;
    }

    if(mode==="board"){
      pill.className="pill show "+(j.ok?"ok":"bad");
      pill.textContent=(j.ok?"✓ tamamlandı":"✗ tamamlanamadı")+" · "+(j.seconds||secs)+"s";
      const c=j.compaction||{};
      let m="";
      m+=metric("Mod","task üretimi");
      m+=metric("Backend", j.backend);
      m+=metric("Planlamada üretilen", j.tasks_created, "good");
      if(j.fn_tasks_run!==undefined) m+=metric("Deterministik fn düğümü", j.fn_tasks_run, "good");
      if(j.agent_tasks_run) m+=metric("LLM ajan düğümü", j.agent_tasks_run, "warn");
      if(j.tasks_spawned_runtime) m+=metric("YÜRÜTME ANINDA üretilen", j.tasks_spawned_runtime, "good");
      m+=metric("Tamamlanan", j.tasks_done, "good");
      if(j.tasks_failed) m+=metric("Başarısız", j.tasks_failed, "bad");
      if(j.retries) m+=metric("Retry", j.retries, "warn");
      if(j.crashes) m+=metric("Çökme", j.crashes, "bad");
      if(j.recovered) m+=metric("Otomatik kurtarma", j.recovered, "good");
      if(c.ham) m+=metric("Compaction kazancı", "%"+c.pct, "good");
      document.getElementById("metrics").innerHTML=m;
      renderBoard(j);
      if(j.note){
        document.getElementById("dispatchlog").innerHTML =
          '<span class="b">'+esc(j.note)+"</span>\n\n"+colorize((j.dispatch_log||[]).join("\n"));
      }
      btn.disabled=false; return;
    }

    document.getElementById("boardwrap").classList.add("hidden");
    document.getElementById("singlewrap").classList.remove("hidden");
    pill.className="pill show "+(j.ok?"ok":"bad");
    pill.textContent=(j.ok?"✓ tamamlandı":"✗ tamamlanamadı")+" · "+(j.seconds||secs)+"s";
    document.getElementById("llmbadge").textContent = j.llm ? ("gerçek LLM: "+j.llm) : "LLM yok";

    // metrikler
    const ce=j.compaction_events||[], tm=j.tm||{};
    const ham=ce.reduce((a,e)=>a+e.before,0), kalan=ce.reduce((a,e)=>a+e.after,0);
    const kaz=ham-kalan, pct=ham? (kaz/ham*100).toFixed(1):0;
    let m="";
    m+=metric("Strateji", j.strategy);
    m+=metric("Backend", j.backend);
    if(ham) m+=metric("Ham tool çıktısı", ham.toLocaleString("tr"));
    if(ham) m+=metric("Compaction sonrası", kalan.toLocaleString("tr"),"good");
    if(ham) m+=metric("Kazanç", "%"+pct,"good");
    if(Object.keys(tm).length){
      m+=metric("Koşan adım", tm.steps_run);
      if(tm.steps_skipped) m+=metric("Checkpoint'ten atlanan", tm.steps_skipped,"good");
      if(tm.retries) m+=metric("Retry", tm.retries,"warn");
      if(tm.crashes) m+=metric("Çökme", tm.crashes,"bad");
      if(tm.recovered) m+=metric("Otomatik kurtarma", tm.recovered,"good");
    }
    document.getElementById("metrics").innerHTML=m;

    document.getElementById("trace").innerHTML=colorize((j.trace||[]).join("\n")||"(iz yok)");
    document.getElementById("tmtitle").style.display="none";
    document.getElementById("tmlog").style.display="none";

    if(ce.length){
      let h="<table><thead><tr><th>#</th><th>Strateji</th><th>Önce</th><th>Sonra</th><th>Kazanç</th><th>Ne yapıldı</th></tr></thead><tbody>";
      ce.forEach((e,i)=>{h+=`<tr><td>${i+1}</td><td>${esc(e.strategy)}</td>
        <td>${e.before.toLocaleString("tr")}</td><td>${e.after.toLocaleString("tr")}</td>
        <td><b>${e.triggered?("%"+e.pct):"—"}</b></td>
        <td style="font-family:var(--mono);font-size:11px">${esc((e.log||[]).slice(0,3).join(" · "))}</td></tr>`;});
      h+="</tbody></table>";
      document.getElementById("cevents").innerHTML=h;
    } else {
      document.getElementById("cevents").innerHTML="<p style='color:var(--faint)'>Compaction olayı yok "+
        "(strateji 'none' ya da bütçe aşılmadı).</p>";
    }
    document.getElementById("answer").textContent=j.answer||"(yanıt yok)";
  }catch(e){
    pill.className="pill show bad"; pill.textContent="✗ "+e;
  }finally{ btn.disabled=false; }
};
loadMeta();
</script></body></html>"""


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print("=" * 74)
    print("BRAIN AGENT — Web demo")
    print(f"  Tarayıcıda aç:  http://127.0.0.1:{PORT}")
    print("  Strateji + backend seç → ▶ Ajanı çalıştır")
    print("  Durdurmak için Ctrl+C")
    print("=" * 74)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nkapatılıyor…")
        srv.shutdown()


if __name__ == "__main__":
    main()
