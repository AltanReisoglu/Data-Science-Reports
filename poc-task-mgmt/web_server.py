#!/usr/bin/env python3
"""
Task-Management POC'larını TARAYICIDAN görsel test etme sunucusu (stdlib, sıfır ek bağımlılık).

Gerçek POC'ları (hermes/temporal/celery) subprocess olarak çalıştırır, çıktıyı
tarayıcıya görsel (adım listesi + sonuç rozetleri) olarak döndürür. Artifact DEĞİL —
çünkü gerçek Python framework'leri (Temporal server, Celery worker) ancak yerelde koşar.

Çalıştır:
    .venv/bin/python poc-task-mgmt/web_server.py
Sonra tarayıcıda:  http://127.0.0.1:8000

Sadece 127.0.0.1'e bağlanır ve yalnızca beyaz-listedeki POC scriptlerini çalıştırır.
"""
from __future__ import annotations
import json, sys, subprocess, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable

# beyaz-liste: sadece bunlar çalıştırılabilir
POCS = {
    # 1. KISIM — framework'ler tek başına
    "hermes":   {"script": HERE / "hermes_real_poc.py",   "label": "Hermes (SQLite kernel)",     "timeout": 60},
    "temporal": {"script": HERE / "temporal_real_poc.py", "label": "Temporal (SDK + dev server)", "timeout": 180},
    "celery":   {"script": HERE / "celery_real_poc.py",   "label": "Celery (worker + broker)",    "timeout": 120},
    # 2. KISIM — beynimizi (brain_chat_V2) altyapılara sarma
    "brain_temporal": {"script": HERE / "brain_on_temporal_poc.py", "label": "brain → Temporal", "timeout": 180},
    "brain_celery":   {"script": HERE / "brain_on_celery_poc.py",   "label": "brain → Celery",   "timeout": 120},
    "brain_hermes":   {"script": HERE / "brain_on_hermes_poc.py",   "label": "brain → Hermes",   "timeout": 60},
    "brain_build":    {"script": HERE / "brain_build_own_poc.py",   "label": "brain → build",    "timeout": 60},
}
# çıktıdan filtrelenecek gürültü
NOISE = re.compile(r"(vulnerable to the WAL|DeprecationWarning|warnings\.warn|pkg_resources|"
                   r"^\[\d{4}-\d\d-\d\d|INFO/|WARNING/|No hostname)")


def run_poc(name: str) -> dict:
    spec = POCS[name]
    try:
        p = subprocess.run([PY, str(spec["script"])], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=spec["timeout"])
        out = p.stdout or ""
        if p.returncode != 0 and p.stderr:
            out += "\n[stderr]\n" + p.stderr[-2000:]
        lines = [l for l in out.splitlines() if not NOISE.search(l)]
        return {"ok": p.returncode == 0, "returncode": p.returncode, "output": "\n".join(lines)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": -1, "output": f"[zaman aşımı: {spec['timeout']}s]"}
    except Exception as e:
        return {"ok": False, "returncode": -1, "output": f"[hata] {e}"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # sessiz
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
            q = parse_qs(u.query)
            name = (q.get("poc") or [""])[0]
            if name not in POCS:
                self._send(400, json.dumps({"ok": False, "output": "bilinmeyen poc"}), "application/json")
                return
            self._send(200, json.dumps(run_poc(name)), "application/json; charset=utf-8")
        else:
            self._send(404, "not found")


PAGE = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Task-Management POC — Görsel Test</title>
<style>
:root{--bg:#f4f6f9;--surface:#fff;--surface2:#eef1f5;--border:#d7dde5;--ink:#1a2230;--soft:#4a5568;--faint:#8a95a5;
--accent:#0d7d78;--accentink:#0a5a56;--accentsoft:#d3ece9;--ok:#2f855a;--bad:#c0392b;--warn:#b7791f;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
--term-bg:#0f141b;--term-ink:#cdd6e2;}
@media(prefers-color-scheme:dark){:root{--bg:#0f141b;--surface:#171e28;--surface2:#1e2732;--border:#2b3543;
--ink:#e6ebf2;--soft:#a7b2c1;--faint:#6b7787;--accent:#2dd4bf;--accentink:#7ff0e3;--accentsoft:#123a37;
--ok:#4ade80;--bad:#f87171;--warn:#e0a94a;--term-bg:#0b0f15;--term-ink:#cdd6e2;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.5}
.wrap{max-width:1060px;margin:0 auto;padding:24px 18px 60px}
h1{font-size:22px;margin:0 0 2px}h1 span{color:var(--accent)}
.sub{color:var(--soft);margin:0 0 8px;max-width:75ch}
.scenario{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:10px 14px;font-size:13px;margin:14px 0 20px}
.scenario code{font-family:var(--mono);background:var(--surface);padding:1px 5px;border-radius:4px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;overflow:hidden;display:flex;flex-direction:column}
.chead{padding:12px 15px;border-bottom:1px solid var(--border)}
.chead h2{font-size:15px;margin:0 0 2px}.chead .p{font-size:12px;color:var(--faint);margin:0}
.cbody{padding:12px 15px;display:flex;flex-direction:column;gap:10px}
.run{align-self:flex-start;font-family:inherit;font-size:13px;font-weight:650;padding:8px 15px;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
:root[data-x] .run,.run{color:#052}@media(prefers-color-scheme:dark){.run{color:#05201e}}
.run:disabled{opacity:.5;cursor:default}
.pill{font-size:11px;font-weight:700;padding:2px 9px;border-radius:20px;display:none;align-items:center;gap:5px}
.pill.show{display:inline-flex}
.pill.run-{background:var(--accentsoft);color:var(--accentink)}
.pill.ok{background:#d7f0e0;color:#1c6b3f}.pill.bad{background:#fbdcd7;color:#b0311f}
@media(prefers-color-scheme:dark){.pill.ok{background:#123a24;color:#4ade80}.pill.bad{background:#3a1a15;color:#f87171}}
.badges{display:flex;flex-wrap:wrap;gap:6px}
.badge{font-size:11px;font-weight:650;padding:3px 9px;border-radius:20px;background:var(--surface2);border:1px solid var(--border);color:var(--ink)}
.badge.good{background:var(--accentsoft);color:var(--accentink);border-color:var(--accentsoft)}
.badge.warn{background:#fdefd6;color:#8a5a08;border-color:#f0dcae}
@media(prefers-color-scheme:dark){.badge.warn{background:#2f2410;color:#e0a94a;border-color:#3a2c12}}
.term{background:var(--term-bg);color:var(--term-ink);font-family:var(--mono);font-size:11px;line-height:1.45;
padding:11px 13px;border-radius:9px;white-space:pre-wrap;word-break:break-word;max-height:340px;overflow:auto;display:none}
.term.show{display:block}
.term .h{color:#79b8ff}.term .k{color:#4ade80;font-weight:700}.term .b{color:#f0a35e}
.spin{width:13px;height:13px;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;display:inline-block;animation:sp .7s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px}th,td{border:1px solid var(--border);padding:5px 8px;text-align:left}th{background:var(--surface2)}
.note{font-size:12px;color:var(--faint);margin-top:6px}
.airflow{grid-column:1/-1}
footer{margin-top:22px;color:var(--faint);font-size:12px}
</style></head><body><div class="wrap">
<h1>Task-Management POC <span>· Görsel Test</span></h1>
<p class="sub">Aşağıdaki butonlar <b>gerçek framework'leri</b> (Hermes kanban_db · Temporal SDK+server · Celery worker+broker) yerelde çalıştırır ve sonucu adım-adım + sonuç rozetleriyle gösterir. Simülasyon değil.</p>
<div class="scenario"><b>Senaryo (hepsi aynı):</b> 3-adımlı iş <code>fetch → process → deliver</code>; <code>process</code> ilk denemede geçici hata verir → <b>retry</b>. Ek: worker <b>çökmesi</b> sonrası ne olur. Kilit soru: <b>retry'da <code>fetch</code> kaç kez koşar?</b></div>

<h2 style="margin:24px 0 6px;font-size:17px">1. KISIM — Framework'ler tek başına</h2>
<div class="grid" id="cards"></div>

<div class="card airflow" style="margin-top:14px">
  <div class="chead"><h2>Airflow (referans DAG — burada koşturulmaz)</h2><p class="p">Kurulum ağır; DAG gerçek/çalıştırılabilir. Ayırt edici: cron scheduling + backfill; sadece hatalı düğüm retry olur, upstream done kalır.</p></div>
  <div class="cbody"><div class="term show"><span class="k">$ pip install "apache-airflow==2.10.*" &amp;&amp; airflow db migrate</span>
<span class="k">$ cp poc-task-mgmt/airflow_dag.py $AIRFLOW_HOME/dags/</span>
<span class="k">$ airflow dags test daily_sales 2026-08-09</span>
schedule="0 8 * * *" · retries=2 · catchup=False · fetch→process→deliver
→ process retry olur; fetch düğümü DONE kalır (statik-DAG kısmi resume)</div></div>
</div>

<h2 style="margin:22px 0 4px;font-size:16px">Çapraz karşılaştırma — retry'da <code>fetch</code> kaç kez?</h2>
<table><thead><tr><th>POC</th><th>Çökme/retry sonrası</th><th>fetch (retry'da)</th></tr></thead><tbody>
<tr><td><b>Hermes</b></td><td>otomatik reclaim + handoff</td><td>başka worker devralır (run#2)</td></tr>
<tr><td><b>Temporal</b></td><td>replay, biten activity atlanır</td><td><b>1×</b> (sadece process 2×)</td></tr>
<tr><td><b>Celery</b></td><td>broker redelivery</td><td><b>2×</b> (baştan)</td></tr>
<tr><td><b>Airflow</b></td><td>sadece hatalı düğüm retry</td><td><b>1×</b> (düğüm ayrı)</td></tr>
</tbody></table>

<h2 style="margin:34px 0 4px;font-size:18px">2. KISIM — Beynimizi (brain_chat_V2) altyapılara sarma <span style="color:var(--accent)">·</span></h2>
<p class="sub">Aynı beyni (<code>brain_core.py</code>) dört altyapıya sarıyoruz — üçü gerçek framework, biri kendi kurduğumuz çekirdek. Beyin 4 adımlı bir iş: <code>retrieve → reason → act → respond</code>; <code>reason</code> (= LLM adımı) ilk denemede geçici hata verir.</p>
<div class="scenario"><b>Kilit soru:</b> çökme/retry sonrası <b>pahalı <code>retrieve</code> kaç kez koşar?</b> — tamamlanan işi koruyor muyuz (1×), yoksa baştan mı başlıyoruz (2×)?</div>

<div class="grid" id="cards2"></div>

<h2 style="margin:22px 0 4px;font-size:16px">Karşılaştırma — aynı beyin, <code>retrieve</code> kaç kez?</h2>
<table><thead><tr><th>Rota</th><th>Taksonomi</th><th>Çökme/retry sonrası</th><th>retrieve</th></tr></thead><tbody>
<tr><td><b>brain → Temporal</b></td><td>buy (durable motora bin)</td><td>replay biten activity'yi atlar</td><td><b>1×</b></td></tr>
<tr><td><b>brain → Hermes</b></td><td>build (hazır motor)</td><td>otomatik reclaim + handoff taşır</td><td><b>1×</b></td></tr>
<tr><td><b>brain → build</b></td><td>build (sıfırdan ~200 satır)</td><td>recover_stale + checkpoint</td><td><b>1×</b></td></tr>
<tr><td><b>brain → Celery</b></td><td>buy (kuyruğa devret)</td><td>self.retry() baştan koşar</td><td><b>2×</b> ⚠️</td></tr>
</tbody></table>
<p class="note">Pahalı <code>retrieve</code> üç dayanıklı rotada 1×, naif Celery'de 2×. Fark: "kaldığı yerden devam" altyapıdan <b>yerleşik gelir</b> (Temporal/Hermes/build) mi, yoksa <b>sen mi kurarsın</b> (Celery)? Tam kod + anlatım: <code>report/brain-chat-v2-task-management-entegrasyon.md</code>.</p>

<footer>Sunucu: yerel stdlib http.server (yalnız 127.0.0.1). POC'lar gerçek framework koşar; Temporal ilk çalıştırmada dev server indirir (~15-30s).</footer>
</div>

<script>
const CARDS=[
 {id:"hermes",title:"Hermes — SQLite durable kernel",prove:"create→claim(CAS/lease)→CRASH→otomatik reclaim→worker-B devralır→done"},
 {id:"temporal",title:"Temporal — durable execution",prove:"workflow + activity RetryPolicy; process 2× ama fetch/deliver 1× (kaldığı yerden)"},
 {id:"celery",title:"Celery — queue + worker",prove:".delay()→worker→self.retry(); retry BAŞTAN koşar (fetch 2×)"},
];
const BADGE={
 hermes:(o)=>{const b=[];
   if(/at-most-once/.test(o))b.push(["at-most-once ✓","good"]);
   const m=o.match(/release_stale_claims\(\)\s*→\s*(\d+)/);if(m)b.push([`crash-recovery: ${m[1]} reclaim ✓`,"good"]);
   if(/worker-B KAZANDI/.test(o))b.push(["worker-B devraldı ✓","good"]);
   if(/run#2\s+status=done/.test(o))b.push(["2 deneme: reclaimed+done","good"]);
   return b;},
 temporal:(o)=>{const b=[];let m;
   if(m=o.match(/process\s+:\s*(\d+) kez/))b.push([`process ×${m[1]} (retry)`,"good"]);
   if(m=o.match(/fetch_data\s+:\s*(\d+) kez/))b.push([`fetch ×${m[1]}`,"good"]);
   if(m=o.match(/toplam (\d+) durable event/))b.push([`${m[1]} durable event`,"good"]);
   return b;},
 celery:(o)=>{const b=[];let m;
   if(/run_order sonucu/.test(o))b.push(["tamamlandı ✓","good"]);
   if(m=o.match(/fetch KAÇ KEZ koştu:\s*(\d+)/))b.push([`fetch ×${m[1]} (BAŞTAN)`,"warn"]);
   return b;},
 brain_temporal:(o)=>{const b=[];let m;
   if(m=o.match(/retrieve\s+:\s*(\d+) kez/))b.push([`retrieve ×${m[1]}`,m[1]==="1"?"good":"warn"]);
   if(m=o.match(/reason\s+:\s*(\d+) kez/))b.push([`reason ×${m[1]} (retry)`,"good"]);
   if(m=o.match(/toplam (\d+) durable event/))b.push([`${m[1]} durable event`,"good"]);
   return b;},
 brain_celery:(o)=>{const b=[];let m;
   if(/run_brain sonucu/.test(o))b.push(["tamamlandı ✓","good"]);
   if(m=o.match(/retrieve KAÇ KEZ koştu:\s*(\d+)/))b.push([`retrieve ×${m[1]} (BAŞTAN)`,m[1]==="1"?"good":"warn"]);
   return b;},
 brain_hermes:(o)=>{const b=[];let m;
   if(m=o.match(/release_stale_claims\(\)\s*→\s*(\d+)/))b.push([`crash-recovery: ${m[1]} ✓`,"good"]);
   if(/handoff okundu/.test(o))b.push(["handoff taşıdı ✓","good"]);
   if(/run#2\s+status=done/.test(o))b.push(["worker-B devraldı ✓","good"]);
   if(m=o.match(/retrieve\s+:\s*(\d+) kez/))b.push([`retrieve ×${m[1]}`,m[1]==="1"?"good":"warn"]);
   return b;},
 brain_build:(o)=>{const b=[];let m;
   if(/at-most-once/.test(o))b.push(["at-most-once ✓","good"]);
   if(m=o.match(/recover_stale\(\)\s*→\s*(\d+)/))b.push([`crash-recovery: ${m[1]} ✓`,"good"]);
   if(m=o.match(/retrieve\s+:\s*(\d+) kez/))b.push([`retrieve ×${m[1]} (checkpoint)`,m[1]==="1"?"good":"warn"]);
   if(m=o.match(/reason\s+:\s*(\d+) kez/))b.push([`reason ×${m[1]}`,"good"]);
   if(/status=done/.test(o))b.push(["done ✓","good"]);
   return b;},
};
function esc(s){return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function colorize(s){return esc(s)
  .replace(/^(──.*|=+|#+.*)$/gm,'<span class="h">$1</span>')
  .replace(/(KANIT:|✓|KAZANDI|done|completed|reclaimed)/g,'<span class="k">$1</span>')
  .replace(/(HATA|None|BAŞTAN|✗)/g,'<span class="b">$1</span>');}
const CARDS2=[
 {id:"brain_temporal",title:"brain → Temporal (buy)",prove:"brain adımları = activity; replay biten activity'yi atlar → retrieve 1×"},
 {id:"brain_hermes",title:"brain → Hermes (build/hazır motor)",prove:"brain işi = kanban kartı; handoff partial-state taşır → retrieve 1×"},
 {id:"brain_build",title:"brain → KENDİ çekirdeğimiz (build)",prove:"~200 satır SQLite: CAS+lease+recover_stale+checkpoint → retrieve 1×"},
 {id:"brain_celery",title:"brain → Celery (buy)",prove:"brain işi = tek task; self.retry() BAŞTAN koşar → retrieve 2×"},
];
function renderCards(list, containerId){
  const wrap=document.getElementById(containerId);
  for(const c of list){
    const el=document.createElement("div");el.className="card";
    el.innerHTML=`<div class="chead"><h2>${c.title}</h2><p class="p">${c.prove}</p></div>
    <div class="cbody">
      <div style="display:flex;align-items:center;gap:10px">
        <button class="run" data-id="${c.id}">▶ Çalıştır</button>
        <span class="pill" id="pill-${c.id}"></span>
      </div>
      <div class="badges" id="badges-${c.id}"></div>
      <div class="term" id="term-${c.id}"></div>
    </div>`;
    wrap.appendChild(el);
  }
}
renderCards(CARDS,"cards");
renderCards(CARDS2,"cards2");
document.querySelectorAll(".run").forEach(btn=>btn.onclick=async()=>{
  const id=btn.dataset.id;
  const pill=document.getElementById("pill-"+id), term=document.getElementById("term-"+id), badges=document.getElementById("badges-"+id);
  btn.disabled=true;badges.innerHTML="";
  pill.className="pill run- show";pill.innerHTML='<span class="spin"></span> çalışıyor…';
  term.className="term show";term.textContent = /temporal/.test(id) ? "Temporal dev server hazırlanıyor (~15-30s)…" : "çalışıyor…";
  const t0=performance.now();
  try{
    const r=await fetch("/run?poc="+id);const j=await r.json();
    const secs=((performance.now()-t0)/1000).toFixed(1);
    term.innerHTML=colorize(j.output||"(çıktı yok)");
    pill.className="pill show "+(j.ok?"ok":"bad");pill.textContent=(j.ok?"✓ başarılı":"✗ hata (rc="+j.returncode+")")+" · "+secs+"s";
    const bs=(BADGE[id]||(()=>[]))(j.output||"");
    for(const [txt,kind] of bs){const s=document.createElement("span");s.className="badge "+kind;s.textContent=txt;badges.appendChild(s);}
  }catch(e){pill.className="pill show bad";pill.textContent="✗ "+e;term.textContent=String(e);}
  finally{btn.disabled=false;}
});
</script></body></html>"""


def main():
    port = 8000
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print("=" * 70)
    print("Task-Management POC — Görsel Test sunucusu")
    print(f"  Tarayıcıda aç:  http://127.0.0.1:{port}")
    print("  Butona bas → gerçek POC koşar → adımlar + sonuç rozetleri görünür.")
    print("  Durdurmak için: Ctrl+C")
    print("=" * 70)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nkapatılıyor…")
        srv.shutdown()


if __name__ == "__main__":
    main()
