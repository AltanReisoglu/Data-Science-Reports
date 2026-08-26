"""
HUD — masaüstünün sağında açılan canlı izleme paneli.

NEDEN TARAYICI: `tkinter` bu makinede kurulu değil ve kurulum istemek için
fazla küçük bir iş. Chrome zaten var ve `--app=` modu çerçevesiz, konumlanabilir
bir pencere veriyor. Sıfır kurulum.

NEDEN localhost SUNUCU: `file://` üzerinden JS `fetch` CORS'a takılıyor;
tek çare sayfanın tamamını `meta refresh` ile yeniden yüklemek olurdu ve her
saniye titrerdi. `http.server` stdlib — 127.0.0.1'e bağlanan küçük bir sunucu
JSON'u pürüzsüz besliyor.

GERİ BESLEME TUZAĞI: panel ekranda duruyor, yani ajanın aldığı ekran
görüntüsünde panelin kendisi de görünür — ve panel o görüntüyü gösterdiği için
sonsuz ayna oluşur. Ajanın modeli kendi HUD'unu "arayüz" sanıp ona tıklamaya
çalışabilir. Çözüm: yakalama alanı panelin SOLUNDA kesiliyor
(`X11Sandbox.capture_width`). Ajan paneli hiç görmüyor.

PANELDE NE VAR:
    ekran goruntusu + hedefe artı işareti   ajan NEREYE bakıyor
    eylem · hedef · aktif pencere            ne yapmak üzere
    adım / token / maliyet + bütçe çubukları  ne harcadı
    engellenen eylemler (kırmızı)             neye izin verilmedi
    son eylem akışı                           nereden geldi
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer

HUD_WIDTH = 470          # panel genişliği (px) — yakalama bu kadar kısalıyor


def _bos_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, hud, *a, **kw):
        self.hud = hud
        super().__init__(*a, **kw)

    def log_message(self, *a):        # sessiz — terminali kirletmesin
        pass

    def do_GET(self):
        if self.path.startswith("/state"):
            govde = json.dumps(self.hud.state).encode()
            tip = "application/json"
        else:
            govde = SAYFA.encode()
            tip = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", tip)
        self.send_header("Content-Length", str(len(govde)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(govde)


class Hud:
    """Sağ kenarda açılan canlı panel."""

    def __init__(self, screen_w: int, screen_h: int, width: int = HUD_WIDTH,
                 auto_open: bool = True):
        self.width = width
        self.screen_w, self.screen_h = screen_w, screen_h
        self.state: dict = {"faz": "hazir", "eylem": "-", "hedef": None,
                            "pencere": "", "adim": 0, "token": 0, "cost": 0.0,
                            "limit": {}, "akis": [], "engel": [], "png": None,
                            "olcek": 1.0, "strateji": "", "gorev": "",
                            "durum": None, "sebep": ""}
        self.port = _bos_port()
        self._srv = HTTPServer(("127.0.0.1", self.port), partial(_Handler, self))
        self._t = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._t.start()
        self._proc = None
        if auto_open:
            self.ac()

    # -- pencere -----------------------------------------------------------

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def ac(self) -> None:
        """Chrome'u çerçevesiz uygulama penceresi olarak SAĞ KENARDA aç."""
        chrome = next((c for c in ("google-chrome", "chromium", "chromium-browser",
                                   "brave-browser", "microsoft-edge")
                       if shutil.which(c)), None)
        if not chrome:
            print(f"  ! tarayici bulunamadi — paneli elle ac: {self.url}")
            return
        x = max(0, self.screen_w - self.width)
        self._proc = subprocess.Popen(
            [chrome, f"--app={self.url}",
             f"--window-position={x},0",
             f"--window-size={self.width},{self.screen_h}",
             "--user-data-dir=" + os.path.join(
                 os.environ.get("TMPDIR", "/tmp"), "cua_hud_profile"),
             "--no-first-run", "--no-default-browser-check",
             "--disable-features=Translate,MediaRouter"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.2)          # pencere yerleşsin

    def kapat(self) -> None:
        if self._proc:
            self._proc.terminate()
        self._srv.shutdown()

    # -- güncelleme --------------------------------------------------------

    def guncelle(self, **alanlar) -> None:
        self.state.update(alanlar)

    def kare(self, png: bytes, olcek: float = 1.0) -> None:
        """Ajanın gördüğü kareyi panele koy."""
        self.state["png"] = base64.b64encode(png).decode()
        self.state["olcek"] = olcek

    def olay(self, metin: str, tur: str = "bilgi") -> None:
        akis = self.state["akis"]
        akis.append({"t": time.strftime("%H:%M:%S"), "metin": metin, "tur": tur})
        self.state["akis"] = akis[-40:]
        if tur == "engel":
            self.state["engel"] = (self.state["engel"] + [metin])[-10:]


SAYFA = """<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>cua-lab</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{font:12px/1.4 "Noto Sans",system-ui,sans-serif;background:#0d1117;color:#e6edf3;
     padding:10px;overflow-x:hidden}
h1{font-size:13px;letter-spacing:.4px;color:#7dd3a0;margin-bottom:2px}
.sub{font-size:10px;color:#7d8590;margin-bottom:8px;word-break:break-word}
.kutu{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:8px;margin-bottom:8px}
.etiket{font-size:9px;text-transform:uppercase;letter-spacing:.7px;color:#7d8590;margin-bottom:3px}
.buyuk{font-size:15px;font-weight:700}
#tuval{position:relative;width:100%;border-radius:4px;overflow:hidden;background:#000;
       border:1px solid #30363d}
#ss{width:100%;display:block}
#arti{position:absolute;width:26px;height:26px;margin:-13px 0 0 -13px;pointer-events:none;
      border:2px solid #f85149;border-radius:50%;box-shadow:0 0 0 2px rgba(248,81,73,.25);
      transition:left .18s,top .18s}
#arti:before,#arti:after{content:"";position:absolute;background:#f85149}
#arti:before{left:11px;top:-8px;width:2px;height:8px}
#arti:after{left:-8px;top:11px;width:8px;height:2px}
.satir{display:flex;justify-content:space-between;gap:6px;margin-bottom:3px}
.satir span:last-child{color:#7d8590;text-align:right;word-break:break-all}
.cubuk{height:5px;background:#30363d;border-radius:3px;overflow:hidden;margin-top:3px}
.cubuk i{display:block;height:100%;background:#3fb950;transition:width .2s}
.cubuk i.orta{background:#d29922}.cubuk i.dolu{background:#f85149}
#akis{max-height:190px;overflow-y:auto;font:10px/1.5 "DejaVu Sans Mono",monospace}
#akis div{padding:1px 0;border-bottom:1px solid #21262d;word-break:break-all}
.t{color:#484f58;margin-right:5px}
.engel{color:#f85149}.eylem{color:#79c0ff}.iyi{color:#3fb950}
.rozet{display:inline-block;padding:2px 7px;border-radius:10px;font-size:10px;font-weight:700}
.OK{background:#238636}.STUCK{background:#9e6a03}.NEEDS_INPUT{background:#1f6feb}
.BUDGET_EXHAUSTED{background:#8250df}.DEGRADED{background:#9e6a03}.CEILING{background:#da3633}
</style></head><body>
<h1>cua-lab · canli izleme</h1>
<div class="sub" id="gorev">—</div>

<div class="kutu">
  <div class="etiket">ajan nereye bakiyor</div>
  <div id="tuval"><img id="ss"><div id="arti" style="display:none"></div></div>
  <div class="satir" style="margin-top:6px">
    <span class="buyuk" id="eylem">—</span><span id="hedef"></span>
  </div>
  <div class="satir"><span style="color:#7d8590">aktif pencere</span><span id="pencere">—</span></div>
</div>

<div class="kutu">
  <div class="etiket">butce</div>
  <div id="butce"></div>
</div>

<div class="kutu" id="engelkutu" style="display:none">
  <div class="etiket" style="color:#f85149">engellenen eylemler</div>
  <div id="engel" style="font-size:11px"></div>
</div>

<div class="kutu">
  <div class="etiket">akis</div>
  <div id="akis"></div>
</div>

<div class="sub" style="text-align:center;color:#484f58">
  kacis: fareyi SOL UST KOSEYE tasi
</div>

<script>
const $ = i => document.getElementById(i);
function cubuk(ad, k, l){
  if(!l) return "";
  const o = Math.min(k/l, 1), c = o<.6?"":(o<.85?"orta":"dolu");
  const g = (typeof k === "number" && k % 1) ? k.toFixed(3) : k;
  return `<div class="satir"><span>${ad}</span><span>${g} / ${l}</span></div>
          <div class="cubuk"><i class="${c}" style="width:${o*100}%"></i></div>`;
}
async function tik(){
  let s; try { s = await (await fetch("/state")).json(); } catch(e){ return; }
  $("gorev").textContent = (s.strateji||"") + (s.gorev? " · "+s.gorev : "");
  $("eylem").textContent = s.eylem || "—";
  $("eylem").className = "buyuk " + (s.faz==="engellendi" ? "engel":"eylem");
  $("hedef").textContent = s.hedef ? `(${s.hedef[0]}, ${s.hedef[1]})` : "";
  $("pencere").textContent = s.pencere || "—";

  if(s.png){
    const img = $("ss");
    img.src = "data:image/png;base64," + s.png;
    if(s.hedef){
      const a = $("arti");
      // hedef GERCEK ekran uzayinda; goruntu olcekli ve panele sigdirilmis
      const gx = s.hedef[0] / (s.olcek||1), gy = s.hedef[1] / (s.olcek||1);
      const oran = img.clientWidth / (img.naturalWidth||1);
      a.style.left = (gx*oran) + "px"; a.style.top = (gy*oran) + "px";
      a.style.display = "block";
    }
  }
  let b = cubuk("adim", s.adim, s.limit.steps) + cubuk("token", s.token, s.limit.tokens)
        + cubuk("maliyet $", s.cost, s.limit.cost) + cubuk("sure sn", s.sure, s.limit.seconds);
  if(s.durum) b = `<div class="satir"><span class="rozet ${s.durum}">${s.durum}</span>
                   <span>${s.sebep||""}</span></div>` + b;
  $("butce").innerHTML = b;

  $("engelkutu").style.display = (s.engel||[]).length ? "block":"none";
  $("engel").innerHTML = (s.engel||[]).map(e=>`<div class="engel">✕ ${e}</div>`).join("");

  $("akis").innerHTML = (s.akis||[]).slice().reverse()
    .map(o=>`<div><span class="t">${o.t}</span><span class="${o.tur==='engel'?'engel':
      (o.tur==='iyi'?'iyi':'')}">${o.metin}</span></div>`).join("");
}
setInterval(tik, 400); tik();
</script></body></html>"""
