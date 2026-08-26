"""
Canlı izleyici — ajanın NE GÖRDÜĞÜNÜ ve NEREYE DOKUNACAĞINI ekranda göster.

NEDEN TERMINAL PANELI YETMIYOR: `panel.py` kısıtların doluluk oranını ve son
araç çağrılarını METİN olarak gösteriyor. Ama bir computer-use ajanında asıl
soru görsel: modele hangi kare gitti, ve o karenin neresine tıklamak üzere.
Bunu terminalde göstermek mümkün değil — GNOME Terminal sixel/kitty grafik
protokolü desteklemiyor, yani ekran görüntüsü hiçbir şekilde basılamıyor.

ÇÖZÜM: localhost'ta küçük bir sunucu + ekranın sağ şeridine sabitlenmiş bir
Chrome `--app` penceresi. Sayfa `/durum`u yokluyor, son kareyi çiziyor ve
hedefin üstüne artı işareti koyuyor.

TASARIM KARARI — eylem ÖNCE gösteriliyor: `x11.py` tıklamadan önce imleci
hedefe götürüp `dwell_seconds` kadar bekliyor. İzleyici o aralıkta "şuraya
tıklayacak" diyor, yani kullanıcı olaya YETİŞEBİLİYOR. Olan biteni sonradan
raporlayan bir gösterge, kaçış kolu olarak işe yaramaz.

SINIR: `--desktop` açıkken ajan TÜM ekranın görüntüsünü alıyor, dolayısıyla bu
pencere de o karenin içinde kalıyor. Kendini gösteren bir ayna oluşuyor.
Pencere sağ şeride sıkıştırıldı ve dar tutuldu ki modelin gördüğü asıl alanı
kapatmasın; yine de tam ekran yakalamada görünür.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _ekran_w() -> int:
    from .tools.cdp import _ekran_boyutu
    return _ekran_boyutu()[0]

SAYFA = """<!doctype html><meta charset="utf-8"><title>ajan</title>
<style>
  :root{--bg:#141018;--fg:#ece7f2;--dim:#8f86a0;--line:#2c2536;
        --v:#c295e8;--w:#e3a863;--r:#f09393;--k:#77ce86}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:12px ui-monospace,"IBM Plex Mono",Menlo,monospace;
       display:flex;flex-direction:column;height:100vh;overflow:hidden}
  header{padding:8px 10px;border-bottom:1px solid var(--line);
         display:flex;gap:10px;align-items:baseline}
  header b{color:var(--v);font-size:13px}
  header span{color:var(--dim);margin-left:auto}
  #tuval{flex:0 0 auto;width:100%;background:#000;display:block}
  #eylem{padding:9px 10px;border-bottom:1px solid var(--line);min-height:44px}
  #eylem .f{color:var(--dim);font-size:10px;letter-spacing:.09em;text-transform:uppercase}
  #eylem .m{margin-top:3px;word-break:break-word;line-height:1.45}
  .bekle{color:var(--w)} .engel{color:var(--r)} .yap{color:var(--k)}
  #kisit{flex:1;overflow-y:auto;padding:8px 10px}
  .satir{display:grid;grid-template-columns:1fr 84px 52px;gap:8px;
         align-items:center;padding:3px 0;color:var(--dim)}
  .bar{height:7px;background:#221c2c;border:1px solid var(--line);position:relative}
  .bar i{position:absolute;inset:0 auto 0 0;background:var(--v)}
  .bar.w i{background:var(--w)} .bar.r i{background:var(--r)}
  .sag{text-align:right;color:var(--fg);font-variant-numeric:tabular-nums}
  footer{padding:6px 10px;border-top:1px solid var(--line);color:var(--dim);font-size:10.5px}
</style>
<header><b>AJAN</b><span id="adim">—</span></header>
<canvas id="tuval"></canvas>
<div id="eylem"><div class="f">sıradaki eylem</div><div class="m" id="em">bekleniyor…</div></div>
<div id="kisit"></div>
<footer>kaçış: imleci sol üst köşeye taşı</footer>
<script>
const tuval=document.getElementById('tuval'), ctx=tuval.getContext('2d');
const img=new Image(); let hedef=null, damga=-1;
img.onload=()=>{
  const w=tuval.clientWidth, o=img.height/img.width;
  tuval.width=w; tuval.height=Math.round(w*o);
  tuval.style.height=tuval.height+'px';
  ctx.drawImage(img,0,0,tuval.width,tuval.height);
  if(hedef){
    // Hedef, MODELE GIDEN karenin piksel uzayinda geliyor; tuval o kareyi
    // olcekleyerek ciziyor. Ayni olcegi isarete de uygulamazsak artı yanlis
    // yere duser — kullanicinin guvendigi tek gorsel ipucu bu.
    const s=tuval.width/img.width, x=hedef[0]*s, y=hedef[1]*s;
    ctx.strokeStyle='#f09393'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.arc(x,y,13,0,6.2832); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x-22,y); ctx.lineTo(x-6,y); ctx.moveTo(x+6,y); ctx.lineTo(x+22,y);
    ctx.moveTo(x,y-22); ctx.lineTo(x,y-6); ctx.moveTo(x,y+6); ctx.lineTo(x,y+22);
    ctx.stroke();
  }
};
async function tik(){
  try{
    const d=await (await fetch('/durum',{cache:'no-store'})).json();
    document.getElementById('adim').textContent=d.adim||'—';
    const em=document.getElementById('em');
    em.textContent=d.eylem||'—';
    em.className='m '+(d.faz||'');
    hedef=d.hedef||null;
    if(d.damga!==damga){ damga=d.damga; img.src='/kare.png?t='+damga; }
    else if(hedef){ img.onload(); }
    document.getElementById('kisit').innerHTML=(d.kisitlar||[]).map(k=>{
      const o=k[2]?Math.min(1,k[1]/k[2]):0;
      const s=o>=.8?'r':(o>=.5?'w':'');
      return `<div class="satir"><span>${k[0]}</span>`+
             `<span class="bar ${s}"><i style="width:${o*100}%"></i></span>`+
             `<span class="sag">${k[1]}/${k[2]}</span></div>`;
    }).join('');
  }catch(e){}
}
setInterval(tik,350); tik();
</script>"""


class Izleyici:
    """Ajanın gördüğü kareyi ve sıradaki eylemi canlı yayınlar."""

    def __init__(self, konum: str = "sag", genislik: int = 540):
        self.konum, self.genislik = konum, genislik
        self._kare: bytes = b""
        self._durum = {"adim": "", "eylem": "", "faz": "", "hedef": None,
                       "kisitlar": [], "damga": 0}
        self._kilit = threading.Lock()
        self._srv = None
        self._chrome = None
        self.port = 0

    # -- yaşam döngüsü -----------------------------------------------------

    def basla(self) -> str:
        izleyici = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):        # sunucu logu terminale sızmasın
                pass

            def _gonder(self, kod, tip, govde):
                self.send_response(kod)
                self.send_header("Content-Type", tip)
                self.send_header("Content-Length", str(len(govde)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(govde)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                yol = self.path.split("?")[0]
                if yol == "/durum":
                    with izleyici._kilit:
                        govde = json.dumps(izleyici._durum).encode()
                    return self._gonder(200, "application/json", govde)
                if yol == "/kare.png":
                    with izleyici._kilit:
                        govde = izleyici._kare
                    if not govde:
                        return self._gonder(404, "text/plain", b"kare yok")
                    return self._gonder(200, "image/png", govde)
                return self._gonder(200, "text/html; charset=utf-8",
                                    SAYFA.encode("utf-8"))

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self._srv.server_address[1]
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        adres = f"http://127.0.0.1:{self.port}/"
        self._pencere_ac(adres)
        return adres

    def _pencere_ac(self, adres: str) -> None:
        import shutil
        from .tools.cdp import _ekran_boyutu
        ikili = next((c for c in ("google-chrome", "chromium", "chromium-browser")
                      if shutil.which(c)), None)
        if not ikili:
            return                                  # Chrome yoksa sunucu yine ayakta
        ekran_w, ekran_h = _ekran_boyutu()
        x = ekran_w - self.genislik if self.konum == "sag" else 0
        try:
            self._chrome = subprocess.Popen(
                [ikili, f"--app={adres}",
                 f"--window-size={self.genislik},{ekran_h - 60}",
                 f"--window-position={x},0",
                 "--user-data-dir=/tmp/agentcli-izle-profil",
                 "--no-first-run", "--no-default-browser-check"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            self._chrome = None
            return
        # Pencere yoneticisi Chrome'un istedigi konumu EZIYOR — olculdu, x=960
        # istenip x=836 alinmisti. Pencere haritalandiktan SONRA zorla yerlestir.
        threading.Thread(target=self._zorla, args=(x, ekran_h), daemon=True).start()

    def _zorla(self, x: int, ekran_h: int) -> None:
        """Pencereyi sağ şeride yapıştır ve TUTTUĞUNU DOĞRULA.

        Tek atışlık `windowmove` yetmiyor: ölçüldü, 460x1020 @(1460,0) istenip
        532x1000 @(1402,86) alındı. Pencere yöneticisi hem boyutu hem konumu
        eziyor, üstelik pencere haritalandıktan SONRA. Bu yüzden her denemeden
        sonra geometri geri okunuyor ve sapma varsa tekrar deneniyor.
        """
        wid = self._pencere_bul()
        if not wid:
            return
        h = ekran_h - 40
        # AYNI koordinati tekrar gondermek yakinsamiyor: `xdotool windowmove`
        # CERCEVEYI tasiyor, `getwindowgeometry` ISTEMCI alanini okuyor ve
        # aradaki dekorasyon farki sabit kaliyor — olculdu, X'te 14px Y'de 86px.
        # Cozum sapmayi olcup TELAFI ETMEK: bir sonraki komutta farki dus.
        istenen_x, istenen_y = x, 0
        gonderilen_x, gonderilen_y = x, 0
        for _ in range(5):
            for argv in (["xdotool", "windowsize", wid,
                          str(self.genislik), str(h)],
                         ["xdotool", "windowmove", wid,
                          str(gonderilen_x), str(gonderilen_y)]):
                subprocess.run(argv, capture_output=True, timeout=3)
            time.sleep(0.3)
            g = self._geometri(wid)
            if not g:
                return
            gw, _, gx, gy = g
            # Chrome'un alt genislik siniri var; gercek degeri kabul edip
            # hedefi ona gore kaydiriyoruz, yoksa sonsuz cekisme oluyor.
            if gw > self.genislik:
                self.genislik = gw
                istenen_x = max(0, _ekran_w() - gw) if self.konum == "sag" else 0
            if abs(gx - istenen_x) <= 4 and abs(gy - istenen_y) <= 4:
                break
            gonderilen_x -= (gx - istenen_x)
            if gy > istenen_y and gonderilen_y <= -60:
                # Y bir turlu 0'a inmiyorsa masaustunun UST PANELI orayi
                # kapatiyor demektir (olculdu: calisma alani y=86'da basliyor).
                # WM'le cekismek yerine calisma alanini kabul edip YUKSEKLIGI
                # ona uyduruyoruz — yoksa pencerenin alti ekranin disina tasiyor.
                istenen_y = gy
                h = max(320, ekran_h - gy - 12)
                gonderilen_y = gy
                continue
            gonderilen_y -= (gy - istenen_y)
            gonderilen_y = max(gonderilen_y, -120)   # WM cok negatifi kirpiyor
        # Yakinsasin ya da yakinsamasin: tasma duzeltmesi HER DURUMDA calissin.
        # Once yalniz yakinsama dalindaydi ve dongu 5 turda bitince atlaniyordu.
        self._alta_oturt(wid, ekran_h)

    def _alta_oturt(self, wid: str, ekran_h: int) -> None:
        """Pencerenin ALTI ekrandan taşmasın.

        Yükseklik, konum yakınsamadan önce hesaplandığı için son ölçülen `y`
        ile uyuşmuyor ve 1px taşma bırakıyordu. Burada gerçek `y` ile
        yeniden ölçülüp bir kez daraltılıyor.
        """
        for _ in range(3):
            g = self._geometri(wid)
            if not g:
                return
            _, gh, _, gy = g
            tasma = (gy + gh) - ekran_h
            if tasma <= 0:
                return
            subprocess.run(["xdotool", "windowsize", wid,
                            str(self.genislik), str(max(320, gh - tasma - 4))],
                           capture_output=True, timeout=3)
            time.sleep(0.25)

    def _pencere_bul(self) -> str | None:
        for _ in range(28):
            time.sleep(0.25)
            try:
                ids = subprocess.run(
                    ["xdotool", "search", "--onlyvisible", "--class", "chrome"],
                    capture_output=True, text=True, timeout=3).stdout.split()
            except Exception:
                return None
            for wid in reversed(ids):
                try:
                    ad = subprocess.run(["xdotool", "getwindowname", wid],
                                        capture_output=True, text=True,
                                        timeout=3).stdout.strip()
                except Exception:
                    continue
                if ad.startswith("ajan"):
                    return wid
        return None

    @staticmethod
    def _geometri(wid: str):
        try:
            g = subprocess.run(["xdotool", "getwindowgeometry", "--shell", wid],
                               capture_output=True, text=True, timeout=3).stdout
            d = dict(l.split("=", 1) for l in g.splitlines() if "=" in l)
            return (int(d["WIDTH"]), int(d["HEIGHT"]), int(d["X"]), int(d["Y"]))
        except Exception:
            return None

    def kapat(self) -> None:
        if self._chrome:
            self._chrome.terminate()
            self._chrome = None
        if self._srv:
            self._srv.shutdown()
            self._srv = None

    # -- besleme -----------------------------------------------------------

    def kare(self, png: bytes | None) -> None:
        if not png:
            return
        with self._kilit:
            self._kare = png
            self._durum["damga"] = self._durum["damga"] + 1

    def eylem(self, metin: str, faz: str = "bekle",
              hedef: tuple[int, int] | None = None) -> None:
        """`faz`: bekle (yapılmak üzere) · yap (yapıldı) · engel (reddedildi)."""
        with self._kilit:
            self._durum["eylem"] = metin
            self._durum["faz"] = faz
            self._durum["hedef"] = list(hedef) if hedef else None

    def adim(self, n, kisitlar: list[tuple]) -> None:
        with self._kilit:
            self._durum["adim"] = f"adım {n}"
            self._durum["kisitlar"] = [list(k) for k in kisitlar]

    # -- x11 gözlemci köprüsü ----------------------------------------------

    def gozlemci(self):
        """`X11Sandbox(gozlemci=…)` imzasına uyan geri çağrı.

        `_bildir(faz, metin, sandbox)` çağrılıyor; bizim tarafta faz adları
        farklı olduğu için burada eşleniyor.
        """
        # `x11.py` bu adlari kullaniyor: baslat · bakiyor · engellendi · bitti.
        # `bakiyor` eylemden ONCE, `dwell_seconds` beklemesinin basinda geliyor —
        # kullanicinin yetisebildigi tek an orasi.
        esle = {"bakiyor": "bekle", "engellendi": "engel",
                "baslat": "yap", "bitti": "yap"}

        def kanca(faz, metin, sandbox):
            hedef = None
            if "→" in metin and "(" in metin:
                try:
                    ham = metin.split("(")[-1].rstrip(")")
                    a, b = ham.split(",")
                    hedef = (int(a), int(b))
                except Exception:
                    hedef = None
            self.eylem(metin, esle.get(faz, "bekle"), hedef)
        return kanca
